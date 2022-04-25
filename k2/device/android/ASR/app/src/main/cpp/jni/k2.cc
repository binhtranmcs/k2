/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <android/log.h>
#include <jni.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/utils.h"
#include "k2/torch/csrc/wave_reader.h"
#include "kaldifeat/csrc/feature-fbank.h"
#include "sentencepiece_processor.h"  // NOLINT
#include "torch/all.h"
#include "torch/script.h"
#include "torch/utils.h"

#define TAG "k2"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace k2 {
namespace jni {
std::shared_ptr<torch::jit::script::Module> g_module;
std::shared_ptr<rnnt_decoding::RnntDecodingConfig> g_config;
std::shared_ptr<FsaClass> g_decoding_graph;
std::shared_ptr<sentencepiece::SentencePieceProcessor> g_bpe_processor;

void init(JNIEnv *env, jobject, jstring jModelPath, jstring jBpePath) {
  torch::Device device(torch::kCPU);
  const char *pModelPath = (env)->GetStringUTFChars(jModelPath, nullptr);
  std::string modelPath = std::string(pModelPath);

  LOGI("model path: %s\n", modelPath.c_str());
  auto module = torch::jit::load(modelPath);
  module.eval();
  module.to(device);
  g_module = std::make_shared<torch::jit::script::Module>(module);

  const char *pBpePath = (env)->GetStringUTFChars(jBpePath, nullptr);
  std::string bpePath = std::string(pBpePath);

  LOGI("bpe path : %s\n", bpePath.c_str());
  g_bpe_processor = std::make_shared<sentencepiece::SentencePieceProcessor>();
  auto status = g_bpe_processor->Load(bpePath);
  if (!status.ok()) {
    K2_LOG(FATAL) << status.ToString();
  }

  int32_t vocab_size = g_module->attr("vocab_size").toInt();
  int32_t context_size = g_module->attr("context_size").toInt();
  auto decoding_graph = TrivialGraph(vocab_size - 1, device);
  g_decoding_graph = std::make_shared<FsaClass>(decoding_graph);

  float beam = 5.0;
  int max_contexts = 10;
  int max_states = 50;
  rnnt_decoding::RnntDecodingConfig config(vocab_size, context_size, beam,
                                           max_states, max_contexts);
  g_config = std::make_shared<rnnt_decoding::RnntDecodingConfig>(config);
}

jstring decode(JNIEnv *env, jobject, jfloatArray jWaveform) {
  auto device = torch::Device(torch::kCPU);
  jsize size = env->GetArrayLength(jWaveform);
  std::vector<float> waveform(size);
  env->GetFloatArrayRegion(jWaveform, 0, size, &waveform[0]);
  auto wave_data =
      torch::from_blob(waveform.data(), {size}, torch::kFloat).to(device);

  int32_t subsampling_factor = g_module->attr("subsampling_factor").toInt();
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = 16000;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = 10.0;
  fbank_opts.frame_opts.frame_length_ms = 25.0;
  fbank_opts.mel_opts.num_bins = 80;
  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  LOGI("Compute features.\n");

  std::vector<int64_t> num_frames;
  std::vector<torch::Tensor> wave_datas;
  wave_datas.push_back(wave_data);
  auto features_vec = ComputeFeatures(fbank, wave_datas, &num_frames);
  auto features = torch::stack(features_vec);

  auto input_lengths = torch::from_blob(num_frames.data(), {1}, torch::kLong)
                           .to(torch::kInt)
                           .to(device);

  LOGI("Compute encoder outs.\n");
  // the output for module.encoder.forward() is a tuple of 2 tensors
  auto outputs =
      g_module->run_method("encoder_forward", features, input_lengths)
          .toTuple();
  assert(outputs->elements().size() == 2u);

  auto encoder_outs = outputs->elements()[0].toTensor();

  LOGI("Build rnnt stream.\n");
  std::vector<k2::FsaClass> current_graphs;
  current_graphs.push_back(*g_decoding_graph);
  std::vector<std::shared_ptr<rnnt_decoding::RnntDecodingStream>>
      current_streams;
  current_streams.push_back(rnnt_decoding::CreateStream(g_decoding_graph->fsa));

  LOGI("Build rnnt streams.\n");
  auto streams = rnnt_decoding::RnntDecodingStreams(current_streams, *g_config);

  LOGI("Decoding.\n");
  LOGI("streams %d\tdim %d\tsize0 %d\n", streams.NumStreams(),
       encoder_outs.dim(), encoder_outs.size(0));

  DecodeOneChunk(streams, *g_module, encoder_outs);

  LOGI("Generate output.\n");
  FsaVec ofsa;
  Ragged<int32_t> out_map;
  int32_t frames =
      min(num_frames[0] / subsampling_factor, encoder_outs.size(1));
  LOGI("frames : %d \t T : %d\n", frames, encoder_outs.size(1));
  std::vector<int32_t> current_num_frames({frames});
  streams.FormatOutput(current_num_frames, &ofsa, &out_map);

  LOGI("Generate lattice.\n");
  FsaClass lattice(ofsa);
  lattice.CopyAttrs(current_graphs, out_map);
  lattice = ShortestPath(lattice);
  auto ragged_aux_labels = k2::GetTexts(lattice);
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();
  std::string text;
  auto status = g_bpe_processor->Decode(aux_labels_vec[0], &text);
  if (!status.ok()) {
    K2_LOG(FATAL) << status.ToString();
  }
  return env->NewStringUTF(text.c_str());
}
}  // namespace jni
}  // namespace k2

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("com/xiaomi/k2/Recognizer");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;Ljava/lang/String;)V",
       reinterpret_cast<void *>(k2::jni::init)},
      {"decode", "([F)Ljava/lang/String;",
       reinterpret_cast<void *>(k2::jni::decode)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
