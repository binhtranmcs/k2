/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <dirent.h>

#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/wave_reader.h"
#include "torch/all.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
This file implements decoding with an HLG decoding graph.

Usage:
  ./bin/hlg_decode \
    --use_gpu true \
    --nn_model <path to torch scripted pt file> \
    --hlg <path to HLG.pt> \
    --word_table <path to words.txt> \
    <path to foo.wav> \
    <path to bar.wav> \
    <more waves if any>

To see all possible options, use
  ./bin/hlg_decode --help

Caution:
 - Only sound files (*.wav) with single channel are supported.
 - It assumes the model is conformer_ctc/transformer.py from icefall.
   If you use a different model, you have to change the code
   related to `model.forward` in this file.
)";

C10_DEFINE_bool(use_gpu, false, "true to use GPU; false to use CPU");
C10_DEFINE_string(nn_model, "", "Path to the model exported by torch script.");
C10_DEFINE_string(hlg, "", "Path to HLG.pt.");
C10_DEFINE_string(word_table, "", "Path to words.txt.");

// Fsa decoding related
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 8, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 10000,
               "max_activate_states in IntersectDensePruned");
// Fbank related
// NOTE: These parameters must match those used in training
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");
C10_DEFINE_string(audio_dir, "/home/cpu13266/binhtt4/clone/k2/build/tmp/tmpwav/", "")

static void CheckArgs() {
#if !defined(K2_WITH_CUDA)
  if (FLAGS_use_gpu) {
    std::cerr << "k2 was not compiled with CUDA. "
                 "Please use --use_gpu false";
    exit(EXIT_FAILURE);
  }
#endif

  if (FLAGS_nn_model.empty()) {
    std::cerr << "Please provide --nn_model\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_hlg.empty()) {
    std::cerr << "Please provide --hlg\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::SetUsageMessage(kUsageMessage);
  torch::ParseCommandLineFlags(&argc, &argv);
  CheckArgs();

  torch::Device device(torch::kCPU);
  if (FLAGS_use_gpu) {
    K2_LOG(INFO) << "Use GPU";
    device = torch::Device(torch::kCUDA, 0);
  }

  K2_LOG(INFO) << "Device: " << device;

  int32_t num_waves = argc - 1;
  K2_CHECK_GE(num_waves, 1) << "You have to provide at least one wave file";
  std::vector<std::string> wave_filenames(num_waves);
  for (int32_t i = 0; i != num_waves; ++i) {
    wave_filenames[i] = argv[i + 1];
  }

  K2_LOG(INFO) << "Build Fbank computer";
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = FLAGS_sample_rate;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = FLAGS_frame_shift_ms;
  fbank_opts.frame_opts.frame_length_ms = FLAGS_frame_length_ms;
  fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  K2_LOG(INFO) << "Load neural network model";
  torch::jit::script::Module module = torch::jit::load(FLAGS_nn_model);
  module.eval();
  module.to(device);

  K2_LOG(INFO) << "Load " << FLAGS_hlg;
  k2::FsaClass decoding_graph = k2::LoadFsa(FLAGS_hlg, device);
  K2_CHECK(decoding_graph.HasTensorAttr("aux_labels") ||
           decoding_graph.HasRaggedTensorAttr("aux_labels"));

  DIR *dir; struct dirent *diread;
  std::string path = FLAGS_audio_dir;
  if ((dir = opendir(path.c_str())) != nullptr) {
    while ((diread = readdir(dir)) != nullptr) {
      std::string filename = path + diread->d_name;
      if (filename.find(".wav") == std::string::npos) continue;

      std::cout << diread->d_name << '\n';
      auto wave_data = k2::ReadWave(std::vector<std::string>(1, filename), FLAGS_sample_rate);
      for (auto& w : wave_data) {
        w = w.to(torch::kCUDA);
      }

      K2_LOG(INFO) << "Compute features";
      std::vector<int64_t> num_frames;
      auto features_vec = k2::ComputeFeatures(fbank, wave_data, &num_frames);

      auto features = torch::stack(features_vec, 0);

      int32_t subsampling_factor = module.attr("subsampling_factor").toInt();
      torch::Dict<std::string, torch::Tensor> sup;
      sup.insert("sequence_idx", torch::arange(num_waves, torch::kInt));
      sup.insert("start_frame", torch::zeros({num_waves}, torch::kInt));
      sup.insert("num_frames",
                 torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
                     .to(torch::kInt));

      torch::IValue supervisions(sup);

      K2_LOG(INFO) << "Compute nnet_output";
      auto outputs = module.run_method("forward", features, supervisions).toTuple();
      assert(outputs->elements().size() == 3u);

      auto nnet_output = outputs->elements()[0].toTensor();
      auto memory = outputs->elements()[1].toTensor();

      torch::Tensor supervision_segments =
          k2::GetSupervisionSegments(supervisions, subsampling_factor);

      auto start = std::chrono::system_clock::now();
      K2_LOG(INFO) << "Decoding";
      k2::FsaClass lattice = k2::GetLattice(
          nnet_output, decoding_graph, supervision_segments, FLAGS_search_beam,
          FLAGS_output_beam, FLAGS_min_activate_states, FLAGS_max_activate_states,
          subsampling_factor);

      auto nbest = k2::Nbest::FromLattice(lattice, 10, 1.0);
      nbest.Intersect(&lattice);

      auto ragged_aux_labels = k2::GetTexts(nbest.fsa);
      auto aux_labels_vec = ragged_aux_labels.ToVecVec();

      K2_LOG(INFO) << "?????????????? " << aux_labels_vec.size();

      auto end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed_seconds = end-start;

      K2_LOG(INFO) << "elapsed time: " << elapsed_seconds.count() << "s";
    }
  }

  return 0;
}
