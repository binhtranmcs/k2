/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang, Wei Kang)
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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <dirent.h>

#include "k2/csrc/intersect_dense_pruned.h"
#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/utils.h"
#include "k2/torch/csrc/wave_reader.h"
#include "kaldifeat/csrc/feature-fbank.h"
#include "torch/all.h"
#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_bool(use_gpu, false, "True to use GPU. False to use CPU");
C10_DEFINE_string(jit_pt, "", "Path to exported jit file.");
C10_DEFINE_bool(use_ctc_decoding, true, "True to use CTC decoding");
C10_DEFINE_string(hlg, "/home/cpu13266/binhtt4/clone/k2/build/hlg/my_HLG.pt",
                  "Path to HLG.pt. Needed if --use_ctc_decoding is false");
C10_DEFINE_string(word_table, "/home/cpu13266/binhtt4/clone/k2/build/hlg/my_words.txt",
                  "Path to words.txt. Needed if --use_ctc_decoding is false");
C10_DEFINE_string(tokens, "",
                  "Path to a tokens.txt. Needed if --use_ctc_decoding is true");
// Fsa decoding related
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 20, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 10000,
               "max_activate_states in IntersectDensePruned");
// fbank related
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");
C10_DEFINE_int(num_streams, 1, "Number of concurrent streams");

C10_DEFINE_string(audio_dir,
                  "/home/cpu13266/binhtt4/asr/e2e/asr-end2end-service/build/btt4_audios/",
                  "audio directory for testing");

static void CheckArgs() {
#if !defined(K2_WITH_CUDA)
  if (FLAGS_use_gpu) {
    std::cerr << "k2 was not compiled with CUDA"
              << "\n";
    std::cerr << "Please use --use_gpu 0"
              << "\n";
    exit(EXIT_FAILURE);
  }
#endif

  if (FLAGS_jit_pt.empty()) {
    std::cerr << "Please provide --jit_pt"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_ctc_decoding && FLAGS_tokens.empty()) {
    std::cout << "Please provide --tokens"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_hlg.empty()) {
    std::cerr << "Please provide --hlg"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }
}


std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}


int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string usage = R"(
  (1) CTC decoding
    ./bin/online_decode \
      --use_ctc_decoding true \
      --jit_pt <path to exported torch script pt file> \
      --tokens <path to tokens.txt> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
  (2) HLG decoding
    ./bin/online_decode \
      --use_ctc_decoding false \
      --jit_pt <path to exported torch script pt file> \
      --hlg <path to HLG.pt> \
      --word_table <path to words.txt> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
   --use_gpu false to use CPU
   --use_gpu true to use GPU
  )";
  torch::SetUsageMessage(usage);

  torch::ParseCommandLineFlags(&argc, &argv);
  CheckArgs();

  torch::Device device(torch::kCPU);
  if (FLAGS_use_gpu) {
    device = torch::Device(torch::kCUDA, 0);
  }

  K2_LOG(INFO) << "Device: " << device;

  int32_t num_waves = 1;

  std::ofstream fout("fixk2.txt");
  std::vector<int64_t> num_frames;

  /// feat
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = FLAGS_sample_rate;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = FLAGS_frame_shift_ms;
  fbank_opts.frame_opts.frame_length_ms = FLAGS_frame_length_ms;
  fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
  fbank_opts.device = device;
  kaldifeat::Fbank fbank(fbank_opts);

  /// jit
  K2_LOG(INFO) << "Load neural network model";
  torch::jit::script::Module module = torch::jit::load(FLAGS_jit_pt);
  module.eval();
  module.to(device);

  /// fst
  K2_LOG(INFO) << "Load " << FLAGS_hlg;
  k2::FsaClass decoding_graph = k2::LoadFsa(FLAGS_hlg, device);
  K2_CHECK(decoding_graph.HasTensorAttr("aux_labels") ||
           decoding_graph.HasRaggedTensorAttr("aux_labels"));
  decoding_graph.SetTensorAttr("lm_scores", decoding_graph.Scores().clone());
  auto decoding_fsa = k2::FsaToFsaVec(decoding_graph.fsa);
  k2::OnlineDenseIntersecter decoder(
      decoding_fsa, FLAGS_num_streams, FLAGS_search_beam, FLAGS_output_beam,
      FLAGS_min_activate_states, FLAGS_max_activate_states);

  /// simulation
  DIR *dir; struct dirent *diread;
  std::string path = FLAGS_audio_dir;
//  std::string path = "/home/cpu13266/Downloads/test/";
//  std::string path = "/home/cpu13266/binhtt4/clone/k2/build/tmp/tmpwav/";
  if ((dir = opendir(path.c_str())) == nullptr) {
    assert(false);
  }
  while ((diread = readdir(dir)) != nullptr) {
//  while (true) {
    std::string filename = path + diread->d_name;
    if (filename.find(".wav") == std::string::npos) continue;

    std::cout << diread->d_name << '\n';
    fout << diread->d_name << ' ';

    K2_LOG(INFO) << "Load wave files " << diread->d_name;
    auto wave_data = k2::ReadWave(std::vector<std::string>(1, filename), FLAGS_sample_rate);
    for (auto &w : wave_data) w = w.to(device);

    {
      /// feat
      K2_LOG(INFO) << "Compute features";
      std::vector<int64_t> num_frames;
      auto features_vec = k2::ComputeFeatures(fbank, wave_data, &num_frames);
      // Note: math.log(1e-10) is -23.025850929940457
      auto features = torch::nn::utils::rnn::pad_sequence(features_vec, true,
          -23.025850929940457f);

      /// jit
      int32_t subsampling_factor = module.attr("subsampling_factor").toInt();
      torch::Dict<std::string, torch::Tensor> sup;
      sup.insert("sequence_idx", torch::arange(num_waves, torch::kInt));
      sup.insert("start_frame", torch::zeros({num_waves}, torch::kInt));
      sup.insert("num_frames",
          torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
              .to(torch::kInt));
      torch::IValue supervisions(sup);
      std::vector<torch::IValue> inputs;
      inputs.emplace_back(std::move(features));
      inputs.emplace_back(supervisions);
      K2_LOG(INFO) << "Compute nnet_output";
      // the output for module.forward() is a tuple of 3 tensors
      auto outputs = module.forward(inputs).toTuple();
      assert(outputs->elements().size() == 3u);
      auto nnet_output = outputs->elements()[0].toTensor();

//      int subsampling_factor = 4;
//      num_frames.push_back(96 * 4);
//      std::vector<char> f = get_the_bytes("/home/cpu13266/binhtt4/clone/k2/build/tmp.pt");
//      torch::IValue x = torch::pickle_load(f);
//      torch::Tensor nnet_output = x.toTensor();

      K2_LOG(INFO) << "Sizes: " << nnet_output.sizes() << " | " << num_frames[0];

      /// fst
      // store decode states for each waves
      std::vector<k2::DecodeStateInfo> states_info(num_waves);
      // decocding results for each waves
      std::vector<std::string> texts(num_waves, "");
      std::vector<int32_t> positions(num_waves, 0);
      int32_t T = nnet_output.size(1);
      int32_t chunk_size = 10;

      /// simulate asynchronous decoding
      while (true) {
        k2::DecodeStateInfo dummy_state_info;
        std::vector<k2::DecodeStateInfo*> current_states_info;
        std::vector<int64_t> num_frame;
        std::vector<torch::Tensor> current_nnet_output;
        // which waves we are decoding now
        std::vector<int32_t> current_wave_ids;

        {
          for (int32_t i = 0; i < num_waves; ++i) {
            // this wave is done
            if (num_frames[i] == 0) continue;

            current_states_info.push_back(&states_info[i]);
            current_wave_ids.push_back(i);

            if (num_frames[i] <= chunk_size * subsampling_factor) {
              num_frame.push_back(num_frames[i]);
              num_frames[i] = 0;
            } else {
              num_frame.push_back(chunk_size * subsampling_factor);
              num_frames[i] -= chunk_size * subsampling_factor;
            }

            int32_t start = positions[i],
                    end = start + chunk_size >= T ? T : start + chunk_size;
            positions[i] = end;
            auto sub_output = nnet_output.index(
                {i, torch::indexing::Slice(start, end), torch::indexing::Slice()});

            // padding T axis to chunk_size if needed
            namespace F = torch::nn::functional;
            sub_output = F::pad(sub_output,
                                F::PadFuncOptions({0, 0, 0, chunk_size - end + start})
                                    .mode(torch::kConstant));

            current_nnet_output.push_back(sub_output);

            // we can only decode `FLAGS_num_streams` waves at a time
            if (static_cast<int32_t>(current_wave_ids.size()) >= FLAGS_num_streams)
              break;
          }
          if (current_wave_ids.size() == 0) break;  // finished
          // no enough waves, feed in garbage data
          while (static_cast<int32_t>(num_frame.size()) < FLAGS_num_streams) {
            assert(false);
            num_frame.push_back(0);
            auto opts = torch::TensorOptions().dtype(nnet_output.dtype())
                            .device(nnet_output.device());
            current_nnet_output.push_back(
                torch::zeros({chunk_size, nnet_output.size(2)}, opts));
            current_states_info.push_back(&dummy_state_info);
          }
        }

        /// sup
        auto sub_nnet_output = torch::stack(current_nnet_output);
        torch::Dict<std::string, torch::Tensor> sup;
        sup.insert("sequence_idx", torch::arange(FLAGS_num_streams, torch::kInt));
        sup.insert("start_frame", torch::zeros({FLAGS_num_streams}, torch::kInt));
        sup.insert("num_frames",
            torch::from_blob(num_frame.data(), {FLAGS_num_streams},
                torch::kLong).to(torch::kInt));
        torch::IValue supervision(sup);
        torch::Tensor supervision_segments =
            k2::GetSupervisionSegments(supervision, subsampling_factor);

        /// get lattice
        K2_LOG(INFO) << "Decoding";
        k2::DenseFsaVec dense_fsa_vec = k2::CreateDenseFsaVec(
            sub_nnet_output, supervision_segments, subsampling_factor - 1);
        k2::FsaVec fsa;
        k2::Array1<int32_t> graph_arc_map;
        decoder.Decode(dense_fsa_vec, &current_states_info, &fsa, &graph_arc_map);
        k2::FsaClass lattice(fsa);
        lattice.CopyAttrs(decoding_graph, k2::Array1ToTorch<int32_t>(graph_arc_map));

        /// nbest
        // lattice = k2::ShortestPath(lattice);
        auto nbest = k2::Nbest::FromLattice(lattice, 3, 1.0);
        nbest.Intersect(&lattice);
        // must hold the array first before getting the underlying data
        auto am_scores_arr = k2::Array1FromTorch<float>(nbest.ComputeAmScores()).To(k2::GetCpuContext());
        auto lm_scores_arr = k2::Array1FromTorch<float>(nbest.ComputeLmScores()).To(k2::GetCpuContext());
        auto row_split = nbest.shape.RowSplits(1).To(k2::GetCpuContext());

        auto ragged_aux_labels = k2::GetTexts(nbest.fsa);
        K2_LOG(INFO) << ragged_aux_labels.shape;
        auto aux_labels_vec = ragged_aux_labels.ToVecVec();
      }
    }
  }

  return 0;
}
