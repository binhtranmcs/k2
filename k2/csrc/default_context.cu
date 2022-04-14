/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
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

#include <cstdlib>
#include <mutex>  // NOLINT

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/pytorch_context.h"

namespace k2 {

static constexpr std::size_t kAlignment = 64;

// TODO(haowen): most of implementations below should be updated later.
class CpuContext : public Context {
 public:
  CpuContext() = default;
//  ContextPtr GetCpuContext() override { return shared_from_this(); }
  DeviceType GetDeviceType() const override { return kCpu; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
    if (bytes) {
      int32_t ret = posix_memalign(&p, kAlignment, bytes);
      K2_CHECK_EQ(ret, 0);
    }
    if (deleter_context != nullptr) *deleter_context = nullptr;
    return p;
  }

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCpu;
  }

  void Deallocate(void *data, void * deleter_context) override {
      if (deleter_context != nullptr) {
          // a non-empty `deleter_context` indicates that
          // the memory is passed from a `torch::Tensor`
          delete reinterpret_cast<ManagedTensor *>(deleter_context);
      } else {
          free(data);
      }
  }

  void CopyDataTo(size_t num_bytes, const void *src,
                  ContextPtr dst_context, void *dst) override {
      memcpy(dst, src, num_bytes);
  };

};

class CudaContext : public Context {
 public:
  explicit CudaContext(int32_t gpu_id) : gpu_id_(gpu_id) {
#ifdef K2_WITH_CUDA
    if (gpu_id_ != -1) {
      auto ret = cudaSetDevice(gpu_id_);
      K2_CHECK_CUDA_ERROR(ret);
    }
    // TODO(haowen): choose one from available GPUs if gpu_id == -1?
    // and handle GPU ids from multiple machines.
    auto ret = cudaStreamCreate(&stream_);
    K2_CHECK_CUDA_ERROR(ret);
#endif
  }
//  ContextPtr GetCpuContext() override { return k2::GetCpuContext(); }
  DeviceType GetDeviceType() const override { return kCuda; }
  int32_t GetDeviceId() const override { return gpu_id_; }

  void *Allocate(std::size_t bytes, void **deleter_context) override {
    void *p = nullptr;
#ifdef K2_WITH_CUDA
    if (bytes) {
      auto ret = cudaMalloc(&p, bytes);
      K2_CHECK_CUDA_ERROR(ret);
    }
    if (deleter_context != nullptr) *deleter_context = nullptr;
#endif
    return p;
  }

    void CopyDataTo(size_t num_bytes, const void *src,
                    ContextPtr dst_context, void *dst) override {};

  bool IsCompatible(const Context &other) const override {
    return other.GetDeviceType() == kCuda && other.GetDeviceId() == gpu_id_;
  }

  void Deallocate(void *data, void * /*deleter_context*/) override {
      #ifdef K2_WITH_CUDA
    auto ret = cudaFree(data);
    K2_CHECK_CUDA_ERROR(ret);
      #endif
  }

  cudaStream_t GetCudaStream() const override {
      #ifdef K2_WITH_CUDA
    return g_stream_override.OverrideStream(stream_);
      #else
      return cudaStream_t{};
      #endif
  }

  void Sync() const override {
      #ifdef K2_WITH_CUDA
    auto ret = cudaStreamSynchronize(stream_);
    K2_CHECK_CUDA_ERROR(ret);
      #endif
  }

  ~CudaContext() {
      #ifdef K2_WITH_CUDA
    auto ret = cudaStreamDestroy(stream_);
    K2_CHECK_CUDA_ERROR(ret);
      #endif
  }

 private:
  int32_t gpu_id_;
  cudaStream_t stream_;
};

ContextPtr GetCpuContext() { return std::make_shared<CpuContext>(); }

ContextPtr GetCudaContext(int32_t gpu_id /*= -1*/) {
#ifdef K2_WITH_CUDA
  static std::once_flag has_cuda_init_flag;
  static bool has_cuda = false;
  std::call_once(has_cuda_init_flag, []() {
    int n = 0;
    auto ret = cudaGetDeviceCount(&n);
    if (ret == cudaSuccess && n > 0)
      has_cuda = true;
    else
      K2_LOG(WARNING) << "CUDA is not available. Return a CPU context.";
  });

  if (has_cuda) return std::make_shared<CudaContext>(gpu_id);
#endif
  return GetCpuContext();
}

RegionPtr NewRegion(torch::Tensor tensor) {
    auto ans = std::make_shared<Region>();
    if (tensor.device().type() == torch::kCPU) {
        ans->context = GetCpuContext();
    } else if (tensor.is_cuda()) {
        ans->context = GetCudaContext(tensor.device().index());
    } else {
        K2_LOG(FATAL) << "Unsupported device: " << tensor.device()
                      << "\nOnly CPU and CUDA are supported";
    }

    // NOTE: the tensor is passed from Python and we have
    // to retain it to avoid potential segmentation fault.
    //
    // It will be freed in `Context::Deallocate`.
    auto *managed_tensor = new ManagedTensor(tensor);
    ans->data = tensor.data_ptr();
    ans->deleter_context = managed_tensor;
#if K2_TORCH_VERSION_MAJOR > 1 || \
(K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR > 5)
    // nbytes() is available only for torch > 1.5
// see https://github.com/pytorch/pytorch/pull/37028
ans->num_bytes = tensor.storage().nbytes();
#else
    // capacity() is available only for torch <= 1.5.0
    ans->num_bytes = tensor.storage().capacity();
#endif
    ans->bytes_used = ans->num_bytes;
    return ans;
}

}  // namespace k2
