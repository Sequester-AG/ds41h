// Host-side helpers for ds41h torch extensions.
// Adapted from BRRR (Apache-2.0, src/brrr/kernels/host.cuh).
#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>

template <class T>
T* zeroed(long count) {
  void* ptr = nullptr;
  TORCH_CHECK(cudaMalloc(&ptr, count * sizeof(T)) == cudaSuccess);
  TORCH_CHECK(cudaMemset(ptr, 0, count * sizeof(T)) == cudaSuccess);
  return static_cast<T*>(ptr);
}

template <class T>
T* data_as(const torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <class T>
const T* cptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "expected contiguous CUDA tensor");
  return data_as<const T>(t);
}

template <class T>
T* mptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "expected contiguous CUDA tensor");
  return data_as<T>(t);
}

// Raw cudaMalloc'd buffer (not from the caching allocator, so cudaIpcGetMemHandle is valid).
// Returns [ptr, handle bytes...] as int64 list for exchange via torch.distributed.
inline std::vector<int64_t> alloc_ipc(int64_t bytes) {
  void* ptr = zeroed<uint8_t>(bytes);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(cudaIpcGetMemHandle(&handle, ptr) == cudaSuccess);
  std::vector<int64_t> out{reinterpret_cast<int64_t>(ptr)};
  const auto* raw = reinterpret_cast<const unsigned char*>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i) out.push_back(raw[i]);
  return out;
}

inline int64_t open_ipc(std::vector<int64_t> handle_bytes) {
  cudaIpcMemHandle_t handle;
  auto* raw = reinterpret_cast<unsigned char*>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i) raw[i] = static_cast<unsigned char>(handle_bytes[i]);
  void* ptr = nullptr;
  TORCH_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess);
  return reinterpret_cast<int64_t>(ptr);
}

template <auto Kernel>
void opt_in_smem(int bytes) {
  static const bool done = [bytes] {
    TORCH_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    return true;
  }();
  (void)done;
}

// Launch on the current torch stream with programmatic stream serialization (PDL) enabled,
// so the kernel may start while the previous one drains; kernels must call pdl_wait() before
// consuming upstream outputs. Captured fine by CUDA graphs.
template <class... Params, class... Args>
void launch_pdl(const char* name, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, Args... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = c10::cuda::getCurrentCUDAStream();
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
  TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}
