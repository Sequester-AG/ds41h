// fpr_cu_launch.h -- cuLaunchKernelEx without linking libcuda: kernel_capture.h and
// triton_replay.h relaunch driver-API functions from the C++ mixed-step driver, whose extension links no -lcuda. The
// entry point comes from cudart (cudaGetDriverEntryPointByVersion, CUDA 12.0 ABI of cuLaunchKernelEx), resolved once per
// process: in C++ a function-local static (thread-safe initialization; immutable afterwards).
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef CUresult (*FprCuLaunchKernelExFn)(const CUlaunchConfig*, CUfunction, void**, void**);

static inline FprCuLaunchKernelExFn fpr_cu_resolve_launch_kernel_ex(void) {
  void* p = NULL;
  enum cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  if (cudaGetDriverEntryPointByVersion("cuLaunchKernelEx", &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
      q != cudaDriverEntryPointSuccess)
    return NULL;
  return (FprCuLaunchKernelExFn)p;
}

static inline CUresult fpr_cu_launch_kernel_ex(const CUlaunchConfig* cfg, CUfunction f, void** params, void** extra) {
#ifdef __cplusplus
  static const FprCuLaunchKernelExFn fn = fpr_cu_resolve_launch_kernel_ex();
#else
  FprCuLaunchKernelExFn fn = fpr_cu_resolve_launch_kernel_ex();
#endif
  return fn ? fn(cfg, f, params, extra) : CUDA_ERROR_NOT_FOUND;
}
