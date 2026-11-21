// pdl2.cuh: programmatic dependent launch (PDL) for the engine's own decode kernels.
//
// Kernel side: pdl2::enter() at the top of a kernel = griddepcontrol.launch_dependents (the next kernel in the stream may
// be scheduled now; it blocks in its own griddepcontrol.wait) + griddepcontrol.wait (block until the previous kernel has
// completed and its memory is visible). Both are no-ops when the kernel was launched without the PDL attribute, so a
// kernel with enter() is correct under either launch. Everything after enter() sees exactly what a normal stream-ordered
// launch would see: the arithmetic is unchanged (bitwise).
// Host side: pdl2::launch(kernel, grid, block, smem, stream, args...) = cudaLaunchKernelEx with
// cudaLaunchAttributeProgrammaticStreamSerialization = ENG_PDL2 (read once). Under stream capture this becomes a
// programmatic graph edge. Only kernels that call enter() (or griddepcontrol.wait) before touching any data written by
// earlier kernels may be launched this way.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdlib>

namespace pdl2 {
__device__ __forceinline__ void enter() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
inline bool on() {
  static int v = -1;
  if (v < 0) {
    const char* e = std::getenv("ENG_PDL2");
    v = (e && e[0] == '1') ? 1 : 0;
  }
  return v == 1;
}
template <typename... KArgs, typename... Args>
inline void launch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = on() ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...));
}
}  // namespace pdl2
