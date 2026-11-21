// kernel_capture.h -- relaunch a library kernel captured once at warm-up: the C++ mixed-step
// driver's path for kernels no C ABI or Triton replay reaches (CuTe DSL: vLLM's fused_indexer_q_rope_quant /
// dequant_gather_k on this deployment; CuTe DSL 4.8 has no AOT export), when every argument is constant per (bucket,
// layer) because the piece owns persistent buffers. kernel_capture.py captures the Python call on those buffers into a
// CUDA graph (thread-local capture mode), and kernel_capture_ext reads each kernel node (driver API): the function (or
// CUkernel -> function of the current context), grid, block, dynamic shared memory, cluster dimension, and the parameter
// bytes laid out by cuFuncGetParamInfo. fpr_kc_launch relaunches exactly those bytes with cuLaunchKernelEx on any stream
// of that device: bitwise by construction (same cubin, same parameters). Records are per device.
#pragma once
#include <cuda.h>
#include <stdint.h>
#include <string.h>

#include "fpr_cabi.h"
#include "fpr_cu_launch.h"   // (cuLaunchKernelEx from cudart: no -lcuda in the driver)

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_KC_MAX_PARAMS 64
#define FPR_KC_PARAM_BYTES 4096

typedef struct {
  CUfunction function;
  int32_t device;
  uint32_t grid[3], block[3];
  uint32_t shared;
  uint32_t cluster[3];        // 0 = no cluster attribute
  int32_t n_params;
  int32_t param_off[FPR_KC_MAX_PARAMS];
  int32_t param_size[FPR_KC_MAX_PARAMS];
  int32_t bytes_used;
  uint8_t bytes[FPR_KC_PARAM_BYTES] __attribute__((aligned(16)));
  char name[96];
} FprKcRecord;

static inline int fpr_kc_launch(const FprKcRecord* r, CUstream stream) {
  if (r == NULL || r->function == NULL || r->n_params > FPR_KC_MAX_PARAMS) return FPR_E_ARG;
  void* params[FPR_KC_MAX_PARAMS];
  for (int i = 0; i < r->n_params; ++i) params[i] = (void*)(r->bytes + r->param_off[i]);
  CUlaunchAttribute attr[1];
  int na = 0;
  if (r->cluster[0] > 0) {
    memset(&attr[0], 0, sizeof(attr[0]));
    attr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr[0].value.clusterDim.x = r->cluster[0];
    attr[0].value.clusterDim.y = r->cluster[1] ? r->cluster[1] : 1;
    attr[0].value.clusterDim.z = r->cluster[2] ? r->cluster[2] : 1;
    na = 1;
  }
  CUlaunchConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDimX = r->grid[0];
  cfg.gridDimY = r->grid[1];
  cfg.gridDimZ = r->grid[2];
  cfg.blockDimX = r->block[0];
  cfg.blockDimY = r->block[1];
  cfg.blockDimZ = r->block[2];
  cfg.sharedMemBytes = r->shared;
  cfg.hStream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = na;
  return fpr_cu_launch_kernel_ex(&cfg, r->function, params, 0) == CUDA_SUCCESS ? FPR_OK : FPR_E_CUDA;
}

#ifdef __cplusplus
}
#endif
