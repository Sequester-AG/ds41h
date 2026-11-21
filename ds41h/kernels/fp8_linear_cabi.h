// fp8_linear_cabi.h -- C ABI of ds41h_fp8_linear's act_quant_128 (fp8_linear.cu) for the FPR C++ mixed-step driver, under
// the rules of fpr/engine/mixed_cxx/fpr_cabi.h. fp8_linear.cu compiles this part only when that
// header is present (a ds41h tree without fpr/ builds as before).
//
// Table: fpr_fp8_linear_table() -> ext "fp8_linear", function "fp8_linear.act_quant_128" (no scratch). The pybind function
// allocates q / scales itself (unchanged); here they are the caller's (rule 2). One launch on args->call.stream (grid m,
// round_up(kp / 128 * 8, 32) threads), none when m == 0. Returns FPR_OK, FPR_E_ARG (the pybind path's checks),
// FPR_E_CUDA (cudaGetLastError right after the launch, as the pybind path).
#pragma once
#include "../../fpr/engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

// bf16 [m, k] -> q e4m3 [m, kp] (contiguous; columns [k, kp) = +0) and fp32 scales per (row, 128-wide K block), MN-major
// TMA aligned: (row, kb) at scales[kb * ma + row], ma = round_up(m, 4)   (kp / 128 * ma floats).
typedef struct {
  FprCall call;
  const void* x;      // bf16 [m, k], unit column stride, row stride ldx elements (% 8 == 0), 16-B aligned
  int64_t ldx;
  int64_t m;
  int64_t k;          // % 16 == 0
  int64_t kp;         // k rounded up to 128 (<= 16,384)
  int32_t ue8m0;      // 1: power-of-two (UE8M0) scales; 0: amax / 448
  uint8_t* q;
  float* scales;
} FprFp8LinearActQuant128Args;

const FprExtTable* fpr_fp8_linear_table(void);

#ifdef __cplusplus
}
#endif
