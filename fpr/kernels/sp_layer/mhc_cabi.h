// mhc_cabi.h -- C ABI of kernels/sp_layer/mhc.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The final mHC boundary of a mixed step (Engine.p_final, R > 320 rows per chunk): k1a_x (K1a phase 1: hc_post -> xn, the
// critical path) + k1b_out (collapse with the lagged pre + RMSNorm -> out, optional prescale exponents). Pointers: device,
// rows contiguous (the pybind wrappers take row-chunk views [c0:c1] of contiguous tensors). Build variants (MHC_*) are
// compiled in; no per-device state (no dynamic SMEM), so no init entry. Table: fpr_mhc_table(); "mhc.k1a_x", "mhc.k1b_out".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind k1a_x(y, rs, post, comb, xn)
  FprCall c;
  const void* y;                       // bf16 [R, D]
  const void* rs;                      // bf16 [R, 4, D] (the residual stream)
  const float* post;                   // [R, 4]
  const float* comb;                   // [R, 4, 4]
  void* xn;                            // bf16 [R, 4, D] out
  int32_t R;                           // MHC_ROWS_OK(R): 1 .. 2^20 (else FPR_E_SHAPE)
} FprMhcK1aXArgs;

typedef struct {                       // = pybind k1b_out(xn, lag, normw, eps, out, ptok_o)
  FprCall c;
  const void* xn;                      // bf16 [R, 4, D]
  const float* lag;                    // [R, 4] (the lagged pre)
  const float* normw;                  // [D]
  float norm_eps;
  void* out;                           // bf16 [R, D]
  int32_t* ptok_o;                     // [R] prescale exponents, or NULL
  int32_t R;                           // grid (xn.size(0))
} FprMhcK1bOutArgs;

const FprExtTable* fpr_mhc_table(void);

#ifdef __cplusplus
}
#endif
