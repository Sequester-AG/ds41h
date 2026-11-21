// head_mhc2_cabi.h -- C ABI of kernels/head/vendor/mhc2.cu, the HSD drafter's head_mhc2, per
// fpr/engine/mixed_cxx/fpr_cabi.h. vendor/mhc2.cu was byte-identical to sp_layer/mhc.cu before its C ABI and
// carries that conversion (k1a_x, k1b_out: sp_layer/mhc_cabi.h's FprMhcK1aXArgs / FprMhcK1bOutArgs) plus the drafter's
// boundary pieces below (drafter.py _bx() with HSD mhc2 on: boundary, boundary_rs, split k1a / k1a_rs + k1b_out + k1b_mix).
// Any rows per call (MHC_ROWS_OK: 1 .. 2^20, else FPR_E_SHAPE): K1a v2 in 64-row CTA slabs. Pointers: device, rows
// contiguous (row-chunk views of contiguous tensors are fine). part: [160 chunks, R, 25] fp32 = 16,000 R bytes, written
// by K1a, read by K1b (boundary / boundary_rs: scratch, ws_bytes; k1a / k1a_rs -> k1b_mix: the caller's hand-over buffer).
// fn: prep_fn(hc_*_fn) (load time, pybind only): float2 B fragments [160][16][3][2][32] (hi / lo tf32).
// Build variants (MHC_*) are compiled in; no per-device state, so no init entry.
// Table: fpr_head_mhc2_table(); "head_mhc2.boundary", "head_mhc2.boundary_rs", "head_mhc2.k1a", "head_mhc2.k1a_rs",
// "head_mhc2.k1a_x", "head_mhc2.k1b_out", "head_mhc2.k1b_mix".
#pragma once
#include "../../sp_layer/mhc_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {   // = pybind boundary(y, rs, post, comb, lag, fn, scale, base, normw, eps, hc_eps, iters, xn, pre_o, post_o, comb_o, out, part)
  FprCall c;
  const void* y;                       // bf16 [R, D] (the sublayer output)
  const void* rs;                      // bf16 [R, 4, D] (the residual stream)
  const float* post;                   // [R, 4]
  const float* comb;                   // [R, 4, 4]
  const float* lag;                    // [R, 4] (the lagged pre)
  const void* fn;                      // prep_fn(hc_fn)
  const float* scale;                  // [3]
  const float* base;                   // [24]
  const float* normw;                  // [D]
  float norm_eps;
  float hc_eps;
  int32_t iters;
  void* xn;                            // bf16 [R, 4, D] out (the new residual stream)
  float* pre_o;                        // [R, 4] out
  float* post_o;                       // [R, 4] out
  float* comb_o;                       // [R, 4, 4] out
  void* out;                           // bf16 [R, D] out (the next sublayer's input)
  float* part;                         // scratch [160, R, 25] fp32 (ws_bytes)
  int32_t R;                           // y.size(0)
} FprHeadMhc2BoundaryArgs;

typedef struct {   // = pybind boundary_rs(recv, flag, epoch, signals, rs, post, comb, lag, fn, scale, base, normw, eps, hc_eps, iters, xn, pre_o, post_o, comb_o, out, part, ptok_o)
  FprCall c;
  const void* recv;                    // bf16 [4, R, D] RS receive slots (slot = source rank), rows contiguous
  int64_t recv_sstride;                // elements between the slots (recv.stride(0)): R * D, or R_total * D for a row chunk
  const uint32_t* flag;                // the RS flag (waits for flag >= *epoch * signals)
  const uint32_t* epoch;               // this rank's RS round counter
  uint32_t signals;                    // per round (W * ncta)
  const void* rs;                      // bf16 [R, 4, D]
  const float* post;
  const float* comb;
  const float* lag;
  const void* fn;
  const float* scale;
  const float* base;
  const float* normw;
  float norm_eps;
  float hc_eps;
  int32_t iters;
  void* xn;
  float* pre_o;
  float* post_o;
  float* comb_o;
  void* out;
  float* part;                         // scratch [160, R, 25] fp32 (ws_bytes)
  int32_t* ptok_o;                     // [R] prescale exponents, or NULL
  int32_t R;                           // recv.size(1)
} FprHeadMhc2BoundaryRsArgs;

typedef struct {   // = pybind k1a(y, rs, post, comb, fn, xn, part)
  FprCall c;
  const void* y;
  const void* rs;
  const float* post;
  const float* comb;
  const void* fn;
  void* xn;                            // bf16 [R, 4, D] out
  float* part;                         // [160, R, 25] fp32 out (-> k1b_mix)
  int32_t R;                           // y.size(0)
} FprHeadMhc2K1aArgs;

typedef struct {   // = pybind k1a_rs(recv, flag, epoch, signals, rs, post, comb, fn, xn, part)
  FprCall c;
  const void* recv;                    // bf16 [4, R, D], rows contiguous
  int64_t recv_sstride;                // recv.stride(0) (elements)
  const uint32_t* flag;
  const uint32_t* epoch;
  uint32_t signals;
  const void* rs;
  const float* post;
  const float* comb;
  const void* fn;
  void* xn;
  float* part;                         // [160, R, 25] fp32 out
  int32_t R;                           // recv.size(1)
} FprHeadMhc2K1aRsArgs;

typedef struct {   // = pybind k1b_mix(part, R, scale, base, eps, hc_eps, iters, pre_o, post_o, comb_o)
  FprCall c;
  const float* part;                   // [160, R, 25] fp32 (k1a / k1a_rs)
  const float* scale;
  const float* base;
  float norm_eps;
  float hc_eps;
  int32_t iters;
  float* pre_o;                        // [R, 4] out
  float* post_o;                       // [R, 4] out
  float* comb_o;                       // [R, 4, 4] out
  int32_t R;                           // grid (the pybind R)
} FprHeadMhc2K1bMixArgs;

const FprExtTable* fpr_head_mhc2_table(void);

#ifdef __cplusplus
}
#endif
