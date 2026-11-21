// mhcf_cabi.h -- C ABI of kernels/mhc_fuse/mhcf.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The mHC boundary of a mixed step (Engine._split_bd): R > 320: fa + fb_out (main stream) + fb_mix (side stream);
// R <= 320: frow + fmix + fb_mix (side). Pointers: device, rows-contiguous, 16-B aligned (the pybind wrappers check
// these; the C path trusts the plan builder). Table: fpr_mhcf_table(); functions "mhcf.init", "mhcf.fa", "mhcf.frow",
// "mhcf.fmix", "mhcf.fb_out", "mhcf.fb_mix". The mode globals (set_nrs / set_fbo / set_fbm) are init-time settings.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { FprCall c; } FprMhcfInitArgs;                       // dynamic-smem opt-ins of every kernel on c.device

typedef struct {                                                      // = pybind fa2(y, rs, post, comb, lag, fnf, xn, scratch, nrs)
  FprCall c;
  const void* y;          // bf16 [R, D]
  const void* rs;         // bf16 [R, HC * D]
  float* post;            // [R, HC]
  float* comb;            // [R, 16]
  float* lag;             // [R, HC]
  const float* fnf;       // prep_f(fn): [NCH * FN_FLOATS / 2]
  void* xn;               // bf16 [>= R, HC * D]
  float* scratch;         // >= in_off(R) + R * D / 2 floats
  int32_t R;
  int32_t nrs;            // row splits (0 = auto: set_nrs or 3)
} FprMhcfFaArgs;

typedef struct {                                                      // = pybind frow(y, rs, post, comb, lag, normw, eps, xn, out, ptok_o)
  FprCall c;
  const void* y; const void* rs; float* post; float* comb; float* lag; const float* normw;
  float norm_eps;
  void* xn;               // bf16 [>= R, HC * D]
  void* out;              // bf16 [>= R, D]
  int32_t* ptok_o;        // [>= R] or NULL
  int32_t R;
} FprMhcfFrowArgs;

typedef struct {                                                      // = pybind fmix(xn, R, fnf, scratch, nrs)
  FprCall c;
  const void* xn; const float* fnf; float* scratch; int32_t R; int32_t nrs;
} FprMhcfFmixArgs;

typedef struct {                                                      // = pybind fb_out(scratch, R, normw, eps, out, ptok_o)
  FprCall c;
  float* scratch; const float* normw; float norm_eps; void* out; int32_t* ptok_o; int32_t R;
} FprMhcfFbOutArgs;

typedef struct {                                                      // = pybind fb_mix(scratch, R, scale, base, eps, hc_eps, iters, pre, post, comb)
  FprCall c;
  float* scratch; const float* scale; const float* base; float norm_eps; float hc_eps; int32_t iters;
  float* pre_o; float* post_o; float* comb_o; int32_t R;
} FprMhcfFbMixArgs;

typedef struct {   // = pybind frow_rs(slots, flag, epoch, signals, rs, post, comb, lag, normw, eps, xn, out, ptok_o): frow whose y
                   // is the 4-slot reduce-scatter sum, waited in-kernel on the round's flag (k_frow_rs<f32>)
  FprCall c;
  const void* slots;      // [4, R, D] fp32 (slots_f32 = 1) or bf16 (0), contiguous
  int32_t slots_f32;
  int32_t R;              // rs.size(0); slots.size(1) == R
  const uint32_t* flag;   // the RS round's flag word
  const uint32_t* epoch;  // its round counter
  uint32_t signals;       // per round
  const void* rs;         // bf16 [R, HC * D]
  float* post; float* comb; float* lag; const float* normw;
  float norm_eps;
  void* xn;               // bf16 [>= R, HC * D]
  void* out;              // bf16 [>= R, D]
  int32_t* ptok_o;        // [>= R] or NULL
} FprMhcfFrowRsArgs;

const FprExtTable* fpr_mhcf_table(void);

#ifdef __cplusplus
}
#endif
