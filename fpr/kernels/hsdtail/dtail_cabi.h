// dtail_cabi.h -- C ABI of kernels/hsdtail/dtail.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// HSD's draft head (e4m3, per-row quantized activations x per-vocab-row scales) for <= 12 draft rows, optionally with the
// k Markov draws + their lamport exchange (markov = 1; HSDT's "dtail" switch) or head only. Persistent grid of 132 CTAs (one per SM, co-resident: grid barriers)
// so it must not run concurrently with another persistent kernel on the device. Scratch (part / bar / xloc) and the
// exchange peers are the caller's (HSDT.configure: torch tensors; bar is a monotonic counter, never reset). Per-device state:
// the kernel's max dynamic SMEM attribute, set by the first call on each device (atomic mask; no init entry needed).
// Table: fpr_dtail_table(); "dtail.dtail".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind dtail(...) (dtail.cu); pointers as the tensors' data_ptr, strides in elements
  FprCall c;                           // c.rank: this rank (its slot in every exchange buffer)
  const void* hd; int32_t R;           // bf16 [R = n * k, 5120]
  const void* wq; const float* ws;     // e4m3 head slice [loc, 5120], fp32 scales [loc]
  int32_t loc, vstart;
  const void* w2; const void* w1;      // bf16 [loc, 256], bf16 [V, 256]
  const void* bias0; int64_t bias0_rs; // bf16 [n, loc] slot-0 bias, row stride
  const float* noise; int64_t nz_s0, nz_s1;
  const int32_t* idx; int64_t idx_s0, idx_s1;
  const float* temp;
  void* cache; int64_t c_s0, c_s1;     // bf16
  int64_t* tokens; int64_t tok_s0;
  float* tlogit; float* lse; int64_t vp_s0; int32_t sink;
  void* base_out;                      // bf16 [R, loc] or NULL (markov mode only)
  float* part;                         // >= 4 * 132 * 5 floats
  uint64_t* bar;                       // [1] monotonic arrival counter
  float* xloc;                         // this rank's exchange buffer [2][4][4][4][8] words
  float* xpeers[4];                    // every rank's exchange buffer
  int32_t n, k;                        // requests (<= 4), Markov slots (<= 4)
  uint64_t* dbg;                       // [132][16] timestamps or NULL
  int32_t markov;                      // 1: Markov draws + exchange; 0: head only
  const void* xq; const float* xsg;    // pre-quantized rows e4m3 [>= R, 5120] + scales [>= R], or NULL (quantize inside)
} FprDtailArgs;

const FprExtTable* fpr_dtail_table(void);

#ifdef __cplusplus
}
#endif
