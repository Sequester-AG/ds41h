// rtag_cabi.h -- C ABI of kernels/hsdtail/rtag.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// HSDT's rtag switch: the drafter router + its all-gather as ONE kernel per GPU for drafter buckets of <= 4 rows per rank
// (f push to every rank, gate GEMV, top-3 by the last CTA, routing-row push; the consumer then only waits). Gather buffers /
// flags of every rank and the epoch / count / logits scratch are the caller's (HSDT.configure(rtag=True)). Peer access on.
// Table: fpr_rtag_table(); "rtag.rtag".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_RTAG_W 4

typedef struct {                       // = pybind rtag(f, gate, gbias, lg, cnt, rows, dslot, r0, kq, rank, nc, fbufs, rbufs, flags, epoch)
  FprCall c;                           // c.rank: this rank
  const void* f; int32_t R;            // bf16 [R <= 4, 5120] contiguous
  const void* gate;                    // bf16 [128, 5120]
  const float* gbias;                  // [128]
  float* lg;                           // [W * 128] logits scratch
  int32_t* cnt;                        // [1] CTA arrival counter (int32)
  int32_t* rows;                       // [R, 16] routing rows (int32)
  const int64_t* dslot;                // dmask: the step's drafting-slot table (g_dslot) or NULL
  int32_t r0, kq, nc;                  // dmask row base (rank * D_pad), rows per request (k), CTAs per destination
  void* fbufs[FPR_RTAG_W];             // f gather buffer on each rank
  void* rbufs[FPR_RTAG_W];             // routing-row gather buffer on each rank
  uint32_t* flags[FPR_RTAG_W];         // flag on each rank
  uint32_t* epoch;                     // this rank's round counter
} FprRtagArgs;

const FprExtTable* fpr_rtag_table(void);

#ifdef __cplusplus
}
#endif
