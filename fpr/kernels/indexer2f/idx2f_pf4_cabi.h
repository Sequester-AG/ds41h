// idx2f_pf4_cabi.h -- C ABI of the idx2f_pf4 extension (idx2f_pf4.cu: prefill dense logits with key tiles shared by groups
// of <= 4 pseudo-requests of one sequence; IDX2F_PF=1) for the C++ mixed-step driver, under the rules of
// fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_idx2f_pf4_table() -> ext "idx2f_pf4": "idx2f_pf4.init", "idx2f_pf4.logits_prefill".
//   init: once per device (current device = call.device): resolves cuTensorMapEncodeTiled and sets the dynamic SMEM
//     attribute of every ABL instantiation at its largest size (the pybind path keeps setting it lazily). No launch.
//   logits_prefill: two tensor maps + ONE launch on call.stream (one persistent CTA per SM), none when b == 0 or g == 0;
//     no allocation, no device change. The group starts (gstart) are the step plan's (idx2f.pf_groups / _pf_gstart: the
//     pre-step uploads them; today a pinned H2D per slice). Returns FPR_OK, FPR_E_ARG (the pybind path's value checks),
//     FPR_E_INIT, FPR_E_CUDA.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
} FprIdx2fPf4InitArgs;

typedef struct {
  FprCall call;
  const void* q;               // e4m3 [b, 4, 32, 128] contiguous
  int64_t b;                   // pseudo-requests
  const uint8_t* kv;           // fused index-K pages [num_pages, page_tok, 1, 132], page_tok 64 / 32, 16-B aligned
  int64_t num_pages;
  int64_t page_stride;         // bytes, % 16 == 0, >= page_tok * 132
  int64_t page_tok;
  const float* weights;        // [>= 4b, 32] contiguous rows, 16-B aligned
  const int32_t* ctx;          // [>= 4b]
  const int32_t* block_table;  // [b, bt_cols] (the members of a group share their sequence's pages: its first row is read)
  int64_t ld_bt;
  int64_t bt_cols;
  const int32_t* gstart;       // [g + 1] group starts (each group <= 4 pseudo-requests of one sequence)
  int64_t g;                   // 0 .. 1,024
  float* out;                  // [>= 4b, >= max ctx], ld_out % 4 == 0
  int64_t ld_out;
  int64_t abl;                 // 0 = production (1, 2, 3, 4, 6, 7: timing ablations)
} FprIdx2fPf4LogitsPrefillArgs;

const FprExtTable* fpr_idx2f_pf4_table(void);

#ifdef __cplusplus
}
#endif
