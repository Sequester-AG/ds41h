// idx2f_fcand8_cabi.h -- C ABI of the idx2f_fcand8 extension (idx2f_fcand8.cu: the f8 logits family, IDX2F_LOGITS=f8) for
// the C++ mixed-step driver, under the rules of fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_idx2f_fcand8_table() -> ext "idx2f_fcand8": "idx2f_fcand8.init", ".prep", ".logits", ".logits_dense", ".remap".
//   init: once per device (current device = call.device): resolves cuTensorMapEncodeTiled and sets the dynamic SMEM
//     attributes of every prep / logits instantiation (next_n 1 / 2 / 4) at the pybind path's sizes. No launch.
//   Every other function: the pybind path's host work (tensor maps) and ONE launch on call.stream with its grid / block /
//     SMEM (logits: one persistent CTA per SM), none when there are no requests / rows; no allocation, no device change.
//   Returns FPR_OK, FPR_E_ARG (the pybind path's value checks), FPR_E_INIT, FPR_E_CUDA (tensor-map encode, or
//   cudaGetLastError after the launch, as the pybind path's C10_CUDA_KERNEL_LAUNCH_CHECK). int = int32; matrices
//   row-major with unit column stride, row strides in elements.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
} FprIdx2fFcand8InitArgs;

// union / per-row compaction of the candidate blocks: rows = next_n x requests
typedef struct {
  FprCall call;
  const int32_t* cand;   // [rows, k] block ids (any order, -1 padded)
  int64_t ld_cand;
  int64_t rows;
  int64_t k;             // 1 .. 8,192; next_n * k % 32 == 0
  const int32_t* ctx;    // [rows] row ends
  int64_t next_n;        // 1, 2 or 4
  int64_t nwords_max;    // 1 .. 8,192 (bitmap words per row)
  int32_t* ublk;         // >= requests * next_n * k
  int32_t* uslot;        // >= requests * next_n * k * next_n
  int32_t* nunion;       // >= requests
  int32_t* cblk;         // [rows, >= k]
  int64_t ld_cblk;
  int32_t* clen;         // >= rows
} FprIdx2fFcand8PrepArgs;

// candidate-only paged MQA logits (= DeepGEMM sm90_fp8_paged_mqa_logits arithmetic); writes out[row, s < clen[row]]
typedef struct {
  FprCall call;
  const void* q;               // e4m3 [b, next_n, 32, 128] contiguous
  int64_t b;                   // requests, <= 1,024 per call
  int64_t next_n;              // 1, 2 or 4
  const uint8_t* kv;           // fused index-K pages [num_pages, 64, 1, 132], page stride page_stride bytes (% 16), 16-B aligned
  int64_t num_pages;
  int64_t page_stride;
  const float* weights;        // [>= b * next_n, 32] contiguous rows, 16-B aligned
  const int32_t* ctx;          // [b * next_n]
  const int32_t* block_table;  // [b, *]
  int64_t ld_bt;
  const int32_t* ublk;         // [b, ustride] (prep)
  const int32_t* uslot;        // [b, ustride, next_n]
  const int32_t* nunion;       // [b]
  int64_t ustride;             // ublk elements per request (= next_n * k)
  float* out;                  // [b * next_n, >= ustride / next_n * 8]
  int64_t ld_out;
  int64_t out_cols;
  const int32_t* upk;          // [b, ustride] packed (page << 3) | (blk & 7) (prep2) or NULL
} FprIdx2fFcand8LogitsArgs;

// dense paged MQA logits of every position [0, ctx[row]) (layers 2 / 8 / 14 / 20)
typedef struct {
  FprCall call;
  const void* q;               // e4m3 [b, next_n, 32, 128] contiguous
  int64_t b;                   // <= 1,024 per call
  int64_t next_n;
  const uint8_t* kv;           // [num_pages, page_tok, 1, 132], page_tok 64 (ratio 1) or 32 (ratio 2)
  int64_t num_pages;
  int64_t page_stride;
  int64_t page_tok;
  const float* weights;        // [>= b * next_n, 32]
  const int32_t* ctx;          // [b * next_n]
  const int32_t* block_table;  // [b, bt_cols]
  int64_t ld_bt;
  int64_t bt_cols;
  float* out;                  // [b * next_n, >= max ctx], ld_out % 4 == 0
  int64_t ld_out;
} FprIdx2fFcand8LogitsDenseArgs;

// compact top-k index -> request-local position, in place
typedef struct {
  FprCall call;
  int32_t* idx;          // [rows, k]
  int64_t ld_idx;
  int64_t rows;
  int64_t k;
  const int32_t* cblk;   // [rows, *] (prep)
  int64_t ld_cblk;
} FprIdx2fFcand8RemapArgs;

const FprExtTable* fpr_idx2f_fcand8_table(void);

#ifdef __cplusplus
}
#endif
