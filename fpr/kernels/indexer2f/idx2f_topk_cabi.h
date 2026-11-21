// idx2f_topk_cabi.h -- C ABI of the idx2f_topk extension (idx2f_topk.cu) for the C++ mixed-step driver, under the rules of
// fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_idx2f_topk_table() -> ext "idx2f_topk": "idx2f_topk.init", ".block_max", ".topk", ".select_blocks", ".prep2".
//   init: once per device (current device = call.device): the dynamic shared-memory attributes of topk / select / prep2
//     (the pybind path keeps setting them lazily on first use). No launch.
//   Every other function: one launch on call.stream with the pybind path's grid / block / shared memory, none when the
//     row count is 0; no allocation, no device change (the pybind path's CUDAGuard is the caller's job here). Returns
//     FPR_OK, FPR_E_ARG (the pybind path's value checks), FPR_E_INIT (init missing), FPR_E_CUDA (cudaGetLastError after
//     the launch, as the pybind path's C10_CUDA_KERNEL_LAUNCH_CHECK). All matrices: row-major, unit column stride, row
//     strides in elements; int = int32.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
} FprIdx2fTopkInitArgs;

// out[r, b] = max of logits[r, 8b .. 8b + 7] within lens[r] (stock select's block score); rows <= 1,024 per call
typedef struct {
  FprCall call;
  const float* logits;  // [rows, >= 8 * ceil(max_len / 8)], ld % 4 == 0, 16-B aligned
  int64_t ld_logits;
  const int32_t* lens;  // [rows]
  int64_t rows;
  float* out;           // [rows, >= ceil(max_len / 8)]
  int64_t ld_out;
  int64_t max_len;
} FprIdx2fTopkBlockMaxArgs;

// exact top-k (key desc, lowest position on ties) of each row via the block-max threshold, ascending positions
typedef struct {
  FprCall call;
  const float* logits;  // [rows, *]
  int64_t ld_logits;
  const int32_t* lens;  // [rows]
  const float* bmax;    // [rows, nblocks <= 32,768]
  int64_t ld_bmax;
  int64_t nblocks;      // bmax.size(1) (the bound check of the pybind path)
  int32_t* out;         // [rows, >= k]
  int64_t ld_out;
  const int32_t* cblk;  // [rows, *] candidate blocks (positions = their tokens) or NULL (positions = column indices)
  int64_t ld_cblk;
  int64_t k;            // 1 .. 8,192
  int32_t* slow;        // >= 1 int32 (slow-path counter)
  int64_t stop;         // debug stage stop (0 = all)
  int64_t rows;
} FprIdx2fTopkTopkArgs;

// layer-20 candidate blocks from the block maxima (stock select's set), ascending, -1 padded
typedef struct {
  FprCall call;
  const float* bmax;       // [rows, nblocks <= 32,768]
  int64_t ld_bmax;
  int64_t nblocks;
  const int32_t* nb_lens;  // [rows]
  int32_t* out;            // [rows, >= k]
  int64_t ld_out;
  int64_t k;
  int64_t rows;
} FprIdx2fTopkSelectBlocksArgs;

// consumer union / compaction from ascending candidate lists: rows = 4 x requests
typedef struct {
  FprCall call;
  const int32_t* cand;         // [>= rows, 2048]
  int64_t ld_cand;
  const int32_t* lens;         // [rows]
  int64_t rows;                // % 4 == 0
  int32_t* ublk;               // >= rows * 2048
  int32_t* uslot;              // >= rows * 2048 * 4
  int32_t* nunion;             // >= rows / 4
  int32_t* clen;               // >= rows
  const int32_t* block_table;  // [requests, *] or NULL (with upk)
  int64_t ld_bt;
  int32_t* upk;                // >= rows * 2048 or NULL (with block_table)
} FprIdx2fTopkPrep2Args;

const FprExtTable* fpr_idx2f_topk_table(void);

#ifdef __cplusplus
}
#endif
