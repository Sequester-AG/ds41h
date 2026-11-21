// gather_k_cabi.h -- C ABI of gather_k.cu, rules of fpr/engine/mixed_cxx/fpr_cabi.h.
//
// vLLM's dequantize_and_gather_k_cache for the DSv4 fp8 record (584 B per token: 448 e4m3 | 64 bf16 RoPE | then, per page,
// block_size x 8 ue8m0 scale bytes after the page's data region), the CuTe DSL kernel this deployment runs
// (dequant_gather_k_cutedsl, fp8_dim 448, group 64), as a plain CUDA kernel callable without Python. The same per-element
// PTX (cvt.rn.f16x2.e4m3x2, cvt.rn.bf16.f16, mul.rn.bf16x2 by the bf16 power of two (s << 7)), so BITWISE equal:
//   for req r, i in [0, gl), gl = gather_lens ? gather_lens[r] : seq_lens[r], pos = seq_lens[r] - gl + i:
//     page = block_table[r * ld_bt + pos / block_size], o = pos % block_size
//     out[r * out_sr + (offset + i) * out_st + j] = bf16(e4m3 data[j]) * 2^(scale[j / 64] - 127)  (j < 448)
//                                                  = the record's bf16 RoPE element j - 448             (j >= 448)
//   other rows of out are not written.
// Table fpr_gatherk_table() -> ext "gatherk", function "gatherk.gather" (no scratch, no init). One launch on
// args->call.stream, grid (num_reqs, workers), 256 threads; none when num_reqs == 0. FPR_OK / FPR_E_ARG / FPR_E_CUDA.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
  void* out;                  // bf16, element strides out_sr (request) / out_st (row), unit inner stride, 16-B aligned rows
  int64_t out_sr, out_st;     // (out_sr, out_st % 8 == 0)
  int64_t max_rows;           // rows of out per request after `offset` (the CuTe kernel's out.shape[1] - offset: workers)
  const uint8_t* k_cache;     // [num_blocks] pages of page_stride bytes (k_cache.stride(0)), 8-B aligned
  int64_t page_stride;
  const int32_t* seq_lens;    // [num_reqs]
  const int32_t* gather_lens; // [num_reqs] or NULL (= seq_lens)
  const int32_t* block_table; // [num_reqs, ld_bt]
  int64_t ld_bt;
  int64_t num_reqs;
  int64_t block_size;         // tokens per page of this cache
  int64_t offset;             // first output row
} FprGatherkGatherArgs;

const FprExtTable* fpr_gatherk_table(void);

#ifdef __cplusplus
}
#endif
