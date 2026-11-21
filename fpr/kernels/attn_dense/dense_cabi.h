// dense_cabi.h -- C ABI of the attn_dense extension (dense.cu: attn_dense_ext / attn_dense_ext_pdl) for the C++ mixed-step
// driver, under the rules of fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_attn_dense_table() -> ext "attn_dense" (DENSE_PDL 0) or "attn_dense_pdl" (DENSE_PDL 1), functions
//   "attn_dense.norm_rope2", "attn_dense.norm_rope", "attn_dense.rope_q_insert", "attn_dense.inv_rope".
// Every function: grid / block exactly as the pybind wrapper (which builds these structs and calls the same functions),
// one cudaLaunchKernelEx on args->call.stream (PDL attribute = DENSE_PDL), no scratch, no allocation, no host sync.
// Returns FPR_OK, FPR_E_ARG (shape rules below) or FPR_E_CUDA (the launch's own return code; no cudaGetLastError, so
// a pending error of an unrelated earlier launch is not consumed: same as the pybind path).
// Tensors are passed as raw device pointers with element strides (bf16 = 2 B, f32 = 4 B, int64 positions / slots).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

// both attention-prologue norms in one launch (grid 2 * rows): oq = bf16(RMSNorm(xq) * wq);
// ok = RoPE_last64(bf16(RMSNorm(xk) * wk)) with cos_sin[pos[r]]. dk >= 64.
typedef struct {
  FprCall call;
  const void* xq;        // bf16 [rows, dq], row stride ldxq (elements), unit column stride
  int64_t ldxq;
  const float* wq;       // f32 [dq]
  void* oq;              // bf16 [rows, dq], row stride ldoq
  int64_t ldoq;
  int64_t dq;
  const void* xk;        // bf16 [rows, dk], row stride ldxk
  int64_t ldxk;
  const float* wk;       // f32 [dk]
  void* ok;              // bf16 [rows, dk], row stride ldok
  int64_t ldok;
  int64_t dk;
  int64_t rows;
  float eps;
  const int64_t* pos;    // [rows]
  const float* cos_sin;  // f32 [positions, 64]
} FprAttnDenseNormRope2Args;

// one norm (grid rows): out = bf16(RMSNorm(x) * w), then RoPE of the last 64 dims when pos != NULL (d >= 64).
typedef struct {
  FprCall call;
  const void* x;         // bf16 [rows, d], row stride ldx
  int64_t ldx;
  const float* w;        // f32 [d]
  void* out;             // bf16 [rows, d], row stride ldo
  int64_t ldo;
  int64_t d;
  int64_t rows;
  float eps;
  const int64_t* pos;    // [rows] or NULL
  const float* cos_sin;  // f32 [positions, 64] (with pos)
} FprAttnDenseNormRopeArgs;

// RoPE(q) in place on dims 448..511 of every head (grid t), then (do_insert) the window KV record of kv[t] at
// slot_mapping[t] (< 0 = skip): page = slot >> shift, 576-B records, 8-B scale groups after 576 << shift bytes.
typedef struct {
  FprCall call;
  void* q;                    // bf16 [t, nh, 512] contiguous, nh 16 or 64
  const void* kv;             // bf16 [t, 512] contiguous
  const int64_t* pos;         // [t]
  const float* cos_sin;       // f32 [positions, 64]
  const int64_t* slot_mapping;  // [t]
  uint8_t* cache;             // window KV pages
  int64_t page_stride;        // bytes
  int64_t shift;              // log2(tokens per page)
  int64_t t;
  int64_t nh;
  int32_t do_insert;          // 0 / 1
} FprAttnDenseRopeQInsertArgs;

// inverse RoPE in place on o[t, :, 448:512] (grid t).
typedef struct {
  FprCall call;
  void* o;                    // bf16 [t, nh, 512] contiguous, nh 16 or 64
  const int64_t* pos;         // [t]
  const float* cos_sin;       // f32 [positions, 64]
  int64_t t;
  int64_t nh;
} FprAttnDenseInvRopeArgs;

const FprExtTable* fpr_attn_dense_table(void);

#ifdef __cplusplus
}
#endif
