// ctx3_cabi.h -- C ABI of kernels/hsdtail/ctx3.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// HSDT's ctx3f switch: the 3 drafter blocks' context-KV norm_rope + window-record insert (fp8 records into the drafter's
// window pool pages) as ONE launch, from the batched wkv GEMM's output kv3 [R, 3 x 512] (hsdt_c1e). No init, no scratch.
// Table: fpr_ctx3_table(); "ctx3.norm_insert3".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind norm_insert3(kv3, w, out, pos, cos_sin, slot, cache, page_stride, shift, eps)
  FprCall c;
  const void* kv3;                     // bf16 [R, 3 x 512], column stride 1
  int64_t ld;                          // kv3 row stride (elements)
  int32_t R;                           // rows (0: nothing launched)
  int32_t shift;                       // log2(window page tokens)
  const float* w[3];                   // kv_norm weights [512] fp32 per block
  void* out[3];                        // bf16 [>= R, 512] per block (the normed rows; block 2's = kv_ctx)
  const int64_t* pos;                  // [>= R] positions
  const float* cos_sin;                // [max_pos, 64] fp32
  const int64_t* slot;                 // [>= R] window-pool slots (-1: no insert)
  void* cache[3];                      // uint8 window pool base per block (drafter layers 40..42)
  int64_t page_stride;                 // bytes per window page
  float eps;
} FprCtx3NormInsert3Args;

const FprExtTable* fpr_ctx3_table(void);

#ifdef __cplusplus
}
#endif
