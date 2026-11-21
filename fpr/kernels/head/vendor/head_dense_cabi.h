// head_dense_cabi.h -- C ABI of kernels/head/vendor/dense.cu, the HSD drafter's head_dense (built with -DDENSE_PDL=0), per fpr/engine/mixed_cxx/fpr_cabi.h.
// One cudaLaunchKernelEx per function on call.stream, its own return code
// (no cudaGetLastError); no scratch, no allocation, no host sync, no per-device state (no init entry).
// Table: fpr_head_dense_table() -> ext "head_dense"; "head_dense.norm_rope2", "head_dense.norm_rope",
// "head_dense.rope_q_insert", "head_dense.inv_rope".
#pragma once
#include "../../attn_dense/dense_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

const FprExtTable* fpr_head_dense_table(void);

#ifdef __cplusplus
}
#endif
