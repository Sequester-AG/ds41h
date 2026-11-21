// dg_cabi.h -- C ABI wrapper of DeepGEMM's two DENSE8 calls (vllm.third_party.deep_gemm 2.8.0, TORCH_LIBRARY ops) for the
// C++ mixed-step driver, under fpr/engine/mixed_cxx/fpr_cabi.h with ONE documented exception: DeepGEMM's num_sms is process-global, so every call runs under one process mutex
//   old = get_num_sms(); set_num_sms(args.num_sms); <the op>; set_num_sms(old)
// (args.num_sms = today's per-device value base - 2 * device: kernel choice follows it and it is the INVALID_HANDLE
// workaround of attn_act8._dg_device). Only the host launch work is serialized.
//
// Table: fpr_dg_table() -> ext "dg": "dg.fp8_gemm_nt" (wo_b / W8A8 linears: DeepGEMM fp8_fp4_gemm_nt, recipe None,
// compiled_dims "nk", disable_ue8m0_cast true = vLLM's fp8_gemm_nt(..., is_deep_gemm_e8m0_used=False)) and "dg.fp8_einsum"
// (wo_a: fp8_einsum("bhr,hdr->bhd", recipe (1, 128, 128))). Both: the dispatcher's op (the same impl the Python
// torch.ops call reaches), on from_blob views of the caller's buffers, under CUDAStreamGuard(args.call.stream) (DeepGEMM
// launches on torch's current stream). No allocation by the wrapper.
// Warm-up (rule 2's one-time state): the pybind dg_cabi.warm_* functions (Python thread, engine / bucket init) run the
// same calls and record each (op, n, k, m, num_sms, device) key; a C call whose key is not warmed returns FPR_E_SHAPE
// without launching, so DeepGEMM's first-use JIT compile can never happen inside a driver step. The C call also returns
// FPR_E_SHAPE if the op allocated through the caching allocator (a scale-layout transform: the plan's layouts are wrong).
// Returns FPR_OK, FPR_E_ARG, FPR_E_SHAPE, FPR_E_CUDA (an exception from the op).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { FPR_DG_E4M3 = 1, FPR_DG_F32 = 2, FPR_DG_BF16 = 3 };

// a strided tensor view (rank <= 4): sizes / strides in elements
typedef struct {
  void* ptr;
  int64_t sizes[4];
  int64_t strides[4];
  int32_t ndim;
  int32_t dtype;   // FPR_DG_*
} FprDgView;

// d [m, n] bf16 = (a [m, k] e4m3, sfa [m, k/128] f32 MN-major TMA aligned: act_quant_128) @ (b [n, k] e4m3, sfb: the
// FoldedWeight ws layout)^T
typedef struct {
  FprCall call;
  FprDgView a, sfa, b, sfb, d;
  int64_t num_sms;
} FprDgGemmArgs;

// d [m, h, dd] bf16 = einsum("bhr,hdr->bhd", (a [m, h, r] e4m3, sfa), (b [h, dd, r] e4m3, sfb)), recipe (1, 128, 128)
typedef struct {
  FprCall call;
  FprDgView a, sfa, b, sfb, d;
  int64_t num_sms;
} FprDgEinsumArgs;

const FprExtTable* fpr_dg_table(void);

#ifdef __cplusplus
}
#endif
