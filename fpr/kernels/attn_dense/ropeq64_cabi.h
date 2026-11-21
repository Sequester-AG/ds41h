// ropeq64_cabi.h -- C ABI of the ropeq64 extension (ropeq64.cu: ropeq64_ext) for the C++ mixed-step driver, under the rules
// of fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_ropeq64_table() -> ext "ropeq64", function "ropeq64.inv_rope_fp8_quant" (no scratch). The pybind function
// allocates out / scales itself (unchanged); here they are the caller's (rule 2), sizes below. One launch on
// args->call.stream (grid (ta, n_groups), 256 threads), none when t == 0. Returns FPR_OK, FPR_E_ARG, or FPR_E_CUDA
// (cudaGetLastError right after the launch, as the pybind path's C10_CUDA_KERNEL_LAUNCH_CHECK).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

// inverse RoPE (dims 448..511) + 1x128 FP8 quant of the attention output for the DENSE8 wo_a einsum, n_groups x 8 heads.
//   out    e4m3 [n_groups, t, 8 * 512] group-major: (g, r, j) at (g * t + r) * 4096 + j   (n_groups * t * 4096 bytes)
//   scales f32, ta = round_up(t, 4): (g, r, c) at g * 32 * ta + c * ta + r, c = 0..31     (n_groups * 32 * ta floats)
// (the pybind function returns out.transpose(0, 1) and the strided [n_groups, t, 32] scale view, transposed)
typedef struct {
  FprCall call;
  const void* o;              // bf16 [t, 8 * n_groups, 512], element strides o_stride_t / o_stride_h (both % 8 == 0),
  int64_t o_stride_t;         // unit last stride, 16-B aligned base
  int64_t o_stride_h;
  const int64_t* positions;   // [>= t]
  const float* cos_sin;       // f32 [max_pos, 64], row stride cs_stride elements (% 4 == 0), unit column stride
  int64_t cs_stride;
  int64_t t;
  int64_t n_groups;
  uint8_t* out;
  float* scales;
} FprRopeq64InvRopeFp8QuantArgs;

const FprExtTable* fpr_ropeq64_table(void);

#ifdef __cplusplus
}
#endif
