// w8a16_cabi.h -- C ABI of the w8a16 extension (w8a16.cu: w8a16_ext and its -D variants) for the C++ mixed-step driver,
// under the rules of fpr/engine/mixed_cxx/fpr_cabi.h.
//
// Table: fpr_w8a16_table() -> ext "w8a16", functions "w8a16.init" (FprW8a16InitArgs) and "w8a16.gemm" (FprW8a16GemmArgs).
//   init: once per device (current device = args->call.device) before any C launch: the dynamic shared-memory attribute of
//     every kernel instantiation of this build (the pybind path keeps setting it lazily on first use, as before). No launch.
//   gemm: the fused W8A16 GEMM: y[t, n] = bf16(sum_k x[t, k] * W[n, k]) with W the packed fp8 weights (exact dequant, bf16
//     MMA, fp32 accumulation), optionally RoPE of y's dims 448..511 per 512 (q_b: pos) + the window KV insert (kv / slot /
//     cache), or the inverse RoPE of x (wo_a: inv_pos). One cudaLaunchKernelEx on args->call.stream (PDL attribute =
//     W8_PDL); host work = one cuTensorMapEncodeTiled. No allocation: split-K partials (ws) and the per-tile arrival
//     counters (cnt, monotonic across calls) are the caller's, as in the pybind path.
//   Returns FPR_OK; FPR_E_ARG (the pybind path's checks below); FPR_E_SHAPE (token tile not built: W8_NTMIN); FPR_E_INIT
//   (init not called on this device); FPR_E_CUDA (tensor-map encode, or the launch / cudaGetLastError after it).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
} FprW8a16InitArgs;

typedef struct {
  FprCall call;
  const void* x;              // bf16 [ntok, kx], row stride x_ld elements (% 8 == 0 for the TMA map), unit column stride
  int64_t ntok;               // >= 1; > 128 needs splits == 1 (grid z = 128-row slices, <= 65,535)
  int64_t kx;
  int64_t x_ld;
  const uint8_t* packed;      // [nchunk, tiles, chunk_bytes] contiguous (w8a16.chunk_bytes() / W8A16_CHUNK_BYTES)
  int64_t nchunk;
  int64_t tiles;
  void* y;                    // bf16 [ntok, >= tiles * 256], row stride ldy elements, unit column stride
  int64_t ldy;
  int64_t y_cols;
  int64_t rows_per_group;
  int64_t kx_per_group;
  int64_t splits;             // 1 .. W8_SMAX (8), <= nchunk; > 1: splits * tiles <= SMs (one resident wave)
  const int64_t* pos;         // RoPE of y (q_b): [ntok] or NULL
  const float* cs;            // f32 [positions, 64] (with pos or inv_pos)
  const void* kv;             // bf16 [ntok, 512] contiguous (window KV insert) or NULL
  const int64_t* slot;        // [ntok] (with kv)
  uint8_t* cache;             // window KV pages (with kv)
  int64_t page_stride;
  int64_t shift;
  const int64_t* inv_pos;     // inverse RoPE of x (wo_a): [ntok] or NULL (exclusive with pos)
  float* ws;                  // split-K: f32 >= splits * tiles * 128 * 260
  int64_t ws_floats;
  int32_t* cnt;               // split-K: >= tiles arrival counters
  int64_t cnt_n;
  int64_t* prof;              // W8_PROF builds only, else NULL
} FprW8a16GemmArgs;

#define W8A16_CHUNK_BYTES 32896  // w8::CHUNK (2 x (16,384 B of fragments + 64 B of scales)); w8a16.chunk_bytes() returns it

const FprExtTable* fpr_w8a16_table(void);

#ifdef __cplusplus
}
#endif
