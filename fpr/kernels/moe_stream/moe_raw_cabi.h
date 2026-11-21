// moe_raw_cabi.h -- C ABI of kernels/moe_stream/moe_raw.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The mixed step's MoE launches (Engine.p_router / p_moe, T = W R gathered rows): router_top_pack3 (gate logits -> ids /
// weights / 64-B routing row), swiglu (shared expert), prepr5 (token images by k_gather5, route2's EC numbering), gemm1r /
// gemm2r (FP4 GEMMs on the raw expert layout and route2's tables: route2_cabi.h), combine (6 slots + shared -> bf16
// partial). Raw layout: w13 u8 [E, 1152, 2560], s13 [E, 1152, 160], w2 [E, 5120, 288], s2 [E, 5120, 18].
// Table: fpr_moe_raw_table(); "moe_raw.init" (per device: the k_gemm<1> / k_gemm<2> dynamic-SMEM opt-ins; once per process:
// the SM count, the cuTensorMapEncodeTiled entry point, pdl2's ENG_PDL2 latch), "moe_raw.router_top_pack3", "moe_raw.swiglu",
// "moe_raw.prepr5", "moe_raw.gemm1r", "moe_raw.gemm2r", "moe_raw.combine". Build variants (MS3_* / MSR_*) are compiled in.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { FprCall c; } FprMoeRawInitArgs;

typedef struct {                       // = pybind router_top_pack3(logits, bias, ids, wts, route_scale, row, valid, ptok)
  FprCall c;
  const float* logits;                 // [T, 384] fp32
  const float* bias;                   // [384]
  int32_t* ids;                        // [T, 6] out
  float* wts;                          // [T, 6] out
  float route_scale;
  void* row;                           // [T, 64 B] routing rows out
  const void* valid;                   // [T] bool / uint8
  const int32_t* ptok;                 // [T] prescale exponents (into row bytes 48..51)
  int32_t T;
} FprMoeRawRouterTopPack3Args;

typedef struct {                       // = pybind swiglu(gu, h)
  FprCall c;
  const void* gu;                      // bf16 [T, 2 I]
  void* h;                             // bf16 [T, I] out
  int32_t T;
  int32_t I;                           // % 8 == 0 (else FPR_E_ARG)
} FprMoeRawSwigluArgs;

typedef struct {                       // = pybind prepr5(x, ecs, counts, img, rows, sms, ctr)
  FprCall c;
  const void* x;                       // bf16 [T, 5120]
  void* ecs;                           // route2's EC records (ec.inv written)
  const int32_t* counts;               // route2's counts
  void* img;                           // token image out
  const void* rows;                    // [T, 64 B] routing rows (prescale int32 at byte 48)
  int32_t sms;                         // grid = MSR_G5B x sms
  int32_t* ctr;                        // int32 [2] zeros if built with MSR_G5BS > 0 (else FPR_E_ARG), may be NULL otherwise
} FprMoeRawPrepr5Args;

typedef struct {                       // = pybind gemm1r(w13, s13, ecs, tiles, counts, img, himg, grid)
  FprCall c;
  const void* w13;                     // raw u8 [E, 1152, 2560]
  const void* s13;                     // raw ue8m0 [E, 1152, 160]
  int64_t E;                           // experts in w13 / s13 (w13.size(0))
  const void* ecs;
  const void* tiles;                   // route2's t1
  const int32_t* counts;               // counts[1] = GEMM1 tiles
  const void* img;                     // token image (prepr5)
  void* himg;                          // H image out
  int32_t grid;                        // CTAs (0 = SM count)
} FprMoeRawGemm1rArgs;

typedef struct {                       // = pybind gemm2r(w2, s2, ecs, tiles, counts, himg, yout, grid)
  FprCall c;
  const void* w2;                      // raw u8 [E, 5120, 288]
  const void* s2;                      // raw ue8m0 [E, 5120, 18]
  int64_t E;                           // experts in w2 / s2 (w2.size(0))
  const void* ecs;
  const void* tiles;                   // route2's t2
  const int32_t* counts;               // counts[2] = GEMM2 tiles
  const void* himg;                    // H image (gemm1r)
  void* yout;                          // bf16 [slots, 5120] out
  int32_t grid;                        // CTAs (0 = SM count)
} FprMoeRawGemm2rArgs;

typedef struct {                       // = pybind combine(Y, slotmap, sh, out)
  FprCall c;
  const void* Y;                       // bf16 [slots, 5120]
  const int32_t* slotmap;              // [T, 6]
  const void* sh;                      // bf16 [T, 5120] shared expert, or NULL
  void* out;                           // bf16 [T, 5120] out
  int32_t T;                           // grid (out.size(0))
} FprMoeRawCombineArgs;

const FprExtTable* fpr_moe_raw_table(void);

#ifdef __cplusplus
}
#endif
