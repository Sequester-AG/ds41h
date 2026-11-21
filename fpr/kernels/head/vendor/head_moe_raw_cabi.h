// head_moe_raw_cabi.h -- C ABI of kernels/head/vendor/moe_raw.cu, the HSD drafter's head_moe_raw
// per fpr/engine/mixed_cxx/fpr_cabi.h. vendor/moe_raw.cu is moe_stream/moe_raw.cu before pdl2 / router_top_pack3 (plain
// launches, no ENG_PDL2): swiglu / gemm1r / gemm2r / combine take moe_stream/moe_raw_cabi.h's structs (same signatures and
// launch arguments; swiglu / combine launch without the PDL attribute here), init takes FprMoeRawInitArgs; the drafter's
// other launches (drafter.py moe(): route for moe_mode "base", prepr, gemm1r8 / gemm2r8 for "n8") have their own below.
// Raw layout: w13 u8 [E, 1152, 2560], s13 [E, 1152, 160], w2 [E, 5120, 288], s2 [E, 5120, 18]. Buffers as moe_raw's
// route: route_max_ec() = 480 EC records (800 B) / 9x / 40x tiles (24 B), counts int32 >= 6 (>= 7 for gemm*r8: counts[6]
// = n ec8 from route2x), slotmap int32 [6T], rts int32 scratch >= route_scratch_ints() (ws_bytes of "head_moe_raw.route").
// route_max_ec / route_scratch_ints are host constants (load-time sizing, pybind only).
// Table: fpr_head_moe_raw_table(); "head_moe_raw.init" (per device: the k_route / k_gemm<1> / k_gemm<2> / k_gemm1r8 /
// k_gemm2r8 dynamic-SMEM opt-ins, atomic per-device masks; once per process, before the driver threads: the SM count and the
// cuTensorMapEncodeTiled entry point), "head_moe_raw.route", "head_moe_raw.swiglu", "head_moe_raw.prepr",
// "head_moe_raw.gemm1r", "head_moe_raw.gemm1r8", "head_moe_raw.gemm2r", "head_moe_raw.gemm2r8", "head_moe_raw.combine".
// gemm1r / gemm1r8 / gemm2r encode their TMA tensor maps on the host per call (pure host work, capture-safe).
// Build variants (MS3_* / MSR_*) are compiled in.
#pragma once
#include "../../moe_stream/moe_raw_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind route(ids, w, T, ecs, t1, t2, counts, slotmap, rts, grid, inv)
  FprCall c;
  const int32_t* ids;                  // [T, 6] contiguous
  const float* w;                      // [T, 6] contiguous
  int64_t T;                           // 6 T <= 6,144 (else FPR_E_ARG)
  void* ecs;                           // EC records out
  void* t1;                            // GEMM1 tiles out
  void* t2;                            // GEMM2 tiles out
  int32_t* counts;                     // {n_ec, ntiles1, ntiles2, ...}
  int32_t* slotmap;                    // [6T]
  int32_t* rts;                        // scratch (ws)
  int64_t rts_bytes;                   // >= route_scratch_ints() x 4 (else FPR_E_WS)
  int32_t grid;                        // snake grid (0 = SM count)
  const float* inv;                    // [T] or NULL
} FprHeadMoeRawRouteArgs;

typedef struct {                       // = pybind prepr(x, ecs, counts, max_ec, img, ptok, inv)
  FprCall c;
  const void* x;                       // bf16 [T, 5120]
  void* ecs;                           // route's EC records (ec.inv written)
  const int32_t* counts;               // route's counts (counts[0] = n_ec, read on the device)
  int32_t max_ec;                      // grid.x (route_max_ec())
  void* img;                           // token image out
  int32_t* ptok;                       // [T] prescale exponents out
  float* inv;                          // [T] out
  int32_t T;                           // x.size(0) (k_prescale grid)
} FprHeadMoeRawPreprArgs;

typedef struct {                       // = pybind gemm1r8(w13, s13, ecs, ec8, counts, img, himg, grid)
  FprCall c;
  const void* w13;                     // raw u8 [E, 1152, 2560]
  const void* s13;                     // raw ue8m0 [E, 1152, 160]
  int64_t E;                           // w13.size(0)
  const void* ecs;
  const int32_t* ec8;                  // route2x's n_pad-8 EC indices
  const int32_t* counts;               // >= 7 (counts[6] = n ec8)
  const void* img;                     // token image (prepr)
  void* himg;                          // H image out
  int32_t grid;                        // CTAs (0 = SM count)
} FprHeadMoeRawGemm1r8Args;

typedef struct {                       // = pybind gemm2r8(w2, s2, ecs, ec8, counts, himg, yout, grid)
  FprCall c;
  const void* w2;                      // raw u8 [E, 5120, 288]
  const void* s2;                      // raw ue8m0 [E, 5120, 18]
  const void* ecs;
  const int32_t* ec8;
  const int32_t* counts;               // >= 7
  const void* himg;                    // H image (gemm1r / gemm1r8)
  void* yout;                          // bf16 [slots, 5120] out
  int32_t grid;                        // CTAs (0 = SM count)
} FprHeadMoeRawGemm2r8Args;

const FprExtTable* fpr_head_moe_raw_table(void);

#ifdef __cplusplus
}
#endif
