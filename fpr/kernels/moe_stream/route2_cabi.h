// route2_cabi.h -- C ABI of kernels/moe_stream/route2.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The routing tables of the mixed step's MoE (Engine.p_moe, T = W R gathered rows): T <= route2_one_max_t (1365): route2r
// (one kernel, ids / weights read in place from the 64-B routing rows); larger T: route2 (k_rank + k_fill on contiguous ids
// / weights). Caller-owned buffers, max_ec(T) = 384 + ceil(6 T / 64): ecs >= max_ec(T) EC records (800 B, 16-B aligned),
// t1 >= 9 max_ec(T) and t2 >= 40 max_ec(T) tiles (24 B, 8-B aligned), counts int32 >= 6 (>= 7 with ec8), slotmap int32
// >= 6T; route2's int32 scratch rts (16-B aligned) >= route2_ws_bytes (= route2_scratch_ints() x 4).
// Table: fpr_route2_table(); "route2.init" (per device: the k_fill / k_one dynamic-SMEM opt-ins; the SM count),
// "route2.route2", "route2.route2r". The tuning globals of route2_config (pdl, k_fill CTAs per SM, ipt, one_max_t; the
// engine never calls it) are init-time settings: set them before the first launch, never while the driver runs.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { FprCall c; } FprRoute2InitArgs;   // on c.device (current): attributes + SM count

typedef struct {                       // = pybind route2(ids, w, T, ecs, t1, t2, counts, slotmap, rts, grid, inv) / route2x(.., ec8)
  FprCall c;
  const int32_t* ids;                  // [T, 6] contiguous
  const float* w;                      // [T, 6] contiguous
  int64_t T;                           // 0 .. 8192 (else FPR_E_ARG)
  void* ecs;                           // EC records out
  void* t1;                            // GEMM1 tiles out
  void* t2;                            // GEMM2 tiles out
  int32_t* counts;                     // {n_ec, ntiles1, ntiles2, img bytes, h bytes, P (, n ec8)}
  int32_t* slotmap;                    // [6T]
  int32_t* rts;                        // scratch (ws)
  int64_t rts_bytes;                   // >= route2_ws_bytes (else FPR_E_WS)
  int32_t grid;                        // snake grid (0 = SM count)
  const float* inv;                    // [T] or NULL
  int32_t* ec8;                        // route2x: >= max_ec(T) EC indices with n_pad 8 (needs counts >= 7); NULL = route2
} FprRoute2Args;

typedef struct {                       // = pybind route2r(rows, T, ecs, t1, t2, counts, slotmap, rts, grid) (rts: unread)
  FprCall c;
  const void* rows;                    // [T, 64 B] routing rows (int32 ids at bytes 0..23, fp32 weights at 24..47)
  int64_t T;                           // 1 .. route2_one_max_t (else FPR_E_SHAPE: take route2)
  void* ecs;
  void* t1;
  void* t2;
  int32_t* counts;                     // >= 6
  int32_t* slotmap;                    // [6T]
  int32_t grid;                        // snake grid (0 = SM count)
} FprRoute2rArgs;

const FprExtTable* fpr_route2_table(void);

#ifdef __cplusplus
}
#endif
