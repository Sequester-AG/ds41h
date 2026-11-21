// moe_decide.h -- the moe piece's per-step variant in C++.
// = fpr/engine/mixed_cxx/mixed_cxx_moe.py decide(drv, s) over an explicit snapshot (torch-free, no Python):
//   MoeSnap    what decide() reads from the engine / driver / flags (fixed after init): mixed_cxx_moe_ensure.snapshot(drv)
//   MoeBucket  what it reads from the bucket state besides vcount / ndec + the view's A8sh layer mask:
//              mixed_cxx_moe_ensure.bucket_of(s) (after ensure_moe; also s["_cxx_moe_export"]["bucket"])
//   moe_decide(snap, bucket, vcount, ndec) -> FprMoeVariant (mixed_plan.h) or -1 (= the Python p_moe; natively: refuse)
// The executor's set_step fills FprStepParams: moe_variant = moe_decide(...), moe_vcount = vcount (p0::build_vidx)
// moe_n_wait = 0 (as mixed_cxx_moe.step_params).
// The per-LAYER refusal of today's _prepare (a fork variant on a layer with the 8-bit shared slice lw["A8sh"] runs Python)
// is moe_bucket_layer_refuses() / moe_bucket_refuses() (the mask rides in MoeBucket); the per-view pre-folded form of the
// same rule is engine/mixed_cxx/pieces_moe_decide.h.
// Golden: engine/mixed_cxx/moe_decide_test.py part 1: Python decide() vs this (ctypes, tests/moe_decide_capi.cpp) on the same
// snapshots over the flag x R x vcount x ndec sweep.
// Both structs are C layout (fixed-width fields) so the exporter can hand them over as bytes: keep in step with the ctypes
// mirrors in mixed_cxx_moe_ensure.py (sizes / offsets checked by the golden).
#pragma once
#include <cstdint>

namespace fpr {
namespace p0 {

constexpr int kMoeMaxDev = 8;
enum MoeVariant : int32_t { MOE_PY = -1, MOE_FORK_V4 = 0, MOE_FORK_RAW = 1, MOE_PROD_RSF = 2, MOE_PROD_FUSED = 3 };

struct MoeSnap {
  int32_t version;                 // kMoeSnapVersion
  int32_t W;                       // len(E.devs)
  int64_t prefill_moe_switch;      // E.cfg.prefill_moe_switch
  int32_t prod_ok;                 // drv._moe_prod_ok (agrs_pmws_c tables + every rank's init + moe_set_nv)
  int32_t fork_ok;                 // drv._moe_fork_ok
  int32_t pm_fused;                // engine.PM_FUSED
  int32_t pf_act8;                 // engine.PF_ACT8
  int64_t small_T;                 // E.prod_moe().SMALL_T (read only when prod_ok && pm_fused; else 0)
  int32_t nt;                      // str(PM_NT) != "0" or agrs_engine.nt_mode(EM) != "0" (AGRS_PMNT env)
  int32_t ws;                      // agrs_ws's mode-2 workspace exists (mixed_cxx_moe._ws() is not None)
  int64_t ws_cap[kMoeMaxDev];      // its rows per device index (fr.cap[d]; 0 = none)
  int32_t agrs_rsf;                // engine.AGRS_RSF
  int32_t cxx_fused;               // ENG_MOE_CXX_FUSED == "1"
  int64_t min_r;                   // agrs_engine.MIN_R (AGRS_MIN_R)
  int32_t moe_v4;                  // E.cfg.moe_v4
  int32_t v4_mixed;                // engine.V4_MIXED or E.cfg.moe_v4_mixed
  int32_t dump_moe;                // bool(engine.DUMP_MOE)
  int32_t pad_;
};
constexpr int32_t kMoeSnapVersion = 1;

struct MoeBucket {
  int32_t prod;                    // s["moe"] == "prod"
  int32_t has_ecs;                 // "ecs" in s (the fork tables)
  int32_t rs_lb128;                // "rs_lb128" in s
  int32_t device;                  // torch.device(s["dev"]).index (-1: none)
  int64_t R;                       // s["R"]
  uint64_t a8sh_layers[2];         // bit L (L < 128): E.lw[rank][L]["A8sh"] is set, over the view's role layers (ensure_moe);
                                   // all ones = unknown (bucket_of without the layer weights / ensure): fork refuses everywhere
};

int32_t moe_decide(const MoeSnap& m, const MoeBucket& b, int64_t vcount, int64_t ndec);
// _prepare: a fork variant on a layer whose lw["A8sh"] is set -> the Python p_moe (natively: refuse the step). NOTE: the
// handle of such a layer is NOT -1 (the same plain handle serves PROD_FUSED there); this rule is the only refusal.
inline bool moe_layer_refuses(int32_t variant, bool layer_a8sh) {
  return layer_a8sh && (variant == MOE_FORK_V4 || variant == MOE_FORK_RAW);
}
inline bool moe_bucket_layer_refuses(const MoeBucket& b, int32_t variant, int32_t layer) {
  const bool a8 = layer < 0 || layer >= 128 || ((b.a8sh_layers[layer >> 6] >> (layer & 63)) & 1u);
  return moe_layer_refuses(variant, a8);
}
// phase 1 for a whole step: any of the view's role layers refuses this variant
inline bool moe_bucket_refuses(const MoeBucket& b, int32_t variant) {
  return (b.a8sh_layers[0] | b.a8sh_layers[1]) != 0 && moe_layer_refuses(variant, true);
}

}  // namespace p0
}  // namespace fpr
