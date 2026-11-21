// pf_build.h -- P1 builder: the HOST side of a mixed step's prefill-attention plan. = engine.Engine._attn_pf_plans + kernels/attn_dense/attn_prefill.Plan (ENG_MIXED_PRESTEP: no device
// reads), decision for decision: the rank's prefill entries (kind != "decode", row order; CP helper sub-chunks included),
// pf_ab, the block-table staging [S, max nb] int32, row_lo / row_hi / n, firsts / ends / g0s (max(replay_start,
// first - 127, 0)) / gather lengths / qsl, the window block-table width, the first rows' window lengths k0 (= first - g0
// + 1), and per ratio (0 window-only, 1, 2) Plan.group_plan (PREFILL_CHUNK_SIZE / WS_CAP) and Plan.group_ws's layout
// (areas, disjoint / shared, the views' element offsets, the prealloc fallbacks or the per-step need). The device work
// (the blob upload, the wbt fill from si / sl, the inv scatter, to_local, the gathers) stays with the caller (P1 kernels).
// Golden: core/tests/replay_pf.cpp on tests/p1_record_gt.py traces (pf.jsonl).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "../../include/tables.h"

namespace fpr {
namespace p1 {

struct PfConsts {                       // attn_prefill.py module constants (environment-dependent ones as served)
  int64_t win = 128, wpt = 32, page_tokens = 64;
  int64_t chunk = 4;                    // PREFILL_CHUNK_SIZE
  int64_t ws_cap = 1024LL << 20;        // ATTN2B_PF_WS_MB (bytes per request group)
  int64_t ws_total = 3072LL << 20;      // ATTN2B_PF_WS_TOTAL_MB
  int64_t ws_round = 0;                 // ENG_PF_WS_ROUND_MB in bf16 elements (0 = off)
};

struct PfGroup { int64_t a, b, N, M, area, off; };   // requests [a, b), max compressed / window extents, bf16 elements
struct PfRatio {
  std::vector<PfGroup> groups;
  bool disjoint = true;
  int64_t need = 0;                     // elements the layout needs (sum if disjoint else max)
  bool shared_fallback = false;         // prealloc: disjoint did not fit, the shared layout instead
  bool alloc_fallback = false;          // prealloc: one group larger than the preallocation (Python allocates; counted)
};
struct PfHost {
  bool has_pf = false;                  // any prefill entry on the rank (else s["pf"] = None)
  int32_t pf_a = 0, pf_b = 0;           // s["ws"]["pf_ab"]
  int64_t S = 0, row_lo = 0, row_hi = 0, n = 0;
  std::vector<int64_t> firsts, ends, g0s, gls, qsl, k0;
  std::vector<int32_t> bt;              // [S, bt_cols] (zero padded)
  int64_t bt_cols = 0;
  int64_t wmax = 0;                     // wbt columns: max(ends) // 32 + 1
  std::array<PfRatio, 3> ratio;         // by ratio 0 / 1 / 2
};

// sb: the rank's StepBatch (seqs with block tables); ws_numel: the engine's pf_ws[r] elements (ENG_PF_WS_PREALLOC_MB;
// 0 = none: the per-step allocation of the need). BAD_ARG where Plan asserts (non-contiguous prefill rows, empty entry).
Status pf_host(const StepBatch& sb, int64_t ws_numel, const PfConsts& k, PfHost* out);

// ---- nd's native mixed step, phase 1: from fpr::p0::build_mixed's PfPlanIn of the rank (the
// tuples + bt blob its golden checks) and the rank's StepBatch (pf_ab). BAD_ARG (the step runs embedded, decided before
// any device work): a PfPlanIn that does not match the batch's prefill entries, Plan's asserts, and -- a native step
// allocates nothing -- prefill rows without a persistent workspace (!snap.ws: ENG_PF_WS_PREALLOC_MB=0, Python's per-step
// workspace) or a layout that does not fit it (Python's alloc fallback). Pure (no host state).
struct PfSnap {
  PfConsts k;
  bool ws = false;                      // the engine's pf_ws[r] exists (the handover's FprEhRankState.pf_ws.data != NULL)
  int64_t ws_numel = 0;                 // its bf16 elements
  int64_t S_max = 0, bt_cols_max = 0, wbt_cols_max = 0;   // FprEhMixedPf's sizes (0 = unchecked)
};
struct PfIn {                           // = fpr::p0::PfPlanIn's fields (kept free of p0's header)
  const std::vector<std::array<int64_t, 4>>* seqs = nullptr;   // (row0, n_rows, start, replay_start)
  int64_t bt_rows = 0, bt_cols = 0;
  const std::vector<int32_t>* bt = nullptr;
  bool ws = false;
};
Status pf_host(const PfSnap& snap, const PfIn& in, const StepBatch& sb, PfHost* out);

}  // namespace p1
}  // namespace fpr
