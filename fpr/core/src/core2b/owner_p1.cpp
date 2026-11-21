// owner_p1.cpp -- UPLOAD_GENERAL
// PF_PLAN, CARRY_CANDIDATES, CP_PLAN, CP_DONE. No CUDA here: device work goes through nd (IMixed2b::h2d / cabi), the C ABI
// tables of libs "idx2f_tab" (idx2f_tab_cabi.h) and "p1" (kernels/p1/p1_cabi.h), and the driver's host setters exported by
// lib "mixed_cxx" (pieces_attn.cpp: fpr_attn_set_idxa_step / set_idx_step / set_pf_step / cp_step / cp_done / tr_check).
// The structs those take are p1_cmirror.h's mirrors (asserted against the real headers by tests/p1_mirror_check.cpp).
// Host math = fpr/core/src/p1 (golden-replayed against the Python functions):
//   phase 1a (prepare_plan): libs / setters / handover views exported, the env knobs Python reads, bucket A's tables
//            through gt_host (pure), capacities the plan-only plan already decides
//   phase 1b (prepare):      gt_host A / T, idxa_step / idx_step, pf_host / pf_step (comb_tpl[r], FPCT v1), cp_host / cp_fill;
//                            every size against its device buffer / pinned set (nd's h2d would be FATAL in phase 2)
//   phase 2 (ops, issue only, = Python's order):
//     UPLOAD_GENERAL A r: idxa gst h2d, fpr_attn_set_idxa_step, [rank without a prefill plan] fpr_attn_set_pf_step(NULL);
//                    T r: fpr_attn_set_idx_step; both: idx2f_tab.wait + the staging block into this side's ring slot +
//                         idx2f_tab.run (tables2f.GeneralTables.upload's device half). Idle tables: setters only.
//     PF_PLAN r:     the plan blob + the compact block table h2d, p1.pf_wbt, p1.pf_inv, fpr_attn_set_pf_step.
//     CARRY_CANDIDATES r: p1.carry (ranks[r].tail_ix, staged by T_STAGE just before).
//     CP_PLAN:       per rank the index blob h2d into cp[r].cp_idx, then fpr_attn_cp_step (the step's FprCpMsg array).
//     CP_DONE:       fpr_attn_cp_done (after the step's sync).
// Python's step_params sets the same structs at a segment's start; natively they are set at these ops, which precede the
// segment's ISSUE ops in the plan (A: UPLOAD_GENERAL / PF_PLAN before ISSUE A; T: UPLOAD_GENERAL T after ISSUE A).
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mixed_owner.h"
#include "../p0/inputs_build.h"
#include "../p1/cp_build.h"
#include "../p1/gt_build.h"
#include "../p1/p1_cmirror.h"
#include "../p1/pf_build.h"
#include "../p1/steps_build.h"

namespace fpr {
namespace core2b {
namespace {

namespace cm = p1::cm;
constexpr int W = FPR_EH_MAX_RANKS;
constexpr int64_t kMaxCpMsgsOld = 1024;
// the struct capacities the builders refuse beyond (= the mirrors' arrays)
static_assert(sizeof(cm::IdxAStep::sl[0]) / sizeof(cm::IdxASlice) == 64, "FPR_IDXA_MAX_SLICES = IdxaCfg::max_slices");
static_assert(sizeof(cm::AttnIdxStep::ch) / sizeof(cm::IdxChunk) == 16, "FPR_IDX_MAX_CHUNKS = IdxCfg::max_chunks");

// serve/geometry.py (DeepSeek-V4.1 as served; = tests/p1_record_cp.py's recorded sections). Guarded at phase 1 by the
// handover's pool page sizes (nd_export._pools: G.PAGE_BYTES / G.WINDOW_PAGE_BYTES): any geometry change refuses CP steps.
constexpr int64_t kPageBytes = 116928, kWindowPageBytes = 817344, kWindowLayerPageB = 19008;
constexpr p1::CpSection kCpSections[4] = {{2, 0, 18688, 94464, 4224},
                                          {8, 19008, 18688, 99072, 4224},
                                          {14, 38016, 18688, 103680, 4224},
                                          {20, 57024, 37376, 108288, 8448}};

// the module-level knobs Python reads at import (same parse, same defaults)
struct P1Env {
  int32_t logits = 0;                   // idx2f.LOGITS as hooks_idx._set_logits leaves it: 1 "f8", 2 "cand", 0 other
  bool idx_pf = false;                  // IDX2F_PF == "1" (OwnedIndexer.pf)
  bool mirror = false;                  // FPR_KV_IDX_MIRROR > 0 (lever (a): stays Python)
  bool idxa_stock = false;              // ENG_MIXED_CXX_IDXA_STOCK == "1": rows > LONG_ROW take the idx_a piece's stock
                                        // branch (refuse 2) instead of refusing the step (refuse 1)
  bool carry_asc = true;                // ENG_IDX_CARRY_ASC
                                        // T's cand_ascending = A's layer-20 order ("0" = the old behaviour: always True)
  int64_t ws_cap = 0, ws_total = 0, ws_round = 0;   // attn_prefill WS_CAP / WS_TOTAL (bytes), WS_ROUND (bf16 elements)
  std::string err;
};
std::string env_s(const char* k, const char* def) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string(def);
}
int64_t env_mb_bytes(const char* k, const char* def, std::string* err) {   // int(float(os.environ.get(k, def)) * 2**20)
  const std::string s = env_s(k, def);
  char* end = nullptr;
  const double d = std::strtod(s.c_str(), &end);
  if (s.empty() || end == s.c_str() || *end != 0) { *err = std::string(k) + " not a number"; return 0; }
  return (int64_t)(d * 1048576.0);
}
P1Env read_env() {
  P1Env e;
  const std::string lg = env_s("IDX2F_LOGITS", "cand");
  const std::string eff = (env_s("ENG_IDX_F8", "0") == "1" || lg == "f8") ? "f8" : lg;
  e.logits = eff == "f8" ? 1 : eff == "cand" ? 2 : 0;
  e.idx_pf = env_s("IDX2F_PF", "0") == "1";
  const std::string m = env_s("FPR_KV_IDX_MIRROR", "0");
  e.mirror = !m.empty() && std::atoll(m.c_str()) != 0;
  e.idxa_stock = env_s("ENG_MIXED_CXX_IDXA_STOCK", "0") == "1";
  e.carry_asc = env_s("ENG_IDX_CARRY_ASC", "1") == "1";   // (the same flag and default as hooks_idx.CARRY_ASC)
  e.ws_cap = env_mb_bytes("ATTN2B_PF_WS_MB", "1024", &e.err);
  e.ws_total = env_mb_bytes("ATTN2B_PF_WS_TOTAL_MB", "3072", &e.err);
  e.ws_round = env_mb_bytes("ENG_PF_WS_ROUND_MB", "0", &e.err) / 2;
  return e;
}

struct P1State {
  bool resolved = false;
  P1Env env;
  // resolved once (phase 1a): C ABI functions and the driver's setters
  CabiFn tab_wait = nullptr, tab_run = nullptr, pf_wbt = nullptr, pf_inv = nullptr, carry = nullptr;
  int (*set_idxa)(int, const void*) = nullptr;
  int (*set_idx)(int, const void*) = nullptr;
  int (*set_pf)(int, const void*) = nullptr;
  int (*cp_step)(const void*, int64_t, const void* const*) = nullptr;
  int (*cp_done)() = nullptr;
  int (*tr_check)(const void*, const uint64_t*) = nullptr;
  int64_t cp_max = kMaxCpMsgsOld;               // the driver's FPR_CP_MAX_MSGS (fpr_attn_cp_max_msgs, optional)
  bool caps_done = false;                       // the init-time worst-case bounds below, evaluated once
  std::string caps_why;                         //      "" = every served step fits the step structs' capacities
  // the combine templates per rank (FPCT v1), re-parsed when comb_gen moves (append-only buffers: pointers stay valid)
  uint64_t tpl_gen = ~0ull;
  std::array<std::vector<p1::CombTemplate>, W> tpl;
  // the step's products (phase 1b -> phase 2)
  std::array<p1::GtHost, W> gtA, gtT;
  std::array<p1::IdxaStepHost, W> idxa;
  std::array<cm::IdxAStep, W> idxa_c;
  std::array<cm::AttnIdxStep, W> idx_c;
  std::array<p1::PfHost, W> pf;
  std::array<std::unique_ptr<cm::AttnPfStep>, W> pf_c;   // (41.8 KB each; nullptr = no prefill plan on the rank)
  std::array<std::vector<int32_t>, W> blob;             // seq_lens | gather_lens | qsl (S + 1) | seq_lens // 2, S_max-based
  std::array<std::vector<int32_t>, W> bt;               // the compact block table [S, bt_cols] (Python's Plan.bt layout)
  std::array<int64_t, W> ring_cursor{};                 // this side's idx2f_tab ring cursor (Python keeps its own)
  bool cp = false;
  p1::CpPlanHost cpp;
  std::vector<p1::CpMsgDev> cpm;
  std::array<const void*, W> cp_streams{};
};

bool tensor_ok(const FprEhTensor& t) { return t.data != nullptr && t.ndim > 0; }
bool sets_fit(const FprEhSets& s, int64_t bytes) {    // nd's h2d alternates the two sets: both must hold the upload
  return bytes <= 0 || (s.h[0].ptr && s.h[1].ptr && s.ev[0] && s.ev[1] && s.h[0].bytes >= bytes && s.h[1].bytes >= bytes);
}
std::string rk(const char* what, int r) { return std::string(what) + " (rank " + std::to_string(r) + ")"; }

std::string need_fn(CabiLibs* L, const char* lib, const char* name, int64_t sz, CabiFn* out) {
  std::string why;
  *out = L->fn(lib, name, sz, &why);
  return *out ? "" : why;
}
template <class F>
std::string need_sym(CabiLibs* L, const char* sym, F* out) {
  std::string why;
  *out = reinterpret_cast<F>(L->sym("mixed_cxx", sym, &why));
  return *out ? "" : std::string("mixed_cxx ") + sym + ": " + (why.empty() ? "not exported" : why);
}
std::string resolve(P1State& s, CabiLibs* L) {
  if (L == nullptr) return "no C ABI libraries";
  std::string w;
  if (!(w = need_fn(L, "idx2f_tab", "idx2f_tab.wait", sizeof(cm::TabWaitArgs), &s.tab_wait)).empty()) return w;
  if (!(w = need_fn(L, "idx2f_tab", "idx2f_tab.run", sizeof(cm::TabRunArgs), &s.tab_run)).empty()) return w;
  if (!(w = need_fn(L, "p1", "p1.pf_wbt", sizeof(cm::PfWbtArgs), &s.pf_wbt)).empty()) return w;
  if (!(w = need_fn(L, "p1", "p1.pf_inv", sizeof(cm::PfInvArgs), &s.pf_inv)).empty()) return w;
  if (!(w = need_fn(L, "p1", "p1.carry", sizeof(cm::CarryArgs), &s.carry)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_set_idxa_step", &s.set_idxa)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_set_idx_step", &s.set_idx)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_set_pf_step", &s.set_pf)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_cp_step", &s.cp_step)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_cp_done", &s.cp_done)).empty()) return w;
  if (!(w = need_sym(L, "fpr_attn_tr_check", &s.tr_check)).empty()) return w;
  {
    std::string why;
    if (auto f = reinterpret_cast<int (*)()>(L->sym("mixed_cxx", "fpr_attn_cp_max_msgs", &why))) s.cp_max = f();
  }
  s.env = read_env();
  if (!s.env.err.empty()) return s.env.err;
  s.resolved = true;
  return "";
}

p1::GtSnap gt_snap(const P1State& s, const FprEhMixedBucketRank& v, const FprEhRankState& rs) {
  p1::GtSnap g;
  g.Bc = v.idx.Bc, g.mirror = s.env.mirror, g.has_deltas = true, g.ring_cap = rs.idx_ring_cap;
  return g;
}
p1::PfConsts pf_consts(const P1State& s, const MixedOwnerCtx& x) {
  p1::PfConsts k;
  k.win = x.snap->sliding_window, k.wpt = x.snap->window_page_tokens, k.page_tokens = x.h->page_tokens;
  k.ws_cap = s.env.ws_cap, k.ws_total = s.env.ws_total, k.ws_round = s.env.ws_round;
  return k;
}

// the views a role / rank needs (the engine exports them at ensure; zero = not exported = refuse)
std::string check_views(const MixedOwnerCtx& x, char role, int r, bool pf) {
  const FprEhMixedBucketRank* v = x.view(role, r);
  if (v == nullptr) return rk(role == 'A' ? "no A view" : "no T view", r);
  const FprEhRankState& rs = x.h->ranks[r];
  const FprEhMixedIdx& d = v->idx;
  if (d.Bc <= 0 || d.tab_hid == 0 || !tensor_ok(d.cand) || !tensor_ok(d.ir2row_c) || !tensor_ok(d.valid_ir) ||
      !tensor_ok(d.row2ir_d))
    return rk("indexer views not exported", r);
  if (rs.idx_ring_n <= 0 || rs.idx_ring_n > FPR_EH_IDX_RING_MAX || rs.idx_ring_cap <= 0 || !tensor_ok(rs.ews_f32) ||
      !tensor_ok(rs.ews_bmax) || !tensor_ok(rs.ews_prep))
    return rk("idx2f_tab ring / EagerWorkspace not exported", r);
  for (int k = 0; k < rs.idx_ring_n; ++k)
    if (rs.idx_ring_host[k] == nullptr || rs.idx_ring_dev[k] == nullptr) return rk("idx2f_tab ring slot not exported", r);
  if (role == 'A' && !tensor_ok(v->idxa_gst)) return rk("idxa_gst not exported", r);
  if (pf) {
    const FprEhMixedPf& p = v->pf;
    if (p.blob.ptr == nullptr || p.blob.bytes < 4 * (4 * p.S_max + 1) || !tensor_ok(p.bt) || !tensor_ok(p.wbt) ||
        !tensor_ok(p.loc) || !tensor_ok(p.comb640) || !tensor_ok(p.clen640) || !tensor_ok(p.comb128) ||
        !tensor_ok(p.clen128) || !tensor_ok(rs.pf_inv) || !tensor_ok(rs.pf_ws) || v->in_d.ptr == nullptr)
      return rk("attn_pf views / pf_ws not exported", r);
    if (p.bt.shape[0] * p.bt.stride[0] < p.S_max * p.bt_cols_max) return rk("attn_pf bt view smaller than S_max x bt_cols_max", r);
  }
  return "";
}

void reparse_templates(P1State& s, const FprEngineHandover* h) {
  const uint64_t g = h->comb_gen ? __atomic_load_n(h->comb_gen, __ATOMIC_ACQUIRE) : 0;
  if (g == s.tpl_gen) return;
  for (int r = 0; r < h->n_ranks && r < W; ++r)
    if (h->comb_tpl[r] == nullptr ||
        p1::parse_comb_templates(h->comb_tpl[r], h->comb_tpl_bytes[r], sizeof(cm::TrRecordBytes), &s.tpl[r]) != Status::OK)
      s.tpl[r].clear();                          // (no templates: pf_step refuses every prefill rank)
  s.tpl_gen = g;
}

std::string cp_views(const MixedOwnerCtx& x, int r, p1::CpRankViews* w) {
  const FprEhRankState& rs = x.h->ranks[r];
  const FprEhCpRank& cp = x.h->cp[r];
  if (!tensor_ok(cp.cp_idx) || cp.cap <= 0 || cp.cp_mail_stream == nullptr) return rk("CP views not exported", r);
  *w = p1::CpRankViews();
  w->device = rs.device, w->idx_dev = static_cast<const int64_t*>(cp.cp_idx.data), w->idx_cap = cp.cap;
  w->mail_stream = cp.cp_mail_stream;
  for (int i = 0; i < rs.n_pools && i < FPR_EH_MAX_POOLS; ++i) {
    const auto& pl = rs.pools[i];
    if (std::strncmp(pl.name, "paged", sizeof(pl.name)) == 0) {
      if (pl.page_bytes != kPageBytes) return rk("CP: paged page bytes != geometry", r);
      w->paged = pl.base, w->paged_rows = pl.n_pages, w->paged_ld = pl.page_bytes;          // cp.py _pg(r)
    } else if (std::strncmp(pl.name, "window", sizeof(pl.name)) == 0) {
      if (pl.page_bytes != kWindowPageBytes) return rk("CP: window page bytes != geometry", r);
      w->window = pl.base, w->window_rows = pl.n_pages, w->window_ld = pl.page_bytes;       // _wsec(r): n_window + 1
    }
  }
  if (w->paged == nullptr || w->window == nullptr) return rk("CP: paged / window pool not exported", r);
  return "";
}

// init-time worst cases of the step structs' fixed capacities, from the served configuration (rows cap, tail rows
// cap, the exporter's S_max, max_model_len, the EagerWorkspace budget): the T idx piece's chunks (FPR_IDX_MAX_CHUNKS 16) and
// the idx_a slices per ratio set (FPR_IDXA_MAX_SLICES 64). "" = no served step can exceed them; else native mixed steps stay
// off for this engine with that reason (one fixed refusal class, never a per-step surprise). CP messages are checked per
// step against the driver's own FPR_CP_MAX_MSGS.
std::string caps_check(const MixedOwnerCtx& x) {
  const int64_t budget = x.h->ranks[0].ews_f32.shape[0] * 4;
  const int64_t smax = std::max<int64_t>(x.view('A', 0)->pf.S_max, 1);
  const int64_t ra = std::max<int64_t>(x.snap->max_mixed_rows, 1), rt = std::max<int64_t>(x.snap->max_tail_rows, ra);
  const int64_t ba = (ra + 3) / 4 + smax, bt = (rt + 3) / 4 + smax;   // pseudo-requests: <= rows / 4 + sequences
  const int64_t cb = std::min<int64_t>({512, 1024, std::max<int64_t>(1, budget / (4 * 2048 * 8 * 4))});
  if ((bt + cb - 1) / cb > 16)
    return "p1 caps: T idx chunks could exceed 16 (tail rows " + std::to_string(rt) + ", budget " + std::to_string(budget) + ")";
  const int64_t mml = std::max<int64_t>(x.h->max_model_len, 1);
  for (int64_t ratio : {2, 1}) {                // the ratio-2 set (layers 2 / 8 / 14) and the ratio-1 set (layer 20)
    const int64_t width = ((mml / ratio) + 255) / 256 * 256;
    const int64_t bstep = std::min<int64_t>(std::max<int64_t>(1, budget / (4 * width * 4)), 1024);
    if ((ba + bstep - 1) / bstep > 64)
      return "p1 caps: idx_a slices could exceed 64 (rows " + std::to_string(ra) + ", max_model_len " + std::to_string(mml) + ")";
  }
  return "";
}

// the same bounds
// for THIS step, still in phase 1a (a refusal moves nothing): per rank, pseudo-requests <= rows / 4 + sequences and the
// index width from the step's largest position instead of max_model_len. "" = this step fits (phase 1b's idx_step /
// idxa_step keep their exact checks for the rare step the bound passes but the real slicing would not)
std::string caps_step(const MixedOwnerCtx& x, const std::vector<StepBatch>& rb) {
  const int64_t budget = x.h->ranks[0].ews_f32.shape[0] * 4;
  for (size_t r = 0; r < rb.size(); ++r) {
    int64_t maxpos = 0;
    for (int64_t p : rb[r].positions) maxpos = std::max(maxpos, p);
    const int64_t ba = ((int64_t)rb[r].n_rows + 3) / 4 + (int64_t)rb[r].seqs.size();
    const int64_t cb = std::min<int64_t>({512, 1024, std::max<int64_t>(1, budget / (4 * 2048 * 8 * 4))});
    if ((ba + cb - 1) / cb > 16)
      return "p1 caps: T idx chunks exceed 16 (rank " + std::to_string(r) + ", rows " + std::to_string(rb[r].n_rows) + ")";
    for (int64_t ratio : {2, 1}) {
      const int64_t width = (((maxpos + 1) / ratio) + 255) / 256 * 256;
      const int64_t bstep = std::min<int64_t>(std::max<int64_t>(1, budget / (4 * std::max<int64_t>(width, 256) * 4)), 1024);
      if ((ba + bstep - 1) / bstep > 64)
        return "p1 caps: idx_a slices exceed 64 (rank " + std::to_string(r) + ", rows " + std::to_string(rb[r].n_rows) +
               ", max position " + std::to_string(maxpos) + ")";
    }
  }
  return "";
}

void reg(MixedOwnerRegistry& R) {
  auto st = std::make_shared<P1State>();

  // ---------------------------------------------------------------- phase 1a
  R.prepare_plan("p1", [st](const MixedOwnerCtx& x, const p0::MixedPlan& P) -> std::string {
    P1State& s = *st;
    if (!s.resolved) {
      const std::string w = resolve(s, x.libs);
      if (!w.empty()) return w;
    }
    const int nr = (int)P.A.size();          // (the plan's ranks: snap.W)
    if (nr <= 0 || nr > W || nr > x.h->n_ranks)
      return "ranks: plan " + std::to_string(nr) + ", handover n_ranks " + std::to_string(x.h->n_ranks);
    bool any_ug = false, any_ugT = false;
    for (const p0::MixedOp& o : P.ops)
      if (o.kind == p0::MixedOpKind::UPLOAD_GENERAL) any_ug = true, any_ugT |= o.role == 'T';
    if (!any_ug) return "no UPLOAD_GENERAL ops (indexer off): the attn_pf step clear rides on them";
    {
      if (s.env.mirror) return "FPR_KV_IDX_MIRROR (the mirror map stays Python)";
      if (s.env.logits == 0) return "idx2f.LOGITS neither f8 nor cand";
      if (any_ugT && s.env.logits != 1) return "idx2f.LOGITS not f8 (the T idx piece)";
    }
    const auto& rb = x.mx->resolved(x.c);
    const auto& tabs = x.mx->tables(x.c);
    if (!s.caps_done && x.view('A', 0) != nullptr && tensor_ok(x.h->ranks[0].ews_f32)) {
      s.caps_why = caps_check(x);
      s.caps_done = true;
    }
    if (tensor_ok(x.h->ranks[0].ews_f32)) {
      const std::string w = caps_step(x, rb);
      if (!w.empty()) return w;
    }
    for (int r = 0; r < nr && r < (int)P.A.size(); ++r) {
      const bool pf = P.A[r].pf;
      std::string w = check_views(x, 'A', r, pf);
      if (!w.empty()) return w;
      if (pf && !P.A[r].pf_in.ws) return rk("pf without a persistent pf_ws", r);
      if (pf) {
        const FprEhMixedPf& q = x.view('A', r)->pf;
        if ((int64_t)P.A[r].pf_in.seqs.size() > q.S_max) return rk("attn_pf S over S_max", r);
        if (P.RA > q.loc.shape[0] || P.RA > q.comb640.shape[0] || P.RA > q.comb128.shape[0] || P.RA > q.clen640.shape[0] ||
            P.RA > q.clen128.shape[0])
          return rk("attn_pf rows over R_A", r);
        if (P.A[r].pf_in.bt_cols > q.bt_cols_max) return rk("attn_pf block table over bt_cols_max", r);
      }
      // bucket A's tables are phase-1 inputs: gt_host (pure) refuses here (no deltas / host path / capacity / ring)
      if (r < (int)tabs.size() && r < (int)rb.size()) {
        p1::GtHost g;
        if (p1::gt_host(gt_snap(s, *x.view('A', r), x.h->ranks[r]), 'A', r, P.RA, rb[r], tabs[r], &g) != Status::OK)
          return rk("bucket A indexer tables: host path / capacity", r);
      }
    }
    for (int r = 0; r < nr && r < (int)P.T.size(); ++r) {
      std::string w = check_views(x, 'T', r, false);
      if (!w.empty()) return w;
    }
    for (const p0::MixedOp& o : P.ops)
      if (o.kind == p0::MixedOpKind::CARRY_CANDIDATES && !tensor_ok(x.h->ranks[o.rank].tail_ix)) return rk("tail_ix not exported", o.rank);
    if (P.cp) {
      for (int r = 0; r < nr; ++r) {
        p1::CpRankViews w;
        const std::string e = cp_views(x, r, &w);
        if (!e.empty()) return e;
        if (x.h->ranks[r].step_stream == nullptr) return rk("no step stream", r);
      }
    }
    return "";
  });

  // ---------------------------------------------------------------- phase 1b
  R.prepare("p1", [st](const MixedOwnerCtx& x, const p0::MixedPlan& P) -> std::string {
    P1State& s = *st;
    const int nr = (int)P.A.size();
    const auto& rb = x.mx->resolved(x.c);
    const auto& tabs = x.mx->tables(x.c);
    if ((int)rb.size() < nr || (int)tabs.size() < nr) return "resolved batches / tables < n_ranks";
    reparse_templates(s, x.h);
    std::vector<int32_t> ratios(x.snap->ratio.begin(), x.snap->ratio.end());   // sorted({c["ratio"] for c in caches})
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());
    std::vector<int32_t> segA;                    // the A segment's index layers (KV sources 2 / 8 / 14 / 20)
    for (int32_t L : x.snap->layers)
      if (L <= 20) segA.push_back(L);
    for (int r = 0; r < W; ++r) s.pf_c[r].reset();
    int64_t grows = 0;                            // hooks_idx.upload_general's global rows: every rank's batch rows
    for (int r = 0; r < nr; ++r) grows += rb[r].n_rows;
    for (int r = 0; r < nr && r < (int)P.A.size(); ++r) {
      const FprEhMixedBucketRank& v = *x.view('A', r);
      const FprEhRankState& rs = x.h->ranks[r];
      if (p1::gt_host(gt_snap(s, v, rs), 'A', r, P.RA, rb[r], tabs[r], &s.gtA[r]) != Status::OK) return rk("gt_host A", r);
      // ---- FprIdxAStep (mixed_cxx_attn._idxa_step)
      p1::IdxaCfg ic;
      ic.budget = rs.ews_f32.shape[0] * 4, ic.logits = s.env.logits, ic.pf = s.env.idx_pf;
      ic.stock = s.env.idxa_stock, ic.global_rows = grows;
      if (s.gtA[r].status == p1::GtStatus::IDLE) {
        s.idxa[r] = p1::IdxaStepHost();
        s.idxa[r].idle = 1;
      } else if (p1::idxa_step(s.gtA[r], ic, &s.idxa[r]) != Status::OK) {
        return rk("idx_a: more slices than FprIdxAStep holds", r);
      }
      const p1::IdxaStepHost& h = s.idxa[r];
      for (int i = 0; i < 4; ++i)
        if (h.refuse[i] == 1) return rk("idx_a: rows > LONG_ROW (stock branch off: ENG_MIXED_CXX_IDXA_STOCK=0)", r);
      const int64_t gb = (int64_t)(h.gst.size() * sizeof(int32_t));
      if ((int64_t)h.gst.size() > v.idxa_gst.shape[0] || !sets_fit(v.idxa_gst_up, gb)) return rk("idx_a group starts > idxa_gst", r);
      cm::IdxAStep& c = s.idxa_c[r];
      std::memset(&c, 0, sizeof(c));
      c.valid = h.valid, c.idle = h.idle, c.logits = h.logits, c.B = h.B, c.IR = h.IR, c.nv = h.nv, c.nreq = h.nreq;
      c.global_rows = h.global_rows, c.max_seq_len = h.max_seq_len;
      for (int i = 0; i < 4; ++i) c.skip[i] = h.skip[i], c.refuse[i] = h.refuse[i], c.max_len[i] = h.max_len[i], c.width[i] = h.width[i];
      for (int k = 0; k < 2; ++k) {
        c.nsl[k] = (int32_t)h.sl[k].size();
        for (size_t j = 0; j < h.sl[k].size(); ++j) c.sl[k][j] = {h.sl[k][j].b0, h.sl[k][j].b1, h.sl[k][j].gst_off, h.sl[k][j].g};
      }
      c.gst = h.gst.empty() ? nullptr : static_cast<const int32_t*>(v.idxa_gst.data);
      // ---- attn_pf: Plan (pf_host) + FprAttnPfStep (mixed_cxx_attn._pf_step)
      if (!P.A[r].pf) continue;
      p1::PfSnap ps;
      ps.k = pf_consts(s, x);
      ps.ws = true, ps.ws_numel = rs.pf_ws.shape[0], ps.S_max = v.pf.S_max, ps.bt_cols_max = v.pf.bt_cols_max;
      ps.wbt_cols_max = v.pf.wbt_cols_max;
      p1::PfIn in;
      in.seqs = &P.A[r].pf_in.seqs, in.bt_rows = P.A[r].pf_in.bt_rows, in.bt_cols = P.A[r].pf_in.bt_cols;
      in.bt = &P.A[r].pf_in.bt, in.ws = P.A[r].pf_in.ws;
      p1::PfHost& pf = s.pf[r];
      if (p1::pf_host(ps, in, rb[r], &pf) != Status::OK) return rk("pf_host (workspace fallback / sizes)", r);
      if (!pf.has_pf) continue;
      const int64_t Sm = v.pf.S_max, S = pf.S;
      if (S <= 0 || S > Sm || pf.bt_cols > v.pf.bt_cols_max || pf.wmax > v.pf.wbt_cols_max || (int64_t)pf.qsl.size() != S + 1 ||
          (int64_t)pf.bt.size() != S * pf.bt_cols)
        return rk("attn_pf plan > its views", r);
      const int32_t* blob = static_cast<const int32_t*>(v.pf.blob.ptr);
      p1::PfStepViews pv;
      pv.ws = (uint64_t)rs.pf_ws.data, pv.comb[0] = (uint64_t)v.pf.comb640.data, pv.comb[1] = (uint64_t)v.pf.comb128.data;
      pv.clen[0] = (uint64_t)v.pf.clen640.data, pv.clen[1] = (uint64_t)v.pf.clen128.data, pv.loc = (uint64_t)v.pf.loc.data;
      pv.loc_ld = v.pf.loc.stride[0], pv.qsl = (uint64_t)(blob + 2 * Sm), pv.seq_lens = (uint64_t)blob;
      pv.gather_lens = (uint64_t)(blob + Sm);
      p1::PfStepHost ph;
      if (p1::pf_step(pf, ratios, pv, s.tpl[r], s.tr_check, ps.k, &ph) != Status::OK)
        return rk("attn_pf step: an unrecorded combine key class / sizes", r);
      auto c2 = std::make_unique<cm::AttnPfStep>();
      std::memset(c2.get(), 0, sizeof(cm::AttnPfStep));
      cm::AttnPfStep& f = *c2;
      const int64_t ld_bt = std::max<int64_t>(pf.bt_cols, 1);   // compact, as Python's Plan.bt
      f.valid = 1, f.nlayouts = (int32_t)ph.lay.size(), f.nrecs = (int32_t)ph.recs.size();
      f.n = ph.n, f.row_lo = ph.row_lo, f.row_hi = ph.row_hi;
      f.seq_lens = blob, f.gather_lens = blob + Sm, f.bt = v.pf.bt.data, f.wbt = v.pf.wbt.data, f.inv = rs.pf_inv.data;
      f.loc = v.pf.loc.data, f.ld_bt = ld_bt, f.ld_wbt = v.pf.wbt.stride[0], f.n_inv = rs.pf_inv.shape[0];
      f.comb[0] = v.pf.comb640.data, f.clen[0] = v.pf.clen640.data, f.cw[0] = 640;
      f.comb[1] = v.pf.comb128.data, f.clen[1] = v.pf.clen128.data, f.cw[1] = 128;
      for (size_t li = 0; li < ph.lay.size(); ++li) {   // (pf_step: <= 4 layouts, <= 32 groups, <= 16 values, <= 8 records)
        const auto& L = ph.lay[li];
        f.seq_lens_r[li] = L.ratio == 1 ? (const void*)blob : L.ratio == 2 ? (const void*)(blob + 3 * Sm + 1) : nullptr;
        f.lay[li].ratio = L.ratio, f.lay[li].ngroups = (int32_t)L.g.size(), f.lay[li].disjoint = L.disjoint;
        for (size_t gi = 0; gi < L.g.size(); ++gi) {
          const auto& G = L.g[gi];
          auto& E = f.lay[li].g[gi];
          E.a = G.a, E.b = G.b, E.N = G.N, E.M = G.M, E.kv = (void*)G.kv, E.q0 = G.q0, E.q1 = G.q1, E.rec = G.rec, E.gy = G.gy;
          for (size_t i = 0; i < G.vals.size(); ++i) E.vals[i] = G.vals[i];
        }
      }
      for (size_t i = 0; i < ph.recs.size(); ++i) std::memcpy(&f.rec[i], s.tpl[r][(size_t)ph.recs[i]].record, sizeof(cm::TrRecordBytes));
      f.ws_ratio = f.comp_src = -1;               // (the piece's reuse state: the setter resets it)
      // the pinned images PF_PLAN uploads
      s.blob[r].assign((size_t)(4 * Sm + 1), 0);
      for (int64_t k = 0; k < S; ++k) {
        s.blob[r][(size_t)k] = (int32_t)pf.ends[(size_t)k];
        s.blob[r][(size_t)(Sm + k)] = (int32_t)pf.gls[(size_t)k];
        s.blob[r][(size_t)(3 * Sm + 1 + k)] = (int32_t)(pf.ends[(size_t)k] / 2);   // _pf_step's seq_lens // 2 (ratio 2)
      }
      for (int64_t k = 0; k <= S; ++k) s.blob[r][(size_t)(2 * Sm + k)] = (int32_t)pf.qsl[(size_t)k];
      s.bt[r] = pf.bt;
      if (!sets_fit(v.pf.up, (int64_t)(s.blob[r].size() * 4)) || !sets_fit(v.pf.up, (int64_t)(s.bt[r].size() * 4)))
        return rk("attn_pf pinned sets smaller than the plan's uploads", r);
      s.pf_c[r] = std::move(c2);
    }
    for (int r = 0; r < nr && r < (int)P.T.size(); ++r) {
      const FprEhMixedBucketRank& v = *x.view('T', r);
      const FprEhRankState& rs = x.h->ranks[r];
      if (p1::gt_host(gt_snap(s, v, rs), 'T', r, P.RT, P.T[r].tb, P.T[r].tab, &s.gtT[r]) != Status::OK) return rk("gt_host T", r);
      // ---- FprAttnIdxStep (mixed_cxx_attn._idx_step)
      cm::AttnIdxStep& c = s.idx_c[r];
      std::memset(&c, 0, sizeof(c));
      if (s.gtT[r].status == p1::GtStatus::IDLE) {
        c.valid = 1, c.idle = 1;
      } else {
        p1::IdxCfg ic;
        ic.budget = rs.ews_f32.shape[0] * 4;
        // layer 20 ran the stock select in bucket A (rows > LONG_ROW on this rank): its candidate lists are in
        // score order. With ENG_IDX_CARRY_ASC (= hooks_idx.carry_candidates' fix) T takes the order-insensitive prep.
        if (s.env.carry_asc && r < (int)P.A.size() && s.gtA[r].status == p1::GtStatus::OK &&
            s.gtA[r].max_len[3] > p1::IdxaCfg().long_row)
          ic.cand_ascending = false;
        p1::IdxWs ws{(uint64_t)rs.ews_prep.data, (uint64_t)rs.ews_f32.data, (uint64_t)rs.ews_bmax.data, (uint64_t)v.idx.cand.data};
        p1::AttnIdxStepHost ih;
        if (p1::idx_step(s.gtT[r], ic, ws, &ih) != Status::OK) return rk("idx: more chunks than FprAttnIdxStep holds", r);
        c.valid = ih.valid, c.idle = ih.idle, c.use2 = ih.use2, c.single = ih.single, c.nchunks = (int32_t)ih.ch.size();
        c.nwords = ih.nwords, c.B = ih.B, c.IR = ih.IR, c.nv = ih.nv;
        for (int i = 0; i < 8; ++i) c.skip[i] = ih.skip[i];
        for (size_t k = 0; k < ih.ch.size(); ++k) {
          const auto& hk = ih.ch[k];
          c.ch[k] = {hk.b0, hk.b1, (void*)hk.ublk, (void*)hk.uslot, (void*)hk.nunion, (void*)hk.cblk, (void*)hk.clen,
                     (void*)hk.upk, (void*)hk.comp, (void*)hk.bm, hk.ld_cblk};
        }
      }
      // ---- carry_candidates' sizes
      const int64_t nt = (int64_t)P.T[r].stage.ix.size();
      if (P.T[r].stage.carry && nt > 0) {
        if (nt > x.h->ranks[r].tail_ix.shape[0]) return rk("carry: tail rows > tail_ix", r);
        if (4 * s.gtT[r].B > v.idx.cand.shape[0]) return rk("carry: T indexer rows > cand", r);
      }
    }
    // ---- CP (CPPlan.__init__ + cp_step's message order)
    s.cp = false;
    s.cpm.clear();
    if (P.cp) {
      p1::CpGeom g;
      g.sources.assign(std::begin(kCpSections), std::end(kCpSections));
      g.wlb = kWindowLayerPageB, g.win = x.snap->sliding_window, g.wpt = x.snap->window_page_tokens, g.page_tokens = x.h->page_tokens;
      if (p1::cp_host(rb, segA, x.snap->idx, g, &s.cpp) != Status::OK) return "CP plan: a group / block not on its ranks";
      std::vector<p1::CpRankViews> views((size_t)nr);
      for (int r = 0; r < nr; ++r) {
        std::string e = cp_views(x, r, &views[(size_t)r]);
        if (!e.empty()) return e;
        const int64_t bytes = (int64_t)(s.cpp.blob[(size_t)r].size() * sizeof(int64_t));
        if ((int64_t)s.cpp.blob[(size_t)r].size() > x.h->cp[r].cap || !sets_fit(x.h->cp[r].cp_idx_up, bytes))
          return rk("CP index blob > cp_idx", r);
        s.cp_streams[(size_t)r] = x.h->ranks[r].step_stream;
      }
      if (p1::cp_fill(s.cpp, views, &s.cpm) != Status::OK) return "CP messages: a section / offset outside its pool";
      if ((int64_t)s.cpm.size() > s.cp_max) return "CP messages > FPR_CP_MAX_MSGS";
      s.cp = true;
    }
    return "";
  });

  // ---------------------------------------------------------------- phase 2
  R.op(p0::MixedOpKind::UPLOAD_GENERAL, [st](const MixedOwnerCtx& x, const p0::MixedOp& o, const p0::MixedPlan& P,
                                            std::string* err) -> Status {
    P1State& s = *st;
    const int r = o.rank;
    const FprEhMixedBucketRank* v = x.view(o.role, r);
    const FprEhRankState& rs = x.h->ranks[r];
    const p1::GtHost& g = o.role == 'A' ? s.gtA[r] : s.gtT[r];
    if (o.role == 'A') {
      const auto& gst = s.idxa[r].gst;
      if (!gst.empty() && x.mx->h2d(x.c, r, &v->idxa_gst_up, v->idxa_gst.data, gst.data(),
                                    (int64_t)(gst.size() * sizeof(int32_t)), err) != Status::OK)
        return Status::INTERNAL;
      if (s.set_idxa(r, &s.idxa_c[r]) != 0) { *err = "fpr_attn_set_idxa_step"; return Status::INTERNAL; }
      if (!s.pf_c[r] && s.set_pf(r, nullptr) != 0) { *err = "fpr_attn_set_pf_step(NULL)"; return Status::INTERNAL; }
    } else if (s.set_idx(r, &s.idx_c[r]) != 0) {
      *err = "fpr_attn_set_idx_step";
      return Status::INTERNAL;
    }
    if (g.status == p1::GtStatus::IDLE) return Status::OK;   // (GeneralTables.upload's idle entry: no run())
    const int64_t slot = s.ring_cursor[r]++ % rs.idx_ring_n;
    cm::TabWaitArgs w{};
    w.ring_id = rs.idx_ring_id, w.slot = slot;
    if (x.mx->cabi(x.c, r, s.tab_wait, &w, err) != Status::OK) return Status::INTERNAL;   // the slot's last use is done
    if (!g.stage.empty()) std::memcpy(rs.idx_ring_host[slot], g.stage.data(), g.stage.size() * sizeof(int32_t));
    cm::TabRunArgs a{};
    a.hid = v->idx.tab_hid, a.ring_id = rs.idx_ring_id, a.slot = slot, a.host = rs.idx_ring_host[slot];
    a.dev = rs.idx_ring_dev[slot], a.n = g.n, a.S = g.S, a.U = g.U, a.P = g.P, a.B = g.B;
    (void)P;
    return x.mx->cabi(x.c, r, s.tab_run, &a, err);
  });

  R.op(p0::MixedOpKind::PF_PLAN, [st](const MixedOwnerCtx& x, const p0::MixedOp& o, const p0::MixedPlan& P,
                                     std::string* err) -> Status {
    P1State& s = *st;
    const int r = o.rank;
    (void)P;
    if (!s.pf_c[r]) {                             // (pf_host: no prefill rows on this rank)
      if (s.set_pf(r, nullptr) != 0) { *err = "fpr_attn_set_pf_step(NULL)"; return Status::INTERNAL; }
      return Status::OK;
    }
    const FprEhMixedBucketRank* v = x.view('A', r);
    const FprEhRankState& rs = x.h->ranks[r];
    const p1::PfHost& pf = s.pf[r];
    if (x.mx->h2d(x.c, r, &v->pf.up, v->pf.blob.ptr, s.blob[r].data(), (int64_t)(s.blob[r].size() * sizeof(int32_t)), err) !=
            Status::OK ||
        x.mx->h2d(x.c, r, &v->pf.up, v->pf.bt.data, s.bt[r].data(), (int64_t)(s.bt[r].size() * sizeof(int32_t)), err) !=
            Status::OK)
      return Status::INTERNAL;
    // si / sl of bucket A: views of its in_d at p0::in_layout(R) (si int32 [R, SLIDING_WINDOW])
    const p0::InLayout lay = p0::in_layout(v->R);
    const char* in = static_cast<const char*>(v->in_d.ptr);
    const int32_t* blob = static_cast<const int32_t*>(v->pf.blob.ptr);
    cm::PfWbtArgs a{};                             // Plan's window block table (zeroed [S, wmax], then filled)
    a.si = reinterpret_cast<const int32_t*>(in + lay.off[p0::IN_SI]), a.ld_si = p0::kSlidingWindow;
    a.sl = reinterpret_cast<const int32_t*>(in + lay.off[p0::IN_SL]), a.seq = nullptr, a.S = pf.S;
    a.wbt = static_cast<int32_t*>(v->pf.wbt.data), a.ld_wbt = v->pf.wbt.stride[0], a.wmax = pf.wmax;
    a.wpt = x.snap->window_page_tokens, a.seq_lens = blob, a.gather_lens = blob + v->pf.S_max;
    a.qsl = blob + 2 * v->pf.S_max, a.row_lo = pf.row_lo;
    if (x.mx->cabi(x.c, r, s.pf_wbt, &a, err) != Status::OK) return Status::INTERNAL;
    cm::PfInvArgs b{};                             // Plan.inv over the pool's pages + 1
    b.inv = static_cast<int32_t*>(rs.pf_inv.data), b.n_inv = rs.pf_inv.shape[0], b.bt = static_cast<const int32_t*>(v->pf.bt.data);
    b.ld_bt = s.pf_c[r]->ld_bt, b.nb = nullptr, b.S = pf.S, b.max_nb = pf.bt_cols, b.seq_lens = blob;
    if (x.mx->cabi(x.c, r, s.pf_inv, &b, err) != Status::OK) return Status::INTERNAL;
    if (s.set_pf(r, s.pf_c[r].get()) != 0) { *err = "fpr_attn_set_pf_step"; return Status::INTERNAL; }
    return Status::OK;
  });

  R.op(p0::MixedOpKind::CARRY_CANDIDATES, [st](const MixedOwnerCtx& x, const p0::MixedOp& o, const p0::MixedPlan& P,
                                              std::string* err) -> Status {
    P1State& s = *st;
    const int r = o.rank;
    if (s.gtA[r].status == p1::GtStatus::IDLE || s.gtT[r].status == p1::GtStatus::IDLE || r >= (int)P.T.size() ||
        P.T[r].stage.ix.empty())
      return Status::OK;                         // (carry_candidates' early return)
    const FprEhMixedBucketRank* A = x.view('A', r);
    const FprEhMixedBucketRank* T = x.view('T', r);
    cm::CarryArgs a{};
    a.ir2row_T = static_cast<const int64_t*>(T->idx.ir2row_c.data), a.valid_T = static_cast<const uint8_t*>(T->idx.valid_ir.data);
    a.ix = static_cast<const int64_t*>(x.h->ranks[r].tail_ix.data), a.row2ir_A = static_cast<const int64_t*>(A->idx.row2ir_d.data);
    a.cand_A = static_cast<const int32_t*>(A->idx.cand.data), a.cand_T = static_cast<int32_t*>(T->idx.cand.data);
    a.IR_T = 4 * s.gtT[r].B;
    return x.mx->cabi(x.c, r, s.carry, &a, err);
  });

  R.op(p0::MixedOpKind::CP_PLAN, [st](const MixedOwnerCtx& x, const p0::MixedOp&, const p0::MixedPlan& P, std::string* err) -> Status {
    P1State& s = *st;
    if (!s.cp) { *err = "p1: CP_PLAN without a prepared CP plan"; return Status::INTERNAL; }
    for (int r = 0; r < (int)P.A.size(); ++r) {
      const auto& bl = s.cpp.blob[(size_t)r];
      if (!bl.empty() && x.mx->h2d(x.c, r, &x.h->cp[r].cp_idx_up, x.h->cp[r].cp_idx.data, bl.data(),
                                   (int64_t)(bl.size() * sizeof(int64_t)), err) != Status::OK)
        return Status::INTERNAL;
    }
    if (s.cp_step(s.cpm.data(), (int64_t)s.cpm.size(), s.cp_streams.data()) != 0) { *err = "fpr_attn_cp_step"; return Status::INTERNAL; }
    return Status::OK;
  });

  R.op(p0::MixedOpKind::CP_DONE, [st](const MixedOwnerCtx&, const p0::MixedOp&, const p0::MixedPlan&, std::string* err) -> Status {
    P1State& s = *st;
    if (s.cp_done() != 0) { *err = "fpr_attn_cp_done"; return Status::INTERNAL; }
    s.cp = false;
    return Status::OK;
  });
}

const bool registered = add_mixed_owner("p1", &reg);

}  // namespace
}  // namespace core2b
}  // namespace fpr
