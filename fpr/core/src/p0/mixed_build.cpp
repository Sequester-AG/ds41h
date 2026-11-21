// mixed_build.cpp -- see mixed_build.h. Build: -O2 -std=c++17 -ffp-contract=off.
// Python references: Engine.step_mixed
// prepare, _upload_v2, _set_vidx, _attn_pf_plans, _new_scheds
// Hooks.device_tokens / _device_tokens_v2, tail_batch, dec_prefix, _in_layout.
#include "mixed_build.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace fpr {
namespace p0 {

namespace {
int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }   // a >= 0, b > 0
int64_t rnd(const MixedSnap& s, int64_t n) { return std::max(s.mixed_round, cdiv(n, s.mixed_round) * s.mixed_round); }

MixedOp op(MixedOpKind k, char role = 0, int rank = -1) {
  MixedOp o;
  o.kind = k;
  o.role = role;
  o.rank = rank;
  return o;
}
MixedOp bucket_op(int64_t R, char role) {
  MixedOp o = op(MixedOpKind::BUCKET, role);
  o.R = R;
  return o;
}
MixedOp issue_op(MixedOpKind k, char role, const std::vector<int32_t>& layers, bool final_, bool embed) {
  MixedOp o = op(k, role);
  if (!layers.empty()) { o.lay0 = layers.front(); o.lay1 = layers.back(); o.nlay = (int32_t)layers.size(); }
  o.final_ = final_;
  o.embed = embed;
  return o;
}
int64_t dec_prefix_(const StepBatch& sb) {        // engine.dec_prefix
  int64_t nd = 0;
  for (const SeqEntry& se : sb.seqs)
    if (se.kind == "decode") nd += se.n_rows;
  for (const SeqEntry& se : sb.seqs)
    if (se.kind == "decode" && se.row0 + se.n_rows > nd) return -1;
  return nd;
}
// build_inputs (inputs_build.h) for one (bucket, rank): InputsSnap from the step snapshot + prev_of of the table's decode rows
Status upload_rank(const MixedSnap& s, int r, int64_t R, const RankTables& tab, const PrevFn& prev, BucketRank* B,
                   std::string* e) {
  InputsSnap sn;
  sn.R = R;
  sn.scratch = s.scratch[(size_t)r];
  sn.hsd = s.hsd;
  sn.k1 = (int32_t)s.k1;
  sn.rank = r;
  if (prev)
    for (const RankSeqRow& q : tab.seq_rows) {
      if (q.kind != "decode") continue;
      PrevSeq p;
      if (prev(q.rid, &p)) sn.prev[q.rid] = PrevOf{p.rank, (int32_t)p.seq_index};
    }
  B->R = R;
  return build_inputs(sn, tab, &B->up, e);
}
void canonical(std::vector<int64_t>* a, std::vector<int64_t>* b) {   // first appearance, a[i] before b[i]
  std::unordered_map<int64_t, int64_t> ids;
  auto f = [&](int64_t& x) {
    if (x < 0) return;
    auto it = ids.find(x);
    if (it == ids.end()) it = ids.emplace(x, (int64_t)ids.size()).first;
    x = it->second;
  };
  for (size_t i = 0; i < a->size(); ++i) { f((*a)[i]); f((*b)[i]); }
}
}  // namespace

const char* mixed_op_name(MixedOpKind k) {
  switch (k) {
    case MixedOpKind::BUCKET: return "bucket";
    case MixedOpKind::POISON: return "poison";
    case MixedOpKind::CP_PLAN: return "cp_plan";
    case MixedOpKind::UPLOAD: return "upload";
    case MixedOpKind::UPLOAD_GENERAL: return "upload_general";
    case MixedOpKind::PREPARE_ENGRAM: return "prepare_engram";
    case MixedOpKind::SET_VIDX: return "set_vidx";
    case MixedOpKind::IMG_ROWS_A: return "img_rows_A";
    case MixedOpKind::PF_PLANS: return "pf_plans";
    case MixedOpKind::PF_PLAN: return "pf_plan";
    case MixedOpKind::NEW_SCHEDS: return "new_scheds";
    case MixedOpKind::TAIL_BATCH: return "tail_batch";
    case MixedOpKind::SET_STEP: return "set_step";
    case MixedOpKind::ISSUE: return "issue";
    case MixedOpKind::ISSUE_MIXED: return "issue_mixed";
    case MixedOpKind::L20_HEAD: return "l20_head";
    case MixedOpKind::L20_TAIL: return "l20_tail";
    case MixedOpKind::SYNC: return "sync";
    case MixedOpKind::CP_DONE: return "cp_done";
    case MixedOpKind::HEAD_RESULT: return "head_result";
    case MixedOpKind::TB_TAIL: return "tb_tail";
    case MixedOpKind::IMG_ROWS_T: return "img_rows_T";
    case MixedOpKind::JOIN_MIX: return "join_mix";
    case MixedOpKind::T_STAGE: return "t_stage";
    case MixedOpKind::CARRY_CANDIDATES: return "carry_candidates";
    case MixedOpKind::PREPARE_HEAD: return "prepare_head";
    case MixedOpKind::HEAD_EAGER: return "head_eager";
    case MixedOpKind::AFTER_SYNC: return "after_sync";
    case MixedOpKind::MEM_AFTER_STEP: return "mem_after_step";
    case MixedOpKind::LOG: return "log";
  }
  return "?";
}

uint32_t crc32(const void* data, size_t n, uint32_t c) {
  static uint32_t T[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t x = i;
      for (int k = 0; k < 8; ++k) x = (x & 1) ? 0xEDB88320u ^ (x >> 1) : x >> 1;
      T[i] = x;
    }
    init = true;
  }
  const uint8_t* p = (const uint8_t*)data;
  c = ~c;
  for (size_t i = 0; i < n; ++i) c = T[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return ~c;
}

Status tail_batch(const StepBatch& sb, bool tail_bound, int64_t sliding_window, StepBatch* out, std::string* err) {
  StepBatch tb;                                     // StepBatch(sb.rank, sb.step, 0)
  tb.rank = sb.rank;
  tb.step = sb.step;
  tb.n_rows = 0;
  std::unordered_map<int32_t, int32_t> new_of;      // {row: i for i, row in enumerate(sb.tail_rows)}
  new_of.reserve(sb.tail_rows.size() * 2 + 1);
  for (size_t i = 0; i < sb.tail_rows.size(); ++i) new_of[sb.tail_rows[i]] = (int32_t)i;   // a repeated row: the last i wins
  for (const SeqEntry& se : sb.seqs) {
    std::vector<int32_t> rows;                      // [x for x in range(row0, row0 + n_rows) if x in new_of]
    for (int32_t x = se.row0; x < se.row0 + se.n_rows; ++x)
      if (new_of.count(x)) rows.push_back(x);
    if (rows.empty()) continue;
    if ((size_t)rows[0] >= sb.positions.size()) {
      if (err) *err = "tail_batch: row " + std::to_string(rows[0]) + " beyond positions";
      return Status::BAD_ARG;
    }
    SeqEntry ne = se;                               // dataclasses.replace(se, row0=..., n_rows=..., start=...)
    ne.row0 = new_of[rows[0]];
    ne.n_rows = (int32_t)rows.size();
    ne.start = sb.positions[rows[0]];
    if (tail_bound && se.kind != "decode" && !se.score && (int32_t)rows.size() < se.n_rows)
      ne.replay_start = std::max(se.replay_start, se.start + (int64_t)se.n_rows - sliding_window);
    ne.tail_row0 = ne.row0;
    tb.seqs.push_back(std::move(ne));
    const int32_t slot = (int32_t)tb.seqs.size() - 1;
    for (int32_t x : rows) {
      if ((size_t)x >= sb.positions.size() || (size_t)x >= sb.paged_slot.size() || (size_t)x >= sb.window_slot.size()) {
        if (err) *err = "tail_batch: row " + std::to_string(x) + " beyond the row arrays";
        return Status::BAD_ARG;
      }
      tb.positions.push_back(sb.positions[x]);
      tb.seq_slot.push_back(slot);
      tb.paged_slot.push_back(sb.paged_slot[x]);
      tb.window_slot.push_back(sb.window_slot[x]);
      tb.tail_rows.push_back(new_of[x]);
    }
  }
  for (int32_t x : sb.sample_rows) {                // [new_of[x] for x in sb.sample_rows] (KeyError -> BAD_ARG)
    auto it = new_of.find(x);
    if (it == new_of.end()) {
      if (err) *err = "tail_batch: sample row " + std::to_string(x) + " is not a tail row";
      return Status::BAD_ARG;
    }
    tb.sample_rows.push_back(it->second);
  }
  tb.n_rows = (int32_t)sb.tail_rows.size();
  *out = std::move(tb);
  return Status::OK;
}

void build_scheds(const MixedSnap& s, bool per_layer, bool l20, bool pf, std::vector<int64_t>* sched,
                  std::vector<int64_t>* dec) {
  int64_t next = 0;
  const size_t nL = s.layers.size();
  sched->assign(nL, -1);
  dec->assign(nL, -1);
  if (per_layer && s.mixed_plans == "source") {
    std::map<std::array<int64_t, 3>, std::pair<int64_t, int64_t>> keys;
    for (size_t i = 0; i < nL; ++i) {
      const std::array<int64_t, 3> k{s.ratio[i], s.index_source[i], (int64_t)(l20 && s.layers[i] == 20)};
      auto it = keys.find(k);
      if (it == keys.end()) {
        const int64_t a = next++;
        const int64_t b = pf ? next++ : -1;
        it = keys.emplace(k, std::make_pair(a, b)).first;
      }
      (*sched)[i] = it->second.first;
      (*dec)[i] = it->second.second;
    }
  } else {
    int64_t sch[3], schd[3];
    for (int c = 0; c < 3; ++c) sch[c] = next++;
    for (int c = 0; c < 3; ++c) schd[c] = pf ? next++ : -1;
    for (size_t i = 0; i < nL; ++i) {
      const int c = s.ratio[i];
      const bool own = per_layer || (l20 && s.layers[i] == 20);
      (*sched)[i] = own ? next++ : sch[c];
      (*dec)[i] = !pf ? -1 : (per_layer ? next++ : schd[c]);
    }
  }
  canonical(sched, dec);
}

// MM_ON (PHYS_MM=1): a step without image rows runs exactly the text path (engine._img_rows_A / _T: mm_emb from the step's
// images, mm_rt = rows whose HOST token is in IMAGE_ROUTE_IDS); natively refuse only steps that have either. The T tail rows
// are A rows, so A's tokens cover both buckets.
static bool mm_rows(const MixedSnap& s, const std::vector<StepBatch>& b, const std::vector<RankTables>& t) {
  for (const auto& sb : b)
    for (const auto& se : sb.seqs)
      if (!se.images.empty() && se.kind != "decode" && se.start >= 0 && se.n_rows > 0) return true;
  for (size_t r = 0; r < b.size() && r < t.size(); ++r)
    for (int64_t i = 0; i < (int64_t)b[r].n_rows && i < (int64_t)t[r].tokens.size(); ++i) {
      const int64_t x = t[r].tokens[(size_t)i];
      if (x >= s.img_route_lo && x <= s.img_route_hi) return true;
    }
  return false;
}

Status build_mixed(const MixedSnap& s, const std::vector<StepBatch>& b, const std::vector<RankTables>& tables,
                   const PrevFn& prev, ITables* tb_tail, int64_t mixed_steps_before, MixedPlan* out) {
  MixedPlan& P = *out;
  P = MixedPlan();
  P.plan_only = tb_tail == nullptr;     // no T tables / uploads: the structure only (tb_tail's state untouched)
  const int W = s.W;
  auto bad = [&](const std::string& e) {
    P.status = MixedStatus::BAD_ARG;
    P.err = e;
    return Status::BAD_ARG;
  };
  if ((int)b.size() != W || (int)tables.size() != W) return bad("batches / tables per rank");
  for (int r = 0; r < W; ++r) {
    if ((size_t)r >= s.scratch.size()) return bad("snapshot scratch per rank");
    if (s.attn_pf && (size_t)r >= s.n_paged.size()) return bad("snapshot n_paged per rank");
  }
  if (s.ratio.size() != s.layers.size() || s.index_source.size() != s.layers.size()) return bad("snapshot layer maps");
  auto stat = [&](const char* k) { P.stats[k] += 1; };
  auto has = [](const std::vector<int32_t>& v, int32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); };
  // ---- _mixed_body: the plan
  int64_t mx = 0;
  for (int r = 0; r < W; ++r) { P.nA.push_back(b[r].n_rows); mx = std::max<int64_t>(mx, b[r].n_rows); }
  if (mx > s.max_mixed_rows)
    return bad(std::to_string(mx) + " rows per rank > max_mixed_rows " + std::to_string(s.max_mixed_rows));
  P.RA = rnd(s, mx);
  for (int32_t L : s.layers) (L <= 20 ? P.segA : P.segT).push_back(L);
  if (s.poison) {
    P.ops.push_back(bucket_op(P.RA, 'A'));
    for (int r = 0; r < W; ++r) P.ops.push_back(op(MixedOpKind::POISON, 'A', r));
  }
  for (int r = 0; r < W && !P.cp; ++r)
    for (const SeqEntry& se : b[r].seqs)
      if (se.has_cp) { P.cp = s.cp; break; }
  if (P.cp) {
    P.ops.push_back(op(MixedOpKind::CP_PLAN));
    stat("cp_steps");
    if (s.mixed_graphs) return bad("CP prefill steps are eager (mixed_graphs off)");
  }
  // ---- prepare(batches, tables, R_A, moe="prod", hooks=False, role="A")
  P.ops.push_back(bucket_op(P.RA, 'A'));
  P.A.resize(W);
  for (int r = 0; r < W; ++r) {
    if (tables[r].n != b[r].n_rows) return bad("prepare: tab.n != sb.n_rows");
    P.ops.push_back(op(MixedOpKind::UPLOAD, 'A', r));
    std::string e;
    if (!P.plan_only && upload_rank(s, r, P.RA, tables[r], prev, &P.A[r], &e) != Status::OK) return bad("A upload: " + e);
    if (s.idx) P.ops.push_back(op(MixedOpKind::UPLOAD_GENERAL, 'A', r));
  }
  P.ops.push_back(op(MixedOpKind::PREPARE_ENGRAM, 'A'));
  for (int r = 0; r < W; ++r) { P.A[r].cp_on = P.cp ? 1 : 0; P.A[r].chain_input = false; }
  P.ops.push_back(op(MixedOpKind::SET_VIDX, 'A'));
  {
    std::vector<int64_t> ns, nds;
    for (int r = 0; r < W; ++r) { ns.push_back(b[r].n_rows); nds.push_back(dec_prefix_(b[r])); }
    P.vA = build_vidx(ns, &nds, P.RA, s.pf_act8);
    if (P.vA.act8_fallback) stat("act8_fallback_steps");
  }
  P.ops.push_back(op(MixedOpKind::IMG_ROWS_A));
  // ---- _attn_pf_plans
  P.ops.push_back(op(MixedOpKind::PF_PLANS, 'A'));
  for (int r = 0; r < W; ++r) {
    const SeqEntry* f = nullptr;
    const SeqEntry* l = nullptr;
    for (const SeqEntry& se : b[r].seqs)
      if (se.kind != "decode") { if (!f) f = &se; l = &se; }
    BucketRank& A = P.A[r];
    A.has_pf_ab = f != nullptr;
    if (f) { A.pf_a = f->row0; A.pf_b = l->row0 + l->n_rows; }
  }
  if (s.attn_pf) {
    if (s.mixed_graphs) return bad("ENG_ATTN_PF: eager mixed steps only");
    for (int r = 0; r < W; ++r) {
      std::vector<const SeqEntry*> pfs;
      for (const SeqEntry& se : b[r].seqs)
        if (se.kind != "decode") pfs.push_back(&se);
      if (pfs.empty()) continue;
      PfPlanIn& pi = P.A[r].pf_in;
      std::vector<int64_t> nbs;
      int64_t mxb = 0;
      for (const SeqEntry* se : pfs) { nbs.push_back(cdiv(se->start + se->n_rows, 64)); mxb = std::max(mxb, nbs.back()); }
      pi.bt_rows = (int64_t)pfs.size();
      pi.bt_cols = mxb;
      pi.bt.assign((size_t)(pi.bt_rows * mxb), 0);
      for (size_t i = 0; i < pfs.size(); ++i) {
        const auto& btv = pfs[i]->block_table;
        const int64_t have = btv ? (int64_t)btv->size() : 0;
        if (have < nbs[i]) return bad("attn_pf: block_table shorter than the sequence's pages");
        for (int64_t j = 0; j < nbs[i]; ++j) pi.bt[(size_t)(i * mxb + j)] = (int32_t)(*btv)[(size_t)j];
        pi.seqs.push_back({(int64_t)pfs[i]->row0, (int64_t)pfs[i]->n_rows, pfs[i]->start, pfs[i]->replay_start});
      }
      pi.n_paged = s.n_paged[(size_t)r];
      pi.ws = s.pf_ws;
      P.A[r].pf = true;
      P.ops.push_back(op(MixedOpKind::PF_PLAN, 'A', r));
    }
    stat("attn_pf_steps");
  }
  P.ops.push_back(op(MixedOpKind::NEW_SCHEDS, 'A'));
  P.ops.back().per_layer = s.per_layer_plans;
  for (int r = 0; r < W; ++r) build_scheds(s, s.per_layer_plans, false, P.A[r].pf, &P.A[r].sched, &P.A[r].sched_dec);
  P.l20 = s.l20_tail && !s.mixed_graphs && !s.dbg_sync && !P.cp && has(P.segA, 20) && !P.segT.empty() && P.segA.back() == 20;
  // ---- the T bucket (ENG_MIXED_PRESTEP: before A's issue)
  std::vector<StepBatch> tb;
  auto tail_plan = [&]() -> Status {
    tb.resize(W);
    P.nT.clear();
    int64_t mt = 0;
    for (int r = 0; r < W; ++r) {
      P.ops.push_back(op(MixedOpKind::TAIL_BATCH, 0, r));
      std::string e;
      if (tail_batch(b[r], s.tail_bound, s.sliding_window, &tb[r], &e) != Status::OK) return bad(e);
      P.nT.push_back(tb[r].n_rows);
      mt = std::max<int64_t>(mt, tb[r].n_rows);
    }
    P.RT = rnd(s, mt);
    P.tgr = s.mixed_graphs_t && !s.mixed_graphs && !s.dbg_sync && !P.cp && !P.l20;
    if (P.tgr) P.RT = std::max(P.RT, std::min(cdiv(mt, s.mixed_t_round) * s.mixed_t_round, s.max_tail_rows));
    P.ops.push_back(bucket_op(P.RT, 'T'));
    return Status::OK;
  };
  if (s.mixed_prestep && !P.segT.empty() && !s.mixed_graphs) {
    P.prestep_T = true;
    if (tail_plan() != Status::OK) return Status::BAD_ARG;
  }
  if (s.cxx_pieces) P.ops.push_back(op(MixedOpKind::SET_STEP, 'A'));
  if (s.mixed_graphs && !s.dbg_sync) {
    P.ops.push_back(issue_op(MixedOpKind::ISSUE_MIXED, 'A', P.segA, P.segT.empty(), true));
  } else if (P.l20) {
    std::vector<int32_t> s1(P.segA.begin(), P.segA.end() - 1);
    P.ops.push_back(issue_op(MixedOpKind::ISSUE, 'A', s1, false, true));
    P.ops.push_back(op(MixedOpKind::L20_HEAD, 'A'));
    stat("l20_tail_steps");
  } else {
    P.ops.push_back(issue_op(MixedOpKind::ISSUE, 'A', P.segA, P.segT.empty(), true));
  }
  if (P.segT.empty()) {                             // no tail segment: bucket A's head ran in the body (no stats)
    P.ops.push_back(op(MixedOpKind::SYNC));
    if (P.cp) P.ops.push_back(op(MixedOpKind::CP_DONE));
    for (int r = 0; r < W; ++r) P.ops.push_back(op(MixedOpKind::HEAD_RESULT, 'A', r));
    P.native_refusal = "no tail segment";
    return Status::OK;
  }
  if (!P.prestep_T && tail_plan() != Status::OK) return Status::BAD_ARG;
  if (s.poison)
    for (int r = 0; r < W; ++r) P.ops.push_back(op(MixedOpKind::POISON, 'T', r));
  P.T.resize(W);
  const bool c20_exists = has(s.layers, 20);
  for (int r = 0; r < W; ++r) {
    BucketRank& T = P.T[r];
    std::vector<int64_t> rt;                        // [int(tables[r].tokens[i]) for i in batches[r].tail_rows]
    for (int32_t i : b[r].tail_rows) {
      if (i < 0 || i >= tables[r].n) return bad("tail row beyond the A tables");
      rt.push_back(tables[r].tokens[(size_t)i]);
    }
    P.ops.push_back(op(MixedOpKind::TB_TAIL, 0, r));
    T.tb = tb[r];
    std::string e;
    if (!P.plan_only && tb_tail->build(tb[r], &rt, &T.tab, &e) != Status::OK) return bad("tb_tail.build: " + e);
    P.ops.push_back(op(MixedOpKind::UPLOAD, 'T', r));
    if (!P.plan_only && upload_rank(s, r, P.RT, T.tab, prev, &T, &e) != Status::OK) return bad("T upload: " + e);
    P.ops.push_back(op(MixedOpKind::IMG_ROWS_T, 0, r));
    if (s.idx) P.ops.push_back(op(MixedOpKind::UPLOAD_GENERAL, 'T', r));
    P.ops.push_back(op(MixedOpKind::JOIN_MIX, 'A', r));
    P.ops.push_back(op(MixedOpKind::T_STAGE, 'T', r));
    TailStage& g = T.stage;
    const int64_t n = P.nT[(size_t)r];
    g.l20 = P.l20;
    g.c20 = P.l20 ? "si20" : (c20_exists ? "si" : "");
    T.l20 = P.l20 ? 1 : 0;
    T.has_pf_ab = false;
    if (n) {
      g.ix.assign(b[r].tail_rows.begin(), b[r].tail_rows.end());
      if (P.l20) g.keys = {"X", "a_pre", "a_post", "a_comb", "ws.q", "si20", "sl20"};
      else g.keys = {"moe_out", "xm2", "f_post", "f_comb", "f_pre"};
      for (int32_t il : s.index_layers)
        if (il <= 20) { g.keys.push_back("ti." + std::to_string(il)); g.keys.push_back("tl." + std::to_string(il)); }
      g.carry = s.idx && has(P.segA, 20);
      if (g.carry) P.ops.push_back(op(MixedOpKind::CARRY_CANDIDATES, 'T', r));
    }
    if (s.clear_tail_padding)
      for (int32_t il : s.index_layers)
        if (il <= 20) g.clear_layers.push_back(il);
    T.chain_input = true;
  }
  P.ops.push_back(op(MixedOpKind::SET_VIDX, 'T'));
  {
    std::vector<int64_t> ns, nds;
    for (int r = 0; r < W; ++r) { ns.push_back(tb[r].n_rows); nds.push_back(dec_prefix_(tb[r])); }
    P.vT = build_vidx(ns, &nds, P.RT, s.pf_act8);
    if (P.vT.act8_fallback) stat("act8_fallback_steps");
  }
  P.ops.push_back(op(MixedOpKind::PREPARE_HEAD, 'T'));   // (step_mixed: prep_head; nd's native step does it itself)
  P.ops.push_back(op(MixedOpKind::NEW_SCHEDS, 'T'));
  P.ops.back().per_layer = s.per_layer_plans;
  for (int r = 0; r < W; ++r) build_scheds(s, s.per_layer_plans, P.l20, false, &P.T[r].sched, &P.T[r].sched_dec);
  if (s.cxx_pieces) P.ops.push_back(op(MixedOpKind::SET_STEP, 'T'));
  if ((s.mixed_graphs || P.tgr) && !s.dbg_sync) {
    P.ops.push_back(issue_op(MixedOpKind::ISSUE_MIXED, 'T', P.segT, true, false));
    if (P.tgr) stat("mixed_t_graph_steps");
  } else {
    if (P.l20) P.ops.push_back(op(MixedOpKind::L20_TAIL, 'T'));
    P.ops.push_back(issue_op(MixedOpKind::ISSUE, 'T', P.segT, true, false));
  }
  // ---- _mixed_post (head_eager = True: step_mixed's own path)
  P.ops.push_back(op(MixedOpKind::HEAD_EAGER, 'T'));
  P.ops.push_back(op(MixedOpKind::SYNC));
  if (s.mixed_cxx_head) P.ops.push_back(op(MixedOpKind::AFTER_SYNC, 'T'));
  if (P.cp) P.ops.push_back(op(MixedOpKind::CP_DONE));
  if (s.pf_ws && mixed_steps_before % 200 == 199) P.ops.push_back(op(MixedOpKind::LOG));
  P.ops.push_back(op(MixedOpKind::MEM_AFTER_STEP));
  stat("steps");
  stat("mixed_steps");
  for (int r = 0; r < W; ++r) P.ops.push_back(op(MixedOpKind::HEAD_RESULT, 'T', r));
  // ---- native refusal (nd's pre-step): what this builder does not cover yet stays embedded
  if (s.mixed_graphs) P.native_refusal = "mixed_graphs";
  else if (s.dbg_sync) P.native_refusal = "dbg_sync";
  else if (s.poison) P.native_refusal = "poison";
  else if (s.mm_on && mm_rows(s, b, tables)) P.native_refusal = "mm: image rows";
  else if (!s.upload_v2) P.native_refusal = "upload_v1";
  return Status::OK;
}

}  // namespace p0
}  // namespace fpr
