// steps_build.cpp -- see steps_build.h.
#include "steps_build.h"

#include <algorithm>
#include <cstring>
#include <set>

namespace fpr {
namespace p1 {

Status idxa_step(const GtHost& gt, const IdxaCfg& cfg, IdxaStepHost* out) {
  IdxaStepHost& v = *out;
  v = IdxaStepHost();
  v.logits = cfg.logits;
  v.global_rows = (int32_t)cfg.global_rows, v.max_seq_len = gt.max_seq_len;   // (_idxa_values passes them through)
  if (gt.status == GtStatus::IDLE) {
    v.idle = 1;
    return Status::OK;
  }
  if (gt.status != GtStatus::OK) return Status::BAD_ARG;
  v.B = gt.B, v.IR = gt.IR, v.nv = gt.n, v.nreq = std::max<int64_t>(gt.S, 1);   // qsl[: max(S, 1) + 1].numel() - 1
  for (int i = 0; i < 4; ++i) {                 // IDXA_SOURCES = the first four index layers (2, 8, 14, 20)
    v.max_len[i] = gt.max_len[i], v.width[i] = gt.width[i];
    v.skip[i] = gt.max_len[i] <= 0;
    v.refuse[i] = gt.max_len[i] > cfg.long_row ? (cfg.stock ? 2 : 1) : 0;
  }
  const int src_of[2] = {0, 3};                 // ratio-2 set: layer 2's width; ratio-1 set: layer 20's
  for (int k = 0; k < 2; ++k) {
    const int64_t per_b = cfg.next_n * v.width[src_of[k]] * 4;
    const int64_t bstep = std::min<int64_t>(std::max<int64_t>(1, cfg.budget / per_b), cfg.max_call_b);
    for (int64_t b0 = 0; b0 < v.B; b0 += bstep) {
      const int64_t b1 = std::min<int64_t>(v.B, b0 + bstep);
      int64_t off = -1, g = 0;
      if (cfg.pf) {                             // OwnedIndexer._pf_gstart's rule
        std::set<int64_t> gs = {b0, b1};
        for (int64_t x : gt.pf_groups)
          if (b0 < x && x < b1) gs.insert(x);
        if (2 * ((int64_t)gs.size() - 1) <= b1 - b0) {
          off = (int64_t)v.gst.size(), g = (int64_t)gs.size() - 1;
          for (int64_t x : gs) v.gst.push_back((int32_t)(x - b0));
        }
      }
      v.sl[k].push_back({b0, b1, off, g});
    }
    if ((int64_t)v.sl[k].size() > cfg.max_slices) return Status::BAD_ARG;
  }
  return Status::OK;
}

Status idx_step(const GtHost& gtT, const IdxCfg& cfg, const IdxWs& ws, AttnIdxStepHost* out) {
  constexpr int64_t NEXT_N = 4, CAND_K = 2048, CAND_BLOCK = 8, MAX_CALL_B = 1024;
  AttnIdxStepHost& st = *out;
  st = AttnIdxStepHost();
  if (gtT.status == GtStatus::IDLE) {
    st.idle = 1;
    return Status::OK;
  }
  if (gtT.status != GtStatus::OK) return Status::BAD_ARG;
  for (int i = 0; i < 8; ++i) st.skip[i] = gtT.max_len[i] <= 0;
  const int64_t ml = gtT.max_len[4];            // layer 24 (= 28 / 32 / 36: plan_bounds, ratio 1)
  const bool use2 = cfg.cand_ascending && ml <= cfg.long_row;
  const int64_t cb = std::max<int64_t>(1, std::min<int64_t>(std::min<int64_t>(cfg.max_cb, MAX_CALL_B),
                                                             cfg.budget / (NEXT_N * CAND_K * CAND_BLOCK * 4)));
  const int64_t B = gtT.B;
  const int64_t nch = (B + cb - 1) / cb;
  if (nch > cfg.max_chunks) return Status::BAD_ARG;
  st.use2 = use2, st.single = B <= cb;
  const int64_t nb = (ml + CAND_BLOCK - 1) / CAND_BLOCK;   // (max_len.get(24, 1): set)
  st.nwords = (int32_t)((nb + 31) / 32);
  st.B = B, st.IR = gtT.IR, st.nv = gtT.n;
  const int64_t K = NEXT_N * CAND_K;
  auto c4 = [](int64_t n) { return (n + 3) / 4 * 4; };
  for (int64_t b0 = 0; b0 < B; b0 += cb) {
    const int64_t b1 = std::min(B, b0 + cb), Bc = b1 - b0;
    IdxChunkHost k;                             // EagerWorkspace.prep_bufs(Bc, cblk=not use2, upk=use2): int32 carving
    k.b0 = b0, k.b1 = b1;
    int64_t o = 0;
    auto take = [&](int64_t n) { const uint64_t a = ws.prep + 4 * (uint64_t)o; o += c4(n); return a; };
    k.ublk = take(Bc * K);
    k.uslot = take(Bc * K * NEXT_N);
    k.nunion = take(Bc);
    const uint64_t cblk_ws = !use2 ? take(Bc * NEXT_N * CAND_K) : 0;
    k.clen = take(Bc * NEXT_N);
    k.upk = use2 ? take(Bc * K) : 0;
    if (use2) k.cblk = ws.cand + 4 * (uint64_t)(4 * b0 * CAND_K), k.ld_cblk = CAND_K;
    else k.cblk = cblk_ws, k.ld_cblk = CAND_K;  // (prep_bufs' cblk view [Bc * 4, CAND_K]: stride CAND_K)
    k.comp = ws.f32;                            // ws.logits(4 * Bc, CAND_K * CAND_BLOCK): the f32 base
    k.bm = ws.bmax;                             // ws.bm(4 * Bc, CAND_K): the bmax base
    st.ch.push_back(k);
  }
  return Status::OK;
}

Status pf_step(const PfHost& pf, const std::vector<int32_t>& ratios_in, const PfStepViews& v, const std::vector<CombTemplate>& tpl,
               TrCheck check, const PfConsts& k, PfStepHost* out) {
  PfStepHost& st = *out;
  st = PfStepHost();
  std::vector<int32_t> ratios(ratios_in);
  std::sort(ratios.begin(), ratios.end());
  ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());
  if (!pf.has_pf || pf.n <= 0 || ratios.size() > 4) return Status::BAD_ARG;
  for (int32_t ra : ratios)
    if (ra < 0 || ra > 2 || pf.ratio[ra].groups.size() > 32) return Status::BAD_ARG;
  auto need0 = [&](int32_t ra) {                // _pf_step's need(): Plan.group_ws without the preallocation
    int64_t sum = 0, mx = 0;
    for (const auto& g : pf.ratio[ra].groups) sum += g.area, mx = std::max(mx, g.area);
    const bool dis = pf.ratio[ra].groups.size() == 1 || sum * 2 <= k.ws_total;
    return dis ? sum : mx;
  };
  std::stable_sort(ratios.begin(), ratios.end(), [&](int32_t x, int32_t y) { return need0(x) > need0(y); });
  st.n = pf.n, st.row_lo = pf.row_lo, st.row_hi = pf.row_hi;
  for (int32_t ra : ratios) {
    const PfRatio& R = pf.ratio[ra];
    if (R.alloc_fallback || R.shared_fallback) return Status::BAD_ARG;   // (Python: stats_ws moved -> None)
    PfLayoutHost L;
    L.ratio = ra, L.disjoint = R.disjoint;
    const int32_t top_k = ra ? 512 : 0;
    const int64_t cw = (top_k + k.win + 127) / 128 * 128;
    const int ci = cw == 640 ? 0 : cw == 128 ? 1 : -1;
    if (ci < 0) return Status::BAD_ARG;
    for (const auto& g : R.groups) {
      PfGroupHost G;
      G.a = g.a, G.b = g.b, G.N = g.N, G.M = g.M, G.kv = v.ws + 2 * (uint64_t)g.off;
      G.q0 = pf.qsl[(size_t)g.a], G.q1 = pf.qsl[(size_t)g.b];
      // the overrides (mixed_cxx_attn._pf_step's ov)
      const std::pair<const char*, uint64_t> ov[] = {
          {"combined_indices_ptr", v.comb[ci] + 4 * (uint64_t)(G.q0 * cw)}, {"combined_indices_stride", (uint64_t)cw},
          {"combined_lens_ptr", v.clen[ci] + 4 * (uint64_t)G.q0},
          {"topk_indices_ptr", v.loc + 4 * (uint64_t)(G.q0 * v.loc_ld)}, {"topk_indices_stride", (uint64_t)v.loc_ld},
          {"query_start_loc_ptr", v.qsl + 4 * (uint64_t)g.a}, {"seq_lens_ptr", v.seq_lens + 4 * (uint64_t)g.a},
          {"gather_lens_ptr", v.gather_lens + 4 * (uint64_t)g.a}, {"M", (uint64_t)G.M}, {"N", (uint64_t)G.N}};
      int found = -1;
      for (size_t t = 0; t < tpl.size() && found < 0; ++t) {
        const CombTemplate& T = tpl[t];
        if (T.ratio != ra || T.top_k != top_k || T.names.size() != T.base.size() || T.base.size() > 16) continue;
        std::vector<uint64_t> vals(T.base);
        for (const auto& o : ov)
          for (size_t i = 0; i < T.names.size(); ++i)
            if (T.names[i] == o.first) vals[i] = o.second;
        if (check == nullptr || check(T.record, vals.data()) == 0) {
          found = (int)t;
          G.vals = vals;
          G.gy = T.gy;
        }
      }
      if (found < 0) return Status::BAD_ARG;   // (a key class not recorded at ensure time: the step runs embedded)
      auto it = std::find(st.recs.begin(), st.recs.end(), found);
      if (it == st.recs.end()) {
        if (st.recs.size() >= 8) return Status::BAD_ARG;
        st.recs.push_back(found);
        it = st.recs.end() - 1;
      }
      G.rec = (int32_t)(it - st.recs.begin());
      L.g.push_back(G);
    }
    st.lay.push_back(L);
  }
  return Status::OK;
}

Status parse_comb_templates(const void* buf, int64_t bytes, int64_t record_size, std::vector<CombTemplate>* out) {
  out->clear();
  if (buf == nullptr || bytes == 0) return Status::OK;   // (nothing exported yet)
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  const uint8_t* end = p + bytes;
  auto need = [&](int64_t n) { return end - p >= n; };
  auto u32 = [&]() { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
  auto i32 = [&]() { int32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
  if (!need(16)) return Status::BAD_ARG;
  const uint32_t magic = u32(), version = u32(), count = u32(), rsz = u32();
  if (magic != COMB_TPL_MAGIC || version != COMB_TPL_VERSION || (int64_t)rsz != record_size) return Status::BAD_ARG;
  auto align8 = [&]() { const uintptr_t o = (uintptr_t)(p - static_cast<const uint8_t*>(buf)); p += (8 - o % 8) % 8; };
  for (uint32_t t = 0; t < count; ++t) {
    if (!need(16)) return Status::BAD_ARG;
    CombTemplate T;
    T.ratio = i32(), T.top_k = i32(), T.gy = i32();
    const int32_t n = i32();
    if (n < 0 || n > 64 || !need(rsz)) return Status::BAD_ARG;
    T.record = p;
    p += rsz;
    for (int32_t i = 0; i < n; ++i) {
      if (!need(2)) return Status::BAD_ARG;
      uint16_t len;
      std::memcpy(&len, p, 2);
      p += 2;
      if (!need(len)) return Status::BAD_ARG;
      T.names.emplace_back(reinterpret_cast<const char*>(p), len);
      p += len;
    }
    align8();
    if (!need(8LL * n)) return Status::BAD_ARG;
    for (int32_t i = 0; i < n; ++i) {
      uint64_t v;
      std::memcpy(&v, p, 8);
      p += 8;
      T.base.push_back(v);
    }
    out->push_back(std::move(T));
  }
  return Status::OK;
}

}  // namespace p1
}  // namespace fpr
