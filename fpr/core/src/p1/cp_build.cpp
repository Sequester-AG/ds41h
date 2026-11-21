// cp_build.cpp -- see cp_build.h.
#include "cp_build.h"

#include <algorithm>
#include <map>

namespace fpr {
namespace p1 {

static int64_t floordiv(int64_t a, int64_t b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }   // Python //

namespace {
struct Group {
  int64_t id;
  std::vector<std::tuple<int32_t, int64_t, int64_t>> subs;
  std::map<int32_t, const SeqEntry*> ents;
  std::vector<int32_t> ranks;
  std::map<std::pair<int32_t, int32_t>, std::vector<int64_t>> pages;   // (r, q) -> rank r's pages of sub q's blocks
  std::map<int32_t, std::vector<int64_t>> halo_src, halo_dst;
};
}  // namespace

Status cp_host(const std::vector<StepBatch>& batches, const std::vector<int32_t>& layers, bool idx_on, const CpGeom& geom,
               CpPlanHost* out) {
  CpPlanHost& P = *out;
  P = CpPlanHost();
  std::vector<Group> groups;                     // first-seen order (Python dict insertion order)
  for (size_t r = 0; r < batches.size(); ++r)
    for (const SeqEntry& se : batches[r].seqs) {
      if (!se.has_cp) continue;
      auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) { return g.id == se.cp.group; });
      if (it == groups.end()) {
        Group g;
        g.id = se.cp.group;
        g.subs = se.cp.subs;
        groups.push_back(g);
        it = groups.end() - 1;
      }
      it->ents[(int32_t)r] = &se;
    }
  for (Group& g : groups) {
    for (auto& s : g.subs) g.ranks.push_back(std::get<0>(s));
    std::vector<int32_t> a, b(g.ranks);
    for (auto& kv : g.ents) a.push_back(kv.first);
    std::sort(b.begin(), b.end());
    if (a != b) return Status::BAD_ARG;          // "CP group: entries on ranks ..., subs ..."
    for (int32_t r : g.ranks) {
      const auto& bt = *g.ents[r]->block_table;
      for (auto& s : g.subs) {
        const int32_t q = std::get<0>(s);
        const int64_t st = std::get<1>(s), n = std::get<2>(s);
        std::vector<int64_t> v;
        for (int64_t blk = st / geom.page_tokens; blk < (st + n + geom.page_tokens - 1) / geom.page_tokens; ++blk) {
          if (blk < 0 || blk >= (int64_t)bt.size()) return Status::BAD_ARG;
          v.push_back(bt[(size_t)blk]);
        }
        g.pages[{r, q}] = v;
      }
    }
    for (size_t i = 1; i < g.ranks.size(); ++i) {
      const int32_t p = g.ranks[i - 1], q = g.ranks[i];
      const int64_t st = std::get<1>(g.subs[i]);
      std::vector<int64_t> hb;
      for (int64_t blk = floordiv(st - geom.win, geom.wpt); blk < floordiv(st, geom.wpt); ++blk)
        if (blk >= 0) hb.push_back(blk);
      if (hb.empty()) continue;
      std::vector<int64_t> hs, hd;
      for (int64_t blk : hb) {
        const WindowTable& wp = *g.ents[p]->window_table;
        const WindowTable& wq = *g.ents[q]->window_table;
        if (!wp.contains(blk) || !wq.contains(blk)) return Status::BAD_ARG;   // (KeyError)
        hs.push_back(wp.get(blk, -1));
        hd.push_back(wq.get(blk, -1));
      }
      g.halo_src[p] = hs;
      g.halo_dst[q] = hd;
    }
  }
  P.n_groups = (int64_t)groups.size();
  // the per-rank blobs: a list is stored once per (rank, list key), in first-use order
  std::map<std::tuple<int, int64_t, int, int32_t, int32_t>, int64_t> where;   // (rank, group, kind, a, b) -> offset
  auto put = [&](int rank, int64_t gid, int kind, int32_t a, int32_t b, const std::vector<int64_t>& v) -> int64_t {
    auto key = std::make_tuple(rank, gid, kind, a, b);
    auto it = where.find(key);
    if (it != where.end()) return it->second;
    auto& bl = P.blob[(size_t)rank];
    const int64_t off = (int64_t)bl.size();
    bl.insert(bl.end(), v.begin(), v.end());
    where[key] = off;
    return off;
  };
  for (int32_t L : layers) {
    if (L > 20) continue;
    const CpSection* sec = nullptr;
    for (const auto& c : geom.sources)
      if (c.layer == L) sec = &c;
    if (idx_on && sec != nullptr) {              // x1a_send: r -> every later rank of its group, comp then idx section
      for (Group& g : groups)
        for (size_t i = 0; i < g.ranks.size(); ++i) {
          const int32_t r = g.ranks[i];
          for (size_t j = i + 1; j < g.ranks.size(); ++j) {
            const int32_t q = g.ranks[j];
            const auto& src = g.pages[{r, r}];
            const auto& dst = g.pages[{q, r}];
            const int64_t so = put(r, g.id, 1, r, r, src), dof = put(q, g.id, 1, q, r, dst);
            P.msgs.push_back({r, q, L, CP_X1A, (int64_t)src.size(), sec->comp_len, sec->comp_off, so, dof});
            P.msgs.push_back({r, q, L, CP_X1A, (int64_t)src.size(), sec->idx_len, sec->idx_off, so, dof});
          }
        }
    }
    for (Group& g : groups)                      // x2_send: r -> the next rank (its window halo of layer L)
      for (size_t i = 0; i < g.ranks.size(); ++i) {
        const int32_t r = g.ranks[i];
        auto hs = g.halo_src.find(r);
        if (hs == g.halo_src.end()) continue;
        if (i + 1 >= g.ranks.size()) return Status::BAD_ARG;
        const int32_t q = g.ranks[i + 1];
        auto hd = g.halo_dst.find(q);
        if (hd == g.halo_dst.end()) return Status::BAD_ARG;
        const int64_t so = put(r, g.id, 2, r, -1, hs->second), dof = put(q, g.id, 3, q, -1, hd->second);
        P.msgs.push_back({r, q, L, CP_X2, (int64_t)hs->second.size(), geom.wlb, (int64_t)L * geom.wlb, so, dof});
      }
  }
  return Status::OK;
}

Status cp_fill(const CpPlanHost& plan, const std::vector<CpRankViews>& views, std::vector<CpMsgDev>* out) {
  out->clear();
  for (size_t r = 0; r < views.size() && r < plan.blob.size(); ++r)
    if (!plan.blob[r].empty() && (views[r].idx_dev == nullptr || (int64_t)plan.blob[r].size() > views[r].idx_cap))
      return Status::BAD_ARG;
  for (const CpMsgT& m : plan.msgs) {
    if (m.src_rank < 0 || m.dst_rank < 0 || (size_t)m.src_rank >= views.size() || (size_t)m.dst_rank >= views.size())
      return Status::BAD_ARG;
    const CpRankViews& S = views[(size_t)m.src_rank];
    const CpRankViews& D = views[(size_t)m.dst_rank];
    CpMsgDev d{};
    d.src_rank = m.src_rank, d.dst_rank = m.dst_rank, d.layer = m.layer, d.kind = m.kind;
    d.n = m.n, d.len = m.len, d.off = m.off;
    d.src_idx = S.idx_dev + m.src_off, d.dst_idx = D.idx_dev + m.dst_off;
    const bool x1a = m.kind == CP_X1A;
    d.src_base = x1a ? S.paged : S.window, d.dst_base = x1a ? D.paged : D.window;
    d.src_rows = x1a ? S.paged_rows : S.window_rows, d.src_ld = x1a ? S.paged_ld : S.window_ld;
    d.dst_rows = x1a ? D.paged_rows : D.window_rows, d.dst_ld = x1a ? D.paged_ld : D.window_ld;
    d.src_dev = S.device, d.dst_dev = D.device;
    d.mail_stream = D.mail_stream;
    if (d.src_base == nullptr || d.dst_base == nullptr || d.mail_stream == nullptr || m.off < 0 || m.off + m.len > d.src_ld ||
        m.off + m.len > d.dst_ld)
      return Status::BAD_ARG;                    // (= pieces_attn attn_cp_step's range check, before any device work)
    out->push_back(d);
  }
  return Status::OK;
}

}  // namespace p1
}  // namespace fpr
