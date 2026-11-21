// pf_build.cpp -- see pf_build.h.
#include "pf_build.h"

#include <algorithm>

namespace fpr {
namespace p1 {

static void group_plan(const PfHost& p, int ratio, const PfConsts& k, std::vector<PfGroup>* out) {
  // Plan.group_plan: consecutive requests sharing one workspace (<= chunk, area cap; a request alone is never split)
  const int64_t S = p.S;
  std::vector<int64_t> comp(S);
  for (int64_t s = 0; s < S; ++s) comp[s] = ratio ? p.ends[s] / ratio : 0;
  int64_t a = 0;
  while (a < S) {
    int64_t N = comp[a], G = p.gls[a], b = a + 1;
    while (b < S && b - a < k.chunk) {
      const int64_t N2 = std::max(N, comp[b]), G2 = std::max(G, p.gls[b]);
      if ((b - a + 1) * (N2 + G2) * 1024 > k.ws_cap) break;
      N = N2, G = G2, ++b;
    }
    out->push_back({a, b, N, N + G, 0, 0});
    a = b;
  }
}

// Plan.__init__'s host part + group_plan / group_ws on the (row0, n, start, replay_start) tuples (p.bt / bt_cols set)
static Status pf_plan(const std::vector<std::array<int64_t, 4>>& seqs, int64_t ws_numel, const PfConsts& k, PfHost* out);

Status pf_host(const StepBatch& sb, int64_t ws_numel, const PfConsts& k, PfHost* out) {
  PfHost& p = *out;
  p = PfHost();
  std::vector<const SeqEntry*> pfs;
  for (const SeqEntry& se : sb.seqs)
    if (se.kind != "decode") pfs.push_back(&se);
  if (pfs.empty()) return Status::OK;
  p.has_pf = true;
  p.pf_a = pfs.front()->row0, p.pf_b = pfs.back()->row0 + pfs.back()->n_rows;
  p.S = (int64_t)pfs.size();
  // _attn_pf_plans: nbs = ceil((start + n_rows) / 64); bt [S, max(nbs)] int32, zero padded
  std::vector<int64_t> nbs;
  for (const SeqEntry* se : pfs) nbs.push_back((se->start + se->n_rows + 63) / 64);
  p.bt_cols = *std::max_element(nbs.begin(), nbs.end());
  p.bt.assign((size_t)(p.S * p.bt_cols), 0);
  for (int64_t s = 0; s < p.S; ++s) {
    const auto& tb = *pfs[s]->block_table;
    if ((int64_t)tb.size() < nbs[s]) return Status::BAD_ARG;     // (Python: as_tensor(block_table[:nb]) shape mismatch)
    for (int64_t j = 0; j < nbs[s]; ++j) p.bt[s * p.bt_cols + j] = (int32_t)tb[j];
  }
  std::vector<std::array<int64_t, 4>> seqs;
  for (const SeqEntry* se : pfs) seqs.push_back({se->row0, se->n_rows, se->start, se->replay_start});
  return pf_plan(seqs, ws_numel, k, out);
}

Status pf_host(const PfSnap& snap, const PfIn& in, const StepBatch& sb, PfHost* out) {
  PfHost& p = *out;
  p = PfHost();
  std::vector<const SeqEntry*> pfs;
  for (const SeqEntry& se : sb.seqs)
    if (se.kind != "decode") pfs.push_back(&se);
  if (pfs.empty()) return (in.seqs == nullptr || in.seqs->empty()) ? Status::OK : Status::BAD_ARG;
  if (in.seqs == nullptr || in.bt == nullptr || in.seqs->size() != pfs.size() || in.bt_rows != (int64_t)pfs.size() ||
      (int64_t)in.bt->size() != in.bt_rows * in.bt_cols)
    return Status::BAD_ARG;
  for (size_t i = 0; i < pfs.size(); ++i) {
    const auto& t = (*in.seqs)[i];
    if (t[0] != pfs[i]->row0 || t[1] != pfs[i]->n_rows || t[2] != pfs[i]->start || t[3] != pfs[i]->replay_start)
      return Status::BAD_ARG;
  }
  if (!snap.ws || snap.ws_numel <= 0) return Status::BAD_ARG;   // (native: no per-step workspace allocation)
  p.has_pf = true;
  p.pf_a = pfs.front()->row0, p.pf_b = pfs.back()->row0 + pfs.back()->n_rows;
  p.S = (int64_t)pfs.size();
  p.bt = *in.bt, p.bt_cols = in.bt_cols;
  const Status rc = pf_plan(*in.seqs, snap.ws_numel, snap.k, out);
  if (rc != Status::OK) return rc;
  for (const PfRatio& R : p.ratio)
    if (R.alloc_fallback) return Status::BAD_ARG;              // (Python would allocate past the preallocation)
  if ((snap.S_max > 0 && p.S > snap.S_max) || (snap.bt_cols_max > 0 && p.bt_cols > snap.bt_cols_max) ||
      (snap.wbt_cols_max > 0 && p.wmax > snap.wbt_cols_max))
    return Status::BAD_ARG;                                    // (the persistent plan tables' sizes)
  return Status::OK;
}

static Status pf_plan(const std::vector<std::array<int64_t, 4>>& seqs, int64_t ws_numel, const PfConsts& k, PfHost* out) {
  PfHost& p = *out;
  // Plan.__init__ (host part)
  p.row_lo = seqs.front()[0];
  int64_t r = p.row_lo;
  p.qsl.push_back(0);
  for (const auto& t : seqs) {
    const int64_t row0 = t[0], n = t[1], first = t[2], rs = t[3];
    if (row0 != r || n <= 0) return Status::BAD_ARG;              // "prefill rows must be contiguous, in sequence order"
    r += n;
    p.firsts.push_back(first);
    p.ends.push_back(first + n);
    p.g0s.push_back(std::max<int64_t>(std::max<int64_t>(rs, first - (k.win - 1)), 0));
    p.qsl.push_back(r - p.row_lo);
  }
  p.row_hi = r, p.n = r - p.row_lo;
  int64_t mx = 0;
  for (int64_t s = 0; s < p.S; ++s) {
    p.gls.push_back(p.ends[s] - p.g0s[s]);
    p.k0.push_back(p.firsts[s] - p.g0s[s] + 1);
    mx = std::max(mx, p.ends[s]);
  }
  p.wmax = mx / k.wpt + 1;
  // Plan.group_plan / group_ws per ratio
  for (int ratio = 0; ratio <= 2; ++ratio) {
    PfRatio& R = p.ratio[ratio];
    group_plan(p, ratio, k, &R.groups);
    int64_t sum = 0, mxa = 0;
    for (auto& g : R.groups) {
      g.area = (g.b - g.a) * g.M * 512;
      sum += g.area;
      mxa = std::max(mxa, g.area);
    }
    R.disjoint = R.groups.size() == 1 || sum * 2 <= k.ws_total;
    R.need = R.disjoint ? sum : mxa;
    if (ws_numel > 0) {
      if (R.need > ws_numel && R.disjoint && R.groups.size() > 1) {
        R.disjoint = false, R.need = mxa, R.shared_fallback = true;
      }
      if (R.need > ws_numel) R.alloc_fallback = true;
    }
    int64_t off = 0;
    for (auto& g : R.groups) {
      g.off = off;
      off += R.disjoint ? g.area : 0;
    }
  }
  return Status::OK;
}

}  // namespace p1
}  // namespace fpr
