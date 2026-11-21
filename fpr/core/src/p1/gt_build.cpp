// gt_build.cpp -- see gt_build.h.
#include "gt_build.h"

#include <algorithm>

namespace fpr {
namespace p1 {

const char* gt_status_name(GtStatus s) {
  switch (s) {
    case GtStatus::OK: return "ok";
    case GtStatus::NO_DELTAS: return "no_deltas";
    case GtStatus::HOST_PATH: return "host_path";
    case GtStatus::IDLE: return "idle";
  }
  return "?";
}

std::array<int64_t, 13> gt_layout(int64_t n, int64_t S, int64_t U, int64_t P) {
  const int64_t sizes[12] = {n, n, S, S, S, S + 1, S + 1, U, U, U, U, P};
  std::array<int64_t, 13> o{};
  int64_t x = 0;
  for (int i = 0; i < 12; ++i) {
    o[i] = x;
    x += sizes[i];
  }
  o[12] = x;
  return o;
}

GtStatus gt_host(const RankTables& tab, bool has_deltas, int64_t R, int64_t Bc_in, GtHost* out) {
  GtHost& g = *out;
  g = GtHost();
  g.Bc = Bc_in;
  if (!has_deltas) return g.status = GtStatus::NO_DELTAS;
  const int64_t n = tab.n;
  const int64_t S = n ? (int64_t)tab.seq_rows.size() : 0;
  g.n = n, g.S = S;
  if (!n) {                                          // (the dict's idle entry)
    g.status = GtStatus::IDLE;
    g.B = 1, g.IR = 4, g.max_seq_len = 1;
    g.row2ir.assign((size_t)std::max<int64_t>(R, 0), -1);
    return g.status;
  }
  // nr = n_rows[:S]; rows sequence-contiguous in sequence order (seq_slot == repeat(arange(S), nr)), sum(nr) == n
  const auto& nrv = tab.seq.n_rows;
  if ((int64_t)nrv.size() < S || (int64_t)tab.seq_slot.size() < n) return g.status = GtStatus::HOST_PATH;
  int64_t tot = 0;
  for (int64_t s = 0; s < S; ++s) tot += nrv[s];
  if (tot != n) return g.status = GtStatus::HOST_PATH;
  {
    int64_t row = 0;
    for (int64_t s = 0; s < S; ++s)
      for (int32_t j = 0; j < nrv[s]; ++j, ++row)
        if (tab.seq_slot[row] != s) return g.status = GtStatus::HOST_PATH;
  }
  std::vector<int64_t> crow(S + 1, 0), pr0(S + 1, 0);
  for (int64_t s = 0; s < S; ++s) {
    crow[s + 1] = crow[s] + nrv[s];
    pr0[s + 1] = pr0[s] + (nrv[s] + 3) / 4;
  }
  g.B = pr0[S], g.IR = 4 * g.B;
  int64_t maxp = tab.positions[0];
  for (int64_t i = 1; i < n; ++i) maxp = std::max(maxp, tab.positions[i]);
  g.max_seq_len = maxp + 1;
  if (g.B > g.Bc) {                                 // GeneralTables._alloc(-(-B // 64) * 64)
    g.Bc = (g.B + 63) / 64 * 64;
    g.grew = true;
  }
  const auto& q = tab.seq;
  const int64_t U = (int64_t)q.bt_upd_seq.size(), P = (int64_t)q.bt_upd_pages.size();
  g.U = U, g.P = P;
  g.o = gt_layout(n, S, U, P);
  const auto& o = g.o;
  g.stage.assign((size_t)o[12], 0);
  int32_t* h = g.stage.data();
  for (int64_t i = 0; i < n; ++i) h[o[0] + i] = (int32_t)tab.paged_slot[i];       // _fill: ps
  for (int64_t s = 0; s < S; ++s) h[o[2] + s] = q.state_slot[s];                  // sslot
  for (int64_t i = 0; i < n; ++i) h[o[1] + i] = tab.seq_slot[i];                  // seqslot
  for (int64_t s = 0; s < S; ++s) {
    h[o[3] + s] = (int32_t)crow[s];                                                // row0 = crow[:S]
    h[o[4] + s] = nrv[s];                                                          // nrows
  }
  for (int64_t s = 0; s <= S; ++s) {
    h[o[5] + s] = (int32_t)pr0[s];
    h[o[6] + s] = (int32_t)crow[s];
  }
  if (U) {
    int64_t acc = 0;
    for (int64_t u = 0; u < U; ++u) {
      h[o[7] + u] = q.bt_upd_seq[u];
      h[o[8] + u] = q.bt_upd_start[u];
      h[o[9] + u] = q.bt_upd_len[u];
      h[o[10] + u] = (int32_t)acc;                                                 // uo = [0, cumsum(ul[:-1])]
      acc += q.bt_upd_len[u];
    }
    for (int64_t p = 0; p < P; ++p) h[o[11] + p] = q.bt_upd_pages[p];
  }
  // row2ir[:n] = repeat(4 * pr0[:S] - crow[:S], nr) + arange(n), -1 past n
  g.row2ir.assign((size_t)std::max<int64_t>(R, n), -1);
  g.row2ir.resize((size_t)std::max<int64_t>(R, 0));
  {
    int64_t row = 0;
    for (int64_t s = 0; s < S; ++s)
      for (int32_t j = 0; j < nrv[s]; ++j, ++row)
        if (row < (int64_t)g.row2ir.size()) g.row2ir[row] = 4 * pr0[s] - crow[s] + row;
  }
  // idx2f.pf_groups([(pr0[s], pr0[s + 1]) for s in range(S)])
  for (int64_t s = 0; s < S; ++s)
    for (int64_t x = pr0[s]; x < pr0[s + 1]; x += 4) g.pf_groups.push_back(x);
  if (S) g.pf_groups.push_back(pr0[S]);
  // OwnedIndexer.plan_bounds(ins, max_len1 = maxp + 1)
  for (int i = 0; i < GT_NLAYERS; ++i) g.max_len[i] = g.max_seq_len / GT_RATIO[i];
  for (int i = 0; i < GT_NSRC; ++i) g.width[i] = std::max<int64_t>(256, (g.max_len[i] + 255) / 256 * 256);
  return g.status = GtStatus::OK;
}

Status gt_host(const GtSnap& snap, char role, int rank, int64_t R, const StepBatch& sb, const RankTables& tab, GtHost* out) {
  if ((role != 'A' && role != 'T') || snap.mirror || tab.n != sb.n_rows) return Status::BAD_ARG;
  const GtStatus st = gt_host(tab, snap.has_deltas, R, snap.Bc, out);
  if (st == GtStatus::NO_DELTAS || st == GtStatus::HOST_PATH || out->grew) return Status::BAD_ARG;
  if (st == GtStatus::OK && snap.ring_cap > 0 && out->o[12] > snap.ring_cap) return Status::BAD_ARG;   // (no _grow mid-step)
  return Status::OK;
}

}  // namespace p1
}  // namespace fpr
