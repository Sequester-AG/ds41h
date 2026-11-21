// engram_host.cpp -- see engram_host.h.
#include "engram_host.h"

#include <cstring>
#include <sstream>

namespace fpr {
namespace p0 {

namespace {

Status fail(std::string* err, const std::string& m) {
  if (err) *err = m;
  return Status::BAD_ARG;
}

// look[i] = [lb[c] if start - 1 - c >= 0 else -1 for c in range(3)]  (assigned into an np.int32 array)
void fill_look(int32_t* look3, const std::array<int64_t, 3>& lb, int64_t st) {
  for (int c = 0; c < 3; ++c) look3[c] = (st - 1 - c >= 0) ? static_cast<int32_t>(lb[c]) : -1;
}

template <class T> std::string pylist(const std::vector<T>& v, size_t k) {   // Python's list.__str__ of v[:k]
  std::ostringstream o;
  o << '[';
  for (size_t i = 0; i < v.size() && i < k; ++i) o << (i ? ", " : "") << v[i];
  o << ']';
  return o.str();
}

}  // namespace

EgLayout eg_layout(int32_t R, int32_t S) {
  EgLayout L;
  L.max_rows = R;
  L.max_seqs = S;
  L.ids = 0;
  L.row0 = R;
  L.nrows = (int64_t)R + S;
  L.start = (int64_t)R + 2 * (int64_t)S;
  L.slot = (int64_t)R + 3 * (int64_t)S;
  L.look = (int64_t)R + 4 * (int64_t)S;
  L.counts = (int64_t)R + 7 * (int64_t)S;
  L.pk_n = (int64_t)R + 7 * (int64_t)S + 4;
  return L;
}

Status eg_args_from_tables(const RankTables& tab, int64_t pad_to, const EgLookback& lookback, EgArgs* out, std::string* err) {
  // sq = tab.seq; n = tab.n; S = len(tab.seq_rows) if n else 0
  const int64_t n = tab.n;
  const size_t S = n ? tab.seq_rows.size() : 0;
  const auto& q = tab.seq;
  if (q.n_rows.size() < S || q.start.size() < S || q.state_slot.size() < S || q.kind.size() < S)
    return fail(err, "eg_args_from_tables: seq arrays shorter than seq_rows");
  EgArgs& a = *out;
  a = EgArgs();
  a.nrows.assign(q.n_rows.begin(), q.n_rows.begin() + S);                 // nr = sq["n_rows"][:S] (int32)
  a.row0.assign(S, 0);                                                     // row0 = zeros(S, int32); cumsum(nr[:-1]) -> row0[1:]
  int32_t acc = 0;
  for (size_t i = 1; i < S; ++i) {
    acc += a.nrows[i - 1];
    a.row0[i] = acc;
  }
  a.start.resize(S);                                                       // start = sq["start"][:S] (int64; written as int32)
  for (size_t i = 0; i < S; ++i) a.start[i] = static_cast<int32_t>(q.start[i]);
  a.slot.assign(q.state_slot.begin(), q.state_slot.begin() + S);
  a.look.assign(3 * S, -2);
  for (size_t i = 0; i < S; ++i) {                                         // host = (kind != 0) | (nr > RING_W)
    if (q.kind[i] != 0 || q.n_rows[i] > EG_RING_W) {
      if (!lookback) return fail(err, "prefill chunks need host lookback ids");
      const int64_t st = q.start[i];                                       // int(start[i]) (the int64 value)
      fill_look(&a.look[3 * i], lookback(tab.seq_rows[i].rid, st), st);
    }
  }
  a.T = n > pad_to ? n : pad_to;
  a.n = n;
  a.rids.reserve(S);
  for (size_t i = 0; i < S; ++i) a.rids.push_back(tab.seq_rows[i].rid);
  return Status::OK;
}

Status eg_args_from_compact(const CompactTables& ct, int64_t pad_to, const EgLookback& lookback, EgArgs* out, std::string* err) {
  const size_t S = ct.slot.size();
  if (ct.n_rows.size() < S || ct.row0.size() < S || ct.start.size() < S || ct.kind.size() < S)
    return fail(err, "eg_args_from_compact: arrays shorter than slot");
  EgArgs& a = *out;
  a = EgArgs();
  a.nrows.resize(S);
  a.row0.resize(S);
  a.start.resize(S);
  a.slot.resize(S);
  for (size_t i = 0; i < S; ++i) {                                         // np.asarray(..., np.int32) / .astype(np.int32)
    a.nrows[i] = static_cast<int32_t>(ct.n_rows[i]);
    a.row0[i] = static_cast<int32_t>(ct.row0[i]);
    a.start[i] = static_cast<int32_t>(ct.start[i]);
    a.slot[i] = static_cast<int32_t>(ct.slot[i]);
  }
  a.look.assign(3 * S, -2);
  for (size_t i = 0; i < S; ++i) {
    if (ct.kind[i] != 0 || a.nrows[i] > EG_RING_W) {
      if (!lookback) return fail(err, "prefill chunks need host lookback ids");
      const int64_t st = a.start[i];                                       // int(start[i]) of the int32 array
      if (st < 0) return fail(err, "a prefill / long sequence needs a host start");
      if (i >= ct.rids.size()) return fail(err, "eg_args_from_compact: rids shorter than slot");
      fill_look(&a.look[3 * i], lookback(ct.rids[i], st), st);
    }
  }
  a.n = ct.n;
  a.T = ct.n > pad_to ? ct.n : pad_to;
  a.rids.assign(ct.rids.begin(), ct.rids.begin() + (ct.rids.size() < S ? ct.rids.size() : S));
  return Status::OK;
}

Status eg_pack(EgArgs* ap, const EgLayout& L, EgHostState* hs, int32_t* const staging[2], const int32_t* host_ids, EgPack* out,
               std::string* err) {
  EgArgs& a = *ap;
  const int64_t T = a.T;
  const size_t S = a.row0.size();
  const int64_t n = a.n >= 0 ? a.n : T;
  if (T > L.max_rows || (int64_t)S > L.max_seqs) return fail(err, "eg_pack: T / S beyond max_rows / max_seqs");
  if (a.nrows.size() != S || a.start.size() != S || a.slot.size() != S || a.look.size() != 3 * S)
    return fail(err, "eg_pack: StepArgs arrays of different lengths");
  *out = EgPack();
  hs->gen = (hs->gen + 1) & 0x0FFFFFFF;                                    // self._gen = (self._gen + 1) & 0x0FFFFFFF
  auto rid = [&](size_t i) { return i < a.rids.size() ? a.rids[i] : std::string("?"); };
  if (hs->guard) {                                                         // ENGRAM_GUARD: _guard_host
    bool pf = false;
    for (size_t i = 0; i < S && !pf; ++i) pf = a.look[3 * i] != -2;
    out->prefill = S > 0 && pf;
    int32_t smax = 0;
    for (size_t i = 0; i < S; ++i) smax = (i == 0 || a.slot[i] > smax) ? a.slot[i] : smax;
    if (S && smax >= hs->nring) {
      int k = 0;
      for (size_t i = 0; i < S; ++i) {
        if (a.slot[i] < hs->nring) continue;
        if (k++ < 4) {
          std::ostringstream o;
          o << "kind=ring-slot(host) layer=- seq=" << i << " row=" << a.row0[i] << " idx=" << a.slot[i] << " size=" << hs->nring
            << " start=" << a.start[i] << " n_rows=" << a.nrows[i] << " gen=" << hs->gen << " req=" << rid(i);
          out->logs.push_back(o.str());
        }
      }
      for (size_t i = 0; i < S; ++i)
        if (a.slot[i] >= hs->nring) a.slot[i] = -1;
    }
    if (out->prefill) {
      std::vector<int64_t> r0(S), nr(S);
      bool bad = false;
      for (size_t i = 0; i < S; ++i) {
        r0[i] = a.row0[i];
        nr[i] = a.nrows[i];
        if (nr[i] < 0) bad = true;
        if (i > 0 && r0[i] != r0[i - 1] + nr[i - 1]) bad = true;
      }
      if (r0[0] != 0 || r0[S - 1] + nr[S - 1] > n) bad = true;
      if (bad) {
        std::ostringstream o;
        o << "kind=row-layout(host) layer=- row0=" << pylist(r0, 8) << " nrows=" << pylist(nr, 8) << " n=" << n << " T=" << T
          << " gen=" << hs->gen << " req=" << rid(0);
        out->logs.push_back(o.str());
      }
      int k = 0;
      for (size_t i = 0; i < S && k < 4; ++i) {
        if (a.start[i] < 0 && a.look[3 * i] != -2) {
          ++k;
          std::ostringstream o;
          o << "kind=start(host) layer=- seq=" << i << " row=" << a.row0[i] << " idx=" << a.start[i] << " size=- start="
            << a.start[i] << " gen=" << hs->gen << " req=" << rid(i);
          out->logs.push_back(o.str());
        }
      }
    }
  }
  const int32_t i = hs->stage_i;                                           // i = self._stage_i; self._stage_i ^= 1
  hs->stage_i ^= 1;
  int32_t* h = staging[i];
  if (S) {
    std::memcpy(h + L.row0, a.row0.data(), S * sizeof(int32_t));
    std::memcpy(h + L.nrows, a.nrows.data(), S * sizeof(int32_t));
    std::memcpy(h + L.start, a.start.data(), S * sizeof(int32_t));
    std::memcpy(h + L.slot, a.slot.data(), S * sizeof(int32_t));
    std::memcpy(h + L.look, a.look.data(), 3 * S * sizeof(int32_t));
  }
  h[L.counts + 0] = static_cast<int32_t>(S);                               // (S, n, gen if guard else 0)
  h[L.counts + 1] = static_cast<int32_t>(n);
  h[L.counts + 2] = hs->guard ? hs->gen : 0;
  if (host_ids != nullptr) {                                               // host ids: h[:T] = a.ids[:T]; lo = 0
    std::memcpy(h + L.ids, host_ids, (size_t)T * sizeof(int32_t));
    out->lo = 0;
  } else {
    out->lo = L.row0;
  }
  out->stage = i;
  out->n_copy = L.pk_n - out->lo;
  return Status::OK;
}

}  // namespace p0
}  // namespace fpr
