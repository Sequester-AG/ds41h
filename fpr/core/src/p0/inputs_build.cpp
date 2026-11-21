// inputs_build.cpp -- see inputs_build.h
#include "inputs_build.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace fpr {
namespace p0 {

namespace {
Status bad(std::string* err, const std::string& m) {
  if (err) *err = m;
  return Status::BAD_ARG;
}
}  // namespace

InLayout in_layout(int64_t R) {
  InLayout L;
  int64_t off = 0;
  for (int f = 0; f < IN_NFIELDS; ++f) {
    L.off[f] = off;
    off += (R * kInCols[f] * kInElem[f] + 7) / 8 * 8;
  }
  L.bytes = off;
  return L;
}

Status build_inputs(const InputsSnap& snap, const RankTables& tab, InputsOut* out, std::string* err) {
  const int64_t R = snap.R, n = tab.n, W = kSlidingWindow;
  if (R <= 0) return bad(err, "R <= 0");
  if (n < 0 || n > R) return bad(err, "n " + std::to_string(n) + " > R " + std::to_string(R));
  if (n && ((int64_t)tab.tokens.size() != n || (int64_t)tab.positions.size() != n || (int64_t)tab.own_slot.size() != n ||
            (int64_t)tab.si.size() != n * W || (int64_t)tab.sl.size() != n))
    return bad(err, "table field sizes != n");
  if (snap.scratch < std::numeric_limits<int32_t>::min() || snap.scratch > std::numeric_limits<int32_t>::max())
    return bad(err, "scratch slot outside int32 (si[:, 0] = scratch)");
  const int64_t ns = n ? (int64_t)tab.sample_rows.size() : 0;
  if (ns > R) return bad(err, "sample rows > R");
  // _upload_v2 (dev_pos False): every field rewritten
  std::vector<int64_t> pos(R, 0), tok(R, 0), own(R, snap.scratch), samp(R, 0);
  std::vector<int32_t> si(R * W, -1), sl(R, 1);
  std::vector<uint8_t> valid(R, 0);
  for (int64_t i = 0; i < R; ++i) si[i * W] = (int32_t)snap.scratch;
  if (n) {
    const int64_t repl = snap.hsd ? 0 : 1000;   // decode rows: -1 = the engine's device next ids (HSD); else a fixed id
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t = tab.tokens[i];
      tok[i] = t < 0 ? repl : t;
      pos[i] = tab.positions[i];
      own[i] = tab.own_slot[i];
      sl[i] = tab.sl[i];
      valid[i] = 1;
    }
    std::copy(tab.si.begin(), tab.si.begin() + n * W, si.begin());
    for (int64_t j = 0; j < ns; ++j) samp[j] = tab.sample_rows[j];
  }
  const InLayout L = in_layout(R);
  out->blob.assign(L.bytes, 0);
  std::memcpy(out->blob.data() + L.off[IN_POS], pos.data(), R * 8);
  std::memcpy(out->blob.data() + L.off[IN_TOK], tok.data(), R * 8);
  std::memcpy(out->blob.data() + L.off[IN_OWN], own.data(), R * 8);
  std::memcpy(out->blob.data() + L.off[IN_SAMP], samp.data(), R * 8);
  std::memcpy(out->blob.data() + L.off[IN_SI], si.data(), R * W * 4);
  std::memcpy(out->blob.data() + L.off[IN_SL], sl.data(), R * 4);
  std::memcpy(out->blob.data() + L.off[IN_VALID], valid.data(), R);
  out->n = n;
  out->n_sample = ns;
  out->rows.clear();
  out->src.clear();
  out->mig.clear();
  // Hooks.device_tokens -> _device_tokens_v2 (returns at once without HSD or rows)
  if (!snap.hsd || !n) return Status::OK;
  for (const RankSeqRow& q : tab.seq_rows) {
    if (q.kind != "decode") continue;
    auto it = snap.prev.find(q.rid);
    if (it == snap.prev.end() || it->second.rank < 0) continue;          // prev_of None
    const int64_t a = q.row0, e = std::min<int64_t>((int64_t)q.row0 + q.n_rows, n);
    if (a < 0 || a >= e) return bad(err, "min() of an empty token slice (decode sequence " + q.rid + ")");
    int64_t mn = tab.tokens[a];
    for (int64_t i = a + 1; i < e; ++i) mn = std::min(mn, tab.tokens[i]);
    if (mn >= 0) continue;                                              // forced / host-given ids win
    const int32_t r0 = it->second.rank, si0 = it->second.seq;
    if (r0 != snap.rank) {                                              // migrated: one cross-device copy
      out->mig.push_back(MigCopy{r0, si0, q.row0, q.n_rows});
      continue;
    }
    const int64_t b = (int64_t)si0 * snap.k1;
    for (int32_t j = 0; j < q.n_rows; ++j) {
      out->rows.push_back((int64_t)q.row0 + j);
      out->src.push_back(b + j);
    }
  }
  return Status::OK;
}

void apply_device_tokens(std::vector<uint8_t>* blob, int64_t R, int32_t rank, const InputsOut& out,
                         const std::vector<const int64_t*>& next_ids, int32_t k1) {
  const InLayout L = in_layout(R);
  uint8_t* t = blob->data() + L.off[IN_TOK];
  auto put = [&](int64_t row, int64_t v) { std::memcpy(t + row * 8, &v, 8); };
  for (const MigCopy& m : out.mig)
    for (int32_t j = 0; j < m.n_rows; ++j) put((int64_t)m.row0 + j, next_ids[m.r0][(int64_t)m.si0 * k1 + j]);
  for (size_t i = 0; i < out.rows.size(); ++i) put(out.rows[i], next_ids[rank][out.src[i]]);
}

VidxOut build_vidx(const std::vector<int64_t>& ns, const std::vector<int64_t>* nds, int64_t R, bool act8_on) {
  VidxOut o;
  for (size_t r = 0; r < ns.size(); ++r) {
    for (int64_t i = 0; i < ns[r]; ++i) o.vidx.push_back((int64_t)r * R + i);
    o.nv.push_back((int32_t)ns[r]);
  }
  bool act8 = act8_on && nds != nullptr && !nds->empty() && *std::min_element(nds->begin(), nds->end()) >= 0;
  if (act8) {
    o.has_nd = true;
    int64_t s = 0;
    for (int64_t x : *nds) o.nd.push_back((int32_t)x), s += x;
    o.ndec = s;
  }
  o.vcount = (int64_t)o.vidx.size();
  o.act8_fallback = act8_on && nds != nullptr && !act8;
  return o;
}

}  // namespace p0
}  // namespace fpr
