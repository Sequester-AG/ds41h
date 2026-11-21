// inputs_build.cpp -- see inputs_build.h
#include "inputs_build.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

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
  // _upload_v2 (dev_pos False): every field rewritten. Written straight into the blob: each field's bytes
  // once (rows < n from the tables, the rest the engine's fill values), the 8-byte padding after each field zeroed -> the
  // same bytes as per-field vectors + memcpy into a zeroed blob, without the temporaries (si alone is R x 512 B) and the
  // zero-then-overwrite pass. (A reused out->blob of the same size is not re-zeroed: every byte is written here.)
  const InLayout L = in_layout(R);
  out->blob.resize(L.bytes);
  uint8_t* const B = out->blob.data();
  int64_t end = 0;                              // zero every gap: before each field (none today) and its 8-byte padding
  for (int f = 0; f < IN_NFIELDS; ++f) {
    if (L.off[f] > end) std::memset(B + end, 0, (size_t)(L.off[f] - end));
    end = L.off[f] + R * kInCols[f] * kInElem[f];
  }
  if (L.bytes > end) std::memset(B + end, 0, (size_t)(L.bytes - end));
  auto put64 = [&](int f, int64_t i, int64_t v) { std::memcpy(B + L.off[f] + i * 8, &v, 8); };
  const int64_t repl = snap.hsd ? 0 : 1000;   // decode rows: -1 = the engine's device next ids (HSD); else a fixed id
  for (int64_t i = 0; i < n; ++i) {
    const int64_t t = tab.tokens[i];
    put64(IN_POS, i, tab.positions[i]);
    put64(IN_TOK, i, t < 0 ? repl : t);
    put64(IN_OWN, i, tab.own_slot[i]);
  }
  for (int64_t i = n; i < R; ++i) put64(IN_POS, i, 0), put64(IN_TOK, i, 0), put64(IN_OWN, i, snap.scratch);
  for (int64_t j = 0; j < ns; ++j) put64(IN_SAMP, j, tab.sample_rows[j]);
  if (R > ns) std::memset(B + L.off[IN_SAMP] + ns * 8, 0, (size_t)((R - ns) * 8));
  // si / sl are copied raw (the original converted element-wise): pin the element types the blob expects
  static_assert(std::is_same<decltype(RankTables::si)::value_type, int32_t>::value && kInElem[IN_SI] == 4,
                "build_inputs: RankTables::si must be int32 (raw copy into the blob)");
  static_assert(std::is_same<decltype(RankTables::sl)::value_type, int32_t>::value && kInElem[IN_SL] == 4,
                "build_inputs: RankTables::sl must be int32 (raw copy into the blob)");
  uint8_t* const si = B + L.off[IN_SI];
  if (n) std::memcpy(si, tab.si.data(), (size_t)(n * W * 4));
  if (R > n) {                                  // rows >= n: [scratch, -1 x (W - 1)]
    int32_t row[kSlidingWindow];
    row[0] = (int32_t)snap.scratch;
    for (int k = 1; k < W; ++k) row[k] = -1;
    for (int64_t i = n; i < R; ++i) std::memcpy(si + i * W * 4, row, sizeof(row));
  }
  uint8_t* const sl = B + L.off[IN_SL];
  if (n) std::memcpy(sl, tab.sl.data(), (size_t)(n * 4));
  for (int64_t i = n; i < R; ++i) {
    const int32_t one = 1;
    std::memcpy(sl + i * 4, &one, 4);
  }
  uint8_t* const valid = B + L.off[IN_VALID];
  if (n) std::memset(valid, 1, (size_t)n);
  if (R > n) std::memset(valid + n, 0, (size_t)(R - n));
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
