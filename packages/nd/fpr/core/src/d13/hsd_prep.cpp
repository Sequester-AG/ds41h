// hsd_prep.cpp -- see hsd_prep.h. HSD._build statement by statement (the numpy expression in the comments).
#include "hsd_prep.h"

#include <algorithm>
#include <cstring>
#include <set>

#include "d13_host.h"

namespace fpr {
namespace d13 {

namespace {
constexpr int64_t NBW = 6;   // hsd.NBW: logical window blocks per drafting sequence
struct Spec { const char* name; int kind; char dim; int64_t cols; };
const Spec SPEC[] = {   // HSD._SPEC (dims: W = WD, T, M = MR, D, N = NT)
    {"cs", HsdPrep::I32, 'W', 1},       {"ce", HsdPrep::I32, 'W', 1},      {"req_idx", HsdPrep::I32, 'W', 1},
    {"g_dslot", HsdPrep::I64, 'W', 1},  {"g_p0", HsdPrep::I64, 'W', 1},    {"row_req", HsdPrep::I32, 'T', 1},
    {"row_local", HsdPrep::I32, 'T', 1}, {"pos", HsdPrep::I64, 'T', 1},    {"tok_slot", HsdPrep::I64, 'T', 1},
    {"tok_j", HsdPrep::I64, 'T', 1},    {"temp", HsdPrep::F32, 'M', 1},    {"seeds", HsdPrep::I64, 'M', 1},
    // ---- rank-specific
    {"wbase", HsdPrep::I64, 'D', 1},    {"wpages", HsdPrep::I64, 'D', NBW}, {"rstart", HsdPrep::I64, 'D', 1},
    {"lslot", HsdPrep::I64, 'D', 1},    {"seq_of_d", HsdPrep::I64, 'D', 1}, {"tpos", HsdPrep::I64, 'N', 1},
    {"tslot", HsdPrep::I64, 'N', 1}};
enum { CS = 0, CE, REQ_IDX, G_DSLOT, G_P0, ROW_REQ, ROW_LOCAL, POS, TOK_SLOT, TOK_J, TEMP, SEEDS, WBASE, WPAGES, RSTART,
       LSLOT, SEQ_OF_D, TPOS, TSLOT };
const char* SHARED_LAST = "seeds";

bool bucket(int64_t n, const std::vector<int64_t>& bs, int64_t* out) {   // hsd._bucket
  for (auto b : bs)
    if (n <= b) { *out = b; return true; }
  return false;
}

// lst[i] with Python's negative wrap; false = IndexError
template <class V>
bool pyget(const V& v, int64_t i, int64_t* out) {
  const int64_t n = (int64_t)v.size();
  if (i < 0) i += n;
  if (i < 0 || i >= n) return false;
  *out = (int64_t)v[i];
  return true;
}
}  // namespace

HsdPrep::HsdPrep(const HsdConfig& c) : c_(c) {
  int64_t off = 0;
  for (const auto& s : SPEC) {
    Field f{s.name, s.kind, s.dim, s.cols, off, dim(s.dim) * s.cols};
    const int64_t isz = s.kind == I64 ? 8 : 4;
    off += -fdiv(-(f.count * isz), 64) * 64;   // -(-n * itemsize // 64) * 64
    lay_.push_back(f);
    if (f.name == SHARED_LAST) shared_end_ = off;
  }
  nbytes_ = off;
  for (int i = 0; i < 2; ++i)
    for (int r = 0; r < W; ++r) {
      own_[i][r].assign(nbytes_, 0);
      set_[i][r] = own_[i][r].data();
      float* t = f<float>(set_[i][r], TEMP);   // _init_tables: self._hv[i][r]["temp"][:] = 1.0
      for (int64_t j = 0; j < lay_[TEMP].count; ++j) t[j] = 1.0f;
    }
  temp_.assign(MR(), 1.0f);
  seed_.assign(MR(), 0);
  key_.assign(MR(), 0);
  key_set_.assign(MR(), false);
}

int64_t HsdPrep::dim(char d) const {
  const int64_t S_max = c_.row_buckets.empty() ? 0 : c_.row_buckets.back();
  const int64_t D_max = c_.seq_buckets.empty() ? 0 : c_.seq_buckets.back();
  switch (d) {
    case 'T': return W * S_max;
    case 'W': return W * D_max;
    case 'D': return D_max;
    case 'N': return c_.nt_max;
    case 'M': return MR();
  }
  return 0;
}

void HsdPrep::bind(int set, int rank, uint8_t* p) {
  std::memcpy(p, set_[set][rank], nbytes_);
  set_[set][rank] = p;
}

void HsdPrep::restore(const std::vector<float>& temp, const std::vector<int64_t>& seed, const std::vector<uint64_t>& keys) {
  for (size_t g = 0; g < temp_.size(); ++g) {
    if (g < temp.size()) temp_[g] = temp[g];
    if (g < seed.size()) seed_[g] = seed[g];
    if (g < keys.size()) {
      key_[g] = keys[g];
      key_set_[g] = keys[g] != 0;
    }
  }
}

int HsdPrep::fidx(const char* name) const {
  for (size_t i = 0; i < lay_.size(); ++i)
    if (lay_[i].name == name) return (int)i;
  return -1;
}

std::string HsdPrep::where(int64_t b) const {
  for (const auto& f : lay_) {
    const int64_t isz = f.kind == I64 ? 8 : 4;
    if (b >= f.off && b < f.off + f.count * isz) {
      const int64_t e = (b - f.off) / isz;
      return f.name + "[" + (f.cols > 1 ? std::to_string(e / f.cols) + "][" + std::to_string(e % f.cols)
                                          : std::to_string(e)) + "]";
    }
  }
  return "padding@" + std::to_string(b);
}

std::vector<int64_t> HsdPrep::tail_buckets(int64_t S_pad) const {
  const int64_t mt = c_.nt_max;   // sorted({min(S_pad + 128 * i, mt) for i in range(0, mt // 128 + 2)})
  std::set<int64_t> s;
  for (int64_t i = 0; i < fdiv(mt, 128) + 2; ++i) s.insert(std::min(S_pad + 128 * i, mt));
  return std::vector<int64_t>(s.begin(), s.end());
}

Status HsdPrep::build(const std::vector<StepBatch>& B, const std::vector<std::vector<HsdState>>& states,
                      const std::vector<HsdBitmask>* bitmasks, HsdStep* cur, std::string* err) {
  *cur = HsdStep();
  const int64_t k = c_.k, MS1 = c_.max_seqs + 1, MS = c_.max_seqs;
  auto fail = [&](const std::string& m) { if (err) *err = m; return Status::BAD_ARG; };
  struct Per { std::vector<int64_t> sr, ss, first; bool all_rows; };
  std::vector<Per> per;
  for (const auto& b : B) {
    Per p;
    const auto& srl = b.sample_rows;
    const int64_t n = (int64_t)srl.size();
    // all_rows = n == len(b.positions) and (n == 0 or (srl[0] == 0 and srl[-1] == n - 1))
    p.all_rows = n == (int64_t)b.positions.size() && (n == 0 || (srl[0] == 0 && srl[n - 1] == n - 1));
    if (p.all_rows) for (int64_t i = 0; i < n; ++i) p.sr.push_back(i);
    else p.sr.assign(srl.begin(), srl.end());
    if (n) {   // ss = seq_slot if all_rows else _gather(seq_slot, srl)
      if (p.all_rows) p.ss.assign(b.seq_slot.begin(), b.seq_slot.end());
      else
        for (auto i : srl) {
          int64_t v;
          if (!pyget(b.seq_slot, i, &v)) return fail("sample row out of range");
          p.ss.push_back(v);
        }
    }
    if (n > 1) {
      bool bad = false;
      if (!p.all_rows)
        for (size_t i = 1; i < p.sr.size() && !bad; ++i) bad = p.sr[i] <= p.sr[i - 1];
      for (size_t i = 1; i < p.ss.size() && !bad; ++i) bad = p.ss[i] < p.ss[i - 1];
      if (bad) { cur->v0 = true; return Status::OK; }   // return None: the per-element version
    }
    for (size_t i = 0; i < p.ss.size(); ++i)   // first = np.flatnonzero(np.diff(ss, prepend=-1))
      if (p.ss[i] != (i ? p.ss[i - 1] : -1)) p.first.push_back((int64_t)i);
    per.push_back(std::move(p));
  }
  const int nr = (int)B.size();
  int64_t mx_s = 1, mx_d = 1, mx_t = 1;
  for (int r = 0; r < nr; ++r) {
    cur->n_s.push_back((int64_t)per[r].sr.size());
    mx_s = std::max<int64_t>(mx_s, per[r].sr.size());
    mx_d = std::max<int64_t>(mx_d, per[r].first.size());
    cur->n_t.push_back((int64_t)B[r].tail_rows.size());
    mx_t = std::max<int64_t>(mx_t, B[r].tail_rows.size());
  }
  int64_t S_pad, D_pad, NT;
  if (!bucket(mx_s, c_.row_buckets, &S_pad)) return fail(std::to_string(mx_s) + " exceeds the largest bucket");
  if (!bucket(mx_d, c_.seq_buckets, &D_pad)) return fail(std::to_string(mx_d) + " exceeds the largest bucket");
  if (!bucket(mx_t, tail_buckets(S_pad), &NT)) return fail(std::to_string(mx_t) + " exceeds the largest bucket");
  const int64_t T = W * S_pad, WD = W * D_pad;
  const int hi = hi_;
  uint8_t* h0 = set_[hi][0];
  int32_t *cs = f<int32_t>(h0, CS), *ce = f<int32_t>(h0, CE), *req_idx = f<int32_t>(h0, REQ_IDX);
  int64_t *g_dslot = f<int64_t>(h0, G_DSLOT), *g_p0 = f<int64_t>(h0, G_P0);
  int32_t *row_req = f<int32_t>(h0, ROW_REQ), *row_local = f<int32_t>(h0, ROW_LOCAL);
  int64_t *pos = f<int64_t>(h0, POS), *tok_slot = f<int64_t>(h0, TOK_SLOT), *tok_j = f<int64_t>(h0, TOK_J);
  cur->rids.assign(WD, "");
  cur->dseqs.assign(nr, {});
  for (int r = 0; r < nr; ++r) {
    const StepBatch& b = B[r];
    const Per& p = per[r];
    const int64_t n = (int64_t)p.sr.size(), nd = (int64_t)p.first.size();
    const int64_t sink = r * MS1 + MS, q0 = r * D_pad, r0 = r * S_pad;
    const auto& seqs = b.seqs;
    std::vector<int64_t> dsi(nd), cnt(nd), p0(nd), wb(nd), qmax(nd);
    for (int64_t j = 0; j < nd; ++j) dsi[j] = p.ss[p.first[j]];
    for (int64_t j = 0; j < nd; ++j) cnt[j] = (j + 1 < nd ? p.first[j + 1] : n) - p.first[j];   // diff(append(first, n))
    std::vector<int64_t> pos_s;   // positions if all_rows else _gather(positions, sample_rows)
    if (n) {
      if (p.all_rows) pos_s.assign(b.positions.begin(), b.positions.end());
      else
        for (auto i : b.sample_rows) {
          int64_t v;
          if (!pyget(b.positions, i, &v)) return fail("sample row out of range");
          pos_s.push_back(v);
        }
    }
    for (int64_t j = 0; j < nd; ++j) {
      p0[j] = pos_s[p.first[j]];
      wb[j] = fdiv(std::max<int64_t>(p0[j] + 1 - c_.window, 0), 32);   // np.maximum(p0 + 1 - WINDOW, 0) // 32
      qmax[j] = p0[j] + cnt[j] + k - 1;
    }
    std::vector<int64_t> slots(nd, 0), lim(nd, k), rst(nd, 0), wrow;
    std::vector<bool> valid(nd, false);
    for (int64_t j = 0; j < nd; ++j) {   // the one per-sequence loop
      int64_t si = dsi[j];
      if (si < 0) si += (int64_t)seqs.size();
      if (si < 0 || si >= (int64_t)seqs.size()) return fail("seq index out of range");
      const SeqEntry& se = seqs[si];
      const int64_t sl = se.state_slot;
      if (sl >= MS) return fail("state_slot >= HSD max_seqs");
      const int64_t g = r * MS1 + sl;
      slots[j] = g;
      cur->rids[q0 + j] = se.rid;
      if (r >= (int)states.size() || si >= (int64_t)states[r].size()) return fail(se.rid + ": no sampling state");
      const HsdState& st = states[r][si];
      valid[j] = se.kind == "decode" && st.drafts_valid;
      if (st.has_num_acceptable) lim[j] = st.num_acceptable;
      if (!(key_set_[g] && st.ident != 0 && key_[g] == st.ident)) {   // `skey[g] is not st_`
        key_[g] = st.ident;
        key_set_[g] = true;
        temp_[g] = (float)st.temperature;   // float(temperature) -> float32
        seed_[g] = st.seed;
      }
      WindowIndex wg(se.window_table.get());
      const int64_t w0 = wb[j];
      for (int64_t q = 0; q < NBW; ++q) wrow.push_back(wg.get_or(w0 + q, -1));
      const int64_t rs_ = se.replay_start;
      rst[j] = rs_;
      if (c_.pool) {
        const int64_t lo = rs_ > 0 ? rs_ : 0;
        for (int64_t p_ : {p0[j], qmax[j]})
          if (p_ >= lo && wg.get_or(fdiv(p_, 32), -1) < 0)
            return fail(se.rid + ": window page for position " + std::to_string(p_) +
                        " missing (allocate the window to start + 2k + 1)");
      }
    }
    for (int64_t j = 0; j < nd; ++j) cur->dseqs[r].push_back({dsi[j], q0 + j});
    // requests (drafting sequences) of rank r; dummies: empty span at the rank's first row, sink slot
    for (int64_t j = 0; j < nd; ++j) {
      cs[q0 + j] = (int32_t)(r0 + p.first[j]);
      ce[q0 + j] = (int32_t)(r0 + p.first[j] + cnt[j]);
      req_idx[q0 + j] = (int32_t)slots[j];
      g_dslot[q0 + j] = slots[j];
      g_p0[q0 + j] = p0[j];
    }
    for (int64_t q = q0 + nd; q < q0 + D_pad; ++q) {
      cs[q] = (int32_t)r0;
      ce[q] = (int32_t)r0;
      req_idx[q] = (int32_t)sink;
      g_dslot[q] = -1;
      g_p0[q] = 0;
    }
    // verify rows of rank r: [r0, r0 + n) real, [r0 + n, r0 + S_pad) pad (sink)
    int64_t covered = 0;   // np.repeat(gs, cnt) must cover the n rows (else numpy's broadcast fails)
    for (int64_t j = 0; j < nd; ++j) covered += cnt[j];
    if (covered != n || (nd && p.first[0] != 0)) return fail("sample rows do not start a sequence");
    for (int64_t j = 0; j < nd; ++j)
      for (int64_t i = p.first[j]; i < p.first[j] + cnt[j]; ++i) {
        const int64_t jj = i - p.first[j];
        const bool ok = jj == 0 || (valid[j] && jj <= lim[j]);
        row_req[r0 + i] = (int32_t)slots[j];
        row_local[r0 + i] = (int32_t)jj;
        pos[r0 + i] = pos_s[i];
        tok_slot[r0 + i] = slots[j];
        tok_j[r0 + i] = ok ? jj : -1;
      }
    for (int64_t i = r0 + n; i < r0 + S_pad; ++i) {
      row_req[i] = (int32_t)sink;
      row_local[i] = 0;
      pos[i] = 0;
      tok_slot[i] = 0;
      tok_j[i] = -1;
    }
    // rank-specific: drafting tables (D_pad) and tail tables (NT)
    uint8_t* h = set_[hi][r];
    int64_t* wp = f<int64_t>(h, WPAGES);
    for (int64_t q = 0; q < nd * NBW; ++q) wp[q] = wrow[q];
    for (int64_t q = nd * NBW; q < D_pad * NBW; ++q) wp[q] = -1;
    int64_t *wbase = f<int64_t>(h, WBASE), *rstart = f<int64_t>(h, RSTART), *lslot = f<int64_t>(h, LSLOT),
            *seq_of_d = f<int64_t>(h, SEQ_OF_D), *tpos = f<int64_t>(h, TPOS), *tslot = f<int64_t>(h, TSLOT);
    for (int64_t j = 0; j < nd; ++j) {
      wbase[j] = wb[j];
      rstart[j] = rst[j];
      lslot[j] = slots[j] - r * MS1;
      seq_of_d[j] = dsi[j];
    }
    for (int64_t j = nd; j < D_pad; ++j) {
      wbase[j] = 0;
      rstart[j] = 0;
      lslot[j] = MS;
      seq_of_d[j] = MS;
    }
    const int64_t nt = cur->n_t[r];
    if (nt) {
      const auto& trl = b.tail_rows;
      const bool t_all = nt == (int64_t)b.positions.size() && trl[0] == 0 && trl[nt - 1] == nt - 1;
      std::vector<int64_t> tp;
      if (t_all && p.all_rows) tp = pos_s;
      else if (t_all) tp.assign(b.positions.begin(), b.positions.end());
      else
        for (auto i : trl) {
          int64_t v;
          if (!pyget(b.positions, i, &v)) return fail("tail row out of range");
          tp.push_back(v);
        }
      if ((int64_t)tp.size() != nt) return fail("tail positions length");
      for (int64_t i = 0; i < nt; ++i) tpos[i] = tp[i];
      if (c_.pool) {
        std::vector<int64_t> ws;
        if (t_all) ws.assign(b.window_slot.begin(), b.window_slot.end());
        else
          for (auto i : trl) {
            int64_t v;
            if (!pyget(b.window_slot, i, &v)) return fail("tail row out of range");
            ws.push_back(v);
          }
        if ((int64_t)ws.size() != nt) return fail("tail window slots length");
        for (int64_t i = 0; i < nt; ++i) tslot[i] = ws[i];
      } else {   // private ring: ssl[tss] * RING + tp % RING (ssl = the seqs' state slots, numpy index wraps)
        const int64_t S = (int64_t)seqs.size();
        for (int64_t i = 0; i < nt; ++i) {
          int64_t s;
          if (!pyget(b.seq_slot, t_all ? i : (int64_t)trl[i], &s)) return fail("tail row out of range");
          if (s < 0) s += S;
          if (s < 0 || s >= S) return fail("tail seq out of range");
          tslot[i] = (int64_t)seqs[s].state_slot * c_.ring + fmod(tp[i], c_.ring);
        }
      }
    }
    for (int64_t i = nt; i < NT; ++i) {
      tpos[i] = 0;
      tslot[i] = -1;
    }
  }
  std::memcpy(f<float>(h0, TEMP), temp_.data(), sizeof(float) * temp_.size());     // h0["temp"][:] = tnp
  std::memcpy(f<int64_t>(h0, SEEDS), seed_.data(), sizeof(int64_t) * seed_.size());   // h0["seeds"][:] = snp
  for (int r = 1; r < W; ++r) std::memcpy(set_[hi][r], h0, shared_end_);   // the shared section on every rank
  cur->hi = hi;
  cur->S_pad = S_pad;
  cur->D_pad = D_pad;
  cur->NT = NT;
  cur->n_req = WD;
  cur->T = T;
  for (const auto& b : B) {
    cur->n_seq.push_back((int64_t)b.seqs.size());
    cur->step.push_back(b.step);
  }
  bool has_bm = false;
  if (bitmasks)
    for (const auto& x : *bitmasks) has_bm = has_bm || x.rows != nullptr;
  cur->has_bm = has_bm;
  if (has_bm) {
    if (!bm_) {   // torch.empty: only [:T] is ever read
      bm_own_.assign((size_t)dim('T') * BM_WORDS, 0);
      bm_ = bm_own_.data();
    }
    for (size_t r2 = 0; r2 < bitmasks->size(); ++r2) {
      const HsdBitmask& x = (*bitmasks)[r2];
      const int64_t ns = x.rows ? x.n_s : 0;
      if (ns) {
        if (x.words != BM_WORDS) return fail("bitmask words != ceil(V / 32)");
        std::memcpy(bm_ + (size_t)(r2 * S_pad) * BM_WORDS, x.rows, sizeof(int32_t) * ns * BM_WORDS);
      }
      for (int64_t i = r2 * S_pad + ns; i < (int64_t)(r2 + 1) * S_pad; ++i)
        std::fill(bm_ + i * BM_WORDS, bm_ + (i + 1) * BM_WORDS, -1);   // all allowed
    }
  }
  return Status::OK;
}

}  // namespace d13
}  // namespace fpr
