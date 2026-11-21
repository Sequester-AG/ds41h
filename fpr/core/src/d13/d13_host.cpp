// d13_host.cpp -- see d13_host.h. Every function names the Python it mirrors; comments
// quote the Python expression where the translation is not obvious.
#include "d13_host.h"

#include <algorithm>

namespace fpr {
namespace d13 {

// ------------------------------------------------------------------------------------------ WindowIndex
WindowIndex::WindowIndex(const WindowTable* wt) : wt_(wt) {
  if (wt_ && wt_->size() > 16)
    for (const auto& kv : wt_->items) idx_.emplace(kv.first, kv.second);
}

bool WindowIndex::get(int64_t blk, int64_t* page) const {
  if (!wt_) return false;
  if (!idx_.empty()) {
    auto it = idx_.find(blk);
    if (it == idx_.end()) return false;
    *page = it->second;
    return true;
  }
  for (const auto& kv : wt_->items)
    if (kv.first == blk) { *page = kv.second; return true; }
  return false;
}

static int64_t n_blocks_of(const SeqEntry& se) {   // se.n_blocks or len(se.block_table)
  if (se.n_blocks) return se.n_blocks;
  return se.block_table ? (int64_t)se.block_table->size() : 0;
}

static int64_t sum_rows(const CompactTables& ct) {   // CompactTables.n (property: n_rows.sum())
  int64_t n = 0;
  for (auto x : ct.n_rows) n += x;
  return n;
}

// ------------------------------------------------------------------------------------------ _async_ok
std::string D13Host::async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& cts) const {
  if (!cfg_.graphs) return "graphs off";
  int64_t mx = 0;
  const size_t nr = std::min(b.size(), cts.size());   // zip(batches, cts)
  for (size_t r = 0; r < nr; ++r) {
    const StepBatch& sb = b[r];
    const CompactTables& ct = cts[r];
    if (!sb.copies.empty()) return "page copies";
    if (sb.n_rows > cfg_.largest_bucket) return "rows > largest bucket";
    const size_t S = sb.seqs.size();
    if (!S) continue;
    const size_t s1 = std::min(S, ct.kind.size());
    bool bad = false;   // np.any(ct.kind[:S] != 0) or np.any(ct.forced[:S] != 0) or np.any(ct.score[:S] != 0)
    for (size_t i = 0; i < s1 && !bad; ++i) bad = ct.kind[i] != 0;
    for (size_t i = 0; i < std::min(S, ct.forced.size()) && !bad; ++i) bad = ct.forced[i] != 0;
    for (size_t i = 0; i < std::min(S, ct.score.size()) && !bad; ++i) bad = ct.score[i] != 0;
    if (bad) return "prefill / forced / score rows";
    for (size_t i = 0; i < std::min(S, ct.n_rows.size()) && !bad; ++i) bad = ct.n_rows[i] != cfg_.k1;
    for (size_t i = 0; i < std::min(S, ct.row0.size()) && !bad; ++i) bad = ct.row0[i] != cfg_.k1 * (int64_t)i;
    if (bad) return "decode layout";
    for (size_t i = 0; i < std::min(S, ct.start.size()); ++i) {   // zip(sb.seqs, ct.start[:S])
      const int64_t st = ct.start[i];
      mx = std::max(mx, (st >= 0 ? st : sb.seqs[i].known + cfg_.k1) + cfg_.k1);
    }
  }
  if (mx > cfg_.graph_max_ctx) return "context beyond the decode graphs' indexer bound";
  return "";
}

// ------------------------------------------------------------------------------------------ _deltas
void D13Host::deltas(int rank, const CompactTables& ct, Deltas* o, MirrorFn mirror, void* user) const {
  *o = Deltas();
  if (ct.bt_upd_slot.empty()) return;
  o->any = true;
  int64_t nu = 0;
  for (auto l : ct.bt_upd_len) nu += l;
  o->nu = nu;
  if (!nu) {
    if (mirror) {   // zero-length deltas: a new sequence with no pages yet in that slot = a reset for the mirror
      MirrorCall c{rank, {}, {}, {}, ct.bt_upd_slot, ct.bt_upd_start};
      mirror(user, c);
    }
    return;
  }
  o->rows.reserve(nu);
  o->cols.reserve(nu);
  for (size_t i = 0; i < ct.bt_upd_slot.size() && i < ct.bt_upd_len.size(); ++i)
    for (int64_t j = 0; j < ct.bt_upd_len[i]; ++j) {   // np.repeat(slot, ln); (arange(nu) - repeat(cumsum - ln)) + repeat(start)
      o->rows.push_back(ct.bt_upd_slot[i]);
      o->cols.push_back(ct.bt_upd_start[i] + j);
    }
  const int64_t np_ = std::min<int64_t>(nu, (int64_t)ct.bt_upd_pages.size());   // bt_upd_pages[:nu]
  o->pages.resize(np_);
  for (int64_t i = 0; i < np_; ++i) o->pages[i] = (int32_t)ct.bt_upd_pages[i];
  if (mirror) {
    MirrorCall c{rank, o->rows, o->cols, o->pages, ct.bt_upd_slot, ct.bt_upd_start};
    mirror(user, c);
  }
  o->direct = nu > cfg_.delta_cap;
}

// ------------------------------------------------------------------------------------------ RowExp.desc_of_compact
void D13Host::desc_of_compact(const CompactTables& ct, Desc* d) {
  *d = Desc();
  const size_t S = ct.slot.size();
  d->S = (int64_t)S;
  d->rows = sum_rows(ct);
  auto i32 = [](const std::vector<int64_t>& v, std::vector<int32_t>* o) {
    o->resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) (*o)[i] = (int32_t)v[i];
  };
  i32(ct.slot, &d->slot);
  i32(ct.row0, &d->row0);
  i32(ct.n_rows, &d->nrows);
  i32(ct.kind, &d->kind);
  d->start = ct.start;
  d->replay = ct.replay_start;
  d->cached = ct.cached_upto;
  if (!ct.win_slot.empty()) {   // np.stack([win_slot, win_blk, win_page], 1).astype(np.int32)
    const size_t V = ct.win_slot.size();
    d->ops.resize(3 * V);
    for (size_t i = 0; i < V; ++i) {
      d->ops[3 * i] = (int32_t)ct.win_slot[i];
      d->ops[3 * i + 1] = (int32_t)ct.win_blk[i];
      d->ops[3 * i + 2] = (int32_t)ct.win_page[i];
    }
  }
}

// ------------------------------------------------------------------------------------------ HostDesc.build
Status HostDesc::build(const StepBatch& sb, Desc* d, std::string* err) {
  *d = Desc();
  const size_t S = sb.seqs.size();
  d->S = (int64_t)S;
  d->rows = sb.n_rows;
  d->slot.resize(S); d->row0.resize(S); d->nrows.resize(S); d->kind.resize(S);
  d->start.resize(S); d->replay.resize(S); d->cached.resize(S);
  for (size_t k = 0; k < S; ++k) {
    const SeqEntry& se = sb.seqs[k];
    const WindowTable* wtab = se.window_table.get();
    const int64_t sl = se.state_slot;
    const std::pair<int64_t, int64_t> uid(se.uid, se.inc);   // identity = (uid, inc)
    d->slot[k] = (int32_t)sl;
    d->row0[k] = (int32_t)se.row0;
    d->nrows[k] = (int32_t)se.n_rows;
    d->kind[k] = se.kind == "decode" ? 0 : 1;
    auto ow = owner_.find(sl);
    const bool dev_ok = device_start_ && se.kind == "decode" && ow != owner_.end() && ow->second == uid;
    const int64_t host_start = se.start >= 0 ? se.start : se.known;
    if (!(dev_ok || host_start >= 0)) {
      if (err) *err = se.rid + ": no host start for a sequence new to slot " + std::to_string(sl);
      return Status::BAD_ARG;
    }
    d->start[k] = dev_ok ? -1 : host_start;
    d->replay[k] = se.replay_start;
    d->cached[k] = se.cached_upto;
    owner_[sl] = uid;
    auto pv = sent_.find(sl);
    const int64_t nb = n_blocks_of(se);
    const int64_t wver = se.wver;
    std::map<int64_t, int64_t> have;
    std::optional<int64_t> last_wver;
    if (pv == sent_.end() || pv->second.uid != uid || pv->second.nb > nb) {   // ring clear (D12 bt_reset trigger)
      d->ops.insert(d->ops.end(), {(int32_t)sl, -1, -1});
    } else {
      have = std::move(pv->second.have);
      last_wver = pv->second.wver;
    }
    if (wver < 0 || !last_wver || wver != *last_wver) {   // the window table changed since the last diff
      if (wtab)
        for (const auto& bp : wtab->items) {
          auto h = have.find(bp.first);
          if (h == have.end() || h->second != bp.second) {
            d->ops.insert(d->ops.end(), {(int32_t)sl, (int32_t)bp.first, (int32_t)bp.second});
            have[bp.first] = bp.second;
          }
        }
    }
    const size_t nwt = wtab ? wtab->size() : 0;
    if (have.size() > 2 * nwt + 16) {   // forget released blocks (host memory only)
      WindowIndex wi(wtab);
      for (auto it = have.begin(); it != have.end();) {
        int64_t p;
        if (!wi.get(it->first, &p)) it = have.erase(it);
        else ++it;
      }
    }
    Sent s;
    s.uid = uid; s.have = std::move(have); s.nb = nb; s.wver = wver;
    sent_[sl] = std::move(s);   // (Python stores the tuple before the diff; `have` is shared by reference: same result)
  }
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ RowExp.upload (host half)
DescStaging::DescStaging(int64_t max_seqs, int64_t max_ops) : S_(max_seqs), O_(max_ops) {
  for (int i = 0; i < 2; ++i) {
    h32_[i].assign(n32_total(), 0);
    h64_[i].assign(n64_total(), 0);
  }
}

Status DescStaging::pack(const Desc& d, int64_t max_rows, int64_t* n32, std::string* err) {
  if (!(d.S <= S_ && d.rows <= max_rows && d.n_ops() <= O_)) return pack_into(d, S_, O_, max_rows, nullptr, nullptr, n32, err);
  const int i = hi_;
  hi_ ^= 1;
  last_ = i;
  return pack_into(d, S_, O_, max_rows, h32_[i].data(), h64_[i].data(), n32, err);
}

Status DescStaging::pack_into(const Desc& d, int64_t S_max, int64_t O_max, int64_t max_rows, int32_t* h32, int64_t* h64,
                              int64_t* n32, std::string* err) {
  const int64_t S = d.S, n_ops = d.n_ops();
  if (!(S <= S_max && d.rows <= max_rows && n_ops <= O_max)) {
    if (err) *err = "descriptor too large: S " + std::to_string(S) + " rows " + std::to_string(d.rows) + " ops " +
                    std::to_string(n_ops);
    return Status::BAD_ARG;
  }
  const int64_t S_ = S_max;
  const int64_t o_slot = 0, o_row0 = S_, o_nrows = 2 * S_, o_kind = 3 * S_, o_counts = 4 * S_, o_ops = 4 * S_ + 4;
  for (int64_t k = 0; k < S; ++k) {
    h32[o_slot + k] = d.slot[k];
    h32[o_row0 + k] = d.row0[k];
    h32[o_nrows + k] = d.nrows[k];
    h32[o_kind + k] = d.kind[k];
  }
  h32[o_counts] = (int32_t)S;
  h32[o_counts + 1] = (int32_t)d.rows;
  h32[o_counts + 2] = (int32_t)n_ops;
  for (int64_t k = 0; k < 3 * n_ops; ++k) h32[o_ops + k] = d.ops[k];
  for (int64_t k = 0; k < S; ++k) {
    h64[k] = d.start[k];
    h64[S_ + k] = d.replay[k];
    h64[2 * S_ + k] = d.cached[k];
  }
  *n32 = o_ops + 3 * n_ops;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ owner / empty / guard
bool OwnerSig::operator==(const OwnerSig& o) const {
  if (n != o.n || sample_idx != o.sample_idx || seqs.size() != o.seqs.size()) return false;
  for (size_t i = 0; i < seqs.size(); ++i) {
    const auto &a = seqs[i], &b = o.seqs[i];
    if (a.rid != b.rid || a.state_slot != b.state_slot || a.row0 != b.row0 || a.n_rows != b.n_rows ||
        a.n_blocks != b.n_blocks || a.prev_index != b.prev_index)
      return false;
  }
  return true;
}

bool D13Host::empty_eligible(const StepBatch& sb, const CompactTables& ct) const {
  return sb.seqs.empty() && !sb.n_rows && ct.win_slot.empty() && ct.bt_upd_slot.empty();
}

std::optional<OwnerSig> D13Host::owner_sig(int rank, const StepBatch& sb, const CompactTables& ct,
                                           const std::vector<PrevOf>& prev) const {
  if (sb.seqs.empty() || !ct.win_slot.empty() || !ct.bt_upd_slot.empty()) return std::nullopt;
  const int64_t n = sb.n_rows;
  for (int64_t i = 0; i < n && i < (int64_t)ct.tokens.size(); ++i)   // (np.asarray(ct.tokens[:n]) < 0).all()
    if (ct.tokens[i] >= 0) return std::nullopt;
  OwnerSig sig;
  sig.n = n;
  for (size_t k = 0; k < sb.seqs.size(); ++k) {
    const SeqEntry& se = sb.seqs[k];
    if (se.kind != "decode" || se.n_rows != cfg_.k1) return std::nullopt;
    const PrevOf* p = nullptr;
    for (const auto& x : prev)
      if (x.rid == se.rid) { p = &x; break; }
    if (!p || !p->has || p->rank != rank) return std::nullopt;
    sig.seqs.push_back({se.rid, se.state_slot, se.row0, se.n_rows, n_blocks_of(se), p->index});
  }
  sig.sample_idx = ct.sample_idx;   // tuple(np.asarray(ct.sample_idx).tolist())
  return sig;
}

std::vector<int64_t> D13Host::guard_blocks(const StepBatch& sb) const {
  std::vector<int64_t> o;
  for (const auto& se : sb.seqs) o.push_back(n_blocks_of(se));
  return o;
}

// ------------------------------------------------------------------------------------------ _positions
void D13Host::positions(const StepBatch& sb, const std::vector<int64_t>& pos, const std::vector<int64_t>& own,
                        StepBatch* out) const {
  *out = sb;
  const int64_t n = sb.n_rows;
  if (!n) return;   // `... if n else sb`
  out->positions.assign(pos.begin(), pos.begin() + std::min<int64_t>(n, (int64_t)pos.size()));      // ph[:n].tolist()
  out->window_slot.assign(own.begin(), own.begin() + std::min<int64_t>(n, (int64_t)own.size()));    // oh[:n].tolist()
}

// ------------------------------------------------------------------------------------------ _fallback
Status D13Host::fallback_resolve(const std::vector<StepBatch>& b, const std::vector<std::vector<int64_t>>& starts,
                                 std::vector<StepBatch>* rb, std::string* err) const {
  rb->clear();
  for (size_t r = 0; r < b.size(); ++r) {
    const StepBatch& sb = b[r];
    const std::vector<int64_t> none;
    const std::vector<int64_t>& st = r < starts.size() ? starts[r] : none;
    const int64_t scratch = r < (size_t)W ? cfg_.scratch[r] : 0;
    const bool reuse = cfg_.fb_reuse && (int64_t)sb.positions.size() == sb.n_rows;
    StepBatch sb2 = sb;
    sb2.seqs.clear();
    sb2.positions.clear();
    sb2.paged_slot.clear();
    sb2.window_slot.clear();
    const size_t S = std::min(sb.seqs.size(), st.size());   // zip(sb.seqs, starts[r])
    for (size_t k = 0; k < S; ++k) {
      SeqEntry se = sb.seqs[k];
      if (reuse && se.start >= 0) {   // host-known: build_step_batch's rows (FB_REUSE)
        sb2.seqs.push_back(se);
        const int64_t a = se.row0, e = se.row0 + se.n_rows;
        auto sl = [&](const std::vector<int64_t>& v, std::vector<int64_t>* o) {   // python slice v[a:e]
          const int64_t lo = std::min<int64_t>(std::max<int64_t>(a, 0), (int64_t)v.size());
          const int64_t hi = std::min<int64_t>(std::max<int64_t>(e, 0), (int64_t)v.size());
          for (int64_t i = lo; i < hi; ++i) o->push_back(v[i]);
        };
        sl(sb.positions, &sb2.positions);
        sl(sb.paged_slot, &sb2.paged_slot);
        sl(sb.window_slot, &sb2.window_slot);
        continue;
      }
      const std::vector<int64_t> empty;
      const std::vector<int64_t>& pg = se.block_table ? *se.block_table : empty;
      if (se.start < 0) {   // replace(se, start=s0, num_computed=s0, n_blocks=min(-(-(s0 + n_rows) // 64), len(bt)))
        const int64_t s0 = st[k];
        se.start = s0;
        se.num_computed = s0;
        se.n_blocks = std::min<int64_t>(-fdiv(-(s0 + se.n_rows), 64), (int64_t)pg.size());
      }
      sb2.seqs.push_back(se);
      const int64_t cu = se.cached_upto;
      const int64_t nbk = se.n_blocks ? se.n_blocks : (int64_t)pg.size();
      const int64_t lim = std::min<int64_t>(nbk, (int64_t)pg.size());
      WindowIndex wi(se.window_table.get());
      for (int64_t p = se.start; p < se.start + se.n_rows; ++p) {
        sb2.positions.push_back(p);
        const int64_t q = fdiv(p, 64);
        if (p < cu || q >= lim) {
          sb2.paged_slot.push_back(PAD);
        } else {
          int64_t qi = q < 0 ? q + (int64_t)pg.size() : q;   // python list index
          if (qi < 0 || qi >= (int64_t)pg.size()) { if (err) *err = "block index out of range"; return Status::BAD_ARG; }
          sb2.paged_slot.push_back(pg[qi] * 64 + fmod(p, 64));
        }
        int64_t w;
        sb2.window_slot.push_back(wi.get(fdiv(p, 32), &w) ? w * 32 + fmod(p, 32) : scratch);
      }
    }
    rb->push_back(std::move(sb2));
  }
  return Status::OK;
}

Status D13Host::fallback_tables(const std::vector<StepBatch>& rb, const std::vector<CompactTables>& ct,
                                std::vector<RankTables>* out, std::string* err) {
  out->assign(rb.size(), RankTables());
  for (size_t r = 0; r < rb.size(); ++r) {   // self.tb.build(sb2, [int(x) for x in ct.tokens[:sb.n_rows]])
    const std::vector<int64_t>& t = ct[r].tokens;
    const std::vector<int64_t> tok(t.begin(), t.begin() + std::min<size_t>(std::max<int32_t>(rb[r].n_rows, 0), t.size()));
    const Status s = tb_.build(rb[r], &tok, &(*out)[r], err);
    if (s != Status::OK) return s;
  }
  return Status::OK;
}

std::vector<int64_t> D13Host::fallback_accepted(const StepBatch& rb, const std::vector<std::pair<Rid, int32_t>>& acc) const {
  std::vector<int64_t> a;
  for (const auto& se : rb.seqs) {
    int64_t v = se.n_rows;
    if (se.kind == "decode")
      for (const auto& x : acc)
        if (x.first == se.rid) { v = x.second; break; }
    a.push_back(v);
  }
  return a;
}

std::vector<int64_t> D13Host::advance_index(int rank, int64_t S, int64_t D_pad,
                                            const std::vector<std::pair<int64_t, int64_t>>& ds, bool* slice) const {
  const int64_t q0 = rank * D_pad;
  bool sl = (int64_t)ds.size() == S;
  for (size_t j = 0; j < ds.size() && sl; ++j) sl = ds[j].first == (int64_t)j && ds[j].second == q0 + (int64_t)j;
  if (slice) *slice = sl;
  int64_t mx = S;
  for (const auto& x : ds) mx = std::max(mx, x.first + 1);
  std::vector<int64_t> q(mx, 0);   // q[:S] = 0; for si, qi in dseqs: q[si] = qi
  for (const auto& x : ds)
    if (x.first >= 0) q[x.first] = x.second;
  q.resize(S);
  return q;
}

bool D13Host::steady_batches(const std::vector<StepBatch>& b, const std::vector<StepBatch>& prev,
                             const std::vector<std::string>& structured) const {
  for (const auto& sb : b)
    for (const auto& se : sb.seqs) {
      if (se.kind != "decode" || se.n_rows != 4) return false;
      if (std::find(structured.begin(), structured.end(), se.rid) != structured.end()) return false;
    }
  if (b.size() != prev.size()) return false;   // signature(batches) != signature(inflight.batches)
  for (size_t r = 0; r < b.size(); ++r) {
    const auto &x = b[r].seqs, &y = prev[r].seqs;
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i)
      if (x[i].rid != y[i].rid || x[i].state_slot != y[i].state_slot || x[i].kind != y[i].kind ||
          x[i].n_rows != y[i].n_rows || x[i].row0 != y[i].row0)
        return false;
  }
  return true;
}

}  // namespace d13
}  // namespace fpr
