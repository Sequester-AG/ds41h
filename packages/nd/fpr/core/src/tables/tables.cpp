// tables.cpp
// See include/tables.h.
#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "../../include/tables.h"

namespace fpr {
namespace {

inline int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; }
inline int64_t pmod(int64_t a, int64_t b) { int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }
inline int64_t cdiv(int64_t a, int64_t b) { return -fdiv(-a, b); }     // -(-a // b)

// Python list slice a[lo:hi] (clamped; negative indices from the end)
inline void py_slice(const std::vector<int64_t>& a, int64_t lo, int64_t hi, std::vector<int64_t>* out) {
  const int64_t n = (int64_t)a.size();
  if (lo < 0) lo = std::max<int64_t>(0, lo + n);
  if (hi < 0) hi = std::max<int64_t>(0, hi + n);
  lo = std::min(lo, n);
  hi = std::min(hi, n);
  for (int64_t i = lo; i < hi; ++i) out->push_back(a[(size_t)i]);
}
// numpy fancy index (wraps negatives; out of range = IndexError)
inline bool np_idx(int64_t i, int64_t n, int64_t* out) {
  if (i < -n || i >= n) return false;
  *out = i < 0 ? i + n : i;
  return true;
}
inline int ratio_of(int src) {
  for (int i = 0; i < 4; ++i) if (KV_SOURCE_LAYERS[i] == src) return KV_SOURCE_RATIO[i];
  return 0;
}
inline Status bad(std::string* err, const std::string& m) { if (err) *err = m; return Status::BAD_ARG; }

}  // namespace

// ------------------------------------------------------------------------------------------------ build_step_batch
Status build_step_batch(const StepPlan& plan, const KVViews& kv, int64_t bt, int64_t wt, int rank, StepBatch* out,
                        std::string* err) {
  StepBatch& sb = *out;
  sb = StepBatch{};
  sb.rank = rank;
  sb.step = plan.step;
  std::vector<const SeqWork*> work;
  for (const SeqWork& w : plan.work) if (w.rank == rank) work.push_back(&w);
  std::stable_sort(work.begin(), work.end(),
                   [](const SeqWork* a, const SeqWork* b) { return (a->kind != "decode") < (b->kind != "decode"); });
  int64_t row = 0;
  for (const SeqWork* wp : work) {
    const SeqWork& w = *wp;
    const bool helper = w.kind == "cp";
    const SeqKV* sk = kv.seq ? kv.seq(w.rid) : nullptr;
    if (!sk) return bad(err, "build_step_batch: no KV sequence " + w.rid);
    SeqEntry se;
    se.rid = helper ? w.cp.rid : w.rid;
    se.kind = helper ? "prefill" : w.kind;
    se.row0 = (int32_t)row;
    se.n_rows = (int32_t)w.n;
    se.start = w.start;
    se.num_computed = w.start;
    se.cached_upto = helper ? 0 : w.cached_upto;
    se.replay_start = w.replay_start;
    se.last_chunk = helper ? false : w.last_chunk;
    se.block_table = sk->pages;
    se.window_table = sk->window;
    se.state_slot = sk->state_slot;
    se.has_cp = w.has_cp;
    if (w.has_cp) se.cp = w.cp;
    const bool dev = w.start < 0;                                   // async decode: positions resolved on the device
    se.n_blocks = dev ? (int64_t)sk->pages->size() : cdiv(w.start + w.n, bt);
    if (w.has_cp) {
      if (w.cp.subs.empty()) return bad(err, "build_step_batch: CP role without subs");
      const auto& last = w.cp.subs.back();
      se.n_blocks = cdiv(std::get<1>(last) + std::get<2>(last), bt);
    }
    se.known = w.known;
    se.images = w.images;
    se.score = w.score;
    se.forced = w.forced;
    se.uid = w.uid;
    se.wver = sk->wver;
    se.inc = sk->inc;
    se.tail_row0 = (int32_t)((w.kind == "decode" || se.score) ? row : row + std::max<int64_t>(0, w.n - SLIDING_WINDOW));
    const int32_t slot = (int32_t)sb.seqs.size();
    const std::vector<int64_t>& pages = *sk->pages;
    const WindowTable& win = *sk->window;
    const int64_t cu = w.cached_upto;
    for (int64_t i = 0; i < w.n; ++i) sb.seq_slot.push_back(slot);
    if (dev) {
      for (int64_t i = 0; i < w.n; ++i) {
        sb.positions.push_back(PAD);
        sb.paged_slot.push_back(PAD);
        sb.window_slot.push_back(PAD);
      }
    } else {
      for (int64_t p = w.start; p < w.start + w.n; ++p) sb.positions.push_back(p);
      for (int64_t p = w.start; p < w.start + w.n; ++p) {
        if (p < cu) { sb.paged_slot.push_back(PAD); continue; }
        int64_t ix;
        if (!np_idx(fdiv(p, bt), (int64_t)pages.size(), &ix)) return bad(err, "build_step_batch: page index out of range");
        sb.paged_slot.push_back(pages[(size_t)ix] * bt + pmod(p, bt));
      }
      for (int64_t p = w.start; p < w.start + w.n; ++p) {
        const int64_t i = win.find(fdiv(p, wt));
        if (i < 0) return bad(err, "build_step_batch: window block missing (KeyError) for " + w.rid);
        sb.window_slot.push_back(win.items[(size_t)i].second * wt + pmod(p, wt));
      }
    }
    if (helper) se.tail_row0 = (int32_t)(row + w.n);              // helpers: layers 0..20 only, nothing sampled
    for (int64_t r = se.tail_row0; r < row + w.n; ++r) sb.tail_rows.push_back((int32_t)r);
    if (helper) {
    } else if (w.kind == "decode" || se.score) {
      for (int64_t r = row; r < row + w.n; ++r) sb.sample_rows.push_back((int32_t)r);
    } else if (w.last_chunk) {
      sb.sample_rows.push_back((int32_t)(row + w.n - 1));
    }
    sb.seqs.push_back(std::move(se));
    row += w.n;
  }
  sb.n_rows = (int32_t)row;
  for (const CopyOp& c : plan.copies)
    if (c.src_rank == rank || c.dst_rank == rank) sb.copies.push_back(c);
  if (kv.take_poison) sb.poison = kv.take_poison(rank);
  return Status::OK;
}

Status build_step_batch(const StepPlan& plan, IKVManager& kv, int rank, StepBatch* out, std::string* err) {
  KVViews v;
  v.seq = [&kv](const Rid& id) -> const SeqKV* { return static_cast<const IKVManager&>(kv).seq(id); };
  v.take_poison = [&kv](int r) { return kv.take_poison(r); };
  return build_step_batch(plan, v, kv.geom().page_tokens, kv.geom().window_page_tokens, rank, out, err);
}

// ------------------------------------------------------------------------------------------------ TableBuilder.build
Status TableBuilder::build(const StepBatch& sb, const std::vector<int64_t>* row_tokens, RankTables* out, std::string* err) {
  RankTables& t = *out;
  t = RankTables{};
  const int64_t n = sb.n_rows;
  const int64_t WT = WINDOW_PAGE_TOKENS, BT = PAGE_TOKENS, SW = SLIDING_WINDOW;
  const size_t S = sb.seqs.size(), NS = sources.size();
  std::vector<int64_t> flat_w;
  std::vector<std::vector<int64_t>> flat_c(NS), coff(NS), clo(NS);
  std::vector<std::array<int64_t, 3>> upd;
  std::vector<int64_t> upd_pages;
  std::vector<std::array<int64_t, 14>> rec;
  std::vector<int64_t> tokens;
  const bool stub = stub_indexer;
  if (!row_tokens && !tokens_of) return bad(err, "TableBuilder.build: no row_tokens and no tokens_of");
  for (size_t k = 0; k < S; ++k) {
    const SeqEntry& se = sb.seqs[k];
    const int64_t first = se.start, nr = se.n_rows, last = first + nr - 1, rs = se.replay_start;
    const int64_t lo_b = fdiv(std::max(rs, first - (SW - 1)), WT);
    const WindowTable& wtab = *se.window_table;
    const int64_t woff = (int64_t)flat_w.size();
    for (int64_t b = lo_b; b < fdiv(last, WT) + 1; ++b) flat_w.push_back(wtab.get(b, -1));
    const std::vector<int64_t>& bt = *se.block_table;
    const int64_t nb = se.n_blocks ? se.n_blocks : (int64_t)bt.size();
    if (stub) {
      for (size_t si = 0; si < NS; ++si) {
        const int64_t c = ratio_of(sources[si]);
        const int64_t nc0 = c > 1 ? fdiv(first + 1, c) : first;
        const int64_t nc1 = c > 1 ? fdiv(last + 1, c) : last;
        const int64_t lo_tok = std::max<int64_t>(0, nc0 - 512) * c;
        const int64_t hi_tok = std::max<int64_t>(0, nc1 - 1) * c;
        coff[si].push_back((int64_t)flat_c[si].size());
        clo[si].push_back(fdiv(lo_tok, BT));
        py_slice(bt, fdiv(lo_tok, BT), fdiv(hi_tok, BT) + 1, &flat_c[si]);
      }
    }
    const auto key = std::make_pair((int32_t)sb.rank, se.state_slot);
    auto it = sent_.find(key);
    const bool reset = it == sent_.end() || it->second.uid != se.uid || it->second.inc != se.inc || it->second.nb > nb;
    const int64_t s0 = reset ? 0 : it->second.nb;
    if (nb > s0 || reset) {
      upd.push_back({(int64_t)k, s0, nb - s0});
      py_slice(bt, s0, nb, &upd_pages);
    }
    sent_[key] = Sent{se.uid, se.inc, nb};
    rec.push_back({se.state_slot, se.uid, first, nr, se.kind == "decode" ? 0 : 1, se.last_chunk, se.score, se.forced, rs,
                   se.cached_upto, nb, reset, woff, lo_b});
    if (!row_tokens)
      for (int64_t p = first; p < last + 1; ++p) tokens.push_back(tokens_of(se.rid, p));
  }
  if (row_tokens) {
    if ((int64_t)row_tokens->size() != n) return bad(err, "TableBuilder.build: len(row_tokens) != n_rows");
    tokens = *row_tokens;
  }
  // tok = np.asarray(tokens if n else [0])[:n]
  std::vector<int64_t> tok(tokens.begin(), tokens.begin() + std::min<int64_t>(n, (int64_t)tokens.size()));
  if ((int64_t)tok.size() != n) return bad(err, "TableBuilder.build: fewer tokens than rows");
  std::vector<int64_t> targets((size_t)n, 0);
  for (int64_t i = 0; i + 1 < n; ++i) targets[(size_t)i] = tok[(size_t)i + 1];
  for (size_t k = 0; k < S; ++k) {                                 // the last row of each sequence predicts beyond its rows
    const SeqEntry& se = sb.seqs[k];
    int64_t ix;
    if (!np_idx((int64_t)se.row0 + se.n_rows - 1, n, &ix)) return bad(err, "TableBuilder.build: target row out of range");
    targets[(size_t)ix] = row_tokens ? -1 : tokens_of(se.rid, se.start + se.n_rows);
  }
  const std::vector<int64_t>& pos = sb.positions;
  const std::vector<int32_t>& seq_of = sb.seq_slot;
  if ((int64_t)pos.size() != n || (int64_t)seq_of.size() != n) return bad(err, "TableBuilder.build: row arrays != n_rows");
  t.si.assign((size_t)(n * SW), -1);
  t.sl.assign((size_t)n, 0);
  t.ti_sources.assign(sources.begin(), sources.end());
  if (stub) { t.ti.assign(NS, {}); t.tl.assign(NS, {}); }
  if (n) {
    flat_w.push_back(-1);
    const int64_t nfw = (int64_t)flat_w.size();
    for (int64_t i = 0; i < n; ++i) {
      const auto& R = rec[(size_t)seq_of[(size_t)i]];
      const int64_t p = pos[(size_t)i];
      const int64_t lo = std::max(R[8], p - (SW - 1));
      for (int64_t j = 0; j < SW; ++j) {
        const int64_t idx = lo + j;
        const bool ok = idx <= p;
        const int64_t idc = std::min(idx, p);
        int64_t fi;
        if (!np_idx(R[12] + fdiv(idc, WT) - R[13], nfw, &fi)) return bad(err, "TableBuilder.build: window index out of range");
        const int64_t slot = flat_w[(size_t)fi] * WT + pmod(idc, WT);
        if (ok && slot < 0) return bad(err, "window page missing");
        t.si[(size_t)(i * SW + j)] = ok ? (int32_t)slot : -1;
      }
      t.sl[(size_t)i] = (int32_t)(p - lo + 1);
    }
    for (size_t s = 0; s < (stub ? NS : 0); ++s) {
      const int64_t c = ratio_of(sources[s]), rpp = PAGE_TOKENS / c;
      std::vector<int64_t> fc = flat_c[s];
      fc.push_back(0);
      const int64_t nfc = (int64_t)fc.size();
      t.ti[s].assign((size_t)(n * 512), -1);
      t.tl[s].assign((size_t)n, 0);
      for (int64_t i = 0; i < n; ++i) {
        const int64_t p = pos[(size_t)i];
        const int64_t q = seq_of[(size_t)i];
        const int64_t nc = c > 1 ? fdiv(p + 1, c) : p;
        const int64_t lo_c = std::max<int64_t>(nc - 512, 0);
        for (int64_t j = 0; j < 512; ++j) {
          const int64_t jj = lo_c + j;
          const bool okc = jj < nc;
          const int64_t tk = okc ? jj * c : 0;
          const int64_t blk = okc ? fdiv(tk, BT) - clo[s][(size_t)q] : 0;
          int64_t fi;
          if (!np_idx(coff[s][(size_t)q] + blk, nfc, &fi)) return bad(err, "TableBuilder.build: stub index out of range");
          t.ti[s][(size_t)(i * 512 + j)] = okc ? (int32_t)(fc[(size_t)fi] * rpp + fdiv(pmod(tk, BT), c)) : -1;
        }
        t.tl[s][(size_t)i] = (int32_t)(nc - lo_c);
      }
    }
  }
  // seq dict (Rs = R, or one zero row for an empty batch)
  RankTablesSeq& q = t.seq;
  std::vector<std::array<int64_t, 14>> Rs = rec;
  if (Rs.empty()) Rs.push_back({});
  for (const auto& R : Rs) {
    q.state_slot.push_back((int32_t)R[0]); q.uid.push_back((int32_t)R[1]); q.start.push_back(R[2]);
    q.n_rows.push_back((int32_t)R[3]); q.kind.push_back((int32_t)R[4]); q.last_chunk.push_back((int32_t)R[5]);
    q.score.push_back((int32_t)R[6]); q.forced.push_back((int32_t)R[7]); q.replay_start.push_back(R[8]);
    q.cached_upto.push_back(R[9]); q.n_blocks.push_back((int32_t)R[10]); q.bt_reset.push_back((int32_t)R[11]);
  }
  for (const auto& u : upd) {
    q.bt_upd_seq.push_back((int32_t)u[0]); q.bt_upd_start.push_back((int32_t)u[1]); q.bt_upd_len.push_back((int32_t)u[2]);
  }
  for (int64_t p : upd_pages) q.bt_upd_pages.push_back((int32_t)p);
  if (full_block_tables) {
    int64_t nbm = 0;
    for (const auto& se : sb.seqs) nbm = std::max<int64_t>(nbm, (int64_t)se.block_table->size());
    if (sb.seqs.empty()) nbm = 1;
    q.has_block_table = true;
    q.bt_rows = std::max<int64_t>((int64_t)S, 1);
    q.bt_cols = nbm;
    q.block_table.assign((size_t)(q.bt_rows * nbm), -1);
    for (size_t k = 0; k < S; ++k) {
      const auto& b = *sb.seqs[k].block_table;
      for (size_t c = 0; c < b.size(); ++c) q.block_table[k * (size_t)nbm + c] = (int32_t)b[c];
    }
  }
  t.n = (int32_t)n;
  t.positions = pos;
  t.tokens = tok;
  t.own_slot = sb.window_slot;
  for (int32_t r : sb.sample_rows) t.sample_rows.push_back(r);
  for (const auto& se : sb.seqs) t.seq_rows.push_back(RankSeqRow{se.rid, se.row0, se.n_rows, se.kind, se.last_chunk});
  t.seq_slot = seq_of;
  t.paged_slot = sb.paged_slot;
  for (int32_t r : sb.tail_rows) t.tail_rows.push_back(r);
  for (int32_t r : sb.sample_rows) t.sample_idx.push_back(r);
  t.targets = std::move(targets);
  return Status::OK;
}

// ------------------------------------------------------------------------------------------------ build_compact
Status TableBuilder::build_compact(const StepBatch& sb, int32_t k1, const std::vector<int64_t>* row_tokens,
                                   CompactTables* out, std::string* err) {
  CompactTables& ct = *out;
  ct = CompactTables{};
  ct.rank = sb.rank;
  ct.step = sb.step;
  ct.k1 = k1;
  std::vector<std::array<int64_t, 3>> clears, sets, btu;
  std::vector<int64_t> btp;
  std::vector<std::array<int64_t, 11>> rec;
  for (const SeqEntry& se : sb.seqs) {
    const int32_t slot = se.state_slot;
    const auto key = std::make_pair((int32_t)sb.rank, slot);
    auto it = cstate_.find(key);
    int64_t start = se.start;
    const std::vector<int64_t>& bt = *se.block_table;
    const int64_t nb = se.n_blocks ? se.n_blocks : (int64_t)bt.size();
    if (it == cstate_.end() || it->second.uid != se.uid || it->second.inc != se.inc) {   // new to this slot
      clears.push_back({slot, -1, -1});
      it = cstate_.insert_or_assign(key, CState{se.uid, se.inc, -1, -2, WindowTable{}}).first;
      if (start < 0) start = se.known;
    }
    CState& e = it->second;
    const int64_t s0 = e.blocks;
    if (s0 < 0 || s0 > nb) {                                     // first table, or it shrank: the full table
      btu.push_back({slot, 0, nb});
      py_slice(bt, 0, nb, &btp);
      e.blocks = nb;
    } else if (nb > s0) {
      btu.push_back({slot, s0, nb - s0});
      py_slice(bt, s0, nb, &btp);
      e.blocks = nb;
    }
    const int64_t wv = se.wver;
    if (wv < 0 || e.wver != wv) {                                // the window changed since it was last sent
      WindowTable& ws = e.ws;
      const WindowTable& wt = *se.window_table;
      for (const auto& bp : wt.items) {
        const int64_t i = ws.find(bp.first);
        if (i < 0 || ws.items[(size_t)i].second != bp.second) {
          sets.push_back({slot, bp.first, bp.second});
          ws.set(bp.first, bp.second);
        }
      }
      if (ws.size() > wt.size()) {
        std::vector<int64_t> drop;
        for (const auto& bp : ws.items) if (!wt.contains(bp.first)) drop.push_back(bp.first);
        for (int64_t b : drop) ws.erase(b);
      }
      e.wver = wv;
    }
    rec.push_back({slot, se.row0, se.n_rows, se.kind == "decode" ? 0 : 1, start, se.replay_start, se.cached_upto, se.uid,
                   se.last_chunk, se.score, se.forced});
  }
  for (const auto& R : rec) {
    ct.slot.push_back(R[0]); ct.row0.push_back(R[1]); ct.n_rows.push_back(R[2]); ct.kind.push_back(R[3]);
    ct.start.push_back(R[4]); ct.replay_start.push_back(R[5]); ct.cached_upto.push_back(R[6]); ct.uid.push_back(R[7]);
    ct.last_chunk.push_back(R[8]); ct.score.push_back(R[9]); ct.forced.push_back(R[10]);
  }
  for (const auto* V : {&clears, &sets})
    for (const auto& v : *V) { ct.win_slot.push_back(v[0]); ct.win_blk.push_back(v[1]); ct.win_page.push_back(v[2]); }
  for (const auto& b : btu) { ct.bt_upd_slot.push_back(b[0]); ct.bt_upd_start.push_back(b[1]); ct.bt_upd_len.push_back(b[2]); }
  ct.bt_upd_pages = btp;
  const int64_t n = sb.n_rows;
  if (row_tokens) {
    if (n) ct.tokens.assign(row_tokens->begin(), row_tokens->begin() + std::min<int64_t>(n, (int64_t)row_tokens->size()));
  } else {                                                      // host-known rows only: explicit starts
    if (!tokens_of) return bad(err, "build_compact: no row_tokens and no tokens_of");
    std::vector<int64_t> tok;
    for (size_t k = 0; k < sb.seqs.size(); ++k) {
      const SeqEntry& se = sb.seqs[k];
      const int64_t st = rec[k][4];
      for (int64_t i = 0; i < se.n_rows; ++i) tok.push_back(st >= 0 ? tokens_of(se.rid, st + i) : -1);
    }
    if (tok.empty()) tok.push_back(0);
    ct.tokens.assign(tok.begin(), tok.begin() + std::min<int64_t>(n, (int64_t)tok.size()));
  }
  // targets: row i predicts the token of row i + 1 of the same sequence (rows of a sequence are contiguous)
  ct.targets.assign((size_t)n, -1);
  if (n) {
    if ((int64_t)ct.tokens.size() != n) return bad(err, "build_compact: tokens != n_rows");
    for (int64_t i = 0; i + 1 < n; ++i) ct.targets[(size_t)i] = ct.tokens[(size_t)i + 1];
    for (size_t k = 0; k < rec.size(); ++k) {
      int64_t ix;
      if (!np_idx(rec[k][1] + rec[k][2] - 1, n, &ix)) return bad(err, "build_compact: target row out of range");
      ct.targets[(size_t)ix] = -1;
    }
  }
  for (int32_t r : sb.tail_rows) ct.tail_rows.push_back(r);
  for (int32_t r : sb.sample_rows) ct.sample_idx.push_back(r);
  for (const auto& se : sb.seqs) ct.rids.push_back(se.rid);
  int64_t sum = 0;
  for (int64_t x : ct.n_rows) sum += x;
  ct.n = sum;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------------ Worker.copies
void dedup_copies(const std::vector<StepBatch>& batches, std::vector<CopyOp>* ops) {
  ops->clear();
  std::map<std::tuple<int32_t, int32_t, int64_t, int32_t, int64_t>, size_t> at;
  for (const auto& sb : batches)
    for (const auto& c : sb.copies) {
      auto k = std::make_tuple(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page);
      auto it = at.find(k);
      if (it == at.end()) { at[k] = ops->size(); ops->push_back(c); }
      else (*ops)[it->second] = c;                                  // dict: the key keeps its place, the value is replaced
    }
}

void page_copy_batches(const std::vector<CopyOp>& ops, std::vector<PageCopyBatch>* out) {
  out->clear();
  std::map<std::tuple<int32_t, int32_t, int32_t>, size_t> at;
  for (const auto& c : ops) {
    auto k = std::make_tuple(c.pool, c.src_rank, c.dst_rank);
    auto it = at.find(k);
    if (it == at.end()) {
      at[k] = out->size();
      const int64_t pb = c.pool == POOL_PAGED ? PAGE_BYTES : c.pool == POOL_WINDOW ? WINDOW_PAGE_BYTES : STATE_SLOT_BYTES;
      out->push_back(PageCopyBatch{c.pool, c.src_rank, c.dst_rank, {}, {}, pb});
      it = at.find(k);
    }
    (*out)[it->second].src_pages.push_back(c.src_page);
    (*out)[it->second].dst_pages.push_back(c.dst_page);
  }
}

void copy_runs(const std::vector<CopyOp>& copies, std::vector<CopyRun>* out) {
  out->clear();
  std::vector<CopyOp> s = copies;
  std::stable_sort(s.begin(), s.end(), [](const CopyOp& a, const CopyOp& b) {
    return std::make_tuple(pool_sort_rank(a.pool), a.src_rank, a.dst_rank, a.src_page) <
           std::make_tuple(pool_sort_rank(b.pool), b.src_rank, b.dst_rank, b.src_page);
  });
  for (const auto& c : s) {
    if (!out->empty()) {
      CopyRun& r = out->back();
      if (r.pool == c.pool && r.src_rank == c.src_rank && r.dst_rank == c.dst_rank && c.src_page == r.src_page0 + r.n &&
          c.dst_page == r.dst_page0 + r.n) {
        r.n += 1;
        continue;
      }
    }
    out->push_back(CopyRun{c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page, 1});
  }
}

void state_slot_copies(const std::vector<CopyOp>& ops, std::vector<std::array<int64_t, 4>>* out) {
  out->clear();
  for (const auto& c : ops)
    if (c.pool == POOL_STATE) out->push_back({c.src_rank, c.src_page, c.dst_rank, c.dst_page});
}

// the delta state as one copy: save before a native build, restore on the Python fallback
struct TableBuilder::State {
  static constexpr uint64_t kMagic = 0x5442535441544531ull;   // "TBSTATE1"
  uint64_t magic = kMagic;
  std::map<std::pair<int32_t, int32_t>, Sent> sent;
  std::map<std::pair<int32_t, int32_t>, CState> cstate;
};

std::shared_ptr<const void> TableBuilder::save_state() const {
  auto p = std::make_shared<State>();
  p->sent = sent_;
  p->cstate = cstate_;
  return p;
}

void TableBuilder::restore_state(const std::shared_ptr<const void>& state) {
  if (!state) return;
  const auto* p = static_cast<const State*>(state.get());
  if (p->magic != State::kMagic) throw std::invalid_argument("TableBuilder::restore_state: not a TableBuilder state");
  sent_ = p->sent;
  cstate_ = p->cstate;
}

std::unique_ptr<ITables> make_tables() { return std::make_unique<TableBuilder>(); }

}  // namespace fpr
