// nd_hist.cpp -- see nd_hist.h. No CUDA.
#include "nd_hist.h"

#include <algorithm>
#include <set>
#include <utility>

namespace fpr {
namespace nd {

void observe_requests(const StepHost& host, std::map<Rid, Req>* reqs, std::map<Rid, PrevOf>* prev,
                      std::vector<Rid>* native_drops) {
  for (const auto& f : host.fresh) {   // on_ctx `new`: eng.hist[rid] = prompt + output; samp_ctx; prev_seq.pop
    Req& q = (*reqs)[f.rid];
    q.rid = f.rid;
    q.serial = f.serial;
    q.sampling = f.sampling;   // the caller's seed (vLLM draws one at admission when none is given)
    q.structured = f.structured;
    q.hist_pending.clear();   // (a re-admission: the old incarnation's tokens die with Python's on_ctx too)
    q.hist.assign(f.prompt.begin(), f.prompt.end());   // eng.hist[rid] = list(prompt_ids) + list(output_ids)
    q.hist.insert(q.hist.end(), f.output.begin(), f.output.end());
    prev->erase(f.rid);
  }
  for (const auto& rid : host.dropped) {   // on_ctx `dropped`: hist / prev_seq / samp_ctx / structured popped
    reqs->erase(rid);
    prev->erase(rid);
    if (native_drops != nullptr) native_drops->push_back(rid);
  }
}

bool new_dropped_disjoint(const StepHost& host, Rid* clash) {
  if (host.fresh.empty() || host.dropped.empty()) return true;
  std::set<Rid> d(host.dropped.begin(), host.dropped.end());
  for (const auto& f : host.fresh)
    if (d.count(f.rid)) {
      if (clash) *clash = f.rid;
      return false;
    }
  return true;
}

void results_and_history(const d13::HsdStep& c, int n_ranks, int64_t K1, const int32_t* const* ns,
                         const int64_t* const* smp, std::map<Rid, Req>* reqs, std::vector<StepResult>* out) {
  out->assign(n_ranks, StepResult{});
  for (int r = 0; r < n_ranks; ++r) {
    StepResult& o = (*out)[r];
    o.rank = r;
    o.step = r < (int)c.step.size() ? c.step[r] : 0;
    if (ns[r] != nullptr && r < (int)c.dseqs.size())
      for (const auto& p : c.dseqs[r]) {
        const int64_t qi = p.second;
        const Rid& rid = c.rids[qi];
        const int32_t a_ = ns[r][qi];
        std::vector<int32_t> toks;   // smp[qi][:a_] (a Python slice: clamps at the row's k + 1 entries)
        for (int64_t j = 0; j < std::min<int64_t>(a_, K1); ++j) toks.push_back((int32_t)smp[r][qi * K1 + j]);
        auto ia = std::find_if(o.accepted.begin(), o.accepted.end(), [&](const auto& x) { return x.first == rid; });
        if (ia == o.accepted.end()) o.accepted.emplace_back(rid, a_); else ia->second = a_;
        auto it = std::find_if(o.tokens.begin(), o.tokens.end(), [&](const auto& x) { return x.first == rid; });
        if (it == o.tokens.end()) o.tokens.emplace_back(rid, std::move(toks)); else it->second = std::move(toks);
      }
    for (const auto& kv : o.tokens) {   // head_result: `if rid in self.eng.hist: self.eng.hist[rid].extend(toks)`
      auto it = reqs->find(kv.first);
      if (it == reqs->end()) continue;
      it->second.hist_pending.insert(it->second.hist_pending.end(), kv.second.begin(), kv.second.end());
      it->second.hist.insert(it->second.hist.end(), kv.second.begin(), kv.second.end());
    }
  }
}

void push_hist(const std::map<Rid, Req>& reqs, const std::vector<Rid>& native_drops, std::vector<FprEhHist>* out) {
  out->clear();
  for (const auto& kv : reqs)
    if (!kv.second.hist_pending.empty())
      out->push_back(FprEhHist{kv.first.c_str(), kv.second.hist_pending.data(), (int64_t)kv.second.hist_pending.size()});
  std::set<Rid> seen;
  for (const auto& rid : native_drops)
    if (seen.insert(rid).second) out->push_back(FprEhHist{rid.c_str(), nullptr, -1});
}

int64_t pull_hist(const FprEhHist* hist, int64_t n_hist, std::map<Rid, Req>* reqs) {
  int64_t unknown = 0;
  for (int64_t i = 0; i < n_hist; ++i) {
    if (hist[i].n < 0 || hist[i].rid == nullptr) continue;   // (drops travel push-wards only)
    auto it = reqs->find(Rid(hist[i].rid));
    if (it == reqs->end()) {
      ++unknown;
      continue;
    }
    it->second.hist.insert(it->second.hist.end(), hist[i].tokens, hist[i].tokens + hist[i].n);
  }
  return unknown;
}

const GrammarRows* grammar_of(const std::vector<GrammarRows>& g, const Rid& rid) {
  const GrammarRows* hit = nullptr;
  for (const GrammarRows& x : g)
    if (x.rid == rid) hit = &x;   // (dict(ctx.grammar): a repeated rid keeps its last value)
  return hit;
}

int grammar_bitmasks(const std::vector<StepBatch>& b, const std::vector<GrammarRows>& g,
                      std::vector<std::vector<int32_t>>* store, std::vector<d13::HsdBitmask>* out) {
  store->assign(b.size(), {});
  out->assign(b.size(), d13::HsdBitmask{});
  bool any = false;
  for (size_t r = 0; r < b.size(); ++r) {
    const StepBatch& sb = b[r];
    const std::vector<int32_t>& sr = sb.sample_rows;
    if (sr.empty()) continue;
    std::map<int64_t, int64_t> pos_in;   // {row: i for i, row in enumerate(sr)} (a repeated row keeps its last i)
    for (size_t i = 0; i < sr.size(); ++i) pos_in[sr[i]] = (int64_t)i;
    std::vector<int32_t>& bm = (*store)[r];
    int64_t words = -1;
    for (const SeqEntry& se : sb.seqs) {
      const GrammarRows* gr = grammar_of(g, se.rid);
      if (gr == nullptr) continue;
      if (words < 0) {
        words = gr->words;
        bm.assign((size_t)(sr.size() * words), -1);
      }
      if (gr->words != words || gr->words <= 0 || gr->n_rows < 0 || (int64_t)gr->rows.size() < gr->n_rows * gr->words)
        return -1;   // (numpy would raise on the row assignment / the reshape: the caller refuses)
      int64_t j = 0;
      for (int64_t x = se.row0; x < (int64_t)se.row0 + se.n_rows && j < gr->n_rows; ++x) {
        auto it = pos_in.find(x);
        if (it == pos_in.end()) continue;
        std::copy(gr->rows.begin() + j * words, gr->rows.begin() + (j + 1) * words, bm.begin() + it->second * words);
        ++j;
      }
    }
    if (words >= 0) {
      (*out)[r] = d13::HsdBitmask{(int64_t)sr.size(), words, bm.data()};
      any = true;
    }
  }
  return any ? 1 : 0;
}

int32_t c1m_owner_of(const std::vector<StepBatch>& b, int64_t nrm) {
  int32_t act = -1, n_act = 0;
  for (size_t r = 0; r < b.size(); ++r)
    if (b[r].n_rows > 0) {
      ++n_act;
      act = (int32_t)r;
    }
  if (n_act != 1 || b[(size_t)act].n_rows > nrm) return -1;
  return act;
}

bool lookback_of(const std::map<Rid, Req>& reqs, const std::vector<StepBatch>& b, const Rid& rid, int64_t start,
                 int64_t out[3]) {
  // cp.alias, cp.CPPlan.__init__ literally: groups[cp.group] in first-seen order, each ents[r] = the LAST entry of that
  // group on batch r (r = the batch's index, ranks first-seen); then for every group, every ents entry whose cp.rid
  // differs from its rid: alias[se.rid] = cp.rid (later assignments win). Only rid's own entries matter here.
  struct G {
    int64_t g;
    std::vector<std::pair<size_t, const SeqEntry*>> ents;
  };
  std::vector<G> gs;
  for (size_t r = 0; r < b.size(); ++r)
    for (const SeqEntry& se : b[r].seqs) {
      if (!se.has_cp) continue;
      G* gp = nullptr;
      for (G& x : gs)
        if (x.g == se.cp.group) gp = &x;
      if (gp == nullptr) {
        gs.push_back(G{se.cp.group, {}});
        gp = &gs.back();
      }
      bool hit = false;
      for (auto& e : gp->ents)
        if (e.first == r) {
          e.second = &se;
          hit = true;
        }
      if (!hit) gp->ents.emplace_back(r, &se);
    }
  const Rid* key = &rid;
  for (const G& x : gs)
    for (const auto& e : x.ents)
      if (e.second->rid == rid && e.second->cp.rid != e.second->rid) key = &e.second->cp.rid;
  auto it = reqs.find(*key);
  if (it == reqs.end()) return false;
  const std::vector<int32_t>& h = it->second.hist;
  if (start > (int64_t)h.size()) return false;
  for (int c = 0; c < 3; ++c) {
    const int64_t p = start - 1 - c;
    out[c] = p >= 0 ? (int64_t)h[(size_t)p] : -1;
  }
  return true;
}

int long_level_of(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const LongCfg& c,
                  int64_t* mx_out) {
  int64_t hi = 0, lo = 0, rows = 0;
  const size_t nr = std::min(b.size(), ct.size());
  for (size_t r = 0; r < b.size(); ++r) rows += b[r].n_rows;   // sum(sb.n_rows for sb in batches)
  for (size_t r = 0; r < nr; ++r) {                             // zip(batches, cts)
    const size_t S = std::min(b[r].seqs.size(), ct[r].start.size());
    for (size_t i = 0; i < S; ++i) {
      const int64_t st = ct[r].start[i];
      if (st >= 0) {
        hi = std::max(hi, st + c.k1);
        lo = std::max(lo, st + c.k1);
      } else {
        hi = std::max(hi, b[r].seqs[i].known + 2 * c.k1);
        lo = std::max(lo, b[r].seqs[i].known + c.k1);
      }
    }
  }
  if (mx_out) *mx_out = hi;   // (= D13._async_ok's bound)
  auto lv = [&](int64_t m) {   // 0 if m <= GRAPH_MAX_CTX else next((i for i, c in enumerate(LONG_LEVEL_CTX) if m <= c), -1)
    if (m <= c.graph_max_ctx) return 0;
    for (size_t i = 0; i < c.level_ctx.size(); ++i)
      if (m <= c.level_ctx[i]) return (int)i;
    return -1;
  };
  const int lvl = lv(hi);
  if (lv(lo) != lvl) return kLongEdge;
  if (lvl == 0) return 0;
  const bool ok = c.long_graphs && lvl > 0 && c.split_hsd && c.hsd_on && !c.dbg_sync && (rows > 64 || c.long_any_rows);
  if (!ok) return -1;
  if (c.c1_lane) return kLongLane;
  return lvl;
}

}  // namespace nd
}  // namespace fpr
