// nd_hist.h -- NativeDecode's request bookkeeping and host token history (handover J), CUDA-free.
// The parts of nd::Engine that must match the embedded Python engine byte for byte on the HOST side, split out so the
// J replay (tests/nd_hist_test.cpp) runs on any CPU:
//   observe_requests      hooks.on_ctx's admissions / drops (eng.hist, eng.prev_seq, samp_ctx, structured)
//   results_and_history   HSD.result's formula per rank + head_result / extract_deferred's `eng.hist[rid].extend(toks)`
//   push_hist             push_state's history entries: the tokens extracted natively since the last handoff, then the
//                         requests DROPPED on native steps (n = -1). A drop seen only by nd (a request finishing during
//                         native decode) never reaches Python's on_ctx: without it eng.hist / samp_ctx / structured keep
//                         the request forever (unbounded growth) and J differs from the all-Python run.
//   pull_hist             (v0.5) pull_state's history DELTAS: the tokens an embedded step appended to eng.hist, so Req.hist
//                         (the full prompt + output, as eng.hist[rid]) stays equal to Python's on every step boundary:
//                         native mixed steps build Engram's host lookback from it.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../include/d13.h"
#include "../core2b/engine_handover.h"
#include "../core2b/nd_api.h"

namespace fpr {
namespace nd {

struct Req {            // nd's per-request state (StepHost.fresh .. dropped)
  Rid rid;
  int64_t serial = 0;
  Sampling sampling;
  bool structured = false;
  std::vector<int32_t> hist_pending;  // tokens extracted natively since the last push_state (J: -> eng.hist[rid])
  std::vector<int32_t> hist;          // (v0.5) = eng.hist[rid]: prompt + output at admission, + native extractions,
                                      // + pull_state's deltas (embedded steps). Engram's host lookback on native mixed steps
};

struct PrevOf { int64_t rank = -1, index = -1, serial = -1; };   // eng.prev_seq[rid]

// hooks.on_ctx: fresh -> a new request state (eng.hist = prompt + output is Python's), prev-of dropped; dropped -> every per-request entry dropped. When
// `native_drops` is given (a NATIVE step: Python's on_ctx never sees this StepHost) the drops are queued for push_hist.
void observe_requests(const StepHost& host, std::map<Rid, Req>* reqs, std::map<Rid, PrevOf>* prev,
                      std::vector<Rid>* native_drops);

// FPR's on_ctx invariant: a context never drops AND admits the same rid
// nd refuses such a StepHost loudly
// (Status BAD_ARG) instead of silently losing the request's history (on_ctx handles `new` before `dropped`).
bool new_dropped_disjoint(const StepHost& host, Rid* clash);

// HSD.result per rank r: for (si, qi) in dseqs[r]: acc[rid] = ns[r][qi], tok[rid] = smp[r][qi][:acc] (dict semantics: a
// repeated rid keeps its first position, takes the last value); ns[r] == nullptr = no copies for rank r (RES_OWNER /
// deferred: ranks without drafting requests). Then, rank by rank, rid in reqs -> its pending tokens extended.
void results_and_history(const d13::HsdStep& c, int n_ranks, int64_t K1, const int32_t* const* ns,
                         const int64_t* const* smp, std::map<Rid, Req>* reqs, std::vector<StepResult>* out);

// push_state's J entries (pointers into reqs / drops: valid until either changes): live requests with pending tokens
// (n >= 0) in rid order, then the native drops (tokens = nullptr, n = -1), each once.
void push_hist(const std::map<Rid, Req>& reqs, const std::vector<Rid>& native_drops, std::vector<FprEhHist>* out);

// (v0.5) pull_state's deltas: hist[i] (n >= 0) appends its tokens to the live request's Req.hist. Returns the number of
// entries whose rid nd does not know (0 when both sides agree; > 0 is a bookkeeping bug the caller reports).
int64_t pull_hist(const FprEhHist* hist, int64_t n_hist, std::map<Rid, Req>* reqs);

// the host ids at positions start-1, start-2, start-3 of request
// rid (-1 below position 0), engine.py Hooks._lookback literally: rid = cp.alias.get(rid, rid) with cp.alias built as
// cp.CPPlan does from the step's batches (an entry with a CP role whose cp.rid differs from its own rid -> cp.rid; batches
// in rank order, seqs in order, the last assignment wins), then h = eng.hist[rid] (= Req.hist); start <= len(h) -> the
// three ids. false = no history or start > len(h) (Python's prev_tokens fallback: the caller refuses the native step).
bool lookback_of(const std::map<Rid, Req>& reqs, const std::vector<StepBatch>& b, const Rid& rid, int64_t start,
                 int64_t out[3]);

// c1mega_eng.owner_of's batch half: the ranks with rows (sb.n_rows > 0, in
// batch order); exactly one, holding <= nrm rows -> that rank, else -1. The state half (R == NRM, pf None, not cp_on, moe
// raw, lvl 0) is the caller's: decode buckets never carry pf / cp_on, are moe raw, and a long variant's key is R + lvl 2^20.
int32_t c1m_owner_of(const std::vector<StepBatch>& b, int64_t nrm);

// engine.Hooks._bitmasks + prepare_head's num_acceptable_drafts, literally: per rank, the grammar
// rows of every sequence with an entry in g (a dict by rid: the LAST entry of a rid wins) placed at its sample rows
// (the seq's rows [row0, row0 + n_rows) that are in sb.sample_rows, in row order, the first rows.n_rows of them), every
// other sample row -1 (all allowed); words = the rank's first grammar entry's. A rank with no sample rows or no grammar
// sequence -> rows == nullptr. Returns 1 = some rank has a mask, 0 = none (Python: None -> no bitmask variant), -1 = a
// malformed entry (rows not [n_rows, words] / a rank mixing word counts: numpy raises; the caller refuses natively).
// store: owns the masks the returned HsdBitmask point into (valid while store lives).
int grammar_bitmasks(const std::vector<StepBatch>& b, const std::vector<GrammarRows>& g,
                      std::vector<std::vector<int32_t>>* store, std::vector<d13::HsdBitmask>* out);
const GrammarRows* grammar_of(const std::vector<GrammarRows>& g, const Rid& rid);   // the dict lookup (last entry wins)

// (v3.30, long-context decode, design slice 1: notes long-context-native-decode-design-16nov.md) the long-row LEVEL of an
// async decode step = the proposed d13.D13._long_level (engine.step's sync rule, engine.py np11 :3708-3717, applied to
// D13's own context bound): mx = max over the step's decode sequences of (ct.start >= 0 ? start : se.known + k1) + k1
// (D13._async_ok's formula); 0 = within lvl 0's indexer bound (graph_max_ctx); lvl > 0 = the first LONG_LEVEL_CTX entry
// holding mx, when engine.step would take the long graphed variant (ENG_LONG_GRAPHS, split HSD, HSD on, not DBG_SYNC,
// > 64 rows in all or FPR_LONG_ANY_ROWS); -1 = no variant (Python's eager mixed path). *mx_out = mx.
// a device-resolved start lies in (known, known + K1]: per
// sequence lo = known + K1, hi = known + 2 K1 (host-known starts: both start + K1); when lo's and hi's levels differ the step
// stays sync (kLongEdge: async == sync exactly, no lvl 0 / 1 tie at an edge); a C1 lane (FPR_C1_KMAX > 3) or variable verify
// rows present -> sync (kLongLane).
struct LongCfg {
  int64_t graph_max_ctx = 262144;      // engine.GRAPH_MAX_CTX
  std::vector<int64_t> level_ctx;      // engine.LONG_LEVEL_CTX (= hooks_idx.LEVEL_CTX): lvl i's indexer bound
  int64_t k1 = 4;
  bool long_graphs = false, split_hsd = false, hsd_on = false, long_any_rows = false, dbg_sync = false;
  bool c1_lane = false;                // (v3.34) D13 has a C1 lane HSD or variable verify rows: long steps stay sync
};
constexpr int kLongEdge = -2;          // (v3.34) "long level edge (device-resolved start)"
constexpr int kLongLane = -3;          // (v3.34) "... (C1 lane / variable rows: sync)"
constexpr int64_t kLvlTag = int64_t(1) << 20;   // engine.LVL_TAG: a long variant's decode bucket key = R + lvl * LVL_TAG
inline int64_t long_key(int64_t R, int lvl) { return R + (int64_t)lvl * kLvlTag; }
int long_level_of(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const LongCfg& c,
                  int64_t* mx_out = nullptr);

}  // namespace nd
}  // namespace fpr
