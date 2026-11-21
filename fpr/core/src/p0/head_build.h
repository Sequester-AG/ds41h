// head_build.h -- P0 builder of the mixed step's head prepare in C++.
// = fpr/engine/p0_head.py build_prepare_head, decision for decision: engine.Hooks.prepare_head's sampling-state derivation
// (drafts_valid from engine.prev_of / host-given ids, num_acceptable_drafts from the grammar entry, the default state) and
// its two non-ok outcomes, from an explicit snapshot (no engine reads). The HSD blob itself is HSD._build =
// fpr/core/src/d13/hsd_prep.cpp (HsdPrep::build), fed with to_hsd_states(); its upload = head_records() (named
// destinations, FPR's mixed_fields.h check). Golden: core/tests/replay_head.cpp on record_d13_cpu.py P0_HEAD_OUT=1 traces.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../include/common.h"
#include "../../include/d13.h"
#include "../../include/tables.h"

namespace fpr {
namespace p0 {

struct HeadAttrs {                      // a sampling state object as the builder reads it (p0_head._attrs)
  double temperature = 1.0;
  int64_t seed = 0;
  bool drafts_valid = true;
  bool has_num_acceptable = false;
  int64_t num_acceptable = 0;
};
struct HeadRid {                        // snap["rids"][rid]
  bool has_sampling = false; HeadAttrs sampling;   // vLLM's sampling dict entry (authoritative)
  bool has_ctx = false; HeadAttrs ctx;             // the StepContext dict entry (samp_ctx)
  bool dev_prev = false;                           // engine.prev_of(rid) is not None
  bool has_grammar = false; int64_t grammar = 0;   // int(grammar[rid][1])
};
struct HeadSnap {
  bool hsd = true, gate_only = false, grammar_any = false;
  bool bm_ok = false;   // the caller builds the masks (graph H1's "|bm" variant): no REFUSED_BITMASK
  std::map<Rid, HeadRid> rids;
};
enum class HeadStatus { OK = 0, SKIP_GATE = 1, REFUSED_BITMASK = 2 };
enum class HeadSrc { SAMPLING = 0, CTX = 1, DEFAULT = 2 };
struct HeadState { HeadAttrs v; HeadSrc src = HeadSrc::DEFAULT; };
struct HeadOut {
  HeadStatus status = HeadStatus::OK;
  std::map<Rid, HeadState> states;      // per rid (a rid on several ranks: the last one, as Python's dict)
  int64_t n_dec = 0, n_resume = 0, n_host = 0;
};
struct HeadRecord { std::string op, src, dst; int rank = 0; int64_t bytes = 0; };

const char* head_status_name(HeadStatus s);   // "ok" / "skip: gate-only" / "refused: bitmask" (p0_head.py's strings)
const char* head_src_name(HeadSrc s);         // "sampling" / "ctx" / "default"
// tokens[r]: rank r's T table host row ids (nullptr = no table / an empty one)
HeadOut build_prepare_head(const HeadSnap& snap, const std::vector<StepBatch>& b,
                           const std::vector<const std::vector<int64_t>*>& tokens);
// HsdPrep::build's states (per rank, per sequence of batch.seqs). ident: a caller-provided identity per (rid, src) so the
// slot's temperature / seed are re-read exactly when Python's state object changes (0 = always re-read, the default state)
std::vector<std::vector<d13::HsdState>> to_hsd_states(const HeadOut& out, const std::vector<StepBatch>& b,
                                                      uint64_t (*ident)(const Rid&, HeadSrc) = nullptr);
std::vector<HeadRecord> head_records(int64_t blob_bytes, int hi, int n_ranks);   // p0_head.head_records

}  // namespace p0
}  // namespace fpr
