// mixed_exec.h -- the native mixed step's executor.
// A mixed step runs in two phases inside NativeWorker::mixed_step_, after nd's begin (resolved batches + bucket A's tables):
//   phase 1 (host only): a PLAN-ONLY build_mixed first (no tb_tail: every structural refusal before the shared tail
//            TableBuilder's state moves), then the real build_mixed (plan, op order, T tables, blobs, vidx, gather spec, ...) + the owners' host
//            builders (IMixedIssue::prepare: P1 gt_host / pf_host, the Engram host prep, ...).
//   phase 2 (issue): every op of plan.ops in engine order through IMixedIssue::issue (the owners' C ABI device halves: no
//            allocation, no host reads after issue). A failure here is FATAL (the step is half issued).
// The backend is abstract: production = NdMixedIssue (nd's upload_inputs / devtok / vidx_issue / head, p1_issue, FPR's
// TAIL_GATHER, the span program); tests = a recording backend (core/tests/replay_mixed.cpp --exec) whose op stream must equal
// the Python step's recorded calls (record_mixed_cpu.py) line by line.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../p0/mixed_build.h"

namespace fpr {
namespace core2b {

struct MixedStepIn {                    // phase 1's inputs
  const p0::MixedSnap* snap = nullptr;
  const std::vector<StepBatch>* rb = nullptr;          // nd's resolved batches (begin)
  const std::vector<RankTables>* tables = nullptr;     // bucket A's RankTables (nd's mixed_begin tables)
  p0::PrevFn prev;                                     // engine.prev_of
  ITables* tb_tail = nullptr;                          // the engine's ONE tail TableBuilder
  int64_t mixed_steps_before = 0;                      // stats["mixed_steps"] before the step
  // tb_tail's state around the real build: a refusal AFTER the build (an owner's
  // prepare hook) restores it, so the fallback's Python body rebuilds T's tables from the same D12 state. Unset = no
  // post-build refusal is allowed to leave the state advanced: such a refusal is FATAL instead.
  std::function<std::shared_ptr<const void>()> tb_save;
  std::function<void(const std::shared_ptr<const void>&)> tb_restore;
};

class IMixedIssue {
 public:
  virtual ~IMixedIssue() = default;
  // phase 1a, on the PLAN-ONLY plan (tb_tail untouched): checks that need only the structure (R_A / R_T, the op list), e.g.
  // ensure_mixed + fill_mixed of the step's views. "" = OK, else the refusal (nothing moved)
  virtual std::string prepare_plan(const p0::MixedPlan& plan, const MixedStepIn& in) { return ""; }
  // phase 1b, after the real build: the owners' host builders for this plan. "" = OK, else the refusal reason (nothing
  // issued; tb_tail's state is restored through MixedStepIn::tb_restore)
  virtual std::string prepare(const p0::MixedPlan& plan, const MixedStepIn& in) = 0;
  // phase 2, before the first op (phase 1 passed): work that precedes the step's ops but must not run for a refused step
  // Non-OK = fatal
  virtual Status begin_issue(const p0::MixedPlan& plan, std::string* err) { return Status::OK; }
  // phase 2: one op, in plan order. Non-OK = fatal (err says which op / rank)
  virtual Status issue(const p0::MixedOp& op, const p0::MixedPlan& plan, std::string* err) = 0;
  // which op kinds this backend issues natively; an op kind it does not cover makes phase 1 refuse ("op <name> not native")
  virtual bool covers(p0::MixedOpKind k) const = 0;
};

enum class MixedExec { NATIVE = 0, REFUSED = 1, FATAL = 2 };

// L0: host time of a step phase / hook / op kind (us), for worker_stats
struct StepTimer {
  int64_t n = 0;
  double us = 0, max_us = 0;
  void add(double x) { ++n; us += x; if (x > max_us) max_us = x; }
};

struct MixedExecStats {
  int64_t steps = 0, native = 0, refused = 0, fatal = 0, ops = 0;
  std::map<std::string, StepTimer> times;   // "build_mixed plan-only" / "build_mixed real" / "phase 2 issue"
  std::map<std::string, int64_t> refusals;             // reason class -> count
  std::map<std::string, int64_t> op_count;             // op name -> count (native steps)
};

// The refusal reason's class for the stats table: up to the first ':' / '(' / digit (as NativeWorker::note_refusal).
std::string refusal_class(const std::string& why);

// Phase 1 + phase 2 for one step. plan: filled in phase 1 (kept for the caller / checks). why: the refusal or fatal text.
MixedExec run_mixed(const MixedStepIn& in, IMixedIssue* io, p0::MixedPlan* plan, std::string* why,
                    MixedExecStats* st = nullptr);

}  // namespace core2b
}  // namespace fpr
