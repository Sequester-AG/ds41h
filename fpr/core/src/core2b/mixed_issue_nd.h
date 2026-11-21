// mixed_issue_nd.h -- the production issue backend of the native mixed step.
// prepare() runs the owners' phase-1 hooks in registration order (P1 gt_host / pf_host, the Engram host prep, ...).
#pragma once
#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "mixed_exec.h"

namespace fpr {
namespace core2b {

using MixedOpFn = std::function<Status(const p0::MixedOp& op, const p0::MixedPlan& plan, std::string* err)>;

using MixedPrepFn = std::function<std::string(const p0::MixedPlan& plan, const MixedStepIn& in)>;

constexpr int kMixedOpKinds = (int)p0::MixedOpKind::LOG + 1;

class NdMixedIssue final : public IMixedIssue {
 public:
  NdMixedIssue();
  // an owner's issue function for one op kind (replaces any earlier one)
  void set(p0::MixedOpKind k, MixedOpFn fn) { fn_[(int)k] = std::move(fn); ++gen_[(int)k]; }
  // (L4) bumped by every set(): NativeWorker checks that no owner replaced its no-op HEAD_RESULT / MEM_AFTER_STEP
  uint64_t gen(p0::MixedOpKind k) const { return gen_[(int)k]; }
  // an owner's phase-1 hook ("" = OK): after the real build (add_prepare) or on the plan-only plan (add_prepare_plan)
  void add_prepare(std::string owner, MixedPrepFn fn) { prep_.emplace_back(std::move(owner), std::move(fn)); }
  void add_prepare_plan(std::string owner, MixedPrepFn fn) { prep_plan_.emplace_back(std::move(owner), std::move(fn)); }
  std::string prepare_plan(const p0::MixedPlan& plan, const MixedStepIn& in) override;
  void set_begin(std::function<Status(const p0::MixedPlan&, std::string*)> fn) { begin_ = std::move(fn); }
  Status begin_issue(const p0::MixedPlan& plan, std::string* err) override { return begin_ ? begin_(plan, err) : Status::OK; }
  // the host-only kinds (phase 1 did them): issue = nothing
  static bool host_only(p0::MixedOpKind k);
  bool covers(p0::MixedOpKind k) const override { return host_only(k) || (bool)fn_[(int)k]; }
  std::string prepare(const p0::MixedPlan& plan, const MixedStepIn& in) override;
  Status issue(const p0::MixedOp& op, const p0::MixedPlan& plan, std::string* err) override;
  // the kinds still missing for a plan (the "not native" table of the stats line)
  std::vector<std::string> missing(const p0::MixedPlan& plan) const;
  // per hook ("hook <owner>", "hook1b <owner>") and op kind ("op <name>"): host time; the caller serializes (run_mixed)
  const std::map<std::string, StepTimer>& times() const { return times_; }

 private:
  std::array<MixedOpFn, kMixedOpKinds> fn_;
  std::array<uint64_t, kMixedOpKinds> gen_{};
  std::vector<std::pair<std::string, MixedPrepFn>> prep_, prep_plan_;
  std::function<Status(const p0::MixedPlan&, std::string*)> begin_;
  std::map<std::string, StepTimer> times_;
};

}  // namespace core2b
}  // namespace fpr
