// mixed_issue_nd.cpp -- see mixed_issue_nd.h. Build: -O2 -std=c++17.
#include "mixed_issue_nd.h"

#include <chrono>
#include <set>

namespace fpr {
namespace core2b {

namespace {
double us_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

NdMixedIssue::NdMixedIssue() {}

bool NdMixedIssue::host_only(p0::MixedOpKind k) {
  using K = p0::MixedOpKind;
  switch (k) {
    case K::BUCKET:              // the views: fetched (ensure_mixed + fill_mixed) by the BUCKET-phase hook in prepare()
    case K::TAIL_BATCH:          // build_mixed (phase 1)
    case K::TB_TAIL:             // build_mixed through the engine's tb_tail (phase 1)
    case K::PF_PLANS:            // the per-rank PF_PLAN ops carry the work
    case K::IMG_ROWS_A:          // no device work without images (build_mixed refuses mm_on steps: native_refusal)
    case K::IMG_ROWS_T:
    case K::AFTER_SYNC:          // the H1 capture stays embedded (native steps run captured keys only)
    case K::LOG:
      return true;
    default:
      return false;
  }
}

std::string NdMixedIssue::prepare_plan(const p0::MixedPlan& plan, const MixedStepIn& in) {
  for (auto& p : prep_plan_) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string w = p.second(plan, in);
    times_["hook " + p.first].add(us_since(t0));
    if (!w.empty()) return p.first + ": " + w;
  }
  return "";
}

std::string NdMixedIssue::prepare(const p0::MixedPlan& plan, const MixedStepIn& in) {
  for (auto& p : prep_) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string w = p.second(plan, in);
    times_["hook1b " + p.first].add(us_since(t0));
    if (!w.empty()) return p.first + ": " + w;
  }
  return "";
}

Status NdMixedIssue::issue(const p0::MixedOp& op, const p0::MixedPlan& plan, std::string* err) {
  if (host_only(op.kind)) return Status::OK;
  const MixedOpFn& f = fn_[(int)op.kind];
  if (!f) {                      // (run_mixed's covers() check makes this unreachable)
    if (err) *err = std::string("no issue function for ") + p0::mixed_op_name(op.kind);
    return Status::INTERNAL;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const Status s = f(op, plan, err);
  times_[std::string("op ") + p0::mixed_op_name(op.kind)].add(us_since(t0));
  return s;
}

std::vector<std::string> NdMixedIssue::missing(const p0::MixedPlan& plan) const {
  std::set<std::string> m;
  for (const p0::MixedOp& o : plan.ops)
    if (!covers(o.kind)) m.insert(p0::mixed_op_name(o.kind));
  return std::vector<std::string>(m.begin(), m.end());
}

}  // namespace core2b
}  // namespace fpr
