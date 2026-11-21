// mixed_exec.cpp -- see mixed_exec.h. Build: -O2 -std=c++17 -ffp-contract=off.
#include "mixed_exec.h"

#include <algorithm>
#include <chrono>

namespace fpr {
namespace core2b {

std::string refusal_class(const std::string& why) {
  size_t e = why.find_first_of(":(0123456789");
  std::string k = why.substr(0, std::min(e, (size_t)48));
  while (!k.empty() && k.back() == ' ') k.pop_back();
  return k.empty() ? why.substr(0, 48) : k;
}

MixedExec run_mixed(const MixedStepIn& in, IMixedIssue* io, p0::MixedPlan* plan, std::string* why, MixedExecStats* st) {
  auto refuse = [&](const std::string& w) {
    *why = w;
    if (st) { ++st->refused; ++st->refusals[refusal_class(w)]; }
    return MixedExec::REFUSED;
  };
  if (st) ++st->steps;
  if (!in.snap || !in.rb || !in.tables || !io) return refuse("executor: missing inputs");
  // ---- phase 1a: plan only (tb_tail untouched): every structural refusal
  using Clk = std::chrono::steady_clock;
  auto us = [](Clk::time_point t0) { return std::chrono::duration<double, std::micro>(Clk::now() - t0).count(); };
  auto t0 = Clk::now();
  p0::build_mixed(*in.snap, *in.rb, *in.tables, in.prev, nullptr, in.mixed_steps_before, plan);
  if (st) st->times["build_mixed plan-only"].add(us(t0));
  if (plan->status != p0::MixedStatus::OK) return refuse("build_mixed: " + plan->err);
  if (!plan->native_refusal.empty()) return refuse(plan->native_refusal);
  for (const p0::MixedOp& o : plan->ops)                // every op must have a native issue function
    if (!io->covers(o.kind)) return refuse(std::string("op ") + p0::mixed_op_name(o.kind) + " not native");
  if (!in.tb_tail) return refuse("tb_tail not shared (ENG_CXX_TB off / check)");
  {
    std::string w = io->prepare_plan(*plan, in);
    if (!w.empty()) return refuse(w);
  }
  // ---- phase 1b: the real build (T tables through the shared tb_tail, blobs) + the owners' host builders
  std::shared_ptr<const void> saved = in.tb_save ? in.tb_save() : nullptr;
  t0 = Clk::now();
  p0::build_mixed(*in.snap, *in.rb, *in.tables, in.prev, in.tb_tail, in.mixed_steps_before, plan);
  if (st) st->times["build_mixed real"].add(us(t0));
  std::string p = plan->status != p0::MixedStatus::OK ? "build_mixed: " + plan->err : io->prepare(*plan, in);
  if (!p.empty()) {
    if (in.tb_restore && saved) {                       // the fallback rebuilds T's tables from the same D12 state
      in.tb_restore(saved);
      return refuse(p);
    }
    *why = "refused after tb_tail moved (no save / restore): " + p;
    if (st) ++st->fatal;
    return MixedExec::FATAL;
  }
  // ---- phase 2: issue in engine order
  t0 = Clk::now();
  {
    std::string e;
    if (io->begin_issue(*plan, &e) != Status::OK) {
      *why = "begin_issue: " + e;
      if (st) ++st->fatal;
      return MixedExec::FATAL;
    }
  }
  for (const p0::MixedOp& o : plan->ops) {
    std::string e;
    if (io->issue(o, *plan, &e) != Status::OK) {
      *why = std::string("issue ") + p0::mixed_op_name(o.kind) + (o.rank >= 0 ? " rank " + std::to_string(o.rank) : "") +
             (o.role ? std::string(" ") + o.role : "") + ": " + e;
      if (st) ++st->fatal;
      return MixedExec::FATAL;
    }
    if (st) { ++st->ops; ++st->op_count[p0::mixed_op_name(o.kind)]; }
  }
  if (st) { ++st->native; st->times["phase 2 issue"].add(us(t0)); }
  why->clear();
  return MixedExec::NATIVE;
}

}  // namespace core2b
}  // namespace fpr
