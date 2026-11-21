// mixed_owner.cpp -- see mixed_owner.h. Build: -O2 -std=c++17.
#include "mixed_owner.h"

#include <cstdio>
#include <map>

namespace fpr {
namespace core2b {

namespace {
std::map<std::string, OwnerRegisterFn>& owners() {   // function-local: safe from the static-init order
  static std::map<std::string, OwnerRegisterFn> m;
  return m;
}
}  // namespace

void MixedOwnerRegistry::op(p0::MixedOpKind k, OwnerOpFn fn) {
  if (io_->covers(k) && !NdMixedIssue::host_only(k))
    std::fprintf(stderr, "core2b mixed owners: op %s registered twice (the later one wins)\n", p0::mixed_op_name(k));
  auto c = ctx_;
  io_->set(k, [c, fn](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) { return fn(c(), o, P, err); });
}

void MixedOwnerRegistry::prepare_plan(const std::string& owner, OwnerPrepFn fn) {
  auto c = ctx_;
  io_->add_prepare_plan(owner, [c, fn](const p0::MixedPlan& P, const MixedStepIn&) { return fn(c(), P); });
}

void MixedOwnerRegistry::prepare(const std::string& owner, OwnerPrepFn fn) {
  auto c = ctx_;
  io_->add_prepare(owner, [c, fn](const p0::MixedPlan& P, const MixedStepIn&) { return fn(c(), P); });
}

bool add_mixed_owner(const char* name, OwnerRegisterFn fn) {
  owners()[name] = fn;
  return true;
}

std::string register_mixed_owners(MixedOwnerRegistry& r) {
  std::string names;
  for (const auto& kv : owners()) {
    kv.second(r);
    names += (names.empty() ? "" : " ") + kv.first;
  }
  return names;
}

}  // namespace core2b
}  // namespace fpr
