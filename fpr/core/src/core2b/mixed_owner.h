// mixed_owner.h -- how an owner adds its device ops to the native mixed step.
//
// An owner writes ONE file core2b/owner_<name>.cpp (+ one line in core.mk) and touches nothing else:
//
//   #include "mixed_owner.h"
//   namespace fpr { namespace core2b {
//   namespace {
//   struct P1State { ... };                       // the step's host products (phase 1b -> phase 2), one per worker
//   void reg(MixedOwnerRegistry& R) {
//     auto st = std::make_shared<P1State>();
//     R.prepare_plan("p1", [st](const MixedOwnerCtx& x, const p0::MixedPlan& P) -> std::string {
//       ... structural checks only (libs present, handover fields exported, sizes): "" = OK, else the refusal ...
//
//     R.prepare("p1", [st](const MixedOwnerCtx& x, const p0::MixedPlan& P) -> std::string {
//       ... the host builders on the REAL plan (P.A / P.T tables, blobs): fill *st; "" = OK ...
//
//     R.op(p0::MixedOpKind::UPLOAD_GENERAL, [st](const MixedOwnerCtx& x, const p0::MixedOp& o, const p0::MixedPlan& P,
//                                                 std::string* err) -> Status {
//       const FprEhMixedBucketRank* v = x.view(o.role, o.rank);
//       if (x.mx->h2d(x.c, o.rank, &v->..._up, v->....data, st->blob.data(), bytes, err) != Status::OK) return Status::INTERNAL;
//       MyArgs a{};                                 // the C ABI Args mirror: CabiCall first (nd fills it)
//
//       return x.mx->cabi(x.c, o.rank, st->fn, &a, err);
//
//   const bool registered = add_mixed_owner("p1", &reg);
//   }  // namespace
//   }}  // namespace fpr::core2b
//
// Rules:
//  - prepare_plan (phase 1a: plan only, tb_tail untouched; the step's views are already filled): every refusal you can make
//    without the real build goes HERE. A refusal = the step runs embedded, nothing moved.
//  - prepare (phase 1b: after the real build): a refusal here restores tb_tail through ITables::restore_state, but only
//    when the handover's tables_abi matches (else the step is FATAL): refuse in phase 1a whenever you can.
//  - op (phase 2): issue only; any non-OK is FATAL (the engine dies). Nothing may fail here that phase 1 could have seen.
//  - x is valid only inside the call (the step's nd context); keep host products in your own state object (one
//    NativeWorker per process: one state object per registration is enough; the executor runs one step at a time).
//  - An op kind can have one handler: a second registration replaces it (stderr note). Registration order = owner name.
//  - Refusal texts are prefixed with the owner name by the executor ("p1: ..."): return the reason only.
//  - Unit test: tests/mixed_owner_test.cpp (registry + NdMixedIssue dispatch, CPU).
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "cabi_libs.h"
#include "engine_handover.h"
#include "mixed_issue_nd.h"
#include "nd_api.h"

namespace fpr {
namespace core2b {

struct MixedOwnerCtx {                  // valid only inside a hook / op call
  nd::IMixed2b* mx = nullptr;           // nd's mixed interface: cabi / h2d / set_tail_ix / ...
  nd::IMixed2b::Ctx* c = nullptr;       // the open step
  const FprEngineHandover* h = nullptr;
  const p0::MixedSnap* snap = nullptr;
  std::function<const FprEhMixedBucketRank*(char role, int rank)> view;   // this step's views ('A' / 'T'); nullptr = none
  CabiLibs* libs = nullptr;             // the owners' C ABI tables (handover libs[])
};
using OwnerOpFn = std::function<Status(const MixedOwnerCtx& x, const p0::MixedOp& op, const p0::MixedPlan& plan,
                                       std::string* err)>;
using OwnerPrepFn = std::function<std::string(const MixedOwnerCtx& x, const p0::MixedPlan& plan)>;

class MixedOwnerRegistry {
 public:
  MixedOwnerRegistry(NdMixedIssue* io, std::function<MixedOwnerCtx()> ctx) : io_(io), ctx_(std::move(ctx)) {}
  void op(p0::MixedOpKind k, OwnerOpFn fn);
  void prepare_plan(const std::string& owner, OwnerPrepFn fn);
  void prepare(const std::string& owner, OwnerPrepFn fn);

 private:
  NdMixedIssue* io_;
  std::function<MixedOwnerCtx()> ctx_;
};

using OwnerRegisterFn = void (*)(MixedOwnerRegistry& r);
// static-init registration from an owner's file (returns true: assign it to a namespace-scope const bool)
bool add_mixed_owner(const char* name, OwnerRegisterFn fn);
// NativeWorker::set_mixed_exec: every registered owner, in name order; returns their names (the stats line)
std::string register_mixed_owners(MixedOwnerRegistry& r);

}  // namespace core2b
}  // namespace fpr
