// native_worker.h -- 2b's IWorker over NativeDecode: serve/worker.py Worker.step's compact
// path in C++, with no Python per step.
//   - CompactTables per rank from this worker's own TableBuilder. The instance
//     state (sent / cstate) tracks what the device tables hold; the worker owns it.
//   - nd::IEngine2b::dispatch / wait (nd routes async vs the embedded fallback).
// Routing: reason = nd.async_ok(b, ct, host); "" -> nd.dispatch / wait (native)
// else nd.embedded_begin(host) -> the embedded Python engine step (EmbeddedFn: Worker.copies + backend.step_compact with
// THESE CompactTables, on the engine's main thread) -> nd.embedded_end(). The compact tables are built once, here, for
// every step: this worker's TableBuilder is the only one whose sent / cstate track the device (as Python's Worker.step
// builds them for every step, native or fallback).
// Images: a step whose batches have image rows (a prefill entry with images: SeqEntry.images =
// the prompt's images overlapping it, CP helper sub-chunks included) runs EMBEDDED with reason "mm" whatever
// nd.async_ok says; the EmbeddedFn plans + encodes them in Python (MMPlanner / EncoderCache).
// Native mixed steps, with
// cfg.mixed_native and nd->mixed() != nullptr: a step with prefill rows (no images) whose can_head + begin agree runs
//   the H1 guard (Python, MixedGuardFn) -> begin (nd: D13._fallback's front + push_state + serial) -> MixedBodyFn (Python: Worker.copies, StepContext,
//   Engine.step's prologue, Engine._mixed_body(prep_head=False), exporter.ensure_mixed) -> prepare_head (nd) -> per rank
//   fill_mixed (the handover's T view) + tail_batch + h1_launch (nd) -> MixedPostFn (Python: Engine._mixed_post(
//   head_eager=False): end events + sync) -> finish (nd) -> PostEmbedded (graphs the step captured) -> a COMPLETED handle.
// can_head false / begin BAD_ARG: nothing happened, the step runs embedded as before. Any failure after begin: fatal.
// Not here: Worker.poison (debug).
#pragma once
#include <atomic>
#include <random>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "core.h"
#include "engine_handover.h"
#include "cabi_libs.h"
#include "driver_bridge.h"
#include "mixed_exec.h"
#include "mixed_issue_nd.h"
#include "mixed_owner.h"
#include "span_program.h"
#include "native_mixed.h"
#include "nd_api.h"

namespace fpr {
namespace core2b {

struct NativeWorkerConfig {           // the serving worker's TableBuilder config (serve/worker.py, read by Python at init)
  int32_t k1 = 4;                     // k + 1
  bool draw_seeds = false;            // nd: NewSeq without a seed -> a drawn int64 (nd_wrap's _nd_host rule, per admission)
  // PHYS_CXX_ONCTX_LAZY=1: a NATIVE step's admissions skip Python's hooks.on_ctx
  // (nd's observe_requests is the authoritative copy); they are queued and replayed into Python (MixedReplayFn) right before
  // the next push into Python (push_now_, embedded_begin). Drops ride nd's J entries. PHYS_CXX_ONCTX_LAZY_CHECK=1 also runs
  // the eager on_ctx into Python's shadow state and compares at each replay (core2b_proc).
  bool onctx_lazy = false;
  bool onctx_lazy_check = false;
  std::vector<int> sources{2, 8, 14, 20};
  bool stub_indexer = false, full_block_tables = false;
  bool mixed_native = false;
  bool mixed_exec = false;            // the whole step through
                                      // core2b::run_mixed when every op is native
  bool tail_bound = true;             // engine.TAIL_BOUND (ENG_TAIL_BOUND) for tail_batch
  bool mixed_async = false;           // L4 (PHYS_CXX_MIXED_ASYNC=1, with mixed_exec; default off): a native mixed step that
                                      // ran no Python since before its begin ends at its LAST SYNC op when only host no-ops
                                      // follow (not CP steps: CP_DONE): that sync + nd's finish run at the next dispatch
                                      // (before anything else there) or at the step's wait, so the loop's host work (update,
                                      // schedule, build_input of the next step) overlaps the step's GPU time. The same nd
                                      // calls in the same order: only the host's sync point moves.
};

// the embedded step: the Python engine's synchronous step for these batches / tables / host, on the engine's main thread
using EmbeddedFn = std::function<Status(const StepInput& in, const std::vector<CompactTables>& cts,
                                        const nd::StepHost& host, std::vector<StepResult>* out)>;

using MmFrees = std::function<void(const std::vector<Rid>& finished, const std::vector<Rid>& preempted)>;
// after an embedded step's nd.embedded_end (a step boundary, nothing in flight): the process publishes what the Python
// step created (exporter.ensure_bucket for new buckets + refresh_graphs: nd_wrap's per-dispatch calls, moved here)
using PostEmbedded = std::function<Status()>;
// the Python halves of a native mixed step, on the engine's main thread. Body: the step's StepInput / tables /
// host + nd's resolved batches and full tables -> the step's mixed buckets (R_A, R_T, T's key text) after
// Engine._mixed_body + exporter.ensure_mixed. Post: Engine._mixed_post(head_eager=False) (its device-wide sync).
struct MixedKeys { int64_t r_a = 0, r_t = 0; std::string key_t; };
using MixedBodyFn = std::function<Status(const StepInput& in, const std::vector<CompactTables>& cts, const nd::StepHost& host,
                                         const std::vector<StepBatch>& rb, const std::vector<RankTables>& tables,
                                         MixedKeys* out)>;
using MixedPostFn = std::function<Status()>;
// Python's hooks.on_ctx for a NATIVE mixed step's admissions / drops (core2b_proc._Embedded.on_ctx: the step's
// StepContext -> eng.hooks.on_ctx: hist / samp_ctx / prev_seq / structured), once, in phase 2's begin (after every refusal
// point), after nd's push. Only for steps with host.fresh / host.dropped
using MixedOnCtxFn = std::function<Status(const StepInput& in, const nd::StepHost& host)>;
// H2: PHYS_CXX_ONCTX_CHECK's probe: run the post-step publication anyway and report what it changed
// (*changed = number of changed items: buckets ensured, graphs added / retired, buckets_epoch, the per-request dicts)
using OnCtxProbeFn = std::function<Status(int64_t* changed, std::string* what)>;
// one queued admission: ids / n = nd's replay_view (Req.hist minus hist_pending: eng.hist as it must
// be BEFORE push_state's J), valid until the next nd call; the ADD frames come from the process's Payloads (Python decodes
// the float64 SamplingParams from them, as _ctx does)
using LzRid = decltype(nd::NewSeq{}.rid);
struct LazyAdmission {
  LzRid rid;
  int64_t serial = 0, seed = 0;
  bool has_seed = false, structured = false;
  const int32_t* ids = nullptr;
  int64_t n = 0;
};
using MixedReplayFn = std::function<Status(const std::vector<LazyAdmission>& adm)>;
using MixedGuardFn = std::function<bool(const std::string& key, std::string* why)>;
// the step's mixed views at a step boundary (Python's exporter.ensure_mixed([(R_A, "A"), (R_T, "T")]) through the
// process's run_python: rule (b) point (i))
// stats = the engine counters the native steps since the last call moved (p0::MixedPlan.stats summed: mixed_steps,
// attn_pf_steps, cp_steps, ...; checklist D5): Python adds them to Engine.stats first (one crossing per native step)
using MixedEnsureFn = std::function<Status(int64_t RA, int64_t RT, const std::map<std::string, int64_t>& stats, std::string* why)>;
// engine.prev_of for the step's decode rids (nd owns prev-of in 2b; nullptr = nd's own IMixed2b::prev_of, v3.7)
using MixedPrevFn = std::function<bool(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, const Rid& rid, p0::PrevSeq* out)>;
// the device-wide sync of the step (Engine.sync: every rank's step stream) before nd's finish (nullptr = nd's own
// IMixed2b::sync, v3.8)
using MixedSyncFn = std::function<Status(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, std::string* err)>;
// nullptr = p0::moe_decide over the handover's
// moe_snap + each view's moe_bucket
using MoeVariantFn = std::function<int32_t(char role, int64_t R, int64_t vcount, int64_t ndec, int rank)>;
struct SpanEngineFlags {                // the engine switches of _issue_span (core2b_proc's snapshot)
  bool l2pf = false, nancheck = false, prof_pieces = false, tp_on = false, mega_on = false, router_pack3 = true;
  bool agrs_on = false, agrs_ag32 = false, agrs_agce = false;
  int64_t agrs_min_r = 0;
};

// rows of the step that take image embeddings (StepContext.mm)
bool has_image_rows(const std::vector<StepBatch>& b);
// any prefill (non-decode) entry: a mixed step
bool has_prefill(const std::vector<StepBatch>& b);

class NativeWorker final : public IWorker {
 public:
  NativeWorker(nd::IEngine2b* nd, const NativeWorkerConfig& cfg, EmbeddedFn embedded = nullptr, MmFrees frees = nullptr,
               PostEmbedded post = nullptr);
  ~NativeWorker();                    // L4: a deferred native mixed step is drained (sync + finish) first
  Status dispatch(const StepInput& in, StepHandle** h) override;
  Status wait(StepHandle* h, std::vector<StepResult>* out) override;
  const std::string& last_error() const { return err_; }
  int64_t n_native() const { return n_native_; }
  int64_t n_embedded() const { return n_embedded_; }
  int64_t n_embedded_mm() const { return n_mm_; }
  // the shared "previous AG bucket": "A|R" / "T|R" of the last natively issued segment if a native ISSUE ran
  // since the last take, else "" (Python keeps its own E._agrs_st_last); set_last_seg = Python's after its step ("" = none)
  std::string take_last_seg() {
    std::lock_guard<std::mutex> g(seg_mu_);
    if (!seg_native_) return std::string();
    seg_native_ = false;
    return last_seg_;
  }
  void set_last_seg(const std::string& seg) {
    std::lock_guard<std::mutex> g(seg_mu_);
    last_seg_ = seg;
    seg_native_ = false;
  }
  // the Python halves + the handover (fill_mixed)
  // (+ the nd API guard: nd's stats_json() "nd_api" must equal FPR_ND_API_VERSION, else native mixed stays off; then
  // nd's lazy push is switched on: NativeWorker pushes once per step, before the step's first Python)
  void set_mixed(MixedBodyFn body, MixedPostFn post, MixedGuardFn guard, const FprEngineHandover* h);
  int64_t n_mixed_native() const { return n_mixed_; }
  // the engine snapshot build_mixed reads (core2b_proc: record_mixed_cpu's _snap of the
  // live engine), the view ensure, nd's prev-of + sync. Registers the native handlers that exist; the other owners add theirs through
  // mixed_issue() as they land
  void set_mixed_exec(const p0::MixedSnap& snap, MixedEnsureFn ensure, MixedPrevFn prev, MixedSyncFn sync);
  // the driver, the moe variant
  // the span flags. Registers SET_STEP / ISSUE / JOIN_MIX when the bridge loads and moe is set (else they refuse)
  void set_mixed_driver(const std::string& driver_so, MoeVariantFn moe, const SpanEngineFlags& flags);
  NdMixedIssue& mixed_issue() { return issue_; }
  void set_mixed_onctx(MixedOnCtxFn f);
  // H2: hooks.on_ctx only writes per-request host dicts. An on_ctx job that was the only Python between its
  // epoch before and after (and left buckets_epoch alone) is epoch-TRANSPARENT: the ensure cache valid before it stays
  // valid, and a native mixed step whose whole epoch delta is transparent jobs skips the publication ("state-only").
  // PHYS_CXX_ONCTX_SKIP=0: off. PHYS_CXX_ONCTX_CHECK=1: state-only steps run the probe instead (= the
  // publication + what it changed; must be 0)
  void set_onctx_probe(OnCtxProbeFn f) { onctx_probe_ = std::move(f); }
  // H3: mixed_cxx_head's H1 epoch block (4 x uint64: [0] epoch, [1] verified, [2] buckets epoch at that
  // verify, [3] magic 0x48314550), owned by the Python module; 0 / no magic = no fast path.
  // PHYS_CXX_H1_EPOCH_VERIFY=1: the fast path runs the Python guard anyway and counts what it would have skipped
  void set_h1_epoch(uint64_t addr);
  // decode-side embedded steps by class (the first full text of each class once to stderr)
  std::map<std::string, int64_t> decode_refusals() const;
  void set_mixed_replay(MixedReplayFn f) { mixed_replay_ = std::move(f); }   // on_ctx lazy's replay
  int64_t lazy_queued() const { return (int64_t)lazy_q_.size(); }
  // L3 (engine-loop plan): the process's count of Python jobs run (PyExec::runs: nd's run_python included). With it,
  // NativeWorker skips the Python calls whose answer cannot have changed since no Python ran: the H1 guard (per passed
  // key), ensure_mixed (same R_A / R_T, buckets_epoch unchanged) and the post-step publication (a native step that ran no
  // Python). Unset: only NativeWorker's own Python calls are counted (tests)
  void set_py_epoch(std::function<uint64_t()> f) { py_epoch_fn_ = std::move(f); }
  int64_t n_mixed_exec() const { return n_exec_; }
  int64_t n_mixed_async() const { return n_async_; }   // L4: native mixed steps whose sync + finish were deferred
  MixedExecStats mixed_exec_stats() const;
  std::map<std::string, StepTimer> timers() const;
  std::map<std::string, int64_t> mixed_refusals() const;   // can_head / begin refusal reasons -> counts

 private:
  Status dispatch_(const StepInput& in, StepHandle** h);
  Status wait_(StepHandle* h, std::vector<StepResult>* out);
  // BAD_ARG = begin refused (nothing done: run embedded); INTERNAL = fatal (nd mid-step)
  void note_refusal(const std::string& why);
  void note_decode_refusal(const std::string& why);
  bool h1_guard_ok_(nd::IMixed2b* mx, const std::vector<StepBatch>& b, const nd::StepHost& host, std::string* why);
  Status mixed_step_(nd::IMixed2b* mx, const StepInput& in, const std::vector<CompactTables>& cts, const nd::StepHost& host,
                     std::vector<StepResult>* out);
  // *native = true = the step ran and finished natively
  // Non-OK = fatal.
  Status exec_step_(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, std::vector<StepResult>* out, bool* native);
  const FprEhMixedBucketRank* view_(char role, int r) const;
  nd::IEngine2b* nd_;                 // not owned (the process owns nd; it outlives the worker)
  NativeWorkerConfig cfg_;
  EmbeddedFn embedded_;
  MmFrees frees_;
  PostEmbedded post_;
  MixedBodyFn mixed_body_;
  MixedPostFn mixed_post_;
  MixedGuardFn mixed_guard_;
  const FprEngineHandover* h_ = nullptr;
  TableBuilder tb_;
  std::mt19937_64 seeds_{std::random_device{}()};   // a new sequence without a seed gets one HERE (nd + Python agree)
  int64_t serial_ = 0;
  std::atomic<int64_t> n_native_{0}, n_embedded_{0}, n_mm_{0}, n_mixed_{0};   // read by stats from other threads
  NdMixedIssue issue_;
  p0::MixedSnap snap_;
  bool snap_ok_ = false;
  MixedEnsureFn ensure_;
  MixedPrevFn prev_;
  MixedSyncFn sync_;
  nd::IMixed2b* cur_mx_ = nullptr;    // the open step's nd context (handlers run inside run_mixed only)
  nd::IMixed2b::Ctx* cur_c_ = nullptr;
  std::vector<FprEhMixedBucketRank> views_a_, views_t_;
  DriverBridge drv_;
  MoeVariantFn moe_;
  SpanEngineFlags span_flags_;
  std::vector<std::vector<std::vector<SpanEntry>>> spans_;   // the plan's ISSUE programs (phase 1), in op order
  std::vector<std::vector<int32_t>> variants_;               // per ISSUE: the moe variant per rank (SET_STEP)
  CabiLibs libs_;                                            // the owners' C ABI tables (handover libs[])
  std::string owners_;                                       // the owners' op files installed (mixed_owner.h)
  MixedOnCtxFn mixed_onctx_;
  // H2 (see set_onctx_probe)
  Status onctx_call_(const StepInput& in, const nd::StepHost& host);
  OnCtxProbeFn onctx_probe_;
  bool onctx_skip_ = true, onctx_check_ = false;
  // P1: native pure-hit ensure skip (see native_worker.cc 'views')
  struct EnsCacheE { uint64_t body = 0, bep = 0; std::vector<std::string> sig; };
  std::map<std::string, EnsCacheE> ens_cache_;                 // key text -> its state at the last Python ensure
  std::vector<std::pair<int64_t, int64_t>> touch_log_;        // (R_A, R_T) of native-hit steps not yet replayed
  static constexpr size_t kTouchCap = 256;                   // flush the touch log at this many pending hits
  size_t touch_cap_ = 0;                                      // resolved cap (0 = not read yet)
  int n_touch_range_ = 0;                                     // refused hits (touch key outside 24 / 39 bits)
  uint64_t pybody_ = 0;                                       // Python bodies run so far
  int ens_native_ = -1;                                       // -1 = not read yet; 0 off; 1 on; 2 on + check
  int64_t n_ens_changed_ = 0;
  std::string view_sig_(const FprEhMixedBucketRank& v) const;
  void flush_touches_(const char* where);   // pending native-hit touches before a Python body
  bool ens_fill_(int64_t R, char role, std::vector<FprEhMixedBucketRank>* out, std::string* why);
  uint64_t step_transparent_ = 0;                            // the open mixed step's transparent epoch increments
  int64_t n_check_changed_ = 0;
  const uint64_t* h1_ep_ = nullptr;   // H3 (see set_h1_epoch)
  bool h1_verify_ = false;
  // on_ctx lazy: admissions of native steps not yet replayed into Python (live rids only: a drop cancels its
  // entry, so the queue is bounded by the live request count); flushed before any push into Python
  MixedReplayFn mixed_replay_;
  std::map<LzRid, LazyAdmission> lazy_q_;
  bool lazy_on_() const { return cfg_.onctx_lazy && (bool)mixed_replay_; }
  void lazy_queue_(const nd::StepHost& host);
  Status flush_lazy_(std::string* err);
  const StepInput* cur_in_ = nullptr;                        // the open step's input / host (phase-1 / begin hooks)
  const nd::StepHost* cur_host_ = nullptr;
  // before any Python of an open step; a no-op without it
  Status push_now_(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, std::string* err);
  nd::IMixed2b::Ctx* pushed_c_ = nullptr;                    // the step whose push already ran (push once per step)
  bool lazy_push_ = false;
  // L4 (mixed_async): the native mixed step whose last SYNC + finish are deferred (pend_.c != nullptr from exec_step_ to
  // the drain); drain_mixed_ runs them and writes the results onto the step's handle (pend_.h, the .cc's NwHandle)
  struct PendingMixed { nd::IMixed2b* mx = nullptr; nd::IMixed2b::Ctx* c = nullptr; void* h = nullptr; };
  PendingMixed pend_;
  Status drain_mixed_();
  bool sync_deferrable_(const p0::MixedOp& o, const p0::MixedPlan& P) const;
  bool defer_sync_ = false;           // set by the SYNC handler for the open step
  uint64_t step_pe0_ = 0;             // the Python epoch before the open step's begin
  uint64_t own_gen_hr_ = 0, own_gen_mem_ = 0;   // issue_.gen of HEAD_RESULT / MEM_AFTER_STEP after the worker's own set
  std::atomic<int64_t> n_async_{0};
  // L3: the Python epoch (see set_py_epoch) and what was valid at which epoch
  uint64_t py_epoch_() const { return (py_epoch_fn_ ? py_epoch_fn_() : 0) + own_py_.load(); }
  std::function<uint64_t()> py_epoch_fn_;
  std::atomic<uint64_t> own_py_{0};                          // NativeWorker's own Python calls (counted when unset too)
  uint64_t guard_epoch_ = ~0ull;                             // the epoch the passed H1 keys were checked at
  uint64_t guard_bep_ = ~0ull;                               // ... and the exporter's buckets_epoch then
  std::set<std::string> guard_keys_;
  bool ens_valid_ = false;                                   // the last ensure_mixed: keys, Python epoch, buckets epoch
  int64_t ens_ra_ = -1, ens_rt_ = -1;
  uint64_t ens_py_ = 0, ens_ep_ = 0;
  std::map<std::string, int64_t> stats_pending_;             // D5: native steps' Engine.stats deltas not yet in Python
  void tick_(const char* k, double us);
  mutable std::mutex tm_mu_;
  std::map<std::string, StepTimer> timers_;
  CabiFn tstage_fn_ = nullptr;                               // tail_gather.t_stage
  int (*tstage_args_)(const void*, void*) = nullptr;         // fpr_t_stage_args (host-only) when exported
  int (*reset_plans_)(int, const int64_t*, int) = nullptr;   // NEW_SCHEDS: fpr_attn_reset_plans (driver, host-only)
  std::vector<PageCopyBatch> pc_groups_;                     // the step's Worker.copies groups (dedup_copies +
                                                             // page_copy_batches over StepInput.batches); empty = none
  size_t span_i_ = 0, sp_i_ = 0;                             // phase 2 cursors
  std::string last_seg_;                                     // the last issued segment ("A|R" / "T|R"): AGCE ce
  // Guarded by seg_mu_ (Python thread vs core thread).
  mutable std::mutex seg_mu_;
  bool seg_native_ = false;
  int64_t mixed_steps_ = 0;
  std::atomic<int64_t> n_exec_{0};
  mutable std::mutex ex_mu_;
  MixedExecStats ex_stats_;
  std::map<std::string, StepTimer> issue_times_;   // issue_.times() snapshot for timers(), published under ex_mu_
  mutable std::mutex refusals_mu_;
  std::map<std::string, int64_t> refusals_;
  std::map<std::string, int64_t> decode_refusals_;   // under refusals_mu_
  std::string err_;
};

}  // namespace core2b
}  // namespace fpr
