// nd_api.h --: the ND-3 API that 2b's NativeWorker calls.
//  - nd owns T_head and the per-rank "head enqueued" counters;
//  - nd routes every step as D13.dispatch does: async when its async_ok passes, else the embedded Python engine's
//    synchronous step. That fallback is drain() -> push_state -> the Python call on the engine's main thread via
//    run_python -> pull_state.
// Change anything here freely; the NativeWorker adapts.
#pragma once
#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "engine.h"           // StepResult
#include "tables.h"           // StepBatch, CompactTables
#include "engine_handover.h"
#include "../p0/inputs_build.h"   // (v3.4) p0::InputsOut

namespace fpr {
namespace nd {

struct NewSeq {               // interfaces.SeqInit
  Rid rid;
  int64_t serial = 0;         // a stable per-admission identity: HsdPrep's slot key (not id(obj))
  std::vector<int32_t> prompt, output;
  Sampling sampling;          // interfaces.sampling_dict subset (common.h)
  bool structured = false;
};
struct GrammarRows {          // StepContext.grammar[rid]: rows int32 [n_rows, words], acceptable drafts
  Rid rid;
  std::vector<int32_t> rows;
  int64_t n_rows = 0, words = 0, num_acceptable = 0;
};
struct StepHost {             // StepContext minus mm (images: embedded fallback in α)
  std::vector<NewSeq> fresh;
  std::vector<Rid> dropped;   // drop engine-side state
  std::vector<GrammarRows> grammar;
  std::vector<std::pair<Rid, std::vector<int32_t>>> prev_tokens;   // Engram context
  std::vector<Rid> order;     // informational
  std::vector<std::pair<Rid, int32_t>> num_spec;   // PhysStep.num_spec: spec tokens per scheduled request (the fake)
  // the worker ran Python's hooks.on_ctx for THIS host before dispatch (its on_ctx job: admissions
  // and drops applied to eng.hist / samp_ctx / prev_seq / structured). Then a decode step with admissions is native, and
  // nd observes the host as Python's (no native drops queued). false = as before (admissions refused, drops queued).
  bool on_ctx_done = false;
  // (jZr, API 326; FPR's on_ctx lazy replay, core2b PHYS_CXX_ONCTX_LAZY) the worker will replay this host's ADMISSIONS into
  // Python lazily (replay_view) instead of running on_ctx before dispatch: a decode step with admissions is native
  // (async_ok), and nd observes the host as native (drops queued for push_hist's J, which push_state applies with on_ctx's
  // drop semantics); a native mixed step's begin queues them too. false = as before.
  bool on_ctx_lazy = false;
};

// run fn on the engine process's main thread (the one that built the device engine), with the GIL; false = stopped.
// The 2b process provides it (fpr_core2b PyExec); nd uses it for every embedded-engine call.
using RunPython = std::function<bool(const std::function<void()>&)>;

struct Handle;                // nd's in-flight step

// Per step
//   reason = nd.async_ok(b, ct, host)
//   "" -> nd.dispatch(...) / nd.wait(...)                  (native)
//   else -> nd.embedded_begin(host) (drain + push_state); the worker's Python step; nd.embedded_end() (pull_state +
//           nd's HsdPrep slot-cache invalidation)
// async_ok = ID13Host::async_ok (incl. "page copies": Worker.copies steps) PLUS what only native decode lacks:
// grammar rows (no graph H), an HSD key / graph-L key (C1 owner) without a published graph, mm -> embedded. Engram
// decode steps stay NATIVE: async decode rows need no host lookback, so their step args + upload are in ND-1 (nd.cpp).
// StepHost is observed on EVERY step (native or embedded): nd's per-request state (serials, prompt + output token
// history for the Engram lookback, sampling) must follow admissions / drops either way.
// The handover's GIL callbacks (ensure_bucket / push_state / pull_state) are ALWAYS invoked by nd through run_python
// (the engine's main thread: its rank streams are current there, and the caching allocator tags ensure_bucket's
// allocations with that thread's stream); the shim's own GIL acquire inside is a harmless nested acquire.
// nd's half of a NATIVE MIXED STEP
// reachable across the .so boundary through IEngine2b::mixed().
// Status: BAD_ARG from can_head / begin = refused with NOTHING done (run the step embedded: embedded_begin / the Python
// step / embedded_end). Any non-OK after begin returned OK is fatal (nd's state is mid-step): INTERNAL.
// State handoff: begin does embedded_begin's push_state (nd -> Python: set indices, holders, prev-of + serial, J history)
// so the Python body sees current state; finish updates nd's own state and there is NO pull_state. Python's mirror is then
// stale exactly as after a native decode step, until the next push.
class IMixed2b {
 public:
  struct Ctx;   // valid from begin to finish
  virtual ~IMixed2b() = default;
  // No device op, no state change. Every refusal of a native mixed step: native mixed steps off (a missing piece), a
  // step already open, a context that drops and admits one request, grammar rows, structured requests, forced / score
  // sequences (Python's GateHead), HSD rows outside the vectorized layout, the H1 graph of the step's HSD key unpublished.
  virtual bool can_head(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                        std::string* why) = 0;
  // D13._fallback's front + Engine.step's serial bump: observe host (Python's on_ctx sees the drops: none queued), drain
  // (the in-flight step's results stay on its handle), push_state, epoch / skip-state resets, D12 deltas, rowexp resolve,
  // _fb_rows, the full tables (D13.tb.build), then step_serial + 1. BAD_ARG only for its own refusals, before the drain.
  virtual Status begin(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                       Ctx** c, std::string* err) = 0;
  // (v3.1) the step's HSD / H1 key text "S_pad|D_pad|NT" (hsd_key_of on host shapes; "" when it has none). No device op.
  // the worker asks mixed_cxx_head.guard_ok(hooks, key) once before begin; False = refuse (embedded), where the
  // Python head's pointer guard retires / re-captures as today.
  virtual std::string h1_key(const std::vector<StepBatch>& b) const = 0;
  // the same key for a step with host grammar rows: + "|bm" when the step has sampled masks (graph H1's
  // bitmask variant, ENG_HEAD_BM_GRAPH); the worker's pre-begin guard check uses this one. Default: h1_key(b).
  virtual std::string h1_key_for(const std::vector<StepBatch>& b, const StepHost& host) const {
    (void)host;
    return h1_key(b);
  }
  virtual const std::vector<StepBatch>& resolved(const Ctx* c) const = 0;   // rb: the batches Engine.step gets
  virtual const std::vector<RankTables>& tables(const Ctx* c) const = 0;    // the full tables of rb
  // Hooks.prepare_head(stT, rb) natively: the P0 builder (T's host ids = tables(c)[r].tokens[resolved(c)[r].tail_rows],
  // computed here), HSD._build into the next blob set, its H2D on every rank's step stream (call it after T's issue).
  virtual Status prepare_head(Ctx* c, std::string* err) = 0;
  // mixed_cxx_head.head_eager for rank r on its step stream after T: h_own[:n_s] = t.h[tail_b.sample_rows], aux, H1 graph.
  // t = bucket T's view on rank r (the handover's fill_mixed), tail_b = tail_batch(resolved(c)[r]) (T.h's row order).
  virtual Status h1_launch(Ctx* c, int r, const FprEhMixedBucketRank& t, const StepBatch& tail_b, std::string* err) = 0;
  // bucket b's
  // device tokens on rank r's step stream, after b's blob H2D: the migrated copies first (row0, n, rank, seq: tok[row0 :
  // row0 + n] = next_ids[rank][seq, :n], peer copies), then the pairs' H2D + tok.index_copy_(0, rows, next_ids[r].flat[src]).
  // tok = b's in_d + in_off_tok (b.R rows). rows / src / mig = build_mixed's InUpload.devtok (p0::build_inputs). Between
  // begin and finish; b must be current (fpr_eh_mixed_current) and of rank r. Non-OK = INTERNAL.
  virtual Status devtok(Ctx* c, int r, const FprEhMixedBucketRank& b, const std::vector<int64_t>& rows,
                        const std::vector<int64_t>& src, const std::vector<std::array<int64_t, 4>>& mig,
                        std::string* err) = 0;
  // bucket b's step inputs on rank r: io = the bucket's
  // p0::build_inputs output (build_mixed's BucketRank.up). The blob goes through nd's OWN pinned set of b's role (wait its
  // previous H2D, memcpy, H2D into b.in_d, re-record) and then devtok as above, on rank r's step stream (= _upload_v2's op
  // order). nd's sets are its own: Python's in_h sets keep alternating on the steps Python runs (no set-index handoff; in_d
  // writes are stream-ordered). b current and of rank r; io.blob = in_layout(b.R) = b.in_d's bytes <= config
  // mixed_max_rows' layout. Between begin and finish; non-OK = INTERNAL. (s["n"] / s["n_sample"] = io.n / io.n_sample.)
  virtual Status upload(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::InputsOut& io, std::string* err) = 0;
  // (v3.4 / v3.5) SET_VIDX natively = Engine::vidx_issue inside an open step: bucket b's Engine._set_vidx values
  // (p0::build_vidx, the same on every rank) into b's v0.6 vidx_d on rank r, the whole layout in ONE H2D (vidx[:vcount],
  // zeros up to W R, nv at 8 W R, nd at 8 W R + ceil8(4 W) or zeros) through nd's own pinned set on rank r's step stream.
  // vidx_d.ptr NULL or a layout mismatch: INTERNAL (can_head refuses such a plan before begin).
  virtual Status set_vidx(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::VidxOut& v, std::string* err) = 0;
  // (v3.4) the T stage's gather index natively: ix (bucket T's A rows = tail_rows of rank r) into the rank's v0.6 tail_ix
  // (one device copy per step, shared by FPR_OP_TAIL_GATHER and carry) through nd's own pinned set, on rank r's step
  // stream. tail_ix.data NULL or ix longer than it: INTERNAL.
  virtual Status set_tail_ix(Ctx* c, int r, const std::vector<int64_t>& ix, std::string* err) = 0;
  // the prev-of map exactly as
  // Hooks._device_tokens_v2 reads it at this step: nd's map with this step's serial (begin bumped it before the step opened:
  // entries produced by step_serial - 1, PREV_STEP_CHECK). Between begin and finish; false = None or no open step.
  virtual bool prev_of(const Ctx* c, const Rid& rid, int32_t* rank, int64_t* seq) const = 0;
  // the step's sync before finish, = Engine.sync() in _mixed_post: torch.cuda.synchronize(d) for every
  // device) cudaDeviceSynchronize on every rank's device. Between begin and finish; a device error -> INTERNAL (fatal).
  virtual Status sync(Ctx* c, std::string* err) = 0;
  // groups = tables.h dedup_copies + page_copy_batches over the
  // step's batches. page_copies_ok: the phase-1 check (no device op): the pagecopy ext's "pagecopy.copy" loaded, no state-pool
  // group (the Engram ring follow stays Python's), each group's pools in ranks[sr / dr].pools[] with its page_bytes, page ids
  // in [0, n_pages), n <= nd's id staging. page_copies (between begin and finish, FIRST in the step, after begin's drain;
  // v3.10b: every touched source / destination device is synchronized device-wide first, as begin's drain covers only
  // step_stream + xstream; groups are rare): per group on the DESTINATION rank's step stream, the ids through nd's own pinned set + one H2D into nd's device id buffer, then
  // pagecopy.copy (pagecopy_cabi.h) with the pools' bases; then, for Python's device syncs, one event per destination rank
  // waited on by EVERY rank's step stream (sources are not reused before the copy read them; destinations are read by the
  // step). BAD_ARG = page_copies_ok refused (nothing issued); INTERNAL = an error after issue.
  virtual bool page_copies_ok(const std::vector<PageCopyBatch>& groups, std::string* why) const = 0;
  virtual Status page_copies(Ctx* c, const std::vector<PageCopyBatch>& groups, std::string* err) = 0;
  // Both between begin and finish.
  // cabi: C ABI rule 3 for core2b (no CUDA there): CUDAGuard(ranks[r].device), args' leading FprCall (fpr_cabi.h, 16 B at
  // offset 0) = {ranks[r].step_stream, device, r}, rc = fn(args); rc != 0 -> INTERNAL "cabi rc N". fn / args NULL: BAD_ARG.
  virtual Status cabi(Ctx* c, int r, int (*fn)(const void*), void* args, std::string* err) {
    (void)c; (void)r; (void)fn; (void)args;
    if (err) *err = "nd: cabi not implemented";
    return Status::INTERNAL;
  }
  // h2d: the FprEhSets rule on rank r's step stream, nd's alternation index kept per `sets` pointer (the events are the
  // engine's persistent ones, shared with Python: safe whichever side used either set last): cudaEventSynchronize(ev[i]);
  // bytes > h[i].bytes -> INTERNAL; memcpy src -> h[i]; cudaMemcpyAsync H2D h[i] -> dst; cudaEventRecord(ev[i]); flip.
  // bytes 0: OK, nothing issued. sets / src / dst NULL, a NULL event or host buffer, dst not device memory of rank r's
  // device: BAD_ARG (nothing issued).
  virtual Status h2d(Ctx* c, int r, const FprEhSets* sets, void* dst, const void* src, int64_t bytes, std::string* err) {
    (void)c; (void)r; (void)sets; (void)dst; (void)src; (void)bytes;
    if (err) *err = "nd: h2d not implemented";
    return Status::INTERNAL;
  }
  // the host token ids of request rid at positions start-1, start-2
  // start-3 (CP shadow rids mapped to the prompt's rid, as cp.alias), -1 below position 0, from nd's per-request history
  // (prompt + outputs + the J deltas). false = no history covering start (Python would fall back to prev_tokens): the
  // owner refuses the native step. No device op; between begin and finish.
#define FPR_ND_LOOKBACK 1
  virtual bool lookback(const Ctx* c, const Rid& rid, int64_t start, int64_t out[3]) const {
    (void)c; (void)rid; (void)start; (void)out;
    return false;
  }
  // lazy push: set_lazy_push(true) once (NativeWorker init) -> begin skips push_state (the handoff
  // of nd's set indices, prev-of map + serial and the natively extracted tokens / native drops, J). Python needs it only
  // when Python runs: the caller calls push_now(c) BEFORE any Python of that step; idempotent per step (the first call pushes). A step that runs no Python never
  // pushes: the pending state accumulates (hist_pending, native drops) until the next push_now or embedded_begin (which
  // always pushes). Default bodies: off / OK (fakes compile unchanged).
#define FPR_ND_LAZY_PUSH 1
// this header's IMixed2b / IEngine2b contract: nd reports it as "nd_api" in
// IEngine2b::stats_json() (a stable slot); NativeWorker compares it with its own at init and disables native mixed on a
// mismatch. BUMP on ANY change of IMixed2b's / IEngine2b's virtuals (order, signature, added) or of a call contract.
#define FPR_ND_API_VERSION 326   /* (jZr) 326: IEngine2b::replay_view + StepHost.on_ctx_lazy appended */
  virtual void set_lazy_push(bool on) { (void)on; }
  virtual Status push_now(Ctx* c, std::string* err) {
    (void)c; (void)err;
    return Status::OK;
  }
  // After the step's sync: StepResults [W] (head_result's formula + the J history), prev-of (CP helpers excluded), rowexp
  // advance + device sync, HSD's step record reset (the next native step prepares on the host). Frees c.
  virtual Status finish(Ctx* c, std::vector<StepResult>* out, std::string* err) = 0;
};

// what nd knows of one request for core2b's replay into Python, read at a
// Python sync point BEFORE push_state's J. ids / n = nd's Req.hist minus its trailing hist_pending = what eng.hist must hold
// before push_state appends the pending tokens; valid until the next nd call. seed = the int64 core2b drew (core2b asserts
// equality; the float64 temperature stays core2b's: nd's is float32). known = false: nd has no such request (a
// bookkeeping bug: core2b reports it and falls back to the eager on_ctx).
struct ReplayEntry {
  Rid rid;
  const int32_t* ids = nullptr;
  int64_t n = 0;
  int64_t seed = 0;
  bool structured = false;
  bool known = false;
};

class IEngine2b {
 public:
  virtual ~IEngine2b() = default;
  virtual IMixed2b* mixed() { return nullptr; }   // nullptr = no native mixed steps (every mixed step embedded)
  virtual Status init(const FprEngineHandover* h, RunPython run_python, std::string* err) = 0;
  virtual std::string async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct,
                               const StepHost& host) = 0;   // "" = native; else the reason (-> embedded)
  virtual Status dispatch(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                          Handle** h, std::string* err) = 0;   // only when async_ok(...) == ""
  virtual Status wait(Handle* h, std::vector<StepResult>* out /* [W] */, std::string* err) = 0;
  virtual Status drain(std::string* err) = 0;   // every inflight step extracted + every rank's streams synchronized
  virtual Status embedded_begin(const StepHost& host, std::string* err) = 0;   // observe host; drain; push_state
  virtual Status embedded_end(std::string* err) = 0;                          // pull_state (+ slot-cache invalidation)
  virtual std::string stats_json() const = 0;
  // (jZr, API 326) read-only, no insert; out gets one entry per rid in order. Default (fakes): every entry unknown
  virtual void replay_view(const std::vector<Rid>& rids, std::vector<ReplayEntry>* out) const {
    out->clear();
    for (const auto& r : rids) {
      ReplayEntry e;
      e.rid = r;
      out->push_back(e);
    }
  }
};

}  // namespace nd
}  // namespace fpr
