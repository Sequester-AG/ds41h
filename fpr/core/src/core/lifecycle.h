// lifecycle.h -- the request lifecycle of the engine core: what adapter/phys_scheduler.py
// (PhysScheduler / PhysAsyncScheduler) and the vLLM Scheduler / AsyncScheduler code it inherits do outside the rank
// scheduler, in C++, decision for decision:
//   add_request, finish_requests (aborts, errors)
//   _schedule, _update_after_schedule (sync / async placeholders), update_from_output (spec-decode accounting
//   check_stop, stop trimming, grammar accept, prefill stats, logprobs slicing, outputs, error outputs, scheduler stats),
//   the rs.update `accepted` counts, get_grammar_bitmask, update_draft_token_ids (+ _in_output, the async drafts path).
// Images: content keys from the adapter (add_request's images). FORCE (DS41H_FORCE_FILE): the adapter matches the force
// file at add_request and passes the forced ids; the rank scheduler's `forced` flag, the forced row tokens / emitted ids /
// ForcedReq counters are here (row_tokens), the dump stays in Python (adapter/force.py ForcedReq, the executor).
// Not here (a follow-up, rejected with vLLM's error path instead): streaming input / resumable requests, LoRA, pooling,
// prompt embeds, KV connectors, DP waves.
// Time is an input everywhere (no clock reads): `mono` = time.monotonic(), `wall` = time.time() of the caller.
#pragma once
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "scheduler.h"   // fpr/core/include (on the include path)
#include "tables.h"
#include "engine.h"
#include "../grammar/grammar.h"
#include "../v1wire/v1wire.h"

namespace fpr {
namespace core {

// vllm/v1/request.py RequestStatus (same values)
enum class RS : int {
  WAITING = 1, WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, WAITING_FOR_REMOTE_KVS, WAITING_FOR_STREAMING_REQ, RUNNING,
  PREEMPTED, FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED, FINISHED_ERROR,
  FINISHED_REPETITION
};
inline bool is_finished(RS s) { return (int)s > (int)RS::PREEMPTED; }
int finish_reason_of(RS s);   // v1::FinishReason or -1

struct VReq {                           // the vLLM Request fields the core keeps (one per live request)
  Rid rid;
  int64_t client_index = 0;
  int64_t priority = 0;
  double arrival_time = 0.0;
  std::vector<int32_t> prompt;          // prompt_token_ids
  std::vector<int32_t> all;             // _all_token_ids (prompt + outputs)
  int64_t num_prompt = 0;
  v1::SamplingParams sp;
  int64_t max_tokens = 0;               // request.max_tokens
  RS status = RS::WAITING;
  int64_t num_computed_tokens = 0, num_output_placeholders = 0, num_in_flight_tokens = 0, num_stale_output_tokens = 0;
  bool drop_stale_output = false;
  std::vector<int32_t> spec_token_ids;
  bool is_prefill_chunk = false;
  int64_t num_preemptions = 0;
  v1::StopReason stop_reason;
  std::vector<v1::Event> events;
  std::optional<v1::PrefillStats> prefill_stats = v1::PrefillStats();
  std::optional<v1::RequestSpecDecodeMetrics> spec_decode_metrics;
  std::optional<v1::StrMap> trace_headers;
  int64_t replay_start = 0;
  int64_t num_nans_in_logits = 0;
  bool skip_reading_prefix_cache = false;
  std::string cache_salt;
  std::shared_ptr<grammar::StructuredReq> so;   // structured outputs (use_structured_output)
  std::vector<Image> images;            // (offset, length, content key) of the prompt's images, by offset
  std::vector<Hash> block_hashes;       // full blocks of all tokens (prompt + outputs)
  std::vector<int32_t> forced_ids;      // FORCE: the force file's "forced" ids (the adapter matched the prompt)
  bool force_match = false;             // matched at add_request: to_rs sends Request.forced, keeps it if rs accepts
  bool forced = false;                  // = PhysScheduler.forced has the rid (a ForcedReq): rs accepted the forced request
  // num_output_placeholders as PhysScheduler._phys_step saw it: the PhysStep is built BEFORE _update_after_schedule
  // (which adds this step's placeholders); row_tokens reads it for the step it was taken in (ph_step)
  int64_t ph_sched = 0, ph_step = -1;
  size_t hashes_sent = 0;               // block_hashes already given to the rank scheduler

  int64_t num_tokens() const { return (int64_t)all.size(); }
  int64_t num_output_tokens() const { return (int64_t)all.size() - num_prompt; }
  bool use_so() const { return (bool)so; }
};

struct LifecycleConfig {
  int32_t num_spec_tokens = 3;          // vllm_config.num_speculative_tokens (k)
  int64_t max_model_len = 655360;
  int32_t page_tokens = 64;             // hash_block_size
  bool async_mode = true;               // PhysAsyncScheduler (--async-scheduling)
  bool log_stats = true;                // not --disable-log-stats
  std::string spec_decode_metrics_level = "none";   // observability per_request_spec_decode_metrics
  bool include_finished_set = false;    // EngineCoreProc: data_parallel_size > 1
  bool grammar_hold = true;             // PHYS_ASYNC_GRAMMAR=hold (async only)
  double grammar_max_frac = 0.34;       // PHYS_ASYNC_GRAMMAR_MAX_FRAC
  int32_t vocab_size = 0;
  bool images_supported = false;        // PHYS_MM=1 + worker image support (images are a follow-up: rejected here)
  bool has_sampling_caps = false;       // bridge.get_sampling_caps() is not None
  std::set<std::string> sampling_caps;
  bool enable_in_reasoning = false;     // structured_outputs_config.enable_in_reasoning
  // the reasoning parser's token rules per reasoning_parser_kwargs: key "" = None / {}, else json.dumps(kwargs)
  // (tests/gen_grammar.py reasoner_specs()); empty = no reasoning parser (vLLM: reasoner_cls None)
  std::vector<std::pair<std::string, grammar::ReasonerSpec>> reasoners;
  double t0_mono = 0.0;                 // PhysScheduler._t0 (time.monotonic() at construction)
  // test hook (replays): a compiled grammar counts as ready only when this says so (the recorded compile timing)
  std::function<bool(const Rid&)> grammar_gate;
};

// the KV facade calls (PhysKVFacade) that read the KV manager outside IScheduler
struct KVFacade {
  std::function<double()> usage;                              // sum(num_used) / sum(num_pages) over the paged pools
  std::function<int64_t(const Rid&)> estimate_cached_tokens;  // kv.seqs[rid].num_registered * page_tokens, 0 if none
  std::function<bool()> reset_prefix_cache;                   // rs idle -> a fresh KV manager (IScheduler::reset_kv)
};

// The facade on the rank scheduler's KV manager (kv.h v4.1 paged_usage / estimate_cached_tokens). `kv` is a getter
// because reset_prefix_cache swaps the manager (IScheduler::reset_kv); `reset` performs that reset (rs idle check
// included), nullptr = unsupported (returns false).
KVFacade kv_facade_of(std::function<const IKVManager*()> kv, std::function<bool()> reset = nullptr);

// the block hasher (vLLM's request_block_hasher): full blocks of `all`, appended from index first
using BlockHasher = std::function<void(const VReq& r, std::vector<Hash>* hashes)>;

// ------------------------------------------------------------------ step I/O
struct ReqInit {                      // bridge.ReqInit (+ the engine's sampling subset: interfaces.sampling_dict)
  Rid rid; int32_t rank; std::vector<int32_t> prompt, output; bool structured;
  Sampling sampling;                  // 2b's NativeWorker (SeqInit.sampling); filled as to_rs fills Request.sampling
};
struct ForcedEmit {                     // PhysStep.forced[rid] (the ids the step emits) + the ForcedReq counter increments
  Rid rid;
  std::vector<int32_t> ids;
  int64_t checked_rows = 0, input_mismatch = 0;
};
struct SchedOut {                       // SchedulerOutput (the fields the executor / engine read) + PhysStep's host part
  int64_t phys_step = -1;               // plan.step, -1 = no plan
  bool has_plan = false;
  StepPlan plan;
  std::vector<std::pair<Rid, int64_t>> num_scheduled;           // insertion order (= req_ids order of the step)
  std::vector<std::pair<Rid, std::vector<int32_t>>> spec;       // scheduled_spec_decode_tokens, insertion order
  std::vector<std::pair<Rid, int64_t>> num_invalid_spec_tokens; // update_draft_token_ids_in_output
  int64_t total = 0;
  std::vector<Rid> finished_req_ids, preempted_req_ids;         // (sets: insertion order here)
  std::vector<Rid> phys_preempted;
  std::vector<ReqInit> new_reqs;
  bool has_structured_output_requests = false, pending_structured_output_tokens = false;
  int32_t num_spec_tokens_to_schedule = 0;
  std::vector<std::pair<Rid, bool>> forced_to_rs;   // FORCE: (rid, rs accepted) of the matched requests _to_rs saw
};
struct LpRow { std::vector<int64_t> ids; std::vector<float> vals; int64_t rank; };   // one generated token's row
struct ModelOut {                       // ModelRunnerOutput as merge_results builds it
  std::vector<Rid> req_ids;             // = num_scheduled order
  std::vector<std::vector<int32_t>> sampled;
  bool has_logprobs = false;            // LogprobsLists: rows in req order, cu = first row per request
  int64_t lp_width = 0;
  std::vector<LpRow> lp_rows;
  std::vector<int64_t> lp_cu;
  bool has_drafts = false;              // phys_drafts / take_draft_token_ids (DraftTokenIds)
  std::vector<std::pair<Rid, std::vector<int32_t>>> drafts;
};
struct GrammarOut {                     // GrammarOutput
  std::vector<Rid> rids;
  std::vector<int32_t> bitmask;         // rows x words
  int64_t rows = 0, words = 0;
  std::vector<int64_t> num_acceptable_drafts;
};
struct UpdateOut {
  std::vector<std::pair<int64_t, v1::EngineCoreOutputs>> outputs;   // client index -> EngineCoreOutputs (dict order)
  std::vector<std::pair<Rid, int32_t>> accepted;                    // what went to rs.update
  bool called_rs_update = false;
};

class Lifecycle {
 public:
  Lifecycle(const LifecycleConfig& cfg, IScheduler* rs, KVFacade kv, BlockHasher hasher, grammar::Manager* som);
  // EngineCore.add_request(preprocess_add_request(request)); mono/wall: the clocks at the call
  // images: the request's images with content keys ((offset, length, key), by offset), computed by the adapter (2a:
  // adapter/mm.py from vLLM's processed tensors); mm_reject: the adapter's rejection reason for its multimodal inputs
  // ("" = none). Without either, a request carrying mm_features is rejected as text-only.
  void add_request(const v1::EngineCoreRequest& r, double mono, double wall, const std::vector<Image>* images = nullptr,
                   const std::string& mm_reject = "", const std::vector<int32_t>* forced = nullptr);
  // EngineCore.abort_requests -> finish_requests(ids, FINISHED_ABORTED)
  void abort_requests(const std::vector<Rid>& rids);
  // wall: time.time() at the call (the rank scheduler's arrival = arrival_time - wall0, wall0 = wall - (mono - t0));
  // NaN = pass the raw arrival_time (replays: the rank-scheduler stand-in does not read it)
  SchedOut schedule(double mono, double wall = std::nan(""));
  GrammarOut get_grammar_bitmask(const SchedOut& so);
  // update_from_output: mono_rs = the clock read for rs.update's `now`; ts_outputs[i] = EngineCoreOutputs i's timestamp
  UpdateOut update_from_output(const SchedOut& so, const ModelOut& mo, double mono_rs,
                               const std::vector<double>& ts_outputs);
  // the same, timestamps drawn one per EngineCoreOutputs message as it is created (time.monotonic() in __post_init__)
  UpdateOut update_from_output(const SchedOut& so, const ModelOut& mo, double mono_rs,
                               const std::function<double()>& next_ts);
  void update_draft_token_ids(const std::vector<std::pair<Rid, std::vector<int32_t>>>& drafts);
  void update_draft_token_ids_in_output(const std::vector<std::pair<Rid, std::vector<int32_t>>>& drafts, SchedOut* so);
  bool has_requests() const { return !requests_.empty() || !finished_req_ids_.empty(); }
  int64_t num_unfinished() const;
  bool reset_prefix_cache();
  enum class Pause { UNPAUSED, PAUSED_NEW, PAUSED_ALL };
  void set_pause(Pause p) { pause_ = p; }
  // the row token ids of a step (PhysStep.row_tokens / prev_tokens), from the batches ITables built for so.plan; emits:
  // PhysStep.forced (dict order: first insertion, last value) + the ForcedReq counter increments (nullptr: no FORCE)
  void row_tokens(const std::vector<StepBatch>& batches, std::vector<std::vector<int32_t>>* rows,
                  std::vector<std::pair<Rid, std::vector<int32_t>>>* prev, std::vector<ForcedEmit>* emits = nullptr);
  const VReq* get(const Rid& rid) const;
  std::vector<Rid> running_ids() const;
  int64_t current_step() const { return current_step_; }
  // PhysScheduler.phys_stats counters the lifecycle owns (same names, same increments)
  struct Counters { int64_t grammar_held = 0, grammar_hold_off = 0, grammar_host_ids = 0, grammar_resume_no_ids = 0,
                    settled_host_t0 = 0, deferred_steps = 0, rejected = 0; };
  const Counters& counters() const { return ctr_; }

 private:
  struct Q {                            // vLLM's FCFS request queue mirror (counts, membership)
    std::deque<VReq*> d;
    void add(VReq* r) { d.push_back(r); }
    void prepend(VReq* r) { d.push_front(r); }
    bool remove(VReq* r);
    size_t size() const { return d.size(); }
  };
  std::string unsupported(const v1::EngineCoreRequest& r, bool images_given) const;
  void reject(VReq* r, const std::string& why);
  void to_rs(VReq* r, double wall0);
  Sampling engine_sampling(const VReq* r, int64_t max_tokens) const;   // the Sampling subset of r's SamplingParams
  void enqueue_waiting(VReq* r);
  bool handle_blocked(VReq* r);
  void preempt(VReq* r, double ts);
  void free_request(VReq* r);
  void phys_free(const Rid& rid);
  std::vector<VReq*> finish_requests(const std::vector<Rid>& ids, RS status);
  bool check_stop(VReq* r);
  void hold_structured();
  void after_update_drafts(const ModelOut& mo);
  void record_event(VReq* r, int type, double ts);
  void append_output(VReq* r, int32_t tok);
  grammar::ReqView view(VReq* r) { return grammar::ReqView{r->so.get(), &r->prompt, &r->all}; }

  LifecycleConfig cfg_;
  IScheduler* rs_;
  KVFacade kv_;
  BlockHasher hasher_;
  grammar::Manager* som_;
  const grammar::ReasonerSpec* reasoner_for(const mp::Val& kwargs) const;

  std::unordered_map<Rid, std::unique_ptr<VReq>> requests_;
  std::vector<VReq*> running_;
  Q waiting_, kv_holding_waiting_;
  std::set<VReq*> deferred_waiting_;
  std::vector<Rid> finished_req_ids_;   // set (insertion order)
  std::map<int64_t, std::vector<Rid>> finished_req_ids_dict_;
  std::vector<Rid> reset_preempted_req_ids_;
  std::vector<Rid> grammar_compile_error_reqs_;   // set (insertion order)
  // PhysScheduler state
  std::vector<VReq*> pending_add_;
  std::set<Rid> started_;
  bool in_update_ = false, preempting_ = false;
  std::vector<Rid> deferred_free_;      // set (insertion order)
  std::set<Rid> rs_waiting_;
  std::map<Rid, std::vector<int32_t>> dev_drafts_;
  std::set<Rid> has_drafts_;
  v1::PrefixCacheStats prefix_stats_;
  int64_t current_step_ = 0;
  Pause pause_ = Pause::UNPAUSED;
  std::vector<int32_t> spec_placeholders_;
  Counters ctr_;
  std::vector<std::pair<Rid, bool>> forced_to_rs_;   // FORCE: to_rs results of this schedule (-> SchedOut.forced_to_rs)
};

void merge_results(const SchedOut& so, const std::vector<StepResult>& results, ModelOut* mo);

// ghost fix: process-wide; ghosts = defensive saves (a gate asserts 0), pending_dropped = the
// finished requests removed from pending_add_ before their VReq was freed
int64_t lifecycle_ghosts();
int64_t lifecycle_pending_dropped();

}  // namespace core
}  // namespace fpr
