// lifecycle.cc -- see lifecycle.h. Line references are to the installed vLLM
// (vllm/v1/core/sched/scheduler.py, async_scheduler.py, utils.py) and adapter/phys_scheduler.py.
#include "lifecycle.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <set>

namespace fpr {
namespace core {

int finish_reason_of(RS s) {
  switch (s) {
    case RS::FINISHED_STOPPED: return v1::FR_STOP;
    case RS::FINISHED_LENGTH_CAPPED: return v1::FR_LENGTH;
    case RS::FINISHED_ABORTED: return v1::FR_ABORT;
    case RS::FINISHED_IGNORED: return v1::FR_LENGTH;
    case RS::FINISHED_ERROR: return v1::FR_ERROR;
    case RS::WAITING_FOR_STREAMING_REQ: return v1::FR_STOP;
    case RS::FINISHED_REPETITION: return v1::FR_REPETITION;
    default: return v1::FR_NONE;
  }
}

namespace {
template <class T>
bool set_add(std::vector<T>* s, const T& x) {          // an insertion-ordered set
  if (std::find(s->begin(), s->end(), x) != s->end()) return false;
  s->push_back(x);
  return true;
}
template <class T>
bool set_has(const std::vector<T>& s, const T& x) { return std::find(s.begin(), s.end(), x) != s.end(); }
// ghost fix: process-wide counters (core2b worker_stats prints them at shutdown, also when 0)
std::atomic<int64_t> g_ghosts{0}, g_pend_dropped{0};
void note_ghost(const Rid& rid, const char* where) {
  static std::mutex mu;
  static std::set<Rid> seen;
  g_ghosts.fetch_add(1, std::memory_order_relaxed);
  bool first;
  {
    std::lock_guard<std::mutex> g(mu);
    first = seen.size() < 4096 && seen.insert(rid).second;
  }
  if (first)
    std::fprintf(stderr, "core2b lifecycle: GHOST rid %s at %s (skipped, rs abort deferred; never terminate)\n",
                 rid.c_str(), where);
}
}  // namespace

int64_t lifecycle_ghosts() { return g_ghosts.load(std::memory_order_relaxed); }
int64_t lifecycle_pending_dropped() { return g_pend_dropped.load(std::memory_order_relaxed); }

KVFacade kv_facade_of(std::function<const IKVManager*()> kv, std::function<bool()> reset) {
  KVFacade f;
  f.usage = [kv]() {   // PhysKVFacade.usage: sum(num_used) / sum(num_pages), Python int / int (correctly rounded)
    int64_t used = 0, total = 0;
    if (const IKVManager* m = kv()) m->paged_usage(&used, &total);
    return total ? (double)used / (double)total : 0.0;
  };
  f.estimate_cached_tokens = [kv](const Rid& rid) -> int64_t {
    const IKVManager* m = kv();
    return m ? m->estimate_cached_tokens(rid) : 0;
  };
  f.reset_prefix_cache = reset ? reset : std::function<bool()>([] { return false; });
  return f;
}

bool Lifecycle::Q::remove(VReq* r) {
  auto it = std::find(d.begin(), d.end(), r);
  if (it == d.end()) return false;
  d.erase(it);
  return true;
}

Lifecycle::Lifecycle(const LifecycleConfig& cfg, IScheduler* rs, KVFacade kv, BlockHasher hasher, grammar::Manager* som)
    : cfg_(cfg), rs_(rs), kv_(std::move(kv)), hasher_(std::move(hasher)), som_(som) {
  spec_placeholders_.assign((size_t)cfg_.num_spec_tokens, -1);
  if (!cfg_.async_mode) cfg_.grammar_hold = false;   // _grammar_hold = async_mode and PHYS_ASYNC_GRAMMAR == hold
}

const VReq* Lifecycle::get(const Rid& rid) const {
  auto it = requests_.find(rid);
  return it == requests_.end() ? nullptr : it->second.get();
}

std::vector<Rid> Lifecycle::running_ids() const {
  std::vector<Rid> out;
  for (auto* r : running_) out.push_back(r->rid);
  return out;
}

int64_t Lifecycle::num_unfinished() const {   // get_num_unfinished_requests
  if (pause_ == Pause::PAUSED_ALL) return 0;
  int64_t w = (int64_t)(waiting_.size() + kv_holding_waiting_.size());
  if (pause_ == Pause::PAUSED_NEW) w = (int64_t)kv_holding_waiting_.size();
  return w + (int64_t)running_.size();
}

const grammar::ReasonerSpec* Lifecycle::reasoner_for(const mp::Val& kwargs) const {
  // StructuredOutputManager._get_reasoner: reasoner_cls(tokenizer, **(reasoning_parser_kwargs or {}))
  if (cfg_.reasoners.empty()) return nullptr;
  std::string key;
  if (!(kwargs.is_nil() || (kwargs.kind == mp::Val::MAP && kwargs.a.empty()))) v1::py_json_dumps(kwargs, &key);
  for (const auto& kv : cfg_.reasoners)
    if (kv.first == key) return &kv.second;
  return &cfg_.reasoners.front().second;   // an unknown variant: the default rules
}

void Lifecycle::record_event(VReq* r, int type, double ts) {
  if (cfg_.log_stats) r->events.push_back(v1::Event{type, ts});
}

void Lifecycle::append_output(VReq* r, int32_t tok) {   // Request.append_output_token_ids + update_block_hashes
  r->all.push_back(tok);
  if (hasher_) hasher_(*r, &r->block_hashes);
}

void Lifecycle::enqueue_waiting(VReq* r) {   // _enqueue_waiting_request
  if (r->status == RS::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR || r->status == RS::WAITING_FOR_REMOTE_KVS ||
      r->status == RS::WAITING_FOR_STREAMING_REQ)
    deferred_waiting_.insert(r);
  bool holds = r->num_computed_tokens > 0 || r->status == RS::WAITING_FOR_REMOTE_KVS;   // _holds_kv_blocks
  if (holds) kv_holding_waiting_.add(r);
  else waiting_.add(r);
}

// ------------------------------------------------------------------ requests in
std::string Lifecycle::unsupported(const v1::EngineCoreRequest& r, bool images_given) const {   // PhysScheduler._unsupported
  if (r.has_mm_features && !images_given) return "multimodal inputs are not supported by the phys engine (text only)";
  if (r.has_prompt_embeds) return "prompt embeddings are not supported by the phys engine";
  if (r.has_lora_request) return "LoRA is not supported by the phys engine";
  if (r.has_pooling_params || !r.has_sampling_params) return "pooling requests are not supported by the phys engine";
  if (r.resumable) return "resumable (streaming input) requests are not supported by the phys engine";
  int64_t np = (int64_t)r.prompt_token_ids.size();
  if (np >= cfg_.max_model_len)
    return "prompt of " + std::to_string(np) + " tokens >= max_model_len " + std::to_string(cfg_.max_model_len);
  if (cfg_.has_sampling_caps) {                // bridge.sampling_features(sp) - caps, sorted
    const auto& sp = r.sampling_params;
    std::set<std::string> used;
    if (sp.top_p < 1.0) used.insert("top_p");
    if (sp.top_k > 0) used.insert("top_k");
    if (sp.min_p > 0.0) used.insert("min_p");
    if (sp.presence_penalty != 0.0 || sp.frequency_penalty != 0.0 || sp.repetition_penalty != 1.0) used.insert("penalties");
    if (sp.has_logit_bias) used.insert("logit_bias");
    if (sp.has_bad_words) used.insert("bad_words");
    if (sp.has_allowed_token_ids) used.insert("allowed_token_ids");
    if (sp.has_logprobs) used.insert("logprobs");
    if (sp.has_prompt_logprobs) used.insert("prompt_logprobs");
    if (sp.min_tokens > 0) used.insert("min_tokens");
    std::string missing;
    for (const auto& f : used)                  // std::set iterates sorted, as sorted() in Python
      if (!cfg_.sampling_caps.count(f)) missing += (missing.empty() ? "" : ", ") + f;
    if (!missing.empty()) return "sampling features not supported by the phys engine's sampler yet: " + missing;
  }
  return "";
}

void Lifecycle::add_request(const v1::EngineCoreRequest& er, double mono, double wall, const std::vector<Image>* images,
                            const std::string& mm_reject, const std::vector<int32_t>* forced) {
  (void)wall;
  auto it = requests_.find(er.request_id);
  if (it != requests_.end()) {
    // PhysScheduler: "streaming input continuations are not supported" -> _reject on the EXISTING id: the stock
    // add_request would treat it as a streaming update; PhysScheduler rejects before calling super
    reject(it->second.get(), "streaming input continuations are not supported by the phys engine");
    return;
  }
  auto up = std::make_unique<VReq>();
  VReq* r = up.get();
  r->rid = er.request_id;
  r->client_index = er.client_index;
  r->priority = er.priority;
  r->arrival_time = er.arrival_time;
  r->prompt = er.prompt_token_ids;
  r->all = er.prompt_token_ids;
  r->num_prompt = (int64_t)er.prompt_token_ids.size();
  r->sp = er.sampling_params;
  r->trace_headers = er.trace_headers;
  r->cache_salt = er.cache_salt;
  if (images) r->images = *images;
  if (forced) {                         // FORCE: the adapter's ForceSpec.match(prompt) (PhysScheduler: at _to_rs)
    r->forced_ids = *forced;
    r->force_match = true;
  }
  r->skip_reading_prefix_cache = er.sampling_params.has_skip_reading_prefix_cache && er.sampling_params.skip_reading_prefix_cache;
  r->max_tokens = er.sampling_params.has_max_tokens ? er.sampling_params.max_tokens : 16;
  r->status = RS::WAITING;
  if (er.sampling_params.structured_outputs) {   // Request.__init__: StructuredOutputRequest + WAITING_FOR_..._GRAMMAR
    auto so = std::make_shared<grammar::StructuredReq>();
    so->type = er.sampling_params.structured_outputs->type;
    so->spec = er.sampling_params.structured_outputs->spec;
    so->stop_ids = er.sampling_params.all_stop_token_ids;
    so->reasoning_ended = er.reasoning_ended;
    so->reasoner = reasoner_for(er.reasoning_parser_kwargs);
    r->so = so;
    r->status = RS::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR;
  }
  if (hasher_) hasher_(*r, &r->block_hashes);
  // preprocess_add_request: grammar_init (the compile starts here, off the loop thread)
  bool so_ok = true;
  if (r->so) {
    const auto& sop = *er.sampling_params.structured_outputs;
    if (!sop.backend.empty() && sop.backend != "xgrammar") {
      so_ok = false;   // only the xgrammar backend exists in the C++ core: fails like a compile error
      std::lock_guard<std::mutex> g(r->so->mu);
      r->so->done = true;
      r->so->error = "structured output backend " + sop.backend + " is not supported by the C++ core";
    } else if (som_ == nullptr) {
      std::lock_guard<std::mutex> g(r->so->mu);
      r->so->done = true;
      r->so->error = "structured outputs are not configured";
    } else {
      som_->grammar_init(r->so);
    }
  }
  (void)so_ok;
  // Scheduler.add_request
  enqueue_waiting(r);
  requests_[r->rid] = std::move(up);
  if (cfg_.spec_decode_metrics_level != "none") {
    v1::RequestSpecDecodeMetrics m;
    m.num_spec_tokens = cfg_.num_spec_tokens;
    m.histogram.assign((size_t)cfg_.num_spec_tokens + 1, 0);
    r->spec_decode_metrics = m;
  }
  record_event(r, 1 /* QUEUED */, mono);
  // PhysScheduler.add_request: rejections, then the pending-add list
  std::string why = mm_reject.empty() ? unsupported(er, images != nullptr) : mm_reject;
  if (!why.empty()) {
    reject(r, why);
    return;
  }
  pending_add_.push_back(r);
  if (er.abort_immediately) abort_requests({r->rid});
}

void Lifecycle::reject(VReq* r, const std::string& why) {   // PhysScheduler._reject: error output in the next update
  (void)why;   // (logged by the caller in Python; the frontend sees finish_reason "error")
  ctr_.rejected += 1;
  set_add(&grammar_compile_error_reqs_, r->rid);
}

Sampling Lifecycle::engine_sampling(const VReq* r, int64_t max_tokens) const {
  Sampling s;
  s.max_tokens = max_tokens;
  s.temperature = (float)r->sp.temperature;
  s.top_p = (float)r->sp.top_p;
  s.top_k = (int32_t)r->sp.top_k;
  s.has_seed = r->sp.has_seed;
  s.seed = r->sp.seed;
  s.logprobs = r->sp.has_logprobs ? (int32_t)r->sp.logprobs : -1;
  s.prompt_logprobs = r->sp.has_prompt_logprobs;
  s.stop_token_ids = r->sp.stop_token_ids;
  s.ignore_eos = r->sp.ignore_eos;
  s.eos_token_id = r->sp.has_eos ? r->sp.eos_token_id : -1;
  return s;
}

void Lifecycle::to_rs(VReq* r, double wall0) {   // PhysScheduler._to_rs
  int64_t k1 = std::max<int64_t>(0, cfg_.num_spec_tokens - 1);
  int64_t max_new = std::max<int64_t>(1, std::min<int64_t>(r->max_tokens, cfg_.max_model_len - r->num_prompt - k1));
  if (max_new < r->max_tokens) r->max_tokens = max_new;
  Request q;
  q.rid = r->rid;
  q.prompt_ids = r->prompt;
  q.block_hashes = r->block_hashes;
  r->hashes_sent = r->block_hashes.size();
  q.sampling = engine_sampling(r, max_new);   // prompt_logprobs -> RankScheduler Request.score
  q.arrival = std::isnan(wall0) ? r->arrival_time : r->arrival_time - wall0;   // req.arrival_time - _wall0()
  q.priority = (int32_t)r->priority;
  q.structured = r->use_so();
  q.cache_salt = r->cache_salt;
  q.images = r->images;                 // the rank scheduler salts block keys with them (ImageBlockKeys) and checks spans
  q.forced = r->force_match;            // q.forced = True (ForceSpec matched); the ForcedReq lives while rs holds it
  // private key space (skip_reading_prefix_cache): the block hashes are replaced by the hasher (see BlockHasher)
  std::string why;
  Status st = rs_->add_request(q, &why);
  if (r->force_match) {                 // rs.add_request False -> self.forced.pop(rid)
    r->forced = st == Status::OK;
    forced_to_rs_.emplace_back(r->rid, r->forced);
    if (!r->forced) r->forced_ids.clear();
  }
  if (st != Status::OK) {
    reject(r, "request too large for the phys engine's KV pools: " + (why.empty() ? "no rank can hold this request" : why));
  }
}

bool Lifecycle::handle_blocked(VReq* r) {   // _handle_blocked_waiting_request (grammar case)
  if (r->status == RS::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR) {
    if (!r->so || !r->so->ready()) return false;
    if (cfg_.grammar_gate && !cfg_.grammar_gate(r->rid)) return false;
    std::string err;
    {
      std::lock_guard<std::mutex> g(r->so->mu);
      err = r->so->error;
    }
    if (!err.empty()) {
      set_add(&grammar_compile_error_reqs_, r->rid);
      return false;
    }
    r->status = RS::WAITING;
    return true;
  }
  if (r->status == RS::WAITING_FOR_REMOTE_KVS || r->status == RS::WAITING_FOR_STREAMING_REQ) return false;
  return true;
}

// ------------------------------------------------------------------ frees / finishes
void Lifecycle::phys_free(const Rid& rid) {   // PhysScheduler._phys_free (via PhysKVFacade.free)
  dev_drafts_.erase(rid);
  has_drafts_.erase(rid);
  if (preempting_) return;
  if (in_update_) {
    set_add(&deferred_free_, rid);
    return;
  }
  ReqState st;
  if (rs_->get(rid, &st)) rs_->abort(rid);
  started_.erase(rid);
  rs_waiting_.erase(rid);
}

void Lifecycle::free_request(VReq* r) {   // Scheduler._free_request + _free_blocks (no connectors)
  set_add(&finished_req_ids_, r->rid);
  if (cfg_.include_finished_set) set_add(&finished_req_ids_dict_[r->client_index], r->rid);
  phys_free(r->rid);
  // del self.requests[rid]: the caller keeps using r until it returns, so the erase is deferred to the end of the
  // public call (graveyard_)
}

std::vector<VReq*> Lifecycle::finish_requests(const std::vector<Rid>& ids_in, RS status) {
  // request_ids = set(request_ids): iteration order of a Python str set is hash order; insertion order here (the
  // only observable effect is the order of error outputs within one step: the tests compare that tail as a set)
  std::vector<Rid> ids;
  for (const auto& x : ids_in) set_add(&ids, x);
  std::vector<VReq*> valid;
  std::set<VReq*> run_rm;
  std::vector<VReq*> wait_rm;
  for (const auto& rid : ids) {
    auto it = requests_.find(rid);
    if (it == requests_.end() || is_finished(it->second->status)) continue;
    VReq* r = it->second.get();
    valid.push_back(r);
    if (r->status == RS::RUNNING) run_rm.insert(r);
    else wait_rm.push_back(r);
  }
  if (!run_rm.empty())
    running_.erase(std::remove_if(running_.begin(), running_.end(), [&](VReq* r) { return run_rm.count(r) > 0; }),
                   running_.end());
  for (VReq* r : wait_rm) {
    waiting_.remove(r);
    kv_holding_waiting_.remove(r);
    deferred_waiting_.erase(r);
  }
  for (VReq* r : valid) {
    r->status = status;
    free_request(r);
  }
  if (!valid.empty() && !pending_add_.empty()) {   // ghost fix (1): the caller frees these VReqs (now or deferred)
    std::set<VReq*> fin(valid.begin(), valid.end());
    const size_t n0 = pending_add_.size();
    pending_add_.erase(std::remove_if(pending_add_.begin(), pending_add_.end(), [&](VReq* r) { return fin.count(r) > 0; }),
                       pending_add_.end());
    g_pend_dropped.fetch_add((int64_t)(n0 - pending_add_.size()), std::memory_order_relaxed);
  }
  return valid;
}

void Lifecycle::abort_requests(const std::vector<Rid>& rids) {
  auto done = finish_requests(rids, RS::FINISHED_ABORTED);
  for (VReq* r : done) requests_.erase(r->rid);
}

// ------------------------------------------------------------------ schedule
void Lifecycle::preempt(VReq* r, double ts) {   // Scheduler._preempt_request (drop_stale_output=False)
  phys_free(r->rid);                            // _free_request_blocks -> facade.free
  r->status = RS::PREEMPTED;
  r->num_computed_tokens = 0;
  if (!r->spec_token_ids.empty()) r->spec_token_ids.clear();
  r->drop_stale_output = r->drop_stale_output && r->num_stale_output_tokens > 0;
  r->num_stale_output_tokens = r->num_in_flight_tokens;
  r->num_output_placeholders = 0;
  r->num_preemptions += 1;
  record_event(r, 3 , ts);
  waiting_.prepend(r);
  set_add(&reset_preempted_req_ids_, r->rid);
}

void Lifecycle::hold_structured() {   // PhysScheduler._hold_structured
  std::vector<Rid> held;
  int64_t dec = 0;
  for (const auto& rid : rs_->running()) {
    ReqState st;
    if (!rs_->get(rid, &st) || st.phase != Phase::DECODE) continue;
    ++dec;
    const VReq* r = get(rid);
    if (r && r->use_so() && r->num_output_placeholders > 0 && has_drafts_.count(rid)) held.push_back(rid);
  }
  if (held.empty()) return;
  if ((double)held.size() > cfg_.grammar_max_frac * (double)dec) {
    ctr_.grammar_hold_off += 1;
    return;
  }
  int64_t nxt = rs_->step_id() + 1;
  for (const auto& rid : held) {
    ReqState st;
    if (rs_->get(rid, &st) && st.skip_until_step < nxt) rs_->hold_until(rid, nxt);
  }
  ctr_.grammar_held += (int64_t)held.size();
}

SchedOut Lifecycle::schedule(double ts, double wall) {
  current_step_ += 1;
  double now = ts - cfg_.t0_mono;
  double wall0 = std::isnan(wall) ? wall : wall - (ts - cfg_.t0_mono);
  // pending adds
  std::vector<VReq*> still;
  std::vector<VReq*> pend;
  pend.swap(pending_add_);
  std::set<VReq*> pend_seen;              // ghost fix: a VReq is to_rs'd at most once
  for (VReq* r : pend) {
    if (!pend_seen.insert(r).second) {
      note_ghost(r->rid, "schedule: duplicate pending add");
      continue;
    }
    auto it = requests_.find(r->rid);
    if (it == requests_.end() || it->second.get() != r || is_finished(r->status)) continue;
    {
      ReqState st0;
      if (r->status == RS::WAITING && rs_->get(r->rid, &st0)) {   // rs still holds an earlier instance: a ghost; drop it
        note_ghost(r->rid, "schedule: add of a rid rs already holds");   // there (before rs.schedule) and add this one
        rs_->abort(r->rid);
        started_.erase(r->rid);
        rs_waiting_.erase(r->rid);
      }
    }
    bool blocked = r->status == RS::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR || r->status == RS::WAITING_FOR_REMOTE_KVS ||
                   r->status == RS::WAITING_FOR_STREAMING_REQ;
    if (blocked && !handle_blocked(r)) {
      still.push_back(r);
      continue;
    }
    if (r->status == RS::WAITING) to_rs(r, wall0);
  }
  pending_add_ = std::move(still);
  if (cfg_.async_mode) {
    for (const auto& rid : rs_waiting_) {
      const VReq* r = get(rid);
      ReqState st;
      if (r && rs_->get(rid, &st) && st.phase == Phase::WAITING && st.num_output != r->num_output_tokens())
        rs_->sync_num_output(rid, r->num_output_tokens());
    }
  }
  SchedOut so;
  so.forced_to_rs.swap(forced_to_rs_);
  if (pause_ != Pause::PAUSED_ALL) {
    bool pn = pause_ == Pause::PAUSED_NEW;
    if (pn) rs_->set_paused_new(true);
    if (cfg_.grammar_hold) hold_structured();
    so.plan = rs_->schedule(now);
    so.has_plan = true;
    if (pn) rs_->set_paused_new(false);
  }
  // rs -> vLLM request states
  std::set<Rid> preempted;
  if (so.has_plan && !so.plan.preempted.empty()) {
    preempting_ = true;
    for (const auto& rid : so.plan.preempted) {
      auto it = requests_.find(rid);
      preempted.insert(rid);
      started_.erase(rid);
      rs_waiting_.insert(rid);
      if (it != requests_.end() && it->second->status == RS::RUNNING) {
        VReq* r = it->second.get();
        running_.erase(std::find(running_.begin(), running_.end(), r));
        preempt(r, ts);
      }
    }
    preempting_ = false;
  }
  for (const auto& rid : rs_->running()) {
    auto it = requests_.find(rid);
    if (it == requests_.end()) continue;
    VReq* r = it->second.get();
    if (r->status != RS::WAITING && r->status != RS::PREEMPTED) continue;
    ReqState st;
    rs_->get(rid, &st);
    bool resumed = r->status == RS::PREEMPTED;
    if (!waiting_.remove(r)) kv_holding_waiting_.remove(r);
    deferred_waiting_.erase(r);
    rs_waiting_.erase(rid);
    running_.push_back(r);
    r->status = RS::RUNNING;
    r->num_computed_tokens = st.replay_start;
    r->replay_start = st.replay_start;
    record_event(r, 2 /* SCHEDULED */, ts);
    if (r->prefill_stats && r->num_preemptions <= 0) {   // PrefillStats.set(num_prompt, min(cached, num_prompt), 0)
      int64_t loc = std::min<int64_t>(st.cached_tokens, r->num_prompt);
      auto& p = *r->prefill_stats;
      p.num_prompt_tokens = r->num_prompt;
      p.num_computed_tokens = r->num_prompt - loc;
      p.num_cached_tokens = loc;
      p.num_local_cached_tokens = loc;
      p.num_external_cached_tokens = 0;
    }
    if (cfg_.log_stats) prefix_stats_.record(r->num_tokens(), st.cached_tokens, resumed);
  }
  // the plan as vLLM's SchedulerOutput
  auto bump = [&](const Rid& rid, int64_t n) {
    for (auto& kv : so.num_scheduled)
      if (kv.first == rid) {
        kv.second += n;
        return;
      }
    so.num_scheduled.emplace_back(rid, n);
  };
  if (so.has_plan) {
    for (const auto& w : so.plan.work) {
      if (w.kind == "cp") {   // CP helper sub-chunk: counts for the owner's request
        bump(w.cp.rid, w.n);
        continue;
      }
      auto wit = requests_.find(w.rid);   // ghost fix (2): was requests_.at (terminate on a ghost)
      if (wit == requests_.end()) {
        note_ghost(w.rid, "schedule: plan work");
        set_add(&deferred_free_, w.rid);    // rs.abort after this plan's rs.update (update_from_output's deferred frees)
        continue;
      }
      VReq* r = wit->second.get();
      bump(w.rid, w.n);
      if (w.kind == "decode") {
        int64_t ns = w.n - 1;
        if (ns > 0) {
          std::vector<int32_t> d;
          if (r->use_so())
            for (int64_t j = 0; j < ns && j < (int64_t)r->spec_token_ids.size(); ++j) d.push_back(r->spec_token_ids[j]);
          while ((int64_t)d.size() < ns) d.push_back(-1);
          bool found = false;
          for (auto& kv : so.spec)
            if (kv.first == w.rid) { kv.second = d; found = true; }
          if (!found) so.spec.emplace_back(w.rid, d);
        }
        r->spec_token_ids.clear();
      }
      if (!started_.count(w.rid)) {
        started_.insert(w.rid);
        ReqInit ri;
        ri.rid = w.rid;
        ri.rank = w.rank;
        ri.prompt = r->prompt;
        ri.output.assign(r->all.begin() + r->num_prompt, r->all.end());
        ri.structured = r->use_so();
        ri.sampling = engine_sampling(r, r->max_tokens);
        so.new_reqs.push_back(std::move(ri));
      }
    }
    so.phys_step = so.plan.step;
  }
  for (const auto& kv : so.num_scheduled) so.total += kv.second;
  so.preempted_req_ids = reset_preempted_req_ids_;
  so.finished_req_ids = finished_req_ids_;
  so.phys_preempted.assign(preempted.begin(), preempted.end());
  so.num_spec_tokens_to_schedule = cfg_.num_spec_tokens;
  // _update_after_schedule (Scheduler); the PhysStep's row tokens see the state before it (ph_sched)
  for (const auto& kv : so.num_scheduled) {
    auto sit = requests_.find(kv.first);   // ghost fix (2) (CP helper rows bump the owner's rid)
    if (sit == requests_.end()) {
      note_ghost(kv.first, "_update_after_schedule");
      set_add(&deferred_free_, kv.first);
      continue;
    }
    VReq* r = sit->second.get();
    r->ph_sched = r->num_output_placeholders;
    r->ph_step = current_step_;
    r->num_computed_tokens += kv.second;
    r->num_in_flight_tokens += kv.second;
    r->is_prefill_chunk = r->num_computed_tokens < r->num_tokens() + r->num_output_placeholders;
    so.has_structured_output_requests |= r->use_so() && !r->is_prefill_chunk;
  }
  finished_req_ids_.clear();
  reset_preempted_req_ids_.clear();
  if (cfg_.async_mode) {   // AsyncScheduler._update_after_schedule
    spec_placeholders_.assign((size_t)so.num_spec_tokens_to_schedule, -1);
    for (const auto& kv : so.num_scheduled) {
      auto ait = requests_.find(kv.first);   // ghost fix (2) (counted once above)
      if (ait == requests_.end()) continue;
      VReq* r = ait->second.get();
      if (r->is_prefill_chunk) continue;
      so.pending_structured_output_tokens |= r->use_so() && r->num_output_placeholders > 0;
      int64_t cur = 0;
      for (const auto& s : so.spec)
        if (s.first == kv.first) cur = (int64_t)s.second.size();
      r->num_output_placeholders += 1 + cur;
      r->spec_token_ids = spec_placeholders_;
    }
    if (so.pending_structured_output_tokens) ctr_.deferred_steps += 1;   // PhysAsyncScheduler._update_after_schedule
  }
  return so;
}

void Lifecycle::row_tokens(const std::vector<StepBatch>& batches, std::vector<std::vector<int32_t>>* rows,
                           std::vector<std::pair<Rid, std::vector<int32_t>>>* prev, std::vector<ForcedEmit>* emits) {
  // PhysScheduler._phys_step's token part (FORCE: forced[n0 - 1] as the in-flight token, the next forced ids as the drafts,
  // the emitted ids, ForcedReq.checked_rows / input_mismatch)
  rows->clear();
  prev->clear();
  std::vector<ForcedEmit> own;
  if (emits == nullptr) emits = &own;   // (callers without FORCE output: same row tokens)
  emits->clear();
  auto emit_slot = [emits](const Rid& rid) -> ForcedEmit& {   // dict semantics: first insertion keeps its place
    for (auto& e : *emits)
      if (e.rid == rid) return e;
    emits->push_back(ForcedEmit{rid, {}, 0, 0});
    return emits->back();
  };
  for (const auto& sb : batches) {
    std::vector<int32_t> toks;
    for (const auto& se : sb.seqs) {
      auto rit = requests_.find(se.rid);    // ghost fix (2): keep the row shape (token 0), no prev / emit
      if (rit == requests_.end()) {
        note_ghost(se.rid, "row_tokens");
        set_add(&deferred_free_, se.rid);
        for (int64_t j = 0; j < (int64_t)se.n_rows; ++j) toks.push_back(0);
        continue;
      }
      VReq* r = rit->second.get();
      const auto& all = r->all;
      // fr = self.forced.get(se.rid); n0 = outputs incl. in-flight ones (async)
      const std::vector<int32_t>* fr = r->forced ? &r->forced_ids : nullptr;
      const int64_t ph = r->ph_step == current_step_ ? r->ph_sched : r->num_output_placeholders;   // before this step's
      int64_t n0 = r->num_output_tokens() + ph;
      if (se.kind == "decode") {
        std::vector<int32_t> hd;
        bool use_hd = false;
        int32_t t0;
        auto dd = dev_drafts_.find(se.rid);
        if (se.start >= 0 && se.start < (int64_t)all.size()) {
          t0 = all[se.start];
        } else if (fr && 1 <= n0 && n0 <= (int64_t)fr->size()) {
          t0 = (*fr)[n0 - 1];                 // forced: the host knows the in-flight token
        } else if (se.start < 0 && ph == 0 && dd != dev_drafts_.end()) {
          hd = dd->second;
          dev_drafts_.erase(dd);
          t0 = all.back();
          use_hd = (int64_t)hd.size() >= se.n_rows - 1;
          if (!use_hd) t0 = -1;
        } else if (se.start < 0 && ph == 0 && !all.empty()) {
          t0 = all.back();
          ctr_.settled_host_t0 += 1;
          if (r->use_so() && has_drafts_.count(se.rid)) ctr_.grammar_resume_no_ids += 1;
        } else {
          t0 = -1;
        }
        toks.push_back(t0);
        if (use_hd) {
          for (int32_t j = 0; j < se.n_rows - 1; ++j) toks.push_back(hd[j]);
          ctr_.grammar_host_ids += 1;
          continue;
        }
        if (fr && n0 + se.n_rows <= (int64_t)fr->size()) {   // e = fr.emit(n0, se.n_rows)
          const int32_t* e = fr->data() + n0;
          ForcedEmit& fe = emit_slot(se.rid);
          fe.ids.assign(e, e + se.n_rows);
          for (int32_t j = 0; j + 1 < se.n_rows; ++j) toks.push_back(e[j]);   // drafts := the next forced tokens
          fe.checked_rows += se.n_rows;                                        // production counts verify rows
          // fed = [t0] + e[:-1]; want = ([forced[n0 - 1]] if n0 >= 1 else [t0]) + forced[n0:n0 + n_rows - 1]: the tails are
          // the same ids (e = forced[n0:n0 + n_rows]), so only row 0 can differ
          if (n0 >= 1 && t0 != (*fr)[n0 - 1]) fe.input_mismatch += 1;
        } else {
          for (int32_t j = 0; j < se.n_rows - 1; ++j) toks.push_back(-1);
        }
      } else {
        for (int64_t j = se.start; j < se.start + se.n_rows; ++j) toks.push_back(all[j]);
        if (fr && se.last_chunk && n0 + 1 <= (int64_t)fr->size()) {   // e = fr.emit(n0, 1)
          ForcedEmit& fe = emit_slot(se.rid);
          fe.ids.assign(1, (*fr)[n0]);
        }
      }
      if (se.start >= 0 && (!se.has_cp || se.cp.owner == sb.rank)) {
        int64_t lo = std::max<int64_t>(0, se.start - 3), hi = std::min<int64_t>(se.start, (int64_t)all.size());
        std::vector<int32_t> p(all.begin() + lo, all.begin() + std::max(lo, hi));
        bool f = false;
        for (auto& kv : *prev)
          if (kv.first == se.rid) { kv.second = p; f = true; }
        if (!f) prev->emplace_back(se.rid, p);
      }
    }
    rows->push_back(std::move(toks));
  }
}

GrammarOut Lifecycle::get_grammar_bitmask(const SchedOut& so) {
  GrammarOut g;
  if (!so.has_structured_output_requests || som_ == nullptr) return g;
  std::vector<grammar::ReqView> views;
  std::vector<const std::vector<int32_t>*> spec;
  for (const auto& kv : so.num_scheduled) {
    auto it = requests_.find(kv.first);
    if (it == requests_.end()) continue;
    VReq* r = it->second.get();
    if (!r->use_so() || r->is_prefill_chunk) continue;
    g.rids.push_back(r->rid);
    views.push_back(view(r));
    const std::vector<int32_t>* s = nullptr;
    for (const auto& sk : so.spec)
      if (sk.first == r->rid) s = &sk.second;
    spec.push_back(s);
  }
  if (g.rids.empty()) return g;
  g.words = som_->backend()->bitmask_words();
  g.rows = som_->grammar_bitmask(views, spec, &g.bitmask);
  for (const auto* s : spec) {
    int64_t n = 0;
    if (s)
      for (int32_t t : *s) {
        if (t < 0) break;
        ++n;
      }
    g.num_acceptable_drafts.push_back(n);
  }
  return g;
}

// ------------------------------------------------------------------ update
bool Lifecycle::check_stop(VReq* r) {   // vllm/v1/core/sched/utils.py check_stop
  int32_t last = r->all.back();
  if (r->sp.has_eos && last == r->sp.eos_token_id) {
    r->status = RS::FINISHED_STOPPED;
    return true;
  }
  if (std::find(r->sp.stop_token_ids.begin(), r->sp.stop_token_ids.end(), last) != r->sp.stop_token_ids.end()) {
    r->status = RS::FINISHED_STOPPED;
    r->stop_reason.kind = v1::StopReason::INT;
    r->stop_reason.i = last;
    return true;
  }
  if (r->num_tokens() >= cfg_.max_model_len || r->num_output_tokens() >= r->max_tokens) {
    r->status = RS::FINISHED_LENGTH_CAPPED;
    return true;
  }
  if (r->num_output_tokens() < r->sp.min_tokens) return false;
  if (r->sp.repetition_detection) {   // check_sequence_repetition on the output ids
    const auto& p = *r->sp.repetition_detection;
    int64_t mx = p.max_pattern_size, mn = p.min_pattern_size <= 0 ? 1 : p.min_pattern_size, cnt = p.min_count;
    const int32_t* out = r->all.data() + r->num_prompt;
    int64_t n = r->num_output_tokens();
    if (mx > 0 && cnt >= 2 && mn <= mx) {
      for (int64_t L = mn; L <= mx; ++L) {
        if (L * cnt > n) break;
        bool rep = true;
        for (int64_t a = 1; a <= L && rep; ++a) {
          int32_t target = out[n - a];
          for (int64_t m = 1; m < cnt; ++m)
            if (out[n - (L * m + a)] != target) { rep = false; break; }
        }
        if (rep) {
          r->status = RS::FINISHED_REPETITION;
          r->stop_reason.kind = v1::StopReason::STR;
          r->stop_reason.s = "repetition_detected";
          return true;
        }
      }
    }
  }
  return false;
}

void Lifecycle::after_update_drafts(const ModelOut& mo) {   // PhysScheduler._after_update_drafts
  if (!mo.has_drafts) return;
  std::vector<std::pair<Rid, std::vector<int32_t>>> upd;
  for (const auto& kv : mo.drafts) {
    auto it = requests_.find(kv.first);
    VReq* r = it == requests_.end() ? nullptr : it->second.get();
    if (r == nullptr || is_finished(r->status) || !r->use_so()) {
      dev_drafts_.erase(kv.first);
      continue;
    }
    has_drafts_.insert(kv.first);
    if (r->num_output_placeholders == 0) {
      dev_drafts_[kv.first] = kv.second;
      upd.push_back(kv);
    } else {
      dev_drafts_.erase(kv.first);
    }
  }
  if (!upd.empty()) update_draft_token_ids(upd);
}

UpdateOut Lifecycle::update_from_output(const SchedOut& so, const ModelOut& mo, double mono_rs,
                                        const std::vector<double>& ts_outputs) {
  size_t ti = 0;
  return update_from_output(so, mo, mono_rs, [&]() { return ti < ts_outputs.size() ? ts_outputs[ti++] : 0.0; });
}

UpdateOut Lifecycle::update_from_output(const SchedOut& so, const ModelOut& mo, double mono_rs,
                                        const std::function<double()>& next_ts) {
  UpdateOut res;
  // PhysScheduler._phys_update: output counts before vLLM's update
  std::vector<std::pair<Rid, int64_t>> before;
  if (so.has_plan)
    for (const auto& w : so.plan.work) {
      auto it = requests_.find(w.rid);
      if (it == requests_.end()) continue;
      bool f = false;
      for (auto& b : before)
        if (b.first == w.rid) { b.second = it->second->num_output_tokens(); f = true; }
      if (!f) before.emplace_back(w.rid, it->second->num_output_tokens());
    }
  in_update_ = true;
  // ---- Scheduler.update_from_output
  std::vector<std::pair<int64_t, std::vector<v1::EngineCoreOutput>>> outs;   // client -> outputs (insertion order)
  auto out_for = [&](int64_t c) -> std::vector<v1::EngineCoreOutput>& {
    for (auto& kv : outs)
      if (kv.first == c) return kv.second;
    outs.emplace_back(c, std::vector<v1::EngineCoreOutput>());
    return outs.back().second;
  };
  std::optional<v1::SpecDecodingStats> spec_stats;
  std::vector<VReq*> stopped_running, stopped_preempted, to_erase;
  for (size_t idx_s = 0; idx_s < so.num_scheduled.size(); ++idx_s) {
    const Rid& rid = so.num_scheduled[idx_s].first;
    int64_t n_sched = so.num_scheduled[idx_s].second;
    auto it = requests_.find(rid);
    VReq* r = it == requests_.end() ? nullptr : it->second.get();
    bool stale = false;
    if (r) {
      r->num_in_flight_tokens -= n_sched;
      if (r->num_stale_output_tokens > 0) {
        stale = true;
        r->num_stale_output_tokens -= n_sched;
      }
    }
    if (r == nullptr || is_finished(r->status)) continue;
    if (stale && r->drop_stale_output) continue;
    size_t idx = 0;
    while (idx < mo.req_ids.size() && mo.req_ids[idx] != rid) ++idx;
    std::vector<int32_t> gen = idx < mo.sampled.size() ? mo.sampled[idx] : std::vector<int32_t>();
    const std::vector<int32_t>* sched_spec = nullptr;
    for (const auto& s : so.spec)
      if (s.first == rid) sched_spec = &s.second;
    if (sched_spec && !sched_spec->empty() && !gen.empty()) {
      int64_t nd = (int64_t)sched_spec->size();
      int64_t na = std::max<int64_t>((int64_t)gen.size() - 1, 0);
      int64_t nr = nd - na;
      if (!stale) {
        if (r->num_computed_tokens > 0) r->num_computed_tokens -= nr;
        if (r->num_output_placeholders > 0) r->num_output_placeholders -= nr;
      }
      int64_t inval = 0;
      for (const auto& kv : so.num_invalid_spec_tokens)
        if (kv.first == rid) inval = kv.second;
      if (cfg_.log_stats && nd) {   // make_spec_decoding_stats
        if (!spec_stats) spec_stats = v1::SpecDecodingStats::make(cfg_.num_spec_tokens);
        spec_stats->observe_draft(nd - (so.num_invalid_spec_tokens.empty() ? 0 : inval), na);
      }
      if (r->spec_decode_metrics) {
        int64_t adj = nd - (so.num_invalid_spec_tokens.empty() ? 0 : inval);
        auto& m = *r->spec_decode_metrics;
        m.histogram[(size_t)na] += 1;
        m.num_draft_tokens += adj;
        if (cfg_.spec_decode_metrics_level == "detailed") {
          m.per_step_accepted.push_back(na);
          m.per_step_drafted.push_back(adj);
        }
      }
    }
    bool stopped = false;
    std::vector<int32_t> new_ids = gen;
    RS status_before = r->status;
    if (!new_ids.empty()) {   // _update_request_with_output (+ AsyncScheduler's)
      size_t keep = new_ids.size();
      for (size_t j = 0; j < new_ids.size(); ++j) {
        append_output(r, new_ids[j]);
        if (check_stop(r)) {
          stopped = true;
          keep = j + 1;
          break;
        }
      }
      new_ids.resize(keep);
      if (cfg_.async_mode && !stale) r->num_output_placeholders -= (int64_t)new_ids.size();
    }
    if (!new_ids.empty() && r->use_so() && som_ && !som_->accept_tokens(view(r), new_ids)) {
      r->status = RS::FINISHED_ERROR;
      stopped = true;
    }
    bool emit = !new_ids.empty() || stopped;
    std::optional<v1::PrefillStats> pstats;
    if (emit && r->prefill_stats) {   // take_prefill_stats + finalize(estimate_cached_tokens)
      pstats = r->prefill_stats;
      r->prefill_stats.reset();
      int64_t c = kv_.estimate_cached_tokens ? kv_.estimate_cached_tokens(rid) : 0;
      pstats->num_cache_creation_tokens =
          std::max<int64_t>(0, std::min<int64_t>(c, pstats->num_prompt_tokens) - pstats->num_cached_tokens);
    }
    std::optional<v1::LogprobsLists> nlp;
    if (r->sp.has_logprobs && mo.has_logprobs) {   // logprobs.slice_request(req_index, len(new_token_ids))
      int64_t s0 = idx < mo.lp_cu.size() ? mo.lp_cu[idx] : (int64_t)idx;
      int64_t n = (int64_t)new_ids.size();
      int64_t e = std::min<int64_t>(s0 + n, (int64_t)mo.lp_rows.size());
      if (s0 > e) s0 = e;
      v1::LogprobsLists l;
      l.logprob_token_ids.dtype = "<i8";
      l.logprobs.dtype = "<f4";
      l.sampled_token_ranks.dtype = "<i8";
      int64_t m = e - s0;
      l.logprob_token_ids.shape = {m, mo.lp_width};
      l.logprobs.shape = {m, mo.lp_width};
      l.sampled_token_ranks.shape = {m};
      for (int64_t j = s0; j < e; ++j) {
        const LpRow& row = mo.lp_rows[(size_t)j];
        for (int64_t c = 0; c < mo.lp_width; ++c) {
          int64_t id = c < (int64_t)row.ids.size() ? row.ids[c] : 0;
          float v = c < (int64_t)row.vals.size() ? row.vals[c] : -INFINITY;
          l.logprob_token_ids.data.append((const char*)&id, 8);
          l.logprobs.data.append((const char*)&v, 4);
        }
        int64_t rk = row.rank;
        l.sampled_token_ranks.data.append((const char*)&rk, 8);
      }
      nlp = l;
    }
    int fr = v1::FR_NONE;
    if (stopped) {
      fr = finish_reason_of(r->status);
      free_request(r);                       // not resumable: finished
      to_erase.push_back(r);
      if (status_before == RS::RUNNING) stopped_running.push_back(r);
      else stopped_preempted.push_back(r);
    }
    if (emit) {
      v1::EngineCoreOutput o;
      o.request_id = rid;
      o.new_token_ids = new_ids;
      o.new_logprobs = nlp;
      o.finish_reason = fr;
      o.stop_reason = r->stop_reason;
      if (!r->events.empty()) {
        o.events = r->events;
        r->events.clear();
      }
      o.trace_headers = r->trace_headers;
      o.prefill_stats = pstats;
      o.num_nans_in_logits = r->num_nans_in_logits;
      if (fr != v1::FR_NONE) o.spec_decode_metrics = r->spec_decode_metrics;
      out_for(r->client_index).push_back(std::move(o));
    }
  }
  if (!stopped_running.empty())
    running_.erase(std::remove_if(running_.begin(), running_.end(),
                                  [&](VReq* x) {
                                    return std::find(stopped_running.begin(), stopped_running.end(), x) !=
                                           stopped_running.end();
                                  }),
                   running_.end());
  for (VReq* x : stopped_preempted) {
    waiting_.remove(x);
    kv_holding_waiting_.remove(x);
    deferred_waiting_.erase(x);
  }
  std::vector<Rid> err_ids;
  err_ids.swap(grammar_compile_error_reqs_);
  if (!err_ids.empty()) {
    auto errs = finish_requests(err_ids, RS::FINISHED_ERROR);
    for (VReq* r : errs) {
      v1::EngineCoreOutput o;
      o.request_id = r->rid;
      o.finish_reason = finish_reason_of(r->status);
      if (!r->events.empty()) {
        o.events = r->events;
        r->events.clear();
      }
      o.trace_headers = r->trace_headers;
      out_for(r->client_index).push_back(std::move(o));
      to_erase.push_back(r);
    }
  }
  // EngineCoreOutputs per client, finished sets, stats
  for (auto& kv : outs) {
    v1::EngineCoreOutputs e;
    e.outputs = std::move(kv.second);
    e.timestamp = next_ts();
    res.outputs.emplace_back(kv.first, std::move(e));
  }
  if (cfg_.include_finished_set && !finished_req_ids_dict_.empty()) {
    for (auto& kv : finished_req_ids_dict_) {
      bool f = false;
      for (auto& o : res.outputs)
        if (o.first == kv.first) { o.second.finished_requests = kv.second; f = true; }
      if (!f) {
        v1::EngineCoreOutputs e;
        e.finished_requests = kv.second;
        e.timestamp = next_ts();
        res.outputs.emplace_back(kv.first, std::move(e));
      }
    }
    finished_req_ids_dict_.clear();
  }
  if (cfg_.log_stats) {   // make_stats
    v1::SchedulerStats s;
    s.num_running_reqs = (int64_t)running_.size();
    int64_t nw = (int64_t)(waiting_.size() + kv_holding_waiting_.size());
    int64_t nd = (int64_t)deferred_waiting_.size();
    s.num_waiting_reqs = nw - nd;
    s.num_skipped_waiting_reqs = nd;
    s.num_kv_fetch_reqs_by_stage = {{"waiting_to_start", 0}, {"in_progress", 0}, {"completed_waiting", 0}};   // no connector
    s.kv_cache_usage = kv_.usage ? kv_.usage() : 0.0;
    s.prefix_cache_stats = prefix_stats_;
    prefix_stats_ = v1::PrefixCacheStats();
    s.spec_decoding_stats = spec_stats;
    if (res.outputs.empty()) {
      v1::EngineCoreOutputs e;
      e.timestamp = next_ts();
      res.outputs.emplace_back(0, std::move(e));
    }
    res.outputs.front().second.scheduler_stats = s;
  }
  in_update_ = false;
  // ---- PhysScheduler: drafts (hold), rs.update with the tokens vLLM kept, deferred frees
  if (cfg_.grammar_hold) after_update_drafts(mo);
  if (so.has_plan) {
    for (const auto& w : so.plan.work) {
      if (w.kind != "decode") continue;
      const std::pair<Rid, int64_t>* b = nullptr;
      for (const auto& x : before)
        if (x.first == w.rid) b = &x;
      if (!b) continue;
      ReqState st;
      if (!rs_->get(w.rid, &st) || st.phase == Phase::FINISHED) continue;
      auto run = rs_->running();
      if (std::find(run.begin(), run.end(), w.rid) == run.end()) continue;
      // req: the request object of `before` (still alive: deferred erase)
      int64_t now_out = -1;
      for (VReq* x : to_erase)
        if (x->rid == w.rid) now_out = x->num_output_tokens();
      if (now_out < 0) {
        auto it = requests_.find(w.rid);
        if (it == requests_.end()) continue;
        now_out = it->second->num_output_tokens();
      }
      int64_t a = now_out - b->second;
      bool f = false;
      for (auto& acc : res.accepted)
        if (acc.first == w.rid) { acc.second = (int32_t)a; f = true; }
      if (!f) res.accepted.emplace_back(w.rid, (int32_t)a);
    }
    // the outputs' new full blocks go to rs before update() commits them (2a: vLLM's block_hashes)
    for (const auto& kv : so.num_scheduled) {
      VReq* r = nullptr;
      for (VReq* x : to_erase)
        if (x->rid == kv.first) r = x;
      if (!r) {
        auto it = requests_.find(kv.first);
        if (it != requests_.end()) r = it->second.get();
      }
      if (r && r->block_hashes.size() > r->hashes_sent) {
        std::vector<Hash> add(r->block_hashes.begin() + r->hashes_sent, r->block_hashes.end());
        rs_->extend_block_hashes(r->rid, (int64_t)r->hashes_sent, add);
        r->hashes_sent = r->block_hashes.size();
      }
    }
    rs_->update(so.plan, res.accepted, mono_rs - cfg_.t0_mono);
    res.called_rs_update = true;
  }
  std::vector<Rid> df;
  df.swap(deferred_free_);
  for (const auto& rid : df) {
    ReqState st;
    if (rs_->get(rid, &st)) rs_->abort(rid);
    started_.erase(rid);
    rs_waiting_.erase(rid);
  }
  for (VReq* r : to_erase) requests_.erase(r->rid);   // del self.requests[rid] (deferred until here)
  return res;
}

void Lifecycle::update_draft_token_ids(const std::vector<std::pair<Rid, std::vector<int32_t>>>& drafts) {
  for (const auto& kv : drafts) {
    auto it = requests_.find(kv.first);
    if (it == requests_.end() || is_finished(it->second->status)) continue;
    VReq* r = it->second.get();
    if (r->is_prefill_chunk) {
      if (!r->spec_token_ids.empty()) r->spec_token_ids.clear();
      continue;
    }
    r->spec_token_ids = (r->use_so() && som_) ? som_->validate_tokens(view(r), kv.second) : kv.second;
  }
}

void Lifecycle::update_draft_token_ids_in_output(const std::vector<std::pair<Rid, std::vector<int32_t>>>& drafts,
                                                 SchedOut* so) {
  so->num_invalid_spec_tokens.clear();
  for (const auto& kv : drafts) {
    auto it = requests_.find(kv.first);
    if (it == requests_.end() || is_finished(it->second->status)) continue;
    VReq* r = it->second.get();
    std::vector<int32_t>* ph = nullptr;
    for (auto& s : so->spec)
      if (s.first == kv.first) ph = &s.second;
    if (ph == nullptr || ph->empty()) continue;
    size_t orig = ph->size();
    std::vector<int32_t> t(kv.second.begin(), kv.second.begin() + std::min(orig, kv.second.size()));
    if (r->use_so() && som_) t = som_->validate_tokens(view(r), t);
    // (validate_tokens returns spec_tokens unchanged for unstructured requests)
    size_t inval = orig - t.size();
    if (inval) {
      t.resize(orig, -1);
      so->num_invalid_spec_tokens.emplace_back(kv.first, (int64_t)inval);
    }
    *ph = t;
  }
}

bool Lifecycle::reset_prefix_cache() { return kv_.reset_prefix_cache ? kv_.reset_prefix_cache() : false; }

// ------------------------------------------------------------------ merge_results (adapter/phys_executor.py, no FORCE)
void merge_results(const SchedOut& so, const std::vector<StepResult>& results, ModelOut* mo) {
  *mo = ModelOut();
  std::map<Rid, const std::vector<int32_t>*> tokens;
  std::map<Rid, const std::vector<std::vector<std::pair<int32_t, float>>>*> logp;
  std::map<Rid, const std::vector<int32_t>*> drafts;
  for (const auto& res : results) {
    for (const auto& kv : res.tokens) tokens[kv.first] = &kv.second;
    for (const auto& kv : res.logprobs) logp[kv.first] = &kv.second;
    for (const auto& kv : res.drafts) drafts[kv.first] = &kv.second;
  }
  for (const auto& kv : so.num_scheduled) {
    mo->req_ids.push_back(kv.first);
    auto it = tokens.find(kv.first);
    mo->sampled.push_back(it == tokens.end() ? std::vector<int32_t>() : *it->second);
  }
  if (!logp.empty()) {
    // _lp_rows: generated token j of a step <- the engine's row j (top-k (id, logprob)); sampled token first
    std::map<Rid, std::vector<LpRow>> rows;
    int64_t width = 1;
    for (size_t q = 0; q < mo->req_ids.size(); ++q) {
      auto lt = logp.find(mo->req_ids[q]);
      if (lt == logp.end()) continue;
      std::vector<LpRow>& out = rows[mo->req_ids[q]];
      const auto& toks = mo->sampled[q];
      for (size_t j = 0; j < toks.size() && j < lt->second->size(); ++j) {
        const auto& lst = (*lt->second)[j];
        LpRow r;
        int32_t t = toks[j];
        int64_t k = -1;
        for (size_t i = 0; i < lst.size(); ++i)
          if (lst[i].first == t) { k = (int64_t)i; break; }
        float own;
        if (k >= 0) { own = lst[(size_t)k].second; r.rank = k + 1; }
        else { own = lst.empty() ? 0.0f : lst.back().second; r.rank = (int64_t)lst.size() + 1; }
        r.ids.push_back(t);
        r.vals.push_back(own);
        for (const auto& p : lst) { r.ids.push_back(p.first); r.vals.push_back(p.second); }
        width = std::max<int64_t>(width, (int64_t)r.ids.size());
        out.push_back(std::move(r));
      }
    }
    mo->has_logprobs = true;
    mo->lp_width = width;
    int64_t n = 0;
    for (size_t q = 0; q < mo->req_ids.size(); ++q) {
      mo->lp_cu.push_back(n);
      auto rt = rows.find(mo->req_ids[q]);
      const auto& toks = mo->sampled[q];
      for (size_t j = 0; j < toks.size(); ++j) {
        if (rt != rows.end() && j < rt->second.size()) mo->lp_rows.push_back(rt->second[j]);
        else mo->lp_rows.push_back(LpRow{{toks[j]}, {0.0f}, 1});
        ++n;
      }
    }
  }
  if (!drafts.empty()) {
    mo->has_drafts = true;
    for (const auto& rid : mo->req_ids) {
      auto it = drafts.find(rid);
      if (it != drafts.end()) mo->drafts.emplace_back(rid, *it->second);
    }
  }
}

}  // namespace core
}  // namespace fpr
