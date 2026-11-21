#include "engine/engine.h"

#ifdef FE_HAVE_TOKENIZER
#include "tokenizer/tokenizer.h"
#endif

namespace fe::engine {

EngineRequest build_engine_request(const BuildInput& in, const SamplingContext& ctx) {
  static const mp::Value empty = mp::Value::make_map();
  const mp::Value& body = in.body ? *in.body : empty;
  EngineRequest r;
  InputChecks checks{in.prompt_ids, in.prompt_oov_id, in.data_parallel_rank, in.data_parallel_rank_text};
  r.sampling = sampling_from_request(in.endpoint, body, int64_t(in.prompt_ids.size()), ctx, &checks);
  if (in.structural_tag && !r.sampling.structured_outputs) {
    StructuredOutputs so;
    so.structural_tag = *in.structural_tag;
    r.sampling.structured_outputs = so;
  }
  r.external_req_id = in.external_req_id;
  r.request_id = make_internal_request_id(in.external_req_id);
  r.prompt_ids = in.prompt_ids;
  r.mm_features = in.mm_features;
  r.arrival_time = in.arrival_time != 0 ? in.arrival_time : wall_time();
  r.cache_salt = in.cache_salt;
  if (!r.cache_salt)
    if (auto* cs = body.get("cache_salt"); cs && cs->is_str()) r.cache_salt = cs->s;
  r.data_parallel_rank = in.data_parallel_rank;
  r.priority = in.priority;
  r.reasoning_ended = in.reasoning_ended;
  r.reasoning_parser_kwargs = in.reasoning_parser_kwargs;
  r.session_id = in.session_id;
  r.client_index = 0;
  return r;
}

mp::Value spec_decode_to_dict(const mp::Value& m) {
  if (!m.is_map()) return mp::Value::nil();
  auto geti = [&](const char* k) -> int64_t {
    auto* v = m.get(k);
    return v && v->is_num() ? v->as_int() : 0;
  };
  std::vector<int64_t> hist;
  if (auto* h = m.get("histogram"); h && h->is_array())
    for (auto& e : h->arr) hist.push_back(e.as_int());
  int64_t steps = 0, accepted = 0;
  for (size_t j = 0; j < hist.size(); ++j) {
    steps += hist[j];
    accepted += int64_t(j) * hist[j];
  }
  int64_t drafted = geti("num_draft_tokens");
  mp::Value o = mp::Value::make_map();
  o.set("mean_acceptance_length", mp::Value::from_double(steps ? 1.0 + double(accepted) / double(steps) : 1.0));
  o.set("draft_acceptance_rate", mp::Value::from_double(drafted ? double(accepted) / double(drafted) : 0.0));
  mp::Value ha = mp::Value::make_array();
  for (auto x : hist) ha.arr.push_back(mp::Value::from_int(x));
  o.set("acceptance_histogram", ha);
  o.set("num_spec_steps", mp::Value::from_int(steps));
  o.set("num_accepted_draft_tokens", mp::Value::from_int(accepted));
  o.set("num_draft_tokens", mp::Value::from_int(drafted));
  o.set("num_spec_tokens", mp::Value::from_int(geti("num_spec_tokens")));
  if (auto* pa = m.get("per_step_accepted"); pa && pa->is_array() && !pa->arr.empty()) {
    o.set("per_step_accepted", *pa);
    if (auto* pd = m.get("per_step_drafted")) o.set("per_step_drafted", *pd);
  }
  return o;
}

AsyncEngine::AsyncEngine(EngineOptions opt)
    : opt_(std::move(opt)), op_(opt_.log_stats, opt_.stream_interval) {
  client_ = std::make_unique<EngineClient>(opt_.client);
  metrics_ = std::make_unique<Metrics>(opt_.metrics);
  max_model_len_ = opt_.metrics.max_model_len;
  client_->set_batch_handler([this](EngineCoreOutputs& o) { on_batch(o); });
  client_->set_dead_callback([this] { on_dead(); });
}

AsyncEngine::~AsyncEngine() {
  client_->set_batch_handler(nullptr);
  client_->set_dead_callback(nullptr);
  client_.reset();  // joins the IO thread
}

const ReadyResponse& AsyncEngine::start() {
  const ReadyResponse& r = client_->wait_ready();
  if (r.max_model_len > 0) max_model_len_ = std::min(max_model_len_, r.max_model_len);
  tasks_ = client_->get_supported_tasks(opt_.startup_utility_timeout_s);
  metrics_->set_ready(r);
  return r;
}

int64_t AsyncEngine::max_model_len() const { return max_model_len_; }

size_t AsyncEngine::num_unfinished() const {
  std::lock_guard<std::mutex> lk(mu_);
  return op_.num_unfinished();
}

void AsyncEngine::add_request(EngineRequest req, OutputFn on_output) {
  if (client_->dead()) throw EngineDeadError();
  if (req.arrival_time == 0) req.arrival_time = wall_time();
  std::unique_ptr<TokenDecoder> dec;
#ifdef FE_HAVE_TOKENIZER
  if (opt_.tokenizer && req.sampling.detokenize) dec = make_fast_decoder(*opt_.tokenizer, req.prompt_ids, req.sampling);
#endif
  {
    std::lock_guard<std::mutex> lk(mu_);
    op_.add_request(req, std::move(dec));
    fns_[req.request_id] = std::make_shared<OutputFn>(std::move(on_output));
  }
  try {
    client_->add(req, OutputCallback());
  } catch (...) {
    std::lock_guard<std::mutex> lk(mu_);
    op_.abort_requests({req.request_id});
    fns_.erase(req.request_id);
    throw;
  }
}

void AsyncEngine::abort(const std::string& internal_id, bool emit_final) {
  std::vector<RequestOutput> finals;
  std::shared_ptr<OutputFn> fn;
  {
    std::lock_guard<std::mutex> lk(mu_);
    finals = op_.abort_requests({internal_id});
    auto it = fns_.find(internal_id);
    if (it != fns_.end()) {
      fn = it->second;
      fns_.erase(it);
    }
  }
  client_->abort({internal_id});
  if (emit_final && fn && *fn)
    for (auto& f : finals) (*fn)(std::move(f));
}

void AsyncEngine::on_batch(EngineCoreOutputs& outs) {
  // AsyncLLM.output_handler: process_outputs -> abort stop-string requests -> logger.record
  std::vector<std::pair<std::shared_ptr<OutputFn>, RequestOutput>> deliver;
  std::vector<std::string> to_abort;
  IterationStats it;
  IterationStats* itp = nullptr;
  if (opt_.log_stats && !outs.outputs.empty()) {
    it.iteration_timestamp = wall_time();
    itp = &it;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    op_.process_outputs(outs.outputs, outs.timestamp, itp, to_abort, [&](RequestOutput&& ro) {
      auto f = fns_.find(ro.internal_id);
      if (f == fns_.end()) return;
      std::shared_ptr<OutputFn> fn = f->second;
      if (ro.finished) fns_.erase(f);
      deliver.emplace_back(std::move(fn), std::move(ro));
    });
    // requests finished without an output (FINAL_ONLY never misses one, but stay safe)
    for (auto& o : outs.outputs)
      if (o.finished() && !op_.has_request(o.request_id)) fns_.erase(o.request_id);
  }
  for (auto& [fn, ro] : deliver)
    if (fn && *fn) (*fn)(std::move(ro));
  if (!to_abort.empty()) client_->abort(to_abort);
  if (opt_.log_stats) metrics_->record(outs.scheduler_stats.is_nil() ? nullptr : &outs.scheduler_stats, itp);
}

void AsyncEngine::on_dead() {
  std::vector<std::pair<std::shared_ptr<OutputFn>, RequestOutput>> deliver;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& ro : op_.propagate_error("EngineCore encountered an issue. See stack trace (above) for the root cause.")) {
      auto f = fns_.find(ro.internal_id);
      if (f == fns_.end()) continue;
      deliver.emplace_back(f->second, std::move(ro));
    }
    fns_.clear();
  }
  for (auto& [fn, ro] : deliver)
    if (fn && *fn) (*fn)(std::move(ro));
}

}  // namespace fe::engine
