#include "engine/output_processor.h"

#include <chrono>

namespace fe::engine {

static double default_wall() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
static double (*g_wall)() = default_wall;
double wall_time() { return g_wall(); }
void set_wall_clock_for_tests(double (*fn)()) { g_wall = fn ? fn : default_wall; }

void RequestOutput::merge(const RequestOutput& next) {
  finished = finished || next.finished;
  text += next.text;
  token_ids.insert(token_ids.end(), next.token_ids.begin(), next.token_ids.end());
  if (!next.spec_decode_metrics.is_nil()) spec_decode_metrics = next.spec_decode_metrics;
  finish_reason = next.finish_reason;
  stop_reason = next.stop_reason;
  num_output_tokens = next.num_output_tokens;
  metrics = next.metrics;
  if (next.engine_dead) {
    engine_dead = true;
    error = next.error;
  }
}

// ------------------------------------------------------------------ stats.py
void iteration_update_from_output(IterationStats& it, const EngineOutput& o, double ts, bool is_prefilling,
                                  RequestStateStats& rs) {
  int64_t n_new = int64_t(o.new_token_ids.size());
  it.num_generation_tokens += n_new;
  if (is_prefilling) {
    if (o.prefill_stats) {
      it.pt_computed += o.prefill_stats->num_computed_tokens;
      it.pt_cached_tokens += o.prefill_stats->num_cached_tokens;
      it.pt_total += o.prefill_stats->num_prompt_tokens;
      it.pt_local_cache_hit += o.prefill_stats->num_local_cached_tokens;
      it.pt_external_kv_transfer += o.prefill_stats->num_external_cached_tokens;
    }
    double ftl = it.iteration_timestamp - rs.arrival_time;
    it.time_to_first_tokens_iter.push_back(ftl);
    rs.first_token_latency = ftl;
  }
  rs.num_generation_tokens += n_new;
  // VLLM_COMPUTE_NANS_IN_LOGITS is off in production: is_corrupted never set.
  for (auto& ev : o.events) {
    if (ev.type == 1) {  // QUEUED
      rs.queued_ts = ev.timestamp;
    } else if (ev.type == 2) {  // SCHEDULED
      if (rs.scheduled_ts == 0.0) rs.scheduled_ts = ev.timestamp;
    } else if (ev.type == 3) {
      it.num_preempted_reqs += 1;
      rs.num_preemptions += 1;
    }
  }
  if (is_prefilling) {
    rs.first_token_ts = ts;
  } else {
    it.inter_token_latencies_iter.push_back(ts - rs.last_token_ts);
  }
  rs.last_token_ts = ts;
}

void iteration_update_from_finished(IterationStats& it, FinishReason r, const std::string& external_id,
                                    int64_t num_prompt_tokens, std::optional<int64_t> max_tokens_param,
                                    const RequestStateStats& rs, int64_t num_cached_tokens) {
  FinishedRequestStats f;
  f.finish_reason = r;
  f.request_id = external_id;
  f.e2e_latency = it.iteration_timestamp - rs.arrival_time;
  f.queued_time = rs.scheduled_ts - rs.queued_ts;
  f.prefill_time = rs.first_token_ts - rs.scheduled_ts;
  f.decode_time = rs.last_token_ts - rs.first_token_ts;
  f.inference_time = rs.last_token_ts - rs.scheduled_ts;
  f.mean_time_per_output_token = rs.num_generation_tokens - 1 > 0 ? f.decode_time / double(rs.num_generation_tokens - 1) : 0;
  f.num_prompt_tokens = num_prompt_tokens;
  f.num_preemptions = rs.num_preemptions;
  f.num_generation_tokens = rs.num_generation_tokens;
  f.max_tokens_param = max_tokens_param;
  f.is_corrupted = rs.is_corrupted;
  f.num_cached_tokens = num_cached_tokens;
  it.finished_requests.push_back(std::move(f));
  if (rs.is_corrupted) it.num_corrupted_reqs += 1;
  it.num_not_watermarked_reqs += 1;  // no watermark config
  // ParentRequest.observe_finished_request(None, ...): n == 1
  it.max_num_generation_tokens_iter.push_back(rs.num_generation_tokens);
  it.n_params_iter.push_back(1);
}

// ------------------------------------------------------------------ RequestState
RequestState::RequestState(const EngineRequest& req, std::unique_ptr<TokenDecoder> decoder,
                           int64_t engine_stream_interval, bool log_stats)
    : request_id(req.request_id),
      external_req_id(req.external_req_id),
      output_kind(req.sampling.output_kind),
      prompt_len(int64_t(req.prompt_ids.size())),
      max_tokens_param(req.sampling.max_tokens),
      n(req.sampling.n),
      detok(req.sampling.detokenize ? std::move(decoder) : nullptr, req.sampling) {
  stream_interval = engine_stream_interval;
  if (req.sampling.stream_interval) stream_interval = std::max(*req.sampling.stream_interval, engine_stream_interval);
  if (log_stats) {
    stats.emplace();
    stats->arrival_time = req.arrival_time;
  }
}

std::optional<RequestOutput> RequestState::make_request_output(std::vector<int32_t> new_token_ids,
                                                               std::optional<FinishReason> finish_reason,
                                                               const mp::Value& stop_reason,
                                                               const mp::Value* spec_metrics) {
  bool finished = finish_reason.has_value();
  bool final_only = output_kind == OutputKind::FinalOnly;
  if (!finished && final_only) return std::nullopt;
  bool delta = output_kind == OutputKind::Delta;
  if (stream_interval > 1) {
    if (!(finished || sent_tokens_offset == 0 ||
          int64_t(detok.num_output_tokens()) - sent_tokens_offset >= stream_interval))
      return std::nullopt;
    if (delta) {
      const auto& all = detok.output_token_ids();
      new_token_ids.assign(all.begin() + std::min<int64_t>(sent_tokens_offset, int64_t(all.size())), all.end());
      sent_tokens_offset = int64_t(detok.num_output_tokens());
    }
  }
  RequestOutput ro;
  ro.request_id = external_req_id;
  ro.internal_id = request_id;
  ro.text = detok.get_next_output_text(finished, delta);
  ro.token_ids = delta ? std::move(new_token_ids) : detok.output_token_ids();
  ro.finished = finished;
  if (finished) {
    ro.finish_reason = finish_reason_str(*finish_reason);
    ro.stop_reason = stop_reason;
    if (spec_metrics) ro.spec_decode_metrics = *spec_metrics;
  }
  ro.num_cached_tokens = num_cached_tokens;
  ro.num_cache_creation_tokens = num_cache_creation_tokens;
  ro.prompt_len = prompt_len;
  ro.num_output_tokens = int64_t(detok.num_output_tokens());
  if (stats) ro.metrics = *stats;
  return ro;
}

// ------------------------------------------------------------------ OutputProcessor
void OutputProcessor::add_request(const EngineRequest& req, std::unique_ptr<TokenDecoder> decoder) {
  if (states_.count(req.request_id)) return;  // streaming-input updates are not supported (resumable=False)
  states_[req.request_id] = std::make_unique<RequestState>(req, std::move(decoder), stream_interval_, log_stats_);
}

bool OutputProcessor::process_one(EngineOutput& o, double ts, IterationStats* iteration,
                                  std::vector<std::string>& reqs_to_abort, RequestOutput& out) {
  auto it = states_.find(o.request_id);
  if (it == states_.end()) return false;  // already aborted
  RequestState& st = *it->second;
  // 1) stats
  if (iteration && st.stats) iteration_update_from_output(*iteration, o, ts, st.is_prefilling, *st.stats);
  std::optional<FinishReason> finish = o.finish_reason;
  mp::Value stop_reason = o.stop_reason;
  if (st.is_prefilling) {
    if (o.prefill_stats) {
      st.num_cached_tokens = o.prefill_stats->num_cached_tokens;
      st.num_cache_creation_tokens = o.prefill_stats->num_cache_creation_tokens;
    }
    st.is_prefilling = false;
  }
  if (!o.spec_decode_metrics.is_nil()) st.spec_decode_metrics = o.spec_decode_metrics;
  // 2) detokenize + stop strings
  size_t num_prev = st.detok.num_output_tokens();
  auto stop_string = st.detok.update(o.new_token_ids, finish == FinishReason::Stop);
  if (stop_string) {
    finish = FinishReason::Stop;
    stop_reason = mp::Value::from_str(*stop_string);
    size_t num_kept = st.detok.num_output_tokens() - num_prev;  // _trim_surplus_tokens
    if (o.new_token_ids.size() > num_kept) o.new_token_ids.resize(num_kept);
  }
  // 4) RequestOutput
  auto ro = st.make_request_output(o.new_token_ids, finish, stop_reason,
                                   st.spec_decode_metrics.is_nil() ? nullptr : &st.spec_decode_metrics);
  bool emitted = false;
  if (ro) {
    ro->mm_cache_miss_hashes = o.mm_cache_miss_hashes;
    out = std::move(*ro);
    emitted = true;
  }
  if (finish) {
    if (!o.finished()) reqs_to_abort.push_back(o.request_id);
    if (iteration && st.stats)
      iteration_update_from_finished(*iteration, *finish, st.external_req_id, st.prompt_len, st.max_tokens_param,
                                     *st.stats, st.num_cached_tokens);
    states_.erase(it);
  }
  return emitted;
}

std::vector<RequestOutput> OutputProcessor::abort_requests(const std::vector<std::string>& ids) {
  std::vector<RequestOutput> outs;
  for (auto& id : ids) {
    auto it = states_.find(id);
    if (it == states_.end()) continue;
    auto ro = it->second->make_request_output({}, FinishReason::Abort, mp::Value::nil(), nullptr);
    if (ro) outs.push_back(std::move(*ro));
    states_.erase(it);
  }
  return outs;
}

std::vector<RequestOutput> OutputProcessor::propagate_error(const std::string& error) {
  std::vector<RequestOutput> outs;
  for (auto& [id, st] : states_) {
    RequestOutput ro;
    ro.request_id = st->external_req_id;
    ro.internal_id = id;
    ro.engine_dead = true;
    ro.error = error;
    ro.finished = true;
    outs.push_back(std::move(ro));
  }
  states_.clear();
  return outs;
}

}  // namespace fe::engine
