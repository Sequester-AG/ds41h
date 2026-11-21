// OutputProcessor: vLLM v1/engine/output_processor.py RequestState + process_outputs, and the
// per-request / per-iteration stats of v1/metrics/stats.py (RequestStateStats, IterationStats,
// FinishedRequestStats) that feed the Prometheus histograms (metrics.h).
//
// Python runs this on the asyncio loop thread; here it runs on the EngineClient IO thread (see engine.h).
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/detok.h"
#include "engine/protocol.h"

namespace fe::engine {

struct RequestStateStats {
  int64_t num_generation_tokens = 0;
  int64_t num_preemptions = 0;
  double arrival_time = 0;  // wall clock (frontend)
  double queued_ts = 0, scheduled_ts = 0, first_token_ts = 0, last_token_ts = 0;  // engine monotonic
  double first_token_latency = 0;
  bool is_corrupted = false;
};

struct FinishedRequestStats {
  FinishReason finish_reason = FinishReason::Stop;
  std::string request_id;  // external
  double e2e_latency = 0;
  int64_t num_prompt_tokens = 0, num_preemptions = 0, num_generation_tokens = 0;
  std::optional<int64_t> max_tokens_param;
  double queued_time = 0, prefill_time = 0, inference_time = 0, decode_time = 0, mean_time_per_output_token = 0;
  bool is_corrupted = false;
  int64_t num_cached_tokens = 0;
};

// IterationStats for one EngineCoreOutputs (created only when it has outputs, like AsyncLLM.output_handler).
struct IterationStats {
  double iteration_timestamp = 0;  // time.time() when the batch is processed
  int64_t num_generation_tokens = 0;
  // PromptTokenStats
  int64_t pt_computed = 0, pt_local_cache_hit = 0, pt_external_kv_transfer = 0, pt_cached_tokens = 0, pt_total = 0;
  int64_t num_preempted_reqs = 0;
  std::vector<FinishedRequestStats> finished_requests;
  std::vector<int64_t> max_num_generation_tokens_iter, n_params_iter;
  std::vector<double> time_to_first_tokens_iter, inter_token_latencies_iter;
  int64_t num_corrupted_reqs = 0;
  int64_t num_watermarked_reqs = 0, num_watermark_skipped_reqs = 0, num_not_watermarked_reqs = 0;
};

// What the API layer consumes: vLLM's RequestOutput with one CompletionOutput (n == 1).
struct RequestOutput {
  std::string request_id;  // EXTERNAL id (RequestOutput.request_id)
  std::string internal_id;
  std::string text;               // delta text (DELTA) or the full text (FINAL_ONLY)
  std::vector<int32_t> token_ids; // delta ids (DELTA) or all output ids (FINAL_ONLY)
  bool finished = false;
  std::optional<std::string> finish_reason;  // "stop" "length" "abort" "error" "repetition" when finished
  mp::Value stop_reason;                     // nil | int (stop token id) | str (stop string); nil unless finished
  int64_t num_cached_tokens = 0, num_cache_creation_tokens = 0;
  int64_t prompt_len = 0;
  int64_t num_output_tokens = 0;  // detokenizer.num_output_tokens() after this output (incl. EOS)
  mp::Value spec_decode_metrics;  // RequestSpecDecodeMetrics map on the final output, else nil
  std::vector<std::string> mm_cache_miss_hashes;
  RequestStateStats metrics;      // RequestOutput.metrics snapshot
  bool engine_dead = false;       // the engine died: no more outputs; API returns its EngineDeadError path
  std::string error;              // with engine_dead
  // RequestOutput.add(next, aggregate=True) for a lagging consumer (DELTA): concatenates text/ids, takes the
  // latest finish/stop/spec metrics.
  void merge(const RequestOutput& next);
};

class RequestState {
 public:
  RequestState(const EngineRequest& req, std::unique_ptr<TokenDecoder> decoder, int64_t engine_stream_interval,
               bool log_stats);
  // Builds RequestState.make_request_output(); nullopt when nothing is due (FINAL_ONLY / stream_interval).
  std::optional<RequestOutput> make_request_output(std::vector<int32_t> new_token_ids,
                                                   std::optional<FinishReason> finish_reason,
                                                   const mp::Value& stop_reason, const mp::Value* spec_metrics);
  std::string request_id, external_req_id;
  OutputKind output_kind = OutputKind::FinalOnly;
  int64_t prompt_len = 0;
  std::optional<int64_t> max_tokens_param;
  int64_t n = 1;
  bool is_prefilling = true;
  int64_t num_cached_tokens = 0, num_cache_creation_tokens = 0;
  int64_t stream_interval = 1, sent_tokens_offset = 0;
  mp::Value spec_decode_metrics;
  std::optional<RequestStateStats> stats;
  IncrementalDetokenizer detok;
};

class OutputProcessor {
 public:
  explicit OutputProcessor(bool log_stats = true, int64_t stream_interval = 1)
      : log_stats_(log_stats), stream_interval_(stream_interval) {}

  void add_request(const EngineRequest& req, std::unique_ptr<TokenDecoder> decoder);
  bool has_request(const std::string& internal_id) const { return states_.count(internal_id) != 0; }
  size_t num_unfinished() const { return states_.size(); }

  // process_outputs(): for each EngineOutput of the batch, in order. `emit(RequestOutput&&)` receives the
  // outputs to deliver; `reqs_to_abort` gets the internal ids finished by a stop string (send ABORT for them).
  // `iteration` may be null (log_stats off or no outputs).
  template <class Emit>
  void process_outputs(std::vector<EngineOutput>& outputs, double engine_core_timestamp, IterationStats* iteration,
                       std::vector<std::string>& reqs_to_abort, Emit&& emit) {
    for (auto& o : outputs) {
      RequestOutput ro;
      if (process_one(o, engine_core_timestamp, iteration, reqs_to_abort, ro)) emit(std::move(ro));
    }
  }
  // abort_requests(internal=True): returns the final ABORT outputs (Python puts them in the queues).
  std::vector<RequestOutput> abort_requests(const std::vector<std::string>& internal_ids);
  // propagate_error(EngineDeadError): one engine_dead output per live request; clears all state.
  std::vector<RequestOutput> propagate_error(const std::string& error);

  bool process_one(EngineOutput& o, double engine_core_timestamp, IterationStats* iteration,
                   std::vector<std::string>& reqs_to_abort, RequestOutput& out);

 private:
  bool log_stats_;
  int64_t stream_interval_;
  std::unordered_map<std::string, std::unique_ptr<RequestState>> states_;
};

// stats.py helpers
void iteration_update_from_output(IterationStats& it, const EngineOutput& o, double engine_core_timestamp,
                                  bool is_prefilling, RequestStateStats& rs);
void iteration_update_from_finished(IterationStats& it, FinishReason r, const std::string& external_id,
                                    int64_t num_prompt_tokens, std::optional<int64_t> max_tokens_param,
                                    const RequestStateStats& rs, int64_t num_cached_tokens);

// Wall clock used for arrival_time / iteration_timestamp (time.time()); replaceable for golden tests.
double wall_time();
void set_wall_clock_for_tests(double (*fn)());

}  // namespace fe::engine
