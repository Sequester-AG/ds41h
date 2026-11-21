// AsyncEngine: what the API layer talks to. The C++ counterpart of vLLM's AsyncLLM frontend half
// EngineClient (ZMQ) + OutputProcessor (detokenizer, stop strings, finish reasons, usage counts) + Metrics.
//
//   AsyncEngine eng(opts);                 // binds/adopts the sockets
//   eng.start();                           // waits for EngineCoreReadyResponse, then get_supported_tasks
//   EngineRequest r = ...;                 // sampling_from_request(), make_internal_request_id(), prompt ids ...
//   eng.add_request(std::move(r), [](RequestOutput&& o) { ... });   // DELTA or FINAL_ONLY outputs
//   eng.abort(internal_id);                // client disconnect (AsyncLLM.abort(internal=True))
//
// Threading: add_request()/abort()/metrics() are thread-safe. on_output runs on the IO thread, sequentially
// per request (and globally); it must not block. After an output with finished == true no more outputs come.
// Usage (chat): prompt_tokens = prompt_len; completion_tokens = sum of token_ids sizes over the DELTA outputs
// (or token_ids.size() of the FINAL_ONLY output) - this includes the EOS token, exactly like vLLM;
// prompt_tokens_details.cached_tokens = num_cached_tokens (from the engine's prefill_stats).
#pragma once
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/client.h"
#include "engine/metrics.h"
#include "engine/output_processor.h"

namespace fe {
class Tokenizer;
}

namespace fe::engine {

struct EngineOptions {
  ClientOptions client;
  const fe::Tokenizer* tokenizer = nullptr;  // required for detokenization (fe_tokenizer)
  MetricsConfig metrics;
  bool log_stats = true;          // --disable-log-stats not set in production
  int64_t stream_interval = 1;    // --stream-interval (scheduler_config.stream_interval)
  double startup_utility_timeout_s = 600;
};

using OutputFn = std::function<void(RequestOutput&&)>;

// Everything the API layer knows about one generation request -> the exact EngineCoreRequest vLLM builds
// (to_sampling_params + __post_init__ + InputProcessor.process_inputs + AsyncLLM.add_request fields).
struct BuildInput {
  Endpoint endpoint = Endpoint::Chat;
  const mp::Value* body = nullptr;  // validated request JSON (to_mp(fe::Json) from json_bridge.h)
  std::vector<int32_t> prompt_ids;
  // A /v1/completions token-id prompt holds Python ints of any size. When some do not fit int32, prompt_ids holds
  // clamped placeholders and this is str() of the id validate_token_ids reports (min if negative, else max).
  std::optional<std::string> prompt_oov_id;
  std::string external_req_id;      // "chatcmpl-<16 hex>" / "cmpl-..." / "resp_..." (or X-Request-Id)
  std::optional<bool> reasoning_ended;
  std::optional<mp::Value> reasoning_parser_kwargs;  // {"chat_template_kwargs": {...}} or none
  // Parser._apply_structural_tag result (xgrammar structural tag JSON): becomes
  // StructuredOutputsParams(structural_tag=...) unless the body already carries structured outputs.
  std::optional<std::string> structural_tag;
  int64_t priority = 0;                 // header / body priority
  std::optional<std::string> session_id;
  std::optional<int64_t> data_parallel_rank;   // X-data-parallel-rank, clamped to int64
  std::string data_parallel_rank_text;         // its str() when it does not fit int64 ("" otherwise)
  std::optional<std::string> cache_salt;
  std::vector<MmFeature> mm_features;   // sorted by offset
  double arrival_time = 0;              // 0 -> wall_time()
};
// Throws ValidationError / PromptTooLongError with vLLM's exact messages (HTTP 400).
EngineRequest build_engine_request(const BuildInput& in, const SamplingContext& ctx);

// RequestSpecDecodeMetrics.to_dict() (the response "metrics.speculative_decoding" payload); nil for nil.
mp::Value spec_decode_to_dict(const mp::Value& m);

class AsyncEngine {
 public:
  explicit AsyncEngine(EngineOptions opt);
  ~AsyncEngine();

  // MPClient ready wait + get_supported_tasks(); also fills cache_config_info. Throws EngineTimeoutError /
  // EngineDeadError.
  const ReadyResponse& start();
  const std::vector<std::string>& supported_tasks() const { return tasks_; }
  int64_t max_model_len() const;  // min(--max-model-len, ready.max_model_len) (_apply_ready_response)

  // AsyncLLM.add_request: req.request_id must be the internal id (make_internal_request_id(external)).
  // arrival_time == 0 -> wall_time(). Throws EngineDeadError when the engine is dead.
  void add_request(EngineRequest req, OutputFn on_output);
  // AsyncLLM.abort(internal=True). emit_final: deliver the final ABORT output (Python builds it but the
  // cancelled generator never reads it).
  void abort(const std::string& internal_id, bool emit_final = false);

  bool dead() const { return client_->dead(); }
  Metrics& metrics() { return *metrics_; }
  EngineClient& client() { return *client_; }
  size_t num_unfinished() const;

 private:
  void on_batch(EngineCoreOutputs& outs);
  void on_dead();

  EngineOptions opt_;
  std::unique_ptr<EngineClient> client_;
  std::unique_ptr<Metrics> metrics_;
  mutable std::mutex mu_;  // op_ + fns_
  OutputProcessor op_;
  std::unordered_map<std::string, std::shared_ptr<OutputFn>> fns_;
  std::vector<std::string> tasks_;
  int64_t max_model_len_ = 0;
};

}  // namespace fe::engine
