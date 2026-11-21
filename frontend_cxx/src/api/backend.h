// The API layer's view of generation: one request in, RequestOutput-like deltas out.
// EngineBackend (backend_engine.h) implements it over fe_engine's AsyncEngine; tests can
// substitute a scripted fake.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/json.h"

namespace fe {
struct ImageFeatures;
}

namespace fe::api {

struct GenRequest {
  std::string external_id;    // RequestOutput.request_id: "chatcmpl-..", "cmpl-..-0", "resp_.._0"
  std::string endpoint;       // "chat" | "completion" | "responses"
  Json body;                  // validated + coerced request body (structured_outputs set for tool tags)
  std::vector<int32_t> prompt_ids;
  std::optional<std::string> prompt_oov_id;  // completions token-id prompt with ids outside int32 (BuildInput)
  std::vector<std::shared_ptr<const fe::ImageFeatures>> images;
  std::vector<int64_t> image_offsets;
  std::optional<bool> reasoning_ended;
  Json reasoning_parser_kwargs;  // null or {"chat_template_kwargs": {...}}
  std::optional<std::string> structural_tag;  // DelegatingParser._apply_structural_tag result
  int64_t priority = 0;
  std::optional<std::string> session_id;
  std::optional<int64_t> data_parallel_rank;   // X-data-parallel-rank (chat / completions), clamped to int64
  std::string data_parallel_rank_text;         // its str() when it does not fit int64
  double arrival_time = 0;    // time.time()
};

// == vllm.outputs.RequestOutput for n=1 (DELTA when streaming, FINAL_ONLY otherwise).
struct GenOutput {
  std::string text;
  std::vector<int32_t> token_ids;
  std::optional<std::string> finish_reason;  // "stop" | "length" | "abort" | "error" | "repetition"
  Json stop_reason;                          // null | int | str
  bool finished = false;
  int64_t num_cached_tokens = 0, num_cache_creation_tokens = 0;
  Json spec_decode;                          // RequestSpecDecodeMetrics map or null
  bool engine_dead = false;
};

// SamplingParams values the Responses API echoes (temperature, top_p, max_tokens, ...).
struct SamplingSummary {
  double temperature = 1.0, top_p = 1.0, presence_penalty = 0.0, frequency_penalty = 0.0;
  int64_t max_tokens = 0;
  std::optional<int64_t> logprobs;
};

using GenCallback = std::function<void(GenOutput&&)>;

class GenBackend {
 public:
  virtual ~GenBackend() = default;
  // Builds SamplingParams + EngineCoreRequest and adds it. Returns the internal id.
  // Throws ApiError for input-processing failures (400) and engine death (500).
  // Callbacks for one request are serialized; they run on the backend's thread.
  virtual std::string start(GenRequest&& req, GenCallback cb) = 0;
  virtual void abort(const std::string& internal_id) = 0;
  virtual bool dead() const = 0;
  virtual SamplingSummary sampling_summary(const std::string& endpoint, const Json& body, int64_t prompt_len) = 0;
  // streams: the phys engine's sampling-caps 400 before the SSE headers (other errors keep their place)
  virtual void precheck_caps(const std::string& endpoint, const Json& body, int64_t prompt_len) {}
  virtual int64_t max_model_len() const = 0;
  virtual std::string render_vllm_metrics() = 0;
  virtual size_t num_unfinished() const = 0;
};

}  // namespace fe::api
