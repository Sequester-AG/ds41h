// Small helpers shared by the route implementations.
#pragma once
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "api/backend.h"
#include "api/errors.h"
#include "api/jw.h"
#include "common/json.h"
#include "http/http_server.h"

namespace fe::api {

// Python truthiness of a decoded JSON value (data.get(k) in a mode="before" validator).
inline bool py_truthy(const Json* v) {
  if (!v) return false;
  switch (v->type()) {
    case Json::Type::Null: return false;
    case Json::Type::Bool: return v->as_bool();
    case Json::Type::Int: return v->as_int() != 0;
    case Json::Type::BigInt: return true;
    case Json::Type::Float: return v->as_double() != 0.0;
    case Json::Type::String: return !v->as_string().empty();
    default: return v->size() != 0;
  }
}

// Python str() of a raw JSON value as used in f-strings ({data['tool_choice']}).
std::string py_fstr(const Json& v);

// A /v1/completions token-id prompt (pydantic list[int]: Python ints of any size) as the engine's int32 ids.
// Ids outside int32 are out of vocabulary whatever the limit: they become INT32_MIN / INT32_MAX placeholders and
// `oov_id` gets str() of the id validate_token_ids reports for the whole list (the min if negative, else the max),
// which fe_engine raises at the input processor's point (BuildInput::prompt_oov_id). Non-integers become 0.
// [begin, end) of the list only (TokenizeParams truncation keeps a slice).
std::vector<int32_t> token_prompt_ids(const Json& ids, std::optional<std::string>& oov_id, size_t begin = 0,
                                      size_t end = SIZE_MAX);

// Python int(s) of a header value (base 10: surrounding whitespace, one sign, single '_' between digits);
// nullopt = ValueError. `text` gets str() of the value; the result is clamped to int64.
std::optional<int64_t> py_int_header(std::string_view s, std::string* text);

// TokenizeParams._truncation_slice for a prompt of n tokens: the [begin, end) kept. `truncate`: the request's
// truncate_prompt_tokens (-1 = max_model_len - 1, or max_model_len when max_tokens is 0); `left`: truncation_side
// "left" (the request's, else the tokenizer's: "left" as vLLM loads V4.1).
std::pair<size_t, size_t> truncation_slice(size_t n, std::optional<int64_t> truncate, bool left,
                                           int64_t max_model_len, int64_t max_output_tokens);

// SSE response headers (StreamingResponse(media_type="text/event-stream")).
inline http::HeaderList sse_headers() { return {{"content-type", "text/event-stream; charset=utf-8"}}; }

// Merge `b` into `a` (RequestOutput.add with aggregate=True).
inline void merge_output(GenOutput& a, GenOutput&& b) {
  a.text += b.text;
  a.token_ids.insert(a.token_ids.end(), b.token_ids.begin(), b.token_ids.end());
  if (!b.spec_decode.is_null()) a.spec_decode = std::move(b.spec_decode);
  a.finish_reason = std::move(b.finish_reason);
  a.stop_reason = std::move(b.stop_reason);
  a.finished |= b.finished;
  a.engine_dead |= b.engine_dead;
  // RequestOutput.add keeps the first output's num_cached_tokens (set at the prefill output)
}

// A per-request output queue with backpressure-aware pumping. Outputs that arrive while
// the connection is congested are merged (vLLM's RequestOutputCollector behaviour when the
// consumer lags). `process` returns false when the stream is finished.
// Lifetime: the backend's output callback owns the Pump, and the engine drops that callback once the
// request finishes. Output parked behind congestion therefore keeps the Pump alive itself (self_) until
// it is delivered (kick() on writable) or the client is gone (the HTTP layer wakes on_writable on
// disconnect).
class Pump : public std::enable_shared_from_this<Pump> {
 public:
  using Fn = std::function<bool(GenOutput&&)>;
  Pump(std::shared_ptr<http::Response> r, Fn fn) : resp_(std::move(r)), fn_(std::move(fn)) {}
  void push(GenOutput&& o);
  void kick();  // on writable
  bool finished() const { return done_; }

 private:
  void run();
  std::shared_ptr<http::Response> resp_;
  Fn fn_;
  std::mutex mu_;
  std::deque<GenOutput> q_;
  bool running_ = false;
  bool done_ = false;
  std::shared_ptr<Pump> self_;  // set while output is parked behind congestion (under mu_)
};

// RequestSpecDecodeMetrics.to_dict() -> SpeculativeDecodingMetrics fields, written into w.
// pydantic=true: model_dump_json(exclude_unset) (no per_step_* unless set).
void write_spec_decode(JW& w, const Json& m, bool full_dump);
bool spec_decode_present(const Json& m);
// usage.prompt_tokens_details: stream (exclude_none, omitted when None) / full (model_dump).
void write_prompt_details_stream(JW& w, bool enabled, int64_t cached, int64_t creation, int64_t mm_image_tokens);
void write_prompt_details_full(JW& w, bool enabled, int64_t cached, int64_t creation, int64_t mm_image_tokens);
// "metrics": PerRequestMetrics model_dump() or null (full responses).
void write_metrics_full(JW& w, const Json& spec_decode);

}  // namespace fe::api
