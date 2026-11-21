// fe_api: the vLLM 0.32.0 OpenAI-compatible routes on top of fe_http, wired to the
// tokenizer (B), renderer (C), parser (D), image (F) and engine (E) components.
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "api/backend.h"
#include "api/config.h"
#include "api/errors.h"
#include "api/http_metrics.h"
#include "api/store.h"
#include "common/json.h"
#include "http/http_server.h"
#include "image/image.h"
#include "parser/parser.h"
#include "render/render.h"
#include "tokenizer/tokenizer.h"

namespace fe::api {

struct Ctx {
  std::shared_ptr<http::Request> req;
  std::shared_ptr<http::Response> resp;
  std::string route_path;
};

struct Components {
  std::shared_ptr<fe::Tokenizer> tokenizer;
  std::shared_ptr<const fe::ParserVocab> vocab;
  fe::RenderOptions render;
  fe::VisionConfig vision;
};

class App {
 public:
  App(Config cfg, Components comp, std::shared_ptr<GenBackend> backend);
  ~App();
  // fe_http handler (runs on a worker thread).
  void handle(std::shared_ptr<http::Request> req, std::shared_ptr<http::Response> resp);
  const Config& config() const { return cfg_; }
  bool engine_dead() const { return backend_ && backend_->dead(); }

  // --- shared helpers used by the route files ---
  void send_json(Ctx& c, int status, std::string body, http::HeaderList extra = {});
  void send_error(Ctx& c, const ApiError& e);
  Json parse_json_body(Ctx& c);   // FastAPI body read + validate_json_request. Throws ApiError.
  void check_model(const Json& body);  // _check_model (404)
  std::string base_request_id(Ctx& c, const Json* body_request_id);
  void preflight();
  void stream_event_log(Ctx& c, std::shared_ptr<EventLog> log, std::optional<int64_t> starting_after);
  // ParserOptions::on_tool_parser_invocation for one request's parsers (empty without --tool-call-parser).
  // Non-streaming parser.parse(enable_auto_tools=...) reaches extract_tool_calls only with enable_auto_tools or
  // tool_choice "none" (DelegatingParser._extract_tool_calls); streaming always does in the tool-call phase.
  std::function<void(bool, bool)> tool_parser_hook(ToolParserRequestType rt, fe::ToolChoice tc) const;

  // routes
  void route_chat(Ctx& c);
  void route_completions(Ctx& c);
  void route_responses_create(Ctx& c);
  void route_responses_get(Ctx& c, const std::string& id);
  void route_responses_cancel(Ctx& c, const std::string& id);
  void route_tokenize(Ctx& c);
  void route_detokenize(Ctx& c);
  void route_models(Ctx& c);
  void route_metrics(Ctx& c);

  Config cfg_;
  Components comp_;
  std::shared_ptr<GenBackend> backend_;
  std::unique_ptr<HttpMetrics> http_metrics_;
  std::shared_ptr<ToolParserMetrics> tool_parser_metrics_;  // set iff --tool-call-parser (init_app_state)
  std::unique_ptr<ResponsesStore> store_;
  std::vector<std::string> api_key_hashes_;
  std::mutex bg_mu_;
  std::unordered_map<std::string, std::string> bg_internal_;  // response id -> internal engine id
};

// vllm.utils.random_uuid(): 16 lowercase hex digits (low 64 bits of a uuid4).
std::string random_uuid_hex();
// time.time()
double now_unix();
// RenderError (fe_render) -> ApiError
ApiError from_render_error(const fe::RenderError& e);

}  // namespace fe::api
