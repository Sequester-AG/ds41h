// Server configuration: the subset of vLLM's CLI args (from --args-json) and environment
// that the API layer needs.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/json.h"

namespace fe::api {

struct Config {
  // model identity
  std::string model_path;                 // model_tag / model (also /v1/models "root")
  std::vector<std::string> served_names;  // --served-model-name (default: [model_path])
  std::string tokenizer_path;             // --tokenizer (default model_path)
  int64_t max_model_len = 0;              // filled from args or the engine's ready message
  // auth / routing
  std::vector<std::string> api_keys;      // --api-key, else VLLM_API_KEY
  std::string root_path;                  // --root-path
  // behaviour switches
  std::string response_role = "assistant";
  bool enable_prompt_tokens_details = false;
  bool enable_force_include_usage = false;
  bool enable_auto_tools = false;
  std::string tool_parser;       // --tool-call-parser
  std::string reasoning_parser;  // --reasoning-parser
  bool return_tokens_as_token_ids = false;
  bool exclude_tools_when_tool_choice_none = false;
  Json default_chat_template_kwargs;  // --default-chat-template-kwargs (object or null)
  bool enable_log_requests = false;
  bool disable_log_stats = false;
  double sse_keep_alive_interval = 0;
  // Responses store
  bool enable_store = false;     // VLLM_ENABLE_RESPONSES_API_STORE
  size_t store_max = 4096;       // PHYS_RESPONSES_STORE_MAX (BoundedStore installed via --middleware)
  bool bounded_store = false;    // --middleware adapter.responses_store.BoundedStore present
  // fingerprint ("vllm-0.32.0-<hash8>"); FE_SYSTEM_FINGERPRINT overrides (see DEVIATIONS.md)
  std::optional<std::string> system_fingerprint;
  // EdgeGuard parity: route allowlist checked before auth. Empty = every route (default).
  // FE_ROUTE_ALLOW="/v1/,/health,/metrics,/version": an entry ending in "/" allows paths under it that
  // posixpath.normpath leaves unchanged (up to a trailing "/"); any other entry matches exactly. "edgeguard" = that
  // list; "off" = none. Without FE_ROUTE_ALLOW, vLLM's --middleware ...edge_guard.EdgeGuard selects the EdgeGuard list.
  std::vector<std::string> route_allow;
  std::string route_allow_source;  // "FE_ROUTE_ALLOW" or "--middleware" (for the startup log)
  // http
  double http_timeout_keep_alive = 5.0;
  double engine_ready_timeout_s = 600.0;
  // speculative decoding (for spec-decode metric families)
  int num_speculative_tokens = 0;
  std::string vllm_version = "0.32.0";
  // raw args for anyone who needs more
  Json args;

  const std::string& model_name() const { return served_names.front(); }
  bool is_base_model(std::string_view name) const;
};

// Parses the --args-json object (vLLM's jsonify_non_default_args) plus environment.
Config config_from_args(const Json& args);

// EdgeGuard's path check (serve/edge_guard/edge_guard.py path_ok) generalised to an allowlist, and Python's
// posixpath.normpath (POSIX rules: "//" kept at the start only, "." dropped, ".." resolved, no trailing "/").
std::string py_normpath(std::string_view path);
bool route_allowed(const std::vector<std::string>& allow, std::string_view path);
// The rejection EdgeGuard sends: 404, {"detail": "Not Found"} (json.dumps spacing, unlike FastAPI's own 404 body).
inline constexpr std::string_view kEdgeGuardBody = "{\"detail\": \"Not Found\"}";

}  // namespace fe::api
