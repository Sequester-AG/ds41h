// RENDER component: request validation + chat template for /v1/chat/completions, /v1/responses and the
// /tokenize messages form, byte-identical to vLLM 0.32.0 + the production dsml_roundtrip v2.2 plugin.
//
// What runs here, in the Python order:
//   ChatCompletionRequest model validators (before, reverse definition order) -> pydantic field validation (incl. the
//   ChatCompletionMessageParam smart union) -> after validators -> dsml_roundtrip _api_normalize (effort map,
//   top-level thinking, DSML_STRICT) -> build_chat_params -> OnlineRenderer.render_chat / preprocess_chat checks ->
//   parse_chat_messages side effects (content-part errors, image items, tool-call `arguments` decoded in place) ->
//   DeepseekV41 apply_chat_template (_normalize_messages, tools on the first system message, thinking / effort) ->
//   plugin encode_messages (tool `strict` not rendered) -> deepseek_v41_encoding.
// Tokenization is NOT done here (the prompt string goes to fe::Tokenizer::encode, add_special_tokens=False unless the
// request says otherwise); neither is media decoding (images are returned in placeholder order for fe_image).
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"

namespace fe {

class Tokenizer;

// An HTTP error exactly as the Python frontend returns it: body {"error": {"message", "type", "param", "code"}}.
struct RenderError : std::runtime_error {
  int status;          // 400, 404, 500, 501
  std::string type;    // "BadRequestError", "Bad Request" (pydantic RequestValidationError), "InternalServerError",
                       // "NotImplementedError", "invalid_request_error" (Responses 404)
  Json param;          // null or string
  std::string message; // already sanitized (serve/exception_handling/utils.py sanitize_message)
  RenderError(int s, std::string t, Json p, std::string m)
      : std::runtime_error(m), status(s), type(std::move(t)), param(std::move(p)), message(std::move(m)) {}
  Json error_body() const;  // {"error": {...}} in ErrorResponse.model_dump() key order
};

struct RenderImage {
  std::string url;          // image_url.url / input_image.image_url (data: URI or URL), "" if absent
  Json uuid;                // null or str
  size_t message_index;     // index in the request messages
};
// Callbacks that let the caller run its own steps at the exact point the Python frontend runs them. A callback may
// throw RenderError (or anything else); render_* lets it propagate unchanged.
struct RenderHooks {
  // After pydantic validation succeeded, before anything else: OpenAIServing._check_model (404 for an unknown model)
  // and, for /v1/responses, nothing else (the store lookup result is passed as ResponsesPrev).
  std::function<void()> after_validation;
  // After the content parts were parsed (images known), before the chat template: MultiModalItemTracker.resolve_items
  // fetches and decodes the images here, so an image error wins over a template error, as in Python.
  std::function<void(const std::vector<RenderImage>&)> before_template;
};

struct RenderOptions {
  int64_t max_model_len = 655360;
  std::string dsml_strict_env = "vllm";  // env DSML_STRICT of the API server (dsml_roundtrip v2.2)
  int64_t max_token_id = 129279;         // max(tokenizer.max_token_id, hf vocab_size - 1): kv_transfer_params ids check
  const RenderHooks* hooks = nullptr;
};


struct ChatRender {
  std::string prompt;            // the rendered prompt text (encode with add_special_tokens below)
  bool thinking = true;          // thinking_mode == "thinking": the output starts inside the reasoning
  std::string effort;            // str(reasoning_effort) as rendered: "high", "max", "low", "xhigh" or "1".."100"
  Json effort_value;             // the same as a Json (str or int)
  bool drop_thinking = true;     // truthiness of chat_template_kwargs.drop_thinking (before the tools override)
  Json chat_template_kwargs;     // effective kwargs: the parser's and EngineCoreRequest.reasoning_parser_kwargs
                                 // {"chat_template_kwargs": ...} (serving.py _effective_chat_template_kwargs)
  Json tools;                    // request tools after DSML_STRICT, ChatCompletionToolsParam.model_dump() form, or null
  Json tool_choice;              // resolved tool_choice ("none" / "auto" / "required" / {named}) or null
  bool add_special_tokens = false;
  std::vector<RenderImage> images;  // in placeholder order
  Json messages;                 // /v1/responses: the message list msg_store keeps for this response (else null)
  std::optional<std::vector<int32_t>> reused_prompt_token_ids;  // kv_transfer_params.prompt_token_ids: no render
  bool store = false;            // /v1/responses: request.store as validated (before the env-flag downgrade)
};

// /v1/chat/completions. Throws RenderError.
ChatRender render_chat(const Json& request, const RenderOptions& opt = {});
ChatRender render_chat(Json&& request, const RenderOptions& opt = {});  // no copy of the body
// DESIGN.md signature. The tokenizer is not needed for the prompt string (it is accepted for interface stability).
ChatRender render_chat(const Json& request, const Tokenizer& tok);

// /v1/responses. `prev` is the store entry of previous_response_id (looked up by the caller):
//   prev == nullptr with previous_response_id set -> RenderError 404 "Response with id '...' not found."
//   prev->messages: msg_store[prev_id] (may be null when the messages were evicted), prev->output: response.output
struct ResponsesPrev {
  const Json* messages = nullptr;  // array or null
  const Json* output = nullptr;    // array of output items (ResponsesResponse.output JSON)
};
ChatRender render_responses(const Json& request, const ResponsesPrev* prev, const RenderOptions& opt = {});
ChatRender render_responses(Json&& request, const ResponsesPrev* prev, const RenderOptions& opt = {});

// /tokenize, messages form (TokenizeChatRequest). The prompt form is not a render.
ChatRender render_tokenize_chat(const Json& request, const RenderOptions& opt = {});

// The bare encoder (deepseek_v41_encoding.encode_messages) for tests and tools. Throws RenderError.
std::string encode_messages_v41(const Json& messages, bool thinking, bool drop_thinking, const Json& reasoning_effort);

}  // namespace fe
