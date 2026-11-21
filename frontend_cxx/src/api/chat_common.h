// Shared request-side helpers for chat / completions / responses / tokenize.
#pragma once
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "api/backend.h"
#include "api/pyd.h"
#include "render/render.h"
#include "common/json.h"

namespace fe::api {

class App;
struct Ctx;

// ---- pydantic field helpers (write coerced values back into the body) ----
void coerce_int(Json& d, Pyd& v, const char* f, bool nullable, std::optional<int64_t> ge = std::nullopt,
                std::optional<int64_t> le = std::nullopt);
void coerce_float(Json& d, Pyd& v, const char* f, bool nullable = true);
void coerce_bool(Json& d, Pyd& v, const char* f, bool nullable);
void validate_int_list(Pyd& v, Json& d, const char* f, bool nullable);
void validate_str_list(Pyd& v, Json& d, const char* f, bool nullable);
void validate_stop(Pyd& v, const Json* stop);
void validate_messages(Pyd& v, const Json& msgs);
void validate_tools(Pyd& v, const Json& tools);
// CompletionRequest.prompt (completions.cpp): the union with token ids >= 0; errors in pydantic's order
void validate_completion_prompt(Pyd& v, const Json* prompt);
// validate_cache_salt() (before-validator: raises VLLMValidationError)
void validate_cache_salt(const Json* salt);
// cache_salt: str | None, min_length=1, max_length=1024 (field)
void validate_cache_salt_field(Pyd& v, const Json& d);

// ---- prompt preparation ----
struct Prepared {
  std::string prompt;             // rendered prompt text
  Json chat_template_kwargs;      // effective kwargs (parser + EngineCoreRequest[18])
  std::vector<int32_t> prompt_ids;
  std::vector<std::shared_ptr<const fe::ImageFeatures>> images;
  std::vector<int64_t> image_offsets;
  int64_t mm_tokens = 0;          // usage.prompt_tokens_details.multimodal_tokens.image
};

// encode (B) -> images (F) -> TokenizeParams length check. Throws ApiError.
Prepared prepare_prompt(App& app, const fe::ChatRender& r, bool check_len = true);
// TokenizeParams checks (max_output_tokens vs max_model_len; prompt vs max_model_len).
void check_output_tokens_param(const App& app, std::optional<int64_t> max_out, const char* param);
void check_prompt_len(const App& app, size_t n_tokens);
int64_t get_max_tokens(int64_t max_model_len, std::optional<int64_t> requested, int64_t prompt_len);
int64_t app_max_model_len(const App& app);

GenRequest make_gen_request(App& app, Ctx& c, const Json& body, const Prepared& pp, const std::string& endpoint,
                            const std::string& external_id);
// reasoning_ended / reasoning_parser_kwargs for chat-like endpoints
void set_reasoning_fields(App& app, GenRequest& g, const Json& body, const Prepared& pp);
// OnlineRenderer.preprocess_chat -> parser.adjust_request (chat + responses): with the deepseek_v41 ParserEngine
// adapters configured, should_adjust_request = reasoning parser set or tool_choice != "none", and
// ParserEngine.adjust_request sets request.skip_special_tokens = False. The streaming DSML scanner relies on it: a
// special-token id (e.g. EOS under ignore_eos) whose text is missing from delta_text is deferred and all later text
// is held until the end of the stream. Call before make_gen_request.
void apply_parser_adjust_request(const App& app, Json& body, bool responses = false);

// ---- non-streaming accumulation: calls `done(output)` once on the backend's thread ----
struct FullAcc : std::enable_shared_from_this<FullAcc> {
  std::mutex mu;
  GenOutput all;
  bool any = false, fired = false;
  std::function<void(GenOutput&&)> done;
  void on(GenOutput&& o);
  void cancel();
};

}  // namespace fe::api
