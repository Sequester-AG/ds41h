// RENDER component: vLLM 0.32.0 chat / responses / tokenize request path up to the prompt string.
// See render.h for the order of operations. Python references are given as file:line of the installed venv.
#include "render/render.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>

#include "render/encoding.h"
#include "render/pyd.h"
#include "render/pyval.h"

namespace fe {

using render::PyError;
using render::py_hashable;
using render::py_repr;
using render::py_repr_str;
using render::py_str;
using render::py_truthy;
using render::py_type_name;
using render::raise_py;
using render::raise_vllm;
namespace pyd = render::pyd;

Json RenderError::error_body() const {
  Json e = Json::object();
  e.set("message", message);
  e.set("type", type);
  e.set("param", param);
  e.set("code", Json(static_cast<int64_t>(status)));
  return Json::object({{"error", std::move(e)}});
}

namespace {

// A FastAPI RequestValidationError (pydantic errors with loc prefixed by "body").
struct ReqValidation {
  pyd::Errors errs;
};

// pydantic_core.ValidationError raised and turned into a 400 by create_error_response (str(exc)).
[[noreturn]] void raise_native(const std::string& title, const pyd::Errors& errs) {
  raise_py("ValidationError", pyd::format_native_errors(title, errs));
}

RenderError to_render_error(const PyError& e) {
  const std::string& c = e.cls;
  std::string msg = render::sanitize_message(e.str());
  if (c == "VLLMValidationError") return RenderError(400, "BadRequestError", e.param ? Json(*e.param) : Json(), msg);
  if (c == "ValueError" || c == "TypeError" || c == "OverflowError" || c == "ValidationError" ||
      c == "JSONDecodeError" || c == "UnicodeDecodeError")
    return RenderError(400, "BadRequestError", Json(), msg);
  if (c == "NotImplementedError") return RenderError(501, "NotImplementedError", Json(), msg);
  // OpenAIServingResponses.create_error_response(err_type="invalid_request_error", ...)
  if (c == "InvalidRequest404") return RenderError(404, "invalid_request_error", e.param ? Json(*e.param) : Json(), msg);
  if (c == "InvalidRequest400") return RenderError(400, "invalid_request_error", e.param ? Json(*e.param) : Json(), msg);
  return RenderError(500, "InternalServerError", Json(), msg);
}

RenderError to_render_error(const ReqValidation& v) {
  pyd::Errors errs = v.errs;
  for (auto& e : errs) e.loc.insert(e.loc.begin(), Json("body"));
  std::string param;
  std::string msg = render::sanitize_message(pyd::format_request_errors(errs, &param));
  return RenderError(400, "Bad Request", errs.empty() ? Json() : Json(param), msg);
}

// ------------------------------------------------------------------ Python helpers on request data
const Json& get(const Json& d, std::string_view k) {
  if (!d.is_object()) return Json::null_ref();
  const Json* p = d.get(k);
  return p ? *p : Json::null_ref();
}
bool has(const Json& d, std::string_view k) { return d.is_object() && d.get(k) != nullptr; }
bool is_str(const Json& v, std::string_view s) { return v.is_string() && v.as_string() == s; }

// `for x in v` over a JSON value (TypeError when not iterable)
JsonArray py_iter(const Json& v) {
  if (v.is_array()) return v.as_array();
  JsonArray out;
  if (v.is_object()) {
    for (const auto& [k, x] : v.as_object()) out.emplace_back(k);
    return out;
  }
  if (v.is_string()) {
    const std::string& s = v.as_string();
    for (size_t i = 0; i < s.size();) {
      size_t j = i + 1;
      while (j < s.size() && (static_cast<unsigned char>(s[j]) & 0xC0) == 0x80) ++j;
      out.emplace_back(std::string_view(s).substr(i, j - i));
      i = j;
    }
    return out;
  }
  raise_py("TypeError", std::string("'") + py_type_name(v) + "' object is not iterable");
}

// renderers/params.py merge_kwargs: defaults | {k: v for k, v in overrides.items() if v not in (None, "auto")}
Json merge_kwargs(const Json& defaults, const Json& overrides) {
  Json out = defaults.is_object() ? defaults : Json::object();
  if (overrides.is_object())
    for (const auto& [k, v] : overrides.as_object())
      if (!v.is_null() && !is_str(v, "auto")) out.set(k, v);
  return out;
}

// ------------------------------------------------------------------ chat_utils constants
constexpr std::string_view kMMParserTypes[] = {"text", "thinking", "input_text", "output_text", "input_image",
                                               "image_url", "image_embeds", "audio_embeds", "video_embeds",
                                               "prompt_embeds", "image_pil", "audio_url", "input_audio", "refusal",
                                               "video_url", "tool_reference"};
constexpr std::string_view kTextPartTypes[] = {"text", "input_text", "output_text", "refusal", "thinking"};
constexpr std::string_view kMediaKeys[] = {"input_image", "image_url", "image_embeds", "audio_embeds", "video_embeds",
                                           "prompt_embeds", "image_pil", "audio_url", "input_audio", "video_url"};
constexpr std::string_view kTextContentTypes[] = {"text", "input_text", "output_text", "refusal", "thinking",
                                                  "tool_reference"};

template <size_t N>
bool in_set(const Json& v, const std::string_view (&set)[N]) {
  if (!v.is_string()) return false;
  for (auto s : set)
    if (v.as_string() == s) return true;
  return false;
}

// chat_utils.py has_non_text_content
bool has_non_text_content(const Json& messages) {
  if (!messages.is_array()) return false;
  for (const Json& msg : messages.as_array()) {
    const Json& content = msg.is_object() ? get(msg, "content") : Json::null_ref();
    if (!content.is_array()) continue;
    for (const Json& part : content.as_array()) {
      if (!part.is_object()) continue;
      for (auto k : kMediaKeys)
        if (part.get(k)) return true;
      const Json* t = part.get("type");
      if (!t) continue;  // default "text"
      if (!t->is_string() || !in_set(*t, kTextContentTypes)) return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------ request schemas (pydantic field validation)
struct ReqSchemas {
  pyd::Builder b;
  const pyd::Schema* chat;       // ChatCompletionRequest minus `messages`
  const pyd::Schema* tokenize;   // TokenizeChatRequest minus `messages`
  const pyd::Schema* responses;  // ResponsesRequest minus `input`
};

const ReqSchemas& req_schemas() {
  static ReqSchemas* rs = [] {
    auto* r = new ReqSchemas();
    pyd::Builder& b = r->b;
    using pyd::K;
    const auto* str = b.mk(K::Str);
    const auto* integer = b.mk(K::Int);
    const auto* flt = b.mk(K::Float);
    const auto* boolean = b.mk(K::Bool);
    const auto* any = b.mk(K::Any);
    auto opt = [&](const pyd::Schema* s) { return b.wrap(K::Nullable, s); };
    const auto* dict_any = b.mk(K::Dict);
    Json i64min = Json::integer_from_text("-9223372036854775808"), i64max(INT64_MAX), m1(int64_t{-1}), z(int64_t{0}),
         one(int64_t{1}), m2(-2.0), p2(2.0);
    const auto* i64 = b.int_range(&i64min, &i64max);
    const auto* list_int = b.wrap(K::List, integer);
    const auto* list_str = b.wrap(K::List, str);
    const auto* stop = opt(b.uni({str, list_str}));
    const auto* cache_salt = [&] {
      auto* s = b.mk(K::Str);
      s->min_length = 1;
      s->max_length = 1024;
      return s;
    }();
    const auto& S = pyd::schemas();
    const auto* tool_choice = opt(b.uni({b.lit({"none"}), b.lit({"auto"}), b.lit({"required"}), S.named_tool_choice}));
    // StreamOptions (OpenAIBaseModel): a model, so a non-dict is model_attributes_type and the two flags are bools (A)
    const auto* stream_options =
        b.td("StreamOptions", {{"include_usage", opt(boolean), false}, {"continuous_usage_stats", opt(boolean), false}},
             K::Model);
    // openai.types.responses.Tool: the 16-member union as pydantic sees it.
    // Only required fields and the `type` / `environment` literals are typed; optional fields accept anything.
    const auto* responses_tool = [&] {
      auto M = [&](const char* name, std::initializer_list<pyd::Field> f) { return b.td(name, f, K::Model); };
      return b.uni({
          M("FunctionTool", {{"name", str, true}, {"parameters", any, false}, {"strict", any, false},
                             {"type", b.lit({"function"}), true}}),
          M("FileSearchTool", {{"type", b.lit({"file_search"}), true}, {"vector_store_ids", any, true}}),
          M("ComputerTool", {{"type", b.lit({"computer"}), true}}),
          M("ComputerUsePreviewTool", {{"display_height", any, true}, {"display_width", any, true},
                                       {"environment", b.lit({"windows", "mac", "linux", "ubuntu", "browser"}), true},
                                       {"type", b.lit({"computer_use_preview"}), true}}),
          M("WebSearchTool", {{"type", b.lit({"web_search", "web_search_2025_08_26"}), true}}),
          M("Mcp", {{"server_label", any, true}, {"type", b.lit({"mcp"}), true}}),
          M("CodeInterpreter", {{"container", any, true}, {"type", b.lit({"code_interpreter"}), true}}),
          M("ProgrammaticToolCalling", {{"type", b.lit({"programmatic_tool_calling"}), true}}),
          M("ImageGeneration", {{"type", b.lit({"image_generation"}), true}}),
          M("LocalShell", {{"type", b.lit({"local_shell"}), true}}),
          M("FunctionShellTool", {{"type", b.lit({"shell"}), true}}),
          M("CustomTool", {{"name", any, true}, {"type", b.lit({"custom"}), true}}),
          M("NamespaceTool", {{"description", any, true}, {"name", any, true}, {"tools", any, true},
                              {"type", b.lit({"namespace"}), true}}),
          M("ToolSearchTool", {{"type", b.lit({"tool_search"}), true}}),
          M("WebSearchPreviewTool", {{"type", b.lit({"web_search_preview", "web_search_preview_2025_03_11"}), true}}),
          M("ApplyPatchTool", {{"type", b.lit({"apply_patch"}), true}}),
      });
    }();
    r->chat = b.td(
        "ChatCompletionRequest",
        {{"model", opt(str), false},
         {"frequency_penalty", opt(flt), false},
         {"logit_bias", opt(b.wrap(K::Dict, flt)), false},
         {"logprobs", opt(boolean), false},
         {"top_logprobs", any, false},
         {"max_tokens", opt(integer), false},
         {"max_completion_tokens", opt(integer), false},
         {"n", opt(integer), false},
         {"presence_penalty", opt(flt), false},
         {"response_format", any, false},
         {"seed", opt(i64), false},
         {"stop", stop, false},
         {"stream", opt(boolean), false},
         {"stream_options", opt(stream_options), false},
         {"temperature", opt(flt), false},
         {"top_p", opt(flt), false},
         {"tools", opt(b.wrap(K::List, S.tools_param)), false},
         {"tool_choice", tool_choice, false},
         {"reasoning_effort", opt(b.lit({"none", "minimal", "low", "medium", "high", "xhigh", "max"})), false},
         {"thinking_token_budget", any, false},
         {"include_reasoning", boolean, false},
         {"parallel_tool_calls", opt(boolean), false},
         {"user", opt(str), false},
         {"use_beam_search", boolean, false},
         {"top_k", opt(integer), false},
         {"min_p", opt(flt), false},
         {"repetition_penalty", opt(flt), false},
         {"watermarking", opt(boolean), false},
         {"length_penalty", flt, false},
         {"stop_token_ids", opt(list_int), false},
         {"include_stop_str_in_output", boolean, false},
         {"ignore_eos", boolean, false},
         {"min_tokens", integer, false},
         {"skip_special_tokens", boolean, false},
         {"spaces_between_special_tokens", boolean, false},
         {"truncate_prompt_tokens", opt(b.int_range(&m1, &i64max)), false},
         {"truncation_side", opt(b.lit({"left", "right"})), false},
         {"prompt_logprobs", opt(integer), false},
         {"logprob_token_ids", opt(list_int), false},
         {"allowed_token_ids", opt(list_int), false},
         {"bad_words", list_str, false},
         {"echo", boolean, false},
         {"add_generation_prompt", boolean, false},
         {"continue_final_message", boolean, false},
         {"add_special_tokens", boolean, false},
         {"documents", opt(b.wrap(K::List, b.wrap(K::Dict, str))), false},
         {"chat_template", opt(str), false},
         {"chat_template_kwargs", opt(dict_any), false},
         {"media_io_kwargs", opt(b.wrap(K::Dict, dict_any)), false},
         {"mm_processor_kwargs", opt(dict_any), false},
         {"structured_outputs", any, false},
         {"priority", i64, false},
         {"request_id", str, false},
         {"session_id", opt(str), false},
         {"return_tokens_as_token_ids", opt(boolean), false},
         {"return_token_ids", opt(boolean), false},
         {"routed_experts_prompt_start", b.int_range(&z, nullptr), false},
         {"return_token_offsets", opt(boolean), false},
         {"return_prompt_text", opt(boolean), false},
         {"mm_payload", b.lit({"tensors", "none"}), false},
         {"return_mm_kwargs", opt(boolean), false},
         {"cache_salt", opt(cache_salt), false},
         {"kv_transfer_params", opt(dict_any), false},
         {"ec_transfer_params", opt(dict_any), false},
         {"vllm_xargs", any, false},
         {"repetition_detection", any, false},
         {"stream_interval", opt(b.int_range(&one, nullptr)), false}},
        K::Model);
    r->tokenize = b.td("TokenizeChatRequest",
                       {{"model", opt(str), false},
                        {"add_generation_prompt", boolean, false},
                        {"return_token_strs", opt(boolean), false},
                        {"continue_final_message", boolean, false},
                        {"add_special_tokens", boolean, false},
                        {"chat_template", opt(str), false},
                        {"chat_template_kwargs", opt(dict_any), false},
                        {"media_io_kwargs", opt(b.wrap(K::Dict, dict_any)), false},
                        {"mm_processor_kwargs", opt(dict_any), false},
                        {"tools", opt(b.wrap(K::List, S.tools_param)), false}},
                       K::Model);
    const auto* reasoning = b.td(
        "Reasoning",
        {{"context", opt(b.lit({"auto", "current_turn", "all_turns"})), false},
         {"effort", opt(b.lit({"none", "minimal", "low", "medium", "high", "xhigh", "max"})), false},
         {"generate_summary", opt(b.lit({"auto", "concise", "detailed"})), false},
         {"summary", opt(b.lit({"auto", "concise", "detailed"})), false}},
        K::Model);
    const auto* pen = [&] {
      auto* s = b.mk(K::Float);
      return s;
    }();
    r->responses = b.td("ResponsesRequest",
                        {{"background", opt(boolean), false},
                         {"include", opt(b.wrap(K::List, any)), false},
                         {"instructions", opt(str), false},
                         {"max_output_tokens", opt(integer), false},
                         {"max_tool_calls", opt(integer), false},
                         {"metadata", any, false},
                         {"model", opt(str), false},
                         {"logit_bias", opt(b.wrap(K::Dict, flt)), false},
                         {"parallel_tool_calls", opt(boolean), false},
                         {"previous_response_id", opt(str), false},
                         {"prompt", any, false},
                         {"reasoning", opt(reasoning), false},
                         {"include_reasoning", boolean, false},
                         {"service_tier", b.lit({"auto", "default", "flex", "scale", "priority"}), false},
                         {"store", opt(boolean), false},
                         {"stream", opt(boolean), false},
                         {"temperature", opt(flt), false},
                         {"text", any, false},
                         {"tool_choice", any, false},
                         {"tools", b.wrap(K::List, responses_tool), false},
                         {"top_logprobs", any, false},
                         {"top_p", opt(flt), false},
                         {"top_k", opt(integer), false},
                         {"min_p", opt(flt), false},
                         {"truncation", opt(b.lit({"auto", "disabled"})), false},
                         {"user", opt(str), false},
                         {"skip_special_tokens", boolean, false},
                         {"include_stop_str_in_output", boolean, false},
                         {"presence_penalty", opt(pen), false},
                         {"frequency_penalty", opt(pen), false},
                         {"prompt_cache_key", opt(str), false},
                         {"watermarking", opt(boolean), false},
                         {"request_id", str, false},
                         {"session_id", opt(str), false},
                         {"media_io_kwargs", opt(b.wrap(K::Dict, dict_any)), false},
                         {"mm_processor_kwargs", opt(dict_any), false},
                         {"priority", i64, false},
                         {"mm_payload", b.lit({"tensors", "none"}), false},
                         {"return_mm_kwargs", opt(boolean), false},
                         {"cache_salt", opt(cache_salt), false},
                         {"enable_response_messages", boolean, false},
                         {"previous_input_messages", opt(b.wrap(K::List, any)), false},
                         {"structured_outputs", any, false},
                         {"repetition_penalty", opt(flt), false},
                         {"seed", opt(i64), false},
                         {"stop", stop, false},
                         {"stop_token_ids", opt(list_int), false},
                         {"ignore_eos", boolean, false},
                         {"vllm_xargs", any, false},
                         {"kv_transfer_params", opt(dict_any), false},
                         {"ec_transfer_params", opt(dict_any), false},
                         {"chat_template_kwargs", opt(dict_any), false}},
                        K::Model);
    (void)m2;
    (void)p2;
    return r;
  }();
  return *rs;
}

// Field validation of a request model. Returns the validated data (lax coercions applied).
Json validate_fields(const pyd::Schema& model, const Json& data, pyd::Errors& errs) {
  pyd::Result r = pyd::validate(model, data, {}, errs);
  return r.coerced ? r.value : data;
}

// ------------------------------------------------------------------ validated messages
struct Msg {
  Json m;                     // the message dict as validated (a copy of the request's)
  int branch = -1;            // ChatCompletionMessageParam member
  bool lazy_content = false;  // content is a ValidatorIterator (OpenAI member, list / dict content)
  bool lazy_tool_calls = false;
};

// list[ChatCompletionMessageParam] (field `messages`). Appends errors; returns validated messages.
// The messages are moved out of `data` (the field is not read again).
std::vector<Msg> validate_messages(Json& data, const char* field, pyd::Errors& errs) {
  std::vector<Msg> out;
  Json* msgs = data.get(field);
  if (!msgs) {
    errs.push_back(pyd::Error{"missing", {Json(field)}, "Field required", data, Json()});
    return out;
  }
  if (!msgs->is_array()) {
    errs.push_back(pyd::Error{"list_type", {Json(field)}, "Input should be a valid list", *msgs, Json()});
    return out;
  }
  const auto& S = pyd::schemas();
  JsonArray& a = msgs->as_array();
  out.reserve(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    pyd::Result r = pyd::validate_consume(*S.message_union, a[i], {Json(field), Json(static_cast<int64_t>(i))}, errs);
    if (!r.ok) continue;
    Msg m;
    m.m = r.coerced ? std::move(r.value) : std::move(a[i]);
    m.branch = r.member;
    if (m.branch < pyd::kCustom) {
      const Json& c = get(m.m, "content");
      m.lazy_content = c.is_array() || c.is_object();
      if (m.branch == pyd::kAssistant) m.lazy_tool_calls = has(m.m, "tool_calls") && !get(m.m, "tool_calls").is_null();
    }
    out.push_back(std::move(m));
  }
  return out;
}

// chat_completion/protocol.py:581 _materialize_tool_calls_after: list(ValidatorIterator) of lazy tool_calls.
void materialize_tool_calls(std::vector<Msg>& msgs) {
  const auto& S = pyd::schemas();
  for (Msg& m : msgs) {
    if (!m.lazy_tool_calls) continue;
    const Json& tcs = get(m.m, "tool_calls");
    JsonArray items = py_iter(tcs);
    for (size_t i = 0; i < items.size(); ++i) {
      pyd::Errors errs;
      if (!pyd::validate(*S.tool_call_union, items[i], {Json(static_cast<int64_t>(i))}, errs).ok)
        throw ReqValidation{std::move(errs)};
    }
    m.m.set("tool_calls", Json(std::move(items)));
    m.lazy_tool_calls = false;
  }
}

// ------------------------------------------------------------------ ChatCompletionRequest before-validators
// pydantic runs mode="before" model validators in reverse definition order (protocol.py:541-1100).
void chat_before_validators(Json& data) {
  if (!data.is_object()) return;
  // check_system_message_content_type (:1053): only iterates (TypeError for a non-iterable `messages`)
  {
    const Json* m = data.get("messages");
    if (m) (void)py_iter(*m);
  }
  // drop_prompt_token_ids_with_media (:1033)
  {
    Json* kv = data.get("kv_transfer_params");
    if (kv && kv->is_object()) {
      const Json* ids = kv->get("prompt_token_ids");
      if (ids && !ids->is_null() && has_non_text_content(get(data, "messages"))) kv->erase("prompt_token_ids");
    }
  }
  // check_generation_prompt (:1021)
  if (py_truthy(get(data, "continue_final_message")) && py_truthy(get(data, "add_generation_prompt")))
    raise_vllm("Cannot set both `continue_final_message` and `add_generation_prompt` to True.");
  // check_tool_usage (:935)
  {
    const Json& tools = get(data, "tools");
    if (tools.is_array() && tools.as_array().empty())
      raise_vllm("`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.",
                 "tools");
    if (!has(data, "tool_choice") && py_truthy(tools)) data.set("tool_choice", "auto");
    const Json* tc = data.get("tool_choice");
    if (!(tc && is_str(*tc, "none")) && tc && !tc->is_null()) {
      Json choice = *tc;
      if (!has(data, "tools") || get(data, "tools").is_null())
        raise_vllm("When using `tool_choice`, `tools` must be set.", "tool_choice");
      if (!is_str(choice, "auto") && !is_str(choice, "required") && !choice.is_object())
        raise_vllm("Invalid value for `tool_choice`: " + py_str(choice) +
                       "! Only named tools, \"none\", \"auto\" or \"required\" are supported.",
                   "tool_choice");
      const char* usage = "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
      if (choice.is_object()) {
        const Json& function = get(choice, "function");
        if (!function.is_object())
          raise_vllm("Invalid value for `function`: `" + py_str(function) + "` in `tool_choice`! " + usage,
                     "tool_choice.function");
        if (!has(function, "name"))
          raise_vllm(std::string("Expected field `name` in `function` in `tool_choice`! ") + usage,
                     "tool_choice.function.name");
        const Json& fname = get(function, "name");
        if (!fname.is_string() || fname.as_string().empty())
          raise_vllm("Invalid `name` in `function`: `" + py_str(fname) + "` in `tool_choice`! " + usage,
                     "tool_choice.function.name");
        bool valid = false;
        for (const Json& tool : py_iter(get(data, "tools"))) {
          if (tool.is_string()) raise_py("TypeError", "string indices must be integers, not 'str'");
          if (tool.is_array()) raise_py("TypeError", "list indices must be integers or slices, not str");
          if (!tool.is_object())
            raise_py("TypeError", std::string("'") + py_type_name(tool) + "' object is not subscriptable");
          const Json* fn = tool.get("function");
          if (!fn) raise_py("KeyError", "'function'");
          if (fn->is_string()) raise_py("TypeError", "string indices must be integers, not 'str'");
          if (!fn->is_object())
            raise_py("TypeError", std::string("'") + py_type_name(*fn) + "' object is not subscriptable");
          const Json* n = fn->get("name");
          if (!n) raise_py("KeyError", "'name'");
          if (n->py_equal(fname)) { valid = true; break; }
        }
        if (!valid)
          raise_vllm("The tool specified in `tool_choice` does not match any of the specified `tools`", "tool_choice");
      }
    }
  }
  // check_structured_outputs_count (:892)
  {
    const Json& so = get(data, "structured_outputs");
    if (!so.is_null()) {
      if (!so.is_object()) raise_py("AttributeError", std::string("'") + py_type_name(so) + "' object has no attribute 'get'");
      int count = 0;
      for (const char* k : {"json", "regex", "choice"})
        if (!get(so, k).is_null()) ++count;
      if (count > 1) raise_vllm("You can only use one kind of constraints for structured outputs ('json', 'regex' or 'choice').");
      const Json* tc = data.get("tool_choice");
      Json choice = tc ? *tc : Json("none");
      if (count > 0 && !(is_str(choice, "none") || is_str(choice, "auto") || is_str(choice, "required")))
        raise_vllm("You can only either use constraints for structured outputs or tools, not both.");
    }
  }
  // check_logprobs (:834)
  {
    if (py_truthy(get(data, "logprob_token_ids")) && py_truthy(get(data, "use_beam_search")))
      raise_vllm("`logprob_token_ids` is not supported with beam search.", "logprob_token_ids");
    if (py_truthy(get(data, "logprob_token_ids")) && !py_truthy(get(data, "logprobs")))
      raise_vllm("when using `logprob_token_ids`, `logprobs` must be set to true.", "logprob_token_ids");
    for (const char* f : {"prompt_logprobs", "top_logprobs"}) {
      const Json& v = get(data, f);
      if (!v.is_null() && !v.is_number() && !v.is_bool())
        raise_vllm(std::string("`") + f + "` must be an integer.", f, py_str(v));
    }
    auto num = [](const Json& v) { return v.is_bool() ? (v.as_bool() ? 1.0 : 0.0) : v.as_double(); };
    const Json& pl = get(data, "prompt_logprobs");
    if (!pl.is_null()) {
      double p = num(pl);
      if (py_truthy(get(data, "stream")) && (p > 0 || p == -1))
        raise_vllm("`prompt_logprobs` are not available when `stream=True`.", "prompt_logprobs");
      if (p < 0 && p != -1) raise_vllm("`prompt_logprobs` must be a positive value or -1.", "prompt_logprobs", py_str(pl));
    }
    const Json& tl = get(data, "top_logprobs");
    if (!tl.is_null()) {
      double t = num(tl);
      if (t < 0 && t != -1) raise_vllm("`top_logprobs` must be a positive value or -1.", "top_logprobs", py_str(tl));
      if ((t == -1 || t > 0) && !py_truthy(get(data, "logprobs")))
        raise_vllm("when using `top_logprobs`, `logprobs` must be set to true.", "top_logprobs");
    }
  }
  // validate_stream_options (:821)
  if (py_truthy(get(data, "stream_options")) && !py_truthy(get(data, "stream")))
    raise_vllm("Stream options can only be defined when `stream=True`.", "stream_options");
  // validate_response_format (:788)
  {
    const Json& rf = get(data, "response_format");
    if (!rf.is_null()) {
      const Json& t = rf.is_object() ? get(rf, "type") : Json::null_ref();
      if (is_str(t, "json_schema") && get(rf, "json_schema").is_null())
        raise_vllm("When response_format type is 'json_schema', the 'json_schema' field must be provided.",
                   "response_format");
    }
  }
  // _normalize_messages_before (:554)
  {
    Json* msgs = data.get("messages");
    if (msgs && msgs->is_array()) {
      for (Json& msg : msgs->as_array()) {
        if (!msg.is_object()) continue;
        const Json* tcs = msg.get("tool_calls");
        if (tcs && !tcs->is_null() && !tcs->is_array()) msg.set("tool_calls", Json(py_iter(*tcs)));
        Json* rcp = msg.get("reasoning_content");
        Json rc = rcp ? std::move(*rcp) : Json();
        msg.erase("reasoning_content");
        if (!rc.is_null() && get(msg, "reasoning").is_null()) msg.set("reasoning", std::move(rc));
      }
    }
  }
  // check_cache_salt_support (:546)
  {
    const Json& cs = get(data, "cache_salt");
    if (!cs.is_null()) {
      if (!cs.is_string() || cs.as_string().empty())
        raise_vllm("Parameter 'cache_salt' must be a non-empty string if provided.", "cache_salt");
      const std::string& s = cs.as_string();
      if (py_len(s) > 128 || s.find_first_of(std::string("@/\\\0", 4)) != std::string::npos)
        raise_vllm("Parameter 'cache_salt' must be at most 128 characters and must not contain '@', '/', '\\\\', or NUL.",
                   "cache_salt");
    }
  }
  // apply_return_mm_kwargs_alias (:541)
  if (get(data, "return_mm_kwargs").is_bool() && !get(data, "return_mm_kwargs").as_bool() && !has(data, "mm_payload"))
    data.set("mm_payload", "none");
}

// ------------------------------------------------------------------ parse_chat_messages emulation
struct ParseState {
  std::vector<RenderImage> images;
  int n_image = 0;
};

[[noreturn]] void mm_limit(const char* modality, int limit) {
  raise_vllm("At most " + std::to_string(limit) + " " + modality + "(s) may be provided in one prompt.", modality);
}

void track_image(ParseState& st, const Json& url, const Json& uuid, size_t mi) {
  if (++st.n_image > 999) mm_limit("image", 999);
  st.images.push_back(RenderImage{url.is_string() ? url.as_string() : std::string(), uuid, mi});
}

// chat_utils.py:1691 _parse_chat_message_content_mm_part + :1907 _parse_chat_message_content_part
void parse_part(const Json& part, ParseState& st, size_t mi) {
  if (part.is_string()) return;
  if (!part.is_object()) raise_py("AssertionError", "");
  const auto& S = pyd::schemas();
  const Json& part_type = get(part, "type");
  const Json& uuid0 = get(part, "uuid");
  std::string ptype;
  Json content;
  bool have_content = false;
  if (part_type.is_string() && in_set(part_type, kMMParserTypes) && uuid0.is_null()) {
    ptype = part_type.as_string();
    if (ptype == "input_image") {
      pyd::Errors errs;
      if (!pyd::validate(*S.responses_input_image, part, {}, errs).ok) raise_native("ResponseInputImageParam", errs);
      content = get(part, "image_url");
    } else if (ptype == "image_url") {
      pyd::Errors errs;
      if (!pyd::validate(*S.image_part, part, {}, errs).ok) raise_native("ChatCompletionContentPartImageParam", errs);
      content = get(get(part, "image_url"), "url");
    } else if (ptype == "audio_url") {
      pyd::Errors errs;
      if (!pyd::validate(*S.audio_part, part, {}, errs).ok) raise_native("ChatCompletionContentPartAudioParam", errs);
      content = get(get(part, "audio_url"), "url");
    } else if (ptype == "video_url") {
      pyd::Errors errs;
      if (!pyd::validate(*S.video_part, part, {}, errs).ok) raise_native("ChatCompletionContentPartVideoParam", errs);
      content = get(get(part, "video_url"), "url");
    } else if (ptype == "text" || ptype == "input_text" || ptype == "output_text") {
      content = get(part, "text");
    } else if (ptype == "thinking") {
      content = get(part, "thinking");
    } else if (ptype == "refusal") {
      content = get(part, "refusal");
    } else if (ptype == "prompt_embeds") {
      content = get(part, "data");
    } else if (ptype == "tool_reference") {
      content = get(part, "name");
    } else {
      content = get(part, ptype);  // image_embeds / audio_embeds / video_embeds / image_pil / input_audio
    }
    have_content = true;
  }
  if (!have_content) {
    if (part_type.is_null() || !uuid0.is_null()) {
      auto direct = [&](const char* key, const char* type, bool url_dict) {
        if (!part.get(key)) return false;
        ptype = type;
        content = get(part, key);
        if (url_dict && content.is_object()) content = get(content, "url");
        return true;
      };
      if (!(direct("image_url", "image_url", true) || direct("image_pil", "image_pil", false) ||
            direct("image_embeds", "image_embeds", false) || direct("audio_embeds", "audio_embeds", false) ||
            direct("video_embeds", "video_embeds", false))) {
        if (part.get("prompt_embeds")) {
          ptype = "prompt_embeds";
          content = get(part, "data");
        } else if (!(direct("audio_url", "audio_url", true) || direct("input_audio", "input_audio", false) ||
                     direct("video_url", "video_url", true))) {
          if (part.get("tool_reference")) {
            ptype = "tool_reference";
            content = get(part, "name");
          } else {
            raise_vllm("Missing 'type' field in multimodal part.", "type");
          }
        }
      }
    } else if (!part_type.is_string()) {
      raise_vllm("Invalid 'type' field in multimodal part.", "type");
    } else {
      ptype = part_type.as_string();
      content = Json("unknown part_type content");
    }
  }
  if ((ptype == "text" || ptype == "refusal") && content.is_null()) return;  // skipped with a warning
  for (auto t : kTextPartTypes)
    if (ptype == t) return;
  Json uuid = get(part, "uuid");
  if (!uuid.is_null()) uuid = Json(py_str(uuid));
  if (ptype == "image_pil" || ptype == "image_url" || ptype == "input_image") {
    track_image(st, content, uuid, mi);
  } else if (ptype == "image_embeds" || ptype == "audio_embeds" || ptype == "video_embeds") {
    raise_vllm("You must set `--enable-mm-embeds` to input `" + ptype + "`", ptype);
  } else if (ptype == "prompt_embeds") {
    if (!py_truthy(content))
      raise_vllm("prompt_embeds content part requires a non-empty `data` field with base64-encoded tensor bytes.",
                 "prompt_embeds");
    raise_vllm("You must set `--enable-prompt-embeds` to input `prompt_embeds`", "prompt_embeds");
  } else if (ptype == "audio_url" || ptype == "input_audio") {
    mm_limit("audio", 0);
  } else if (ptype == "video_url") {
    mm_limit("video", 0);
  } else if (ptype == "tool_reference") {
    return;
  } else {
    raise_vllm("Unsupported chat content part type: " + py_repr_str(ptype) +
                   ". Supported types: audio_embeds, audio_url, image_embeds, image_pil, image_url, input_audio, "
                   "input_image, input_text, output_text, prompt_embeds, refusal, text, thinking, tool_reference, "
                   "video_embeds, video_url.",
               "type", ptype);
  }
}

// The OpenAI part schema a lazy content iterator validates against, by message member.
const pyd::Schema* lazy_part_schema(int branch) {
  const auto& S = pyd::schemas();
  if (branch == pyd::kUser) return S.openai_user_parts;
  if (branch == pyd::kAssistant) return S.openai_assistant_parts;
  return S.text_part;
}

// next(ValidatorIterator): the validated item (a TypedDict output, keys reordered)
Json validate_lazy_item(const Msg& m, const Json& item, size_t idx) {
  pyd::Errors errs;
  pyd::Result r = pyd::validate(*lazy_part_schema(m.branch), item, {Json(static_cast<int64_t>(idx))}, errs);
  if (!r.ok) raise_native("ValidatorIterator", errs);
  return r.coerced ? std::move(r.value) : item;
}

// chat_utils.py:2230 parse_chat_messages (conversation side): errors, image items, and _postprocess_messages, which
// decodes assistant tool-call `arguments` IN PLACE in the request messages (the conversation shares the dicts).
void parse_chat_messages(std::vector<Msg>& msgs, ParseState& st) {
  for (size_t mi = 0; mi < msgs.size(); ++mi) {
    const Msg& m = msgs[mi];
    const Json* role = m.m.get("role");
    if (!role) raise_py("KeyError", "'role'");
    const Json& content = get(m.m, "content");
    if (m.lazy_content) {
      JsonArray items = py_iter(content);
      if (is_str(*role, "system")) {  // _strip_claude_code_billing_header consumes the iterator first
        for (size_t i = 0; i < items.size(); ++i) items[i] = validate_lazy_item(m, items[i], i);
        for (const Json& p : items) parse_part(p, st, mi);
      } else {
        for (size_t i = 0; i < items.size(); ++i) parse_part(validate_lazy_item(m, items[i], i), st, mi);
      }
    } else if (content.is_array()) {
      for (const Json& p : content.as_array()) parse_part(p, st, mi);
    }
  }
  // _postprocess_messages (chat_utils.py:2153)
  for (Msg& m : msgs) {
    if (!is_str(get(m.m, "role"), "assistant")) continue;
    Json* tcs = m.m.get("tool_calls");
    if (!tcs || tcs->is_null() || !tcs->is_array()) continue;
    if (tcs->as_array().empty()) continue;  // popped from the conversation only
    for (Json& item : tcs->as_array()) {
      if (!item.is_object()) raise_vllm("assistant tool_calls entries must be objects.", "tool_calls");
      Json* function = item.get("function");
      const Json* t = item.get("type");
      bool type_ok = !t || is_str(*t, "function");
      if (!type_ok || !function || !function->is_object())
        raise_vllm("chat completions only support assistant tool_calls of type 'function'.", "tool_calls");
      const Json& args = get(*function, "arguments");
      Json parsed;
      if (py_truthy(args)) {
        if (args.is_object()) {
          parsed = args;
        } else {
          if (args.is_string()) {
            try {
              parsed = json_loads_str(args.as_string());
            } catch (const JsonError& e) {
              if (e.kind == JsonError::Kind::Decode) parsed = Json();
              else if (e.kind == JsonError::Kind::Recursion) raise_py("RecursionError", e.msg);
              else raise_py("ValueError", e.msg);
            }
          } else {
            parsed = args;
          }
          if (!parsed.is_object()) parsed = Json::object();
        }
      } else {
        parsed = Json::object();
      }
      function->set("arguments", std::move(parsed));
    }
  }
}

// ------------------------------------------------------------------ DeepseekV41 apply_chat_template
struct TemplateOut {
  std::string prompt;
  bool thinking;
  Json effort;
  bool drop_thinking;
};

// tokenizers/deepseek_v41.py:53-101 + the plugin's _encode_messages_no_strict (dsml_roundtrip.py:351-366)
TemplateOut apply_chat_template_v41(std::vector<Msg>&& msgs, bool any_lazy, const Json& kwargs) {
  // _normalize_messages: copy.deepcopy(messages) first (a ValidatorIterator cannot be pickled)
  if (any_lazy) raise_py("TypeError", "cannot pickle 'pydantic_core._pydantic_core.ValidatorIterator' object");
  JsonArray conv;
  conv.reserve(msgs.size() + 1);
  for (Msg& mm : msgs) {
    Json message = std::move(mm.m);  // copy.deepcopy(messages): the validated list is not used afterwards
    const Json& role = get(message, "role");
    if (!(is_str(role, "system") || is_str(role, "developer") || is_str(role, "user") || is_str(role, "assistant") ||
          is_str(role, "tool")))
      raise_py("ValueError", "Invalid role: " + py_str(role));
    if (has(message, "reasoning")) message.set("reasoning_content", get(message, "reasoning"));
    const Json& content = get(message, "content");
    if (content.is_array()) {
      JsonArray parts;
      for (const Json& block : content.as_array()) {
        if (!block.is_object())
          raise_py("AttributeError", std::string("'") + py_type_name(block) + "' object has no attribute 'get'");
        const Json& pt = get(block, "type");
        if (is_str(pt, "text") || is_str(pt, "input_text") || is_str(pt, "output_text")) {
          const Json* t = block.get("text");
          parts.push_back(t ? *t : Json(""));
        } else if (is_str(pt, "image_url") || is_str(pt, "input_image") || is_str(pt, "image_pil")) {
          parts.push_back(Json(render::kImagePlaceholder));
        } else {
          raise_py("ValueError", "DeepSeek V4.1 supports text and image content only; got " + py_repr(pt));
        }
      }
      std::string joined;
      for (size_t i = 0; i < parts.size(); ++i) {
        if (!parts[i].is_string())
          raise_py("TypeError", "sequence item " + std::to_string(i) + ": expected str instance, " +
                                    py_type_name(parts[i]) + " found");
        if (i) joined += "\n\n";
        joined += parts[i].as_string();
      }
      message.set("content", Json(std::move(joined)));
    }
    conv.push_back(std::move(message));
  }
  const Json& tools = get(kwargs, "tools");
  if (py_truthy(tools)) {
    size_t si = conv.size();
    for (size_t i = 0; i < conv.size(); ++i)
      if (is_str(get(conv[i], "role"), "system")) { si = i; break; }
    if (si == conv.size()) {
      conv.insert(conv.begin(), Json::object({{"role", Json("system")}, {"content", Json("")}}));
      si = 0;
    }
    conv[si].set("tools", tools);
  }
  bool thinking = py_truthy(get(kwargs, "thinking")) || py_truthy(get(kwargs, "enable_thinking"));
  if (!has(kwargs, "thinking") && !has(kwargs, "enable_thinking")) thinking = true;
  Json effort = get(kwargs, "reasoning_effort");
  if (is_str(effort, "none")) {
    thinking = false;
    effort = Json();
  }
  if (effort.is_null()) effort = Json("high");
  bool valid = (effort.is_int() && effort.as_int() >= 1 && effort.as_int() <= 100) ||
               (effort.is_string() && (effort.as_string() == "low" || effort.as_string() == "high" ||
                                       effort.as_string() == "xhigh" || effort.as_string() == "max"));
  if (!valid)
    raise_py("ValueError",
             "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in "
             "chat_template_kwargs");
  const Json* dt = kwargs.get("drop_thinking");
  bool drop = dt ? py_truthy(*dt) : true;
  // plugin: strip "strict" from the rendered tool schemas of every message with tools
  for (Json& m : conv) {
    const Json& mt = get(m, "tools");
    if (!py_truthy(mt)) continue;
    JsonArray stripped;
    for (const Json& t : py_iter(mt)) {
      const Json& fn = t.is_object() ? get(t, "function") : Json::null_ref();
      if (fn.is_object() && fn.get("strict")) {
        Json nfn = fn;
        nfn.erase("strict");
        Json nt = t;
        nt.set("function", std::move(nfn));
        stripped.push_back(std::move(nt));
      } else {
        stripped.push_back(t);
      }
    }
    m.set("tools", Json(std::move(stripped)));
  }
  TemplateOut out;
  out.prompt = render::encode_messages(std::move(conv), thinking, drop, effort);
  out.thinking = thinking;
  out.effort = effort;
  out.drop_thinking = drop;
  return out;
}

// ------------------------------------------------------------------ request helpers
const Json& field(const Json& req, std::string_view k) { return get(req, k); }

bool bool_field(const Json& req, std::string_view k, bool def) {
  const Json* v = req.get(k);
  return v ? py_truthy(*v) : def;
}

// ChatCompletionToolsParam.model_dump() (protocol.py:191-208 + FunctionDefinition._serialize)
Json dump_tools(const Json& tools) {
  if (tools.is_null()) return Json();
  JsonArray out;
  for (const Json& t : tools.as_array()) {
    const Json& fn = get(t, "function");
    Json f = Json::object();
    f.set("name", get(fn, "name"));
    f.set("description", get(fn, "description"));
    f.set("parameters", get(fn, "parameters"));
    if (!get(fn, "strict").is_null()) f.set("strict", get(fn, "strict"));
    Json defer = get(fn, "defer_loading");
    if (defer.is_null()) defer = get(t, "defer_loading");  // _propagate_defer_loading
    if (!defer.is_null()) f.set("defer_loading", defer);
    Json d = Json::object();
    d.set("type", "function");
    d.set("function", std::move(f));
    if (!get(t, "defer_loading").is_null()) d.set("defer_loading", get(t, "defer_loading"));
    out.push_back(std::move(d));
  }
  return Json(std::move(out));
}

// pydantic model_dump of a model with extra="allow": declared fields (declaration order), then the extras.
Json dump_model(const Json& v, std::initializer_list<std::pair<const char*, Json>> fields) {
  Json out = Json::object();
  for (const auto& [k, def] : fields) out.set(k, has(v, k) ? get(v, k) : def);
  for (const auto& [k, x] : v.as_object()) {
    bool declared = false;
    for (const auto& [fk, d] : fields)
      if (k == fk) declared = true;
    if (!declared) out.set(k, x);
  }
  return out;
}

// ChatCompletionNamedToolChoiceParam.model_dump()
Json dump_named_choice(const Json& tc) {
  if (!tc.is_object()) return tc;
  Json fn = get(tc, "function");
  Json out = dump_model(tc, {{"function", Json()}, {"type", Json("function")}});
  if (fn.is_object()) out.set("function", dump_model(fn, {{"name", Json()}}));
  return out;
}

// renderers/params.py TokenizeParams.__post_init__ via build_tok_params
void check_tok_params(const Json& max_out, const char* param, const Json& truncate, int64_t max_model_len) {
  if (max_out.is_integer() && py_truthy(max_out)) {
    bool gt = max_out.is_big_int() ? max_out.big_int_text()[0] != '-' : max_out.as_int() > max_model_len;
    if (gt)
      raise_vllm(std::string(param) + "=" + py_str(max_out) + " cannot be greater than max_model_len=" +
                     std::to_string(max_model_len) + ". Please request fewer output tokens.",
                 param, py_str(max_out));
  }
  if (truncate.is_integer()) {
    bool gt = truncate.is_big_int() ? truncate.big_int_text()[0] != '-' : truncate.as_int() > max_model_len;
    if (gt)
      raise_vllm("truncate_prompt_tokens=" + py_str(truncate) + " cannot be greater than max_model_len = " +
                     std::to_string(max_model_len) + ". Please request a smaller truncation size.",
                 "truncate_prompt_tokens", py_str(truncate));
  }
}

void check_chat_template(const Json& request_template, const Json& ctk) {
  if (!request_template.is_null() || (py_truthy(ctk) && !get(ctk, "chat_template").is_null()))
    raise_py("ValueError",
             "Chat template is passed with request, but --trust-request-chat-template is not set. Refused request "
             "with untrusted chat template.");
}

void check_mm_kwargs(const Json& req) {
  if (py_truthy(field(req, "mm_processor_kwargs")))
    raise_vllm("Per-request mm_processor_kwargs are disabled by default because they can change multimodal "
               "preprocessing resource usage. Start the server with --trust-request-mm-kwargs only when clients are "
               "trusted.",
               "mm_processor_kwargs");
  if (py_truthy(field(req, "media_io_kwargs")))
    raise_vllm("Per-request media_io_kwargs are disabled by default because they can change multimodal media "
               "loading resource usage. Start the server with --trust-request-mm-kwargs only when clients are "
               "trusted.",
               "media_io_kwargs");
}

Json messages_json(const std::vector<Msg>& msgs);

// online_renderer.py:105 _reused_prompt_token_ids
std::optional<std::vector<int32_t>> reused_ids(Json& req, const std::vector<Msg>& msgs, int64_t max_token_id) {
  Json* kv = req.get("kv_transfer_params");
  if (!kv || !kv->is_object()) return std::nullopt;
  Json ids = get(*kv, "prompt_token_ids");
  kv->erase("prompt_token_ids");
  if (ids.is_null()) return std::nullopt;
  if (py_truthy(field(req, "echo"))) return std::nullopt;
  if (has_non_text_content(messages_json(msgs))) return std::nullopt;
  bool ok = ids.is_array() && !ids.as_array().empty();
  if (ok)
    for (const Json& x : ids.as_array())
      if (!x.is_integer() || (x.is_int() && x.as_int() < 0) || (x.is_big_int() && x.big_int_text()[0] == '-')) ok = false;
  if (!ok)
    raise_vllm("`kv_transfer_params['prompt_token_ids']` must be a non-empty list of non-negative integers.",
               "kv_transfer_params.prompt_token_ids");
  // BaseRenderer.validate_token_ids (renderers/base.py:420): max id vs max(tokenizer.max_token_id, vocab - 1)
  const Json* mx = &ids.as_array()[0];
  for (const Json& x : ids.as_array()) {
    bool bigger = x.is_big_int() ? !mx->is_big_int() || x.big_int_text().size() > mx->big_int_text().size() ||
                                       (x.big_int_text().size() == mx->big_int_text().size() && x.big_int_text() > mx->big_int_text())
                                 : (!mx->is_big_int() && x.as_int() > mx->as_int());
    if (bigger) mx = &x;
  }
  if (mx->is_big_int() || mx->as_int() > max_token_id)
    raise_vllm("Token id " + py_str(*mx) + " is out of vocabulary", "kv_transfer_params.prompt_token_ids");
  std::vector<int32_t> out;
  for (const Json& x : ids.as_array()) out.push_back(static_cast<int32_t>(x.as_int()));
  return out;
}

Json messages_json(const std::vector<Msg>& msgs) {
  JsonArray a;
  for (const Msg& m : msgs) a.push_back(m.m);
  return Json(std::move(a));
}

void fill_template(ChatRender& out, TemplateOut&& t) {
  out.prompt = std::move(t.prompt);
  out.thinking = t.thinking;
  out.effort = py_str(t.effort);
  out.effort_value = t.effort;
  out.drop_thinking = t.drop_thinking;
}

// OnlineRenderer.preprocess_chat (online_renderer.py:752) from the request checks to the prompt.
void preprocess_chat(ChatRender& out, Json& req, std::vector<Msg>& msgs, const Json& default_kwargs,
                     const Json& tool_dicts, const Json& chat_kwargs_built, const Json& max_out, const char* max_param,
                     const Json& truncate, const RenderOptions& opt, bool keep_messages = false) {
  check_mm_kwargs(req);
  Json defaults = merge_kwargs(default_kwargs, Json::object({{"tools", tool_dicts}, {"tokenize", Json(false)}}));
  check_tok_params(max_out, max_param, truncate, opt.max_model_len);
  // build_chat_params(...).with_defaults(defaults): defaults are never empty (tokenize=False)
  Json kwargs = merge_kwargs(defaults, chat_kwargs_built);
  out.reused_prompt_token_ids = reused_ids(req, msgs, opt.max_token_id);
  if (out.reused_prompt_token_ids) {
    if (keep_messages) out.messages = messages_json(msgs);
    return;
  }
  ParseState st;
  parse_chat_messages(msgs, st);
  out.images = std::move(st.images);
  if (keep_messages) out.messages = messages_json(msgs);  // msg_store keeps them after the in-place argument decode
  if (opt.hooks && opt.hooks->before_template) opt.hooks->before_template(out.images);
  bool any_lazy = false;
  for (const Msg& m : msgs) any_lazy = any_lazy || m.lazy_content || m.lazy_tool_calls;
  fill_template(out, apply_chat_template_v41(std::move(msgs), any_lazy, kwargs));
}

// ChatCompletionRequest.build_chat_params (protocol.py:621) -> ChatParams.chat_template_kwargs
Json chat_build_kwargs(const Json& req) {
  Json extra = Json::object();
  extra.set("add_generation_prompt", Json(bool_field(req, "add_generation_prompt", true)));
  extra.set("continue_final_message", Json(bool_field(req, "continue_final_message", false)));
  extra.set("documents", field(req, "documents"));
  extra.set("reasoning_effort", field(req, "reasoning_effort"));
  const Json& user = field(req, "chat_template_kwargs");
  if (!field(req, "reasoning_effort").is_null() && !has(user, "enable_thinking"))
    extra.set("enable_thinking", Json(!is_str(field(req, "reasoning_effort"), "none")));
  return merge_kwargs(user, extra);
}

// dsml_roundtrip.py:325 _api_normalize (on the validated request)
void api_normalize(Json& req, const RenderOptions& opt) {
  Json mode = field(req, "dsml_strict");
  std::string m = py_truthy(mode) ? (mode.is_string() ? mode.as_string() : py_str(mode)) : opt.dsml_strict_env;
  if (py_truthy(mode) && !mode.is_string()) m = "";  // a non-str truthy value never equals "api"
  if (m == "api") {
    Json* tools = req.get("tools");
    if (tools && tools->is_array())
      for (Json& t : tools->as_array()) {
        Json* fn = t.get("function");
        if (fn && fn->is_object() && fn->get("strict") && !fn->get("strict")->is_null()) fn->set("strict", Json());
      }
  }
  const Json& eff = field(req, "reasoning_effort");
  if (eff.is_string()) {
    const std::string& e = eff.as_string();
    const char* mapped = e == "minimal" ? "low" : e == "medium" ? "high" : e == "xhigh" ? "high" : nullptr;
    if (mapped) req.set("reasoning_effort", mapped);
  }
  const Json& th = field(req, "thinking");
  if (th.is_object()) {
    const Json& t = get(th, "type");
    if (is_str(t, "enabled") || is_str(t, "disabled")) {
      Json kw = field(req, "chat_template_kwargs").is_object() ? field(req, "chat_template_kwargs") : Json::object();
      if (!has(kw, "thinking") && !has(kw, "enable_thinking")) {
        bool on = is_str(t, "enabled");
        kw.set("thinking", Json(on));
        kw.set("enable_thinking", Json(on));
        req.set("chat_template_kwargs", std::move(kw));
      }
    }
  }
}

template <class F>
ChatRender guarded(F&& f) {
  try {
    return f();
  } catch (const PyError& e) {
    throw to_render_error(e);
  } catch (const ReqValidation& v) {
    throw to_render_error(v);
  } catch (const JsonError& e) {
    throw RenderError(500, "InternalServerError", Json(), render::sanitize_message(e.what()));
  }
}

// ================================================================== /v1/chat/completions
ChatRender chat_impl(Json data, const RenderOptions& opt) {
  chat_before_validators(data);
  if (!data.is_object())
    throw ReqValidation{{pyd::Error{"model_attributes_type", {},
                                    "Input should be a valid dictionary or object to extract fields from", data, Json()}}};
  pyd::Errors errs;
  std::vector<Msg> msgs = validate_messages(data, "messages", errs);
  Json req = validate_fields(*req_schemas().chat, data, errs);
  if (!errs.empty()) throw ReqValidation{std::move(errs)};
  materialize_tool_calls(msgs);
  if (opt.hooks && opt.hooks->after_validation) opt.hooks->after_validation();

  ChatRender out;
  // OpenAIServingChat._create_chat_completion: _effective_chat_template_kwargs (the plugin normalises here)
  api_normalize(req, opt);
  out.chat_template_kwargs = chat_build_kwargs(req);
  out.tools = dump_tools(field(req, "tools"));
  out.tool_choice = dump_named_choice(has(req, "tool_choice") ? field(req, "tool_choice") : Json("none"));
  out.add_special_tokens = bool_field(req, "add_special_tokens", false);
  // OnlineRenderer.render_chat
  Json tool_dicts = out.tools;
  check_chat_template(field(req, "chat_template"), field(req, "chat_template_kwargs"));
  const Json& mct = field(req, "max_completion_tokens");
  bool use_mct = !mct.is_null();
  preprocess_chat(out, req, msgs, Json::object(), tool_dicts, chat_build_kwargs(req),
                  use_mct ? mct : field(req, "max_tokens"), use_mct ? "max_completion_tokens" : "max_tokens",
                  field(req, "truncate_prompt_tokens"), opt);
  return out;
}

// ================================================================== /tokenize (messages form)
ChatRender tokenize_impl(Json data, const RenderOptions& opt) {
  if (data.is_object() && py_truthy(get(data, "continue_final_message")) && py_truthy(get(data, "add_generation_prompt")))
    raise_vllm("Cannot set both `continue_final_message` and `add_generation_prompt` to True.");
  if (!data.is_object())
    throw ReqValidation{{pyd::Error{"model_attributes_type", {},
                                    "Input should be a valid dictionary or object to extract fields from", data, Json()}}};
  pyd::Errors errs;
  std::vector<Msg> msgs = validate_messages(data, "messages", errs);
  Json req = validate_fields(*req_schemas().tokenize, data, errs);
  if (!errs.empty()) throw ReqValidation{std::move(errs)};
  if (opt.hooks && opt.hooks->after_validation) opt.hooks->after_validation();
  ChatRender out;
  out.tools = dump_tools(field(req, "tools"));
  out.add_special_tokens = bool_field(req, "add_special_tokens", false);
  check_chat_template(field(req, "chat_template"), field(req, "chat_template_kwargs"));
  Json extra = Json::object({{"add_generation_prompt", Json(bool_field(req, "add_generation_prompt", true))},
                             {"continue_final_message", Json(bool_field(req, "continue_final_message", false))}});
  Json built = merge_kwargs(field(req, "chat_template_kwargs"), extra);
  check_mm_kwargs(req);
  Json defaults = merge_kwargs(Json::object(), Json::object({{"tools", out.tools}, {"tokenize", Json(false)}}));
  Json kwargs = merge_kwargs(defaults, built);
  out.chat_template_kwargs = kwargs;
  ParseState st;
  parse_chat_messages(msgs, st);
  out.images = std::move(st.images);
  if (opt.hooks && opt.hooks->before_template) opt.hooks->before_template(out.images);
  bool any_lazy = false;
  for (const Msg& m : msgs) any_lazy = any_lazy || m.lazy_content || m.lazy_tool_calls;
  fill_template(out, apply_chat_template_v41(std::move(msgs), any_lazy, kwargs));
  return out;
}

// ================================================================== /v1/responses
#include "render/responses.inc"

}  // namespace

ChatRender render_chat(const Json& request, const RenderOptions& opt) {
  return guarded([&] { return chat_impl(request, opt); });
}

ChatRender render_chat(Json&& request, const RenderOptions& opt) {
  return guarded([&] { return chat_impl(std::move(request), opt); });
}

ChatRender render_chat(const Json& request, const Tokenizer&) { return render_chat(request, RenderOptions{}); }

ChatRender render_tokenize_chat(const Json& request, const RenderOptions& opt) {
  return guarded([&] { return tokenize_impl(request, opt); });
}

ChatRender render_responses(const Json& request, const ResponsesPrev* prev, const RenderOptions& opt) {
  return guarded([&] { return responses_impl(request, prev, opt); });
}

ChatRender render_responses(Json&& request, const ResponsesPrev* prev, const RenderOptions& opt) {
  return guarded([&] { return responses_impl(std::move(request), prev, opt); });
}

std::string encode_messages_v41(const Json& messages, bool thinking, bool drop_thinking, const Json& effort) {
  try {
    if (!messages.is_array()) raise_py("TypeError", "messages must be a list");
    return render::encode_messages(messages.as_array(), thinking, drop_thinking, effort);
  } catch (const PyError& e) {
    throw RenderError(400, e.cls, Json(), e.str());
  }
}

}  // namespace fe
