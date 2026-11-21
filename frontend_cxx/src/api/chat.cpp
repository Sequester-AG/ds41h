// POST /v1/chat/completions (openai/chat_completion/serving.py, vLLM 0.32.0).
// Validation + rendering: fe_render (C). Output parsing: fe_parser (D). Engine: GenBackend (E).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "api/app.h"
#include "api/chat_common.h"
#include "api/pyd.h"
#include "api/util.h"

namespace fe::api {

namespace {

using L = std::vector<std::string>;

// ---------------------------------------------------------------------------------------
// The pydantic layer is fe_render's (exact error bodies). These local copies only coerce the
// body the way pydantic would (e.g. "max_tokens": "12" -> 12, tool_choice defaulting) for
// sampling_from_request(); they run after fe_render accepted the request.
// mode="before" validators, in pydantic's execution order (reverse of definition order).

void before_validators(Json& d) {
  // check_system_message_content_type: warning only.
  // drop_prompt_token_ids_with_media: kv_transfer_params unused here.
  // check_generation_prompt
  if (py_truthy(d.get("continue_final_message")) && py_truthy(d.get("add_generation_prompt")))
    throw vllm_validation_error("Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
                                std::nullopt);
  // check_tool_usage
  {
    const Json* tools = d.get("tools");
    if (tools && tools->is_array() && tools->size() == 0)
      throw vllm_validation_error(
          "`tools` must not be an empty array. Either provide at least one tool or omit the field entirely.",
          "tools");
    if (!d.get("tool_choice") && py_truthy(tools)) d.set("tool_choice", Json("auto"));
    const Json* tc = d.get("tool_choice");
    bool none = tc && tc->is_string() && tc->as_string() == "none";
    if (!none && tc && !tc->is_null()) {
      tools = d.get("tools");
      if (!tools || tools->is_null())
        throw vllm_validation_error("When using `tool_choice`, `tools` must be set.", "tool_choice");
      bool auto_req = tc->is_string() && (tc->as_string() == "auto" || tc->as_string() == "required");
      if (!auto_req && !tc->is_object())
        throw vllm_validation_error("Invalid value for `tool_choice`: " + py_fstr(*tc) +
                                        "! Only named tools, \"none\", \"auto\" or \"required\" are supported.",
                                    "tool_choice");
      const std::string usage = "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
      if (tc->is_object()) {
        const Json* fn = tc->get("function");
        if (!fn || !fn->is_object())
          throw vllm_validation_error("Invalid value for `function`: `" + (fn ? py_fstr(*fn) : std::string("None")) +
                                          "` in `tool_choice`! " + usage,
                                      "tool_choice.function");
        const Json* name = fn->get("name");
        if (!name)
          throw vllm_validation_error("Expected field `name` in `function` in `tool_choice`! " + usage,
                                      "tool_choice.function.name");
        if (!name->is_string() || name->as_string().empty())
          throw vllm_validation_error("Invalid `name` in `function`: `" + py_fstr(*name) + "` in `tool_choice`! " + usage,
                                      "tool_choice.function.name");
        bool valid = false;
        if (tools->is_array())
          for (auto& t : tools->as_array()) {
            const Json& tn = t["function"]["name"];
            if (tn.is_string() && tn.as_string() == name->as_string()) {
              valid = true;
              break;
            }
          }
        if (!valid)
          throw vllm_validation_error("The tool specified in `tool_choice` does not match any of the specified `tools`",
                                      "tool_choice");
      }
    }
  }
  // check_structured_outputs_count
  if (const Json* so = d.get("structured_outputs"); so && !so->is_null() && so->is_object()) {
    int count = 0;
    for (const char* k : {"json", "regex", "choice"})
      if (const Json* x = so->get(k); x && !x->is_null()) ++count;
    if (count > 1)
      throw vllm_validation_error(
          "You can only use one kind of constraints for structured outputs ('json', 'regex' or 'choice').",
          std::nullopt);
    const Json* tc = d.get("tool_choice");
    bool ok_tc = !tc || (tc->is_string() && (tc->as_string() == "none" || tc->as_string() == "auto" ||
                                             tc->as_string() == "required"));
    if (count > 0 && !ok_tc)
      throw vllm_validation_error("You can only either use constraints for structured outputs or tools, not both.",
                                  std::nullopt);
  }
  // check_logprobs
  {
    const Json* lti = d.get("logprob_token_ids");
    if (py_truthy(lti) && py_truthy(d.get("use_beam_search")))
      throw vllm_validation_error("`logprob_token_ids` is not supported with beam search.", "logprob_token_ids");
    if (py_truthy(lti) && !py_truthy(d.get("logprobs")))
      throw vllm_validation_error("when using `logprob_token_ids`, `logprobs` must be set to true.",
                                  "logprob_token_ids");
    for (const char* f : {"prompt_logprobs", "top_logprobs"}) {
      const Json* v = d.get(f);
      if (v && !v->is_null() && !v->is_number() && !v->is_bool())
        throw vllm_validation_error(std::string("`") + f + "` must be an integer.", f, py_fstr(*v));
    }
    if (const Json* pl = d.get("prompt_logprobs"); pl && !pl->is_null()) {
      double p = pl->is_bool() ? (pl->as_bool() ? 1 : 0) : pl->as_double();
      if (py_truthy(d.get("stream")) && (p > 0 || p == -1))
        throw vllm_validation_error("`prompt_logprobs` are not available when `stream=True`.", "prompt_logprobs");
      if (p < 0 && p != -1)
        throw vllm_validation_error("`prompt_logprobs` must be a positive value or -1.", "prompt_logprobs",
                                    py_fstr(*pl));
    }
    if (const Json* tl = d.get("top_logprobs"); tl && !tl->is_null()) {
      double t = tl->is_bool() ? (tl->as_bool() ? 1 : 0) : tl->as_double();
      if (t < 0 && t != -1)
        throw vllm_validation_error("`top_logprobs` must be a positive value or -1.", "top_logprobs", py_fstr(*tl));
      if ((t == -1 || t > 0) && !py_truthy(d.get("logprobs")))
        throw vllm_validation_error("when using `top_logprobs`, `logprobs` must be set to true.", "top_logprobs");
    }
  }
  // validate_stream_options
  if (py_truthy(d.get("stream_options")) && !py_truthy(d.get("stream")))
    throw vllm_validation_error("Stream options can only be defined when `stream=True`.", "stream_options");
  // validate_response_format
  if (const Json* rf = d.get("response_format"); rf && rf->is_object()) {
    const Json& t = (*rf)["type"];
    if (t.is_string() && t.as_string() == "json_schema" && (!rf->get("json_schema") || (*rf)["json_schema"].is_null()))
      throw vllm_validation_error(
          "When response_format type is 'json_schema', the 'json_schema' field must be provided.", "response_format");
  }
  // _normalize_messages_before: reasoning_content -> reasoning
  if (Json* msgs = d.get("messages"); msgs && msgs->is_array()) {
    for (auto& m : msgs->as_array()) {
      if (!m.is_object()) continue;
      Json* rc = m.get("reasoning_content");
      if (!rc) continue;
      Json val = *rc;
      m.erase("reasoning_content");
      if (!val.is_null()) {
        const Json* r = m.get("reasoning");
        if (!r || r->is_null()) m.set("reasoning", std::move(val));
      }
    }
  }
  validate_cache_salt(d.get("cache_salt"));
}

// ---------------------------------------------------------------------------------------
// Field validation (declaration order); coerced values are written back into the body.

void field_validation(Json& d) {
  Pyd v;
  auto B = [](const char* f) { return L{"body", f}; };
  // messages
  const Json* msgs = d.get("messages");
  if (!msgs)
    v.missing(B("messages"), d);
  else if (v.is_list(msgs, B("messages"), false))
    validate_messages(v, *msgs);
  v.to_str(d.get("model"), B("model"), true);
  coerce_float(d, v, "frequency_penalty");
  if (const Json* lb = d.get("logit_bias"); lb && !lb->is_null()) v.is_dict(lb, B("logit_bias"), true);
  coerce_bool(d, v, "logprobs", true);
  if (const Json* tl = d.get("top_logprobs"); tl && !tl->is_null()) coerce_int(d, v, "top_logprobs", false);
  coerce_int(d, v, "max_tokens", true);
  coerce_int(d, v, "max_completion_tokens", true);
  coerce_int(d, v, "n", true);
  coerce_float(d, v, "presence_penalty");
  if (const Json* rf = d.get("response_format"); rf && !rf->is_null()) v.is_dict(rf, B("response_format"), true);
  coerce_int(d, v, "seed", true, INT64_MIN, INT64_MAX);
  validate_stop(v, d.get("stop"));
  coerce_bool(d, v, "stream", true);
  if (Json* so = d.get("stream_options"); so && v.is_model(so, B("stream_options"), true)) {
    for (const char* f : {"include_usage", "continuous_usage_stats"}) {
      if (Json* x = so->get(f)) {
        auto b = v.to_bool(x, L{"body", "stream_options", f}, true);
        if (b) so->set(f, Json(*b));
      }
    }
  }
  coerce_float(d, v, "temperature");
  coerce_float(d, v, "top_p");
  if (const Json* tools = d.get("tools"); tools && v.is_list(tools, B("tools"), true)) validate_tools(v, *tools);
  // tool_choice union: "none" | "auto" | "required" | named | None (before-validator already
  // rejected other strings and non-dicts)
  if (Json* tc = d.get("tool_choice"); tc && tc->is_object()) {
    const Json* fn = tc->get("function");
    if (fn && fn->is_object()) v.to_str(fn->get("name"), L{"body", "tool_choice", "function", "name"}, false);
  }
  if (const Json* re = d.get("reasoning_effort"))
    v.to_literal(re, B("reasoning_effort"), true, {"none", "minimal", "low", "medium", "high", "xhigh", "max"});
  coerce_bool(d, v, "include_reasoning", false);
  coerce_bool(d, v, "parallel_tool_calls", true);
  v.to_str(d.get("user"), B("user"), true);
  coerce_bool(d, v, "use_beam_search", false);
  coerce_int(d, v, "top_k", true);
  coerce_float(d, v, "min_p");
  coerce_float(d, v, "repetition_penalty");
  coerce_bool(d, v, "watermarking", true);
  coerce_float(d, v, "length_penalty", false);
  validate_int_list(v, d, "stop_token_ids", true);
  coerce_bool(d, v, "include_stop_str_in_output", false);
  coerce_bool(d, v, "ignore_eos", false);
  coerce_int(d, v, "min_tokens", false);
  coerce_bool(d, v, "skip_special_tokens", false);
  coerce_bool(d, v, "spaces_between_special_tokens", false);
  coerce_int(d, v, "truncate_prompt_tokens", true, -1, INT64_MAX);
  if (const Json* ts = d.get("truncation_side")) v.to_literal(ts, B("truncation_side"), true, {"left", "right"});
  coerce_int(d, v, "prompt_logprobs", true);
  validate_int_list(v, d, "logprob_token_ids", true);
  validate_int_list(v, d, "allowed_token_ids", true);
  validate_str_list(v, d, "bad_words", false);
  coerce_bool(d, v, "echo", false);
  coerce_bool(d, v, "add_generation_prompt", false);
  coerce_bool(d, v, "continue_final_message", false);
  coerce_bool(d, v, "add_special_tokens", false);
  if (const Json* docs = d.get("documents"); docs && !docs->is_null()) v.is_list(docs, B("documents"), true);
  v.to_str(d.get("chat_template"), B("chat_template"), true);
  if (const Json* x = d.get("chat_template_kwargs")) v.is_dict(x, B("chat_template_kwargs"), true);
  if (const Json* x = d.get("media_io_kwargs")) v.is_dict(x, B("media_io_kwargs"), true);
  if (const Json* x = d.get("mm_processor_kwargs")) v.is_dict(x, B("mm_processor_kwargs"), true);
  if (const Json* x = d.get("structured_outputs")) v.is_dict(x, B("structured_outputs"), true);
  coerce_int(d, v, "priority", false, INT64_MIN, INT64_MAX);
  v.to_str(d.get("request_id"), B("request_id"), false);
  v.to_str(d.get("session_id"), B("session_id"), true);
  coerce_bool(d, v, "return_tokens_as_token_ids", true);
  coerce_bool(d, v, "return_token_ids", true);
  coerce_int(d, v, "routed_experts_prompt_start", false, 0, std::nullopt);
  coerce_bool(d, v, "return_token_offsets", true);
  coerce_bool(d, v, "return_prompt_text", true);
  if (const Json* x = d.get("mm_payload")) v.to_literal(x, B("mm_payload"), false, {"tensors", "none"});
  coerce_bool(d, v, "return_mm_kwargs", true);
  validate_cache_salt_field(v, d);
  if (const Json* x = d.get("kv_transfer_params")) v.is_dict(x, B("kv_transfer_params"), true);
  if (const Json* x = d.get("ec_transfer_params")) v.is_dict(x, B("ec_transfer_params"), true);
  if (const Json* x = d.get("vllm_xargs")) v.is_dict(x, B("vllm_xargs"), true);
  coerce_int(d, v, "stream_interval", true, 1, std::nullopt);
  if (!v.ok()) throw validation_error(v.errs);
}


static bool debug_stream() {
  static const bool on = [] {
    const char* v = getenv("FE_DEBUG_STREAM");
    return v && *v && *v != '0';
  }();
  return on;
}

struct ChatOpts {
  std::string id, model, role;
  int64_t created = 0;
  bool include_usage = false, cont_usage = false;
  bool return_token_ids = false, return_prompt_text = false, include_reasoning = true;
  bool parallel_false = false;
  std::optional<std::string> fingerprint;
  bool reasoning_details = true;  // _include_reasoning_tokens_details (reasoning parser configured)
  bool prompt_details = false;
  std::string echo_content;       // last message content when echo applies
};

void write_usage_stream(JW& w, int64_t prompt, int64_t completion, std::optional<int64_t> reasoning) {
  w.k("usage").ob().ki("prompt_tokens", prompt).ki("total_tokens", prompt + completion).ki("completion_tokens", completion);
  if (reasoning) w.k("completion_tokens_details").ob().ki("reasoning_tokens", *reasoning).oe();
  w.oe();
}

class ChatStream {
 public:
  std::shared_ptr<http::Response> resp;
  ChatOpts o;
  std::unique_ptr<fe::Parser> parser;
  std::vector<int32_t> prompt_ids;
  std::string prompt_text;
  int64_t mm_tokens = 0;
  bool first = true, finish_sent = false;
  int64_t prev_tokens = 0, prev_reasoning = 0;
  bool any_generated = false;
  int64_t cached = 0, creation = 0;
  Json last_spec;

  void head(JW& w) {
    w.ob().kstr("id", o.id).kstr("object", "chat.completion.chunk").ki("created", o.created).kstr("model", o.model);
  }
  void emit(const std::string& json) { resp->write("data: " + json + "\n\n"); }
  void emit_error(const ApiError& e) {
    emit(error_body_stream(e));
    resp->write("data: [DONE]\n\n");
    resp->end();
  }

  void first_chunk() {
    JW w(true, false, FloatStyle::Ryu);
    head(w);
    w.k("choices").ab().ob().ki("index", 0);
    w.k("delta").ob().kstr("role", o.role).kstr("content", "").oe();
    w.knull("logprobs").knull("finish_reason").oe().ae();
    std::optional<int64_t> rz = o.reasoning_details ? std::optional<int64_t>(0) : std::nullopt;
    if (o.cont_usage) write_usage_stream(w, int64_t(prompt_ids.size()), 0, rz);
    if (o.return_token_ids)
      w.k("prompt_token_ids").int_array(prompt_ids);
    else
      w.knull("prompt_token_ids");
    if (o.return_prompt_text)
      w.kstr("prompt_text", prompt_text);
    else
      w.knull("prompt_text");
    w.oe();
    emit(w.take());
    if (!o.echo_content.empty()) {
      JW e(true, false, FloatStyle::Ryu);
      head(e);
      e.k("choices").ab().ob().ki("index", 0).k("delta").ob().kstr("content", o.echo_content).oe();
      e.knull("logprobs").knull("finish_reason").oe().ae();
      if (o.cont_usage) write_usage_stream(e, int64_t(prompt_ids.size()), 0, rz);
      e.oe();
      emit(e.take());
    }
  }

  // returns false when the stream is complete
  bool on_output(GenOutput&& out) {
    if (debug_stream()) {  // FE_DEBUG_STREAM=1: sizes only (never text)
      static const auto t00 = std::chrono::steady_clock::now();
      fprintf(stderr, "[frontend_cxx] stream %s +%.1f ms: text %zu B, %zu ids, finished %d\n", o.id.c_str(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t00).count(),
              out.text.size(), out.token_ids.size(), int(out.finished));
    }
    if (resp->disconnected()) return false;
    if (out.engine_dead) {
      emit_error(ApiError(500, "InternalServerError",
                          "EngineCore encountered an issue. See stack trace (above) for the root cause."));
      return false;
    }
    if (first) {
      cached = out.num_cached_tokens;
      creation = out.num_cache_creation_tokens;
      first_chunk();
      first = false;
    }
    if (!out.spec_decode.is_null()) last_spec = out.spec_decode;
    if (!finish_sent) {
      if (out.finish_reason && *out.finish_reason == "error") {
        emit_error(ApiError(500, "InternalServerError", "Internal server error"));
        return false;
      }
      bool skip = out.text.empty() && out.token_ids.empty() && prev_tokens == 0;
      if (!skip) process_delta(out);
    }
    if (out.finished) {
      finish();
      return false;
    }
    return true;
  }

  void process_delta(GenOutput& out) {
    bool finished = out.finish_reason.has_value();
    std::string delta_json;
    bool present = true;
    if (parser) {
      fe::ParserDelta d = parser->feed(out.text, out.token_ids, finished);
      present = d.present;
      if (present) delta_json = d.to_json(!o.parallel_false);
      if (debug_stream() && !present) fprintf(stderr, "[frontend_cxx] stream %s: parser delta absent\n", o.id.c_str());
    } else {
      JW w(true, false, FloatStyle::Ryu);
      w.ob().kstr("content", out.text).oe();
      delta_json = w.take();
    }
    prev_tokens += int64_t(out.token_ids.size());
    if (!out.token_ids.empty()) any_generated = true;
    if (parser && o.reasoning_details && o.cont_usage) prev_reasoning = parser->reasoning_tokens();
    bool hide = !o.include_reasoning && parser;
    if (!present) {
      if (!finished && (!o.return_token_ids || hide)) return;
      delta_json = "{}";
    }
    bool include_ids = o.return_token_ids && !hide;
    JW w(true, false, FloatStyle::Ryu);
    head(w);
    w.k("choices").ab().ob().ki("index", 0);
    w.kraw("delta", delta_json);
    w.knull("logprobs");
    if (!finished) {
      w.knull("finish_reason");
    } else {
      std::string fr = parser ? parser->stream_finish_reason(*out.finish_reason)
                              : (out.finish_reason->empty() ? std::string("stop") : *out.finish_reason);
      w.kstr("finish_reason", fr);
      w.kjson("stop_reason", out.stop_reason);
      finish_sent = true;
    }
    if (include_ids)
      w.k("token_ids").int_array(out.token_ids);
    else
      w.knull("token_ids");
    w.oe().ae();
    if (o.cont_usage)
      write_usage_stream(w, int64_t(prompt_ids.size()), prev_tokens,
                         o.reasoning_details ? std::optional<int64_t>(prev_reasoning) : std::nullopt);
    if (!o.include_usage && o.fingerprint && finished) w.kstr("system_fingerprint", *o.fingerprint);
    w.oe();
    emit(w.take());
  }

  void finish() {
    if (o.reasoning_details && !o.cont_usage && parser && any_generated) prev_reasoning = parser->reasoning_tokens();
    if (o.include_usage) {
      JW w(true, false, FloatStyle::Ryu);
      head(w);
      w.k("choices").ab().ae();
      int64_t p = int64_t(prompt_ids.size());
      w.k("usage").ob().ki("prompt_tokens", p).ki("total_tokens", p + prev_tokens).ki("completion_tokens", prev_tokens);
      write_prompt_details_stream(w, o.prompt_details, cached, creation, mm_tokens);
      if (o.reasoning_details) w.k("completion_tokens_details").ob().ki("reasoning_tokens", prev_reasoning).oe();
      w.oe();
      if (o.fingerprint) w.kstr("system_fingerprint", *o.fingerprint);
      if (spec_decode_present(last_spec)) {
        w.k("metrics").ob().k("speculative_decoding");
        write_spec_decode(w, last_spec, false);
        w.oe();
      }
      w.oe();
      emit(w.take());
    }
    resp->write("data: [DONE]\n\n");
    resp->end();
  }
};

struct FullState {
  ChatOpts o;
  std::shared_ptr<const fe::ParserVocab> vocab;
  fe::ParserOptions popts;
  bool use_parser = false;
  std::vector<int32_t> prompt_ids;
  std::string prompt_text;
  int64_t mm_tokens = 0;
};

std::string build_chat_full(FullState& fs, GenOutput&& out) {
  const ChatOpts& o = fs.o;
  if (out.engine_dead)
    throw ApiError(500, "InternalServerError", "EngineCore encountered an issue. See stack trace (above) for the root cause.");
  if (out.finish_reason && *out.finish_reason == "error") throw generation_error();
  std::string engine_reason = out.finish_reason ? *out.finish_reason : "stop";
  std::optional<std::string> reasoning, content;
  std::vector<fe::ToolCall> tools;
  std::string finish = engine_reason.empty() ? "stop" : engine_reason;
  int64_t reasoning_tokens = 0;
  bool suppress = false;
  if (fs.use_parser) {
    fe::Parser p(fs.vocab, fs.popts);
    fe::Parser::Full f = p.parse_full(out.text, out.token_ids, engine_reason);
    reasoning = f.reasoning;
    content = f.content;
    tools = std::move(f.tool_calls);
    finish = f.finish_reason;
    reasoning_tokens = f.reasoning_tokens;
    suppress = !o.include_reasoning;
  } else {
    content = out.text;
  }
  if (!o.echo_content.empty()) content = o.echo_content + content.value_or("");
  int64_t p = int64_t(fs.prompt_ids.size());
  int64_t n = int64_t(out.token_ids.size());
  JW w(true, false, FloatStyle::PyRepr);
  w.ob().kstr("id", o.id).kstr("object", "chat.completion").ki("created", o.created).kstr("model", o.model);
  w.k("choices").ab().ob().ki("index", 0);
  w.k("message").ob().kstr("role", o.role);
  if (content)
    w.kstr("content", *content);
  else
    w.knull("content");
  w.knull("refusal").knull("annotations").knull("audio").knull("function_call");
  if (!tools.empty()) {
    w.k("tool_calls").ab();
    for (auto& t : tools) {
      w.ob().kstr("id", t.id.empty() ? fe::random_tool_call_id() : t.id).kstr("type", "function");
      w.k("function").ob().kstr("name", t.name).kstr("arguments", t.arguments).oe().oe();
    }
    w.ae();
  }
  if (reasoning)
    w.kstr("reasoning", *reasoning);
  else
    w.knull("reasoning");
  w.oe();
  w.knull("logprobs").kstr("finish_reason", finish).kjson("stop_reason", out.stop_reason);
  if (o.return_token_ids && !suppress)
    w.k("token_ids").int_array(out.token_ids);
  else
    w.knull("token_ids");
  w.knull("routed_experts");
  w.oe().ae();
  w.knull("service_tier");
  if (o.fingerprint)
    w.kstr("system_fingerprint", *o.fingerprint);
  else
    w.knull("system_fingerprint");
  w.k("usage").ob().ki("prompt_tokens", p).ki("total_tokens", p + n).ki("completion_tokens", n);
  write_prompt_details_full(w, o.prompt_details, out.num_cached_tokens, out.num_cache_creation_tokens, fs.mm_tokens);
  if (o.reasoning_details)
    w.k("completion_tokens_details").ob().ki("reasoning_tokens", reasoning_tokens).oe();
  else
    w.knull("completion_tokens_details");
  w.oe();
  w.knull("prompt_logprobs");
  if (o.return_token_ids)
    w.k("prompt_token_ids").int_array(fs.prompt_ids);
  else
    w.knull("prompt_token_ids");
  if (o.return_prompt_text)
    w.kstr("prompt_text", fs.prompt_text);
  else
    w.knull("prompt_text");
  w.knull("kv_transfer_params").knull("ec_transfer_params");
  write_metrics_full(w, out.spec_decode);
  w.oe();
  return w.take();
}

}  // namespace

// ---------------------------------------------------------------------------------------

void App::route_chat(Ctx& c) {
  Json body = parse_json_body(c);
  fe::ChatRender r;
  try {
    r = fe::render_chat(body, comp_.render);
  } catch (const fe::RenderError& e) {
    throw from_render_error(e);
  }
  check_model(body);
  preflight();
  if (auto n = body.get_int("n"); n && *n > 1)
    throw bad_request("frontend_cxx: n > 1 is not supported (DEVIATIONS.md)", std::string("n"));
  // pydantic coercion of the body for SamplingParams (fe_render already validated it)
  Json eb = body;
  try {
    before_validators(eb);
    field_validation(eb);
  } catch (const ApiError&) {
  }
  {
    const Json* mct = eb.get("max_completion_tokens");
    const Json* mt = eb.get("max_tokens");
    if (mct && mct->is_int())
      check_output_tokens_param(*this, mct->as_int(), "max_completion_tokens");
    else if (mt && mt->is_int())
      check_output_tokens_param(*this, mt->as_int(), "max_tokens");
  }
  Prepared pp = prepare_prompt(*this, r);

  std::string request_id = "chatcmpl-" + base_request_id(c, body.get("request_id"));
  ChatOpts o;
  o.id = request_id;
  o.model = cfg_.model_name();
  o.created = int64_t(now_unix());
  const Json& msgs = eb["messages"];
  if (eb.get_bool("add_generation_prompt").value_or(true)) {
    o.role = cfg_.response_role;
  } else if (msgs.is_array() && msgs.size() > 0) {
    const Json& rl = msgs[msgs.size() - 1]["role"];
    o.role = rl.is_string() ? rl.as_string() : "";
  }
  bool stream = eb.get_bool("stream").value_or(false);
  bool force = cfg_.enable_force_include_usage;
  const Json& so = eb["stream_options"];
  bool inc = so.is_object() && so.get_bool("include_usage").value_or(false);
  o.include_usage = force || inc;
  o.cont_usage = force || (inc && so.get_bool("continuous_usage_stats").value_or(false));
  o.return_token_ids = eb.get_bool("return_token_ids").value_or(false);
  o.return_prompt_text = eb.get_bool("return_prompt_text").value_or(false);
  o.include_reasoning = eb.get_bool("include_reasoning").value_or(true);
  const Json* ptc = eb.get("parallel_tool_calls");
  o.parallel_false = ptc && ptc->is_bool() && !ptc->as_bool();
  o.fingerprint = cfg_.system_fingerprint;
  o.reasoning_details = !cfg_.reasoning_parser.empty();
  o.prompt_details = cfg_.enable_prompt_tokens_details;
  if (eb.get_bool("echo").value_or(false) && msgs.is_array() && msgs.size() > 0) {
    const Json& last = msgs[msgs.size() - 1];
    if (last["role"].is_string() && last["role"].as_string() == o.role && last["content"].is_string())
      o.echo_content = last["content"].as_string();
  }

  bool use_parser = !cfg_.reasoning_parser.empty() || !cfg_.tool_parser.empty();
  fe::ParserOptions popts = fe::parser_options_for(eb, r.chat_template_kwargs);
  popts.on_tool_parser_invocation = tool_parser_hook(ToolParserRequestType::ChatCompletions, popts.tool_choice);
  // structural tag for required / named / strict tools (DelegatingParser._apply_structural_tag)
  std::optional<std::string> stag;
  if (use_parser) {
    std::optional<std::string> tag;
    try {
      tag = fe::structural_tag_for(eb);
    } catch (const std::invalid_argument& e) {
      throw bad_request(e.what());
    } catch (const std::runtime_error& e) {
      throw ApiError(400, "BadRequestError", e.what());
    }
    if (tag) {  // request.structured_outputs = StructuredOutputsParams(structural_tag=...); response_format = None
      eb.erase("response_format");
      eb.erase("structured_outputs");
    }
    stag = tag;
  }
  apply_parser_adjust_request(*this, eb);
  GenRequest gr = make_gen_request(*this, c, eb, pp, "chat", request_id);
  gr.structural_tag = stag;
  set_reasoning_fields(*this, gr, eb, pp);

  if (stream) backend_->precheck_caps("chat", gr.body, int64_t(pp.prompt_ids.size()));   // caps 400 first
  if (stream) {
    auto st = std::make_shared<ChatStream>();
    st->resp = c.resp;
    st->o = o;
    if (use_parser) {
      st->parser = std::make_unique<fe::Parser>(comp_.vocab, popts);
      st->parser->set_prompt_token_ids(pp.prompt_ids);
    }
    st->prompt_ids = pp.prompt_ids;
    st->prompt_text = pp.prompt;
    st->mm_tokens = pp.mm_tokens;
    c.resp->start(200, sse_headers());
    auto pump = std::make_shared<Pump>(c.resp, [st](GenOutput&& out) { return st->on_output(std::move(out)); });
    std::weak_ptr<Pump> wp = pump;
    c.resp->on_writable([wp] {
      if (auto p = wp.lock()) p->kick();
    });
    std::string internal;
    try {
      internal = backend_->start(std::move(gr), [pump](GenOutput&& out) { pump->push(std::move(out)); });
    } catch (const ApiError& e) {
      st->emit_error(e);  // errors inside the stream generator become an error chunk
      return;
    }
    auto be = backend_;
    c.resp->on_disconnect([be, internal] { be->abort(internal); });
    return;
  }

  auto fs = std::make_shared<FullState>();
  fs->o = o;
  fs->vocab = comp_.vocab;
  fs->popts = popts;
  fs->use_parser = use_parser;
  fs->prompt_ids = pp.prompt_ids;
  fs->prompt_text = pp.prompt;
  fs->mm_tokens = pp.mm_tokens;
  auto resp = c.resp;
  App* self = this;
  auto acc = std::make_shared<FullAcc>();
  acc->done = [self, resp, fs](GenOutput&& out) {
    Ctx c2{nullptr, resp, ""};
    try {
      if (resp->disconnected()) return;
      self->send_json(c2, 200, build_chat_full(*fs, std::move(out)));
    } catch (const ApiError& e) {
      if (!resp->started()) self->send_error(c2, e);
    } catch (const std::exception& e) {
      if (!resp->started()) self->send_error(c2, ApiError(500, "InternalServerError", e.what()));
    }
  };
  std::string internal = backend_->start(std::move(gr), [acc](GenOutput&& out) { acc->on(std::move(out)); });
  auto be = backend_;
  std::weak_ptr<FullAcc> wacc = acc;
  c.resp->on_disconnect([be, internal, wacc] {
    be->abort(internal);
    if (auto a = wacc.lock()) a->cancel();
  });
}

}  // namespace fe::api
