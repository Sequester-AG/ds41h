// /v1/responses (openai/responses/{protocol,serving,streaming_events,utils}.py, SimpleContext path),
// GET /v1/responses/{id}, POST /v1/responses/{id}/cancel, and the BoundedStore semantics.
#include <cmath>
#include <algorithm>
#include <thread>

#include "api/app.h"
#include "api/chat_common.h"
#include "api/pyd.h"
#include "api/util.h"

namespace fe::api {

namespace {
using L = std::vector<std::string>;

// ---------------------------------------------------------------------------------------
// request validation

void before_validators(Json& d) {
  // check_tool_usage
  {
    const Json* tools = d.get("tools");
    Json tc = d.get("tool_choice") ? *d.get("tool_choice") : Json("auto");
    bool has_tools = tools && !tools->is_null() && tools->size() > 0;
    bool named = tc.is_object() && tc["type"].is_string() &&
                 (tc["type"].as_string() == "function" || tc["type"].as_string() == "custom");
    if (!has_tools) {
      if (tc.is_string() && (tc.as_string() == "auto" || tc.as_string() == "none"))
        d.set("tool_choice", Json("none"));
      else if (tc.is_string() && tc.as_string() == "required")
        throw vllm_validation_error("Tool choice 'required' must be specified with 'tools' parameter.", "tool_choice");
      else if (named)
        throw vllm_validation_error("Tool choice 'function' not found in 'tools' parameter.", "tool_choice");
    } else if (named) {
      const Json& n = tc["name"];
      if (!n.is_string() || n.as_string().empty())
        throw vllm_validation_error("Tool choice 'function' not found in 'tools' parameter.", "tool_choice");
    }
  }
  // check_custom_tool_format
  if (const Json* tools = d.get("tools"); tools && tools->is_array())
    for (auto& t : tools->as_array())
      if (t.is_object() && t["type"].is_string() && t["type"].as_string() == "custom" && t["format"].is_object() &&
          t["format"]["type"].is_string() && t["format"]["type"].as_string() == "grammar")
        throw vllm_validation_error(
            "Custom tools with a grammar format are not supported; only free-form text custom tools are accepted.",
            "tools");
  // input_item_parsing: normalisation of reasoning / assistant message items
  if (Json* in = d.get("input"); in && in->is_array()) {
    for (auto& item : in->as_array()) {
      if (!item.is_object()) continue;
      const Json& t = item["type"];
      std::string type = t.is_string() ? t.as_string() : "";
      if (type == "reasoning") {
        if (!item.get("id")) item.set("id", Json("rs_" + random_uuid_hex()));
      } else if (type == "message" && item["role"].is_string() && item["role"].as_string() == "assistant") {
        if (!item["content"].is_array()) continue;
        if (!item.get("id")) item.set("id", Json("msg_" + random_uuid_hex()));
        if (!item.get("status")) item.set("status", Json("completed"));
        for (auto& c : item.get("content")->as_array())
          if (c.is_object() && c["type"].is_string() && c["type"].as_string() == "output_text" && !c.get("annotations"))
            c.set("annotations", Json::array());
      }
    }
  }
  // validate_prompt
  if (const Json* p = d.get("prompt"); p && !p->is_null())
    throw vllm_validation_error("prompt template is not supported", "prompt");
  // validate_background
  if (py_truthy(d.get("background"))) {
    const Json* st = d.get("store");
    if (st && !py_truthy(st)) throw vllm_validation_error("background can only be used when `store` is true", "background");
  }
  validate_cache_salt(d.get("cache_salt"));
}

void field_validation(Json& d) {
  Pyd v;
  auto B = [](const char* f) { return L{"body", f}; };
  coerce_bool(d, v, "background", true);
  if (const Json* in = d.get("input"); !in) {
    v.missing(B("input"), d);
  } else if (!in->is_string() && !in->is_array()) {
    v.add("string_type", B("input"), "Input should be a valid string", in);
    v.add("list_type", B("input"), "Input should be a valid list", in);
  }
  v.to_str(d.get("instructions"), B("instructions"), true);
  coerce_int(d, v, "max_output_tokens", true);
  coerce_int(d, v, "max_tool_calls", true);
  if (const Json* x = d.get("metadata")) v.is_dict(x, B("metadata"), true);
  v.to_str(d.get("model"), B("model"), true);
  coerce_bool(d, v, "parallel_tool_calls", true);
  v.to_str(d.get("previous_response_id"), B("previous_response_id"), true);
  if (Json* r = d.get("reasoning"); r && v.is_model(r, B("reasoning"), true)) {
    if (const Json* e = r->get("effort"))
      v.to_literal(e, L{"body", "reasoning", "effort"}, true, {"none", "minimal", "low", "medium", "high", "xhigh", "max"});
  }
  coerce_bool(d, v, "include_reasoning", false);
  if (const Json* x = d.get("service_tier"))
    v.to_literal(x, B("service_tier"), false, {"auto", "default", "flex", "scale", "priority"});
  coerce_bool(d, v, "store", true);
  coerce_bool(d, v, "stream", true);
  coerce_float(d, v, "temperature");
  if (const Json* x = d.get("text")) v.is_model(x, B("text"), true);
  if (const Json* x = d.get("tools")) v.is_list(x, B("tools"), false);
  if (const Json* tl = d.get("top_logprobs"); tl && !tl->is_null()) coerce_int(d, v, "top_logprobs", false);
  coerce_float(d, v, "top_p");
  coerce_int(d, v, "top_k", true);
  coerce_float(d, v, "min_p");
  if (const Json* x = d.get("truncation")) v.to_literal(x, B("truncation"), true, {"auto", "disabled"});
  v.to_str(d.get("user"), B("user"), true);
  coerce_bool(d, v, "skip_special_tokens", false);
  coerce_bool(d, v, "include_stop_str_in_output", false);
  coerce_float(d, v, "presence_penalty");
  coerce_float(d, v, "frequency_penalty");
  v.to_str(d.get("request_id"), B("request_id"), false);
  coerce_int(d, v, "priority", false, INT64_MIN, INT64_MAX);
  validate_cache_salt_field(v, d);
  coerce_bool(d, v, "enable_response_messages", false);
  coerce_float(d, v, "repetition_penalty");
  coerce_int(d, v, "seed", true);
  validate_stop(v, d.get("stop"));
  validate_int_list(v, d, "stop_token_ids", true);
  coerce_bool(d, v, "ignore_eos", false);
  if (const Json* x = d.get("chat_template_kwargs")) v.is_dict(x, B("chat_template_kwargs"), true);
  if (!v.ok()) throw validation_error(v.errs);
}

// ---------------------------------------------------------------------------------------
// response serialisation (model_dump(mode="json", by_alias=True))

struct RespCtx {
  std::string id, model;
  int64_t created_at = 0;
  Json body;               // validated request
  double temperature = 1.0, top_p = 1.0, presence_penalty = 0.0, frequency_penalty = 0.0;
  int64_t max_output_tokens = 0;
  std::optional<int64_t> top_logprobs;
  bool include_reasoning = true;
  bool parallel_false = false;
  bool store = false, background = false;
  std::vector<int32_t> prompt_ids;
  bool reasoning_details = true;
};

void write_tool(JW& w, const Json& t) {
  if (t["type"].is_string() && t["type"].as_string() == "function") {
    w.ob();
    w.kjson("name", t["name"]).kjson("parameters", t["parameters"]).kjson("strict", t["strict"]);
    w.kstr("type", "function").kjson("allowed_callers", t["allowed_callers"]).kjson("async", t["async"]);
    w.kjson("defer_loading", t["defer_loading"]).kjson("description", t["description"]);
    w.kjson("output_schema", t["output_schema"]);
    w.oe();
  } else {
    w.json(t);
  }
}

void write_tool_choice(JW& w, const Json& tc) {
  if (tc.is_object() && tc["type"].is_string() && tc["type"].as_string() == "function") {
    w.ob().kjson("name", tc["name"]).kstr("type", "function").oe();
  } else {
    w.json(tc);
  }
}

void write_reasoning(JW& w, const Json& r) {
  if (!r.is_object()) {
    w.null();
    return;
  }
  w.ob().kjson("context", r["context"]).kjson("effort", r["effort"]).kjson("generate_summary", r["generate_summary"]);
  w.kjson("mode", r["mode"]).kjson("summary", r["summary"]).oe();
}

void write_text_cfg(JW& w, const Json& t) {
  if (!t.is_object()) {
    w.null();
    return;
  }
  w.ob().k("format");
  const Json& f = t["format"];
  if (!f.is_object()) {
    w.null();
  } else {
    std::string type = f["type"].is_string() ? f["type"].as_string() : "";
    if (type == "json_schema") {
      w.ob().kjson("name", f["name"]).kjson("schema", f["schema"]).kstr("type", "json_schema");
      w.kjson("description", f["description"]).kjson("strict", f["strict"]).oe();
    } else {
      w.ob().kstr("type", type).oe();
    }
  }
  w.kjson("verbosity", t["verbosity"]).oe();
}

struct Usage {
  int64_t in = 0, out = 0, cached = 0, cache_write = 0, reasoning = 0;
};

void write_usage(JW& w, const Usage& u) {
  w.ob().ki("input_tokens", u.in);
  w.k("input_tokens_details").ob().ki("cached_tokens", u.cached).ki("cache_write_tokens", u.cache_write);
  w.k("input_tokens_per_turn").ab().ae().k("cached_tokens_per_turn").ab().ae().oe();
  w.ki("output_tokens", u.out);
  w.k("output_tokens_details").ob().ki("reasoning_tokens", u.reasoning).ki("tool_output_tokens", 0);
  w.k("output_tokens_per_turn").ab().ae().k("tool_output_tokens_per_turn").ab().ae().oe();
  w.ki("total_tokens", u.in + u.out).oe();
}

// The full response object (dialect: Py for JSON bodies, Ryu for SSE events).
std::string response_json(const RespCtx& r, const std::string& status, const Json& output,
                          const std::optional<Usage>& usage, FloatStyle fs) {
  const Json& b = r.body;
  JW w(true, false, fs);
  w.ob().kstr("id", r.id).ki("created_at", r.created_at);
  if (status == "incomplete")
    w.k("incomplete_details").ob().kstr("reason", "max_output_tokens").oe();
  else
    w.knull("incomplete_details");
  w.kjson("instructions", b["instructions"]).kjson("metadata", b["metadata"]).kstr("model", r.model);
  w.kstr("object", "response").kjson("output", output);
  const Json& ptc = b["parallel_tool_calls"];
  w.kb("parallel_tool_calls", ptc.is_bool() ? ptc.as_bool() : true);
  w.k("temperature").f(r.temperature);
  w.k("tool_choice");
  write_tool_choice(w, b.get("tool_choice") ? b["tool_choice"] : Json("auto"));
  w.k("tools").ab();
  if (b["tools"].is_array())
    for (auto& t : b["tools"].as_array()) write_tool(w, t);
  w.ae();
  w.k("top_p").f(r.top_p);
  const Json& bg = b["background"];
  w.kb("background", bg.is_bool() && bg.as_bool());
  w.ki("max_output_tokens", r.max_output_tokens);
  w.kjson("max_tool_calls", b["max_tool_calls"]).kjson("previous_response_id", b["previous_response_id"]);
  w.knull("prompt");
  w.k("reasoning");
  write_reasoning(w, b["reasoning"]);
  w.kstr("service_tier", b["service_tier"].is_string() ? b["service_tier"].as_string() : "auto");
  w.kstr("status", status);
  w.k("text");
  write_text_cfg(w, b["text"]);
  if (r.top_logprobs)
    w.ki("top_logprobs", *r.top_logprobs);
  else
    w.knull("top_logprobs");
  w.kstr("truncation", b["truncation"].is_string() ? b["truncation"].as_string() : "disabled");
  if (usage) {
    w.k("usage");
    write_usage(w, *usage);
  } else {
    w.knull("usage");
  }
  w.kjson("user", b["user"]);
  w.k("presence_penalty").f(r.presence_penalty);
  w.k("frequency_penalty").f(r.frequency_penalty);
  w.knull("kv_transfer_params").knull("ec_transfer_params").knull("input_messages").knull("output_messages");
  w.oe();
  return w.take();
}

// build_response_output_items (non-streaming)
Json output_items(const RespCtx& r, const fe::Parser::Full& pf) {
  Json out = Json::array();
  if (pf.reasoning && !pf.reasoning->empty()) {
    Json content = Json::array({Json::object({{"text", Json(*pf.reasoning)}, {"type", Json("reasoning_text")}})});
    out.push_back(Json::object({{"id", Json("rs_" + random_uuid_hex())},
                                {"summary", Json::array()},
                                {"type", Json("reasoning")},
                                {"content", content},
                                {"encrypted_content", Json()},
                                {"status", Json()}}));
  }
  if (pf.content && !pf.content->empty()) {
    Json part = Json::object({{"annotations", Json::array()},
                              {"text", Json(*pf.content)},
                              {"type", Json("output_text")},
                              {"logprobs", Json()}});
    out.push_back(Json::object({{"id", Json("msg_" + random_uuid_hex())},
                                {"content", Json::array({part})},
                                {"role", Json("assistant")},
                                {"status", Json("completed")},
                                {"type", Json("message")},
                                {"phase", Json()}}));
  }
  size_t n = pf.tool_calls.size();
  if (r.parallel_false && n > 1) n = 1;
  for (size_t i = 0; i < n; ++i) {
    auto& tc = pf.tool_calls[i];
    std::string call_id = tc.id.empty() ? fe::random_tool_call_id() : tc.id;
    out.push_back(Json::object({{"arguments", Json(tc.arguments)},
                                {"call_id", Json(call_id)},
                                {"name", Json(tc.name)},
                                {"type", Json("function_call")},
                                {"id", Json("fc_" + random_uuid_hex())},
                                {"async", Json()},
                                {"caller", Json()},
                                {"namespace", Json()},
                                {"status", Json("completed")}}));
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// streaming (SimpleStreamingEventProcessor)

class RespStream {
 public:
  App* app;
  std::shared_ptr<http::Response> resp;  // may be null for background-only
  std::shared_ptr<EventLog> log;          // background stream events
  RespCtx r;
  std::shared_ptr<fe::Parser> parser;
  int64_t seq = 0;
  // state
  enum St { NONE, CONTENT, REASONING, TOOL } st = NONE;
  int64_t output_index = 0;
  std::string item_id, acc, tool_call_id, tool_name;
  std::optional<int> tool_index;
  Json items = Json::array();
  // accounting
  std::vector<int32_t> all_ids;
  std::string all_text;
  int64_t n_out = 0;
  int64_t cached = 0, creation = 0;
  bool first_out = true;
  std::string finish;
  std::function<void(const StoredResponse&)> on_final;
  // background without stream: Python builds the result with responses_full_generator (one non-streaming
  // parser.parse); called with the full text / ids / finish reason to record that parse's tool-parser invocation.
  std::function<void(const std::string&, const std::vector<int32_t>&, const std::string&)> full_parse_metric;

  void ev(const char* type, const std::function<void(JW&)>& fields_before_seq,
          const std::function<void(JW&)>& fields_after_seq = nullptr,
          const std::function<void(JW&)>& extras = nullptr) {
    JW w(true, false, FloatStyle::Ryu);
    w.ob();
    fields_before_seq(w);
    w.ki("sequence_number", seq);
    if (fields_after_seq) fields_after_seq(w);
    w.kstr("type", type);
    if (extras) extras(w);
    w.oe();
    std::string data = std::string("event: ") + type + "\ndata: " + w.take() + "\n\n";
    if (log) {
      std::lock_guard<std::mutex> lk(log->mu);
      log->events.emplace_back(seq, data);
      log->cv.notify_all();
    }
    if (resp) resp->write(data);
    ++seq;
  }

  void start() {
    std::string initial = response_json(r, "in_progress", Json::array(), std::nullopt, FloatStyle::Ryu);
    ev("response.created", [&](JW& w) { w.kraw("response", initial); });
    ev("response.in_progress", [&](JW& w) { w.kraw("response", initial); });
  }

  void close_current() {
    if (st == CONTENT) {
      std::string text = acc;
      ev("response.output_text.done",
         [&](JW& w) { w.ki("content_index", 0).kstr("item_id", item_id).k("logprobs").ab().ae().ki("output_index", output_index); },
         [&](JW& w) { w.kstr("text", text); });
      ev("response.content_part.done",
         [&](JW& w) {
           w.ki("content_index", 0).kstr("item_id", item_id).ki("output_index", output_index);
           w.k("part").ob().k("annotations").ab().ae().kstr("text", text).kstr("type", "output_text").knull("logprobs").oe();
         });
      Json part = Json::object({{"annotations", Json::array()}, {"text", Json(text)}, {"type", Json("output_text")},
                                {"logprobs", Json()}});
      Json item = Json::object({{"id", Json(item_id)},
                                {"content", text.empty() ? Json::array() : Json::array({part})},
                                {"role", Json("assistant")},
                                {"status", Json("completed")},
                                {"type", Json("message")},
                                {"phase", Json()}});
      ev("response.output_item.done", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
      items.push_back(item);
      ++output_index;
    } else if (st == REASONING) {
      std::string text = acc;
      ev("response.reasoning_text.done",
         [&](JW& w) { w.ki("content_index", 0).kstr("item_id", item_id).ki("output_index", output_index); },
         [&](JW& w) { w.kstr("text", text); });
      ev("response.content_part.done",
         [&](JW& w) {
           w.ki("content_index", 0).kstr("item_id", item_id).ki("output_index", output_index);
           w.k("part").ob().kstr("text", text).kstr("type", "reasoning_text").oe();
         });
      Json item = Json::object({{"id", Json(item_id)},
                                {"summary", Json::array()},
                                {"type", Json("reasoning")},
                                {"content", Json::array({Json::object({{"text", Json(text)}, {"type", Json("reasoning_text")}})})},
                                {"encrypted_content", Json()},
                                {"status", Json("completed")}});
      ev("response.output_item.done", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
      items.push_back(item);
      ++output_index;
    } else if (st == TOOL) {
      std::string args = acc;
      // `name` is not an openai field of this event: pydantic keeps it as an extra, after the declared fields
      ev("response.function_call_arguments.done",
         [&](JW& w) { w.kstr("arguments", args).kstr("item_id", item_id).ki("output_index", output_index); },
         nullptr, [&](JW& w) { w.kstr("name", tool_name); });
      Json item = Json::object({{"arguments", Json(args)},
                                {"call_id", Json(tool_call_id)},
                                {"name", Json(tool_name)},
                                {"type", Json("function_call")},
                                {"id", Json(item_id)},
                                {"async", Json()},
                                {"caller", Json()},
                                {"namespace", Json()},
                                {"status", Json("completed")}});
      ev("response.output_item.done", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
      items.push_back(item);
      ++output_index;
      tool_index.reset();
    }
    st = NONE;
  }

  void open(St target, const fe::ToolCallDelta* tc) {
    st = target;
    acc.clear();
    if (target == CONTENT) {
      item_id = "msg_" + random_uuid_hex();
      Json item = Json::object({{"id", Json(item_id)}, {"content", Json::array()}, {"role", Json("assistant")},
                                {"status", Json("in_progress")}, {"type", Json("message")}, {"phase", Json()}});
      ev("response.output_item.added", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
      ev("response.content_part.added", [&](JW& w) {
        w.ki("content_index", 0).kstr("item_id", item_id).ki("output_index", output_index);
        w.k("part").ob().k("annotations").ab().ae().kstr("text", "").kstr("type", "output_text").k("logprobs").ab().ae().oe();
      });
    } else if (target == REASONING) {
      item_id = "rs_" + random_uuid_hex();
      Json item = Json::object({{"id", Json(item_id)}, {"summary", Json::array()}, {"type", Json("reasoning")},
                                {"content", Json()}, {"encrypted_content", Json()}, {"status", Json("in_progress")}});
      ev("response.output_item.added", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
      ev("response.content_part.added", [&](JW& w) {
        w.ki("content_index", 0).kstr("item_id", item_id).ki("output_index", output_index);
        w.k("part").ob().kstr("text", "").kstr("type", "reasoning_text").oe();
      });
    } else if (target == TOOL) {
      item_id = "fc_" + random_uuid_hex();
      tool_call_id = (tc && tc->id) ? *tc->id : fe::random_tool_call_id();
      tool_name = (tc && tc->name) ? *tc->name : "";
      tool_index = tc ? std::optional<int>(tc->index) : std::nullopt;
      Json item = Json::object({{"arguments", Json("")}, {"call_id", Json(tool_call_id)}, {"name", Json(tool_name)},
                                {"type", Json("function_call")}, {"id", Json(item_id)}, {"async", Json()},
                                {"caller", Json()}, {"namespace", Json()}, {"status", Json("in_progress")}});
      ev("response.output_item.added", [&](JW& w) { w.kjson("item", item).ki("output_index", output_index); });
    }
  }

  void handle_delta(const fe::ParserDelta& dm) {
    St target = NONE;
    const fe::ToolCallDelta* tc0 = nullptr;
    if (!dm.tool_calls.empty()) {
      target = TOOL;
      tc0 = &dm.tool_calls[0];
    } else if (dm.reasoning) {
      target = REASONING;
    } else if (dm.content && !dm.content->empty()) {
      target = CONTENT;
    }
    if (target == NONE) return;
    bool transition = st != target || (target == TOOL && tc0 && tool_index && *tool_index != tc0->index);
    if (transition) {
      close_current();
      open(target, tc0);
    }
    if (st == TOOL) {
      std::string combined;
      for (auto& t : dm.tool_calls)
        if (t.arguments) combined += *t.arguments;
      if (!combined.empty()) {
        acc += combined;
        ev("response.function_call_arguments.delta",
           [&](JW& w) { w.kstr("delta", combined).kstr("item_id", item_id).ki("output_index", output_index); });
      }
    } else if (st == REASONING) {
      acc += *dm.reasoning;
      ev("response.reasoning_text.delta", [&](JW& w) {
        w.ki("content_index", 0).kstr("delta", *dm.reasoning).kstr("item_id", item_id).ki("output_index", output_index);
      });
    } else if (st == CONTENT) {
      acc += *dm.content;
      ev("response.output_text.delta", [&](JW& w) {
        w.ki("content_index", 0).kstr("delta", *dm.content).kstr("item_id", item_id).k("logprobs").ab().ae();
        w.ki("output_index", output_index);
      });
    }
  }

  // split_delta: reasoning -> content -> tool_calls grouped by index
  void process(const fe::ParserDelta& d) {
    int parts = int(d.reasoning.has_value()) + int(d.content.has_value()) + int(!d.tool_calls.empty());
    std::vector<int> idxs;
    for (auto& t : d.tool_calls)
      if (std::find(idxs.begin(), idxs.end(), t.index) == idxs.end()) idxs.push_back(t.index);
    if (parts <= 1 && idxs.size() <= 1) {
      handle_delta(d);
      return;
    }
    if (d.reasoning) {
      fe::ParserDelta x;
      x.present = true;
      x.reasoning = d.reasoning;
      handle_delta(x);
    }
    if (d.content) {
      fe::ParserDelta x;
      x.present = true;
      x.content = d.content;
      handle_delta(x);
    }
    for (int i : idxs) {
      fe::ParserDelta x;
      x.present = true;
      for (auto& t : d.tool_calls)
        if (t.index == i) x.tool_calls.push_back(t);
      handle_delta(x);
    }
  }

  void error_event(const ApiError& e) {
    // TypeAdapter(StreamingResponsesResponse).validate_json(error_json): an error payload
    // does not validate into any event type; vLLM raises here (stream ends without an event).
    (void)e;
    finish_stream();
  }

  void finish_stream() {
    if (resp) resp->end();
    if (log) {
      std::lock_guard<std::mutex> lk(log->mu);
      log->done = true;
      log->cv.notify_all();
    }
  }

  bool on_output(GenOutput&& out) {
    if (resp && resp->disconnected() && !r.background) return false;
    if (out.engine_dead || (out.finish_reason && *out.finish_reason == "error")) {
      error_event(generation_error());
      return false;
    }
    if (first_out) {
      cached = out.num_cached_tokens;
      creation = out.num_cache_creation_tokens;
      first_out = false;
    }
    n_out += int64_t(out.token_ids.size());
    all_ids.insert(all_ids.end(), out.token_ids.begin(), out.token_ids.end());
    all_text += out.text;
    if (out.finish_reason) finish = *out.finish_reason;
    bool finished = out.finish_reason.has_value();
    fe::ParserDelta dm;
    if (parser) {
      dm = parser->feed(out.text, out.token_ids, finished);
    } else {
      dm.present = true;
      dm.content = out.text;
    }
    if (dm.present) {
      if (r.parallel_false) {
        std::vector<fe::ToolCallDelta> keep;
        for (auto& t : dm.tool_calls)
          if (t.index == 0) keep.push_back(t);
        dm.tool_calls = std::move(keep);
      }
      process(dm);
    }
    if (out.finished) {
      if (full_parse_metric && finish != "abort") full_parse_metric(all_text, all_ids, finish);
      close_current();
      Usage u;
      u.in = int64_t(r.prompt_ids.size());
      u.out = n_out;
      u.cached = cached;
      u.cache_write = creation;
      if (parser) u.reasoning = parser->reasoning_tokens();
      std::string status = finish == "length" ? "incomplete" : "completed";
      std::string final_resp = response_json(r, status, items, u, FloatStyle::Ryu);
      ev("response.completed", [&](JW& w) { w.kraw("response", final_resp); });
      if (r.store && on_final) {
        StoredResponse sr;
        sr.id = r.id;
        sr.status = status;
        sr.body = response_json(r, status, items, u, FloatStyle::PyRepr);
        sr.output = items;
        sr.created_at = r.created_at;
        on_final(sr);
      }
      finish_stream();
      return false;
    }
    return true;
  }
};

}  // namespace

// ---------------------------------------------------------------------------------------

void App::route_responses_create(Ctx& c) {
  Json body = parse_json_body(c);
  // previous_response_id lookup: fe_render validates first, then raises the 404 itself when
  // prev is null (serving.py order: validation, model check, store checks, lookup).
  std::optional<StoredResponse> prev;
  std::optional<Json> prev_msgs;
  if (auto pid = body.get_string("previous_response_id")) {
    prev = store_->get_response(std::string(*pid));
    if (prev) prev_msgs = store_->get_messages(prev->id);
  }
  fe::ChatRender r;
  {
    fe::ResponsesPrev pv;
    Json null_msgs;
    pv.messages = prev_msgs ? &*prev_msgs : &null_msgs;
    pv.output = prev ? &prev->output : nullptr;
    try {
      r = fe::render_responses(body, prev ? &pv : nullptr, comp_.render);
    } catch (const fe::RenderError& e) {
      throw from_render_error(e);
    }
  }
  check_model(body);
  Json eb = body;
  try {
    before_validators(eb);
    field_validation(eb);
  } catch (const ApiError&) {
  }
  bool store = r.store;
  bool background = eb.get_bool("background").value_or(false);
  if (store && !cfg_.enable_store && background)
    throw ApiError(400, "invalid_request_error",
                   "This vLLM engine does not support `store=True` and therefore does not support the background "
                   "mode. To enable these features, set the environment variable `VLLM_ENABLE_RESPONSES_API_STORE=1` "
                   "when launching the vLLM server.",
                   "background");
  preflight();
  if (store && !cfg_.enable_store) {
    store = false;
    eb.set("store", Json(false));
  }
  std::string request_id = eb.get_string("request_id") ? std::string(*eb.get_string("request_id"))
                                                       : "resp_" + random_uuid_hex();
  if (auto mot = eb.get_int("max_output_tokens")) check_output_tokens_param(*this, *mot, "max_output_tokens");
  Prepared pp = prepare_prompt(*this, r, false);
  int64_t L = app_max_model_len(*this);
  bool truncate = eb["truncation"].is_string() && eb["truncation"].as_string() != "disabled";
  if (!truncate) check_prompt_len(*this, pp.prompt_ids.size());
  if (L > 0 && int64_t(pp.prompt_ids.size()) >= L)
    throw ApiError(400, "invalid_request_error",
                   "The engine prompt length " + std::to_string(pp.prompt_ids.size()) + " exceeds the max_model_len " +
                       std::to_string(L) + ". Please reduce prompt.",
                   "input");

  auto rc = std::make_shared<RespCtx>();
  rc->id = request_id;
  rc->model = cfg_.model_name();
  rc->created_at = int64_t(now_unix());
  rc->body = eb;
  rc->prompt_ids = pp.prompt_ids;
  rc->include_reasoning = eb.get_bool("include_reasoning").value_or(true);
  const Json* ptc = eb.get("parallel_tool_calls");
  rc->parallel_false = ptc && ptc->is_bool() && !ptc->as_bool();
  rc->store = store;
  rc->background = background;
  SamplingSummary ss = backend_->sampling_summary("responses", eb, int64_t(pp.prompt_ids.size()));
  rc->temperature = ss.temperature;
  rc->top_p = ss.top_p;
  rc->presence_penalty = ss.presence_penalty;
  rc->frequency_penalty = ss.frequency_penalty;
  rc->max_output_tokens = ss.max_tokens;
  rc->top_logprobs = ss.logprobs;

  // parser over the Responses tools / tool_choice
  bool use_parser = !cfg_.reasoning_parser.empty() || !cfg_.tool_parser.empty();
  fe::ParserOptions popts;
  {
    std::string kw = pp.chat_template_kwargs.is_null() ? "{}" : dumps_py(pp.chat_template_kwargs, false);
    popts.thinking_initial = fe::parser_thinking(kw);
    popts.include_reasoning = rc->include_reasoning;
    popts.has_tools = eb["tools"].is_array() && eb["tools"].size() > 0;
    const Json& tc = eb.get("tool_choice") ? eb["tool_choice"] : Json("auto");
    if (tc.is_null())
      popts.tool_choice = fe::ToolChoice::Null;
    else if (tc.is_string())
      popts.tool_choice = tc.as_string() == "none"       ? fe::ToolChoice::None
                          : tc.as_string() == "required" ? fe::ToolChoice::Required
                                                         : fe::ToolChoice::Auto;
    else if (tc.is_object()) {
      popts.tool_choice = fe::ToolChoice::Named;
      if (tc["name"].is_string()) popts.named_function = tc["name"].as_string();
    }
    popts.parallel_tool_calls = !rc->parallel_false;
    popts.on_tool_parser_invocation = tool_parser_hook(ToolParserRequestType::Responses, popts.tool_choice);
  }
  // background without stream: built here from the streaming state (DEVIATIONS A8), but Python records the
  // tool-parser invocation of its one non-streaming parse, not one per delta (RespStream::full_parse_metric)
  bool bg_full = background && !eb.get_bool("stream").value_or(false);
  std::shared_ptr<fe::Parser> parser;
  if (use_parser) {
    fe::ParserOptions sopts = popts;
    if (bg_full) sopts.on_tool_parser_invocation = nullptr;
    parser = std::make_shared<fe::Parser>(comp_.vocab, std::move(sopts));
    parser->set_prompt_token_ids(pp.prompt_ids);
  }
  if (store && r.messages.is_array()) store_->put_messages(request_id, r.messages);

  // OnlineRenderer.preprocess_chat -> parser.adjust_request -> DelegatingParser._apply_structural_tag runs for
  // Responses requests too: required / named / strict tools get the deepseek_v4_1 structural tag, and the request's
  // `text` becomes None (echoed as null). fe_parser builds the tag from the chat form of tools / tool_choice.
  std::optional<std::string> stag;
  if (use_parser && eb["tools"].is_array() && eb["tools"].size() > 0) {
    Json cb = Json::object();
    Json ct = Json::array();
    for (auto& t : eb["tools"].as_array()) {
      if (!(t["type"].is_string() && t["type"].as_string() == "function")) continue;
      Json fn = Json::object();
      for (const char* k : {"name", "description", "parameters", "strict"})
        if (const Json* v = t.get(k)) fn.set(k, *v);
      ct.push_back(Json::object({{"type", Json("function")}, {"function", fn}}));
    }
    if (ct.size() > 0) cb.set("tools", ct);
    const Json& tc = eb["tool_choice"];
    if (!eb.get("tool_choice") || tc.is_null()) {  // ResponsesRequest.tool_choice defaults to "auto" (strict tools)
      cb.set("tool_choice", Json("auto"));
    } else if (tc.is_string()) {
      cb.set("tool_choice", tc);
    } else if (tc.is_object() && tc["type"].is_string() && tc["type"].as_string() == "function") {
      cb.set("tool_choice", Json::object({{"type", Json("function")},
                                          {"function", Json::object({{"name", tc["name"]}})}}));
    }
    if (const Json* ptc = eb.get("parallel_tool_calls")) cb.set("parallel_tool_calls", *ptc);
    if (const Json* so = eb.get("structured_outputs"); so && !so->is_null()) cb.set("structured_outputs", *so);
    const Json& fmt = eb["text"]["format"];
    if (fmt.is_object() && fmt["type"].is_string()) {
      if (fmt["type"].as_string() == "json_schema") {
        Json js = Json::object();
        for (const char* k : {"name", "description", "schema", "strict"})
          if (const Json* v = fmt.get(k)) js.set(k, *v);
        cb.set("response_format", Json::object({{"type", Json("json_schema")}, {"json_schema", js}}));
      } else if (fmt["type"].as_string() == "json_object") {
        cb.set("response_format", Json::object({{"type", Json("json_object")}}));
      }
    }
    if (const Json* ds = eb.get("dsml_strict")) cb.set("dsml_strict", *ds);
    if (cb.get("tools") && cb.get("tool_choice")) {
      try {
        stag = fe::structural_tag_for(cb);
      } catch (const std::invalid_argument& e) {
        throw bad_request(e.what());
      } catch (const std::runtime_error& e) {
        throw ApiError(400, "BadRequestError", e.what());
      }
    }
    if (stag) {  // request.structured_outputs = StructuredOutputsParams(structural_tag=...); request.text = None
      eb.erase("text");
      eb.erase("structured_outputs");
      rc->body.erase("text");
    }
  }

  apply_parser_adjust_request(*this, eb, true);
  GenRequest gr = make_gen_request(*this, c, eb, pp, "responses", request_id + "_0");
  gr.structural_tag = stag;
  set_reasoning_fields(*this, gr, eb, pp);
  bool stream = eb.get_bool("stream").value_or(false);
  ResponsesStore* st = store_.get();
  auto stored_cb = [st](const StoredResponse& sr) { st->put_response_unless_cancelled(sr); };

  if (background) {
    StoredResponse q;
    q.id = request_id;
    q.status = "queued";
    q.body = response_json(*rc, "queued", Json::array(), std::nullopt, FloatStyle::PyRepr);
    q.output = Json::array();
    q.created_at = rc->created_at;
    store_->put_response(q);
  }

  if (stream || background) {
    auto rs = std::make_shared<RespStream>();
    rs->app = this;
    rs->r = *rc;
    rs->parser = parser;
    rs->on_final = stored_cb;
    if (bg_full && use_parser && popts.on_tool_parser_invocation) {
      auto vocab = comp_.vocab;
      rs->full_parse_metric = [vocab, popts](const std::string& text, const std::vector<int32_t>& ids,
                                             const std::string& fin) {
        try {
          fe::Parser p(vocab, popts);
          (void)p.parse_full(text, ids, fin.empty() ? "stop" : fin);
        } catch (...) {  // metrics only: the invocation is recorded before a parser exception propagates
        }
      };
    }
    if (background) {
      rs->log = std::make_shared<EventLog>();
      store_->put_events(request_id, rs->log);
    }
    if (stream && !background) {
      rs->resp = c.resp;
      c.resp->start(200, sse_headers());
    }
    rs->start();
    auto pump = std::make_shared<Pump>(rs->resp ? rs->resp : c.resp,
                                       [rs](GenOutput&& out) { return rs->on_output(std::move(out)); });
    if (rs->resp) {
      std::weak_ptr<Pump> wp = pump;
      c.resp->on_writable([wp] {
        if (auto p = wp.lock()) p->kick();
      });
    }
    std::string internal;
    try {
      internal = backend_->start(std::move(gr), [pump](GenOutput&& out) { pump->push(std::move(out)); });
    } catch (const ApiError&) {
      if (stream && !background) {
        rs->finish_stream();
        return;
      }
      throw;
    }
    {
      std::lock_guard<std::mutex> lk(bg_mu_);
      bg_internal_[request_id] = internal;
    }
    if (!background) {
      auto be = backend_;
      c.resp->on_disconnect([be, internal] { be->abort(internal); });
      return;
    }
    if (!stream) {
      send_json(c, 200, response_json(*rc, "queued", Json::array(), std::nullopt, FloatStyle::PyRepr));
      return;
    }
    stream_event_log(c, rs->log, std::nullopt);
    return;
  }

  // ---- non-streaming
  auto resp = c.resp;
  App* self = this;
  auto vocab = comp_.vocab;
  auto acc = std::make_shared<FullAcc>();
  acc->done = [self, resp, rc, use_parser, vocab, popts, stored_cb](GenOutput&& out) {
    Ctx c2{nullptr, resp, ""};
    try {
      if (out.engine_dead || (out.finish_reason && *out.finish_reason == "error")) throw generation_error();
      Json items;
      int64_t rtok = 0;
      if (use_parser) {
        fe::Parser p(vocab, popts);
        fe::Parser::Full pf = p.parse_full(out.text, out.token_ids, out.finish_reason ? *out.finish_reason : "stop");
        rtok = pf.reasoning_tokens;
        items = output_items(*rc, pf);
      } else {
        items = Json::array();
        Json content = out.text.empty() ? Json::array()
                                        : Json::array({Json::object({{"annotations", Json::array()},
                                                                     {"text", Json(out.text)},
                                                                     {"type", Json("output_text")},
                                                                     {"logprobs", Json()}})});
        items.push_back(Json::object({{"id", Json("msg_" + random_uuid_hex())}, {"content", content},
                                      {"role", Json("assistant")}, {"status", Json("completed")},
                                      {"type", Json("message")}, {"phase", Json()}}));
      }
      std::string status = (out.finish_reason && *out.finish_reason == "length") ? "incomplete" : "completed";
      Usage u;
      u.in = int64_t(rc->prompt_ids.size());
      u.out = int64_t(out.token_ids.size());
      u.cached = out.num_cached_tokens;
      u.cache_write = out.num_cache_creation_tokens;
      u.reasoning = rtok;
      std::string b = response_json(*rc, status, items, u, FloatStyle::PyRepr);
      if (rc->store) {
        StoredResponse sr;
        sr.id = rc->id;
        sr.status = status;
        sr.body = b;
        sr.output = items;
        sr.created_at = rc->created_at;
        stored_cb(sr);
      }
      if (resp->disconnected()) return;
      self->send_json(c2, 200, std::move(b));
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

void App::stream_event_log(Ctx& c, std::shared_ptr<EventLog> log, std::optional<int64_t> starting_after) {
  c.resp->start(200, sse_headers());
  auto resp = c.resp;
  // replay on a dedicated thread (rare: background streams)
  std::thread([resp, log, starting_after] {
    size_t idx = 0;
    int64_t start = starting_after ? *starting_after + 1 : 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(log->mu);
      log->cv.wait_for(lk, std::chrono::milliseconds(500),
                       [&] { return idx < log->events.size() || log->done || resp->disconnected(); });
      while (idx < log->events.size()) {
        auto ev = log->events[idx++];
        if (ev.first < start) continue;
        lk.unlock();
        resp->write(ev.second);
        if (ev.second.rfind("event: response.completed\n", 0) == 0) {
          resp->end();
          return;
        }
        lk.lock();
      }
      if (log->done || resp->disconnected()) {
        lk.unlock();
        resp->end();
        return;
      }
    }
  }).detach();
}

void App::route_responses_get(Ctx& c, const std::string& id) {
  auto r = store_->get_response(id);
  if (!r) throw ApiError(404, "invalid_request_error", "Response with id '" + id + "' not found.", "response_id");
  auto stream = c.req->query_param("stream");
  bool s = false;
  if (stream) {
    std::string v = *stream;
    for (auto& ch : v) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    s = v == "true" || v == "1" || v == "yes" || v == "on" || v == "t" || v == "y";
  }
  if (s) {
    auto log = store_->get_events(id);
    if (!log)
      throw vllm_validation_error("Unknown response_id: " + id, "response_id", id);
    std::optional<int64_t> sa;
    if (auto q = c.req->query_param("starting_after")) {
      try {
        sa = std::stoll(*q);
      } catch (...) {
      }
    }
    stream_event_log(c, log, sa);
    return;
  }
  send_json(c, 200, r->body);
}

void App::route_responses_cancel(Ctx& c, const std::string& id) {
  auto r = store_->get_response(id);
  if (!r) throw ApiError(404, "invalid_request_error", "Response with id '" + id + "' not found.", "response_id");
  if (r->status != "queued" && r->status != "in_progress")
    throw ApiError(400, "invalid_request_error", "Cannot cancel a synchronous response.", "response_id");
  r->status = "cancelled";
  // rewrite the status field in the stored body
  size_t pos = r->body.find("\"status\":\"");
  while (pos != std::string::npos) {
    // the top-level status follows "service_tier"
    size_t st = r->body.rfind("\"service_tier\":", pos);
    if (st != std::string::npos) break;
    pos = r->body.find("\"status\":\"", pos + 1);
  }
  if (pos != std::string::npos) {
    size_t vs = pos + 10, ve = r->body.find('"', vs);
    r->body.replace(vs, ve - vs, "cancelled");
  }
  store_->put_response(*r);
  std::string internal;
  {
    std::lock_guard<std::mutex> lk(bg_mu_);
    auto it = bg_internal_.find(id);
    if (it != bg_internal_.end()) internal = it->second;
  }
  if (!internal.empty()) backend_->abort(internal);
  send_json(c, 200, r->body);
}

}  // namespace fe::api
