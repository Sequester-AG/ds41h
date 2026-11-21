#include "api/chat_common.h"

#include <algorithm>

#include "api/app.h"
#include "api/util.h"
#include "image/image.h"
#include "tokenizer/tokenizer.h"

namespace fe::api {

using L = std::vector<std::string>;

static std::optional<int64_t> opt_int(const Json* x) {
  if (!x) return std::nullopt;
  if (x->is_int()) return x->as_int();
  if (x->is_big_int()) return x->big_int_text()[0] == '-' ? INT64_MIN : INT64_MAX;
  return std::nullopt;
}

void coerce_int(Json& d, Pyd& v, const char* f, bool nullable, std::optional<int64_t> ge, std::optional<int64_t> le) {
  Json* x = d.get(f);
  if (!x) return;
  auto r = v.to_int(x, L{"body", f}, nullable, ge, le);
  if (r && !x->is_int() && !x->is_big_int()) d.set(f, Json(*r));
}

void coerce_float(Json& d, Pyd& v, const char* f, bool nullable) {
  Json* x = d.get(f);
  if (!x) return;
  auto r = v.to_float(x, L{"body", f}, nullable);
  if (r && !x->is_float()) d.set(f, Json(*r));
}

void coerce_bool(Json& d, Pyd& v, const char* f, bool nullable) {
  Json* x = d.get(f);
  if (!x) return;
  auto r = v.to_bool(x, L{"body", f}, nullable);
  if (r && !x->is_bool()) d.set(f, Json(*r));
}

void validate_int_list(Pyd& v, Json& d, const char* f, bool nullable) {
  Json* x = d.get(f);
  if (!x) return;
  if (!v.is_list(x, L{"body", f}, nullable)) return;
  auto& arr = x->as_array();
  for (size_t i = 0; i < arr.size(); ++i) {
    auto r = v.to_int(&arr[i], L{"body", f, std::to_string(i)}, false);
    if (r && !arr[i].is_int()) arr[i] = Json(*r);
  }
}

void validate_str_list(Pyd& v, Json& d, const char* f, bool nullable) {
  Json* x = d.get(f);
  if (!x) return;
  if (!v.is_list(x, L{"body", f}, nullable)) return;
  auto& arr = x->as_array();
  for (size_t i = 0; i < arr.size(); ++i) v.to_str(&arr[i], L{"body", f, std::to_string(i)}, false);
}

void validate_stop(Pyd& v, const Json* stop) {
  // StopParam = str | list[str] (max VLLM_MAX_STOP_STRINGS) | None
  if (!stop || stop->is_null() || stop->is_string()) return;
  if (stop->is_array()) {
    Pyd tmp;
    for (size_t i = 0; i < stop->size(); ++i) tmp.to_str(&(*stop)[i], L{"body", "stop", std::to_string(i)}, false);
    if (!tmp.ok()) {
      // union: str branch first, then list branch errors
      v.add("string_type", L{"body", "stop"}, "Input should be a valid string", stop);
      for (auto& e : tmp.errs) v.errs.push_back(e);
    }
    return;
  }
  v.add("string_type", L{"body", "stop"}, "Input should be a valid string", stop);
  v.add("list_type", L{"body", "stop"}, "Input should be a valid list", stop);
}

void validate_messages(Pyd& v, const Json& msgs) {
  static const char* kUnion[] = {"ChatCompletionDeveloperMessageParam", "ChatCompletionSystemMessageParam",
                                 "ChatCompletionUserMessageParam",      "ChatCompletionAssistantMessageParam",
                                 "ChatCompletionToolMessageParam",      "ChatCompletionFunctionMessageParam",
                                 "CustomChatCompletionMessageParam"};
  for (size_t i = 0; i < msgs.size(); ++i) {
    const Json& m = msgs[i];
    std::string idx = std::to_string(i);
    if (!m.is_object()) {
      for (auto* u : kUnion) v.add("dict_type", L{"body", "messages", idx, u}, "Input should be a valid dictionary", &m);
      v.add("model_attributes_type", L{"body", "messages", idx, "Message"},
            "Input should be a valid dictionary or object to extract fields from", &m);
      continue;
    }
    if (!m.get("role")) {
      auto miss = [&](const char* u, const char* f) { v.missing(L{"body", "messages", idx, u, f}, m); };
      miss("ChatCompletionDeveloperMessageParam", "role");
      miss("ChatCompletionSystemMessageParam", "role");
      miss("ChatCompletionUserMessageParam", "role");
      miss("ChatCompletionAssistantMessageParam", "role");
      miss("ChatCompletionToolMessageParam", "role");
      if (!m.get("tool_call_id")) miss("ChatCompletionToolMessageParam", "tool_call_id");
      if (!m.get("name")) miss("ChatCompletionFunctionMessageParam", "name");
      miss("ChatCompletionFunctionMessageParam", "role");
      miss("CustomChatCompletionMessageParam", "role");
      if (!m.get("author")) miss("Message", "author");
      // (pydantic reports one more union branch error here; only the count is visible)
      v.missing(L{"body", "messages", idx, "Message", "content"}, m);
    }
  }
}

void validate_tools(Pyd& v, const Json& tools) {
  for (size_t i = 0; i < tools.size(); ++i) {
    const Json& t = tools[i];
    std::string idx = std::to_string(i);
    if (!v.is_model(&t, L{"body", "tools", idx}, false)) continue;
    if (const Json* ty = t.get("type")) v.to_literal(ty, L{"body", "tools", idx, "type"}, false, {"function"});
    const Json* fn = t.get("function");
    if (!fn) {
      v.missing(L{"body", "tools", idx, "function"}, t);
      continue;
    }
    if (!v.is_model(fn, L{"body", "tools", idx, "function"}, false)) continue;
    if (!fn->get("name"))
      v.missing(L{"body", "tools", idx, "function", "name"}, *fn);
    else
      v.to_str(fn->get("name"), L{"body", "tools", idx, "function", "name"}, false);
    v.to_str(fn->get("description"), L{"body", "tools", idx, "function", "description"}, true);
    if (const Json* p = fn->get("parameters")) v.is_dict(p, L{"body", "tools", idx, "function", "parameters"}, true);
    v.to_bool(fn->get("strict"), L{"body", "tools", idx, "function", "strict"}, true);
  }
}

void validate_cache_salt(const Json* salt) {
  if (!salt || salt->is_null()) return;
  if (!salt->is_string() || salt->as_string().empty())
    throw vllm_validation_error("Parameter 'cache_salt' must be a non-empty string if provided.", "cache_salt");
  const std::string& s = salt->as_string();
  bool bad = py_len(s) > 128;
  for (char ch : s)
    if (ch == '@' || ch == '/' || ch == '\\' || ch == '\0') bad = true;
  if (bad)
    throw vllm_validation_error(
        "Parameter 'cache_salt' must be at most 128 characters and must not contain '@', '/', '\\\\', or NUL.",
        "cache_salt");
}

void validate_cache_salt_field(Pyd& v, const Json& d) {
  const Json* x = d.get("cache_salt");
  if (!x) return;
  v.to_str(x, L{"body", "cache_salt"}, true);
}

// ---------------------------------------------------------------------------------------

int64_t app_max_model_len(const App& app) {
  if (app.backend_ && app.backend_->max_model_len() > 0) return app.backend_->max_model_len();
  return app.cfg_.max_model_len;
}

void check_output_tokens_param(const App& app, std::optional<int64_t> max_out, const char* param) {
  int64_t L = app_max_model_len(app);
  int64_t mo = max_out.value_or(0);
  if (L > 0 && mo > L)
    throw vllm_validation_error(std::string(param) + "=" + std::to_string(mo) + " cannot be greater than max_model_len=" +
                                    std::to_string(L) + ". Please request fewer output tokens.",
                                std::string(param), std::to_string(mo));
}

void check_prompt_len(const App& app, size_t n) {
  int64_t L = app_max_model_len(app);
  if (L <= 0 || int64_t(n) <= L) return;
  // The tokenizer truncates to max_length = L + 1, hence "at least".
  int64_t count = std::min<int64_t>(int64_t(n), L + 1);
  std::string q = count == L + 1 ? "at least " : "";
  throw vllm_validation_error("This model's maximum context length is " + std::to_string(L) +
                                  " tokens. However, your prompt contains " + q + std::to_string(count) +
                                  " input tokens, which exceeds this limit even without any output tokens. "
                                  "Please reduce the length of the input prompt.",
                              "input_tokens", std::to_string(count));
}

int64_t get_max_tokens(int64_t L, std::optional<int64_t> requested, int64_t prompt_len) {
  if (L < prompt_len)
    throw bad_request("Input length (" + std::to_string(prompt_len) + ") exceeds model's maximum context length (" +
                      std::to_string(L) + ").");
  int64_t m = L - prompt_len;
  if (requested) m = std::min(m, *requested);
  return m;
}

Prepared prepare_prompt(App& app, const fe::ChatRender& r, bool check_len) {
  Prepared pp;
  pp.prompt = r.prompt;
  pp.chat_template_kwargs = r.chat_template_kwargs;
  if (r.reused_prompt_token_ids) {
    pp.prompt_ids = *r.reused_prompt_token_ids;
  } else {
    try {
      pp.prompt_ids = app.comp_.tokenizer->encode(r.prompt);
    } catch (const fe::TokenizerError& e) {
      // Python: TypeError from the Rust tokenizer -> 400
      throw ApiError(400, "BadRequestError", e.what());
    }
  }
  if (check_len) check_prompt_len(app, pp.prompt_ids.size());
  if (!r.images.empty()) {
    std::vector<fe::ImageFeatures> feats;
    for (auto& im : r.images) {
      try {
        if (!fe::is_data_url(im.url))
          throw ApiError(400, "BadRequestError",
                         "frontend_cxx: only data: image URLs are supported (remote image fetch is not implemented)");
        auto bytes = fe::decode_data_url(im.url);
        feats.push_back(fe::preprocess_image(bytes, app.comp_.vision));
      } catch (const fe::ImageError& e) {
        throw ApiError(400, "BadRequestError", e.what());
      }
    }
    try {
      pp.image_offsets = fe::expand_image_placeholders(pp.prompt_ids, feats, app.comp_.vision.image_token_id);
    } catch (const fe::ImageError& e) {
      throw ApiError(400, "BadRequestError", e.what());
    }
    for (auto& f : feats) {
      pp.mm_tokens += f.num_tokens;
      pp.images.push_back(std::make_shared<const fe::ImageFeatures>(std::move(f)));
    }
  }
  return pp;
}

void set_reasoning_fields(App& app, GenRequest& g, const Json& body, const Prepared& pp) {
  if (app.cfg_.reasoning_parser.empty()) return;
  bool include_reasoning = body.get_bool("include_reasoning").value_or(true);
  std::string kw = pp.chat_template_kwargs.is_null() ? "{}" : dumps_py(pp.chat_template_kwargs, false);
  bool thinking = fe::parser_thinking(kw);
  g.reasoning_ended = fe::engine_reasoning_ended(pp.prompt_ids, *app.comp_.vocab, thinking, include_reasoning);
  Json k = Json::object();
  k.set("chat_template_kwargs", pp.chat_template_kwargs.is_null() ? Json::object() : pp.chat_template_kwargs);
  g.reasoning_parser_kwargs = std::move(k);
}

void apply_parser_adjust_request(const App& app, Json& body, bool responses) {
  const Config& cfg = app.config();
  bool reasoning = !cfg.reasoning_parser.empty();
  bool tools = !cfg.tool_parser.empty();
  if (!reasoning && !tools) return;
  if (!reasoning) {
    // tool_choice after validation: chat: absent -> "auto" with tools, else "none"; ResponsesRequest: "auto"
    const Json& tc = body["tool_choice"];
    std::string choice;
    if (tc.is_string()) {
      choice = tc.as_string();
    } else if (tc.is_null() || !body.get("tool_choice")) {
      const Json& t = body["tools"];
      choice = (responses || (t.is_array() && t.size() > 0)) ? "auto" : "none";
    } else {
      choice = "named";
    }
    if (choice == "none") return;
  }
  body.set("skip_special_tokens", Json(false));
}

GenRequest make_gen_request(App& app, Ctx& c, const Json& body, const Prepared& pp, const std::string& endpoint,
                            const std::string& external_id) {
  GenRequest g;
  g.external_id = external_id;
  g.endpoint = endpoint;
  g.body = body;
  g.prompt_ids = pp.prompt_ids;
  g.images = pp.images;
  g.image_offsets = pp.image_offsets;
  g.priority = body.get_int("priority").value_or(0);
  if (c.req) {
    if (const std::string* ph = c.req->header("x-vllm-priority")) {
      try {
        size_t pos = 0;
        long long v = std::stoll(*ph, &pos);
        if (pos == ph->size()) g.priority = v;
      } catch (...) {
      }
    }
    // _get_data_parallel_rank (generate/base/serving.py:220): int(header), None on ValueError. Chat and completions
    // only: the Responses API never reads it. The range check runs in fe_engine (process_inputs).
    if (const std::string* dr = c.req->header("x-data-parallel-rank"); dr && endpoint != "responses") {
      g.data_parallel_rank = py_int_header(*dr, &g.data_parallel_rank_text);
      if (!g.data_parallel_rank) g.data_parallel_rank_text.clear();
    }
  }
  if (auto s = body.get_string("session_id"); s && !s->empty())
    g.session_id = std::string(*s);
  else if (c.req && c.req->header("x-session-id") && !c.req->header("x-session-id")->empty())
    g.session_id = *c.req->header("x-session-id");
  else if (const Json& xs = body["vllm_xargs"]["session_id"]; xs.is_string() && !xs.as_string().empty())
    g.session_id = xs.as_string();
  g.arrival_time = now_unix();
  return g;
}

void FullAcc::on(GenOutput&& o) {
  std::function<void(GenOutput&&)> fn;
  GenOutput fin;
  {
    std::lock_guard<std::mutex> lk(mu);
    if (fired) return;
    if (!any) {
      all = std::move(o);
      any = true;
    } else {
      merge_output(all, std::move(o));
    }
    if (!(all.finished || all.engine_dead)) return;
    fired = true;
    fin = std::move(all);
    fn = std::move(done);
  }
  if (fn) fn(std::move(fin));
}

void FullAcc::cancel() {
  std::lock_guard<std::mutex> lk(mu);
  fired = true;
  done = nullptr;
}

}  // namespace fe::api
