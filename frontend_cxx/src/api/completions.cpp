// POST /v1/completions (openai/completion/{protocol,serving}.py). Single prompt (string or
// token-id list); n=1. Used by warm.py and the eval scripts.
#include "api/app.h"
#include "api/chat_common.h"
#include "api/pyd.h"
#include "api/util.h"
#include "tokenizer/tokenizer.h"

namespace fe::api {

namespace {
using L = std::vector<std::string>;

void before_validators(Json& d) {
  // reverse definition order: validate_prompt_list_length, validate_prompt_and_prompt_embeds,
  // validate_stream_options, check_logprobs, check_structured_outputs_count,
  // validate_response_format, normalize_null_max_tokens, check_cache_salt_support, alias
  const Json* prompt = d.get("prompt");
  if (prompt && prompt->is_array() && prompt->size() > 0) {
    bool all_int = true;
    for (auto& x : prompt->as_array())
      if (!x.is_integer()) all_int = false;
    const char* mp = std::getenv("VLLM_MAX_COMPLETION_PROMPTS");
    size_t max_prompts = mp ? size_t(std::atol(mp)) : 1024;
    if (!all_int && prompt->size() > max_prompts)
      throw vllm_validation_error("prompt list length " + std::to_string(prompt->size()) +
                                      " exceeds the maximum allowed count of " + std::to_string(max_prompts) +
                                      ". To increase this limit, set the VLLM_MAX_COMPLETION_PROMPTS environment variable.",
                                  "prompt");
  }
  {
    bool prompt_empty = !prompt || prompt->is_null() || (prompt->is_string() && prompt->as_string().empty());
    const Json* pe = d.get("prompt_embeds");
    bool embeds_empty = !pe || pe->is_null() || (pe->is_array() && pe->size() == 0);
    if (prompt_empty && embeds_empty)
      throw vllm_validation_error("Either prompt or prompt_embeds must be provided and non-empty.", "prompt");
  }
  if (py_truthy(d.get("stream_options")) && !py_truthy(d.get("stream")))
    throw vllm_validation_error("Stream options can only be defined when `stream=True`.", "stream_options");
  {
    const Json* lti = d.get("logprob_token_ids");
    if (py_truthy(lti) && py_truthy(d.get("use_beam_search")))
      throw vllm_validation_error("`logprob_token_ids` is not supported with beam search.", "logprob_token_ids");
    const Json* lp = d.get("logprobs");
    if (py_truthy(d.get("use_beam_search")) && py_truthy(d.get("echo")) && lp && !lp->is_null())
      throw vllm_validation_error(
          "`echo` with `logprobs` is not supported with beam search because beam search does not compute prompt logprobs.",
          "echo");
    if (py_truthy(lti) && (!lp || lp->is_null()))
      throw vllm_validation_error("when using `logprob_token_ids`, `logprobs` must be set.", "logprob_token_ids");
    for (const char* f : {"prompt_logprobs", "logprobs"}) {
      const Json* v = d.get(f);
      if (v && !v->is_null() && !v->is_number() && !v->is_bool())
        throw vllm_validation_error(std::string("`") + f + "` must be an integer.", f, py_fstr(*v));
    }
    if (const Json* pl = d.get("prompt_logprobs"); pl && !pl->is_null()) {
      double p = pl->is_bool() ? (pl->as_bool() ? 1 : 0) : pl->as_double();
      if (py_truthy(d.get("stream")) && (p > 0 || p == -1))
        throw vllm_validation_error("`prompt_logprobs` are not available when `stream=True`.", "prompt_logprobs");
      if (p < 0 && p != -1)
        throw vllm_validation_error("`prompt_logprobs` must be a positive value or -1.", "prompt_logprobs", py_fstr(*pl));
    }
    if (lp && !lp->is_null()) {
      double l = lp->is_bool() ? (lp->as_bool() ? 1 : 0) : lp->as_double();
      if (l < 0 && l != -1)
        throw vllm_validation_error("`logprobs` must be a positive value or -1.", "logprobs", py_fstr(*lp));
    }
  }
  if (const Json* so = d.get("structured_outputs"); so && so->is_object()) {
    int count = 0;
    for (const char* k : {"json", "regex", "choice"})
      if (const Json* x = so->get(k); x && !x->is_null()) ++count;
    if (count > 1)
      throw vllm_validation_error(
          "You can only use one kind of constraints for structured outputs ('json', 'regex' or 'choice').",
          "structured_outputs");
  }
  if (const Json* rf = d.get("response_format"); rf && rf->is_object()) {
    const Json& t = (*rf)["type"];
    if (t.is_string() && t.as_string() == "json_schema" && (*rf)["json_schema"].is_null())
      throw vllm_validation_error(
          "When response_format type is 'json_schema', the 'json_schema' field must be provided.", "response_format");
  }
  if (const Json* mt = d.get("max_tokens"); !mt || mt->is_null()) d.set("max_tokens", Json(16));
  validate_cache_salt(d.get("cache_salt"));
}

}  // namespace

void validate_completion_prompt(Pyd& v, const Json* p) {
  auto B = [](const char* f) { return L{"body", f}; };
  // prompt: list[Annotated[int, Field(ge=0)]] | list[list[Annotated[int, Field(ge=0)]]] | str | list[str] | None.
  // Smart union: accepted when a member matches strictly; otherwise the lax errors of every member in order.
  // (A lax-only match - 2.0, "5" or true where an int goes - is not coerced: rejected with those errors.)
  if (p && !p->is_null() && !p->is_string()) {
    if (!p->is_array()) {
      v.add("list_type", B("prompt"), "Input should be a valid list", p);
      v.add("list_type", B("prompt"), "Input should be a valid list", p);
      v.add("string_type", B("prompt"), "Input should be a valid string", p);
      v.add("list_type", B("prompt"), "Input should be a valid list", p);
    } else {
      auto token_id = [](const Json& x) {  // strict int >= 0
        return x.is_int() ? x.as_int() >= 0 : x.is_big_int() && x.big_int_text()[0] != '-';
      };
      bool ids = true, str_list = true, id_lists = true;
      for (auto& x : p->as_array()) {
        if (!token_id(x)) ids = false;
        if (!x.is_string()) str_list = false;
        if (!x.is_array()) id_lists = false;
        else
          for (auto& y : x.as_array())
            if (!token_id(y)) id_lists = false;
      }
      if (!ids && !str_list && !id_lists) {
        Pyd a, b, cc;
        for (size_t i = 0; i < p->size(); ++i) a.to_int(&(*p)[i], L{"body", "prompt", std::to_string(i)}, false, 0);
        for (size_t i = 0; i < p->size(); ++i) {
          const Json& x = (*p)[i];
          if (b.is_list(&x, L{"body", "prompt", std::to_string(i)}, false))
            for (size_t j = 0; j < x.size(); ++j)
              b.to_int(&x[j], L{"body", "prompt", std::to_string(i), std::to_string(j)}, false, 0);
        }
        for (auto& e : a.errs) v.errs.push_back(e);
        for (auto& e : b.errs) v.errs.push_back(e);
        v.add("string_type", B("prompt"), "Input should be a valid string", p);
        for (size_t i = 0; i < p->size(); ++i) cc.to_str(&(*p)[i], L{"body", "prompt", std::to_string(i)}, false);
        for (auto& e : cc.errs) v.errs.push_back(e);
      }
    }
  }
}

namespace {

void field_validation(Json& d) {
  Pyd v;
  auto B = [](const char* f) { return L{"body", f}; };
  v.to_str(d.get("model"), B("model"), true);
  validate_completion_prompt(v, d.get("prompt"));
  coerce_bool(d, v, "echo", true);
  coerce_float(d, v, "frequency_penalty");
  if (const Json* lb = d.get("logit_bias")) v.is_dict(lb, B("logit_bias"), true);
  coerce_int(d, v, "logprobs", true);
  coerce_int(d, v, "max_tokens", true);
  coerce_int(d, v, "n", false);
  coerce_float(d, v, "presence_penalty");
  coerce_int(d, v, "seed", true, INT64_MIN, INT64_MAX);
  validate_stop(v, d.get("stop"));
  coerce_bool(d, v, "stream", true);
  if (Json* so = d.get("stream_options"); so && v.is_model(so, B("stream_options"), true)) {
    for (const char* f : {"include_usage", "continuous_usage_stats"})
      if (Json* x = so->get(f)) {
        auto b = v.to_bool(x, L{"body", "stream_options", f}, true);
        if (b) so->set(f, Json(*b));
      }
  }
  v.to_str(d.get("suffix"), B("suffix"), true);
  coerce_float(d, v, "temperature");
  coerce_float(d, v, "top_p");
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
  validate_int_list(v, d, "allowed_token_ids", true);
  coerce_int(d, v, "prompt_logprobs", true);
  validate_int_list(v, d, "logprob_token_ids", true);
  validate_str_list(v, d, "bad_words", false);
  coerce_bool(d, v, "add_special_tokens", false);
  coerce_int(d, v, "priority", false, INT64_MIN, INT64_MAX);
  v.to_str(d.get("request_id"), B("request_id"), false);
  coerce_bool(d, v, "return_tokens_as_token_ids", true);
  coerce_bool(d, v, "return_token_ids", true);
  coerce_int(d, v, "routed_experts_prompt_start", false, 0, std::nullopt);
  validate_cache_salt_field(v, d);
  coerce_int(d, v, "stream_interval", true, 1, std::nullopt);
  if (!v.ok()) throw validation_error(v.errs);
}

struct CompOpts {
  std::string id, model;
  int64_t created = 0;
  bool include_usage = false, cont_usage = false, return_token_ids = false, echo = false;
  int64_t max_tokens_req = 16;
  std::optional<std::string> fingerprint;
  bool prompt_details = false;
  std::vector<int32_t> prompt_ids;
  std::string prompt_text;
};

class CompStream {
 public:
  std::shared_ptr<http::Response> resp;
  CompOpts o;
  bool first = true, has_echoed = false;
  int64_t prev_tokens = 0;
  int64_t cached = 0;
  Json last_spec;

  void emit(const std::string& j) { resp->write("data: " + j + "\n\n"); }
  void emit_error(const ApiError& e) {
    emit(error_body_stream(e));
    resp->write("data: [DONE]\n\n");
    resp->end();
  }
  bool on_output(GenOutput&& out) {
    if (resp->disconnected()) return false;
    if (out.engine_dead) {
      emit_error(ApiError(500, "InternalServerError",
                          "EngineCore encountered an issue. See stack trace (above) for the root cause."));
      return false;
    }
    if (first) {
      cached = out.num_cached_tokens;
      first = false;
    }
    if (!out.spec_decode.is_null()) last_spec = out.spec_decode;
    if (out.finish_reason && *out.finish_reason == "error") {
      emit_error(ApiError(500, "InternalServerError", "Internal server error"));
      return false;
    }
    std::string text;
    const std::vector<int32_t>* ids = &out.token_ids;
    std::vector<int32_t> echo_ids;
    bool ret_prompt_ids = false;
    bool skip = false;
    if (o.echo && !has_echoed) {
      std::string ptext = o.return_token_ids ? "" : o.prompt_text;
      if (o.max_tokens_req == 0) {
        text = ptext;
        echo_ids = o.prompt_ids;
      } else {
        text = ptext + out.text;
        echo_ids = o.prompt_ids;
        echo_ids.insert(echo_ids.end(), out.token_ids.begin(), out.token_ids.end());
      }
      (void)echo_ids;
      ret_prompt_ids = true;
      has_echoed = true;
    } else {
      text = out.text;
      if (!has_echoed && o.return_token_ids) {
        ret_prompt_ids = true;
        has_echoed = true;
      }
      if (text.empty() && out.token_ids.empty() && prev_tokens == 0) skip = true;
    }
    if (!skip) {
      prev_tokens += int64_t(out.token_ids.size());
      JW w(true, false, FloatStyle::Ryu);
      w.ob().kstr("id", o.id).kstr("object", "text_completion").ki("created", o.created).kstr("model", o.model);
      w.k("choices").ab().ob().ki("index", 0).kstr("text", text).knull("logprobs");
      if (out.finish_reason)
        w.kstr("finish_reason", *out.finish_reason);
      else
        w.knull("finish_reason");
      w.kjson("stop_reason", out.stop_reason);
      if (ret_prompt_ids)
        w.k("prompt_token_ids").int_array(o.prompt_ids);
      else
        w.knull("prompt_token_ids");
      if (o.return_token_ids)
        w.k("token_ids").int_array(*ids);
      else
        w.knull("token_ids");
      w.oe().ae();
      if (o.cont_usage) {
        int64_t p = int64_t(o.prompt_ids.size());
        w.k("usage").ob().ki("prompt_tokens", p).ki("total_tokens", p + prev_tokens).ki("completion_tokens", prev_tokens).oe();
      }
      if (!o.include_usage && o.fingerprint && out.finish_reason) w.kstr("system_fingerprint", *o.fingerprint);
      w.oe();
      emit(w.take());
    }
    if (out.finished) {
      if (o.include_usage) {
        int64_t p = int64_t(o.prompt_ids.size());
        JW w(true, false, FloatStyle::Ryu);
        w.ob().kstr("id", o.id).kstr("object", "text_completion").ki("created", o.created).kstr("model", o.model);
        w.k("choices").ab().ae();
        w.k("usage").ob().ki("prompt_tokens", p).ki("total_tokens", p + prev_tokens).ki("completion_tokens", prev_tokens);
        if (o.prompt_details) w.k("prompt_tokens_details").ob().ki("cached_tokens", cached).oe();
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
      return false;
    }
    return true;
  }
};

std::string build_comp_full(const CompOpts& o, GenOutput&& out) {
  if (out.engine_dead)
    throw ApiError(500, "InternalServerError", "EngineCore encountered an issue. See stack trace (above) for the root cause.");
  if (out.finish_reason && *out.finish_reason == "error") throw generation_error();
  std::string text = out.text;
  if (o.echo) {
    std::string ptext = o.return_token_ids ? "" : o.prompt_text;
    text = o.max_tokens_req == 0 ? ptext : ptext + out.text;
  }
  int64_t p = int64_t(o.prompt_ids.size());
  int64_t n = int64_t(out.token_ids.size());
  JW w(true, false, FloatStyle::PyRepr);
  w.ob().kstr("id", o.id).kstr("object", "text_completion").ki("created", o.created).kstr("model", o.model);
  w.k("choices").ab().ob().ki("index", 0).kstr("text", text).knull("logprobs");
  if (out.finish_reason)
    w.kstr("finish_reason", *out.finish_reason);
  else
    w.knull("finish_reason");
  w.kjson("stop_reason", out.stop_reason);
  if (o.return_token_ids)
    w.k("token_ids").int_array(out.token_ids);
  else
    w.knull("token_ids");
  w.knull("prompt_logprobs");
  if (o.return_token_ids)
    w.k("prompt_token_ids").int_array(o.prompt_ids);
  else
    w.knull("prompt_token_ids");
  w.knull("routed_experts");
  w.oe().ae();
  w.knull("service_tier");
  if (o.fingerprint)
    w.kstr("system_fingerprint", *o.fingerprint);
  else
    w.knull("system_fingerprint");
  w.k("usage").ob().ki("prompt_tokens", p).ki("total_tokens", p + n).ki("completion_tokens", n);
  if (o.prompt_details)
    w.k("prompt_tokens_details").ob().ki("cached_tokens", out.num_cached_tokens).knull("created_cache_tokens").knull("multimodal_tokens").oe();
  else
    w.knull("prompt_tokens_details");
  w.knull("completion_tokens_details");
  w.oe();
  w.knull("kv_transfer_params").knull("ec_transfer_params");
  write_metrics_full(w, out.spec_decode);
  w.oe();
  return w.take();
}
}  // namespace

void App::route_completions(Ctx& c) {
  Json body = parse_json_body(c);
  before_validators(body);
  field_validation(body);
  bool stream = body.get_bool("stream").value_or(false);
  if (stream && body.get_bool("use_beam_search").value_or(false))
    throw bad_request("Streaming is not currently supported with beam search");
  check_model(body);
  preflight();
  if (auto n = body.get_int("n"); n && *n > 1)
    throw bad_request("frontend_cxx: n > 1 is not supported (DEVIATIONS.md)", std::string("n"));
  const Json& prompt = body["prompt"];
  int64_t req_max = body.get_int("max_tokens").value_or(16);
  check_output_tokens_param(*this, req_max, "max_tokens");
  // TokenizeParams.__post_init__ (build_tok_params): truncate_prompt_tokens <= max_model_len
  std::optional<int64_t> truncate = body.get_int("truncate_prompt_tokens");
  int64_t L = app_max_model_len(*this);
  if (truncate && L > 0 && *truncate > L)
    throw vllm_validation_error("truncate_prompt_tokens=" + std::to_string(*truncate) +
                                    " cannot be greater than max_model_len = " + std::to_string(L) +
                                    ". Please request a smaller truncation size.",
                                std::string("truncate_prompt_tokens"), std::to_string(*truncate));
  // render_prompts: prompt_to_seq([]) is an empty batch
  if (prompt.is_array() && prompt.size() == 0) throw bad_request("You must pass at least one prompt");
  // TokenizeParams._token_truncation: the request's side, else the tokenizer's ("left" as vLLM loads V4.1)
  const Json& side = body["truncation_side"];
  bool left = !(side.is_string() && side.as_string() == "right");
  auto cut = [&](size_t n) { return truncation_slice(n, truncate, left, L, req_max); };
  bool echo = body.get_bool("echo").value_or(false);
  Prepared pp;
  std::optional<std::string> prompt_oov_id;
  auto enc = [&](const std::string& t) {
    std::vector<int32_t> ids;
    try {
      ids = comp_.tokenizer->encode(t);
    } catch (const fe::TokenizerError& e) {
      throw ApiError(400, "BadRequestError", e.what());
    }
    auto [b, e] = cut(ids.size());
    if (b != 0 || e != ids.size()) ids = std::vector<int32_t>(ids.begin() + b, ids.begin() + e);
    return ids;
  };
  // Token-id prompt: the echo text is the detokenized prompt BEFORE truncation (_tokenize_singleton_prompt).
  auto ids_prompt = [&](const Json& arr) {
    auto [b, e] = cut(arr.size());
    pp.prompt_ids = token_prompt_ids(arr, prompt_oov_id, b, e);
    if (echo) {
      std::optional<std::string> unused;
      pp.prompt = comp_.tokenizer->decode(b == 0 && e == arr.size() ? pp.prompt_ids : token_prompt_ids(arr, unused),
                                          false);
    }
  };
  if (prompt.is_string()) {
    pp.prompt = prompt.as_string();
    pp.prompt_ids = enc(prompt.as_string());
  } else if (prompt.is_array() && prompt[0].is_integer()) {
    ids_prompt(prompt);
  } else if (prompt.is_array() && prompt.size() == 1) {
    const Json& p0 = prompt[0];
    if (p0.is_string()) {
      pp.prompt = p0.as_string();
      pp.prompt_ids = enc(p0.as_string());
    } else if (p0.is_array()) {
      ids_prompt(p0);
    }
  } else {
    throw bad_request("Only a single prompt is supported by this frontend.", "prompt");
  }
  check_prompt_len(*this, pp.prompt_ids.size());
  std::string request_id = "cmpl-" + base_request_id(c, body.get("request_id"));
  CompOpts o;
  o.id = request_id;
  o.model = cfg_.model_name();
  o.created = int64_t(now_unix());
  const Json& so = body["stream_options"];
  bool force = cfg_.enable_force_include_usage;
  o.include_usage = force || (so.is_object() && so.get_bool("include_usage").value_or(false));
  o.cont_usage = force || (o.include_usage && so.is_object() && so.get_bool("continuous_usage_stats").value_or(false));
  o.return_token_ids = body.get_bool("return_token_ids").value_or(false);
  o.echo = body.get_bool("echo").value_or(false);
  o.max_tokens_req = req_max;
  o.fingerprint = cfg_.system_fingerprint;
  o.prompt_details = cfg_.enable_prompt_tokens_details;
  o.prompt_ids = pp.prompt_ids;
  o.prompt_text = pp.prompt;

  GenRequest gr = make_gen_request(*this, c, body, pp, "completion", request_id + "-0");
  gr.prompt_oov_id = std::move(prompt_oov_id);
  if (stream) backend_->precheck_caps("completion", gr.body, int64_t(pp.prompt_ids.size()));   // caps 400 first
  if (stream) {
    auto st = std::make_shared<CompStream>();
    st->resp = c.resp;
    st->o = o;
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
      st->emit_error(e);
      return;
    }
    auto be = backend_;
    c.resp->on_disconnect([be, internal] { be->abort(internal); });
    return;
  }
  auto resp = c.resp;
  App* self = this;
  auto acc = std::make_shared<FullAcc>();
  acc->done = [self, resp, o](GenOutput&& out) {
    Ctx c2{nullptr, resp, ""};
    try {
      if (resp->disconnected()) return;
      self->send_json(c2, 200, build_comp_full(o, std::move(out)));
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
