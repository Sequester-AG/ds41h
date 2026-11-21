// POST /tokenize, /detokenize (serve/tokenize/*).
#include "api/app.h"
#include "api/chat_common.h"
#include "api/pyd.h"
#include "api/util.h"
#include "tokenizer/tokenizer.h"

namespace fe::api {

namespace {
using L = std::vector<std::string>;
}

void App::route_tokenize(Ctx& c) {
  Json body = parse_json_body(c);
  // TokenizeRequest = TokenizeCompletionRequest | TokenizeChatRequest (smart union: the
  // first branch that validates wins; both branches' errors are reported otherwise).
  bool is_chat = false;
  {
    Pyd a;
    const Json* prompt = body.get("prompt");
    if (!prompt)
      a.missing(L{"body", "prompt"}, body);
    else
      a.to_str(prompt, L{"body", "prompt"}, false);
    a.to_str(body.get("model"), L{"body", "model"}, true);
    a.to_bool(body.get("add_special_tokens"), L{"body", "add_special_tokens"}, false);
    a.to_bool(body.get("return_token_strs"), L{"body", "return_token_strs"}, true);
    if (!a.ok()) {
      // chat branch: its before-validator runs first
      if (py_truthy(body.get("continue_final_message")) && py_truthy(body.get("add_generation_prompt")))
        throw vllm_validation_error("Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
                                    std::nullopt);
      Pyd b;
      const Json* msgs = body.get("messages");
      if (!msgs)
        b.missing(L{"body", "messages"}, body);
      else if (b.is_list(msgs, L{"body", "messages"}, false))
        validate_messages(b, *msgs);
      b.to_str(body.get("model"), L{"body", "model"}, true);
      if (const Json* t = body.get("tools"); t && b.is_list(t, L{"body", "tools"}, true)) validate_tools(b, *t);
      if (const Json* k = body.get("chat_template_kwargs")) b.is_dict(k, L{"body", "chat_template_kwargs"}, true);
      if (!b.ok()) {
        // pydantic reports the errors of the first branch, then the second, per field location
        std::vector<PydErr> all = a.errs;
        for (auto& e : b.errs) all.push_back(e);
        throw validation_error(all);
      }
      is_chat = true;
    }
  }
  std::vector<int32_t> ids;
  if (is_chat) {
    fe::ChatRender r;
    try {
      r = fe::render_tokenize_chat(body, comp_.render);
    } catch (const fe::RenderError& e) {
      throw from_render_error(e);
    }
    check_model(body);
    Prepared pp = prepare_prompt(*this, r, false);  // max_total_tokens=None: no length check
    ids = std::move(pp.prompt_ids);
  } else {
    check_model(body);
    try {
      ids = comp_.tokenizer->encode(body["prompt"].as_string());
    } catch (const fe::TokenizerError& e) {
      throw ApiError(400, "BadRequestError", e.what());
    }
  }
  bool strs = body.get_bool("return_token_strs").value_or(false);
  JW w(true, false);
  w.ob().ki("count", int64_t(ids.size())).ki("max_model_len", app_max_model_len(*this));
  w.k("tokens").int_array(ids);
  if (strs) {
    w.k("token_strs").ab();
    for (auto id : ids) {
      auto t = comp_.tokenizer->id_to_token(id);
      if (t)
        w.str(*t);
      else
        w.null();
    }
    w.ae();
  } else {
    w.knull("token_strs");
  }
  w.oe();
  send_json(c, 200, w.take());
}

void App::route_detokenize(Ctx& c) {
  Json body = parse_json_body(c);
  Pyd v;
  v.to_str(body.get("model"), L{"body", "model"}, true);
  const Json* toks = body.get("tokens");
  std::vector<int32_t> ids;
  if (!toks) {
    v.missing(L{"body", "tokens"}, body);
  } else if (v.is_list(toks, L{"body", "tokens"}, false)) {
    for (size_t i = 0; i < toks->size(); ++i) {
      auto r = v.to_int(&(*toks)[i], L{"body", "tokens", std::to_string(i)}, false, 0, INT64_MAX);
      if (r) ids.push_back(int32_t(*r > INT32_MAX ? -1 : *r));
    }
  }
  if (!v.ok()) throw validation_error(v.errs);
  check_model(body);
  std::string text = comp_.tokenizer->decode(ids, false);
  JW w(true, false);
  w.ob().kstr("prompt", text).oe();
  send_json(c, 200, w.take());
}

}  // namespace fe::api
