// /v1/responses request path. Included into render.cpp's anonymous namespace.
// Python: responses/protocol.py ResponsesRequest validators, responses/serving.py _create_responses up to rendering,
// renderers/online_renderer.py render_responses, responses/utils.py construct_input_messages.

// An input item after ResponsesRequest.input_item_parsing (protocol.py:621): a pydantic object or a dict.
struct RItem {
  enum Kind { Dict, FunctionCall, Reasoning, OutputMessage } kind = Dict;
  Json v;  // the item (dict form, with the defaults the models add)
};

// pydantic model checks of input_item_parsing (failures leave the dict for the union validation).
bool valid_function_call(const Json& it) {
  for (const char* k : {"arguments", "call_id", "name"})
    if (!get(it, k).is_string()) return false;
  if (!is_str(get(it, "type"), "function_call")) return false;
  for (const char* k : {"id", "namespace"})
    if (has(it, k) && !get(it, k).is_null() && !get(it, k).is_string()) return false;
  const Json& st = get(it, "status");
  if (!st.is_null() && !(is_str(st, "in_progress") || is_str(st, "completed") || is_str(st, "incomplete"))) return false;
  return true;
}

bool valid_text_list(const Json& l, const char* type) {
  if (!l.is_array()) return false;
  for (const Json& c : l.as_array())
    if (!c.is_object() || !get(c, "text").is_string() || !is_str(get(c, "type"), type)) return false;
  return true;
}

bool valid_reasoning(const Json& it) {
  if (!get(it, "id").is_string() || !is_str(get(it, "type"), "reasoning")) return false;
  if (!valid_text_list(get(it, "summary"), "summary_text")) return false;
  const Json& c = get(it, "content");
  if (!c.is_null() && !valid_text_list(c, "reasoning_text")) return false;
  const Json& e = get(it, "encrypted_content");
  if (!e.is_null() && !e.is_string()) return false;
  const Json& st = get(it, "status");
  if (!st.is_null() && !(is_str(st, "in_progress") || is_str(st, "completed") || is_str(st, "incomplete"))) return false;
  return true;
}

bool valid_output_message(const Json& it) {
  if (!get(it, "id").is_string() || !is_str(get(it, "role"), "assistant") || !is_str(get(it, "type"), "message"))
    return false;
  const Json& st = get(it, "status");
  if (!(is_str(st, "in_progress") || is_str(st, "completed") || is_str(st, "incomplete"))) return false;
  const Json& content = get(it, "content");
  if (!content.is_array()) return false;
  for (const Json& c : content.as_array()) {
    if (!c.is_object()) return false;
    if (is_str(get(c, "type"), "output_text")) {
      if (!get(c, "text").is_string() || !get(c, "annotations").is_array()) return false;
    } else if (is_str(get(c, "type"), "refusal")) {
      if (!get(c, "refusal").is_string()) return false;
    } else {
      return false;
    }
  }
  return true;
}

std::string random_hex_id() { return std::string(32, '0'); }  // ids of auto-created items never reach the prompt

// protocol.py:141 _default_input_image_details
Json default_input_image_details(const Json& value) {
  if (!value.is_object()) return value;
  const Json& content = get(value, "content");
  if (!content.is_array()) return value;
  bool changed = false;
  JsonArray nc;
  for (const Json& part : content.as_array()) {
    if (part.is_object() && is_str(get(part, "type"), "input_image") && !has(part, "detail")) {
      Json np = part;
      np.set("detail", "auto");
      nc.push_back(std::move(np));
      changed = true;
    } else {
      nc.push_back(part);
    }
  }
  if (!changed) return value;
  Json out = value;
  out.set("content", Json(std::move(nc)));
  return out;
}

// The dict members of ResponseInputOutputItem a plain item validates as. Messages and function-call outputs use the
// real TypedDict schemas (the validated dict lists declared fields first, as pydantic does); other item kinds are
// checked loosely. Returns the validated item.
Json validate_input_dict(const Json& input, const Json& it, size_t i, pyd::Errors& errs) {
  JsonArray loc{Json("input"), Json(static_cast<int64_t>(i))};
  auto fail_union = [&] {
    // pydantic reports every member of str | list[ResponseInputOutputItem] (~170-230 errors); this port reports the
    // first one (same param) and one summary error.
    if (errs.empty())
      errs.push_back(pyd::Error{"string_type", {Json("input"), Json("str")}, "Input should be a valid string", input, Json()});
    errs.push_back(pyd::Error{"union_tag_invalid", loc, "Input should be a valid response input item", it, Json()});
  };
  if (!it.is_object()) {
    fail_union();
    return it;
  }
  const Json& type = get(it, "type");
  if (has(it, "role") || is_str(type, "function_call_output")) {
    pyd::Errors local;
    pyd::Result r = pyd::validate(*pyd::schemas().responses_dict_item, it, loc, local);
    if (r.ok) return r.coerced ? std::move(r.value) : it;
  } else if (is_str(type, "custom_tool_call_output")) {
    const Json& out = get(it, "output");
    if (get(it, "call_id").is_string() && (out.is_string() || out.is_array())) return it;
  } else if (is_str(type, "custom_tool_call")) {
    if (get(it, "call_id").is_string() && get(it, "input").is_string() && get(it, "name").is_string()) return it;
  } else if (is_str(type, "item_reference")) {
    if (get(it, "id").is_string()) return it;
  }
  fail_union();
  return it;
}

// responses/utils.py:248 construct_input_messages + :311 _construct_message_from_response_item
JsonArray construct_input_messages(const Json& instructions, const Json& input, const std::vector<RItem>& items,
                                   const Json* prev_msgs, const Json* prev_out) {
  JsonArray messages;
  if (py_truthy(instructions))
    messages.push_back(Json::object({{"role", Json("system")}, {"content", instructions}}));
  if (prev_msgs && prev_msgs->is_array())
    for (const Json& m : prev_msgs->as_array())
      if (!is_str(get(m, "role"), "system")) messages.push_back(m);
  if (prev_out && prev_out->is_array()) {
    for (const Json& o : prev_out->as_array()) {
      if (!is_str(get(o, "type"), "message")) continue;  // reasoning and function_call outputs are skipped
      for (const Json& c : get(o, "content").as_array()) {
        if (!has(c, "text")) raise_py("AttributeError", "'ResponseOutputRefusal' object has no attribute 'text'");
        messages.push_back(Json::object({{"role", Json("assistant")}, {"content", get(c, "text")}}));
      }
    }
  }
  if (input.is_string()) {
    messages.push_back(Json::object({{"role", Json("user")}, {"content", input}}));
    return messages;
  }
  size_t start = messages.size();
  for (const RItem& ri : items) {
    const Json& item = ri.v;
    // prev_msg is the last message produced from THIS input list (construct_chat_messages_with_tool_call)
    Json* prev_asst = nullptr;
    if (messages.size() > start && is_str(get(messages.back(), "role"), "assistant")) prev_asst = &messages.back();
    Json tool_call;
    if (ri.kind == RItem::FunctionCall) {
      std::string name = get(item, "name").as_string();
      if (py_truthy(get(item, "namespace"))) name = get(item, "namespace").as_string() + "__" + name;
      tool_call = Json::object({{"id", get(item, "call_id")},
                                {"function", Json::object({{"name", Json(name)}, {"arguments", get(item, "arguments")}})},
                                {"type", Json("function")}});
    } else if (ri.kind == RItem::Dict && is_str(get(item, "type"), "custom_tool_call")) {
      const Json* cid = item.get("call_id");
      if (!cid) raise_py("KeyError", "'call_id'");
      const Json* nm = item.get("name");
      if (!nm) raise_py("KeyError", "'name'");
      std::string args = "{\"input\": ";
      append_py_string(args, py_truthy(get(item, "input")) ? get(item, "input").as_string() : std::string(), false);
      args += "}";
      tool_call = Json::object({{"id", *cid},
                                {"function", Json::object({{"name", *nm}, {"arguments", Json(args)}})},
                                {"type", Json("function")}});
    }
    if (!tool_call.is_null()) {
      if (prev_asst) {
        Json* tcs = prev_asst->get("tool_calls");
        if (!tcs || tcs->is_null()) {
          prev_asst->set("tool_calls", Json::array({tool_call}));
          continue;
        }
        if (tcs->is_array()) {
          tcs->push_back(tool_call);
          continue;
        }
      }
      messages.push_back(Json::object({{"role", Json("assistant")}, {"tool_calls", Json::array({tool_call})}}));
      continue;
    }
    if (ri.kind == RItem::Reasoning) {
      std::string reasoning;
      if (py_truthy(get(item, "encrypted_content"))) raise_vllm("Encrypted content is not supported.", "input");
      const Json& content = get(item, "content");
      if (content.is_array() && !content.as_array().empty()) reasoning = get(content.as_array()[0], "text").as_string();
      else if (!get(item, "summary").as_array().empty()) reasoning = get(get(item, "summary").as_array()[0], "text").as_string();
      if (prev_asst && get(*prev_asst, "reasoning").is_null()) {
        prev_asst->set("reasoning", Json(reasoning));
        continue;
      }
      messages.push_back(Json::object({{"role", Json("assistant")}, {"reasoning", Json(reasoning)}}));
      continue;
    }
    if (ri.kind == RItem::OutputMessage) {
      const JsonArray& content = get(item, "content").as_array();
      if (content.empty()) raise_py("IndexError", "list index out of range");
      if (!has(content[0], "text")) raise_py("AttributeError", "'ResponseOutputRefusal' object has no attribute 'text'");
      const Json& text = get(content[0], "text");
      if (prev_asst && get(*prev_asst, "content").is_null()) {
        prev_asst->set("content", text);
        continue;
      }
      messages.push_back(Json::object({{"role", Json("assistant")}, {"content", text}}));
      continue;
    }
    const Json& type = get(item, "type");
    if (is_str(type, "function_call_output") || is_str(type, "custom_tool_call_output")) {
      messages.push_back(Json::object(
          {{"role", Json("tool")}, {"content", get(item, "output")}, {"tool_call_id", get(item, "call_id")}}));
      continue;
    }
    if (is_str(get(item, "role"), "assistant")) {
      const Json& content = get(item, "content");
      Json text;
      if (content.is_string()) text = content;
      else if (content.is_array() && !content.as_array().empty()) text = get(content.as_array()[0], "text");
      if (!text.is_null()) {
        if (prev_asst && get(*prev_asst, "content").is_null()) {
          prev_asst->set("content", text);
          continue;
        }
        messages.push_back(Json::object({{"role", Json("assistant")}, {"content", text}}));
        continue;
      }
    }
    if (has(item, "role")) {
      messages.push_back(item);
      continue;
    }
    raise_vllm("Unsupported input item type: " + py_str(type), "input");
  }
  return messages;
}

// tool_parsers/utils.py get_function_tools + responses/utils.py construct_tool_dicts
Json responses_tool_dicts(const Json& tools) {
  if (!py_truthy(tools)) return Json();
  JsonArray out;
  auto add = [&](const Json& t, const Json& name) {
    Json f = Json::object();
    f.set("name", name);
    f.set("description", get(t, "description"));
    f.set("parameters", get(t, "parameters"));
    if (!get(t, "strict").is_null()) f.set("strict", get(t, "strict"));
    if (!get(t, "defer_loading").is_null()) f.set("defer_loading", get(t, "defer_loading"));
    out.push_back(Json::object({{"type", Json("function")}, {"function", std::move(f)}}));
  };
  for (const Json& t : tools.as_array()) {
    const Json& type = get(t, "type");
    if (is_str(type, "function")) {
      add(t, get(t, "name"));
    } else if (is_str(type, "namespace")) {
      for (const Json& nt : get(t, "tools").as_array())
        if (is_str(get(nt, "type"), "function"))
          add(nt, Json(get(t, "name").as_string() + "__" + get(nt, "name").as_string()));
    } else if (is_str(type, "custom")) {
      Json shim = Json::object();
      shim.set("description", get(t, "description"));
      shim.set("parameters",  // tool_parsers/utils.py custom_tool_parameters
               Json::object({{"type", Json("object")},
                             {"properties",
                              Json::object({{"input", Json::object({{"type", Json("string")},
                                                                    {"description",
                                                                     Json("The freeform text payload for this tool.")}})}})},
                             {"required", Json::array({Json("input")})},
                             {"additionalProperties", Json(false)}}));
      add(shim, get(t, "name"));
    }
  }
  return Json(std::move(out));
}

// responses/protocol.py:170 _resolve_named_tool_choice
Json resolve_named_tool_choice(const Json& tool_name, const Json& tools) {
  std::vector<std::string> exact;
  std::vector<std::pair<std::string, std::string>> local;  // local name -> flat name
  for (const Json& tool : py_iter(tools)) {
    const Json& type = tool.is_object() ? get(tool, "type") : Json::null_ref();
    if (is_str(type, "namespace")) {
      const Json& ns = get(tool, "name");
      const Json& nts = get(tool, "tools");
      if (!nts.is_array() || !ns.is_string()) return tool_name;
      for (const Json& nt : nts.as_array()) {
        const Json& ln = get(nt, "name");
        if (!ln.is_string()) continue;
        std::string flat = ns.as_string() + "__" + ln.as_string();
        exact.push_back(flat);
        local.emplace_back(ln.as_string(), flat);
      }
    } else {
      const Json& n = get(tool, "name");
      if (n.is_string()) exact.push_back(n.as_string());
    }
  }
  if (tool_name.is_string())
    for (const auto& e : exact)
      if (e == tool_name.as_string()) return tool_name;
  std::vector<std::string> cands;
  if (tool_name.is_string())
    for (const auto& [l, f] : local)
      if (l == tool_name.as_string()) cands.push_back(f);
  if (cands.size() == 1) return Json(cands[0]);
  if (cands.size() > 1) {
    std::string joined;
    for (size_t i = 0; i < cands.size(); ++i) joined += (i ? ", " : "") + cands[i];
    raise_vllm("Tool choice '" + py_str(tool_name) + "' is ambiguous: it is defined in more than one namespace (" +
                   joined + "). Select it by its full '<namespace>__<name>' name.",
               "tool_choice");
  }
  raise_vllm("Tool choice 'function' not found in 'tools' parameter.", "tool_choice");
}

// ResponsesRequest model validators, mode="before", reverse definition order (protocol.py:583-787)
std::vector<RItem> responses_before_validators(Json& data) {
  std::vector<RItem> items;
  if (!data.is_object()) return items;
  // check_tool_usage
  {
    const Json& tools = get(data, "tools");
    Json tool_choice = has(data, "tool_choice") ? get(data, "tool_choice") : Json("auto");
    bool has_tools = false;
    if (!tools.is_null()) {
      if (tools.is_array() || tools.is_object()) has_tools = tools.size() > 0;
      else if (tools.is_string()) has_tools = !tools.as_string().empty();
      else raise_py("TypeError", std::string("object of type '") + py_type_name(tools) + "' has no len()");
    }
    bool named = tool_choice.is_object() && (is_str(get(tool_choice, "type"), "function") ||
                                             is_str(get(tool_choice, "type"), "custom"));
    if (!has_tools) {
      if (is_str(tool_choice, "auto") || is_str(tool_choice, "none")) data.set("tool_choice", "none");
      else if (is_str(tool_choice, "required"))
        raise_vllm("Tool choice 'required' must be specified with 'tools' parameter.", "tool_choice");
      else if (named) raise_vllm("Tool choice 'function' not found in 'tools' parameter.", "tool_choice");
    } else if (named) {
      const Json& name = get(tool_choice, "name");
      if (!py_truthy(name)) raise_vllm("Tool choice 'function' not found in 'tools' parameter.", "tool_choice");
      Json resolved = resolve_named_tool_choice(name, tools);
      if (!resolved.py_equal(name)) {
        Json tc = tool_choice;
        tc.set("name", resolved);
        data.set("tool_choice", std::move(tc));
      }
    }
  }
  // check_custom_tool_format
  {
    const Json& tools = get(data, "tools");
    if (tools.is_array())
      for (const Json& t : tools.as_array()) {
        if (!t.is_object()) continue;
        const Json& fmt = get(t, "format");
        if (is_str(get(t, "type"), "custom") && fmt.is_object() && is_str(get(fmt, "type"), "grammar"))
          raise_vllm("Custom tools with a grammar format are not supported; only free-form text custom tools are "
                     "accepted.",
                     "tools");
      }
  }
  // input_item_parsing
  {
    Json* input = data.get("input");
    if (input && input->is_array()) {
      for (const Json& it0 : input->as_array()) {
        RItem ri;
        ri.v = it0;
        if (it0.is_object()) {
          const Json& type = get(it0, "type");
          if (is_str(type, "function_call")) {
            if (valid_function_call(it0)) ri.kind = RItem::FunctionCall;
          } else if (is_str(type, "reasoning")) {
            Json it = it0;
            if (!has(it, "id")) it.set("id", Json("rs_" + random_hex_id()));
            if (valid_reasoning(it)) {
              ri.kind = RItem::Reasoning;
              ri.v = std::move(it);
            }
          } else if (is_str(type, "message") && is_str(get(it0, "role"), "assistant") && get(it0, "content").is_array()) {
            Json it = it0;
            if (!has(it, "id")) it.set("id", Json("msg_" + random_hex_id()));
            if (!has(it, "status")) it.set("status", "completed");
            JsonArray nc;
            for (const Json& c : get(it0, "content").as_array()) {
              if (c.is_object() && is_str(get(c, "type"), "output_text") && !has(c, "annotations")) {
                Json cc = c;
                cc.set("annotations", Json::array());
                nc.push_back(std::move(cc));
              } else {
                nc.push_back(c);
              }
            }
            it.set("content", Json(std::move(nc)));
            if (valid_output_message(it)) {
              ri.kind = RItem::OutputMessage;
              ri.v = std::move(it);
            }
          }
        }
        items.push_back(std::move(ri));
      }
    }
  }
  // validate_prompt
  if (!get(data, "prompt").is_null()) raise_vllm("prompt template is not supported", "prompt");
  // validate_background
  if (py_truthy(get(data, "background"))) {
    const Json* st = data.get("store");
    if (st && !py_truthy(*st)) raise_vllm("background can only be used when `store` is true", "background");
  }
  // check_cache_salt_support
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
  // apply_return_mm_kwargs_alias
  if (get(data, "return_mm_kwargs").is_bool() && !get(data, "return_mm_kwargs").as_bool() && !has(data, "mm_payload"))
    data.set("mm_payload", "none");
  return items;
}

// responses/utils.py:201 should_continue_final_message
bool should_continue_final(const Json& input, const std::vector<RItem>& items) {
  if (input.is_string() || items.empty()) return false;
  const RItem& last = items.back();
  if (last.kind == RItem::OutputMessage || last.kind == RItem::Reasoning) {
    const Json& st = get(last.v, "status");
    return is_str(st, "in_progress") || is_str(st, "incomplete");
  }
  if (last.kind == RItem::FunctionCall || !last.v.is_object()) return false;
  const Json* t = last.v.get("type");
  Json type = t ? *t : Json("message");
  if (!(is_str(type, "message") || is_str(type, "reasoning"))) return false;
  const Json& st = get(last.v, "status");
  return is_str(st, "in_progress") || is_str(st, "incomplete");
}

ChatRender responses_impl(Json data, const ResponsesPrev* prev, const RenderOptions& opt) {
  std::vector<RItem> items = responses_before_validators(data);
  if (!data.is_object())
    throw ReqValidation{{pyd::Error{"model_attributes_type", {},
                                    "Input should be a valid dictionary or object to extract fields from", data, Json()}}};
  pyd::Errors errs;
  const Json* input0 = data.get("input");
  if (!input0) {
    errs.push_back(pyd::Error{"missing", {Json("input")}, "Field required", data, Json()});
  } else if (input0->is_array()) {
    for (size_t i = 0; i < items.size(); ++i) {
      items[i].v = items[i].kind == RItem::Dict ? default_input_image_details(items[i].v) : items[i].v;
      if (items[i].kind == RItem::Dict) items[i].v = validate_input_dict(*input0, items[i].v, i, errs);
    }
  } else if (!input0->is_string()) {
    errs.push_back(pyd::Error{"string_type", {Json("input"), Json("str")}, "Input should be a valid string", *input0, Json()});
    errs.push_back(pyd::Error{"list_type", {Json("input"), Json("list[...]")}, "Input should be a valid list", *input0, Json()});
  }
  Json req = validate_fields(*req_schemas().responses, data, errs);
  if (!errs.empty()) throw ReqValidation{std::move(errs)};
  if (opt.hooks && opt.hooks->after_validation) opt.hooks->after_validation();

  ChatRender out;
  out.store = has(req, "store") ? py_truthy(field(req, "store")) : true;
  // _validate_create_responses_input (serving.py:252)
  if (py_truthy(field(req, "previous_input_messages")) && py_truthy(field(req, "previous_response_id")))
    throw PyError("InvalidRequest400", "Only one of `previous_input_messages` and `previous_response_id` can be set.",
                  std::string("previous_response_id"));
  const Json& ctk = field(req, "chat_template_kwargs");
  check_chat_template(Json(), ctk);
  if (!field(req, "previous_response_id").is_null() && !prev)
    throw PyError("InvalidRequest404", "Response with id '" + py_str(field(req, "previous_response_id")) + "' not found.",
                  std::string("response_id"));
  if (field(req, "previous_response_id").is_null()) prev = nullptr;
  // OnlineRenderer.render_responses
  check_chat_template(Json(), ctk);
  Json tool_dicts = responses_tool_dicts(field(req, "tools"));
  const Json& input = field(req, "input");
  JsonArray built = construct_input_messages(field(req, "instructions"), input, items, prev ? prev->messages : nullptr,
                                             prev ? prev->output : nullptr);
  // ResponsesRequest.build_chat_params (protocol.py:420)
  bool cf = should_continue_final(input, items);
  Json effort = get(field(req, "reasoning"), "effort");
  Json extra = Json::object({{"add_generation_prompt", Json(!cf)}, {"continue_final_message", Json(cf)},
                             {"reasoning_effort", effort}});
  if (!effort.is_null() && !has(ctk, "enable_thinking")) extra.set("enable_thinking", Json(!is_str(effort, "none")));
  Json kwargs_built = merge_kwargs(ctk, extra);
  out.chat_template_kwargs = kwargs_built;
  out.tools = tool_dicts;
  out.tool_choice = has(req, "tool_choice") ? field(req, "tool_choice") : Json("auto");
  if (out.tool_choice.is_object() && (is_str(get(out.tool_choice, "type"), "function") ||
                                      is_str(get(out.tool_choice, "type"), "custom")))
    out.tool_choice = dump_model(out.tool_choice, {{"name", Json()}, {"type", Json()}});
  std::vector<Msg> msgs;
  msgs.reserve(built.size());
  for (Json& m : built) {
    Msg mm;
    mm.m = std::move(m);
    msgs.push_back(std::move(mm));
  }
  Json truncate = is_str(field(req, "truncation"), "disabled") || !has(req, "truncation") ? Json() : Json(int64_t{-1});
  preprocess_chat(out, req, msgs, kwargs_built, tool_dicts, kwargs_built, field(req, "max_output_tokens"),
                  "max_output_tokens", truncate, opt, /*keep_messages=*/true);
  return out;
}
