// The tool-call structural tag of DelegatingParser._apply_structural_tag (parser/abstract_parser.py:498-606) for
// tool_parser.structural_tag_model = "deepseek_v4_1": tool_parsers/structural_tag_registry.py:120-278 +
// xgrammar 0.2.8 builtin_structural_tag.py (normalize_tool_choice, _get_deepseek_v4_structural_tag v4_1=True with
// reasoning="disabled"), serialised as json.dumps(StructuralTag.model_dump()).
// Assumptions (production): --tool-strict-level auto, VLLM_ENFORCE_STRICT_TOOL_CALLING on (default), and the
// structured-output constraint of a tool_choice=auto request passes xgrammar validation.
#include <cstdlib>
#include <stdexcept>

#include "parser/parser.h"
#include "parser/pyjson.h"

namespace fe {

using pyjson::Value;

namespace {

const std::string kInvokeBeginPrefix = "<｜DSML｜ invoke name=\"";
const std::string kInvokeBeginSuffix = "\">\n";
const std::string kInvokeEnd = "</｜DSML｜ invoke>\n";
const std::string kToolCallsPrefix = "\n\n";
const std::string kCallsBegin = "<｜DSML｜ calls>\n";
const std::string kCallsEnd = "</｜DSML｜ calls>";
const std::string kCallsTrigger = "<｜DSML｜ calls>";

Value S(const std::string& s) { return Value::str(s); }
Value B(bool b) { return Value::boolean(b); }
Value strs(std::initializer_list<const char*> l) {
  Value a = Value::array();
  for (auto* s : l) a.push(S(s));
  return a;
}
Value obj(std::initializer_list<std::pair<const char*, Value>> l) {
  Value o = Value::object();
  for (auto& [k, v] : l) o.set(k, v);
  return o;
}

Value json_schema_fmt(const Value& schema, const char* style) {
  return obj({{"type", S("json_schema")}, {"json_schema", schema}, {"style", S(style)}, {"any_order", B(false)},
              {"max_whitespace_cnt", Value::null()}, {"excludes", Value::array()}});
}
Value const_str(const std::string& s) { return obj({{"type", S("const_string")}, {"value", S(s)}}); }
Value tag(const std::string& begin, Value content, const std::string& end) {
  return obj({{"type", S("tag")}, {"begin", S(begin)}, {"content", std::move(content)}, {"end", S(end)}});
}

struct Tool {
  std::string name;
  const Value* parameters = nullptr;  // nullptr / JSON null -> None
  std::optional<bool> strict;         // as received (None when absent / null)
  bool pinned = false;                // resolve_tool_strictness
};

Value params_of(const Tool& t) {  // xgrammar _get_function_parameters
  if (!t.pinned) return B(true);   // strict is False (pinned) -> unconstrained
  if (!t.parameters || t.parameters->is_null()) return B(true);
  return *t.parameters;
}

Value invoke_tag(const Tool& t) {
  return tag(kInvokeBeginPrefix + t.name + kInvokeBeginSuffix, json_schema_fmt(params_of(t), "deepseek_v4_1_xml"),
             kInvokeEnd);
}

void limit_to_single(Value& v) {  // limit_to_single_tool_call
  if (v.is_obj()) {
    const Value* ty = v.get("type");
    if (ty && ty->is_str() && (ty->s == "triggered_tags" || ty->s == "token_triggered_tags" || ty->s == "tags_with_separator"))
      v.set("stop_after_first", B(true));
    for (auto& x : v.items) limit_to_single(x);
  } else if (v.is_arr()) {
    for (auto& x : v.items) limit_to_single(x);
  }
}

}  // namespace

std::optional<std::string> structural_tag_for_text(std::string_view request_json) {
  Value req;
  if (!pyjson::loads(request_json, req) || !req.is_obj()) throw std::invalid_argument("request body is not a JSON object");
  const Value* tools_v = req.get("tools");
  if (!tools_v || !tools_v->is_arr() || tools_v->items.empty()) return std::nullopt;  // `not request.tools`

  enum class TC { Null, None, Auto, Required, Named } choice;
  std::string named;
  const Value* tc = req.get("tool_choice");
  if (!tc) choice = TC::Auto;  // pydantic validator: tools given, tool_choice absent -> "auto"
  else if (tc->is_null()) choice = TC::Null;
  else if (tc->is_str() && tc->s == "none") choice = TC::None;
  else if (tc->is_str() && tc->s == "auto") choice = TC::Auto;
  else if (tc->is_str() && tc->s == "required") choice = TC::Required;
  else if (tc->is_obj()) {
    const Value* f = tc->get("function");
    const Value* n = f ? f->get("name") : nullptr;
    if (!n || !n->is_str()) throw std::invalid_argument("Invalid tool_choice");
    named = n->s;
    choice = TC::Named;
  } else {
    throw std::invalid_argument("Invalid value for `tool_choice`");
  }
  if (choice != TC::Auto && choice != TC::Required && choice != TC::Named) return std::nullopt;

  // dsml_roundtrip v2.2 _api_normalize: DSML_STRICT=api (env) or "dsml_strict": "api" clears tool.function.strict
  std::string mode = "vllm";
  if (const Value* ds = req.get("dsml_strict"); ds && ds->truthy()) mode = ds->is_str() ? ds->s : std::string("<non-str>");
  else if (const char* e = std::getenv("DSML_STRICT")) mode = e;
  bool api = mode == "api";

  std::vector<Tool> tools;
  for (const Value& t : tools_v->items) {
    const Value* f = t.is_obj() ? t.get("function") : nullptr;
    const Value* n = f ? f->get("name") : nullptr;
    if (!n || !n->is_str()) throw std::invalid_argument("invalid tool: function.name");
    Tool x;
    x.name = n->s;
    x.parameters = f->get("parameters");
    if (const Value* s = f->get("strict"); s && s->is_bool() && !api) x.strict = s->b;
    tools.push_back(std::move(x));
  }
  if (choice == TC::Named) {
    bool found = false;
    for (auto& t : tools) found |= t.name == named;
    if (!found) throw std::invalid_argument("The tool specified in `tool_choice` does not match any of the specified `tools`");
  }

  // request.extract_structured_outputs(): structured_outputs with response_format overrides
  std::optional<Value> so;
  if (const Value* s = req.get("structured_outputs"); s && !s->is_null()) so = *s;
  if (const Value* rf = req.get("response_format"); rf && rf->is_obj()) {
    const Value* ty = rf->get("type");
    std::string t = ty && ty->is_str() ? ty->s : "";
    if (t != "text") {
      if (!so) so = Value::object();
      if (t == "json_object") so->set("json_object", B(true));
      else if (t == "json_schema") {
        const Value* js = rf->get("json_schema");
        const Value* sc = js ? js->get("schema") : nullptr;
        so->set("json", sc ? *sc : Value::null());
      } else {
        so->set("structural_tag", S("<response_format structural tag>"));
      }
    }
  }

  bool is_auto = choice == TC::Auto;
  bool single = false;
  if (const Value* p = req.get("parallel_tool_calls"); p && p->is_bool() && !p->b) single = true;
  int level = 0;  // ToolStrictLevel.AUTO (no --tool-strict-level); default_tool_strict_level None
  if (single && !(is_auto && so)) level = 1;  // FUNCTION

  // resolve_tool_strictness(get_structural_tag_tools(...)): all chat tools are function tools
  bool any_strict = false;
  for (auto& t : tools) any_strict |= t.strict.value_or(false);
  if (choice == TC::Auto && level == 0 && !any_strict) return std::nullopt;
  for (auto& t : tools) t.pinned = level >= 2 || t.strict.value_or(false);

  std::optional<Value> output_format;
  if (is_auto && so) {
    const Value* jo = so->get("json_object");
    const Value* js = so->get("json");
    if (jo && jo->truthy()) {
      output_format = json_schema_fmt(obj({{"type", S("object")}}), "json");
    } else if (js && !js->is_null()) {
      Value schema = *js;
      if (js->is_str() && !pyjson::loads(js->s, schema)) throw std::invalid_argument("invalid JSON schema string");
      output_format = json_schema_fmt(schema, "json");
    } else if (const Value* rx = so->get("regex"); rx && rx->is_str()) {
      output_format = obj({{"type", S("regex")}, {"pattern", *rx}});
    } else if (const Value* ch = so->get("choice"); ch && ch->is_arr()) {
      Value els = Value::array();
      for (auto& c : ch->items) els.push(obj({{"type", S("const_string")}, {"value", c}}));
      output_format = obj({{"type", S("or")}, {"elements", els}});
    } else {
      throw std::runtime_error("unsupported: structured outputs (grammar / structural_tag) combined with tools");
    }
  }

  // tool_parser.get_structural_tag(tag_request, reasoning=False, strict_level): tool_choice "required" when an
  // output format is or-ed in
  TC sel = output_format ? TC::Required : choice;
  Value suffix;
  if (sel == TC::Auto) {
    Value tags = Value::array();
    for (auto& t : tools) tags.push(invoke_tag(t));
    Value tws = obj({{"type", S("tags_with_separator")}, {"tags", tags}, {"separator", S("")},
                     {"at_least_one", B(true)}, {"stop_after_first", B(false)}});
    Value outer = Value::array();
    outer.push(tag(kCallsBegin, tws, kCallsEnd));
    Value trig = Value::array();
    trig.push(S(kCallsTrigger));
    suffix = obj({{"type", S("triggered_tags")}, {"triggers", trig}, {"tags", outer}, {"at_least_one", B(false)},
                  {"stop_after_first", B(false)}, {"excludes", strs({"<think>", "</think>"})}});
  } else if (sel == TC::Named) {
    std::vector<const Tool*> m;
    for (auto& t : tools) if (t.name == named) m.push_back(&t);
    if (m.size() != 1) throw std::invalid_argument("Forced tool choice must resolve to exactly one tool.");
    Value els = Value::array();
    els.push(const_str(kToolCallsPrefix + kCallsBegin));
    els.push(invoke_tag(*m[0]));
    els.push(const_str(kCallsEnd));
    suffix = obj({{"type", S("sequence")}, {"elements", els}});
  } else {
    Value tags = Value::array();
    for (auto& t : tools) tags.push(invoke_tag(t));
    Value els = Value::array();
    els.push(const_str(kToolCallsPrefix + kCallsBegin));
    els.push(obj({{"type", S("tags_with_separator")}, {"tags", tags}, {"separator", S("")}, {"at_least_one", B(true)},
                  {"stop_after_first", B(false)}}));
    els.push(const_str(kCallsEnd));
    suffix = obj({{"type", S("sequence")}, {"elements", els}});
  }
  if (single) limit_to_single(suffix);
  Value format = suffix;
  if (output_format) {
    Value els = Value::array();
    els.push(std::move(suffix));
    els.push(std::move(*output_format));
    format = obj({{"type", S("or")}, {"elements", els}});
  }
  Value st = obj({{"type", S("structural_tag")}, {"format", std::move(format)}});
  return pyjson::dumps(st, true);
}

}  // namespace fe
