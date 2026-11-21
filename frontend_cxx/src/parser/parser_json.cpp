// fe::Json convenience entry points (common/json.h): request -> ParserOptions, request -> structural tag.
#include "common/json.h"
#include "parser/parser.h"

namespace fe {

std::optional<std::string> structural_tag_for(const Json& request) {
  Json small = Json::object();
  for (const char* k : {"tools", "tool_choice", "parallel_tool_calls", "response_format", "structured_outputs", "dsml_strict"})
    if (const Json* v = request.get(k)) small.set(k, *v);
  return structural_tag_for_text(dumps_py(small, false));
}

ParserOptions parser_options_for(const Json& request, const Json& chat_template_kwargs) {
  ParserOptions o;
  o.thinking_initial = parser_thinking(chat_template_kwargs.is_object() ? dumps_py(chat_template_kwargs, false) : "{}");
  if (const Json* ir = request.get("include_reasoning"); ir && ir->is_bool()) o.include_reasoning = ir->as_bool();
  const Json* tools = request.get("tools");
  o.has_tools = tools && tools->is_array() && tools->size() > 0;
  const Json* tc = request.get("tool_choice");
  if (!tc) o.tool_choice = o.has_tools ? ToolChoice::Auto : ToolChoice::None;
  else if (tc->is_null()) o.tool_choice = ToolChoice::Null;
  else if (tc->is_string()) {
    std::string_view s = tc->as_str();
    o.tool_choice = s == "none" ? ToolChoice::None : s == "auto" ? ToolChoice::Auto : ToolChoice::Required;
  } else if (tc->is_object()) {
    o.tool_choice = ToolChoice::Named;
    const Json& n = (*tc)["function"]["name"];
    if (n.is_string()) o.named_function = n.as_string();
  }
  if (const Json* p = request.get("parallel_tool_calls"); p && p->is_bool() && !p->as_bool()) o.parallel_tool_calls = false;
  // dsml_roundtrip v2.3: getattr(request.stream_options, "continuous_usage_stats", False)
  if (const Json* so = request.get("stream_options"); so && so->is_object())
    o.continuous_usage_stats = so->get_bool("continuous_usage_stats").value_or(false);
  return o;
}

}  // namespace fe
