// DeepSeek V4.1 prompt encoder, port of vllm/tokenizers/deepseek_v41_encoding.py. Owner C. See encoding.h.
#include "render/encoding.h"

#include <algorithm>
#include <string_view>
#include <utility>
#include <vector>

#include "render/pyval.h"

namespace fe::render {

#define FW "\xef\xbd\x9c"  // U+FF5C FULLWIDTH VERTICAL LINE

namespace {

constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kUser = "<" FW "User" FW ">";
constexpr std::string_view kAssistant = "<" FW "Assistant" FW ">";
constexpr std::string_view kSystem = "<" FW "System" FW ">";
constexpr std::string_view kLatestReminder = "<" FW "latest_reminder" FW ">";
constexpr std::string_view kDsml = FW "DSML" FW;

constexpr std::pair<std::string_view, std::string_view> kTasks[] = {
    {"action", "<" FW "action" FW ">"}, {"query", "<" FW "query" FW ">"},   {"authority", "<" FW "authority" FW ">"},
    {"domain", "<" FW "domain" FW ">"}, {"title", "<" FW "title" FW ">"},   {"read_url", "<" FW "read_url" FW ">"},
};

constexpr std::pair<std::string_view, int> kEffort[] = {{"low", 50}, {"high", 75}, {"xhigh", 75}, {"max", 100}};

constexpr std::string_view kToolsPre =
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" FW
    "DSML" FW " calls>\" block like the following:\n"
    "\n"
    "<" FW "DSML" FW " calls>\n"
    "<" FW "DSML" FW " invoke name=\"$TOOL_NAME\">\n"
    "<" FW "DSML" FW " parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" FW "DSML" FW
    " parameter>\n"
    "...\n"
    "</" FW "DSML" FW " invoke>\n"
    "<" FW "DSML" FW " invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</" FW "DSML" FW " invoke>\n"
    "</" FW "DSML" FW " calls>\n"
    "\n"
    "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, "
    "arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n"
    "\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
    "<think>...</think> BEFORE any tool calls or final response.\n"
    "\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n";
constexpr std::string_view kToolsPost =
    "\n"
    "\n"
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

constexpr std::string_view kResponseFormatPre =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

// ------------------------------------------------------------------ Python dict / list helpers
[[noreturn]] void attr_error(const Json& v, const char* attr) {
  raise_py("AttributeError", std::string("'") + py_type_name(v) + "' object has no attribute '" + attr + "'");
}

// d.get(k): None when absent. AttributeError when d is not a dict.
const Json& dget(const Json& d, std::string_view k) {
  if (!d.is_object()) attr_error(d, "get");
  const Json* p = d.get(k);
  return p ? *p : Json::null_ref();
}
// d.get(k, default)
Json dget_or(const Json& d, std::string_view k, Json def) {
  if (!d.is_object()) attr_error(d, "get");
  const Json* p = d.get(k);
  return p ? *p : std::move(def);
}
// d[k]
const Json& dindex(const Json& d, std::string_view k) {
  if (d.is_object()) {
    const Json* p = d.get(k);
    if (!p) raise_py("KeyError", py_repr_str(k));
    return *p;
  }
  if (d.is_array()) raise_py("TypeError", "list indices must be integers or slices, not str");
  if (d.is_string()) raise_py("TypeError", "string indices must be integers, not 'str'");
  raise_py("TypeError", std::string("'") + py_type_name(d) + "' object is not subscriptable");
}

bool is_str(const Json& v, std::string_view s) { return v.is_string() && v.as_string() == s; }

// iter(x) for a JSON value: list items, dict keys, str chars.
JsonArray py_iter(const Json& v) {
  if (v.is_array()) return v.as_array();
  JsonArray out;
  if (v.is_object()) {
    for (const auto& [k, x] : v.as_object()) out.emplace_back(k);
    return out;
  }
  if (v.is_string()) {
    const std::string& s = v.as_string();
    size_t i = 0;
    while (i < s.size()) {
      size_t j = i + 1;
      while (j < s.size() && (static_cast<unsigned char>(s[j]) & 0xC0) == 0x80) ++j;
      out.emplace_back(std::string_view(s).substr(i, j - i));
      i = j;
    }
    return out;
  }
  raise_py("TypeError", std::string("'") + py_type_name(v) + "' object is not iterable");
}

// iter(x) without copying a list
struct Iter {
  const JsonArray* ref = nullptr;
  JsonArray own;
  const JsonArray& items() const { return ref ? *ref : own; }
};
Iter py_iter_ref(const Json& v) {
  Iter it;
  if (v.is_array()) it.ref = &v.as_array();
  else it.own = py_iter(v);
  return it;
}

void list_append(Json& lst, Json v) {
  if (!lst.is_array()) attr_error(lst, "append");
  lst.as_array().push_back(std::move(v));
}

void list_extend(Json& lst, const Json& it) {
  if (!lst.is_array()) attr_error(lst, "extend");
  if (&lst == &it) {  // a.extend(a)
    JsonArray copy = it.as_array();
    for (auto& x : copy) lst.as_array().push_back(std::move(x));
    return;
  }
  for (auto& x : py_iter(it)) lst.as_array().push_back(std::move(x));
}

// "sep".join(items): TypeError for non-str items.
std::string str_join(std::string_view sep, const JsonArray& items) {
  std::string out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (!items[i].is_string())
      raise_py("TypeError", "sequence item " + std::to_string(i) + ": expected str instance, " +
                                py_type_name(items[i]) + " found");
  }
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out += sep;
    out += items[i].as_string();
  }
  return out;
}

// s += x (Python str concatenation): TypeError unless x is a str.
void str_concat(std::string& s, const Json& x) {
  if (!x.is_string())
    raise_py("TypeError", std::string("can only concatenate str (not \"") + py_type_name(x) + "\") to str");
  s += x.as_string();
}

// `x or ""` as a Json
const Json& or_empty(const Json& x) {
  static const Json empty("");
  return py_truthy(x) ? x : empty;
}

const JsonDumpOptions kToJson = [] {
  JsonDumpOptions o;
  o.ensure_ascii = false;
  return o;
}();
std::string to_json(const Json& v) { return dumps_py(v, false); }

// Python dict with hashable scalar keys and `==` key matching (1 == 1.0 == True).
struct PyDict {
  std::vector<std::pair<Json, Json>> items;
  void set(const Json& k, Json v) {
    if (!py_hashable(k)) raise_py("TypeError", std::string("unhashable type: '") + py_type_name(k) + "'");
    for (auto& [kk, vv] : items)
      if (kk.py_equal(k)) { vv = std::move(v); return; }
    items.emplace_back(k, std::move(v));
  }
  const Json* get(const Json& k) const {
    if (!py_hashable(k)) raise_py("TypeError", std::string("unhashable type: '") + py_type_name(k) + "'");
    for (auto& [kk, vv] : items)
      if (kk.py_equal(k)) return &vv;
    return nullptr;
  }
  bool empty() const { return items.empty(); }
};

// ------------------------------------------------------------------ encoder functions
JsonArray merge_tool_messages(JsonArray&& messages) {
  JsonArray merged;
  merged.reserve(messages.size());
  for (Json& m0 : messages) {
    Json msg = std::move(m0);  // copy.deepcopy(msg): the caller's list is not used afterwards
    const Json& role = dget(msg, "role");
    if (is_str(role, "tool")) {
      Json block = Json::object();
      block.set("type", "tool_result");
      block.set("tool_use_id", dget_or(msg, "tool_call_id", Json("")));
      Json* c = msg.get("content");
      block.set("content", c ? std::move(*c) : Json(""));  // the tool message itself is dropped
      if (!merged.empty() && is_str(dget(merged.back(), "role"), "user") && merged.back().contains("content_blocks")) {
        list_append(*merged.back().get("content_blocks"), std::move(block));
      } else {
        Json nm = Json::object();
        nm.set("role", "user");
        nm.set("content_blocks", Json::array({std::move(block)}));
        merged.push_back(std::move(nm));
      }
    } else if (is_str(role, "user")) {
      Json content_blocks = dget(msg, "content_blocks");
      if (content_blocks.is_null()) {
        Json t = Json::object();
        t.set("type", "text");
        // the message's own "content" is never read again (content_blocks is a non-empty list): move, not copy
        Json* c = msg.get("content");
        t.set("text", c ? std::move(*c) : Json(""));
        content_blocks = Json::array({std::move(t)});
      }
      if (!merged.empty() && is_str(dget(merged.back(), "role"), "user") && merged.back().contains("content_blocks") &&
          dget(merged.back(), "task").is_null()) {
        list_extend(*merged.back().get("content_blocks"), content_blocks);
      } else {
        msg.set("content_blocks", std::move(content_blocks));
        merged.push_back(std::move(msg));
      }
    } else {
      merged.push_back(std::move(msg));
    }
  }
  return merged;
}

void sort_tool_results_by_call_order(JsonArray& messages) {
  PyDict order;
  for (Json& msg : messages) {
    const Json& role = dget(msg, "role");
    if (is_str(role, "assistant") && py_truthy(dget(msg, "tool_calls"))) {
      order = PyDict{};
      Iter tci = py_iter_ref(dindex(msg, "tool_calls"));
      const JsonArray& tcs = tci.items();
      for (size_t idx = 0; idx < tcs.size(); ++idx) {
        const Json& tc = tcs[idx];
        Json tc_id = dget(tc, "id");
        if (!py_truthy(tc_id)) {
          Json fn = dget_or(tc, "function", Json::object());
          tc_id = dget_or(fn, "id", Json(""));
        }
        if (py_truthy(tc_id)) order.set(tc_id, Json(static_cast<int64_t>(idx)));
      }
    } else if (is_str(role, "user") && py_truthy(dget(msg, "content_blocks"))) {
      Json& blocks_json = *msg.get("content_blocks");
      Iter bi = py_iter_ref(blocks_json);
      const JsonArray& blocks = bi.items();
      std::vector<const Json*> tool_blocks;
      for (const Json& b : blocks)
        if (is_str(dget(b, "type"), "tool_result")) tool_blocks.push_back(&b);
      if (tool_blocks.size() > 1 && !order.empty()) {
        std::vector<std::pair<int64_t, const Json*>> keyed;
        for (const Json* b : tool_blocks) {
          Json id = dget_or(*b, "tool_use_id", Json(""));
          const Json* o = order.get(id);
          keyed.emplace_back(o ? o->as_int() : 0, b);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        JsonArray nb;
        nb.reserve(blocks.size());
        size_t si = 0;
        for (const Json& b : blocks) {
          if (is_str(dget(b, "type"), "tool_result")) nb.push_back(*keyed[si++].second);
          else nb.push_back(b);
        }
        blocks_json = Json(std::move(nb));
      }
    }
  }
}

int find_last_user_index(const JsonArray& messages) {
  for (int idx = static_cast<int>(messages.size()) - 1; idx >= 0; --idx) {
    const Json& role = dget(messages[idx], "role");
    if (is_str(role, "user") || is_str(role, "developer") || (is_str(role, "system") && idx > 0)) return idx;
  }
  return -1;
}

std::string render_reasoning_effort(size_t index, bool thinking, const Json& effort_in) {
  Json effort = effort_in.is_null() ? Json("high") : effort_in;
  bool ok = false;
  int budget = 0;
  if (effort.is_int() && effort.as_int() >= 1 && effort.as_int() <= 100) {
    ok = true;
    budget = static_cast<int>(effort.as_int());
  } else {
    if (!py_hashable(effort)) raise_py("TypeError", std::string("unhashable type: '") + py_type_name(effort) + "'");
    if (effort.is_string())
      for (auto& [k, b] : kEffort)
        if (effort.as_string() == k) { ok = true; budget = b; }
  }
  if (!ok)
    raise_py("AssertionError", "Invalid reasoning effort for deepseek_v41: " + py_str(effort) +
                                   ", should be int within [1,100] or ['low', 'high', 'xhigh', 'max']");
  if (index == 0 && thinking)
    return "Reasoning Effort: " + std::to_string(budget) +
           " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
  return "";
}

std::string encode_arguments_to_dsml(const Json& orig) {
  const Json* argp = &orig;
  Json arguments;
  if (!orig.is_object()) {
    arguments = orig;
    for (int k = 0; k < 2; ++k) {
      if (!arguments.is_string()) break;
      try {
        arguments = json_loads_str(arguments.as_string());
      } catch (const JsonError&) {
        break;
      }
    }
    if (!arguments.is_object()) {
      Json w = Json::object();
      w.set("arguments", orig);
      arguments = std::move(w);
    }
    argp = &arguments;
  }
  std::string out;
  bool first = true;
  for (const auto& [k, v] : argp->as_object()) {
    if (!first) out += '\n';
    first = false;
    out += "<";
    out += kDsml;
    out += " parameter name=\"";
    out += k;
    out += "\" string=\"";
    out += v.is_string() ? "true" : "false";
    out += "\">";
    if (v.is_string()) out += v.as_string();
    else out += to_json(v);
    out += "</";
    out += kDsml;
    out += " parameter>";
  }
  return out;
}

void render_message(std::string& prompt, size_t index, const JsonArray& messages, bool thinking, bool drop_thinking,
                    const Json& reasoning_effort, int last_user_idx) {
  const Json& msg = messages[index];
  const Json& role = dget(msg, "role");
  const Json& content = dget(msg, "content");
  const Json& tools_v = dget(msg, "tools");
  const Json& response_format = dget(msg, "response_format");
  const Json& tool_calls_v = dget(msg, "tool_calls");
  const Json& reasoning_content = dget(msg, "reasoning_content");
  Json wo_eos = dget_or(msg, "wo_eos", Json(false));

  std::vector<const Json*> tools;  // tools_from_openai_format
  Iter ti;
  if (py_truthy(tools_v)) {
    ti = py_iter_ref(tools_v);
    for (const Json& t : ti.items()) tools.push_back(&dindex(t, "function"));
  }
  std::vector<std::pair<const Json*, const Json*>> tool_calls;  // tool_calls_from_openai_format: (name, arguments)
  Iter tci;
  if (py_truthy(tool_calls_v)) {
    tci = py_iter_ref(tool_calls_v);
    for (const Json& tc : tci.items()) {
      const Json& fn = dindex(tc, "function");
      tool_calls.emplace_back(&dindex(fn, "name"), &dindex(fn, "arguments"));
    }
  }

  std::string effort_prompt = render_reasoning_effort(index, thinking, reasoning_effort);
  if (index == 0 && (!effort_prompt.empty() || is_str(role, "system"))) prompt += kSystem;
  prompt += effort_prompt;

  if (is_str(role, "system")) {
    if (index > 0) prompt += kSystem;
    py_str_append(prompt, or_empty(content));
    if (!tools.empty()) {
      prompt += "\n\n";
      prompt += render_tools_ptrs(tools);
    }
    if (py_truthy(response_format)) {
      prompt += "\n\n";
      prompt += kResponseFormatPre;
      prompt += to_json(response_format);
    }
  } else if (is_str(role, "developer")) {
    if (!py_truthy(content)) {
      raise_py("AssertionError", "Invalid message for role `developer`: " + py_repr(msg));
    }
    std::string cd(kUser);
    str_concat(cd, content);
    if (!tools.empty()) {
      cd += "\n\n";
      cd += render_tools_ptrs(tools);
    }
    if (py_truthy(response_format)) {
      cd += "\n\n";
      cd += kResponseFormatPre;
      cd += to_json(response_format);
    }
    prompt += cd;
  } else if (is_str(role, "user")) {
    prompt += kUser;
    const Json& content_blocks = dget(msg, "content_blocks");
    if (py_truthy(content_blocks)) {
      // parts as references (a part is either a value from the message or an owned string); joined after all
      // parts are built, like "\n\n".join(parts)
      struct Part {
        const Json* j = nullptr;
        std::string own;
      };
      static const Json empty("");
      std::vector<Part> parts;
      Iter cbi = py_iter_ref(content_blocks);
      for (const Json& block : cbi.items()) {
        const Json& bt = dget(block, "type");
        Part p;
        if (is_str(bt, "text")) {
          const Json* t = block.get("text");
          p.j = t ? t : &empty;
        } else if (is_str(bt, "tool_result")) {
          const Json* tc = block.get("content");
          const Json& tool_content = tc ? *tc : empty;
          p.own = "<tool_result>";
          if (tool_content.is_array()) {
            JsonArray text_parts;
            for (const Json& b : tool_content.as_array()) {
              if (is_str(dget(b, "type"), "text")) text_parts.push_back(dget_or(b, "text", Json("")));
              else text_parts.push_back(Json("[Unsupported " + py_str(dget(b, "type")) + "]"));
            }
            p.own += str_join("\n\n", text_parts);
          } else {
            py_str_append(p.own, tool_content);
          }
          p.own += "</tool_result>";
        } else {
          p.own = "[Unsupported " + py_str(bt) + "]";
        }
        parts.push_back(std::move(p));
      }
      for (size_t i = 0; i < parts.size(); ++i)
        if (parts[i].j && !parts[i].j->is_string())
          raise_py("TypeError", "sequence item " + std::to_string(i) + ": expected str instance, " +
                                    py_type_name(*parts[i].j) + " found");
      for (size_t i = 0; i < parts.size(); ++i) {
        if (i) prompt += "\n\n";
        prompt += parts[i].j ? parts[i].j->as_string() : parts[i].own;
      }
    } else {
      str_concat(prompt, or_empty(content));
    }
  } else if (is_str(role, "latest_reminder")) {
    prompt += kLatestReminder;
    py_str_append(prompt, content);
  } else if (is_str(role, "tool")) {
    raise_py("NotImplementedError",
             "deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()");
  } else if (is_str(role, "assistant")) {
    std::string tc_content;
    if (!tool_calls.empty()) {
      std::string tcl;
      bool first = true;
      for (const auto& [name, args] : tool_calls) {
        if (!first) tcl += '\n';
        first = false;
        tcl += "<";
        tcl += kDsml;
        tcl += " invoke name=\"";
        py_str_append(tcl, *name);
        tcl += "\">\n";
        tcl += encode_arguments_to_dsml(*args);
        tcl += "\n</";
        tcl += kDsml;
        tcl += " invoke>";
      }
      tc_content += "\n\n<";
      tc_content += kDsml;
      tc_content += " calls>\n";
      tc_content += tcl;
      tc_content += "\n</";
      tc_content += kDsml;
      tc_content += " calls>";
    }
    const Json& summary = or_empty(content);
    const Json& rc = or_empty(reasoning_content);
    bool prev_has_task = index >= 1 && !dget(messages[index - 1], "task").is_null();
    if (thinking && !prev_has_task) {
      if (!drop_thinking || static_cast<int>(index) > last_user_idx) {
        py_str_append(prompt, rc);
        prompt += kThinkEnd;
      }
    }
    py_str_append(prompt, summary);
    prompt += tc_content;
    if (!py_truthy(wo_eos)) prompt += kEos;
  } else {
    raise_py("NotImplementedError", "Unknown role: " + py_str(role));
  }

  // transition tokens
  if (index + 1 < messages.size()) {
    const Json& nr = dget(messages[index + 1], "role");
    if (!(is_str(nr, "assistant") || is_str(nr, "latest_reminder"))) return;
  }
  const Json& task = dget(messages[index], "task");
  if (!task.is_null()) {
    if (!py_hashable(task)) raise_py("TypeError", std::string("unhashable type: '") + py_type_name(task) + "'");
    std::string_view tok;
    for (auto& [name, t] : kTasks)
      if (is_str(task, name)) tok = t;
    if (tok.empty())
      raise_py("AssertionError", "Invalid task: '" + py_str(task) +
                                     "'. Valid tasks are: ['action', 'query', 'authority', 'domain', 'title', "
                                     "'read_url']");
    if (!is_str(task, "action")) {
      prompt += tok;
    } else {
      prompt += kAssistant;
      prompt += thinking ? kThinkStart : kThinkEnd;
      prompt += tok;
    }
  } else if (is_str(role, "user") || is_str(role, "developer") || (is_str(role, "system") && index > 0)) {
    prompt += kAssistant;
    if (!drop_thinking && thinking) prompt += kThinkStart;
    else if (drop_thinking && thinking && static_cast<int>(index) >= last_user_idx) prompt += kThinkStart;
    else prompt += kThinkEnd;
  }
}

JsonArray drop_thinking_messages(JsonArray&& messages) {
  int last_user_idx = find_last_user_index(messages);
  JsonArray result;
  result.reserve(messages.size());
  for (size_t idx = 0; idx < messages.size(); ++idx) {
    Json& msg = messages[idx];
    const Json& role = dget(msg, "role");
    if (!py_hashable(role)) raise_py("TypeError", std::string("unhashable type: '") + py_type_name(role) + "'");
    bool keep = is_str(role, "user") || is_str(role, "system") || is_str(role, "tool") ||
                is_str(role, "latest_reminder") || is_str(role, "direct_search_results");
    if (keep || static_cast<int>(idx) >= last_user_idx) {
      result.push_back(std::move(msg));
    } else if (is_str(role, "assistant")) {
      msg.erase("reasoning_content");  // copy.copy(msg).pop(...): the caller's list is not used afterwards
      result.push_back(std::move(msg));
    }
  }
  return result;
}

}  // namespace

std::string render_tools_ptrs(const std::vector<const Json*>& functions) {
  std::string out(kToolsPre);
  bool first = true;
  for (const Json* t : functions) {
    if (!first) out += '\n';
    first = false;
    dump_append(out, *t, kToJson);
  }
  out += kToolsPost;
  return out;
}

std::string render_tools(const Json& functions) {
  std::vector<const Json*> p;
  Iter it = py_iter_ref(functions);
  for (const Json& t : it.items()) p.push_back(&t);
  return render_tools_ptrs(p);
}

std::string encode_messages(JsonArray messages, bool thinking, bool drop_thinking, const Json& reasoning_effort) {
  JsonArray merged = merge_tool_messages(std::move(messages));
  messages.clear();
  sort_tool_results_by_call_order(merged);
  std::string prompt(kBos);
  bool effective_drop = drop_thinking;
  for (const Json& m : merged)
    if (py_truthy(dget(m, "tools"))) { effective_drop = false; break; }
  JsonArray full = (thinking && effective_drop) ? drop_thinking_messages(std::move(merged)) : std::move(merged);
  int last_user_idx = find_last_user_index(full);
  size_t total = 0;
  for (const Json& m : full)
    for (const char* k : {"content", "reasoning_content"})
      if (const Json* c = m.get(k); c && c->is_string()) total += c->as_string().size();
  prompt.reserve(total + total / 8 + 4096);
  for (size_t idx = 0; idx < full.size(); ++idx)
    render_message(prompt, idx, full, thinking, effective_drop, reasoning_effort, last_user_idx);
  return prompt;
}

}  // namespace fe::render
