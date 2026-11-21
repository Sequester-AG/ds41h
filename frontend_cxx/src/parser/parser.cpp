// fe::Parser = vLLM's DelegatingParser (parser/abstract_parser.py:339-1052, with vllm-0.32.0-reasoning-count.patch)
// composed of the DeepSeekV41 reasoning and tool adapters, plus the parser-side rules of
// entrypoints/openai/chat_completion/serving.py (finish_reason, message construction, parallel_tool_calls filter).
#include "parser/parser.h"

#include <random>
#include <stdexcept>

#include "parser/dsml_engine.h"
#include "parser/pyjson.h"

namespace fe {

using dsml::Delta;
using dsml::ReqView;

void finalize_parser_vocab(ParserVocab& v) { v.internal = std::make_shared<const dsml::Shape>(v); }

std::string random_tool_call_id() {
  // vllm.utils.random_uuid(): f"{uuid.uuid4().int & MASK_64_BITS:016x}" -- the low 64 bits of a uuid4, whose top
  // two bits are the RFC 4122 variant "10".
  static thread_local std::mt19937_64 rng{std::random_device{}() ^ ((uint64_t)std::random_device{}() << 32)};
  uint64_t x = rng();
  x = (x & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;
  static const char* H = "0123456789abcdef";
  std::string s = "chatcmpl-tool-";
  for (int i = 15; i >= 0; --i) s += H[(x >> (i * 4)) & 15];
  return s;
}

// ------------------------------------------------------------------------------------------------ delta serialisation
// pydantic-core JSON string: '"' and '\\' escaped, \b \f \n \r \t short escapes, other C0 controls as \u00xx, all
// other characters (DEL, non-ASCII) raw.
static void pyd_str(std::string& o, std::string_view s) {
  static const char* H = "0123456789abcdef";
  o += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { o += "\\u00"; o += H[c >> 4]; o += H[c & 15]; }
        else o += (char)c;
    }
  }
  o += '"';
}

std::string ParserDelta::to_json(bool parallel_tool_calls) const {
  std::string o = "{";
  bool first = true;
  auto key = [&](const char* k) {
    if (!first) o += ',';
    first = false;
    o += '"'; o += k; o += "\":";
  };
  if (content_set || content) {
    key("content");
    if (content) pyd_str(o, *content); else o += "null";
  }
  if (reasoning_set || reasoning) {
    key("reasoning");
    if (reasoning) pyd_str(o, *reasoning); else o += "null";
  }
  std::vector<const ToolCallDelta*> tcs;
  for (auto& t : tool_calls)
    if (parallel_tool_calls || t.index == 0) tcs.push_back(&t);
  if (!tcs.empty()) {
    key("tool_calls");
    o += '[';
    for (size_t i = 0; i < tcs.size(); ++i) {
      const ToolCallDelta& t = *tcs[i];
      if (i) o += ',';
      o += '{';
      if (t.id) { o += "\"id\":"; pyd_str(o, *t.id); o += ",\"type\":\"function\","; }
      o += "\"index\":" + std::to_string(t.index) + ",\"function\":{";
      if (t.name) { o += "\"name\":"; pyd_str(o, *t.name); }
      if (t.arguments) { if (t.name) o += ','; o += "\"arguments\":"; pyd_str(o, *t.arguments); }
      o += "}}";
    }
    o += ']';
  }
  o += '}';
  return o;
}

// ------------------------------------------------------------------------------------------------ Parser
struct Parser::Impl {
  std::shared_ptr<const ParserVocab> v;
  std::shared_ptr<const dsml::Shape> sh;
  ParserOptions o;
  std::function<std::string()> idgen;
  ReqView req;
  dsml::ReasoningAdapter rp;
  dsml::ToolAdapter tp;
  // StreamState
  bool reasoning_ended = false, prompt_ended_reasoning = false, stream_finished = false;
  bool tool_call_text_started = false, prompt_reasoning_checked = false;
  int history_tool_call_cnt = 0;
  bool have_prompt = false;
  std::vector<int32_t> prompt;
  std::vector<int32_t> generated;

  static std::shared_ptr<const dsml::Shape> shape_of(const std::shared_ptr<const ParserVocab>& v) {
    if (!v) throw std::invalid_argument("fe::Parser: null vocab");
    if (v->internal) return std::static_pointer_cast<const dsml::Shape>(v->internal);
    return std::make_shared<const dsml::Shape>(*v);
  }
  Impl(std::shared_ptr<const ParserVocab> vv, ParserOptions oo)
      : v(std::move(vv)), sh(shape_of(v)), o(std::move(oo)),
        idgen(o.make_tool_call_id ? o.make_tool_call_id : std::function<std::string()>(random_tool_call_id)),
        req{o.has_tools, o.tool_choice}, rp(sh, v, o.thinking_initial, &idgen), tp(sh, v, o.thinking_initial, &idgen) {}

  // record_tool_parser_invocation (DelegatingParser.extract_tool_calls[_streaming]'s finally:)
  void record_invocation(bool streaming, bool tool_called) {
    if (o.on_tool_parser_invocation) o.on_tool_parser_invocation(streaming, tool_called);
  }

  // DelegatingParser.extract_tool_calls_streaming: is_tool_called = bool(result and result.tool_calls)
  Delta tool_adapter_streaming(std::string_view dtext, std::span<const int32_t> dids) {
    Delta d;
    try {
      d = tp.extract_tool_calls_streaming(dtext, dids, req);
    } catch (...) {
      record_invocation(true, false);
      throw;
    }
    record_invocation(true, d && !d->tool_calls.empty());
    return d;
  }

  // DelegatingParser.extract_tool_calls: is_tool_called = bool(result.tools_called)
  dsml::ExtractedInfo tool_adapter_full(std::string_view content) {
    dsml::ExtractedInfo info;
    try {
      info = tp.extract_tool_calls(content, req);
    } catch (...) {
      record_invocation(false, false);
      throw;
    }
    record_invocation(false, info.tools_called);
    return info;
  }

  Delta extract_tool_calls_streaming(std::string_view dtext, std::span<const int32_t> dids) {
    if (o.tool_choice == ToolChoice::None) {
      Delta d = tool_adapter_streaming(dtext, dids);
      if (d) d->tool_calls.clear();  // delta_message.tool_calls = []
      return d;
    }
    return tool_adapter_streaming(dtext, dids);  // supports_required_and_named=False: as auto
  }

  Delta flush_engine_parsers(Delta dm) {
    bool ended = reasoning_ended;
    for (int k = 0; k < 2; ++k) {
      if (k == 0 && ended) continue;
      Delta fd = k == 0 ? rp.finish_streaming() : tp.finish_streaming();
      if (!fd) continue;
      if (!dm) { dm = std::move(fd); continue; }
      if (fd->content && !fd->content->empty()) dm->set_content(dm->content.value_or("") + *fd->content);
      if (fd->reasoning && !fd->reasoning->empty()) dm->set_reasoning(dm->reasoning.value_or("") + *fd->reasoning);
      for (auto& t : fd->tool_calls) dm->tool_calls.push_back(std::move(t));
    }
    return dm;
  }

  Delta parse_delta(std::string_view delta_text, std::span<const int32_t> delta_ids, bool finished) {
    stream_finished = finished;
    if (!prompt_reasoning_checked && have_prompt) {
      prompt_reasoning_checked = true;
      if (rp.is_reasoning_end(prompt)) {
        reasoning_ended = true;
        prompt_ended_reasoning = true;
      }
      // else: adjust_initial_state_from_prompt is a no-op for ParserEngine
    }
    std::string current_text(delta_text);
    std::vector<int32_t> current_ids(delta_ids.begin(), delta_ids.end());
    std::string dtext(delta_text);
    std::vector<int32_t> dids(current_ids);
    Delta dm;
    if (!reasoning_ended) {
      dm = rp.extract_reasoning_streaming(dtext, dids);
      if (rp.has_engine_confirmed_reasoning_end()) {
        reasoning_ended = true;
        current_ids = rp.extract_content_ids(dids);
        Delta flush = rp.finish_streaming();
        current_text = (dm && dm->content ? *dm->content : std::string()) +
                       (flush && flush->content ? *flush->content : std::string());
        if (dm) dm->set_content(std::nullopt);
      }
    }
    if (reasoning_ended) {
      if (!tool_call_text_started) {
        tool_call_text_started = true;
        dtext = current_text;
        dids = current_ids;
      }
      std::optional<std::string> r_batch = dm ? dm->reasoning : std::nullopt;
      dm = extract_tool_calls_streaming(dtext, dids);
      if (r_batch && !r_batch->empty()) {
        if (!dm) {
          ParserDelta x;
          x.present = true;
          x.set_reasoning(r_batch);
          dm = std::move(x);
        } else if (!dm->reasoning || dm->reasoning->empty()) {
          dm->set_reasoning(r_batch);
        }
      }
      if (dm)
        for (auto& t : dm->tool_calls) history_tool_call_cnt += t.id.has_value();
    }
    if (finished) {
      // finalize_generation: no fallback content; _append_unstreamed_tool_args (arguments None -> "")
      if (dm && !dm->tool_calls.empty()) {
        auto& last = dm->tool_calls.back();
        last.arguments = last.arguments.value_or("");
      }
      dm = flush_engine_parsers(std::move(dm));
    }
    if (dm && !o.include_reasoning) {
      dm->set_reasoning(std::nullopt);
      if ((!dm->content || dm->content->empty()) && dm->tool_calls.empty()) dm.reset();
    }
    return dm;
  }

  int count_reasoning_tokens() {
    if (prompt_ended_reasoning) return rp.count_after_prompt_end(generated, stream_finished);
    return rp.count_reasoning_tokens(generated);
  }
};

Parser::Parser(std::shared_ptr<const ParserVocab> vocab, ParserOptions opts)
    : d_(std::make_unique<Impl>(std::move(vocab), std::move(opts))) {}
Parser::~Parser() = default;
Parser::Parser(Parser&&) noexcept = default;
Parser& Parser::operator=(Parser&&) noexcept = default;

void Parser::set_prompt_token_ids(std::span<const int32_t> prompt_ids) {
  if (d_->prompt_reasoning_checked) return;
  d_->prompt.assign(prompt_ids.begin(), prompt_ids.end());
  d_->have_prompt = true;
}

ParserDelta Parser::feed(std::string_view delta_text, std::span<const int32_t> delta_ids, bool finished) {
  Delta d = d_->parse_delta(delta_text, delta_ids, finished);
  d_->generated.insert(d_->generated.end(), delta_ids.begin(), delta_ids.end());
  if (!d) {
    // dsml_roundtrip v2.3 keep-alive: None -> DeltaMessage() when not finished and continuous_usage_stats is on
    ParserDelta x;
    x.present = !finished && d_->o.continuous_usage_stats;
    return x;
  }
  if (!d->tool_calls.empty()) tools_streamed_ = true;
  return std::move(*d);
}

ParserDelta Parser::finish() { return feed("", {}, true); }

std::string Parser::stream_finish_reason(std::string_view engine_reason) const {
  bool named = d_->o.tool_choice == ToolChoice::Named && !d_->o.named_function.empty();
  if (tools_streamed_ && !named && engine_reason == "stop") return "tool_calls";
  return engine_reason.empty() ? std::string("stop") : std::string(engine_reason);
}

int Parser::reasoning_tokens() { return d_->count_reasoning_tokens(); }

Parser::Full Parser::parse_full(std::string_view text, std::span<const int32_t> ids, std::string_view engine_reason) {
  Impl& I = *d_;
  const ParserOptions& o = I.o;
  // DelegatingParser.parse: extract_reasoning (text only), then _extract_tool_calls(enable_auto_tools=True)
  auto [reasoning, content] = I.rp.extract_reasoning(text);
  std::optional<std::vector<ToolCall>> tool_calls;
  std::optional<std::string> content2;
  bool named = o.tool_choice == ToolChoice::Named;
  bool required = o.tool_choice == ToolChoice::Required;
  if (o.tool_choice == ToolChoice::None) {
    auto info = I.tool_adapter_full(content ? *content : std::string_view());
    tool_calls = std::vector<ToolCall>{};
    content2 = info.content;
  } else {
    auto info = I.tool_adapter_full(content ? *content : std::string_view());
    if (info.tools_called) {
      tool_calls = info.tool_calls;
      content2 = info.content;
      if (content2 && !content2->empty() && pyjson::py_is_blank(*content2)) content2.reset();
    } else if ((required || named) && (!content || pyjson::py_is_blank(*content))) {
      tool_calls = std::vector<ToolCall>{};
      content2.reset();
    } else {
      tool_calls.reset();
      content2 = info.content && !info.content->empty() ? info.content : std::nullopt;
    }
  }
  I.generated.assign(ids.begin(), ids.end());
  Full f;
  f.reasoning_tokens = I.rp.count_reasoning_tokens(ids);
  // chat_completion_full_generator
  f.reasoning = o.include_reasoning ? reasoning : std::nullopt;
  if (named || required) {
    f.content = content2.value_or("");
    if (tool_calls) f.tool_calls = *tool_calls;
  } else if (o.tool_choice == ToolChoice::Null || o.tool_choice == ToolChoice::None) {
    f.content = content2;
  } else if (o.has_tools && o.tool_choice == ToolChoice::Auto) {
    f.auto_tools_called = tool_calls && !tool_calls->empty();
    if (f.auto_tools_called) f.tool_calls = *tool_calls;
    f.content = content2;
  } else {
    f.content = content2;
  }
  if (!o.parallel_tool_calls && f.tool_calls.size() > 1) f.tool_calls.resize(1);
  bool tc = f.auto_tools_called || (required && engine_reason == "stop");
  f.finish_reason = tc ? std::string("tool_calls") : engine_reason.empty() ? std::string("stop") : std::string(engine_reason);
  return f;
}

// ------------------------------------------------------------------------------------------------ free functions
bool parser_thinking(std::string_view kwargs_json) {
  pyjson::Value kw;
  if (kwargs_json.empty() || !pyjson::loads(kwargs_json, kw) || !kw.is_obj()) kw = pyjson::Value::object();
  const pyjson::Value* t = kw.get("thinking");
  const pyjson::Value* e = kw.get("enable_thinking");
  bool thinking = (t && t->truthy()) || (e && e->truthy());
  if (!t && !e) thinking = true;
  const pyjson::Value* eff = kw.get("reasoning_effort");
  if (eff && eff->is_str() && eff->s == "none") thinking = false;
  return thinking;
}

bool is_reasoning_end(std::span<const int32_t> prompt_ids, const ParserVocab& vocab, bool thinking) {
  auto sh = vocab.internal ? std::static_pointer_cast<const dsml::Shape>(vocab.internal)
                           : std::make_shared<const dsml::Shape>(vocab);
  std::shared_ptr<const ParserVocab> vp(&vocab, [](const ParserVocab*) {});
  std::function<std::string()> idgen = random_tool_call_id;
  dsml::ParserEngine e(sh, vp, thinking, &idgen);
  return e.is_reasoning_end(prompt_ids);
}

bool engine_reasoning_ended(std::span<const int32_t> prompt_ids, const ParserVocab& vocab, bool thinking,
                            bool include_reasoning) {
  if (!include_reasoning) return true;
  return is_reasoning_end(prompt_ids, vocab, thinking);
}

}  // namespace fe
