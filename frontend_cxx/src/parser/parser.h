// fe_parser: reasoning + DSML tool-call parser for DeepSeek-V4.1.
//
// ParserManager's DelegatingParser over the DeepSeekV41 reasoning and tool adapters (two ParserEngine instances)
// with the plugin's v1/v2 patches and the vllm-0.32.0-reasoning-count.patch. Spec: tools/inventory.md 1.8, 1.9,
// 1.11, 5.1. Golden test: tools/golden/parser_dump.py + tests/parser/.
//
// Use, per chat request (one Parser per choice, like serving.py):
//   auto vocab = fe::make_parser_vocab(tokenizer);                    // once at startup
//   fe::ParserOptions o = fe::parser_options_for(request, effective_chat_template_kwargs);
//   fe::Parser p(vocab, o);
//   p.set_prompt_token_ids(prompt_ids);                               // before the first feed (streaming)
//   streaming: for each engine output: d = p.feed(text, ids, finished); if d.present -> chunk delta d.to_json()
//              tools_streamed = p.tools_called(); finish = p.stream_finish_reason(engine_reason)
//              usage: p.reasoning_tokens() once at the end (or after every feed with continuous_usage_stats)
//   non-streaming: auto full = p.parse_full(text, all_ids, engine_reason)  (a fresh Parser; no set_prompt needed)
//   grammar: fe::structural_tag_for(request) -> StructuredOutputsParams.structural_tag (and drop response_format)
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fe {

class Json;  // common/json.h

// What the parser needs from the tokenizer. Build once (make_parser_vocab) and share.
struct ParserVocab {
  std::function<std::optional<int32_t>(std::string_view)> token_to_id;  // vocab lookup (added tokens included)
  std::function<std::string(int32_t)> decode_one;                        // == tokenizer.decode([id]) (special kept)
  std::function<std::string(std::span<const int32_t>)> decode;           // == tokenizer.decode(ids) (special kept)
  // tokenizer.all_special_tokens / all_special_ids (HF: bos, eos, unk, pad, additional; deduplicated, in order).
  // For the DeepSeek-V4.1 checkpoint: [("<｜begin▁of▁sentence｜>", 0), ("<｜end▁of▁sentence｜>", 1)].
  std::vector<std::pair<std::string, int32_t>> special_tokens;
  std::shared_ptr<const void> internal;  // set by finalize_parser_vocab()
};
// Precomputes the lexer/scanner tables; call after filling the fields (make_parser_vocab does it).
void finalize_parser_vocab(ParserVocab& v);

// Generic adapter over a tokenizer with token_to_id(sv) -> optional<int32_t> and decode(span, bool skip_special)
// (fe::Tokenizer). The vocab keeps a reference: the tokenizer must outlive it.
template <class Tok>
std::shared_ptr<const ParserVocab> make_parser_vocab(const Tok& tok) {
  auto v = std::make_shared<ParserVocab>();
  v->token_to_id = [&tok](std::string_view s) -> std::optional<int32_t> {
    auto r = tok.token_to_id(s);
    if (!r) return std::nullopt;
    return (int32_t)*r;
  };
  v->decode_one = [&tok](int32_t id) { int32_t a[1] = {id}; return std::string(tok.decode(std::span<const int32_t>(a, 1), false)); };
  v->decode = [&tok](std::span<const int32_t> ids) { return std::string(tok.decode(ids, false)); };
  for (const char* s : {"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"})
    if (auto id = tok.token_to_id(s)) v->special_tokens.emplace_back(s, (int32_t)*id);
  finalize_parser_vocab(*v);
  return v;
}

// request.tool_choice after pydantic validation: absent -> "auto" when tools are given, else "none"; explicit null
// stays Null (parsed like auto, but the non-streaming message then carries no tool_calls, as in serving.py).
enum class ToolChoice : uint8_t { Null, None, Auto, Required, Named };

struct ParserOptions {
  bool thinking_initial = true;    // DeepSeekV4Parser rule over the effective chat_template_kwargs: parser_thinking()
  bool include_reasoning = true;   // request.include_reasoning
  bool has_tools = false;          // bool(request.tools)
  ToolChoice tool_choice = ToolChoice::None;
  std::string named_function;      // tool_choice.function.name when Named
  bool parallel_tool_calls = true; // false only when the request sets parallel_tool_calls=false explicitly
  // dsml_roundtrip v2.3 usage keep-alive (DelegatingParser.parse_delta patch): request.stream_options is set and its
  // continuous_usage_stats is true. feed() then returns an empty delta (present, "{}") instead of None for a step that
  // is not the final one, so serving_chat still sends that step's chunk (and its usage). Streaming chat only.
  bool continuous_usage_stats = false;
  // Tool-call id generator. Default: "chatcmpl-tool-" + vllm.utils.random_uuid() = 16 lowercase hex digits of the
  // low 64 bits of a uuid4. Injectable for tests.
  std::function<std::string()> make_tool_call_id;
  // vllm:tool_call_parser_invocations_total hook (vllm/parser/metrics.py record_tool_parser_invocation), called
  // where DelegatingParser records it (parser/abstract_parser.py:628-690), i.e. once per tool-parser call:
  //   feed():       once per call made in the tool-call phase (reasoning ended, incl. the delta that ends it):
  //                 streaming=true, tool_called = the tool adapter's delta carried tool_calls (before
  //                 tool_choice=none strips them);
  //   parse_full(): once: streaming=false, tool_called = ExtractedToolCallInformation.tools_called.
  // A tool-parser exception counts as tool_called=false and is rethrown. Empty = nothing recorded.
  std::function<void(bool streaming, bool tool_called)> on_tool_parser_invocation;
};

// DeltaFunctionCall / DeltaToolCall: "id" present implies "type":"function"; name / arguments present iff set.
struct ToolCallDelta {
  int index = 0;
  std::optional<std::string> id, name, arguments;
};

// A DeltaMessage. present=false is Python's None (no delta; serving.py skips the chunk unless it must send one).
// content_set / reasoning_set: the field is in pydantic's fields_set (exclude_unset dump), i.e. it is serialised,
// as JSON null when the optional is empty.
struct ParserDelta {
  bool present = false;
  std::optional<std::string> content, reasoning;
  bool content_set = false, reasoning_set = false;
  std::vector<ToolCallDelta> tool_calls;
  // The chunk's "delta" object exactly as choice.model_dump_json(exclude_unset=True) writes it (compact, UTF-8).
  // parallel_tool_calls=false applies maybe_filter_parallel_tool_calls (tool_calls with index 0 only).
  std::string to_json(bool parallel_tool_calls = true) const;
  void set_content(std::optional<std::string> v) { content = std::move(v); content_set = true; }
  void set_reasoning(std::optional<std::string> v) { reasoning = std::move(v); reasoning_set = true; }
};

struct ToolCall {
  std::string id, name, arguments;  // arguments: the dsml_roundtrip v2 official bytes
};

class Parser {
 public:
  Parser(std::shared_ptr<const ParserVocab> vocab, ParserOptions opts);
  template <class Tok, class = decltype(std::declval<const Tok&>().token_to_id(std::string_view()))>
  Parser(const Tok& tok, ParserOptions opts) : Parser(make_parser_vocab(tok), std::move(opts)) {}
  ~Parser();
  Parser(Parser&&) noexcept;
  Parser& operator=(Parser&&) noexcept;

  // Streaming. Must be called with the prompt before the first feed() (serving.py passes res.prompt_token_ids;
  // only the first call matters). Without it the first feed behaves as if prompt_token_ids were None.
  void set_prompt_token_ids(std::span<const int32_t> prompt_ids);
  // == DelegatingParser.parse_delta(delta_text, delta_token_ids, request, prompt_token_ids, finished=...)
  // delta_text = detokenizer delta (skip_special_tokens=False, stop token text excluded), delta_ids = all new ids.
  // finished = (engine finish_reason is not None): the final delta and the flush happen in this one call.
  ParserDelta feed(std::string_view delta_text, std::span<const int32_t> delta_ids, bool finished = false);
  // Convenience for callers that learn about the end separately: feed("", {}, true). Not identical to passing
  // finished=true with the last engine output; serving code should use feed(..., true).
  ParserDelta finish();
  bool tools_called() const { return tools_streamed_; }  // any streamed delta carried tool_calls (tools_streamed)
  // finish_reason of the final chunk: "tool_calls" when tools were streamed, tool_choice is not named and the engine
  // said "stop"; else the engine's reason (or "stop").
  std::string stream_finish_reason(std::string_view engine_reason) const;
  // count_reasoning_tokens(all generated ids so far) -- call exactly where serving.py does: once after the final
  // feed, or after every feed when stream_options.continuous_usage_stats is on (the patched incremental count
  // depends on the call pattern). 0 when no token was generated.
  int reasoning_tokens();

  // Non-streaming: parser.parse(text, request, enable_auto_tools=True, ids) + chat_completion_full_generator's
  // message rules + count_reasoning_tokens(ids). Use a fresh Parser.
  struct Full {
    std::optional<std::string> reasoning;  // None when include_reasoning=false
    std::optional<std::string> content;    // "" instead of None for tool_choice required / named
    std::vector<ToolCall> tool_calls;      // message.tool_calls (after the parallel_tool_calls=false filter)
    bool auto_tools_called = false;
    std::string finish_reason;
    int reasoning_tokens = 0;
  };
  // engine_finish_reason: the engine's "stop" / "length" / ...; it decides finish_reason ("tool_calls" rule).
  Full parse_full(std::string_view text, std::span<const int32_t> ids, std::string_view engine_finish_reason = "stop");

 private:
  struct Impl;
  std::unique_ptr<Impl> d_;
  bool tools_streamed_ = false;
};

// DeepSeekV4Parser.__init__: thinking = bool(kw.thinking or kw.enable_thinking); True when neither key is present;
// and reasoning_effort != "none". kwargs_json = the effective chat_template_kwargs (after build_chat_params and the
// dsml_roundtrip effort/thinking mapping) as a JSON object; "" or "null" = {}.
bool parser_thinking(std::string_view chat_template_kwargs_json);

// ParserEngine.is_reasoning_end(prompt_ids) of the reasoning parser (thinking = parser_thinking(...)).
bool is_reasoning_end(std::span<const int32_t> prompt_ids, const ParserVocab& vocab, bool thinking);
// EngineCoreRequest.reasoning_ended = request.resolve_reasoning_ended(parser, prompt):
// True when include_reasoning=false, else is_reasoning_end(prompt).
bool engine_reasoning_ended(std::span<const int32_t> prompt_ids, const ParserVocab& vocab, bool thinking,
                            bool include_reasoning);

// The xgrammar deepseek_v4_1 structural tag that DelegatingParser._apply_structural_tag puts into
// request.structured_outputs.structural_tag (json.dumps(tag.model_dump()), ensure_ascii, ", " / ": "), or nullopt
// when Python sets none. request_json = the chat request body (tools, tool_choice, parallel_tool_calls,
// response_format, structured_outputs, dsml_strict are read; the env DSML_STRICT is honoured like the plugin).
// When a tag is returned, the request's response_format is dropped (Python sets it to None).
// Throws std::invalid_argument for inputs Python rejects in adjust_request (message = Python's) and
// std::runtime_error("unsupported: ...") for structured-output forms not ported (regex/choice/grammar/structural_tag
// combined with tools + tool_choice=auto).
std::optional<std::string> structural_tag_for_text(std::string_view request_json);
// Same, from the parsed request (only tools / tool_choice / parallel_tool_calls / response_format /
// structured_outputs / dsml_strict are read). Needs fe_common.
std::optional<std::string> structural_tag_for(const Json& request);

// ParserOptions from the validated chat request body and the effective chat_template_kwargs (the dict the encoder
// renders with: request.build_chat_params(...).chat_template_kwargs incl. the dsml_roundtrip mapping).
// make_tool_call_id is left empty (random). Needs fe_common.
ParserOptions parser_options_for(const Json& request, const Json& chat_template_kwargs);

// The default tool-call id ("chatcmpl-tool-" + 16 hex).
std::string random_tool_call_id();

}  // namespace fe
