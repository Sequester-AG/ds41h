// every Python str operation used here (find, slicing at match boundaries, prefix tests)
// gives the same result on valid UTF-8 bytes as on code points. Per-character token counts are kept per byte, on the
// first byte of each character.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "parser/parser.h"

namespace fe::dsml {

enum class St : uint8_t { CONTENT, REASONING, MESSAGE_HEADER, TOOL_PREAMBLE, TOOL_NAME, TOOL_ARGS, TOOL_BETWEEN };
enum class Tm : uint8_t {
  CONTENT, THINK_START, THINK_END, TOOL_START, TOOL_END, INVOKE_PREFIX, INVOKE_NAME_END, INVOKE_END,
  PARAM_START, PARAM_CLOSE, DROP
};
enum class Ev : uint8_t {
  TEXT_CHUNK, REASONING_START, REASONING_CHUNK, REASONING_END, TOOL_CALL_START, TOOL_NAME, ARG_VALUE_CHUNK, TOOL_CALL_END
};

inline bool is_tool_state(St s) {
  return s == St::TOOL_PREAMBLE || s == St::TOOL_NAME || s == St::TOOL_ARGS || s == St::TOOL_BETWEEN;
}

struct Event { Ev type; std::string value; int tool_index = -1; int token_count = 0; };
using Events = std::vector<Event>;

struct LexToken { Tm terminal; std::string value; int token_count = 0; };
struct TextChunk { std::string text; std::vector<std::string> token_texts; int token_count = 0; };
struct PreLexed { Tm terminal; int32_t token_id; std::string text; };
using LexInput = std::variant<TextChunk, PreLexed>;

// The v41 terminal set + the drop terminals of the tokenizer's special tokens (precomputed once per vocab).
struct Shape {
  std::vector<std::pair<std::string, Tm>> literals;  // every spelling
  bool first[128] = {};                              // first byte of each literal (all ASCII here)
  std::map<int32_t, Tm> token_ids;                   // resolved token-id terminals (incl. DROP)
  std::optional<int32_t> think_start_id, think_end_id;
  std::vector<int32_t> reasoning_end_ids;            // _derive_reasoning_end_token_ids
  bool has_drops = false;
  bool tid_name[16] = {};                             // _token_id_terminal_names
  explicit Shape(const ParserVocab& v);
  bool any_first(std::string_view s) const {
    for (unsigned char c : s) if (c < 128 && first[c]) return true;
    return false;
  }
};

struct Transition { St next; std::vector<Ev> events; };
const Transition* transition(St s, Tm t);

class Lexer {
 public:
  explicit Lexer(const Shape& sh) : sh_(sh) {}
  void reset() { buffer.clear(); counts_.clear(); }
  std::vector<LexToken> feed(std::string_view text, const std::vector<std::string>& token_texts, int token_count);
  std::vector<LexToken> flush();
  std::string buffer;
 private:
  std::vector<int> char_counts(std::string_view text, const std::vector<std::string>& tt, int token_count) const;
  int sum_range(size_t from, size_t to) const;
  int sum_all() const;
  std::vector<LexToken> drain(bool final_);
  size_t content_boundary(std::string_view buf) const;
  bool in_prefix_set(std::string_view b) const;
  const Shape& sh_;
  std::vector<int> counts_;
};

class Scanner {
 public:
  Scanner(const Shape& sh, const ParserVocab& v) : sh_(sh), v_(v) {}
  void reset() { deferred_.clear(); deferred_prefix_counts_.clear(); deferred_trailing_ = 0; deferred_post_text_.clear(); }
  std::vector<LexInput> scan(std::string_view delta_text, std::span<const int32_t> ids);
  std::vector<LexInput> flush_pending();
  bool has_deferred() const { return !deferred_.empty(); }
 private:
  const std::string& decode(int32_t id);
  std::vector<std::string> decode_all(std::span<const int32_t> ids);
  void resolve_deferred(std::string_view delta_text, int cur_count, std::vector<LexInput>& results,
                        std::string& remaining, int& trailing);
  std::vector<LexInput> recover_holdback(std::string_view delta_text, std::vector<LexInput> results);
  std::vector<LexInput> rebuild_from_anchors(std::string_view delta_text, const std::vector<LexInput>& results);
  const Shape& sh_;
  const ParserVocab& v_;
  std::map<int32_t, std::string> cache_;
  std::vector<PreLexed> deferred_;
  std::vector<int> deferred_prefix_counts_;
  int deferred_trailing_ = 0;
  std::string deferred_post_text_;
};

class StreamingEngine {
 public:
  StreamingEngine(const Shape& sh, const ParserVocab& v, St config_initial);
  Events feed(std::string_view delta_text, std::span<const int32_t> ids);
  Events finish();
  void reset(std::optional<St> initial);
  St state = St::CONTENT;
  int tool_index = -1;
  bool skip_tool_parsing = false;
  bool skip_reasoning_parsing = false;
  int reasoning_token_count = 0;
 private:
  void record(const Events& ev);
  Events process_scanner_items(std::vector<LexInput>& items);
  Events process_lex_tokens(std::vector<LexToken>& toks);
  Events on_terminal(Tm terminal, const std::string& value, int token_count);
  Events emit_for_state(const std::string& text, int token_count);
  Events on_content(const std::string& text, int token_count);
  Events apply_transition(const Transition& tr, const std::string& value, int token_count);
  const Shape& sh_;
  St config_initial_;
  Scanner scanner_;
  Lexer lexer_;
  bool ever_had_token_ids_ = false;
  bool in_skipped_tool_span_ = false;
  std::vector<std::string> v41_raw_;  // dsml_roundtrip: text emitted in TOOL_ARGS for the current invoke
};

// dsml_roundtrip _Scan: the official parameter loop over a raw invoke body, incremental over append-only text.
struct ParamScan {
  size_t pos = 0;
  long open_at = -1;
  size_t end_from = 0;
  std::vector<std::string> names;
  std::vector<std::string> entries;
  size_t emitted = 0;
  bool closed = false;
  void advance(std::string_view raw);
  std::optional<std::string> tail_entry(std::string_view raw) const;
};
std::string official_arguments(std::string_view raw);
bool param_open(std::string_view raw);

struct Slot {
  std::string id, name, args, streamed_json;
  bool name_sent = false;
};

struct ReqView {
  bool has_tools = false;
  ToolChoice tool_choice = ToolChoice::None;
};

using Delta = std::optional<ParserDelta>;  // DeltaMessage | None

struct ExtractedInfo {
  bool tools_called = false;
  std::vector<ToolCall> tool_calls;
  std::optional<std::string> content;
};

class ParserEngine {
 public:
  ParserEngine(std::shared_ptr<const Shape> sh, std::shared_ptr<const ParserVocab> v, bool thinking,
               std::function<std::string()>* idgen);
  // lifecycle
  void initialize_streaming(std::optional<St> initial = std::nullopt);
  Delta finish_streaming();
  void reset(std::optional<St> initial = std::nullopt);
  Events feed(std::string_view t, std::span<const int32_t> ids) { return engine.feed(t, ids); }
  // reasoning
  std::pair<std::optional<std::string>, std::optional<std::string>> extract_reasoning(std::string_view text);
  Delta extract_reasoning_streaming(std::string_view delta_text, std::span<const int32_t> ids);
  bool is_reasoning_end(std::span<const int32_t> ids) const;
  std::vector<int32_t> extract_content_ids(std::span<const int32_t> ids) const;
  int count_reasoning_tokens() const { return engine.reasoning_token_count; }
  // tools
  void check_skip_tool_parsing(const ReqView& r);
  ExtractedInfo extract_tool_calls_from_content(std::string_view content, const ReqView& r);
  Delta extract_tool_calls_streaming(std::string_view delta_text, std::span<const int32_t> ids, const ReqView& r);
  struct SinglePass { std::optional<std::string> reasoning, content; ExtractedInfo info; };
  SinglePass single_pass_parse(std::string_view text, std::span<const int32_t> ids, std::optional<St> initial);

  StreamingEngine engine;
  bool reasoning_ended = false;

 private:
  Delta events_to_delta(Events& events, bool finished);
  Delta events_to_delta_orig(Events& events, bool finished);
  std::optional<std::string> strip_content_whitespace(std::string content, bool tools_called);
  void ensure_slot(int idx);
  void ensure_tool_id(Slot& s);
  void emit_name_delta(int idx, std::vector<ToolCallDelta>& deltas, const std::optional<std::string>& name);
  void handle_arg_chunk(const Event& ev, std::vector<ToolCallDelta>& deltas);
  void handle_tool_end(const Event& ev, std::vector<ToolCallDelta>& deltas);
  std::optional<std::string> compute_arg_delta(int idx, const std::string& raw_delta);
  std::optional<std::string> flush_arg_converter(int idx);
  std::optional<std::string> try_extract_name(int idx) const;
  ExtractedInfo build_extracted_result();
  Slot& slot(int idx);

  std::shared_ptr<const Shape> sh_;
  std::shared_ptr<const ParserVocab> v_;
  bool thinking_;
  std::function<std::string()>* idgen_;
  bool streaming_initialized_ = false;
  std::vector<Slot> slots_;
  std::string deferred_content_;
  bool content_has_nonws_ = false;
  bool suppress_tool_calls_ = false;
  bool has_tools_ = false;
  bool v41_notools_ = false;  // dsml_roundtrip v2.3/v2.4: skip_tool_parsing was set (no tools, or tool_choice none)
  // dsml_roundtrip
  std::string rt_nl_;
  bool rt_done_ = false;
  bool rt_stripped_ = false;
  std::map<int, ParamScan> v41_scan_;
};

class ReasoningAdapter {
 public:
  ReasoningAdapter(std::shared_ptr<const Shape> sh, std::shared_ptr<const ParserVocab> v, bool thinking,
                   std::function<std::string()>* idgen);
  bool is_reasoning_end(std::span<const int32_t> ids) const { return engine_->is_reasoning_end(ids); }
  std::vector<int32_t> extract_content_ids(std::span<const int32_t> ids) const { return engine_->extract_content_ids(ids); }
  std::pair<std::optional<std::string>, std::optional<std::string>> extract_reasoning(std::string_view text);
  Delta extract_reasoning_streaming(std::string_view delta_text, std::span<const int32_t> ids);
  bool has_engine_confirmed_reasoning_end() const { return engine_->reasoning_ended; }
  Delta finish_streaming();
  int count_reasoning_tokens(std::span<const int32_t> ids);
  int count_after_prompt_end(std::span<const int32_t> ids, bool finished);
 private:
  std::unique_ptr<ParserEngine> make() const;
  std::shared_ptr<const Shape> sh_;
  std::shared_ptr<const ParserVocab> v_;
  bool thinking_;
  std::function<std::string()>* idgen_;
  std::unique_ptr<ParserEngine> engine_, counting_, incremental_;
  size_t incremental_counted_ = 0;
  bool incremental_flushed_ = false;
  bool streaming_count_valid_ = false;
};

class ToolAdapter {
 public:
  ToolAdapter(std::shared_ptr<const Shape> sh, std::shared_ptr<const ParserVocab> v, bool thinking,
              std::function<std::string()>* idgen)
      : engine_(std::make_unique<ParserEngine>(std::move(sh), std::move(v), thinking, idgen)) {}
  ExtractedInfo extract_tool_calls(std::string_view text, const ReqView& r) { return engine_->extract_tool_calls_from_content(text, r); }
  Delta extract_tool_calls_streaming(std::string_view delta_text, std::span<const int32_t> ids, const ReqView& r) {
    engine_->initialize_streaming(St::CONTENT);
    return engine_->extract_tool_calls_streaming(delta_text, ids, r);
  }
  Delta finish_streaming() { return engine_->finish_streaming(); }
 private:
  std::unique_ptr<ParserEngine> engine_;
};

}  // namespace fe::dsml
