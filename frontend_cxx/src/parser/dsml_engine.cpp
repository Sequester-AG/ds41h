#include "parser/dsml_engine.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "parser/pyjson.h"

namespace fe::dsml {

using pyjson::dump_str;
using pyjson::py_is_blank;

namespace {

const std::string kThinkStart = "<think>";
const std::string kThinkEnd = "</think>";
const std::string kToolStart = "<｜DSML｜ calls>";
const std::string kToolEnd = "</｜DSML｜ calls>";
const std::string kInvokePrefix = "<｜DSML｜ invoke name=\"";
const std::string kInvokeNameEnd = "\">";
const std::string kInvokeEnd = "</｜DSML｜ invoke>";
const std::string kParamStart = "<｜DSML｜ parameter";
const std::string kParamClose = "</｜DSML｜ parameter>";
// dsml_roundtrip official grammar
const std::string kPStart = "<｜DSML｜ parameter";
const std::string kPEnd = "/｜DSML｜ parameter";
const std::string kPClose = "</｜DSML｜ parameter>";

bool starts_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
bool ends_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// deepseek_v4_config transitions, in declaration order
struct TEntry { St s; Tm t; Transition tr; };
const std::vector<TEntry>& table() {
  static const std::vector<TEntry> T = {
      {St::CONTENT, Tm::THINK_START, {St::REASONING, {Ev::REASONING_START}}},
      {St::CONTENT, Tm::THINK_END, {St::CONTENT, {}}},
      {St::REASONING, Tm::THINK_START, {St::REASONING, {}}},
      {St::REASONING, Tm::THINK_END, {St::CONTENT, {Ev::REASONING_END}}},
      {St::REASONING, Tm::TOOL_START, {St::TOOL_PREAMBLE, {Ev::REASONING_END}}},
      {St::CONTENT, Tm::TOOL_START, {St::TOOL_PREAMBLE, {}}},
      {St::TOOL_PREAMBLE, Tm::INVOKE_PREFIX, {St::TOOL_NAME, {Ev::TOOL_CALL_START}}},
      {St::CONTENT, Tm::INVOKE_PREFIX, {St::TOOL_NAME, {Ev::TOOL_CALL_START}}},
      {St::TOOL_NAME, Tm::INVOKE_NAME_END, {St::TOOL_ARGS, {}}},
      {St::TOOL_ARGS, Tm::INVOKE_END, {St::TOOL_BETWEEN, {Ev::TOOL_CALL_END}}},
      {St::TOOL_ARGS, Tm::TOOL_END, {St::TOOL_BETWEEN, {Ev::TOOL_CALL_END}}},
      {St::TOOL_BETWEEN, Tm::INVOKE_PREFIX, {St::TOOL_NAME, {Ev::TOOL_CALL_START}}},
      {St::TOOL_BETWEEN, Tm::TOOL_END, {St::TOOL_BETWEEN, {}}},
      {St::TOOL_BETWEEN, Tm::TOOL_START, {St::TOOL_PREAMBLE, {}}},
  };
  return T;
}

struct Derived {
  bool tool_terminal[16] = {};
  bool tool_exit_terminal[16] = {};
  bool reasoning_markup[16] = {};
  Derived() {
    for (auto& e : table()) {
      if (is_tool_state(e.tr.next) || is_tool_state(e.s)) tool_terminal[(int)e.t] = true;
      if (is_tool_state(e.s) && !is_tool_state(e.tr.next)) tool_exit_terminal[(int)e.t] = true;
    }
    auto plain = [](St s) { return s == St::CONTENT || s == St::REASONING; };
    bool markers[16] = {};
    for (auto& e : table())
      if ((e.s == St::REASONING || e.tr.next == St::REASONING) && !is_tool_state(e.tr.next)) markers[(int)e.t] = true;
    bool pure = true;
    for (auto& e : table()) {
      if (!markers[(int)e.t]) continue;
      bool evs_ok = true;
      for (Ev x : e.tr.events) if (x != Ev::REASONING_START && x != Ev::REASONING_END) evs_ok = false;
      if (!plain(e.s) || !plain(e.tr.next) || !evs_ok) pure = false;
    }
    if (pure) for (int i = 0; i < 16; ++i) reasoning_markup[i] = markers[i];
  }
};
const Derived& derived() { static const Derived D; return D; }

std::optional<Ev> content_event(St s) {
  switch (s) {
    case St::CONTENT: return Ev::TEXT_CHUNK;
    case St::REASONING: return Ev::REASONING_CHUNK;
    case St::TOOL_NAME: return Ev::TOOL_NAME;
    case St::TOOL_ARGS: return Ev::ARG_VALUE_CHUNK;
    default: return std::nullopt;
  }
}

template <class T>
void append(std::vector<T>& a, std::vector<T>&& b) {
  if (a.empty()) { a = std::move(b); return; }
  for (auto& x : b) a.push_back(std::move(x));
}

}  // namespace

const Transition* transition(St s, Tm t) {
  for (auto& e : table())
    if (e.s == s && e.t == t) return &e.tr;
  return nullptr;
}

// ---------------------------------------------------------------------------------------------------------- Shape
Shape::Shape(const ParserVocab& v) {
  literals = {{kThinkStart, Tm::THINK_START}, {kThinkEnd, Tm::THINK_END},   {kToolStart, Tm::TOOL_START},
              {kToolEnd, Tm::TOOL_END},       {kInvokePrefix, Tm::INVOKE_PREFIX}, {kInvokeNameEnd, Tm::INVOKE_NAME_END},
              {kInvokeEnd, Tm::INVOKE_END},   {kParamStart, Tm::PARAM_START},   {kParamClose, Tm::PARAM_CLOSE}};
  // token_id_terminals: THINK_START, THINK_END, TOOL_START, TOOL_END (deepseek_v41 overrides the tool spellings)
  const std::pair<std::string, Tm> tid_terms[] = {
      {kThinkStart, Tm::THINK_START}, {kThinkEnd, Tm::THINK_END}, {kToolStart, Tm::TOOL_START}, {kToolEnd, Tm::TOOL_END}};
  std::map<Tm, std::optional<int32_t>> tid_of;
  for (auto& [text, t] : tid_terms) {
    auto id = v.token_to_id ? v.token_to_id(text) : std::nullopt;
    tid_of[t] = id;
    if (id) token_ids[*id] = t;
  }
  think_start_id = tid_of[Tm::THINK_START];
  think_end_id = tid_of[Tm::THINK_END];
  // _build_drop_info: special tokens not among the configured texts are DROP terminals (text and id)
  std::set<std::string> configured;
  for (auto& [text, t] : tid_terms) configured.insert(text);
  for (auto& [text, t] : literals) configured.insert(text);
  std::vector<std::string> drop_texts;
  for (auto& [text, id] : v.special_tokens) {
    if (configured.count(text)) continue;
    token_ids[id] = Tm::DROP;
    if (std::find(drop_texts.begin(), drop_texts.end(), text) == drop_texts.end()) drop_texts.push_back(text);
  }
  if (!drop_texts.empty()) {
    has_drops = true;
    for (auto& t : drop_texts) literals.emplace_back(t, Tm::DROP);
  }
  for (auto& [id, t] : token_ids) tid_name[(int)t] = true;
  for (auto& [text, t] : literals) {
    if (text.empty()) continue;
    unsigned char c = (unsigned char)text[0];
    if (c >= 128) throw std::runtime_error("fe_parser: non-ASCII first character in a terminal literal");
    first[c] = true;
  }
  // _derive_reasoning_end_token_ids
  std::vector<std::pair<Tm, St>> end_terms;
  bool none = false;
  for (auto& e : table()) {
    if (e.s != St::REASONING) continue;
    bool has_end = std::find(e.tr.events.begin(), e.tr.events.end(), Ev::REASONING_END) != e.tr.events.end();
    if (has_end) {
      if (std::none_of(end_terms.begin(), end_terms.end(), [&](auto& p) { return p.first == e.t; }))
        end_terms.emplace_back(e.t, e.tr.next);
    } else if (e.tr.next != St::REASONING) {
      none = true;
      break;
    }
  }
  if (!none) {
    for (auto& [t, next] : end_terms) {
      std::optional<int32_t> id;
      auto it = tid_of.find(t);
      if (it != tid_of.end()) id = it->second;
      if (id) {
        if (std::find(reasoning_end_ids.begin(), reasoning_end_ids.end(), *id) == reasoning_end_ids.end())
          reasoning_end_ids.push_back(*id);
      } else if (next == St::CONTENT) {
        reasoning_end_ids.clear();
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------- Lexer
std::vector<int> Lexer::char_counts(std::string_view text, const std::vector<std::string>& tt, int token_count) const {
  std::vector<int> counts(text.size(), 0);
  if (text.empty()) return counts;
  if (!tt.empty()) {
    size_t pos = 0;
    int assigned = 0;
    for (auto& t : tt) {
      if (t.empty()) continue;
      size_t found = text.find(t, pos);
      if (found == std::string_view::npos) continue;
      counts[found] += 1;
      assigned += 1;
      pos = found + t.size();
    }
    int missing = token_count - assigned;
    if (missing > 0) counts[0] += missing;
  } else if (token_count) {
    counts[0] = token_count;
  }
  return counts;
}

int Lexer::sum_range(size_t from, size_t to) const {
  int s = 0;
  to = std::min(to, counts_.size());
  for (size_t i = from; i < to; ++i) s += counts_[i];
  return s;
}
int Lexer::sum_all() const { return sum_range(0, counts_.size()); }

bool Lexer::in_prefix_set(std::string_view b) const {
  if (b.empty()) return false;
  for (auto& [lit, t] : sh_.literals)
    if (b.size() < lit.size() && starts_with(lit, b)) return true;
  return false;
}

size_t Lexer::content_boundary(std::string_view buf) const {
  size_t n = buf.size();
  for (size_t i = 1; i < n; ++i) {
    unsigned char c = (unsigned char)buf[i];
    if (c >= 128 || !sh_.first[c]) continue;
    size_t remaining = n - i;
    for (auto& [lit, t] : sh_.literals) {
      size_t cl = std::min(remaining, lit.size());
      if (buf.compare(i, cl, lit, 0, cl) == 0) return i;
    }
  }
  return n;
}

std::vector<LexToken> Lexer::feed(std::string_view text, const std::vector<std::string>& tt, int token_count) {
  std::vector<int> counts = char_counts(text, tt, token_count);
  if (buffer.empty() && !sh_.literals.empty()) {
    if (!sh_.any_first(text)) {
      int s = 0;
      for (int c : counts) s += c;
      return {LexToken{Tm::CONTENT, std::string(text), s}};
    }
  }
  buffer.append(text);
  counts_.insert(counts_.end(), counts.begin(), counts.end());
  return drain(false);
}

std::vector<LexToken> Lexer::flush() {
  std::vector<LexToken> toks;
  if (!buffer.empty()) toks = drain(true);
  if (!buffer.empty()) {
    toks.push_back(LexToken{Tm::CONTENT, buffer, sum_all()});
    buffer.clear();
    counts_.clear();
  }
  return toks;
}

// IncrementalLexer._drain. Python re-slices the buffer after every token; here the consumed prefix is tracked as an
// offset and erased once at the end (same tokens, linear time).
std::vector<LexToken> Lexer::drain(bool final_) {
  std::vector<LexToken> toks;
  size_t off = 0;
  auto take = [&](Tm t, size_t n) {
    toks.push_back(LexToken{t, buffer.substr(off, n), sum_range(off, off + n)});
    off += n;
  };
  while (off < buffer.size()) {
    std::string_view b = std::string_view(buffer).substr(off);
    if (!sh_.any_first(b)) {
      take(Tm::CONTENT, b.size());
      break;
    }
    const std::pair<std::string, Tm>* best = nullptr;
    for (auto& lit : sh_.literals)
      if (starts_with(b, lit.first) && (!best || lit.first.size() > best->first.size())) best = &lit;
    if (!final_ && in_prefix_set(b)) {
      if (best) {
        bool longer = false;
        for (auto& lit : sh_.literals)
          if (lit.first.size() > best->first.size() && starts_with(lit.first, b)) { longer = true; break; }
        if (!longer) {
          take(best->second, best->first.size());
          continue;
        }
      }
      break;
    }
    if (best) {
      take(best->second, best->first.size());
    } else {
      size_t ce = content_boundary(b);
      if (ce == 0) ce = pyjson::cp_len_at(b, 0);
      take(Tm::CONTENT, ce);
    }
  }
  buffer.erase(0, off);
  counts_.erase(counts_.begin(), counts_.begin() + (long)std::min(off, counts_.size()));
  return toks;
}

// ---------------------------------------------------------------------------------------------------------- Scanner
const std::string& Scanner::decode(int32_t id) {
  auto it = cache_.find(id);
  if (it != cache_.end()) return it->second;
  return cache_[id] = v_.decode_one ? v_.decode_one(id) : std::string();
}

std::vector<std::string> Scanner::decode_all(std::span<const int32_t> ids) {
  std::vector<std::string> out;
  out.reserve(ids.size());
  for (int32_t id : ids) out.push_back(decode(id));
  return out;
}

static std::string text_of(const LexInput& x) {
  if (auto* c = std::get_if<TextChunk>(&x)) return c->text;
  return std::get<PreLexed>(x).text;
}

std::vector<LexInput> Scanner::scan(std::string_view delta_text, std::span<const int32_t> ids) {
  std::vector<LexInput> prefix_items;
  std::string effective(delta_text);
  int deferred_trailing = 0;
  if (!deferred_.empty()) resolve_deferred(delta_text, (int)ids.size(), prefix_items, effective, deferred_trailing);

  if (sh_.token_ids.empty()) {
    if (!effective.empty()) {
      auto tt = decode_all(ids);
      int n = (int)tt.size();
      prefix_items.push_back(TextChunk{effective, std::move(tt), deferred_trailing + n});
    }
    return prefix_items;
  }
  bool has_special = false;
  for (int32_t id : ids)
    if (sh_.token_ids.count(id)) { has_special = true; break; }
  if (!has_special) {
    if (!effective.empty()) {
      auto tt = decode_all(ids);
      int n = (int)tt.size();
      if (prefix_items.empty()) return {TextChunk{effective, std::move(tt), n}};
      prefix_items.push_back(TextChunk{effective, std::move(tt), deferred_trailing + n});
    }
    return prefix_items;
  }
  std::vector<std::string> decoded = decode_all(ids);
  std::vector<LexInput> results;
  std::vector<std::string> text_accum;
  for (size_t idx = 0; idx < ids.size(); ++idx) {
    auto it = sh_.token_ids.find(ids[idx]);
    if (it != sh_.token_ids.end()) {
      if (!text_accum.empty()) {
        std::string joined;
        for (auto& s : text_accum) joined += s;
        if (!joined.empty()) {
          int n = (int)text_accum.size();
          results.push_back(TextChunk{joined, text_accum, n});
        }
        text_accum.clear();
      }
      results.push_back(PreLexed{it->second, ids[idx], decoded[idx]});
    } else {
      text_accum.push_back(decoded[idx]);
    }
  }
  if (!text_accum.empty()) {
    std::string joined;
    for (auto& s : text_accum) joined += s;
    if (!joined.empty()) {
      int n = (int)text_accum.size();
      results.push_back(TextChunk{joined, text_accum, n});
    }
  }
  if (deferred_trailing) results.insert(results.begin(), TextChunk{"", {}, deferred_trailing});
  if (!effective.empty()) {
    results = recover_holdback(effective, std::move(results));
  } else {
    int ptc = 0;
    for (auto& r : results) {
      if (auto* c = std::get_if<TextChunk>(&r)) ptc += c->token_count;
      if (auto* p = std::get_if<PreLexed>(&r)) {
        deferred_.push_back(*p);
        deferred_prefix_counts_.push_back(ptc);
        ptc = 0;
      }
    }
    deferred_trailing_ += ptc;
    results.clear();
  }
  append(prefix_items, std::move(results));
  return prefix_items;
}

std::vector<LexInput> Scanner::flush_pending() {
  if (deferred_.empty() && deferred_post_text_.empty()) return {};
  std::vector<LexInput> results;
  if (!deferred_post_text_.empty()) {
    int pc = deferred_prefix_counts_.empty() ? 0 : deferred_prefix_counts_[0];
    results.push_back(TextChunk{deferred_post_text_, {}, pc});
    deferred_post_text_.clear();
  }
  for (auto& t : deferred_) results.push_back(t);
  if (deferred_trailing_) results.push_back(TextChunk{"", {}, deferred_trailing_});
  deferred_.clear();
  deferred_prefix_counts_.clear();
  deferred_trailing_ = 0;
  return results;
}

void Scanner::resolve_deferred(std::string_view delta_text, int cur_count, std::vector<LexInput>& results,
                               std::string& remaining, int& trailing_out) {
  std::vector<PreLexed> deferred = std::move(deferred_);
  std::vector<int> pcs = std::move(deferred_prefix_counts_);
  int trailing = deferred_trailing_;
  deferred_.clear();
  deferred_prefix_counts_.clear();
  deferred_trailing_ = 0;
  remaining = std::string(delta_text);
  if (!deferred_post_text_.empty()) {
    remaining = deferred_post_text_ + remaining;
    deferred_post_text_.clear();
  }
  for (size_t idx = 0; idx < deferred.size(); ++idx) {
    int ptc = pcs[idx];
    const PreLexed& term = deferred[idx];
    size_t pos = remaining.find(term.text);
    if (pos != std::string::npos && pos > 0) {
      results.push_back(TextChunk{remaining.substr(0, pos), {}, ptc});
      results.push_back(term);
      remaining = remaining.substr(pos + term.text.size());
    } else if (pos == 0) {
      results.push_back(term);
      remaining = remaining.substr(term.text.size());
    } else {
      if (!remaining.empty()) {
        deferred_post_text_ += remaining;
        ptc += cur_count;
        remaining.clear();
      }
      deferred_.push_back(term);
      deferred_prefix_counts_.push_back(ptc);
    }
  }
  if (!deferred_.empty()) {
    deferred_trailing_ = trailing;
    trailing = 0;
  }
  trailing_out = trailing;
}

std::vector<LexInput> Scanner::recover_holdback(std::string_view delta_text, std::vector<LexInput> results) {
  if (results.empty()) return {TextChunk{std::string(delta_text), {}, 0}};
  std::string reconstructed;
  for (auto& r : results) reconstructed += text_of(r);
  if (reconstructed.empty()) {
    std::vector<LexInput> out{TextChunk{std::string(delta_text), {}, 0}};
    append(out, std::move(results));
    return out;
  }
  size_t pos = delta_text.find(reconstructed);
  if (pos != std::string_view::npos && pos > 0) {
    std::vector<LexInput> out{TextChunk{std::string(delta_text.substr(0, pos)), {}, 0}};
    append(out, std::move(results));
    return out;
  }
  if (pos == 0) return results;
  return rebuild_from_anchors(delta_text, results);
}

std::vector<LexInput> Scanner::rebuild_from_anchors(std::string_view delta_text, const std::vector<LexInput>& results) {
  std::vector<const PreLexed*> anchors;
  for (auto& r : results)
    if (auto* p = std::get_if<PreLexed>(&r)) anchors.push_back(p);
  if (anchors.empty()) return {TextChunk{std::string(delta_text), {}, 0}};
  std::vector<std::vector<std::string>> groups(anchors.size() + 1);
  std::vector<int> counts(anchors.size() + 1, 0);
  size_t g = 0;
  for (auto& r : results) {
    if (std::holds_alternative<PreLexed>(r)) { ++g; continue; }
    auto& c = std::get<TextChunk>(r);
    groups[g].insert(groups[g].end(), c.token_texts.begin(), c.token_texts.end());
    counts[g] += c.token_count;
  }
  std::vector<long> positions(anchors.size(), -1);
  size_t search_end = delta_text.size();
  for (long i = (long)anchors.size() - 1; i >= 0; --i) {
    const std::string& t = anchors[i]->text;
    if (t.size() > search_end) continue;
    size_t pos = delta_text.rfind(t, search_end - t.size());
    if (pos != std::string_view::npos) {
      positions[i] = (long)pos;
      search_end = pos;
    }
  }
  std::vector<LexInput> out;
  size_t consumed = 0;
  for (size_t i = 0; i < anchors.size(); ++i) {
    long pos = positions[i];
    if (pos >= (long)consumed) {
      if (pos > (long)consumed)
        out.push_back(TextChunk{std::string(delta_text.substr(consumed, pos - consumed)), groups[i], counts[i]});
      out.push_back(*anchors[i]);
      consumed = pos + anchors[i]->text.size();
    } else {
      bool later = false;
      for (size_t k = i + 1; k < anchors.size(); ++k) if (positions[k] >= 0) later = true;
      if (!later && consumed < delta_text.size() && anchors[i]->terminal != Tm::DROP) {
        deferred_post_text_ += std::string(delta_text.substr(consumed));
        consumed = delta_text.size();
      }
      deferred_.push_back(*anchors[i]);
      deferred_prefix_counts_.push_back(counts[i]);
    }
  }
  if (consumed < delta_text.size())
    out.push_back(TextChunk{std::string(delta_text.substr(consumed)), groups.back(), counts.back()});
  return out;
}

// ---------------------------------------------------------------------------------------------------------- engine
StreamingEngine::StreamingEngine(const Shape& sh, const ParserVocab& v, St config_initial)
    : sh_(sh), config_initial_(config_initial), scanner_(sh, v), lexer_(sh) {
  reset(std::nullopt);
}

void StreamingEngine::reset(std::optional<St> initial) {
  v41_raw_.clear();
  state = initial.value_or(config_initial_);
  tool_index = -1;
  ever_had_token_ids_ = false;
  reasoning_token_count = 0;
  scanner_.reset();
  lexer_.reset();
  in_skipped_tool_span_ = false;
}

void StreamingEngine::record(const Events& ev) {
  for (auto& e : ev)
    if (e.type == Ev::REASONING_CHUNK) reasoning_token_count += e.token_count;
}

Events StreamingEngine::feed(std::string_view delta_text, std::span<const int32_t> ids) {
  if (!ids.empty()) ever_had_token_ids_ = true;
  if (!delta_text.empty() && lexer_.buffer.empty() && !scanner_.has_deferred() && !sh_.any_first(delta_text)) {
    bool has_special = false;
    for (int32_t id : ids)
      if (sh_.token_ids.count(id)) { has_special = true; break; }
    if (!has_special) {
      Events ev = emit_for_state(std::string(delta_text), (int)ids.size());
      record(ev);
      return ev;
    }
  }
  std::vector<LexInput> items = scanner_.scan(delta_text, ids);
  if (items.size() == 1 && std::holds_alternative<TextChunk>(items[0])) {
    auto& it = std::get<TextChunk>(items[0]);
    std::vector<LexToken> lex = lexer_.feed(it.text, it.token_texts, it.token_count);
    Events ev;
    if (lex.size() == 1 && lex[0].terminal == Tm::CONTENT) ev = emit_for_state(lex[0].value, lex[0].token_count);
    else ev = process_lex_tokens(lex);
    record(ev);
    return ev;
  }
  Events ev = process_scanner_items(items);
  record(ev);
  return ev;
}

Events StreamingEngine::process_scanner_items(std::vector<LexInput>& items) {
  Events ev;
  for (auto& item : items) {
    if (auto* p = std::get_if<PreLexed>(&item)) {
      auto fl = lexer_.flush();
      append(ev, process_lex_tokens(fl));
      append(ev, on_terminal(p->terminal, p->text, 0));
    } else {
      auto& c = std::get<TextChunk>(item);
      if (c.text.empty() && c.token_count) {
        append(ev, emit_for_state("", c.token_count));
      } else {
        auto lex = lexer_.feed(c.text, c.token_texts, c.token_count);
        append(ev, process_lex_tokens(lex));
      }
    }
  }
  return ev;
}

Events StreamingEngine::finish() {
  auto items = scanner_.flush_pending();
  Events ev = process_scanner_items(items);
  auto fl = lexer_.flush();
  append(ev, process_lex_tokens(fl));
  if (is_tool_state(state)) {
    if (tool_index >= 0) ev.push_back(Event{Ev::TOOL_CALL_END, "", tool_index, 0});
    state = St::CONTENT;
  } else if (state == St::REASONING) {
    ev.push_back(Event{Ev::REASONING_END, "", tool_index, 0});
    state = St::CONTENT;
  } else if (state == St::MESSAGE_HEADER) {
    state = St::CONTENT;
  }
  record(ev);
  return ev;
}

Events StreamingEngine::process_lex_tokens(std::vector<LexToken>& toks) {
  Events ev;
  bool strict = ever_had_token_ids_;
  for (auto& t : toks) {
    if (t.terminal == Tm::CONTENT || (strict && sh_.tid_name[(int)t.terminal])) append(ev, on_content(t.value, t.token_count));
    else append(ev, on_terminal(t.terminal, t.value, t.token_count));
  }
  return ev;
}

Events StreamingEngine::on_terminal(Tm terminal, const std::string& value, int token_count) {
  // dsml_roundtrip: an invoke / calls end tag inside an open parameter value is value text
  if ((terminal == Tm::INVOKE_END || terminal == Tm::TOOL_END) && state == St::TOOL_ARGS) {
    std::string raw;
    for (auto& s : v41_raw_) raw += s;
    if (param_open(raw)) return emit_for_state(value, token_count);
  }
  const Transition* tr = transition(state, terminal);
  const Derived& D = derived();
  if (!tr) {
    if (sh_.has_drops && terminal == Tm::DROP) return {};
    if (skip_tool_parsing && D.tool_exit_terminal[(int)terminal]) in_skipped_tool_span_ = false;
    return emit_for_state(value, token_count);
  }
  if (skip_reasoning_parsing && D.reasoning_markup[(int)terminal]) return emit_for_state(value, token_count);
  if (skip_tool_parsing && D.tool_terminal[(int)terminal]) {
    bool is_opener = is_tool_state(tr->next);
    bool is_exit = D.tool_exit_terminal[(int)terminal];
    bool plain_closer = is_exit && !is_opener && !in_skipped_tool_span_;
    if (!plain_closer) {
      if (is_opener) in_skipped_tool_span_ = true;
      else if (is_exit) in_skipped_tool_span_ = false;
      bool leaving_header = state == St::MESSAGE_HEADER;
      if (std::find(tr->events.begin(), tr->events.end(), Ev::REASONING_END) != tr->events.end()) {
        state = St::CONTENT;
        return {Event{Ev::REASONING_END, value, tool_index, 0}, Event{Ev::TEXT_CHUNK, value, tool_index, 0}};
      } else if (leaving_header) {
        state = St::CONTENT;
        return {Event{Ev::TEXT_CHUNK, value, tool_index, 0}};
      }
      auto ct = content_event(state);
      if (ct) return {Event{*ct, value, tool_index, 0}};
      return {};
    }
  }
  return apply_transition(*tr, value, token_count);
}

Events StreamingEngine::emit_for_state(const std::string& text, int token_count) {
  if (!text.empty() && state == St::TOOL_ARGS) v41_raw_.push_back(text);  // dsml_roundtrip _spe_emit_for_state
  if (state == St::MESSAGE_HEADER) return {};
  if (state == St::TOOL_ARGS) return {Event{Ev::ARG_VALUE_CHUNK, text, tool_index, token_count}};
  auto ct = content_event(state);
  if (ct) return {Event{*ct, text, tool_index, token_count}};
  return {};
}

Events StreamingEngine::on_content(const std::string& text, int token_count) {
  if (text.empty()) return {};
  return emit_for_state(text, token_count);
}

Events StreamingEngine::apply_transition(const Transition& tr, const std::string& value, int token_count) {
  Events ev;
  state = tr.next;
  for (Ev et : tr.events) {
    if (et == Ev::TOOL_CALL_START) tool_index += 1;
    if (et == Ev::TEXT_CHUNK && value.empty()) continue;
    ev.push_back(Event{et, value, tool_index, token_count});
  }
  if (state == St::TOOL_ARGS) v41_raw_.clear();  // dsml_roundtrip _spe_apply_transition: a new invoke body
  return ev;
}

// ---------------------------------------------------------------------------------------------------------- _Scan
namespace {
struct PM { std::string name; std::optional<std::string> is_str; std::string value; };

bool ws_at(std::string_view s, size_t i, size_t& l) {
  uint32_t cp = pyjson::decode_cp(s, i, l);
  return pyjson::re_space(cp);
}
size_t skip_ws(std::string_view s, size_t i) {
  size_t l;
  while (i < s.size() && ws_at(s, i, l)) i += l;
  return i;
}
// end anchor of "(.*?)<$" (DOTALL; $ also matches before a final "\n"): index of that "<", or npos
size_t lt_end(std::string_view c) {
  size_t n = c.size();
  if (n >= 2 && c[n - 2] == '<' && c[n - 1] == '\n') return n - 2;
  if (n >= 1 && c[n - 1] == '<') return n - 1;
  return std::string_view::npos;
}
// _PARAM_RE: ^ name="(.*?)" string="(true|false)">(.*?)<$
std::optional<PM> match_strict(std::string_view c) {
  static const std::string H = " name=\"", T = "\" string=\"true\">", F = "\" string=\"false\">";
  if (!starts_with(c, H)) return std::nullopt;
  size_t p = lt_end(c);
  if (p == std::string_view::npos) return std::nullopt;
  size_t jt = c.find(T, H.size()), jf = c.find(F, H.size());
  size_t j = std::min(jt, jf);
  if (j == std::string_view::npos) return std::nullopt;
  bool is_true = j == jt;
  size_t body = j + (is_true ? T.size() : F.size());
  if (body > p) return std::nullopt;
  return PM{std::string(c.substr(H.size(), j - H.size())), std::string(is_true ? "true" : "false"),
            std::string(c.substr(body, p - body))};
}
// head of _LOOSE_RE / _TAIL_RE: ^\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>   -> body start
bool match_head(std::string_view c, PM& m, size_t& body) {
  size_t i = skip_ws(c, 0);
  if (i == 0) return false;
  if (c.compare(i, 6, "name=\"") != 0) return false;
  size_t j = i + 6;
  size_t k = c.find('"', j);
  if (k == std::string_view::npos || k == j) return false;
  m.name = std::string(c.substr(j, k - j));
  size_t q0 = k + 1;
  size_t q = skip_ws(c, q0);
  if (q > q0 && c.compare(q, 8, "string=\"") == 0) {
    size_t r = c.find('"', q + 8);
    if (r != std::string_view::npos) {
      size_t t = skip_ws(c, r + 1);
      if (t < c.size() && c[t] == '>') {
        m.is_str = std::string(c.substr(q + 8, r - (q + 8)));
        body = t + 1;
        return true;
      }
    }
  }
  size_t t = skip_ws(c, q0);
  if (t < c.size() && c[t] == '>') {
    m.is_str.reset();
    body = t + 1;
    return true;
  }
  return false;
}
std::optional<PM> match_loose(std::string_view c) {
  PM m;
  size_t body;
  if (!match_head(c, m, body)) return std::nullopt;
  size_t p = lt_end(c);
  if (p == std::string_view::npos || body > p) return std::nullopt;
  m.value = std::string(c.substr(body, p - body));
  return m;
}
std::optional<PM> match_tail(std::string_view c) {
  PM m;
  size_t body;
  if (!match_head(c, m, body)) return std::nullopt;
  m.value = std::string(c.substr(body));
  return m;
}

std::string entry(const std::string& name, const std::optional<std::string>& is_str, const std::string& value,
                  bool strict) {
  if (is_str && *is_str == "true") return dump_str(name, false) + ": " + dump_str(value, false);
  if (is_str && *is_str == "false" && strict) return dump_str(name, false) + ": " + value;
  if (pyjson::loads_ok(value)) return dump_str(name, false) + ": " + value;
  return dump_str(name, false) + ": " + dump_str(value, false);
}
}  // namespace

void ParamScan::advance(std::string_view raw) {
  for (;;) {
    if (open_at < 0) {
      size_t s = raw.find(kPStart, pos);
      if (s == std::string_view::npos) {
        long np = (long)raw.size() - (long)kPStart.size() + 1;
        if (np > (long)pos) pos = (size_t)np;
        return;
      }
      open_at = (long)s;
      end_from = s + kPStart.size();
    }
    size_t e = raw.find(kPEnd, end_from);
    if (e == std::string_view::npos) {
      long ne = (long)raw.size() - (long)kPEnd.size() + 1;
      if (ne > (long)end_from) end_from = (size_t)ne;
      return;
    }
    size_t cs = (size_t)open_at + kPStart.size();
    std::string_view content = raw.substr(cs, e - cs);
    auto m = match_strict(content);
    bool strict = m.has_value();
    if (!m) m = match_loose(content);
    if (m && std::find(names.begin(), names.end(), m->name) == names.end()) {
      names.push_back(m->name);
      entries.push_back(entry(m->name, m->is_str, m->value, strict));
    }
    pos = e + kPEnd.size();
    open_at = -1;
  }
}

std::optional<std::string> ParamScan::tail_entry(std::string_view raw) const {
  if (open_at < 0) return std::nullopt;
  auto m = match_tail(raw.substr((size_t)open_at + kPStart.size()));
  if (!m || std::find(names.begin(), names.end(), m->name) != names.end()) return std::nullopt;
  std::string value = m->value;
  for (size_t k = kPClose.size() - 1; k > 0; --k) {
    if (ends_with(value, std::string_view(kPClose).substr(0, k))) {
      value.resize(value.size() - k);
      break;
    }
  }
  if (m->is_str && *m->is_str == "true") return dump_str(m->name, false) + ": " + dump_str(value, false);
  if (!pyjson::loads_ok(value)) return std::nullopt;
  return dump_str(m->name, false) + ": " + value;
}

static std::string join(const std::vector<std::string>& v, size_t from, const char* sep) {
  std::string o;
  for (size_t i = from; i < v.size(); ++i) {
    if (i > from) o += sep;
    o += v[i];
  }
  return o;
}

std::string official_arguments(std::string_view raw) {
  ParamScan st;
  st.advance(raw);
  std::vector<std::string> entries = st.entries;
  if (auto t = st.tail_entry(raw)) entries.push_back(*t);
  return "{" + join(entries, 0, ", ") + "}";
}

bool param_open(std::string_view raw) {
  ParamScan st;
  st.advance(raw);
  return st.open_at >= 0;
}

// ---------------------------------------------------------------------------------------------------------- ParserEngine
ParserEngine::ParserEngine(std::shared_ptr<const Shape> sh, std::shared_ptr<const ParserVocab> v, bool thinking,
                           std::function<std::string()>* idgen)
    : engine(*sh, *v, thinking ? St::REASONING : St::CONTENT), sh_(std::move(sh)), v_(std::move(v)),
      thinking_(thinking), idgen_(idgen) {}

void ParserEngine::initialize_streaming(std::optional<St> initial) {
  if (!streaming_initialized_) {
    streaming_initialized_ = true;
    reset(initial);
  }
}

void ParserEngine::reset(std::optional<St> initial) {
  engine.reset(initial);
  reasoning_ended = false;
  slots_.clear();
  deferred_content_.clear();
  content_has_nonws_ = false;
  rt_nl_.clear();
  rt_done_ = false;
  rt_stripped_ = false;
  v41_scan_.clear();
}

Delta ParserEngine::finish_streaming() {
  if (!rt_nl_.empty() && deferred_content_.empty()) {
    Events ev = engine.finish();
    return events_to_delta(ev, true);
  }
  Events ev = engine.finish();
  if (!ev.empty() || !deferred_content_.empty()) return events_to_delta(ev, true);
  return std::nullopt;
}

std::pair<std::optional<std::string>, std::optional<std::string>> ParserEngine::extract_reasoning(std::string_view text) {
  reset();
  Events ev = feed(text, {});
  append(ev, engine.finish());
  std::string r, c;
  for (auto& e : ev) {
    if (e.type == Ev::REASONING_CHUNK) r += e.value;
    else if (e.type == Ev::TEXT_CHUNK) c += e.value;
    else if (e.type == Ev::REASONING_END) reasoning_ended = true;
  }
  return {r.empty() ? std::nullopt : std::optional<std::string>(r),
          c.empty() ? std::nullopt : std::optional<std::string>(c)};
}

Delta ParserEngine::extract_reasoning_streaming(std::string_view delta_text, std::span<const int32_t> ids) {
  initialize_streaming();
  Events ev = feed(delta_text, ids);
  return events_to_delta(ev, false);
}

void ParserEngine::check_skip_tool_parsing(const ReqView& r) {
  // dsml_roundtrip v2.3: a request without tools never enters tool-call parsing (DSML spans stay plain content).
  // v2.4: tools + tool_choice "none" too (stock suppress_tool_calls_ dropped the block: neither content nor tool calls).
  // Set once per parser, before its first feed; the original check then runs unchanged (and never sets suppress).
  if ((!r.has_tools || r.tool_choice == ToolChoice::None) && !v41_notools_) {
    v41_notools_ = true;
    engine.skip_tool_parsing = true;
  }
  if (r.has_tools) has_tools_ = true;
  if (!engine.skip_tool_parsing && !suppress_tool_calls_)
    if (r.tool_choice == ToolChoice::None && r.has_tools) suppress_tool_calls_ = true;
}

ExtractedInfo ParserEngine::extract_tool_calls_from_content(std::string_view content, const ReqView& r) {
  check_skip_tool_parsing(r);
  SinglePass sp = single_pass_parse(content, {}, St::CONTENT);
  ExtractedInfo info = std::move(sp.info);
  if (sp.content && !info.content) info.content = sp.content;
  return info;
}

Delta ParserEngine::extract_tool_calls_streaming(std::string_view delta_text, std::span<const int32_t> ids,
                                                 const ReqView& r) {
  initialize_streaming();
  check_skip_tool_parsing(r);
  Events ev = feed(delta_text, ids);
  return events_to_delta(ev, false);
}

ParserEngine::SinglePass ParserEngine::single_pass_parse(std::string_view text, std::span<const int32_t> ids,
                                                         std::optional<St> initial) {
  reset(initial);
  Events ev = feed(text, ids);
  append(ev, engine.finish());
  Delta d = events_to_delta(ev, true);
  SinglePass sp;
  sp.info = build_extracted_result();
  if (d) sp.reasoning = d->reasoning;
  std::optional<std::string> content = d ? d->content : std::nullopt;
  if (content && !content->empty()) content = strip_content_whitespace(*content, sp.info.tools_called);
  sp.content = content;
  return sp;
}

bool ParserEngine::is_reasoning_end(std::span<const int32_t> ids) const {
  bool wait = thinking_;
  St initial = thinking_ ? St::REASONING : St::CONTENT;
  const Transition* st = transition(initial, Tm::THINK_START);
  bool start_opens = st && st->next == St::REASONING;
  for (size_t k = ids.size(); k-- > 0;) {
    int32_t id = ids[k];
    if (std::find(sh_->reasoning_end_ids.begin(), sh_->reasoning_end_ids.end(), id) != sh_->reasoning_end_ids.end())
      return true;
    if (sh_->think_start_id && id == *sh_->think_start_id) {
      if (start_opens) return false;
      break;
    }
  }
  return !wait;
}

std::vector<int32_t> ParserEngine::extract_content_ids(std::span<const int32_t> ids) const {
  if (!thinking_) return {ids.begin(), ids.end()};
  if (sh_->think_end_id) {
    for (size_t k = ids.size(); k-- > 0;)
      if (ids[k] == *sh_->think_end_id) return {ids.begin() + k + 1, ids.end()};
  }
  const auto& E = sh_->reasoning_end_ids;
  if (!E.empty()) {
    for (size_t k = 0; k < ids.size(); ++k)
      if (std::find(E.begin(), E.end(), ids[k]) != E.end()) return {ids.begin() + k, ids.end()};
  }
  return {ids.begin(), ids.end()};
}

std::optional<std::string> ParserEngine::strip_content_whitespace(std::string content, bool tools_called) {
  if (tools_called && !content.empty() && !rt_stripped_ && ends_with(content, "\n\n")) {
    content.resize(content.size() - 2);
    rt_stripped_ = true;
  }
  if (tools_called && py_is_blank(content)) content.clear();
  if (content.empty()) return std::nullopt;
  return content;
}

static Delta with_content(Delta d, const std::string& content) {
  if (!d) {
    if (content.empty()) return std::nullopt;
    ParserDelta x;
    x.present = true;
    x.set_content(content);
    return x;
  }
  d->set_content(content.empty() ? std::nullopt : std::optional<std::string>(content));
  if (!d->content && !d->reasoning && d->tool_calls.empty()) return std::nullopt;
  return d;
}

Delta ParserEngine::events_to_delta(Events& events, bool finished) {
  Delta d = events_to_delta_orig(events, finished);
  if (rt_done_) return d;
  std::string pending = rt_nl_;
  std::string content = d && d->content ? *d->content : std::string();
  bool tool = is_tool_state(engine.state);
  if (pending.empty() && content.empty() && !tool) return d;
  std::string combined = pending + content;
  rt_nl_.clear();
  if (tool) {
    rt_done_ = true;
    if (ends_with(combined, "\n\n")) {
      combined.resize(combined.size() - 2);
      rt_stripped_ = true;
    }
    if (py_is_blank(combined) && !content_has_nonws_) combined.clear();
  } else if (!finished && content_has_nonws_) {
    size_t e = combined.size();
    while (e > 0 && combined[e - 1] == '\n') --e;
    if (e < combined.size()) {
      rt_nl_ = combined.substr(e);
      combined.resize(e);
    }
  }
  return with_content(std::move(d), combined);
}

Slot& ParserEngine::slot(int idx) {
  if (idx < 0 || (size_t)idx >= slots_.size()) throw std::runtime_error("fe_parser: tool slot index out of range");
  return slots_[(size_t)idx];
}

void ParserEngine::ensure_slot(int idx) {
  while ((long)slots_.size() <= (long)idx) slots_.emplace_back();
}

void ParserEngine::ensure_tool_id(Slot& s) {
  if (s.id.empty()) s.id = (*idgen_)();
}

void ParserEngine::emit_name_delta(int idx, std::vector<ToolCallDelta>& deltas, const std::optional<std::string>& name) {
  if (!name || name->empty()) return;
  std::string nm = *name;
  Slot& s = slot(idx);
  s.name = nm;
  s.name_sent = true;
  ensure_tool_id(s);
  deltas.push_back(ToolCallDelta{idx, s.id, nm, std::nullopt});
}

void ParserEngine::handle_arg_chunk(const Event& ev, std::vector<ToolCallDelta>& deltas) {
  int idx = ev.tool_index;
  Slot& s = slot(idx);
  if (!ev.value.empty()) s.args += ev.value;
  if (!s.name_sent) {
    if (!s.name.empty()) emit_name_delta(idx, deltas, s.name);
    else if (!ev.value.empty()) emit_name_delta(idx, deltas, try_extract_name(idx));
  } else if (!ev.value.empty()) {
    auto ad = compute_arg_delta(idx, ev.value);
    if (ad && !ad->empty()) deltas.push_back(ToolCallDelta{idx, std::nullopt, std::nullopt, *ad});
  }
}

void ParserEngine::handle_tool_end(const Event& ev, std::vector<ToolCallDelta>& deltas) {
  int idx = ev.tool_index;
  if ((long)idx >= (long)slots_.size() || idx < 0) return;  // idx < 0 cannot occur (TOOL_CALL_END follows a start)
  std::optional<std::string> remaining = flush_arg_converter(idx);
  Slot& s = slot(idx);
  if (!s.name_sent) {
    std::string name = !s.name.empty() ? s.name : try_extract_name(idx).value_or("");
    if (!name.empty()) {
      s.name = name;
      s.name_sent = true;
      ensure_tool_id(s);
      deltas.push_back(ToolCallDelta{idx, s.id, name, remaining && !remaining->empty() ? *remaining : std::string()});
      remaining.reset();
    }
  }
  if (remaining && !remaining->empty() && s.name_sent)
    deltas.push_back(ToolCallDelta{idx, std::nullopt, std::nullopt, *remaining});
}

std::optional<std::string> ParserEngine::compute_arg_delta(int idx, const std::string& raw_delta) {
  if (raw_delta.find('>') == std::string::npos) return std::nullopt;
  Slot& s = slot(idx);
  ParamScan& st = v41_scan_[idx];
  if (st.closed) return std::nullopt;
  st.advance(s.args);
  if (st.emitted >= st.entries.size()) return std::nullopt;
  std::string diff = (st.emitted == 0 ? "{" : ", ") + join(st.entries, st.emitted, ", ");
  st.emitted = st.entries.size();
  s.streamed_json += diff;
  return diff;
}

std::optional<std::string> ParserEngine::flush_arg_converter(int idx) {
  Slot& s = slot(idx);
  ParamScan& st = v41_scan_[idx];
  if (st.closed) return std::nullopt;
  st.advance(s.args);
  std::vector<std::string> rest(st.entries.begin() + (long)std::min(st.emitted, st.entries.size()), st.entries.end());
  auto tail = st.tail_entry(s.args);
  if (tail) rest.push_back(*tail);
  std::string diff;
  if (st.emitted == 0) diff = "{" + join(rest, 0, ", ") + "}";
  else diff = (rest.empty() ? std::string() : ", " + join(rest, 0, ", ")) + "}";
  st.emitted = st.entries.size() + (tail ? 1 : 0);
  st.closed = true;
  s.streamed_json += diff;
  return diff;
}

std::optional<std::string> ParserEngine::try_extract_name(int idx) const {
  const std::string& a = slots_.at((size_t)idx).args;
  size_t from = 0;
  for (;;) {
    size_t p = a.find("\"name\"", from);
    if (p == std::string::npos) return std::nullopt;
    size_t q = skip_ws(a, p + 6);
    if (q < a.size() && a[q] == ':') {
      q = skip_ws(a, q + 1);
      if (q < a.size() && a[q] == '"') {
        size_t r = a.find('"', q + 1);
        if (r != std::string::npos) {
          std::string nm = a.substr(q + 1, r - q - 1);
          if (nm.empty()) return std::nullopt;
          return nm;
        }
      }
    }
    from = p + 1;
  }
}

static std::pair<std::string, std::string> extract_name_and_args(const std::string& raw_body) {
  std::string raw(pyjson::py_strip(raw_body));
  pyjson::Value p;
  if (!pyjson::loads(raw, p) || !p.is_obj()) return {"", raw};
  std::string name;
  if (const pyjson::Value* n = p.get("name")) {
    if (n->is_str()) name = n->s;
    else if (n->truthy())
      throw std::runtime_error("fe_parser: tool call name is not a string (Python raises a ValidationError)");
  }
  for (const char* key : {"arguments", "parameters"}) {
    if (const pyjson::Value* v = p.get(key)) {
      if (v->is_str()) return {name, v->s};
      return {name, pyjson::dumps(*v, false)};
    }
  }
  pyjson::Value without = pyjson::Value::object();
  for (size_t i = 0; i < p.keys.size(); ++i)
    if (p.keys[i] != "name") without.set(p.keys[i], p.items[i]);
  return {name, pyjson::dumps(without, false)};
}

ExtractedInfo ParserEngine::build_extracted_result() {
  ExtractedInfo info;
  for (size_t idx = 0; idx < slots_.size(); ++idx) {
    Slot& s = slots_[idx];
    if (s.name.empty() && s.args.empty()) continue;
    std::string name(pyjson::py_strip(s.name));
    const std::string& raw = s.args;
    std::string args_json;
    if (name.empty() && !py_is_blank(raw)) {
      auto na = extract_name_and_args(raw);
      name = na.first;
      args_json = na.second;
    } else if (!py_is_blank(raw)) {
      args_json = official_arguments(raw);
    } else {
      args_json = "{}";
    }
    if (!name.empty()) {
      ensure_tool_id(s);
      info.tool_calls.push_back(ToolCall{s.id, name, args_json});
    }
  }
  info.tools_called = !info.tool_calls.empty();
  info.content = std::nullopt;  // _strip_content_whitespace("") -> None
  return info;
}

static void coalesce(std::vector<ToolCallDelta>& deltas) {
  std::vector<ToolCallDelta> merged;
  std::map<int, size_t> pos;
  for (auto& tc : deltas) {
    auto it = pos.find(tc.index);
    if (it == pos.end()) {
      pos[tc.index] = merged.size();
      merged.push_back(tc);
      continue;
    }
    ToolCallDelta& ex = merged[it->second];
    if (tc.id && !ex.id) ex.id = tc.id;
    if (tc.name && !ex.name) ex.name = tc.name;
    if (tc.arguments) {
      if (!ex.arguments) ex.arguments = tc.arguments;
      else *ex.arguments += *tc.arguments;
    }
  }
  if (merged.size() != deltas.size()) deltas = std::move(merged);
}

Delta ParserEngine::events_to_delta_orig(Events& events, bool finished) {
  if (events.empty() && deferred_content_.empty()) return std::nullopt;
  std::vector<ToolCallDelta> tcd;
  std::vector<std::string> parts;
  std::string rparts;
  bool any_reasoning = false;
  std::string carried = deferred_content_;
  bool seen = false;
  bool suppress = suppress_tool_calls_;
  for (auto& e : events) {
    switch (e.type) {
      case Ev::TEXT_CHUNK:
        if (seen) deferred_content_ += e.value;
        else parts.push_back(e.value);
        break;
      case Ev::REASONING_CHUNK:
        rparts += e.value;
        any_reasoning = true;
        break;
      case Ev::REASONING_END: reasoning_ended = true; break;
      case Ev::TOOL_CALL_START:
        if (!suppress) { seen = true; ensure_slot(e.tool_index); }
        break;
      case Ev::TOOL_NAME:
        if (!suppress) { seen = true; slot(e.tool_index).name += e.value; }
        break;
      case Ev::ARG_VALUE_CHUNK:
        if (!suppress) { seen = true; handle_arg_chunk(e, tcd); }
        break;
      case Ev::TOOL_CALL_END:
        if (!suppress) { seen = true; handle_tool_end(e, tcd); }
        break;
      case Ev::REASONING_START: break;
    }
  }
  (void)any_reasoning;
  if (tcd.size() > 1) coalesce(tcd);
  if (!deferred_content_.empty() && (!seen || tcd.empty())) {
    std::string now = deferred_content_.substr(std::min(carried.size(), deferred_content_.size()));
    if (!carried.empty()) parts.insert(parts.begin(), carried);
    if (!now.empty()) parts.push_back(now);
    deferred_content_.clear();
  }
  std::string content_str;
  for (auto& p : parts) content_str += p;
  if (content_has_nonws_) {
  } else if (!content_str.empty()) {
    if (!py_is_blank(content_str)) content_has_nonws_ = true;
    else if (!slots_.empty()) content_str.clear();
    else if (!finished) { deferred_content_ = content_str; content_str.clear(); }
  }
  if (!content_str.empty() || !tcd.empty() || !rparts.empty()) {
    ParserDelta d;
    d.present = true;
    if (!content_str.empty()) d.set_content(content_str);
    if (!rparts.empty()) d.set_reasoning(rparts);
    d.tool_calls = std::move(tcd);
    return d;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------- adapters
ReasoningAdapter::ReasoningAdapter(std::shared_ptr<const Shape> sh, std::shared_ptr<const ParserVocab> v, bool thinking,
                                   std::function<std::string()>* idgen)
    : sh_(std::move(sh)), v_(std::move(v)), thinking_(thinking), idgen_(idgen) {
  engine_ = make();
}

std::unique_ptr<ParserEngine> ReasoningAdapter::make() const {
  return std::make_unique<ParserEngine>(sh_, v_, thinking_, idgen_);
}

namespace {
struct SkipTools {
  StreamingEngine& e;
  bool saved;
  explicit SkipTools(StreamingEngine& x) : e(x), saved(x.skip_tool_parsing) { e.skip_tool_parsing = true; }
  ~SkipTools() { e.skip_tool_parsing = saved; }
};
}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>> ReasoningAdapter::extract_reasoning(std::string_view text) {
  streaming_count_valid_ = false;
  SkipTools g(engine_->engine);
  return engine_->extract_reasoning(text);
}

Delta ReasoningAdapter::extract_reasoning_streaming(std::string_view delta_text, std::span<const int32_t> ids) {
  streaming_count_valid_ = true;
  SkipTools g(engine_->engine);
  return engine_->extract_reasoning_streaming(delta_text, ids);
}

Delta ReasoningAdapter::finish_streaming() {
  SkipTools g(engine_->engine);
  return engine_->finish_streaming();
}

int ReasoningAdapter::count_after_prompt_end(std::span<const int32_t> ids, bool finished) {
  if (!incremental_ || ids.size() < incremental_counted_ || (incremental_flushed_ && ids.size() > incremental_counted_)) {
    if (!incremental_) incremental_ = make();
    incremental_->reset(St::CONTENT);
    incremental_counted_ = 0;
    incremental_flushed_ = false;
  }
  auto new_ids = ids.subspan(incremental_counted_);
  if (!new_ids.empty()) {
    incremental_->feed(v_->decode(new_ids), new_ids);
    incremental_counted_ = ids.size();
  }
  if (finished && !incremental_flushed_) {
    incremental_->engine.finish();
    incremental_flushed_ = true;
  }
  return incremental_->count_reasoning_tokens();
}

int ReasoningAdapter::count_reasoning_tokens(std::span<const int32_t> ids) {
  if (streaming_count_valid_) return engine_->count_reasoning_tokens();
  if (ids.empty()) return 0;
  if (!counting_) counting_ = make();
  counting_->single_pass_parse(v_->decode(ids), ids, std::nullopt);
  return counting_->count_reasoning_tokens();
}

}  // namespace fe::dsml
