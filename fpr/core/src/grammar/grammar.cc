// grammar.cc -- see grammar.h
#include "grammar.h"

#include <dlpack/dlpack.h>
#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <regex>
#include <sstream>

namespace fpr {
namespace grammar {

// ------------------------------------------------------------------ helpers
std::vector<int32_t> strip_speculative_padding(const std::vector<int32_t>& t) {
  for (size_t i = 0; i < t.size(); ++i)
    if (t[i] < 0) return std::vector<int32_t>(t.begin(), t.begin() + i);
  return t;
}

bool grammar_is_likely_lark(const std::string& g) {
  if (g.empty()) return false;
  static const std::regex comment("(#|//).*$");
  std::stringstream ss(g);
  std::string line;
  // Python str.split("\n"): a trailing "\n" yields a last empty line (skipped either way)
  while (std::getline(ss, line, '\n')) {
    line = std::regex_replace(line, comment, "");
    size_t a = line.find_first_not_of(" \t\r\f\v");
    if (a == std::string::npos) continue;
    if (line.find("::=") != std::string::npos) return false;
  }
  return true;
}

bool ReasonerSpec::is_end_id(int32_t t) const {
  return std::find(end_ids.begin(), end_ids.end(), t) != end_ids.end();
}
bool ReasonerSpec::is_boundary(int32_t t) const {
  return std::find(boundary_ids.begin(), boundary_ids.end(), t) != boundary_ids.end();
}
bool ReasonerSpec::is_reasoning_end(const std::vector<int32_t>& ids) const {
  for (size_t j = ids.size(); j-- > 0;) {
    int32_t t = ids[j];
    if (is_end_id(t)) return true;
    if (start_id >= 0 && t == start_id) {
      if (start_opens_reasoning) return false;
      break;
    }
    if (is_boundary(t)) break;
  }
  return !wait_for_reasoning;
}
bool ReasonerSpec::find_reasoning_end(const int32_t* ids, size_t n, size_t* offset, bool* implicit) const {
  if (end_ids.empty()) return false;
  for (size_t j = 0; j < n; ++j) {
    if (is_end_id(ids[j])) {
      *offset = j;
      *implicit = ids[j] != end_token_id;
      return true;
    }
  }
  *offset = n;
  *implicit = false;
  return true;
}

// ------------------------------------------------------------------ xgrammar backend
namespace {

class XGrammar final : public Grammar {
 public:
  XGrammar(xgrammar::GrammarMatcher m, int32_t vocab, int32_t words) : m_(std::move(m)), vocab_(vocab), words_(words) {}
  bool accept_tokens(const std::vector<int32_t>& tokens) override {
    if (term_) return true;
    for (int32_t t : tokens) {
      if (!m_.AcceptToken(t)) return false;
      ++processed_;
      term_ = m_.IsTerminated();
      if (term_) break;
    }
    return true;
  }
  std::vector<int32_t> validate_tokens(const std::vector<int32_t>& tokens) override {
    std::vector<int32_t> acc;
    if (term_) return acc;
    for (int32_t t : tokens) {
      if (m_.AcceptToken(t)) {
        acc.push_back(t);
        if (m_.IsTerminated()) break;
      } else {
        break;
      }
    }
    if (!acc.empty()) m_.Rollback((int)acc.size());
    return acc;
  }
  void rollback(int n) override {
    m_.Rollback(n);
    processed_ -= n;
    term_ = m_.IsTerminated();
  }
  void fill_bitmask(int32_t* row) override {
    int64_t shape[2] = {1, words_};
    DLTensor t;
    t.data = row;
    t.device = DLDevice{kDLCPU, 0};
    t.ndim = 2;
    t.dtype = xgrammar::GetBitmaskDLType();
    t.shape = shape;
    t.strides = nullptr;
    t.byte_offset = 0;
    m_.FillNextTokenBitmask(&t, 0);
  }
  bool is_terminated() const override { return term_; }
  void reset() override {
    m_.Reset();
    processed_ = 0;
    term_ = false;
  }
  int64_t num_processed_tokens() const override { return processed_; }

 private:
  xgrammar::GrammarMatcher m_;
  int32_t vocab_, words_;
  int64_t processed_ = 0;
  bool term_ = false;
};

// legacy {"structures": [{begin, schema, end}], "triggers": [...]} -> the structural-tag JSON xgrammar's
// StructuralTag.from_legacy_structural_tag builds (semantically; pydantic's extra default fields are xgrammar's
// C++ defaults too)
bool legacy_tag(const picojson::value& v, std::string* out, std::string* err) {
  const auto& o = v.get<picojson::object>();
  picojson::array tags;
  for (const auto& s : o.at("structures").get<picojson::array>()) {
    const auto& so = s.get<picojson::object>();
    picojson::object content;
    content["type"] = picojson::value("json_schema");
    content["json_schema"] = so.at("schema");     // vLLM json.dumps it, xgrammar json.loads it back: the same value
    picojson::object tag;
    tag["type"] = picojson::value("tag");
    tag["begin"] = so.at("begin");
    tag["content"] = picojson::value(content);
    tag["end"] = so.at("end");
    tags.push_back(picojson::value(tag));
  }
  picojson::object fmt;
  fmt["type"] = picojson::value("triggered_tags");
  fmt["triggers"] = o.at("triggers");
  fmt["tags"] = picojson::value(tags);
  picojson::object st;
  st["type"] = picojson::value("structural_tag");
  st["format"] = picojson::value(fmt);
  *out = picojson::value(st).serialize();
  (void)err;
  return true;
}

class XBackend final : public Backend {
 public:
  XBackend(const BackendConfig& c, xgrammar::TokenizerInfo ti)
      : cfg_(c), ti_(ti), comp_(ti, c.compile_threads, true, c.cache_limit_bytes) {}
  int32_t vocab_size() const override { return cfg_.vocab_size; }

  std::shared_ptr<Grammar> compile(SOType type, const std::string& spec, const std::vector<int32_t>& stop_ids,
                                   std::string* err) override {
    try {
      std::optional<std::pair<std::string, std::string>> seps;
      if (cfg_.disable_any_whitespace) seps = std::make_pair(std::string(","), std::string(":"));
      bool anyws = !cfg_.disable_any_whitespace;
      std::optional<xgrammar::CompiledGrammar> ctx;
      switch (type) {
        case SOType::JSON: ctx = comp_.CompileJSONSchema(spec, anyws, std::nullopt, seps, true, std::nullopt, false); break;
        case SOType::JSON_OBJECT:
          ctx = comp_.CompileJSONSchema("{\"type\": \"object\"}", anyws, std::nullopt, seps, true, std::nullopt, false);
          break;
        case SOType::GRAMMAR:
          if (grammar_is_likely_lark(spec)) ctx = comp_.CompileLark(spec);
          else ctx = comp_.CompileGrammar(spec, "root");
          break;
        case SOType::REGEX: ctx = comp_.CompileRegex(spec); break;
        case SOType::STRUCTURAL_TAG: {
          picojson::value v;
          std::string perr = picojson::parse(v, spec);
          if (!perr.empty()) {
            *err = "structural tag: invalid JSON: " + perr;
            return nullptr;
          }
          if (v.is<picojson::object>() && v.get<picojson::object>().count("structures")) {
            std::string conv;
            if (!legacy_tag(v, &conv, err)) return nullptr;
            ctx = comp_.CompileStructuralTag(conv);
          } else {
            ctx = comp_.CompileStructuralTag(spec);
          }
          break;
        }
        default:
          *err = "grammar is not of valid supported types";
          return nullptr;
      }
      std::optional<std::vector<int>> stops;
      if (!stop_ids.empty()) stops = std::vector<int>(stop_ids.begin(), stop_ids.end());
      xgrammar::GrammarMatcher m(*ctx, stops, false, cfg_.num_spec_tokens);
      return std::make_shared<XGrammar>(std::move(m), cfg_.vocab_size, bitmask_words());
    } catch (const std::exception& e) {
      *err = e.what();
      return nullptr;
    } catch (...) {
      *err = "xgrammar: unknown error";
      return nullptr;
    }
  }

 private:
  BackendConfig cfg_;
  xgrammar::TokenizerInfo ti_;
  xgrammar::GrammarCompiler comp_;
};

}  // namespace

std::unique_ptr<Backend> make_xgrammar_backend(const BackendConfig& cfg, std::string* err) {
  try {
    auto r = xgrammar::TokenizerInfo::DeserializeJSON(cfg.tokenizer_info_json);
    if (std::holds_alternative<xgrammar::SerializationError>(r)) {
      *err = "tokenizer info: cannot deserialize";
      return nullptr;
    }
    return std::unique_ptr<Backend>(new XBackend(cfg, std::get<xgrammar::TokenizerInfo>(r)));
  } catch (const std::exception& e) {
    *err = e.what();
    return nullptr;
  }
}

// ------------------------------------------------------------------ manager
Manager::Manager(std::unique_ptr<Backend> backend, int32_t k, bool enable_in_reasoning, bool async_compile)
    : backend_(std::move(backend)), k_(k), enable_in_reasoning_(enable_in_reasoning), async_(async_compile) {
  if (async_)
    for (int i = 0; i < 4; ++i) pool_.emplace_back([this] { worker(); });
}

Manager::~Manager() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : pool_) t.join();
}

void Manager::compile_one(const std::shared_ptr<StructuredReq>& so) {
  std::string err;
  auto g = backend_->compile(so->type, so->spec, so->stop_ids, &err);
  std::lock_guard<std::mutex> l(so->mu);
  so->grammar = g;
  so->error = g ? "" : (err.empty() ? "grammar compile failed" : err);
  so->done = true;
}

void Manager::worker() {
  for (;;) {
    std::shared_ptr<StructuredReq> so;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [this] { return stop_ || !q_.empty(); });
      if (stop_ && q_.empty()) return;
      so = q_.front();
      q_.pop_front();
    }
    compile_one(so);
  }
}

void Manager::grammar_init(const std::shared_ptr<StructuredReq>& so) {
  if (!async_) {
    compile_one(so);
    return;
  }
  {
    std::lock_guard<std::mutex> g(mu_);
    q_.push_back(so);
  }
  cv_.notify_one();
}

void Manager::constraint_bounds(const ReqView& r, const std::vector<int32_t>& spec, bool committed, size_t* gs,
                                size_t* cs) {
  auto set = [&](size_t a, size_t b) { *gs = a; *cs = b; };
  if (enable_in_reasoning_) return set(0, 0);
  StructuredReq* so = r.so;
  if (so->reasoning_ended == 1) return set(0, 0);
  const ReasonerSpec* rs = so->reasoner;
  if (rs == nullptr || !rs->present) return set(0, 0);
  if (so->reasoning_ended < 0) {
    if (rs->is_reasoning_end(*r.prompt)) return set(0, 0);
    so->reasoning_ended = 0;
  }
  size_t n = spec.size();
  if (n == 0) return set(1, 1);
  if (rs->adapter && !rs->end_ids.empty()) {
    size_t off;
    bool impl;
    if (rs->find_reasoning_end(spec.data(), n, &off, &impl)) {
      size_t c = off + 1;
      if (impl) return set(off, c);
      return set(c, c);
    }
  }
  std::vector<int32_t> input(*r.all);
  if (committed) {
    if (!rs->is_reasoning_end(input)) return set(n + 1, n + 1);   // is_reasoning_end_streaming = is_reasoning_end
  } else {
    input.insert(input.end(), spec.begin(), spec.end());
    if (!rs->is_reasoning_end(input)) return set(n + 1, n + 1);
  }
  for (size_t i = n - 1; i > 0; --i) {
    input.pop_back();
    if (!rs->is_reasoning_end(input)) return set(i + 1, i + 1);
  }
  return set(1, 1);
}

std::vector<int32_t> Manager::validate_tokens(const ReqView& r, const std::vector<int32_t>& spec_in) {
  std::vector<int32_t> spec = strip_speculative_padding(spec_in);
  size_t gs, cs;
  constraint_bounds(r, spec, false, &gs, &cs);
  if (gs >= spec.size()) return spec;
  std::vector<int32_t> out(spec.begin(), spec.begin() + gs);
  std::vector<int32_t> tail(spec.begin() + gs, spec.end());
  auto v = r.so->grammar->validate_tokens(tail);
  out.insert(out.end(), v.begin(), v.end());
  return out;
}

bool Manager::accept_tokens(const ReqView& r, const std::vector<int32_t>& toks) {
  if (toks.empty()) return true;
  size_t gs, cs;
  constraint_bounds(r, toks, true, &gs, &cs);
  if (gs > toks.size()) return true;
  r.so->reasoning_ended = 1;
  return r.so->grammar->accept_tokens(std::vector<int32_t>(toks.begin() + gs, toks.end()));
}

int64_t Manager::grammar_bitmask(const std::vector<ReqView>& reqs, const std::vector<const std::vector<int32_t>*>& spec,
                                 std::vector<int32_t>* out) {
  const int32_t W = backend_->bitmask_words();
  const size_t base = out->size();
  int64_t rows = 0;
  auto row = [&](int64_t i) { return out->data() + base + (size_t)i * W; };
  auto add_row = [&]() {
    out->resize(base + (size_t)(rows + 1) * W);
    return rows++;
  };
  auto fill = [&](Grammar* g, int64_t i, bool apply) {
    if (apply && !g->is_terminated()) g->fill_bitmask(row(i));
    else std::fill(row(i), row(i) + W, -1);
  };
  static const std::vector<int32_t> none;
  for (size_t q = 0; q < reqs.size(); ++q) {
    const ReqView& r = reqs[q];
    Grammar* g = r.so->grammar.get();
    const std::vector<int32_t>& toks = spec[q] ? *spec[q] : none;
    size_t gs, cs;
    constraint_bounds(r, strip_speculative_padding(toks), false, &gs, &cs);
    int adv = 0;
    bool seen_padding = false;
    int64_t failed = -1;
    for (size_t i = 0; i < toks.size(); ++i) {
      int64_t cur = add_row();
      if (failed >= 0) {
        std::copy(row(failed), row(failed) + W, row(cur));
        continue;
      }
      bool apply = !seen_padding && i >= cs;
      fill(g, cur, apply);
      int32_t t = toks[i];
      if (t == -1) {
        seen_padding = true;
      } else if (!seen_padding && i >= gs) {
        if (!g->is_terminated() && g->accept_tokens({t})) ++adv;
        else failed = cur;
      }
    }
    int64_t cur = add_row();
    if (failed >= 0) {
      std::copy(row(failed), row(failed) + W, row(cur));
    } else {
      bool bonus = !seen_padding && cs <= toks.size();
      fill(g, cur, bonus);
    }
    if (adv > 0) g->rollback(adv);
  }
  return rows;
}

}  // namespace grammar
}  // namespace fpr
