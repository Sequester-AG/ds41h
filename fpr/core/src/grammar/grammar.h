// grammar.h -- structured outputs on xgrammar's C++ API, with vLLM's semantics.
//
// One-to-one with the installed vLLM (vllm/v1/structured_output/__init__.py StructuredOutputManager,
// backend_xgrammar.py XgrammarBackend / XgrammarGrammar, request.py StructuredOutputRequest) on the same xgrammar
// (0.2.8, commit 97787376f, built from source as a static library: tests/Makefile):
//   * compile_grammar: JSON (any_whitespace = !disable_any_whitespace, separators (",", ":") when disabled),
//     JSON_OBJECT ('{"type": "object"}'), GRAMMAR (lark if grammar_is_likely_lark, else EBNF), REGEX, STRUCTURAL_TAG
//     (the legacy {"structures", "triggers"} form is rejected: it needs xgrammar's Python-side conversion);
//     GrammarMatcher(override_stop_tokens = all_stop_token_ids or None, max_rollback_tokens = num_spec_tokens).
//   * the per-step bitmask rows of the decode group (num_spec + 1 rows per request, -1 padding, grammar / constraint
//     bounds from the reasoning parser, failed-index copies, rollback of the probe advances), validate_tokens (drafts:
//     the longest valid prefix after the reasoning bound), accept_tokens (reasoning_ended latch).
//   * the reasoning bounds (_get_constraint_bounds) for vLLM's ParserEngineReasoningAdapter family (deepseek_v41): a
//     token-id-only rule set (ReasonerSpec) dumped once from Python by tests/gen_grammar.py (reasoner_spec()).
// Bitmask rows are int32 words, bit (t & 31) of word (t >> 5) = token t allowed (xgrammar's layout); -1 = all allowed.
// Grammar compilation runs on a thread pool (vLLM: ThreadPoolExecutor); is_ready() polls, as the scheduler does.
// No exceptions leave this module: xgrammar's are caught at the call and turned into error strings.
#pragma once
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../v1wire/v1wire.h"

namespace fpr {
namespace grammar {

using v1::SOType;

struct ReasonerSpec {                  // ParserEngine's token-level reasoning facts (one reasoning_parser_kwargs variant)
  bool present = false;                // a reasoner exists for the request (reasoning parser set)
  bool adapter = true;                 // ParserEngineReasoningAdapter (find_reasoning_end path)
  std::vector<int32_t> end_ids;        // _reasoning_end_token_ids
  int32_t end_token_id = -1;           // _reasoning_end_token_id (THINK_END), -1 = None
  int32_t start_id = -1;               // _reasoning_start_token_id, -1 = None
  std::vector<int32_t> boundary_ids;   // _turn_boundary_token_ids
  bool wait_for_reasoning = true;
  bool start_opens_reasoning = false;

  bool is_end_id(int32_t t) const;
  bool is_boundary(int32_t t) const;
  bool is_reasoning_end(const std::vector<int32_t>& ids) const;               // ParserEngine.is_reasoning_end
  bool find_reasoning_end(const int32_t* ids, size_t n, size_t* offset, bool* implicit) const;   // false = None
};

class Grammar {                         // XgrammarGrammar
 public:
  virtual ~Grammar() = default;
  virtual bool accept_tokens(const std::vector<int32_t>& tokens) = 0;
  virtual std::vector<int32_t> validate_tokens(const std::vector<int32_t>& tokens) = 0;
  virtual void rollback(int n) = 0;
  virtual void fill_bitmask(int32_t* row) = 0;      // one row of bitmask_words() int32
  virtual bool is_terminated() const = 0;
  virtual void reset() = 0;
  virtual int64_t num_processed_tokens() const = 0;
};

struct BackendConfig {
  std::string tokenizer_info_json;     // xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size).serialize_json()
  int32_t vocab_size = 0;              // model_config.get_vocab_size()
  int32_t num_spec_tokens = 0;         // max_rollback_tokens
  bool disable_any_whitespace = false; // structured_outputs_config.disable_any_whitespace
  int64_t cache_limit_bytes = 512ll << 20;   // VLLM_XGRAMMAR_CACHE_MB (default 512)
  int compile_threads = 8;             // GrammarCompiler max_threads
  int pool_threads = 4;                // the grammar_init executor
};

class Backend {                         // XgrammarBackend
 public:
  virtual ~Backend() = default;
  // nullptr + *err on failure
  virtual std::shared_ptr<Grammar> compile(SOType type, const std::string& spec, const std::vector<int32_t>& stop_ids,
                                           std::string* err) = 0;
  virtual int32_t vocab_size() const = 0;
  int32_t bitmask_words() const { return (vocab_size() + 31) / 32; }
};
std::unique_ptr<Backend> make_xgrammar_backend(const BackendConfig& cfg, std::string* err);

bool grammar_is_likely_lark(const std::string& g);   // vllm/v1/structured_output/utils.py

// StructuredOutputRequest: the compiled grammar (future), the reasoning latch and the request's reasoner
struct StructuredReq {
  SOType type = SOType::NONE;
  std::string spec;
  std::vector<int32_t> stop_ids;       // all_stop_token_ids
  int reasoning_ended = -1;            // None / False / True
  const ReasonerSpec* reasoner = nullptr;   // nullptr = no reasoner (vLLM: reasoner_cls None)
  // compile result
  std::mutex mu;
  bool done = false;
  std::shared_ptr<Grammar> grammar;
  std::string error;                   // non-empty = the compile raised
  bool ready() { std::lock_guard<std::mutex> g(mu); return done; }
};

// one request's view for the manager (the vLLM Request fields it reads)
struct ReqView {
  StructuredReq* so;
  const std::vector<int32_t>* prompt;  // prompt_token_ids
  const std::vector<int32_t>* all;     // all_token_ids
};

class Manager {                         // StructuredOutputManager (xgrammar backend)
 public:
  Manager(std::unique_ptr<Backend> backend, int32_t num_spec_tokens, bool enable_in_reasoning, bool async_compile = true);
  ~Manager();
  Backend* backend() { return backend_.get(); }
  // grammar_init: compile on the pool (or inline when async_compile is false)
  void grammar_init(const std::shared_ptr<StructuredReq>& so);
  // _get_constraint_bounds
  void constraint_bounds(const ReqView& r, const std::vector<int32_t>& spec_tokens, bool committed, size_t* grammar_start,
                         size_t* constraint_start);
  std::vector<int32_t> validate_tokens(const ReqView& r, const std::vector<int32_t>& spec_tokens);
  bool accept_tokens(const ReqView& r, const std::vector<int32_t>& new_token_ids);
  // grammar_bitmask: rows for `reqs` (structured_output_request_ids order), spec[i] = scheduled_spec_decode_tokens of
  // reqs[i] (empty if none). Appends rows to *out (words per row = backend()->bitmask_words()); returns the rows written.
  int64_t grammar_bitmask(const std::vector<ReqView>& reqs, const std::vector<const std::vector<int32_t>*>& spec,
                          std::vector<int32_t>* out);

 private:
  void worker();
  void compile_one(const std::shared_ptr<StructuredReq>& so);
  std::unique_ptr<Backend> backend_;
  int32_t k_;
  bool enable_in_reasoning_;
  bool async_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<StructuredReq>> q_;
  std::vector<std::thread> pool_;
  bool stop_ = false;
};

std::vector<int32_t> strip_speculative_padding(const std::vector<int32_t>& t);

}  // namespace grammar
}  // namespace fpr
