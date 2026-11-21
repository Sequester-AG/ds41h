// Internals of fe_tokenizer. Exposed for tests/tokenizer only; other components use tokenizer.h.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace fe::tok {

// ---- Unicode classes (tables probed from the Oniguruma inside HF tokenizers; == vendored onig 6.9.10) ----
enum : uint16_t {
  kL = 1u << 0,           // \p{L}
  kM = 1u << 1,           // \p{M}
  kN = 1u << 2,           // \p{N}
  kP = 1u << 3,           // \p{P}
  kS = 1u << 4,           // \p{S}
  kWS = 1u << 5,          // \s
  kCJK = 1u << 6,         // [一-龥぀-ゟ゠-ヿ]
  kW = 1u << 7,           // Rust regex \w (AddedVocabulary single_word; != onig \w: superscripts, ZWJ/ZWNJ)
  kAsciiAlpha = 1u << 8,  // [A-Za-z]
  kAsciiPunct = 1u << 9,  // [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]
  kNL = 1u << 10,         // [\r\n]
  kSP = 1u << 11,         // ' '
  kRS = 1u << 12,         // Rust regex \s (AddedVocabulary lstrip / rstrip; == onig \s)
};
uint16_t cp_class(uint32_t cp);
const uint16_t* ascii_class_table();  // 128 entries

// Valid UTF-8 (Rust str rules: no surrogates, no overlongs, <= U+10FFFF).
bool utf8_valid(const char* s, size_t n);
// Rust String::from_utf8_lossy.
std::string utf8_lossy(std::string_view bytes);

// ---- Leftmost-longest trie (== aho-corasick MatchKind::LeftmostLongest, non-overlapping find_iter) ----
class Trie {
 public:
  void build(const std::vector<std::string>& patterns);
  bool empty() const { return npat_ == 0; }
  // First match starting at or after pos. Returns false if none.
  bool find(const uint8_t* s, size_t n, size_t pos, size_t& start, size_t& end, int32_t& pat) const;

 private:
  struct Node {
    int32_t term = -1;
    uint32_t first = 0;
    uint32_t count = 0;
  };
  std::vector<Node> nodes_;
  std::vector<uint8_t> ebyte_;
  std::vector<uint32_t> enode_;
  uint32_t root_[256] = {};  // child of root per byte (0 = none)
  size_t npat_ = 0;
};

struct AddedTok {
  std::string content;
  bool single_word = false, lstrip = false, rstrip = false, normalized = false, special = false;
  bool same(const AddedTok& o) const {
    return content == o.content && single_word == o.single_word && lstrip == o.lstrip && rstrip == o.rstrip &&
           normalized == o.normalized && special == o.special;
  }
};

// ---- BPE merges: (a, b) -> (rank, new id), open addressing ----
class MergeMap {
 public:
  void reserve(size_t n);
  void put(uint32_t a, uint32_t b, uint32_t rank, uint32_t nid);  // overwrites (HashMap collect semantics)
  inline bool get(uint32_t a, uint32_t b, uint32_t& rank, uint32_t& nid) const {
    uint64_t k = (uint64_t(a) << 32) | b;
    size_t h = size_t((k * 0x9E3779B97F4A7C15ull) >> shift_);
    for (;;) {
      const Slot& s = slots_[h];
      if (s.key == k) {
        rank = s.rank;
        nid = s.nid;
        return true;
      }
      if (s.key == kEmpty) return false;
      h = (h + 1) & mask_;
    }
  }

 private:
  static constexpr uint64_t kEmpty = ~0ull;
  struct Slot {
    uint64_t key = kEmpty;
    uint32_t rank = 0, nid = 0;
  };
  std::vector<Slot> slots_;
  size_t mask_ = 0;
  int shift_ = 64;
  size_t count_ = 0;
};

// ---- small fixed pool for chunk-parallel encode ----
class TaskPool {
 public:
  explicit TaskPool(int nthreads);
  ~TaskPool();
  int size() const { return int(threads_.size()) + 1; }
  // Runs f(0..n-1); the caller participates. Rethrows the first exception.
  void run(size_t n, const std::function<void(size_t)>& f);

 private:
  struct Job {
    const std::function<void(size_t)>* f = nullptr;
    size_t n = 0;
    std::atomic<size_t> next{0}, done{0};
    std::mutex m;
    std::condition_variable cv;
    std::exception_ptr err;
  };
  void work_on(Job& j);
  void loop();
  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> q_;
  bool stop_ = false;
};

struct Split {
  int32_t id;  // -1: plain text
  size_t a, b;
};

}  // namespace fe::tok

namespace fe {

class TokenizerImpl {
 public:
  void load(std::string_view tokenizer_json, std::string_view config_json);

  // Whole-string HF encode (added vocabulary + pre-tokenizer + BPE), appends to out. Text must be valid UTF-8.
  void encode_text(std::string_view text, std::vector<int32_t>& out) const;
  // Pre-tokenizer pieces (byte ranges) of a plain segment (no added tokens): fast path or onig.
  void pretokenize(std::string_view seg, std::vector<std::pair<uint32_t, uint32_t>>& out) const;
  void pretokenize_onig(std::string_view seg, std::vector<std::pair<uint32_t, uint32_t>>& out) const;
  bool has_fast_pretok() const { return fast_pretok_; }
  bool has_onig_pretok() const { return !regexes_.empty(); }
  // BPE of one pre-token (raw bytes). word_naive / word_heap are the two merge algorithms (tests compare them).
  void bpe(const uint8_t* p, size_t n, std::vector<int32_t>& out) const;  // with cache
  void bpe_naive(const uint8_t* p, size_t n, std::vector<int32_t>& out) const;
  void bpe_heap(const uint8_t* p, size_t n, std::vector<int32_t>& out) const;
  // AddedVocabulary find_matches on one pass (normalized=false: raw-text pass; true: second pass).
  void find_added(std::string_view s, bool normalized_pass, std::vector<tok::Split>& out) const;

  std::string decode(const int32_t* ids, size_t n, bool skip_special) const;
  void decode_raw_append(const int32_t* ids, size_t n, bool skip_special, std::string& raw) const;

  // vocab
  std::unordered_map<std::string, int32_t> vocab_;        // BPE model vocab
  std::unordered_map<std::string, int32_t> added_map_;    // added_tokens_map (content -> id)
  std::unordered_map<int32_t, tok::AddedTok> added_r_;    // added_tokens_map_r (id -> token)
  std::vector<tok::AddedTok> special_tokens_, added_tokens_;
  std::unordered_map<std::string, bool> special_set_;
  int32_t vocab_size_with_added_ = 0, max_id_ = -1, model_vocab_size_ = 0;

  // decoder table, index = id (0..max_id_)
  struct DecEntry {
    uint32_t off = 0;
    int32_t len = -1;  // -1: no token for this id
    bool special = false;
  };
  std::vector<DecEntry> dec_;
  std::string dec_arena_;
  std::vector<uint32_t> tok_off_;  // id_to_token strings (as stored)
  std::string tok_arena_;

  // added vocabulary tries
  tok::Trie trie_raw_, trie_norm_;
  std::vector<int32_t> trie_raw_ids_, trie_norm_ids_;

  // pre-tokenizer
  bool fast_pretok_ = false;
  std::vector<void*> regexes_;  // onig regex_t* per Split stage
  ~TokenizerImpl();

  // BPE
  int32_t byte_id_[256];
  tok::MergeMap merges_;
  bool ignore_merges_ = false;
  uint64_t instance_id_ = 0;

  // pool
  mutable std::mutex pool_m_;
  mutable std::shared_ptr<tok::TaskPool> pool_;
  int parallelism_ = 1;
  std::shared_ptr<tok::TaskPool> get_pool() const;

 private:
  template <class F>
  void stage1(const uint16_t* cls, uint32_t a, uint32_t b, F&& f) const;
  template <class F>
  void stage2(const uint16_t* cls, uint32_t a, uint32_t b, F&& f) const;
  template <class F>
  void stage3(const uint16_t* cls, uint32_t a, uint32_t b, F&& f) const;
  void encode_plain(std::string_view seg, std::vector<int32_t>& out) const;
};

}  // namespace fe
