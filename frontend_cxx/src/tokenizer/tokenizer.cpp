// fe_tokenizer implementation. Ports, with the same semantics
//   tokenizers 0.23.3: AddedVocabulary::{add_tokens, refresh_added_tokens, find_matches, extract_and_normalize},
//   pre_tokenizers::{Split (onig find_iter, Isolated), ByteLevel(use_regex=false)}, models::bpe::{merge_word,
//   Word::merge_all}, decoders::ByteLevel, Tokenizer::decode; ds41h/patch_serving.py split_chunks.
#include "tokenizer/tokenizer_impl.h"

#include <oniguruma.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>

#include "tokenizer/json_mini.h"

namespace fe::tok {

// ------------------------------------------------------------------------------------------------ Unicode tables
struct CpRange {
  uint32_t lo, hi;
};
#include "tokenizer/unicode_tables.inc"

namespace {

struct ClassTable {
  std::vector<uint16_t> stage1;  // cp >> 8 -> block
  std::vector<uint16_t> stage2;  // block * 256 + (cp & 255)
  uint16_t ascii[128];
  ClassTable() {
    std::vector<uint16_t> full(0x110000, 0);
    auto set = [&](const CpRange* r, size_t n, uint16_t bit) {
      for (size_t i = 0; i < n; i++)
        for (uint32_t c = r[i].lo; c <= r[i].hi; c++) full[c] |= bit;
    };
#define FE_SET(name, bit) set(kRanges_##name, sizeof(kRanges_##name) / sizeof(CpRange), bit)
    FE_SET(L, kL);
    FE_SET(M, kM);
    FE_SET(N, kN);
    FE_SET(P, kP);
    FE_SET(S, kS);
    FE_SET(s, kWS);
    FE_SET(rw, kW);
    FE_SET(rs, kRS);
#undef FE_SET
    for (uint32_t c = 0x4E00; c <= 0x9FA5; c++) full[c] |= kCJK;
    for (uint32_t c = 0x3040; c <= 0x309F; c++) full[c] |= kCJK;
    for (uint32_t c = 0x30A0; c <= 0x30FF; c++) full[c] |= kCJK;
    for (uint32_t c = 'A'; c <= 'Z'; c++) full[c] |= kAsciiAlpha;
    for (uint32_t c = 'a'; c <= 'z'; c++) full[c] |= kAsciiAlpha;
    for (const char* p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"; *p; p++) full[uint8_t(*p)] |= kAsciiPunct;
    full['\r'] |= kNL;
    full['\n'] |= kNL;
    full[' '] |= kSP;
    stage1.resize(0x1100);
    std::unordered_map<std::string, uint16_t> seen;
    for (uint32_t blk = 0; blk < 0x1100; blk++) {
      std::string key(reinterpret_cast<const char*>(&full[blk * 256]), 512);
      auto it = seen.find(key);
      if (it == seen.end()) {
        uint16_t id = uint16_t(stage2.size() / 256);
        stage2.insert(stage2.end(), full.begin() + blk * 256, full.begin() + (blk + 1) * 256);
        it = seen.emplace(std::move(key), id).first;
      }
      stage1[blk] = it->second;
    }
    for (int c = 0; c < 128; c++) ascii[c] = full[c];
  }
};

const ClassTable& class_table() {
  static const ClassTable t;
  return t;
}

}  // namespace

uint16_t cp_class(uint32_t cp) {
  if (cp >= 0x110000) return 0;
  const ClassTable& t = class_table();
  return t.stage2[size_t(t.stage1[cp >> 8]) * 256 + (cp & 255)];
}
const uint16_t* ascii_class_table() { return class_table().ascii; }

// ------------------------------------------------------------------------------------------------ UTF-8
bool utf8_valid(const char* s0, size_t n) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(s0);
  size_t i = 0;
  while (i < n) {
    if (i + 8 <= n) {
      uint64_t v;
      memcpy(&v, s + i, 8);
      if ((v & 0x8080808080808080ull) == 0) {
        i += 8;
        continue;
      }
    }
    uint8_t c = s[i];
    if (c < 0x80) {
      i++;
    } else if (c >= 0xC2 && c <= 0xDF) {
      if (i + 1 >= n || (s[i + 1] & 0xC0) != 0x80) return false;
      i += 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
      if (i + 2 >= n) return false;
      uint8_t c1 = s[i + 1];
      if (c == 0xE0 && (c1 < 0xA0 || c1 > 0xBF)) return false;
      if (c == 0xED && (c1 < 0x80 || c1 > 0x9F)) return false;
      if ((c1 & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80) return false;
      i += 3;
    } else if (c >= 0xF0 && c <= 0xF4) {
      if (i + 3 >= n) return false;
      uint8_t c1 = s[i + 1];
      if (c == 0xF0 && (c1 < 0x90 || c1 > 0xBF)) return false;
      if (c == 0xF4 && (c1 < 0x80 || c1 > 0x8F)) return false;
      if ((c1 & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80 || (s[i + 3] & 0xC0) != 0x80) return false;
      i += 4;
    } else {
      return false;
    }
  }
  return true;
}

// Rust core::str::lossy::Utf8Chunks + String::from_utf8_lossy: every maximal invalid prefix -> one U+FFFD.
std::string utf8_lossy(std::string_view bytes) {
  if (utf8_valid(bytes.data(), bytes.size())) return std::string(bytes);
  const uint8_t* s = reinterpret_cast<const uint8_t*>(bytes.data());
  const size_t n = bytes.size();
  auto get = [&](size_t i) -> uint8_t { return i < n ? s[i] : 0; };
  std::string out;
  out.reserve(n + 16);
  size_t pos = 0;
  while (pos < n) {
    size_t i = pos, valid_up_to = pos;
    while (i < n) {
      uint8_t byte = s[i];
      i++;
      if (byte >= 128) {
        int w = (byte >= 0xC2 && byte <= 0xDF) ? 2 : (byte >= 0xE0 && byte <= 0xEF) ? 3 : (byte >= 0xF0 && byte <= 0xF4) ? 4 : 0;
        bool bad = false;
        if (w == 2) {
          if (int8_t(get(i)) >= -64) bad = true;
          else i++;
        } else if (w == 3) {
          uint8_t b1 = get(i);
          bool ok = (byte == 0xE0 && b1 >= 0xA0 && b1 <= 0xBF) || (byte >= 0xE1 && byte <= 0xEC && b1 >= 0x80 && b1 <= 0xBF) ||
                    (byte == 0xED && b1 >= 0x80 && b1 <= 0x9F) || (byte >= 0xEE && byte <= 0xEF && b1 >= 0x80 && b1 <= 0xBF);
          if (!ok) bad = true;
          else {
            i++;
            if (int8_t(get(i)) >= -64) bad = true;
            else i++;
          }
        } else if (w == 4) {
          uint8_t b1 = get(i);
          bool ok = (byte == 0xF0 && b1 >= 0x90 && b1 <= 0xBF) || (byte >= 0xF1 && byte <= 0xF3 && b1 >= 0x80 && b1 <= 0xBF) ||
                    (byte == 0xF4 && b1 >= 0x80 && b1 <= 0x8F);
          if (!ok) bad = true;
          else {
            i++;
            if (int8_t(get(i)) >= -64) bad = true;
            else {
              i++;
              if (int8_t(get(i)) >= -64) bad = true;
              else i++;
            }
          }
        } else {
          bad = true;
        }
        if (bad) break;
      }
      valid_up_to = i;
    }
    out.append(bytes.data() + pos, valid_up_to - pos);
    if (i > valid_up_to) out += "\xEF\xBF\xBD";
    pos = i;
  }
  return out;
}

static inline void put_utf8(std::string& o, uint32_t cp) {
  if (cp < 0x80) {
    o += char(cp);
  } else if (cp < 0x800) {
    o += char(0xC0 | (cp >> 6));
    o += char(0x80 | (cp & 63));
  } else if (cp < 0x10000) {
    o += char(0xE0 | (cp >> 12));
    o += char(0x80 | ((cp >> 6) & 63));
    o += char(0x80 | (cp & 63));
  } else {
    o += char(0xF0 | (cp >> 18));
    o += char(0x80 | ((cp >> 12) & 63));
    o += char(0x80 | ((cp >> 6) & 63));
    o += char(0x80 | (cp & 63));
  }
}

// decode one code point of valid UTF-8 at s[i], advancing i
static inline uint32_t next_cp(const uint8_t* s, size_t& i) {
  uint8_t c = s[i];
  if (c < 0x80) {
    i += 1;
    return c;
  }
  if (c < 0xE0) {
    uint32_t cp = (uint32_t(c & 0x1F) << 6) | (s[i + 1] & 0x3F);
    i += 2;
    return cp;
  }
  if (c < 0xF0) {
    uint32_t cp = (uint32_t(c & 0x0F) << 12) | (uint32_t(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
    i += 3;
    return cp;
  }
  uint32_t cp = (uint32_t(c & 0x07) << 18) | (uint32_t(s[i + 1] & 0x3F) << 12) | (uint32_t(s[i + 2] & 0x3F) << 6) |
                (s[i + 3] & 0x3F);
  i += 4;
  return cp;
}

// ------------------------------------------------------------------------------------------------ Trie
void Trie::build(const std::vector<std::string>& patterns) {
  // temporary tree with sorted child maps, then flattened breadth-first
  struct TNode {
    int32_t term = -1;
    std::vector<std::pair<uint8_t, uint32_t>> kids;
  };
  std::vector<TNode> t(1);
  npat_ = 0;
  for (size_t pi = 0; pi < patterns.size(); pi++) {
    const std::string& p = patterns[pi];
    if (p.empty()) continue;
    npat_++;
    uint32_t cur = 0;
    for (unsigned char c : p) {
      auto& k = t[cur].kids;
      auto it = std::lower_bound(k.begin(), k.end(), std::make_pair(c, uint32_t(0)),
                                 [](auto& a, auto& b) { return a.first < b.first; });
      if (it != k.end() && it->first == c) {
        cur = it->second;
      } else {
        uint32_t nn = uint32_t(t.size());
        k.insert(it, {c, nn});
        t.emplace_back();
        cur = nn;
      }
    }
    // identical patterns: aho-corasick reports the first pattern id among equal matches
    if (t[cur].term < 0) t[cur].term = int32_t(pi);
  }
  // flatten (BFS order)
  nodes_.assign(t.size(), Node{});
  ebyte_.clear();
  enode_.clear();
  std::vector<uint32_t> order{0}, newid(t.size(), 0);
  for (size_t qi = 0; qi < order.size(); qi++)
    for (auto& kv : t[order[qi]].kids) {
      newid[kv.second] = uint32_t(order.size());
      order.push_back(kv.second);
    }
  for (size_t qi = 0; qi < order.size(); qi++) {
    const TNode& tn = t[order[qi]];
    Node& n = nodes_[qi];
    n.term = tn.term;
    n.first = uint32_t(ebyte_.size());
    n.count = uint32_t(tn.kids.size());
    for (auto& kv : tn.kids) {
      ebyte_.push_back(kv.first);
      enode_.push_back(newid[kv.second]);
    }
  }
  memset(root_, 0, sizeof(root_));
  const Node& r = nodes_[0];
  for (uint32_t k = 0; k < r.count; k++) root_[ebyte_[r.first + k]] = enode_[r.first + k];
}

bool Trie::find(const uint8_t* s, size_t n, size_t pos, size_t& start, size_t& end, int32_t& pat) const {
  if (npat_ == 0) return false;
  for (size_t i = pos; i < n; i++) {
    uint32_t cur = root_[s[i]];
    if (!cur) continue;
    int32_t best = -1;
    size_t best_end = 0;
    size_t j = i + 1;
    for (;;) {
      const Node& nd = nodes_[cur];
      if (nd.term >= 0) {
        best = nd.term;
        best_end = j;
      }
      if (j >= n || nd.count == 0) break;
      uint8_t c = s[j];
      const uint8_t* eb = ebyte_.data() + nd.first;
      uint32_t found = 0;
      if (nd.count <= 8) {
        for (uint32_t k = 0; k < nd.count; k++)
          if (eb[k] == c) {
            found = enode_[nd.first + k];
            break;
          }
      } else {
        const uint8_t* it = std::lower_bound(eb, eb + nd.count, c);
        if (it != eb + nd.count && *it == c) found = enode_[nd.first + uint32_t(it - eb)];
      }
      if (!found) break;
      cur = found;
      j++;
    }
    if (best >= 0) {
      start = i;
      end = best_end;
      pat = best;
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------ MergeMap
void MergeMap::reserve(size_t n) {
  size_t cap = 16;
  while (cap < n * 2) cap <<= 1;
  slots_.assign(cap, Slot{});
  mask_ = cap - 1;
  shift_ = 64;
  for (size_t c = cap; c > 1; c >>= 1) shift_--;
  count_ = 0;
}

void MergeMap::put(uint32_t a, uint32_t b, uint32_t rank, uint32_t nid) {
  if ((count_ + 1) * 2 > slots_.size()) {
    std::vector<Slot> old = std::move(slots_);
    reserve(std::max<size_t>(16, (count_ + 1) * 2));
    for (auto& s : old)
      if (s.key != kEmpty) put(uint32_t(s.key >> 32), uint32_t(s.key), s.rank, s.nid);
  }
  uint64_t k = (uint64_t(a) << 32) | b;
  size_t h = size_t((k * 0x9E3779B97F4A7C15ull) >> shift_);
  for (;;) {
    Slot& s = slots_[h];
    if (s.key == k) {
      s.rank = rank;
      s.nid = nid;
      return;
    }
    if (s.key == kEmpty) {
      s.key = k;
      s.rank = rank;
      s.nid = nid;
      count_++;
      return;
    }
    h = (h + 1) & mask_;
  }
}

// ------------------------------------------------------------------------------------------------ TaskPool
TaskPool::TaskPool(int nthreads) {
  for (int i = 1; i < nthreads; i++) threads_.emplace_back([this] { loop(); });
}
TaskPool::~TaskPool() {
  {
    std::lock_guard<std::mutex> lk(m_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}
void TaskPool::work_on(Job& j) {
  for (;;) {
    size_t i = j.next.fetch_add(1);
    if (i >= j.n) return;
    try {
      (*j.f)(i);
    } catch (...) {
      std::lock_guard<std::mutex> lk(j.m);
      if (!j.err) j.err = std::current_exception();
    }
    if (j.done.fetch_add(1) + 1 == j.n) {
      std::lock_guard<std::mutex> lk(j.m);
      j.cv.notify_all();
    }
  }
}
void TaskPool::loop() {
  for (;;) {
    std::shared_ptr<Job> j;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
      if (q_.empty()) return;  // stop_
      j = q_.front();
      if (j->next.load() >= j->n) {
        q_.pop_front();
        continue;
      }
    }
    work_on(*j);
    {
      std::lock_guard<std::mutex> lk(m_);
      if (!q_.empty() && q_.front() == j) q_.pop_front();
    }
  }
}
void TaskPool::run(size_t n, const std::function<void(size_t)>& f) {
  if (n == 0) return;
  if (threads_.empty() || n == 1) {
    for (size_t i = 0; i < n; i++) f(i);
    return;
  }
  auto j = std::make_shared<Job>();
  j->f = &f;
  j->n = n;
  {
    std::lock_guard<std::mutex> lk(m_);
    q_.push_back(j);
  }
  cv_.notify_all();
  work_on(*j);
  {
    std::unique_lock<std::mutex> lk(j->m);
    j->cv.wait(lk, [&] { return j->done.load() == n; });
  }
  {
    std::lock_guard<std::mutex> lk(m_);
    auto it = std::find(q_.begin(), q_.end(), j);
    if (it != q_.end()) q_.erase(it);
  }
  if (j->err) std::rethrow_exception(j->err);
}

// ------------------------------------------------------------------------------------------------ word cache
namespace {

inline uint64_t hash_bytes(const uint8_t* p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t(n) * 0xff51afd7ed558ccdull);
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    h = (h ^ v) * 0xbf58476d1ce4e5b9ull;
    h ^= h >> 31;
    p += 8;
    n -= 8;
  }
  uint64_t v = 0;
  memcpy(&v, p, n);
  h = (h ^ v) * 0x94d049bb133111ebull;
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return h;
}

struct WordCache {
  static constexpr size_t kSlots = 1u << 16;  // 1.5 MB per thread + key/id arenas (<= 8 MB)
  static constexpr size_t kMaxKey = 64;
  struct Slot {
    uint64_t h;
    uint32_t koff, ioff;
    uint16_t klen, nids;  // klen == 0: empty
  };
  uint64_t owner = 0;
  std::vector<Slot> slots;
  std::vector<uint8_t> keys;
  std::vector<int32_t> ids;
  size_t count = 0;
  void reset(uint64_t o) {
    owner = o;
    slots.assign(kSlots, Slot{0, 0, 0, 0, 0});
    keys.clear();
    ids.clear();
    count = 0;
  }
};
thread_local WordCache tl_cache;

struct Scratch {
  std::vector<uint16_t> cls;
  std::vector<uint32_t> boff;
  std::vector<Split> s1, s2;
  std::vector<std::pair<uint32_t, uint32_t>> pieces;
};
thread_local Scratch tl_scratch;

std::atomic<uint64_t> g_instance{1};

}  // namespace

}  // namespace fe::tok

namespace fe {

using namespace fe::tok;

// ------------------------------------------------------------------------------------------------ byte level
namespace {

struct ByteLevelTables {
  uint32_t byte_to_cp[256];
  int16_t cp_to_byte[324];
  ByteLevelTables() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++)
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n++);
      }
    for (int i = 0; i < 324; i++) cp_to_byte[i] = -1;
    for (size_t i = 0; i < bs.size(); i++) {
      byte_to_cp[bs[i]] = uint32_t(cs[i]);
      cp_to_byte[cs[i]] = int16_t(bs[i]);
    }
  }
  // ByteLevel decoder for one token: bytes if every char maps, else the raw UTF-8 of the token.
  std::string token_bytes(const std::string& tok) const {
    std::string out;
    const uint8_t* s = reinterpret_cast<const uint8_t*>(tok.data());
    size_t i = 0;
    while (i < tok.size()) {
      uint32_t cp = next_cp(s, i);
      if (cp >= 324 || cp_to_byte[cp] < 0) return tok;
      out += char(uint8_t(cp_to_byte[cp]));
    }
    return out;
  }
};

const ByteLevelTables& bl() {
  static const ByteLevelTables t;
  return t;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw TokenizerError("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

const char* const kDsSplits[3] = {
    "\\p{N}{1,3}",
    "[\xE4\xB8\x80-\xE9\xBE\xA5\xE3\x81\x80-\xE3\x82\x9F\xE3\x82\xA0-\xE3\x83\xBF]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
};

std::once_flag g_onig_once;

}  // namespace

TokenizerImpl::~TokenizerImpl() {
  for (void* r : regexes_) onig_free(static_cast<OnigRegex>(r));
}

void TokenizerImpl::load(std::string_view tj_text, std::string_view cfg_text) {
  using tokjson::Value;
  instance_id_ = g_instance.fetch_add(1);
  for (int i = 0; i < 256; i++) byte_id_[i] = -1;
  Value root = tokjson::parse(tj_text);
  if (!cfg_text.empty()) (void)tokjson::parse(cfg_text);  // validated; nothing in it changes encode / decode ids
  if (root.type != Value::Object) throw TokenizerError("tokenizer.json: not an object");

  auto need_null = [&](const char* k) {
    const Value* v = root.get(k);
    if (v && !v->is_null()) throw TokenizerError(std::string("tokenizer.json: unsupported non-null ") + k);
  };
  need_null("truncation");
  need_null("padding");

  // normalizer: null or empty Sequence
  if (const Value* n = root.get("normalizer"); n && !n->is_null()) {
    const Value* t = n->get("type");
    const Value* l = n->get("normalizers");
    if (!t || t->as_str() != "Sequence" || !l || l->type != Value::Array || !l->arr.empty())
      throw TokenizerError("tokenizer.json: unsupported normalizer");
  }

  // pre_tokenizer: Sequence[Split(Regex, Isolated)*, ByteLevel(add_prefix_space=false, use_regex=false)]
  std::vector<std::string> split_patterns;
  {
    const Value* pt = root.get("pre_tokenizer");
    if (!pt || pt->is_null()) throw TokenizerError("tokenizer.json: pre_tokenizer missing (unsupported)");
    std::vector<const Value*> stages;
    if (pt->get("type") && pt->get("type")->as_str() == "Sequence") {
      const Value* l = pt->get("pretokenizers");
      if (!l || l->type != Value::Array) throw TokenizerError("tokenizer.json: bad pre_tokenizer Sequence");
      for (auto& s : l->arr) stages.push_back(&s);
    } else {
      stages.push_back(pt);
    }
    bool have_bl = false;
    for (const Value* s : stages) {
      const std::string& ty = s->get("type")->as_str();
      if (have_bl) throw TokenizerError("tokenizer.json: pre-tokenizer stage after ByteLevel (unsupported)");
      if (ty == "Split") {
        const Value* pat = s->get("pattern");
        const Value* rx = pat ? pat->get("Regex") : nullptr;
        if (!rx) throw TokenizerError("tokenizer.json: Split with a String pattern (unsupported)");
        if (s->get("behavior")->as_str() != "Isolated") throw TokenizerError("tokenizer.json: Split behavior not Isolated (unsupported)");
        if (s->get("invert") && s->get("invert")->truthy()) throw TokenizerError("tokenizer.json: Split invert (unsupported)");
        split_patterns.push_back(rx->as_str());
      } else if (ty == "ByteLevel") {
        if ((s->get("add_prefix_space") && s->get("add_prefix_space")->truthy()) || (s->get("use_regex") && s->get("use_regex")->truthy()))
          throw TokenizerError("tokenizer.json: ByteLevel pre-tokenizer with prefix space / regex (unsupported)");
        have_bl = true;
      } else {
        throw TokenizerError("tokenizer.json: unsupported pre-tokenizer " + ty);
      }
    }
    if (!have_bl) throw TokenizerError("tokenizer.json: no ByteLevel pre-tokenizer (unsupported)");
  }
  fast_pretok_ = split_patterns.size() == 3 && split_patterns[0] == kDsSplits[0] && split_patterns[1] == kDsSplits[1] &&
                 split_patterns[2] == kDsSplits[2];
  if (!split_patterns.empty()) {
    std::call_once(g_onig_once, [] {
      OnigEncoding use[] = {ONIG_ENCODING_UTF8};
      onig_initialize(use, 1);
    });
    for (auto& p : split_patterns) {
      OnigRegex re = nullptr;
      OnigErrorInfo ei;
      const UChar* ps = reinterpret_cast<const UChar*>(p.data());
      int r = onig_new(&re, ps, ps + p.size(), ONIG_OPTION_NONE, ONIG_ENCODING_UTF8, ONIG_SYNTAX_DEFAULT, &ei);
      if (r != ONIG_NORMAL) {
        UChar msg[ONIG_MAX_ERROR_MESSAGE_LEN];
        onig_error_code_to_str(msg, r, &ei);
        throw TokenizerError(std::string("tokenizer.json: bad Split regex: ") + reinterpret_cast<char*>(msg));
      }
      regexes_.push_back(re);
    }
  }

  // decoder: ByteLevel
  {
    const Value* d = root.get("decoder");
    if (!d || d->is_null() || d->get("type")->as_str() != "ByteLevel") throw TokenizerError("tokenizer.json: decoder must be ByteLevel");
  }

  // model: BPE
  const Value* m = root.get("model");
  if (!m || m->get("type")->as_str() != "BPE") throw TokenizerError("tokenizer.json: model must be BPE");
  if (const Value* d = m->get("dropout"); d && !d->is_null() && std::stod(d->str) != 0.0) throw TokenizerError("BPE dropout (unsupported)");
  for (const char* k : {"continuing_subword_prefix", "end_of_word_suffix"})
    if (const Value* v = m->get(k); v && !v->is_null() && !v->as_str().empty()) throw TokenizerError(std::string("BPE ") + k + " (unsupported)");
  if (const Value* v = m->get("byte_fallback"); v && v->truthy()) throw TokenizerError("BPE byte_fallback (unsupported)");
  ignore_merges_ = m->get("ignore_merges") && m->get("ignore_merges")->truthy();
  const Value* vocab = m->get("vocab");
  if (!vocab || vocab->type != Value::Object) throw TokenizerError("BPE vocab missing");
  vocab_.reserve(vocab->obj.size() * 2);
  for (auto& kv : vocab->obj) vocab_[kv.first] = int32_t(kv.second.as_int());
  model_vocab_size_ = int32_t(vocab_.size());
  std::vector<const std::string*> vocab_r;
  {
    int32_t mx = -1;
    for (auto& kv : vocab_) mx = std::max(mx, kv.second);
    vocab_r.assign(size_t(mx + 1), nullptr);
    for (auto& kv : vocab_) vocab_r[size_t(kv.second)] = &kv.first;
  }
  bool all_bytes = true;
  for (int b = 0; b < 256; b++) {
    std::string s;
    put_utf8(s, bl().byte_to_cp[b]);
    auto it = vocab_.find(s);
    if (it != vocab_.end()) byte_id_[b] = it->second;
    else all_bytes = false;
  }
  if (!all_bytes) {
    if (const Value* u = m->get("unk_token"); u && !u->is_null())
      throw TokenizerError("BPE with unk_token and missing byte symbols (unsupported)");
    // no unk: HF drops the missing symbols, as bpe_* do (byte_id_ = -1)
  }
  const Value* merges = m->get("merges");
  if (!merges || merges->type != Value::Array) throw TokenizerError("BPE merges missing");
  merges_.reserve(merges->arr.size());
  for (size_t i = 0; i < merges->arr.size(); i++) {
    const Value& mv = merges->arr[i];
    std::string a, b;
    if (mv.type == Value::String) {
      const std::string& line = mv.str;
      size_t sp = line.find(' ');
      if (sp == std::string::npos || line.find(' ', sp + 1) != std::string::npos) throw TokenizerError("BPE merge: bad line");
      a = line.substr(0, sp);
      b = line.substr(sp + 1);
    } else if (mv.type == Value::Array && mv.arr.size() == 2) {
      a = mv.arr[0].as_str();
      b = mv.arr[1].as_str();
    } else {
      throw TokenizerError("BPE merge: bad entry");
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), in = vocab_.find(a + b);
    if (ia == vocab_.end() || ib == vocab_.end() || in == vocab_.end()) throw TokenizerError("BPE merge token out of vocabulary");
    merges_.put(uint32_t(ia->second), uint32_t(ib->second), uint32_t(i), uint32_t(in->second));
  }

  // added tokens: one AddedVocabulary::add_tokens call with all entries (TokenizerImpl deserializer)
  if (const Value* at = root.get("added_tokens"); at && at->type == Value::Array) {
    std::vector<AddedTok> toks;
    for (auto& e : at->arr) {
      AddedTok t;
      t.content = e.get("content")->as_str();
      auto flag = [&](const char* k) { const Value* v = e.get(k); return v && v->truthy(); };
      t.single_word = flag("single_word");
      t.lstrip = flag("lstrip");
      t.rstrip = flag("rstrip");
      t.normalized = flag("normalized");
      t.special = flag("special");
      toks.push_back(std::move(t));
    }
    for (auto& t : toks)
      if (t.special && !t.content.empty() && !special_set_.count(t.content)) {
        special_tokens_.push_back(t);
        special_set_[t.content] = true;
      }
    for (auto& t : toks) {
      if (t.content.empty()) continue;
      bool dup = false;
      for (auto& kv : added_r_)
        if (kv.second.same(t)) {
          dup = true;
          break;
        }
      if (dup) continue;
      int32_t new_id;
      if (auto it = added_map_.find(t.content); it != added_map_.end()) new_id = it->second;
      else if (auto iv = vocab_.find(t.content); iv != vocab_.end()) new_id = iv->second;
      else {
        int32_t mx = -1;
        for (auto& kv : added_map_) mx = std::max(mx, kv.second);
        if (mx < 0) new_id = model_vocab_size_;
        else new_id = (mx >= model_vocab_size_ || model_vocab_size_ == 0) ? mx + 1 : model_vocab_size_;
      }
      added_map_[t.content] = new_id;
      added_r_[new_id] = t;
      if (!special_set_.count(t.content)) added_tokens_.push_back(t);
    }
  }
  // refresh_added_tokens: special_tokens ++ added_tokens, partitioned by `normalized`
  {
    std::vector<std::string> raw_p, norm_p;
    trie_raw_ids_.clear();
    trie_norm_ids_.clear();
    auto add = [&](const AddedTok& t) {
      int32_t id;
      if (auto it = added_map_.find(t.content); it != added_map_.end()) id = it->second;
      else if (auto iv = vocab_.find(t.content); iv != vocab_.end()) id = iv->second;
      else throw TokenizerError("Missing additional token");
      if (t.normalized) {
        norm_p.push_back(t.content);  // normalizer is identity (empty Sequence / null)
        trie_norm_ids_.push_back(id);
      } else {
        raw_p.push_back(t.content);
        trie_raw_ids_.push_back(id);
      }
    };
    for (auto& t : special_tokens_) add(t);
    for (auto& t : added_tokens_) add(t);
    trie_raw_.build(raw_p);
    trie_norm_.build(norm_p);
  }

  // id tables
  {
    std::unordered_map<std::string, int32_t> uni = vocab_;
    for (auto& kv : added_map_) uni[kv.first] = kv.second;
    vocab_size_with_added_ = int32_t(uni.size());
    max_id_ = -1;
    for (auto& kv : uni) max_id_ = std::max(max_id_, kv.second);
    for (auto& kv : added_r_) max_id_ = std::max(max_id_, kv.first);
    dec_.assign(size_t(max_id_ + 1), DecEntry{});
    tok_off_.assign(size_t(max_id_ + 2), 0);
    dec_arena_.clear();
    tok_arena_.clear();
    for (int32_t id = 0; id <= max_id_; id++) {
      tok_off_[size_t(id)] = uint32_t(tok_arena_.size());
      const std::string* t = nullptr;
      if (auto it = added_r_.find(id); it != added_r_.end()) t = &it->second.content;
      else if (size_t(id) < vocab_r.size()) t = vocab_r[size_t(id)];
      if (!t) continue;
      tok_arena_ += *t;
      std::string b = bl().token_bytes(*t);
      DecEntry& e = dec_[size_t(id)];
      e.off = uint32_t(dec_arena_.size());
      e.len = int32_t(b.size());
      e.special = special_set_.count(*t) > 0;
      dec_arena_ += b;
    }
    tok_off_[size_t(max_id_ + 1)] = uint32_t(tok_arena_.size());
  }

  // parallelism default
  {
    int n = int(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    if (const char* e = getenv("FE_TOKENIZER_THREADS"); e && *e) n = std::max(1, atoi(e));
    parallelism_ = n;
  }
}

std::shared_ptr<TaskPool> TokenizerImpl::get_pool() const {
  std::lock_guard<std::mutex> lk(pool_m_);
  if (parallelism_ <= 1) return nullptr;
  if (!pool_ || pool_->size() != parallelism_) pool_ = std::make_shared<TaskPool>(parallelism_);
  return pool_;
}

// ------------------------------------------------------------------------------------------------ added vocabulary
namespace {
// Rust regex \w / \s helpers of added_vocabulary.rs (kW / kRS tables derived from HF's own single_word / lstrip
// behaviour over every code point; Rust \w != onig \w)
inline uint32_t last_cp(std::string_view s) {  // s non-empty, valid UTF-8
  size_t j = s.size() - 1;
  while (j > 0 && (uint8_t(s[j]) & 0xC0) == 0x80) j--;
  size_t p = j;
  return next_cp(reinterpret_cast<const uint8_t*>(s.data()), p);
}
inline bool ends_with_word(std::string_view s) { return !s.empty() && (cp_class(last_cp(s)) & kW); }
inline bool starts_with_word(std::string_view s) {
  if (s.empty()) return false;
  size_t p = 0;
  return cp_class(next_cp(reinterpret_cast<const uint8_t*>(s.data()), p)) & kW;
}
// start (byte) of the trailing \s* run
inline size_t space_leftmost_at_end(std::string_view s) {
  size_t end = s.size();
  while (end > 0) {
    size_t j = end;
    do { j--; } while (j > 0 && (uint8_t(s[j]) & 0xC0) == 0x80);
    size_t p = j;
    if (!(cp_class(next_cp(reinterpret_cast<const uint8_t*>(s.data()), p)) & kRS)) break;
    end = j;
  }
  return end;
}
// length (bytes) of the leading \s* run
inline size_t space_rightmost_at_start(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    size_t p = i;
    if (!(cp_class(next_cp(reinterpret_cast<const uint8_t*>(s.data()), p)) & kRS)) break;
    i = p;
  }
  return i;
}
}  // namespace

void TokenizerImpl::find_added(std::string_view s, bool normalized_pass, std::vector<Split>& out) const {
  out.clear();
  if (s.empty()) {
    out.push_back({-1, 0, 0});
    return;
  }
  const Trie& trie = normalized_pass ? trie_norm_ : trie_raw_;
  const std::vector<int32_t>& pid = normalized_pass ? trie_norm_ids_ : trie_raw_ids_;
  const uint8_t* b = reinterpret_cast<const uint8_t*>(s.data());
  size_t start_offset = 0, pos = 0, ms, me;
  int32_t pat;
  while (trie.find(b, s.size(), pos, ms, me, pat)) {
    pos = me;
    size_t start = ms, stop = me;
    int32_t id = pid[size_t(pat)];
    auto it = added_r_.find(id);
    const AddedTok& t = it->second;
    if (t.single_word) {
      bool start_space = start == 0 || !ends_with_word(s.substr(0, start));
      bool stop_space = stop == s.size() || !starts_with_word(s.substr(stop));
      if (!stop_space || !start_space) continue;
    }
    if (t.lstrip) {
      size_t ns = space_leftmost_at_end(s.substr(0, start));
      start = std::max(ns, start_offset);
    }
    if (t.rstrip) stop += space_rightmost_at_start(s.substr(stop));
    if (start_offset < start) out.push_back({-1, start_offset, start});
    out.push_back({id, start, stop});
    start_offset = stop;
  }
  if (start_offset != s.size()) out.push_back({-1, start_offset, s.size()});
}

// ------------------------------------------------------------------------------------------------ pre-tokenizer
template <class F>
inline void TokenizerImpl::stage1(const uint16_t* cls, uint32_t a, uint32_t b, F&& f) const {
  uint32_t i = a, gap = a;
  while (i < b) {
    if (cls[i] & kN) {
      if (gap < i) stage2(cls, gap, i, f);
      uint32_t j = i + 1, lim = std::min(b, i + 3);
      while (j < lim && (cls[j] & kN)) j++;
      stage2(cls, i, j, f);
      i = gap = j;
    } else {
      i++;
    }
  }
  if (gap < b) stage2(cls, gap, b, f);
}

template <class F>
inline void TokenizerImpl::stage2(const uint16_t* cls, uint32_t a, uint32_t b, F&& f) const {
  uint32_t i = a, gap = a;
  while (i < b) {
    if (cls[i] & kCJK) {
      if (gap < i) stage3(cls, gap, i, f);
      uint32_t j = i + 1;
      while (j < b && (cls[j] & kCJK)) j++;
      stage3(cls, i, j, f);
      i = gap = j;
    } else {
      i++;
    }
  }
  if (gap < b) stage3(cls, gap, b, f);
}

// The third Split regex, alternatives in order (onig leftmost-first, greedy with backtracking):
//  A1 [ascii punct][A-Za-z]+   A2 [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   A3 ' '?[\p{P}\p{S}]+[\r\n]*
//  A4 \s*[\r\n]+               A5 \s+(?!\S)                                  A6 \s+
template <class F>
inline void TokenizerImpl::stage3(const uint16_t* cls, uint32_t a, uint32_t e, F&& f) const {
  uint32_t pos = a, gap = a;
  while (pos < e) {
    const uint16_t k0 = cls[pos];
    uint32_t end = 0;
    if ((k0 & kAsciiPunct) && pos + 1 < e && (cls[pos + 1] & kAsciiAlpha)) {
      uint32_t j = pos + 2;
      while (j < e && (cls[j] & kAsciiAlpha)) j++;
      end = j;
    } else if (k0 & (kL | kM)) {
      uint32_t j = pos + 1;
      while (j < e && (cls[j] & (kL | kM))) j++;
      end = j;
    } else if (!(k0 & (kNL | kL | kP | kS)) && pos + 1 < e && (cls[pos + 1] & (kL | kM))) {
      uint32_t j = pos + 2;
      while (j < e && (cls[j] & (kL | kM))) j++;
      end = j;
    } else if ((k0 & kSP) && pos + 1 < e && (cls[pos + 1] & (kP | kS))) {
      uint32_t j = pos + 2;
      while (j < e && (cls[j] & (kP | kS))) j++;
      while (j < e && (cls[j] & kNL)) j++;
      end = j;
    } else if (k0 & (kP | kS)) {
      uint32_t j = pos + 1;
      while (j < e && (cls[j] & (kP | kS))) j++;
      while (j < e && (cls[j] & kNL)) j++;
      end = j;
    } else if (k0 & kWS) {
      uint32_t j = pos + 1;
      while (j < e && (cls[j] & kWS)) j++;
      uint32_t q = j;
      while (q > pos && !(cls[q - 1] & kNL)) q--;
      if (q > pos) end = q;                 // A4: up to the last newline of the run
      else if (j == e) end = j;             // A5 at the piece end
      else if (j - pos >= 2) end = j - 1;   // A5: leave one space for the next token
      else end = j;                         // A6
    }
    if (end) {
      if (gap < pos) f(gap, pos);
      f(pos, end);
      pos = gap = end;
    } else {
      pos++;
    }
  }
  if (gap < e) f(gap, e);
}

namespace {
// code point classes + byte offsets of a valid UTF-8 segment; returns the number of code points
inline size_t classify_n(std::string_view seg, std::vector<uint16_t>& cls, std::vector<uint32_t>& boff) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(seg.data());
  const size_t n = seg.size();
  if (cls.size() < n + 1) {
    cls.resize(n + 1);
    boff.resize(n + 1);
  }
  const uint16_t* asc = ascii_class_table();
  uint16_t* c = cls.data();
  uint32_t* o = boff.data();
  size_t nc = 0, i = 0;
  while (i < n) {
    uint8_t b = s[i];
    o[nc] = uint32_t(i);
    if (b < 0x80) {
      c[nc++] = asc[b];
      i++;
    } else {
      c[nc++] = cp_class(next_cp(s, i));
    }
  }
  o[nc] = uint32_t(n);
  c[nc] = 0;
  return nc;
}
}  // namespace

void TokenizerImpl::pretokenize(std::string_view seg, std::vector<std::pair<uint32_t, uint32_t>>& out) const {
  out.clear();
  if (seg.empty()) return;
  if (!fast_pretok_) {
    pretokenize_onig(seg, out);
    return;
  }
  Scratch& sc = tl_scratch;
  size_t nc = classify_n(seg, sc.cls, sc.boff);
  const uint32_t* bo = sc.boff.data();
  stage1(sc.cls.data(), 0, uint32_t(nc), [&](uint32_t a, uint32_t b) { out.push_back({bo[a], bo[b]}); });
}

void TokenizerImpl::pretokenize_onig(std::string_view seg, std::vector<std::pair<uint32_t, uint32_t>>& out) const {
  out.clear();
  if (seg.empty()) return;
  out.push_back({0, uint32_t(seg.size())});
  std::vector<std::pair<uint32_t, uint32_t>> next;
  OnigRegion* region = onig_region_new();
  const UChar* base = reinterpret_cast<const UChar*>(seg.data());
  for (void* rv : regexes_) {
    OnigRegex re = static_cast<OnigRegex>(rv);
    next.clear();
    for (auto [a, b] : out) {
      // tokenizers SysRegex::find_matches over the piece alone (lookaround ends at the piece) with the onig crate's
      // FindMatches iterator; Isolated keeps matches and gaps, empty pieces are dropped below.
      const UChar* str = base + a;
      const UChar* end = base + b;
      const size_t len = b - a;
      size_t last_end = 0, prev = 0;
      bool have_last = false;
      size_t last_match_end = 0;
      for (;;) {
        if (last_end > len) break;
        onig_region_clear(region);
        int r = onig_search(re, str, end, str + last_end, end, region, ONIG_OPTION_NONE);
        if (r < 0) break;
        size_t ms = size_t(region->beg[0]), me = size_t(region->end[0]);
        if (me == ms && have_last && last_match_end == me) {
          if (last_end >= len) {
            last_end = len + 1;
          } else {
            size_t p = a + last_end;
            next_cp(base, p);
            last_end = p - a;
          }
          continue;
        }
        last_end = me;
        have_last = true;
        last_match_end = me;
        if (prev != ms) next.push_back({uint32_t(a + prev), uint32_t(a + ms)});
        next.push_back({uint32_t(a + ms), uint32_t(a + me)});
        prev = me;
      }
      if (prev != len) next.push_back({uint32_t(a + prev), uint32_t(b)});
    }
    out.clear();
    for (auto& p : next)
      if (p.second > p.first) out.push_back(p);
  }
  onig_region_free(region, 1);
}

// ------------------------------------------------------------------------------------------------ BPE
void TokenizerImpl::bpe_naive(const uint8_t* p, size_t n, std::vector<int32_t>& out) const {
  int32_t symb[64];
  uint32_t rk[64], nid[64];
  int32_t* sym = symb;
  std::vector<int32_t> vs;
  std::vector<uint32_t> vr, vn;
  uint32_t* R = rk;
  uint32_t* NI = nid;
  if (n > 64) {
    vs.resize(n);
    vr.resize(n);
    vn.resize(n);
    sym = vs.data();
    R = vr.data();
    NI = vn.data();
  }
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    int32_t id = byte_id_[p[i]];
    if (id >= 0) sym[m++] = id;
  }
  if (m == 0) return;
  constexpr uint32_t INF = std::numeric_limits<uint32_t>::max();
  for (size_t i = 0; i + 1 < m; i++)
    if (!merges_.get(uint32_t(sym[i]), uint32_t(sym[i + 1]), R[i], NI[i])) R[i] = INF;
  while (m > 1) {
    uint32_t best = INF;
    size_t bi = 0;
    for (size_t i = 0; i + 1 < m; i++)
      if (R[i] < best) {
        best = R[i];
        bi = i;
      }
    if (best == INF) break;
    sym[bi] = int32_t(NI[bi]);
    for (size_t k = bi + 1; k + 1 < m; k++) sym[k] = sym[k + 1];
    for (size_t k = bi + 1; k + 2 < m; k++) {
      R[k] = R[k + 1];
      NI[k] = NI[k + 1];
    }
    m--;
    if (bi > 0 && !merges_.get(uint32_t(sym[bi - 1]), uint32_t(sym[bi]), R[bi - 1], NI[bi - 1])) R[bi - 1] = INF;
    if (bi + 1 < m && !merges_.get(uint32_t(sym[bi]), uint32_t(sym[bi + 1]), R[bi], NI[bi])) R[bi] = INF;
  }
  out.insert(out.end(), sym, sym + m);
}

// Port of models/bpe/word.rs Word::merge_all (dropout = None).
void TokenizerImpl::bpe_heap(const uint8_t* p, size_t n, std::vector<int32_t>& out) const {
  struct Sym {
    uint32_t c;
    int64_t prev, next;
    size_t len;
  };
  struct Merge {
    uint32_t rank;
    size_t pos;
    uint32_t new_id;
  };
  struct Cmp {
    bool operator()(const Merge& a, const Merge& b) const {
      if (a.rank != b.rank) return a.rank > b.rank;
      return a.pos > b.pos;
    }
  };
  std::vector<Sym> sy;
  sy.reserve(n);
  for (size_t i = 0; i < n; i++) {
    int32_t id = byte_id_[p[i]];
    if (id < 0) continue;
    int64_t len = int64_t(sy.size());
    if (!sy.empty()) sy.back().next = len;
    sy.push_back({uint32_t(id), sy.empty() ? -1 : len - 1, -1, 1});
  }
  std::priority_queue<Merge, std::vector<Merge>, Cmp> q;
  uint32_t rank, nid;
  for (size_t i = 0; i + 1 < sy.size(); i++)
    if (merges_.get(sy[i].c, sy[i + 1].c, rank, nid)) q.push({rank, i, nid});
  while (!q.empty()) {
    Merge top = q.top();
    q.pop();
    if (sy[top.pos].len == 0) continue;
    if (sy[top.pos].next == -1) continue;
    size_t next_pos = size_t(sy[top.pos].next);
    Sym right = sy[next_pos];
    if (!merges_.get(sy[top.pos].c, right.c, rank, nid) || nid != top.new_id) continue;
    // merge_with
    sy[top.pos].c = top.new_id;
    sy[top.pos].len += right.len;
    sy[top.pos].next = right.next;
    sy[next_pos].len = 0;
    if (right.next > -1 && size_t(right.next) < sy.size()) sy[size_t(right.next)].prev = int64_t(top.pos);
    const Sym& cur = sy[top.pos];
    if (cur.prev >= 0) {
      size_t pv = size_t(cur.prev);
      if (merges_.get(sy[pv].c, cur.c, rank, nid)) q.push({rank, pv, nid});
    }
    if (cur.next >= 0 && size_t(cur.next) < sy.size()) {
      size_t nx = size_t(cur.next);
      if (merges_.get(cur.c, sy[nx].c, rank, nid)) q.push({rank, top.pos, nid});
    }
  }
  for (auto& s : sy)
    if (s.len != 0) out.push_back(int32_t(s.c));
}

void TokenizerImpl::bpe(const uint8_t* p, size_t n, std::vector<int32_t>& out) const {
  if (n == 0) return;
  if (n == 1) {
    if (byte_id_[p[0]] >= 0) out.push_back(byte_id_[p[0]]);
    return;
  }
  if (ignore_merges_) {
    std::string s;
    for (size_t i = 0; i < n; i++) put_utf8(s, bl().byte_to_cp[p[i]]);
    if (auto it = vocab_.find(s); it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  if (n > WordCache::kMaxKey) {
    if (n <= 256) bpe_naive(p, n, out);
    else bpe_heap(p, n, out);
    return;
  }
  WordCache& c = tl_cache;
  if (c.owner != instance_id_) c.reset(instance_id_);
  uint64_t h = hash_bytes(p, n);
  size_t mask = WordCache::kSlots - 1, idx = size_t(h) & mask;
  for (;;) {
    WordCache::Slot& s = c.slots[idx];
    if (s.klen == 0) break;
    if (s.h == h && s.klen == n && memcmp(c.keys.data() + s.koff, p, n) == 0) {
      out.insert(out.end(), c.ids.begin() + s.ioff, c.ids.begin() + s.ioff + s.nids);
      return;
    }
    idx = (idx + 1) & mask;
  }
  size_t before = out.size();
  bpe_naive(p, n, out);
  if (c.count * 2 >= WordCache::kSlots || c.keys.size() > (4u << 20)) {
    c.reset(instance_id_);
    idx = size_t(h) & mask;
  }
  WordCache::Slot& s = c.slots[idx];
  s.h = h;
  s.koff = uint32_t(c.keys.size());
  s.klen = uint16_t(n);
  s.ioff = uint32_t(c.ids.size());
  s.nids = uint16_t(out.size() - before);
  c.keys.insert(c.keys.end(), p, p + n);
  c.ids.insert(c.ids.end(), out.begin() + long(before), out.end());
  c.count++;
}

// ------------------------------------------------------------------------------------------------ encode
void TokenizerImpl::encode_plain(std::string_view seg, std::vector<int32_t>& out) const {
  if (seg.empty()) return;
  const uint8_t* s = reinterpret_cast<const uint8_t*>(seg.data());
  if (fast_pretok_) {
    Scratch& sc = tl_scratch;
    size_t nc = classify_n(seg, sc.cls, sc.boff);
    const uint32_t* bo = sc.boff.data();
    stage1(sc.cls.data(), 0, uint32_t(nc), [&](uint32_t a, uint32_t b) { bpe(s + bo[a], bo[b] - bo[a], out); });
    return;
  }
  if (regexes_.empty()) {
    bpe(s, seg.size(), out);
    return;
  }
  std::vector<std::pair<uint32_t, uint32_t>> pieces;
  pretokenize_onig(seg, pieces);
  for (auto [a, b] : pieces) bpe(s + a, b - a, out);
}

void TokenizerImpl::encode_text(std::string_view text, std::vector<int32_t>& out) const {
  if (text.empty()) return;
  std::vector<Split> s1, s2;  // local: encode_plain uses the thread-local scratch
  find_added(text, false, s1);
  for (const Split& a : s1) {
    if (a.b <= a.a) continue;
    if (a.id >= 0) {
      out.push_back(a.id);
      continue;
    }
    std::string_view seg = text.substr(a.a, a.b - a.a);
    if (trie_norm_.empty()) {
      encode_plain(seg, out);
      continue;
    }
    find_added(seg, true, s2);
    for (const Split& b : s2) {
      if (b.b <= b.a) continue;
      if (b.id >= 0) out.push_back(b.id);
      else encode_plain(seg.substr(b.a, b.b - b.a), out);
    }
  }
}

// ------------------------------------------------------------------------------------------------ decode
void TokenizerImpl::decode_raw_append(const int32_t* ids, size_t n, bool skip_special, std::string& raw) const {
  for (size_t i = 0; i < n; i++) {
    int32_t id = ids[i];
    if (id < 0 || id > max_id_) continue;
    const DecEntry& e = dec_[size_t(id)];
    if (e.len < 0) continue;
    if (skip_special && e.special) continue;
    raw.append(dec_arena_.data() + e.off, size_t(e.len));
  }
}

std::string TokenizerImpl::decode(const int32_t* ids, size_t n, bool skip_special) const {
  std::string raw;
  decode_raw_append(ids, n, skip_special, raw);
  if (utf8_valid(raw.data(), raw.size())) return raw;
  return utf8_lossy(raw);
}

// ------------------------------------------------------------------------------------------------ Tokenizer
std::unique_ptr<Tokenizer> Tokenizer::from_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json) {
  std::unique_ptr<Tokenizer> t(new Tokenizer());
  t->impl_ = std::make_unique<TokenizerImpl>();
  t->impl_->load(tokenizer_json, tokenizer_config_json);
  return t;
}

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& tokenizer_json, const std::string& tokenizer_config_json) {
  std::string a = read_file(tokenizer_json);
  std::string b = tokenizer_config_json.empty() ? std::string() : read_file(tokenizer_config_json);
  return from_json(a, b);
}

Tokenizer::~Tokenizer() = default;

namespace {
inline size_t leads8(uint64_t v) {  // non-continuation bytes among 8
  return 8 - size_t(__builtin_popcountll(v & ~(v << 1) & 0x8080808080808080ull));
}
inline uint64_t has_nl8(uint64_t v) {  // nonzero iff some byte is '\n' or '\r'
  auto hz = [](uint64_t x) { return (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull; };
  return hz(v ^ 0x0a0a0a0a0a0a0a0aull) | hz(v ^ 0x0d0d0d0d0d0d0d0dull);
}
// byte offset k code points after the char boundary p (or n)
inline size_t advance_chars(const uint8_t* s, size_t n, size_t p, size_t k) {
  while (p + 8 <= n) {
    uint64_t v;
    memcpy(&v, s + p, 8);
    size_t l = leads8(v);
    if (l >= k) break;
    p += 8;
    k -= l;
  }
  while (p < n) {
    if ((s[p] & 0xC0) != 0x80) {
      if (k == 0) break;
      k--;
    }
    p++;
  }
  return p;
}
}  // namespace

size_t Tokenizer::count_chars(std::string_view text) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(text.data());
  size_t n = text.size(), i = 0, c = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    memcpy(&v, s + i, 8);
    c += leads8(v);
  }
  for (; i < n; i++) c += (s[i] & 0xC0) != 0x80;
  return c;
}

// ds41h/patch_serving.py split_chunks (indices are code points there; here byte offsets of the same cut points).
std::vector<size_t> Tokenizer::split_chunks(std::string_view text, size_t min_chars, size_t target) {
  std::vector<size_t> starts{0};
  const size_t nchars = count_chars(text);
  if (nchars < min_chars) return starts;
  const uint8_t* s = reinterpret_cast<const uint8_t*>(text.data());
  const size_t n = text.size();
  size_t start_c = 0, start_b = 0;
  while (double(nchars - start_c) > 1.5 * double(target)) {
    size_t pb = advance_chars(s, n, start_b, target);  // byte offset of code point index start_c + target
    // smallest q >= pb with text[q-1] in "\r\n" and text[q] in [A-Za-z]  (_SAFE = (?<=[\r\n])(?=[A-Za-z]))
    size_t q = 0;
    bool found = false;
    for (size_t j = pb - 1; j + 1 < n;) {  // j: candidate newline (text[q-1]), q = j + 1; pb >= 1
      if (j + 9 <= n) {
        uint64_t v;
        memcpy(&v, s + j, 8);
        if (!has_nl8(v)) {
          j += 8;
          continue;
        }
      }
      if ((s[j] == '\n' || s[j] == '\r') && (s[j + 1] | 0x20) >= 'a' && (s[j + 1] | 0x20) <= 'z') {
        q = j + 1;
        found = true;
        break;
      }
      j++;
    }
    if (!found) break;
    start_c += target + count_chars(text.substr(pb, q - pb));
    start_b = q;
    starts.push_back(q);
  }
  return starts;
}

std::vector<int32_t> Tokenizer::encode_unchunked(std::string_view text) const {
  if (!utf8_valid(text.data(), text.size())) throw TokenizerInputError();
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 8);
  impl_->encode_text(text, out);
  return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  if (!utf8_valid(text.data(), text.size())) throw TokenizerInputError();
  std::vector<int32_t> out;
  std::vector<size_t> starts = split_chunks(text);
  if (starts.size() <= 1) {
    out.reserve(text.size() / 3 + 8);
    impl_->encode_text(text, out);
    return out;
  }
  const size_t nch = starts.size();
  auto chunk = [&](size_t k) {
    size_t a = starts[k], b = k + 1 < nch ? starts[k + 1] : text.size();
    return text.substr(a, b - a);
  };
  std::shared_ptr<TaskPool> pool = impl_->get_pool();
  if (!pool) {
    out.reserve(text.size() / 3 + 8);
    for (size_t k = 0; k < nch; k++) impl_->encode_text(chunk(k), out);
    return out;
  }
  std::vector<std::vector<int32_t>> parts(nch);
  std::function<void(size_t)> f = [&](size_t k) {
    parts[k].reserve(chunk(k).size() / 3 + 8);
    impl_->encode_text(chunk(k), parts[k]);
  };
  pool->run(nch, f);
  size_t tot = 0;
  for (auto& p : parts) tot += p.size();
  out.reserve(tot);
  for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
  return impl_->decode(ids.data(), ids.size(), skip_special);
}

std::optional<int32_t> Tokenizer::token_to_id(std::string_view token) const {
  std::string k(token);
  if (auto it = impl_->added_map_.find(k); it != impl_->added_map_.end()) return it->second;
  if (auto it = impl_->vocab_.find(k); it != impl_->vocab_.end()) return it->second;
  return std::nullopt;
}

std::optional<std::string> Tokenizer::id_to_token(int32_t id) const {
  if (id < 0 || id > impl_->max_id_ || impl_->dec_[size_t(id)].len < 0) return std::nullopt;
  size_t a = impl_->tok_off_[size_t(id)], b = impl_->tok_off_[size_t(id) + 1];
  return impl_->tok_arena_.substr(a, b - a);
}

bool Tokenizer::is_special(int32_t id) const {
  if (id < 0 || id > impl_->max_id_) return false;
  const auto& e = impl_->dec_[size_t(id)];
  return e.len >= 0 && e.special;
}

std::optional<std::string_view> Tokenizer::added_token_content(int32_t id) const {
  auto it = impl_->added_r_.find(id);
  if (it == impl_->added_r_.end()) return std::nullopt;
  return std::string_view(it->second.content);
}

int32_t Tokenizer::vocab_size() const { return impl_->vocab_size_with_added_; }
int32_t Tokenizer::max_token_id() const { return impl_->max_id_; }

void Tokenizer::set_parallelism(int nthreads) {
  std::lock_guard<std::mutex> lk(impl_->pool_m_);
  impl_->parallelism_ = std::max(1, nthreads);
}
int Tokenizer::parallelism() const {
  std::lock_guard<std::mutex> lk(impl_->pool_m_);
  return impl_->parallelism_;
}

}  // namespace fe
