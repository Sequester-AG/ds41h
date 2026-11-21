// kv_manager.h -- the C++ OwnedKVManager.
// Interface: fpr/core/include/kv.h. Golden-trace replay: fpr/core/tests/replay_kv.cpp.
//
// Container choices (each one reproduces the Python container's ORDER, which is part of the decisions):
//   PagePool.free_uncached  deque (FIFO)            -> std::deque
//                           list + heapq (state)    -> binary min-heap on a vector (pop order = the set's order)
//                           TierFree (FPR_KV_HBM)   -> min-heap of HBM ids + FIFO deque of host ids
//   PagePool.lru            OrderedDict             -> intrusive doubly-linked list over page ids (O(1) pop / append)
//   PagePool.cached         dict (never iterated in a decision) -> unordered_map
//   SeqKV.window            dict (insertion order)  -> WindowTable (vector of pairs)
//   OwnedKVManager.deferred dict step -> list (flush iterates keys in insertion order) -> vector of (step, list)
//   pinned / pending        dict keyed by step (only popped / counted) -> std::map
#pragma once
#include <deque>
#include <map>
#include <tuple>
#include <unordered_map>

#include "../../include/kv.h"

namespace fpr {
namespace kvimpl {

inline int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; }
inline int64_t pmod(int64_t a, int64_t b) { int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }

struct FreeList {                     // deque / heapq list / TierFree
  int mode = 0;                       // 0 FIFO deque, 1 min-heap (state: lowest first), 2 TierFree (heap HBM + FIFO host)
  int64_t n_hbm = 0;
  std::vector<int64_t> heap;          // modes 1 and 2 (HBM ids)
  std::deque<int64_t> dq;             // modes 0 and 2 (host ids)
  size_t size() const { return heap.size() + dq.size(); }
  bool empty() const { return heap.empty() && dq.empty(); }
  int64_t heap_pop();
  void heap_push(int64_t p);
  int64_t popleft();                  // alloc's pop: heappop (mode 1) / popleft (modes 0, 2)
  void append(int64_t p);             // release's push: heappush (mode 1) / append (0) / TierFree.append (2)
  int64_t pop_hbm() { return heap.empty() ? -1 : heap_pop(); }
  int64_t pop_host() { if (dq.empty()) return -1; int64_t p = dq.front(); dq.pop_front(); return p; }
};

struct Lru {                          // OrderedDict[int, None]: front = evicted first
  std::vector<int64_t> prv, nxt; std::vector<char> in; int64_t head = -1, tail = -1; size_t n = 0;
  void init(int64_t N) { prv.assign((size_t)N, -1); nxt.assign((size_t)N, -1); in.assign((size_t)N, 0); }
  bool contains(int64_t p) const { return p >= 0 && p < (int64_t)in.size() && in[(size_t)p]; }
  void push_back(int64_t p);          // lru[p] = None (an existing key keeps its place, as in Python)
  void erase(int64_t p);              // lru.pop(p)
  int64_t pop_front();                // lru.popitem(last=False)
};

struct PagePool {
  PoolSpec spec; int rank = 0;
  std::vector<int32_t> ref; std::vector<Hash> hash_of; std::vector<char> has_hash;
  bool low_first = false; int64_t n_hbm = 0;
  std::vector<int64_t> demote_q;
  FreeList free_uncached; Lru lru;
  std::unordered_map<Hash, int64_t, HashHasher> cached;
  int64_t reserved = 0;
  bool poison_on = false; std::vector<int64_t> poison;

  PagePool(const PoolSpec& s, int r, const KVGeometry& g);
  int64_t num_pages() const { return spec.num_pages; }
  int64_t num_free() const { return (int64_t)(free_uncached.size() + lru.n); }
  int64_t num_used() const { return num_pages() - num_free(); }
  bool cached_get(const Hash& h, int64_t* p) const { auto it = cached.find(h); if (it == cached.end()) return false; *p = it->second; return true; }
  bool caches_page(int64_t p) const { int64_t q; return has_hash[(size_t)p] && cached_get(hash_of[(size_t)p], &q) && q == p; }
  bool alloc(int64_t n, std::vector<int64_t>* out);
  void touch(int64_t p);
  void release(int64_t p);
  int64_t host_dst();
  void move_hash(int64_t src, int64_t dst);
  bool register_(int64_t p, const Hash& h);
};

struct Pin { int32_t pool, rank; int64_t page; };

struct Heads {                        // replay aid: free-list heads in pop order, LRU head, counts
  std::vector<int64_t> free, host, lru; int64_t nfree = 0, nlru = 0, ncached = 0, reserved = 0, demote = 0, npoison = 0;
};

class OwnedKVManager final : public IKVManager {
 public:
  explicit OwnedKVManager(const KVGeometry& g);
  const KVGeometry& geom() const override { return g_; }
  std::vector<int64_t> lookup(const std::vector<Hash>& block_hashes) const override;
  int64_t free_pages(int rank) const override;
  KVUsage usage(int rank) const override;
  int64_t pages_for(int64_t tokens) const override { return -fdiv(-tokens, g_.page_tokens); }
  void hit_after_cap(int64_t hit_blocks, int64_t prompt_len, int64_t* hit, int64_t* cached, int64_t* replay_start) const override;
  const std::vector<int64_t>& block_table(const Rid& id) const override;
  const WindowTable& window_table(const Rid& id) const override;
  int32_t state_slot(const Rid& id) const override;
  SeqKV* seq(const Rid& key) override { auto it = seqs_.find(key); return it == seqs_.end() ? nullptr : it->second.get(); }
  const SeqKV* seq(const Rid& key) const override { auto it = seqs_.find(key); return it == seqs_.end() ? nullptr : it->second.get(); }
  size_t num_seqs() const override { return seqs_.size(); }
  bool has_pinned() const override { return !pinned_.empty(); }
  int64_t num_pages(int32_t pool, int rank) const override { return P(pool, rank).num_pages(); }
  int64_t num_free(int32_t pool, int rank) const override { return P(pool, rank).num_free(); }
  int64_t num_used(int32_t pool, int rank) const override { return P(pool, rank).num_used(); }
  const KVStats& stats() const override { return stats_; }
  Status admit(const Rid& id, int rank, const std::vector<Hash>& block_hashes, int64_t prompt_len, int64_t step,
               int copy_from, AdmitResult* out) override;
  bool can_allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) const override;
  bool allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) override;
  void commit(const Rid& id, int64_t num_computed, const std::vector<Hash>& block_hashes) override;
  void shrink_reservation(const Rid& id) override;
  bool open_shadow(const Rid& key, int rank, int64_t replay_start) override;
  void trim_window(const Rid& id, int64_t lo_pos) override;
  void free_now(const Rid& id) override;
  void free_deferred(const Rid& id, int64_t defer_until) override;
  void flush_deferred(int64_t upto_step) override;
  Status migrate(const Rid& id, int dst, int64_t step, int64_t num_computed, std::vector<CopyOp>* copies) override;
  std::vector<CopyOp> tier_copies(int64_t step, int64_t max_pages = 4096) override;
  void release_copies(int64_t step) override;
  std::vector<std::pair<int32_t, int64_t>> take_poison(int rank) override;
  Status check() const override;
  const std::string& last_error() const override { return err_; }

  // replay aids (not in IKVManager)
  void heads(int32_t pool, int rank, int k, Heads* out) const;
  size_t num_deferred_seqs() const;
  int64_t errors() const { return nerr_; }
  void set_poison(int32_t pool, int rank, std::vector<int64_t> v) { P(pool, rank).poison = std::move(v); }   // "_ext"

  // introspection (the Python binding's views: kv.seqs, kv.paged[r].ref / .cached / .lru / .hash_of / .poison, kv.pinned,
  // kv.pending, kv.deferred); none of these changes a decision except poison_append (the tests' hand-made re-queue)
  std::shared_ptr<SeqKV> seq_ptr(const Rid& key) const { auto it = seqs_.find(key); return it == seqs_.end() ? nullptr : it->second; }
  std::vector<Rid> seq_keys() const { std::vector<Rid> k; for (auto& e : seqs_) k.push_back(e.first); return k; }
  const PagePool& pool(int32_t pool, int rank) const { return P(pool, rank); }
  void poison_append(int32_t pool, int rank, int64_t page) { P(pool, rank).poison.push_back(page); }
  std::vector<int64_t> lru_list(int32_t pool, int rank) const {
    std::vector<int64_t> out; const auto& l = P(pool, rank).lru;
    for (int64_t q = l.head; q >= 0; q = l.nxt[(size_t)q]) out.push_back(q);
    return out;
  }
  std::vector<std::pair<int64_t, std::vector<Pin>>> pinned_list() const {
    std::vector<std::pair<int64_t, std::vector<Pin>>> out; for (auto& e : pinned_) out.emplace_back(e.first, e.second); return out;
  }
  std::vector<int64_t> pending_steps() const { std::vector<int64_t> out; for (auto& e : pending_) out.push_back(e.first); return out; }
  const std::vector<std::pair<int64_t, std::vector<std::shared_ptr<SeqKV>>>>& deferred_list() const { return deferred_; }   // failed calls so far (last_error() = the latest)

 private:
  PagePool& P(int32_t pool, int rank) { return pool == POOL_PAGED ? paged_[(size_t)rank] : pool == POOL_WINDOW ? window_[(size_t)rank] : state_[(size_t)rank]; }
  const PagePool& P(int32_t pool, int rank) const { return pool == POOL_PAGED ? paged_[(size_t)rank] : pool == POOL_WINDOW ? window_[(size_t)rank] : state_[(size_t)rank]; }
  void window_need(const SeqKV& sk, int64_t start, int64_t end, int64_t* lo, int64_t* hi) const;
  void release_seq(SeqKV& sk);
  CopyOp copy(int32_t pool, int32_t src, int64_t s, int32_t dst, int64_t d, int64_t step);
  bool fail(const std::string& m) const { err_ = m; ++nerr_; return false; }

  KVGeometry g_;
  std::vector<PagePool> paged_, window_, state_;
  std::unordered_map<Rid, std::shared_ptr<SeqKV>> seqs_;
  std::map<int64_t, std::vector<Pin>> pinned_;
  std::map<int64_t, std::map<std::tuple<int32_t, int32_t, int64_t>, std::pair<int32_t, int64_t>>> pending_;
  std::vector<std::pair<int64_t, std::vector<std::shared_ptr<SeqKV>>>> deferred_;
  KVStats stats_;
  int64_t inc_ = 0;
  mutable std::string err_;
  mutable int64_t nerr_ = 0;
};

}  // namespace kvimpl
}  // namespace fpr
