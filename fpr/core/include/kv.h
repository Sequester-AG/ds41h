// kv.h -- serve/kv_manager.py.
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
// v4: the full OwnedKVManager surface (every public method the scheduler / adapter / worker call)
// SeqKV by reference (shared_ptr page lists / window tables: Python holds the list / dict objects), shadow keys, pool
// specs, stats; v4.1: paged_usage / estimate_cached_tokens (the adapter facade). Implementation: fpr/core/src/kv/kv_manager.cpp (make_kv_manager); replay: fpr/core/tests/replay_kv.cpp.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "geometry.h"

namespace fpr {

// ---------------------------------------------------------------- pools
// Python names pools by string. C++ uses the int ids below; where Python SORTS by pool name (interfaces.copy_runs:
// "paged" < "state" < "window") use pool_sort_rank(), never the id.
enum : int32_t { POOL_PAGED = 0, POOL_WINDOW = 1, POOL_STATE = 2 };
inline const char* pool_name(int32_t p) { return p == POOL_PAGED ? "paged" : p == POOL_WINDOW ? "window" : "state"; }
inline int pool_sort_rank(int32_t p) { return p == POOL_PAGED ? 0 : p == POOL_STATE ? 1 : 2; }
inline int32_t pool_id(const std::string& name) { return name == "paged" ? POOL_PAGED : name == "window" ? POOL_WINDOW : POOL_STATE; }

// The Geometry fields OwnedKVManager reads plus the two environment knobs
// PagePool reads at construction (Python reads os.environ there; C++ components never read the environment).
struct KVGeometry {
  int32_t n_ranks = 4;
  int64_t page_tokens = 64, window = 128, window_page_tokens = 32, replay_tokens = 128;
  double reserve_watermark = 0.01;    // admit / migrate: int(reserve_watermark * num_pages) (double, truncated)
  PoolSpec paged, window_pool, state; // Geometry.pools() in this order
  int64_t hbm_pages = 0;              // FPR_KV_HBM_PAGES (kv_manager.hbm_pages(); paged pool only)
  std::string poison_mode;            // PHYS_POISON_ON_FREE: "" | "1" (all pools) | "pages" (paged + window)
  const PoolSpec& spec(int32_t pool) const { return pool == POOL_PAGED ? paged : pool == POOL_WINDOW ? window_pool : state; }
};
KVGeometry kv_geometry(const Geometry& g, int64_t hbm_pages = 0, const std::string& poison_mode = "");   // kv_manager.cpp

// ---------------------------------------------------------------- value types
struct CopyOp {                       // kv_manager.CopyOp, the same field order
  int32_t pool;                       // POOL_* (Python str)
  int32_t src_rank; int64_t src_page; int32_t dst_rank; int64_t dst_page; int64_t step;
};
struct AdmitResult { int32_t rank; int64_t hit_blocks, cached_tokens, replay_start, copied_blocks; std::vector<CopyOp> copies; };
struct KVUsage { int64_t used, cached_free, free, reserved, total, window_used, state_used; };   // usage(rank)
struct KVStats { int64_t copies = 0, copy_bytes = 0, evictions = 0, promoted = 0, demoted = 0; };   // OwnedKVManager.stats

// SeqKV.window: dict[int, int] logical 32-token block -> window page, in INSERTION order (Python dict order is part of
// the decisions: release order = free-list order, build_compact's window-op order). Small (<= ~130 entries).
class WindowTable {
 public:
  std::vector<std::pair<int64_t, int64_t>> items;     // (block, page), insertion order
  int64_t find(int64_t b) const { for (size_t i = 0; i < items.size(); ++i) if (items[i].first == b) return (int64_t)i; return -1; }
  bool contains(int64_t b) const { return find(b) >= 0; }
  int64_t get(int64_t b, int64_t dflt) const { int64_t i = find(b); return i < 0 ? dflt : items[(size_t)i].second; }
  void set(int64_t b, int64_t p) { int64_t i = find(b); if (i < 0) items.emplace_back(b, p); else items[(size_t)i].second = p; }
  int64_t pop(int64_t b) { int64_t i = find(b); int64_t p = items[(size_t)i].second; items.erase(items.begin() + i); return p; }
  void erase(int64_t b) { int64_t i = find(b); if (i >= 0) items.erase(items.begin() + i); }
  size_t size() const { return items.size(); }
  bool empty() const { return items.empty(); }
  int64_t min_key() const { int64_t m = items[0].first; for (auto& kv : items) if (kv.first < m) m = kv.first; return m; }
};

// kv_manager.SeqKV, every field. pages / window are shared by reference exactly as Python shares the list / dict
// objects: allocate() appends to *pages in place, commit() / trim_window() pop from *window in place, migrate() swaps in
// NEW objects (holders of the old ones keep them), free() moves the SeqKV to the deferred list. A StepBatch's SeqEntry
// holds the same shared_ptrs.
struct SeqKV {
  Rid seq_id; int32_t rank = 0;
  std::shared_ptr<std::vector<int64_t>> pages = std::make_shared<std::vector<int64_t>>();   // paged pool: page of block i
  int64_t num_registered = 0;         // leading blocks registered (or found) in the prefix cache
  std::shared_ptr<WindowTable> window = std::make_shared<WindowTable>();
  int32_t state_slot = -1;            // Python None = -1
  int64_t reserved = 0, cached_tokens = 0, replay_start = 0;
  int64_t wver = 0;                   // bumped on every change of `window`
  bool shadow = false;                // a CP helper's shadow (never hashed)
  int64_t cp_have = 0;                // shadow: leading blocks of the prompt its replica holds (WRITTEN BY THE SCHEDULER)
  int64_t inc = 0;                    // incarnation: unique per admit / open_shadow / migrate
  int64_t capacity_tokens() const { return (int64_t)pages->size(); }   // (sic: Python returns len(pages))
};

// CP helper shadow key: Python's tuple ("cp", rid, rank) as one Rid string.
inline Rid shadow_key(const Rid& rid, int rank) { return "cp\x1f" + rid + "\x1f" + std::to_string(rank); }

// ---------------------------------------------------------------- the manager
// OwnedKVManager, method for method: same arguments, same decisions, same free-list / LRU orders. Python returning None
// = Status::NO_ROOM (admit, migrate); Python assertion / KeyError = Status::BAD_ARG (unknown id) or INTERNAL (check()).
class IKVManager {
 public:
  virtual ~IKVManager() = default;
  virtual const KVGeometry& geom() const = 0;
  // ---- queries
  virtual std::vector<int64_t> lookup(const std::vector<Hash>& block_hashes) const = 0;   // per rank: leading hit blocks
  virtual int64_t free_pages(int rank) const = 0;                                          // num_free - reserved
  virtual KVUsage usage(int rank) const = 0;
  virtual int64_t pages_for(int64_t tokens) const = 0;                                     // ceil(tokens / 64)
  virtual void hit_after_cap(int64_t hit_blocks, int64_t prompt_len, int64_t* hit, int64_t* cached, int64_t* replay_start) const = 0;
  virtual const std::vector<int64_t>& block_table(const Rid& id) const = 0;              // = *seq(id)->pages
  virtual const WindowTable& window_table(const Rid& id) const = 0;                      // = *seq(id)->window
  virtual int32_t state_slot(const Rid& id) const = 0;
  virtual SeqKV* seq(const Rid& key) = 0;             // kv.seqs[key] (nullptr: absent); the scheduler writes cp_have
  virtual const SeqKV* seq(const Rid& key) const = 0;
  virtual size_t num_seqs() const = 0;                // bool(kv.seqs)
  virtual bool has_pinned() const = 0;                // bool(kv.pinned)
  virtual int64_t num_pages(int32_t pool, int rank) const = 0;   // kv.<pool>[rank].num_pages
  virtual int64_t num_free(int32_t pool, int rank) const = 0;    // .num_free(): uncached free + LRU
  virtual int64_t num_used(int32_t pool, int rank) const = 0;    // .num_used()
  virtual const KVStats& stats() const = 0;
  // adapter PhysKVFacade reads (defaults are exact; implementations need not override):
  //   usage = used / total if total else 0.0, summed over the ranks' paged pools
  virtual void paged_usage(int64_t* used, int64_t* total) const {
    *used = *total = 0;
    for (int r = 0; r < geom().n_ranks; ++r) { *used += num_used(POOL_PAGED, r); *total += num_pages(POOL_PAGED, r); }
  }
  //   estimate_cached_tokens: kv.seqs[rid].num_registered * page_tokens, 0 for an unknown rid (deferred frees: unknown)
  virtual int64_t estimate_cached_tokens(const Rid& id) const {
    const SeqKV* sk = seq(id);
    return sk ? sk->num_registered * geom().page_tokens : 0;
  }
  // ---- admission / growth (window_end: Python None = pass `end`)
  virtual Status admit(const Rid& id, int rank, const std::vector<Hash>& block_hashes, int64_t prompt_len, int64_t step,
                       int copy_from /* -1 = None */, AdmitResult* out) = 0;
  virtual bool can_allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) const = 0;
  virtual bool allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) = 0;
  // commit: Python asks block_hash(i) only for i in [num_registered, min(num_computed // 64, len(pages))): the vector
  // must cover those indices (index < min(nc // 64, pages)); other entries are not read.
  virtual void commit(const Rid& id, int64_t num_computed, const std::vector<Hash>& block_hashes) = 0;
  virtual void shrink_reservation(const Rid& id) = 0;
  // ---- CP shadows
  virtual bool open_shadow(const Rid& key, int rank, int64_t replay_start) = 0;
  virtual void trim_window(const Rid& id, int64_t lo_pos) = 0;
  // ---- release
  virtual void free_now(const Rid& id) = 0;                           // free(id)
  virtual void free_deferred(const Rid& id, int64_t defer_until) = 0;  // free(id, defer_until=s): any s, negative included
  virtual void flush_deferred(int64_t upto_step) = 0;
  // ---- moves and copies
  virtual Status migrate(const Rid& id, int dst, int64_t step, int64_t num_computed, std::vector<CopyOp>* copies) = 0;
  virtual std::vector<CopyOp> tier_copies(int64_t step, int64_t max_pages = 4096) = 0;
  virtual void release_copies(int64_t step) = 0;
  // ---- debug ($PHYS_POISON_ON_FREE): (pool, page) of `rank` freed / evicted since the last call
  virtual std::vector<std::pair<int32_t, int64_t>> take_poison(int rank) = 0;
  // ---- invariants (Python check(); the first failing assertion's text in last_error())
  virtual Status check() const = 0;
  virtual const std::string& last_error() const = 0;
};

std::unique_ptr<IKVManager> make_kv_manager(const KVGeometry& g);

}  // namespace fpr
