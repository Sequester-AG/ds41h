// Do not edit: re-run the generator.
// Build: -O2 -std=c++17 -ffp-contract=off.
#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fpr {

// serve/geometry.py model / page constants (SLIDING_WINDOW, PAGE_TOKENS, WINDOW_PAGE_TOKENS: common.h)
constexpr int N_LAYERS = 40;
constexpr int HIDDEN = 5120;
constexpr int N_HEADS = 64;
constexpr int LAST_KV_SOURCE = 20;
constexpr int KV_SOURCE_LAYERS[4] = {2, 8, 14, 20};
constexpr int KV_SOURCE_RATIO[4] = {2, 2, 2, 1};   // per KV_SOURCE_LAYERS entry
constexpr int CANDIDATE_LAYERS[4] = {24, 28, 32, 36};
constexpr int CANDIDATES = 16384;
constexpr int SPEC_K = 3;
constexpr int64_t MAX_MODEL_LEN = 655360;
constexpr int64_t PAGE_BYTES = 116928;
constexpr int64_t WINDOW_PAGE_BYTES = 817344;
constexpr int64_t STATE_SLOT_BYTES = 98304;

// serve/geometry.py PoolSpec (kind: "paged" | "window" | "state"; index = CopyOp.pool)
struct PoolSpec { std::string name, kind; int64_t tokens_per_page, bytes_per_page, num_pages; };

// serve/geometry.py Geometry, field for field
struct Geometry {
  int64_t n_ranks = 4;
  int64_t spec_k = 3;
  int64_t page_tokens = 64;
  int64_t window = 128;
  int64_t window_page_tokens = 32;
  int64_t kv_bytes_per_rank = 34000000000;
  int64_t window_pool_bytes = 1400000000;
  int64_t state_pool_slots = 512;
  int64_t max_model_len = 655360;
  int64_t max_seqs = 256;
  int64_t max_seqs_per_rank = 80;
  int64_t max_prefill_tokens_per_rank = 4096;
  int64_t max_rows_per_rank = 0;
  int64_t max_tail_rows_per_rank = 1536;
  int64_t replay_tokens = 128;
  int64_t draft_window_tokens = 3;
  double reserve_watermark = 0.01;

  // SmallGeometry (tests) / any other subclass: its pools() exactly; empty = Geometry.pools()
  std::vector<PoolSpec> pool_override;

  int64_t rows_per_decode_seq() const { return spec_k + 1; }
  int64_t rows_cap() const {
    return max_rows_per_rank ? max_rows_per_rank : rows_per_decode_seq() * max_seqs_per_rank + max_prefill_tokens_per_rank;
  }
  std::vector<PoolSpec> pools() const {
    if (!pool_override.empty()) return pool_override;
    int64_t num = kv_bytes_per_rank - window_pool_bytes - state_pool_slots * STATE_SLOT_BYTES;
    int64_t n_pages = num / PAGE_BYTES - ((num % PAGE_BYTES != 0 && num < 0) ? 1 : 0);      // Python //
    int64_t wn = window_pool_bytes / WINDOW_PAGE_BYTES - ((window_pool_bytes % WINDOW_PAGE_BYTES != 0 && window_pool_bytes < 0) ? 1 : 0);
    return {PoolSpec{"paged", "paged", page_tokens, PAGE_BYTES, n_pages},
            PoolSpec{"window", "window", window_page_tokens, WINDOW_PAGE_BYTES, wn},
            PoolSpec{"state", "state", 0, STATE_SLOT_BYTES, state_pool_slots}};
  }

  // every field with its Python name, in declaration order: v(const char* name, T& member)
  template <class C, class V> static void visit(C& c, V&& v) {
    v("n_ranks", c.n_ranks);
    v("spec_k", c.spec_k);
    v("page_tokens", c.page_tokens);
    v("window", c.window);
    v("window_page_tokens", c.window_page_tokens);
    v("kv_bytes_per_rank", c.kv_bytes_per_rank);
    v("window_pool_bytes", c.window_pool_bytes);
    v("state_pool_slots", c.state_pool_slots);
    v("max_model_len", c.max_model_len);
    v("max_seqs", c.max_seqs);
    v("max_seqs_per_rank", c.max_seqs_per_rank);
    v("max_prefill_tokens_per_rank", c.max_prefill_tokens_per_rank);
    v("max_rows_per_rank", c.max_rows_per_rank);
    v("max_tail_rows_per_rank", c.max_tail_rows_per_rank);
    v("replay_tokens", c.replay_tokens);
    v("draft_window_tokens", c.draft_window_tokens);
    v("reserve_watermark", c.reserve_watermark);
  }
};

}  // namespace fpr
