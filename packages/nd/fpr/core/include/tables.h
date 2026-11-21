// tables.h -- serve/interfaces.py build_step_batch, serve/worker.py TableBuilder / build_compact / copies.
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
// v4: every Python field (SeqEntry.tail_row0 / block_table / window_table by reference, StepBatch.poison
// RankTables, CompactTables rank / step / k1 / tail_rows / sample_idx / rids), build_step_batch, TableBuilder (build +
// build_compact, per-instance `sent` / `cstate`), the Worker.copies grouping. Implementation: fpr/core/src/tables/tables.cpp;
// replay: fpr/core/tests/replay_tables.cpp (+ src/tables/gtrace_tables.py, the recorder).
#pragma once
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "kv.h"
#include "scheduler.h"

namespace fpr {

constexpr int64_t PAD = -1;                        // interfaces.PAD

// ---------------------------------------------------------------- interfaces.SeqEntry / StepBatch
struct SeqEntry {                     // interfaces.SeqEntry, every field, Python order
  Rid rid; std::string kind;          // "decode" | "prefill" (a CP helper's entry is "prefill" with its prompt's rid)
  int32_t row0 = 0, n_rows = 0;
  int64_t start = 0, num_computed = 0, cached_upto = 0, replay_start = 0;
  bool last_chunk = false;
  // the KV manager's SeqKV.pages / .window objects BY REFERENCE (Python holds the list / dict): later appends / pops are
  // seen, a migrate's new objects are not (valid until the next schedule(), as in Python)
  std::shared_ptr<const std::vector<int64_t>> block_table;
  std::shared_ptr<const WindowTable> window_table;
  int32_t state_slot = -1;            // None = -1
  int32_t tail_row0 = 0;              // first row that also runs layers 21..39
  int64_t n_blocks = 0;               // pages of block_table this step covers
  bool score = false, forced = false;
  int32_t uid = -1;
  int64_t known = -1, wver = -1;
  bool has_cp = false; CPRole cp;     // Python cp = None <-> has_cp = false
  std::vector<Image> images;
  int64_t inc = 0;
};
struct StepBatch {                    // interfaces.StepBatch: rows of one rank, host order
  int32_t rank = 0; int64_t step = 0; int32_t n_rows = 0;
  std::vector<int64_t> positions;
  std::vector<int32_t> seq_slot;
  std::vector<int64_t> paged_slot, window_slot;
  std::vector<int32_t> tail_rows, sample_rows;
  std::vector<SeqEntry> seqs;
  std::vector<CopyOp> copies;
  std::vector<std::pair<int32_t, int64_t>> poison;   // debug: (pool, page) to fill with 0xFF before this step's copies
};

// What build_step_batch reads from the KV manager: kv.seqs[w.rid] and kv.take_poison(rank) (Python: getattr, optional).
struct KVViews {
  std::function<const SeqKV*(const Rid&)> seq;
  std::function<std::vector<std::pair<int32_t, int64_t>>(int)> take_poison;    // empty: no poison
};
// interfaces.build_step_batch(plan, kv, geom, rank). BAD_ARG where Python raises (unknown sequence, missing window page).
Status build_step_batch(const StepPlan& plan, const KVViews& kv, int64_t page_tokens, int64_t window_page_tokens, int rank,
                        StepBatch* out, std::string* err = nullptr);
Status build_step_batch(const StepPlan& plan, IKVManager& kv, int rank, StepBatch* out, std::string* err = nullptr);

// ---------------------------------------------------------------- worker.RankTables (TableBuilder.build)
struct RankTablesSeq {                // RankTables.seq dict, Python key names; dtypes as the torch tensors
  std::vector<int32_t> state_slot, uid; std::vector<int64_t> start; std::vector<int32_t> n_rows, kind, last_chunk, score, forced;
  std::vector<int64_t> replay_start, cached_upto; std::vector<int32_t> n_blocks, bt_reset;
  std::vector<int32_t> bt_upd_seq, bt_upd_start, bt_upd_len, bt_upd_pages;
  bool has_block_table = false; int64_t bt_rows = 0, bt_cols = 0; std::vector<int32_t> block_table;   // full_block_tables
};
struct RankSeqRow { Rid rid; int32_t row0, n_rows; std::string kind; bool last_chunk; };
struct RankTables {                   // worker.RankTables (CPU tensors there; flat row-major vectors here)
  int32_t n = 0;
  std::vector<int64_t> positions, tokens, own_slot;
  std::vector<int32_t> si;            // [n, 128]
  std::vector<int32_t> sl;
  std::vector<int32_t> ti_sources;    // stub_indexer: the sources in TableBuilder.sources order
  std::vector<std::vector<int32_t>> ti, tl;   // per source: ti [n, 512], tl [n]
  std::vector<int64_t> sample_rows;
  std::vector<RankSeqRow> seq_rows;
  std::vector<int32_t> seq_slot;
  std::vector<int64_t> paged_slot, tail_rows, sample_idx;
  RankTablesSeq seq;
  std::vector<int64_t> targets;
};

// ---------------------------------------------------------------- rowexp_ref.CompactTables (build_compact)
struct CompactTables {                // D13 / D12 format: per-seq records + deltas + window ops (every Python field)
  int32_t rank = 0; int64_t step = 0; int32_t k1 = 0;
  std::vector<int64_t> slot, row0, n_rows, kind, start, replay_start, cached_upto;
  std::vector<int64_t> win_slot, win_blk, win_page;                  // clears (blk -1, page -1) first, then sets
  std::vector<int64_t> bt_upd_slot, bt_upd_start, bt_upd_len, bt_upd_pages;
  std::vector<int64_t> tokens, tail_rows, sample_idx, targets;
  std::vector<Rid> rids;
  std::vector<int64_t> uid, last_chunk, score, forced;
  int64_t n = 0;                      // Python property: sum(n_rows)
};

// ---------------------------------------------------------------- Worker.copies grouping
struct PageCopyBatch {                // FPR_PAGECOPY: copies grouped by (pool, src_rank, dst_rank) -> one kernel each
  int32_t pool, src_rank, dst_rank; std::vector<int64_t> src_pages, dst_pages; int64_t page_bytes;
};
struct CopyRun { int32_t pool, src_rank; int64_t src_page0; int32_t dst_rank; int64_t dst_page0, n; };   // copy_runs
// Worker.copies: the ops dict over every batch's copies (key (pool, src_rank, src_page, dst_rank, dst_page), first-seen
// order, the last op of a key wins), then the FPR_PAGECOPY groups in first-seen order, or interfaces.copy_runs (sorted by
// pool NAME, then src_rank, dst_rank, src_page: pool_sort_rank), and the engine hook's state-slot copies.
void dedup_copies(const std::vector<StepBatch>& batches, std::vector<CopyOp>* ops);
void page_copy_batches(const std::vector<CopyOp>& ops, std::vector<PageCopyBatch>* out);
void copy_runs(const std::vector<CopyOp>& copies, std::vector<CopyRun>* out);
void state_slot_copies(const std::vector<CopyOp>& ops, std::vector<std::array<int64_t, 4>>* out);   // (sr, sp, dr, dp)

// ---------------------------------------------------------------- worker.TableBuilder
using TokensOf = std::function<int64_t(const Rid& rid, int64_t pos)>;   // TableBuilder.tokens_of (Python's default is
                                                                         // its per-process str hash: pass row_tokens)
// The ITables vtable ABI this header declares: 1 = build / build_compact; 2 = +
// save_state / restore_state (appended). Bump on ANY change of ITables' virtuals. fpr_kvtables exports the value it was
// built with (extern "C" fpr_tables_abi(), tables_abi()); callers holding an ITables* from another .so compare it with
// theirs before calling a virtual beyond slot 1 (an older .so's vtable is shorter: calling past it is undefined).
#define FPR_TABLES_ABI 2

class ITables {                       // one TableBuilder instance (the worker's, D13's private one, engine tb / tb_tail)
 public:
  virtual ~ITables() = default;
  // TableBuilder.build(sb, row_tokens): row_tokens nullptr = tokens_of(rid, pos)
  virtual Status build(const StepBatch& sb, const std::vector<int64_t>* row_tokens, RankTables* out, std::string* err = nullptr) = 0;
  // worker.build_compact(tb, sb, k1, row_tokens): shares this instance's state (cstate)
  virtual Status build_compact(const StepBatch& sb, int32_t k1, const std::vector<int64_t>* row_tokens, CompactTables* out,
                               std::string* err = nullptr) = 0;
  // an opaque copy of this builder's delta state (build's D12 `sent_`
  // build_compact's `cstate_`): save before a native build, restore when the step falls back to Python, so Python's rebuild
  // of the same step emits exactly the deltas it would have emitted without the native attempt. Restore only a state this
  // builder saved. Appended LAST: the earlier vtable slots are unchanged. Defaults: nullptr = not supported (the caller
  // refuses the native build), restoring nullptr = no-op.
  virtual std::shared_ptr<const void> save_state() const { return nullptr; }
  virtual void restore_state(const std::shared_ptr<const void>& state) {}
};

class TableBuilder final : public ITables {
 public:
  std::vector<int> sources{2, 8, 14, 20};
  TokensOf tokens_of;
  bool stub_indexer = false, full_block_tables = false;

  Status build(const StepBatch& sb, const std::vector<int64_t>* row_tokens, RankTables* out, std::string* err = nullptr) override;
  Status build_compact(const StepBatch& sb, int32_t k1, const std::vector<int64_t>* row_tokens, CompactTables* out,
                       std::string* err = nullptr) override;
  std::shared_ptr<const void> save_state() const override;                   // copy of sent_ + cstate_
  void restore_state(const std::shared_ptr<const void>& state) override;    // throws std::invalid_argument on a foreign state

 private:
  struct State;
  struct Sent { int32_t uid; int64_t inc; int64_t nb; };
  std::map<std::pair<int32_t, int32_t>, Sent> sent_;            // (rank, state_slot) -> ((uid, inc), blocks sent)
  struct CState { int32_t uid; int64_t inc; int64_t blocks; int64_t wver; WindowTable ws; };
  std::map<std::pair<int32_t, int32_t>, CState> cstate_;        // (rank, slot) -> [ident, blocks, wver, {blk: page}]
};

std::unique_ptr<ITables> make_tables();

}  // namespace fpr
