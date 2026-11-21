// d13.h -- engine/d13.py host side + kernels/head/hsd.py _build.
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
//
// v3: the interface as d13.py / d13s.py / rowexp.py / hsd.py compute it (v1's 4 methods lacked
// inputs the Python reads), on tables.h v4. Engine state the Python reads -> d13::Config (eng.cfg graphs / largest
// bucket, GRAPH_MAX_CTX, the scratch window slots), d13::PrevOf (engine.prev_of: device-token provenance), d13::HsdState
// (HSD's sampling states).
// Device work stays in the Python engine (rowexp kernels, HSD kernels, uploads): only host decisions and host blobs here.
// Implementation: fpr/core/src/d13/ (D13Host, HsdPrep); golden traces: src/d13/gtrace_d13.py; replay: tests/replay_d13.cpp.
#pragma once
#include <map>
#include <optional>

#include "common.h"
#include "tables.h"
#include "engine.h"

namespace fpr {
namespace d13 {

struct Config {                        // what d13.py reads from the engine / its module constants
  bool graphs = true;                  // eng.cfg.graphs
  int64_t largest_bucket = 320;        // eng.cfg.buckets[-1]
  int64_t graph_max_ctx = 262144;      // engine.GRAPH_MAX_CTX
  int64_t k1 = 4;                      // d13.K1
  int64_t delta_cap = 65536;           // d13.DELTA_CAP
  bool fb_reuse = true;                // d13.FB_REUSE (ENG_D13_FB_REUSE, default 1)
  int64_t scratch[W] = {0, 0, 0, 0};   // rx[r].scratch = pools[r].scratch_window_page * 32
};

struct Deltas {                        // d13._deltas: D12 block-table deltas -> per-slot device table coordinates
  std::vector<int64_t> rows, cols;     // np.repeat(slot, ln), start + arange
  std::vector<int32_t> pages;          // np.asarray(bt_upd_pages[:nu], np.int32)
  int64_t nu = 0;
  bool any = false;                    // bt_upd_slot non-empty (else nothing happens)
  bool direct = false;                 // nu > DELTA_CAP: fresh device tensors instead of the pinned staging
};
struct MirrorCall {                    // hooks_idx.mirror_deltas(r, rows, cols, pages, bt_upd_slot, bt_upd_start)
  int rank; std::vector<int64_t> rows, cols; std::vector<int32_t> pages; std::vector<int64_t> slot, start;
};
typedef void (*MirrorFn)(void* user, const MirrorCall& call);   // lever (a) (index-key mirror); nullptr = off

struct Desc {                          // kernels/rowexp Desc: one rank's per-step descriptor, sequences in row order
  int64_t S = 0, rows = 0;
  std::vector<int32_t> slot, row0, nrows, kind;
  std::vector<int64_t> start, replay, cached;
  std::vector<int32_t> ops;            // [n_ops * 3] (slot, block, page); block -1 = clear the slot's ring
  int64_t n_ops() const { return (int64_t)ops.size() / 3; }
};

struct PrevOf { Rid rid; bool has = false; int64_t rank = -1, index = -1; };   // engine.prev_of(rid)
struct OwnerSigSeq { Rid rid; int64_t state_slot, row0, n_rows, n_blocks, prev_index; };
struct OwnerSig {                      // d13._owner_sig: (n, ((rid, slot, row0, n_rows, n_blocks, prev index), ...), sample_idx)
  int64_t n = 0; std::vector<OwnerSigSeq> seqs; std::vector<int64_t> sample_idx;
  bool operator==(const OwnerSig& o) const;
};

// ---------------------------------------------------------------- HSD prepare (kernels/head/hsd.py HSD._build)
struct HsdConfig {                     // the HSD object's fixed state that _build reads
  int64_t k = 3;
  int64_t max_seqs = 80;               // HSD max_seqs (state slots per rank); MR = W * (max_seqs + 1)
  std::vector<int64_t> seq_buckets;    // HSD.seq_buckets (after FPR_HSD_CAP_BUCKETS)
  std::vector<int64_t> row_buckets;    // HSD.row_buckets = seq_buckets * (k + 1)
  int64_t nt_max = 1536;               // ranks0[0].b["aux"].shape[0] (tail rows): tail_buckets()
  bool pool = true;                    // window pools bound (served); false = private rings (tests)
  int64_t window = 128;                // drafter.WINDOW
  int64_t ring = 160;                  // drafter.RING (private-ring tail slots)
};
struct HsdState {                      // the sampling state object of a drafting sequence, as _build reads it
  double temperature = 1.0;            // getattr(st, "temperature", 1.0)
  int64_t seed = 0;                    // getattr(st, "seed", 0)
  bool drafts_valid = true;            // getattr(st, "drafts_valid", True)
  bool has_num_acceptable = false;     // getattr(st, "num_acceptable_drafts", None) is not None
  int64_t num_acceptable = 0;
  uint64_t ident = 0;                  // identity of the state object (_build re-reads temperature / seed only when the
                                       // slot's object changed: `skey[g] is not st_`); 0 = unknown -> always re-read
};
struct HsdBitmask {                    // one rank's grammar bitmask rows [n_s, words] in sample_rows order (xgrammar)
  int64_t n_s = 0, words = 0; const int32_t* rows = nullptr;
};
struct HsdStep {                       // hsd._Step (self.cur) + _build's return value
  bool v0 = false;                     // _build returned None: the per-element _prepare_v0 path (rows out of layout)
  int hi = 0;                          // the host blob set written (hsd._hi)
  int64_t S_pad = 0, D_pad = 0, NT = 0, n_req = 0, T = 0;
  std::vector<int64_t> n_s, n_seq, n_t, step;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> dseqs;   // per rank: (seq index in batch.seqs, request index)
  std::vector<std::string> rids;       // [WD]: rid per drafting request ("" = None)
  bool has_bm = false;
};

}  // namespace d13

class ID13Host {                       // engine/d13.py host side, decision for decision (+ HSD's host prepare)
 public:
  virtual ~ID13Host() = default;
  // d13._async_ok: "" = async, else the reason (the same strings)
  virtual std::string async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct) const = 0;
  // d13._deltas (rows / cols / pages; the mirror hook also gets the zero-length resets)
  virtual void deltas(int rank, const CompactTables& ct, d13::Deltas* out, d13::MirrorFn mirror = nullptr,
                      void* user = nullptr) const = 0;
  // RowExp.desc_of_compact (the descriptor D13 uploads per rank)
  virtual void desc(const CompactTables& ct, d13::Desc* out) const = 0;
  // d13._empty_eligible / _owner_sig / _guard_host (block counts) / _positions
  virtual bool empty_eligible(const StepBatch& sb, const CompactTables& ct) const = 0;
  virtual std::optional<d13::OwnerSig> owner_sig(int rank, const StepBatch& sb, const CompactTables& ct,
                                                 const std::vector<d13::PrevOf>& prev) const = 0;
  virtual std::vector<int64_t> guard_blocks(const StepBatch& sb) const = 0;
  virtual void positions(const StepBatch& sb, const std::vector<int64_t>& pos, const std::vector<int64_t>& own,
                         StepBatch* out) const = 0;
  // d13._fallback: device-resolved starts -> resolved StepBatches; their full tables (D13's private TableBuilder,
  // tables.h, row tokens = ct.tokens[:n]); accepted counts for k_advance
  virtual Status fallback_resolve(const std::vector<StepBatch>& b, const std::vector<std::vector<int64_t>>& starts,
                                  std::vector<StepBatch>* rb, std::string* err) const = 0;
  virtual Status fallback_tables(const std::vector<StepBatch>& rb, const std::vector<CompactTables>& ct,
                                 std::vector<RankTables>* out, std::string* err) = 0;
  virtual std::vector<int64_t> fallback_accepted(const StepBatch& rb,
                                                 const std::vector<std::pair<Rid, int32_t>>& accepted) const = 0;
  // d13._advance: HSD request index per sequence; *slice = ENG_D13_ADV_SLICE applies
  virtual std::vector<int64_t> advance_index(int rank, int64_t S, int64_t D_pad,
                                             const std::vector<std::pair<int64_t, int64_t>>& dseqs, bool* slice) const = 0;
  // d13s.Steady: signature equality + is_steady's batch-only conditions
  virtual bool steady_batches(const std::vector<StepBatch>& b, const std::vector<StepBatch>& prev,
                              const std::vector<Rid>& structured) const = 0;
  // hsd.py HSD._build into host blob set hsd_set() (the caller uploads it, then hsd_uploaded() = _upload's set flip)
  virtual Status hsd_prepare(const std::vector<StepBatch>& b,
                             const std::vector<std::vector<d13::HsdState>>& states /* [W][seqs] */,
                             const std::vector<d13::HsdBitmask>* bitmasks, d13::HsdStep* cur, std::string* err) = 0;
  virtual void hsd_uploaded() = 0;
  virtual uint8_t* hsd_blob(int set, int rank) = 0;      // host blob (pinned in serving), hsd_blob_bytes() each
  virtual int64_t hsd_blob_bytes() const = 0;
};

}  // namespace fpr
