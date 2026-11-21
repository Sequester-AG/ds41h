// gt_build.h -- P1 builder: the HOST side of a mixed step's indexer tables. = fpr/kernels/indexer2f/tables2f.py GeneralTables.upload up to (not including) the device work, decision
// for decision: _deltas (D12 block-table deltas present?), the sequence-contiguity check (else hooks_idx's host path),
// crow / pr0 / B / IR / max position, the capacity rule (B > Bc -> Bc = ceil64(B)), layout() + _fill() of the pinned
// int32 staging block (idx2f_tab.cu reads it: keep in sync), the host row -> indexer-row map (carry_candidates), the
// shared-key prefill groups (idx2f.pf_groups) and OwnedIndexer.plan_bounds' host bounds (max_len per index layer, width
// per KV source). The device part (one cudaMemcpyAsync of the block + idx2f_tab's 2 kernels) and the OwnedIndexer row
// views stay with the caller (nd's pre-step: the handover's FprEhMixedBucketRank views).
// Golden: core/tests/replay_gt.cpp on tests/p1_record_gt.py traces (record_d13_cpu.py's workloads, every D13 fallback
// step's RankTables of every rank next to tables2f's own host code).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "../../include/tables.h"

namespace fpr {
namespace p1 {

enum class GtStatus : int {
  OK = 0,          // tables built
  NO_DELTAS = 1,   // the tables carry no D12 deltas: GeneralTables.upload -> None (hooks_idx keeps its host path)
  HOST_PATH = 2,   // rows not sequence-contiguous in sequence order: -> None (host path)
  IDLE = 3,        // no rows on this rank (the dict's idle entry: B 1, IR 4, max_seq_len 1, row2ir all -1)
};
const char* gt_status_name(GtStatus s);

constexpr int GT_NLAYERS = 8;                           // idx2f.LAYERS (2, 8, 14, 20, 24, 28, 32, 36)
constexpr int GT_LAYERS[GT_NLAYERS] = {2, 8, 14, 20, 24, 28, 32, 36};
constexpr int GT_RATIO[GT_NLAYERS] = {2, 2, 2, 1, 1, 1, 1, 1};
constexpr int GT_NSRC = 4;                              // KV sources = the first 4 layers

struct GtHost {
  GtStatus status = GtStatus::OK;
  int64_t n = 0, S = 0, U = 0, P = 0;                   // rows, sequences, delta records, delta pages
  int64_t B = 0, IR = 0;                                // pseudo-requests (4 rows each), indexer rows
  int64_t max_seq_len = 1;                              // max position + 1 (the dict's max_seq_len)
  int64_t Bc = 0;                                       // capacity after this step (grows to ceil64(B) when B > Bc in)
  bool grew = false;
  std::array<int64_t, 13> o{};                          // layout(): word offsets ps, seqslot, sslot, row0, nrows, pr0,
                                                        // crow, uk, us, ul, uo, pages, total
  std::vector<int32_t> stage;                           // the staging block, o[12] words
  std::vector<int64_t> row2ir;                          // [R]: row -> 4 * pr0[s] + offset in its sequence, -1 past n
  std::vector<int64_t> pf_groups;                       // idx2f.pf_groups over the sequences' [pr0[s], pr0[s + 1])
  std::array<int64_t, GT_NLAYERS> max_len{};            // plan_bounds: max_seq_len // ratio
  std::array<int64_t, GT_NSRC> width{};                 // plan_bounds: max(256, ceil256(max_len)) of the KV sources
};

// tab: the rank's RankTables (TableBuilder.build: seq.bt_upd_* = the D12 deltas, has_deltas = the Python dict had
// "bt_upd_seq"); R: the bucket's rows (row2ir length); Bc_in: the table set's capacity before the step.
GtStatus gt_host(const RankTables& tab, bool has_deltas, int64_t R, int64_t Bc_in, GtHost* out);

// tables2f.layout(n, S, U, P)
std::array<int64_t, 13> gt_layout(int64_t n, int64_t S, int64_t U, int64_t P);

// ---- nd's native mixed step, phase 1. BAD_ARG = the whole
// step runs embedded, decided before any device work: no D12 deltas or rows out of sequence order (Python's
// _upload_general_host), B > snap.Bc (a native table set never regrows), the HBM index-key mirror (its host map is Python
// state). An idle rank is OK (the idle entry).
// STATE RULE: the builder is PURE (snap + inputs -> outputs), no hidden host state:
//  - capacity: Bc is the bucket's (the handover view's, fixed at ensure). An embedded step that grows a set (Python
//    GeneralTables._alloc) re-creates its tensors; the handover's generation bump re-exports the views (and snap.Bc);
//  - staging ring: tables2f.Stage's ring of the device (idx2f_tab.cu ring id, pinned + device blocks) is SHARED by
//    Python and nd; a slot is safe once its event completed (E.wait before every write), so each side keeps its own
//    cursor and they interleave freely; no cursor resync;
//  - the mirror map (FPR_KV_IDX_MIRROR, off when served) stays Python: mirror on = BAD_ARG.
struct GtSnap {
  int64_t Bc = 0;                       // the bucket's table capacity in pseudo-requests (the view's)
  bool mirror = false;                  // FPR_KV_IDX_MIRROR on
  bool has_deltas = true;               // the tables carry D12 deltas (TableBuilder: always; false only for test tables)
  int64_t ring_cap = 0;                 // FprEhRankState.idx_ring_cap (int32 words per slot); 0 = unchecked
};
Status gt_host(const GtSnap& snap, char role, int rank, int64_t R, const StepBatch& sb, const RankTables& tab, GtHost* out);

}  // namespace p1
}  // namespace fpr
