// cp_build.h -- P1 builder: the HOST side of a CP prefill step's exchanges. = engine/cp.py CPPlan.__init__ (the step's CP groups from the batches' CP roles, every rank's page lists
// pages[r][q] of every sub-chunk's blocks, the window halo src / dst pages of every later sub-chunk) + the message order of
// mixed_cxx_attn.cp_step (= cp.py's x1a_send / x2_send loops, layer by layer): one message per (layer, group, sender,
// receiver, section). The index lists go into one int64 blob per rank (the device that reads them: a sender's own lists,
// a receiver's own lists); a message names its lists by blob offset. The issue side (the CP exchange pieces,
// pieces_attn.h FprCpMsg) fills the device pointers from the uploaded blobs and the pool views.
// Golden: core/tests/replay_cp.cpp on tests/p1_record_cp.py traces.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "../../include/tables.h"

namespace fpr {
namespace p1 {

enum { CP_X1A = 1, CP_X2 = 2 };         // = pieces_attn.h FPR_CP_X1A / FPR_CP_X2

struct CpSection { int32_t layer; int64_t comp_off, comp_len, idx_off, idx_len; };   // cp.py _sections(L), KV sources only
struct CpGeom {
  std::vector<CpSection> sources;       // (2, 8, 14, 20): PAGE_OFFSETS["comp{L}" / "idx{L}"], RECORDS_PER_PAGE[L] * 584 / 132
  int64_t wlb = 0;                      // WINDOW_LAYER_PAGE_B
  int64_t win = 128, wpt = 32, page_tokens = 64;
};
struct CpMsgT {                         // FprCpMsg without the device pointers
  int32_t src_rank, dst_rank, layer, kind;
  int64_t n, len, off;
  int64_t src_off, dst_off;             // int64 offsets into blob[src_rank] / blob[dst_rank]
};
struct CpPlanHost {
  int64_t n_groups = 0;
  std::array<std::vector<int64_t>, 8> blob;   // per rank: the index lists it reads (pages / halo window pages)
  std::vector<CpMsgT> msgs;
};

// batches: the step's StepBatch per rank (seqs with CP roles, block / window tables); layers: the segment's layers
// (messages for L <= 20); idx_on: the engine has an indexer (X1a on the KV-source index layers). BAD_ARG where CPPlan
// asserts (a group's entries not on its subs' ranks) or a referenced block / window block is missing (Python: IndexError /
// KeyError).
Status cp_host(const std::vector<StepBatch>& batches, const std::vector<int32_t>& layers, bool idx_on, const CpGeom& geom,
               CpPlanHost* out);

// ---- the issue side (no device work here): the step's FprCpMsg array (pieces_attn.h; CpMsgDev has its exact layout,
// 120 B) from the plan, the per-rank DEVICE copies of plan.blob[r] (uploaded by the caller, idx_dev[r]) and the rank's
// pool views: X1a = cp.py _pg(r) (the paged pool viewed [n_pages, page_bytes]), X2 = _wsec(r) (the window pool viewed
// [n_window + 1, WINDOW_PAGE_BYTES]); mail_stream[r] = the receiver's CP-only stream (cp.py _mail_stream). The array goes
// to pieces_attn's attn_cp_step (its C entry), the exchange pieces read it.
struct CpRankViews {
  int32_t device = -1;
  const int64_t* idx_dev = nullptr;     // device copy of plan.blob[rank]
  int64_t idx_cap = 0;                  // its int64 capacity
  void* paged = nullptr; int64_t paged_rows = 0, paged_ld = 0;     // _pg(r)
  void* window = nullptr; int64_t window_rows = 0, window_ld = 0;  // _wsec(r)
  void* mail_stream = nullptr;          // cudaStream_t
};
struct CpMsgDev {                       // = pieces_attn.h FprCpMsg, field for field
  int32_t src_rank, dst_rank, layer, kind;
  int64_t n, len, off;
  const void* src_idx;
  const void* dst_idx;
  void* src_base;
  void* dst_base;
  int64_t src_rows, src_ld, dst_rows, dst_ld;
  int32_t src_dev, dst_dev;
  void* mail_stream;
};
static_assert(sizeof(CpMsgDev) == 120, "CpMsgDev = FprCpMsg (120 B)");
// BAD_ARG: a missing view, a blob larger than its device copy, an offset / section outside the pool's row.
Status cp_fill(const CpPlanHost& plan, const std::vector<CpRankViews>& views, std::vector<CpMsgDev>* out);

}  // namespace p1
}  // namespace fpr
