// p1_cmirror.h -- CUDA-free layout mirrors of the C structs P1's native ops hand to the GPU side
// the pieces' per-step setters' structs (fpr/engine/mixed_cxx/pieces_attn.h) and the C ABI Args of kernels/p1 (p1_cabi.h)
// and kernels/indexer2f/idx2f_tab_cabi.h, for core2b's owner_p1.cpp (no CUDA there). Every size / offset below is
// asserted against the real headers by core/tests/p1_mirror_check.cpp (compiled WITH the CUDA headers: run it whenever
// either side changes). FprTrRecord (CUfunction + key class) is carried as its bytes.
#pragma once
#include <cstddef>
#include <cstdint>

namespace fpr {
namespace p1 {
namespace cm {

struct Call { void* stream; int32_t device, rank; };                     // = FprCall (16 B)
struct alignas(8) TrRecordBytes { uint8_t b[2120]; };                    // = FprTrRecord

struct IdxASlice { int64_t b0, b1, gst_off, g; };
struct IdxAStep {                                                        // = FprIdxAStep
  int32_t valid, idle, logits, global_rows;
  int64_t B, IR, nv, nreq;
  int32_t skip[4], refuse[4];
  int64_t max_len[4], width[4];
  int32_t nsl[2], pad1[2];
  IdxASlice sl[2][64];
  const int32_t* gst;
  int64_t max_seq_len;
};
struct IdxChunk { int64_t b0, b1; void *ublk, *uslot, *nunion, *cblk, *clen, *upk, *comp, *bm; int64_t ld_cblk; };
struct AttnIdxStep {                                                     // = FprAttnIdxStep
  int32_t valid, idle, use2, single, nchunks, nwords;
  int64_t B, IR, nv;
  int32_t skip[8];
  int32_t prep_done, pad1;
  IdxChunk ch[16];
};
struct PfGroup { int64_t a, b, N, M; void* kv; int64_t q0, q1; int32_t rec, gy; uint64_t vals[16]; };
struct PfLayout { int32_t ratio, ngroups, disjoint, pad; PfGroup g[32]; };
struct AttnPfStep {                                                      // = FprAttnPfStep
  int32_t valid, nlayouts, nrecs, pad0;
  int64_t n, row_lo, row_hi;
  const void *seq_lens, *gather_lens, *bt, *wbt, *inv;
  void* loc;
  int64_t ld_bt, ld_wbt, n_inv;
  const void* seq_lens_r[4];
  void* comb[2];
  void* clen[2];
  int64_t cw[2];
  PfLayout lay[4];
  TrRecordBytes rec[8];
  int32_t ws_ratio, comp_src;
  int32_t ck_valid[2], ck_ratio[2], ck_isrc[2];
};
// C ABI Args (nd's cabi fills `call`)
struct TabWaitArgs { Call call; int64_t ring_id, slot; };                                         // FprIdx2fTabWaitArgs
struct TabRunArgs { Call call; int64_t hid, ring_id, slot; const void* host; void* dev; int64_t n, S, U, P, B; };   // FprIdx2fTabRunArgs
struct PfWbtArgs {                                                                                // FprP1PfWbtArgs
  Call call; const int32_t* si; int64_t ld_si; const int32_t* sl; const int64_t* seq; int64_t S; int32_t* wbt;
  int64_t ld_wbt, wmax, wpt; const int32_t* seq_lens; const int32_t* gather_lens; const int32_t* qsl; int64_t row_lo;
};
struct PfInvArgs { Call call; int32_t* inv; int64_t n_inv; const int32_t* bt; int64_t ld_bt; const int32_t* nb; int64_t S, max_nb;
                   const int32_t* seq_lens; };                                                    // FprP1PfInvArgs
struct CarryArgs { Call call; const int64_t* ir2row_T; const uint8_t* valid_T; const int64_t* ix; const int64_t* row2ir_A;
                   const int32_t* cand_A; int32_t* cand_T; int64_t IR_T; };                        // FprP1CarryArgs

static_assert(sizeof(Call) == 16 && sizeof(TrRecordBytes) == 2120, "mirror");
static_assert(sizeof(IdxAStep) == 4272 && offsetof(IdxAStep, sl) == 160 && offsetof(IdxAStep, gst) == 4256, "FprIdxAStep");
static_assert(sizeof(AttnIdxStep) == 1496 && offsetof(AttnIdxStep, ch) == 88 && sizeof(IdxChunk) == 88, "FprAttnIdxStep");
static_assert(sizeof(AttnPfStep) == 41824 && offsetof(AttnPfStep, lay) == 192 && offsetof(AttnPfStep, rec) == 24832 &&
                  offsetof(AttnPfStep, ws_ratio) == 41792 && sizeof(PfGroup) == 192 && sizeof(PfLayout) == 6160, "FprAttnPfStep");
static_assert(sizeof(TabWaitArgs) == 32 && sizeof(TabRunArgs) == 96 && sizeof(PfWbtArgs) == 120 && sizeof(PfInvArgs) == 80 &&
                  sizeof(CarryArgs) == 72, "C ABI Args");

}  // namespace cm
}  // namespace p1
}  // namespace fpr
