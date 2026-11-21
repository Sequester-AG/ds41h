// steps_build.h -- P1 builders of the attention / indexer PIECES' per-step structs: what mixed_cxx_attn.step_params computes in Python before a segment's span, from phase 1's host
// results. Torch-free; goldens: core/tests/replay_steps.cpp on tests/p1_record_steps.py traces (the Python functions
// themselves on recorded steps).
//   idxa_step  FprIdxAStep (Stage B, A-role index layers): mixed_cxx_attn._idxa_values
//   idx_step   FprAttnIdxStep (T-role consumer layers 24 .. 36): mixed_cxx_attn._idx_step (pointers = the EagerWorkspace's
//              carving: prep_bufs / logits / bm, and the candidate rows)
//   pf_step    FprAttnPfStep (bucket A's prefill attention, ENG_ATTN_PF): mixed_cxx_attn._pf_step's layouts (ratios by
//              their no-prealloc need, largest first), groups (a, b, N, M, the workspace view, q0 / q1) and per group the
//              combine kernel's Triton values from a recorded template + the group's overrides; the template = the first
//              one whose key class accepts them (the caller's check = triton_replay.h fpr_tr_check)
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gt_build.h"
#include "pf_build.h"

namespace fpr {
namespace p1 {

struct IdxaSlice { int64_t b0, b1, gst_off, g; };
struct IdxaStepHost {                   // FprIdxAStep's values (pieces_attn.h) + the pf4 group starts to upload
  int32_t valid = 1, idle = 0, logits = 0;
  int64_t B = 0, IR = 0, nv = 0, nreq = 0;
  std::array<int32_t, 4> skip{}, refuse{};   // refuse: rows > LONG_ROW -> 1 (Python) / 2
  std::array<int64_t, 4> max_len{}, width{};
  std::array<std::vector<IdxaSlice>, 2> sl;   // [0] ratio 2 (layers 2 / 8 / 14), [1] ratio 1 (layer 20)
  int32_t global_rows = 0;              // LayerIn.global_rows (cfg) and max_seq_len (gt): the stock top-k's inputs
  int64_t max_seq_len = 0;
  std::vector<int32_t> gst;             // every pf4 slice's group starts, relative to its b0 (int32 device buffer)
};
struct IdxaCfg {
  int64_t budget = 256LL << 20;         // EagerWorkspace budget (bytes of fp32 dense logits per call)
  int32_t logits = 1;                   // 1 = idx2f.LOGITS "f8", 2 = "cand"
  bool pf = true;                       // OwnedIndexer.pf (IDX2F_PF) and its pf_groups set
  int64_t long_row = 262144, next_n = 4, max_call_b = 1024, max_slices = 64;
  bool stock = false;                   // ENG_MIXED_CXX_IDXA_STOCK: long layers take the piece's stock branch (refuse 2)
  int64_t global_rows = 0;              // the step's rows of every rank (hooks_idx.upload_general's grows)
};
// gt: the A bucket's gt_host result (OK or IDLE). BAD_ARG: more slices than FprIdxAStep holds (Python: None).
Status idxa_step(const GtHost& gt, const IdxaCfg& cfg, IdxaStepHost* out);

struct IdxChunkHost {                   // FprAttnIdxChunk: device addresses (the EagerWorkspace's views)
  int64_t b0 = 0, b1 = 0;
  uint64_t ublk = 0, uslot = 0, nunion = 0, cblk = 0, clen = 0, upk = 0, comp = 0, bm = 0;
  int64_t ld_cblk = 0;
};
struct AttnIdxStepHost {                // FprAttnIdxStep's values
  int32_t valid = 1, idle = 0, use2 = 0, single = 0, nwords = 0;
  int64_t B = 0, IR = 0, nv = 0;
  std::array<int32_t, 8> skip{};
  std::vector<IdxChunkHost> ch;         // nchunks = ch.size() (<= 16)
};
struct IdxCfg {
  int64_t budget = 256LL << 20, max_cb = 512;   // EagerWorkspace.of(dev, budget, max_cb)
  bool cand_ascending = true;           // bucket A's layer 20 wrote ascending candidates (not rows > LONG_ROW)
  int64_t long_row = 262144, max_chunks = 16;
};
struct IdxWs { uint64_t prep = 0, f32 = 0, bmax = 0, cand = 0; };   // EagerWorkspace .prep / .f32 / .bmax, the T cand rows
// gtT: the T bucket's gt_host result. The caller has checked the served-branch conditions (fast mode, unfused, no mirror,
// idx2f.LOGITS "f8", cand rows of CAND_K). BAD_ARG: more than 16 chunks (Python: None).
Status idx_step(const GtHost& gtT, const IdxCfg& cfg, const IdxWs& ws, AttnIdxStepHost* out);

struct PfGroupHost {
  int64_t a = 0, b = 0, N = 0, M = 0;
  uint64_t kv = 0;                      // the group's workspace view (bf16): ws + 2 * offset
  int64_t q0 = 0, q1 = 0;               // its rows [qsl[a], qsl[b])
  int32_t rec = -1, gy = 0;             // index into PfStepHost.recs, the record's grid y
  std::vector<uint64_t> vals;           // the combine's Triton values (Python argument order)
};
struct PfLayoutHost { int32_t ratio = 0; bool disjoint = true; std::vector<PfGroupHost> g; };
struct PfStepHost {
  int64_t n = 0, row_lo = 0, row_hi = 0;
  std::vector<PfLayoutHost> lay;        // <= 4, in _pf_step's order
  std::vector<int32_t> recs;            // template indices used (<= 8): FprAttnPfStep.rec[i] = templates[recs[i]].record
};
struct PfStepViews {                    // device addresses (FprEhMixedPf / FprEhRankState views)
  uint64_t ws = 0;                      // pf_ws[r] (bf16)
  uint64_t comb[2] = {0, 0}, clen[2] = {0, 0};   // index widths 640 / 128 (int32 [R_A, cw], [R_A])
  uint64_t loc = 0; int64_t loc_ld = 512;        // int32 [R_A, 512]
  uint64_t qsl = 0, seq_lens = 0, gather_lens = 0;   // the plan blob's int32 arrays
};
struct CombTemplate {                   // one recorded key class of the combine kernel (recorded at ensure time)
  const void* record = nullptr;         // FprTrRecord
  int32_t ratio = 0, top_k = 0, gy = 0;
  std::vector<std::string> names;       // Python-level argument names (triton_replay Rec.names)
  std::vector<uint64_t> base;           // its recorded values
};
using TrCheck = int (*)(const void* record, const uint64_t* vals);   // FPR_OK = the values fall in the key class
// Format "FPCT" v1, little endian, 8-B aligned sections
//   u32 magic 0x54435046 ("FPCT"), u32 version 1, u32 count, u32 record_size (= sizeof(FprTrRecord) of the exporter)
//   per template: i32 ratio, i32 top_k, i32 gy, i32 n | record_size bytes (FprTrRecord) | n x (u16 len, len bytes)
//                 padded to 8 | n x u64 base values
// The parsed templates' `record` points INTO `buf` (append-only, alive with the engine). BAD_ARG: magic / version /
// record_size / a truncated buffer.
constexpr uint32_t COMB_TPL_MAGIC = 0x54435046u, COMB_TPL_VERSION = 1u;
Status parse_comb_templates(const void* buf, int64_t bytes, int64_t record_size, std::vector<CombTemplate>* out);
// pf: phase 1's PfHost (prealloc workspace); ratios: the A bucket's layer ratios (0 / 1 / 2). BAD_ARG (Python: None):
// more than 4 ratios or 32 groups, no template accepting a group's values, more than 8 records, > 16 values.
Status pf_step(const PfHost& pf, const std::vector<int32_t>& ratios, const PfStepViews& v, const std::vector<CombTemplate>& tpl,
               TrCheck check, const PfConsts& k, PfStepHost* out);

}  // namespace p1
}  // namespace fpr
