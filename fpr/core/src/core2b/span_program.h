// span_program.h -- the C++ span program of one mixed-step bucket segment: engine/mixed_cxx/mixed_cxx.py _issue_span's per-layer decisions for the case where EVERY piece is C++ (no Python between
// pieces), over piece handles built at ENSURE time (checklist C2: exported per (bucket view, rank, layer, kind)).
//
// Per segment (bucket A: layers 0..20, bucket T: 21..39 + final), per rank, in mixed_cxx's order:
//   [embed]                                   owner "embed" (the first segment only)
//   per layer L:
//     engram_bd (L in ENGRAM_LAYERS, Engram on, joins) | bd (which 0)
//     attn_a, [idx (L in INDEX_LAYERS, indexer on)], attn_b
//     bd (which 1)
//     router (arg = AGCE ce), ag (arg = ce)            every rank's router first, then every rank's ag
//     agce_join
//     moe (owner kind)                                 rsf = the bucket's moe variant == PROD_RSF (set_step's decide)
//     rs (arg = rsf)                                   handle by agrs_fl_rsf (the "fl" variant when the state has it)
//   [final]                                            (T: the last segment)
//   join                                               (_join_mix's driver half; no Python event: nothing waits)
// A CP step replaces, on layers L <= 20 only, the
// layer's attn_a .. attn_b part with:
//   attn_a
//   [idx_c (h = idx), cp_x1a_send (-1), cp_x1a_recv (-1), idx_s (h = idx)]      L an index layer (the KV sources)
//   attn_bi (h = attn_b / attn_b_pf as above), cp_x2_send (-1), cp_x2_recv (-1), attn_bo (the same h)
// (the cross-rank order is the driver's: cp_send publishes, cp_recv spins on the flag; no barrier entries). Any CP kind or
// handle missing: refused.
// Entries are added rank 0 .. W-1 for each kind (mixed_cxx._Span.add order). A segment that needs ANY Python piece (a
// handle missing, a flag that makes _issue_span return False) is refused: the executor then runs the step embedded.
// Golden: record_mixed_live.py's "spans" (the real _issue_span on the GPU: add / py / flush / segment) through replay_span.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../../include/common.h"

namespace fpr {
namespace core2b {

struct SpanKinds {                      // the driver's kind ids: mixed_cxx K_* (builtin) + drv.x.owner_kind(name) (owners)
  int32_t bd = 1, router = 2, ag = 3, agce_join = 4, rs = 5, final_ = 6, join = 7, waitev = 8;
  int32_t embed = -1, engram_bd = -1, attn_a = -1, idx = -1, attn_b = -1, moe = -1;   // -1 = the owner has no kind
  // CP steps: -1 = not registered -> a CP segment refuses
  int32_t attn_bi = -1, attn_bo = -1, idx_c = -1, idx_s = -1, cp_x1a_send = -1, cp_x1a_recv = -1, cp_x2_send = -1,
          cp_x2_recv = -1;
};

struct SpanLayerHandles {               // one rank, one layer: handles built at ensure (-1 = absent -> refusal)
  int32_t layer = -1;
  int64_t bd0 = -1, engram_bd = -1, attn_a = -1, idx = -1, attn_b = -1, bd1 = -1;
  int64_t router = -1, router_ce = -1, ag = -1, moe = -1, rs = -1, rs_fl = -1;
  int64_t moe_rsf = -1, attn_b_pf = -1;   // the PROD_RSF moe handle, the ATTN_PF attn_b (A views)
};
struct SpanRankHandles {                // one rank of one bucket view
  int64_t embed = -1, final_ = -1;
  std::vector<SpanLayerHandles> layers; // in the segment's layer order
};

struct SpanSegment {                    // the decision inputs of one segment (host data only)
  std::vector<int32_t> layers;
  bool final_ = false, embed = false;
  // the engine's switches _issue_span checks (record_mixed_live's snapshot + the bucket's state flags)
  bool moe_prod = true, capturing = false, l2pf = false, nancheck = false, dbg_sync = false, prof_pieces = false;
  bool cp_on = false, tp_on = false, mega_on = false, router_pack3 = true, mm_rt = false, mext_ok = true;   // cp_on: CP layout
  bool agrs_on = false, agrs_ag32 = false, agrs_agce = false;
  int64_t R = 0, agrs_min_r = 0;
  std::vector<int32_t> engram_layers, index_layers;
  bool engram_on = false, idx_on = false;
  // per rank: AGCE ce (the previous segment's state list is this one: mixed_cxx's stl[r] is s), the moe variant's RSF,
  // the state's agrs_fl_rsf (the rs "fl" handle)
  std::vector<uint8_t> ce, rsf, fl_rsf;
  std::vector<uint8_t> pf;              // per rank: the A bucket's rank has prefill rows under ENG_ATTN_PF (ws["pf_ab"]): attn_b_pf
};

struct SpanEntry { int32_t rank, kind, layer; int64_t handle, arg; bool joins; };

// The segment's program per rank (prog[r] = the entries rank r's driver thread runs, in order). "" = OK, else the refusal
// (the piece / flag that needs Python).
std::string build_span(const SpanSegment& seg, const std::vector<SpanRankHandles>& h, const SpanKinds& k,
                       std::vector<std::vector<SpanEntry>>* prog);

}  // namespace core2b
}  // namespace fpr
