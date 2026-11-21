// mixed_build.h -- P0 builders of the mixed step's host side in C++.
// = engine.py Engine.step_mixed (_mixed_body + _mixed_post) as the step's host program, decision for decision:
//   - the plan: bucket rows R_A / R_T (mixed_round, ENG_MIXED_GRAPHS_T + MIXED_T_ROUND, max_tail_rows), the A / T layer
//     segments, ENG_L20_TAIL, ENG_MIXED_PRESTEP's early T lookup, CP, the issue variant of each bucket;
//   - per bucket _new_scheds' FlashMLA plan sharing per layer;
//   - bucket A: _attn_pf_plans' host inputs; bucket T: tail_batch, tb_tail.build, the tail
//     gather (keys + ix), the index-layer padding clears, layer 20's window aliasing;
//   - the ordered op list (record_mixed_cpu.py's call list: what a native executor issues, in engine order) and the
//     stats counters the step moves.
// No device work, no engine reads: explicit inputs (MixedSnap, the batches, bucket A's RankTables, prev_of, tb_tail).
// Golden: core/tests/replay_mixed.cpp on core/tests/record_mixed_cpu.py traces (engine.py's REAL step_mixed on a stub
// engine inside record_d13_cpu.py's serving harness).
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../../include/common.h"
#include "../../include/tables.h"
#include "inputs_build.h"

namespace fpr {
namespace p0 {

// ---------------------------------------------------------------- inputs
struct MixedSnap {                      // the engine state step_mixed reads (record_mixed_cpu.py Recorder._snap)
  int W = 4;
  int64_t mixed_round = 64, max_mixed_rows = 4416, max_tail_rows = 0, mixed_t_round = 128;
  std::vector<int32_t> layers;          // Engine.layers (contiguous)
  bool mixed_graphs = false, cp = true, dbg_sync = false, l20_tail = false, mixed_prestep = true, mixed_graphs_t = false;
  bool pf_act8 = false, attn_pf = true, mm_on = false, poison = false, cxx_pieces = true, mixed_cxx_head = true;
  int64_t img_route_lo = 129264, img_route_hi = 129268;   // engine.IMAGE_ROUTE_IDS (rows routed with bias_vl under MM_ON)
  bool tail_bound = true, devtok_v2 = true, per_layer_plans = true, clear_tail_padding = true, upload_v2 = true;
  bool pf_ws = true, hsd = true, idx = true;
  std::string mixed_plans = "source";
  std::vector<int32_t> index_layers;    // INDEX_LAYERS
  std::vector<int32_t> ratio;           // RATIO[L] for each entry of layers
  std::vector<int32_t> index_source;    // index_source_of(L) for each entry of layers (-1 = None)
  int64_t sliding_window = 128, window_page_tokens = 32;
  std::vector<int64_t> scratch;         // per rank: pools[r].scratch_window_page * WINDOW_PAGE_TOKENS
  std::vector<int64_t> n_paged;         // per rank: pools[r].n_paged (attn_pf Plan)
  int64_t k1 = 4;                       // hsd.next_ids[r].shape[1]
};
struct PrevSeq { int32_t rank = -1; int64_t seq_index = -1; };   // engine.prev_of(rid) = (rank, seq index) or None
using PrevFn = std::function<bool(const Rid& rid, PrevSeq* out)>;  // false = None

// ---------------------------------------------------------------- outputs
struct PfPlanIn {                       // attn_prefill.Plan's host inputs of one rank of bucket A (ENG_ATTN_PF)
  std::vector<std::array<int64_t, 4>> seqs;   // (row0, n_rows, start, replay_start) of the prefill sequences
  int64_t bt_rows = 0, bt_cols = 0;
  std::vector<int32_t> bt;              // [rows, cols] int32, block_table[:ceil((start + n_rows) / 64)] per sequence, 0 padded
  int64_t n_paged = 0;
  bool ws = false;                      // the engine's pf_ws[r] (ENG_PF_WS_PREALLOC_MB)
};
struct TailStage {                      // bucket T of one rank: the gather from bucket A after layer 20
  std::vector<int64_t> ix;              // = batch.tail_rows (A rows, T row i <- A row ix[i])
  std::vector<std::string> keys;        // per-row state keys copied (A -> T rows [0, n)): "X" / "ws.q" / "si20" / "ti.2" ...
  std::vector<int32_t> clear_layers;    // index layers whose T padding rows [n, R) are cleared (ti -1, tl 0)
  bool l20 = false;                     // si20 / sl20 = T's si / sl, rows [0, n) from A's; caches[20] reads them
  std::string c20 = "";                 // caches[20]'s window views: "si20" | "si" | "" (no layer 20 in T)
  bool carry = false;                   // hooks.idx.carry_candidates(sA, sT, tail_rows, ix)
};
struct BucketRank {                     // one rank of one bucket (A or T)
  int64_t R = 0;
  InputsOut up;                         // build_inputs (inputs_build.h): the blob, n / n_sample, device-token pairs / copies
  bool chain_input = false;
  int cp_on = -1;                       // A: CP plan present (0 / 1); T: -1 (not set)
  int l20 = -1;                         // T: sT["l20"] (0 / 1); A: -1 (not set)
  bool has_pf_ab = false; int64_t pf_a = 0, pf_b = 0;   // ws["pf_ab"]: the bucket's prefill rows [a, b)
  bool pf = false; PfPlanIn pf_in;      // A: attn_pf Plan built for this rank
  std::vector<int64_t> sched, sched_dec;   // per layer: canonical FlashMLA plan ids (first appearance); -1 = None
  // T only
  StepBatch tb;                         // tail_batch(sb)
  RankTables tab;                       // tb_tail.build(tb, [tables_A.tokens[i] for i in sb.tail_rows])
  TailStage stage;
};
enum class MixedOpKind {
  BUCKET, POISON, CP_PLAN, UPLOAD, UPLOAD_GENERAL, PREPARE_ENGRAM, SET_VIDX, IMG_ROWS_A, PF_PLANS, PF_PLAN, NEW_SCHEDS,
  TAIL_BATCH, SET_STEP, ISSUE, ISSUE_MIXED, L20_HEAD, L20_TAIL, SYNC, CP_DONE, HEAD_RESULT, TB_TAIL, IMG_ROWS_T, JOIN_MIX,
  T_STAGE, CARRY_CANDIDATES, PREPARE_HEAD, HEAD_EAGER, AFTER_SYNC, MEM_AFTER_STEP, LOG
};
struct MixedOp {                        // one step of the host program (record_mixed_cpu.py's call list + T_STAGE)
  MixedOpKind kind;
  char role = 0;                        // 'A' / 'T' / 0
  int rank = -1;
  int64_t R = 0;                        // BUCKET
  int32_t lay0 = -1, lay1 = -1, nlay = 0;   // ISSUE / ISSUE_MIXED: the layer range (nlay 0 = None)
  bool final_ = false, embed = false, per_layer = false;
};
const char* mixed_op_name(MixedOpKind k);  // record_mixed_cpu.py's names ("bucket", "upload_general", ...); T_STAGE "t_stage"

enum class MixedStatus { OK = 0, BAD_ARG = 1 };
struct MixedPlan {
  MixedStatus status = MixedStatus::OK;
  std::string err;                      // BAD_ARG: where Python raises (the assertion text)
  int64_t RA = 0, RT = 0;
  std::vector<int32_t> nA, nT;
  std::vector<int32_t> segA, segT;
  bool cp = false, l20 = false, tgr = false, prestep_T = false;
  VidxOut vA, vT;                       // build_vidx (inputs_build.h)
  std::vector<BucketRank> A, T;         // per rank (T empty without a tail segment)
  std::vector<MixedOp> ops;
  std::map<std::string, int64_t> stats; // counters moved: steps, mixed_steps, attn_pf_steps, cp_steps, l20_tail_steps, ...
  // native refusal (nd's pre-step): "" = a native step can run this plan; else why it stays embedded
  std::string native_refusal;
  bool plan_only = false;               // built with tb_tail == nullptr: ops / plan / vidx / gather spec, no T tables, no blobs
};

// engine.tail_batch(sb) 1:1 (TAIL_BOUND = tail_bound). BAD_ARG where Python raises (a sample row that is not a tail row).
Status tail_batch(const StepBatch& sb, bool tail_bound, int64_t sliding_window, StepBatch* out, std::string* err = nullptr);
// Engine._new_scheds on one rank's state: per layer the plan identity (raw creation ids; canonical() renumbers)
void build_scheds(const MixedSnap& s, bool per_layer, bool l20, bool pf, std::vector<int64_t>* sched,
                  std::vector<int64_t>* sched_dec);

// The whole step. tb_tail: the engine's tail TableBuilder (persistent across steps; called once per rank, rank order);
// nullptr = PLAN ONLY (the same op list and decisions without T tables or upload blobs, tb_tail's state untouched: the
// executor's pre-check, so a refusal never advances the shared D12 state).
// mixed_steps_before: stats["mixed_steps"] before the step (the pf_ws log every 200 steps).
Status build_mixed(const MixedSnap& s, const std::vector<StepBatch>& b, const std::vector<RankTables>& tables,
                   const PrevFn& prev, ITables* tb_tail, int64_t mixed_steps_before, MixedPlan* out);

// crc32 (zlib's) for golden comparisons
uint32_t crc32(const void* data, size_t n, uint32_t c = 0);

}  // namespace p0
}  // namespace fpr
