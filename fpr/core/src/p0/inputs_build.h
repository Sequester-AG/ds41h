// inputs_build.h -- P0 builders of a step's input uploads in C++.
// = fpr/engine/p0_inputs.py, byte for byte:
//  - build_inputs: engine.Engine._upload_v2(r, s, tab) with dev_pos False (every eager / mixed step; D13's async decode
//    passes dev_pos True and is not covered) -> the in_h set's bytes in engine._in_layout(R), plus Hooks._device_tokens_v2's
//    host half (ENG_DEVTOK=1): the (row, next_ids index) pairs and the migrated sequences' cross-device copies;
//  - build_vidx: engine.Engine._set_vidx's values (vidx / nv / nd / ndec / vcount).
// The device side (P1 / the driver) issues, on rank r's step stream: the blob H2D into s["in_d"]; per migrated sequence
// next_ids[r0][si0, :nr] -> tok[row0:row0 + nr] (seq_rows order); if pairs: their H2D, then
// tok.index_copy_(0, rows, next_ids[r].view(-1).index_select(0, src)).
// Check: core/tests/p0_inputs_capi.cpp (ctypes) inside p0_inputs_test.py, against today's engine code on CPU fakes.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../include/common.h"
#include "../../include/tables.h"

namespace fpr {
namespace p0 {

constexpr int kSlidingWindow = 128;               // engine.SLIDING_WINDOW
// engine._IN_FIELDS order: pos int64, tok int64, own int64, samp int64, si int32 [W], sl int32, valid bool
enum InFieldId { IN_POS = 0, IN_TOK, IN_OWN, IN_SAMP, IN_SI, IN_SL, IN_VALID, IN_NFIELDS };   // (not InField: mixed_build.h's struct)
inline constexpr const char* kInName[IN_NFIELDS] = {"pos", "tok", "own", "samp", "si", "sl", "valid"};
inline constexpr int64_t kInElem[IN_NFIELDS] = {8, 8, 8, 8, 4, 4, 1};             // element bytes (numpy itemsize)
inline constexpr int64_t kInCols[IN_NFIELDS] = {1, 1, 1, 1, kSlidingWindow, 1, 1};
// The single source of truth is Python's engine._in_layout(R). The contract check (p0_inputs_check.Ctx)
// compares every field's (name, offset, R * cols * elem bytes) and the blob size for every R in 1..4096 (the decode bucket
// lists, the C1 lane R, every mixed A / T size); mixed_fields.h holds no blob offsets.
struct InLayout {
  int64_t off[IN_NFIELDS] = {0};   // byte offsets (each field padded to 8 bytes)
  int64_t bytes = 0;               // the blob size
};
InLayout in_layout(int64_t R);     // = engine._in_layout(R)

struct PrevOf { int32_t rank = -1, seq = -1; };   // engine.prev_of(rid): rank -1 = None
struct InputsSnap {                // p0_inputs.snap_inputs
  int64_t R = 0;
  int64_t scratch = 0;             // pools[r].scratch_window_page * WINDOW_PAGE_TOKENS
  bool hsd = false;                // hooks.hsd is not None
  int32_t k1 = 0;                  // hsd.next_ids[r].shape[1]
  int32_t rank = 0;
  std::map<Rid, PrevOf> prev;      // decode sequences of tab (absent = None)
};
struct MigCopy { int32_t r0 = 0, si0 = 0, row0 = 0, n_rows = 0; };
struct InputsOut {
  std::vector<uint8_t> blob;       // in_layout(R).bytes
  int64_t n = 0, n_sample = 0;     // s["n"], s["n_sample"]
  std::vector<int64_t> rows, src;  // device-token pairs (index_copy rows, next_ids flat indices)
  std::vector<MigCopy> mig;        // migrated copies, in seq_rows order
};
// BAD_ARG where Python would raise (n > R, a table shorter than n, sample rows past R, an empty decode sequence's min()).
Status build_inputs(const InputsSnap& snap, const RankTables& tab, InputsOut* out, std::string* err = nullptr);
// The device result of the token ops on a blob (checks / the CPU model): next_ids[r] = rank r's [S, k1] row-major ids.
void apply_device_tokens(std::vector<uint8_t>* blob, int64_t R, int32_t rank, const InputsOut& out,
                         const std::vector<const int64_t*>& next_ids, int32_t k1);

struct VidxOut {
  std::vector<int64_t> vidx;       // rank-major rows: r * R + i
  std::vector<int32_t> nv;         // rows per rank
  bool has_nd = false;
  std::vector<int32_t> nd;         // PF_ACT8: decode rows per rank (all ranks' decode rows a prefix)
  int64_t ndec = -1, vcount = 0;
  bool act8_fallback = false;      // the step engine.stats act8_fallback_steps counts
};
// nds: dec_prefix per rank (nullptr = not given); act8_on = engine.PF_ACT8
VidxOut build_vidx(const std::vector<int64_t>& ns, const std::vector<int64_t>* nds, int64_t R, bool act8_on);

}  // namespace p0
}  // namespace fpr
