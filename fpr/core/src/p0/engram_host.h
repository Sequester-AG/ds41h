// engram_host.h -- Engram's per-step HOST side in C++
// kernels/engram/engram.py
//   step_args_from_tables(tab, lookback, pad_to)  -> eg_args_from_tables   (RankTables of bucket A / D13's fallback)
//   step_args_from_compact(ct, lookback, pad_to)  -> eg_args_from_compact  (D13 CompactTables)
//   EngramPhys.upload's host part (gen, the guard's exact host checks, the pinned staging writes, H2D range) -> eg_pack
// The device part of upload (wait for the staging set's copy of two uploads ago, ONE H2D from the set into _pk at `lo`,
// record the set's event) and Engram's device prepare stay with the caller (nd: the handover's engram_stage /
// engram_pk_d / estream). Byte-identical staging vs Python: tests/eg_check.py (CPU golden, libp0eg.so via ctypes).
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "tables.h"

namespace fpr {
namespace p0 {

constexpr int EG_RING_W = 8;            // engram.py RING_W: rows per sequence written to the device ring per step

// EngramPhys._pk_off (int32 offsets into the packed per-step block) for its max_rows R / max_seqs S:
// [ids R | row0 S | nrows S | start S | slot S | look 3S | counts 4]
struct EgLayout {
  int64_t ids = 0, row0 = 0, nrows = 0, start = 0, slot = 0, look = 0, counts = 0, pk_n = 0;
  int32_t max_rows = 0, max_seqs = 0;
};
EgLayout eg_layout(int32_t max_rows, int32_t max_seqs);

// engram.StepArgs (int32 arrays as numpy holds them; start / look cast to int32 exactly as numpy's assignment does)
struct EgArgs {
  int64_t T = 0, n = -1;                // T = max(n, pad_to); n = real rows
  std::vector<int32_t> row0, nrows, start, slot;
  std::vector<int32_t> look;            // [S, 3]: raw ids at start-1..start-3 (-1 below position 0), -2 = the device ring
  std::vector<Rid> rids;                // log lines only
};

// lookback(rid, start): the host's token ids at positions start-1, start-2, start-3 (values for negative positions are
// ignored: the builder writes -1 there, as engram.py does). Called only for prefill / > EG_RING_W-row sequences.
using EgLookback = std::function<std::array<int64_t, 3>(const Rid& rid, int64_t start)>;

Status eg_args_from_tables(const RankTables& tab, int64_t pad_to, const EgLookback& lookback, EgArgs* out,
                           std::string* err = nullptr);
Status eg_args_from_compact(const CompactTables& ct, int64_t pad_to, const EgLookback& lookback, EgArgs* out,
                            std::string* err = nullptr);

// EngramPhys's host state that upload advances: _gen (every upload), _stage_i (alternates), GUARD + ring rows
struct EgHostState {
  int32_t gen = 0;
  int32_t stage_i = 0;
  bool guard = true;                    // engram.GUARD (ENGRAM_GUARD, default 1)
  int32_t nring = 0;                    // EngramPhys.ring.shape[0] (the guard's host slot check)
};

struct EgPack {
  int32_t stage = 0;                    // the staging set written (the caller copies it, then records its event)
  int64_t lo = 0;                       // first int32 the H2D copies: 0 with host ids, else the row0 offset
  int64_t n_copy = 0;                   // int32s from lo to the end of the block
  bool prefill = false;                 // the guard's pf (a row with a host lookback): its device record read is due
  std::vector<std::string> logs;        // "ENGRAM OOB ..." bodies (the caller appends " rank=<r>")
};

// EngramPhys.upload's host part on a (possibly modified: the guard clamps slots) EgArgs: gen += 1; guard host checks;
// staging[stage_i] writes (only the fields upload writes: stale values elsewhere stay, as in Python); stage_i ^= 1.
// host_ids: the step's input ids (T values) when the module has no bound id buffer (production binds: nullptr).
// The caller must have waited for the chosen set's previous H2D (EngramPhys: _stage_ev[i].synchronize()).
Status eg_pack(EgArgs* a, const EgLayout& L, EgHostState* hs, int32_t* const staging[2], const int32_t* host_ids, EgPack* out,
               std::string* err = nullptr);

}  // namespace p0
}  // namespace fpr
