// native_mixed.h -- the NativeWorker's native mixed step. nd's hooks natively; the A -> tail gather -> T middle in Python via the process's PyExec. This header: the host-side pieces of the step that core2b owns.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "tables.h"

namespace fpr {
namespace core2b {

// engine.tail_batch(sb), 1:1: the StepBatch of sb's tail rows (layers 21..39). Rows = sb.tail_rows in order, each sequence
// cut to its tail rows (row0 / n_rows / start remapped, tail_row0 = row0), sample rows remapped. tail_bound = engine's
// TAIL_BOUND (ENG_TAIL_BOUND, default on): a trimmed prefill / score-free sequence's replay_start rises to
// start + n_rows - sliding_window (production's decoder-side bounded replay). BAD_ARG where Python raises (a sample row that
// is not a tail row: KeyError).
Status tail_batch(const StepBatch& sb, bool tail_bound, int64_t sliding_window, StepBatch* out, std::string* err = nullptr);

// The T tables' host tokens prepare_head reads: [tables.tokens[i] for i in sb.tail_rows] (engine.step_mixed: tb_tail.build's
// token argument), per rank.
std::vector<int32_t> tail_tokens(const StepBatch& sb, const std::vector<int32_t>& tokens);

}  // namespace core2b
}  // namespace fpr
