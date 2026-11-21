// engine.h -- the device engine (stage 2a: today's Python Engine; the stage-1 mixed driver under it).
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
#pragma once
#include "common.h"
#include "tables.h"

namespace fpr {

// ---------------------------------------------------------------- engine (device side) + D13 host
struct StepResult {                   // interfaces.StepResult (per rank)
  int32_t rank; int64_t step;
  std::vector<std::pair<Rid, int32_t>> accepted;                       // rid -> tokens appended (1 .. k + 1)
  std::vector<std::pair<Rid, std::vector<int32_t>>> tokens;
  std::vector<std::pair<Rid, std::vector<std::vector<std::pair<int32_t, float>>>>> logprobs;   // per token: top-k;
                                      // row j = generated (accepted) token j of this step, in `tokens` order
  std::vector<std::pair<Rid, std::vector<int32_t>>> drafts;            // structured requests: next k drafts
};
struct StepHandle;                    // an issued async step (D13): extract() later
class IEngine {
 public:
  virtual ~IEngine() = default;
  virtual Status step_compact(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, StepHandle** h) = 0;
  virtual Status extract(StepHandle* h, std::vector<StepResult>* out /* [W] */) = 0;
  virtual Status step(const std::vector<StepBatch>& b, std::vector<StepResult>* out) = 0;    // synchronous (mixed)
  virtual Status copies(const std::vector<PageCopyBatch>& c) = 0;
};

}  // namespace fpr
