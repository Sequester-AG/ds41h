// core.h -- request lifecycle / stop checks / grammar / V1 output assembly / the loop.
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
//
// v2: the core speaks the V1 wire types (fpr/core/src/v1wire/v1wire.h), because vLLM's outputs
// carry more than CoreOutput did (events with timestamps, the full prefill-stats breakdown, logprob ranks, per-step
// scheduler stats, utility replies) and byte identity needs all of it. Inputs: the frontend's EngineCoreRequest (2b: the
// ZMQ payload; 2a: the adapter encodes vLLM's Request back with MsgpackEncoder). Outputs: (client index, frames of an
// EngineCoreOutputs message) exactly as vLLM's output thread sends them, plus the decoded struct for 2a.
//
// the step batches come from BatchBuilder
// (tables.h build_step_batch on the rank scheduler's KV manager). The lifecycle itself: src/core/lifecycle.h.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "scheduler.h"
#include "kv.h"
#include "tables.h"
#include "engine.h"
#include "../src/core/lifecycle.h"
#include "../src/grammar/grammar.h"
#include "../src/v1wire/v1wire.h"

namespace fpr {

// ---------------------------------------------------------------- what the core hands the device side per step
struct GrammarRows { Rid rid; std::vector<int32_t> rows; int64_t n_rows = 0, words = 0, num_acceptable = 0; };
struct StepInput {                    // adapter/bridge.py PhysStep (host part) + phys_executor.split_bitmask
  int64_t step = -1;
  const StepPlan* plan = nullptr;
  std::vector<StepBatch> batches;     // ITables::build (interfaces.build_step_batch per rank)
  std::vector<std::vector<int32_t>> row_tokens;                    // per rank, -1 = device-resolved
  std::vector<std::pair<Rid, std::vector<int32_t>>> prev_tokens;   // up to 3 tokens before each sequence's first row
  std::vector<core::ReqInit> new_reqs;
  std::vector<Rid> finished, preempted, order;                     // order = SchedulerOutput.num_scheduled_tokens
  std::vector<std::pair<Rid, int32_t>> num_spec;
  std::vector<GrammarRows> grammar;   // structured requests' bitmask rows (empty: no grammar this step)
  double est_us = 0.0;
  int64_t total_rows = 0;             // total_num_scheduled_tokens (0 = frees only)
};
struct StepHandle;                    // the worker's in-flight step
class IWorker {
 public:
  virtual ~IWorker() = default;
  virtual Status dispatch(const StepInput& in, StepHandle** h) = 0;            // async: returns once issued
  virtual Status wait(StepHandle* h, std::vector<StepResult>* out /* [W] */) = 0;
};

// Where the loop reads the clock (each = one time.monotonic() / time.time() read of vLLM's code). Unset = the real
// clocks. Golden tests feed the recorded values per purpose, in order.
struct CoreClocks {
  std::function<double()> add;        // Request.record_event(QUEUED) at add_request
  std::function<double()> sched;      // PhysScheduler._schedule's ts
  std::function<double()> update;     // _phys_update's `now` for rs.update
  std::function<double()> output;     // each EngineCoreOutputs.__post_init__ (step outputs and utility replies)
  std::function<double()> wall;       // time.time() (the rank scheduler's arrival base)
};
// Test / embedding seams of the loop (all optional)
struct CoreHooks {
  // after each lifecycle call of the loop: "schedule" / "bitmask" / "update" / "drafts" (post_step) / "drafts_in_output"
  std::function<void(const char* call, const core::SchedOut& so, const core::GrammarOut* g, const core::UpdateOut* u)> on_call;
  // replaces merge_results (StepResults -> ModelRunnerOutput); tests feed the recorded ModelRunnerOutput
  std::function<void(const core::SchedOut& so, const std::vector<StepResult>& res, core::ModelOut* mo)> merge;
};

struct CoreConfig {
  core::LifecycleConfig lifecycle;
  CoreClocks clocks;
  CoreHooks hooks;
  bool grammar_async = true;          // compile grammars on the pool (vLLM); false = at add (golden tests)
  bool structured_outputs = true;     // build the xgrammar backend (grammar.tokenizer_info_json required)
  grammar::BackendConfig grammar;
  int64_t engine_index = 0;           // EngineCoreOutputs.engine_index (the DP rank)
  int64_t zero_copy_threshold = v1::kZeroCopyThreshold;
  bool check_for_draft_tokens = true; // EngineCore: speculative decoding on
  // UTILITY methods whose answer is fixed for the engine's lifetime (2b: computed by Python at init, e.g.
  // get_supported_tasks -> ("generate",)); answered before the built-in ones
  std::map<std::string, mp::Val> static_utilities;
  // v2.2: a JSONL step log of the loop thread ("" = off; 2b's PHYS_CXX_STEP_LOG): one line per
  // step_batch_queue / step_sync call that scheduled or waited: monotonic t (s, CLOCK_MONOTONIC = Python's
  // time.monotonic()), the schedule / dispatch / wait / update ms, rows, decode sequences, batch-queue depth
  std::string step_log;
};

struct WireOut { int64_t client_index; std::vector<std::string> frames; v1::EngineCoreOutputs msg; };

class ICore {
 public:
  virtual ~ICore() = default;
  // frontend messages (the V1 input socket): ADD / ABORT / UTILITY payloads (msgpack, after the type byte)
  virtual Status add(const v1::EngineCoreRequest& r, std::string* err) = 0;
  virtual void abort(const std::vector<Rid>& rids) = 0;
  virtual Status utility(const v1::UtilityCall& call, std::string* err) = 0;   // the reply arrives on the output queue
  virtual Status input(uint8_t type, const std::string& payload, std::string* err) = 0;   // decode + one of the above
  virtual void start() = 0;           // the loop thread (EngineCoreProc.run_busy_loop: step / step_with_batch_queue)
  // one loop iteration on the caller's thread, without the loop thread: drain the input queue (no blocking), then
  // step_fn + post_step, as vLLM's in-process client (InprocClient.get_output). Returns model_executed.
  virtual bool step_once() = 0;
  virtual void stop() = 0;
  virtual bool pop_outputs(std::vector<WireOut>* batch, double timeout_s) = 0;   // what the output thread sends
  // non-empty once the loop thread stopped on a worker failure (EngineCoreProc: executor failed -> ENGINE_CORE_DEAD);
  // 2b's process checks it and sends ENGINE_CORE_DEAD
  virtual std::string fatal() const { return std::string(); }
  // requests in the lifecycle or steps in flight, as the loop thread last saw them (EngineCore.has_work, for the
  // drain-mode shutdown; v2.1 additive)
  virtual bool has_work() const { return false; }
  // v2.2: an ADD whose multimodal inputs the process admitted -- 2b's Python
  // admitter decoded the aux frames with vLLM's decoder and computed the content keys (adapter/mm.py) -> the lifecycle's
  // add_request(r, ..., &images, mm_reject); mm_reject != "" rejects with vLLM's error output. Default: add(r) (a core
  // without images rejects a request carrying mm_features as text-only).
  virtual Status add_mm(const v1::EngineCoreRequest& r, const std::vector<Image>& images, const std::string& mm_reject,
                        std::string* err) {
    (void)images, (void)mm_reject;
    return add(r, err);
  }
  // v2.2: one whole V1 input message [type, payload, aux...] (2b's T_in). Default: the payload alone; aux frames only on
  // an ADD (the lifecycle then rejects its mm_features unless an override admitted them)
  virtual Status input_frames(const std::vector<std::string>& frames, std::string* err) {
    uint8_t type = frames.empty() || frames[0].empty() ? 0xff : (uint8_t)frames[0][0];
    if (frames.size() < 2 || (frames.size() > 2 && type != v1::ADD)) {
      if (err) *err = "aux frames not supported";
      return Status::BAD_ARG;
    }
    return input(type, frames[1], err);
  }
};
// interfaces.build_step_batch(plan, rs.kv, geom, rank): 2a/2b = tables.h build_step_batch on the rank scheduler's KV manager
using BatchBuilder = std::function<Status(const StepPlan& plan, int rank, StepBatch* out)>;
std::unique_ptr<ICore> make_core(const CoreConfig& cfg, std::unique_ptr<IScheduler> rs, BatchBuilder batches,
                                 std::unique_ptr<IWorker> worker, core::KVFacade kv, core::BlockHasher hasher,
                                 std::string* err);

}  // namespace fpr
