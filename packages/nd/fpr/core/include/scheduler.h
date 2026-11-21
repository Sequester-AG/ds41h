// scheduler.h -- serve/scheduler.py + serve/cost_model.py.
// Part of fpr/core/include/fpr_core.h (the umbrella). Mirrors the Python 1:1; build: -O2 -std=c++17 -ffp-contract=off.
#pragma once
#include "common.h"
#include "kv.h"
#include "geometry.h"
#include "cost_model.h"
#include "sched_config.h"

namespace fpr {

// ---------------------------------------------------------------- scheduling (serve/scheduler.py)
struct CPRole {                       // interfaces.CPRole
  int64_t group; Rid rid; int32_t owner;
  std::vector<std::tuple<int32_t, int64_t, int64_t>> subs;   // (rank, start, n) in position order, the owner last
  int32_t me;
};
struct SeqWork {                      // scheduler.SeqWork, the same fields
  Rid rid; int32_t rank; std::string kind;          // "decode" | "prefill" | "cp" (a CP helper shadow); score is a flag
  int64_t start;                      // -1 = device-resolved (D13 async decode)
  int64_t n, cached_upto = 0, replay_start = 0;
  bool last_chunk = false, score = false, forced = false;
  int32_t uid = -1; int64_t known = -1;
  bool has_cp = false; CPRole cp;
  std::vector<Image> images;
};
struct StepPlan {                     // scheduler.StepPlan
  int64_t step = 0;
  std::vector<SeqWork> work;          // insertion order IS a decision
  std::vector<CopyOp> copies;
  std::vector<Rid> preempted, migrated;
  std::vector<RankLoad> loads;        // one per rank (cost_model.h)
  double est_us = 0.0, idle_frac = 0.0;
  int64_t rows(int rank) const {
    int64_t s = 0;
    for (const SeqWork& w : work) if (w.rank == rank) s += w.n;
    return s;
  }
};
struct ReqState {                     // the per-request fields the adapter reads / writes today
  Phase phase; int32_t rank, uid; int64_t num_computed, num_output, inflight, replay_start, cached_tokens, skip_until_step;
  bool first_pending;
};

class IScheduler {                    // RankScheduler
 public:
  virtual ~IScheduler() = default;
  virtual Status add_request(const Request& r, std::string* reject_reason) = 0;
  virtual void abort(const Rid& rid) = 0;
  virtual StepPlan schedule(double now) = 0;
  // accepted[rid] = tokens appended (1 .. k + 1) for decode work of that plan; returns the requests that finished
  virtual std::vector<Rid> update(const StepPlan& plan, const std::vector<std::pair<Rid, int32_t>>& accepted, double now) = 0;
  virtual int64_t max_new_cap(int64_t prompt_len) const = 0;
  virtual int32_t seats_per_rank() const = 0;
  // what the 2a adapter does to rs today (adapter/phys_scheduler.py)
  virtual void extend_block_hashes(const Rid& rid, int64_t first_index, const std::vector<Hash>& hashes) = 0;  // before update()
  virtual void hold_until(const Rid& rid, int64_t step) = 0;        // structured-output hold (skip_until_step)
  virtual void sync_num_output(const Rid& rid, int64_t n) = 0;      // waiting requests' outputs
  virtual void set_paused_new(bool paused) = 0;                     // PAUSED_NEW: no new admissions
  virtual void reset_kv(std::unique_ptr<IKVManager> kv) = 0;        // reset_prefix_cache
  virtual bool get(const Rid& rid, ReqState* out) const = 0;
  virtual std::vector<Rid> running() const = 0;
  virtual int64_t step_id() const = 0;
  virtual std::string stats_json() const = 0;
};

// The C++ RankScheduler (fpr/core/src/scheduler/rank_scheduler.cpp): owns `kv`; all decisions as serve/scheduler.py.
std::unique_ptr<IScheduler> make_scheduler(const Geometry& g, const CostParams& cost, const SchedConfig& cfg,
                                           std::unique_ptr<IKVManager> kv);

}  // namespace fpr
