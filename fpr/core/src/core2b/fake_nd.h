// fake_nd.h -- a CPU stand-in for NativeDecode: adapter/fake_worker.py's random-mode model in
// C++, behind nd::IEngine2b. With it the 2b process runs with NO Python per step (C++ loop + NativeWorker + this), and
// its streams must equal the Python fake worker's token for token (test_core2b arm "native"). It also measures the C++
// host cost per step on the CPU.
//   token at position p of a request with prompt key K: 1000 + crc32("seed/K/p") % max(1, vocab - 2000)
//   drafts for the group at next_pos: u = random.Random(crc32("seed/rid/draft/next_pos")).random() (Python's MT19937);
//     n_ok = #{p in (0.80, 0.60, 0.43)[:k] : u < p}; draft j = true token, or a wrong one for j >= n_ok
//   a grammar-masked target takes the first allowed token
// Not covered (as the fake's defaults): scripts (PHYS_FAKE_SCRIPT), the KV content model, images, FORCE, logprob rows.
//
// SIMULATED DEVICE (SimConfig.step_ms > 0; 2b step 3, the overlap proof): every step occupies a simulated GPU timeline
// shaped like a real D13 async step
//   graph T + positions D2H (t_ms) -> graph L ((1 - h_frac) * step_ms) -> graph H (h_frac * step_ms).
// - T_HEAD is a real thread: it waits until the step's positions are ready, spends prep_ms of CPU (the HSD host
//   prepare), then enqueues H at max(L end, prepare done).
// - dispatch(N + 1) waits until T_head has enqueued H(N), then starts N + 1's graph T at max(now, H(N) end).
// - wait(N) returns at H(N) end.
// Measured per decode class:
//   cycle   = consecutive H ends;
//   GPU idle before graph T and before graph H (what the host did not hide);
//   loop busy = the loop thread's time outside dispatch / wait per cycle (schedule, tables, lifecycle, wire, IO, and
//     the fake's own token math).
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nd_api.h"

namespace fpr {
namespace core2b {

uint32_t crc32(const void* data, size_t n, uint32_t crc = 0);   // zlib.crc32
double py_random(uint32_t seed);                                // random.Random(seed).random(), seed < 2^32

struct SimConfig {
  double step_ms = 0.0;               // 0 = the device is instant (no timeline, no T_head)
  double h_frac = 0.15;               // graph H's share of the step
  double t_ms = 0.3;                  // graph T + the positions D2H before graph L
  double prep_ms = 0.3;               // T_head's HSD host prepare (CPU spin)
};

class FakeNd final : public nd::IEngine2b {
 public:
  FakeNd(int64_t vocab, int32_t k, int64_t seed = 0, SimConfig sim = SimConfig());
  ~FakeNd() override;
  Status init(const FprEngineHandover*, nd::RunPython, std::string*) override { return Status::OK; }
  // the fake runs every step natively (prefill + decode): no embedded fallback, no Python per step
  std::string async_ok(const std::vector<StepBatch>&, const std::vector<CompactTables>&, const nd::StepHost&) override {
    return std::string();
  }
  Status embedded_begin(const nd::StepHost&, std::string*) override { return Status::OK; }
  Status embedded_end(std::string*) override { return Status::OK; }
  Status dispatch(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const nd::StepHost& host,
                  nd::Handle** h, std::string* err) override;
  Status wait(nd::Handle* h, std::vector<StepResult>* out, std::string* err) override;
  Status drain(std::string*) override { return Status::OK; }
  std::string stats_json() const override;

 private:
  struct Req {
    uint32_t key = 0;
    bool structured = false;
    bool has_pos = false;
    int64_t pos = 0;                  // the "device state": the real start of the next decode group
    int32_t last = -1;
    std::vector<int32_t> drafts;
  };
  int32_t true_tok(const Req& st, int64_t pos) const;
  int32_t target(const Req& st, int64_t pos, const nd::GrammarRows* g, int64_t i, std::string* err) const;
  void new_drafts(Req* st, const Rid& rid, int64_t next_pos, StepResult* res) const;
  int64_t vocab_;
  int32_t k_;
  int64_t seed_;
  std::map<Rid, Req> reqs_;
  int64_t steps_ = 0, groups_ = 0, dev_groups_ = 0;
  // host cost: the interval between consecutive dispatches, by the step's decode-group class (st_cn's classes). With
  // the fake device instant, it is the C++ loop's whole cycle (schedule, tables, lifecycle, wire, IO)
  std::chrono::steady_clock::time_point last_{};
  bool has_last_ = false;
  std::map<int, std::pair<double, int64_t>> cyc_;   // class upper bound -> (sum ms, n)

  // ---- the simulated device (sim_.step_ms > 0)
 public:
  using TP = std::chrono::steady_clock::time_point;
  struct SimStep {
    int cls = 0;                      // decode-class upper bound (0 = has prefill rows)
    TP dev_start, pos_ready, L_end, H_start, H_end;
    double idle_T = 0.0;              // GPU idle before this step's graph T (ms)
    bool head_done = false;
  };

 private:
  void head_loop();
  void account_entry();               // loop busy: the time since the last FakeNd return
  SimConfig sim_;
  std::thread thead_;
  std::mutex smu_;
  std::condition_variable scv_;
  std::deque<std::shared_ptr<SimStep>> hq_;
  bool stop_ = false;
  std::shared_ptr<SimStep> prev_;     // the last dispatched step
  TP last_return_{}, last_H_end_{};
  bool has_return_ = false, has_H_end_ = false;
  double busy_since_wait_ = 0.0;
  struct SimStat { int64_t n = 0; double cycle = 0, idle_T = 0, idle_H = 0, idle_max = 0, busy = 0, busy_max = 0; };
  std::map<int, SimStat> sst_;
  double head_busy_ = 0.0;
  int64_t head_steps_ = 0;
};

}  // namespace core2b
}  // namespace fpr
