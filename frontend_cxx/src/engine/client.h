// EngineClient: the AsyncMPClient equivalent. ZMQ ROUTER (input) + PULL (output) adopted from the
// listener fds the Python supervisor passes through the seam (tools/SEAM.md), one dedicated IO thread.
//
// Threading contract (answers tools/REQUESTS.md "A -> E"):
//   * add(), abort(), call_utility(), scheduler_stats(), dead() are thread-safe; call them from any thread.
//   * Callbacks (per-request OutputCallback, the StepObserver, the dead callback) run on the IO thread, one at a
//     time, in engine order. They must not block; calling add()/abort() from inside a callback is allowed.
//   * For one request, callbacks are strictly sequential; after a finished output (or abort()) no more arrive.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "engine/protocol.h"

namespace fe::engine {

struct EngineDeadError : std::runtime_error {
  EngineDeadError() : std::runtime_error("EngineCore encountered an issue. See stack trace (above) for the root cause.") {}
};
struct EngineTimeoutError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct UtilityError : std::runtime_error {  // failure_message from the engine
  using std::runtime_error::runtime_error;
};

struct ClientOptions {
  // Inherited listener fds (seam). When < 0, the client binds `*_endpoint` itself (tests, mock engine).
  int input_listener_fd = -1, output_listener_fd = -1;
  std::string input_endpoint, output_endpoint;
  int engine_start_index = 0, engine_count = 1;
  double ready_timeout_s = 600;  // args-json engine_ready_timeout_secs (VLLM_ENGINE_READY_TIMEOUT_S)
  int64_t client_index = 0;
  int io_threads = 2;            // zmq.Context(io_threads=2) like MPClient
};

// One delivery to a request: an engine output, or (output == nullptr) a fatal client error / engine death.
struct Delivery {
  const EngineOutput* output = nullptr;
  const EngineCoreOutputs* step = nullptr;  // the batch it arrived in (timestamp, engine_index, ...)
  const char* error = nullptr;              // set when output == nullptr ("EngineDeadError", decode error text)
  bool engine_dead = false;
};
using OutputCallback = std::function<void(const Delivery&)>;
using StepObserver = std::function<void(const EngineCoreOutputs&)>;  // every EngineCoreOutputs, before dispatch

class EngineClient {
 public:
  explicit EngineClient(ClientOptions opt);
  ~EngineClient();
  EngineClient(const EngineClient&) = delete;
  EngineClient& operator=(const EngineClient&) = delete;

  // Blocks until every expected engine identity sent its EngineCoreReadyResponse on the input ROUTER
  // (MPClient.__init__), then starts the IO thread. Throws EngineTimeoutError after ready_timeout_s.
  const ReadyResponse& wait_ready();
  const ReadyResponse& ready() const { return ready_; }

  // UTILITY round trip (AsyncMPClient._call_utility_async). Returns the result value (UtilityResult.result).
  mp::Value call_utility(const std::string& method, const std::vector<mp::Value>& args, double timeout_s = 600);
  std::vector<std::string> get_supported_tasks(double timeout_s = 600);  // ("generate",) for our engine

  // Registers `cb` for req.request_id (the INTERNAL id) and sends ADD. Throws EngineDeadError when dead.
  void add(const EngineRequest& req, OutputCallback cb);
  // Sends ABORT for internal ids and drops their callbacks (no further deliveries). Idempotent.
  void abort(const std::vector<std::string>& internal_ids);
  // Forget callbacks without sending ABORT (used after a finished output).
  void forget(const std::string& internal_id);

  void set_step_observer(StepObserver obs);
  void set_dead_callback(std::function<void()> cb);
  // When set, every non-utility EngineCoreOutputs goes to this handler (on the IO thread) INSTEAD of the
  // per-request callbacks (AsyncEngine uses it to run the OutputProcessor batch-wise, like AsyncLLM).
  using BatchHandler = std::function<void(EngineCoreOutputs&)>;
  void set_batch_handler(BatchHandler h);

  bool dead() const { return dead_.load(); }
  SchedulerStatsView scheduler_stats() const;  // latest non-null SchedulerStats (for /metrics gauges)
  mp::Value scheduler_stats_raw() const;
  size_t num_inflight() const;

 private:
  void io_loop();
  void send_now(const std::vector<std::string>& frames);
  void enqueue(std::vector<std::string> frames);
  void wake();
  void handle_outputs(std::vector<std::string>&& frames);
  void mark_dead(const char* why);
  void* bind_socket(int type, int fd, const std::string& endpoint);

  ClientOptions opt_;
  void* ctx_ = nullptr;
  void* in_ = nullptr;   // ROUTER
  void* out_ = nullptr;  // PULL
  int wake_fd_ = -1;
  std::string engine_id_;
  ReadyResponse ready_;
  bool ready_done_ = false;

  std::thread io_;
  std::atomic<bool> running_{false};
  std::atomic<bool> dead_{false};

  std::mutex send_mu_;
  std::deque<std::vector<std::string>> send_q_;

  mutable std::mutex mu_;  // callbacks, utility, stats
  std::unordered_map<std::string, std::shared_ptr<OutputCallback>> cbs_;
  struct Pending {
    bool done = false;
    std::optional<std::string> failure;
    mp::Value value;
  };
  std::map<uint64_t, std::shared_ptr<Pending>> utility_;
  std::condition_variable utility_cv_;
  mp::Value sched_stats_;
  StepObserver observer_;
  BatchHandler batch_;
  std::function<void()> dead_cb_;
};

}  // namespace fe::engine
