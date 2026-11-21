// server.h -- the stage-2b engine process's V1 socket IO. Replaces EngineCoreProc.process_input_sockets / process_output_sockets after vLLM's own
// Python handshake (init only):
//   start():  one DEALER per input address (identity = engine index, 2 bytes LE), the ready payload
//             (msgpack EngineCoreReadyResponse, built by Python's _make_ready_response) sent on each, as vLLM's input
//             thread does before the frontend's ROUTER can address the engine; one PUSH per output address (client i).
//   T_in:     [type, payload, aux...] -> ICore::input (buffered in order until attach()).
//   T_out:    ICore::pop_outputs -> the client's PUSH socket (frames as v1wire encoded them).
//   dead():   ENGINE_CORE_DEAD on every output socket (EngineCoreProc._send_engine_dead).
// DP = 1 only. Each socket is used by exactly one thread.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core.h"
#include "zio.h"

namespace fpr {
namespace core2b {

struct ServerConfig {
  std::vector<std::string> inputs;    // EngineZmqAddresses.inputs (the frontend's ROUTERs)
  std::vector<std::string> outputs;   // EngineZmqAddresses.outputs, index = client_index
  std::string identity;               // engine_index.to_bytes(2, "little")
  std::string ready_payload;          // msgspec.msgpack.encode(EngineCoreReadyResponse)
  int poll_ms = 50;                   // T_in poll / T_out pop timeout (stop latency)
};

struct ServerStats {
  std::atomic<int64_t> in_msgs{0}, in_bad{0}, in_buffered{0}, out_msgs{0}, out_frames{0}, out_bytes{0}, out_dropped{0};
};

class Server {
 public:
  Server() = default;
  ~Server() { stop(); }
  bool start(const ServerConfig& cfg, std::string* err);
  void attach(ICore* core);           // from now on T_in feeds the core, T_out drains it
  void dead();                        // ENGINE_CORE_DEAD to every client (then the outputs stop)
  void stop();
  const ServerStats& stats() const { return st_; }
  std::string last_error() const;

 private:
  void run_in();
  void run_out();
  void feed(const std::vector<std::string>& frames);
  void set_error(const std::string& e);

  ServerConfig cfg_;
  zio::Ctx ctx_;
  std::vector<zio::Sock> in_, out_;
  std::thread tin_, tout_;
  std::atomic<bool> run_{false}, dead_{false};
  std::atomic<ICore*> core_{nullptr};
  std::mutex mu_;                     // the attach point: buffered inputs before attach(), T_in's core_ check
  std::condition_variable cv_;
  std::deque<std::vector<std::string>> pending_;
  mutable std::mutex emu_;            // last error (set from T_in / T_out / feed under mu_)
  std::string err_;
  ServerStats st_;
};

}  // namespace core2b
}  // namespace fpr
