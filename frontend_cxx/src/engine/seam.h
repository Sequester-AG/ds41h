// The external-frontend seam of vLLM 0.32.0 (VLLM_USE_RUST_FRONTEND=1, VLLM_RUST_FRONTEND_PATH=<exe>).
// Full contract: tools/SEAM.md. This header gives main.cpp the CLI parser and the fd helpers.
#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace fe::engine {

struct SeamError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// `<exe> frontend --listen-fd N --input-listener-fd N --output-listener-fd N --engine-start-index I
//   --engine-count C --data-parallel-size D [--grpc-listen-fd N] --args-json JSON`
// (v1/utils.py:310-363). Also accepts `--flag=value`. Throws SeamError on a malformed command line.
struct FrontendArgs {
  int listen_fd = -1;
  int input_listener_fd = -1;
  int output_listener_fd = -1;
  std::optional<int> grpc_listen_fd;
  int engine_start_index = 0;
  int engine_count = 1;
  int data_parallel_size = 1;  // defaults to engine_count when absent (vllm-rs --help)
  std::optional<std::string> coordinator_address;  // stats_update_address; null for DP=1
  std::string args_json;  // json.dumps(non-default CLI args, sort_keys=True) + model_tag,
                          // engine_ready_timeout_secs, http_timeout_keep_alive. MAY CONTAIN api_key: never log it.
};
FrontendArgs parse_frontend_args(int argc, char** argv);  // argv[1] must be "frontend"

// ZMQ endpoint for an inherited, bound+listening socket fd (getsockname):
//   AF_UNIX -> "ipc://<path>", AF_INET -> "tcp://a.b.c.d:port", AF_INET6 -> "tcp://[addr]:port" (*ipv6 = true).
// libzmq's ZMQ_USE_FD needs the matching endpoint string at zmq_bind() time.
std::string zmq_endpoint_for_fd(int fd, bool* ipv6 = nullptr);

// vLLM's _get_zmq_socket_buffer_size(): 512 MiB when MemTotal > 32 GiB and MemAvailable > 16 GiB, else -1.
int64_t zmq_socket_buffer_size();

// Engine identity on the ROUTER: data-parallel rank as 2 little-endian bytes (rank 0 -> "\0\0").
std::string engine_identity(int rank);

}  // namespace fe::engine
