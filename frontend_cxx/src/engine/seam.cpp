#include "engine/seam.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>

namespace fe::engine {

static int parse_int(std::string_view flag, const std::string& v) {
  try {
    size_t pos = 0;
    int x = std::stoi(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("trailing");
    return x;
  } catch (const std::exception&) {
    throw SeamError("invalid integer for " + std::string(flag) + ": '" + v + "'");
  }
}

FrontendArgs parse_frontend_args(int argc, char** argv) {
  if (argc < 2 || std::string_view(argv[1]) != "frontend")
    throw SeamError("usage: frontend_cxx frontend --listen-fd N --input-listener-fd N --output-listener-fd N "
                    "[--engine-start-index I] [--engine-count C] [--data-parallel-size D] [--grpc-listen-fd N] "
                    "[--coordinator-address A] --args-json JSON");
  FrontendArgs a;
  bool have_listen = false, have_in = false, have_out = false, have_json = false, have_dp = false;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    std::string val;
    size_t eq = arg.find('=');
    if (arg.rfind("--", 0) == 0 && eq != std::string::npos) {
      val = arg.substr(eq + 1);
      arg = arg.substr(0, eq);
    } else {
      if (i + 1 >= argc) throw SeamError("missing value for " + arg);
      val = argv[++i];
    }
    if (arg == "--listen-fd") { a.listen_fd = parse_int(arg, val); have_listen = true; }
    else if (arg == "--input-listener-fd") { a.input_listener_fd = parse_int(arg, val); have_in = true; }
    else if (arg == "--output-listener-fd") { a.output_listener_fd = parse_int(arg, val); have_out = true; }
    else if (arg == "--grpc-listen-fd") a.grpc_listen_fd = parse_int(arg, val);
    else if (arg == "--engine-start-index") a.engine_start_index = parse_int(arg, val);
    else if (arg == "--engine-count") a.engine_count = parse_int(arg, val);
    else if (arg == "--data-parallel-size") { a.data_parallel_size = parse_int(arg, val); have_dp = true; }
    else if (arg == "--coordinator-address") a.coordinator_address = val;
    else if (arg == "--args-json") { a.args_json = val; have_json = true; }
    else throw SeamError("unknown argument " + arg);
  }
  if (!have_listen) throw SeamError("--listen-fd is required");
  if (!have_in) throw SeamError("--input-listener-fd is required");
  if (!have_out) throw SeamError("--output-listener-fd is required");
  if (!have_json) throw SeamError("--args-json is required");
  if (!have_dp) a.data_parallel_size = a.engine_count;
  if (a.engine_count < 1) throw SeamError("--engine-count must be >= 1");
  return a;
}

std::string zmq_endpoint_for_fd(int fd, bool* ipv6) {
  sockaddr_storage ss{};
  socklen_t len = sizeof ss;
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&ss), &len) != 0)
    throw SeamError("getsockname(" + std::to_string(fd) + ") failed: " + std::strerror(errno));
  if (ipv6) *ipv6 = false;
  if (ss.ss_family == AF_UNIX) {
    auto* un = reinterpret_cast<sockaddr_un*>(&ss);
    size_t n = len > offsetof(sockaddr_un, sun_path) ? len - offsetof(sockaddr_un, sun_path) : 0;
    std::string path(un->sun_path, strnlen(un->sun_path, n));
    if (path.empty()) throw SeamError("inherited AF_UNIX listener has no path (abstract/unnamed socket)");
    return "ipc://" + path;
  }
  char host[INET6_ADDRSTRLEN] = {0};
  if (ss.ss_family == AF_INET) {
    auto* in = reinterpret_cast<sockaddr_in*>(&ss);
    inet_ntop(AF_INET, &in->sin_addr, host, sizeof host);
    return "tcp://" + std::string(host) + ":" + std::to_string(ntohs(in->sin_port));
  }
  if (ss.ss_family == AF_INET6) {
    auto* in6 = reinterpret_cast<sockaddr_in6*>(&ss);
    inet_ntop(AF_INET6, &in6->sin6_addr, host, sizeof host);
    if (ipv6) *ipv6 = true;
    return "tcp://[" + std::string(host) + "]:" + std::to_string(ntohs(in6->sin6_port));
  }
  throw SeamError("unsupported inherited listener domain " + std::to_string(ss.ss_family));
}

int64_t zmq_socket_buffer_size() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  int64_t val = 0, total = 0, avail = 0;
  std::string unit;
  while (f >> key >> val >> unit) {
    if (key == "MemTotal:") total = val * 1024;
    else if (key == "MemAvailable:") avail = val * 1024;
  }
  const int64_t GiB = int64_t(1) << 30;
  if (total > 32 * GiB && avail > 16 * GiB) return 512 * (int64_t(1) << 20);
  return -1;
}

std::string engine_identity(int rank) {
  std::string id(2, '\0');
  id[0] = char(rank & 0xff);
  id[1] = char((rank >> 8) & 0xff);
  return id;
}

}  // namespace fe::engine
