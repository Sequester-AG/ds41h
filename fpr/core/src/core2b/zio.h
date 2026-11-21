// zio.h -- a minimal RAII layer over libzmq's C API for the stage-2b engine process.
// libzmq 4.3.5: the process links pyzmq's bundled libzmq (the one vLLM's Python already loaded), headers from
// frontend_cxx/third_party/build/zeromq-4.3.5/include. No exceptions: every call returns bool / int, errors via err().
#pragma once
#include <zmq.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fpr {
namespace zio {

inline std::string err() { return zmq_strerror(zmq_errno()); }

class Ctx {
 public:
  Ctx() : c_(zmq_ctx_new()) {}
  ~Ctx() {
    if (c_) zmq_ctx_term(c_);
  }
  Ctx(const Ctx&) = delete;
  Ctx& operator=(const Ctx&) = delete;
  void* get() const { return c_; }

 private:
  void* c_;
};

class Sock {
 public:
  Sock() = default;
  Sock(void* ctx, int type) : s_(zmq_socket(ctx, type)) {}
  ~Sock() { close(); }
  Sock(Sock&& o) noexcept : s_(o.s_) { o.s_ = nullptr; }
  Sock& operator=(Sock&& o) noexcept {
    if (this != &o) {
      close();
      s_ = o.s_;
      o.s_ = nullptr;
    }
    return *this;
  }
  Sock(const Sock&) = delete;
  Sock& operator=(const Sock&) = delete;
  void close() {
    if (s_) {
      int z = 0;
      zmq_setsockopt(s_, ZMQ_LINGER, &z, sizeof z);
      zmq_close(s_);
      s_ = nullptr;
    }
  }
  void* get() const { return s_; }
  bool ok() const { return s_ != nullptr; }
  bool set_int(int opt, int v) { return zmq_setsockopt(s_, opt, &v, sizeof v) == 0; }
  bool set_bytes(int opt, const std::string& v) { return zmq_setsockopt(s_, opt, v.data(), v.size()) == 0; }
  bool connect(const std::string& addr) { return zmq_connect(s_, addr.c_str()) == 0; }
  bool bind(const std::string& addr) { return zmq_bind(s_, addr.c_str()) == 0; }
  // all frames of one message; false on error (EAGAIN with flags = ZMQ_DONTWAIT included)
  bool send_multipart(const std::vector<std::string>& frames, int flags = 0) {
    for (size_t i = 0; i < frames.size(); ++i) {
      int f = flags | (i + 1 < frames.size() ? ZMQ_SNDMORE : 0);
      if (zmq_send(s_, frames[i].data(), frames[i].size(), f) < 0) return false;
    }
    return true;
  }
  bool send(const std::string& b, int flags = 0) { return zmq_send(s_, b.data(), b.size(), flags) >= 0; }
  bool recv_multipart(std::vector<std::string>* frames, int flags = 0) {
    frames->clear();
    for (;;) {
      zmq_msg_t m;
      zmq_msg_init(&m);
      if (zmq_msg_recv(&m, s_, flags) < 0) {
        zmq_msg_close(&m);
        return false;
      }
      frames->emplace_back(static_cast<const char*>(zmq_msg_data(&m)), zmq_msg_size(&m));
      bool more = zmq_msg_more(&m) != 0;
      zmq_msg_close(&m);
      if (!more) return true;
    }
  }

 private:
  void* s_ = nullptr;
};

}  // namespace zio
}  // namespace fpr
