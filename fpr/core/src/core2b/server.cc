// server.cc -- see server.h.
#include "server.h"

#include <chrono>
#include <cstdio>

namespace fpr {
namespace core2b {

static const char kDead[] = "ENGINE_CORE_DEAD";   // EngineCoreProc.ENGINE_CORE_DEAD

bool Server::start(const ServerConfig& cfg, std::string* err) {
  cfg_ = cfg;
  if (!ctx_.get()) {
    if (err) *err = "zmq_ctx_new failed";
    return false;
  }
  for (const auto& a : cfg_.inputs) {   // make_zmq_socket(ctx, addr, zmq.DEALER, identity=identity, bind=False)
    zio::Sock s(ctx_.get(), ZMQ_DEALER);
    if (!s.ok() || !s.set_bytes(ZMQ_IDENTITY, cfg_.identity) || !s.connect(a)) {
      if (err) *err = "input socket " + a + ": " + zio::err();
      return false;
    }
    // the initial message to each input socket: required before the frontend's ROUTER can send to us
    if (!s.send(cfg_.ready_payload)) {
      if (err) *err = "ready payload on " + a + ": " + zio::err();
      return false;
    }
    in_.push_back(std::move(s));
  }
  for (const auto& a : cfg_.outputs) {   // make_zmq_socket(ctx, addr, zmq.PUSH, linger=4000)
    zio::Sock s(ctx_.get(), ZMQ_PUSH);
    if (!s.ok() || !s.set_int(ZMQ_LINGER, 4000) || !s.connect(a)) {
      if (err) *err = "output socket " + a + ": " + zio::err();
      return false;
    }
    out_.push_back(std::move(s));
  }
  run_ = true;
  tin_ = std::thread([this] { run_in(); });
  tout_ = std::thread([this] { run_out(); });
  return true;
}

void Server::attach(ICore* core) {
  std::lock_guard<std::mutex> g(mu_);   // T_in checks core_ under mu_: it blocks until the buffered inputs are in
  core_.store(core);
  for (auto& f : pending_) feed(f);     // in arrival order
  pending_.clear();
  cv_.notify_all();
}

void Server::feed(const std::vector<std::string>& frames) {   // core_ set; caller holds mu_ or is T_in after attach
  ICore* c = core_.load();
  std::string e;
  uint8_t type = frames[0].empty() ? 0xff : (uint8_t)frames[0][0];
  // [type, payload, aux...]: aux frames carry out-of-band tensors (images). The whole message goes to the core
  // (ICore v2.2 input_frames): 2b's core admits images through the Python admitter (design §5a); a core without one
  // decodes the main frame (mm_features opaque) and the lifecycle rejects them with vLLM's error output (the client gets
  // finish_reason "error" instead of waiting forever)
  if (c->input_frames(frames, &e) != Status::OK) {
    st_.in_bad++;
    set_error("input type " + std::to_string((int)type) + " (" + std::to_string(frames.size()) + " frames): " +
              (e.empty() ? std::string("aux frames not supported") : e));
    return;
  }
  st_.in_msgs++;
}

void Server::run_in() {
  std::vector<zmq_pollitem_t> items;
  for (auto& s : in_) items.push_back({s.get(), 0, ZMQ_POLLIN, 0});
  std::vector<std::string> frames;
  while (run_) {
    int n = zmq_poll(items.data(), (int)items.size(), cfg_.poll_ms);
    if (n < 0) {
      if (zmq_errno() == EINTR) continue;
      set_error("zmq_poll: " + zio::err());
      break;
    }
    for (size_t i = 0; i < items.size(); ++i) {
      if (!(items[i].revents & ZMQ_POLLIN)) continue;
      while (in_[i].recv_multipart(&frames, ZMQ_DONTWAIT)) {
        if (frames.empty()) continue;
        if (frames[0] == "READY") continue;
        std::unique_lock<std::mutex> l(mu_);
        if (!core_.load()) {
          pending_.push_back(frames);
          st_.in_buffered++;
          continue;
        }
        l.unlock();
        feed(frames);
      }
    }
  }
}

void Server::run_out() {
  {
    std::unique_lock<std::mutex> l(mu_);
    cv_.wait(l, [this] { return core_.load() != nullptr || !run_ || dead_; });
  }
  std::vector<WireOut> batch;
  while (run_ && !dead_) {
    ICore* c = core_.load();
    if (!c) break;
    if (!c->pop_outputs(&batch, cfg_.poll_ms / 1000.0)) continue;
    for (const auto& w : batch) {
      if (w.client_index < 0 || (size_t)w.client_index >= out_.size()) {
        st_.out_dropped++;
        continue;
      }
      if (!out_[w.client_index].send_multipart(w.frames)) {
        set_error("output send: " + zio::err());
        st_.out_dropped++;
        continue;
      }
      st_.out_msgs++;
      st_.out_frames += (int64_t)w.frames.size();
      for (const auto& f : w.frames) st_.out_bytes += (int64_t)f.size();
    }
  }
  if (dead_)
    for (auto& s : out_) s.send(std::string(kDead, sizeof(kDead) - 1));
}

void Server::dead() {
  dead_ = true;
  cv_.notify_all();
  if (tout_.joinable()) tout_.join();
}

void Server::stop() {
  run_ = false;
  cv_.notify_all();
  if (tin_.joinable()) tin_.join();
  if (tout_.joinable()) tout_.join();
  in_.clear();
  out_.clear();
}

void Server::set_error(const std::string& e) {
  std::lock_guard<std::mutex> g(emu_);
  err_ = e;
  std::fprintf(stderr, "core2b server: %s\n", e.c_str());
}

std::string Server::last_error() const {
  std::lock_guard<std::mutex> g(emu_);
  return err_;
}

}  // namespace core2b
}  // namespace fpr
