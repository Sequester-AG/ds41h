#include "engine/client.h"

#include <sys/eventfd.h>
#include <unistd.h>
#include <zmq.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <set>

#include "engine/seam.h"

namespace fe::engine {

namespace {
std::string zerr(const char* what) { return std::string(what) + ": " + zmq_strerror(zmq_errno()); }

// Receive one multipart message (all frames). Returns false on EAGAIN when nonblocking.
bool recv_multipart(void* sock, std::vector<std::string>& frames, int flags) {
  frames.clear();
  while (true) {
    zmq_msg_t m;
    zmq_msg_init(&m);
    int rc = zmq_msg_recv(&m, sock, frames.empty() ? flags : 0);
    if (rc < 0) {
      int e = zmq_errno();
      zmq_msg_close(&m);
      if (e == EAGAIN && frames.empty()) return false;
      if (e == EINTR) {
        if (frames.empty()) return false;
        continue;
      }
      throw std::runtime_error(zerr("zmq_msg_recv"));
    }
    frames.emplace_back(static_cast<const char*>(zmq_msg_data(&m)), zmq_msg_size(&m));
    int more = zmq_msg_more(&m);
    zmq_msg_close(&m);
    if (!more) return true;
  }
}
}  // namespace

void* EngineClient::bind_socket(int type, int fd, const std::string& endpoint) {
  void* s = zmq_socket(ctx_, type);
  if (!s) throw std::runtime_error(zerr("zmq_socket"));
  int zero = 0;
  // make_zmq_socket(): HWM 0 (unbounded) and the large-buffer policy.
  zmq_setsockopt(s, ZMQ_RCVHWM, &zero, sizeof zero);
  int buf = int(zmq_socket_buffer_size());
  if (buf > 0) zmq_setsockopt(s, ZMQ_RCVBUF, &buf, sizeof buf);
  if (type == ZMQ_ROUTER) {
    zmq_setsockopt(s, ZMQ_SNDHWM, &zero, sizeof zero);
    if (buf > 0) zmq_setsockopt(s, ZMQ_SNDBUF, &buf, sizeof buf);
  }
  int linger = 0;
  zmq_setsockopt(s, ZMQ_LINGER, &linger, sizeof linger);
  std::string ep = endpoint;
  if (fd >= 0) {
    bool v6 = false;
    ep = zmq_endpoint_for_fd(fd, &v6);
    if (v6) {
      int one = 1;
      zmq_setsockopt(s, ZMQ_IPV6, &one, sizeof one);
    }
    if (zmq_setsockopt(s, ZMQ_USE_FD, &fd, sizeof fd) != 0) throw std::runtime_error(zerr("ZMQ_USE_FD"));
  }
  if (zmq_bind(s, ep.c_str()) != 0) throw std::runtime_error(zerr(("zmq_bind " + ep).c_str()));
  return s;
}

EngineClient::EngineClient(ClientOptions opt) : opt_(std::move(opt)) {
  if (opt_.engine_count != 1)
    throw std::runtime_error("frontend_cxx supports exactly one EngineCore (DP=1); got engine_count=" +
                             std::to_string(opt_.engine_count));
  ctx_ = zmq_ctx_new();
  zmq_ctx_set(ctx_, ZMQ_IO_THREADS, opt_.io_threads);
  in_ = bind_socket(ZMQ_ROUTER, opt_.input_listener_fd, opt_.input_endpoint);
  out_ = bind_socket(ZMQ_PULL, opt_.output_listener_fd, opt_.output_endpoint);
  wake_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (wake_fd_ < 0) throw std::runtime_error("eventfd failed");
  engine_id_ = engine_identity(opt_.engine_start_index);
}

EngineClient::~EngineClient() {
  running_ = false;
  wake();
  if (io_.joinable()) io_.join();
  if (in_) zmq_close(in_);
  if (out_) zmq_close(out_);
  if (ctx_) zmq_ctx_term(ctx_);
  if (wake_fd_ >= 0) ::close(wake_fd_);
}

const ReadyResponse& EngineClient::wait_ready() {
  if (ready_done_) return ready_;
  std::set<std::string> want;
  for (int r = opt_.engine_start_index; r < opt_.engine_start_index + opt_.engine_count; ++r)
    want.insert(engine_identity(r));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(opt_.ready_timeout_s);
  std::vector<std::string> frames;
  while (!want.empty()) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (left.count() <= 0)
      throw EngineTimeoutError("Timed out waiting for engine core processes to start. This is often caused by slow "
                               "weight loading for large models. Waited " +
                               std::to_string(int64_t(opt_.ready_timeout_s)) +
                               "s (configured by VLLM_ENGINE_READY_TIMEOUT_S). To increase the timeout, set the "
                               "environment variable: VLLM_ENGINE_READY_TIMEOUT_S=<seconds>");
    zmq_pollitem_t it[2] = {{in_, 0, ZMQ_POLLIN, 0}, {out_, 0, ZMQ_POLLIN, 0}};
    int rc = zmq_poll(it, 2, long(std::min<int64_t>(left.count(), 1000)));
    if (rc < 0) {
      if (zmq_errno() == EINTR) continue;
      throw std::runtime_error(zerr("zmq_poll"));
    }
    if (it[1].revents & ZMQ_POLLIN) {  // ENGINE_CORE_DEAD can arrive before READY if startup fails
      if (recv_multipart(out_, frames, ZMQ_DONTWAIT) && is_engine_dead_message(frames)) {
        mark_dead("ENGINE_CORE_DEAD during startup");
        throw EngineDeadError();
      }
    }
    if (!(it[0].revents & ZMQ_POLLIN)) continue;
    if (!recv_multipart(in_, frames, ZMQ_DONTWAIT)) continue;
    if (frames.size() < 2) continue;  // [identity, payload]
    auto itw = want.find(frames[0]);
    if (itw == want.end()) continue;
    want.erase(itw);
    ready_ = decode_ready_response(frames[1]);
  }
  ready_done_ = true;
  running_ = true;
  io_ = std::thread([this] { io_loop(); });
  return ready_;
}

void EngineClient::wake() {
  if (wake_fd_ < 0) return;
  uint64_t one = 1;
  ssize_t r = ::write(wake_fd_, &one, sizeof one);
  (void)r;
}

void EngineClient::enqueue(std::vector<std::string> frames) {
  {
    std::lock_guard<std::mutex> lk(send_mu_);
    send_q_.push_back(std::move(frames));
  }
  wake();
}

void EngineClient::send_now(const std::vector<std::string>& frames) {
  // [engine identity, type, body, aux...]
  if (zmq_send(in_, engine_id_.data(), engine_id_.size(), ZMQ_SNDMORE) < 0) throw std::runtime_error(zerr("zmq_send"));
  for (size_t i = 0; i < frames.size(); ++i) {
    int fl = i + 1 < frames.size() ? ZMQ_SNDMORE : 0;
    while (zmq_send(in_, frames[i].data(), frames[i].size(), fl) < 0) {
      if (zmq_errno() == EINTR) continue;
      throw std::runtime_error(zerr("zmq_send"));
    }
  }
}

void EngineClient::add(const EngineRequest& req, OutputCallback cb) {
  if (dead_) throw EngineDeadError();
  auto frames = encode_add(req);
  {
    std::lock_guard<std::mutex> lk(mu_);
    cbs_[req.request_id] = std::make_shared<OutputCallback>(std::move(cb));
  }
  enqueue(std::move(frames));
}

void EngineClient::abort(const std::vector<std::string>& ids) {
  if (ids.empty()) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& id : ids) cbs_.erase(id);
  }
  if (dead_) return;  // abort_requests_async: skipped when the engine is dead
  enqueue(encode_abort(ids));
}

void EngineClient::forget(const std::string& id) {
  std::lock_guard<std::mutex> lk(mu_);
  cbs_.erase(id);
}

mp::Value EngineClient::call_utility(const std::string& method, const std::vector<mp::Value>& args, double timeout_s) {
  if (dead_) throw EngineDeadError();
  uint64_t call_id = uuid1_call_id();
  auto p = std::make_shared<Pending>();
  {
    std::lock_guard<std::mutex> lk(mu_);
    utility_[call_id] = p;
  }
  enqueue(encode_utility(opt_.client_index, call_id, method, args));
  std::unique_lock<std::mutex> lk(mu_);
  bool ok = utility_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return p->done || dead_.load(); });
  utility_.erase(call_id);
  if (!ok) throw EngineTimeoutError("utility call " + method + " timed out");
  if (!p->done) throw EngineDeadError();
  if (p->failure) throw UtilityError(*p->failure);
  return p->value;
}

std::vector<std::string> EngineClient::get_supported_tasks(double timeout_s) {
  mp::Value v = call_utility("get_supported_tasks", {}, timeout_s);
  std::vector<std::string> out;
  if (v.is_array())
    for (auto& e : v.arr)
      if (e.is_str()) out.push_back(e.s);
  return out;
}

void EngineClient::set_step_observer(StepObserver obs) {
  std::lock_guard<std::mutex> lk(mu_);
  observer_ = std::move(obs);
}
void EngineClient::set_batch_handler(BatchHandler h) {
  std::lock_guard<std::mutex> lk(mu_);
  batch_ = std::move(h);
}
void EngineClient::set_dead_callback(std::function<void()> cb) {
  std::lock_guard<std::mutex> lk(mu_);
  dead_cb_ = std::move(cb);
}

SchedulerStatsView EngineClient::scheduler_stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return view_scheduler_stats(sched_stats_);
}
mp::Value EngineClient::scheduler_stats_raw() const {
  std::lock_guard<std::mutex> lk(mu_);
  return sched_stats_;
}
size_t EngineClient::num_inflight() const {
  std::lock_guard<std::mutex> lk(mu_);
  return cbs_.size();
}

void EngineClient::mark_dead(const char* why) {
  if (dead_.exchange(true)) return;
  std::unordered_map<std::string, std::shared_ptr<OutputCallback>> cbs;
  std::function<void()> dcb;
  {
    std::lock_guard<std::mutex> lk(mu_);
    cbs.swap(cbs_);
    dcb = dead_cb_;
  }
  utility_cv_.notify_all();
  Delivery d;
  d.error = why;
  d.engine_dead = true;
  for (auto& [id, cb] : cbs)
    if (cb && *cb) (*cb)(d);  // OutputProcessor.propagate_error(EngineDeadError)
  if (dcb) dcb();
}

void EngineClient::handle_outputs(std::vector<std::string>&& frames) {
  if (is_engine_dead_message(frames)) {
    mark_dead("EngineDeadError");
    return;
  }
  EngineCoreOutputs outs;
  try {
    outs = decode_outputs(frames);
  } catch (const std::exception& e) {
    // process_outputs_socket: an exception is forwarded and kills the output handler -> every request errors.
    static thread_local std::string msg;
    msg = std::string("failed to decode EngineCoreOutputs: ") + e.what();
    mark_dead(msg.c_str());
    return;
  }
  if (outs.utility_output) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = utility_.find(outs.utility_output->call_id);
    if (it != utility_.end()) {
      it->second->done = true;
      it->second->failure = outs.utility_output->failure_message;
      if (auto* v = outs.utility_output->utility_value()) it->second->value = *v;
    }
    utility_cv_.notify_all();
    return;  // `continue` in process_outputs_socket: utility frames carry nothing else
  }
  StepObserver obs;
  BatchHandler batch;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!outs.scheduler_stats.is_nil()) sched_stats_ = outs.scheduler_stats;
    obs = observer_;
    batch = batch_;
    if (batch)
      for (auto& o : outs.outputs)
        if (o.finished()) cbs_.erase(o.request_id);
  }
  if (obs) obs(outs);
  if (batch) {
    batch(outs);
    return;
  }
  for (auto& o : outs.outputs) {
    std::shared_ptr<OutputCallback> cb;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = cbs_.find(o.request_id);
      if (it == cbs_.end()) continue;  // already aborted / unknown: ignored like OutputProcessor
      cb = it->second;
      if (o.finished()) cbs_.erase(it);
    }
    Delivery d;
    d.output = &o;
    d.step = &outs;
    if (*cb) (*cb)(d);
  }
}

void EngineClient::io_loop() {
  std::vector<std::string> frames;
  while (running_) {
    zmq_pollitem_t it[3] = {{out_, 0, ZMQ_POLLIN, 0}, {nullptr, wake_fd_, ZMQ_POLLIN, 0}, {in_, 0, ZMQ_POLLIN, 0}};
    int rc = zmq_poll(it, 3, 1000);
    if (rc < 0) {
      if (zmq_errno() == EINTR) continue;
      if (zmq_errno() == ETERM) break;
      continue;
    }
    if (it[1].revents & ZMQ_POLLIN) {
      uint64_t v;
      while (::read(wake_fd_, &v, sizeof v) > 0) {
      }
    }
    // 1) sends (ADD / ABORT / UTILITY), in submission order
    while (true) {
      std::vector<std::string> msg;
      {
        std::lock_guard<std::mutex> lk(send_mu_);
        if (send_q_.empty()) break;
        msg = std::move(send_q_.front());
        send_q_.pop_front();
      }
      try {
        send_now(msg);
      } catch (const std::exception& e) {
        static thread_local std::string m;
        m = e.what();
        mark_dead(m.c_str());
      }
    }
    // 2) engine -> frontend frames
    if (it[0].revents & ZMQ_POLLIN) {
      for (int k = 0; k < 256; ++k) {
        if (!recv_multipart(out_, frames, ZMQ_DONTWAIT)) break;
        handle_outputs(std::move(frames));
      }
    }
    // 3) late frames on the input socket (another engine identity re-registering): drain and ignore
    if (it[2].revents & ZMQ_POLLIN) {
      while (recv_multipart(in_, frames, ZMQ_DONTWAIT)) {
      }
    }
  }
}

}  // namespace fe::engine
