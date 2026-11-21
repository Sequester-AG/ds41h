// Responses API store: vLLM's response_store / msg_store / event_store, with the
// BoundedStore LRU semantics when enabled
// caps n / 2n / n, read or write marks recently used, evicting a response drops its
// messages and events.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/json.h"

namespace fe::api {

// The stored ResponsesResponse, kept as the fields we re-serialize.
struct StoredResponse {
  std::string id;
  std::string status;   // "queued" | "in_progress" | "completed" | "incomplete" | "cancelled" | "failed"
  std::string body;     // model_dump(mode="json") compact JSON (Starlette dump) of the response
  Json output;          // response output items (for previous_response_id replay)
  int64_t created_at = 0;
};

// Background stream events (event_store): (sequence_number, SSE text) list + completion flag.
struct EventLog {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<int64_t, std::string>> events;  // (sequence_number, "event: ...\ndata: ...\n\n")
  bool done = false;
};

template <class V>
class Lru {
 public:
  explicit Lru(size_t cap = 0) : cap_(cap) {}  // cap 0 = unbounded
  // returns evicted keys
  std::vector<std::string> put(const std::string& k, V v) {
    auto it = map_.find(k);
    if (it != map_.end()) {
      it->second.first = std::move(v);
      order_.splice(order_.end(), order_, it->second.second);
    } else {
      order_.push_back(k);
      map_.emplace(k, std::make_pair(std::move(v), std::prev(order_.end())));
    }
    std::vector<std::string> ev;
    while (cap_ && map_.size() > cap_) {
      std::string old = order_.front();
      order_.pop_front();
      map_.erase(old);
      ev.push_back(std::move(old));
    }
    return ev;
  }
  V* get(const std::string& k) {  // marks recently used
    auto it = map_.find(k);
    if (it == map_.end()) return nullptr;
    order_.splice(order_.end(), order_, it->second.second);
    return &it->second.first;
  }
  bool contains(const std::string& k) const { return map_.count(k) != 0; }
  void erase(const std::string& k) {
    auto it = map_.find(k);
    if (it == map_.end()) return;
    order_.erase(it->second.second);
    map_.erase(it);
  }
  size_t size() const { return map_.size(); }

 private:
  size_t cap_;
  std::list<std::string> order_;
  std::unordered_map<std::string, std::pair<V, std::list<std::string>::iterator>> map_;
};

class ResponsesStore {
 public:
  ResponsesStore(bool bounded, size_t n)
      : responses_(bounded ? n : 0), msgs_(bounded ? 2 * n : 0), events_(bounded ? n : 0) {}

  std::optional<StoredResponse> get_response(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto* r = responses_.get(id);
    if (!r) return std::nullopt;
    return *r;
  }
  void put_response(const StoredResponse& r) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& old : responses_.put(r.id, r)) {
      msgs_.erase(old);
      events_.erase(old);
    }
  }
  // "If the response is already cancelled, don't update it."
  void put_response_unless_cancelled(const StoredResponse& r) {
    std::lock_guard<std::mutex> lk(mu_);
    auto* cur = responses_.get(r.id);
    if (cur && cur->status == "cancelled") return;
    for (auto& old : responses_.put(r.id, r)) {
      msgs_.erase(old);
      events_.erase(old);
    }
  }
  std::optional<Json> get_messages(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto* m = msgs_.get(id);
    if (!m) return std::nullopt;
    return *m;
  }
  void put_messages(const std::string& id, Json msgs) {
    std::lock_guard<std::mutex> lk(mu_);
    msgs_.put(id, std::move(msgs));
  }
  std::shared_ptr<EventLog> get_events(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto* e = events_.get(id);
    return e ? *e : nullptr;
  }
  void put_events(const std::string& id, std::shared_ptr<EventLog> e) {
    std::lock_guard<std::mutex> lk(mu_);
    events_.put(id, std::move(e));
  }
  size_t num_responses() {
    std::lock_guard<std::mutex> lk(mu_);
    return responses_.size();
  }

 private:
  std::mutex mu_;
  Lru<StoredResponse> responses_;
  Lru<Json> msgs_;
  Lru<std::shared_ptr<EventLog>> events_;
};

}  // namespace fe::api
