// rank_scheduler.h -- the C++ port of serve/scheduler.py RankScheduler.
// Decision for decision: the same order of operations, the same float expressions (build with -ffp-contract=off), the
// same tie-breaks (Python's max / min keep the first extreme, sorted() is stable: std::stable_sort, key tuples compared
// lexicographically), the same insertion orders. Python `x or y` on an optional float is truthiness: None and 0.0 both take y (orv() below).
// Golden-trace replay: fpr/core/tests/replay_scheduler.cpp (KV calls checked against the recorded ones).
#pragma once
#include <climits>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "scheduler.h"

namespace fpr {

using OptD = std::optional<double>;

struct SReq {                         // serve/scheduler.py Request (rid / prompt / output / state fields, same names)
  Rid rid;
  int64_t prompt_len = 0, max_new_tokens = 0;
  double arrival = 0.0;
  int32_t priority = 0;
  bool score = false, forced = false;
  std::vector<Image> images;
  Phase phase = Phase::WAITING;
  int32_t rank = -1;                  // None = -1
  int64_t num_computed = 0, num_output = 0, prefill_end = 0, cached_tokens = 0, replay_start = 0;
  int64_t skip_until_step = -1;
  OptD first_token_time, admit_time, pf_progress_time, first_chunk_time, finish_time;
  bool cp_forced = false;
  int64_t last_pf_step = -(int64_t(1) << 30);
  int32_t home_rank = -1;             // None = -1
  int64_t admit_steps_waited = 0;
  std::vector<std::pair<int32_t, std::pair<Rid, int32_t>>> cp_shadows;   // helper rank -> (shadow key, shadow uid), insertion order
  std::vector<std::tuple<int32_t, int64_t, int64_t>> cp_prev_subs;
  int64_t n_preempt = 0, n_migrate = 0, inflight = 0;
  bool first_pending = false;
  int32_t uid = -1;
  int64_t hit_tokens_total = 0, copied_tokens_total = 0;
  // block keys: base hashes (vLLM's in 2a) + ImageBlockKeys salting; keys[i] cached once computed
  std::vector<Hash> base;
  std::vector<uint8_t> have;
  bool image_keys = false;
  bool final_keys = false;            // base hashes ARE the final keys (2a facade: Python's ImageBlockKeys applied there)
  std::vector<int64_t> img_off;       // ImageBlockKeys.offsets
  std::vector<Hash> img_prefix;       // ImageBlockKeys.prefix[k] (k = 0 .. n images), as a 128-bit digest
  std::vector<Hash> keys;
  std::vector<uint8_t> key_real;      // keys[i] came from a known base hash (else a private placeholder)
  int64_t ph_min = INT64_MAX;          // lowest index of a placeholder key (INT64_MAX: none)
  // membership (Python: identity tests `q in self.running`, the prefilling dict)
  bool in_running = false;
  int64_t pf_slot = -1;

  int64_t num_tokens() const { return prompt_len + num_output; }
  int64_t prefill_left() const { return phase == Phase::PREFILL ? prefill_end - num_computed : 0; }
  bool pinned() const { return !images.empty() && phase != Phase::DECODE; }
  const std::pair<Rid, int32_t>* shadow(int32_t ri) const {
    for (auto& e : cp_shadows) if (e.first == ri) return &e.second;
    return nullptr;
  }
};
using ReqP = std::shared_ptr<SReq>;

class RankScheduler : public IScheduler {
 public:
  RankScheduler(const Geometry& g, const CostParams& cost, const SchedConfig& cfg, std::unique_ptr<IKVManager> kv);
  // non-owning KV (the 2a binding: the KV manager belongs to its own Python object, which outlives the scheduler)
  RankScheduler(const Geometry& g, const CostParams& cost, const SchedConfig& cfg, IKVManager* kv);

  // IScheduler
  Status add_request(const Request& r, std::string* reject_reason) override;
  void abort(const Rid& rid) override;
  StepPlan schedule(double now) override;
  std::vector<Rid> update(const StepPlan& plan, const std::vector<std::pair<Rid, int32_t>>& accepted, double now) override;
  int64_t max_new_cap(int64_t prompt_len) const override;
  int32_t seats_per_rank() const override;
  void extend_block_hashes(const Rid& rid, int64_t first_index, const std::vector<Hash>& hashes) override;
  void hold_until(const Rid& rid, int64_t step) override;
  void sync_num_output(const Rid& rid, int64_t n) override;
  void set_paused_new(bool paused) override { paused_new_ = paused; }
  void reset_kv(std::unique_ptr<IKVManager> kv) override { kv_owned_ = std::move(kv); kv_ = kv_owned_.get(); }
  void reset_kv_unowned(IKVManager* kv) { kv_owned_.reset(); kv_ = kv; }
  IKVManager* kv() const { return kv_; }

  // Base block hashes the scheduler was not given (2a: vLLM's, through the adapter's hash_fn): fetch blocks
  // [first, first + n) of a request; returns how many leading ones it filled (fewer = not available yet). Called with
  // the scheduler's state consistent; must not call back into the scheduler.
  std::function<int64_t(const SReq& q, int64_t first, int64_t n, std::vector<Hash>* out)> hash_source;
  bool get(const Rid& rid, ReqState* out) const override;
  std::vector<Rid> running() const override;
  int64_t step_id() const override { return step_id_; }
  std::string stats_json() const override;

  // add_request without prompt ids (the scheduler only needs the length): the replay and the 2a adapter
  Status add(const Rid& rid, int64_t prompt_len, int64_t max_new_tokens, double arrival, int32_t priority, bool score,
             bool forced, const std::vector<Image>& images, const std::vector<Hash>& block_hashes, std::string* why,
             int64_t* max_new_after = nullptr, bool final_keys = false);
  int64_t max_request_tokens() const;
  const std::string& error() const { return error_; }   // a Python assert / KeyError equivalent ("" = none)
  void clear_error() { error_.clear(); }
  // the private helpers tests call (RankScheduler._slack / _decode_load / _prefill_sorted / rank_loads)
  std::vector<double> rank_loads_pub() const { return rank_loads(); }
  double slack_pub(double base, double backlog, const double* frac) const { return slack(base, backlog, frac); }
  double decode_load_pub(const SReq& q) const { return decode_load(q); }
  std::vector<SReq*> prefill_sorted_pub(const std::vector<SReq*>& qs, int64_t step, double now) const {
    return prefill_sorted(qs, step, now);
  }

  // ---- state (public for the golden-trace replay, which reads and, for out-of-band changes, writes it)
  Geometry g;
  CostModel cost;
  SchedConfig cfg;
  std::unique_ptr<IKVManager> kv_owned_;
  IKVManager* kv_ = nullptr;
  std::unordered_map<Rid, ReqP> reqs;
  std::vector<Rid> reqs_order;        // insertion order of reqs (Python dict order: snapshots only)
  std::deque<ReqP> waiting;
  std::vector<ReqP> running_;
  int64_t step_id_ = 0, rr = 0, imbalance_steps = 0;
  std::vector<std::pair<std::string, int64_t>> stats;   // insertion order, as the Python dict
  std::unordered_map<Rid, std::string> rejected;
  std::deque<int32_t> free_uids, free_shadow_uids;
  std::vector<SReq*> pf_list;         // prefilling, insertion order (nullptr = removed)
  int64_t pf_count = 0;
  double pf_credit = 0.0;
  int64_t dec_since_mixed = int64_t(1) << 30;
  double lam = 1.0;
  bool paused_new_ = false;
  int64_t private_hash_ = 0;

  // helpers the replay uses too
  void pf_add(SReq* q);
  void pf_discard(SReq* q);
  std::vector<SReq*> prefilling() const;
  void set_running(const std::vector<ReqP>& v);
  int64_t& stat(const std::string& k);
  void rebuild_keys(SReq& q);         // after an out-of-band change of images / base hashes
  void setup_image_keys(SReq& q);

 private:
  std::string error_;
  int R() const { return (int)g.n_ranks; }
  int64_t k1() const { return g.rows_per_decode_seq(); }
  int64_t paged_bpp_ = 0, window_bpp_ = 0, state_bpp_ = 0, paged_pages_ = 0;

  void fail(const std::string& why) { if (error_.empty()) error_ = why; }
  const Hash& key_at(SReq& q, int64_t i);
  void ensure_base(SReq& q, int64_t n_blocks);       // ask hash_source for missing base hashes below n_blocks
  std::vector<Hash> block_hashes(SReq& q, int64_t n_tokens);
  const std::vector<Hash>& keys_upto(SReq& q, int64_t n_blocks);
  void take_uid(SReq& q);
  void free_uid(SReq& q);
  std::vector<SReq*> on_rank(int r) const;
  int64_t n_on_rank(int r) const;
  void running_remove(SReq* q);
  double decode_load(const SReq& q) const;
  double backlog_us(const SReq& q) const;
  std::vector<double> rank_loads() const;
  std::optional<int64_t> defer(bool in_schedule = false) const;   // None = free now
  void kv_free(const Rid& id, std::optional<int64_t> d) { if (d) kv_->free_deferred(id, *d); else kv_->free_now(id); }

  bool allocate_or_relieve(SReq& q, int64_t start, int64_t end, StepPlan& plan, int64_t& budget, int64_t window_end);
  void unschedule(StepPlan& plan, const Rid& rid);
  void preempt(SReq& q, StepPlan& plan);
  bool move(SReq& q, StepPlan& plan, int64_t& budget, int exclude, int dst = -1);
  // SchedConfig.bucket_aware
  std::vector<int64_t> buckets_() const;
  static int64_t bucket_of_(int64_t n, const std::vector<int64_t>& B);
  std::vector<int64_t> seat_rows_() const;
  int64_t goal_bucket_(const std::vector<int64_t>& B, const std::vector<int64_t>& rows, int64_t add) const;
  bool past_goal_(const std::vector<int64_t>& B, const std::vector<int64_t>& rows, int64_t G, int r) const;
  bool bucket_rebalance(StepPlan& plan, int64_t& budget, const std::vector<int64_t>& B);
  void maybe_rebalance(StepPlan& plan, int64_t& budget);
  void admit(double now, StepPlan& plan, int64_t& budget);
  std::vector<std::pair<int, int>> placement_order(SReq& q, const std::vector<int64_t>& hits,
                                                  const std::vector<int64_t>* rows = nullptr);
  double slack(double base_us, double backlog_us, const double* frac = nullptr) const;
  void steal(StepPlan& plan, int64_t& budget);
  std::vector<SReq*> prefill_sorted(const std::vector<SReq*>& qs, int64_t step, double now = 0.0) const;
  void fill_prefill(StepPlan& plan, std::vector<RankLoad>& loads, double now);
  int64_t cp_force_min() const;
  bool cp_forced_(const SReq& q) const;
  bool cp_prefill(StepPlan& plan, std::vector<RankLoad>& loads, std::vector<std::vector<SReq*>>& pend, double limit,
                  std::vector<int64_t>& rank_pf, std::vector<int64_t>& rank_tail, int64_t cap, double now);
  void drop_shadows(SReq& q);
  double step_us(const std::vector<RankLoad>& loads, int64_t copy_bytes = 0) const { return cost.step(loads, copy_bytes).total_us; }
  std::vector<Image> images_in(const SReq& q, int64_t lo, int64_t hi) const;
};

Hash image_prefix_digest(const std::vector<Image>& images, size_t k);
Hash image_key(const Hash& base, const Hash& prefix);

}  // namespace fpr
