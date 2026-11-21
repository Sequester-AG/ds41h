// rank_scheduler.cpp -- serve/scheduler.py RankScheduler in C++. See rank_scheduler.h for the
// parity rules. Each method below is the Python method of the same name; comments quote the Python where the
// translation is not literal. Build: -O2 -std=c++17 -ffp-contract=off.
#include "rank_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <tuple>

namespace fpr {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

// Python `x or d` on an Optional[float]: None and 0.0 are falsy
inline double orv(const OptD& x, double d) { return (x.has_value() && *x != 0.0) ? *x : d; }
// Python `x if x is not None else d`
inline double isv(const OptD& x, double d) { return x.has_value() ? *x : d; }
// Python // and % on ints (floor)
inline int64_t fdiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}
inline int64_t fmod_(int64_t a, int64_t b) { return a - fdiv(a, b) * b; }

inline uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline void fnv(uint64_t& h, const void* p, size_t n) {
  const unsigned char* c = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 0x100000001B3ull; }
}

}  // namespace

Hash image_prefix_digest(const std::vector<Image>& images, size_t k) {
  // ImageBlockKeys.prefix[k] = tuple((offset, identifier) for the first k images)
  uint64_t a = 0xCBF29CE484222325ull, b = 0x84222325CBF29CE4ull;
  uint64_t n = k;
  fnv(a, &n, 8);
  fnv(b, &n, 8);
  for (size_t i = 0; i < k; ++i) {
    int64_t off = images[i].offset;
    uint64_t len = images[i].key.size();
    fnv(a, &off, 8); fnv(a, &len, 8); fnv(a, images[i].key.data(), images[i].key.size());
    fnv(b, &len, 8); fnv(b, images[i].key.data(), images[i].key.size()); fnv(b, &off, 8);
  }
  return Hash{splitmix(a), splitmix(b ^ 0x5DEECE66Dull)};
}

Hash image_key(const Hash& base, const Hash& prefix) {   // ("mm", base, prefix) as 128 bits
  uint64_t hi = splitmix(base.hi ^ splitmix(prefix.lo ^ 0x6D6D000000000001ull) ^ splitmix(base.lo + 0x632BE59BD9B4E019ull));
  uint64_t lo = splitmix(base.lo ^ splitmix(prefix.hi ^ 0x6D6D000000000002ull) ^ splitmix(base.hi + 0x8CB92BA72F3D8DD7ull) ^ prefix.lo);
  return Hash{hi, lo};
}

// ------------------------------------------------------------------------------------------------ construction
RankScheduler::RankScheduler(const Geometry& g_, const CostParams& cost_, const SchedConfig& cfg_,
                             std::unique_ptr<IKVManager> kv)
    : RankScheduler(g_, cost_, cfg_, kv.get()) {
  kv_owned_ = std::move(kv);
}

RankScheduler::RankScheduler(const Geometry& g_, const CostParams& cost_, const SchedConfig& cfg_, IKVManager* kv)
    : g(g_), cost(cost_), cfg(cfg_), kv_(kv) {
  for (const char* k : {"preemptions", "migrations", "prefix_copies", "admitted", "finished", "copy_bytes",
                        "rejected_admit", "steals", "copy_waits", "copy_over_budget", "rejected_too_long", "cp_chunks",
                        "cp_rows", "cp_copy_pages"})
    stats.emplace_back(k, 0);
  for (int64_t i = 0; i < g.max_seqs; ++i) free_uids.push_back((int32_t)i);
  for (int64_t i = g.max_seqs; i < 4 * g.max_seqs; ++i) free_shadow_uids.push_back((int32_t)i);
  std::vector<PoolSpec> ps = g.pools();
  for (const PoolSpec& p : ps) {
    if (p.kind == "paged") { paged_bpp_ = p.bytes_per_page; paged_pages_ = p.num_pages; }
    else if (p.kind == "window") window_bpp_ = p.bytes_per_page;
    else if (p.kind == "state") state_bpp_ = p.bytes_per_page;
  }
}

std::unique_ptr<IScheduler> make_scheduler(const Geometry& g, const CostParams& cost, const SchedConfig& cfg,
                                           std::unique_ptr<IKVManager> kv) {
  return std::make_unique<RankScheduler>(g, cost, cfg, std::move(kv));
}

int64_t& RankScheduler::stat(const std::string& k) {
  for (auto& e : stats) if (e.first == k) return e.second;
  stats.emplace_back(k, 0);
  return stats.back().second;
}

std::string RankScheduler::stats_json() const {
  std::ostringstream o;
  o << "{";
  for (size_t i = 0; i < stats.size(); ++i) o << (i ? ", " : "") << "\"" << stats[i].first << "\": " << stats[i].second;
  o << "}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ prefilling (ordered)
void RankScheduler::pf_add(SReq* q) {
  if (q->pf_slot >= 0) return;          // dict assignment of a present key keeps its position
  q->pf_slot = (int64_t)pf_list.size();
  pf_list.push_back(q);
  ++pf_count;
}
void RankScheduler::pf_discard(SReq* q) {
  if (q->pf_slot < 0) return;
  pf_list[q->pf_slot] = nullptr;
  q->pf_slot = -1;
  --pf_count;
  if (pf_list.size() > 64 && (int64_t)pf_list.size() > 2 * pf_count) {   // compact (order kept)
    std::vector<SReq*> v;
    v.reserve(pf_count);
    for (SReq* x : pf_list) if (x) { x->pf_slot = (int64_t)v.size(); v.push_back(x); }
    pf_list.swap(v);
  }
}
std::vector<SReq*> RankScheduler::prefilling() const {
  std::vector<SReq*> v;
  v.reserve(pf_count);
  for (SReq* x : pf_list) if (x) v.push_back(x);
  return v;
}
void RankScheduler::set_running(const std::vector<ReqP>& v) {
  for (auto& q : running_) q->in_running = false;
  running_ = v;
  for (auto& q : running_) q->in_running = true;
}

// ------------------------------------------------------------------------------------------------ block keys
void RankScheduler::setup_image_keys(SReq& q) {
  q.image_keys = true;
  q.img_off.clear();
  q.img_prefix.clear();
  for (const Image& im : q.images) q.img_off.push_back(im.offset);
  for (size_t k = 0; k <= q.images.size(); ++k) q.img_prefix.push_back(image_prefix_digest(q.images, k));
  q.keys.clear();
  q.key_real.clear();
}
void RankScheduler::rebuild_keys(SReq& q) {
  q.keys.clear();
  q.key_real.clear();
}

const Hash& RankScheduler::key_at(SReq& q, int64_t i) {
  while ((int64_t)q.keys.size() <= i) {
    int64_t j = (int64_t)q.keys.size();
    bool real = j < (int64_t)q.have.size() && q.have[j];
    // a block hash nobody supplied: a private key (never shared, never hit) instead of a wrong prefix reuse
    Hash b = real ? q.base[j] : Hash{0xFFFFFFFFFFFFFFFFull, (uint64_t)(++private_hash_)};
    if (q.image_keys && !q.final_keys) {
      // ImageBlockKeys: k = bisect_left(offsets, (i + 1) * page)  (images with offset < the block's end)
      int64_t end = (j + 1) * g.page_tokens;
      size_t k = (size_t)(std::lower_bound(q.img_off.begin(), q.img_off.end(), end) - q.img_off.begin());
      if (k != 0) b = image_key(b, q.img_prefix[k]);
    }
    q.keys.push_back(b);
    q.key_real.push_back(real ? 1 : 0);
    if (!real && j < q.ph_min) q.ph_min = j;
  }
  return q.keys[i];
}
void RankScheduler::ensure_base(SReq& q, int64_t n_blocks) {
  if (!hash_source) return;
  // keys below min(keys.size(), ph_min) are final; from there on, fetch what the request lacks
  int64_t first = std::min<int64_t>((int64_t)q.keys.size(), q.ph_min);
  while (first < n_blocks && first < (int64_t)q.have.size() && q.have[first]) ++first;
  if (first >= n_blocks) return;
  std::vector<Hash> got;
  int64_t n = hash_source(q, first, n_blocks - first, &got);
  if (n <= 0) return;
  if ((int64_t)q.base.size() < first + n) { q.base.resize(first + n); q.have.resize(first + n, 0); }
  for (int64_t i = 0; i < n && i < (int64_t)got.size(); ++i) { q.base[first + i] = got[i]; q.have[first + i] = 1; }
  if (first < (int64_t)q.keys.size()) {            // placeholders handed out from here on: recompute
    q.keys.resize(first);
    q.key_real.resize(first);
    q.ph_min = INT64_MAX;
  }
}
std::vector<Hash> RankScheduler::block_hashes(SReq& q, int64_t n_tokens) {
  int64_t n = fdiv(n_tokens, 64);                  // Request.block_hashes: range(n_tokens // 64) (literal 64)
  ensure_base(q, n);
  std::vector<Hash> out;
  out.reserve(std::max<int64_t>(n, 0));
  for (int64_t i = 0; i < n; ++i) out.push_back(key_at(q, i));
  return out;
}
const std::vector<Hash>& RankScheduler::keys_upto(SReq& q, int64_t n_blocks) {
  ensure_base(q, n_blocks);
  if (n_blocks > 0) key_at(q, n_blocks - 1);
  return q.keys;
}
void RankScheduler::extend_block_hashes(const Rid& rid, int64_t first, const std::vector<Hash>& hs) {
  auto it = reqs.find(rid);
  if (it == reqs.end()) return;
  SReq& q = *it->second;
  int64_t need = first + (int64_t)hs.size();
  if ((int64_t)q.base.size() < need) { q.base.resize(need); q.have.resize(need, 0); }
  for (size_t i = 0; i < hs.size(); ++i) {
    int64_t j = first + (int64_t)i;
    q.base[j] = hs[i];
    q.have[j] = 1;
    if (j < (int64_t)q.keys.size() && !q.key_real[j]) {   // a placeholder was handed out: recompute from here
      q.keys.resize(j);
      q.key_real.resize(j);
      if (q.ph_min >= j) q.ph_min = INT64_MAX;     // (a lower placeholder stays tracked)
    }
  }
}

// ------------------------------------------------------------------------------------------------ requests
void RankScheduler::take_uid(SReq& q) {
  q.uid = free_uids.front();
  free_uids.pop_front();
}
void RankScheduler::free_uid(SReq& q) {
  if (q.uid >= 0) {
    free_uids.push_back(q.uid);
    q.uid = -1;
  }
}

int64_t RankScheduler::max_request_tokens() const {
  int64_t pool = paged_pages_;
  int64_t wm = (int64_t)(g.reserve_watermark * (double)pool);
  return std::min(g.max_model_len, (pool - wm) * g.page_tokens - g.rows_per_decode_seq() - g.draft_window_tokens);
}

int64_t RankScheduler::max_new_cap(int64_t prompt_len) const {
  return std::max<int64_t>(1, g.max_model_len - prompt_len - std::max<int64_t>(0, g.spec_k - 1));
}

static std::string check_images(const std::vector<Image>& images, int64_t prompt_len) {
  int64_t end = 0;
  for (const Image& im : images) {
    int64_t off = im.offset, n = im.length;
    if (n <= 0 || off < end || off + n > prompt_len)
      return "image span (" + std::to_string(off) + ", " + std::to_string(n) +
             ") out of order, overlapping or outside the prompt (" + std::to_string(prompt_len) + " tokens)";
    if (im.key.empty()) return "image without an identifier";
    end = off + n;
  }
  return "";
}

Status RankScheduler::add_request(const Request& r, std::string* why) {
  return add(r.rid, (int64_t)r.prompt_ids.size(), r.sampling.max_tokens, r.arrival, r.priority, r.sampling.prompt_logprobs,
             r.forced, r.images, r.block_hashes, why);
}

Status RankScheduler::add(const Rid& rid, int64_t prompt_len, int64_t max_new_tokens, double arrival, int32_t priority,
                          bool score, bool forced, const std::vector<Image>& images,
                          const std::vector<Hash>& block_hashes_, std::string* why, int64_t* max_new_after,
                          bool final_keys) {
  auto q = std::make_shared<SReq>();
  q->rid = rid;
  q->prompt_len = prompt_len;
  q->arrival = arrival;
  q->priority = priority;
  q->score = score;
  q->forced = forced;
  q->images = images;
  q->base = block_hashes_;
  q->final_keys = final_keys;
  q->have.assign(block_hashes_.size(), 1);
  q->max_new_tokens = std::min(max_new_tokens, max_new_cap(prompt_len));
  if (max_new_after) *max_new_after = q->max_new_tokens;
  int64_t cap = max_request_tokens();
  if (prompt_len + q->max_new_tokens > cap) {
    std::string w = "prompt " + std::to_string(prompt_len) + " + max_tokens " + std::to_string(q->max_new_tokens) +
                    " tokens > " + std::to_string(cap) + " (one rank's KV pool / max_model_len)";
    rejected[rid] = w;
    if (why) *why = w;
    stat("rejected_too_long") += 1;
    q->phase = Phase::FINISHED;
    return Status::REJECTED;
  }
  if (!images.empty()) {
    std::string w = check_images(images, prompt_len);
    if (!w.empty()) {
      rejected[rid] = w;
      if (why) *why = w;
      stat("rejected_images") += 1;
      q->phase = Phase::FINISHED;
      return Status::REJECTED;
    }
    setup_image_keys(*q);
  }
  if (!reqs.count(rid)) reqs_order.push_back(rid);
  reqs[rid] = q;
  waiting.push_back(q);
  return Status::OK;
}

void RankScheduler::running_remove(SReq* q) {
  for (size_t i = 0; i < running_.size(); ++i)
    if (running_[i].get() == q) {
      running_.erase(running_.begin() + (long)i);
      q->in_running = false;
      return;
    }
}

void RankScheduler::abort(const Rid& rid) {
  auto it = reqs.find(rid);
  if (it == reqs.end()) { fail("abort: KeyError " + rid); return; }
  ReqP req = it->second;
  reqs.erase(it);
  reqs_order.erase(std::find(reqs_order.begin(), reqs_order.end(), rid));
  for (size_t i = 0; i < waiting.size(); ++i)
    if (waiting[i] == req) { waiting.erase(waiting.begin() + (long)i); break; }
  if (req->in_running) {
    running_remove(req.get());
    if (!req->cp_shadows.empty()) drop_shadows(*req);
    kv_free(rid, defer());
    free_uid(*req);
  }
  pf_discard(req.get());
  req->phase = Phase::FINISHED;
}

int32_t RankScheduler::seats_per_rank() const {
  return (int32_t)std::min(g.max_seqs_per_rank, g.rows_cap() / g.rows_per_decode_seq());
}

std::vector<SReq*> RankScheduler::on_rank(int r) const {
  std::vector<SReq*> v;
  for (auto& q : running_) if (q->rank == r) v.push_back(q.get());
  return v;
}
int64_t RankScheduler::n_on_rank(int r) const {
  int64_t n = 0;
  for (auto& q : running_) n += q->rank == r;
  return n;
}

void RankScheduler::hold_until(const Rid& rid, int64_t step) {
  auto it = reqs.find(rid);
  if (it != reqs.end() && it->second->skip_until_step < step) it->second->skip_until_step = step;
}
void RankScheduler::sync_num_output(const Rid& rid, int64_t n) {
  auto it = reqs.find(rid);
  if (it != reqs.end() && it->second->phase == Phase::WAITING) it->second->num_output = n;
}
bool RankScheduler::get(const Rid& rid, ReqState* out) const {
  auto it = reqs.find(rid);
  if (it == reqs.end()) return false;
  const SReq& q = *it->second;
  *out = ReqState{q.phase, q.rank, q.uid, q.num_computed, q.num_output, q.inflight, q.replay_start, q.cached_tokens,
                  q.skip_until_step, q.first_pending};
  return true;
}
std::vector<Rid> RankScheduler::running() const {
  std::vector<Rid> v;
  for (auto& q : running_) v.push_back(q->rid);
  return v;
}

// ------------------------------------------------------------------------------------------------ loads
double RankScheduler::decode_load(const SReq& q) const { return cost.decode_seq_rank_us(q.num_tokens(), k1()); }
double RankScheduler::backlog_us(const SReq& q) const {
  if (q.phase != Phase::PREFILL) return 0.0;
  return cost.prefill_rank_us(q.num_computed, q.prefill_left());
}
std::vector<double> RankScheduler::rank_loads() const {
  std::vector<double> out(R(), 0.0);
  for (auto& q : running_) out[q->rank] += decode_load(*q) + backlog_us(*q) / (double)cfg.horizon_steps;
  return out;
}
std::optional<int64_t> RankScheduler::defer(bool in_schedule) const {
  if (!cfg.async_mode) return std::nullopt;
  return step_id_ - (in_schedule ? 2 : 1);
}

std::vector<Image> RankScheduler::images_in(const SReq& q, int64_t lo, int64_t hi) const {
  std::vector<Image> v;
  for (const Image& im : q.images) if (im.offset < hi && im.offset + im.length > lo) v.push_back(im);
  return v;
}

// ------------------------------------------------------------------------------------------------ step
StepPlan RankScheduler::schedule(double now) {
  int64_t step = step_id_;
  step_id_ += 1;
  StepPlan plan;
  plan.step = step;
  int64_t copy_budget = cfg.max_copy_bytes_per_step;
  plan.loads.assign(R(), RankLoad());
  int64_t K1 = k1();

  // 1. decode rows
  std::vector<ReqP> snap = running_;
  for (auto& qp : snap) {
    SReq& q = *qp;
    if (q.phase != Phase::DECODE || !q.in_running) continue;
    if (q.skip_until_step > step) continue;
    int64_t n, end, start;
    if (cfg.async_mode) {
      if (q.num_output + q.inflight + (int64_t)q.first_pending >= q.max_new_tokens) continue;
      int64_t lead = (q.inflight + 1) * K1;
      int64_t room = g.max_model_len - q.num_computed - q.inflight * K1;
      if (room < K1) continue;
      n = K1;
      end = std::min(q.num_computed + lead, g.max_model_len);
      start = q.forced ? q.num_computed + q.inflight * K1 : -1;
    } else {
      n = std::min(K1, g.max_model_len - q.num_computed);
      end = std::min(q.num_computed + n, g.max_model_len);
      start = q.num_computed;
    }
    int64_t wend = end + g.draft_window_tokens;
    if (!allocate_or_relieve(q, q.num_computed, end, plan, copy_budget, wend)) continue;
    if (!error_.empty()) return plan;
    if (q.phase != Phase::DECODE || q.skip_until_step > step) continue;
    SeqWork w;
    w.rid = q.rid; w.rank = q.rank; w.kind = "decode"; w.start = start; w.n = n; w.cached_upto = q.cached_tokens;
    w.replay_start = q.replay_start; w.forced = q.forced; w.uid = q.uid; w.known = q.num_computed;
    plan.work.push_back(std::move(w));
    plan.loads[q.rank].add_decode(n, q.num_tokens());
    if (cfg.async_mode) q.inflight += 1;
  }

  // 2. rebalance
  if (cfg.rebalance && cfg.placement == "balanced" && fmod_(step, cfg.rebalance_every) == 0 && !cfg.async_mode)
    maybe_rebalance(plan, copy_budget);

  // 3. admission
  if (plan.preempted.empty() && !paused_new_) admit(now, plan, copy_budget);
  if (!error_.empty()) return plan;

  // 3b. prefill stealing
  if (cfg.steal && cfg.placement == "balanced" && pf_count) steal(plan, copy_budget);

  // 4. prefill quota
  fill_prefill(plan, plan.loads, now);
  if (!error_.empty()) return plan;

  if (!plan.work.empty()) {
    std::vector<CopyOp> t = kv_->tier_copies(plan.step);
    plan.copies.insert(plan.copies.end(), t.begin(), t.end());
  }
  int64_t cb = 0;
  for (const CopyOp& o : plan.copies) cb += o.pool == 0 ? paged_bpp_ : o.pool == 1 ? window_bpp_ : state_bpp_;
  StepCost c = cost.step(plan.loads, cb);
  plan.est_us = c.total_us;
  plan.idle_frac = c.idle_fraction();
  bool mixed = false;
  for (const SeqWork& w : plan.work) if (w.kind != "decode") { mixed = true; break; }
  if (mixed) dec_since_mixed = 0;
  else if (!plan.work.empty()) dec_since_mixed += 1;
  return plan;
}

// -- 1. growth / pressure
bool RankScheduler::allocate_or_relieve(SReq& q, int64_t start, int64_t end, StepPlan& plan, int64_t& budget,
                                        int64_t window_end) {
  while (!kv_->allocate(q.rid, start, end, window_end)) {
    // victims = sorted(on_rank(q.rank), key=(priority, admit_time or 0.0, str(rid))); victim = victims[-1]
    std::vector<SReq*> vs = on_rank(q.rank);
    if (vs.empty()) { fail("allocate_or_relieve: no victim on rank"); return false; }
    SReq* victim = vs[0];
    auto key = [](const SReq* x) { return std::make_tuple(x->priority, orv(x->admit_time, 0.0), x->rid); };
    for (size_t i = 1; i < vs.size(); ++i)
      if (!(key(vs[i]) < key(victim))) victim = vs[i];   // the last of the maximal keys (stable sort, [-1])
    bool moved = (cfg.async_mode && victim->phase == Phase::DECODE) ? false : move(*victim, plan, budget, q.rank);
    if (!moved) preempt(*victim, plan);
    unschedule(plan, victim->rid);
    if (victim == &q || moved || cfg.async_mode) return false;
  }
  return true;
}

void RankScheduler::unschedule(StepPlan& plan, const Rid& rid) {
  std::vector<int> ranks;
  bool any = false;
  for (const SeqWork& w : plan.work)
    if (w.rid == rid) {
      any = true;
      if (std::find(ranks.begin(), ranks.end(), w.rank) == ranks.end()) ranks.push_back(w.rank);
    }
  if (!any) return;
  plan.work.erase(std::remove_if(plan.work.begin(), plan.work.end(), [&](const SeqWork& w) { return w.rid == rid; }),
                  plan.work.end());
  for (int r : ranks) {
    RankLoad ld;
    for (const SeqWork& w : plan.work)
      if (w.rank == r && w.kind == "decode") ld.add_decode(w.n, reqs.at(w.rid)->num_tokens());
    plan.loads[r] = ld;
  }
}

void RankScheduler::preempt(SReq& q, StepPlan& plan) {
  if (!q.cp_shadows.empty()) drop_shadows(q);
  kv_free(q.rid, defer(true));
  q.inflight = 0;
  q.first_pending = false;
  ReqP keep = nullptr;
  for (auto& x : running_) if (x.get() == &q) { keep = x; break; }
  running_remove(&q);
  free_uid(q);
  pf_discard(&q);
  q.phase = Phase::WAITING;
  q.rank = -1;
  q.num_computed = 0;
  q.cached_tokens = 0;
  q.replay_start = 0;
  q.n_preempt += 1;
  waiting.push_front(keep);
  plan.preempted.push_back(q.rid);
  stat("preemptions") += 1;
}

bool RankScheduler::move(SReq& q, StepPlan& plan, int64_t& budget, int exclude, int dst) {
  if (!q.cp_shadows.empty()) return false;
  if (cfg.image_pin && q.pinned()) {
    stat("image_moves_refused") += 1;
    return false;
  }
  int64_t need_bytes = (int64_t)kv_->block_table(q.rid).size() * paged_bpp_ +
                       (int64_t)kv_->window_table(q.rid).size() * window_bpp_;
  if (need_bytes > budget) return false;
  std::vector<int> cands;
  if (dst >= 0) {
    cands.push_back(dst);
  } else {
    // sorted(..., key=lambda r: (past goal bucket, -free_pages(r))), stable; bucket_aware off: past = false
    std::vector<int64_t> B = buckets_(), rows;
    int64_t G = 0;
    if (!B.empty()) {
      rows = seat_rows_();
      G = goal_bucket_(B, rows, 0);
    }
    std::vector<std::tuple<bool, int64_t, int>> kr;
    for (int r = 0; r < R(); ++r) if (r != exclude) kr.emplace_back(past_goal_(B, rows, G, r), -kv_->free_pages(r), r);
    std::stable_sort(kr.begin(), kr.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(std::get<0>(a), std::get<1>(a)) < std::make_tuple(std::get<0>(b), std::get<1>(b));
    });
    for (auto& e : kr) cands.push_back(std::get<2>(e));
  }
  for (int r : cands) {
    if (n_on_rank(r) >= seats_per_rank()) continue;
    std::vector<CopyOp> ops;
    if (kv_->migrate(q.rid, r, plan.step, q.num_computed, &ops) != Status::OK) continue;
    q.rank = r;
    q.skip_until_step = plan.step + 1;
    q.n_migrate += 1;
    plan.copies.insert(plan.copies.end(), ops.begin(), ops.end());
    plan.migrated.push_back(q.rid);
    budget -= need_bytes;
    stat("migrations") += 1;
    stat("copy_bytes") += need_bytes;
    return true;
  }
  return false;
}

// -- decode buckets (SchedConfig.bucket_aware)
std::vector<int64_t> RankScheduler::buckets_() const {
  std::vector<int64_t> B;
  if (!cfg.bucket_aware || cfg.decode_buckets.empty()) return B;
  std::stringstream ss(cfg.decode_buckets);
  std::string x;
  while (std::getline(ss, x, ':')) {
    if (x.find_first_not_of(" \t\r\n") == std::string::npos) continue;   // `if x.strip()`
    B.push_back(std::stoll(x));
  }
  std::sort(B.begin(), B.end());
  return B;
}
int64_t RankScheduler::bucket_of_(int64_t n, const std::vector<int64_t>& B) {
  for (int64_t b : B) if (n <= b) return b;
  return n;
}
std::vector<int64_t> RankScheduler::seat_rows_() const {
  std::vector<int64_t> rows(R(), 0);
  for (auto& q : running_) rows[q->rank] += k1();
  return rows;
}
int64_t RankScheduler::goal_bucket_(const std::vector<int64_t>& B, const std::vector<int64_t>& rows, int64_t add) const {
  int64_t tot = add, per = (int64_t)R() * k1();
  for (int64_t x : rows) tot += x;
  int64_t G = bucket_of_((tot + per - 1) / per * k1(), B);   // -(-(sum + add) // (R * k1)) * k1
  return G >= cfg.bucket_min_rows ? G : 0;
}
bool RankScheduler::past_goal_(const std::vector<int64_t>& B, const std::vector<int64_t>& rows, int64_t G, int r) const {
  return !B.empty() && G > 0 && bucket_of_(rows[r] + k1(), B) > G;
}
bool RankScheduler::bucket_rebalance(StepPlan& plan, int64_t& budget, const std::vector<int64_t>& B) {
  std::vector<int64_t> rows = seat_rows_();
  int64_t G = goal_bucket_(B, rows, 0);
  int hi = 0, lo = 0;
  for (int i = 1; i < R(); ++i) {
    if (rows[i] > rows[hi]) hi = i;
    if (rows[i] < rows[lo]) lo = i;
  }
  if (G == 0 || bucket_of_(rows[hi], B) <= G) return false;
  if (past_goal_(B, rows, G, lo)) return false;
  SReq* q = nullptr;                                // min(cands, key=len(block_table)): the first smallest
  size_t qbt = 0;
  for (SReq* x : on_rank(hi)) {
    if (x->phase != Phase::DECODE || x->skip_until_step > plan.step) continue;
    size_t b = kv_->block_table(x->rid).size();
    if (!q || b < qbt) { q = x; qbt = b; }
  }
  if (!q) return false;
  if (!move(*q, plan, budget, hi, lo)) return false;
  unschedule(plan, q->rid);
  imbalance_steps = 0;
  stat("bucket_moves") += 1;
  return true;
}

// -- 2. rebalance
void RankScheduler::maybe_rebalance(StepPlan& plan, int64_t& budget) {
  std::vector<int64_t> B = buckets_();
  if (!B.empty() && bucket_rebalance(plan, budget, B)) return;
  std::vector<double> dec(R(), 0.0);
  for (auto& q : running_) if (q->phase == Phase::DECODE) dec[q->rank] += decode_load(*q);
  PySum sum;                                        // sum(dec) (CPython 3.12 sum: cost_model.h PySum)
  for (double d : dec) sum.add(d);
  double mean = sum.value() / (double)dec.size();
  if (mean <= 0) return;
  int hi = 0, lo = 0;
  for (int i = 1; i < R(); ++i) {
    if (dec[i] > dec[hi]) hi = i;
    if (dec[i] < dec[lo]) lo = i;
  }
  if ((dec[hi] - dec[lo]) / mean < cfg.rebalance_threshold) {
    imbalance_steps = 0;
    return;
  }
  imbalance_steps += 1;
  if (imbalance_steps < cfg.rebalance_patience) return;
  double gap = dec[hi] - dec[lo];
  SReq* best = nullptr;
  double best_after = dec[hi] - 0.0;
  for (SReq* q : on_rank(hi)) {
    if (q->phase != Phase::DECODE || q->skip_until_step > plan.step) continue;
    double c = decode_load(*q);
    double after = std::max(dec[hi] - c, dec[lo] + c);
    if (after < best_after - 1e-9 && c < gap) { best = q; best_after = after; }
  }
  if (!best) return;
  if (!B.empty()) {                                 // bucket_aware: a load move never takes `lo` past the goal bucket
    std::vector<int64_t> rows = seat_rows_();
    if (past_goal_(B, rows, goal_bucket_(B, rows, 0), lo)) return;
  }
  if (move(*best, plan, budget, hi, lo)) {
    unschedule(plan, best->rid);
    imbalance_steps = 0;
  }
}

// -- 3. admission
void RankScheduler::admit(double now, StepPlan& plan, int64_t& budget) {
  int64_t n_admit = 0;
  std::vector<int64_t> B = buckets_(), rows;       // bucket_aware: seat rows, kept up to date per admission
  if (!B.empty()) rows = seat_rows_();
  while (!waiting.empty() && n_admit < cfg.max_waiting_admit_per_step) {
    if ((int64_t)running_.size() >= g.max_seqs) return;
    ReqP qp = waiting.front();
    SReq& q = *qp;
    q.prefill_end = q.num_tokens();
    std::vector<Hash> hashes = block_hashes(q, q.prefill_end);
    std::vector<int64_t> hits = kv_->lookup(hashes);
    bool have_res = false, wait = false;
    AdmitResult res{};
    int64_t G = B.empty() ? 0 : goal_bucket_(B, rows, k1());
    for (auto& rs : placement_order(q, hits, B.empty() ? nullptr : &rows)) {
      int r = rs.first, src = rs.second;
      if (wait && past_goal_(B, rows, G, r)) {        // bucket_aware: wait a step for the copy budget instead
        stat("bucket_waits") += 1;
        break;
      }
      if (src >= 0) {
        int64_t cb = (hits[src] - hits[r]) * paged_bpp_;
        if (cb > budget && budget < cfg.max_copy_bytes_per_step) {
          wait = true;
          stat("copy_over_budget") += 1;
          continue;
        }
      }
      if (kv_->admit(q.rid, r, hashes, q.prefill_end, plan.step, src, &res) == Status::OK) {
        have_res = true;
        break;
      }
    }
    if (!have_res) {
      stat(wait ? "copy_waits" : "rejected_admit") += 1;
      return;
    }
    waiting.pop_front();
    n_admit += 1;
    q.rank = res.rank;
    q.phase = Phase::PREFILL;
    q.admit_time = now;
    q.pf_progress_time = now;
    int64_t mx = 0;
    int arg = 0;
    for (size_t i = 0; i < hits.size(); ++i) if (hits[i] > hits[arg]) arg = (int)i;
    for (int64_t h : hits) mx = std::max(mx, h);
    q.home_rank = mx > 0 ? arg : -1;
    q.cached_tokens = res.cached_tokens;
    q.replay_start = res.replay_start;
    q.num_computed = res.replay_start;
    q.hit_tokens_total += res.cached_tokens;
    if (res.copied_blocks) {
      q.copied_tokens_total += res.copied_blocks * g.page_tokens;
      budget -= res.copied_blocks * paged_bpp_;
      stat("prefix_copies") += 1;
      stat("copy_bytes") += res.copied_blocks * paged_bpp_;
    }
    plan.copies.insert(plan.copies.end(), res.copies.begin(), res.copies.end());
    take_uid(q);
    running_.push_back(qp);
    if (!B.empty()) rows[q.rank] += k1();
    q.in_running = true;
    pf_add(&q);
    stat("admitted") += 1;
  }
}

std::vector<std::pair<int, int>> RankScheduler::placement_order(SReq& q, const std::vector<int64_t>& hits,
                                                                const std::vector<int64_t>* rows_in) {
  const int Rn = R();
  std::vector<char> seats(Rn);
  for (int r = 0; r < Rn; ++r) seats[r] = n_on_rank(r) < seats_per_rank();
  std::vector<std::pair<int, int>> out;
  if (cfg.placement == "round_robin") {
    std::vector<int> order;
    for (int i = 0; i < Rn; ++i) order.push_back((int)fmod_(rr + i, Rn));
    rr += 1;
    for (int r : order) if (seats[r]) out.emplace_back(r, -1);
    return out;
  }
  if (cfg.placement == "affinity") {
    std::vector<double> loads = rank_loads();
    std::vector<int> order(Rn);
    for (int r = 0; r < Rn; ++r) order[r] = r;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return std::make_tuple(-hits[a], loads[a]) < std::make_tuple(-hits[b], loads[b]);
    });
    for (int r : order) if (seats[r]) out.emplace_back(r, -1);
    return out;
  }
  std::vector<double> dec(Rn, 0.0), back(Rn, 0.0);
  for (auto& x : running_) {
    dec[x->rank] += decode_load(*x);
    back[x->rank] += backlog_us(*x);
  }
  double add = cost.decode_seq_rank_us(q.prefill_end, k1());
  int best_src = 0;
  for (int r = 1; r < Rn; ++r) if (hits[r] > hits[best_src]) best_src = r;
  struct Cand { double peak, pf; bool copy; int r, src; bool past = false; };
  std::vector<Cand> cands;
  for (int r = 0; r < Rn; ++r) {
    if (!seats[r]) continue;
    int src = hits[best_src] - hits[r] >= cfg.migrate_min_blocks ? best_src : -1;
    int64_t eff = src >= 0 ? hits[src] : hits[r];
    int64_t hit, cached, replay;
    kv_->hit_after_cap(eff, q.prefill_end, &hit, &cached, &replay);
    double peak = 0.0;
    for (int i = 0; i < Rn; ++i) {
      double v = dec[i] + (i == r ? add : 0.0);
      peak = i ? std::max(peak, v) : v;
    }
    double pf = back[r] + cost.prefill_rank_us(replay, q.prefill_end - replay);
    cands.push_back(Cand{peak, pf, src >= 0, r, src});
  }
  if (cands.empty()) return out;
  std::vector<int64_t> B = buckets_(), rows;       // bucket_aware: would this rank's next seat go past the goal bucket?
  int64_t G = 0;
  if (!B.empty()) {
    rows = rows_in ? *rows_in : seat_rows_();      // the caller's current seat rows (admit keeps them), else computed
    G = goal_bucket_(B, rows, k1());
  }
  bool any_keep = false;
  for (auto& c : cands) {
    c.past = past_goal_(B, rows, G, c.r);
    any_keep = any_keep || !c.past;
  }
  bool have = false;                                // best over the candidates within G (all of them if none is)
  double best = 0.0;
  for (auto& c : cands) {
    if (any_keep && c.past) continue;
    best = have ? std::min(best, c.peak) : c.peak;
    have = true;
  }
  double tol = cfg.placement_tol * std::max(best, 1.0);
  std::stable_sort(cands.begin(), cands.end(), [&](const Cand& a, const Cand& b) {
    return std::make_tuple(a.past, a.peak > best + tol, a.pf, a.copy, a.peak, a.r) <
           std::make_tuple(b.past, b.peak > best + tol, b.pf, b.copy, b.peak, b.r);
  });
  for (auto& c : cands) out.emplace_back(c.r, c.src);
  return out;
}

// -- 3b. prefill stealing
double RankScheduler::slack(double base_us, double backlog, const double* frac_) const {
  double want = std::max(cfg.min_slack_us, backlog / (double)cfg.backlog_drain_steps);
  double frac = frac_ ? *frac_ : cfg.max_slack_frac;
  double cap = std::max((1.0 + frac) * base_us, cfg.prefill_floor_us) - base_us;
  if (base_us <= 0.0) return cap;
  return std::max(std::min(want, cap), 0.0);
}

void RankScheduler::steal(StepPlan& plan, int64_t& budget) {
  const int Rn = R();
  for (int64_t it = 0; it < cfg.max_steals_per_step; ++it) {
    std::vector<double> back(Rn, 0.0);
    for (SReq* q : prefilling()) back[q->rank] += backlog_us(*q);
    double base = step_us(plan.loads);
    PySum sb;                                       // sum(back)
    for (double b : back) sb.add(b);
    double sl = slack(base, sb.value());
    std::vector<int> idle;
    for (int r = 0; r < Rn; ++r) if (back[r] < 0.5 * sl && n_on_rank(r) < seats_per_rank()) idle.push_back(r);
    int donor = 0;
    for (int r = 1; r < Rn; ++r) if (back[r] > back[donor]) donor = r;
    if (idle.empty() || back[donor] < cfg.steal_min_backlog_steps * sl) return;
    std::vector<SReq*> cands;
    int64_t n_donor = 0;
    for (SReq* q : prefilling()) {
      if (q->rank == donor) ++n_donor;
      if (q->rank == donor && !(cfg.image_pin && q->pinned()) && q->num_computed == q->replay_start &&
          q->skip_until_step <= plan.step)
        cands.push_back(q);
    }
    std::vector<SReq*> order = prefill_sorted(cands, plan.step);
    if ((int64_t)order.size() == n_donor) cands.assign(order.begin() + (order.empty() ? 0 : 1), order.end());
    else cands = order;
    if (cands.empty()) return;
    SReq* q = cands[0];
    size_t qbt = kv_->block_table(q->rid).size();
    for (size_t i = 1; i < cands.size(); ++i) {
      size_t b = kv_->block_table(cands[i]->rid).size();
      if (b < qbt) { q = cands[i]; qbt = b; }
    }
    std::vector<double> dec(Rn, 0.0);
    for (auto& x : running_) dec[x->rank] += decode_load(*x);
    double c = decode_load(*q);
    double mx = dec[0];
    for (double d : dec) mx = std::max(mx, d);
    std::vector<int> ok;
    for (int r : idle) if (dec[r] + c <= mx * (1.0 + cfg.steal_decode_tol)) ok.push_back(r);
    if (ok.empty()) return;
    int dst = ok[0];
    for (size_t i = 1; i < ok.size(); ++i)
      if (std::make_tuple(back[ok[i]], dec[ok[i]]) < std::make_tuple(back[dst], dec[dst])) dst = ok[i];
    if (!move(*q, plan, budget, donor, dst)) return;
    stat("steals") += 1;
  }
}

std::vector<SReq*> RankScheduler::prefill_sorted(const std::vector<SReq*>& qs, int64_t /*step*/, double now) const {
  std::vector<SReq*> v = qs;
  if (cfg.prefill_order == "fcfs") {
    std::stable_sort(v.begin(), v.end(), [](const SReq* a, const SReq* b) {
      return std::make_tuple(orv(a->admit_time, 0.0), std::cref(a->rid)) < std::make_tuple(orv(b->admit_time, 0.0), std::cref(b->rid));
    });
    return v;
  }
  double a = cfg.prefill_aging_tok_per_s;
  int64_t small = cfg.small_prefill_tokens;
  if (cfg.long_order == "fcfs") {
    auto key = [&](const SReq* q) {
      bool lng = q->prefill_left() > small;
      return std::make_tuple(lng, lng ? orv(q->admit_time, 0.0) : (double)q->prefill_left(), std::cref(q->rid));
    };
    std::stable_sort(v.begin(), v.end(), [&](const SReq* x, const SReq* y) { return key(x) < key(y); });
    return v;
  }
  auto key = [&](const SReq* q) {
    return std::make_tuple(q->prefill_left() > small, (double)q->prefill_left() - a * (now - orv(q->admit_time, now)),
                           std::cref(q->rid));
  };
  std::stable_sort(v.begin(), v.end(), [&](const SReq* x, const SReq* y) { return key(x) < key(y); });
  return v;
}

// -- 4. prefill quota
void RankScheduler::fill_prefill(StepPlan& plan, std::vector<RankLoad>& loads, double now) {
  if (!pf_count) return;
  const int Rn = R();
  std::vector<std::vector<SReq*>> pend(Rn);
  std::vector<SReq*> pfl = prefilling();
  for (int r = 0; r < Rn; ++r) {
    std::vector<SReq*> v;
    for (SReq* q : pfl) if (q->rank == r && q->skip_until_step <= plan.step) v.push_back(q);
    pend[r] = prefill_sorted(v, plan.step, now);
  }
  bool anyp = false;
  for (auto& v : pend) anyp = anyp || !v.empty();
  if (!anyp) return;
  double base = step_us(loads);
  PySum backlog_;                                   // sum(backlog_us(q) + prefill_shared_us(...) for ...)
  for (auto& v : pend) for (SReq* q : v) backlog_.add(backlog_us(*q) + cost.prefill_shared_us(q->prefill_left()));
  double backlog = backlog_.value();
  int64_t small_ = cfg.small_prefill_tokens;
  bool starving_long = false;
  for (auto& v : pend) for (SReq* q : v)
    if (!starving_long && q->prefill_left() > small_ && now - isv(q->admit_time, now) >= cfg.starve_wait_s) starving_long = true;
  double fr = std::max(cfg.max_slack_frac, cfg.starve_slack_frac);
  double sl = slack(base, backlog, starving_long ? &fr : nullptr);
  double fixed = cost.mixed_fixed_us();
  double amort = std::min(fixed * (1.0 + cfg.amortize_mixed), cfg.max_mixed_step_us);
  bool banked = fixed >= cfg.amortize_min_fixed_us && sl < amort;
  bool has_stop = false;
  int64_t stop_rows = 0;
  int64_t target = cfg.mixed_target_tokens;
  int64_t lo_ = cfg.mixed_adapt_lo, hi_ = cfg.mixed_adapt_hi;
  if (hi_ > lo_ && lo_ > 0) {
    int64_t n_dec = 0;
    for (auto& q : running_) n_dec += q->phase == Phase::DECODE;
    lam = std::min(1.0, std::max(cfg.mixed_adapt_min, (double)(n_dec - lo_) / (double)(hi_ - lo_)));
  }
  double lm = lam;
  bool any_dec = false;
  for (auto& q : running_) if (q->phase == Phase::DECODE) { any_dec = true; break; }
  double limit = 0.0;
  if (target > 0 && base > 0.0 && any_dec) {
    int64_t cap_ = g.max_prefill_tokens_per_rank;
    int64_t pending = 0;
    if (cfg.mixed_target_total) {
      for (auto& v : pend) for (SReq* q : v) pending += q->prefill_left();
    } else {
      for (auto& v : pend) {
        int64_t s = 0;
        for (SReq* q : v) s += std::min(q->prefill_left(), cap_);
        pending += std::min(cap_, s);
      }
    }
    bool starving = false, draining = false, full = false, big = false;
    bool until_done = cfg.backlog_drain_until == "finished";
    for (auto& v : pend) for (SReq* q : v) {
      if (now - isv(q->pf_progress_time, now) > lm * cfg.mixed_target_wait_s) starving = true;
      if ((until_done || !q->first_chunk_time.has_value()) && q->prefill_left() > cfg.small_prefill_tokens &&
          now - isv(q->admit_time, now) >= cfg.backlog_drain_wait_s)
        draining = true;
    }
    double fw = cfg.mixed_target_full_wait_s;
    if (fw < INF) {
      for (auto& v : pend) {
        int64_t s = 0;
        for (SReq* q : v) s += std::min(q->prefill_left(), cap_);
        bool waited = false;
        for (SReq* q : v) if (now - isv(q->pf_progress_time, now) >= fw) { waited = true; break; }
        if (s >= cap_ && waited) { full = true; break; }
      }
    }
    if (cfg.cp_force_decode_steps >= 0)
      for (auto& v : pend) for (SReq* q : v) if (!big && cp_forced_(*q)) big = true;
    if ((double)pending < lm * (double)target && !starving && !draining && !full && !big) return;
    int64_t floor0 = (int64_t)std::nearbyint(lm * (double)cfg.mixed_min_decode_steps);
    int64_t floor_ = big ? std::min(floor0, cfg.cp_force_decode_steps) : floor0;
    if (cfg.mixed_target_fill && dec_since_mixed < floor_ && !draining) return;
    banked = false;
    has_stop = !cfg.mixed_target_fill;
    stop_rows = target;
    limit = base + cfg.max_mixed_step_us;
  } else if (banked) {
    double budget;
    if (base <= 0.0 || !any_dec) {
      budget = cfg.max_mixed_step_us;
    } else {
      pf_credit = std::min(pf_credit + sl, cfg.max_mixed_step_us);
      bool starving = false;
      for (SReq* q : pfl) if (now - isv(q->pf_progress_time, now) > cfg.max_prefill_wait_s) { starving = true; break; }
      if (pf_credit < amort && !starving) return;
      budget = pf_credit >= amort ? pf_credit : std::max(pf_credit, 3.0 * fixed);
    }
    limit = base + budget;
  } else {
    limit = base + sl;
  }

  // sched: rid -> [start, n], insertion order
  std::vector<std::pair<SReq*, std::pair<int64_t, int64_t>>> sched;
  auto sched_find = [&](SReq* q) -> std::pair<int64_t, int64_t>* {
    for (auto& e : sched) if (e.first->rid == q->rid) return &e.second;
    return nullptr;
  };
  std::vector<char> done(Rn, 0);
  int64_t cap = g.max_prefill_tokens_per_rank;
  std::vector<int64_t> rank_pf(Rn, 0);
  int64_t tail_cap = g.max_tail_rows_per_rank;
  std::vector<int64_t> rank_tail(Rn);
  for (int r = 0; r < Rn; ++r) rank_tail[r] = loads[r].dec_rows;
  bool any_sched = false;
  if (cfg.cp_prefill) {
    any_sched = cp_prefill(plan, loads, pend, limit, rank_pf, rank_tail, cap, now);
    if (!error_.empty()) return;
    if (cfg.async_mode) {
      for (int r = 0; r < Rn; ++r) {
        std::vector<SReq*> v;
        for (SReq* q : pend[r])
          if (!(q->prefill_left() >= cfg.cp_min && !q->score && !(cfg.image_pin && !q->images.empty()) &&
                q->last_pf_step > plan.step - cfg.cp_async_settle_steps))
            v.push_back(q);
        pend[r] = v;
      }
    }
  }
  // anti-starvation share
  std::vector<SReq*> big(Rn, nullptr);
  std::vector<int64_t> big_left(Rn, 0);
  std::vector<char> has_big(Rn, 0);
  double share_us = cfg.large_share_frac * std::max(0.0, limit - base);
  int64_t small = cfg.small_prefill_tokens;
  if (cfg.large_min_rows > 0) {
    for (int r = 0; r < Rn; ++r) {
      std::vector<SReq*> longs;
      for (SReq* q : pend[r])
        if (q->prefill_left() > small && q->cp_shadows.empty() && now - isv(q->admit_time, now) >= cfg.large_share_after_s)
          longs.push_back(q);
      if (!longs.empty() && std::find(longs.begin(), longs.end(), pend[r][0]) == longs.end()) {
        SReq* b = longs[0];
        for (size_t i = 1; i < longs.size(); ++i)
          if (std::make_tuple(orv(longs[i]->admit_time, 0.0), std::cref(longs[i]->rid)) <
              std::make_tuple(orv(b->admit_time, 0.0), std::cref(b->rid)))
            b = longs[i];
        big[r] = b;
        big_left[r] = cfg.large_min_rows;
        has_big[r] = 1;
      }
    }
  }
  auto in_pend = [&](int r, SReq* q) { return std::find(pend[r].begin(), pend[r].end(), q) != pend[r].end(); };
  while (true) {
    std::vector<int> cand;
    for (int r = 0; r < Rn; ++r) if (!done[r] && !pend[r].empty()) cand.push_back(r);
    if (cand.empty()) break;
    if (has_stop) {
      int64_t s = 0;
      for (int64_t x : rank_pf) s += x;
      if (s >= stop_rows) break;
    }
    int r = cand[0];
    double rbest = cost.rank_sides_sum(loads[r]);
    for (size_t i = 1; i < cand.size(); ++i) {
      double v = cost.rank_sides_sum(loads[cand[i]]);
      if (v < rbest) { r = cand[i]; rbest = v; }
    }
    bool share = has_big[r] && big_left[r] > 0 && in_pend(r, big[r]);
    SReq* q = share ? big[r] : pend[r][0];
    std::pair<int64_t, int64_t>* stp = sched_find(q);
    std::pair<int64_t, int64_t> st = stp ? *stp : std::make_pair(q->num_computed, (int64_t)0);
    int64_t pos = st.first + st.second;
    int64_t left = q->prefill_end - pos;
    int64_t room = std::min(cap - rank_pf[r], g.rows_cap() - loads[r].dec_rows - rank_pf[r]);
    if (room <= 0) { done[r] = 1; continue; }
    int64_t qa = cfg.quantum;
    int64_t pm = fmod_(pos, qa);
    int64_t n = std::min(std::min(left, room), pm ? qa - pm : qa);
    if (share) n = std::min(n, big_left[r]);
    bool cut = cfg.cp_prefill && cfg.cp_replay_first && !q->score && !(cfg.image_pin && !q->images.empty()) &&
               pos < q->cached_tokens && fmod_(q->cached_tokens, g.page_tokens) == 0 &&
               q->prefill_end - q->cached_tokens >= cfg.cp_min;
    if (cut) n = std::min(n, q->cached_tokens - pos);
    int64_t had = st.second;
    int64_t tail_add = q->score ? n : std::min<int64_t>(had + n, SLIDING_WINDOW) - std::min<int64_t>(had, SLIDING_WINDOW);
    if (rank_tail[r] + tail_add > tail_cap) {
      if (share) { big_left[r] = 0; continue; }
      done[r] = 1;
      continue;
    }
    RankLoad trial = loads[r];
    trial.add_prefill(pos, n);
    auto with = [&](int rr_, const RankLoad& ld) {
      return cost.step_of((size_t)Rn, [&](size_t i) -> const RankLoad& { return (int)i == rr_ ? ld : loads[i]; }).total_us;
    };
    double nw = with(r, trial);
    double inc = 0.0;
    if (share) {
      double cur = step_us(loads);
      if (cur <= 0.0) {
        RankLoad probe = loads[r];
        probe.add_prefill(pos, 1);
        cur = with(r, probe);
      }
      inc = nw - cur;
      if (inc > share_us || nw > limit) { big_left[r] = 0; continue; }
    }
    if (nw > limit && any_sched) { done[r] = 1; continue; }
    bool last = pos + n >= q->prefill_end;
    if (!kv_->allocate(q->rid, st.first, pos + n, pos + n + (last ? g.draft_window_tokens : 0))) {
      if (share) { big_left[r] = 0; continue; }
      done[r] = 1;
      continue;
    }
    if (share) {
      big_left[r] -= n;
      share_us -= inc;
    }
    loads[r] = trial;
    rank_pf[r] += n;
    st.second += n;
    if (stp) *stp = st;
    else sched.emplace_back(q, st);
    rank_tail[r] += tail_add;
    q->pf_progress_time = now;
    if (!q->first_chunk_time.has_value()) q->first_chunk_time = now;
    any_sched = true;
    if (pos + n >= q->prefill_end || (cut && pos + n >= q->cached_tokens))
      pend[r].erase(std::find(pend[r].begin(), pend[r].end(), q));
  }
  if (banked && any_sched) pf_credit = std::max(0.0, pf_credit - (step_us(loads) - base));
  for (auto& e : sched) {
    SReq& q = *e.first;
    int64_t start = e.second.first, n = e.second.second;
    bool ok = kv_->allocate(q.rid, start, start + n, start + n + (start + n >= q.prefill_end ? g.draft_window_tokens : 0));
    if (!ok) { fail("fill_prefill: AssertionError " + q.rid); return; }
    bool last = start + n >= q.prefill_end;
    SeqWork w;
    w.rid = q.rid; w.rank = q.rank; w.kind = "prefill"; w.start = start; w.n = n; w.cached_upto = q.cached_tokens;
    w.replay_start = q.replay_start; w.last_chunk = last; w.score = q.score; w.forced = q.forced; w.uid = q.uid;
    w.images = images_in(q, start, start + n);
    plan.work.push_back(std::move(w));
    q.last_pf_step = plan.step;
    if (cfg.async_mode) {
      q.num_computed = start + n;
      if (last) {
        q.phase = Phase::DECODE;
        q.first_pending = true;
        pf_discard(&q);
        kv_->shrink_reservation(q.rid);
      }
    }
  }
}

int64_t RankScheduler::cp_force_min() const {
  int64_t f = cfg.cp_force_min, lo = cfg.mixed_adapt_force_min;
  if (f > 0 && 0 < lo && lo < f && lam < 1.0) return (int64_t)((double)lo + lam * (double)(f - lo));
  return f;
}
bool RankScheduler::cp_forced_(const SReq& q) const {
  int64_t f = cp_force_min();
  return cfg.cp_prefill && f > 0 && !q.score && !(cfg.image_pin && !q.images.empty()) &&
         (q.prefill_left() >= f || (q.cp_forced && q.prefill_left() >= cfg.cp_min));
}

bool RankScheduler::cp_prefill(StepPlan& plan, std::vector<RankLoad>& loads, std::vector<std::vector<SReq*>>& pend,
                               double limit, std::vector<int64_t>& rank_pf, std::vector<int64_t>& rank_tail, int64_t cap,
                               double now) {
  const int Rn = R();
  const int64_t P = g.page_tokens, Wn = g.window;
  bool any_sched = false;
  int64_t big_min = cfg.cp_force_min;
  // fkey(q) as a flat numeric tuple (every mode's tuples have a fixed length, so flat comparison = Python's)
  auto fkey = [&](const SReq* q) -> std::vector<double> {
    if (cfg.cp_force_order == "srpt") {
      double hold = cfg.cp_force_srpt_hold;
      int64_t keep = cfg.cp_force_srpt_min_left;
      if (!q->cp_shadows.empty()) return {q->prefill_left() < keep ? -1.0 : (double)q->prefill_left() * hold};
      return {(double)q->prefill_left()};
    }
    if (cfg.cp_force_order == "tiered") {
      int64_t huge = cfg.cp_force_srpt_min_left ? cfg.cp_force_srpt_min_left : 131072;
      return {(double)(q->prefill_left() >= huge), orv(q->admit_time, 0.0)};
    }
    if (cfg.cp_force_order == "split") {
      if (q->prefill_left() < big_min) return {0.0, (double)q->prefill_left()};
      return {1.0, orv(q->admit_time, 0.0)};
    }
    return {orv(q->admit_time, 0.0)};
  };
  int64_t gap = cfg.cp_async_settle_steps;
  auto settled = [&](const SReq* q) { return cfg.async_mode ? q->last_pf_step <= plan.step - gap : true; };
  std::vector<SReq*> cands;
  for (int r = 0; r < Rn; ++r) {
    std::vector<SReq*> el;
    for (SReq* q : pend[r])
      if (q->prefill_left() >= cfg.cp_min && !q->score && !(cfg.image_pin && !q->images.empty()) &&
          q->num_computed >= q->cached_tokens && fmod_(q->num_computed, P) == 0 && settled(q))
        el.push_back(q);
    if (el.empty()) continue;
    std::vector<SReq*> fq;
    for (SReq* q : el) if (cp_forced_(*q)) fq.push_back(q);
    if (!fq.empty()) {
      SReq* b = fq[0];
      for (size_t i = 1; i < fq.size(); ++i)
        if (std::make_tuple(fkey(fq[i]), std::cref(fq[i]->rid)) < std::make_tuple(fkey(b), std::cref(b->rid))) b = fq[i];
      cands.push_back(b);
    } else {
      cands.push_back(el[0]);
    }
  }
  {
    // sorted(cands, key=((0, fkey(q)) if forced(q) else (1, q.prefill_left), str(q.rid)))
    std::vector<std::pair<std::vector<double>, SReq*>> keyed;
    for (SReq* q : cands) {
      std::vector<double> k;
      if (cp_forced_(*q)) { k.push_back(0.0); for (double x : fkey(q)) k.push_back(x); }
      else { k.push_back(1.0); k.push_back((double)q->prefill_left()); }
      keyed.emplace_back(k, q);
    }
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(std::cref(a.first), std::cref(a.second->rid)) < std::make_tuple(std::cref(b.first), std::cref(b.second->rid));
    });
    cands.clear();
    for (auto& e : keyed) cands.push_back(e.second);
  }
  for (SReq* qp : cands) {
    SReq& q = *qp;
    int owner = q.rank;
    std::vector<int> helpers;
    for (int i = 0; i < Rn; ++i) if (i != owner) helpers.push_back(i);
    double thr = cfg.cp_light_frac > 0 ? cfg.cp_light_frac * (double)q.prefill_left() : (double)cap;
    int64_t n_light = 0;
    for (int i : helpers) {
      int64_t s = 0;
      for (SReq* x : pend[i]) if (x != qp) s += x->prefill_left();
      if ((double)s < thr) ++n_light;
    }
    if (n_light < 2 && !cp_forced_(q)) continue;
    int64_t small = cfg.small_prefill_tokens;
    std::vector<int64_t> short_(Rn, 0);
    for (int i = 0; i < Rn; ++i)
      for (SReq* x : pend[i]) if (x != qp && x->prefill_left() <= small) short_[i] += std::min(x->prefill_left(), small);
    if (cp_forced_(q)) {
      int64_t keep = (int64_t)((1.0 - cfg.cp_force_room_frac) * (double)cap);
      for (auto& x : short_) x = std::min(x, keep);
    }
    int64_t room = 0;
    for (int i = 0; i < Rn; ++i) {
      int64_t v = std::min(cap - rank_pf[i] - short_[i], g.rows_cap() - loads[i].dec_rows - rank_pf[i] - short_[i]);
      room = i ? std::min(room, v) : v;
    }
    int64_t per = fdiv(room, P) * P;
    int64_t start = q.num_computed, left = q.prefill_end - q.num_computed;
    int64_t N = std::min(left, (int64_t)Rn * per);
    if (N < left) N = fdiv(N, cfg.quantum) * cfg.quantum;
    int64_t h = 0, owner_n = 0;
    bool broke = false;
    while (N >= (int64_t)Rn * SLIDING_WINDOW) {
      h = fdiv(fdiv(N, Rn), P) * P;
      owner_n = N - (int64_t)(Rn - 1) * h;
      if (owner_n <= per) { broke = true; break; }
      N -= N < left ? cfg.quantum : P;
    }
    if (!broke) continue;
    if (h < SLIDING_WINDOW || owner_n < SLIDING_WINDOW) continue;
    std::vector<std::tuple<int32_t, int64_t, int64_t>> subs;
    for (size_t j = 0; j < helpers.size(); ++j) subs.emplace_back(helpers[j], start + (int64_t)j * h, h);
    subs.emplace_back(owner, start + (int64_t)(Rn - 1) * h, owner_n);
    int64_t end = start + N;
    bool last = end >= q.prefill_end;
    if (rank_tail[owner] + std::min<int64_t>(owner_n, SLIDING_WINDOW) > g.max_tail_rows_per_rank) continue;
    std::vector<RankLoad> trials = loads;
    for (auto& s : subs) {
      int ri = std::get<0>(s);
      trials[ri].add_prefill(std::get<1>(s), std::get<2>(s));
      if (ri != owner) {
        trials[ri].pf_tail_rows -= std::min<int64_t>(std::get<2>(s), SLIDING_WINDOW);
        trials[ri].pf_cand_pos = loads[ri].pf_cand_pos;
      }
    }
    double nw = step_us(trials) + cfg.cp_exchange_us;
    if (nw > limit && any_sched) continue;
    int64_t owner_start = std::get<1>(subs.back());
    int64_t wend = end + (last ? g.draft_window_tokens : 0);
    if (!kv_->can_allocate(q.rid, owner_start, end, wend)) continue;
    std::vector<int> opened;
    bool ok = true;
    for (size_t j = 0; j < helpers.size(); ++j) {
      int ri = helpers[j];
      if (!q.shadow(ri)) {
        if (free_shadow_uids.empty()) { ok = false; break; }
        Rid key = shadow_key(q.rid, ri);
        if (!kv_->open_shadow(key, ri, q.replay_start)) { ok = false; break; }
        q.cp_shadows.emplace_back(ri, std::make_pair(key, free_shadow_uids.front()));
        free_shadow_uids.pop_front();
        opened.push_back(ri);
      }
      Rid key = q.shadow(ri)->first;
      int64_t s0 = std::get<1>(subs[j]), n0 = std::get<2>(subs[j]);
      kv_->trim_window(key, std::max(q.replay_start, s0 - (Wn - 1)));
      if (!kv_->can_allocate(key, s0, end, s0 + n0)) { ok = false; break; }
    }
    if (!ok) {
      for (int ri : opened) {
        for (size_t i = 0; i < q.cp_shadows.size(); ++i)
          if (q.cp_shadows[i].first == ri) {
            auto e = q.cp_shadows[i].second;
            q.cp_shadows.erase(q.cp_shadows.begin() + (long)i);
            kv_->free_now(e.first);
            free_shadow_uids.push_back(e.second);
            break;
          }
      }
      continue;
    }
    if (!kv_->allocate(q.rid, owner_start, end, wend)) { fail("cp_prefill: AssertionError " + q.rid); return any_sched; }
    std::vector<int64_t> osk_pages = kv_->block_table(q.rid);
    WindowTable osk_win = kv_->window_table(q.rid);
    for (size_t j = 0; j < helpers.size(); ++j) {
      int ri = helpers[j];
      Rid key = q.shadow(ri)->first;
      int32_t suid = q.shadow(ri)->second;
      int64_t s0 = std::get<1>(subs[j]), n0 = std::get<2>(subs[j]);
      if (!kv_->allocate(key, s0, end, s0 + n0)) { fail("cp_prefill: AssertionError " + key); return any_sched; }
      SeqKV* hsk = kv_->seq(key);
      if (!hsk) { fail("cp_prefill: no SeqKV for " + key); return any_sched; }
      int64_t have = hsk->cp_have;
      for (int64_t b = have; b < fdiv(start, P); ++b) {
        int src = owner;
        int64_t spage = osk_pages.at((size_t)b);
        for (auto& ps : q.cp_prev_subs) {
          int pr = std::get<0>(ps);
          int64_t ps0 = std::get<1>(ps), pn0 = std::get<2>(ps);
          if (fdiv(ps0, P) <= b && b < fdiv(ps0 + pn0, P) && pr != ri) {
            if (pr != owner) {
              src = pr;
              spage = kv_->block_table(q.shadow(pr)->first).at((size_t)b);
            }
            break;
          }
        }
        plan.copies.push_back(CopyOp{0, src, spage, ri, kv_->block_table(key).at((size_t)b), plan.step});
        stat("cp_copy_pages") += 1;
      }
      hsk->cp_have = fdiv(s0 + n0, P);
      if (j == 0 && start > q.replay_start) {
        int64_t wt = g.window_page_tokens;
        const WindowTable& hwin = kv_->window_table(key);
        for (int64_t b = fdiv(std::max(q.replay_start, start - (Wn - 1)), wt); b < fdiv(start - 1, wt) + 1; ++b) {
          if (osk_win.contains(b) && hwin.contains(b))
            plan.copies.push_back(CopyOp{POOL_WINDOW, owner, osk_win.get(b, -1), ri, hwin.get(b, -1), plan.step});
        }
      }
      SeqWork w;
      w.rid = key; w.rank = ri; w.kind = "cp"; w.start = s0; w.n = n0; w.cached_upto = 0; w.replay_start = q.replay_start;
      w.uid = suid;
      w.has_cp = true;
      w.cp = CPRole{q.uid, q.rid, owner, subs, (int32_t)j};
      w.images = images_in(q, s0, s0 + n0);
      plan.work.push_back(std::move(w));
      rank_pf[ri] += n0;
    }
    SeqWork w;
    w.rid = q.rid; w.rank = owner; w.kind = "prefill"; w.start = owner_start; w.n = owner_n; w.cached_upto = q.cached_tokens;
    w.replay_start = q.replay_start; w.last_chunk = last; w.score = false; w.forced = q.forced; w.uid = q.uid;
    w.has_cp = true;
    w.cp = CPRole{q.uid, q.rid, owner, subs, (int32_t)(Rn - 1)};
    w.images = images_in(q, owner_start, end);
    plan.work.push_back(std::move(w));
    rank_pf[owner] += owner_n;
    rank_tail[owner] += std::min<int64_t>(owner_n, SLIDING_WINDOW);
    for (int i = 0; i < Rn; ++i) loads[i] = trials[i];
    q.pf_progress_time = now;
    if (!q.first_chunk_time.has_value()) q.first_chunk_time = now;
    pend[owner].erase(std::find(pend[owner].begin(), pend[owner].end(), qp));
    q.cp_prev_subs = subs;
    q.last_pf_step = plan.step;
    q.cp_forced = q.cp_forced || cp_forced_(q);
    if (cfg.async_mode) {
      q.num_computed = end;
      if (last) {
        q.phase = Phase::DECODE;
        q.first_pending = true;
        pf_discard(&q);
        kv_->shrink_reservation(q.rid);
      }
    }
    stat("cp_chunks") += 1;
    stat("cp_rows") += N;
    any_sched = true;
  }
  return any_sched;
}

void RankScheduler::drop_shadows(SReq& q) {
  auto sh = q.cp_shadows;
  for (auto& e : sh) {
    kv_free(e.second.first, defer());
    free_shadow_uids.push_back(e.second.second);
  }
  q.cp_shadows.clear();
  q.cp_prev_subs.clear();
}

// ------------------------------------------------------------------------------------------------ results
std::vector<Rid> RankScheduler::update(const StepPlan& plan, const std::vector<std::pair<Rid, int32_t>>& accepted,
                                       double now) {
  std::vector<Rid> finished;
  std::vector<ReqP> fin_objs;
  for (const SeqWork& w : plan.work) {
    auto it = reqs.find(w.rid);
    if (it == reqs.end()) continue;
    ReqP qp = it->second;
    SReq& q = *qp;
    if (q.phase == Phase::FINISHED || !q.in_running) continue;
    if (w.kind == "decode") {
      const std::pair<Rid, int32_t>* acc = nullptr;
      for (auto& a : accepted) if (a.first == w.rid) acc = &a;   // (dict: the last value of a key wins)
      if (!acc) { fail("update: KeyError " + w.rid); return finished; }
      int64_t a = acc->second;
      if (!(1 <= a && a <= w.n)) { fail("update: AssertionError " + w.rid); return finished; }
      a = std::min(a, q.max_new_tokens - q.num_output);
      q.num_computed += a;
      q.num_output += a;
      if (cfg.async_mode) q.inflight -= 1;
    } else if (cfg.async_mode) {
      if (w.last_chunk) {
        q.num_output += 1;
        q.first_pending = false;
        q.first_token_time = now;
        if (!q.cp_shadows.empty()) drop_shadows(q);
      }
    } else {
      q.num_computed = w.start + w.n;
      if (w.last_chunk) {
        q.phase = Phase::DECODE;
        pf_discard(&q);
        q.num_output += 1;
        q.first_token_time = now;
        kv_->shrink_reservation(q.rid);
        if (!q.cp_shadows.empty()) drop_shadows(q);
      }
    }
    int64_t settled = (cfg.async_mode && w.kind == "prefill") ? w.start + w.n : q.num_computed;
    int64_t nb = std::min<int64_t>(fdiv(settled, g.page_tokens), (int64_t)kv_->block_table(q.rid).size());
    kv_->commit(q.rid, settled, keys_upto(q, nb));
    if (q.num_output >= q.max_new_tokens) {
      pf_discard(&q);
      q.phase = Phase::FINISHED;
      q.finish_time = now;
      kv_free(q.rid, defer());
      running_remove(&q);
      free_uid(q);
      finished.push_back(q.rid);
      fin_objs.push_back(qp);
      stat("finished") += 1;
    }
  }
  kv_->release_copies(plan.step);
  kv_->flush_deferred(plan.step);
  for (auto& q : fin_objs) {
    auto it = reqs.find(q->rid);
    if (it != reqs.end()) {
      reqs.erase(it);
      reqs_order.erase(std::find(reqs_order.begin(), reqs_order.end(), q->rid));
    }
  }
  return finished;
}

}  // namespace fpr
