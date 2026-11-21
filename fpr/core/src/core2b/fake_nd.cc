// fake_nd.cc -- see fake_nd.h.
#include "fake_nd.h"

#include <cstdio>
#include <memory>

namespace fpr {
namespace core2b {

// ------------------------------------------------------------------ zlib.crc32 (IEEE 802.3, reflected)
uint32_t crc32(const void* data, size_t n, uint32_t crc) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int j = 0; j < 8; ++j) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  const uint8_t* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
  return ~crc;
}

static uint32_t crc32s(const std::string& s) { return crc32(s.data(), s.size()); }

// ------------------------------------------------------------------ CPython's random.Random(int).random()
namespace {
struct MT {
  uint32_t mt[624];
  int mti = 625;
  void init_genrand(uint32_t s) {
    mt[0] = s;
    for (mti = 1; mti < 624; ++mti) mt[mti] = 1812433253u * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + (uint32_t)mti;
  }
  void init_by_array(const uint32_t* key, size_t len) {   // _randommodule.c init_by_array
    init_genrand(19650218u);
    size_t i = 1, j = 0;
    size_t k = 624 > len ? 624 : len;
    for (; k; --k) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + (uint32_t)j;
      ++i;
      ++j;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
      if (j >= len) j = 0;
    }
    for (k = 623; k; --k) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - (uint32_t)i;
      ++i;
      if (i >= 624) {
        mt[0] = mt[623];
        i = 1;
      }
    }
    mt[0] = 0x80000000u;
    mti = 624;
  }
  uint32_t next() {
    static const uint32_t mag01[2] = {0u, 0x9908b0dfu};
    uint32_t y;
    if (mti >= 624) {
      int kk;
      for (kk = 0; kk < 624 - 397; ++kk) {
        y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1u];
      }
      for (; kk < 623; ++kk) {
        y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1u];
      }
      y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
      mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1u];
      mti = 0;
    }
    y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
  }
};
}  // namespace

double py_random(uint32_t seed) {   // random_seed(): key = abs(n) as 32-bit words ([0] for 0); n < 2^32 -> one word
  MT m;
  uint32_t key[1] = {seed};
  m.init_by_array(key, 1);
  uint32_t a = m.next() >> 5, b = m.next() >> 6;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// ------------------------------------------------------------------ the fake model
struct FakeHandleImpl {
  std::vector<StepResult> results;
  std::shared_ptr<FakeNd::SimStep> sim;
};

static double ms_between(FakeNd::TP a, FakeNd::TP b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
static FakeNd::TP plus_ms(FakeNd::TP t, double ms) {
  return t + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

FakeNd::FakeNd(int64_t vocab, int32_t k, int64_t seed, SimConfig sim) : vocab_(vocab), k_(k), seed_(seed), sim_(sim) {
  if (sim_.step_ms > 0) thead_ = std::thread([this] { head_loop(); });
}

FakeNd::~FakeNd() {
  {
    std::lock_guard<std::mutex> g(smu_);
    stop_ = true;
  }
  scv_.notify_all();
  if (thead_.joinable()) thead_.join();
}

void FakeNd::head_loop() {            // T_head: positions -> HSD prepare (CPU) -> enqueue graph H
  for (;;) {
    std::shared_ptr<SimStep> s;
    {
      std::unique_lock<std::mutex> l(smu_);
      scv_.wait(l, [this] { return stop_ || !hq_.empty(); });
      if (stop_) return;
      s = hq_.front();
      hq_.pop_front();
    }
    std::this_thread::sleep_until(s->pos_ready);
    TP t0 = std::chrono::steady_clock::now();
    TP until = plus_ms(t0, sim_.prep_ms);
    volatile uint64_t spin = 0;
    while (std::chrono::steady_clock::now() < until) spin = spin + 1;   // the prepare's CPU work
    TP done = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(smu_);
      s->H_start = done > s->L_end ? done : s->L_end;
      s->H_end = plus_ms(s->H_start, sim_.h_frac * sim_.step_ms);
      s->head_done = true;
      head_busy_ += ms_between(t0, done);
      ++head_steps_;
    }
    scv_.notify_all();
  }
}

void FakeNd::account_entry() {
  if (sim_.step_ms <= 0) return;
  TP now = std::chrono::steady_clock::now();
  if (has_return_) busy_since_wait_ += ms_between(last_return_, now);
}

int32_t FakeNd::true_tok(const Req& st, int64_t pos) const {   // fake_token(seed, key, pos, vocab)
  std::string s = std::to_string(seed_) + "/" + std::to_string(st.key) + "/" + std::to_string(pos);
  int64_t m = vocab_ - 2000 > 1 ? vocab_ - 2000 : 1;
  return (int32_t)(1000 + (int64_t)crc32s(s) % m);
}

int32_t FakeNd::target(const Req& st, int64_t pos, const nd::GrammarRows* g, int64_t i, std::string* err) const {
  int32_t t = true_tok(st, pos);
  if (!g) return t;
  const int32_t* row = g->rows.data() + i * g->words;
  if ((((uint32_t)row[t >> 5]) >> (t & 31)) & 1u) return t;
  for (int64_t w = 0; w < g->words; ++w) {                     // first allowed token
    uint32_t word = (uint32_t)row[w];
    if (word) return (int32_t)(w * 32 + __builtin_ctz(word));
  }
  if (err) *err = "grammar allows no token";
  return t;
}

void FakeNd::new_drafts(Req* st, const Rid& rid, int64_t next_pos, StepResult* res) const {
  static const double tail[3] = {0.80, 0.60, 0.43};             // ACCEPT_TAIL
  std::string s = std::to_string(seed_) + "/" + rid + "/draft/" + std::to_string(next_pos);
  double u = py_random(crc32s(s));
  int n_ok = 0;
  for (int j = 0; j < k_ && j < 3; ++j)
    if (u < tail[j]) ++n_ok;
  std::vector<int32_t> d;
  int64_t m = vocab_ - 2000 > 1 ? vocab_ - 2000 : 1;
  for (int j = 0; j < k_; ++j) {
    int32_t t = true_tok(*st, next_pos + 1 + j);
    if (j >= n_ok) t = (int32_t)(1000 + ((int64_t)t + 7 + j) % m);
    d.push_back(t);
  }
  st->drafts = d;
  if (st->structured) res->drafts.emplace_back(rid, d);
}

Status FakeNd::dispatch(const std::vector<StepBatch>& b, const std::vector<CompactTables>&, const nd::StepHost& host,
                        nd::Handle** h, std::string* err) {
  account_entry();
  ++steps_;
  {
    auto now = std::chrono::steady_clock::now();
    int dec = 0, pf = 0;
    for (const auto& sb : b)
      for (const auto& se : sb.seqs) (se.kind == "decode" ? dec : pf) += 1;
    if (has_last_ && pf == 0 && dec > 0) {
      static const int ub[] = {1, 8, 32, 64, 128, 256, 1 << 30};
      int c = 0;
      while (dec > ub[c]) ++c;
      auto& x = cyc_[ub[c]];
      x.first += std::chrono::duration<double, std::milli>(now - last_).count();
      x.second += 1;
    }
    last_ = now;
    has_last_ = true;
  }
  for (const auto& rid : host.dropped) reqs_.erase(rid);
  for (const auto& q : host.fresh) {                           // _Req(ri, key=prompt_key(prompt))
    Req st;
    std::string bytes;
    bytes.reserve(q.prompt.size() * 4);
    for (int32_t t : q.prompt) {
      uint32_t v = (uint32_t)t;
      char c[4] = {(char)(v & 0xff), (char)((v >> 8) & 0xff), (char)((v >> 16) & 0xff), (char)((v >> 24) & 0xff)};
      bytes.append(c, 4);
    }
    st.key = crc32s(bytes);
    st.structured = q.structured;
    reqs_[q.rid] = st;
  }
  std::map<Rid, const nd::GrammarRows*> gram;
  for (const auto& g : host.grammar) gram[g.rid] = &g;
  std::map<Rid, int32_t> nspec(host.num_spec.begin(), host.num_spec.end());
  auto out = std::make_unique<FakeHandleImpl>();
  for (const auto& sb : b) {
    StepResult res;
    res.rank = sb.rank;
    res.step = sb.step;
    for (const auto& se : sb.seqs) {
      auto it = reqs_.find(se.rid);
      if (it == reqs_.end()) {
        if (err) *err = "fake nd: unknown request " + se.rid;
        return Status::BAD_ARG;
      }
      Req& st = it->second;
      auto gi = gram.find(se.rid);
      const nd::GrammarRows* g = gi == gram.end() ? nullptr : gi->second;
      if (se.kind == "decode") {                               // _decode
        ++groups_;
        int64_t start;
        if (se.start < 0) {
          if (!st.has_pos || st.pos < se.known) {
            if (err) *err = "fake nd: device position behind the host's for " + se.rid;
            return Status::INTERNAL;
          }
          start = st.pos;
          ++dev_groups_;
        } else {
          start = se.start;
        }
        auto ns = nspec.find(se.rid);
        int32_t num_spec = ns == nspec.end() ? se.n_rows - 1 : ns->second;
        int64_t acceptable = g ? std::min<int64_t>(num_spec, g->num_acceptable) : num_spec;
        std::vector<int32_t> toks;
        for (int32_t i = 0; i <= num_spec; ++i) {
          int32_t t = target(st, start + 1 + i, g, i, err);
          toks.push_back(t);
          if (i >= acceptable || i >= (int32_t)st.drafts.size() || st.drafts[i] != t) break;
        }
        res.accepted.emplace_back(se.rid, (int32_t)toks.size());
        res.tokens.emplace_back(se.rid, toks);
        st.has_pos = true;
        st.pos = start + (int64_t)toks.size();
        st.last = toks.back();
        new_drafts(&st, se.rid, st.pos, &res);
      } else if (se.last_chunk) {                              // the final prefill chunk: the first token
        int32_t t = target(st, se.start + se.n_rows, g, 0, err);
        st.has_pos = true;
        st.pos = se.start + se.n_rows;
        st.last = t;
        res.tokens.emplace_back(se.rid, std::vector<int32_t>{t});
        new_drafts(&st, se.rid, st.pos, &res);
      }
    }
    out->results.push_back(std::move(res));
  }
  int sdec = 0, spf = 0;
  for (const auto& sb : b)
    for (const auto& se : sb.seqs) (se.kind == "decode" ? sdec : spf) += 1;
  if (sim_.step_ms > 0 && sdec + spf > 0) {   // the simulated device (steps with rows; frees-only steps run nothing)
    auto s = std::make_shared<SimStep>();
    int dec = sdec, pf = spf;
    if (pf == 0 && dec > 0) {
      static const int ub[] = {1, 8, 32, 64, 128, 256, 1 << 30};
      int c = 0;
      while (dec > ub[c]) ++c;
      s->cls = ub[c];
    }
    std::unique_lock<std::mutex> l(smu_);
    if (prev_) scv_.wait(l, [this] { return prev_->head_done; });   // condition 3: N's head enqueued first
    TP now = std::chrono::steady_clock::now();
    s->dev_start = now;
    if (prev_ && prev_->H_end > now) s->dev_start = prev_->H_end;
    if (prev_) s->idle_T = std::max(0.0, ms_between(prev_->H_end, s->dev_start));
    s->pos_ready = plus_ms(s->dev_start, sim_.t_ms);
    s->L_end = plus_ms(s->dev_start, (1.0 - sim_.h_frac) * sim_.step_ms);
    hq_.push_back(s);
    prev_ = s;
    l.unlock();
    scv_.notify_all();
    out->sim = s;
  }
  *h = reinterpret_cast<nd::Handle*>(out.release());
  if (sim_.step_ms > 0) {
    last_return_ = std::chrono::steady_clock::now();
    has_return_ = true;
  }
  return Status::OK;
}

Status FakeNd::wait(nd::Handle* h, std::vector<StepResult>* out, std::string*) {
  std::unique_ptr<FakeHandleImpl> own(reinterpret_cast<FakeHandleImpl*>(h));
  if (own->sim) {
    account_entry();
    auto s = own->sim;
    {
      std::unique_lock<std::mutex> l(smu_);
      scv_.wait(l, [&] { return s->head_done || stop_; });
    }
    std::this_thread::sleep_until(plus_ms(s->H_end, 0.05));    // + the results' D2H
    if (s->cls > 0 && has_H_end_) {
      SimStat& x = sst_[s->cls];
      double idle_H = std::max(0.0, ms_between(s->L_end, s->H_start));
      x.n += 1;
      x.cycle += ms_between(last_H_end_, s->H_end);
      x.idle_T += s->idle_T;
      x.idle_H += idle_H;
      x.idle_max = std::max(x.idle_max, s->idle_T + idle_H);
      x.busy += busy_since_wait_;
      x.busy_max = std::max(x.busy_max, busy_since_wait_);
    }
    last_H_end_ = s->H_end;
    has_H_end_ = true;
    busy_since_wait_ = 0.0;
    last_return_ = std::chrono::steady_clock::now();
    has_return_ = true;
  }
  *out = std::move(own->results);
  return Status::OK;
}

std::string FakeNd::stats_json() const {
  std::string c = "{";
  for (const auto& kv : cyc_) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s\"dec<=%d\": [%.4f, %lld]", c.size() > 1 ? ", " : "", kv.first,
                  kv.second.second ? kv.second.first / kv.second.second : 0.0, (long long)kv.second.second);
    c += buf;
  }
  c += "}";
  std::string sm = "null";
  if (sim_.step_ms > 0) {
    char head[256];
    std::snprintf(head, sizeof head, "{\"step_ms\": %.2f, \"h_frac\": %.2f, \"t_ms\": %.2f, \"prep_ms\": %.2f, "
                  "\"head_busy_ms\": %.4f, \"classes\": {", sim_.step_ms, sim_.h_frac, sim_.t_ms, sim_.prep_ms,
                  head_steps_ ? head_busy_ / head_steps_ : 0.0);
    sm = head;
    bool first = true;
    for (const auto& kv : sst_) {
      const SimStat& x = kv.second;
      double n = x.n ? (double)x.n : 1.0;
      char buf[320];
      std::snprintf(buf, sizeof buf, "%s\"dec<=%d\": {\"n\": %lld, \"cycle\": %.3f, \"gpu_idle_T\": %.4f, "
                    "\"gpu_idle_H\": %.4f, \"gpu_idle_max\": %.3f, \"loop_busy\": %.3f, \"loop_busy_max\": %.3f}",
                    first ? "" : ", ", kv.first, (long long)x.n, x.cycle / n, x.idle_T / n, x.idle_H / n, x.idle_max,
                    x.busy / n, x.busy_max);
      sm += buf;
      first = false;
    }
    sm += "}}";
  }
  return "{\"steps\": " + std::to_string(steps_) + ", \"groups\": " + std::to_string(groups_) +
         ", \"dev_groups\": " + std::to_string(dev_groups_) + ", \"reqs\": " + std::to_string(reqs_.size()) +
         ", \"cycle_ms_by_class\": " + c + ", \"sim\": " + sm + "}";
}

}  // namespace core2b
}  // namespace fpr
