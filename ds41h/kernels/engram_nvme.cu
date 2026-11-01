// ds41h ENGRAM_NVME: Engram n-gram table rows read from the NVMe drive at lookup time (io_uring poller thread),
// CUDA-graph-safe GPU <-> host signalling through host-mapped words, bit-exact stock dequant on the GPU.
//
// Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0, LICENSES/BRRR-Apache-2.0.txt):
// src/brrr/engram_nvme.py (_issue_kernel / _wait_kernel with host-mapped generation flags, _dequant_rows_kernel) and
// src/brrr/kernels/engram_nvme.cpp (io_uring poller thread, O_DIRECT + buffered descriptors, aligned staging slots).
// Changes: raw io_uring syscalls (no liburing on the host), ONE poller per process serving every Engram layer of the
// process in layer order, re-laid-out 272-B records (e4m3 row + ue8m0 scales in one read, ds41h/engram_nvme.py),
// per-request de-duplication, stock vLLM 0.32.0 head-shard ownership rule, stock-exact dequant pinned in PTX (same
// instruction chain as kernels/engram_host.cu, verified bit-exact against stock `_engram_lookup_kernel`), multi-CTA
// publish, GPU timeout trap and stall / service-time statistics.
//
// Protocol per channel (= one Engram layer in this process), all on the caller's CUDA stream:
//   issue_kernel   ids[k] = hash[t, head0 + h] (k = t * part + h) -> pinned `ids`; fence.sc.sys; req_n; fence;
//                  st.release.sys req_gen = ++counter (device-memory counter => identical work on every graph replay)
//   poller thread  sees req_gen != seen: de-duplicates the owned ids (ids outside [vstart, vend) -> src = -1 = zero
//                  row, stock rule), reads every unique record (O_DIRECT: the 1-2 sectors holding it into a staging
//                  slot + memcpy; buffered: 264 B straight into the pinned rows buffer), writes src[k] = unique index,
//                  mfence, done = gen (release)
//   wait_dequant   thread 0 of every CTA spins on done (ld.acquire.sys) until done >= gen, bar.sync, then every row
//                  out[k] = dequant(rows[src[k]]) (zeros for src < 0) with .relaxed.sys loads (the buffers are
//                  rewritten every step: no stale cached lines)
// At most one request per channel is in flight (the next issue is stream-ordered after the previous wait), so the
// poller never overwrites a buffer the GPU still reads.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

constexpr int kDim = 256;                         // engram_head_dim: e4m3 bytes per row
constexpr int kQuant = 32;                        // ue8m0 block (8 scale bytes per row)
constexpr int kPayload = kDim + kDim / kQuant;    // 264 bytes read per row
constexpr int kRecStride = 272;                   // row stride of the pinned rows buffer (16-B aligned)
constexpr int64_t kSector = 512;
constexpr int kSlot = 1024;                       // O_DIRECT staging slot: a 264-B record spans <= 2 sectors
constexpr int kHist = 128;                        // service-time histogram: 4 buckets per octave of microseconds

// ------------------------------------------------------------------------------------------- host <-> GPU control
struct Ctl {                                      // pinned + mapped; one 4 KiB page per channel
  unsigned long long req_n;                       // [GPU -> CPU] ids in the request
  unsigned long long req_gen;                     // [GPU -> CPU] request generation (release, after ids + req_n)
  unsigned long long pad0[6];
  unsigned long long done;                        // [CPU -> GPU] generation whose src / rows are complete
  unsigned long long pad1[7];
  unsigned long long waits, spun, stall_ns, stall_max_ns;  // [GPU -> CPU] mirror of DevState stats (posted writes)
  unsigned long long d_waits, d_spun, d_stall_ns, d_stall_max_ns;  // same, decode-sized requests only (<= kSmallRows)
};
static_assert(sizeof(Ctl) == 192, "Ctl layout");

struct DevState {                                 // device memory (torch int64 [16]), one per channel
  unsigned long long counter;                     // last issued generation
  unsigned long long gen;                         // generation the next wait must see
  unsigned int arrive;                            // multi-CTA issue: CTAs done writing ids
  unsigned int pad;
  unsigned long long waits, spun, stall_ns, stall_max_ns;
  unsigned long long consumed;                    // generation the last wait consumed (protocol check)
  unsigned long long d_waits, d_spun, d_stall_ns, d_stall_max_ns;
  unsigned long long pad2[4];
};
static_assert(sizeof(DevState) == 128, "DevState layout");
constexpr int kSmallRows = 768;                   // "decode-sized" request: <= 128 tokens at 6 columns per rank

inline int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

inline int hist_bucket(int64_t us) {  // 4 buckets per octave: [2^e (1 + m/4), 2^e (1 + (m + 1)/4))
  if (us < 1) us = 1;
  const int e = 63 - __builtin_clzll(uint64_t(us));
  const int m = e >= 2 ? int((uint64_t(us) >> (e - 2)) & 3) : int((uint64_t(us) << (2 - e)) & 3);
  return std::min(4 * e + m, kHist - 1);
}

[[noreturn]] void fatal(const char* what, int err, const std::string& extra = "") {
  std::fprintf(stderr, "ds41h engram_nvme FATAL: %s (%d: %s) %s\n", what, err, err > 0 ? strerror(err) : "", extra.c_str());
  std::fflush(stderr);
  std::abort();
}

// ------------------------------------------------------------------------------------------- raw io_uring
struct Ring {
  int fd = -1;
  unsigned flags = 0;
  unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_array = nullptr, sq_mask = 0, sq_entries = 0;
  unsigned *cq_head = nullptr, *cq_tail = nullptr, cq_mask = 0;
  io_uring_sqe* sqes = nullptr;
  io_uring_cqe* cqes = nullptr;
  unsigned local_tail = 0;

  bool setup(unsigned entries, std::string& err) {
    const unsigned tries[] = {
        IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_SUBMIT_ALL | IORING_SETUP_CQSIZE,
        IORING_SETUP_SUBMIT_ALL | IORING_SETUP_CQSIZE, IORING_SETUP_CQSIZE, 0u};
    io_uring_params p{};
    for (unsigned f : tries) {
      std::memset(&p, 0, sizeof p);
      p.flags = f;
      p.cq_entries = entries * 2;
      fd = int(syscall(__NR_io_uring_setup, entries, &p));
      if (fd >= 0) {
        flags = f;
        break;
      }
    }
    if (fd < 0) {
      err = std::string("io_uring_setup: ") + strerror(errno);
      return false;
    }
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
    if (single) sq_sz = cq_sz = std::max(sq_sz, cq_sz);
    void* sq = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (sq == MAP_FAILED) {
      err = std::string("mmap sq: ") + strerror(errno);
      return false;
    }
    void* cq = sq;
    if (!single) {
      cq = mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
      if (cq == MAP_FAILED) {
        err = std::string("mmap cq: ") + strerror(errno);
        return false;
      }
    }
    void* se = mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                    IORING_OFF_SQES);
    if (se == MAP_FAILED) {
      err = std::string("mmap sqes: ") + strerror(errno);
      return false;
    }
    char* s = static_cast<char*>(sq);
    char* c = static_cast<char*>(cq);
    sq_head = reinterpret_cast<unsigned*>(s + p.sq_off.head);
    sq_tail = reinterpret_cast<unsigned*>(s + p.sq_off.tail);
    sq_mask = *reinterpret_cast<unsigned*>(s + p.sq_off.ring_mask);
    sq_entries = *reinterpret_cast<unsigned*>(s + p.sq_off.ring_entries);
    sq_array = reinterpret_cast<unsigned*>(s + p.sq_off.array);
    cq_head = reinterpret_cast<unsigned*>(c + p.cq_off.head);
    cq_tail = reinterpret_cast<unsigned*>(c + p.cq_off.tail);
    cq_mask = *reinterpret_cast<unsigned*>(c + p.cq_off.ring_mask);
    cqes = reinterpret_cast<io_uring_cqe*>(c + p.cq_off.cqes);
    sqes = static_cast<io_uring_sqe*>(se);
    local_tail = *sq_tail;
    return true;
  }

  unsigned unsubmitted() const { return local_tail - __atomic_load_n(sq_head, __ATOMIC_ACQUIRE); }

  io_uring_sqe* next_sqe() {
    if (unsubmitted() >= sq_entries) return nullptr;
    const unsigned idx = local_tail & sq_mask;
    io_uring_sqe* q = &sqes[idx];
    std::memset(q, 0, sizeof *q);
    sq_array[idx] = idx;
    ++local_tail;
    return q;
  }

  // publish the local SQEs, submit them and wait for >= min_complete completions (GETEVENTS also runs the deferred
  // task work of DEFER_TASKRUN rings). Retries interrupted calls; on a full CQ the caller reaps and calls again.
  void submit_wait(unsigned min_complete) {
    __atomic_store_n(sq_tail, local_tail, __ATOMIC_RELEASE);
    for (;;) {
      const unsigned pending = unsubmitted();
      const int r = int(syscall(__NR_io_uring_enter, fd, pending, min_complete, IORING_ENTER_GETEVENTS, nullptr, 0));
      if (r >= 0) return;
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EBUSY) return;  // resources / CQ full: reap first, the caller loops
      fatal("io_uring_enter", errno);
    }
  }

  template <class F>
  unsigned reap(F&& f) {
    unsigned head = *cq_head;
    const unsigned tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
    unsigned n = 0;
    for (; head != tail; ++head, ++n) f(cqes[head & cq_mask]);
    __atomic_store_n(cq_head, head, __ATOMIC_RELEASE);
    return n;
  }
};

// ------------------------------------------------------------------------------------------- engine
struct Channel {
  int layer = 0;
  std::string path;
  int fd_direct = -1, fd_buffered = -1;  // raw descriptors
  int fx_direct = -1, fx_buffered = -1;  // registered (fixed) indices, -1 = use the raw descriptor
  int64_t data_off = 0, rows = 0, per_block = 0, block = 0, rec = 0, vstart = 0, vend = 0, max_rows = 0;
  bool cuda = false;
  Ctl* ctl = nullptr;  // host pointers (== device pointers under UVA for cudaHostAlloc'ed memory)
  int32_t* ids = nullptr;
  int32_t* src = nullptr;
  uint8_t* rows_buf = nullptr;
  Ctl* d_ctl = nullptr;  // device-visible aliases (cudaHostGetDevicePointer)
  int32_t* d_ids = nullptr;
  int32_t* d_src = nullptr;
  uint8_t* d_rows = nullptr;
  uint64_t seen = 0;
  // de-duplication scratch (poller only)
  std::vector<int32_t> uniq, hkey, hval;
  std::vector<uint32_t> htag;
  uint32_t tag = 0;
  std::vector<int32_t> miss;  // unique indices that need a read (not in the record cache)
  int64_t insert_pending = 0;  // misses of the last request not yet inserted into the record cache (deferred)
  int64_t insert_done = 0;     // of those, already inserted (idle passes insert in bounded chunks)
  // record cache (poller only): 2-way set-associative, exact row-id keys, 264-B records; ways = 2 * sets
  int64_t c_sets = 0;
  int c_shift = 64;            // set = (id * golden) >> c_shift  (64 - log2(sets))
  std::vector<int32_t> c_key;  // [2 * sets] row id or -1
  std::vector<uint8_t> c_lru;  // [sets] way to evict next
  uint8_t* c_data = nullptr;   // [2 * sets, kPayload]
  // statistics (poller writes, any thread reads)
  std::atomic<uint64_t> st_req{0}, st_ids{0}, st_owned{0}, st_uniq{0}, st_ios{0}, st_direct{0}, st_bytes{0}, st_svc{0},
      st_svc_max{0}, st_io{0}, st_hits{0};
  std::atomic<uint64_t> hist[kHist];
  // prefetch (opt-in): hash columns of this channel and ids waiting to be read into the record cache
  int li = -1, head0 = 0, part = 0;  // layer index in the hash config, this rank's hash columns [head0, head0 + part)
  std::vector<int32_t> pf_ids;
  size_t pf_pos = 0;
  std::atomic<uint64_t> st_pf_ids{0}, st_pf_reads{0}, st_pf_late{0};
  uint64_t log_req = 0;  // st_req at the last periodic log
  int64_t offset(int64_t r) const { return data_off + (r / per_block) * block + (r % per_block) * rec; }
  Channel() {
    for (auto& h : hist) h.store(0);
  }
};

struct Engine {
  int rank = 0, depth = 512, mode = 2;  // mode: 0 O_DIRECT, 1 buffered, 2 auto (O_DIRECT if unique <= auto_max)
  int64_t auto_max = 512, spin_ns = 50000000, idle_ns = 200000, stats_ns = 0, cache_bytes = 0;
  std::vector<Channel*> ch;
  Ring ring;
  bool fixed_files = false, fixed_buf = false;
  uint8_t* staging = nullptr;
  std::vector<unsigned> free_slots;
  std::vector<int64_t> retry;  // unique indices to resubmit (EAGAIN)
  std::thread thr;
  std::atomic<bool> stop{false};
  bool started = false;
  std::mutex mu;
  // prefetch (opt-in, engine_set_hash + engine_prefetch): stock n-gram hash of future prompt tokens on the CPU ->
  // O_DIRECT reads of their records straight into the record cache, below every GPU request in priority
  struct PfJob {
    std::vector<int32_t> tok;  // [history (<= max_ngram - 1 tokens) | tokens to prefetch]
    int64_t hist = 0;
  };
  std::vector<int32_t> h_tmap;
  std::vector<int64_t> h_mult, h_prime, h_off;
  int64_t h_pad = 0;
  int h_nl = 0, h_ng = 0, h_nh = 0;
  std::atomic<bool> hash_ok{false};
  std::mutex pf_mu;
  std::deque<PfJob> pf_q;
  PfJob pf_cur;
  int64_t pf_cur_pos = 0;
  int64_t pf_inflight = 0, pf_max = 128, pf_chunk = 1024;
  int64_t last_req_ns = 0;  // last GPU request: prefetch backs off to pf_busy reads in flight while requests are frequent
  int64_t pf_busy = 32, pf_busy_window_ns = 20000000;
  std::atomic<uint64_t> st_pf_jobs{0}, st_pf_tokens{0}, st_pf_err{0};

  void read_rows(Channel& c, int64_t U, bool direct);
  void process(Channel& c, uint64_t gen, int64_t n);
  void cache_insert(Channel& c, int64_t max_n = INT64_MAX);
  void cache_put(Channel& c, int32_t id, const uint8_t* rec);
  bool cache_has(const Channel& c, int32_t id) const;
  void pf_complete(const io_uring_cqe& e);
  void pf_hash(int64_t a, int64_t b);
  bool pf_step();
  void loop();
  void log_stats(int64_t window_ns);
};

std::vector<Engine*> g_engines;    // never freed: worker processes exit without tearing the engine down
std::vector<Channel*> g_channels;
std::mutex g_mu;

Engine& engine(int64_t h) {
  TORCH_CHECK(h >= 0 && h < int64_t(g_engines.size()), "engram_nvme: bad engine handle");
  return *g_engines[h];
}
Channel& channel(int64_t h) {
  TORCH_CHECK(h >= 0 && h < int64_t(g_channels.size()), "engram_nvme: bad channel handle");
  return *g_channels[h];
}

void Engine::read_rows(Channel& c, int64_t U, bool direct) {
  int64_t next = 0, inflight = 0, done = 0;
  const int fd = direct ? (c.fx_direct >= 0 ? c.fx_direct : c.fd_direct) : (c.fx_buffered >= 0 ? c.fx_buffered : c.fd_buffered);
  const bool fixed = direct ? c.fx_direct >= 0 : c.fx_buffered >= 0;
  retry.clear();
  while (done < U) {
    while ((next < U || !retry.empty()) && !free_slots.empty()) {
      io_uring_sqe* q = ring.next_sqe();
      if (!q) break;
      int64_t u;
      if (!retry.empty()) {
        u = retry.back();
        retry.pop_back();
      } else {
        u = c.miss[next++];
      }
      const unsigned slot = free_slots.back();
      free_slots.pop_back();
      const int64_t off = c.offset(c.uniq[u]);
      q->fd = fd;
      q->flags = fixed ? IOSQE_FIXED_FILE : 0;
      if (direct) {
        const int64_t a0 = off & ~(kSector - 1), a1 = (off + kPayload + kSector - 1) & ~(kSector - 1);
        q->opcode = fixed_buf ? IORING_OP_READ_FIXED : IORING_OP_READ;
        q->addr = reinterpret_cast<uint64_t>(staging + size_t(slot) * kSlot);
        q->len = unsigned(a1 - a0);
        q->off = uint64_t(a0);
        q->buf_index = 0;
      } else {
        q->opcode = IORING_OP_READ;
        q->addr = reinterpret_cast<uint64_t>(c.rows_buf + u * kRecStride);
        q->len = kPayload;
        q->off = uint64_t(off);
      }
      q->user_data = (uint64_t(u) << 16) | slot;
      ++inflight;
    }
    // all queued: wait for every outstanding read (one wake-up for a decode-sized request); else batch completions
    const bool all_queued = next >= U && retry.empty();
    const unsigned want = unsigned(all_queued ? inflight : std::min<int64_t>(inflight, std::max(1, depth / 8)));
    ring.submit_wait(want);
    ring.reap([&](const io_uring_cqe& e) {
      if (e.user_data >> 63) {  // a prefetch read that was in flight when this request arrived
        pf_complete(e);
        return;
      }
      const int64_t u = int64_t(e.user_data >> 16);
      const unsigned slot = unsigned(e.user_data & 0xffff);
      free_slots.push_back(slot);
      --inflight;
      if (e.res == -EAGAIN || e.res == -EINTR) {
        retry.push_back(u);
        return;
      }
      const int64_t off = c.offset(c.uniq[u]);
      if (e.res < 0) fatal("engram row read", -e.res, c.path + " offset " + std::to_string(off));
      if (direct) {
        const int64_t skew = off & (kSector - 1);
        if (e.res < skew + kPayload) fatal("short O_DIRECT row read", 0, c.path + " offset " + std::to_string(off));
        std::memcpy(c.rows_buf + u * kRecStride, staging + size_t(slot) * kSlot + skew, kPayload);
      } else if (e.res < kPayload) {
        fatal("short buffered row read", 0, c.path + " offset " + std::to_string(off));
      }
      ++done;
    });
  }
  c.st_ios.fetch_add(uint64_t(U), std::memory_order_relaxed);  // U = reads issued (cache misses)
  if (direct) c.st_direct.fetch_add(uint64_t(U), std::memory_order_relaxed);
}

void Engine::process(Channel& c, uint64_t gen, int64_t n) {
  const int64_t t0 = now_ns();
  if (n < 0 || n > c.max_rows) fatal("request larger than the channel", 0, std::to_string(n) + " > " + std::to_string(c.max_rows));
  // de-duplicate owned ids (open addressing, generation tags: no clearing between requests)
  size_t cap = 64;
  while (cap < size_t(2 * n)) cap <<= 1;
  const uint32_t mask = uint32_t(cap - 1);
  if (++c.tag == 0) {
    std::fill(c.htag.begin(), c.htag.end(), 0u);
    c.tag = 1;
  }
  const uint32_t tag = c.tag;
  int64_t U = 0, owned = 0;
  for (int64_t k = 0; k < n; ++k) {
    const int64_t id = c.ids[k];
    if (id < c.vstart || id >= c.vend || id >= c.rows) {
      c.src[k] = -1;  // stock: rows outside this rank's head shard (and DEAD_ID) look up zeros
      continue;
    }
    ++owned;
    uint32_t h = uint32_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 32) & mask;
    while (c.htag[h] == tag && c.hkey[h] != int32_t(id)) h = (h + 1) & mask;
    if (c.htag[h] == tag) {
      c.src[k] = c.hval[h];
    } else {
      c.htag[h] = tag;
      c.hkey[h] = int32_t(id);
      c.hval[h] = int32_t(U);
      c.uniq[U] = int32_t(id);
      c.src[k] = int32_t(U);
      ++U;
    }
  }
  // record cache: hits are copied now, misses are read and then inserted
  int64_t M = 0;
  if (c.c_sets) {
    for (int64_t u = 0; u < U; ++u) {
      const int32_t id = c.uniq[u];
      const int64_t set = int64_t((uint64_t(uint32_t(id)) * 0x9E3779B97F4A7C15ull) >> c.c_shift);
      int way = c.c_key[2 * set] == id ? 0 : (c.c_key[2 * set + 1] == id ? 1 : -1);
      if (way >= 0) {
        std::memcpy(c.rows_buf + u * kRecStride, c.c_data + size_t(2 * set + way) * kPayload, kPayload);
        c.c_lru[set] = uint8_t(way ^ 1);
      } else {
        c.miss[M++] = int32_t(u);
      }
    }
  } else {
    for (int64_t u = 0; u < U; ++u) c.miss[M++] = int32_t(u);
  }
  c.st_hits.fetch_add(uint64_t(U - M), std::memory_order_relaxed);
  const int64_t t1 = now_ns();
  if (M) read_rows(c, M, mode == 0 || (mode == 2 && M <= auto_max));
  __atomic_thread_fence(__ATOMIC_SEQ_CST);  // mfence: src / rows (incl. any non-temporal memcpy stores) before done
  __atomic_store_n(&c.ctl->done, gen, __ATOMIC_RELEASE);
  c.seen = gen;
  c.insert_pending = c.c_sets ? M : 0;  // inserted when the poller is idle / before this channel's next request
  c.insert_done = 0;
  const int64_t t2 = now_ns();
  const uint64_t svc = uint64_t(t2 - t0);
  c.st_req.fetch_add(1, std::memory_order_relaxed);
  c.st_ids.fetch_add(uint64_t(n), std::memory_order_relaxed);
  c.st_owned.fetch_add(uint64_t(owned), std::memory_order_relaxed);
  c.st_uniq.fetch_add(uint64_t(U), std::memory_order_relaxed);
  c.st_bytes.fetch_add(uint64_t(M) * kPayload, std::memory_order_relaxed);
  c.st_svc.fetch_add(svc, std::memory_order_relaxed);
  c.st_io.fetch_add(uint64_t(t2 - t1), std::memory_order_relaxed);
  uint64_t mx = c.st_svc_max.load(std::memory_order_relaxed);
  while (svc > mx && !c.st_svc_max.compare_exchange_weak(mx, svc)) {
  }
  c.hist[hist_bucket(int64_t(svc / 1000))].fetch_add(1, std::memory_order_relaxed);
}

// Insert the last request's misses (still in rows_buf: only this channel's next request overwrites it) into the record
// cache. Deferred out of process() so `done` (and the other layer's request) never waits for it.
void Engine::cache_put(Channel& c, int32_t id, const uint8_t* rec) {
  const int64_t set = int64_t((uint64_t(uint32_t(id)) * 0x9E3779B97F4A7C15ull) >> c.c_shift);
  const int way = c.c_key[2 * set] == id ? 0 : (c.c_key[2 * set + 1] == id ? 1 : (c.c_key[2 * set] < 0 ? 0 : (c.c_key[2 * set + 1] < 0 ? 1 : c.c_lru[set])));
  c.c_key[2 * set + way] = id;
  std::memcpy(c.c_data + size_t(2 * set + way) * kPayload, rec, kPayload);
  c.c_lru[set] = uint8_t(way ^ 1);
}

bool Engine::cache_has(const Channel& c, int32_t id) const {
  if (!c.c_sets) return false;
  const int64_t set = int64_t((uint64_t(uint32_t(id)) * 0x9E3779B97F4A7C15ull) >> c.c_shift);
  return c.c_key[2 * set] == id || c.c_key[2 * set + 1] == id;
}

void Engine::cache_insert(Channel& c, int64_t max_n) {
  const int64_t end = std::min(c.insert_pending, c.insert_done + max_n);
  for (int64_t i = c.insert_done; i < end; ++i) {
    const int64_t u = c.miss[i];
    cache_put(c, c.uniq[u], c.rows_buf + u * kRecStride);
  }
  c.insert_done = end;
  if (end >= c.insert_pending) c.insert_pending = c.insert_done = 0;
}

// prefetch read finished: record -> cache (best effort: errors are counted, never fatal)
void Engine::pf_complete(const io_uring_cqe& e) {
  const unsigned slot = unsigned(e.user_data & 0xffff);
  const int ci = int((e.user_data >> 48) & 0x7fff);
  const int32_t row = int32_t(uint32_t((e.user_data >> 16) & 0xffffffffull));
  free_slots.push_back(slot);
  --pf_inflight;
  if (ci < 0 || ci >= int(ch.size())) return;
  Channel& c = *ch[ci];
  const int64_t skew = c.offset(row) & (kSector - 1);
  if (e.res < skew + kPayload) {
    st_pf_err.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  cache_put(c, row, staging + size_t(slot) * kSlot + skew);
}

// stock `_hash_ids_kernel` for job tokens [a, b) (vllm/models/deepseek_v41/common/engram.py:288-420): rolling XOR of
// compressed id x multiplier over the last max_ngram tokens, n-grams reaching before the request's first token cut with
// the pad id; id = rolling % prime + bucket offset. uint64 products = Triton's int64 wraparound bits; signed % = srem.
// Only this process's hash columns are produced; owned, not cached, unique within the chunk -> c.pf_ids.
void Engine::pf_hash(int64_t a, int64_t b) {
  const std::vector<int32_t>& tok = pf_cur.tok;
  const int ncols = (h_ng - 1) * h_nh;
  for (Channel* c : ch) {
    c->pf_ids.clear();
    c->pf_pos = 0;
    if (++c->tag == 0) {
      std::fill(c->htag.begin(), c->htag.end(), 0u);
      c->tag = 1;
    }
  }
  int64_t rs[8];
  for (int64_t p = a; p < b; ++p) {
    for (Channel* c : ch) {
      if (c->li < 0 || c->li >= h_nl || !c->c_sets) continue;
      const int li = c->li;
      uint64_t rolling = 0;
      bool blocked = false;
      for (int sh = 0; sh < h_ng && sh < 8; ++sh) {
        const int64_t j = p - sh;
        if (j < 0) blocked = true;
        int64_t v = h_pad;
        if (!blocked) {
          const int32_t t = tok[j];
          v = (t >= 0 && size_t(t) < h_tmap.size()) ? int64_t(h_tmap[t]) : h_pad;
        }
        rolling ^= uint64_t(v) * uint64_t(h_mult[li * h_ng + sh]);
        rs[sh] = int64_t(rolling);
      }
      const uint32_t mask = uint32_t(c->htag.size() - 1);
      for (int col = c->head0; col < c->head0 + c->part && col < ncols; ++col) {
        const int sh = col / h_nh + 1, h = col % h_nh;
        const int64_t id = rs[sh] % h_prime[(int64_t(li) * (h_ng - 1) + sh - 1) * h_nh + h] + h_off[int64_t(li) * ncols + col];
        if (id < c->vstart || id >= c->vend || id >= c->rows) continue;
        uint32_t hh = uint32_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 32) & mask;
        while (c->htag[hh] == c->tag && c->hkey[hh] != int32_t(id)) hh = (hh + 1) & mask;
        if (c->htag[hh] == c->tag) continue;  // already queued in this chunk
        c->htag[hh] = c->tag;
        c->hkey[hh] = int32_t(id);
        if (!cache_has(*c, int32_t(id))) c->pf_ids.push_back(int32_t(id));
      }
    }
  }
  for (Channel* c : ch) c->st_pf_ids.fetch_add(c->pf_ids.size(), std::memory_order_relaxed);
}

// one non-blocking round of prefetch work (only called while no GPU request is pending); true if it did anything
bool Engine::pf_step() {
  bool did = false;
  if (pf_inflight > 0) {
    ring.submit_wait(0);  // GETEVENTS: runs deferred completions without blocking
    if (ring.reap([&](const io_uring_cqe& e) { pf_complete(e); })) did = true;
  }
  bool pending = false;
  for (Channel* c : ch) pending |= c->pf_pos < c->pf_ids.size();
  if (!pending && hash_ok.load(std::memory_order_acquire)) {
    if (pf_cur_pos >= int64_t(pf_cur.tok.size())) {
      std::lock_guard<std::mutex> g(pf_mu);
      if (!pf_q.empty()) {
        pf_cur = std::move(pf_q.front());
        pf_q.pop_front();
        pf_cur_pos = pf_cur.hist;
        st_pf_jobs.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (pf_cur_pos < int64_t(pf_cur.tok.size())) {
      int64_t chunk = pf_chunk;  // keep the per-chunk de-duplication table at most half full
      for (Channel* c : ch)
        if (c->part > 0) chunk = std::min<int64_t>(chunk, int64_t(c->htag.size()) / (2 * c->part));
      const int64_t b = std::min<int64_t>(int64_t(pf_cur.tok.size()), pf_cur_pos + std::max<int64_t>(1, chunk));
      pf_hash(pf_cur_pos, b);
      st_pf_tokens.fetch_add(uint64_t(b - pf_cur_pos), std::memory_order_relaxed);
      pf_cur_pos = b;
      did = true;
    }
  }
  bool submitted = false;
  // decode-rate GPU requests (one per layer every ~10 ms) keep the drive queue short: fewer prefetch reads in flight
  const int64_t cap = (now_ns() - last_req_ns < pf_busy_window_ns) ? std::min(pf_busy, pf_max) : pf_max;
  for (size_t ci = 0; ci < ch.size(); ++ci) {
    Channel& c = *ch[ci];
    while (c.pf_pos < c.pf_ids.size() && pf_inflight < cap && !free_slots.empty()) {
      const int32_t row = c.pf_ids[c.pf_pos];
      if (cache_has(c, row)) {  // a GPU request read it meanwhile
        ++c.pf_pos;
        c.st_pf_late.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      io_uring_sqe* q = ring.next_sqe();
      if (!q) break;
      ++c.pf_pos;
      const unsigned slot = free_slots.back();
      free_slots.pop_back();
      const int64_t off = c.offset(row);
      const int64_t a0 = off & ~(kSector - 1), a1 = (off + kPayload + kSector - 1) & ~(kSector - 1);
      q->fd = c.fx_direct >= 0 ? c.fx_direct : c.fd_direct;
      q->flags = c.fx_direct >= 0 ? IOSQE_FIXED_FILE : 0;
      q->opcode = fixed_buf ? IORING_OP_READ_FIXED : IORING_OP_READ;
      q->addr = reinterpret_cast<uint64_t>(staging + size_t(slot) * kSlot);
      q->len = unsigned(a1 - a0);
      q->off = uint64_t(a0);
      q->buf_index = 0;
      q->user_data = (1ull << 63) | (uint64_t(ci) << 48) | (uint64_t(uint32_t(row)) << 16) | slot;
      ++pf_inflight;
      c.st_pf_reads.fetch_add(1, std::memory_order_relaxed);
      submitted = true;
    }
  }
  if (submitted) {
    ring.submit_wait(0);
    did = true;
  }
  return did;
}

void Engine::log_stats(int64_t window_ns) {
  for (Channel* c : ch) {
    const uint64_t req = c->st_req.load();
    if (req == c->log_req) continue;
    c->log_req = req;
    uint64_t h[kHist];
    uint64_t tot = 0;
    for (int i = 0; i < kHist; ++i) tot += (h[i] = c->hist[i].load());
    auto q = [&](double f) {
      uint64_t acc = 0;
      for (int i = 0; i < kHist; ++i) {
        acc += h[i];
        if (acc >= f * double(tot)) return double(1ull << (i / 4)) * (1.0 + double(i % 4 + 1) / 4.0);
      }
      return 0.0;
    };
    const uint64_t ids = c->st_ids.load(), uniq = c->st_uniq.load(), ios = c->st_ios.load();
    const uint64_t waits = c->ctl->waits, spun = c->ctl->spun, stall = c->ctl->stall_ns, smax = c->ctl->stall_max_ns;
    timespec rt;
    clock_gettime(CLOCK_REALTIME, &rt);
    tm utc;
    gmtime_r(&rt.tv_sec, &utc);
    char line[1024];  // one write() per line: 4 worker processes share the log, fprintf(stderr) would interleave mid-line
    const int len = std::snprintf(line, sizeof line,
                 "ds41h engram_nvme %02d:%02d:%02d.%03ld r%d L%d (cumulative): %llu req, %llu ids (%.1f%% unique, record cache "
                 "hits %.1f%%), %llu reads (%.0f%% O_DIRECT, %.2f GB), svc p50 %.0f / p99 %.0f / max %.0f us | GPU waits %llu, "
                 "stalled %llu (%.1f%%), stall mean %.1f / max %.1f us; decode-sized: waits %llu, stalled %.1f%%, stall per "
                 "wait %.2f us, max %.1f us | stall_ns %llu svc_ns %llu d_waits %llu d_spun %llu d_stall_ns %llu\n",
                 utc.tm_hour, utc.tm_min, utc.tm_sec, long(rt.tv_nsec / 1000000),
                 rank, c->layer, (unsigned long long)req, (unsigned long long)ids, ids ? 100.0 * double(uniq) / double(ids) : 0.0,
                 uniq ? 100.0 * double(c->st_hits.load()) / double(uniq) : 0.0,
                 (unsigned long long)ios, ios ? 100.0 * double(c->st_direct.load()) / double(ios) : 0.0,
                 double(c->st_bytes.load()) / 1e9, q(0.5), q(0.99), double(c->st_svc_max.load()) / 1e3,
                 (unsigned long long)waits, (unsigned long long)spun, waits ? 100.0 * double(spun) / double(waits) : 0.0,
                 spun ? double(stall) / double(spun) / 1e3 : 0.0, double(smax) / 1e3, (unsigned long long)c->ctl->d_waits,
                 c->ctl->d_waits ? 100.0 * double(c->ctl->d_spun) / double(c->ctl->d_waits) : 0.0,
                 c->ctl->d_waits ? double(c->ctl->d_stall_ns) / double(c->ctl->d_waits) / 1e3 : 0.0,
                 double(c->ctl->d_stall_max_ns) / 1e3, (unsigned long long)stall, (unsigned long long)c->st_svc.load(),
                 (unsigned long long)c->ctl->d_waits, (unsigned long long)c->ctl->d_spun,
                 (unsigned long long)c->ctl->d_stall_ns);
    if (len > 0) {
      ssize_t w = ::write(2, line, size_t(std::min<int>(len, int(sizeof line) - 1)));
      (void)w;
    }
  }
  (void)window_ns;
}

void Engine::loop() {
  prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);  // 1 us timer slack: idle polls (nanosleep) wake when asked, not 50 us late
  int64_t last = now_ns(), last_log = last;
  while (!stop.load(std::memory_order_relaxed)) {
    Channel* hit = nullptr;
    uint64_t g = 0;
    for (Channel* c : ch) {  // layer order: layer 1's request is always served before layer 14's
      const uint64_t v = __atomic_load_n(&c->ctl->req_gen, __ATOMIC_ACQUIRE);
      if (v != c->seen) {
        hit = c;
        g = v;
        break;
      }
    }
    if (hit) {
      const uint64_t n = __atomic_load_n(&hit->ctl->req_n, __ATOMIC_ACQUIRE);
      if (hit->insert_pending) cache_insert(*hit);  // its rows_buf is about to be overwritten
      process(*hit, g, int64_t(n));
      last = last_req_ns = now_ns();
      continue;
    }
    int64_t t = now_ns();
    if (stats_ns > 0 && t - last_log > stats_ns) {  // no request pending: a good moment for the periodic stats line
      log_stats(t - last_log);
      last_log = t;
    }
    bool inserted = false;  // idle: deferred record-cache inserts (one channel per pass, then re-check requests)
    for (Channel* c : ch) {
      if (c->insert_pending) {
        cache_insert(*c, 4096);  // ~0.2-0.4 ms per pass, then re-check for GPU requests
        inserted = true;
        break;
      }
    }
    if (inserted) continue;
    if (hash_ok.load(std::memory_order_relaxed) && pf_step()) {
      last = now_ns();
      continue;
    }
    t = now_ns();
    if (t - last < spin_ns) {
      for (int i = 0; i < 8; ++i) __builtin_ia32_pause();
    } else {
      timespec ts{0, long(idle_ns)};
      nanosleep(&ts, nullptr);
    }
  }
}

void* alloc_shared(size_t bytes, bool cuda, void** dev) {
  void* p = nullptr;
  if (cuda) {
    TORCH_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess,
                "engram_nvme: cudaHostAlloc(", bytes, ") failed: ", cudaGetErrorString(cudaGetLastError()));
    TORCH_CHECK(cudaHostGetDevicePointer(dev, p, 0) == cudaSuccess, "engram_nvme: cudaHostGetDevicePointer failed");
  } else {
    TORCH_CHECK(posix_memalign(&p, 4096, bytes) == 0, "engram_nvme: posix_memalign failed");
    *dev = p;
  }
  std::memset(p, 0, bytes);
  return p;
}

// ------------------------------------------------------------------------------------------- python API (host)
int64_t engine_new(int64_t depth, int64_t mode, int64_t auto_max, int64_t spin_ns, int64_t idle_ns, int64_t stats_ns,
                   int64_t rank, int64_t cache_bytes) {
  TORCH_CHECK(depth >= 8 && depth <= 4096 && (depth & (depth - 1)) == 0, "engram_nvme: depth must be a power of 2 in [8, 4096]");
  TORCH_CHECK(mode >= 0 && mode <= 2, "engram_nvme: mode 0 (direct) / 1 (buffered) / 2 (auto)");
  auto* e = new Engine();
  e->depth = int(depth);
  e->mode = int(mode);
  e->auto_max = auto_max;
  e->spin_ns = spin_ns;
  e->idle_ns = std::max<int64_t>(1000, std::min<int64_t>(idle_ns, 999999999));
  e->stats_ns = stats_ns;
  e->rank = int(rank);
  e->cache_bytes = std::max<int64_t>(0, cache_bytes);
  std::lock_guard<std::mutex> g(g_mu);
  g_engines.push_back(e);
  return int64_t(g_engines.size() - 1);
}

int64_t engine_add_channel(int64_t h, int64_t layer, std::string path, int64_t data_off, int64_t rows, int64_t per_block,
                           int64_t block, int64_t rec, int64_t vstart, int64_t vend, int64_t max_rows, bool cuda,
                           int64_t cache_bytes) {
  Engine& e = engine(h);
  std::lock_guard<std::mutex> g(e.mu);
  TORCH_CHECK(!e.started, "engram_nvme: add channels before engine_start");
  TORCH_CHECK(rec >= kPayload && rec <= kRecStride * 2 && max_rows > 0 && max_rows < (1ll << 31) && vstart >= 0 && vend >= vstart,
              "engram_nvme: bad channel geometry");
  auto* c = new Channel();
  c->layer = int(layer);
  c->path = path;
  c->data_off = data_off;
  c->rows = rows;
  c->per_block = per_block;
  c->block = block;
  c->rec = rec;
  c->vstart = vstart;
  c->vend = vend;
  c->max_rows = max_rows;
  c->cuda = cuda;
  c->fd_direct = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  c->fd_buffered = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  TORCH_CHECK(c->fd_direct >= 0 && c->fd_buffered >= 0, "engram_nvme: cannot open ", path, ": ", strerror(errno));
  struct stat st {};
  TORCH_CHECK(fstat(c->fd_buffered, &st) == 0, "engram_nvme: fstat failed");
  const int64_t last = c->offset(rows - 1) + kPayload;
  TORCH_CHECK(st.st_size >= ((last + kSector - 1) & ~(kSector - 1)), "engram_nvme: ", path, " is too small for ", rows, " rows");
  posix_fadvise(c->fd_buffered, 0, 0, POSIX_FADV_RANDOM);  // no read-ahead around random rows
  void* dev = nullptr;
  c->ctl = static_cast<Ctl*>(alloc_shared(4096, cuda, &dev));
  c->d_ctl = static_cast<Ctl*>(dev);
  c->ids = static_cast<int32_t*>(alloc_shared(size_t(max_rows) * 4, cuda, &dev));
  c->d_ids = static_cast<int32_t*>(dev);
  c->src = static_cast<int32_t*>(alloc_shared(size_t(max_rows) * 4, cuda, &dev));
  c->d_src = static_cast<int32_t*>(dev);
  c->rows_buf = static_cast<uint8_t*>(alloc_shared(size_t(max_rows) * kRecStride, cuda, &dev));
  c->d_rows = static_cast<uint8_t*>(dev);
  size_t cap = 64;
  while (cap < size_t(2 * max_rows)) cap <<= 1;
  c->uniq.assign(size_t(max_rows), 0);
  c->hkey.assign(cap, 0);
  c->hval.assign(cap, 0);
  c->htag.assign(cap, 0u);
  c->miss.assign(size_t(max_rows), 0);
  if (cache_bytes >= 2 * kPayload * 1024) {  // record cache of this channel (sets: power of two)
    int64_t sets = 2;
    int k = 1;
    while (2 * (2 * sets) * int64_t(kPayload) <= cache_bytes) {
      sets <<= 1;
      ++k;
    }
    c->c_sets = sets;
    c->c_shift = 64 - k;
    c->c_key.assign(size_t(2 * sets), -1);
    c->c_lru.assign(size_t(sets), 0);
    TORCH_CHECK(posix_memalign(reinterpret_cast<void**>(&c->c_data), 2 << 20, size_t(2 * sets) * kPayload) == 0,
                "engram_nvme: record cache allocation failed");
    madvise(c->c_data, size_t(2 * sets) * kPayload, MADV_HUGEPAGE);
    // prefault now (kernel-zeroed), so first-touch faults never land in the poller's request path
    if (madvise(c->c_data, size_t(2 * sets) * kPayload, 23 /* MADV_POPULATE_WRITE */) != 0)
      std::memset(c->c_data, 0, size_t(2 * sets) * kPayload);
  }
  e.ch.push_back(c);
  std::lock_guard<std::mutex> g2(g_mu);
  g_channels.push_back(c);
  return int64_t(g_channels.size() - 1);
}

// Ring setup + registrations; runs ON the poller thread (IORING_SETUP_SINGLE_ISSUER binds the ring to its creator).
bool engine_setup(Engine& e, std::string& note) {
  std::string err;
  if (!e.ring.setup(unsigned(e.depth), err)) {
    note = err;
    return false;
  }
  std::vector<int> fds;
  for (Channel* c : e.ch) {
    fds.push_back(c->fd_direct);
    fds.push_back(c->fd_buffered);
  }
  std::string notes;
  if (!fds.empty() &&
      syscall(__NR_io_uring_register, e.ring.fd, IORING_REGISTER_FILES, fds.data(), unsigned(fds.size())) == 0) {
    e.fixed_files = true;
    for (size_t i = 0; i < e.ch.size(); ++i) {
      e.ch[i]->fx_direct = int(2 * i);
      e.ch[i]->fx_buffered = int(2 * i + 1);
    }
  } else {
    notes += std::string(" no fixed files (") + strerror(errno) + ")";
  }
  if (posix_memalign(reinterpret_cast<void**>(&e.staging), 4096, size_t(e.depth) * kSlot) != 0) {
    note = "staging allocation failed";
    return false;
  }
  std::memset(e.staging, 0, size_t(e.depth) * kSlot);
  iovec iov{e.staging, size_t(e.depth) * kSlot};
  if (syscall(__NR_io_uring_register, e.ring.fd, IORING_REGISTER_BUFFERS, &iov, 1u) == 0) {
    e.fixed_buf = true;
  } else {
    notes += std::string(" no fixed buffer (") + strerror(errno) + ")";
  }
  e.free_slots.clear();
  for (int s = e.depth - 1; s >= 0; --s) e.free_slots.push_back(unsigned(s));
  char buf[256];
  std::snprintf(buf, sizeof buf, "io_uring depth %d flags 0x%x%s%s", e.depth, e.ring.flags,
                (e.ring.flags & IORING_SETUP_DEFER_TASKRUN) ? " (single issuer, deferred task run)" : "", notes.c_str());
  note = buf;
  return true;
}

std::string engine_start(int64_t h) {
  Engine& e = engine(h);
  std::lock_guard<std::mutex> g(e.mu);
  if (e.started) return "";
  std::atomic<int> state{0};  // 0 pending, 1 ok, -1 failed
  std::string note;
  e.stop = false;
  e.thr = std::thread([&e, &state, &note] {
    std::string n;
    const bool ok = engine_setup(e, n);
    note = n;
    state.store(ok ? 1 : -1, std::memory_order_release);
    if (ok) e.loop();
  });
  while (state.load(std::memory_order_acquire) == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
  if (state.load() < 0) {
    e.thr.join();
    TORCH_CHECK(false, "engram_nvme: ", note);
  }
  e.started = true;
  return note;
}

void engine_stop(int64_t h) {
  Engine& e = engine(h);
  std::lock_guard<std::mutex> g(e.mu);
  if (!e.started) return;
  e.stop = true;
  if (e.thr.joinable()) e.thr.join();
  e.started = false;
  if (e.stats_ns > 0) e.log_stats(0);
}

// stock hash parameters for prefetch: token map, pad id, multipliers [nl, ng], primes [nl, ng-1, nh], offsets [nl, (ng-1) nh]
void engine_set_hash(int64_t h, torch::Tensor tmap, int64_t pad, torch::Tensor mult, torch::Tensor primes,
                     torch::Tensor offsets, int64_t max_inflight, int64_t chunk) {
  Engine& e = engine(h);
  std::lock_guard<std::mutex> g(e.mu);
  TORCH_CHECK(!e.hash_ok.load(), "engram_nvme: hash config already set");  // written once; the poller reads it after hash_ok
  TORCH_CHECK(tmap.dtype() == torch::kInt32 && mult.dtype() == torch::kInt64 && primes.dtype() == torch::kInt64 &&
                  offsets.dtype() == torch::kInt64 && mult.dim() == 2 && primes.dim() == 3 && offsets.dim() == 2,
              "engram_nvme: hash config dtypes / shapes");
  auto tm = tmap.contiguous().cpu(), mu = mult.contiguous().cpu(), pr = primes.contiguous().cpu(), of = offsets.contiguous().cpu();
  e.h_tmap.assign(tm.data_ptr<int32_t>(), tm.data_ptr<int32_t>() + tm.numel());
  e.h_mult.assign(mu.data_ptr<int64_t>(), mu.data_ptr<int64_t>() + mu.numel());
  e.h_prime.assign(pr.data_ptr<int64_t>(), pr.data_ptr<int64_t>() + pr.numel());
  e.h_off.assign(of.data_ptr<int64_t>(), of.data_ptr<int64_t>() + of.numel());
  e.h_pad = pad;
  e.h_nl = int(mult.size(0));
  e.h_ng = int(mult.size(1));
  e.h_nh = int(primes.size(2));
  TORCH_CHECK(e.h_ng >= 2 && e.h_ng <= 8 && primes.size(1) == e.h_ng - 1 && offsets.size(1) == (e.h_ng - 1) * e.h_nh,
              "engram_nvme: hash config geometry");
  e.pf_max = std::max<int64_t>(1, std::min<int64_t>(max_inflight, e.depth / 2));
  e.pf_chunk = std::max<int64_t>(16, chunk);
  e.hash_ok.store(true, std::memory_order_release);
}

void channel_set_cols(int64_t ch, int64_t layer_index, int64_t head0, int64_t part) {
  Channel& c = channel(ch);
  c.li = int(layer_index);
  c.head0 = int(head0);
  c.part = int(part);
}

// queue a prefetch job: tokens = [history | tokens to prefetch]; history = up to max_ngram - 1 earlier tokens of the
// same request (0 when the chunk starts at the request's first token)
int64_t engine_prefetch(int64_t h, torch::Tensor tokens, int64_t hist) {
  Engine& e = engine(h);
  if (!e.hash_ok.load()) return -1;
  TORCH_CHECK(!tokens.is_cuda() && tokens.dtype() == torch::kInt32 && tokens.dim() == 1, "engram_nvme: tokens int32 CPU [n]");
  auto t = tokens.contiguous();
  Engine::PfJob j;
  j.tok.assign(t.data_ptr<int32_t>(), t.data_ptr<int32_t>() + t.numel());
  j.hist = std::max<int64_t>(0, std::min<int64_t>(hist, t.numel()));
  std::lock_guard<std::mutex> g(e.pf_mu);
  e.pf_q.push_back(std::move(j));
  return int64_t(e.pf_q.size());
}

// prefetch counters: (jobs, tokens, queued jobs, in flight, errors) + per channel (ids, reads, late)
std::vector<int64_t> engine_pf_stats(int64_t h) {
  Engine& e = engine(h);
  std::vector<int64_t> v = {int64_t(e.st_pf_jobs.load()), int64_t(e.st_pf_tokens.load()), 0, e.pf_inflight,
                            int64_t(e.st_pf_err.load())};
  {
    std::lock_guard<std::mutex> g(e.pf_mu);
    v[2] = int64_t(e.pf_q.size());
  }
  for (Channel* c : e.ch) {
    v.push_back(int64_t(c->st_pf_ids.load()));
    v.push_back(int64_t(c->st_pf_reads.load()));
    v.push_back(int64_t(c->st_pf_late.load()));
  }
  return v;
}

// how many of `ids` are in the channel's record cache (tests; racy against the poller, use when idle)
int64_t channel_cache_count(int64_t ch, torch::Tensor ids) {
  Channel& c = channel(ch);
  auto t = ids.contiguous().to(torch::kInt32);
  int64_t n = 0;
  for (int64_t i = 0; i < t.numel(); ++i) {
    const int32_t id = t.data_ptr<int32_t>()[i];
    if (!c.c_sets) break;
    const int64_t set = int64_t((uint64_t(uint32_t(id)) * 0x9E3779B97F4A7C15ull) >> c.c_shift);
    n += c.c_key[2 * set] == id || c.c_key[2 * set + 1] == id;
  }
  return n;
}

// raw counters + service-time histogram of a channel
std::tuple<std::vector<int64_t>, std::vector<int64_t>> channel_stats(int64_t ch) {
  Channel& c = channel(ch);
  std::vector<int64_t> v = {int64_t(c.st_req.load()),  int64_t(c.st_ids.load()),    int64_t(c.st_owned.load()),
                            int64_t(c.st_uniq.load()), int64_t(c.st_ios.load()),    int64_t(c.st_direct.load()),
                            int64_t(c.st_bytes.load()), int64_t(c.st_svc.load()),   int64_t(c.st_svc_max.load()),
                            int64_t(c.st_io.load()),   int64_t(c.ctl->waits),       int64_t(c.ctl->spun),
                            int64_t(c.ctl->stall_ns),  int64_t(c.ctl->stall_max_ns), int64_t(c.st_hits.load()),
                            int64_t(2 * c.c_sets),     int64_t(c.ctl->d_waits),     int64_t(c.ctl->d_spun),
                            int64_t(c.ctl->d_stall_ns), int64_t(c.ctl->d_stall_max_ns)};
  std::vector<int64_t> hh(kHist);
  for (int i = 0; i < kHist; ++i) hh[i] = int64_t(c.hist[i].load());
  return {v, hh};
}

void channel_reset_stats(int64_t ch) {
  Channel& c = channel(ch);
  for (auto* a : {&c.st_req, &c.st_ids, &c.st_owned, &c.st_uniq, &c.st_ios, &c.st_direct, &c.st_bytes, &c.st_svc,
                  &c.st_svc_max, &c.st_io, &c.st_hits})
    a->store(0);
  for (auto& x : c.hist) x.store(0);
  c.log_req = 0;
}

// (src, rows, ids) of a channel as non-owning CPU tensors (tests)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> channel_buffers(int64_t ch) {
  Channel& c = channel(ch);
  auto o32 = torch::TensorOptions().dtype(torch::kInt32);
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8);
  return {torch::from_blob(c.src, {c.max_rows}, o32), torch::from_blob(c.rows_buf, {c.max_rows, kRecStride}, o8),
          torch::from_blob(c.ids, {c.max_rows}, o32)};
}

// CPU-side requests (tests / I/O benchmark without a GPU), emulating the issue / wait kernels.
std::vector<int64_t> g_cpu_t0(4096, 0);  // publish time per channel handle
uint64_t cpu_publish(int64_t ch, torch::Tensor ids) {
  Channel& c = channel(ch);
  TORCH_CHECK(!ids.is_cuda() && ids.dtype() == torch::kInt32 && ids.is_contiguous(), "ids: contiguous int32 CPU tensor");
  const int64_t n = ids.numel();
  TORCH_CHECK(n <= c.max_rows && ch < int64_t(g_cpu_t0.size()), "too many ids");
  std::memcpy(c.ids, ids.data_ptr<int32_t>(), size_t(n) * 4);
  const uint64_t g = __atomic_load_n(&c.ctl->req_gen, __ATOMIC_RELAXED) + 1;
  __atomic_store_n(&c.ctl->req_n, uint64_t(n), __ATOMIC_RELAXED);
  g_cpu_t0[ch] = now_ns();
  __atomic_store_n(&c.ctl->req_gen, g, __ATOMIC_RELEASE);
  return g;
}
// wait until the channel's done >= gen; returns ns since the matching cpu_publish (-1 on timeout)
int64_t cpu_wait(int64_t ch, uint64_t gen, double timeout_s) {
  Channel& c = channel(ch);
  const int64_t t0 = g_cpu_t0[ch];
  pybind11::gil_scoped_release nogil;
  while (__atomic_load_n(&c.ctl->done, __ATOMIC_ACQUIRE) < gen) {
    if (double(now_ns() - t0) > timeout_s * 1e9) return -1;
    __builtin_ia32_pause();
  }
  return now_ns() - t0;
}
int64_t cpu_request(int64_t ch, torch::Tensor ids, double timeout_s) { return cpu_wait(ch, cpu_publish(ch, ids), timeout_s); }

// ------------------------------------------------------------------------------------------- device side
__device__ __forceinline__ unsigned long long ld_acquire_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ int ld_relaxed_sys_s32(const int* p) {
  int v;
  asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys_u8(const uint8_t* p) {
  uint16_t v;
  asm volatile("ld.relaxed.sys.global.u8 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// two e4m3 (low byte = element 0) times an exact power-of-two fp32 scale -> packed bf16x2 (element 0 low).
// Identical instruction chain to kernels/engram_host.cu dequant2 (verified bit-exact against stock
// `_engram_lookup_kernel` over every e4m3 code x every ue8m0 byte, incl. NaN / inf / fp32-subnormal products):
// cvt.rn.f16x2.e4m3x2 -> cvt.f32.f16 (exact) -> mul.rn.f32 (no .ftz) -> cvt.rn.bf16x2.f32.
__device__ __forceinline__ uint32_t dequant2(uint32_t pair16, float scale) {
  uint32_t h2;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(pair16));
  float f0, f1;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }" : "=f"(f0), "=f"(f1) : "r"(h2));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f0) : "f"(scale));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f1) : "f"(scale));
  uint32_t b2;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b2) : "f"(f1), "f"(f0));  // d = {hi: f1, lo: f0}
  return b2;
}

// ids[k] = hash[t, head0 + h] for k = t * part + h (heads >= total -> -1, stock's padded heads), then publish.
__global__ void __launch_bounds__(1024) issue_kernel(const int* __restrict__ hash, long long st, long long sh, int head0,
                                                     int part, int total, int T, int* ids, Ctl* ctl, DevState* ds) {
  const int n = T * part;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
    const int t = k / part, h = k - t * part, head = head0 + h;
    ids[k] = head < total ? __ldg(hash + (long long)t * st + (long long)head * sh) : -1;
  }
  __threadfence_system();  // this thread's id stores are performed (host-visible) before the CTA arrives
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned prev = atomicAdd(&ds->arrive, 1u);
    if (prev == gridDim.x - 1) {  // last CTA: every id of the request is host-visible
      __threadfence();
      ds->arrive = 0;
      const unsigned long long g = ds->counter + 1;
      ds->counter = g;
      ds->gen = g;
      st_relaxed_sys(&ctl->req_n, (unsigned long long)n);
      __threadfence_system();
      st_release_sys(&ctl->req_gen, g);
    }
  }
}

// Wait for the poller's rows of the current generation, then out[k, :] = dequant(rows[src[k]]) (zeros for src < 0).
// 16 lanes per row (16 e4m3 bytes -> 16 bf16 each), U rows in flight per lane group, grid-stride over rows.
template <int U>
__global__ void __launch_bounds__(256) wait_dequant_kernel(Ctl* ctl, DevState* ds, const int* src, const uint8_t* rows,
                                                           __nv_bfloat16* out, int n, unsigned long long timeout_ns) {
  constexpr int LANES = 16;
  if (threadIdx.x == 0) {
    const unsigned long long g = ds->gen;
    if (blockIdx.x == 0 && g == ds->consumed) {  // no issue since the last wait (or issued on another stream): never
      printf("ds41h engram_nvme: wait without a new issue (generation %llu already consumed) -> trap\n", g);  // stale rows
      __trap();
    }
    unsigned long long d = ld_acquire_sys(&ctl->done);
    unsigned long long stall = 0;
    if (d < g) {
      const unsigned long long t0 = globaltimer();
      unsigned long long t = t0;
      while (d < g) {
        __nanosleep(64);
        d = ld_acquire_sys(&ctl->done);
        t = globaltimer();
        if (t - t0 > timeout_ns) {
          printf("ds41h engram_nvme: GPU timeout waiting for Engram rows (generation %llu, done %llu) -> trap\n", g, d);
          __trap();
        }
      }
      stall = t - t0;
    }
    if (blockIdx.x == 0) {  // single writer per channel (stream-ordered): device-side accumulation, posted mirror
      ds->consumed = g;
      ds->waits += 1;
      if (stall) {
        ds->spun += 1;
        ds->stall_ns += stall;
        if (stall > ds->stall_max_ns) ds->stall_max_ns = stall;
      }
      if (n <= kSmallRows) {
        ds->d_waits += 1;
        if (stall) {
          ds->d_spun += 1;
          ds->d_stall_ns += stall;
          if (stall > ds->d_stall_max_ns) ds->d_stall_max_ns = stall;
        }
        st_relaxed_sys(&ctl->d_waits, ds->d_waits);
        st_relaxed_sys(&ctl->d_spun, ds->d_spun);
        st_relaxed_sys(&ctl->d_stall_ns, ds->d_stall_ns);
        st_relaxed_sys(&ctl->d_stall_max_ns, ds->d_stall_max_ns);
      }
      st_relaxed_sys(&ctl->waits, ds->waits);
      st_relaxed_sys(&ctl->spun, ds->spun);
      st_relaxed_sys(&ctl->stall_ns, ds->stall_ns);
      st_relaxed_sys(&ctl->stall_max_ns, ds->stall_max_ns);
    }
  }
  __syncthreads();
  const int lane = threadIdx.x % LANES;
  const int grp = blockIdx.x * (blockDim.x / LANES) + threadIdx.x / LANES;
  const int ngrp = gridDim.x * (blockDim.x / LANES);
  for (int base = 0; base < n; base += ngrp * U) {
    int u_[U];
    uint4 v[U];
    uint32_t sb[U];
#pragma unroll
    for (int i = 0; i < U; ++i) {
      const int k = base + i * ngrp + grp;
      u_[i] = k < n ? ld_relaxed_sys_s32(src + k) : -1;
    }
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if (u_[i] >= 0) {
        const uint8_t* r = rows + (long long)u_[i] * kRecStride;
        v[i] = ld_relaxed_sys_v4(r + lane * 16);
        sb[i] = ld_relaxed_sys_u8(r + kDim + lane / 2);  // lane covers elements [16 lane, 16 lane + 16): block lane / 2
      } else {
        v[i] = make_uint4(0u, 0u, 0u, 0u);
        sb[i] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < U; ++i) {
      const int k = base + i * ngrp + grp;
      if (k >= n) continue;
      uint32_t o[8];
      if (u_[i] >= 0) {
        const float scale = __uint_as_float(sb[i] << 23);  // ue8m0 byte = the fp32 exponent field (0 -> 0, 255 -> inf)
        const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          o[2 * j] = dequant2(w[j] & 0xffffu, scale);
          o[2 * j + 1] = dequant2(w[j] >> 16, scale);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = 0u;
      }
      uint4* dst = reinterpret_cast<uint4*>(out + (long long)k * kDim + lane * 16);
      dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
      dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
  }
}

// ------------------------------------------------------------------------------------------- python API (launch)
void check_dstate(const torch::Tensor& ds) {
  TORCH_CHECK(ds.is_cuda() && ds.dtype() == torch::kInt64 && ds.numel() >= 16 && ds.is_contiguous(),
              "engram_nvme: dstate must be a contiguous int64 CUDA tensor of >= 16 elements");
}

// hash: int32 CUDA [T, ncols] (any strides; the stock [T, 2, 24] hash view of one layer)
void issue(int64_t ch, torch::Tensor hash, int64_t head0, int64_t part, int64_t total, torch::Tensor dstate) {
  Channel& c = channel(ch);
  TORCH_CHECK(c.cuda, "engram_nvme: CPU-only channel");
  TORCH_CHECK(hash.is_cuda() && hash.dtype() == torch::kInt32 && hash.dim() == 2, "engram_nvme: hash int32 CUDA [T, cols]");
  check_dstate(dstate);
  const int64_t T = hash.size(0), n = T * part;
  TORCH_CHECK(part >= 1 && head0 >= 0 && total >= 1 && total <= hash.size(1) && n <= c.max_rows, "engram_nvme: issue shape");
  if (n == 0) return;
  const int threads = 1024;
  const int blocks = int(std::min<int64_t>(16, (n + threads - 1) / threads));
  auto st = c10::cuda::getCurrentCUDAStream(hash.device().index()).stream();
  issue_kernel<<<blocks, threads, 0, st>>>(hash.data_ptr<int>(), hash.stride(0), hash.stride(1), int(head0), int(part),
                                           int(total), int(T), c.d_ids, c.d_ctl, reinterpret_cast<DevState*>(dstate.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out: bf16 CUDA, contiguous, >= n * 256 elements ([T, part, 256] staging rows); n = T * part
void wait_dequant(int64_t ch, torch::Tensor dstate, torch::Tensor out, int64_t n, int64_t timeout_ms) {
  Channel& c = channel(ch);
  TORCH_CHECK(c.cuda, "engram_nvme: CPU-only channel");
  check_dstate(dstate);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() >= n * kDim,
              "engram_nvme: out must be a contiguous bf16 CUDA tensor with >= n * 256 elements");
  TORCH_CHECK(n >= 0 && n <= c.max_rows, "engram_nvme: n");
  if (n == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream(out.device().index()).stream();
  const unsigned long long to = (unsigned long long)std::max<int64_t>(timeout_ms, 1) * 1000000ull;
  auto* ds = reinterpret_cast<DevState*>(dstate.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  constexpr int rows_per_cta = 256 / 16;
  if (n <= 4096) {
    const int blocks = int((n + rows_per_cta - 1) / rows_per_cta);
    wait_dequant_kernel<1><<<blocks, 256, 0, st>>>(c.d_ctl, ds, c.d_src, c.d_rows, o, int(n), to);
  } else {
    int sms = 132;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, out.device().index());
    const int blocks = int(std::min<int64_t>(2 * sms, (n + rows_per_cta * 4 - 1) / (rows_per_cta * 4)));
    wait_dequant_kernel<4><<<blocks, 256, 0, st>>>(c.d_ctl, ds, c.d_src, c.d_rows, o, int(n), to);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RECORD_STRIDE") = kRecStride;
  m.attr("PAYLOAD") = kPayload;
  m.def("engine_new", &engine_new,
        "engine (depth, mode 0 direct/1 buffered/2 auto, auto_max, spin_ns, idle_ns, stats_ns, rank, cache_bytes)");
  m.def("engine_add_channel", &engine_add_channel,
        "(engine, layer, path, data_off, rows, per_block, block, record, vstart, vend, max_rows, cuda, cache_bytes) -> channel");
  m.def("engine_start", &engine_start, "set up io_uring + start the poller thread; returns a config note");
  m.def("engine_stop", &engine_stop);
  m.def("channel_stats", &channel_stats);
  m.def("engine_set_hash", &engine_set_hash, "prefetch: stock hash config (before engine_start)");
  m.def("channel_set_cols", &channel_set_cols, "prefetch: (channel, layer index, head0, part)");
  m.def("engine_prefetch", &engine_prefetch, "prefetch job: (engine, int32 tokens [hist | new], hist) -> queue length");
  m.def("engine_pf_stats", &engine_pf_stats);
  m.def("channel_cache_count", &channel_cache_count);
  m.def("channel_reset_stats", &channel_reset_stats);
  m.def("channel_buffers", &channel_buffers, "(src, rows, ids) CPU views (tests)");
  m.def("cpu_request", &cpu_request, "CPU-driven request (tests / I/O bench): ids -> wait done; returns ns (-1 timeout)");
  m.def("cpu_publish", &cpu_publish, "CPU-driven request: ids -> publish; returns the generation");
  m.def("cpu_wait", &cpu_wait, "wait for a cpu_publish generation; returns ns since the publish (-1 timeout)");
  m.def("issue", &issue, "issue kernel on the current stream");
  m.def("wait_dequant", &wait_dequant, "wait + dequant kernel on the current stream");
}
