// kv_manager.cpp -- serve/kv_manager.py OwnedKVManager in C++, line for line. See kv_manager.h.
#include "kv_manager.h"

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>

namespace fpr {
namespace kvimpl {

// ------------------------------------------------------------------------------------------------ FreeList / Lru
int64_t FreeList::heap_pop() {
  std::pop_heap(heap.begin(), heap.end(), std::greater<int64_t>());
  int64_t p = heap.back();
  heap.pop_back();
  return p;
}
void FreeList::heap_push(int64_t p) {
  heap.push_back(p);
  std::push_heap(heap.begin(), heap.end(), std::greater<int64_t>());
}
int64_t FreeList::popleft() {
  if (mode == 1) return heap_pop();
  if (mode == 2 && !heap.empty()) return heap_pop();
  int64_t p = dq.front();
  dq.pop_front();
  return p;
}
void FreeList::append(int64_t p) {
  if (mode == 1 || (mode == 2 && p < n_hbm)) heap_push(p);
  else dq.push_back(p);
}

void Lru::push_back(int64_t p) {
  if (in[(size_t)p]) return;
  in[(size_t)p] = 1;
  prv[(size_t)p] = tail;
  nxt[(size_t)p] = -1;
  if (tail >= 0) nxt[(size_t)tail] = p; else head = p;
  tail = p;
  ++n;
}
void Lru::erase(int64_t p) {
  if (!in[(size_t)p]) return;
  int64_t a = prv[(size_t)p], b = nxt[(size_t)p];
  if (a >= 0) nxt[(size_t)a] = b; else head = b;
  if (b >= 0) prv[(size_t)b] = a; else tail = a;
  in[(size_t)p] = 0;
  prv[(size_t)p] = nxt[(size_t)p] = -1;
  --n;
}
int64_t Lru::pop_front() {
  int64_t p = head;
  erase(p);
  return p;
}

// ------------------------------------------------------------------------------------------------ PagePool
PagePool::PagePool(const PoolSpec& s, int r, const KVGeometry& g) : spec(s), rank(r) {
  const int64_t n = s.num_pages;
  ref.assign((size_t)n, 0);
  hash_of.assign((size_t)n, Hash{});
  has_hash.assign((size_t)n, 0);
  low_first = s.kind == "state";
  n_hbm = s.kind == "paged" ? g.hbm_pages : 0;
  if (n_hbm) {
    free_uncached.mode = 2;
    free_uncached.n_hbm = n_hbm;
    const int64_t m = std::min(n_hbm, n);
    for (int64_t i = 0; i < m; ++i) free_uncached.heap.push_back(i);       // sorted = a valid heap
    for (int64_t i = std::max<int64_t>(m, 0); i < n; ++i) free_uncached.dq.push_back(i);
  } else if (low_first) {
    free_uncached.mode = 1;
    for (int64_t i = 0; i < n; ++i) free_uncached.heap.push_back(i);
  } else {
    free_uncached.mode = 0;
    for (int64_t i = 0; i < n; ++i) free_uncached.dq.push_back(i);
  }
  lru.init(n);
  poison_on = g.poison_mode == "1" || (g.poison_mode == "pages" && s.kind != "state");
}

bool PagePool::alloc(int64_t n, std::vector<int64_t>* out) {
  out->clear();
  if (n > num_free()) return false;
  while ((int64_t)out->size() < n && !free_uncached.empty()) out->push_back(free_uncached.popleft());
  while ((int64_t)out->size() < n) {
    int64_t p = lru.pop_front();
    if (caches_page(p)) cached.erase(hash_of[(size_t)p]);
    has_hash[(size_t)p] = 0;
    if (poison_on) poison.push_back(p);       // an evicted prefix-cache page changes owner
    out->push_back(p);
  }
  for (int64_t p : *out) ref[(size_t)p] = 1;  // (Python asserts ref == 0)
  return true;
}

void PagePool::touch(int64_t p) {
  if (ref[(size_t)p] == 0) lru.erase(p);
  ref[(size_t)p] += 1;
}

void PagePool::release(int64_t p) {
  ref[(size_t)p] -= 1;
  if (ref[(size_t)p] == 0) {
    if (caches_page(p)) {
      lru.push_back(p);                          // to the back: most recently freed = evicted last
      if (p < n_hbm) demote_q.push_back(p);      // tiered pool: copy to a host page at the next step (tier_copies)
    } else {
      has_hash[(size_t)p] = 0;
      if (poison_on) poison.push_back(p);
      free_uncached.append(p);
    }
  }
}

int64_t PagePool::host_dst() {
  int64_t q = free_uncached.pop_host();
  if (q < 0) {
    for (int64_t c = lru.head; c >= 0; c = lru.nxt[(size_t)c])
      if (c >= n_hbm) { q = c; break; }
    if (q < 0) return -1;
    lru.erase(q);
    if (caches_page(q)) cached.erase(hash_of[(size_t)q]);
    has_hash[(size_t)q] = 0;
    if (poison_on) poison.push_back(q);
  }
  return q;
}

void PagePool::move_hash(int64_t src, int64_t dst) {
  Hash h = hash_of[(size_t)src];
  has_hash[(size_t)src] = 0;
  hash_of[(size_t)dst] = h;
  has_hash[(size_t)dst] = 1;
  cached[h] = dst;
}

bool PagePool::register_(int64_t p, const Hash& h) {
  int64_t q;
  if (cached_get(h, &q)) return q == p;
  cached[h] = p;
  hash_of[(size_t)p] = h;
  has_hash[(size_t)p] = 1;
  return true;
}

// ------------------------------------------------------------------------------------------------ OwnedKVManager
OwnedKVManager::OwnedKVManager(const KVGeometry& g) : g_(g) {
  for (int r = 0; r < g.n_ranks; ++r) {
    paged_.emplace_back(g.paged, r, g);
    window_.emplace_back(g.window_pool, r, g);
    state_.emplace_back(g.state, r, g);
  }
}

std::vector<int64_t> OwnedKVManager::lookup(const std::vector<Hash>& block_hashes) const {
  std::vector<int64_t> out;
  for (const auto& pool : paged_) {
    int64_t n = 0;
    for (const auto& h : block_hashes) {
      if (!pool.cached.count(h)) break;
      ++n;
    }
    out.push_back(n);
  }
  return out;
}

int64_t OwnedKVManager::free_pages(int rank) const {
  const auto& p = paged_[(size_t)rank];
  return p.num_free() - p.reserved;
}

KVUsage OwnedKVManager::usage(int rank) const {
  const auto& p = paged_[(size_t)rank];
  return KVUsage{p.num_used(), (int64_t)p.lru.n, (int64_t)p.free_uncached.size(), p.reserved, p.num_pages(),
                 window_[(size_t)rank].num_used(), state_[(size_t)rank].num_used()};
}

static const std::vector<int64_t> kEmptyPages;
static const WindowTable kEmptyWindow;

const std::vector<int64_t>& OwnedKVManager::block_table(const Rid& id) const {
  const SeqKV* sk = seq(id);
  if (!sk) { fail("block_table: unknown " + id); return kEmptyPages; }
  return *sk->pages;
}
const WindowTable& OwnedKVManager::window_table(const Rid& id) const {
  const SeqKV* sk = seq(id);
  if (!sk) { fail("window_table: unknown " + id); return kEmptyWindow; }
  return *sk->window;
}
int32_t OwnedKVManager::state_slot(const Rid& id) const {
  const SeqKV* sk = seq(id);
  return sk ? sk->state_slot : -1;
}

void OwnedKVManager::hit_after_cap(int64_t hit_blocks, int64_t prompt_len, int64_t* hit, int64_t* cached,
                                   int64_t* replay_start) const {
  const int64_t bt = g_.page_tokens;
  const int64_t h = std::min(hit_blocks, std::max<int64_t>(0, fdiv(prompt_len - 1, bt)));
  const int64_t c = h * bt;
  *hit = h;
  *cached = c;
  *replay_start = c - std::min(g_.replay_tokens, c);
}

Status OwnedKVManager::admit(const Rid& id, int rank, const std::vector<Hash>& block_hashes, int64_t prompt_len,
                             int64_t step, int copy_from, AdmitResult* out) {
  if (seqs_.count(id)) { fail("admit: already admitted " + id); return Status::BAD_ARG; }
  const std::vector<int64_t> hits = lookup(block_hashes);
  const int64_t local = hits[(size_t)rank];
  const int src = (copy_from >= 0 && hits[(size_t)copy_from] > local) ? copy_from : -1;
  const int64_t want = src >= 0 ? hits[(size_t)src] : local;
  int64_t keep, cached, replay_start;
  hit_after_cap(want, prompt_len, &keep, &cached, &replay_start);
  const int64_t n_local = std::min(local, keep);
  const int64_t n_copy = keep - n_local;
  const int64_t need_total = pages_for(prompt_len) - n_local;    // pages to allocate: copies + the rest of the prompt
  PagePool& pool = paged_[(size_t)rank];
  const int64_t wm = (int64_t)(g_.reserve_watermark * (double)pool.num_pages());
  // the request's own prefix-hit pages in the LRU count as free in num_free() but touching them takes them out of it
  int64_t lru_hits = 0;
  for (int64_t i = 0; i < n_local; ++i)
    if (pool.ref[(size_t)pool.cached.at(block_hashes[(size_t)i])] == 0) ++lru_hits;
  if (need_total + wm > free_pages(rank) - lru_hits || !state_[(size_t)rank].num_free()) return Status::NO_ROOM;
  auto sk = std::make_shared<SeqKV>();
  sk->seq_id = id;
  sk->rank = rank;
  sk->inc = ++inc_;
  std::vector<CopyOp> copies;
  int64_t promote = 0;
  if (pool.n_hbm) {   // tiered pool: the hit's host pages move to HBM pages if the whole prompt fits there
    int64_t n_host_hits = 0;
    for (int64_t i = 0; i < n_local; ++i)
      if (pool.cached.at(block_hashes[(size_t)i]) >= pool.n_hbm) ++n_host_hits;
    if (n_host_hits && (int64_t)pool.free_uncached.heap.size() >= n_host_hits + need_total &&
        need_total + wm + n_host_hits <= free_pages(rank) - lru_hits)
      promote = n_host_hits;
  }
  for (int64_t i = 0; i < n_local; ++i) {
    const int64_t p = pool.cached.at(block_hashes[(size_t)i]);
    if (promote && p >= pool.n_hbm) {
      const int64_t d = pool.free_uncached.pop_hbm();
      pool.ref[(size_t)d] = 1;                           // the sequence's reference
      pool.touch(p);                                     // the copy's source: pinned until release_copies
      pinned_[step].push_back(Pin{POOL_PAGED, rank, p});
      copies.push_back(copy(POOL_PAGED, rank, p, rank, d, step));
      pool.move_hash(p, d);
      sk->pages->push_back(d);
      stats_.promoted += 1;
    } else {
      pool.touch(p);
      sk->pages->push_back(p);
    }
  }
  sk->num_registered = n_local;
  if (n_copy) {
    std::vector<int64_t> dst;
    pool.alloc(n_copy, &dst);
    PagePool& spool = paged_[(size_t)src];
    for (size_t i = 0; i < dst.size(); ++i) {
      const Hash& h = block_hashes[(size_t)n_local + i];
      const int64_t s = spool.cached.at(h);
      spool.touch(s);                                    // pinned until release_copies(step)
      pinned_[step].push_back(Pin{POOL_PAGED, src, s});
      copies.push_back(copy(POOL_PAGED, src, s, rank, dst[i], step));
      pool.register_(dst[i], h);
      sk->pages->push_back(dst[i]);
    }
    sk->num_registered = keep;
    stats_.copies += n_copy;
    stats_.copy_bytes += n_copy * pool.spec.bytes_per_page;
  }
  std::vector<int64_t> st;
  state_[(size_t)rank].alloc(1, &st);
  sk->state_slot = (int32_t)st[0];
  sk->reserved = need_total - n_copy;
  sk->cached_tokens = cached;
  sk->replay_start = replay_start;
  pool.reserved += sk->reserved;
  seqs_[id] = sk;
  *out = AdmitResult{rank, keep, cached, replay_start, n_copy, std::move(copies)};
  return Status::OK;
}

void OwnedKVManager::window_need(const SeqKV& sk, int64_t start, int64_t end, int64_t* lo, int64_t* hi) const {
  const int64_t wt = g_.window_page_tokens;
  *lo = fdiv(std::max(sk.replay_start, start - (g_.window - 1)), wt);
  *hi = fdiv(end - 1, wt);
}

bool OwnedKVManager::can_allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) const {
  const SeqKV* sk = seq(id);
  if (!sk) return fail("can_allocate: unknown " + id);
  const PagePool& pp = paged_[(size_t)sk->rank];
  const int64_t need = std::max<int64_t>(0, pages_for(end) - (int64_t)sk->pages->size());
  const int64_t avail = pp.num_free() - pp.reserved + std::min(sk->reserved, need);
  int64_t lo, hi;
  window_need(*sk, start, window_end, &lo, &hi);
  int64_t wneed = 0;
  for (int64_t b = lo; b <= hi; ++b)
    if (!sk->window->contains(b)) ++wneed;
  return need <= avail && wneed <= window_[(size_t)sk->rank].num_free();
}

bool OwnedKVManager::allocate(const Rid& id, int64_t start, int64_t end, int64_t window_end) {
  SeqKV* sk = seq(id);
  if (!sk) return fail("allocate: unknown " + id);
  const int64_t wend = window_end;
  // fast path: pages and window blocks already there (blocks are allocated in order and released from the bottom)
  if (-fdiv(-end, g_.page_tokens) <= (int64_t)sk->pages->size()) {
    const int64_t wt = g_.window_page_tokens;
    const int64_t hi = fdiv(wend - 1, wt);
    const int64_t lo = fdiv(std::max(sk->replay_start, start - (g_.window - 1)), wt);
    if (sk->window->contains(hi) && sk->window->contains(lo)) return true;
  }
  if (!can_allocate(id, start, end, wend)) return false;
  PagePool& pool = paged_[(size_t)sk->rank];
  const int64_t need = std::max<int64_t>(0, pages_for(end) - (int64_t)sk->pages->size());
  if (need) {
    const int64_t use_res = std::min(sk->reserved, need);
    sk->reserved -= use_res;
    pool.reserved -= use_res;
    const int64_t evict = std::max<int64_t>(0, need - (int64_t)pool.free_uncached.size());
    std::vector<int64_t> pages;
    pool.alloc(need, &pages);
    stats_.evictions += evict;
    sk->pages->insert(sk->pages->end(), pages.begin(), pages.end());
  }
  PagePool& wpool = window_[(size_t)sk->rank];
  int64_t lo, hi;
  window_need(*sk, start, wend, &lo, &hi);
  std::vector<int64_t> missing;
  for (int64_t b = lo; b <= hi; ++b)
    if (!sk->window->contains(b)) missing.push_back(b);
  if (!missing.empty()) {
    std::vector<int64_t> wp;
    wpool.alloc((int64_t)missing.size(), &wp);
    for (size_t i = 0; i < missing.size() && i < wp.size(); ++i) sk->window->set(missing[i], wp[i]);
    sk->wver += 1;
  }
  return true;
}

void OwnedKVManager::commit(const Rid& id, int64_t num_computed, const std::vector<Hash>& block_hashes) {
  SeqKV* sk = seq(id);
  if (!sk) { fail("commit: unknown " + id); return; }
  PagePool& pool = paged_[(size_t)sk->rank];
  const int64_t full = std::min(fdiv(num_computed, g_.page_tokens), (int64_t)sk->pages->size());
  while (sk->num_registered < full) {
    if (sk->num_registered >= (int64_t)block_hashes.size()) { fail("commit: block hash missing"); return; }
    pool.register_((*sk->pages)[(size_t)sk->num_registered], block_hashes[(size_t)sk->num_registered]);
    sk->num_registered += 1;
  }
  int64_t lo, hi;
  window_need(*sk, num_computed, num_computed + 1, &lo, &hi);
  WindowTable& w = *sk->window;
  if (!w.empty() && w.min_key() < lo) {
    std::vector<int64_t> drop;
    for (auto& kv : w.items) if (kv.first < lo) drop.push_back(kv.first);
    for (int64_t b : drop) window_[(size_t)sk->rank].release(w.pop(b));
    sk->wver += 1;
  }
}

void OwnedKVManager::shrink_reservation(const Rid& id) {
  SeqKV* sk = seq(id);
  if (!sk) { fail("shrink_reservation: unknown " + id); return; }
  paged_[(size_t)sk->rank].reserved -= sk->reserved;
  sk->reserved = 0;
}

bool OwnedKVManager::open_shadow(const Rid& key, int rank, int64_t replay_start) {
  if (seqs_.count(key)) return fail("open_shadow: exists " + key);
  if (!state_[(size_t)rank].num_free()) return false;
  auto sk = std::make_shared<SeqKV>();
  sk->seq_id = key;
  sk->rank = rank;
  sk->inc = ++inc_;
  std::vector<int64_t> st;
  state_[(size_t)rank].alloc(1, &st);
  sk->state_slot = (int32_t)st[0];
  sk->replay_start = replay_start;
  sk->shadow = true;
  seqs_[key] = sk;
  return true;
}

void OwnedKVManager::trim_window(const Rid& id, int64_t lo_pos) {
  SeqKV* sk = seq(id);
  if (!sk) { fail("trim_window: unknown " + id); return; }
  const int64_t wt = g_.window_page_tokens;
  WindowTable& w = *sk->window;
  std::vector<int64_t> drop;
  for (auto& kv : w.items) if ((kv.first + 1) * wt <= lo_pos) drop.push_back(kv.first);
  for (int64_t b : drop) window_[(size_t)sk->rank].release(w.pop(b));
  if (!drop.empty()) sk->wver += 1;
}

void OwnedKVManager::free_now(const Rid& id) {
  auto it = seqs_.find(id);
  if (it == seqs_.end()) { fail("free: unknown " + id); return; }
  std::shared_ptr<SeqKV> sk = it->second;
  seqs_.erase(it);
  release_seq(*sk);
}

void OwnedKVManager::free_deferred(const Rid& id, int64_t defer_until) {
  auto it = seqs_.find(id);
  if (it == seqs_.end()) { fail("free: unknown " + id); return; }
  std::shared_ptr<SeqKV> sk = it->second;
  seqs_.erase(it);
  for (auto& e : deferred_)
    if (e.first == defer_until) { e.second.push_back(sk); return; }
  deferred_.emplace_back(defer_until, std::vector<std::shared_ptr<SeqKV>>{sk});
}

void OwnedKVManager::flush_deferred(int64_t upto_step) {
  std::vector<int64_t> steps;
  for (auto& e : deferred_) if (e.first <= upto_step) steps.push_back(e.first);
  for (int64_t s : steps) {
    std::vector<std::shared_ptr<SeqKV>> lst;
    for (size_t i = 0; i < deferred_.size(); ++i)
      if (deferred_[i].first == s) { lst = std::move(deferred_[i].second); deferred_.erase(deferred_.begin() + (long)i); break; }
    for (auto& sk : lst) release_seq(*sk);
  }
}

void OwnedKVManager::release_seq(SeqKV& sk) {
  PagePool& pool = paged_[(size_t)sk.rank];
  pool.reserved -= sk.reserved;
  for (auto it = sk.pages->rbegin(); it != sk.pages->rend(); ++it) pool.release(*it);
  for (auto& kv : sk.window->items) window_[(size_t)sk.rank].release(kv.second);
  if (sk.state_slot >= 0) state_[(size_t)sk.rank].release(sk.state_slot);
}

Status OwnedKVManager::migrate(const Rid& id, int dst, int64_t step, int64_t num_computed, std::vector<CopyOp>* copies) {
  copies->clear();
  SeqKV* sk = seq(id);
  if (!sk) { fail("migrate: unknown " + id); return Status::BAD_ARG; }
  if (dst == sk->rank) return Status::OK;
  const int src = sk->rank;
  const int64_t n_pages = (int64_t)sk->pages->size();
  PagePool& dpool = paged_[(size_t)dst];
  PagePool& wpool = window_[(size_t)dst];
  const int64_t wm = (int64_t)(g_.reserve_watermark * (double)dpool.num_pages());
  if (n_pages + sk->reserved + wm > free_pages(dst) || (int64_t)sk->window->size() > wpool.num_free() ||
      !state_[(size_t)dst].num_free())
    return Status::NO_ROOM;
  std::vector<CopyOp>& ops = *copies;
  auto new_pages = std::make_shared<std::vector<int64_t>>();
  dpool.alloc(n_pages, new_pages.get());
  // pages holding KV: the computed positions AND the whole cached prefix (see the Python comment)
  const int64_t valid = std::max({num_computed, sk->cached_tokens, sk->num_registered * g_.page_tokens});
  const int64_t n_full = std::min(fdiv(valid, g_.page_tokens), n_pages);
  PagePool& spool = paged_[(size_t)src];
  for (int64_t i = 0; i < n_pages; ++i) {
    const int64_t s = (*sk->pages)[(size_t)i], d = (*new_pages)[(size_t)i];
    if (i < n_full || i * g_.page_tokens < valid) ops.push_back(copy(POOL_PAGED, src, s, dst, d, step));
    if (i < sk->num_registered && spool.has_hash[(size_t)s]) dpool.register_(d, spool.hash_of[(size_t)s]);
  }
  auto new_win = std::make_shared<WindowTable>();
  std::vector<std::pair<int64_t, int64_t>> sorted_w = sk->window->items;
  std::sort(sorted_w.begin(), sorted_w.end());
  std::vector<int64_t> wp;
  wpool.alloc((int64_t)sk->window->size(), &wp);
  for (size_t i = 0; i < sorted_w.size() && i < wp.size(); ++i) {
    ops.push_back(copy(POOL_WINDOW, src, sorted_w[i].second, dst, wp[i], step));
    new_win->set(sorted_w[i].first, wp[i]);
  }
  std::vector<int64_t> st;
  state_[(size_t)dst].alloc(1, &st);
  const int64_t new_state = st[0];
  ops.push_back(copy(POOL_STATE, src, sk->state_slot, dst, new_state, step));
  // source pages: keep our reference until the copy ran (pinned), then release
  auto& pin = pinned_[step];
  for (int64_t s : *sk->pages) pin.push_back(Pin{POOL_PAGED, src, s});
  for (auto& kv : sk->window->items) pin.push_back(Pin{POOL_WINDOW, src, kv.second});
  pin.push_back(Pin{POOL_STATE, src, sk->state_slot});
  spool.reserved -= sk->reserved;
  dpool.reserved += sk->reserved;
  sk->rank = dst;
  sk->pages = new_pages;
  sk->window = new_win;
  sk->state_slot = (int32_t)new_state;
  sk->wver += 1;
  sk->inc = ++inc_;
  stats_.copies += (int64_t)ops.size();
  for (auto& o : ops) stats_.copy_bytes += P(o.pool, dst).spec.bytes_per_page;
  return Status::OK;
}

std::vector<CopyOp> OwnedKVManager::tier_copies(int64_t step, int64_t max_pages) {
  std::vector<CopyOp> ops;
  for (int r = 0; r < (int)paged_.size(); ++r) {
    PagePool& pool = paged_[(size_t)r];
    if (!pool.n_hbm || pool.demote_q.empty()) continue;
    std::vector<int64_t> q;
    q.swap(pool.demote_q);
    for (size_t i = 0; i < q.size(); ++i) {
      const int64_t p = q[i];
      if ((int64_t)ops.size() >= max_pages) {
        pool.demote_q.insert(pool.demote_q.end(), q.begin() + (long)i, q.end());
        break;
      }
      if (pool.ref[(size_t)p] != 0 || !pool.lru.contains(p) || !pool.has_hash[(size_t)p]) continue;   // re-used / evicted
      const int64_t d = pool.host_dst();
      if (d < 0) break;
      pool.touch(p);                                     // out of the LRU, pinned until release_copies
      pinned_[step].push_back(Pin{POOL_PAGED, r, p});
      ops.push_back(copy(POOL_PAGED, r, p, r, d, step));
      pool.move_hash(p, d);
      pool.lru.push_back(d);                             // the host copy is the cached-free page now
    }
    stats_.demoted += (int64_t)ops.size();             // (sic: Python adds the running total once per tiered rank)
  }
  return ops;
}

void OwnedKVManager::release_copies(int64_t step) {
  auto it = pinned_.find(step);
  if (it != pinned_.end()) {
    std::vector<Pin> pins = std::move(it->second);
    pinned_.erase(it);
    for (auto& pn : pins) P(pn.pool, pn.rank).release(pn.page);
  }
  pending_.erase(step);
}

CopyOp OwnedKVManager::copy(int32_t pool, int32_t src, int64_t s, int32_t dst, int64_t d, int64_t step) {
  auto& pend = pending_[step];
  for (;;) {
    auto it = pend.find(std::make_tuple(pool, src, s));
    if (it == pend.end()) break;
    src = it->second.first;
    s = it->second.second;
  }
  pend[std::make_tuple(pool, dst, d)] = std::make_pair(src, s);
  return CopyOp{pool, src, s, dst, d, step};
}

std::vector<std::pair<int32_t, int64_t>> OwnedKVManager::take_poison(int rank) {
  std::vector<std::pair<int32_t, int64_t>> out;
  for (int32_t pool : {POOL_PAGED, POOL_WINDOW, POOL_STATE}) {
    PagePool& pl = P(pool, rank);
    for (int64_t p : pl.poison) out.emplace_back(pool, p);
    pl.poison.clear();
  }
  return out;
}

Status OwnedKVManager::check() const {
  std::vector<const SeqKV*> all;
  for (auto& kv : seqs_) all.push_back(kv.second.get());
  for (auto& e : deferred_) for (auto& sk : e.second) all.push_back(sk.get());
  for (int32_t name : {POOL_PAGED, POOL_WINDOW, POOL_STATE}) {
    for (int r = 0; r < g_.n_ranks; ++r) {
      const PagePool& pool = P(name, r);
      const int64_t N = pool.num_pages();
      std::vector<int64_t> refs((size_t)N, 0);
      auto bump = [&](int64_t p) { if (p >= 0 && p < N) refs[(size_t)p] += 1; };
      for (const SeqKV* sk : all) {
        if (sk->rank != r) continue;
        if (name == POOL_PAGED) for (int64_t p : *sk->pages) bump(p);
        else if (name == POOL_WINDOW) for (auto& kv : sk->window->items) bump(kv.second);
        else if (sk->state_slot >= 0) bump(sk->state_slot);
      }
      for (auto& e : pinned_) for (auto& pn : e.second) if (pn.pool == name && pn.rank == r) bump(pn.page);
      std::ostringstream os;
      for (int64_t p = 0; p < N; ++p)
        if (refs[(size_t)p] != pool.ref[(size_t)p]) {
          os << "check: refs " << pool_name(name) << " rank " << r << " page " << p << " seqs+pins " << refs[(size_t)p]
             << " ref " << pool.ref[(size_t)p];
          fail(os.str());
          return Status::INTERNAL;
        }
      std::vector<char> in_free((size_t)N, 0);
      size_t nf = 0;
      auto mark = [&](int64_t p) { if (p >= 0 && p < N) { if (in_free[(size_t)p]) nf = (size_t)-1; else { in_free[(size_t)p] = 1; if (nf != (size_t)-1) ++nf; } } };
      for (int64_t p : pool.free_uncached.heap) mark(p);
      for (int64_t p : pool.free_uncached.dq) mark(p);
      if (nf != pool.free_uncached.size()) { fail("check: free list has duplicates"); return Status::INTERNAL; }
      for (int64_t p = pool.lru.head; p >= 0; p = pool.lru.nxt[(size_t)p])
        if (in_free[(size_t)p]) { fail("check: free and lru intersect"); return Status::INTERNAL; }
      for (int64_t p = 0; p < N; ++p) {
        const bool f = in_free[(size_t)p] || pool.lru.contains(p);
        if (f != (pool.ref[(size_t)p] == 0)) {
          os << "check: free state " << pool_name(name) << " rank " << r << " page " << p;
          fail(os.str());
          return Status::INTERNAL;
        }
      }
      for (int64_t p = pool.lru.head; p >= 0; p = pool.lru.nxt[(size_t)p])
        if (!pool.caches_page(p)) { fail("check: lru page without its cache entry"); return Status::INTERNAL; }
      for (auto& kv : pool.cached)
        if (!pool.has_hash[(size_t)kv.second] || pool.hash_of[(size_t)kv.second] != kv.first) {
          fail("check: cache entry names a page without that hash");
          return Status::INTERNAL;
        }
      if (name == POOL_PAGED) {
        int64_t res = 0;
        for (const SeqKV* sk : all) if (sk->rank == r) res += sk->reserved;
        if (pool.reserved != res) { fail("check: reservations do not add up"); return Status::INTERNAL; }
        if (pool.reserved < 0) { fail("check: negative reservation"); return Status::INTERNAL; }
        int64_t pins = 0;
        for (auto& e : pinned_) for (auto& pn : e.second) if (pn.pool == POOL_PAGED && pn.rank == r) ++pins;
        if (pool.reserved > pool.num_free() + pins) { fail("check: reservation exceeds free + pins"); return Status::INTERNAL; }
      }
    }
  }
  return Status::OK;
}

void OwnedKVManager::heads(int32_t pool, int rank, int k, Heads* out) const {
  const PagePool& p = P(pool, rank);
  *out = Heads{};
  if (p.free_uncached.mode == 0) {
    for (size_t i = 0; i < p.free_uncached.dq.size() && (int)i < k; ++i) out->free.push_back(p.free_uncached.dq[i]);
  } else {
    std::vector<int64_t> h = p.free_uncached.heap;
    std::sort(h.begin(), h.end());
    for (size_t i = 0; i < h.size() && (int)i < k; ++i) out->free.push_back(h[i]);
    if (p.free_uncached.mode == 2)
      for (size_t i = 0; i < p.free_uncached.dq.size() && (int)i < k; ++i) out->host.push_back(p.free_uncached.dq[i]);
  }
  int c = 0;
  for (int64_t q = p.lru.head; q >= 0 && c < k; q = p.lru.nxt[(size_t)q], ++c) out->lru.push_back(q);
  out->nfree = (int64_t)p.free_uncached.size();
  out->nlru = (int64_t)p.lru.n;
  out->ncached = (int64_t)p.cached.size();
  out->reserved = p.reserved;
  out->demote = (int64_t)p.demote_q.size();
  out->npoison = (int64_t)p.poison.size();
}

size_t OwnedKVManager::num_deferred_seqs() const {
  size_t n = 0;
  for (auto& e : deferred_) n += e.second.size();
  return n;
}

}  // namespace kvimpl

KVGeometry kv_geometry(const Geometry& g, int64_t hbm_pages, const std::string& poison_mode) {
  KVGeometry k;
  k.n_ranks = (int32_t)g.n_ranks;
  k.page_tokens = g.page_tokens; k.window = g.window; k.window_page_tokens = g.window_page_tokens;
  k.replay_tokens = g.replay_tokens; k.reserve_watermark = g.reserve_watermark;
  std::vector<PoolSpec> p = g.pools();
  for (auto& s : p) { if (s.name == "paged") k.paged = s; else if (s.name == "window") k.window_pool = s; else k.state = s; }
  k.hbm_pages = hbm_pages; k.poison_mode = poison_mode;
  return k;
}

std::unique_ptr<IKVManager> make_kv_manager(const KVGeometry& g) { return std::make_unique<kvimpl::OwnedKVManager>(g); }

}  // namespace fpr
