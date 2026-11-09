"""Sequence-owned KV manager for the ds41h-phys serving engine.

Every sequence lives on exactly one rank (its owner). Its KV is stored once, on that rank only:
  * "paged" pool  -- compressed KV of the 4 source layers + their index keys, one page per 64 tokens, in the packed
                     fp8_ds_mla page layout FlashMLA / the indexer already read (data rows then scale rows). Pages are
                     prefix-cacheable: a full page carries its block hash (vLLM's chained block hash) and survives the
                     sequence that wrote it, ref-counted, in a per-rank LRU of free cached pages.
  * "window" pool -- the 128-token sliding window of all 40 layers (one page id = 32 tokens in each of the 40 per-layer
                     tensors). Pages below the window are released as the sequence advances. Not prefix-cached.
  * "state" pool  -- one fixed slot per sequence (compressor running state). Not prefix-cached.

Prefix caching is global over ranks: lookup() returns, per rank, how many leading blocks of a request are cached there.
A request can then be placed on any rank; if the chosen rank holds fewer blocks than another one, the missing pages are
COPIED rank-to-rank (NVLink copy engine, ~1.8 KB per token: a 140K-token prefix is ~250 MB, ~2 ms) instead of being
recomputed (~2-3 s of prefill). Copies are returned to the caller as CopyOp lists that the step program executes before
the step's kernels; source pages stay pinned until release_copies(step) is called.

After a hit the window and compressor state of the cached prefix do not exist on the owner rank. As in production
(vLLM 0.32 "SWA bounded replay", Scheduler._mark_prefix_replay / DeepseekV41ModelState), every hit page is referenced, and
the last `replay_tokens` (128) tokens of the hit are re-run with `replay_start` = hit - 128: their paged slots are padded
(-1, the cached KV stays as is), their window KV and compressor state are written, and no window KV below replay_start
is read.

Everything is stdlib Python and deterministic; check() verifies every invariant (used by the tests after every op).
"""
from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

from geometry import Geometry, PoolSpec


@dataclass(frozen=True)
class CopyOp:
    """Copy one page of `pool` from (src_rank, src_page) to (dst_rank, dst_page) before step `step` runs."""
    pool: str
    src_rank: int
    src_page: int
    dst_rank: int
    dst_page: int
    step: int


class PagePool:
    """Pages of one pool kind on one rank: ref counts, hashes, free list, LRU of cached free pages."""

    def __init__(self, spec: PoolSpec, rank: int):
        self.spec, self.rank = spec, rank
        n = spec.num_pages
        self.ref = [0] * n
        self.hash_of: list[Optional[Hashable]] = [None] * n
        # state slots are handed out lowest-first (a heap): indices stay <= live sequences + in-flight copy pins, so
        # engine-side tables keyed by the slot stay small (the engine's, HSD's); the other pools stay FIFO / LRU
        self.low_first = spec.kind == "state"
        self.free_uncached = list(range(n)) if self.low_first else deque(range(n))
        self.lru: OrderedDict[int, None] = OrderedDict()   # ref == 0 and hashed; evict from the front
        self.cached: dict[Hashable, int] = {}              # block hash -> page (at most one page per hash per rank)
        self.reserved = 0                                   # pages promised to admitted prefills (see OwnedKVManager)
        # debug ($PHYS_POISON_ON_FREE=1: all pools, =pages: paged + window): pages that leave an owner for the free list,
        # and prefix-cache pages evicted for a new owner, are queued here; the worker fills them with 0xFF before the
        # next step's kernels (a read of a previous owner's leftovers becomes NaN; prefix-cache pages are left intact)
        import os as _os
        mode = _os.environ.get("PHYS_POISON_ON_FREE", "")
        self.poison_on = mode == "1" or (mode == "pages" and spec.kind != "state")
        self.poison: list[int] = []

    # -- capacity
    @property
    def num_pages(self) -> int:
        return self.spec.num_pages

    def num_free(self) -> int:
        """Pages allocatable now (never-hashed free pages + evictable cached pages)."""
        return len(self.free_uncached) + len(self.lru)

    def num_used(self) -> int:
        return self.num_pages - self.num_free()

    # -- allocation
    def alloc(self, n: int) -> Optional[list[int]]:
        """n pages (ref 1 each) or None (nothing changes) if fewer than n are allocatable. Uncached free pages first,
        then cached pages in LRU order (their hash entries are dropped)."""
        if n > self.num_free():
            return None
        out = []
        while len(out) < n and self.free_uncached:
            out.append(heapq.heappop(self.free_uncached) if self.low_first else self.free_uncached.popleft())
        while len(out) < n:
            p, _ = self.lru.popitem(last=False)
            h = self.hash_of[p]
            if h is not None and self.cached.get(h) == p:
                del self.cached[h]
            self.hash_of[p] = None
            if self.poison_on:
                self.poison.append(p)                 # an evicted prefix-cache page changes owner
            out.append(p)
        for p in out:
            assert self.ref[p] == 0
            self.ref[p] = 1
        return out

    def touch(self, p: int) -> None:
        """Take a reference on an existing (cached) page."""
        if self.ref[p] == 0:
            self.lru.pop(p)
        self.ref[p] += 1

    def release(self, p: int) -> None:
        assert self.ref[p] > 0, (self.spec.name, self.rank, p)
        self.ref[p] -= 1
        if self.ref[p] == 0:
            if self.hash_of[p] is not None and self.cached.get(self.hash_of[p]) == p:
                self.lru[p] = None            # to the back: most recently freed = evicted last
            else:
                self.hash_of[p] = None
                if self.poison_on:
                    self.poison.append(p)
                if self.low_first:
                    heapq.heappush(self.free_uncached, p)
                else:
                    self.free_uncached.append(p)

    def register(self, p: int, h: Hashable) -> bool:
        """Mark page p (ref > 0, full) as holding block hash h. If another page already holds h on this rank the cache
        keeps the existing page and p stays private (released to the uncached free list later)."""
        if h in self.cached:
            return self.cached[h] == p
        self.cached[h] = p
        self.hash_of[p] = h
        return True


@dataclass
class SeqKV:
    seq_id: Hashable
    rank: int
    pages: list[int] = field(default_factory=list)      # paged pool: page of logical block i
    num_registered: int = 0                             # leading blocks registered (or found) in the prefix cache
    window: dict[int, int] = field(default_factory=dict)  # window pool: logical 32-token block -> page
    state_slot: Optional[int] = None
    reserved: int = 0                                   # pages still reserved for this sequence's prompt
    cached_tokens: int = 0                              # positions < cached_tokens: paged KV came from the cache (no writes)
    replay_start: int = 0                               # no window KV below this position (bounded replay)
    wver: int = 0                                       # bumped on every change of `window` (D13 deltas: no dict diff)
    shadow: bool = False                                # a CP helper's shadow of a prompt (never hashed)
    cp_have: int = 0                                    # shadow: leading blocks of the prompt its replica holds
    inc: int = 0                                        # incarnation: unique per admit / move (uids and slots recycle)

    @property
    def capacity_tokens(self) -> int:
        return len(self.pages)


@dataclass
class AdmitResult:
    rank: int
    hit_blocks: int            # cached blocks this sequence now references on its rank (local + copied)
    cached_tokens: int         # = hit_blocks * 64: paged KV of these positions exists (never rewritten)
    replay_start: int          # first position to schedule: cached_tokens - min(128, cached_tokens)
    copied_blocks: int         # of hit_blocks, how many were copied from another rank
    copies: list[CopyOp]


class OwnedKVManager:
    def __init__(self, geom: Geometry):
        self.geom = geom
        specs = {p.name: p for p in geom.pools()}
        self.paged = [PagePool(specs["paged"], r) for r in range(geom.n_ranks)]
        self.window = [PagePool(specs["window"], r) for r in range(geom.n_ranks)]
        self.state = [PagePool(specs["state"], r) for r in range(geom.n_ranks)]
        self.seqs: dict[Hashable, SeqKV] = {}
        self.pinned: dict[int, list[tuple[str, int, int]]] = {}  # step -> pages pinned for copies (pool, rank, page)
        # step -> {(pool, rank, page): (src rank, src page)}: pages that this step's copies fill, with their ORIGINAL
        # source. A later copy of the same step that would read such a page reads the original instead, so no copy of a
        # step depends on another (e.g. a request admitted with a prefix copy and moved in the same step must not have
        # its pages copied before the prefix copy filled them)
        self.pending: dict[int, dict[tuple, tuple[int, int]]] = {}
        self.deferred: dict[int, list] = {}                       # step -> SeqKV freed later (async scheduling)
        self.stats = dict(copies=0, copy_bytes=0, evictions=0)
        self._inc = 0                                             # SeqKV incarnations handed out

    def take_poison(self, rank: int) -> list:
        """Debug ($PHYS_POISON_ON_FREE): [(pool, page)] of `rank` freed / evicted since the last call (see PagePool)."""
        out = []
        for name, pools in (("paged", self.paged), ("window", self.window), ("state", self.state)):
            pl = pools[rank]
            if pl.poison:
                out.extend((name, p) for p in pl.poison)
                pl.poison = []
        return out

    # ------------------------------------------------------------------------------------------ queries
    def lookup(self, block_hashes: list[Hashable]) -> list[int]:
        """Leading cached blocks of `block_hashes` on each rank."""
        out = []
        for pool in self.paged:
            n = 0
            for h in block_hashes:
                if h not in pool.cached:
                    break
                n += 1
            out.append(n)
        return out

    def free_pages(self, rank: int) -> int:
        """Allocatable paged pages on `rank` minus reservations of admitted prompts."""
        p = self.paged[rank]
        return p.num_free() - p.reserved

    def usage(self, rank: int) -> dict:
        p = self.paged[rank]
        return dict(used=p.num_used(), cached_free=len(p.lru), free=len(p.free_uncached), reserved=p.reserved,
                    total=p.num_pages, window_used=self.window[rank].num_used(), state_used=self.state[rank].num_used())

    def block_table(self, seq_id) -> list[int]:
        return self.seqs[seq_id].pages

    def window_table(self, seq_id) -> dict[int, int]:
        return self.seqs[seq_id].window

    # ------------------------------------------------------------------------------------------ admission
    def pages_for(self, tokens: int) -> int:
        return -(-tokens // self.geom.page_tokens)

    def hit_after_cap(self, hit_blocks: int, prompt_len: int) -> tuple[int, int, int]:
        """(blocks referenced, cached tokens, replay start) for a hit of `hit_blocks` blocks. The last prompt token is
        always computed (it produces the first logits), so a hit covers at most (prompt_len - 1) // 64 blocks; the last
        `replay_tokens` of the hit are re-run for the window and compressor state (production's bounded replay)."""
        bt = self.geom.page_tokens
        hit = min(hit_blocks, max(0, (prompt_len - 1) // bt))
        cached = hit * bt
        return hit, cached, cached - min(self.geom.replay_tokens, cached)

    def admit(self, seq_id, rank: int, block_hashes: list[Hashable], prompt_len: int, step: int = 0,
              copy_from: Optional[int] = None) -> Optional[AdmitResult]:
        """Create the sequence on `rank`: reference its cached prefix there (and, with copy_from, copy the longer prefix
        held by rank `copy_from`), allocate its state slot and reserve the pages its remaining prompt needs.
        Returns None (nothing changes) if the rank cannot hold the prompt."""
        assert seq_id not in self.seqs, seq_id
        hits = self.lookup(block_hashes)
        local = hits[rank]
        src = copy_from if copy_from is not None and hits[copy_from] > local else None
        want = hits[src] if src is not None else local
        keep, cached, replay_start = self.hit_after_cap(want, prompt_len)
        n_local = min(local, keep)
        n_copy = keep - n_local
        need_total = self.pages_for(prompt_len) - n_local      # pages to allocate: copies + the rest of the prompt
        pool = self.paged[rank]
        wm = int(self.geom.reserve_watermark * pool.num_pages)
        # the request's own prefix-hit pages that sit in the LRU (ref 0) count as free in num_free(), but touching them
        # below takes them out of it: count them as used BEFORE granting the reservation (otherwise a preempted request
        # re-admitted onto its own cached pages leaves reserved > num_free, a prefill deadlock)
        lru_hits = sum(1 for h in block_hashes[:n_local] if pool.ref[pool.cached[h]] == 0)
        if need_total + wm > self.free_pages(rank) - lru_hits or not self.state[rank].num_free():
            return None
        sk = SeqKV(seq_id, rank)
        self._inc += 1
        sk.inc = self._inc
        for h in block_hashes[:n_local]:
            p = pool.cached[h]
            pool.touch(p)
            sk.pages.append(p)
        sk.num_registered = n_local
        copies: list[CopyOp] = []
        if n_copy:
            dst = pool.alloc(n_copy)
            assert dst is not None
            spool = self.paged[src]
            for i, d in enumerate(dst):
                h = block_hashes[n_local + i]
                s = spool.cached[h]
                spool.touch(s)                      # pinned until release_copies(step)
                self.pinned.setdefault(step, []).append(("paged", src, s))
                copies.append(self._copy("paged", src, s, rank, d, step))
                pool.register(d, h)
                sk.pages.append(d)
            sk.num_registered = keep
            self.stats["copies"] += n_copy
            self.stats["copy_bytes"] += n_copy * pool.spec.bytes_per_page
        sk.state_slot = self.state[rank].alloc(1)[0]
        sk.reserved = need_total - n_copy
        sk.cached_tokens, sk.replay_start = cached, replay_start
        pool.reserved += sk.reserved
        self.seqs[seq_id] = sk
        return AdmitResult(rank, keep, cached, replay_start, n_copy, copies)

    # ------------------------------------------------------------------------------------------ growth
    def _window_need(self, sk: SeqKV, start: int, end: int) -> tuple[int, int]:
        """Logical window blocks a step computing positions [start, end) touches: [lo, hi] (none below replay_start)."""
        wt = self.geom.window_page_tokens
        lo = max(sk.replay_start, start - (self.geom.window - 1)) // wt
        return lo, (end - 1) // wt

    def can_allocate(self, seq_id, start: int, end: int, window_end: int | None = None) -> bool:
        sk = self.seqs[seq_id]
        need = max(0, self.pages_for(end) - len(sk.pages))
        avail = self.paged[sk.rank].num_free() - self.paged[sk.rank].reserved + min(sk.reserved, need)
        lo, hi = self._window_need(sk, start, window_end if window_end is not None else end)
        wneed = sum(1 for b in range(lo, hi + 1) if b not in sk.window)
        return need <= avail and wneed <= self.window[sk.rank].num_free()

    def allocate(self, seq_id, start: int, end: int, window_end: int | None = None) -> bool:
        """Make room for a step that computes positions [start, end) of the sequence (prefill chunk or the k+1 decode
        rows). window_end (>= end): window pages also up to there (the DSpark drafter writes window KV -- layers 40-42 --
        past the verified rows; it writes no paged KV). All-or-nothing: returns False and changes nothing if the owner
        rank is short of pages."""
        sk = self.seqs[seq_id]
        wend = window_end if window_end is not None else end
        # fast path (most decode steps): pages and window blocks already there. Window blocks are allocated in order
        # and released from the bottom, so the lowest and highest needed being present means all are.
        if -(-end // self.geom.page_tokens) <= len(sk.pages):
            wt = self.geom.window_page_tokens
            hi = (wend - 1) // wt
            lo = max(sk.replay_start, start - (self.geom.window - 1)) // wt
            w = sk.window
            if hi in w and lo in w:
                return True
        if not self.can_allocate(seq_id, start, end, wend):
            return False
        sk = self.seqs[seq_id]
        pool = self.paged[sk.rank]
        need = max(0, self.pages_for(end) - len(sk.pages))
        if need:
            use_res = min(sk.reserved, need)
            sk.reserved -= use_res
            pool.reserved -= use_res
            evict = max(0, need - len(pool.free_uncached))
            pages = pool.alloc(need)
            assert pages is not None
            self.stats["evictions"] += evict
            sk.pages.extend(pages)
        wpool = self.window[sk.rank]
        lo, hi = self._window_need(sk, start, wend)
        missing = [b for b in range(lo, hi + 1) if b not in sk.window]
        if missing:
            for b, p in zip(missing, wpool.alloc(len(missing))):
                sk.window[b] = p
            sk.wver += 1
        return True

    def commit(self, seq_id, num_computed: int, block_hash: Callable[[int], Hashable]) -> None:
        """After a step: `num_computed` tokens of the sequence have valid KV. Registers the newly full blocks under their
        hashes (block_hash(i) = chained hash of logical block i; only the new blocks are asked for) and releases window
        pages that fell below the window of the next step."""
        sk = self.seqs[seq_id]
        pool = self.paged[sk.rank]
        full = min(num_computed // self.geom.page_tokens, len(sk.pages))
        while sk.num_registered < full:
            pool.register(sk.pages[sk.num_registered], block_hash(sk.num_registered))
            sk.num_registered += 1
        lo, _ = self._window_need(sk, num_computed, num_computed + 1)
        w = sk.window
        if w and min(w) < lo:
            for b in [b for b in w if b < lo]:
                self.window[sk.rank].release(w.pop(b))
            sk.wver += 1

    def shrink_reservation(self, seq_id) -> None:
        """Drop what is left of the prompt reservation (prefill finished)."""
        sk = self.seqs[seq_id]
        self.paged[sk.rank].reserved -= sk.reserved
        sk.reserved = 0

    # ------------------------------------------------------------------------------------------ release
    # ------------------------------------------------------------------------------------------ CP shadows
    def open_shadow(self, key, rank: int, replay_start: int = 0) -> bool:
        """A CP helper's SHADOW of a prompt on `rank`: its own state slot (scratch ring), replica
        paged pages (allocate() grows them like the owner's) and scratch window pages; never hashed / prefix-cached.
        False (nothing changes) without a free state slot."""
        assert key not in self.seqs, key
        if not self.state[rank].num_free():
            return False
        sk = SeqKV(key, rank)
        self._inc += 1
        sk.inc = self._inc
        sk.state_slot = self.state[rank].alloc(1)[0]
        sk.replay_start = replay_start
        sk.shadow = True
        self.seqs[key] = sk
        return True

    def trim_window(self, seq_id, lo_pos: int) -> None:
        """Release window pages of blocks wholly below position lo_pos (a shadow's scratch window between CP chunks)."""
        sk = self.seqs[seq_id]
        wt = self.geom.window_page_tokens
        w = sk.window
        drop = [b for b in w if (b + 1) * wt <= lo_pos]
        for b in drop:
            self.window[sk.rank].release(w.pop(b))
        if drop:
            sk.wver += 1

    def free(self, seq_id, defer_until: int | None = None) -> None:
        """Sequence finished or preempted: release everything. Registered full pages stay in the rank's prefix cache
        (LRU, the sequence's LAST blocks are evicted first); unregistered pages go back to the free list.
        defer_until=s (async scheduling, D13 guarantee b): the sequence leaves the tables now, but its pages stay
        referenced until flush_deferred(s') with s' >= s, i.e. until the in-flight step that may still touch them is
        settled."""
        sk = self.seqs.pop(seq_id)
        if defer_until is not None:
            self.deferred.setdefault(defer_until, []).append(sk)
            return
        self._release(sk)

    def flush_deferred(self, upto_step: int) -> None:
        for s in [s for s in self.deferred if s <= upto_step]:
            for sk in self.deferred.pop(s):
                self._release(sk)

    def _release(self, sk: "SeqKV") -> None:
        pool = self.paged[sk.rank]
        pool.reserved -= sk.reserved
        for p in reversed(sk.pages):
            pool.release(p)
        for p in sk.window.values():
            self.window[sk.rank].release(p)
        if sk.state_slot is not None:
            self.state[sk.rank].release(sk.state_slot)

    def migrate(self, seq_id, dst: int, step: int, num_computed: int) -> Optional[list[CopyOp]]:
        """Move a running sequence to rank `dst`: copy its computed pages (paged + window + state slot) and switch its
        tables; the source pages are released after release_copies(step). None if `dst` lacks room."""
        sk = self.seqs[seq_id]
        if dst == sk.rank:
            return []
        src = sk.rank
        n_pages = len(sk.pages)
        dpool, wpool = self.paged[dst], self.window[dst]
        wm = int(self.geom.reserve_watermark * dpool.num_pages)
        if n_pages + sk.reserved + wm > self.free_pages(dst) or len(sk.window) > wpool.num_free() \
                or not self.state[dst].num_free():
            return None
        ops: list[CopyOp] = []
        new_pages = dpool.alloc(n_pages)
        # pages holding KV: the computed positions AND the whole cached prefix. A not-yet-started prefill (prefill
        # stealing) has num_computed = replay_start = hit - 128, and the bounded replay never writes paged KV below
        # cached_tokens: copying only below num_computed would leave the last 2 hit pages uncopied on dst yet registered
        # there (the moved request and every later hit on dst would read 128 positions of stale KV)
        valid = max(num_computed, sk.cached_tokens, sk.num_registered * self.geom.page_tokens)
        n_full = min(valid // self.geom.page_tokens, n_pages)
        spool = self.paged[src]
        for i, (s, d) in enumerate(zip(sk.pages, new_pages)):
            if i < n_full or i * self.geom.page_tokens < valid:
                ops.append(self._copy("paged", src, s, dst, d, step))
            if i < sk.num_registered and spool.hash_of[s] is not None:
                dpool.register(d, spool.hash_of[s])
        new_win = {}
        for (b, s), d in zip(sorted(sk.window.items()), wpool.alloc(len(sk.window))):
            ops.append(self._copy("window", src, s, dst, d, step))
            new_win[b] = d
        new_state = self.state[dst].alloc(1)[0]
        ops.append(self._copy("state", src, sk.state_slot, dst, new_state, step))
        # source pages: keep our reference until the copy ran (pinned), then release
        pin = self.pinned.setdefault(step, [])
        pin.extend(("paged", src, s) for s in sk.pages)
        pin.extend(("window", src, s) for s in sk.window.values())
        pin.append(("state", src, sk.state_slot))
        spool.reserved -= sk.reserved
        dpool.reserved += sk.reserved
        sk.rank, sk.pages, sk.window, sk.state_slot = dst, new_pages, new_win, new_state
        sk.wver += 1
        self._inc += 1
        sk.inc = self._inc
        self.stats["copies"] += len(ops)
        self.stats["copy_bytes"] += sum(self._pool(o.pool, dst).spec.bytes_per_page for o in ops)
        return ops

    def release_copies(self, step: int) -> None:
        """The copies planned for `step` have executed: drop the pins on their source pages."""
        for pool, r, p in self.pinned.pop(step, []):
            self._pool(pool, r).release(p)
        self.pending.pop(step, None)

    def _copy(self, pool: str, src: int, s: int, dst: int, d: int, step: int) -> CopyOp:
        """A copy of `step` from (src, s) to (dst, d), re-sourced to the original page if (src, s) is itself filled by a
        copy of this step; records (dst, d) as filled by it."""
        pend = self.pending.setdefault(step, {})
        while (pool, src, s) in pend:
            src, s = pend[(pool, src, s)]
        pend[(pool, dst, d)] = (src, s)
        return CopyOp(pool, src, s, dst, d, step)

    def _pool(self, name: str, rank: int) -> PagePool:
        return {"paged": self.paged, "window": self.window, "state": self.state}[name][rank]

    # ------------------------------------------------------------------------------------------ invariants
    def check(self) -> None:
        """Every page's ref count equals the references held by sequences + copy pins; free structures are disjoint and
        complete; every cached hash points at a page that carries it; reservations add up."""
        for name, pools in (("paged", self.paged), ("window", self.window), ("state", self.state)):
            for r, pool in enumerate(pools):
                refs = [0] * pool.num_pages
                for sk in list(self.seqs.values()) + [x for v in self.deferred.values() for x in v]:
                    if sk.rank != r:
                        continue
                    if name == "paged":
                        for p in sk.pages:
                            refs[p] += 1
                    elif name == "window":
                        for p in sk.window.values():
                            refs[p] += 1
                    elif sk.state_slot is not None:
                        refs[sk.state_slot] += 1
                for pins in self.pinned.values():
                    for pn, pr, p in pins:
                        if pn == name and pr == r:
                            refs[p] += 1
                assert refs == pool.ref, (name, r, [i for i in range(pool.num_pages) if refs[i] != pool.ref[i]][:8])
                free = set(pool.free_uncached)
                lru = set(pool.lru)
                assert len(free) == len(pool.free_uncached) and not (free & lru)
                for p in range(pool.num_pages):
                    in_free = p in free or p in lru
                    assert in_free == (pool.ref[p] == 0), (name, r, p, pool.ref[p], in_free)
                for p in lru:
                    assert pool.hash_of[p] is not None and pool.cached[pool.hash_of[p]] == p
                for h, p in pool.cached.items():
                    assert pool.hash_of[p] == h
                if name == "paged":
                    assert pool.reserved == sum(sk.reserved for sk in list(self.seqs.values())
                                                + [x for v in self.deferred.values() for x in v] if sk.rank == r), (r,)
                    assert pool.reserved >= 0
                    # every reservation can be honoured (copy pins on this rank may hold pages back until their step ran)
                    pins = sum(1 for v in self.pinned.values() for pn, pr, _ in v if pn == "paged" and pr == r)
                    assert pool.reserved <= pool.num_free() + pins, (r, pool.reserved, pool.num_free(), pins)
