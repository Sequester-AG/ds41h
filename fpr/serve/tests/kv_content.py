"""A device-side model of what every KV page holds, for scheduler tests.

The scheduler only plans: it hands out block tables, copy lists (CopyOp) and per-sequence work (SeqWork), and the
engine executes them. KVContent executes the same plans on a content model instead of tensors, so a test can check
that every read of a step finds the right data where the tables point:

  * a paged page is 64 slots; a slot holds the block hash of the position written there. The block hash identifies
    the token stream up to that block, so two requests that share a prefix write the same value, and a page that
    serves a prefix-cache hit must hold its hash in all 64 slots;
  * a window page is 32 slots of (block hash, position);
  * a state slot is (sequence, positions processed): the compressor / token-ring state is valid only for the
    sequence and position it was built for.

Execution order per step, as on the device:
  1. the plan's copies, all reading the state before the step (copy engines run them in any order, so a copy
     must never read a page another copy of the same step writes);
  2. every work item's reads are checked (paged KV of all earlier positions, the sliding window of the last 127
     positions, the state slot), then all writes are applied (paged slots >= cached_upto, window, state).
  3. after the scheduler's update(): every page registered in a rank's prefix cache must hold its hash.

By default a block's content is its block hash (the cache key). A request may carry `content_fn` (block index -> what
the block's KV really depends on, e.g. its tokens AND the images before it); the model then writes and expects that
content, keeps which content every cache key was written with, and reports a key used for two different contents
(two prompts that would share a page although their KV differs). Registered pages must hold their key's content.

Context-parallel (CP) prefill chunks follow the CP data flow: an entry reads the paged KV below the chunk start from
its own rank (the owner's pages or the helper's replica pages), the lowest helper reads its window halo from its own
window pages, and after the step every participant holds the chunk's records from its own sub-chunk and the lower
ones (in-step exchange); the owner holds the whole chunk.
"""
from __future__ import annotations

from collections import Counter

PAGE = 64
WIN_PAGE = 32
WINDOW = 128
_MIXED = object()          # a page whose slots do not all hold the same block hash


class KVContent:
    def __init__(self, sched, max_errors: int = 20):
        self.s = sched
        self.kv = sched.kv
        n = sched.g.n_ranks
        self.paged = [dict() for _ in range(n)]       # page -> [64 slots]
        self.window = [dict() for _ in range(n)]      # page -> [32 slots]
        self.state = [dict() for _ in range(n)]       # slot -> (seq key, positions processed)
        self.errors: list = []
        self.max_errors = max_errors
        self.stats = Counter()
        self._touched: set = set()
        self._sum: dict = {}                          # (rank, page) -> the hash all 64 slots hold, or _MIXED
        self._seen: dict = {}                         # seq key -> (rank, incarnation, blocks verified)
        self.key_content: dict = {}                   # cache key -> the content its block was written with (content_fn)
        self.step = 0
        self.sample_every = 16                        # decode reads: every block again each this many steps

    def _page_val(self, r, pg):
        key = (r, pg)
        val = self._sum.get(key)
        if val is None:
            v = self.paged[r].get(pg)
            val = v[0] if v is not None and v[0] is not None and all(x == v[0] for x in v) else _MIXED
            self._sum[key] = val
        return val

    # ------------------------------------------------------------------ helpers
    def _err(self, *msg) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(msg)
        self.stats["errors"] += 1

    def _req(self, rid):
        """The request whose token stream a sequence key carries (CP helper shadows: ("cp", rid, rank))."""
        if isinstance(rid, tuple) and len(rid) == 3 and rid[0] == "cp":
            return self.s.reqs[rid[1]]
        return self.s.reqs[rid]

    @staticmethod
    def _content(q):
        return getattr(q, "content_fn", None) or q.hash_fn

    def _pools(self, name):
        return {"paged": self.paged, "window": self.window, "state": self.state}[name]

    # ------------------------------------------------------------------ one step
    def apply(self, plan, acc: dict) -> None:
        """Run `plan` (as returned by schedule()) on the model; `acc` = accepted tokens per decode rid."""
        self.step += 1
        self._touched = set()
        self._copies(plan)
        for w in plan.work:
            self._check_reads(plan, w)
        for w in plan.work:
            self._write(w, acc)

    def _copies(self, plan) -> None:
        reads = {(c.pool, c.src_rank, c.src_page) for c in plan.copies}
        writes = {(c.pool, c.dst_rank, c.dst_page) for c in plan.copies}
        both = reads & writes
        if both:
            self._err("copy reads a page another copy of the same step writes", plan.step, sorted(both)[:3])
        vals = []
        for c in plan.copies:
            v = self._pools(c.pool)[c.src_rank].get(c.src_page)
            vals.append(list(v) if isinstance(v, list) else v)
        for c, v in zip(plan.copies, vals):
            self._pools(c.pool)[c.dst_rank][c.dst_page] = v
            self._touched.add((c.pool, c.dst_rank, c.dst_page))
            if c.pool == "paged":
                self._sum.pop((c.dst_rank, c.dst_page), None)
            self.stats[f"copy_{c.pool}"] += 1

    def _check_reads(self, plan, w) -> None:
        if w.kind not in ("prefill", "decode", "cp"):
            return
        sk = self.kv.seqs.get(w.rid)
        if sk is None:
            self._err("work for a sequence without KV", plan.step, w.rid)
            return
        if sk.rank != w.rank:
            self._err("work on a rank that does not own the sequence", plan.step, w.rid, w.rank, sk.rank)
            return
        q = self._req(w.rid)
        hf = self._content(q)
        role = w.cp
        if role is not None:
            chunk = role.subs[0][1]
            lowest = role.me == 0
        else:
            chunk, lowest = w.start, True
        r = w.rank
        # paged KV of every position below the chunk start (the indexer scores all of them). A decode step re-checks
        # only the blocks completed since its last check on this rank and incarnation, and all of them every
        # sample_every steps; prefill chunks and the first step after an admission or a move check every block
        seen = self._seen.get(w.rid)
        first = 0
        if w.kind == "decode" and seen is not None and seen[:2] == (r, sk.inc) \
                and (self.step + hash(str(w.rid))) % self.sample_every:
            first = seen[2]
        for b in range(first, chunk // PAGE):
            if b >= len(sk.pages):
                self._err("block table shorter than the context", plan.step, w.rid, b)
                return
            if self._page_val(r, sk.pages[b]) != hf(b):
                v = self.paged[r].get(sk.pages[b])
                self._err("stale or missing paged KV", plan.step, w.rid, "rank", r, "block", b,
                          "want", hf(b), "got", None if v is None else Counter(v).most_common(2))
                return
        self._seen[w.rid] = (r, sk.inc, chunk // PAGE)
        if chunk % PAGE:
            b = chunk // PAGE
            v = self.paged[r].get(sk.pages[b]) if b < len(sk.pages) else None
            if v is None or any(v[i] != hf(b) for i in range(chunk % PAGE)):
                self._err("stale or missing paged KV (partial block)", plan.step, w.rid, "rank", r, "block", b)
                return
        # the sliding window of the last 127 positions (none below replay_start); for CP only the lowest entry reads
        # it from its own pages (the others get the halo from the sub-chunk below them in-step)
        if lowest:
            for pos in range(max(w.replay_start, chunk - (WINDOW - 1)), chunk):
                pg = sk.window.get(pos // WIN_PAGE)
                v = self.window[r].get(pg) if pg is not None else None
                if v is None or v[pos % WIN_PAGE] != (hf(pos // PAGE), pos):
                    self._err("stale or missing window KV", plan.step, w.rid, "rank", r, "pos", pos,
                              "page", pg, "got", None if v is None else v[pos % WIN_PAGE])
                    return
        # the state slot: built for this sequence up to the chunk start, or rebuilt by a bounded replay
        owner_entry = role is None or role.me == len(role.subs) - 1
        if owner_entry:
            st = self.state[r].get(sk.state_slot)
            if st != (w.rid, chunk) and not (w.kind == "prefill" and chunk == w.replay_start):
                self._err("state slot not carried", plan.step, w.rid, "rank", r, "slot", sk.state_slot,
                          "want", (w.rid, chunk), "got", st)
        self.stats[f"reads_{w.kind}"] += 1

    def _write(self, w, acc: dict) -> None:
        if w.kind not in ("prefill", "decode", "cp"):
            return
        sk = self.kv.seqs.get(w.rid)
        if sk is None or sk.rank != w.rank:
            return
        q = self._req(w.rid)
        hf = self._content(q)
        r = w.rank
        role = w.cp
        if role is not None:
            lo = role.subs[0][1]
            hi = w.start + w.n                       # its own sub-chunk and every lower one (in-step exchange)
        else:
            lo, hi = w.start, w.start + w.n
        if getattr(q, "content_fn", None) is not None:
            # blocks completed by this write: the cache key they will be registered under, and their real content
            for b in range(max(lo, w.cached_upto) // PAGE, hi // PAGE):
                key, val = q.hash_fn(b), hf(b)
                prev = self.key_content.setdefault(key, val)
                if prev != val:
                    self._err("one prefix-cache key for two different contents", w.rid, "block", b, "key", key)
        for pos in range(max(lo, w.cached_upto), hi):
            b = pos // PAGE
            pg = sk.pages[b]
            v = self.paged[r].get(pg)
            if v is None:
                v = self.paged[r][pg] = [None] * PAGE
            v[pos % PAGE] = hf(b)
            self._sum.pop((r, pg), None)
            self._touched.add(("paged", r, pg))
        for pos in range(w.start, w.start + w.n):
            pg = sk.window.get(pos // WIN_PAGE)
            if pg is None:
                self._err("window block missing for a computed position", w.rid, pos)
                continue
            v = self.window[r].get(pg)
            if v is None:
                v = self.window[r][pg] = [None] * WIN_PAGE
            v[pos % WIN_PAGE] = (hf(pos // PAGE), pos)
        if role is None or role.me == len(role.subs) - 1:
            upto = w.start + (acc.get(w.rid, w.n) if w.kind == "decode" else w.n)
            self.state[r][sk.state_slot] = (w.rid, upto)

    # ------------------------------------------------------------------ after update()
    def check_registered(self, full: bool = False) -> None:
        """Every page in a rank's prefix cache holds its block hash (only pages touched this step unless `full`)."""
        for r, pool in enumerate(self.kv.paged):
            items = pool.cached.items() if full else (
                (pool.hash_of[p], p) for (name, rr, p) in self._touched
                if name == "paged" and rr == r and pool.hash_of[p] is not None)
            for h, p in items:
                if pool.cached.get(h) != p:
                    continue
                if self._page_val(r, p) != self.key_content.get(h, h):
                    v = self.paged[r].get(p)
                    self._err("registered page without its content", "rank", r, "page", p, "hash", h,
                              "got", None if v is None else Counter(v).most_common(2))
                    return
                self.stats["registered_checked"] += 1


def run_checked(s, steps: int, rng, arrivals=None, dt: float = 0.05, full_every: int = 25, on_step=None):
    """Drive scheduler `s` for `steps` steps on a KVContent model (random acceptance 1 .. k + 1); `arrivals` maps a step
    to requests added before it. Returns (model, finished requests)."""
    m = KVContent(s)
    done, now = [], 0.0
    arrivals = arrivals or {}
    for i in range(steps):
        for q in arrivals.get(i, ()):
            s.add_request(q)
        if not (s.running or s.waiting) and not any(k > i for k in arrivals):
            break
        p = s.schedule(now)
        acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
        m.apply(p, acc)
        if on_step is not None:
            on_step(i, p)
        now += max(p.est_us / 1e6, dt)
        done += s.update(p, acc, now)
        m.check_registered(full=(i % full_every == 0))
        if i % full_every == 0:
            s.kv.check()
    m.check_registered(full=True)
    s.kv.check()
    return m, done
