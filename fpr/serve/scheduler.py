"""Rank-aware scheduler of the ds41h-phys serving engine: sequence-to-rank placement, per-rank prefill quota,
admission, preemption, migration.

Pure policy over abstract requests (stdlib only, CPU-testable). The vLLM adapter (adapter/phys_scheduler.py) wraps it in
a vLLM SchedulerInterface; a CPU simulation can drive it with a long-context agentic workload.

One step:
  1. every decoding sequence gets its k + 1 = 4 rows on its owner rank (KV for 4 more positions; on a full rank:
     free cached pages are evicted first, then the newest sequence of THAT rank is moved to another rank, or preempted
     by recompute if no rank has room -- never a sequence on another rank);
  2. optional rebalance: when the per-rank load spread stays above a threshold, one decoding sequence moves from the
     heaviest to the lightest rank (pages copied rank-to-rank, the sequence skips one step);
  3. admission, FCFS: each waiting request goes to the rank that minimizes the largest projected rank load (its rows,
     its indexer context, its uncached prefill spread over a horizon); a longer cached prefix on another rank is copied
     instead of recomputed when that saves >= migrate_min_blocks pages;
  4. prefill quota: chunks of the pending prefills are added water-filling style, always to the currently lightest
     rank that has backlog, while the projected step stays within (decode-only step + slack); the slack grows with the
     prefill backlog (bounded), so prefill keeps up with arrivals without stretching every decode step.
The cost model (cost_model.py) supplies all times; nothing here depends on its constants.

Image prompts (Request.images: one (offset, length, identifier) per image, the identifier a content hash of the image):
  * prefix-cache keys: every block at or after an image's first position is keyed by its token-stream hash AND the
    identifiers and offsets of all images up to the block's end (ImageBlockKeys). Prompts whose placeholder tokens are
    the same but whose images differ never share a page, on admission, prefix copies or moves; blocks before the first
    image keep their text keys, so a shared text prefix (a system prompt) still hits;
  * every prefill work item lists the images overlapping its positions (SeqWork.images), the bounded replay below a
    cache hit included: the engine needs those images' encoder outputs for that step;
  * a prompt with images prefills on its owner rank only (no context-parallel chunks) and is never moved (stolen,
    rebalanced or relieved by a move) before its prefill completes; a memory-relief victim among them is preempted
    (recomputed) instead. After the prefill it moves like any other sequence. This is a performance limit only: a long
    image prompt prefills at one rank's speed. SchedConfig.image_pin (default on) holds these two rules.
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

from cost_model import CostModel, RankLoad
from geometry import SLIDING_WINDOW, Geometry
from kv_manager import CopyOp, OwnedKVManager

WAITING, PREFILL, DECODE, FINISHED = "waiting", "prefill", "decode", "finished"


@dataclass(eq=False)          # identity equality: `q in running` / remove() must not compare every field
class Request:
    rid: Hashable
    prompt_len: int
    max_new_tokens: int
    arrival: float
    hash_fn: Callable[[int], Hashable]     # block index -> chained block hash of the request's token stream
    priority: int = 0
    score: bool = False                    # teacher-forced scoring (prompt logprobs): every prompt row runs all 40 layers
                                           # and is sampled for logprobs (exactness checks); no decoder-side tail cut
    forced: bool = False                   # forced decode (= production's patch_force): drafts := the next forced
                                           # tokens, all accepted; raw logits of every verify row returned
    images: tuple = ()                     # ((offset, length, identifier), ...) of the prompt's images, by offset; the
                                           # identifier is the image's content hash (vLLM: mm_feature.identifier)
    # state
    phase: str = WAITING
    rank: Optional[int] = None
    num_computed: int = 0                  # positions whose KV is final (window + paged)
    num_output: int = 0
    prefill_end: int = 0                   # positions to (re)compute before decoding: prompt_len, or after a
                                           # preemption prompt_len + num_output (vLLM recompute semantics)
    cached_tokens: int = 0                 # prefix-cache hit of the current admission (positions)
    replay_start: int = 0
    skip_until_step: int = -1              # migrated: resumes at this step
    first_token_time: Optional[float] = None
    admit_time: Optional[float] = None
    pf_progress_time: Optional[float] = None   # last time a prefill chunk of this request was scheduled (or admission)
    first_chunk_time: Optional[float] = None   # accounting: when its first prefill chunk was scheduled
    cp_forced: bool = False                    # CP prefill: it went CP under cp_force_min (stays forced to its end)
    last_pf_step: int = -(1 << 30)             # the step of its last scheduled prefill chunk (async CP: a CP chunk's
                                               # replica copies read the previous chunk's pages, so it must be settled)
    home_rank: Optional[int] = None            # accounting: the rank holding the longest cached prefix at admission
    admit_steps_waited: int = 0                # accounting: schedule() calls it spent waiting for admission
    cp_shadows: dict = field(default_factory=dict)   # CP prefill: helper rank -> (shadow key, shadow uid)
    cp_prev_subs: list = field(default_factory=list)   # CP prefill: the previous CP chunk's subs (X1b sources)
    finish_time: Optional[float] = None
    n_preempt: int = 0
    n_migrate: int = 0
    inflight: int = 0                      # async mode: scheduled decode steps not settled yet
    first_pending: bool = False            # async mode: last prefill chunk scheduled, its sampled token not known yet
    uid: int = -1                          # global sequence slot 0 .. max_seqs - 1, stable across rank moves (the
                                           # engine keys rank-independent per-sequence state by it, e.g. the head's
                                           # vocab-sharded sampler state); new uid after a preemption
    hit_tokens_total: int = 0              # accounting: prompt tokens served from the prefix cache (incl. copies)
    copied_tokens_total: int = 0

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + self.num_output

    def block_hashes(self, n_tokens: int) -> list:
        return [self.hash_fn(i) for i in range(n_tokens // 64)]

    @property
    def prefill_left(self) -> int:
        return self.prefill_end - self.num_computed if self.phase == PREFILL else 0

    def images_in(self, lo: int, hi: int) -> tuple:
        """The images overlapping positions [lo, hi)."""
        return tuple(im for im in self.images if im[0] < hi and im[0] + im[1] > lo)

    @property
    def pinned(self) -> bool:
        """An image prompt before its prefill completed: owner-only prefill, never moved."""
        return bool(self.images) and self.phase != DECODE


def check_images(images, prompt_len: int) -> Optional[str]:
    """Why an image list is malformed (None if fine): sorted, non-overlapping, non-empty spans inside the prompt."""
    end = 0
    for im in images:
        if len(im) != 3:
            return f"image entry {im!r} is not (offset, length, identifier)"
        off, n, ident = im
        if not isinstance(off, int) or not isinstance(n, int) or n <= 0 or off < end or off + n > prompt_len:
            return f"image span ({off}, {n}) out of order, overlapping or outside the prompt ({prompt_len} tokens)"
        if ident is None or ident == "":
            return "image without an identifier"
        end = off + n
    return None


class ImageBlockKeys:
    """Prefix-cache key of block i for a prompt with images: the token-stream key base(i) for blocks that end at or before
    the first image's offset; for every later block (base(i), the (offset, identifier) of every image starting before
    the block's end). KV at a position depends on every input before it, so a block after an image differs whenever
    any earlier image differs, even when the token streams (placeholder ids) are equal. vLLM's block hashes already
    carry the image identifiers; this keeps the guarantee for any base key (a text-only hash included)."""

    __slots__ = ("base", "offsets", "prefix", "page")

    def __init__(self, base: Callable[[int], Hashable], images: tuple, page: int = 64):
        self.base, self.page = base, page
        self.offsets = [im[0] for im in images]
        self.prefix = [tuple((im[0], im[2]) for im in images[:k]) for k in range(len(images) + 1)]

    def __call__(self, i: int) -> Hashable:
        k = bisect.bisect_left(self.offsets, (i + 1) * self.page)     # images with offset < the block's end
        b = self.base(i)
        return b if k == 0 else ("mm", b, self.prefix[k])


@dataclass
class SchedConfig:
    placement: str = "balanced"            # balanced | affinity | round_robin
    migrate_min_blocks: int = 16           # copy a cached prefix from another rank if it saves >= this many pages
    max_copy_bytes_per_step: int = 600_000_000   # NVLink copy budget per step (~5 ms at 123 GB/s, copy engines)
    horizon_steps: int = 12                # placement spreads a request's prefill over this many steps
    min_slack_us: float = 3000.0           # prefill time always allowed per step when there is backlog
    max_slack_frac: float = 0.8            # a step with prefill may grow to (1 + this) x the decode-only step ...
    prefill_floor_us: float = 60_000.0     # ... or to this total, whichever is larger (light decode load, cold start)
    steal_decode_tol: float = 0.05         # a steal may raise the largest per-rank decode load by this fraction
    placement_tol: float = 0.02            # ranks within this fraction of the best decode peak count as tied
    backlog_drain_steps: int = 10          # slack = backlog / drain_steps (bounded by the two above)
    quantum: int = 256                     # prefill chunk granularity (tokens; multiple of the 64-token page)
    rebalance: bool = True
    rebalance_threshold: float = 0.10      # (max - min) / mean of the per-rank decode load
    rebalance_patience: int = 8            # checks above threshold before moving a sequence
    rebalance_every: int = 8               # check the decode balance every this many steps (host time)
    max_waiting_admit_per_step: int = 64
    async_mode: bool = False               # D13: schedule step N+1 before step N is settled (decode positions resolved on
                                           # the device; worst-case KV; deferred frees; no decode rank moves)
    prefill_order: str = "srpt"            # per-rank prefill order: srpt (shortest remaining first, with aging) | fcfs
    prefill_aging_tok_per_s: float = 40_000.0   # srpt: a waiting prefill gains this many tokens of priority per second
    small_prefill_tokens: int = 4096 + 256     # srpt: prefills with at most this many tokens left go before every longer one
    large_min_rows: int = 0                    # > 0: the OLDEST longer prefill on each rank gets up to this many rows of
                                               # every step that prefills, ahead of the short ones (an anti-starvation
                                               # share; the sim, 3 seeds: no better than none, so off; see starve_* below)
    large_share_after_s: float = 0.0           # ... once it has waited this long since admission
    large_share_frac: float = 0.25             # ... and at most this share of the step's prefill time budget
    starve_wait_s: float = 4.0                 # anti-starvation by capacity: while a longer prefill (> small_prefill_tokens)
    starve_slack_frac: float = 1.2             # has waited this long since admission, the slack cap is this x the decode
                                               # step instead of max_slack_frac (a served replay near prefill saturation:
                                               # 4-16K TTFT p95 86 s under short-first; 1.6 cut the TTFT more but cost
                                               # 4-6 % throughput at full load). starve_wait_s = inf: off
    amortize_mixed: float = 32.0           # a step that carries prefill gets prefill worth >= this x the cost model's
                                           # fixed mixed-step cost (banked from the decode-only steps' slack). With the
                                           # MoE(T) cost, tok/s at full load (replay workload, 160 seats):
                                           # am 2 3,402 / 4 3,413 / 8 3,928 / 16 4,008 / 32 4,108 (TTFT p95 < 4K ~1.6 s)
    mixed_target_tokens: int = 0           # > 0: token-target mixed steps instead of the time bank: no prefill until the
                                           # pending prefill reaches this many tokens (or a prefill waited mixed_target_wait_s),
                                           # then fill the step to it (bigger mixed steps amortize the MoE weight stream)
    mixed_target_wait_s: float = 1.0       # ... the starvation guard of that mode
    mixed_target_full_wait_s: float = float("inf")   # ... a rank holding a full chunk (>= max_prefill_tokens_per_rank
                                           # rows) whose prefill made no progress for this long fires the step too (inf:
                                           # off). Per-rank pending is capped at one chunk, so a lone long prompt (4,096 <
                                           # 12,288) fired on the 1 s guard alone: one chunk per second, ~4K tok/s without
                                           # CP, ~16K with (measured). Waiting cannot add rows to a
                                           # full rank, and the prompt needs ceil(left / cap) mixed steps either way
    mixed_target_fill: bool = False        # ... True: a step that fires fills to the row caps (demand-driven prefill
                                           # share: mixed steps fire as often as the pending prefill amortizes them)
    mixed_target_total: bool = False       # ... count the whole pending prefill against the threshold (False: at most
                                           # one step's rows per rank, so a backlog on two ranks never reaches 12K and the
                                           # step fires on the wait alone: 1.3 mixed steps / s at saturation)
    long_order: str = "srpt"               # order of the prefills above small_prefill_tokens: "srpt" (aged shortest-left
                                           # first) or "fcfs" (oldest first: a startup backlog drains in arrival order)
    backlog_drain_until: str = "first_chunk"   # "first_chunk": drain while a long prefill admitted >= the wait ago has
                                           # no first chunk yet; "finished": ... while it is still prefilling (with CP
                                           # the first chunk starts early and the prompt then progresses slowly:
                                           # 192 sequences with CP: > 64K first-chunk wait p50 94 s but prefill 74 s)
    backlog_drain_wait_s: float = float("inf")  # target mode: while a long prefill has waited this long for its first
                                           # chunk, the decode floor is off and a step fires on any pending prefill
                                           # (drain a backlog at full prefill share, e.g. a startup burst)
    mixed_adapt_lo: int = 0                # load-adaptive target mode (off: 0): with <= lo decoding sequences the
    mixed_adapt_hi: int = 0                # controller fires on any pending prefill, every step (no 12K threshold,
                                           # no wait, no decode floor: production's prefill-in-every-step); with >= hi
                                           # it batches as configured; linear in between (lambda = (n_dec - lo) /
                                           # (hi - lo)). Batching amortizes a mixed step's fixed cost, which only pays
                                           # near full load (served decode-only steps: p50 79 sequences at concurrency
                                           # 144, 118 at 192)
    mixed_adapt_min: float = 0.0           # ... lambda never below this (served eager mixed steps cost ~40-50 ms more
                                           # than a graphed decode step even for 128 prefill rows, so light batching
                                           # can still pay at low load)
    mixed_adapt_force_min: int = 0         # ... and the forced-CP threshold slides from this (lambda 0) up to
                                           # cp_force_min (lambda 1); 0: cp_force_min at every load
    mixed_min_decode_steps: int = 2        # ... the decode floor of that mode: at least this many decode-only steps
                                           # between two mixed steps (a backlog cannot run mixed steps back to back)
    amortize_min_fixed_us: float = 5_000.0  # ... only when that fixed cost is material (eager mixed steps ~34 ms; with
                                           # graph-quality mixed steps, ~0.5 ms, banking would only add delay)
    max_mixed_step_us: float = 600_000.0   # ... and never more than this (decoders stall for one step; production's
                                           # 16,896-token chunked-prefill steps are ~300-400 ms)
    max_prefill_wait_s: float = float("inf")   # TTFT guard (off): a prefill without progress this long gets a mixed
                                           # step now, bank or not. Off by default: on the replay-calibrated sim it cuts TTFT
                                           # p95 (10.6 -> 5.2 s at 1 s) but adds eager mixed steps, so mean call 24.2 -> 26.2 s
                                           # and tok/s -11 %. Enable via PHYS_SCHED="max_prefill_wait_s=..." if TTFT matters
    cp_async_settle_steps: int = 2         # async mode: a CP chunk only for a prompt whose previous chunk was scheduled
                                           # >= this many steps ago (2: settled before this schedule, so its pages are
                                           # final whatever the worker's copy ordering; 1: back-to-back, safe when every
                                           # prefill / copy step is a synchronous fallback run on the scheduler's thread
                                           # -- the engine's D13 guarantee)
    cp_prefill: bool = False               # context-parallel prefill (engine/cp.py): a long prompt's chunk
                                           # is split over all ranks (helpers: shadow sequences), owner last
    cp_min: int = 8192                     # ... for prompts with at least this many prefill tokens left
    cp_force_min: int = 0                  # ... > 0: a prompt with at least this many tokens left goes CP even when the
                                           # helpers are busy (no light-helper gate), the oldest such prompt first, and
                                           # its chunk takes >= cp_force_room_frac of every rank's chunk room before the
                                           # short prefills (> 64K TTFT: served 20-30 s at 6-16K tok/s per prompt)
    cp_force_room_frac: float = 0.5
    cp_force_srpt_min_left: int = 0        # ... and a prompt in CP with fewer tokens left than this is never overtaken
                                           # (131072: only a huge prompt yields; 32-64K arrivals preempting a 100K one
                                           # cost the 64-128K class in the sim)
    cp_force_srpt_hold: float = 1.0        # ... cp_force_order "srpt": a prompt already in CP keeps its slot until a
                                           # waiting forced prompt has fewer than hold x its tokens left (1.0: plain
                                           # srpt; 0.5: an 89K prompt overtakes a 240K one mid-prefill, not a 100K one;
                                           # in a replay the 89K waited 3.95 s behind ~200K tokens left)
    cp_force_order: str = "fcfs"           # ... which forced prompt goes first: "fcfs" (oldest), "srpt" (fewest
                                           # tokens left: a 95K prompt does not wait behind a 300K re-prefill) or
                                           # "split" (those under cp_force_min, forced at low load by
                                           # mixed_adapt_force_min, shortest first; the big ones after them, oldest
                                           # first: srpt let a stream of 2-16K prompts starve the big ones) or
                                           # "tiered" (prompts with < cp_force_srpt_min_left tokens left first, the
                                           # huge ones after them, oldest first in each tier: a 95K prompt never waits
                                           # behind a 240K one, and nothing overtakes a mid-size prompt mid-prefill)
    cp_force_decode_steps: int = -1        # ... >= 0: while a forced prompt is pending, target mode fires a mixed step
                                           # after this many decode-only steps (the token threshold waived; -1: the
                                           # ordinary mixed_min_decode_steps cadence): its CP chunks near back-to-back
    cp_replay_first: bool = True           # ... a CP candidate still inside its bounded replay prefills only up to its
                                           # cached boundary on its own rank (the replay rows), and goes CP from the next
                                           # mixed step: follow-ups' first chunks were never CP-eligible (a served replay:
                                           # lone 4,096-row chunks on one rank in 24 % of mixed steps)
    cp_exchange_us: float = 200.0          # ... and this per-step cost of its X1 / X2 exchanges (cost model)
    cp_light_frac: float = 0.5             # a helper rank qualifies if its own pending prefill is < this x the prompt's
                                           # remaining (0: < one per-rank chunk)
    image_pin: bool = True                 # image prompts (Request.images): no CP chunks, and never moved (stolen,
                                           # rebalanced, relieved) before their prefill completed (v1.1: off needs the
                                           # engine's image rows on CP helpers and per-rank encoder residency)
    steal: bool = True                     # a rank without prefill backlog takes a not-yet-started prefill elsewhere
    steal_min_backlog_steps: float = 2.0   # ... if the donor's backlog exceeds this many steps of prefill slack
    max_steals_per_step: int = 2


@dataclass
class SeqWork:
    rid: Hashable
    rank: int
    kind: str                              # "decode" | "prefill"
    start: int                             # first position computed this step
    n: int                                 # positions computed (decode: k + 1 rows)
    cached_upto: int = 0                   # positions < cached_upto: no paged writes (bounded replay)
    replay_start: int = 0                  # no window KV read below this position
    last_chunk: bool = False               # prefill: produces the first output token
    score: bool = False                    # all rows run layers 21..39 and return logprobs
    forced: bool = False                   # forced decode: accept every row, return raw logits
    uid: int = -1                          # Request.uid
    known: int = -1                        # async decode work: the host's settled num_computed (start = -1: device)
    cp: Optional[object] = None            # interfaces.CPRole for CP prefill work (kind "prefill" owner / "cp" helper)
    images: tuple = ()                     # prefill: the request's images overlapping [start, start + n) (their encoder
                                           # outputs are inputs of this step, the bounded replay's rows included)


@dataclass
class StepPlan:
    step: int
    work: list = field(default_factory=list)          # SeqWork
    copies: list = field(default_factory=list)        # CopyOp, run before the step's kernels
    preempted: list = field(default_factory=list)
    migrated: list = field(default_factory=list)
    loads: list = field(default_factory=list)         # RankLoad per rank
    est_us: float = 0.0
    idle_frac: float = 0.0

    def rows(self, rank: int) -> int:
        return sum(w.n for w in self.work if w.rank == rank)


class RankScheduler:
    def __init__(self, geom: Geometry, cost: CostModel, cfg: SchedConfig | None = None):
        self.g, self.cost, self.cfg = geom, cost, cfg or SchedConfig()
        self.kv = OwnedKVManager(geom)
        self.reqs: dict[Hashable, Request] = {}
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []          # admitted, by admission order
        self.step_id = 0
        self.rr = 0
        self.imbalance_steps = 0
        self.stats = dict(preemptions=0, migrations=0, prefix_copies=0, admitted=0, finished=0,
                          copy_bytes=0, rejected_admit=0, steals=0, copy_waits=0,
                          copy_over_budget=0, rejected_too_long=0, cp_chunks=0, cp_rows=0, cp_copy_pages=0)
        self.rejected: dict = {}                  # rid -> reason (add_request: no rank could ever hold it)
        self.free_uids: deque[int] = deque(range(geom.max_seqs))
        self.prefilling: set = set()              # running requests in phase PREFILL (host time: no O(n) scans)
        self.pf_credit = 0.0                      # banked prefill time (us) while the mixed-step fixed cost is amortized
        self.dec_since_mixed = 1 << 30            # decode-only steps since the last step that carried prefill
        self.free_shadow_uids: deque[int] = deque(range(geom.max_seqs, 4 * geom.max_seqs))   # CP shadows' uids
        self.lam = 1.0                            # load-adaptive target mode: this step's batching factor (0 .. 1)

    def take_uid(self, q: Request) -> None:
        q.uid = self.free_uids.popleft()

    def free_uid(self, q: Request) -> None:
        if q.uid >= 0:
            self.free_uids.append(q.uid)
            q.uid = -1

    # ------------------------------------------------------------------------------------------ requests
    def max_request_tokens(self) -> int:
        """Prompt + output tokens one sequence may reach: an EMPTY rank's paged pool minus the reserve watermark and the
        drafter's lookahead, capped at max_model_len."""
        pool = min(p.num_pages for p in self.kv.paged)
        wm = int(self.g.reserve_watermark * pool)
        return min(self.g.max_model_len, (pool - wm) * self.g.page_tokens - self.g.rows_per_decode_seq - self.g.draft_window_tokens)

    def add_request(self, req: Request) -> bool:
        """Queue a request. A request no rank could ever hold (prompt + max output > max_request_tokens) is NOT queued:
        at the head of the FCFS queue it would block every request behind it forever (e.g. a 398K-token item on
        2 GB mock pools). Returns False and records the reason in self.rejected[rid]; the caller reports it (vLLM: an
        error output, as for max_model_len)."""
        cap = self.max_request_tokens()
        if req.prompt_len + req.max_new_tokens > cap:
            self.rejected[req.rid] = (f"prompt {req.prompt_len} + max_tokens {req.max_new_tokens} tokens > {cap} "
                                      f"(one rank's KV pool / max_model_len)")
            self.stats["rejected_too_long"] += 1
            req.phase = FINISHED
            return False
        if req.images:
            why = check_images(req.images, req.prompt_len)
            if why is not None:
                self.rejected[req.rid] = why
                self.stats["rejected_images"] = self.stats.get("rejected_images", 0) + 1
                req.phase = FINISHED
                return False
            if not isinstance(req.hash_fn, ImageBlockKeys):
                req.hash_fn = ImageBlockKeys(req.hash_fn, tuple(req.images), self.g.page_tokens)
        self.reqs[req.rid] = req
        self.waiting.append(req)
        return True

    def abort(self, rid) -> None:
        req = self.reqs.pop(rid)
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
            if req.cp_shadows:
                self._drop_shadows(req)
            self.kv.free(rid, defer_until=self._defer())
            self.free_uid(req)
        self.prefilling.discard(req)
        req.phase = FINISHED

    def seats_per_rank(self) -> int:
        """Sequences a rank may hold: its seat cap, and all of them decoding must fit the per-rank row cap."""
        return min(self.g.max_seqs_per_rank, self.g.rows_cap // self.g.rows_per_decode_seq)

    def on_rank(self, r: int) -> list[Request]:
        return [q for q in self.running if q.rank == r]

    # ------------------------------------------------------------------------------------------ loads
    def _decode_load(self, q: Request) -> float:
        return self.cost.decode_seq_rank_us(q.num_tokens, self.g.rows_per_decode_seq)

    def _backlog_us(self, q: Request) -> float:
        if q.phase != PREFILL:
            return 0.0
        return self.cost.prefill_rank_us(q.num_computed, q.prefill_left)

    def rank_loads(self) -> list[float]:
        """Projected per-rank cost per step: decode rows + indexer of running sequences + pending prefill over the
        horizon. Prefilling sequences count with their future decode load too (they will decode soon)."""
        out = [0.0] * self.g.n_ranks
        for q in self.running:
            out[q.rank] += self._decode_load(q) + self._backlog_us(q) / self.cfg.horizon_steps
        return out

    # ------------------------------------------------------------------------------------------ step
    def schedule(self, now: float = 0.0) -> StepPlan:
        step = self.step_id
        self.step_id += 1
        plan = StepPlan(step)
        copy_budget = [self.cfg.max_copy_bytes_per_step]
        plan.loads = loads = [RankLoad() for _ in range(self.g.n_ranks)]
        k1 = self.g.rows_per_decode_seq

        # 1. decode rows
        for q in list(self.running):
            if q.phase != DECODE or q not in self.running:
                continue
            if q.skip_until_step > step:
                continue
            if self.cfg.async_mode:
                # D13 (d): skip if the in-flight steps certainly finish it (each appends >= 1 token)
                if q.num_output + q.inflight + int(q.first_pending) >= q.max_new_tokens:
                    continue
                lead = (q.inflight + 1) * k1               # worst case: every in-flight step accepted k + 1
                n = min(k1, max(1, self.g.max_model_len - q.num_computed - q.inflight * k1))
                end = min(q.num_computed + lead, self.g.max_model_len)
                # resolved on the device (rowexp); forced sequences accept every row, so their start is known here
                start = q.num_computed + q.inflight * k1 if q.forced else -1
            else:
                n = min(k1, self.g.max_model_len - q.num_computed)
                end = min(q.num_computed + n, self.g.max_model_len)
                start = q.num_computed
            wend = min(end + self.g.draft_window_tokens, self.g.max_model_len)    # + the drafter's block (window only)
            if not self._allocate_or_relieve(q, q.num_computed, end, plan, copy_budget, wend):
                continue
            if q.phase != DECODE or q.skip_until_step > step:     # moved while relieving
                continue
            plan.work.append(SeqWork(q.rid, q.rank, "decode", start, n, q.cached_tokens, q.replay_start,
                                     forced=q.forced, uid=q.uid, known=q.num_computed))
            plan.loads[q.rank].add_decode(n, q.num_tokens)
            if self.cfg.async_mode:
                q.inflight += 1
        loads = plan.loads

        # 2. rebalance (every rebalance_every steps)
        if self.cfg.rebalance and self.cfg.placement == "balanced" and step % self.cfg.rebalance_every == 0 \
                and not self.cfg.async_mode:
            self._maybe_rebalance(plan, copy_budget)
        loads = plan.loads

        # 3. admission (not in a step that had to preempt: vLLM does the same, it avoids preempt / re-admit thrash)
        if not plan.preempted:
            self._admit(now, plan, copy_budget)

        # 3b. prefill stealing
        if self.cfg.steal and self.cfg.placement == "balanced" and self.prefilling:
            self._steal(plan, copy_budget)
        loads = plan.loads

        # 4. prefill quota
        self._fill_prefill(plan, loads, now)

        cb = sum(self.kv._pool(o.pool, o.dst_rank).spec.bytes_per_page for o in plan.copies)
        c = self.cost.step(loads, cb)
        plan.loads, plan.est_us, plan.idle_frac = loads, c.total_us, c.idle_fraction
        if any(w.kind != "decode" for w in plan.work):
            self.dec_since_mixed = 0
        elif plan.work:
            self.dec_since_mixed += 1
        return plan

    # -- 1. growth / pressure
    def _allocate_or_relieve(self, q: Request, start: int, end: int, plan: StepPlan, budget: list,
                             window_end: int | None = None) -> bool:
        while not self.kv.allocate(q.rid, start, end, window_end):
            victims = sorted(self.on_rank(q.rank), key=lambda x: (x.priority, x.admit_time or 0.0, str(x.rid)))
            victim = victims[-1]                         # newest (lowest priority) sequence of THIS rank
            moved = False if (self.cfg.async_mode and victim.phase == DECODE) else \
                self._move(victim, plan, budget, exclude=q.rank)
            if not moved:
                self._preempt(victim, plan)
            self._unschedule(plan, victim.rid)
            if victim is q or moved or self.cfg.async_mode:
                # a moved victim's source pages stay pinned until its copies ran, and in async mode a preempted
                # victim's pages are freed only after the in-flight step settles: the room appears next step
                # (without this, async mode would preempt the whole rank in one step: a livelock)
                return False
        return True

    def _unschedule(self, plan: StepPlan, rid) -> None:
        """Drop work already planned this step for a sequence that moved or was preempted; rebuild its rank's load."""
        gone = [w for w in plan.work if w.rid == rid]
        if not gone:
            return
        plan.work = [w for w in plan.work if w.rid != rid]
        for r in {w.rank for w in gone}:
            ld = RankLoad()
            for w in plan.work:
                if w.rank == r and w.kind == "decode":
                    ld.add_decode(w.n, self.reqs[w.rid].num_tokens)
            plan.loads[r] = ld

    def _defer(self, in_schedule: bool = False):
        """async mode: frees wait until no dispatched step can touch the pages (D13 guarantee b). From inside
        schedule() (a preemption) the victim is not in the plan being built, so only the in-flight step N matters
        (step_id - 2); from update() (a finish) the sequence may be in the newest dispatched plan (step_id - 1)."""
        if not self.cfg.async_mode:
            return None
        return self.step_id - (2 if in_schedule else 1)

    def _preempt(self, q: Request, plan: StepPlan) -> None:
        if q.cp_shadows:
            self._drop_shadows(q)
        self.kv.free(q.rid, defer_until=self._defer(in_schedule=True))
        q.inflight, q.first_pending = 0, False
        self.running.remove(q)
        self.free_uid(q)
        self.prefilling.discard(q)
        q.phase, q.rank, q.num_computed, q.cached_tokens, q.replay_start = WAITING, None, 0, 0, 0
        q.n_preempt += 1
        self.waiting.appendleft(q)
        plan.preempted.append(q.rid)
        self.stats["preemptions"] += 1

    def _move(self, q: Request, plan: StepPlan, budget: list, exclude: int, dst: Optional[int] = None) -> bool:
        """Copy a running sequence to another rank (the one with the most free pages unless dst is given)."""
        if q.cp_shadows:                                 # a CP prompt mid-prefill: its helpers' shadows stay put
            return False
        if self.cfg.image_pin and q.pinned:              # an image prompt before its prefill completed stays put
            self.stats["image_moves_refused"] = self.stats.get("image_moves_refused", 0) + 1
            return False
        need_bytes = (len(self.kv.block_table(q.rid)) * self.kv.paged[0].spec.bytes_per_page
                      + len(self.kv.window_table(q.rid)) * self.kv.window[0].spec.bytes_per_page)
        if need_bytes > budget[0]:
            return False
        cands = [dst] if dst is not None else sorted(
            (r for r in range(self.g.n_ranks) if r != exclude), key=lambda r: -self.kv.free_pages(r))
        for r in cands:
            if len(self.on_rank(r)) >= self.seats_per_rank():
                continue
            ops = self.kv.migrate(q.rid, r, plan.step, q.num_computed)
            if ops is None:
                continue
            q.rank = r
            q.skip_until_step = plan.step + 1
            q.n_migrate += 1
            plan.copies.extend(ops)
            plan.migrated.append(q.rid)
            budget[0] -= need_bytes
            self.stats["migrations"] += 1
            self.stats["copy_bytes"] += need_bytes
            return True
        return False

    # -- 2. rebalance
    def _maybe_rebalance(self, plan: StepPlan, budget: list) -> None:
        dec = [0.0] * self.g.n_ranks
        for q in self.running:
            if q.phase == DECODE:
                dec[q.rank] += self._decode_load(q)
        mean = sum(dec) / len(dec)
        if mean <= 0:
            return
        hi, lo = max(range(len(dec)), key=dec.__getitem__), min(range(len(dec)), key=dec.__getitem__)
        if (dec[hi] - dec[lo]) / mean < self.cfg.rebalance_threshold:
            self.imbalance_steps = 0
            return
        self.imbalance_steps += 1
        if self.imbalance_steps < self.cfg.rebalance_patience:
            return
        gap = dec[hi] - dec[lo]
        best, best_after = None, dec[hi] - 0.0
        for q in self.on_rank(hi):
            if q.phase != DECODE or q.skip_until_step > plan.step:
                continue
            c = self._decode_load(q)
            after = max(dec[hi] - c, dec[lo] + c)
            if after < best_after - 1e-9 and c < gap:
                best, best_after = q, after
        if best is None:
            return
        # the sequence skips this step (its pages are copied before the step's kernels) and resumes on `lo`
        if self._move(best, plan, budget, exclude=hi, dst=lo):
            self._unschedule(plan, best.rid)
            self.imbalance_steps = 0

    # -- 3. admission
    def _admit(self, now: float, plan: StepPlan, budget: list) -> None:
        n_admit = 0
        while self.waiting and n_admit < self.cfg.max_waiting_admit_per_step:
            if len(self.running) >= self.g.max_seqs:
                return
            q = self.waiting[0]
            q.prefill_end = q.num_tokens                 # prompt, or prompt + outputs after a preemption
            hashes = q.block_hashes(q.prefill_end)
            hits = self.kv.lookup(hashes)
            res, wait = None, False
            for r, src in self._placement_order(q, hits):
                if src is not None:
                    cb = (hits[src] - hits[r]) * self.kv.paged[0].spec.bytes_per_page
                    # a copy beyond what is left of this step's budget waits for the next step (one step of delay)
                    # instead of recomputing the prefix there (its whole prefill, ~1,000x the copy: a fan-out of
                    # 112 calls over 8 shared prompts recomputed 275K tokens). One copy larger than the whole budget
                    # goes alone, in a step whose budget is untouched.
                    if cb > budget[0] and budget[0] < self.cfg.max_copy_bytes_per_step:
                        wait = True
                        self.stats["copy_over_budget"] += 1
                        continue
                res = self.kv.admit(q.rid, r, hashes, q.prefill_end, plan.step, copy_from=src)
                if res is not None:
                    break
            if res is None:
                self.stats["copy_waits" if wait else "rejected_admit"] += 1
                return                                   # head of line waits (FCFS)
            self.waiting.popleft()
            n_admit += 1
            q.rank, q.phase, q.admit_time = res.rank, PREFILL, now
            q.pf_progress_time = now
            q.home_rank = max(range(len(hits)), key=lambda i: hits[i]) if max(hits) > 0 else None
            q.cached_tokens, q.replay_start = res.cached_tokens, res.replay_start
            q.num_computed = res.replay_start            # the bounded replay re-runs [replay_start, cached_tokens)
            q.hit_tokens_total += res.cached_tokens
            if res.copied_blocks:
                q.copied_tokens_total += res.copied_blocks * self.g.page_tokens
                budget[0] -= res.copied_blocks * self.kv.paged[0].spec.bytes_per_page
                self.stats["prefix_copies"] += 1
                self.stats["copy_bytes"] += res.copied_blocks * self.kv.paged[0].spec.bytes_per_page
            plan.copies.extend(res.copies)
            self.take_uid(q)
            self.running.append(q)
            self.prefilling.add(q)
            self.stats["admitted"] += 1

    def _placement_order(self, q: Request, hits: list[int]):
        """Ranks to try, best first, each with the rank to copy a longer cached prefix from (or None)."""
        R = self.g.n_ranks
        seats = [len(self.on_rank(r)) < self.seats_per_rank() for r in range(R)]
        mode = self.cfg.placement
        if mode == "round_robin":
            order = [(self.rr + i) % R for i in range(R)]
            self.rr += 1
            return [(r, None) for r in order if seats[r]]
        if mode == "affinity":
            loads = self.rank_loads()
            order = sorted(range(R), key=lambda r: (-hits[r], loads[r]))
            return [(r, None) for r in order if seats[r]]
        # Balanced: the decode load lasts the whole call (~400 steps), a prefill backlog only until it drains (and
        # prefill stealing evens it out), so the projected DECODE load decides; backlog and copy size break near-ties.
        dec = [0.0] * R
        back = [0.0] * R
        for x in self.running:
            dec[x.rank] += self._decode_load(x)
            back[x.rank] += self._backlog_us(x)
        add = self.cost.decode_seq_rank_us(q.prefill_end, self.g.rows_per_decode_seq)
        best_src = max(range(R), key=lambda r: hits[r])
        cands = []
        for r in range(R):
            if not seats[r]:
                continue
            src = best_src if hits[best_src] - hits[r] >= self.cfg.migrate_min_blocks else None
            eff = hits[src] if src is not None else hits[r]
            _, cached, replay = self.kv.hit_after_cap(eff, q.prefill_end)
            peak = max(dec[i] + (add if i == r else 0.0) for i in range(R))
            pf = back[r] + self.cost.prefill_rank_us(replay, q.prefill_end - replay)
            cands.append([peak, pf, src is not None, r, src])
        if not cands:
            return []
        best = min(c[0] for c in cands)
        tol = self.cfg.placement_tol * max(best, 1.0)
        cands.sort(key=lambda c: (c[0] > best + tol, c[1], c[2], c[0], c[3]))
        return [(c[3], c[4]) for c in cands]

    # -- 3b. prefill stealing
    def _slack(self, base_us: float, backlog_us: float, frac: float | None = None) -> float:
        """Prefill time allowed on top of the decode-only step `base_us`."""
        want = max(self.cfg.min_slack_us, backlog_us / self.cfg.backlog_drain_steps)
        frac = self.cfg.max_slack_frac if frac is None else frac
        cap = max((1.0 + frac) * base_us, self.cfg.prefill_floor_us) - base_us
        if base_us <= 0.0:                  # nothing decodes: no stream to protect, fill the step
            return cap
        return max(min(want, cap), 0.0)

    def _steal(self, plan: StepPlan, budget: list) -> None:
        R = self.g.n_ranks
        for _ in range(self.cfg.max_steals_per_step):
            back = [0.0] * R
            for q in self.prefilling:
                back[q.rank] += self._backlog_us(q)
            base = self.cost.step(plan.loads).total_us
            slack = self._slack(base, sum(back))
            idle = [r for r in range(R) if back[r] < 0.5 * slack and len(self.on_rank(r)) < self.seats_per_rank()]
            donor = max(range(R), key=back.__getitem__)
            if not idle or back[donor] < self.cfg.steal_min_backlog_steps * slack:
                return
            # not-yet-started prefills of the donor (only their cached prefix is copied), the cheapest copy first,
            # keeping the one the donor works on now
            cands = [q for q in self.prefilling if q.rank == donor and not (self.cfg.image_pin and q.pinned)
                     and q.num_computed == q.replay_start and q.skip_until_step <= plan.step]
            order = self._prefill_sorted(cands, plan.step)
            cands = order[1:] if len(order) == sum(1 for q in self.prefilling if q.rank == donor) else order
            if not cands:
                return
            q = min(cands, key=lambda x: len(self.kv.block_table(x.rid)))
            dec = [0.0] * R
            for x in self.running:
                dec[x.rank] += self._decode_load(x)
            c = self._decode_load(q)
            ok = [r for r in idle if dec[r] + c <= max(dec) * (1.0 + self.cfg.steal_decode_tol)]
            if not ok:
                return
            dst = min(ok, key=lambda r: (back[r], dec[r]))
            if not self._move(q, plan, budget, exclude=donor, dst=dst):
                return
            self.stats["steals"] = self.stats.get("steals", 0) + 1

    def _prefill_sorted(self, qs: list, step: int, now: float = 0.0) -> list:
        if self.cfg.prefill_order == "fcfs":
            return sorted(qs, key=lambda q: (q.admit_time or 0.0, str(q.rid)))
        a = self.cfg.prefill_aging_tok_per_s
        small = self.cfg.small_prefill_tokens
        # short prefills (the follow-up calls of a conversation: a few hundred to a few thousand new tokens after a cached
        # prefix) always go first: aging alone let a long cold prompt that had waited ~1.5 s overtake them and then fill
        # the rank's whole per-step room for all its chunks (the replay's < 4K follow-ups: TTFT p95 20.8 s vs 0.79 s)
        if self.cfg.long_order == "fcfs":              # long prefills oldest first (a startup backlog drains in order)
            return sorted(qs, key=lambda q: (q.prefill_left > small,
                                             (q.admit_time or 0.0) if q.prefill_left > small else q.prefill_left,
                                             str(q.rid)))
        return sorted(qs, key=lambda q: (q.prefill_left > small,
                                         q.prefill_left - a * (now - (q.admit_time or now)), str(q.rid)))

    # -- 4. prefill quota
    def _fill_prefill(self, plan: StepPlan, loads: list, now: float = 0.0) -> None:
        if not self.prefilling:
            return
        pend = {r: self._prefill_sorted([q for q in self.prefilling if q.rank == r and q.skip_until_step <= plan.step],
                                        plan.step, now)
                for r in range(self.g.n_ranks)}
        if not any(pend.values()):
            return
        base = self.cost.step(loads).total_us
        # the drain estimate counts the shared side of prefill too (0 with the MoE(T) curve; the ceiling's per-token cost)
        backlog = sum(self._backlog_us(q) + self.cost.prefill_shared_us(q.prefill_left) for v in pend.values() for q in v)
        small_ = self.cfg.small_prefill_tokens
        starving_long = any(q.prefill_left > small_ and now - (q.admit_time if q.admit_time is not None else now)
                            >= self.cfg.starve_wait_s for v in pend.values() for q in v)
        slack = self._slack(base, backlog, max(self.cfg.max_slack_frac, self.cfg.starve_slack_frac) if starving_long else None)
        fixed = self.cost.mixed_fixed_us() if hasattr(self.cost, "mixed_fixed_us") else 0.0
        # (never above the largest mixed step allowed, or the bank could never fire)
        amort = min(fixed * (1.0 + self.cfg.amortize_mixed), self.cfg.max_mixed_step_us)
        banked = fixed >= self.cfg.amortize_min_fixed_us and slack < amort
        stop_rows = None
        target = self.cfg.mixed_target_tokens
        lo_, hi_ = self.cfg.mixed_adapt_lo, self.cfg.mixed_adapt_hi
        if hi_ > lo_ > 0:
            n_dec = sum(1 for q in self.running if q.phase == DECODE)
            self.lam = min(1.0, max(self.cfg.mixed_adapt_min, (n_dec - lo_) / (hi_ - lo_)))
        lam = self.lam
        if target > 0 and base > 0.0 and any(q.phase == DECODE for q in self.running):
            cap_ = self.g.max_prefill_tokens_per_rank
            if self.cfg.mixed_target_total:
                pending = sum(q.prefill_left for v in pend.values() for q in v)
            else:
                pending = sum(min(cap_, sum(min(q.prefill_left, cap_) for q in v)) for v in pend.values())
            starving = any(now - (q.pf_progress_time if q.pf_progress_time is not None else now)
                           > lam * self.cfg.mixed_target_wait_s for v in pend.values() for q in v)
            until_done = self.cfg.backlog_drain_until == "finished"
            draining = any((until_done or q.first_chunk_time is None) and q.prefill_left > self.cfg.small_prefill_tokens
                           and now - (q.admit_time if q.admit_time is not None else now) >= self.cfg.backlog_drain_wait_s
                           for v in pend.values() for q in v)
            fw = self.cfg.mixed_target_full_wait_s
            full = fw < float("inf") and any(
                sum(min(q.prefill_left, cap_) for q in v) >= cap_
                and any(now - (q.pf_progress_time if q.pf_progress_time is not None else now) >= fw for q in v)
                for v in pend.values())
            big = self.cfg.cp_force_decode_steps >= 0 and any(self._cp_forced(q) for v in pend.values() for q in v)
            if pending < lam * target and not starving and not draining and not full and not big:
                return
            floor0 = int(round(lam * self.cfg.mixed_min_decode_steps))
            floor = min(floor0, self.cfg.cp_force_decode_steps) if big else floor0
            if self.cfg.mixed_target_fill and self.dec_since_mixed < floor and not draining:
                return
            banked, stop_rows = False, (None if self.cfg.mixed_target_fill else target)
            limit = base + self.cfg.max_mixed_step_us
        elif banked:
            # The step's fixed cost of carrying prefill exceeds what this step may spend (eager mixed steps: ~41 ms):
            # prefill in fewer, larger steps. Bank the decode-only steps' slack; a mixed step runs once the bank covers
            # the fixed cost plus `amortize_mixed` times it in rows. Nothing decoding: nothing to protect, fill the step.
            if base <= 0.0 or not any(q.phase == DECODE for q in self.running):
                budget = self.cfg.max_mixed_step_us
            else:
                self.pf_credit = min(self.pf_credit + slack, self.cfg.max_mixed_step_us)
                starving = any(now - (q.pf_progress_time if q.pf_progress_time is not None else now)
                               > self.cfg.max_prefill_wait_s for q in self.prefilling)
                if self.pf_credit < amort and not starving:
                    return
                # a starving prefill fires the step with what is banked, at least enough to amortize the fixed cost twice
                budget = self.pf_credit if self.pf_credit >= amort else max(self.pf_credit, 3.0 * fixed)
            limit = base + budget
        elif not (target > 0 and base > 0.0 and any(q.phase == DECODE for q in self.running)):
            limit = base + slack
        sched: dict = {}                                  # rid -> [start, n]
        done = set()
        cap = self.g.max_prefill_tokens_per_rank
        rank_pf = [0] * self.g.n_ranks
        tail_cap = self.g.max_tail_rows_per_rank
        rank_tail = [ld.dec_rows for ld in loads]        # rows running layers 21..39 + the head this step, per rank
        any_sched = False
        if self.cfg.cp_prefill:                           # (async mode too: _cp_prefill keeps the D13 guarantees)
            any_sched = self._cp_prefill(plan, loads, pend, limit, rank_pf, rank_tail, cap, now)
            if self.cfg.async_mode:
                # a CP-size prompt whose chunk is still in flight sits this step out (back-to-back mixed steps while
                # draining): an owner-only chunk now would keep it unsettled, and so out of CP, for good
                for r in range(self.g.n_ranks):
                    pend[r] = [q for q in pend[r] if not (q.prefill_left >= self.cfg.cp_min and not q.score
                                                          and not (self.cfg.image_pin and q.images)
                                                          and q.last_pf_step > plan.step - self.cfg.cp_async_settle_steps)]
        # anti-starvation share: the oldest longer prefill per rank gets up to large_min_rows ahead of the short ones
        big, big_left = {}, {}
        share_us = self.cfg.large_share_frac * max(0.0, limit - base)    # time the shares may take, all ranks
        small = self.cfg.small_prefill_tokens
        if self.cfg.large_min_rows > 0:
            for r in range(self.g.n_ranks):
                longs = [q for q in pend[r] if q.prefill_left > small and q.cp_shadows == {}
                         and now - (q.admit_time if q.admit_time is not None else now) >= self.cfg.large_share_after_s]
                if longs and pend[r][0] not in longs:
                    big[r] = min(longs, key=lambda q: (q.admit_time or 0.0, str(q.rid)))
                    big_left[r] = self.cfg.large_min_rows
        while True:
            cand = [r for r in range(self.g.n_ranks) if r not in done and pend[r]]
            if not cand:
                break
            if stop_rows is not None and sum(rank_pf) >= stop_rows:
                break
            r = min(cand, key=lambda x: sum(self.cost.rank_sides(loads[x])))
            share = big_left.get(r, 0) > 0 and big[r] in pend[r]
            q = big[r] if share else pend[r][0]
            st = sched.get(q.rid, [q.num_computed, 0])
            pos = st[0] + st[1]
            left = q.prefill_end - pos
            room = min(cap - rank_pf[r], self.g.rows_cap - loads[r].dec_rows - rank_pf[r])
            if room <= 0:
                done.add(r)
                continue
            qa = self.cfg.quantum
            n = min(left, room, qa - (pos % qa) if pos % qa else qa)
            if share:
                n = min(n, big_left[r])
            cut = (self.cfg.cp_prefill and self.cfg.cp_replay_first and not q.score
                   and not (self.cfg.image_pin and q.images)
                   and pos < q.cached_tokens and q.cached_tokens % self.g.page_tokens == 0
                   and q.prefill_end - q.cached_tokens >= self.cfg.cp_min)
            if cut:
                n = min(n, q.cached_tokens - pos)        # the replay rows only; the rest goes CP next mixed step
            # tail rows: a chunk's last 128 rows (every row of a score chunk) run layers 21..39 and the drafter's head;
            # many replays (128 rows each) in one step would pass the engine's tail bucket (HSD: 1,536 per rank)
            had = st[1]
            tail_add = n if q.score else min(had + n, SLIDING_WINDOW) - min(had, SLIDING_WINDOW)
            if rank_tail[r] + tail_add > tail_cap:
                if share:                                 # the share does not fit: the short ones may
                    big_left[r] = 0
                    continue
                done.add(r)
                continue
            trial = loads[r].copy()
            trial.add_prefill(pos, n)
            new = self.cost.step([trial if i == r else loads[i] for i in range(self.g.n_ranks)]).total_us
            if share:
                cur = self.cost.step(loads).total_us
                if cur <= 0.0:                            # an empty step's fixed cost is not the share's: probe 1 row
                    probe = loads[r].copy()
                    probe.add_prefill(pos, 1)
                    cur = self.cost.step([probe if i == r else loads[i] for i in range(self.g.n_ranks)]).total_us
                inc = new - cur
                if inc > share_us or new > limit:
                    big_left[r] = 0
                    continue
            if new > limit and any_sched:
                done.add(r)
                continue
            last = pos + n >= q.prefill_end
            # allocate now (not after the loop): window pages are not reserved at admission, so chunks of two prompts on
            # one rank each passing a can_allocate check could together overrun the window pool (seen with forced CP:
            # the shadows' scratch windows); an accepted chunk is never rolled back, so this is the same decision
            if not self.kv.allocate(q.rid, st[0], pos + n,
                                    min(pos + n + (self.g.draft_window_tokens if last else 0), self.g.max_model_len)):
                if share:
                    big_left[r] = 0
                    continue
                done.add(r)
                continue
            if share:
                big_left[r] -= n
                share_us -= inc
            loads[r] = trial
            rank_pf[r] += n
            st[1] += n
            sched[q.rid] = st
            rank_tail[r] += tail_add
            q.pf_progress_time = now
            if q.first_chunk_time is None:
                q.first_chunk_time = now
            any_sched = True
            if pos + n >= q.prefill_end or (cut and pos + n >= q.cached_tokens):
                pend[r].remove(q)
        if banked and any_sched:
            self.pf_credit = max(0.0, self.pf_credit - (self.cost.step(loads).total_us - base))
        for rid, (start, n) in sched.items():
            q = self.reqs[rid]
            ok = self.kv.allocate(rid, start, start + n,
                                  min(start + n + (self.g.draft_window_tokens if start + n >= q.prefill_end else 0),
                                      self.g.max_model_len))
            assert ok, rid
            last = start + n >= q.prefill_end
            plan.work.append(SeqWork(rid, q.rank, "prefill", start, n, q.cached_tokens, q.replay_start,
                                     last_chunk=last, score=q.score, forced=q.forced, uid=q.uid,
                                     images=q.images_in(start, start + n)))
            q.last_pf_step = plan.step
            if self.cfg.async_mode:                     # prefill progress is deterministic: advance at schedule time
                q.num_computed = start + n
                if last:
                    q.phase, q.first_pending = DECODE, True
                    self.prefilling.discard(q)
                    self.kv.shrink_reservation(q.rid)

    def _cp_force_min(self) -> int:
        f, lo = self.cfg.cp_force_min, self.cfg.mixed_adapt_force_min
        if f > 0 and 0 < lo < f and self.lam < 1.0:
            return int(lo + self.lam * (f - lo))
        return f

    def _cp_forced(self, q: Request) -> bool:
        """cp_force_min: a prompt with that many tokens left (or one already forced) goes CP without the light gate."""
        f = self._cp_force_min()
        return (self.cfg.cp_prefill and f > 0 and not q.score and not (self.cfg.image_pin and q.images)
                and (q.prefill_left >= f or (q.cp_forced and q.prefill_left >= self.cfg.cp_min)))

    def _cp_prefill(self, plan: StepPlan, loads: list, pend: dict, limit: float, rank_pf: list, rank_tail: list,
                    cap: int, now: float) -> bool:
        """Context-parallel prefill chunks (engine/cp.py), before the per-rank quota: a prompt with
        >= cp_min tokens left, a page-aligned start past its bounded replay, and >= 2 other ranks with less than a chunk
        of their own pending prefill gets a chunk of N rows split into 4 contiguous sub-chunks (helpers in rank order,
        64-aligned, the owner the highest, >= 128 rows). The owner's entry is the prompt's ordinary chunk cut to its
        sub-chunk; each helper's is a SHADOW sequence (replica pages to the chunk end, scratch window / state). Copies:
        the replica catch-up (prefix / earlier chunks, owner -> helper) and the lowest helper's window halo."""
        from interfaces import CPRole

        R, P, W = self.g.n_ranks, self.g.page_tokens, self.g.window
        any_sched = False
        forced = self._cp_forced
        big_min = self.cfg.cp_force_min
        if self.cfg.cp_force_order == "srpt":
            hold, keep = self.cfg.cp_force_srpt_hold, self.cfg.cp_force_srpt_min_left
            # a prompt already in CP counts hold x its tokens left, and below `keep` tokens left it is never overtaken
            fkey = lambda q: ((-1.0 if q.prefill_left < keep else q.prefill_left * hold) if q.cp_shadows  # noqa: E731
                              else float(q.prefill_left),)
        elif self.cfg.cp_force_order == "tiered":   # < cp_force_srpt_min_left tokens left first, then the huge ones;
            huge = self.cfg.cp_force_srpt_min_left or 131072      # oldest first in each tier
            fkey = lambda q: (int(q.prefill_left >= huge), q.admit_time or 0.0)  # noqa: E731
        elif self.cfg.cp_force_order == "split":     # < cp_force_min (forced only at low load): shortest first; big
            fkey = lambda q: ((0, q.prefill_left) if q.prefill_left < big_min  # noqa: E731
                              else (1, q.admit_time or 0.0))  # ones after them, oldest first
        else:
            fkey = lambda q: (q.admit_time or 0.0,)  # noqa: E731
        # async mode (D13): the replica catch-up / halo copies read the pages of the prompt's previous chunk; copies run
        # before the step's kernels but are not ordered after the in-flight step, so that chunk must be settled
        # (scheduled <= step - 2: update(step - 2) ran before this schedule)
        gap = self.cfg.cp_async_settle_steps
        settled = (lambda q: q.last_pf_step <= plan.step - gap) if self.cfg.async_mode else (lambda q: True)  # noqa: E731
        cands = []
        for r in range(R):
            el = [q for q in pend[r] if q.prefill_left >= self.cfg.cp_min and not q.score
                  and not (self.cfg.image_pin and q.images)
                  and q.num_computed >= q.cached_tokens and q.num_computed % P == 0 and settled(q)]
            if not el:
                continue
            fq = [q for q in el if forced(q)]
            cands.append(min(fq, key=lambda q: (fkey(q), str(q.rid))) if fq else el[0])
        for q in sorted(cands, key=lambda q: ((0, fkey(q)) if forced(q) else (1, q.prefill_left), str(q.rid))):
            owner = q.rank
            helpers = [i for i in range(R) if i != owner]
            thr = self.cfg.cp_light_frac * q.prefill_left if self.cfg.cp_light_frac > 0 else cap
            light = [i for i in helpers if sum(x.prefill_left for x in pend[i] if x is not q) < thr]
            if len(light) < 2 and not forced(q):
                continue
            # short-first holds on every rank: the CP chunk takes only the room left after each rank's short prefills
            # (a forced prompt: at least cp_force_room_frac of the chunk room; the short ones get the rest)
            small = self.cfg.small_prefill_tokens
            short = [sum(min(x.prefill_left, small) for x in pend[i] if x is not q and x.prefill_left <= small)
                     for i in range(R)]
            if forced(q):
                keep = int((1.0 - self.cfg.cp_force_room_frac) * cap)
                short = [min(x, keep) for x in short]
            room = min(min(cap - rank_pf[i] - short[i], self.g.rows_cap - loads[i].dec_rows - rank_pf[i] - short[i])
                       for i in range(R))
            per = (room // P) * P
            start, left = q.num_computed, q.prefill_end - q.num_computed
            N = min(left, R * per)
            if N < left:
                N = (N // self.cfg.quantum) * self.cfg.quantum
            while N >= R * SLIDING_WINDOW:         # every sub-chunk >= 128 rows (X2: the halo of a sub-chunk comes
                                                   # from the previous sub-chunk's rank alone)
                h = ((N // R) // P) * P
                owner_n = N - (R - 1) * h
                if owner_n <= per:
                    break
                N -= self.cfg.quantum if N < left else P
            else:
                continue
            if h < SLIDING_WINDOW or owner_n < SLIDING_WINDOW:
                continue
            subs = [(ri, start + j * h, h) for j, ri in enumerate(helpers)] + [(owner, start + (R - 1) * h, owner_n)]
            end = start + N
            last = end >= q.prefill_end
            if rank_tail[owner] + min(owner_n, SLIDING_WINDOW) > self.g.max_tail_rows_per_rank:
                continue
            trials = [ld.copy() for ld in loads]
            for ri, s0, n0 in subs:
                trials[ri].add_prefill(s0, n0)
                if ri != owner:                          # layers 21..39 + the head: the owner's last rows only
                    trials[ri].pf_tail_rows -= min(n0, SLIDING_WINDOW)
                    trials[ri].pf_cand_pos = loads[ri].pf_cand_pos
            new = self.cost.step(trials).total_us + self.cfg.cp_exchange_us
            if new > limit and any_sched:
                continue
            owner_start = subs[-1][1]
            wend = min(end + (self.g.draft_window_tokens if last else 0), self.g.max_model_len)
            if not self.kv.can_allocate(q.rid, owner_start, end, wend):
                continue
            # shadows: open on the prompt's first CP chunk, replica pages up to `end`, scratch window for the sub-chunk
            opened = []
            ok = True
            for j, ri in enumerate(helpers):
                if ri not in q.cp_shadows:
                    if not self.free_shadow_uids:
                        ok = False
                        break
                    key = ("cp", q.rid, ri)
                    if not self.kv.open_shadow(key, ri, q.replay_start):
                        ok = False
                        break
                    q.cp_shadows[ri] = (key, self.free_shadow_uids.popleft())
                    opened.append(ri)
                key = q.cp_shadows[ri][0]
                s0, n0 = subs[j][1], subs[j][2]
                self.kv.trim_window(key, max(q.replay_start, s0 - (W - 1)))
                if not self.kv.can_allocate(key, s0, end, s0 + n0):
                    ok = False
                    break
            if not ok:
                for ri in opened:
                    key, uid = q.cp_shadows.pop(ri)
                    self.kv.free(key)
                    self.free_shadow_uids.append(uid)
                continue
            assert self.kv.allocate(q.rid, owner_start, end, wend), q.rid
            osk = self.kv.seqs[q.rid]
            for j, ri in enumerate(helpers):
                key, suid = q.cp_shadows[ri]
                s0, n0 = subs[j][1], subs[j][2]
                assert self.kv.allocate(key, s0, end, s0 + n0), key
                hsk = self.kv.seqs[key]
                for b in range(hsk.cp_have, start // P):       # replica catch-up: blocks the helper lacks below the chunk
                    # X1b: a block of the previous CP chunk comes from the rank that computed it; the prefix and
                    # owner-only chunks from the owner
                    src, spage = owner, osk.pages[b]
                    for pr, ps0, pn0 in q.cp_prev_subs:
                        if ps0 // P <= b < (ps0 + pn0) // P and pr != ri:
                            if pr != owner:
                                src, spage = pr, self.kv.seqs[q.cp_shadows[pr][0]].pages[b]
                            break
                    plan.copies.append(CopyOp("paged", src, spage, ri, hsk.pages[b], plan.step))
                    self.stats["cp_copy_pages"] += 1
                hsk.cp_have = (s0 + n0) // P                  # after this step: its own + the lower sub-chunks (X1a)
                if j == 0 and start > q.replay_start:          # the lowest helper's window halo, owner -> helper
                    wt = self.g.window_page_tokens
                    for b in range(max(q.replay_start, start - (W - 1)) // wt, (start - 1) // wt + 1):
                        if b in osk.window and b in hsk.window:
                            plan.copies.append(CopyOp("window", owner, osk.window[b], ri, hsk.window[b], plan.step))
                role = CPRole(q.uid, q.rid, owner, subs, j)
                plan.work.append(SeqWork(key, ri, "cp", s0, n0, 0, q.replay_start, uid=suid, cp=role,
                                         images=q.images_in(s0, s0 + n0)))
                rank_pf[ri] += n0
            plan.work.append(SeqWork(q.rid, owner, "prefill", owner_start, owner_n, q.cached_tokens, q.replay_start,
                                     last_chunk=last, score=False, forced=q.forced, uid=q.uid,
                                     cp=CPRole(q.uid, q.rid, owner, subs, R - 1),
                                     images=q.images_in(owner_start, end)))
            rank_pf[owner] += owner_n
            rank_tail[owner] += min(owner_n, SLIDING_WINDOW)
            for i in range(R):
                loads[i] = trials[i]
            q.pf_progress_time = now
            if q.first_chunk_time is None:
                q.first_chunk_time = now
            pend[owner].remove(q)
            q.cp_prev_subs = list(subs)
            q.last_pf_step = plan.step
            q.cp_forced = q.cp_forced or forced(q)
            if self.cfg.async_mode:                       # as an owner-only chunk: progress advanced at schedule time
                q.num_computed = end
                if last:                                  # (its shadows go when this step settles: update())
                    q.phase, q.first_pending = DECODE, True
                    self.prefilling.discard(q)
                    self.kv.shrink_reservation(q.rid)
            self.stats["cp_chunks"] += 1
            self.stats["cp_rows"] += N
            any_sched = True
        return any_sched

    def _drop_shadows(self, q: Request) -> None:
        for ri, (key, uid) in list(q.cp_shadows.items()):
            self.kv.free(key, defer_until=self._defer())
            self.free_shadow_uids.append(uid)
        q.cp_shadows.clear()
        q.cp_prev_subs = []

    # ------------------------------------------------------------------------------------------ results
    def update(self, plan: StepPlan, accepted: dict, now: float = 0.0) -> list[Request]:
        """Apply a step's results. accepted[rid] = tokens appended by a decode row group (1 .. k + 1: the accepted
        drafts + the bonus token); prefill chunks need no entry. Returns the requests that finished."""
        finished = []
        for w in plan.work:
            q = self.reqs.get(w.rid)
            if q is None or q.phase == FINISHED or q not in self.running:
                continue
            if w.kind == "decode":
                a = int(accepted[w.rid])
                assert 1 <= a <= w.n
                a = min(a, q.max_new_tokens - q.num_output)
                q.num_computed += a
                q.num_output += a
                if self.cfg.async_mode:
                    q.inflight -= 1
            elif self.cfg.async_mode:                    # already advanced when scheduled
                if w.last_chunk:
                    q.num_output += 1
                    q.first_pending = False
                    q.first_token_time = now
                    if q.cp_shadows:                      # a CP prompt's last chunk settled: its shadows go (no
                        self._drop_shadows(q)             # newer plan holds them: the prompt decodes there)
            else:
                q.num_computed = w.start + w.n
                if w.last_chunk:
                    q.phase = DECODE
                    self.prefilling.discard(q)
                    q.num_output += 1                   # the first token, sampled from the last prompt row
                    q.first_token_time = now
                    self.kv.shrink_reservation(q.rid)
                    if q.cp_shadows:
                        self._drop_shadows(q)
            # async: a prefill sequence's num_computed already includes chunks still in flight -> register hashes and
            # release window pages only up to what this settled step computed (D13: no window page released while a
            # pending step can read it; kernels/rowexp flags violations as rx.err)
            settled = (w.start + w.n) if (self.cfg.async_mode and w.kind == "prefill") else q.num_computed
            self.kv.commit(q.rid, settled, q.hash_fn)
            if q.num_output >= q.max_new_tokens:
                self.prefilling.discard(q)
                q.phase, q.finish_time = FINISHED, now
                self.kv.free(q.rid, defer_until=self._defer())
                self.running.remove(q)
                self.free_uid(q)
                finished.append(q)
                self.stats["finished"] += 1
        self.kv.release_copies(plan.step)
        self.kv.flush_deferred(plan.step)
        if finished:
            self._req_log(finished, now)
        for q in finished:
            del self.reqs[q.rid]
        return finished

    def _req_log(self, finished: list, now: float) -> None:
        """$PHYS_REQ_LOG=<path>: one JSON line per finished request, numbers only (no ids, no tokens): uncached prompt
        tokens, and its TTFT split into the wait for admission, the wait for its first chunk and its prefill. Never
        fails a step."""
        import json as _json
        import os as _os

        path = _os.environ.get("PHYS_REQ_LOG")
        if not path or getattr(self, "_req_off", False):
            return
        try:
            if getattr(self, "_req_f", None) is None:
                _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
                self._req_f = open(path, "a", buffering=1)
            for q in finished:
                a, ad = q.arrival, q.admit_time
                fc, ft = q.first_chunk_time, q.first_token_time
                r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
                self._req_f.write(_json.dumps(dict(
                    t=r3(now), prompt=q.prompt_len, uncached=q.prompt_len - min(q.hit_tokens_total, q.prompt_len),
                    out=q.num_output, rank=q.rank,
                    home=(None if q.home_rank is None else q.home_rank == q.rank), hit=q.hit_tokens_total,
                    copied=q.copied_tokens_total,
                    admit_wait=r3(None if ad is None else ad - a),
                    first_chunk_wait=r3(None if fc is None or ad is None else fc - ad),
                    prefill_s=r3(None if ft is None or fc is None else ft - fc),
                    ttft=r3(None if ft is None else ft - a), call_s=r3(now - a))) + "\n")
        except Exception:  # noqa: BLE001
            self._req_off = True
