"""PhysScheduler: vLLM 0.32's scheduler interface on top of serve/scheduler.py (RankScheduler), sequence-owned KV.

Plugged in with `--scheduler-cls adapter.phys_scheduler.PhysScheduler` (SchedulerConfig.scheduler_cls accepts a qualname;
vllm/config/scheduler.py get_scheduler_cls). Nothing in the vLLM package is patched.

What is vLLM's and what is ours:
  * vLLM (inherited from Scheduler, unchanged): add_request bookkeeping, update_from_output (stop checks via check_stop:
    EOS / stop token ids / max_tokens / max_model_len; spec-decode accounting num_computed -= rejected; grammar
    accept_tokens; logprobs slicing; EngineCoreOutputs, events, stats), finish_requests (aborts, stop strings from the
    detokenizer), get_grammar_bitmask (StructuredOutputManager), update_draft_token_ids (grammar validation of drafts),
    make_stats, the request queues used for counts.
  * ours: schedule() -- RankScheduler places, admits, preempts, migrates and fills prefill per rank; the plan becomes a
    vLLM SchedulerOutput (num_scheduled_tokens = rows, k spec placeholders per decode group, new / cached request data)
    plus `so.phys` (adapter/bridge.py PhysStep: one StepBatch per rank, row token ids, new requests, frees).
  * self.kv_cache_manager is a facade (PhysKVFacade): the calls vLLM's scheduler makes outside schedule() (free,
    usage, prefix-cache stats, estimate_cached_tokens, events) are answered from the OwnedKVManager. vLLM's own
    KVCacheManager is built on the executor's dummy spec (one small 64-token group) and only serves EngineCore's
    start-up introspection (its single_type_managers).

Lifecycle mapping (vLLM status <-> RankScheduler phase):
  add_request               vLLM WAITING (queue mirror) -> rs.add_request (after the grammar compiled, if any)
  rs admits                 vLLM RUNNING, num_computed_tokens = replay_start (bounded replay after a prefix hit),
                            prefill_stats = the hit (prompt_tokens_details.cached_tokens)
  rs schedules work         num_scheduled_tokens[rid] = chunk rows (prefill) or k + 1 (decode, spec = k x -1, or the
                            grammar-validated drafts for structured outputs)
  update_from_output        vLLM appends tokens + stop checks first, then rs.update(plan, accepted) with accepted =
                            tokens vLLM kept this step (so commits never pass the token stream vLLM's block hashes cover)
  vLLM finishes / aborts    rs.abort(rid) (deferred until rs.update ran when it happens inside update_from_output)
  rs preempts               vLLM _preempt_request (PREEMPTED, num_computed_tokens = 0, back to the waiting mirror);
                            rs keeps the request and re-admits it as a recompute of prompt + outputs
  rs migrates / steals      invisible to vLLM (the sequence skips a step; copies run before the next step's kernels)

Images (PHYS_MM=1 with a worker that supports them, adapter/mm.py): a request's images get content keys (blake2b of
vLLM's processed tensors, hashed off the engine loop thread before the request enters the rank scheduler). The rank
scheduler gets them as Request.images = ((offset, length, content key), ...) and salts the KV block keys with them
(serve/scheduler.py ImageBlockKeys: stock vLLM 0.32 names an image by the client's uuid when one is given). MMPlanner
turns every step's prefill rows into the image work of PhysStep.mm (encode / rows / drop per rank). vLLM's encoder
scheduling stays unused.

Synchronous engine loop only (EngineCore.step, max_concurrent_batches == 1): PhysExecutor.supports_async_scheduling()
is False, so the launcher passes --no-async-scheduling. The StepBatch is position-explicit; async scheduling would need
the positions of step N+1 before step N's acceptance is known (PhysAsyncScheduler below).
"""
from __future__ import annotations

import dataclasses
import gc
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import RequestStatus

from adapter import bridge, steptimes  # noqa: F401  (bridge puts serve/ on sys.path)
from adapter import mm as MM
from adapter.bridge import PhysStep, ReqInit
from adapter.force import ForcedReq, ForceSpec

from cost_model import CostModel, params_from_env  # noqa: E402  (serve/)
from interfaces import build_step_batch  # noqa: E402
from kv_manager import OwnedKVManager  # noqa: E402
from scheduler import DECODE, FINISHED, RankScheduler, Request as RSRequest, SchedConfig  # noqa: E402

logger = init_logger("vllm.phys_adapter")
CHECK = os.environ.get("PHYS_ADAPTER_CHECK", "0") == "1"


class _HashFn:
    """RankScheduler's hash_fn: vLLM's own chained block hash of block i (request.block_hashes, extra keys included).
    `private`: a per-request key space (requests that must not read the prefix cache, e.g. prompt logprobs: vLLM's
    skip_reading_prefix_cache; they neither hit nor publish cache entries). Requests with images: the rank scheduler
    salts these keys with Request.images (content keys), in one place (serve/scheduler.py ImageBlockKeys)."""

    __slots__ = ("req", "private")

    def __init__(self, req, private: bool):
        self.req, self.private = req, private

    def __call__(self, i: int):
        if self.private:
            return ("private", self.req.request_id, i)
        bh = self.req.block_hashes
        if i < len(bh):
            return bh[i]
        raise IndexError(f"phys adapter: block hash {i} of {self.req.request_id} not computed "
                         f"({len(bh)} hashes, {self.req.num_tokens} tokens)")


class _RSReq(RSRequest):
    """serve's Request with identity equality: RankScheduler keeps requests in lists and tests membership
    (`q in self.running`, `.remove(q)`); the dataclass __eq__ compares every field, which was 44 % of the host time
    per step at 256 sequences. Identity is the intended semantics (one object per rid)."""
    __eq__ = object.__eq__
    __hash__ = object.__hash__


class PhysKVFacade:
    """The KV-manager surface vLLM's Scheduler uses outside schedule(), backed by the RankScheduler's OwnedKVManager.
    Anything else is delegated to vLLM's (dummy) KVCacheManager, with a warning on first use."""

    _DELEGATE_OK = {"coordinator", "block_pool", "empty_kv_cache_blocks", "kv_cache_config", "enable_caching",
                    "block_size", "num_kv_cache_groups", "mamba_shared_prefix_checkpoint"}

    def __init__(self, sched: "PhysScheduler", stock):
        self._sched, self._stock = sched, stock
        self._prefix_stats = PrefixCacheStats() if sched.log_stats else None
        self._warned: set = set()

    # -- calls of update_from_output / finish_requests / _preempt_request / make_stats
    def free(self, request) -> None:
        self._sched._phys_free(request.request_id)

    def cache_blocks(self, request, num_computed_tokens: int) -> None:   # AsyncScheduler only; rs.update commits
        return None

    @property
    def usage(self) -> float:
        kv = self._sched.rs.kv
        used = sum(p.num_used() for p in kv.paged)
        total = sum(p.num_pages for p in kv.paged)
        return used / total if total else 0.0

    def make_prefix_cache_stats(self):
        if self._prefix_stats is None:
            return None
        s, self._prefix_stats = self._prefix_stats, PrefixCacheStats()
        return s

    def record_prefix(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        if self._prefix_stats is not None:
            self._prefix_stats.record(num_tokens=num_tokens, num_hits=num_hits, preempted=preempted)

    def take_events(self):
        return None

    def estimate_cached_tokens(self, request) -> int:
        sk = self._sched.rs.kv.seqs.get(request.request_id)
        return 0 if sk is None else sk.num_registered * self._sched.geom.page_tokens

    def get_block_ids(self, request_id: str, *a, **k):
        return ([],)

    def reset_prefix_cache(self) -> bool:
        return self._sched._phys_reset_prefix_cache()

    # -- schedule()-internal calls of the stock path (not reached: schedule() is ours); harmless answers
    def new_step_starts(self) -> None:
        return None

    def take_new_block_ids(self):
        return []

    def take_kv_cache_block_copies(self):
        return [], []

    def take_boundary_state_offloads(self):
        return {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._DELEGATE_OK and name not in self._warned:
            self._warned.add(name)
            logger.warning("phys adapter: kv_cache_manager.%s delegated to vLLM's dummy KV manager", name)
        return getattr(self._stock, name)


class PhysScheduler(Scheduler):
    async_mode = False                       # PhysAsyncScheduler: step N+1 is planned before step N's results

    def __init__(self, vllm_config, kv_cache_config, structured_output_manager, block_size: int,
                 hash_block_size: int | None = None, mm_registry=MULTIMODAL_REGISTRY, include_finished_set: bool = False,
                 log_stats: bool = False) -> None:
        super().__init__(vllm_config=vllm_config, kv_cache_config=kv_cache_config,
                         structured_output_manager=structured_output_manager, block_size=block_size,
                         hash_block_size=hash_block_size, mm_registry=mm_registry,
                         include_finished_set=include_finished_set, log_stats=log_stats)
        g = bridge.get_geometry()
        self.geom = bridge.geometry_overrides(dataclasses.replace(
            g, spec_k=self.num_spec_tokens, max_model_len=self.max_model_len, max_seqs=self.max_num_running_reqs))
        assert self.hash_block_size == self.geom.page_tokens, (
            f"phys adapter: vLLM hashes {self.hash_block_size}-token blocks, the paged pool has "
            f"{self.geom.page_tokens}-token pages (the executor's KV spec must use block_size {self.geom.page_tokens})")
        cfg = sched_config_from_env()
        if self.async_mode:
            assert hasattr(cfg, "async_mode"), "serve/scheduler.py has no SchedConfig.async_mode (D13 pending step)"
            cfg.async_mode = True
        # the same step-cost model as serve's OfflineEngine (the worker's default, $PHYS_COST overrides): the prefill
        # chunk boundaries follow from it, and the served fork matches the offline one only on the same boundaries
        self.cost_params = params_from_env(bridge.get_cost_default())
        self.rs = RankScheduler(self.geom, CostModel(self.cost_params), cfg)
        logger.info("phys adapter: cost model %s", self.cost_params)
        self._stock_kv = self.kv_cache_manager
        self.kv_cache_manager = PhysKVFacade(self, self._stock_kv)
        self._pending_add: list = []        # added to vLLM, not yet to rs (grammar still compiling)
        self._started: set = set()          # rids whose worker-side state exists (ReqInit sent)
        self._in_update = False
        self._deferred_free: set = set()
        self._preempting = False
        self._t0 = time.monotonic()
        self.force = ForceSpec()            # production's DS41H_FORCE_FILE protocol (adapter/force.py)
        self.forced: dict = {}              # rid -> ForcedReq
        self.vocab_size = self.vllm_config.model_config.get_vocab_size()
        # images (adapter/mm.py): PHYS_MM=1 and a worker that encodes them; else image requests are rejected
        self.mm = None
        self._mm_keys: dict = {}            # rid -> Future: the request's images with content keys (or an error string)
        self._mm_why_off = "multimodal inputs are not supported by the phys engine (text only)"
        if os.environ.get("PHYS_MM", "0") == "1":
            if bridge.get_image_support():
                cap = int(float(os.environ.get("PHYS_MM_CACHE_MB", "512")) * 2 ** 20)
                row_bytes = 2 * self.vllm_config.model_config.get_hidden_size()        # bf16 span rows
                self.mm = MM.MMPlanner(self.geom.n_ranks, cap, row_bytes, self.geom.page_tokens)
                self._mm_pool = ThreadPoolExecutor(1, thread_name_prefix="phys-mm-keys")
                logger.info("phys adapter: images on (encoder cache %d MiB per rank)", cap >> 20)
            else:
                self._mm_why_off = "image inputs are not supported by this phys worker (PHYS_MM=1, no image support)"
                logger.warning("phys adapter: PHYS_MM=1 but the worker does not support images: images are rejected")
        self.phys_stats = dict(steps=0, empty_steps=0, rejected=0, forced=0, gc_full=0, gc_full_ms=0.0, gc_cyclic=0,
                               cp_chunks=0, cp_helper_rows=0, grammar_held=0, grammar_hold_off=0, grammar_host_ids=0,
                               grammar_resume_no_ids=0, settled_host_t0=0)
        # ASYNC + structured outputs: vLLM 0.32 defers a step's sample_tokens (our D13 dispatch) until the previous
        # step's update_from_output whenever a scheduled structured-output request has tokens in flight, which
        # serializes the pipeline (an agentic load: 4.1 % strict-tool calls = ~5 of ~125 decoding sequences, so
        # nearly every step). PHYS_ASYNC_GRAMMAR=hold (default): an in-flight structured request skips one step instead
        # (it decodes every other step, with exactly the sync semantics: settled grammar state, validated drafts),
        # unless they are more than PHYS_ASYNC_GRAMMAR_MAX_FRAC of the decoding sequences (then vLLM's deferral).
        # =defer: vLLM's behaviour.
        self._grammar_hold = self.async_mode and os.environ.get("PHYS_ASYNC_GRAMMAR", "hold") == "hold"
        self._grammar_max_frac = float(os.environ.get("PHYS_ASYNC_GRAMMAR_MAX_FRAC", "0.34"))
        self._dev_drafts: dict = {}         # rid -> the engine's last drafts of a settled structured request (its resume)
        self._has_drafts: set = set()       # structured rids whose step results carried drafts (only those are held)
        # GC: vLLM's requests hold their whole token streams in Python lists (15M ids at 256 x 60K); every full
        # collection traverses them (~300 ms each, 7 per 300 steps = 7 ms per step). gc.freeze() (O(1)) every
        # PHYS_GC_FREEZE_EVERY steps moves the long-lived objects out of the collector's view; a full unfreeze +
        # collect every PHYS_GC_FULL_S seconds reclaims cyclic garbage that was frozen meanwhile.
        self._gc_every = int(os.environ.get("PHYS_GC_FREEZE_EVERY", "64"))
        self._gc_full_s = float(os.environ.get("PHYS_GC_FULL_S", "900"))
        self._gc_last_full = time.monotonic()
        logger.info("phys adapter: PhysScheduler on %d ranks, k=%d, max_seqs %d (%d per rank), paged pool %s pages per "
                    "rank, prefill <= %d rows per rank", self.geom.n_ranks, self.geom.spec_k, self.geom.max_seqs,
                    self.geom.max_seqs_per_rank, self.geom.pools()[0].num_pages, self.geom.max_prefill_tokens_per_rank)

    # ------------------------------------------------------------------------------------------ requests in
    def add_request(self, request) -> None:
        rid = request.request_id
        if rid in self.requests:                           # streaming-input continuation of an existing request
            self._reject(request, "streaming input continuations are not supported by the phys engine")
            return
        super().add_request(request)
        why = self._unsupported(request)
        if why:
            self._reject(request, why)
            return
        if request.mm_features:
            self._mm_keys[rid] = self._mm_pool.submit(self._mm_images, request)
        self._pending_add.append(request)

    def _unsupported(self, request) -> str | None:
        if request.mm_features:
            if self.mm is None:
                return self._mm_why_off
            other = sorted({f.modality for f in request.mm_features} - {"image"})
            if other:
                return f"only image inputs are supported by the phys engine (got {', '.join(other)})"
            if any(f.data is None for f in request.mm_features):
                return "image data missing in the engine (a processor-cache entry without its data)"
        if request.prompt_embeds is not None:
            return "prompt embeddings are not supported by the phys engine"
        if request.lora_request is not None:
            return "LoRA is not supported by the phys engine"
        if request.pooling_params is not None or request.sampling_params is None:
            return "pooling requests are not supported by the phys engine"
        if request.resumable:
            return "resumable (streaming input) requests are not supported by the phys engine"
        if request.num_prompt_tokens >= self.max_model_len:
            return f"prompt of {request.num_prompt_tokens} tokens >= max_model_len {self.max_model_len}"
        caps = bridge.get_sampling_caps()
        if caps is not None:
            missing = sorted(bridge.sampling_features(request.sampling_params) - caps)
            if missing:
                return f"sampling features not supported by the phys engine's sampler yet: {', '.join(missing)}"
        return None

    def _mm_images(self, request):
        """The request's images with content keys (runs on the key thread). An error string rejects the request."""
        ids = request.prompt_token_ids or ()
        out = []
        for f in request.mm_features:
            t = MM.item_tensors(f.data)
            pos = f.mm_position
            n = MM.span_length(t)
            if n != pos.length or pos.offset + n > len(ids):
                return f"image span {pos.offset}+{pos.length} does not match its {n} processed positions"
            if any(ids[j] != MM.IMAGE_TOKEN_ID for j in range(pos.offset, pos.offset + n)):
                return f"image span {pos.offset}+{n} holds tokens other than {MM.IMAGE_TOKEN_ID}"
            out.append(MM.Image(pos.offset, n, MM.content_key(t), t))
        return out

    def _reject(self, request, why: str) -> None:
        logger.error("phys adapter: request %s rejected: %s", request.request_id, why)
        self.phys_stats["rejected"] += 1
        if request.request_id not in self.requests:
            self.requests[request.request_id] = request
            self._enqueue_waiting_request(request)
        # finished as FINISHED_ERROR with an output in the next update_from_output (vLLM's own error path)
        self.grammar_compile_error_reqs.add(request.request_id)

    def _to_rs(self, req) -> None:
        sp = req.sampling_params
        images = None
        fut = self._mm_keys.pop(req.request_id, None)
        if fut is not None:
            try:
                got = fut.result()
            except Exception as e:  # noqa: BLE001
                got = f"image keys failed: {e!r}"
            if isinstance(got, str):
                self._reject(req, got)
                return
            images = self.mm.add(req.request_id, got)
        max_new = max(1, min(req.max_tokens, self.max_model_len - req.num_prompt_tokens))
        q = _RSReq(req.request_id, req.num_prompt_tokens, max_new, req.arrival_time - self._wall0(),
                      _HashFn(req, private=bool(req.skip_reading_prefix_cache)), priority=req.priority)
        if images is not None:
            assert hasattr(q, "images"), "serve/scheduler.py has no Request.images (the image-aware rank scheduler)"
            q.images = tuple((im.offset, im.length, im.key) for im in images.images)
        if sp is not None and sp.prompt_logprobs is not None and hasattr(q, "score"):
            q.score = True                                # teacher-forced scoring: every prompt row runs the head
        if self.force.enabled:
            spec = self.force.match(req.prompt_token_ids or ())
            if spec is not None:
                q.forced = True
                self.forced[req.request_id] = ForcedReq(req.request_id, spec, req.num_prompt_tokens)
                self.phys_stats["forced"] += 1
                logger.info("phys adapter: request %s forced (tag %s, %d forced tokens, n_dump %d)", req.request_id,
                            spec.get("tag"), len(spec.get("forced", ())), int(spec.get("n_dump", 0)))
        if self.rs.add_request(q) is False:              # no rank could ever hold it (serve: sched.rejected[rid])
            self.forced.pop(req.request_id, None)
            if self.mm is not None:
                self.mm.forget(req.request_id)
            why = getattr(self.rs, "rejected", {}).get(req.request_id, "no rank can hold this request")
            self._reject(req, f"request too large for the phys engine's KV pools: {why}")

    def _wall0(self) -> float:
        return time.time() - (time.monotonic() - self._t0)

    # ------------------------------------------------------------------------------------------ schedule
    def _gc(self, ts: float) -> None:
        if ts - self._gc_last_full >= self._gc_full_s:
            t0 = time.perf_counter()
            gc.unfreeze()
            n = gc.collect()
            gc.freeze()
            ms = (time.perf_counter() - t0) * 1e3
            self._gc_last_full = ts
            st = self.phys_stats
            st["gc_full"] += 1
            st["gc_full_ms"] += ms
            st["gc_cyclic"] += n
            logger.info("phys adapter: full GC %.0f ms, %d cyclic objects reclaimed (%d frozen)", ms, n, gc.get_freeze_count())
        else:
            gc.freeze()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if steptimes.T is None and steptimes.P is None:
            return self._schedule(throttle_prefills)
        t0 = steptimes.pc()                          # PHYS_STEP_TIMES / PHYS_PROFILE_DIR (adapter/steptimes.py)
        if steptimes.P is not None:
            steptimes.P.on_step_start(self.current_step + 1)
        with steptimes.step_range(self.current_step + 1):
            so = self._schedule(throttle_prefills)
        ps = getattr(so, "phys", None)
        if steptimes.T is not None and ps is not None and ps.step >= 0 and ps.plan is not None:
            r = steptimes.T.start(ps.step, t0)
            steptimes.T.mark(ps.step, "s1")
            wk = ps.plan.work
            r.update(rows=so.total_num_scheduled_tokens, seqs=len(so.num_scheduled_tokens),
                     dec=sum(1 for w in wk if w.kind == "decode"), pf=sum(w.n for w in wk if w.kind != "decode"),
                     cp=sum(1 for w in wk if getattr(w, "cp", None) is not None), copies=len(ps.plan.copies))
        return so

    def _schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        ts = time.monotonic()
        if self._gc_every and self.current_step % self._gc_every == 0:
            self._gc(ts)
        now = ts - self._t0

        still = []
        for req in self._pending_add:
            if req.is_finished() or self.requests.get(req.request_id) is not req:
                continue
            if self._is_blocked_waiting_status(req.status) and not self._handle_blocked_waiting_request(req):
                still.append(req)
                continue
            fut = self._mm_keys.get(req.request_id)
            if fut is not None and not fut.done():
                still.append(req)                   # image content keys still hashing (off this thread)
                continue
            if req.status == RequestStatus.WAITING:
                self._to_rs(req)
        self._pending_add = still
        if self.async_mode:
            self._sync_waiting()

        plan = None
        if self._pause_state != PauseState.PAUSED_ALL:
            hidden = None
            if self._pause_state == PauseState.PAUSED_NEW:
                hidden, self.rs.waiting = self.rs.waiting, deque()
            try:
                if self._grammar_hold:
                    self._hold_structured()
                plan = self.rs.schedule(now)
            finally:
                if hidden is not None:
                    self.rs.waiting = hidden

        # rs -> vLLM request states
        preempted = set()
        if plan is not None and plan.preempted:
            self._preempting = True
            try:
                for rid in plan.preempted:
                    req = self.requests.get(rid)
                    preempted.add(rid)
                    self._started.discard(rid)
                    if req is not None and req.status == RequestStatus.RUNNING:
                        self.running.remove(req)
                        self._preempt_request(req, ts)
            finally:
                self._preempting = False
        for q in self.rs.running:
            req = self.requests.get(q.rid)
            if req is None or req.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                continue
            resumed = req.status == RequestStatus.PREEMPTED
            try:
                self.waiting.remove_request(req)
            except ValueError:
                self.kv_holding_waiting.remove_request(req)
            self.deferred_waiting.discard(req)
            self.running.append(req)
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens = q.replay_start     # the admission's start (async mode advanced q at schedule)
            req.replay_start = q.replay_start
            if self.log_stats:
                req.record_event(EngineCoreEventType.SCHEDULED, ts)
            if req.prefill_stats and req.num_preemptions <= 0:
                req.prefill_stats.set(num_prompt_tokens=req.num_prompt_tokens,
                                      num_local_cached_tokens=min(q.cached_tokens, req.num_prompt_tokens),
                                      num_external_cached_tokens=0)
            self.kv_cache_manager.record_prefix(req.num_tokens, q.cached_tokens, resumed)

        # the plan as vLLM's SchedulerOutput
        num_sched: dict = {}
        spec: dict = {}
        new_data, new_inits = [], []
        c_ids, c_comp, c_out, c_blocks = [], [], [], []
        work = plan.work if plan is not None else []
        for w in work:
            cp = getattr(w, "cp", None)
            if w.kind == "cp":
                # CP prefill helper: a sub-chunk of the owner's prompt computed on another
                # rank under a shadow key ("cp", rid, rank). vLLM sees ONE request scheduled for the whole chunk: the
                # sub-chunks add to the owner's count; the worker gets the entry under the prompt's rid
                # (build_step_batch), so nothing else is per-helper here
                assert cp is not None and cp.rid in self.requests, (w.rid, cp)
                num_sched[cp.rid] = num_sched.get(cp.rid, 0) + w.n
                self.phys_stats["cp_helper_rows"] += w.n
                continue
            req = self.requests[w.rid]
            assert req.status == RequestStatus.RUNNING, (w.rid, req.status)
            if cp is not None:                        # the CP owner's entry: its sub-chunk is the chunk's last
                assert req.num_computed_tokens == cp.subs[0][1], (w.rid, req.num_computed_tokens, cp.subs)
                assert w.start == cp.subs[-1][1] and w.n == cp.subs[-1][2], (w.rid, w.start, w.n, cp.subs)
                self.phys_stats["cp_chunks"] += 1
            elif w.start >= 0:
                assert req.num_computed_tokens == w.start, (w.rid, req.num_computed_tokens, w.start, w.kind)
            else:                                     # async decode: resolved on the device from the settled count
                assert self.async_mode and w.kind == "decode" and req.num_computed_tokens >= w.known, (
                    w.rid, req.num_computed_tokens, w.known)
            num_sched[w.rid] = num_sched.get(w.rid, 0) + w.n
            if w.kind == "decode":
                if w.start >= 0:
                    assert w.start == req.num_tokens + req.num_output_placeholders - 1, (
                        w.rid, w.start, req.num_tokens, req.num_output_placeholders)
                ns = w.n - 1
                if ns > 0:
                    drafts = list(req.spec_token_ids[:ns]) if req.use_structured_output else []
                    spec[w.rid] = drafts + [-1] * (ns - len(drafts))
                req.spec_token_ids = []
            if w.rid not in self._started:
                self._started.add(w.rid)
                new_data.append(NewRequestData.from_request(req, ([],), req._all_token_ids,
                                                            uses_mrope=self.model_uses_mrope))
                new_inits.append(ReqInit(w.rid, w.rank, list(req.prompt_token_ids or ()), list(req.output_token_ids),
                                         req.sampling_params, req.use_structured_output))
            else:
                c_ids.append(w.rid)
                c_comp.append(req.num_computed_tokens)
                c_out.append(req.num_output_tokens)
                c_blocks.append(None)
        cached = CachedRequestData(req_ids=c_ids, resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
                                   new_block_ids=c_blocks, num_computed_tokens=c_comp, num_output_tokens=c_out)
        so = SchedulerOutput(
            scheduled_new_reqs=new_data,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_sched,
            total_num_scheduled_tokens=sum(num_sched.values()),
            scheduled_spec_decode_tokens=spec,
            scheduled_encoder_inputs={},
            preempted_req_ids=self.reset_preempted_req_ids,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            num_spec_tokens_to_schedule=self.num_spec_tokens,
        )
        so.phys = self._phys_step(plan, so, new_inits, preempted, spec)
        self.phys_stats["steps"] += 1
        if self.phys_stats["steps"] % 2000 == 0:
            logger.info("phys adapter: scheduler stats %s", {k: v for k, v in self.phys_stats.items()
                                                            if k.startswith(("grammar", "deferred", "cp_", "steps", "settled"))})
            if self.mm is not None:
                logger.info("phys adapter: image stats %s, resident MiB per rank %s", self.mm.stats,
                            [round(u / 2 ** 20, 1) for u in self.mm.used])
        if not num_sched:
            self.phys_stats["empty_steps"] += 1
        self._update_after_schedule(so)
        return so

    def _phys_step(self, plan, so, new_inits, preempted, spec) -> PhysStep:
        R = self.geom.n_ranks
        if plan is None or not plan.work:
            batches = [build_step_batch(plan, self.rs.kv, self.geom, r) for r in range(R)] if plan is not None else []
            return PhysStep(step=plan.step if plan is not None else -1, batches=batches, row_tokens=[[] for _ in batches],
                            new_reqs=new_inits, finished=set(so.finished_req_ids), preempted=preempted, order=[],
                            plan=plan)
        batches = [build_step_batch(plan, self.rs.kv, self.geom, r) for r in range(R)]
        row_tokens, prev, emits = [], {}, {}
        for sb in batches:
            toks = []
            for se in sb.seqs:
                req = self.requests[se.rid]
                all_ids = req._all_token_ids
                fr = self.forced.get(se.rid)
                n0 = req.num_output_tokens + req.num_output_placeholders   # outputs incl. in-flight ones (async)
                if se.kind == "decode":
                    hd = None
                    if 0 <= se.start < len(all_ids):
                        t0 = all_ids[se.start]
                    elif fr is not None and 1 <= n0 <= len(fr.forced):
                        t0 = fr.forced[n0 - 1]                # forced: the host knows the in-flight token
                    elif se.start < 0 and req.num_output_placeholders == 0 and se.rid in self._dev_drafts:
                        # a structured request resuming after a held step (nothing of it in flight): every input id
                        # is host-given (its last token + the engine's drafts of its last step), so the engine does
                        # not read its batch-order next ids, which the held step rewrote
                        hd = self._dev_drafts.pop(se.rid)
                        t0 = all_ids[-1]
                        if len(hd) < se.n_rows - 1:
                            hd = None
                            t0 = -1
                    elif se.start < 0 and req.num_output_placeholders == 0 and all_ids:
                        # async, nothing of it in flight: it missed the previous step (a hold, or no KV room), so the
                        # engine's next ids for it are stale. Its last token is known here; the drafts stay -1
                        # (the engine's prev-step check: rows of a sequence absent from the previous step keep the host
                        # ids, drafts invalid)
                        t0 = all_ids[-1]
                        self.phys_stats["settled_host_t0"] += 1
                        if req.use_structured_output and se.rid in self._has_drafts:
                            self.phys_stats["grammar_resume_no_ids"] += 1
                    else:
                        t0 = -1                               # async: sampled by the step in flight (device side)
                    toks.append(t0)
                    if hd is not None:
                        toks.extend(int(x) for x in hd[:se.n_rows - 1])
                        self.phys_stats["grammar_host_ids"] += 1
                        continue
                    e = fr.emit(n0, se.n_rows) if fr is not None else None
                    if e is not None:
                        emits[se.rid] = e
                        toks.extend(e[:-1])                   # drafts := the next forced tokens
                        fr.checked_rows += se.n_rows             # production counts verify rows
                        fed = [t0] + e[:-1]
                        want = ([fr.forced[n0 - 1]] if n0 >= 1 else [t0]) + fr.forced[n0:n0 + se.n_rows - 1]
                        fr.input_mismatch += sum(1 for a, b in zip(fed, want) if a != b)
                    else:
                        toks.extend([-1] * (se.n_rows - 1))
                else:
                    toks.extend(all_ids[se.start:se.start + se.n_rows])
                    if fr is not None and se.last_chunk:
                        e = fr.emit(n0, 1)
                        if e is not None:
                            emits[se.rid] = e
                cp = getattr(se, "cp", None)
                if se.start >= 0 and (cp is None or cp.owner == sb.rank):   # device rows: the engine's own ring;
                    # CP helpers share the prompt's rid (the engine reads their context from its token history)
                    prev[se.rid] = all_ids[max(0, se.start - 3):min(se.start, len(all_ids))]
            assert len(toks) == sb.n_rows, (sb.rank, len(toks), sb.n_rows)
            row_tokens.append(toks)
        ps = PhysStep(step=plan.step, batches=batches, row_tokens=row_tokens, prev_tokens=prev, new_reqs=new_inits,
                      finished=set(so.finished_req_ids), preempted=preempted, order=list(so.num_scheduled_tokens),
                      num_spec={rid: len(v) for rid, v in spec.items()}, est_us=plan.est_us, plan=plan)
        ps.forced = emits
        ps.forced_reqs = {rid: self.forced[rid] for rid in emits}
        ps.vocab_size = self.vocab_size
        if self.mm is not None and self.mm.reqs:
            ps.mm = self.mm.plan(batches)
        return ps

    # ------------------------------------------------------------------------------------------ results
    def update_from_output(self, scheduler_output, model_runner_output):
        if steptimes.T is None and steptimes.P is None:
            return self._phys_update(scheduler_output, model_runner_output)
        ps = getattr(scheduler_output, "phys", None)
        st = ps.step if ps is not None else -1
        if steptimes.T is not None:
            steptimes.T.mark(st, "u0")
        with steptimes.rf("phys:update"):
            out = self._phys_update(scheduler_output, model_runner_output)
        if steptimes.T is not None:
            steptimes.T.finish(st, steptimes.pc())
        if steptimes.P is not None:
            steptimes.P.on_step_end()
        return out

    def _phys_update(self, scheduler_output, model_runner_output):
        ps = getattr(scheduler_output, "phys", None)
        plan = ps.plan if ps is not None else None
        before = {}
        if plan is not None:
            for w in plan.work:
                req = self.requests.get(w.rid)
                if req is not None:
                    before[w.rid] = (req, req.num_output_tokens)
        self._in_update = True
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self._in_update = False
        if self._grammar_hold:
            self._after_update_drafts(model_runner_output)
        if plan is not None:
            accepted = {}
            for w in plan.work:
                if w.kind != "decode" or w.rid not in before:
                    continue
                q = self.rs.reqs.get(w.rid)
                if q is None or q.phase == FINISHED or q not in self.rs.running:
                    continue                                # gone, or preempted with its in-flight output stale
                req, n0 = before[w.rid]
                a = req.num_output_tokens - n0              # tokens vLLM kept (stop / max_tokens trimmed)
                assert 1 <= a <= w.n, (w.rid, a, w.n, "the worker returned no token for a decode group")
                accepted[w.rid] = a
            self.rs.update(plan, accepted, time.monotonic() - self._t0)
        for rid in self._deferred_free:
            if rid in self.rs.reqs:
                self.rs.abort(rid)
            self._started.discard(rid)
        self._deferred_free.clear()
        if CHECK and plan is not None:
            self._check(plan)
        return outputs

    def _check(self, plan) -> None:
        self.rs.kv.check()
        if self.mm is not None:
            self.mm.check()
            stray = set(self.mm.reqs) - set(self.rs.reqs)
            assert not stray, sorted(stray)[:5]
        for q in self.rs.running:
            req = self.requests.get(q.rid)
            assert req is not None and req.status == RequestStatus.RUNNING, (q.rid, req and req.status)
            if q.phase == DECODE:
                ahead = getattr(q, "inflight", 0) * self.geom.rows_per_decode_seq    # async: vLLM counts in-flight rows
                assert q.num_computed + ahead == req.num_computed_tokens, (
                    q.rid, q.num_computed, ahead, req.num_computed_tokens)
                assert q.num_output + ahead + int(getattr(q, "first_pending", False)) == \
                    req.num_output_tokens + req.num_output_placeholders, (
                    q.rid, q.num_output, ahead, req.num_output_tokens, req.num_output_placeholders)
        rs_ids = {q.rid for q in self.rs.running}
        for req in self.running:
            assert req.request_id in rs_ids, req.request_id

    # ------------------------------------------------------------------------------------------ frees
    def _phys_free(self, rid: str) -> None:
        self._dev_drafts.pop(rid, None)
        self._has_drafts.discard(rid)
        if self.mm is not None:
            if self._preempting:             # re-admitted later: keep its images, release its encoder-cache refs
                self.mm.release(rid)
            else:
                self.mm.forget(rid)
                fut = self._mm_keys.pop(rid, None)
                if fut is not None:
                    fut.cancel()
        if self._preempting:                 # rs already released the KV of the sequences it preempts
            return
        fr = self.forced.pop(rid, None)
        if fr is not None and not fr.flushed:
            fr.flush(self.vocab_size, partial=not fr.done())
        if self._in_update:                  # finished inside update_from_output: after rs.update committed the step
            self._deferred_free.add(rid)
            return
        if rid in self.rs.reqs:
            self.rs.abort(rid)
        self._started.discard(rid)

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        if reset_running_requests and self.running:
            logger.warning("phys adapter: reset_prefix_cache(reset_running_requests=True) is not supported")
            return False
        return self._phys_reset_prefix_cache()

    def _phys_reset_prefix_cache(self) -> bool:
        if self.rs.running or self.rs.kv.seqs or self.rs.kv.pinned:
            return False
        self.rs.kv = OwnedKVManager(self.geom)
        logger.info("phys adapter: prefix cache reset")
        return True

    def _hold_structured(self) -> None:
        """ASYNC: structured-output requests with tokens in flight skip this step (rank scheduler skip_until_step), so
        vLLM never defers a step's sampling for them. Only requests whose drafts came back in a step result (their
        resume step gets host-given ids) and only while they are at most PHYS_ASYNC_GRAMMAR_MAX_FRAC of the decoding
        sequences (else holding would halve most of the batch: vLLM's deferral is the better trade)."""
        held, dec = [], 0
        for q in self.rs.running:
            if q.phase != DECODE:
                continue
            dec += 1
            req = self.requests.get(q.rid)
            if req is not None and req.use_structured_output and req.num_output_placeholders > 0 \
                    and q.rid in self._has_drafts:
                held.append(q)
        if not held:
            return
        if len(held) > self._grammar_max_frac * dec:
            self.phys_stats["grammar_hold_off"] += 1
            return
        nxt = self.rs.step_id + 1                    # the step about to be planned is rs.step_id
        for q in held:
            if q.skip_until_step < nxt:
                q.skip_until_step = nxt
        self.phys_stats["grammar_held"] += len(held)

    def _after_update_drafts(self, model_runner_output) -> None:
        """ASYNC + hold: a structured request with nothing in flight after this update (it is held in the step already
        dispatched) gets the engine's drafts grammar-validated into spec_token_ids, as sync mode's post_step does
        (EngineCore skips that under async scheduling; without it the next bitmask would constrain only position 0),
        and keeps a host copy of them for its resume step's input ids."""
        dr = getattr(model_runner_output, "phys_drafts", None)
        if dr is None:
            return
        ids, toks = [], []
        for rid, d in zip(dr.req_ids, dr.draft_token_ids):
            req = self.requests.get(rid)
            if req is None or req.is_finished() or not req.use_structured_output:
                self._dev_drafts.pop(rid, None)
                continue
            self._has_drafts.add(rid)
            if req.num_output_placeholders == 0:
                self._dev_drafts[rid] = list(d)
                ids.append(rid)
                toks.append(list(d))
            else:
                self._dev_drafts.pop(rid, None)
        if ids:
            self.update_draft_token_ids(DraftTokenIds(ids, toks))

    # ------------------------------------------------------------------------------------------ async (D13, vLLM side)
    # vLLM's AsyncScheduler plans step N+1 before step N's results: it advances num_computed_tokens by all scheduled
    # tokens and rolls rejected drafts back in update_from_output. The rank scheduler's native async mode
    # (SchedConfig.async_mode, D13) keeps the SETTLED num_computed per sequence plus `inflight` decode steps:
    # in-flight decode groups get start = -1 (positions / slots resolved on the device by the engine's row expansion),
    # forced ones their known start; prefill progress and the first token are advanced at schedule time; frees and
    # preemptions are deferred until the newest scheduled step settled.
    def _sync_waiting(self) -> None:
        """Preempted requests wait in the rank scheduler with vLLM's output count: in-flight (stale) outputs that vLLM
        appended after the preemption are recomputed as prefill on re-admission (vLLM recompute semantics)."""
        for q in self.rs.waiting:
            req = self.requests.get(q.rid)
            if req is not None and q.num_output != req.num_output_tokens:
                q.num_output = req.num_output_tokens

    def shutdown(self) -> None:
        logger.info("phys adapter: scheduler stats %s, rank scheduler %s", self.phys_stats, self.rs.stats)
        if self.mm is not None:
            logger.info("phys adapter: image stats %s", self.mm.stats)
            self._mm_pool.shutdown(wait=False, cancel_futures=True)
        super().shutdown()


def sched_config_from_env() -> SchedConfig:
    """PHYS_SCHED='key=value,...' overrides SchedConfig fields (placement, rebalance, steal, quantum, ...)."""
    cfg = SchedConfig()
    spec = os.environ.get("PHYS_SCHED", "")
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes", "on")
        else:
            val = type(cur)(v)
        setattr(cfg, k, val)
    return cfg


class PhysAsyncScheduler(PhysScheduler, AsyncScheduler):
    """PhysScheduler with vLLM's async scheduling (--async-scheduling, max_concurrent_batches = 2): AsyncScheduler's
    placeholders / -1 spec tokens / rejection rollback / deferred structured-output sampling, on the rank scheduler's
    native async mode (D13). Use with PhysExecutor and PHYS_ASYNC=1, and a worker that resolves decode
    positions on the device (device_positions: the engine's row expansion; the fake worker emulates it)."""
    async_mode = True

    def _update_after_schedule(self, scheduler_output) -> None:
        super()._update_after_schedule(scheduler_output)
        if scheduler_output.pending_structured_output_tokens:       # vLLM will defer this step's sampling (serialized)
            self.phys_stats["deferred_steps"] = self.phys_stats.get("deferred_steps", 0) + 1
