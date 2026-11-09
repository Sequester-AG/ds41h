"""PhysExecutor: vLLM 0.32's Executor interface over the ds41h-phys worker (4 GPUs, one process).

Plugged in with `--distributed-executor-backend adapter.phys_executor.PhysExecutor` (ParallelConfig accepts an Executor
subclass or its qualname; vllm/v1/executor/abstract.py Executor.get_class). EngineCore builds it in the EngineCore
process; the worker lives in the same process (serve/worker.py drives all 4 GPUs from one process: the fork's
collectives use peer pointers), so no RPC layer is needed and `so.phys` reaches the worker as a Python object.

Worker selection (PHYS_ADAPTER_WORKER, see make_worker): fake (CPU, adapter/fake_worker.py) or the GPU worker of
adapter/stage4_worker.py (serve/worker.py on the GPUs).

Start-up contract with EngineCore._initialize_kv_caches:
  get_kv_cache_specs        one 64-token FullAttentionSpec ("phys.paged", 1 head x 64 B, int8): vLLM hashes 64-token
                            blocks (= the paged pool's page), its own block pool stays a small dummy
  determine_available_memory  bytes for PHYS_VLLM_BLOCKS dummy blocks (default: max_model_len / 64 + 64), so vLLM's
                            max_model_len check passes; the real pools are sized by the worker (Geometry)
  initialize_from_config / compile_or_warm_up_model / set_kv_cache_layout: worker warm-up, layouts are irrelevant
Step contract with EngineCore.step (synchronous loop, max_concurrent_batches == 1):
  execute_model(so)         keeps `so.phys`; returns None (EngineCore computes the grammar bitmask, then calls
                            sample_tokens) or the empty output for a step without rows (frees only)
  sample_tokens(grammar)    splits the bitmask into per-request rows, runs worker.step(phys, grammar), merges the 4
                            ranks' StepResults into one ModelRunnerOutput in vLLM's request order
  take_draft_token_ids()    the real drafts of structured-output requests (vLLM validates them against the grammar
                            before the next step); None when the batch has none (the other drafts stay on the device)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds, LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes

from adapter import bridge, runtime_state, slotdump, steptimes
from adapter.bridge import GrammarRows
_SLOTDUMP = bool(os.environ.get("PHYS_D13_SLOTDUMP"))

logger = init_logger("vllm.phys_adapter")

_PAGE_HEAD_BYTES = 64          # dummy spec: 64 tokens x 1 head x 64 B int8 x (K, V) = 8 KiB per block


def warm_thread_blas(devs, streams) -> None:
    """Create THIS thread's cuBLAS / cuBLASLt handles and their per-stream workspaces on each (device, stream), eagerly.
    torch keeps cuBLAS handles per thread and device: a thread's first matmul inside a CUDA-graph capture has to create
    one there, which fails (CUBLAS_STATUS_NOT_INITIALIZED, then cudaErrorStreamCaptureInvalidated). Under ASYNC the
    merge thread can run D13's _extract, which captures a new HSD graph on the rank's step stream."""
    for d, s in zip(devs, streams):
        with torch.cuda.device(d), torch.cuda.stream(s):
            a = torch.ones(64, 64, device=d, dtype=torch.bfloat16)
            torch.mm(a, a.t())
            torch.mm(a, a.t(), out_dtype=torch.float32)
            torch.addmm(a[0], a, a.t())
            torch.nn.functional.linear(a, a, a[0])
            f = a.float()
            _ = f @ f.t()
        torch.cuda.synchronize(d)


class _LazyFuture(Future):
    """A step's output computed on the thread that asks for it (vLLM's engine loop: step_with_batch_queue only calls
    result()), like vLLM's own AsyncOutputFuture (uniproc_executor.py). D13: the handle's extraction, and any graph
    capture it does, then always runs on the engine loop thread that dispatches (on the merge thread a capture hit
    CUBLAS_STATUS_NOT_INITIALIZED, and a capture there could race the loop thread's CUDA calls)."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def result(self, timeout=None):
        if not super().done():
            fn, self._fn = self._fn, None
            if fn is not None:
                try:
                    self.set_result(fn())
                except Exception as e:  # noqa: BLE001
                    self.set_exception(e)
        return super().result(timeout)


def _done(value) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


class PhysExecutor(Executor):
    uses_ray = False
    supports_pp = False

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """PHYS_ASYNC=1: vLLM's async scheduling (max_concurrent_batches = 2, with --async-scheduling and
        --scheduler-cls adapter.phys_scheduler.PhysAsyncScheduler). Needs a worker that resolves decode positions on the
        device (D13). Default off: the engine side does not exist yet."""
        return os.environ.get("PHYS_ASYNC", "0") == "1"

    # ------------------------------------------------------------------------------------------ start-up
    def _init_executor(self) -> None:
        self.kind = os.environ.get("PHYS_ADAPTER_WORKER", "fake")
        self.k = self.vllm_config.num_speculative_tokens
        from adapter import debug_patches            # opt-in diagnosis patches (PHYS_WS_PER_DEVICE ...), before the engine
        dp = debug_patches.install()
        if dp:
            logger.warning("phys adapter: debug patches active: %s", dp)
        runtime_state.dump("pre_worker")             # PHYS_DUMP_RUNTIME=<file>: served-vs-offline process diff
        self.worker = make_worker(self.kind, self.vllm_config)
        runtime_state.dump("post_worker")
        # production's ds41h.moe reads DS41H_MOE_DEC at import (vLLM's ds41h plugin imports it first). The launcher
        # exports it; the engine's prod_moe() also resets the module global on its first call.
        # A mismatch here is a warning (PHYS_MOE_ENV_CHECK=strict: refuse to start).
        real = bool(getattr(self.worker, "_real_engine", False))      # calls production's ds41h.moe (prefill MoE)
        if not runtime_state.check_moe_env(logger.warning, "1" if real else None) and \
                os.environ.get("PHYS_MOE_ENV_CHECK", "warn") == "strict":
            raise RuntimeError("phys adapter: production's ds41h.moe kernel switches differ from the environment "
                               "(see the warning above)")
        self._rt_first = bool(os.environ.get("PHYS_DUMP_RUNTIME"))
        bridge.set_geometry(self.worker.geom)
        bridge.set_sampling_caps(getattr(self.worker, "sampling_caps", None))
        bridge.set_cost_default(getattr(self.worker, "cost_default", None))
        bridge.set_image_support(bool(getattr(self.worker, "supports_images", False)))
        self._phys = None
        self._so = None
        self._drafts: DraftTokenIds | None = None
        self.step_times: list = []
        self._disp_times: list = []          # D13: engine-loop thread time inside dispatch (waits for the previous step)
        self._failure_callback = None
        self._log_every = int(os.environ.get("PHYS_LOG_EVERY", "500"))
        self._win = self._new_window()
        self.async_mode = bool(self.vllm_config.scheduler_config.async_scheduling)
        self._main_dispatch = False
        self._lazy_merge = False
        if self.async_mode:
            if not getattr(self.worker, "device_positions", False):
                raise RuntimeError("phys adapter: async scheduling needs a worker that resolves decode positions on the "
                                   "device (D13 row expansion); this worker does not declare device_positions")
            # one step thread: steps run in order; the engine loop plans step N+1 while step N runs there
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="phys-step")
            self._fifo: deque = deque()                 # (so, ps) executed, not yet sampled (EngineCore pairs them FIFO)
            self._wait_prepared = bool(getattr(self.worker, "signals_prepared", False))
            # D13 dispatch model: worker.step() on THIS thread queues step N+1 on the GPU and waits for N; the pool
            # thread merges N's handle.result() into vLLM's output (all worker calls stay on one thread)
            self._main_dispatch = bool(getattr(self.worker, "main_thread_dispatch", False))
            # PHYS_MERGE_THREAD=lazy (default): a dispatched step's merge (D13 extraction + merge_results) runs when vLLM
            # asks for the step's output, on the engine loop thread; =pool: on the step thread (the earlier behaviour)
            self._lazy_merge = self._main_dispatch and os.environ.get("PHYS_MERGE_THREAD", "lazy") == "lazy"
            eng = getattr(getattr(self.worker, "w", None), "backend", None)
            devs, streams = getattr(eng, "devs", None), getattr(eng, "streams", None)
            if self._main_dispatch and not self._lazy_merge and devs and streams and torch.cuda.is_available():
                # the merge thread may extract a D13 step and capture a graph on the step streams (warm_thread_blas)
                self._pool.submit(warm_thread_blas, list(devs), list(streams)).result()
                logger.info("phys adapter: cuBLAS handles warmed on the merge thread (%d step streams)", len(streams))
            logger.info("phys adapter: async scheduling on (%s)", ("dispatch and merge on the engine loop thread "
                        "(lazy merge)" if self._lazy_merge else "dispatch on the engine loop thread, merge on the "
                        "step thread") if self._main_dispatch else
                        f"step thread, wait for host prepare: {self._wait_prepared}")
        logger.info("phys adapter: PhysExecutor with the %s worker, k=%d", self.kind, self.k)

    def register_failure_callback(self, callback) -> None:
        self._failure_callback = callback

    def get_kv_cache_specs(self):
        page = bridge.get_geometry().page_tokens
        spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=_PAGE_HEAD_BYTES, dtype=torch.int8)
        return [{"phys.paged": spec}]

    def determine_available_memory(self) -> list[int]:
        page = bridge.get_geometry().page_tokens
        n = int(os.environ.get("PHYS_VLLM_BLOCKS", "0")) or (-(-self.model_config.max_model_len // page) + 64)
        return [n * 2 * page * _PAGE_HEAD_BYTES]

    def get_supported_kv_cache_layouts(self):
        return [["NHD"]]

    def set_kv_cache_layout(self, layout_name: str) -> None:
        return None

    def initialize_from_config(self, kv_cache_configs) -> None:
        return None

    def compile_or_warm_up_model(self) -> None:
        t0 = time.time()
        self.worker.warmup()
        self.vllm_config.compilation_config.compilation_time = time.time() - t0
        runtime_state.dump("warmup")

    @property
    def supported_tasks(self):
        return ("generate",)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
        kwargs = kwargs or {}
        if callable(method):
            raise NotImplementedError(f"phys adapter: collective_rpc with a callable ({method})")
        fn = {
            "check_health": lambda: self.worker.check_health(),
            "get_supported_tasks": lambda: ("generate",),
            "get_kv_connector_handshake_metadata": lambda: None,
            "reset_mm_cache": lambda: None,
            "reset_encoder_cache": lambda: None,
            "shutdown": lambda: self.worker.shutdown(),
            "compile_or_warm_up_model": lambda: CompilationTimes(0.0, 0.0),
            "supports_draft_weight_updates": lambda: False,
            "list_loras": lambda: set(),
            "update_max_model_len": lambda *a: None,
            "profile": lambda *a, **k: None,
        }.get(method)
        if fn is None:
            raise NotImplementedError(f"phys adapter: collective_rpc('{method}') is not supported")
        out = [fn(*args, **kwargs)]
        return _done(out) if non_block else out

    def check_health(self) -> None:
        self.worker.check_health()

    def shutdown(self) -> None:
        if self.step_times:
            t = np.array(self.step_times)
            logger.info("phys adapter: %d steps, step wall ms p50 %.2f p95 %.2f", len(t), np.percentile(t, 50) * 1e3,
                        np.percentile(t, 95) * 1e3)
        if self._disp_times:
            t = np.array(self._disp_times)
            logger.info("phys adapter: D13 dispatch (engine loop thread) ms p50 %.2f p95 %.2f", np.percentile(t, 50) * 1e3,
                        np.percentile(t, 95) * 1e3)
        if getattr(self, "async_mode", False):
            self._pool.shutdown(wait=True)
        if steptimes.T is not None:
            steptimes.T.close()
        self.worker.shutdown()

    def execute_dummy_batch(self) -> None:
        return None

    def take_draft_token_ids(self):
        d, self._drafts = self._drafts, None
        return d

    # ------------------------------------------------------------------------------------------ steps
    def execute_model(self, scheduler_output, non_block: bool = False):
        ps = getattr(scheduler_output, "phys", None)
        assert ps is not None, "phys adapter: SchedulerOutput without .phys (is --scheduler-cls PhysScheduler set?)"
        if self.async_mode:
            if scheduler_output.total_num_scheduled_tokens == 0:
                def frees():
                    if ps.finished or ps.preempted or ps.new_reqs:
                        _dump(ps, None)
                        self.worker.step(ps, None)
                    return EMPTY_MODEL_RUNNER_OUTPUT
                if self._main_dispatch:                  # worker calls stay on this thread; nothing is queued
                    out = frees()
                    return _done(out) if non_block else out
                fut = self._pool.submit(frees)           # in order with the steps in flight
                return fut if non_block else fut.result()
            if scheduler_output.pending_structured_output_tokens:
                # vLLM defers this step's sampling until the previous step's update_from_output, which commits KV and
                # releases window blocks: the StepBatch references the live tables (D12), so freeze a copy now
                snapshot_tables(ps)
            self._fifo.append((scheduler_output, ps))
            return _done(None) if non_block else None
        self._so, self._phys = scheduler_output, ps
        out = None
        if scheduler_output.total_num_scheduled_tokens == 0:
            if ps.finished or ps.preempted or ps.new_reqs:
                _dump(ps, None)
                self.worker.step(ps, None)               # frees only
            self._so = self._phys = None
            out = EMPTY_MODEL_RUNNER_OUTPUT
        return _done(out) if non_block else out

    def sample_tokens(self, grammar_output, non_block: bool = False):
        if self.async_mode:
            so, ps = self._fifo.popleft()
            grammar = split_bitmask(grammar_output, so) if grammar_output is not None else None
            if self._main_dispatch:
                return self._dispatch(so, ps, grammar, non_block)
            if self._wait_prepared:
                ps.prepared = threading.Event()
            fut = self._pool.submit(self._run_async, so, ps, grammar)
            if self._wait_prepared:
                # the StepBatch references the KV manager's live block / window tables (D12): the worker must have
                # consumed them before the engine loop plans the next step
                while not ps.prepared.wait(0.5):
                    if fut.done():
                        break
            return fut if non_block else fut.result()
        try:
            out = self._run(grammar_output)
        except Exception:
            logger.exception("phys adapter: step failed")
            raise
        return _done(out) if non_block else out

    def _dispatch(self, so, ps, grammar, non_block: bool):
        """D13: queue this step (worker.step on the engine loop's thread: it also waits for the previous step and
        extracts its results), then merge its handle on the pool thread, in order."""
        _dump(ps, grammar)
        if self._rt_first:
            self._rt_first = False
            runtime_state.dump("first_step")
        t0 = time.perf_counter()
        try:
            with steptimes.rf("phys:dispatch"):
                handle = self.worker.step(ps, grammar)
            if _SLOTDUMP:
                slotdump.maybe_dump(self, ps)            # PHYS_D13_SLOTDUMP: debug, synchronizes every device
        except Exception:
            logger.exception("phys adapter: dispatch failed")
            raise
        t_disp = time.perf_counter() - t0
        if self._lazy_merge:
            fut = _LazyFuture(lambda: self._merge_handle(so, ps, handle, t0, t_disp))
        else:
            fut = self._pool.submit(self._merge_handle, so, ps, handle, t0, t_disp)
        return fut if non_block else fut.result()

    def _merge_handle(self, so, ps, handle, t0: float, t_disp: float):
        try:
            results = handle.result() if hasattr(handle, "result") else handle
            r1 = time.perf_counter()
            dt = r1 - t0                                 # dispatch -> results on the host
            self.step_times.append(dt)
            self._disp_times.append(t_disp)
            out = merge_results(so, ps, results, self)
            if steptimes.T is not None:
                self._st_rec(ps, t0, t0 + t_disp, time.perf_counter(), handle, r1)
            self._window(ps, so, out, dt)
            return out
        except Exception:
            logger.exception("phys adapter: step failed")
            raise

    def _run_async(self, so, ps, grammar):
        try:
            t0 = time.perf_counter()
            results = self.worker.step(ps, grammar)
            t1 = time.perf_counter()
            dt = t1 - t0
            self.step_times.append(dt)
            out = merge_results(so, ps, results, self)
            if steptimes.T is not None:
                self._st_rec(ps, t0, t1, time.perf_counter())
            self._window(ps, so, out, dt)
            return out
        except Exception:
            logger.exception("phys adapter: step failed")
            raise
        finally:
            ev = getattr(ps, "prepared", None)
            if ev is not None:
                ev.set()

    def _run(self, grammar_output) -> ModelRunnerOutput:
        so, ps = self._so, self._phys
        assert so is not None, "sample_tokens without execute_model"
        self._so = self._phys = None
        grammar = split_bitmask(grammar_output, so) if grammar_output is not None else None
        _dump(ps, grammar)
        if self._rt_first:
            self._rt_first = False
            runtime_state.dump("first_step")
        t0 = time.perf_counter()
        with steptimes.rf("phys:worker"):
            results = self.worker.step(ps, grammar)
        t1 = time.perf_counter()
        dt = t1 - t0
        self.step_times.append(dt)
        with steptimes.rf("phys:merge"):
            out = merge_results(so, ps, results, self)
        if steptimes.T is not None:
            self._st_rec(ps, t0, t1, time.perf_counter())
        self._window(ps, so, out, dt)
        return out

    def _st_rec(self, ps, x0: float, x1: float, m1: float, handle=None, r1: float | None = None) -> None:
        """PHYS_STEP_TIMES: this step's executor marks (+ the engine's own split of a sync decode step / D13's)."""
        T, st = steptimes.T, ps.step
        T.mark(st, "x0", x0)
        T.mark(st, "x1", x1)
        if r1 is not None:
            T.mark(st, "r1", r1)
        T.mark(st, "m1", m1)
        kw = {}
        lm = getattr(self.worker, "last_ms", None)
        if lm and not self.async_mode:
            kw["worker"] = [round(x, 3) for x in lm]
        eng = getattr(getattr(self.worker, "w", None), "backend", None)
        dl = getattr(eng, "stats", {}).get("decode_last") if eng is not None else None
        if dl is not None and id(dl) != getattr(self, "_st_dl", None):
            self._st_dl = id(dl)
            kw["eng"] = dl
        fb = getattr(eng, "stats", {}).get("d13_fb_last") if eng is not None else None   # D13 fallback phases
        if fb is not None and id(fb) != getattr(self, "_st_fb", None):
            self._st_fb = id(fb)
            kw["fb"] = fb
        if handle is not None and getattr(handle, "t", None):
            kw["d13"] = handle.t
        if kw:
            T.put(st, **kw)

    def _window(self, ps, so, out, dt: float) -> None:
        """One log line every PHYS_LOG_EVERY steps: worker wall per step, rows, decode / prefill mix, tokens out."""
        w = self._win
        w["n"] += 1
        w["wall"] += dt
        w["est"] += ps.est_us / 1e6
        w["rows"] += so.total_num_scheduled_tokens
        w["tok"] += sum(len(t) for t in out.sampled_token_ids)
        w["seqs"] += len(so.num_scheduled_tokens)
        if w["n"] >= self._log_every:
            n = w["n"]
            el = time.perf_counter() - w["t0"]
            lm = getattr(self.worker, "last_ms", None)
            logger.info("phys adapter: last %d steps: worker %.2f ms/step (cost model %.2f), engine-loop wall %.2f ms/step, "
                        "%.0f rows / %.0f seqs per step, %.0f tok/s out%s", n, w["wall"] / n * 1e3, w["est"] / n * 1e3,
                        el / n * 1e3, w["rows"] / n, w["seqs"] / n, w["tok"] / el if el > 0 else 0.0,
                        f"; last worker step (copies, tables, ctx, engine) ms {lm}" if lm else "")
            self._win = self._new_window()

    @staticmethod
    def _new_window() -> dict:
        return dict(n=0, wall=0.0, est=0.0, rows=0, tok=0, seqs=0, t0=time.perf_counter())


_DUMP = {"n": 0}


def _dump(ps, grammar) -> None:
    """PHYS_DUMP_STEPS=<dir>[:N]: pickle the first N (default 4) worker inputs (PhysStep, grammar rows) as
    <dir>/step{i}.pkl, for replaying a step through serve's Worker.step without vLLM (engine debugging). The pickles
    hold prompt token ids: keep <dir> in a mode-700 place when the prompts are real traffic."""
    spec = os.environ.get("PHYS_DUMP_STEPS", "")
    if not spec:
        return
    d, _, n = spec.partition(":")
    if _DUMP["n"] >= int(n or 4):
        return
    import pickle

    os.makedirs(d, mode=0o700, exist_ok=True)
    i = _DUMP["n"]
    _DUMP["n"] += 1
    with open(os.path.join(d, f"step{i}.pkl"), "wb") as f:
        pickle.dump(dict(ps=light_step(ps), grammar=grammar), f)
    logger.info("phys adapter: dumped worker input %d to %s", i, d)


def light_step(ps):
    """The worker input without the host-side bookkeeping: no scheduler plan, no Event, and per forced request only its
    spec (ForcedReq.spec(): tag, dump dir, forced ids, n_dump, prompt length) instead of the object, whose recorded
    logits grow with every step (0.5 MB per row)."""
    import copy

    q = copy.copy(ps)
    q.plan = None
    if hasattr(q, "prepared"):
        q.prepared = None
    fr = getattr(ps, "forced_reqs", None)
    if fr:
        q.forced_reqs = {rid: f.spec() for rid, f in fr.items()}
    return q


def snapshot_tables(ps) -> None:
    """Replace the StepBatches' by-reference block / window tables with copies (the part this step uses)."""
    for sb in ps.batches:
        for se in sb.seqs:
            nb = getattr(se, "n_blocks", None)
            se.block_table = list(se.block_table[:nb] if nb is not None else se.block_table)
            se.window_table = dict(se.window_table)


def split_bitmask(go, so) -> dict:
    """vLLM's GrammarOutput -> rid -> GrammarRows. Rows are laid out per structured request in
    structured_output_request_ids order, 1 + len(scheduled_spec_decode_tokens[rid]) rows each
    (StructuredOutputManager.grammar_bitmask)."""
    out = {}
    bm = go.grammar_bitmask
    row = 0
    acc = go.num_acceptable_drafts or [None] * len(go.structured_output_request_ids)
    for rid, a in zip(go.structured_output_request_ids, acc):
        n = 1 + len(so.scheduled_spec_decode_tokens.get(rid, ()))
        out[rid] = GrammarRows(np.array(bm[row:row + n], copy=True), n - 1 if a is None else int(a))
        row += n
    assert row == bm.shape[0], (row, bm.shape)
    return out


def merge_results(so, ps, results: list, ex: PhysExecutor) -> ModelRunnerOutput:
    """The ranks' StepResults -> one ModelRunnerOutput in the order of so.num_scheduled_tokens."""
    tokens, logp, drafts = {}, {}, {}
    for res in results:
        tokens.update(res.tokens)
        logp.update(res.logprobs or {})
        drafts.update(getattr(res, "drafts", None) or {})
    for rid, e in (getattr(ps, "forced", None) or {}).items():     # FORCE hook: the forced ids are what was sampled
        fr = ps.forced_reqs[rid]
        if list(tokens.get(rid, ())) != e:
            fr.overrides += 1
        fed = None
        for res in results:
            fed = (getattr(res, "fed", None) or {}).get(rid, fed)
        if fed is not None:
            fr.input_mismatch += sum(1 for a, b in zip(fed[1:], e[:-1]) if a != b)
        tokens[rid] = list(e)
    for fr in (getattr(ps, "forced_reqs", None) or {}).values():
        fr.record(results)
        if fr.done() and not fr.flushed:
            fr.flush(ps.vocab_size)
    req_ids = list(so.num_scheduled_tokens)
    sampled = [list(tokens.get(rid, ())) for rid in req_ids]
    if ps.plan is not None:
        for w in ps.plan.work:
            got = len(tokens.get(w.rid, ()))
            if w.kind == "decode":
                assert 1 <= got <= w.n, (w.rid, got, w.n, "decode group without 1..k+1 tokens")
            else:
                assert got == (1 if w.last_chunk else 0), (w.rid, got, w.last_chunk)
    lp = None
    if logp:
        base = {}
        if ps.plan is not None:
            for w in ps.plan.work:
                base[w.rid] = w.start if w.kind == "decode" else w.start + w.n - 1
        logp = {rid: _lp_rows(rows, tokens.get(rid, ()), base.get(rid, 0)) for rid, rows in logp.items()}
        width = 1 + max((len(r[0]) - 1 for v in logp.values() for r in v), default=0)
        ids, vals, ranks, cu = [], [], [], []
        n = 0
        for rid, toks in zip(req_ids, sampled):
            cu.append(n)
            rows = logp.get(rid)
            for j in range(len(toks)):
                if rows is not None and j < len(rows):
                    i_, v_, r_ = rows[j]
                else:
                    i_, v_, r_ = [toks[j]], [0.0], 1
                ids.append(list(i_) + [0] * (width - len(i_)))
                vals.append(list(v_) + [float("-inf")] * (width - len(v_)))
                ranks.append(r_)
                n += 1
        lp = LogprobsLists(np.asarray(ids, dtype=np.int64), np.asarray(vals, dtype=np.float32),
                           np.asarray(ranks, dtype=np.int64), cu)
    if drafts:
        ids = [rid for rid in req_ids if rid in drafts]
        ex._drafts = DraftTokenIds(ids, [list(drafts[rid]) for rid in ids])
    else:
        ex._drafts = None
    out = ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                            sampled_token_ids=sampled, logprobs=lp)
    if ex.async_mode:            # PhysScheduler (ASYNC + PHYS_ASYNC_GRAMMAR=hold) validates them after the update
        out.phys_drafts = ex._drafts
    return out


_LP_WARNED = []


def _lp_rows(rows, toks, base: int) -> list:
    """Per generated token (ids [1 + n], logprobs [1 + n], rank of the sampled token), sampled token first (vLLM's
    LogprobsLists row). Accepts serve's StepResult format [(position, [(id, logprob)] x top-k)] (the row at position p
    scores the token at p + 1; generated token j of a step comes from the row at base + j) or rows already in this
    form."""
    if not rows:
        return []
    r0 = rows[0]
    if not (isinstance(r0, (tuple, list)) and len(r0) == 2 and isinstance(r0[1], (list, tuple))):
        return rows
    bypos = {int(p): lst for p, lst in rows}
    out = []
    for j, t in enumerate(toks):
        lst = bypos.get(base + j, [])
        ids = [int(i) for i, _ in lst]
        lps = [float(v) for _, v in lst]
        if t in ids:
            k = ids.index(t)
            own, rank = lps[k], k + 1
        else:
            if not _LP_WARNED:
                _LP_WARNED.append(1)
                logger.warning("phys adapter: a sampled token is not in its row's top-k logprobs; its logprob is "
                               "reported as the k-th one (an upper bound): the head should return the sampled token's own")
            own, rank = (lps[-1] if lps else 0.0), len(ids) + 1
        out.append(([t] + ids, [own] + lps, rank))
    return out


# ---------------------------------------------------------------------------------------------- workers
def make_worker(kind: str, vllm_config):
    k = vllm_config.num_speculative_tokens
    if kind == "fake":
        from adapter.fake_worker import FakeWorker

        geom = bridge.geometry_from_env()
        tok = None
        if os.environ.get("PHYS_FAKE_SCRIPT"):
            from vllm.tokenizers import get_tokenizer

            mc = vllm_config.model_config
            tok = get_tokenizer(mc.tokenizer, tokenizer_mode=mc.tokenizer_mode, trust_remote_code=mc.trust_remote_code)
        eos = _eos_id(vllm_config)
        vocab = vllm_config.model_config.get_vocab_size()
        return FakeWorker(geom, vocab, eos, k, tok)
    if kind == "stage4":
        from adapter.stage4_worker import Stage4Worker

        return Stage4Worker(vllm_config)
    raise ValueError(f"PHYS_ADAPTER_WORKER={kind!r}")


def _eos_id(vllm_config):
    hf = vllm_config.model_config.hf_config
    gen = getattr(vllm_config.model_config, "try_get_generation_config", None)
    eos = None
    if gen is not None:
        try:
            eos = gen().get("eos_token_id")
        except Exception:  # noqa: BLE001
            eos = None
    if eos is None:
        eos = getattr(hf, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return eos
