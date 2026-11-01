"""DRAFT TAIL hook: the tail of the DSpark draft step (draft head + Markov sampling) runs on a side
stream, concurrently with the next step's eager input preparation on the main stream. DS41H_DRAFT_TAIL=1 (default 0).

Stock order on the main stream (vLLM 0.32.0 V2 runner, decode):
  verify chain -> drafter FULL graph [backbone forward | draft head + Markov loop] -> post-propose (draft_tokens
  index_put, set_draft_tokens) -> next step: prepare_inputs (... combine_sampled_and_draft_tokens ...), prepare_attn,
  model_state.prepare_attn (attention metadata), model_state.prepare_inputs -> target forward.
Of the next step's preparation only combine_sampled_and_draft_tokens (input_ids <- draft tokens) reads the drafter's
tail output; the rest depends on the verification result and the scheduler only (~35 eager launches, ~130 us at 350 W).

With the flag
  * DSparkSpeculator.capture: the FULL graph of each batch descriptor holds only the backbone forward (graph A, the
    stock manager's graph); _sample_sequential (draft head + Markov loop) is captured per descriptor as graph B in a
    pool of its own. Eager (non-captured) draft steps run stock (_generate_draft unchanged outside capture).
  * propose -> query_cudagraph_manager.run_fullgraph(desc): replays A on the main stream, then -- inside
    GPUModelRunner.sample_tokens -- B on the tail stream (after the main stream), and makes the tail stream current for
    the rest of sample_tokens (post-propose ops), then records an event there and restores the main stream. Outside
    sample_tokens (dummy / warm-up runs) B replays on the main stream right after A.
  * next step: combine_sampled_and_draft_tokens (module global of model_runner) is deferred: logits_indices is
    allocated as stock and returned, and the stock kernel is launched with the same arguments after the main stream
    waits for the tail event, at the start of model_state.prepare_inputs (after the attention metadata build, before
    the target forward). Between the stock call site and that point nothing writes the kernel's inputs or reads its
    outputs (input_ids, logits_indices). Steps with new / finished / preempted / resumed requests, encoder inputs or
    dummy runs wait for the tail event before any work instead.
Target numerics: unchanged (same kernels, same arguments, same values; only the stream schedule changes).
Drafter numerics: unchanged (graph A + graph B = the stock graph's kernels).
Guards: source hashes of the stock functions whose structure this relies on; any mismatch leaves everything stock.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.tail")

TAIL = flags.ENABLED and flags._flag("DS41H_DRAFT_TAIL", "0")
DEFER = TAIL and flags._flag("DS41H_DRAFT_TAIL_DEFER", "1")
# DS41H_DRAFT_TAIL=2 (FULL): the WHOLE draft step (backbone graph A + head/Markov graph B + post-propose) runs on the tail
# stream, so the next step's eager input preparation on the main stream (~45 launches, ~120-170 us at 350 W) overlaps the
# drafter's backbone instead of only its head + Markov loop. The drafter then reads the query slot mappings and the draft
# KV-cache groups' block tables from PRIVATE copies (DraftBlockTables): the main stream's next-step
# gather_block_tables / compute_slot_mappings rewrite the runner's shared BlockTables buffers for every group while the
# drafter graph is still running.
FULL = TAIL and os.environ.get("DS41H_DRAFT_TAIL", "0").strip() == "2"
_EAGER_EMB: list = [None]  # patch_drafter.eager_embed when DS41H_DRAFTER_EMBED_EAGER is active (set at capture)
# LATE FORK (default with FULL; DS41H_DRAFT_TAIL_FORK=early = the original FULL schedule): the main stream
# resumes the next step's eager prep only at a fork point INSIDE graph A -- an external event recorded (captured as an event-record
# node) right before the LAST drafter layer's FFN (after its v3 prologue / sparse attention / o-proj / AR / FFN-side mHC). The
# early-fork FULL schedule deadlocked in situ (dr4a_t2m, 17:24: "ds41h prologue HANG: wait 2 ... value 96 target 128" -> trap):
# the v3 prologue triggers launch_dependents at its start and needs all its CTA clusters co-resident; with eager prep kernels of
# the main stream occupying SMs, the PDL-launched sparse attention's CTAs took the SMs the remaining prologue clusters needed.
# After the fork point graph A + B only run kernels without that hazard (o-proj / AR / tilelang mHC / router / MoE / draft head /
# Markov loop -- the latter two already ran next to the eager prep in TAIL=1), so the prep (~160 us with META, ~270 without)
# overlaps the last layer's FFN + graph B (~175 us) instead of the whole drafter. A graph captured without the fork node (no
# drafter FFN override on its last layer) falls back to "main waits for the end of graph A" (= the TAIL=1 schedule).
LATE_FORK = FULL and os.environ.get("DS41H_DRAFT_TAIL_FORK", "late").strip().lower() != "early"
# DS41H_DRAFT_TAIL_EARLY_COMBINE=1 (opt-in, with DEFER; touches the runner's request state -> correctness gate): the fused Markov
# kernel also writes the draft tokens into the runner's req_states.draft_tokens at each request-state index (the values the stock
# post-propose `req_states.draft_tokens[idx_mapping] = draft_tokens` writes later on the tail stream), and the next step's
# deferred combine waits for an event recorded right after graph B instead of the end of the post-propose ops (in situ the
# index_put + gather + event sit between the Markov loop and the combine: ~13 us of the FULL-tail critical path). Only for steps
# that schedule all num_speculative_steps drafts (no -1 masking after propose) and graphs B captured with the fused kernel.
EARLY_COMBINE = DEFER and flags._flag("DS41H_DRAFT_TAIL_EARLY_COMBINE", "0")
# DS41H_DRAFT_TAIL_MERGE=1 (opt-in, FULL only): graph A and graph B are captured as ONE graph (the stock _generate_draft
# capture: backbone forward + draft head + Markov loop). In FULL both replayed back to back on the tail stream anyway (the
# late fork lets main resume inside A), so the split only cost a graph-to-graph launch boundary on the critical path
# (~5 us in situ: _hc_head_reduce_store end 630.2 -> graph B's first kernel 635.2, dr4b_late).
MERGE = FULL and flags._flag("DS41H_DRAFT_TAIL_MERGE", "0")

# sha1[:16] of inspect.getsource(...) in vLLM 0.32.0
_SHA = {
    "DSparkSpeculator._generate_draft": "0307ef7bb12d28ca",
    "DFlashSpeculator.capture": "e20cc5a413eb1497",
    "DFlashCudaGraphManager.capture": "bf4b9e5badd93888",
    "combine_sampled_and_draft_tokens": "28f19ec59c2ad224",
    "GPUModelRunner.sample_tokens": "5bfae876cc8cbf08",
}

# process state (one TP rank / device per worker process)
_ST: dict = {
    "active": False,  # inside GPUModelRunner.sample_tokens
    "switched": None,  # (main, tail) streams while the tail stream is current
    "event": None,  # tail event not yet waited on by the main stream
    "deferred": None,  # deferred combine_sampled_and_draft_tokens launch
    "defer_ok": False,  # set by the execute_model wrapper for steps whose combine may be deferred
    "stats": {"tail": 0, "deferred": 0, "early_wait": 0, "late_flush": 0},
    "main_wait": None,  # FULL late fork: event the main stream waits for when sample_tokens restores it
    "early_ok": False,  # EARLY_COMBINE: this step schedules all drafts (set by sample_tokens)
    "ev_combine": None,  # EARLY_COMBINE: tail event right after graph B (the deferred combine waits for it)
}
_STREAMS: dict = {}
_FORK: dict = {"event": None, "rec": False, "capturing": False}  # external fork event; recorded in the graph A being captured
# Xid 31 fix. PyTorch (2.13) caches ONE cuBLAS/cuBLASLt workspace (32 MiB on sm90) per
# (handle, stream) and frees the one of a graph's capture stream when that torch.cuda.CUDAGraph is destroyed (CUDAGraph::reset);
# graphs still alive that captured cuBLAS calls on that stream keep the freed pointer. The old capture() kept the graphs B of
# vLLM's CUDA-graph memory-profiling pass in a dict on the speculator and replaced them one by one while capturing the real
# ones, all on torch.cuda.graph's shared default capture stream: the first real graph B (the max_num_seqs draft batch) captured
# the profiling workspace, the replacement right after freed it, the profiling pool was released at the first allocator cache
# release, and cuBLASLt's draft-head GEMM (M = 3 x max_num_seqs >= 320 rows) then zeroed a word at its base -> Xid 31
# (HUBCLIENT_FE VIRT_WRITE). Now: (1) every capture() captures its graphs B on a stream of its own (workspace out of reach of
# any other graph's destruction), (2) graphs B of a dropped cudagraph manager are destroyed BEFORE the new captures, (3) a key
# re-captured by the same manager retires its old graph to _RETIRED instead of destroying it while its stream is in use.
_RETIRED: list = []


def mark_fork() -> None:
    """Called by the drafter FFN override of the LAST drafter layer, before its first launch: during the capture of a FULL-tail
    graph A, record the external fork event there (an event-record node of the graph)."""
    if not (LATE_FORK and _FORK["capturing"]):
        return
    import torch

    if not torch.cuda.is_current_stream_capturing():
        return
    ev = _FORK["event"]
    if ev is None:
        ev = _FORK["event"] = torch.cuda.Event(external=True)
    ev.record()
    _FORK["rec"] = True


def _sha(fn) -> str:
    try:
        return hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "?"


def _check(name: str, fn) -> bool:
    got = _sha(fn)
    if got != _SHA[name]:
        logger.warning("ds41h tail: %s source hash %s != %s (vLLM changed); draft tail stays stock", name, got, _SHA[name])
        return False
    return True


def tail_stream(dev):
    import torch

    st = _STREAMS.get(str(dev))
    if st is None:
        st = _STREAMS[str(dev)] = torch.cuda.Stream(device=dev)
    return st


def wait_tail() -> None:
    """Make the current (main) stream wait for the last draft tail (no-op if none is pending)."""
    ev = _ST["event"]
    if ev is not None:
        import torch

        torch.cuda.current_stream().wait_event(ev)
        _ST["event"] = None


# ------------------------------------------------------------------------------------------------ speculator
def patch_dspark(module) -> None:
    import torch

    from ds41h.patch_core import mark_active
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dfcg
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    cls = module.DSparkSpeculator
    if not (_check("DSparkSpeculator._generate_draft", cls._generate_draft)
            and _check("DFlashSpeculator.capture", DFlashSpeculator.capture)
            and _check("DFlashCudaGraphManager.capture", dfcg.DFlashCudaGraphManager.capture)):
        return
    base_gen = cls._generate_draft
    base_capture = cls.capture

    def _generate_draft(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                        cudagraph_runtime_mode=None):
        stash = self.__dict__.get("_ds41h_tail_stash")
        if cudagraph_runtime_mode is None:
            from vllm.config.compilation import CUDAGraphMode

            cudagraph_runtime_mode = CUDAGraphMode.NONE
        if stash is None or not torch.cuda.is_current_stream_capturing():
            return base_gen(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                            cudagraph_runtime_mode)
        _FORK["capturing"], _FORK["rec"] = True, False
        if MERGE:
            # FULL + MERGE: the whole stock draft step (backbone + draft head + Markov loop) in this one graph
            self.__dict__.pop("_ds41h_req_draft_used", None)
            try:
                out = base_gen(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                               cudagraph_runtime_mode)
            finally:
                _FORK["capturing"] = False
            early = bool(self.__dict__.pop("_ds41h_req_draft_used", False))
            stash.append((num_reqs, num_tokens_padded, None, _FORK["rec"], ("merged", early)))
            return out
        # graph A: the backbone forward only (stock _generate_draft minus _sample_sequential)
        try:
            head_hidden = self._run_model(num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                                          cudagraph_runtime_mode)
        finally:
            _FORK["capturing"] = False
        stash.append((num_reqs, num_tokens_padded, head_hidden, _FORK["rec"], None))

    def capture(self):
        try:
            from ds41h import patch_drafter as _pd

            _EAGER_EMB[0] = _pd.eager_embed if _pd.EMBED_EAGER else None
        except Exception:  # noqa: BLE001
            _EAGER_EMB[0] = None
        # Xid 31 fix (2): graphs B of a cudagraph manager vLLM has dropped (its CUDA-graph memory-profiling pass) can never be
        # replayed again -- destroy them now, before any capture below, never in the middle of it.
        import gc
        import weakref

        mgr_now = getattr(self, "query_cudagraph_manager", None)
        owner = self.__dict__.get("_ds41h_tail_owner")
        if owner is None or owner() is not mgr_now:
            stale = self.__dict__.pop("_ds41h_tail_graphs", None)
            if stale:
                n_stale = len(stale)
                stale.clear()
                del stale
                gc.collect()
                torch.cuda.synchronize()
                logger.info("ds41h tail: released %d draft-tail graphs of a dropped cudagraph manager before capturing", n_stale)
            self._ds41h_tail_owner = weakref.ref(mgr_now) if mgr_now is not None else None
        self._ds41h_tail_stash = []
        try:
            base_capture(self)
        finally:
            stash = self.__dict__.pop("_ds41h_tail_stash", None) or []
        mgr = self.query_cudagraph_manager
        # kept across capture() calls of the SAME manager: every backbone-only graph it may still replay needs its tail graph
        graphs: dict = self.__dict__.setdefault("_ds41h_tail_graphs", {})
        if stash:
            pool = torch.cuda.graph_pool_handle()
            # Xid 31 fix (1): a capture stream of this capture() call only (kept alive with the graphs)
            cap_stream = torch.cuda.Stream()
            self.__dict__.setdefault("_ds41h_tail_cap_streams", []).append(cap_stream)
            for num_reqs, num_tokens, head_hidden, fork, merged in stash:
                old = graphs.pop((num_reqs, num_tokens), None)
                if old is not None:  # Xid 31 fix (3): retire, never destroy, a graph while graphs of its stream are in use
                    _RETIRED.append(old)
                if merged is not None:  # MERGE: graph B lives inside the manager's graph
                    graphs[(num_reqs, num_tokens)] = (None, None, fork, merged[1], True)
                    continue
                g = torch.cuda.CUDAGraph()
                self.__dict__.pop("_ds41h_req_draft_used", None)
                with torch.cuda.graph(g, pool=pool, stream=cap_stream):
                    self._sample_sequential(num_reqs, head_hidden)
                early = bool(self.__dict__.pop("_ds41h_req_draft_used", False))  # EARLY_COMBINE: graph B writes req_states
                graphs[(num_reqs, num_tokens)] = (g, head_hidden, fork, early)
            torch.cuda.synchronize()
        base_run = mgr.__dict__.get("_ds41h_base_run")
        if base_run is None:
            base_run = mgr._ds41h_base_run = mgr.run_fullgraph
        max_reqs = mgr.max_num_reqs

        def switch() -> None:
            if _ST["active"] and _ST["switched"] is None:
                main = torch.cuda.current_stream()
                tail = tail_stream(main.device)
                tail.wait_stream(main)
                torch.cuda.set_stream(tail)
                _ST["switched"] = (main, tail)
                _ST["stats"]["tail"] += 1

        full = FULL and isinstance(getattr(self, "block_tables", None), DraftBlockTables)

        def run_fullgraph(desc, *args, **kwargs):
            if _ST["switched"] is None:
                wait_tail()  # a dummy / warm-up draft step after a real one: its graphs reuse the tail's buffers
            key_reqs = desc.num_reqs or min(desc.num_tokens, max_reqs)
            entry = graphs.get((key_reqs, desc.num_tokens))
            if entry is None:
                raise RuntimeError(f"ds41h tail: no draft-tail graph for {desc}")
            if _EAGER_EMB[0] is not None:
                # DS41H_DRAFTER_EMBED_EAGER: the drafter input embedding on a side stream (forked from main, before the
                # switch to the tail stream); graph A waits for it through its external event-wait node
                _EAGER_EMB[0](self, key_reqs)
            if full:
                bt = _SYNC["bt"]
                if bt is not None:  # no batched prepare consumed the private-table sync: plain copies (main, before the fork)
                    bt.sync(_SYNC["n"])
                    _SYNC["bt"] = None
                    _ST["stats"]["plain_sync"] = _ST["stats"].get("plain_sync", 0) + 1
                switch()  # graph A on the tail stream too (FULL)
            out = base_run(desc, *args, **kwargs)
            if full and LATE_FORK and _ST["switched"] is not None:
                if entry[2] and _FORK["event"] is not None:
                    _ST["main_wait"] = _FORK["event"]  # main resumes at the fork node inside graph A
                    _ST["stats"]["late_fork"] = _ST["stats"].get("late_fork", 0) + 1
                else:
                    ev_a = torch.cuda.Event()
                    ev_a.record(_ST["switched"][1])  # no fork node in this graph: main resumes after graph A (TAIL=1)
                    _ST["main_wait"] = ev_a
                    _ST["stats"]["fork_fallback"] = _ST["stats"].get("fork_fallback", 0) + 1
            switch()
            if not (len(entry) > 4 and entry[4]):  # MERGE: the manager's graph already ran graph B's part
                entry[0].replay()
            if EARLY_COMBINE and len(entry) > 3 and entry[3] and _ST["early_ok"] and _ST["switched"] is not None:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())  # right after graph B (req_states.draft_tokens written by it)
                _ST["ev_combine"] = ev
                _ST["stats"]["early_combine"] = _ST["stats"].get("early_combine", 0) + 1
            return out

        mgr.run_fullgraph = run_fullgraph
        logger.info("ds41h tail: %d draft-tail graphs (draft head + Markov) captured apart from the backbone graphs",
                    len(graphs))

    _generate_draft.__wrapped__ = base_gen
    capture.__wrapped__ = base_capture
    cls._generate_draft = _generate_draft
    cls.capture = capture
    if FULL:
        patch_full(cls)
    mark_active("tail", "DSparkSpeculator.{_generate_draft,capture}(draft tail graph on a side stream)")


# ------------------------------------------------------------------------------------------------ private block tables
class DraftBlockTables:
    """The drafter's view of the runner's BlockTables (FULL tail): private slot_mappings [groups, max_tokens] and private
    input block tables for the draft KV-cache groups; every other attribute / method is the runner's. The private draft
    rows are refreshed from the runner's gathered tables at the start of each propose (main stream, before the fork), i.e.
    the drafter sees exactly the tables stock would read; the drafter graphs are captured against these buffers."""

    def __init__(self, real, draft_gids):
        import torch

        self._real = real
        self._gids = tuple(int(g) for g in draft_gids)
        self.slot_mappings = torch.zeros_like(real.slot_mappings)
        self.input_block_tables = list(real.input_block_tables)
        for g in self._gids:
            self.input_block_tables[g] = torch.zeros_like(real.input_block_tables[g])

    def __getattr__(self, name):
        return getattr(self.__dict__["_real"], name)

    def get_dummy_block_tables(self, num_reqs: int):
        return tuple(t[:num_reqs].zero_() if i in self._gids else t[:num_reqs] for i, t in enumerate(self.input_block_tables))

    def get_dummy_slot_mappings(self, num_tokens: int):
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]

    def gather_block_tables(self, *a, **kw):
        raise RuntimeError("ds41h tail: DraftBlockTables.gather_block_tables (pcp) is not supported in FULL mode")

    def sync(self, num_reqs: int) -> None:
        """Copy the runner's gathered rows of the draft groups (this step's batch) into the private tables."""
        real = self._real.input_block_tables
        for g in self._gids:
            self.input_block_tables[g][:num_reqs].copy_(real[g][:num_reqs])

    def sources(self) -> dict:
        return {self.input_block_tables[g].data_ptr(): self._real.input_block_tables[g] for g in self._gids}


# A FULL-tail propose whose private-table sync is folded into the drafter's batched prepare kernel (patch_drafter
# _launch_prepare: the kernel reads the runner's rows and copies them); if no prepare launch consumed it, the propose
# wrapper syncs with plain copies before returning control to the graph replay (see patch_full.propose).
_SYNC: dict = {"bt": None, "n": 0}


def pending_sources() -> dict | None:
    bt = _SYNC["bt"]
    return bt.sources() if bt is not None else None


def mark_synced() -> None:
    _SYNC["bt"] = None


def patch_full(cls) -> None:
    """FULL tail: the drafter gets DraftBlockTables right after set_attn (before its graphs are captured)."""
    from ds41h.patch_core import mark_active

    base_set_attn = cls.set_attn
    base_propose = cls.propose

    def set_attn(self, *args, **kwargs):
        base_set_attn(self, *args, **kwargs)
        bt = self.block_tables
        if (getattr(self, "pcp_manager", None) is None and getattr(bt, "cp_size", 1) == 1 and not isinstance(bt, DraftBlockTables)
                and getattr(self, "draft_kv_cache_group_ids", None)):
            self.block_tables = DraftBlockTables(bt, self.draft_kv_cache_group_ids)
            logger.info("ds41h tail: FULL draft tail -- drafter uses private slot mappings / block tables of groups %s",
                        list(self.draft_kv_cache_group_ids))
        else:
            logger.warning("ds41h tail: FULL draft tail unavailable (pcp / cp / no draft groups); head + Markov tail only")

    def propose(self, input_batch, *args, **kwargs):
        # the memory-profile run proposes before set_attn (no block_tables yet, dummy_run=True): stock path
        bt = None if kwargs.get("dummy_run", False) else getattr(self, "block_tables", None)
        if isinstance(bt, DraftBlockTables):
            from ds41h import patch_drafter

            if patch_drafter.PREP:
                _SYNC["bt"] = bt  # folded into the batched prepare launch (reads the runner's rows, writes the private ones)
                _SYNC["n"] = input_batch.num_reqs
            else:
                bt.sync(input_batch.num_reqs)
        try:
            return base_propose(self, input_batch, *args, **kwargs)
        finally:
            _SYNC["bt"] = None

    set_attn.__wrapped__ = base_set_attn
    propose.__wrapped__ = base_propose
    cls.set_attn = set_attn
    cls.propose = propose
    mark_active("tail", "DSparkSpeculator.{set_attn,propose}(FULL: private draft block tables)")


# ------------------------------------------------------------------------------------------------ runner
def _steady(so) -> bool:
    """Decode step with the same request set as the previous one (no state writes that could race the tail)."""
    if so is None or so.scheduled_new_reqs or so.finished_req_ids or getattr(so, "preempted_req_ids", None):
        return False
    if so.scheduled_encoder_inputs or getattr(so.scheduled_cached_reqs, "resumed_req_ids", None):
        return False
    return True


def _flush(late: bool = False) -> None:
    d = _ST["deferred"]
    if d is None:
        return
    _ST["deferred"] = None
    ev = _ST["ev_combine"]
    _ST["ev_combine"] = None
    if ev is not None:  # EARLY_COMBINE: graph B already wrote req_states.draft_tokens; the full tail event stays pending
        import torch

        torch.cuda.current_stream().wait_event(ev)
    else:
        wait_tail()
    import triton

    from vllm.v1.worker.gpu.input_batch import _combine_sampled_and_draft_tokens_kernel

    (input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, draft_tokens, cu_num_logits,
     logits_indices, num_new) = d
    # stock combine_sampled_and_draft_tokens (input_batch.py:466-504) with the pre-allocated logits_indices
    _combine_sampled_and_draft_tokens_kernel[(idx_mapping.shape[0],)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new,
        BLOCK_SIZE=triton.next_power_of_2(draft_tokens.shape[-1] + num_new),
    )
    if late:
        _ST["stats"]["late_flush"] += 1
        if _ST["stats"]["late_flush"] <= 3:
            logger.error("ds41h tail: deferred combine flushed after the forward was issued (model_state.prepare_inputs "
                         "not called?) -- set DS41H_DRAFT_TAIL_DEFER=0")


def patch_runner(module) -> None:
    import torch

    from ds41h.patch_core import mark_active

    cls = module.GPUModelRunner
    if not (_check("GPUModelRunner.sample_tokens", cls.sample_tokens)
            and _check("combine_sampled_and_draft_tokens", module.combine_sampled_and_draft_tokens)):
        return
    base_sample_tokens = cls.sample_tokens
    base_execute = cls.execute_model
    base_combine = module.combine_sampled_and_draft_tokens

    def sample_tokens(self, *args, **kwargs):
        wait_tail()  # normally consumed by this step's combine already
        _ST["ev_combine"] = None
        st = getattr(self, "execute_model_state", None)
        ib = getattr(st, "input_batch", None) if st is not None else None
        nst = getattr(st, "num_spec_tokens_to_schedule", None) if st is not None else None
        _ST["early_ok"] = bool(EARLY_COMBINE and nst is not None and nst == getattr(self, "num_speculative_steps", -1))
        _ST["active"] = True
        try:
            return base_sample_tokens(self, *args, **kwargs)
        finally:
            _ST["active"] = False
            sw, _ST["switched"] = _ST["switched"], None
            if sw is not None:
                main, tail = sw
                if ib is not None:  # main-stream allocations read by the post-propose ops on the tail stream
                    for v in vars(ib).values():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            v.record_stream(tail)
                ev = torch.cuda.Event()
                ev.record(tail)
                torch.cuda.set_stream(main)
                _ST["event"] = ev
                mw, _ST["main_wait"] = _ST.get("main_wait"), None
                if mw is not None:  # FULL late fork: the next step's prep starts at the fork point (see LATE_FORK)
                    main.wait_event(mw)

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = bool(kwargs.get("dummy_run", args[1] if len(args) > 1 else False))
        # deferral only where nothing between combine and model_state.prepare_inputs reads input_ids / logits_indices
        ok = (DEFER and not dummy and _steady(scheduler_output) and getattr(self, "fast_prefill", None) is None
              and not getattr(self, "uses_inputs_embeds", False) and getattr(self, "pcp_manager", None) is None
              and getattr(self, "ubatch_runner", None) is None and getattr(self, "is_first_pp_rank", True))
        if _ST["event"] is not None and not ok:
            wait_tail()
            _ST["stats"]["early_wait"] += 1
        _ST["defer_ok"] = ok
        ms = getattr(self, "model_state", None)
        if DEFER and ms is not None and "_ds41h_tail_pi" not in ms.__dict__:
            base_pi = ms.prepare_inputs

            def prepare_inputs(*a, **kw):
                _flush()
                return base_pi(*a, **kw)

            ms._ds41h_tail_pi = base_pi
            ms.prepare_inputs = prepare_inputs
        try:
            return base_execute(self, scheduler_output, *args, **kwargs)
        finally:
            _ST["defer_ok"] = False
            if _ST["deferred"] is not None:
                _flush(late=True)

    def combine_sampled_and_draft_tokens(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
                                         prefill_len, draft_tokens, cu_num_logits, num_logits,
                                         num_new_sampled_tokens=1):
        if _ST["event"] is None or not _ST["defer_ok"] or _ST["deferred"] is not None or num_new_sampled_tokens not in (0, 1):
            wait_tail()
            return base_combine(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len,
                                draft_tokens, cu_num_logits, num_logits, num_new_sampled_tokens)
        logits_indices = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
        _ST["deferred"] = (input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len,
                           draft_tokens, cu_num_logits, logits_indices, num_new_sampled_tokens)
        _ST["stats"]["deferred"] += 1
        return logits_indices

    sample_tokens.__wrapped__ = base_sample_tokens
    execute_model.__wrapped__ = base_execute
    combine_sampled_and_draft_tokens.__wrapped__ = base_combine
    cls.sample_tokens = sample_tokens
    cls.execute_model = execute_model
    if EARLY_COMBINE and hasattr(cls, "capture_model"):
        base_capture_model = cls.capture_model

        def capture_model(self, *a, **kw):
            # the speculator's graph B captures the fused Markov kernel with the runner's req_states.draft_tokens as an extra output
            sp = getattr(self, "speculator", None)
            rd = getattr(getattr(self, "req_states", None), "draft_tokens", None)
            if sp is not None and isinstance(rd, torch.Tensor) and rd.dtype == torch.int64 and rd.dim() == 2 and rd.stride(1) == 1:
                sp._ds41h_req_draft = rd
            return base_capture_model(self, *a, **kw)

        cls.capture_model = capture_model
        mark_active("tail", "GPUModelRunner.capture_model(early combine: graph B writes req_states.draft_tokens)")
    module.combine_sampled_and_draft_tokens = combine_sampled_and_draft_tokens
    mark_active("tail", "GPUModelRunner.{sample_tokens,execute_model} + combine_sampled_and_draft_tokens (deferred)")


def _register(name: str, fn) -> None:
    import sys

    register(name)(fn)
    if name in sys.modules:  # imported before the hook saw it
        fn(sys.modules[name])


if TAIL:
    _register("vllm.v1.worker.gpu.spec_decode.dspark.speculator", patch_dspark)
    _register("vllm.v1.worker.gpu.model_runner", patch_runner)
