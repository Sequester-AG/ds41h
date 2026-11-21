"""core_sched.py

  * per step Python only converts at the vLLM / executor boundary: SchedulerOutput fields, serve's StepPlan objects (the
    executor's merge and step-time marks read them), the StepBatches, the PhysStep, and
    the EngineCoreOutputs (C++-encoded msgpack, decoded with vLLM's own MsgpackDecoder);
  * vLLM's Request objects are kept only for the worker's ReqInit (sampling params); their state is not maintained.
Requires PHYS_CXX_SCHED=1 + PHYS_CXX_KV=1 (+ PHYS_CXX_TABLES=1 for the C++ batches). Images (PHYS_MM=1): content keys and
the encoder-cache planner stay in Python (adapter/mm.py, as PhysScheduler), the keys go to the C++ rank scheduler (which
salts the block keys with them). FORCE (DS41H_FORCE_FILE): the force file is matched here at add_request (PhysScheduler:
at _to_rs, the next schedule), the forced ids go to the C++ lifecycle (rank scheduler flag, row tokens, emitted ids,
counters); the ForcedReq objects, the executor's id override and the dumps stay Python (adapter/force.py). Not covered
(rejected with vLLM's error output): streaming input, LoRA, pooling, prompt embeds.
The C++ lifecycle is golden-trace identical to PhysScheduler + vLLM's Scheduler (fpr/core/tests/README_core.md); this
file's conversions are checked by the adapter tests with PHYS_CXX_CORE=1.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.environ.get("PHYS_CXX_CORE_DIR") or os.environ.get("PHYS_CXX_BUILD") or os.path.join(_HERE, "..", "..", "build_py")
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
import fpr_core_py as C  # noqa: E402

from vllm.logger import init_logger  # noqa: E402
from vllm.v1.core.sched.output import CachedRequestData, GrammarOutput, SchedulerOutput  # noqa: E402
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402

from adapter import bridge  # noqa: E402
from adapter.bridge import PhysStep, ReqInit  # noqa: E402
from adapter.force import ForcedReq  # noqa: E402

from interfaces import CPRole, build_step_batch  # noqa: E402  (serve/)
from kv_manager import CopyOp  # noqa: E402
from scheduler import SeqWork, StepPlan  # noqa: E402

logger = init_logger("vllm.phys_adapter")
POOLS = ("paged", "window", "state")
KWARG_VARIANTS = (None, {"thinking": True}, {"thinking": False})
_SEP = "\x1f"
_TINFO: dict = {}                       # (tokenizer name, vocab size) -> TokenizerInfo JSON (once per process)


def reasoner_spec(reasoner) -> dict | None:
    """vLLM's ParserEngine token rules (is_reasoning_end / find_reasoning_end) for the C++ grammar bounds; None when the
    reasoner is not a ParserEngineReasoningAdapter (the C++ rules do not cover it)."""
    from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
    from vllm.parser.engine.parser_engine_config import ParserState

    if not isinstance(reasoner, ParserEngineReasoningAdapter):
        return None
    pe = reasoner._parser_engine
    cfg = pe.parser_engine_config
    wait = cfg.wait_for_reasoning
    if wait is None:
        wait = cfg.initial_state is ParserState.REASONING
    st = cfg.transitions.get((cfg.initial_state, "THINK_START"))
    return {"present": True, "adapter": True, "end_ids": sorted(pe._reasoning_end_token_ids),
            "end_token_id": -1 if pe._reasoning_end_token_id is None else pe._reasoning_end_token_id,
            "start_id": -1 if pe._reasoning_start_token_id is None else pe._reasoning_start_token_id,
            "boundary_ids": sorted(pe._turn_boundary_token_ids), "wait_for_reasoning": bool(wait),
            "start_opens_reasoning": bool(st is not None and st.next_state is ParserState.REASONING)}


def _rid_obj(s: str):
    """the C++ Rid of a CP helper shadow ("cp\\x1f<rid>\\x1f<rank>") -> Python's ("cp", rid, rank) tuple"""
    if s.startswith("cp" + _SEP):
        a = s.split(_SEP)
        if len(a) == 3:
            return ("cp", a[1], int(a[2]))
    return s


class _CoreMixin:
    """PhysScheduler / PhysAsyncScheduler with the request lifecycle in C++ (fpr_core_py.CoreX)."""

    def _core_init(self) -> None:
        assert type(self.rs).__name__ == "CxxRankScheduler", (
            "PHYS_CXX_CORE=1 needs the C++ rank scheduler (PHYS_CXX_SCHED=1, PHYS_CXX_KV=1, a plain CostModel)")
        som = self.structured_output_manager
        tinfo, reasoners = None, []
        tok = getattr(som, "tokenizer", None)
        if tok is not None:
            import xgrammar as xgr
            key = (getattr(tok, "name_or_path", id(tok)), self.vocab_size)
            if key not in _TINFO:
                _TINFO[key] = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=self.vocab_size).serialize_json()
            tinfo = _TINFO[key]
            if som.reasoner_cls is not None and not som.enable_in_reasoning:
                for kw in KWARG_VARIANTS:
                    spec = reasoner_spec(som.reasoner_cls(tokenizer=tok, **(kw or {})))
                    assert spec is not None, "PHYS_CXX_CORE=1: the reasoning parser is not a ParserEngine adapter"
                    reasoners.append(("" if not kw else json.dumps(kw), spec))
            # the C++ core compiles grammars itself (xgrammar 0.2.8's C++ API): EngineCore's Python compile is skipped
            som.grammar_init = lambda request: None
        caps = bridge.get_sampling_caps()
        cfg = dict(k=self.num_spec_tokens, max_model_len=self.max_model_len, page_tokens=self.geom.page_tokens,
                   window_page_tokens=getattr(self.geom, "window_page_tokens", 32), **{"async": self.async_mode},
                   log_stats=bool(self.log_stats), spec_decode_metrics_level=self.spec_decode_metrics_level,
                   include_finished_set=self.finished_req_ids_dict is not None, grammar_hold=bool(self._grammar_hold),
                   grammar_max_frac=self._grammar_max_frac, vocab_size=self.vocab_size, t0=self._t0,
                   sampling_caps=None if caps is None else sorted(caps),
                   enable_in_reasoning=bool(getattr(som, "enable_in_reasoning", False)),
                   disable_any_whitespace=bool(self.vllm_config.structured_outputs_config.disable_any_whitespace))
        cx = self.rs._cx
        if hasattr(self.rs, "external_views"):           # rs.reqs / running / waiting read
            self.rs.external_views()                     # live from C++ (the core drives the scheduler through the capsule)
        self._core_cfg = (cfg, tinfo, reasoners)         # PHYS_CXX_LOOP=1 (2b): the C++ loop's core uses the same config
        self.core = C.CoreX(cfg, cx._ischeduler(), cx.kv_capsule(), self.rs, tinfo, reasoners)
        self._enc = MsgpackEncoder()
        self._dec = MsgpackDecoder(EngineCoreOutputs)
        # rid -> vLLM Request (the worker's ReqInit sampling params): vLLM's own `requests` dict, so tools and tests that
        # read scheduler.requests see the live requests (entries leave on finish / abort; their state is the C++ core's)
        self._vreqs = self.requests
        self._sid: dict = {}                 # id(SchedulerOutput) -> the core's handle
        self._mm_live: set = set()           # rids registered with the image planner (self.mm, PHYS_MM=1)
        self._force_pending: dict = {}       # rid -> force spec matched at add_request (-> self.forced at the C++ _to_rs)
        logger.info("phys adapter: PHYS_CXX_CORE=1: the request lifecycle in C++ (%s; grammar %s, %d reasoning variants)",
                    C.__file__, "on" if tinfo else "off", len(reasoners))

    # ------------------------------------------------------------------ requests in
    def add_request(self, request) -> None:
        sor = request.structured_output_request
        rid = request.request_id
        images, mm_why = None, ""
        if request.mm_features:
            # PhysScheduler's rules (adapter/mm.py): content keys from vLLM's processed tensors, the planner's residency.
            # (Synchronous here; PhysScheduler hashes them on a side thread: a follow-up if large images show up in
            # the host time.)
            mm_why = self._unsupported(request) or ""
            if not mm_why:
                got = self._mm_images(request)
                if isinstance(got, str):
                    mm_why = got
                else:
                    reg = self.mm.add(rid, got)
                    images = [(im.offset, im.length, im.key) for im in reg.images]
                    self._mm_live.add(rid)
            if mm_why:
                logger.error("phys adapter: request %s rejected: %s", rid, mm_why)
        ecr = EngineCoreRequest(
            request_id=request.request_id, prompt_token_ids=request.prompt_token_ids,
            mm_features=None, sampling_params=request.sampling_params,
            pooling_params=request.pooling_params, arrival_time=request.arrival_time, lora_request=request.lora_request,
            cache_salt=request.cache_salt, data_parallel_rank=None, prompt_embeds=request.prompt_embeds,
            client_index=request.client_index, priority=request.priority, trace_headers=request.trace_headers,
            resumable=request.resumable, reasoning_ended=None if sor is None else sor.reasoning_ended,
            reasoning_parser_kwargs=None if sor is None else sor.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately)
        forced = None
        if self.force.enabled:               # PhysScheduler._to_rs's ForceSpec.match (the file as it is now)
            spec = self.force.match(request.prompt_token_ids or ())
            if spec is not None:
                self._force_pending[rid] = spec
                forced = [int(x) for x in spec["forced"]]
        self._vreqs[rid] = request
        if forced is None:
            self.core.add_request(self._enc.encode(ecr)[0], time.monotonic(), time.time(), images, mm_why)
        else:
            self.core.add_request(self._enc.encode(ecr)[0], time.monotonic(), time.time(), images, mm_why, forced)

    def finish_requests(self, request_ids, finished_status):
        if request_ids is None:
            ids = list(self._vreqs)
        elif isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)
        if finished_status != RequestStatus.FINISHED_ABORTED:
            logger.warning("phys adapter (core): finish_requests(%s) treated as an abort", finished_status)
        self.core.abort(ids)
        for r in ids:
            self._req_end(r)
        return [self._vreqs.pop(r) for r in ids if r in self._vreqs]

    def _req_end(self, rid) -> None:
        """PhysScheduler._phys_free's Python part for a request that ended, at the
        same point (inside update / finish_requests): its ForcedReq, and its images' encoder-cache entries."""
        if self.forced or self._force_pending:
            self._force_end(rid)
        if rid in self._mm_live:
            self._mm_live.discard(rid)
            self.mm.forget(rid)

    def _force_end(self, rid) -> None:
        """PhysScheduler._phys_free's FORCE part: the request ended -> pop its
        ForcedReq, write a partial dump if the executor has not written the full one."""
        self._force_pending.pop(rid, None)
        fr = self.forced.pop(rid, None)
        if fr is not None and not fr.flushed:
            fr.flush(self.vocab_size, partial=not fr.done())

    # ------------------------------------------------------------------ schedule
    def _plan(self, t) -> StepPlan:
        step, work, copies, preempted, migrated, est_us, idle_frac = t
        p = StepPlan(step)
        for (rid, rank, kind, start, n, cached_upto, replay_start, last_chunk, score, forced, uid, known, cp, ims) in work:
            w = SeqWork(_rid_obj(rid), rank, kind, start, n, cached_upto, replay_start, last_chunk, score, forced, uid,
                        known)
            if cp is not None:
                g, crid, owner, subs, me = cp
                w.cp = CPRole(g, crid, owner, [tuple(s) for s in subs], me)
            if ims:
                w.images = tuple(tuple(x) for x in ims)
            p.work.append(w)
        p.copies = [CopyOp(POOLS[c[0]], c[1], c[2], c[3], c[4], c[5]) for c in copies]
        p.preempted = list(preempted)
        p.migrated = list(migrated)
        p.est_us, p.idle_frac = est_us, idle_frac
        return p

    def _schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        ts = time.monotonic()
        if self._gc_every and self.current_step % self._gc_every == 0:
            self._gc(ts)
        h, sd, plan_t, rows, prev, nr = self.core.schedule(ts, time.time())
        plan = self._plan(plan_t) if plan_t is not None else None
        num_sched = dict(sd["num_scheduled"])
        spec = {rid: list(t) for rid, t in sd["spec"]}
        so = SchedulerOutput(
            scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_sched, total_num_scheduled_tokens=sd["total"], scheduled_spec_decode_tokens=spec,
            scheduled_encoder_inputs={}, preempted_req_ids=set(sd["preempted"]), finished_req_ids=set(sd["finished"]),
            free_encoder_mm_hashes=[], num_spec_tokens_to_schedule=self.num_spec_tokens)
        so.has_structured_output_requests = sd["has_so"]
        so.pending_structured_output_tokens = sd["pending_so"]
        R = self.geom.n_ranks
        inits = []
        for rid, rank, prompt, output, structured in nr:
            vr = self._vreqs.get(rid)
            inits.append(ReqInit(rid, rank, list(prompt), list(output), None if vr is None else vr.sampling_params,
                                 structured))
        for rid, ok in sd.get("forced_to_rs", ()):   # PhysScheduler._to_rs: matched requests the rank scheduler saw
            fspec = self._force_pending.pop(rid)
            vr = self._vreqs.get(rid)
            self.phys_stats["forced"] += 1
            logger.info("phys adapter: request %s forced (tag %s, %d forced tokens, n_dump %d)", rid, fspec.get("tag"),
                        len(fspec.get("forced", ())), int(fspec.get("n_dump", 0)))
            if ok:
                self.forced[rid] = ForcedReq(rid, fspec, vr.num_prompt_tokens if vr is not None else 0)
        finished = set(sd["finished"])
        for rid in finished:                       # (normally done in update / finish_requests already: _req_end)
            self._vreqs.pop(rid, None)
            if rid in self.forced or rid in self._force_pending or rid in self._mm_live:
                self._req_end(rid)
        if self._mm_live:
            for rid in sd["phys_preempted"]:          # re-admitted later, keep the images, drop cache refs
                if rid in self._mm_live:
                    self.mm.release(rid)
        if plan is None or not plan.work:
            batches = [build_step_batch(plan, self.rs.kv, self.geom, r) for r in range(R)] if plan is not None else []
            ps = PhysStep(step=plan.step if plan is not None else -1, batches=batches, row_tokens=[[] for _ in batches],
                          new_reqs=inits, finished=finished, preempted=set(sd["phys_preempted"]), order=[], plan=plan)
        else:
            batches = [build_step_batch(plan, self.rs.kv, self.geom, r) for r in range(R)]
            ps = PhysStep(step=plan.step, batches=batches, row_tokens=[list(x) for x in rows], prev_tokens=dict(prev),
                          new_reqs=inits, finished=finished, preempted=set(sd["phys_preempted"]), order=list(num_sched),
                          num_spec={rid: len(v) for rid, v in spec.items()}, est_us=plan.est_us, plan=plan)
        ps.forced, ps.forced_reqs, ps.vocab_size = {}, {}, self.vocab_size
        for rid, e, checked, mismatch in sd.get("forced_emits", ()):   # PhysScheduler._phys_step's emits
            fr = self.forced[rid]
            fr.checked_rows += checked
            fr.input_mismatch += mismatch
            ps.forced[rid] = list(e)
            ps.forced_reqs[rid] = fr
        if plan is not None and plan.work and self.mm is not None and self.mm.reqs:   # (PhysScheduler: steps with work)
            ps.mm = self.mm.plan(batches)
        so.phys = ps
        self._sid[id(so)] = h
        self.phys_stats["steps"] += 1
        if not num_sched:
            self.phys_stats["empty_steps"] += 1
        return so

    def get_grammar_bitmask(self, scheduler_output):
        h = self._sid.get(id(scheduler_output))
        if h is None or not scheduler_output.has_structured_output_requests:
            return None
        rids, rows, n, words, acc = self.core.grammar_bitmask(h)
        if not rids:
            return None
        return GrammarOutput(list(rids), np.frombuffer(rows, dtype=np.int32).reshape(n, words), list(acc))

    # ------------------------------------------------------------------ results
    def _phys_update(self, scheduler_output, model_runner_output):
        h = self._sid.pop(id(scheduler_output))
        mo = model_runner_output
        lp = None
        if mo.logprobs is not None:
            l = mo.logprobs
            lp = (l.logprob_token_ids.tolist(), l.logprobs.tolist(), l.sampled_token_ranks.tolist(),
                  list(l.cu_num_generated_tokens or range(len(mo.req_ids))))
        dr = getattr(mo, "phys_drafts", None)
        drafts = None if dr is None else [(r, list(d)) for r, d in zip(dr.req_ids, dr.draft_token_ids)]
        sampled = [list(t) for t in (mo.sampled_token_ids or [[] for _ in mo.req_ids])]
        t = time.monotonic()
        msgs, accepted = self.core.update(h, list(mo.req_ids), sampled, lp, drafts, t, t)
        self.phys_stats.update(self.core.stats())          # the counters the C++ lifecycle owns (PhysScheduler names)
        out = {}
        for client, frames in msgs:
            eco = self._dec.decode(frames)
            for o in eco.outputs:
                if o.finish_reason is not None:
                    self._vreqs.pop(o.request_id, None)
                    self._req_end(o.request_id)
            out[client] = eco
        return out

    def update_draft_token_ids(self, draft_token_ids) -> None:
        self.core.update_draft_token_ids([(r, list(d)) for r, d in zip(draft_token_ids.req_ids,
                                                                        draft_token_ids.draft_token_ids)])

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output) -> None:
        h = self._sid[id(scheduler_output)]
        d = self.core.update_draft_token_ids_in_output(h, [(r, list(x)) for r, x in zip(draft_token_ids.req_ids,
                                                                                         draft_token_ids.draft_token_ids)])
        scheduler_output.scheduled_spec_decode_tokens = {rid: list(t) for rid, t in d["spec"]}
        scheduler_output.num_invalid_spec_tokens = dict(d["num_invalid"])

    # ------------------------------------------------------------------ queries / controls
    def has_requests(self) -> bool:
        return self.core.has_requests()

    def has_unfinished_requests(self) -> bool:
        return self.core.num_unfinished() > 0

    def has_finished_requests(self) -> bool:
        return self.core.has_requests() and self.core.num_unfinished() == 0

    def get_num_unfinished_requests(self) -> int:
        return self.core.num_unfinished()

    def get_request_counts(self):
        running, unfinished = self.core.counts()
        return running, max(0, unfinished - running)

    def set_pause_state(self, pause_state) -> None:
        super().set_pause_state(pause_state)
        self.core.set_pause({"UNPAUSED": 0, "PAUSED_NEW": 1, "PAUSED_ALL": 2}.get(pause_state.name, 0))

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        logger.warning("phys adapter (core): reset_prefix_cache is not supported with PHYS_CXX_CORE=1")
        return False


def install(PS) -> None:
    """adapter/phys_scheduler.py's PHYS_CXX_CORE=1 hook: replace its two scheduler classes by the core subclasses."""
    base, base_async = PS.PhysScheduler, PS.PhysAsyncScheduler

    class CorePhysScheduler(_CoreMixin, base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._core_init()

    class CorePhysAsyncScheduler(_CoreMixin, base_async):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._core_init()

    CorePhysScheduler.__qualname__ = CorePhysScheduler.__name__ = "PhysScheduler"
    CorePhysAsyncScheduler.__qualname__ = CorePhysAsyncScheduler.__name__ = "PhysAsyncScheduler"
    PS.PyPhysScheduler, PS.PyPhysAsyncScheduler = base, base_async
    PS.PhysScheduler, PS.PhysAsyncScheduler = CorePhysScheduler, CorePhysAsyncScheduler
    print(f"phys_scheduler: PHYS_CXX_CORE=1: the request lifecycle in C++ ({C.__file__})", flush=True)
    if os.environ.get("PHYS_CXX_LOOP", "0") == "1":      # stage 2b-α (default off): the engine process's IO + loop in C++
        import importlib.util as ilu
        # PHYS_CXX_CORE2B_PROC=<path>: a frozen core2b_proc.py; default the tree's
        p = os.environ.get("PHYS_CXX_CORE2B_PROC") or os.path.join(_HERE, "..", "core2b", "core2b_proc.py")
        mod = sys.modules.get("core2b_proc")
        if mod is None:
            spec = ilu.spec_from_file_location("core2b_proc", p)
            mod = ilu.module_from_spec(spec)
            sys.modules["core2b_proc"] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop("core2b_proc", None)
                raise
        from vllm.v1.engine.core import EngineCoreProc
        mod.install(EngineCoreProc)
