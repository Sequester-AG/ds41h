"""core2b_proc.py -- stage 2b-α: vLLM's engine process with the busy loop and the socket IO in C++.

PHYS_CXX_LOOP=1 (with PHYS_CXX_CORE=1, PHYS_CXX_SCHED=1, PHYS_CXX_KV=1): core_sched.install() calls install() below,
which replaces three EngineCoreProc methods in the engine process (vLLM's launcher still spawns it, Python still runs
its init: config, the handshake (HELLO / init metadata / READY), the device engine, the ready response):
  process_input_sockets   -> fpr_core2b.Server.start(): the C++ DEALERs (ready payload sent on each), PUSHes, T_in /
                             T_out; sets ready_event (vLLM's __init__ waits for it)
  process_output_sockets  -> nothing (T_out owns the output sockets)
  run_busy_loop           -> fpr_core2b.Core2b on the scheduler's C++ rank scheduler / KV manager (capsules), attached
                             to the server; the C++ loop thread runs; Python's main thread only sleeps (signals)
The device worker (2b-α0, the interim): _Worker below. The C++ loop posts its calls (prebuild at schedule time, dispatch,
wait) to this process's main thread, which built the engine and serves them in run_busy_loop (core.serve_python); it
rebuilds the PhysStep (serve's StepPlan / StepBatches) and calls the executor's worker, as PhysExecutor does. NativeDecode (C++)
replaces it for decode steps (2b-α), the stage-1 driver C API for mixed ones.
Images: _MM below. T_in hands an ADD carrying mm_features to _MM.admit on this thread (vLLM's
decoder incl. aux frames, PhysScheduler's checks + content keys -> MMPlanner) and adds the request with its keys; the
planner + encoder caches run in Python on embedded steps (native) or every step (_Worker); frees are posted.
Not covered: FORCE (2a has it; 2b: test streams only).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

_ONCTX_CHECK = os.environ.get("PHYS_CXX_ONCTX_LAZY_CHECK", "0") == "1"   # on_ctx lazy's check mode

_HERE = os.path.dirname(os.path.abspath(__file__))
PRECREATE_HOLD = os.environ.get("ENG_NATIVE_BOUNDARY", "0") == "1"   # see post())
ND_LONG = os.environ.get("ND_LONG", "0") == "1"   # long-context decode variants (post())
_BUILD = (os.environ.get("PHYS_CXX_CORE2B_DIR") or os.environ.get("PHYS_CXX_CORE_DIR") or os.environ.get("PHYS_CXX_BUILD")
          or os.path.join(_HERE, "..", "..", "build_py"))
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
import fpr_core2b as X  # noqa: E402

import msgspec  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.v1.engine import EngineCoreRequest  # noqa: E402
from vllm.v1.serial_utils import MsgpackDecoder  # noqa: E402

logger = init_logger("vllm.phys_adapter")


def enabled() -> bool:
    return os.environ.get("PHYS_CXX_LOOP", "0") == "1"


# ---------------------------------------------------------------------------------------------- shared conversions
_PV: dict = {}             # one Python view per C++ page list / window table (cxx_kv's identity rule), by address
_WV: dict = {}
_NACC = [0]


def _view(cache: dict, ptr: int, v):
    got = cache.get(ptr)
    if got is None:
        got = cache[ptr] = v
    _NACC[0] += 1
    if _NACC[0] % 4096 == 0:              # a cached view keeps its C++ object alive: drop the ones only we hold
        for d in (_PV, _WV):
            for k in [k for k, x in d.items() if x.use_count() <= 1]:
                del d[k]
    return got


def _assemble(plan, tuples):
    """fpr_core2b's StepBatch tuples (one per rank, = fpr_kvtables.build_step_batch's) -> serve's StepBatch objects,
    exactly as cxx_kv.wrap_build_step_batch assembles them (the plan's work items give the Python-side fields)."""
    from interfaces import SeqEntry, StepBatch

    out = []
    for rank, t in enumerate(tuples):
        works = [w for w in plan.work if w.rank == rank]
        n_rows, pos, seq_slot, paged, win, tail, samp, seqs = t
        sb = StepBatch(rank, plan.step, n_rows)
        sb.positions, sb.seq_slot, sb.paged_slot, sb.window_slot, sb.tail_rows, sb.sample_rows = \
            pos, seq_slot, paged, win, tail, samp
        for i, row0, n_blocks, tail_row0, wver, inc, slot, pages, window, bp, wp in seqs:
            pages = _view(_PV, bp, pages)
            window = _view(_WV, wp, window)
            w = works[i]
            helper = w.kind == "cp"
            cp = getattr(w, "cp", None)
            se = SeqEntry(cp.rid if helper else w.rid, "prefill" if helper else w.kind, row0, w.n, w.start, w.start,
                          0 if helper else w.cached_upto, w.replay_start, False if helper else w.last_chunk, pages, window,
                          None if slot < 0 else slot)
            se.cp = cp
            se.n_blocks = n_blocks
            se.known = int(getattr(w, "known", -1))
            se.images = tuple(getattr(w, "images", ()))
            se.score = bool(getattr(w, "score", False))
            se.forced = bool(getattr(w, "forced", False))
            se.uid = int(getattr(w, "uid", -1))
            se.wver, se.inc = wver, inc
            se.tail_row0 = tail_row0
            sb.seqs.append(se)
        sb.copies = [c for c in plan.copies if rank in (c.src_rank, c.dst_rank)]
        out.append(sb)
    return out


def _cts(batches, dicts, k1):
    """fpr_core2b's CompactTables dicts -> rowexp_ref.CompactTables (cxx_kv.CxxTableBuilder.build_compact's assembly)."""
    from rowexp_ref import CompactTables

    out = []
    for sb, d in zip(batches, dicts):
        ct = CompactTables(sb.rank, sb.step, k1)
        for f, v in d.items():
            setattr(ct, f, v)
        ct.rids = [se.rid for se in sb.seqs]
        out.append(ct)
    return out


def _results(plan, results) -> list:
    """serve's StepResult per rank -> the dicts fpr_core2b reads (logprob rows: [(position, top-k)] -> row j = token j)."""
    base = {}
    if plan is not None:
        for w in plan.work:
            base[w.rid] = w.start if w.kind == "decode" else w.start + w.n - 1
    out = []
    for res in results or ():
        lp = []
        for rid, rows in (res.logprobs or {}).items():
            bypos = {int(p): lst for p, lst in rows}
            b = base.get(rid, 0)
            n = len(res.tokens.get(rid, ()))
            lp.append((rid, [[(int(i), float(v)) for i, v in bypos.get(b + j, [])] for j in range(n)]))
        out.append(dict(rank=int(res.rank), step=int(res.step),
                        accepted=[(r, int(n)) for r, n in res.accepted.items()],
                        tokens=[(r, [int(t) for t in v]) for r, v in res.tokens.items()],
                        logprobs=lp,
                        drafts=[(r, [int(t) for t in v]) for r, v in (getattr(res, "drafts", None) or {}).items()]))
    return out


def _check_batches(step, py_batches, cxx_batches, py_cts, cxx_cts) -> None:
    """PHYS_CXX_CHECK_EMB=1: the embedded step's inputs built from C++ (2b's tuples / views, NativeWorker's TableBuilder)
    must equal the Python worker's own, field by field (raises on the first difference)."""
    def cp_t(cp):
        return None if cp is None else (cp.group, str(cp.rid), cp.owner, [tuple(x) for x in cp.subs], cp.me)
    assert len(py_batches) == len(cxx_batches), (step, len(py_batches), len(cxx_batches))
    for a, b in zip(py_batches, cxx_batches):
        for f in ("rank", "step", "n_rows", "positions", "seq_slot", "paged_slot", "window_slot", "tail_rows", "sample_rows"):
            x, y = getattr(a, f), getattr(b, f)
            assert list(x) == list(y) if isinstance(x, (list, tuple)) or hasattr(x, "__len__") else x == y, (step, a.rank, f)
        assert [(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page) for c in a.copies] == \
            [(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page) for c in b.copies], (step, a.rank, "copies")
        assert len(a.seqs) == len(b.seqs), (step, a.rank, "seqs")
        for u, v in zip(a.seqs, b.seqs):
            for f in ("rid", "kind", "row0", "n_rows", "start", "num_computed", "cached_upto", "replay_start", "last_chunk",
                      "state_slot", "n_blocks", "known", "images", "score", "forced", "uid", "wver", "inc", "tail_row0"):
                assert getattr(u, f, None) == getattr(v, f, None), (step, a.rank, u.rid, f, getattr(u, f, None), getattr(v, f, None))
            assert list(u.block_table) == list(v.block_table), (step, a.rank, u.rid, "block_table")
            assert list(u.window_table.items()) == list(v.window_table.items()), (step, a.rank, u.rid, "window_table")
            assert cp_t(getattr(u, "cp", None)) == cp_t(getattr(v, "cp", None)), (step, a.rank, u.rid, "cp")
    for a, b in zip(py_cts, cxx_cts):
        for f, y in b.items():
            x = getattr(a, f)
            assert [int(t) for t in x] == [int(t) for t in y], (step, a.rank, "ct", f)


def _rank_tables(d, sb):
    """fpr_core2b.rank_tables_dict (nd's full tables of a resolved batch, = fpr_kvtables TableBuilder.build's dict) ->
    worker.RankTables, exactly as cxx_kv.TableBuilder.build wraps that dict (sample rows / seq rows from the batch)"""
    import torch
    from worker import RankTables

    T = torch.from_numpy
    seq = {k: T(v) for k, v in d["seq"].items()}
    ti = {int(s_): T(v) for s_, v in d["ti"].items()}
    tl = {int(s_): T(v) for s_, v in d["tl"].items()}
    return RankTables(d["n"], T(d["positions"]), T(d["tokens"]), T(d["own_slot"]), T(d["si"]), T(d["sl"]), ti, tl,
                      list(sb.sample_rows), [(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs],
                      seq_slot=T(d["seq_slot"]), paged_slot=T(d["paged_slot"]), tail_rows=T(d["tail_rows"]),
                      sample_idx=T(d["sample_idx"]), seq=seq, targets=T(d["targets"]))


def _mixed_snap(eng) -> dict:
    """the engine state fpr::p0::build_mixed reads (p0::MixedSnap; = core/tests/record_mixed_live.py snap_of: the
    same keys, read from the live engine once at init; the flags are import-time constants)."""
    E = sys.modules[type(eng).__module__]
    cfg = eng.cfg
    hs = eng.hooks.hsd
    return dict(W=E.W, mixed_round=cfg.mixed_round, max_mixed_rows=cfg.max_mixed_rows, max_tail_rows=eng.max_tail_rows,
                mixed_t_round=E.MIXED_T_ROUND, layers=list(eng.layers), mixed_graphs=bool(cfg.mixed_graphs), cp=bool(cfg.cp),
                dbg_sync=bool(E.DBG_SYNC), l20_tail=bool(E.L20_TAIL), mixed_prestep=bool(E.MIXED_PRESTEP),
                mixed_graphs_t=bool(E.MIXED_GRAPHS_T), pf_act8=bool(E.PF_ACT8), attn_pf=bool(E.ATTN_PF), mm_on=bool(E.MM_ON),
                image_route=list(E.IMAGE_ROUTE_IDS) if hasattr(E, "IMAGE_ROUTE_IDS") else None,
                poison=bool(E.POISON), cxx_pieces=bool(E.CXX_PIECES), mixed_cxx_head=bool(E.MIXED_CXX_HEAD),
                tail_bound=bool(E.TAIL_BOUND), devtok_v2=bool(E.DEVTOK_V2), per_layer_plans=bool(E.BISECT["per_layer_plans"]),
                clear_tail_padding=bool(E.BISECT["clear_tail_padding"]), mixed_plans=str(cfg.mixed_plans),
                upload_v2=bool(cfg.upload_v2), index_layers=list(E.INDEX_LAYERS), ratio=[int(E.RATIO[L]) for L in eng.layers],
                index_source=[-1 if E.index_source_of(L) is None else int(E.index_source_of(L)) for L in eng.layers],
                sliding_window=int(E.SLIDING_WINDOW), window_page_tokens=int(E.WINDOW_PAGE_TOKENS),
                scratch=[int(p.scratch_window_page * E.WINDOW_PAGE_TOKENS) for p in eng.pools],
                n_paged=[int(p.n_paged) for p in eng.pools], pf_ws=getattr(eng, "pf_ws", None) is not None,
                hsd=hs is not None, idx=eng.hooks.idx is not None,
                k1=int(hs.next_ids[0].shape[1]) if hs is not None else 4,
                span_flags=_span_flags(eng, E), driver_so=_driver_so(eng))


def _span_flags(eng, E) -> dict:
    """mixed_cxx._issue_span's engine switches (the span program refuses where it would return False)."""
    try:
        import agrs_engine as AGE
        min_r = int(getattr(AGE, "MIN_R", 0))
    except ImportError:
        min_r = 0
    return dict(l2pf=bool(getattr(E, "C1_L2PF", False)), nancheck=bool(getattr(E, "NANCHECK", False)),
                prof_pieces=bool(getattr(E, "PROF_PIECES", False)), tp_on=False, mega_on=False,
                router_pack3=bool(getattr(eng.cfg, "router_pack3", True)),
                agrs_on=bool(getattr(type(eng), "_agrs_installed", False)), agrs_ag32=bool(getattr(E, "AGRS_AG32", False)),
                agrs_agce=bool(getattr(E, "AGRS_AGCE", False)), agrs_min_r=min_r)


def _driver_so(eng) -> str:
    """the loaded C++ mixed driver (mixed_cxx.ext().__file__), "" if not loaded yet."""
    m = sys.modules.get("mixed_cxx")
    try:
        return str(m.ext().__file__) if m is not None else ""
    except Exception:  # noqa: BLE001
        return ""


class _Embedded:
    """2b + nd (PHYS_CXX_NATIVE=nd): the embedded engine step (design §5 / nd_api.h). The C++ NativeWorker routes every
    step nd's async_ok refuses here, between nd.embedded_begin (drain + push_state) and nd.embedded_end (pull_state):
    serve Worker.step's compact path with the NativeWorker's own CompactTables (its TableBuilder is the one tracking the
    device tables) and serve StepBatches over the C++ batches' page lists; images: _MM.step -> StepContext.mm.
    post() (after embedded_end, a step boundary) = nd_wrap's per-dispatch exporter calls: ensure_bucket for buckets the
    Python step created, refresh_graphs for graphs it captured."""

    def __init__(self, sched, w, eng, ex, mm, k1: int):
        self.s, self.w, self.eng, self.ex, self.mm, self.k1 = sched, w, eng, ex, mm, k1
        self.dec = MsgpackDecoder(EngineCoreRequest)
        self.stats = dict(steps=0, admissions=0, buckets=0, graphs=0, mixed_native=0, agce_pulls=0)
        self.core = None                         # the Core2b, set after construction: AGCE's shared previous bucket
        self._mx = None                          # (Engine._mixed_body's context, rb) between body and post

    # AGCE: mixed_cxx._issue_span's layer-0 ce = (E._agrs_st_last is this bucket's states). A native step
    # issues AGs Python never sees, and a Python step issues AGs core2b never sees: ONE shared "previous AG bucket". Before a
    # Python step: take core2b's last native segment (if any since) as E._agrs_st_last (looked up read-only: no bucket() side
    # effects). After it: hand Python's back. Native steps pay nothing (they only read / write core2b's copy).
    def _agce_pull(self) -> None:
        c, eng = self.core, self.eng
        if c is None or eng is None or not hasattr(c, "take_last_seg"):
            return
        seg = c.take_last_seg()
        if not seg:
            return
        role, R = seg.split("|")
        key = (int(R), "prod", role)
        st = None
        for d in (getattr(eng, "bk_view", None) or {}, getattr(eng, "bk", None) or {}):
            for v in d.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("bkey") == key:
                    st = v
                    break
            if st is not None:
                break
        eng._agrs_st_last = st                   # None: the bucket is gone -> Python's layer 0 is non-CE (as native's would be)
        self.stats["agce_pulls"] += 1

    def _agce_push(self) -> None:
        c, eng = self.core, self.eng
        if c is None or eng is None or not hasattr(c, "set_last_seg"):
            return
        st = getattr(eng, "_agrs_st_last", None)
        bk = st[0].get("bkey") if st else None
        c.set_last_seg(f"{bk[2]}|{int(bk[0])}" if isinstance(bk, tuple) and len(bk) > 2 and bk[1] == "prod" else "")

    def step(self, step, plan_t, btuples, ctds, host, mmv):
        self._agce_pull()
        try:
            return self._step(step, plan_t, btuples, ctds, host, mmv)
        finally:
            self._agce_push()

    def _step(self, step, plan_t, btuples, ctds, host, mmv):
        w = self.w
        plan = self.s._plan(plan_t) if plan_t is not None else None
        batches = _assemble(plan, btuples) if plan is not None else []
        cts = _cts(batches, ctds, self.k1)
        ctx = self._ctx(host, batches, mmv)
        w.copies(batches)
        res = w.backend.step_compact(batches, cts, ctx=ctx)
        res = res.result() if hasattr(res, "result") else res   # synchronous in 2b-α (pull_state drains D13 anyway)
        self.stats["steps"] += 1
        return _results(plan, res)

    def _ctx(self, host, batches, mmv, note=True):
        """the step's StepContext from the C++ host dict (fresh sequences with their ADD frames' SamplingParams and nd's
        seed, drops, grammar rows, prev tokens, order, images)"""
        from adapter.bridge import GrammarRows
        from interfaces import SeqInit, StepContext, sampling_dict

        w = self.w
        new = []
        for rid, serial, prompt, output, seed, structured, frames in host["fresh"]:
            sp = sampling_dict(self.dec.decode(list(frames)).sampling_params) if frames else {}
            if sp.get("seed") is None and seed is not None:   # nd's NewSeq got this seed (NativeWorker draw_seeds)
                sp["seed"] = int(seed)
            new.append(SeqInit(rid, list(prompt), list(output), sp, bool(structured)))
            if note:                             # not for on_ctx lazy's check-mode shadow
                if self.ex is not None:
                    self.ex.note_rid(rid)
                self.stats["admissions"] += 1
        grammar = {rid: GrammarRows(np.frombuffer(rows, dtype=np.int32).reshape(n, wd).copy(), int(acc))
                   for rid, rows, n, wd, acc in host["grammar"]}
        mm_dev = self.mm.step(mmv) if (mmv is not None and self.mm is not None) else None
        return StepContext(new=new, dropped=set(host["dropped"]), grammar=grammar,
                           prev_tokens={r: list(t) for r, t in host["prev"]}, order=list(host["order"]),
                           mm=w._mm_ctx(mm_dev, batches))

    def mixed_body(self, step, plan_t, btuples, ctds, host, rbt, tds):
        """the Python middle of a NATIVE mixed step. nd's begin already ran
        D13._fallback's front natively (drain + push_state, D12 deltas, rowexp resolve, _fb_rows, the full tables, the
        serial bump). This runs what _fallback's eng.step(rb, tables, ctx=ctx) runs up to the head, in its order:
        Worker.copies, the StepContext, Engine.step's prologue (bump_serial, hooks.on_ctx, _mm), then
        Engine._mixed_body(rb, tables, prep_head=False); then the step's mixed views. Returns (R_A, R_T, bucket T's key text) for nd's fill_mixed."""
        import dataclasses

        import nd_export

        self._agce_pull()                        # AGCE: the body issues this step's AGs in Python
        w, eng = self.w, self.eng
        plan = self.s._plan(plan_t)
        batches = _assemble(plan, btuples)
        ctx = self._ctx(host, batches, None)     # (image steps never route here: the NativeWorker runs them embedded)
        w.copies(batches)
        rb = []                                  # D13._fb_rows as nd resolved it (the same replace rule)
        for sb, (pos, paged, win, seqv) in zip(batches, rbt):
            seqs = [se if se.start >= 0 else dataclasses.replace(se, start=int(s0), num_computed=int(nc), n_blocks=int(nb))
                    for se, (s0, nc, nb) in zip(sb.seqs, seqv)]
            rb.append(dataclasses.replace(sb, seqs=seqs, positions=list(pos), paged_slot=list(paged), window_slot=list(win)))
        tables = [_rank_tables(d, sb) for d, sb in zip(tds, rb)]
        eng.bump_serial(rb)                      # Engine.step's prologue (the push carried the pre-bump serial)
        eng.hooks.on_ctx(ctx, None, None)
        eng._mm = getattr(ctx, "mm", None)
        c = eng._mixed_body(rb, tables, prep_head=False)
        if c["out"] is not None:
            raise RuntimeError("beta0a: a mixed step without a tail segment (layer subset engines are not native)")
        RA, RT = int(c["stA"][0]["R"]), int(c["stT"][0]["R"])
        kA, kT = nd_export._key_text((RA, "prod", "A")), nd_export._key_text((RT, "prod", "T"))
        if self.ex.ensure_mixed([(RA, 0, kA), (RT, 1, kT)]) != 0:   # both exist: a pure view fill, never a refusal
            raise RuntimeError(f"beta0a: ensure_mixed refused {kA} / {kT} after the body")
        self._mx = (c, rb)
        return RA, RT, kT

    def ensure_mixed_views(self, RA, RT, stats=None):
        """Returns (ok, why)
        stats: the Engine.stats counters the native steps since the previous call moved
        (checklist D5; step_mixed's own increments, which a native step does not run), added first."""
        import nd_export

        # P1: the native hits' touch log first (in order), then the optional check flag
        touches = sorted((k, v) for k, v in (stats or {}).items() if k.startswith("__touch_"))
        check = bool((stats or {}).get("__check"))
        if stats:
            stats = {k: v for k, v in stats.items() if not k.startswith("__")}
        if touches:
            self._replay_touches([(int(v) >> 24, int(v) & 0xFFFFFF) for _, v in touches])
        if stats:
            es = self.eng.stats
            for k, v in stats.items():
                es[k] = es.get(k, 0) + int(v)
        if int(RA) < 0:   # replay-only (before a Python body): no exporter ensure
            self.stats["p1_replay_only"] = self.stats.get("p1_replay_only", 0) + 1
            return True, ""
        if check:
            self._replay_touches([(int(RA), int(RT))])   # this step's own touch (the native path would have logged it)
            h0 = self._bk_hash()

        keys = [(int(RA), 0, nd_export._key_text((int(RA), "prod", "A")))]
        if int(RT) > 0:
            keys.append((int(RT), 1, nd_export._key_text((int(RT), "prod", "T"))))
        rc = self.ex.ensure_mixed(keys)
        if check and rc == 0:
            h1 = self._bk_hash()
            n_ = self.__dict__.get("_p1_chk", [0, 0])
            n_[0 if h0 == h1 else 1] += 1
            self._p1_chk = n_
            if h0 != h1 and n_[1] <= 20:
                print(f"core2b ENS check: bucket state moved under a replayed hit (RA {RA}, RT {RT}): {h0[1]} -> {h1[1]}",
                      file=sys.stderr, flush=True)
            if sum(n_) % 200 == 1:
                print(f"core2b ENS check: bucket-state hash equal {n_[0]}, differ {n_[1]}", file=sys.stderr, flush=True)
        return rc == 0, "" if rc == 0 else f"ensure_mixed rc {rc} ({', '.join(k[2] for k in keys)})"

    def _replay_touches(self, steps):
        """P1: exactly exporter.ensure_mixed's state effects of a pure hit, per native-hit
        step in order: eng._busy = set(keys); eng.bucket() per key (LRU re-insert / onebk view path, a pure hit by construction:
        buckets_epoch did not move); the exporter's ensure_mixed counter. A replayed bucket() that created / evicted is logged
        (must never happen: the native hit required an unchanged buckets_epoch)."""
        eng, ex = self.eng, self.ex
        for RA, RT in steps:
            want = [(int(RA), "prod", "A")] + ([(int(RT), "prod", "T")] if int(RT) > 0 else [])
            k0 = set(dict.keys(eng.bk)); v0 = len(getattr(eng, "bk_view", None) or {})
            eng._busy = set(want)
            for R, moe, ro in want:
                eng.bucket(R, moe, ro)
            if set(dict.keys(eng.bk)) != k0 or len(getattr(eng, "bk_view", None) or {}) != v0:
                print(f"core2b P1 replay: bucket() CREATED / EVICTED while replaying a native hit (RA {RA}, RT {RT})",
                      file=sys.stderr, flush=True)
            st = getattr(ex, "stats", None)
            if isinstance(st, dict):
                st["ensure_mixed"] = st.get("ensure_mixed", 0) + 1
        fl = getattr(ex, "_flush", None)
        if fl is not None:
            fl()
        self.stats["p1_replayed"] = self.stats.get("p1_replayed", 0) + len(steps)

    def _bk_hash(self):
        """(order of eng.bk, order of bk_view, _busy, bk_bytes) -> (sha, short text)"""
        eng = self.eng
        t = repr((list(dict.keys(eng.bk)), list(dict.keys(getattr(eng, "bk_view", None) or {})), sorted(map(repr, eng._busy)),
                  sorted(map(repr, getattr(eng, "bk_bytes", {}).items()))))
        return hashlib.sha256(t.encode()).hexdigest()[:16], f"{len(eng.bk)} bk / {len(getattr(eng, 'bk_view', None) or {})} views / busy {len(eng._busy)}"

    def h1_guard_ok(self, key):
        """Returns (ok, why)
        no guard in this build -> (False, ...): the step runs embedded, never an unchecked native replay."""
        EM = sys.modules[type(self.eng).__module__]
        mxh = EM._mxh() if getattr(EM, "MIXED_CXX_HEAD", False) else None
        if mxh is None or not hasattr(mxh, "guard_ok"):
            return False, "mixed_cxx_head.guard_ok unavailable"
        ok = bool(mxh.guard_ok(self.eng.hooks, str(key)))   # STALE: the guard retires the key itself (as head_eager)
        return ok, "" if ok else f"guard: {key} stale"

    def on_ctx(self, step, plan_t, btuples, host):
        """a NATIVE mixed step's admissions / drops for Python's per-request state. Runs
        once, in phase 2's begin (after nd's push), only when host has fresh / dropped requests: the StepContext (SeqInit +
        nd's seed, drops, prev tokens, order; admissions counted, rids noted) -> eng.hooks.on_ctx (eng.hist, samp_ctx,
        prev_seq, structured). No bump_serial (nd's push carries the serial), no Worker.copies (native), no _mm (no images:
        the NativeWorker runs image steps embedded). A refused step's mixed_body runs on_ctx instead (exactly once)."""
        plan = self.s._plan(plan_t)
        batches = _assemble(plan, btuples)
        if host.get("lazy"):             # on_ctx lazy: core2b calls on_ctx then ONLY for the check mode: the eager
            if not _ONCTX_CHECK:            # on_ctx into the SHADOW (the real state comes from replay)
                raise RuntimeError("on_ctx lazy: on_ctx called on a lazy step without PHYS_CXX_ONCTX_LAZY_CHECK")
            ctx = self._ctx(host, batches, None, note=False)   # compared at the replay
            type(self.eng.hooks).on_ctx(self._shadow(), ctx, None, None)
            self.stats["onctx_shadow"] = self.stats.get("onctx_shadow", 0) + 1
            return
        ctx = self._ctx(host, batches, None)
        self.eng.hooks.on_ctx(ctx, None, None)
        self.stats["mixed_onctx"] = self.stats.get("mixed_onctx", 0) + 1

    def _shadow(self):
        sh = getattr(self, "_shadow_", None)
        if sh is None:
            from types import SimpleNamespace as NS
            sh = self._shadow_ = NS(eng=NS(hist={}, prev_seq={}), samp_ctx={}, structured=set(), sampling=None, lookback=None,
                                    grammar={}, prev_tokens={})
        return sh

    def replay(self, ents):
        """on_ctx lazy (PHYS_CXX_ONCTX_LAZY): the admissions NATIVE steps queued since the last sync, replayed into
        Python's per-request state in one call, right before the next push into Python (nd's J then appends the pending
        tokens; drops ride J). ents = [(rid, serial, ids, seed or None, structured, ADD frames)]: ids = nd's Req.hist minus
        hist_pending (eng.hist as it must be before the push). The SamplingParams are decoded from the frames exactly as _ctx
        does (float64 temperature; the request's own seed, else the worker's drawn one), so the state equals eager on_ctx's.
        Check mode (PHYS_CXX_ONCTX_LAZY_CHECK): each replayed rid is compared with the shadow the eager on_ctx built."""
        from interfaces import sampling_dict
        H = self.eng.hooks
        fresh = []
        for rid, serial, ids, seed, structured, frames in ents:
            sp = sampling_dict(self.dec.decode(list(frames)).sampling_params) if frames else {}
            if sp.get("seed") is None and seed is not None:   # nd's NewSeq got this seed (NativeWorker draw_seeds)
                sp["seed"] = int(seed)
            if sp.get("seed") is None:
                raise RuntimeError(f"on_ctx lazy: {rid!r} has no seed (draw_seeds off?): eager on_ctx would draw one in Python")
            fresh.append((rid, ids, float(sp.get("temperature", 1.0)), int(sp["seed"]), bool(structured)))
            if self.ex is not None:
                self.ex.note_rid(rid)
            self.stats["admissions"] += 1
        H.replay(fresh, ())
        self.stats["lazy_replays"] = self.stats.get("lazy_replays", 0) + 1
        self.stats["lazy_replayed"] = self.stats.get("lazy_replayed", 0) + len(fresh)
        if _ONCTX_CHECK:
            sh = self._shadow()
            rids = [f[0] for f in fresh]
            a = H.per_request_state(rids)
            b = type(H).per_request_state(sh, rids)
            bad = [r for r in rids if a[r][:5] != b[r][:5]]   # prev_seq excluded: push_state rebuilds it from nd right after
            self.stats["onctx_check_rids"] = self.stats.get("onctx_check_rids", 0) + len(rids)
            if bad:
                n_ = self.stats["onctx_check_bad"] = self.stats.get("onctx_check_bad", 0) + len(bad)
                logger.warning("core2b on_ctx lazy CHECK: %d / %d replayed rids differ from eager on_ctx (e.g. %r: lazy %s / "
                               "eager %s) [%d so far]", len(bad), len(rids), bad[0], str(a[bad[0]][1:5]), str(b[bad[0]][1:5]), n_)
            for r in rids:                 # the shadow keeps only not-yet-replayed admissions
                sh.eng.hist.pop(r, None)
                sh.samp_ctx.pop(r, None)
                sh.structured.discard(r)
        n_ = self.stats["lazy_replays"]
        if n_ == 1 or n_ % 64 == 0:
            logger.info("core2b on_ctx lazy: replays %d, replayed %d, check %s: rids %d bad %d, shadow on_ctx %d", n_,
                        self.stats["lazy_replayed"], "on" if _ONCTX_CHECK else "off", self.stats.get("onctx_check_rids", 0),
                        self.stats.get("onctx_check_bad", 0), self.stats.get("onctx_shadow", 0))

    def mixed_post(self):
        """Engine._mixed_post(head_eager=False) after nd's native head launch: end events, the device-wide sync
        (every stream the body used), head-key capture, bookkeeping; nd's finish then computes the head results."""
        c, rb = self._mx
        self._mx = None
        self.eng._mixed_post(c, rb, head_eager=False)
        self._agce_push()                        # AGCE: hand the body's last AG bucket back to core2b
        self.stats["mixed_native"] += 1

    def _post_sig(self):
        """"""
        ex, eng = self.ex, self.eng
        H = eng.hooks
        pend = sum(1 for k in list(eng.bk) if not isinstance(k, tuple) and k < (1 << 20) and k not in ex._state_of)
        st = getattr(ex, "stats", {}) or {}
        req = (repr(sorted((str(k), repr(v)) for k, v in getattr(H, "samp_ctx", {}).items())),
               repr(sorted(str(k) for k in getattr(H, "structured", ()))),
               repr(sorted((str(k), repr(v)) for k, v in eng.prev_seq.items())),
               repr(sorted((str(k), len(v)) for k, v in eng.hist.items())))
        return dict(pending_buckets=pend, pub=len(ex._pub), graph_count=int(ex.x.graph_count()),
                    h1_retired=int(st.get("h1_guard_retired", 0)), req=hash(req))

    def post_probe(self):
        """H2 (PHYS_CXX_ONCTX_CHECK=1): a state-only step's publication, run anyway: (number of changed items, names).
        The publication itself is the normal post(), so the engine state ends exactly as without the skip."""
        if self.ex is None:
            return 0, ""
        a = self._post_sig()
        b0, g0 = self.stats["buckets"], self.stats["graphs"]
        ex = self.ex
        x, calls = ex.x, []

        class _Rec:
            def __getattr__(_s, n):
                f = getattr(x, n)
                if not callable(f):
                    return f

                def w(*aa, **kk):
                    calls.append(n)
                    return f(*aa, **kk)
                return w
        ex.x = _Rec()
        try:
            self.post()
        finally:
            ex.x = x
        b = self._post_sig()
        ch = [k for k in a if a[k] != b[k]]
        ch += sorted({"x." + n for n in calls if n not in ("graph_count", "mixed_keys", "mixed_view", "ptr")})
        if self.stats["buckets"] != b0:
            ch.append("buckets_ensured")
        if self.stats["graphs"] != g0:
            ch.append("graphs_net")
        self.stats["h2_probes"] = self.stats.get("h2_probes", 0) + 1
        if ch:
            self.stats["h2_changed"] = self.stats.get("h2_changed", 0) + 1
        return len(ch), "".join(" " + c for c in ch)

    def post(self):
        ex, eng = self.ex, self.eng
        if ex is None:                           # embed_all: no nd, no exporter
            return
        # ENG_NATIVE_BOUNDARY: the precreate
        # hold: ENG_DECODE_PRECREATE decides in Python's bucket_of, which native decode never calls -> while the engine's warm-up
        # is due (precreate deciding / an H1 precapture due) the decode buckets stay unpublished: nd refuses them cleanly
        # ("bucket not ensured") and they run embedded, so precreate completes exactly as in Python mode (its done point runs
        # MIX_PRECREATE / C1 / H1 precaptures). This init round (after nd.init) is the first post(): no decode bucket is
        # published before the hold lifts.
        if PRECREATE_HOLD and getattr(eng, "_precap_extra", None) is None and hasattr(ex, "precapture_all"):
            eng._precap_extra = ex.precapture_all   # ENG_PRECAP_DEFER's job runs nd's precapture inside its window
        nb = getattr(eng, "native_boundary", None) if PRECREATE_HOLD else None
        wd = getattr(eng, "warmup_due", None) if PRECREATE_HOLD else None
        if (nb is None) != (wd is None):         # post() raising = ENGINE_CORE_DEAD) a half engine: no hold
            if not getattr(self, "_nb_half_logged", False):
                self._nb_half_logged = True
                logger.warning("phys adapter: ENG_NATIVE_BOUNDARY: the engine has %s but not %s: no precreate hold",
                               "native_boundary" if nb is not None else "warmup_due",
                               "warmup_due" if nb is not None else "native_boundary")
            nb = wd = None
        if nb is not None:
            nb()                                 # np12: a due H1 precapture / the stalled-precreate decision at this boundary
        held = wd is not None and bool(wd())
        if held:
            self.stats["precreate_hold"] = self.stats.get("precreate_hold", 0) + 1
        else:
            for k in list(eng.bk):     # (rule (a)(i): a step boundary; nd's own steps never create buckets)
                # ND_LONG: also the long
                # variants Rk = R + lvl * 2^20 (lvl 1 / 2), keys / views at Rk, sizes at R; without it only plain decode buckets (k < 2^20)
                if not isinstance(k, tuple) and (k < (1 << 20) or (ND_LONG and 1 <= (k >> 20) <= 2)) and k not in ex._state_of:
                    ex.ensure_bucket(int(k))
                    self.stats["buckets"] += 1
        self.stats["graphs"] += int(ex.refresh_graphs() or 0)
        if not held:
            self._native_done_transition(ex, eng)
        self._late_create_check(eng, held)

    def _native_done_transition(self, ex, eng):
        """once, at the first unheld post after precreate decided (eng._pc None): with the engine's
        ENG_PRECAP_DEFER the precapture job (incl. nd's precapture_all, registered as eng._precap_extra) already ran inside the
        memory-gated window; without it, nd's precapture_all runs here (fallback). Then nd's mark_native_done: later publishes
        count as LATE (cross-check of the late-create guard). hasattr-guarded (nd < 343: no-op)."""
        if getattr(self, "_nd_marked", False) or getattr(eng, "_pc", None) is not None:
            return
        self._nd_marked = True
        EM = sys.modules.get(type(eng).__module__)
        defer = bool(getattr(EM, "PRECAP_DEFER", False))
        if not defer and hasattr(ex, "precapture_all"):
            try:
                ex.precapture_all()
            except Exception as e:  # noqa: BLE001  (its keys stay first-use; logged)
                logger.warning("phys adapter: precapture_all failed at the native-done transition: %s: %s", type(e).__name__, str(e)[:200])
            # publish what it created / captured NOW, before mark_native_done: later native steps may not post
            # (py-free / H2 state-only), and a publish after the mark would count as a false LATE
            for k in list(eng.bk):
                if not isinstance(k, tuple) and (k < (1 << 20) or (ND_LONG and 1 <= (k >> 20) <= 2)) and k not in ex._state_of:
                    ex.ensure_bucket(int(k))
                    self.stats["buckets"] += 1
            self.stats["graphs"] += int(ex.refresh_graphs() or 0)
        if hasattr(ex, "mark_native_done"):
            ex.mark_native_done()
        logger.info("phys adapter: native-done transition (step serial %s): precapture %s, nd late-publish counters armed",
                    getattr(eng, "step_serial", "?"), "inside ENG_PRECAP_DEFER's job" if defer else
                    ("precapture_all (fallback)" if hasattr(ex, "precapture_all") else "none (nd < 343)"))

    def _late_create_check(self, eng, held):
        """after warm-up (precreate
        decided: eng._pc is None, no hold), every bucket key created (eng.bk / eng.bk_view) or evicted since the previous
        boundary is a serve-time Python creation: stats late_create / late_evict count them and one warning line names them.
        The first boundary after warm-up only takes the snapshot (and logs 'late-create guard armed'). Observation only: it
        never changes what runs. Gates assert late_create == late_evict == 0 after the armed line."""
        if held or getattr(eng, "_pc", None) is not None:
            self._lc_snap = None                 # (warm-up: creations are expected)
            return
        keys = set(eng.bk) | set(getattr(eng, "bk_view", None) or ())
        snap = getattr(self, "_lc_snap", None)
        self._lc_snap = keys
        if snap is None:
            if not getattr(self, "_lc_armed", False):
                self._lc_armed = True
                self.stats.setdefault("late_create", 0)
                self.stats.setdefault("late_evict", 0)
                nv = len(getattr(eng, "bk_view", None) or ())
                try:   # the onebk universe for the config in use: (rnd(max_mixed_rows) / round - 1) + (rnd(max_tail_rows) / round - 1)
                    c_ = eng.cfg
                    rnd = lambda n: max(c_.mixed_round, -(-int(n) // c_.mixed_round) * c_.mixed_round)   # noqa: E731
                    ne = rnd(c_.max_mixed_rows) // c_.mixed_round - 1 + rnd(eng.max_tail_rows) // c_.mixed_round - 1
                except Exception:  # noqa: BLE001
                    ne = -1
                self.stats["onebk_views_armed"], self.stats["onebk_views_expected"] = nv, ne
                logger.info("phys adapter: late-create guard armed: %d bucket / view keys after warm-up (%d onebk views; expected %d)",
                            len(keys), nv, ne)
            return
        new, gone = keys - snap, snap - keys
        if new or gone:
            self.stats["late_create"] = self.stats.get("late_create", 0) + len(new)
            self.stats["late_evict"] = self.stats.get("late_evict", 0) + len(gone)
            logger.warning("phys adapter: LATE CREATE after warm-up (step serial %s): new %s, evicted %s",
                           getattr(eng, "step_serial", "?"), sorted(map(str, new))[:8], sorted(map(str, gone))[:8])


class _MM:
    """Images in 2b (design §5a): admission (vLLM's decoder + PhysScheduler's image checks and content keys ->
    MMPlanner.add), frees (forget / release, posted by the C++ loop for its image rids) and the embedded step's plan +
    every rank's EncoderCache. Runs on the engine's main thread (PyExec)."""

    def __init__(self, engine, sched, worker):
        from types import SimpleNamespace
        self.NS = SimpleNamespace
        self.s, self.mm = sched, getattr(sched, "mm", None)
        # EngineCoreProc.process_input_sockets' add decoder (tensor IPC when the frontend uses it; aux frames otherwise)
        self.dec = MsgpackDecoder(EngineCoreRequest, oob_tensor_provider=getattr(engine, "tensor_ipc_receiver", None))
        self.rcache = getattr(engine, "mm_receiver_cache", None)      # None with --mm-processor-cache-gb 0
        self.caches = getattr(worker, "mm_caches", None) if worker is not None else None
        self.stats = dict(admitted=0, rejected=0, plans=0, mm_steps=0, forgets=0, releases=0)

    def _why(self, req):                 # PhysScheduler._unsupported's image part (same texts)
        if self.mm is None:
            return getattr(self.s, "_mm_why_off", "multimodal inputs are not supported by the phys engine (text only)")
        other = sorted({f.modality for f in req.mm_features} - {"image"})
        if other:
            return f"only image inputs are supported by the phys engine (got {', '.join(other)})"
        if any(f.data is None for f in req.mm_features):
            return "image data missing in the engine (a processor-cache entry without its data)"
        return None

    def admit(self, frames):
        """[payload, aux...] of an ADD with mm_features -> ([(offset, length, key)], "") or (None, reason)."""
        req = self.dec.decode(frames)
        if self.rcache is not None and req.mm_features:   # EngineCore.preprocess_add_request
            req.mm_features = self.rcache.get_and_update_features(req.mm_features)
        why = self._why(req)
        got = None
        if not why:
            try:
                got = self.s._mm_images(req)               # content keys + span checks (EngineCoreRequest has its fields)
            except Exception as e:  # noqa: BLE001  (PhysScheduler._to_rs: the key future's exception)
                got = f"image keys failed: {e!r}"
            if isinstance(got, str):
                why = got
        if why:
            self.stats["rejected"] += 1
            logger.error("phys adapter: request %s rejected: %s", req.request_id, why)
            return None, why
        self.mm.add(req.request_id, got)
        self.stats["admitted"] += 1
        return [(im.offset, im.length, im.key) for im in got], ""

    def free(self, finished, preempted):
        """"""
        if self.mm is None:
            return
        for rid in finished:
            self.mm.forget(rid)
            self.stats["forgets"] += 1
        for rid in preempted:
            self.mm.release(rid)
            self.stats["releases"] += 1

    def plan(self, batches):
        if self.mm is None or not self.mm.reqs:
            return None
        self.stats["plans"] += 1
        return self.mm.plan(batches)

    def step(self, view):
        """An embedded image step (the C++ fake's EmbeddedFn): plan this step's image entries, apply drop / encode /
        gather on every rank's EncoderCache; -> per rank (idx, emb) | None (StepContext.mm)."""
        NS = self.NS
        batches = [NS(rank=r, seqs=[NS(rid=rid, kind=k, start=st, n_rows=n, row0=r0, cp=None if o < 0 else NS(owner=o))
                                    for rid, k, st, n, r0, o in seqs]) for r, seqs in view]
        st = self.plan(batches)
        if st is None or not self.caches:
            return None
        self.stats["mm_steps"] += 1
        return [c.apply(st) for c in self.caches]


class _Worker:
    """The C++ loop's IWorker in Python (2b-α0): StepInput dict -> PhysStep -> executor.worker.step -> results."""

    def __init__(self, sched, executor, mm=None):
        from adapter.bridge import PhysStep, ReqInit, GrammarRows   # noqa: F401  (adapter/)
        from interfaces import build_step_batch                    # serve/
        self.sched, self.ex, self.mm = sched, executor, mm
        self.PhysStep, self.ReqInit, self.build = PhysStep, ReqInit, build_step_batch
        self.dec = MsgpackDecoder(EngineCoreRequest)
        self.R = sched.geom.n_ranks
        self._pre: dict = {}                     # plan step -> (StepPlan, StepBatches) built at schedule time
        self._shadow, self._bc, self.checked = None, None, 0

    def prebuild(self, t):
        """At schedule time (the C++ BatchBuilder, before rank 0's batch): the PhysStep's StepPlan + StepBatches, as
        PhysScheduler._schedule builds them; dispatch uses these (a deferred dispatch runs after the next update)."""
        s = self.sched
        plan = s._plan(t)
        self._pre[plan.step] = (plan, [self.build(plan, s.rs.kv, s.geom, r) for r in range(self.R)])

    def dispatch(self, d):
        s = self.sched
        pre = self._pre.pop(d["plan"][0], None) if d["plan"] is not None else None
        plan = pre[0] if pre is not None else (s._plan(d["plan"]) if d["plan"] is not None else None)
        if self.mm is not None and (d["finished"] or d["preempted"]):   # PhysScheduler._phys_free (images) before the plan
            self.mm.free(d["finished"], d["preempted"])
        inits = []
        for rid, rank, prompt, output, structured, payload in d["new_reqs"]:
            sp = self.dec.decode(payload).sampling_params if payload else None
            inits.append(self.ReqInit(rid, rank, list(prompt), list(output), sp, structured))
        R = self.R
        if plan is None or not plan.work:
            batches = (pre[1] if pre is not None else [self.build(plan, s.rs.kv, s.geom, r) for r in range(R)]) \
                if plan is not None else []
            ps = self.PhysStep(step=plan.step if plan is not None else -1, batches=batches,
                               row_tokens=[[] for _ in batches], new_reqs=inits, finished=set(d["finished"]),
                               preempted=set(d["preempted"]), order=[], plan=plan)
        else:
            batches = pre[1] if pre is not None else [self.build(plan, s.rs.kv, s.geom, r) for r in range(R)]
            ps = self.PhysStep(step=plan.step, batches=batches, row_tokens=[list(x) for x in d["rows"]],
                               prev_tokens={r: list(t) for r, t in d["prev"]}, new_reqs=inits,
                               finished=set(d["finished"]), preempted=set(d["preempted"]), order=list(d["order"]),
                               num_spec=dict(d["num_spec"]), est_us=d["est_us"], plan=plan)
        ps.forced, ps.forced_reqs, ps.vocab_size = {}, {}, s.vocab_size
        if self.mm is not None and plan is not None and plan.work:   # PhysScheduler._phys_step: ps.mm every step
            mm_step = self.mm.plan(batches)
            if mm_step is not None:
                ps.mm = mm_step
        grammar = None
        if d["grammar"]:
            from adapter.bridge import GrammarRows
            grammar = {rid: GrammarRows(np.frombuffer(rows, dtype=np.int32).reshape(n, w).copy(), int(acc))
                       for rid, rows, n, w, acc in d["grammar"]}
        if "cxx_batches" in d:                               # PHYS_CXX_CHECK_EMB=1
            if self._shadow is None:
                import worker as WK                          # serve/worker.py (TableBuilder = C++ with PHYS_CXX_TABLES)
                self._shadow, self._bc = WK.TableBuilder(), WK.build_compact
            k1 = int(s.num_spec_tokens) + 1
            py_cts = [self._bc(self._shadow, sb, k1, ps.row_tokens[r]) for r, sb in enumerate(batches)]
            got, gct = _assemble(plan, d["cxx_batches"]), d["cxx_cts"]
            mut = os.environ.get("PHYS_CXX_CHECK_MUTANT", "")   # the check's own test: a perturbed field must fail it
            if mut == "row0" and got and got[0].seqs:
                got[0].seqs[0].row0 += 1
            elif mut == "tokens" and gct and len(gct[0]["tokens"]):
                gct[0]["tokens"][0] ^= 1
            elif mut == "pages" and any(se.block_table for sb in got for se in sb.seqs):
                se = next(se for sb in got for se in sb.seqs if se.block_table)
                se.block_table = list(se.block_table)[:-1]
            _check_batches(plan.step, batches, got, py_cts, gct)
            self.checked += 1
        if d["total"] == 0:                                  # PhysExecutor: frees only (or nothing)
            if ps.finished or ps.preempted or ps.new_reqs:
                self.ex.worker.step(ps, None)
            return (ps, [])
        return (ps, self.ex.worker.step(ps, grammar))

    def wait(self, h):
        ps, handle = h
        results = handle.result() if hasattr(handle, "result") else handle
        return _results(ps.plan, results)


def install(ECP) -> None:
    """Swap EngineCoreProc's IO threads and busy loop for the C++ ones (this process only)."""
    if getattr(ECP, "_cxx_loop_installed", False):
        return

    def process_input_sockets(self, input_addresses, coord_input_address, identity, ready_event):
        try:
            if coord_input_address is not None:
                raise RuntimeError("PHYS_CXX_LOOP=1: a DP coordinator (DP > 1) is not supported")
            ready = msgspec.msgpack.encode(self._make_ready_response())
            srv = X.Server()
            srv.start(list(input_addresses), list(self.addresses.outputs), bytes(identity), ready, 50)
            self._cxx_srv = srv
            logger.info("phys adapter: PHYS_CXX_LOOP=1: engine IO in C++ (inputs %s, outputs %s)", list(input_addresses),
                        list(self.addresses.outputs))
        except Exception:
            logger.exception("phys adapter: PHYS_CXX_LOOP=1: the C++ IO did not start")
            self._cxx_srv = None
            raise
        finally:
            ready_event.set()

    def process_output_sockets(self, output_paths, coord_output_path, engine_index):
        return                                               # T_out (C++) owns the output sockets

    def run_busy_loop(self):
        while getattr(self, "_cxx_srv", "pending") == "pending":
            time.sleep(0.01)
        if self._cxx_srv is None:
            raise RuntimeError("PHYS_CXX_LOOP=1: no C++ IO")
        s = self.scheduler
        cfg, tinfo, reasoners = s._core_cfg
        cfg = dict(cfg)
        # the EngineCore utilities whose answers are fixed at init (vLLM computes them from the executor / config)
        cfg["static_utilities"] = {"get_supported_tasks": tuple(self.get_supported_tasks())}
        try:                                     # the sampling-caps 400: the C++ IO answers it itself
            import adapter.bridge as _br
            _caps = _br.get_sampling_caps() if hasattr(_br, "get_sampling_caps") else None
            cfg["static_utilities"]["phys_sampling_caps"] = None if _caps is None else sorted(_caps)
            logger.info("phys adapter: phys_sampling_caps (static utility of the C++ loop): %s",
                        "None (no check)" if _caps is None else sorted(_caps))
        except Exception as e:  # noqa: BLE001  (no caps utility: the guard logs it and skips the API-side check)
            logger.warning("phys adapter: phys_sampling_caps not served: %s", str(e)[:160])
        if os.environ.get("PHYS_CXX_STEP_LOG"):              # the C++ loop's per-step timing (JSONL), default off
            cfg["step_log"] = os.environ["PHYS_CXX_STEP_LOG"]
        mm = _MM(self, s, getattr(self.model_executor, "worker", None))
        cfg["mm"] = mm
        native = os.environ.get("PHYS_CXX_NATIVE", "")
        if native == "fake":                     # CPU: NativeWorker over the C++ fake device, no Python per step
            cfg["native"] = dict(kind="fake", vocab=int(s.vocab_size), k=int(s.num_spec_tokens), seed=0)
            for key in ("step_ms", "h_frac", "t_ms", "prep_ms"):   # PHYS_CXX_SIM_*: FakeNd's simulated device timeline
                v = os.environ.get(f"PHYS_CXX_SIM_{key.upper()}")
                if v:
                    cfg["native"][f"sim_{key}"] = float(v)
            worker = None
        elif native == "nd":
            here = os.path.dirname(os.path.abspath(__file__))
            # PHYS_CXX_ND_PREBUILT=1: load
            # its build/nd/fpr_nd*.so + build/nd2/fpr_nd2*.so AS BUILT (a frozen, verified package: no torch JIT, which would
            # rebuild whenever the sources moved)
            nd_dir = os.environ.get("PHYS_CXX_ND_DIR") or os.path.join(here, "..", "nd")
            if nd_dir not in sys.path:
                sys.path.insert(0, nd_dir)
            import nd_cxx                        # noqa: E402  (fpr/core/src/nd: fpr_nd2 = nd::Engine, JIT-built)
            if os.environ.get("PHYS_CXX_ND_PREBUILT", "0") == "1":
                import glob
                import importlib.util as ilu
                for attr, sub, name in (("_X", "nd", "fpr_nd"), ("_X2", "nd2", "fpr_nd2")):
                    so = sorted(glob.glob(os.path.join(nd_dir, "build", sub, name + "*.so")))[0]
                    spec = ilu.spec_from_file_location(name, so)
                    m = ilu.module_from_spec(spec)
                    spec.loader.exec_module(m)
                    setattr(nd_cxx, attr, m)     # nd_cxx.ext() / ext2() return these (their module-level caches)
                logger.info("phys adapter: PHYS_CXX_NATIVE=nd: prebuilt nd modules from %s", nd_dir)
            import nd_export                     # noqa: E402
            st4 = self.model_executor.worker
            w = getattr(st4, "w", st4)           # serve's Worker (stage-4 worker wraps it)
            eng = w.backend
            # nd's handover capacity: n_gens / graph_capacity sized for the precaptured H1 grid (each key
            # takes a generation slot; the nd defaults 256 / 8192 overflow at 820+ keys). Host-side arrays; env-tunable.
            import inspect as _insp
            _kw = {}
            _ps = _insp.signature(nd_export.Exporter.__init__).parameters
            if "n_gens" in _ps:
                _kw["n_gens"] = int(os.environ.get("PHYS_CXX_ND_GENS", "4096"))
            if "graph_capacity" in _ps:
                _kw["graph_capacity"] = int(os.environ.get("PHYS_CXX_ND_GRAPH_CAP", "32768"))
            ex = nd_export.Exporter(eng, **_kw)
            logger.info("phys adapter: nd handover capacity %s", _kw or "nd defaults")
            ndo = nd_cxx.ext2().Engine2b()
            emb = _Embedded(s, w, eng, ex, mm, int(s.num_spec_tokens) + 1)
            tb = w.tb
            cfg["native"] = dict(kind="nd", engine=ndo.iengine(), owner=ndo, handover=int(ex.ptr()), embedded=emb,
                                 k=int(s.num_spec_tokens), sources=[int(x) for x in tb.sources],
                                 stub_indexer=bool(tb.stub_indexer), full_block_tables=bool(tb.full_block_tables))
            # native mixed steps (nd's IMixed2b + _Embedded.mixed_body / mixed_post)
            if os.environ.get("PHYS_CXX_MIXED_NATIVE", "0") == "1":
                if not hasattr(eng, "_mixed_body"):
                    raise RuntimeError("PHYS_CXX_MIXED_NATIVE=1 needs the engine's step_mixed split (_mixed_body / _mixed_post)")
                cfg["native"]["mixed"] = True
                cfg["native"]["tail_bound"] = bool(getattr(sys.modules[type(eng).__module__], "TAIL_BOUND", True))
                logger.info("phys adapter: PHYS_CXX_MIXED_NATIVE=1: native mixed steps (beta0a)")
                if os.environ.get("PHYS_CXX_MIXED_EXEC", "0") == "1":   # the whole step natively
                    cfg["native"]["mixed_exec"] = _mixed_snap(eng)          # when every op is
                    logger.info("phys adapter: PHYS_CXX_MIXED_EXEC=1: beta0b executor (core2b::run_mixed)")
                    # owner_p1's C ABI table: nd's exporter lists libs["p1"] only when p1k is loaded
                    # in-process, and nothing else in serving imports it: load it here (the root's prebuild built it). A
                    # failure is not fatal: owner_p1 then refuses every native mixed step with its class.
                    try:
                        _p1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
                            sys.modules[type(eng).__module__].__file__))), "kernels", "p1")
                        if _p1 not in sys.path:
                            sys.path.insert(0, _p1)
                        import p1k as _p1k
                        _p1k.build()
                        logger.info("phys adapter: p1k loaded from %s (owner_p1's C ABI table)", _p1)
                    except Exception as e:   # noqa: BLE001
                        logger.warning("phys adapter: p1k not loaded (%r): native mixed steps refuse at owner_p1", e)
                    if os.environ.get("PHYS_CXX_MIXED_ASYNC", "0") == "1":   # L4 (default off): a py-free native mixed
                        cfg["native"]["mixed_async"] = True                 # step's last sync + finish move to the
                        logger.info("phys adapter: PHYS_CXX_MIXED_ASYNC=1: L4 deferred sync + finish")   # next dispatch
            self._cxx_nd = (ex, ndo, emb)
            logger.info("phys adapter: PHYS_CXX_NATIVE=nd: nd::Engine (handover %#x, %d buckets, emb %s)", ex.ptr(),
                        len(ex._state_of), emb.stats)
            worker = None
        elif native == "embed_all":              # every step through _Embedded (serve Worker + its engine)
            st4 = self.model_executor.worker
            w = getattr(st4, "w", st4)
            emb = _Embedded(s, w, w.backend, None, mm, int(s.num_spec_tokens) + 1)
            tb = w.tb
            cfg["native"] = dict(kind="embed_all", embedded=emb, k=int(s.num_spec_tokens),
                                 sources=[int(x) for x in tb.sources], stub_indexer=bool(tb.stub_indexer),
                                 full_block_tables=bool(tb.full_block_tables))
            self._cxx_nd = (None, None, emb)
            worker = None
        elif native:
            raise RuntimeError(f"PHYS_CXX_NATIVE={native}: 'fake', 'nd' or 'embed_all'")
        else:
            worker = _Worker(s, self.model_executor, mm)
            if os.environ.get("PHYS_CXX_CHECK_EMB", "0") == "1":   # C++-built embedded inputs vs Python's
                cfg["check_k1"] = int(s.num_spec_tokens) + 1
        core = X.Core2b(cfg, s.rs._cx._ischeduler(), s.rs._cx.kv_capsule(), s.rs, tinfo, reasoners, worker,
                        int(self.engine_index))
        self._cxx_core = core
        nd_t_ = getattr(self, "_cxx_nd", None)
        if nd_t_ is not None and len(nd_t_) > 2 and nd_t_[2] is not None:
            nd_t_[2].core = core                 # AGCE: _Embedded's shared previous-AG-bucket calls
            # H3: the H1 epoch fast path, when the engine's mixed_cxx_head has the epoch block (v5+)
            try:
                EM_ = sys.modules[type(nd_t_[2].eng).__module__]
                mxh_ = EM_._mxh() if getattr(EM_, "MIXED_CXX_HEAD", False) else None
                # OFF unless PHYS_CXX_H1_EPOCH=1: push_state sets hsd._hi / resets hsd._slot_key
                # without an HSD epoch bump, so the fast path is unproven; default = the Python guard
                h1ep_ = os.environ.get("PHYS_CXX_H1_EPOCH", "0") == "1"
                if h1ep_ and mxh_ is not None and hasattr(mxh_, "epoch_addr") and hasattr(core, "set_h1_epoch"):
                    core.set_h1_epoch(int(mxh_.epoch_addr()))
                    logger.info("phys adapter: H1 epoch block wired (mixed_cxx_head.epoch_addr)")
                elif mxh_ is not None and hasattr(mxh_, "epoch_addr"):
                    logger.info("phys adapter: H1 epoch fast path OFF (PHYS_CXX_H1_EPOCH != 1): the Python H1 guard runs")
            except Exception as e:  # noqa: BLE001  (no fast path: the Python guard as before)
                logger.warning("phys adapter: H1 epoch block not wired: %s", str(e)[:160])
        nd_t = getattr(self, "_cxx_nd", None)
        if nd_t is not None and nd_t[0] is not None:
            # ONE ensure_bucket + refresh_graphs round right after nd.init (Core2b's constructor) and before the
            # first step, for what warm-up created / captured (main thread, nothing of nd in flight); later rounds run
            # after every embedded step (NativeWorker's post hook)
            nd_t[2].post()
            logger.info("phys adapter: PHYS_CXX_NATIVE=nd: after init: %d buckets ensured, graphs %s",
                        len(nd_t[0]._state_of), nd_t[2].stats)
        self._cxx_srv.attach(core)               # the inputs T_in buffered since the handshake go in first, in order
        core.start()
        logger.info("phys adapter: PHYS_CXX_LOOP=1: the engine loop runs in C++ (fpr_core2b %s)", X.__file__)
        io_log = float(os.environ.get("PHYS_CXX_IO_LOG", "0") or 0)   # > 0: the C++ IO counters every io_log s (hangs)
        t_io = time.monotonic()
        # PHYS_CXX_PROF_TRIGGER=<path> (default off): GPU-only kernel traces of the C++ loop's steps, native ones included
        # (the engine's FPR_PROF_* profiler only ticks on Python steps). Window w starts once "<path>.<w>" exists and lasts
        # PHYS_CXX_PROF_S seconds; written as $FPR_PROF_DIR/core2b_trace_w<w>.json.gz. CUPTI records every thread's
        # kernels, so the trace holds the native steps the C++ loop issues while this thread only serves Python jobs
        prof_trig = os.environ.get("PHYS_CXX_PROF_TRIGGER", "")
        prof_s = float(os.environ.get("PHYS_CXX_PROF_S", "2") or 2)
        prof = dict(p=None, t=0.0, w=0)
        try:
            while True:
                if prof_trig:
                    if prof["p"] is None and os.path.exists(f"{prof_trig}.{prof['w'] + 1}"):
                        from torch.profiler import ProfilerActivity, profile
                        prof["p"] = profile(activities=[ProfilerActivity.CUDA])
                        prof["p"].__enter__()
                        prof["t"], prof["w"] = time.monotonic(), prof["w"] + 1
                        logger.info("phys adapter: PHYS_CXX_PROF: window %d started (%.1f s)", prof["w"], prof_s)
                    elif prof["p"] is not None and time.monotonic() - prof["t"] >= prof_s:
                        prof["p"].__exit__(None, None, None)
                        path = os.path.join(os.environ.get("FPR_PROF_DIR", "/tmp"), f"core2b_trace_w{prof['w']}.json.gz")
                        prof["p"].export_chrome_trace(path)
                        prof["p"] = None
                        logger.info("phys adapter: PHYS_CXX_PROF: window %d -> %s", prof["w"], path)
                if io_log > 0 and time.monotonic() - t_io >= io_log:
                    t_io = time.monotonic()
                    logger.info("phys adapter: PHYS_CXX_IO_LOG: server %s has_work %s worker %s", self._cxx_srv.stats(),
                                core.has_work(), core.worker_stats())
                # this thread built the device engine (its CUDA current devices / streams are thread-local): every
                # Python call the C++ loop needs (prebuild / dispatch / wait) runs here; between jobs, signals
                # (SIGTERM -> SystemExit) and the loop's fatal state are checked
                for _ in range(64):
                    if core.serve_python(0.2) <= 0:
                        break
                why = core.fatal()
                if why:                                      # the C++ loop stopped: the executor failed
                    raise RuntimeError(f"PHYS_CXX_LOOP=1: engine loop failed: {why}")
                # vLLM's SIGTERM / SIGINT handler only sets shutdown_state (EngineCoreProc._handle_shutdown): abort mode
                # (shutdown_timeout 0) exits now, drain mode once the C++ loop has no work left
                st = getattr(self, "shutdown_state", None)
                if st is not None and getattr(st, "name", "RUNNING") != "RUNNING":
                    timeout = getattr(self.vllm_config, "shutdown_timeout", 0) or 0
                    if timeout == 0 or not core.has_work():
                        logger.info("[shutdown] EngineCore (PHYS_CXX_LOOP=1): leaving the C++ loop (mode=%s)",
                                    "abort" if timeout == 0 else "drain")
                        raise SystemExit
        except BaseException as e:
            # vLLM's run_engine_core: SystemExit (SIGTERM / SIGINT shutdown) exits quietly; any other exception tells the
            # clients ENGINE_CORE_DEAD (EngineCoreProc._send_engine_dead)
            try:
                core.stop()
                if isinstance(e, Exception):
                    self._cxx_srv.dead()
            finally:
                self._cxx_srv.stop()
                # each line on its own
                lines = (("core2b worker stats", lambda: f"{core.worker_stats()}; mm {mm.stats}; planner "
                                                         f"{getattr(mm.mm, 'stats', None)}"),
                         ("core2b native stats", lambda: core.native_stats()),
                         ("core2b nd", lambda: f"embedded {self._cxx_nd[2].stats}" if getattr(self, "_cxx_nd", None) else "-"),
                         ("core2b on_ctx lazy", lambda: (lambda st: f"replays {st.get('lazy_replays', 0)}, replayed "
                                                       f"{st.get('lazy_replayed', 0)}, check {'on' if _ONCTX_CHECK else 'off'}: "
                                                       f"rids {st.get('onctx_check_rids', 0)} bad {st.get('onctx_check_bad', 0)}, "
                                                       f"shadow on_ctx {st.get('onctx_shadow', 0)} (final)")(self._cxx_nd[2].stats)
                          if getattr(self, "_cxx_nd", None) else "-"),
                         ("core2b IO", lambda: self._cxx_srv.stats()),
                         ("core2b rank scheduler", lambda: s.rs.stats),
                         ("core2b check", lambda: f"{worker.checked} steps' embedded inputs IDENTICAL (C++ vs Python)"
                          if worker is not None and getattr(worker, "checked", 0) else "-"))
                for name, fn in lines:
                    try:
                        print(f"{name}: {fn()}", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        print(f"{name}: unavailable ({ex!r})", flush=True)
            raise

    ECP.process_input_sockets = process_input_sockets
    ECP.process_output_sockets = process_output_sockets
    ECP.run_busy_loop = run_busy_loop
    ECP._cxx_loop_installed = True
    print(f"phys_scheduler: PHYS_CXX_LOOP=1: EngineCoreProc IO + busy loop in C++ ({X.__file__})", flush=True)
