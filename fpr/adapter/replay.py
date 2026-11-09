"""Replay the served path's worker inputs through serve's Worker without vLLM, and write the forced requests' logits as
patch_force dumps: the decisive served-vs-offline A/B.

  served run:  PHYS_DUMP_STEPS=<dir>:100000 scripts/serve_fpr.sh ...    every worker input, pickled in order
  replay:      python adapter/tests/replay_step.py <dir> --force-dump <dump dir>   (GPUs: the real engine)

The replay drives the Worker exactly as the worker wrapper (adapter/stage4_worker.py) does in the server: the same PhysSteps, frees-only steps
held for the next step with rows, -1 draft rows filled on the host only for an engine without device drafts. The
forced requests are recorded with the adapter's own ForcedReq (adapter/force.py -> serve/force_dump.write_item), so the
replayed dump has the served dump's format and tag naming, so the same comparison reads both unchanged.
  replay == served dump  -> the served process computed what its inputs say: look at the inputs (vs offline's)
  replay == offline dump -> same inputs, different result: the served process's runtime state differs
The pickles hold prompt token ids: keep them in mode-700 directories when the prompts are real traffic.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from adapter import bridge  # noqa: E402,F401  (serve/ on sys.path)
from adapter.force import ForcedReq  # noqa: E402


def step_files(paths) -> list:
    """Dump files in step order: directories expand to their step{i}.pkl, sorted by i."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            fs = [f for f in os.listdir(p) if re.fullmatch(r"step\d+\.pkl", f)]
            out += [os.path.join(p, f) for f in sorted(fs, key=lambda f: int(f[4:-4]))]
        else:
            out.append(p)
    return out


def is_async(files: list, probe: int = 400) -> bool:
    """Dumps of an ASYNC=1 server: some decode group left its position to the device (start -1)."""
    for f in files[:probe]:
        with open(f, "rb") as fh:
            ps = pickle.load(fh)["ps"]
        if any(se.kind == "decode" and se.start < 0 for sb in ps.batches for se in sb.seqs):
            return True
    return False


def make_worker(engine_cls: str = "engine.engine.Engine", engine_kw: dict | None = None, max_rows: int = 320,
                max_seqs: int = 160, log=print, async_mode: bool = False):
    """serve's Worker(backend="engine") with the served geometry (scripts/serve_fpr.sh: max_model_len 655,360, k = 3, MAX_SEQS
    160) and the worker wrapper's max_rows."""
    import dataclasses

    import geometry as G
    import worker as W

    cfg = W.WorkerConfig()
    cfg = dataclasses.replace(cfg, engine_cls=engine_cls, engine_kw=dict(engine_kw or {}))
    if async_mode:                                    # D13: compact tables, engine.step_compact = dispatch
        cfg = dataclasses.replace(cfg, async_mode=True)
    if os.environ.get("DS41H_MODEL_DIR"):
        cfg = dataclasses.replace(cfg, model=os.environ["DS41H_MODEL_DIR"])
    geom = G.Geometry(max_model_len=655360, spec_k=3, max_seqs=max_seqs)
    return W.Worker(cfg, geom, "engine", max_rows=max_rows, log=log)


def _device_drafts(w) -> bool:
    from adapter.stage4_worker import Stage4Worker

    class _Shim:
        pass
    s = _Shim()
    s.w = w
    return Stage4Worker._device_drafts(s)


def _describe(ps) -> str:
    """One step's composition (no token ids): per rank rows, sequences by kind, async / first decodes, uids, slots."""
    out = []
    for sb in ps.batches:
        kinds = {}
        for se in sb.seqs:
            k = "dec-1" if se.kind == "decode" and se.start < 0 else se.kind
            kinds[k] = kinds.get(k, 0) + 1
        uids = sorted(int(getattr(se, "uid", -1)) for se in sb.seqs)
        slots = sorted(int(se.state_slot) for se in sb.seqs)
        out.append(f"r{sb.rank}: {sb.n_rows} rows {kinds} uids {uids[:3]}..{uids[-3:]} slots max {max(slots, default=-1)}")
    return f"step {ps.step}: new {len(ps.new_reqs)} dropped {len(ps.finished) + len(ps.preempted)} | " + " | ".join(out)


def replay_async(files: list, w, sync_each: bool = False, log=print) -> dict:
    """ASYNC=1 dumps through the D13 dispatch model, as PhysExecutor drives it: worker.step on one thread in step order
    (dispatch(N + 1) queues N + 1, waits for N, extracts it), each handle's result() collected after the next dispatch
    (the served step thread). sync_each: synchronize every device after each dispatch, so a CUDA fault is pinned to the
    step that queued it (slower; with CUDA_LAUNCH_BLOCKING=1 to the kernel)."""
    import torch

    pending: set = set()
    steps = 0
    prev = None
    last = []
    t0 = time.time()
    devs = list(getattr(getattr(w, "backend", None), "devs", []) or [])
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        ps, grammar = d["ps"], d.get("grammar")
        if not ps.batches or not any(sb.n_rows for sb in ps.batches):      # frees only: as the worker wrapper
            pending |= set(ps.finished) | set(ps.preempted)
            continue
        if pending:
            import copy

            ps = copy.copy(ps)
            ps.finished = set(ps.finished) | pending
            pending = set()
        desc = _describe(ps)
        last = (last + [f"{os.path.basename(f)} {desc}"])[-6:]
        try:
            h = w.step(ps, grammar)
            if prev is not None and hasattr(prev, "result"):
                prev.result()
            if sync_each and torch.cuda.is_available():
                for dv in devs:
                    torch.cuda.synchronize(dv)
        except Exception as e:  # noqa: BLE001
            log(f"replay: FAULT at dump {os.path.basename(f)} (#{i}, {steps} steps run): {type(e).__name__}: "
                f"{str(e).splitlines()[0][:200]}")
            for ln in last:
                log(f"  {ln}")
            return dict(fault=os.path.basename(f), index=i, steps=steps, error=str(e).splitlines()[0][:200],
                        last=last, seconds=round(time.time() - t0, 1))
        prev = h
        steps += 1
        if steps % 200 == 0:
            log(f"replay: {steps} steps ({os.path.basename(f)}), {time.time() - t0:.0f} s")
    if prev is not None and hasattr(prev, "result"):
        prev.result()
    out = dict(fault=None, files=len(files), steps=steps, seconds=round(time.time() - t0, 1))
    log(f"replay: {out}")
    return out


def replay(files: list, w, force_dump: str | None = None, log=print) -> dict:
    """Run every dumped PhysStep through Worker w in order; forced requests -> dumps under force_dump (default: the
    served dump dir recorded in the step, which you do NOT want to overwrite: pass force_dump)."""
    from adapter.stage4_worker import Stage4Worker

    dev = _device_drafts(w)
    frs: dict = {}
    pending: set = set()
    steps = rows = 0
    t0 = time.time()
    vocab = None
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        ps, grammar = d["ps"], d.get("grammar")
        vocab = getattr(ps, "vocab_size", vocab)
        for sb in ps.batches:
            for se in sb.seqs:
                assert not (se.kind == "decode" and se.start < 0), \
                    f"{f}: an async-mode step (device-resolved positions) cannot be replayed on the host"
        if not ps.batches or not any(sb.n_rows for sb in ps.batches):      # frees only: as the worker wrapper
            pending |= set(ps.finished) | set(ps.preempted)
            continue
        if pending:
            import copy

            ps = copy.copy(ps)
            ps.finished = set(ps.finished) | pending
            pending = set()
        res = w.step(ps if dev else Stage4Worker._fill_drafts(ps), grammar)
        res = res.result() if hasattr(res, "result") else list(res)
        steps += 1
        rows += sum(sb.n_rows for sb in ps.batches)
        specs = getattr(ps, "forced_reqs", None) or {}
        for rid, e in (getattr(ps, "forced", None) or {}).items():           # as merge_results
            if rid not in frs:
                sp = specs[rid]
                sp = sp.spec() if isinstance(sp, ForcedReq) else sp                # dumps from before 16:55 held objects
                frs[rid] = ForcedReq.from_spec(rid, sp, force_dump)
            fr = frs[rid]
            toks = [t for r in res for t in r.tokens.get(rid, ())]
            if toks != list(e):
                fr.overrides += 1
            fed = None
            for r in res:
                fed = (getattr(r, "fed", None) or {}).get(rid, fed)
            if fed is not None:
                fr.input_mismatch += sum(1 for a, b in zip(fed[1:], e[:-1]) if a != b)
        for rid in specs:
            fr = frs.get(rid)
            if fr is None:
                continue
            fr.record(res)
            if fr.done() and not fr.flushed:
                fr.flush(vocab)
    for fr in frs.values():
        if not fr.flushed:
            fr.flush(vocab, partial=True)
    out = dict(files=len(files), steps=steps, rows=rows, seconds=round(time.time() - t0, 1),
               forced={fr.tag: dict(n=len(fr.idx), n_dump=fr.n_dump, input_mismatch=fr.input_mismatch,
                                    dump_dir=fr.dump_dir) for fr in frs.values()})
    log(f"replay: {out}")
    return out
