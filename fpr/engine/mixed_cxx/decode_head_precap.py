"""decode_head_precap: the DECODE graph H of every decode bucket R captured
BEFORE its first use, so no decode step of a fresh process is embedded for "graph H not captured / published".

ENG_DECODE_HEAD_PRECAPTURE=1 (default 0), called from mixed_cxx_head.precapture.

Decode graph H is keyed (R, "H", hooks.hsd_key()) and only used when hsd_key's S_pad == R (Hooks.hsd_graph_ok). With the
production buckets (HSD seq buckets = R / 4 of the decode buckets, row buckets = 4 x seq buckets) every decode step of bucket
R has the key (R, R / 4, NT = R) + gvar; gvar is (("c1d", owner),) for a C1 step (D_pad 1, one owner rank, S_pad <= SMAX:
kernels/hsdtail/hsdt_c1d.py), else (). ENG_DECODE_PRECREATE creates every R but captures graph H only for the keys its
PADDED warm-up steps happen to have (their natural, smaller keys), so each R's real key is first-use.

Per decode bucket R whose state exists (eng.bk, created by PRECREATE) and per natural sequence count d of R (bucket_of(d * (k
+ 1)) == R), on every owner rank for d == 1 (the c1d variants), else rank 0: a synthetic decode HSD.prepare (rank `owner`
holds d decode sequences of k + 1 rows, all sampled; the other ranks empty; only HSD's per-step tables / blob are written),
then hkey = eng._hkey(R) with hooks.bm_step forced False, and eng._capture_split(R, st, False, hkey) for a key not captured
yet (CUDA graph capture only: nothing executes; Hooks.hsd_graph records the copies of the bucket's h / aux into HSD's inputs
and HSD's run). HSD's per-step state is saved / restored around it all (mixed_cxx_head.hsd_save / hsd_restore).
Skipped, with the reason: no state for R, S_pad != R (no graph H for that key), captured keys, keys whose drafter state a
first EAGER run would create (HSD_SKW8, Rd <= 32, no _skws yet). Grammar ("bm") variants stay first-use.
Returns dict(captured=[hkey...], skipped={(R, d, owner): why}).

v2: ENG_DECODE_HEAD_PRECAPTURE_GRID=full (default) plans
EVERY reachable decode key, not only the natural full-draft steps: per bucket R and per S = R_prev + 1 .. R rows (S_pad == R)
and d = ceil(S / K1) .. min(S, max_seqs) sequences of 1 .. K1 rows (k-truncated drafts near max-len / structured requests
raise d at the same rows), one representative (S, d) per distinct (R, D_pad); d == 1 on every owner rank (the c1d variants).
Under the engine's ENG_HEAD_BM_GRAPH every planned key also gets its grammar twin (R, "H", key + ("bm",)): hooks.bm_step
forced True and an all-allowed synthetic mask per sampled row (the variant's capture records the mask H2D's consumer, not the
mask values). GRID=natural = v1's plan (full-draft steps only, no bm twins). Logs keys / seconds / every GPU's free GiB before
-> after; the result dict carries them (s, free_gib).
"""
from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace as NS

import torch

ON = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE", "0") == "1"
# TRACE: after the precapture, log every FIRST-USE decode graph H capture (a key not captured yet
# R a decode bucket: the embedded first step this module removes) and the first replay of each precaptured key, by wrapping
# this engine instance's _capture_split / replay. DRY: the trace without any precapture.
# the replay wrapper adds a
# Python call to every decode graph replay (the hot path)
TRACE = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE_TRACE", "0") == "1"
GRID = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE_GRID", "full")   # (v2) full | natural
DRY = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE_DRY", "0") == "1"
# ENG_DECODE_HEAD_PRECAPTURE_LVL1=1: also the long-row level-1 variants Rk = R + LVL_TAG of every
# planned bucket R whose Rk state ALREADY exists: graph H (Rk, "H", key) with the same synthetic plan (S_pad == R % LVL_TAG). Default 0 = v3 exactly.
LVL1 = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE_LVL1", "0") == "1"
# ENG_DECODE_HEAD_PRECAPTURE_PRELOAD=1: before each key's
# capture, head_preload.preload(eng, hooks.hsd_graph body): a throwaway probe capture records the key's Triton kernels (compiled
# with the same args, no launch) and matmul problems, then loads them OUTSIDE any capture (Triton _init_handles; matmuls replayed
# on scratch tensors of the same shape / stride / dtype / pointer phase). No live buffer is executed on. Default 0 = v4.
PRELOAD = os.environ.get("ENG_DECODE_HEAD_PRECAPTURE_PRELOAD", "0") == "1"


def _install_trace(eng, pre, lg) -> None:
    if eng.__dict__.get("_dhp_trace") is not None:
        return
    tr = eng._dhp_trace = dict(pre=set(pre), first_use=0, replayed=set())
    cap0, rep0 = eng._capture_split, eng.replay
    bs = {int(x) for x in eng.cfg.buckets}

    def cap(R, st, layers, hkey, lkey=None):
        if isinstance(hkey, tuple) and len(hkey) == 3 and hkey[1] == "H" and hkey[0] in bs and hkey not in eng.graphs:
            tr["first_use"] += 1
            lg(f"decode_head_precap trace: first-use decode graph H capture {hkey} (#{tr['first_use']})")
        return cap0(R, st, layers, hkey) if lkey is None else cap0(R, st, layers, hkey, lkey=lkey)

    def rep(key, *a, **k):
        if key in tr["pre"] and key not in tr["replayed"]:
            tr["replayed"].add(key)
            lg(f"decode_head_precap trace: precaptured {key} replayed ({len(tr['replayed'])} of {len(tr['pre'])})")
        return rep0(key, *a, **k)
    eng._capture_split, eng.replay = cap, rep
    lg(f"decode_head_precap trace: installed ({len(tr['pre'])} precaptured keys{', DRY: none captured' if DRY else ''})")


def decode_batches(d: int, K1: int, owner: int, W: int, anypage, scratch_pages, rows=None):
    """Synthetic StepBatches + states: rank `owner` = d decode sequences of K1 rows (v2: rows[i] rows each, 1 .. K1) (all
    sampled, all tail rows), others empty. Positions / slots arbitrary but valid (nothing executes)."""
    rows = list(rows) if rows is not None else [K1] * d
    assert len(rows) == d and all(1 <= x <= K1 for x in rows), rows
    # real interfaces dataclasses when that module is loaded
    # paged_slot = PAD (no paged KV write), empty block tables; else the namespaces with the same fields
    IF = sys.modules.get("interfaces")
    PAD = int(getattr(IF, "PAD", -1))

    def mk(rank, **kw):
        kw.setdefault("paged_slot", [PAD] * len(kw["positions"]))
        if IF is not None and hasattr(IF, "StepBatch"):
            return IF.StepBatch(rank, 0, len(kw["positions"]), **kw)
        return NS(rank=rank, step=0, n_rows=len(kw["positions"]), copies=[], poison=[], **kw)

    def se(**kw):
        full = dict(start=1000, num_computed=1000, cached_upto=0, last_chunk=False, block_table=[], tail_row0=kw["row0"],
                    n_blocks=0)
        full.update(kw)
        if IF is not None and hasattr(IF, "SeqEntry"):
            import dataclasses
            names = {f.name for f in dataclasses.fields(IF.SeqEntry)}
            return IF.SeqEntry(**{k: v for k, v in full.items() if k in names})
        return NS(**full)

    def empty(r):
        return mk(r, positions=[], seq_slot=[], sample_rows=[], tail_rows=[], window_slot=[], seqs=[])
    seqs, positions, seq_slot, sample_rows, states = [], [], [], [], {}
    wt = anypage(scratch_pages[owner])
    row0 = 0
    for i in range(d):
        rid = f"__dprecap_{owner}_{i}"
        seqs.append(se(rid=rid, kind="decode", row0=row0, n_rows=rows[i], state_slot=i, window_table=wt, replay_start=0))
        for j in range(rows[i]):
            positions.append(1000 + j)
            seq_slot.append(i)
            sample_rows.append(row0 + j)
        row0 += rows[i]
        states[rid] = NS(temperature=1.0, seed=0, drafts_valid=False, num_acceptable_drafts=None)
    n = row0
    b = mk(owner, positions=positions, seq_slot=seq_slot, sample_rows=sample_rows, tail_rows=list(range(n)),
           window_slot=[0] * n, seqs=seqs)
    return [b if r == owner else empty(r) for r in range(W)], states


def plan(buckets, K1: int, max_seqs: int, W: int):
    """[(R, d, owner)] of the natural decode steps: every d whose bucket_of(d * K1) is R; d == 1 on every owner rank."""
    bs = sorted(int(x) for x in buckets)
    out = []
    for d in range(1, max_seqs + 1):
        R = next((x for x in bs if d * K1 <= x), None)
        if R is None:
            break
        for owner in (range(W) if d == 1 else (0,)):
            out.append((R, d, owner))
    return out


def plan_full(buckets, K1: int, seq_buckets, row_buckets, max_seqs: int, W: int):
    """(v2) [(R, rows, owner)]: one representative synthetic step per reachable decode key (R, S_pad = R, D_pad) (module
    docstring); rows = the per-sequence row counts (sum S, each 1 .. K1)"""
    bs = sorted(int(x) for x in buckets)
    bkt = lambda n, xs: next((x for x in xs if n <= x), None)   # noqa: E731
    out, seen = [], set()
    prev = 0
    for R in bs:
        for S in range(prev + 1, R + 1):
            if bkt(S, row_buckets) != R:          # no graph H unless HSD's S_pad == the engine bucket
                continue
            for d in range(-(-S // K1), min(S, max_seqs) + 1):
                Dp = bkt(d, seq_buckets)
                if Dp is None or (R, Dp) in seen:
                    continue
                seen.add((R, Dp))
                q, m = divmod(S, d)
                rows = tuple([q + 1] * m + [q] * (d - m))
                for owner in (range(W) if d == 1 else (0,)):
                    out.append((R, rows, owner))
        prev = R
    return out


def _bm_words(hsd):
    for klass in type(hsd).__mro__:
        m = sys.modules.get(klass.__module__)
        if m is not None and hasattr(m, "BM_WORDS"):
            return int(m.BM_WORDS)
    return None


def _free_gib(devs):
    out = []
    for d in devs:
        try:
            out.append(round(torch.cuda.mem_get_info(torch.device(d))[0] / 2 ** 30, 3))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def precapture(eng, save, restore, anypage, log=None) -> dict:
    lg = log or eng.log
    out = dict(captured=[], skipped={})
    H = eng.hooks
    hsd = getattr(H, "hsd", None)
    if not ON or hsd is None or not hasattr(eng, "_capture_split") or not hasattr(eng, "_hkey"):
        return out
    if DRY:
        if TRACE:
            _install_trace(eng, (), lg)
        return out
    W = len(eng.devs)
    K1 = hsd.k + 1
    skw8 = bool(getattr(sys.modules.get("drafter"), "HSD_SKW8", False))
    sp = [int(eng.pools[r].scratch_window_page) for r in range(W)]
    import time
    EM = sys.modules.get(type(eng).__module__)
    bmg = bool(getattr(EM, "HEAD_BM_GRAPH", False)) and GRID == "full"
    bmw = _bm_words(hsd) if bmg else None
    ms = int(getattr(hsd, "max_seqs", 80))
    if GRID == "full":
        items = [(R, rows, owner) for R, rows, owner in plan_full(eng.cfg.buckets, K1, hsd.seq_buckets, hsd.row_buckets, ms, W)]
    else:
        items = [(R, (K1,) * d, owner) for R, d, owner in plan(eng.cfg.buckets, K1, ms, W)]
    tagv = int(getattr(EM, "LVL_TAG", 1 << 20))
    if LVL1:   # (v4) the level-1 twins of the plan, for the Rk states that exist (never created here)
        items = items + [(R + tagv, rows, owner) for R, rows, owner in items if dict.__contains__(eng.bk, R + tagv)]
    for dv in eng.devs:
        torch.cuda.synchronize(dv)
    t0, f0 = time.perf_counter(), _free_gib(eng.devs)
    sv = save(hsd)
    bm0 = getattr(H, "bm_step", False)
    seen = set()
    try:
        H.bm_step = False
        for R, rows, owner, bm in [(R, rows, o, b) for R, rows, o in items for b in ((False, True) if bmw else (False,))]:
            d = len(rows)
            tag = (R, d, owner, sum(rows)) + (("bm",) if bm else ())
            H.bm_step = bm
            st = dict.get(eng.bk, R)   # (no LRU bump, never a creation: PRECREATE made them)
            if st is None:
                out["skipped"][tag] = "no decode state for R"
                continue
            batches, states = decode_batches(d, K1, owner, W, anypage, sp, rows=rows)
            bms = None
            if bm:   # (v2) an all-allowed mask per sampled row of the owner rank, sample-row order (HSD.prepare's contract)
                bms = [torch.full((len(b.sample_rows), bmw), -1, dtype=torch.int32) if b.sample_rows else None for b in batches]
            Dp = next((x for x in hsd.seq_buckets if d <= x), None)
            if Dp is None:
                out["skipped"][tag] = "d beyond HSD's seq buckets"
                continue
            ds = [hsd.draft_rank(r, Dp) for r in range(W)]
            if skw8 and ds[0].Rd <= 32 and any(getattr(x, "_skws", None) is None for x in ds):
                out["skipped"][tag] = "HSD_SKW8 workspace not created yet (needs one eager run of this D_pad)"
                continue
            for x in list(hsd.ranks0) + list(getattr(hsd, "drk", {}).values()):
                x.mix_ev = None            # (GPUs synced: a previous call's mix events are done; not a capture input)
            try:
                with contextlib.ExitStack() as es:
                    for r in range(W):
                        es.enter_context(torch.cuda.stream(eng.streams[r]))
                    hsd.prepare(batches, states, bms)
                for dv in eng.devs:
                    torch.cuda.synchronize(dv)
            except Exception as e:  # noqa: BLE001
                out["skipped"][tag] = f"synthetic prepare: {type(e).__name__}: {str(e)[:120]}"
                continue
            k = H.hsd_key()
            if k is None or k[0] != R % tagv:   # (v4: Rk's rows = R % LVL_TAG; == R below level 1)
                out["skipped"][tag] = f"no graph H for key {k} (S_pad != R)"
                continue
            hkey = eng._hkey(R)
            if hkey is None:
                out["skipped"][tag] = "no H key"
                continue
            if hkey in seen or hkey in eng.graphs:
                continue                   # (the same key from another d / owner, or already captured: not a skip)
            seen.add(hkey)
            if PRELOAD:   # (v5) the key's modules loaded outside the capture, on scratch only
                try:
                    _hd = os.path.dirname(os.path.abspath(__file__))
                    if _hd not in sys.path:
                        sys.path.append(_hd)
                    import head_preload
                    pr = head_preload.preload(eng, lambda r, st_=st: H.hsd_graph(r, st_[r]), tag=f"dhp {hkey}",
                                              pools=[eng.gpool[r] for r in range(W)], roots=[hsd])
                    out["preload"] = {k_: out.get("preload", {}).get(k_, 0) + v_ for k_, v_ in pr.items()
                                      if k_ not in ("s", "t0", "t1")}
                    for x in list(hsd.ranks0) + list(getattr(hsd, "drk", {}).values()):
                        x.mix_ev = None    # events recorded in the destroyed probe capture: never waited on
                except Exception as e:  # noqa: BLE001
                    out["skipped"][tag] = f"preload: {type(e).__name__}: {str(e)[:120]}"
                    continue
            try:
                eng._capture_split(R, st, False, hkey)
            except Exception as e:  # noqa: BLE001
                out["skipped"][tag] = f"capture: {type(e).__name__}: {str(e)[:120]}"
                continue
            if hkey in eng.graphs:
                out["captured"].append(hkey)
    finally:
        H.bm_step = bm0
        restore(hsd, sv)
        for dv in eng.devs:
            torch.cuda.synchronize(dv)
    out["s"] = round(time.perf_counter() - t0, 2)
    out["free_gib"] = dict(before=f0, after=_free_gib(eng.devs))
    out["grid"] = GRID
    out["lvl1"] = LVL1
    out["preload_on"] = PRELOAD
    eng.stats["decode_h_precaptured"] = len(out["captured"])
    if TRACE:
        _install_trace(eng, out["captured"], lg)
    lg(f"decode_head_precap: GRID={GRID} {len(out['captured'])} keys in {out['s']} s (preload {PRELOAD}: {out.get('preload')}), "
       f"free GiB per GPU {out['free_gib']}")
    lg(f"decode_head_precap: {len(out['captured'])} decode graph H keys captured {out['captured']}; skipped "
       f"{len(out['skipped'])}: {dict(list(out['skipped'].items())[:6])}")
    return out
