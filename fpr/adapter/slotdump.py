"""PHYS_D13_SLOTDUMP=<file.jsonl>: per-slot device state of sequences whose last prefill chunk just ran, for the D13
slot-reuse fault (the per-slot device state: rowexp ring / nc / dbt, HSD by_slot, Engram ring).

After every dispatch that contains a last prefill chunk, those (rank, state_slot) are watched for 3 dispatches (the
fallback itself, the first pipelined decode, one more). Each watched dispatch: all devices synchronized (debug only:
no overlap), then one JSON line per watched slot with the device state next to the host tables of that step:
  rowexp nc[slot], the ring entries (block, page) the device holds for the slot vs the host window table, dbt[slot]
  columns 0 .. n_blocks + 8 vs the host block table, rowexp err; HSD by_slot[g] and next_ids row; Engram ring[slot].
A CUDA fault inside the synchronize names the dispatch whose work faulted. Read only; never changes engine state."""
from __future__ import annotations

import json
import os
import time

_WATCH: dict = {}
_N = {"i": 0}
_HIST: list = []          # the host's entries of the last dispatches: [(dispatch, {rid: entry})] (finish tracking)
_DROPWATCH = {"left": 0}  # dispatches still to sync + range-check after one that carried drops (finishes / preemptions)


def _eng(executor):
    w = getattr(executor.worker, "w", None)
    return getattr(w, "backend", None)


from adapter import debug_patches  # noqa: E402


class SlotDumpFault(RuntimeError):
    """A CUDA fault that surfaced at the slot dump's synchronize (the dispatch whose work faulted)."""


_WARNED = {"x": False}


def maybe_dump(executor, ps) -> None:
    """Debug hook: never fails the server except for a real CUDA fault surfacing at its synchronize."""
    try:
        _maybe_dump(executor, ps)
    except SlotDumpFault:
        raise
    except Exception as e:  # noqa: BLE001
        if not _WARNED["x"]:
            _WARNED["x"] = True
            print(f"phys adapter: slot dump error (ignored from now on): {type(e).__name__}: {str(e)[:200]}", flush=True)


def _maybe_dump(executor, ps) -> None:
    path = os.environ.get("PHYS_D13_SLOTDUMP", "")
    if not path:
        return
    eng = _eng(executor)
    d13 = getattr(eng, "_d13_obj", None)
    if d13 is None:
        return
    import torch

    if os.environ.get("PHYS_D13_SUBSYNC") == "1" and not _SUB["on"]:
        install_subsync(eng, d13, path)
    _SUB["step"] = int(ps.step) + 1                       # the next dispatch's step (phases run inside it)
    _N["i"] += 1
    debug_patches.STATE["dispatch"] = _N["i"]
    if any(se.kind != "decode" for sb in ps.batches for se in sb.seqs):
        _PRE["after_fb"] = 3                               # a fallback (mixed) dispatch: check the next 3 pre phases
    ent = {}
    for sb in ps.batches:
        for se in sb.seqs:
            nb = int(se.n_blocks or len(se.block_table))
            ent[se.rid] = (sb.rank, int(se.state_slot), int(getattr(se, "uid", -1)), int(getattr(se, "inc", 0)), se.kind,
                           int(se.start), int(getattr(se, "known", -1)), nb, len(se.block_table),
                           max(se.window_table) if se.window_table else -1)
            if se.kind == "prefill" and se.last_chunk:
                _WATCH[(sb.rank, int(se.state_slot))] = dict(rid=str(se.rid), left=3, end=int(se.start + se.n_rows),
                                                              since=_N["i"])
    _HIST.append((_N["i"], ent))
    del _HIST[:-6]
    _shape(eng, ps, path)
    dropped = set(ps.finished) | set(ps.preempted)
    if dropped:
        _DROPWATCH["left"] = 3            # this dispatch and the 2 after it
    allw = os.environ.get("PHYS_D13_SLOTDUMP_ALL") == "1"
    if not _WATCH and _DROPWATCH["left"] <= 0 and not allw:
        return
    _DROPWATCH["left"] -= 1
    t0 = time.time()
    fault = None
    try:
        for d in eng.devs:
            torch.cuda.synchronize(d)
    except Exception as e:  # noqa: BLE001
        fault = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    host = {}
    for sb in ps.batches:
        for se in sb.seqs:
            host[(sb.rank, int(se.state_slot))] = se
    out = []
    for key, w in list(_WATCH.items()):
        r, slot = key
        rec = dict(dispatch=_N["i"], step=int(ps.step), rank=r, slot=slot, rid=w["rid"], end=w["end"],
                   since=w["since"], fault=fault)
        se = host.get(key)
        if se is not None:
            rec.update(host_rid=str(se.rid), kind=se.kind, start=int(se.start), known=int(getattr(se, "known", -1)),
                       n_blocks=int(se.n_blocks or len(se.block_table)), host_bt=list(se.block_table[: (se.n_blocks or len(se.block_table)) + 2]),
                       host_wt=sorted((int(b), int(p)) for b, p in se.window_table.items()))
        if fault is None:
            try:
                rx = d13.rx[r]
                rec["nc"] = int(rx.nc[slot].item())
                rg = rx.ring[slot].cpu()
                rec["ring"] = sorted((int(b), int(p)) for b, p in rg.tolist() if b >= 0)
                nb = rec.get("n_blocks", -(-(w["end"] + 8) // 64))
                rec["dbt"] = rx.dbt[slot, : nb + 8].cpu().tolist()
                rec["rx_err"] = rx.err.cpu().tolist()
                hsd = getattr(eng.hooks, "hsd", None)
                if hsd is not None:
                    g = r * (hsd.max_seqs + 1) + slot
                    rec["hsd_by_slot"] = hsd.by_slot[r][g].cpu().tolist()
                eg = getattr(eng.hooks, "engram", None)
                if eg is not None:
                    rec["engram_ring"] = [tuple(x) for x in eg[r].ring[slot].cpu().tolist() if x[0] >= 0]
                    rec["engram_err"] = eg[r].err.cpu().tolist()
            except Exception as e:  # noqa: BLE001
                rec["read_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        out.append(rec)
        w["left"] -= 1
        if w["left"] <= 0:
            _WATCH.pop(key, None)
    rows = {f"r{sb.rank}": (sb.n_rows, len(sb.seqs)) for sb in ps.batches}
    for rid in sorted(dropped, key=str):
        last = next(((d, e[rid]) for d, e in reversed(_HIST[:-1]) if rid in e), None)
        rec = dict(dispatch=_N["i"], step=int(ps.step), dropped=str(rid), fault=fault, rows=rows)
        if last is not None:
            d0, e0 = last
            r, slot, uid, inc, kind, start, known, nb, nbt, wmax = e0
            rec.update(last_dispatch=d0, rank=r, slot=slot, uid=uid, inc=inc, kind=kind, known=known, n_blocks=nb,
                       bt_len=nbt, paged_cover=nb * 64, window_cover=32 * (wmax + 1))
            if fault is None:
                try:
                    rx = d13.rx[r]
                    nc = int(rx.nc[slot].item())
                    rec["nc"] = nc
                    rec["nc_past_paged"] = nc + 4 > nb * 64        # the in-flight rows ran past the block table
                    rec["nc_past_window"] = nc + 4 > 32 * (wmax + 1)
                    rec["dbt_tail"] = rx.dbt[slot, max(0, nb - 1): nb + 3].cpu().tolist()
                    rec["rx_err"] = rx.err.cpu().tolist()
                except Exception as e:  # noqa: BLE001
                    rec["read_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        out.append(rec)
    if fault is None:
        try:
            out.append(index_check(eng, ps))
        except Exception as e:  # noqa: BLE001
            out.append(dict(index_check_error=f"{type(e).__name__}: {str(e)[:200]}"))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            for rec in out:
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps(dict(dispatch=_N["i"], step=int(ps.step), watched=len(out), fault=fault, rows=rows,
                                    drops=len(dropped),
                                    dump_ms=round((time.time() - t0) * 1e3, 1))) + "\n")
    except Exception:  # noqa: BLE001
        pass
    if fault is not None:
        print("phys adapter: " + _fault_memhist(path), flush=True)
        raise SlotDumpFault(f"phys adapter: CUDA fault surfaced at slot-dump sync after dispatch {_N['i']} "
                           f"(step {ps.step}): {fault}")


def index_check(eng, ps) -> dict:
    """The write / read indices this dispatch actually used (the bucket state of the last dispatch, after rowexp, the
    guard and the indexer's device tables), every row of every rank, against the pools they index: positions, own
    window slots, window read slots (within sl), paged slots, the indexer's compressor ring slots and compressed slots.
    Out-of-range values (a wild write: Xid 31 FAULT_PDE VIRT_WRITE) with the first offenders' row / sequence / rid."""
    import torch

    st = getattr(eng, "last_st", None)
    rec = dict(index_check=True, step=int(ps.step))
    if st is None:
        rec["note"] = "no last_st"
        return rec
    for r in range(len(st)):
        s = st[r]
        sb = ps.batches[r] if r < len(ps.batches) else None
        n = sb.n_rows if sb is not None else 0
        if not n:
            continue
        pool = eng.pools[r]
        lim = dict(own=(pool.n_window + 1) * 32, si=(pool.n_window + 1) * 32, paged=pool.n_paged * 64,
                   cslot1=pool.n_paged * 64, cslot2=pool.n_paged * 32, rslot=pool.n_state * 8, pos=655_360 + 64)
        R = int(s["R"])
        vals = {}
        with torch.cuda.device(s["dev"]):
            vals["pos"] = s["pos"][:n].long().cpu()
            vals["own"] = s["own"][:n].long().cpu()
            if "paged_d" in s:
                vals["paged"] = s["paged_d"][:n].long().cpu()
            sl = s["sl"][:n].long().cpu()
            si = s["si"].view(R, -1)[:n].long().cpu()
            d = s.get("idx") or {}
            if "rslot" in d:
                vals["rslot"] = d["rslot"][:n].long().cpu()
            if "cslot" in d:
                vals["cslot1"] = d["cslot"][1].view(-1)[:n].long().cpu()
                vals["cslot2"] = d["cslot"][2].view(-1)[:n].long().cpu()
        rr = dict(n=n, R=R)
        for k, v in vals.items():
            lo_ok = -1 if k in ("paged", "rslot", "cslot1", "cslot2") else 0
            bad = ((v < lo_ok) | (v >= lim[k])).nonzero().flatten()
            rr[k] = (int(v.min()), int(v.max()), lim[k], int(bad.numel()))
            if bad.numel():
                i = int(bad[0])
                se = sb.seqs[sb.seq_slot[i]] if i < len(sb.seq_slot) else None
                rr[k + "_first_bad"] = (i, int(v[i]), str(getattr(se, "rid", "?")), getattr(se, "kind", "?"),
                                        int(getattr(se, "state_slot", -1)), int(getattr(se, "known", -1)))
        inlen = torch.arange(si.shape[1])[None, :] < sl[:, None]
        sbad = (inlen & ((si < 0) | (si >= lim["si"]))).any(1).nonzero().flatten()
        rr["si_bad_rows"] = int(sbad.numel())
        if sbad.numel():
            i = int(sbad[0])
            row = si[i][: int(sl[i])]
            rr["si_first_bad"] = (i, int(row.min()), int(row.max()), int(sl[i]))
        rec[f"r{r}"] = rr
    return rec


def _shape(eng, ps, path) -> None:
    """Every dispatch, host side only (no sync): per-rank rows / sequences / sample rows / tail rows, the bucket R and
    bkey each rank's state carries, HSD's key (S_pad, D_pad, NT): rank-different shapes in a faulting step. One compact
    JSON line ("shape")."""
    st = getattr(eng, "last_st", None)
    rk = []
    for r, sb in enumerate(ps.batches):
        s = st[r] if st is not None and r < len(st) else {}
        rk.append([sb.n_rows, len(sb.seqs), len(sb.sample_rows), len(sb.tail_rows), s.get("R"), s.get("bkey"),
                   sum(1 for se in sb.seqs if se.kind != "decode")])
    hsd = getattr(getattr(eng, "hooks", None), "hsd", None)
    cur = getattr(hsd, "cur", None)
    key = None if cur is None else [getattr(cur, "S_pad", None), getattr(cur, "D_pad", None), getattr(cur, "NT", None)]
    rec = dict(shape=_N["i"], step=int(ps.step), ranks=rk, hsd_key=key, new=len(ps.new_reqs),
               drop=len(ps.finished) + len(ps.preempted), ws=_ws_state(), res_mb=_reserved_mb(eng),
               kg=debug_patches.kept_graph_stats())
    try:
        fd = os.open(path + ".shape", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ phase syncs
_SUB = {"on": False, "step": -1}


def install_subsync(eng, d13, path) -> None:
    """PHYS_D13_SUBSYNC=1 (debug): wrap the D13 dispatch's phases on THIS process's engine objects (no engine file is
    changed) so every device is synchronized after each one: D13._pre, Engine.replay (graph L, then graph H),
    Engine._prepare_head_on_streams (HSD's prepare), hooks.head_eager, D13._advance. A CUDA fault then surfaces at the
    phase whose work faulted; one JSON line {"subsync_fault": phase, step} goes to <path>.sub. No overlap: slower."""
    if _SUB["on"]:
        return
    import functools

    import torch

    def sync_after(obj, name, label, post=None):
        fn = getattr(obj, name)

        @functools.wraps(fn)
        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            if post is not None:
                try:
                    for dv in eng.devs:
                        torch.cuda.synchronize(dv)
                    post(a)
                except Exception:  # noqa: BLE001  (a fault surfaces at the sync below)
                    pass
            try:
                for dv in eng.devs:
                    torch.cuda.synchronize(dv)
            except Exception as e:  # noqa: BLE001
                lab = label(a) if callable(label) else label
                rec = dict(subsync_fault=lab, step=_SUB["step"], dispatch=_N["i"] + 1,
                           error=f"{type(e).__name__}: {str(e).splitlines()[0][:160]}", memhist=_fault_memhist(path))
                try:
                    with open(path + ".sub", "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:  # noqa: BLE001
                    pass
                raise SlotDumpFault(f"phys adapter: CUDA fault in D13 phase {lab} (step {_SUB['step']}): {rec['error']}")
            return out
        setattr(obj, name, wrapped)

    sync_after(d13, "_pre", "pre", post=lambda a: _precheck(eng, path, a))
    sync_after(eng, "replay", lambda a: f"replay {a[0] if a else '?'}")
    sync_after(eng, "_prepare_head_on_streams", "hsd_prepare")
    sync_after(eng.hooks, "head_eager", "head_eager")
    sync_after(d13, "_advance", "advance")
    _SUB["on"] = True


_PRE = {"n": 0, "after_fb": 0}


def _precheck(eng, path, a) -> None:
    """Right after D13._pre (synchronized): the indices graph L is about to use (positions, own window slots, window
    read slots, paged slots, the indexer's ring / compressed slots of the bucket state _pre just filled), every rank,
    against the pools (index_check). Written when anything is out of range, and for the first 3 async dispatches after
    a fallback (non-decode) dispatch, so the faulting replay's inputs are on record."""
    st, batches = a[0], a[1]
    _PRE["n"] += 1

    class _PS:
        pass
    ps = _PS()
    ps.batches, ps.step = batches, (batches[0].step if batches else -1)
    saved = getattr(eng, "last_st", None)
    try:
        eng.last_st = st
        rec = index_check(eng, ps)
    finally:
        eng.last_st = saved
    nbad = sum(v[3] for rr in rec.values() if isinstance(rr, dict) for k, v in rr.items()
               if isinstance(v, tuple) and len(v) == 4 and not k.endswith("_first_bad")) + \
        sum(rr.get("si_bad_rows", 0) for rr in rec.values() if isinstance(rr, dict))
    if nbad or _PRE["after_fb"] > 0:
        _PRE["after_fb"] = max(0, _PRE["after_fb"] - 1)
        rec.update(precheck=_PRE["n"], dispatch=_N["i"] + 1, bad=nbad)
        fd = os.open(path + ".pre", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _reserved_mb(eng):
    """torch's reserved bytes per device in MiB (host side): a drop between dispatches = segments returned to the driver."""
    try:
        import torch
        return [int(torch.cuda.memory_reserved(d) >> 20) for d in eng.devs]
    except Exception:  # noqa: BLE001
        return None


def _fault_memhist(path) -> str:
    return debug_patches.dump_memhist(path + ".memsnap_fault.pickle")


def _ws_state():
    """vLLM's WorkspaceManager(s), host side (no sync): [device, [[data_ptr, numel] per ubatch / lane slot]] for the
    global manager (or the per-device ones under PHYS_WS_PER_DEVICE): a growth = a new pointer / size."""
    try:
        import vllm.v1.worker.workspace as W
    except Exception:  # noqa: BLE001
        return None
    mgrs = dict(getattr(W, "_per_device_managers", {}) or {})
    if getattr(W, "_manager", None) is not None:
        mgrs["global"] = W._manager
    out = []
    for k, m in sorted(mgrs.items(), key=lambda x: str(x[0])):
        cur = getattr(m, "_current_workspaces", None) or []
        out.append([str(getattr(m, "_device", k)), [[t.data_ptr(), t.numel()] if t is not None else None for t in cur]])
    return out
