"""c1m_precapture: the C1 mega graph L of every owner key captured
BEFORE serving, so native steps (nd) never meet an unpublished "R|LA|M|<owner>" and fall back to Python on a C1 step.

CAPTURE ONLY. Nothing executes: no synthetic row, no dummy sequence. No KV, Engram (ring / gathers), HSD or D13 state is
touched. (A dummy C1 row would need a sequence slot, KV pages and an Engram state slot of its own. c1mega's A4 insert and the
index layers' compressor / key writes go wherever the bucket's tables point.) A capture only records launches on pointers:
the R=4 bucket state's persistent buffers / tables, which every replay of that bucket reads anyway.

What the D13 first step (d13.D13._first) does for a new mega key, minus its eager step:
  1. tag the bucket state with the owner (Engine._tag_mega: the capture's layer() runs C1Mega.run on it);
  2. the adapter's launch blobs EAGERLY, outside the capture (C1Mega.run builds them on its first call, with .to(dev) H2D
     copies that a stream capture refuses), keyed exactly as run() keys them;
  3. Engine._capture_split(R, st, True, None, lkey=(R, kind, "M", owner) + _ab_key()). It captures graph L (p_embed with
     e0_prepare on estream, the c1mega launch(es) with their Engram event waits and aux means, p_final, the joins) on every
     rank, stores and keeps it, and syncs. Graph H is the bucket's (hkey None: unchanged).
The first replay of such a graph runs exactly what a graph captured after an eager step runs: same code path, same tensors.

Preconditions (else skipped / logged, never raised):
  - ENG_C1_MEGA on and the active adapter built (Engine.c1m());
  - bucket R=4 created and populated by at least one decode step: the blobs bake its tables (caches[L] ti / tl, s["idx"],
    pos / own / si / sl, moe_out ...). The D13 path keeps those buffers across steps; c1mega_eng's pack cache relies on it;
  - an idle point: no step in flight on any GPU and no other thread issuing CUDA work (torch's graph capture is in global
    error mode). ENG_DECODE_PRECREATE's done point qualifies.
A bucket state that is later evicted / re-created drops its graphs as today: the next C1 step then takes the Python first
step again.

  from c1m_precapture import precapture
  res = precapture(eng)   # {"captured": [keys], "skipped": [(key, why)], "failed": [(key, error)], "s": seconds}
"""
from __future__ import annotations

import sys
import time

import torch


def precapture(eng, owners=None, kinds=("LA",), log=print) -> dict:
    """Capture graph L of the C1 mega keys (R, kind, "M", owner) + Engine._ab_key() for every owner in `owners` (default:
    all ranks) and every kind (D13: "LA"; the synchronous path's "L"). Keys already in eng.graphs are skipped.
    Never raises: failures are logged and returned. A failed owner keeps today's behaviour (the Python first step)."""
    t0 = time.perf_counter()
    res = dict(captured=[], skipped=[], failed=[], s=0.0)
    E_ = sys.modules.get(type(eng).__module__)
    if not getattr(E_, "C1_MEGA", False):
        res["skipped"].append((None, "ENG_C1_MEGA off"))
        return res
    try:
        cm = eng.c1m()
    except Exception as e:  # noqa: BLE001
        res["failed"].append((None, f"adapter: {type(e).__name__}: {e}"))
        log(f"c1m_precapture: no adapter ({type(e).__name__}: {e})")
        return res
    R = int(getattr(sys.modules.get(type(cm).__module__), "NRM", 4))   # the C1 bucket (owner_of: R == NRM)
    if R not in eng.bk:
        res["skipped"].append((None, f"bucket {R} not created yet"))
        log(f"c1m_precapture: bucket {R} not created yet: nothing captured")
        return res
    st = eng.bucket(R)
    W = len(st)
    mem0 = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
    for o in (range(W) if owners is None else owners):
        for kind in kinds:
            lk = (R, kind, "M", int(o)) + tuple(eng._ab_key())
            if lk in eng.graphs:
                res["skipped"].append((lk, "already captured"))
                continue
            try:
                eng._tag_mega(st, int(o))
                cm._weights()
                cm._buffers()
                for k in range(len(cm.runs)):   # C1Mega.run's pack key (s["R"] is the bucket's R on every rank)
                    key = (st[0]["R"], int(o), k, st[int(o)]["moe_out"].data_ptr())
                    if key not in cm._packs:
                        cm._packs[key] = cm._pack(st, int(o), k)
                for d in eng.devs:
                    torch.cuda.synchronize(d)
                eng._capture_split(R, st, True, None, lkey=lk)
                res["captured"].append(lk)
            except Exception as e:  # noqa: BLE001
                res["failed"].append((lk, f"{type(e).__name__}: {e}"))
                log(f"c1m_precapture: {lk} FAILED ({type(e).__name__}: {e}); its first C1 step stays Python")
            finally:
                eng._tag_mega(st, None)
    res["s"] = time.perf_counter() - t0
    mem1 = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
    res["mem_mb"] = [((a1 - a0) / 2**20, (r1 - r0) / 2**20) for (a0, r0), (a1, r1) in zip(mem0, mem1)]   # (allocated, reserved)
    log(f"c1m_precapture: captured {len(res['captured'])} ({res['captured']}), skipped {len(res['skipped'])}, failed "
        f"{len(res['failed'])} in {res['s']:.1f} s; memory delta per GPU (allocated / reserved MiB): "
        + " ".join(f"{a:+.1f}/{r:+.1f}" for a, r in res["mem_mb"]))
    return res
