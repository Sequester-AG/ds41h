"""Opt-in runtime patches for diagnosis, applied in THIS process only (no engine or vLLM file changes).

PHYS_WS_PER_DEVICE=1  one vLLM WorkspaceManager per GPU (A/B for the D13 fault).
  vLLM keeps ONE global WorkspaceManager bound to one device (vllm/v1/worker/workspace.py). idx2f's _workspace(dev)
  creates it for the first device it sees and only marks the others, so every rank's stock top-k
  (indexer_topk.py: cooperative_topk / persistent_topk scratch = current_workspace_manager().get_simultaneous(1 MiB))
  gets the SAME 1 MiB buffer on one GPU: four GPUs' kernels share one scratch over peer access. This patch gives each
  CUDA device its own manager (keyed by the current device), for the A/B of whether that sharing is the D13 fault.
  Install before the engine is built (graphs capture views of the workspace).
  Result: still faults at the replay of graph (32, 'LA'); the stock top-k never asked for the workspace in that run.

PHYS_EC_LOG=<file>    one JSON line per torch.cuda.empty_cache() call: caller frames, the dispatch counter, reserved bytes
                      per device before / after (a drop = segments returned to the driver = unmapped).
PHYS_NO_EMPTY_CACHE=1 torch.cuda.empty_cache() does nothing (still logged): A/B of whether the D13 fault needs the
                      allocator to unmap freed segments (Xid 31 FAULT_PDE = a write to an unmapped page).
PHYS_KEEP_GRAPHS=1    (the D13 Xid 31 fix, runtime side) every torch.cuda.CUDAGraph made in this
                      process is kept alive until exit. torch 2.13's ~CUDAGraph frees the cuBLAS workspace of the stream
                      the graph was captured on, although every other graph captured on that stream still writes into it
                      (reproduced on 1 GPU). The engine overwrites graph H when a sync fallback
                      step recaptures an HSD key D13 already captured (engine.py _capture_split, self.graphs[hkey] = gs):
                      the step stream's 32 MiB workspace was freed, a mixed-bucket eviction's empty_cache unmapped it, and
                      graph L (32, 'LA') wrote there (allocator history of a faulting run). Kept graphs
                      cost their exec objects only (all engine graphs share each rank's pool). The engine-side fix (never
                      destroy a graph captured on a rank stream; reuse an existing graph H) belongs in engine/engine.py.
PHYS_MEMHIST=<n>      record the CUDA caching allocator's history (alloc / free / segment events, Python stacks, last n
                      entries, all devices) from engine start; dump_memhist(path) writes torch's snapshot pickle (slotdump
                      does at a D13 fault: <slotdump>.memsnap.pickle). Match the Xid fault addresses to segment_free /
                      free events in it.
"""
from __future__ import annotations

import os

_DONE = {"ws": False}


def install() -> list:
    out = []
    if os.environ.get("PHYS_WS_PER_DEVICE") == "1" and not _DONE["ws"]:
        _ws_per_device()
        _DONE["ws"] = True
        out.append("ws_per_device")
    if os.environ.get("PHYS_KEEP_GRAPHS") == "1" and not _DONE.get("kg"):
        _keep_graphs()
        _DONE["kg"] = True
        out.append("keep_graphs")
    if os.environ.get("PHYS_MEMHIST", "0") not in ("", "0") and not _DONE.get("mh"):
        import torch
        n = int(os.environ["PHYS_MEMHIST"]) if os.environ["PHYS_MEMHIST"].isdigit() and int(os.environ["PHYS_MEMHIST"]) > 1 \
            else 3_000_000
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=n)
            _DONE["mh"] = True
            out.append(f"memhist({n})")
        except Exception as e:  # noqa: BLE001
            out.append(f"memhist unavailable ({type(e).__name__}: {str(e)[:80]})")
    if (os.environ.get("PHYS_EC_LOG") or os.environ.get("PHYS_NO_EMPTY_CACHE") == "1") and not _DONE.get("ec"):
        # PHYS_EC_LOG + PHYS_MEMHIST: the allocator history is also dumped right after each empty_cache that returned
        # memory (<PHYS_EC_LOG>.memsnap_<n>.pickle; the first 3 and a rolling last one): the unmapped segments are in it
        # even if the CUDA context cannot be read after the fault
        _empty_cache_hook(os.environ.get("PHYS_EC_LOG"), os.environ.get("PHYS_NO_EMPTY_CACHE") == "1")
        _DONE["ec"] = True
        out.append("empty_cache_" + ("noop" if os.environ.get("PHYS_NO_EMPTY_CACHE") == "1" else "log"))
    return out


STATE = {"dispatch": -1, "ec_calls": 0}     # slotdump sets "dispatch"


KEPT_GRAPHS: list = []


def _keep_graphs() -> None:
    import torch
    import torch.cuda.graphs as G

    base = G.CUDAGraph
    if getattr(base, "_phys_kept", False):
        return

    class CUDAGraph(base):              # same name: torch.cuda.graph() and isinstance checks see a CUDAGraph
        _phys_kept = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            KEPT_GRAPHS.append(self)    # never destroyed before exit: ~CUDAGraph would free its stream's cuBLAS workspace

    CUDAGraph.__module__ = base.__module__
    G.CUDAGraph = CUDAGraph
    torch.cuda.CUDAGraph = CUDAGraph


def kept_graph_stats():
    """[graphs kept, of them only alive through the keep list (= the ones the engine dropped)], or None (patch off)."""
    if not _DONE.get("kg"):
        return None
    import sys
    return [len(KEPT_GRAPHS), sum(1 for g in KEPT_GRAPHS if sys.getrefcount(g) <= 3)]


def _empty_cache_hook(path, noop: bool) -> None:
    import json
    import time
    import traceback

    import torch

    orig = torch.cuda.empty_cache

    def reserved():
        try:
            return [int(torch.cuda.memory_reserved(d)) for d in range(torch.cuda.device_count())]
        except Exception:  # noqa: BLE001
            return None

    def empty_cache():
        STATE["ec_calls"] += 1
        fr = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_stack(limit=6)[:-1]]
        r0 = reserved()
        if not noop:
            orig()
        r1 = reserved()
        dumped = None
        if path and _DONE.get("mh") and r0 and r1 and any(a > b for a, b in zip(r0, r1)):
            STATE["ec_drops"] = STATE.get("ec_drops", 0) + 1
            k = STATE["ec_drops"]
            dumped = dump_memhist(f"{path}.memsnap_{k if k <= 3 else 'last'}.pickle")
        if path:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as f:
                    f.write(json.dumps(dict(t=round(time.time(), 3), n=STATE["ec_calls"], dispatch=STATE["dispatch"],
                                            noop=noop, caller=fr, reserved_before=r0, reserved_after=r1,
                                            memhist=dumped)) + "\n")
            except Exception:  # noqa: BLE001
                pass

    torch.cuda.empty_cache = empty_cache
    torch.cuda.memory.empty_cache = empty_cache


def dump_memhist(path) -> str:
    """Write the allocator history (PHYS_MEMHIST) to path; returns a short status (never raises)."""
    if not _DONE.get("mh"):
        return "memhist off"
    try:
        import torch
        torch.cuda.memory._dump_snapshot(path)
        return f"memhist -> {path} ({os.path.getsize(path)} bytes)"
    except Exception as e:  # noqa: BLE001
        return f"memhist dump failed: {type(e).__name__}: {str(e)[:160]}"


def _ws_per_device() -> None:
    from contextlib import nullcontext

    import torch
    import vllm.model_executor.layers.indexer_topk as IT
    import vllm.v1.worker.workspace as W

    mgrs: dict = {}

    def current():
        d = torch.cuda.current_device()
        m = mgrs.get(d)
        if m is None:
            m = mgrs[d] = W.WorkspaceManager(torch.device("cuda", d), None, 1, nullcontext)
        return m

    def init(device=None, *a, **k):          # idx2f's _workspace(dev): nothing to bind globally any more
        return None

    W.current_workspace_manager = current
    W.is_workspace_manager_initialized = lambda: True
    W.init_workspace_manager = init
    W._manager = None
    IT.current_workspace_manager = current    # bound by name at import in indexer_topk
    W._per_device_managers = mgrs             # (for inspection)
