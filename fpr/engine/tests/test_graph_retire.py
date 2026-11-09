"""A replaced CUDA graph is retired (kept alive), never destroyed while other graphs on its stream live.

Bug guarded: with the pipelined decode path, a synchronous fallback step could recapture a bucket's graph and the old
CUDAGraph object was dropped. In the PyTorch release this engine runs on, destroying a graph frees the cuBLAS
workspace of its capture stream even though other graphs captured on the same stream still write into it; after the
allocator returned that memory to the driver, the next replay of a surviving graph faulted (Xid 31, a write to an
unmapped address). The fix (Engine._store_graph and the mixed-bucket eviction) moves every replaced graph to
Engine._retired, where it lives until the whole run's graphs are dropped together.
"""
from __future__ import annotations

import gc
import types

import pytest


def _stub():
    return types.SimpleNamespace(graphs={}, _retired=[], stats={})


def test_store_graph_retires_the_replaced_graph():
    """CPU: storing a graph under an existing key keeps the old one referenced in _retired."""
    from engine import engine as E
    eng = _stub()
    g1, g2, g3 = object(), object(), object()
    E.Engine._store_graph(eng, (64, True), g1)
    assert eng.graphs[(64, True)] is g1 and eng._retired == []
    E.Engine._store_graph(eng, (64, True), g2)
    E.Engine._store_graph(eng, (64, True), g3)
    assert eng.graphs[(64, True)] is g3
    assert eng._retired[0] is g1 and eng._retired[1] is g2
    assert eng.stats["graphs_retired"] == 2


def _big_frees(n0: int) -> list:
    import torch
    tr = torch.cuda.memory._snapshot()["device_traces"][0]
    return [e for e in tr[n0:] if e["action"] in ("free_requested", "free_completed", "segment_free")
            and int(e.get("size", 0)) >= (1 << 20)]


@pytest.mark.gpu(n=1)
def test_replaced_graph_keeps_the_stream_workspace():
    """GPU: an eager GEMM on stream s, then graphs g1 and g2 captured on s (shared pool, as the engine does). g1 is
    replaced through Engine._store_graph and the last other reference is dropped: no workspace-sized block is freed,
    and g2 (and an eager GEMM after it) still give the eager result bit for bit."""
    import torch
    from engine import engine as E
    dev = torch.device("cuda", 0)
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=100_000)
    try:
        g = torch.Generator(device=dev).manual_seed(0)
        a = torch.randn(512, 5120, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(1792, 5120, device=dev, dtype=torch.bfloat16, generator=g)
        s = torch.cuda.Stream(device=dev)
        out, o1, o2 = (torch.empty(512, 1792, device=dev, dtype=torch.bfloat16) for _ in range(3))
        with torch.cuda.stream(s):
            torch.mm(a, w.t(), out=out)                   # creates the (cuBLAS handle, s) workspace
        torch.cuda.synchronize(dev)
        ref = out.clone()
        pool = torch.cuda.graph_pool_handle()
        g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1, pool=pool, stream=s):
            torch.mm(a, w.t(), out=o1)
        with torch.cuda.graph(g2, pool=pool, stream=s):
            torch.mm(a, w.t(), out=o2)
        torch.cuda.synchronize(dev)
        n0 = len(torch.cuda.memory._snapshot()["device_traces"][0])
        eng = _stub()
        E.Engine._store_graph(eng, "bucket", g1)
        E.Engine._store_graph(eng, "bucket", g2)          # g1 replaced: retired
        del g1
        gc.collect()
        torch.cuda.synchronize(dev)
        frees = _big_frees(n0)
        assert not frees, f"a workspace-sized block was freed after the replacement: {frees[:2]}"
        for _ in range(2):
            o2.zero_()
            eng.graphs["bucket"].replay()
            torch.cuda.synchronize(dev)
            assert torch.equal(o2, ref)
            with torch.cuda.stream(s):
                torch.mm(a, w.t(), out=out)
            torch.cuda.synchronize(dev)
            assert torch.equal(out, ref)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
