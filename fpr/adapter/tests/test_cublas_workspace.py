"""Destroying a CUDA graph must not free the cuBLAS workspace that other graphs captured on the same stream still use.

Guards the ASYNC (D13) Xid 31 fault: a synchronous fallback step recaptured a graph D13 had already captured, so the
old torch.cuda.CUDAGraph was destroyed; in torch 2.13 that frees the cuBLAS workspace of the stream it was captured on
(32 MiB on the rank step stream, first allocated by an eager matmul), although every other graph captured on that
stream keeps writing into it. A later empty_cache() unmapped the block, and the next replay wrote to an unmapped page
(before the unmap, the writes land in memory the allocator may hand to other tensors).

The runtime stopgap (adapter/debug_patches.py, PHYS_KEEP_GRAPHS=1) keeps every CUDAGraph alive until exit; the engine
retires replaced graphs instead of destroying them. Each case runs in a child process: the stopgap patches torch
process-wide. 1 GPU, a few seconds.
  python -m pytest -q adapter/tests/test_cublas_workspace.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _child(keep: bool) -> dict:
    """In this process: eager mm on stream s, graphs g1 and g2 captured on s (one shared pool, as the engine), del g1,
    then: was a >= 1 MiB block freed, and does g2 still replay to the eager result (within bf16 rounding)?"""
    import gc

    import torch

    if keep:
        sys.path.insert(0, ROOT)
        os.environ["PHYS_KEEP_GRAPHS"] = "1"
        from adapter import debug_patches

        assert "keep_graphs" in debug_patches.install()
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=100_000)
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(512, 5120, device=dev, dtype=torch.bfloat16, generator=g)
    w = torch.randn(1792, 5120, device=dev, dtype=torch.bfloat16, generator=g)
    out = torch.empty(512, 1792, device=dev, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    torch.cuda.synchronize()

    def events():
        return torch.cuda.memory._snapshot()["device_traces"][0]

    with torch.cuda.stream(s):
        torch.mm(a, w.t(), out=out)                   # eager: allocates the (cuBLAS handle, s) workspace
    torch.cuda.synchronize()
    ref = out.clone()
    pool = torch.cuda.graph_pool_handle()
    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool, stream=s):
        torch.mm(a, w.t(), out=out)
    with torch.cuda.graph(g2, pool=pool, stream=s):
        torch.mm(a, w.t(), out=out)
    n0 = len(events())
    del g1
    gc.collect()
    torch.cuda.synchronize()
    freed = [round(int(e["size"]) / 2**20, 2) for e in events()[n0:]
             if e["action"] == "free_requested" and int(e.get("size", 0)) >= (1 << 20)]
    out.zero_()
    g2.replay()                                       # no empty_cache in between: a freed block is still mapped
    torch.cuda.synchronize()
    return dict(freed_mib=freed, replay_close=bool(torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)),
                replay_equal=bool(torch.equal(out, ref)), torch=torch.__version__)


def _run(keep: bool) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "keep" if keep else "plain"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.gpu
def test_keep_graphs_stopgap_keeps_workspace():
    r = _run(keep=True)
    assert r["freed_mib"] == [], f"a graph destroy freed {r['freed_mib']} MiB with PHYS_KEEP_GRAPHS=1 ({r['torch']})"
    assert r["replay_close"], r


@pytest.mark.gpu
def test_graph_destroy_hazard():
    """Without the stopgap: if this torch frees the shared workspace on a graph destroy, the freed block is a
    workspace-sized block that the surviving graph still uses. Skipped when torch no longer frees it (the stopgap and
    the engine's retire list are then belt and braces)."""
    r = _run(keep=False)
    if not r["freed_mib"]:
        pytest.skip(f"torch {r['torch']}: destroying a graph no longer frees the stream's cuBLAS workspace")
    assert all(x >= 1 for x in r["freed_mib"]), r
    assert r["replay_close"], r


if __name__ == "__main__":
    print(json.dumps(_child(sys.argv[1] == "keep")))
