"""Does destroying one CUDA graph free the cuBLAS workspace other graphs captured on the same stream still use?
(the root cause of a D13 Xid 31 fault: the faulting addresses were the rank step stream's 32 MiB cuBLAS
workspace, allocated by an eager torch.mm, freed while graph L still used it, then unmapped by empty_cache.)

1 GPU, ~20 s, no fault is provoked (no empty_cache while a dangling graph exists):
  eager mm on stream s (allocates the (handle, s) workspace) -> capture g1 and g2 on s (shared pool, like the engine)
  -> del g1 + gc -> is the workspace block freed? -> next eager mm: a new workspace?
On the GPUs, from fpr/: python adapter/tests/cublas_ws_graph.py [--keep]
--keep: with adapter/debug_patches.py PHYS_KEEP_GRAPHS (every CUDAGraph kept alive): step 3 must show no free.
Result with torch 2.13.0+cu130: step 3 FREED the workspace g2 still uses; a graph on a fresh stream gets its own.
"""
import gc

import torch


def big_events(n0):
    tr = torch.cuda.memory._snapshot()["device_traces"][0]
    out = []
    for i, e in enumerate(tr[n0:], n0):
        if int(e.get("size", 0)) >= (1 << 20) and e["action"] in ("alloc", "free_requested", "segment_alloc", "segment_free"):
            fr = [f"{f['filename'].split('/')[-1]}:{f['line']}" for f in (e.get("frames") or [])[:3]]
            out.append((i, e["action"], hex(int(e["addr"])), round(int(e["size"]) / 2**20, 2), fr))
    return out, len(tr)


def main():
    import os
    import sys
    if "--keep" in sys.argv:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        os.environ["PHYS_KEEP_GRAPHS"] = "1"
        from adapter import debug_patches
        print("patches:", debug_patches.install())
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=200_000)
    dev = torch.device("cuda", 0)
    bf = torch.bfloat16
    a = torch.randn(512, 5120, device=dev, dtype=bf)
    w = torch.randn(1792, 5120, device=dev, dtype=bf)
    out = torch.empty(512, 1792, device=dev, dtype=bf)
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    _, n0 = big_events(0)
    with torch.cuda.stream(s):
        torch.mm(a, w.t(), out=out)                     # eager: the (handle, s) workspace
    torch.cuda.synchronize()
    ev, n1 = big_events(n0)
    print("1. eager mm on s:", ev)
    pool = torch.cuda.graph_pool_handle()
    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool, stream=s):
        torch.mm(a, w.t(), out=out)
    with torch.cuda.graph(g2, pool=pool, stream=s):
        torch.mm(a, w.t(), out=out)
    ev, n2 = big_events(n1)
    print("2. capture g1, g2 on s:", ev)
    del g1
    gc.collect()
    ev, n3 = big_events(n2)
    print("3. del g1 + gc:", ev, "-> FREED the workspace" if any(x[1] == "free_requested" for x in ev) else "-> no free")
    with torch.cuda.stream(s):
        torch.mm(a, w.t(), out=out)
    torch.cuda.synchronize()
    ev, n4 = big_events(n3)
    print("4. eager mm on s again:", ev)
    g2.replay()
    torch.cuda.synchronize()
    print("5. g2 replay ok (no empty_cache in between, so a freed workspace is still mapped)")
    # the variant the engine could use: capture on a dedicated stream per evictable group
    s2 = torch.cuda.Stream()
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3, pool=pool, stream=s2):
        torch.mm(a, w.t(), out=out)
    ev, n5 = big_events(n4)
    print("6. capture g3 on a fresh stream s2:", ev)
    del g3
    gc.collect()
    ev, n6 = big_events(n5)
    print("7. del g3 + gc:", ev)
    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
