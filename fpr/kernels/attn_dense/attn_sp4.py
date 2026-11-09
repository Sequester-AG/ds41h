#!/usr/bin/env python3
"""Build / call wrapper for attn_sp4.cu (v4: v2 + a producer warpgroup for the record gather; ATTN2B_SPA=4 in
attn_half). SPA4_DEFS="SPA_NR=4,SPA_PROF=1" adds -D overrides (separate build dir per option set)."""
from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = {}


def build(defs: str | None = None, verbose: bool = False):
    defs = os.environ.get("SPA4_DEFS", "") if defs is None else defs
    key = defs
    if key not in _EXT:
        from torch.utils.cpp_extension import load

        dl = [d.strip() for d in defs.split(",") if d.strip()]
        suf = ("_" + "_".join(d.lower().replace("=", "") for d in dl)) if dl else ""
        bdir = os.path.join(HERE, "build_spa4" + suf)
        os.makedirs(bdir, exist_ok=True)
        flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"] + [f"-D{d}" for d in dl]
        _EXT[key] = load(name="attn_sp4_ext" + suf, sources=[os.path.join(HERE, "attn_sp4.cu")], extra_cuda_cflags=flags,
                         build_directory=bdir, verbose=verbose)
    return _EXT[key]


def attn(ext, q, out, cache, sink, positions=None, cos_sin=None, prof=None, prefetch=None, prefetch_bytes=0):
    """o [R, 1, 64, 512] for the rank cache of attn_half.build_rank_cache (inverse RoPE fused when cos_sin is given).
    prefetch: a tensor whose first prefetch_bytes the CTAs pull into L2 at kernel start (the next GEMM's weights)."""
    R = cache["R"]
    # tile 0 is gathered before the PDL wait unless one of its window entries may be a record the previous kernel
    # inserts (the last `ins` window entries of a row: one per row of its sequence; cache["ins"], default 4)
    ext.attn_sp(q.view(R, 64, 512), out.view(R, 64, 512), cache["sbuf"], cache["cbuf"], cache["stride"],
                cache["si"].view(R, -1), cache["ti"].view(R, -1), cache["sl"], cache["tl"], sink, cache["scale"],
                positions, cos_sin, prof, prefetch, int(prefetch_bytes), int(cache.get("ins", 4)))
    return out
