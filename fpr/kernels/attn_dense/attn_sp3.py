#!/usr/bin/env python3
"""Build / call wrapper for attn_sp3.cu (v3: QK warpgroup + 2 P V warpgroups; ATTN2B_SPA=2 in attn_half).
SPA3_DEFS="SPA_NQR=20,SPA_PROF=1" adds -D overrides (separate build dir per option set)."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = {}


def build(defs: str | None = None, verbose: bool = False):
    defs = os.environ.get("SPA3_DEFS", "") if defs is None else defs
    if defs not in _EXT:
        from torch.utils.cpp_extension import load

        dl = [d.strip() for d in defs.split(",") if d.strip()]
        suf = ("_" + "_".join(d.lower().replace("=", "") for d in dl)) if dl else ""
        bdir = os.path.join(HERE, "build_spa3" + suf)
        os.makedirs(bdir, exist_ok=True)
        flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"] + [f"-D{d}" for d in dl]
        _EXT[defs] = load(name="attn_sp3_ext" + suf, sources=[os.path.join(HERE, "attn_sp3.cu")], extra_cuda_cflags=flags,
                          extra_include_paths=[HERE], build_directory=bdir, verbose=verbose)
    return _EXT[defs]


def attn(ext, q, out, cache, sink, positions=None, cos_sin=None, prof=None):
    R = cache["R"]
    ext.attn_sp3(q.view(R, 64, 512), out.view(R, 64, 512), cache["sbuf"], cache["cbuf"], cache["stride"],
                 cache["si"].view(R, -1), cache["ti"].view(R, -1), cache["sl"], cache["tl"], sink, cache["scale"],
                 positions, cos_sin, prof, int(cache.get("ins", 4)))
    return out
