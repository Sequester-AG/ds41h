#!/usr/bin/env python3
"""w8dq2: build + call of w8dq2.cu: w8dq.cu's exact bf16 [N, K] of a w8a16-packed fp8
weight as a persistent kernel with a bulk-copy ring of W8DQ2_NS stages. Same API as w8dq.py. Build root
W8DQ2_BUILD (default <this dir>/build_w8dq2)."""
from __future__ import annotations

import functools
import os

HERE = os.path.dirname(os.path.abspath(__file__))


@functools.cache
def ext(ns: int | None = None):
    from torch.utils.cpp_extension import load

    ns = int(ns or os.environ.get("W8DQ2_NS", "4"))
    assert ns in (2, 3, 4), ns
    bdir = os.path.join(os.environ.get("W8DQ2_BUILD", os.path.join(HERE, "build_w8dq2")), f"ns{ns}")
    os.makedirs(bdir, exist_ok=True)
    return load(name=f"w8dq2_ext_ns{ns}", sources=[os.path.join(HERE, "w8dq2.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   f"-DW8DQ2_NS={ns}"],
                build_directory=bdir, verbose=bool(os.environ.get("W8DQ2_VERBOSE")))


def dequant_packed(packed, out, grid: int = 0, ns: int | None = None):
    """packed uint8 [K / 128, N / 256, 32896] -> out bf16 [N, >= K] (written in place), returns out."""
    ext(ns).dequant_packed(packed, out, grid)
    return out
