#!/usr/bin/env python3
"""w8dq: build + call of w8dq.cu: exact bf16 [N, K] of a w8a16-packed fp8 weight
(w8a16.Gemm.packed), so mixed / prefill rows can run the bf16 copies' own cuBLAS call on a shared scratch (one resident fp8
copy, bitwise with the bf16 copies). Build root: W8DQ_BUILD (default <this dir>/build_w8dq).
W8DQ_IMPL=2: ext() returns w8dq2.cu's extension instead. Its build root: W8DQ2_BUILD."""
from __future__ import annotations

import functools
import os

HERE = os.path.dirname(os.path.abspath(__file__))


@functools.cache
def ext():
    if os.environ.get("W8DQ_IMPL", "1") == "2":
        import w8dq2
        return w8dq2.ext()
    from torch.utils.cpp_extension import load

    bdir = os.environ.get("W8DQ_BUILD", os.path.join(HERE, "build_w8dq"))
    os.makedirs(bdir, exist_ok=True)
    return load(name="w8dq_ext", sources=[os.path.join(HERE, "w8dq.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                build_directory=bdir, verbose=bool(os.environ.get("W8DQ_VERBOSE")))


def dequant_packed(packed, out):
    """packed uint8 [K / 128, N / 256, 32896] -> out bf16 [N, >= K] (written in place), returns out."""
    ext().dequant_packed(packed, out)
    return out
