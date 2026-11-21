"""Batched page copies between FPR pools: kernels/pagecopy/pagecopy.cu.

copy_pages(groups) issues ONE kernel per (pool, src rank, dst rank) group instead of one copy_ per contiguous run (serve's
Worker.copies: with scattered free lists nearly every run is a single page, ~60 us of host time each as an HtoH copy_).
Each group runs on the destination GPU's current stream; the page id lists go up through pinned staging (non_blocking).
Bitwise: a byte copy. The caller keeps its own ordering rules (Worker.copies synchronizes the touched devices after).
"""
from __future__ import annotations

import os
from functools import cache

HERE = os.path.dirname(os.path.abspath(__file__))


@cache
def ext():
    from torch.utils.cpp_extension import load

    bd = os.path.join(HERE, "build")
    os.makedirs(bd, exist_ok=True)
    return load(name="fpr_pagecopy", sources=[os.path.join(HERE, "pagecopy.cu")], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])


def copy_pages(src, dst, src_pages, dst_pages, page_bytes: int) -> None:
    """Copy whole pages src[src_pages[i]] -> dst[dst_pages[i]] (page_bytes each). src / dst: the pools' flat uint8 tensors
    (any devices with peer / host-mapped access from dst's GPU); the launch runs on dst's GPU, current stream."""
    import torch

    n = len(src_pages)
    assert n == len(dst_pages)
    if not n:
        return
    dev = dst.device
    with torch.cuda.device(dev):
        sp = torch.tensor(src_pages, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
        dp = torch.tensor(dst_pages, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
        ext().pagecopy(int(src.data_ptr()), int(dst.data_ptr()), sp, dp, int(page_bytes))
