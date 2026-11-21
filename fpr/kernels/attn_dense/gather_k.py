"""gather_k: build of the DSv4 fp8 K-cache dequantize + gather.

build() -> the extension (gatherk_ext; C ABI table fpr_gatherk_table for the C++ mixed-step driver; pybind gather(out,
k_cache, seq_lens, gather_lens, block_table, block_size, offset) = vLLM's dequantize_and_gather_k_cache, bit for bit, for the
584-B fp8 record). Build directory: GATHERK_BUILD or <here>/build_gatherk.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT: dict = {}


def build():
    if "m" not in _EXT:
        from torch.utils.cpp_extension import load

        bdir = os.environ.get("GATHERK_BUILD") or os.path.join(HERE, "build_gatherk")
        os.makedirs(bdir, exist_ok=True)
        _EXT["m"] = load(name="gatherk_ext", sources=[os.path.join(HERE, "gather_k.cu")], extra_cflags=["-O3", "-std=c++17"],
                         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                         extra_include_paths=[HERE], build_directory=bdir, verbose=False)
    return _EXT["m"]
