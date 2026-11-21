"""p1k: build of the P1 pre-step kernels.

build() -> the extension (fpr_p1k; C ABI table fpr_p1_table for nd's pre-step / the C++ driver; pybind pf_wbt / pf_inv /
to_local / carry for p1k_test.py). Build directory: P1K_BUILD or <here>/build_p1k.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT: dict = {}


def build():
    if "m" not in _EXT:
        from torch.utils.cpp_extension import load

        bdir = os.environ.get("P1K_BUILD") or os.path.join(HERE, "build_p1k")
        os.makedirs(bdir, exist_ok=True)
        _EXT["m"] = load(name="fpr_p1k", sources=[os.path.join(HERE, "p1_kernels.cu")], extra_cflags=["-O3", "-std=c++17"],
                         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                         extra_include_paths=[HERE], build_directory=bdir, verbose=False)
    return _EXT["m"]
