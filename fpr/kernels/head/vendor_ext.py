"""JIT builds of the fork kernels kernels/head reuses, from the snapshot in kernels/head/vendor/ (SOURCES.txt has the md5
of each file and the time it was taken from the live tree), under head_* extension names in kernels/head/build_*/.

  comm()   sp_layer/comm.cu      peer-store AG / RS (ag_push, ag_push2, rs_push, rs_sum, wait, enable_peers)
  mhc()    sp_layer/mhc.cu       fused mHC sublayer boundary (boundary, prep_fn)
  dense()  attn_dense/dense.cu   RMSNorm(+RoPE), RoPE(q) + fp8_ds_mla window-KV record insert, inverse RoPE (PDL off)
  moe()    moe_stream/moe_raw.cu routed MoE on the raw ds41h layout (route, prepr, gemm1r, gemm2r, combine, swiglu,
           and the n_pad-8 row-stream kernels gemm1r8 / gemm2r8)
  route2() moe_stream/route2.cu  multi-CTA routing tables (route2, route2x: n_pad <= 8 chunks into ec8)
"""
from __future__ import annotations

import os

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
VEN = os.path.join(HERE, "vendor")
_E: dict = {}
FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"]


def _load(name, src, extra=()):
    if name not in _E:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        bd = os.path.join(HERE, "build_" + name)
        os.makedirs(bd, exist_ok=True)
        _E[name] = load(name=name, sources=[os.path.join(VEN, src)], build_directory=bd, verbose=False,
                        extra_cuda_cflags=FLAGS + list(extra), extra_include_paths=[VEN])
    return _E[name]


def comm():
    return _load("head_comm", "comm.cu")


def mhc():
    return _load("head_mhc", "mhc.cu")


def mhc2():
    """Newer sp_layer/mhc.cu (vendor/mhc2.cu): any rows per call (K1a v2 in 64-row slabs, partials [160 x R x 25]),
    k1a_x / k1a_mix; same host API and prep_fn as mhc()."""
    return _load("head_mhc2", "mhc2.cu")


def dense():
    return _load("head_dense", "dense.cu", ["-DDENSE_PDL=0"])


def moe():
    return _load("head_moe_raw", "moe_raw.cu", ["--expt-relaxed-constexpr"])


def route2():
    return _load("head_route2", "route2.cu", ["--expt-relaxed-constexpr"])
