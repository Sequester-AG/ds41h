#!/usr/bin/env python3
"""moe_v4 drop-in: the fork's routed-MoE GEMM kernels with the moe_v4 options compiled in.

Buffer sizes are unchanged (pair lists are shorter). EC records, slot map
image / H layouts and Y are byte-identical to moe_raw + route2, so prepr*, combine and phys_moe's buffers work as before.
Not supported with these lists: route2x / gemm1r8 / gemm2r8 (the n_pad-8 row-stream kernels).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"]

# the measured best configuration; MS4_HQ / MS4_YWARP default to 1 in moe_v4.cu
DEFAULT = {"MS4_G2P": "1", "MS3_S2": "2", "MS4_SPOL": "1"}
ROUTE_KEYS = {"MS4_G1P": "R2_G1PAIR", "MS4_G2P": "R2_G2PAIR"}


def load(defs=None, build_root=None):
    from torch.utils.cpp_extension import load as _load
    d = dict(DEFAULT if defs is None else defs)
    if defs is None and os.environ.get("MOE_V4_DEFS"):   # MOE_V4_DEFS=MS3_S=4,MS4_PDL=1
        d.update(kv.split("=", 1) for kv in os.environ["MOE_V4_DEFS"].split(",") if "=" in kv)
    root = build_root or os.environ.get("MOE_V4_BUILD", os.path.join(HERE, "build"))
    tag = "".join(f"_{k.lower()}{v}" for k, v in sorted(d.items()))
    # MOE_V4_SRC: another source file in this directory (e.g.
    # moe_v4_dqmix.cu); its name goes into the extension tag, so the served build is never overwritten
    src = os.environ.get("MOE_V4_SRC", "moe_v4.cu")
    if src != "moe_v4.cu":
        tag += "_src" + os.path.splitext(os.path.basename(src))[0].replace("moe_v4_", "")
    bd = os.path.join(root, "v4api" + tag)
    os.makedirs(bd, exist_ok=True)
    ext = _load(name="moe_v4_api" + tag, sources=[os.path.join(HERE, src)], build_directory=bd, verbose=False,
                extra_cuda_cflags=CFLAGS + [f"-D{k}={v}" for k, v in sorted(d.items())])
    rd = {ROUTE_KEYS[k]: v for k, v in d.items() if k in ROUTE_KEYS}
    rtag = "".join(f"_{k.lower()}{v}" for k, v in sorted(rd.items()))
    rbd = os.path.join(root, "v4route" + rtag)
    os.makedirs(rbd, exist_ok=True)
    r2 = _load(name="moe_v4_route" + rtag, sources=[os.path.join(HERE, "route2.cu")], build_directory=rbd, verbose=False,
               extra_cuda_cflags=CFLAGS + [f"-D{k}={v}" for k, v in sorted(rd.items())])
    return ext, r2
