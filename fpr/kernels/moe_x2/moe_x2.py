#!/usr/bin/env python3
"""moe_x2: JIT loader for the C128 routed-expert GEMMs on the pre-placed FP4 layout (moe_x2.cu), standalone prototype.

    from moe_x2 import moe_x2
    x2 = moe_x2.load({"X2_W8": "1"})                     # one build per define set, under $MOE_X2_BUILD or ./build
    x2.preplace(w, 0)                                    # in place: raw checkpoint expert bytes -> pre-placed (1: back)
    x2.gather_w8(x, ecs, counts, img8, asc, 0)           # W4A8 token image (X2_W8 builds)
    x2.gemm1(w13p, s13, ecs, counts, img, himg, ctr, asc, 0)   # img: gather6's f16 image, or img8 with X2_W8=1
    x2.gemm2(w2p, s2, ecs, counts, himg, Y, ctr, 0)

Defines (top of moe_x2.cu): X2_PRE (default 1), X2_W8 (0), X2_ASC (1), X2_PS (2), and moe_x's MX_* (MX_NWG1, MX_NWG2,
MX_MMA2, MX_PDL, ablations MX_NOSTREAM / MX_NOW / MX_NOMMA).

    python fpr/kernels/moe_x2/moe_x2.py build "X2_W8=1" "X2_PRE=0" ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"]


def parse(spec):
    """'A=1,B=2' -> {'A': '1', 'B': '2'}"""
    return dict(kv.split("=", 1) for kv in (spec or "").split(",") if "=" in kv)


def load(defs=None, build_root=None, verbose=False):
    from torch.utils.cpp_extension import load as _load
    d = dict(defs or {})
    root = build_root or os.environ.get("MOE_X2_BUILD", os.path.join(HERE, "build"))
    tag = "".join(f"_{k.lower().replace('mx_', '').replace('x2_', '')}{v}" for k, v in sorted(d.items()))
    bd = os.path.join(root, "x2" + tag)
    os.makedirs(bd, exist_ok=True)
    return _load(name="moe_x2" + tag, sources=[os.path.join(HERE, "moe_x2.cu")], build_directory=bd, verbose=verbose,
                 extra_cuda_cflags=CFLAGS + [f"-D{k}={v}" for k, v in sorted(d.items())])


def load_pcsamp(build_root=None):
    """CUPTI PC sampling extension (pcsamp.cpp): start(period) / stop(top) around kernel launches in this context"""
    from torch.utils.cpp_extension import load as _load, CUDA_HOME
    root = build_root or os.environ.get("MOE_X2_BUILD", os.path.join(HERE, "build"))
    bd = os.path.join(root, "pcsamp")
    os.makedirs(bd, exist_ok=True)
    return _load(name="moe_x2_pcsamp", sources=[os.path.join(HERE, "pcsamp.cpp")], build_directory=bd, verbose=False,
                 extra_include_paths=[os.path.join(CUDA_HOME, "include")],
                 extra_ldflags=[f"-L{os.path.join(CUDA_HOME, 'lib')}", "-lcupti", "-lcuda"])


if __name__ == "__main__":
    for spec in (sys.argv[2:] or [""]):
        ext = load(parse(spec), verbose=bool(os.environ.get("MX_VERBOSE")))
        print("built", repr(spec), "config", list(ext.config()), flush=True)
