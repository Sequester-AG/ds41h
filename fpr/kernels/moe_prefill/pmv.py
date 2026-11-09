#!/usr/bin/env python3
"""pmv: production's MoE extension (private pm2 copy of production's ds41h sources, unchanged kernels) built with extra -D defines,
one extension per define set (mpf_pmv_<tag>, build/mpf_pmv_<tag>), and timed against production at prefill shapes.
Defines tried: moe_gemm16's W16_DEPTH (k16 wgmma groups in flight, production 1) and W16_BN (tokens per remainder tile).
  CUDA_VISIBLE_DEVICES= python3 pmv.py --build W16_DEPTH=3 [W16_BN=64 ...]
  python3 pmv.py --run W16_DEPTH=3                               # one GPU
AGGREGATE OUTPUT ONLY.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

try:                                   # engine: from moe_prefill import pmv (kernels/ on sys.path)
    from . import pm2, pm_fused as P
    from . import pmv_nt
except ImportError:                    # scripts in this directory
    import pm2
    import pm_fused as P
    import pmv_nt


# source patches of the pmv copy (build/pmv_src; never pm2_src, which the engine's mpf_pm2 build hashes), each gated by a
# -D define so a build without it is production's code:
#   PMV_W16_CHUNK: hybrid remainders of more than W16_BN rows run as ceil(r / BN) gemm16 tiles (production: one tile per
#   expert, so W16_BN < S would drop rows)
PATCHES = [
    ("""  const int r = n % a.tsplit_bn;
  return (r > 0 && r <= a.tsplit) ? 1 : 0;
}""", """  const int r = n % a.tsplit_bn;
#ifdef PMV_W16_CHUNK
  return (r > 0 && r <= a.tsplit) ? (r + w16::BN - 1) / w16::BN : 0;
#else
  return (r > 0 && r <= a.tsplit) ? 1 : 0;
#endif
}"""),
]
# (pmv_nt: the PMV_* source patches, gated by #ifdef)

PATCHES += pmv_nt.PATCHES
SRC3 = P.HERE / "build" / "pmv_src"


def sync_sources3():
    src2 = pm2.sync_sources2()
    SRC3.mkdir(parents=True, exist_ok=True)
    for f in sorted(src2.iterdir()):
        data = f.read_bytes()
        if f.name == "moe_prefill.cu":
            txt = data.decode()
            for o, n in PATCHES:
                assert txt.count(o) == 1, o[:60]
                txt = txt.replace(o, n)
            txt = pmv_nt.arv_patch(txt)
            data = txt.encode()
        P._write_if_changed(SRC3 / f.name, data)
    return SRC3


def tag(defs):
    return "_".join(d.replace("=", "").lower() for d in defs) or "base"


@functools.cache
def ext(defs: tuple):
    from torch.utils.cpp_extension import load

    cuext = P._load_file("mpf_pmv_cuext", P.DEPLOY / "cuext.py")
    src = sync_sources3() if any(d.startswith("PMV_") for d in defs) else pm2.sync_sources2()
    name = "mpf_pmv_" + tag(defs)
    bd = P.HERE / "build" / name
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(P.HERE / "pm2.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS) + ["-D" + d for d in defs],
                extra_include_paths=[str(src)], build_directory=str(bd), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="*")
    ap.add_argument("--run", nargs="*")
    ap.add_argument("--freeze1", nargs="*")
    ap.add_argument("--freeze2", nargs="*")
    ap.add_argument("--freeze3", nargs="*")
    ap.add_argument("--T", default="8192,16384")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    if a.freeze3 is not None:
        for z in a.freeze3:
            ext_frozen3(tuple(z.split(",")))
            print("built frozen3", z, flush=True)
        return
    if a.freeze2 is not None:
        for z in a.freeze2:
            ext_frozen2(tuple(z.split(",")))
            print("built frozen2", z, flush=True)
        return
    if a.freeze1 is not None:
        for z in a.freeze1:
            ext_frozen1(tuple(z.split(",")))
            print("built frozen1", z, flush=True)
        return
    sets = [tuple(z.split(",")) for z in (a.build if a.build is not None else a.run or [])]
    if a.build is not None:
        for s in sets:
            ext(s)
            print("built", tag(s), flush=True)
        return
    import torch

    import common as C
    import pm_test as PT
    from ds41h import moe as PM

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", 0)
    w13, s13, w2, s2 = C.load_weights(dev)
    inp = C.Inputs(dev)
    exts = {tag(s): ext(s) for s in sets}
    for e in exts.values():
        e.set_hybrid(PM.HYBRID, PM.HYBRID_ENGINE)
    for T in [int(z) for z in a.T.split(",")]:
        x, w, ids = inp.get(T, "cap")
        sh = (torch.randn(T, 5120, device=dev) * 0.1).to(torch.bfloat16)
        prod = lambda: PM.moe_forward_cuda(x, w, ids, w13, s13, w2, s2, 10.0, 0xB9, addend=sh)  # noqa: E731
        base = prod()
        torch.cuda.synchronize()
        bus = sorted(PT.gpu_us(prod, a.iters) for _ in range(3))[1]
        row = dict(test="pmv", T=T, prod_us=bus)
        for tg, e in exts.items():
            fn = lambda e=e: e.moe_forward(x, w, ids, w13, s13, w2, s2, 10.0, 0xB9, sh)  # noqa: E731
            y = fn()
            torch.cuda.synchronize()
            us = sorted(PT.gpu_us(fn, a.iters) for _ in range(3))[1]
            row[tg] = dict(us=us, rel=round(us / bus - 1, 4), bit_equal=PT.cmp(base, y)["eq"],
                           kernels={k: v for k, v in C.kprofile(fn, 3).items() if v >= 5})
        C.jdump(os.path.join(C.HERE, "results", "pmv.jsonl"), row)
    print("done", flush=True)



# frozen snapshot of the pmv sources for engine use (FusedV): later edits of pmv_nt.py (tuning) never change these bytes, so
# the engine's build never goes stale or rebuilds on GPU time. Snapshot 1: PMV_NT_BOXN validated.
FROZEN1 = P.HERE / "build" / "pmv_frozen1_src"


def freeze1():
    import shutil

    if not (FROZEN1 / "moe_prefill.cu").exists():
        src = sync_sources3()
        FROZEN1.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, FROZEN1 / f.name)
        shutil.copy2(P.HERE / "pm2.cu", FROZEN1 / "pm2.cu")
    return FROZEN1


@functools.cache
def ext_frozen1(defs: tuple):
    from torch.utils.cpp_extension import load

    cuext = P._load_file("mpf_pmvf1_cuext", P.DEPLOY / "cuext.py")
    src = freeze1()
    name = "mpf_pmvf1_" + tag(defs)
    bd = P.HERE / "build" / name
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(src / "pm2.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS) + ["-D" + d for d in defs],
                extra_include_paths=[str(src)], build_directory=str(bd), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


# snapshot 2: + PMV_WAITCNT / PMV_WAITFAST (validated bit-equal at 640-16K rows)
FROZEN2 = P.HERE / "build" / "pmv_frozen2_src"


def freeze2():
    import shutil

    if not (FROZEN2 / "moe_prefill.cu").exists():
        src = sync_sources3()
        FROZEN2.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, FROZEN2 / f.name)
        shutil.copy2(P.HERE / "pm2.cu", FROZEN2 / "pm2.cu")
    return FROZEN2


@functools.cache
def ext_frozen2(defs: tuple):
    from torch.utils.cpp_extension import load

    cuext = P._load_file("mpf_pmvf2_cuext", P.DEPLOY / "cuext.py")
    src = freeze2()
    name = "mpf_pmvf2_" + tag(defs)
    bd = P.HERE / "build" / name
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(src / "pm2.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS) + ["-D" + d for d in defs],
                extra_include_paths=[str(src)], build_directory=str(bd), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


# snapshot 3: + PMV_WAITCNT2 (give-up record inside the wait's asm; validated bit-equal at 640-16K rows)
FROZEN3 = P.HERE / "build" / "pmv_frozen3_src"


def freeze3():
    import shutil

    if not (FROZEN3 / "moe_prefill.cu").exists():
        src = sync_sources3()
        FROZEN3.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, FROZEN3 / f.name)
        shutil.copy2(P.HERE / "pm2.cu", FROZEN3 / "pm2.cu")
    return FROZEN3


@functools.cache
def ext_frozen3(defs: tuple):
    from torch.utils.cpp_extension import load

    cuext = P._load_file("mpf_pmvf3_cuext", P.DEPLOY / "cuext.py")
    src = freeze3()
    name = "mpf_pmvf3_" + tag(defs)
    bd = P.HERE / "build" / name
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(src / "pm2.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS) + ["-D" + d for d in defs],
                extra_include_paths=[str(src)], build_directory=str(bd), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


# engine choices (ENG_PM_NT, frozen snapshot 3): "1" = safe (counted-spin waits that give up after 2^30 polls and record the
# barrier in a device flag instead of hanging), "fast" = plain-spin waits (no device watchdog)
NT_SETS = {"1": ("PMV_NT_BOXN", "PMV_WAITCNT2"), "fast": ("PMV_NT_BOXN", "PMV_WAITFAST")}
NT_SNAP = 3


class FusedV:
    """pm_fused.Fused on a pmv extension (production's kernels + the given PMV_* source patches, e.g. ("PMV_NT_BOXN",)): the
    same layer / layer8 entry points (pm2.cu) and production's W4A16 variant / hybrid setting. Engine: ENG_PM_NT=1."""

    def __init__(self, PM=None, defs=("PMV_NT_BOXN",), snap=1):
        if PM is None:
            from ds41h import moe as PM
        self.variant, hs, he = P.prod_settings(PM)
        # frozen snapshot (prebuild: CUDA_VISIBLE_DEVICES= python3 pmv.py --freeze<snap> <defines>, e.g. snapshot 1 with
        # PMV_NT_BOXN, snapshot 2 with PMV_NT_BOXN,PMV_WAITCNT,PMV_ARV)
        self.e = {1: ext_frozen1, 2: ext_frozen2, 3: ext_frozen3}[snap](tuple(defs))
        self.e.set_hybrid(hs, he)
        self.e.set_swiglu(P.swiglu_ext().launcher())
        self.vdec = 0

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0, fvbuf=None):
        self.e.layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)

    def layer8(self, fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit=10.0, v8=P.ACT8_VARIANT):
        self.e.layer8(fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, v8, self.vdec)


if __name__ == "__main__":
    sys.exit(main())
