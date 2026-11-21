"""  load()      (api, route): the two extensions, built exactly as v4.load() builds the served pair (the same sources, one TU
              each, the same CFLAGS and production defines, v4.DEFAULT / its route keys), plus host code only:
                moe_v4c_api<tag>   moe_v4_cabi.cu  (moe_v4.cu + gemm1r / gemm2r / swiglu / prepr6 / combine C ABI)
                                                                                      table fpr_moe_v4_table  ("moe_v4")
                moe_v4c_route<tag> route2_cabi.cu  (route2.cu + route2r C ABI)            table fpr_moe_v4r_table ("moe_v4r")
              in kernels/moe_v4/build_cabi (never the served engine/build/moe_v4). Every served binding stays (v4_bind /
              r2_bind), plus cabi_* thin callers of the C functions.

  build       CUDA_VISIBLE_DEVICES= python3 v4_cabi.py
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v4 as V4  # noqa: E402  (the served loader: its CFLAGS / DEFAULT / ROUTE_KEYS)

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(HERE)))
CABI_DIR = os.path.join(PHYS, "engine", "mixed_cxx")
ROOT = os.path.join(HERE, "build_cabi")


def _md5_header(bd: str) -> None:
    h = hashlib.md5()
    for f in ("moe_v4.cu", "route2.cu", "moe_v4_cabi.cu", "route2_cabi.cu", "moe_v4_cabi.h"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    for f in sorted(os.listdir(HERE)):
        if f.endswith((".cuh", ".h")) and f != "moe_v4_cabi.h":
            h.update(open(os.path.join(HERE, f), "rb").read())
    txt = f'#pragma once\n#define MOE_V4C_SRC_MD5 "{h.hexdigest()}"\n'
    p = os.path.join(bd, "moe_v4c_md5.h")
    if not os.path.exists(p) or open(p).read() != txt:
        open(p, "w").write(txt)


@functools.cache
def load():
    from torch.utils.cpp_extension import load as _load
    d = dict(V4.DEFAULT)                          # production: no MOE_V4_DEFS / MOE_V4_SRC
    tag = "".join(f"_{k.lower()}{v}" for k, v in sorted(d.items()))
    bd = os.path.join(ROOT, "v4capi" + tag)
    os.makedirs(bd, exist_ok=True)
    _md5_header(bd)
    api = _load(name="moe_v4c_api" + tag, sources=[os.path.join(HERE, "moe_v4_cabi.cu")], build_directory=bd, verbose=False,
                extra_cuda_cflags=V4.CFLAGS + [f"-D{k}={v}" for k, v in sorted(d.items())],
                extra_include_paths=[HERE, CABI_DIR, bd])
    rd = {V4.ROUTE_KEYS[k]: v for k, v in d.items() if k in V4.ROUTE_KEYS}
    rtag = "".join(f"_{k.lower()}{v}" for k, v in sorted(rd.items()))
    rbd = os.path.join(ROOT, "v4croute" + rtag)
    os.makedirs(rbd, exist_ok=True)
    _md5_header(rbd)
    route = _load(name="moe_v4c_route" + rtag, sources=[os.path.join(HERE, "route2_cabi.cu")], build_directory=rbd, verbose=False,
                  extra_cuda_cflags=V4.CFLAGS + [f"-D{k}={v}" for k, v in sorted(rd.items())],
                  extra_include_paths=[HERE, CABI_DIR, rbd])
    return api, route


def lib_paths():
    api, route = load()
    return api.__file__, route.__file__


# ---- ctypes mirrors of moe_v4_cabi.h
class FprCall(ctypes.Structure):
    _fields_ = [("stream", ctypes.c_void_p), ("device", ctypes.c_int), ("rank", ctypes.c_int)]


class MoeV4InitArgs(ctypes.Structure):
    _fields_ = [("device", ctypes.c_int32), ("pad_", ctypes.c_int32)]


class MoeV4Gemm1rArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("w13", ctypes.c_void_p), ("s13", ctypes.c_void_p), ("E", ctypes.c_int64),
                ("ecs", ctypes.c_void_p), ("tiles", ctypes.c_void_p), ("counts", ctypes.c_void_p), ("img", ctypes.c_void_p),
                ("himg", ctypes.c_void_p), ("grid", ctypes.c_int64)]


class MoeV4Gemm2rArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("w2", ctypes.c_void_p), ("s2", ctypes.c_void_p), ("E", ctypes.c_int64),
                ("ecs", ctypes.c_void_p), ("tiles", ctypes.c_void_p), ("counts", ctypes.c_void_p), ("himg", ctypes.c_void_p),
                ("yout", ctypes.c_void_p), ("grid", ctypes.c_int64)]


class MoeV4SwigluArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("gu", ctypes.c_void_p), ("h", ctypes.c_void_p), ("T", ctypes.c_int32), ("I", ctypes.c_int32)]


class MoeV4Prepr6Args(ctypes.Structure):
    _fields_ = [("call", FprCall), ("x", ctypes.c_void_p), ("ecs", ctypes.c_void_p), ("counts", ctypes.c_void_p),
                ("img", ctypes.c_void_p), ("ptok", ctypes.c_void_p), ("inv", ctypes.c_void_p), ("T", ctypes.c_int32),
                ("sms", ctypes.c_int32)]


class MoeV4CombineArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("Y", ctypes.c_void_p), ("slotmap", ctypes.c_void_p), ("sh", ctypes.c_void_p),
                ("out", ctypes.c_void_p), ("T", ctypes.c_int32), ("pad_", ctypes.c_int32)]


class MoeV4Route2rArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("rows", ctypes.c_void_p), ("T", ctypes.c_int64),
                ("ecs", ctypes.c_void_p), ("t1", ctypes.c_void_p), ("t2", ctypes.c_void_p), ("counts", ctypes.c_void_p),
                ("slotmap", ctypes.c_void_p),
                ("ecs_bytes", ctypes.c_int64), ("t1_bytes", ctypes.c_int64), ("t2_bytes", ctypes.c_int64),
                ("counts_n", ctypes.c_int64), ("slotmap_n", ctypes.c_int64),
                ("grid", ctypes.c_int64), ("pdl", ctypes.c_int32), ("one_max_t", ctypes.c_int32)]


def _b(s) -> bytes:
    return ctypes.string_at(ctypes.addressof(s), ctypes.sizeof(s))


def check() -> None:
    api, route = load()
    assert list(api.cabi_args_sizes()) == [ctypes.sizeof(t) for t in (MoeV4InitArgs, MoeV4Gemm1rArgs, MoeV4Gemm2rArgs,
                                                                     MoeV4SwigluArgs, MoeV4Prepr6Args, MoeV4CombineArgs)], \
        api.cabi_args_sizes()
    assert list(route.cabi_args_sizes()) == [ctypes.sizeof(MoeV4InitArgs), ctypes.sizeof(MoeV4Route2rArgs)], route.cabi_args_sizes()


def _call(t, stream):
    import torch
    st = torch.cuda.current_stream(t.device) if stream is None else stream
    return FprCall(st.cuda_stream, t.device.index, 0)


def pack_init(device) -> bytes:
    return _b(MoeV4InitArgs(int(device), 0))


def pack_gemm1r(w13, s13, ecs, tiles, counts, img, himg, grid=0, stream=None) -> bytes:
    return _b(MoeV4Gemm1rArgs(_call(w13, stream), w13.data_ptr(), s13.data_ptr(), w13.shape[0], ecs.data_ptr(), tiles.data_ptr(),
                              counts.data_ptr(), img.data_ptr(), himg.data_ptr(), int(grid)))


def pack_gemm2r(w2, s2, ecs, tiles, counts, himg, yout, grid=0, stream=None) -> bytes:
    return _b(MoeV4Gemm2rArgs(_call(w2, stream), w2.data_ptr(), s2.data_ptr(), w2.shape[0], ecs.data_ptr(), tiles.data_ptr(),
                              counts.data_ptr(), himg.data_ptr(), yout.data_ptr(), int(grid)))


def pack_swiglu(gu, h, stream=None) -> bytes:
    return _b(MoeV4SwigluArgs(_call(h, stream), gu.data_ptr(), h.data_ptr(), h.shape[0], h.shape[1]))


def pack_prepr6(x, ecs, counts, img, ptok, inv, sms, stream=None) -> bytes:
    return _b(MoeV4Prepr6Args(_call(x, stream), x.data_ptr(), ecs.data_ptr(), counts.data_ptr(), img.data_ptr(), ptok.data_ptr(),
                              inv.data_ptr(), x.shape[0], int(sms)))


def pack_combine(Y, slotmap, sh, out, stream=None) -> bytes:
    return _b(MoeV4CombineArgs(_call(out, stream), Y.data_ptr(), slotmap.data_ptr(),
                               sh.data_ptr() if sh is not None and sh.numel() else None, out.data_ptr(), out.shape[0], 0))


def pack_route2r_served(rows, T, ecs, t1, t2, counts, slotmap, rts, grid, stream=None, pdl=3, one_max_t=1365) -> bytes:
    """The MoeV4Route2rArgs the SERVED route2r(rows, T, ecs, t1, t2, counts, slotmap, rts, grid) launches with: grid = that call's last argument (k_one's slot-group count sg; the launch grid max(S, min(ceil(nE /
    F_THREADS), sms)) is computed inside moe_v4r.route2r from T and the device's SM count, as the served wrapper computes
    it from its g_sms); pdl / one_max_t = the served module's route2_config values, which no served path sets (defaults
    3 / 1365). rts is only size-checked by the served wrapper (>= route2_scratch_ints()) and not read by its launch, so it
    is not an Args field (checked here instead, when the served route module's value is known: 64 ints is its RS_INTS floor)."""
    assert rts.numel() > 0, "route2r: rts (the served wrapper checks rts.numel() >= route2_scratch_ints())"
    return pack_route2r(rows, T, ecs, t1, t2, counts, slotmap, grid=grid, pdl=pdl, one_max_t=one_max_t, stream=stream)


def pack_route2r(rows, T, ecs, t1, t2, counts, slotmap, grid=0, pdl=3, one_max_t=1365, stream=None) -> bytes:
    return _b(MoeV4Route2rArgs(_call(rows, stream), rows.data_ptr(), int(T), ecs.data_ptr(), t1.data_ptr(), t2.data_ptr(),
                               counts.data_ptr(), slotmap.data_ptr(), ecs.nbytes, t1.nbytes, t2.nbytes, counts.numel(),
                               slotmap.numel(), int(grid), int(pdl), int(one_max_t)))


if __name__ == "__main__":
    a, r = load()
    print(f"moe_v4c built: {a.__file__} ; {r.__file__}", flush=True)
