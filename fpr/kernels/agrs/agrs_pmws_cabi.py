"""  ext()             the extension agrs_pmws_c: three translation units as the served agrs_pm_rs, each source #included
                    unchanged with host code added: moe_route_ws.cu (production's moe_route.cu private copy, synced as
                    pm_fused.sync_sources does into kernels/agrs/build/pmc_src), moe_prefill_ws.cu (moe_prefill.cu),
                    pm_rs_cabi.cu (agrs/pm_rs.cu + the C ABI); production's NVCC flags; build/agrs_pmws_c. Table symbol
                    fpr_agrs_pmws_table (src_md5 = md5 of the sources it was built from, build/pmc_src/agrs_pmws_md5.h).
  FusedRSC(PM)      agrs_pm.FusedRS's interface (layer / layer_rs) through the C functions (pybind cabi_layer / cabi_layer_rs),
                    with one caller-owned workspace per device (reserve(devs, T)); init() per device as the driver does.

  build             CUDA_VISIBLE_DEVICES= python3 agrs_pmws_cabi.py
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHYS = Path(os.environ.get("PHYS_ROOT", str(HERE.parent.parent)))
if str(PHYS / "kernels") not in sys.path:
    sys.path.insert(0, str(PHYS / "kernels"))
SRC = HERE / "build" / "pmc_src"
BUILD = HERE / "build" / "agrs_pmws_c"
NAME = "agrs_pmws_c"
CABI_DIR = PHYS / "engine" / "mixed_cxx"
H_, E_, I_ = 5120, 384, 576
MAXW = 4


def _pf():
    from moe_prefill import pm_fused as PF   # read-only import (kernels/moe_prefill)
    return PF


def sync_sources() -> Path:
    """pm_fused.sync_sources into kernels/agrs/build/pmc_src (production's sources, the same patches), plus the md5 header."""
    PF = _pf()
    old = PF.SRC
    try:
        PF.SRC = SRC
        src = PF.sync_sources()
    finally:
        PF.SRC = old
    h = hashlib.md5()
    for p in (src / "moe_route.cu", src / "moe_prefill.cu", src / "host.cuh", src / "common.cuh", HERE / "pm_rs.cu",
              HERE / "moe_route_ws.cu", HERE / "moe_prefill_ws.cu", HERE / "pm_rs_cabi.cu", HERE / "pm_rs_cabi.h"):
        h.update(p.read_bytes())
    for p in (HERE / "pm_rs_lt.cpp", CABI_DIR / "fpr_lt.h"):   # fpr_lt's host TU + header when present: a
        if p.exists():                                           # change to either must change the stamp (no stale .so passes)
            h.update(p.read_bytes())
    txt = f'#pragma once\n#define AGRS_PMWS_SRC_MD5 "{h.hexdigest()}"\n'
    f = src / "agrs_pmws_md5.h"
    if not f.exists() or f.read_text() != txt:
        f.write_text(txt)
    return src


@functools.cache
def ext():
    from torch.utils.cpp_extension import load
    PF = _pf()
    cuext = PF._load_file("agrs_pmws_c_cuext", PF.DEPLOY / "cuext.py")    # production's NVCC flags
    src = sync_sources()
    BUILD.mkdir(parents=True, exist_ok=True)
    srcs = ["moe_route_ws.cu", "moe_prefill_ws.cu", "pm_rs_cabi.cu"]
    if (HERE / "pm_rs_lt.cpp").exists():   # fpr_lt's host TU: pm_rs_cabi.cu's agrs_lt_mm + the fpr_agrs_lt_warm entry
        srcs.append("pm_rs_lt.cpp")
    return load(name=NAME, sources=[str(HERE / f) for f in srcs],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=list(cuext.NVCC_FLAGS),
                extra_include_paths=[str(src), str(HERE), str(CABI_DIR)], build_directory=str(BUILD),
                verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


def lib_path() -> str:
    return ext().__file__


# ---- ctypes mirrors of pm_rs_cabi.h (sizes checked against the extension's sizeof in check())
class FprCall(ctypes.Structure):
    _fields_ = [("stream", ctypes.c_void_p), ("device", ctypes.c_int), ("rank", ctypes.c_int)]


class AgrsPmwsInitArgs(ctypes.Structure):
    _fields_ = [("device", ctypes.c_int32), ("pad_", ctypes.c_int32), ("hybrid_split", ctypes.c_int64),
                ("hybrid_engine", ctypes.c_int64), ("swiglu_fn", ctypes.c_int64)]


class AgrsPmwsThreadArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("ws", ctypes.c_void_p), ("ws_bytes", ctypes.c_int64)]


class AgrsPmwsLayerArgs(ctypes.Structure):
    _fields_ = [("call", FprCall),
                ("fall", ctypes.c_void_p), ("gr", ctypes.c_void_p), ("nv", ctypes.c_void_p),
                ("W", ctypes.c_int64), ("R", ctypes.c_int64), ("vcount", ctypes.c_int64), ("H", ctypes.c_int64),
                ("w13", ctypes.c_void_p), ("s13", ctypes.c_void_p), ("w2", ctypes.c_void_p), ("s2", ctypes.c_void_p),
                ("E", ctypes.c_int64), ("I", ctypes.c_int64),
                ("sw13", ctypes.c_void_p), ("sw2", ctypes.c_void_p), ("Is", ctypes.c_int64),
                ("gu", ctypes.c_void_p), ("hsh", ctypes.c_void_p), ("sh", ctypes.c_void_p), ("scratch_rows", ctypes.c_int64),
                ("part", ctypes.c_void_p),
                ("rs_buf", ctypes.c_void_p * MAXW), ("rs_flag", ctypes.c_void_p * MAXW), ("epoch", ctypes.c_void_p),
                ("grid", ctypes.c_int64),
                ("limit", ctypes.c_double), ("variant", ctypes.c_int64), ("padded", ctypes.c_int32), ("pad_", ctypes.c_int32),
                ("ws", ctypes.c_void_p), ("ws_bytes", ctypes.c_int64)]


def _b(s) -> bytes:
    return ctypes.string_at(ctypes.addressof(s), ctypes.sizeof(s))


def check() -> None:
    assert ctypes.sizeof(AgrsPmwsLayerArgs) == ext().cabi_args_size(), (ctypes.sizeof(AgrsPmwsLayerArgs), ext().cabi_args_size())


def pack_init(device, hybrid_split, hybrid_engine, swiglu_fn) -> bytes:
    return _b(AgrsPmwsInitArgs(int(device), 0, int(hybrid_split), int(hybrid_engine), int(swiglu_fn)))


def pack_thread(ws, stream=None) -> bytes:
    import torch
    st = torch.cuda.current_stream(ws.device) if stream is None else stream
    return _b(AgrsPmwsThreadArgs(FprCall(st.cuda_stream, ws.device.index, 0), ws.data_ptr(), ws.numel()))


def pack_layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, ws, part=None, bufs=None, flags=None,
               epoch=None, rank=0, grid=0, limit=10.0, variant=0xB9, padded=False, stream=None) -> bytes:
    """The AgrsPmwsLayerArgs bytes of one call (layer: part; layer_rs: bufs / flags / epoch / rank / grid)."""
    import torch
    st = torch.cuda.current_stream(fall.device) if stream is None else stream
    a = AgrsPmwsLayerArgs()
    a.call = FprCall(st.cuda_stream, fall.device.index, int(rank))
    a.fall, a.gr, a.nv = fall.data_ptr(), gr.data_ptr(), nv.data_ptr()
    a.W, a.R, a.vcount, a.H = nv.numel(), int(R), int(vcount), fall.shape[1]
    a.w13, a.s13, a.w2, a.s2 = w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr()
    a.E, a.I = w13.shape[0], w13.shape[1] // 2
    a.sw13, a.sw2, a.Is = sw13.data_ptr(), sw2.data_ptr(), sw2.shape[1]
    a.gu, a.hsh, a.sh, a.scratch_rows = gu.data_ptr(), hsh.data_ptr(), sh.data_ptr(), gu.shape[0]
    a.part = part.data_ptr() if part is not None else None
    if bufs is not None:
        for q in range(nv.numel()):
            a.rs_buf[q] = bufs[q].data_ptr()
            a.rs_flag[q] = flags[q].data_ptr()
        a.epoch = epoch.data_ptr()
    a.grid = int(grid)
    a.limit, a.variant, a.padded = float(limit), int(variant), int(bool(padded))
    a.ws, a.ws_bytes = ws.data_ptr(), ws.numel()
    return _b(a)


def ws_bytes(T: int, H: int = H_, E: int = E_, I: int = I_) -> int:
    return int(ext().cabi_ws_bytes(int(T), int(H), int(E), int(I)))


class FusedRSC:
    """agrs_pm.FusedRS's interface through the C ABI: layer / layer_rs with the same arguments, buffers from one caller-owned
    workspace per device (reserve(devs, T) before the first call). As agrs_ws's ENG_AGRS_WS=1 workspace: a call above the
    reserved rows, on a capturing stream, on a device without a workspace or with an fvbuf runs the extension's allocating
    layer / layer_rs (pm_rs.cu's pm_layer_impl, the same kernels), counted in self.st."""

    def __init__(self, PM=None, devices=None):
        import torch
        PF = _pf()
        if PM is None:
            from ds41h import moe as PM
        self.variant, self.hs, self.he = PF.prod_settings(PM)
        self.e = ext()
        self.e.set_hybrid(self.hs, self.he)              # the pybind path's production setter (same statics)
        self.swiglu = PF.swiglu_ext().launcher()
        self.e.set_swiglu(self.swiglu)
        for d in (range(torch.cuda.device_count()) if devices is None else devices):
            self.e.cabi_init(int(d), int(self.hs), int(self.he), int(self.swiglu))
        self.ws, self.cap = {}, {}
        self.st = dict(ws_calls=0, fb_size=0, fb_capture=0, fb_no_ws=0, stream_switches=0, max_T=0)

    def reserve(self, devs, T: int) -> int:
        import torch
        n = ws_bytes(T)
        for d in devs:
            d = torch.device(d)
            with torch.cuda.device(d):
                self.ws[d.index] = torch.empty(n, dtype=torch.uint8, device=d)
            self.cap[d.index] = int(T)
        return n

    def _ws(self, fall, T: int, fvbuf):
        import torch
        d = fall.device.index
        if d not in self.ws or fvbuf is not None:
            self.st["fb_no_ws"] += 1
            return None
        if torch.cuda.is_current_stream_capturing():
            self.st["fb_capture"] += 1
            return None
        if T > self.cap[d]:
            self.st["fb_size"] += 1
            return None
        self.st["ws_calls"] += 1
        self.st["max_T"] = max(self.st["max_T"], int(T))
        return self.ws[d]

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0, fvbuf=None):
        ws = self._ws(fall, nv.numel() * R if padded else vcount, fvbuf)
        if ws is None:
            self.e.layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)
            return
        self.e.cabi_layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, ws)

    def layer_rs(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                 padded=False, limit=10.0, fvbuf=None):
        ws = self._ws(fall, nv.numel() * R if padded else vcount, fvbuf)
        if ws is None:
            self.e.layer_rs(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                            limit, self.variant, padded, fvbuf)
            return
        self.e.cabi_layer_rs(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, bufs, flags, epoch, rank, grid, limit,
                             self.variant, padded, ws)


if __name__ == "__main__":
    ext()
    print(f"{NAME} built in {BUILD}: {lib_path()}", flush=True)
