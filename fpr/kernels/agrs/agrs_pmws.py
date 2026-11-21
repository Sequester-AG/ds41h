"""agrs WS: the production-MoE layer of agrs_pm.FusedRS with its per-call buffers in ONE persistent
workspace per GPU.

  ext()           the extension agrs_pm_rs_ws: production's moe_route.cu + moe_prefill.cu (private copies synced exactly as
                  pm_fused.sync_sources does, into kernels/agrs/build/pmws_src; agrs_pm's build/pm_src is not touched) +
                  pm_rs_ws.cu (which #includes pm_rs.cu unchanged), production's NVCC flags, built in
                  kernels/agrs/build/agrs_pm_rs_ws
  FusedRSWS()     agrs_pm.FusedRS's interface: layer() / layer_rs() with the same arguments, served from the workspace
                  (fallback to the allocating layer when the device has none, the stream is capturing or T is above its
                  rows; counted in stats())
  ws_bytes(T, ...), reserve(devs, T): size / allocate + bind one workspace per device for T call rows (all ranks)
  build           CUDA_VISIBLE_DEVICES= python3 agrs_pmws.py

Per row of T (H 5120, I 576, top-6): x 10,240 + h 6,912 + xp / slot 61,440 + ids / tw / perm_w / meta 144 = 78,736 B.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHYS = Path(os.environ.get("PHYS_ROOT", str(HERE.parent.parent)))
if str(PHYS / "kernels") not in sys.path:
    sys.path.insert(0, str(PHYS / "kernels"))
SRC = HERE / "build" / "pmws_src"
BUILD = HERE / "build" / "agrs_pm_rs_ws"
NAME = "agrs_pm_rs_ws"
H_, E_, I_ = 5120, 384, 576           # hidden, routed experts, MoE intermediate per tensor-parallel rank (ds41h)


def _pf():
    from moe_prefill import pm_fused as PF   # read-only import (kernels/moe_prefill)
    return PF


def sync_sources() -> Path:
    """pm_fused.sync_sources with its output directory pointed at kernels/agrs/build/pmws_src."""
    PF = _pf()
    old = PF.SRC
    try:
        PF.SRC = SRC
        return PF.sync_sources()
    finally:
        PF.SRC = old


@functools.cache
def ext():
    from torch.utils.cpp_extension import load
    PF = _pf()
    cuext = PF._load_file("agrs_pmws_cuext", PF.DEPLOY / "cuext.py")    # production's NVCC flags
    src = sync_sources()
    BUILD.mkdir(parents=True, exist_ok=True)
    return load(name=NAME, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(HERE / "pm_rs_ws.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS), extra_include_paths=[str(src)],
                build_directory=str(BUILD), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


def ws_bytes(T: int, H: int = H_, E: int = E_, I: int = I_) -> int:
    return int(ext().ws_bytes(int(T), int(H), int(E), int(I)))


def reserve(devs, T: int, H: int = H_, E: int = E_, I: int = I_) -> int:
    """One workspace per device for T call rows, bound in the extension (it keeps the reference). Returns bytes per GPU."""
    import torch
    e = ext()
    n = ws_bytes(T, H, E, I)
    for d in devs:
        d = torch.device(d)
        e.ws_clear(d.index)
        with torch.cuda.device(d):
            buf = torch.empty(n, dtype=torch.uint8, device=d)
        e.ws_set(buf, int(T), int(H), int(E), int(I))
    return n


def release(devs) -> None:
    import torch
    for d in devs:
        ext().ws_clear(torch.device(d).index)


def stats() -> dict:
    return dict(ext().ws_stats())


class FusedRSWS:
    """agrs_pm.FusedRS with layer / layer_rs on the workspace entries (same arguments, same kernels)."""

    def __init__(self, PM=None):
        PF = _pf()
        if PM is None:
            from ds41h import moe as PM
        self.variant, hs, he = PF.prod_settings(PM)
        self.e = ext()
        self.e.set_hybrid(hs, he)
        self.e.set_swiglu(PF.swiglu_ext().launcher())

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0, fvbuf=None):
        self.e.layer_ws(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)

    def layer_rs(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                 padded=False, limit=10.0, fvbuf=None):
        self.e.layer_rs_ws(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                           limit, self.variant, padded, fvbuf)


def build() -> None:
    ext()


if __name__ == "__main__":
    build()
    print(f"{NAME} built in {BUILD}", flush=True)
