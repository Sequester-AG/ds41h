"""agrs RSF: the production-MoE layer of kernels/moe_prefill/pm_fused with the RS push fused into its combine (pm_rs.cu).

  ext()        the extension agrs_pm_rs: production's moe_route.cu + moe_prefill.cu (private copies synced exactly as
               pm_fused.sync_sources does, into kernels/agrs/build/pm_src) + pm_rs.cu, production's NVCC flags
  FusedRS()    per process: the extension with production's W4A16 variant and hybrid setting (pm_fused.prod_settings) and
               pm_fused's swiglu launcher (moe_prefill's extension, already loaded by the engine's pmf())
  build()      CUDA_VISIBLE_DEVICES= python3 agrs_pm.py

FusedRS.layer_rs(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid)
  = pm_fused.Fused.layer(...) with part's rows stored into bufs[q][rank] (owner q's RS receive slot) instead, plus
    comm.cu's release signals (owners wait for W * grid signals per round on their flag of this path).
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
SRC = HERE / "build" / "pm_src"
BUILD = HERE / "build" / "agrs_pm_rs"
NAME = "agrs_pm_rs"


def _pf():
    from moe_prefill import pm_fused as PF   # read-only import (kernels/moe_prefill)
    return PF


def sync_sources() -> Path:
    """pm_fused.sync_sources with its output directory pointed at kernels/agrs/build/pm_src (moe_prefill's own copy is
    not touched): the same production sources, the same host.cuh / module-block patches, only changed bytes rewritten."""
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
    cuext = PF._load_file("agrs_pm_cuext", PF.DEPLOY / "cuext.py")    # production's NVCC flags
    src = sync_sources()
    BUILD.mkdir(parents=True, exist_ok=True)
    return load(name=NAME, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(HERE / "pm_rs.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS), extra_include_paths=[str(src)],
                build_directory=str(BUILD), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


class FusedRS:
    def __init__(self, PM=None):
        PF = _pf()
        if PM is None:
            from ds41h import moe as PM
        self.variant, hs, he = PF.prod_settings(PM)
        self.e = ext()
        self.e.set_hybrid(hs, he)
        self.e.set_swiglu(PF.swiglu_ext().launcher())

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0, fvbuf=None):
        self.e.layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)

    def layer_rs(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                 padded=False, limit=10.0, fvbuf=None):
        self.e.layer_rs(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                        limit, self.variant, padded, fvbuf)


def build() -> None:
    ext()


if __name__ == "__main__":
    build()
    print(f"{NAME} built in {BUILD}", flush=True)
