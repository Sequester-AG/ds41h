"""agrs RSF on the narrow-tile MoE (ENG_PM_NT): pm2_rs.cu (gen_pm2_rs.py: moe_prefill's pm2.cu +
pm_rs.cu's RSF additions) built with moe_prefill's frozen snapshot 3 of production's MoE sources (pmv_frozen3_src: the
PMV_* patches, gated by -D) and pmv.NT_SETS's defines.

  ext(mode)       mode "1" (PMV_NT_BOXN + PMV_WAITCNT2, the recommended safe set) or "fast" (PMV_NT_BOXN + PMV_WAITFAST):
                  extension agrs_pmnt_<mode> in kernels/agrs/build/agrs_pmnt_<mode>, sources copied once from the frozen
                  snapshot into kernels/agrs/build/pmnt3_src (kernels/moe_prefill is only read)
  FusedRSNT(mode) as agrs_pm.FusedRS: layer() (= pmv.FusedV.layer, same kernels) and layer_rs()
  build           CUDA_VISIBLE_DEVICES= python3 agrs_pmnt.py [1 fast]
"""
from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHYS = Path(os.environ.get("PHYS_ROOT", str(HERE.parent.parent)))
if str(PHYS / "kernels") not in sys.path:
    sys.path.insert(0, str(PHYS / "kernels"))
FROZEN3 = PHYS / "kernels" / "moe_prefill" / "build" / "pmv_frozen3_src"
SRC = HERE / "build" / "pmnt3_src"
MODES = {"1": ("PMV_NT_BOXN", "PMV_WAITCNT2"), "fast": ("PMV_NT_BOXN", "PMV_WAITFAST")}   # = pmv.NT_SETS (asserted)


def _pf():
    from moe_prefill import pm_fused as PF   # read-only imports (kernels/moe_prefill)
    return PF


def sources() -> Path:
    """The frozen snapshot's files, copied once (never rewritten: the build stays valid if the snapshot dir changes), and
    pm2_rs.cu generated from the copy's pm2.cu."""
    if not (SRC / "moe_prefill.cu").exists():
        SRC.mkdir(parents=True, exist_ok=True)
        for f in FROZEN3.iterdir():
            shutil.copy2(f, SRC / f.name)
    import gen_pm2_rs
    txt = gen_pm2_rs.generate(SRC / "pm2.cu")
    out = HERE / "pm2_rs.cu"
    if not out.exists() or out.read_text() != txt:
        out.write_text(txt)
    return SRC


@functools.cache
def ext(mode: str = "1"):
    from torch.utils.cpp_extension import load
    from moe_prefill import pmv
    assert pmv.NT_SETS[mode] == MODES[mode] and pmv.NT_SNAP == 3, "moe_prefill's NT set / snapshot changed: regenerate"
    PF = _pf()
    cuext = PF._load_file("agrs_pmnt_cuext", PF.DEPLOY / "cuext.py")    # production's NVCC flags
    src = sources()
    name = "agrs_pmnt_" + mode
    bd = HERE / "build" / name
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(HERE / "pm2_rs.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS) + ["-D" + d for d in MODES[mode]],
                extra_include_paths=[str(src)], build_directory=str(bd), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


class FusedRSNT:
    def __init__(self, PM=None, mode: str = "1"):
        PF = _pf()
        if PM is None:
            from ds41h import moe as PM
        self.mode = mode
        self.variant, hs, he = PF.prod_settings(PM)
        self.e = ext(mode)
        self.e.set_hybrid(hs, he)
        self.e.set_swiglu(PF.swiglu_ext().launcher())

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0, fvbuf=None):
        self.e.layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)

    def layer_rs(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                 padded=False, limit=10.0, fvbuf=None):
        self.e.layer_rs(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, bufs, flags, epoch, rank, grid,
                        limit, self.variant, padded, fvbuf)


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["1"]):
        ext(m)
        print(f"agrs_pmnt_{m} built in {HERE / 'build' / ('agrs_pmnt_' + m)}", flush=True)
