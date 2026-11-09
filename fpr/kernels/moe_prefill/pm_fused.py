"""pm_fused: the engine's production-MoE layer (engine.py p_moe_prod) as one host call per layer per rank.

Unfused, p_moe_prod issues ~13 Python-level ops per layer per rank (index_select x2, contiguous x2, torch.mm x2, swiglu,
torch.empty, moe_forward_w4a16 (whose C++ launches permute x2, act_perm16, GEMM1 + remainder, GEMM2 + remainder,
combine), zero_, index_copy_). pm_fused.layer does the same work in one call: k_pm_prep (row gather + route-row unpack),
cuBLAS x2 + the fork's swiglu (shared expert slice), production's permute / act_perm16 / GEMMs (unmodified sources,
production's flags), k_pm_combine (production's combine arithmetic, writing the rank partial `part` in place with
zeros on padding rows). See pm_fused.cu for the two row modes (compact = bit-identical by construction, padded =
graph-capturable).

  ext()          the extension (mpf_pm_fused): production's moe_route.cu + moe_prefill.cu (private copies, host.cuh with
                 engine/prod_moe_ext.py's per-device smem patch) + pm_fused.cu; production's bindings are in it too
  swiglu_ext()   mpf_pm_swiglu (pm_swiglu.cu, moe_raw.cu's flags)
  Fused()        per process: the extension with production's W4A16 variant and hybrid setting (from ds41h.moe)
  build()        both extensions on the CPU (CUDA_VISIBLE_DEVICES= python3 pm_fused.py)

Call (engine bucket state s of rank r, layer L; nv = int32 [W] valid rows per rank on the rank's device):
  pmf.layer(s["gf"].view(torch.bfloat16).view(Tall, D), s["gr"], s["nv"], R, vcount, w13, s13, w2, s2, sw13, sw2,
            s["gu"], s["hsh"], s["sh"], s["part"], padded=False)
"""
from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHYS = Path(os.environ.get("PHYS_ROOT", str(HERE.parent.parent)))
DEPLOY = Path(os.environ.get("DS41H_DEPLOY", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))) / "ds41h"
SRC = HERE / "build" / "pm_src"
BUILD = HERE / "build" / "mpf_pm_fused"
BUILD_SW = HERE / "build" / "mpf_pm_swiglu"
NAME, NAME_SW = "mpf_pm_fused", "mpf_pm_swiglu"
BIND_ORIG = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"
BIND_PATCH = "void ds41h_prefill_bind(pybind11::module_& m) {  // pm_fused: production's module body, registered by pm_fused.cu"
# bench.build_extr's flags for moe_raw.cu (the engine's fork MoE extension, which holds the current swiglu)
SW_FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v", "--expt-relaxed-constexpr"]
# act8 (layer8): production's deterministic W4A8 variant (ds41h.moe DS41H_MOE_DETERMINISTIC=1: GEMM1 5 = TMA, activations
# in routed-row order; GEMM2 7 = TMA into slot rows + combine). Production's W4A8 default 0x95 (GEMM2 9: bf16 atomic
# reductions into the output) is neither deterministic nor fp32-combined, so it is not offered here.
ACT8_VARIANT = int(os.environ.get("PM_ACT8_VARIANT", "0x75"), 0)


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_if_changed(p: Path, data: bytes) -> None:
    if not p.exists() or p.read_bytes() != data:
        p.write_bytes(data)


def sync_sources() -> Path:
    """Production's kernels/*.cu|cuh|h|hpp (the ds41h package, read-only) -> SRC, with host.cuh's smem opt-in per device
    (engine/prod_moe_ext.py ORIG -> PATCH, read-only import) and moe_prefill.cu's module renamed (BIND_ORIG -> BIND_PATCH).
    Only changed bytes are rewritten (torch's build hash)."""
    pme = _load_file("mpf_prod_moe_ext", PHYS / "engine" / "prod_moe_ext.py")
    SRC.mkdir(parents=True, exist_ok=True)
    for f in sorted((DEPLOY / "kernels").iterdir()):
        if f.suffix not in (".cu", ".cuh", ".h", ".hpp") or not f.is_file():
            continue
        data = f.read_bytes()
        if f.name == "host.cuh":
            txt = data.decode()
            assert txt.count(pme.ORIG) == 1, "deploy host.cuh opt_in_smem changed: re-check engine/prod_moe_ext.py and pm_fused"
            data = txt.replace(pme.ORIG, pme.PATCH).encode()
        elif f.name == "moe_prefill.cu":
            txt = data.decode()
            assert txt.count(BIND_ORIG) == 1, "deploy moe_prefill.cu module block changed: re-check pm_fused"
            data = txt.replace(BIND_ORIG, BIND_PATCH).encode()
        _write_if_changed(SRC / f.name, data)
    return SRC


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    cuext = _load_file("mpf_pm_cuext", DEPLOY / "cuext.py")   # production's NVCC flags (stdlib-only module)
    src = sync_sources()
    BUILD.mkdir(parents=True, exist_ok=True)
    return load(name=NAME, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(HERE / "pm_fused.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS), extra_include_paths=[str(src)],
                build_directory=str(BUILD), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


@functools.cache
def swiglu_ext():
    from torch.utils.cpp_extension import load

    BUILD_SW.mkdir(parents=True, exist_ok=True)
    return load(name=NAME_SW, sources=[str(HERE / "pm_swiglu.cu")], extra_cuda_cflags=SW_FLAGS, build_directory=str(BUILD_SW),
                verbose=False)


def prod_settings(PM) -> tuple[int, int, int]:
    """(variant, hybrid split, hybrid engine) that ds41h.moe.moe_forward_w4a16 uses for calls above SMALL_T rows."""
    if getattr(PM, "_CTL_PATH", ""):
        raise RuntimeError("pm_fused: DS41H_PREFILL_CTL is set (runtime-tunable production knobs); not supported")
    v = PM.VARIANT if PM.VARIANT and PM.is_w4a16_variant(PM.VARIANT) else PM.W4A16_VARIANT
    return int(v), int(PM.HYBRID), int(PM.HYBRID_ENGINE)


class Fused:
    """One per process (all devices). layer(...) = pm_fused.cu pm_layer with production's variant."""

    def __init__(self, PM=None):
        if PM is None:
            from ds41h import moe as PM
        self.variant, hs, he = prod_settings(PM)
        self.e = ext()
        self.e.set_hybrid(hs, he)            # = production's _apply_hybrid (this extension's own copy of the global)
        self.e.set_swiglu(swiglu_ext().launcher())

    def layer(self, fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, padded=False, limit=10.0,
              fvbuf=None):
        self.e.layer(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, padded, fvbuf)

    def layer8(self, fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit=10.0, v8=ACT8_VARIANT):
        """ENG_PF_ACT8 (opt-in, LOSSY on prefill rows): nd = int32 [W] decode rows per rank on the device (a prefix of each
        rank's valid rows), ndec = sum(nd) (host). Decode rows: this W4A16 variant; prefill rows: production's W4A8 v8."""
        self.e.layer8(fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, v8)


def build() -> None:
    ext()
    swiglu_ext()


if __name__ == "__main__":
    build()
    print(f"{NAME} built in {BUILD}; {NAME_SW} in {BUILD_SW}", flush=True)
