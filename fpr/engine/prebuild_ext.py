#!/usr/bin/env python3
"""Compile every CUDA extension the engine loads, on the CPU, before any GPU time.

The engine and the kernels it imports build their extensions with torch's JIT loader on first use. This script calls
the SAME loader functions with the same arguments (it imports the engine first, so its environment defaults apply),
so every build directory gets the build.ninja the engine would write; at GPU time ninja then finds everything up to
date and compiles nothing.

  CUDA_VISIBLE_DEVICES= python engine/prebuild_ext.py            # build all (about 30 extensions, sm_90a)
  python engine/prebuild_ext.py --list                           # the targets and where each one lands
  CUDA_VISIBLE_DEVICES= python engine/prebuild_ext.py --only head_comm,idx2f_tab
  CUDA_VISIBLE_DEVICES= python engine/prebuild_ext.py --root /fast/disk

Build locations:
  - extensions whose loaders take a configurable directory go under --root (default: the repository):
      <root>/engine/build/idx2f      (IDX2F_BUILD)
      <root>/engine/build/moe_v4     (MOE_V4_BUILD)
      <root>/build/torch_extensions  (TORCH_EXTENSIONS_DIR: loaders without an explicit directory)
    An already-set environment variable wins. The script writes <root>/build/prebuilt.env with these exports: source
    it before serving, or the engine looks elsewhere and compiles again.
  - the other kernels build next to their sources (kernels/*/build*, engine/build/*), as their loaders fix it.

Requirements: the CUDA toolkit the installed torch was built with (nvcc on PATH or CUDA_HOME), ninja, and the
production kernel package `ds41h` (its directory is found through `import ds41h`; DS41H_DEPLOY overrides). No GPU is
needed; TORCH_CUDA_ARCH_LIST defaults to 9.0a (H100 / H200), the architecture every loader also passes explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # engine/prebuild_ext.py -> the repository


def _env(root: Path) -> dict:
    out = {"IDX2F_BUILD": str(root / "engine/build/idx2f"), "MOE_V4_BUILD": str(root / "engine/build/moe_v4"),
           "TORCH_EXTENSIONS_DIR": str(root / "build/torch_extensions")}
    return {k: os.environ.get(k, v) for k, v in out.items()}


def _targets(E):
    """(name, where the .so lands, builder) for the extension set of the release configuration, in load order."""
    import importlib

    k = ROOT / "kernels"
    ext_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    idx, v4 = Path(os.environ["IDX2F_BUILD"]), Path(os.environ["MOE_V4_BUILD"])
    stub = types.SimpleNamespace(log=print)

    def mod(name, path=None):
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return importlib.import_module(name)

    def route2():                                   # Engine.__init__'s call, verbatim
        from torch.utils.cpp_extension import load
        return load(name="route2_ext", sources=[os.path.join(E.PHYS, "kernels/moe_stream/route2.cu")], verbose=False,
                    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--expt-relaxed-constexpr"])

    def prod():                                     # production's package, located as Engine.prod_moe does
        E.Engine.prod_moe(stub)
        return importlib.import_module("ds41h.attn"), importlib.import_module("ds41h.candidates")

    def v4tag():
        from moe_v4 import v4 as V
        d = V.DEFAULT
        t = "".join(f"_{a.lower()}{b}" for a, b in sorted(d.items()))
        rt = "".join(f"_{V.ROUTE_KEYS[a].lower()}{b}" for a, b in sorted(d.items()) if a in V.ROUTE_KEYS)
        return t, rt

    sys.path.insert(0, str(k))
    t, rt = v4tag()
    X = lambda: mod("idx2f", k / "indexer2f")      # noqa: E731
    head = lambda: mod("vendor_ext", k / "head")   # noqa: E731
    return [
        ("eng_mhcf", ROOT / "engine/build/eng_mhcf/ds41h_phys_mhcf_default.so",
         lambda: mod("mhcf", k / "mhc_fuse").build(build_directory=os.path.join(E.PHYS, "engine/build/eng_mhcf"))),
        ("sp_mhc", k / "sp_layer/build_mhc/default/ds41h_phys_mhc_default.so", lambda: mod("test_mhc").ext()),
        ("sp_comm", k / "sp_layer/build_comm/ds41h_phys_comm.so", lambda: mod("test_comm").ext()),
        ("route2_ext", ext_dir / "route2_ext/route2_ext.so", route2),
        ("moe_raw_ext", ext_dir / "moe_raw_ext/moe_raw_ext.so", lambda: mod("bench").build_extr()),
        ("w8a16_ext", k / "attn_dense/build_w8a16/w8a16_ext.so", lambda: mod("w8a16").build()),
        ("attn_dense", k / "attn_dense/build_pdl/attn_dense_ext_pdl.so", lambda: mod("attn_half")._ext()),
        ("attn_seq_ext", k / "attn_dense/build/attn_seq_ext.so", lambda: mod("attn_half")._ext()),
        ("eng_moe_g6", ROOT / "engine/build/eng_moe_g6/eng_moe_g6s8b2.so", lambda: E.Engine.g6(stub)),
        ("moe_v4_api", v4 / f"v4api{t}/moe_v4_api{t}.so", lambda: E.Engine.moe_v4(stub)),
        ("moe_v4_route", v4 / f"v4route{rt}/moe_v4_route{rt}.so", lambda: E.Engine.moe_v4(stub)),
        ("idx2f_fcand8", idx / "idx2f_fcand8/idx2f_fcand8.so", lambda: X().ext_fcand8()),
        ("idx2f_ftopk", idx / "idx2f_ftopk/idx2f_ftopk.so", lambda: X().ext_ftopk()),
        ("idx2f_topk", idx / "idx2f_topk/idx2f_topk.so", lambda: X().ext_topk()),
        ("idx2f_pf4", idx / "idx2f_pf4/idx2f_pf4.so", lambda: X().ext_pf()),
        ("idx2f_tab", idx / "idx2f_tab/idx2f_tab.so", lambda: mod("tables2f", k / "indexer2f").ext_tab()),
        ("engram_phys_ext", k / "engram/build_ext/engram_phys_ext.so", lambda: mod("engram", k / "engram").ext()),
        ("engram_w8a16_ext", k / "engram/build_w8a16/engram_w8a16_ext.so",
         lambda: mod("engram", k / "engram").w8a16_mod()),
        ("head_comm", k / "head/build_head_comm/head_comm.so", lambda: head().comm()),
        ("head_mhc", k / "head/build_head_mhc/head_mhc.so", lambda: head().mhc()),
        ("head_mhc2", k / "head/build_head_mhc2/head_mhc2.so", lambda: head().mhc2()),
        ("head_dense", k / "head/build_head_dense/head_dense.so", lambda: head().dense()),
        ("head_moe_raw", k / "head/build_head_moe_raw/head_moe_raw.so", lambda: head().moe()),
        ("head_route2", k / "head/build_head_route2/head_route2.so", lambda: head().route2()),
        ("mpf_pm_fused", k / "moe_prefill/build/mpf_pm_fused/mpf_pm_fused.so",
         lambda: importlib.import_module("moe_prefill.pm_fused").ext()),
        ("mpf_pm_swiglu", k / "moe_prefill/build/mpf_pm_swiglu/mpf_pm_swiglu.so",
         lambda: importlib.import_module("moe_prefill.pm_fused").swiglu_ext()),
        ("agrs_comm", k / "agrs/build/agrs_comm.so", lambda: mod("commbench", k / "agrs").ext()),
        ("agrs_pm_rs", k / "agrs/build/agrs_pm_rs/agrs_pm_rs.so", lambda: mod("agrs_pm", k / "agrs").ext()),
        ("ds41h_sparse_decode", ext_dir / "ds41h_sparse_decode/ds41h_sparse_decode.so", lambda: prod()[0].ext()),
        ("ds41h_candidates", ext_dir / "ds41h_candidates/ds41h_candidates.so", lambda: prod()[1].ext()),
    ]


def _async_targets(E):
    """The pipelined decode path (engine/d13.py) adds one extension."""
    def rowexp():
        from engine import d13
        return d13.rowexp_ext()
    return [("eng_rowexp", ROOT / "engine/build/eng_rowexp/eng_rowexp.so", rowexp)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="base of the configurable build directories (default: repo)")
    ap.add_argument("--only", default="", help="comma-separated target names (see --list)")
    ap.add_argument("--list", action="store_true", help="print the targets and exit")
    ap.add_argument("--async", dest="asyn", action="store_true", help="also the pipelined decode path's extension")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "" and not a.list:
        print("prebuild_ext: run on the CPU: CUDA_VISIBLE_DEVICES= (empty)", file=sys.stderr)
        return 2
    stray = sorted(k_ for k_ in os.environ if k_.startswith(("MHC_", "SPC_", "MS3_", "MSR_", "MS_BUILD_DIR", "W8A16_DEFS",
                                                             "W8A16_BUILD", "ATTN2B_BUILD", "IDX2F_PF_EXT")))
    if stray:                                     # these select differently named builds in the loaders
        print(f"prebuild_ext: unset {stray}: they change which builds the loaders select", file=sys.stderr)
        return 2
    root = a.root.resolve()
    env = _env(root)
    os.environ.update(env)
    os.environ.setdefault("PHYS_ROOT", str(ROOT))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("ENG_MHCF", "1")
    sys.path.insert(0, str(ROOT))
    from engine import engine as E               # paths and the engine's environment defaults
    if "DS41H_DEPLOY" not in os.environ:          # the production package's parent directory, if `import ds41h` finds it
        import ds41h
        for p in list(ds41h.__path__):
            if (Path(p) / "cuext.py").exists():
                os.environ["DS41H_DEPLOY"] = env["DS41H_DEPLOY"] = str(Path(p).parent)
                break
    targets = _targets(E) + (_async_targets(E) if a.asyn else [])
    if a.only:
        want = set(a.only.split(","))
        unknown = want - {n for n, _, _ in targets}
        if unknown:
            print(f"prebuild_ext: unknown targets {sorted(unknown)}", file=sys.stderr)
            return 2
        targets = [t for t in targets if t[0] in want]
    if a.list:
        for n, so, _ in targets:
            print(f"{n:22s} {so}")
        return 0
    for k_, v in env.items():
        Path(v).mkdir(parents=True, exist_ok=True)
    (root / "build").mkdir(parents=True, exist_ok=True)
    (root / "build/prebuilt.env").write_text("".join(f"export {k_}={v}\n" for k_, v in env.items()))
    t00, bad = time.time(), []
    for n, so, fn in targets:
        t0 = time.time()
        try:
            fn()
            ok = so.exists()
            msg = "ok" if ok else f"built, but {so} is missing"
        except Exception as e:                    # report every target, then fail
            ok, msg = False, f"{type(e).__name__}: {e}"
        if not ok:
            bad.append(n)
        print(f"prebuild_ext: {n:22s} {time.time() - t0:7.1f} s  {msg}", flush=True)
    print(f"prebuild_ext: {len(targets) - len(bad)} / {len(targets)} built in {time.time() - t00:.0f} s; "
          f"environment for serving: {root / 'build/prebuilt.env'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
