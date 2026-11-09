"""pm2: pm_fused's extension (pm2.cu = pm_fused.cu + layer8's decode-row sub-call variant vdec) built against a private copy of production's MoE sources with one
added hybrid remainder engine: set_hybrid(S, engine) with engine = a small16k GEMM variant (18..23, 26..31; moe_prefill.cu
launch_small16k_v) runs each expert's <= S-row remainder tile on moe_small16k_kernel instead of moe_gemm16_kernel
(engine 0) / moe_small16_kernel (engine 1). small16k processes a remainder in NT-row chunks, so S may exceed 128 (the
gemm16 engine takes one <= 128-row tile per expert). Same per-element math (production: small16k == 0x500 == LD 4).

Separate extension (mpf_pm2, build/mpf_pm2, sources build/pm2_src) so the engine's mpf_pm_fused build is never touched.
  CUDA_VISIBLE_DEVICES= python3 pm2.py      # build on the CPU
"""
from __future__ import annotations

import functools
import os

try:                                   # engine: from moe_prefill import pm2 (kernels/ on sys.path)
    from . import pm_fused as P
except ImportError:                    # scripts in this directory
    import pm_fused as P

SRC2 = P.HERE / "build" / "pm2_src"
BUILD2 = P.HERE / "build" / "mpf_pm2"
NAME2 = "mpf_pm2"
HYB_ORIG = "    if (g_hybrid_engine == 1 && small16_ok(mode, a2)) {\n"
HYB_PATCH = ("    if (small16k_variant(g_hybrid_engine) && small16_ok(mode, a2)) {  // mpf_pm2: small16k remainder engine\n"
             "      if (mode == 1)\n"
             "        launch_small16k_v<1>(g_hybrid_engine, a2, R);\n"
             "      else\n"
             "        launch_small16k_v<2>(g_hybrid_engine, a2, R);\n"
             "    } else if (g_hybrid_engine == 1 && small16_ok(mode, a2)) {\n")


def sync_sources2():
    pme = P._load_file("mpf_prod_moe_ext2", P.PHYS / "engine" / "prod_moe_ext.py")
    SRC2.mkdir(parents=True, exist_ok=True)
    for f in sorted((P.DEPLOY / "kernels").iterdir()):
        if f.suffix not in (".cu", ".cuh", ".h", ".hpp") or not f.is_file():
            continue
        data = f.read_bytes()
        if f.name == "host.cuh":
            txt = data.decode()
            assert txt.count(pme.ORIG) == 1
            data = txt.replace(pme.ORIG, pme.PATCH).encode()
        elif f.name == "moe_prefill.cu":
            txt = data.decode()
            assert txt.count(P.BIND_ORIG) == 1 and txt.count(HYB_ORIG) == 1, "deploy moe_prefill.cu changed: re-check pm2"
            data = txt.replace(P.BIND_ORIG, P.BIND_PATCH).replace(HYB_ORIG, HYB_PATCH).encode()
        P._write_if_changed(SRC2 / f.name, data)
    return SRC2


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    cuext = P._load_file("mpf_pm2_cuext", P.DEPLOY / "cuext.py")
    src = sync_sources2()
    BUILD2.mkdir(parents=True, exist_ok=True)
    return load(name=NAME2, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu"), str(P.HERE / "pm2.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS), extra_include_paths=[str(src)],
                build_directory=str(BUILD2), verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


if __name__ == "__main__":
    ext()
    P.swiglu_ext()
    print(f"{NAME2} built in {BUILD2}", flush=True)


class Fused2(P.Fused):
    """pm_fused.Fused on the pm2 extension: production's variant, hybrid split S on remainder engine `engine` (0 = moe_gemm16,
    small16k variant = pm2's engine), layer8's decode-row sub-call on small16k variant `vdec` (0 = the W4A16 variant)."""

    def __init__(self, PM=None, S=None, engine=None, vdec=0):
        if PM is None:
            from ds41h import moe as PM
        self.variant, hs, he = P.prod_settings(PM)
        self.e = ext()
        self.hybrid = (int(hs if S is None else S), int(he if engine is None else engine))
        self.e.set_hybrid(*self.hybrid)
        self.e.set_swiglu(P.swiglu_ext().launcher())
        self.vdec = int(vdec)

    def layer8(self, fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit=10.0, v8=P.ACT8_VARIANT):
        self.e.layer8(fall, gr, nv, nd, R, vcount, ndec, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, self.variant, v8,
                      self.vdec)
