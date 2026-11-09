"""Production's routed-MoE extension (the ds41h package's ds41h_moe: moe_route.cu + moe_prefill.cu) for ONE process
driving 4 GPUs.

Why: production's kernels/host.cuh opt_in_smem() sets a kernel's dynamic shared-memory attribute ONCE PER PROCESS
(`static const bool done`), but cudaFuncSetAttribute applies to the current device only. vLLM runs one process per GPU,
so production never sees it; in the engine the first GPU that launches a moe_gemm kernel gets the attribute and the
other three fail with "moe_gemm launch failed: invalid argument" (seen on a prefill step with > 2,560 valid rows).
Same class of bug as idx2f_cand's smem attribute.

Fix: a private copy of the ds41h package's kernel sources (read-only originals, copied into engine/build), host.cuh's
opt_in_smem patched to remember the attribute per device, built as `eng_ds41h_moe`. Kernels, launch configurations and
arithmetic are untouched (bit-identical results); the other per-process statics in these files are device properties
(SM count, co-resident clusters, a driver entry point), equal on the 4 identical H200s.

  ext()       the patched extension (production's ds41h.moe.ext() is pointed here by Engine.prod_moe)
  build()     copy + patch + build on the CPU (before any GPU work)
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPLOY = Path(os.environ.get("DS41H_DEPLOY", os.path.dirname(PHYS))) / "ds41h"
SRC = Path(PHYS) / "engine" / "build" / "eng_prod_moe_src"
BUILD = Path(PHYS) / "engine" / "build" / "eng_ds41h_moe"
NAME = "eng_ds41h_moe"

ORIG = """  static const bool done = [bytes] {
    TORCH_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    return true;
  }();
  (void)done;"""
PATCH = """  // eng_ds41h_moe: the attribute is per DEVICE; one process drives 4 GPUs
  static bool done[64] = {};
  int dev = 0;
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64);
  if (!done[dev]) {
    TORCH_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    done[dev] = true;
  }"""


def _write_if_changed(p: Path, data: bytes) -> None:
    if not p.exists() or p.read_bytes() != data:
        p.write_bytes(data)


def sync_sources() -> Path:
    """Copy the ds41h package's kernels/*.cu|*.cuh|*.h into SRC (only files whose bytes changed: torch's build hash), with
    host.cuh patched; asserts the patch site exists exactly once."""
    SRC.mkdir(parents=True, exist_ok=True)
    kd = DEPLOY / "kernels"
    for f in sorted(kd.iterdir()):
        if f.suffix not in (".cu", ".cuh", ".h", ".hpp") or not f.is_file():
            continue
        data = f.read_bytes()
        if f.name == "host.cuh":
            txt = data.decode()
            assert txt.count(ORIG) == 1, "deploy host.cuh opt_in_smem changed: re-check engine/prod_moe_ext.py"
            data = txt.replace(ORIG, PATCH).encode()
        _write_if_changed(SRC / f.name, data)
    return SRC


@functools.cache
def ext():
    import importlib.util

    from torch.utils.cpp_extension import load

    spec = importlib.util.spec_from_file_location("eng_prod_cuext", str(DEPLOY / "cuext.py"))
    cuext = importlib.util.module_from_spec(spec)      # production's build flags (read-only; stdlib-only module)
    spec.loader.exec_module(cuext)

    src = sync_sources()
    BUILD.mkdir(parents=True, exist_ok=True)
    return load(name=NAME, sources=[str(src / "moe_route.cu"), str(src / "moe_prefill.cu")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=list(cuext.NVCC_FLAGS),
                extra_include_paths=[str(src)], build_directory=str(BUILD),
                verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


def build() -> None:
    ext()


if __name__ == "__main__":
    build()
    print(f"{NAME} built in {BUILD}", flush=True)
