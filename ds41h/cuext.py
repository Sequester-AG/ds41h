"""JIT build helper for ds41h CUDA extensions (sm_90a) and CUDA-IPC buffer exchange.

Extensions are built with torch.utils.cpp_extension.load into TORCH_EXTENSIONS_DIR
($DS41H_HOME/cache/torch_extensions via scripts/env.sh). Sources live in ds41h/kernels.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

KERNELS = Path(__file__).parent / "kernels"

NVCC_FLAGS = [
    "-O3",
    "-std=c++20",
    "--use_fast_math",
    "-gencode=arch=compute_90a,code=sm_90a",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "-Xptxas=-v" if os.environ.get("DS41H_PTXAS_VERBOSE") else "-Xptxas=-O3",
]


@functools.cache
def load(name: str, *sources: str, extra_cuda_cflags: tuple[str, ...] = ()):
    """Build (or reuse) extension `name` from files in ds41h/kernels."""
    from torch.utils.cpp_extension import load as _load

    return _load(
        name=name,
        sources=[str(KERNELS / s) for s in sources],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=NVCC_FLAGS + list(extra_cuda_cflags),
        extra_include_paths=[str(KERNELS)],
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")),
    )


def exchange_ipc_handles(handle: bytes, cpu_group) -> list[bytes]:
    """All-gather a cudaIpcMemHandle (64 bytes) across the TP CPU group; rank order."""
    import torch.distributed as dist

    out: list = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(out, handle, group=cpu_group)
    return out
