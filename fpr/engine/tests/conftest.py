"""Shared setup for the engine regression tests.

The engine reads its feature flags from the environment when `engine.engine` is imported, so the release's flag set is
set here (setdefault: an explicit environment wins) before any test module imports the engine.

  CPU tests:  CUDA_VISIBLE_DEVICES= python -m pytest engine/tests -m "not gpu"
  GPU tests:  python -m pytest engine/tests/<one module>       (4 GPUs; one engine per process, see drive.engine_once)

GPU tests are marked `gpu` and skipped when CUDA (or the GPU count they need) is missing. Every engine-level GPU module
builds ONE engine for all of its tests; run each such module in its own process (a second engine in the same process
would need a second copy of the weights in GPU memory).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]          # the repository: engine/tests/conftest.py -> ../..
for p in (ROOT, ROOT / "serve"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("PHYS_ROOT", str(ROOT))
os.environ.setdefault("IDX2F_BUILD", str(ROOT / "engine/build/idx2f"))
os.environ.setdefault("MOE_V4_BUILD", str(ROOT / "engine/build/moe_v4"))
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# The release's engine flag set (16-bit arithmetic throughout). Every flag below is bitwise-neutral against the
# engine's reference path; they select faster kernels and launch layouts.
RELEASE_FLAGS = {
    "ENG_AGRS_RSF": "1", "ENG_AGRS_SUM528": "1", "ENG_AGRS_AGCE": "1", "ENG_QB16_MIXED": "1", "ENG_ATTN_PF": "1",
    "ENG_MHCF": "1", "ENG_QB16_DECODE": "1", "IDX2F_PF": "1", "ENG_IDX_FUSE": "1", "ENG_IDX_SIDE": "1",
    "ENG_IDX_F8": "1", "ENG_G6": "1", "ENG_AGRS_RSD": "1", "ENG_LAUNCH_SPLIT": "1", "ENG_LONG_GRAPHS": "1",
    "IDX2F_CANON_STOCK": "1",
}
for k, v in RELEASE_FLAGS.items():
    os.environ.setdefault(k, v)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu(n): needs CUDA and n GPUs (default 4); skipped otherwise")


def gpu_count() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return 0
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    have = None
    for it in items:
        m = it.get_closest_marker("gpu")
        if m is None:
            continue
        need = int(m.kwargs.get("n", m.args[0] if m.args else 4))
        if have is None:
            have = gpu_count()
        if have < need:
            it.add_marker(pytest.mark.skip(reason=f"needs CUDA with {need} GPU(s), found {have}"))

