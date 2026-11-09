"""Fixtures for the sp_layer kernel tests (comm.cu: the sequence-parallel collectives; mhc.cu: the fused mHC boundary).

The kernels are compiled for sm_90a. Every test is marked `gpu` and skips without such a device; the collective tests
also need 4 GPUs with peer access, driven from one process (one stream per device), as the engine drives them.
The extensions load through the modules the engine itself uses (test_comm.ext, test_mhc.ext) with their default build
options: SPC_* / MHC_* variables, which select experimental build variants, are hidden while loading.
"""
from __future__ import annotations

import os
import sys
import time

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)                         # kernels/sp_layer
if KDIR not in sys.path:
    sys.path.insert(0, KDIR)
W = 4


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs an sm_90a CUDA device (skipped without one)")


def _sm90(i: int) -> bool:
    return torch.cuda.get_device_capability(i) == (9, 0)


def _have_gpu() -> bool:
    return torch.cuda.is_available() and _sm90(0)


def _have_4() -> bool:
    if not _have_gpu() or torch.cuda.device_count() < W or not all(_sm90(i) for i in range(W)):
        return False
    return all(torch.cuda.can_device_access_peer(a, b) for a in range(W) for b in range(W) if a != b)


def _load_default(prefix: str, loader):
    """loader() with every <prefix>* environment variable hidden (the default build, as served)."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(prefix)}
    try:
        return loader()
    finally:
        os.environ.update(saved)


@pytest.fixture(scope="session")
def dev():
    if not _have_gpu():
        pytest.skip("needs an sm_90a CUDA device")
    return torch.device("cuda", 0)


@pytest.fixture(scope="session")
def devs():
    if not _have_4():
        pytest.skip("needs 4 sm_90a GPUs with peer access")
    return [torch.device("cuda", i) for i in range(W)]


@pytest.fixture(scope="session")
def comm(dev):
    import test_comm
    return _load_default("SPC_", test_comm.ext)


@pytest.fixture(scope="session")
def mhc(dev):
    import test_mhc
    return _load_default("MHC_", test_mhc.ext)


@pytest.fixture(scope="session")
def peers(comm, devs):
    comm.enable_peers(W)
    torch.cuda.set_device(devs[0])                    # enable_peers leaves the last device current
    return devs


@pytest.fixture(scope="session")
def sync():
    """sync(streams_or_devices, timeout_s): wait for the work queued so far. A spin-wait whose flag never reaches its
    target would hang the process, so this polls with a deadline and stops the session if it passes."""
    def _sync(items, timeout: float = 60.0):
        evs = []
        for it in items:
            st = it if isinstance(it, torch.cuda.Stream) else torch.cuda.current_stream(it)
            e = torch.cuda.Event()
            e.record(st)
            evs.append(e)
        t0 = time.monotonic()
        while not all(e.query() for e in evs):
            if time.monotonic() - t0 > timeout:
                pytest.exit(f"GPU work did not finish within {timeout:.0f} s (a flag wait never reached its target)",
                            returncode=3)
            time.sleep(0.0005)
    return _sync


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (NaN payloads and signed zeros included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


@pytest.fixture(scope="session")
def same():
    return bits_equal
