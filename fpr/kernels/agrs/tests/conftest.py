"""Fixtures for the agrs tests: the AG / RS fused paths the serving engine installs (agrs_engine.py, agrs_decode.py) and their
kernels (agrs.cu: combine_push_dm, ce_copy; pm_rs.cu: the production MoE layer with the RS push fused into its combine).

The references are the engine's own default path: kernels/sp_layer/comm.cu (AG / RS), kernels/moe_stream's moe_raw
combine (decode MoE), kernels/moe_prefill's pm_fused layer (production MoE), plus torch re-statements of the arithmetic.
Expert weights are synthetic MXFP4 (random e2m1 bytes, ue8m0 scale bytes 118..124), so no checkpoint is needed.
Every test is marked `gpu`; the fused path tests also need 4 GPUs with peer access, driven from one process as the engine
drives them.
"""
from __future__ import annotations

import os
import sys
import time
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)                        # kernels/agrs
KROOT = os.path.dirname(KDIR)                       # kernels
ROOT = os.path.dirname(KROOT)
for _p in (os.path.join(KROOT, "moe_stream"), os.path.join(KROOT, "sp_layer"), KROOT, ROOT, KDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
W, D = 4, 5120
EXPERTS, INTER = 384, 576                           # routed experts; MoE intermediate size per tensor-parallel rank


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


def _load_default(prefixes, loader):
    """loader() with the environment variables that select experimental build variants hidden (the served build)."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(prefixes)}
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
    import test_comm                                # kernels/sp_layer: the engine's collectives
    return _load_default(("SPC_",), test_comm.ext)


@pytest.fixture(scope="session")
def agrs(dev):
    import commbench                                # kernels/agrs: agrs.cu, as agrs_engine / agrs_decode load it
    return commbench.ext()


@pytest.fixture(scope="session")
def moe_raw(dev):
    """The decode MoE's extension (its combine is the call the decode RS fused path replaces)."""
    try:
        import bench
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"kernels/moe_stream not importable ({type(ex).__name__}: {ex})")
    return _load_default(("MS3_", "MSR_", "MS_BUILD_DIR"), bench.build_extr)


@pytest.fixture(scope="session")
def PM(dev):
    """Production's MoE module exactly as the engine resolves it (Engine.prod_moe: deploy package, production env)."""
    try:
        import engine.engine as EE
        return EE.Engine.prod_moe(types.SimpleNamespace(log=lambda *a, **k: None))
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"the engine's production MoE is not importable ({type(ex).__name__}: {ex})")


@pytest.fixture(scope="session")
def peers(comm, agrs, devs):
    comm.enable_peers(W)
    torch.cuda.set_device(devs[0])                  # enable_peers leaves the last device current
    return devs


@pytest.fixture(scope="session")
def experts():
    """experts(dev, seed) -> dict(w13, s13, w2, s2, sw13, sw2): one rank's synthetic expert slices (cached)."""
    cache = {}

    def make(d, seed):
        key = (str(d), seed)
        if key not in cache:
            g = torch.Generator(device=d).manual_seed(seed)

            def u8(shape, lo, hi):
                return torch.randint(lo, hi, shape, dtype=torch.uint8, device=d, generator=g)
            cache[key] = dict(
                w13=u8((EXPERTS, 2 * INTER, D // 2), 0, 256), s13=u8((EXPERTS, 2 * INTER, D // 32), 118, 125),
                w2=u8((EXPERTS, D, INTER // 2), 0, 256), s2=u8((EXPERTS, D, INTER // 32), 118, 125),
                sw13=(torch.randn(2 * INTER, D, generator=g, device=d) * 0.02).bfloat16(),
                sw2=(torch.randn(D, INTER, generator=g, device=d) * 0.04).bfloat16())
        return cache[key]
    return make


@pytest.fixture(scope="session")
def sync():
    """sync(streams_or_devices, timeout_s): wait for the work issued so far; stops the session if a flag wait never
    finishes."""
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
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


@pytest.fixture(scope="session")
def same():
    return bits_equal


# ---- shared data builders (fixtures returning functions: the test modules stay free of path-dependent imports)
def rs_ref(slots):
    """comm.cu k_rs_sum's arithmetic: fp32 sum in slot order 0..3, one bf16 rounding."""
    a = slots[0].float()
    for t in slots[1:]:
        a = a + t.float()
    return a.bfloat16()


def combine_ref(Y, slotmap, sh):
    """The decode MoE combine's arithmetic: the 6 slot rows of Y in slot order (fp32, missing slot = 0), then the
    shared-expert addend, one bf16 rounding."""
    acc = torch.zeros(slotmap.shape[0], Y.shape[1], dtype=torch.float32, device=Y.device)
    for k in range(slotmap.shape[1]):
        sl = slotmap[:, k].long()
        v = Y.index_select(0, sl.clamp_min(0)).float()
        acc = acc + torch.where((sl >= 0)[:, None], v, torch.zeros((), device=Y.device))
    return (acc + sh.float()).bfloat16()


def slotmap(T, nv, R, g, d):
    """Distinct slot rows for the valid rows (owner q's rows [q R, q R + nv[q])); -1 for padding rows and ~2 % of the
    valid rows' slots."""
    sm = torch.randperm(6 * T, generator=g, device=d).to(torch.int32).view(T, 6)
    lr = torch.arange(T, device=d) % R
    q = torch.arange(T, device=d) // R
    valid = lr < torch.tensor(nv, device=d)[q]
    drop = torch.rand(T, 6, generator=g, device=d) < 0.02
    return torch.where(valid[:, None] & ~drop, sm, torch.full_like(sm, -1)).contiguous()


def route_rows(R, nvalid, g, d):
    """One rank's 64-B routing rows: 6 distinct expert ids (skewed popularity, sampled without replacement: no ties) and
    6 fp32 weights for the first nvalid rows; padding rows get ids -1 and weights 0, as the engine's router writes them."""
    pop = 1.0 / torch.arange(1, EXPERTS + 1, device=d).float() ** 0.7
    pop = pop[torch.randperm(EXPERTS, generator=g, device=d)]
    ids = torch.multinomial(pop.expand(R, EXPERTS), 6, replacement=False, generator=g).to(torch.int32)
    wts = torch.rand(R, 6, generator=g, device=d)
    wts = (wts / wts.sum(1, keepdim=True) * 1.5).float()
    ids[nvalid:] = -1
    wts[nvalid:] = 0
    rr = torch.zeros(R, 64, dtype=torch.uint8, device=d)
    rr[:, :24] = ids.view(torch.uint8).view(R, 24)
    rr[:, 24:48] = wts.view(torch.uint8).view(R, 24)
    return rr


@pytest.fixture(scope="session")
def ref():
    return types.SimpleNamespace(rs=rs_ref, combine=combine_ref, slotmap=slotmap, route_rows=route_rows)
