"""Long-row top-k on 4 GPUs at once: each device must use its own radix workspace.

Bug guarded: the indexer's stock top-k (rows past 262,144 positions, eager and graphed) takes its 1 MB radix workspace
from vLLM's process-global WorkspaceManager, which is created on the first device. All four ranks then ran the
persistent top-k on one device-0 buffer at the same time. Persistent kernels coordinate their CTAs through that
buffer, so concurrent ranks corrupted each other: wrong top-512 sets at low load and, in serving, a crash.

The fix (engine/hooks_idx.py, Idx2f._wrap_stock_topk) gives every device its own locked manager and swaps it in per
call; the global manager gets a canary that raises when another device asks it for workspace.

Inputs are tie-free: each row's logits are a rotation of one random permutation of 0 .. W-1 (distinct, exact in fp32),
so the top-512 set of every row is unique and an exact CPU top-k is the reference.
"""
from __future__ import annotations

import types

import pytest

W = 655_360                                  # logits width (the longest context)
K = 512
MIN_LEN = 262_145                            # every row is past the fast top-k's 262,144 positions


def _setup():
    import torch
    from engine import hooks_idx as HI
    import idx2f as X                        # kernels/indexer2f (hooks_idx puts it on sys.path)
    devs = [torch.device("cuda", i) for i in range(4)]
    for i in devs:                           # peer access between every pair, as the engine's collectives enable it
        for j in devs:
            if i != j:
                with torch.cuda.device(i):
                    torch.zeros(4, device=i).copy_(torch.ones(4, device=j))
    for d in devs:
        torch.cuda.synchronize(d)
    hooks = types.SimpleNamespace()
    HI.Idx2f._wrap_stock_topk(hooks, X, types.SimpleNamespace(devs=devs))
    return X, devs, hooks


@pytest.fixture(scope="module")
def topk():
    return _setup()


def _inputs(devs, rows: int, seed: int):
    import torch
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in devs:
        perm = torch.randperm(W, generator=g).to(torch.float32)
        shifts = torch.randint(0, W, (rows,), generator=g)
        lg = torch.stack([perm.roll(int(s)) for s in shifts])                     # [rows, W], distinct per row
        sl = torch.randint(MIN_LEN, W + 1, (rows,), generator=g, dtype=torch.int32)
        ref = []
        for i in range(rows):                                                     # exact reference on the CPU
            n = int(sl[i])
            ref.append(torch.topk(lg[i, :n], K).indices.sort().values.to(torch.int32))
        out.append(dict(lg=lg.to(d), sl=sl.to(d), out=torch.full((rows, K), -1, dtype=torch.int32, device=d),
                        mx=int(sl.max()), ref=torch.stack(ref)))
    return out


def _canon(t):
    import torch
    return torch.where(t >= 0, t, torch.full_like(t, 1 << 30)).sort(dim=1).values.cpu()


@pytest.mark.gpu(n=4)
@pytest.mark.parametrize("rows", [8, 32, 128])
def test_concurrent_long_row_topk_is_exact(topk, rows):
    """All 4 devices launch the persistent top-k at once (rows per device = 8 / 32 / 128): every row's set equals the
    exact top-512, on every repeat, with no device fault."""
    import torch
    X, devs, _ = topk
    inp = _inputs(devs, rows, seed=rows)
    bad = 0
    for _ in range(10):
        for i, d in enumerate(devs):                       # launch on every device, then wait for all
            with torch.cuda.device(d):
                inp[i]["out"].fill_(-1)
                X.stock_topk(inp[i]["lg"], inp[i]["sl"], inp[i]["out"], inp[i]["mx"], "persistent")
        for d in devs:
            torch.cuda.synchronize(d)
        for i in range(len(devs)):
            bad += int((_canon(inp[i]["out"]) != inp[i]["ref"]).any(1).sum())
    assert bad == 0, f"{bad} rows differ from the exact top-{K} (rows per device {rows})"


@pytest.mark.gpu(n=4)
def test_each_device_has_its_own_workspace(topk):
    """One locked manager per device, each on its own device."""
    _, devs, hooks = topk
    mgrs = hooks.topk_ws
    assert sorted(mgrs) == [d.index for d in devs]
    assert len({id(m) for m in mgrs.values()}) == len(devs)
    for idx, m in mgrs.items():
        assert m._device.index == idx


@pytest.mark.gpu(n=4)
def test_global_manager_refuses_other_devices(topk):
    """The canary: the process-global manager (created on device 0) raises when device 1 asks it for workspace."""
    import torch
    from vllm.model_executor.layers import indexer_topk as IT
    from vllm.v1.worker import workspace as WS
    g = WS._manager
    assert g is not None and getattr(g, "_eng_canary", False)
    with torch.cuda.device(1):
        with pytest.raises(AssertionError, match="cross-device workspace"):
            g.get_simultaneous(((IT.RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8))
