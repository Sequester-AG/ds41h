"""Tests + micro-benchmark for ds41h.grouped_decode (4 decode rows x 16 heads per FlashMLA 64-row tile over the key-set union;
GPU only (sm_90a), stock FlashMLA from vLLM 0.32.0 as the comparison.

  python -m pytest tests/test_grouped_decode.py -x -q
  python tests/test_grouped_decode.py --bench          # us/call + mJ/call (NVML) vs stock FlashMLA, 16 / 64 requests x 8 rows

Contract: every row attends exactly to its own keys (SWA window + top-k): within the fp32-summation-order bound of the fp64
reference (tests/test_attn.py ref_decode / assert_close_ref) and close to stock FlashMLA; correct for rows of different
requests in one group, padded (invalid) rows, short contexts, holes in the top-k; bit-repeatable incl. CUDA-graph replay.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import test_attn as ta  # noqa: E402

from ds41h import grouped_decode as gd  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

HAS = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _skip(cond, why):
    return pytest.mark.skipif(cond, reason=why) if pytest is not None else (lambda f: f)


def _param(n, v):
    return pytest.mark.parametrize(n, v) if pytest is not None else (lambda f: f)


gpu = _skip(not HAS, "needs sm_90a")


def run_grouped(case, out=None):
    """Grouped decode of a test_attn case (T padded to a multiple of 4 with invalid rows by the caller)."""
    T = case["T"]
    q16 = case["q"][:, : ta.HEADS].contiguous()
    out16 = torch.empty(T, ta.HEADS, ta.DIM, dtype=torch.bfloat16, device=q16.device) if out is None else out
    sidx = case["swa_idx"].reshape(T, -1)
    su, sm, sl = gd.union(sidx, case["swa_lens"], gd._width(sidx.shape[1]), case["swa"].shape[0] * case["swa"].shape[1])
    extra = None
    if "ex" in case:
        eidx = case["ex_glob"].reshape(T, -1)
        eu, em, el = gd.union(eidx, case["ex_lens"], gd._width(eidx.shape[1]), case["ex"].shape[0] * case["ex"].shape[1])
        extra = (case["ex"].unsqueeze(-2), eu, el, em)
    gd.decode(q16, case["swa"].unsqueeze(-2), su, sl, sm, case["sink"], ta.SCALE, out16, extra)
    return out16


CASES = {
    "c16x8_r1": ([2048] * 16, [8] * 16, dict(ratio=1)),
    "c4x8_r2_holes": ([1900] * 4, [8] * 4, dict(ratio=2, holes=True)),
    "short_ctx": ([40, 300, 41, 120], [8, 8, 8, 8], dict(ratio=2)),       # < 512 compressed, < 128 window
    "mixed_reqs": ([700, 1200, 900], [6, 6, 4], dict(ratio=1)),            # groups straddle requests
    "invalid_rows": ([800, 1000], [8, 8], dict(ratio=1, invalid_rows=(3, 9, 15))),
    "swa_only": ([300, 2000], [8, 8], dict(topk=0)),
    "sink_big_peaky": ([900] * 2, [8] * 2, dict(ratio=2, sink="big", q_gain=6.0)),
    "sink_none": ([1300] * 2, [8] * 2, dict(ratio=1, sink="none")),
}


@gpu
@_param("name", list(CASES))
def test_vs_reference_and_stock(name):
    seq, ql, kw = CASES[name]
    case = ta.make_case(seq, ql, seed=23, use_stock_writer=False, **kw)
    assert case["T"] % 4 == 0
    ours = run_grouped(case)
    torch.cuda.synchronize()
    ref = ta.ref_decode(case)
    ta.assert_close_ref(ours, ref, f"{name}/grouped")
    if ta.flashmla_available():
        stock = ta.run_flashmla(case)
        torch.cuda.synchronize()
        ta.assert_close_ref(stock, ref, f"{name}/stock")
        ta.assert_close_ref(ours, (stock[:, : ta.HEADS].double(), ref[1]), f"{name}/grouped-vs-stock", rtol=2e-2)


@gpu
def test_sink_cache_not_stale():
    """_sink64 never returns the values of a freed sink for a new tensor at the same address, and follows in-place updates
    (test_vs_reference_and_stock[short_ctx] failed after test_gdec_share.py in the same process)."""
    for i in range(8):
        s = torch.full((64,), float(i) - 3.5, device="cuda")
        got = gd._sink64(s)
        assert torch.equal(got, s[:16].float().repeat(gd.GQ)), i
        del s, got
    s = torch.zeros(64, device="cuda")
    gd._sink64(s)
    s.add_(1.25)
    assert torch.equal(gd._sink64(s), s[:16].float().repeat(gd.GQ))


@gpu
def test_union_builder():
    T, W = 16, 128
    g = torch.Generator(device="cuda").manual_seed(5)
    idx = torch.randint(0, 3000, (T, W), generator=g, device="cuda", dtype=torch.int32)
    idx[2, 5] = -1
    lens = torch.randint(0, W + 1, (T,), generator=g, device="cuda", dtype=torch.int32)
    u, m, n = gd.union(idx, lens, gd.SWA_WIDTH, 3000)
    for grp in range(T // 4):
        want = {}
        for r in range(4):
            row = 4 * grp + r
            for j in range(int(lens[row])):
                v = int(idx[row, j])
                if v >= 0:
                    want[v] = want.get(v, 0) | (1 << r)
        keys = sorted(want)
        assert int(n[grp]) == len(keys), (grp, int(n[grp]), len(keys))
        assert u[grp, : len(keys)].tolist() == keys
        assert m[grp, : len(keys)].tolist() == [want[k] for k in keys]


@gpu
def test_bit_repeatable_and_graph():
    case = ta.make_case([2048] * 8, [8] * 8, ratio=1, seed=31, use_stock_writer=False)
    y0 = run_grouped(case).clone()
    side = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for i in range(10):
        if i % 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                a @ a
        assert torch.equal(run_grouped(case), y0), i
        torch.cuda.current_stream().wait_stream(side)
    out = torch.empty_like(y0)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        run_grouped(case, out=out)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        run_grouped(case, out=out)
    for _ in range(3):
        out.zero_()
        gr.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, y0)


def _energy_handle():
    import pynvml

    pynvml.nvmlInit()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        pci = pynvml.nvmlDeviceGetPciInfo(h)
        if pci.bus == props.pci_bus_id and pci.device == props.pci_device_id and pci.domain == props.pci_domain_id:
            return pynvml, h
    return None, None


def _measure(fn, secs, nv, h):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h) if nv else 0
    a.record()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        for _ in range(8):
            fn()
        n += 8
    b.record()
    b.synchronize()
    e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h) if nv else 0
    s = a.elapsed_time(b) / 1e3
    return s / n * 1e6, (e1 - e0) / n, (e1 - e0) / 1e3 / s


def bench(secs=2.0):
    nv, h = _energy_handle()
    for nreq, ratio, ctx in ((16, 1, 2048), (64, 1, 2048), (64, 2, 2048), (16, 1, 16384)):
        case = ta.make_case([ctx] * nreq, [8] * nreq, ratio=ratio, seed=41, use_stock_writer=False)
        out = torch.empty(case["T"], ta.HEADS, ta.DIM, dtype=torch.bfloat16, device="cuda")
        su, sm, sl = gd.union(case["swa_idx"].reshape(case["T"], -1), case["swa_lens"], gd._width(case["swa_idx"].shape[-1]),
                              case["swa"].shape[0] * case["swa"].shape[1])
        line = f"{nreq:3d} req x 8 rows ratio {ratio} ctx {ctx:6d}: union |swa| {float(sl.float().mean()):5.1f}"
        if "ex" in case:
            eu, em, el = gd.union(case["ex_glob"].reshape(case["T"], -1), case["ex_lens"], gd._width(case["ex_glob"].shape[-1]),
                                  case["ex"].shape[0] * case["ex"].shape[1])
            line += f" |topk| {float(el.float().mean()):6.1f} (of {4 * 512})"
        print(line, flush=True)
        res = {}
        if ta.flashmla_available():
            res["stock FlashMLA (64 padded heads, planner each call)"] = _measure(lambda: ta.run_flashmla(case), secs, nv, h)
        res["grouped incl. unions + planner"] = _measure(lambda: run_grouped(case, out=out), secs, nv, h)
        q16 = case["q"][:, : ta.HEADS].contiguous()
        extra = (case["ex"].unsqueeze(-2), eu, el, em) if "ex" in case else None
        res["grouped kernel only (+ planner)"] = _measure(
            lambda: gd.decode(q16, case["swa"].unsqueeze(-2), su, sl, sm, case["sink"], ta.SCALE, out, extra), secs, nv, h)
        for k, (us, mj, w) in res.items():
            print(f"  {k:52s} {us:8.1f} us  {mj:7.3f} mJ  {w:5.0f} W", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--secs", type=float, default=2.0)
    a = ap.parse_args()
    if a.bench:
        bench(a.secs)
    else:
        sys.exit(pytest.main([__file__, "-x", "-q"]) if pytest else 0)
