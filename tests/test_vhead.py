"""tests for ds41h/vhead.py (lossless 12-bit verification head). 1 GPU.

  python tests/test_vhead.py            correctness (real head.weight shards + synthetic escape stress) + bench
  python tests/test_vhead.py --nobench  correctness only
Real weights: DS41H_MODEL_DIR (default $DS41H_HOME/model), tensor head.weight, TP shard r = rows
[r * 32320, (r + 1) * 32320).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ds41h import vhead  # noqa: E402

MODEL = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
N_SHARD = 32320


def load_head_shard(rank: int) -> torch.Tensor | None:
    try:
        from safetensors import safe_open
    except Exception:  # noqa: BLE001
        return None
    idx = os.path.join(MODEL, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    fn = json.load(open(idx))["weight_map"]["head.weight"]
    with safe_open(os.path.join(MODEL, fn), "pt") as f:
        sl = f.get_slice("head.weight")
        w = sl[rank * N_SHARD : (rank + 1) * N_SHARD]
    return w.contiguous().cuda()


def synthetic(n: int = 512, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, 5120, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    # escapes: zeros, subnormal / tiny, |w| >= 2 (E >= 128), a dense escape tile, signs
    w[3, :100] = 0.0
    w[5, 7] = -0.0
    w[17, ::7] = 1e-7
    w[18, 3] = -3.0
    w[19, 4095] = 2.5e3
    w[33, 5000] = 1.0  # E = 127 is a regular code (15)
    w[34, 11] = -1.75
    w[64:80, :200] = 0.0  # 3200 escapes in one tile (pack capacity 4096 per tile)
    return w


def check_unpack(w: torch.Tensor, name: str) -> "vhead.PackedHead":
    p = vhead.PackedHead(w)
    u = p.unpack()
    torch.cuda.synchronize()
    eq = torch.equal(u.view(torch.int16), w.view(torch.int16))
    assert eq, f"{name}: unpack != W at {(u.view(torch.int16) != w.view(torch.int16)).nonzero()[:8].tolist()}"
    print(f"{name}: unpack == W bitwise ({p.n} rows, {p.nesc} escapes, packed {p.bytes / 1e6:.1f} MB vs bf16 {w.numel() * 2 / 1e6:.1f} MB)")
    return p


def check_gemv(p: "vhead.PackedHead", w: torch.Tensor, name: str, seed: int = 1) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    hs = (torch.randn(40, 5120, generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    hs[3, 17] = 60.0  # an outlier channel
    for m in (1, 2, 5, 7, 8):
        idx = torch.randperm(40, generator=g, device="cuda")[:m].to(torch.int64)
        x = hs[idx]
        ref = F.linear(x, w)  # cuBLAS bf16 (stock compute_logits_local)
        r64 = x.double() @ w.double().t()
        for var in (0, 1, 2, 3):
            y = p(hs, idx, m, pdl=False, variant=var)
            y2 = p(x.contiguous(), None, m, pdl=False, variant=var)
            torch.cuda.synchronize()
            assert torch.equal(y, y2), f"{name}: gather vs plain differ (m={m}, var={var})"
            e_y = (y.double() - r64).abs()
            e_r = (ref.double() - r64).abs()
            mag = torch.maximum(y.float().abs(), ref.float().abs()).clamp(min=2.0**-20)
            bf_ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
            ulp = (y.float() - ref.float()).abs() / bf_ulp
            n_bad = int(((y.float() - ref.float()).abs() > 2 * bf_ulp + 1e-4).sum())
            assert n_bad == 0, f"{name}: {n_bad} logits differ from cuBLAS by > 2 bf16 ulp (m={m}, var={var})"
            assert e_y.mean() <= e_r.mean() * 1.05 + 1e-9, f"{name}: mean |err| vs fp64 {e_y.mean():.3e} > cuBLAS {e_r.mean():.3e}"
            if var == 0 and m == 8:
                rn = r64.to(torch.bfloat16)
                print(f"{name} m={m}: max ulp vs cuBLAS {float(ulp.max()):.2f}; == bf16(fp64) {float((y == rn).float().mean()):.5f} "
                      f"(cuBLAS {float((ref == rn).float().mean()):.5f}); mean|err| {float(e_y.mean()):.3e} (cuBLAS {float(e_r.mean()):.3e})")


ROWS_VARIANTS = (0, 1, 2, 3, 4, 5)


def check_rows(w: torch.Tensor, name: str, seed: int = 2) -> None:
    """DS41H_VERIFY_HEAD=3 (vrh row-range GEMV on the plain bf16 weight): vs cuBLAS / fp64, gather == plain, bitwise
    repeatable (3 launches + a graph replay), row split edge cases (CTA counts that leave partial / empty tiles)."""
    ext = vhead.ext()
    g = torch.Generator(device="cuda").manual_seed(seed)
    hs = (torch.randn(40, 5120, generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    hs[3, 17] = 60.0
    n = w.shape[0]
    for m in (1, 2, 5, 7, 8):
        idx = torch.randperm(40, generator=g, device="cuda")[:m].to(torch.int64)
        x = hs[idx]
        ref = F.linear(x, w)
        r64 = x.double() @ w.double().t()
        for var in ROWS_VARIANTS:
            for ctas in (0, 7, 200):
                y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                ext.gemv_rows(hs, idx, w, y, False, var, ctas)
                y2 = torch.empty_like(y)
                ext.gemv_rows(x.contiguous(), None, w, y2, True, var, ctas)
                torch.cuda.synchronize()
                assert torch.equal(y, y2), f"{name}: rows gather vs plain differ (m={m}, var={var}, ctas={ctas})"
                if ctas == 0:
                    y0 = y
                else:
                    assert torch.equal(y, y0), f"{name}: rows result depends on the CTA count (m={m}, var={var}, ctas={ctas})"
            e_y = (y0.double() - r64).abs()
            e_r = (ref.double() - r64).abs()
            mag = torch.maximum(y0.float().abs(), ref.float().abs()).clamp(min=2.0**-20)
            bf_ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
            n_far = int(((y0.float() - ref.float()).abs() > 2 * bf_ulp + 1e-4).sum())
            # stock precision = fp32 accumulation of the exact bf16 products in SOME order + one bf16 rounding: every logit must
            # lie within the rigorous bound of that class around the fp64 value (any fp32 order: |err| <= (K - 1) u sum|w x|,
            # u = 2^-24, plus half a bf16 ulp of the result for the final rounding); near-cancelling logits can legitimately
            # differ from cuBLAS by several ulps (counted, informational)
            s64 = x.double().abs() @ w.double().abs().t()
            ulp64 = torch.exp2(torch.floor(torch.log2(r64.abs().clamp(min=2.0**-40))) - 7)
            bound = 0.5 * ulp64 + 5119 * 2.0**-24 * s64 + 1e-30
            n_out = int((e_y > bound).sum())
            assert n_out == 0, f"{name}: rows: {n_out} logits outside the fp32-accumulation bound (m={m}, var={var})"
            if n_far:
                print(f"  {name} rows m={m} var={var}: {n_far} logits differ from cuBLAS by > 2 bf16 ulp (inside the fp32 bound; "
                      f"max |rows - fp64| / bound {float((e_y / bound).max()):.3f}, cuBLAS {float((e_r / bound).max()):.3f})")
            assert e_y.mean() <= e_r.mean() * 1.05 + 1e-9, f"{name}: rows mean |err| vs fp64 {e_y.mean():.3e} > cuBLAS {e_r.mean():.3e}"
            if var == 0 and m == 8:
                ulp = (y0.float() - ref.float()).abs() / bf_ulp
                rn = r64.to(torch.bfloat16)
                print(f"{name} rows m={m}: max ulp vs cuBLAS {float(ulp.max()):.2f}; == bf16(fp64) {float((y0 == rn).float().mean()):.5f} "
                      f"(cuBLAS {float((ref == rn).float().mean()):.5f}); mean|err| {float(e_y.mean()):.3e} (cuBLAS {float(e_r.mean()):.3e})")
    # bitwise repeatability: back-to-back launches and CUDA-graph replays (the exactness gate's determinism requirement)
    m = 8
    idx = torch.arange(m, device="cuda", dtype=torch.int64) * 4
    outs = []
    for _ in range(3):
        y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        ext.gemv_rows(hs, idx, w, y, True, 0, 0)
        outs.append(y)
    yg = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        ext.gemv_rows(hs, idx, w, yg, True, 0, 0)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ext.gemv_rows(hs, idx, w, yg, True, 0, 0)
    for _ in range(5):
        yg.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, outs[0]), f"{name}: rows graph replay differs"
    assert all(torch.equal(o, outs[0]) for o in outs[1:]), f"{name}: rows launches not bitwise repeatable"
    print(f"{name}: rows kernel PASS (m 1/2/5/7/8 x variants {ROWS_VARIANTS} x CTA counts 132/7/200, bitwise repeatable, graph replay)")


def bench(p: "vhead.PackedHead", w: torch.Tensor, hot: bool) -> None:
    from ds41h import drafter

    hs = (torch.randn(64, 5120, device="cuda") * 1.5).to(torch.bfloat16)
    idx = torch.arange(8, device="cuda", dtype=torch.int64) * 3
    heat = torch.empty(3 << 30, dtype=torch.uint8, device="cuda") if hot else None
    heat2 = torch.empty_like(heat) if hot else None
    y = torch.empty(8, w.shape[0], dtype=torch.bfloat16, device="cuda")
    fns = {
        "cuBLAS F.linear(hs[idx])": lambda: F.linear(hs[idx], w),
        "dvh head_bf16": lambda: drafter.ext().head_bf16(hs, idx, w, y, True),
    }
    for v in (0, 1, 2, 3):
        fns[f"vhead packed v{v}"] = (lambda vv: (lambda: p(hs, idx, 8, True, vv)))(v)
    for v in ROWS_VARIANTS:
        fns[f"vrh rows v{v}"] = (lambda vv: (lambda: vhead.ext().gemv_rows(hs, idx, w, y, True, vv, 0)))(v)
    res = {}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            if hot:
                heat2.copy_(heat)  # HBM heater: the in-situ power state right before the head
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e3)
        times.sort()
        t = times[len(times) // 2]
        nbytes = p.bytes if "packed" in name else w.numel() * 2
        res[name] = t
        print(f"bench{' hot' if hot else ''}: {name:28s} {t:7.1f} us  ({nbytes / t / 1e6:.2f} TB/s of its own bytes, "
              f"{w.numel() * 2 / t / 1e6:.2f} TB/s bf16-equivalent)")


def main() -> int:
    torch.manual_seed(0)
    w = synthetic()
    check_rows(w, "synthetic")
    check_rows(synthetic(1000, seed=3), "synthetic n=1000")
    if "--rows" in sys.argv:  # rows kernel only (no packing)
        wr = load_head_shard(int(os.environ.get("VHEAD_SHARDS", "0").split(",")[0]))
        if wr is not None:
            check_rows(wr, "head shard")
            if "--nobench" not in sys.argv:
                p = vhead.PackedHead(wr)
                for hot in (False, True):
                    bench(p, wr, hot=hot)
        print("vhead rows: ALL PASS")
        return 0
    p = check_unpack(w, "synthetic")
    check_gemv(p, w, "synthetic")
    shards = [int(r) for r in os.environ.get("VHEAD_SHARDS", "0,3").split(",") if r != ""]
    last = None
    for r in shards:
        t = time.time()
        wr = load_head_shard(r)
        if wr is None:
            print(f"SKIP real shard {r}: checkpoint unavailable")
            continue
        pr = check_unpack(wr, f"head shard {r} (load {time.time() - t:.0f} s)")
        check_gemv(pr, wr, f"head shard {r}")
        check_rows(wr, f"head shard {r}")
        last = (pr, wr)
    print("vhead: ALL PASS")
    if "--nobench" not in sys.argv and last is not None:
        for hot in (False, True):
            bench(*last, hot=hot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
