"""Tests + micro-benchmark for ds41h.moe_dec (decode-shaped small-T W4A16 routed MoE).

  GPU tests:  DS41H_MODEL_DIR=<checkpoint> python -m pytest tests/test_moe_dec.py -x -q
  CPU-only subset:                     CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_moe_dec.py -q -k cpu
  Benchmark:                           python tests/test_moe_dec.py --bench [--layers 10,20] [--cfgs 0x00,0x22,...]

Correctness contract: moe_dec must reproduce ds41h.moe.moe_forward_cuda(..., 0x500) (small16,
the certified small-T path) BIT FOR BIT for every T, routing, variant and token-chunking, be bit-repeatable under a
concurrent side-stream load and in CUDA-graph replays, and agree with the exact torch reference within the fp32-order bound.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import test_moe as tm  # noqa: E402  (raw_layer, stock_route, constants, pytest shim)
from ds41h import moe, moe_dec  # noqa: E402

pytest = tm.pytest
gpu = tm.gpu
H, I_R, E, TOPK, LIMIT = tm.H, tm.I_R, tm.E, tm.TOPK, tm.LIMIT
VARIANTS = list(range(10))  # moe_dec.cu launch_dec_v


# ===================================================================================== CPU: the expansion identity
def _small16_pair(w: int, p: int) -> int:  # moe_prefill.cu fp4_pair_bf16_raw<p>
    ms, ss = 6 - 4 * p, 12 - 4 * p
    mag = ((w << ms) if ms >= 0 else (w >> -ms)) & 0x01C001C0
    sgn = (w << ss) & 0x80008000
    return (mag | sgn) & 0xFFFFFFFF


def _dec_pair(w: int, p: int) -> int:  # moe_dec.cu: pairs 0/1 of w, 2/3 = pairs 0/1 of w >> 8 (IMAD multipliers)
    x = w if p < 2 else (w >> 8)
    if p % 2 == 0:
        return (((x & 0x000F000F) * 0x1040) & 0xFFFFFFFF) & 0x81C081C0
    return (((x & 0x00F000F0) * 0x104) & 0xFFFFFFFF) & 0x81C081C0


def test_cpu_imad_expansion_matches_small16():
    """Every pair of nibbles (P, P+4) of a word, every code combination, with random other nibbles: identical raw bits."""
    import random

    rnd = random.Random(0)
    for p in range(4):
        for lo in range(16):
            for hi in range(16):
                for _ in range(8):
                    w = rnd.getrandbits(32)
                    w &= ~((0xF << (4 * p)) | (0xF << (4 * p + 16)))
                    w |= (lo << (4 * p)) | (hi << (4 * p + 16))
                    assert _dec_pair(w, p) == _small16_pair(w, p), (p, lo, hi, hex(w))


def test_cpu_row_permutations_conflict_free():
    """The lane-row -> stage-row maps give bank-conflict-free 8-byte weight loads (2 wavefronts per LDS.64) for every
    K block of a stage, and cover each stage row exactly once."""
    for kps, perm in ((1, list(range(8))), (2, [0, 1, 4, 5, 2, 3, 6, 7]), (4, [0, 2, 4, 6, 1, 3, 5, 7])):
        rb = 32 * kps

        def woff(r, j, t):
            c = 2 * j + (t >> 1)
            pc = c if kps == 1 else (c ^ ((r >> 1) & 3)) if kps == 2 else (c ^ (r & 7))
            return r * rb + 16 * pc + 8 * (t & 1)

        for base in (0, 8, 16, 40):
            rows = [base + perm[g] for g in range(8)]
            assert sorted(r - base for r in rows) == list(range(8))
            for j in range(kps):
                for half in (0, 1):  # half warp = lanes 16 half .. 16 half + 15 = g in [4 half, 4 half + 4)
                    slots = set()
                    for g in range(4 * half, 4 * half + 4):
                        for t in range(4):
                            a = woff(rows[g], j, t)
                            slots.add((a % 128) // 8)
                    assert len(slots) == 16, (kps, base, j, half, slots)


# ===================================================================================== GPU helpers
def _inputs(T: int, seed: int, mode: str = "random"):
    """x [T, H] bf16 and stock routing (real gate of the test layer). mode: random | streams (8 consecutive positions per
    stream share a base vector: correlated routing like k=7 decode) | hot (most tokens share one direction: hot experts)."""
    w13, s13, w2, s2, gate, bias = tm.raw_layer()
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "random":
        x = torch.randn(T, H, device="cuda", generator=g) * 0.5
    elif mode == "streams":
        ns = max(1, (T + 7) // 8)
        base = torch.randn(ns, H, device="cuda", generator=g).repeat_interleave(8, 0)[:T]
        x = (base + 0.35 * torch.randn(T, H, device="cuda", generator=g)) * 0.5
    else:
        common = torch.randn(1, H, device="cuda", generator=g)
        x = (common + 0.3 * torch.randn(T, H, device="cuda", generator=g)) * 0.5
    x = x.to(torch.bfloat16)
    _, tw, ids = tm.stock_route(x, gate, bias)
    return x, tw.float().contiguous(), ids.int().contiguous()


def _ref16(x, tw, ids):
    w13, s13, w2, s2, *_ = tm.raw_layer()
    return moe.ext().moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, 0x500)


def _dec(x, tw, ids, cfg, addend=None):
    w13, s13, w2, s2, *_ = tm.raw_layer()
    return moe_dec.ext().moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, cfg, addend)


def _ndiff(a, b) -> int:
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


# ===================================================================================== GPU: bit identity with small16 0x500
@gpu
@pytest.mark.parametrize("T", [1, 3, 8, 16, 38, 65, 128, 300, 512, 777])
def test_forward_bit_identical_to_small16(T):
    for mode in ("random", "streams", "hot"):
        x, tw, ids = _inputs(T, 100 + T, mode)
        ref = _ref16(x, tw, ids)
        bad = {}
        for v in VARIANTS:
            cfg = v | (v << 4)
            n = _ndiff(_dec(x, tw, ids, cfg), ref)
            if n:
                bad[hex(cfg)] = n
        torch.cuda.synchronize()
        assert not bad, (T, mode, bad)


@gpu
@pytest.mark.parametrize("T", [16, 128, 512])
def test_mixed_variants_and_gemm_entry(T):
    """GEMM1 / GEMM2 variants combined freely; the per-GEMM entry equals the forward's intermediate (h) of small16."""
    x, tw, ids = _inputs(T, 7 + T, "streams")
    ref = _ref16(x, tw, ids)
    for cfg in (0x20, 0x02, 0x51, 0x37, 0x94, 0x68):
        assert _ndiff(_dec(x, tw, ids, cfg), ref) == 0, hex(cfg)
    # GEMM-level: h of GEMM1 vs small16's GEMM1 (variant 17) on the same permutation
    w13, s13, w2, s2, *_ = tm.raw_layer()
    e16, ed = moe.ext(), moe_dec.ext()
    R = T * TOPK
    meta = torch.empty(e16.meta_size(E, R), dtype=torch.int32, device="cuda")
    pw = torch.empty(R, dtype=torch.float32, device="cuda")
    e16.moe_permute(ids, tw, E, meta, pw)
    t0 = 2 * (E + 1) + 8
    xp = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
    e16.act_perm16(x, meta[t0 + 2 * R : t0 + 3 * R], TOPK, xp)
    h16 = torch.full((R, I_R), float("nan"), dtype=torch.bfloat16, device="cuda")
    e16.moe_gemm(1, xp, pw, w13, s13, meta, pw, R, 17, LIMIT, h16)
    for v in VARIANTS:
        hd = torch.full((R, I_R), float("nan"), dtype=torch.bfloat16, device="cuda")
        assert ed.moe_gemm(1, xp, w13, s13, meta, R, v, LIMIT, hd)
        assert _ndiff(hd, h16) == 0, ("gemm1", v)
        s16 = torch.full((R, H), float("nan"), dtype=torch.bfloat16, device="cuda")
        sd = torch.full((R, H), float("nan"), dtype=torch.bfloat16, device="cuda")
        e16.moe_gemm(2, h16, pw, w2, s2, meta, pw, R, 17, LIMIT, s16)
        assert ed.moe_gemm(2, h16, w2, s2, meta, R, v, LIMIT, sd)
        assert _ndiff(sd, s16) == 0, ("gemm2", v)


@gpu
def test_hot_expert_chunks():
    """Experts with many rows (chunks of NT = 16 / 32 tokens, all NG dispatch paths 1..NGMAX, partial last groups)."""
    w13, s13, w2, s2, gate, bias = tm.raw_layer()
    for T, nexp in ((40, 6), (100, 8), (257, 12), (512, 24)):
        g = torch.Generator(device="cuda").manual_seed(T)
        x = (torch.randn(T, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        hot = torch.randperm(E, generator=torch.Generator().manual_seed(T))[:nexp].cuda()
        ids = torch.stack([hot[torch.randperm(nexp, generator=torch.Generator().manual_seed(t))[:TOPK].cuda()] for t in range(T)]).int()
        tw = torch.rand(T, TOPK, device="cuda", generator=g).float() * 0.3
        ref = _ref16(x, tw, ids)
        for v in VARIANTS:
            assert _ndiff(_dec(x, tw, ids, v | (v << 4)), ref) == 0, (T, nexp, v)


@gpu
def test_dropped_ids_addend_tiny():
    w13, s13, w2, s2, *_ = tm.raw_layer()
    for T in (1, 2, 5, 33):
        x, tw, ids = _inputs(T, 55 + T, "random")
        ids = ids.clone()
        ids[0, 1] = -1
        if T > 1:
            ids[T - 1, 0] = E + 7
        ad = (torch.randn(T, H, device="cuda") * 0.1).to(torch.bfloat16)
        ref = moe.ext().moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, 0x500, ad)
        for v in (0, 2, 5):
            assert _ndiff(_dec(x, tw, ids, v | (v << 4), ad), ref) == 0, (T, v)


@gpu
@pytest.mark.parametrize("T", [24, 128, 512])
def test_vs_exact_reference(T):
    """Against the fp32 torch reference (exact dequantized weights): within the bound of the certified W4A16 path
    (bf16 rounding of h and y, fp32 accumulation order) -- the same tolerance test_moe uses for the W4A16 kernels."""
    w13, s13, w2, s2, *_ = tm.raw_layer()
    x, tw, ids = _inputs(T, 300 + T, "streams")
    ref = moe.moe_forward_ref(x.float(), tw, ids, w13, s13, w2, s2, LIMIT)  # float32 compute
    y = _dec(x, tw, ids, moe_dec.CFG)
    d = (y.float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    rel = (d.norm() / ref.float().norm()).item()
    assert rel < 4e-3 and d.max().item() <= 0.02 * scale + 1e-3, (T, rel, d.max().item(), scale)


def _side_load(n: int = 3):
    return tm._side_load(n)


@gpu
@pytest.mark.parametrize("T", [16, 128, 512, 1000])
def test_bit_repeatable_stress(T):
    """Identical inputs N times -> bit-identical outputs: back to back, under a concurrent side-stream GEMM / copy load
    (the cross-proxy race signature), interleaved with other sizes, and CUDA-graph replays interleaved with eager calls."""
    N = int(os.environ.get("DS41H_STRESS_N", "20"))
    cfgs = [int(c, 0) for c in os.environ.get("DS41H_DEC_STRESS_CFGS", "0x22,0x00,0x55").split(",")]
    x, tw, ids = _inputs(T, 900 + T, "streams")
    x2, tw2, ids2 = _inputs(3 * T + 17, 901 + T, "random")
    ref = _ref16(x, tw, ids)
    for cfg in cfgs:
        fwd = lambda a, b, c: _dec(a, b, c, cfg)
        bad = {}
        y0 = fwd(x, tw, ids).clone()
        assert _ndiff(y0, ref) == 0, (T, hex(cfg))
        for mode in ("plain", "side", "interleaved"):
            n = 0
            for _ in range(N):
                if mode == "side":
                    s = _side_load()
                if mode == "interleaved":
                    fwd(x2, tw2, ids2)
                    fwd(x[: max(1, T // 3)], tw[: max(1, T // 3)], ids[: max(1, T // 3)])
                y = fwd(x, tw, ids)
                if mode == "side":
                    torch.cuda.current_stream().wait_stream(s)
                n += _ndiff(y, y0) != 0
            bad[mode] = n
        sx, sw, si = x.clone(), tw.clone(), ids.clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fwd(sx, sw, si)
        torch.cuda.current_stream().wait_stream(st)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sy = fwd(sx, sw, si)
        n = 0
        for _ in range(N):
            fwd(x2, tw2, ids2)
            graph.replay()
            n += _ndiff(sy, y0) != 0
        bad["graph"] = n
        torch.cuda.synchronize()
        assert all(v == 0 for v in bad.values()), (T, hex(cfg), bad)


# ===================================================================================== benchmark
def _route_stats(ids: torch.Tensor) -> dict:
    cnt = torch.bincount(ids.reshape(-1).long().clamp(0, E - 1), minlength=E)
    nz = cnt[cnt > 0]
    return {"touched": int((cnt > 0).sum()), "max_rows": int(cnt.max()), "mean_rows": float(nz.float().mean()),
            "chunks16": int(((cnt + 15) // 16).sum()), "chunks32": int(((cnt + 31) // 32).sum())}


def bench(layers=(10, 20), cfgs=(0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99), sizes=(16, 32, 64, 128, 256, 384, 512),
          modes=("streams", "random"), iters=20, s16k=()):
    """Time the full small-T forward (permute + act_perm16 + GEMM1 + GEMM2 + combine) per call, alternating between the
    weights of several layers (no L2 reuse between calls), for small16 0x500, optional small16k variants and moe_dec cfgs.
    Reports us per call and the effective weight stream (touched experts' w13 + w2 + scales / time)."""
    per_e = (2 * I_R * H // 2 + 2 * I_R * H // 32) + (H * I_R // 2 + H * I_R // 32)
    lays = []
    for L in layers:
        w13, s13, w2, s2, gate, bias = tm.raw_layer(L)
        lays.append((w13, s13, w2, s2, gate, bias))
    e16, ed = moe.ext(), moe_dec.ext()
    rows = []
    for mode in modes:
        for T in sizes:
            inp = []
            for li, (w13, s13, w2, s2, gate, bias) in enumerate(lays):
                g = torch.Generator(device="cuda").manual_seed(1000 * li + T)
                if mode == "random":
                    x = torch.randn(T, H, device="cuda", generator=g) * 0.5
                else:
                    ns = max(1, (T + 7) // 8)
                    base = torch.randn(ns, H, device="cuda", generator=g).repeat_interleave(8, 0)[:T]
                    x = (base + 0.35 * torch.randn(T, H, device="cuda", generator=g)) * 0.5
                x = x.to(torch.bfloat16)
                _, tw, ids = tm.stock_route(x, gate, bias)
                inp.append((x, tw.float().contiguous(), ids.int().contiguous()))
            st = _route_stats(inp[0][2])
            nbytes = statistics.mean(_route_stats(i[2])["touched"] for i in inp) * per_e

            def timeit(fn):
                for _ in range(3):
                    for li in range(len(lays)):
                        fn(li)
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(iters):
                    for li in range(len(lays)):
                        fn(li)
                b.record()
                torch.cuda.synchronize()
                return a.elapsed_time(b) * 1e3 / (iters * len(lays))

            res = {}
            res["s16"] = timeit(lambda li: e16.moe_forward(*inp[li], *lays[li][:4], LIMIT, 0x500))
            for sv in s16k:
                res[f"k{sv:#x}"] = timeit(lambda li, sv=sv: e16.moe_forward(*inp[li], *lays[li][:4], LIMIT, sv))
            for c in cfgs:
                res[f"d{c:#04x}"] = timeit(lambda li, c=c: ed.moe_forward(*inp[li], *lays[li][:4], LIMIT, c, None))
            best = min((v, k) for k, v in res.items() if k.startswith("d"))
            line = (f"{mode:7s} T={T:4d} touched {st['touched']:3d} max {st['max_rows']:3d} mean {st['mean_rows']:5.1f} "
                    f"ch16 {st['chunks16']:3d} ch32 {st['chunks32']:3d} | " + " ".join(f"{k} {v:6.0f}" for k, v in res.items())
                    + f" | best {best[1]} {best[0]:.0f} us = {nbytes / best[0] / 1e6:.2f} TB/s (s16 {nbytes / res['s16'] / 1e6:.2f})")
            print(line, flush=True)
            rows.append((mode, T, st, res))
    return rows


def bench_gemm(layers=(10, 20), variants=tuple(range(10)), sizes=(16, 64, 128, 256, 512), mode="streams", iters=20, hint=(0,)):
    """Per-GEMM timing (GEMM1 = w13 + SwiGLU, GEMM2 = w2) of moe_dec variants vs small16 (GEMM variant 17) on the same
    permutation, alternating between the layers' weights. Prints us per call and the effective weight stream per GEMM."""
    lays = [tm.raw_layer(L) for L in layers]
    e16, ed = moe.ext(), moe_dec.ext()
    b1 = (2 * I_R * H // 2 + 2 * I_R * H // 32)  # w13 + s13 bytes per expert
    b2 = (H * I_R // 2 + H * I_R // 32)
    for T in sizes:
        st = []
        for li, (w13, s13, w2, s2, gate, bias) in enumerate(lays):
            g = torch.Generator(device="cuda").manual_seed(2000 * li + T)
            if mode == "random":
                x = torch.randn(T, H, device="cuda", generator=g) * 0.5
            else:
                ns = max(1, (T + 7) // 8)
                base = torch.randn(ns, H, device="cuda", generator=g).repeat_interleave(8, 0)[:T]
                x = (base + 0.35 * torch.randn(T, H, device="cuda", generator=g)) * 0.5
            x = x.to(torch.bfloat16)
            _, tw, ids = tm.stock_route(x, gate, bias)
            tw, ids = tw.float().contiguous(), ids.int().contiguous()
            R = T * TOPK
            meta = torch.empty(e16.meta_size(E, R), dtype=torch.int32, device="cuda")
            pw = torch.empty(R, dtype=torch.float32, device="cuda")
            e16.moe_permute(ids, tw, E, meta, pw)
            t0 = 2 * (E + 1) + 8
            xp = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
            e16.act_perm16(x, meta[t0 + 2 * R : t0 + 3 * R], TOPK, xp)
            h = torch.empty(R, I_R, dtype=torch.bfloat16, device="cuda")
            sl = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
            e16.moe_gemm(1, xp, pw, w13, s13, meta, pw, R, 17, LIMIT, h)
            touched = int((torch.bincount(ids.reshape(-1).long(), minlength=E) > 0).sum())
            st.append(dict(meta=meta, pw=pw, xp=xp, h=h, sl=sl, R=R, touched=touched))

        def timeit(fn):
            for _ in range(3):
                for li in range(len(lays)):
                    fn(li)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(iters):
                for li in range(len(lays)):
                    fn(li)
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) * 1e3 / (iters * len(lays))

        touched = statistics.mean(d["touched"] for d in st)
        for mode_g, (wi, si, bpe) in ((1, (0, 1, b1)), (2, (2, 3, b2))):
            act = (lambda li: st[li]["xp"]) if mode_g == 1 else (lambda li: st[li]["h"])
            out = (lambda li: st[li]["h"]) if mode_g == 1 else (lambda li: st[li]["sl"])
            res = {"s16": timeit(lambda li: e16.moe_gemm(mode_g, act(li), st[li]["pw"], lays[li][wi], lays[li][si], st[li]["meta"],
                                                         st[li]["pw"], st[li]["R"], 17, LIMIT, out(li)))}
            for v in variants:
                for hb in hint:
                    res[f"v{v}{'h' if hb else ''}"] = timeit(lambda li, v=v, hb=hb: ed.moe_gemm(mode_g, act(li), lays[li][wi], lays[li][si], st[li]["meta"],
                                                                                         st[li]["R"], v | (16 if hb else 0), LIMIT, out(li)))
            best = min((t, k) for k, t in res.items() if k != "s16")
            print(f"GEMM{mode_g} {mode} T={T:4d} touched {touched:5.1f} | " + " ".join(f"{k} {t:5.0f}" for k, t in res.items())
                  + f" | best {best[1]} {best[0]:.0f} us = {touched * bpe / best[0] / 1e6:.2f} TB/s (s16 {touched * bpe / res['s16'] / 1e6:.2f})",
                  flush=True)


def bench_prof(layers=(10, 20), variants=(2, 5, 8), sizes=(128, 512), mode="streams"):
    """Where the time goes (diagnostic build ds41h_moe_dec_prof, clock64 per warp): consumer FULL / SFULL waits, stage tail
    (fence + arrive), compute (rest of the K loops), epilogue, outside units; producer EMPTY / SEMPTY waits. Fractions of the
    per-warp kernel time, summed over all warps."""
    lays = [tm.raw_layer(L) for L in layers]
    e16, ep = moe.ext(), moe_dec.ext_prof()
    for T in sizes:
        for li, (w13, s13, w2, s2, gate, bias) in enumerate(lays[:1]):
            g = torch.Generator(device="cuda").manual_seed(3000 + T)
            if mode == "random":
                x = torch.randn(T, H, device="cuda", generator=g) * 0.5
            else:
                ns = max(1, (T + 7) // 8)
                base = torch.randn(ns, H, device="cuda", generator=g).repeat_interleave(8, 0)[:T]
                x = (base + 0.35 * torch.randn(T, H, device="cuda", generator=g)) * 0.5
            x = x.to(torch.bfloat16)
            _, tw, ids = tm.stock_route(x, gate, bias)
            tw, ids = tw.float().contiguous(), ids.int().contiguous()
            R = T * TOPK
            meta = torch.empty(e16.meta_size(E, R), dtype=torch.int32, device="cuda")
            pw = torch.empty(R, dtype=torch.float32, device="cuda")
            e16.moe_permute(ids, tw, E, meta, pw)
            t0 = 2 * (E + 1) + 8
            xp = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
            e16.act_perm16(x, meta[t0 + 2 * R : t0 + 3 * R], TOPK, xp)
            h = torch.empty(R, I_R, dtype=torch.bfloat16, device="cuda")
            sl = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
            for mode_g in (1, 2):
                act, out = (xp, h) if mode_g == 1 else (h, sl)
                wi, si = (0, 1) if mode_g == 1 else (2, 3)
                for v in variants:
                    for _ in range(3):  # warm (clocks, L2 state like the timed runs)
                        ep.moe_gemm(mode_g, act, w13 if wi == 0 else w2, s13 if si == 1 else s2, meta, R, v, LIMIT, out)
                    torch.cuda.synchronize()
                    ep.prof_reset()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    n = 10
                    for _ in range(n):
                        ep.moe_gemm(mode_g, act, w13 if wi == 0 else w2, s13 if si == 1 else s2, meta, R, v, LIMIT, out)
                    b.record()
                    torch.cuda.synchronize()
                    us = a.elapsed_time(b) * 1e3 / n
                    c = ep.prof_read()
                    tot = max(1, c[5])
                    comp = c[3] - c[0] - c[1] - c[2]
                    ptot = max(1, c[10])
                    print(f"GEMM{mode_g} {mode} T={T:4d} v{v}: {us:6.0f} us/call (prof build) | consumer warps {c[6] // n:4d} units "
                          f"{c[7] // n:5d}: FULL wait {100 * c[0] / tot:4.1f}% SFULL {100 * c[1] / tot:4.1f}% tail {100 * c[2] / tot:4.1f}% "
                          f"compute {100 * comp / tot:4.1f}% epilogue {100 * c[4] / tot:4.1f}% outside {100 * (c[5] - c[3] - c[4]) / tot:4.1f}% "
                          f"| producer EMPTY wait {100 * c[8] / ptot:4.1f}% SEMPTY {100 * c[9] / ptot:4.1f}% "
                          f"| avg consumer warp {c[5] / max(1, c[6]) / 1e3:.0f} kclk", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--layers", default="10,20")
    ap.add_argument("--cfgs", default="0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99")
    ap.add_argument("--sizes", default="16,32,64,128,256,384,512")
    ap.add_argument("--modes", default="streams,random")
    ap.add_argument("--s16k", default="")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--gemm", action="store_true", help="per-GEMM timing (bench_gemm)")
    ap.add_argument("--variants", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--hint", default="0")
    ap.add_argument("--prof", action="store_true", help="clock64 breakdown (diagnostic build)")
    a = ap.parse_args()
    if a.prof:
        bench_prof(tuple(int(v) for v in a.layers.split(",")), tuple(int(v) for v in a.variants.split(",")),
                   tuple(int(v) for v in a.sizes.split(",")), a.modes.split(",")[0])
    elif a.gemm:
        bench_gemm(tuple(int(v) for v in a.layers.split(",")), tuple(int(v) for v in a.variants.split(",")),
                   tuple(int(v) for v in a.sizes.split(",")), a.modes.split(",")[0], a.iters, tuple(int(v) for v in a.hint.split(",")))
    elif a.bench:
        bench(tuple(int(v) for v in a.layers.split(",")), tuple(int(v, 0) for v in a.cfgs.split(",")),
              tuple(int(v) for v in a.sizes.split(",")), tuple(a.modes.split(",")), a.iters,
              tuple(int(v, 0) for v in a.s16k.split(",") if v))
    else:
        sys.exit(pytest.main([__file__, "-x", "-q"]))
