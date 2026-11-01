"""Tests and micro-benchmark for ds41h 'prologue' (fused decode prologue + skinny FP8 GEMM, sm90a).

    python -m pytest tests/test_prologue.py -q      # all tests (GPU tests skip without a GPU)
    python tests/test_prologue.py                   # same via the built-in runner (no pytest needed)
    python tests/test_prologue.py --bench           # per-layer prologue time at M = 1, 6, 8 + skinny GEMM GB/s
                                                    # (hmma / wgmma x fast 0 / 1, both GEMM flavours)
    python tests/test_prologue.py cpu               # only tests whose name contains 'cpu'
    python tests/test_prologue.py wgmma fast        # only the RS-wgmma / fast-path tests (filters are OR'd)

Switch-dependent tests take the flavour explicitly (mma='hmma'|'wgmma', fast=0|3), so one run covers both
paths regardless of DS41H_PROLOGUE_MMA / DS41H_PROLOGUE_FAST; the plain tests follow the environment.

CPU tests (run anywhere, CUDA_VISIBLE_DEVICES=""): skinny packing round trip, float64 emulation of the
kernel's activation staging / per-32 post-scale bookkeeping, and the equivalence of the KV-record
exponent formula with stock's ceilf(log2f(amax / 448)) over every bf16 amax; float64 emulation of the
RS-wgmma data path (packed lane words -> A fragments, stage_e4m3 -> B tile -> no-swizzle descriptor,
C fragments, promotion factors) against act_quant(x) @ W^T; the rcp Sinkhorn vs the IEEE one.

GPU tests: skinny GEMM vs a float64 reference on random and real checkpoint
weights (every decode GEMM shape); the full prologue vs a float32 torch reference built from ds41h.ref on
real layer weights (layers 3 and 20, M = 1..8), the SWA records byte-exact vs an exact torch replica and
vs stock vLLM's own CUDA writer (torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert), the
mHC outputs vs stock mhc_fused_post_pre_delayed_tilelang, slot -1 / short slot mappings, profile runs
without a cache, and CUDA-graph replay (counter reset across replays).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from common import (  # noqa: E402
    D,
    bench,
    compare,
    fmt_bw,
    fmt_metrics,
    has_cuda,
    random_pre_mix,
    reader,
    realistic_normed,
    realistic_stream,
    ref,
    require_cuda,
    run_module_tests,
    skip,
)

from ds41h import prologue, skinny  # noqa: E402

HC, QR, KVW, HEADS, HDIM, NOPE, ROPE = 4, 1280, 512, 16, 512, 448, 64
V4_DATA, V4_SCALES = 576, 8

# =========================================================================================== helpers


def rand_e4m3(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Random e4m3 bytes (all finite codes, incl. subnormals and both zeros) as uint8 [n, k]."""
    b = torch.randint(0, 256, (n, k), generator=g, dtype=torch.int32)
    b = torch.where((b & 0x7F) == 0x7F, b ^ 0x01, b)  # drop the NaN codes 0x7f / 0xff
    return b.to(torch.uint8)


def rand_scales(n: int, k: int, g: torch.Generator, lo: int = 112, hi: int = 124) -> torch.Tensor:
    """Checkpoint-form ue8m0 block scales [n/32, k/32] (uint8)."""
    return torch.randint(lo, hi + 1, (-(-n // 32), k // 32), generator=g, dtype=torch.int32).to(torch.uint8)


def rand_act(m: int, k: int, g: torch.Generator, kind: str = "normal") -> torch.Tensor:
    """bf16 activations; 'mixed' adds huge/tiny/zero blocks to stress the per-32 f16 staging."""
    x = torch.randn(m, k, generator=g)
    if kind == "mixed":
        blocks = x.view(m, k // 32, 32)
        sel = torch.rand(m, k // 32, generator=g)
        blocks[sel < 0.05] *= 3.0e4  # would overflow f16 without the staging scale
        blocks[(sel >= 0.05) & (sel < 0.10)] *= 1.0e-30  # far below the f16 range
        blocks[(sel >= 0.10) & (sel < 0.12)] = 0.0
        out = (torch.rand(m, k, generator=g) < 0.002).float() * 200.0  # sparse outliers inside blocks
        x = (blocks.view(m, k) + out * x.sign())
    return x.to(torch.bfloat16)


def ulp_mismatch(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(#elements differing, max |difference| in bf16 ulps) between two bf16 tensors."""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    # map sign-magnitude to a monotonic integer line
    ai = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    d = (ai - bi).abs()
    return int((d > 0).sum()), int(d.max()) if d.numel() else 0


def fma_rope_bf16(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Stock sm_90 RoPE on the last 64 dims, bit-exact emulation (float64 FMA, see prologue.cu):
    even' = fma(e, c, -fl(o*s)), odd' = fma(e, s, fl(o*c)), each rounded to fp32, then to bf16.
    x [..., 512] bf16-valued, cos/sin [rows, 32] broadcastable to x[..., :32]."""
    x = x.float().clone()
    r = x[..., NOPE:].unflatten(-1, (32, 2))
    e, o = r[..., 0].double(), r[..., 1].double()
    c, s = cos.double(), sin.double()
    os_ = (o * s).float().double()
    oc = (o * c).float().double()
    ne = (e * c - os_).float()
    no = (e * s + oc).float()
    x[..., NOPE:] = torch.stack([ne, no], dim=-1).flatten(-2)
    return x.to(torch.bfloat16)


def cos_sin_table(ratio: int, max_pos: int, device) -> torch.Tensor:
    """[max_pos, 64] fp32 cos | sin table for a layer's compress ratio (ds41h.ref rope frequencies)."""
    c, s = ref.rope_cos_sin(torch.arange(max_pos), ratio)
    return torch.cat([c, s], dim=-1).float().contiguous().to(device)


# =========================================================================================== CPU tests


def test_cpu_pack_roundtrip():
    g = torch.Generator().manual_seed(1)
    for n, k in [(32, 64), (48, 96), (1792, 5120), (5120, 576)]:
        w = rand_e4m3(n, k, g)
        s = rand_scales(n, k, g)
        sw = skinny.pack(w, s)
        assert sw.w.numel() == n * k and sw.s.shape == (n // 16, -(-(k // 32) // 4) * 4)
        w2, st = skinny.unpack(sw)
        assert torch.equal(w2, w), f"pack/unpack mismatch {n}x{k}"
        assert torch.equal(st, s.repeat_interleave(2, 0)[: n // 16])
        # (row-expanded scale form: ref.dequant_fp8 infers the block size from the scale shape)
        assert torch.equal(skinny.dequant(sw), ref.dequant_fp8(w.view(torch.float8_e4m3fn), ref.vllm_scale_layout(s)[:n]))
        # vLLM's row-expanded scale form gives the same packing
        sw2 = skinny.pack(w.view(torch.float8_e4m3fn), ref.vllm_scale_layout(s)[:n])
        assert torch.equal(sw2.w, sw.w) and torch.equal(sw2.s, sw.s)
        # fragment layout spot check (skinny.cu header): chunk (rt, kb), lane l = 4g + t
        kb_n = k // 32
        for rt, kb, gg, t in [(0, 0, 0, 0), (n // 16 - 1, kb_n - 1, 7, 3), (n // 32, kb_n // 2, 3, 1)]:
            base = ((rt * kb_n + kb) * 32 + gg * 4 + t) * 16
            assert torch.equal(sw.w[base : base + 8], w[16 * rt + gg, 32 * kb + 8 * t : 32 * kb + 8 * t + 8])
            assert torch.equal(sw.w[base + 8 : base + 16], w[16 * rt + gg + 8, 32 * kb + 8 * t : 32 * kb + 8 * t + 8])
    # a scale that varies inside a 16-row tile is rejected
    w = rand_e4m3(32, 64, g)
    s = torch.full((32, 2), 120, dtype=torch.uint8)
    s[3, 0] = 121
    try:
        skinny.pack(w, s)
    except ValueError:
        pass
    else:
        raise AssertionError("non-uniform tile scale accepted")


def emulate_skinny(x: torch.Tensor, sw) -> torch.Tensor:
    """float64 emulation of skinny.cu: convert_act staging (per (token, k32) power of two, f16 RN) and
    mma_frag's post-scale 2^(eb + sw - 254); products and sums exact in float64."""
    m, k = x.shape
    kb = k // 32
    w8, st = skinny.unpack(sw)
    wf = w8.view(torch.float8_e4m3fn).double()  # [N, K] exact e4m3 values (cvt e4m3 -> f16 is exact)
    xf = x.float()
    amax = xf.view(m, kb, 32).abs().amax(-1)
    ea = ((amax.view(torch.int32) >> 23) & 0xFF).clamp_min(14)
    mul = torch.ldexp(torch.ones_like(amax), (141 - ea))  # float32 2^(141 - ea)
    h = (xf.view(m, kb, 32) * mul.unsqueeze(-1)).half().double()  # f32 multiply, f16 RN
    eb = (ea - 14).double()  # [m, kb]
    d = torch.einsum("mbk,nbk->mnb", h, wf.view(-1, kb, 32))  # per-block HMMA sums
    s_rows = st.repeat_interleave(16, 0).double()  # [N, kb]
    e = (eb.unsqueeze(1) + s_rows.unsqueeze(0) - 127).clamp(0, 254)  # biased exponent of the factor
    factor = torch.where(e > 0, torch.ldexp(torch.ones_like(e), (e - 127).long()), torch.zeros_like(e))
    return (d * factor).sum(-1)


def test_cpu_skinny_numerics_emulation():
    """The f16 staging + per-32 post-scale bookkeeping is exact (vs float64 x @ W^T)."""
    g = torch.Generator().manual_seed(2)
    for n, k, kind in [(64, 5120, "normal"), (48, 1280, "mixed"), (32, 576, "mixed"), (16, 2048, "normal")]:
        w = rand_e4m3(n, k, g)
        s = rand_scales(n, k, g, 100, 130)
        sw = skinny.pack(w, s)
        for m in (1, 5, 8):
            x = rand_act(m, k, g, kind)
            y = emulate_skinny(x, sw)
            yr = x.double() @ ref.dequant_fp8(w.view(torch.float8_e4m3fn), ref.vllm_scale_layout(s)[:n]).double().t()
            err = (y - yr).abs().max() / yr.abs().max().clamp_min(1e-300)
            # f16 staging is exact unless a value sits > 2^31 below its block max (the 1e-30 blocks
            # next to 2e2 outliers can lose low bits): bound the relative error at f32-level.
            assert err < 1e-6, f"{n}x{k} m={m} {kind}: rel err {err:.3e}"


def test_cpu_kv_exponent_matches_stock_formula():
    """Every bf16 amax >= 1e-4: ceil(log2(fl32(amax / 448))) (stock processDeepseekV4Slot) equals the
    integer formula used by prologue.cu (common.cuh quant_exp: fl32(amax * fl32(1/448)), exponent + (mant != 0))."""
    allbits = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16)  # positive finite bf16
    a = allbits.view(torch.bfloat16).float()
    a = torch.maximum(a, torch.tensor(1e-4, dtype=torch.float32))  # stock: fmaxf(absmax, 1e-4f)
    q = (a / torch.tensor(448.0)).double()  # IEEE fp32 division
    stock = torch.ceil(torch.log2(q))
    prod = a * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = prod.view(torch.int32)
    ours = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    bad = (stock.long() != ours.long()).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} mismatches, first amax={a[bad[0]].item()!r}"


def test_cpu_v4_writer_replica_matches_ref():
    """The test's own record replica (fma_rope_bf16 + ref.v4_pack) is consistent with ref.v4_record_quant."""
    g = torch.Generator().manual_seed(3)
    kv = (torch.randn(6, 512, generator=g) * 3).to(torch.bfloat16)
    c, s = ref.rope_cos_sin(torch.tensor([0, 1, 5, 100, 4097, 65535]), 2)
    roped = fma_rope_bf16(kv, c, s)
    data, scales = ref.v4_pack(roped.float())
    assert data.shape == (6, V4_DATA) and scales.shape == (6, V4_SCALES)
    back = ref.v4_unpack(data, scales)
    assert torch.equal(back[:, NOPE:], roped[:, NOPE:].float())
    rel = (back - roped.float()).abs().max() / roped.float().abs().max()
    assert rel < 0.07  # e4m3 quantization


# ------------------------------------------------------------------------------- CPU: RS-wgmma data path

WG_KB_PITCH = 272  # kernels/skinny.cu


def _e4m3_values(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float8_e4m3fn).double()


def emulate_wgmma(sw, x: torch.Tensor, tiles: list[int]) -> torch.Tensor:
    """float64 emulation of the RS-wgmma path (kernels/skinny.cu) for one warpgroup = 4 tiles over all K:
    returns y [64, 8] (rows of `tiles`, tokens; tokens >= m must come out 0).

    A: lane l = 4g + t of warp w reads the packed 16-byte word {x, y, z, w} of (tile, kb) and feeds
       {a0, a1, a2, a3} = {x, z, y, w}; PTX/CuTe 8-bit A fragment: a0 = (g, 4t+b), a1 = (g+8, 4t+b),
       a2 = (g, 16+4t+b), a3 = (g+8, 16+4t+b) (byte b of the register).
    B: bytes exactly as stage_e4m3 writes them, read back through the no-swizzle K-major descriptor
       address(n, j) = kb * 272 + (j // 16) * LBO(128) + (n % 8) * 16 + (n // 8) * SBO + j % 16.
    D -> C fragment (d0..d3 = (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)) -> acc += d * fac."""
    n_all, k = sw.n, sw.k
    kbn, m = k // 32, x.shape[0]
    q, sx = ref.quant.fp8_quant(x.float(), 32)  # reference act_quant: e4m3 [m, k], scale [m, kb]
    qb = q.view(torch.uint8)
    bt = torch.zeros(kbn * WG_KB_PITCH, dtype=torch.uint8)
    for tok in range(m):
        for kb in range(kbn):
            for t in range(4):
                base = kb * WG_KB_PITCH + tok * 16 + 4 * t
                bt[base : base + 4] = qb[tok, 32 * kb + 8 * t : 32 * kb + 8 * t + 4]
                bt[base + 128 : base + 132] = qb[tok, 32 * kb + 8 * t + 4 : 32 * kb + 8 * t + 8]
    j = torch.arange(32).view(32, 1)
    nn = torch.arange(8).view(1, 8)
    addr = (j // 16) * 128 + (nn % 8) * 16 + (nn // 8) * 256 + j % 16  # [32, 8]
    B = torch.stack([_e4m3_values(bt[kb * WG_KB_PITCH + addr.flatten()]).view(32, 8) for kb in range(kbn)])
    # A fragment index maps (lane, reg, byte) -> (row, k) and source byte inside the lane word
    lanes = torch.arange(32).view(32, 1, 1)
    regs = torch.arange(4).view(1, 4, 1)
    byt = torch.arange(4).view(1, 1, 4)
    g, t = lanes // 4, lanes % 4
    row = g + 8 * (regs % 2)
    kk = 4 * t + byt + 16 * (regs // 2)
    word = torch.tensor([0, 2, 1, 3]).view(1, 4, 1)  # {a0, a1, a2, a3} = {x, z, y, w}
    src = lanes * 16 + word * 4 + byt
    row, kk, src = (v.expand(32, 4, 4).flatten() for v in (row, kk, src))
    w8, st = skinny.unpack(sw)
    sfac = torch.zeros(8, kbn, dtype=torch.float64)
    sfac[:m] = sx.double()
    y = torch.zeros(64, 8, dtype=torch.float64)
    for wi, tile in enumerate(tiles):
        if tile >= n_all // 16:
            continue
        for kb in range(kbn):
            chunk = sw.w[(tile * kbn + kb) * 512 : (tile * kbn + kb + 1) * 512]
            A = torch.zeros(16, 32, dtype=torch.float64)
            A[row, kk] = _e4m3_values(chunk[src])
            D = A @ B[kb]  # [16, 8]
            fac = torch.ldexp(torch.ones(()), (st[tile, kb].long() - 127)).double() * sfac[:, kb]  # [8]
            # C fragment round trip (lane (g, t) holds rows g, g+8 x tokens 2t, 2t+1)
            for lane in range(32):
                gg, tt = lane // 4, lane % 4
                for rr, cc in ((gg, 2 * tt), (gg, 2 * tt + 1), (gg + 8, 2 * tt), (gg + 8, 2 * tt + 1)):
                    y[16 * wi + rr, cc] += D[rr, cc] * fac[cc]
    return y


def test_cpu_wgmma_fragment_emulation():
    """The wgmma path's byte plumbing (no weight repack, logical-k permutation applied to the B tile,
    descriptor LBO/SBO, C fragment, promotion factors) computes act_quant(x) @ W^T exactly."""
    g = torch.Generator().manual_seed(7)
    for n, k, m, tiles in [(64, 64, 3, [0, 1, 2, 3]), (96, 128, 8, [5, 2, 0, 4]), (48, 96, 1, [1, 2, 0, 3])]:
        w = rand_e4m3(n, k, g)
        s = rand_scales(n, k, g, 100, 130)
        sw = skinny.pack(w, s)
        x = rand_act(m, k, g, "mixed")
        y = emulate_wgmma(sw, x, tiles)
        wd = skinny.dequant(sw).double()
        yr = ref.quant.act_quant_sim(x.float(), 32).double() @ wd.t()  # [m, n]
        for wi, tile in enumerate(tiles):
            if tile >= n // 16:
                assert y[16 * wi : 16 * wi + 16].abs().max() == 0
                continue
            got = y[16 * wi : 16 * wi + 16, :m].t()
            exp = yr[:, 16 * tile : 16 * tile + 16]
            err = (got - exp).abs().max() / exp.abs().max().clamp_min(1e-300)
            assert err < 1e-12, f"{n}x{k} m={m} tile {tile}: rel err {err:.3e}"
            if m < 8:
                assert y[16 * wi : 16 * wi + 16, m:].abs().max() == 0, "padding tokens must stay zero"


def _sinkhorn_emul(logits: torch.Tensor, eps: float, iters: int, fast: bool) -> torch.Tensor:
    """float32 emulation of prologue.cu sinkhorn_job (exact) / sinkhorn_fast (rcp; fp32 1/x stands in for
    rcp.approx, which is within 1 ulp of it). logits [T, 4, 4]."""
    f32 = torch.float32
    c = logits.to(f32)
    e = torch.exp(c - c.amax(-1, keepdim=True))
    rs = (e[..., 0] + e[..., 1]) + (e[..., 2] + e[..., 3])
    if fast:
        c = (e.double() * (1.0 / rs).to(f32).double()[..., None] + eps).to(f32)  # fmaf(e, rcp, eps)
    else:
        c = e / rs[..., None] + torch.tensor(eps, dtype=f32)

    def cols(c):
        cs = (c[:, 0, :] + c[:, 1, :]) + (c[:, 2, :] + c[:, 3, :]) + torch.tensor(eps, dtype=f32)
        return c * (1.0 / cs).to(f32)[:, None, :] if fast else c / cs[:, None, :]

    def rows(c):
        rs = (c[..., 0] + c[..., 1]) + (c[..., 2] + c[..., 3]) + torch.tensor(eps, dtype=f32)
        return c * (1.0 / rs).to(f32)[..., None] if fast else c / rs[..., None]

    c = cols(c)
    for _ in range(iters - 1):
        c = cols(rows(c))
    return c


def test_cpu_default_is_lossless():
    """The default prologue / skinny configuration is the lossless one (docs/EXACTNESS.md: target model at least as
    precise as stock): HMMA with 16-bit activations (no e4m3 activation quant), IEEE Sinkhorn (no rcp.approx), no
    deferred-norm e4m3 staging. The lossy fast paths (wgmma, FAST bit 2, 4096) stay opt-in."""
    import os

    if "DS41H_PROLOGUE_MMA" not in os.environ:
        assert prologue.MMA == "hmma" and skinny.MMA == "hmma"
    assert prologue.FAST_DEFAULT & (2 | 4096) == 0, f"lossy bits in FAST_DEFAULT {prologue.FAST_DEFAULT}"
    if "DS41H_PROLOGUE_FAST" not in os.environ:
        assert prologue.FAST == prologue.FAST_DEFAULT


def test_cpu_fast_sinkhorn_within_stock_tolerance():
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(4096, 4, 4, generator=g) * 3.0
    exact = _sinkhorn_emul(logits, 1e-6, 20, fast=False)
    fast = _sinkhorn_emul(logits, 1e-6, 20, fast=True)
    d = (exact - fast).abs().max().item()
    assert d < 1e-5, f"rcp Sinkhorn deviates by {d:.3e} (stock tolerance used by the GPU tests: 5e-5)"


def _lane_transposition_totals(vals: torch.Tensor) -> torch.Tensor:
    """float32 emulation of prologue.cu's P1b lane transposition (first step folded into the dot loop,
    then lane_sums_step<8..1>): vals [32 lanes, 32 entries] -> totals [32] (lane L holds entry L)."""
    lanes = torch.arange(32)
    hi16 = (lanes & 16) != 0
    lo, up = vals[:, :16], vals[:, 16:]
    keep = torch.where(hi16[:, None], up, lo)
    send = torch.where(hi16[:, None], lo, up)
    v = torch.zeros(32, 32, dtype=torch.float32)
    v[:, :16] = keep + send[lanes ^ 16]
    for w in (8, 4, 2, 1):
        hi = ((lanes & w) != 0)[:, None]
        keep = torch.where(hi, v[:, w : 2 * w], v[:, :w])
        send = torch.where(hi, v[:, :w], v[:, w : 2 * w])
        v[:, :w] = keep + send[lanes ^ w]
    return v[:, 0]


def _warp_sum_lanes(x: torch.Tensor) -> torch.Tensor:
    lanes = torch.arange(32)
    for o in (16, 8, 4, 2, 1):
        x = x + x[lanes ^ o]
    return x


def test_cpu_p1_split_xsq_bit_identical():
    """P1a's warp_sum (xor butterfly: every lane ends with the same bits) reproduces Phase 1's xsq (entry 25
    of the P1b transposition) bit for bit, so the default path keeps its numerics after the P1a / P1b split."""
    g = torch.Generator().manual_seed(9)
    for _ in range(200):
        xf = (torch.randn(32, generator=g) * torch.rand(32, generator=g) * 10).float()
        vals = torch.randn(32, 32, generator=g).float()
        vals[:, 25] = xf * xf
        old = _lane_transposition_totals(vals)[25].view(torch.int32)
        new = _warp_sum_lanes(xf * xf).view(torch.int32)
        assert bool((new == old).all()), "P1a collapse square sum differs from Phase 1"


def test_cpu_wgmma_split_configs():
    """Every decode GEMM shape gets a feasible wgmma decomposition (needs the JIT-built extension)."""
    try:
        e = skinny.ext()
    except Exception as ex:  # noqa: BLE001 - no toolchain here
        skip(f"skinny extension not buildable: {ex!r}")
    for name, (n, k) in SKINNY_SHAPES.items():
        kw, kp, s, ctas = (int(v) for v in e.choose_split_wg(n, k))
        kb = k // 32
        assert kw % 4 == 0 and 4 <= kw <= 20 and kp in (1, 2) and s in (1, 2, 4, 8) and (kp == 2 or s == 1), name
        assert kw * kp * s >= kb, f"{name}: parts do not cover K"
        print(f"  {name:<16s} {n}x{k}: kw={kw} kp={kp} cluster={s} ctas={ctas}")


# =========================================================================================== GPU: skinny


SKINNY_SHAPES = {
    "fused_wqa_wkv": (1792, 5120),
    "wq_b": (8192, 1280),
    "indexer.wq_b": (4096, 1280),
    "wo_b": (5120, 2048),
    "shared.gate_up": (1152, 5120),
    "shared.down": (5120, 576),
}


def _check_skinny(sw, x: torch.Tensor, name: str, mma: str | None = None) -> dict:
    """hmma: vs x @ W^T (W8A16, products exact). wgmma: vs act_quant(x, 32) @ W^T (the reference W8A8
    numerics; Hopper's FP8 tensor-core accumulator is narrower than fp32, so each k32 product sum gets
    ~2^-13 of its magnitude as slack), plus a loose closeness check to the bf16-activation product."""
    mma = (mma or skinny.MMA).lower()
    y = skinny.gemm(x, sw, mma=mma)
    torch.cuda.synchronize()
    wd = skinny.dequant(sw).double()
    xin = ref.quant.act_quant_sim(x.float(), 32).double() if mma == "wgmma" else x.double()
    yr = xin @ wd.t()
    mag = xin.abs() @ wd.abs().t()  # accumulation magnitude per output
    m = compare(y.float(), yr.float())
    # products exact, fp32 accumulation, bf16 output: within 1 bf16 ulp + fp32 accumulation error
    assert m["finite"] and m["rel_rmse"] < 3e-3 and m["cos"] > 0.99999, f"{name} [{mma}]: {fmt_metrics(m)}"
    tol = yr.abs() * 2.0**-7 + mag * (2.0**-12 if mma == "wgmma" else 2.0**-18)
    bad = int(((y.double() - yr).abs() > tol).sum())
    assert bad == 0, f"{name} [{mma}]: {bad} elements beyond 1 bf16 ulp; {fmt_metrics(m)}"
    if mma == "wgmma":  # e4m3 activations vs the exact bf16 product: quantization noise only
        mb = compare(y.float(), (x.double() @ wd.t()).float())
        assert mb["cos"] > 0.998, f"{name} [wgmma vs bf16 activations]: {fmt_metrics(mb)}"
    return m


def test_skinny_random_shapes():
    require_cuda()
    g = torch.Generator().manual_seed(4)
    for name, (n, k) in SKINNY_SHAPES.items():
        sw = skinny.pack(rand_e4m3(n, k, g).cuda(), rand_scales(n, k, g).cuda())
        for m in (1, 2, 6, 7, 8):
            for kind in ("normal", "mixed"):
                _check_skinny(sw, rand_act(m, k, g, kind).cuda(), f"{name} m={m} {kind}")
    # strided input rows (a column slice of a wider tensor, as qr out of qr_kv)
    sw = skinny.pack(rand_e4m3(8192, 1280, g).cuda(), rand_scales(8192, 1280, g).cuda())
    wide = rand_act(4, 1792, g).cuda()
    _check_skinny(sw, wide[:, :1280], "strided rows")


def test_skinny_checkpoint_weights():
    require_cuda()
    rd = reader()
    dev = "cuda"
    cases = [
        ("fused_wqa_wkv L3", ["layers.3.attn.wq_a.weight", "layers.3.attn.wkv.weight"], None),
        ("wq_b L3 r1", ["layers.3.attn.wq_b.weight"], 1),
        ("indexer.wq_b L2", ["layers.2.attn.indexer.wq_b.weight"], None),
        ("wo_b L3 r2", ["layers.3.attn.wo_b.weight"], 2),
        ("shared w1|w3 L3 r0", ["layers.3.ffn.shared_experts.w1.weight", "layers.3.ffn.shared_experts.w3.weight"], 0),
        ("shared w2 L3 r3", ["layers.3.ffn.shared_experts.w2.weight"], 3),
    ]
    norm = rd.get("layers.3.attn_norm.weight").float()
    for name, names, rank in cases:
        sw = skinny.pack_checkpoint(rd, names, rank, dev)
        for m in (1, 8):
            x = realistic_normed(m, torch.ones(sw.k) if sw.k != D else norm, seed=m, device=dev).to(torch.bfloat16)
            print(f"  {name:<22s} m={m}: {fmt_metrics(_check_skinny(sw, x, name))}")


def test_skinny_wgmma_random_shapes():
    """RS-wgmma skinny GEMM (every decode shape: single CTA, 2-warpgroup k split, cluster k split)."""
    require_cuda()
    g = torch.Generator().manual_seed(14)
    for name, (n, k) in SKINNY_SHAPES.items():
        sw = skinny.pack(rand_e4m3(n, k, g).cuda(), rand_scales(n, k, g).cuda())
        print(f"  {name}: wgmma split {skinny.split_config(n, k, 'wgmma')}")
        for m in (1, 2, 6, 7, 8):
            for kind in ("normal", "mixed"):
                _check_skinny(sw, rand_act(m, k, g, kind).cuda(), f"{name} m={m} {kind}", mma="wgmma")
    # partial last tile quad (N % 64 != 0) and a K needing zero-filled tail blocks
    for n, k in ((1808, 5120), (80, 576), (16, 1280)):
        sw = skinny.pack(rand_e4m3(n, k, g).cuda(), rand_scales(n, k, g).cuda())
        _check_skinny(sw, rand_act(5, k, g).cuda(), f"odd {n}x{k}", mma="wgmma")
    sw = skinny.pack(rand_e4m3(8192, 1280, g).cuda(), rand_scales(8192, 1280, g).cuda())
    wide = rand_act(4, 1792, g).cuda()
    _check_skinny(sw, wide[:, :1280], "strided rows", mma="wgmma")
    # deterministic (cluster reduction in rank order)
    x = rand_act(6, 5120, g).cuda()
    sw = skinny.pack(rand_e4m3(1152, 5120, g).cuda(), rand_scales(1152, 5120, g).cuda())
    y0 = skinny.gemm(x, sw, mma="wgmma")
    for _ in range(3):
        assert torch.equal(skinny.gemm(x, sw, mma="wgmma"), y0), "wgmma skinny GEMM is not deterministic"


def test_skinny_wgmma_checkpoint_weights():
    require_cuda()
    rd = reader()
    norm = rd.get("layers.3.attn_norm.weight").float()
    for name, names, rank in [
        ("fused_wqa_wkv L3", ["layers.3.attn.wq_a.weight", "layers.3.attn.wkv.weight"], None),
        ("wq_b L3 r1", ["layers.3.attn.wq_b.weight"], 1),
        ("indexer.wq_b L2", ["layers.2.attn.indexer.wq_b.weight"], None),
        ("shared w2 L3 r3", ["layers.3.ffn.shared_experts.w2.weight"], 3),
    ]:
        sw = skinny.pack_checkpoint(rd, names, rank, "cuda")
        for m in (1, 8):
            x = realistic_normed(m, torch.ones(sw.k) if sw.k != D else norm, seed=m, device="cuda").to(torch.bfloat16)
            mw = _check_skinny(sw, x, name, mma="wgmma")
            mh = compare(skinny.gemm(x, sw, mma="wgmma").float(), skinny.gemm(x, sw, mma="hmma").float())
            print(f"  {name:<22s} m={m}: vs W8A8 ref {fmt_metrics(mw)} | wgmma vs hmma {fmt_metrics(mh)}")


def test_skinny_eligibility_and_dropin():
    """ds41h.skinny.install wraps ModelOptLinearMethod: stand-in method/layer objects."""
    require_cuda()
    try:
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static  # noqa: F401
    except Exception as e:  # noqa: BLE001
        skip(f"vLLM not importable: {e!r}")
    g = torch.Generator().manual_seed(5)

    class FakeMethod:
        def __init__(self):
            from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static

            self.spec = type("S", (), {"weight": kMxfp8Static})()
            self.calls = 0

        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)  # "repack"

        def apply(self, layer, x, bias=None):
            self.calls += 1
            return torch.zeros(x.shape[0], 1792, dtype=x.dtype, device=x.device)

    # CAPTURE / APPLY follow DS41H=1 + DS41H_PROLOGUE / DS41H_SKINNY at import; the unit test runs without
    # the hook environment, so switch both on for the duration of the test (the wrappers read them per call).
    saved = (skinny.CAPTURE, skinny.APPLY)
    skinny.CAPTURE, skinny.APPLY = True, True
    try:
        _skinny_dropin_body(FakeMethod, g)
    finally:
        skinny.CAPTURE, skinny.APPLY = saved


def _skinny_dropin_body(FakeMethod, g):
    module = type("M", (), {})()
    module.ModelOptLinearMethod = FakeMethod
    skinny.install(module)
    method = module.ModelOptLinearMethod()
    layer = torch.nn.Module()
    layer.prefix = "model.layers.3.attn.fused_wqa_wkv"
    w = rand_e4m3(1792, 5120, g).cuda()
    s = rand_scales(1792, 5120, g).cuda()
    layer.weight = torch.nn.Parameter(w.view(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ref.vllm_scale_layout(s), requires_grad=False)
    method.process_weights_after_loading(layer)
    sw = layer.__dict__.get("_ds41h_skinny")
    assert sw is not None and torch.equal(skinny.unpack(sw)[0], w)
    x = rand_act(3, 5120, g).cuda()
    y = method.apply(layer, x)
    if skinny.APPLY:
        assert method.calls == 0 and torch.allclose(y.float(), skinny.gemm(x, sw).float())
    xbig = rand_act(16, 5120, g).cuda()
    method.apply(layer, xbig)  # > 8 rows -> stock
    assert method.calls == (1 if skinny.APPLY else 2)
    other = torch.nn.Module()
    other.prefix = "model.layers.3.attn.wo_a"
    other.is_bmm = True
    assert skinny.capture(method, other) is None


# =========================================================================================== GPU: prologue


def _layer_info(layer_id: int, rank: int, max_pos: int, dev: str = "cuda") -> prologue.PrologueLayer:
    rd = reader()
    pre = f"layers.{layer_id}."
    g = lambda n: rd.get_tp(pre + n, rank)  # noqa: E731
    cfg = rd.config
    return prologue.PrologueLayer(
        layer_id=layer_id,
        fn=g("hc_attn_fn").float().contiguous().to(dev),
        hc_scale=g("hc_attn_scale").float().contiguous().to(dev),
        hc_base=g("hc_attn_base").float().contiguous().to(dev),
        attn_norm=g("attn_norm.weight").to(torch.bfloat16).contiguous().to(dev),
        wqa=skinny.pack_checkpoint(rd, [pre + "attn.wq_a.weight", pre + "attn.wkv.weight"], None, dev),
        q_norm=g("attn.q_norm.weight").to(torch.bfloat16).contiguous().to(dev),
        kv_norm=g("attn.kv_norm.weight").to(torch.bfloat16).contiguous().to(dev),
        wqb=skinny.pack_checkpoint(rd, [pre + "attn.wq_b.weight"], rank, dev),
        cos_sin=cos_sin_table(cfg["compress_ratios"][layer_id], max_pos, dev),
        rms_eps=float(cfg["rms_norm_eps"]),
        hc_eps=float(cfg["hc_eps"]),
        norm_eps=float(cfg["rms_norm_eps"]),
        qkv_eps=float(cfg["rms_norm_eps"]),
        post_mult=2.0,
        iters=int(cfg["hc_sinkhorn_iters"]),
    )


def _inputs(m: int, seed: int, max_pos: int, dev: str = "cuda") -> dict:
    g = torch.Generator().manual_seed(seed)
    resid = realistic_stream(m, seed=seed).to(torch.bfloat16)
    x_prev = (torch.randn(m, D, generator=g) * resid.float().square().mean().sqrt() * 0.5).to(torch.bfloat16)
    post = (2 * torch.sigmoid(torch.randn(m, HC, 1, generator=g))).float()
    comb = ref.sinkhorn(torch.randn(m, HC, HC, generator=g) * 2).float()
    pre = random_pre_mix(m, seed=seed)
    pos = torch.randint(0, max_pos, (m,), generator=g, dtype=torch.int64)
    return {k: v.to(dev).contiguous() for k, v in
            dict(x_prev=x_prev, residual=resid, post_mix=post, res_mix=comb, pre_mix=pre, positions=pos).items()}  # fmt: skip


def _reference(info: prologue.PrologueLayer, inp: dict, w8a8: bool = False, qr_defer: bool = False) -> dict:
    """float32 torch reference with stock vLLM's bf16 rounding points (see prologue.cu header).
    w8a8: the GEMM inputs a / qr go through the reference act_quant (the wgmma path's numerics)."""
    with ref.strict_fp32():
        return _reference_impl(info, inp, w8a8, qr_defer)


def hc_post_fma_chain(x: torch.Tensor, resid: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Bit-exact emulation of the stock fused post GEMM (tilelang_kernels.py:745-750, same order in
    prologue.cu p1_collapse): t = fl32(post[j] * x); t = fma32(comb[k, j], resid[k], t) for k = 0..3.
    Each fp32 FMA is emulated in float64 (the product of two fp32 values is exact in float64, then one
    rounding of the sum to 53 bits and one to 24 bits: double rounding can differ from a single rounding
    only on ~2^-29 of the inputs). torch's einsum sums in another order without FMA, which under
    cancellation (|new_r| << |terms|) moves the fp32 value by more than one bf16 ulp of the result."""
    x64, r64 = x.double(), resid.double()
    p64, c64 = post.double().reshape(x.shape[0], HC), comb.double()
    out = []
    for j in range(HC):
        t = (p64[:, j : j + 1] * x64).float().double()
        for k in range(HC):
            t = (c64[:, k, j : j + 1] * r64[:, k] + t).float().double()
        out.append(t.float())
    return torch.stack(out, dim=1)


def _reference_impl(info: prologue.PrologueLayer, inp: dict, w8a8: bool = False, qr_defer: bool = False) -> dict:
    aq = (lambda v: ref.quant.act_quant_sim(v, 32)) if w8a8 else (lambda v: v)  # noqa: E731
    n_bf = ref.Numerics(act_quant=False, bf16=True, kv_format="none", idx_format="none")
    x, resid = inp["x_prev"].float(), inp["residual"].float()
    post, comb, pre_in = inp["post_mix"].float(), inp["res_mix"].float(), inp["pre_mix"].float()
    stream = ref.hc_post(x, resid, post, comb)  # fp32 post-mapped stream (feeds the mixes)
    # the bf16 stream (residual output, collapse input) with stock's exact FMA order
    stream_bf = ref.quant.bf16_round(hc_post_fma_chain(x, resid, post, comb))
    pre, post_n, comb_n, _ = ref.hc_mixes(stream, info.fn, info.hc_scale, info.hc_base, norm_eps=info.rms_eps,
                                          hc_eps=info.hc_eps, iters=info.iters)
    collapse = ref.hc_collapse(stream_bf, pre_in, n_bf)
    a = ref.rmsnorm(collapse, info.attn_norm, info.norm_eps, n_bf)
    qkv = ref.quant.bf16_round((aq(a).double() @ skinny.dequant(info.wqa).double().t()).float())
    qr = ref.rmsnorm(qkv[:, :QR], info.q_norm, info.qkv_eps, n_bf)
    kv = ref.rmsnorm(qkv[:, QR:], info.kv_norm, info.qkv_eps, n_bf)
    if qr_defer:  # FAST_QR_DEFER: wq_b on act_quant(bf16(qkv * q_norm_w)), the q_norm rsqrt applied to the fp32 result
        u = ref.quant.bf16_round(qkv[:, :QR] * info.q_norm.float())
        rq = torch.rsqrt(qkv[:, :QR].square().sum(-1, keepdim=True) / QR + info.qkv_eps)
        q = ref.quant.bf16_round(((aq(u).double() @ skinny.dequant(info.wqb).double().t()) * rq.double()).float()).view(-1, HEADS, HDIM)
    else:
        q = ref.quant.bf16_round((aq(qr).double() @ skinny.dequant(info.wqb).double().t()).float()).view(-1, HEADS, HDIM)
    cs = info.cos_sin[inp["positions"]]
    q_rope = fma_rope_bf16(q.to(torch.bfloat16), cs[:, None, :32], cs[:, None, 32:])
    return dict(residual=stream_bf, attn_pre=pre, post_mix=post_n.unsqueeze(-1), res_mix=comb_n, a=a, qr=qr, kv=kv,
                q_pre=q, q=q_rope.float())  # fmt: skip


def _swa_cache(num_blocks: int, block_size: int, dev: str = "cuda") -> torch.Tensor:
    page = -(-block_size * (V4_DATA + V4_SCALES) // 576) * 576  # stock rounds pages to 576 B (attention.py:514)
    return torch.full((num_blocks, page), 0xA5, dtype=torch.uint8, device=dev)


def _expected_records(kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    cs = cos_sin[positions]
    roped = fma_rope_bf16(kv, cs[:, :32], cs[:, 32:])
    return ref.v4_pack(roped.float())


def _check_records(cache: torch.Tensor, before: torch.Tensor, block_size: int, slots: torch.Tensor, n_insert: int,
                   data: torch.Tensor, scales: torch.Tensor, name: str) -> None:
    exp = before.clone()
    for i in range(n_insert):
        s = int(slots[i])
        if s < 0:
            continue
        b, o = s // block_size, s % block_size
        exp[b, o * V4_DATA : (o + 1) * V4_DATA] = data[i]
        base = block_size * V4_DATA + o * V4_SCALES
        exp[b, base : base + V4_SCALES] = scales[i]
    diff = (cache != exp).nonzero()
    assert diff.shape[0] == 0, f"{name}: {diff.shape[0]} cache bytes differ, first at {diff[0].tolist()}"


def _stock_insert_op():
    try:
        import vllm._C_stable_libtorch  # noqa: F401  (registers torch.ops._C)

        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    except Exception as e:  # noqa: BLE001
        print(f"  (stock insert op unavailable: {e!r})")
        return None


def _gemm_close(y: torch.Tensor, xin: torch.Tensor, sw, mma: str, what: str) -> dict:
    """y (kernel, bf16) vs bf16(xin @ W^T) in float64 with xin the GEMM input the kernel saw: 1 bf16 ulp of the
    output + the accumulation slack of _check_skinny (FP8 tensor-core accumulator for wgmma)."""
    wd = skinny.dequant(sw).double()
    yr = xin.double() @ wd.t()
    mag = xin.double().abs() @ wd.abs().t()
    tol = yr.abs() * 2.0**-7 + mag * (2.0**-12 if mma == "wgmma" else 2.0**-18)
    bad = int(((y.double() - yr).abs() > tol).sum())
    m = compare(y.float(), yr.float())
    assert m["finite"] and bad == 0 and m["rel_rmse"] < 3e-3, f"{what}: {bad} beyond tolerance; {fmt_metrics(m)}"
    return m


def _check_chain(info, out: dict, q_pre: torch.Tensor, mma: str, tag: str, qr_defer: bool = False, a_defer: bool = False) -> None:
    """Like-for-like GEMM checks: each GEMM against the reference fed with the kernel's own input (a for
    wq_a|wkv, qr for wq_b), through the same activation quantization (act_quant per token per 32 for wgmma).
    a_defer (FAST_V3_ADEFER): the wq_a|wkv input is act_quant(bf16(collapse * attn_norm_w)) and the attn_norm rsqrt
    multiplies the GEMM output (the kernel's own collapse is read from its scratch)."""
    aq = (lambda v: ref.quant.act_quant_sim(v.float(), 32)) if mma == "wgmma" else (lambda v: v.float())  # noqa: E731
    m = out["a"].shape[0]
    n_bf = ref.Numerics(act_quant=False, bf16=True, kv_format="none", idx_format="none")
    if a_defer:
        sc = prologue.ensure_device(out["a"].device)
        off = int(prologue.ext().OFF_XCB)
        xc = sc[off : off + 8 * D * 2].view(torch.bfloat16).view(8, D)[:m].float()
        u = ref.quant.bf16_round(xc * info.attn_norm.float())
        rs = torch.rsqrt(xc.double().square().mean(-1, keepdim=True) + info.norm_eps)
        xin = aq(u).double() * rs
    else:
        xin = aq(out["a"]).double()
    wd = skinny.dequant(info.wqa).double()
    y64 = xin @ wd.t()
    # per-output bound on the kernel's qkv before its bf16 rounding (fp32 order; FP8 tensor-core accumulator)
    acc_tol = (xin.abs() @ wd.abs().t()) * (2.0**-12 if mma == "wgmma" else 2.0**-18)
    with ref.strict_fp32():
        qkv = ref.quant.bf16_round(y64.float())
        qr = ref.rmsnorm(qkv[:, :QR], info.q_norm, info.qkv_eps, n_bf)
        kv = ref.rmsnorm(qkv[:, QR:], info.kv_norm, info.qkv_eps, n_bf)
    for k, v, sl, w in (("qr", qr, slice(0, QR), info.q_norm), ("kv", kv, slice(QR, None), info.kv_norm)):
        mt = compare(out[k].float(), v.float())
        # bound: qkv error (accumulation + one bf16 rounding of qkv, 2^-8 of |qkv|) through x * r * w, plus the
        # final bf16 rounding (2^-8 of |out|) and the norm statistics (2^-10 relative)
        rr = torch.rsqrt(qkv[:, sl].double().square().mean(-1, keepdim=True) + info.qkv_eps)
        tol = (acc_tol[:, sl] + qkv[:, sl].double().abs() * 2.0**-8) * rr * w.double().abs() + v.double().abs() * (2.0**-8 + 2.0**-10)
        bad = int(((out[k].double() - v.double()).abs() > tol).sum())
        assert mt["cos"] >= 0.99999 and mt["rel_rmse"] <= 3e-3 and bad == 0, f"{tag} chain {k}: {bad} beyond bound {fmt_metrics(mt)}"
    if qr_defer:
        # the kernel's own qkv (scratch) -> u = bf16(qkv * w) -> act_quant -> GEMM -> x rsqrt(mean(qkv_q^2) + eps)
        sc = prologue.ensure_device(out["a"].device)
        off = int(prologue.ext().OFF_QKV)
        qkv_k = sc[off : off + 8 * (QR + KVW) * 2].view(torch.bfloat16).view(8, QR + KVW)[:m].float()
        u = ref.quant.bf16_round(qkv_k[:, :QR] * info.q_norm.float())
        rq = torch.rsqrt(qkv_k[:, :QR].double().square().sum(-1, keepdim=True) / QR + info.qkv_eps)
        _gemm_close(q_pre.view(m, -1).double() / rq, aq(u), info.wqb, mma, f"{tag} chain wq_b (deferred q_norm)")
    else:
        _gemm_close(q_pre.view(m, -1), aq(out["qr"]), info.wqb, mma, f"{tag} chain wq_b")


def _run_case(info, m: int, seed: int, max_pos: int, block_size: int = 64, n_slots: int | None = None,
              with_cache: bool = True, holes: bool = False, mma: str | None = None, fast: int | None = None) -> dict:
    dev = "cuda"
    inp = _inputs(m, seed, max_pos, dev)
    num_blocks = 12
    cache = _swa_cache(num_blocks, block_size, dev)
    g = torch.Generator().manual_seed(seed + 100)
    slots = torch.randperm(num_blocks * block_size, generator=g)[: (n_slots or m)].to(torch.int64)
    if holes and slots.numel() > 1:
        slots[1] = -1
    slots = slots.to(dev)
    before = cache.clone()
    q_pre = torch.empty(m, HEADS * HDIM, dtype=torch.bfloat16, device=dev)
    mma = (mma or prologue.MMA).lower()
    fast = prologue.FAST if fast is None else fast
    out = prologue.run_raw(**{k: inp[k] for k in ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")},
                           info=info, swa=(cache, slots, block_size) if with_cache else None, q_pre=q_pre, mma=mma,
                           fast=fast)
    torch.cuda.synchronize()
    # FAST_QR_DEFER needs FAST_CLUSTER (both bits); FAST_V3 (512) always defers q_norm with wgmma
    qr_defer = mma == "wgmma" and ((fast & 192) == 192 or bool(fast & 512))
    a_defer = mma == "wgmma" and bool(fast & 512) and bool(fast & 4096)
    r = _reference(info, inp, w8a8=mma == "wgmma", qr_defer=qr_defer)
    tag = f"L{info.layer_id} m={m} seed={seed} [{mma} fast={fast}]"
    # residual: same fp32 FMA chain as stock, then bf16 -> bit-exact up to rare 1-ulp ties
    nd, mu = ulp_mismatch(out["residual"], r["residual"].to(torch.bfloat16))
    assert mu <= 1 and nd <= max(2, out["residual"].numel() // 10000), f"{tag} residual: {nd} differ, max {mu} ulp"
    for k, tol in (("attn_pre", 2e-5), ("post_mix", 2e-5), ("res_mix", 5e-5)):
        mt = compare(out[k], r[k])
        assert mt["max_abs"] < tol, f"{tag} {k}: {fmt_metrics(mt)}"
    # End to end vs the reference. With wgmma, qr is re-quantized to e4m3 (per token per 32) before wq_b: a
    # 1-ulp bf16 difference upstream (fp32 accumulation order in wq_a|wkv) flips whole e4m3 steps (2^-4
    # relative) or a block's ue8m0 scale, so q differs from the reference by more than the GEMM's own
    # error; the like-for-like checks below feed the reference GEMMs the kernel's own a / qr instead.
    q_rr, q_cos = (2e-2, 0.9998) if mma == "wgmma" else (6e-3, 0.9999)
    qkv_rr, qkv_cos = 4e-3, 0.99995
    if a_defer:  # a is quantized before its rsqrt scaling: an independent e4m3 rounding of every activation, the size of
        # the W8A8-vs-W8A16 difference itself (the like-for-like chain checks below stay exact)
        q_rr, q_cos, qkv_rr, qkv_cos = 6e-2, 0.998, 6e-2, 0.998
    for k, cos_min, rr in (("a", 0.99999, 2e-3), ("qr", qkv_cos, qkv_rr), ("kv", qkv_cos, qkv_rr),
                           ("q", q_cos, q_rr)):  # fmt: skip
        mt = compare(out[k].float(), r[k].float().view_as(out[k]))
        assert mt["finite"] and mt["cos"] >= cos_min and mt["rel_rmse"] <= rr, f"{tag} {k}: {fmt_metrics(mt)}"
    _check_chain(info, out, q_pre, mma, tag, qr_defer, a_defer)
    # q RoPE: exact given the kernel's own pre-RoPE q
    cs = info.cos_sin[inp["positions"]]
    q_exp = fma_rope_bf16(q_pre.view(m, HEADS, HDIM), cs[:, None, :32], cs[:, None, 32:])
    assert torch.equal(out["q"], q_exp), f"{tag}: q RoPE not exact"
    if with_cache:
        data, scales = _expected_records(out["kv"], inp["positions"], info.cos_sin)
        _check_records(cache, before, block_size, slots.cpu(), min(m, slots.numel()), data, scales, tag)
        op = _stock_insert_op()
        if op is not None:
            cache2 = before.clone()
            qp = op(q_pre.view(m, HEADS, HDIM), out["kv"], cache2, slots, inp["positions"], info.cos_sin, 64,
                    info.qkv_eps, block_size, False, False, True, False)
            torch.cuda.synchronize()
            assert torch.equal(cache2, cache), f"{tag}: cache differs from stock writer"
            assert torch.equal(qp[:, :HEADS], out["q"]), f"{tag}: q differs from stock RoPE"
            assert int(qp[:, HEADS:].abs().sum()) == 0
    else:
        assert torch.equal(cache, before), f"{tag}: cache written without metadata"
    print(f"  {tag}: residual {nd} ulp-diffs; a {fmt_metrics(compare(out['a'].float(), r['a']))}")
    out["q_pre"] = q_pre
    out["cache"] = cache
    return out


def test_prologue_vs_reference():
    require_cuda()
    max_pos = 70000
    for layer_id in (3, 20):
        info = _layer_info(layer_id, rank=1, max_pos=max_pos)
        for m in (1, 2, 3, 6, 7, 8):
            _run_case(info, m, seed=10 * layer_id + m, max_pos=max_pos)


def test_prologue_cache_edge_cases():
    require_cuda()
    info = _layer_info(5, rank=0, max_pos=4096)
    _run_case(info, 6, seed=1, max_pos=4096, holes=True)  # slot -1 skipped
    _run_case(info, 6, seed=2, max_pos=4096, n_slots=4)  # shorter slot mapping (padding rows)
    _run_case(info, 5, seed=3, max_pos=4096, with_cache=False)  # profile run: no insert
    _run_case(info, 8, seed=4, max_pos=4096, block_size=32)  # other page size


def test_prologue_wgmma_vs_reference():
    """RS-wgmma prologue vs the W8A8 reference (act_quant on a and qr), both latency modes."""
    require_cuda()
    max_pos = 70000
    for layer_id in (3, 20):
        info = _layer_info(layer_id, rank=1, max_pos=max_pos)
        for m in (1, 2, 3, 6, 7, 8):
            for fast in (0, 99, 227, 512, 1536, 2562, 6658, 6658 | 1024 | 8192, 6658 | 16384 | 1024):
                _run_case(info, m, seed=10 * layer_id + m, max_pos=max_pos, mma="wgmma", fast=fast)
    info = _layer_info(5, rank=0, max_pos=4096)
    _run_case(info, 6, seed=1, max_pos=4096, holes=True, mma="wgmma", fast=227)
    _run_case(info, 5, seed=3, max_pos=4096, with_cache=False, mma="wgmma", fast=227)
    _run_case(info, 6, seed=1, max_pos=4096, holes=True, mma="wgmma", fast=512)
    _run_case(info, 5, seed=3, max_pos=4096, with_cache=False, mma="wgmma", fast=512)


def test_prologue_v3_vs_reference():
    """FAST_V3 schedule (own collapse, smem weights, DSMEM reductions) vs the reference, HMMA (stock numerics)."""
    require_cuda()
    prologue.ensure_device("cuda")
    e = prologue.ext()
    for code in (0, 1):  # otherwise the launcher silently runs the regular schedule
        assert int(e.max_resident_ctas_v3(code)) >= int(e.CTAS_V3), f"v3 grid does not fit (mma {code}): {int(e.max_resident_ctas_v3(code))}"
    max_pos = 70000
    for layer_id in (3, 20):
        info = _layer_info(layer_id, rank=1, max_pos=max_pos)
        for m in (1, 2, 3, 6, 7, 8):
            _run_case(info, m, seed=10 * layer_id + m, max_pos=max_pos, mma="hmma", fast=512)
    info = _layer_info(5, rank=0, max_pos=4096)
    _run_case(info, 6, seed=1, max_pos=4096, holes=True, mma="hmma", fast=512)
    _run_case(info, 6, seed=2, max_pos=4096, n_slots=4, mma="hmma", fast=512)
    _run_case(info, 5, seed=3, max_pos=4096, with_cache=False, mma="hmma", fast=512)
    _run_case(info, 8, seed=4, max_pos=4096, block_size=32, mma="hmma", fast=512 | 1024)
    for m in (1, 6, 8):  # FAST_V3_HOP: P1a hop instead of the own collapse
        _run_case(info, m, seed=20 + m, max_pos=4096, mma="hmma", fast=512 | 2048)
        _run_case(info, m, seed=20 + m, max_pos=4096, mma="wgmma", fast=512 | 2048 | 2)


def test_prologue_fast_mode_equivalence():
    """DS41H_PROLOGUE_FAST only reorders independent work (early G1) and swaps the Sinkhorn divisions for
    rcp: every output except res_mix is bit-identical to fast=0; res_mix within the stock tolerance."""
    require_cuda()
    info = _layer_info(9, rank=2, max_pos=8192)
    base = {}
    for mma in ("hmma", "wgmma"):
        for m in (1, 6, 8):
            ref_out = _run_case(info, m, seed=70 + m, max_pos=8192, mma=mma, fast=0)
            for fast in (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 35, 64, 99, 355):
                out = _run_case(info, m, seed=70 + m, max_pos=8192, mma=mma, fast=fast)
                for k in ("residual", "post_mix", "attn_pre", "a", "qr", "kv", "q", "q_pre", "cache"):
                    assert torch.equal(out[k], ref_out[k]), f"{mma} m={m} fast={fast}: {k} differs from fast=0"
                d = (out["res_mix"] - ref_out["res_mix"]).abs().max().item()
                assert (d == 0.0) if not (fast & 2) else (d < 2e-5), f"{mma} m={m} fast={fast}: res_mix off by {d:.3e}"
            # FAST_V3: HMMA bit-identical to fast=0; wgmma bit-identical to the cluster + QR_DEFER schedule (224)
            v3_ref = ref_out if mma == "hmma" else _run_case(info, m, seed=70 + m, max_pos=8192, mma=mma, fast=224)
            for fast in (512, 512 | 1024, 512 | 3, 512 | 2048, 512 | 1024 | 2048, 512 | 1024 | 2048 | 8192, 512 | 2048 | 16384,
                         512 | 16384, 18944 | 65536, 18944 | 131072, 18944 | 65536 | 131072, 512 | 65536 | 131072,
                         18944 | 262144, 18944 | 65536 | 131072 | 262144, 18944 | 524288, 18944 | 65536 | 131072 | 524288,
                         18944 | 65536 | 131072 | 262144 | 524288, 512 | 524288, 1985024, 4082176, 8276480, 18944 | 4194304,
                         512 | 2048 | 4194304, 512 | 4194304, 18944 | 1048576, 18944 | 2097152, 4082176 | 8388608,
                         18944 | 1048576 | 2097152 | 8388608, 4082176 | 1024, 12470784 | 1024,
                         4083200 | 1, 4083200 | 33554432, 4083200 | 67108864, 4081152 | 67108864, 4083200 | 134217728,
                         4081152 | 134217728, 3100160 | 134217728):
                # 4083200 = default (with DS41H_PROLOGUE_FIXED=1: the specialized kernel); | 1: runtime bits; 33554432 ACQPOLL;
                # 67108864 SKREG / 134217728 SKROW (Sinkhorn with shared-reciprocal exact divisions: res_mix bit-identical)
                out = _run_case(info, m, seed=70 + m, max_pos=8192, mma=mma, fast=fast)
                for k in ("residual", "post_mix", "attn_pre", "a", "qr", "kv", "q", "q_pre", "cache"):
                    assert torch.equal(out[k], v3_ref[k]), f"{mma} m={m} fast={fast}: {k} differs from the reference schedule"
                d = (out["res_mix"] - v3_ref["res_mix"]).abs().max().item()
                assert (d == 0.0) if not (fast & 2) else (d < 2e-5), f"{mma} m={m} fast={fast}: res_mix off by {d:.3e}"
            # everything upstream of the GEMMs is MMA-independent (same code / same rounding)
            if m in base:
                for k in ("residual", "post_mix", "res_mix", "attn_pre", "a"):
                    assert torch.equal(ref_out[k], base[m][k]), f"m={m}: {k} differs between hmma and wgmma"
                mq = compare(ref_out["q"].float(), base[m]["q"].float())
                print(f"  m={m}: q wgmma vs hmma {fmt_metrics(mq)}")
            base[m] = ref_out
    # SKREG / SKROW range guard: hc_eps below 2^-60 sends every token to the reference Sinkhorn (sinkhorn_job) -> identical
    import dataclasses

    info0 = dataclasses.replace(info, hc_eps=1e-30)
    ref0 = _run_case(info0, 6, seed=77, max_pos=8192, mma="hmma", fast=0)
    for fast in (4083200 | 67108864, 4083200 | 134217728):
        out = _run_case(info0, 6, seed=77, max_pos=8192, mma="hmma", fast=fast)
        for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "q"):
            assert torch.equal(out[k], ref0[k]), f"hc_eps=1e-30 fast={fast}: {k} differs (range-guard fallback)"


def test_prologue_vs_stock_mhc():
    """mHC outputs vs stock mhc_fused_post_pre_delayed_tilelang (the sm90 mhc_shifted_post_pre path)."""
    require_cuda()
    try:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_delayed_tilelang
    except Exception as e:  # noqa: BLE001
        skip(f"stock mHC unavailable: {e!r}")
    info = _layer_info(7, rank=2, max_pos=4096)
    for m in (1, 6, 8):
        inp = _inputs(m, 50 + m, 4096)
        try:
            res, post, comb, a, pre, _ = mhc_fused_post_pre_delayed_tilelang(
                inp["x_prev"], inp["residual"], inp["post_mix"], inp["res_mix"], info.fn, info.hc_scale, info.hc_base,
                info.rms_eps, info.hc_eps, info.hc_eps, info.post_mult, info.iters, pre_mix=inp["pre_mix"],
                norm_weight=info.attn_norm, norm_eps=info.norm_eps)  # fmt: skip
        except Exception as e:  # noqa: BLE001 (needs a vLLM config context / TileLang JIT)
            skip(f"stock mHC call failed: {e!r}")
        for fast in (0, 99, prologue.FAST_DEFAULT, prologue.FAST_DEFAULT | 65536 | 131072):
            out = prologue.run_raw(**{k: inp[k] for k in ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")},
                                   info=info, fast=fast)
            torch.cuda.synchronize()
            print(f"  m={m} fast={fast} vs stock mHC: " + " | ".join(
                f"{k} max_abs {(out[k] - s.view_as(out[k])).abs().max().item():.2e}"
                for k, s in (("post_mix", post), ("res_mix", comb), ("attn_pre", pre))))
            nd, mu = ulp_mismatch(out["residual"], res)
            assert mu <= 1 and nd <= max(2, res.numel() // 10000), f"m={m} residual vs stock: {nd} differ ({mu} ulp)"
            for k, s, tol in (("post_mix", post, 2e-5), ("res_mix", comb, 5e-5), ("attn_pre", pre, 2e-5)):
                mt = compare(out[k], s.view_as(out[k]))
                assert mt["max_abs"] < tol, f"m={m} fast={fast} {k} vs stock: {fmt_metrics(mt)}"
            nd, mu = ulp_mismatch(out["a"], a)
            assert mu <= 2 and nd <= max(4, a.numel() // 1000), f"m={m} a vs stock: {nd} differ ({mu} ulp)"
            print(f"  m={m} fast={fast}: residual/a vs stock ok ({nd} a-diffs)")


def _graph_replay_case(mma: str | None, fast: int | None) -> None:
    info = _layer_info(4, rank=3, max_pos=4096)
    m = 6
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    static = _inputs(m, 0, 4096)
    cache = _swa_cache(8, 64)
    slots = torch.arange(m, dtype=torch.int64, device="cuda") * 7
    kw = dict(info=info, swa=(cache, slots, 64), mma=mma, fast=fast)
    prologue.run_raw(**{k: static[k] for k in keys}, **kw)  # warm up / build
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = prologue.run_raw(**{k: static[k] for k in keys}, **kw)
    for seed in (11, 12, 13):
        new = _inputs(m, seed, 4096)
        for k in keys:
            static[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        cache_g = cache.clone()
        eager = prologue.run_raw(**{k: new[k] for k in keys}, **kw)
        torch.cuda.synchronize()
        for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
            assert torch.equal(gout[k], eager[k]), f"replay seed={seed} [{mma} fast={fast}]: {k} differs from eager"
        assert torch.equal(cache_g, cache), f"replay seed={seed} [{mma} fast={fast}]: cache differs from eager"


def test_prologue_cuda_graph_replay():
    """Capture once, replay with new inputs several times: outputs equal eager runs (counters reset)."""
    require_cuda()
    _graph_replay_case(None, None)


def test_prologue_wgmma_cuda_graph_replay():
    require_cuda()
    _graph_replay_case("wgmma", 99)
    _graph_replay_case("wgmma", 227)
    _graph_replay_case("hmma", 99)
    _graph_replay_case("hmma", 33)
    _graph_replay_case("wgmma", 33)
    _graph_replay_case("hmma", 512)
    _graph_replay_case("wgmma", 512)
    _graph_replay_case("hmma", 18946)  # v3 + P1a hop + paced weight stream + rcp Sinkhorn
    _graph_replay_case("wgmma", 18946)
    _graph_replay_case("wgmma", 23042)  # + ADEFER
    _graph_replay_case("hmma", 18944 | 65536 | 131072)  # + Sinkhorn on idle G2 CTAs + precomputed HMMA factors
    _graph_replay_case("hmma", 18944 | 65536 | 131072 | 262144)  # + interleaved G1 / G2 clusters
    _graph_replay_case("hmma", 18944 | 65536 | 131072 | 524288)  # + pushed k-part partials
    _graph_replay_case("hmma", 18944 | 65536 | 131072 | 262144 | 524288)
    _graph_replay_case("hmma", 8276480)  # + Sinkhorn / KV insert spread over G1 CTAs + L2-prefetched weights
    _graph_replay_case("hmma", 4082176)  # the default (no L2PF)
    _graph_replay_case("hmma", 4082176 | 1)  # + legacy EARLY bit (ignored by v3)
    _graph_replay_case("hmma", 4082176 | 8388608)  # + Sinkhorn / KV hosts spread over the kv clusters
    _graph_replay_case("hmma", 4083200)  # default (+ LATE_W2)


def test_prologue_attn_flag_handoff():
    """DS41H_ATTN_FLAG: a prologue launched with aflag bumps the handoff counter once per G2 CTA + once per KV insert; the
    flag-mode sparse decode attention on that prologue's q equals the PDL-wait attention bit for bit and leaves the counter
    at 0 (eager pairs for m = 1 / 6 / 8 and three fast modes incl. the regular kernel, a reset after an unconsumed launch,
    and a replayed CUDA graph of two prologue -> attention layers with new inputs per replay)."""
    require_cuda()
    import test_attn as ta

    from ds41h import attn

    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    infos = [_layer_info(i, rank=0, max_pos=4096) for i in (4, 5)]
    ctr = prologue.handoff_counter("cuda")
    ctr.zero_()
    for m in (1, 6, 8):
        tgt = prologue.handoff_target(m)
        cache = _swa_cache(8, 64)
        slots = torch.arange(m, dtype=torch.int64, device="cuda") * 7
        case = ta.make_case([1500], [m], ratio=1, seed=31 + m)
        aout = torch.empty(m, 64, 512, dtype=torch.bfloat16, device="cuda")
        for fast in (None, 18944, 33):
            inp = _inputs(m, 100 + m, 4096)
            o = prologue.run_raw(**{k: inp[k] for k in keys}, info=infos[0], swa=(cache, slots, 64), fast=fast, aflag=True)
            ta.run_ds41h({**case, "q": o["q"]}, out=aout, aflag=ctr, atarget=tgt)
            torch.cuda.synchronize()
            assert ctr.tolist() == [0, 0], f"m={m} fast={fast}: counter not reset: {ctr.tolist()}"
            ref_out = ta.run_ds41h({**case, "q": o["q"]})
            torch.cuda.synchronize()
            assert torch.equal(aout[:, :HEADS], ref_out[:, :HEADS]), f"m={m} fast={fast}: flag-mode attention differs"
            o2 = prologue.run_raw(**{k: inp[k] for k in keys}, info=infos[0], swa=(cache, slots, 64), fast=fast)
            for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
                assert torch.equal(o[k], o2[k]), f"m={m} fast={fast}: {k} differs with aflag"
        # an unconsumed flag-mode launch (stock attention fallback) is reset by the reset kernel
        prologue.run_raw(**{k: inp[k] for k in keys}, info=infos[0], swa=(cache, slots, 64), aflag=True)
        attn.ext().aflag_reset(ctr)
        torch.cuda.synchronize()
        assert ctr.tolist() == [0, 0]
    # CUDA graph: two layers of [prologue (aflag) -> flag-mode attention], replayed with new inputs
    m = 6
    tgt = prologue.handoff_target(m)
    cache = _swa_cache(8, 64)
    slots = torch.arange(m, dtype=torch.int64, device="cuda") * 7
    case = ta.make_case([1500], [m], ratio=1, seed=77)
    static = _inputs(m, 0, 4096)
    aouts = [torch.empty(m, 64, 512, dtype=torch.bfloat16, device="cuda") for _ in infos]
    sep = torch.zeros(256, device="cuda")

    def step(inputs, outs, flag):
        res = []
        for info, ao in zip(infos, outs):
            o = prologue.run_raw(**{k: inputs[k] for k in keys}, info=info, swa=(cache, slots, 64), aflag=flag)
            ta.run_ds41h({**case, "q": o["q"]}, out=ao, **(dict(aflag=ctr, atarget=tgt) if flag else {}))
            # a regular (non-PDL) kernel between layers, like the decoder layer kernel in production: a v3 prologue
            # PDL-launched DIRECTLY behind a sparse decode attention (whose CTAs trigger at entry) can deadlock -- with or
            # without the flag -- because its 30 clusters of 8 are partly placed next to the resident attention CTAs and
            # the last clusters then find no GPC with 8 free slots (P->A->P graphs hang, P->A->X->P do not)
            sep.add_(1.0)
            res.append(o)
        return res

    step(static, aouts, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gres = step(static, aouts, True)
    for seed in (21, 22, 23):
        new = _inputs(m, seed, 4096)
        for k in keys:
            static[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        assert ctr.tolist() == [0, 0], f"graph replay seed={seed}: counter not reset"
        g_q = [r["q"].clone() for r in gres]
        g_a = [a.clone() for a in aouts]
        e_outs = [torch.empty_like(a) for a in aouts]
        eres = step(new, e_outs, False)
        torch.cuda.synchronize()
        for i in range(len(infos)):
            assert torch.equal(g_q[i], eres[i]["q"]), f"graph replay seed={seed} layer {i}: q differs"
            assert torch.equal(g_a[i][:, :HEADS], e_outs[i][:, :HEADS]), f"graph replay seed={seed} layer {i}: attention differs"


def test_prologue_aux():
    """DS41H_PROLOGUE_AUX: out["aux"] (written by the P1a jobs / the own-collapse group) equals stock's capture_aux expression
    residual.mean(dim=1) bit for bit, and every other output is unchanged, for the HOP, own-collapse and regular schedules."""
    require_cuda()
    info = _layer_info(36, rank=0, max_pos=4096)
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    for m in (1, 6, 8):
        for fast in (None, 4081152, 33, 0):
            inp = _inputs(m, 500 + m, 4096)
            out = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, fast=fast, aux=True)
            torch.cuda.synchronize()  # never PDL-chain two v3 prologues back to back (cluster placement can deadlock)
            base = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, fast=fast)
            torch.cuda.synchronize()
            ref = out["residual"].mean(dim=1)
            assert torch.equal(out["aux"], ref), f"m={m} fast={fast}: aux != residual.mean(dim=1) ({(out['aux'].float() - ref.float()).abs().max().item():.3e})"
            for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
                assert torch.equal(out[k], base[k]), f"m={m} fast={fast}: {k} differs with aux"


def test_prologue_first_layer():
    """first_layer=True (model entry: stock `residual is None` with pre_mix None, the DSpark drafter's first layer): every
    output bit-identical to the regular launch on the explicit broadcast stream (residual = x x4, post 0, comb I, pre e0),
    residual == x broadcast, and post / comb / pre / a vs stock mhc_pre_delayed_tilelang (tf32 prenorm GEMM -> tolerance)."""
    require_cuda()
    info = _layer_info(9, rank=1, max_pos=4096)
    keys = ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q")
    try:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    except Exception as e:  # noqa: BLE001
        mhc_pre_delayed_tilelang = None
        print(f"  stock mhc_pre unavailable ({e!r}): explicit-stream equivalence only")
    for m in (1, 6, 8):
        inp = _inputs(m, 900 + m, 4096)
        x = inp["residual"][:, 0].contiguous()  # embedding-like magnitudes
        for fast in (None, 4081152, 0):
            out = prologue.run_raw(x, None, None, None, None, inp["positions"], info=info, fast=fast, first_layer=True)
            eye = torch.eye(HC, device=x.device).expand(m, HC, HC).contiguous()
            pre_e0 = torch.zeros(m, HC, device=x.device)
            pre_e0[:, 0] = 1.0
            base = prologue.run_raw(x, x.unsqueeze(1).expand(-1, HC, -1).contiguous(), torch.zeros(m, HC, 1, device=x.device),
                                    eye, pre_e0, inp["positions"], info=info, fast=fast)
            torch.cuda.synchronize()
            assert torch.equal(out["residual"], x.unsqueeze(1).expand(-1, HC, -1)), f"m={m} fast={fast}: residual != x broadcast"
            for k in keys:
                assert torch.equal(out[k], base[k]), f"m={m} fast={fast}: {k} differs from the explicit broadcast stream"
        if mhc_pre_delayed_tilelang is None:
            continue
        try:
            post, comb, a, pre = mhc_pre_delayed_tilelang(
                x.unsqueeze(1).expand(-1, HC, -1).contiguous(), info.fn, info.hc_scale, info.hc_base, info.rms_eps,
                info.hc_eps, info.hc_eps, info.post_mult, info.iters, pre_mix=None, norm_weight=info.attn_norm,
                norm_eps=info.norm_eps)  # fmt: skip
        except Exception as e:  # noqa: BLE001 (needs a vLLM config context / TileLang JIT)
            print(f"  stock mhc_pre call failed ({e!r}): explicit-stream equivalence only")
            mhc_pre_delayed_tilelang = None
            continue
        torch.cuda.synchronize()
        msg = " | ".join(f"{k} max_abs {(out[k] - st.view_as(out[k])).abs().max().item():.2e}"
                         for k, st in (("post_mix", post), ("res_mix", comb), ("attn_pre", pre)))  # fmt: skip
        nd, mu = ulp_mismatch(out["a"], a)
        print(f"  m={m} first_layer vs stock mhc_pre (tf32 mixes): {msg} | a {nd} differ, max {mu} ulp")
        for k, st in (("post_mix", post), ("res_mix", comb), ("attn_pre", pre)):
            mt = compare(out[k], st.view_as(out[k]))
            assert mt["max_abs"] < 5e-3, f"m={m} {k} vs stock mhc_pre: {fmt_metrics(mt)}"
        assert mu <= 2 and nd <= max(4, a.numel() // 1000), f"m={m} a vs stock mhc_pre: {nd} differ ({mu} ulp)"


def test_prologue_ext_p1a():
    """FAST_V3_EXT_P1A (experimental: residual / xcb / xsq produced by the previous grid): after a regular launch on the same
    inputs (which leaves xcb / xsq in the scratch), an EXT_P1A launch gives bit-identical outputs (its residual output is
    not written: the producer owns it)."""
    require_cuda()
    info = _layer_info(9, rank=2, max_pos=4096)
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    ext_bit = int(prologue.ext().FAST_V3_EXT_P1A)
    for m in (1, 6, 8):
        for fast in (4083200, 4083200 | 134217728):
            inp = _inputs(m, 700 + m, 4096)
            base = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, fast=fast)
            # p1a_buffers must alias the scratch the kernel reads: save, clobber, restore through the views
            xcb, xsq = prologue.p1a_buffers(inp["x_prev"].device)
            assert xcb.shape == (8, 5120) and xsq.shape == (8, 160) and xcb.dtype == torch.bfloat16 and xsq.dtype == torch.float32
            sx, sq = xcb.clone(), xsq.clone()
            xcb.zero_()
            xsq.zero_()
            xcb.copy_(sx)
            xsq.copy_(sq)
            out = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, fast=fast | ext_bit)
            torch.cuda.synchronize()
            for k in ("post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
                assert torch.equal(out[k], base[k]), f"m={m} fast={fast}|EXT_P1A: {k} differs"
            # and the views really are what the kernel reads: a zeroed collapse must change a
            xcb.zero_()
            bad = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, fast=fast | ext_bit)
            torch.cuda.synchronize()
            assert not torch.equal(bad["a"], base["a"]), f"m={m}: p1a_buffers xcb does not alias the kernel's collapse input"


def test_prologue_indexer_projections():
    """DS41H_PROLOGUE_INDEXER: on an index-source layer the prologue also returns idx_q = indexer.wq_b(qr) and idx_w =
    indexer.weights_proj(a). idx_q vs the skinny drop-in (the GEMM stock runs on qr) and a float64 reference, idx_w vs the
    fp32 GEMM of the bf16 inputs (exact products) rounded to bf16; every other output bit-identical to the launch without
    the projections; m = 1 / 6 / 8, three fast modes, and a CUDA-graph replay with new inputs."""
    require_cuda()
    rd = reader()
    lid = 24  # an index-source layer (config index_source_layer_ids) without compressor
    info = _layer_info(lid, rank=0, max_pos=4096)
    info.iwq = skinny.pack_checkpoint(rd, [f"layers.{lid}.attn.indexer.wq_b.weight"], None, "cuda")
    info.iwp = rd.get_tp(f"layers.{lid}.attn.indexer.weights_proj.weight", None).to(torch.bfloat16).contiguous().cuda()
    assert (info.iwq.n, info.iwq.k) == (4096, QR) and tuple(info.iwp.shape) == (32, 5120)
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    for m in (1, 6, 8):
        cache = _swa_cache(8, 64)
        slots = torch.arange(m, dtype=torch.int64, device="cuda") * 5
        for fast in (None, 4083200 | 16777216, 4081152):
            inp = _inputs(m, 300 + m, 4096)
            c1 = cache.clone()
            out = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(c1, slots, 64), fast=fast, indexer=True)
            torch.cuda.synchronize()  # no back-to-back PDL v3 prologues (cluster placement deadlock)
            c2 = cache.clone()
            base = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(c2, slots, 64), fast=fast)
            torch.cuda.synchronize()
            tag = f"m={m} fast={fast}"
            assert "idx_q" in out and "idx_w" in out, f"{tag}: no indexer outputs"
            for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
                assert torch.equal(out[k], base[k]), f"{tag}: {k} differs with the indexer projections"
            assert torch.equal(c1, c2), f"{tag}: SWA cache differs with the indexer projections"
            ref_q = skinny.gemm(out["qr"], info.iwq, mma="hmma")
            q64 = out["qr"].double() @ skinny.dequant(info.iwq).double().T
            mq = compare(out["idx_q"].float(), ref_q.float())
            nd, mu = ulp_mismatch(out["idx_q"], ref_q)
            # bit-identical: the kernel reproduces the drop-in's k-partition (4 parts of 12 k32 blocks, kernels/prologue.cu IDXQ_KPC)
            assert torch.equal(out["idx_q"], ref_q), f"{tag}: idx_q vs skinny drop-in: {nd} differ ({mu} ulp), {fmt_metrics(mq)}"
            mq64 = compare(out["idx_q"].double(), q64)
            assert mq64["rel_rmse"] < 3e-3, f"{tag}: idx_q vs float64: {fmt_metrics(mq64)}"
            ref_w = (out["a"].float() @ info.iwp.float().T).to(torch.bfloat16)
            nd, mu = ulp_mismatch(out["idx_w"], ref_w)
            assert mu <= 1 and nd <= max(4, out["idx_w"].numel() // 50), f"{tag}: idx_w vs fp32 GEMM: {nd} differ ({mu} ulp)"
            print(f"  {tag}: idx_q vs skinny {fmt_metrics(mq)}; idx_w ok")
            # DS41H_PROLOGUE_INDEXER default (no INDEXER_W): idx_q only, weights_proj stays stock; same idx_q, other outputs unchanged
            c3 = cache.clone()
            out_q = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(c3, slots, 64), fast=fast, indexer=True, indexer_w=False)
            torch.cuda.synchronize()
            assert "idx_w" not in out_q and torch.equal(out_q["idx_q"], out["idx_q"]), f"{tag}: idx_q-only launch differs"
            for k in ("residual", "post_mix", "res_mix", "attn_pre", "a", "qr", "kv", "q"):
                assert torch.equal(out_q[k], base[k]), f"{tag}: {k} differs in the idx_q-only launch"
    # graph replay with new inputs
    m = 6
    cache = _swa_cache(8, 64)
    slots = torch.arange(m, dtype=torch.int64, device="cuda") * 5
    static = _inputs(m, 0, 4096)
    kw = dict(info=info, swa=(cache, slots, 64), indexer=True)
    prologue.run_raw(**{k: static[k] for k in keys}, **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = prologue.run_raw(**{k: static[k] for k in keys}, **kw)
    for seed in (41, 42):
        new = _inputs(m, seed, 4096)
        for k in keys:
            static[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        eager = prologue.run_raw(**{k: new[k] for k in keys}, **kw)
        torch.cuda.synchronize()
        for k in ("q", "qr", "idx_q", "idx_w"):
            assert torch.equal(gout[k], eager[k]), f"indexer graph replay seed={seed}: {k} differs from eager"


def test_prologue_indexer_fold():
    """DS41H_PROLOGUE_INDEXER_FOLD: the prologue's indexer q (RoPE + per-head FP8 quant) and weights_out are bit-identical to stock
    fused_indexer_q_rope_quant (vLLM's CuTe IndexerQFp8Kernel on sm90) applied to the prologue's idx_q / idx_w, and the top-k
    buffer rows [0, m) are cleared to -1 (rows >= m untouched); m = 1 / 6 / 8 and a CUDA-graph replay."""
    require_cuda()
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant

    rd = reader()
    lid = 24
    info = _layer_info(lid, rank=0, max_pos=4096)
    info.iwq = skinny.pack_checkpoint(rd, [f"layers.{lid}.attn.indexer.wq_b.weight"], None, "cuda")
    info.iwp = rd.get_tp(f"layers.{lid}.attn.indexer.weights_proj.weight", None).to(torch.bfloat16).contiguous().cuda()
    g = torch.Generator(device="cuda").manual_seed(5)
    info.icos = (torch.rand(4096, ROPE, generator=g, device="cuda") * 2 - 1).contiguous()
    info.iwscale = float(128 ** -0.5 * 32 ** -0.5)
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    for m in (1, 6, 8):
        for seed in (0, 1, 2):
            info.itopk = torch.full((16, 512), 7, dtype=torch.int32, device="cuda")
            cache = _swa_cache(8, 64)
            slots = torch.arange(m, dtype=torch.int64, device="cuda") * 5
            inp = _inputs(m, 700 + 10 * m + seed, 4096)
            out = prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(cache, slots, 64), indexer=True, fold=True)
            torch.cuda.synchronize()
            rq, rw = fused_indexer_q_rope_quant(inp["positions"][:m], out["idx_q"].view(m, 32, 128), info.icos, out["idx_w"],
                                                128 ** -0.5, 32 ** -0.5)
            torch.cuda.synchronize()
            tag = f"m={m} seed={seed}"
            dq = (out["idx_qf"].view(torch.uint8).view(m, 32, 128) != rq.view(torch.uint8)).sum().item()
            dw = (out["idx_wf"].view(torch.int32) != rw.view(torch.int32)).sum().item()
            assert dq == 0 and dw == 0, f"{tag}: fold vs stock CuTe: {dq} q bytes / {dw} weights differ"
            assert (info.itopk[:m] == -1).all() and (info.itopk[m:] == 7).all(), f"{tag}: top-k buffer clear wrong"
            print(f"  {tag}: fold bit-identical to the stock CuTe kernel")
    m = 8
    static = _inputs(m, 0, 4096)
    cache = _swa_cache(8, 64)
    slots = torch.arange(m, dtype=torch.int64, device="cuda") * 5
    kw = dict(info=info, swa=(cache, slots, 64), indexer=True, fold=True)
    prologue.run_raw(**{k: static[k] for k in keys}, **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = prologue.run_raw(**{k: static[k] for k in keys}, **kw)
    for seed in (51, 52):
        new = _inputs(m, seed, 4096)
        for k in keys:
            static[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        eager = prologue.run_raw(**{k: new[k] for k in keys}, **kw)
        torch.cuda.synchronize()
        for k in ("idx_qf", "idx_wf", "idx_q", "idx_w"):
            assert torch.equal(gout[k].view(torch.uint8), eager[k].view(torch.uint8)), f"fold graph replay seed={seed}: {k} differs"


# =========================================================================================== benchmark


def _graph_ms(fn, n: int, iters: int) -> float:
    """Median ms per call of fn(i), i = 0..n-1 captured back to back in one CUDA graph."""
    for i in range(n):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(n):
            fn(i)
    ms = bench(graph.replay, warmup=3, iters=max(10, iters // 10)) / n
    del graph
    return ms


def bench_main(iters: int) -> None:
    require_cuda()
    dev = "cuda"
    print("== skinny GEMM (weights rotated over copies to defeat L2), hmma vs wgmma")
    g = torch.Generator().manual_seed(0)
    for name, (n, k) in SKINNY_SHAPES.items():
        copies = max(2, int(math.ceil(160e6 / (n * k))))
        sws = [skinny.pack(rand_e4m3(n, k, g).to(dev), rand_scales(n, k, g).to(dev)) for _ in range(copies)]
        wbf = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        for m in (1, 6, 8):
            x = rand_act(m, k, g).to(dev)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            ms_bf = _graph_ms(lambda i: torch.matmul(x, wbf.t()), 8, iters)
            for mma in ("hmma", "wgmma"):
                # graph of `copies` back-to-back calls on rotating weight copies (no host gaps, weights from HBM)
                ms = _graph_ms(lambda i, mma=mma: skinny.gemm(x, sws[i], out, mma=mma), copies, iters)
                print(fmt_bw(f"{name} {n}x{k} m={m} {mma:<5s} {skinny.split_config(n, k, mma)}", ms, nbytes=n * k + n * k // 512)
                      + f"   (cuBLAS bf16 W, L2-hot: {ms_bf * 1e3:.1f} us)")
        del sws
    print("== fused prologue (layers 2..9 rotated); mma x fast (3: early G1 + rcp Sinkhorn, 7: + entry wq_b prefetch)")
    infos = [_layer_info(i, rank=0, max_pos=8192) for i in range(2, 10)]
    wbytes = infos[0].fn.numel() * 4 + infos[0].wqa.nbytes + infos[0].wqb.nbytes
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    for m in (1, 6, 8):
        inp = _inputs(m, m, 8192)
        cache = _swa_cache(16, 64)
        slots = torch.arange(m, dtype=torch.int64, device=dev)
        for mma in ("hmma", "wgmma"):
            for fast in (0, 3, 35, 99, 227):
                it = [0]

                def runp(mma=mma, fast=fast):
                    it[0] = (it[0] + 1) % len(infos)
                    prologue.run_raw(**{k: inp[k] for k in keys}, info=infos[it[0]], swa=(cache, slots, 64), mma=mma, fast=fast)

                ms = bench(runp, warmup=10, iters=iters)
                # CUDA-graph replay of 8 layers back to back (what decode sees: no host gaps)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for info in infos:
                        prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(cache, slots, 64), mma=mma, fast=fast)
                ms_g = bench(graph.replay, warmup=5, iters=max(10, iters // 4)) / len(infos)
                print(fmt_bw(f"prologue m={m} {mma:<5s} fast={fast} (eager)", ms, nbytes=wbytes))
                print(fmt_bw(f"prologue m={m} {mma:<5s} fast={fast} (graph, per layer)", ms_g, nbytes=wbytes))
                del graph


PROF_NAMES = {
    "G1": {1: "pdl", 2: "P1A seen", 3: "act smem", 4: "staged", 5: "mma done", 10: "part st", 6: "partial", 9: "tile fin",
           7: "qr: q seen", 11: "qr: built", 8: "qr: pub", 15: "exit"},
    "G2": {1: "pdl", 9: "P1a t0", 10: "P1a bar", 11: "P1a fence", 2: "P1a pub", 3: "P1b pub", 5: "qr seen", 6: "qr smem",
           7: "mma done", 8: "red", 15: "exit"},
    "H0": {1: "pdl", 2: "mix seen", 3: "sinkhorn", 15: "exit"},
    "H1": {1: "pdl", 4: "kv seen", 5: "kv done", 15: "exit"},
}


def prof_main(iters: int, mma_list: list[str], fast_list: list[int], ms_list: list[int], hot: bool = False) -> None:
    """Phase timeline of one prologue launch inside a CUDA graph of 8 layers (the 5th launch is stamped),
    from a -DDS41H_PROLOGUE_PROF build: per role, median / max over CTAs of each phase stamp relative to the
    earliest CTA entry, plus the SM clock (clock64 / globaltimer over each CTA's lifetime)."""
    require_cuda()
    from ds41h import cuext

    e = cuext.load("ds41h_prologue_prof", "prologue.cu", extra_cuda_cflags=("-DDS41H_PROLOGUE_PROF",))
    assert bool(e.PROFILING)
    e.init()
    prologue.ext = lambda: e  # run_raw / ensure_device use the profiling build
    G1, G2, NS = 112, 128, int(e.PROF_SLOTS)
    # hot: the same layer 8 times (its 21.6 MB stay L2-resident: the sync / latency chain alone)
    infos = [_layer_info(2, rank=0, max_pos=8192)] * 8 if hot else [_layer_info(i, rank=0, max_pos=8192) for i in range(2, 10)]
    keys = ("x_prev", "residual", "post_mix", "res_mix", "pre_mix", "positions")
    buf = torch.zeros(max(int(e.CTAS), int(e.CTAS_V3)) * NS, dtype=torch.int64, device="cuda")
    for m in ms_list:
        inp = _inputs(m, m, 8192)
        cache = _swa_cache(16, 64)
        slots = torch.arange(m, dtype=torch.int64, device="cuda")
        for mma in mma_list:
            for fast in fast_list:
                graph = torch.cuda.CUDAGraph()
                for info in infos:  # warm up
                    prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(cache, slots, 64), mma=mma, fast=fast)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    for i, info in enumerate(infos):
                        e.set_prof(buf if i == 4 else None)
                        prologue.run_raw(**{k: inp[k] for k in keys}, info=info, swa=(cache, slots, 64), mma=mma, fast=fast)
                e.set_prof(None)
                acc = []
                for _ in range(iters):
                    buf.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    acc.append(buf.view(-1, NS).cpu().clone())
                ms_g = bench(graph.replay, warmup=5, iters=50) / len(infos)
                st = torch.stack(acc).double()  # [iters, CTAS, NS]
                t0 = torch.where(st[:, :, 0] > 0, st[:, :, 0], torch.full_like(st[:, :, 0], float("inf"))).min(dim=1, keepdim=True).values
                rel = (st - t0[..., None]) / 1e3  # us
                rel[st == 0] = float("nan")
                clk = (st[:, :, 14] - st[:, :, 13]) / (st[:, :, 15] - st[:, :, 0])  # GHz
                print(f"\n== prologue m={m} {mma} fast={fast}{' HOT' if hot else ''}: {ms_g * 1e3:.2f} us/layer (graph); SM clock "
                      f"{clk.nanmedian().item() * 1e3:.0f} MHz (median over CTAs/iters); stamps in us from the first CTA entry "
                      f"(median over {iters} replays of the per-role median / max over CTAs)")
                # (FAST_CLUSTER: the H jobs run on G1 CTAs 110 / 111; their H stamps share those CTAs' slots)
                h0 = G1 - 2 if fast & 64 else G1 + G2
                roles = {"G1": slice(0, G1), "G2": slice(G1, G1 + G2), "H0": slice(h0, h0 + 1), "H1": slice(h0 + 1, h0 + 2)}
                for role, sl in roles.items():
                    parts = []
                    for slot, name in PROF_NAMES[role].items():
                        v = rel[:, sl, slot]
                        if torch.isnan(v).all():
                            continue
                        med = v.nanmedian(dim=1).values.nanmedian().item()
                        mx = torch.nan_to_num(v, nan=-1e9).max(dim=1).values.median().item()
                        parts.append(f"{name} {med:.1f}/{mx:.1f}")
                    ent = rel[:, sl, 0]
                    print(f"  {role}: entry {ent.nanmedian(dim=1).values.median().item():.1f}/"
                          f"{torch.nan_to_num(ent, nan=-1e9).max(dim=1).values.median().item():.1f} | " + " | ".join(parts))
                del graph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--prof", action="store_true", help="phase timeline from a -DDS41H_PROLOGUE_PROF build")
    ap.add_argument("--mma", default="wgmma,hmma")
    ap.add_argument("--fast", default="3")
    ap.add_argument("--m", default="1,6,8")
    ap.add_argument("--hot", action="store_true", help="--prof: same layer every launch (weights L2-resident)")
    ap.add_argument("--iters", type=int, default=200)
    args, rest = ap.parse_known_args()
    if args.prof:
        prof_main(min(args.iters, 50), args.mma.split(","), [int(v) for v in args.fast.split(",")], [int(v) for v in args.m.split(",")],
                  hot=args.hot)
        return 0
    if args.bench:
        bench_main(args.iters)
        return 0
    return run_module_tests(globals(), rest)


if __name__ == "__main__":
    sys.exit(main())
