"""mhc.cu, the fused mHC sublayer boundary on a rank's own rows: exactness, and the equivalence of its entry points.

One boundary, per row (mhc.cu's header): xn[j] = bf16(post[j] y + sum_i comb[i][j] rs[i]) (the next residual stream);
mixes = (xn.flat @ fn^T) rsqrt(mean(xn.flat^2) + eps); pre / post / comb = sigmoids and a Sinkhorn of the mixes; the
sublayer input = RMSNorm(bf16(sum_c lag[c] xn[c])) * w.

Guards:
  - xn is bit-exact. The kernel computes post[j] * y, then one fmaf per source copy; the inputs here keep every exact
    intermediate sum within float64's 53 bits (magnitudes in [2^-3, 2^3], comb >= 0.05), so float64 sums followed by a
    single float32 rounding reproduce each fmaf exactly.
  - pre / post / comb match a float64 reference of the same formulas within 5e-4 (the mixes run as split-tf32
    tensor-core products); the sublayer input matches an emulation of the kernel's fp32 collapse chain within 1 bf16 ulp
    (the norm's rsqrtf is not correctly rounded), and is exact in at least 99.9 % of the elements.
  - The MoE prescale exponent ptok = 14 - frexp exponent of each output row's max |value| (0 for an all-zero row), exactly.
  - Every entry point the engine combines gives the same bits at any row count and in row chunks: boundary ==
    boundary2 == k1a + k1b_out + k1b_mix == k1a_x + k1a_mix + k1b_out + k1b_mix. The RS-folded variants (boundary_rs,
    k1a_rs, k1a_x_rs, including a row-chunk view of the receive buffer) == comm.cu rs_sum followed by the plain variant.
    boundary_push stores boundary's output rows into slot `rank` of every gather buffer and signals each flag once per row.
Row counts: decode buckets, odd counts, and mixed-step sizes up to the 4,288-row bucket.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

D, HC, NMIX, NP, NCH = 5120, 4, 24, 25, 160
EPS, HC_EPS, ITERS = 1e-6, 1e-6, 20
ROWS = (1, 4, 32, 127, 128, 320, 1024, 4288)


def _inputs(R, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)

    def mag(shape, lo, hi):
        return torch.exp2(torch.rand(shape, generator=g, device=dev) * (hi - lo) + lo)

    def sgn(shape):
        return torch.randint(0, 2, shape, generator=g, device=dev).float() * 2 - 1

    return dict(
        y=(sgn((R, D)) * mag((R, D), -3, 3)).bfloat16(),
        rs=(sgn((R, HC, D)) * mag((R, HC, D), -3, 3)).bfloat16(),
        post=torch.rand(R, HC, generator=g, device=dev) * 1.75 + 0.25,
        comb=torch.rand(R, HC, HC, generator=g, device=dev) * 0.95 + 0.05,
        lag=torch.rand(R, HC, generator=g, device=dev) * 0.9 + 0.1,
        fn=torch.randn(NMIX, HC * D, generator=g, device=dev) / (HC * D) ** 0.5,
        scale=torch.rand(3, generator=g, device=dev) * 0.7 + 0.3,
        base=torch.randn(NMIX, generator=g, device=dev) * 0.5,
        normw=torch.rand(D, generator=g, device=dev) + 0.5,
    )


def _outs(R, dev):
    nan = float("nan")
    return dict(xn=torch.full((R, HC, D), nan, dtype=torch.bfloat16, device=dev),
                pre=torch.full((R, HC), nan, device=dev), post=torch.full((R, HC), nan, device=dev),
                comb=torch.full((R, HC, HC), nan, device=dev), out=torch.full((R, D), nan, dtype=torch.bfloat16, device=dev),
                part=torch.full((NCH * R * NP,), nan, device=dev), ptok=torch.full((R,), -999, dtype=torch.int32, device=dev))


def xn_exact(y, rs, post, comb):
    """The kernel's hc_post, emulated exactly: fp32 post[j] * y, then for c = 0..3 fmaf(comb[c][j], rs[c], a)."""
    a = post[:, :, None] * y.float()[:, None, :]                            # [R, j, D], one fp32 rounding
    for c in range(HC):
        p = comb[:, c, :, None].double() * rs[:, c, None, :].double()       # exact products
        a = (p + a.double()).float()                                         # exact sum, one fp32 rounding = fmaf
    return a.bfloat16()


def ref_rest(xn, lag, fn, scale, base, normw):
    """float64 reference of the mixes (pre / post / comb) and the sublayer input, from the kernel's own xn."""
    X = xn.double().flatten(1)
    mixes = (X @ fn.double().t()) * torch.rsqrt(X.square().mean(1, keepdim=True) + EPS)
    sc, bs = scale.double(), base.double()
    pre = torch.sigmoid(mixes[:, :HC] * sc[0] + bs[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * sc[1] + bs[HC:2 * HC])
    c = torch.softmax((mixes[:, 2 * HC:] * sc[2] + bs[2 * HC:]).view(-1, HC, HC), -1) + HC_EPS
    c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        c = c / (c.sum(-1, keepdim=True) + HC_EPS)
        c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    # collapse: the kernel's fmaf chain over the 4 copies (a = 0; a = fmaf(lag[c], xn[c], a)), each step an exact
    # float64 sum and one fp32 rounding, then one bf16 rounding (a cancelled sum keeps the chain's rounding error);
    # RMSNorm with the kernel's two fp32 products, from a float64 rsqrt (the kernel's rsqrtf differs by an fp32 ulp)
    a = torch.zeros(xn.shape[0], xn.shape[2], dtype=torch.float32, device=xn.device)
    for k in range(HC):
        a = (lag[:, k, None].double() * xn[:, k].double() + a.double()).float()
    v = a.bfloat16().float()
    rr = torch.rsqrt(v.double().square().mean(1, keepdim=True) + EPS).float()
    out = ((v * rr) * normw).bfloat16()
    return pre, post, c, out


def ptok_ref(out):
    m = out.float().abs().amax(1)
    _, ex = torch.frexp(m)
    return torch.where(m > 0, 14 - ex, torch.zeros_like(ex)).to(torch.int32)


def bf16_ulps(a, b):
    """|a - b| in bf16 ulps of |b| (elementwise)."""
    a, b = a.float(), b.float()
    u = torch.exp2(torch.floor(torch.log2(b.abs().clamp_min(2.0 ** -100))) - 7)
    return (a - b).abs() / u


def run_boundary2(e, x, fnp, o):
    e.boundary2(x["y"], x["rs"], x["post"], x["comb"], x["lag"], fnp, x["scale"], x["base"], x["normw"], EPS, HC_EPS, ITERS,
                o["xn"], o["pre"], o["post"], o["comb"], o["out"], o["part"], o["ptok"])


def run_split(e, x, fnp, o, chunk, kx):
    """The engine's split boundary in row chunks: k1a (or k1a_x + k1a_mix), k1b_out (+ ptok), k1b_mix."""
    R = x["y"].shape[0]
    for c0 in range(0, R, chunk):
        c1 = min(R, c0 + chunk)
        part = o["part"][NCH * NP * c0:NCH * NP * c1]
        if kx:
            e.k1a_x(x["y"][c0:c1], x["rs"][c0:c1], x["post"][c0:c1], x["comb"][c0:c1], o["xn"][c0:c1])
            e.k1a_mix(o["xn"][c0:c1], c1 - c0, fnp, part)
        else:
            e.k1a(x["y"][c0:c1], x["rs"][c0:c1], x["post"][c0:c1], x["comb"][c0:c1], fnp, o["xn"][c0:c1], part)
        e.k1b_out(o["xn"][c0:c1], x["lag"][c0:c1], x["normw"], EPS, o["out"][c0:c1], o["ptok"][c0:c1])
        e.k1b_mix(part, c1 - c0, x["scale"], x["base"], EPS, HC_EPS, ITERS, o["pre"][c0:c1], o["post"][c0:c1], o["comb"][c0:c1])


@pytest.mark.parametrize("R", ROWS)
def test_xn_exact_and_reference(mhc, dev, sync, same, R):
    """xn bit-exact; mixes and sublayer input within tolerance of float64; ptok exact."""
    x = _inputs(R, dev, 11 + R)
    fnp = mhc.prep_fn(x["fn"])
    o = _outs(R, dev)
    run_boundary2(mhc, x, fnp, o)
    sync([dev])
    for c0 in range(0, R, 1024):                                     # the float64 emulation in row chunks (memory)
        c1 = min(R, c0 + 1024)
        ref = xn_exact(x["y"][c0:c1], x["rs"][c0:c1], x["post"][c0:c1], x["comb"][c0:c1])
        assert same(o["xn"][c0:c1], ref), f"xn rows [{c0}, {c1}) not bit-exact"
    pre, post, comb, out = ref_rest(o["xn"], x["lag"], x["fn"], x["scale"], x["base"], x["normw"])
    assert float((o["pre"].double() - pre).abs().max()) < 5e-4
    assert float((o["post"].double() - post).abs().max()) < 5e-4
    assert float((o["comb"].double() - comb).abs().max()) < 5e-4
    u = bf16_ulps(o["out"], out)
    exact = float((u == 0).float().mean())
    assert float(u.max()) <= 1 and exact >= 0.999, f"out: max {float(u.max())} ulps, {exact:.6f} exact"
    assert same(o["ptok"], ptok_ref(o["out"]))


@pytest.mark.parametrize("R", ROWS)
def test_entry_points_agree(mhc, dev, sync, same, R):
    """boundary, boundary2, and the split / chunked / K1a-split compositions: identical bits for every output."""
    x = _inputs(R, dev, 23 + R)
    fnp = mhc.prep_fn(x["fn"])
    ref = _outs(R, dev)
    run_boundary2(mhc, x, fnp, ref)
    b = _outs(R, dev)
    mhc.boundary(x["y"], x["rs"], x["post"], x["comb"], x["lag"], fnp, x["scale"], x["base"], x["normw"], EPS, HC_EPS, ITERS,
                 b["xn"], b["pre"], b["post"], b["comb"], b["out"], b["part"])
    arms = {"boundary": b}
    for name, chunk, kx in (("split", R, False), ("split k1a_x", R, True), ("split 128-row chunks", 128, False),
                            ("split 64-row chunks k1a_x", 64, True)):
        o = _outs(R, dev)
        run_split(mhc, x, fnp, o, chunk, kx)
        arms[name] = o
    sync([dev])
    for name, o in arms.items():
        for k in ("xn", "pre", "post", "comb", "out") + (("ptok",) if name != "boundary" else ()):
            assert same(o[k], ref[k]), f"{name}: {k} differs from boundary2"
    for name in ("boundary", "split", "split k1a_x"):
        assert same(arms[name]["part"], ref["part"]), f"{name}: mix partials differ"


@pytest.mark.parametrize("R", (32, 320, 1024))
def test_rs_fold(mhc, comm, dev, sync, same, R):
    """boundary_rs / k1a_rs / k1a_x_rs (the RS sum folded into K1a) == comm.cu rs_sum, then the plain entry point."""
    x = _inputs(R, dev, 37 + R)
    fnp = mhc.prep_fn(x["fn"])
    g = torch.Generator(device=dev).manual_seed(41 + R)
    recv = (torch.randn(4, R, D, generator=g, device=dev) * 0.7).bfloat16()
    SIG = 64
    flag = torch.tensor([SIG, 0, 0, 0], dtype=torch.int32, device=dev)     # the round already complete: no wait
    epoch = torch.ones(1, dtype=torch.int32, device=dev)
    y = torch.empty(R, D, dtype=torch.bfloat16, device=dev)
    comm.rs_sum(recv, flag, epoch, SIG, y, 132)
    sync([dev])
    a = recv[0].float()
    for s in range(1, 4):
        a = a + recv[s].float()
    assert same(y, a.bfloat16()), "rs_sum differs from the fixed-order sum"
    xy = dict(x, y=y)
    ref = _outs(R, dev)
    run_boundary2(mhc, xy, fnp, ref)
    br = _outs(R, dev)
    mhc.boundary_rs(recv, flag, epoch, SIG, x["rs"], x["post"], x["comb"], x["lag"], fnp, x["scale"], x["base"], x["normw"],
                    EPS, HC_EPS, ITERS, br["xn"], br["pre"], br["post"], br["comb"], br["out"], br["part"], br["ptok"])
    k1 = _outs(R, dev)
    mhc.k1a(y, x["rs"], x["post"], x["comb"], fnp, k1["xn"], k1["part"])
    k1r = _outs(R, dev)
    mhc.k1a_rs(recv, flag, epoch, SIG, x["rs"], x["post"], x["comb"], fnp, k1r["xn"], k1r["part"])
    kx = _outs(R, dev)
    mhc.k1a_x(y, x["rs"], x["post"], x["comb"], kx["xn"])
    kxr = _outs(R, dev)
    mhc.k1a_x_rs(recv, flag, epoch, SIG, x["rs"], x["post"], x["comb"], kxr["xn"])
    c0, c1 = R // 4, R // 4 + max(1, R // 2)                                  # a row-chunk view of the receive buffer
    ch = _outs(c1 - c0, dev)
    xc = {k: (v[c0:c1] if k in ("rs", "post", "comb", "lag") else v) for k, v in x.items()}
    mhc.boundary_rs(recv[:, c0:c1], flag, epoch, SIG, xc["rs"], xc["post"], xc["comb"], xc["lag"], fnp, x["scale"], x["base"],
                    x["normw"], EPS, HC_EPS, ITERS, ch["xn"], ch["pre"], ch["post"], ch["comb"], ch["out"], ch["part"], ch["ptok"])
    sync([dev])
    for k in ("xn", "pre", "post", "comb", "out", "ptok", "part"):
        assert same(br[k], ref[k]), f"boundary_rs: {k} differs"
    for k in ("xn", "pre", "post", "comb", "out", "ptok"):
        assert same(ch[k], ref[k][c0:c1]), f"boundary_rs on a row-chunk view: {k} differs"
    assert same(k1r["xn"], k1["xn"]) and same(k1r["part"], k1["part"]), "k1a_rs differs from k1a"
    assert same(kxr["xn"], kx["xn"]) and same(kx["xn"], ref["xn"]), "k1a_x_rs / k1a_x differ"


@pytest.mark.parametrize("R", (4, 128))
def test_boundary_push(mhc, dev, sync, same, R):
    """boundary_push: boundary's output rows land in slot `rank` of all 4 gather buffers (other slots untouched), each
    flag gets one signal per row, and xn / pre / post / comb are boundary's."""
    x = _inputs(R, dev, 53 + R)
    fnp = mhc.prep_fn(x["fn"])
    ref = _outs(R, dev)
    run_boundary2(mhc, x, fnp, ref)
    rank = 2
    bufs = [torch.full((4 * R, D), float("nan"), dtype=torch.bfloat16, device=dev) for _ in range(4)]
    flags = [torch.zeros(4, dtype=torch.int32, device=dev) for _ in range(4)]
    o = _outs(R, dev)
    mhc.boundary_push(x["y"], x["rs"], x["post"], x["comb"], x["lag"], fnp, x["scale"], x["base"], x["normw"], EPS, HC_EPS,
                      ITERS, o["xn"], o["pre"], o["post"], o["comb"], o["part"], rank, bufs, flags)
    sync([dev])
    for k in ("xn", "pre", "post", "comb"):
        assert same(o[k], ref[k])
    for d in range(4):
        assert same(bufs[d][rank * R:(rank + 1) * R], ref["out"]), f"gather buffer {d}: slot {rank} differs"
        others = torch.cat([bufs[d][:rank * R], bufs[d][(rank + 1) * R:]])
        assert bool(others.isnan().all()), f"gather buffer {d}: other slots were written"
        assert int(flags[d][0]) == R
