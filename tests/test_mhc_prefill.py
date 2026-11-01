"""Tests + micro-benchmark for the fused mHC prefill boundary (ds41h/kernels/mhc_prefill.cu, ds41h/mhc_prefill.py).

    CUDA_VISIBLE_DEVICES="" python tests/test_mhc_prefill.py cpu     # CPU-only subset
    python tests/test_mhc_prefill.py                                 # all tests (GPU tests need an sm90 GPU)
    python tests/test_mhc_prefill.py --bench [--layer 10] [--tokens 8192]
        stock (mhc_post_tilelang + DeepGEMM tf32_hc_prenorm_gemm + mhc_pre_big_fuse_with_norm) vs fused, per kernel

References: (a) ``mhc_prefill.boundary_ref`` (torch, stock rounding points, fp64 GEMV), (b) the stock vLLM 0.32.0
path itself (``mhc_fused_post_pre_delayed_tilelang`` / ``mhc_pre_delayed_tilelang`` with > 32 tokens = the
unfused TileLang + DeepGEMM kernels), with real hc weights of layer DS41H_TEST_LAYER (default 10) when the
checkpoint is available (random weights of the same scale otherwise).
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import compare, fmt_metrics, has_cuda, run_module_tests, skip  # noqa: E402

from ds41h import mhc_prefill as M  # noqa: E402

LAYER = int(os.environ.get("DS41H_TEST_LAYER", "10"))
HC, HID, MIX = 4, 5120, 24
EPS = dict(rms_eps=1e-20, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20, norm_eps=1e-20)
torch.backends.cuda.matmul.allow_tf32 = False


# ===================================================================================== helpers
def require_gpu():
    if not has_cuda() or torch.cuda.get_device_capability() != (9, 0):
        skip("needs an sm90 GPU")


def hc_weights(which: str = "attn", layer: int = LAYER, device: str = "cpu") -> dict[str, torch.Tensor]:
    """fn [24, 20480] f32, scale [3], base [24], norm weight bf16 [5120] of one boundary (real if available)."""
    try:
        from common import reader

        rd = reader()
        p = f"layers.{layer}."
        norm = "attn_norm.weight" if which == "attn" else "ffn_norm.weight"
        w = {"fn": rd.get(p + f"hc_{which}_fn").float(), "scale": rd.get(p + f"hc_{which}_scale").float(),
             "base": rd.get(p + f"hc_{which}_base").float(), "norm": rd.get(p + norm).to(torch.bfloat16)}
    except Exception:  # noqa: BLE001 - no checkpoint: random weights of a similar scale
        g = torch.Generator().manual_seed(layer)
        w = {"fn": torch.randn(MIX, HC * HID, generator=g) * 2e-3, "scale": torch.tensor([0.5, 0.5, 2.0]),
             "base": torch.randn(MIX, generator=g) * 0.5, "norm": (1 + 0.1 * torch.randn(HID, generator=g)).to(torch.bfloat16)}
    return {k: v.to(device).contiguous() for k, v in w.items()}


def inputs(T: int, seed: int = 0, device: str = "cuda", scale: float = 1.0):
    """x [T, 5120] bf16, residual [T, 4, 5120] bf16, post [T, 4, 1], comb [T, 4, 4], pre_mix [T, 4] (fp32)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(T, 1, HID, generator=g) * 0.6
    res = base + 0.25 * torch.randn(T, HC, HID, generator=g)
    res = res * (1 + 6 * (torch.rand(HID, generator=g) < 0.003))  # a few large channels
    x = torch.randn(T, HID, generator=g) * 0.3
    post = 2 * torch.sigmoid(torch.randn(T, HC, 1, generator=g))
    logits = torch.randn(T, HC, HC, generator=g)
    comb = torch.softmax(logits, -1) + 1e-6
    for _ in range(20):
        comb = comb / comb.sum(-2, keepdim=True)
        comb = comb / comb.sum(-1, keepdim=True)
    pre = torch.sigmoid(torch.randn(T, HC, generator=g)) + 1e-6
    bf = lambda t: (t * scale).to(torch.bfloat16).to(device).contiguous()  # noqa: E731
    return bf(x), bf(res), post.to(device), comb.contiguous().to(device), pre.to(device)


def stock_available() -> bool:
    try:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_delayed_tilelang  # noqa: F401
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        return bool(is_deep_gemm_supported())
    except Exception:  # noqa: BLE001
        return False


def stock_post_pre(x, res, post, comb, pre, w, aux=False):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_delayed_tilelang

    return mhc_fused_post_pre_delayed_tilelang(x, res, post, comb, w["fn"], w["scale"], w["base"], EPS["rms_eps"], EPS["hc_pre_eps"],
                                               EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"],
                                               pre_mix=pre, norm_weight=w["norm"], norm_eps=EPS["norm_eps"], capture_aux=aux)


IMPLS = tuple(int(v) for v in os.environ.get("DS41H_TEST_MHC_IMPLS", "4,3,2,1").split(","))


def fused(x, res, post, comb, pre, w, aux=False, impl=None):
    return M.boundary(x, res, post, comb, pre, w["fn"], w["scale"], w["base"], w["norm"], EPS["rms_eps"], EPS["hc_pre_eps"],
                      EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"], EPS["norm_eps"], aux, impl)


def reference(x, res, post, comb, pre, w, aux=False, gemm_dtype=None):
    return M.boundary_ref(x, res, post, comb, pre, w["fn"], w["scale"], w["base"], w["norm"], EPS["rms_eps"], EPS["hc_pre_eps"],
                          EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"], EPS["norm_eps"], aux,
                          gemm_dtype=gemm_dtype)


NAMES = ("residual", "post", "comb", "layer_input", "pre", "aux")


def report(tag: str, test, refv, names=NAMES) -> dict:
    out = {}
    for n, a, b in zip(names, test, refv, strict=False):
        if a is None or b is None or a.numel() == 0:
            continue
        m = compare(a.float(), b.float())
        out[n] = m
        print(f"  {tag:>14s} {n:<12s} {fmt_metrics(m)}")
    return out


def residual_check(out: torch.Tensor, x, res, post, comb, tag: str = "") -> float:
    """Post-mapped residual vs the exact (fp64) post map: every element within 1 bf16 ulp of the exact value plus
    the fp32 accumulation error of 5 terms (2^-21 of the absolute sum |post x| + sum |comb res|, which matters only
    under cancellation). Returns the fraction of elements that differ from round-to-nearest of the exact value."""
    T = res.shape[0]
    p = post.reshape(T, HC, 1).double()
    c = comb.reshape(T, HC, HC).double()
    xd, rd = x.double().unsqueeze(1), res.double()
    exact = p * xd + torch.einsum("tio,tih->toh", c, rd)
    absum = (p * xd).abs() + torch.einsum("tio,tih->toh", c.abs(), rd.abs())
    o = out.double()
    mag = torch.maximum(exact.abs(), o.abs()).clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
    err = (o - exact).abs()
    bad = err > ulp + 2.0 ** -21 * absum
    nbad = int(bad.sum())
    frac = float((out != exact.to(torch.bfloat16)).float().mean())
    print(f"  {tag} residual: {frac:.2e} differ from rn(exact), max err/ulp {float((err / ulp).max()):.2f}, out of bound {nbad}")
    assert nbad == 0, (tag, nbad)
    return frac


def bf16_mismatch(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(fraction of differing bf16 elements, max difference in bf16 ulps of the reference)."""
    d = a.view(torch.int16).int() - b.view(torch.int16).int()
    frac = float((d != 0).float().mean())
    return frac, float(d.abs().max())


# ===================================================================================== CPU tests
def test_cpu_pack_roundtrip_and_layout():
    g = torch.Generator().manual_seed(0)
    fn = torch.randn(MIX, HC * HID, generator=g) * 3e-3
    for impl in (1, 2):
        p = M.pack_fn(fn, impl)
        assert p.dtype == torch.bfloat16 and p.numel() * 2 == 160 * 12288
        back = M.unpack_fn(p, impl)
        rel = float((back - fn).norm() / fn.norm())
        assert rel < 1e-5, (impl, rel)  # hi + lo keeps ~16 mantissa bits
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    flat = M.pack_fn(fn, 1).view(-1, 8)  # 16-byte lane groups
    for gs, o, j, hl, gg, t in ((0, 0, 0, 0, 0, 0), (7, 2, 1, 1, 5, 3), (159, 3, 2, 0, 7, 1)):
        blk = ((gs * HC + o) * 3 + j) * 2 + hl
        lane_vals = flat[blk * 32 + gg * 4 + t].float()
        n, k0 = 8 * j + gg, o * HID + 32 * gs + 8 * t
        exp = (hi if hl == 0 else lo)[n, k0 : k0 + 8].float()
        assert torch.equal(lane_vals, exp), (gs, o, j, hl, gg, t)
    w48 = torch.cat([hi, lo])
    flat = M.pack_fn(fn, 2).view(-1, 8)
    for gs, o, mt, kb, gg, t in ((0, 0, 0, 0, 0, 0), (7, 2, 1, 1, 5, 3), (159, 3, 2, 0, 7, 1), (80, 1, 2, 1, 3, 2)):
        blk = ((gs * HC + o) * 3 + mt) * 2 + kb
        lane_vals = flat[blk * 32 + gg * 4 + t].float()
        for wd in range(4):
            pp, rh = wd // 2, wd % 2
            row, col = 16 * mt + 8 * rh + gg, o * HID + 32 * gs + 8 * t + 4 * kb + 2 * pp
            assert torch.equal(lane_vals[2 * wd : 2 * wd + 2], w48[row, col : col + 2].float()), (gs, o, mt, kb, gg, t, wd)


def test_cpu_reference_matches_ref_layer():
    """boundary_ref == ds41h.ref.layer hc_post + hc_mixes + hc_collapse + rmsnorm (same math)."""
    from ds41h import ref

    w = hc_weights("attn", device="cpu")
    x, res, post, comb, pre = inputs(9, device="cpu")
    out = reference(x, res, post, comb, pre, w)
    r = ref.hc_post(x, res, post, comb)
    assert torch.equal(out[0], r.to(torch.bfloat16)) or bf16_mismatch(out[0], r.to(torch.bfloat16))[0] < 2e-3
    pre_n, post_n, comb_n, _ = ref.hc_mixes(out[0], w["fn"], w["scale"], w["base"], norm_eps=EPS["rms_eps"])
    assert torch.allclose(out[1].squeeze(-1), post_n, atol=1e-5)
    assert torch.allclose(out[2], comb_n, atol=1e-5)
    assert torch.allclose(out[4], pre_n, atol=1e-6)
    y = ref.rmsnorm(ref.hc_collapse(out[0], pre), w["norm"], EPS["norm_eps"])
    m = compare(out[3].float(), y.float())
    assert m["rel_rmse"] < 5e-3, fmt_metrics(m)


def test_cpu_patch_wrappers_fall_back_on_cpu_tensors():
    calls = []
    stock = lambda *a, **k: calls.append(("post_pre", k.get("stream"))) or "stock"  # noqa: E731
    stock_pre = lambda *a, **k: calls.append(("pre", k.get("x") is not None)) or "stock_pre"  # noqa: E731
    f = M.make_shifted_post_pre(stock)
    fp = M.make_pre_delayed(stock_pre)
    w = hc_weights("attn", device="cpu")
    x, res, post, comb, pre = inputs(80, device="cpu")
    assert f(x, res, post, comb, w["fn"], w["scale"], w["base"], 1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix=pre, norm_weight=w["norm"]) == "stock"
    assert fp(res, w["fn"], w["scale"], w["base"], 1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix=pre, norm_weight=w["norm"]) == "stock_pre"
    assert f.__wrapped__ is stock and fp.__wrapped__ is stock_pre
    assert calls == [("post_pre", None), ("pre", False)]


# ===================================================================================== GPU tests
def test_fused_vs_reference():
    """POST mode (incl. partial tiles, aux, one-hot pre) vs the fp64-GEMV torch reference."""
    require_gpu()
    w = hc_weights("attn", device="cuda")
    for impl, T, aux, onehot in [(i, *c) for i in IMPLS for c in ((64, False, False), (200, True, False), (1000, False, True),
                                                                   (4096, False, False), (77, True, True))]:
        x, res, post, comb, pre = inputs(T, seed=T)
        pre_arg = None if onehot else pre
        out = fused(x, res, post, comb, pre_arg, w, aux, impl)
        torch.cuda.synchronize()
        refv = reference(x, res, post, comb, pre_arg, w, aux)
        frac = residual_check(out[0], x, res, post, comb, f"impl {impl} T={T} aux={aux} onehot={onehot}")
        assert frac < 1e-3, frac
        m = report("fused vs ref", out[1:], refv[1:], NAMES[1:])
        # stock (TF32 GEMV) is 2-5x further from the fp64 reference than this on the same inputs (test_fused_vs_stock_vllm)
        assert m["post"]["max_abs"] < 2e-4 and m["pre"]["max_abs"] < 3e-4 and m["comb"]["max_abs"] < 1e-4, m
        assert m["layer_input"]["rel_rmse"] < 3e-3 and m["layer_input"]["cos"] > 0.99999, m["layer_input"]
        if aux:
            assert m["aux"]["rel_rmse"] < 3e-3, m["aux"]


def test_pre_mode_vs_reference():
    """PRE mode (Engram-layer / non-broadcast mhc_pre_delayed_tilelang): residual as is."""
    require_gpu()
    w = hc_weights("ffn", device="cuda")
    for impl, T in [(i, T) for i in IMPLS for T in (64, 333)]:
        _, res, _, _, pre = inputs(T, seed=7 + T)
        out = fused(None, res, None, None, pre, w, impl=impl)
        assert out[0].data_ptr() == res.data_ptr()
        refv = reference(None, res, None, None, pre, w)
        m = report(f"pre T={T}", out[1:5], refv[1:5], NAMES[1:5])
        assert m["post"]["max_abs"] < 2e-4 and m["comb"]["max_abs"] < 1e-4 and m["layer_input"]["rel_rmse"] < 3e-3, m


def test_fused_vs_stock_vllm():
    """Against the stock vLLM 0.32.0 unfused path (TileLang post + DeepGEMM TF32 GEMM + TileLang pre-norm)."""
    require_gpu()
    if not stock_available():
        skip("stock TileLang/DeepGEMM mHC kernels unavailable")
    w = hc_weights("attn", device="cuda")
    for impl, T in [(i, T) for i in IMPLS for T in (64, 1000, 8192)]:
        print(f"-- impl {impl}")
        x, res, post, comb, pre = inputs(T, seed=3 * T)
        st = stock_post_pre(x, res, post, comb, pre, w)
        out = fused(x, res, post, comb, pre, w, impl=impl)
        refv = reference(x, res, post, comb, pre, w)
        torch.cuda.synchronize()
        frac, ulps = bf16_mismatch(out[0], st[0])
        print(f"T={T}: residual vs stock bf16 mismatch {frac:.2e} (max {ulps:.0f} ulp)")
        residual_check(out[0], x, res, post, comb, f"fused T={T}")
        residual_check(st[0], x, res, post, comb, f"stock T={T}")
        assert frac < 1e-3
        ms = report("fused vs stock", out[1:5], st[1:5], NAMES[1:5])
        mr = report("stock vs ref", st[1:5], refv[1:5], NAMES[1:5])
        mf = report("fused vs ref", out[1:5], refv[1:5], NAMES[1:5])
        for n in ("post", "comb", "pre"):
            assert mf[n]["max_abs"] <= max(2 * mr[n]["max_abs"], 1e-5), (n, mf[n], mr[n])
        assert ms["layer_input"]["rel_rmse"] < 5e-3 and ms["layer_input"]["cos"] > 0.9999, ms["layer_input"]
        lf, ls = bf16_mismatch(out[3], st[3])
        print(f"T={T}: layer_input vs stock bf16 mismatch {lf:.2e} (max {ls:.0f} ulp)")


def test_patch_on_stand_in_module():
    """install(): module-global wrappers dispatch eligible calls to the fused kernel, the rest to stock."""
    require_gpu()
    calls = []
    mod = types.ModuleType("ds41h_test_fake_model")

    def stock_spp(*a, **k):
        calls.append("spp")
        return "stock"

    def stock_pre(*a, **k):
        calls.append("pre")
        return "stock_pre"

    class DeepseekV4Model:
        def __init__(self, layers):
            self.layers = layers

        def finalize_mhc_broadcast_weights(self):
            calls.append("finalize")

    mod.mhc_shifted_post_pre, mod.mhc_pre_delayed_tilelang, mod.DeepseekV4Model = stock_spp, stock_pre, DeepseekV4Model
    M.install(mod)
    M.install(mod)  # idempotent
    w = hc_weights("attn", device="cuda")
    layer = types.SimpleNamespace(hc_attn_fn=w["fn"].clone(), hc_ffn_fn=w["fn"].clone())
    mod.DeepseekV4Model([layer]).finalize_mhc_broadcast_weights()
    attr = f"{M.PACKED_ATTR}{M.packing_of(M.IMPL)}"
    assert getattr(layer.hc_attn_fn, attr, None) is not None and getattr(layer.hc_ffn_fn, attr, None) is not None
    x, res, post, comb, pre = inputs(M.MIN_T + 5, seed=1)
    hc = (EPS["rms_eps"], EPS["hc_pre_eps"], EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"])
    out = mod.mhc_shifted_post_pre(x, res, post, comb, layer.hc_attn_fn, w["scale"], w["base"], *hc, pre_mix=pre, norm_weight=w["norm"],
                                   norm_eps=EPS["norm_eps"], stream=None, reduce_results=False)
    assert isinstance(out, tuple) and len(out) == 6
    small = inputs(M.MIN_T - 1, seed=2)
    assert mod.mhc_shifted_post_pre(small[0], small[1], small[2], small[3], layer.hc_attn_fn, w["scale"], w["base"], *hc,
                                    pre_mix=small[4], norm_weight=w["norm"], norm_eps=EPS["norm_eps"]) == "stock"
    assert mod.mhc_shifted_post_pre(x, res, post, comb, layer.hc_attn_fn, w["scale"], w["base"], *hc, pre_mix=pre, norm_weight=w["norm"],
                                    norm_eps=EPS["norm_eps"], reduce_results=True) == "stock"
    o2 = mod.mhc_pre_delayed_tilelang(res, layer.hc_attn_fn, w["scale"], w["base"], *hc, pre_mix=pre, norm_weight=w["norm"],
                                      norm_eps=EPS["norm_eps"])
    assert isinstance(o2, tuple) and len(o2) == 4
    fn_b = layer.hc_attn_fn.view(MIX, HC, HID).sum(1)
    assert mod.mhc_pre_delayed_tilelang(res, fn_b, w["scale"], w["base"], *hc, x=res[:, 0], norm_weight=w["norm"]) == "stock_pre"
    assert calls == ["finalize", "spp", "spp", "pre"], calls


def test_stress_scales():
    """Large / tiny residual magnitudes and T not a multiple of the tile."""
    require_gpu()
    w = hc_weights("attn", device="cuda")
    for impl, scale in [(i, sc) for i in IMPLS for sc in (1e-3, 30.0)]:
        x, res, post, comb, pre = inputs(77, seed=11, scale=scale)
        out = fused(x, res, post, comb, pre, w, impl=impl)
        refv = reference(x, res, post, comb, pre, w)
        frac = residual_check(out[0], x, res, post, comb, f"scale={scale}")
        assert frac < 1e-3, (scale, frac)
        m = report(f"scale={scale}", out[1:5], refv[1:5], NAMES[1:5])
        assert m["comb"]["max_abs"] < 1e-4 and m["layer_input"]["rel_rmse"] < 3e-3, (scale, m)


def test_impls_agree_and_deterministic():
    """impl 2 / 3 are bitwise run-to-run deterministic and agree with impl 1 to rounding (same hi/lo fn)."""
    require_gpu()
    w = hc_weights("ffn", device="cuda")
    x, res, post, comb, pre = inputs(1500, seed=21)
    c = fused(x, res, post, comb, pre, w, True, 1)
    for impl in (2, 3, 4):
        a = fused(x, res, post, comb, pre, w, True, impl)
        b = fused(x, res, post, comb, pre, w, True, impl)
        for u, v in zip(a, b, strict=True):
            assert torch.equal(u, v), impl
        frac, _ = bf16_mismatch(a[0], c[0])
        assert frac < 1e-4, (impl, frac)
        m = report(f"impl{impl} vs impl1", a[1:], c[1:], NAMES[1:])
        assert m["comb"]["max_abs"] < 2e-5 and m["layer_input"]["rel_rmse"] < 1e-3, (impl, m)


def test_lossless_error_not_above_stock():
    """LOSSLESS gate: per output, the fused boundary's error vs the fp64-GEMV reference (stock rounding points) is not
    larger than stock vLLM's own error (post / comb / pre: max abs; residual / layer_input / aux: rel RMSE and the
    fraction of bf16 elements that differ from the reference), for every impl, synthetic and real-scale inputs."""
    require_gpu()
    if not stock_available():
        skip("stock TileLang/DeepGEMM mHC kernels unavailable")
    w = hc_weights("attn", device="cuda")
    for T, scale in ((1000, 1.0), (8192, 1.0), (2048, 30.0)):
        x, res, post, comb, pre = inputs(T, seed=7 * T, scale=scale)
        st = stock_post_pre(x, res, post, comb, pre, w, aux=True)
        refv = reference(x, res, post, comb, pre, w, aux=True)
        mr = report("stock vs ref", st[:6], refv[:6])
        for impl in IMPLS:
            out = fused(x, res, post, comb, pre, w, aux=True, impl=impl)
            mf = report(f"impl{impl} vs ref", out[:6], refv[:6])
            for n in ("post", "comb", "pre"):
                assert mf[n]["max_abs"] <= mr[n]["max_abs"] * 1.0 + 1e-7, (T, impl, n, mf[n], mr[n])
            for i, n in ((0, "residual"), (3, "layer_input"), (5, "aux")):
                assert mf[n]["rel_rmse"] <= mr[n]["rel_rmse"] * 1.05 + 1e-6, (T, impl, n, mf[n], mr[n])
                ff, _ = bf16_mismatch(out[i], refv[i].to(torch.bfloat16))
                fs, _ = bf16_mismatch(st[i], refv[i].to(torch.bfloat16))
                print(f"  T={T} x{scale} impl{impl} {n}: differs from ref in {ff:.2e} (stock {fs:.2e})")
                assert ff <= fs * 1.1 + 2e-5, (T, impl, n, ff, fs)


def test_real_stream_vs_stock():
    """Residual stream built from real embedding rows (tests/common.realistic_stream) through a real layer's attn mHC."""
    require_gpu()
    if not stock_available():
        skip("stock TileLang/DeepGEMM mHC kernels unavailable")
    from common import realistic_stream

    w = hc_weights("attn", device="cuda")
    T = 2048
    try:
        res = realistic_stream(T, seed=3).to(torch.bfloat16).cuda().contiguous()
    except Exception as e:  # noqa: BLE001
        skip(f"realistic_stream unavailable: {e}")
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(T, HID, generator=g) * float(res.float().std())).to(torch.bfloat16).cuda()
    _, _, post, comb, pre = inputs(T, seed=4)
    st = stock_post_pre(x, res, post, comb, pre, w)
    out = fused(x, res, post, comb, pre, w)
    refv = reference(x, res, post, comb, pre, w)
    residual_check(out[0], x, res, post, comb, "real stream")
    mf = report("fused vs ref", out[1:5], refv[1:5], NAMES[1:5])
    mr = report("stock vs ref", st[1:5], refv[1:5], NAMES[1:5])
    for n in ("post", "comb", "pre"):
        assert mf[n]["max_abs"] <= max(2 * mr[n]["max_abs"], 1e-5), (n, mf[n], mr[n])
    lf, _ = bf16_mismatch(out[3], st[3])
    print(f"  layer_input vs stock: {lf:.2e} of elements differ")
    assert lf < 1e-2


# ===================================================================================== benchmarks
def bench(layer: int, tokens: list[int]) -> None:
    """Stock (3 kernels) vs fused, sustained at the 200 W cap (tests/test_prefill.py time_sustained)."""
    from test_prefill import time_sustained
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    from vllm.model_executor.kernels.mhc.tilelang_kernels import _MHC_POST_TILELANG_KERNEL
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL, compute_mhc_pre_num_splits
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    w = hc_weights("attn", layer, device="cuda")
    M.packed(w["fn"], 1), M.packed(w["fn"], 2)
    print(f"mHC boundary, layer {layer} hc_attn weights (sustained us per call; GB/s of the minimal HBM traffic: "
          f"POST 100 KB/token, PRE 50 KB/token)")
    for T in tokens:
        x, res, post, comb, pre = inputs(T, seed=5)
        n = T
        res_cur = torch.empty_like(res)
        splits = compute_mhc_pre_num_splits(HC * HID, n)
        mixes = torch.empty(splits, n, MIX, device="cuda")
        sqr = torch.empty(splits, n, device="cuda")
        post_o, comb_o, pre_o = (torch.empty(n, HC, device="cuda"), torch.empty(n, HC * HC, device="cuda"), torch.empty(n, HC, device="cuda"))
        li = torch.empty(n, HID, dtype=torch.bfloat16, device="cuda")

        def k_post():
            _MHC_POST_TILELANG_KERNEL(comb, res, post.view(n, HC), x, res_cur, HC, HID)

        def k_gemm():
            tf32_hc_prenorm_gemm(res_cur.view(n, HC * HID), w["fn"], mixes, sqr, splits)

        def k_pre():
            MHC_PRE_NORM_KERNEL(mixes, sqr, w["scale"], w["base"], res_cur, post_o, comb_o, li, w["norm"], pre, pre_o, li,
                                hidden_size=HID, rms_eps=EPS["rms_eps"], hc_pre_eps=EPS["hc_pre_eps"], hc_sinkhorn_eps=EPS["hc_sinkhorn_eps"],
                                hc_post_mult_value=EPS["hc_post_mult_value"], sinkhorn_repeat=EPS["sinkhorn_repeat"], norm_eps=EPS["norm_eps"],
                                hc_mult=HC, use_pre_mix_in=True, save_pre_mix=True, rms_numel=HC * HID)

        (t1, _), (t2, _), (t3, _) = time_sustained(k_post, 0.3), time_sustained(k_gemm, 0.3), time_sustained(k_pre, 0.3)
        t_stock, c_stock = time_sustained(lambda: stock_post_pre(x, res, post, comb, pre, w))
        t_f1, c_f1 = time_sustained(lambda: fused(x, res, post, comb, pre, w, impl=1))
        t_f2, c_f2 = time_sustained(lambda: fused(x, res, post, comb, pre, w, impl=2))
        t_f3, c_f3 = time_sustained(lambda: fused(x, res, post, comb, pre, w, impl=3))
        t_f, c_f = time_sustained(lambda: fused(x, res, post, comb, pre, w, impl=4))
        print(f"  T={T:6d}: fused impl 1 {t_f1:7.1f} us ({c_f1}) | impl 2 {t_f2:7.1f} us ({c_f2}) | impl 3 {t_f3:7.1f} us ({c_f3}) "
              f"| impl 4 below")
        t_pre_stock, _ = time_sustained(lambda: mhc_pre_delayed_tilelang(
            res, w["fn"], w["scale"], w["base"], EPS["rms_eps"], EPS["hc_pre_eps"], EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"],
            EPS["sinkhorn_repeat"], pre_mix=pre, norm_weight=w["norm"], norm_eps=EPS["norm_eps"]))
        t_pre_f, _ = time_sustained(lambda: fused(None, res, None, None, pre, w, impl=4))
        t_cp, c_cp = time_sustained(lambda: res_cur.copy_(res), 0.3)
        print(f"  T={T:6d}: torch copy of the residual (80 KB/token) {t_cp:7.1f} us = {T * 80 * 1024 / t_cp / 1e3:5.0f} GB/s ({c_cp})")
        nb = T * 100 * 1024
        print(f"  T={T:6d}: stock post {t1:7.1f} + tf32 gemm {t2:7.1f} (splits {splits}) + pre-norm {t3:7.1f} = {t1 + t2 + t3:7.1f}"
              f" (call {t_stock:7.1f}, {c_stock}) | fused {t_f:7.1f} [{nb / t_f / 1e3:5.0f} GB/s] ({c_f}) x{t_stock / t_f:4.2f}"
              f" | PRE: stock {t_pre_stock:7.1f} fused {t_pre_f:7.1f} [{T * 50 * 1024 / t_pre_f / 1e3:5.0f} GB/s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 2048, 16384])
    args, rest = ap.parse_known_args()
    if args.bench:
        require_gpu()
        bench(args.layer, args.tokens)
    else:
        sys.exit(run_module_tests(globals(), rest))
