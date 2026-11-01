"""Tests + micro-benchmarks for the ds41h 'prefill' component (W8A8 dense linears, FP8 wo_a, small-head
sparse prefill, PR #61875 MoE port).

    CUDA_VISIBLE_DEVICES="" python tests/test_prefill.py cpu      # CPU-only subset (real checkpoint weights)
    python tests/test_prefill.py                                  # all tests (GPU tests need an sm90 GPU)
    python -m pytest tests/test_prefill.py -q                     # same, if pytest is installed
    python tests/test_prefill.py --bench [--layer 10] [--rank 0]  # GPU timings: dense linears (stock Humming
                                                                  # W8A16 vs Humming W8A8 vs DeepGEMM W8A8),
                                                                  # wo_a (BF16 bmm vs FP8 einsum), sparse
                                                                  # prefill (padded 64 heads vs 16 heads)

Env: DS41H_TEST_LAYER (default 10), DS41H_TEST_RANK (default 0), DS41H_MODEL_DIR (checkpoint).
GPU tests need ~6 GB on one device; the attention tests need the FlashMLA small-head build
(scripts/build_flashmla_sm90.sh), the linear tests DeepGEMM (vendored in vLLM) and Humming.
References: exact checkpoint weights (ds41h.ref.weights dequant, 32x32 ue8m0), the stock sm90 kernels
themselves (HummingMxfp8LinearKernel on a stand-in layer, vendored FlashMLA padded to 64 heads, the stock
deep_gemm_fp8_o_proj BF16 branch), and fp32 torch references of the quantized operands.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import assert_close, compare, fmt_metrics, has_cuda, reader, run_module_tests, skip  # noqa: E402

from ds41h import linear, prefill_attn  # noqa: E402
from ds41h.ref.quant import as_u8  # noqa: E402
from ds41h.ref.weights import dequant_fp8, vllm_scale_layout  # noqa: E402

LAYER = int(os.environ.get("DS41H_TEST_LAYER", "10"))
RANK = int(os.environ.get("DS41H_TEST_RANK", "0"))
ENGRAM_LAYER = 1
torch.backends.cuda.matmul.allow_tf32 = False


# ===================================================================================== helpers
def require_gpu():
    if not has_cuda() or torch.cuda.get_device_capability() != (9, 0):
        skip("needs an sm90 GPU")


def require_deepgemm():
    require_gpu()
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        skip("DeepGEMM not supported/available (VLLM_USE_DEEP_GEMM=0?)")


def dense_weights(layer: int = LAYER, rank: int = RANK) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{name: (e4m3 [N, K], ue8m0 block scale [N/32, K/32])} of the dense FP8 linears of one TP rank, as
    stock vLLM 0.32.0 shards them (fused_wqa_wkv stacked like MergedColumnParallelLinear)."""
    rd = reader()
    p = f"layers.{layer}."
    g = lambda n: rd.get_tp(p + n, rank)  # noqa: E731
    out = {
        "fused_wqa_wkv": (torch.cat([g("attn.wq_a.weight"), g("attn.wkv.weight")]),
                          torch.cat([as_u8(g("attn.wq_a.scale")), as_u8(g("attn.wkv.scale"))])),
        "wq_b": (g("attn.wq_b.weight"), as_u8(g("attn.wq_b.scale"))),
        "wo_b": (g("attn.wo_b.weight"), as_u8(g("attn.wo_b.scale"))),
        "shared.gate_up": (torch.cat([g("ffn.shared_experts.w1.weight"), g("ffn.shared_experts.w3.weight")]),
                           torch.cat([as_u8(g("ffn.shared_experts.w1.scale")), as_u8(g("ffn.shared_experts.w3.scale"))])),
        "shared.down": (g("ffn.shared_experts.w2.weight"), as_u8(g("ffn.shared_experts.w2.scale"))),
    }
    if (p + "attn.indexer.wq_b.weight") in rd:
        out["indexer.wq_b"] = (g("attn.indexer.wq_b.weight"), as_u8(g("attn.indexer.wq_b.scale")))
    return out


def engram_wkv(rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    rd = reader()
    p = f"layers.{ENGRAM_LAYER}.engram.wkv."
    return rd.get_tp(p + "weight", rank), as_u8(rd.get_tp(p + "scale", rank))


def woa_weight(layer: int = LAYER, rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    rd = reader()
    p = f"layers.{layer}.attn.wo_a."
    return rd.get_tp(p + "weight", rank), as_u8(rd.get_tp(p + "scale", rank))


def realistic_x(M: int, K: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """bf16 activations with a few large outlier channels (like normed residual streams / SwiGLU outputs)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g)
    x = x * (1 + 15 * (torch.rand(K, generator=g) < 0.005))
    return x.to(torch.bfloat16).to(device)


def humming_stand_in(w: torch.Tensor, s_rows: torch.Tensor, w8a8: bool):
    """Stock ``HummingMxfp8LinearKernel`` on a minimal stand-in layer (the attributes Humming reads from a
    vLLM LinearBase). ``w8a8`` uses ds41h's MXFP8-activation post-load instead of the stock BF16 fallback."""
    from vllm.model_executor.kernels.linear.mxfp8.humming import HummingMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig

    n, k = w.shape
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s_rows.clone(), requires_grad=False)
    layer.input_size_per_partition = k
    layer.input_size = k
    layer.output_partition_sizes = [n]
    layer.params_dtype = torch.bfloat16
    layer.has_bias = False
    kernel = HummingMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    if w8a8:
        linear._humming_mxfp8_process(kernel, layer)
    else:
        kernel.process_weights_after_loading(layer)
    return lambda x: kernel.apply_weights(layer, x)


def rope_cache(max_pos: int, rope_dim: int = 64, device: str = "cuda") -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], -1).float().to(device)


# ===================================================================================== CPU tests
def test_cpu_fold_synthetic_exact_and_padding():
    """Uniform exponents fold exactly; K = 576 pads to 640 with zero columns; row-expanded == block scales."""
    g = torch.Generator().manual_seed(1)
    n, k = 256, 576
    w = (torch.randn(n, k, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.full((n // 32, k // 32), 120, dtype=torch.uint8)
    wf, s128, st = linear.fold_block_scales(w, s)
    assert wf.shape == (n, 640) and s128.shape == (2, 5), (wf.shape, s128.shape)
    assert st["changed"] == 0 and st["spread"] == 0, st
    assert torch.equal(as_u8(wf[:, :k]), as_u8(w)) and not as_u8(wf[:, k:]).any()
    assert torch.allclose(s128, torch.full_like(s128, 2.0 ** (120 - 127)))
    wf2, s2, _ = linear.fold_block_scales(w, s.repeat_interleave(32, 0))
    assert torch.equal(as_u8(wf2), as_u8(wf)) and torch.equal(s2, s128)
    # a sub-block one binade lower: values are halved and re-rounded; dequant matches except sub-normals
    s[1, 2] = 119
    wf3, s3, st3 = linear.fold_block_scales(w, s)
    ref = linear.dequant_checkpoint(w, s)
    fw = linear.FoldedWeight(w=wf3, ws=s3, s128=s3, n=n, k=k, kp=640)
    err = (linear.dequant_folded(fw) - ref).abs()
    assert st3["spread"] == 1
    assert float(err.max()) <= 2.0 ** -10 * 2.0 ** (120 - 127) + 1e-12, float(err.max())
    assert int((err > 0).sum()) == st3["changed"]


def test_cpu_fold_real_weights_near_exact():
    """Real dense FP8 weights: 128x128 folding changes < 1e-4 of the elements, rel L2 < 1e-5."""
    ws = dense_weights()
    for name, (w, s) in ws.items():
        wf, s128, st = linear.fold_block_scales(w, s)
        fw = linear.FoldedWeight(w=wf, ws=s128, s128=s128, n=w.shape[0], k=w.shape[1], kp=wf.shape[1])
        m = compare(linear.dequant_folded(fw), dequant_fp8(w, s))
        print(f"  {name:16s} {tuple(w.shape)} changed {st['changed']}/{st['numel']} spread {st['spread']} "
              f"rel_l2 {st['rel_l2']:.2e} (check {m['rel_rmse']:.2e})")
        assert st["changed"] / st["numel"] < 1e-4, (name, st)
        assert st["rel_l2"] < 1e-5 and abs(m["rel_rmse"] - st["rel_l2"]) < 1e-6 + 0.01 * st["rel_l2"], (name, st, m)


def test_cpu_act_quant_ref_properties():
    x = realistic_x(37, 576, seed=2)
    for ue8m0 in (True, False):
        q, s = linear.act_quant_ref(x, 640, ue8m0)
        assert q.shape == (37, 640) and s.shape == (37, 5)
        assert not as_u8(q[:, 576:]).any()
        if ue8m0:
            assert torch.equal(torch.exp2(torch.log2(s).round()), s), "scales must be powers of two"
        xd = (q.float().view(37, 5, 128) * s[..., None]).view(37, 640)[:, :576]
        m = compare(xd, x.float())
        assert m["rel_rmse"] < 0.05, fmt_metrics(m)


def test_cpu_gemm_ref_vs_exact_real_weights():
    """W8A8 (1x128 act, 128x128 folded weights) vs exact W8A16 math on real weights + realistic inputs."""
    for name, (w, s) in dense_weights().items():
        fw, _ = linear.fold_weight(w, s, with_deepgemm_layout=False)
        x = realistic_x(64, w.shape[1], seed=3)
        y = linear.gemm_w8a8_ref(x, fw)
        y_exact = x.float() @ dequant_fp8(w, s).t()
        m = assert_close(y, y_exact, cos_min=0.999, rel_rmse_max=0.04, name=name)
        print(f"  {name:16s} {fmt_metrics(m)}")


def test_cpu_raw_capture_exact_dequant():
    """Lossless W8A16 weights: capture_raw + dequant_bf16_ref == exact checkpoint value (bf16 holds it exactly) for every
    dense FP8 weight of a real layer; block and row-expanded scale forms agree; out-of-range scales are rejected."""
    for name, (w, s) in dense_weights().items():
        rw = linear.capture_raw(w, s)
        assert rw.s.shape == (-(-w.shape[0] // 32), w.shape[1] // 32) and rw.s.dtype == torch.uint8
        exact = dequant_fp8(w, s)
        got = linear.dequant_bf16_ref(rw)
        assert torch.equal(got.float(), exact), (name, int((got.float() != exact).sum()))
        rw2 = linear.capture_raw(w, vllm_scale_layout(s))
        assert torch.equal(rw2.s, rw.s) and torch.equal(rw2.w, rw.w)
    w = torch.zeros(64, 64, dtype=torch.float8_e4m3fn)
    for bad in (9, 247):
        try:
            linear.capture_raw(w, torch.full((2, 2), bad, dtype=torch.uint8))
            raise AssertionError(f"scale byte {bad} must be rejected")
        except ValueError:
            pass


def test_cpu_sharded_rows_apply():
    """WQA_SHARD row sharding: per-rank shard sizes / padding and the all-gathered result == the unsharded op (M not a
    multiple of the TP size, M < TP size, N-D inputs)."""
    g = torch.Generator().manual_seed(3)
    w = torch.randn(48, 32, generator=g)
    fn = lambda xs: (xs.float() @ w.t()).to(torch.bfloat16)  # noqa: E731 - row-wise op
    for shape in ((1024, 32), (1023, 32), (5, 32), (3, 32), (1, 32), (2, 7, 32)):
        x = torch.randn(*shape, generator=g).to(torch.bfloat16)
        locs: dict[int, torch.Tensor] = {}

        class Rec:
            def __init__(self, r):
                self.r = r

            def all_gather(self, y, dim=0):
                assert dim == 0 and y.is_contiguous()
                locs[self.r] = y.clone()
                return torch.cat([y] * 4)

        for r in range(4):
            linear.sharded_rows_apply(fn, x, 48, (r, 4, Rec(r)))
        assert len({tuple(v.shape) for v in locs.values()}) == 1  # equal shard sizes (padded)
        full = torch.cat([locs[r] for r in range(4)])

        class Cat:
            def all_gather(self, y, dim=0):
                return full

        ref = fn(x.reshape(-1, 32)).view(*shape[:-1], 48)
        for r in range(4):
            y = linear.sharded_rows_apply(fn, x, 48, (r, 4, Cat()))
            assert y.shape == ref.shape and torch.equal(y, ref), (shape, r)


def test_cpu_inv_rope_ref_formula():
    """inv_rope_bf16_ref == the stock Triton formula (x * cos +- partner * sin on dims >= 448) in fp64, rounded to bf16."""
    g = torch.Generator().manual_seed(5)
    o = (torch.randn(7, 16, 512, generator=g) * 0.7).to(torch.bfloat16)
    pos = torch.randint(0, 100, (7,), generator=g)
    cs = rope_cache(100, device="cpu")
    got = linear.inv_rope_bf16_ref(o, pos, cs)
    x = o.double()
    c, sn = cs[pos][:, None, :32].double(), cs[pos][:, None, 32:].double()
    exp = x.clone()
    exp[..., 448::2] = x[..., 448::2] * c + x[..., 449::2] * sn
    exp[..., 449::2] = x[..., 449::2] * c - x[..., 448::2] * sn
    assert torch.equal(got[..., :448], o[..., :448])
    d = (got.double() - exp).abs()
    ulp = torch.exp2(torch.floor(torch.log2(exp.abs().clamp_min(1e-30))) - 7)
    assert float((d / ulp).max()) <= 0.5 + 1e-6  # correctly rounded up to fp32 intermediate rounding


def test_cpu_modelopt_wrapper_passthrough():
    """install_modelopt is idempotent and leaves layers without a captured weight on the previous path."""
    calls = []

    class ModelOptLinearMethod:
        def process_weights_after_loading(self, layer):
            calls.append("pwal")

        def apply(self, layer, x, bias=None):
            calls.append("apply")
            return "stock"

    mod = types.ModuleType("fake_modelopt")
    mod.ModelOptLinearMethod = ModelOptLinearMethod
    mod.kMxfp8Static = object()
    mod.WEIGHT, mod.ACT = "weight", "activation"
    mod.is_weights_pre_processed = lambda: False
    linear.install_modelopt(mod)
    linear.install_modelopt(mod)
    assert ModelOptLinearMethod.apply.__wrapped__.__name__ == "apply"
    assert getattr(ModelOptLinearMethod.apply.__wrapped__, "__wrapped__", None) is None, "wrapped twice"
    m = ModelOptLinearMethod()
    layer = torch.nn.Module()
    layer.prefix = "model.layers.3.attn.wq_b"
    m.process_weights_after_loading(layer)
    assert m.apply(layer, torch.zeros(4, 8)) == "stock" and calls == ["pwal", "apply"], calls
    # a wo_a already converted to FP8 einsum weights never reaches the stock (BF16 emulation) post-load again
    woa = torch.nn.Module()
    woa.prefix = "model.layers.3.attn.wo_a"
    woa._ds41h_woa_fp8 = True
    m.process_weights_after_loading(woa)
    assert calls == ["pwal", "apply"], calls
    assert linear.MIN_M >= 1


def test_cpu_linear_layer_selection():
    """Default capture = full-row layers: < cut (max kv_source_layer_ids) + the cut layer's fused_wqa_wkv only."""
    saved = linear._LAYERS, linear._LAYERS_ALL, dict(linear._CFG_CACHE)
    try:
        linear._LAYERS, linear._LAYERS_ALL = None, False
        linear._CFG_CACHE.update({"replay_cut": 20, "layers": 40})
        tl = linear.target_layer
        assert tl("model.layers.0.attn.wq_b") == 0 and tl("model.layers.19.ffn.shared_experts.down_proj") == 19
        assert tl("model.layers.14.engram.wkv") == 14 and tl("model.layers.8.attn.indexer.wq_b") == 8
        assert tl("model.layers.20.attn.fused_wqa_wkv") == 20 and tl("model.layers.20.attn.wq_b") is None
        assert tl("model.layers.21.attn.fused_wqa_wkv") is None and tl("model.layers.40.attn.wq_b") is None
        assert tl("model.layers.30.attn.wo_a", linear.WOA_RE) == 30  # wo_a conversion is not restricted
        linear._LAYERS_ALL = True
        assert tl("model.layers.30.attn.wq_b") == 30 and tl("model.layers.40.attn.wq_b") is None
        linear._LAYERS_ALL, linear._LAYERS = False, {3, 5}
        assert tl("model.layers.3.attn.wq_b") == 3 and tl("model.layers.4.attn.wq_b") is None
    finally:
        linear._LAYERS, linear._LAYERS_ALL = saved[0], saved[1]
        linear._CFG_CACHE.clear()
        linear._CFG_CACHE.update(saved[2])


def test_cpu_moe_fp8act_port():
    """PR #61875 port: the stock Mxfp4MoEMethod (Humming, sm90, no env override) gets kFp8Dynamic128Sym."""
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
    from vllm.platforms import current_platform

    class Mxfp4MoEMethod:
        def __init__(self, backend):
            self.activation_key = None
            self.mxfp4_backend = backend

    class Other:
        activation_key = None

    class DeepseekV4FP8Config:
        def get_quant_method(self, layer, prefix):
            return layer

    mod = types.ModuleType("fake_quant_config")
    mod.DeepseekV4FP8Config, mod.Mxfp4MoEMethod = DeepseekV4FP8Config, Mxfp4MoEMethod
    linear.install_moe_fp8_activations(mod)
    linear.install_moe_fp8_activations(mod)
    cfg = DeepseekV4FP8Config()
    saved = current_platform.is_device_capability_family
    saved_env = os.environ.pop("VLLM_HUMMING_INPUT_QUANT_CONFIG", None)
    try:
        current_platform.is_device_capability_family = lambda cap: cap == 90  # type: ignore[method-assign]
        if hasattr(envs, "__getattr__") and hasattr(envs, "disable_envs_cache"):
            envs.disable_envs_cache()
        a = cfg.get_quant_method(Mxfp4MoEMethod(Mxfp4MoeBackend.HUMMING), "model.layers.0.ffn.experts")
        b = cfg.get_quant_method(Mxfp4MoEMethod(Mxfp4MoeBackend.MARLIN), "model.layers.0.ffn.experts")
        c = cfg.get_quant_method(Other(), "model.layers.0.ffn.experts")
        assert a.activation_key == kFp8Dynamic128Sym, a.activation_key
        assert b.activation_key is None and c.activation_key is None
        current_platform.is_device_capability_family = lambda cap: cap == 100  # type: ignore[method-assign]
        d = cfg.get_quant_method(Mxfp4MoEMethod(Mxfp4MoeBackend.HUMMING), "x")
        assert d.activation_key is None
    finally:
        current_platform.is_device_capability_family = saved  # type: ignore[method-assign]
        if saved_env is not None:
            os.environ["VLLM_HUMMING_INPUT_QUANT_CONFIG"] = saved_env


def test_cpu_humming_schema_support():
    """The facts linear.py relies on (installed humming, sm90 faked): MXFP8 x MXFP8 and per-token FP8 x
    MXFP8 are accepted, FP8-group-128 x MXFP8 is not (only MXFP4 weights take it, the #61875 MoE case);
    vLLM's stock schema check falls back to BF16 activations unless allow_fallback=False."""
    import humming.schema.humming as hs
    from vllm.model_executor.layers.quantization.utils import humming as vh
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    saved_dev, saved_cap = hs.current_device, current_platform.get_device_capability
    hs.current_device = types.SimpleNamespace(sm_version=90, is_ppu=False)
    current_platform.get_device_capability = lambda *a, **k: DeviceCapability(9, 0)  # type: ignore[method-assign]
    try:
        w8 = hs.HummingWeightSchema(b_dtype="float8e4m3", bs_dtype="float8e8m0", weight_scale_group_size=32)
        w4 = hs.HummingWeightSchema(b_dtype="float4e2m1", bs_dtype="float8e8m0", weight_scale_group_size=32)
        mx = hs.HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=32, input_scale_dtype="float8e8m0",
                                   input_quant_mode="dynamic_group")
        blk = hs.HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=128, input_quant_mode="dynamic_group")
        tok = hs.HummingInputSchema(a_dtype="float8e4m3", input_scale_dtype="float32", input_quant_mode="dynamic_token")
        assert hs.is_humming_schema_compatible(mx, w8, torch.bfloat16)
        assert hs.is_humming_schema_compatible(tok, w8, torch.bfloat16)
        assert not hs.is_humming_schema_compatible(blk, w8, torch.bfloat16)
        assert hs.is_humming_schema_compatible(blk, w4, torch.bfloat16)
        fb = vh.check_and_fallback_input_schema(w8, mx, torch.bfloat16, allow_fallback=True)
        assert fb.a_dtype is None or fb.a_dtype.num_bits == 16, fb  # stock: W8A16
        keep = vh.check_and_fallback_input_schema(w8, mx, torch.bfloat16, allow_fallback=False)
        assert keep.a_dtype is not None and keep.a_dtype.num_bits == 8 and keep.input_scale_group_size == 32, keep
    finally:
        hs.current_device = saved_dev
        current_platform.get_device_capability = saved_cap  # type: ignore[method-assign]


def test_cpu_sparse_ref_vs_loop():
    g = torch.Generator().manual_seed(4)
    S, H, D, NKV, TOPK = 5, 4, 512, 40, 16
    q = torch.randn(S, H, D, generator=g)
    kv = torch.randn(NKV, 1, D, generator=g)
    idx = torch.randint(-1, NKV + 3, (S, 1, TOPK), generator=g, dtype=torch.int32)
    tl = torch.randint(1, TOPK + 1, (S,), generator=g, dtype=torch.int32)
    sink = torch.randn(H, generator=g)
    out, ml, lse = prefill_attn.sparse_prefill_ref(q, kv, idx, 0.05, 512, sink, tl)
    for i in range(S):
        sel = [int(j) for j in idx[i, 0, : int(tl[i])] if 0 <= int(j) < NKV]
        if not sel:
            assert torch.all(out[i] == 0)
            continue
        k = kv[sel, 0]
        lg = q[i] @ k.t() * 0.05
        l_ = torch.logsumexp(lg, -1)
        o = torch.softmax(lg, -1) @ k * (torch.exp(l_) / (torch.exp(l_) + torch.exp(sink)))[:, None]
        assert torch.allclose(o, out[i], atol=1e-4, rtol=1e-4) and torch.allclose(l_, lse[i], atol=1e-5)


def test_cpu_attn_dispatch_falls_back_to_stock():
    """CPU tensors (or no extension) -> the dispatcher calls the stock function with the unchanged padded q."""
    seen = {}

    def stock(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None):
        seen["h"] = q.shape[1]
        return out, None, None

    d = prefill_attn.make_dispatch(stock)
    prefill_attn._STATE.layer = types.SimpleNamespace(n_local_heads=16)
    try:
        d(q=torch.zeros(2, 64, 512), kv=torch.zeros(3, 1, 512), indices=torch.zeros(2, 1, 128, dtype=torch.int32),
          sm_scale=0.1, attn_sink=torch.zeros(64), topk_length=None, out=torch.zeros(2, 64, 512))
    finally:
        prefill_attn._STATE.layer = None
    assert seen["h"] == 64  # CPU tensors never take the CUDA kernels (q.is_cuda gate), with or without the extension


def test_cpu_nopad_gating():
    """NOPAD only in fully eager prefill-only forwards: never in decode / mixed steps, under a stream capture,
    with an open breakable capture, or when the forward context's cudagraph mode is not NONE (breakable
    eager segments re-run _fused_qnorm_rope_kv_insert in Python at every replay, while the graph-recorded
    _alloc_attn_out buffer keeps the padded head count)."""
    import vllm.forward_context as fc
    from vllm.config import CUDAGraphMode

    import ds41h.patch_core as pc

    base = dict(num_tokens=8192, decode_only=False, num_reqs=1, num_decode_tokens=0, max_decode_query_len=None,
                cudagraph_mode="NONE", capturing=False, has_attn_metadata=True)
    cur = {"step": pc.StepInfo(**base), "mode": CUDAGraphMode.NONE}
    layer = types.SimpleNamespace(n_local_heads=16, padded_heads=64)
    saved = (prefill_attn.NOPAD, prefill_attn.available, pc.current_step, fc.get_forward_context,
             fc.is_forward_context_available)
    try:
        prefill_attn.NOPAD = True
        prefill_attn.available = lambda: True
        pc.current_step = lambda: cur["step"]
        fc.is_forward_context_available = lambda: True
        fc.get_forward_context = lambda: types.SimpleNamespace(cudagraph_runtime_mode=cur["mode"])
        assert prefill_attn._nopad_active(layer)
        assert not prefill_attn._nopad_active(types.SimpleNamespace(n_local_heads=64, padded_heads=64))
        for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
            cur["mode"] = mode
            assert not prefill_attn._nopad_active(layer), mode
        cur["mode"] = CUDAGraphMode.NONE
        for kw in ({"num_decode_tokens": 3}, {"decode_only": True}, {"capturing": True}, {"has_attn_metadata": False}):
            cur["step"] = pc.StepInfo(**{**base, **kw})
            assert not prefill_attn._nopad_active(layer), kw
        cur["step"] = pc.StepInfo(**base)
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        BreakableCUDAGraphCapture._tls.active = object()  # inside an eager segment of a breakable capture
        try:
            assert not prefill_attn._nopad_active(layer)
        finally:
            BreakableCUDAGraphCapture._tls.active = None
        assert prefill_attn._nopad_active(layer)
    finally:
        (prefill_attn.NOPAD, prefill_attn.available, pc.current_step, fc.get_forward_context,
         fc.is_forward_context_available) = saved


def test_cpu_prefill_profile_script():
    """scripts/prefill_profile.py on a synthetic Kineto trace: correlation-based step attribution + categories."""
    import gzip
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location("prefill_profile", Path(__file__).resolve().parents[1] / "scripts" / "prefill_profile.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": "execute_context_1(8192)_generation_0(0)", "ts": 0, "dur": 1000, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "user_annotation", "name": "execute_context_0(0)_generation_1(6)", "ts": 2000, "dur": 100, "pid": 1, "tid": 1},
    ]
    kernels = [  # (name, launch ts, gpu ts, dur)
        ("void humming_kernel<128, 256>(...)", 10, 1100, 300),
        ("sparse_attn_fwd_small_heads_kernel<512, true, 16>", 20, 1400, 200),
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 30, 1600, 100),
        ("moe_gemm_kernel<1>", 40, 1700, 400),
        ("sm90_fp8_mqa_logits", 50, 2100, 50),
        ("mhc_shifted_post_pre_kernel", 60, 2150, 30),
        ("_engram_lookup_kernel", 70, 2180, 10),
        ("vectorized_elementwise_kernel", 80, 2190, 5),
        ("decode_only_kernel", 2010, 2300, 7),  # launched inside the decode step
    ]
    for i, (name, lts, gts, dur) in enumerate(kernels):
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": lts, "dur": 2, "pid": 1, "tid": 1,
                   "args": {"correlation": 100 + i}})
        ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": gts, "dur": dur, "pid": 0, "tid": 7,
                   "args": {"correlation": 100 + i}})
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rank0.pt.trace.json.gz"
        with gzip.open(p, "wt") as f:
            json.dump({"traceEvents": ev}, f)
        steps = pp.split_steps(pp.load_trace(p), "auto", 3.0)
        assert len(steps) == 2 and steps[0].ctx_tokens == 8192 and len(steps[0].kernels) == 8
        r = pp.aggregate(pp.pick_step(steps, "largest"), pp.compile_rules(None))
    cats = {c: round(v["ms"] * 1e3) for c, v in r["categories"].items()}
    assert cats == {"moe gemm": 400, "dense gemm": 300, "attention": 200, "indexer": 50, "mhc": 30, "allreduce": 100,
                    "engram": 10, "other": 5}, cats


# ===================================================================================== GPU tests: linears
def test_act_quant_exact():
    """CUDA act_quant_128 == torch reference (bits), incl. K tails, strided rows, both scale modes."""
    require_gpu()
    for K in (576, 1280, 2048, 5120, 6144):
        kp = -(-K // 128) * 128
        for M in (1, 3, 256, 1000):
            base = realistic_x(M, K + 64, seed=K + M, device="cuda") * 3
            for x in (base[:, :K].contiguous(), base[:, :K]):  # contiguous and row stride K + 64
                for ue8m0 in (True, False):
                    q, s = linear.act_quant(x, kp, ue8m0)
                    qr, sr = linear.act_quant_ref(x, kp, ue8m0)
                    assert s.stride() == (1, -(-M // 4) * 4), s.stride()
                    if ue8m0:
                        assert torch.equal(s, sr), (K, M)
                        assert torch.equal(as_u8(q), as_u8(qr)), (K, M, int((as_u8(q) != as_u8(qr)).sum()))
                    else:  # fast-math 448/amax may differ from IEEE in the last bit
                        assert torch.allclose(s, sr, rtol=1e-6, atol=0)
                        # at most one e4m3 step apart: a step is 1/16..1/8 of the value (2^-9 in the subnormals)
                        bad = (q.float() - qr.float()).abs() > (qr.float().abs() * 0.125 + 2.0 ** -9)
                        assert int(bad.sum()) == 0, (K, M)
                        assert float((as_u8(q) != as_u8(qr)).float().mean()) < 1e-3


def test_gemm_w8a8_vs_ref_and_stock():
    """DeepGEMM W8A8 vs (a) fp32 math on the same quantized operands (tight), (b) the stock Humming W8A16
    kernel and exact W8A16 math (FP8-activation tolerance), for every dense linear of a real layer."""
    require_deepgemm()
    weights = dense_weights()
    weights["engram.wkv"] = engram_wkv()
    for name, (w, s) in weights.items():
        w, s = w.cuda(), s.cuda()
        fw, st = linear.fold_weight(w, s)
        for M in (256, 1000, 4096):
            x = realistic_x(M, w.shape[1], seed=M, device="cuda")
            y = linear.gemm_w8a8(x, fw)
            m_ref = assert_close(y.float(), linear.gemm_w8a8_ref(x, fw), cos_min=0.99999, rel_rmse_max=4e-3,
                                 name=f"{name} M={M} vs ref")
            y_exact = (x.float() @ dequant_fp8(w, s).t())
            m_ex = assert_close(y.float(), y_exact, cos_min=0.999, rel_rmse_max=0.04, name=f"{name} M={M} vs exact")
            if M == 1000:
                y_stock = humming_stand_in(w, vllm_scale_layout(s), w8a8=False)(x)
                m_st = assert_close(y.float(), y_stock.float(), cos_min=0.999, rel_rmse_max=0.04, name=f"{name} vs stock")
                print(f"  {name:16s} ref {m_ref['rel_rmse']:.1e}  exact {m_ex['rel_rmse']:.2e}  stock {m_st['rel_rmse']:.2e} "
                      f"(fold changed {st['changed']})")


def test_humming_w8a8_vs_stock():
    """impl humming (MXFP8 x MXFP8): close to the exact W8A16 math (reference recipe, 1x32 ue8m0 activations)."""
    require_gpu()
    try:
        import humming

        ver = getattr(humming, "__version__", "0")
    except ImportError:
        skip("humming not installed")
    if tuple(int(v) for v in ver.split(".")[:3] if v.isdigit()) <= (0, 1, 17):
        # measured: "no legal grouped-scale SM90 schedule: WGMMA input scales must use float32 storage,
        # got float8e8m0" for every candidate -> impl "humming" is unusable on 0.1.17 (default impl is deepgemm)
        skip(f"humming {ver} has no sm90 schedule for e8m0 input scales")
    for name, (w, s) in dense_weights().items():
        if w.shape[1] % 128:
            continue  # humming 0.1.17 packed e8m0 input-scale padding bug (K/32 % 4 != 0), see linear.py
        w, s = w.cuda(), s.cuda()
        x = realistic_x(512, w.shape[1], seed=5, device="cuda")
        y = humming_stand_in(w, vllm_scale_layout(s), w8a8=True)(x)
        m = assert_close(y.float(), x.float() @ dequant_fp8(w, s).t(), cos_min=0.999, rel_rmse_max=0.04, name=name)
        print(f"  {name:16s} {fmt_metrics(m)}")


def test_modelopt_wrapper_dispatch_gpu():
    """install_modelopt on a stand-in method: capture at load, W8A8 for M >= MIN_M, previous apply below."""
    require_deepgemm()
    w, s = dense_weights()["wq_b"]
    w, s = w.cuda(), vllm_scale_layout(s.cuda())

    class ModelOptLinearMethod:
        def __init__(self):
            self.spec = types.SimpleNamespace(weight=mod.kMxfp8Static)

        def process_weights_after_loading(self, layer):
            layer.stock_w = dequant_fp8(layer.weight.data, layer.weight_scale.data).to(torch.bfloat16)

        def apply(self, layer, x, bias=None):
            return x @ layer.stock_w.t()

    mod = types.ModuleType("fake_modelopt_gpu")
    mod.ModelOptLinearMethod = ModelOptLinearMethod
    mod.kMxfp8Static = object()
    mod.WEIGHT, mod.ACT = "weight", "activation"
    mod.is_weights_pre_processed = lambda: False
    saved = linear.LINEAR, linear.IMPL
    linear.LINEAR, linear.IMPL = True, "deepgemm"
    try:
        linear.install_modelopt(mod)
        layer = torch.nn.Module()
        layer.prefix = f"model.layers.{LAYER}.attn.wq_b"
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        meth = ModelOptLinearMethod()
        meth.process_weights_after_loading(layer)
        assert "_ds41h_w8a8" in layer.__dict__
        fw = layer._ds41h_w8a8
        big = realistic_x(linear.MIN_M + 7, w.shape[1], seed=6, device="cuda")
        small = big[: linear.MIN_M - 1]
        assert torch.equal(meth.apply(layer, big), linear.gemm_w8a8(big, fw))
        assert torch.equal(meth.apply(layer, small), small @ layer.stock_w.t())
        y3 = meth.apply(layer, big.view(1, *big.shape))  # N-D input keeps its leading dims
        assert y3.shape == (1, big.shape[0], w.shape[0])
    finally:
        linear.LINEAR, linear.IMPL = saved


def test_woa_fp8_einsum_vs_stock_bf16():
    """wo_a converted by convert_woa + stock deep_gemm_fp8_o_proj FP8 branch vs the stock BF16 branch."""
    require_deepgemm()
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    w, s = woa_weight()
    w, s = w.cuda(), s.cuda()
    T = 1000
    g = torch.Generator(device="cuda").manual_seed(7)
    o = (torch.randn(T, 16, 512, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64) * 7
    cs = rope_cache(8 * T)

    def run(wo_a):
        return deep_gemm_fp8_o_proj(o, positions, cs, wo_a, lambda z: z, n_groups=2, heads_per_group=8, nope_dim=448,
                                    rope_dim=64, o_lora_rank=1024, einsum_recipe=(1, 128, 128), tma_aligned_scales=False)

    stock = torch.nn.Module()
    stock.weight = torch.nn.Parameter(dequant_fp8(w, s).to(torch.bfloat16), requires_grad=False)  # Emulation at load
    fp8 = torch.nn.Module()
    fp8.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    fp8.weight_scale = torch.nn.Parameter(vllm_scale_layout(s), requires_grad=False)
    fp8.bmm_batch_size = 2
    fp8.prefix = f"model.layers.{LAYER}.attn.wo_a"
    assert linear.convert_woa(fp8) and fp8.weight.dtype == torch.float8_e4m3fn and fp8.weight.dim() == 3
    m = assert_close(run(fp8).float(), run(stock).float(), cos_min=0.999, rel_rmse_max=0.04, name="wo_a")
    print(f"  wo_a FP8 einsum vs stock BF16 bmm: {fmt_metrics(m)}")


def test_inv_rope_fp8_quant_vs_stock():
    """ds41h inv_rope_fp8_quant == vLLM's Triton fused_inv_rope_fp8_quant (SM90 layout): scales and e4m3 bytes."""
    require_gpu()
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    g = torch.Generator(device="cuda").manual_seed(9)
    cs = rope_cache(70000)
    for T, strided in ((1, False), (37, False), (1000, True), (8188, False)):
        base = torch.randn(T, 64 if strided else 16, 512, device="cuda", generator=g) * 0.7
        base[:, :, 300] *= 40  # outlier channel
        o = base.to(torch.bfloat16)[:, :16]
        pos = torch.randint(0, 70000, (T,), device="cuda", generator=g)
        ref_q, ref_s = fused_inv_rope_fp8_quant(o, pos, cs, n_groups=2, heads_per_group=8)
        q, sc = linear.inv_rope_fp8_quant(o, pos, cs, 2)
        assert q.shape == ref_q.shape and sc.shape == ref_s.shape and q.stride() == ref_q.stride() and sc.stride() == ref_s.stride()
        assert torch.equal(sc, ref_s), (T, float((sc != ref_s).float().mean()))
        d = (as_u8(q).int() - as_u8(ref_q).int()).abs()
        frac = float((d != 0).float().mean())
        print(f"  T={T}: e4m3 bytes differing {frac:.2e} (max code diff {int(d.max())})")
        assert frac < 1e-4 and int(d.max()) <= 1, (T, frac)


# ===================================================================================== GPU tests: lossless linears
def _err_vs(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(rel RMSE, max |diff| in units of the reference's bf16 ulp) of y vs an fp64 reference."""
    d = (y.double() - ref).abs()
    rel = float(d.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-300))
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)  # bf16 ulp of |ref|
    return rel, float((d / ulp).max())


def test_dequant_fp8_bf16_exact():
    """dequant_fp8_bf16 == exact checkpoint value (bits) for every dense FP8 weight of a real layer + engram wkv."""
    require_gpu()
    weights = dense_weights()
    weights["engram.wkv"] = engram_wkv()
    for name, (w, s) in weights.items():
        rw = linear.capture_raw(w.cuda(), s.cuda())
        got = linear.dequant_bf16(rw)
        exact = dequant_fp8(w.cuda(), s.cuda())  # fp32 exact value
        assert torch.equal(got.float(), exact), (name, int((got.float() != exact).sum()))
        assert torch.equal(got, linear.dequant_bf16_ref(rw)), name
        # row-expanded (vLLM) scale form, strided output
        rw2 = linear.capture_raw(w.cuda(), vllm_scale_layout(s.cuda()))
        out = torch.empty(rw2.n, rw2.k + 64, dtype=torch.bfloat16, device="cuda")[:, : rw2.k]
        linear.dequant_bf16(rw2, out)
        assert torch.equal(out, got), name
    # e4m3 sub-normals and the full code range with extreme (but exact-range) scales
    codes = torch.arange(256, dtype=torch.uint8, device="cuda").repeat(64, 2)[:, :512]  # [64, 512]
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    for sb in (10, 120, 127, 246):
        sc = torch.full((2, 16), sb, dtype=torch.uint8, device="cuda")
        rw = linear.capture_raw(codes.view(torch.float8_e4m3fn), sc)
        exact = codes.view(torch.float8_e4m3fn).float() * 2.0 ** (sb - 127)
        assert torch.equal(linear.dequant_bf16(rw).float(), exact), sb
    try:
        linear.capture_raw(codes.view(torch.float8_e4m3fn), torch.full((2, 16), 250, dtype=torch.uint8, device="cuda"))
        raise AssertionError("scale outside the exact range must be rejected")
    except ValueError:
        pass


def test_gemm_w8a16_lossless_vs_stock():
    """Lossless W8A16 (exact dequant + bf16 GEMM, deepgemm and cublas) vs an fp64 reference of the exact W8A16 math:
    error no larger than stock Humming W8A16's (both are bf16-output-rounding bound), and within a few output ulps of
    stock element-wise."""
    require_deepgemm()
    weights = dense_weights()
    weights["engram.wkv"] = engram_wkv()
    for name, (w, s) in weights.items():
        w, s = w.cuda(), s.cuda()
        rw = linear.capture_raw(w, s)
        stock = humming_stand_in(w, vllm_scale_layout(s), w8a8=False)
        wexact = dequant_fp8(w, s).double()
        for M in (256, 1000, 4096):
            x = realistic_x(M, w.shape[1], seed=M + 1, device="cuda")
            ref = x.double() @ wexact.t()
            y_st = stock(x)
            e_st = _err_vs(y_st, ref)
            for impl in ("deepgemm", "cublas"):
                y = linear.gemm_w8a16(x, rw, impl=impl)
                e = _err_vs(y, ref)
                assert e[0] <= e_st[0] * 1.03 + 1e-7, (name, M, impl, e, e_st)
                assert e[1] <= max(e_st[1], 1.0) + 0.5, (name, M, impl, e, e_st)  # fp32 summation order only
                if M == 1000:
                    print(f"  {name:16s} M={M} {impl:8s} rel {e[0]:.3e} max {e[1]:.2f} ulp | stock rel {e_st[0]:.3e} max {e_st[1]:.2f} ulp")
        y3 = linear.gemm_w8a16(realistic_x(300, w.shape[1], device="cuda").view(3, 100, -1), rw)  # N-D input
        assert y3.shape == (3, 100, w.shape[0])
        del stock
        torch.cuda.empty_cache()


def test_modelopt_wrapper_dispatch_linear16_gpu():
    """install_modelopt with LINEAR16: raw capture at load, lossless W8A16 for M >= MIN_M, previous apply below."""
    require_deepgemm()
    w, s = dense_weights()["wq_b"]
    w, s = w.cuda(), vllm_scale_layout(s.cuda())

    class ModelOptLinearMethod:
        def __init__(self):
            self.spec = types.SimpleNamespace(weight=mod.kMxfp8Static)

        def process_weights_after_loading(self, layer):
            layer.stock_w = dequant_fp8(layer.weight.data, layer.weight_scale.data).to(torch.bfloat16)

        def apply(self, layer, x, bias=None):
            return x @ layer.stock_w.t()

    mod = types.ModuleType("fake_modelopt_gpu16")
    mod.ModelOptLinearMethod = ModelOptLinearMethod
    mod.kMxfp8Static = object()
    mod.WEIGHT, mod.ACT = "weight", "activation"
    mod.is_weights_pre_processed = lambda: False
    saved = linear.LINEAR, linear.LINEAR16
    linear.LINEAR, linear.LINEAR16 = False, True
    try:
        linear.install_modelopt(mod)
        layer = torch.nn.Module()
        layer.prefix = f"model.layers.{LAYER}.attn.wq_b"
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        meth = ModelOptLinearMethod()
        meth.process_weights_after_loading(layer)
        assert "_ds41h_w8a16" in layer.__dict__ and "_ds41h_w8a8" not in layer.__dict__
        rw = layer._ds41h_w8a16
        big = realistic_x(linear.MIN_M + 7, w.shape[1], seed=6, device="cuda")
        small = big[: linear.MIN_M - 1]
        assert torch.equal(meth.apply(layer, big), linear.gemm_w8a16(big, rw))
        assert torch.equal(meth.apply(layer, small), small @ layer.stock_w.t())
    finally:
        linear.LINEAR, linear.LINEAR16 = saved


def test_inv_rope_bf16_vs_stock():
    """In-place inv_rope_bf16_ == vLLM's Triton fused_inv_rope_fp8_quant(quantize=False) (bits), contiguous and
    padded-head (strided) attention outputs; no-RoPE dims untouched."""
    require_gpu()
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    g = torch.Generator(device="cuda").manual_seed(19)
    cs = rope_cache(70000)
    for T, strided in ((1, False), (37, True), (1000, False), (8188, True)):
        base = (torch.randn(T, 64 if strided else 16, 512, device="cuda", generator=g) * 0.7).to(torch.bfloat16)
        o = base[:, :16]
        pos = torch.randint(0, 70000, (T,), device="cuda", generator=g)
        ref, _ = fused_inv_rope_fp8_quant(o, pos, cs, n_groups=2, heads_per_group=8, quantize=False)  # [T, 2, 4096]
        before, pad = o.clone(), base[:, 16:].clone()
        linear.inv_rope_bf16_(o, pos, cs)
        got = o.reshape(T, 2, 4096)
        nd = int((got != ref).sum())
        assert torch.equal(o[..., :448], before[..., :448])
        assert torch.equal(base[:, 16:], pad)  # padding heads untouched
        assert nd == 0, (T, nd, float((got.float() - ref.float()).abs().max()))
        r2 = linear.inv_rope_bf16_ref(before, pos, cs).reshape(T, 2, 4096)
        frac = float((r2 != ref).float().mean())
        print(f"  T={T} strided={strided}: bit-identical to stock; torch (unfused) ref differs in {frac:.1e} (FMA contraction)")


def test_woa16_vs_stock_bf16():
    """Lossless wo_a (in-place inverse RoPE + strided bf16 group GEMMs) vs the stock BF16 branch of
    deep_gemm_fp8_o_proj and an fp64 reference: same error class (bf16 output rounding)."""
    require_deepgemm()
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    w, s = woa_weight()
    wb = dequant_fp8(w.cuda(), s.cuda()).to(torch.bfloat16)  # what stock's Emulation kernel keeps
    stock = torch.nn.Module()
    stock.weight = torch.nn.Parameter(wb, requires_grad=False)
    g = torch.Generator(device="cuda").manual_seed(7)
    cs = rope_cache(70000)
    for T, padded in ((300, True), (1000, False), (4096, False)):
        base = (torch.randn(T, 64 if padded else 16, 512, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        o = base[:, :16]
        pos = torch.randint(0, 70000, (T,), device="cuda", generator=g)
        z_st = deep_gemm_fp8_o_proj(o, pos, cs, stock, lambda z: z, n_groups=2, heads_per_group=8, nope_dim=448, rope_dim=64,
                                    o_lora_rank=1024, einsum_recipe=(1, 128, 128), tma_aligned_scales=False)
        ref = linear.inv_rope_bf16_ref(o, pos, cs).reshape(T, 2, 4096).double()  # stock's bf16 rounding of the rotation
        ref = torch.einsum("tgk,grk->tgr", ref, wb.double().view(2, 1024, 4096)).reshape(T, 2048)
        e_st = _err_vs(z_st.reshape(T, 2048), ref)
        for impl in ("deepgemm", "cublas"):
            oi = o.clone()
            z = linear.woa16(oi, pos, cs, wb, 2, 1024, impl=impl).reshape(T, 2048)
            e = _err_vs(z, ref)
            print(f"  T={T} padded={padded} {impl}: rel {e[0]:.3e} max {e[1]:.2f} ulp | stock rel {e_st[0]:.3e} max {e_st[1]:.2f} ulp")
            assert e[0] <= e_st[0] * 1.05 + 1e-6, (T, impl, e, e_st)


def test_o_proj16_patch_dispatch():
    """make_o_proj16: bf16 wo_a weight + >= MIN_M tokens -> woa16 (== direct call); FP8 weight / few tokens -> stock."""
    require_deepgemm()
    w, s = woa_weight()
    wb = dequant_fp8(w.cuda(), s.cuda()).to(torch.bfloat16)
    calls = []

    def stock(o, positions, cs, wo_a, wo_b, **kw):
        calls.append(o.shape[0])
        return wo_b(torch.zeros(o.shape[0], 2048, device="cuda", dtype=torch.bfloat16))

    fn = linear.make_o_proj16(stock)
    layer = types.SimpleNamespace(weight=wb)
    cs = rope_cache(4096)
    kw = dict(n_groups=2, heads_per_group=8, nope_dim=448, rope_dim=64, o_lora_rank=1024, einsum_recipe=(1, 128, 128),
              tma_aligned_scales=False)
    for T, expect_stock in ((linear.MIN_M, False), (linear.MIN_M - 1, True)):
        o = (torch.randn(T, 16, 512, device="cuda") * 0.5).to(torch.bfloat16)
        pos = torch.arange(T, device="cuda", dtype=torch.int64)
        o2 = o.clone()
        calls.clear()
        y = fn(o, pos, cs, layer, lambda z: z, **kw)
        assert (calls == [T]) == expect_stock, (T, calls)
        if not expect_stock:
            assert torch.equal(y, linear.woa16(o2, pos, cs, wb, 2, 1024).flatten(1))


# ===================================================================================== GPU tests: attention
def _attn_inputs(S: int, s_kv: int, topk: int, H: int = 16, HP: int = 64, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.zeros(S, HP, 512, device="cuda", dtype=torch.bfloat16)
    q[:, :H] = (torch.randn(S, H, 512, device="cuda", generator=g)).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, 512, device="cuda", generator=g)).to(torch.bfloat16)
    idx = torch.randint(0, s_kv, (S, 1, topk), device="cuda", generator=g, dtype=torch.int32)
    idx[:, :, -17:] = -1  # invalid tail entries
    tl = torch.randint(topk // 2, topk + 1, (S,), device="cuda", generator=g, dtype=torch.int32)
    sink = torch.full((HP,), float("-inf"), device="cuda")
    sink[:H] = torch.randn(H, device="cuda", generator=g)
    return q, kv, idx, tl, sink


def _stock_sparse():
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd


def require_small_heads():
    require_gpu()
    if not prefill_attn.available():
        skip("FlashMLA small-head extension unavailable (run scripts/build_flashmla_sm90.sh)")


def test_small_heads_vs_stock_padded():
    """16 heads on strided head slices == stock FlashMLA padded to 64 heads (+ fp32 reference)."""
    require_small_heads()
    scale = 512 ** -0.5
    for S, s_kv, topk, use_tl in ((7, 300, 128, True), (1024, 9000, 640, True), (513, 4096, 640, False)):
        q, kv, idx, tl, sink = _attn_inputs(S, s_kv, topk, seed=S)
        tl_arg = tl if use_tl else None
        out64 = torch.zeros_like(q)
        _, _, lse64 = _stock_sparse()(q=q, kv=kv, indices=idx, sm_scale=scale, attn_sink=sink, topk_length=tl_arg, out=out64)
        outp = torch.full_like(q, 7.0)  # padded buffer: heads >= 16 must stay untouched
        o16, _, lse16 = prefill_attn.sparse_prefill_fwd(q[:, :16], kv, idx, scale, 512, sink[:16], tl_arg, outp[:, :16])
        m = assert_close(outp[:, :16].float(), out64[:, :16].float(), cos_min=0.9999, rel_rmse_max=1e-2, name=f"S={S}")
        assert torch.all(outp[:, 16:] == 7.0), "small-head kernel wrote outside its head slice"
        assert torch.allclose(lse16, lse64[:, :16], atol=2e-3, rtol=1e-3)
        ref, _, lse_ref = prefill_attn.sparse_prefill_ref(q[:, :16], kv, idx, scale, 512, sink[:16], tl_arg)
        m2 = assert_close(outp[:, :16].float(), ref, cos_min=0.9999, rel_rmse_max=1e-2, name=f"S={S} ref")
        print(f"  S={S} topk={topk}: vs stock-64 {fmt_metrics(m)} | vs fp32 ref rel_rmse {m2['rel_rmse']:.2e}")


def test_small_heads_patch_on_stand_in_class():
    """install_flashmla_prefill: a class whose _forward_prefill calls the module-global flash_mla_sparse_fwd
    exactly like flashmla.py:368-376 gets the small-head kernel for n_local_heads=16 and the stock kernel for 64."""
    require_small_heads()
    mod = types.ModuleType("ds41h_test_fake_flashmla")
    exec(
        "class DeepseekV4FlashMLAAttention:\n"
        "    def __init__(self, h, hp):\n"
        "        self.n_local_heads, self.padded_heads = h, hp\n"
        "    def _forward_prefill(self, q, kv, idx, scale, sink, tl, out):\n"
        "        flash_mla_sparse_fwd(q=q, kv=kv, indices=idx.unsqueeze(1) if idx.dim() == 2 else idx, sm_scale=scale,\n"
        "                             attn_sink=sink, topk_length=tl, out=out)\n",
        mod.__dict__,
    )
    calls = []
    stock = _stock_sparse()

    def counting_stock(**kw):
        calls.append(kw["q"].shape[1])
        return stock(**kw)

    mod.flash_mla_sparse_fwd = counting_stock
    prefill_attn.install_flashmla_prefill(mod)
    prefill_attn.install_flashmla_prefill(mod)  # idempotent
    q, kv, idx, tl, sink = _attn_inputs(256, 2048, 640, seed=11)
    scale = 512 ** -0.5
    out_small, out_stock = torch.zeros_like(q), torch.zeros_like(q)
    mod.DeepseekV4FlashMLAAttention(16, 64)._forward_prefill(q, kv, idx, scale, sink, tl, out_small)
    mod.DeepseekV4FlashMLAAttention(64, 64)._forward_prefill(q, kv, idx, scale, sink, tl, out_stock)
    assert calls == [64], calls  # only the 64-head layer reached the stock kernel
    assert_close(out_small[:, :16].float(), out_stock[:, :16].float(), cos_min=0.9999, rel_rmse_max=1e-2, name="patched")


def realistic_indices(S: int, C: int, topk_c: int = 512, swa: int = 128, alpha: float = 0.3, seed: int = 0, P0: int | None = None):
    """Combined prefill indices like vLLM's combine_topk_swa_indices for one request: query p (absolute position
    P0 + p) gets top-`topk_c` of the causal compressed positions [0, (P0 + p + 1) // 4) by score = shared + alpha *
    per-query noise (alpha controls how much consecutive queries' sets overlap), then its `swa` window positions
    C + (p .. p + swa - 1) of the SWA region (kv workspace = [C compressed | S + swa - 1 window rows]).
    Returns (indices [S, 1, topk_c + swa] int32 with -1 padding, lens [S] int32, s_kv)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    P0 = 4 * C - S if P0 is None else P0
    shared = torch.randn(C, device="cuda", generator=g)
    idx = torch.full((S, topk_c + swa), -1, dtype=torch.int32, device="cuda")
    lens = torch.empty(S, dtype=torch.int32, device="cuda")
    for a in range(0, S, 1024):
        b = min(S, a + 1024)
        sc = shared[None, :] + alpha * torch.randn(b - a, C, device="cuda", generator=g)
        ncausal = ((P0 + torch.arange(a, b, device="cuda") + 1) // 4).clamp(max=C)
        sc = sc.masked_fill(torch.arange(C, device="cuda")[None, :] >= ncausal[:, None], float("-inf"))
        k = min(topk_c, C)
        top = sc.topk(k, dim=-1)
        sel = top.indices.int().masked_fill(top.values == float("-inf"), -1)
        nvalid = (sel >= 0).sum(-1, keepdim=True)
        sel = sel.sort(dim=-1, descending=True).values  # valid (>= 0) first
        j = torch.arange(topk_c + swa, device="cuda")[None, :]
        selp = torch.full((b - a, topk_c + swa), -1, dtype=torch.int32, device="cuda")
        selp[:, :k] = sel
        win = j - nvalid  # position inside the window part
        wval = (C + torch.arange(a, b, device="cuda")[:, None] + win).int()
        idx[a:b] = torch.where(j < nvalid, selp, torch.where(win < swa, wval, torch.full_like(wval, -1)))
        lens[a:b] = (nvalid[:, 0] + swa).int()
    return idx.unsqueeze(1), lens, C + S + swa - 1


def test_build_union_vs_torch():
    """build_union == per-group sorted unique valid indices + membership bits (torch reference)."""
    require_small_heads()
    g = torch.Generator(device="cuda").manual_seed(3)
    for S, topk, s_kv, use_len in ((37, 640, 3000, True), (64, 128, 70, False), (8, 640, 65536, True)):
        idx = torch.randint(-50, s_kv + 50, (S, 1, topk), device="cuda", generator=g, dtype=torch.int32)
        idx[:, :, :5] = idx[:, :, 5:10]  # duplicates inside a row
        lens = torch.randint(0, topk + 1, (S,), device="cuda", generator=g, dtype=torch.int32) if use_len else None
        uidx, umask, ulen = prefill_attn.build_union(idx, lens, s_kv)
        assert uidx.shape[1] % 128 == 0 and uidx.shape[0] == -(-S // 4)
        for b in range(uidx.shape[0]):
            sets = []
            for j in range(4):
                q = 4 * b + j
                if q >= S:
                    sets.append(set())
                    continue
                n = int(lens[q]) if lens is not None else topk
                row = idx[q, 0, :n].tolist()
                sets.append({v for v in row if 0 <= v < s_kv})
            union = sorted(set().union(*sets))
            n = int(ulen[b])
            assert n == len(union), (b, n, len(union))
            assert uidx[b, :n].tolist() == union, b
            exp = [sum(1 << j for j in range(4) if v in sets[j]) for v in union]
            assert umask[b, :n].tolist() == exp, b


def test_grouped_vs_small_heads_and_ref():
    """Grouped kernel (4 queries x 16 heads over the key-set union) == small-head kernel / fp32 reference."""
    require_small_heads()
    scale = 512 ** -0.5
    for S, C, alpha, with_sink in ((256, 1024, 0.3, True), (1024, 4096, 1.0, True), (512, 300, 0.3, False), (4096, 4096, 0.3, True)):
        idx, lens, s_kv = realistic_indices(S, C, alpha=alpha, seed=S)
        g = torch.Generator(device="cuda").manual_seed(S + 1)
        q = torch.randn(S, 16, 512, device="cuda", generator=g).to(torch.bfloat16)
        kv = torch.randn(s_kv, 1, 512, device="cuda", generator=g).to(torch.bfloat16)
        sink = torch.randn(64, device="cuda", generator=g) if with_sink else None
        out_g = torch.full_like(q, 7.0)
        prefill_attn.sparse_prefill_grouped(q, kv, idx, scale, sink, lens, out_g)
        out_s = torch.zeros_like(q)
        prefill_attn.sparse_prefill_fwd(q, kv, idx, scale, 512, sink[:16] if sink is not None else None, lens, out_s)
        m = assert_close(out_g.float(), out_s.float(), cos_min=0.9999, rel_rmse_max=1e-2, name=f"S={S} grouped vs small")
        uidx, umask, ulen = prefill_attn.build_union(idx, lens, s_kv)
        print(f"  S={S} C={C} alpha={alpha}: union mean {ulen.float().mean():.0f} (of 4 x {idx.shape[2]}) | "
              f"vs small-head {fmt_metrics(m)}")
        if S <= 512:
            ref, _, _ = prefill_attn.sparse_prefill_ref(q, kv, idx, scale, 512, sink[:16] if sink is not None else None, lens)
            assert_close(out_g.float(), ref, cos_min=0.9999, rel_rmse_max=1e-2, name=f"S={S} grouped vs ref")


def _sparse_ref64(q, kv, idx, scale, sink, lens, chunk: int = 128) -> torch.Tensor:
    """fp64 reference of the sparse prefill output (FlashMLA semantics incl. the sink), chunked over queries."""
    outs = []
    for a in range(0, q.shape[0], chunk):
        b = min(q.shape[0], a + chunk)
        o, _, _ = prefill_attn.sparse_prefill_ref(q[a:b].double(), kv.double(), idx[a:b], scale, 512,
                                                  sink.double() if sink is not None else None,
                                                  lens[a:b] if lens is not None else None)
        outs.append(o)
    return torch.cat(outs)


def test_attn_error_not_above_stock():
    """LOSSLESS gate for the prefill attention: vs an fp64 reference, the small-head (16) and grouped kernels' errors
    are not larger than stock's (vendored FlashMLA, 16 heads padded to 64): same bf16 Q/K/V, fp32 softmax, bf16 P,
    fp32 accumulation -- only the key-block order / partitioning differs. Also reports whether this build's 64-head
    kernel is bit-identical to the vendored one (same kernel source)."""
    require_small_heads()
    scale = 512 ** -0.5
    for S, C, alpha in ((1024, 4096, 0.3), (1024, 4096, 1.0), (512, 300, 0.3)):
        idx, lens, s_kv = realistic_indices(S, C, alpha=alpha, seed=S + 3)
        g = torch.Generator(device="cuda").manual_seed(S + 4)
        q64 = torch.zeros(S, 64, 512, device="cuda", dtype=torch.bfloat16)
        q64[:, :16] = (torch.randn(S, 16, 512, device="cuda", generator=g) * 2).to(torch.bfloat16)  # peaked softmax
        q16 = q64[:, :16].contiguous()
        kv = torch.randn(s_kv, 1, 512, device="cuda", generator=g).to(torch.bfloat16)
        sink = torch.full((64,), float("-inf"), device="cuda")
        sink[:16] = torch.randn(16, device="cuda", generator=g)
        ref = _sparse_ref64(q16, kv, idx, scale, sink[:16], lens)
        out64 = torch.zeros_like(q64)
        _stock_sparse()(q=q64, kv=kv, indices=idx, sm_scale=scale, attn_sink=sink, topk_length=lens, out=out64)
        o64b = torch.zeros_like(q64)
        prefill_attn.sparse_prefill_fwd(q64, kv, idx, scale, 512, sink, lens, o64b)
        same64 = torch.equal(o64b, out64)
        o16 = torch.zeros_like(q16)
        prefill_attn.sparse_prefill_fwd(q16, kv, idx, scale, 512, sink[:16], lens, o16)
        og = torch.zeros_like(q16)
        prefill_attn.sparse_prefill_grouped(q16, kv, idx, scale, sink, lens, og)
        e_st = _err_vs(out64[:, :16], ref)
        e16, eg = _err_vs(o16, ref), _err_vs(og, ref)
        rn = ref.to(torch.bfloat16)
        f = lambda o: float((o != rn).float().mean())  # noqa: E731
        print(f"  S={S} C={C} alpha={alpha}: rel err stock-64 {e_st[0]:.3e} ({f(out64[:, :16]):.3f} != rn(ref), max {e_st[1]:.1f} ulp) | "
              f"small-16 {e16[0]:.3e} ({f(o16):.3f}, {e16[1]:.1f}) | grouped {eg[0]:.3e} ({f(og):.3f}, {eg[1]:.1f}) | "
              f"ds41h 64-head build == vendored: {same64}")
        for name, e, o in (("small-16", e16, o16), ("grouped", eg, og)):
            assert e[0] <= e_st[0] * 1.05 + 1e-6, (S, alpha, name, e, e_st)
            assert f(o) <= f(out64[:, :16]) * 1.05 + 1e-3, (S, alpha, name, f(o), f(out64[:, :16]))


def test_grouped_dispatch_gating():
    """dispatch takes the grouped kernel only for contiguous [T, 16, 512] q / out with T % 4 == 0 (GROUPED on)."""
    require_small_heads()
    calls = []
    saved = prefill_attn.GROUPED, prefill_attn.GROUPED_MIN_T
    prefill_attn.GROUPED, prefill_attn.GROUPED_MIN_T = True, 4
    orig = prefill_attn.sparse_prefill_grouped
    prefill_attn.sparse_prefill_grouped = lambda *a, **k: calls.append("g") or orig(*a, **k)
    try:
        d = prefill_attn.make_dispatch(lambda **kw: calls.append("stock"))
        prefill_attn._STATE.layer = types.SimpleNamespace(n_local_heads=16)
        idx, lens, s_kv = realistic_indices(64, 256, seed=1)
        kv = torch.randn(s_kv, 1, 512, device="cuda").to(torch.bfloat16)
        for T, contiguous, expect in ((64, True, "g"), (62, True, "g"), (3, True, None), (64, False, None)):
            q = torch.randn(T, 16, 512, device="cuda").to(torch.bfloat16)
            out = torch.zeros_like(q) if contiguous else torch.zeros(T, 64, 512, device="cuda", dtype=torch.bfloat16)[:, :16]
            calls.clear()
            d(q=q, kv=kv, indices=idx[:T], sm_scale=0.04, attn_sink=None, topk_length=lens[:T], out=out)
            assert (calls == ["g"]) == (expect == "g"), (T, contiguous, calls)
            if expect == "g":  # incl. the small-head tail for T % 4 != 0
                ref = torch.zeros_like(q)
                prefill_attn.sparse_prefill_fwd(q, kv, idx[:T], 0.04, 512, None, lens[:T], ref)
                assert_close(out.float(), ref.float(), cos_min=0.9999, rel_rmse_max=1e-2, name=f"dispatch T={T}")
    finally:
        prefill_attn.GROUPED, prefill_attn.GROUPED_MIN_T = saved
        prefill_attn.sparse_prefill_grouped = orig
        prefill_attn._STATE.layer = None


# ===================================================================================== benchmarks
def _time_us(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


class _ClockSampler:
    """Background NVML sampler of the SM clock / power of the current device (mean over the window)."""

    def __init__(self, period_s: float = 0.005):
        import threading

        self.period, self.samples, self._stop = period_s, [], threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            idx = int(vis.split(",")[torch.cuda.current_device()]) if vis and vis[0].isdigit() else torch.cuda.current_device()
            self.h, self.nv = pynvml.nvmlDeviceGetHandleByIndex(idx), pynvml
        except Exception:  # noqa: BLE001
            self.h = None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM),
                                     self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            except Exception:  # noqa: BLE001
                return
            self._stop.wait(self.period)

    def __enter__(self):
        if self.h is not None:
            self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.h is not None:
            self._thr.join()

    def summary(self) -> str:
        if not self.samples:
            return "clk n/a"
        clk = sorted(c for c, _ in self.samples)
        pw = sum(p for _, p in self.samples) / len(self.samples)
        return f"clk p50 {clk[len(clk) // 2]:4d} MHz, {pw:5.0f} W"


SUSTAIN_S = float(os.environ.get("DS41H_BENCH_SUSTAIN_S", "0.5"))


def time_sustained(fn, seconds: float | None = None, warm_s: float | None = None) -> tuple[float, str]:
    """Mean us per call after `warm_s` of back-to-back calls, measured over `seconds` of back-to-back calls.

    At the 200 W cap the SM clock / HBM rate settle within tens of ms of continuous load (a 20-iteration burst
    reads ~1.5x faster than the in-server rate), so every benchmark here is timed in that steady state."""
    seconds = SUSTAIN_S if seconds is None else seconds
    warm_s = 0.6 * seconds if warm_s is None else warm_s
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(3):
        fn()
    b.record()
    torch.cuda.synchronize()
    t0 = max(a.elapsed_time(b) / 3e3, 1e-6)  # s per call (burst)
    nw, nm = max(3, int(warm_s / t0)), max(5, int(seconds / t0))
    for _ in range(nw):
        fn()
    with _ClockSampler() as cs:
        a.record()
        for _ in range(nm):
            fn()
        b.record()
        torch.cuda.synchronize()
    return a.elapsed_time(b) / nm * 1e3, cs.summary()


def bench_linears(layer: int, rank: int, ms=(8192, 2048)) -> None:
    """Stock Humming W8A16 vs ds41h DeepGEMM W8A8 (act_quant_128 + fp8_gemm_nt), sustained at the 200 W cap.
    (Humming MXFP8 x MXFP8 = impl "humming" is not runnable on humming 0.1.17 sm90: WGMMA input scales must be
    fp32, so it is not benchmarked.)"""
    from vllm.utils.deep_gemm import fp8_gemm_nt

    weights = dense_weights(layer, rank)
    weights["engram.wkv"] = engram_wkv(rank)
    print(f"dense linears, layer {layer} rank {rank} (sustained us per call; TFLOPS in brackets)")
    print(f"{'name':16s} {'N x K':>12s} {'M':>6s} {'stock W8A16':>16s} {'ds41h W8A8':>16s} {'= quant + gemm':>18s} "
          f"{'bf16 cuBLAS':>16s}  clocks (stock | ds41h)")
    tot = {}
    for name, (w, s) in weights.items():
        w, s = w.cuda(), s.cuda()
        n, k = w.shape
        stock = humming_stand_in(w, vllm_scale_layout(s), w8a8=False)
        fw, _ = linear.fold_weight(w, s)
        wb = dequant_fp8(w, s).to(torch.bfloat16)
        for M in ms:
            x = realistic_x(M, k, seed=M, device="cuda")
            fl = 2.0 * M * n * k
            t_st, c_st = time_sustained(lambda: stock(x))
            t_d, c_d = time_sustained(lambda: linear.gemm_w8a8(x, fw))
            xq, xs = linear.act_quant(x, fw.kp)
            y = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            t_q, _ = time_sustained(lambda: linear.act_quant(x, fw.kp), 0.2)
            t_g, _ = time_sustained(lambda: fp8_gemm_nt((xq, xs), (fw.w, fw.ws), y, is_deep_gemm_e8m0_used=False))
            t_b, _ = time_sustained(lambda: x @ wb.t(), 0.25)
            f = lambda t: f"{t:8.1f} [{fl / t / 1e6:5.0f}]"  # noqa: E731
            print(f"{name:16s} {n:5d}x{k:<6d} {M:6d} {f(t_st)} {f(t_d)} {t_q:7.1f}+{t_g:8.1f}  {f(t_b)}  {c_st} | {c_d}",
                  flush=True)
            if M == ms[0]:
                tot[name] = (t_st, t_d)
        del stock, fw, wb
        torch.cuda.empty_cache()
    st, d = sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values())
    print(f"sum over one layer's linears at M={ms[0]} (engram wkv counted once): stock {st:.0f} us, ds41h {d:.0f} us "
          f"({st / d:.2f}x)")


def bench_linear16(layer: int, rank: int, ms=(16384, 8192)) -> None:
    """LOSSLESS dense linears at the power cap (sustained): stock Humming W8A16 vs exact bf16 dequant + DeepGEMM bf16
    / cuBLAS bf16 (fp32 reductions), plus the lossy W8A8 path for reference."""
    from vllm.third_party.deep_gemm import bf16_gemm_nt

    weights = dense_weights(layer, rank)
    weights["engram.wkv"] = engram_wkv(rank)
    print(f"dense linears (lossless), layer {layer} rank {rank} (sustained us per call; TFLOPS in brackets)")
    print(f"{'name':16s} {'N x K':>12s} {'M':>6s} {'stock W8A16':>16s} {'dequant':>8s} {'DeepGEMM bf16':>16s} {'cuBLAS bf16':>16s} "
          f"{'W8A8 (lossy)':>16s}  clocks (stock | deepgemm | cublas)")
    tot: dict[int, list[float]] = {}
    for name, (w, s) in weights.items():
        w, s = w.cuda(), s.cuda()
        n, k = w.shape
        stock = humming_stand_in(w, vllm_scale_layout(s), w8a8=False)
        rw = linear.capture_raw(w, s)
        fw, _ = linear.fold_weight(w, s)
        wb = linear.dequant_bf16(rw)
        for M in ms:
            x = realistic_x(M, k, seed=M, device="cuda")
            y = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
            fl = 2.0 * M * n * k
            t_st, c_st = time_sustained(lambda: stock(x))
            t_dq, _ = time_sustained(lambda: linear.dequant_bf16(rw, wb), 0.2)
            t_dg, c_dg = time_sustained(lambda: bf16_gemm_nt(x, wb, y))
            t_cb, c_cb = time_sustained(lambda: linear.bf16_gemm(x, wb, y, impl="cublas"))
            t_8, _ = time_sustained(lambda: linear.gemm_w8a8(x, fw), 0.3)
            f = lambda t: f"{t:8.1f} [{fl / t / 1e6:5.0f}]"  # noqa: E731
            print(f"{name:16s} {n:5d}x{k:<6d} {M:6d} {f(t_st)} {t_dq:8.1f} {f(t_dg)} {f(t_cb)} {f(t_8)}  {c_st} | {c_dg} | {c_cb}",
                  flush=True)
            tot.setdefault(M, [0.0] * 5)
            for i, t in enumerate((t_st, t_dq + t_dg, t_dq + t_cb, t_8, t_dq)):
                tot[M][i] += t
        del stock, fw, wb, rw
        torch.cuda.empty_cache()
    for M, (st, dg, cb, w8, dq) in tot.items():
        print(f"sum over one layer's linears (+ engram wkv once) at M={M}: stock {st:.0f} us | lossless deepgemm {dg:.0f} "
              f"({st / dg:.2f}x) | lossless cublas {cb:.0f} ({st / cb:.2f}x) | (dequant {dq:.0f}) | lossy W8A8 {w8:.0f}")


def bench_woa16(layer: int, rank: int, ms=(16384, 8192)) -> None:
    """wo_a at the power cap (sustained): stock BF16 (Triton inverse RoPE copy + bmm) vs lossless woa16 (in-place
    RoPE + strided DeepGEMM / cuBLAS group GEMMs) vs lossy FP8 einsum."""
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    w, s = woa_weight(layer, rank)
    w, s = w.cuda(), s.cuda()
    wb = dequant_fp8(w, s).to(torch.bfloat16)
    stock = torch.nn.Module()
    stock.weight = torch.nn.Parameter(wb, requires_grad=False)
    fp8 = torch.nn.Module()
    fp8.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    fp8.weight_scale = torch.nn.Parameter(vllm_scale_layout(s), requires_grad=False)
    fp8.bmm_batch_size = 2
    linear.convert_woa(fp8)
    print("wo_a (inverse RoPE + 2 groups x 1024 x 4096), sustained")
    for M in ms:
        o = torch.randn(M, 16, 512, device="cuda").to(torch.bfloat16)
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        cs = rope_cache(M)
        run = lambda m: deep_gemm_fp8_o_proj(o, pos, cs, m, lambda z: z, n_groups=2, heads_per_group=8, nope_dim=448,  # noqa: E731
                                             rope_dim=64, o_lora_rank=1024, einsum_recipe=(1, 128, 128), tma_aligned_scales=False)
        fl = 2.0 * M * 2048 * 4096
        t_s, c_s = time_sustained(lambda: run(stock))
        t_r, _ = time_sustained(lambda: linear.inv_rope_bf16_(o, pos, cs), 0.2)  # (rotates o repeatedly: timing only)
        t_d, c_d = time_sustained(lambda: linear.woa16(o, pos, cs, wb, 2, 1024, impl="deepgemm"))
        t_c, c_c = time_sustained(lambda: linear.woa16(o, pos, cs, wb, 2, 1024, impl="cublas"))
        t_f, _ = time_sustained(lambda: run(fp8), 0.3)
        f = lambda t: f"{t:8.1f} us [{fl / t / 1e6:4.0f} TF]"  # noqa: E731
        print(f"  M={M:5d}: stock BF16 {f(t_s)} ({c_s}) | lossless deepgemm {f(t_d)} ({c_d}) cublas {f(t_c)} ({c_c}) "
              f"[in-place RoPE {t_r:.1f} us] | lossy FP8 {f(t_f)} -> {t_s / min(t_d, t_c):4.2f}x", flush=True)


def bench_ropeq(ms=(8188, 2048)) -> None:
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    print("inverse RoPE + FP8 quant of the attention output (16 heads x 512), sustained")
    cs = rope_cache(70000)
    for M in ms:
        o = torch.randn(M, 16, 512, device="cuda").to(torch.bfloat16)
        pos = torch.randint(0, 70000, (M,), device="cuda")
        t_s, c_s = time_sustained(lambda: fused_inv_rope_fp8_quant(o, pos, cs, n_groups=2, heads_per_group=8), 0.3)
        t_d, c_d = time_sustained(lambda: linear.inv_rope_fp8_quant(o, pos, cs, 2), 0.3)
        nb = M * 16 * 512 * 3
        print(f"  M={M}: stock Triton {t_s:7.1f} us ({nb / t_s / 1e3:5.0f} GB/s, {c_s})  ds41h {t_d:7.1f} us ({nb / t_d / 1e3:5.0f} GB/s, "
              f"{c_d})  {t_s / t_d:4.2f}x", flush=True)


def bench_woa(layer: int, rank: int, ms=(8192, 2048)) -> None:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    w, s = woa_weight(layer, rank)
    w, s = w.cuda(), s.cuda()
    stock = torch.nn.Module()
    stock.weight = torch.nn.Parameter(dequant_fp8(w, s).to(torch.bfloat16), requires_grad=False)
    fp8 = torch.nn.Module()
    fp8.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    fp8.weight_scale = torch.nn.Parameter(vllm_scale_layout(s), requires_grad=False)
    fp8.bmm_batch_size = 2
    linear.convert_woa(fp8)
    print("wo_a (inverse RoPE + grouped projection, 2 groups x 1024 x 4096), sustained")
    for M in ms:
        o = torch.randn(M, 16, 512, device="cuda").to(torch.bfloat16)
        pos = torch.arange(M, device="cuda", dtype=torch.int64)
        cs = rope_cache(M)
        run = lambda m: deep_gemm_fp8_o_proj(o, pos, cs, m, lambda z: z, n_groups=2, heads_per_group=8, nope_dim=448,  # noqa: E731
                                             rope_dim=64, o_lora_rank=1024, einsum_recipe=(1, 128, 128), tma_aligned_scales=False)
        fl = 2.0 * M * 2048 * 4096
        (t_s, c_s), (t_f, c_f) = time_sustained(lambda: run(stock)), time_sustained(lambda: run(fp8))
        print(f"  M={M:5d}: stock BF16 bmm {t_s:8.1f} us [{fl / t_s / 1e6:5.0f} TF] ({c_s})   FP8 einsum {t_f:8.1f} us "
              f"[{fl / t_f / 1e6:5.0f} TF] ({c_f})  {t_s / t_f:4.2f}x", flush=True)


def bench_attn_grouped(cases=((8192, 4096, 0.3), (8192, 4096, 1.0), (8192, 16384, 0.3), (8192, 4096, 3.0))) -> None:
    """Realistic index sets (realistic_indices: causal top-512 compressed + 128-window, alpha = per-query score noise):
    stock FlashMLA (64 padded heads) vs small-head (16) vs grouped (4 queries x 16 heads over the union, incl. the
    union build), sustained."""
    print("sparse prefill with realistic index overlap (us, sustained; useful TF = 4 * S * 16 * len * 512)")
    scale = 512 ** -0.5
    for S, C, alpha in cases:
        idx, lens, s_kv = realistic_indices(S, C, alpha=alpha, seed=2)
        g = torch.Generator(device="cuda").manual_seed(5)
        q64 = torch.zeros(S, 64, 512, device="cuda", dtype=torch.bfloat16)
        q64[:, :16] = torch.randn(S, 16, 512, device="cuda", generator=g).to(torch.bfloat16)
        q16 = q64[:, :16].contiguous()
        kv = torch.randn(s_kv, 1, 512, device="cuda", generator=g).to(torch.bfloat16)
        sink = torch.randn(64, device="cuda", generator=g)
        out64, out16 = torch.zeros_like(q64), torch.zeros_like(q16)
        fl = 4.0 * 16 * 512 * float(lens.float().sum())
        t_s, c_s = time_sustained(lambda: _stock_sparse()(q=q64, kv=kv, indices=idx, sm_scale=scale, attn_sink=sink, topk_length=lens,
                                                          out=out64))
        t_n, c_n = time_sustained(lambda: prefill_attn.sparse_prefill_fwd(q16, kv, idx, scale, 512, sink[:16], lens, out16))
        t_g, c_g = time_sustained(lambda: prefill_attn.sparse_prefill_grouped(q16, kv, idx, scale, sink, lens, out16))
        t_u, _ = time_sustained(lambda: prefill_attn.build_union(idx, lens, s_kv), 0.2)
        ulen = prefill_attn.build_union(idx, lens, s_kv)[2].float().mean()
        print(f"  S={S} C={C} alpha={alpha}: union {ulen:.0f} keys/4 queries | stock-64 {t_s:8.1f} [{fl / t_s / 1e6:4.0f} TF] ({c_s}) "
              f"small-16 {t_n:8.1f} [{fl / t_n / 1e6:4.0f}] ({c_n}) grouped {t_g:8.1f} [{fl / t_g / 1e6:4.0f}] ({c_g}; union build "
              f"{t_u:.0f}) -> {t_n / t_g:4.2f}x vs small, {t_s / t_g:4.2f}x vs stock", flush=True)


def bench_attn(cases=((8192, 16384, 640), (8192, 16384, 512), (8192, 2048, 128), (2048, 16384, 640))) -> None:
    """Sparse prefill, stock (vendored FlashMLA, 64 padded heads) vs small-head kernel (16 heads), sustained.
    topk 640 = 512 indexer + 128 SWA keys (the CSA layers), 128 = SWA-only layers."""
    print("sparse prefill (16 real heads; stock pads to 64) d_qk=512, sustained")
    scale = 512 ** -0.5
    for S, s_kv, topk in cases:
        q, kv, idx, tl, sink = _attn_inputs(S, s_kv, topk, seed=1)
        tl = torch.full_like(tl, topk)  # full top-k (prefill rows past 640 tokens use all slots)
        out = torch.zeros_like(q)
        fl = 4.0 * S * 16 * topk * 512  # QK + PV useful flops
        t_s, c_s = time_sustained(lambda: _stock_sparse()(q=q, kv=kv, indices=idx, sm_scale=scale, attn_sink=sink, topk_length=tl,
                                                          out=out))
        t_n, c_n = float("nan"), ""
        if prefill_attn.available():
            t_n, c_n = time_sustained(lambda: prefill_attn.sparse_prefill_fwd(q[:, :16], kv, idx, scale, 512, sink[:16], tl, out[:, :16]))
        print(f"  S={S:5d} s_kv={s_kv:6d} topk={topk:4d}: stock-64 {t_s:9.1f} us [{fl / t_s / 1e6:5.0f} useful TF] ({c_s})   "
              f"small-16 {t_n:9.1f} us [{fl / t_n / 1e6:5.0f} TF] ({c_n})  speedup {t_s / t_n:4.2f}x", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--what", default="linear,woa,attn")
    args, rest = ap.parse_known_args()
    if args.bench:
        require_gpu()
        what = args.what.split(",")
        if "linear" in what:
            bench_linears(args.layer, args.rank)
        if "linear16" in what:
            bench_linear16(args.layer, args.rank)
        if "woa16" in what:
            bench_woa16(args.layer, args.rank)
        if "woa" in what:
            bench_woa(args.layer, args.rank)
        if "attn" in what:
            bench_attn()
        if "grouped" in what:
            bench_attn_grouped()
        if "ropeq" in what:
            bench_ropeq()
    else:
        sys.exit(run_module_tests(globals(), rest))
