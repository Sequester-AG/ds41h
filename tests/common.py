"""Shared helpers for ds41h unit tests (import as ``from common import ...`` from tests/).

Weights: ``load_layer`` (dequantized float32 LayerWeights for the reference, unsharded or one TP
rank), ``load_layer_raw`` (checkpoint-format TP-rank tensors exactly as ds41h kernels consume them,
docs/DESIGN.md "Weight layouts"), ``raw_experts``.
Inputs: ``realistic_stream`` (residual stream built from real embedding rows), ``realistic_normed``
(RMSNorm outputs with a layer's real norm weight), ``random_pre_mix``.
Metrics: ``compare`` / ``assert_close`` (max abs/rel, rel-RMSE, cosine), ``kl_top1``, ``topk_overlap``.
Bench: ``bench`` (CUDA events or wall clock), ``fmt_bw``.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from ds41h import ref  # noqa: E402
from ds41h.ref import CheckpointReader, LayerWeights, ModelConfig, RawExperts  # noqa: E402

try:  # pytest is optional: the test files also run standalone
    import pytest

    SkipTest = pytest.skip.Exception
except ImportError:  # pragma: no cover
    import unittest

    pytest = None
    SkipTest = unittest.SkipTest

D = 5120


def skip(msg: str):
    raise SkipTest(msg)


def has_cuda() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" and torch.cuda.is_available()


def require_cuda():
    if not has_cuda():
        skip("CUDA not available")


def device() -> str:
    return "cuda" if has_cuda() else "cpu"


@lru_cache(maxsize=1)
def reader() -> CheckpointReader:
    if not Path(ref.MODEL_DIR, "model.safetensors.index.json").exists():
        skip(f"model not found at {ref.MODEL_DIR} (set DS41H_MODEL_DIR)")
    return CheckpointReader(ref.MODEL_DIR)


@lru_cache(maxsize=1)
def cfg() -> ModelConfig:
    return ModelConfig.from_hf(reader().config)


def free():
    gc.collect()
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        torch.cuda.empty_cache()


# ===================================================================================== weights


def load_layer(layer_id: int, rank: int | None = None, device: str = "cpu", experts: str = "lazy",
               mtp: int | None = None) -> LayerWeights:
    """Dequantized float32 weights of one layer for the reference (rank None = unsharded)."""
    return ref.load_layer_weights(reader(), layer_id, rank, device=device, cfg=cfg(), mtp=mtp, experts=experts)


def load_layer_raw(layer_id: int, rank: int, device: str = "cpu", experts: bool | list[int] = False,
                   mtp: int | None = None) -> dict[str, torch.Tensor]:
    """Checkpoint-format tensors of TP rank `rank` (stock vLLM 0.32.0 sharding, ds41h layouts).

    Dense FP8: e4m3 [N, K] + ue8m0 uint8 block scales [N/32, K/32] (checkpoint form; use
    ref.vllm_scale_layout for stock vLLM's row-expanded form). Keys:
      wqa_wkv/wqa_wkv_s  [1792, 5120]/[56, 160] (rows 0..1279 = wq_a, 1280.. = wkv; replicated)
      wq_b/wq_b_s        [8192, 1280]/[256, 40]   (heads 16r..16r+15)
      wo_a/wo_a_s        [2048, 4096]/[64, 128]   (groups 2r, 2r+1)
      wo_b/wo_b_s        [5120, 2048]/[160, 64]   (K-split)
      sh_w13/sh_w13_s    [1152, 5120]/[36, 160]   (rows 0..575 = w1, 576.. = w3)
      sh_w2/sh_w2_s      [5120, 576]/[160, 18]
      gate (bf16 [E, 5120]), gate_bias (f32), attn_sink (f32 [16]), norms (bf16), hc_* (f32),
      compressor/indexer tensors (replicated) where present,
      w13/s13/w2/s2      routed experts, raw layout [E, 1152, 2560]/[E, 1152, 160]/[E, 5120, 288]/
                         [E, 5120, 18] uint8 (only if experts=True or a list of expert ids).
    """
    rd = reader()
    pre = f"mtp.{mtp}." if mtp is not None else f"layers.{layer_id}."
    g = lambda n: rd.get_tp(pre + n, rank)  # noqa: E731
    u8 = lambda t: ref.quant.as_u8(t)  # noqa: E731
    out: dict[str, torch.Tensor] = {}
    out["wqa_wkv"] = torch.cat([g("attn.wq_a.weight"), g("attn.wkv.weight")], 0)
    out["wqa_wkv_s"] = torch.cat([u8(g("attn.wq_a.scale")), u8(g("attn.wkv.scale"))], 0)
    for k, n in (("wq_b", "attn.wq_b"), ("wo_a", "attn.wo_a"), ("wo_b", "attn.wo_b"), ("sh_w2", "ffn.shared_experts.w2")):
        out[k], out[k + "_s"] = g(n + ".weight"), u8(g(n + ".scale"))
    out["sh_w13"] = torch.cat([g("ffn.shared_experts.w1.weight"), g("ffn.shared_experts.w3.weight")], 0)
    out["sh_w13_s"] = torch.cat([u8(g("ffn.shared_experts.w1.scale")), u8(g("ffn.shared_experts.w3.scale"))], 0)
    for k, n in (("gate", "ffn.gate.weight"), ("gate_bias", "ffn.gate.bias"), ("attn_sink", "attn.attn_sink"),
                 ("attn_norm", "attn_norm.weight"), ("ffn_norm", "ffn_norm.weight"), ("q_norm", "attn.q_norm.weight"),
                 ("kv_norm", "attn.kv_norm.weight"), ("hc_attn_fn", "hc_attn_fn"), ("hc_attn_scale", "hc_attn_scale"),
                 ("hc_attn_base", "hc_attn_base"), ("hc_ffn_fn", "hc_ffn_fn"), ("hc_ffn_scale", "hc_ffn_scale"),
                 ("hc_ffn_base", "hc_ffn_base")):
        out[k] = g(n)
    for n in rd.names(pre + "attn.compressor.") + rd.names(pre + "attn.indexer."):
        out[n[len(pre) :]] = rd.get(n)
    if experts is not False:
        ids = list(range(out["gate"].shape[0])) if experts is True else list(experts)
        out.update(_stack_experts(layer_id, rank, ids, device, mtp))
    return {k: v.to(device) for k, v in out.items()}


def _stack_experts(layer_id: int, rank: int, ids: list[int], device: str, mtp: int | None = None):
    rd = reader()
    pre = f"mtp.{mtp}." if mtp is not None else f"layers.{layer_id}."
    first = f"{pre}ffn.experts.{ids[0]}."
    sh = lambda n: ref.tp_shape(first + n, rd.info(first + n).shape, rank)  # noqa: E731
    i2, k2 = sh("w1.weight")
    w13 = torch.empty(len(ids), 2 * i2, k2, dtype=torch.uint8, device=device)
    s13 = torch.empty(len(ids), 2 * i2, sh("w1.scale")[1], dtype=torch.uint8, device=device)
    w2 = torch.empty(len(ids), *sh("w2.weight"), dtype=torch.uint8, device=device)
    s2 = torch.empty(len(ids), *sh("w2.scale"), dtype=torch.uint8, device=device)
    for j, e in enumerate(ids):
        p = f"{pre}ffn.experts.{e}."
        u = lambda n: ref.quant.as_u8(rd.get_tp(p + n, rank)).to(device)  # noqa: E731
        w13[j, :i2], w13[j, i2:] = u("w1.weight"), u("w3.weight")
        s13[j, :i2], s13[j, i2:] = u("w1.scale"), u("w3.scale")
        w2[j], s2[j] = u("w2.weight"), u("w2.scale")
    return {"w13": w13, "s13": s13, "w2": w2, "s2": s2, "expert_ids": torch.tensor(ids)}


def raw_experts(layer_id: int, rank: int, ids: list[int], device: str = "cpu") -> RawExperts:
    """RawExperts over a subset of experts; index with the position in `ids`."""
    t = _stack_experts(layer_id, rank, ids, device)
    return RawExperts(t["w13"], t["s13"], t["w2"], t["s2"])


# ===================================================================================== inputs


def embed_rows(token_ids: torch.Tensor) -> torch.Tensor:
    """Rows of the (bf16) embedding table as float32 [T, D]."""
    return reader().gather_rows("embed.weight", token_ids.long().cpu()).float()


def realistic_stream(T: int, seed: int = 0, scale: float = 1.0, copy_noise: float = 0.25,
                     device: str = "cpu") -> torch.Tensor:
    """Residual stream X [T, 4, D] (bf16-representable float32) built like the real one: embedding
    rows of random token ids broadcast to the 4 hc copies (model.py :1261), plus copy-specific
    gaussian noise of `copy_noise` x the row RMS (deeper layers' copies diverge), times `scale`."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 120000, (T,), generator=g)
    e = embed_rows(ids)
    rms = e.square().mean(-1, keepdim=True).sqrt()
    noise = torch.randn(T, 4, D, generator=g) * rms.unsqueeze(1) * copy_noise
    X = (e.unsqueeze(1) + noise) * scale
    return ref.quant.bf16_round(X).to(device)


def realistic_normed(T: int, norm_w: torch.Tensor, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """RMSNorm outputs with a real norm weight: what a sublayer actually receives, [T, D]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, norm_w.shape[-1], generator=g)
    x = x * (1 + 3 * (torch.rand(norm_w.shape[-1], generator=g) < 0.01))  # a few outlier channels
    return ref.quant.bf16_round(ref.rmsnorm(x, norm_w.cpu().float(), 1e-20)).to(device)


def random_pre_mix(T: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """pre-mix like a real one: sigmoid(...) + 1e-6, [T, 4] float32."""
    g = torch.Generator().manual_seed(seed)
    return (torch.sigmoid(torch.randn(T, 4, generator=g)) + 1e-6).to(device)


def decode_positions(p0: int, T: int, device: str = "cpu") -> torch.Tensor:
    return torch.arange(p0, p0 + T, device=device)


# ===================================================================================== metrics


def compare(test: torch.Tensor, refv: torch.Tensor) -> dict:
    a, b = test.detach().double().flatten(), refv.detach().double().flatten()
    d = (a - b).abs()
    finite = torch.isfinite(a).all() and torch.isfinite(b).all()
    denom = b.abs().clamp_min(1e-30)
    return {
        "max_abs": float(d.max()) if d.numel() else 0.0,
        "max_rel": float((d / denom).max()) if d.numel() else 0.0,
        "mean_abs": float(d.mean()) if d.numel() else 0.0,
        "rel_rmse": float(d.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)),
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)) if a.numel() else 1.0,
        "n_nan": int((~torch.isfinite(a)).sum()),
        "finite": bool(finite),
    }


def assert_close(test: torch.Tensor, refv: torch.Tensor, *, cos_min: float = 0.999, rel_rmse_max: float = 1e-2,
                 max_abs: float | None = None, name: str = "") -> dict:
    assert test.shape == refv.shape, f"{name}: shape {tuple(test.shape)} != {tuple(refv.shape)}"
    m = compare(test, refv)
    msg = f"{name}: {fmt_metrics(m)}"
    assert m["finite"], msg
    assert m["cos"] >= cos_min, msg
    assert m["rel_rmse"] <= rel_rmse_max, msg
    if max_abs is not None:
        assert m["max_abs"] <= max_abs, msg
    return m


def fmt_metrics(m: dict) -> str:
    return (f"max_abs={m['max_abs']:.3e} max_rel={m['max_rel']:.3e} rel_rmse={m['rel_rmse']:.3e} "
            f"cos={m['cos']:.6f} nan={m['n_nan']}")


def kl_top1(ref_logits: torch.Tensor, test_logits: torch.Tensor, temperature: float = 1.0) -> dict:
    """Mean/max KL(ref || test) over rows of full-vocab logits and top-1 agreement."""
    lp = torch.log_softmax(ref_logits.double() / temperature, -1)
    lq = torch.log_softmax(test_logits.double() / temperature, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (ref_logits.argmax(-1) == test_logits.argmax(-1)).double().mean()
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(top1)}


def topk_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row overlap |A & B| / |A| of index sets (entries < 0 ignored)."""
    vals = []
    for ra, rb in zip(a.tolist(), b.tolist(), strict=True):
        sa, sb = {x for x in ra if x >= 0}, {x for x in rb if x >= 0}
        vals.append(len(sa & sb) / max(1, len(sa)) if sa or sb else 1.0)
    return sum(vals) / max(1, len(vals))


# ===================================================================================== bench


def bench(fn, warmup: int = 5, iters: int = 50) -> float:
    """Median milliseconds per call (CUDA events when on GPU, else wall clock)."""
    for _ in range(warmup):
        fn()
    times = []
    if has_cuda():
        torch.cuda.synchronize()
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def fmt_bw(name: str, ms: float, nbytes: float | None = None, flops: float | None = None) -> str:
    s = f"{name:<40s} {ms * 1e3:10.1f} us"
    if nbytes:
        s += f"  {nbytes / ms / 1e6:9.1f} GB/s"
    if flops:
        s += f"  {flops / ms / 1e9:9.2f} TFLOPS"
    return s


def run_module_tests(module_globals: dict, argv: list[str] | None = None) -> int:
    """Minimal runner so test files work without pytest: runs every test_* function."""
    argv = sys.argv[1:] if argv is None else argv
    sel = [a for a in argv if not a.startswith("-")]
    names = [n for n in module_globals if n.startswith("test_") and callable(module_globals[n])]
    if sel:
        names = [n for n in names if any(s in n for s in sel)]
    fails = skips = 0
    for n in names:
        t = time.perf_counter()
        try:
            module_globals[n]()
            status = "PASS"
        except SkipTest as e:  # type: ignore[misc]
            status = f"SKIP ({e})"
            skips += 1
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            status = f"FAIL ({type(e).__name__}: {e})"
            fails += 1
        print(f"{n:<45s} {status}  [{time.perf_counter() - t:.1f}s]", flush=True)
        free()
    print(f"{len(names) - fails - skips} passed, {skips} skipped, {fails} failed", flush=True)
    return 1 if fails else 0


__all__ = [n for n in dir() if not n.startswith("_")] + ["math"]
