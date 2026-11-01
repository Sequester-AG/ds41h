"""GPU unit test + micro-benchmark for the ds41h decoder layer kernel (ds41h/kernels/decoder.cu).

  python -m pytest tests/test_decoder.py -v           # single GPU (solo all-reduce) + 4-GPU if free
  torchrun --nproc-per-node 4 tests/test_decoder.py --dist          # full reduced outputs
  python tests/test_decoder.py --bench [--layer 3] [--m 1 6 8]      # single GPU timing
  torchrun --nproc-per-node 4 tests/test_decoder.py --bench --dist  # 4-GPU timing (NVLink IPC)
  python tests/test_decoder.py --bench-routed [--layers 40 --experts 30 --m 6 8 --rpe 8]
      # routed-expert GEMM loop only (HMMA vs RS-wgmma inner loops, synthetic weights), GB/s
  DS41H_DECODER_DEFINES=ROUTED_MMA=1,DENSE_MMA=1 python -m pytest tests/test_decoder.py -v
      # the same layer tests on the RS-wgmma variants (reference simulates their e4m3 activations)

Env: DS41H_MODEL_DIR (default $DS41H_HOME/model), DS41H_TEST_LAYER (default 3),
DS41H_TEST_RANK (TP rank whose weights the single-GPU test uses, default 1).

Reference: a pure-torch float32 model of exactly what stock vLLM 0.32.0 computes on sm90 for the
post-attention half of DeepseekV4DecoderLayer (bf16 rounding at the same points: o_rot, z,
wo_b output, all-reduce, residual, collapse, ffn input, gate/up, SwiGLU, down, combine), built
on dequantized checkpoint weights (ds41h.ref.weights.dequant_fp8/fp4 when available). The mHC
pieces are cross-checked against ds41h.ref.layer (official model.py semantics) when importable.
Routing near-ties (6th/7th key within 1e-3) may differ between kernel and reference fp32 sums;
the MoE reference then uses the kernel's expert ids (weights recomputed from reference scores).
RS-wgmma variants (ROUTED_MMA / DENSE_MMA = 1): the reference quantizes the GEMM activations
exactly like the kernel (e4m3: o_rot per (token, 32) ue8m0, z with its f16 staging exponent + 6,
x / xs, SwiGLU outputs * 4) wherever the kernel takes the wgmma path (<= 16 tokens per group /
expert entry), so the tolerances stay those of the HMMA path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:  # script mode (--bench / --dist under torchrun) must not need pytest

    class _Mark:
        @staticmethod
        def skipif(*args, **kwargs):
            return lambda fn: fn

        @staticmethod
        def parametrize(*args, **kwargs):
            return lambda fn: fn

    class pytest:  # noqa: N801 - minimal stand-in for the decorators used at import time
        mark = _Mark

        @staticmethod
        def fixture(*args, **kwargs):
            return lambda fn: fn

        @staticmethod
        def skip(reason: str):
            raise RuntimeError(f"skip: {reason}")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
LAYER = int(os.environ.get("DS41H_TEST_LAYER", "3"))
TEST_RANK = int(os.environ.get("DS41H_TEST_RANK", "1"))
HID, HC, INTER, EXPERTS, TOPK = 5120, 4, 576, 384, 6
EXPERT_BYTES = 2 * INTER * HID // 2 + 2 * INTER * HID // 32 + HID * INTER // 2 + HID * INTER // 32  # per rank


def _cuda_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (9, 0)


# ------------------------------------------------------------------------------- dequant helpers
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _e8m0(s: torch.Tensor) -> torch.Tensor:
    return torch.exp2(s.view(torch.uint8).to(torch.float32) - 127.0)


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] x ue8m0 [N/32, K/32] -> fp32."""
    try:
        from ds41h.ref.weights import dequant_fp8 as ref_fp8

        return ref_fp8(w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu))
    except ImportError:
        wf = w.view(torch.float8_e4m3fn).float()
        sf = _e8m0(s).repeat_interleave(32, 0)[: wf.shape[0]].repeat_interleave(32, 1)[:, : wf.shape[1]]
        return wf * sf


def dequant_fp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 [N, K/2] (low nibble = even k) x ue8m0 [N, K/32] -> fp32 [N, K]."""
    try:
        from ds41h.ref.weights import dequant_fp4 as ref_fp4

        return ref_fp4(w.view(torch.uint8), s.view(torch.float8_e8m0fnu))
    except ImportError:
        b = w.view(torch.uint8).long()
        table = torch.tensor(_FP4, device=w.device)
        v = torch.stack([table[b & 15], table[b >> 4]], dim=-1).flatten(-2)
        return v * _e8m0(s).repeat_interleave(32, -1)


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16, keep the working dtype (fp32, or fp64 for the exact reference)."""
    return x.to(torch.bfloat16).to(x.dtype if x.dtype == torch.float64 else torch.float32)


def sinkhorn(logits: torch.Tensor, iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    c = torch.softmax(logits, dim=-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return c


# ------------------------------------------------------------------------------- wgmma activation quant
WG_MAX_TOKENS = 16  # the kernel's wgmma paths take <= 2 x 8 tokens per group / expert entry
WG_GROUP = 32  # decoder.cu OTOK: dense GEMM token groups (wo_b / shared expert dispatch per group)
COARSE_M = 8  # decoder.cu COARSE_M: coarse routed path (e4m3 x / h for every expert) for m <= 8


def wg_dense_rows(m: int, device=None) -> torch.Tensor:
    """Per-token mask of the dense GEMMs (wo_b, shared w1/w3/w2) that take the wgmma path: the
    kernel dispatches per 32-token group and uses wgmma for groups of <= WG_MAX_TOKENS tokens
    (identical to m <= 16 for the default MAXM = 32; differs for MAXM > 32, e.g. m = 40)."""
    t = torch.arange(m, device=device)
    count = torch.clamp(m - (t // WG_GROUP) * WG_GROUP, max=WG_GROUP)
    return count <= WG_MAX_TOKENS


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3 value (round to nearest even, satfinite), as cvt.rn.satfinite.e4m3x2.f32."""
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def _ceil_exp(v: torch.Tensor) -> torch.Tensor:
    """Smallest e with 2^e >= v (v > 0), exactly (decoder quant_exp)."""
    m, e = torch.frexp(v)
    return torch.where(m == 0.5, e - 1, e)


def quant_ue8m0_32(x: torch.Tensor) -> torch.Tensor:
    """Official act_quant rule per 32 values (decoder.cu wo_a wgmma path): scale 2^ceil(log2(max(amax,
    1e-4) / 448)), e4m3 of x / scale, dequantized."""
    xb = x.float().unflatten(-1, (-1, 32))
    v = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    s = torch.ldexp(torch.ones_like(v), _ceil_exp(v))
    return (e4m3(xb / s) * s).flatten(-2)


def quant_z(z: torch.Tensor) -> torch.Tensor:
    """wo_b wgmma input: per 32 block the f16 staging exponent e (amax * 2^-e in [2^13, 2^14),
    -100 for zeros, clamped to +-100) plus 6, e4m3 of z * 2^-(e+6), dequantized."""
    zb = z.float().unflatten(-1, (-1, 32))
    amax = zb.abs().amax(-1, keepdim=True)
    _, ex = torch.frexp(amax)
    e = torch.where(amax > 0, ex - 14, torch.full_like(ex, -100)).clamp(-100, 100) + 6
    s = torch.ldexp(torch.ones_like(amax), e)
    return (e4m3(zb / s) * s).flatten(-2)


def quant_static(x: torch.Tensor, scale: float) -> torch.Tensor:
    return e4m3(x.float() / scale) * scale


def rope_table(n: int, device, theta: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / theta ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64)
    f = torch.arange(n, dtype=torch.float64)[:, None] * inv[None]
    return torch.cat([f.cos(), f.sin()], dim=-1).float().to(device)


# ------------------------------------------------------------------------------- weights
class RefWeights:
    """Dequantized fp32 weights of one TP rank (reference side)."""

    def __init__(self, raw: dict, experts: tuple):
        self.woa = dequant_fp8(raw["woa"], raw["woa_s"])  # [2048, 4096]
        self.wob = dequant_fp8(raw["wob"], raw["wob_s"])  # [5120, 2048]
        self.w1 = dequant_fp8(raw["w1"], raw["s1"])
        self.w3 = dequant_fp8(raw["w3"], raw["s3"])
        self.w2 = dequant_fp8(raw["w2"], raw["s2"])
        self.gate = raw["gate"].float()
        self.gate_b, self.fn, self.hc_scale, self.hc_base = raw["gate_b"], raw["fn"], raw["hc_scale"], raw["hc_base"]
        self.norm = raw["norm"].float()
        self.experts = experts  # raw (w13, s13, w2, s2), dequantized per expert on demand

    def expert(self, e: int):
        w13, s13, w2, s2 = self.experts
        a = dequant_fp4(w13[e], s13[e]).to(self.dtype)
        return a[:INTER], a[INTER:], dequant_fp4(w2[e], s2[e]).to(self.dtype)

    dtype = torch.float32

    def to64(self) -> "RefWeights":
        """The same weights in float64 (exact GEMM reference: only the bf16 rounding points remain), cached."""
        if getattr(self, "_f64", None) is None:
            w = object.__new__(RefWeights)
            for k, v in vars(self).items():
                setattr(w, k, v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            w.dtype = torch.float64
            self._f64 = w
        return self._f64


def checkpoint_raw(reader, layer: int, rank: int, device) -> dict:
    """Unstaged rank-`rank` tensors (for the reference) + DenseWeights inputs."""
    from ds41h.decoder import _get_tp, _u8

    p = f"layers.{layer}"
    g = lambda n: _get_tp(reader, f"{p}.{n}", rank)  # noqa: E731
    raw = {
        "woa": _u8(g("attn.wo_a.weight")), "woa_s": _u8(g("attn.wo_a.scale")),
        "wob": _u8(g("attn.wo_b.weight")), "wob_s": _u8(g("attn.wo_b.scale")),
        "w1": _u8(g("ffn.shared_experts.w1.weight")), "s1": _u8(g("ffn.shared_experts.w1.scale")),
        "w3": _u8(g("ffn.shared_experts.w3.weight")), "s3": _u8(g("ffn.shared_experts.w3.scale")),
        "w2": _u8(g("ffn.shared_experts.w2.weight")), "s2": _u8(g("ffn.shared_experts.w2.scale")),
        "gate": reader.get(f"{p}.ffn.gate.weight"), "gate_b": reader.get(f"{p}.ffn.gate.bias"),
        "fn": reader.get(f"{p}.hc_ffn_fn"), "hc_scale": reader.get(f"{p}.hc_ffn_scale"),
        "hc_base": reader.get(f"{p}.hc_ffn_base"), "norm": reader.get(f"{p}.ffn_norm.weight"),
    }  # fmt: skip
    return {k: v.to(device) for k, v in raw.items()}


def random_raw(device, seed: int = 0, concentrate: bool = False) -> tuple[dict, tuple]:
    """Synthetic weights with checkpoint dtypes/shapes (rank slice) for fast stress tests."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def e4m3(*shape):
        b = torch.randint(0, 256, shape, generator=gen, dtype=torch.int32)
        b = torch.where((b & 0x7F) == 0x7F, b - 1, b)  # no NaN codes
        return b.to(torch.uint8)

    def scales(lo, hi, *shape):
        return torch.randint(lo, hi + 1, shape, generator=gen, dtype=torch.int32).to(torch.uint8)

    raw = {
        "woa": e4m3(2048, 4096), "woa_s": scales(115, 118, 64, 128),
        "wob": e4m3(HID, 2048), "wob_s": scales(115, 118, 160, 64),
        "w1": e4m3(INTER, HID), "s1": scales(114, 117, 18, 160),
        "w3": e4m3(INTER, HID), "s3": scales(114, 117, 18, 160),
        "w2": e4m3(HID, INTER), "s2": scales(115, 118, 160, 18),
        "gate": (torch.randn(EXPERTS, HID, generator=gen) * 0.02).to(torch.bfloat16),
        "gate_b": torch.randn(EXPERTS, generator=gen) * 0.01,
        "fn": torch.randn(24, 4 * HID, generator=gen) * 1e-3,
        "hc_scale": torch.tensor([0.7, 0.5, 0.9]),
        "hc_base": torch.randn(24, generator=gen) * 0.2,
        "norm": (1.0 + 0.1 * torch.randn(HID, generator=gen)).to(torch.bfloat16),
    }  # fmt: skip
    if concentrate:  # every token picks experts 3, 77, 150, 200, 301, 383 -> 32 routes per entry
        raw["gate_b"][[3, 77, 150, 200, 301, 383]] += 100.0
    experts = (
        torch.randint(0, 256, (EXPERTS, 2 * INTER, HID // 2), generator=gen, dtype=torch.int32).to(torch.uint8),
        scales(118, 121, EXPERTS, 2 * INTER, HID // 32),
        torch.randint(0, 256, (EXPERTS, HID, INTER // 2), generator=gen, dtype=torch.int32).to(torch.uint8),
        scales(118, 121, EXPERTS, HID, INTER // 32),
    )
    return {k: v.to(device) for k, v in raw.items()}, tuple(t.to(device) for t in experts)


def dense_from_raw(raw: dict):
    from ds41h.decoder import DenseWeights, interleave, stage_fp8

    return DenseWeights(
        woa=stage_fp8(raw["woa"]), woa_s=raw["woa_s"].contiguous(),
        wob=stage_fp8(raw["wob"]), wob_s=raw["wob_s"].contiguous(),
        gate=stage_fp8(raw["gate"].view(torch.uint8), perm8=False), gate_b=raw["gate_b"].float().contiguous(),
        sh13=stage_fp8(interleave(torch.cat([raw["w1"], raw["w3"]]), 32)),
        sh13_s=interleave(torch.cat([raw["s1"], raw["s3"]]), 1),
        sh2=stage_fp8(raw["w2"]), sh2_s=F.pad(raw["s2"], (0, 2)).contiguous(),
        fn=raw["fn"].float().contiguous(), hc_scale=raw["hc_scale"].float().contiguous(),
        hc_base=raw["hc_base"].float().contiguous(), norm=raw["norm"].contiguous(),
    )  # fmt: skip


def make_inputs(m: int, device, seed: int = 1, o_seed: int | None = None) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ogen = torch.Generator(device="cpu").manual_seed(seed if o_seed is None else o_seed)
    positions = torch.randint(0, 4096, (m,), generator=gen)
    inp = {
        "o": (torch.randn(m, 64, 512, generator=ogen) * 0.5).to(torch.bfloat16),
        "positions": positions,
        "cos_sin": rope_table(4096, "cpu"),
        "residual": (torch.randn(m, HC, HID, generator=gen) * 4.0).to(torch.bfloat16),
        "post_mix": (2 * torch.sigmoid(torch.randn(m, HC, generator=gen))).unsqueeze(-1),
        "res_mix": sinkhorn(torch.randn(m, HC, HC, generator=gen)),
        "attn_pre": torch.sigmoid(torch.randn(m, HC, generator=gen)) + 1e-6,
    }
    return {k: v.to(device).contiguous() for k, v in inp.items()}


# ------------------------------------------------------------------------------- reference
def reference(W: RefWeights, inp: dict, reduce=None, routing=None, quant: dict | None = None, dtype=torch.float32) -> dict:
    """Stock-vLLM-sm90 numerics of model.py:591-625 for one TP rank; `reduce` sums partials
    across ranks (fp32 sum of bf16 partials -> bf16); None = solo (identity). `quant`
    ({"dense": bool, "routed": bool, "xs": float}) simulates the RS-wgmma variants' e4m3
    activations where the kernel uses them: wo_a for m <= WOA_SMALL (16, decoder.cu woa_group), wo_b
    and the shared expert per 32-token group of <= WG_MAX_TOKENS tokens, routed experts per expert
    entry of <= WG_MAX_TOKENS routes (entries are chunks of 32 routes in atomic order, so for
    MAXM > 32 an expert with > 32 routes is modelled as all-HMMA; only reachable with m > 32)."""
    red = (lambda t: bf(t)) if reduce is None else (lambda t: bf(reduce(t)))
    o = inp["o"].to(dtype)
    m = o.shape[0]
    dense = bool(quant and quant.get("dense"))
    dq = dense and m <= WG_MAX_TOKENS  # wo_a (WOA_SMALL B tile)
    dq_rows = wg_dense_rows(m, o.device)[:, None]  # wo_b / shared expert rows on the wgmma path
    dsel = (lambda q, v: torch.where(dq_rows, q(v), v)) if dense else (lambda q, v: v)  # noqa: E731
    rq = bool(quant and quant.get("routed"))
    # coarse routed path (decoder.cu COARSE, m <= 8): routed experts and the shared expert always on
    # e4m3 x / h (resident x tile / bt2 B tiles, RS-wgmma).
    cq = bool(quant and quant.get("coarse")) and m <= COARSE_M
    xs = float(quant["xs"]) if quant else 1.0
    pos = inp["positions"].clamp(0, inp["cos_sin"].shape[0] - 1)
    cs = inp["cos_sin"][pos]
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    o16 = o[:, :16]
    pr = o16[..., 448:].unflatten(-1, (32, 2))
    a, b = pr[..., 0], pr[..., 1]
    rot = torch.stack([a * cos + b * sin, b * cos - a * sin], dim=-1).flatten(-2)
    o_rot = bf(torch.cat([o16[..., :448], rot], dim=-1))  # fused_inv_rope (bf16 out)
    o_in = quant_ue8m0_32(o_rot.reshape(m, 2, 4096)) if dq else o_rot.reshape(m, 2, 4096)
    z = bf(torch.einsum("tgd,grd->tgr", o_in, W.woa.view(2, 1024, 4096))).flatten(1)  # bf16 bmm
    y = red(bf(dsel(quant_z, z) @ W.wob.t()))  # wo_b (bf16 out) + all-reduce #1
    # FFN mHC post (fp32 stream feeds the GEMV/RMS; bf16 copy is the residual and collapse input)
    r = inp["post_mix"] * y[:, None, :] + torch.einsum("tij,tid->tjd", inp["res_mix"].to(dtype), inp["residual"].to(dtype))
    resid1 = bf(r)
    xf = r.flatten(1)
    mixes = (xf @ W.fn.t()) * torch.rsqrt(xf.square().sum(-1, keepdim=True) / (HC * HID) + 1e-20)
    s, bb = W.hc_scale, W.hc_base
    pre = torch.sigmoid(mixes[:, :4] * s[0] + bb[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * s[1] + bb[4:8])
    comb = sinkhorn(mixes[:, 8:].unflatten(-1, (4, 4)) * s[2] + bb[8:].view(4, 4))
    xc = bf((inp["attn_pre"][:, :, None] * resid1).sum(1))
    x = bf(xc * torch.rsqrt(xc.square().mean(-1, keepdim=True) + 1e-20) * W.norm)
    logits = x @ W.gate.t()
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt()
    keys = scores + W.gate_b
    order = keys.topk(TOPK + 1, dim=-1)
    ids = order.indices[:, :TOPK] if routing is None else routing
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True) * 1.5

    def swiglu(g, u):
        return bf(F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0))

    x_sh = quant_static(x, xs) if cq else dsel(lambda v: quant_static(v, xs), x)
    h_sh = swiglu(bf(x_sh @ W.w1.t()), bf(x_sh @ W.w3.t()))
    sh = bf((quant_static(h_sh, 0.25) if cq else dsel(lambda v: quant_static(v, 0.25), h_sh)) @ W.w2.t())
    routed = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        tok, slot = torch.where(ids == e)
        w1, w3, w2 = W.expert(e)
        eq = cq or (rq and tok.numel() <= WG_MAX_TOKENS)
        xe = quant_static(x[tok], xs) if eq else x[tok]
        h = swiglu(bf(xe @ w1.t()), bf(xe @ w3.t()))
        routed.index_add_(0, tok, w[tok, slot, None] * ((quant_static(h, 0.25) if eq else h) @ w2.t()))
    out = red(bf(bf(routed) + sh))
    return {"x": out, "residual": resid1, "post": post, "comb": comb, "pre": pre, "y": y, "xin": x,
            "ids": order.indices[:, :TOPK], "weights": w, "keys": keys, "gap": order.values[:, TOPK - 1] - order.values[:, TOPK]}  # fmt: skip


def _cross_check_ref(W: RefWeights, inp: dict, ref: dict) -> None:
    """Our stock-numerics reference vs ds41h.ref.layer (official semantics) on the mHC part."""
    try:
        from ds41h.ref import layer as L
    except ImportError:
        return
    r = L.hc_post(ref["y"], inp["residual"], inp["post_mix"], inp["res_mix"])
    assert (bf(r) - ref["residual"]).abs().max() <= 2 ** -6 * ref["residual"].abs().max()
    pre, post, comb, _ = L.hc_mixes(r, W.fn, W.hc_scale, W.hc_base)
    torch.testing.assert_close(pre, ref["pre"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(post, ref["post"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb, ref["comb"], atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------------------- comparison
def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


_P1A_BUFS: dict = {}


def run_kernel(eng, idx: int, inp: dict):
    p1a = None
    if os.environ.get("DS41H_TEST_P1A", "0") == "1":  # bench: also write the next prologue's P1a (decoder.cu xf_p1a)
        m, dev = inp["residual"].shape[0], inp["residual"].device
        if m not in _P1A_BUFS:
            _P1A_BUFS[m] = (torch.empty((m, HC, HID), dtype=torch.bfloat16, device=dev), torch.empty((m, HID), dtype=torch.bfloat16, device=dev),
                            torch.empty((m, HID // 32), dtype=torch.float32, device=dev))
        p1a = _P1A_BUFS[m]
    return eng.run(idx, inp["o"], inp["positions"], inp["cos_sin"], inp["residual"], inp["post_mix"], inp["res_mix"], inp["attn_pre"], p1a=p1a)


def engine_quant(eng, idx: int) -> dict | None:
    """Reference quantization switches for the engine's compiled variant (None = lossless build: every GEMM
    gets 16-bit activations, the reference is the plain W8A16 / W4A16 stock numerics)."""
    v = eng.variant
    coarse_q = bool(v["coarse"]) and not v.get("lossless", 0)  # LOSSLESS=0: the e4m3 coarse path
    if not (v["routed_mma"] or v["dense_mma"] or coarse_q):
        return None
    return {"dense": bool(v["dense_mma"]), "routed": bool(v["routed_mma"]), "coarse": coarse_q, "xs": eng.layer_xs(idx)}


# Strict tolerances of lossless builds against the 16-bit (W8A16 / W4A16, stock sm90) reference: kernel and
# reference differ only in fp32 summation order, i.e. in occasional 1-ulp bf16 rounding flips at the bf16
# rounding points (z, wo_b out, all-reduce, gate/up, SwiGLU, expert outputs) and their downstream effect.
# Any e4m3 activation path is ~1e-2 (a8_dev) and fails these by an order of magnitude.
LOSSLESS_TOL = {"x": float(os.environ.get("DS41H_TEST_LOSSLESS_X", "2.5e-3")), "residual": 1e-3}


def check(eng, idx: int, W: RefWeights, inp: dict, reduce=None, tag: str = "") -> dict:
    x, resid1, post, comb, pre = run_kernel(eng, idx, inp)
    torch.cuda.synchronize()
    m = inp["residual"].shape[0]
    kid, kw = (t[:m] for t in eng.debug_topk())
    quant = engine_quant(eng, idx)
    if quant:
        tag = f"{tag}-wg{'R' if quant['routed'] else ''}{'D' if quant['dense'] else ''}{'C' if quant['coarse'] and m <= COARSE_M else ''}"
    ref = reference(W, inp, reduce, quant=quant)
    _cross_check_ref(W, inp, ref)
    # Routing: identical sets unless the reference 6th/7th keys nearly tie.
    same = (kid.sort(-1).values == ref["ids"].sort(-1).values).all(-1)
    for t in torch.where(~same)[0].tolist():
        assert float(ref["gap"][t]) < 1e-3 * max(1.0, float(ref["keys"][t].abs().max())), f"token {t}: routing differs {kid[t].tolist()} vs {ref['ids'][t].tolist()}"
    if bool(same.all()):
        ref_ids = ref["ids"]
    else:
        ref = reference(W, inp, reduce, routing=kid.long(), quant=quant)
        ref_ids = kid.long()
    # Routing weights, ordered by expert id on both sides.
    kw_sorted = kw.gather(1, kid.long().argsort(-1))
    rw_sorted = ref["weights"].gather(1, ref_ids.argsort(-1))
    stats = {
        "residual": rel_err(resid1, ref["residual"]),
        "x": rel_err(x, ref["x"]),
        "post": float((post.squeeze(-1) - ref["post"]).abs().max()),
        "comb": float((comb - ref["comb"]).abs().max()),
        "pre": float((pre - ref["pre"]).abs().max()),
        "topk_w": float((kw_sorted - rw_sorted).abs().max()),
        "route_flips": int((~same).sum()),
    }
    if quant:  # diagnostic only: deviation of the e4m3-activation variant from the HMMA numerics
        ref16 = reference(W, inp, reduce, routing=ref_ids)
        stats["a8_dev_x"] = rel_err(x, ref16["x"])
        stats["a8_dev_res"] = rel_err(resid1, ref16["residual"])
    if quant is None:  # lossless build: also against the exact (fp64 GEMM) model with the same bf16 rounding points
        tag += "-lossless"
        ref64 = reference(W.to64(), inp, reduce, routing=ref_ids, dtype=torch.float64)
        stats["x_f64"] = rel_err(x, ref64["x"])
        stats["res_f64"] = rel_err(resid1, ref64["residual"])
        stats["ref32_x_f64"] = rel_err(ref["x"], ref64["x"])  # noise floor: an fp32 implementation of the same model
        stats["ref32_res_f64"] = rel_err(ref["residual"], ref64["residual"])
    print(f"[decoder {tag} m={m}] " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    assert torch.isfinite(x).all() and torch.isfinite(resid1).all()
    assert stats["residual"] < 4e-3, stats
    assert stats["post"] < 2e-4 and stats["comb"] < 2e-4 and stats["pre"] < 2e-4, stats
    assert stats["topk_w"] < 5e-4, stats
    assert stats["x"] < 1.5e-2, stats
    if quant is None:  # strict lossless mode: 16-bit activations everywhere
        # Pass if within the absolute tolerances of the fp32 reference, or (bf16 partial sums across ranks amplify
        # rounding flips when they cancel) as close to the exact model as the fp32 reference itself is (3x its
        # deviation): both differ from it only by fp32 summation order. e4m3 activation paths miss both by ~10x.
        within_abs = stats["x"] < LOSSLESS_TOL["x"] and stats["residual"] < LOSSLESS_TOL["residual"]
        within_floor = stats["x_f64"] <= 3 * stats["ref32_x_f64"] + 2e-4 and stats["res_f64"] <= 3 * stats["ref32_res_f64"] + 2e-4
        assert within_abs or within_floor, ("lossless tolerance", stats)
    return stats


def fill_routed_experts(eng, idx: int, W: RefWeights, inp: dict, loader) -> None:
    """Routing does not depend on expert weights: run once, then load only the chosen experts."""
    run_kernel(eng, idx, inp)
    torch.cuda.synchronize()
    m = inp["residual"].shape[0]
    ids = torch.unique(eng.debug_topk()[0][:m]).tolist()
    loader(ids)


# ------------------------------------------------------------------------------- pytest
def test_stage_layout_cpu():
    """stage_fp8 places byte (row, k) exactly where decoder.cu's mma32 expects it: fragment
    (K32 block q, 16-row tile rt) of (64-row block, 128-K group), lane (g, t), byte h*8 + i =
    row rt*16 + h*8 + g, k = 32q + 8t + P8[i]; activations use the same P8 order, so every HMMA
    slot pairs W[row, k] with x[token, k] (see mma32's slot comment)."""
    from ds41h.decoder import P8, stage_fp8

    for n, k, perm in ((128, 288, True), (64, 256, False)):
        w = torch.randint(0, 256, (n, k), dtype=torch.uint8)
        kp = -(-k // 128) * 128
        wp = F.pad(w, (0, kp - k))
        st = stage_fp8(w, perm8=perm).view(n // 64, kp // 128, 4, 4, 8, 4, 2, 8)  # rb kg q rt g t h i
        rb, kg, q, rt, g, t, h, i = torch.meshgrid(*(torch.arange(s) for s in st.shape), indexing="ij")
        p8 = torch.tensor(P8 if perm else tuple(range(8)))
        rows = rb * 64 + rt * 16 + h * 8 + g
        cols = kg * 128 + q * 32 + t * 8 + p8[i]
        assert torch.equal(st, wp[rows, cols])


def test_p8_activation_order_cpu():
    """decoder.cu writes activations as f16 pairs (x0,x2)(x4,x6)(x1,x3)(x5,x7) per 8-group
    (pack8_p8 / bf16x8_f16_p8 / p8_pos): element c lands at position (c>>1) + (c&1)*4 = P8^-1."""
    from ds41h.decoder import P8

    pos = [(c >> 1) + (c & 1) * 4 for c in range(8)]
    assert [P8[p] for p in pos] == list(range(8))
    assert [P8.index(c) for c in range(8)] == pos


def _prmt(a: int, b: int, sel: int) -> int:
    """PTX prmt.b32 (default mode): result byte i = byte (sel nibble i & 7) of {b, a}, sign-replicated if bit 3."""
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        nib = (sel >> (4 * i)) & 0xF
        byte = pool[nib & 7]
        if nib & 8:
            byte = 0xFF if byte & 0x80 else 0
        out |= byte << (8 * i)
    return out


def _fp4_f16_p8(w: int) -> list[int]:  # decoder.cu fp4_f16_p8
    ve = ((w << 4) & 0x80808080) | ((w << 1) & 0x0E0E0E0E)
    vo = (w & 0x80808080) | ((w >> 3) & 0x0E0E0E0E)
    return [_prmt(ve, 0, 0x1404), _prmt(ve, 0, 0x3424), _prmt(vo, 0, 0x1404), _prmt(vo, 0, 0x3424)]


def test_fp4_f16_p8_cpu():
    """The lossless coarse path's FP4 -> f16 expansion (decoder.cu fp4_f16_p8) is exact: every e2m1 code
    becomes value * 2^-14 as f16 (incl. the 0.5 subnormal and -0), in the P8 pairs (k0,k2) (k4,k6) (k1,k3)
    (k5,k7) that the f16 activation images use (mma32_fp4 slots == mma32 slots)."""
    import random
    import struct

    rnd = random.Random(0)
    words = [rnd.getrandbits(32) for _ in range(4000)] + [0x76543210, 0xFEDCBA98, 0, 0xFFFFFFFF, 0x88888888]
    pairs = ((0, 2), (4, 6), (1, 3), (5, 7))
    for w in words:
        for reg, (ka, kb) in zip(_fp4_f16_p8(w), pairs):
            for half, k in ((reg & 0xFFFF, ka), (reg >> 16, kb)):
                got = struct.unpack("<e", half.to_bytes(2, "little"))[0]
                code = (w >> (4 * k)) & 0xF
                want = _FP4[code] * 2.0**-14
                assert got == want and (half >> 15) == (code >> 3), (hex(w), k, got, want)


def _fp4_lo_e4m3(w: int) -> int:  # sm90.cuh fp4_lo_e4m3 on a 32-bit word
    return ((w << 4) & 0x80808080) | ((w << 2) & 0x1C1C1C1C)


def _fp4_hi_e4m3(w: int) -> int:
    return (w & 0x80808080) | ((w >> 2) & 0x1C1C1C1C)


def _bt_wg_off(tok: int, chunk: int, nt: int) -> int:  # decoder.cu bt_wg_off
    return (chunk * nt + (tok >> 3)) * 128 + (tok & 7) * 16


def _wg_frag_cols(regs, g: int):
    """wgmma .e4m3 A fragment (PTX ISA, m64nNk32, per warp): a0 = (row g, cols 4t..4t+3), a1 = (row g+8,
    same), a2 = (row g, cols 16+4t..), a3 = (row g+8, 16+4t..); byte i of a register = column +i.
    Yields (row, col, byte) for one thread (t given by the caller via regs order)."""
    for r, (row, c0) in enumerate(((g, 0), (g + 8, 0), (g, 16), (g + 8, 16))):
        for i in range(4):
            yield r, row, c0 + i, (regs[r] >> (8 * i)) & 0xFF


def test_wg_fragment_order_cpu():
    """RS-wgmma operand layout of decoder.cu (ROUTED_MMA / DENSE_MMA = 1), emulated per thread:
    (1) FP4: the ldmatrix word of row g (k = 8t..8t+7, low nibble first) expanded with the e4m3 bit
    trick into a0 = lo(w_g), a1 = lo(w_g8), a2 = hi(w_g), a3 = hi(w_g8) puts W[row][WG_PERM[col]]
    * 2^-6 at every fragment column; (2) the P8-staged dense fragment v = (x, y, z, w) as (x, z, y,
    w) gives W[row][WG_PERM[col]]; (3) wg_order / quant_e4m3_wg put x[WG_PERM[p]] at position p and
    the kernel-side writers (wg_pos, store_wg8 word split) agree; (4) the B tile placement
    (bt_wg_off, woa in-job staging) is the canonical no-swizzle K-major layout of desc_b (LBO = NT x
    128 B between K chunks, SBO = 128 B between 8-token groups, block kb at kb x NT x 256 B)."""
    from ds41h.decoder import P8, WG_PERM, quant_e4m3_wg, stage_fp8, wg_order

    torch.manual_seed(0)
    perm = list(WG_PERM)
    assert sorted(perm) == list(range(32))
    # (1) FP4 rows: 16 rows x 32 k (16 bytes), the HMMA path's ldmatrix fragment words
    raw = torch.randint(0, 256, (16, 16), dtype=torch.uint8)
    w_val = dequant_fp4(raw, torch.full((16, 1), 127, dtype=torch.uint8))  # unit scale
    a_frag = torch.full((16, 32), float("nan"))
    for g in range(8):
        for t in range(4):
            wg_, wg8 = (int.from_bytes(bytes(raw[r, 4 * t:4 * t + 4].tolist()), "little") for r in (g, g + 8))
            regs = (_fp4_lo_e4m3(wg_), _fp4_lo_e4m3(wg8), _fp4_hi_e4m3(wg_), _fp4_hi_e4m3(wg8))
            for _, row, col, byte in _wg_frag_cols(regs, g):
                a_frag[row, 4 * t + col if col < 16 else 16 + 4 * t + (col - 16)] = float(torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float())
    assert torch.equal(a_frag * 64.0, w_val[:, perm]), "FP4 wgmma fragment order"
    # (2) dense: stage_fp8 fragment (64-row block, K group 0, k32 block q=0, 16-row tile rt=0)
    wd = torch.randint(0, 256, (64, 128), dtype=torch.uint8)
    wd[(wd & 0x7F) == 0x7F] = 0
    st = stage_fp8(wd).view(-1, 16, 32, 16)[0, 0]  # [lane, 16 bytes]
    d_frag = torch.zeros(16, 32, dtype=torch.uint8)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        v = [int.from_bytes(bytes(st[lane, 4 * j:4 * j + 4].tolist()), "little") for j in range(4)]
        for _, row, col, byte in _wg_frag_cols((v[0], v[2], v[1], v[3]), g):
            d_frag[row, 4 * t + col if col < 16 else 16 + 4 * t + (col - 16)] = byte
    assert torch.equal(d_frag, wd[:16, :32][:, perm]), "dense wgmma fragment order"
    # (3) activation order: python helpers vs the kernel writers
    x = torch.randn(3, 64)
    assert torch.equal(wg_order(x)[:, :32], x[:, :32][:, perm])
    q = quant_e4m3_wg(x, 1.0 / 64)
    assert torch.equal(q.view(torch.float8_e4m3fn).float(), wg_order((x * 64).to(torch.float8_e4m3fn).float()))
    pos = [(k & ~31) | ((k & 1) << 4) | (((k >> 3) & 3) << 2) | ((k & 7) >> 1) for k in range(64)]  # wg_pos
    assert all(perm[pos[k] % 32] == k % 32 for k in range(64))
    for t in range(4):  # store_wg8 / e4m3_wg8: elements 0,2,4,6 at 4t.., 1,3,5,7 at 16+4t..
        assert [pos[8 * t + j] for j in (0, 2, 4, 6)] == [4 * t + i for i in range(4)]
        assert [pos[8 * t + j] for j in (1, 3, 5, 7)] == [16 + 4 * t + i for i in range(4)]
    assert [P8[p] for p in range(8)] == [0, 2, 4, 6, 1, 3, 5, 7]
    # (4) B tile = canonical K-major no-swizzle layout addressed by desc_b + kb * NT * 16
    for nt in (1, 2):
        lbo, sbo = nt * 128, 128
        for kb in range(18):
            for n in range(8 * nt):
                for c in range(32):
                    canon = kb * nt * 256 + (c // 16) * lbo + (n // 8) * sbo + (n % 8) * 16 + c % 16
                    byte_in_row = kb * 32 + c  # wg-ordered activation row, as fill_wg copies it
                    assert _bt_wg_off(n, byte_in_row // 16, nt) + byte_in_row % 16 == canon
                for t in range(4):  # woa in-job staging: lo word -> chunk 2kb, hi word -> chunk 2kb+1
                    base = (n & 7) * 16 + 4 * t
                    assert base + (2 * kb * nt + (n >> 3)) * 128 == kb * nt * 256 + (n // 8) * sbo + (n % 8) * 16 + 4 * t
                    assert base + ((2 * kb + 1) * nt + (n >> 3)) * 128 == kb * nt * 256 + lbo + (n // 8) * sbo + (n % 8) * 16 + 4 * t


class _RoutedPlan:  # mirror of decoder.cu RoutedPlan<R2, NT>
    def __init__(self, r2: bool, nt: int):
        self.r2, self.NKU = r2, 4 if nt == 1 else 2
        self.PER, self.HEAD = 2 * (4 // self.NKU), 2 if r2 else 0
        self.U = self.HEAD + (5 - 1 if r2 else 16 // 4) * self.PER

    def stage(self, u): return 0 if u < self.HEAD else (u - self.HEAD) // self.PER + (1 if self.r2 else 0)  # noqa: E704
    def mt(self, u): return u if u < self.HEAD else (u - self.HEAD) % self.PER // (4 // self.NKU)  # noqa: E704
    def q0(self, u): return 0 if u < self.HEAD else (u - self.HEAD) % (4 // self.NKU) * self.NKU  # noqa: E704
    def nk(self, u): return 2 if u < self.HEAD else self.NKU  # noqa: E704
    def first(self, u): return u == 0 or self.stage(u) != self.stage(u - 1)  # noqa: E704
    def last(self, u): return u == self.U - 1 or self.stage(u) != self.stage(u + 1)  # noqa: E704

    def kb(self, u):
        s = self.stage(u)
        return (16 if s == 0 else 4 * (s - 1)) + self.q0(u) if self.r2 else 4 * s + self.q0(u)


class _DensePlan:  # mirror of decoder.cu DensePlan<KS, NT>
    def __init__(self, ks: int, nt: int):
        self.KS, self.NKU = ks, 4 if nt == 1 else 2
        self.PER, self.U = 4 // self.NKU, -(-ks // self.NKU)

    def stage(self, u): return u // self.PER  # noqa: E704
    def q0(self, u): return u % self.PER * self.NKU  # noqa: E704
    def nk(self, u): return min(self.NKU, self.KS - u * self.NKU)  # noqa: E704
    def first(self, u): return u % self.PER == 0  # noqa: E704
    def last(self, u): return u == self.U - 1 or u % self.PER == self.PER - 1  # noqa: E704
    def kb(self, u): return u * self.NKU  # noqa: E704
    def mt(self, u): return 0  # noqa: E704


def _simulate_pipe(plan, stage_blocks):
    """WgPipe event order (load u, issue u, wait_group 1, retire u-1; final wait 0, retire U-1)
    against the ring protocol (acquire on the first unit of a stage, release after its last load)
    and the per-warp float tables (built on the first unit, parity = stage & 1)."""
    acquired, released, table, covered = [0], [], {}, []
    inflight = []  # (unit, buffer)
    for u in range(plan.U + 1):
        if u < plan.U:
            s = plan.stage(u)
            if plan.first(u):
                if u > 0:
                    acquired.append(s)
                table[s & 1] = s
            assert acquired[-1] == s and s not in released, (u, s)
            base, nblk = stage_blocks(s)
            assert 0 <= plan.q0(u) and plan.q0(u) + plan.nk(u) <= nblk and plan.kb(u) == base + plan.q0(u), (u, s)
            buf = u & 1
            assert all(b != buf for _, b in inflight), f"unit {u} overwrites buffer {buf} in flight"
            covered += [(plan.mt(u), plan.kb(u) + q) for q in range(plan.nk(u))]
            if plan.last(u):
                released.append(s)
            inflight.append((u, buf))  # issue
        while len(inflight) > (1 if u < plan.U else 0):  # wait_group 1 (0 at the end), then retire
            v, _ = inflight.pop(0)
            assert table[plan.stage(v) & 1] == plan.stage(v), f"table of stage {plan.stage(v)} overwritten before unit {v} retired"
    assert acquired == released == sorted(set(acquired))
    return sorted(covered), len(acquired)


def test_wg_plans_cpu():
    """The unrolled wgmma schedules cover every (m-tile, k32 block) exactly once, follow the
    producer's stage order (R1: 4 stages of 4 blocks; R2: stage 0 = blocks 16-17, stages 1-4 =
    blocks 4(s-1)..; dense: stages of 4 blocks), never overwrite an in-flight register buffer and
    never rebuild a scale table before its stage retired."""
    for nt in (1, 2):
        cov, ns = _simulate_pipe(_RoutedPlan(False, nt), lambda s: (4 * s, 4))
        assert ns == 4 and cov == sorted((mt, k) for mt in (0, 1) for k in range(16))
        cov, ns = _simulate_pipe(_RoutedPlan(True, nt), lambda s: (16, 2) if s == 0 else (4 * (s - 1), 4))
        assert ns == 5 and cov == sorted((mt, k) for mt in (0, 1) for k in range(18))
        for ks in (16, 18):
            cov, ns = _simulate_pipe(_DensePlan(ks, nt), lambda s: (4 * s, 4))
            assert ns == -(-ks // 4) and cov == [(0, k) for k in range(ks)]


def test_wg_reference_quant_cpu():
    """The reference's e4m3 activation models match the kernel rules on edge cases."""
    from ds41h.decoder import x_scale

    x = torch.tensor([448.0 * 8, 3.0, -0.001, 0.0] + [0.0] * 28)
    q = quant_ue8m0_32(x)  # amax 3584 -> scale 8: exact
    assert float(q[0]) == 3584.0 and float(q[1]) == 3.0
    z = torch.zeros(64)
    z[0], z[40] = 1.0, 3e-3  # block 0 amax 1 -> e = -13 (+6), block 1 amax 3e-3
    qz = quant_z(z)
    assert float(qz[0]) == 1.0 and abs(float(qz[40]) - 3e-3) <= 3e-3 / 16
    norm = torch.full((HID,), 2.0, dtype=torch.bfloat16)
    xs = x_scale(norm)  # 2 * sqrt(5120) * 1.01 / 448 = 0.3228 -> 0.5
    assert xs == 0.5
    # dense wgmma dispatch is per 32-token group (decoder.cu dense_wg_dispatch on group_of(m, tg))
    assert wg_dense_rows(13).all() and not wg_dense_rows(17).any()
    assert wg_dense_rows(40).tolist() == [False] * 32 + [True] * 8
    assert wg_dense_rows(49).tolist() == [False] * 49


needs_gpu = pytest.mark.skipif(not _cuda_ok(), reason="needs an sm90 GPU")
needs_model = pytest.mark.skipif(not Path(MODEL_DIR, "model.safetensors.index.json").exists(), reason="checkpoint not found")


@pytest.fixture(scope="module")
def solo_engine():
    from ds41h.decoder import DecoderEngine

    torch.backends.cuda.matmul.allow_tf32 = False
    return DecoderEngine(rank=TEST_RANK, world=1)


@pytest.fixture(scope="module")
def random_layers(solo_engine):
    out = {}
    for conc in (False, True):
        raw, experts = random_raw("cuda", seed=7 + conc, concentrate=conc)
        out[conc] = (solo_engine.add_layer(dense_from_raw(raw), experts), RefWeights(raw, experts))
    return out


@needs_gpu
@pytest.mark.parametrize("m", [1, 2, 6, 8, 13, 17, 32])
@pytest.mark.parametrize("concentrate", [False, True])
def test_random_weights(solo_engine, random_layers, m, concentrate):
    if m > solo_engine.maxm:
        pytest.skip("beyond MAXM")
    idx, W = random_layers[concentrate]
    check(solo_engine, idx, W, make_inputs(m, "cuda", seed=100 + m), tag=f"random{'-conc' if concentrate else ''}")


def p1a_reference(y, resid1, post, comb, pre):
    """prologue.cu p1_collapse (MODE 0) on the decoder's outputs: new_r[j] = post[j] * y, then fma(comb[k][j], resid1[k], .)
    (k = 0..3) -> bf16; xc = pre[0] * r[0], then fma(pre[j], r[j], .) over the bf16 streams -> bf16; per 32-column slice
    xor-butterfly sum of bf16(xc)^2. fma emulated in fp64 (exact product; the final fp32 rounding can differ from a true
    fma only by double rounding, vanishingly rare)."""
    def fma(a, b, c):
        return (a.double() * b.double() + c.double()).float()

    m = y.shape[0]
    yf, rk = y.float(), resid1.float()  # [m, HID], [m, 4, HID]
    post, pre, comb = post.reshape(m, HC).float(), pre.reshape(m, HC).float(), comb.reshape(m, HC, HC).float()
    rb = []
    for j in range(HC):
        t = post[:, j:j + 1] * yf
        for k in range(HC):
            t = fma(comb[:, k, j:j + 1].expand_as(t), rk[:, k], t)
        rb.append(t.to(torch.bfloat16))
    resid_out = torch.stack(rb, 1)
    xc = pre[:, 0:1] * rb[0].float()
    for j in range(1, HC):
        xc = fma(pre[:, j:j + 1].expand_as(xc), rb[j].float(), xc)
    xcb = xc.to(torch.bfloat16)
    v = (xcb.float() * xcb.float()).view(m, HID // 32, 32)
    idx = torch.arange(32, device=v.device)
    for o in (16, 8, 4, 2, 1):
        v = v + v[..., idx ^ o]
    return resid_out, xcb, v[..., 0].contiguous()


@needs_gpu
@pytest.mark.parametrize("m", [1, 2, 6, 8])
def test_xf_p1a(solo_engine, random_layers, m):
    """XF_P1A: the next layer's prologue P1a written by the decoder's XF jobs == p1a_reference (bitwise up to the fp64
    fma emulation) of the decoder's own outputs; the decoder outputs themselves are unchanged."""
    if m > solo_engine.maxm:
        pytest.skip("beyond MAXM")
    idx, W = random_layers[False]
    inp = make_inputs(m, "cuda", seed=300 + m)
    base = [t.clone() for t in run_kernel(solo_engine, idx, inp)]
    r_out = torch.full((m, HC, HID), float("nan"), dtype=torch.bfloat16, device="cuda")
    xcb = torch.full((m, HID), float("nan"), dtype=torch.bfloat16, device="cuda")
    xsq = torch.full((m, HID // 32), float("nan"), dtype=torch.float32, device="cuda")
    out = solo_engine.run(idx, inp["o"], inp["positions"], inp["cos_sin"], inp["residual"], inp["post_mix"], inp["res_mix"], inp["attn_pre"],
                          p1a=(r_out, xcb, xsq))
    torch.cuda.synchronize()
    y, resid1, post, comb, pre = out
    # the layer outputs are unchanged by the extra work (up to fp32 atomic summation order between launches)
    for a, b in zip(out, base):
        assert rel_err(a, b) < 1e-2
    ref_r, ref_x, ref_s = p1a_reference(y, resid1, post, comb, pre)
    for got, want, name in ((r_out, ref_r, "resid_out"), (xcb, ref_x, "xcb")):
        diff = (got.view(torch.int16) != want.view(torch.int16)).float().mean().item()
        assert diff <= 1e-4, f"{name}: {diff:.2e} of the bf16 values differ"
    assert torch.isfinite(xsq).all()
    torch.testing.assert_close(xsq, ref_s, rtol=1e-6, atol=0.0)


@pytest.fixture(scope="module")
def ckpt_layer(solo_engine):
    from ds41h.decoder import load_raw_experts
    from ds41h.ref.weights import CheckpointReader

    reader = CheckpointReader(MODEL_DIR)
    raw = checkpoint_raw(reader, LAYER, TEST_RANK, "cuda")
    experts = load_raw_experts(reader, LAYER, TEST_RANK, "cuda", experts=range(0))  # filled on demand
    idx = solo_engine.add_layer(dense_from_raw(raw), experts)
    loaded: set[int] = set()

    def loader(ids):
        new = [e for e in ids if e not in loaded]
        if new:
            load_raw_experts(reader, LAYER, TEST_RANK, "cuda", experts=new, out=experts)
            loaded.update(new)

    return idx, RefWeights(raw, experts), loader


@needs_gpu
@needs_model
@pytest.mark.parametrize("m", [1, 6, 8])
def test_checkpoint_layer_solo(solo_engine, ckpt_layer, m):
    idx, W, loader = ckpt_layer
    inp = make_inputs(m, "cuda", seed=200 + m)
    fill_routed_experts(solo_engine, idx, W, inp, loader)
    check(solo_engine, idx, W, inp, tag=f"ckpt-L{LAYER}-r{TEST_RANK}")


@needs_gpu
@needs_model
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_gpu_torchrun():
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4", str(Path(__file__).resolve()), "--dist"]
    subprocess.run(cmd, check=True, cwd=str(ROOT), timeout=3600)


# ------------------------------------------------------------------------------- routed GEMM loop (HMMA vs wgmma)
class RoutedProblem:
    """Synthetic routed-expert GEMM jobs for ext.bench_routed: `slots` (expert, routes) entries over
    distinct raw-layout experts (one per slot), token activations x [T, HID] and SwiGLU outputs
    h [6T, INTER] in both kernel formats (f16 P8 for HMMA, e4m3 wg order for wgmma)."""

    def __init__(self, routes: list[list[int]], tokens: int, device="cuda", seed: int = 0, experts: int | None = None):
        from ds41h.decoder import P8, quant_e4m3_wg

        gen = torch.Generator(device=device).manual_seed(seed)
        self.slots = len(routes)
        n_exp = self.slots if experts is None else experts
        u8 = dict(dtype=torch.uint8, device=device, generator=gen)
        self.w13 = torch.randint(0, 256, (n_exp, 2 * INTER, HID // 2), **u8)
        self.s13 = torch.randint(118, 122, (n_exp, 2 * INTER, HID // 32), **u8)
        self.w2 = torch.randint(0, 256, (n_exp, HID, INTER // 2), **u8)
        self.s2 = torch.randint(118, 122, (n_exp, HID, INTER // 32), **u8)
        self.tokens, self.R = tokens, tokens * TOPK
        self.routes = [list(r) for r in routes]
        assert all(1 <= len(r) <= 32 and max(r) < self.R for r in routes)
        self.slot_expert = (torch.arange(self.slots, device=device) % n_exp).int()
        sr = torch.zeros(self.slots, 32, dtype=torch.int32)
        for i, r in enumerate(routes):
            sr[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
        self.slot_routes = sr.to(device)
        self.slot_count = torch.tensor([len(r) for r in routes], dtype=torch.int32, device=device)
        self.x = torch.randn(tokens, HID, device=device, generator=gen).half()
        self.h = (torch.randn(self.R, INTER, device=device, generator=gen) * 2.0).clamp(-100, 100).half()
        self.topk_w = torch.rand(self.R, device=device, generator=gen) + 0.5
        amax = float(self.x.float().abs().max())
        self.xs = float(2.0 ** torch.ceil(torch.log2(torch.tensor(amax / 448.0))))
        p8 = torch.tensor(P8, device=device)
        self.xh = self.x.view(tokens, -1, 8).index_select(2, p8).reshape(tokens, HID).contiguous()
        self.hh = self.h.view(self.R, -1, 8).index_select(2, p8).reshape(self.R, INTER).contiguous()
        self.xq = quant_e4m3_wg(self.x.float(), self.xs)
        self.hq = quant_e4m3_wg(self.h.float(), 0.25)
        self.claim = torch.zeros(1, dtype=torch.int32, device=device)

    def nbytes(self, mode: int = 3) -> int:
        per = (2 * INTER * HID // 2 + 2 * INTER * HID // 32) * (mode & 1) + (HID * INTER // 2 + HID * INTER // 32) * (mode >> 1 & 1)
        return self.slots * per

    def run(self, ext, wg: int, mode: int = 3, reps: int = 1):
        self.hacc = torch.zeros(self.R, 2 * INTER, device=self.x.device)
        self.yracc = torch.zeros(self.tokens, HID, device=self.x.device)
        ext.bench_routed(wg, self.w13, self.s13, self.w2, self.s2, self.slot_expert, self.slot_routes, self.slot_count, self.xh, self.xq,
                         self.hh, self.hq, self.topk_w, self.hacc, self.yracc, self.claim, self.xs, mode, reps)  # fmt: skip
        return self.hacc, self.yracc

    def reference(self, wg: bool, exact: bool = False):
        """fp32 reference of one launch (mode 3). wg: the kernel's e4m3 activations on entries of
        <= WG_MAX_TOKENS routes; exact: unquantized activations everywhere."""
        hacc = torch.zeros(self.R, 2 * INTER, device=self.x.device)
        yracc = torch.zeros(self.tokens, HID, device=self.x.device)
        for i, r in enumerate(self.routes):
            e = int(self.slot_expert[i])
            rt = torch.tensor(r, device=self.x.device)
            q = wg and not exact and len(r) <= WG_MAX_TOKENS
            x = self.x[rt // TOPK].float()
            h = self.h[rt].float()
            if q:
                x, h = quant_static(x, self.xs), quant_static(h, 0.25)
            hacc.index_add_(0, rt, x @ dequant_fp4(self.w13[e], self.s13[e]).t())
            yracc.index_add_(0, rt // TOPK, self.topk_w[rt, None] * (h @ dequant_fp4(self.w2[e], self.s2[e]).t()))
        return hacc, yracc


@needs_gpu
def test_routed_variants():
    """Routed GEMM loops alone (bench kernel, both inner loops in one build): HMMA and RS-wgmma vs the
    fp32 reference on entries of 1..32 routes (wgmma for <= 16, its HMMA fallback above)."""
    from ds41h.decoder import extension

    torch.backends.cuda.matmul.allow_tf32 = False
    counts = [1, 2, 3, 5, 8, 9, 13, 16, 17, 32]
    routes = [[tok * TOPK + i % TOPK for tok in range(c)] for i, c in enumerate(counts)]
    prob = RoutedProblem(routes, tokens=32, seed=3)
    ext = extension()
    out = {}
    for wg in (0, 1):
        hacc, yracc = prob.run(ext, wg)
        torch.cuda.synchronize()
        rh, ry = prob.reference(bool(wg))
        eh, ey = prob.reference(bool(wg), exact=True)
        out[wg] = (hacc.clone(), yracc.clone())
        stats = (rel_err(hacc, rh), rel_err(yracc, ry), rel_err(hacc, eh), rel_err(yracc, ey))
        print(f"[routed {'wgmma' if wg else 'hmma '}] w13 {stats[0]:.2e} w2 {stats[1]:.2e} (vs unquantized activations: {stats[2]:.2e} / {stats[3]:.2e})")
        assert torch.isfinite(hacc).all() and torch.isfinite(yracc).all()
        assert stats[0] < 2e-3 and stats[1] < 2e-3, stats
    print(f"[routed] wgmma vs hmma: w13 {rel_err(out[1][0], out[0][0]):.2e} w2 {rel_err(out[1][1], out[0][1]):.2e}")


def bench_routed(args) -> None:
    """Routed GEMM loop only, `--layers` x `--experts` distinct experts (default 40 x 30, 5.6 GB of
    weights per launch, far beyond L2), for each M: decode routing (M tokens x top-6 spread
    round-robin over the experts of a layer, i.e. 1-2 routes per expert) or `--rpe` routes per
    expert. Times HMMA and RS-wgmma inner loops (bench_routed reps back to back, CUDA events)."""
    from ds41h.decoder import extension

    ext = extension()
    n_slots = args.layers * args.experts
    print(f"[bench-routed] {n_slots} experts, {n_slots * EXPERT_BYTES / 1e9:.2f} GB per launch, STAGES={ext.STAGES} PER_SM={ext.PER_SM}")
    weights = None
    for m in args.m:
        if args.rpe:
            routes = [[tok * TOPK + s % TOPK for tok in range(args.rpe)] for s in range(n_slots)]
            tokens = args.rpe
        else:
            per_layer = [[r for r in range(m * TOPK) if r % args.experts == e] for e in range(args.experts)]
            routes = [per_layer[s % args.experts] or [0] for s in range(n_slots)]
            tokens = m
        prob = RoutedProblem(routes, tokens=tokens, seed=5, experts=None if weights is None else 1)
        if weights is None:
            weights = (prob.w13, prob.s13, prob.w2, prob.s2)
        else:  # reuse the big expert tensors, only the routing/activations change
            prob.w13, prob.s13, prob.w2, prob.s2 = weights
            prob.slot_expert = torch.arange(n_slots, device="cuda", dtype=torch.int32)
        counts = torch.bincount(prob.slot_count.cpu()).tolist()
        res = {}
        for wg in args.wg:
            prob.run(ext, wg, reps=2)
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            ext.bench_routed(wg, prob.w13, prob.s13, prob.w2, prob.s2, prob.slot_expert, prob.slot_routes, prob.slot_count, prob.xh, prob.xq,
                             prob.hh, prob.hq, prob.topk_w, prob.hacc, prob.yracc, prob.claim, prob.xs, 3, args.iters)  # fmt: skip
            t1.record()
            torch.cuda.synchronize()
            us = t0.elapsed_time(t1) * 1e3 / args.iters
            res[wg] = us
            print(f"[bench-routed {'wgmma' if wg else 'hmma '}] m={m} routes/expert hist={counts} {us:9.1f} us/launch {prob.nbytes() / (us * 1e-6) / 1e9:8.1f} GB/s")
        if len(res) == 2:
            print(f"[bench-routed] m={m} wgmma / hmma time = {res[1] / res[0]:.3f}")


# ------------------------------------------------------------------------------- distributed / bench
def _dist_setup():
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    return rank, dist, dist.new_group(backend="gloo")


def _layer_bytes(eng, idx: int, n_experts: int) -> int:
    dense = eng._layers[idx][0].nbytes()
    return dense + n_experts * EXPERT_BYTES


def bench(eng, idx: int, inp: dict, iters: int, label: str) -> None:
    m = inp["residual"].shape[0]
    for _ in range(10):
        run_kernel(eng, idx, inp)
    torch.cuda.synchronize()
    n_exp = len(torch.unique(eng.debug_topk()[0][:m]))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):
            run_kernel(eng, idx, inp)
    g.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = max(1, iters // 8)
    start.record()
    for _ in range(reps):
        g.replay()
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) * 1e3 / (reps * 8)
    gb = _layer_bytes(eng, idx, n_exp) / 1e9
    print(f"[bench {label}] m={m:2d} experts={n_exp:3d} {us:8.1f} us/layer  weights {gb * 1e3:7.1f} MB  {gb / (us * 1e-6):7.1f} GB/s")


_MULTI: dict[int, list[int]] = {}  # engine id -> its layer copies


def _nvml_energy_reader():
    """() -> mJ of the current CUDA device (NVML total energy counter, ctypes), or None if unavailable."""
    import ctypes

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        assert nvml.nvmlInit_v2() == 0
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        bus = f"{pr.pci_domain_id:08X}:{pr.pci_bus_id:02X}:{pr.pci_device_id:02X}.0".encode()
        h = ctypes.c_void_p()
        assert nvml.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(h)) == 0
    except Exception:  # noqa: BLE001
        return None

    def read() -> int:
        e = ctypes.c_ulonglong()
        nvml.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e))
        return int(e.value)

    return read


def bench_multi(eng, raw: dict, experts: tuple, inp: dict, layers: int, seconds: float, label: str, dpf: bool = False) -> None:
    """Steady-state us/layer over `layers` distinct copies of the layer (dense weights and routed
    experts in separate memory, so nothing stays L2-resident between launches, like the real
    40-layer step), CUDA-graph replay: one burst replay, then `seconds` of back-to-back replay
    (the 200 W cap settles the clocks), then the timed replays."""
    import time

    m = inp["residual"].shape[0]
    # Copies are made once per process (they hold the experts loaded so far; later M values only
    # route to experts that are loaded into the copies too, since the loader fills `experts` and the
    # copies share nothing: refresh the copies' experts from `experts` each call).
    multi = _MULTI.setdefault(id(eng), [])
    if len(multi) < layers:
        multi += [eng.add_layer(dense_from_raw(raw), tuple(t.clone() for t in experts)) for _ in range(layers - len(multi))]
    idxs = multi[:layers]
    for i in idxs:
        for dst, src in zip(eng.experts_of(i), experts):
            dst.copy_(src)
    # --dpf: copy k L2-prefetches copy k+1's dense critical-path weights (like layer i -> i+1 in the model)
    for k, i in enumerate(idxs):
        eng.set_prefetch(i, eng.dense_prefetch_tensors(idxs[(k + 1) % len(idxs)]) if dpf else [], dense=True)
    for i in idxs:
        run_kernel(eng, i, inp)
    torch.cuda.synchronize()
    n_exp = len(torch.unique(eng.debug_topk()[0][:m]))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in idxs:
            run_kernel(eng, i, inp)
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    g.replay()
    t1.record()
    torch.cuda.synchronize()
    burst = t0.elapsed_time(t1) * 1e3 / layers
    end, n = time.time() + seconds, 0
    while time.time() < end:
        g.replay()
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
    # timed window ~1.5 s (the 200 W governor oscillates on a ~100 ms scale; short windows are noisy)
    energy = _nvml_energy_reader()
    torch.cuda.synchronize()
    e0 = energy() if energy else None
    t0.record()
    start, reps = time.time(), 0
    while time.time() - start < 1.5:
        g.replay()
        reps += 1
        if reps % 4 == 0:
            torch.cuda.synchronize()
    t1.record()
    torch.cuda.synchronize()
    e1 = energy() if energy else None
    us = t0.elapsed_time(t1) * 1e3 / (reps * layers)
    mj = (e1 - e0) / (reps * layers) if e0 is not None else float("nan")
    gb = _layer_bytes(eng, idxs[0], n_exp) / 1e9
    label += " dpf" if dpf else ""
    print(f"[bench-multi {label}] m={m:2d} experts={n_exp:3d} layers={layers} burst {burst:7.1f} us/layer | steady {us:7.1f} us/layer "
          f"weights {gb * 1e3:6.1f} MB {gb / (us * 1e-6):7.1f} GB/s | {mj:6.1f} mJ/layer {mj / (us * 1e-3):5.0f} W")


def _tok_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token relative L2 difference of [m, ...] tensors (float)."""
    a, b = a.float().flatten(1), b.float().flatten(1)
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def stress_layers(device, n: int) -> list[tuple[dict, tuple]]:
    """Synthetic layers with different routing regimes: random (spread), concentrated (6 experts), and a
    16-expert pool (bias boost, n ~ 16-24 distinct experts at m = 8, like the real model's 17-32)."""
    out = []
    for i in range(n):
        raw, experts = random_raw(device, seed=11 + i, concentrate=(i % 4 == 2))
        if i % 4 == 3:
            gen = torch.Generator(device="cpu").manual_seed(500 + i)
            pool = torch.randperm(EXPERTS, generator=gen)[:16].to(device)
            raw["gate_b"][pool] += 0.6
        out.append((raw, experts))
    return out


def stress(specs: list[str], rank: int, world: int, cpu, reduce, iters: int, ms: list[int], n_layers: int, reps: int, ref_every: int,
           tol: float, seed: int = 0, seconds: float = 0.0) -> int:
    """Race / hazard detector (DENSE_TS1 question): every iteration runs `n_layers` distinct layers x `reps`
    back to back (one stream, no host sync in between: cross-launch scratch reuse, epoch parity, exchange flags under
    world = 4) on fresh random inputs, for every build in `specs` (interleaved, order rotated per iteration). Checks
    (a) every rep against rep 0 of the same build and against rep 0 of the first build (all builds compute the same math:
    only fp32 atomic order may differ -> per-token relative L2 difference ~1e-3), and (b) every `ref_every` iterations every
    output against the fp32 stock-numerics reference. A token above `tol` is reported (race / stale data / missing
    barrier signature), with the reference's routing gap (near-tie routing flips are legitimate and flagged as such).
    Returns the number of flagged tokens (excluding routing near-ties)."""
    import time

    from ds41h.decoder import DecoderEngine

    engines = {s: DecoderEngine(rank=rank, world=world, cpu_group=cpu, spec=s) for s in specs}
    layers = stress_layers("cuda", n_layers)
    dense = [dense_from_raw(raw) for raw, _ in layers]
    idx = {s: [e.add_layer(d, ex) for d, (_, ex) in zip(dense, layers)] for s, e in engines.items()}
    Ws = [RefWeights(raw, ex) for raw, ex in layers]
    keys = ("x", "residual", "post", "comb", "pre")
    worst = {s: {"rep": {k: 0.0 for k in keys}, "xb": {k: 0.0 for k in keys}, "ref_x": 0.0, "ref_res": 0.0} for s in specs}
    flagged, near, launches = 0, 0, 0
    hist = {s: torch.zeros(8, dtype=torch.long) for s in specs}  # per-token rep-diff of x: decades 1e-6 .. 1e1
    deadline = time.time() + seconds if seconds > 0 else float("inf")
    done, last_print = 0, time.time()
    for it in range(iters):
        stop = torch.tensor([1 if time.time() > deadline else 0])
        if world > 1:  # every rank must run the same launches (exchange flags), so the time limit is agreed on
            import torch.distributed as dist

            dist.all_reduce(stop, op=dist.ReduceOp.MAX, group=cpu)
        if int(stop):
            break
        done = it + 1
        m = ms[it % len(ms)]
        s0 = 10_000 + seed * 10_000_000 + it * 64
        inps = [make_inputs(m, "cuda", seed=s0 + i, o_seed=s0 + i + 1_000_000 * rank) for i in range(n_layers)]
        order = specs[it % len(specs):] + specs[:it % len(specs)]
        outs = {}
        for s in order:
            outs[s] = [[run_kernel(engines[s], idx[s][i], inps[i]) for i in range(n_layers)] for _ in range(reps)]
            launches += reps * n_layers
        torch.cuda.synchronize()
        base = outs[specs[0]][0]
        refs: dict = {}

        def gap_of(i: int, t: int) -> float:  # reference routing gap of (layer i, token t), lazily (flags only)
            if i not in refs:
                refs[i] = reference(Ws[i], inps[i], reduce)
            return float(refs[i]["gap"][t] / refs[i]["keys"][t].abs().max().clamp_min(1.0))

        for s in specs:
            for r in range(reps):
                for i in range(n_layers):
                    for k, a, b0, bx in zip(keys, outs[s][r][i], outs[s][0][i], base[i]):
                        a = a.reshape(m, -1)
                        if r:
                            d = _tok_rel(a, b0.reshape(m, -1))
                            worst[s]["rep"][k] = max(worst[s]["rep"][k], float(d.max()))
                            if k == "x":
                                hist[s] += torch.histc(torch.log10(d.cpu().clamp(1e-6, 9.9)), bins=8, min=-6, max=2).long()
                            for t in torch.where(d > tol)[0].tolist():
                                g = gap_of(i, t) if world == 1 else float("nan")
                                tie = g < 1e-3
                                near += tie
                                flagged += not tie
                                print(f"[stress {'TIE ' if tie else 'FLAG'} {s or 'default'} r{rank}] it={it} m={m} layer={i} rep={r} tok={t} {k} rep-vs-rep "
                                      f"{float(d[t]):.3e} gap {g:.2e}")
                        if s != specs[0]:
                            d = _tok_rel(a, bx.reshape(m, -1))
                            worst[s]["xb"][k] = max(worst[s]["xb"][k], float(d.max()))
                            for t in torch.where(d > tol)[0].tolist():
                                g = gap_of(i, t) if world == 1 else float("nan")
                                tie = g < 1e-3
                                near += tie
                                flagged += not tie
                                print(f"[stress {'TIE ' if tie else 'FLAG'} {s or 'default'} r{rank}] it={it} m={m} layer={i} rep={r} tok={t} {k} "
                                      f"vs-{specs[0] or 'default'} {float(d[t]):.3e} gap {g:.2e}")
        if ref_every and it % ref_every == 0:
            for i in range(n_layers):
                ref = refs[i] if i in refs else reference(Ws[i], inps[i], reduce)
                gap = ref["gap"] / ref["keys"].abs().amax(-1).clamp_min(1.0)
                for s in specs:
                    for r in range(reps):
                        x, resid1 = outs[s][r][i][0], outs[s][r][i][1]
                        dx, dr = _tok_rel(x, ref["x"]), _tok_rel(resid1, ref["residual"])
                        worst[s]["ref_x"] = max(worst[s]["ref_x"], float(dx.max()))
                        worst[s]["ref_res"] = max(worst[s]["ref_res"], float(dr.max()))
                        for t in torch.where((dx > 4 * tol) | (dr > tol))[0].tolist():
                            tie = float(gap[t]) < 1e-3
                            near += tie
                            flagged += not tie
                            print(f"[stress {'TIE ' if tie else 'FLAG'} {s or 'default'} r{rank}] it={it} m={m} layer={i} rep={r} tok={t} vs-ref x {float(dx[t]):.3e} "
                                  f"res {float(dr[t]):.3e} gap {float(gap[t]):.2e}")
        if time.time() - last_print > 30:
            last_print = time.time()
            print(f"[stress r{rank}] it={it}/{iters} launches={launches} flagged={flagged} near-ties={near}", flush=True)
    for s in specs:
        w = worst[s]
        print(f"[stress SUMMARY {s or 'default'} r{rank}] iters={done} m={ms} layers={n_layers} reps={reps} | rep-vs-rep max "
              + " ".join(f"{k}={v:.2e}" for k, v in w["rep"].items())
              + (" | vs-first-build max " + " ".join(f"{k}={v:.2e}" for k, v in w["xb"].items()) if s != specs[0] else "")
              + f" | vs-ref max x={w['ref_x']:.2e} res={w['ref_res']:.2e} | x rep-diff decades 1e-6..1e2: {hist[s].tolist()}")
    print(f"[stress r{rank}] TOTAL launches={launches} flagged={flagged} near-ties={near}")
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", action="store_true", help="4-GPU torchrun mode")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-multi", type=int, default=0, help="steady-state timing over N distinct layer copies (cold L2)")
    ap.add_argument("--seconds", type=float, default=2.0, help="--bench-multi: seconds of warm replay before timing")
    ap.add_argument("--dpf", type=int, default=0, help="--bench-multi: 1 = dense next-layer L2 prefetch, 2 = both 0 and 1")
    ap.add_argument("--r1kp", type=int, default=0, help="coarse R1 K parts (0 = engine default)")
    ap.add_argument("--lag", type=int, default=-1, help="coarse R2 expert lag (-1 = engine default, 0 = auto)")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 6, 8])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--random", action="store_true", help="synthetic weights (no checkpoint)")
    ap.add_argument("--bench-routed", action="store_true", help="routed GEMM loop micro-benchmark (HMMA vs RS-wgmma)")
    ap.add_argument("--layers", type=int, default=40, help="--bench-routed: layers worth of experts")
    ap.add_argument("--experts", type=int, default=30, help="--bench-routed: distinct experts per layer")
    ap.add_argument("--rpe", type=int, default=0, help="--bench-routed: fixed routes per expert (0 = decode routing of M tokens)")
    ap.add_argument("--wg", type=int, nargs="+", default=[0, 1], help="--bench-routed: inner loops (0 = HMMA, 1 = wgmma)")
    ap.add_argument("--variants", type=str, default="", help="--bench-multi: ';'-separated DS41H_DECODER_DEFINES specs, one engine each "
                    "in this process (checkpoint loaded once; '' entries = the default build), e.g. ';PROUTE=0;R1F_PIPE=0'")
    ap.add_argument("--rounds", type=int, default=1, help="--variants: repeat the variant list (interleaved A/B rounds)")
    ap.add_argument("--stress", type=int, default=0, help="race detector: N iterations over --stress-layers synthetic layers x --reps, "
                    "every build in --variants (default: the env build), see stress()")
    ap.add_argument("--stress-layers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ref-every", type=int, default=8)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--stress-seed", type=int, default=0)
    ap.add_argument("--stress-seconds", type=float, default=0.0, help="--stress: stop after this many seconds (0 = run all iterations)")
    args = ap.parse_args()
    if args.stress:
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.dist:
            rank, dist, cpu = _dist_setup()

            def reduce(t: torch.Tensor) -> torch.Tensor:
                t = t.clone()
                dist.all_reduce(t)
                return t

            world = 4
        else:
            rank, dist, cpu, world, reduce = TEST_RANK, None, None, 1, None
            torch.cuda.set_device(0)
        specs = [s.strip() for s in args.variants.split(";")] if args.variants else [os.environ.get("DS41H_DECODER_DEFINES", "")]
        flagged = stress(specs, rank, world, cpu, reduce, args.stress, args.m, args.stress_layers, args.reps, args.ref_every, args.tol, args.stress_seed, args.stress_seconds)
        if dist is not None:
            t = torch.tensor([flagged], device="cuda")
            dist.all_reduce(t)
            flagged = int(t.item())
            dist.destroy_process_group()
        sys.exit(1 if flagged else 0)
    if args.bench_routed:
        torch.cuda.set_device(0)
        if args.iters == 400:
            args.iters = 20
        bench_routed(args)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    from ds41h.decoder import DecoderEngine, load_raw_experts

    if args.dist:
        rank, dist, cpu = _dist_setup()
        world = 4

        def reduce(t: torch.Tensor) -> torch.Tensor:  # fp32 sum of the ranks' bf16 partials
            t = t.clone()
            dist.all_reduce(t)
            return t
    else:
        rank, dist, cpu, world, reduce = TEST_RANK, None, None, 1, None
        torch.cuda.set_device(0)
    eng = DecoderEngine(rank=rank, world=world, cpu_group=cpu)
    if args.r1kp or args.lag >= 0:
        eng.set_coarse(args.r1kp or int(os.environ.get("DS41H_DECODER_R1KP", "0")), max(args.lag, 0))
    if args.random:
        raw, experts = random_raw("cuda", seed=7)
        loader = lambda ids: None  # noqa: E731
    else:
        from ds41h.ref.weights import CheckpointReader

        reader = CheckpointReader(MODEL_DIR)
        raw = checkpoint_raw(reader, args.layer, rank, "cuda")
        experts = load_raw_experts(reader, args.layer, rank, "cuda", experts=range(0))
        loaded: set[int] = set()

        def loader(ids):
            if dist is not None:  # same routing on every rank; union anyway for safety
                lst = [None] * world
                dist.all_gather_object(lst, sorted(ids), group=cpu)
                ids = sorted(set().union(*map(set, lst)))
            new = [e for e in ids if e not in loaded]
            if new:
                load_raw_experts(reader, args.layer, rank, "cuda", experts=new, out=experts)
                loaded.update(new)

    idx = eng.add_layer(dense_from_raw(raw), experts)
    W = RefWeights(raw, experts)
    failures = 0
    for m in args.m:
        # Shared stream inputs on every rank; o differs per rank (each rank owns 16 heads).
        inp = make_inputs(m, "cuda", seed=300 + m, o_seed=300 + m + 1000 * rank)
        fill_routed_experts(eng, idx, W, inp, loader)
        if args.bench_multi and args.variants:  # A/B of several builds in one process (interleaved rounds)
            if not hasattr(main, "_engines"):
                main._engines = {}
            for _ in range(args.rounds):
                for spec in args.variants.split(";"):
                    spec = spec.strip()
                    if spec not in main._engines:
                        main._engines[spec] = DecoderEngine(rank=rank, world=world, cpu_group=cpu, spec=spec)
                    bench_multi(main._engines[spec], raw, experts, inp, args.bench_multi, args.seconds,
                                f"{'4gpu' if world == 4 else 'solo'} r{rank} [{spec or 'default'}]", bool(args.dpf))  # fmt: skip
        elif args.bench_multi:
            for dpf in ([False, True] if args.dpf == 2 else [bool(args.dpf)]):
                bench_multi(eng, raw, experts, inp, args.bench_multi, args.seconds, f"{'4gpu' if world == 4 else 'solo'} r{rank}", dpf)
        elif args.bench:
            bench(eng, idx, inp, args.iters, f"{'4gpu' if world == 4 else 'solo'} r{rank}")
        else:
            try:
                check(eng, idx, W, inp, reduce, tag=f"{'4gpu' if world == 4 else 'solo'}-L{args.layer}-r{rank}")
            except AssertionError as e:
                failures += 1
                print(f"[decoder rank {rank} m={m}] FAILED: {e}")
    if dist is not None:
        t = torch.tensor([failures], device="cuda")
        dist.all_reduce(t)
        failures = int(t.item())
        dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
