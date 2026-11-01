"""Pure-torch reference of one DeepSeek-V4.1-Flash decoder layer (decode step), float32 math.

Semantics follow the official ``inference/model.py`` (line numbers below refer to it) unless a
function says otherwise:

* mHC (Block.hc_mixes/hc_pre/hc_post, :950-968; kernel.py hc_split_sinkhorn :406-474): the
  residual stream is X [T, 4, D]; mixes = (flatten(X) @ fn.T) * rsqrt(mean(X^2) + 1e-20);
  pre = sigmoid(m0..3*s0 + b) + 1e-6, post = 2*sigmoid(m4..7*s1 + b), comb = row-softmax + eps,
  column-normalise, then 19 x (row, column) normalisations (20 Sinkhorn iterations, eps 1e-6).
  Each sublayer collapses its input with the *previous* sublayer's pre (Block.forward :970-996):
  attention uses the previous layer's ffn_pre (one-hot [1,0,0,0] for layer 0), the FFN uses this
  layer's attn_pre. hc_post: X'[j] = post[j]*y + sum_i comb[i, j] * X[i].
* RMSNorm eps 1e-20 everywhere (config norm_eps), (w * x*rsqrt(mean(x^2)+eps)).
* Attention (Attention.forward :766-791): qr = q_norm(wq_a a); q = wq_b qr -> [T, H, 512];
  kv = kv_norm(wkv a); interleaved-pair RoPE on the last 64 dims (apply_rotary_emb :392-406);
  ratio-0 layers use theta 10000 without YaRN, compressed layers theta 160000 with YaRN
  (factor 16, original 65536, beta 32/1) (:681-697). Window 128 (decode attends to the ring of the
  last 128 positions, causal), attention sink added to the softmax denominator only
  (kernel.py sparse_attn :382-385), scale 512**-0.5; inverse RoPE on o; o viewed as
  [T, groups, 4096] (8 groups of 8 heads), per-group wo_a [1024, 4096] (einsum, wo_a in bf16),
  then wo_b [5120, 8192].
* MoE (Gate :794-829, Expert :832-853, MoE :856-906): scores = sqrt(softplus(x W^T)) in fp32;
  experts = top-6 of (scores + bias) (noaux_tc); weights = scores[sel] / (sum + 1e-20) * 1.5;
  expert: silu(min(w1 x, 10)) * clamp(w3 x, -10, 10), multiplied by the routing weight *before*
  w2; plus the shared expert (no routing weight).
* Compressor / indexer / candidate selection: :429-611; Engram: :296-365.

Numerics. ``Numerics(act_quant=True, bf16=True)`` (``EXACT``) reproduces model.py's simulation:
FP8 1x32 ue8m0 fake-quant of every FP8/FP4-weight linear input and of o before wo_a, bf16
rounding wherever model.py materialises a bf16 tensor. ``W_ONLY`` keeps weight quantization
only (activations float32, no bf16 rounding) -- the right target for kernels that run FP16/BF16
activations. ``kv_format`` selects the KV-storage simulation ('model' = model.py's FP8 window /
FP4 compressed; 'v4' = stock vLLM sm90 584-byte record; 'none'), ``idx_format`` the indexer q/k
simulation ('mxfp4' = model.py, 'fp8' = stock vLLM sm90 default indexer cache, 'none').
``vllm_rounding`` (set in ``VLLM_SM90``) skips the bf16 roundings model.py has but stock vLLM does
not (compressor projection / pooled row, indexer weights and logits; see Numerics).

TP. Weights are loaded either unsharded (``rank=None``: the whole-model math, i.e. the sum over
ranks) or as one stock-vLLM TP rank (``rank=r``; see weights.tp_slice). The functions are
shard-agnostic: with rank weights, row-parallel outputs (wo_b, MoE) are that rank's partial sums
(pre all-reduce). ``decode_layer_ref`` runs on unsharded weights and, with ``rank=r``, also
returns the rank-r partials, so a single-GPU kernel test can compare pre-all-reduce values.

All functions accept tensors on any device; inputs are promoted to float32.
"""

from __future__ import annotations

import contextlib
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .quant import act_quant_sim, bf16_round, fp4_quant_sim, indexer_k_fp8_sim, v4_record_sim
from .weights import TP, CheckpointReader, dequant_fp4, dequant_fp8, engram_layout

# ===================================================================================== config


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 5120
    vocab_size: int = 129280
    n_layers: int = 40
    n_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora_rank: int = 1280
    o_groups: int = 8
    o_lora_rank: int = 1024
    window: int = 128
    n_routed: int = 384
    n_activated: int = 6
    moe_inter: int = 2304
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-20
    hc_mult: int = 4
    hc_iters: int = 20
    hc_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    compress_ratios: tuple[int, ...] = (0, 0, *([2] * 18), *([1] * 20), 0, 0, 0)
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    engram_layer_ids: tuple[int, ...] = (1, 14)
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed: int = 128
    dspark_n_activated: int = 3
    n_mtp_layers: int = 3

    @classmethod
    def from_hf(cls, c: Mapping) -> "ModelConfig":
        c = c.get("text_config", c)
        rs = c.get("rope_scaling") or {}
        return cls(
            dim=c["hidden_size"], vocab_size=c["vocab_size"], n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"], head_dim=c["head_dim"], rope_dim=c["qk_rope_head_dim"],
            q_lora_rank=c["q_lora_rank"], o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
            window=c["sliding_window"], n_routed=c["n_routed_experts"], n_activated=c["num_experts_per_tok"],
            moe_inter=c["moe_intermediate_size"], route_scale=c["routed_scaling_factor"],
            swiglu_limit=c["swiglu_limit"], norm_eps=c["rms_norm_eps"], hc_mult=c["hc_mult"],
            hc_iters=c["hc_sinkhorn_iters"], hc_eps=c["hc_eps"], rope_theta=c["rope_theta"],
            compress_rope_theta=c["compress_rope_theta"], rope_factor=rs.get("factor", 16),
            original_seq_len=rs.get("original_max_position_embeddings", 65536),
            beta_fast=rs.get("beta_fast", 32), beta_slow=rs.get("beta_slow", 1),
            compress_ratios=tuple(c["compress_ratios"]), kv_source_layers=tuple(c["kv_source_layer_ids"]),
            index_source_layers=tuple(c["index_source_layer_ids"]),
            candidate_source_layer=c["candidate_source_layer_id"],
            candidate_topk_blocks=c["candidate_topk_blocks"], candidate_block_size=c["candidate_block_size"],
            index_n_heads=c["index_n_heads"], index_head_dim=c["index_head_dim"], index_topk=c["index_topk"],
            engram_layer_ids=tuple(c["engram_layer_ids"]), dspark_block_size=c["dspark_block_size"],
            dspark_noise_token_id=c["dspark_noise_token_id"],
            dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]),
            dspark_markov_rank=c["dspark_markov_rank"], dspark_n_routed=c["dspark_n_routed_experts"],
            dspark_n_activated=c["dspark_num_experts_per_tok"], n_mtp_layers=c["num_nextn_predict_layers"],
        )  # fmt: skip

    def ratio(self, layer_id: int) -> int:
        return self.compress_ratios[layer_id] if layer_id < len(self.compress_ratios) else 0


CFG = ModelConfig()


@dataclass(frozen=True)
class Numerics:
    act_quant: bool = False  # FP8 1x32 ue8m0 fake-quant of FP8/FP4-weight linear inputs and o
    bf16: bool = False  # round to bf16 where model.py materialises bf16 tensors
    kv_format: str = "model"  # 'model' | 'v4' | 'none'
    idx_format: str = "mxfp4"  # 'mxfp4' | 'fp8' | 'none'
    # Stock vLLM 0.32.0 keeps fp32 at a few points where model.py rounds to bf16 (only matters
    # with bf16=True): the compressor projection (torch.mm(..., out_dtype=fp32), attention.py:954-959)
    # and the ratio-2 pooled row go into RMSNorm unrounded (fused_compress_quant_cache.py:204-222);
    # the indexer weights are scaled in fp32 (deepseek_v4/common/ops/fused_indexer_q.py:197-207)
    # and the q.k / logit reductions are fp32 (DeepGEMM fp8_mqa_logits). True = follow stock there.
    vllm_rounding: bool = False


EXACT = Numerics(act_quant=True, bf16=True, kv_format="model", idx_format="mxfp4")
W_ONLY = Numerics(act_quant=False, bf16=False, kv_format="model", idx_format="mxfp4")
FP32 = Numerics(act_quant=False, bf16=False, kv_format="none", idx_format="none")
VLLM_SM90 = Numerics(act_quant=False, bf16=True, kv_format="v4", idx_format="fp8", vllm_rounding=True)


class _R:
    """Rounding policy helper."""

    def __init__(self, n: Numerics):
        self.n = n

    def bf(self, x: torch.Tensor) -> torch.Tensor:
        return bf16_round(x) if self.n.bf16 else x

    def aq(self, x: torch.Tensor) -> torch.Tensor:
        return act_quant_sim(x, 32) if self.n.act_quant else x


@contextlib.contextmanager
def strict_fp32():
    """Disable TF32 for the duration (GPU references must be true fp32)."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def _f(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def lin_q(x: torch.Tensor, w: torch.Tensor, n: Numerics = W_ONLY) -> torch.Tensor:
    """FP8/FP4-weight linear (model.py linear(): act_quant + fp8/fp4 gemm -> bf16)."""
    r = _R(n)
    return r.bf(r.aq(_f(x)) @ w.t())


def lin_bf16(x: torch.Tensor, w: torch.Tensor, n: Numerics = W_ONLY) -> torch.Tensor:
    """bf16-weight linear (F.linear in bf16, fp32 accumulate, bf16 out)."""
    return _R(n).bf(_f(x) @ w.t())


# ===================================================================================== basics


def rmsnorm(x: torch.Tensor, w: torch.Tensor | None, eps: float = 1e-20, n: Numerics = W_ONLY) -> torch.Tensor:
    """model.py RMSNorm (:281-293): (w * x * rsqrt(mean(x^2) + eps)), cast to the input dtype."""
    x = _f(x)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * _f(w)
    return _R(n).bf(y)


def sinkhorn(comb_logits: torch.Tensor, iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """kernel.py hc_split_sinkhorn comb part: row-softmax + eps, column norm, then (iters-1) x
    (row norm, column norm). [..., hc, hc] -> doubly-stochastic (approximately)."""
    c = torch.softmax(_f(comb_logits), dim=-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return c


def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int = 4,
                      iters: int = 20, eps: float = 1e-6):
    """mixes [..., (2+hc)*hc] -> (pre [..., hc], post [..., hc], comb [..., hc, hc])."""
    m, s, b = _f(mixes), _f(scale), _f(base)
    pre = torch.sigmoid(m[..., :hc] * s[0] + b[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * s[1] + b[hc : 2 * hc])
    logits = m[..., 2 * hc :].unflatten(-1, (hc, hc)) * s[2] + b[2 * hc :].view(hc, hc)
    return pre, post, sinkhorn(logits, iters, eps)


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, *,
             norm_eps: float = 1e-20, hc_eps: float = 1e-6, iters: int = 20):
    """Block.hc_mixes (:950-957): x [T, hc, D] -> (pre, post, comb, mixes)."""
    hc = x.shape[-2]
    xf = _f(x).flatten(-2)
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (xf @ _f(fn).t()) * rs
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)
    return pre, post, comb, mixes


def hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor | None, n: Numerics = W_ONLY) -> torch.Tensor:
    """Block.hc_pre (:959-962): sum_c pre_mix[c] * x[c]; pre_mix None = one-hot copy 0 (layer 0)."""
    if pre_mix is None:
        return _R(n).bf(_f(x[..., 0, :]))
    return _R(n).bf((_f(pre_mix).unsqueeze(-1) * _f(x)).sum(-2))


def hc_pre(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
           pre_mix: torch.Tensor | None = None, *, norm_eps: float = 1e-20, hc_eps: float = 1e-6,
           iters: int = 20, n: Numerics = W_ONLY):
    """mHC pre for one sublayer: returns (y, post, comb, pre).

    (pre, post, comb) are this sublayer's coefficients computed from `x`; `y` is the collapsed
    sublayer input using the *lagged* `pre_mix` (previous sublayer's pre; None = one-hot copy 0),
    exactly like Block.forward. To get the non-lagged collapse pass ``pre_mix=pre`` yourself.
    """
    pre, post, comb, _ = hc_mixes(x, fn, scale, base, norm_eps=norm_eps, hc_eps=hc_eps, iters=iters)
    return hc_collapse(x, pre_mix, n), post, comb, pre


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
            n: Numerics = W_ONLY) -> torch.Tensor:
    """Block.hc_post (:964-968): out[j] = post[j]*y + sum_i comb[i, j] * residual[i].
    Same as vLLM mhc_post_torch (model_executor/kernels/mhc/torch.py:133-145); post may be [T,hc]
    or vLLM's [T,hc,1]."""
    post = _f(post)
    if post.dim() == residual.dim():  # vLLM [T, hc, 1]
        post = post.squeeze(-1)
    out = post.unsqueeze(-1) * _f(y).unsqueeze(-2) + torch.einsum("...ij,...id->...jd", _f(comb), _f(residual))
    return _R(n).bf(out)


# ===================================================================================== RoPE


@lru_cache(maxsize=16)
def _inv_freq(rope_dim: int, base: float, original_seq_len: int, factor: float, beta_fast: int,
              beta_slow: int) -> torch.Tensor:
    """precompute_freqs_cis (:368-389) frequencies, float32, YaRN when original_seq_len > 0."""
    dim = rope_dim
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_inv_freq(compress_ratio: int, cfg: ModelConfig = CFG) -> torch.Tensor:
    if compress_ratio:
        return _inv_freq(cfg.rope_dim, cfg.compress_rope_theta, cfg.original_seq_len, cfg.rope_factor,
                         cfg.beta_fast, cfg.beta_slow)
    return _inv_freq(cfg.rope_dim, cfg.rope_theta, 0, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)


def rope_cos_sin(positions: torch.Tensor, compress_ratio: int, cfg: ModelConfig = CFG):
    """(cos, sin) [*positions.shape, rope_dim/2] float32; angle = fp32(pos) * fp32(freq)."""
    f = rope_inv_freq(compress_ratio, cfg).to(positions.device)
    ang = positions.to(torch.float32).unsqueeze(-1) * f
    return torch.cos(ang), torch.sin(ang)


def rope_interleaved(x: torch.Tensor, positions: torch.Tensor, compress_ratio: int = 0, *,
                     cfg: ModelConfig = CFG, inverse: bool = False) -> torch.Tensor:
    """Rotate the last cfg.rope_dim (64) dims of x as adjacent (even, odd) pairs (GPT-J style,
    vLLM is_neox_style=False). compress_ratio 0 -> theta 10000 plain; >0 -> theta 160000 + YaRN.

    x [T, ..., D], positions [T]. Returns a new float32 tensor (the NoPE part is copied).
    """
    rd = cfg.rope_dim
    x = _f(x).clone()
    cos, sin = rope_cos_sin(positions, compress_ratio, cfg)
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (rd // 2,)
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    r = x[..., -rd:].unflatten(-1, (rd // 2, 2))
    e, o = r[..., 0], r[..., 1]
    x[..., -rd:] = torch.stack([e * cos - o * sin, e * sin + o * cos], dim=-1).flatten(-2)
    return x


# ===================================================================================== weights


class ExpertSource:
    """Routed-expert weights provider: get(e) -> (w1 [I, D], w3 [I, D], w2 [D, I]) float32."""

    n_experts: int

    def get(self, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CheckpointExperts(ExpertSource):
    """Lazy experts straight from the checkpoint (TP slice for `rank`, or full for rank=None)."""

    def __init__(self, reader: CheckpointReader, prefix: str, n_experts: int, rank: int | None = None,
                 tp: int = TP, device="cpu", cache_size: int = 0):
        self.reader, self.prefix, self.n_experts = reader, prefix, n_experts
        self.rank, self.tp, self.device = rank, tp, device
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self.cache_size = cache_size

    def raw(self, e: int) -> dict[str, torch.Tensor]:
        p = f"{self.prefix}.{e}"
        return {f"{w}.{k}": self.reader.get_tp(f"{p}.{w}.{k}", self.rank, self.tp)
                for w in ("w1", "w2", "w3") for k in ("weight", "scale")}

    def get(self, e: int):
        if e in self._cache:
            self._cache.move_to_end(e)
            return self._cache[e]
        r = self.raw(e)
        d = self.device
        out = tuple(dequant_fp4(r[f"{w}.weight"].to(d), r[f"{w}.scale"].to(d)) for w in ("w1", "w3", "w2"))
        if self.cache_size:
            self._cache[e] = out
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return out


class RawExperts(ExpertSource):
    """Experts in the ds41h raw layout (docs/DESIGN.md "Weight layouts"):
    w13 [E, 2I, D/2] uint8 (rows 0..I-1 = w1, I..2I-1 = w3), s13 [E, 2I, D/32] ue8m0 bytes,
    w2 [E, D, I/2] uint8, s2 [E, D, I/32]."""

    def __init__(self, w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor):
        self.w13, self.s13, self.w2, self.s2 = w13, s13, w2, s2
        self.n_experts = w13.shape[0]

    def get(self, e: int):
        w13 = dequant_fp4(self.w13[e], self.s13[e])
        i = w13.shape[0] // 2
        return w13[:i], w13[i:], dequant_fp4(self.w2[e], self.s2[e])


class ShardedExperts(ExpertSource):
    """TP view (576-row/col slice) of an unsharded ExpertSource."""

    def __init__(self, src: ExpertSource, rank: int, tp: int = TP):
        self.src, self.rank, self.tp, self.n_experts = src, rank, tp, src.n_experts

    def get(self, e: int):
        w1, w3, w2 = self.src.get(e)
        i = w1.shape[0] // self.tp
        sl = slice(self.rank * i, (self.rank + 1) * i)
        return w1[sl], w3[sl], w2[:, sl]


@dataclass
class LayerWeights:
    """Dequantized (float32) weights of one decoder layer (or DSpark block) for one TP rank
    (rank=None: unsharded). Shapes are per-rank. Optional parts are None when absent."""

    layer_id: int
    rank: int | None
    tp: int
    compress_ratio: int
    is_kv_source: bool
    is_index_source: bool
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    wq_a: torch.Tensor  # [1280, D]
    q_norm: torch.Tensor
    wq_b: torch.Tensor  # [H_local*512, 1280]
    wkv: torch.Tensor  # [512, D]
    kv_norm: torch.Tensor
    attn_sink: torch.Tensor  # [H_local]
    wo_a: torch.Tensor  # [G_local, 1024, 4096]
    wo_b: torch.Tensor  # [D, G_local*1024]
    gate_w: torch.Tensor  # [E, D]
    gate_bias: torch.Tensor  # [E]
    sh_w1: torch.Tensor  # [I_local, D]
    sh_w3: torch.Tensor
    sh_w2: torch.Tensor  # [D, I_local]
    experts: ExpertSource | None
    comp_wkv: torch.Tensor | None = None  # [512, D]
    comp_wgate: torch.Tensor | None = None  # [512, D] (ratio 2 only)
    comp_norm: torch.Tensor | None = None
    idx_wq_b: torch.Tensor | None = None  # [32*128, 1280] (replicated in vLLM)
    idx_weights_proj: torch.Tensor | None = None  # [32, D]
    idx_wk: torch.Tensor | None = None  # [128, 512] (KV sources only)
    idx_k_norm: torch.Tensor | None = None
    eng_wkv: torch.Tensor | None = None  # [25600, 6144] (unsharded; vLLM all-gathers the output)
    eng_q: torch.Tensor | None = None  # [4, D]
    eng_k: torch.Tensor | None = None
    gate_bias_vl: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_local_heads(self) -> int:
        return self.wq_b.shape[0] // 512

    @property
    def n_local_groups(self) -> int:
        return self.wo_a.shape[0]

    def to(self, device) -> "LayerWeights":
        kw = {}
        for k, v in self.__dict__.items():
            kw[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        out = LayerWeights(**kw)
        if isinstance(self.experts, CheckpointExperts):
            ex = CheckpointExperts(self.experts.reader, self.experts.prefix, self.experts.n_experts,
                                   self.experts.rank, self.experts.tp, device, self.experts.cache_size)
            out.experts = ex
        return out

    def shard(self, rank: int, tp: int = TP) -> "LayerWeights":
        """TP-rank view of unsharded weights, mirroring weights.tp_slice (32-block aligned, so
        slicing the dequantized tensor equals dequantizing the TP slice)."""
        if self.rank is not None:
            raise ValueError("already sharded")

        def rows(t):
            n = t.shape[0] // tp
            return t[rank * n : (rank + 1) * n]

        def cols(t):
            n = t.shape[1] // tp
            return t[:, rank * n : (rank + 1) * n]

        return replace(
            self, rank=rank, tp=tp, wq_b=rows(self.wq_b), attn_sink=rows(self.attn_sink),
            wo_a=rows(self.wo_a), wo_b=cols(self.wo_b), sh_w1=rows(self.sh_w1), sh_w3=rows(self.sh_w3),
            sh_w2=cols(self.sh_w2), experts=None if self.experts is None else ShardedExperts(self.experts, rank, tp),
        )  # fmt: skip


def load_layer_weights(reader: CheckpointReader, layer_id: int, rank: int | None = None, tp: int = TP,
                       device="cpu", cfg: ModelConfig | None = None, mtp: int | None = None,
                       experts: str = "lazy") -> LayerWeights:
    """Load + dequantize one decoder layer (``mtp=i``: DSpark block mtp.i) for `rank` (None = full).

    experts: 'lazy' (CheckpointExperts, dequantized on demand, nothing resident), 'none'.
    """
    cfg = cfg or ModelConfig.from_hf(reader.config)
    pre = f"mtp.{mtp}." if mtp is not None else f"layers.{layer_id}."
    lid = cfg.n_layers + mtp if mtp is not None else layer_id
    g = lambda n: reader.get_tp(pre + n, rank, tp)  # noqa: E731

    def fp8(n):
        return dequant_fp8(g(n + ".weight").to(device), g(n + ".scale").to(device))

    def f32(n):
        return g(n).to(device).float()

    ratio = cfg.ratio(lid)
    is_backbone = mtp is None
    kv_src = is_backbone and lid in cfg.kv_source_layers
    idx_src = is_backbone and lid in cfg.index_source_layers
    wo_a = fp8("attn.wo_a")
    w = LayerWeights(
        layer_id=lid, rank=rank, tp=tp, compress_ratio=ratio, is_kv_source=kv_src, is_index_source=idx_src,
        attn_norm=f32("attn_norm.weight"), ffn_norm=f32("ffn_norm.weight"),
        hc_attn_fn=f32("hc_attn_fn"), hc_attn_scale=f32("hc_attn_scale"), hc_attn_base=f32("hc_attn_base"),
        hc_ffn_fn=f32("hc_ffn_fn"), hc_ffn_scale=f32("hc_ffn_scale"), hc_ffn_base=f32("hc_ffn_base"),
        wq_a=fp8("attn.wq_a"), q_norm=f32("attn.q_norm.weight"), wq_b=fp8("attn.wq_b"),
        wkv=fp8("attn.wkv"), kv_norm=f32("attn.kv_norm.weight"), attn_sink=f32("attn.attn_sink"),
        wo_a=wo_a.view(-1, cfg.o_lora_rank, wo_a.shape[-1]), wo_b=fp8("attn.wo_b"),
        gate_w=f32("ffn.gate.weight"), gate_bias=f32("ffn.gate.bias"),
        sh_w1=fp8("ffn.shared_experts.w1"), sh_w3=fp8("ffn.shared_experts.w3"), sh_w2=fp8("ffn.shared_experts.w2"),
        experts=None,
    )  # fmt: skip
    if pre + "ffn.gate.bias_vl" in reader:
        w.gate_bias_vl = f32("ffn.gate.bias_vl")
    n_exp = w.gate_w.shape[0]
    if experts == "lazy":
        w.experts = CheckpointExperts(reader, pre + "ffn.experts", n_exp, rank, tp, device)
    elif experts != "none":
        raise ValueError(experts)
    if kv_src:
        w.comp_wkv = f32("attn.compressor.wkv.weight")
        w.comp_norm = f32("attn.compressor.norm.weight")
        if ratio > 1:
            w.comp_wgate = f32("attn.compressor.wgate.weight")
    if idx_src:
        w.idx_wq_b = fp8("attn.indexer.wq_b")
        w.idx_weights_proj = f32("attn.indexer.weights_proj.weight")
        if kv_src:
            w.idx_wk = f32("attn.indexer.wk.weight")
            w.idx_k_norm = f32("attn.indexer.k_norm.weight")
    if pre + "engram.wkv.weight" in reader:
        # vLLM column-shards wkv but all-gathers its output (gather_output=True), so the math
        # needs the full matrix; load it unsharded regardless of rank.
        w.eng_wkv = dequant_fp8(reader.get(pre + "engram.wkv.weight").to(device),
                                reader.get(pre + "engram.wkv.scale").to(device))
        w.eng_q = reader.get(pre + "engram.q_weight").to(device).float()
        w.eng_k = reader.get(pre + "engram.k_weight").to(device).float()
    return w


# ===================================================================================== attention


def kv_store(kv: torch.Tensor, kind: str, n: Numerics) -> torch.Tensor:
    """Storage simulation of post-RoPE KV rows. kind: 'window' | 'compressed'."""
    if n.kv_format == "none":
        return _R(n).bf(kv)
    if n.kv_format == "v4":  # stock vLLM sm90: both caches use the 584 B V4 record
        return v4_record_sim(kv)
    if n.kv_format == "model":
        if kind == "window":  # Attention._window_kv :706-708 (fp8 1x32 over all 512 dims)
            return act_quant_sim(_R(n).bf(kv), 32)
        return fp4_quant_sim(_R(n).bf(kv), 16, "e4m3")  # _compress_kv :760-761
    raise ValueError(n.kv_format)


def idx_store(x: torch.Tensor, n: Numerics) -> torch.Tensor:
    """Indexer q/k simulation (model.py: MXFP4 1x32 ue8m0, :546, :553)."""
    if n.idx_format == "none":
        return _R(n).bf(x)
    if n.idx_format == "mxfp4":
        return fp4_quant_sim(_R(n).bf(x), 32, "ue8m0")
    if n.idx_format == "fp8":
        return indexer_k_fp8_sim(x)
    raise ValueError(n.idx_format)


def q_kv_proj_ref(W: LayerWeights, a: torch.Tensor, positions: torch.Tensor, *, cfg: ModelConfig = CFG,
                  n: Numerics = W_ONLY) -> dict:
    """Attention input projections for normalized input a [T, D] (Attention.forward :771-773,
    _window_kv :701-708). Returns qr [T,1280] (q_norm output), q [T,H_local,512] (post-RoPE),
    kv_raw [T,512] (post-norm, post-RoPE, unquantized), kv [T,512] (stored form)."""
    r = _R(n)
    ratio = W.compress_ratio
    qa = lin_q(a, W.wq_a, n)
    qr = rmsnorm(qa, W.q_norm, cfg.norm_eps, n)
    q = lin_q(qr, W.wq_b, n).unflatten(-1, (-1, cfg.head_dim))
    q = r.bf(rope_interleaved(q, positions, ratio, cfg=cfg))
    kvp = lin_q(a, W.wkv, n)
    kvn = rmsnorm(kvp, W.kv_norm, cfg.norm_eps, n)
    kv_raw = r.bf(rope_interleaved(kvn, positions, ratio, cfg=cfg))
    return {"q_a": qa, "qr": qr, "q": q, "kv_proj": kvp, "kv_raw": kv_raw, "kv": kv_store(kv_raw, "window", n)}


def gather_kv(kv: torch.Tensor, idx: torch.Tensor):
    """kv [N, D], idx [T, K] (int, -1 = empty) -> (rows [T, K, D], valid [T, K])."""
    valid = idx >= 0
    rows = kv[idx.clamp_min(0).long()]
    return rows * valid.unsqueeze(-1), valid


def sparse_attn_ref(q: torch.Tensor, kv_rows: torch.Tensor, sink: torch.Tensor, valid: torch.Tensor,
                    scale: float | None = None) -> torch.Tensor:
    """Sparse MQA with attention sink (kernel.py sparse_attn :311-389), exact softmax in fp32.

    q [T, H, D], kv_rows [T, K, D] (K = V), sink [H], valid [T, K] bool -> o [T, H, D].
    o = sum_j e^{s_j} v_j / (sum_j e^{s_j} + e^{sink}); a row with no valid key gives 0.
    """
    d = q.shape[-1]
    scale = d**-0.5 if scale is None else scale
    qf, kf = _f(q), _f(kv_rows)
    s = torch.einsum("thd,tkd->thk", qf, kf) * scale
    s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sk = _f(sink).view(1, -1, 1).expand(s.shape[0], -1, 1)
    m = torch.maximum(s.amax(-1, keepdim=True), sk)
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(sk - m)
    o = torch.einsum("thk,tkd->thd", p / denom, kf)
    return torch.where(valid.any(-1).view(-1, 1, 1), o, torch.zeros_like(o))


def window_keys(positions: torch.Tensor, window_kv_hist: torch.Tensor, kv_step: torch.Tensor,
                window: int = 128):
    """Window keys of a decode step (causal, ring of the last `window` positions).

    positions [T] consecutive (p0..p0+T-1); window_kv_hist [P, D] = stored window KV of positions
    [p0-P, p0); kv_step [T, D] stored KV of the step. Returns (rows [T, window, D], valid [T, window]).
    """
    T = positions.shape[0]
    p0 = int(positions[0])
    P = window_kv_hist.shape[0]
    allkv = torch.cat([window_kv_hist.to(kv_step.dtype), kv_step], 0)  # rows for positions p0-P .. p0+T-1
    j = torch.arange(window, device=positions.device)
    pos_k = positions.view(T, 1) - (window - 1) + j.view(1, window)  # [T, window]
    row = pos_k - (p0 - P)
    valid = (pos_k >= 0) & (row >= 0)
    if bool(((pos_k >= 0) & (row < 0)).any()):
        raise ValueError("window_kv_hist too short for the window of this step")
    return allkv[row.clamp_min(0)] * valid.unsqueeze(-1), valid


def o_proj_ref(W: LayerWeights, o: torch.Tensor, positions: torch.Tensor, *, cfg: ModelConfig = CFG,
               n: Numerics = W_ONLY) -> dict:
    """Inverse RoPE + grouped wo_a + wo_b (Attention.forward :781-791).

    o [T, H_local, 512] -> o_rot (after inverse RoPE), z [T, G_local*1024], y [T, D] (partial sum
    when W is a TP rank)."""
    r = _R(n)
    o_rot = r.bf(rope_interleaved(o, positions, W.compress_ratio, cfg=cfg, inverse=True))
    og = r.aq(o_rot.reshape(o.shape[0], W.n_local_groups, -1))  # [T, G, 4096]
    z = r.bf(torch.einsum("tgd,grd->tgr", og, W.wo_a)).flatten(1)
    y = lin_q(z, W.wo_b, n)
    return {"o_rot": o_rot, "z": z, "y": y}


# ----------------------------------------------------------------------------- compressor


def compressor_proj_ref(W: LayerWeights, a: torch.Tensor, n: Numerics = W_ONLY) -> torch.Tensor:
    """Compressor projection per token: [T, 512] (ratio 1, bf16 linear) or [T, 1024] = [kv | score]
    (ratio 2, fp32 linear; Compressor.forward :458-466). vLLM keeps these rows in its fp32 state."""
    if W.compress_ratio == 1:
        # model.py: bf16 Linear (bf16 out); stock vLLM: torch.mm(out_dtype=fp32) (attention.py:954-959)
        return _f(a) @ W.comp_wkv.t() if n.vllm_rounding else lin_bf16(a, W.comp_wkv, n)
    x = _f(a)
    return torch.cat([x @ W.comp_wkv.t(), x @ W.comp_wgate.t()], -1)


def compressor_pool_ref(rows: torch.Tensor, ratio: int, norm_w: torch.Tensor, cfg: ModelConfig = CFG,
                        n: Numerics = W_ONLY) -> torch.Tensor:
    """rows [G*ratio, 512*(1 or 2)] of complete groups -> latents [G, 512] (pre-RoPE, normed).

    ratio 2: per-channel softmax over the group's scores, weighted sum of kv (:473-482), cast to bf16
    then RMSNorm (:485). ratio 1: RMSNorm of the projection (:461-462)."""
    if ratio == 1:
        return rmsnorm(rows, norm_w, cfg.norm_eps, n)
    kv, score = rows[..., :512].unflatten(0, (-1, ratio)), rows[..., 512:].unflatten(0, (-1, ratio))
    pooled = (kv * score.softmax(dim=1)).sum(1)
    if not n.vllm_rounding:  # model.py norm(kv.to(bf16)); stock normalizes the fp32 pooled row
        pooled = _R(n).bf(pooled)
    return rmsnorm(pooled, norm_w, cfg.norm_eps, n)


def compressed_rows_ref(latent: torch.Tensor, group_ids: torch.Tensor, ratio: int, cfg: ModelConfig = CFG,
                        n: Numerics = W_ONLY) -> torch.Tensor:
    """Latents -> stored compressed KV rows: RoPE at the group's first position j*ratio, then the
    storage format (:752-762)."""
    rot = _R(n).bf(rope_interleaved(latent, group_ids * ratio, ratio, cfg=cfg))
    return kv_store(rot, "compressed", n)


def index_keys_ref(W: LayerWeights, latent: torch.Tensor, group_ids: torch.Tensor, ratio: int,
                   cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> torch.Tensor:
    """Index keys k = k_norm(wk latent), RoPE at j*ratio on the last 64 of 128 dims, quant
    (Indexer.forward :537-547; vLLM indexer_k_norm_rope_store)."""
    k = rmsnorm(lin_bf16(latent, W.idx_wk, n), W.idx_k_norm, cfg.norm_eps, n)
    k = _R(n).bf(rope_interleaved(k, group_ids * ratio, ratio, cfg=cfg))
    return idx_store(k, n)


def compress_history_ref(W: LayerWeights, a_all: torch.Tensor, cfg: ModelConfig = CFG, n: Numerics = W_ONLY):
    """Run a KV-source layer's compressor over a whole sequence of normalized inputs a_all
    [P, D] (positions 0..P-1). Returns dict(latent [C,512], kv [C,512] stored rows,
    index_k [C,128] or None, proj [P, 512|1024]) with C = P // ratio."""
    ratio = W.compress_ratio
    proj = compressor_proj_ref(W, a_all, n)
    C = a_all.shape[0] // ratio
    latent = compressor_pool_ref(proj[: C * ratio], ratio, W.comp_norm, cfg, n)
    gid = torch.arange(C, device=a_all.device)
    out = {"proj": proj, "latent": latent, "kv": compressed_rows_ref(latent, gid, ratio, cfg, n), "index_k": None}
    if W.idx_wk is not None:
        out["index_k"] = index_keys_ref(W, latent, gid, ratio, cfg, n)
    return out


# ----------------------------------------------------------------------------- indexer


def select_candidate_blocks(logits: torch.Tensor, compress_lens, topk_blocks: int, block_size: int) -> torch.Tensor:
    """model.py select_candidate_blocks (:584-611), verbatim semantics. logits [..., n] with
    unreachable positions at -inf; compress_lens int or broadcastable [..., 1]."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def indexer_ref(W: LayerWeights, a: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor,
                index_k: torch.Tensor, *, cfg: ModelConfig = CFG, n: Numerics = W_ONLY,
                candidates: torch.Tensor | None = None) -> dict:
    """Indexer scores + top-k for decode tokens (Indexer.forward :527-581, multi-token form).

    a [T, D] (attention input after attn_norm), qr [T, 1280], positions [T], index_k [C, 128] stored
    keys of all compressed positions visible to the latest token. All 32 heads (vLLM replicates the
    indexer on every rank). Token t sees compressed positions < (p_t + 1) // ratio.
    Returns dict(q, weights, scores [T, C] (-inf where unreachable/non-candidate), raw_scores,
    compress_lens [T], candidates [T, C] (candidate-source layer only), topk [T, K] (sorted, -1 pad)).
    """
    r = _R(n)
    # model.py materialises weights / q.k / logits in bf16; stock vLLM keeps them fp32
    rs = (lambda t: t) if n.vllm_rounding else r.bf  # noqa: E731
    ratio, H, Dh = W.compress_ratio, cfg.index_n_heads, cfg.index_head_dim
    q = lin_q(qr, W.idx_wq_b, n).unflatten(-1, (H, Dh))
    q = idx_store(r.bf(rope_interleaved(q, positions, ratio, cfg=cfg)), n)
    wts = rs(lin_bf16(a, W.idx_weights_proj, n) * (Dh**-0.5 * H**-0.5))
    s = rs(torch.einsum("thd,cd->thc", q, _f(index_k)))
    raw = rs((s.relu() * wts.unsqueeze(-1)).sum(1))
    C = index_k.shape[0]
    lens = (positions.long() + 1) // ratio  # [T]
    scores = raw.masked_fill(torch.arange(C, device=a.device).view(1, C) >= lens.view(-1, 1), float("-inf"))
    out = {"q": q, "weights": wts, "raw_scores": raw, "compress_lens": lens, "candidates": None}
    if W.layer_id == cfg.candidate_source_layer:
        out["candidates"] = select_candidate_blocks(scores, lens.view(-1, 1), cfg.candidate_topk_blocks,
                                                    cfg.candidate_block_size)
    elif candidates is not None and 0 <= cfg.candidate_source_layer < W.layer_id:
        scores = scores.masked_fill(~candidates[:, :C], float("-inf"))
    out["scores"] = scores
    k = min(cfg.index_topk, int(lens.max()))
    idx = scores.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    out["topk"] = torch.where(idx < lens.view(-1, 1), idx, -1)
    return out


# ----------------------------------------------------------------------------- attention block


@dataclass
class AttnContext:
    """Cache state an attention layer reads in a decode step (step positions p0..p0+T-1).

    window_kv:     [P, 512] stored window KV of positions [p0-P, p0) (P >= min(p0, window-1)).
    compressed_kv: [C, 512] stored compressed rows. KV-source layers: groups completed before the
                   step (C = p0 // ratio); the reference appends the step's own groups. Other
                   compressed layers: everything the source wrote, incl. this step's groups.
    comp_state:    KV-source ratio-2 layers when p0 is odd: [1, 1024] fp32 [kv | score] row of
                   position p0-1 (vLLM CompressorStateCache).
    index_k:       [C, 128] stored index keys (same convention as compressed_kv) for index sources.
    topk_idxs:     [T, K] compressed indices (-1 empty) for compressed layers that are not index
                   sources (published by their index source).
    candidates:    [T, C] bool candidate mask (index sources after the candidate source layer).
    """

    window_kv: torch.Tensor
    compressed_kv: torch.Tensor | None = None
    comp_state: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def attention_ref(W: LayerWeights, a: torch.Tensor, positions: torch.Tensor, ctx: AttnContext, *,
                  cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> dict:
    """One attention sublayer for normalized input a [T, D] (Attention.forward :766-791)."""
    r = _R(n)
    out = q_kv_proj_ref(W, a, positions, cfg=cfg, n=n)
    rows, valid = window_keys(positions, ctx.window_kv, out["kv"], cfg.window)
    ratio = W.compress_ratio
    if ratio:
        comp_kv = ctx.compressed_kv
        index_k = ctx.index_k
        if W.is_kv_source:
            p0, pT = int(positions[0]), int(positions[-1])
            g0, g1 = p0 // ratio, (pT + 1) // ratio  # groups completed in this step: [g0, g1)
            proj = compressor_proj_ref(W, a, n)
            if ratio > 1 and p0 % ratio:
                if ctx.comp_state is None:
                    raise ValueError("comp_state needed: step starts inside a compression group")
                proj = torch.cat([_f(ctx.comp_state).to(proj.device), proj], 0)
            ncomp = (g1 - g0) * ratio
            out["comp_proj"] = proj
            if g1 > g0:
                lat = compressor_pool_ref(proj[:ncomp], ratio, W.comp_norm, cfg, n)
                gid = torch.arange(g0, g1, device=a.device)
                new_kv = compressed_rows_ref(lat, gid, ratio, cfg, n)
                out["comp_latent"], out["comp_new_kv"] = lat, new_kv
                comp_kv = torch.cat([comp_kv[:g0].to(new_kv.dtype), new_kv], 0) if comp_kv is not None else new_kv
                if W.idx_wk is not None:
                    nk = index_keys_ref(W, lat, gid, ratio, cfg, n)
                    out["index_k_new"] = nk
                    index_k = torch.cat([index_k[:g0].to(nk.dtype), nk], 0) if index_k is not None else nk
            out["compressed_kv"], out["index_k"] = comp_kv, index_k
        if W.is_index_source:
            if comp_kv is None or comp_kv.shape[0] == 0 or (positions.max() + 1) // ratio == 0:
                topk = torch.empty(a.shape[0], 0, dtype=torch.long, device=a.device)
            else:
                ix = indexer_ref(W, a, out["qr"], positions, index_k, cfg=cfg, n=n, candidates=ctx.candidates)
                out["indexer"] = ix
                topk = ix["topk"]
        else:
            topk = ctx.topk_idxs
        out["topk_idxs"] = topk
        if topk is not None and topk.numel():
            crow, cval = gather_kv(comp_kv, topk)
            rows, valid = torch.cat([rows, crow.to(rows.dtype)], 1), torch.cat([valid, cval], 1)
    out["kv_rows"], out["kv_valid"] = rows, valid
    o = r.bf(sparse_attn_ref(out["q"], rows, W.attn_sink, valid, cfg.head_dim**-0.5))
    out["o"] = o
    out.update(o_proj_ref(W, o, positions, cfg=cfg, n=n))
    return out


# ===================================================================================== MoE


def gate_ref(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, topk: int = 6,
             route_scale: float = 1.5, bias_vl: torch.Tensor | None = None,
             image_mask: torch.Tensor | None = None):
    """Gate (:811-829): sqrtsoftplus scores (fp32), noaux_tc bias for selection only,
    weights = scores[sel] / (sum + 1e-20) * route_scale. Returns (weights [T,k], idx [T,k], scores)."""
    scores = F.softplus(_f(x) @ _f(gate_w).t()).sqrt()
    b = _f(bias)
    if image_mask is not None and bias_vl is not None:
        b = torch.where(image_mask.unsqueeze(-1), _f(bias_vl), b)
    idx = (scores + b).topk(topk, dim=-1).indices
    wts = scores.gather(-1, idx)
    if topk > 1:
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
    return wts * route_scale, idx, scores


def expert_ffn_ref(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor,
                   weight: torch.Tensor | None = None, limit: float = 10.0, n: Numerics = W_ONLY) -> torch.Tensor:
    """Expert.forward (:843-853): routing weight applied before w2; clamp-SwiGLU at `limit`."""
    r = _R(n)
    xin = r.aq(_f(x))
    gte = r.bf(xin @ w1.t())
    up = r.bf(xin @ w3.t())
    if limit > 0:
        up = up.clamp(-limit, limit)
        gte = gte.clamp(max=limit)
    h = F.silu(gte) * up
    if weight is not None:
        h = _f(weight) * h
    h = r.aq(r.bf(h))
    return r.bf(h @ w2.t())


def moe_ref(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, experts: ExpertSource,
            shared: Sequence[torch.Tensor] | None, *, topk: int = 6, route_scale: float = 1.5,
            swiglu_limit: float = 10.0, n: Numerics = W_ONLY, routing=None, return_all: bool = False):
    """MoE.forward (:891-906) for x [T, D] (after ffn_norm). Returns y [T, D] float32, or with
    return_all=True a dict(y, routed, shared, weights, idx, scores).

    experts: ExpertSource (unsharded -> full output; a TP rank -> that rank's partial).
    shared: (w1, w3, w2) of the shared expert (sharded the same way as experts), or None.
    routing: optional precomputed (weights, idx) (e.g. to reuse the full-model routing for a TP
    rank's partial). y = bf16(routed_sum + shared) in EXACT mode, like model.py's .type_as(x).
    """
    if routing is None:
        wts, idx, scores = gate_ref(x, gate_w, bias, topk, route_scale)
    else:
        (wts, idx), scores = routing, None
    xf = _f(x)
    routed = torch.zeros_like(xf)
    for e in torch.unique(idx).tolist():
        tok, slot = torch.where(idx == e)
        w1, w3, w2 = experts.get(e)
        routed.index_add_(0, tok, expert_ffn_ref(xf[tok], w1, w3, w2, wts[tok, slot, None], swiglu_limit, n))
        del w1, w3, w2
    sh = expert_ffn_ref(xf, *shared, None, swiglu_limit, n) if shared is not None else torch.zeros_like(xf)
    y = _R(n).bf(routed + sh)
    if not return_all:
        return y
    return {"y": y, "routed": routed, "shared": sh, "weights": wts, "idx": idx, "scores": scores}


def routed_experts_ref(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, w13: torch.Tensor,
                       s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, limit: float = 10.0,
                       n: Numerics = W_ONLY) -> torch.Tensor:
    """Routed-expert sum only (no shared expert, no all-reduce) on the DESIGN.md raw layout of one
    TP rank (w13 [E, 2I, D/2], s13 [E, 2I, D/32], w2 [E, D, I/2], s2 [E, D, I/32]) with given
    routing (weights [T, k], ids [T, k]) -> float32 [T, D] (that rank's pre-all-reduce partial)."""
    return moe_ref(x, None, None, RawExperts(w13, s13, w2, s2), None, swiglu_limit=limit, n=n,
                   routing=(_f(topk_weights), topk_ids.long()))


# ===================================================================================== Engram


def engram_ref(X: torch.Tensor, rows: torch.Tensor, wkv: torch.Tensor, q_w: torch.Tensor, k_w: torch.Tensor,
               *, eps: float = 1e-20, clamp_value: float = 1e-6, token_mask: torch.Tensor | None = None,
               n: Numerics = W_ONLY):
    """Engram.forward (:350-365). X [T, hc, D]; rows [T, 24, 256] dequantized table rows (bf16
    values); wkv [(hc+1)*D, 24*256]. Returns (X_out, dict(kv, key, value, gate))."""
    T, hc, D = X.shape
    kv = lin_q(_f(rows).flatten(-2), wkv, n)
    key, value = kv[:, : hc * D].unflatten(-1, (hc, D)), kv[:, hc * D :]
    wqk = _f(q_w) * _f(k_w)
    h = _f(X)
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * wqk * key).sum(-1) * rstd * D**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    out = _R(n).bf(h + gate.unsqueeze(-1) * value.unsqueeze(-2))
    return out, {"kv": kv, "key": key, "value": value, "gate": gate, "dot": dot}


def engram_lookup(reader: CheckpointReader, layer_id: int, hash_ids: torch.Tensor) -> torch.Tensor:
    """Gather + dequantize Engram table rows (ParallelEngramEmbedding.forward :312-325): FP8 row x
    per-32 ue8m0 scale, cast to bf16. hash_ids [..., 24] global row ids -> [..., 24, 256] float32."""
    from .quant import e4m3_to_float, e8m0_to_float

    p = f"layers.{layer_id}.engram.embed."
    w = reader.gather_rows(p + "weight", hash_ids)
    s = reader.gather_rows(p + "scale", hash_ids)
    v = e4m3_to_float(w).unflatten(-1, (-1, 32)) * e8m0_to_float(s).unsqueeze(-1)
    return bf16_round(v.flatten(-2))


def engram_multipliers(layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """engram.py compute_hash_multipliers (:62-82) -> int64 [n_layers, max_ngram_size]."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        rows.append(torch.tensor(g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1))
    return torch.stack(rows)


def engram_hash_ids(compressed_ids: torch.Tensor, cfg_hf: Mapping, pad_compressed_id: int) -> torch.Tensor:
    """NgramHashState.forward (engram.py :152-184) for a whole sequence from position 0.

    compressed_ids [L] int64 (token_map[input_ids]; -1 marks dead/image tokens). Returns global
    table row ids [L, n_engram_layers, n_hash_cols] (hash % prime + bucket offset)."""
    lay = engram_layout(cfg_hf)
    L = compressed_ids.shape[0]
    pos = torch.arange(L)
    toks, blocked = [], torch.zeros(L, dtype=torch.bool)
    for shift in range(lay.max_ngram_size):
        src = compressed_ids[(pos - shift).clamp_min(0)]
        blocked = blocked | (pos < shift) | (src == -1)
        toks.append(torch.where(blocked, pad_compressed_id, src))
    tokens = torch.stack(toks, -1)  # [L, max_ngram]
    mult = engram_multipliers(lay.layer_ids, lay.max_ngram_size, lay.compressed_vocab_size)
    primes = torch.tensor(lay.primes)  # [n_layers, ngram-1, heads]
    offsets = torch.tensor([lay.offsets(i) for i in range(len(lay.layer_ids))])
    prods = tokens.unsqueeze(1) * mult  # [L, n_layers, max_ngram]
    rolling, hashes = prods[..., 0], []
    for i in range(1, lay.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, prods[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, -1) + offsets


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """engram.py build_compressed_token_map (:17-59), needs the `tokenizers` package."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"  # private-use char (engram.py:27-29)
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])  # fmt: skip
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(tid)
        else:
            normed = normalizer.normalize_str(text)
            key = normed if normed else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


# ===================================================================================== layer


def decode_layer_ref(W: LayerWeights, X: torch.Tensor, positions: torch.Tensor, pre_mix: torch.Tensor | None,
                     ctx: AttnContext, *, cfg: ModelConfig = CFG, n: Numerics = W_ONLY, rank: int | None = None,
                     tp: int = TP, engram_rows: torch.Tensor | None = None) -> dict:
    """One decoder layer for a decode step of T tokens of one sequence at consecutive positions.

    W: unsharded LayerWeights (rank=None). X [T, 4, D]: the residual stream entering the layer
    (after the previous layer's FFN hc_post). pre_mix [T, 4]: previous layer's ffn_pre (None for
    layer 0). ctx: cache state (AttnContext). engram_rows [T, 24, 256] for Engram layers 1/14
    (applied to X first, like vLLM between the previous post and this pre).

    Returns a dict of intermediates (float32), including
      'x_in' (after Engram), 'attn_pre','attn_post','attn_comb', 'a' (attn input after attn_norm),
      'attn' (dict from attention_ref: qr, q, kv, o, z, y, topk_idxs, ...), 'x_mid' (stream after
      attention hc_post), 'ffn_pre','ffn_post','ffn_comb', 'f' (MoE input), 'moe' (dict), 'x_out'.
      'aux': mean over the hc copies of the stream entering the layer (before Engram), [T, D]
        (stock previous_aux with capture_previous_aux: nvidia/model.py:541-542, :585-586,
        kernels/mhc/tilelang_kernels.py:270-287; DSpark main_hidden for layers 37-39).
      'vllm': the stock DeepseekV4DecoderLayer.forward return values for this layer, i.e.
        (x = MoE output [T, D], residual = x_mid [T,4,D], post_mix = ffn_post [T,4,1],
         res_mix = ffn_comb [T,4,4], ffn_pre [T,4], aux [T, D]) -- nvidia/model.py:455-625; the
        next layer computes x_out = mhc_post(x, residual, post_mix, res_mix).
      'rank' (when rank is given): rank-r partials with the *same* routing / KV: q, o, z,
        attn_y (wo_b partial), moe_routed, moe_shared, moe_y (pre all-reduce sums).
    """
    if W.rank is not None:
        raise ValueError("decode_layer_ref needs unsharded weights (rank=None); use rank=... for partials")
    with strict_fp32():
        out: dict = {}
        x_in = _R(n).bf(_f(X))
        out["aux"] = _R(n).bf(x_in.mean(-2))
        if engram_rows is not None:
            if W.eng_wkv is None:
                raise ValueError("engram_rows given for a non-Engram layer")
            x_in, out["engram"] = engram_ref(x_in, engram_rows, W.eng_wkv, W.eng_q, W.eng_k, eps=cfg.norm_eps, n=n)
        out["x_in"] = x_in
        hk = dict(norm_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, iters=cfg.hc_iters)
        a_pre, a_post, a_comb, _ = hc_mixes(x_in, W.hc_attn_fn, W.hc_attn_scale, W.hc_attn_base, **hk)
        out.update(attn_pre=a_pre, attn_post=a_post, attn_comb=a_comb)
        out["attn_in"] = hc_collapse(x_in, pre_mix, n)
        a = rmsnorm(out["attn_in"], W.attn_norm, cfg.norm_eps, n)
        out["a"] = a
        att = attention_ref(W, a, positions, ctx, cfg=cfg, n=n)
        out["attn"] = att
        x_mid = hc_post(att["y"], x_in, a_post, a_comb, n)
        out["x_mid"] = x_mid
        f_pre, f_post, f_comb, _ = hc_mixes(x_mid, W.hc_ffn_fn, W.hc_ffn_scale, W.hc_ffn_base, **hk)
        out.update(ffn_pre=f_pre, ffn_post=f_post, ffn_comb=f_comb)
        out["ffn_in"] = hc_collapse(x_mid, a_pre, n)
        f = rmsnorm(out["ffn_in"], W.ffn_norm, cfg.norm_eps, n)
        out["f"] = f
        nact = cfg.n_activated if W.gate_w.shape[0] == cfg.n_routed else cfg.dspark_n_activated
        moe = moe_ref(f, W.gate_w, W.gate_bias, W.experts, (W.sh_w1, W.sh_w3, W.sh_w2), topk=nact,
                      route_scale=cfg.route_scale, swiglu_limit=cfg.swiglu_limit, n=n, return_all=True)
        out["moe"] = moe
        out["x_out"] = hc_post(moe["y"], x_mid, f_post, f_comb, n)
        out["vllm"] = (moe["y"], x_mid, f_post.unsqueeze(-1), f_comb, f_pre, out["aux"])
        if rank is not None:
            Wr = W.shard(rank, tp)
            hl = Wr.n_local_heads
            o_l = att["o"][:, rank * hl : (rank + 1) * hl]
            po = o_proj_ref(Wr, o_l, positions, cfg=cfg, n=n)
            mr = moe_ref(f, Wr.gate_w, Wr.gate_bias, Wr.experts, (Wr.sh_w1, Wr.sh_w3, Wr.sh_w2),
                         swiglu_limit=cfg.swiglu_limit, n=n, routing=(moe["weights"], moe["idx"]), return_all=True)
            out["rank"] = {
                "rank": rank, "q": att["q"][:, rank * hl : (rank + 1) * hl], "o": o_l, "o_rot": po["o_rot"],
                "z": po["z"], "attn_y": po["y"], "moe_routed": mr["routed"], "moe_shared": mr["shared"],
                "moe_y": mr["routed"] + mr["shared"],
            }
        return out


def build_attn_context(W: LayerWeights, a_hist: torch.Tensor, positions: torch.Tensor, *, cfg: ModelConfig = CFG,
                       n: Numerics = W_ONLY, a_step: torch.Tensor | None = None,
                       source: LayerWeights | None = None, index_source: LayerWeights | None = None,
                       candidates: torch.Tensor | None = None) -> AttnContext:
    """Synthesize a consistent cache state for a decode step from normalized history inputs.

    a_hist [P, D]: this layer's attention inputs for positions 0..P-1 (P = positions[0]);
    a_step [T, D]: the step's inputs (needed for non-source compressed layers, whose source writes
    this step's groups before they attend). `source` / `index_source`: the KV / index source
    layers' weights (defaults: W itself when W is one). Their compressor/indexer run on the same
    a_hist/a_step (a test convenience: real layers see different inputs).
    """
    with strict_fp32():
        kvh = q_kv_proj_ref(W, a_hist, torch.arange(a_hist.shape[0], device=a_hist.device), cfg=cfg, n=n)["kv"]
        ctx = AttnContext(window_kv=kvh)
        ratio = W.compress_ratio
        if not ratio:
            return ctx
        p0 = int(positions[0])
        src = W if W.is_kv_source else source
        if src is None:
            raise ValueError(f"layer {W.layer_id}: pass `source` (its KV source layer weights)")
        if W.is_kv_source:
            hist = compress_history_ref(src, a_hist, cfg, n)
            ctx.compressed_kv, ctx.index_k = hist["kv"], hist["index_k"]
            if ratio > 1 and p0 % ratio:
                ctx.comp_state = hist["proj"][p0 - p0 % ratio : p0]
            return ctx
        if a_step is None:
            raise ValueError("a_step needed for a non-source compressed layer")
        a_all = torch.cat([a_hist, a_step], 0)
        full = compress_history_ref(src, a_all, cfg, n)
        ctx.compressed_kv = full["kv"]
        if W.is_index_source:
            ctx.index_k = full["index_k"]
            ctx.candidates = candidates
            return ctx
        isrc = index_source
        if isrc is None:
            raise ValueError(f"layer {W.layer_id}: pass `index_source` (its index source layer weights)")
        qr = q_kv_proj_ref(isrc, a_step, positions, cfg=cfg, n=n)["qr"]
        ix = indexer_ref(isrc, a_step, qr, positions, full["index_k"], cfg=cfg, n=n, candidates=candidates)
        ctx.topk_idxs = ix["topk"]
        return ctx
