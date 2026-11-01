"""ds41h 'prefill' component, part 1: dense FP8 linears + wo_a for prefill on sm90.

NUMERICS (docs/EXACTNESS.md): the default paths are LOSSLESS vs stock vLLM 0.32.0 --
bf16 activations into every GEMM, exact 32x32 ue8m0 weight scales, fp32 accumulation (only the summation order
differs). The FP8-activation (W8A8) paths of phase 3 are kept as explicit opt-ins (or DS41H_LOSSY=1).

Stock vLLM 0.32.0 runs every DeepSeek-V4.1 dense FP8 linear through
``ModelOptLinearMethod(QuantSpec(kMxfp8Static, kMxfp8Dynamic), CkptCtx(scale_block_size=(32, 32)))``
(models/deepseek_v41/quant_config.py:221-243). On sm90 the MXFP8 kernel oracle
(kernels/linear/__init__.py:936-988, Humming ahead of Marlin per quantization/utils/humming/priority.py)
picks ``HummingMxfp8LinearKernel``, whose input schema falls back to BF16 activations
(quantization/utils/humming/schema.py ``check_and_fallback_input_schema``: "prefer wna16 for groupwise
weight or groupwise input" unless sm >= 120), i.e. W8A16. The bmm ``wo_a`` has no sm90 MXFP8 kernel at all and gets
``EmulationMxfp8LinearKernel`` (dequantized to a BF16 weight at load, run as ``torch.bmm`` after vLLM's Triton
inverse RoPE writes a bf16 [G, T, 4096] copy of the attention output).

LOSSLESS paths:
  DS41H_PREFILL_LINEAR16 (W8A16, default OFF -- stock Humming W8A16 is as fast at 250 W, see LINEAR16 below): a captured dense linear with >= MIN_M rows dequantizes its checkpoint FP8 weight
      EXACTLY to bf16 (fp8_linear.cu ``dequant_fp8_bf16``: e4m3 * 2^(S-127) is representable in bf16, so this is the
      weight stock's W8A16 kernel multiplies with) into a scratch buffer (~1-2 % of the GEMM's time at M = 8-16K) and
      runs a bf16 x bf16 -> fp32-accumulate GEMM on it (DS41H_PREFILL_LINEAR16_GEMM: ``deepgemm`` = DeepGEMM sm90
      ``bf16_gemm_nt`` (wgmma, TMA, persistent) or ``cublas`` = torch.mm with reduced-precision reductions disabled).
      Same products as stock Humming W8A16, fp32 accumulation, only the summation order differs.
  DS41H_PREFILL_WOA16 (default ON, 1.29x stock at M = 16384): wo_a keeps stock's bf16 weight and bf16 math; the inverse RoPE is applied IN PLACE to the 64
      RoPE dims of the attention output (``inv_rope_bf16_``, stock's fp32 formula, bf16 rounding; the 448 no-RoPE dims
      are unchanged by stock, so the [G, T, 4096] copy is skipped) and the two group GEMMs read the attention output
      through a strided view (DS41H_PREFILL_WOA16_GEMM: deepgemm | cublas).

LOSSY opt-ins (default OFF; ON with DS41H_LOSSY=1; take precedence over the lossless paths when set):

vLLM PR #61875 ("MXFP8 activations run as block-FP8 on Hopper", humming 0.1.18) only touches the
MoE experts (it passes ``activation_key=kMxfp8Dynamic`` to ``Mxfp4MoEMethod`` and maps it to
``kFp8Dynamic128Sym`` on SM90). Its port is ``install_moe_fp8_activations`` below (switch
DS41H_PREFILL_MOE_FP8ACT); it only matters when the ds41h raw MoE (DS41H_MOE) is off. Humming has no
FP8-group-128 x MXFP8-group-32 dense path in 0.1.17 or 0.1.18 (``is_humming_schema_compatible``
rejects input group != weight group unless the weight is MXFP4), so the dense linears need their own
W8A8 path (DS41H_PREFILL_LINEAR=1):

impl "deepgemm" (default): DeepGEMM sm90 ``fp8_gemm_nt`` 1D2D (vendored, vllm.third_party.deep_gemm),
    activations e4m3 with a 1x128 scale (ds41h/kernels/fp8_linear.cu ``act_quant_128``), weights e4m3
    with one fp32 scale per 128x128 block. The checkpoint has one ue8m0 scale per 32x32 block; folding
    16 of them into one 128x128 scale (reference exponent = the block max, each 32x32 sub-block's
    e4m3 values multiplied by 2^(e - e_ref) <= 1 and re-rounded to e4m3) is exact except for values
    that fall below e4m3's normal range after the shift: measured on layers 0/1/2/10/20/39 (all dense
    FP8 weights incl. indexer wq_b / engram wkv) 1e-5..6e-5 of the elements change, each by at most
    half an e4m3 subnormal step of its block (<= 2^-10 * 2^5 = 0.031 in units where the block absmax
    is 448), relative L2 weight error 3e-8..4e-6 -- four orders of magnitude below the e4m3
    activation rounding (~1.5e-2 relative per element). ``fold_block_scales`` returns these counts.
    The wo_a bmm is converted the same way and then runs the stock DeepSeek-V4 Hopper path
    (models/deepseek_v4/nvidia/ops/o_proj.py: ``fused_inv_rope_fp8_quant`` + DeepGEMM
    ``fp8_einsum("bhr,hdr->bhd", recipe=(1, 128, 128))``), which 0.32.0 already selects whenever
    ``wo_a.weight`` is e4m3 (switch DS41H_PREFILL_WOA).
impl "humming": Humming MXFP8 x MXFP8 (activations e4m3 with a ue8m0 scale per 1x32, promotion every
    K=32): numerically the reference recipe (inference/model.py linear -> act_quant(x, 32, "ue8m0") +
    fp8_gemm(block 32)) with exact weights, but the per-32 promotion costs one FFMA per accumulator per
    K=32 (as much FP32 issue as the tensor-core time at m64n256k32) -- expected slower than "deepgemm"
    at 200 W; GPU benchmark pending. Only layers with K % 128 == 0: humming 0.1.17 leaves the unused
    packed e8m0 input-scale groups uninitialized when K/32 % 4 != 0 (fixed in 0.1.18 by humming #157),
    i.e. the shared expert down_proj (K = 576) stays stock on 0.1.17.

Runtime dispatch: ``ModelOptLinearMethod.apply`` of a captured layer runs the W8A8 (or W8A16) path when the input is
a 2+-D bf16 tensor with >= DS41H_PREFILL_LINEAR_MIN_M rows (default 256, above the default max cudagraph capture size
so FULL decode graphs never contain it) and no CUDA graph is being captured; everything else (decode, the ds41h
skinny M<=8 drop-in, bias, QuantizedActivation inputs) goes to the previous ``apply`` unchanged. The stock (Humming)
weights are kept for those paths, so this costs one extra e4m3 copy of the captured weights (~0.9 GB per GPU for
the default "full" layer set).

Switches (all need DS41H=1 and DS41H_PREFILL != 0):
  DS41H_PREFILL_LINEAR16=0        LOSSLESS W8A16 dense linears for M >= MIN_M (exact bf16 dequant + bf16 GEMM)
  DS41H_PREFILL_LINEAR16_GEMM     deepgemm (default) | cublas
                                  (default OFF: stock Humming W8A16 is the lossless dense path, equally fast at 250 W)
  DS41H_PREFILL_WOA16=1           LOSSLESS wo_a: in-place bf16 inverse RoPE + strided bf16 group GEMMs (M >= MIN_M)
  DS41H_PREFILL_WOA16_GEMM        cublas (default) | deepgemm
  DS41H_PREFILL_WQA_SHARD=1       LOSSLESS token-sharded replicated fused_wqa_wkv (stock kernel on M/tp rows per rank +
                                  TP all-gather), M >= DS41H_PREFILL_WQA_SHARD_MIN_M (1024)
  DS41H_PREFILL_LINEAR=0          LOSSY W8A8 dense linears for M >= MIN_M (fused_wqa_wkv, wq_b, wo_b,
                                  indexer.wq_b, shared expert gate_up/down, engram wkv); wins over LINEAR16
  DS41H_PREFILL_LINEAR_IMPL       deepgemm (default) | humming
  DS41H_PREFILL_LINEAR_MIN_M      rows threshold for the W8A8 / W8A16 paths (default 256)
  DS41H_PREFILL_LINEAR_LAYERS     layer-index ranges to capture, e.g. "0-20,24"; default "full": layers before
                                  the decoder-replay cut + the cut layer's fused_wqa_wkv; "all": every
                                  backbone layer (MTP/DSpark layers are never captured)
  DS41H_PREFILL_LINEAR_UE8M0=0    fp32 amax/448 activation scales instead of power-of-two ones (W8A8)
  DS41H_PREFILL_LINEAR_WARMUP=0   skip the load-time DeepGEMM JIT warmup of the prefill M sizes
  DS41H_PREFILL_WOA=0             LOSSY wo_a bmm as FP8 einsum (all M; replaces the BF16 emulation weight)
  DS41H_PREFILL_WOA_ROPEQ         with WOA: ds41h CUDA inverse RoPE + FP8 quant instead of vLLM's Triton kernel
  DS41H_PREFILL_MOE_FP8ACT=0      LOSSY PR #61875 port: Humming MoE with block-FP8 activations (stock MoE only)

Public API: ``RawWeight``, ``capture_raw``, ``dequant_bf16``, ``bf16_gemm``, ``gemm_w8a16``, ``gemm_w8a16_ref``,
``woa16``, ``inv_rope_bf16_``; lossy: ``fold_block_scales``, ``FoldedWeight``, ``fold_weight``, ``act_quant`` (CUDA) /
``act_quant_ref`` (torch), ``gemm_w8a8``, ``gemm_w8a8_ref``, ``dequant_folded``, ``convert_woa``;
``install_modelopt`` / ``install_o_proj16`` / ``install_moe_fp8_activations`` (called from ds41h/patch_linear.py), ``ext()``.
Imports of torch / vLLM / the extension are lazy (this module is imported from the import hook).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ds41h import flags

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("vllm.ds41h.prefill")

_ON = flags.ENABLED and flags.PREFILL
# Lossy switches: OFF by default (lossless policy), ON as a group with DS41H_LOSSY=1; an explicit env value always wins.
_LOSSY_DEFAULT = "1" if getattr(flags, "LOSSY", False) else "0"
LINEAR = _ON and flags._flag("DS41H_PREFILL_LINEAR", _LOSSY_DEFAULT)  # LOSSY W8A8 (e4m3 activations, 128x128 scale fold)
IMPL = os.environ.get("DS41H_PREFILL_LINEAR_IMPL", "deepgemm").strip().lower()
# >= 1: 0 would hang the load-time warmup (doubling loop) and send empty inputs to DeepGEMM.
MIN_M = max(1, int(os.environ.get("DS41H_PREFILL_LINEAR_MIN_M", "256")))
UE8M0 = flags._flag("DS41H_PREFILL_LINEAR_UE8M0", "1")
WARMUP = flags._flag("DS41H_PREFILL_LINEAR_WARMUP", "1")
LAYERS_SPEC = os.environ.get("DS41H_PREFILL_LINEAR_LAYERS", "").strip()
WOA = _ON and flags._flag("DS41H_PREFILL_WOA", _LOSSY_DEFAULT)  # LOSSY FP8 wo_a einsum (e4m3 activations)
MOE_FP8ACT = _ON and flags._flag("DS41H_PREFILL_MOE_FP8ACT", "0")
# With WOA: ds41h CUDA inverse RoPE + 1x128 FP8 quant of the attention output (fp8_linear.cu inv_rope_fp8_quant)
# instead of vLLM's Triton fused_inv_rope_fp8_quant for >= MIN_M tokens (same semantics; ~2x faster at 200 W).
ROPEQ = WOA and flags._flag("DS41H_PREFILL_WOA_ROPEQ", "1")
# LOSSLESS paths (default ON); the lossy switch of the same op wins when both are set.
# W8A16 via exact dequant + DeepGEMM/cuBLAS bf16: lossless but NOT faster than stock Humming W8A16 at the 250 W cap
# (layer 10, M = 16384: per-layer sum stock 10639 us, deepgemm 10795, cublas 10946; all three run at
# the power-capped bf16 tensor-core rate, 230-250 TFLOPS) and it costs a ~0.9 GB/GPU raw FP8 copy -> default OFF:
# stock Humming IS the lossless dense path.
LINEAR16 = _ON and not LINEAR and flags._flag("DS41H_PREFILL_LINEAR16", "0")
GEMM16 = os.environ.get("DS41H_PREFILL_LINEAR16_GEMM", "deepgemm").strip().lower()
WOA16 = _ON and not WOA and flags._flag("DS41H_PREFILL_WOA16", "1")
WOA16_GEMM = os.environ.get("DS41H_PREFILL_WOA16_GEMM", "cublas").strip().lower()

# LOSSLESS token-sharded fused_wqa_wkv: the q-lora / kv projection is a ReplicatedLinear (every TP rank computes the
# same [M, 1792] GEMM); with WQA_SHARD each rank runs the stock kernel on ceil(M / tp) of the rows and the outputs are
# all-gathered along the token dim (concat = the [M, 1792] layout, no permute copy). Same per-row math (stock Humming
# W8A16), 1/tp of the GEMM energy, + one all-gather of M x 1792 bf16. Rows >= WQA_SHARD_MIN_M only (prefill, eager).
# Default ON (end-to-end, 250 W: 16K-step kernel sum 673 -> 658 ms, per layer
# GEMM 1.16 -> ~0.30 ms + all-gather 0.45 ms on the critical rank; PP c1 16K 21,553 -> 21,627, c16 19,409 -> 20,051;
# KL top1 0.9904 PASS).
WQA_SHARD = _ON and flags._flag("DS41H_PREFILL_WQA_SHARD", "1")
WQA_SHARD_MIN_M = max(1, int(os.environ.get("DS41H_PREFILL_WQA_SHARD_MIN_M", "1024")))
WQA_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.attn\.fused_wqa_wkv$")

for _name, _val in (("DS41H_PREFILL_LINEAR16_GEMM", GEMM16), ("DS41H_PREFILL_WOA16_GEMM", WOA16_GEMM)):
    if _val not in ("deepgemm", "cublas"):
        raise ValueError(f"{_name} must be deepgemm or cublas, got {_val!r}")

if LINEAR and IMPL not in ("deepgemm", "humming"):  # only when the switch is on: never break the stock path
    raise ValueError(f"DS41H_PREFILL_LINEAR_IMPL must be deepgemm or humming, got {IMPL!r}")

BLOCK = 128  # DeepGEMM sm90 1D2D granularity (activation 1x128, weight 128x128)
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4  # reference act_quant floor (inference/kernel.py), same as common.cuh

# vLLM module prefixes of the DeepSeek-V4.1 dense FP8 linears (attention.py:364-403, :1313-1320;
# deepseek_v4/nvidia/model.py:160-176 shared expert; common/engram.py:1555-1563 engram wkv).
TARGET_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:attn\.(?:fused_wqa_wkv|wq_b|wo_b|indexer\.wq_b)"
    r"|ffn\.shared_experts\.(?:gate_up_proj|down_proj)|engram\.wkv)$"
)
WOA_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.attn\.wo_a$")


# ============================================================================================ weights
def _u8(t: "torch.Tensor") -> "torch.Tensor":
    import torch

    if t.element_size() != 1:
        raise TypeError(f"expected a 1-byte tensor, got {t.dtype}")
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def block_exponents(scale: "torch.Tensor", n: int, k: int) -> "torch.Tensor":
    """ue8m0 scales of an [n, k] FP8 weight -> int32 exponent bytes per 32x32 block [ceil(n/32), k/32].

    Accepts the checkpoint block form [ceil(n/32), k/32] and vLLM's row-expanded form [n, k/32]
    (KMxfp8Static.get_scale_weight_loader repeats each block row 32 times, modelopt.py:2278-2292).
    Raises if a row-expanded scale is not constant inside a 32-row block (not a 32x32 block scale)."""
    import torch

    s = _u8(scale.contiguous())
    kb = k // 32
    if s.dim() != 2 or s.shape[1] != kb:
        raise ValueError(f"scale shape {tuple(s.shape)} does not match K/32 = {kb}")
    nb = -(-n // 32)
    if s.shape[0] == nb:
        return s.to(torch.int32)
    if s.shape[0] == n and n % 32 == 0:
        t = s.view(nb, 32, kb)
        if not bool((t == t[:, :1]).all()):
            raise ValueError("weight scale varies inside a 32-row block; not a 32x32 block scale")
        return t[:, 0].to(torch.int32)
    raise ValueError(f"cannot interpret scale with {s.shape[0]} rows for N={n}")


def fold_block_scales(weight: "torch.Tensor", scale: "torch.Tensor", kp: int | None = None) -> tuple["torch.Tensor", "torch.Tensor", dict]:
    """32x32-block e4m3 weight -> 128x128-block e4m3 weight (DeepGEMM 1D2D recipe (1, 128, 128)).

    weight: e4m3 [n, k] (float8_e4m3fn or raw uint8), scale: ue8m0 (see ``block_exponents``).
    Returns (w e4m3 [n, kp] with zero columns in [k, kp), s fp32 [ceil(n/128), kp/128], stats) where
    w[i, j] * s[i/128, j/128] reproduces the checkpoint value except for the few sub-normal casualties
    counted in stats (``changed`` elements, ``max_abs_err`` in checkpoint units, ``rel_l2`` of the
    weight, ``spread`` = max exponent spread inside a 128x128 block). Runs on the weight's device."""
    import torch

    w8 = _u8(weight.contiguous())
    n, k = w8.shape
    if k % 32:
        raise ValueError(f"K = {k} must be a multiple of 32")
    kp = kp or _round_up(k, BLOCK)
    if kp % BLOCK or kp < k:
        raise ValueError(f"kp = {kp} must be a multiple of {BLOCK} and >= K = {k}")
    e = block_exponents(scale.to(w8.device), n, k)  # [nb32, kb32]
    nb32, kb32 = e.shape
    nbp, kbp = _round_up(nb32, 4), kp // 32
    ep = torch.full((nbp, kbp), -1, dtype=torch.int32, device=w8.device)  # -1: padding, never the max
    ep[:nb32, :kb32] = e
    eref = ep.view(nbp // 4, 4, kbp // 4, 4).amax(dim=(1, 3))  # [nb128, kb128]
    d = ep - eref.repeat_interleave(4, 0).repeat_interleave(4, 1)  # <= 0 (garbage only for padding)
    d = d[:nb32, :kb32].clamp(min=-126)
    spread = int(-d.min()) if d.numel() else 0
    wf = w8.view(torch.float8_e4m3fn).float()
    shift = torch.exp2(d.float()).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)  # [n, k]
    v = wf * shift
    vq = v.to(torch.float8_e4m3fn)  # rne; |v| <= 448 so no saturation
    diff = vq.float() - v
    changed = int((diff != 0).sum())
    max_abs_err = float((diff.abs() / shift).max()) if changed else 0.0  # in checkpoint e4m3 units
    # rel L2 of the dequantized weight (common per-block factor 2^(e-127) included)
    blockf = torch.exp2(e.float() - 127.0).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)
    num = (diff / shift * blockf).square().sum()
    den = (wf * blockf).square().sum().clamp_min(1e-30)
    rel_l2 = float((num / den).sqrt())
    w_out = torch.zeros((n, kp), dtype=torch.float8_e4m3fn, device=w8.device)
    w_out[:, :k] = vq
    s_out = torch.exp2(eref.float() - 127.0)[: -(-n // BLOCK)].contiguous()
    stats = {"changed": changed, "numel": n * k, "max_abs_err": max_abs_err, "rel_l2": rel_l2, "spread": spread}
    return w_out, s_out, stats


@dataclass(frozen=True)
class FoldedWeight:
    """A W8A8 weight for the DeepGEMM 1D2D path: w e4m3 [n, kp], ws fp32 scales in the layout
    DeepGEMM wants (vLLM ``deepgemm_post_process_fp8_weight_block``), s128 fp32 [ceil(n/128), kp/128]
    (plain block form, for references), n / k logical shape, kp = k rounded up to 128."""

    w: Any
    ws: Any
    s128: Any
    n: int
    k: int
    kp: int

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.ws.numel() * 4


def _deepgemm_weight_scale(w: "torch.Tensor", s128: "torch.Tensor") -> "torch.Tensor":
    """fp32 [n/128, kp/128] block scales -> the layout DeepGEMM's sm90 fp8_gemm_nt consumes (exactly what
    vLLM's DeepGemmFp8BlockScaledMMKernel does at load, kernels/linear/scaled_mm/deep_gemm.py:84-99)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block

    _, ws = deepgemm_post_process_fp8_weight_block(wq=w, ws=s128, quant_block_shape=(BLOCK, BLOCK), use_e8m0=False)
    return ws


def fold_weight(weight: "torch.Tensor", scale: "torch.Tensor", with_deepgemm_layout: bool = True) -> tuple[FoldedWeight, dict]:
    w, s128, stats = fold_block_scales(weight, scale)
    n, k = weight.shape
    ws = _deepgemm_weight_scale(w, s128) if with_deepgemm_layout else s128
    return FoldedWeight(w=w, ws=ws, s128=s128, n=int(n), k=int(k), kp=int(w.shape[1])), stats


def dequant_folded(fw: FoldedWeight) -> "torch.Tensor":
    """float32 [n, k] value of a folded weight (tests)."""
    s = fw.s128.repeat_interleave(BLOCK, 0)[: fw.n].repeat_interleave(BLOCK, 1)
    return (fw.w.float() * s)[:, : fw.k]


def dequant_checkpoint(weight: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """float32 [n, k] exact value of a 32x32-block e4m3 weight (tests)."""
    import torch

    w8 = _u8(weight)
    n, k = w8.shape
    e = block_exponents(scale.to(w8.device), n, k)
    sf = torch.exp2(e.float() - 127.0).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)
    return w8.view(torch.float8_e4m3fn).float() * sf


# ============================================================================================ activations
@functools.cache
def ext() -> Any:
    """JIT-built activation-quant extension (sm_90a)."""
    from ds41h import cuext

    return cuext.load("ds41h_fp8_linear", "fp8_linear.cu")


def act_quant(x2d: "torch.Tensor", kp: int, ue8m0: bool = UE8M0) -> tuple["torch.Tensor", "torch.Tensor"]:
    """bf16 [M, K] -> (e4m3 [M, kp], fp32 scales [M, kp/128] MN-major TMA-aligned view). CUDA."""
    return tuple(ext().act_quant_128(x2d, kp, ue8m0))  # type: ignore[return-value]


def act_quant_ref(x2d: "torch.Tensor", kp: int, ue8m0: bool = UE8M0) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch reference of ``act_quant`` (any device): same bits (float32 path; scales contiguous [M, kp/128])."""
    import torch

    M, K = x2d.shape
    xf = torch.zeros((M, kp), dtype=torch.float32, device=x2d.device)
    xf[:, :K] = x2d.float()
    xb = xf.view(M, kp // BLOCK, BLOCK)
    amax = xb.abs().amax(-1).clamp_min(AMAX_FLOOR)
    inv448 = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x2d.device)
    if ue8m0:
        bits = (amax * inv448).view(torch.int32)  # common.cuh quant_exp: ceil(log2(amax / 448))
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = torch.exp2(e.float())
        q = (xb * torch.exp2(-e.float()).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)  # satfinite
    else:
        scale = amax * inv448
        q = (xb * (FP8_MAX / amax).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(M, kp), scale


def gemm_w8a8(x: "torch.Tensor", fw: FoldedWeight, out: "torch.Tensor | None" = None) -> "torch.Tensor":
    """y [..., n] bf16 = x [..., k] bf16 @ W^T on the DeepGEMM sm90 1D2D FP8 path."""
    import torch
    from vllm.utils.deep_gemm import fp8_gemm_nt

    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.stride(-1) != 1 or x2d.stride(0) % 8 or x2d.data_ptr() % 16:
        x2d = x2d.contiguous()
    xq, xs = act_quant(x2d, fw.kp)
    y = torch.empty((x2d.shape[0], fw.n), dtype=torch.bfloat16, device=x.device) if out is None else out
    fp8_gemm_nt((xq, xs), (fw.w, fw.ws), y, is_deep_gemm_e8m0_used=False)
    return y.view(*lead, fw.n)


def gemm_w8a8_ref(x: "torch.Tensor", fw: FoldedWeight, ue8m0: bool = UE8M0) -> "torch.Tensor":
    """fp32 reference of ``gemm_w8a8``: dequantized quantized activations @ dequantized folded weight."""
    x2d = x.reshape(-1, x.shape[-1])
    q, s = act_quant_ref(x2d, fw.kp, ue8m0)
    xd = (q.float().view(q.shape[0], -1, BLOCK) * s.unsqueeze(-1)).view(q.shape)[:, : fw.k]
    return (xd @ dequant_folded(fw).t()).view(*x.shape[:-1], fw.n)


# ============================================================================================ lossless W8A16
@dataclass(frozen=True)
class RawWeight:
    """Checkpoint FP8 weight for the lossless W8A16 path: w uint8 (e4m3 bits) [n, k] contiguous, s uint8 ue8m0
    exponent per 32x32 block [ceil(n/32), k/32]."""

    w: Any
    s: Any
    n: int
    k: int

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel()


# dequant_fp8_bf16 multiplies in fp32: exact while e4m3 * 2^(S-127) stays a normal fp32 (S in [10, 246]).
_S_MIN, _S_MAX = 10, 246


def capture_raw(weight: "torch.Tensor", scale: "torch.Tensor") -> RawWeight:
    """Copy of a checkpoint e4m3 [n, k] weight + its 32x32 ue8m0 scales (block or vLLM row-expanded form)."""
    import torch

    w8 = _u8(weight.contiguous())
    n, k = w8.shape
    if k % 32:
        raise ValueError(f"K = {k} must be a multiple of 32")
    e = block_exponents(scale.to(w8.device), n, k)
    lo, hi = int(e.min()), int(e.max())
    if lo < _S_MIN or hi > _S_MAX:
        raise ValueError(f"ue8m0 scale bytes [{lo}, {hi}] outside the exact-dequant range [{_S_MIN}, {_S_MAX}]")
    return RawWeight(w=w8.clone(), s=e.to(torch.uint8).contiguous(), n=int(n), k=int(k))


def dequant_bf16(rw: RawWeight, out: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Exact bf16 [n, k] value of a raw weight (CUDA)."""
    return ext().dequant_fp8_bf16(rw.w, rw.s, out)


def dequant_bf16_ref(rw: RawWeight) -> "torch.Tensor":
    """Torch reference of ``dequant_bf16`` (any device): fp32 exact value, cast to bf16 (exact)."""
    import torch

    sf = torch.exp2(rw.s.float() - 127.0).repeat_interleave(32, 0)[: rw.n].repeat_interleave(32, 1)
    return (rw.w.view(torch.float8_e4m3fn).float() * sf).to(torch.bfloat16)


class _NoReducedPrecision:
    """cuBLAS bf16 GEMMs with fp32 reductions only (torch allows reduced-precision split-K reductions by default)."""

    def __enter__(self):
        import torch

        m = torch.backends.cuda.matmul
        self.saved = m.allow_bf16_reduced_precision_reduction
        m.allow_bf16_reduced_precision_reduction = False

    def __exit__(self, *exc):
        import torch

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.saved


def bf16_gemm(a: "torch.Tensor", b: "torch.Tensor", out: "torch.Tensor | None" = None, impl: str | None = None) -> "torch.Tensor":
    """d [M, N] bf16 = a [M, K] bf16 @ b [N, K]^T with fp32 accumulation (strided rows allowed, unit inner stride)."""
    import torch

    impl = impl or GEMM16
    M, N = a.shape[0], b.shape[0]
    d = torch.empty((M, N), dtype=torch.bfloat16, device=a.device) if out is None else out
    if impl == "deepgemm" and not _DG_BROKEN:
        try:
            from vllm.third_party.deep_gemm import bf16_gemm_nt

            bf16_gemm_nt(a, b, d)
            return d
        except (RuntimeError, AssertionError, ValueError, TypeError, ImportError) as e:  # shape / stride rejected
            _dg_failed(e, a, b, d)
    with _NoReducedPrecision():
        torch.mm(a, b.t(), out=d)
    return d


_DG_BROKEN = False


def _dg_failed(e: Exception, a, b, d) -> None:
    """DeepGEMM bf16 refused a call (raised before launching): use cuBLAS from now on (same lossless math)."""
    global _DG_BROKEN
    _DG_BROKEN = True
    logger.warning("ds41h prefill: DeepGEMM bf16_gemm_nt failed (%s: %s) for a %s/%s b %s d %s/%s; cuBLAS bf16 from now on",
                   type(e).__name__, str(e)[:200], tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(d.shape), tuple(d.stride()))


def gemm_w8a16(x: "torch.Tensor", rw: RawWeight, out: "torch.Tensor | None" = None, impl: str | None = None) -> "torch.Tensor":
    """y [..., n] bf16 = x [..., k] bf16 @ W^T: exact bf16 dequant of the FP8 weight into a scratch buffer + bf16 GEMM."""
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    if not x2d.is_contiguous() or x2d.data_ptr() % 16:  # DeepGEMM / TMA: dense rows (strided A not validated)
        x2d = x2d.contiguous()
    wb = dequant_bf16(rw)
    y = bf16_gemm(x2d, wb, out, impl)
    del wb  # back to the caching allocator; the next linear reuses the block
    return y.view(*lead, rw.n)


def gemm_w8a16_ref(x: "torch.Tensor", rw: RawWeight) -> "torch.Tensor":
    """fp32 reference: bf16 activations @ exact weight (fp32 math)."""
    x2d = x.reshape(-1, x.shape[-1]).float()
    return (x2d @ dequant_bf16_ref(rw).float().t()).view(*x.shape[:-1], rw.n)


def inv_rope_bf16_(o: "torch.Tensor", positions: "torch.Tensor", cos_sin_cache: "torch.Tensor") -> "torch.Tensor":
    """In-place stock bf16 inverse RoPE of the 64 RoPE dims of each 512-dim head of o [T, H, 512] (CUDA)."""
    ext().inv_rope_bf16_(o, positions, cos_sin_cache)
    return o


def inv_rope_bf16_ref(o: "torch.Tensor", positions: "torch.Tensor", cos_sin_cache: "torch.Tensor") -> "torch.Tensor":
    """Torch reference of the stock bf16 inverse RoPE (new tensor; fp32 math, no FMA contraction)."""
    x = o.float()
    cs = cos_sin_cache[positions[: o.shape[0]]].float()  # [T, 64]
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    ev, od = x[..., 448::2], x[..., 449::2]
    out = x.clone()
    out[..., 448::2] = ev * cos + od * sin
    out[..., 449::2] = od * cos - ev * sin
    return out.to(o.dtype)


def woa16(o: "torch.Tensor", positions: "torch.Tensor", cos_sin_cache: "torch.Tensor", weight: "torch.Tensor", n_groups: int,
          o_lora_rank: int, impl: str | None = None) -> "torch.Tensor":
    """Lossless wo_a: z [T, G, R] bf16 from the attention output o [T, G * 8, 512] (RoPE dims rotated back IN PLACE) and
    stock's bf16 weight [G * R, 8 * 512]: z[:, g] = o[:, 8g:8g+8].flatten(1) @ weight[gR:(g+1)R]^T."""
    import torch

    impl = impl or WOA16_GEMM
    T, H, D = o.shape
    inv_rope_bf16_(o, positions, cos_sin_cache)
    a = o.view(T, n_groups, (H // n_groups) * D)  # strided rows (stride(0) = row stride of the [T, H(p), 512] buffer)
    w = weight.view(n_groups, o_lora_rank, -1)
    z = torch.empty((T, n_groups, o_lora_rank), dtype=torch.bfloat16, device=o.device)
    for g in range(n_groups):
        bf16_gemm(a[:, g], w[g], z[:, g], impl)
    return z


def woa16_ref(o: "torch.Tensor", positions: "torch.Tensor", cos_sin_cache: "torch.Tensor", weight: "torch.Tensor",
              n_groups: int, o_lora_rank: int) -> "torch.Tensor":
    """fp32 reference of ``woa16`` (o untouched)."""
    T, H, D = o.shape
    a = inv_rope_bf16_ref(o, positions, cos_sin_cache).float().reshape(T, n_groups, (H // n_groups) * D)
    w = weight.float().view(n_groups, o_lora_rank, -1)
    return (a.transpose(0, 1) @ w.transpose(1, 2)).transpose(0, 1)


# ============================================================================================ vLLM glue
def _parse_layers(spec: str) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


_LAYERS = _parse_layers(LAYERS_SPEC) if LAYERS_SPEC.lower() not in ("", "full", "all") else None
# Default (DS41H_PREFILL_LINEAR_LAYERS unset or "full"): only the layers that see full prefill rows -- 0 .. cut - 1 and
# the KV-only pass of layer `cut` = max(kv_source_layer_ids) (its fused_wqa_wkv); layers past the cut run on each
# request's 128-token replay window (decoder SWA bounded replay), where W8A16 is as fast and the extra FP8 copy
# (~40 MB per layer per GPU) would only cost KV-cache memory. "all" captures every backbone layer.
_LAYERS_ALL = LAYERS_SPEC.lower() == "all"


_CFG_CACHE: dict[str, int] = {}


def _vllm_cfg_int(key: str, getter, env: str, default: int) -> int:
    """A vLLM config value, cached once found. process_weights_after_loading may run outside the
    set_current_vllm_config context, so fall back to an env override / DeepSeek-V4.1-Flash default."""
    if key in _CFG_CACHE:
        return _CFG_CACHE[key]
    if os.environ.get(env):
        return _CFG_CACHE.setdefault(key, int(os.environ[env]))
    try:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            return _CFG_CACHE.setdefault(key, int(getter(cfg)))
    except Exception:  # noqa: BLE001
        pass
    return default


def _num_hidden_layers() -> int:
    return _vllm_cfg_int(
        "layers", lambda c: c.model_config.hf_text_config.num_hidden_layers, "DS41H_NUM_HIDDEN_LAYERS", 40
    )


def _max_batched_tokens() -> int:
    return _vllm_cfg_int(
        "tokens", lambda c: c.scheduler_config.max_num_batched_tokens, "DS41H_PREFILL_LINEAR_MAX_TOKENS", 8192
    )


def _replay_cut() -> int:
    """max(kv_source_layer_ids): the last layer that writes KV for every prefill row (vLLM's decoder replay cut)."""
    def get(c):
        ids = getattr(c.model_config.hf_text_config, "kv_source_layer_ids", None)
        return max(ids) if ids else 10**6
    return _vllm_cfg_int("replay_cut", get, "DS41H_PREFILL_LINEAR_CUT", 20)


def target_layer(prefix: str, pattern: re.Pattern = TARGET_RE) -> int | None:
    """Layer index if `prefix` is a captured dense linear of a backbone layer, else None."""
    m = pattern.search(prefix or "")
    if m is None:
        return None
    idx = int(m.group(1))
    if idx >= _num_hidden_layers():  # MTP / DSpark layers are built as layers.{num_hidden_layers + k}
        return None
    if _LAYERS is not None:
        return idx if idx in _LAYERS else None
    if not _LAYERS_ALL and pattern is TARGET_RE:
        cut = _replay_cut()
        if idx > cut or (idx == cut and not prefix.endswith("fused_wqa_wkv")):
            return None
    return idx


def _is_sm90() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@functools.cache
def _deepgemm_ok() -> bool:
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        return bool(is_deep_gemm_supported())
    except Exception:  # noqa: BLE001
        return False


_WARMED: set[tuple[int, int]] = set()


def _warmup_m_values(n: int, device: Any) -> list[int]:
    max_tokens = _max_batched_tokens()
    try:  # the M values at which DeepGEMM's sm90 heuristic changes config (vLLM "relax" warmup)
        from vllm.model_executor.warmup.deep_gemm_warmup import _generate_optimal_warmup_m_values

        ms = [m for m in _generate_optimal_warmup_m_values(max_tokens, n, device) if m >= MIN_M]
    except Exception:  # noqa: BLE001
        ms = []
    m, pow2 = MIN_M, []
    while m < max_tokens:
        pow2.append(m)
        m *= 2
    return sorted(set(ms) | set(pow2) | {max_tokens})


def _warmup(fw: FoldedWeight) -> None:
    """JIT-compile DeepGEMM for this (n, kp) at the prefill M sizes now (load time, eager), not in the
    first prefill. No-op for shapes already warmed."""
    import torch

    key = (fw.n, fw.kp)
    if not WARMUP or key in _WARMED:
        return
    ms = _warmup_m_values(fw.n, fw.w.device)
    x = torch.zeros((max(ms), fw.k), dtype=torch.bfloat16, device=fw.w.device)
    for m in ms:
        gemm_w8a8(x[:m], fw)
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()  # the warm-up buffers must not stay reserved when vLLM sizes the KV cache
    _WARMED.add(key)
    logger.info("ds41h prefill linear: DeepGEMM warmed up for n=%d k=%d at M=%s", fw.n, fw.kp, ms)


def capture(layer: Any) -> FoldedWeight | None:
    """Fold and keep a W8A8 copy of `layer`'s checkpoint FP8 weight (called before the stock repack)."""
    import torch

    existing = layer.__dict__.get("_ds41h_w8a8")
    if existing is not None:
        return existing
    w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dim() != 2 or w.dtype != torch.float8_e4m3fn or s.element_size() != 1 or not w.is_cuda:
        return None
    n, k = w.shape
    if n % 64 or k % 32:  # DeepGEMM N % 64 (deep_gemm.py should_use_deepgemm_for_fp8_linear)
        return None
    fw, stats = fold_weight(w.data, s.data)
    layer._ds41h_w8a8 = fw
    logger.info(
        "ds41h prefill linear: %s [%d x %d -> %d] folded 32x32->128x128: changed %d/%d (max %.3g ckpt units, rel L2 %.2e)",
        getattr(layer, "prefix", "?"), n, k, fw.kp, stats["changed"], stats["numel"], stats["max_abs_err"], stats["rel_l2"],
    )
    _warmup(fw)
    return fw


_WARMED16: set[tuple[int, int, str]] = set()


def _warmup16(n: int, k: int, device: Any) -> None:
    """JIT-compile DeepGEMM's bf16 GEMM for this (n, k) at the prefill M sizes now (load time)."""
    import torch

    key = (n, k, GEMM16)
    if not WARMUP or GEMM16 != "deepgemm" or key in _WARMED16:
        return
    ms = _warmup_m_values(n, device)
    x = torch.zeros((max(ms), k), dtype=torch.bfloat16, device=device)
    wb = torch.zeros((n, k), dtype=torch.bfloat16, device=device)
    for m in ms:
        bf16_gemm(x[:m], wb)
    torch.cuda.synchronize()
    del x, wb
    torch.cuda.empty_cache()
    _WARMED16.add(key)
    logger.info("ds41h prefill linear16: DeepGEMM bf16 warmed up for n=%d k=%d at M=%s", n, k, ms)


def capture16(layer: Any) -> RawWeight | None:
    """Keep a raw copy of `layer`'s checkpoint FP8 weight for the lossless W8A16 path (before the stock repack)."""
    import torch

    existing = layer.__dict__.get("_ds41h_w8a16")
    if existing is not None:
        return existing
    w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dim() != 2 or w.dtype != torch.float8_e4m3fn or s.element_size() != 1 or not w.is_cuda:
        return None
    n, k = w.shape
    if n % 64 or k % 64:
        return None
    rw = capture_raw(w.data, s.data)
    layer._ds41h_w8a16 = rw
    logger.info("ds41h prefill linear16: %s [%d x %d] raw FP8 copy (%.1f MB) for the lossless W8A16 path",
                getattr(layer, "prefix", "?"), n, k, rw.nbytes / 2**20)
    _warmup16(n, k, w.device)
    return rw


def _tp_info() -> tuple[int, int, Any] | None:
    """(rank, world size, TP GroupCoordinator) or None outside an initialized TP group of size > 1."""
    try:
        from vllm.distributed import parallel_state as ps

        if not ps.model_parallel_is_initialized():
            return None
        g = ps.get_tp_group()
        return (g.rank_in_group, g.world_size, g) if g.world_size > 1 else None
    except Exception:  # noqa: BLE001
        return None


def sharded_rows_apply(apply_fn: Any, x: "torch.Tensor", n: int, tp: tuple[int, int, Any]) -> "torch.Tensor":
    """y [..., n] = apply_fn(x) computed as row shards: rank r runs apply_fn on rows [r c, (r + 1) c), c = ceil(M / tp),
    and the [c, n] results are all-gathered along dim 0 (concat order = row order). Lossless (same per-row kernel)."""
    import torch

    rank, world, group = tp
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    M = x2d.shape[0]
    c = -(-M // world)
    lo, hi = min(rank * c, M), min((rank + 1) * c, M)
    y_loc = apply_fn(x2d[lo:hi]) if hi > lo else None
    if y_loc is None or y_loc.shape[0] != c or not y_loc.is_contiguous():
        pad = torch.empty((c, n), dtype=torch.bfloat16, device=x.device)
        if y_loc is not None:
            pad[: hi - lo].copy_(y_loc.reshape(hi - lo, n))
        y_loc = pad
    y = group.all_gather(y_loc.reshape(c, n), dim=0)  # [world * c, n]
    return y[:M].view(*lead, n)


def eligible_shard(x: Any) -> bool:
    import torch

    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() >= 2
        and x.is_cuda
        and x.numel() // x.shape[-1] >= WQA_SHARD_MIN_M
        and not torch.cuda.is_current_stream_capturing()
    )


def eligible16(x: Any, rw: RawWeight) -> bool:
    import torch

    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() >= 2
        and x.is_cuda
        and x.shape[-1] == rw.k
        and x.numel() // rw.k >= MIN_M
        and not torch.cuda.is_current_stream_capturing()
    )


def eligible(x: Any, fw: FoldedWeight) -> bool:
    import torch

    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() >= 2
        and x.is_cuda
        and x.shape[-1] == fw.k
        and x.numel() // fw.k >= MIN_M
        and not torch.cuda.is_current_stream_capturing()
    )


def convert_woa(layer: Any) -> bool:
    """Replace the bmm ``wo_a``'s checkpoint MXFP8 weight by a 128x128-folded e4m3 weight + fp32 scales
    in DeepGEMM's grouped (einsum) layout, so the stock o_proj takes its FP8 branch
    (deepseek_v4/nvidia/ops/o_proj.py:63-104). Returns False (layer untouched) if not applicable."""
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
    from vllm.model_executor.utils import replace_parameter

    w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
    g = int(getattr(layer, "bmm_batch_size", 0) or 0)
    if w is None or s is None or w.dim() != 2 or w.dtype != torch.float8_e4m3fn or s.element_size() != 1 or g <= 0:
        return False
    n, k = w.shape
    if n % g or (n // g) % BLOCK or k % BLOCK:
        return False
    wq, s128, stats = fold_block_scales(w.data, s.data)
    wq3, ws = deepgemm_post_process_fp8_weight_block(
        wq=wq, ws=s128, quant_block_shape=(BLOCK, BLOCK), use_e8m0=False, is_bmm=True, bmm_batch_size=g
    )
    # Warm up (and thereby validate) the FP8 o-proj path on a stand-in before touching the layer, so a
    # failure leaves the checkpoint weights for the stock BF16 emulation kernel.
    _warmup_woa(types.SimpleNamespace(weight=wq3, weight_scale=ws), g, n // g, k)
    replace_parameter(layer, "weight", wq3)
    replace_parameter(layer, "weight_scale", ws)
    layer._ds41h_woa_fp8 = True
    logger.info(
        "ds41h prefill wo_a: %s -> FP8 einsum [%d x %d x %d] (changed %d/%d, rel L2 %.2e)",
        getattr(layer, "prefix", "?"), g, n // g, k, stats["changed"], stats["numel"], stats["rel_l2"],
    )
    return True


_WARMED_WOA: set[tuple[int, int, int]] = set()


def _warmup_woa(layer: Any, g: int, r: int, k: int) -> None:
    """JIT-compile the stock FP8 o-proj path (fused_inv_rope_fp8_quant + DeepGEMM fp8_einsum) for every
    M config now, at load time: the converted wo_a also runs in decode, i.e. inside FULL CUDA graphs, and
    a DeepGEMM JIT / module load must not happen during capture."""
    import torch
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    key = (g, r, k)
    heads_per_group, head_dim = 8, 512
    if not WARMUP or key in _WARMED_WOA or k != heads_per_group * head_dim:
        return
    max_tokens = _max_batched_tokens()
    try:
        from vllm.model_executor.warmup.deep_gemm_warmup import _generate_optimal_warmup_m_values

        ms = sorted(set(_generate_optimal_warmup_m_values(max_tokens, r, layer.weight.device)) | {max_tokens})
    except Exception:  # noqa: BLE001
        ms = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, max_tokens})
    dev = layer.weight.device
    o = torch.zeros((max(ms), g * heads_per_group, head_dim), dtype=torch.bfloat16, device=dev)
    pos = torch.zeros(max(ms), dtype=torch.int64, device=dev)
    cs = torch.zeros((1, 64), dtype=torch.float32, device=dev)
    for m in ms:
        deep_gemm_fp8_o_proj(o[:m], pos[:m], cs, layer, lambda z: z, n_groups=g, heads_per_group=heads_per_group,
                             nope_dim=448, rope_dim=64, o_lora_rank=r, einsum_recipe=(1, BLOCK, BLOCK),
                             tma_aligned_scales=False)
    torch.cuda.synchronize()
    del o, pos
    torch.cuda.empty_cache()
    _WARMED_WOA.add(key)  # only after success: a failing config is retried (and fails) for every layer
    logger.info("ds41h prefill wo_a: FP8 einsum warmed up for %d M values (max %d)", len(ms), max(ms))


def _humming_mxfp8_process(kernel: Any, layer: Any) -> None:
    """HummingMxfp8LinearKernel.process_weights_after_loading (kernels/linear/mxfp8/humming.py:43-60) with
    the MXFP8 input schema kept (allow_input_schema_fallback=False): W8A8 MXFP8 x MXFP8 on sm90."""
    import torch
    from vllm.model_executor.layers.quantization.utils.humming import (
        convert_linear_layer_to_humming_standard,
        get_humming_linear_compute_config,
        prepare_humming_linear_layer_config,
        quant_key_to_input_schema,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    from vllm.utils.humming import create_humming_locks

    layer.weight_scale.data = layer.weight_scale.data.view(torch.float8_e8m0fnu)
    quant_config = {
        "quant_method": "humming",
        "dtype": "float8e4m3",
        "scale_dtype": "float8e8m0",
        "group_size": 32,
        "weight_scale_type": "group",
    }
    convert_linear_layer_to_humming_standard(layer=layer, name_map={"weight": "weight", "weight_scale": "weight_scale"})
    kernel.layer_config = prepare_humming_linear_layer_config(
        layer, quant_config, input_schema=quant_key_to_input_schema(kMxfp8Dynamic), allow_input_schema_fallback=False
    )
    kernel.compute_config = get_humming_linear_compute_config()
    kernel.locks = create_humming_locks(layer.weight.device)
    layer._ds41h_humming_w8a8 = True


def install_modelopt(module: Any) -> None:
    """Wrap ``module.ModelOptLinearMethod`` (vllm.model_executor.layers.quantization.modelopt) for the
    LINEAR / WOA switches. Composes with the prologue component's skinny wrapper (ours is outermost,
    so it sees the checkpoint weight before anything else). Idempotent."""
    cls = module.ModelOptLinearMethod
    if getattr(cls, "_ds41h_linear_installed", False):
        return
    base_pwal = cls.process_weights_after_loading
    base_apply = cls.apply
    pre_processed = getattr(module, "is_weights_pre_processed", None)
    kMxfp8Static = module.kMxfp8Static

    def process_weights_after_loading(self, layer):
        import torch

        if layer.__dict__.get("_ds41h_woa_fp8", False):
            # Already converted (a repeated post-load, e.g. a weight reload): the stock bmm post-load would
            # run the BF16 emulation dequant on the 3-D e4m3 weight with fp32 128x128 scales.
            return None
        spec_w = getattr(getattr(self, "spec", None), "weight", None)
        prefix = getattr(layer, "prefix", "") or ""
        if spec_w == kMxfp8Static and not (pre_processed is not None and pre_processed()) and _is_sm90():
            try:
                if WOA and getattr(layer, "is_bmm", False) and target_layer(prefix, WOA_RE) is not None and _deepgemm_ok():
                    self.wkey.process(layer, module.WEIGHT)  # stock validation (2-D e4m3 + ue8m0)
                    if convert_woa(layer):
                        return None  # replaces the BF16 emulation kernel's dequant; apply is never used
                elif LINEAR16 and not getattr(layer, "is_bmm", False) and target_layer(prefix) is not None:
                    if GEMM16 == "cublas" or _deepgemm_ok():
                        capture16(layer)
                elif LINEAR and not getattr(layer, "is_bmm", False) and target_layer(prefix) is not None:
                    if IMPL == "deepgemm" and _deepgemm_ok():
                        capture(layer)
                    elif IMPL == "humming":
                        from vllm.model_executor.kernels.linear.mxfp8.humming import HummingMxfp8LinearKernel

                        kernel = self.kernel
                        if isinstance(kernel, HummingMxfp8LinearKernel) and layer.weight.shape[-1] % BLOCK == 0:
                            # The stock lifecycle (and the prologue's skinny capture) runs unchanged; only
                            # the kernel's own post-load step is swapped, via an instance attribute.
                            kernel.process_weights_after_loading = types.MethodType(_humming_mxfp8_process, kernel)
            except Exception:  # noqa: BLE001 - a failed capture must leave the stock path intact
                logger.exception("ds41h prefill linear: setup failed for %s; stock path kept", prefix)
                layer.__dict__.pop("_ds41h_w8a8", None)
                layer.__dict__.pop("_ds41h_w8a16", None)
        if WQA_SHARD and WQA_RE.search(prefix) and target_layer(prefix, WQA_RE) is not None:
            layer._ds41h_wqa_shard = True
        out = base_pwal(self, layer)
        if WOA16 and getattr(layer, "is_bmm", False) and _is_sm90() and target_layer(prefix, WOA_RE) is not None:
            try:  # stock's emulation kernel has dequantized wo_a to bf16 [G * R, K]: warm the strided group GEMMs once
                w, g = getattr(layer, "weight", None), int(getattr(layer, "bmm_batch_size", 0) or 0)
                if w is not None and w.dtype == torch.bfloat16 and w.dim() == 2 and g > 0 and (WOA16_GEMM == "cublas" or _deepgemm_ok()):
                    warmup_woa16(g, w.shape[0] // g, w.shape[1], w.device)
            except Exception:  # noqa: BLE001 - a failed warm-up only means a JIT at the first prefill
                logger.exception("ds41h prefill woa16: warm-up failed for %s", prefix)
        return out

    def apply_local(self, layer, x, bias=None):
        rw = layer.__dict__.get("_ds41h_w8a16")
        if rw is not None and bias is None and eligible16(x, rw):
            return gemm_w8a16(x, rw)
        fw = layer.__dict__.get("_ds41h_w8a8")
        if fw is not None and bias is None and eligible(x, fw):
            return gemm_w8a8(x, fw)
        return base_apply(self, layer, x, bias)

    def apply(self, layer, x, bias=None):
        if bias is None and layer.__dict__.get("_ds41h_wqa_shard", False) and eligible_shard(x):
            tp = _tp_info()
            n = int(sum(getattr(layer, "output_partition_sizes", None) or [0]))
            if tp is not None and n > 0:
                return sharded_rows_apply(lambda xs: apply_local(self, layer, xs, None), x, n, tp)
        return apply_local(self, layer, x, bias)

    process_weights_after_loading.__wrapped__ = base_pwal  # type: ignore[attr-defined]
    apply.__wrapped__ = base_apply  # type: ignore[attr-defined]
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    cls._ds41h_linear_installed = True


def inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups: int):
    """CUDA drop-in for vLLM's fused_inv_rope_fp8_quant (SM90: fp32 scales): (out [T, G, 4096] e4m3, scales [T, G, 32])."""
    return tuple(ext().inv_rope_fp8_quant(o, positions, cos_sin_cache, int(n_groups)))


def make_inv_rope_fp8_quant(stock_fn: Any) -> Any:
    def fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim=448, rope_dim=64,
                                 quant_group_size=128, tma_aligned_scales=False, quantize=True):
        import torch

        if (quantize and not tma_aligned_scales and n_groups == 2 and heads_per_group == 8 and nope_dim == 448 and rope_dim == 64
                and quant_group_size == 128 and isinstance(o, torch.Tensor) and o.is_cuda and o.dim() == 3 and o.shape[0] >= MIN_M
                and o.dtype == torch.bfloat16 and o.stride(-1) == 1 and positions.dtype == torch.int64
                and cos_sin_cache.dtype == torch.float32 and cos_sin_cache.dim() == 2 and cos_sin_cache.stride(-1) == 1
                and not torch.cuda.is_current_stream_capturing()):
            return inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups)
        return stock_fn(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim,
                        quant_group_size=quant_group_size, tma_aligned_scales=tma_aligned_scales, quantize=quantize)

    fused_inv_rope_fp8_quant.__wrapped__ = stock_fn  # type: ignore[attr-defined]
    return fused_inv_rope_fp8_quant


def install_o_proj(module: Any) -> None:
    """Patch ``module`` = vllm.models.deepseek_v4.nvidia.ops.o_proj (module-global fused_inv_rope_fp8_quant). Idempotent."""
    if getattr(module, "_ds41h_ropeq_installed", False):
        return
    module.fused_inv_rope_fp8_quant = make_inv_rope_fp8_quant(module.fused_inv_rope_fp8_quant)
    module._ds41h_ropeq_installed = True


def make_o_proj16(stock_fn: Any) -> Any:
    """Wrap ``deep_gemm_fp8_o_proj`` (module global of vllm.models.deepseek_v41.nvidia.ops.o_proj): eligible prefill
    calls with stock's bf16 wo_a weight run ``woa16`` (lossless: same inverse RoPE + bf16 GEMM math, no copies)."""

    def deep_gemm_fp8_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group, nope_dim, rope_dim,
                             o_lora_rank, einsum_recipe, tma_aligned_scales):
        import torch

        w = getattr(wo_a, "weight", None)
        if (isinstance(o, torch.Tensor) and o.is_cuda and o.dim() == 3 and o.dtype == torch.bfloat16 and o.shape[0] >= MIN_M
                and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
                and tuple(w.shape) == (n_groups * o_lora_rank, heads_per_group * 512) and w.is_contiguous()
                and nope_dim == 448 and rope_dim == 64 and o.shape[1] == n_groups * heads_per_group and o.shape[2] == 512
                and o.stride(2) == 1 and o.stride(1) == 512 and o.stride(0) % 8 == 0 and o.data_ptr() % 16 == 0
                and positions.dtype == torch.int64 and positions.is_contiguous()
                and cos_sin_cache.dtype == torch.float32 and cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
                and cos_sin_cache.stride(1) == 1 and not torch.cuda.is_current_stream_capturing()):
            z = woa16(o, positions, cos_sin_cache, w, n_groups, o_lora_rank)
            return wo_b(z.flatten(1))
        return stock_fn(o, positions, cos_sin_cache, wo_a, wo_b, n_groups=n_groups, heads_per_group=heads_per_group,
                        nope_dim=nope_dim, rope_dim=rope_dim, o_lora_rank=o_lora_rank, einsum_recipe=einsum_recipe,
                        tma_aligned_scales=tma_aligned_scales)

    deep_gemm_fp8_o_proj.__wrapped__ = stock_fn  # type: ignore[attr-defined]
    return deep_gemm_fp8_o_proj


_WARMED_WOA16: set[tuple[int, int, int]] = set()


def warmup_woa16(g: int = 2, r: int = 1024, k: int = 4096, device: Any = None) -> None:
    """JIT-compile DeepGEMM's bf16 GEMM for the wo_a group shape (strided A / D) at the prefill M sizes."""
    import torch

    key = (g, r, k)
    if not WARMUP or WOA16_GEMM != "deepgemm" or key in _WARMED_WOA16:
        return
    device = device or torch.device("cuda", torch.cuda.current_device())
    ms = _warmup_m_values(r, device)
    o = torch.zeros((max(ms), g * 8, 512), dtype=torch.bfloat16, device=device)
    pos = torch.zeros(max(ms), dtype=torch.int64, device=device)
    cs = torch.zeros((1, 64), dtype=torch.float32, device=device)
    w = torch.zeros((g * r, k), dtype=torch.bfloat16, device=device)
    for m in ms:
        woa16(o[:m], pos[:m], cs, w, g, r)
    torch.cuda.synchronize()
    del o, pos, w
    torch.cuda.empty_cache()
    _WARMED_WOA16.add(key)
    logger.info("ds41h prefill woa16: DeepGEMM bf16 warmed up for the wo_a groups at M=%s", ms)


def install_o_proj16(module: Any) -> None:
    """Patch ``module`` = vllm.models.deepseek_v41.nvidia.ops.o_proj (module-global deep_gemm_fp8_o_proj). Idempotent."""
    if getattr(module, "_ds41h_woa16_installed", False):
        return
    module.deep_gemm_fp8_o_proj = make_o_proj16(module.deep_gemm_fp8_o_proj)
    module._ds41h_woa16_installed = True


def install_moe_fp8_activations(module: Any) -> None:
    """Port of vLLM PR #61875 for 0.32.0 (module = vllm.models.deepseek_v41.quant_config): the stock
    ``Mxfp4MoEMethod`` of DeepSeek-V4.1's routed experts gets the checkpoint's MXFP8 activation key,
    run as block FP8 (group 128, fp32 scales) on SM90 -- exactly what #61875's ``_humming_input_schema``
    resolves to on Hopper (convert_weight_to_mxfp4_moe_kernel_format then builds Humming's input schema
    from it, fused_moe/oracle/mxfp4.py:1544-1561). Skipped when VLLM_HUMMING_INPUT_QUANT_CONFIG is set
    (the env still wins, as in the PR), when the backend is not Humming, and for the ds41h raw MoE
    method (not an Mxfp4MoEMethod). Idempotent."""
    cls = module.DeepseekV4FP8Config
    orig = cls.get_quant_method
    if getattr(orig, "_ds41h_moe_fp8act", False):
        return

    def get_quant_method(self, layer, prefix):
        method = orig(self, layer, prefix)
        try:
            if type(method) is module.Mxfp4MoEMethod and getattr(method, "activation_key", None) is None:
                import vllm.envs as envs
                from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
                from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
                from vllm.platforms import current_platform

                if (
                    current_platform.is_device_capability_family(90)
                    and not envs.VLLM_HUMMING_INPUT_QUANT_CONFIG
                    and method.mxfp4_backend == Mxfp4MoeBackend.HUMMING
                ):
                    method.activation_key = kFp8Dynamic128Sym
                    logger.info("ds41h prefill: %s MoE activations run as block FP8 (group 128) on Hopper (PR #61875)", prefix)
        except Exception:  # noqa: BLE001 - keep the stock method on any surprise
            logger.exception("ds41h prefill: MoE FP8-activation patch failed for %s", prefix)
        return method

    get_quant_method._ds41h_moe_fp8act = True  # type: ignore[attr-defined]
    get_quant_method.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.get_quant_method = get_quant_method
