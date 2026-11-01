"""ds41h MoE: DeepSeek-V4.1 routed experts kept in the raw checkpoint layout (TP-4 slice).

Stock vLLM 0.32.0 on sm90 repacks the MXFP4 experts for Humming (or Marlin) in
``Mxfp4MoEMethod.process_weights_after_loading`` (quantization/mxfp4.py:1012-1036 ->
fused_moe/oracle/mxfp4.py convert_weight_to_mxfp4_moe_kernel_format). The routed experts are
~72 GB per GPU, so a second copy is impossible; ds41h instead keeps exactly what the stock weight
loader writes into the ``Mxfp4MoEMethod`` parameters and runs its own kernels on it:

  w13_weight        uint8 [E, 2*I_r, H/2]   e2m1 pairs, low nibble = even k (convert.py FP4_TABLE)
                     rows [0, I_r)      = w1 (gate) rows [r*I_r, (r+1)*I_r) of the checkpoint
                     rows [I_r, 2*I_r)  = w3 (up)   rows [r*I_r, (r+1)*I_r)
                     (RoutedExperts._load_w13, fused_moe/routed_experts.py:523-572: shard_id "w1"
                      -> narrow(0, 0, I_r), "w3" -> narrow(0, I_r, I_r); i.e. [w1; w3], NOT [w3; w1])
  w13_weight_scale  uint8 [E, 2*I_r, H/32]  ue8m0 (e8m0fnu bytes), same row order
  w2_weight         uint8 [E, H, I_r/2]     columns [r*I_r/2, (r+1)*I_r/2) of the packed checkpoint
                     (RoutedExperts._load_w2, routed_experts.py:574-623: narrow(1, r*288, 288))
  w2_weight_scale   uint8 [E, H, I_r/32]    columns [r*I_r/32, (r+1)*I_r/32)
  with H = 5120, I_r = 2304 / 4 = 576, E = 384 (128 for the DSpark drafter layers).

Model semantics (inference/model.py Expert.forward): gate = w1 x, up = w3 x,
h = silu(min(gate, limit)) * clamp(up, -limit, limit), y = w2 (route_weight * h), limit = 10.
vLLM's SiluAndMulWithClamp (layers/activation.py:208-243) uses the same clamps.

Public API
  get_raw_experts(obj) -> (w13, s13, w2, s2)     raw tensors of a DeepseekV4MoE / MoERunner /
                                                  RoutedExperts using the ds41h method
  is_raw_moe(obj) -> bool
  moe_forward(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, impl=None) -> [T, H]
      routed-expert output of this TP rank (partial sum, not all-reduced, no shared expert).
  moe_forward_cuda(..., variant=0)  CUDA path (ds41h/kernels/moe_prefill.cu + moe_route.cu), graph safe;
                          variant 0 = the kernel default = W4A16 (lossless)
  moe_forward_w4a16(...)  CUDA W4A16: bf16 activations, FP4 weights expanded exactly to bf16 (ue8m0 scale folded
                          exactly), bf16 wgmma with fp32 accumulation, stock clamped SwiGLU, bf16 h, unscaled bf16
                          y per routed row, fp32 routing-weighted combine -- stock vLLM W4A16 numerics (only the
                          fp32 summation order differs; the SwiGLU input is the fp32 accumulator instead of bf16)
  moe_forward_w4a8(...)   CUDA W4A8 (LOSSY: e4m3 activations / h), opt-in (DS41H_MOE_PRECISION=fp8 or impl w4a8)
  moe_forward_ref(...)    pure-torch per-expert dequant path (eager only: host sync)
  dequant_fp4(w_packed, scale, dtype) -> [N, K]
  route_topk(logits, bias, topk, scale, renormalize, bias_vl=None, input_ids=None, image_lo=0)
      fused sqrtsoftplus + bias top-k kernel with stock dsv4_topk semantics (standalone)
  load_raw_experts(layer_idx, rank, device, model_dir=None, tp=4, experts=None)
      the same raw tensors read straight from the checkpoint (tests / tools)
  load_raw_shared_expert(layer_idx, rank, device, model_dir=None, tp=4)
      shared expert FP8 tensors of this TP rank in checkpoint layout (for decode kernels; the
      stock shared expert itself stays on vLLM's MergedColumnParallel/RowParallel linears)
  make_raw_mxfp4_moe_method(moe_config) -> FusedMoEMethodBase
      the vLLM quant method (installed by ds41h/patch_prefill.py)

Environment: DS41H_MOE=0 disables the component; DS41H_MOE_IMPL=cuda|w4a16|w4a8|ref selects the routed
forward (default cuda = the CUDA kernels at DS41H_MOE_PRECISION: bf16 = W4A16, lossless, default; fp8 = W4A8,
lossy opt-in); DS41H_MOE_VARIANT=v1+16*v2 forces the GEMM variants (moe_prefill.cu
dispatch_gemm; 0 = the kernel default). W4A16: GEMM1 9/10 = 256/128-token tiles, GEMM2 11/12 = 256/128-token
tiles (TMA loaders, asynchronous epilogue, slot rows + moe_combine; default 0xB9, bit-deterministic).
W4A8: GEMM1: 1 = 256-token tiles, cp.async gather loader; 2/3/4 =
128-token tiles with fp32 promotion every 4/2/1 K blocks; 5/6 = 256/128-token tiles fed by TMA from
activations in routed-row order (act_quant_perm), asynchronous epilogue. GEMM2: 1/2 = 256/128-token
tiles (cp.async); 3/4 = B-stationary kernel (slot output via bulk / 16-byte stores); 5 = B-stationary
+ bf16 reductions into the output (no slot buffer, no combine); 7/8 = TMA-fed 256/128-token tiles,
asynchronous epilogue; 9/10 = 7/8 with the bf16-reduction epilogue. The reduction variants (5, 9, 10)
are not bit-deterministic (accumulation order); DS41H_MOE_DETERMINISTIC=1 selects 0x75 (TMA GEMM1 +
TMA GEMM2 into the slot buffer + moe_combine) unless DS41H_MOE_VARIANT is set.

This module must stay cheap to import (it is imported from the import hook): torch, vLLM and the
CUDA extension are imported lazily.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any

from ds41h import flags

if TYPE_CHECKING:
    import torch


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


ENABLED = flags.ENABLED and (flags.DECODER or flags.PREFILL) and _env_flag("DS41H_MOE")
IMPL = os.environ.get("DS41H_MOE_IMPL", "cuda")
PRECISION = os.environ.get("DS41H_MOE_PRECISION", "bf16").strip().lower()  # bf16 (W4A16, lossless) | fp8 (W4A8, lossy)
if PRECISION not in ("bf16", "fp8"):
    raise ValueError(f"DS41H_MOE_PRECISION={PRECISION!r}: expected 'bf16' or 'fp8'")
VARIANT = int(os.environ.get("DS41H_MOE_VARIANT", "0"), 0)
if not VARIANT and PRECISION == "fp8" and _env_flag("DS41H_MOE_DETERMINISTIC", "0"):
    VARIANT = 0x75
W4A16_VARIANT = 0xB9  # moe_prefill.cu DEFAULT_V1 / DEFAULT_V2
W4A8_VARIANT = 0x95  # moe_prefill.cu FP8_V1 / FP8_V2


def is_w4a16_variant(variant: int) -> bool:
    """GEMM variants v1 + 16 * v2 select the W4A16 kernels (GEMM1 9..12, GEMM2 11..15; 0 = the W4A16 default)."""
    if variant & 0x100:  # small-batch kernel (moe_small16_kernel) for both GEMMs: 0x100 auto units, 0x300 64-token, 0x500 16-token
        # small16k: 0x500 | sub << 11 (moe_prefill.cu GEMM variants 17 + sub; same math, bit-identical
        # to 0x500): sub 1..4 = 16-token units with 2 / 4 K blocks per stage (wide swizzled weight boxes), 5 / 6 = 32-token
        # units (1 / 2 K blocks per stage); + 0x4000 = chunk-fastest unit order (the token chunks of one expert block run on
        # neighbouring CTAs and share its weight stream through L2)
        sub, base, cf = (variant >> 11) & 7, variant & 0x7FF, bool(variant & 0x4000)
        if not 0 <= variant < 0x8000 or (base & 0x600) == 0x600:
            return False
        return (sub == 0 and not cf) or (1 <= sub <= 6 and (base & 0x400) != 0)
    v1, v2 = variant & 15, (variant >> 4) & 15
    return (v1 in (0, 9, 10, 11, 12, 13, 14)) and (v2 in (0, 11, 12, 13, 14, 15))  # GEMM1 13/14: x-gather

HIDDEN = 5120
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@functools.cache
def ext():
    """The ds41h_moe CUDA extension (JIT built for sm_90a on first use)."""
    from ds41h import cuext

    return cuext.load("ds41h_moe", "moe_route.cu", "moe_prefill.cu")


def default_variant() -> int:
    """The GEMM variants (v1 + 16 * v2) moe_forward_cuda / impl 'cuda' use when called with variant=0."""
    if VARIANT:
        return VARIANT
    return W4A16_VARIANT if PRECISION == "bf16" else W4A8_VARIANT


# ============================================================================== raw tensor access
def _routed_experts(obj: Any):
    """RoutedExperts behind a DeepseekV4MoE (.experts = MoERunner), a MoERunner or itself."""
    seen = 0
    while obj is not None and seen < 4:
        if hasattr(obj, "w13_weight") and hasattr(obj, "w2_weight_scale"):
            return obj
        obj = getattr(obj, "routed_experts", None) or getattr(obj, "experts", None)
        seen += 1
    raise TypeError("expected a DeepseekV4MoE, MoERunner or RoutedExperts")


def is_raw_moe(obj: Any) -> bool:
    try:
        return bool(getattr(_routed_experts(obj), "ds41h_raw", False))
    except TypeError:
        return False


def get_raw_experts(obj: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w13, s13, w2, s2) uint8 tensors of this TP rank in raw checkpoint layout (see module doc).

    Valid after process_weights_after_loading of a layer that uses the ds41h raw method; raises
    otherwise (the stock Humming/Marlin layouts are not the raw layout).
    """
    re = _routed_experts(obj)
    if not getattr(re, "ds41h_raw", False):
        raise RuntimeError(f"{getattr(re, 'layer_name', re)}: experts are not in the ds41h raw layout")
    return re.w13_weight.data, re.w13_weight_scale.data, re.w2_weight.data, re.w2_weight_scale.data


def _check_raw(w13, s13, w2, s2) -> tuple[int, int, int]:
    import torch

    E, two_i, h2 = w13.shape
    H, I = h2 * 2, two_i // 2
    want = {
        "w13": (w13, (E, 2 * I, H // 2)),
        "s13": (s13, (E, 2 * I, H // 32)),
        "w2": (w2, (E, H, I // 2)),
        "s2": (s2, (E, H, I // 32)),
    }
    for name, (t, shape) in want.items():
        if tuple(t.shape) != shape or t.dtype != torch.uint8:
            raise ValueError(f"{name}: expected uint8 {shape}, got {t.dtype} {tuple(t.shape)}")
    return E, H, I


# ============================================================================== forward paths
def moe_forward(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float | None, impl: str | None = None):
    """Routed experts of this rank: sum_k topk_weights[t,k] * Expert_{ids[t,k]}(x[t]) -> x.dtype.

    topk_weights already contain renormalization and routed_scaling_factor (stock router output).
    Ids outside [0, E) contribute nothing.
    """
    impl = impl or IMPL
    # inference/model.py Expert.forward clamps only `if self.swiglu_limit > 0`; None / <= 0 = no clamp
    limit = float(swiglu_limit) if swiglu_limit is not None and float(swiglu_limit) > 0 else float("inf")
    if impl in ("cuda", "w4a16", "w4a8") and x.dtype.is_floating_point and x.is_cuda:
        if impl == "cuda":
            return moe_forward_cuda(x, topk_weights, topk_ids, w13, s13, w2, s2, limit, default_variant())
        fn = moe_forward_w4a16 if impl == "w4a16" else moe_forward_w4a8
        return fn(x, topk_weights, topk_ids, w13, s13, w2, s2, limit)
    if impl not in ("ref", "cuda", "w4a16", "w4a8"):
        raise ValueError(f"DS41H_MOE_IMPL={impl!r}: expected 'cuda', 'w4a16', 'w4a8' or 'ref'")
    import torch

    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("ds41h moe: the 'ref' implementation syncs with the host and cannot be graph captured")
    return moe_forward_ref(x, topk_weights, topk_ids, w13, s13, w2, s2, limit)


_A16_SCALES_OK: set = set()


def check_w4a16_scales(*scales) -> None:
    """The W4A16 kernels fold a ue8m0 byte S into the bf16 weights via a bf16 multiplier 2^(S-1): S <= 128 (scale
    <= 2) is required (the checkpoint has S in [117, 126]). Checked once per tensor (host sync; skipped inside a
    CUDA-graph capture, where the eager profile/warm-up run has already checked the same tensors)."""
    import torch

    for s in scales:
        key = (s.data_ptr(), s.numel())
        if key in _A16_SCALES_OK or (s.is_cuda and torch.cuda.is_current_stream_capturing()):
            continue
        mx = int(s.max())
        if mx > 128:
            raise ValueError(f"ds41h moe W4A16: ue8m0 scale byte {mx} > 128 (scale > 2) is not supported by the bf16 expansion")
        _A16_SCALES_OK.add(key)


def moe_forward_cuda(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float = 10.0, variant: int = 0, addend=None):
    """CUDA grouped-GEMM path; variant = v1 + 16 * v2 selects the GEMM1 / GEMM2 kernels (moe_prefill.cu
    dispatch_gemm; 0 = the kernel default = W4A16). W4A16 variants are lossless vs stock (see module doc),
    W4A8 variants quantize x and h to e4m3 (lossy). addend (bf16 [T, H], e.g. the shared-expert output) is added
    inside moe_combine (fp32 sum, one bf16 rounding)."""
    import torch

    _check_raw(w13, s13, w2, s2)
    if is_w4a16_variant(variant):
        check_w4a16_scales(s13, s2)
    xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    H = xb.shape[-1]
    _apply_hybrid()
    ad = None
    if addend is not None:
        ad = addend.reshape(-1, H)
        ad = ad if ad.dtype == torch.bfloat16 and ad.is_contiguous() else ad.to(torch.bfloat16).contiguous()
    out = ext().moe_forward(xb.reshape(-1, H), topk_weights.reshape(-1, topk_weights.shape[-1]), topk_ids.reshape(-1, topk_ids.shape[-1]), w13, s13, w2, s2, float(swiglu_limit), int(variant), ad)
    out = out.view(*x.shape[:-1], H)  # same leading shape as x (the extension works on [T, H])
    return out if out.dtype == x.dtype else out.to(x.dtype)


# Small-batch W4A16 variant: calls with <= SMALL_T tokens (decoder-replay layers: 128 rows per request,
# multi-request decode batches) are weight-streaming bound with a few rows per expert; DS41H_MOE_SMALL_T=0 disables.
SMALL_T = int(os.environ.get("DS41H_MOE_SMALL_T", "512"))  # default since the pf350_spv3 gate
# 0x500 = small16 with 16-token units at every size: the 64-token-unit kernel (0x300, the
# 02:30 default; 0x100 above 256 tokens) returned wrong rows under a concurrent side-stream load at T >= 1000 (tests/test_moe.py
# test_bit_repeatable_stress); 16-token units are clean, bit-identical in situ, and 3-14% faster for
# every T <= 512 (layer 10: T 64 544 vs 624 us, 128 715 vs 830, 256 921 vs 1026, 512 1148 vs 1188).
# small16k (moe_prefill.cu moe_small16k_kernel): the same per-element math as 0x500 (bit-identical:
# tests/test_moe.py test_small16k_bit_identical_to_0x500 + side-stream stress; in situ deterministic-prefill rows identical, k64)
# with 4 K blocks per pipeline stage (0x2500: 128-byte swizzled weight boxes) or 32-token units (0x3500: hot experts expand their
# weights once per 32 routed rows). In situ (k64, 350 W, k=7, NVMe deployment line): c16 decode 2,585 -> 2,799 tok/s with 0x2500
# (0x3500 2,707), c64 5,019 -> 5,779 with 0x3500 (0x2500 5,372). "auto" = 0x2500 up to SMALL_AUTO_T tokens per call, 0x3500
# above: DEFAULT (k64b, same line: c16 2,650 -> 2,840, c32 3,881 -> 4,208 (0x3500 at c32: 4,199),
# c64 5,087 -> 5,640; PP c16 +0.5 %). DS41H_MOE_SMALL_VARIANT=<int> / ctl moe_small_v pins one variant ("0x500" = the old default).
_SV = os.environ.get("DS41H_MOE_SMALL_VARIANT", "auto").strip().lower()
SMALL_AUTO = _SV == "auto"
SMALL_VARIANT = 0x2500 if SMALL_AUTO else int(_SV, 0)
SMALL_AUTO_T = int(os.environ.get("DS41H_MOE_SMALL_AUTO_T", "256"))
SMALL_AUTO_LO, SMALL_AUTO_HI = 0x2500, 0x3500
# moe_dec (decode-shaped kernels, ds41h/moe_dec.py + kernels/moe_dec.cu; outputs bit-identical to small16
# 0x500): DS41H_MOE_SMALL_VARIANT=0x8000 / ctl moe_small_v 0x8000 / DS41H_MOE_DEC=1 / ctl moe_dec run calls with <= SMALL_T tokens on
# ds41h.moe_dec.moe_forward_dec (kernel variants: DS41H_MOE_DEC_CFG); unsupported shapes or a missing module fall back to 0x500.
MOE_DEC_VARIANT = 0x8000
MOE_DEC = int(os.environ.get("DS41H_MOE_DEC", "0") or 0)
_MOE_DEC_WARNED: list = [False]


# Hybrid remainder tiles (moe_prefill.cu set_hybrid): the default W4A16 GEMMs run each expert's last
# partial token tile on moe_gemm16_kernel when it holds <= HYBRID rows (0 = off). DS41H_MOE_HYBRID / ctl moe_hybrid.
# DS41H_MOE_HYBRID_ENGINE / ctl moe_hybrid_eng: remainder kernel 0 = moe_gemm16_kernel, 1 = moe_small16_kernel (mma.sync).
HYBRID = int(os.environ.get("DS41H_MOE_HYBRID", "128"))
HYBRID_ENGINE = int(os.environ.get("DS41H_MOE_HYBRID_ENGINE", "0"))
_HYBRID_SET: list = [None]


def _apply_hybrid() -> None:
    h, eng = HYBRID, HYBRID_ENGINE
    if _CTL_PATH:
        from ds41h import prefill_ctl

        h = int(prefill_ctl.get("moe_hybrid", h))
        eng = int(prefill_ctl.get("moe_hybrid_eng", eng))
    if (h, eng) != _HYBRID_SET[0]:
        e = ext()
        if hasattr(e, "set_hybrid"):
            e.set_hybrid(int(h), int(eng))
        _HYBRID_SET[0] = (h, eng)


def moe_forward_w4a16(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float = 10.0, variant: int = 0, addend=None):
    """W4A16 (lossless): SMALL_VARIANT for <= SMALL_T tokens (unless DS41H_MOE_VARIANT is set), else variant (the
    per-layer A/B variant), else DS41H_MOE_VARIANT if that is a W4A16 variant, else 0xB9."""
    small_t, small_v, auto, dec = SMALL_T, SMALL_VARIANT, SMALL_AUTO, MOE_DEC
    if _CTL_PATH:
        from ds41h import prefill_ctl

        small_t = int(prefill_ctl.get("moe_small_t", small_t))
        sv = prefill_ctl.get("moe_small_v", None)
        if sv is not None:
            auto = str(sv).strip().lower() == "auto"
            small_v = SMALL_AUTO_LO if auto else int(str(sv), 0)
        dec = int(prefill_ctl.get("moe_dec", dec))
    if small_t and x.shape[0] <= small_t and not VARIANT and (dec or small_v == MOE_DEC_VARIANT):
        try:
            from ds41h import moe_dec
        except ImportError:
            moe_dec = None
        if moe_dec is not None and moe_dec.supported(w13, w2):
            return moe_dec.moe_forward_dec(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, addend=addend)
        if not _MOE_DEC_WARNED[0]:
            import logging

            logging.getLogger("ds41h.moe").warning("ds41h moe: moe_dec requested but %s -> small16 0x500",
                                                   "ds41h.moe_dec is missing" if moe_dec is None else "the shape is unsupported")
            _MOE_DEC_WARNED[0] = True
        small_v, auto = 0x500, False
    if auto:  # small16k by call size (both bit-identical to 0x500); the size is static per captured graph
        small_v = SMALL_AUTO_LO if x.shape[0] <= SMALL_AUTO_T else SMALL_AUTO_HI
    if small_t and x.shape[0] <= small_t and is_w4a16_variant(small_v) and not VARIANT:
        variant = small_v  # (also over a per-layer A/B variant: A/B targets the full-row calls)
    v = variant or (VARIANT if VARIANT and is_w4a16_variant(VARIANT) else W4A16_VARIANT)
    if not is_w4a16_variant(v):
        raise ValueError(f"moe_forward_w4a16: variant 0x{v:x} is not a W4A16 variant")
    return moe_forward_cuda(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, v, addend)


def moe_forward_w4a8(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float = 10.0, variant: int = 0):
    """W4A8 grouped-GEMM path (LOSSY opt-in): e4m3 activations (per-token / per-routed-row dynamic scales),
    FP4 weights expanded to e4m3 in-kernel with the ue8m0 scales folded in exactly.
    variant = v1 + 16 * v2 selects the GEMM1 / GEMM2 kernels (0 = DS41H_MOE_VARIANT if that is a W4A8
    variant, else 0x95)."""
    v = variant or (VARIANT if VARIANT and not is_w4a16_variant(VARIANT) else W4A8_VARIANT)
    if is_w4a16_variant(v):
        raise ValueError(f"moe_forward_w4a8: variant 0x{v:x} is a W4A16 variant")
    return moe_forward_cuda(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, v)


def dequant_fp4(w_packed, scale, dtype=None):
    """Packed e2m1 [..., N, K/2] (low nibble = even k) x ue8m0 [..., N, K/32] -> [..., N, K]."""
    import torch

    dtype = dtype or torch.float32
    w = w_packed.view(torch.uint8)
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=w.device)
    vals = torch.stack([lut[(w & 0x0F).long()], lut[(w >> 4).long()]], dim=-1).flatten(-2)
    s = scale.view(torch.uint8).float().sub_(127.0).exp2_()
    return (vals * s.repeat_interleave(32, dim=-1)).to(dtype)


def moe_forward_ref(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float = 10.0, compute_dtype=None):
    """Reference routed-expert forward: per-expert dequant of the raw weights, torch matmuls.

    compute_dtype float32 (default for x.dtype float32) gives the exact-weight reference used by
    the tests; bf16 matches stock W4A16 numerics. Peak extra memory: one dequantized expert.
    """
    import torch
    import torch.nn.functional as F

    E, H, I = _check_raw(w13, s13, w2, s2)
    cdt = compute_dtype or (torch.float32 if x.dtype == torch.float32 else torch.bfloat16)
    x2 = x.reshape(-1, H)
    T, K = topk_ids.shape
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    ids = topk_ids.reshape(-1).long()
    wts = topk_weights.reshape(-1).float()
    valid = (ids >= 0) & (ids < E)
    order = torch.argsort(torch.where(valid, ids, E), stable=True)
    counts = torch.bincount(ids[valid], minlength=E).tolist()  # host sync
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        idx = order[start : start + n]
        start += n
        tok = idx // K
        gu = x2[tok].to(cdt) @ dequant_fp4(w13[e], s13[e], cdt).t()
        g, u = gu.float().split(I, dim=-1)
        h = F.silu(g.clamp(max=swiglu_limit)) * u.clamp(-swiglu_limit, swiglu_limit) * wts[idx, None]
        y = h.to(cdt) @ dequant_fp4(w2[e], s2[e], cdt).t()
        out.index_add_(0, tok, y.float())
    return out.to(x.dtype).reshape(*x.shape[:-1], H)


def route_topk(logits, bias, topk: int, scale: float, renormalize: bool = True, bias_vl=None, input_ids=None, image_lo: int = 0):
    """Fused routing kernel: sqrt(softplus(logits)) + bias top-k, weights renormalized * scale.

    Same semantics as vLLM dsv4_topk (fused_moe/router/dsv4_topk.py) for any E <= 512, topk <= 32.
    Returns (weights f32 [T, topk], ids int32 [T, topk]).
    """
    import torch

    T = logits.shape[0]
    w = torch.empty(T, topk, dtype=torch.float32, device=logits.device)
    ids = torch.empty(T, topk, dtype=torch.int32, device=logits.device)
    if bias_vl is not None:
        # vLLM's input_ids buffer is int32 (the stock Triton kernel casts to int64 on load); the
        # CUDA kernel reads int64 and an f32 contiguous bias_vl.
        if input_ids is None:
            raise ValueError("route_topk: bias_vl routing requires input_ids")
        bias_vl = bias_vl.float().contiguous()
        input_ids = input_ids.reshape(-1).to(torch.int64).contiguous()
    ext().moe_topk(logits.float().contiguous(), bias.float().contiguous(), bias_vl, input_ids, int(image_lo), float(scale), bool(renormalize), w, ids)
    return w, ids


# ============================================================================== checkpoint loaders
def load_raw_experts(layer_idx: int, rank: int, device=None, model_dir: str | None = None, tp: int = 4, experts=None, prefix: str = "layers"):
    """Raw tensors of `layer_idx` for TP rank `rank`, read from the checkpoint with the same slicing
    as the stock vLLM loader (ds41h.ref.weights TP rules). Returns (w13, s13, w2, s2) on `device`."""
    import torch

    from ds41h.ref.weights import CheckpointReader, default_reader

    rd = CheckpointReader(model_dir) if model_dir else default_reader()
    lay = rd.layer(layer_idx) if prefix == "layers" else rd.mtp(layer_idx)
    if experts is None:
        experts = range(int(rd.config["n_routed_experts"] if prefix == "layers" else rd.config["dspark_n_routed_experts"]))
    w13, s13, w2, s2 = [], [], [], []
    u8 = torch.uint8
    for e in experts:
        p = f"ffn.experts.{e}."
        w13.append(torch.cat([lay.tp(p + "w1.weight", rank, tp).view(u8), lay.tp(p + "w3.weight", rank, tp).view(u8)]))
        s13.append(torch.cat([lay.tp(p + "w1.scale", rank, tp).view(u8), lay.tp(p + "w3.scale", rank, tp).view(u8)]))
        w2.append(lay.tp(p + "w2.weight", rank, tp).view(u8))
        s2.append(lay.tp(p + "w2.scale", rank, tp).view(u8))
    return tuple(torch.stack(t).to(device).contiguous() for t in (w13, s13, w2, s2))


def load_raw_shared_expert(layer_idx: int, rank: int, device=None, model_dir: str | None = None, tp: int = 4):
    """Shared expert of `layer_idx`, TP rank `rank`, checkpoint layout:
    (w13 e4m3 [2*576, 5120] = [w1 rows; w3 rows], s13 ue8m0-as-uint8 [36, 160],
     w2 e4m3 [5120, 576], s2 uint8 [160, 18]) -- 32x32 block scales."""
    import torch

    from ds41h.ref.weights import CheckpointReader, default_reader

    rd = CheckpointReader(model_dir) if model_dir else default_reader()
    lay = rd.layer(layer_idx)
    p = "ffn.shared_experts."
    w13 = torch.cat([lay.tp(p + "w1.weight", rank, tp), lay.tp(p + "w3.weight", rank, tp)])
    s13 = torch.cat([lay.tp(p + "w1.scale", rank, tp).view(torch.uint8), lay.tp(p + "w3.scale", rank, tp).view(torch.uint8)])
    w2 = lay.tp(p + "w2.weight", rank, tp)
    s2 = lay.tp(p + "w2.scale", rank, tp).view(torch.uint8)
    return tuple(t.to(device).contiguous() for t in (w13, s13, w2, s2))


# ============================================================================== vLLM integration
@functools.cache
def _method_cls():
    import torch

    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.utils import set_weight_attrs

    logger = init_logger("ds41h.moe")

    class Ds41hRawMxfp4MoEMethod(FusedMoEMethodBase):
        """MXFP4 routed experts kept in the raw checkpoint layout (no Humming/Marlin repack).

        Modular FusedMoE method: the stock router (FusedTopKBiasRouter -> dsv4_topk) selects the
        experts, MoERunner adds the shared expert and does the TP all-reduce
        (runner/moe_runner.py:688-797); apply() returns this rank's routed partial sum.
        """

        def __init__(self, moe):
            super().__init__(moe)
            self.weight_dtype = "mxfp4"

        @property
        def supports_eplb(self) -> bool:
            return False

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
            # Parameter set identical to Mxfp4MoEMethod.create_weights (quantization/mxfp4.py:659-759) so
            # the stock RoutedExperts.weight_loader and expert mapping load the checkpoint unchanged.
            if getattr(self.moe, "has_bias", False):
                raise NotImplementedError("ds41h raw MXFP4 MoE: expert biases are not supported")
            if self.moe.w13_num_shards != 2:
                raise NotImplementedError("ds41h raw MXFP4 MoE: gated (w1/w3) experts only")
            self.num_experts = num_experts
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size_per_partition
            layer.params_dtype = params_dtype
            layer.num_experts = num_experts
            weight_loader = extra_weight_attrs.pop("weight_loader")
            scale_loader = Mxfp4MoEMethod.get_scale_weight_loader(weight_loader)  # e8m0fnu -> uint8 view
            i2 = 2 * intermediate_size_per_partition

            def param(name, shape, loader, block_scale=False):
                p = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.uint8), requires_grad=False)
                layer.register_parameter(name, p)
                set_weight_attrs(p, extra_weight_attrs)
                set_weight_attrs(p, {"weight_loader": loader})
                if block_scale:
                    p.quant_method = "block"  # FusedMoeWeightScaleSupported.BLOCK -> group/block scale loading
                return p

            param("w13_weight", (num_experts, i2, hidden_size // 2), weight_loader)
            param("w13_weight_scale", (num_experts, i2, hidden_size // 32), scale_loader, True)
            param("w2_weight", (num_experts, hidden_size, intermediate_size_per_partition // 2), weight_loader)
            param("w2_weight_scale", (num_experts, hidden_size, intermediate_size_per_partition // 32), scale_loader, True)

        def process_weights_after_loading(self, layer) -> None:
            par = self.moe.moe_parallel_config
            if getattr(par, "use_ep", False) or getattr(par, "ep_size", 1) > 1:
                raise NotImplementedError("ds41h raw MXFP4 MoE supports tensor parallelism only (no EP)")
            act = getattr(layer.activation, "value", str(layer.activation))
            if "silu" not in str(act).lower():
                raise NotImplementedError(f"ds41h raw MXFP4 MoE: activation {act} (expected silu)")
            if getattr(layer, "apply_router_weight_on_input", False):
                raise NotImplementedError("ds41h raw MXFP4 MoE: apply_router_weight_on_input")
            for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
                p = getattr(layer, name)
                if not p.data.is_contiguous():
                    p.data = p.data.contiguous()
            _check_raw(layer.w13_weight, layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale)
            layer.ds41h_raw = True
            layer.ds41h_moe_impl = _glue_impl(layer)
            layer.ds41h_moe_variant = _ab_variant(layer) if layer.ds41h_moe_impl in ("w4a16", "cuda") else 0
            layer.ds41h_layer_idx = _layer_idx(layer)
            if layer.ds41h_moe_impl in ("w4a8", "w4a16", "cuda", "ref_eager") and layer.w13_weight.is_cuda:
                ext()  # JIT-build (or load) the extension now, not inside the first forward / profile run
            logger.info_once(
                "ds41h: MXFP4 routed experts kept in raw checkpoint layout (%s, w13 %s); forward impl %s",
                layer.layer_name,
                tuple(layer.w13_weight.shape),
                layer.ds41h_moe_impl,
            )

        def get_fused_moe_quant_config(self, layer):
            return None

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            impl = getattr(layer, "ds41h_moe_impl", None) or _glue_impl(layer)
            if impl == "ref_eager":
                # Exact cross-check mode: the bf16 per-expert torch reference (host sync) for every forward that contains
                # prefill tokens; decode-only batches and CUDA-graph captures use the (lossless) W4A16 kernel.
                capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
                info = None
                if not capturing:
                    from ds41h.patch_core import current_step

                    info = current_step()
                impl = "w4a16" if (capturing or (info is not None and info.decode_only)) else "ref"
            v = getattr(layer, "ds41h_moe_variant", 0)
            if _CTL_PATH:
                from ds41h import prefill_ctl

                spec = prefill_ctl.get("moe_ab", None)
                if spec is not None:
                    v = _ab_pick(str(spec), getattr(layer, "ds41h_layer_idx", -1))
            if v and impl in ("w4a16", "cuda") and x.is_cuda:
                lim = getattr(layer, "swiglu_limit", None)
                lim = float(lim) if lim is not None and float(lim) > 0 else float("inf")
                return moe_forward_cuda(x, topk_weights, topk_ids, layer.w13_weight, layer.w13_weight_scale, layer.w2_weight,
                                        layer.w2_weight_scale, lim, v)
            return moe_forward(
                x,
                topk_weights,
                topk_ids,
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
                getattr(layer, "swiglu_limit", None),
                impl=impl,
            )

    return Ds41hRawMxfp4MoEMethod


_DRAFTER_FIRST_LAYER = 40  # DeepSeek-V4.1-Flash num_hidden_layers: DSpark drafter layers are model.layers.40-42


def _glue_impl(layer) -> str:
    """Forward impl of one raw-MXFP4 MoE layer (vLLM glue): DS41H_MOE_IMPL if set, else the numerics
    policy (ds41h/flags.py: 'w4a16' lossless default, 'w4a8' with DS41H_LOSSY=1). DSpark drafter layers
    (approximations allowed, outputs stay exact through rejection sampling) use 'w4a8' unless DS41H_MOE_IMPL is set."""
    import re

    env = os.environ.get("DS41H_MOE_IMPL")
    if env:
        return env
    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
    if m and int(m.group(1)) >= _DRAFTER_FIRST_LAYER:
        return "w4a8"
    return flags.MOE_IMPL


_CTL_PATH = os.environ.get("DS41H_PREFILL_CTL", "")


def forward_layer(layer, x, topk_weights, topk_ids, addend=None):
    """Lossless W4A16 forward of one raw-MXFP4 RoutedExperts layer as its apply() would run it (per-layer A/B variant,
    small-T variant), with an optional bf16 addend (the shared-expert output) folded into moe_combine. Prefill SP
    (ds41h/prefill_sp.py) uses it to skip the runner's separate shared + routed add. Returns None if the layer does
    not run the W4A16 CUDA path (the caller then uses the stock runner)."""
    impl = getattr(layer, "ds41h_moe_impl", None) or _glue_impl(layer)
    if impl not in ("w4a16", "cuda") or not x.is_cuda or (impl == "cuda" and PRECISION != "bf16"):
        return None
    v = getattr(layer, "ds41h_moe_variant", 0)
    if _CTL_PATH:
        from ds41h import prefill_ctl

        spec = prefill_ctl.get("moe_ab", None)
        if spec is not None:
            v = _ab_pick(str(spec), getattr(layer, "ds41h_layer_idx", -1))
    lim = getattr(layer, "swiglu_limit", None)
    lim = float(lim) if lim is not None and float(lim) > 0 else float("inf")
    return moe_forward_w4a16(x, topk_weights, topk_ids, layer.w13_weight, layer.w13_weight_scale, layer.w2_weight,
                             layer.w2_weight_scale, lim, v, addend=addend)


def _layer_idx(layer) -> int:
    import re

    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
    return int(m.group(1)) if m else -1


@functools.lru_cache(maxsize=256)
def _ab_pick(spec: str, idx: int) -> int:
    if idx < 0 or idx >= _DRAFTER_FIRST_LAYER:
        return 0
    vs = [int(v, 0) for v in spec.split(",") if v.strip()]
    v = vs[idx % len(vs)] if vs else 0
    return v if v and is_w4a16_variant(v) else 0


def _ab_variant(layer) -> int:
    """Per-layer W4A16 variant for in-server A/B timing: DS41H_MOE_AB="0xB9,0xDB" runs target layer i
    with entry i % n (all W4A16 variants share the lossless numerics; non-W4A16 entries are ignored). 0 = default.
    With DS41H_PREFILL_CTL the control file's "moe_ab" (if present) overrides it per call."""
    spec = os.environ.get("DS41H_MOE_AB", "").strip()
    return _ab_pick(spec, _layer_idx(layer)) if spec else 0


def make_raw_mxfp4_moe_method(moe_config):
    return _method_cls()(moe_config)
