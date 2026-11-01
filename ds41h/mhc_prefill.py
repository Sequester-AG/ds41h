"""ds41h 'prefill' component, part 3: one fused kernel per mHC sublayer boundary in prefill (sm90).

Stock vLLM 0.32.0 runs every mHC boundary with more than 32 tokens (all of prefill) unfused on sm90
(model_executor/kernels/mhc/tilelang.py:269-497 ``mhc_fused_post_pre_delayed_tilelang`` falls back when
``mhc_fused_post_pre_split_config`` returns None): ``mhc_post_tilelang`` writes the post-mapped residual,
DeepGEMM ``tf32_hc_prenorm_gemm`` re-reads it for the 24 mixes + square sum, and
``mhc_pre_big_fuse_with_norm_tilelang`` reads it a third time for the Sinkhorn coefficients and the delayed
collapse + RMSNorm -- ~41 ms per 8192-token chunk at TP4 (41 full-row boundaries). The collapse uses the
PREVIOUS sublayer's pre-mix, so nothing downstream of the GEMV feeds the layer input: ds41h/kernels/mhc_prefill.cu
does all of it in one pass (read x + residual, write residual_out + layer input; see the kernel header for the
work split, the bf16 hi/lo tensor-core GEMV and the cluster-wide reductions).

LOSSLESS: same math as stock -- exact (IEEE) division in softmax / Sinkhorn / RMS means,
accurate expf in softmax / sigmoid (the kernel is built without --use_fast_math, see ``ext``), fp32 everywhere stock
uses fp32, bf16 rounding where stock rounds; the 24-mix GEMV is MORE precise than stock (bf16 hi + lo fn vs TF32 fn).

Numerics vs stock: residual_out uses stock's operation order (fp32 post * x, then one FFMA per stream, round to
bf16) and is bit-identical up to FMA contraction differences; the mixes are computed on the bf16 stream as in
stock but with fn split into bf16 hi + lo (~2^-17 relative per product instead of TF32's 2^-11), so post / comb /
pre are slightly closer to the fp32 reference than stock; the collapse and RMSNorm follow stock's rounding points
(bf16 collapse, fp32 square sum of the rounded values, (u * rsqrt) * w).

Patched (DS41H=1, DS41H_PREFILL != 0, DS41H_PREFILL_MHC=1; ds41h/patch_mhc.py), module
vllm.models.deepseek_v41.nvidia.model:
  mhc_shifted_post_pre        -> fused kernel (POST mode) for plain-tensor x with >= MIN_T rows, no overlap
                                 stream, no fused all-reduce, outside CUDA-graph capture; else stock
  mhc_pre_delayed_tilelang    -> fused kernel (PRE mode: residual as is) for the Engram-layer / non-broadcast
                                 calls with the same conditions (the first layer's broadcast call stays stock)
  DeepseekV4Model.finalize_mhc_broadcast_weights -> also packs every layer's hc_attn_fn / hc_ffn_fn at load time
                                 (1.97 MB each, ~160 MB per GPU for 40 layers) so the KV-cache sizing sees it
Switches: DS41H_PREFILL_MHC (default ON: unit tests + end-to-end KL gate), DS41H_PREFILL_MHC_IMPL (1..4,
default 1), DS41H_PREFILL_MHC_MIN_T (default 64).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

from ds41h import flags

logger = logging.getLogger("vllm.ds41h.prefill")

_ON = flags.ENABLED and flags.PREFILL
MHC = _ON and flags._flag("DS41H_PREFILL_MHC", "1")  # default ON (impl 1: end-to-end KL gate passed 4x)
MIN_T = max(1, int(os.environ.get("DS41H_PREFILL_MHC_MIN_T", "64")))
# 1 (default): 8-CTA cluster per 64-token tile (non-persistent: the scheduler balances it against concurrent work).
# 2: one CTA pair per SM; 3: persistent 16-CTA clusters, fn slice in registers, bulk row-segment loads, DSMEM
# reduction pipeline; 4: impl 3's streaming core with no cross-CTA sync + a finishing kernel.
# Sustained alone at 200 W (T = 16384 / 8192 / 2048 / 128 tokens, us): stock 2391 / 1217 / 341 / 34, impl 1 2552 /
# 1119 / 301 / 42, impl 3 2046 / 1158 / 276 / 29, impl 4 - / 1204 / - / - (all power-capped). IN THE SERVER impl 3 is
# 1.7x slower than impl 1 (end-to-end, 16K step: 164 vs 97 ms; full-row calls 3.4-5.8 ms instead of 2.05):
# its static persistent schedule needs all 7 clusters of 16 SMs at once, which concurrent prefill kernels break.
IMPL = int(os.environ.get("DS41H_PREFILL_MHC_IMPL", "1"))
# Cross-proxy fence audit (docs/EXACTNESS.md, cross-proxy fences): impl 2-4 refill their cp.async.bulk rings after
# generic smem reads without fence.proxy.async (the moe small16 race pattern); impl 1 (default) has no async-proxy smem traffic.
if MHC and IMPL in (2, 3, 4):
    logger.warning("ds41h mHC prefill impl %d: its bulk-copy rings lack the cross-proxy fence (EXACTNESS item 11) -- not for "
                   "production; use the default impl 1", IMPL)

HC, HID, MIX = 4, 5120, 24
SH, GSTEPS = 32, 160
PACKED_ATTR = "_ds41h_mhc_packed"


EXT_NAME = "ds41h_mhc_prefill_exact"


@functools.cache
def ext() -> Any:
    """JIT build of mhc_prefill.cu WITHOUT --use_fast_math (unlike the other ds41h extensions): stock's tilelang mHC
    kernels compile without fast math, i.e. IEEE division, the accurate expf (softmax / sigmoid) and no FTZ; the
    lossless rule requires the same norm / softmax / Sinkhorn math here."""
    from torch.utils.cpp_extension import load as _load

    from ds41h import cuext

    nvcc = [f for f in cuext.NVCC_FLAGS if f != "--use_fast_math"] + ["-ftz=false", "-prec-div=true", "-prec-sqrt=true"]
    return _load(
        name=EXT_NAME,
        sources=[str(cuext.KERNELS / "mhc_prefill.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=nvcc,
        extra_include_paths=[str(cuext.KERNELS)],
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")),
    )


# ============================================================================================ weights
def pack_fn(fn: "Any", impl: int | None = None) -> "Any":
    """fp32 [24, 20480] hc fn -> bf16 hi/lo (hi = bf16(fn), lo = bf16(fn - hi)) in the per-lane fragment layout of
    the kernel. Both layouts are [160 steps of 32 columns][...] with 12 KB per step (contiguous, cp.async.bulk).

    impl 1 (fn = mma B operand): [step][4 streams][3 n-tiles][hi, lo][8 g][4 t][8]: lane g * 4 + t of a
        (step, stream, n-tile, hi/lo) block holds fn_{hi|lo}[8 j + g][stream * 5120 + 32 step + 8 t + e].
    impl 2 (fn = mma A operand, 48 rows = 24 hi + 24 lo): [step][4 streams][3 m-tiles][2 k-blocks][8 g][4 t]
        [4 words][2]: word (p, rh) = 2 p + rh of lane (g, t) holds row 16 m + 8 rh + g, columns
        stream * 5120 + 32 step + 8 t + 4 kb + 2 p + {0, 1}."""
    import torch

    impl = impl or IMPL
    impl = 2 if impl in (3, 4) else impl
    assert fn.shape == (MIX, HC * HID), fn.shape
    f = fn.detach().float()
    hi = f.to(torch.bfloat16)
    lo = (f - hi.float()).to(torch.bfloat16)
    if impl == 1:
        w = torch.stack([hi, lo])  # [hl, n, k]
        w = w.view(2, 3, 8, HC, GSTEPS, 4, 8).permute(4, 3, 1, 0, 2, 5, 6)  # gs, o, j, hl, g, t, e
        return w.contiguous()
    w = torch.cat([hi, lo])  # [48, k]
    w = w.view(3, 2, 8, HC, GSTEPS, 4, 2, 2, 2)  # mt, rh, g, o, gs, t, kb, p, e
    return w.permute(4, 3, 0, 6, 2, 5, 7, 1, 8).contiguous()  # gs, o, mt, kb, g, t, p, rh, e


def unpack_fn(packed: "Any", impl: int | None = None) -> "Any":
    """Inverse of pack_fn (hi + lo in fp32), for tests."""
    impl = impl or IMPL
    if impl == 1:
        w = packed.float().view(GSTEPS, HC, 3, 2, 8, 4, 8).permute(3, 2, 4, 1, 0, 5, 6).reshape(2, MIX, HC * HID)
        return w[0] + w[1]
    w = packed.float().view(GSTEPS, HC, 3, 2, 8, 4, 2, 2, 2).permute(2, 7, 4, 1, 0, 5, 3, 6, 8).reshape(2 * MIX, HC * HID)
    return w[:MIX] + w[MIX:]


def packing_of(impl: int | None = None) -> int:
    """fn packing layout used by `impl` (v3 / v4 use the v2 layout)."""
    impl = impl or IMPL
    return 2 if impl in (3, 4) else impl


def packed(fn: "Any", impl: int | None = None) -> "Any":
    """Packed copy of `fn` for `impl`, cached on the tensor object (packed lazily if the load-time hook did not run)."""
    impl = impl or IMPL
    impl = 2 if impl in (3, 4) else impl  # v3 / v4 use the v2 packing
    attr = f"{PACKED_ATTR}{impl}"
    p = getattr(fn, attr, None)
    if p is None or p.device != fn.device:
        p = pack_fn(fn, impl)
        setattr(fn, attr, p)
    return p


def pack_model(model: Any) -> int:
    """Pack hc_attn_fn / hc_ffn_fn of every decoder layer of a DeepseekV4Model. Returns the number packed."""
    n = 0
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0
    for layer in layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name, None)
            if fn is not None and getattr(fn, "is_cuda", False) and tuple(fn.shape) == (MIX, HC * HID):
                packed(fn, IMPL)
                n += 1
    return n


# ============================================================================================ kernels
def boundary(x, residual, post_mix, comb_mix, pre_mix, fn, hc_scale, hc_base, norm_weight, rms_eps: float,
             hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
             norm_eps: float, capture_aux: bool = False, impl: int | None = None):
    """One boundary. x None -> PRE mode. Returns (residual_out [T,4,5120], post [T,4,1], comb [T,4,4],
    layer_input [T,5120], pre [T,4], aux [T|0, 5120]) -- stock mhc_shifted_post_pre's tuple."""
    post = comb = None
    if x is not None:
        T = residual.shape[0]
        post = post_mix.reshape(T, HC)
        comb = comb_mix.reshape(T, HC, HC)
        post = post if post.is_contiguous() else post.contiguous()
        comb = comb if comb.is_contiguous() else comb.contiguous()
    pre = pre_mix if pre_mix is None or pre_mix.is_contiguous() else pre_mix.contiguous()
    impl = impl or IMPL
    res_out, post_out, comb_out, layer_in, pre_out, aux = ext().mhc_boundary(
        x, residual, post, comb, pre, packed(fn, impl), hc_scale, hc_base, norm_weight, float(rms_eps), float(hc_pre_eps),
        float(hc_sinkhorn_eps), float(hc_post_mult_value), int(sinkhorn_repeat), float(norm_eps), bool(capture_aux), int(impl),
    )  # fmt: skip
    return res_out, post_out.unsqueeze(-1), comb_out, layer_in, pre_out, aux


def boundary_ref(x, residual, post_mix, comb_mix, pre_mix, fn, hc_scale, hc_base, norm_weight, rms_eps: float,
                 hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
                 norm_eps: float, capture_aux: bool = False, gemm_dtype=None):
    """Torch reference with stock's rounding points (fp32 math; the mixes GEMV in `gemm_dtype`, default fp64)."""
    import torch

    T = residual.shape[0]
    if x is not None:
        post = post_mix.reshape(T, HC, 1).float()
        comb = comb_mix.reshape(T, HC, HC).float()
        acc = post * x.float().unsqueeze(1)
        for i in range(HC):  # stock: post * x, then one FFMA per input stream (in order)
            acc = acc + comb[:, i, :].unsqueeze(-1) * residual[:, i, :].float().unsqueeze(1)
        res = acc.to(torch.bfloat16)
    else:
        res = residual
    rf = res.float().reshape(T, HC * HID)
    gd = gemm_dtype or torch.float64
    mixes = (rf.to(gd) @ fn.to(gd).t()).float()
    sq = rf.double().square().sum(-1).float()
    rms = torch.rsqrt(sq / (HC * HID) + rms_eps)
    m = mixes * rms[:, None]
    s, b = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[:, :HC] * s[0] + b[:HC]) + hc_pre_eps
    post_o = torch.sigmoid(m[:, HC : 2 * HC] * s[1] + b[HC : 2 * HC]) * hc_post_mult_value
    cm = (m[:, 2 * HC :] * s[2] + b[2 * HC :]).view(T, HC, HC)
    cm = torch.softmax(cm, -1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    if pre_mix is None:
        u = res[:, 0, :].float()
    else:
        u = (pre_mix.float().unsqueeze(-1) * res.float()).sum(1)
    u = u.to(torch.bfloat16).float()
    rn = torch.rsqrt(u.square().mean(-1, keepdim=True) + norm_eps)
    layer_in = ((u * rn) * norm_weight.float()).to(torch.bfloat16)
    aux = res.float().mean(1).to(torch.bfloat16) if capture_aux else res.new_empty(0, HID)
    return res, post_o.unsqueeze(-1), cm, layer_in, pre, aux


# ============================================================================================ vLLM glue
def _eligible(residual, fn, norm_weight, stream, x=None) -> bool:
    import torch

    if norm_weight is None or stream is not None:
        return False
    if not isinstance(residual, torch.Tensor) or residual.dim() != 3 or tuple(residual.shape[1:]) != (HC, HID):
        return False
    if residual.shape[0] < MIN_T or residual.dtype != torch.bfloat16 or not residual.is_cuda or not residual.is_contiguous():
        return False
    if x is not None and (not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous()):
        return False
    if tuple(fn.shape) != (MIX, HC * HID) or norm_weight.dtype != torch.bfloat16 or norm_weight.numel() != HID:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return available()


@functools.cache
def available() -> bool:
    """Extension builds/loads and runs on sm90 (checked once; logs why not)."""
    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)):
            return False
        ext()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("ds41h prefill mhc: extension unavailable; stock mHC kept")
        return False


_LOGGED: set[str] = set()


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


def make_shifted_post_pre(stock_fn: Any) -> Any:
    def mhc_shifted_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None, norm_weight=None,
                             norm_eps=1e-6, capture_aux=False, *, stream=None, reduce_results=False):
        if not reduce_results and _eligible(residual, fn, norm_weight, stream, x):
            _log_once("post_pre", "ds41h prefill mhc: fused post+pre boundary active (T=%d)", residual.shape[0])
            return boundary(x, residual, post_layer_mix, comb_res_mix, pre_mix, fn, hc_scale, hc_base, norm_weight, rms_eps,
                            hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps, capture_aux)
        return stock_fn(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                        hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps,
                        capture_aux=capture_aux, stream=stream, reduce_results=reduce_results)

    mhc_shifted_post_pre.__wrapped__ = stock_fn  # type: ignore[attr-defined]
    return mhc_shifted_post_pre


def make_pre_delayed(stock_fn: Any) -> Any:
    def mhc_pre_delayed_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                                 sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
        # x given = the first layer's broadcast-embedding call (fn is [24, 5120]): stock.
        if x is None and _eligible(residual, fn, norm_weight, None):
            _log_once("pre", "ds41h prefill mhc: fused pre boundary active (T=%d)", residual.shape[0])
            _, post, comb, layer_in, pre, _ = boundary(None, residual, None, None, pre_mix, fn, hc_scale, hc_base, norm_weight,
                                                       rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                                                       sinkhorn_repeat, norm_eps)
            return post, comb, layer_in, pre
        return stock_fn(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                        sinkhorn_repeat, pre_mix=pre_mix, x=x, norm_weight=norm_weight, norm_eps=norm_eps)

    mhc_pre_delayed_tilelang.__wrapped__ = stock_fn  # type: ignore[attr-defined]
    return mhc_pre_delayed_tilelang


def install(module: Any) -> None:
    """Patch ``module`` = vllm.models.deepseek_v41.nvidia.model (idempotent)."""
    if getattr(module, "_ds41h_mhc_installed", False):
        return
    module.mhc_shifted_post_pre = make_shifted_post_pre(module.mhc_shifted_post_pre)
    module.mhc_pre_delayed_tilelang = make_pre_delayed(module.mhc_pre_delayed_tilelang)
    model_cls = module.DeepseekV4Model
    base_finalize = model_cls.finalize_mhc_broadcast_weights

    def finalize_mhc_broadcast_weights(self) -> None:
        base_finalize(self)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) and available():
                n = pack_model(self)  # after the (load-time) extension build, so prefill never JIT-builds
                logger.info("ds41h prefill mhc: packed %d hc fn weights (%.1f MB)", n, n * MIX * HC * HID * 4 / 2**20)
        except Exception:  # noqa: BLE001 - packing falls back to the first use
            logger.exception("ds41h prefill mhc: load-time packing failed; packing lazily")

    finalize_mhc_broadcast_weights.__wrapped__ = base_finalize  # type: ignore[attr-defined]
    model_cls.finalize_mhc_broadcast_weights = finalize_mhc_broadcast_weights
    module._ds41h_mhc_installed = True
    logger.info("ds41h prefill mhc: patched %s (min_t=%d)", module.__name__, MIN_T)
