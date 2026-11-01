"""Bit-exact number-format helpers for the DeepSeek-V4.1 reference (pure torch, any device).

Everything here is a *simulation in float32*: inputs and outputs are float tensors, and the
quantize/dequantize round trip reproduces the values the real kernels produce.

Sources (paths relative to the official checkpoint's ``inference/`` and to vLLM 0.32.0):
  * FP8 activation quant  -- inference/kernel.py:22-124 (act_quant, fast_round_scale)
  * FP4 activation quant  -- inference/kernel.py:127-204 (fp4_quant_kernel)
  * FP4 weight table      -- inference/convert.py:13-15 (FP4_TABLE, low nibble = even k)
  * vLLM sm90 "V4" KV record (584 B) -- models/deepseek_v41/common/ops/cache_utils.py:37-180,
    fused_compress_quant_cache.py:304-352 (identical math, ``fmaxf(amax, 1e-4)``)
  * vLLM indexer K cache  -- models/deepseek_v41/common/ops/indexer_k_store.py:210-255
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# float32(1/448) and float32(1/6), exactly what the TileLang kernels multiply by
# (kernel.py:46 ``fp8_max_inv = 1 / fp8_max``; :132 ``fp4_max_inv``).
_INV_FP8_MAX = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32).item()
_INV_FP4_MAX = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32).item()

# e2m1 code -> value (convert.py:13). Code 8 is -0 in OCP e2m1; convert.py writes +0.0.
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def fp4_table(device=None) -> torch.Tensor:
    return torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)


def e8m0_table(device=None) -> torch.Tensor:
    """ue8m0 byte -> float32: 2**(b-127); 255 is NaN. 2**-127 is an fp32 subnormal, exact."""
    vals = [2.0 ** (i - 127) for i in range(255)] + [float("nan")]
    return torch.tensor(vals, dtype=torch.float64, device=device).to(torch.float32)


def as_u8(t: torch.Tensor) -> torch.Tensor:
    """Raw bytes of a 1-byte tensor (float8_e8m0fnu / float8_e4m3fn / int8 / uint8)."""
    if t.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {t.dtype}")
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    """Decode ue8m0 scales (float8_e8m0fnu or raw uint8 bytes) to float32 exactly."""
    return e8m0_table(s.device)[as_u8(s).long()]


def e4m3_to_float(w: torch.Tensor) -> torch.Tensor:
    """Decode e4m3fn values (float8_e4m3fn or raw uint8 bytes) to float32 exactly."""
    if w.dtype == torch.float8_e4m3fn:
        return w.float()
    return as_u8(w).view(torch.float8_e4m3fn).float()


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] bytes (int8/uint8, two e2m1 per byte, low nibble = even element) -> [..., K] f32."""
    b = as_u8(packed).long()
    tab = fp4_table(packed.device)
    lo, hi = tab[b & 0xF], tab[(b >> 4) & 0xF]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] e2m1 codes (0..15, int) -> [..., K/2] uint8, low nibble = even element."""
    c = codes.to(torch.uint8).unflatten(-1, (-1, 2))
    return (c[..., 0] | (c[..., 1] << 4)).contiguous()


def fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """Exact e2m1 grid values (as produced by round_to_e2m1) -> codes 0..15 (sign bit = 8)."""
    a = x.abs()
    grid = torch.tensor(FP4_VALUES[:8], dtype=torch.float32, device=x.device)
    idx = (a.unsqueeze(-1) == grid).float().argmax(-1)
    if not bool(((a.unsqueeze(-1) == grid).any(-1)).all()):
        raise ValueError("fp4_codes: input not on the e2m1 grid")
    neg = torch.signbit(x) & (a > 0)
    return idx + 8 * neg.long()


# ----------------------------------------------------------------------------- rounding


def round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round float32 to the nearest e4m3fn value (RNE, like cvt.rn.satfinite.e4m3x2.f32).

    Callers must clamp to [-448, 448] first (all kernels do); torch maps overflow to NaN.
    """
    return x.float().to(torch.float8_e4m3fn).float()


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} with ties-to-even codes, saturating at 6.

    Ties: .25->0, .75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4 (even mantissa code wins), which is
    what cvt.rn.satfinite.e2m1x2.f32 / __nv_cvt_float_to_fp4(cudaRoundNearest) produce.
    """
    a = x.float().abs()
    q = torch.where(
        a <= 0.25, 0.0,
        torch.where(a < 0.75, 0.5,
        torch.where(a <= 1.25, 1.0,
        torch.where(a < 1.75, 1.5,
        torch.where(a <= 2.5, 2.0,
        torch.where(a < 3.5, 3.0,
        torch.where(a <= 5.0, 4.0, 6.0)))))))  # fmt: skip
    return torch.copysign(q, x.float())


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive normal float32 x, exactly (kernel.py:22-37 bit trick)."""
    m, e = torch.frexp(x.float())  # x = m * 2**e, m in [0.5, 1)
    e = e - (m == 0.5).to(e.dtype)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e)


def ue8m0_scale(amax: torch.Tensor, qmax_inv: float) -> torch.Tensor:
    """Power-of-two scale 2**ceil(log2(amax * qmax_inv)) with the fp32 product the kernels use."""
    return pow2_ceil(amax.float() * torch.tensor(qmax_inv, dtype=torch.float32, device=amax.device))


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by block {block}")
    return x.float().unflatten(-1, (-1, block))


# ----------------------------------------------------------------------------- FP8 activations


def fp8_quant(x: torch.Tensor, block: int = 32, amax_min: float = 1e-4, scale_fmt: str = "ue8m0"):
    """FP8 e4m3 per-`block` (last dim) quantization.

    Returns (q float8_e4m3fn [..., K], scale float32 [..., K/block]); dequant = q * scale.
    scale_fmt='ue8m0' matches inference/kernel.py act_quant(round_scale=True): amax clamped at
    1e-4, s = 2**ceil(log2(amax * fp32(1/448))), q = e4m3(clamp(x / s, -448, 448)).
    scale_fmt='fp32': s = amax * fp32(1/448) (kernel.py act_quant without round_scale, the usual
    per-token-group FP8 of W8A8 GEMMs).
    """
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1).clamp_min(amax_min)
    if scale_fmt == "ue8m0":
        s = ue8m0_scale(amax, _INV_FP8_MAX)
    elif scale_fmt == "fp32":
        s = amax * torch.tensor(_INV_FP8_MAX, dtype=torch.float32, device=amax.device)
    else:
        raise ValueError(scale_fmt)
    q = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(-2), s


def act_quant_sim(x: torch.Tensor, block: int = 32, scale_fmt: str = "ue8m0") -> torch.Tensor:
    """Fake-quant (quantize then dequantize) of kernel.py act_quant(..., inplace=True), in float32.

    The kernel writes the dequantized value back in the input dtype (bf16 in model.py); with
    ue8m0 scales the result e4m3 * 2**k is exactly representable in bf16 (no extra rounding).
    """
    q, s = fp8_quant(x, block, scale_fmt=scale_fmt)
    return (q.float().unflatten(-1, (-1, block)) * s.unsqueeze(-1)).flatten(-2)


# ----------------------------------------------------------------------------- FP4 activations


def fp4_quant_sim(x: torch.Tensor, block: int = 32, scale_fmt: str = "ue8m0") -> torch.Tensor:
    """Fake-quant of kernel.py fp4_act_quant(..., inplace=True) in float32.

    scale_fmt='ue8m0' (indexer q/k, block 32): amax >= 6*2**-126, s = 2**ceil(log2(amax/6)).
    scale_fmt='e4m3'  (compressed KV, block 16): amax >= 6*2**-9, s = e4m3(amax / 6).
    Then v = e2m1(clamp(x / s, -6, 6)) * s.
    """
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1)
    if scale_fmt == "ue8m0":
        s = ue8m0_scale(amax.clamp_min(6 * 2.0**-126), _INV_FP4_MAX)
    elif scale_fmt == "e4m3":
        s = round_to_e4m3(amax.clamp_min(6 * 2.0**-9) / FP4_MAX)
    else:
        raise ValueError(scale_fmt)
    v = round_to_e2m1((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)) * s.unsqueeze(-1)
    return v.flatten(-2)


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


# ----------------------------------------------------------------------------- vLLM sm90 KV records

V4_NOPE = 448
V4_ROPE = 64
V4_DATA_BYTES = 576  # 448 e4m3 + 64 bf16
V4_SCALE_BYTES = 8  # 7 ue8m0 (one per 64 NoPE dims) + 1 zero pad
V4_BYTES_PER_TOKEN = V4_DATA_BYTES + V4_SCALE_BYTES  # 584


def v4_record_quant(kv: torch.Tensor):
    """Quantize post-RoPE KV rows [N, 512] the way stock vLLM fills the sm90 V4 record.

    cache_utils.py:124-170 / fused_compress_quant_cache.py:326-351: the 448 NoPE dims are split
    into 7 groups of 64, amax clamped at 1e-4, exponent = ceil(log2(amax/448)), e4m3 of
    clamp(x * 2**-exponent); the 64 RoPE dims are stored as bf16. The kernels read the KV as
    bf16 first, so the input is rounded to bf16 here.
    Returns (nope_e4m3 [N,448] float8, exponents int [N,7] (scale = 2**e), rope_bf16 [N,64]).
    """
    x = bf16_round(kv)
    nope = x[..., :V4_NOPE].unflatten(-1, (7, 64))
    amax = nope.abs().amax(-1).clamp_min(1e-4)
    s = ue8m0_scale(amax, _INV_FP8_MAX)
    q = (nope / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).flatten(-2)
    _, e = torch.frexp(s)
    return q, (e - 1), x[..., V4_NOPE:].to(torch.bfloat16)


def v4_record_dequant(q: torch.Tensor, exps: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    nope = q.float().unflatten(-1, (7, 64)) * torch.ldexp(torch.ones_like(exps, dtype=torch.float32), exps).unsqueeze(-1)
    return torch.cat([nope.flatten(-2), rope.float()], dim=-1)


def v4_record_sim(kv: torch.Tensor) -> torch.Tensor:
    """Round trip through the V4 record (what FlashMLA on sm90 sees for SWA/compressed KV)."""
    return v4_record_dequant(*v4_record_quant(kv))


def v4_pack(kv: torch.Tensor):
    """Post-RoPE KV [N,512] -> (data uint8 [N,576], scales uint8 [N,8]) record bytes."""
    q, e, rope = v4_record_quant(kv)
    data = torch.cat([as_u8(q), rope.contiguous().view(torch.uint8)], dim=-1)
    scales = torch.zeros(*kv.shape[:-1], 8, dtype=torch.uint8, device=kv.device)
    scales[..., :7] = (e + 127).clamp(0, 255).to(torch.uint8)
    return data, scales


def v4_unpack(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    q = data[..., :V4_NOPE].contiguous().view(torch.float8_e4m3fn)
    rope = data[..., V4_NOPE:].contiguous().view(torch.bfloat16)
    e = scales[..., :7].long() - 127
    return v4_record_dequant(q, e, rope)


def v4_cache_insert(cache: torch.Tensor, block_size: int, slots: torch.Tensor, kv: torch.Tensor) -> None:
    """Write rows into a paged V4 cache viewed as uint8 [num_blocks, block_stride] (stride >= B*584).

    Page layout (cache_utils.py:83-116): token data [B*576) then scales [B*8) per page.
    """
    data, scales = v4_pack(kv)
    blk, off = (slots // block_size).long(), (slots % block_size).long()
    for i in range(slots.numel()):
        if int(slots[i]) < 0:
            continue
        b, o = int(blk[i]), int(off[i])
        cache[b, o * V4_DATA_BYTES : (o + 1) * V4_DATA_BYTES] = data[i]
        base = block_size * V4_DATA_BYTES + o * V4_SCALE_BYTES
        cache[b, base : base + V4_SCALE_BYTES] = scales[i]


def v4_cache_gather(cache: torch.Tensor, block_size: int, slots: torch.Tensor) -> torch.Tensor:
    """Read and dequantize rows [N, 512] float32 from a paged V4 cache (see v4_cache_insert)."""
    s = slots.long().reshape(-1)
    blk, off = s // block_size, s % block_size
    didx = off.unsqueeze(-1) * V4_DATA_BYTES + torch.arange(V4_DATA_BYTES, device=cache.device)
    sidx = block_size * V4_DATA_BYTES + off.unsqueeze(-1) * V4_SCALE_BYTES + torch.arange(8, device=cache.device)
    rows = cache[blk]
    data = rows.gather(1, didx)
    scales = rows.gather(1, sidx)
    return v4_unpack(data, scales).reshape(*slots.shape, 512)


def indexer_k_fp8_sim(k: torch.Tensor) -> torch.Tensor:
    """vLLM sm90 default indexer K cache (indexer_k_store.py:238-255): one power-of-two scale per
    128-dim row, amax clamped at 1e-4, e4m3 of clamp(x * 2**-e). Input is the bf16 row."""
    x = bf16_round(k)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = ue8m0_scale(amax, _INV_FP8_MAX)
    return round_to_e4m3((x / s).clamp(-FP8_MAX, FP8_MAX)) * s
