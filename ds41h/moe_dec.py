"""ds41h moe_dec: decode-shaped W4A16 routed-MoE path for small calls.

Multi-request decode at c2..c64 with k=7 runs the vLLM layer path with 16..512 tokens per MoE call (prefill replay chunks
<= SMALL_T too). There the routed experts are a weight stream (~1.5-1.8 GB per layer per GPU, nearly every expert touched
from ~128 tokens on) and the old small-T kernel (moe_prefill.cu moe_small16_kernel, forward variant 0x500) is bound by its
FP4 -> bf16 expansion on the ALU pipe. ds41h/kernels/moe_dec.cu keeps small16's numerics BIT FOR BIT (same exact
expansion, same lane -> physical-k map and k16 order, fp32 accumulation over the whole K, same epilogues and moe_combine)
and changes only speed: an IMAD-based expansion (4 instead of 5 instructions per bf16x2 pair, 2 instead of 2.5+ on the ALU
pipe), units of up to 8 * NGMAX tokens (hot experts are expanded once per unit instead of once per 16 tokens), wide TMA
weight boxes (KPS 64-wide K blocks per stage) and 8-row token boxes.

Public API
  ext()                                   the ds41h_moe_dec CUDA extension (moe_route.cu + moe_dec.cu, JIT built)
  moe_forward_dec(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, cfg=None, addend=None) -> [T, H]
      routed-expert output of this TP rank (= ds41h.moe.moe_forward_cuda(..., 0x500) bit for bit)
  supported(w13, w2) -> bool              shape check (E <= 512, I % 64 == 0, H % 128 == 0, K/32 scale rows <= 18 bytes)

Environment: DS41H_MOE_DEC_CFG = GEMM1 variant | GEMM2 variant << 4 (moe_dec.cu launch_dec_v; default DEFAULT_CFG).
"""

from __future__ import annotations

import functools
import os

DEFAULT_CFG = 0x55  # KPS 2 (64-byte boxes), 3 stages, 32-token units, 3 CTAs / SM for both GEMMs: in situ (ab1q / ab2q, 350 W)
# 0x55 beat 0x22 at c64 (5,781 vs 5,591 tok/s; hot experts expand once per 32 rows) and matched it at c16 / c32
CFG = int(os.environ.get("DS41H_MOE_DEC_CFG", hex(DEFAULT_CFG)), 0)
MAX_E = 512
# DS41H_MOE_DEC_CHECK=1 (debug, in-situ proof of bit identity): every call ALSO runs the certified small16 path (variant
# 0x500) on the same inputs and counts differing output elements on the GPU (graph-capturable, no sync); the counters are
# logged from eager calls (prefill steps) every DS41H_MOE_DEC_CHECK_EVERY (default 20) eager calls. Doubles the MoE cost.
CHECK = os.environ.get("DS41H_MOE_DEC_CHECK", "0") not in ("0", "")
CHECK_EVERY = int(os.environ.get("DS41H_MOE_DEC_CHECK_EVERY", "20"))
_check: dict = {}  # device index -> int64 [3] (calls, differing elements, elements compared); "eager" -> eager call count


@functools.cache
def ext():
    """The ds41h_moe_dec CUDA extension (JIT built for sm_90a on first use)."""
    from ds41h import cuext

    return cuext.load("ds41h_moe_dec", "moe_route.cu", "moe_dec.cu")


@functools.cache
def ext_prof():
    """Diagnostic build (DS41H_MOE_DEC_PROF=1): same kernels + clock64 accounting (prof_reset / prof_read); never served."""
    from ds41h import cuext

    return cuext.load("ds41h_moe_dec_prof", "moe_route.cu", "moe_dec.cu", extra_cuda_cflags=("-DDS41H_MOE_DEC_PROF=1",))


def supported(w13, w2) -> bool:
    E, two_i, h2 = w13.shape
    H, I = 2 * h2, two_i // 2
    return E <= MAX_E and I % 64 == 0 and H % 128 == 0 and (I // 32) <= 18 and H % 64 == 0


def moe_forward_dec(x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit: float = 10.0, cfg: int | None = None, addend=None):
    """W4A16 routed experts via the moe_dec kernels (same outputs as moe_forward_cuda(..., variant=0x500)).
    topk_weights include renormalization and the routed scaling factor; ids outside [0, E) contribute nothing.
    addend (bf16 [T, H], e.g. the shared-expert output) is added inside moe_combine (fp32 sum, one bf16 rounding)."""
    import torch

    from ds41h import moe

    moe._check_raw(w13, s13, w2, s2)
    moe.check_w4a16_scales(s13, s2)
    xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    H = xb.shape[-1]
    ad = None
    if addend is not None:
        ad = addend.reshape(-1, H)
        ad = ad if ad.dtype == torch.bfloat16 and ad.is_contiguous() else ad.to(torch.bfloat16).contiguous()
    lim = float(swiglu_limit) if swiglu_limit is not None and float(swiglu_limit) > 0 else float("inf")
    x2, tw2, ids2 = xb.reshape(-1, H), topk_weights.reshape(-1, topk_weights.shape[-1]), topk_ids.reshape(-1, topk_ids.shape[-1])
    out = ext().moe_forward(x2, tw2, ids2, w13, s13, w2, s2, lim, int(CFG if cfg is None else cfg), ad)
    if CHECK:
        _check_against_small16(out, x2, tw2, ids2, w13, s13, w2, s2, lim, ad)
    out = out.view(*x.shape[:-1], H)
    return out if out.dtype == x.dtype else out.to(x.dtype)


def _check_against_small16(out, x, tw, ids, w13, s13, w2, s2, lim, ad) -> None:
    import logging

    import torch

    from ds41h import moe

    ref = moe.ext().moe_forward(x, tw.float().contiguous(), ids.int().contiguous(), w13, s13, w2, s2, lim, 0x500, ad)
    dev = out.device.index if out.device.index is not None else torch.cuda.current_device()
    cnt = _check.get(dev)
    if cnt is None:
        cnt = _check[dev] = torch.zeros(3, dtype=torch.int64, device=out.device)
    cnt[0] += 1
    cnt[1] += (out.view(torch.int16) != ref.view(torch.int16)).sum()
    cnt[2] += out.numel()
    if not torch.cuda.is_current_stream_capturing():
        _check["eager"] = _check.get("eager", 0) + 1
        if _check["eager"] % CHECK_EVERY == 1:
            c = cnt.tolist()
            logging.getLogger("ds41h.moe_dec").warning(
                "ds41h moe_dec CHECK (dev %d): %d calls compared with small16 0x500 (graph replays included), %d of %d output "
                "elements differ", dev, c[0], c[1], c[2])
