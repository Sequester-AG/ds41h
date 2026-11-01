"""ds41h 'prologue' component, part 1: skinny FP8-weight GEMM for decode (M <= 8 tokens) on sm90a.

Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/skinny.py). Kernel: ds41h/kernels/skinny.cu (FP16 HMMA with exact e4m3 -> f16 weight expansion
and a per-32 post-scale; products exact, fp32 accumulation, bf16 out -- the W8A16 numerics class of
stock vLLM's sm90 path).

Weights. Stock vLLM 0.32.0 runs the DeepSeek-V4.1 dense FP8 linears through
``ModelOptLinearMethod(QuantSpec(kMxfp8Static, kMxfp8Dynamic))`` (models/deepseek_v41/quant_config.py
:221-243) whose sm90 kernel (Humming) repacks ``layer.weight`` in ``process_weights_after_loading``.
Right before that, the loader has left exactly the checkpoint bytes of this TP rank in
``layer.weight`` (e4m3 [N_local, K_local]) and ``layer.weight_scale`` (ue8m0 bytes, row-expanded to
[N_local, K_local/32] by KMxfp8Static.get_scale_weight_loader, quantization/modelopt.py:2278-2292), so
``install()`` wraps that method and keeps a skinny-packed copy (``layer._ds41h_skinny``) before the
stock kernel repacks. This happens at load time, i.e. before vLLM profiles memory and before CUDA
graph capture. ``pack_checkpoint()`` builds the same copy from the checkpoint
(ds41h.ref.weights.CheckpointReader + tp_slice) for tests and as a fallback.

MMA flavour (env DS41H_PROLOGUE_MMA, shared with ds41h.prologue; default 'hmma'): 'wgmma' runs the same
packed weights through RS-wgmma m64n8k32 e4m3 with the reference per-(token, 32) ue8m0 activation quant
(official W8A8 numerics; ~3x fewer instructions per weight byte, matters at the 345 MHz power-capped
clock). K splits beyond one CTA reduce through thread-block-cluster shared memory (deterministic).

Drop-in. The wrapped ``ModelOptLinearMethod.apply`` routes 2-D bf16 inputs with 1..8 rows to
``gemm()`` for the DeepSeek-V4.1 decode GEMMs (fused wq_a|wkv, wq_b, indexer wq_b, wo_b, shared expert
gate_up / down; MTP layers included); anything else (prefill, bias, bmm wo_a) stays stock. The launch
allocates only its output (graph-pool safe), runs on the current stream and has no data-dependent host
control flow, so FULL decode CUDA graphs capture it.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import torch

from ds41h import cuext, flags

logger = logging.getLogger("vllm.ds41h.prologue")

MAX_TOKENS = 8  # B operand n8 (ds41h/kernels/skinny.cu MAXM)
KW_MAX = 20  # k32 blocks per warp kept in registers
MAX_K = 8 * KW_MAX * 32  # 5120: one CTA covers all of K
MMA_CODES = {"hmma": 0, "wgmma": 1}  # kernels/skinny.cu MMA_HMMA / MMA_WGMMA
_MMA_ENV = os.environ.get("DS41H_PROLOGUE_MMA", "hmma").strip().lower() or "hmma"
MMA = _MMA_ENV if _MMA_ENV in MMA_CODES else "hmma"

# Capture the checkpoint weights whenever the prologue component is on (the fused prologue kernel
# reads the fused_wqa_wkv / wq_b copies); route stock decode GEMMs to the skinny kernel only if
# DS41H_SKINNY is also on (default on).
CAPTURE = flags.PROLOGUE
APPLY = flags.PROLOGUE and flags._flag("DS41H_SKINNY")

# vLLM module prefixes of the DeepSeek-V4.1 FP8 linears that run as dense decode GEMMs
# (attention.py:364-403 / :1313-1320; deepseek_v4/nvidia/model.py:155-176), MTP layers included.
TARGET_RE = re.compile(
    r"(?:^|\.)layers\.\d+\.(?:attn\.(?:fused_wqa_wkv|wq_b|wo_b|indexer\.wq_b)"
    r"|ffn\.shared_experts\.(?:gate_up_proj|down_proj))$"
)


@dataclass(frozen=True)
class SkinnyWeight:
    """A skinny-packed FP8 weight: w uint8 [N/16 * K/32 * 512] (fragment-major), s uint8 [N/16, KBP]."""

    w: torch.Tensor
    s: torch.Tensor
    n: int
    k: int

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel()


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def _u8(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 1:
        raise TypeError(f"expected a 1-byte tensor, got {t.dtype}")
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def tile_scales(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """ue8m0 scales in checkpoint block form [ceil(N/32), K/32], vLLM's row-expanded form [N, K/32]
    or per-16-row-tile form [N/16, K/32] -> [N/16, K/32] uint8 (one byte per 16-row tile and k32 block).
    Raises if the scale varies inside a 16-row tile (the kernel applies one scale per tile)."""
    s = _u8(scale.contiguous())
    kb = k // 32
    if s.dim() != 2 or s.shape[1] != kb:
        raise ValueError(f"scale shape {tuple(s.shape)} does not match K/32 = {kb}")
    rows = s.shape[0]
    if rows == n:
        t = s.view(n // 16, 16, kb)
        if not bool((t == t[:, :1]).all()):
            raise ValueError("weight scale varies inside a 16-row tile; not a 32x32 (or 16-row) block scale")
        return t[:, 0].contiguous()
    if rows == n // 16:
        return s
    if rows == -(-n // 32):
        return s.repeat_interleave(2, dim=0)[: n // 16].contiguous()
    raise ValueError(f"cannot interpret scale with {rows} rows for N={n}")


def pack(weight: torch.Tensor, scale: torch.Tensor) -> SkinnyWeight:
    """e4m3 weight [N, K] (float8_e4m3fn or raw uint8) + ue8m0 scale -> SkinnyWeight on weight's device.

    Fragment-major layout (ds41h/kernels/skinny.cu header): chunk (tile rt, block kb) of 512 bytes, lane
    l = 4g + t holds W[16rt + g][32kb + 8t : +8] followed by W[16rt + g + 8][32kb + 8t : +8]."""
    if weight.dim() != 2:
        raise ValueError("weight must be 2-D [N, K]")
    n, k = weight.shape
    if n % 16 or k % 32:
        raise ValueError(f"skinny pack needs N % 16 == 0 and K % 32 == 0, got {n}x{k}")
    kb = k // 32
    w8 = _u8(weight.contiguous())
    w = w8.view(n // 16, 2, 8, kb, 4, 8).permute(0, 3, 2, 4, 1, 5).contiguous().view(-1)
    st = tile_scales(scale.to(weight.device), n, k)
    s = torch.zeros(n // 16, _round_up(kb, 4), dtype=torch.uint8, device=weight.device)
    s[:, :kb] = st
    return SkinnyWeight(w=w, s=s, n=int(n), k=int(k))


def unpack(sw: SkinnyWeight) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack: (e4m3 bytes uint8 [N, K], tile scales uint8 [N/16, K/32]) (tests)."""
    n, k = sw.n, sw.k
    kb = k // 32
    w = sw.w.view(n // 16, kb, 8, 4, 2, 8).permute(0, 4, 2, 1, 3, 5).contiguous().view(n, k)
    return w, sw.s[:, :kb].contiguous()


def dequant(sw: SkinnyWeight) -> torch.Tensor:
    """float32 [N, K] value of a packed weight (tests / references)."""
    w, s = unpack(sw)
    sf = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
    return w.view(torch.float8_e4m3fn).float() * sf.repeat_interleave(16, 0).repeat_interleave(32, 1)


def pack_checkpoint(reader: Any, names: list[str], rank: int | None, device: Any = "cuda", tp: int = 4) -> SkinnyWeight:
    """Pack checkpoint FP8 weights `names` (each '<prefix>.weight' with a '<prefix>.scale'), stacked along
    N in the given order, as TP rank `rank` of stock vLLM loads them (ds41h.ref.weights.CheckpointReader /
    tp_slice; rank None = unsharded). E.g. ['layers.3.attn.wq_a.weight', 'layers.3.attn.wkv.weight'] is
    the fused_wqa_wkv of layer 3."""
    ws, ss = [], []
    for name in names:
        w = reader.get_tp(name, rank, tp)
        s = reader.get_tp(name.removesuffix(".weight") + ".scale", rank, tp)
        ws.append(_u8(w))
        ss.append(_u8(s))
    return pack(torch.cat(ws, 0).to(device), torch.cat(ss, 0).to(device))


@functools.cache
def ext() -> Any:
    """The JIT-built extension (sm_90a); sets the kernels' smem attributes on first use (needs a GPU)."""
    e = cuext.load("ds41h_skinny", "skinny.cu")
    if torch.cuda.is_available():
        e.init()
    return e


def gemm(x: torch.Tensor, sw: SkinnyWeight, out: torch.Tensor | None = None, mma: str | None = None) -> torch.Tensor:
    """y [M, N] bf16 = x [M, K] bf16 @ W^T for 1 <= M <= 8 (x rows unit-stride and 16-byte aligned).
    mma: 'hmma' | 'wgmma' (default DS41H_PROLOGUE_MMA)."""
    return ext().gemm(x, sw.w, sw.s, sw.n, sw.k, out, MMA_CODES[(mma or MMA).lower()])


def split_config(n: int, k: int, mma: str | None = None) -> tuple[int, ...]:
    """Launch decomposition for an [n, k] weight: hmma -> (k-parts per CTA,); wgmma -> (kw, kp, s, ctas)."""
    if (mma or MMA).lower() == "wgmma":
        return tuple(int(v) for v in ext().choose_split_wg(n, k))
    return (int(ext().choose_split(n, k)),)


def eligible(x: Any, sw: SkinnyWeight) -> bool:
    """Whether gemm() can take `x` for weight `sw` (host-side, capture-time-static checks only)."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.shape[1] == sw.k
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and x.device == sw.w.device
    )


def is_target(prefix: str) -> bool:
    return bool(TARGET_RE.search(prefix or ""))


def capture(method: Any, layer: torch.nn.Module) -> SkinnyWeight | None:
    """Keep a skinny copy of `layer`'s checkpoint FP8 weight (called before the stock repack)."""
    existing = layer.__dict__.get("_ds41h_skinny")
    if existing is not None:
        return existing
    prefix = getattr(layer, "prefix", "") or ""
    if not is_target(prefix) or getattr(layer, "is_bmm", False):
        return None
    if not APPLY:  # without the drop-in only the fused prologue's two weights are needed
        prologue_weight = prefix.endswith("attn.fused_wqa_wkv") or (
            prefix.endswith("attn.wq_b") and not prefix.endswith("indexer.wq_b")
        )
        if not prologue_weight:
            return None
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static

    if getattr(getattr(method, "spec", None), "weight", None) != kMxfp8Static:
        return None
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dim() != 2 or w.dtype != torch.float8_e4m3fn or s.element_size() != 1 or not w.is_cuda:
        return None
    n, k = w.shape
    if n % 16 or k % 32 or k > MAX_K:
        return None
    sw = pack(w.data, s.data)
    layer._ds41h_skinny = sw
    if APPLY:
        ext()  # JIT build + smem attributes now, not during CUDA graph capture
    if prefix.endswith("attn.fused_wqa_wkv"):
        # Allocate the prologue scratch and build both extensions now: before vLLM's memory profile
        # and long before CUDA graph capture (no cudaMalloc / JIT during capture).
        from ds41h import prologue

        prologue.ensure_device(w.device)
    return sw


def install(module: Any) -> None:
    """Wrap ModelOptLinearMethod.process_weights_after_loading / apply of the imported
    vllm.model_executor.layers.quantization.modelopt module (idempotent)."""
    cls = module.ModelOptLinearMethod
    if getattr(cls, "_ds41h_skinny_installed", False):
        return
    base_pwal = cls.process_weights_after_loading
    base_apply = cls.apply
    pre_processed = getattr(module, "is_weights_pre_processed", None)

    def process_weights_after_loading(self, layer):
        if CAPTURE and not (pre_processed is not None and pre_processed()):
            try:
                sw = capture(self, layer)
                if sw is not None:
                    logger.debug("ds41h skinny: captured %s [%d x %d]", getattr(layer, "prefix", "?"), sw.n, sw.k)
            except Exception:  # noqa: BLE001 - a failed capture must leave the stock path intact
                logger.exception("ds41h skinny: weight capture failed for %s; stock path kept", getattr(layer, "prefix", "?"))
                layer.__dict__.pop("_ds41h_skinny", None)
        return base_pwal(self, layer)

    def apply(self, layer, x, bias=None):
        sw = layer.__dict__.get("_ds41h_skinny")
        if APPLY and sw is not None and bias is None and eligible(x, sw):
            return gemm(x, sw)
        return base_apply(self, layer, x, bias)

    process_weights_after_loading.__wrapped__ = base_pwal  # type: ignore[attr-defined]
    apply.__wrapped__ = base_apply  # type: ignore[attr-defined]
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    cls._ds41h_skinny_installed = True
