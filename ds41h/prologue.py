"""ds41h 'prologue' component, part 2: fused decode prologue of a DeepSeek-V4.1 decoder layer (sm90a).

One persistent launch (ds41h/kernels/prologue.cu, adapted from BRRR's prologue_kernel, Apache-2.0)
replaces, for M <= 8 tokens on non-Engram layers with a residual stream, everything stock vLLM 0.32.0
runs between the previous layer's MoE output and sparse attention:

  mhc_shifted_post_pre(...)                       models/deepseek_v41/nvidia/model.py:552-575
    (sm90: mhc_fused_post_pre_delayed_tilelang, model_executor/kernels/mhc/tilelang.py:269-497)
  fused_wqa_wkv GEMM                              attention.py:908-912 (replicated, disable_tp=True)
  fused_q_kv_rmsnorm                              attention.py:773-806, models/common/ops/fused_qk_rmsnorm.py
  wq_b GEMM (TP slice, 16 heads)                  attention.py:845-849, :924-930
  fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (q RoPE, KV RoPE + FP8 V4 record + SWA insert)
                                                  attention.py:1011-1078

and returns tensors with exactly the stock dtypes/shapes, plus q in the ds41h attention layout
(contiguous [M, 16, 512] bf16 with RoPE, no 16 -> 64 head padding). The decoder component calls
``run_prologue`` when ``eligible`` (and the stock code otherwise); KV-source / index-source layers feed
``a`` (attention input, = stock ``hidden_states`` of DeepseekV4Attention.forward) and ``qr`` (q_norm
output, bf16, what the stock indexer receives since sm90 never fuses the query quant,
query_quant.py:130-151) to the stock compressor / indexer calls.

Weights: the fused wq_a|wkv and wq_b copies are the ones ds41h.skinny captured from the checkpoint
bytes at load time (``attn.fused_wqa_wkv._ds41h_skinny`` / ``attn.wq_b._ds41h_skinny``); if missing they
are packed from the checkpoint (ds41h.ref.weights.CheckpointReader + tp_slice) on first use (eager only).

Switches (environment, read at import; both default to the Phase-1 behaviour until validated on GPU):
  DS41H_PROLOGUE_MMA=hmma|wgmma  GEMM flavour of the fused prologue AND of the skinny drop-in
      (ds41h.skinny): 'hmma' = FP16 HMMA, bf16 activations (stock W8A16 numerics); 'wgmma' = RS-wgmma
      m64n8k32 e4m3 with the reference per-(token, 32) ue8m0 activation quant (official W8A8 numerics,
      ~3x fewer SASS instructions per weight byte, see kernels/skinny.cu).
  DS41H_PROLOGUE_FAST=<bitmask>  kernels/prologue.cu `fast` bits (unset: FAST_DEFAULT = 4083200 = V3 | V3_HOP |
      V3_PACE | V3_LATE_W2 | V3_FAC | V3_ILV | V3_PUSH | V3_SK_G1 | V3_KV_SPREAD, bit-identical to 0 with hmma): 1 EARLY (wq_a|wkv starts as soon as the collapse, which uses the
      incoming pre_mix, is published), 2 SINKHORN (rcp.approx instead of IEEE divisions: LOSSY, opt-in only),
      32 LIGHT_SYNC, 64 CLUSTER, 128 QR_DEFER (wgmma), 512 V3 (separate kernel, see prologue_kernel_v3), 1024
      V3_LATE_W2, 2048 V3_HOP, 4096 V3_ADEFER (wgmma, LOSSY), 8192 V3_W1_LATE, 16384 V3_PACE, 32768 V3_PACE1, and
      (all bit-identical with hmma) 65536 V3_SK_G2 (Sinkhorn on idle G2 CTAs), 131072 V3_FAC (precomputed
      HMMA promotion factors), 262144 V3_ILV (G1 / G2 clusters interleaved in the grid), 524288 V3_PUSH (G1 k-part
      partials pushed by st.async into the finisher CTA), 1048576 V3_SK_G1 (Sinkhorn per token on G1 CTAs after their
      push), 2097152 V3_KV_SPREAD (KV insert per token on separate G1 CTAs), 4194304 V3_L2PF (weights HBM -> L2 by
      bulk prefetch early, L2 -> smem after the latency-critical copies). See the kernel header.
      '1' keeps its legacy meaning (= 3); 33 = the previous default (regular kernel).
  Numerics (lossless rule): with hmma every GEMM gets 16-bit activations (bf16 values staged exactly as
  power-of-two-scaled f16), exact ue8m0 weight scales, fp32 accumulation; the mHC Sinkhorn / sigmoid use IEEE
  divisions and libdevice's accurate expf exactly like stock's TileLang kernels (built without fast math).
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from ds41h import cuext, flags
from ds41h.skinny import SkinnyWeight, pack_checkpoint

logger = logging.getLogger("vllm.ds41h.prologue")

MAX_TOKENS = min(8, flags.MAX_DECODE_TOKENS)  # kernel limit 8 (B operand n8)

MMA_CODES = {"hmma": 0, "wgmma": 1}  # kernels/skinny.cu MMA_HMMA / MMA_WGMMA
FAST_EARLY, FAST_SINKHORN = 1, 2  # kernels/prologue.cu fast bits


def _mma_from_env() -> str:
    v = os.environ.get("DS41H_PROLOGUE_MMA", "hmma").strip().lower() or "hmma"
    if v not in MMA_CODES:
        logging.getLogger("vllm.ds41h.prologue").warning("DS41H_PROLOGUE_MMA=%r unknown; using 'hmma'", v)
        v = "hmma"
    return v


MMA = _mma_from_env()
FAST_LIGHT_SYNC = 32
FAST_V3, FAST_V3_HOP, FAST_V3_PACE = 512, 2048, 16384
# Default fast bits: the v3 schedule (prologue_kernel_v3: clusters of 8, weights -> smem by bulk copies paced to <= 2 x
# 10 KB in flight per CTA so the latency-critical loads are not queued behind the weight stream, P1a hop, DSMEM k
# reduction; G1 CTAs exit early so a PDL-launched attention can start). HMMA: bit-identical to fast=0 (stock
# rounding). Measured end-to-end-like (m=6, previous-layer kernel end -> attention end, cold code / warm code):
# hmma fast=0-style f33 46.4 / 39.9 us -> 18944 42.8 / 37.7 us (with DS41H_DECODE_NOPAD + DS41H_ATTN_EARLY: 37.5 / 33.6).
# Falls back to the regular kernel (with the same bits' EARLY | LIGHT_SYNC meaning ignored) if the 30-cluster grid
# does not fit the device.
FAST_V3_LATE_W2 = 1024
FAST_V3_FAC, FAST_V3_ILV, FAST_V3_PUSH, FAST_V3_SK_G1, FAST_V3_KV_SPREAD = 131072, 262144, 524288, 1048576, 2097152
# Measured (250 W, end-to-end-like, previous-layer kernel end -> attention end, 2 rounds, all bit-identical with hmma):
#   m=6 cold code (torch heater): 18944 38.5 us -> 4082176 36.2 -> 4083200 (+LATE_W2) 35.8 us;
#   m=6 warm code (L2 evict_first heater, closest to end-to-end): 33.3 -> 32.1 -> 31.3 us;
#   4082176 at m=1 / m=8 (warm): 30.3 -> 27.8 / 35.8 -> 34.6 us. Prologue kernel m=6 warm 22.2 -> 19.0 us.
# (1001984 = SK_G2 instead of SK_G1 | KV_SPREAD: -2.0 us cold but +0.9 us warm / ~0 in end-to-end -> replaced.)
FAST_DEFAULT = (FAST_V3 | FAST_V3_HOP | FAST_V3_PACE | FAST_V3_FAC | FAST_V3_ILV | FAST_V3_PUSH | FAST_V3_SK_G1 |
                FAST_V3_KV_SPREAD | FAST_V3_LATE_W2)


def _fast_from_env() -> int:
    """DS41H_PROLOGUE_FAST: unset -> FAST_DEFAULT; '0' -> 0 (Phase-1 schedule); '1' -> early + rcp Sinkhorn (legacy);
    any other integer -> that kernels/prologue.cu fast bitmask (bits the launcher does not support are dropped there)."""
    v = os.environ.get("DS41H_PROLOGUE_FAST", "").strip().lower()
    if v == "":
        return FAST_DEFAULT
    if v in ("1", "true", "yes", "on"):
        return FAST_EARLY | FAST_SINKHORN
    if v in ("false", "no", "off"):
        return 0
    try:
        return int(v, 0)
    except ValueError:
        logging.getLogger("vllm.ds41h.prologue").warning("DS41H_PROLOGUE_FAST=%r unknown; using %d", v, FAST_DEFAULT)
        return FAST_DEFAULT


FAST = _fast_from_env()

# DS41H_ATTN_FLAG=1: prologue -> sparse decode attention handoff through a counter instead of griddepcontrol.wait. The prologue
# bumps C_AQ (its scratch's counter block) once per G2 CTA after its q rows are stored and once per token after its KV insert;
# the flag-mode attention (ds41h/attn.py) polls C_AQ >= G2_CTAS + m, so it starts while the prologue's tail (Sinkhorn host CTAs,
# grid completion + flush) is still running, then waits for the prologue grid at its own end. Its last CTA resets the counter.
# Pairing: every flag-mode prologue registers a pending handoff (below) that the attention of the SAME attention module takes;
# anything else (stock attention fallback, another layer's attention, the next prologue) resets the counter with a regular
# launch first, so a stale count can never release an attention early. Bit-identical outputs (only the wait changes).
ATTN_FLAG = os.environ.get("DS41H_ATTN_FLAG", "0").strip().lower() not in ("0", "", "false", "no", "off")

# DS41H_PROLOGUE_INDEXER=1 (opt-in): on index-source layers the prologue also computes the indexer's two projections that the
# stock glue runs right after it -- idx_w = indexer.weights_proj(a) (bf16 [m, 32]) and idx_q = indexer.wq_b(qr) (bf16
# [m, 4096], no RoPE) -- inside the same launch (kernels/prologue.cu "indexer projections"); run_prologue returns them as
# out["idx_w"] / out["idx_q"] and the decoder glue hands them to the stock indexer (patch_prologue.py: DeepseekV4Indexer.
# _wq_b_proj returns a pending idx_q). Numerics: idx_q = the skinny drop-in's (same HMMA staging as wq_b); idx_w = bf16
# HMMA with exact products and fp32 accumulation (cuBLAS bf16 GEMM class), bf16 output.
INDEXER = os.environ.get("DS41H_PROLOGUE_INDEXER", "0").strip().lower() not in ("0", "", "false", "no", "off")
# INDEXER must be bit-identical in values to the reference path:
#  * idx_q now reproduces the skinny drop-in's k-partition (kernels/prologue.cu IDXQ_KPC): bit-identical to indexer.wq_b(qr);
#  * idx_w (weights_proj in the prologue: exact bf16 products, fp32 accumulation in another order than cuBLAS) only with
#    DS41H_PROLOGUE_INDEXER_W=1; by default the stock cuBLAS weights_proj runs (out has no "idx_w") -> DS41H_PROLOGUE_INDEXER=1 is
#    value-identical to the reference config (only the schedule changes).
INDEXER_W = os.environ.get("DS41H_PROLOGUE_INDEXER_W", "0").strip().lower() not in ("0", "", "false", "no", "off")
# DS41H_PROLOGUE_INDEXER_FOLD=1 (opt-in, needs INDEXER_W: weights_out folds the prologue's idx_w): the stock fused indexer q RoPE + FP8
# quant (CuTe IndexerQFp8Kernel, bit-reproduced: kernels/prologue.cu v3_idx_q) and the sparse indexer's top-k buffer clear run in the
# prologue; DeepseekV4Indexer.forward_q returns the prologue's (q_fp8, None, weights_out) and SparseAttnIndexer.forward_cuda skips its
# clear for that q (patch_prologue.py). -2 kernels (+ gaps) per index-source layer.
INDEXER_FOLD = INDEXER_W and os.environ.get("DS41H_PROLOGUE_INDEXER_FOLD", "0").strip().lower() not in ("0", "", "false", "no", "off")
IDX_N, IDXW_N = 4096, 32
# DS41H_PROLOGUE_AUX=1 (opt-in): run_prologue(aux=True) also returns out["aux"] = residual.mean(dim=1) (the capture_aux of the
# last layers) computed by the P1a jobs from the bf16 streams they write anyway (one fewer reduce kernel per such layer).
AUX = os.environ.get("DS41H_PROLOGUE_AUX", "0").strip().lower() not in ("0", "", "false", "no", "off")
# DS41H_P1A_EXT (opt-in, needs the decoder's XF_P1A = decoder.ffn(..., p1a=...)): the layer kernel of layer L computes layer
# L+1's P1a (post-mapped residual + collapse xcb / xsq into p1a_buffers) and layer L+1's prologue skips it (run_prologue
# p1a_done, FAST_V3_EXT_P1A). Glue: ds41h/patch_decoder.py (forward + _prologue_attention_side).
P1A_EXT = os.environ.get("DS41H_P1A_EXT", "0").strip().lower() not in ("0", "", "false", "no", "off")


def mma_code(mma: str | int | None) -> int:
    """Kernel MMA selector for 'hmma' / 'wgmma' / a raw code (None: the DS41H_PROLOGUE_MMA default)."""
    if mma is None:
        mma = MMA
    if isinstance(mma, str):
        return MMA_CODES[mma.lower()]
    if int(mma) not in MMA_CODES.values():
        raise ValueError(f"unknown mma code {mma}")
    return int(mma)
HC, HID, QR, KVW, HEADS, HDIM, ROPE = 4, 5120, 1280, 512, 16, 512, 64
C_AQ, G2_CTAS = 8, 128  # kernels/prologue.cu: handoff counter word (u32 index; C_AACK = C_AQ + 1) and the G2 CTA count
V4_RECORD_BYTES = 584  # stock sm90 SWA record (attention.py:512-513)

_lock = threading.Lock()
_scratch: dict[int, torch.Tensor] = {}
# Model-entry constants per device (first_layer=True), allocated next to the scratch (before any graph capture):
# residual 0 [8, 4, 5120] bf16, post 1 [8, 4, 1], comb 0 [8, 4, 4], pre e0 [8, 4] (f32). See entry_inputs.
_entry: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
ENTRY_ROWS = 8  # kernel token limit


# DS41H_PA_TRACE_DUMP=<prefix> (diagnostics, end-to-end in-situ timeline of the prologue + sparse decode attention): both kernels are
# built with their phase stamps (-DDS41H_PROLOGUE_PROF / -DDS41H_ATTN_PROF; %globaltimer + %clock64 + %smid per CTA), every launch
# writes into a per-layer slot of a device buffer (_trace / _atrace, allocated with the scratch, before graph capture), and a daemon
# thread waits for the file <prefix>.go, then saves <prefix>_r<rank>_<k>.npz (pro [layers, CTAS, 32], att [layers, 128, 32], and
# dec = the decoder's trace ring if the decoder runs a TRACE=1 build) 6 times, 1 s apart. Analysis: scratch pa5/pa_trace.py.
TRACE_DUMP = os.environ.get("DS41H_PA_TRACE_DUMP", "").strip()
TRACE_LAYERS = 48
ATRACE_CTAS = 128  # rows (<= 8) x cluster (<= 16) of a traced attention launch
_trace: dict[int, torch.Tensor] = {}
_atrace: dict[int, torch.Tensor] = {}


@functools.cache
def ext() -> Any:
    """The JIT-built extension (sm_90a)."""
    if TRACE_DUMP:
        return cuext.load("ds41h_prologue_prof", "prologue.cu", extra_cuda_cflags=("-DDS41H_PROLOGUE_PROF",))
    return cuext.load("ds41h_prologue", "prologue.cu")


def _tp_rank(idx: int) -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return idx


def _start_trace_dumper(idx: int, prefix: str) -> None:
    """See TRACE_DUMP. Diagnostics only: never raises into the server."""
    import threading
    import time

    import numpy as np

    def run():
        torch.cuda.set_device(idx)
        while not os.path.exists(prefix + ".go"):
            time.sleep(0.25)
        rank = _tp_rank(idx)
        stream = torch.cuda.Stream(idx)
        for k in range(6):
            time.sleep(1.0)
            try:
                with torch.cuda.stream(stream):
                    dec = None
                    try:
                        from ds41h import decoder as _dec

                        if getattr(_dec, "_ENGINE", None) is not None:
                            dec = _dec._ENGINE.trace_dump(False)  # synchronizes the device; the ring of the last 8 launches
                    except Exception:  # noqa: BLE001
                        dec = None
                    pro, att = _trace[idx].clone(), _atrace[idx].clone()
                stream.synchronize()
                np.savez(f"{prefix}_r{rank}_{k}.npz", pro=pro.cpu().numpy(), att=att.cpu().numpy(),
                         dec=dec.cpu().numpy() if dec is not None and dec.numel() else np.zeros(0, np.int32))
            except Exception as e:  # noqa: BLE001
                print(f"ds41h prologue/attn trace dump failed: {e}", flush=True)
                return

    threading.Thread(target=run, name="ds41h-pa-trace-dump", daemon=True).start()


def ensure_device(device: torch.device | int | str) -> torch.Tensor:
    """Per-device zeroed scratch (phase counters + partial sums), allocated once and never freed.

    Must first run outside CUDA graph capture (ds41h.skinny.capture calls it at weight-load time)."""
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _scratch.get(idx)
    if buf is None:
        with _lock:
            buf = _scratch.get(idx)
            if buf is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("ds41h prologue scratch must be allocated before CUDA graph capture")
                e = ext()
                ndev = torch.device("cuda", idx)
                with torch.cuda.device(ndev):
                    buf = torch.zeros(int(e.scratch_bytes()), dtype=torch.uint8, device=ndev)
                    # sets both instantiations' smem attributes now (never first inside a graph capture), and the
                    # cluster / v3 grids' residency checks
                    if hasattr(e, "init"):
                        e.init()
                    for code in MMA_CODES.values():
                        fits = int(e.max_resident_ctas(code))
                        if code == mma_code(None) and fits < int(e.CTAS):
                            raise RuntimeError(f"ds41h prologue needs {int(e.CTAS)} co-resident CTAs, device fits {fits}")
                    f32 = dict(dtype=torch.float32, device=ndev)
                    pre_e0 = torch.zeros(ENTRY_ROWS, HC, **f32)
                    pre_e0[:, 0] = 1.0
                    _entry[idx] = (
                        torch.zeros(ENTRY_ROWS, HC, HID, dtype=torch.bfloat16, device=ndev),
                        torch.ones(ENTRY_ROWS, HC, 1, **f32),
                        torch.zeros(ENTRY_ROWS, HC, HC, **f32),
                        pre_e0,
                    )
                    if TRACE_DUMP:
                        _trace[idx] = torch.zeros(TRACE_LAYERS, max(int(e.CTAS), int(e.CTAS_V3)), int(e.PROF_SLOTS), dtype=torch.int64, device=ndev)
                        _atrace[idx] = torch.zeros(TRACE_LAYERS, ATRACE_CTAS, 32, dtype=torch.int64, device=ndev)
                        _start_trace_dumper(idx, TRACE_DUMP)
                _scratch[idx] = buf
    return buf


def trace_slot(kind: str, device: torch.device | int | str, layer_id: int, rows: int = 0) -> torch.Tensor | None:
    """TRACE_DUMP: the stamp buffer slot of layer `layer_id` for the prologue ('pro') or the attention ('att'), else None."""
    if not TRACE_DUMP or not (0 <= int(layer_id) < TRACE_LAYERS):
        return None
    d = (_trace if kind == "pro" else _atrace).get(_dev_index(device))
    if d is None:
        return None
    return d[int(layer_id)]


def entry_inputs(device: torch.device | int | str, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(residual, post_mix, res_mix, pre_mix) that turn the shifted prologue into stock's model-entry mHC pre (first_layer):
    residual 0, post 1, comb 0, pre e0. The kernel then computes the post-mapped stream new_r[j] = 1 * x + sum_k 0 * 0 = x
    exactly (stock: the stream is the input broadcast to the 4 copies, model.py `residual is None` branch), collapses it
    with e0 to copy 0 = x exactly (stock mhc_pre_delayed_tilelang with pre_mix=None selects stream 0), and derives this
    sublayer's post / comb / pre from the fp32 stream as on every layer (stock: tf32 prenorm GEMM; same values, more
    precise)."""
    ensure_device(device)
    r, p, c, e = _entry[_dev_index(device)]
    return r[:m], p[:m], c[:m], e[:m]


FAST_V3_EXT_P1A = 268435456  # kernels/prologue.cu: P1a (residual output, collapse xcb / xsq) produced by the previous grid
SLICES = HID // 32


def p1a_buffers(device: torch.device | int | str) -> tuple[torch.Tensor, torch.Tensor]:
    """(xcb bf16 [8, 5120], xsq f32 [8, 160]) views of the device's prologue scratch that a producer of the P1a outputs
    (decoder layer kernel XF_P1A, decoder.py DecoderEngine.run(p1a=(resid_out, xcb, xsq))) fills for the NEXT layer's
    prologue, which then runs with run_prologue(..., p1a_done=resid_out) (FAST_V3_EXT_P1A: no P1a, no C_P1A hop)."""
    buf = ensure_device(device)
    e = ext()
    xcb = buf[int(e.OFF_XCB) : int(e.OFF_XCB) + ENTRY_ROWS * HID * 2].view(torch.bfloat16).view(ENTRY_ROWS, HID)
    xsq = buf[int(e.OFF_XSQ) : int(e.OFF_XSQ) + ENTRY_ROWS * SLICES * 4].view(torch.float32).view(ENTRY_ROWS, SLICES)
    return xcb, xsq


_p1a_pending: dict[int, tuple[Any, torch.Tensor]] = {}  # device -> (next layer, resid_out) whose P1a the decoder produced


def set_p1a_pending(device: torch.device | int | str, next_layer: Any, resid_out: torch.Tensor) -> None:
    """The layer kernel just launched with p1a=(resid_out, *p1a_buffers(device)) for `next_layer`'s prologue."""
    _p1a_pending[_dev_index(device)] = (next_layer, resid_out)


def take_p1a_pending(layer: Any, device: torch.device | int | str, m: int) -> torch.Tensor | None:
    """resid_out if the previous layer kernel produced this layer's P1a (else None; a stale entry is dropped)."""
    e = _p1a_pending.pop(_dev_index(device), None)
    if e is None or e[0] is not layer or e[1].shape[0] != m:
        return None
    return e[1]


def handoff_counter(device: torch.device | int | str) -> torch.Tensor:
    """int32 [2] view (C_AQ, C_AACK) of the device's prologue scratch: the flag-mode attention's wait counter."""
    return ensure_device(device)[C_AQ * 4 : C_AQ * 4 + 8].view(torch.int32)


def handoff_target(m: int) -> int:
    """C_AQ value at which the q rows (G2_CTAS signals) and the m KV inserts of a flag-mode prologue are stored."""
    return G2_CTAS + int(m)


@dataclass
class Handoff:
    owner: Any  # the attention module whose decode attention consumes it
    counter: torch.Tensor
    target: int


_handoff: dict[int, Handoff] = {}  # device index -> pending handoff of the last flag-mode prologue


def _dev_index(device: torch.device | int | str) -> int:
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def reset_handoff(device: torch.device | int | str) -> None:
    """Drop a pending handoff of `device` and reset its counter (regular launch: runs after the producing prologue)."""
    h = _handoff.pop(_dev_index(device), None)
    if h is not None:
        from ds41h import attn

        attn.ext().aflag_reset(h.counter)


def take_handoff(owner: Any, device: torch.device | int | str) -> Handoff | None:
    """The pending handoff if `owner` (an attention module) is the one its prologue registered, else None (a pending handoff
    of another module is reset first, so the caller can run the regular PDL-wait attention)."""
    idx = _dev_index(device)
    h = _handoff.get(idx)
    if h is None:
        return None
    if h.owner is not owner:
        reset_handoff(idx)
        return None
    del _handoff[idx]
    return h


@dataclass
class PrologueLayer:
    """Per-layer pointers the kernel needs (built once per layer, no copies of large tensors)."""

    layer_id: int
    fn: torch.Tensor  # hc_attn_fn [24, 20480] f32
    hc_scale: torch.Tensor  # [3] f32
    hc_base: torch.Tensor  # [24] f32
    attn_norm: torch.Tensor  # [5120] bf16
    wqa: SkinnyWeight  # fused wq_a|wkv [1792, 5120]
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wqb: SkinnyWeight  # wq_b TP slice [8192, 1280]
    cos_sin: torch.Tensor  # rotary_emb.cos_sin_cache [max_pos, 64] f32
    rms_eps: float
    hc_eps: float
    norm_eps: float
    qkv_eps: float
    post_mult: float
    iters: int
    iwq: SkinnyWeight | None = None  # indexer.wq_b [4096, 1280] (index-source layers)
    iwp: torch.Tensor | None = None  # indexer.weights_proj.weight [32, 5120] bf16
    icos: torch.Tensor | None = None  # indexer_rotary_emb.cos_sin_cache [max_pos, 64] f32 (INDEXER_FOLD)
    iwscale: float = 0.0  # indexer softmax_scale * n_head ** -0.5 (INDEXER_FOLD)
    itopk: torch.Tensor | None = None  # the shared top-k buffer [rows, 512] int32 (INDEXER_FOLD clear)


def _contig(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A persistent kernel argument: must already be contiguous/aligned when called under graph capture
    (a copy made during capture would only be filled when that graph replays)."""
    t = t.detach()
    if t.dtype != dtype:
        raise TypeError(f"expected {dtype}, got {t.dtype}")
    if torch.cuda.is_current_stream_capturing() and not (t.is_contiguous() and t.data_ptr() % 16 == 0):
        raise RuntimeError("ds41h prologue: layer tensors need a copy; prepare layers before CUDA graph capture")
    return _aligned(t)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16-byte aligned base (the kernel uses vector loads); copies only if needed."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _checkpoint_weights(layer: torch.nn.Module) -> tuple[SkinnyWeight, SkinnyWeight]:
    """Fallback: pack fused wq_a|wkv and this rank's wq_b slice straight from the checkpoint."""
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    from ds41h.ref.weights import CheckpointReader

    attn = layer.attn
    i = attn.layer_id
    num_layers = get_current_vllm_config().model_config.hf_text_config.num_hidden_layers
    pre = f"layers.{i}." if i < num_layers else f"mtp.{i - num_layers}."
    reader = CheckpointReader(get_current_vllm_config().model_config.model)
    dev = attn.fused_wqa_wkv.weight.device
    rank, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    wqa = pack_checkpoint(reader, [pre + "attn.wq_a.weight", pre + "attn.wkv.weight"], None, dev)
    wqb = pack_checkpoint(reader, [pre + "attn.wq_b.weight"], rank, dev, tp)
    return wqa, wqb


def prepare_layer(layer: torch.nn.Module) -> PrologueLayer | None:
    """Cached per-layer kernel arguments, or None if this layer cannot use the fused prologue."""
    cached = layer.__dict__.get("_ds41h_prologue", False)
    if cached is not False:
        return cached
    info = None
    try:
        info = _build(layer)
    except Exception:  # noqa: BLE001 - ineligible layers fall back to stock
        logger.exception("ds41h prologue: layer %s not prepared; stock path kept", getattr(layer.attn, "layer_id", "?"))
        info = None
    layer.__dict__["_ds41h_prologue"] = info
    return info


def _build(layer: torch.nn.Module) -> PrologueLayer | None:
    attn = layer.attn
    if getattr(layer, "engram", None) is not None:
        return None
    if (
        attn.n_local_heads != HEADS
        or attn.head_dim != HDIM
        or attn.rope_head_dim != ROPE
        or attn.q_lora_rank != QR
        or attn.hidden_size != HID
        or layer.hc_mult != HC
        or getattr(attn, "kv_mxfp8", True)
        or getattr(attn, "swa_bytes_per_token", 0) != V4_RECORD_BYTES
        or getattr(attn, "kv_cache_dtype", None) != "fp8_ds_mla"
        or getattr(attn, "accepts_unnormed_unroped_query", False)
    ):
        return None
    wqa = attn.fused_wqa_wkv.__dict__.get("_ds41h_skinny")
    wqb = attn.wq_b.__dict__.get("_ds41h_skinny")
    if wqa is None or wqb is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ds41h prologue: weights missing during CUDA graph capture")
        logger.warning("ds41h prologue: layer %d weights not captured at load; packing from the checkpoint", attn.layer_id)
        wqa, wqb = _checkpoint_weights(layer)
    if (wqa.n, wqa.k) != (QR + KVW, HID) or (wqb.n, wqb.k) != (HEADS * HDIM, QR):
        return None
    cos_sin = attn.rotary_emb.cos_sin_cache
    if cos_sin.dtype != torch.float32 or cos_sin.dim() != 2 or cos_sin.shape[1] != ROPE or not cos_sin.is_cuda:
        return None  # stock asserts the same (fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu)
    ensure_device(wqa.w.device)
    iwq = iwp = None
    indexer = getattr(attn, "indexer", None)
    if indexer is not None:
        cand = getattr(getattr(indexer, "wq_b", None), "__dict__", {}).get("_ds41h_skinny")
        wp = getattr(getattr(indexer, "weights_proj", None), "weight", None)
        if (cand is not None and (cand.n, cand.k) == (IDX_N, QR) and isinstance(wp, torch.Tensor) and wp.dtype == torch.bfloat16
                and tuple(wp.shape) == (IDXW_N, HID) and wp.is_cuda):
            iwq, iwp = cand, _contig(wp, torch.bfloat16)
    icos = itopk = None
    iwscale = 0.0
    if iwq is not None:
        ircs = getattr(getattr(attn, "indexer_rotary_emb", None), "cos_sin_cache", None)
        tb = getattr(indexer, "topk_indices_buffer", None)
        if isinstance(ircs, torch.Tensor) and ircs.dtype == torch.float32 and ircs.dim() == 2 and ircs.shape[1] == ROPE and ircs.is_cuda:
            icos = _contig(ircs, torch.float32)
            iwscale = float(indexer.softmax_scale * indexer.n_head ** -0.5)
        if (isinstance(tb, torch.Tensor) and tb.dtype == torch.int32 and tb.dim() == 2 and tb.stride(1) == 1 and tb.shape[1] % 4 == 0
                and tb.stride(0) % 4 == 0 and tb.data_ptr() % 16 == 0):
            itopk = tb
    return PrologueLayer(
        layer_id=int(attn.layer_id),
        fn=_contig(layer.hc_attn_fn, torch.float32),
        hc_scale=_contig(layer.hc_attn_scale, torch.float32),
        hc_base=_contig(layer.hc_attn_base, torch.float32),
        attn_norm=_contig(layer.attn_norm.weight, torch.bfloat16),
        wqa=wqa,
        q_norm=_contig(attn.q_norm.weight, torch.bfloat16),
        kv_norm=_contig(attn.kv_norm.weight, torch.bfloat16),
        wqb=wqb,
        cos_sin=_contig(cos_sin, torch.float32),
        rms_eps=float(layer.rms_norm_eps),
        hc_eps=float(layer.hc_eps),
        norm_eps=float(layer.attn_norm.variance_epsilon),
        qkv_eps=float(attn.eps),
        post_mult=float(layer.hc_post_alpha),
        iters=int(layer.hc_sinkhorn_iters),
        iwq=iwq,
        iwp=iwp,
        icos=icos,
        iwscale=iwscale,
        itopk=itopk,
    )


def prepare_all(model: torch.nn.Module) -> int:
    """Prepare every DeepSeek-V4.1 decoder layer of `model` (eagerly, before CUDA graph capture).
    Returns the number of layers that can use the fused prologue."""
    n = 0
    for module in model.modules():
        if hasattr(module, "attn") and hasattr(module, "hc_attn_fn") and hasattr(module.attn, "fused_wqa_wkv"):
            n += prepare_layer(module) is not None
    return n


def eligible(layer: torch.nn.Module, x_prev: Any, residual: Any, positions: torch.Tensor, *, first_layer: bool = False) -> bool:
    """Host-side check (static per captured graph) that run_prologue can replace the stock prologue
    of this forward: prologue flag, a plain-tensor previous output, a residual stream, no Engram, no
    sequence parallel / fused mHC all-reduce, 1..8 tokens. first_layer: model-entry mode (no residual; x_prev = the
    [M, 5120] bf16 layer input, see run_prologue)."""
    if first_layer:
        if residual is not None:
            return False
        if not flags.PROLOGUE or not isinstance(x_prev, torch.Tensor) or x_prev.dim() != 2:
            return False
        if not (0 < x_prev.shape[0] <= MAX_TOKENS) or x_prev.shape[1] != HID or x_prev.dtype != torch.bfloat16:
            return False
        if not x_prev.is_cuda or positions.device != x_prev.device:
            return False
        if positions.dtype != torch.int64 or positions.dim() != 1 or positions.shape[0] < x_prev.shape[0]:
            return False
        if getattr(layer, "engram", None) is not None or getattr(layer, "use_sequence_parallel", False):
            return False
        if getattr(layer, "fuse_mhc_all_reduce", False):
            return False
        return prepare_layer(layer) is not None
    if not flags.PROLOGUE or residual is None or not isinstance(x_prev, torch.Tensor):
        return False
    if getattr(layer, "engram", None) is not None or getattr(layer, "use_sequence_parallel", False):
        return False
    if getattr(layer, "fuse_mhc_all_reduce", False):
        return False
    m = x_prev.shape[0]
    if not (0 < m <= MAX_TOKENS) or x_prev.dim() != 2 or x_prev.shape[1] != HID or x_prev.dtype != torch.bfloat16:
        return False
    if not x_prev.is_cuda:
        return False
    # The kernel TORCH_CHECKs these; reject here so an odd step falls back to stock instead of raising.
    if not isinstance(residual, torch.Tensor) or residual.dtype != torch.bfloat16 or tuple(residual.shape) != (m, HC, HID):
        return False
    if residual.device != x_prev.device or positions.device != x_prev.device:
        return False
    if positions.dtype != torch.int64 or positions.dim() != 1 or positions.shape[0] < m:
        return False
    return prepare_layer(layer) is not None


def _swa_target(layer: torch.nn.Module, attn_metadata: Any):
    """(cache uint8 [blocks, page_bytes], slot_mapping int64, block_size) or None (profile run)."""
    if attn_metadata is None:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None  # stock skips the insert too (attention.py:1026-1042)
    swa_layer = layer.attn.swa_cache_layer
    md = attn_metadata.get(swa_layer.prefix)
    if md is None:
        # A real step without this layer's SWA metadata: stock asserts (attention.py:1047). Never
        # skip the insert silently -- attention would then read a stale window.
        raise RuntimeError(f"ds41h prologue: no SWA metadata for {swa_layer.prefix}")
    cache = swa_layer.kv_cache
    if cache.numel() == 0:
        return None  # KV cache not allocated yet (memory profiling)
    if cache.dtype != torch.uint8:
        raise RuntimeError(f"ds41h prologue: SWA cache dtype {cache.dtype}, expected the uint8 V4 record pages")
    cache2d = cache.view(cache.shape[0], -1)  # the view stock passes (attention.py:1052)
    return cache2d, md.slot_mapping, int(md.block_size)


def run_prologue(
    layer: torch.nn.Module,
    x_prev: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    pre_mix: torch.Tensor,
    positions: torch.Tensor,
    attn_metadata: Any = None,
    *,
    q_pre: torch.Tensor | None = None,
    mma: str | int | None = None,
    fast: int | None = None,
    indexer: bool | None = None,
    aux: bool = False,
    first_layer: bool = False,
    p1a_done: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Fused prologue of DeepseekV4DecoderLayer.forward for one decode step (call only if eligible()).

    Inputs (exactly what the stock forward receives for layers > 0 with residual not None):
      x_prev    [M, 5120] bf16  previous layer's MoE output (all-reduced)
      residual  [M, 4, 5120] bf16 residual stream
      post_mix  [M, 4, 1] f32, res_mix [M, 4, 4] f32  previous FFN post / comb coefficients
      pre_mix   [M, 4] f32      previous FFN pre-mix (collapses this attention input)
      positions [M] int64       attn_metadata: forward-context dict (default: current context)
    Returns a dict:
      residual  [M, 4, 5120] bf16  post-mapped stream (stock mhc_shifted_post_pre `residual`)
      post_mix  [M, 4, 1] f32, res_mix [M, 4, 4] f32, attn_pre [M, 4] f32  this attention's mHC coefficients
      a         [M, 5120] bf16     attn_norm(collapse): the attention input (stock `hidden_states`)
      qr        [M, 1280] bf16     q_norm(wq_a a)
      kv        [M, 512] bf16      kv_norm(wkv a), pre-RoPE (stock `kv`, unused by sm90 attention)
      q         [M, 16, 512] bf16  wq_b qr with RoPE (heads of this TP rank, contiguous)
      kv_inserted bool             whether this step's KV went into the SWA cache (False: profile run)
    mma / fast override DS41H_PROLOGUE_MMA / DS41H_PROLOGUE_FAST (tests, A/B benchmarks).
    With DS41H_ATTN_FLAG the launch also registers the handoff that layer.attn's decode attention takes (see ATTN_FLAG).
    With DS41H_PROLOGUE_INDEXER (or indexer=True) on an index-source layer, also idx_w [M, 32] / idx_q [M, 4096] (INDEXER).
    first_layer=True: model-entry mode = stock's `residual is None` branch with pre_mix None (e.g. the DSpark drafter's
    first layer): pass x_prev = the layer input [M, 5120] bf16 (the stream is x broadcast to the 4 copies) and None for
    residual / post_mix / res_mix / pre_mix (eligible(..., first_layer=True)). Outputs as above: residual = x broadcast,
    a = attn_norm(x) (stream 0), post_mix / res_mix / attn_pre = this attention's mHC coefficients (see entry_inputs).
    p1a_done = the post-mapped residual [M, 4, 5120] bf16 that the previous layer kernel already wrote together with xcb / xsq
    (p1a_buffers): the launch skips P1a (FAST_V3_EXT_P1A) and returns it as out["residual"]; no "aux" output then.
    """
    info = prepare_layer(layer)
    if info is None:
        raise RuntimeError("run_prologue called for an ineligible layer")
    m = x_prev.shape[0]
    dev = x_prev.device
    if first_layer:
        residual, post_mix, res_mix, pre_mix = entry_inputs(dev, m)
    scratch = ensure_device(dev)
    target = _swa_target(layer, attn_metadata)
    f32 = dict(dtype=torch.float32, device=dev)
    bf16 = dict(dtype=torch.bfloat16, device=dev)
    out = {
        "residual": p1a_done if p1a_done is not None else torch.empty_like(residual, memory_format=torch.contiguous_format),
        "post_mix": torch.empty(m, HC, 1, **f32),
        "res_mix": torch.empty(m, HC, HC, **f32),
        "attn_pre": torch.empty(m, HC, **f32),
        "a": torch.empty(m, HID, **bf16),
        "qr": torch.empty(m, QR, **bf16),
        "kv": torch.empty(m, KVW, **bf16),
        "q": torch.empty(m, HEADS, HDIM, **bf16),
    }
    cache, slots, block_size = target if target is not None else (None, None, 0)
    if _dev_index(dev) in _handoff:
        reset_handoff(dev)  # an unconsumed handoff (its attention never ran in flag mode): reset before this launch
    aflag = ATTN_FLAG
    idx = _idx_args(info, m, dev, mma, fast, INDEXER if indexer is None else indexer, INDEXER_W)
    fold = (None, None, None, 0.0, None)
    if idx[3] is not None:
        out["idx_q"] = idx[3]
        if idx[4] is not None:
            out["idx_w"] = idx[4]
        # the pending idx_q is consumed by the patched DeepseekV4Indexer._wq_b_proj only for THIS qr tensor (patch_prologue.py)
        layer.attn.indexer.__dict__["_ds41h_idx_q"] = (out["qr"], idx[3])
        if INDEXER_FOLD and idx[4] is not None and info.icos is not None:
            fold = _fold_args(info, m, dev)
            out["idx_qf"], out["idx_wf"] = fold[0], fold[1]
            ind = layer.attn.indexer
            ind.__dict__["_ds41h_fold"] = (out["qr"], fold[0], fold[1])  # patched forward_q: only for this qr
            if fold[4] is not None:
                getattr(ind, "indexer_op").__dict__["_ds41h_fold_q"] = fold[0]  # patched forward_cuda: clear done for this q
    aux_t = torch.empty(m, HID, **bf16) if (aux and AUX and p1a_done is None) else None
    if aux_t is not None:
        out["aux"] = aux_t
    fast_v = FAST if fast is None else int(fast)
    if p1a_done is not None:
        fast_v |= FAST_V3_EXT_P1A
    tslot = trace_slot("pro", dev, info.layer_id) if TRACE_DUMP else None
    if tslot is not None:
        ext().set_prof(tslot)
    ext().prologue(
        _aligned(x_prev),
        _aligned(residual),
        _aligned(post_mix),
        _aligned(res_mix),
        _aligned(pre_mix),
        info.fn,
        info.hc_scale,
        info.hc_base,
        info.attn_norm,
        info.wqa.w,
        info.wqa.s,
        info.q_norm,
        info.kv_norm,
        info.wqb.w,
        info.wqb.s,
        positions if positions.is_contiguous() else positions.contiguous(),
        info.cos_sin,
        cache,
        slots if slots is None or slots.is_contiguous() else slots.contiguous(),
        block_size,
        scratch,
        out["residual"],
        out["post_mix"],
        out["res_mix"],
        out["attn_pre"],
        out["a"],
        out["qr"],
        out["kv"],
        out["q"],
        q_pre,
        info.rms_eps,
        info.hc_eps,
        info.norm_eps,
        info.qkv_eps,
        info.post_mult,
        info.iters,
        mma_code(mma),
        fast_v,
        1 if aflag else 0,
        *idx,
        aux_t,
        *fold,
    )
    if tslot is not None:
        ext().set_prof(None)
    if aflag:
        _handoff[_dev_index(dev)] = Handoff(layer.attn, handoff_counter(dev), handoff_target(m))
    out["kv_inserted"] = target is not None
    return out


@functools.cache
def _idxq_partition_ok() -> bool:
    """The kernel's idx_q reproduces the skinny drop-in's k-partition for [4096, 1280] (4 k-parts of 12 blocks, kernels/prologue.cu
    IDXQ_KPC / IDXQ_KW): bit-identical to indexer.wq_b(qr) of the reference path only if the drop-in still picks that split."""
    from ds41h import skinny

    kpc = int(skinny.ext().choose_split(IDX_N, QR))
    if kpc != 4:
        logger.warning("ds41h prologue: skinny split for [%d, %d] is %d (kernel assumes 4): indexer projections disabled", IDX_N, QR, kpc)
    return kpc == 4


def _fold_args(info: PrologueLayer, m: int, dev: torch.device, clear: bool = True) -> tuple:
    """(idx_qf e4m3 [m, 32, 128], idx_wf f32 [m, 32], indexer cos_sin, wscale, top-k buffer or None) for INDEXER_FOLD."""
    qf = torch.empty(m, IDXW_N, IDX_N // IDXW_N, dtype=torch.float8_e4m3fn, device=dev)
    wf = torch.empty(m, IDXW_N, dtype=torch.float32, device=dev)
    return (qf, wf, info.icos, float(info.iwscale), info.itopk if clear else None)


def _idx_args(info: PrologueLayer, m: int, dev: torch.device, mma: str | int | None, fast: int | None, want: bool,
              with_w: bool = True) -> tuple:
    """(idx_wqb, idx_wqb_s, idx_wp, idx_q, idx_w) for the kernel: tensors when the indexer projections run in this launch
    (requested, the layer has them, hmma + the v3 schedule), else Nones; idx_wp / idx_w None unless with_w."""
    f = FAST if fast is None else int(fast)
    if not (want and info.iwq is not None and mma_code(mma) == MMA_CODES["hmma"] and f & FAST_V3 and _idxq_partition_ok()):
        return (None, None, None, None, None)
    bf16 = dict(dtype=torch.bfloat16, device=dev)
    if not with_w:
        return (info.iwq.w, info.iwq.s, None, torch.empty(m, IDX_N, **bf16), None)
    return (info.iwq.w, info.iwq.s, info.iwp, torch.empty(m, IDX_N, **bf16), torch.empty(m, IDXW_N, **bf16))


def run_raw(
    x_prev: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    pre_mix: torch.Tensor,
    positions: torch.Tensor,
    info: PrologueLayer,
    swa: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    q_pre: torch.Tensor | None = None,
    mma: str | int | None = None,
    fast: int | None = None,
    aflag: bool = False,
    indexer: bool = False,
    aux: bool = False,
    first_layer: bool = False,
    indexer_w: bool = True,
    fold: bool = False,
) -> dict[str, torch.Tensor]:
    """Kernel entry without vLLM objects (tests / benchmarks): same outputs as run_prologue. aflag: signal the handoff
    counter (handoff_counter / handoff_target); the caller must then run exactly one flag-mode attention (or
    attn.ext().aflag_reset) before the next aflag launch on this device. first_layer: as run_prologue (residual /
    post_mix / res_mix / pre_mix ignored)."""
    m, dev = x_prev.shape[0], x_prev.device
    if first_layer:
        residual, post_mix, res_mix, pre_mix = entry_inputs(dev, m)
    f32 = dict(dtype=torch.float32, device=dev)
    bf16 = dict(dtype=torch.bfloat16, device=dev)
    out = {
        "residual": torch.empty(m, HC, HID, **bf16),
        "post_mix": torch.empty(m, HC, 1, **f32),
        "res_mix": torch.empty(m, HC, HC, **f32),
        "attn_pre": torch.empty(m, HC, **f32),
        "a": torch.empty(m, HID, **bf16),
        "qr": torch.empty(m, QR, **bf16),
        "kv": torch.empty(m, KVW, **bf16),
        "q": torch.empty(m, HEADS, HDIM, **bf16),
    }
    cache, slots, block_size = swa if swa is not None else (None, None, 0)
    idx = _idx_args(info, m, dev, mma, fast, indexer, indexer_w)
    fa = (None, None, None, 0.0, None)
    if idx[3] is not None:
        out["idx_q"] = idx[3]
        if idx[4] is not None:
            out["idx_w"] = idx[4]
            if fold:
                fa = _fold_args(info, m, dev)
                out["idx_qf"], out["idx_wf"] = fa[0], fa[1]
    aux_t = torch.empty(m, HID, **bf16) if aux else None
    if aux_t is not None:
        out["aux"] = aux_t
    ext().prologue(
        _aligned(x_prev), _aligned(residual), _aligned(post_mix), _aligned(res_mix), _aligned(pre_mix),
        info.fn, info.hc_scale, info.hc_base, info.attn_norm, info.wqa.w, info.wqa.s, info.q_norm, info.kv_norm,
        info.wqb.w, info.wqb.s, positions.contiguous(), info.cos_sin, cache, slots, block_size, ensure_device(dev),
        out["residual"], out["post_mix"], out["res_mix"], out["attn_pre"], out["a"], out["qr"], out["kv"], out["q"],
        q_pre, info.rms_eps, info.hc_eps, info.norm_eps, info.qkv_eps, info.post_mult, info.iters,
        mma_code(mma), FAST if fast is None else int(fast), 1 if aflag else 0, *idx, aux_t, *fa,
    )  # fmt: skip
    return out
