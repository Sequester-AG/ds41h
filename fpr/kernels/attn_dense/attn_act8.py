#!/usr/bin/env python3
"""ds41h-phys: 8-bit activations for the engine's PREFILL rows' attention GEMMs (opt-in, LOSSY).

Production's own 8-bit prefill arm (ds41h/linear.py, DS41H_PREFILL_LINEAR): e4m3 activations with 1x128
ue8m0 scales (act_quant_128, ds41h/kernels/fp8_linear.cu), the checkpoint's 32x32 ue8m0 weight scales folded into one
fp32 scale per 128x128 block (fold_block_scales: 1e-5..6e-5 of the weight elements re-rounded), DeepGEMM sm90
fp8_gemm_nt 1D2D. This module calls those functions (no copies of their code), for:
  q_a | wkv (fused_wqa_wkv), q_b, wo_b  of layers 0..19, and q_a | wkv of layer 20
(production's default capture set "full": the layers before the decoder-replay cut + the cut layer's fused_wqa_wkv),
for calls of >= MIN_M = 256 prefill rows (production's DS41H_PREFILL_LINEAR_MIN_M). Decode rows stay 16-bit (the
default arm). wo_a in FP8 (production's DS41H_PREFILL_WOA + ROPEQ) is a sub-option, ENG_PF_ACT8_WOA=1:
ropeq64.cu (production's CUDA inverse-RoPE + 1x128 quant, launch reshaped for 64 heads; bytes and scales == production's
ROPEQ per TP4 rank, bitwise) + DeepGEMM fp8_einsum; it saves little (372 vs 403 us at 4,096 rows) but matches
production's 8-bit arm.

Switches (module level, read at import; the engine reads ENABLED and may flip it per arm with set_enabled()):
  ENG_PF_ACT8_DENSE=1   on (default 0: nothing changes)
Engine API:
  raw = cached(f"act8_L{L}_raw_v2", lambda: build_raw(L), device=dev)   for L in 0..20 (CPU build first: checkpoint
                                           e4m3 + ue8m0 scales of gemms_of(L); v1 entries = without wo_a, also valid)
  P = prepare(raw)                         per layer on the device: folded e4m3 weights in DeepGEMM's layout (93 MB for
                                           layers 0..19, 9 MB for layer 20: 1.87 GB per GPU)
  applies(L, which, n_rows) -> bool        which in ("qkv", "q_b", "wo_b")
  qkv(P, a, out) / q_b(P, qr, out) / wo_b(P, z, out)   bf16 in, bf16 out (row slices of the step's buffers);
          q_b leaves RoPE(q) + the window insert to dense.cu rope_q_insert (as attn_half's bf16 branch).
  wo_a(P, o, pos, cos_sin, z)              (ENG_PF_ACT8_WOA=1, prepare(raw, woa=True)) o = the attention output BEFORE the
                                           inverse RoPE (fused here), z [n, 8, 1024]

Measured (pf8_check.py, layer 10, 1 GPU, us per GEMM, 8-bit vs the 16-bit cuBLAS bf16 path):
  4,096 rows: q_a|wkv 95 vs 121, q_b 307 vs 489 (W8A16 743), wo_b 292 vs 463  -> -379 us per layer, -7.6 ms per step
  2,048 rows: 59 vs 64, 163 vs 260, 184 vs 233 (-151 us); 1,024: 37 vs 34, 91 vs 134, 80 vs 122; 256: about even.
  Exactness: vs the 8-bit fp32 reference 93 % bit-exact, row max 1 ulp; vs exact 16-bit rel. RMSE 2.67 % (the A8 cost);
  deterministic; each row's result independent of the call's row count (first 256 rows == the 256-row call); q_b
  columns == a GEMM on production's TP4 column shard, bitwise (so per token == production's 8-bit q_b / q_a|wkv; wo_b
  is K-sharded in TP4 and differs by construction, as in 16-bit).
  wo_a (pf8_woa.py): 372 vs 403 us at 4,096 rows (ropeq64 150 + einsum 230; vLLM's Triton quant alone 378), 180 vs 202 at
  2,048; vs its fp32 reference 93 % bit-exact, row max 1 ulp; rel. RMSE 2.67 %.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("ENG_PF_ACT8_DENSE", "0") == "1"
MIN_M = 256
NAMES = {"qkv": ("wq_a", "wkv"), "q_b": ("wq_b",), "wo_a": ("wo_a",), "wo_b": ("wo_b",)}
WOA8 = os.environ.get("ENG_PF_ACT8_WOA", "0") == "1"   # wo_a in FP8 too (ropeq64.cu inverse RoPE + quant + fp8_einsum)
_EXT = {}


def set_enabled(on: bool) -> None:
    global ENABLED
    ENABLED = bool(on)


def deploy_linear():
    """Production's ds41h/linear.py (as engine.py reaches ds41h.moe: its package dir appended to ds41h.__path__)."""
    import ds41h

    dt = os.environ.get("DS41H_DEPLOY", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))) + "/ds41h"
    if dt not in list(ds41h.__path__):
        ds41h.__path__.append(dt)
    from ds41h import linear as DL

    return DL


def gemms_of(layer: int) -> tuple:
    """The captured GEMMs of a layer (production's "full" set): all three for 0..19, q_a | wkv for 20, none after."""
    return ("qkv", "q_b", "wo_a", "wo_b") if layer < 20 else (("qkv",) if layer == 20 else ())


def build_raw(layer: int, model: str | None = None) -> dict:
    """Checkpoint e4m3 weights + ue8m0 scales (uint8, CPU) of one layer's captured GEMMs."""
    import bench2b as B

    rd = B.CheckpointReader(model or B.MODEL)
    out = {}
    for g in gemms_of(layer):
        ns = NAMES[g]
        w = torch.cat([rd.get(f"layers.{layer}.attn.{n}.weight") for n in ns], 0).contiguous()
        s = torch.cat([rd.get(f"layers.{layer}.attn.{n}.scale") for n in ns], 0).contiguous()
        out[g] = (w.view(torch.uint8).reshape(-1, w.shape[-1]).contiguous(),
                  s.view(torch.uint8).reshape(-1, s.shape[-1]).contiguous())
    return out


def prepare(raw: dict, woa: bool | None = None) -> dict:
    """Folded e4m3 weights on raw's device: DeepGEMM 1D2D layout (production's fold_weight) for q_a|wkv / q_b / wo_b, the
    grouped einsum layout (production's convert_woa fold + deepgemm_post_process_fp8_weight_block(is_bmm)) for wo_a (only
    with woa / WOA8), plus the fold statistics."""
    DL = deploy_linear()
    woa = WOA8 if woa is None else woa
    P, stats = {}, {}
    for g, (w, s) in raw.items():
        if g == "wo_a":
            if not woa:
                continue
            from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block

            wq, s128, stats[g] = DL.fold_block_scales(w, s)
            P[g] = deepgemm_post_process_fp8_weight_block(wq=wq, ws=s128, quant_block_shape=(128, 128), use_e8m0=False,
                                                          is_bmm=True, bmm_batch_size=8)
            P["_woa_ref"] = (wq, s128)
            continue
        P[g], stats[g] = DL.fold_weight(w, s)
    P["_stats"] = stats
    return P


def ropeq():
    """ropeq64.cu: production's inverse RoPE + 1x128 FP8 quant for G x 8 heads (built with production's flags)."""
    if "rq" not in _EXT:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        bdir = os.path.join(here, "build_rq")
        os.makedirs(bdir, exist_ok=True)
        _EXT["rq"] = load(name="ropeq64_ext", sources=[os.path.join(here, "ropeq64.cu")],
                          extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a",
                                             "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-O3"],
                          build_directory=bdir, verbose=False)
    return _EXT["rq"]


def wo_a(P: dict, o: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """z [n, 8, 1024] bf16 = FP8 wo_a of the attention output o [n, 64, 512] bf16 (RoPE dims still rotated: the inverse
    RoPE is fused into the quant). Production's DS41H_PREFILL_WOA + ROPEQ path per rank."""
    from vllm.utils.deep_gemm import fp8_einsum

    oq, osc = ropeq().inv_rope_fp8_quant(o, pos, cos_sin, 8)
    fp8_einsum("bhr,hdr->bhd", (oq, osc), P["wo_a"], z, recipe=(1, 128, 128))
    return z


def applies(layer: int, which: str, n_rows: int) -> bool:
    if not ENABLED or n_rows < MIN_M:
        return False
    return layer < 20 or (layer == 20 and which == "qkv")


def _gemm(x: torch.Tensor, fw, out: torch.Tensor) -> torch.Tensor:
    return deploy_linear().gemm_w8a8(x, fw, out=out)


def qkv(P: dict, a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return _gemm(a, P["qkv"], out)


def q_b(P: dict, qr: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return _gemm(qr, P["q_b"], out)


def wo_b(P: dict, z: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return _gemm(z, P["wo_b"], out)
