"""Dense indexer logits (index layers 2 / 8 / 14 / 20) as a Triton kernel, for ONE process driving 4 GPUs.

Why: DeepGEMM's JIT runtime (deep_jit, vendored in vLLM) caches each kernel with a CUfunction obtained by
cuKernelGetFunction in the context of the device that first loaded it; launching it on another device of the same
process fails with CUDA_ERROR_INVALID_HANDLE (measured). vLLM runs one process per GPU, so production never hits
it. Triton compiles and loads per device.

Math = DeepGEMM's sm90_fp8_paged_mqa_logits (and vLLM's fp8_paged_mqa_logits_torch reference):
    logit[r, c] = kscale[c] * sum_h w[r, h] * relu(q[r, h, :] . k[c, :])        for c < n_r (fp32 accumulation)
q e4m3 [B, 4, 32, 128], k e4m3 [128] + fp32 scale per position in the paged index-K cache (per page: values, then scales;
page stride = the packed 116,928-B page), w = weights_out [4B, 32] fp32. Each program takes one pseudo-request's 4 rows
(DeepGEMM's next_n = 4: the rows share the block table) and a tile of BLOCK_KV positions: one e4m3 dot [BLOCK_KV, 128] x
[128, 4 x 32] -> fp32, relu, x w, per-row head sum, x kscale. Products are exact in fp32; only the summation order
differs from DeepGEMM's (wgmma k32 chunks, 4-way in-thread + 2 xor-shuffle head sums); agreement measured below.
Columns >= n_r are not written (as DeepGEMM with clean_logits=False). Graph-capturable (no host sync).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _eng_paged_mqa_logits(q_ptr, kv_ptr, ks_ptr, w_ptr, len_ptr, bt_ptr, out_ptr,
                          stride_qb, stride_bt, stride_out, kv_page_stride, ks_page_stride, ks_off, max_blocks,
                          NEXT_N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                          BLOCK_KV: tl.constexpr, N_SPLITS: tl.constexpr, NUM_STAGES: tl.constexpr):
    b = tl.program_id(0)
    split = tl.program_id(1)
    NH: tl.constexpr = NEXT_N * H
    j = tl.arange(0, NH)                      # row-major (row n, head h)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + j[:, None] * D + d[None, :])          # [NH, D] e4m3
    w = tl.load(w_ptr + b * NH + j)                                           # [NH] fp32 (rows b*4 .. b*4+3)
    nr = tl.arange(0, NEXT_N)
    lens = tl.load(len_ptr + b * NEXT_N + nr)                                 # [NEXT_N]
    lmax = tl.max(lens, 0)
    kv_col = tl.arange(0, BLOCK_KV)
    for kv_start in tl.range(split * BLOCK_KV, lmax, N_SPLITS * BLOCK_KV, num_stages=NUM_STAGES):
        pos = kv_start + kv_col
        blk = pos // BLOCK_SIZE                                               # a tile may span several pages
        ok_blk = blk < max_blocks
        page = tl.load(bt_ptr + b * stride_bt + blk, mask=ok_blk, other=0).to(tl.int64)
        pib = pos - blk * BLOCK_SIZE
        m = (pos < lmax) & ok_blk
        k = tl.load(kv_ptr + page[:, None] * kv_page_stride + pib[:, None] * D + d[None, :], mask=m[:, None], other=0.0)
        sc = tl.load(ks_ptr + page * ks_page_stride + ks_off + pib, mask=m, other=0.0)          # [BLOCK_KV] fp32
        s = tl.dot(k, tl.trans(q))                                            # [BLOCK_KV, NH] fp32
        s = tl.maximum(s, 0.0) * w[None, :]
        s = tl.sum(tl.reshape(s, (BLOCK_KV, NEXT_N, H)), axis=2)              # [BLOCK_KV, NEXT_N]
        s = s * sc[:, None]
        rows = b * NEXT_N + nr
        om = (pos[:, None] < lens[None, :]) & ok_blk[:, None]
        tl.store(out_ptr + rows[None, :].to(tl.int64) * stride_out + pos[:, None], s, mask=om)


def paged_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, seq_lens: torch.Tensor,
                     block_table: torch.Tensor, block_size: int, width: int, out: torch.Tensor | None = None,
                     n_splits: int | None = None, block_kv: int | None = None, num_stages: int | None = None,
                     num_warps: int = 8) -> torch.Tensor:
    """q e4m3 [B, 4, 32, 128]; kv uint8 [pages, block_size, 1, 132] (strided pages OK: inner dims contiguous);
    weights fp32 [4B, 32]; seq_lens int32 [B, 4]; block_table int32 [B, P] -> logits fp32 [4B, width]."""
    B, NN, H, D = q.shape
    assert q.is_contiguous() and weights.is_contiguous() and seq_lens.is_contiguous()
    assert kv.dtype == torch.uint8 and kv.stride(1) == D + 4 and kv.stride(-1) == 1
    pages = kv.shape[0]
    flat = torch.as_strided(kv, (pages, block_size * (D + 4)), (kv.stride(0), 1))
    kv8 = flat.view(torch.float8_e4m3fn)
    assert kv.stride(0) % 4 == 0 and kv.storage_offset() % 4 == 0
    ks32 = flat.view(torch.float32)
    if out is None:
        out = torch.empty(B * NN, width, dtype=torch.float32, device=q.device)
    # default: 128-position tiles spanning pages, 3 stages -> measured top-512 sets == production on 100 % of
    # rows of all 8 layers, 1.38-1.40 ms per rank for the 8 index layers (32 / 64-position tiles: 99.9 %, 1.86 ms)
    BLOCK_KV = block_kv or int(os.environ.get("ENG_IDXL_BKV", "128"))
    num_stages = num_stages or int(os.environ.get("ENG_IDXL_ST", "3"))
    if n_splits is None:
        tiles = max(1, triton.cdiv(width, BLOCK_KV))
        n_splits = max(1, min(tiles, max(1, 2 * 132 // max(1, B)) * 4))
    _eng_paged_mqa_logits[(B, n_splits)](
        q, kv8, ks32, weights, seq_lens, block_table, out,
        q.stride(0), block_table.stride(0), out.stride(0), kv8.stride(0), ks32.stride(0), (block_size * D) // 4,
        block_table.shape[1], NEXT_N=NN, H=H, D=D, BLOCK_SIZE=block_size, BLOCK_KV=BLOCK_KV, N_SPLITS=n_splits,
        NUM_STAGES=num_stages, num_warps=num_warps)
    return out
