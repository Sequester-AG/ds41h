"""ds41h-phys: the DeepSeek-V4.1 indexer (+ compressor) on OWNED sequences.

Every GPU runs the indexer only for the decode rows of the sequences it owns, against its own compact index-K pages
(no KV replication, no exchange: a sequence's index keys, compressed KV and compressor ring live on its owner only).

Per index layer L in (2, 8, 14, 20, 24, 28, 32, 36), per owned request b (next_n = 4 rows p..p+3):
  dense layers 2 / 8 / 14 (ratio 2, n = (p + 1) // 2) and 20 (ratio 1, n = p + 1):
      logits = DeepGEMM sm90_fp8_paged_mqa_logits on the owned rows (the production kernel, bit-identical per row: a
               logit depends only on its q row, weights row and K entry), width = the longest owned row
      L20 only: candidates = ds41h select_blocks (production's semantics: block key-max over [0, n), newest block pinned,
               top-2048 blocks, lowest block id on ties) -- production runs stock's Triton + torch.topk because its
               655,360-wide buffer is past the fused kernel's 32,768-block envelope; owned rows are not
      top-512 = the stock vLLM top-k ("auto" -> persistent / sampled) with production's host max_seq_len
  consumer layers 24 / 28 / 32 / 36 (ratio 1, layer 20's index-K pages and candidates):
      prep (union of the 4 rows' candidate blocks, per-row compaction), candidate-only logits (DeepGEMM's arithmetic on
      gathered 8-token blocks: kernels/indexer2f/idx2f_cand.cu = the IDX_CAND kernel), the stock top-k on compact rows
      (<= 16,384), remap to positions. Selected sets are those of dense + mask + stock top-k (proved; measured here).

Public API (step program): see `OwnedIndexer` at the bottom.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BUILD = Path(os.environ.get("IDX2F_BUILD", str(HERE / "build")))
LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
KV_SOURCES = (2, 8, 14, 20)
CONSUMERS = (24, 28, 32, 36)
RATIO = {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}
K_SOURCE = {2: 2, 8: 8, 14: 14, 20: 20, 24: 20, 28: 20, 32: 20, 36: 20}
TOPK = 512
LONG_ROW = 262_144   # compressed positions: rows above it take the stock select / top-k (block keys > 32,768)
CAND_K = 2048
CAND_BLOCK = 8
HEADS, HEAD_DIM = 32, 128
NEXT_N = 4

# DeepGEMM JIT flags of sm90_fp8_paged_mqa_logits (no --use_fast_math: the epilogue must not flush denormals)
DG_FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,186,940"]
# ds41h flags (integer key math only in the select kernels)
DS_FLAGS = ["-O3", "-std=c++20", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
            "-lineinfo", "-Xptxas=-O3"]


def deep_gemm_include() -> Path:
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    inc = Path(spec.origin).parent / "third_party" / "deep_gemm" / "include"
    assert (inc / "deep_gemm" / "impls" / "sm90_fp8_paged_mqa_logits.cuh").exists(), inc
    return inc


def _load(name: str, src: str, flags: list[str], inc: list[str]):
    from torch.utils.cpp_extension import load

    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(HERE / src)], extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=flags,
                extra_include_paths=inc, build_directory=str(d), verbose=bool(os.environ.get("IDX2F_VERBOSE")))


@functools.cache
def ext_cand():
    return _load("idx2f_cand", "idx2f_cand.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_sel():
    return _load("idx2f_sel", "idx2f_sel.cu", DS_FLAGS, [str(HERE)])


@functools.cache
def ext_topk():
    return _load("idx2f_topk", "idx2f_topk.cu", DS_FLAGS, [str(HERE)])


@functools.cache
def ext_fcand():
    """gen_fused.py: logits_dense / logits with the block maxima fused into the epilogue (OwnedIndexer.fuse)."""
    return _load("idx2f_fcand", "idx2f_fcand.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_fcand2():
    """gen_fused.py: fcand with a redux block max and the weights in registers (experiment, topk2f.py)."""
    return _load("idx2f_fcand2", "idx2f_fcand2.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_dl():
    """gen_fused.py: one host call per eager dense layer (the original logits / block max / select / top-k kernels, the
    slices issued from C++; OwnedIndexer.dl)."""
    return _load("idx2f_dl", "idx2f_dl.cu", DG_FLAGS, [str(deep_gemm_include()), str(HERE)])


@functools.cache
def ext_fcand3():
    """gen_fused.py: logits with 4 rows per wgmma (N = 128), 3 math warpgroups (prefill experiment)."""
    return _load("idx2f_fcand3", "idx2f_fcand3.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_fcand4():
    """gen_fused.py: fcand3 + the weights in registers (prefill experiment)."""
    return _load("idx2f_fcand4", "idx2f_fcand4.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_fcand5():
    """gen_fused.py: fcand3 software-pipelined (2 math warpgroups, MMA of the next tile under the epilogue)."""
    return _load("idx2f_fcand5", "idx2f_fcand5.cu", DG_FLAGS, [str(deep_gemm_include())])


PF_EXT = os.environ.get("IDX2F_PF_EXT", "idx2f_pf4")   # the shared-key prefill logits kernel


@functools.cache
def ext_pf():
    """Prefill dense logits with key tiles shared by groups of <= 4 pseudo-requests of one sequence (bit-identical)."""
    return _load(PF_EXT, PF_EXT + ".cu", DG_FLAGS, [str(deep_gemm_include())])


def pf_groups(seq_pr: list[tuple[int, int]]) -> list[int]:
    """Group starts for the shared-key kernel from each sequence's pseudo-request range [p0, p1): chunks of <= 4."""
    gs = []
    for p0, p1 in seq_pr:
        gs.extend(range(p0, p1, 4))
    if seq_pr:
        gs.append(seq_pr[-1][1])
    return gs


@functools.cache
def ext_fcand8():
    """gen_fused.py fcand8: 4 rows per m64n128k32, per-warp stage arrives, producer block-table cache, quad-row stores
    (decode / single-request logits, bit-identical; IDX2F_LOGITS=f8)."""
    return _load("idx2f_fcand8", "idx2f_fcand8.cu", DG_FLAGS, [str(deep_gemm_include())])


@functools.cache
def ext_ftopk():
    """gen_fused.py: exact top-512 that also writes FlashMLA's global slots + lengths (OwnedIndexer.fuse)."""
    return _load("idx2f_ftopk", "idx2f_ftopk.cu", DS_FLAGS, [str(HERE)])


@functools.cache
def ext_q2():
    """idx2f_q2.cu: wq_b GEMM (4-CTA cluster K-split, every weight byte read once) + the stock RoPE / quant epilogue."""
    return _load("idx2f_q2", "idx2f_q2.cu", DG_FLAGS, [str(HERE)])


Q_WSCALE = float(128 ** -0.5 * 32 ** -0.5)   # softmax_scale * n_head^-0.5 (as idx2f_q.cu / the stock op)


def qpath2(qr: torch.Tensor, wq: torch.Tensor, w: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor,
           q8: torch.Tensor | None = None, wo: torch.Tensor | None = None, qin: torch.Tensor | None = None,
           qdbg: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The indexer q path of one layer in one kernel: = fused_indexer_q_rope_quant(pos, F.linear(qr, wq), cos_sin, w,
    128^-0.5, 32^-0.5) up to the GEMM's fp32 summation order (the epilogue is the stock op's, bit for bit).
    qr [R, 1280] bf16, wq [4096, 1280] bf16, w = weights_proj(x) [R, 32] bf16, pos [R] int64 -> q8 [R, 32, 128] e4m3,
    wo [R, 32] fp32."""
    R = qr.shape[0]
    if q8 is None:
        q8 = torch.empty((R, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=qr.device)
    if wo is None:
        wo = torch.empty((R, HEADS), dtype=torch.float32, device=qr.device)
    ext_q2().qpath2(qr, wq, w, pos, cos_sin, Q_WSCALE, q8, wo, qin, qdbg)
    return q8, wo


@functools.cache
def ext_q():
    return _load("idx2f_q", "idx2f_q.cu", DG_FLAGS, [str(HERE)])


@functools.cache
def ext_comp():
    return _load("idx2f_comp", "idx2f_comp.cu", DS_FLAGS, [str(HERE)])


# logits kernel family for dense + candidate logits (bit-identical variants): "cand" = idx2f_cand.cu
# (the default), "n128" = 4 rows per wgmma (idx2f_fcand3.cu), "n128h" = n128 + the weights in registers (idx2f_fcand4.cu),
# "f8" = idx2f_fcand8.cu (n128 + per-warp arrives + producer block-table cache + quad-row stores: -16 % decode)
LOGITS = os.environ.get("IDX2F_LOGITS", "cand")


def ext_logits():
    return {"cand": ext_cand, "n128": ext_fcand3, "n128h": ext_fcand4, "pipe": ext_fcand5, "f8": ext_fcand8}[LOGITS]()


# ------------------------------------------------------------------------------------------------------- ownership
def assign_requests(ctx: list[int], ranks: int = 4) -> list[list[int]]:
    """Greedy sequence ownership: longest context first, to the rank with the smallest context sum."""
    order = sorted(range(len(ctx)), key=lambda b: -ctx[b])
    load = [0] * ranks
    out: list[list[int]] = [[] for _ in range(ranks)]
    for b in order:
        k = min(range(ranks), key=lambda i: load[i])
        out[k].append(b)
        load[k] += ctx[b]
    return [sorted(o) for o in out]


@dataclass
class CompactK:
    """A rank's own index-K pages for one kv-source layer: kv [n_pages, bs, 1, 132] uint8 contiguous, and the owned
    requests' block tables remapped to it (page 0 = a zero page standing in for the null block)."""
    kv: torch.Tensor
    block_table: torch.Tensor  # [B_r, P] int32
    block_size: int


def compact_pages(pages: torch.Tensor, ids: torch.Tensor, block_table: torch.Tensor, reqs: list[int], bs: int,
                  dev) -> CompactK:
    """pages [n, bs*132] uint8 (raw captured page bytes for block ids `ids`), block_table [B, P] (captured, block ids)."""
    bt = block_table[reqs].to(torch.int64)
    used = torch.unique(bt[bt > 0])
    pos = torch.searchsorted(ids, used)
    assert torch.equal(ids[pos], used), "page missing from the capture"
    kv = torch.zeros((used.numel() + 1, bs * (HEAD_DIM + 4)), dtype=torch.uint8)
    kv[1:] = pages[pos]
    remap = torch.zeros(int(ids.max()) + 2, dtype=torch.int64)
    remap[used] = torch.arange(1, used.numel() + 1)
    nbt = torch.where(bt > 0, remap[bt.clamp(min=0)], torch.zeros_like(bt)).to(torch.int32)
    return CompactK(kv.view(-1, bs, 1, HEAD_DIM + 4).to(dev), nbt.contiguous().to(dev), bs)


# ------------------------------------------------------------------------------------------------------ kernels
def dg_logits(q: torch.Tensor, ck: CompactK, weights: torch.Tensor, seq_lens: torch.Tensor, width: int,
              sched: torch.Tensor | None = None) -> torch.Tensor:
    """DeepGEMM paged MQA logits on owned rows: q [B, 4, 32, 128] fp8, weights [B*4, 32] fp32, seq_lens [B, 4] int32."""
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata

    if sched is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        sched = get_paged_mqa_logits_metadata(seq_lens, ck.block_size, sms)
    return fp8_fp4_paged_mqa_logits((q, None), ck.kv, weights, seq_lens, ck.block_table, sched, max_model_len=width,
                                    clean_logits=False)


def dense_logits(q: torch.Tensor, ck: CompactK, weights: torch.Tensor, seq_lens: torch.Tensor, width: int,
                 out: torch.Tensor | None = None, bm: torch.Tensor | None = None) -> torch.Tensor:
    """Dense paged MQA logits with DeepGEMM's arithmetic in our own extension (idx2f_cand.cu logits_dense): bit-identical
    to dg_logits per value, and it runs on every device of one process (DeepGEMM's JIT caches its CUfunction in the
    first device's context). Pages of 64 (ratio 1) or 32 (ratio 2) compressed positions; no schedule metadata."""
    rows = q.shape[0] * q.shape[1]
    if out is None:                                    # out: [rows, >= width] fp32, rows 16-B aligned (a workspace view)
        out = torch.empty((rows, max(int(width), 256)), dtype=torch.float32, device=q.device)
    if bm is not None:   # fused block maxima (= block_max(out, lens, bm, max_len), bit for bit)
        ext_fcand().logits_dense(q, ck.kv, weights, seq_lens.reshape(-1), ck.block_table, out, bm)
    else:
        ext_logits().logits_dense(q, ck.kv, weights, seq_lens.reshape(-1), ck.block_table, out)
    return out


def dg_schedule(seq_lens: torch.Tensor, block_size: int) -> torch.Tensor:
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    sms = torch.cuda.get_device_properties(seq_lens.device).multi_processor_count
    return get_paged_mqa_logits_metadata(seq_lens, block_size, sms)


_WS: set = set()


def _workspace(dev) -> None:
    if dev in _WS:
        return
    from vllm.v1.worker import workspace as ws

    if getattr(ws, "_manager", None) is None:
        ws.init_workspace_manager(dev)
    _WS.add(dev)


def production_backend(global_rows: int) -> str:
    """The stock top-k kernel production runs for this step: "auto" = cooperative_topk for <= 64 rows of the WHOLE batch,
    persistent_topk above. A rank owns fewer rows than the batch, so 'auto' on its rows would switch to the cooperative
    kernel (inexact on threshold-bin overflow in vLLM 0.32.0, #59645) at loads where production runs persistent."""
    return "persistent" if global_rows > 64 else "auto"


def stock_topk(logits: torch.Tensor, seq_lens: torch.Tensor, out: torch.Tensor, max_seq_len: int,
               backend: str = "persistent") -> None:
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk

    _workspace(logits.device)
    get_indexer_topk(backend)(logits, seq_lens, NEXT_N, out, TOPK, int(max_seq_len))


def canon_topk(out: torch.Tensor) -> None:
    """In place: each row's top-k list in ascending position order, -1 padding last (the fast path's order). The stock
    persistent top-k writes its set in a run-to-run varying order, and FlashMLA's sums follow the list
    order: canonical lists make long-row steps deterministic and eager == graph bit for bit."""
    big = torch.iinfo(torch.int32).max
    key = torch.where(out >= 0, out, torch.full_like(out, big))
    s = key.sort(dim=1).values
    out.copy_(torch.where(s == big, torch.full_like(s, -1), s))


def select_candidates(logits: torch.Tensor, seq_lens: torch.Tensor, out: torch.Tensor) -> None:
    """Layer 20: production semantics (ds41h select_blocks = stock select_candidate_blocks' set), ascending ids."""
    ext_sel().select_blocks(logits, None, seq_lens.reshape(-1), CAND_K, CAND_BLOCK, out, 1)


def select_candidates_stock(logits: torch.Tensor, seq_lens: torch.Tensor, out: torch.Tensor) -> None:
    """Layer 20 for rows of any length: vLLM's own select_candidate_blocks (Triton block scores + torch.topk, what
    production runs at width 655,360). Output in descending score order (consumers then need the order-insensitive
    cand_prep)."""
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks as stock

    stock = getattr(stock, "_ds41h_stock", stock)
    stock(logits, None, seq_lens.reshape(-1), CAND_K, CAND_BLOCK, out, 1)


@dataclass
class CandPrep:
    ublk: torch.Tensor
    uslot: torch.Tensor
    nunion: torch.Tensor
    cblk: torch.Tensor
    clen: torch.Tensor
    upk: torch.Tensor | None = None  # packed (page << 3) | (blk & 7) per union block (prep2 with a block table)


MAX_CALL_ROWS = 1024   # block_max: rows per call (device-side prefix in shared memory)
MAX_CALL_B = 1024      # logits_dense / candidate logits: pseudo-requests per call (task prefix in shared memory)


def block_max(logits: torch.Tensor, lens: torch.Tensor, out: torch.Tensor, max_len: int) -> None:
    rows = logits.shape[0]
    for r0 in range(0, max(rows, 1), MAX_CALL_ROWS):   # one call for <= 1,024 rows (decode graphs); eager prefill: slices
        r1 = min(rows, r0 + MAX_CALL_ROWS)
        ext_topk().block_max(logits[r0:r1], lens[r0:r1], out[r0:r1], int(max_len))


def exact_topk(logits: torch.Tensor, lens: torch.Tensor, bmax: torch.Tensor, out: torch.Tensor, slow: torch.Tensor,
               cblk: torch.Tensor | None = None, stop: int = 0, glob: "GlobalOut | None" = None) -> None:
    if glob is not None:   # + FlashMLA's global slots / lengths in the same kernel (= compute_global_topk_indices_and_lens)
        ext_ftopk().topk(logits, lens, bmax, TOPK, out, cblk, slow, stop, glob.gout, glob.glen, glob.t2r, glob.bt,
                         int(glob.bs), glob.valid)
        return
    ext_topk().topk(logits, lens, bmax, TOPK, out, cblk, slow, stop)


@dataclass
class GlobalOut:
    """FlashMLA's sparse inputs of one index layer (what compute_global_topk_indices_and_lens returns), written by the
    top-k when OwnedIndexer.fuse: gout [rows, 512] int32 global compressed-KV slots, glen [rows] int32, from t2r [rows]
    int32 (row -> block-table row), bt [*, P] int32 (the rank's compressed-KV block table), bs (positions per page),
    valid [rows] bool (padding rows get length 0)."""
    gout: torch.Tensor
    glen: torch.Tensor
    t2r: torch.Tensor
    bt: torch.Tensor
    bs: int
    valid: torch.Tensor


def global_topk(top: torch.Tensor, glob: GlobalOut) -> None:
    """Unfused: vLLM's compute_global_topk_indices_and_lens into glob (the engine hook's path before fuse)."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens

    g, ln = compute_global_topk_indices_and_lens(top, glob.t2r, glob.bt, int(glob.bs), glob.valid)
    glob.gout.copy_(g)
    glob.glen.copy_(ln)


def select_candidates_bmax(bmax: torch.Tensor, nb_lens: torch.Tensor, out: torch.Tensor, from_lens: bool = False) -> None:
    """Layer 20 from the block maxima (idx2f_topk.cu select_blocks): the same block keys, pin, ties and -inf drop as
    select_candidate_blocks over the logits; ascending block ids, -1 padded. from_lens: nb_lens holds the ratio-1 row
    lengths (the fused path: no separate ceil(len / 8) kernels)."""
    if from_lens:
        ext_ftopk().select_blocks(bmax, nb_lens, CAND_K, out, 1)
        return
    ext_topk().select_blocks(bmax, nb_lens, CAND_K, out)


def cand_prep(cand: torch.Tensor, seq_lens: torch.Tensor, max_len: int, ws: "EagerWorkspace | None" = None) -> CandPrep:
    B = seq_lens.shape[0]
    rows = B * NEXT_N
    dev = cand.device
    i32 = dict(dtype=torch.int32, device=dev)
    if ws is not None:
        p = ws.prep_bufs(B, cblk=True, upk=False)
    else:
        p = CandPrep(torch.empty((B, NEXT_N * CAND_K), **i32), torch.empty((B, NEXT_N * CAND_K, NEXT_N), **i32),
                     torch.empty((B,), **i32), torch.empty((rows, CAND_K), **i32), torch.empty((rows,), **i32))
    nwords = -(-(-(-int(max_len) // CAND_BLOCK)) // 32)
    ext_cand().prep(cand[:rows], seq_lens.reshape(-1), NEXT_N, nwords, p.ublk, p.uslot, p.nunion, p.cblk, p.clen)
    return p


def cand_prep2(cand: torch.Tensor, seq_lens: torch.Tensor, block_table: torch.Tensor | None = None,
               ws: "EagerWorkspace | None" = None, fused: bool = False) -> CandPrep:
    """prep from ASCENDING candidate lists (select_blocks' output): no atomics; cblk = the candidate rows themselves.
    With the layer-20 block table it also packs each union block's page for the candidate-logits producer (upk)."""
    B = seq_lens.shape[0]
    rows = B * NEXT_N
    dev = cand.device
    i32 = dict(dtype=torch.int32, device=dev)
    if ws is not None:
        p = ws.prep_bufs(B, cblk=False, upk=block_table is not None)
        p.cblk = cand[:rows]
    else:
        p = CandPrep(torch.empty((B, NEXT_N * CAND_K), **i32), torch.empty((B, NEXT_N * CAND_K, NEXT_N), **i32),
                     torch.empty((B,), **i32), cand[:rows], torch.empty((rows,), **i32),
                     torch.empty((B, NEXT_N * CAND_K), **i32) if block_table is not None else None)
    # fused (OwnedIndexer.fuse): the same tables, the per-union-entry part lane-parallel (gen_fused.py)
    (ext_ftopk() if fused else ext_topk()).prep2(cand, seq_lens.reshape(-1), p.ublk, p.uslot, p.nunion, p.clen,
                                                 block_table, p.upk)
    return p


def cand_logits(q: torch.Tensor, ck: CompactK, weights: torch.Tensor, seq_lens: torch.Tensor, p: CandPrep,
                out: torch.Tensor, bm: torch.Tensor | None = None) -> None:
    """bm: fused block maxima of the compact rows (= block_max(out, p.clen, bm, 16384), bit for bit)."""
    B = q.shape[0]
    if B <= MAX_CALL_B:
        if bm is not None:
            ext_fcand().logits(q, ck.kv, weights, seq_lens.reshape(-1), ck.block_table, p.ublk, p.uslot, p.nunion, out,
                               p.upk, bm)
        else:
            ext_logits().logits(q, ck.kv, weights, seq_lens.reshape(-1), ck.block_table, p.ublk, p.uslot, p.nunion,
                                out, p.upk)
        return
    assert bm is None, "fused block maxima: one call (<= 1,024 pseudo-requests)"
    for b0 in range(0, B, MAX_CALL_B):                 # eager prefill steps with > 1,024 pseudo-requests: slices
        b1 = min(B, b0 + MAX_CALL_B)
        r0, r1 = NEXT_N * b0, NEXT_N * b1
        ext_logits().logits(q[b0:b1], ck.kv, weights[r0:r1], seq_lens[b0:b1].reshape(-1), ck.block_table[b0:b1],
                          p.ublk[b0:b1], p.uslot[b0:b1], p.nunion[b0:b1], out[r0:r1],
                          p.upk[b0:b1] if p.upk is not None else None)


def cand_remap(idx: torch.Tensor, p: CandPrep) -> None:
    ext_cand().remap(idx, p.cblk)


class EagerWorkspace:
    """Fixed scratch of one device for eager (lean) indexer steps: dense-logit row slices and the
    consumers' compact chunks (fp32, `budget` bytes), block maxima, candidate-prep tables. Every eager layer call works
    in views of it, so the indexer's memory is CONSTANT whatever the rows or the context (no allocator growth from
    context-sized blocks). One stream per device (stream-ordered reuse); graphs keep their own buffers."""

    _per: dict = {}
    PREP_PER_B = 3 * NEXT_N * CAND_K + NEXT_N * CAND_K * NEXT_N + NEXT_N * CAND_K + 1 + NEXT_N  # i32 / pseudo-request

    @classmethod
    def of(cls, dev, budget: int = 256 << 20, max_cb: int = 512) -> "EagerWorkspace":
        dev = torch.device(dev)
        if dev.index is None:                          # "cuda" == the current device: one workspace per physical GPU
            dev = torch.device("cuda", torch.cuda.current_device())
        key = (dev.index, budget, max_cb)
        w = cls._per.get(key)
        if w is None:
            w = cls._per[key] = cls(dev, budget, max_cb)
        return w

    def __init__(self, dev: torch.device, budget: int, max_cb: int):
        self.dev, self.budget = dev, int(budget)
        # consumer chunk: pseudo-requests whose [4 rows, 16,384] fp32 compact rows fit in the budget
        self.cb = max(1, min(max_cb, MAX_CALL_B, self.budget // (NEXT_N * CAND_K * CAND_BLOCK * 4)))
        with torch.cuda.device(dev):
            self.f32 = torch.empty(self.budget // 4, dtype=torch.float32, device=dev)
            self.bmax = torch.empty(max(self.budget // 32, self.cb * NEXT_N * CAND_K), dtype=torch.float32, device=dev)
            self.prep = torch.empty(self.cb * self.PREP_PER_B, dtype=torch.int32, device=dev)

    def bytes(self) -> int:
        return self.f32.numel() * 4 + self.bmax.numel() * 4 + self.prep.numel() * 4

    def logits(self, rows: int, width: int) -> torch.Tensor:
        assert rows * width <= self.f32.numel(), (rows, width)
        return self.f32[: rows * width].view(rows, width)

    def bm(self, rows: int, nb: int) -> torch.Tensor:
        nb = max(int(nb), 1)
        assert rows * nb <= self.bmax.numel(), (rows, nb)
        return self.bmax[: rows * nb].view(rows, nb)

    def prep_bufs(self, B: int, cblk: bool, upk: bool) -> CandPrep:
        assert B <= self.cb, (B, self.cb)
        K = NEXT_N * CAND_K
        o = 0

        def take(n, shape):
            nonlocal o
            t = self.prep[o: o + n].view(shape)
            o += -(-n // 4) * 4                         # 16-B aligned pieces
            return t
        ublk = take(B * K, (B, K))
        uslot = take(B * K * NEXT_N, (B, K, NEXT_N))
        nunion = take(B, (B,))
        cb = take(B * NEXT_N * CAND_K, (B * NEXT_N, CAND_K)) if cblk else None
        clen = take(B * NEXT_N, (B * NEXT_N,))
        up = take(B * K, (B, K)) if upk else None
        return CandPrep(ublk, uslot, nunion, cb, clen, up)


# ------------------------------------------------------------------------------------------------- decode graphs
GRAPH_LEVELS = (262_144, 524_288, 655_360)   # context bounds of the decode-graph variants


def graph_level(max_ctx: int) -> int:
    """The decode-graph variant a step needs: max_ctx = the step's largest position + 1 over all live rows. 0 = the base
    graphs (every layer on the fast kernels), 1 = layer 20 + consumers on the stock long-row kernels (ratio-2 layers
    fast), 2 = every dense layer stock. The same per-layer rule as the eager path (stock when a layer's longest row
    exceeds LONG_ROW compressed positions), so a step's kernels are the eager path's at every level."""
    for i, c in enumerate(GRAPH_LEVELS):
        if max_ctx <= c:
            return i
    raise ValueError(f"context {max_ctx} > max_model_len {GRAPH_LEVELS[-1]}")


def decode_graph_bounds(oi: "OwnedIndexer", max_ctx: int, bmax: torch.Tensor | None = None) -> dict:
    """Fixed bounds of a decode-graph bucket valid for every row up to max_ctx (a GRAPH_LEVELS entry): max_len / width
    per layer (layers whose bound exceeds LONG_ROW take the stock select / top-k statically, as eager does), the block
    maxima for the fast layers and the consumers, nb_lens. Returns the buffer sizes (bytes): the shared logits scratch
    this bucket needs (rows x the widest dense layer x 4 B; pass one per device as oi.logits_scratch), bmax, and the
    stock path's per-call graph-pool temporaries. The caller fixes the stock top-k's host bound
    (LayerIn.max_seq_len = max_ctx) and its backend (LayerIn.global_rows, production_backend)."""
    rows = oi.B * NEXT_N
    for L in LAYERS:
        oi.max_len[L] = int(max_ctx) // RATIO[L]
    for L in KV_SOURCES:
        oi.width[L] = max(256, -(-oi.max_len[L] // 256) * 256)
    fast_nb = [-(-oi.max_len[L] // CAND_BLOCK) for L in KV_SOURCES if oi.max_len[L] <= LONG_ROW]
    nbw = max(fast_nb + [CAND_K])
    if oi.mode in ("fast", "hybrid"):
        oi.bmax = bmax if bmax is not None else torch.empty((rows, nbw), dtype=torch.float32, device=oi.dev)
        oi.nb_lens = torch.zeros(rows, dtype=torch.int32, device=oi.dev)
    long_layers = [L for L in KV_SOURCES if oi.max_len[L] > LONG_ROW]
    stock_tmp = sum(rows * -(-oi.width[L] // CAND_BLOCK) * 4 for L in long_layers if L == 20)  # block scores
    prep_tmp = oi.B * (NEXT_N * CAND_K * (1 + NEXT_N + 1) + NEXT_N) * 4 if long_layers else 0
    return dict(logits_scratch=rows * max(oi.width[L] for L in KV_SOURCES) * 4, bmax=rows * nbw * 4,
                long_layers=long_layers, stock_temp=stock_tmp, prep_temp=prep_tmp)


def logits_scratch_bytes(rows_max: int, max_ctx: int = GRAPH_LEVELS[-1]) -> int:
    """One shared scratch per device for every long-row decode-graph variant up to rows_max rows and max_ctx context."""
    return rows_max * max(256, -(-int(max_ctx) // 256) * 256) * 4


# ------------------------------------------------------------------------------------------------------ step API
@dataclass
class LayerIn:
    """Inputs of one index layer on one rank (owned rows only, request-major, next_n = 4 rows per request)."""
    q: torch.Tensor          # [B_r, 4, 32, 128] float8_e4m3fn (fused RoPE + per-head ue8m0 quant of wq_b(qr))
    weights: torch.Tensor    # [B_r * 4, 32] fp32 (weights_proj(x) * q_scale * 128^-0.5 * 32^-0.5)
    seq_lens: torch.Tensor   # [B_r, 4] int32 compressed row lengths n = (p + 1) // ratio
    max_seq_len: int         # production's host bound for the top-k dispatch (global batch max)
    global_rows: int = 512   # decode rows of the whole step (all ranks): picks production's top-k kernel


@dataclass
class OwnedIndexer:
    """Per-rank indexer state for one decode step shape. `ks[L]` are the rank's compact index-K pages per kv source;
    `cand` [B_r * 4, 2048] int32 is written by layer 20 and read by 24-36; `topk[L]` [B_r * 4, 512] int32 is the output
    (request-local compressed positions, -1 padded) that the attention half reads (one buffer per index layer here so a
    test can keep them all; the step program can share one per index group).

    mode "stock": production's kernels on the owned rows (DeepGEMM logits, ds41h select_blocks on the logits for layer
      20, the stock top-k with production's dispatch, cand_prep / candidate logits / stock top-k / remap for 24-36).
    mode "fast": the same logits, then block_max (one read of the logits) -> layer 20 candidates from the block maxima,
      and the exact block-threshold top-512 (idx2f_topk.cu) instead of the stock top-k (remap fused for 24-36).
    mode "hybrid": fast on layers 2 / 8 / 14 / 20 (block max shared by layer 20's select and top-k), prep2 (packed pages)
      + candidate logits + the stock top-k + remap on 24-36 (cheaper than block max + exact top-k on 16K-wide rows).
    plan() (once per step, outside the graph) takes the host-side bounds; layer() then has no host sync.
    Rows longer than LONG_ROW (262,144) compressed positions (max_len[L] > LONG_ROW): the layer takes production's stock
    select (any width) and stock top-k, and that step's consumers the order-insensitive cand_prep, in every mode.
    keep_logits: keep each dense layer's fp32 logits in self.logits (tests); off, a layer's logits are freed after it."""
    ks: dict[int, CompactK]
    B: int
    dev: torch.device
    mode: str = "stock"
    width: dict[int, int] = field(default_factory=dict)
    max_len: dict[int, int] = field(default_factory=dict)
    cand: torch.Tensor | None = None
    topk: dict[int, torch.Tensor] = field(default_factory=dict)
    sched: dict[int, torch.Tensor] = field(default_factory=dict)
    prep: CandPrep | None = None
    compact: torch.Tensor | None = None
    compact_lens: torch.Tensor | None = None
    logits: dict[int, torch.Tensor] = field(default_factory=dict)
    bmax: torch.Tensor | None = None
    nb_lens: torch.Tensor | None = None
    slow: torch.Tensor | None = None
    keep_logits: bool = False
    cand_ascending: bool = True   # layer 20 wrote ascending candidate ids (prep2 needs them; stock select does not)
    dense_impl: str = "deepgemm"  # "deepgemm" | "idx2f" (multi-device processes: DeepGEMM's JIT works on one device only)
    logits_budget: int = 1 << 30  # bytes of fp32 dense logits per call; above it the dense layers run in row slices of
                                  # whole pseudo-requests (prefill chunks; exact: every kernel is per row). Decode graph
                                  # buckets stay below it (320 rows x 262,144 x 4 B = 335 MB)
    ws: "EagerWorkspace | None" = None  # lean + ws: every per-call buffer is a view of the device's fixed workspace
    fuse: bool = False            # fast mode, dense_impl "idx2f": with layer(..., glob) FlashMLA's global
                                  # slots / lengths from the top-k (no compute_global_topk_indices_and_lens + copies),
                                  # layer-20 select from the row lengths, two-phase prep2; same outputs bit for bit
    canon_stock: bool = field(default_factory=lambda: os.environ.get("IDX2F_CANON_STOCK", "0") == "1")
                                  # stock top-k lists (rows past LONG_ROW) sorted ascending: deterministic, eager ==
                                  # the long-row decode graphs bit for bit
    logits_scratch: torch.Tensor | None = None  # flat fp32: the unsliced dense layers' logits are views of it (one
                                  # scratch per device shared by every layer and every long-row decode graph variant,
                                  # see GRAPH_LEVELS); None: a fresh [rows, width] tensor per layer call (graph pool)
    dl: bool = field(default_factory=lambda: os.environ.get("IDX2F_DL", "0") == "1")
                                  # eager (lean + ws) dense layers on the fast path: ONE host call per layer, the slices
                                  # issued from C++ (idx2f_dl.cu: the same kernels and slices, bit for bit); for long
                                  # contexts the per-slice Python / dispatch cost dominated the step's issue time
    pf_groups: list | None = None  # eager dense layers: host group starts (pf_groups(), over this step's B pseudo-
                                  # requests) -> the logits of each slice from the shared-key prefill kernel (ext_pf:
                                  # the same logits bit for bit, key tiles read once per <= 16 rows of one sequence)
    pf: bool = field(default_factory=lambda: os.environ.get("IDX2F_PF", "0") == "1")
    fuse_bm: bool = False         # + block maxima from the logits epilogue (exact, but the logits kernels are epilogue-
                                  # bound: +55 % logits time vs a 10 to 17 us block_max pass; off)
    lean: bool = False            # eager steps: no persistent compact / block-max buffers (allocated per
                                  # layer call, sized to the call's rows and width), ONE top-512 buffer shared by the 8
                                  # layers (the caller consumes it before the next layer); transient memory per layer
                                  # call ~1.2 x logits_budget + the stock select's scratch on long rows

    def __post_init__(self):
        rows = self.B * NEXT_N
        i32 = dict(dtype=torch.int32, device=self.dev)
        self.cand = torch.empty((rows, CAND_K), **i32)
        if self.lean:
            t = torch.empty((rows, TOPK), **i32)
            for L in LAYERS:
                self.topk[L] = t
            self.compact = None          # allocated by the first consumer layer (eager tail buckets only)
        else:
            for L in LAYERS:
                self.topk[L] = torch.empty((rows, TOPK), **i32)
            self.compact = torch.empty((rows, CAND_K * CAND_BLOCK), dtype=torch.float32, device=self.dev)
        self.slow = torch.zeros((1,), **i32)

    def plan(self, ins: dict[int, LayerIn]) -> None:
        """Once per step (outside the graph): DeepGEMM schedules, widths and the host bounds of the dense layers."""
        mx = 0
        for L in LAYERS:
            self.max_len[L] = int(ins[L].seq_lens.max())
        for L in (2, 8, 14, 20):
            sl = ins[L].seq_lens
            self.width[L] = max(256, int(-(-self.max_len[L] // 256) * 256))   # >= 256: 1-token prompts (length 0)
            if self.dense_impl == "deepgemm":
                self.sched[L] = dg_schedule(sl, self.ks[L].block_size)
            mx = max(mx, self.max_len[L])
        if self.mode in ("fast", "hybrid"):
            rows = self.B * NEXT_N
            nbw = max(-(-min(mx, LONG_ROW) // CAND_BLOCK), CAND_K)   # >= 2,048 blocks: the fast-mode consumers' rows
            self.bmax = None if self.lean else torch.empty((rows, nbw), dtype=torch.float32, device=self.dev)
            self.nb_lens = ((ins[20].seq_lens.reshape(-1) + CAND_BLOCK - 1) // CAND_BLOCK).to(torch.int32).contiguous()

    def plan_bounds(self, ins: dict[int, LayerIn], max_len1: int, nb_lens: torch.Tensor | None = None,
                    bmax: torch.Tensor | None = None) -> None:
        """plan() from host bounds, no device reads (tables2f.GeneralTables): max_len1 = the largest ratio-1 row length
        (max position + 1; ratio 2 = max_len1 // 2, positions ascend in a group); nb_lens = (seq_lens[1] + 7) // 8 when
        the caller derived it on the device, bmax = a [rows, >= nbw] fp32 buffer (else allocated here)."""
        mx = 0
        for L in LAYERS:
            self.max_len[L] = int(max_len1) // RATIO[L]
        for L in KV_SOURCES:
            self.width[L] = max(256, int(-(-self.max_len[L] // 256) * 256))   # >= 256: 1-token prompts (length 0)
            if self.dense_impl == "deepgemm":
                self.sched[L] = dg_schedule(ins[L].seq_lens, self.ks[L].block_size)
            mx = max(mx, self.max_len[L])
        if self.mode in ("fast", "hybrid"):
            rows = self.B * NEXT_N
            nbw = max(-(-min(mx, LONG_ROW) // CAND_BLOCK), CAND_K)
            self.bmax = (bmax.view(-1)[: rows * nbw].view(rows, nbw) if bmax is not None and bmax.numel() >= rows * nbw
                         else None if self.lean else torch.empty((rows, nbw), dtype=torch.float32, device=self.dev))
            self.nb_lens = (nb_lens if nb_lens is not None else
                            ((ins[20].seq_lens.reshape(-1) + CAND_BLOCK - 1) // CAND_BLOCK).to(torch.int32).contiguous())

    def rows_view(self, B: int, ks: dict[int, CompactK]) -> "OwnedIndexer":
        """A per-step indexer over the first B pseudo-requests of this persistent (larger) one: views of its buffers,
        fresh per-step state, no allocation (mixed steps: tables2f.GeneralTables)."""
        import copy

        assert B <= self.B, (B, self.B)
        o = copy.copy(self)
        rows = B * NEXT_N
        o.B, o.ks = B, ks
        o.cand = self.cand[:rows]
        o.compact = self.compact[:rows] if self.compact is not None else None
        o.topk = {L: t[:rows] for L, t in self.topk.items()}
        o.width, o.max_len, o.sched, o.logits = {}, {}, {}, {}
        o.prep = o.compact_lens = o.bmax = o.nb_lens = None
        o.cand_ascending = True
        return o

    def _pf_gstart(self, b0: int, b1: int) -> torch.Tensor | None:
        """The slice [b0, b1)'s group starts (groups cut at the slice edges), relative to b0, int32 on the device."""
        key = (b0, b1)
        cache = self.__dict__.setdefault("_pf_cache", {})
        if cache.get("groups") is not self.pf_groups:
            cache.clear()
            cache["groups"] = self.pf_groups
        if key not in cache:
            g = sorted({b0, b1} | {s for s in self.pf_groups if b0 < s < b1})
            # decode pseudo-requests are groups of 1 (3 of the kernel's 4 warpgroups would idle on their tiles): the
            # shared-key kernel only for slices of mostly multi-member groups
            cache[key] = (torch.tensor([s - b0 for s in g], dtype=torch.int32).pin_memory().to(self.dev, non_blocking=True)
                          if 2 * (len(g) - 1) <= b1 - b0 else None)
        return cache[key]

    def _scratch(self, L: int) -> torch.Tensor | None:
        """The layer's [rows, width] logits as a view of logits_scratch (None: allocate per call)."""
        if self.logits_scratch is None:
            return None
        rows, w = self.B * NEXT_N, max(int(self.width[L]), 256)
        flat = self.logits_scratch.view(-1)            # any contiguous scratch (flat, or a [R, bound] view of one)
        assert flat.numel() >= rows * w, ("logits_scratch too small", rows, w)
        return flat[: rows * w].view(rows, w)

    def _bm(self, r0: int, r1: int, nb: int) -> torch.Tensor:
        """Block maxima rows r0..r1 (>= nb blocks): a view of the persistent buffer, or (lean) a transient one."""
        if self.bmax is not None:
            return self.bmax[r0:r1]
        return torch.empty((r1 - r0, max(nb, 1)), dtype=torch.float32, device=self.dev)

    def layer(self, L: int, x: LayerIn, glob: GlobalOut | None = None) -> torch.Tensor:
        """The indexer of index layer L for the owned rows; returns the top-512 buffer of L. glob: also write FlashMLA's
        global slots / lengths (fused into the top-k when self.fuse and the layer takes the fast path)."""
        top = self._layer(L, x, glob)
        if glob is not None and not self._glob_done:
            global_topk(top, glob)
        return top

    def _layer(self, L: int, x: LayerIn, glob: GlobalOut | None) -> torch.Tensor:
        out = self.topk[L]
        fast = self.mode in ("fast", "hybrid")
        long_rows = self.max_len.get(L, 0) > LONG_ROW
        self._glob_done = False
        if L in CONSUMERS and self.ws is not None and not torch.cuda.is_current_stream_capturing():
            return self._consumer_ws(L, x, out, fast)
        if L in CONSUMERS:
            if L == 24 or self.prep is None:  # one prep per step (the 4 consumers share layer 20's candidates)
                if fast and self.cand_ascending and self.max_len.get(L, 0) <= LONG_ROW:
                    self.prep = cand_prep2(self.cand, x.seq_lens, self.ks[20].block_table, fused=self.fuse)
                else:
                    self.prep = cand_prep(self.cand, x.seq_lens, self.max_len[L])
                self.compact_lens = self.prep.clen.view(self.B, NEXT_N)
            if self.compact is None:
                self.compact = torch.empty((self.B * NEXT_N, CAND_K * CAND_BLOCK), dtype=torch.float32, device=self.dev)
            if self.mode == "fast" and self.fuse and self.B <= MAX_CALL_B:
                bm = self._bm(0, self.B * NEXT_N, CAND_K)
                if self.fuse_bm:
                    cand_logits(x.q, self.ks[20], x.weights, x.seq_lens, self.prep, self.compact, bm)
                else:
                    cand_logits(x.q, self.ks[20], x.weights, x.seq_lens, self.prep, self.compact)
                    block_max(self.compact, self.prep.clen, bm, CAND_K * CAND_BLOCK)
                exact_topk(self.compact, self.prep.clen, bm, out, self.slow, self.prep.cblk, glob=glob)
                self._glob_done = glob is not None
                return out
            cand_logits(x.q, self.ks[20], x.weights, x.seq_lens, self.prep, self.compact)
            if self.mode == "fast":
                bm = self._bm(0, self.B * NEXT_N, CAND_K)
                block_max(self.compact, self.prep.clen, bm, CAND_K * CAND_BLOCK)
                exact_topk(self.compact, self.prep.clen, bm, out, self.slow, self.prep.cblk)
            else:
                stock_topk(self.compact, self.compact_lens, out, min(int(x.max_seq_len), CAND_K * CAND_BLOCK),
                           production_backend(x.global_rows))
                cand_remap(out, self.prep)
            return out
        per_b = NEXT_N * self.width[L] * 4
        budget = self.ws.budget if self.ws is not None else self.logits_budget
        if (self.ws is not None or per_b * self.B > budget or self.B > MAX_CALL_B) and \
                not torch.cuda.is_current_stream_capturing():
            return self._dense_sliced(L, x, min(max(1, budget // per_b), MAX_CALL_B), fast and not long_rows, long_rows)
        if self.fuse and fast and not long_rows and self.dense_impl == "idx2f" and self.B <= MAX_CALL_B:
            lens = x.seq_lens.reshape(-1)
            bm = self._bm(0, self.B * NEXT_N, -(-self.max_len[L] // CAND_BLOCK))
            if self.fuse_bm:
                lg = dense_logits(x.q, self.ks[L], x.weights, x.seq_lens, self.width[L], self._scratch(L), bm=bm)
            else:
                lg = dense_logits(x.q, self.ks[L], x.weights, x.seq_lens, self.width[L], self._scratch(L))
                block_max(lg, lens, bm, self.max_len[L])
            if self.keep_logits:
                self.logits[L] = lg
            if L == 20:
                select_candidates_bmax(bm, lens, self.cand, from_lens=True)
                self.prep, self.cand_ascending = None, True
            exact_topk(lg, lens, bm, out, self.slow, glob=glob)
            self._glob_done = glob is not None
            return out
        if self.dense_impl == "idx2f":
            lg = dense_logits(x.q, self.ks[L], x.weights, x.seq_lens, self.width[L], self._scratch(L))
        else:
            lg = dg_logits(x.q, self.ks[L], x.weights, x.seq_lens, self.width[L], self.sched[L])
        if self.keep_logits:
            self.logits[L] = lg
        if fast and not long_rows:
            lens = x.seq_lens.reshape(-1)
            bm = self._bm(0, self.B * NEXT_N, -(-self.max_len[L] // CAND_BLOCK))
            block_max(lg, lens, bm, self.max_len[L])
            if L == 20:
                select_candidates_bmax(bm, self.nb_lens, self.cand)
                self.prep, self.cand_ascending = None, True
            exact_topk(lg, lens, bm, out, self.slow)
            return out
        if L == 20:
            if long_rows:
                select_candidates_stock(lg, x.seq_lens, self.cand)
                self.cand_ascending = False
            else:
                select_candidates(lg, x.seq_lens, self.cand)
                self.cand_ascending = True
            self.prep = None
        stock_topk(lg, x.seq_lens, out, x.max_seq_len, production_backend(x.global_rows))
        if self.canon_stock:
            canon_topk(out)
        return out

    def _consumer_ws(self, L: int, x: LayerIn, out: torch.Tensor, fast: bool) -> torch.Tensor:
        """Consumer layer (24-36) in chunks of ws.cb pseudo-requests, every buffer a view of the workspace (eager only).
        One chunk: the prep is computed once per step (layer 24) as in layer(); several: per chunk and layer."""
        ws = self.ws
        use2 = fast and self.cand_ascending and self.max_len.get(L, 0) <= LONG_ROW
        k20 = self.ks[20]
        single = self.B <= ws.cb
        for b0 in range(0, self.B, ws.cb):
            b1 = min(self.B, b0 + ws.cb)
            r0, r1 = NEXT_N * b0, NEXT_N * b1
            sl = x.seq_lens[b0:b1]
            bt = k20.block_table[b0:b1]
            if single and L != 24 and self.prep is not None:
                p = self.prep
            else:
                p = (cand_prep2(self.cand[r0:r1], sl, bt, ws) if use2 else
                     cand_prep(self.cand[r0:r1], sl, self.max_len[L], ws))
                if single:
                    self.prep = p
            comp = ws.logits(r1 - r0, CAND_K * CAND_BLOCK)
            cand_logits(x.q[b0:b1], CompactK(k20.kv, bt, k20.block_size), x.weights[r0:r1], sl, p, comp)
            if self.mode == "fast":
                bm = ws.bm(r1 - r0, CAND_K)
                block_max(comp, p.clen, bm, CAND_K * CAND_BLOCK)
                exact_topk(comp, p.clen, bm, out[r0:r1], self.slow, p.cblk)
            else:
                stock_topk(comp, p.clen.view(b1 - b0, NEXT_N), out[r0:r1], min(int(x.max_seq_len), CAND_K * CAND_BLOCK),
                           production_backend(x.global_rows))
                cand_remap(out[r0:r1], p)
        return out

    def _dense_sliced(self, L: int, x: LayerIn, bstep: int, fast: bool, long_rows: bool) -> torch.Tensor:
        """Dense layer in slices of `bstep` pseudo-requests (eager only: the schedules are recomputed per slice)."""
        out = self.topk[L]
        ck = self.ks[L]
        if L == 20:
            self.prep = None
            self.cand_ascending = not long_rows
        if (self.dl and fast and self.ws is not None and self.bmax is None and self.dense_impl == "idx2f"
                and self.nb_lens is not None):
            ext_dl().dense_layer(x.q, ck.kv, x.weights, x.seq_lens.reshape(-1), ck.block_table, self.ws.f32,
                                 self.ws.bmax, bstep, self.width[L], self.max_len[L], TOPK, out, self.slow,
                                 self.nb_lens if L == 20 else None, self.cand if L == 20 else None, CAND_K)
            return out
        for b0 in range(0, self.B, bstep):
            b1 = min(self.B, b0 + bstep)
            r0, r1 = NEXT_N * b0, NEXT_N * b1
            sl = x.seq_lens[b0:b1]
            sub = CompactK(ck.kv, ck.block_table[b0:b1], ck.block_size)
            gst = (self._pf_gstart(b0, b1) if self.dense_impl == "idx2f" and self.pf and self.pf_groups is not None
                   and self.ws is not None else None)
            if gst is not None:   # (slices whose groups average < 2 pseudo-requests keep the default kernel: _pf_gstart None)
                lg = self.ws.logits(r1 - r0, self.width[L])
                ext_pf().logits_prefill(x.q[b0:b1], ck.kv, x.weights[r0:r1], sl.reshape(-1), ck.block_table[b0:b1],
                                        gst, lg, 0)
            elif self.dense_impl == "idx2f":
                lg = dense_logits(x.q[b0:b1], sub, x.weights[r0:r1], sl, self.width[L],
                                  self.ws.logits(r1 - r0, self.width[L]) if self.ws is not None else None)
            else:
                lg = dg_logits(x.q[b0:b1], sub, x.weights[r0:r1], sl, self.width[L], dg_schedule(sl, ck.block_size))
            lens = sl.reshape(-1)
            if fast:
                nb = -(-self.max_len[L] // CAND_BLOCK)
                bm = self.ws.bm(r1 - r0, nb) if self.ws is not None and self.bmax is None else self._bm(r0, r1, nb)
                block_max(lg, lens, bm, self.max_len[L])
                if L == 20:
                    select_candidates_bmax(bm, self.nb_lens[r0:r1], self.cand[r0:r1])
                exact_topk(lg, lens, bm, out[r0:r1], self.slow)
            else:
                if L == 20:
                    (select_candidates_stock if long_rows else select_candidates)(lg, sl, self.cand[r0:r1])
                stock_topk(lg, sl, out[r0:r1], x.max_seq_len, production_backend(x.global_rows))
                if self.canon_stock:
                    canon_topk(out[r0:r1])
            del lg
        return out
