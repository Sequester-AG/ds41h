"""Vocab-parallel DSpark drafting + block verification for TP=4 (stock vLLM 0.32.0).

Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/vp_sampling.py). See LICENSES/BRRR-Apache-2.0.txt.

Stock path being replaced
-------------------------
Drafting (DSparkSpeculator._sample_sequential, spec_decode/dspark/speculator.py:156-199): full-vocab base
logits (all-gathered over TP), then per draft step a full [V, 256] Markov GEMV with the replicated
markov_w2 (qwen3_dspark.py:48-90, disable_tp=True), a bf16 add, and gumbel_sample over the full vocab
that also caches the bf16 draft logits [max_reqs, k, V] (gumbel.py:336-409).
Verification (RejectionSampler.__call__, rejection_sampler.py:407 -> rejection_sample(block verification),
rejection_sampler_utils.py:1015-1328) on full-vocab target logits (all-gathered over TP).

Vocab-parallel path
-------------------
Rank r owns token ids [r*V/4, (r+1)*V/4) (the lm_head TP shard). It keeps its local slice of markov_w2
and its local slice of the draft-logit cache; w1 stays replicated. Per draft step each rank computes its
local logits = bf16(base + bf16(emb . w2_r)) (the stock rounding sequence), caches them, and reduces them
to 5 words per row: Gumbel-argmax value and token (noise keyed by *global* token id with stock's
gumbel_noised_argmax, so the vocab-parallel argmax is bit-identical to stock's full-vocab argmax given
equal logits), the winner's raw logit, and (max, sum-exp) of logits/T. The 5 words are LL-gathered and
every rank picks the same winner. Verification gathers 3 stat words per logit row twice (target
max/sum-exp/owned-draft-logit; residual mass/resample Gumbel value/token) and then runs exactly stock's
block-verification decision (_rejection_kernel USE_BLOCK_VERIFICATION, rejection_sampler_utils.py:
488-697) and residual resampling (_resample_kernel, :734-947) with stock's RNG (tl_rand32 for u, Gumbel
noise keyed by global token id at the row's position).

Exactness: identical math to stock up to fp32 summation order in the log-sum-exp and residual-mass
reductions (and the order of the 256-term Markov dot product), so drafts and verification outcomes
match stock bit-for-bit except at measure-zero ties/thresholds; the result is an exact block-verification
sampler of the target distribution either way.

Optional DS41H_SAMPLING_W2_FP8=1 stores the local markov_w2 slice as per-row-scaled e4m3 (halves the
per-step weight read); the drafts then follow a slightly different q, verification stays exact for p.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.triton_utils import tl, tldevice, triton

try:  # griddepcontrol.wait for the PDL-launched verify kernels (DS41H_SAMPLING_RESID_PDL)
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
except ImportError:  # pragma: no cover - older Triton: the flag stays off
    _gdc_wait = None
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable, gumbel_noised_argmax, murmur3_uniform32, tl_rand32

VERIFY_BLOCK = 2048
DRAFT_BLOCK = 128
RANK_CHUNK = 32
DRAFT_WORDS = 5
STAT_WORDS = 3
W2_FP8 = os.environ.get("DS41H_SAMPLING_W2_FP8", "0") not in ("0", "", "false")
HEAD_NORM_FUSED = os.environ.get("DS41H_DRAFT_HEAD_NORM", "0") not in ("0", "", "false")  # final RMSNorm inside the FP8 draft head
# DS41H_SAMPLING_NOISE_EARLY=1: the residual-resampling Gumbel noise of every verified row (depends only on seed, position and
# token id) is computed on the side stream while the verification head GEMV runs; _target_residual_kernel loads it instead
# of hashing per element on the critical path. Bitwise identical (same Triton expressions, tests/test_sampling.py).
NOISE_EARLY = os.environ.get("DS41H_SAMPLING_NOISE_EARLY", "0") not in ("0", "", "false")
# DS41H_SAMPLING_RESID_PDL=1 (verification schedule only; bitwise identical): the two LL all-gathers of the verify chain release
# their dependent right after their own PDL wait (all_gather_early) and the residual / reject kernels are PDL-launched: they load
# every gather-independent input (target logits, draft-logit cache, noise, request parameters, drafted tokens) while the
# exchange runs, then griddepcontrol.wait before touching the gathered words.
RESID_PDL = os.environ.get("DS41H_SAMPLING_RESID_PDL", "0") not in ("0", "", "false") and _gdc_wait is not None
NOISE_ROWS = 16  # rows covered by the early noise buffer (more rows: in-kernel noise as before)
# DS41H_VERIFY_HEAD=1: the verification lm_head GEMV (+ the hidden_states[logits_indices] gather) as one ds41h bf16 HMMA
# GEMV (kernels/drafter.cu dvh, fp32 accumulation = stock numerics class) instead of gather + cuBLAS; <= 8 rows.
# DS41H_VERIFY_HEAD=2: the same GEMV on the LOSSLESS 12-bit packed lm_head (ds41h/vhead.py, kernels/vhead.cu: -25 % bytes,
# weights decoded bit-exactly, packed once at load); DS41H_VHEAD_CTL switches stock / packed / alternate at runtime.
VERIFY_HEAD = os.environ.get("DS41H_VERIFY_HEAD", "0") not in ("0", "", "false")
VERIFY_HEAD_MODE = (int(os.environ["DS41H_VERIFY_HEAD"]) if os.environ.get("DS41H_VERIFY_HEAD", "0").isdigit() else 1) if VERIFY_HEAD else 0
# DS41H_SAMPLING_POST_FUSE=1: the stock post_update of GPUModelRunner.postprocess_sampled (input_batch.py:557-614:
# last_sampled_tokens, total_len, all_token_ids, output_bin_counts, num_computed_tokens) runs inside the rejection kernel
# (same integer state; bin counts via atomics = same totals); patch_sampling skips the stock launch for that step
POST_FUSE = os.environ.get("DS41H_SAMPLING_POST_FUSE", "0") not in ("0", "", "false")

Gather = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (inp [*S], out [R, *S]) -> out

# data_ptr of every vocab-parallel draft-logit cache (stock RejectionSampler must all-gather these).
LOCAL_DRAFT_LOGITS: dict[int, "VPShard"] = {}


@dataclass
class VPShard:
    rank: int
    ranks: int
    start: int  # first global token id owned by this rank
    local: int  # tokens owned by this rank
    vocab: int
    max_reqs: int
    n_spec: int
    w1: torch.Tensor  # [vocab, R] replicated Markov embedding
    w2: torch.Tensor  # [local, R] bf16 slice (or e4m3 with w2_scale)
    w2_scale: torch.Tensor  # [local] fp32 (ones unless FP8)
    w2_fp8: bool
    logits: torch.Tensor  # [max_reqs, n_spec, local] cached pre-temperature draft logits (head dtype)
    round_bf16: bool  # emulate stock bf16 rounding of bias and sum (cache dtype is bf16)
    token_logit: torch.Tensor  # [max_reqs + 1, n_spec] fp32 raw logit of the drafted token (row max_reqs = sink)
    lse: torch.Tensor  # [max_reqs + 1, n_spec] fp32 log-sum-exp of draft logits / T
    draft_blocks: int
    draft_counts: torch.Tensor  # [max_reqs] int32, zero; reset by the last program of each row
    draft_partials: torch.Tensor  # [max_reqs, draft_blocks, 5] fp32
    draft_out: torch.Tensor  # [max_reqs, 5] fp32
    draft_gathered: torch.Tensor  # [ranks * max_reqs * 5] fp32
    emb: torch.Tensor  # [2, max_reqs, R] ping-pong Markov embeddings
    verify_blocks: int
    max_logits: int
    verify_counts: torch.Tensor  # [max_logits] int32
    verify_partials: torch.Tensor  # [max_logits, verify_blocks, 3] fp32
    stats: torch.Tensor  # [max_logits, 3]
    resid: torch.Tensor  # [max_logits, 3]
    g1: torch.Tensor  # [ranks * max_logits * 3]
    g2: torch.Tensor  # [ranks * max_logits * 3]
    temp: torch.Tensor  # [max_reqs] fp32 staged sampling temperatures (device copy of the UVA tensor)
    seeds: torch.Tensor  # [max_reqs] int64 staged seeds
    gather: Gather
    fused: Any = None  # ds41h.drafter.MarkovFused (whole Markov loop in one launch) or None
    verify_fused: Any = None  # ds41h.drafter.VerifyFused (whole verification in one launch) or None
    head: Any = None  # ds41h.drafter.HeadFP8 (e4m3 per-row-scaled lm_head shard) for drafting, or None

    @classmethod
    def create(
        cls,
        *,
        rank: int,
        ranks: int,
        w1: torch.Tensor,
        w2_full: torch.Tensor,
        vocab: int,
        max_reqs: int,
        n_spec: int,
        cache_dtype: torch.dtype,
        gather: Gather,
        w2_fp8: bool = W2_FP8,
    ) -> "VPShard":
        assert vocab % ranks == 0 and w2_full.shape[0] == vocab and w1.shape[0] == vocab
        device = w1.device
        local = vocab // ranks
        start = rank * local
        r_dim = w1.shape[1]
        assert w2_full.shape[1] == r_dim and r_dim % RANK_CHUNK == 0
        w2_local = w2_full[start : start + local]
        if w2_fp8:
            weight = w2_local.float()
            scale = weight.abs().amax(dim=1).clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
            w2 = (weight / scale[:, None]).to(torch.float8_e4m3fn).contiguous()
        else:
            w2 = w2_local.contiguous().clone()
            scale = torch.ones(local, dtype=torch.float32, device=device)
        draft_blocks = triton.next_power_of_2(triton.cdiv(local, DRAFT_BLOCK))
        verify_blocks = triton.next_power_of_2(triton.cdiv(local, VERIFY_BLOCK))
        max_logits = max_reqs * (n_spec + 1)
        f32 = dict(dtype=torch.float32, device=device)
        shard = cls(
            rank=rank,
            ranks=ranks,
            start=start,
            local=local,
            vocab=vocab,
            max_reqs=max_reqs,
            n_spec=n_spec,
            w1=w1.contiguous(),
            w2=w2,
            w2_scale=scale.float().contiguous(),
            w2_fp8=w2_fp8,
            logits=torch.zeros((max_reqs, n_spec, local), dtype=cache_dtype, device=device),
            round_bf16=cache_dtype == torch.bfloat16,
            token_logit=torch.zeros((max_reqs + 1, n_spec), **f32),
            lse=torch.zeros((max_reqs + 1, n_spec), **f32),
            draft_blocks=draft_blocks,
            draft_counts=torch.zeros(max_reqs, dtype=torch.int32, device=device),
            draft_partials=torch.empty((max_reqs, draft_blocks, DRAFT_WORDS), **f32),
            draft_out=torch.empty((max_reqs, DRAFT_WORDS), **f32),
            draft_gathered=torch.empty((ranks * max_reqs * DRAFT_WORDS,), **f32),
            emb=torch.zeros((2, max_reqs, r_dim), dtype=w1.dtype, device=device),
            verify_blocks=verify_blocks,
            max_logits=max_logits,
            verify_counts=torch.zeros(max_logits, dtype=torch.int32, device=device),
            verify_partials=torch.empty((max_logits, verify_blocks, STAT_WORDS), **f32),
            stats=torch.empty((max_logits, STAT_WORDS), **f32),
            resid=torch.empty((max_logits, STAT_WORDS), **f32),
            g1=torch.empty((ranks * max_logits * STAT_WORDS,), **f32),
            g2=torch.empty((ranks * max_logits * STAT_WORDS,), **f32),
            temp=torch.zeros(max_reqs, **f32),
            seeds=torch.zeros(max_reqs, dtype=torch.int64, device=device),
            gather=gather,
        )
        return shard


# ================================================================================================ shared jit
@triton.jit
def _last_program(counts_ptr, row, NB: tl.constexpr):
    # Stores of this program precede the barrier; the acq_rel RMW publishes them (bar.sync + release is
    # cumulative) and lets the last arriving program read every other program's partials.
    tl.debug_barrier()
    last = tl.atomic_add(counts_ptr + row, 1, sem="acq_rel", scope="gpu") == NB - 1
    if last:
        tl.store(counts_ptr + row, 0)
    return last


@triton.jit
def _block_stats(x):
    m = tl.max(x, axis=0)
    return m, tl.where(m > float("-inf"), tl.sum(tl.exp(x - m), axis=0), 0.0)


@triton.jit
def _merge(m, s):
    top = tl.max(m, axis=0)
    return top, tl.sum(tl.where(m > float("-inf"), s * tl.exp(m - top), 0.0), axis=0)


@triton.jit
def _logsumexp(m, s):
    top, total = _merge(m, s)
    return top + tl.log(total)


@triton.jit
def _pick(v, lanes, i):
    return tl.sum(tl.where(lanes == i, v, 0), axis=0)


@triton.jit
def _nan_to_neg_inf(v):
    return tl.where(v != v, float("-inf"), v)


# ================================================================================================ drafting
@triton.jit
def _draft_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    base_ptr,
    base_stride,
    emb_ptr,
    emb_stride,
    w2_ptr,
    w2_scale_ptr,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col,
    idx_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    temp_ptr,
    local_vocab,
    vocab_start,
    RANK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    W2_FP8: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    # Markov bias = emb . w2[token] for this rank's tokens (stock: F.linear(emb, markov_w2) in bf16).
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in tl.range(0, RANK, CHUNK):  # rolled: an unrolled 256-wide K blows the register budget
        k = k0 + tl.arange(0, CHUNK)
        w = tl.load(w2_ptr + offs[:, None].to(tl.int64) * RANK + k[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        e = tl.load(emb_ptr + row * emb_stride + k).to(tl.float32)
        acc += tl.sum(w * e[None, :], axis=1)
    if W2_FP8:
        acc = acc * tl.load(w2_scale_ptr + offs, mask=mask, other=0.0)
    base = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    if ROUND_BF16:
        # stock: bias = bf16(linear), logits_i = bf16(base + bias) (speculator.py:184-185)
        bias = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        logits = (base + bias).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    else:
        logits = base + acc
    logits = tl.where(mask, logits, float("-inf"))
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    # Cache pre-temperature logits (gumbel_block_argmax, gumbel.py:236-256).
    tl.store(
        cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs,
        logits.to(cache_ptr.dtype.element_ty),
        mask=mask & valid,
    )
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    # sample_pos is the predicted position P; draws are keyed by P - 1 (speculator.py:136-148).
    pos = tl.load(pos_ptr + row * pos_stride) - 1
    value, idx = gumbel_noised_argmax(
        logits, vocab_start + offs, mask, seed, pos, temp, IS_DRAFTING=True, USE_FP64=False, APPLY_TEMPERATURE=True
    )
    t = tl.where(temp > 0.0, temp, 1.0)
    m, s = _block_stats(tl.where(mask, logits / t, float("-inf")))
    p = part_ptr + (row * NB + b) * 5
    tl.store(p, value)
    tl.store(p + 1, (vocab_start + b * BLOCK + idx).to(tl.float32))
    tl.store(p + 2, _pick(logits, lanes, idx))
    tl.store(p + 3, m)
    tl.store(p + 4, s)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 5
        vals = tl.load(q, volatile=True)
        win = tl.argmax(vals, axis=0)
        o = out_ptr + row * 5
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


@triton.jit
def _draft_combine_kernel(
    gathered_ptr,
    rows,
    tokens_ptr,
    tokens_stride,
    col,
    emb_ptr,
    emb_stride,
    w1_ptr,
    idx_ptr,
    idx_stride,
    token_logit_ptr,
    lse_ptr,
    vp_stride,
    sink,
    TP: tl.constexpr,
    RANK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, TP)
    q = gathered_ptr + (ranks * rows + row) * 5
    # Lowest rank wins ties == lowest global token id (stock: first max over vocab blocks).
    win = tl.argmax(tl.load(q), axis=0)
    token = _pick(tl.load(q + 1), ranks, win).to(tl.int64)
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    slot = tl.where(req >= 0, req, sink)
    tl.store(tokens_ptr + row * tokens_stride + col, token)
    tl.store(token_logit_ptr + slot * vp_stride + col, _pick(tl.load(q + 2), ranks, win))
    tl.store(lse_ptr + slot * vp_stride + col, _logsumexp(tl.load(q + 3), tl.load(q + 4)))
    r = tl.arange(0, RANK)
    tl.store(emb_ptr + row * emb_stride + r, tl.load(w1_ptr + token * RANK + r))


def draft_step_local(
    shard: VPShard,
    base: torch.Tensor,  # [rows, local] (row stride arbitrary, unit column stride)
    emb: torch.Tensor,  # [rows, R]
    idx_map: torch.Tensor,  # [rows] request-state index (-1 = padding)
    sample_pos: torch.Tensor,  # [rows] predicted positions
    step: int,
    temperature: torch.Tensor,  # [max_reqs]
    seeds: torch.Tensor,  # [max_reqs]
) -> torch.Tensor:
    rows = base.shape[0]
    assert base.shape[1] == shard.local and base.stride(1) == 1 and rows <= shard.max_reqs
    out = shard.draft_out[:rows]
    _draft_kernel[(rows, shard.draft_blocks)](
        out,
        shard.draft_partials,
        shard.draft_counts,
        base,
        base.stride(0),
        emb,
        emb.stride(0),
        shard.w2,
        shard.w2_scale,
        shard.logits,
        shard.logits.stride(0),
        shard.logits.stride(1),
        step,
        idx_map,
        idx_map.stride(0),
        sample_pos,
        sample_pos.stride(0),
        seeds,
        temperature,
        shard.local,
        shard.start,
        RANK=shard.w1.shape[1],
        CHUNK=RANK_CHUNK,
        BLOCK=DRAFT_BLOCK,
        NB=shard.draft_blocks,
        W2_FP8=shard.w2_fp8,
        ROUND_BF16=shard.round_bf16,
        num_warps=4,
    )
    return out


def draft_combine(
    shard: VPShard,
    gathered: torch.Tensor,  # [ranks, rows, 5]
    draft_tokens: torch.Tensor,  # [>=rows, n_spec] int64
    step: int,
    emb_out: torch.Tensor,  # [rows, R]
    idx_map: torch.Tensor,  # [rows]
) -> None:
    rows = gathered.shape[1]
    _draft_combine_kernel[(rows,)](
        gathered,
        rows,
        draft_tokens,
        draft_tokens.stride(0),
        step,
        emb_out,
        emb_out.stride(0),
        shard.w1,
        idx_map,
        idx_map.stride(0),
        shard.token_logit,
        shard.lse,
        shard.lse.stride(0),
        shard.max_reqs,
        TP=shard.ranks,
        RANK=shard.w1.shape[1],
        num_warps=1,
    )


def draft_markov(
    shard: VPShard,
    base: torch.Tensor,  # [num_reqs, n_spec, local]
    prev: torch.Tensor,  # [num_reqs] anchor token ids
    idx_map: torch.Tensor,  # [num_reqs, n_spec]
    sample_pos: torch.Tensor,  # [num_reqs, n_spec]
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    draft_tokens: torch.Tensor,  # [max_reqs, n_spec] int64
) -> None:
    """Sequential Markov sampling over all draft steps (one rank; gathers through shard.gather)."""
    num_reqs = base.shape[0]
    emb = shard.emb[0, :num_reqs]
    torch.index_select(shard.w1, 0, prev, out=emb)
    gathered = shard.draft_gathered[: shard.ranks * num_reqs * DRAFT_WORDS].view(shard.ranks, num_reqs, DRAFT_WORDS)
    for i in range(base.shape[1]):
        out = draft_step_local(shard, base[:, i], emb, idx_map[:, i], sample_pos[:, i], i, temperature, seeds)
        shard.gather(out, gathered)
        emb_next = shard.emb[(i + 1) % 2, :num_reqs]
        draft_combine(shard, gathered, draft_tokens, i, emb_next, idx_map[:, i])
        emb = emb_next


def _lm(model):
    lm = getattr(model, "language_model", None)
    return lm if lm is not None and hasattr(lm, "lm_head") else model


def _head_weight(lm, hidden_states: torch.Tensor):
    """The plain bf16 lm_head shard if the stock logits path is exactly x @ W^T (no bias / soft cap / scale), else None."""
    head, proc = getattr(lm, "lm_head", None), getattr(lm, "logits_processor", None)
    w = getattr(head, "weight", None)
    if (
        isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == 5120 and w.stride(1) == 1
        and w.stride(0) % 8 == 0 and w.data_ptr() % 16 == 0 and getattr(head, "bias", None) is None
        and proc is not None and proc.soft_cap is None and proc.scale == 1.0 and not proc.logits_as_input
        and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2 and hidden_states.shape[1] == 5120
        and hidden_states.stride(1) == 1 and hidden_states.stride(0) % 8 == 0 and hidden_states.data_ptr() % 16 == 0
        and hidden_states.device == w.device
    ):
        return w
    return None


def verify_logits(model: torch.nn.Module, hidden_states: torch.Tensor, logits_indices: torch.Tensor) -> torch.Tensor:
    """This rank's target logits of the verified rows (stock: model.compute_logits_local(hidden_states[logits_indices]))."""
    m = int(logits_indices.shape[0])
    if VERIFY_HEAD_MODE == 3 and 0 < m <= 8:
        # exact row-range GEMV on the stock bf16 weight (kernels/vhead.cu vrh: fp32 accumulation, one bf16 rounding,
        # deterministic); DS41H_VHEAD_CTL {"mode": "stock"} switches back to cuBLAS at runtime (same-server A/B)
        w = _head_weight(_lm(model), hidden_states)
        if w is not None:
            from ds41h import drafter, vhead

            if vhead.ctl_mode() != "stock":
                idx = logits_indices if logits_indices.dtype == torch.int64 and logits_indices.is_contiguous() else logits_indices.to(torch.int64).contiguous()
                y = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device=hidden_states.device)
                vhead.ext().gemv_rows(hidden_states, idx, w, y, drafter.PDL, vhead.ROWS_VARIANT, 0)
                return y
        return model.compute_logits_local(hidden_states[logits_indices])
    if VERIFY_HEAD_MODE == 2 and 0 < m <= 8:
        lm = _lm(model)
        packed = lm.__dict__.get("_ds41h_vhc")
        proc = getattr(lm, "logits_processor", None)
        if (
            packed is not None and proc is not None and proc.soft_cap is None and proc.scale == 1.0 and not proc.logits_as_input
            and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2 and hidden_states.shape[1] == 5120
            and hidden_states.stride(1) == 1 and hidden_states.stride(0) % 8 == 0 and hidden_states.data_ptr() % 16 == 0
        ):
            from ds41h import drafter, vhead

            if vhead.ctl_mode() != "stock":
                idx = logits_indices if logits_indices.dtype == torch.int64 and logits_indices.is_contiguous() else logits_indices.to(torch.int64).contiguous()
                return packed(hidden_states, idx, m, pdl=drafter.PDL)
        return model.compute_logits_local(hidden_states[logits_indices])
    if VERIFY_HEAD and 0 < m <= 8:
        lm = _lm(model)
        head, proc = getattr(lm, "lm_head", None), getattr(lm, "logits_processor", None)
        w = getattr(head, "weight", None)
        if (
            isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == 5120 and w.stride(1) == 1
            and w.shape[0] % 16 == 0 and w.stride(0) % 8 == 0 and w.data_ptr() % 16 == 0 and getattr(head, "bias", None) is None
            and proc is not None and proc.soft_cap is None and proc.scale == 1.0 and not proc.logits_as_input
            and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2 and hidden_states.shape[1] == 5120
            and hidden_states.stride(1) == 1 and hidden_states.stride(0) % 8 == 0 and hidden_states.data_ptr() % 16 == 0
        ):
            from ds41h import drafter

            idx = logits_indices if logits_indices.dtype == torch.int64 and logits_indices.is_contiguous() else logits_indices.to(torch.int64).contiguous()
            y = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device=hidden_states.device)
            drafter.ext().head_bf16(hidden_states, idx, w, y, drafter.PDL)
            return y
    return model.compute_logits_local(hidden_states[logits_indices])


def local_logits(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """This rank's slice of lm_head(norm(hidden)) (stock compute_logits without the TP gather)."""
    return model.logits_processor(model.lm_head, model.model.norm(hidden), skip_gather=True)


def draft_logits_local(spec, shard: VPShard, sample_hidden: torch.Tensor) -> torch.Tensor:
    """This rank's draft-head logits [rows, local]: the e4m3 draft head (skinny GEMM, <= 8 rows) when
    present, else the bf16 lm_head (stock compute_logits without the TP gather)."""
    if shard.head is not None and sample_hidden.shape[0] <= shard.head.MAX_ROWS:
        md = spec.model.model.__dict__  # DS41H_DRAFTER_FINAL_FUSE: rms_norm(c) written together with c by the drafter forward
        last = md.get("_ds41h_normed")
        # captured c -> y pairs only while capturing (split draft-tail graphs capture every graph A before the graphs B)
        keep = md.get("_ds41h_normed_keep") if torch.cuda.is_current_stream_capturing() else None
        ptr = sample_hidden.data_ptr()
        hit = (last[1], last[2]) if last is not None and last[0] == ptr else (keep.get(ptr) if keep else None)
        if hit is not None and sample_hidden.shape[0] <= hit[0] and sample_hidden.is_contiguous():
            x = hit[1][: sample_hidden.shape[0]]
            if shard.head.eligible(x):
                return shard.head(x)
        norm = spec.model.model.norm
        if HEAD_NORM_FUSED and shard.head.eligible(sample_hidden) and shard.head.norm_fusable(norm):
            return shard.head(sample_hidden, norm.weight, norm.variance_epsilon)  # RMSNorm fused into the head kernel
        x = norm(sample_hidden)
        if shard.head.eligible(x):
            return shard.head(x)
    return local_logits(spec.model, sample_hidden)


def draft(spec, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """Replacement for DSparkSpeculator._sample_sequential (stock speculator.py:156-199)."""
    shard: VPShard = spec._ds41h_vp
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    if getattr(spec, "sample_from_anchor", False) and spec.num_query_per_req == n_spec:
        # DSpark anchor-as-first layout: sample index of (request r, step j) = query index r * n_spec + j, i.e.
        # sample_indices[:num_sample] == arange(num_sample) (dflash/speculator.py _prepare_dflash_inputs_kernel)
        sample_hidden = head_hidden[:num_sample]
    else:
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base = draft_logits_local(spec, shard, sample_hidden)
    if shard.fused is not None and shard.fused.eligible(num_reqs, n_spec):
        # Whole Markov loop in one launch (ds41h/kernels/drafter.cu); anchors = input_ids[r * num_query_per_req].
        # DS41H_DRAFT_TAIL_EARLY_COMBINE: the runner's req_states.draft_tokens (set on the speculator by patch_tail) is written
        # by the same kernel at each request-state index, so the next step's combine need not wait for the stock index_put.
        req_draft = spec.__dict__.get("_ds41h_req_draft")
        shard.fused(
            base,
            spec.input_buffers.input_ids,
            spec.num_query_per_req,
            spec.sample_idx_mapping,
            spec.sample_pos,
            spec.temperature,
            spec.seeds,
            spec.draft_tokens,
            num_reqs,
            n_spec,
            req_draft=req_draft,
        )
        if req_draft is not None:
            spec._ds41h_req_draft_used = True
        return
    base = base.view(num_reqs, n_spec, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    draft_markov(shard, base, prev, idx_map, sample_pos, spec.temperature, spec.seeds, spec.draft_tokens)


# ================================================================================================ verification
@triton.jit
def _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, count, temp, TP: tl.constexpr, S: tl.constexpr):
    # Per drafted position s < count of a request: log p_target(x_{s+1}) - log q_draft(x_{s+1}).
    lanes = tl.arange(0, S)
    live = lanes < count
    q = g1_ptr + (tl.arange(0, TP)[:, None] * num_logits + start + lanes[None, :]) * 3
    m = tl.load(q, mask=live[None, :], other=float("-inf"))
    s = tl.load(q + 1, mask=live[None, :], other=0.0)
    lse = _logsumexp(m, s)
    owned = tl.max(tl.load(q + 2, mask=live[None, :], other=float("-inf")), axis=0)
    d_lp = tl.load(vp_logit_ptr + req * vp_stride + lanes, mask=live, other=0.0) / temp - tl.load(
        vp_lse_ptr + req * vp_stride + lanes, mask=live, other=0.0
    )
    return owned - lse - d_lp, m, s


@triton.jit
def _target_stats_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    logits_ptr,
    logits_stride,
    draft_sampled_ptr,
    num_logits,
    expanded_idx_ptr,
    temp_ptr,
    local_vocab,
    vocab_start,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
):
    # Words per row: (max of logits/T, sum exp(logits/T - max)) or for greedy (max, argmax token),
    # and the logit/T of the next row's draft token if this rank owns it (else -inf).
    # Mirrors _compute_local_logits_stats_kernel (rejection_sampler_utils.py:194-304); target logits are
    # divided by T exactly like _temperature_kernel (gumbel.py:22-44) for T not in {0, 1}.
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    temp = tl.load(temp_ptr + tl.load(expanded_idx_ptr + row)).to(tl.float32)
    t = tl.where(temp > 0.0, temp, 1.0)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) / t
    if temp == 0.0:
        m, i = tl.max(x, axis=0, return_indices=True)
        s = (vocab_start + b * BLOCK + i).to(tl.float32)
    else:
        m, s = _block_stats(x)
    d = tl.load(draft_sampled_ptr + row + 1, mask=row + 1 < num_logits, other=-1) - vocab_start - b * BLOCK
    owned = (
        tl.load(
            logits_ptr + row * logits_stride + b * BLOCK + d,
            mask=(d >= 0) & (d < BLOCK) & (b * BLOCK + d < local_vocab),
            other=float("-inf"),
        ).to(tl.float32)
        / t
    )
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, m)
    tl.store(p + 1, s)
    tl.store(p + 2, owned)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        ms = tl.load(q, volatile=True)
        ss = tl.load(q + 1, volatile=True)
        o = out_ptr + row * 3
        if temp == 0.0:
            clean = _nan_to_neg_inf(ms)
            tl.store(o, tl.max(clean, axis=0))
            tl.store(o + 1, _pick(ss, blocks, tl.argmax(clean, axis=0)))
        else:
            top, total = _merge(ms, ss)
            tl.store(o, top)
            tl.store(o + 1, total)
        tl.store(o + 2, tl.max(tl.load(q + 2, volatile=True), axis=0))


@triton.jit
def _target_residual_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    g1_ptr,
    logits_ptr,
    logits_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    expanded_idx_ptr,
    local_pos_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    local_vocab,
    vocab_start,
    num_spec,
    noise_ptr,
    noise_stride,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    TP: tl.constexpr,
    S: tl.constexpr,
    NOISE: tl.constexpr,
):
    # Words per row: residual-mass partial sum_x max(p_{<row} * M_b(x) - M_s(x), 0) for this rank's tokens
    # (_compute_local_residual_mass_kernel, :388-485), and the Gumbel-argmax (value, token) of the residual
    # distribution this row would be resampled from if it were the first rejected/bonus row
    # (_resample_kernel, :734-947).
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(expanded_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    step = tl.load(local_pos_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(local_pos_ptr + row + 1, mask=has_next, other=0)
    is_bonus = (row + 1 >= num_logits) | (next_pos == 0)  # last row of its request
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_sampled_ptr + row + 1, mask=has_next, other=-1)
    lanes = tl.arange(0, S)
    terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP, S)
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=lanes < step, other=-1)
    # log p_{step} = cumulative min(0, sum of ratio terms) over the valid drafts before this row
    # (_compute_cumulative_log_p_kernel, :307-385; -1 placeholders carry the previous value).
    log_p = 0.0
    for s in tl.static_range(S):
        take = (temp != 0.0) & (s < step) & (_pick(drafted, lanes, s) >= 0)
        log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) / tl.where(
        temp > 0.0, temp, 1.0
    )
    stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * 3
    lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
    use_draft = (temp != 0.0) & not_bonus & (rejected >= 0)
    t_lp = x - lse
    d_lp = tl.load(
        cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask & use_draft, other=float("-inf")
    ).to(tl.float32) / temp - tl.load(vp_lse_ptr + req * vp_stride + step, mask=use_draft, other=0.0)
    mass = 0.0
    if use_draft & (step >= 1) & (step < num_spec):
        mass = tl.sum(tl.maximum(tl.exp(log_p) * tl.exp(t_lp) - tl.exp(d_lp), 0.0), axis=0)
    value = float("-inf")
    token = vocab_start + b * BLOCK
    if (temp != 0.0) | is_bonus:
        if use_draft:
            # log max(p_tau * M_b - M_s, 0) = a + log1p(-exp(b - a)) (stock :356-379)
            shifted = t_lp + log_p
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x  # bonus row or -1 placeholder: target distribution (stock :336-339)
        if NOISE:
            # gumbel_noised_argmax(APPLY_TEMPERATURE=False) with the noise precomputed by _resid_noise_kernel
            nz = tl.load(noise_ptr + row * noise_stride + offs, mask=mask & (temp != 0.0), other=0.0)
            noised = tl.where(temp != 0.0, tl.where(mask, residual + nz, float("-inf")), residual)
            value, i = tl.max(noised, axis=0, return_indices=True)
        else:
            value, i = gumbel_noised_argmax(
                residual, vocab_start + offs, mask, seed, tl.load(pos_ptr + row), temp, IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False
            )
        token = vocab_start + b * BLOCK + i
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, mass)
    tl.store(p + 1, value)
    tl.store(p + 2, token.to(tl.float32))
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1, volatile=True))
        o = out_ptr + row * 3
        tl.store(o, tl.sum(tl.load(q, volatile=True), axis=0))
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))



@triton.jit
def _target_residual_kernel_pdl(
    out_ptr,
    part_ptr,
    counts_ptr,
    g1_ptr,
    logits_ptr,
    logits_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    expanded_idx_ptr,
    local_pos_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    local_vocab,
    vocab_start,
    num_spec,
    noise_ptr,
    noise_stride,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    TP: tl.constexpr,
    S: tl.constexpr,
    NOISE: tl.constexpr,
):
    # Words per row: residual-mass partial sum_x max(p_{<row} * M_b(x) - M_s(x), 0) for this rank's tokens
    # (_compute_local_residual_mass_kernel, :388-485), and the Gumbel-argmax (value, token) of the residual
    # distribution this row would be resampled from if it were the first rejected/bonus row
    # (_resample_kernel, :734-947).
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(expanded_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    step = tl.load(local_pos_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(local_pos_ptr + row + 1, mask=has_next, other=0)
    is_bonus = (row + 1 >= num_logits) | (next_pos == 0)  # last row of its request
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_sampled_ptr + row + 1, mask=has_next, other=-1)
    lanes = tl.arange(0, S)
    # every input that does not depend on the gathered stats g1 first (same values / math as _target_residual_kernel),
    # then griddepcontrol.wait (the previous kernel is the LL all-gather launched with the early release)
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=lanes < step, other=-1)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) / tl.where(
        temp > 0.0, temp, 1.0
    )
    use_draft = (temp != 0.0) & not_bonus & (rejected >= 0)
    d_lp = tl.load(
        cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask & use_draft, other=float("-inf")
    ).to(tl.float32) / temp - tl.load(vp_lse_ptr + req * vp_stride + step, mask=use_draft, other=0.0)
    if NOISE:
        nz = tl.load(noise_ptr + row * noise_stride + offs, mask=mask & (temp != 0.0), other=0.0)
    _gdc_wait()
    terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP, S)
    log_p = 0.0
    for s in tl.static_range(S):
        take = (temp != 0.0) & (s < step) & (_pick(drafted, lanes, s) >= 0)
        log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
    stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * 3
    lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
    t_lp = x - lse
    mass = 0.0
    if use_draft & (step >= 1) & (step < num_spec):
        mass = tl.sum(tl.maximum(tl.exp(log_p) * tl.exp(t_lp) - tl.exp(d_lp), 0.0), axis=0)
    value = float("-inf")
    token = vocab_start + b * BLOCK
    if (temp != 0.0) | is_bonus:
        if use_draft:
            # log max(p_tau * M_b - M_s, 0) = a + log1p(-exp(b - a)) (stock :356-379)
            shifted = t_lp + log_p
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x  # bonus row or -1 placeholder: target distribution (stock :336-339)
        if NOISE:
            # gumbel_noised_argmax(APPLY_TEMPERATURE=False) with the noise precomputed by _resid_noise_kernel
            noised = tl.where(temp != 0.0, tl.where(mask, residual + nz, float("-inf")), residual)
            value, i = tl.max(noised, axis=0, return_indices=True)
        else:
            value, i = gumbel_noised_argmax(
                residual, vocab_start + offs, mask, seed, tl.load(pos_ptr + row), temp, IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False
            )
        token = vocab_start + b * BLOCK + i
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, mass)
    tl.store(p + 1, value)
    tl.store(p + 2, token.to(tl.float32))
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1, volatile=True))
        o = out_ptr + row * 3
        tl.store(o, tl.sum(tl.load(q, volatile=True), axis=0))
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))


@triton.jit
def _resid_noise_kernel(noise_ptr, noise_stride, pos_ptr, expanded_idx_ptr, temp_ptr, seeds_ptr, local_vocab, vocab_start, BLOCK: tl.constexpr):
    # The Gumbel noise gumbel_noised_argmax(IS_DRAFTING=False, USE_FP64=False) draws for (row, token) in
    # _target_residual_kernel: keyed by (seed, pos of the row, global token id) only -> computable before the logits.
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(expanded_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req)
        u = murmur3_uniform32(seed, tl.load(pos_ptr + row), vocab_start + offs)
        tl.store(noise_ptr + row * noise_stride + offs, -tl.log(-_log1p_neg_stable(u)), mask=mask)


_NOISE_BUF: dict = {}


def residual_noise(shard: "VPShard", pos: torch.Tensor, expanded_idx: torch.Tensor, num_logits: int) -> torch.Tensor | None:
    """Precompute the verification residual Gumbel noise [num_logits, local] fp32 (uses the staged shard.temp / seeds),
    or None if num_logits exceeds the buffer (the residual kernel then hashes in place)."""
    if num_logits > NOISE_ROWS or num_logits == 0:
        return None
    key = (shard.logits.device, shard.local)
    buf = _NOISE_BUF.get(key)
    if buf is None:
        buf = _NOISE_BUF[key] = torch.empty((NOISE_ROWS, shard.local), dtype=torch.float32, device=shard.logits.device)
    _resid_noise_kernel[(num_logits, shard.verify_blocks)](
        buf, buf.stride(0), pos, expanded_idx, shard.temp, shard.seeds, shard.local, shard.start, BLOCK=VERIFY_BLOCK, num_warps=8
    )
    return buf[:num_logits]


@triton.jit
def _target_reject_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    g1_ptr,
    g2_ptr,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    cu_ptr,
    idx_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    nct_ptr,
    last_ptr,
    bins_ptr,
    bins_stride,
    tok_ptr,
    tok_stride,
    total_ptr,
    qsl_ptr,
    TP: tl.constexpr,
    S: tl.constexpr,
    COUNTS: tl.constexpr,
    POST: tl.constexpr,
    BINS: tl.constexpr,
):
    # _rejection_kernel (rejection_sampler_utils.py:488-697) with USE_BLOCK_VERIFICATION, followed by
    # _insert_resampled_kernel (:949-1012); the resampled token is the global Gumbel argmax of row
    # start + accepted from the gathered residual words. COUNTS: also the stock post-processing
    # _get_num_sampled_and_rejected_kernel (input_batch.py:507-533) for this request (saves a launch).
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    start = tl.load(cu_ptr + r).to(tl.int64)
    end = tl.load(cu_ptr + r + 1).to(tl.int64)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    greedy = temp == 0.0
    lanes = tl.arange(0, S)
    ranks = tl.arange(0, TP)
    live = lanes < num_draft
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_sampled_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    terms, m, s = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp, TP, S)
    best = _pick(s, ranks[:, None], tl.argmax(_nan_to_neg_inf(m), axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(
        tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * 3, mask=(lanes < num_draft - 1)[None, :], other=0.0),
        axis=0,
    )
    accepted = tl.zeros((), tl.int64)
    verifying = num_draft >= 0
    log_p = tl.zeros((), tl.float32)
    for i in tl.static_range(S):
        if i < num_draft:
            raw = _pick(drafted, lanes, i)
            valid = raw >= 0
            token = tl.maximum(raw, 0)
            if greedy:
                if verifying:
                    top = _pick(best, lanes, i)
                    ok = (top == token) & valid
                    verifying = ok
                    accepted += ok
                    tl.store(sampled_ptr + r * sampled_stride + i, tl.where(ok, token, top))
            else:
                if valid:
                    log_p = tl.minimum(log_p + _pick(terms, lanes, i), 0.0)
                verifying &= valid
                if verifying:
                    prefix = tl.exp(log_p)
                    if _pick(following, lanes, i) >= 0:
                        weight = _pick(mass, lanes, i)
                        denom = weight + 1.0 - prefix
                        h = tl.where(denom > 0.0, weight / denom, 1.0)
                    else:
                        h = prefix
                    accepted = tl.where(_pick(u, lanes, i) <= h, i + 1, accepted)
                    tl.store(sampled_ptr + r * sampled_stride + i, token)
    n_sampled = (accepted + 1).to(tl.int32)
    n_rejected = tl.zeros((), tl.int32)
    if COUNTS:
        chunked = tl.load(seq_lens_ptr + r) < tl.load(prefill_len_ptr + req)
        n_sampled = tl.where(chunked, 0, n_sampled)
        n_rejected = tl.where(chunked, 0, (end - start).to(tl.int32) - n_sampled)
        tl.store(num_rejected_ptr + r, n_rejected)
    tl.store(num_sampled_ptr + r, n_sampled)
    row = start + accepted
    if (temp != 0.0) | (row == end - 1):
        q = g2_ptr + (ranks * num_logits + row) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1))
        token = _pick(tl.load(q + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + accepted, token)
    if POST:
        # stock post_update (input_batch.py:557-614) for this request, from registers: sampled[i] = drafted[i] for
        # i < accepted, and the token at `accepted` = the residual / bonus argmax (or, greedy mismatch, the target top)
        if req >= 0:
            use_g2 = (temp != 0.0) | (row == end - 1)
            q2 = g2_ptr + (ranks * num_logits + row) * 3
            g2_token = _pick(tl.load(q2 + 2), ranks, tl.argmax(_nan_to_neg_inf(tl.load(q2 + 1)), axis=0)).to(tl.int64)
            fin = tl.where(use_g2, g2_token, _pick(best, lanes, accepted))
            toks = tl.where(lanes < accepted, drafted, fin)
            total = tl.load(total_ptr + req)
            if n_sampled > 0:
                tl.store(last_ptr + req, _pick(toks, lanes, n_sampled - 1))
                tl.store(total_ptr + req, total + n_sampled)
            keep = lanes < n_sampled
            tl.store(tok_ptr + req * tok_stride + total + lanes, toks, mask=keep)
            if BINS:
                tl.atomic_add(bins_ptr + req * bins_stride + toks, 1, mask=keep)
            delta = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r) - n_rejected
            if delta != 0:
                tl.store(nct_ptr + req, tl.load(nct_ptr + req) + delta)



@triton.jit
def _target_reject_kernel_pdl(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    g1_ptr,
    g2_ptr,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    cu_ptr,
    idx_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    nct_ptr,
    last_ptr,
    bins_ptr,
    bins_stride,
    tok_ptr,
    tok_stride,
    total_ptr,
    qsl_ptr,
    TP: tl.constexpr,
    S: tl.constexpr,
    COUNTS: tl.constexpr,
    POST: tl.constexpr,
    BINS: tl.constexpr,
):
    # _rejection_kernel (rejection_sampler_utils.py:488-697) with USE_BLOCK_VERIFICATION, followed by
    # _insert_resampled_kernel (:949-1012); the resampled token is the global Gumbel argmax of row
    # start + accepted from the gathered residual words. COUNTS: also the stock post-processing
    # _get_num_sampled_and_rejected_kernel (input_batch.py:507-533) for this request (saves a launch).
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    start = tl.load(cu_ptr + r).to(tl.int64)
    end = tl.load(cu_ptr + r + 1).to(tl.int64)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    greedy = temp == 0.0
    lanes = tl.arange(0, S)
    ranks = tl.arange(0, TP)
    live = lanes < num_draft
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_sampled_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    _gdc_wait()  # the gathered residual words (previous kernel = the LL all-gather with the early release) are visible
    terms, m, s = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp, TP, S)
    best = _pick(s, ranks[:, None], tl.argmax(_nan_to_neg_inf(m), axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(
        tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * 3, mask=(lanes < num_draft - 1)[None, :], other=0.0),
        axis=0,
    )
    accepted = tl.zeros((), tl.int64)
    verifying = num_draft >= 0
    log_p = tl.zeros((), tl.float32)
    for i in tl.static_range(S):
        if i < num_draft:
            raw = _pick(drafted, lanes, i)
            valid = raw >= 0
            token = tl.maximum(raw, 0)
            if greedy:
                if verifying:
                    top = _pick(best, lanes, i)
                    ok = (top == token) & valid
                    verifying = ok
                    accepted += ok
                    tl.store(sampled_ptr + r * sampled_stride + i, tl.where(ok, token, top))
            else:
                if valid:
                    log_p = tl.minimum(log_p + _pick(terms, lanes, i), 0.0)
                verifying &= valid
                if verifying:
                    prefix = tl.exp(log_p)
                    if _pick(following, lanes, i) >= 0:
                        weight = _pick(mass, lanes, i)
                        denom = weight + 1.0 - prefix
                        h = tl.where(denom > 0.0, weight / denom, 1.0)
                    else:
                        h = prefix
                    accepted = tl.where(_pick(u, lanes, i) <= h, i + 1, accepted)
                    tl.store(sampled_ptr + r * sampled_stride + i, token)
    n_sampled = (accepted + 1).to(tl.int32)
    n_rejected = tl.zeros((), tl.int32)
    if COUNTS:
        chunked = tl.load(seq_lens_ptr + r) < tl.load(prefill_len_ptr + req)
        n_sampled = tl.where(chunked, 0, n_sampled)
        n_rejected = tl.where(chunked, 0, (end - start).to(tl.int32) - n_sampled)
        tl.store(num_rejected_ptr + r, n_rejected)
    tl.store(num_sampled_ptr + r, n_sampled)
    row = start + accepted
    if (temp != 0.0) | (row == end - 1):
        q = g2_ptr + (ranks * num_logits + row) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1))
        token = _pick(tl.load(q + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + accepted, token)
    if POST:
        # stock post_update (input_batch.py:557-614) for this request, from registers: sampled[i] = drafted[i] for
        # i < accepted, and the token at `accepted` = the residual / bonus argmax (or, greedy mismatch, the target top)
        if req >= 0:
            use_g2 = (temp != 0.0) | (row == end - 1)
            q2 = g2_ptr + (ranks * num_logits + row) * 3
            g2_token = _pick(tl.load(q2 + 2), ranks, tl.argmax(_nan_to_neg_inf(tl.load(q2 + 1)), axis=0)).to(tl.int64)
            fin = tl.where(use_g2, g2_token, _pick(best, lanes, accepted))
            toks = tl.where(lanes < accepted, drafted, fin)
            total = tl.load(total_ptr + req)
            if n_sampled > 0:
                tl.store(last_ptr + req, _pick(toks, lanes, n_sampled - 1))
                tl.store(total_ptr + req, total + n_sampled)
            keep = lanes < n_sampled
            tl.store(tok_ptr + req * tok_stride + total + lanes, toks, mask=keep)
            if BINS:
                tl.atomic_add(bins_ptr + req * bins_stride + toks, 1, mask=keep)
            delta = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r) - n_rejected
            if delta != 0:
                tl.store(nct_ptr + req, tl.load(nct_ptr + req) + delta)


@triton.jit
def _stage_kernel(temp_out_ptr, seed_out_ptr, temp_ptr, seeds_ptr, n, N: tl.constexpr):
    i = tl.arange(0, N)
    live = i < n
    tl.store(temp_out_ptr + i, tl.load(temp_ptr + i, mask=live, other=0.0), mask=live)
    tl.store(seed_out_ptr + i, tl.load(seeds_ptr + i, mask=live, other=0), mask=live)


@triton.jit
def _gather_stage_kernel(
    draft_sampled_ptr,
    pos_ptr,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    prefill_len_ptr,
    num_logits,
    temp_out_ptr,
    seed_out_ptr,
    temp_ptr,
    seeds_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
):
    # stock _gather_draft_sampled_kernel (spec_decode/rejection_sampler.py:77-112) + _stage_kernel in one launch
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)
    if tl.program_id(0) == 0:
        i = tl.arange(0, N)
        live = i < n
        tl.store(temp_out_ptr + i, tl.load(temp_ptr + i, mask=live, other=0.0), mask=live)
        tl.store(seed_out_ptr + i, tl.load(seeds_ptr + i, mask=live, other=0), mask=live)


def gather_and_stage(
    shard: VPShard,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stock gather_draft_sampled (draft_sampled, pos) + stage_sampling_params, one launch."""
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    n = min(temperature.shape[0], shard.max_reqs)
    BLOCK_SIZE = 1024
    _gather_stage_kernel[(max(1, triton.cdiv(num_logits, BLOCK_SIZE)),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        shard.temp,
        shard.seeds,
        temperature,
        seeds,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        N=triton.next_power_of_2(n),
    )
    return draft_sampled, pos


def stage_sampling_params(shard: VPShard, temperature: torch.Tensor, seeds: torch.Tensor) -> None:
    """Copy the (UVA) per-request temperature / seed tensors into device buffers, one launch."""
    n = min(temperature.shape[0], shard.max_reqs)
    _stage_kernel[(1,)](shard.temp, shard.seeds, temperature, seeds, n, N=triton.next_power_of_2(n), num_warps=1)


def verify_stats_local(shard: VPShard, logits: torch.Tensor, draft_sampled: torch.Tensor, expanded_idx: torch.Tensor) -> torch.Tensor:
    num_logits = logits.shape[0]
    stats = shard.stats[:num_logits]
    _target_stats_kernel[(num_logits, shard.verify_blocks)](
        stats,
        shard.verify_partials,
        shard.verify_counts,
        logits,
        logits.stride(0),
        draft_sampled,
        num_logits,
        expanded_idx,
        shard.temp,
        shard.local,
        shard.start,
        BLOCK=VERIFY_BLOCK,
        NB=shard.verify_blocks,
        num_warps=8,
    )
    return stats


def verify_residual_local(
    shard: VPShard,
    logits: torch.Tensor,
    g1: torch.Tensor,
    draft_sampled: torch.Tensor,
    expanded_idx: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    pos: torch.Tensor,
    noise: torch.Tensor | None = None,
    pdl: bool = False,
) -> torch.Tensor:
    num_logits = logits.shape[0]
    resid = shard.resid[:num_logits]
    nz = noise if noise is not None else resid
    kern = _target_residual_kernel_pdl if pdl else _target_residual_kernel
    kern[(num_logits, shard.verify_blocks)](
        resid,
        shard.verify_partials,
        shard.verify_counts,
        g1,
        logits,
        logits.stride(0),
        shard.logits,
        shard.logits.stride(0),
        shard.logits.stride(1),
        shard.token_logit,
        shard.lse,
        shard.lse.stride(0),
        draft_sampled,
        num_logits,
        expanded_idx,
        expanded_local_pos,
        pos,
        shard.temp,
        shard.seeds,
        shard.local,
        shard.start,
        shard.n_spec,
        nz,
        nz.stride(0),
        BLOCK=VERIFY_BLOCK,
        NB=shard.verify_blocks,
        TP=shard.ranks,
        S=triton.next_power_of_2(shard.n_spec + 1),
        NOISE=noise is not None,
        num_warps=8,
        launch_pdl=pdl,
    )
    return resid


def verify_reject(
    shard: VPShard,
    g1: torch.Tensor,
    g2: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_reqs: int,
    num_logits: int,
    counts: tuple[torch.Tensor, torch.Tensor] | None = None,
    post: dict | None = None,
    pdl: bool = False,
) -> tuple[torch.Tensor, ...]:
    """(sampled, num_sampled) or, with counts = (seq_lens, prefill_len), (sampled, num_sampled, num_rejected) after
    the stock get_num_sampled_and_rejected adjustment. post (needs counts): the stock post_update state tensors
    (num_computed_tokens, last_sampled_tokens, output_bin_counts or None, all_token_ids, total_len, query_start_loc),
    updated in the same launch."""
    sampled = torch.empty((num_reqs, shard.n_spec + 1), dtype=torch.int64, device=g1.device)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=g1.device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=g1.device) if counts is not None else num_sampled
    seq_lens, prefill_len = counts if counts is not None else (num_sampled, num_sampled)
    if post is not None and counts is None:
        post = None
    bins = post["bins"] if post is not None else None
    pd = post or {}
    dummy = num_sampled
    kern = _target_reject_kernel_pdl if pdl else _target_reject_kernel
    kern[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        g1,
        g2,
        shard.token_logit,
        shard.lse,
        shard.lse.stride(0),
        draft_sampled,
        num_logits,
        cu_num_logits,
        idx_mapping,
        pos,
        shard.temp,
        shard.seeds,
        num_rejected,
        seq_lens,
        prefill_len,
        pd.get("num_computed", dummy),
        pd.get("last_sampled", dummy),
        bins if bins is not None else dummy,
        bins.stride(0) if bins is not None else 0,
        pd.get("all_token_ids", dummy),
        pd["all_token_ids"].stride(0) if post is not None else 0,
        pd.get("total_len", dummy),
        pd.get("query_start_loc", dummy),
        TP=shard.ranks,
        S=triton.next_power_of_2(shard.n_spec + 1),
        COUNTS=counts is not None,
        POST=post is not None,
        BINS=bins is not None,
        num_warps=1,
        launch_pdl=pdl,
    )
    if counts is not None:
        return sampled, num_sampled, num_rejected
    return sampled, num_sampled


def verify(
    shard: VPShard,
    logits: torch.Tensor,  # [num_logits, local] this rank's target logits (unprocessed)
    draft_sampled: torch.Tensor,  # [num_logits] from stock gather_draft_sampled (-1 placeholders)
    pos: torch.Tensor,  # [num_logits]
    expanded_idx: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    staged: bool = False,
    counts: tuple[torch.Tensor, torch.Tensor] | None = None,
    noise: torch.Tensor | None = None,
    post: dict | None = None,
) -> tuple[torch.Tensor, ...]:
    """Vocab-parallel block verification; returns (sampled [num_reqs, k+1] int64, num_sampled [num_reqs]) or, with
    counts = (seq_lens, prefill_len) on the Triton path, also num_rejected (stock get_num_sampled_and_rejected fused).
    staged: temperature / seeds were already copied to shard.temp / shard.seeds (gather_and_stage)."""
    num_logits = logits.shape[0]
    assert logits.shape[1] == shard.local and logits.stride(1) == 1 and num_logits <= shard.max_logits
    if shard.verify_fused is not None and shard.verify_fused.eligible(logits, num_reqs):
        return shard.verify_fused(logits, draft_sampled, pos, expanded_idx, expanded_local_pos, cu_num_logits, idx_mapping, num_reqs, temperature, seeds)
    if not staged:
        stage_sampling_params(shard, temperature, seeds)
    stats = verify_stats_local(shard, logits, draft_sampled, expanded_idx)
    pdl = RESID_PDL and getattr(shard.gather, "_ds41h_early", False)
    g1 = _gather(shard, stats, shard.g1[: shard.ranks * num_logits * STAT_WORDS].view(shard.ranks, num_logits, STAT_WORDS), pdl)
    if noise is not None and noise.shape[0] != num_logits:
        noise = None
    resid = verify_residual_local(shard, logits, g1, draft_sampled, expanded_idx, expanded_local_pos, pos, noise, pdl=pdl)
    g2 = _gather(shard, resid, shard.g2[: shard.ranks * num_logits * STAT_WORDS].view(shard.ranks, num_logits, STAT_WORDS), pdl)
    return verify_reject(shard, g1, g2, draft_sampled, cu_num_logits, idx_mapping, pos, num_reqs, num_logits, counts, post, pdl=pdl)


def _gather(shard: VPShard, t: torch.Tensor, out: torch.Tensor, early: bool) -> torch.Tensor:
    """shard.gather, with the early PDL release (DS41H_SAMPLING_RESID_PDL) when the gather supports it."""
    if early:
        return shard.gather(t, out, early=True)
    return shard.gather(t, out)


# ================================================================================================ setup
def ll_gather() -> Gather | None:
    """LL word gather over the TP group, or None if the LL comm is disabled (the LLComm constructor
    agrees on that across ranks, so every rank gets the same answer)."""
    from ds41h.allreduce import sampling_comm

    comm = sampling_comm()
    if comm.disabled:
        return None

    def gather(inp: torch.Tensor, out: torch.Tensor, early: bool = False) -> torch.Tensor:
        return comm.gather_words(inp, out, early=early)

    gather._ds41h_early = True  # supports the early PDL release (DS41H_SAMPLING_RESID_PDL)
    return gather


def eligibility(spec, model) -> str | None:
    """Reason the vocab-parallel path cannot serve this speculator, or None if it can.

    Every condition is identical on all TP ranks (configs and shapes only)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    from ds41h.patch_core import tp_size

    if tp_size() != 4:
        return "tp_size != 4"
    if spec.draft_logits is None:
        return "draft_sample_method is not probabilistic"
    if getattr(model, "draft_id_to_target_id", None) is not None or spec._d2t_scatter_index is not None:
        return "reduced draft vocabulary"
    if spec.draft_watermarker is not None:
        return "draft watermarking"
    if spec._draft_topk is not None:
        return "dspark_draft_topk"
    if spec.use_fp64_gumbel:
        return "fp64 gumbel"
    if spec.enable_adaptive_verification or spec.use_acceptance_estimator or spec.use_confidence_head:
        return "adaptive verification"
    if not hasattr(model, "model") or not hasattr(model.model, "markov_head") or not hasattr(model, "logits_processor"):
        return "not a DSpark DeepSeek-V4 draft model"
    head = getattr(model, "lm_head", None)
    si = getattr(head, "shard_indices", None)
    if head is None or si is None or getattr(head, "tp_size", 1) != 4:
        return "lm_head is not vocab-parallel over 4 ranks"
    if si.num_org_vocab_padding != 0 or si.num_added_elements != 0 or si.num_org_elements * 4 != spec.vocab_size:
        return "lm_head shard has padding"
    if head.weight.shape[0] != si.num_org_elements:
        return "lm_head weight shape"
    proc = model.logits_processor
    if proc.soft_cap is not None or proc.scale != 1.0 or proc.logits_as_input:
        return "logits processor scaling"
    markov = model.model.markov_head
    w1, w2 = markov.markov_w1, markov.markov_w2
    if not isinstance(w1, torch.nn.Embedding) or getattr(w2, "tp_size", 1) != 1:
        return "markov head layout"
    if not isinstance(getattr(w2, "quant_method", None), UnquantizedEmbeddingMethod) or getattr(w2, "is_w4a16_nvfp4", False):
        return "quantized markov_w2"
    if w2.weight.shape != (spec.vocab_size, w1.weight.shape[1]) or w1.weight.shape[0] != spec.vocab_size:
        return "markov shapes"
    if w1.weight.shape[1] % RANK_CHUNK != 0 or w2.weight.dtype != w1.weight.dtype:
        return "markov rank/dtype"
    if spec.draft_logits.dtype not in (torch.bfloat16, torch.float32):
        return "draft logits dtype"
    return None


def shard_drafts(spec, model) -> VPShard | None:
    """Build this rank's VPShard for a loaded DSpark model (collective: creates the LL gather comm)."""
    import logging

    from ds41h.patch_core import tp_rank

    logger = logging.getLogger("ds41h.sampling")
    reason = eligibility(spec, model)
    if reason is not None:
        logger.info("ds41h sampling: vocab-parallel DSpark disabled (%s); using stock sampling", reason)
        return None
    gather = ll_gather()
    if gather is None:
        logger.warning("ds41h sampling: LL gather unavailable; vocab-parallel DSpark disabled, using stock sampling")
        return None
    rank = tp_rank()
    markov = model.model.markov_head
    shard = VPShard.create(
        rank=rank,
        ranks=4,
        w1=markov.markov_w1.weight.detach(),
        w2_full=markov.markov_w2.weight.detach(),
        vocab=spec.vocab_size,
        max_reqs=spec.max_num_reqs,
        n_spec=spec.num_speculative_steps,
        cache_dtype=spec.draft_logits.dtype,
        gather=gather,
    )
    assert shard.start == model.lm_head.shard_indices.org_vocab_start_index
    spec.draft_logits = shard.logits  # local slice replaces the full [max_reqs, k, V] cache
    LOCAL_DRAFT_LOGITS[shard.logits.data_ptr()] = shard
    from ds41h import drafter

    if drafter.FUSED and not shard.w2_fp8 and shard.logits.dtype == torch.bfloat16:
        comm = drafter.draft_comm("draft")  # collective (all ranks reach this point: eligibility is rank-invariant)
        if not comm.disabled:
            shard.fused = drafter.MarkovFused(comm, shard)
    if drafter.VERIFY_FUSED and shard.logits.dtype == torch.bfloat16:
        comm = drafter.draft_comm("verify")  # collective, separate slots/epoch from drafting
        if not comm.disabled:
            shard.verify_fused = drafter.VerifyFused(comm, shard)
    if drafter.HEAD_FP8:
        try:
            shard.head = drafter.fp8_head(model.lm_head.weight)
        except Exception:  # noqa: BLE001 - rank-local choice (no collective depends on it)
            logger.exception("ds41h sampling: FP8 draft head unavailable; drafting with the bf16 head")
            shard.head = None
    logger.info(
        "ds41h sampling: vocab-parallel DSpark on rank %d: tokens [%d, %d), markov_w2 slice %s, k=%d, max_reqs=%d, "
        "fused markov loop %s, fused verify %s, draft head %s",
        rank,
        shard.start,
        shard.start + shard.local,
        "e4m3" if shard.w2_fp8 else str(shard.w2.dtype),
        shard.n_spec,
        shard.max_reqs,
        shard.fused is not None,
        shard.verify_fused is not None,
        "e4m3" if shard.head is not None else "bf16",
    )
    return shard
