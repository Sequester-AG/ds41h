"""Vocab-parallel DSpark sampling for the ds41h-phys sequence-parallel step program (kernels/head).

The verify head and the draft head are vocab-sharded: rank r owns token ids [r * V/4, (r+1) * V/4) and computes the
logits of ALL rows on its slice (the head is compute/energy-bound, so the replicated head would only add
3/4 GB of weight reads per pass). Sampling then needs only a few words per row from each rank:

  verification (block verification, Sun et al. 2024, exactly stock's rejection_sample(USE_BLOCK_VERIFICATION)):
    k_stats    per row: (max, sum exp) of logits/T on the slice (greedy: max, argmax) + logit/T of the next row's draft
    exchange 1 (all-gather of 4 words per row)
    k_resid    per row: residual-mass partial and the Gumbel argmax of the residual distribution on the slice
    exchange 2
    k_reject   per request (redundantly on every rank): accepted length + resampled / bonus token
  drafting (DSparkSpeculator._sample_sequential: sequential Markov sampling, Gumbel-max keyed by global token id):
    per draft step: k_draft (bias = emb . w2_slice, logits = bf16(base + bf16(bias)), cache, partial Gumbel argmax,
    (max, sum exp) of logits/T) -> exchange (8 words per row) -> k_draft_combine (winner, cached token logit and lse,
    next Markov embedding = w1[token]).

The math, the rounding points and the RNG are those of production's ds41h/vp_sampling.py (itself adapted from BRRR,
github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0), which reproduces stock vLLM 0.32.0 bit for bit except
fp32 summation order in the log-sum-exp / residual-mass reductions. Added here: the grammar bitmask (stock
_apply_grammar_bitmask_kernel semantics: bit 0 = token disallowed -> logit -inf on the TARGET rows only) applied inside
k_stats / k_resid, so a structured-output batch does not need a separate masking pass or the stock fallback path.
Invalid drafts (stock grammar_invalid_drafts) are passed as draft_sampled = -1, as stock's verify_draft_sampled.

Gathers: `gather(site, inps [W] x [rows, WS] fp32, outs [W] x contiguous [W, rows, WS] fp32, streams)` -> every rank
gets every rank's words (site names the collective instance: its own flag / epoch in the peer-store version). WS = 4 (verify)
or 8 (draft) so each row is a 16-B multiple for the peer-store all-gather (comm.cu).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable, gumbel_noised_argmax, murmur3_uniform32, tl_rand32

VERIFY_BLOCK = 2048
DRAFT_BLOCK = 128
RANK_CHUNK = 32
WS_V = 4  # words per verify row (3 used)
WS_D = 8  # words per draft row (5 used)


# ============================================================================================ shared jit helpers
@triton.jit
def _last_program(counts_ptr, row, NB: tl.constexpr):
    # bar.sync + acq_rel RMW publishes this program's partial stores; the last arriving program merges all partials
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
def _div_t(x, t):
    # x / t, skipped at t == 1 (x / 1.0 == x exactly in IEEE fp32, incl. +-inf / NaN): real traffic samples at 1.0, and
    # the fp32 division of every logit is a large share of the sampler passes' ALU work
    if t != 1.0:
        x = x / t
    return x


@triton.jit
def _nan_to_neg_inf(v):
    return tl.where(v != v, float("-inf"), v)


@triton.jit
def _allowed(bm_ptr, bm_stride, row, offs, mask, HAS_MASK: tl.constexpr):
    # stock _apply_grammar_bitmask_kernel: bit (token % 32) of word token // 32; 0 = disallowed. offs are GLOBAL ids.
    if HAS_MASK:
        words = tl.load(bm_ptr + row * bm_stride + (offs >> 5), mask=mask, other=-1)
        return ((words >> (offs & 31)) & 1) != 0
    else:
        return mask


# ============================================================================================ verification
@triton.jit
def _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, count, temp, TP: tl.constexpr,
                 S: tl.constexpr, WS: tl.constexpr):
    # per drafted position s < count: log p_target(x_{s+1}) - log q_draft(x_{s+1})
    lanes = tl.arange(0, S)
    live = lanes < count
    q = g1_ptr + (tl.arange(0, TP)[:, None] * num_logits + start + lanes[None, :]) * WS
    m = tl.load(q, mask=live[None, :], other=float("-inf"))
    s = tl.load(q + 1, mask=live[None, :], other=0.0)
    lse = _logsumexp(m, s)
    owned = tl.max(tl.load(q + 2, mask=live[None, :], other=float("-inf")), axis=0)
    d_lp = tl.load(vp_logit_ptr + req * vp_stride + lanes, mask=live, other=0.0) / temp - tl.load(
        vp_lse_ptr + req * vp_stride + lanes, mask=live, other=0.0)
    return owned - lse - d_lp, m, s


@triton.jit
def _stats_kernel(out_ptr, part_ptr, counts_ptr, logits_ptr, logits_stride, draft_ptr, num_logits, row_req_ptr, temp_ptr,
                  bm_ptr, bm_stride, local_vocab, vocab_start,
                  BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr, HAS_MASK: tl.constexpr):
    # (max of logits/T, sum exp(logits/T - max)) or greedy (max, argmax token); logit/T of the next row's draft token if
    # this rank owns it (else -inf). = production _target_stats_kernel (stock _compute_local_logits_stats_kernel) + mask.
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
    temp = tl.load(temp_ptr + tl.load(row_req_ptr + row)).to(tl.float32)
    t = tl.where(temp > 0.0, temp, 1.0)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    x = tl.where(ok, x, float("-inf")) / t
    if temp == 0.0:
        m, i = tl.max(x, axis=0, return_indices=True)
        s = (vocab_start + b * BLOCK + i).to(tl.float32)
    else:
        m, s = _block_stats(x)
    d = tl.load(draft_ptr + row + 1, mask=row + 1 < num_logits, other=-1) - vocab_start - b * BLOCK
    dmask = (d >= 0) & (d < BLOCK) & (b * BLOCK + d < local_vocab)
    owned = tl.load(logits_ptr + row * logits_stride + b * BLOCK + d, mask=dmask, other=float("-inf")).to(tl.float32)
    if HAS_MASK:
        dok = _allowed(bm_ptr, bm_stride, row, vocab_start + b * BLOCK + d, dmask, HAS_MASK)
        owned = tl.where(dok, owned, float("-inf"))
    owned = owned / t
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, m)
    tl.store(p + 1, s)
    tl.store(p + 2, owned)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        ms = tl.load(q, volatile=True)
        ss = tl.load(q + 1, volatile=True)
        o = out_ptr + row * WS
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
def _resid_kernel(out_ptr, part_ptr, counts_ptr, g1_ptr, logits_ptr, logits_stride, cache_ptr, cache_stride_0,
                  cache_stride_1, vp_logit_ptr, vp_lse_ptr, vp_stride, draft_ptr, num_logits, row_req_ptr, row_local_ptr,
                  pos_ptr, temp_ptr, seeds_ptr, bm_ptr, bm_stride, local_vocab, vocab_start, num_spec,
                  BLOCK: tl.constexpr, NB: tl.constexpr, TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr,
                  HAS_MASK: tl.constexpr):
    # residual-mass partial sum_x max(p_{<row} M_b(x) - M_s(x), 0) on the slice and the Gumbel argmax (value, token) of
    # the residual distribution this row would be resampled from (production _target_residual_kernel + mask).
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    step = tl.load(row_local_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
    is_bonus = (row + 1 >= num_logits) | (next_pos == 0)
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
    lanes = tl.arange(0, S)
    terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP, S, WS)
    drafted = tl.load(draft_ptr + start + lanes + 1, mask=lanes < step, other=-1)
    log_p = 0.0
    for s in tl.static_range(S):
        take = (temp != 0.0) & (s < step) & (_pick(drafted, lanes, s) >= 0)
        log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    x = tl.where(ok, x, float("-inf")) / tl.where(temp > 0.0, temp, 1.0)
    stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
    lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
    use_draft = (temp != 0.0) & not_bonus & (rejected >= 0)
    t_lp = x - lse
    d_lp = tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask & use_draft,
                   other=float("-inf")).to(tl.float32) / temp - tl.load(vp_lse_ptr + req * vp_stride + step,
                                                                         mask=use_draft, other=0.0)
    mass = 0.0
    if use_draft & (step >= 1) & (step < num_spec):
        mass = tl.sum(tl.maximum(tl.exp(log_p) * tl.exp(t_lp) - tl.exp(d_lp), 0.0), axis=0)
    value = float("-inf")
    token = vocab_start + b * BLOCK
    if (temp != 0.0) | is_bonus:
        if use_draft:
            shifted = t_lp + log_p
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x
        value, i = gumbel_noised_argmax(residual, vocab_start + offs, mask, seed, tl.load(pos_ptr + row), temp,
                                        IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False)
        token = vocab_start + b * BLOCK + i
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, mass)
    tl.store(p + 1, value)
    tl.store(p + 2, token.to(tl.float32))
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1, volatile=True))
        o = out_ptr + row * WS
        tl.store(o, tl.sum(tl.load(q, volatile=True), axis=0))
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))


@triton.jit
def _reject_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, last_ptr, g1_ptr, g2_ptr, vp_logit_ptr, vp_lse_ptr,
                   vp_stride, draft_ptr, num_logits, cu_ptr, idx_ptr, pos_ptr, temp_ptr, seeds_ptr,
                   TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr):
    # stock _rejection_kernel(USE_BLOCK_VERIFICATION) + _insert_resampled_kernel (production _target_reject_kernel);
    # also writes the last sampled token per request (the next draft step's anchor).
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
    drafted = tl.load(draft_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    terms, m, s = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp, TP,
                               S, WS)
    best = _pick(s, ranks[:, None], tl.argmax(_nan_to_neg_inf(m), axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * WS,
                          mask=(lanes < num_draft - 1)[None, :], other=0.0), axis=0)
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
                    okk = (top == token) & valid
                    verifying = okk
                    accepted += okk
                    tl.store(sampled_ptr + r * sampled_stride + i, tl.where(okk, token, top))
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
    tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))
    row = start + accepted
    last = _pick(best, lanes, accepted)  # greedy mismatch: the target top (already stored above)
    if (temp != 0.0) | (row == end - 1):
        q = g2_ptr + (ranks * num_logits + row) * WS
        vals = _nan_to_neg_inf(tl.load(q + 1))
        last = _pick(tl.load(q + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + accepted, last)
    tl.store(last_ptr + r, last)


# ---------------------------------------------------------------------------------------- v2 verification (high load)
# Production's k_resid computes the residual Gumbel argmax for EVERY row (murmur3 + 2 logs per element: ALU-bound,
# 127 us at T = 456 on one rank's slice) although only ONE row per request is resampled. v2 splits it:
#   k_mass     residual mass only, only on the rows that need it (1 <= local < k, temp != 0, valid next draft)
#   exchange 2 (mass words)
#   k_decide   per request: accepted length tau (exactly the block-verification loop of k_reject), drafts -> sampled,
#              greedy: the target top at tau; tau_row = start + tau
#   k_resample per request: the residual (or target) distribution of row tau_row on the slice, Gumbel argmax
#   exchange 3 (value, token)
#   k_finish   the global winner -> sampled[tau], last
# Same formulas, same RNG keys, same rows as production -> the same tokens (bit-identical decisions).
@triton.jit
def _mass_kernel(out_ptr, part_ptr, counts_ptr, g1_ptr, logits_ptr, logits_stride, cache_ptr, cache_stride_0,
                 cache_stride_1, vp_logit_ptr, vp_lse_ptr, vp_stride, draft_ptr, num_logits, row_req_ptr, row_local_ptr,
                 temp_ptr, bm_ptr, bm_stride, local_vocab, vocab_start, num_spec,
                 BLOCK: tl.constexpr, NB: tl.constexpr, TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr,
                 HAS_MASK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    step = tl.load(row_local_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
    need = (temp != 0.0) & not_bonus & (rejected >= 0) & (step >= 1) & (step < num_spec)
    mass = 0.0
    if need:
        offs = b * BLOCK + tl.arange(0, BLOCK)
        mask = offs < local_vocab
        ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
        lanes = tl.arange(0, S)
        terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP,
                                   S, WS)
        drafted = tl.load(draft_ptr + start + lanes + 1, mask=lanes < step, other=-1)
        log_p = 0.0
        for s in tl.static_range(S):
            take = (s < step) & (_pick(drafted, lanes, s) >= 0)
            log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = _div_t(tl.where(ok, x, float("-inf")), temp)
        stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
        lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
        t_lp = x - lse
        d_lp = _div_t(tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask,
                              other=float("-inf")).to(tl.float32), temp) - tl.load(vp_lse_ptr + req * vp_stride + step)
        mass = tl.sum(tl.maximum(tl.exp(log_p) * tl.exp(t_lp) - tl.exp(d_lp), 0.0), axis=0)
    tl.store(part_ptr + (row * NB + b) * 3, mass)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        tl.store(out_ptr + row * WS, tl.sum(tl.load(part_ptr + (row * NB + blocks) * 3, volatile=True), axis=0))


@triton.jit
def _decide_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, tau_ptr, g1_ptr, g2_ptr, vp_logit_ptr, vp_lse_ptr,
                   vp_stride, draft_ptr, num_logits, cs_ptr, ce_ptr, idx_ptr, pos_ptr, temp_ptr, seeds_ptr,
                   TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr):
    # cs / ce: first / end row of each request (= cu[r], cu[r + 1] when the requests tile the rows; hsd.py pads each
    # rank's rows for the all-gather, so its requests are not contiguous over the global rows)
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    start = tl.load(cs_ptr + r).to(tl.int64)
    end = tl.load(ce_ptr + r).to(tl.int64)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    greedy = temp == 0.0
    lanes = tl.arange(0, S)
    ranks = tl.arange(0, TP)
    live = lanes < num_draft
    drafted = tl.load(draft_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    terms, m, s = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp, TP,
                               S, WS)
    # greedy: the target argmax (stats word 1 = global id) of EVERY verified row incl. the bonus row (lane num_draft),
    # from the rank with the row's max. _ratio_terms' m / s cover only the num_draft drafted rows: the bonus lane must be
    # loaded here (otherwise it reads as token 0 = BOS for finishing prefills and fully accepted steps)
    bq = g1_ptr + (ranks[:, None] * num_logits + start + lanes[None, :]) * WS
    blive = (lanes < num_draft + 1)[None, :]
    bm = _nan_to_neg_inf(tl.load(bq, mask=blive, other=float("-inf")))
    best = _pick(tl.load(bq + 1, mask=blive, other=0.0), ranks[:, None], tl.argmax(bm, axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * WS,
                          mask=(lanes < num_draft - 1)[None, :], other=0.0), axis=0)
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
                    okk = (top == token) & valid
                    verifying = okk
                    accepted += okk
                    tl.store(sampled_ptr + r * sampled_stride + i, tl.where(okk, token, top))
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
    tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))
    if greedy & (start + accepted == end - 1):
        # greedy bonus row: the argmax of the target row (= the temp-0 Gumbel argmax stock takes from the residual pass)
        tl.store(sampled_ptr + r * sampled_stride + accepted, _pick(best, lanes, accepted))
    tl.store(tau_ptr + r, (start + accepted).to(tl.int32))


@triton.jit
def _resample_kernel(out_ptr, part_ptr, counts_ptr, tau_ptr, g1_ptr, logits_ptr, logits_stride, cache_ptr,
                     cache_stride_0, cache_stride_1, vp_logit_ptr, vp_lse_ptr, vp_stride, draft_ptr, num_logits,
                     row_req_ptr, row_local_ptr, pos_ptr, temp_ptr, seeds_ptr, bm_ptr, bm_stride, local_vocab,
                     vocab_start, BLOCK: tl.constexpr, NB: tl.constexpr, TP: tl.constexpr, S: tl.constexpr,
                     WS: tl.constexpr, HAS_MASK: tl.constexpr):
    # = the per-row Gumbel-argmax half of production's k_resid, evaluated only at row tau_row[r]
    r = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    row = tl.load(tau_ptr + r).to(tl.int64)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(row_req_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    value = float("-inf")
    token = vocab_start + b * BLOCK
    if temp != 0.0:
        ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
        seed = tl.load(seeds_ptr + req)
        step = tl.load(row_local_ptr + row).to(tl.int64)
        start = row - step
        has_next = row + 1 < num_logits
        next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
        not_bonus = has_next & (next_pos != 0)
        rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
        use_draft = not_bonus & (rejected >= 0)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = _div_t(tl.where(ok, x, float("-inf")), temp)
        if use_draft:
            lanes = tl.arange(0, S)
            terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp,
                                       TP, S, WS)
            drafted = tl.load(draft_ptr + start + lanes + 1, mask=lanes < step, other=-1)
            log_p = 0.0
            for s in tl.static_range(S):
                take = (s < step) & (_pick(drafted, lanes, s) >= 0)
                log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
            stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
            lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
            t_lp = x - lse
            d_lp = _div_t(tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask,
                                  other=float("-inf")).to(tl.float32), temp) - tl.load(vp_lse_ptr + req * vp_stride + step)
            shifted = t_lp + log_p
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x
        value, i = gumbel_noised_argmax(residual, vocab_start + offs, mask, seed, tl.load(pos_ptr + row), temp,
                                        IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False)
        token = vocab_start + b * BLOCK + i
    p = part_ptr + (r * NB + b) * 3
    tl.store(p + 1, value)
    tl.store(p + 2, token.to(tl.float32))
    if _last_program(counts_ptr, r, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (r * NB + blocks) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1, volatile=True))
        o = out_ptr + r * WS
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))


@triton.jit
def _finish_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, last_ptr, g3_ptr, nreq, idx_ptr, temp_ptr,
                   TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_ptr + r).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    ns = tl.load(num_sampled_ptr + r).to(tl.int64)
    ranks = tl.arange(0, TP)
    last = tl.load(sampled_ptr + r * sampled_stride + ns - 1)  # greedy: written by k_decide
    if temp != 0.0:
        q = g3_ptr + (ranks * nreq + r) * WS
        vals = _nan_to_neg_inf(tl.load(q + 1))
        last = _pick(tl.load(q + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + ns - 1, last)
    tl.store(last_ptr + r, last)


@triton.jit
def _draft2_kernel(out_ptr, part_ptr, counts_ptr, base_ptr, base_stride, bias_ptr, bias_stride, cache_ptr,
                   cache_stride_0, cache_stride_1, col, idx_ptr, idx_stride, pos_ptr, pos_stride, seeds_ptr, temp_ptr,
                   local_vocab, vocab_start, BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr):
    # production _draft_kernel with the Markov bias precomputed by one cuBLAS GEMM (bias = bf16(emb @ w2_slice^T), stock's
    # F.linear): at 100+ rows the per-row GEMV re-read the 16.5 MB w2 slice once per row (284 us per draft step)
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    base = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    bias = tl.load(bias_ptr + row * bias_stride + offs, mask=mask, other=0.0).to(tl.float32)
    logits = (base + bias).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    logits = tl.where(mask, logits, float("-inf"))
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    tl.store(cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs, logits.to(cache_ptr.dtype.element_ty),
             mask=mask & valid)
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + row * pos_stride) - 1
    value, idx = gumbel_noised_argmax(logits, vocab_start + offs, mask, seed, pos, temp, IS_DRAFTING=True,
                                      USE_FP64=False, APPLY_TEMPERATURE=True)
    t = tl.where(temp > 0.0, temp, 1.0)
    m, s = _block_stats(tl.where(mask, _div_t(logits, t), float("-inf")))
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
        o = out_ptr + row * WS
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


# ---------------------------------------------------------------------------------------- v3: row-loop passes
# Same per-block (BLOCK = 2048) partials and the same final reductions as v1/v2 (so the same bits), but one program
# per row loops over the NB blocks and keeps the partials in registers: no per-block programs, atomics or partial
# round trips (v2 stats: 7,296 programs of 4 KB each at T = 456).
@triton.jit
def _stats_rows_kernel(out_ptr, logits_ptr, logits_stride, draft_ptr, num_logits, row_req_ptr, temp_ptr, bm_ptr,
                       bm_stride, local_vocab, vocab_start, BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr,
                       HAS_MASK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    temp = tl.load(temp_ptr + tl.load(row_req_ptr + row)).to(tl.float32)
    t = tl.where(temp > 0.0, temp, 1.0)
    blocks = tl.arange(0, NB)
    ms = tl.full((NB,), float("-inf"), tl.float32)
    ss = tl.zeros((NB,), tl.float32)
    ow = tl.full((NB,), float("-inf"), tl.float32)
    dtok = tl.load(draft_ptr + row + 1, mask=row + 1 < num_logits, other=-1)
    for b in tl.static_range(NB):
        offs = b * BLOCK + tl.arange(0, BLOCK)
        mask = offs < local_vocab
        ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = _div_t(tl.where(ok, x, float("-inf")), t)
        if temp == 0.0:
            m, i = tl.max(x, axis=0, return_indices=True)
            sv = (vocab_start + b * BLOCK + i).to(tl.float32)
        else:
            m, sv = _block_stats(x)
        d = dtok - vocab_start - b * BLOCK
        dmask = (d >= 0) & (d < BLOCK) & (b * BLOCK + d < local_vocab)
        owned = tl.load(logits_ptr + row * logits_stride + b * BLOCK + d, mask=dmask, other=float("-inf")).to(tl.float32)
        if HAS_MASK:
            dok = _allowed(bm_ptr, bm_stride, row, vocab_start + b * BLOCK + d, dmask, HAS_MASK)
            owned = tl.where(dok, owned, float("-inf"))
        ms = tl.where(blocks == b, m, ms)
        ss = tl.where(blocks == b, sv, ss)
        ow = tl.where(blocks == b, owned / t, ow)
    o = out_ptr + row * WS
    if temp == 0.0:
        clean = _nan_to_neg_inf(ms)
        tl.store(o, tl.max(clean, axis=0))
        tl.store(o + 1, _pick(ss, blocks, tl.argmax(clean, axis=0)))
    else:
        top, total = _merge(ms, ss)
        tl.store(o, top)
        tl.store(o + 1, total)
    tl.store(o + 2, tl.max(ow, axis=0))


@triton.jit
def _mass_rows_kernel(out_ptr, g1_ptr, logits_ptr, logits_stride, cache_ptr, cache_stride_0, cache_stride_1, vp_logit_ptr,
                      vp_lse_ptr, vp_stride, draft_ptr, num_logits, row_req_ptr, row_local_ptr, temp_ptr, bm_ptr,
                      bm_stride, local_vocab, vocab_start, num_spec, BLOCK: tl.constexpr, NB: tl.constexpr,
                      TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr, HAS_MASK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    step = tl.load(row_local_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
    need = (temp != 0.0) & not_bonus & (rejected >= 0) & (step >= 1) & (step < num_spec)
    blocks = tl.arange(0, NB)
    parts = tl.zeros((NB,), tl.float32)
    if need:
        lanes = tl.arange(0, S)
        terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP,
                                   S, WS)
        drafted = tl.load(draft_ptr + start + lanes + 1, mask=lanes < step, other=-1)
        log_p = 0.0
        for s in tl.static_range(S):
            take = (s < step) & (_pick(drafted, lanes, s) >= 0)
            log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
        stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
        lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
        d_lse = tl.load(vp_lse_ptr + req * vp_stride + step)
        p_tau = tl.exp(log_p)
        for b in tl.static_range(NB):
            offs = b * BLOCK + tl.arange(0, BLOCK)
            mask = offs < local_vocab
            ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
            x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
            x = _div_t(tl.where(ok, x, float("-inf")), temp)
            d_lp = _div_t(tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask,
                                  other=float("-inf")).to(tl.float32), temp) - d_lse
            mb = tl.sum(tl.maximum(p_tau * tl.exp(x - lse) - tl.exp(d_lp), 0.0), axis=0)
            parts = tl.where(blocks == b, mb, parts)
    tl.store(out_ptr + row * WS, tl.sum(parts, axis=0))


def verify3_stats(st: "RankSampler", logits: torch.Tensor, m: "VerifyMeta") -> torch.Tensor:
    T = logits.shape[0]
    bm = m.bitmask
    _stats_rows_kernel[(T,)](
        st.stats, logits, logits.stride(0), m.draft_sampled, T, m.row_req, st.temp,
        bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start,
        BLOCK=VERIFY_BLOCK, NB=st.vblocks, WS=WS_V, HAS_MASK=bm is not None, num_warps=8)
    return st.stats[:T]


def verify3_mass(st: "RankSampler", logits: torch.Tensor, g1: torch.Tensor, m: "VerifyMeta") -> torch.Tensor:
    T = logits.shape[0]
    bm = m.bitmask
    _mass_rows_kernel[(T,)](
        st.resid, g1, logits, logits.stride(0), st.cache, st.cache.stride(0), st.cache.stride(1), st.token_logit, st.lse,
        st.lse.stride(0), m.draft_sampled, T, m.row_req, m.row_local, st.temp,
        bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start, st.n_spec,
        BLOCK=VERIFY_BLOCK, NB=st.vblocks, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V,
        HAS_MASK=bm is not None, num_warps=8)
    return st.resid[:T]


# ---------------------------------------------------------------------------------------- draft noise ahead of time
# The Gumbel noise of draft slot i depends only on (seed, P_i - 1 + 2^30, global token id): k_draft_noise computes all k
# slots on a side stream while the DSpark blocks run (positions are known once the verification is done), and
# k_draft3 adds it exactly like gumbel_noised_argmax does (logits / T + noise in fp32) -> the same draws, the same bits.
@triton.jit
def _draft_noise_kernel(noise_ptr, noise_stride_slot, noise_stride_row, idx_ptr, idx_stride, pos_ptr, pos_stride, seeds_ptr,
                        temp_ptr, col_stride, local_vocab, vocab_start, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(idx_ptr + row * idx_stride + col * col_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    if valid & (temp != 0.0):
        seed = tl.load(seeds_ptr + req)
        pos = tl.load(pos_ptr + row * pos_stride + col * col_stride) - 1 + (1 << 30)
        u = murmur3_uniform32(seed, pos, vocab_start + offs)
        nz = -tl.log(-_log1p_neg_stable(u))
        tl.store(noise_ptr + col * noise_stride_slot + row * noise_stride_row + offs, nz, mask=mask)


@triton.jit
def _draft3_kernel(out_ptr, part_ptr, counts_ptr, base_ptr, base_stride, bias_ptr, bias_stride, noise_ptr, noise_stride,
                   cache_ptr, cache_stride_0, cache_stride_1, col, idx_ptr, idx_stride, temp_ptr, local_vocab, vocab_start,
                   BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    base = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    bias = tl.load(bias_ptr + row * bias_stride + offs, mask=mask, other=0.0).to(tl.float32)
    logits = (base + bias).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    logits = tl.where(mask, logits, float("-inf"))
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    tl.store(cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs, logits.to(cache_ptr.dtype.element_ty),
             mask=mask & valid)
    # = gumbel_noised_argmax(logits, ids, mask, seed, pos, temp, IS_DRAFTING=True, APPLY_TEMPERATURE=True)
    x = logits
    if temp != 0.0:
        x = _div_t(x, temp)
        nz = tl.load(noise_ptr + row * noise_stride + offs, mask=mask, other=0.0)
        x = tl.where(mask, x + nz, float("-inf"))
    value, idx = tl.max(x, axis=0, return_indices=True)
    t = tl.where(temp > 0.0, temp, 1.0)
    m, s = _block_stats(tl.where(mask, _div_t(logits, t), float("-inf")))
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
        o = out_ptr + row * WS
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


def draft_noise(st: "RankSampler", noise: torch.Tensor, m: "DraftMeta", block: int = 2048) -> None:
    """noise [k, >= n, local] fp32: the Gumbel noise of every draft slot (side stream, after the verification)."""
    n, k = m.num_reqs, st.n_spec
    nb = triton.cdiv(st.local, block)
    _draft_noise_kernel[(n, k, nb)](
        noise, noise.stride(0), noise.stride(1), m.idx, m.idx.stride(0), m.sample_pos, m.sample_pos.stride(0), st.seeds,
        st.temp, m.idx.stride(1), st.local, st.start, BLOCK=block, num_warps=4)


def draft3_local(st: "RankSampler", base: torch.Tensor, emb: torch.Tensor, idx: torch.Tensor, noise: torch.Tensor, col: int,
                 block: int = 2048, bias_ready: bool = False) -> torch.Tensor:
    """bias_ready: st.bias[:rows] already holds bf16(emb @ w2^T) (e.g. slot 0's, computed on a side stream)."""
    rows = base.shape[0]
    bias = st.bias[:rows]
    if not bias_ready:
        torch.mm(emb, st.w2.t(), out=bias)
    nb = triton.next_power_of_2(triton.cdiv(st.local, block))
    _draft3_kernel[(rows, nb)](
        st.d_out, st.d_part, st.d_counts, base, base.stride(0), bias, bias.stride(0), noise, noise.stride(0), st.cache,
        st.cache.stride(0), st.cache.stride(1), col, idx, idx.stride(0), st.temp, st.local, st.start, BLOCK=block, NB=nb,
        WS=WS_D, num_warps=4)
    return st.d_out[:rows]


# ============================================================================================ drafting
@triton.jit
def _draft_kernel(out_ptr, part_ptr, counts_ptr, base_ptr, base_stride, emb_ptr, emb_stride, w2_ptr, cache_ptr,
                  cache_stride_0, cache_stride_1, col, idx_ptr, idx_stride, pos_ptr, pos_stride, seeds_ptr, temp_ptr,
                  local_vocab, vocab_start, RANK: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr,
                  ROUND_BF16: tl.constexpr):
    # one sequential Markov step for one request row (production _draft_kernel, stock speculator.py:178-199):
    # bias = bf16(emb . w2[token]); logits = bf16(base + bias); cache; Gumbel argmax keyed by (seed, P-1+salt, token)
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in tl.range(0, RANK, CHUNK):
        k = k0 + tl.arange(0, CHUNK)
        w = tl.load(w2_ptr + offs[:, None].to(tl.int64) * RANK + k[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        e = tl.load(emb_ptr + row * emb_stride + k).to(tl.float32)
        acc += tl.sum(w * e[None, :], axis=1)
    base = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    if ROUND_BF16:
        bias = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        logits = (base + bias).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    else:
        logits = base + acc
    logits = tl.where(mask, logits, float("-inf"))
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    tl.store(cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs, logits.to(cache_ptr.dtype.element_ty),
             mask=mask & valid)
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + row * pos_stride) - 1
    value, idx = gumbel_noised_argmax(logits, vocab_start + offs, mask, seed, pos, temp, IS_DRAFTING=True,
                                      USE_FP64=False, APPLY_TEMPERATURE=True)
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
        o = out_ptr + row * WS
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


@triton.jit
def _draft_combine_kernel(gathered_ptr, rows, tokens_ptr, tokens_stride, col, emb_ptr, emb_stride, w1_ptr, idx_ptr,
                          idx_stride, token_logit_ptr, lse_ptr, vp_stride, sink, TP: tl.constexpr, RANK: tl.constexpr,
                          WS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, TP)
    q = gathered_ptr + (ranks * rows + row) * WS
    win = tl.argmax(tl.load(q), axis=0)  # lowest rank wins ties == lowest global token id (stock first max)
    token = _pick(tl.load(q + 1), ranks, win).to(tl.int64)
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    slot = tl.where(req >= 0, req, sink)
    tl.store(tokens_ptr + row * tokens_stride + col, token)
    tl.store(token_logit_ptr + slot * vp_stride + col, _pick(tl.load(q + 2), ranks, win))
    tl.store(lse_ptr + slot * vp_stride + col, _logsumexp(tl.load(q + 3), tl.load(q + 4)))
    r = tl.arange(0, RANK)
    tl.store(emb_ptr + row * emb_stride + r, tl.load(w1_ptr + token * RANK + r))


# ============================================================================================ per-rank state
@dataclass
class RankSampler:
    """Per-rank buffers of the vocab-parallel sampler (one per GPU; all shapes static -> graph-capturable)."""
    rank: int
    ranks: int
    start: int
    local: int
    vocab: int
    max_reqs: int
    n_spec: int
    max_rows: int
    w1: torch.Tensor  # [V, 256] bf16 replicated Markov embedding
    w2: torch.Tensor  # [local, 256] bf16 slice of the Markov head
    cache: torch.Tensor  # [max_reqs, n_spec, local] bf16 cached pre-temperature draft logits (by request-state index)
    token_logit: torch.Tensor  # [max_reqs + 1, n_spec] fp32 raw draft logit of each drafted token (replicated)
    lse: torch.Tensor  # [max_reqs + 1, n_spec] fp32 log-sum-exp of draft logits / T (replicated)
    temp: torch.Tensor  # [max_reqs] fp32
    seeds: torch.Tensor  # [max_reqs] int64
    v_counts: torch.Tensor
    v_part: torch.Tensor
    stats: torch.Tensor  # [max_rows, WS_V]
    resid: torch.Tensor  # [max_rows, WS_V]
    g1f: torch.Tensor  # flat [W * max_rows * WS_V]; viewed [W, T, WS_V] per step (gather layout)
    g2f: torch.Tensor
    d_counts: torch.Tensor
    d_part: torch.Tensor
    d_out: torch.Tensor  # [max_reqs, WS_D]
    d_gathf: torch.Tensor  # flat [n_spec, W * max_reqs * WS_D]: ONE gather buffer per draft slot (see d_gath)
    emb: torch.Tensor  # [2, max_reqs, 256]
    sampled: torch.Tensor  # [max_reqs, n_spec + 1] int64
    num_sampled: torch.Tensor  # [max_reqs] int32
    last: torch.Tensor  # [max_reqs] int64 (anchor of the next draft)
    draft_tokens: torch.Tensor  # [max_reqs, n_spec] int64
    tau: torch.Tensor = None  # [max_reqs] int32 row of the resampled / bonus token (v2)
    out3: torch.Tensor = None  # [max_reqs, WS_V] (v2 resample words)
    g3f: torch.Tensor = None  # flat [W * max_reqs * WS_V]
    bias: torch.Tensor = None  # [max_reqs, local] bf16 Markov bias (v2 drafting)

    @classmethod
    def create(cls, rank, ranks, w1, w2_full, max_reqs, n_spec, device):
        vocab = w1.shape[0]
        local = vocab // ranks
        start = rank * local
        R = w1.shape[1]
        max_rows = max_reqs * (n_spec + 1)
        vb = triton.next_power_of_2(triton.cdiv(local, VERIFY_BLOCK))
        db = triton.next_power_of_2(triton.cdiv(local, DRAFT_BLOCK))
        f32 = dict(dtype=torch.float32, device=device)
        z = lambda *s, **k: torch.zeros(*s, **{**f32, **k})  # noqa: E731
        return cls(
            rank=rank, ranks=ranks, start=start, local=local, vocab=vocab, max_reqs=max_reqs, n_spec=n_spec,
            max_rows=max_rows, w1=w1.to(device).contiguous(), w2=w2_full[start:start + local].to(device).contiguous(),
            cache=torch.zeros(max_reqs, n_spec, local, dtype=torch.bfloat16, device=device),
            token_logit=z(max_reqs + 1, n_spec), lse=z(max_reqs + 1, n_spec), temp=z(max_reqs),
            seeds=torch.zeros(max_reqs, dtype=torch.int64, device=device),
            v_counts=torch.zeros(max_rows, dtype=torch.int32, device=device), v_part=z(max_rows, vb, 3),
            stats=z(max_rows, WS_V), resid=z(max_rows, WS_V), g1f=z(ranks * max_rows * WS_V), g2f=z(ranks * max_rows * WS_V),
            d_counts=torch.zeros(max_reqs, dtype=torch.int32, device=device), d_part=z(max_reqs, db, 5),
            d_out=z(max_reqs, WS_D), d_gathf=z(n_spec * ranks * max_reqs * WS_D),
            emb=torch.zeros(2, max_reqs, R, dtype=w1.dtype, device=device),
            sampled=torch.zeros(max_reqs, n_spec + 1, dtype=torch.int64, device=device),
            num_sampled=torch.zeros(max_reqs, dtype=torch.int32, device=device),
            last=torch.zeros(max_reqs, dtype=torch.int64, device=device),
            draft_tokens=torch.zeros(max_reqs, n_spec, dtype=torch.int64, device=device),
            tau=torch.zeros(max_reqs, dtype=torch.int32, device=device), out3=z(max_reqs, WS_V),
            g3f=z(ranks * max_reqs * WS_V),
            bias=torch.zeros(max_reqs, local, dtype=torch.bfloat16, device=device),
        )

    def g1(self, T):
        return self.g1f[: self.ranks * T * WS_V].view(self.ranks, T, WS_V)

    def g2(self, T):
        return self.g2f[: self.ranks * T * WS_V].view(self.ranks, T, WS_V)

    def g3(self, n):
        return self.g3f[: self.ranks * n * WS_V].view(self.ranks, n, WS_V)

    def d_gath(self, n, i=0):
        """Draft slot i's all-gather buffer [W, n, WS_D]. One per slot: with a shared buffer a fast peer pushes slot i + 1's
        words (it only waits for everyone's slot-i push) while a slow rank still reads slot i in draft_combine (the
        observed race: corrupted drafts / q in mixed steps)."""
        per = self.ranks * self.max_reqs * WS_D
        return self.d_gathf[i * per: i * per + self.ranks * n * WS_D].view(self.ranks, n, WS_D)

    @property
    def vblocks(self):
        return self.v_part.shape[1]

    @property
    def dblocks(self):
        return self.d_part.shape[1]


@dataclass
class VerifyMeta:
    """Per-step verification metadata (identical on every rank). Rows = the verified rows in step order."""
    draft_sampled: torch.Tensor  # [T] int64: input token of each row (row j >= 1 of a request = draft j); -1 = invalid
    pos: torch.Tensor  # [T] int64 position of each row
    row_req: torch.Tensor  # [T] int32 request-state index of each row
    row_local: torch.Tensor  # [T] int32 0..k within the request
    cu: torch.Tensor  # [reqs + 1] int32
    req_idx: torch.Tensor  # [reqs] int32 request-state index of each request
    num_reqs: int
    bitmask: torch.Tensor | None = None  # [T, ceil(V/32)] int32 (stock layout, global token ids) or None
    cu_end: torch.Tensor | None = None  # [reqs] end row of each request when the requests do not tile the rows (cu = starts)


@dataclass
class DraftMeta:
    idx: torch.Tensor  # [reqs, n_spec] int32 request-state index per (request, slot) (-1 = padding)
    sample_pos: torch.Tensor  # [reqs, n_spec] int64 predicted position P of each slot (draw keyed by P - 1)
    num_reqs: int


def verify_local_stats(st: RankSampler, logits: torch.Tensor, m: VerifyMeta, block: int = VERIFY_BLOCK,
                       warps: int = 8) -> torch.Tensor:
    T = logits.shape[0]
    bm = m.bitmask
    nb = triton.next_power_of_2(triton.cdiv(st.local, block))
    _stats_kernel[(T, nb)](
        st.stats, st.v_part, st.v_counts, logits, logits.stride(0), m.draft_sampled, T, m.row_req, st.temp,
        bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start,
        BLOCK=block, NB=nb, WS=WS_V, HAS_MASK=bm is not None, num_warps=warps)
    return st.stats[:T]


def verify_local_resid(st: RankSampler, logits: torch.Tensor, g1: torch.Tensor, m: VerifyMeta) -> torch.Tensor:
    T = logits.shape[0]
    bm = m.bitmask
    _resid_kernel[(T, st.vblocks)](
        st.resid, st.v_part, st.v_counts, g1, logits, logits.stride(0), st.cache, st.cache.stride(0), st.cache.stride(1),
        st.token_logit, st.lse, st.lse.stride(0), m.draft_sampled, T, m.row_req, m.row_local, m.pos, st.temp, st.seeds,
        bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start, st.n_spec,
        BLOCK=VERIFY_BLOCK, NB=st.vblocks, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V,
        HAS_MASK=bm is not None, num_warps=8)
    return st.resid[:T]


def verify_reject(st: RankSampler, g1: torch.Tensor, g2: torch.Tensor, m: VerifyMeta, T: int) -> None:
    _reject_kernel[(m.num_reqs,)](
        st.sampled, st.sampled.stride(0), st.num_sampled, st.last, g1, g2, st.token_logit, st.lse, st.lse.stride(0),
        m.draft_sampled, T, m.cu, m.req_idx, m.pos, st.temp, st.seeds,
        TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V, num_warps=1)


def draft_local(st: RankSampler, base: torch.Tensor, emb: torch.Tensor, idx: torch.Tensor, sample_pos: torch.Tensor,
                col: int) -> torch.Tensor:
    rows = base.shape[0]
    _draft_kernel[(rows, st.dblocks)](
        st.d_out, st.d_part, st.d_counts, base, base.stride(0), emb, emb.stride(0), st.w2, st.cache, st.cache.stride(0),
        st.cache.stride(1), col, idx, idx.stride(0), sample_pos, sample_pos.stride(0), st.seeds, st.temp, st.local, st.start,
        RANK=st.w1.shape[1], CHUNK=RANK_CHUNK, BLOCK=DRAFT_BLOCK, NB=st.dblocks, WS=WS_D,
        ROUND_BF16=st.cache.dtype == torch.bfloat16, num_warps=4)
    return st.d_out[:rows]


def draft_combine(st: RankSampler, gath: torch.Tensor, rows: int, col: int, emb_out: torch.Tensor,
                  idx: torch.Tensor) -> None:
    _draft_combine_kernel[(rows,)](
        gath, rows, st.draft_tokens, st.draft_tokens.stride(0), col, emb_out, emb_out.stride(0), st.w1, idx, idx.stride(0),
        st.token_logit, st.lse, st.lse.stride(0), st.max_reqs, TP=st.ranks, RANK=st.w1.shape[1], WS=WS_D, num_warps=1)


# ---------------------------------------------------------------------------------------- v2 wrappers
def verify2_mass(st: RankSampler, logits: torch.Tensor, g1: torch.Tensor, m: VerifyMeta) -> torch.Tensor:
    T = logits.shape[0]
    bm = m.bitmask
    _mass_kernel[(T, st.vblocks)](
        st.resid, st.v_part, st.v_counts, g1, logits, logits.stride(0), st.cache, st.cache.stride(0), st.cache.stride(1),
        st.token_logit, st.lse, st.lse.stride(0), m.draft_sampled, T, m.row_req, m.row_local, st.temp,
        bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start, st.n_spec,
        BLOCK=VERIFY_BLOCK, NB=st.vblocks, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V,
        HAS_MASK=bm is not None, num_warps=8)
    return st.resid[:T]


def verify2_decide(st: RankSampler, g1: torch.Tensor, g2: torch.Tensor, m: VerifyMeta, T: int) -> None:
    ce = m.cu_end if m.cu_end is not None else m.cu[1:]
    _decide_kernel[(m.num_reqs,)](
        st.sampled, st.sampled.stride(0), st.num_sampled, st.tau, g1, g2, st.token_logit, st.lse, st.lse.stride(0),
        m.draft_sampled, T, m.cu, ce, m.req_idx, m.pos, st.temp, st.seeds,
        TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V, num_warps=1)


def verify2_resample(st: RankSampler, logits: torch.Tensor, g1: torch.Tensor, m: VerifyMeta) -> torch.Tensor:
    T, n = logits.shape[0], m.num_reqs
    bm = m.bitmask
    _resample_kernel[(n, st.vblocks)](
        st.out3, st.v_part, st.v_counts, st.tau, g1, logits, logits.stride(0), st.cache, st.cache.stride(0),
        st.cache.stride(1), st.token_logit, st.lse, st.lse.stride(0), m.draft_sampled, T, m.row_req, m.row_local, m.pos,
        st.temp, st.seeds, bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start,
        BLOCK=VERIFY_BLOCK, NB=st.vblocks, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V,
        HAS_MASK=bm is not None, num_warps=8)
    return st.out3[:n]


def verify2_finish(st: RankSampler, g3: torch.Tensor, m: VerifyMeta) -> None:
    _finish_kernel[(m.num_reqs,)](st.sampled, st.sampled.stride(0), st.num_sampled, st.last, g3, m.num_reqs, m.req_idx,
                                  st.temp, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=WS_V, num_warps=1)


def draft2_local(st: RankSampler, base: torch.Tensor, emb: torch.Tensor, idx: torch.Tensor, sample_pos: torch.Tensor,
                 col: int, block: int = 1024) -> torch.Tensor:
    rows = base.shape[0]
    bias = st.bias[:rows]
    torch.mm(emb, st.w2.t(), out=bias)  # stock F.linear(emb, markov_w2) in bf16 (cuBLAS, fp32 accumulation)
    nb = triton.next_power_of_2(triton.cdiv(st.local, block))
    assert nb <= st.d_part.shape[1]
    _draft2_kernel[(rows, nb)](
        st.d_out, st.d_part, st.d_counts, base, base.stride(0), bias, bias.stride(0), st.cache, st.cache.stride(0),
        st.cache.stride(1), col, idx, idx.stride(0), sample_pos, sample_pos.stride(0), st.seeds, st.temp, st.local,
        st.start, BLOCK=block, NB=nb, WS=WS_D, num_warps=4)
    return st.d_out[:rows]


# ============================================================================================ multi-rank drivers
# One process drives the W devices (development harness, like kernels/sp_layer). Each stage is issued on every rank
# before the next stage, so a gather's push on all ranks precedes every rank's wait (no cross-device deadlock).
def verify_all(sts: list[RankSampler], logits: list[torch.Tensor], metas: list[VerifyMeta], gather, streams):
    """Block verification on every rank (rank r: its vocab slice of all T rows). Result in st.sampled / num_sampled /
    last ([:num_reqs], identical on every rank)."""
    W = len(sts)
    T = logits[0].shape[0]
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            verify_local_stats(sts[r], logits[r], metas[r])
    gather("g1", [st.stats[:T] for st in sts], [st.g1(T) for st in sts], streams)
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            verify_local_resid(sts[r], logits[r], sts[r].g1(T), metas[r])
    gather("g2", [st.resid[:T] for st in sts], [st.g2(T) for st in sts], streams)
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            verify_reject(sts[r], sts[r].g1(T), sts[r].g2(T), metas[r], T)


def draft_all(sts: list[RankSampler], base: list[torch.Tensor], anchors: list[torch.Tensor], metas: list[DraftMeta],
              gather, streams):
    """Sequential Markov sampling of all n_spec slots. base[r]: [reqs, n_spec, local] draft-head logits of rank r's
    slice; anchors[r]: [reqs] int64 anchor token per request (same on every rank). Result: st.draft_tokens[:reqs],
    the cache / token_logit / lse updated for the next verification."""
    W = len(sts)
    n = metas[0].num_reqs
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            torch.index_select(sts[r].w1, 0, anchors[r][:n], out=sts[r].emb[0, :n])
    for i in range(sts[0].n_spec):
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft_local(sts[r], base[r][:n, i], sts[r].emb[i % 2, :n], metas[r].idx[:n, i],
                            metas[r].sample_pos[:n, i], i)
        gather(f"d{i}", [st.d_out[:n] for st in sts], [st.d_gath(n, i) for st in sts], streams)
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft_combine(sts[r], sts[r].d_gath(n, i), n, i, sts[r].emb[(i + 1) % 2, :n], metas[r].idx[:n, i])


def verify_all2(sts, logits, metas, gather, streams):
    """v2 verification (mass-only pass, per-request resample): same results as verify_all."""
    W = len(sts)
    T = logits[0].shape[0]
    n = metas[0].num_reqs

    def each(fn):
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                fn(r)
    each(lambda r: verify_local_stats(sts[r], logits[r], metas[r]))
    gather("g1", [st.stats[:T] for st in sts], [st.g1(T) for st in sts], streams)
    each(lambda r: verify2_mass(sts[r], logits[r], sts[r].g1(T), metas[r]))
    gather("g2", [st.resid[:T] for st in sts], [st.g2(T) for st in sts], streams)
    each(lambda r: verify2_decide(sts[r], sts[r].g1(T), sts[r].g2(T), metas[r], T))
    each(lambda r: verify2_resample(sts[r], logits[r], sts[r].g1(T), metas[r]))
    gather("g3", [st.out3[:n] for st in sts], [st.g3(n) for st in sts], streams)
    each(lambda r: verify2_finish(sts[r], sts[r].g3(n), metas[r]))


def draft_all2(sts, base, anchors, metas, gather, streams, block: int = 1024):
    W = len(sts)
    n = metas[0].num_reqs
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            torch.index_select(sts[r].w1, 0, anchors[r][:n], out=sts[r].emb[0, :n])
    for i in range(sts[0].n_spec):
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft2_local(sts[r], base[r][:n, i], sts[r].emb[i % 2, :n], metas[r].idx[:n, i], metas[r].sample_pos[:n, i],
                             i, block)
        gather(f"d{i}", [st.d_out[:n] for st in sts], [st.d_gath(n, i) for st in sts], streams)
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft_combine(sts[r], sts[r].d_gath(n, i), n, i, sts[r].emb[(i + 1) % 2, :n], metas[r].idx[:n, i])


def verify_all3(sts, logits, metas, gather, streams):
    """v3 = v2 with the row-loop stats / mass passes (same bits as v1 / v2 / production)."""
    W = len(sts)
    T = logits[0].shape[0]
    n = metas[0].num_reqs

    def each(fn):
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                fn(r)
    each(lambda r: verify3_stats(sts[r], logits[r], metas[r]))
    gather("g1", [st.stats[:T] for st in sts], [st.g1(T) for st in sts], streams)
    each(lambda r: verify3_mass(sts[r], logits[r], sts[r].g1(T), metas[r]))
    gather("g2", [st.resid[:T] for st in sts], [st.g2(T) for st in sts], streams)
    each(lambda r: verify2_decide(sts[r], sts[r].g1(T), sts[r].g2(T), metas[r], T))
    each(lambda r: verify2_resample(sts[r], logits[r], sts[r].g1(T), metas[r]))
    gather("g3", [st.out3[:n] for st in sts], [st.g3(n) for st in sts], streams)
    each(lambda r: verify2_finish(sts[r], sts[r].g3(n), metas[r]))


def draft_all3(sts, base, anchors, metas, gather, streams, noise, block: int = 2048):
    """v3 drafting: noise precomputed (draft_noise), same draws as draft_all / draft_all2. noise[r]: [k, n, local] fp32."""
    W = len(sts)
    n = metas[0].num_reqs
    for r in range(W):
        with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
            draft_noise(sts[r], noise[r], metas[r], block)
            torch.index_select(sts[r].w1, 0, anchors[r][:n], out=sts[r].emb[0, :n])
    for i in range(sts[0].n_spec):
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft3_local(sts[r], base[r][:n, i], sts[r].emb[i % 2, :n], metas[r].idx[:n, i], noise[r][i], i, block)
        gather(f"d{i}", [st.d_out[:n] for st in sts], [st.d_gath(n, i) for st in sts], streams)
        for r in range(W):
            with torch.cuda.device(sts[r].w1.device), torch.cuda.stream(streams[r]):
                draft_combine(sts[r], sts[r].d_gath(n, i), n, i, sts[r].emb[(i + 1) % 2, :n], metas[r].idx[:n, i])
