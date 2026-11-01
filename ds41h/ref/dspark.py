"""Reference of the DSpark drafter forward and of stock vLLM 0.32.0's DSpark sampling math.

Drafter (official inference/model.py DSparkAttention/DSparkBlock/forward_spec, :1022-1159, and
vLLM models/deepseek_v41/nvidia/dspark.py):
  main_hidden = cat_l(mean_c X_in[l]) for l in (37, 38, 39) -- block *inputs* (after Engram);
  main_x      = main_norm(main_proj(main_hidden))            (mtp.0 only; FP8 [5120, 15360]);
  ring KV     = kv_store(RoPE(kv_norm(wkv(main_x)))) per block, positions <= p, window 128;
  draft ids   = [anchor, noise x (N-1)] (noise 128799), embedded and repeated to 4 hc copies;
  3 blocks: ratio-0 attention (theta 10000) where all N queries attend to the ring and to all N
              block KVs (bidirectional), MoE with 128 experts top-3 + shared expert, own mHC;
  head_hidden = hc_collapse(X, last ffn_pre); base logits = head(norm(head_hidden)) (tied head);
  sequential: logits_i = base_i + markov_head(markov_embed(prev)), prev = sample(logits_i).

Sampling (vLLM v1/worker/gpu/sample/gumbel.py, spec_decode/dspark/speculator.py:116-199,
spec_decode/rejection_sampler_utils.py): Gumbel-max sampling with noise keyed by
(seed, position, *global* token id) via murmur3, so a vocab-parallel implementation can be
bit-compatible with full-vocab sampling; block verification (Sun et al. 2024) with a Philox
uniform per verified row (Triton tl.rand). The integer parts (murmur3, Philox) are bit-exact;
float parts follow the Triton code operation by operation in fp32 (libdevice log vs torch log may
differ by an ulp, which only matters for exact Gumbel ties).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .layer import (
    CFG,
    W_ONLY,
    LayerWeights,
    ModelConfig,
    Numerics,
    _R,
    hc_collapse,
    hc_mixes,
    hc_post,
    lin_q,
    load_layer_weights,
    moe_ref,
    o_proj_ref,
    q_kv_proj_ref,
    rmsnorm,
    sparse_attn_ref,
    strict_fp32,
)
from .weights import CheckpointReader, dequant_fp8

# ===================================================================================== weights


@dataclass
class DSparkWeights:
    blocks: list[LayerWeights]  # mtp.0..2 (unsharded or one rank)
    main_proj: torch.Tensor  # [D, 3D] float32
    main_norm: torch.Tensor  # [D]
    norm: torch.Tensor  # [D] final norm (mtp.2.norm)
    markov_embed: torch.Tensor  # [V, 256] (bf16 values in float32)
    markov_head: torch.Tensor  # [V, 256]
    confidence: torch.Tensor | None  # [1, D + 256]


def load_dspark_weights(reader: CheckpointReader, rank: int | None = None, device="cpu",
                        cfg: ModelConfig | None = None, markov: bool = True) -> DSparkWeights:
    cfg = cfg or ModelConfig.from_hf(reader.config)
    blocks = [load_layer_weights(reader, cfg.n_layers + i, rank, device=device, cfg=cfg, mtp=i)
              for i in range(cfg.n_mtp_layers)]
    last = f"mtp.{cfg.n_mtp_layers - 1}."
    mp = dequant_fp8(reader.get("mtp.0.main_proj.weight").to(device), reader.get("mtp.0.main_proj.scale").to(device))
    f = lambda n: reader.get(n).to(device).float()  # noqa: E731
    me = f(last + "markov_head.embed.weight") if markov else torch.empty(0, device=device)
    mh = f(last + "markov_head.head.weight") if markov else torch.empty(0, device=device)
    conf = f(last + "confidence_head.proj.weight") if last + "confidence_head.proj.weight" in reader else None
    return DSparkWeights(blocks, mp, f("mtp.0.main_norm.weight"), f(last + "norm.weight"), me, mh, conf)


# ===================================================================================== drafter


def dspark_main_x_ref(DW: DSparkWeights, main_hidden: torch.Tensor, *, cfg: ModelConfig = CFG,
                      n: Numerics = W_ONLY) -> torch.Tensor:
    """main_x = main_norm(main_proj(main_hidden)) (DSparkBlock.forward_embed :1131-1138;
    vLLM combine_hidden_states). main_hidden [P, 3D] -> [P, D]."""
    return rmsnorm(lin_q(main_hidden, DW.main_proj, n), DW.main_norm, cfg.norm_eps, n)


def dspark_context_kv_ref(W: LayerWeights, main_x: torch.Tensor, positions: torch.Tensor, *,
                          cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> torch.Tensor:
    """Ring KV rows of one draft block for target positions `positions` (DSparkAttention :1042-1044;
    vLLM precompute_and_store_context_kv): stored kv_store(RoPE(kv_norm(wkv(main_x))))."""
    return q_kv_proj_ref(W, main_x, positions, cfg=cfg, n=n)["kv"]


def dspark_block_ref(W: LayerWeights, X: torch.Tensor, pre_mix: torch.Tensor | None, block_positions: torch.Tensor,
                     ring_kv: torch.Tensor, *, cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> dict:
    """One draft block (Block.forward with DSparkAttention :1056-1077) on the N block tokens.

    X [N, 4, D]; block_positions [N] (p+1 .. p+N); ring_kv [R, 512] stored context KV of the last
    R <= 128 target positions (all valid). Every query attends to all R ring rows and all N
    block KVs (bidirectional). Returns dict incl. x_out [N,4,D] and ffn_pre [N,4].
    """
    r = _R(n)
    hk = dict(norm_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, iters=cfg.hc_iters)
    out: dict = {}
    a_pre, a_post, a_comb, _ = hc_mixes(X, W.hc_attn_fn, W.hc_attn_scale, W.hc_attn_base, **hk)
    a = rmsnorm(hc_collapse(X, pre_mix, n), W.attn_norm, cfg.norm_eps, n)
    p = q_kv_proj_ref(W, a, block_positions, cfg=cfg, n=n)
    N = X.shape[0]
    keys = torch.cat([ring_kv.to(p["kv"].dtype), p["kv"]], 0)
    rows = keys.unsqueeze(0).expand(N, -1, -1)
    valid = torch.ones(rows.shape[:2], dtype=torch.bool, device=X.device)
    o = r.bf(sparse_attn_ref(p["q"], rows, W.attn_sink, valid, cfg.head_dim**-0.5))
    po = o_proj_ref(W, o, block_positions, cfg=cfg, n=n)
    x_mid = hc_post(po["y"], X, a_post, a_comb, n)
    f_pre, f_post, f_comb, _ = hc_mixes(x_mid, W.hc_ffn_fn, W.hc_ffn_scale, W.hc_ffn_base, **hk)
    f = rmsnorm(hc_collapse(x_mid, a_pre, n), W.ffn_norm, cfg.norm_eps, n)
    moe = moe_ref(f, W.gate_w, W.gate_bias, W.experts, (W.sh_w1, W.sh_w3, W.sh_w2), topk=cfg.dspark_n_activated,
                  route_scale=cfg.route_scale, swiglu_limit=cfg.swiglu_limit, n=n, return_all=True)
    out.update(a=a, q=p["q"], kv=p["kv"], o=o, attn_y=po["y"], x_mid=x_mid, f=f, moe=moe, attn_pre=a_pre,
               ffn_pre=f_pre, ffn_post=f_post, ffn_comb=f_comb,
               x_out=hc_post(moe["y"], x_mid, f_post, f_comb, n))
    return out


def dspark_forward_ref(DW: DSparkWeights, embed_rows: torch.Tensor, anchor_pos: int,
                       ring_kv: Sequence[torch.Tensor], *, cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> dict:
    """Drafter forward for one request (forward_spec :1277-1285 / vLLM DSparkDeepseekV4Model.forward).

    embed_rows [N, D]: embeddings of [anchor, noise, ...] (N = number of query slots, 5 for the
    trained block); anchor_pos: position of the anchor token (= p + 1, p = last target position);
    ring_kv: per block, [R, 512] stored context KV. Returns dict(head_hidden [N, D] (pre-norm,
    what vLLM's forward returns), blocks: list of block dicts, pre_mix).
    """
    with strict_fp32():
        N = embed_rows.shape[0]
        pos = torch.arange(anchor_pos, anchor_pos + N, device=embed_rows.device)
        X = _R(n).bf(embed_rows.float()).unsqueeze(1).repeat(1, cfg.hc_mult, 1)
        pre_mix, blocks = None, []
        for W, rk in zip(DW.blocks, ring_kv, strict=True):
            b = dspark_block_ref(W, X, pre_mix, pos, rk, cfg=cfg, n=n)
            blocks.append(b)
            X, pre_mix = b["x_out"], b["ffn_pre"]
        head_hidden = hc_collapse(X, pre_mix, n)
        return {"head_hidden": head_hidden, "blocks": blocks, "pre_mix": pre_mix, "positions": pos}


def dspark_logits_ref(head_hidden: torch.Tensor, norm_w: torch.Tensor, head_w: torch.Tensor,
                      cfg: ModelConfig = CFG, n: Numerics = W_ONLY) -> torch.Tensor:
    """Base draft logits = head(norm(head_hidden)) for a vocab slice head_w [V_slice, D]
    (ParallelHead is fp32 in model.py; vocab-parallel in vLLM)."""
    return rmsnorm(head_hidden, norm_w, cfg.norm_eps, n) @ head_w.float().t()


def markov_bias_ref(prev: torch.Tensor, markov_embed: torch.Tensor, markov_head: torch.Tensor):
    """(bias [B, V_slice], embed [B, 256]) for previous tokens (DSparkMarkovHead :1080-1089)."""
    e = markov_embed[prev.long()].float()
    return e @ markov_head.float().t(), e


def confidence_ref(head_hidden: torch.Tensor, markov_embed: torch.Tensor, proj: torch.Tensor, sigmoid: bool = True):
    """DSparkConfidenceHead (:1092-1100); vLLM applies a sigmoid (nvidia/dspark.py:345-350)."""
    c = torch.cat([head_hidden.float(), markov_embed.float()], -1) @ proj.float().t()
    c = c.squeeze(-1)
    return torch.sigmoid(c) if sigmoid else c


# ===================================================================================== RNG (bit-exact)

M32 = 0xFFFFFFFF
DRAFT_NOISE_SALT = 1 << 30  # gumbel.py:18
TL_RAND_MIN = np.float32(4.6566127342e-10)


def _mul32(a: torch.Tensor, b: int) -> torch.Tensor:
    """(a * b) mod 2**32 for int64 tensors holding uint32 values, overflow-free."""
    return ((a * (b & 0xFFFF)) + (((a * (b >> 16)) & 0xFFFF) << 16)) & M32


def _rotl32(v: torch.Tensor, s: int) -> torch.Tensor:
    return ((v << s) | (v >> (32 - s))) & M32


def _mmix(h: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    key = _mul32(key, 0xCC9E2D51)
    key = _rotl32(key, 15)
    key = _mul32(key, 0x1B873593)
    h = h ^ key
    h = _rotl32(h, 13)
    return (_mul32(h, 5) + 0xE6546B64) & M32


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    h = h ^ (h >> 16)
    h = _mul32(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = _mul32(h, 0xC2B2AE35)
    return h ^ (h >> 16)


def murmur3_hash32(seed: int, pos: int, offset: torch.Tensor, domain: int = 0) -> torch.Tensor:
    """gumbel.py murmur3_hash32 (:86-98); offset = token ids (uint32). Returns int64 in [0, 2**32)."""
    off = offset.long() & M32
    h = torch.full_like(off, domain & M32)
    for k in (seed & M32, (seed >> 32) & M32, pos & M32):
        h = _mmix(h, torch.full_like(off, k))
    h = _mmix(h, off)
    return _fmix32(h ^ 16)


def murmur3_uniform32(seed: int, pos: int, offset: torch.Tensor) -> torch.Tensor:
    """gumbel.py:101-114: fp32 hi16*2**-16 + (lo16+0.5)*2**-32 (both products exact)."""
    r = murmur3_hash32(seed, pos, offset)
    hi, lo = (r >> 16).to(torch.float32), (r & 0xFFFF).to(torch.float32)
    return hi * 1.52587890625e-05 + (lo + 0.5) * 2.3283064365386963e-10


def murmur3_uniform64(seed: int, pos: int, offset: torch.Tensor) -> torch.Tensor:
    lo = murmur3_hash32(seed, pos, offset)
    hi = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9)
    r53 = ((hi << 32) | lo) >> 11  # < 2**53, fits int64
    u = (r53.to(torch.float64) + 0.5) * 1.1102230246251565e-16
    return torch.where(u == 1.0, u - 1.1102230246251565e-16, u)


def _log1p_neg_stable(v: torch.Tensor) -> torch.Tensor:
    """gumbel.py:127-144 (fp32 Horner series below 0.25, direct log above)."""
    p = torch.full_like(v, 1.0 / 8.0)
    for c in (1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0):
        p = torch.tensor(c, dtype=torch.float32) + v * p
    series = -v * p
    direct = torch.log(torch.clamp(1.0 - v, min=5.960464477539063e-08))
    return torch.where(v < 0.25, series, direct)


def gumbel_noise(seed: int, pos: int, token_ids: torch.Tensor, *, is_drafting: bool = False,
                 use_fp64: bool = False) -> torch.Tensor:
    """The Gumbel noise vLLM adds to logit `token_ids` of a row (gumbel_noised_argmax :147-185)."""
    if is_drafting:
        pos = pos + DRAFT_NOISE_SALT
    if use_fp64:
        u = murmur3_uniform64(seed, pos, token_ids)
        return -torch.log(-torch.log(u))
    u = murmur3_uniform32(seed, pos, token_ids)
    return -torch.log(-_log1p_neg_stable(u))


def gumbel_sample_ref(logits: torch.Tensor, temperature: float, seed: int, pos: int, *, is_drafting: bool,
                      apply_temperature: bool = True, use_fp64: bool = False, vocab_offset: int = 0):
    """One row of vLLM gumbel_sample (gumbel.py:333-411). logits [V] (pre-temperature when
    apply_temperature). `vocab_offset`: global id of logits[0] (for a vocab shard: the noise is
    keyed by the global id, so shard winners merge exactly). Returns (token_id, noisy_value)."""
    x = logits.float()
    if temperature != 0.0 and apply_temperature:
        x = x / temperature
    if use_fp64:
        x = x.double()
    if temperature != 0.0:
        ids = torch.arange(vocab_offset, vocab_offset + x.shape[-1], device=x.device)
        x = x + gumbel_noise(seed, pos, ids, is_drafting=is_drafting, use_fp64=use_fp64).to(x.device, x.dtype)
    v, i = x.max(-1)
    return int(i) + vocab_offset, float(v)


def philox_u32(seed: int, offset: int, rounds: int = 10) -> int:
    """Triton tl.randint(seed, offset) for a 64-bit offset (triton/language/random.py, Triton 3.7.1):
    philox(key=(seed_lo, seed_hi), counter=(offset_lo, offset_hi, 0, 0)), first word."""
    A, B, KA, KB = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    s = seed & 0xFFFFFFFFFFFFFFFF
    k0, k1 = s & M32, (s >> 32) & M32
    o = offset & 0xFFFFFFFFFFFFFFFF
    c0, c1, c2, c3 = o & M32, (o >> 32) & M32, 0, 0
    for _ in range(rounds):
        _c0, _c2 = c0, c2
        c0 = ((B * _c2) >> 32) ^ c1 ^ k0
        c2 = ((A * _c0) >> 32) ^ c3 ^ k1
        c1 = (B * _c2) & M32
        c3 = (A * _c0) & M32
        k0 = (k0 + KA) & M32
        k1 = (k1 + KB) & M32
    return c0


def tl_rand32(seed: int, offset: int) -> float:
    """vLLM tl_rand32(seed, offset, includes_zero=False) (gumbel.py:64-69) -> np.float32."""
    x = philox_u32(seed, offset)
    if x >= 1 << 31:
        x -= 1 << 32
    if x < 0:
        x = -x - 1
    u = np.float32(x) * np.float32(4.6566127342e-10)
    return max(u, TL_RAND_MIN)


# ===================================================================================== draft sampling


def draft_sample_sequential_ref(base_logits: torch.Tensor, anchor: int, markov_embed: torch.Tensor,
                                markov_head: torch.Tensor, *, temperature: float, seed: int,
                                sample_pos: Sequence[int], use_fp64: bool = False,
                                logits_dtype: torch.dtype | None = torch.bfloat16) -> dict:
    """DSparkSpeculator._sample_sequential (speculator.py:156-199) for one request.

    base_logits [N, V] (draft-vocab logits from the backbone); sample_pos[i] = position P of the
    token predicted at slot i (the draw is keyed by P-1, drafting salt). Returns dict(tokens [N],
    logits [N, V] = the cached pre-temperature draft logits (base + Markov bias) used by verification,
    markov_embeds [N, 256]).

    logits_dtype: stock vLLM computes the base logits and the Markov bias through LogitsProcessor
    in the model dtype (head_dtype defaults to bf16, config/model.py:2503-2524) and adds them in
    bf16 (speculator.py:184-185), so the sampled and cached logits are bf16(bf16(base) + bf16(bias)).
    None keeps everything in fp32.
    """
    def rnd(t: torch.Tensor) -> torch.Tensor:
        return t.float() if logits_dtype is None else t.to(logits_dtype).float()

    prev = anchor
    toks, cached, embeds = [], [], []
    for i in range(base_logits.shape[0]):
        bias, e = markov_bias_ref(torch.tensor([prev], device=base_logits.device), markov_embed, markov_head)
        li = rnd(rnd(base_logits[i]) + rnd(bias[0]))
        t, _ = gumbel_sample_ref(li, temperature, seed, int(sample_pos[i]) - 1, is_drafting=True, use_fp64=use_fp64)
        toks.append(t)
        cached.append(li)
        embeds.append(e[0])
        prev = t
    return {"tokens": toks, "logits": torch.stack(cached), "markov_embeds": torch.stack(embeds)}


# ===================================================================================== verification


def _lse(x: torch.Tensor) -> torch.Tensor:
    m = x.max(-1, keepdim=True).values
    s = torch.where(m > float("-inf"), torch.exp(x - m).sum(-1, keepdim=True), torch.zeros_like(m))
    return (m + torch.log(s)).squeeze(-1)


def block_verify_ref(target_logits: torch.Tensor, draft_logits: torch.Tensor | None, draft_tokens: Sequence[int],
                     *, temperature: float, seed: int, positions: Sequence[int], use_fp64: bool = False,
                     block: bool = True) -> dict:
    """Stock rejection_sample() for one request (rejection_sampler_utils.py:1015-1328).

    target_logits [K+1, V]: *processed* target logits (temperature already applied by
    Sampler.apply_sampling_params); draft_logits [K, V]: the cached draft logits (pre-temperature,
    divided by `temperature` here like the kernels) or None for one-hot drafts; draft_tokens [K]
    (-1 = placeholder); positions [K+1]: position of each logits row (keys the RNG).
    block=True: block verification (rejection_sample_method="block"); False: standard
    Leviathan et al. token-wise rejection.
    Returns dict(tokens (accepted drafts + resampled/bonus token), num_sampled, accepted,
    cum_log_p [K], h [K], u [K], residual_mass {row: value}).
    """
    tl = target_logits.float()
    K = len(draft_tokens)
    V = tl.shape[-1] if draft_logits is None else min(tl.shape[-1], draft_logits.shape[-1])
    tl = tl[:, :V]
    d = list(int(t) for t in draft_tokens)
    info: dict = {"u": [None] * K, "h": [None] * K, "residual_mass": {}, "cum_log_p": [None] * K}
    if temperature == 0.0:  # greedy (:567-586)
        acc = 0
        for i in range(K):
            am = int(tl[i].argmax())
            ok = d[i] >= 0 and am == d[i]
            if not ok:
                return {"tokens": d[:i] + [am], "num_sampled": i + 1, "accepted": i, **info}
            acc += 1
        return {"tokens": d + [int(tl[K].argmax())], "num_sampled": K + 1, "accepted": K, **info}

    t_lp = tl - _lse(tl).unsqueeze(-1)  # target log-probs per row
    if draft_logits is not None:
        dl = draft_logits.float()[:, :V] / temperature
        d_lp = dl - _lse(dl).unsqueeze(-1)
    else:
        d_lp = None

    def lq(row: int, tok: int) -> float:
        return 0.0 if d_lp is None else float(d_lp[row, tok])

    # cumulative_log_p (:306-385): log p_{i+1} = min(0, log p_i + log Mb(x_i) - log Ms(x_i)).
    # Scalar math below is fp32 like the Triton kernels (np.float32 ops round to fp32).
    f32 = np.float32
    cum, lp = [f32(0.0)] * K, f32(0.0)
    for s in range(K):
        if d[s] >= 0:
            lp = min(lp + (f32(t_lp[s, d[s]]) - f32(lq(s, d[s]))), f32(0.0))
        cum[s] = lp
    info["cum_log_p"] = [float(c) for c in cum]

    def residual_mass(row: int, p_prefix: float) -> float:
        # (:387-484) for full draft distributions; closed form for one-hot drafts (:48-91)
        if d_lp is None:
            return p_prefix * (1.0 - float(torch.exp(t_lp[row, d[row]])))
        return float(torch.clamp(p_prefix * torch.exp(t_lp[row]) - torch.exp(d_lp[row]), min=0).sum())

    accepted, verifying = 0, True
    for i in range(K):
        verifying = verifying and d[i] >= 0
        if not verifying:
            continue
        u = float(tl_rand32(seed, int(positions[i])))
        info["u"][i] = u
        if block:
            p = np.exp(cum[i])  # fp32 (:569-571)
            nxt = d[i + 1] if i < K - 1 else -1
            if nxt >= 0:
                rm = f32(residual_mass(i + 1, float(p)))
                info["residual_mass"][i + 1] = float(rm)
                denom = rm + f32(1.0) - p  # (:589-590), fp32
                h = rm / denom if denom > 0.0 else f32(1.0)
            else:
                h = p
            info["h"][i] = float(h)
            if u <= h:
                accepted = i + 1
        else:
            ok = f32(t_lp[i, d[i]]) > np.log(f32(u)) + f32(lq(i, d[i]))
            verifying = ok
            accepted += int(ok)

    tau = accepted
    is_bonus = tau == K
    if is_bonus or d[tau] < 0:
        residual = tl[tau]
    elif d_lp is not None:
        tlp = t_lp[tau].clone()
        if block and tau > 0:
            tlp = tlp + float(cum[tau - 1])
        ratio = torch.exp(d_lp[tau] - tlp)
        residual = torch.where(ratio < 1.0, tlp + torch.log1p(-ratio), torch.full_like(tlp, float("-inf")))
    else:
        residual = tl[tau].clone()
        residual[d[tau]] = float("-inf")
    tok, _ = gumbel_sample_ref(residual, temperature, seed, int(positions[tau]), is_drafting=False,
                               apply_temperature=False, use_fp64=use_fp64)
    return {"tokens": d[:tau] + [tok], "num_sampled": tau + 1, "accepted": tau, "residual_logits": residual, **info}
