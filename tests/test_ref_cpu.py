"""CPU self-checks of ds41h.ref (no GPU, no CUDA context). Run:

    source scripts/env.sh; export PYTHONPATH=$PWD:$PYTHONPATH
    CUDA_VISIBLE_DEVICES="" python tests/test_ref_cpu.py            # all tests (standalone runner)
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_ref_cpu.py  # same, if pytest is installed
    CUDA_VISIBLE_DEVICES="" python tests/test_ref_cpu.py --bench    # reference/dequant throughput

RAM: loads at most ~2.5 GB at a time (one layer's dense weights unsharded in fp32 plus one
dequantized expert); routed experts are read lazily per selected expert.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import torch

from common import (  # noqa: I001
    D, assert_close, cfg, compare, embed_rows, fmt_bw, fmt_metrics, free, load_layer, load_layer_raw, random_pre_mix,
    reader, realistic_normed, realistic_stream, run_module_tests, skip, bench,
)
from ds41h import ref
from ds41h.ref import quant as Q

torch.set_grad_enabled(False)

# expected per-rank parameter shapes of stock vLLM 0.32.0 at TP=4 (checkpoint form; see
# ds41h/ref/weights.py rules for the source lines). None = replicated (full shape).
EXPECTED_RANK_SHAPES = {
    "embed.weight": (32320, 5120),
    "head.weight": (32320, 5120),
    "layers.3.attn.wq_a.weight": None,
    "layers.3.attn.wkv.scale": None,
    "layers.3.attn.wq_b.weight": (8192, 1280),
    "layers.3.attn.wq_b.scale": (256, 40),
    "layers.3.attn.wo_a.weight": (2048, 4096),
    "layers.3.attn.wo_a.scale": (64, 128),
    "layers.3.attn.wo_b.weight": (5120, 2048),
    "layers.3.attn.wo_b.scale": (160, 64),
    "layers.3.attn.attn_sink": (16,),
    "layers.3.ffn.gate.weight": None,
    "layers.3.ffn.gate.bias": None,
    "layers.3.ffn.experts.7.w1.weight": (576, 2560),
    "layers.3.ffn.experts.7.w1.scale": (576, 160),
    "layers.3.ffn.experts.7.w3.weight": (576, 2560),
    "layers.3.ffn.experts.7.w2.weight": (5120, 288),
    "layers.3.ffn.experts.7.w2.scale": (5120, 18),
    "layers.3.ffn.shared_experts.w1.weight": (576, 5120),
    "layers.3.ffn.shared_experts.w1.scale": (18, 160),
    "layers.3.ffn.shared_experts.w2.weight": (5120, 576),
    "layers.3.ffn.shared_experts.w2.scale": (160, 18),
    "layers.3.hc_attn_fn": None,
    "layers.2.attn.compressor.wkv.weight": None,
    "layers.2.attn.compressor.wgate.weight": None,
    "layers.2.attn.indexer.wq_b.weight": None,
    "layers.2.attn.indexer.weights_proj.weight": None,
    "layers.2.attn.indexer.wk.weight": None,
    "layers.1.engram.wkv.weight": (6400, 6144),
    "layers.1.engram.wkv.scale": (200, 192),
    "layers.1.engram.q_weight": None,
    "mtp.0.main_proj.weight": None,
    "mtp.0.attn.wq_b.weight": (8192, 1280),
    "mtp.0.ffn.experts.5.w1.weight": (576, 2560),
    "mtp.2.markov_head.embed.weight": None,
    "mtp.2.markov_head.head.weight": None,
    "mtp.2.confidence_head.proj.weight": None,
}


# ===================================================================================== formats


def test_fp4_table_and_bit_tricks():
    b = torch.arange(256, dtype=torch.uint8).view(1, 256)
    v = Q.unpack_fp4(b).view(256, 2)
    tab = Q.fp4_table()
    for x in range(256):
        assert v[x, 0] == tab[x & 15] and v[x, 1] == tab[x >> 4], x
    # OCP e2m1 decode: s | e(2) | m(1); e==0 -> m*0.5 else 2^(e-1)*(1+m/2)
    for c in range(16):
        s, e, m = c >> 3, (c >> 1) & 3, c & 1
        val = (m * 0.5 if e == 0 else 2.0 ** (e - 1) * (1 + m / 2)) * (-1) ** s
        assert float(tab[c]) == val, c
        # e2m1 -> e4m3 byte trick is exact x 2^-6, e2m1 -> f16 bits trick exact x 2^-14 (DESIGN.md)
        e4 = torch.tensor([((c & 8) << 4) | ((c & 7) << 2)], dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        assert float(e4) * 64 == val, c
        f16 = np.array([((c & 8) << 12) | ((c & 7) << 9)], dtype=np.uint16).view(np.float16).astype(np.float64)
        assert float(f16[0]) * 2**14 == val, c
    # RNE rounding to the e2m1 grid (ties to even code), saturation at 6
    xs = torch.tensor([0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.26, 1.75, 2.5, 2.51, 3.49, 3.5, 5.0, 5.01, 6.0, 7.0, -0.75, -5.0])
    exp = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0, -1.0, -4.0])
    assert torch.equal(Q.round_to_e2m1(xs), exp), Q.round_to_e2m1(xs)
    codes = Q.fp4_codes(Q.round_to_e2m1(torch.randn(64) * 3))
    assert torch.equal(Q.unpack_fp4(Q.pack_fp4_codes(codes)), tab[codes])


def test_e8m0_decode():
    b = torch.arange(256, dtype=torch.uint8)
    mine = Q.e8m0_to_float(b)
    torch_native = b.view(torch.float8_e8m0fnu).float()
    assert torch.isnan(mine[255]) and torch.isnan(torch_native[255])
    assert torch.equal(mine[:255], torch_native[:255])
    for i in (0, 1, 126, 127, 128, 254):
        assert float(mine[i]) == 2.0 ** (i - 127)
    assert float(mine[0]) > 0  # 2^-127 is an fp32 subnormal, not 0


def test_quant_sims():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 512, generator=g) * torch.logspace(-3, 2, 512).view(1, -1)
    q, s = Q.fp8_quant(x, 32)
    m, _ = torch.frexp(s)
    assert torch.all(m == 0.5), "ue8m0 scales must be powers of two"
    assert torch.all(q.float().abs() <= 448)
    xq = Q.act_quant_sim(x, 32)
    assert torch.equal(Q.bf16_round(xq), xq), "act_quant output must be bf16-representable"
    rel = ((xq - x).abs() / x.abs().unflatten(-1, (-1, 32)).amax(-1).repeat_interleave(32, -1)).max()
    assert rel <= 16 / 224 + 1e-6, rel  # half e4m3 ulp (16) over the smallest amax/s (224)
    assert torch.equal(Q.act_quant_sim(xq, 32), xq), "fake quant must be idempotent"
    # scale rounding exactly like kernel.py fast_round_scale (bit trick on amax * fp32(1/448))
    amax = x.abs().unflatten(-1, (-1, 32)).amax(-1).clamp_min(1e-4)
    prod = (amax * torch.tensor(1 / 448, dtype=torch.float32)).numpy().view(np.uint32)
    e = ((prod >> 23) & 0xFF).astype(np.int64) - 127 + ((prod & 0x7FFFFF) != 0)
    assert torch.equal(s, torch.from_numpy(np.ldexp(1.0, e).astype(np.float32)))
    for fmt, blk in (("ue8m0", 32), ("e4m3", 16)):
        y = Q.fp4_quant_sim(x, blk, fmt)
        assert torch.isfinite(y).all(), fmt
        if fmt == "ue8m0":  # pow2 scales make the fake quant idempotent (e4m3 scales need not be)
            assert torch.equal(Q.fp4_quant_sim(y, blk, fmt), y)
        else:
            assert torch.all(Q.round_to_e4m3(y.abs().unflatten(-1, (-1, 16)).amax(-1)) >= 0)
        assert compare(y, x)["cos"] > 0.99, fmt
    # vLLM V4 record: pack/unpack round trip == simulation; paged insert/gather round trip
    kv = torch.randn(10, 512, generator=g) * 3
    sim = Q.v4_record_sim(kv)
    data, sc = Q.v4_pack(kv)
    assert data.shape == (10, 576) and sc.shape == (10, 8) and torch.all(sc[:, 7] == 0)
    assert torch.equal(Q.v4_unpack(data, sc), sim)
    assert torch.equal(sim[:, 448:], Q.bf16_round(kv[:, 448:]))
    B, nb = 64, 3
    cache = torch.zeros(nb, B * 584 + 64, dtype=torch.uint8)  # padded page stride
    slots = torch.tensor([0, 1, 63, 64, 100, 191, 5, 70, 130, 2])
    Q.v4_cache_insert(cache, B, slots, kv)
    assert torch.equal(Q.v4_cache_gather(cache, B, slots), sim)
    k = Q.indexer_k_fp8_sim(torch.randn(4, 128, generator=g))
    assert torch.isfinite(k).all()


def test_dequant_real_weights():
    rd = reader()
    w, s = rd.get("layers.3.attn.wq_a.weight"), rd.get("layers.3.attn.wq_a.scale")
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu and s.shape == (40, 160)
    deq = ref.dequant_fp8(w, s)
    alt = w.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert torch.equal(deq, alt)
    assert torch.equal(ref.dequant_fp8(w, ref.vllm_scale_layout(s)), deq), "vLLM row-expanded scale form"
    assert int(Q.as_u8(s).max()) < 255
    rms = float(deq.square().mean().sqrt())
    assert 1e-3 < rms < 1.0, rms
    w1, s1 = rd.get("layers.3.ffn.experts.0.w1.weight"), rd.get("layers.3.ffn.experts.0.w1.scale")
    assert w1.dtype == torch.int8 and w1.shape == (2304, 2560) and s1.shape == (2304, 160)
    e = ref.dequant_fp4(w1, s1)
    assert e.shape == (2304, 5120) and torch.isfinite(e).all()
    # manual decode of one row
    row = Q.as_u8(w1[5]).long()
    man = torch.stack([Q.fp4_table()[row & 15], Q.fp4_table()[row >> 4]], -1).flatten()
    man = man * Q.e8m0_to_float(s1[5]).repeat_interleave(32)
    assert torch.equal(e[5], man)
    rms = float(e.square().mean().sqrt())
    assert 1e-4 < rms < 1.0, rms


# ===================================================================================== TP rules


def test_tp_slice_shapes_and_reconstruction():
    rd, c = reader(), reader().config
    for name, exp in EXPECTED_RANK_SHAPES.items():
        full = rd.info(name).shape
        for r in range(4):
            got = ref.tp_shape(name, full, r, cfg=c)
            assert got == (full if exp is None else exp), (name, r, got, exp)
    # every text weight of these layers has a rule
    for pre in ("layers.1.", "layers.2.", "layers.3.", "layers.20.", "layers.24.", "mtp.0.", "mtp.2."):
        for n in rd.names(pre):
            ref.tp_rule(n)
    for n in ("embed.weight", "head.weight", "norm.weight"):
        ref.tp_rule(n)
    # slices concatenate back to the full tensor; partial-row reads equal tp_slice(full)
    for name in ("layers.3.attn.wq_b.scale", "layers.3.attn.wo_b.scale", "layers.3.attn.attn_sink",
                 "layers.3.ffn.experts.7.w1.weight", "layers.3.ffn.experts.7.w2.weight",
                 "layers.3.ffn.experts.7.w2.scale", "layers.3.ffn.shared_experts.w2.scale",
                 "layers.3.attn.wo_a.scale", "layers.1.engram.wkv.scale"):
        full = rd.get(name)
        parts = [ref.tp_slice(name, full, r, cfg=c) for r in range(4)]
        dim = ref.tp_rule(name).dim
        assert torch.equal(torch.cat([Q.as_u8(p) if p.element_size() == 1 else p for p in parts], dim),
                           Q.as_u8(full) if full.element_size() == 1 else full), name
        for r in (0, 3):
            a, b = rd.get_tp(name, r), parts[r]
            assert a.dtype == b.dtype and torch.equal(a.view(torch.uint8) if a.element_size() == 1 else a,
                                                      b.view(torch.uint8) if b.element_size() == 1 else b), name
    # wq_b TP slice == heads 16r..16r+15 of the unsharded projection
    w, s = rd.get("layers.3.attn.wq_b.weight"), rd.get("layers.3.attn.wq_b.scale")
    full = ref.dequant_fp8(w, s).view(64, 512, 1280)
    r = 2
    part = ref.dequant_fp8(rd.get_tp("layers.3.attn.wq_b.weight", r), rd.get_tp("layers.3.attn.wq_b.scale", r))
    assert torch.equal(part.view(16, 512, 1280), full[16 * r : 16 * r + 16])
    # Engram: prime buckets reproduce config sizes; 6 complete heads per rank, contiguous
    lay = ref.engram_layout(c)
    for li, lid in enumerate(lay.layer_ids):
        sizes = lay.head_sizes(li)
        assert len(sizes) == 24 and sum(sizes) == lay.num_embeddings[li], (lid, sum(sizes))
        assert all(16_000_000 < p < 16_001_000 for p in sizes)
        ranges = [lay.rank_rows(li, r) for r in range(4)]
        assert ranges[0][0] == 0 and ranges[-1][1] == sum(sizes)
        assert all(ranges[i][1] == ranges[i + 1][0] for i in range(3))
        name = f"layers.{lid}.engram.embed.weight"
        assert ref.tp_shape(name, rd.info(name).shape, 1, cfg=c) == (ranges[1][1] - ranges[1][0], 256)
    assert len({p for li in range(2) for p in lay.head_sizes(li)}) == 48, "primes must never repeat"


def test_layerweights_shard_matches_tp_slice():
    """LayerWeights.shard(r) (used by decode_layer_ref partials) == loading TP rank r directly."""
    full = load_layer(3, None, experts="lazy")
    r = 3
    direct = load_layer(3, r, experts="lazy")
    sh = full.shard(r)
    for k in ("wq_b", "attn_sink", "wo_a", "wo_b", "sh_w1", "sh_w3", "sh_w2", "wq_a", "wkv", "gate_w", "hc_ffn_fn"):
        assert torch.equal(getattr(sh, k), getattr(direct, k)), k
    for e in (0, 200):
        for a, b in zip(sh.experts.get(e), direct.experts.get(e), strict=True):
            assert torch.equal(a, b), e
    assert direct.n_local_heads == 16 and direct.n_local_groups == 2


# ===================================================================================== mHC / RoPE / routing


def _hc_vllm_style(x, fn, scale, base, eps, rms_eps, iters):
    """Independent re-statement of vLLM mhc_pre_delayed_torch (kernels/mhc/torch.py:6-45)."""
    hc = x.shape[1]
    xf = x.flatten(1).float()
    mixes = (xf @ fn.t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]) * 2.0
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * scale[2] + base[2 * hc :].view(1, hc, hc)
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def test_sinkhorn_doubly_stochastic():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1000, 4, 4, generator=g) * 4
    c = ref.sinkhorn(logits, 20, 1e-6)
    assert torch.all(c > 0)
    assert (c.sum(-2) - 1).abs().max() < 1e-5, "columns (last op) normalized"
    row_err = float((c.sum(-1) - 1).abs().max())
    print(f"    random logits*4: max |row sum - 1| after 20 iters = {row_err:.2e}")
    assert row_err < 0.1, row_err
    rd = reader()
    fn, sc, bs = rd.get("layers.3.hc_attn_fn"), rd.get("layers.3.hc_attn_scale"), rd.get("layers.3.hc_attn_base")
    X = realistic_stream(16, seed=3)
    pre, post, comb, _ = ref.hc_mixes(X, fn, sc, bs, norm_eps=1e-20, hc_eps=1e-6, iters=20)
    p2, q2, c2 = _hc_vllm_style(X, fn, sc, bs, 1e-6, 1e-20, 20)
    assert torch.allclose(pre, p2, atol=1e-6) and torch.allclose(post, q2, atol=1e-6) and torch.allclose(comb, c2, atol=1e-6)
    assert torch.all((pre > 0) & (pre <= 1 + 1e-6)) and torch.all((post > 0) & (post < 2))
    row_err = float((comb.sum(-1) - 1).abs().max())
    print(f"    layer-3 hc_attn comb: max |row sum - 1| = {row_err:.2e}")
    assert row_err < 0.05 and (comb.sum(-2) - 1).abs().max() < 1e-5
    # hc_post with comb doubly stochastic and post=0 preserves the copy-sum
    y = ref.hc_post(torch.zeros(16, D), X, torch.zeros(16, 4), comb)
    assert compare(y.sum(1), X.sum(1))["rel_rmse"] < 3e-3
    # lagged collapse semantics: hc_pre(..., pre_mix=None) selects copy 0 (layer 0)
    y0, *_ = ref.hc_pre(X, fn, sc, bs, None)
    assert torch.equal(y0, X[:, 0])


def test_rope():
    c = cfg()
    f0 = ref.layer.rope_inv_freq(0, c)
    assert torch.allclose(f0, 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)))
    # YaRN (compressed layers) vs an independent statement of vLLM's _yarn_* helpers
    base, dim, orig, fac = 160000.0, 64, 65536, 16.0

    def cdim(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(cdim(32)), 0), min(math.ceil(cdim(1)), dim - 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (fac * pos_freqs)
    mask = 1 - ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
    yarn = inter * (1 - mask) + extra * mask
    f2 = ref.layer.rope_inv_freq(2, c)
    assert torch.allclose(f2, yarn, rtol=1e-6), (f2 - yarn).abs().max()
    assert torch.equal(ref.layer.rope_inv_freq(1, c), f2)
    x = torch.randn(5, 3, 512)
    pos = torch.tensor([0, 1, 127, 4096, 70000])
    y = ref.rope_interleaved(x, pos, 2, cfg=c)
    assert torch.equal(y[..., :448], x[..., :448])
    assert torch.allclose(ref.rope_interleaved(y, pos, 2, cfg=c, inverse=True), x, atol=1e-5)
    pair = lambda t: t[..., 448:].unflatten(-1, (32, 2)).norm(dim=-1)  # noqa: E731
    assert torch.allclose(pair(y), pair(x), atol=1e-5)
    assert torch.equal(ref.rope_interleaved(x[:1], pos[:1], 0, cfg=c), x[:1]), "position 0 is identity"


def test_routing_matches_bruteforce():
    rd = reader()
    gw, gb = rd.get("layers.3.ffn.gate.weight"), rd.get("layers.3.ffn.gate.bias")
    x = realistic_normed(64, rd.get("layers.3.ffn_norm.weight").float(), seed=5)
    wts, idx, scores = ref.gate_ref(x, gw, gb, 6, 1.5)
    xs = x.double()
    for t in range(x.shape[0]):
        logits = (xs[t] @ gw.double().t()).tolist()
        sc = [math.sqrt(math.log1p(math.exp(v)) if v < 20 else v) for v in logits]
        order = sorted(range(384), key=lambda e: -(sc[e] + float(gb[e])))[:6]
        assert set(order) == set(idx[t].tolist()), t
        tot = sum(sc[e] for e in order)
        exp_w = {e: sc[e] / tot * 1.5 for e in order}
        for j, e in enumerate(idx[t].tolist()):
            assert abs(float(wts[t, j]) - exp_w[e]) < 1e-5, (t, e)
    assert torch.allclose(wts.sum(-1), torch.full((64,), 1.5), atol=1e-5)


def test_select_candidate_blocks():
    g = torch.Generator().manual_seed(0)
    T, C = 3, 100
    s = torch.randn(T, C, generator=g)
    lens = torch.tensor([[40], [97], [100]])
    s = s.masked_fill(torch.arange(C) >= lens, float("-inf"))
    keep = ref.select_candidate_blocks(s, lens, 4, 8)
    for t in range(T):
        L = int(lens[t])
        blocks = keep[t].view(-1)[: C].tolist()
        kb = {i // 8 for i in range(C) if blocks[i]}
        assert (L - 1) // 8 in kb, "newest (partial) block pinned"
        assert len(kb) == min(4, -(-L // 8))
        assert all(b * 8 < L for b in kb), "no unreachable block"
    keep2 = ref.select_candidate_blocks(s, lens, 100, 8)
    assert torch.equal(keep2, torch.arange(C).view(1, -1) < ((lens + 7) // 8) * 8)


# ===================================================================================== attention / layer


P0, T = 300, 6


def _history(W, seed=11):
    a_hist = realistic_normed(P0, W.attn_norm, seed=seed)
    a_step = realistic_normed(T, W.attn_norm, seed=seed + 1)
    return a_hist, a_step


def test_layer2_compressor_indexer():
    """KV + index source layer (ratio 2): compressor groups, index keys, top-k validity."""
    W = load_layer(2, None, experts="none")
    a_hist, a_step = _history(W)
    pos = torch.arange(P0, P0 + T)
    for n in (ref.EXACT, ref.W_ONLY, ref.VLLM_SM90):
        ctx = ref.build_attn_context(W, a_hist, pos, n=n)
        assert ctx.compressed_kv.shape == (P0 // 2, 512) and ctx.index_k.shape == (P0 // 2, 128)
        att = ref.attention_ref(W, a_step, pos, ctx, n=n)
        C = (P0 + T) // 2
        assert att["compressed_kv"].shape == (C, 512) and att["index_k"].shape == (C, 128)
        # decode compressor == compressing the whole sequence at once
        full = ref.compress_history_ref(W, torch.cat([a_hist, a_step]), n=n)
        assert torch.allclose(att["compressed_kv"], full["kv"], atol=1e-5)
        assert torch.allclose(att["index_k"], full["index_k"], atol=1e-5)
        tk = att["topk_idxs"]
        lens = (pos + 1) // 2
        assert tk.shape == (T, min(512, C))
        for t in range(T):
            v = [i for i in tk[t].tolist() if i >= 0]
            assert v == sorted(set(v)) and len(v) == int(lens[t]) and all(i < int(lens[t]) for i in v)
        assert torch.isfinite(att["y"]).all() and att["y"].shape == (T, D)
    del W
    free()


def test_layer3_decode_ref():
    """Layer 3 (ratio-2 consumer of KV/index source 2), T=6, EXACT vs W_ONLY, TP partials."""
    W = load_layer(3, None, experts="lazy")
    S = load_layer(2, None, experts="none")
    a_hist, a_step = _history(W)
    pos = torch.arange(P0, P0 + T)
    X = realistic_stream(T, seed=21)
    pre = random_pre_mix(T, seed=22)
    outs = {}
    for name, n in (("exact", ref.EXACT), ("wonly", ref.W_ONLY)):
        ctx = ref.build_attn_context(W, a_hist, pos, n=n, a_step=a_step, source=S, index_source=S)
        t0 = time.perf_counter()
        o = ref.decode_layer_ref(W, X, pos, pre, ctx, n=n, rank=1 if name == "wonly" else None)
        print(f"    decode_layer_ref[{name}] {time.perf_counter() - t0:.1f}s, "
              f"{len(torch.unique(o['moe']['idx']))} experts")
        for k in ("x_out", "x_mid", "a", "f"):
            assert torch.isfinite(o[k]).all(), (name, k)
        assert o["x_out"].shape == (T, 4, D) and o["attn"]["q"].shape == (T, 64, 512)
        assert o["attn"]["kv_rows"].shape[1] == 128 + o["attn"]["topk_idxs"].shape[1]
        y, resid, post, comb, ffn_pre, aux = o["vllm"]
        assert post.shape == (T, 4, 1) and comb.shape == (T, 4, 4) and ffn_pre.shape == (T, 4)
        assert aux.shape == (T, D) and torch.equal(aux, o["aux"])
        exp_aux = X.float().mean(1)
        assert torch.equal(aux, ref.quant.bf16_round(exp_aux) if n.bf16 else exp_aux), "aux = mean of input stream"
        assert torch.allclose(ref.hc_post(y, resid, post, comb, n), o["x_out"], atol=1e-6)
        outs[name] = o
    e, w = outs["exact"], outs["wonly"]
    for k in ("a", "x_mid", "f", "x_out"):
        m = compare(e[k], w[k])
        print(f"    exact vs w_only {k:6s} {fmt_metrics(m)}")
        assert m["cos"] > 0.995, k
    m = compare(e["moe"]["y"], w["moe"]["y"])
    print(f"    exact vs w_only moe_y  {fmt_metrics(m)}")
    assert m["cos"] > 0.98
    # TP partials: rank-1 partial from decode_layer_ref, all ranks' attention partials sum to full
    rk = w["rank"]
    assert rk["q"].shape == (T, 16, 512) and rk["z"].shape == (T, 2048) and rk["attn_y"].shape == (T, D)
    o_full = w["attn"]["o"]
    tot = torch.zeros(T, D)
    for r in range(4):
        tot += ref.o_proj_ref(W.shard(r), o_full[:, 16 * r : 16 * r + 16], pos)["y"]
    assert_close(tot, w["attn"]["y"], cos_min=0.999999, rel_rmse_max=1e-5, name="sum_r attn partials")
    assert torch.allclose(rk["attn_y"], ref.o_proj_ref(W.shard(1), o_full[:, 16:32], pos)["y"], atol=1e-6)
    # MoE partials: experts sliced 576/rank sum to the full expert (W_ONLY and EXACT)
    f = w["f"]
    for e_id in w["moe"]["idx"][0, :2].tolist():
        full_w = W.experts.get(e_id)
        for n in (ref.W_ONLY, ref.EXACT):
            yf = ref.expert_ffn_ref(f, *full_w, None, 10.0, n)
            ys = sum(ref.expert_ffn_ref(f, *ref.CheckpointExperts(reader(), "layers.3.ffn.experts", 384, r).get(e_id),
                                        None, 10.0, n) for r in range(4))
            tol = 1e-5 if n is ref.W_ONLY else 2e-2  # EXACT rounds each partial to bf16
            assert_close(ys, yf, cos_min=0.9999 if n is ref.W_ONLY else 0.999, rel_rmse_max=tol,
                         name=f"expert {e_id} partial sum")
    sh = [W.shard(r) for r in range(4)]
    ysh = sum(ref.expert_ffn_ref(f, s.sh_w1, s.sh_w3, s.sh_w2) for s in sh)
    assert_close(ysh, ref.expert_ffn_ref(f, W.sh_w1, W.sh_w3, W.sh_w2), cos_min=0.99999, rel_rmse_max=1e-5,
                 name="shared partial sum")
    del W, S, outs
    free()


def test_routed_experts_ref_raw_layout():
    """routed_experts_ref on the DESIGN raw layout (rank-local) == moe_ref's routed partial."""
    r = 2
    W = load_layer(3, r, experts="lazy")
    x = realistic_normed(4, W.ffn_norm, seed=7)
    wts, idx, _ = ref.gate_ref(x, W.gate_w, W.gate_bias, 6, 1.5)
    ids = sorted(set(idx.flatten().tolist()))
    raw = load_layer_raw(3, r, experts=ids)
    local = torch.tensor([ids.index(e) for e in idx.flatten().tolist()]).view_as(idx)
    y = ref.routed_experts_ref(x.bfloat16(), wts, local, raw["w13"], raw["s13"], raw["w2"], raw["s2"])
    yr = ref.moe_ref(x, W.gate_w, W.gate_bias, W.experts, None, routing=(wts, idx), return_all=True)["routed"]
    assert y.dtype == torch.float32 and y.shape == (4, D)
    assert torch.allclose(y, yr, atol=1e-6, rtol=1e-5), (y - yr).abs().max()
    del W, raw
    free()


def test_layer20_candidate_source():
    """Layer 20 (ratio 1, KV + index + candidate source): candidates published, top-k valid."""
    W = load_layer(20, None, experts="none")
    a_hist, a_step = _history(W, seed=31)
    pos = torch.arange(P0, P0 + T)
    ctx = ref.build_attn_context(W, a_hist, pos, n=ref.W_ONLY)
    assert ctx.comp_state is None and ctx.compressed_kv.shape[0] == P0
    att = ref.attention_ref(W, a_step, pos, ctx, n=ref.W_ONLY)
    ix = att["indexer"]
    assert ix["candidates"] is not None and ix["candidates"].shape == (T, P0 + T)
    assert att["topk_idxs"].shape == (T, 306) and torch.isfinite(att["y"]).all()
    # with 306 <= 2048 blocks*8, every reachable position is a candidate
    assert torch.equal(ix["candidates"], torch.arange(P0 + T).view(1, -1) < ((pos.view(-1, 1) + 1 + 7) // 8) * 8)
    del W
    free()


def test_engram_layer1():
    rd, c = reader(), reader().config
    lay = ref.engram_layout(c)
    ids = torch.tensor([5, 1000, 77, 2, 9, 3141, 42, 7])
    hashes = ref.engram_hash_ids(ids, c, pad_compressed_id=2)  # [8, 2, 24]
    assert hashes.shape == (8, 2, 24)
    for li in range(2):
        offs, sizes = lay.offsets(li), lay.head_sizes(li)
        for col in range(24):
            h = hashes[:, li, col]
            assert torch.all((h >= offs[col]) & (h < offs[col] + sizes[col]))
    # position 0: all n-grams are pad-filled except the current token -> same within an n-gram size
    W_wkv = ref.dequant_fp8(rd.get("layers.1.engram.wkv.weight"), rd.get("layers.1.engram.wkv.scale"))
    rows = ref.engram_lookup(rd, 1, hashes[:2, 0])
    assert rows.shape == (2, 24, 256) and torch.isfinite(rows).all()
    X = realistic_stream(2, seed=41)
    q, k = rd.get("layers.1.engram.q_weight"), rd.get("layers.1.engram.k_weight")
    out, info = ref.engram_ref(X, rows, W_wkv, q, k, n=ref.EXACT)
    assert torch.isfinite(out).all() and torch.all((info["gate"] > 0) & (info["gate"] < 1))
    assert out.shape == X.shape
    print(f"    engram gate mean {float(info['gate'].mean()):.3f}, |value| rms "
          f"{float(info['value'].square().mean().sqrt()):.3e}")
    del W_wkv
    free()


def test_dspark_forward_and_markov():
    c = cfg()
    rd = reader()
    p = 1000  # last target position; anchor at p+1
    N = c.dspark_block_size
    mh = torch.cat([realistic_stream(129, seed=50 + i).mean(1) for i in range(3)], -1)  # [129, 3D]
    main_proj = ref.dequant_fp8(rd.get("mtp.0.main_proj.weight"), rd.get("mtp.0.main_proj.scale"))
    DWlite = ref.DSparkWeights([], main_proj, rd.get("mtp.0.main_norm.weight").float(),
                               rd.get("mtp.2.norm.weight").float(), torch.empty(0), torch.empty(0), None)
    main_x = ref.dspark_main_x_ref(DWlite, mh)
    ctx_pos = torch.arange(p - 128, p + 1)  # ring keeps the last 128 positions <= p
    ids = torch.tensor([4242] + [c.dspark_noise_token_id] * (N - 1))
    X = embed_rows(ids).unsqueeze(1).repeat(1, 4, 1)
    bpos = torch.arange(p + 1, p + 1 + N)
    pre = None
    for i in range(c.n_mtp_layers):
        W = load_layer(c.n_layers + i, None, experts="lazy", mtp=i)
        assert W.gate_w.shape[0] == 128 and W.compress_ratio == 0
        ring = ref.dspark_context_kv_ref(W, main_x[-128:], ctx_pos[-128:])
        b = ref.dspark_block_ref(W, X, pre, bpos, ring)
        assert b["moe"]["idx"].shape == (N, 3)
        X, pre = b["x_out"], b["ffn_pre"]
        del W
        free()
    head_hidden = ref.hc_collapse(X, pre)
    assert torch.isfinite(head_hidden).all() and head_hidden.shape == (N, D)
    # tied head on a vocab slice + Markov bias (replicated in vLLM) + confidence head
    head_slice = rd.get_rows("head.weight", 0, 4096).float()
    base = ref.dspark_logits_ref(head_hidden, DWlite.norm, head_slice)
    me = rd.get("mtp.2.markov_head.embed.weight")
    mhw = rd.get_rows("mtp.2.markov_head.head.weight", 0, 4096)
    bias, emb = ref.markov_bias_ref(torch.tensor([4242]), me, mhw)
    assert base.shape == (N, 4096) and bias.shape == (1, 4096) and emb.shape == (1, 256)
    conf = ref.confidence_ref(head_hidden[:1], emb, rd.get("mtp.2.confidence_head.proj.weight"))
    assert 0 < float(conf) < 1
    d = ref.draft_sample_sequential_ref(base, 4242, me, mhw, temperature=1.0, seed=7,
                                        sample_pos=list(range(p + 2, p + 2 + N)))
    assert len(d["tokens"]) == N and d["logits"].shape == (N, 4096)
    assert torch.equal(d["logits"], d["logits"].bfloat16().float()), "stock adds base + Markov bias in bf16"
    print(f"    dspark head_hidden rms {float(head_hidden.square().mean().sqrt()):.3f}, conf {float(conf):.3f}")


# ===================================================================================== sampling


def test_block_verification_math():
    g = torch.Generator().manual_seed(0)
    V, K = 6, 3
    tgt = torch.randn(K + 1, V, generator=g) * 1.5
    pos = [100, 101, 102, 103]
    # q == p -> every draft accepted (h = 1), bonus sampled from the last row
    r = ref.block_verify_ref(tgt, tgt[:K].clone(), [1, 2, 3], temperature=1.0, seed=5, positions=pos)
    assert r["accepted"] == K and r["num_sampled"] == K + 1
    # greedy: accept while draft == argmax
    am = tgt.argmax(-1).tolist()
    r = ref.block_verify_ref(tgt, None, [am[0], (am[1] + 1) % V, am[2]], temperature=0.0, seed=0, positions=pos)
    assert r["tokens"] == [am[0], am[1]] and r["accepted"] == 1
    # Monte Carlo: first emitted token ~ target row 0; block verification accepts >= token-wise
    drf = tgt[:K] + torch.randn(K, V, generator=g)
    p0 = torch.softmax(tgt[0], -1)
    cnt = torch.zeros(V)
    acc_b = acc_t = 0
    trials = 4000
    for s in range(trials):
        dt = []
        for i in range(K):
            t, _ = ref.gumbel_sample_ref(drf[i], 1.0, 1000 + s, pos[i], is_drafting=True)
            dt.append(t)
        rb = ref.block_verify_ref(tgt, drf, dt, temperature=1.0, seed=1000 + s, positions=pos)
        rt = ref.block_verify_ref(tgt, drf, dt, temperature=1.0, seed=1000 + s, positions=pos, block=False)
        cnt[rb["tokens"][0]] += 1
        acc_b += rb["accepted"]
        acc_t += rt["accepted"]
    emp = cnt / trials
    print(f"    first-token TV distance {float((emp - p0).abs().sum() / 2):.4f}; mean accepted block "
          f"{acc_b / trials:.3f} vs token-wise {acc_t / trials:.3f}")
    assert float((emp - p0).abs().max()) < 0.03
    assert acc_b >= acc_t - 0.02 * trials


def test_rng_bit_exact_properties():
    # murmur3 uniforms are in (0, 1) and depend on (seed, pos, id); philox matches known Triton values
    ids = torch.arange(0, 129280)
    u = ref.dspark.murmur3_uniform32(1234, 77, ids)
    assert float(u.min()) > 0 and float(u.max()) < 1 and abs(float(u.mean()) - 0.5) < 0.01
    assert not torch.equal(u, ref.dspark.murmur3_uniform32(1234, 78, ids))
    us = [float(ref.tl_rand32(s, 5)) for s in range(2000)]
    assert 0 < min(us) and max(us) < 1 and abs(sum(us) / len(us) - 0.5) < 0.02
    # Philox-4x32-10 known-answer test (Random123 kat_vectors: ctr=0, key=0)
    assert ref.philox_u32(0, 0) == 0x6627E8D5


def test_vs_stock_vllm_cpu_pieces():
    """ref vs stock vLLM 0.32.0 code run on CPU: mhc_pre_delayed_torch / mhc_post_torch
    (kernels/mhc/torch.py), the DSV4 RoPE cos/sin cache (common/rope.py -> DeepseekV4Scaling...),
    and the V4 record insert kernel quantize_and_insert_k_cache (Triton interpreter)."""
    code = textwrap.dedent(r"""
        import os, types, torch
        os.environ["TRITON_INTERPRET"] = "1"
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch, mhc_post_torch
        from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
        from vllm.models.deepseek_v41.common.ops.cache_utils import quantize_and_insert_k_cache
        from ds41h import ref
        from ds41h.ref import quant as Q
        rd = ref.CheckpointReader()
        c = ref.ModelConfig.from_hf(rd.config)
        g = torch.Generator().manual_seed(0)
        # --- mHC
        fn, sc, bs = (rd.get(f"layers.3.hc_attn_{k}") for k in ("fn", "scale", "base"))
        X = (torch.randn(7, 4, 5120, generator=g) * 0.05).bfloat16()
        pm = torch.sigmoid(torch.randn(7, 4, generator=g)) + 1e-6
        post_v, comb_v, li_v, pre_v = mhc_pre_delayed_torch(X, fn, sc, bs, 1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix=pm)
        pre, post, comb, _ = ref.hc_mixes(X.float(), fn, sc, bs)
        li = ref.hc_collapse(X.float(), pm, ref.EXACT)
        assert torch.allclose(pre, pre_v, atol=1e-6, rtol=1e-6), (pre - pre_v).abs().max()
        assert torch.allclose(post, post_v.squeeze(-1), atol=1e-6), (post - post_v.squeeze(-1)).abs().max()
        assert torch.allclose(comb, comb_v, atol=1e-6), (comb - comb_v).abs().max()
        assert torch.equal(li, li_v.float()), (li - li_v.float()).abs().max()
        y = torch.randn(7, 5120, generator=g).bfloat16()
        o_v = mhc_post_torch(y, X, post_v, comb_v)
        o = ref.hc_post(y.float(), X.float(), post_v, comb_v, ref.EXACT)
        d = (o - o_v.float()).abs().max()
        assert d <= 2 ** -7 * o.abs().max(), d  # equal up to one bf16 rounding of the fp32 sum
        print("mhc ok", float(d))
        # --- RoPE cos/sin cache
        cfgns = types.SimpleNamespace(rope_parameters={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
            "beta_slow": 1, "original_max_position_embeddings": 65536, "rope_theta": 10000},
            compress_rope_theta=160000, rope_theta=10000)
        pos = torch.tensor([0, 1, 2, 127, 128, 4095, 65535, 300001])
        with set_current_vllm_config(VllmConfig()):
            for ratio in (0, 2):
                r = build_deepseek_v4_rope(cfgns, head_dim=512, rope_head_dim=64,
                                           max_position_embeddings=1048576, compress_ratio=ratio)
                cs = r.cos_sin_cache[pos].float()
                cos, sin = ref.rope_cos_sin(pos, ratio, c)
                dc = max(float((cs[:, :32] - cos).abs().max()), float((cs[:, 32:] - sin).abs().max()))
                assert dc < 2e-3, (ratio, dc)  # angle rounding at large positions; exact at small ones
                small = pos < 4096
                ds = max(float((cs[small, :32] - cos[small]).abs().max()), float((cs[small, 32:] - sin[small]).abs().max()))
                assert ds < 1e-5, (ratio, ds)
                print("rope ok", ratio, dc, ds)
        # --- V4 584 B record insert (stock Triton kernel, interpreted) vs ref bytes
        B, nb = 64, 3
        kv = (torch.randn(9, 512, generator=g) * torch.logspace(-2, 1, 512)).bfloat16()
        slots = torch.tensor([0, 1, 63, 64, 100, 191, 5, 70, 130])
        cache_v = torch.zeros(nb, B * 584, dtype=torch.uint8)
        quantize_and_insert_k_cache(kv, cache_v, slots, block_size=B)
        cache_r = torch.zeros_like(cache_v)
        Q.v4_cache_insert(cache_r, B, slots, kv.float())
        # Scales and bf16 RoPE bytes must match exactly. The Triton *interpreter* emulates the
        # fp32->e4m3 cast in numpy with round-half-up and a broken binade carry (e.g. 42->44,
        # 31.75->16), whereas GPU Triton lowers it to cvt.rn.satfinite (RNE) like the ref. So
        # e4m3 bytes may differ, but then the ref byte must be the true RNE (checked against an
        # independent nearest-value search over all e4m3 codes, ties to even code).
        allv = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        okc = torch.isfinite(allv)
        diffs = total = 0
        for i, s in enumerate(slots.tolist()):
            blk, off = divmod(s, B)
            dr, dv = cache_r[blk, off * 576:(off + 1) * 576], cache_v[blk, off * 576:(off + 1) * 576]
            sr = cache_r[blk, B * 576 + off * 8: B * 576 + off * 8 + 8]
            sv = cache_v[blk, B * 576 + off * 8: B * 576 + off * 8 + 8]
            assert torch.equal(sr, sv) and torch.equal(dr[448:], dv[448:]), (i, sr.tolist(), sv.tolist())
            total += 448
            for j in (dr[:448] != dv[:448]).nonzero().flatten().tolist():
                v = kv[i, j].float().item() / 2.0 ** (int(sr[j // 64]) - 127)
                dist = (allv - v).abs().masked_fill(~okc, float("inf"))
                best = (dist == dist.min()).nonzero().flatten().tolist()
                best = [b for b in best if b % 2 == 0] or best
                assert int(dr[j]) in best, (i, j, v, int(dr[j]), best)
                diffs += 1
        print("v4 record bytes ok", diffs, "of", total, "e4m3 bytes differ only by interpreter rounding")
    """)
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900)
    out = (p.stdout + p.stderr).strip().splitlines()
    for line in out:
        if line.endswith(" ok") or " ok " in line:
            print("    " + line)
    if p.returncode != 0:
        if "ModuleNotFoundError" in p.stderr and "vllm" in p.stderr:
            skip("vllm not importable")
        raise AssertionError("\n".join(out[-15:]))


def test_rng_vs_triton_interpreter():
    """Compare murmur3 / Philox / Gumbel noise with vLLM's own Triton code in TRITON_INTERPRET mode."""
    code = textwrap.dedent(r"""
        import os, sys, torch
        os.environ["TRITON_INTERPRET"] = "1"
        import triton, triton.language as tl
        from vllm.v1.worker.gpu.sample.gumbel import murmur3_hash32, tl_rand32, gumbel_noised_argmax
        @triton.jit
        def k(out_h, out_u, seed, pos, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(out_h + offs, murmur3_hash32(seed, pos, offs).to(tl.int64))
            tl.store(out_u, tl_rand32(seed, pos, includes_zero=False))
        @triton.jit
        def g(logits_ptr, out_v, out_i, seed, pos, N: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(logits_ptr + offs)
            v, i = gumbel_noised_argmax(x, offs, offs < N, seed, pos, 1.0, IS_DRAFTING=True, USE_FP64=False)
            tl.store(out_v, v); tl.store(out_i, i)
        res = []
        for seed, pos in ((0, 0), (123456789, 4242), (-5, 99999), (2**40 + 7, 2**33 + 1)):
            h = torch.zeros(64, dtype=torch.int64); u = torch.zeros(1, dtype=torch.float32)
            k[(1,)](h, u, seed, pos, N=64)
            x = torch.randn(64, generator=torch.Generator().manual_seed(seed & 0xffff))
            v = torch.zeros(1); i = torch.zeros(1, dtype=torch.int64)
            g[(1,)](x, v, i, seed, pos, N=64)
            res.append((seed, pos, (h & 0xffffffff).tolist(), float(u[0]), float(v[0]), int(i[0])))
        print(repr(res))
    """)
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    if p.returncode != 0:
        skip(f"triton interpreter unavailable: {p.stderr.strip().splitlines()[-1] if p.stderr else p.returncode}")
    res = eval(p.stdout.strip().splitlines()[-1])  # noqa: S307
    for seed, pos, h, u, v, i in res:
        mine_h = ref.murmur3_hash32(seed, pos, torch.arange(64)).tolist()
        assert mine_h == h, (seed, pos)
        assert float(ref.tl_rand32(seed, pos)) == u, (seed, pos, float(ref.tl_rand32(seed, pos)), u)
        x = torch.randn(64, generator=torch.Generator().manual_seed(seed & 0xFFFF))
        ti, tv = ref.gumbel_sample_ref(x, 1.0, seed, pos, is_drafting=True)
        assert ti == i and abs(tv - v) <= 1e-5 * max(1.0, abs(v)), (seed, pos, ti, i, tv, v)


def test_ref_gpu_matches_cpu():
    """Needs a GPU: the reference is device-agnostic (layer 3, T=6, W_ONLY, CPU vs CUDA)."""
    from common import has_cuda

    if not has_cuda():
        skip("CUDA not available (run without CUDA_VISIBLE_DEVICES=\"\")")
    W = load_layer(3, None, experts="lazy")
    S = load_layer(2, None, experts="none")
    a_hist, a_step = _history(W)
    pos = torch.arange(P0, P0 + T)
    X, pre = realistic_stream(T, seed=21), random_pre_mix(T, seed=22)
    ctx = ref.build_attn_context(W, a_hist, pos, a_step=a_step, source=S, index_source=S)
    oc = ref.decode_layer_ref(W, X, pos, pre, ctx)
    Wg, Sg = W.to("cuda"), S.to("cuda")
    ctxg = ref.build_attn_context(Wg, a_hist.cuda(), pos.cuda(), a_step=a_step.cuda(), source=Sg, index_source=Sg)
    og = ref.decode_layer_ref(Wg, X.cuda(), pos.cuda(), pre.cuda(), ctxg)
    for k in ("a", "x_mid", "f", "x_out"):
        assert_close(og[k].cpu(), oc[k], cos_min=0.99999, rel_rmse_max=1e-4, name=f"gpu vs cpu {k}")


# ===================================================================================== bench


def bench_main():
    """Reference throughput (the ref is a correctness tool; this sizes test runtimes)."""
    from common import device

    dev = device()
    print(f"device: {dev}")
    rd = reader()
    w1, s1 = rd.get("layers.3.ffn.experts.0.w1.weight").to(dev), rd.get("layers.3.ffn.experts.0.w1.scale").to(dev)
    ms = bench(lambda: ref.dequant_fp4(w1, s1), 1, 5)
    print(fmt_bw("dequant_fp4 2304x5120 (read bytes)", ms, w1.numel() + s1.numel()))
    w, s = rd.get("layers.3.attn.wq_b.weight").to(dev), rd.get("layers.3.attn.wq_b.scale").to(dev)
    ms = bench(lambda: ref.dequant_fp8(w, s), 1, 3)
    print(fmt_bw("dequant_fp8 32768x1280 (read bytes)", ms, w.numel() + s.numel()))
    t = time.perf_counter()
    rd.get_tp("layers.3.attn.wq_b.weight", 1)
    print(fmt_bw("get_tp wq_b rank1 from disk/page cache", (time.perf_counter() - t) * 1e3, 8192 * 1280))
    W = load_layer(3, None, device=dev, experts="lazy")
    x = realistic_normed(6, W.ffn_norm.cpu(), seed=1).to(dev)
    ms = bench(lambda: ref.moe_ref(x, W.gate_w, W.gate_bias, W.experts, (W.sh_w1, W.sh_w3, W.sh_w2)), 0, 1)
    print(fmt_bw(f"moe_ref T=6 ({dev}, lazy experts)", ms))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_main()
        sys.exit(0)
    sys.exit(run_module_tests(globals()))
