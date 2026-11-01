"""GPU tests for ds41h SAMPLING (ds41h/vp_sampling.py): vocab-parallel DSpark drafting + block verification.

Run on one GPU ( the 4 TP ranks are simulated by 4 VPShards whose LL gathers are
replaced by torch.stack, so the per-rank kernels are exercised exactly as on 4 GPUs). DS41H unset.
    python -m pytest tests/test_sampling.py -q
    python tests/test_sampling.py            # same tests without pytest
    python tests/test_sampling.py --bench    # per-step kernel timings vs stock

Tests
  * drafting: vocab-parallel Gumbel argmax is bit-identical to stock gumbel_sample on the full-vocab
    logits (inputs chosen so the Markov dot products are exact), cached draft logits identical,
    winner logit / log-sum-exp consistent;
  * verification: identical (num_sampled, tokens) to stock rejection_sample(block verification) and to
    ds41h.ref.dspark.block_verify_ref, incl. greedy, T=1, T!=1, short/zero-draft requests, -1 drafts;
  * statistics: on a small vocab, the sequence distribution after block verification (+ target
    continuation) matches the target Markov chain (chi-square), for T=1 and T=0.7; greedy exact;
  * CUDA-graph replay of the per-rank draft kernels.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from common import bench, fmt_bw, require_cuda, run_module_tests

TP = 4


def vp():
    from ds41h import vp_sampling

    return vp_sampling


def stock_gumbel():
    try:
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    except Exception:  # noqa: BLE001
        return None
    return gumbel_sample, apply_temperature, rejection_sample


def make_shards(w1, w2, vocab, max_reqs, n_spec, cache_dtype=torch.bfloat16, w2_fp8=False):
    return [
        vp().VPShard.create(
            rank=r, ranks=TP, w1=w1, w2_full=w2, vocab=vocab, max_reqs=max_reqs, n_spec=n_spec, cache_dtype=cache_dtype, gather=None, w2_fp8=w2_fp8
        )
        for r in range(TP)
    ]


# ---------------------------------------------------------------------------------------- drafting
def sim_draft(shards, base, prev, idx_map, sample_pos, temp, seeds):
    """All draft steps on 4 simulated ranks; returns per-rank draft_tokens [max_reqs, n_spec]."""
    V = vp()
    rows, n_spec = base.shape[0], base.shape[1]
    tokens = [torch.zeros(shards[0].max_reqs, n_spec, dtype=torch.int64, device="cuda") for _ in shards]
    for sh in shards:
        torch.index_select(sh.w1, 0, prev, out=sh.emb[0, :rows])
    for i in range(n_spec):
        outs = [V.draft_step_local(sh, base[:, i, sh.start : sh.start + sh.local], sh.emb[i % 2, :rows], idx_map[:, i], sample_pos[:, i], i, temp, seeds).clone() for sh in shards]
        gathered = torch.stack(outs)
        for sh, tok in zip(shards, tokens):
            V.draft_combine(sh, gathered, tok, i, sh.emb[(i + 1) % 2, :rows], idx_map[:, i])
    return tokens


def draft_case(vocab, rows=7, n_spec=5, max_reqs=8, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    R = 256
    # Integer-valued Markov weights: every partial sum of the 256-term dot product is exact in fp32,
    # so the bias is identical whatever the summation order (cuBLAS vs ours).
    w1 = torch.randint(-3, 4, (vocab, R), generator=g, device="cuda").to(torch.bfloat16)
    w2 = torch.randint(-3, 4, (vocab, R), generator=g, device="cuda").to(torch.bfloat16)
    base = (torch.randn(rows, n_spec, vocab, generator=g, device="cuda") * 6).to(torch.bfloat16)
    perm = torch.randperm(max_reqs, generator=torch.Generator().manual_seed(seed))[:rows].to("cuda")
    idx = perm.clone()
    idx[-1] = -1  # padded row (CUDA-graph padding)
    idx_map = idx[:, None].expand(rows, n_spec).contiguous().to(torch.int64)
    temp = torch.tensor([1.0, 0.7, 0.0, 1.3, 1.0, 0.5, 2.0, 1.0], device="cuda")[:max_reqs].contiguous()
    seeds = torch.randint(-(2**62), 2**62, (max_reqs,), generator=g, device="cuda", dtype=torch.int64)
    p0 = torch.randint(100, 60000, (rows, 1), generator=g, device="cuda")
    sample_pos = (p0 + 1 + torch.arange(n_spec, device="cuda")[None, :]).to(torch.int64).contiguous()
    prev = torch.randint(0, vocab, (rows,), generator=g, device="cuda", dtype=torch.int32)
    return w1, w2, base, idx_map, temp, seeds, sample_pos, prev


def stock_draft(w1, w2, base, idx_map, temp, seeds, sample_pos, prev, max_reqs):
    sg = stock_gumbel()
    rows, n_spec, vocab = base.shape
    cache = torch.zeros(max_reqs, n_spec, vocab, dtype=base.dtype, device="cuda")
    toks = torch.zeros(rows, n_spec, dtype=torch.int64, device="cuda")
    cols = torch.arange(n_spec, dtype=torch.int32, device="cuda")
    p = prev
    for i in range(n_spec):
        emb = w1[p.long()]
        bias = torch.nn.functional.linear(emb, w2)  # bf16 out, like LogitsProcessor(markov_w2)
        logits_i = base[:, i] + bias  # bf16 add (speculator.py:185)
        if sg is not None:
            t = sg[0](logits_i, idx_map[:, i], temp, seeds, sample_pos[:, i] - 1, apply_temperature=True, is_drafting=True, logits_cache=cache, logits_cache_col=cols[i])
        else:  # pure reference (bit-exact RNG, fp32 noise math)
            from ds41h.ref.dspark import gumbel_sample_ref

            t = torch.empty(rows, dtype=torch.int64, device="cuda")
            for r in range(rows):
                req = int(idx_map[r, i])
                tt = float(temp[req]) if req >= 0 else 0.0
                sd = int(seeds[req]) if req >= 0 else 0
                t[r] = gumbel_sample_ref(logits_i[r], tt, sd, int(sample_pos[r, i]) - 1, is_drafting=True)[0]
                if req >= 0:
                    cache[req, i] = logits_i[r]
        toks[:, i] = t
        p = t.to(torch.int32)
    return toks, cache


def test_vp_draft_matches_stock():
    require_cuda()
    for vocab, seed in [(8000, 0), (129280, 1)]:
        max_reqs, n_spec = 8, 5
        w1, w2, base, idx_map, temp, seeds, sample_pos, prev = draft_case(vocab, max_reqs=max_reqs, n_spec=n_spec, seed=seed)
        shards = make_shards(w1, w2, vocab, max_reqs, n_spec)
        tokens = sim_draft(shards, base, prev, idx_map, sample_pos, temp, seeds)
        ref_tok, ref_cache = stock_draft(w1, w2, base, idx_map, temp, seeds, sample_pos, prev, max_reqs)
        rows = base.shape[0]
        for r in range(TP):
            assert torch.equal(tokens[r][:rows], ref_tok), f"V={vocab} rank {r}: drafts differ\n{tokens[r][:rows]}\n{ref_tok}"
        cache = torch.cat([sh.logits for sh in shards], dim=-1)
        valid = idx_map[:, 0][idx_map[:, 0] >= 0]
        assert torch.equal(cache[valid].view(torch.int16), ref_cache[valid].view(torch.int16)), "cached draft logits differ"
        for row in range(rows):
            req = int(idx_map[row, 0])
            if req < 0:
                continue
            t = float(temp[req]) if float(temp[req]) > 0 else 1.0
            for i in range(n_spec):
                tok = int(ref_tok[row, i])
                for sh in shards:
                    assert float(sh.token_logit[req, i]) == float(ref_cache[req, i, tok].float())
                    lse = torch.logsumexp(ref_cache[req, i].float() / t, -1)
                    assert math.isclose(float(sh.lse[req, i]), float(lse), rel_tol=1e-5, abs_tol=1e-4), (float(sh.lse[req, i]), float(lse))


def test_vp_draft_fp32_cache_and_fp8_w2_run():
    require_cuda()
    vocab, max_reqs, n_spec = 4000, 8, 3
    w1, w2, base, idx_map, temp, seeds, sample_pos, prev = draft_case(vocab, max_reqs=max_reqs, n_spec=n_spec, seed=3)
    # fp32 head (head_dtype=float32): no bf16 rounding of bias / sum
    shards = make_shards(w1, w2, vocab, max_reqs, n_spec, cache_dtype=torch.float32)
    tokens = sim_draft(shards, base.float(), prev, idx_map, sample_pos, temp, seeds)
    assert all(torch.equal(tokens[0], t) for t in tokens)
    # e4m3 markov_w2 (optional bandwidth mode): runs, ranks agree, tokens in range
    shards = make_shards(w1, w2, vocab, max_reqs, n_spec, w2_fp8=True)
    tokens = sim_draft(shards, base, prev, idx_map, sample_pos, temp, seeds)
    assert all(torch.equal(tokens[0], t) for t in tokens) and int(tokens[0].max()) < vocab and int(tokens[0].min()) >= 0


def test_draft_graph_replay():
    require_cuda()
    V = vp()
    vocab, max_reqs, n_spec = 8000, 8, 5
    w1, w2, base, idx_map, temp, seeds, sample_pos, prev = draft_case(vocab, max_reqs=max_reqs, n_spec=n_spec, seed=5)
    sh = make_shards(w1, w2, vocab, max_reqs, n_spec)[1]

    def replicated(inp, out):  # stands in for the LL gather: every rank contributed `inp`
        return out.copy_(inp.unsqueeze(0).expand_as(out))

    sh.gather = replicated
    local_base = base[:, :, sh.start : sh.start + sh.local].contiguous()
    toks = torch.zeros(max_reqs, n_spec, dtype=torch.int64, device="cuda")
    run = lambda: V.draft_markov(sh, local_base, prev, idx_map, sample_pos, temp, seeds, toks)  # noqa: E731
    run()
    torch.cuda.synchronize()
    eager = toks.clone()
    toks.zero_()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=s):
        run()
    for _ in range(3):
        toks.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(toks, eager)
    assert int(sh.draft_counts.abs().sum()) == 0, "per-row arrival counters must be reset by the last program"


# ---------------------------------------------------------------------------------------- verification
def verify_case(vocab, n_spec=5, num_reqs=16, seed=0, temps=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    max_reqs = num_reqs
    num_draft = [n_spec] * num_reqs
    num_draft[3], num_draft[7] = 2, 0  # short / prefill-only requests
    cu = [0]
    for n in num_draft:
        cu.append(cu[-1] + n + 1)
    num_logits = cu[-1]
    idx_mapping = torch.randperm(max_reqs, generator=cpu).to(torch.int32).cuda()
    if temps is None:
        temps = [0.0, 1.0, 0.6, 1.4]
    temp = torch.tensor([temps[i % len(temps)] for i in range(max_reqs)], dtype=torch.float32, device="cuda")
    seeds = torch.randint(-(2**62), 2**62, (max_reqs,), generator=g, device="cuda", dtype=torch.int64)
    cache = (torch.randn(max_reqs, n_spec, vocab, generator=g, device="cuda") * 3).to(torch.bfloat16)
    logits = torch.empty(num_logits, vocab, dtype=torch.bfloat16, device="cuda")
    draft_sampled = torch.empty(num_logits, dtype=torch.int64, device="cuda")
    expanded_idx = torch.empty(num_logits, dtype=torch.int32, device="cuda")
    local_pos = torch.empty(num_logits, dtype=torch.int32, device="cuda")
    pos = torch.empty(num_logits, dtype=torch.int64, device="cuda")
    drafts = {}
    for r in range(num_reqs):
        req = int(idx_mapping[r])
        s, n = cu[r], num_draft[r]
        t = float(temp[req])
        toks = []
        for j in range(n):
            q = cache[req, j].float()
            toks.append(int(q.argmax()) if t == 0 else int(torch.multinomial(torch.softmax(q / t, -1).cpu(), 1, generator=cpu)))
        if r == 5:
            toks[2] = -1  # placeholder draft
        drafts[r] = toks
        draft_sampled[s] = int(torch.randint(0, vocab, (1,), generator=cpu))
        for j in range(n):
            draft_sampled[s + j + 1] = toks[j]
        p0 = int(torch.randint(1000, 50000, (1,), generator=cpu))
        for j in range(n + 1):
            row = s + j
            expanded_idx[row] = req
            local_pos[row] = j
            pos[row] = p0 + j
            # target close to the draft so that acceptance is non-trivial; bonus row unrelated
            base_row = cache[req, j].float() if j < n else torch.randn(vocab, generator=g, device="cuda") * 3
            logits[row] = (base_row + torch.randn(vocab, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    return dict(
        vocab=vocab, n_spec=n_spec, num_reqs=num_reqs, max_reqs=max_reqs, cu=cu_t, cu_list=cu, idx_mapping=idx_mapping, temp=temp, seeds=seeds,
        cache=cache, logits=logits, draft_sampled=draft_sampled, expanded_idx=expanded_idx, local_pos=local_pos, pos=pos, drafts=drafts,
        num_draft=num_draft,
    )


def fill_shard_draft_state(shards, c):
    """Per-step draft state as the draft step would have left it: local cache slice, winner logit, lse."""
    cache, temp = c["cache"], c["temp"]
    for sh in shards:
        sh.logits.copy_(cache[:, :, sh.start : sh.start + sh.local])
        sh.token_logit.zero_()
        sh.lse.zero_()
    for r in range(c["num_reqs"]):
        req = int(c["idx_mapping"][r])
        t = float(temp[req]) if float(temp[req]) > 0 else 1.0
        for j, tok in enumerate(c["drafts"][r]):
            for sh in shards:
                if tok >= 0:
                    sh.token_logit[req, j] = cache[req, j, tok].float()
                sh.lse[req, j] = torch.logsumexp(cache[req, j].float() / t, -1)


def sim_verify(shards, c):
    V = vp()
    logits = c["logits"]
    for sh in shards:
        V.stage_sampling_params(sh, c["temp"], c["seeds"])
    stats = [V.verify_stats_local(sh, logits[:, sh.start : sh.start + sh.local], c["draft_sampled"], c["expanded_idx"]).clone() for sh in shards]
    g1 = torch.stack(stats)
    resid = [
        V.verify_residual_local(sh, logits[:, sh.start : sh.start + sh.local], g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"]).clone()
        for sh in shards
    ]
    g2 = torch.stack(resid)
    outs = [V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], logits.shape[0]) for sh in shards]
    s0, n0 = outs[0]
    for s, n in outs[1:]:  # every rank must reach the same decision (tokens past num_sampled are unspecified)
        assert torch.equal(n, n0), "ranks disagree on num_sampled"
        for r in range(n0.shape[0]):
            k = int(n0[r])
            assert torch.equal(s[r, :k], s0[r, :k]), "ranks disagree on sampled tokens"
    return s0, n0


def stock_verify(c):
    sg = stock_gumbel()
    if sg is None:
        return None
    _, apply_temperature, rejection_sample = sg
    temp = c["temp"]
    tvals = temp[c["idx_mapping"].long()]
    processed = c["logits"]
    if bool(((tvals != 0) & (tvals != 1)).any()):  # Sampler.apply_sampling_params (sampler.py:238-265)
        processed = c["logits"].float().clone()
        apply_temperature(processed, c["expanded_idx"], temp)
    return rejection_sample(
        processed, c["cache"], c["draft_sampled"], c["cu"], c["pos"], c["idx_mapping"], c["expanded_idx"], c["local_pos"], temp, c["seeds"],
        c["n_spec"], use_block_verification=True,
    )


def ref_verify(c):
    from ds41h.ref.dspark import block_verify_ref

    out = []
    for r in range(c["num_reqs"]):
        req = int(c["idx_mapping"][r])
        s, n = c["cu_list"][r], c["num_draft"][r]
        t = float(c["temp"][req])
        # the reference is scalar python/numpy code: hand it host tensors
        tl = c["logits"][s : s + n + 1].float().cpu()
        if t not in (0.0, 1.0):
            tl = tl / t
        draft = c["cache"][req, :n].cpu() if n else None
        res = block_verify_ref(tl, draft, [int(v) for v in c["drafts"][r]], temperature=t, seed=int(c["seeds"][req]),
                               positions=c["pos"][s : s + n + 1].tolist())
        out.append((res["num_sampled"], res["tokens"]))
    return out


def compare(sampled, num_sampled, ref, label, allowed=0):
    bad = []
    for r, (n_ref, toks_ref) in enumerate(ref):
        n = int(num_sampled[r])
        if n != n_ref or sampled[r, :n].tolist() != list(toks_ref)[:n]:
            bad.append((r, n, sampled[r, :n].tolist(), n_ref, list(toks_ref)))
    assert len(bad) <= allowed, f"{label}: {len(bad)} requests differ: {bad[:4]}"
    return len(bad)


def test_vp_verify_matches_stock():
    require_cuda()
    total = 0
    for vocab, seed in [(8000, 0), (8000, 1), (129280, 2)]:
        c = verify_case(vocab, seed=seed)
        R = 256
        w = torch.zeros(vocab, R, dtype=torch.bfloat16, device="cuda")
        shards = make_shards(w, w, vocab, c["max_reqs"], c["n_spec"])
        fill_shard_draft_state(shards, c)
        sampled, num_sampled = sim_verify(shards, c)
        st = stock_verify(c)
        if st is not None:
            ss, ns = st
            ref = [(int(ns[r]), ss[r, : int(ns[r])].tolist()) for r in range(c["num_reqs"])]
            # identical math up to fp32 reduction order: a mismatch needs u within ~1e-6 of h
            total += compare(sampled, num_sampled, ref, f"vs stock rejection_sample (V={vocab}, seed={seed})", allowed=1)
        if vocab <= 8000:
            total += compare(sampled, num_sampled, ref_verify(c), f"vs ds41h.ref.dspark.block_verify_ref (V={vocab})", allowed=1)
    assert total <= 2, f"{total} near-threshold mismatches in total"



def test_fused_glue_matches_stock():
    """gather_and_stage == stock gather_draft_sampled + stage_sampling_params; verify_reject(counts=...) == verify_reject
    + stock get_num_sampled_and_rejected (incl. a chunked-prefill request)."""
    require_cuda()
    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled
    except Exception:  # noqa: BLE001
        print("SKIP test_fused_glue_matches_stock: vllm unavailable")
        return
    V = vp()
    vocab = 8000
    c = verify_case(vocab, seed=3)
    w = torch.zeros(vocab, 256, dtype=torch.bfloat16, device="cuda")
    shards = make_shards(w, w, vocab, c["max_reqs"], c["n_spec"])
    fill_shard_draft_state(shards, c)
    g = torch.Generator().manual_seed(7)
    num_tokens = 300
    n = c["logits"].shape[0]
    input_ids = torch.randint(0, vocab, (num_tokens,), generator=g, dtype=torch.int32).cuda()
    positions = torch.randint(0, 5000, (num_tokens,), generator=g, dtype=torch.int64).cuda()
    logits_indices = torch.randperm(num_tokens, generator=g)[:n].to(torch.int64).cuda()
    prefill_len = torch.randint(0, 6000, (c["max_reqs"],), generator=g, dtype=torch.int32).cuda()
    ds_ref, pos_ref = gather_draft_sampled(input_ids, positions, logits_indices, c["expanded_idx"], c["local_pos"], prefill_len)
    sh = shards[0]
    sh.temp.zero_()
    sh.seeds.zero_()
    ds, pos = V.gather_and_stage(sh, input_ids, positions, logits_indices, c["expanded_idx"], c["local_pos"], prefill_len, c["temp"], c["seeds"])
    assert torch.equal(ds, ds_ref) and torch.equal(pos, pos_ref), "gather_and_stage != stock gather_draft_sampled"
    m = min(c["temp"].shape[0], sh.max_reqs)
    assert torch.equal(sh.temp[:m], c["temp"][:m]) and torch.equal(sh.seeds[:m], c["seeds"][:m]), "staging differs"
    # fused counts (seq_lens < prefill_len marks chunked-prefill requests)
    logits = c["logits"]
    for x in shards:
        V.stage_sampling_params(x, c["temp"], c["seeds"])
    g1 = torch.stack([V.verify_stats_local(x, logits[:, x.start : x.start + x.local], c["draft_sampled"], c["expanded_idx"]).clone() for x in shards])
    g2 = torch.stack([V.verify_residual_local(x, logits[:, x.start : x.start + x.local], g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"],
                                              c["pos"]).clone() for x in shards])
    pl = torch.full((c["max_reqs"],), 100, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((c["num_reqs"],), 200, dtype=torch.int32, device="cuda")
    seq_lens[2] = 50  # chunked prefill
    s1, n1 = V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], n)
    n1, r1 = get_num_sampled_and_rejected(n1, seq_lens, c["cu"], c["idx_mapping"], pl)
    s2, n2, r2 = V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], n, counts=(seq_lens, pl))
    assert torch.equal(n1, n2) and torch.equal(r1, r2), f"fused counts differ: {n1.tolist()} {n2.tolist()} {r1.tolist()} {r2.tolist()}"
    for r in range(c["num_reqs"]):
        k = int(n1[r])
        assert torch.equal(s1[r, :k], s2[r, :k])
    assert int(n2[2]) == 0 and int(r2[2]) == 0


def test_post_fuse_matches_stock():
    """verify_reject(counts, post=...) (DS41H_SAMPLING_POST_FUSE) leaves exactly the state of verify_reject(counts) + stock
    post_update: num_computed_tokens, last_sampled_tokens, total_len, all_token_ids, output_bin_counts (greedy, T=1 and
    T!=1 rows, a short, a prefill-only, a placeholder-draft and a chunked-prefill request)."""
    require_cuda()
    try:
        from vllm.v1.worker.gpu.input_batch import post_update
    except Exception:  # noqa: BLE001
        print("SKIP test_post_fuse_matches_stock: vllm unavailable")
        return
    V = vp()
    for vocab, seed in ((8000, 3), (129280, 9)):
        c = verify_case(vocab, seed=seed)
        w = torch.zeros(vocab, 256, dtype=torch.bfloat16, device="cuda")
        shards = make_shards(w, w, vocab, c["max_reqs"], c["n_spec"])
        fill_shard_draft_state(shards, c)
        logits = c["logits"]
        n = logits.shape[0]
        for x in shards:
            V.stage_sampling_params(x, c["temp"], c["seeds"])
        g1 = torch.stack([V.verify_stats_local(x, logits[:, x.start : x.start + x.local], c["draft_sampled"], c["expanded_idx"]).clone() for x in shards])
        g2 = torch.stack([V.verify_residual_local(x, logits[:, x.start : x.start + x.local], g1, c["draft_sampled"], c["expanded_idx"],
                                                  c["local_pos"], c["pos"]).clone() for x in shards])
        g = torch.Generator().manual_seed(seed)
        R = c["max_reqs"]
        pl = torch.full((R,), 100, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((c["num_reqs"],), 200, dtype=torch.int32, device="cuda")
        seq_lens[2] = 50  # chunked prefill
        qlen = [c["cu_list"][r + 1] - c["cu_list"][r] for r in range(c["num_reqs"])]
        qlen[2] += 37  # a longer (chunked prefill) query
        qsl = torch.tensor([0] + list(np.cumsum(qlen)), dtype=torch.int32, device="cuda")
        state = dict(
            num_computed=torch.randint(150, 4000, (R,), generator=g, dtype=torch.int32).cuda(),
            last_sampled=torch.randint(0, vocab, (R, 1), generator=g, dtype=torch.int64).cuda(),
            bins=torch.randint(0, 3, (R, vocab), generator=g, dtype=torch.int32).cuda(),
            all_token_ids=torch.randint(0, vocab, (R, 5000), generator=g, dtype=torch.int32).cuda(),
            total_len=torch.randint(200, 4000, (R,), generator=g, dtype=torch.int32).cuda(),
        )
        ref = {k: v.clone() for k, v in state.items()}
        sh = shards[0]
        s1, n1, r1 = V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], n, counts=(seq_lens, pl))
        post_update(c["idx_mapping"], ref["num_computed"], ref["last_sampled"], ref["bins"], s1, n1, r1, qsl, ref["all_token_ids"], ref["total_len"])
        got = dict(state)
        got["query_start_loc"] = qsl
        s2, n2, r2 = V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], n, counts=(seq_lens, pl),
                                     post=got)
        torch.cuda.synchronize()
        assert torch.equal(n1, n2) and torch.equal(r1, r2), "post: counts differ"
        for k in ref:
            assert torch.equal(ref[k], state[k]), f"post-fused {k} differs from stock post_update (vocab {vocab}): {(ref[k] != state[k]).nonzero()[:8].tolist()}"
        assert int(n2[2]) == 0
    print("post fuse: request state bitwise equal to stock post_update")


def test_early_noise_matches():
    """verify_residual_local with the precomputed residual noise (DS41H_SAMPLING_NOISE_EARLY) == in-kernel noise, bitwise
    (greedy, T=1 and T!=1 rows; real vocab shard size)."""
    require_cuda()
    V = vp()
    V.NOISE_ROWS, V._NOISE_BUF = 64, {}  # test cases have up to ~50 rows (serving: <= 16)
    for vocab, num_reqs, seed in ((129280, 8, 5), (8000, 8, 11)):
        c = verify_case(vocab, seed=seed, num_reqs=num_reqs, temps=[0.0, 1.0, 0.7])
        w = torch.zeros(vocab, 256, dtype=torch.bfloat16, device="cuda")
        shards = make_shards(w, w, vocab, c["max_reqs"], c["n_spec"])
        fill_shard_draft_state(shards, c)
        logits = c["logits"]
        n = logits.shape[0]
        assert n <= V.NOISE_ROWS
        for x in shards:
            V.stage_sampling_params(x, c["temp"], c["seeds"])
        g1 = torch.stack([V.verify_stats_local(x, logits[:, x.start : x.start + x.local], c["draft_sampled"], c["expanded_idx"]).clone() for x in shards])
        for x in shards:
            lg = logits[:, x.start : x.start + x.local]
            ref = V.verify_residual_local(x, lg, g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"]).clone()
            nz = V.residual_noise(x, c["pos"], c["expanded_idx"], n)
            got = V.verify_residual_local(x, lg, g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"], nz).clone()
            assert torch.equal(ref, got), f"early-noise residual words differ (vocab {vocab}, rank {x.rank}): {(ref - got).abs().max()}"
    print("early noise: residual words bitwise equal")


# ---------------------------------------------------------------------------------------- statistics
def markov_tables(V, n_rows, seed, spread):
    g = torch.Generator().manual_seed(seed)
    # multiples of 1/8 in [-4, 4): exact in bf16
    T = (torch.randint(-32, 32, (n_rows, V, V), generator=g).float() / 8.0) * spread
    Q = T + torch.randint(-8, 8, (n_rows, V, V), generator=g).float() / 8.0  # draft = perturbed target
    return T, Q


def run_markov_batch(shards, T, Q, temp_value, B, ctx_token, gen, cpu, n_spec):
    """One batch of B requests through simulated VP verification; returns sampled, num_sampled."""
    V = T.shape[-1]
    t = temp_value if temp_value > 0 else 1.0
    dev = "cuda"
    Td, Qd = T.to(dev), Q.to(dev)
    # draft x_1..x_k ~ Q chain (prev_0 = ctx)
    prev = torch.full((B,), ctx_token, dtype=torch.long, device=dev)
    drafts, prevs = [], []
    for j in range(n_spec):
        q = Qd[j, prev]
        prevs.append(prev)
        x = q.argmax(-1) if temp_value == 0 else torch.multinomial(torch.softmax(q / t, -1), 1, generator=gen).squeeze(-1)
        drafts.append(x)
        prev = x
    drafts_t = torch.stack(drafts, 1)  # [B, k]
    num_logits = B * (n_spec + 1)
    rows_prev = torch.cat([torch.full((B, 1), ctx_token, device=dev, dtype=torch.long), drafts_t], 1)  # prev token of row j
    logits = Td[torch.arange(n_spec + 1, device=dev)[None, :].expand(B, -1), rows_prev].reshape(num_logits, V).to(torch.bfloat16)
    draft_sampled = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), drafts_t], 1).reshape(-1)
    req = torch.arange(B, device=dev, dtype=torch.int32)
    expanded_idx = req.repeat_interleave(n_spec + 1)
    local_pos = torch.arange(n_spec + 1, device=dev, dtype=torch.int32).repeat(B)
    p0 = torch.randint(0, 1 << 30, (B, 1), generator=gen, device=dev)
    pos = (p0 + torch.arange(n_spec + 1, device=dev)[None, :]).reshape(-1).to(torch.int64)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32) * (n_spec + 1)
    temp = torch.full((B,), temp_value, dtype=torch.float32, device=dev)
    seeds = torch.randint(-(2**62), 2**62, (B,), generator=gen, device=dev, dtype=torch.int64)
    cache = Qd[torch.arange(n_spec, device=dev)[None, :].expand(B, -1), torch.stack(prevs, 1)].to(torch.bfloat16)  # [B, k, V]
    for sh in shards:
        sh.logits[:B].copy_(cache[:, :, sh.start : sh.start + sh.local])
        sh.token_logit[:B] = torch.gather(cache.float(), 2, drafts_t[:, :, None]).squeeze(-1)
        sh.lse[:B] = torch.logsumexp(cache.float() / t, -1)
    c = dict(logits=logits, temp=temp, seeds=seeds, draft_sampled=draft_sampled, expanded_idx=expanded_idx, local_pos=local_pos, pos=pos, cu=cu,
             idx_mapping=req, num_reqs=B)
    return sim_verify(shards, c), drafts_t


def chi2_threshold(df: int, p: float = 1e-5) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(1 - p, df))
    except Exception:  # noqa: BLE001  Wilson-Hilferty approximation
        z = 4.265  # one-sided 1e-5
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def test_block_verification_distribution():
    require_cuda()
    V, n_spec, B, batches, ctx = 16, 3, 4096, 64, 5
    w = torch.zeros(V, 64, dtype=torch.bfloat16, device="cuda")
    shards = make_shards(w, w, V, B, n_spec)
    for temp_value, spread in [(1.0, 1.0), (0.7, 0.5)]:
        T, Q = markov_tables(V, n_spec + 1, seed=int(temp_value * 10), spread=spread)
        gen = torch.Generator(device="cuda").manual_seed(123)
        counts = torch.zeros(V, V, dtype=torch.float64)
        accepted_total = 0
        for _ in range(batches):
            (sampled, num_sampled), _ = run_markov_batch(shards, T, Q, temp_value, B, ctx, gen, None, n_spec)
            y1 = sampled[:, 0]
            y2 = sampled[:, 1].clone()
            need = num_sampled < 2  # continue from the target chain where the step emitted one token
            if bool(need.any()):
                p2 = torch.softmax(T.to("cuda")[1, y1[need]] / temp_value, -1)
                y2[need] = torch.multinomial(p2, 1, generator=gen).squeeze(-1)
            counts += torch.bincount((y1 * V + y2).cpu(), minlength=V * V).view(V, V).double()
            accepted_total += int((num_sampled - 1).sum())
        p1 = torch.softmax(T[0, ctx] / temp_value, -1).double()
        p2 = torch.softmax(T[1] / temp_value, -1).double()
        expected = p1[:, None] * p2 * (B * batches)
        mask = expected >= 5
        chi2 = float((((counts - expected) ** 2) / expected)[mask].sum())
        df = int(mask.sum()) - 1
        thr = chi2_threshold(df)
        print(f"  T={temp_value}: chi2={chi2:.1f} df={df} threshold={thr:.1f}, mean accepted={accepted_total / (B * batches):.2f}")
        assert chi2 < thr, f"T={temp_value}: output distribution deviates from the target chain (chi2 {chi2:.1f} >= {thr:.1f})"
        assert accepted_total > 0


def test_block_verification_greedy_exact():
    require_cuda()
    V, n_spec, B, ctx = 16, 3, 1024, 2
    w = torch.zeros(V, 64, dtype=torch.bfloat16, device="cuda")
    shards = make_shards(w, w, V, B, n_spec)
    T, Q = markov_tables(V, n_spec + 1, seed=9, spread=1.0)
    gen = torch.Generator(device="cuda").manual_seed(3)
    (sampled, num_sampled), drafts = run_markov_batch(shards, T, Q, 0.0, B, ctx, gen, None, n_spec)
    Td = T.to("cuda")
    for b in range(B):
        prev, n = ctx, int(num_sampled[b])
        for j in range(n):
            best = int(Td[j, prev].to(torch.bfloat16).float().argmax())
            assert int(sampled[b, j]) == best, f"greedy request {b} token {j}"
            if j < n - 1:
                assert int(drafts[b, j]) == best
            prev = best
        if n <= n_spec:  # the last token is the target's correction of a rejected draft
            assert int(drafts[b, n - 1]) != int(sampled[b, n - 1]), f"greedy request {b}: accepted draft reported as rejected"


# ---------------------------------------------------------------------------------------- bench
def run_bench():
    require_cuda()
    V = vp()
    vocab, R, n_spec = 129280, 256, 5
    local = vocab // TP
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(vocab, R, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(vocab, R, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    sg = stock_gumbel()
    for fp8 in (False, True):
        sh = make_shards(w1, w2, vocab, 16, n_spec, w2_fp8=fp8)[0]
        for rows in (1, 16):
            base = (torch.randn(rows, local, generator=g, device="cuda") * 3).to(torch.bfloat16)
            emb = w1[:rows].contiguous()
            idx = torch.arange(rows, device="cuda", dtype=torch.int64)
            pos = torch.full((rows,), 1000, device="cuda", dtype=torch.int64)
            temp = torch.ones(16, device="cuda")
            seeds = torch.arange(16, device="cuda", dtype=torch.int64)
            t = bench(lambda: V.draft_step_local(sh, base, emb, idx, pos, 0, temp, seeds))
            nbytes = local * R * (1 if fp8 else 2) + rows * local * 4
            print(fmt_bw(f"ds41h draft step/rank rows={rows} w2={'e4m3' if fp8 else 'bf16'}", t, nbytes))
    if sg is not None:
        for rows in (1, 16):
            base = (torch.randn(rows, vocab, generator=g, device="cuda") * 3).to(torch.bfloat16)
            emb = w1[:rows].contiguous()
            idx = torch.arange(rows, device="cuda", dtype=torch.int64)
            pos = torch.full((rows,), 1000, device="cuda", dtype=torch.int64)
            temp = torch.ones(16, device="cuda")
            seeds = torch.arange(16, device="cuda", dtype=torch.int64)
            cache = torch.zeros(16, n_spec, vocab, dtype=torch.bfloat16, device="cuda")
            col = torch.zeros((), dtype=torch.int32, device="cuda")

            def stock_step():
                li = base + torch.nn.functional.linear(emb, w2)
                return sg[0](li, idx, temp, seeds, pos, apply_temperature=True, is_drafting=True, logits_cache=cache, logits_cache_col=col)

            t = bench(stock_step)
            print(fmt_bw(f"stock draft step (full vocab, every rank) rows={rows}", t, vocab * R * 2 + rows * vocab * 6))
    for num_reqs in (1, 16):
        c = verify_case(vocab, n_spec=n_spec, num_reqs=max(num_reqs, 8), seed=1)
        w = torch.zeros(vocab, R, dtype=torch.bfloat16, device="cuda")
        shards = make_shards(w, w, vocab, c["max_reqs"], n_spec)
        fill_shard_draft_state(shards, c)
        sh = shards[0]
        logits = c["logits"][:, : sh.local]
        V.stage_sampling_params(sh, c["temp"], c["seeds"])
        stats = V.verify_stats_local(sh, logits, c["draft_sampled"], c["expanded_idx"])
        g1 = torch.stack([stats] * TP)
        resid = V.verify_residual_local(sh, logits, g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"])
        g2 = torch.stack([resid] * TP)

        def ours():
            V.verify_stats_local(sh, logits, c["draft_sampled"], c["expanded_idx"])
            V.verify_residual_local(sh, logits, g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"])
            V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"], c["pos"], c["num_reqs"], logits.shape[0])

        n = logits.shape[0]
        t = bench(ours)
        print(fmt_bw(f"ds41h verify kernels/rank (no gathers) logits={n}", t, n * sh.local * 2 * 2 + n * sh.local * 2))
        if sg is not None:
            t = bench(lambda: stock_verify(c))
            print(fmt_bw(f"stock rejection_sample full vocab logits={n}", t, n * vocab * 2 * 3))


def test_resid_pdl_matches():
    """DS41H_SAMPLING_RESID_PDL: the PDL-launched residual / reject kernels (gather-independent loads hoisted above the
    griddepcontrol.wait) give bitwise the same residual words and decisions as the plain launches (with and without noise)."""
    require_cuda()
    V = vp()
    if V._gdc_wait is None:
        print("SKIP: no triton gdc_wait")
        return
    for vocab, seed in [(8000, 3), (129280, 4)]:
        c = verify_case(vocab, seed=seed)
        R = 256
        w = torch.zeros(vocab, R, dtype=torch.bfloat16, device="cuda")
        shards = make_shards(w, w, vocab, c["max_reqs"], c["n_spec"])
        fill_shard_draft_state(shards, c)
        logits = c["logits"]
        for sh in shards:
            V.stage_sampling_params(sh, c["temp"], c["seeds"])
        stats = [V.verify_stats_local(sh, logits[:, sh.start : sh.start + sh.local], c["draft_sampled"], c["expanded_idx"]).clone()
                 for sh in shards]
        g1 = torch.stack(stats)
        for use_noise in (False, True):
            outs = {}
            for pdl in (False, True):
                res, dec = [], []
                for sh in shards:
                    lg = logits[:, sh.start : sh.start + sh.local]
                    nz = V.residual_noise(sh, c["pos"], c["expanded_idx"], lg.shape[0]) if use_noise else None
                    if use_noise and nz is None:
                        continue
                    res.append(V.verify_residual_local(sh, lg, g1, c["draft_sampled"], c["expanded_idx"], c["local_pos"], c["pos"], nz,
                                                       pdl=pdl).clone())
                g2 = torch.stack(res) if res else None
                if g2 is not None:
                    for sh in shards:
                        dec.append(tuple(t.clone() for t in V.verify_reject(sh, g1, g2, c["draft_sampled"], c["cu"], c["idx_mapping"],
                                                                             c["pos"], c["num_reqs"], logits.shape[0], pdl=pdl)))
                torch.cuda.synchronize()
                outs[pdl] = (res, dec)
            for a_, b_ in zip(outs[False][0], outs[True][0]):
                assert torch.equal(a_.view(torch.int32), b_.view(torch.int32)), f"residual words differ with PDL (V={vocab}, noise={use_noise})"
            for da, db in zip(outs[False][1], outs[True][1]):
                # (sampled, num_sampled): tokens past num_sampled are unspecified (torch.empty), as in sim_verify
                sa, na = da[0], da[1]
                sb, nb = db[0], db[1]
                assert torch.equal(na, nb), f"num_sampled differs with PDL (V={vocab}, noise={use_noise})"
                for r in range(na.shape[0]):
                    k = int(na[r])
                    assert torch.equal(sa[r, :k], sb[r, :k]), f"sampled tokens differ with PDL (V={vocab}, noise={use_noise}, req {r})"
    print("resid/reject PDL launches: PASS (bitwise == plain launches, with / without precomputed noise)")


def test_graph_out_alias():
    """DS41H_GRAPH_OUT_BATCH (patch_sampling.graph_out_alias): inside a captured graph the model outputs are written into the
    graph manager's persistent buffers by ONE batched copy and the stock store_capture_output copies become no-ops; after
    replay the buffers hold exactly the outputs (bitwise), and the replay runs 3 copy kernels fewer than the stock graph."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from ds41h import patch_sampling as ps

    class Mgr:  # the ModelCudaGraphManager attributes that store_capture_output uses
        pass

    dev = "cuda"
    N, n, H = 64, 8, 5120
    x = torch.randn(n, H, device=dev).to(torch.bfloat16)

    def model():
        return (x * 2).contiguous(), [(x + float(i)).contiguous() for i in range(3)]

    def make_mgr():
        m = Mgr()
        m.hidden_states = torch.empty(N, H, dtype=torch.bfloat16, device=dev)
        m.aux_hidden_states = [torch.empty(N, H, dtype=torch.bfloat16, device=dev) for _ in range(3)]
        return m

    def store(m, out):  # == vllm store_capture_output (last PP rank, aux hidden state outputs)
        hidden_states, aux_hidden_states = out
        m.hidden_states[:n] = hidden_states
        for i, aux in enumerate(aux_hidden_states):
            m.aux_hidden_states[i][:n] = aux

    def capture(m, alias: bool):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(st):
            store(m, model())  # warm-up (stock copies, eager)
            with torch.cuda.graph(g, stream=st):
                out = model()
                store(m, ps.graph_out_alias(m, out) if alias else out)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        return g

    def gpu_ops(g):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            g.replay()
            torch.cuda.synchronize()
        return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]

    ms, ma = make_mgr(), make_mgr()
    gs, ga = capture(ms, False), capture(ma, True)
    for m in (ms, ma):
        m.hidden_states.zero_()
        for b in m.aux_hidden_states:
            b.zero_()
    x.copy_(torch.randn(n, H, device=dev).to(torch.bfloat16))
    gs.replay()
    ga.replay()
    torch.cuda.synchronize()
    for m in (ms, ma):
        assert torch.equal(m.hidden_states[:n], x * 2)
        for i in range(3):
            assert torch.equal(m.aux_hidden_states[i][:n], x + float(i))
    assert torch.equal(ms.hidden_states[:n], ma.hidden_states[:n])
    os_, oa = gpu_ops(gs), gpu_ops(ga)
    assert len(oa) == len(os_) - 3, (os_, oa)
    # layout mismatch -> unchanged output (stock copies)
    mb = make_mgr()
    mb.aux_hidden_states = mb.aux_hidden_states[:2]
    out = model()
    assert ps.graph_out_alias(mb, out) is out
    print(f"graph-out alias: PASS (stock replay {len(os_)} GPU ops, batched {len(oa)}; buffers bitwise equal)")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_bench()
    else:
        sys.exit(run_module_tests(globals()))
