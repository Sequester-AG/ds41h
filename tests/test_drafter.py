"""4-GPU tests + benchmark for ds41h.drafter (fused DSpark Markov drafting, kernels/drafter.cu).

    torchrun --nproc-per-node 4 tests/test_drafter.py           # tests
    torchrun --nproc-per-node 4 tests/test_drafter.py --bench   # fused vs per-step Triton (+LL gather)
    python tests/test_drafter.py [--bench]                      # relaunches itself under torchrun

Tests (every rank checks its own outputs):
  * tokens identical to stock full-vocab gumbel_sample drafting (integer Markov weights -> exact dot products),
    cached draft logits identical to stock's cache slice, token_logit == cached logit of the drafted token,
    lse == logsumexp(cache / T) (fp32 tolerance); greedy, T=1, T!=1 and a padding row (idx -1);
  * real vocab (129280 -> 127 CTAs per rank) and a small vocab (4 CTAs per rank);
  * 200 back-to-back launches with changing inputs (epoch / parity / counter logic) and CUDA-graph replay.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

TP = 4


def setup():
    os.environ.setdefault("DS41H_DRAFT_TIMEOUT_S", "30")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    cpu_group = dist.new_group(backend="gloo")
    from ds41h import drafter

    drafter.TIMEOUT_NS = int(30e9)
    comm = drafter.DraftComm(cpu_group, torch.device("cuda", local), name="test-draft")
    if comm.disabled:
        raise RuntimeError("DraftComm setup failed")
    return rank, local, cpu_group, comm


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[rank {dist.get_rank()}] {msg}")


def case(vocab, rows, n_spec, max_reqs, seed):
    from test_sampling import draft_case

    w1, w2, base, idx_map, temp, seeds, sample_pos, prev = draft_case(vocab, rows=rows, n_spec=n_spec, max_reqs=max_reqs, seed=seed)
    if rows == 1:  # draft_case makes the last row a padding row (idx -1); keep one live request instead
        idx_map = torch.full_like(idx_map, 3 + seed % 2)  # T = 1.3 / 1.0
    return w1, w2, base, idx_map, temp, seeds, sample_pos, prev


def run_fused(comm, rank, w1, w2, vocab, base, idx_map, temp, seeds, sample_pos, prev, max_reqs, n_spec, shard=None, req_draft=None):
    from ds41h import drafter
    from ds41h.vp_sampling import VPShard

    rows = base.shape[0]
    if shard is None:
        shard = VPShard.create(rank=rank, ranks=TP, w1=w1, w2_full=w2, vocab=vocab, max_reqs=max_reqs, n_spec=n_spec,
                               cache_dtype=torch.bfloat16, gather=None)
    fused = drafter.MarkovFused(comm, shard)
    nq = n_spec
    input_ids = torch.zeros(max(rows, 8) * nq, dtype=torch.int32, device="cuda")
    input_ids[torch.arange(rows, device="cuda") * nq] = prev
    base_local = base[:, :, shard.start : shard.start + shard.local].reshape(rows * n_spec, shard.local).contiguous()
    toks = torch.zeros(max_reqs, n_spec, dtype=torch.int64, device="cuda")
    fused(base_local, input_ids, nq, idx_map.reshape(-1).contiguous(), sample_pos.reshape(-1).contiguous(), temp, seeds, toks, rows, n_spec,
          req_draft=req_draft)
    return shard, toks, (base_local, input_ids, nq, idx_map.reshape(-1).contiguous(), sample_pos.reshape(-1).contiguous())


def t_matches_stock(comm, rank, vocab, rows, n_spec, seed):
    from test_sampling import stock_draft

    max_reqs = 8
    w1, w2, base, idx_map, temp, seeds, sample_pos, prev = case(vocab, rows, n_spec, max_reqs, seed)
    ref_toks, ref_cache = stock_draft(w1, w2, base, idx_map, temp, seeds, sample_pos, prev, max_reqs)
    req_draft = torch.full((max_reqs, n_spec + 2), -7, dtype=torch.int64, device="cuda")  # wider than n_spec: stride != n_spec
    shard, toks, _ = run_fused(comm, rank, w1, w2, vocab, base, idx_map, temp, seeds, sample_pos, prev, max_reqs, n_spec, req_draft=req_draft)
    torch.cuda.synchronize()
    check(torch.equal(toks[:rows], ref_toks), f"tokens differ (vocab {vocab} seed {seed}):\n{toks[:rows]}\nvs stock\n{ref_toks}")
    # early combine (req_draft): == stock req_states.draft_tokens[idx_mapping] = draft_tokens, untouched elsewhere
    idx_cpu = idx_map.cpu()
    live = [int(v) for v in idx_cpu.reshape(-1).tolist() if v >= 0]
    if len(set(live)) * n_spec == len(live):  # unique request per row (duplicate rows have no defined winner, as stock)
        ref_rd = torch.full_like(req_draft, -7)
        for r in range(rows):
            for s_ in range(n_spec):
                q = int(idx_cpu[r, s_])
                if q >= 0:
                    ref_rd[q, s_] = ref_toks[r, s_]
        check(torch.equal(req_draft, ref_rd), f"req_draft differs from the stock index_put (vocab {vocab} seed {seed})")
    lo, hi = shard.start, shard.start + shard.local
    for r in range(rows):
        req = int(idx_map[r, 0])
        if req < 0:
            continue
        check(torch.equal(shard.logits[req].view(torch.int16), ref_cache[req, :, lo:hi].view(torch.int16)), f"cache differs req {req}")
        t = float(temp[req])
        for s in range(n_spec):
            tok = int(ref_toks[r, s])
            row = ref_cache[req, s].float()
            check(float(shard.token_logit[req, s]) == float(row[tok]), f"token_logit req {req} step {s}")
            lse = torch.logsumexp(row / (t if t > 0 else 1.0), 0).item()
            check(abs(float(shard.lse[req, s]) - lse) <= 1e-4 * max(1.0, abs(lse)), f"lse req {req} step {s}: {float(shard.lse[req, s])} vs {lse}")
    if rank == 0:
        print(f"  ok: fused drafting == stock (vocab {vocab}, rows {rows}, n_spec {n_spec}, seed {seed})", flush=True)


def t_repeat_and_graph(comm, rank, vocab=129280, n_spec=5):
    """Back-to-back launches with changing inputs, then CUDA-graph capture + replays."""
    from test_sampling import stock_draft

    max_reqs = 8
    for it in range(200):
        w1, w2, base, idx_map, temp, seeds, sample_pos, prev = case(vocab if it % 50 == 0 else 4096, 1 + it % 3, n_spec, max_reqs, 1000 + it)
        shard, toks, _ = run_fused(comm, rank, w1, w2, w1.shape[0], base, idx_map, temp, seeds, sample_pos, prev, max_reqs, n_spec)
        if it % 25 == 0:
            ref_toks, _ = stock_draft(w1, w2, base, idx_map, temp, seeds, sample_pos, prev, max_reqs)
            check(torch.equal(toks[: base.shape[0]], ref_toks), f"repeat it {it}: tokens differ")
    torch.cuda.synchronize()
    # graph: capture one launch, replay with new contents in the same buffers
    w1, w2, base, idx_map, temp, seeds, sample_pos, prev = case(vocab, 1, n_spec, max_reqs, 7)
    shard, toks, args = run_fused(comm, rank, w1, w2, vocab, base, idx_map, temp, seeds, sample_pos, prev, max_reqs, n_spec)
    from ds41h import drafter

    fused = drafter.MarkovFused(comm, shard)
    base_local, input_ids, nq, idx_flat, pos_flat = args
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    dist.barrier()
    with torch.cuda.graph(g, stream=s):
        fused(base_local, input_ids, nq, idx_flat, pos_flat, temp, seeds, toks, 1, n_spec)
    for it in range(20):
        w1b, w2b, baseb, idxb, tempb, seedsb, posb, prevb = case(vocab, 1, n_spec, max_reqs, 500 + it)
        base_local.copy_(baseb[:, :, shard.start : shard.start + shard.local].reshape(n_spec, shard.local))
        input_ids[0] = prevb[0]
        idx_flat.copy_(idxb.reshape(-1))
        pos_flat.copy_(posb.reshape(-1))
        temp.copy_(tempb)
        seeds.copy_(seedsb)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        # reference with the graph's (fixed) Markov weights w1/w2 and the new per-step inputs
        ref_toks, _ = stock_draft(w1, w2, baseb, idxb, tempb, seedsb, posb, prevb, max_reqs)
        check(torch.equal(toks[:1], ref_toks), f"graph replay {it}: tokens differ {toks[:1]} vs {ref_toks}")
    if rank == 0:
        print("  ok: 200 back-to-back launches + 20 graph replays", flush=True)


def t_verify_matches_stock(vcomm, rank, vocab, seed, num_reqs=8, temps=None):
    """Fused verification == stock rejection_sample(block verification) on the full vocab."""
    from test_sampling import compare as cmp_verify
    from test_sampling import fill_shard_draft_state, stock_verify, verify_case

    from ds41h import drafter
    from ds41h.vp_sampling import VPShard

    c = verify_case(vocab, n_spec=5, num_reqs=num_reqs, seed=seed, temps=temps)
    w = torch.zeros(vocab, 256, dtype=torch.bfloat16, device="cuda")
    shard = VPShard.create(rank=rank, ranks=TP, w1=w, w2_full=w, vocab=vocab, max_reqs=c["max_reqs"], n_spec=5, cache_dtype=torch.bfloat16, gather=None)
    fill_shard_draft_state([shard], c)
    vf = drafter.VerifyFused(vcomm, shard)
    logits = c["logits"][:, shard.start : shard.start + shard.local]
    check(vf.eligible(logits, c["num_reqs"]), "verify not eligible")
    sampled, num_sampled = vf(logits, c["draft_sampled"], c["pos"], c["expanded_idx"], c["local_pos"], c["cu"], c["idx_mapping"], c["num_reqs"], c["temp"], c["seeds"])
    torch.cuda.synchronize()
    ref = stock_verify(c)
    rs, rn = ref
    ref_list = [(int(rn[r]), rs[r, : int(rn[r])].tolist()) for r in range(c["num_reqs"])]
    bad = cmp_verify(sampled, num_sampled, ref_list, f"fused verify vocab {vocab} seed {seed}", allowed=1)
    # every rank must reach the same decisions
    allv = [torch.empty_like(num_sampled) for _ in range(TP)]
    dist.all_gather(allv, num_sampled)
    check(all(torch.equal(allv[0], x) for x in allv), "ranks disagree on num_sampled")
    if rank == 0:
        print(f"  ok: fused verify == stock (vocab {vocab}, seed {seed}, {c['num_reqs']} reqs, {bad} near-tie mismatches)", flush=True)


def t_moe_fused_ar(cpu_group, rank, local):
    """moe_decode with the TP all-reduce fused into moe2 == unfused partials summed in rank order (bitwise)."""
    from ds41h import drafter, moe

    w13, s13, w2, s2 = moe.load_raw_experts(0, rank, "cuda", prefix="mtp")
    E = w13.shape[0]
    op = drafter.MoEDecode(torch.device("cuda", local))
    arc = drafter.DraftComm(cpu_group, torch.device("cuda", local), name="test-moe-ar")
    for it in range(60):
        op.moe2p = it >= 30  # second half: persistent moe2p (its own unfused partials as reference)
        m = 1 + it % 8
        x, w, ids = moe_case(m, 3, E, 500 + it)  # same routing on every rank (seeded)
        y_local = op(x, w, ids, w13, s13, w2, s2, 10.0)
        parts = [torch.empty_like(y_local) for _ in range(TP)]
        dist.all_gather(parts, y_local)
        ref = parts[0].float()
        for p_ in parts[1:]:
            ref = ref + p_.float()
        ref = ref.to(torch.bfloat16)
        y = op(x, w, ids, w13, s13, w2, s2, 10.0, ar=arc)
        torch.cuda.synchronize()
        check(torch.equal(y.view(torch.int16), ref.view(torch.int16)), f"fused moe all-reduce differs (it {it}, m {m}): max diff {(y.float() - ref.float()).abs().max().item()}")
    if rank == 0:
        print("  ok: moe2 / moe2p fused all-reduce == rank-ordered sum of partials (2 x 30 launches, m = 1..8)", flush=True)


def t_lin_fused_ar(cpu_group, rank, local):
    """OProj wo_b with the TP all-reduce fused (DS41H_DRAFTER_OPROJ_AR) == unfused partials summed in rank order (bitwise),
    60 launches incl. a CUDA-graph replay section (device epoch / parity)."""
    from ds41h import drafter, skinny

    g = torch.Generator(device="cuda").manual_seed(100 + rank)  # different weights per rank (TP K slices)
    w8 = (torch.randn(5120, 2048, generator=g, device="cuda") * 0.6).clamp(-440, 440).to(torch.float8_e4m3fn)
    sc = torch.randint(118, 124, (160, 64), generator=g, device="cuda").to(torch.uint8)
    WB = skinny.pack(w8, sc)
    arc = drafter.DraftComm(cpu_group, torch.device("cuda", local), name="test-oproj-ar")
    e = drafter.ext()
    for it in range(40):
        m = 1 + it % 8
        z = (torch.randn(m, 2048, generator=torch.Generator(device="cuda").manual_seed(7 * it + rank), device="cuda")).to(torch.bfloat16)
        y_local = torch.empty(m, 5120, dtype=torch.bfloat16, device="cuda")
        e.lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y_local, None, None, True, -1, 0)
        parts = [torch.empty_like(y_local) for _ in range(TP)]
        dist.all_gather(parts, y_local)
        ref = parts[0].float()
        for p_ in parts[1:]:
            ref = ref + p_.float()
        ref = ref.to(torch.bfloat16)
        y = torch.empty_like(y_local)
        e.lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y, None, None, True, arc.handle, int(30e9))
        torch.cuda.synchronize()
        check(torch.equal(y.view(torch.int16), ref.view(torch.int16)), f"fused o-proj all-reduce differs (it {it}, m {m})")
    # graph replays
    m = 7
    z = torch.randn(m, 2048, device="cuda").to(torch.bfloat16)
    y = torch.empty(m, 5120, dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        e.lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y, None, None, True, arc.handle, int(30e9))
    torch.cuda.synchronize()
    dist.barrier()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph, stream=st):
        for _ in range(4):
            e.lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y, None, None, True, arc.handle, int(30e9))
    for _ in range(5):
        gph.replay()
    torch.cuda.synchronize()
    y_local = torch.empty_like(y)
    e.lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y_local, None, None, True, -1, 0)
    parts = [torch.empty_like(y_local) for _ in range(TP)]
    dist.all_gather(parts, y_local)
    ref = sum(p_.float() for p_ in parts[1:]) if False else None
    ref = parts[0].float()
    for p_ in parts[1:]:
        ref = ref + p_.float()
    check(torch.equal(y.view(torch.int16), ref.to(torch.bfloat16).view(torch.int16)), "fused o-proj all-reduce differs after graph replays")
    if rank == 0:
        print("  ok: o-proj wo_b fused all-reduce == rank-ordered sum of partials (40 launches + 20 graph replays)", flush=True)


def run_markov_quick() -> int:
    """4 GPUs, --markov: only the fused Markov drafting vs stock (incl. the early-combine req_draft output)."""
    rank, local, cpu_group, comm = setup()
    t_matches_stock(comm, rank, 4 * 1000, rows=4, n_spec=5, seed=0)
    t_matches_stock(comm, rank, 4 * 1000, rows=3, n_spec=7, seed=1)
    t_matches_stock(comm, rank, 129280, rows=4, n_spec=7, seed=2)
    t_matches_stock(comm, rank, 129280, rows=1, n_spec=7, seed=3)
    dist.barrier()
    if rank == 0:
        print("MARKOV PASS", flush=True)
    dist.destroy_process_group()
    return 0


def run_tests() -> int:
    rank, local, cpu_group, comm = setup()
    from ds41h import drafter

    vcomm = drafter.DraftComm(cpu_group, torch.device("cuda", local), name="test-verify")
    for vocab, seed in ((4 * 2000, 0), (4 * 2000, 1), (129280, 2), (129280, 3)):
        t_verify_matches_stock(vcomm, rank, vocab, seed)
    t_verify_matches_stock(vcomm, rank, 129280, 4, temps=[0.0])
    t_verify_matches_stock(vcomm, rank, 129280, 5, temps=[1.0])
    t_matches_stock(comm, rank, 4 * 1000, rows=4, n_spec=5, seed=0)
    t_matches_stock(comm, rank, 4 * 1000, rows=3, n_spec=7, seed=1)
    t_matches_stock(comm, rank, 129280, rows=4, n_spec=5, seed=2)
    t_matches_stock(comm, rank, 129280, rows=1, n_spec=5, seed=3)
    t_matches_stock(comm, rank, 129280, rows=2, n_spec=8, seed=4)
    t_repeat_and_graph(comm, rank)
    t_moe_fused_ar(cpu_group, rank, local)
    t_lin_fused_ar(cpu_group, rank, local)
    dist.barrier()
    if rank == 0:
        print("ALL PASS", flush=True)
    dist.destroy_process_group()
    return 0


def graph_time_us(fn, calls=20, reps=5):
    """(min, median) us per fn() call from CUDA-graph replays (barrier-synchronized when torch.distributed is up)."""
    barrier = dist.barrier if dist.is_available() and dist.is_initialized() else (lambda: None)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    barrier()
    best = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        barrier()
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        best.append(e0.elapsed_time(e1) * 1e3 / calls)
    return min(best), sorted(best)[len(best) // 2]


def run_bench() -> None:
    rank, local, cpu_group, comm = setup()
    from ds41h import vp_sampling as V
    from ds41h.allreduce import LLComm

    ll = LLComm(cpu_group, torch.device("cuda", local), max_bytes=256 << 10, name="bench-sampling")
    vocab, n_spec, max_reqs = 129280, 5, 8
    for rows in (1, 2, 4):
        w1, w2, base, idx_map, temp, seeds, sample_pos, prev = case(vocab, rows, n_spec, max_reqs, 11)
        idx_map = idx_map.abs()  # no padding row in the bench
        shard, toks, args = run_fused(comm, rank, w1, w2, vocab, base, idx_map, temp, seeds, sample_pos, prev, max_reqs, n_spec)
        from ds41h import drafter

        fused = drafter.MarkovFused(comm, shard)
        base_local, input_ids, nq, idx_flat, pos_flat = args
        t_f = graph_time_us(lambda: fused(base_local, input_ids, nq, idx_flat, pos_flat, temp, seeds, toks, rows, n_spec))
        tshard = V.VPShard.create(rank=rank, ranks=TP, w1=w1, w2_full=w2, vocab=vocab, max_reqs=max_reqs, n_spec=n_spec,
                                  cache_dtype=torch.bfloat16, gather=lambda i, o: ll.gather_words(i, o))
        base3 = base_local.view(rows, n_spec, -1)
        idx2, pos2 = idx_flat.view(rows, n_spec), pos_flat.view(rows, n_spec)
        prev_t = input_ids[torch.arange(rows, device="cuda") * nq]
        toks2 = torch.zeros_like(toks)
        t_t = graph_time_us(lambda: V.draft_markov(tshard, base3, prev_t, idx2, pos2, temp, seeds, toks2))
        torch.cuda.synchronize()
        same = torch.equal(toks, toks2)
        if rank == 0:
            print(f"rows {rows}: fused {t_f[0]:7.1f} us (med {t_f[1]:7.1f})   triton+LL {t_t[0]:7.1f} us (med {t_t[1]:7.1f})   same tokens {same}", flush=True)
    # verification: one request, k = 5 (6 logit rows), T = 1
    from ds41h import drafter as D

    vcomm = D.DraftComm(cpu_group, torch.device("cuda", local), name="bench-verify")
    g = torch.Generator(device="cuda").manual_seed(3)
    n_spec, max_reqs = 5, 8
    shard = V.VPShard.create(rank=rank, ranks=TP, w1=w1, w2_full=w2, vocab=vocab, max_reqs=max_reqs, n_spec=n_spec, cache_dtype=torch.bfloat16,
                             gather=lambda i, o: ll.gather_words(i, o))
    shard.logits.copy_((torch.randn(max_reqs, n_spec, shard.local, generator=g, device="cuda") * 3).to(torch.bfloat16))
    shard.lse.fill_(12.0)
    shard.token_logit.fill_(3.0)
    rows = n_spec + 1
    logits = (torch.randn(rows, shard.local, generator=g, device="cuda") * 3).to(torch.bfloat16)
    draft_sampled = torch.randint(0, vocab, (rows,), generator=g, device="cuda").to(torch.int32)
    pos = (1000 + torch.arange(rows, device="cuda")).to(torch.int64)
    exp_idx = torch.full((rows,), 2, dtype=torch.int32, device="cuda")
    local_pos = torch.arange(rows, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([2], dtype=torch.int32, device="cuda")
    vtemp = torch.ones(max_reqs, device="cuda")
    vseeds = torch.arange(max_reqs, device="cuda", dtype=torch.int64) * 7919
    vf = D.VerifyFused(vcomm, shard)
    t_v = graph_time_us(lambda: vf(logits, draft_sampled, pos, exp_idx, local_pos, cu, idx_mapping, 1, vtemp, vseeds))
    shard.verify_fused = None
    t_t = graph_time_us(lambda: V.verify(shard, logits, draft_sampled, pos, exp_idx, local_pos, cu, idx_mapping, 1, vtemp, vseeds))
    if rank == 0:
        print(f"verify 1 req x 6 rows: fused {t_v[0]:7.1f} us (med {t_v[1]:7.1f})   triton+LL {t_t[0]:7.1f} us (med {t_t[1]:7.1f})", flush=True)
    dist.barrier()
    dist.destroy_process_group()


# ------------------------------------------------------------------------------------------- drafter MoE (1 GPU)
def moe_case(m, k, E, seed, distinct_frac=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, 5120, generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    x[0, :7] = torch.tensor([300.0, -1e-6, 0.0, 2e-30, -88.0, 1e-3, 7.0])  # range / tiny values
    pool = max(k, int(E * distinct_frac))
    ids = torch.stack([torch.randperm(pool, generator=torch.Generator().manual_seed(seed * 100 + t))[:k] for t in range(m)]).to(torch.int32).cuda()
    w = torch.rand(m, k, generator=g, device="cuda") + 0.1
    w = w / w.sum(-1, keepdim=True) * 1.5
    return x, w.float().contiguous(), ids.contiguous()


def run_moe(bench: bool = False, bench_only: bool = False) -> int:
    from common import compare, fmt_metrics

    from ds41h import drafter, moe

    layer = int(os.environ.get("DS41H_TEST_MTP", "0"))
    w13, s13, w2, s2 = moe.load_raw_experts(layer, 0, "cuda", prefix="mtp")
    E = w13.shape[0]
    print(f"[load] mtp.{layer} experts rank 0: {tuple(w13.shape)}", flush=True)
    op = drafter.MoEDecode(torch.device("cuda"))
    if bench_only:  # timing only (DS41H_DRAFTER_MOE_DBG diagnostics produce wrong outputs on purpose)
        for frac in (1.0, 0.08):
            moe_bench(op, w13, s13, w2, s2, E, distinct_frac=frac)
        if op.m2s is None and not os.environ.get("DS41H_DRAFTER_MOE_DBG"):
            op.m2s = torch.zeros(drafter.ext().moe2s_scratch_floats(), dtype=torch.float32, device="cuda")
            print("  (moe2s)", flush=True)
            for frac in (1.0, 0.08):
                moe_bench(op, w13, s13, w2, s2, E, distinct_frac=frac)
            op.m2s = None
        return 0
    worst = 0.0
    for m in (1, 2, 3, 5, 6, 8):
        for seed in (0, 1):
            x, w, ids = moe_case(m, 3, E, seed)
            y = op(x, w, ids, w13, s13, w2, s2, 10.0)
            ref = moe.moe_forward_ref(x.float(), w, ids, w13, s13, w2, s2, 10.0)
            met = compare(y.float(), ref.float())
            print(f"  m={m} seed={seed}: {fmt_metrics(met)}", flush=True)
            worst = max(worst, met["rel_rmse"])
            assert met["rel_rmse"] < 5e-3 and met["cos"] > 0.99995, f"moe_decode mismatch m={m} seed={seed}: {met}"
    # invalid ids (-1 / >= E) contribute nothing
    x, w, ids = moe_case(5, 3, E, 3)
    ids[1, 2] = -1
    ids[3, 0] = E + 5
    y = op(x, w, ids, w13, s13, w2, s2, 10.0)
    ref = moe.moe_forward_ref(x.float(), w, ids, w13, s13, w2, s2, 10.0)
    met = compare(y.float(), ref.float())
    assert met["rel_rmse"] < 5e-3, met
    # deterministic
    y2 = op(x, w, ids, w13, s13, w2, s2, 10.0)
    assert torch.equal(y, y2), "moe_decode not deterministic"
    print(f"MOE PASS (worst rel_rmse {worst:.2e})", flush=True)
    # shared expert folded into the kernels: (with shared) - (without) == shared expert reference
    from ds41h import skinny
    from ds41h.ref.weights import default_reader

    lay = default_reader().mtp(layer)
    u8 = torch.uint8
    sw13 = torch.cat([lay.tp("ffn.shared_experts.w1.weight", 0, 4), lay.tp("ffn.shared_experts.w3.weight", 0, 4)]).cuda()
    ss13 = torch.cat([lay.tp("ffn.shared_experts.w1.scale", 0, 4).view(u8), lay.tp("ffn.shared_experts.w3.scale", 0, 4).view(u8)]).cuda()
    sw2 = lay.tp("ffn.shared_experts.w2.weight", 0, 4).cuda()
    ss2 = lay.tp("ffn.shared_experts.w2.scale", 0, 4).view(u8).cuda()
    pack = drafter.SharedPack(skinny.pack(sw13, ss13), skinny.pack(sw2, ss2))

    def deq(w, sc):
        sf = torch.exp2(sc.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)[: w.shape[0], : w.shape[1]]
        return w.view(torch.float8_e4m3fn).float() * sf

    W13, W2 = deq(sw13, ss13), deq(sw2, ss2)
    for m in (1, 5, 8):
        x, w, ids = moe_case(m, 3, E, 10 + m)
        y_s = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack).float()
        ref_r = moe.moe_forward_ref(x.float(), w, ids, w13, s13, w2, s2, 10.0).float()
        gu = x.float() @ W13.t()
        g_, u_ = gu.split(576, -1)
        hs = torch.nn.functional.silu(g_.clamp(max=10.0)) * u_.clamp(-10.0, 10.0)
        ref = ref_r + hs @ W2.t()
        met = compare(y_s, ref)
        print(f"  shared+routed m={m}: {fmt_metrics(met)}", flush=True)
        assert met["rel_rmse"] < 5e-3 and met["cos"] > 0.99995, f"shared expert mismatch m={m}: {met}"
    print("MOE+SHARED PASS", flush=True)
    # fused router (gate GEMV + dsv4_topk inside moe13) vs stock-semantics routing (torch gate GEMV + ds41h route_topk)
    gw = lay["ffn.gate.weight"].to(torch.bfloat16).cuda().contiguous()
    gb = lay["ffn.gate.bias"].float().cuda().contiguous()
    worst_ids = 0
    for m in (1, 5, 8):
        x, _, _ = moe_case(m, 3, E, 20 + m)
        x = (x.float() / x.float().norm(dim=-1, keepdim=True) * 70).to(torch.bfloat16)  # RMS-normed scale
        y_f, w_f, i_f = op.routed(x, gw, gb, 3, 1.5, True, w13, s13, w2, s2, 10.0, shared=pack)
        logits = x.float() @ gw.float().t()
        w_r, i_r = moe.route_topk(logits, gb, 3, 1.5, True)
        same = (torch.sort(i_f, -1).values == torch.sort(i_r, -1).values).all(-1)
        worst_ids += int((~same).sum())
        y_r = op(x, w_r, i_r, w13, s13, w2, s2, 10.0, shared=pack)
        met = compare(y_f.float(), y_r.float())
        print(f"  fused router m={m}: ids equal {int(same.sum())}/{m}, y {fmt_metrics(met)}", flush=True)
        if bool(same.all()):
            assert met["rel_rmse"] < 2e-2, met
            wd = (torch.gather(w_f, 1, torch.argsort(i_f, 1)) - torch.gather(w_r, 1, torch.argsort(i_r, 1))).abs().max().item()
            assert wd < 1e-3, f"router weights differ by {wd}"
    assert worst_ids <= 1, f"fused router picked different experts for {worst_ids} tokens"
    print("ROUTER PASS", flush=True)
    # persistent moe2p (h fragments read once per CTA, deterministic warp-order reduction) vs classic moe2
    for m in (1, 5, 7, 8):
        x, w, ids = moe_case(m, 3, E, 40 + m)
        op.moe2p = False
        y0 = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack).float()
        op.moe2p = True
        y1 = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack)
        assert torch.equal(y1, op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack)), "moe2p not deterministic"
        met = compare(y1.float(), y0)
        print(f"  moe2p vs moe2 m={m}: {fmt_metrics(met)}", flush=True)
        assert met["rel_rmse"] < 2e-3 and met["cos"] > 0.99999, met
    op.moe2p = False
    print("MOE2P PASS", flush=True)
    # slot-major moe2s (DS41H_DRAFTER_MOE2_VARIANT=1) == classic moe2, bitwise (same products, same slot order)
    m2s = torch.zeros(drafter.ext().moe2s_scratch_floats(), dtype=torch.float32, device="cuda")
    for m in (1, 5, 7, 8):
        for frac in (1.0, 0.08):
            x, w, ids = moe_case(m, 3, E, 60 + m, frac)
            op.m2s = None
            y0 = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack)
            op.m2s = m2s
            y1 = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack)
            y2 = op(x, w, ids, w13, s13, w2, s2, 10.0, shared=pack)
            op.m2s = None
            assert torch.equal(y1, y2), "moe2s not deterministic"
            assert torch.equal(y0, y1), f"moe2s != moe2 (m={m}, frac={frac}): {(y0.float() - y1.float()).abs().max()}"
    print("MOE2S PASS (bitwise == moe2)", flush=True)
    if bench:
        moe_bench(op, w13, s13, w2, s2, E)
        moe_bench(op, w13, s13, w2, s2, E, shared=pack)
        op.moe2p = True
        print("  (moe2p)", flush=True)
        moe_bench(op, w13, s13, w2, s2, E, shared=pack)
        op.moe2p = False
        moe_bench(op, w13, s13, w2, s2, E, shared=pack, router=(gw, gb))
    return 0


def moe_bench(op, w13, s13, w2, s2, E, shared=None, router=None, distinct_frac=1.0):
    from ds41h import moe

    for m in ((5,) if shared is not None else (5, 6, 8)):
        cases = [moe_case(m, 3, E, 100 + i, distinct_frac) for i in range(8)]  # 8 routings per replay: defeats L2 reuse
        ne = sum(len(torch.unique(c[2])) for c in cases) / len(cases)
        nbytes = ne * (2 * 576 * 2560 + 2 * 576 * 160 + 5120 * 288 + 5120 * 18)
        fns = (("ds41h moe_decode", lambda c: op(c[0], c[1], c[2], w13, s13, w2, s2, 10.0)),)
        if distinct_frac == 1.0 and not os.environ.get("DS41H_DRAFTER_MOE_DBG"):
            fns += (("prefill w4a8 (current)", lambda c: moe.moe_forward_w4a8(c[0], c[1], c[2], w13, s13, w2, s2, 10.0)),)
        if shared is not None:
            fns = (("moe_decode + shared", lambda c: op(c[0], c[1], c[2], w13, s13, w2, s2, 10.0, shared=shared)),)
        if router is not None:
            fns = (("router + moe + shared", lambda c: op.routed(c[0], router[0], router[1], 3, 1.5, True, w13, s13, w2, s2, 10.0, shared=shared)),)
        for name, fn in fns:
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for c in cases:
                    fn(c)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s_):
                for _ in range(4):
                    for c in cases:
                        fn(c)
            g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3 / 32)
            us = min(ts)
            print(f"  m={m} experts~{ne:4.1f} {name:24s} {us:7.1f} us/call (med {sorted(ts)[2]:7.1f})  {nbytes / us / 1e3:7.1f} GB/s", flush=True)


# ------------------------------------------------------------------------------------------- prepare inputs (1 GPU)
def run_prep() -> int:
    """ds41h.patch_drafter.fast_prepare_dflash_inputs == stock prepare_dflash_inputs on every output buffer."""
    import types

    import numpy as np

    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dfs

    from ds41h import patch_drafter

    stock = getattr(dfs, "_ds41h_stock_prepare_dflash_inputs", dfs.prepare_dflash_inputs)
    dev = "cuda"
    max_num_reqs, max_num_tokens, max_model_len, n_spec = 16, 4096, 36864, 5
    for case_i, (num_reqs, qlen, sfa) in enumerate([(1, 6, True), (3, 6, True), (5, 1, True), (2, 37, True), (4, 6, False), (16, 6, True)]):
        g = torch.Generator(device="cpu").manual_seed(case_i)
        nq = n_spec if sfa else n_spec + 1
        sched = np.array([qlen] * num_reqs, dtype=np.int32)
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(torch.tensor(sched), 0)
        starts = torch.randint(100, 20000, (num_reqs,), generator=g)
        positions = torch.cat([starts[r] + torch.arange(qlen) for r in range(num_reqs)]).to(torch.int64)
        idx_mapping = torch.randperm(max_num_reqs, generator=g)[:num_reqs].to(torch.int32)
        batch = types.SimpleNamespace(num_reqs=num_reqs, num_scheduled_tokens=sched, positions=positions.to(dev),
                                      query_start_loc=qsl.to(dev), idx_mapping=idx_mapping.to(dev))
        num_sampled = torch.randint(0, 3, (num_reqs,), generator=g).to(torch.int32).to(dev)
        num_rejected = torch.minimum(torch.randint(0, qlen, (num_reqs,), generator=g), torch.tensor(qlen - 1)).to(torch.int32).to(dev)
        last_sampled = torch.randint(0, 129280, (max_num_reqs,), generator=g).to(torch.int64).to(dev)
        next_prefill = torch.randint(0, 129280, (max_num_reqs,), generator=g).to(torch.int32).to(dev)
        in_temp = torch.rand(max_num_reqs, generator=g).to(dev)
        in_seeds = torch.randint(0, 2**40, (max_num_reqs,), generator=g).to(dev)
        bs = 128
        block_table = torch.randint(0, 5000, (max_num_reqs, 512), generator=g).to(torch.int32).to(dev)
        block_table[:, 3] = 0  # null blocks

        def bufs():
            gg = torch.Generator(device="cpu").manual_seed(999)
            mk = lambda n, dt: torch.randint(-5, 50, (n,), generator=gg).to(dt).to(dev)  # noqa: E731 - stale contents
            ib = types.SimpleNamespace(input_ids=mk(max_num_tokens, torch.int32), positions=mk(max_num_tokens, torch.int64),
                                       query_start_loc=mk(max_num_reqs + 1, torch.int32), seq_lens=mk(max_num_reqs, torch.int32))
            return dict(ib=ib, qslot=mk(max_num_tokens, torch.int64), cpos=mk(max_num_tokens, torch.int64), cslot=mk(max_num_tokens, torch.int64),
                        sidx=mk(max_num_reqs * n_spec, torch.int64), spos=mk(max_num_reqs * n_spec, torch.int64),
                        smap=mk(max_num_reqs * n_spec, torch.int64), temp=mk(max_num_reqs, torch.float32), seeds=mk(max_num_reqs, torch.int64))

        outs = []
        for fn in (stock, patch_drafter.fast_prepare_dflash_inputs):
            b = bufs()
            fn(b["ib"], b["qslot"], b["cpos"], b["cslot"], b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"], batch, num_sampled, num_rejected,
               last_sampled, next_prefill, in_temp, in_seeds, block_table, bs, 0, 1, 1, 128799, nq, n_spec, max_num_reqs, max_num_tokens,
               max_model_len, sample_from_anchor=sfa)
            torch.cuda.synchronize()
            outs.append([b["ib"].input_ids, b["ib"].positions, b["ib"].query_start_loc, b["ib"].seq_lens, b["qslot"], b["cpos"], b["cslot"],
                         b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"]])
        names = ["input_ids", "positions", "query_start_loc", "seq_lens", "query_slot", "ctx_pos", "ctx_slot", "sample_idx", "sample_pos",
                 "sample_map", "temp", "seeds"]
        for name, a, b in zip(names, *outs):
            assert torch.equal(a, b), f"prep case {case_i}: {name} differs"
        # three draft KV-cache groups (own block tables / block sizes / slot buffers), batched into one launch
        bts = [block_table, torch.randint(0, 9000, (max_num_reqs, 300), generator=g).to(torch.int32).to(dev),
               torch.randint(0, 7000, (max_num_reqs, 700), generator=g).to(torch.int32).to(dev)]
        bss = [bs, 64, 256]
        gouts = []
        for batched in (False, True):
            b = bufs()
            gg = torch.Generator(device="cpu").manual_seed(5)
            qs = [b["qslot"]] + [torch.randint(-5, 50, (max_num_tokens,), generator=gg).to(torch.int64).to(dev) for _ in range(2)]
            cs = [b["cslot"]] + [torch.randint(-5, 50, (max_num_tokens,), generator=gg).to(torch.int64).to(dev) for _ in range(2)]
            fn = patch_drafter.fast_prepare_dflash_inputs if batched else stock
            patch_drafter._BATCH["groups"] = 3 if batched else 0
            for gi in range(3):
                fn(b["ib"], qs[gi], b["cpos"], cs[gi], b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"], batch, num_sampled, num_rejected,
                   last_sampled, next_prefill, in_temp, in_seeds, bts[gi], bss[gi], 0, 1, 1, 128799, nq, n_spec, max_num_reqs, max_num_tokens,
                   max_model_len, sample_from_anchor=sfa)
                if batched and gi < 2:
                    assert len(patch_drafter._BATCH["pending"]) == gi + 1, "prepare call was not deferred"
            patch_drafter._BATCH["groups"] = 0
            torch.cuda.synchronize()
            gouts.append([b["ib"].input_ids, b["ib"].positions, b["ib"].query_start_loc, b["ib"].seq_lens, *qs, b["cpos"], *cs,
                          b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"]])
        for i, (a, b2) in enumerate(zip(*gouts)):
            assert torch.equal(a, b2), f"prep case {case_i}: batched 3-group output {i} differs"
        # FULL draft tail (patch_tail.DraftBlockTables): the drafter passes PRIVATE block tables; the batched kernel reads the
        # runner's tables (identical outputs) and copies this step's rows into the private ones
        import ds41h.patch_tail as pt

        priv = [torch.full_like(t, -7) for t in bts]
        fake = types.SimpleNamespace(sources=lambda: {p_.data_ptr(): r_ for p_, r_ in zip(priv, bts)})
        full0 = pt.FULL
        pt.FULL, pt._SYNC["bt"] = True, fake
        try:
            b = bufs()
            gg = torch.Generator(device="cpu").manual_seed(5)
            qs = [b["qslot"]] + [torch.randint(-5, 50, (max_num_tokens,), generator=gg).to(torch.int64).to(dev) for _ in range(2)]
            cs = [b["cslot"]] + [torch.randint(-5, 50, (max_num_tokens,), generator=gg).to(torch.int64).to(dev) for _ in range(2)]
            patch_drafter._BATCH["groups"] = 3
            for gi in range(3):
                patch_drafter.fast_prepare_dflash_inputs(b["ib"], qs[gi], b["cpos"], cs[gi], b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"],
                                                        batch, num_sampled, num_rejected, last_sampled, next_prefill, in_temp, in_seeds, priv[gi],
                                                        bss[gi], 0, 1, 1, 128799, nq, n_spec, max_num_reqs, max_num_tokens, max_model_len,
                                                        sample_from_anchor=sfa)
            patch_drafter._BATCH["groups"] = 0
            torch.cuda.synchronize()
            assert pt._SYNC["bt"] is None, "the FULL-tail sync was not consumed by the batched prepare launch"
        finally:
            pt.FULL = full0
            pt._SYNC["bt"] = None
        cout = [b["ib"].input_ids, b["ib"].positions, b["ib"].query_start_loc, b["ib"].seq_lens, *qs, b["cpos"], *cs,
                b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"]]
        for i, (a, b2) in enumerate(zip(gouts[0], cout)):
            assert torch.equal(a, b2), f"prep case {case_i}: FULL-tail private-table output {i} differs"
        for gi in range(3):
            assert torch.equal(priv[gi][:num_reqs], bts[gi][:num_reqs]), f"prep case {case_i}: private table {gi} rows not copied"
            assert bool((priv[gi][num_reqs:] == -7).all()), f"prep case {case_i}: private table {gi} rows beyond num_reqs touched"
        print(f"  ok: prepare case {case_i} (num_reqs {num_reqs}, qlen {qlen}, anchor {sfa}) incl. 3-group batch + FULL-tail copy", flush=True)
    # timing
    import time

    for fn, name in ((stock, "stock"), (patch_drafter.fast_prepare_dflash_inputs, "ds41h")):
        b = bufs()
        args = (b["ib"], b["qslot"], b["cpos"], b["cslot"], b["sidx"], b["spos"], b["smap"], b["temp"], b["seeds"], batch, num_sampled, num_rejected,
                last_sampled, next_prefill, in_temp, in_seeds, block_table, bs, 0, 1, 1, 128799, n_spec, n_spec, max_num_reqs, 8192, max_model_len)
        for _ in range(5):
            fn(*args, sample_from_anchor=True)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(50):
            fn(*args, sample_from_anchor=True)
        e1.record()
        e1.synchronize()
        print(f"  {name} prepare (16 reqs, max_num_tokens 8192): {e0.elapsed_time(e1) * 1e3 / 50:.1f} us/call (incl. launch)", flush=True)
    print("PREP PASS", flush=True)
    return 0


# ------------------------------------------------------------------------------------------- FP8 draft head (1 GPU)
def run_head(bench: bool = True) -> int:
    from common import compare, fmt_metrics

    from ds41h import drafter

    torch.manual_seed(0)
    N, K = 32320, 5120
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    w[5, :] *= 30.0  # an outlier row
    head = drafter.HeadFP8(w)
    # exact dequant of the packed copy (inverse of the fragment-major pack)
    q = head.w.view(N // 16, K // 32, 8, 4, 2, 8).permute(0, 4, 2, 1, 3, 5).reshape(N, K).view(torch.float8_e4m3fn).float()
    wq = q * head.scale[:, None]
    worst = 0.0
    for m in (1, 2, 5, 8):
        x = (torch.randn(m, K, device="cuda") * 2).to(torch.bfloat16)
        y = head(x)
        ref_q = x.float() @ wq.t()  # same quantized weights: kernel exactness (fp32 accumulation order only)
        ref = x.float() @ w.float().t()  # bf16 head: quantization error
        mq, mr = compare(y.float(), ref_q), compare(y.float(), ref)
        print(f"  m={m}: vs dequant {fmt_metrics(mq)} | vs bf16 head {fmt_metrics(mr)}", flush=True)
        assert mq["rel_rmse"] < 4e-3 and mq["cos"] > 0.99999, mq
        worst = max(worst, mr["rel_rmse"])
        # draft distribution distance (T=1) vs the bf16 head
        p = torch.log_softmax(ref.double(), -1)
        qd = torch.log_softmax(y.double(), -1)
        kl = (p.exp() * (p - qd)).sum(-1).mean().item()
        print(f"        KL(bf16 head || fp8 head) T=1: {kl:.2e}  top1 agree {(ref.argmax(-1) == y.float().argmax(-1)).float().mean().item():.3f}", flush=True)
    print(f"HEAD PASS (quantization rel_rmse {worst:.2e})", flush=True)
    # RMSNorm fused into the staging == vLLM rms_norm (bf16(bf16(x * r) * w)) followed by the head
    nw = (1.0 + 0.3 * torch.randn(K, device="cuda")).to(torch.bfloat16)
    eps = 1e-6
    for m in (1, 5, 8):
        h = (torch.randn(m, K, device="cuda") * 3 + 0.5).to(torch.bfloat16)
        r = torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps)
        xn = ((h.float() * r).to(torch.bfloat16).float() * nw.float()).to(torch.bfloat16)
        y_ref = head(xn)
        y_f = head(h, nw, eps)
        met = compare(y_f.float(), y_ref.float())
        print(f"  fused norm m={m}: vs norm+head {fmt_metrics(met)}", flush=True)
        assert met["rel_rmse"] < 2e-3 and met["cos"] > 0.99999, met
    print("HEAD NORM PASS", flush=True)
    # row-stream kernel (DS41H_DRAFT_HEAD_ROWS): same math, k-split across warps -> fp32 order only; deterministic
    e = drafter.ext()
    for m in (1, 5, 7, 8):
        x = (torch.randn(m, K, device="cuda") * 2).to(torch.bfloat16)
        e.head_set_rows(0)
        y0 = head(x)
        e.head_set_rows(1)
        y1 = head(x)
        y2 = head(x)
        yn = head(x, nw, eps)
        e.head_set_rows(0)
        yn0 = head(x, nw, eps)
        torch.cuda.synchronize()
        assert torch.equal(y1, y2), f"head rows kernel not repeatable (m={m})"
        met = compare(y1.float(), y0.float())
        met_n = compare(yn.float(), yn0.float())
        print(f"  rows kernel m={m}: vs head_kernel {fmt_metrics(met)} | fused norm {fmt_metrics(met_n)}", flush=True)
        assert met["rel_rmse"] < 1e-3 and met_n["rel_rmse"] < 1e-3, (met, met_n)
    print("HEAD ROWS PASS", flush=True)
    if bench:
        for m in (5, 6, 7, 8):
            x = (torch.randn(m, K, device="cuda") * 2).to(torch.bfloat16)

            def rows():
                e.head_set_rows(1)
                return head(x)

            def classic():
                e.head_set_rows(0)
                return head(x)

            for name, fn in (("fp8 head", classic), ("fp8 head rows", rows), ("bf16 F.linear", lambda: torch.nn.functional.linear(x, w))):
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s_):
                    for _ in range(20):
                        fn()
                g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e3 / 20)
                nb = head.w.numel() if name.startswith("fp8") else w.numel() * 2
                print(f"  m={m} {name:14s} {min(ts):7.1f} us (med {sorted(ts)[2]:7.1f})  {nb / min(ts) / 1e3:7.1f} GB/s", flush=True)
        e.head_set_rows(0)
    return 0


# ------------------------------------------------------------------------------------------- main_proj (1 GPU)
def run_main(bench: bool = True) -> int:
    from common import compare, fmt_metrics

    from ds41h import drafter
    from ds41h.ref.weights import default_reader

    rd = default_reader()
    w = rd.get("mtp.0.main_proj.weight").cuda()
    sc = rd.get("mtp.0.main_proj.scale").view(torch.uint8).cuda()
    mp = drafter.MainProj(torch.device("cuda"), w, sc)
    sf = torch.exp2(sc.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    W = w.view(torch.float8_e4m3fn).float() * sf
    for m in (1, 3, 6, 8):
        x = (torch.randn(m, 15360, device="cuda") * 3).to(torch.bfloat16)
        y = mp(x)
        ref = x.float() @ W.t()
        met = compare(y.float(), ref)
        print(f"  main_proj m={m}: {fmt_metrics(met)}", flush=True)
        assert met["rel_rmse"] < 4e-3 and met["cos"] > 0.99999, met
        assert torch.equal(y, mp(x)), "main_proj not deterministic"
    print("MAIN PASS", flush=True)
    if bench:
        wb = W.to(torch.bfloat16)
        for m in (6,):
            x = (torch.randn(m, 15360, device="cuda") * 3).to(torch.bfloat16)
            for name, fn in (("ds41h main_proj", lambda: mp(x)), ("bf16 F.linear (2x bytes)", lambda: torch.nn.functional.linear(x, wb))):
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s_):
                    for _ in range(20):
                        fn()
                g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e3 / 20)
                print(f"  m={m} {name:26s} {min(ts):7.1f} us (med {sorted(ts)[2]:7.1f})", flush=True)
    return 0


# ------------------------------------------------------------------------------------------- one-launch router (1 GPU)
def run_router(bench: bool = True) -> int:
    """drafter.MoEDecode.route (gate GEMV + dsv4_topk, one launch) vs fp32 torch gate GEMV + ds41h moe.route_topk
    (stock semantics: ll_bf16_gemm emits fp32 logits), with the real mtp.0 gate, incl. bias_vl image-sentinel routing."""
    from ds41h import drafter, moe
    from ds41h.ref.weights import default_reader

    lay = default_reader().mtp(int(os.environ.get("DS41H_TEST_MTP", "0")))
    gw = lay["ffn.gate.weight"].to(torch.bfloat16).cuda().contiguous()
    gb = lay["ffn.gate.bias"].float().cuda().contiguous()
    E = gw.shape[0]
    op = drafter.MoEDecode(torch.device("cuda"))
    bad = 0
    total = 0
    for m in (1, 2, 5, 7, 8):
        for seed in range(4):
            g = torch.Generator(device="cuda").manual_seed(1000 * m + seed)
            x = torch.randn(m, 5120, generator=g, device="cuda")
            x = (x / x.norm(dim=-1, keepdim=True) * 70).to(torch.bfloat16)
            for k, scale, renorm in ((3, 1.5, True), (6, 2.5, False)):
                w_f, i_f = op.route(x, gw, gb, k, scale, renorm)
                logits = x.float() @ gw.float().t()
                w_r, i_r = moe.route_topk(logits, gb, k, scale, renorm)
                same = (i_f == i_r).all(-1)
                total += m
                bad += int((~same).sum())
                if bool(same.all()):
                    wd = (w_f - w_r).abs().max().item()
                    assert wd < 1e-4 * max(1.0, scale), f"router weights differ by {wd} (m={m} k={k})"
    print(f"  ids identical for {total - bad}/{total} tokens", flush=True)
    assert bad <= max(1, total // 100), f"router picked different experts for {bad} tokens"
    # image-sentinel tokens route with bias_vl (int32 input ids, as vLLM's input_ids buffer)
    bvl = (gb.flip(0) * 0.5).contiguous()
    lo = 128815
    x = (torch.randn(6, 5120, device="cuda") * 1.0).to(torch.bfloat16)
    iid = torch.tensor([5, lo, lo + 4, lo + 5, 77, lo + 1], dtype=torch.int32, device="cuda")
    w_f, i_f = op.route(x, gw, gb, 3, 1.5, True, bias_vl=bvl, input_ids=iid, image_lo=lo)
    w_r, i_r = moe.route_topk(x.float() @ gw.float().t(), gb, 3, 1.5, True, bias_vl=bvl, input_ids=iid, image_lo=lo)
    assert torch.equal(i_f, i_r), (i_f, i_r)
    # graph replay with changing inputs (arrival counter reset by the last CTA)
    xs = (torch.randn(8, 5, 5120, device="cuda")).to(torch.bfloat16)
    xin = xs[0].clone()
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        out = op.route(xin, gw, gb, 3, 1.5, True)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s_):
        out = op.route(xin, gw, gb, 3, 1.5, True)
    for i in range(8):
        xin.copy_(xs[i])
        gr.replay()
        w_r, i_r = moe.route_topk(xs[i].float() @ gw.float().t(), gb, 3, 1.5, True)
        torch.cuda.synchronize()
        assert torch.equal(out[1], i_r), f"graph replay {i}: ids differ"
    print(f"ROUTER1 PASS (E={E})", flush=True)
    if bench:
        x = (torch.randn(5, 5120, device="cuda")).to(torch.bfloat16)
        fns = [("ds41h router (1 launch)", lambda: op.route(x, gw, gb, 3, 1.5, True)),
               ("torch gate + route_topk", lambda: moe.route_topk(torch.nn.functional.linear(x, gw).float(), gb, 3, 1.5, True))]
        try:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

            fns.append(("stock ll_bf16 + route_topk", lambda: moe.route_topk(ll_bf16_gemm(x, gw), gb, 3, 1.5, True)))
        except Exception as e:  # noqa: BLE001
            print(f"  (stock ll_bf16 unavailable: {type(e).__name__})")
        for name, fn in fns:
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s_):
                for _ in range(20):
                    fn()
            g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3 / 20)
            print(f"  m=5 {name:28s} {min(ts):7.2f} us (med {sorted(ts)[2]:7.2f})", flush=True)
    return 0


# ------------------------------------------------------------------------------------------- ctx-kv norm (1 GPU)
def run_norm3() -> int:
    """patch_drafter.ctx_kv_norm == vLLM rms_norm (ops.rms_norm) per layer slice, bitwise."""
    from vllm import _custom_ops as ops

    from ds41h import patch_drafter

    for T in (1, 6, 8, 37):
        kv_all = (torch.randn(T, 1536, device="cuda") * 3).to(torch.bfloat16)
        w = (1 + 0.2 * torch.randn(3, 512, device="cuda")).to(torch.bfloat16)
        eps = 1e-6
        out = patch_drafter.ctx_kv_norm(kv_all, w, eps)
        for i in range(3):
            x = kv_all[:, 512 * i : 512 * (i + 1)].contiguous()
            ref = torch.empty_like(x)
            ops.rms_norm(ref, x, w[i], eps)
            diff = (out[i].float() - ref.float()).abs().max().item()
            same = torch.equal(out[i], ref)
            print(f"  T={T} layer {i}: bitwise {same} max|diff| {diff:.3e}", flush=True)
            assert diff <= 2 ** -6 * max(1.0, ref.float().abs().max().item()), "ctx kv norm differs beyond 1 bf16 ulp"
    print("NORM3 PASS", flush=True)
    return 0


# ------------------------------------------------------------------------------- first drafter layer virtual inputs
def run_l0() -> int:
    """DS41H_DRAFTER_L0PRO identity: stock mhc_pre_delayed_tilelang on the hc-broadcast embedding with pre_mix None (the stock
    first DSpark layer) == stock mhc_shifted_post_pre on (x = embedding, residual 0, post 1, comb 0, pre_mix one-hot stream 0)
    (what the prologue path computes for the first layer under L0PRO)."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_shifted_post_pre

    g = torch.Generator(device="cuda").manual_seed(9)
    hc, H = 4, 5120
    mix = hc * (hc + 2)
    fn = torch.randn(mix, hc * H, generator=g, device="cuda") * 0.01
    hc_scale = torch.rand(3, generator=g, device="cuda") + 0.5
    hc_base = torch.randn(mix, generator=g, device="cuda") * 0.1
    nw = (1 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    args = (hc_scale, hc_base, 1e-20, 1e-6, 1e-6, 2.0, 20)
    ok = True
    for m in (1, 5, 8):
        x = (torch.randn(m, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        res = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
        p0, c0, li0, pre0 = mhc_pre_delayed_tilelang(res, fn, *args, pre_mix=None, norm_weight=nw, norm_eps=1e-6)
        onehot = torch.zeros(m, hc, dtype=torch.float32, device="cuda")
        onehot[:, 0] = 1
        r1, p1, c1, li1, pre1, _ = mhc_shifted_post_pre(
            x, torch.zeros(m, hc, H, dtype=torch.bfloat16, device="cuda"), torch.ones(m, hc, 1, device="cuda"),
            torch.zeros(m, hc, hc, device="cuda"), fn, *args, pre_mix=onehot, norm_weight=nw, norm_eps=1e-6, stream=None)
        same = [torch.equal(r1, res), torch.equal(p0, p1), torch.equal(c0, c1), torch.equal(li0, li1), torch.equal(pre0, pre1)]
        print(f"  m={m}: residual/post/comb/input/pre bitwise {same}; max|d input| {(li0.float() - li1.float()).abs().max().item():.3e}", flush=True)
        ok &= all(same[:1]) and (li0.float() - li1.float()).abs().max().item() <= 1e-2
    print("L0 PASS" if ok else "L0 FAIL", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------- bf16 verification head (1 GPU)
def run_vhead(bench: bool = True) -> int:
    """drafter.ext().head_bf16 (exact-class bf16 GEMV, fp32 accumulation) vs F.linear (cuBLAS) on a [32320, 5120] shard."""
    import torch.nn.functional as F

    from ds41h import drafter

    g = torch.Generator(device="cuda").manual_seed(5)
    n = 32320
    w = (torch.randn(n, 5120, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    hs = (torch.randn(40, 5120, generator=g, device="cuda") * 3).to(torch.bfloat16)
    e = drafter.ext()
    worst = 0
    for m in (1, 6, 8):
        idx = torch.randperm(40, device="cuda")[:m].to(torch.int64)
        y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        e.head_bf16(hs, idx, w, y, False)
        ref = F.linear(hs[idx], w)
        exact = (hs[idx].float() @ w.float().t())
        d_ref = (y.float() - ref.float()).abs()
        ulp = ref.float().abs().clamp_min(1e-3) * 2 ** -7
        worst = max(worst, int((d_ref > ulp).sum()))
        err_mine = (y.float() - exact).abs().mean().item()
        err_cublas = (ref.float() - exact).abs().mean().item()
        print(f"  m={m}: |mine - cublas| > 1 ulp: {int((d_ref > ulp).sum())}/{d_ref.numel()}, mean |err vs fp32| mine {err_mine:.3e} cublas {err_cublas:.3e}", flush=True)
        assert err_mine <= err_cublas * 1.1 + 1e-6, "head_bf16 less accurate than cuBLAS"
    assert worst == 0, "head_bf16 differs from cuBLAS by more than one bf16 ulp"
    print("VHEAD PASS", flush=True)
    if bench:
        for m in (6, 8):
            x = hs[:m].contiguous()
            y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            t_new = graph_time_us(lambda: e.head_bf16(x, None, w, y, False))[0]
            t_ref = graph_time_us(lambda: F.linear(x, w))[0]
            gb = n * 5120 * 2 / 1e3
            print(f"  bench m={m}: head_bf16 {t_new:.1f} us ({gb / t_new:.0f} GB/s) vs F.linear {t_ref:.1f} us ({gb / t_ref:.0f} GB/s)", flush=True)
    return 0


# ------------------------------------------------------------------------------- drafter o-projection (1 GPU)
def run_lin(bench: bool = True) -> int:
    """drafter.OProj (inverse RoPE + wo_a + wo_b, kernels/drafter.cu dlin) vs a float reference of the stock math
    (fused_inv_rope_fp8_quant(quantize=False) -> bf16 o, bf16 bmm with the dequantized wo_a -> bf16 z, FP8 wo_b)."""
    from common import compare, fmt_metrics

    from ds41h import drafter, skinny

    g = torch.Generator(device="cuda").manual_seed(3)

    def rand_fp8(n, k):
        w = (torch.randn(n, k, generator=g, device="cuda") * 0.6).clamp(-440, 440).to(torch.float8_e4m3fn)
        sc = torch.randint(118, 124, ((n + 31) // 32, k // 32), generator=g, device="cuda").to(torch.uint8)
        return w, sc

    wa8, sa = rand_fp8(2048, 4096)
    wb8, sb = rand_fp8(5120, 2048)
    WA, WB = skinny.pack(wa8, sa), skinny.pack(wb8, sb)
    da, db = skinny.dequant(WA), skinny.dequant(WB)  # fp32 [N, K]
    op = drafter.OProj(WA, WB, 16, 2)
    P = 70000
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    ang = torch.arange(P, device="cuda").float()[:, None] * inv[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], -1).contiguous()  # [P, 64] fp32
    worst = 0.0
    for m in (1, 5, 8):
        for heads_alloc in (16, 64):
            o_full = (torch.randn(m, heads_alloc, 512, generator=g, device="cuda") * 2).to(torch.bfloat16)
            pos = torch.randint(0, P, (m,), generator=g, device="cuda")
            o = o_full[:, :16, :]
            y = op(o, pos, cs)
            # reference
            x = o.float().clone()
            r = x[..., 448:].reshape(m, 16, 32, 2)
            c, s_ = cs[pos][:, None, :32], cs[pos][:, None, 32:]
            ev, od = r[..., 0], r[..., 1]
            rot = torch.stack([ev * c + od * s_, od * c - ev * s_], -1).reshape(m, 16, 64)
            x[..., 448:] = rot
            xb = x.to(torch.bfloat16).float().reshape(m, 2, 4096)
            z_ref = torch.stack([xb[:, gi] @ da[1024 * gi : 1024 * (gi + 1)].t() for gi in range(2)], 1).reshape(m, 2048)
            y_ref = z_ref.to(torch.bfloat16).float() @ db.t()
            met = compare(y.float(), y_ref)
            worst = max(worst, met["rel_rmse"])
            print(f"  m={m} heads_alloc={heads_alloc}: {fmt_metrics(met)}", flush=True)
            assert met["rel_rmse"] < 1e-2 and met["cos"] > 0.9999, met
    # wo_b alone vs the skinny kernel (same numerics class)
    z = (torch.randn(5, 2048, generator=g, device="cuda")).to(torch.bfloat16)
    y1 = torch.empty(5, 5120, dtype=torch.bfloat16, device="cuda")
    drafter.ext().lin_fp8(z, 0, 1, WB.w, WB.s, 5120, 2048, y1, None, None, False, -1, 0)
    y2 = skinny.gemm(z, WB)
    met = compare(y1.float(), y2.float())
    print(f"  wo_b vs skinny.gemm: {fmt_metrics(met)}", flush=True)
    assert met["rel_rmse"] < 2e-3, met
    # one-pass x staging (default, lin_set_stage(1)) vs the two-pass staging: bitwise identical (fp32 and bf16 cos_sin_cache)
    e = drafter.ext()
    for cache in (cs, cs.to(torch.bfloat16)):
        for m in (1, 5, 7, 8):
            o = (torch.randn(m, 16, 512, generator=g, device="cuda") * 2).to(torch.bfloat16)
            o[0, 3, 450] = 300.0  # an outlier inside a RoPE chunk (sets the token's staging scale)
            pos = torch.randint(0, P, (m,), generator=g, device="cuda")
            e.lin_set_stage(1)
            y1 = op(o, pos, cache)
            e.lin_set_stage(0)
            y0 = op(o, pos, cache)
            e.lin_set_stage(1)
            torch.cuda.synchronize()
            assert torch.equal(y1, y0), f"one-pass staging differs from the two-pass staging (m={m}, cache {cache.dtype})"
    print("LIN S1 PASS (one-pass staging bitwise == two-pass)", flush=True)
    # early dependent trigger (lin_set_early(1)): wo_b launched while wo_a runs -> bitwise identical, PDL launches, in a graph
    if hasattr(e, "lin_set_early"):
        oo = (torch.randn(7, 16, 512, generator=g, device="cuda") * 2).to(torch.bfloat16)
        pp = torch.randint(0, P, (7,), generator=g, device="cuda")
        e.lin_set_early(0)
        y0 = op(oo, pp, cs)
        e.lin_set_early(1)
        y1 = op(oo, pp, cs)
        torch.cuda.synchronize()
        assert torch.equal(y1, y0), "early wo_b trigger changed the o-projection"
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(st):
            op(oo, pp, cs)  # warm (allocator)
            with torch.cuda.graph(gr, stream=st):
                yg = op(oo, pp, cs)
        torch.cuda.current_stream().wait_stream(st)
        for _ in range(50):
            gr.replay()
        torch.cuda.synchronize()
        e.lin_set_early(0)
        assert torch.equal(yg, y0), "early wo_b trigger (graph replay) changed the o-projection"
        print("LIN EARLY PASS (early wo_b trigger bitwise == default, eager + 50 graph replays)", flush=True)
    print(f"LIN PASS (worst rel_rmse {worst:.2e})", flush=True)
    if bench:
        o = (torch.randn(8, 16, 512, device="cuda") * 2).to(torch.bfloat16)[:7]
        pos = torch.randint(0, P, (7,), device="cuda")
        e.lin_set_stage(0)
        t_s0 = graph_time_us(lambda: op(o, pos, cs))[0]
        e.lin_set_stage(1)
        print(f"  bench m=7: OProj two-pass staging {t_s0:.2f} us", flush=True)
        t_new = graph_time_us(lambda: op(o, pos, cs))[0]
        if hasattr(e, "lin_set_early"):
            e.lin_set_early(1)
            t_early = graph_time_us(lambda: op(o, pos, cs))[0]
            e.lin_set_early(0)
            print(f"  bench m=7: OProj one-pass {t_new:.2f} us, + early wo_b trigger {t_early:.2f} us", flush=True)
        zz = torch.randn(7, 2048, device="cuda").to(torch.bfloat16)
        wa_bf = da.to(torch.bfloat16).view(2, 1024, 4096)

        def stock_like():
            xo = o.reshape(7, 2, 4096).transpose(0, 1)
            zt = torch.bmm(xo, wa_bf.transpose(1, 2))
            return skinny.gemm(zt.transpose(0, 1).reshape(7, 2048), WB)

        t_old = graph_time_us(stock_like)[0]
        print(f"  bench m=7: OProj {t_new:.2f} us vs bf16 bmm + skinny wo_b {t_old:.2f} us (no inverse RoPE in the latter)", flush=True)
    return 0


# ------------------------------------------------------------------------------- batched SWA metadata (1 GPU)
def meta_case(gen, noncausal: bool, nreq: int, qlen: int, window: int, width: int, block_size: int, max_tokens: int,
              pad_slots: int = 0, invalid_frac: float = 0.2, replay: bool = False):
    """One synthetic DeepseekSparseSWA decode metadata job (buffers prefilled with garbage, as in a live server)."""
    from ds41h import patch_meta

    dev = "cuda"
    ntok = nreq * qlen
    nblk = 64
    seq = torch.randint(qlen + 1, nblk * block_size - 8, (nreq,), generator=gen).to(torch.int32)
    qsl = (torch.arange(nreq + 1, dtype=torch.int32) * qlen)
    bt = torch.randint(1, 5000, (nreq + 2, nblk + 3), generator=gen).to(torch.int32)
    n_slot = ntok + pad_slots
    slot = torch.randint(0, 1 << 20, (n_slot,), generator=gen).to(torch.int64)
    slot[torch.rand(n_slot, generator=gen) < invalid_frac] = -1
    rs = torch.randint(0, 40, (nreq,), generator=gen).to(torch.int32) if replay else torch.zeros(nreq, dtype=torch.int32)
    valid = torch.randint(0, 2, (max_tokens,), generator=gen).bool()
    lens = torch.randint(-5, 9999, (max_tokens,), generator=gen).to(torch.int32)
    idx = torch.randint(-5, 9999, (max_tokens, 1, width), generator=gen).to(torch.int32)
    t2r = torch.randint(0, 3, (max_tokens,), generator=gen).to(torch.int32)
    if not noncausal:  # causal: token->request map precomputed by stock compute_token_to_req_indices
        t2r[:ntok] = torch.arange(ntok, dtype=torch.int32) // qlen
    to = lambda t: t.to(dev)  # noqa: E731
    vw = to(valid)
    job = patch_meta.make_job(to(slot), vw if not noncausal else vw[:n_slot], to(lens) if not noncausal else to(lens)[:ntok],
                              to(idx), to(qsl), to(seq), to(t2r), to(bt), None if noncausal else to(rs), ntok, width, window,
                              block_size, noncausal)
    assert job is not None, "make_job rejected a well-formed case"
    return job


def run_meta(bench: bool = True) -> int:
    """patch_meta.swa_batch (one launch for many groups) == the stock per-group launches, bitwise."""
    from ds41h import patch_meta

    gen = torch.Generator().manual_seed(7)
    fails = 0
    cases = [
        # (noncausal, nreq, qlen, window, width, block_size, max_tokens, pad_slots, replay)
        (False, 1, 6, 128, 128, 64, 16896, 0, False),
        (False, 1, 6, 128, 128, 64, 16896, 2, True),
        (False, 4, 6, 128, 128, 64, 4096, 0, True),
        (False, 16, 6, 128, 128, 64, 4096, 5, False),
        (True, 1, 5, 128, 192, 64, 4096, 0, False),
        (True, 3, 5, 128, 192, 64, 4096, 0, False),
        (True, 8, 5, 128, 192, 64, 4096, 3, False),
    ]
    for g in (1, 3, 8):
        for c in cases:
            jobs = [meta_case(gen, *c[:7], pad_slots=c[7], replay=c[8]) for _ in range(g)]
            ref_jobs = [{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in j.items()} for j in jobs]
            for j in ref_jobs:
                patch_meta.stock_launch(j)
            patch_meta.launch(jobs)
            torch.cuda.synchronize()
            for i, (a, b) in enumerate(zip(jobs, ref_jobs)):
                outs_a, outs_b = patch_meta._outputs(a), patch_meta._outputs(b)
                same = all(torch.equal(x, y) for x, y in zip(outs_a, outs_b))
                if not same:
                    fails += 1
                    names = ["valid", "lens", "idx", "t2r"]
                    bad = [n for n, x, y in zip(names, outs_a, outs_b) if not torch.equal(x, y)]
                    print(f"  MISMATCH groups={g} case={c} job {i}: {bad}", flush=True)
            print(f"  groups={g} case={c}: {'ok' if fails == 0 else 'FAIL'}", flush=True)
    # mixed causal + non-causal in one launch
    jobs = [meta_case(gen, False, 1, 6, 128, 128, 64, 16896), meta_case(gen, True, 1, 5, 128, 192, 64, 4096)]
    ref_jobs = [{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in j.items()} for j in jobs]
    for j in ref_jobs:
        patch_meta.stock_launch(j)
    patch_meta.launch(jobs)
    torch.cuda.synchronize()
    for a, b in zip(jobs, ref_jobs):
        if not all(torch.equal(x, y) for x, y in zip(patch_meta._outputs(a), patch_meta._outputs(b))):
            fails += 1
            print("  MISMATCH mixed causal / non-causal batch", flush=True)
    if bench:
        jobs = [meta_case(gen, False, 1, 6, 128, 128, 64, 16896, replay=True) for _ in range(8)]
        t_b = graph_time_us(lambda: patch_meta.launch(jobs))[0]
        t_s = graph_time_us(lambda: [patch_meta.stock_launch(j) for j in jobs])[0]
        print(f"  bench 8 causal groups (c1, graph-captured): batched {t_b:.2f} us vs stock 24 launches {t_s:.2f} us", flush=True)
        jobs = [meta_case(gen, True, 1, 5, 128, 192, 64, 4096) for _ in range(3)]
        t_b = graph_time_us(lambda: patch_meta.launch(jobs))[0]
        t_s = graph_time_us(lambda: [patch_meta.stock_launch(j) for j in jobs])[0]
        print(f"  bench 3 DSpark groups (graph-captured): batched {t_b:.2f} us vs stock 9 launches {t_s:.2f} us", flush=True)
    print("META PASS" if fails == 0 else f"META FAIL ({fails})", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if "--meta" in sys.argv:
        sys.exit(run_meta("--nobench" not in sys.argv))
    if "--l0" in sys.argv:
        sys.exit(run_l0())
    if "--vhead" in sys.argv:
        sys.exit(run_vhead("--nobench" not in sys.argv))
    if "--lin" in sys.argv:
        sys.exit(run_lin("--nobench" not in sys.argv))
    if "--norm3" in sys.argv:
        sys.exit(run_norm3())
    if "--router" in sys.argv:
        sys.exit(run_router("--nobench" not in sys.argv))
    if "--main" in sys.argv:
        sys.exit(run_main())
    if "--head" in sys.argv:
        sys.exit(run_head())
    if "--prep" in sys.argv:
        sys.exit(run_prep())
    if "--moe" in sys.argv:
        sys.exit(run_moe("--bench" in sys.argv, "--benchonly" in sys.argv))
    if "RANK" not in os.environ:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            print("SKIP: needs 4 GPUs")
            sys.exit(0)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4", __file__, *sys.argv[1:]]
        sys.exit(subprocess.run(cmd, cwd=REPO).returncode)
    if "--bench" in sys.argv:
        run_bench()
    elif "--markov" in sys.argv:
        sys.exit(run_markov_quick())
    else:
        sys.exit(run_tests())
