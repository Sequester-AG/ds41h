"""Compressor (kv sources 2 / 8 / 14 / 20) and indexer q path (all 8 index layers) on OWNED rows, checked
against a capture (rank 0's captured stock tensors) and timed.

Compressor per kv-source layer, owned rows only (stock vLLM kernels, batch-invariant per row):
  kv_score = x @ [wkv | wgate]^T (bf16 -> fp32, cuBLAS)          [R, 512 * ratio]
  fused_save_compress_norm (Triton): ratio-2 softmax-gated pool of (ring row, current row), RMSNorm -> latent bf16, ring store
  rope_quant_insert (Triton): RoPE(latent) at the group position, 7 x 64 ue8m0-fp8 + 64 bf16 RoPE dims -> 584-B record
  k_pre = latent @ wk^T (bf16), indexer_k_norm_rope_store (Triton): k_norm, RoPE, fp32-scaled fp8 -> 132-B index-K entry
Indexer q path per index layer:
  w = x @ weights_proj^T (bf16), q = qr @ wq_b^T (fp8 32x32-block weights, exactly dequantized to bf16; fp32 accumulate),
  fused_indexer_q_rope_quant (stock): RoPE + per-head ue8m0 fp8 quant, weights_out = w * q_scale * 128^-0.5 * 32^-0.5

  python comp2f.py --step 0 [--rank r]      exactness (bit-exact rates, ulp stats, error vs fp64) + cold timings

Only statistics are printed (no content).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idx2f as X  # noqa: E402

CAP = os.environ.get("CAP_DIR", "cap")
ATTN_DENSE = os.path.join(os.path.dirname(HERE), "attn_dense")
EPS = 1e-20


def _bench2b():
    sys.path.insert(0, ATTN_DENSE)
    import bench2b as B  # noqa: E402  (bench2b's stock RoPE table + checkpoint reader; read-only use)

    return B


def load_weights(L: int, dev) -> dict:
    B = _bench2b()
    rd = B.CheckpointReader(B.MODEL)
    W = B.L.load_layer_weights(rd, L, rank=None, device=dev, experts="none")
    bf = torch.bfloat16
    w = dict(ratio=W.compress_ratio, idx_wq_b=W.idx_wq_b.to(bf).contiguous(),
             idx_wp=W.idx_weights_proj.to(bf).contiguous(), cos_sin=B.stock_cos_sin(W.compress_ratio, dev, 0),
             f32=dict(idx_wq_b=W.idx_wq_b, idx_wp=W.idx_weights_proj))
    assert bool((w["idx_wq_b"].float() == W.idx_wq_b).all()) and bool((w["idx_wp"].float() == W.idx_weights_proj).all())
    if W.comp_wkv is not None:
        parts = [W.comp_wkv] + ([W.comp_wgate] if W.comp_wgate is not None else [])
        wc = torch.cat(parts, 0)
        w.update(comp_w=wc.to(bf).contiguous(), comp_norm=W.comp_norm.to(bf).contiguous(),
                 idx_wk=W.idx_wk.to(bf).contiguous(), idx_k_norm=W.idx_k_norm.to(bf).contiguous())
        assert bool((w["comp_w"].float() == wc).all()) and bool((w["idx_wk"].float() == W.idx_wk).all())
        w["f32"].update(comp_w=wc)
    return w


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """bf16 / fp32 bitwise equality rate and the max distance in units of the dtype's ulp."""
    if a.dtype == torch.bfloat16:
        ia, ib = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
        ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
        ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    else:
        ia, ib = a.contiguous().view(torch.int32).to(torch.int64), b.contiguous().view(torch.int32).to(torch.int64)
        ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
        ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    d = (ia - ib).abs()
    return dict(n=int(a.numel()), exact=round(float((d == 0).float().mean()), 6), max_ulp=int(d.max()) if d.numel() else 0)


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Row-scaled error: max over rows of max|x - ref| / max|ref| (fp64)."""
    x, ref = x.double(), ref.double()
    den = ref.abs().amax(-1).clamp(min=1e-300)
    return float(((x - ref).abs().amax(-1) / den).max())


def cold(fn, iters=15, flush=None) -> float:
    ts = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return round(statistics.median(ts), 1)


def run(args) -> dict:
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant, indexer_k_norm_rope_store
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (fused_save_compress_norm,
                                                                             rope_quant_insert)

    dev = torch.device("cuda")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    rec20 = torch.load(f"{args.cap}/s{args.step}_L20_r0.pt", weights_only=False, mmap=True)
    sl = rec20["logits"]["seq_lens"]
    B = int(sl.shape[0])
    ctx = [int(v) for v in sl[:, -1].tolist()]
    own = X.assign_requests(ctx, 4)
    ranks = [args.rank] if args.rank is not None else list(range(4))
    out = dict(step=args.step, B=B, layers={})
    for L in X.LAYERS:
        rec = torch.load(f"{args.cap}/s{args.step}_L{L}_r0.pt", weights_only=False, mmap=True)
        W = load_weights(L, dev)
        ratio = W["ratio"]
        lay = dict(ratio=ratio, ranks={})
        for r in ranks:
            reqs = own[r]
            rows = torch.tensor([b * X.NEXT_N + j for b in reqs for j in range(X.NEXT_N)], dtype=torch.int64)
            R = rows.numel()
            x = rec["attn_in"]["hidden_states"][rows].to(dev)
            pos = rec["attn_in"]["positions"][rows].to(dev)
            res = {}
            # ---------------- q path
            qr = rec["idx"]["qr"][rows].to(dev)
            w_ours = F.linear(x, W["idx_wp"])
            q_ours = F.linear(qr, W["idx_wq_b"])
            w_cap = rec["idx"]["indexer_weights"][rows].to(dev)
            q_cap = rec["idx"]["wq_b_out"][rows].to(dev)
            res["idx_weights"] = ulp_stats(w_ours, w_cap)
            res["idx_q"] = ulp_stats(q_ours, q_cap)
            ref_q = qr.double() @ W["f32"]["idx_wq_b"].double().T
            res["idx_q_err_vs_fp64"] = dict(ours=rel_err(q_ours, ref_q), stock=rel_err(q_cap, ref_q))
            ref_w = x.double() @ W["f32"]["idx_wp"].double().T
            res["idx_w_err_vs_fp64"] = dict(ours=rel_err(w_ours, ref_w), stock=rel_err(w_cap, ref_w))
            # stock RoPE + quant on the captured q / w (bit-exact replay) and on ours
            qf_c, wo_c = fused_indexer_q_rope_quant(pos, q_cap.view(R, 32, 128), W["cos_sin"], w_cap, 128 ** -0.5,
                                                    32 ** -0.5)
            res["q_fp8_replay_exact"] = float((qf_c.view(torch.uint8) == rec["idx"]["q_fp8"][rows].to(dev).view(
                torch.uint8)).float().mean())
            res["weights_out_replay"] = ulp_stats(wo_c, rec["idx"]["weights_out"][rows].to(dev))
            qf_o, wo_o = fused_indexer_q_rope_quant(pos, q_ours.view(R, 32, 128), W["cos_sin"], w_ours, 128 ** -0.5,
                                                    32 ** -0.5)
            res["q_fp8_ours_exact"] = float((qf_o.view(torch.uint8) == qf_c.view(torch.uint8)).float().mean())
            res["t_us"] = dict(
                weights_proj=cold(lambda: F.linear(x, W["idx_wp"]), flush=flush),
                wq_b=cold(lambda: F.linear(qr, W["idx_wq_b"]), flush=flush),
                rope_quant=cold(lambda: fused_indexer_q_rope_quant(pos, q_ours.view(R, 32, 128), W["cos_sin"], w_ours,
                                                                   128 ** -0.5, 32 ** -0.5), flush=flush))
            # ---------------- compressor (kv sources)
            if L in X.KV_SOURCES and "comp" in rec:
                c = rec["comp"]
                ks_cap = c["kv_score"][rows].to(dev)
                ks_ours = torch.mm(x, W["comp_w"].T, out_dtype=torch.float32)
                res["kv_score"] = ulp_stats(ks_ours, ks_cap)
                ref_ks = x.double() @ W["f32"]["comp_w"].double().T
                res["kv_score_err_vs_fp64"] = dict(ours=rel_err(ks_ours, ref_ks), stock=rel_err(ks_cap, ref_ks))
                # ring (ratio 2): compact per-request ring blocks
                if ratio == 2:
                    slots = c["state_slots"][rows]
                    blk_cap = c["ring_blocks"]
                    sb = int(c["state_block"])
                    ring = torch.zeros((R // X.NEXT_N + 1, sb, 1024), dtype=torch.float32)
                    new_slots = torch.full_like(slots, -1)
                    for i, b in enumerate(reqs):
                        s = c["state_slots"][b * X.NEXT_N: (b + 1) * X.NEXT_N]
                        ok = s >= 0
                        if ok.any():
                            blk = int(s[ok][0]) // sb
                            k = int((blk_cap == blk).nonzero()[0])
                            ring[i + 1] = c["ring_before"][k]
                            new_slots[i * X.NEXT_N: (i + 1) * X.NEXT_N] = torch.where(
                                ok, (i + 1) * sb + s % sb, torch.full_like(s, -1))
                    ring = ring.to(dev)
                    new_slots = new_slots.to(dev)
                    qsl = torch.arange(0, R + 1, X.NEXT_N, dtype=torch.int32, device=dev)
                    t2r = (torch.arange(R, device=dev) // X.NEXT_N).to(torch.int32)
                else:
                    ring = qsl = t2r = None
                    new_slots = torch.where(c["main_slots"][rows] >= 0, torch.arange(R), torch.full((R,), -1)).to(dev)

                def compress(ks, ring_t):
                    lat = torch.empty((R, 512), dtype=torch.bfloat16, device=dev)
                    fused_save_compress_norm(ks, pos, ring_t, new_slots, qsl, t2r, W["comp_norm"], EPS, ratio, lat)
                    return lat

                ring0 = ring.clone() if ring is not None else None
                lat_rep = compress(ks_cap, ring0)
                wslots = c["written_slots"][rows]
                valid = (wslots >= 0).to(dev)
                lat_cap = c["latent"][rows].to(dev)
                res["latent_replay"] = ulp_stats(lat_rep[valid], lat_cap[valid])
                if ring is not None:
                    ra = ring0[1:].cpu()
                    ok = 0
                    tot = 0
                    for i, b in enumerate(reqs):
                        s = c["state_slots"][b * X.NEXT_N: (b + 1) * X.NEXT_N]
                        if (s >= 0).any():
                            blk = int(s[s >= 0][0]) // int(c["state_block"])
                            k = int((c["ring_blocks"] == blk).nonzero()[0])
                            tot += 1
                            ok += int(torch.equal(ra[i], c["ring_after"][k]))
                    res["ring_after_replay_exact_reqs"] = f"{ok}/{tot}"
                ring1 = ring.clone() if ring is not None else None
                lat_ours = compress(ks_ours, ring1)
                res["latent_ours_vs_stock"] = ulp_stats(lat_ours[valid], lat_cap[valid])
                # main record: rope_quant_insert into a compact cache, slot t for written row t
                bs = 64
                cache = torch.zeros(((R + bs - 1) // bs + 1, bs, 584), dtype=torch.uint8, device=dev)
                rs = torch.where(valid, torch.arange(R, device=dev) + bs, torch.full((R,), -1, device=dev))
                rope_quant_insert(lat_rep, pos, W["cos_sin"], cache, rs, ratio)
                got = X_gather(cache, rs)
                exp = c["records"][rows].to(dev)
                res["record_replay_exact_rows"] = f"{int((got == exp).all(-1)[valid].sum())}/{int(valid.sum())}"
                # index K
                kc = torch.zeros(((R + bs - 1) // bs + 1, bs, 132), dtype=torch.uint8, device=dev)
                kpre = F.linear(lat_rep, W["idx_wk"])
                indexer_k_norm_rope_store(kpre, pos, W["cos_sin"], W["idx_k_norm"], EPS, kc, rs, ratio, False)
                gk = X_gather(kc, rs, 128, 4)
                ek = rec["idxk"]["entries"][rows].to(dev)
                res["index_k_replay_exact_rows"] = f"{int((gk == ek).all(-1)[valid].sum())}/{int(valid.sum())}"
                res["t_us"].update(
                    kv_score=cold(lambda: torch.mm(x, W["comp_w"].T, out_dtype=torch.float32), flush=flush),
                    save_compress_norm=cold(lambda: compress(ks_ours, ring.clone() if ring is not None else None),
                                            flush=flush),
                    rope_quant_insert=cold(lambda: rope_quant_insert(lat_rep, pos, W["cos_sin"], cache, rs, ratio),
                                           flush=flush),
                    wk=cold(lambda: F.linear(lat_rep, W["idx_wk"]), flush=flush),
                    k_store=cold(lambda: indexer_k_norm_rope_store(kpre, pos, W["cos_sin"], W["idx_k_norm"], EPS, kc,
                                                                   rs, ratio, False), flush=flush))
            lay["ranks"][r] = res
            print(json.dumps({"layer": L, "rank": r, **res}), flush=True)
        out["layers"][L] = lay
        del W
        torch.cuda.empty_cache()
    return out


def graph_chain(args) -> dict:
    """The q path (8 index layers) + compressor (4 kv sources) of ONE rank's owned rows with the stock kernels, captured
    as the step program would run them: one CUDA graph per layer chain and one for the whole step (cold L2 before each
    replay; read-only flush)."""
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant, indexer_k_norm_rope_store
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (fused_save_compress_norm,
                                                                             rope_quant_insert)

    dev = torch.device("cuda")
    flush = torch.ones(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    rec20 = torch.load(f"{args.cap}/s{args.step}_L20_r0.pt", weights_only=False, mmap=True)
    sl = rec20["logits"]["seq_lens"]
    ctx = [int(v) for v in sl[:, -1].tolist()]
    own = X.assign_requests(ctx, 4)
    reqs = list(range(len(ctx))) if args.all else own[args.rank or 0]
    rows = torch.tensor([b * X.NEXT_N + j for b in reqs for j in range(X.NEXT_N)], dtype=torch.int64)
    R = rows.numel()
    chains = {}
    keep = []
    for L in X.LAYERS:
        rec = torch.load(f"{args.cap}/s{args.step}_L{L}_r0.pt", weights_only=False, mmap=True)
        W = load_weights(L, dev)
        W.pop("f32")
        ratio = W["ratio"]
        x = rec["attn_in"]["hidden_states"][rows].to(dev)
        pos = rec["attn_in"]["positions"][rows].to(dev)
        qr = rec["idx"]["qr"][rows].to(dev)
        bufs = dict(x=x, pos=pos, qr=qr, W=W)
        if L in X.KV_SOURCES:
            bs = 64
            bufs["ring"] = torch.zeros((R // X.NEXT_N + 1, 8, 1024), dtype=torch.float32, device=dev) if ratio == 2 else None
            bufs["slots"] = (torch.arange(R, device=dev) % 8 + (torch.arange(R, device=dev) // X.NEXT_N + 1) * 8
                             if ratio == 2 else torch.arange(R, device=dev))
            bufs["qsl"] = torch.arange(0, R + 1, X.NEXT_N, dtype=torch.int32, device=dev)
            bufs["t2r"] = (torch.arange(R, device=dev) // X.NEXT_N).to(torch.int32)
            bufs["lat"] = torch.empty((R, 512), dtype=torch.bfloat16, device=dev)
            bufs["cache"] = torch.zeros(((R + bs - 1) // bs + 1, bs, 584), dtype=torch.uint8, device=dev)
            bufs["kc"] = torch.zeros(((R + bs - 1) // bs + 1, bs, 132), dtype=torch.uint8, device=dev)
            bufs["rs"] = torch.arange(R, device=dev) + bs

        side = torch.cuda.Stream() if args.overlap else None

        def chain(b=bufs, L=L, ratio=ratio, side=side):
            W = b["W"]
            if side is not None and L in X.KV_SOURCES:  # compressor on a side branch, q path on the main stream
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    comp(b, L, ratio)
            w = F.linear(b["x"], W["idx_wp"])
            q = F.linear(b["qr"], W["idx_wq_b"])
            fused_indexer_q_rope_quant(b["pos"], q.view(R, 32, 128), W["cos_sin"], w, 128 ** -0.5, 32 ** -0.5)
            if side is not None and L in X.KV_SOURCES:
                torch.cuda.current_stream().wait_stream(side)
            elif L in X.KV_SOURCES:
                comp(b, L, ratio)

        def comp(b, L, ratio):
            W = b["W"]
            if True:
                ks = torch.mm(b["x"], W["comp_w"].T, out_dtype=torch.float32)
                fused_save_compress_norm(ks, b["pos"], b["ring"], b["slots"], b["qsl"] if ratio == 2 else None,
                                         b["t2r"] if ratio == 2 else None, W["comp_norm"], EPS, ratio, b["lat"])
                rope_quant_insert(b["lat"], b["pos"], W["cos_sin"], b["cache"], b["rs"], ratio)
                kpre = F.linear(b["lat"], W["idx_wk"])
                indexer_k_norm_rope_store(kpre, b["pos"], W["cos_sin"], W["idx_k_norm"], EPS, b["kc"], b["rs"], ratio,
                                          False)

        chains[L] = chain
        keep.append(bufs)
    res = dict(rank="all (production, replicated)" if args.all else (args.rank or 0), R=R, overlap=bool(args.overlap),
               layer_us={})

    def gtime(fn):
        fn()
        torch.cuda.synchronize()
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            fn()
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(15):
            flush.view(torch.int64).sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        return round(statistics.median(ts), 1)

    for L in X.LAYERS:
        res["layer_us"][L] = gtime(chains[L])
    res["step_us_sum"] = round(sum(res["layer_us"].values()), 1)
    res["step_graph_us"] = gtime(lambda: [chains[L]() for L in X.LAYERS])
    print(json.dumps(res), flush=True)
    return res


def qfused(args) -> dict:
    """The fused q path (idx2f_q.cu) on the owned rows of every index layer: epilogue bit-exactness vs the stock op on the
    same bf16 q / w, GEMM closeness vs fp64 (ours vs cuBLAS vs production), top-k effect left to bench2f; graph timing."""
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant

    dev = torch.device("cuda")
    flush = torch.ones(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    e = X.ext_q()
    rec20 = torch.load(f"{args.cap}/s{args.step}_L20_r0.pt", weights_only=False, mmap=True)
    sl = rec20["logits"]["seq_lens"]
    ctx = [int(v) for v in sl[:, -1].tolist()]
    own = X.assign_requests(ctx, 4)
    ranks = [args.rank] if args.rank is not None else list(range(4))
    out = {}
    wscale = float(128 ** -0.5 * 32 ** -0.5)
    for L in X.LAYERS:
        rec = torch.load(f"{args.cap}/s{args.step}_L{L}_r0.pt", weights_only=False, mmap=True)
        W = load_weights(L, dev)
        for r in ranks:
            reqs = own[r]
            rows = torch.tensor([b * X.NEXT_N + j for b in reqs for j in range(X.NEXT_N)], dtype=torch.int64)
            R = rows.numel()
            x = rec["attn_in"]["hidden_states"][rows].to(dev).contiguous()
            pos = rec["attn_in"]["positions"][rows].to(dev).contiguous()
            qr = rec["idx"]["qr"][rows].to(dev).contiguous()
            part = torch.empty((16, R, 32), dtype=torch.float32, device=dev)
            qf8 = torch.empty((R, 32, 128), dtype=torch.float8_e4m3fn, device=dev)
            wo = torch.empty((R, 32), dtype=torch.float32, device=dev)
            qd = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
            e.qpath(x, qr, W["idx_wp"], W["idx_wq_b"], pos, W["cos_sin"], wscale, part, qf8, wo, qd, 3)
            torch.cuda.synchronize()
            w = part[0].clone()
            for k in range(1, 16):
                w = w + part[k]
            w = w.to(torch.bfloat16)
            qs, ws = fused_indexer_q_rope_quant(pos, qd.view(R, 32, 128), W["cos_sin"], w, 128 ** -0.5, 32 ** -0.5)
            res = dict(q_fp8_vs_stock_epilogue=float((qs.view(torch.uint8) == qf8.view(torch.uint8)).float().mean()),
                       wout_vs_stock_epilogue=ulp_stats(wo, ws))
            ref_q = qr.double() @ W["f32"]["idx_wq_b"].double().T
            ref_w = x.double() @ W["f32"]["idx_wp"].double().T
            q_cap = rec["idx"]["wq_b_out"][rows].to(dev)
            w_cap = rec["idx"]["indexer_weights"][rows].to(dev)
            res["q_err_vs_fp64"] = dict(fused=rel_err(qd, ref_q), cublas=rel_err(F.linear(qr, W["idx_wq_b"]), ref_q),
                                        production=rel_err(q_cap, ref_q))
            res["w_err_vs_fp64"] = dict(fused=rel_err(w, ref_w), cublas=rel_err(F.linear(x, W["idx_wp"]), ref_w),
                                        production=rel_err(w_cap, ref_w))
            res["q_fp8_vs_production"] = float((qf8.view(torch.uint8) == rec["idx"]["q_fp8"][rows].to(dev).view(
                torch.uint8)).float().mean())

            def fused(which=3):
                e.qpath(x, qr, W["idx_wp"], W["idx_wq_b"], pos, W["cos_sin"], wscale, part, qf8, wo, None, which)

            def stock():
                w_ = F.linear(x, W["idx_wp"])
                q_ = F.linear(qr, W["idx_wq_b"])
                fused_indexer_q_rope_quant(pos, q_.view(R, 32, 128), W["cos_sin"], w_, 128 ** -0.5, 32 ** -0.5)

            res["graph_us"] = dict(fused=gtime1(fused, flush), wp_only=gtime1(lambda: fused(1), flush),
                                   q_only=gtime1(lambda: fused(2), flush), stock=gtime1(stock, flush),
                                   stock_wq_b=gtime1(lambda: F.linear(qr, W["idx_wq_b"]), flush),
                                   stock_wp=gtime1(lambda: F.linear(x, W["idx_wp"]), flush))
            out.setdefault(L, {})[r] = res
            print(json.dumps({"layer": L, "rank": r, **res}), flush=True)
        del W
        torch.cuda.empty_cache()
    return out


def gtime1(fn, flush, iters=15) -> float:
    fn()
    torch.cuda.synchronize()
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        fn()
    torch.cuda.current_stream().wait_stream(s_)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(iters):
        flush.view(torch.int64).sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return round(statistics.median(ts), 1)


def X_gather(cache, slots, data=576, scale=8):
    nb, bs = cache.shape[0], cache.shape[1]
    flat = cache.reshape(-1)
    i = slots.clamp(min=0).to(torch.int64)
    page = (i // bs) * (bs * (data + scale))
    p = i % bs
    d = flat[(page + p * data)[:, None] + torch.arange(data, device=cache.device)]
    s = flat[(page + bs * data + p * scale)[:, None] + torch.arange(scale, device=cache.device)]
    return torch.cat([d, s], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--qfused", action="store_true")
    ap.add_argument("--all", action="store_true", help="--graph on every request (production's replicated work)")
    ap.add_argument("--overlap", action="store_true", help="--graph: compressor on a side branch next to the q path")
    args = ap.parse_args()
    res = qfused(args) if args.qfused else graph_chain(args) if args.graph else run(args)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
