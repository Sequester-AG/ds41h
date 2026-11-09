#!/usr/bin/env python3
"""agrs: NVLink AG / RS physics at the mixed step's sizes, all 4 GPUs at once.

Per collective round, every rank sends R rows of 10,240 B to each of the 3 peers (+ a local copy into its own slot) and
waits for its 3 incoming slabs (AG and RS move the same bytes; RS reads rank s's rows from a 4R-row partial).
  SM: agrs.cu k_push (comm.cu's store loop / signals), ncta CTAs per destination, UNR loads in flight, MODE 0 (comm.cu
      order) / 1 (contiguous slab per CTA); then k_wait.
  CE: cudaMemcpyAsync per destination (copy engines; peer access on), on 1 or 4 streams per rank, in 1 / 2 / 4 chunks;
      the receiver's stream waits on every sender's done events.
  Overlap: a bf16 GEMM [R, 5120] x [5120, 6144] on the main stream with the collective on a side stream (GEMM time
      alone vs with each collective path, and the collective's time under the GEMM).
  Sum: the RS sum's own time (comm.cu k_rs_sum arithmetic without the wait) at R rows.
Synthetic bytes only. One process, no model.

  python3 commbench.py [--rows 4288,1024,320] [--iters 10] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = 4
RB = 10240


def ext():
    from torch.utils.cpp_extension import load
    bd = os.path.join(HERE, "build")
    os.makedirs(bd, exist_ok=True)
    return load(name="agrs_comm", sources=[os.path.join(HERE, "agrs.cu")], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="4288,1024,320")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "commbench.json"))
    a = ap.parse_args()
    e = ext()
    if a.dry_run:
        print("dry run ok (extension built)", flush=True)
        return
    e.enable_peers(W)
    dev = [torch.device("cuda", i) for i in range(W)]
    Rmax = max(int(x) for x in a.rows.split(","))
    src = [torch.randint(0, 256, (W * Rmax, RB), dtype=torch.uint8, device=d) for d in dev]
    bufs = [torch.zeros(W * Rmax, RB, dtype=torch.uint8, device=d) for d in dev]
    main_st = [torch.cuda.Stream(device=d) for d in dev]
    side_st = [torch.cuda.Stream(device=d) for d in dev]
    ce_st = [[torch.cuda.Stream(device=d) for _ in range(4)] for d in dev]
    res = dict(rows={}, overlap={}, sum={})

    def sync():
        for d in dev:
            torch.cuda.synchronize(d)

    def timed(issue, N, sts):
        for _ in range(2):
            issue()
        sync()
        e0 = [torch.cuda.Event(enable_timing=True) for _ in dev]
        e1 = [torch.cuda.Event(enable_timing=True) for _ in dev]
        for r in range(W):
            e0[r].record(sts[r])
        for _ in range(N):
            issue()
        for r in range(W):
            e1[r].record(sts[r])
        sync()
        return [e0[r].elapsed_time(e1[r]) / N for r in range(W)]

    def sm_issue(R, kind, ncta, unr, mode, sts):
        n16 = R * RB // 16
        stride = n16 if kind == "rs" else 0
        F = [torch.zeros(4, dtype=torch.int32, device=d) for d in dev]
        EP = [torch.zeros(1, dtype=torch.int32, device=d) for d in dev]
        bv = [b[: W * R] for b in bufs]

        def issue():
            for r in range(W):
                with torch.cuda.device(dev[r]), torch.cuda.stream(sts[r]):
                    e.push(src[r], n16, stride, r, bv, F, EP[r], ncta, unr, mode)
            for r in range(W):
                with torch.cuda.device(dev[r]), torch.cuda.stream(sts[r]):
                    e.wait(F[r], EP[r], W * ncta)
        return issue

    def ce_issue(R, kind, nst, nch, sts):
        nb = R * RB

        def issue():
            done = []
            for r in range(W):
                with torch.cuda.device(dev[r]):
                    ev = torch.cuda.Event()
                    ev.record(sts[r])
                    used = ce_st[r][:nst]
                    for cs in used:
                        cs.wait_event(ev)
                    k = 0
                    for s in range(W):
                        so = (s * nb) if kind == "rs" else 0
                        cb = nb // nch
                        for c in range(nch):
                            o = c * cb
                            n = nb - o if c == nch - 1 else cb
                            cs = used[k % nst]
                            k += 1
                            e.ce_copy(bufs[s], r * nb + o, src[r], so + o, n, cs.cuda_stream)
                    dl = []
                    for cs in used:
                        d_ = torch.cuda.Event()
                        d_.record(cs)
                        dl.append(d_)
                    done.append(dl)
            for r in range(W):
                with torch.cuda.device(dev[r]):
                    for dl in done:
                        for d_ in dl:
                            sts[r].wait_event(d_)
        return issue

    def check(R, kind):
        nb = R * RB
        for s in range(W):
            for r in range(W):
                got = bufs[s].view(-1)[r * nb:(r + 1) * nb].cpu()
                so = (s * nb) if kind == "rs" else 0
                exp = src[r].view(-1)[so:so + nb].cpu()
                if not torch.equal(got, exp):
                    return False
        return True

    variants = [("SM", dict(ncta=n, unr=u, mode=m)) for n in (16, 32, 48) for u in (1, 4) for m in (0, 1)]
    variants += [("CE", dict(nst=s, nch=c)) for s, c in ((1, 1), (4, 1), (4, 2), (4, 4))]
    for R in (int(x) for x in a.rows.split(",")):
        mb = R * RB / 1e6
        out = {}
        for kind in ("ag", "rs"):
            for path, kw in variants:
                for b in bufs:
                    b.zero_()
                iss = sm_issue(R, kind, sts=main_st, **kw) if path == "SM" else ce_issue(R, kind, sts=main_st, **kw)
                ts = timed(iss, a.iters, main_st)
                ok = check(R, kind)
                tm = max(ts)
                key = f"{kind} {path} " + " ".join(f"{k}={v}" for k, v in kw.items())
                out[key] = dict(ms=round(tm, 4), ms_per_rank=[round(t, 4) for t in ts], gbs_per_peer=round(mb / tm, 1), ok=ok)
                print(f"R {R} ({mb:.1f} MB per peer) {key:28s}: {tm * 1000:8.1f} us per round = {mb / tm:6.1f} GB/s per "
                      f"peer per direction {'ok' if ok else 'BYTES WRONG'}", flush=True)
        res["rows"][R] = out

    # overlap: a GEMM on the main stream, the collective on the side stream
    R = Rmax
    A = [torch.randn(R, 5120, device=d).bfloat16() for d in dev]
    B = [torch.randn(5120, 6144, device=d).bfloat16() for d in dev]
    C = [torch.empty(R, 6144, device=d, dtype=torch.bfloat16) for d in dev]

    def gemm_issue(k):
        def issue():
            for r in range(W):
                with torch.cuda.device(dev[r]), torch.cuda.stream(main_st[r]):
                    for _ in range(k):
                        torch.mm(A[r], B[r], out=C[r])
        return issue
    g1 = max(timed(gemm_issue(1), a.iters, main_st))
    res["overlap"]["gemm_alone_ms"] = round(g1, 4)
    print(f"GEMM [{R}, 5120] x [5120, 6144] bf16 alone: {g1 * 1000:.1f} us", flush=True)
    for path, kw in (("SM", dict(ncta=16, unr=1, mode=0)), ("SM", dict(ncta=32, unr=4, mode=0)), ("CE", dict(nst=4, nch=1))):
        coll = sm_issue(R, "rs", sts=side_st, **kw) if path == "SM" else ce_issue(R, "rs", sts=side_st, **kw)
        c1 = max(timed(coll, a.iters, side_st))
        ng = max(1, round(c1 / g1))         # GEMMs per collective round: both streams busy for about the same time

        def both():
            coll()
            gemm_issue(ng)()
        for _ in range(2):
            both()
        sync()
        e0 = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in dev]
        e1 = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in dev]
        for r in range(W):
            e0[r][0].record(main_st[r])
            e0[r][1].record(side_st[r])
        for _ in range(a.iters):
            both()
        for r in range(W):
            e1[r][0].record(main_st[r])
            e1[r][1].record(side_st[r])
        sync()
        gt = max(e0[r][0].elapsed_time(e1[r][0]) for r in range(W)) / (a.iters * ng)
        ct = max(e0[r][1].elapsed_time(e1[r][1]) for r in range(W)) / a.iters
        key = f"{path} " + " ".join(f"{k}={v}" for k, v in kw.items())
        res["overlap"][key] = dict(coll_alone_ms=round(c1, 4), gemms_per_round=ng, gemm_with_ms=round(gt, 4),
                                   coll_with_ms=round(ct, 4), gemm_slowdown=round(gt / g1, 3), coll_slowdown=round(ct / c1, 3))
        print(f"overlap {key:24s}: collective alone {c1 * 1000:.1f} us, under {ng} GEMMs {ct * 1000:.1f} us; GEMM "
              f"{g1 * 1000:.1f} -> {gt * 1000:.1f} us ({gt / g1:.3f}x)", flush=True)

    # RS sum's own time
    for R in (int(x) for x in a.rows.split(",")):
        recv = [torch.randn(W, R, 5120, device=d).bfloat16() for d in dev]
        o = [torch.empty(R, 5120, device=d, dtype=torch.bfloat16) for d in dev]
        for grid in (132, 264, 528):
            def s_issue():
                for r in range(W):
                    with torch.cuda.device(dev[r]), torch.cuda.stream(main_st[r]):
                        e.sum(recv[r], R * 5120 // 2, o[r], grid)
            t = max(timed(s_issue, a.iters, main_st))
            gb = (W + 1) * R * 5120 * 2 / 1e9
            res["sum"][f"{R} grid {grid}"] = round(t, 4)
            print(f"RS sum R {R} grid {grid}: {t * 1000:.1f} us ({gb / t * 1000:.0f} GB/s HBM)", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
