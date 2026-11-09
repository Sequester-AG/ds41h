#!/usr/bin/env python3
"""K5 sequence-parallel collectives (comm.cu) on 4 GPUs in one process: correctness and time of
  AG: every rank pushes its R own rows of f (bf16 [R, 5120]) + routing (6 ids + 6 weights, padded to 64 B) to all ranks
  RS: every rank pushes the 3 x R rows other ranks own of its MoE partial (bf16 [4R, 5120]); owners sum 4 slots
against torch references, and the time per collective (all 4 GPUs, measured with events on GPU 0 around N iterations,
each iteration = push on all ranks + wait / sum on all ranks).

  python3 test_comm.py --rows 128 --iters 200 [--ncta 8]
"""
import argparse
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
W = 4


def ext():
    defs = [f"-D{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("SPC_")]  # e.g. SPC_UNR=8
    tag = "_".join(d[2:].replace("=", "") for d in defs)
    bd = os.path.join(HERE, "build_comm" + (f"_{tag}" if tag else ""))
    os.makedirs(bd, exist_ok=True)
    return load(name="ds41h_phys_comm" + (f"_{tag}" if tag else ""), sources=[os.path.join(HERE, "comm.cu")],
                build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"] + defs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ncta", type=int, default=8)
    a = ap.parse_args()
    e = ext()
    e.enable_peers(W)
    R, H = a.rows, 5120
    dev = [torch.device("cuda", i) for i in range(W)]
    RB = H * 2 + 64  # f row + routing (6 int32 ids + 6 fp32 weights + pad) in one row of bytes
    f = [torch.randint(0, 256, (R, RB), dtype=torch.uint8, device=d) for d in dev]
    gbuf = [torch.zeros(W * R, RB, dtype=torch.uint8, device=d) for d in dev]
    gflag = [torch.zeros(4, dtype=torch.int32, device=d) for d in dev]
    part = [(torch.randn(W * R, H, device=d) * (i + 1)).bfloat16() for i, d in enumerate(dev)]
    rbuf = [torch.zeros(W, R, H, dtype=torch.bfloat16, device=d) for d in dev]
    rflag = [torch.zeros(4, dtype=torch.int32, device=d) for d in dev]
    rout = [torch.empty(R, H, dtype=torch.bfloat16, device=d) for d in dev]
    streams = [torch.cuda.Stream(device=d) for d in dev]
    ag_ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in dev]
    rs_ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in dev]
    SIG = W * a.ncta  # signals per round on each receiver: W sources x ncta CTAs per destination

    def ag_rank(r):
        e.ag_push(f[r], r, gbuf, gflag, ag_ep[r], a.ncta)
        e.wait(gflag[r], ag_ep[r], SIG)

    def rs_rank(r):
        e.rs_push(part[r], r, rbuf, rflag, rs_ep[r], a.ncta)
        e.rs_sum(rbuf[r], rflag[r], rs_ep[r], SIG, rout[r], 132)

    def ag():
        for r in range(W):
            with torch.cuda.device(dev[r]), torch.cuda.stream(streams[r]):
                ag_rank(r)

    def rs():
        for r in range(W):
            with torch.cuda.device(dev[r]), torch.cuda.stream(streams[r]):
                rs_rank(r)

    def sync():
        for r in range(W):
            streams[r].synchronize()

    ag(); rs(); sync()
    ok_ag = all(torch.equal(gbuf[r][s * R:(s + 1) * R], f[s].to(dev[r])) for r in range(W) for s in range(W))
    ok_rs, maxd = True, 0.0
    for r in range(W):
        ref = sum(part[s][r * R:(r + 1) * R].float().to(dev[r]) for s in range(W)).bfloat16()
        ok_rs &= torch.equal(rout[r], ref)
        maxd = max(maxd, float((rout[r].float() - ref.float()).abs().max()))
    print(f"R={R}: AG correct {ok_ag}, RS bit-identical to the fp32 reference sum {ok_rs} (max diff {maxd})")
    K = 50  # rounds per graph: host launches out of the measurement
    for name, rank_fn, mb in (("AG", ag_rank, R * RB * 3 / 1e6), ("RS", rs_rank, R * H * 2 * 3 / 1e6)):
        graphs = []
        for r in range(W):
            with torch.cuda.device(dev[r]):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=streams[r]):
                    for _ in range(K):
                        rank_fn(r)
                graphs.append(g)
        def replay():
            for r in range(W):
                with torch.cuda.device(dev[r]):
                    graphs[r].replay()
        replay(); sync()
        ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(W)]
        ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(W)]
        reps = max(1, a.iters // K)
        for r in range(W):
            with torch.cuda.device(dev[r]):
                ev0[r].record(torch.cuda.current_stream(dev[r]))
        for _ in range(reps):
            replay()
        for r in range(W):
            with torch.cuda.device(dev[r]):
                ev1[r].record(torch.cuda.current_stream(dev[r]))
        sync()
        for r in range(W):
            torch.cuda.synchronize(dev[r])
        t = [ev0[r].elapsed_time(ev1[r]) / (reps * K) * 1000 for r in range(W)]
        print(f"  {name} (graphs, {K} rounds each): {max(t):.1f} us per collective (per rank {', '.join(f'{v:.1f}' for v in t)}); "
              f"egress {mb:.2f} MB per rank = {mb / max(t) * 1e3:.0f} GB/s per rank, {mb / 3 / max(t) * 1e3:.0f} GB/s per link")
    ok_rs = all(torch.equal(rout[r], sum(part[s][r * R:(r + 1) * R].float().to(dev[r]) for s in range(W)).bfloat16())
                for r in range(W))
    print(f"  after the graph runs: RS still bit-identical {ok_rs}")

if __name__ == "__main__":
    main()
