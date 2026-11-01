"""TP4 all-reduce bandwidth for prefill-sized tensors (NCCL via torch.distributed), steady state under the cap.

  bash -c 'for p in default Simple LL128 LL; do \
      NCCL_PROTO=$p torchrun --nproc-per-node 4 scripts/nccl_bench.py; done'

(NCCL_PROTO=default is passed through as unset.) Prints per size: median us per all_reduce, algbw, busbw.
A 2 s all-reduce heat-up runs first. --gemm interleaves a bf16 GEMM (8192 x 5120 x 5120, ~3 ms) before every timed
all-reduce, like prefill (compute -> all-reduce), so the all-reduce runs at the power-capped clock the GEMM leaves.
"""

import os
import statistics
import sys
import time

if os.environ.get("NCCL_PROTO") == "default":
    del os.environ["NCCL_PROTO"]

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r)
    gemm = "--gemm" in sys.argv
    sizes = [(16384, 5120), (8192, 5120)] if gemm else [(8192, 5120), (16384, 5120), (2048, 5120), (128, 5120)]
    ga = torch.randn(8192, 5120, dtype=torch.bfloat16, device="cuda")
    gb = torch.randn(5120, 5120, dtype=torch.bfloat16, device="cuda")
    bufs = {s: torch.randn(*s, dtype=torch.bfloat16, device="cuda") for s in sizes}
    free0 = torch.cuda.mem_get_info()[0]
    dist.all_reduce(bufs[sizes[0]])
    torch.cuda.synchronize()
    nccl_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20
    t_end = time.time() + 2.0
    while time.time() < t_end:  # heat-up
        if gemm:
            for _ in range(4):
                ga @ gb
        dist.all_reduce(bufs[sizes[0]])
    torch.cuda.synchronize()
    for s in sizes:
        x = bufs[s]
        times = []
        for _ in range(30):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            dist.barrier()
            if gemm:
                for _ in range(4):
                    ga @ gb
            e0.record()
            dist.all_reduce(x)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1e3)
        med = statistics.median(times)
        nbytes = x.numel() * 2
        if r == 0:
            alg = nbytes / med / 1e3
            tag = os.environ.get("NCCL_PROTO", "default") + ("" if not os.environ.get("NCCL_MIN_NCHANNELS") else f" ch>={os.environ['NCCL_MIN_NCHANNELS']}") + ("" if not os.environ.get("NCCL_BUFFSIZE") else f" buf={int(os.environ['NCCL_BUFFSIZE']) >> 20}M") + (" +gemm" if gemm else "") + f" [{nccl_mb:.0f} MiB]"
            print(f"{tag:36s} [{s[0]:5d} x {s[1]}] {nbytes / 1e6:6.1f} MB: "
                  f"median {med:8.1f} us  algbw {alg:6.1f} GB/s  busbw {alg * 2 * (w - 1) / w:6.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
