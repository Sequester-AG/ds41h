"""4-GPU tests for ds41h ALLREDUCE (kernels/allreduce.cu, ds41h/allreduce.py).

Run (needs 4 GPUs; DS41H unset):
    python -m pytest tests/test_allreduce.py -q                 # spawns torchrun itself
    torchrun --nproc-per-node 4 tests/test_allreduce.py         # tests directly
    torchrun --nproc-per-node 4 tests/test_allreduce.py --bench # latency sweep vs NCCL / vLLM stock

Checks: bitwise equality with the fixed-order fp32 reference sum (rank 0, 1, 2, 3; one bf16 rounding),
bit-identical outputs on all ranks, in-place use, 2000 back-to-back calls of random sizes (epoch /
parity / line-clearing logic), all-gather layouts for dim 0 / -1 / odd word counts, gather_words, and
CUDA-graph capture + many replays with changing inputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SIZES = [8, 1 << 10, 4 << 10, 16 << 10, 6 * 5120 * 2, 64 << 10, 96 * 5120 * 2, 256 << 10, 1 << 20, 2 << 20]


# ---------------------------------------------------------------------------------------- pytest entry
def test_allreduce_4gpu():
    try:
        import pytest
    except ImportError:  # pragma: no cover
        pytest = None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        if pytest is not None:
            pytest.skip("needs 4 GPUs")
        return
    env = dict(os.environ, DS41H_AR_TIMEOUT_S="60")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4", __file__]
    res = subprocess.run(cmd, env=env, cwd=REPO, timeout=1800)
    assert res.returncode == 0, "torchrun allreduce tests failed"


# ---------------------------------------------------------------------------------------- worker side
def setup():
    os.environ.setdefault("DS41H_AR_TIMEOUT_S", "60")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    cpu_group = dist.new_group(backend="gloo")
    from ds41h.allreduce import LLComm

    comm = LLComm(cpu_group, torch.device("cuda", local), max_bytes=4 << 20, name="test")
    if comm.disabled:  # setup failed on some rank (agreed on all ranks, see the log for the cause)
        raise RuntimeError("LLComm setup failed (disabled on all ranks)")
    return rank, local, cpu_group, comm


def ref_sum(x: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


def same_on_all_ranks(x: torch.Tensor) -> bool:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    return all(torch.equal(parts[0].view(torch.uint8), p.view(torch.uint8)) for p in parts)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[rank {dist.get_rank()}] {msg}")


def t_all_reduce_sizes(comm, rank):
    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    for nbytes in SIZES:
        n = nbytes // 2
        x = torch.randn(n, generator=gen, device="cuda").to(torch.bfloat16) * (rank + 1)
        x[: min(n, 8)] = torch.tensor([0.0, -0.0, 65504.0, -1e-30, 1.0, -1.0, 3.0, 1e-3], device="cuda")[: min(n, 8)].to(torch.bfloat16)
        out = comm.all_reduce(x)
        ref = ref_sum(x)
        check(torch.equal(out.view(torch.int16), ref.view(torch.int16)), f"all_reduce mismatch at {nbytes} bytes")
        check(same_on_all_ranks(out), f"outputs differ across ranks at {nbytes} bytes")
        y = x.clone()
        comm.all_reduce(y, out=y)  # in place
        check(torch.equal(y.view(torch.int16), ref.view(torch.int16)), f"in-place all_reduce mismatch at {nbytes} bytes")


def t_all_reduce_stress(comm, rank):
    gen = torch.Generator().manual_seed(7)  # same size sequence on every rank
    dgen = torch.Generator(device="cuda").manual_seed(rank)
    for it in range(2000):
        n = int(torch.randint(1, 65536, (1,), generator=gen)) * 4
        x = torch.randn(n, generator=dgen, device="cuda").to(torch.bfloat16)
        out = comm.all_reduce(x)
        if it % 97 == 0:
            check(torch.equal(out.view(torch.int16), ref_sum(x).view(torch.int16)), f"stress mismatch at iteration {it}")
    torch.cuda.synchronize()


def t_all_gather(comm, rank, world):
    from ds41h import allreduce as ar

    # (6, 32320) fp32 = 758 KiB per rank (the vocab-parallel logits shard of 6 rows) is above the default
    # routing limit DS41H_AG_MAX_BYTES = 512 KiB, so the stock path keeps it unless the limit is raised; the
    # LL kernel itself handles it (the IPC slot is 4 MiB here), which is checked by calling it directly.
    cases = [((6, 5120), torch.bfloat16, -1), ((6, 5120), torch.bfloat16, 0), ((6, 32320), torch.float32, -1), ((3, 7), torch.int32, 1),
             ((2, 3, 5), torch.float32, 1), ((1,), torch.int32, 0)]
    for shape, dtype, dim in cases:
        x = (torch.arange(int(torch.tensor(shape).prod()), device="cuda") + 1000 * rank).to(dtype).view(shape)
        routed = x.nbytes <= min(comm.max_bytes, ar.AG_MAX_BYTES)
        check(comm.can_all_gather(x, dim) == routed, f"can_all_gather {shape} dim {dim}: expected {routed} "
              f"({x.nbytes} B vs DS41H_AG_MAX_BYTES={ar.AG_MAX_BYTES})")
        check(comm.can_all_gather(x, dim, max_bytes=comm.max_bytes), f"can_all_gather {shape} dim {dim} at the slot capacity")
        out = comm.all_gather(x, dim)
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x)
        ref = torch.cat(parts, dim=dim)
        check(out.shape == ref.shape and torch.equal(out, ref), f"all_gather mismatch {shape} {dtype} dim {dim}")
    x = torch.randn(16, 5, device="cuda") + rank
    g = comm.gather_words(x)
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x)
    check(torch.equal(g, torch.stack(parts)), "gather_words mismatch")


def t_cuda_graph(comm, rank):
    gen = torch.Generator(device="cuda").manual_seed(50 + rank)
    sizes = [61440, 4096, 983040, 61440, 1 << 20]
    ins = [torch.zeros(s // 2, dtype=torch.bfloat16, device="cuda") for s in sizes]
    gin = torch.zeros(6, 333, dtype=torch.float32, device="cuda")
    for t in ins:  # warm-up outside capture
        comm.all_reduce(t)
    comm.all_gather(gin, -1)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [comm.all_reduce(t) for t in ins]
        gout = comm.all_gather(gin, -1)
        outs2 = [comm.all_reduce(o) for o in outs[:2]]  # chained: consumes the previous collective's output
    for it in range(50):
        for t in ins:
            t.copy_(torch.randn(t.shape, generator=gen, device="cuda").to(torch.bfloat16))
        gin.copy_(torch.randn(gin.shape, generator=gen, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        if it % 7 == 0 or it == 49:
            for t, o in zip(ins, outs):
                check(torch.equal(o.view(torch.int16), ref_sum(t).view(torch.int16)), f"graph replay {it}: all_reduce mismatch")
            for o, o2 in zip(outs[:2], outs2):
                check(torch.equal(o2.view(torch.int16), ref_sum(o).view(torch.int16)), f"graph replay {it}: chained mismatch")
            parts = [torch.empty_like(gin) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, gin)
            check(torch.equal(gout, torch.cat(parts, -1)), f"graph replay {it}: all_gather mismatch")
    # eager calls after replays keep working (device epoch stayed in sync)
    x = torch.randn(4096, generator=gen, device="cuda").to(torch.bfloat16)
    check(torch.equal(comm.all_reduce(x).view(torch.int16), ref_sum(x).view(torch.int16)), "eager after graph mismatch")


def run_tests():
    rank, local, cpu_group, comm = setup()
    world = dist.get_world_size()
    fails = 0
    for name, fn in [
        ("all_reduce_sizes", lambda: t_all_reduce_sizes(comm, rank)),
        ("all_reduce_stress", lambda: t_all_reduce_stress(comm, rank)),
        ("all_gather", lambda: t_all_gather(comm, rank, world)),
        ("cuda_graph", lambda: t_cuda_graph(comm, rank)),
    ]:
        try:
            fn()
            status = "PASS"
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            status = f"FAIL ({e})"
            fails += 1
        ok = torch.tensor([0 if status == "PASS" else 1], device="cuda")
        dist.all_reduce(ok)
        if rank == 0:
            print(f"{name:<24s} {'PASS' if int(ok) == 0 else 'FAIL'}", flush=True)
        if int(ok):
            fails += 1 if status == "PASS" else 0
            break  # a failed collective test can leave peers out of step
    comm.close()
    dist.barrier()
    dist.destroy_process_group()
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------- bench
def time_us(fn, iters=200, graph_calls=50, graph_ctx=None):
    """(eager us/call, graph us/call). Graph mode captures `graph_calls` calls (inside `graph_ctx()`,
    e.g. vLLM CustomAllreduce.capture) and replays them."""
    import contextlib

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    eager = s.elapsed_time(e) * 1e3 / iters
    graph_us = float("nan")
    try:
        stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with graph_ctx() if graph_ctx is not None else contextlib.nullcontext():
            with torch.cuda.graph(g, stream=stream):
                for _ in range(graph_calls):
                    fn()
        g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        s.record()
        for _ in range(5):
            g.replay()
        e.record()
        e.synchronize()
        graph_us = s.elapsed_time(e) * 1e3 / (5 * graph_calls)
    except Exception as ex:  # noqa: BLE001
        if dist.get_rank() == 0:
            print(f"  (graph timing unavailable: {ex})")
    return eager, graph_us


def run_bench():
    rank, local, cpu_group, comm = setup()
    device = torch.device("cuda", local)
    stock = {}
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        ca = CustomAllreduce(group=cpu_group, device=device)
        if not ca.disabled:
            stock["vllm_custom_ar"] = ca
    except Exception as ex:  # noqa: BLE001
        if rank == 0:
            print(f"vLLM CustomAllreduce unavailable: {ex}")
    try:
        from vllm.distributed.device_communicators.flashinfer_all_reduce import FlashInferAllReduce

        fi = FlashInferAllReduce(group=cpu_group, device=device)
        if not fi.disabled:
            stock["flashinfer_trtllm"] = fi
    except Exception as ex:  # noqa: BLE001
        if rank == 0:
            print(f"FlashInfer all-reduce unavailable: {ex}")
    if rank == 0:
        print(f"{'bytes':>10s} {'impl':<20s} {'eager us':>9s} {'graph us':>9s} {'algbw GB/s':>11s}")
    for nbytes in [1 << 10, 4 << 10, 16 << 10, 61440, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 983040, 1 << 20, 2 << 20]:
        x = torch.randn(nbytes // 2, device=device).to(torch.bfloat16)
        impls = {"ds41h_ll": lambda: comm.all_reduce(x)}
        y = x.clone()
        impls["nccl"] = lambda: dist.all_reduce(y)
        if "vllm_custom_ar" in stock and stock["vllm_custom_ar"].should_custom_ar(x):
            impls["vllm_custom_ar"] = lambda: stock["vllm_custom_ar"].custom_all_reduce(x)
        if "flashinfer_trtllm" in stock and stock["flashinfer_trtllm"].should_use_fi_ar(x):
            impls["flashinfer_trtllm"] = lambda: stock["flashinfer_trtllm"].all_reduce(x)
        for name, fn in impls.items():
            ctx = stock["vllm_custom_ar"].capture if name == "vllm_custom_ar" else None
            eager, graph_us = time_us(fn, graph_ctx=ctx)
            if rank == 0:
                best = min(eager, graph_us) if graph_us == graph_us else eager
                print(f"{nbytes:>10d} {name:<20s} {eager:9.2f} {graph_us:9.2f} {nbytes / best / 1e3:11.2f}", flush=True)
    if rank == 0:
        print("\nall_gather (dim -1)")
    for shape, dtype in [((6, 5120), torch.bfloat16), ((6, 32320), torch.float32), ((96, 6400), torch.bfloat16)]:
        x = torch.randn(shape, device=device).to(dtype)
        eager, graph_us = time_us(lambda: comm.all_gather(x, -1))
        out = torch.empty((4 * shape[0], shape[1]), dtype=dtype, device=device)
        eager_n, graph_n = time_us(lambda: dist.all_gather_into_tensor(out, x))
        if rank == 0:
            print(f"{str(shape):>14s} {str(dtype):<16s} ds41h {eager:8.2f}/{graph_us:8.2f} us   nccl(dim0, no movedim) {eager_n:8.2f}/{graph_n:8.2f} us")
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:  # plain `python tests/test_allreduce.py`: relaunch under torchrun (or skip)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            print("SKIP: needs 4 GPUs (run with torchrun --nproc-per-node 4)")
            sys.exit(0)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4", __file__, *sys.argv[1:]]
        sys.exit(subprocess.run(cmd, env=dict(os.environ, DS41H_AR_TIMEOUT_S=os.environ.get("DS41H_AR_TIMEOUT_S", "60")), cwd=REPO).returncode)
    if "--bench" in sys.argv:
        run_bench()
    else:
        sys.exit(run_tests())
