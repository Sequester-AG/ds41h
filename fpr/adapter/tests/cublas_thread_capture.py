"""CUDA-graph capture on a thread that never used cuBLAS. The failure this guards against: D13's _extract ran on
PhysExecutor's merge thread (vLLM deferred a structured-output step, so the merge thread waited on step N's handle and
extracted it) and captured a new HSD graph there; the drafter's first torch.mm on that thread had to create its cuBLAS
handle inside the capture -> CUBLAS_STATUS_NOT_INITIALIZED, then cudaErrorStreamCaptureInvalidated.

1 GPU, a few seconds: a fresh thread captures torch.mm on a side stream (A) without and (B) after
adapter.phys_executor.warm_thread_blas on that stream; then replays (B).
On the GPUs, from fpr/: python adapter/tests/cublas_thread_capture.py
"""
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def capture(dev, s, warm: bool, out: dict, key: str):
    try:
        if warm:
            from adapter.phys_executor import warm_thread_blas
            warm_thread_blas([dev], [s])
        a = torch.randn(256, 512, device=dev, dtype=torch.bfloat16)
        w = torch.randn(1024, 512, device=dev, dtype=torch.bfloat16)
        y = torch.empty(256, 1024, device=dev, dtype=torch.bfloat16)
        f = torch.randn(64, 512, device=dev)
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            torch.mm(a, w.t(), out=y)
            torch.mm(a, w.t(), out_dtype=torch.float32)
            _ = f @ f.t()
        g.replay()
        torch.cuda.synchronize(dev)
        ref = a.float() @ w.float().t()
        out[key] = f"ok (max |err| {float((y.float() - ref).abs().max()):.3f})"
    except Exception as e:  # noqa: BLE001
        out[key] = f"FAILED: {str(e).splitlines()[0][:120]}"


def main():
    dev = torch.device("cuda", 0)
    torch.ones(1, device=dev)
    s = torch.cuda.Stream(device=dev)
    # the main thread has used cuBLAS on another stream (like the engine loop thread)
    x = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
    torch.mm(x, x)
    torch.cuda.synchronize(dev)
    out = {}
    for key, warm in (("B_warm", True), ("A_cold", False)):   # B first: A's failure may poison the context
        t = threading.Thread(target=capture, args=(dev, s, warm, out, key))
        t.start()
        t.join()
        print(key, out[key], flush=True)
    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
