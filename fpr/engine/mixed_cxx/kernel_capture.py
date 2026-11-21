"""kernel_capture: capture a library call's kernels at warm-up as FprKcRecord structs (kernel_capture.h), relaunched
verbatim by the C++ driver. For calls whose every argument is constant per (bucket, layer) once
the piece owns persistent buffers (CuTe DSL kernels: vLLM fused_indexer_q_rope_quant; no AOT export in CuTe DSL 4.8).

    recs = capture(lambda: kernel(..persistent buffers..), device)   # list of bytes, one per kernel, in order
    ext().launch(recs[0], stream)                                     # the test entry (the driver includes kernel_capture.h)

capture() runs fn once outside any capture first (CuTe DSL compiles / loads its module on first use), then once inside a
thread-local CUDA graph capture on a side stream of the device; the call must not allocate (persistent buffers only) and
must contain only kernel nodes.
"""
from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT: dict = {}


def ext():
    if "m" not in _EXT:
        from torch.utils.cpp_extension import load

        bdir = os.environ.get("KERNEL_CAPTURE_BUILD") or os.path.join(HERE, "build", "kernel_capture")
        os.makedirs(bdir, exist_ok=True)
        _EXT["m"] = load(name="kernel_capture_ext", sources=[os.path.join(HERE, "kernel_capture_ext.cpp")],
                         extra_cflags=["-O3", "-std=c++17"], extra_include_paths=[HERE], extra_ldflags=["-lcuda"],
                         with_cuda=True, build_directory=bdir, verbose=False)
    return _EXT["m"]



_PRIV: dict = {}   # (device index, tag) -> a NON-POOLED torch.cuda.ExternalStream


def _priv_stream(dev, tag: str = "build"):
    """a stream no other code can be handed: torch.cuda.Stream() draws round-robin from a
    32-stream pool per device and priority, which the engine's side / Engram / AGRS comm streams also come from, so a
    "fresh" builder stream aliased an engine stream after ~32 creations (its waits / syncs then covered unrelated work).
    cudaStreamCreateWithFlags(cudaStreamNonBlocking) once per (device, tag), wrapped as an ExternalStream; never destroyed
    (process lifetime)."""
    import ctypes as _C
    d = torch.device(dev)
    d = torch.device("cuda", d.index if d.index is not None else torch.cuda.current_device())
    k = (d.index, tag)
    s = _PRIV.get(k)
    if s is None:
        rt = None
        for ln in open("/proc/self/maps"):
            p = ln.split()[-1]
            if "libcudart.so" in p:
                rt = _C.CDLL(p)
                break
        if rt is None:
            raise RuntimeError("_priv_stream: no libcudart mapped")
        h = _C.c_void_p()
        prev = _C.c_int()
        rt.cudaGetDevice(_C.byref(prev))
        rt.cudaSetDevice(_C.c_int(d.index))
        try:
            rc = rt.cudaStreamCreateWithFlags(_C.byref(h), _C.c_uint(1))   # cudaStreamNonBlocking
        finally:
            rt.cudaSetDevice(prev)
        if rc != 0 or not h.value:
            raise RuntimeError(f"_priv_stream: cudaStreamCreateWithFlags rc {rc}")
        s = _PRIV[k] = torch.cuda.ExternalStream(h.value, device=d)
    return s

def capture(fn, device, warm=None, sync: bool = True) -> list[bytes]:
    """fn's kernels as FprKcRecord bytes. warm: the warm-up call (default fn itself; give a call of the same kernels on
    SCRATCH buffers when fn's buffers may be in use). sync=False: never block the host (no device synchronize; the warm-up
    runs on a private stream, the capture executes nothing): for builds during a C++ span, where the device may run work
    that waits on work not issued yet, so a host sync would deadlock."""
    E = ext()
    dev = torch.device(device)
    with torch.cuda.device(dev):
        if sync:
            (warm or fn)()                     # warm: JIT compile / module load outside the capture
            torch.cuda.synchronize(dev)
        else:
            ws = _priv_stream(dev, "kc_warm")
            cur = torch.cuda.current_stream(dev)
            with torch.cuda.stream(ws):
                (warm or fn)()                 # (host-side compile / load; its launch runs on ws)
            # the warm's buffers are the CALLER's (scratch allocated on the caller's stream
            # and freed when the caller returns): before, nobody waited for ws, so the caching allocator could hand those
            # blocks out again (to the next allocation on the caller's pooled stream) while ws's warm kernels still wrote
            # them. The caller's stream now waits for ws: a GPU-side wait only (no host sync, no deadlock: ws holds nothing
            # but the warm), so every later use of those blocks on the caller's stream runs after the warm
            cur.wait_stream(ws)
        st = _priv_stream(dev, "kc_capture")
        if sync:
            st.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(st):
            E.begin(st.cuda_stream)
            try:
                fn()
            finally:
                recs = E.end(st.cuda_stream, dev.index)
        if sync:
            torch.cuda.synchronize(dev)
    return list(recs)
