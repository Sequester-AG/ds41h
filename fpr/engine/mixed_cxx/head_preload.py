"""head_preload.

50 module loads landed inside the precapture job's GLOBAL-mode captures (graph H in decode_head_precap, H1 in
mixed_cxx_head): 26 Triton launchers' cuModuleLoadData (first launch of a kernel on a device) and 24 libcuda lazy loads of
cuBLASLt kernels (torch mm -> cublasGemmEx -> cuKernelSetAttribute), every one the first use of that kernel on that device.

preload(eng, body, tag) loads exactly those modules OUTSIDE any capture, executing nothing on live buffers:
  1. PROBE: per rank, `body(r)` (the code the real capture records, e.g. hooks.hsd_graph(r, st[r])) runs inside a THROWAWAY
     CUDA graph capture (private pool; destroyed after) with two interceptions:
       - Triton: JITFunction.run(..., warmup=False) -> run(..., warmup=True) (compiles / loads from the disk cache with the SAME
         arguments = the same specialization and key; returns the CompiledKernel; nothing launched), kernel recorded;
       - torch matmuls (aten mm / addmm / bmm / baddbmm, functional and out=): recorded as (op, args' dtype / shape / stride /
         storage offset / 128-B pointer phase, scalars); not executed (out= returns the out tensor; functional returns an
         empty result from the private pool).
     Nothing else in body() is intercepted: a capture executes nothing anyway.
  2. LOAD, after the probe capture ended, every GPU synced:
       - each recorded Triton kernel: _init_handles() with its device current (the exact cuModuleLoadData its first launch did);
       - each distinct recorded matmul: replayed once on SCRATCH tensors with the same dtype / shape / stride / storage offset
         and pointer phase mod 128 (cuBLAS's heuristic sees the same problem: the same kernel, loaded lazily now), on rank r's
         stream, outside capture. Scratch only: no live buffer is read or written.
Default off; callers gate it (decode_head_precap: ENG_DECODE_HEAD_PRECAPTURE_PRELOAD=1). Returns dict(triton=, mm=, s=).
the probe captures in the SAME pool as the real capture (pool= the caller's, e.g. eng.gpool[r]), so a
buffer the body creates lazily on first use is a normal pool allocation, as in the real capture; snapshots before / after the
probe (roots= objects, e.g. [hooks.hsd]): every CUDA tensor slot reachable (mixed_cxx_head.guard_record: path + data_ptr) and
the roots' scalar attributes -> new_slots / rebound_slots / scalar_changes in the result and in the log line (0 expected; a
nonzero count is reported, never hidden); t0 / t1 = the probe's wall-clock window."""
from __future__ import annotations

import contextlib
import time

import torch

_STATE = {"probe": None}          # the active probe (records) or None
_MM_DONE = set()                  # (v3) matmul problems already replayed in this process (per device): replayed once
# (v3) HEAD_PRELOAD_CHECK=1: the roots' tensor-slot / scalar snapshots around every probe; 0 (default) = no snapshots
import os as _os
CHECK = _os.environ.get("HEAD_PRELOAD_CHECK", "0") == "1"
_PATCHED = [False]
_MM_OPS = None


def _mm_ops():
    global _MM_OPS
    if _MM_OPS is None:
        a = torch.ops.aten
        names = [("mm", "default"), ("mm", "out"), ("addmm", "default"), ("addmm", "out"), ("bmm", "default"),
                 ("bmm", "out"), ("baddbmm", "default"), ("baddbmm", "out")]
        _MM_OPS = {getattr(getattr(a, n), o) for n, o in names if hasattr(getattr(a, n, None), o)}
    return _MM_OPS


def _patch_triton():
    if _PATCHED[0]:
        return
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001
        _PATCHED[0] = True
        return
    orig = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        p = _STATE["probe"]
        if p is None or warmup:
            return orig(self, *args, grid=grid, warmup=warmup, **kwargs)
        k = orig(self, *args, grid=grid, warmup=True, **kwargs)   # compile / cache load only, same key; no launch
        if k is not None:
            p["triton"].append((torch.cuda.current_device(), k))
        return None
    JITFunction.run = run
    _PATCHED[0] = True


def _meta(x):
    if isinstance(x, torch.Tensor):
        return ("T", x.dtype, tuple(x.shape), tuple(x.stride()), int(x.storage_offset()), int(x.data_ptr()) % 128,
                x.device.index)
    return ("V", x)


class _MMRec(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        p = _STATE["probe"]
        if p is not None and func in _mm_ops():
            p["mm"].append((func, tuple(_meta(a) for a in args), {k: _meta(v) for k, v in kwargs.items()}))
            if "out" in kwargs:
                return kwargs["out"]
            with torch.utils._python_dispatch._disable_current_modes():
                mt = func(*[torch.empty_strided(a.shape, a.stride(), dtype=a.dtype, device="meta")
                            if isinstance(a, torch.Tensor) else a for a in args],
                          **{k: v for k, v in kwargs.items()})
                return torch.empty_strided(mt.shape, mt.stride(), dtype=mt.dtype, device=args[0].device)
        return func(*args, **kwargs)


def _scratch(m, dev):
    _, dt, sh, sd, off, phase, _d = m
    need = off + 1 + sum((s - 1) * st for s, st in zip(sh, sd)) if all(sh) else off + 1
    es = torch.empty((), dtype=dt).element_size()
    raw = torch.zeros(need * es + 256, dtype=torch.uint8, device=dev)
    base = raw.data_ptr()
    shift = ((phase - (base + off * es)) % 128) // 1
    if shift % es:   # (a phase that is not a multiple of the element size cannot be matched: use the natural alignment)
        shift = 0
    v = raw[shift: shift + need * es].view(dt)
    return v.as_strided(sh, sd, off), raw


def _slots(roots):
    import sys as _sys
    mx = _sys.modules.get("mixed_cxx_head")
    gr = getattr(mx, "guard_record", None)
    out = {}
    for i, o in enumerate(roots or ()):
        if gr is not None:
            for e in gr(o, f"root{i}"):
                if e[4] is not None:
                    out[e[7]] = e[4]
    return out


def _scalars(roots):
    out = {}
    for i, o in enumerate(roots or ()):
        for k, v in getattr(o, "__dict__", {}).items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                out[f"root{i}.{k}"] = v
    return out


def preload(eng, body, tag="", log=None, pools=None, roots=None, capture_error_mode="global") -> dict:
    """body(r): rank r's capture body (what the real capture records); pools[r]: the real capture's pool (None = a private
    one); roots: objects whose tensor slots / scalars are compared around the probe; capture_error_mode: the REAL
    capture's mode. See the module docstring. The caller must drop any capture-state
    the body records (e.g. HSD DraftRank.mix_ev events) before its real capture: they belong to the destroyed probe graph."""
    lg = log or eng.log
    _patch_triton()
    W = len(eng.devs)
    t0, w0 = time.perf_counter(), time.time()
    if not CHECK:
        roots = None
    s0, c0 = _slots(roots), _scalars(roots)
    tsn = time.perf_counter()
    rec = {"triton": [], "mm": []}
    _STATE["probe"] = rec
    try:
        for r in range(W):
            with torch.cuda.device(eng.devs[r]):
                g = torch.cuda.CUDAGraph()
                with _MMRec(), torch.cuda.graph(g, pool=pools[r] if pools else None, stream=eng.streams[r],
                                                capture_error_mode=capture_error_mode):
                    body(r)
                del g
    finally:
        _STATE["probe"] = None
    w1 = time.time()
    tpr = time.perf_counter()
    s1, c1 = _slots(roots), _scalars(roots)
    tsn2 = time.perf_counter()
    new_ = [k for k in s1 if k not in s0]
    reb_ = [k for k in s1 if k in s0 and s1[k] != s0[k]]
    sch_ = [k for k in c1 if c0.get(k, object()) != c1[k]]
    for d in eng.devs:
        torch.cuda.synchronize(d)
    nt = nm = 0
    for dev, k in rec["triton"]:
        if getattr(k, "module", None) is None:
            with torch.cuda.device(dev):
                k._init_handles()
            nt += 1
    for func, am, km in rec["mm"]:
        key = (str(func), am, tuple(sorted(km.items())))
        if key in _MM_DONE:
            continue
        _MM_DONE.add(key)
        dev = next(m[6] for m in list(am) + list(km.values()) if m[0] == "T")
        keep = []

        def mk(m):
            if m[0] != "T":
                return m[1]
            t, raw = _scratch(m, dev)
            keep.append(raw)
            return t
        r = next((i for i, d_ in enumerate(eng.devs) if torch.device(d_).index == dev), None)
        st_ = eng.streams[r] if r is not None else None
        with torch.cuda.device(dev), (torch.cuda.stream(st_) if st_ is not None else contextlib.nullcontext()):
            func(*[mk(m) for m in am], **{k: mk(m) for k, m in km.items()})
        nm += 1
    for d in eng.devs:
        torch.cuda.synchronize(d)
    out = dict(triton=nt, triton_seen=len(rec["triton"]), mm=nm, mm_seen=len(rec["mm"]), s=round(time.perf_counter() - t0, 3),
               t0=round(w0, 6), t1=round(w1, 6), new_slots=len(new_), rebound_slots=len(reb_), scalar_changes=len(sch_),
               snap_s=round((tsn - t0) + (tsn2 - tpr), 3), probe_s=round(tpr - tsn, 3),
               load_s=round(time.perf_counter() - tsn2, 3), check=CHECK)
    if new_ or reb_ or sch_:
        lg(f"head_preload {tag}: PROBE SIDE EFFECTS new {new_[:6]} rebound {reb_[:6]} scalars {sch_[:8]}")
    if tag:
        lg(f"head_preload {tag}: {out}")
    return out
