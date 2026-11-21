"""mixed_cxx_ltwarm: at Driver init, before any mixed step
warm fpr_lt.h's CLOSED algo set of every .so whose pool-thread GEMMs go through fpr_lt, so no pool-thread GEMM ever first-loads a cuBLASLt kernel module
mid-exchange.

Per .so ("entry"): keys file <base>.<entry>.txt. Recording run: FPR_LT_DUMP=<base> appends every planned key of each .so to
<base>.<entry>.txt every FPR_LT_DUMP_EVERY Python mixed steps (rank 0; gate harnesses kill the engine, so no exit dump).
Warm: FPR_LT_WARM_FILES=<base>[:<base>...]; each distinct site shape (batch, N, K, lda, ldw, ldy, sa, sw, sy, out_f32) is
re-expanded over the full served row set M = decode buckets U multiples of mixed_round up to max(max_mixed_rows,
max_tail_rows) U W x those (gathered-row sites: the shared expert's Tall), then warm(dev, shapes, n, streams, ns) per rank with
the streams the pool threads issue on (E.streams[r], E.side[r], the AGRS comm streams: workspaces pre-allocated).
fpr_lt.h prints "fpr_lt: warm dev d: n shapes, k distinct algos, b not planned" per call. FPR_LT_WARM=0 skips the warm.
"""
from __future__ import annotations

import ctypes as C
import os
import time

import torch

ON = os.environ.get("FPR_LT_WARM", "1") == "1"
FILES = [p for p in os.environ.get("FPR_LT_WARM_FILES", "").split(":") if p]
DUMP = os.environ.get("FPR_LT_DUMP", "")
DUMP_EVERY = int(os.environ.get("FPR_LT_DUMP_EVERY", "200") or 0)


class Shape(C.Structure):
    _fields_ = [(n, C.c_int64) for n in ("batch", "M", "N", "K", "lda", "ldw", "ldy", "sa", "sw", "sy")] + [
        ("out_f32", C.c_int32), ("pad_", C.c_int32)]


def m_set(E, W: int) -> list[int]:
    cfg = E.cfg
    rnd = max(1, int(cfg.mixed_round))
    top = max(int(getattr(E, "max_tail_rows", 0) or 0), -(-int(cfg.max_mixed_rows) // rnd) * rnd)
    ms = {int(b) for b in cfg.buckets} | set(range(rnd, top + 1, rnd))
    return sorted(ms | {W * m for m in ms})


def read_sites(paths: list[str], ms_seen: dict | None = None) -> set[tuple]:
    """key line: batch M N K lda ldw ldy sa sw sy out_f32 [site]; M is dropped (re-expanded); missing files are skipped.
    ms_seen (optional): site -> set of recorded M values (for the ATen sweep of sites whose M is not a bucket)"""
    sites = set()
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            for ln in f:
                v = ln.split()
                if len(v) < 11 or not v[0].lstrip("-").isdigit():
                    continue
                b, m_, n, k, lda, ldw, ldy, sa, sw, sy, of = (int(x) for x in v[:11])
                sites.add((b, n, k, lda, ldw, ldy, sa, sw, sy, of))
                if ms_seen is not None:
                    ms_seen.setdefault((b, n, k, lda, ldw, ldy, sa, sw, sy, of), set()).add(m_)
    return sites


def shapes_for(sites: set[tuple], ms: list[int]) -> list[tuple]:
    out = []
    for (b, n, k, lda, ldw, ldy, sa, sw, sy, of) in sorted(sites):
        for m in ms:
            if b > 1 and ((sa >= lda and m * lda > sa) or (sy >= ldy and m * ldy > sy)):
                continue          # (batched, non-interleaved: a batch stride must hold m rows; interleaved layouts fit any m)
            out.append((b, m, n, k, lda, ldw, ldy, sa, sw, sy, of))
    return out


SWEEP = os.environ.get("FPR_LT_ATEN_SWEEP", "1") == "1"


def m_sweep(top: int) -> list[int]:
    """M values whose stock cuBLAS calls load every kernel the heuristic can pick up to `top` (the largest call: max(recorded M,
    W x max rows)). Now EVERY M in 1..top (exact coverage of every gathered-row call the sites can see)
    ~0.12 ms per GEMM -> ~2 x top GEMMs per device.
    FPR_LT_ATEN_SWEEP_STEP=<n> (> 1) restores the old sparse grid (diag only)."""
    step = int(os.environ.get("FPR_LT_ATEN_SWEEP_STEP", "1") or 1)
    if step <= 1:
        return list(range(1, top + 1))
    ms = set(range(1, min(1024, top) + 1))
    for base in range(1024, top + 1, max(16, step)):
        ms.update(range(base, min(base + 16, top + 1)))
    ms.add(top)
    return sorted(m for m in ms if 1 <= m <= top)


def aten_sweep(E, sites_ms: dict, ms_bucket: list[int], W: int) -> tuple[int, int]:
    """For every 2-D site with a recorded M outside the warmed bucket set (e.g. agrs's prefill shared expert: M = the real
    gathered prefill token count), run the STOCK ATen GEMM (torch.mm / mm out=fp32) for that (N, K, layout) over m_sweep on
    every device, so every cuBLAS(Lt) kernel module the heuristic can pick there is loaded before serving; serving keeps
    calling ATen with the real M (fpr_lt's 'unwarmed key -> ATen' path): bitwise as today, no pool-thread lazy load."""
    bset = set(ms_bucket)
    todo = [(site, ms) for site, ms in sites_ms.items() if site[0] == 1 and any(m not in bset for m in ms)]
    if not todo:
        return 0, 0
    top = max(max(ms) for _, ms in todo)
    top = max(top, max(ms_bucket))   # m_set already holds W * rows -- the old W * max() swept to 4x the
                                     # largest possible call (37,888 vs 9,472 / 13,568 rows): 10,240 M, 23 s of startup)
    msw = m_sweep(top)
    n_calls = 0
    for r, d in enumerate(E.devs):
        dev = torch.device(d)
        t_dev = time.perf_counter()
        with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
            for (b, n, k, lda, ldw, ldy, sa, sw, sy, of), _ms in todo:
                w = torch.zeros(n, k, dtype=torch.bfloat16, device=dev)
                a_big = torch.zeros(top, max(k, lda), dtype=torch.bfloat16, device=dev)
                for m in msw:
                    a = a_big[:m, :k] if lda == k else a_big.as_strided((m, k), (lda, 1))
                    if of:
                        torch.mm(a, w.t(), out_dtype=torch.float32)
                    else:
                        torch.mm(a, w.t())
                    n_calls += 1
                del a_big, w
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()     # hand the sweep's operand / output segments back before serving: no
                                         # sweep-sized cached blocks left to fragment the prefill allocations
        print(f"mixed_cxx_ltwarm: ATen sweep dev {dev.index}: {len(todo)} sites x {len(msw)} M in {time.perf_counter() - t_dev:.2f} s", flush=True)
    return len(todo), n_calls


KWARM = os.environ.get("FPR_KWARM", "1") == "1"


def _loaded_fpr_sos() -> list[str]:
    """every shared object mapped into this process from the FPR tree (fpr/ or ds41h/ build dirs), via /proc/self/maps:
    extension modules AND ctypes-loaded C ABI libraries (fpr_p1k, gatherk, ...), which sys.modules would miss"""
    out = []
    try:
        with open("/proc/self/maps") as f:
            for ln in f:
                p = ln.split()[-1] if len(ln.split()) >= 6 else ""
                if p.endswith(".so") and ("/fpr/" in p or "/ds41h/" in p) and p not in out:
                    out.append(p)
    except OSError:
        pass
    return out


def kernel_warm(E) -> list[str]:
    """(plan part 1) per-.so kernel residency at driver init: every loaded FPR .so that exports
    `extern "C" int fpr_kwarm(int dev)` (cudaFuncGetAttributes on each __global__ it can launch) gets it called on every device,
    so no pool thread first-loads one of its kernel modules mid-exchange; + one dummy launch of the torch elementwise kernels
    the pool threads were seen first-launching (ltg1t: logical_and). Returns the .so basenames that had an entry."""
    had = []
    for p in _loaded_fpr_sos():
        try:
            fn = getattr(C.CDLL(p), "fpr_kwarm", None)
        except OSError:
            fn = None
        if fn is None:
            continue
        fn.argtypes, fn.restype = [C.c_int], C.c_int
        rcs = [int(fn(torch.device(d).index)) for d in E.devs]   # (fpr_kwarm returns the kernel count, < 0 = error)
        bad = [r for r, v in enumerate(rcs) if v < 0]
        had.append(os.path.basename(p) + f" ({rcs[0]} kernels" + (f"; ERROR on ranks {bad}" if bad else "") + ")")
    for r, d in enumerate(E.devs):
        dev = torch.device(d)
        with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
            t = torch.zeros(64, dtype=torch.bool, device=dev)
            torch.logical_and(t, t)
            torch.logical_or(t, t)
            torch.logical_not(t)
            # (ltg1v: the 4 remaining pool-thread loads = one classic-cuBLAS bf16 GEMM per rank at the pool threads' start, i.e.
            # pieces_moe / agrs thread-init's 64 x 128 handle mm: the same call here loads its kernel for this context)
            a_ = torch.zeros(64, 128, dtype=torch.bfloat16, device=dev)
            c_ = torch.empty(64, 64, dtype=torch.bfloat16, device=dev)
            torch.mm(a_, a_.t(), out=c_)
            # 4 rank-thread loads of gpu_reduce_kernel<BFloat16, MeanOps> <- at::mean_out <- driver bd()'s
            # DSpark aux row: aux[:, i] = X.mean(1), X bf16 [R, HC, D], out a strided slice of aux [R, n_aux, D]) the same op at
            # several R (the reduce picks vectorization / launch shapes by size and alignment), into the same strided layout
            HC, Dm, NA = _aux_dims(E)
            for R_ in (1, 7, 64, 333, 1024, 4096, 8192):
                X_ = torch.zeros(R_, HC, Dm, dtype=torch.bfloat16, device=dev)
                aux_ = torch.zeros(R_, NA, Dm, dtype=torch.bfloat16, device=dev)
                torch.mean(X_, 1, out=aux_[:, NA - 1])
                aux_[:, 0].copy_(X_.mean(1))
                del X_, aux_
        torch.cuda.synchronize(dev)
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    return had


def _aux_dims(E):
    """(HC, D, n_aux) of the bd piece's aux mean: X = [R, HC, D] (mHC rows), aux = [R, len(AUX_LAYERS), D]."""
    import sys
    mod = sys.modules.get(type(E).__module__)
    na = len(getattr(mod, "AUX_LAYERS", ()) or ()) or 3
    mc = getattr(E, "mcfg", None)
    d = int(getattr(mc, "hidden_size", 0) or getattr(mc, "dim", 0) or 5120)
    hc = int(getattr(mc, "hc_mult", 0) or getattr(mc, "mhc", 0) or 4)
    return hc, d, na


def _libs(drv):
    """[(entry, CDLL)] of every loaded .so with an fpr_lt warm entry"""
    out = [("fpr_mixed_cxx_lt_warm", C.CDLL(drv.x.__file__))]
    try:
        import agrs_pmws_cabi as APC
        out.append(("fpr_agrs_lt_warm", C.CDLL(APC.ext().__file__)))
    except Exception:  # noqa: BLE001  (no agrs C ABI in this build: nothing to warm there)
        pass
    return out


_DUMPS = []          # [(entry, fn)]
_NSTEP = [0]


def step_params(drv, r, s, sp) -> None:
    """(owners' per-step hook) FPR_LT_DUMP: append each .so's planned keys every FPR_LT_DUMP_EVERY mixed steps (rank 0)"""
    if not _DUMPS or DUMP_EVERY <= 0 or r != 0:
        return
    _NSTEP[0] += 1
    if _NSTEP[0] % DUMP_EVERY == 0:
        for entry, fn in _DUMPS:
            drv.n["lt_dump_keys_" + entry] = fn(f"{DUMP}.{entry}.txt".encode())
        drv.n["lt_dumps"] += 1


KWARM_ALL = os.environ.get("FPR_KWARM_ALL", "1") == "1"


def setup(drv) -> None:
    """The cuBLASLt warm list + kernel warms (_setup), then -- always, as the very last init step, after every owner has
    loaded its .so -- cudaFuncGetAttributes on every nvcc kernel of every ALREADY-mapped .so under the release's fpr/ on every
    device, from this thread (never dlopens). FPR_KWARM_ALL=0 skips."""
    try:
        _setup(drv)
    finally:
        if KWARM_ALL:
            E = drv.eng
            try:
                import kwarm_all
                root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/fpr
                kwarm_all.warm_loaded(root, [torch.device(d).index for d in E.devs], E.log)
            except Exception as e:  # noqa: BLE001
                print(f"mixed_cxx_ltwarm: kwarm_all FAILED ({type(e).__name__}: {str(e)[:160]})", flush=True)


def _setup(drv) -> None:
    E = drv.eng
    if KWARM:
        try:
            had = kernel_warm(E)
            print(f"mixed_cxx_ltwarm: kernel warm: fpr_kwarm in {len(had)} .so {had}; torch logical_* + thread-init 64x128 mm warmed on "
                  f"{len(E.devs)} devices (FPR_KWARM=0 skips)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"mixed_cxx_ltwarm: kernel warm FAILED ({type(e).__name__}: {str(e)[:160]})", flush=True)
    libs = _libs(drv)
    if DUMP:
        for entry, lib in libs:
            f = getattr(lib, entry + "_dump", None)
            if f is not None:
                f.argtypes, f.restype = [C.c_char_p], C.c_int
                _DUMPS.append((entry, f))
        print(f"mixed_cxx_ltwarm: FPR_LT_DUMP={DUMP} every {DUMP_EVERY} mixed steps: {[e for e, _ in _DUMPS]}", flush=True)
    if not ON:
        return
    if not FILES:
        print("mixed_cxx_ltwarm: FPR_LT_WARM_FILES unset: no warm list (record one with FPR_LT_DUMP)", flush=True)
        return
    try:
        import agrs_engine as AGE
    except ImportError:
        AGE = None
    W = len(E.devs)
    ms = m_set(E, W)
    t0 = time.perf_counter()
    for entry, lib in libs:
        fn = getattr(lib, entry, None)
        if fn is None:
            print(f"mixed_cxx_ltwarm: {entry} missing: not warmed", flush=True)
            continue
        fn.argtypes = [C.c_int, C.POINTER(Shape), C.c_int, C.POINTER(C.c_void_p), C.c_int]
        fn.restype = C.c_int
        sites_ms = {}
        sites = read_sites([f"{b}.{entry}.txt" for b in FILES], sites_ms)
        sh = shapes_for(sites, ms)
        if not sh:
            print(f"mixed_cxx_ltwarm: {entry}: no recorded site shapes: not warmed", flush=True)
            continue
        arr = (Shape * len(sh))(*[Shape(*x[:10], x[10], 0) for x in sh])
        for r, d in enumerate(E.devs):
            ss = [E.streams[r]]
            if getattr(E, "side", None) is not None:
                ss.append(E.side[r])
            if AGE is not None:
                ss.extend(AGE._cstreams(E, r))
            sp = (C.c_void_p * len(ss))(*[C.c_void_p(x.cuda_stream) for x in ss])
            dev = torch.device(d).index
            with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
                k = fn(dev, arr, len(sh), sp, len(ss))
                torch.cuda.synchronize(dev)
            drv.n[f"lt_warm_{entry}_r{r}"] = k
        print(f"mixed_cxx_ltwarm: {entry}: {len(sites)} site shapes x {len(ms)} M = {len(sh)} keys per rank", flush=True)
        if SWEEP:
            t1 = time.perf_counter()
            ns, nc = aten_sweep(E, sites_ms, ms, W)
            print(f"mixed_cxx_ltwarm: {entry}: ATen sweep for {ns} non-bucket-M sites: {nc} stock GEMMs over {len(E.devs)} devices "
                  f"in {time.perf_counter() - t1:.1f} s (loads every cuBLAS kernel those shapes can pick; FPR_LT_ATEN_SWEEP=0 skips)", flush=True)
    print(f"mixed_cxx_ltwarm: warm done in {time.perf_counter() - t0:.1f} s ({W} ranks)", flush=True)
