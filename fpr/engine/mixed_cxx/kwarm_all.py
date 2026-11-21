"""kwarm_all: load every CUDA kernel of the given extension
.so files on every device, on the CALLING thread, before any rank thread runs. Under CUDA_MODULE_LOADING=LAZY a kernel's module
loads at its first launch
nvcc emits, per kernel instance (templates included), a host stub
_Z<n>__device_stub__<mangled kernel name minus its leading '_'><stub params> (or, when the stub was inlined, only its
static guard variable _ZGVZL<n>__device_stub__<...>) next to the host handle symbol _<mangled>; the
handle's address is what cudaFuncGetAttributes takes (and loads the module of). Symbols from `nm --defined-only` (local ones
too: anonymous-namespace kernels), load base from /proc/self/maps (offset 0 mapping), the process's own libcudart (the one
the .so registered its fatbins with). Returns {so: (kernels found, warmed on every device)}."""
from __future__ import annotations

import bisect
import ctypes as C
import os
import re
import subprocess

_STUB = re.compile(r"^(?:_Z\d+|_ZGVZL?\d+|_ZZL?\d+)__device_stub__(.+)$")   # the stub, or its static guard variable (stubs nvcc inlined)


def _maps() -> list[tuple[int, int, str]]:
    out = []
    with open("/proc/self/maps") as f:
        for ln in f:
            v = ln.split()
            if len(v) >= 6:
                a, _, off, _, _, path = v[:6]
                out.append((int(a.split("-")[0], 16), int(off, 16), path))
    return out


def _cudart():
    for _, _, p in _maps():
        if os.path.basename(p).startswith("libcudart.so"):
            return C.CDLL(p)
    raise RuntimeError("kwarm_all: no libcudart mapped")


_LNK = re.compile(r"(?<=[0-9A-Za-z_])L(?=[1-9])")


def _canon(x: str) -> str:
    """the internal-linkage marker: the handle of a static / anonymous-namespace kernel mangles as ...L17name..., its stub's
    embedded name as ...17name..."""
    return _LNK.sub("", x)


def kernels(so: str) -> list[tuple[str, int]]:
    """(symbol, address in this process) of every kernel host handle in so (must already be loaded)"""
    base = None
    real = os.path.realpath(so)
    for a, off, p in _maps():
        if off == 0 and os.path.realpath(p) == real:
            base = a if base is None else min(base, a)
    if base is None:
        raise RuntimeError(f"kwarm_all: {so} not mapped")
    txt = subprocess.run(["nm", "--defined-only", real], capture_output=True, text=True, check=True).stdout
    syms = {}
    stubs = []
    for ln in txt.splitlines():
        v = ln.split()
        if len(v) != 3:
            continue
        m = _STUB.match(v[2])
        if m:
            stubs.append(_canon("_" + m.group(1)))
        elif v[1] in "tTwW" and v[2].startswith("_Z"):
            syms[_canon(v[2])] = int(v[0], 16)
    names = sorted(syms)
    out = []
    for s in stubs:
        i = bisect.bisect_right(names, s) - 1
        while i >= 0 and s.startswith(names[i][:8]):     # the longest symbol that is a prefix of the stub's name
            if s.startswith(names[i]):
                out.append((names[i], base + syms[names[i]]))
                break
            i -= 1
    return sorted(set(out))


class _Attr(C.Structure):
    _fields_ = [("raw", C.c_byte * 1024)]


def warm(sos: list[str], devs: list[int]) -> dict:
    rt = _cudart()
    rt.cudaFuncGetAttributes.argtypes = [C.c_void_p, C.c_void_p]
    rt.cudaGetDevice.argtypes = [C.POINTER(C.c_int)]
    rt.cudaSetDevice.argtypes = [C.c_int]
    prev = C.c_int(0)
    rt.cudaGetDevice(C.byref(prev))
    res = {}
    try:
        for so in sos:
            ks = kernels(so)
            ok = 0
            for _, addr in ks:
                good = True
                for d in devs:
                    rt.cudaSetDevice(int(d))
                    good &= rt.cudaFuncGetAttributes(C.byref(_Attr()), C.c_void_p(addr)) == 0
                ok += good
            rt.cudaGetLastError()
            res[os.path.basename(so)] = (len(ks), ok)
    finally:
        rt.cudaSetDevice(prev.value)
    return res


def loaded(root: str) -> list[str]:
    """the .so files under root that this process has mapped (never dlopens anything new: a second build of an extension
    would re-register its TORCH_LIBRARY namespace)"""
    root = os.path.realpath(root)
    out = set()
    for _, off, p in _maps():
        if off == 0 and p.startswith("/"):
            rp = os.path.realpath(p)
            if rp.startswith(root + os.sep) and rp.endswith(".so"):
                out.add(rp)
    return sorted(out)


def warm_loaded(root: str, devs: list[int], log=print) -> dict:
    """warm() over loaded(root); one summary line. Errors per .so are counted, never raised."""
    import time
    t0 = time.time()
    sos = loaded(root)
    res = {}
    bad = []
    for so in sos:
        try:
            res.update(warm([so], devs))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{os.path.basename(so)}: {str(e)[:80]}")
    tot = sum(n for n, _ in res.values())
    ok = sum(g for _, g in res.values())
    log(f"kwarm_all: {len(sos)} loaded .so under {root}: {tot} kernels, {ok} loaded on all {len(devs)} devices in "
        f"{time.time() - t0:.2f} s{'; failed: ' + '; '.join(bad) if bad else ''}")
    return res
