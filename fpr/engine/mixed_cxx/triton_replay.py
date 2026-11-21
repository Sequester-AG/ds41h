"""triton_replay: record Triton's own launches (Python thread, warm-up) as FprTrRecord structs for triton_replay.h, the C++
driver's replay of those kernels.

    with recording() as recs:                     # patches triton JITFunction.run: launches run as usual, and are recorded
        vllm_wrapper(...)
    for r in recs: r.record (ctypes FprTrRecord: CUfunction on that device, launch config, compiled params + key class),
                   r.values() (uint64 per Python-level argument), r.grid, r.bound (name -> value), r.key (readable)
    ext().launch(bytes(r.record), r.values_bytes(vals), *r.grid, stream, 0, 0)   # the test entry (triton_replay_ext.cpp)

A record is valid for every launch whose values fall in its key class (fpr_tr_check; same rules as Triton's binder, see
triton_replay.h). warm_set(recs) / dump(recs): the warmed classes, readable (stage 2's AOT list). Not supported (assert):
tuple arguments, tensor descriptors, non-zero global / profile scratch without a caller buffer.
"""
from __future__ import annotations

import contextlib
import ctypes as C
import os
import struct

import torch

MAXP = 64
TYPES = {"i1": 2, "i8": 3, "i16": 4, "i32": 5, "i64": 6, "u1": 7, "u8": 8, "u16": 9, "u32": 10, "u64": 11, "fp16": 12,
         "bf16": 13, "fp32": 14, "f32": 14, "fp64": 15}
SPEC = {None: 0, "": 1, "D": 2}
HERE = os.path.dirname(os.path.abspath(__file__))


class Param(C.Structure):
    _fields_ = [("type", C.c_int32), ("spec", C.c_int32), ("align", C.c_int32), ("dns", C.c_int32), ("pyarg", C.c_int32),
                ("pad", C.c_int32)]


class Record(C.Structure):
    _fields_ = [("function", C.c_void_p), ("device", C.c_int32), ("num_warps", C.c_int32), ("num_ctas", C.c_int32),
                ("shared", C.c_int32), ("launch_pdl", C.c_int32), ("launch_coop", C.c_int32), ("gscratch_size", C.c_int64),
                ("gscratch_align", C.c_int64), ("pscratch_size", C.c_int64), ("pscratch_align", C.c_int64),
                ("n_params", C.c_int32), ("n_const1", C.c_int32), ("p", Param * MAXP), ("const1_pyarg", C.c_int32 * MAXP),
                ("name", C.c_char * 96), ("key", C.c_char * 160)]


def _bits(v, ty) -> int:
    """The uint64 Triton's extractor stores for a Python value of compiled type ty (pointers: addresses)."""
    if v is None:
        return 0
    if hasattr(v, "data_ptr"):
        return int(v.data_ptr()) & 0xFFFFFFFFFFFFFFFF
    if ty in ("fp32", "f32"):
        return struct.unpack("<I", struct.pack("<f", float(v)))[0]
    if ty == "fp64":
        return struct.unpack("<Q", struct.pack("<d", float(v)))[0]
    if ty == "fp16":
        return struct.unpack("<H", struct.pack("<e", float(v)))[0]
    if ty == "bf16":
        return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16
    return int(v) & 0xFFFFFFFFFFFFFFFF


class Rec:
    """One recorded launch."""

    def __init__(self, fn, kernel, device, bound, specialization, grid):
        self.fn, self.kernel, self.device, self.bound = fn, kernel, int(device), dict(bound)
        self.names = list(self.bound)
        self.spec = [tuple(s) for s in specialization]
        g = tuple(int(x) for x in grid) + (1, 1)
        self.grid = g[:3]
        md = kernel.metadata
        r = Record()
        r.function = int(kernel.function)
        r.device = self.device
        r.num_warps, r.num_ctas, r.shared = int(md.num_warps), int(getattr(md, "num_ctas", 1)), int(md.shared)
        r.launch_pdl = int(bool(getattr(md, "launch_pdl", False)))
        r.launch_coop = int(bool(getattr(md, "launch_cooperative_grid", False)))
        r.gscratch_size, r.gscratch_align = int(md.global_scratch_size), int(md.global_scratch_align)
        r.pscratch_size, r.pscratch_align = int(getattr(md, "profile_scratch_size", 0)), int(getattr(md, "profile_scratch_align", 1))
        kp = {p.name: p for p in fn.params}
        n, nc, keyparts = 0, 0, []
        self.param_types = []
        for i, (name, (ty, sp)) in enumerate(zip(self.names, self.spec)):
            assert not isinstance(ty, tuple) and not str(ty).startswith("tensordesc"), f"{name}: {ty} not supported"
            p = kp[name]
            if ty == "constexpr":
                if not p.is_constexpr and sp == 1 and isinstance(self.bound[name], int) and not isinstance(self.bound[name], bool):
                    r.const1_pyarg[nc] = i        # an int specialized to constexpr 1: must stay 1 for this record
                    nc += 1
                keyparts.append(f"{name}=c{sp!r}"[:24] if not p.is_constexpr else f"{name}={sp!r}"[:24])
                continue
            assert n < MAXP, "too many kernel parameters"
            q = r.p[n]
            q.type = 1 if str(ty).startswith("*") else TYPES[ty]
            q.spec = SPEC[sp]
            q.align = int(not p.do_not_specialize_on_alignment)
            q.dns = int(bool(p.do_not_specialize))
            q.pyarg = i
            self.param_types.append(ty)
            keyparts.append(f"{name}:{ty}{'' if sp is None else '/' + sp}")
            n += 1
        r.n_params, r.n_const1 = n, nc
        r.name = fn.__name__.encode()[:95]
        self.key = (f"{fn.__name__} dev{self.device} w{r.num_warps} " + " ".join(keyparts))
        r.key = self.key.encode()[:159]
        self.record = r

    def values(self, **override) -> list:
        """uint64 per Python-level argument (override: name -> new value, e.g. a remapped pointer / row count)."""
        out = []
        tys = {name: ty for name, (ty, _) in zip(self.names, self.spec)}
        for name in self.names:
            v = override.get(name, self.bound[name])
            out.append(_bits(v, tys[name]))
        return out

    @staticmethod
    def values_bytes(vals) -> bytes:
        return struct.pack(f"<{len(vals)}Q", *vals)


@contextlib.contextmanager
def recording(filter_fn=None):
    """Record every Triton launch (JITFunction.run, warmup False) made inside the block; launches still run."""
    from triton import knobs
    from triton.runtime import driver
    from triton.runtime import jit as J

    recs: list[Rec] = []
    orig = J.JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        kernel = orig(self, *args, grid=grid, warmup=warmup, **kwargs)
        if warmup or kernel is None or (filter_fn is not None and not filter_fn(self)):
            return kernel
        kw = dict(kwargs)
        kw["debug"] = kw.get("debug", self.debug) or knobs.runtime.debug
        kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        device = driver.active.get_current_device()
        binder = self.device_caches[device][4]
        bound, specialization, _ = binder(*args, **kw)
        g = grid(bound) if callable(grid) else grid
        recs.append(Rec(self, kernel, device, bound, specialization, g))
        return kernel

    J.JITFunction.run = run
    try:
        yield recs
    finally:
        J.JITFunction.run = orig


def dump(recs) -> list[str]:
    """The distinct key classes of a recording (readable; stage 2 turns them into AOT cubins)."""
    return sorted({r.key for r in recs})


_EXT: dict = {}


def ext():
    """triton_replay_ext (pybind over triton_replay.h): check(record, vals) / launch(record, vals, gx, gy, gz, stream, gs, ps)."""
    if "m" not in _EXT:
        from torch.utils.cpp_extension import load

        bdir = os.environ.get("TRITON_REPLAY_BUILD") or os.path.join(HERE, "build", "triton_replay")
        os.makedirs(bdir, exist_ok=True)
        _EXT["m"] = load(name="triton_replay_ext", sources=[os.path.join(HERE, "triton_replay_ext.cpp")], extra_cflags=["-O3", "-std=c++17"],
                         extra_include_paths=[HERE], extra_ldflags=["-lcuda"], with_cuda=True, build_directory=bdir, verbose=False)
    return _EXT["m"]
