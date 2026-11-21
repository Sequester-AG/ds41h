"""tail_gather: the mixed step's
T stage in one launch (tail_gather.cu / tail_gather_cabi.h): bucket A's tail rows into bucket T for every key (a byte copy
of whole rows, bitwise equal to torch.index_select of contiguous row-major buffers) plus T's padding rows (index layers'
ti -1 / tl 0; ENG_L20_TAIL's si20 / sl20 rows [n, R) from T's si / sl).
  ext()         the extension (build dir kernels/tail_gather/build or TAIL_GATHER_BUILD); lib_path() its .so (C ABI table
                "tail_gather": "tail_gather.run", "tail_gather.t_stage", for nd's native T_STAGE)
  gather(ix, srcs, dsts, err=None)          explicit keys (pybind -> fpr_tail_gather_run)
  TgKeys / keys_bytes(R, n_idx, slots)      a view's FprTgKeys (ctypes; the bytes the exporter publishes as tkeys)
  t_stage(ix, n, a_keys, t_keys, l20, clear, device, err=None)   pybind -> fpr_t_stage_run (the function nd calls)
Engine wiring: engine.py Engine._t_stage / _tg_keys (ENG_TAIL_GATHER=1 + ENG_TAIL_GATHER_KERNEL=1). The ctypes part needs
no build (the exporter imports it without the extension)."""
from __future__ import annotations

import ctypes as C
import functools
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CABI_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine", "mixed_cxx")
ROOT = os.environ.get("TAIL_GATHER_BUILD", os.path.join(HERE, "build"))

# tail_gather_cabi.h: FprTgKeys slots (FPR_TG_*)
X, A_PRE, A_POST, A_COMB, WS_Q, SI, SL, SI20, SL20, MOE_OUT, XM2, F_POST, F_COMB, F_PRE = range(14)
TI0, TL0, N_KEYS, MAX_IDX, SLOTS, KEYS_VERSION = 14, 18, 22, 4, 24, 1


class TgKeys(C.Structure):
    _fields_ = [("version", C.c_int32), ("n_idx", C.c_int32), ("R", C.c_int64), ("p", C.c_void_p * SLOTS),
                ("row_bytes", C.c_int64 * SLOTS), ("rows", C.c_int64 * SLOTS)]


def keys_bytes(R: int, n_idx: int, slots: dict) -> bytes:
    """slots: {FPR_TG_* slot: tensor or None}. Every tensor contiguous with rows along dim 0."""
    k = TgKeys()
    k.version, k.n_idx, k.R = KEYS_VERSION, n_idx, R
    assert 0 <= n_idx <= MAX_IDX, n_idx
    for sl, t in slots.items():
        if t is None:
            continue
        assert t.is_contiguous() and t.dim() >= 1, (sl, tuple(t.shape), t.stride())
        k.p[sl] = t.data_ptr()
        k.row_bytes[sl] = t.numel() // max(t.shape[0], 1) * t.element_size()
        k.rows[sl] = t.shape[0]
    return bytes(k)


def _md5() -> str:
    h = hashlib.md5()
    for f in ("tail_gather.cu", "tail_gather_cabi.h"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


@functools.cache
def ext():
    from torch.utils.cpp_extension import load
    bd = os.path.join(ROOT, "tail_gather")
    os.makedirs(bd, exist_ok=True)
    txt = f'#pragma once\n#define TAIL_GATHER_SRC_MD5 "{_md5()}"\n'
    hp = os.path.join(bd, "tail_gather_md5.h")
    if not os.path.exists(hp) or open(hp).read() != txt:   # (only when it changed: torch's build hash / ninja's mtimes)
        with open(hp, "w") as fh:
            fh.write(txt)
    m = load(name="tail_gather", sources=[os.path.join(HERE, "tail_gather.cu")], build_directory=bd, verbose=False,
             extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"],
             extra_include_paths=[HERE, CABI_DIR, bd])
    assert m.keys_size() == C.sizeof(TgKeys), (m.keys_size(), C.sizeof(TgKeys))
    return m


def lib_path() -> str:
    return ext().__file__


def gather(ix, srcs, dsts, err=None) -> None:
    ext().gather(ix, list(srcs), list(dsts), err)


def t_stage(ix, n: int, a_keys: bytes, t_keys: bytes, l20: bool, clear: bool, device: int, err=None) -> None:
    ext().t_stage(ix, int(n), a_keys, t_keys, bool(l20), bool(clear), int(device), err)
