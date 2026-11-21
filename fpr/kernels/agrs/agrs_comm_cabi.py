"""  ext()        the extension agrs_commc: agrs_comm_cabi.cu (agrs.cu #included unchanged, its own TU, commbench.ext()'s
               CFLAGS) in kernels/agrs/build/agrs_commc; table fpr_agrs_comm_table ("agrs_comm"); the served bindings
               + cabi_ce_copy(dst, dst_off, src, src_off, nbytes, stream, wait_before, record_after).
  ctypes       AgrsCeCopyArgs + pack_ce_copy(...) -> bytes, for fpr/engine/mixed_cxx/cabi_test.
  build        CUDA_VISIBLE_DEVICES= python3 agrs_comm_cabi.py
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(HERE)))
CABI_DIR = os.path.join(PHYS, "engine", "mixed_cxx")
BUILD = os.path.join(HERE, "build", "agrs_commc")
CFLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"]   # = commbench.ext() (the served agrs_comm)


@functools.cache
def ext():
    from torch.utils.cpp_extension import load
    os.makedirs(BUILD, exist_ok=True)
    h = hashlib.md5()
    for f in ("agrs.cu", "agrs_comm_cabi.cu", "agrs_comm_cabi.h"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    txt = f'#pragma once\n#define AGRS_COMMC_SRC_MD5 "{h.hexdigest()}"\n'
    p = os.path.join(BUILD, "agrs_commc_md5.h")
    if not os.path.exists(p) or open(p).read() != txt:
        open(p, "w").write(txt)
    return load(name="agrs_commc", sources=[os.path.join(HERE, "agrs_comm_cabi.cu")], build_directory=BUILD, verbose=False,
                extra_cuda_cflags=CFLAGS, extra_include_paths=[HERE, CABI_DIR, BUILD])


def lib_path() -> str:
    return ext().__file__


class FprCall(ctypes.Structure):
    _fields_ = [("stream", ctypes.c_void_p), ("device", ctypes.c_int), ("rank", ctypes.c_int)]


class AgrsCeCopyArgs(ctypes.Structure):
    _fields_ = [("call", FprCall), ("dst", ctypes.c_void_p), ("src", ctypes.c_void_p), ("nbytes", ctypes.c_int64),
                ("wait_before", ctypes.c_void_p), ("record_after", ctypes.c_void_p)]


def check() -> None:
    assert ctypes.sizeof(AgrsCeCopyArgs) == ext().cabi_args_size(), (ctypes.sizeof(AgrsCeCopyArgs), ext().cabi_args_size())


def ev_handle(ev) -> int:
    """cudaEvent_t of a torch.cuda.Event (0 = none). The event must exist: torch creates it lazily on the first record."""
    return 0 if ev is None else int(ev.cuda_event)


def pack_ce_copy(dst, dst_off, src, src_off, nbytes, stream, wait_before=None, record_after=None) -> bytes:
    a = AgrsCeCopyArgs(FprCall(stream.cuda_stream, src.device.index, 0), dst.data_ptr() + int(dst_off), src.data_ptr() + int(src_off),
                       int(nbytes), ev_handle(wait_before) or None, ev_handle(record_after) or None)
    return ctypes.string_at(ctypes.addressof(a), ctypes.sizeof(a))


if __name__ == "__main__":
    print(f"agrs_commc built: {ext().__file__}", flush=True)
