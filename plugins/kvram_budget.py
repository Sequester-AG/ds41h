#!/usr/bin/env python3
"""Usable GiB for the KV-RAM host region.

The region is a pre-faulted file in /dev/shm. Two limits can apply: the tmpfs size AND a per-user tmpfs quota
(/dev/shm mounted usrquota) -- exceeding either makes the pre-fault fail (EFAULT).
The shared Engram host tables (2 x 97.3 GiB) are charged to the same user and are (re)created by a server with ENGRAM=host
when absent.

    python3 kvram_budget.py [--engram host|nvme|hbm] [--margin-gib 24]  ->  prints the usable GiB (stdout), details on stderr
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import shutil
import sys

SHM = "/dev/shm"
ENGRAM_TABLE_BYTES = 104_454_946_816  # one packed table (~97.3 GiB); two layers (L1, L14)


class _Dqblk(ctypes.Structure):  # struct if_dqblk
    _fields_ = [("bhard", ctypes.c_uint64), ("bsoft", ctypes.c_uint64), ("cur", ctypes.c_uint64), ("ihard", ctypes.c_uint64),
                ("isoft", ctypes.c_uint64), ("curi", ctypes.c_uint64), ("btime", ctypes.c_uint64), ("itime", ctypes.c_uint64),
                ("valid", ctypes.c_uint32)]


def quota_remaining(path: str = SHM) -> int | None:
    """Bytes left in this user's block quota on `path` (None: no quota / not queryable)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        d = _Dqblk()
        fd = os.open(path, os.O_RDONLY)
        try:
            # quotactl_fd (x86_64 syscall 443), QCMD(Q_GETQUOTA=0x800007, USRQUOTA=0)
            r = libc.syscall(443, fd, ctypes.c_uint((0x800007 << 8) | 0), ctypes.c_uint(os.getuid()), ctypes.byref(d))
        finally:
            os.close(fd)
        if r != 0 or not (d.valid & 1) or d.bhard == 0:  # QIF_BLIMITS
            return None
        return max(0, d.bhard * 1024 - d.cur)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engram", default="host")
    ap.add_argument("--margin-gib", type=int, default=24)
    a = ap.parse_args()
    free = shutil.disk_usage(SHM).free
    q = quota_remaining()
    avail = min(free, q) if q is not None else free
    todo = 0
    if a.engram == "host":
        for layer in ("L1", "L14"):
            files = glob.glob(os.path.join(SHM, f"ds41h_engram_{layer}_packed_*.bin"))
            if files:
                st = os.stat(files[0])
                todo += max(0, st.st_size - st.st_blocks * 512)
            else:
                todo += ENGRAM_TABLE_BYTES
    cap = (avail - todo) // (1 << 30) - a.margin_gib
    print(f"kvram_budget: tmpfs free {free >> 30} GiB, user quota left {q >> 30 if q is not None else 'n/a'} GiB, "
          f"Engram to allocate {todo >> 30} GiB, margin {a.margin_gib} GiB -> usable {cap} GiB", file=sys.stderr)
    print(max(0, cap))


if __name__ == "__main__":
    main()
