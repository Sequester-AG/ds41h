#!/usr/bin/env python3
"""d13tail: D13's graph T' (the steady C1 owner's step prep) as ONE kernel.

Graph T' (d13.py _gto_body -> _gt_body -> hooks_idx.device_tables_d13) is ~45 tiny torch kernels (~67 us per C1 step on
the owner, between graph H and its graph L: the critical path, the peers wait inside c1mega meanwhile). tail() runs the
same operations in the same order on the same buffers in one CTA (d13tail.cu). Integer math only: bitwise by construction;

    import d13tail as DT
    DT.run(**DT.owner_args(d13, r, s, R, n))      # = d13._gto_body(r, s, R, n)
"""
from __future__ import annotations

import os
import sys
from functools import cache

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SW = 128
_ROWEXP = os.path.join(os.path.dirname(HERE), "rowexp")


@cache
def ext():
    from torch.utils.cpp_extension import load
    bd = os.path.join(HERE, "build_ext")
    os.makedirs(bd, exist_ok=True)
    return load(name="d13tail_ext", sources=[os.path.join(HERE, "d13tail.cu")], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])


def _rx_ext():
    if _ROWEXP not in sys.path:
        sys.path.insert(0, _ROWEXP)
    import rowexp
    return rowexp.ext()


KEYS = ("rslot", "rbase", "b_start", "sbase", "tok", "dt", "n_tok", "next_ids", "slot", "row0", "nrows", "kind", "start",
        "replay", "cached", "counts", "ring", "ops", "dbt", "nc", "R", "scratch", "pos", "own", "si", "sl", "paged",
        "start_o", "seq", "err", "has_seqs", "nb", "valid", "sl1", "sl2", "cs1", "cs2")


def run(**a) -> None:
    """The fused kernel on the current stream (all tensors on one device; see KEYS)."""
    ext().tail(a["rslot"], a["rbase"], a["b_start"], a["sbase"], a["tok"], a["dt"], int(a["n_tok"]), a["next_ids"],
               a["slot"], a["row0"], a["nrows"], a["start"], a["replay"], a["cached"], a["counts"], a["ring"], a["ops"],
               a["dbt"], a["nc"], int(a["R"]), int(a["scratch"]), a["pos"], a["own"], a["si"], a["sl"], a["paged"],
               a["start_o"], a["seq"], a["err"], bool(a["has_seqs"]), a["nb"], a["valid"], a["sl1"], a["sl2"], a["cs1"],
               a["cs2"])


def ref(**a) -> None:
    """Graph T' as the engine runs it: _gto_body's resets + device tokens, rowexp begin (ring_ops + expand), _gt_body's
    guard ops, hooks_idx.device_tables_d13 (the exact torch statements)."""
    R, n = int(a["R"]), int(a["n_tok"])
    a["rslot"].copy_(a["rbase"])
    a["b_start"].copy_(a["sbase"])
    dt = a["dt"]
    a["tok"].index_copy_(0, dt[0, :n], a["next_ids"].index_select(0, dt[1, :n]))
    E = _rx_ext()
    E.ring_ops(a["ring"], a["ops"], a["counts"])
    E.expand(a["slot"], a["row0"], a["nrows"], a["kind"], a["start"], a["replay"], a["cached"], a["counts"], a["ring"],
             a["dbt"], a["nc"], R, int(a["scratch"]), a["pos"], a["own"], a["si"], a["sl"], a["paged"], a["start_o"],
             a["seq"], a["err"])
    own = a["own"]
    own.masked_fill_(own < 0, int(a["scratch"]))
    if a["has_seqs"]:
        seq = a["seq"][:R]
        lim = a["nb"].index_select(0, seq.clamp(min=0).long())
        a["paged"].masked_fill_((seq >= 0) & (a["pos"] // 64 >= lim), -1)
    pos, v = a["pos"][:R], a["valid"]
    p1 = pos + 1
    a["sl1"].copy_(torch.where(v, p1, 0))
    a["sl2"].copy_(torch.where(v, p1 // 2, 0))
    paged = a["paged"]
    ok = v & (paged >= 0)
    a["cs1"].copy_(torch.where(ok, paged, -1))
    a["cs2"].copy_(torch.where(ok, (paged // 64) * 32 + (paged % 64) // 2, -1))
    a["rslot"].copy_(torch.where(v, a["rslot"] + pos % 8, -1))


def owner_args(d13, r: int, s: dict, R: int, n: int) -> dict:
    """The buffers d13._gto_body(r, s, R, n) operates on (graph T' of a steady owner rank)."""
    H = d13.eng.hooks
    d = s["idx"]
    rx = d13.rx[r]
    o = rx.out
    return dict(rslot=d["rslot"], rbase=s["_own_rbase"], b_start=H.engram[r].b_start, sbase=s["_own_sbase"], tok=s["tok"],
                dt=H._dt_d[r], n_tok=n, next_ids=H.hsd.next_ids[r].view(-1), slot=rx.d["slot"], row0=rx.d["row0"],
                nrows=rx.d["nrows"], kind=rx.d["kind"], start=rx.d["start"], replay=rx.d["replay"], cached=rx.d["cached"],
                counts=rx.d["counts"], ring=rx.ring, ops=rx.d["ops"], dbt=rx.dbt, nc=rx.nc, R=R, scratch=rx.scratch,
                pos=o["pos"], own=o["own"], si=o["si"], sl=o["sl"], paged=o["paged"], start_o=o["start"], seq=o["seq"],
                err=rx.err, has_seqs=True, nb=d13.nb_d[r], valid=d["valid"], sl1=d["seq_lens"][1].view(-1),
                sl2=d["seq_lens"][2].view(-1), cs1=d["cslot"][1], cs2=d["cslot"][2])


def run_owner(d13, r: int, s: dict, R: int, n: int) -> None:
    run(**owner_args(d13, r, s, R, n))


OUT_KEYS = ("rslot", "b_start", "tok", "ring", "pos", "own", "si", "sl", "paged", "start_o", "seq", "err", "sl1", "sl2",
            "cs1", "cs2")
