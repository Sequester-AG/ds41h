"""mixed_cxx_c1mega: the Python side of pieces_c1mega.cpp.

Kind (ENG_MIXED_CXX_PIECES): "c1mega" = Engine._issue's _mega_on branch, C1Mega.run(r, s, L, st), for the run of layers that
starts at L: _join_mix, the Engram waits, ONE k_layer launch through the package's C ABI, the
owner's DSpark aux copies. Loaded by mixed_cxx.Driver: setup(drv) once (inert unless "c1mega" is in ENG_MIXED_CXX_PIECES),
piece_c1mega(drv, r, s, L) from the Engine hook (1b-i), handle_c1mega(drv, r, s, L) from the span builder (1d).
-1 / False (= Python) for: a member layer that does not start a run (nothing to launch: the persistent kernel runs it), a
package without the C ABI, C1IDX_FOLD_BT=0, an Engram event not recorded yet, a Python boundary's
pending side-stream mix (s["mix_ev"]: Engine._join_mix's Python half, not in the handle; never at a run start with
C1M_IX=1), no bucket states (E._tp_st) for the call. The handle bytes are built by the package's adapter
(C1Mega.piece_bytes), from the same packs / buffers / events its run() uses, cached per bucket state (s["_cxx"]; a
refusal is NOT cached: re-checked per call) and freed in the driver when the state is dropped (s["_cxx_c1m_h"]).
Counts (drv._c1m_n): c1mega = launches through the C++ piece, python = run starts that fell back (py_<reason> each),
inner = member layers inside a run (no launch on either path).
"""
from __future__ import annotations

import collections
import ctypes as C

import mixed_cxx as M


def setup(drv) -> None:
    """Once per driver: inert unless "c1mega" is in ENG_MIXED_CXX_PIECES. The adapter is resolved per call (Engine.c1m()
    can change: ENG_C1_MEGA_AB arms, the in-situ harness's arms): a package with the C ABI registers its table at its
    first call; one C ABI package per process (the C++ resolves "c1mega.run" once)."""
    drv._c1m_on = "c1mega" in M.PIECES
    drv._c1m_ok = {}                       # id(adapter) -> bool (portable: has piece_bytes + exports fpr_c1mega_table)
    drv._c1m_so = None                     # the registered package .so
    drv._c1m_n = collections.Counter(c1mega=0, python=0)


class _Handles:
    """A bucket state's c1mega handles (s["_cxx_c1m_h"]): freed in the driver when the state is dropped (an evicted /
    re-created bucket, reset_state), as mixed_cxx_moe's (mixed_cxx._Cache frees only the FPR / Engram kinds)."""

    def __init__(self, x):
        self.x, self.h = x, []

    def __del__(self):
        try:
            for h in self.h:
                self.x.free_c1mega(h)
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass


def _adapter_ok(drv, cm) -> bool:
    ok = drv._c1m_ok.get(id(cm))
    if ok is not None:
        return ok
    E, ok = drv.eng, False
    so = getattr(getattr(cm, "X", None), "__file__", None)
    if not hasattr(cm, "piece_bytes") or so is None:
        E.log(f"mixed_cxx_c1mega: adapter {getattr(cm, 'HERE', '?')} has no C ABI builder (srv7c+): its runs stay Python")
    else:
        try:
            C.CDLL(so).fpr_c1mega_table
            has = True
        except AttributeError:
            has = False
        if not has:
            E.log(f"mixed_cxx_c1mega: {so} exports no fpr_c1mega_table: its runs stay Python")
        elif drv._c1m_so is not None and drv._c1m_so != so:
            E.log(f"mixed_cxx_c1mega: a second C ABI package ({so}; registered {drv._c1m_so}): its runs stay Python")
        else:
            assert drv.x.sizeof_c1mega() == cm.piece_sizeof(), (drv.x.sizeof_c1mega(), cm.piece_sizeof())
            if drv._c1m_so is None:
                drv.x.register_table(so, "c1mega")
                drv._c1m_so = so
                E.log(f"mixed_cxx_c1mega: c1mega piece up ({so}; FprPieceC1mega {cm.piece_sizeof()} B)")
            ok = True
    drv._c1m_ok[id(cm)] = ok
    return ok


def _handle(drv, r, s, L):
    """(handle, adapter, reason): handle >= 0 = the C++ piece runs this call; else reason = why the Python path runs
    ("inner" = not a run start: nothing to launch on either path)."""
    E = drv.eng
    if not E._mega_on(s):
        return -1, None, "py_not_mega"
    cm = E.c1m()
    k = cm.first.get(L)
    if k is None:
        return -1, cm, "inner"
    if not _adapter_ok(drv, cm):
        return -1, cm, "py_no_cabi"
    st = getattr(E, "_tp_st", None)
    if st is None or st[r] is not s:
        return -1, cm, "py_no_states"
    if s.get("mix_ev") is not None:
        return -1, cm, "py_mix_ev"
    cache = M._cache(drv, s)
    key = ("c1mega", id(cm), L, cm.pack_key(s, st, k))
    h = cache.get(key)
    if h is None:
        b = cm.piece_bytes(r, s, L, st)
        if b is None:
            return -1, cm, "py_no_bytes"         # (not cached: an Engram event not recorded yet is transient)
        own = s.get("_cxx_c1m_h")
        if own is None:
            own = s["_cxx_c1m_h"] = _Handles(drv.x)
        h = cache[key] = drv.x.add_c1mega(b)
        own.h.append(h)
    return h, cm, None


def handle_c1mega(drv, r, s, L) -> int:
    if not getattr(drv, "_c1m_on", False):
        return -1
    return _handle(drv, r, s, L)[0]


def piece_c1mega(drv, r, s, L) -> bool:
    if not getattr(drv, "_c1m_on", False):
        return False
    h, cm, why = _handle(drv, r, s, L)
    if h < 0:
        drv._c1m_n[why] += 1
        if why != "inner":
            drv._c1m_n["python"] += 1
        return False
    rc = drv.x.c1mega(r, h, L)
    assert rc == 0, f"mixed_cxx_c1mega: c1mega rank {r} layer {L}: FPR error {rc}"
    cm.stats["launches"] += 1                    # (= C1Mega.run's count: one k_layer launch)
    drv._c1m_n["c1mega"] += 1
    drv.n["c1mega"] += 1
    return True
