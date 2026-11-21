"""mixed_cxx_moe_ensure: the moe piece's handles at ENSURE time and
its per-step variant rule as data (pieces_moe_decide.h FprMoeDecide).

Today mixed_cxx_moe._prepare builds a handle lazily at the first step that runs (bucket state s, layer L, rsf) and
step_params re-patches every handle's nv pointer when Engine._set_vidx stores a new s["nv"]. The native step needs
every pointer at ensure time and no Python in the step:

  ensure_moe(drv, st, layers=None, nv=None)
      st = one mixed view's bucket states over the ranks (Engine.bucket(R, "prod", role)), called by the engine's
      ensure-time hook after ensure_mixed. Builds, for every rank and layer of the view's role, the handles the
      step can need: plain (FUSED / FORK_*) and RSF, into the SAME cache _prepare reads (s["_cxx"][("moe", L, rsf)],
      s["_cxx_moe_h"]), so the embedded path finds them too and nothing is added lazily in-step. Idempotent. Handles live
      as long as the bucket state (mixed_cxx_moe._Handles frees them when the state is dropped: evicted / re-carved view =
      a new state = a new generation). nv: the view's persistent nv tensor
      None = s["nv"] if present, else a zeroed placeholder kept in the state (step_params patches it at the first step, as
      today). Returns export(drv, st).
  export(drv, st) -> [per rank dict(layer0, n, h [n][2], h_ptr, decide bytes, spec)]
      h[i] = (plain handle, RSF handle) of layer layer0 + i, -1 = none; h_ptr = an int64 host array kept alive in the state
      (for the exporter, D3); decide = the FprMoeDecide bytes.
  decide_spec(drv, r, s, layers) -> dict; pack_decide(dict) -> bytes; variant(spec, vcount, ndec, layer=None) = the header's
      fpr_moe_variant / fpr_moe_layer_variant in Python. CPU golden: moe_decide_test.py (decide() vs the header via ctypes).

The fixed parts of decide() are snapshotted at ensure: flags, cfg, agrs_ws's workspace rows, AGRS_PMNT and
ENG_MOE_CXX_FUSED (decide() reads those two env vars per call; nothing sets them after start-up).
"""
from __future__ import annotations

import ctypes as C
import os

import torch

import mixed_cxx_moe as M

DECIDE_VERSION = 1
NBITS = 128


class MoeDecide(C.Structure):          # pieces_moe_decide.h FprMoeDecide
    _fields_ = [("struct_size", C.c_int32), ("version", C.c_int32), ("on", C.c_int32), ("has_ecs", C.c_int32),
                ("switch_rows", C.c_int64), ("small_T", C.c_int64), ("ws_cap", C.c_int64),
                ("prod_variant", C.c_int32), ("act8_split", C.c_int32), ("fork_variant", C.c_int32),
                ("rsf_needs_agrs_st", C.c_int32), ("a8sh_layers", C.c_uint64 * 2)]


MAXDEV = 8
SNAP_VERSION = 1


class MoeSnapC(C.Structure):          # core/src/p0/moe_decide.h fpr::p0::MoeSnap (C3: the executor's moe_decide)
    _fields_ = [("version", C.c_int32), ("W", C.c_int32), ("prefill_moe_switch", C.c_int64), ("prod_ok", C.c_int32),
                ("fork_ok", C.c_int32), ("pm_fused", C.c_int32), ("pf_act8", C.c_int32), ("small_T", C.c_int64),
                ("nt", C.c_int32), ("ws", C.c_int32), ("ws_cap", C.c_int64 * MAXDEV), ("agrs_rsf", C.c_int32),
                ("cxx_fused", C.c_int32), ("min_r", C.c_int64), ("moe_v4", C.c_int32), ("v4_mixed", C.c_int32),
                ("dump_moe", C.c_int32), ("pad_", C.c_int32)]


class MoeBucketC(C.Structure):        # fpr::p0::MoeBucket
    _fields_ = [("prod", C.c_int32), ("has_ecs", C.c_int32), ("rs_lb128", C.c_int32), ("device", C.c_int32), ("R", C.c_int64),
                ("a8sh_layers", C.c_uint64 * 2)]


def snapshot(drv) -> MoeSnapC:
    """What mixed_cxx_moe.decide() reads from the engine / driver / flags (fixed after init; env AGRS_PMNT and
    ENG_MOE_CXX_FUSED taken now) -> p0::MoeSnap. bytes(snapshot(drv)) for the exporter."""
    E, EM = drv.eng, drv.EM
    m = MoeSnapC()
    m.version, m.W = SNAP_VERSION, len(E.devs)
    m.prefill_moe_switch = int(E.cfg.prefill_moe_switch)
    m.prod_ok, m.fork_ok = int(bool(getattr(drv, "_moe_prod_ok", False))), int(bool(getattr(drv, "_moe_fork_ok", False)))
    m.pm_fused, m.pf_act8 = int(bool(EM.PM_FUSED)), int(bool(getattr(EM, "PF_ACT8", False)))
    m.min_r = 1 << 62
    if m.prod_ok and m.pm_fused:                         # (decide reads these only on this path)
        import agrs_engine as AE
        m.small_T = int(E.prod_moe().SMALL_T)
        m.nt = int(str(getattr(EM, "PM_NT", "0")) != "0" or AE.nt_mode(EM) != "0")
        m.min_r = int(AE.MIN_R)
        fr = M._ws()
        m.ws = int(fr is not None)
        if fr is not None:
            for d, c in fr.cap.items():
                if d is not None and 0 <= int(d) < MAXDEV:
                    m.ws_cap[int(d)] = int(c)
        m.agrs_rsf = int(bool(EM.AGRS_RSF))
    m.cxx_fused = int(os.environ.get("ENG_MOE_CXX_FUSED", "0") == "1")
    m.moe_v4 = int(bool(E.cfg.moe_v4))
    m.v4_mixed = int(bool(getattr(EM, "V4_MIXED", False) or E.cfg.moe_v4_mixed))
    m.dump_moe = int(bool(getattr(EM, "DUMP_MOE", "")))
    return m


def bucket_of(s, lw=None, layers=None) -> MoeBucketC:
    """What decide() reads from the bucket state besides vcount / ndec -> p0::MoeBucket, + the view's A8sh layer mask:
    from lw (= E.lw[rank]) over layers (default: the view's role layers among lw's) when lw is given, else from ensure_moe's spec (s["_cxx_moe_export"]), else ALL ONES
    (unknown: the C++ refuses every fork variant; with no moe piece the variant is -1 anyway)."""
    b = MoeBucketC()
    b.prod, b.has_ecs, b.rs_lb128 = int(s.get("moe") == "prod"), int("ecs" in s), int("rs_lb128" in s)
    di = torch.device(s["dev"]).index if s.get("dev") is not None else None
    b.device = -1 if di is None else int(di)
    b.R = int(s.get("R", 0) or 0)
    if lw is not None:
        if layers is None:                               # default: the view's role layers among lw's (E.lw[r]: dict by layer)
            layers = _by_role(s, sorted(lw) if isinstance(lw, dict) else range(len(lw)))
        m = [0, 0]
        for L in layers:
            if lw[L].get("A8sh") is not None:
                assert 0 <= L < NBITS, L
                m[L >> 6] |= 1 << (L & 63)
    else:
        ex = s.get("_cxx_moe_export")
        m = list(ex["spec"]["a8sh"]) if ex else [(1 << 64) - 1, (1 << 64) - 1]
    b.a8sh_layers[0], b.a8sh_layers[1] = m[0], m[1]
    return b


def _by_role(s, Ls) -> list:
    """Ls filtered to the layers a mixed view runs: A = layers <= 20 (segA), T = layers > 20 (segT); unknown role = all."""
    bk = s.get("bkey")                                    # Engine.bucket / onebk view: (R, "prod", role)
    role = bk[2] if isinstance(bk, tuple) and len(bk) == 3 else s.get("role")
    if role is None:
        key = str(s.get("key", ""))                       # (the handover's canonical text key "704|prod|A")
        role = key.rsplit("|", 1)[-1] if "|" in key else None
    if role == "A":
        return [L for L in Ls if L <= 20]
    if role == "T":
        return [L for L in Ls if L > 20]
    return list(Ls)


def role_layers(E, s) -> list:
    return _by_role(s, E.layers)


def decide_spec(drv, r, s, layers) -> dict:
    """The fixed parts of mixed_cxx_moe.decide(drv, s) for this (view, rank) + the per-layer A8sh mask."""
    E, EM = drv.eng, drv.EM
    d = dict(on=int(s.get("moe") == "prod"), has_ecs=int("ecs" in s), switch_rows=int(E.cfg.prefill_moe_switch), small_T=0,
             ws_cap=0, prod_variant=-1, act8_split=0, fork_variant=-1, rsf_needs_agrs_st=1, a8sh=[0, 0])
    if not d["on"]:
        return d
    # prod branch (vcount > switch or no fork tables)
    if getattr(drv, "_moe_prod_ok", False) and EM.PM_FUSED:
        import agrs_engine as AE
        d["small_T"] = int(E.prod_moe().SMALL_T)
        d["act8_split"] = int(bool(getattr(EM, "PF_ACT8", False)))
        fr = M._ws()
        nt = str(getattr(EM, "PM_NT", "0")) != "0" or AE.nt_mode(EM) != "0"
        if fr is not None and not nt:
            d["ws_cap"] = int(fr.cap.get(torch.device(s["dev"]).index, 0))
            if EM.AGRS_RSF and s["R"] >= AE.MIN_R:
                d["prod_variant"] = M.PROD_RSF
            elif os.environ.get("ENG_MOE_CXX_FUSED", "0") == "1":
                d["prod_variant"] = M.PROD_FUSED
    # fork branch
    if getattr(drv, "_moe_fork_ok", False) and not getattr(EM, "DUMP_MOE", "") and "rs_lb128" not in s:
        Tall = len(E.devs) * s["R"]
        v4 = Tall <= 1365 and E.cfg.moe_v4 and (getattr(EM, "V4_MIXED", False) or E.cfg.moe_v4_mixed)
        d["fork_variant"] = M.FORK_V4 if v4 else M.FORK_RAW
    for L in layers:
        if E.lw[r][L].get("A8sh") is not None:
            assert 0 <= L < NBITS, L
            d["a8sh"][L >> 6] |= 1 << (L & 63)
    return d


def pack_decide(d) -> bytes:
    p = MoeDecide()
    p.struct_size, p.version = C.sizeof(MoeDecide), DECIDE_VERSION
    for k in ("on", "has_ecs", "switch_rows", "small_T", "ws_cap", "prod_variant", "act8_split", "fork_variant",
              "rsf_needs_agrs_st"):
        setattr(p, k, int(d[k]))
    p.a8sh_layers[0], p.a8sh_layers[1] = d["a8sh"][0], d["a8sh"][1]
    return bytes(p)


def variant(d, vcount, ndec=-1, layer=None) -> int:
    """pieces_moe_decide.h fpr_moe_variant (+ fpr_moe_layer_variant when layer is given), in Python."""
    if not d["on"]:
        v = -1
    elif vcount > d["switch_rows"] or not d["has_ecs"]:
        if d["prod_variant"] < 0 or vcount <= d["small_T"] or vcount > d["ws_cap"] or (d["act8_split"] and ndec >= 0):
            v = -1
        else:
            v = d["prod_variant"]
    else:
        v = d["fork_variant"]
    if layer is not None and v in (M.FORK_V4, M.FORK_RAW) and 0 <= layer < NBITS and (d["a8sh"][layer >> 6] >> (layer & 63)) & 1:
        v = -1
    return v


def _build(drv, r, s, L, st, rsf):
    """One handle into _prepare's cache (the same key / bookkeeping as mixed_cxx_moe._prepare's lazy build)."""
    cache = s.setdefault("_cxx", {})
    key = ("moe", L, bool(rsf))
    h = cache.get(key)
    if h is not None:
        return h
    own = s.get("_cxx_moe_h")
    if own is None:
        own = s["_cxx_moe_h"] = M._Handles(drv.x)
    h = cache[key] = drv.x.add_moe(M.build_piece_moe(drv, r, s, L, st if rsf else None))
    own.h.append(h)
    return h


def ensure_moe(drv, st, layers=None, nv=None):
    """Every moe handle of this mixed view (all ranks), built now. Returns export(drv, st)."""
    if not hasattr(drv, "_moe_n"):                       # mixed_cxx_moe.setup did not run (moe not among the C++ pieces)
        return None
    E = drv.eng
    def bound(r, s):                                     # the state's handles already point at the nv this call would use
        own = s.get("_cxx_moe_h")
        if own is None:
            return True
        want = nv[r] if nv is not None and nv[r] is not None else (s["_vidx_v"][1] if s.get("_vidx_v") is not None else None)
        return want is None or own.nvp == want.data_ptr()
    if layers is None and all("_cxx_moe_export" in s and bound(r, s) for r, s in enumerate(st)):
        return export(drv, st)                           # (every step's ensure: already built for this state = generation;
                                                         # a re-allocated ENG_MIX_PERSIST blob (vidx_gen) re-binds below)
    for r, s in enumerate(st):
        Ls = list(layers) if layers is not None else role_layers(E, s)
        d = decide_spec(drv, r, s, Ls)
        plain = d["on"] and (d["prod_variant"] == M.PROD_FUSED or d["fork_variant"] >= 0)
        rsf = d["on"] and d["prod_variant"] == M.PROD_RSF
        vv = s.get("_vidx_v")                            # ENG_MIX_PERSIST: the view's persistent (vidx, nv, nd) views
        nv_t = nv[r] if nv is not None and nv[r] is not None else (vv[1] if vv is not None else s.get("nv"))   # (None entry
        #                                                     = the default: FPR's hook passes None without ENG_MIX_PERSIST)
        if nv_t is None:                                 # no nv yet (the view's first step has not run): a placeholder,
            nv_t = s.get("_cxx_moe_nv0")                 # kept alive in the state; step_params patches it at the first step
            if nv_t is None:
                nv_t = s["_cxx_moe_nv0"] = torch.zeros(len(E.devs), dtype=torch.int32, device=s["dev"])
        had, old = "nv" in s, s.get("nv")
        s["nv"] = nv_t                                   # build_piece_moe reads s["nv"]; restored below
        try:
            hs = []
            for L in Ls:
                h0 = _build(drv, r, s, L, st, False) if plain else -1
                h1 = _build(drv, r, s, L, st, True) if rsf else -1
                hs.append((int(h0), int(h1)))
        finally:
            if had:
                s["nv"] = old
            else:
                del s["nv"]
        own = s.get("_cxx_moe_h")
        if own is not None and own.h:                    # one nv pointer for every handle of the state (earlier lazy ones too);
            p = nv_t.data_ptr()                          # step_params re-patches whenever s["nv"] moves (flag-off path)
            if own.nvp != p:
                for h in own.h:
                    drv.x.moe_set_nv(h, p)
            own.nvp = p
        ht = torch.tensor(hs if hs else [(-1, -1)], dtype=torch.int64)
        s["_cxx_moe_export"] = dict(layer0=Ls[0] if Ls else 0, n=len(Ls), h=hs, h_t=ht, h_ptr=ht.data_ptr(),
                                    decide=pack_decide(d), spec=d, bucket=bytes(bucket_of(s, E.lw[r], Ls)))
    return export(drv, st)


def export(drv, st):
    return [s.get("_cxx_moe_export") for s in st]


def states_of(E, s):
    """The mixed view's states over the ranks that s belongs to (Engine.bk_view[bkey][1] for an onebk view, Engine.bk[bkey]
    for a bucket), or None."""
    bk = s.get("bkey")
    if bk is None:
        return None
    v = (getattr(E, "bk_view", None) or {}).get(bk)
    if isinstance(v, tuple) and len(v) == 2:             # bk_view[key] = (base states, view states)
        v = v[1]
    for st in (v, (getattr(E, "bk", None) or {}).get(bk)):
        if isinstance(st, list) and any(x is s for x in st):
            return st
    return None


def ensure_from_step(drv, s):
    """ENG_MOE_ENSURE (mixed_cxx_moe.step_params, until FPR's ensure-time hook calls ensure_moe): the first set_step of a
    view builds every handle of the view (all ranks) before its span runs; a state outside bk / bk_view keeps the lazy path."""
    import time
    E = drv.eng
    st = states_of(E, s)
    if st is None:
        s["_cxx_moe_export"] = None                      # (not retried; _prepare builds lazily as before)
        E.log(f"mixed_cxx_moe: ENG_MOE_ENSURE: state {s.get('bkey')} R {s.get('R')} not found in bk / bk_view: lazy handles")
        return None
    t0 = time.perf_counter()
    n0 = drv.x.live_moe() if hasattr(drv.x, "live_moe") else -1
    ex = ensure_moe(drv, st)
    if ex and ex[0] is not None:
        n1 = drv.x.live_moe() if hasattr(drv.x, "live_moe") else -1
        nv_src = "persistent vidx_d" if st[0].get("_vidx_v") is not None else ("s['nv']" if st[0].get("nv") is not None else "placeholder")
        E.log(f"mixed_cxx_moe: ENG_MOE_ENSURE: view {st[0].get('bkey')} layers {ex[0]['layer0']}..{ex[0]['layer0'] + ex[0]['n'] - 1}: "
              f"{n1 - n0} handles in {(time.perf_counter() - t0) * 1e3:.1f} ms (live {n1}; nv {nv_src}; "
              f"lazy builds so far {getattr(drv, '_moe_lazy', 0)})")
    return ex
