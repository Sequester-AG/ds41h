"""mixed_cxx_moe: the Python side of the moe piece of the C++ mixed-step driver.
Loaded by mixed_cxx.Driver.__init__ (optional module): setup(drv) once, step_params(drv, r, s, sp) in every set_step,
piece_moe(drv, r, s, L) from the hook at the top of Engine.p_moe (ENG_MIXED_CXX_PIECES=...,moe).

Variant of a mixed ("prod") bucket in a step = the branch today's Python path takes (engine.p_moe / agrs_engine):
  prod path  (vcount > cfg.prefill_moe_switch or "ecs" not in s), PM_FUSED, vcount > SMALL_T:
    RSF      ENG_AGRS_RSF and R >= AGRS_MIN_R (and, at the call, agrs's bucket set _agrs_st holds s at rank r: its p_ag
             wrapper ran with this segment) -> agrs_pmws.layer_rs; sets s["agrs_rsf_pending"] as agrs_engine.p_moe_prod
             does, so the Python p_rs only rs_sum's
    FUSED    otherwise -> agrs_pmws.layer (part)
    both need ENG_AGRS_WS=2 (the piece and the Python path share agrs_ws's per-device workspace) and vcount <= its rows
  fork path  (vcount <= prefill_moe_switch and the bucket has the fork tables):
    FORK_V4  Tall <= 1,365, cfg.moe_v4 and (V4_MIXED or cfg.moe_v4_mixed): moe_v4r.route2r + moe_v4 GEMMs
    FORK_RAW otherwise: route2.route2r (Tall <= 1,365) or the ids / wts copies + route2.route2, moe_raw GEMMs
    both: the shared expert slice on the side stream (ATen mm + moe_raw.swiglu), moe_raw.prepr5, moe_raw.combine
  -1 (the Python p_moe): unfused prod (vcount <= SMALL_T), PF_ACT8's decode-split arm, the narrow-tile kernels (PM_NT /
  AGRS_PMNT != 0), a layer with the 8-bit shared slice (lw["A8sh"]: ENG_C128_DENSE8_ADD sh), ENG_DUMP_MOE, an
  "rs_lb128" bucket, a missing table, a capturing stream (the hook's check).
"""
from __future__ import annotations

import ctypes as C
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kernels")
for _d in ("agrs", "moe_v4"):
    if os.path.join(KROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(KROOT, _d))
import agrs_pmws_cabi as APC  # noqa: E402
import v4_cabi as V4C  # noqa: E402

VERSION = 2
FORK_V4, FORK_RAW, PROD_RSF, PROD_FUSED = 0, 1, 2, 3     # mixed_plan.h FprMoeVariant
NAMES = {FORK_V4: "fork_v4", FORK_RAW: "fork_raw", PROD_RSF: "rsf", PROD_FUSED: "fused"}
D, TOPK = 5120, 6
P = C.c_void_p
FprCall = APC.FprCall


# ---- ctypes mirrors of FPR's moe_stream headers; sizes checked in setup
class RawSwiglu(C.Structure):
    _fields_ = [("c", FprCall), ("gu", P), ("h", P), ("T", C.c_int32), ("I", C.c_int32)]


class RawPrepr5(C.Structure):
    _fields_ = [("c", FprCall), ("x", P), ("ecs", P), ("counts", P), ("img", P), ("rows", P), ("sms", C.c_int32), ("ctr", P)]


class RawGemm1r(C.Structure):
    _fields_ = [("c", FprCall), ("w13", P), ("s13", P), ("E", C.c_int64), ("ecs", P), ("tiles", P), ("counts", P), ("img", P),
                ("himg", P), ("grid", C.c_int32)]


class RawGemm2r(C.Structure):
    _fields_ = [("c", FprCall), ("w2", P), ("s2", P), ("E", C.c_int64), ("ecs", P), ("tiles", P), ("counts", P), ("himg", P),
                ("yout", P), ("grid", C.c_int32)]


class RawCombine(C.Structure):
    _fields_ = [("c", FprCall), ("Y", P), ("slotmap", P), ("sh", P), ("out", P), ("T", C.c_int32)]


class Route2r(C.Structure):
    _fields_ = [("c", FprCall), ("rows", P), ("T", C.c_int64), ("ecs", P), ("t1", P), ("t2", P), ("counts", P), ("slotmap", P),
                ("grid", C.c_int32)]


class Route2(C.Structure):
    _fields_ = [("c", FprCall), ("ids", P), ("w", P), ("T", C.c_int64), ("ecs", P), ("t1", P), ("t2", P), ("counts", P),
                ("slotmap", P), ("rts", P), ("rts_bytes", C.c_int64), ("grid", C.c_int32), ("inv", P), ("ec8", P)]


class PieceMoe(C.Structure):
    _fields_ = [("version", C.c_int32), ("layer", C.c_int32), ("R", C.c_int32), ("rank", C.c_int32),
                ("rsf_ok", C.c_int32), ("fused_ok", C.c_int32), ("fork_ok", C.c_int32), ("fork_route", C.c_int32),
                ("fork_gemm", C.c_int32), ("device", C.c_int32),
                ("main", P), ("side", P), ("ev0", P), ("ev1", P),
                ("prod_rs", APC.AgrsPmwsLayerArgs), ("prod", APC.AgrsPmwsLayerArgs),
                ("fall", P), ("sw13", P), ("sw2", P), ("gu", P), ("hsh", P), ("sh", P),
                ("Tall", C.c_int64), ("H", C.c_int64), ("Is", C.c_int64),
                ("swiglu", RawSwiglu),
                ("v4_route", V4C.MoeV4Route2rArgs), ("r2r", Route2r), ("r2", Route2), ("gr", P), ("ids_buf", P), ("wts_buf", P),
                ("prepr5", RawPrepr5), ("v4_g1", V4C.MoeV4Gemm1rArgs), ("v4_g2", V4C.MoeV4Gemm2rArgs),
                ("raw_g1", RawGemm1r), ("raw_g2", RawGemm2r), ("combine", RawCombine)]


PARTS = (PieceMoe, APC.AgrsPmwsLayerArgs, V4C.MoeV4Route2rArgs, V4C.MoeV4Gemm1rArgs, V4C.MoeV4Gemm2rArgs, Route2r, Route2,
         RawSwiglu, RawPrepr5, RawGemm1r, RawGemm2r, RawCombine)   # = pieces_moe.cpp sizeof_moe_parts()


def _ws():
    import agrs_ws
    st = agrs_ws._state
    return st["fr"] if st.get("mode") == "2" and st.get("fr") is not None else None


def setup(drv) -> None:
    """Once per driver: register the tables, the per-rank events + each table's init (device current)."""
    E, x = drv.eng, drv.x
    drv._moe_prod_ok = drv._moe_fork_ok = False
    drv._moe_n = {k: 0 for k in list(NAMES.values()) + ["python"]}
    sizes = [C.sizeof(t) for t in PARTS]
    assert list(x.sizeof_moe_parts()) == sizes, (list(x.sizeof_moe_parts()), sizes)
    fr = _ws()
    if fr is not None:
        APC.check()
        x.register_table(APC.lib_path(), "agrs_pmws")
    try:   # the fork's tables: moe_v4 / moe_v4r, moe_raw / route2 (the engine's own modules, FPR's ABI)
        V4C.check()
        la, lr = V4C.lib_paths()
        x.register_table(la, "moe_v4")
        x.register_table(lr, "moe_v4r")
        x.register_table(E.mext[0].__file__, "moe_raw")
        x.register_table(E.r2.__file__, "route2")
        fork_tables = True
    except Exception as ex:  # noqa: BLE001
        E.log(f"mixed_cxx_moe: fork tables not registered ({type(ex).__name__}: {ex}): fork variants stay Python")
        fork_tables = False
    prod, fork = [], []
    drv._moe_scratch = []
    for r, d in enumerate(E.devs):
        with torch.cuda.device(d):
            di = torch.device(d).index
            prod.append(bool(x.moe_init_rank(r, di)))
            fork.append(bool(fork_tables and x.moe_init_fork_rank(r, di)))
            sc = torch.zeros(64 * 1024, dtype=torch.uint8, device=d)     # the 1d thread warm-up's mm operands (kept alive)
            drv._moe_scratch.append(sc)
            x.moe_set_thread_init(r, E.streams[r].cuda_stream, E.side[r].cuda_stream, sc.data_ptr(), sc.numel())
    nv_ok = hasattr(x, "moe_set_nv")   # a driver build without it would freeze s["nv"] at add_moe: prod stays Python
    drv._moe_prod_ok, drv._moe_fork_ok = fr is not None and all(prod) and nv_ok, all(fork)
    if fr is not None and not nv_ok:
        E.log("mixed_cxx_moe: this driver build has no moe_set_nv (an older pieces_moe.cpp): prod variants stay Python")
    E.log(f"mixed_cxx_moe: moe piece up: prod RSF / fused {'ON (agrs_pmws_c, ENG_AGRS_WS=2)' if drv._moe_prod_ok else 'off (needs ENG_AGRS_WS=2)'}, "
          f"fork v4 / raw {'ON' if drv._moe_fork_ok else 'off'} (FprPieceMoe {C.sizeof(PieceMoe)} B)")


def decide(drv, s) -> int:
    E, EM = drv.eng, drv.EM
    if s.get("moe") != "prod":
        return -1
    vcount = s.get("vcount", 1 << 30)
    if vcount > E.cfg.prefill_moe_switch or "ecs" not in s:            # Engine.p_moe -> p_moe_prod
        if not getattr(drv, "_moe_prod_ok", False) or not EM.PM_FUSED or vcount <= E.prod_moe().SMALL_T:
            return -1                                                   # unfused prod: Python
        if getattr(EM, "PF_ACT8", False) and s.get("ndec", -1) >= 0:
            return -1                                                   # ENG_PF_ACT8's layer8: Python
        import agrs_engine as AE
        if str(getattr(EM, "PM_NT", "0")) != "0" or AE.nt_mode(EM) != "0":
            return -1                                                   # narrow-tile kernels: Python
        fr = _ws()
        if fr is None or vcount > fr.cap.get(torch.device(s["dev"]).index, 0):
            return -1                                                   # above the workspace's rows: the Python fallback
        if EM.AGRS_RSF and s["R"] >= AE.MIN_R:
            return PROD_RSF
        # PROD_FUSED in C++ diverges late: Python until root-caused; ENG_MOE_CXX_FUSED=1 re-enables it for the debug arms
        return PROD_FUSED if os.environ.get("ENG_MOE_CXX_FUSED", "0") == "1" else -1
    if not getattr(drv, "_moe_fork_ok", False) or getattr(EM, "DUMP_MOE", "") or "rs_lb128" in s:
        return -1
    Tall = len(E.devs) * s["R"]
    if Tall <= 1365 and E.cfg.moe_v4 and (getattr(EM, "V4_MIXED", False) or E.cfg.moe_v4_mixed):
        return FORK_V4
    return FORK_RAW


def step_params(drv, r, s, sp) -> None:
    """Driver.set_step hook: this rank's moe fields of the segment's StepParams."""
    v = decide(drv, s)
    s["_cxx_moe_variant"] = v
    sp.moe_variant = v
    sp.moe_vcount = int(s.get("vcount", 0) or 0)
    sp.moe_n_wait = 0                                        # 1b-i: agrs's p_moe wrapper waits the AGCE events in Python
    own = s.get("_cxx_moe_h")
    if own is not None and own.h and s.get("nv") is not None:
        p = s["nv"].data_ptr()                               # _set_vidx stores a new nv tensor every mixed step
        if own.nvp != p:                                     # patch every handle's prod Args (between spans) before the pieces run
            for h in own.h:
                drv.x.moe_set_nv(h, p)
            own.nvp = p


def _p(t):
    return t.data_ptr() if t is not None else None


def build_piece_moe(drv, r, s, L, st=None) -> bytes:
    """FprPieceMoe for (bucket state s, layer L, rank r): the prod Args (fused; RSF too when st = the segment's 4 states) and
    the fork Args (when the bucket has the fork tables). Pointers only; every buffer is the bucket's / the layer's."""
    E, EM = drv.eng, drv.EM
    dev = torch.device(s["dev"])
    di = dev.index
    lw = E.lw[r][L]
    R, W = s["R"], len(E.devs)
    Tall = W * R
    fall = s["gf"].view(torch.bfloat16).view(Tall, D)
    main, side = E.streams[r], E.side[r]
    p = PieceMoe()
    p.version, p.layer, p.R, p.rank, p.device = VERSION, L, R, r, di
    p.main, p.side = main.cuda_stream, side.cuda_stream
    fr = _ws()
    if getattr(drv, "_moe_prod_ok", False) and fr is not None:
        ws = fr.ws[di]
        common = (fall, s["gr"], s["nv"], R, 0, lw["w13"], lw["s13"], lw["w2"], lw["s2"], lw["sw13"], lw["sw2"], s["gu"],
                  s["hsh"], s["sh"], ws)
        p.prod = APC.AgrsPmwsLayerArgs.from_buffer_copy(APC.pack_layer(*common, part=s["part"], variant=fr.variant, stream=main))
        p.fused_ok = 1
        if st is not None:
            import agrs_engine as AE
            AE._ensure(E, st, "rsf")                         # the RSF flags / epoch words (first use: zeroed + device sync)
            p.prod_rs = APC.AgrsPmwsLayerArgs.from_buffer_copy(APC.pack_layer(
                *common, bufs=[x["rs"] for x in st], flags=[x["agrs_fl_rsf"] for x in st], epoch=s["agrs_ep_rsf"], rank=r,
                grid=AE.RSF_GRID, variant=fr.variant, stream=main))
            p.rsf_ok = 1
    if getattr(drv, "_moe_fork_ok", False) and "ecs" in s:
        cm, cs = FprCall(main.cuda_stream, di, r), FprCall(side.cuda_stream, di, r)
        Is = lw["sw2"].shape[1]
        p.fall, p.sw13, p.sw2, p.gu, p.hsh, p.sh = (_p(fall), _p(lw["sw13"]), _p(lw["sw2"]), _p(s["gu"]), _p(s["hsh"]),
                                                    _p(s["sh"]))
        p.Tall, p.H, p.Is = Tall, D, Is
        assert s["gu"].shape[0] == Tall and s["sh"].shape[0] == Tall and lw["sw13"].is_contiguous() and lw["sw2"].is_contiguous()
        p.swiglu = RawSwiglu(cs, _p(s["gu"]), _p(s["hsh"]), Tall, Is)
        v4 = Tall <= 1365 and E.cfg.moe_v4 and (getattr(EM, "V4_MIXED", False) or E.cfg.moe_v4_mixed)
        sms = int(s["sms"])
        if v4:
            p.fork_route = 0
            p.v4_route = V4C.MoeV4Route2rArgs.from_buffer_copy(V4C.pack_route2r(
                s["gr"], Tall, s["ecs"], s["t1"], s["t2"], s["counts"], s["slotmap"], grid=sms, pdl=3, one_max_t=1365, stream=main))
        elif Tall <= 1365:
            p.fork_route = 1
            p.r2r = Route2r(cm, _p(s["gr"]), Tall, _p(s["ecs"]), _p(s["t1"]), _p(s["t2"]), _p(s["counts"]), _p(s["slotmap"]), sms)
        else:
            p.fork_route = 2
            idw = s.get("_cxx_moe_idw")
            if idw is None or idw.shape[1] != Tall:
                idw = s["_cxx_moe_idw"] = torch.empty(2, Tall, 4 * TOPK, dtype=torch.uint8, device=dev)
            p.gr, p.ids_buf, p.wts_buf = _p(s["gr"]), _p(idw[0]), _p(idw[1])
            assert s["gr"].shape[0] == Tall and s["gr"].shape[1] == 64
            p.r2 = Route2(cm, _p(idw[0]), _p(idw[1]), Tall, _p(s["ecs"]), _p(s["t1"]), _p(s["t2"]), _p(s["counts"]),
                          _p(s["slotmap"]), _p(s["rts"]), s["rts"].nbytes, sms, None, None)
        p.prepr5 = RawPrepr5(cm, _p(fall), _p(s["ecs"]), _p(s["counts"]), _p(s["img"]), _p(s["gr"]), sms, _p(s["g5ctr"]))
        if v4:
            p.fork_gemm = 0
            p.v4_g1 = V4C.MoeV4Gemm1rArgs.from_buffer_copy(V4C.pack_gemm1r(
                lw["w13"], lw["s13"], s["ecs"], s["t1"], s["counts"], s["img"], s["himg"], 0, stream=main))
            p.v4_g2 = V4C.MoeV4Gemm2rArgs.from_buffer_copy(V4C.pack_gemm2r(
                lw["w2"], lw["s2"], s["ecs"], s["t2"], s["counts"], s["himg"], s["Y"], 0, stream=main))
        else:
            p.fork_gemm = 1
            p.raw_g1 = RawGemm1r(cm, _p(lw["w13"]), _p(lw["s13"]), lw["w13"].shape[0], _p(s["ecs"]), _p(s["t1"]), _p(s["counts"]),
                                 _p(s["img"]), _p(s["himg"]), 0)
            p.raw_g2 = RawGemm2r(cm, _p(lw["w2"]), _p(lw["s2"]), lw["w2"].shape[0], _p(s["ecs"]), _p(s["t2"]), _p(s["counts"]),
                                 _p(s["himg"]), _p(s["Y"]), 0)
        p.combine = RawCombine(cm, _p(s["Y"]), _p(s["slotmap"]), _p(s["sh"]), _p(s["part"]), s["part"].shape[0])
        p.fork_ok = 1
    return bytes(p)


class _Handles:
    """The bucket state's moe handles (stored in the state dict): freed in the driver when the state is dropped (an
    evicted / re-created bucket), so handles do not accumulate over a run."""

    def __init__(self, x):
        self.x, self.h, self.nvp = x, [], None   # nvp: the s["nv"] pointer the handles' prod Args hold

    def __del__(self):
        # a GC run on another Python thread can fire this mid-span, while the driver threads
        # read the slot table: frees only QUEUE (defer_free); drain_frees() applies them on the engine thread between spans
        try:
            try:
                from mixed_cxx import defer_free
            except Exception:  # noqa: BLE001  (interpreter shutdown: the module may be gone; free directly)
                defer_free = None
            for h in self.h:
                if defer_free is not None:
                    defer_free(self.x.free_moe, h)
                else:
                    self.x.free_moe(h)
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass


def _prepare(drv, r, s, L):
    """(variant, handle) for this call, or None = the Python p_moe. The checks of today's Python path that depend on the
    call (the layer's 8-bit shared slice; RSF: agrs's bucket set holds s), the cached static struct, and the bookkeeping
    the Python path does (workspace stats, agrs's rsf counter)."""
    v = s.get("_cxx_moe_variant", -1)
    if v < 0:
        return None
    E = drv.eng
    if v in (FORK_V4, FORK_RAW) and E.lw[r][L].get("A8sh") is not None:
        drv._moe_n["python"] += 1
        return None                                          # the 8-bit shared slice (DENSE8_ADD sh): Python
    st = getattr(E, "_agrs_st", None) if v == PROD_RSF else None
    if v == PROD_RSF and not (st is not None and st[r] is s):
        drv._moe_n["python"] += 1
        return None                                          # agrs's p_moe_prod would not take RSF either: Python
    cache = s.setdefault("_cxx", {})
    key = ("moe", L, v == PROD_RSF)
    h = cache.get(key)
    if h is None:
        own = s.get("_cxx_moe_h")
        if own is None:
            own = s["_cxx_moe_h"] = _Handles(drv.x)
        h = cache[key] = drv.x.add_moe(build_piece_moe(drv, r, s, L, st))
        if own.h and own.nvp is not None and s.get("nv") is not None and own.nvp != s["nv"].data_ptr():
            for h0 in own.h:                                 # (the step_params patch normally ran; this keeps them consistent)
                drv.x.moe_set_nv(h0, s["nv"].data_ptr())
        own.h.append(h)
        own.nvp = s["nv"].data_ptr() if s.get("nv") is not None else None
    if v in (PROD_RSF, PROD_FUSED):
        fr = _ws()
        fr.st["ws_calls"] += 1
        fr.st["max_T"] = max(fr.st["max_T"], int(s.get("vcount", 0) or 0))
    if v == PROD_RSF:
        import agrs_engine as AE
        AE.stats["rsf"] += 1
    nm = NAMES[v]
    drv._moe_n[nm] += 1
    if drv._moe_n[nm] == 1:                                  # once per variant: proof the piece ran (harness greps mixed_cxx)
        E.log(f"mixed_cxx_moe: first C++ moe piece, variant {nm}: rank {r} layer {L} R {s['R']} vcount {s.get('vcount')}")
    return v, h


def piece_moe(drv, r, s, L) -> bool:
    """1b-i: the hook's call (top of Engine.p_moe, on the rank stream): True = issued in C++, False = the Python p_moe."""
    ph = _prepare(drv, r, s, L)
    if ph is None:
        return False
    v, h = ph
    rc = drv.x.moe(r, h, L)
    assert rc == 0, f"mixed_cxx_moe: moe rank {r} layer {L} variant {NAMES[v]}: FPR error {rc}"
    if v == PROD_RSF:
        s["agrs_rsf_pending"] = True                         # as agrs_engine.p_moe_prod: the Python p_rs then only rs_sum's
    if v == PROD_FUSED and os.environ.get("ENG_MOE_CXX_CHECK", "0") == "1":
        _check_fused(drv, r, s, L)
    return True


def _check_fused(drv, r, s, L) -> None:
    """after the C++ fused call: the Python layer (pmf().layer
    mode 2) into scratch copies of part / gu / hsh / sh on the same stream, compared bitwise; prints the first differing rows."""
    E = drv.eng
    lw = E.lw[r][L]
    R, W = s["R"], len(E.devs)
    Tall, T = W * R, int(s["vcount"])
    fall = s["gf"].view(torch.bfloat16).view(Tall, D)
    part2, gu2, hsh2, sh2 = (torch.empty_like(s[k]) for k in ("part", "gu", "hsh", "sh"))
    E.pmf().layer(fall, s["gr"], s["nv"], R, T, lw["w13"], lw["s13"], lw["w2"], lw["s2"], lw["sw13"], lw["sw2"], gu2, hsh2, sh2,
                  part2, padded=False)
    res = []
    for nm, a, b, n in (("part", s["part"], part2, Tall), ("gu", s["gu"], gu2, T), ("hsh", s["hsh"], hsh2, T), ("sh", s["sh"], sh2, T)):
        x, y = a[:n].view(torch.int16), b[:n].view(torch.int16)
        bad = (x != y).any(1).nonzero().flatten()
        res.append(f"{nm} {'OK' if bad.numel() == 0 else f'DIFF {bad.numel()} rows, first {bad[:6].tolist()}'}")
    E.log(f"mixed_cxx_moe CHECK fused rank {r} layer {L} R {R} vcount {T} nv {s['nv'].tolist()}: " + "; ".join(res))


def handle_moe(drv, r, s, L) -> int:
    """1d: the span builder's call: this rank's moe handle (the driver thread runs moe_c with it), or -1 = the Python p_moe.
    The span builder reads s["_cxx_moe_variant"] == PROD_RSF for its rs entry (rsf_pending is not set here)."""
    ph = _prepare(drv, r, s, L)
    return -1 if ph is None else ph[1]
