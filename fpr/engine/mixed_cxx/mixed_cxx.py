"""Stage 1b-i of the C++ mixed-step driver: Python side of driver.cpp.

the driver keeps them (add_<kind> -> handle). bd / moe hook inside the Engine
methods; router / ag / rs / final (and the AGCE join before p_moe) wrap the served method chain from outside (install(),
called by engine.py after the agrs wrappers), so the C++ piece replaces exactly what the Python chain issues; anything
not covered (raw buckets, capture, image routing, AG32, RSD, ENG_C1_L2PF) runs the Python chain unchanged.
"""
import ctypes as C
import functools
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P = C.c_void_p
PIECES = frozenset(x for x in os.environ.get("ENG_MIXED_CXX_PIECES", "").split(",") if x)
RUN = os.environ.get("ENG_MIXED_CXX_RUN", "0")   # stage 1d: "1" = spans on the 4 driver threads, "serial" = the same spans
                                                 # issued by the calling thread (reference), "0" = off (1b-i only)
VERSION = 1
STAMP = os.environ.get("ENG_MIXED_CXX_STAMP", "0") == "1"   # run_span wall time and the
    # Python pieces' time between spans, accumulated in the driver counts (span_us / span_max_us / py_us / py_calls)
CP_SPAN = os.environ.get("ENG_MIXED_CXX_CP", "0")   # CP prefill segments (bucket
    # A with engine._cp) in spans too: every C++-covered piece as in non-CP segments. "1": the three CP attention phases
    # (Engine._cp_p1 / _p2 / _p3 for L <= 20) stay Python, each for every rank before the next as Engine._issue orders them;
    # "3": + the X1a / X2 exchanges as pieces too (pieces_attn cp_x1a_* / cp_x2_*; RUN=1 threads only): the indexer's
    # compressor / scoring of index layers stay Python
MAX_CHUNKS = 4
FINAL_MAX_CHUNKS = 16
W = 4


class FprCall(C.Structure):
    _fields_ = [("stream", P), ("device", C.c_int), ("rank", C.c_int)]


class FrowArgs(C.Structure):
    _fields_ = [("c", FprCall), ("y", P), ("rs", P), ("post", P), ("comb", P), ("lag", P), ("normw", P),
                ("norm_eps", C.c_float), ("xn", P), ("out", P), ("ptok_o", P), ("R", C.c_int32)]


class FmixArgs(C.Structure):
    _fields_ = [("c", FprCall), ("xn", P), ("fnf", P), ("scratch", P), ("R", C.c_int32), ("nrs", C.c_int32)]


class FbMixArgs(C.Structure):
    _fields_ = [("c", FprCall), ("scratch", P), ("scale", P), ("base", P), ("norm_eps", C.c_float), ("hc_eps", C.c_float),
                ("iters", C.c_int32), ("pre_o", P), ("post_o", P), ("comb_o", P), ("R", C.c_int32)]


class FbOutArgs(C.Structure):
    _fields_ = [("c", FprCall), ("scratch", P), ("normw", P), ("norm_eps", C.c_float), ("out", P), ("ptok_o", P),
                ("R", C.c_int32)]


class FaArgs(C.Structure):
    _fields_ = [("c", FprCall), ("y", P), ("rs", P), ("post", P), ("comb", P), ("lag", P), ("fnf", P), ("xn", P),
                ("scratch", P), ("R", C.c_int32), ("nrs", C.c_int32)]


class TopPack3Args(C.Structure):      # moe_raw_cabi.h FprMoeRawRouterTopPack3Args
    _fields_ = [("c", FprCall), ("logits", P), ("bias", P), ("ids", P), ("wts", P), ("route_scale", C.c_float), ("row", P),
                ("valid", P), ("ptok", P), ("T", C.c_int32)]


class AgPush2Args(C.Structure):       # comm_cabi.h
    _fields_ = [("c", FprCall), ("src", P), ("src2", P), ("R", C.c_int32), ("RB", C.c_int32), ("RB2", C.c_int32),
                ("bufs", P * W), ("bufs2", P * W), ("flags", P * W), ("epoch", P), ("ncta", C.c_int32)]


class WaitArgs(C.Structure):
    _fields_ = [("c", FprCall), ("flag", P), ("epoch", P), ("signals", C.c_uint32)]


class RsPushArgs(C.Structure):
    _fields_ = [("c", FprCall), ("part", P), ("R", C.c_int32), ("H", C.c_int32), ("bufs", P * W), ("flags", P * W),
                ("epoch", P), ("ncta", C.c_int32)]


class RsSumArgs(C.Structure):
    _fields_ = [("c", FprCall), ("recv", P), ("R", C.c_int32), ("H", C.c_int32), ("flag", P), ("epoch", P),
                ("signals", C.c_uint32), ("out", P), ("ncta", C.c_int32)]


class K1aXArgs(C.Structure):          # mhc_cabi.h
    _fields_ = [("c", FprCall), ("y", P), ("rs", P), ("post", P), ("comb", P), ("xn", P), ("R", C.c_int32)]


class K1bOutArgs(C.Structure):
    _fields_ = [("c", FprCall), ("xn", P), ("lag", P), ("normw", P), ("norm_eps", C.c_float), ("out", P), ("ptok_o", P),
                ("R", C.c_int32)]


class PieceRouter(C.Structure):       # pieces.h
    _fields_ = [("version", C.c_int32), ("layer", C.c_int32), ("R", C.c_int32), ("ce_ok", C.c_int32), ("main", P),
                ("cs", P * W), ("gf_dst", P * W), ("f_src", P), ("nb_f", C.c_int64), ("f", P), ("f_st0", C.c_int64),
                ("f_st1", C.c_int64), ("gw", P), ("gw_st0", C.c_int64), ("gw_st1", C.c_int64), ("lg", P),
                ("lg_st0", C.c_int64), ("lg_st1", C.c_int64), ("D", C.c_int32), ("E", C.c_int32), ("top", TopPack3Args)]


class PieceAg(C.Structure):
    _fields_ = [("version", C.c_int32), ("layer", C.c_int32), ("R", C.c_int32), ("ce_ok", C.c_int32), ("main", P),
                ("cs", P * W), ("gr_dst", P * W), ("rr_src", P), ("nb_r", C.c_int64), ("push", AgPush2Args),
                ("wait", WaitArgs)]


class PieceRs(C.Structure):
    _fields_ = [("version", C.c_int32), ("layer", C.c_int32), ("R", C.c_int32), ("rsf_ok", C.c_int32), ("push", RsPushArgs),
                ("sum", RsSumArgs), ("sum_rsf", RsSumArgs)]


class PieceFinal(C.Structure):
    _fields_ = [("version", C.c_int32), ("R", C.c_int32), ("path", C.c_int32), ("n_chunks", C.c_int32), ("main", P),
                ("frow", FrowArgs), ("k1a", K1aXArgs * FINAL_MAX_CHUNKS), ("k1b", K1bOutArgs * FINAL_MAX_CHUNKS)]


class ProgEntry(C.Structure):         # pieces.h FprProgEntry
    _fields_ = [("kind", C.c_int32), ("layer", C.c_int32), ("handle", C.c_int64), ("arg", C.c_int32), ("pad_", C.c_int32)]


K_BD, K_ROUTER, K_AG, K_AGCE_JOIN, K_RS, K_FINAL, K_JOIN, K_WAITEV = 1, 2, 3, 4, 5, 6, 7, 8


class StepParams(C.Structure):
    _fields_ = [("chain_input", C.c_int32), ("moe_variant", C.c_int32), ("moe_vcount", C.c_int64), ("moe_n_wait", C.c_int32),
                ("pad0_", C.c_int32), ("moe_wait", P * 4)]


class PieceBd(C.Structure):
    _fields_ = [("version", C.c_int32), ("which", C.c_int32), ("layer", C.c_int32), ("R", C.c_int32), ("path", C.c_int32),
                ("first_layer", C.c_int32), ("n_chunks", C.c_int32), ("aux_index", C.c_int32), ("main", P), ("side", P),
                ("frow", FrowArgs * 2), ("fmix", FmixArgs), ("fb_mix", FbMixArgs), ("fa", (FaArgs * MAX_CHUNKS) * 2),
                ("fb_out", FbOutArgs * MAX_CHUNKS), ("fb_mix_c", FbMixArgs * MAX_CHUNKS), ("aux_X", P), ("aux_tmp", P),
                ("aux_dst", P), ("aux_dst_stride0", C.c_int64), ("aux_HC", C.c_int32), ("aux_D", C.c_int32)]


# Deferred handle frees: run_span releases the GIL while the 4 driver threads run, and vLLM's EngineCore
# has other Python threads (input / output sockets). A GC run in one of them can call a dropped state's _Cache.__del__ in
# the middle of a span: a free_handle there would mutate the slot tables the driver threads are reading. So frees from any
# thread only QUEUE (list.append is atomic under the GIL), and drain_frees() applies them on the engine thread between spans
# (Driver.set_step and issue). Owners (moe, c1mega, attn, ...) free their handles through defer_free too.
_PENDING = []


def defer_free(fn, *args) -> None:
    """Queue fn(*args) (a handle free); applied by drain_frees() on the engine thread between spans. Any thread."""
    _PENDING.append((fn, args))


def drain_frees() -> int:
    """Apply the queued frees (engine thread, between spans only). Returns how many ran."""
    n = 0
    while _PENDING:
        fn, args = _PENDING.pop()
        try:
            fn(*args)
        except Exception:  # noqa: BLE001  (an unknown / already freed handle is ignored by the driver)
            pass
        n += 1
    return n


class _Cache(dict):
    """s["_cxx"]: a bucket state's piece handles, keyed (kind, ...). When the state is dropped (bucket eviction), the FPR
    kinds' driver structs are freed (their slots reused); owners free their own."""
    FPR_KINDS = ("bd", "router", "ag", "rs", "final")
    ENGRAM_KINDS = ("embed", "engram_bd")                 # pieces_engram.cpp (free_engram)

    def __init__(self, x, base=()):
        super().__init__(base)
        self._x = x

    def _free(self):
        try:
            for k, h in self.items():
                if isinstance(k, tuple) and k and isinstance(h, int) and h >= 0:
                    if k[0] in self.FPR_KINDS:
                        defer_free(self._x.free_handle, k[0], h)
                    elif k[0] in self.ENGRAM_KINDS:
                        defer_free(self._x.free_engram, k[0], h)
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass

    def clear(self):                     # Engine._drop_state (ENG_MIX_POOL) clears dict values before dropping the state
        self._free()
        super().clear()

    def __del__(self):
        self._free()


AUX_SHARED = os.environ.get("ENG_MIXED_CXX_AUX_SHARED", "0") == "1"
# bd piece's aux mean scratch as ONE buffer per rank shared by every mixed bucket (the pieces of a rank run in its stream
# order and a bd call fills and drains it within the call) instead of a persistent [R, D] per bucket state outside the
# mixed-bucket byte budget. Sized once for the largest served mixed step (geometry prefill cap + the largest decode bucket,
# rounded to mixed_round): a stable address, as cached piece args require. A bucket beyond it keeps its own buffer
# (today's path). Same kernels, same values: bitwise.
_AUX_TMP = {}


def _shared_aux_tmp(E, r, R, X):
    t = _AUX_TMP.get(r)
    if t is None:
        g = getattr(E, "geom", None)
        pf = int(getattr(g, "max_prefill_tokens_per_rank", 0) or 0)
        rnd = max(1, int(E.cfg.mixed_round))
        n = pf + int(E.cfg.buckets[-1]) if pf > 0 else int(E.cfg.max_mixed_rows)
        n = -(-n // rnd) * rnd
        t = _AUX_TMP[r] = torch.empty(n, X.shape[-1], dtype=X.dtype, device=X.device)
    if R > t.shape[0] or t.dtype != X.dtype or t.shape[1] != X.shape[-1] or t.device != X.device:
        return None
    return t


def _cache(drv, s):
    c = s.get("_cxx")
    if not isinstance(c, _Cache):
        c = s["_cxx"] = _Cache(drv.x, c or ())
    return c


@functools.cache
def ext():
    from torch.utils.cpp_extension import load
    bd = os.path.join(HERE, "build", "driver")
    os.makedirs(bd, exist_ok=True)
    kern = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kernels")
    import glob
    srcs = [os.path.join(HERE, "driver.cpp")] + sorted(glob.glob(os.path.join(HERE, "pieces_*.cpp")))   # owners' pieces
    return load(name="fpr_mixed_cxx", sources=srcs, extra_include_paths=[HERE, kern],
                extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"], extra_ldflags=["-ldl", "-lcublasLt"], build_directory=bd,
                with_cuda=True, verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))


def _p(t):
    return C.c_void_p(t.data_ptr()) if t is not None and t.numel() else C.c_void_p(0)


class Driver:
    """One per Engine (rank streams / side streams / devices from it)."""

    def __init__(self, eng, EM):
        self.eng, self.EM, self.x = eng, EM, ext()      # EM = the engine module (MHCF, ENGRAM_LAYERS, AUX_LAYERS)
        assert self.x.sizeof_bd() == C.sizeof(PieceBd), (self.x.sizeof_bd(), C.sizeof(PieceBd))
        if eng.mhcf is not None:
            self.x.register_table(eng.mhcf.__file__, "mhcf")
        for nm, sz in (("router", PieceRouter), ("ag", PieceAg), ("rs", PieceRs), ("final", PieceFinal)):
            assert getattr(self.x, f"sizeof_{nm}")() == C.sizeof(sz), (nm, getattr(self.x, f"sizeof_{nm}")(), C.sizeof(sz))
        self.x.register_table(eng.cext.__file__, "comm")      # ag / rs
        self.x.register_table(eng.hext.__file__, "mhc")       # final (path 2)
        mf = {m.__file__ for m in eng.mext}
        self._mext_ok = len(mf) == 1                          # one moe_raw .so for every rank (bench.build_extr's cache)
        if self._mext_ok:
            self.x.register_table(eng.mext[0].__file__, "moe_raw")   # router
        try:
            import agrs_engine as AGE                         # (kernels/agrs on sys.path: engine._install_agrs)
        except ImportError:
            AGE = None
        self.AGE = AGE
        import atexit
        import collections
        self.n = collections.Counter()          # calls per piece / variant (py_<kind>: prod calls the Python chain took)
        atexit.register(lambda: print(f"mixed_cxx: counts {dict(sorted(self.n.items()))}", flush=True))
        if STAMP and hasattr(self.x, "stamps"):   # (ENG_MIXED_CXX_STAMP: driver.cpp per-rank issue / wait / lag, us)
            atexit.register(lambda: print(f"mixed_cxx: stamps {self.x.stamps()}", flush=True))
        assert self.x.sizeof_prog_entry() == C.sizeof(ProgEntry), (self.x.sizeof_prog_entry(), C.sizeof(ProgEntry))
        for r, d in enumerate(eng.devs):
            self.x.init_rank(r, torch.device(d).index)
            self.x.set_streams(r, eng.streams[r].cuda_stream)
        assert self.x.sizeof_step_params() == C.sizeof(StepParams), (self.x.sizeof_step_params(), C.sizeof(StepParams))
        self._handles = {}                       # stage 1d: owners' handle_<kind>(drv, r, s, L) -> handle or -1
        self._pieces, self._sp_hooks = {}, []   # owners: mixed_cxx_<owner>.py defines piece_<kind>(drv, r, s, L) -> bool,
        import importlib                         # optional setup(drv) (register its tables, per-rank init) and
        for mod in ("mixed_cxx_moe", "mixed_cxx_attn", "mixed_cxx_head", "mixed_cxx_engram", "mixed_cxx_c1mega",
                    "mixed_cxx_ltwarm"):   #    # step_params
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue
            if hasattr(m, "setup"):
                m.setup(self)
            if hasattr(m, "step_params"):
                self._sp_hooks.append(m.step_params)
            for k in dir(m):
                if k.startswith("piece_"):
                    self._pieces[k[len("piece_"):]] = getattr(m, k)
                if k.startswith("handle_"):
                    self._handles[k[len("handle_"):]] = getattr(m, k)
        eng.log(f"mixed_cxx: driver up, pieces {sorted(PIECES)}, owner pieces {sorted(self._pieces)} "
                f"(FprPieceBd {C.sizeof(PieceBd)} B)")

    def set_step(self, st) -> None:
        """The pre-step's per-rank StepParams for a bucket segment (st = the segment's 4 states), before its pieces run."""
        drain_frees()                    # between spans, on the engine thread (see defer_free)
        for r, s in enumerate(st):
            sp = StepParams()
            sp.chain_input = int(bool(s.get("chain_input")))
            sp.moe_variant, sp.moe_vcount = -1, int(s.get("vcount", 0) or 0)
            for h in self._sp_hooks:
                h(self, r, s, sp)
            self.x.set_step_params(r, bytes(sp))

    def piece(self, kind, r, s, L) -> bool:
        """An owner's C++ piece (mixed_cxx_<owner>.py); False = absent or not eligible (the Python piece runs)."""
        f = self._pieces.get(kind)
        return bool(f is not None and f(self, r, s, L))

    # ---- the boundary piece
    def _call(self, r, s, dev):
        return FprCall(C.c_void_p(self.eng.streams[r].cuda_stream), torch.device(dev).index, r)

    def build_bd(self, r, s, L, which, engram_sets=None):
        """FprPieceBd for (bucket state s, layer L, which 0 attn_bd / 1 ffn_bd), mirroring p_attn_bd / p_ffn_bd ->
        _split_bd (ENG_MHCF fused path). path 0 = not eligible (the Python piece runs). engram_sets (mixed_cxx_engram): the
        Engram layers' identity-mode inputs [(zy, X1, zpost, eyec, lag) per _inputs set] after the inject."""
        E, EM = self.eng, self.EM
        lw = E.lw[r][L]
        fnf = lw.get("fa_f") if which == 0 else lw.get("ff_f")
        R = s["R"]
        st = PieceBd()
        st.version, st.which, st.layer, st.R = VERSION, which, L, R
        st.aux_index = -1
        ok = EM.MHCF and fnf is not None and E.mhcf is not None and hasattr(E.mhcf, "frow")
        if which == 0 and L in EM.ENGRAM_LAYERS and E.hooks.engram is not None and engram_sets is None:
            ok = False                       # Engram inject: mixed_cxx_engram's engram_bd piece (or Python)
        if not ok:
            st.path = 0
            return bytes(st)
        cfg = E.mcfg
        dev = s["dev"]
        call = self._call(r, s, dev)
        side = FprCall(C.c_void_p(E.side[r].cuda_stream), torch.device(dev).index, r)   # fmix / fb_mix (Python: side stream)
        st.main = C.c_void_p(E.streams[r].cuda_stream)
        st.side = C.c_void_p(E.side[r].cuda_stream)
        st.first_layer = int(L == E.layers[0])
        if which == 0:
            sets = engram_sets or [(s["x0"], s["res0"], s["post0"], s["comb0"], s["lag0"]),
                                   (s["moe_out"], s["xm2"], s["f_post"], s["f_comb"], s["f_pre"])]
            xn, sc, bs, nw, out, ptok = s["X"], lw["sa"], lw["ba"], lw["na"], s["a"], s["empty_i"]
            pre, post, comb = s["a_pre"], s["a_post"], s["a_comb"]
        else:
            y = s["ws"]["y"]
            sets = [(y, s["X"], s["a_post"], s["a_comb"], s["a_pre"])] * 2
            xn, sc, bs, nw, out, ptok = s["xm2"], lw["sf"], lw["bf"], lw["nf"], s["f"], s["ptok_own"]
            pre, post, comb = s["f_pre"], s["f_post"], s["f_comb"]
        part = s["part_mhc"]
        eps, hc_eps, iters = float(cfg.norm_eps), float(cfg.hc_eps), int(cfg.hc_iters)
        if R <= 320:
            st.path = 1
            for i, (xi, ri, pi, ci, lag) in enumerate(sets):
                st.frow[i] = FrowArgs(call, _p(xi[:R]), _p(ri[:R]), _p(pi[:R]), _p(ci[:R]), _p(lag[:R]), _p(nw), eps,
                                      _p(xn[:R]), _p(out[:R]), _p(ptok[:R]) if ptok.numel() else C.c_void_p(0), R)
            st.fmix = FmixArgs(side, _p(xn[:R]), _p(fnf), _p(part), R, 0)
            st.fb_mix = FbMixArgs(side, _p(part), _p(sc), _p(bs), eps, hc_eps, iters, _p(pre[:R]), _p(post[:R]),
                                  _p(comb[:R]), R)
        else:
            mc = E.cfg.mhc_chunk
            chunks = [(c0, min(R, c0 + mc)) for c0 in range(0, R, mc)]
            if len(chunks) > MAX_CHUNKS:
                st.path = 0
                return bytes(st)
            st.path, st.n_chunks = 2, len(chunks)
            for j, (c0, c1) in enumerate(chunks):
                pt = part[160 * 25 * c0: 160 * 25 * c1]
                for i, (xi, ri, pi, ci, lag) in enumerate(sets):
                    st.fa[i][j] = FaArgs(call, _p(xi[c0:c1]), _p(ri[c0:c1]), _p(pi[c0:c1]), _p(ci[c0:c1]), _p(lag[c0:c1]),
                                         _p(fnf), _p(xn[c0:c1]), _p(pt), c1 - c0, 0)
                st.fb_out[j] = FbOutArgs(call, _p(pt), _p(nw), eps, _p(out[c0:c1]),
                                         _p(ptok[c0:c1]) if ptok.numel() else C.c_void_p(0), c1 - c0)
                st.fb_mix_c[j] = FbMixArgs(side, _p(pt), _p(sc), _p(bs), eps, hc_eps, iters, _p(pre[c0:c1]), _p(post[c0:c1]),
                                           _p(comb[c0:c1]), c1 - c0)
        if which == 0 and L in EM.AUX_LAYERS:
            X = s["X"]
            tmp = _shared_aux_tmp(E, r, R, X) if AUX_SHARED else None
            if tmp is None:
                tmp = s.get("_cxx_aux_tmp")
            if tmp is None:
                tmp = s["_cxx_aux_tmp"] = torch.empty(R, X.shape[-1], dtype=X.dtype, device=X.device)
            dst = s["aux"][:, EM.AUX_LAYERS.index(L)]
            st.aux_index = EM.AUX_LAYERS.index(L)
            st.aux_X, st.aux_tmp, st.aux_dst = _p(X), _p(tmp), C.c_void_p(dst.data_ptr())
            st.aux_dst_stride0, st.aux_HC, st.aux_D = dst.stride(0), X.shape[1], X.shape[2]
        return bytes(st)

    def bd(self, r, s, L, which) -> bool:
        """Run the boundary piece in C++; False = not eligible (the caller runs the Python piece)."""
        cache = _cache(self, s)
        key = ("bd", L, which)
        h = cache.get(key)
        if h is None:
            b = self.build_bd(r, s, L, which)
            st = PieceBd.from_buffer_copy(b)
            h = cache[key] = self.x.add_bd(b) if st.path else -1
        if h < 0:
            return False
        self._py_mix(s)
        rc = self.x.bd(r, h, int(bool(s.get("chain_input"))))
        assert rc == 0, f"mixed_cxx: bd rank {r} layer {L} which {which}: FPR error {rc}"
        self.n["bd"] += 1
        return True

    def join(self, r, stream) -> None:
        self.x.join(r, stream.cuda_stream)

    @staticmethod
    def _py_mix(s) -> None:
        """A PYTHON boundary's pending side-stream mixes (s["mix_ev"]: Engram layers' attn_bd, chunked boundaries): the
        current stream waits them, as the next Python boundary's _join_mix would, before a C++ boundary / final that only
        joins the driver's own pending mix."""
        ev = s.pop("mix_ev", None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    # ---- router / ag / rs / final (prod buckets; pieces.h). Each build_* returns None when the bucket / layer is not
    # covered (the caller runs the Python chain); the handle cache keys include the variant's static inputs.
    def _cached(self, s, key, build, add):
        cache = _cache(self, s)
        h = cache.get(key)
        if h is None:
            b = build()
            h = cache[key] = add(b) if b is not None else -1
        return h

    def _cs(self, r):
        return [c.cuda_stream for c in self.AGE._cstreams(self.eng, r)]

    def build_router(self, r, s, L, st):
        E = self.eng
        lw = E.lw[r][L]
        R = s["R"]
        f, gw, lg, rr, valid, ptok = s["f"], lw["gate_w"], s["logits"], s["route_row"], s["valid"], s["ptok_own"]
        if not (f.dtype == torch.bfloat16 and gw.dtype == torch.bfloat16 and lg.dtype == torch.float32 and f.dim() == 2
                and gw.dim() == 2 and lg.dim() == 2 and f.shape[0] == R and gw.shape[1] == f.shape[1]
                and tuple(lg.shape) == (R, gw.shape[0]) and lw["gate_b"].dtype == torch.float32
                and rr.numel() * rr.element_size() >= R * 64 and valid.numel() >= R and valid.element_size() == 1
                and valid.is_contiguous() and ptok.numel() >= R and ptok.dtype == torch.int32):
            return None
        p = PieceRouter()
        p.version, p.layer, p.R = VERSION, L, R
        p.main = C.c_void_p(E.streams[r].cuda_stream)
        if st is not None:
            nb = R * self.AGE.D * 2                         # agrs_engine.p_router's nb
            for q, (c, x) in enumerate(zip(self._cs(r), st)):
                p.cs[q], p.gf_dst[q] = C.c_void_p(c), C.c_void_p(x["gf"].data_ptr() + r * nb)
            p.f_src, p.nb_f, p.ce_ok = C.c_void_p(f.data_ptr()), nb, 1
        p.f, p.f_st0, p.f_st1 = C.c_void_p(f.data_ptr()), f.stride(0), f.stride(1)
        p.gw, p.gw_st0, p.gw_st1 = C.c_void_p(gw.data_ptr()), gw.stride(0), gw.stride(1)
        p.lg, p.lg_st0, p.lg_st1 = C.c_void_p(lg.data_ptr()), lg.stride(0), lg.stride(1)
        p.D, p.E = f.shape[1], gw.shape[0]
        p.top = TopPack3Args(FprCall(p.main, torch.device(s["dev"]).index, 0), _p(lg), _p(lw["gate_b"]), _p(s["ids"]),
                             _p(s["wts"]), 1.5, _p(rr), _p(valid), _p(ptok), lg.shape[0])
        return bytes(p)

    def router(self, r, s, L, st) -> bool:
        """st = the bucket's 4 states when this call takes AGCE (agrs' agce_on and _agrs_st_last[r] is s), else None."""
        if not self._mext_ok:
            return False
        h = self._cached(s, ("router", L, st is not None), lambda: self.build_router(r, s, L, st), self.x.add_router)
        if h < 0:
            return False
        rc = self.x.router(r, h, int(st is not None))
        assert rc == 0, f"mixed_cxx: router rank {r} layer {L}: FPR error {rc}"
        self.n["router_ce" if st is not None else "router"] += 1
        return True

    def build_ag(self, r, s, L, st):
        E = self.eng
        R, f, rr = s["R"], s["f"], s["route_row"]
        D2 = f.shape[-1] * f.element_size()
        RB2 = rr.numel() // rr.shape[0] * rr.element_size()
        if not (f.is_contiguous() and f.shape[0] == R and rr.shape[0] == R and D2 % 16 == 0 and RB2 % 16 == 0):
            return None
        p = PieceAg()
        p.version, p.layer, p.R = VERSION, L, R
        p.main = C.c_void_p(E.streams[r].cuda_stream)
        dev = torch.device(s["dev"]).index
        if self.AGE is not None and getattr(self.EM, "AGRS_AGCE", False) and R >= self.AGE.MIN_R:
            nb = R * 64                                     # agrs_engine.p_ag's nb
            for q, (c, x) in enumerate(zip(self._cs(r), st)):
                p.cs[q], p.gr_dst[q] = C.c_void_p(c), C.c_void_p(x["gr"].data_ptr() + r * nb)
            p.rr_src, p.nb_r, p.ce_ok = C.c_void_p(rr.data_ptr()), nb, 1
        nc = E.cfg.ncta
        p.push = AgPush2Args(FprCall(p.main, dev, r), _p(f), _p(rr), R, D2, RB2,
                             (P * W)(*[x["gf"].data_ptr() for x in st]), (P * W)(*[x["gr"].data_ptr() for x in st]),
                             (P * W)(*[x["fl_f"].data_ptr() for x in st]), _p(s["ep_f"]), nc)
        p.wait = WaitArgs(FprCall(p.main, dev, 0), _p(s["fl_f"]), _p(s["ep_f"]), W * nc)
        return bytes(p)

    def ag(self, r, s, L, st, ce) -> bool:
        h = self._cached(s, ("ag", L), lambda: self.build_ag(r, s, L, st), self.x.add_ag)
        if h < 0:
            return False
        rc = self.x.ag(r, h, int(ce))
        assert rc == 0, f"mixed_cxx: ag rank {r} layer {L} ce {ce}: FPR error {rc}"
        self.n["ag_ce" if ce else "ag"] += 1
        return True

    def rs_grid(self, s) -> int:
        AGE = self.AGE
        return AGE.SUM_GRID if (AGE is not None and getattr(self.EM, "AGRS_SUM528", False) and s["R"] >= AGE.MIN_R) else 132

    def build_rs(self, r, s, L, st):
        E = self.eng
        part, rs, out = s["part"], s["rs"], s["moe_out"]
        if not (rs.dim() == 3 and rs.shape[0] == W and part.dim() == 2):
            return None
        p = PieceRs()
        p.version, p.layer, p.R = VERSION, L, s["R"]
        main = C.c_void_p(E.streams[r].cuda_stream)
        dev = torch.device(s["dev"]).index
        nc, grid = E.cfg.ncta, self.rs_grid(s)
        p.push = RsPushArgs(FprCall(main, dev, r), _p(part), part.shape[0] // W, part.shape[1],
                            (P * W)(*[x["rs"].data_ptr() for x in st]), (P * W)(*[x["fl_s"].data_ptr() for x in st]),
                            _p(s["ep_s"]), nc)
        p.sum = RsSumArgs(FprCall(main, dev, 0), _p(rs), rs.shape[1], rs.shape[2], _p(s["fl_s"]), _p(s["ep_s"]), W * nc,
                          _p(out), grid)
        if "agrs_fl_rsf" in s:
            p.sum_rsf = RsSumArgs(FprCall(main, dev, 0), _p(rs), rs.shape[1], rs.shape[2], _p(s["agrs_fl_rsf"]),
                                  _p(s["agrs_ep_rsf"]), W * self.AGE.RSF_GRID, _p(out), grid)
            p.rsf_ok = 1
        return bytes(p)

    def rs(self, r, s, L, st, rsf) -> bool:
        h = self._cached(s, ("rs", L, "agrs_fl_rsf" in s), lambda: self.build_rs(r, s, L, st), self.x.add_rs)
        if h < 0:
            return False
        rc = self.x.rs(r, h, int(rsf))
        assert rc == 0, f"mixed_cxx: rs rank {r} layer {L} rsf {rsf}: FPR error {rc}"
        self.n["rs_rsf" if rsf else f"rs_g{self.rs_grid(s)}"] += 1
        return True

    def build_final(self, r, s):
        E, EM = self.eng, self.EM
        R = s["R"]
        main = C.c_void_p(E.streams[r].cuda_stream)
        call = FprCall(main, torch.device(s["dev"]).index, 0)
        y, rs_, post, comb, lag, Xf, h, ptok = (s["moe_out"], s["xm2"], s["f_post"], s["f_comb"], s["f_pre"], s["Xf"], s["h"],
                                                s["empty_i"])
        nw, eps = E.glob[r]["norm_w"], float(E.mcfg.norm_eps)
        p = PieceFinal()
        p.version, p.R, p.main = VERSION, R, main
        if EM.MHCF and E.mhcf is not None and R <= 320 and hasattr(E.mhcf, "frow"):
            p.path = 1
            p.frow = FrowArgs(call, _p(y), _p(rs_), _p(post), _p(comb), _p(lag), _p(nw), eps, _p(Xf), _p(h), _p(ptok),
                              y.shape[0])
            return bytes(p)
        mc = E.cfg.mhc_chunk
        chunks = [(c0, min(R, c0 + mc)) for c0 in range(0, R, mc)]
        if len(chunks) > FINAL_MAX_CHUNKS:
            return None
        p.path, p.n_chunks = 2, len(chunks)
        for j, (c0, c1) in enumerate(chunks):
            p.k1a[j] = K1aXArgs(call, _p(y[c0:c1]), _p(rs_[c0:c1]), _p(post[c0:c1]), _p(comb[c0:c1]), _p(Xf[c0:c1]), c1 - c0)
            p.k1b[j] = K1bOutArgs(call, _p(Xf[c0:c1]), _p(lag[c0:c1]), _p(nw), eps, _p(h[c0:c1]), _p(ptok), c1 - c0)
        return bytes(p)

    def final(self, r, s) -> bool:
        h = self._cached(s, ("final",), lambda: self.build_final(r, s), self.x.add_final)
        if h < 0:
            return False
        self._py_mix(s)
        rc = self.x.final(r, h)
        assert rc == 0, f"mixed_cxx: final rank {r}: FPR error {rc}"
        self.n["final"] += 1
        return True


def issue(drv, st, layers, final, embed) -> bool:
    drain_frees()
    return RUN != "0" and _issue_span(drv, st, layers, final, embed)


def install(EM) -> None:
    """Wrap the served chain of Engine.p_router / p_ag / p_moe / p_rs / p_final (agrs_engine's wrappers included) from
    outside, per ENG_MIXED_CXX_PIECES. Called by engine.py after _install_agrs(). Every wrapper falls through to the chain
    when its piece is off or the call is not covered."""
    E = EM.Engine
    if getattr(E, "_cxx_installed", False) or not PIECES & {"router", "ag", "rs", "final"}:
        return
    o_router, o_ag, o_moe, o_rs, o_final = E.p_router, E.p_ag, E.p_moe, E.p_rs, E.p_final
    cap = torch.cuda.is_current_stream_capturing
    agrs_on = bool(getattr(E, "_agrs_installed", False))

    def p_router(self, r, s, L):
        if s.get("moe") == "prod" and s.get("mm_rt") is None and self.cfg.router_pack3 and not cap():
            d = self._cxx_drv()
            AGE, stl = d.AGE, getattr(self, "_agrs_st_last", None)
            ce = (agrs_on and AGE is not None and getattr(EM, "AGRS_AGCE", False) and s["R"] >= AGE.MIN_R
                  and stl is not None and stl[r] is s)
            if d.router(r, s, L, stl if ce else None):
                if ce:
                    s["agrs_ce_issued"] = True
                return
            d.n["py_router"] += 1
        return o_router(self, r, s, L)

    def p_ag(self, r, s, L, st):
        if s.get("moe") == "prod" and not cap():
            d = self._cxx_drv()
            AGE = d.AGE
            ce = bool(s.get("agrs_ce_issued", False))
            ag32 = agrs_on and AGE is not None and getattr(EM, "AGRS_AG32", False) and s["R"] >= AGE.MIN_R
            if (ce or not ag32) and d.ag(r, s, L, st, ce):
                s.pop("agrs_ce_issued", None)
                if agrs_on:
                    self._agrs_st = st
                    self._agrs_st_last = st
                    if ce:
                        AGE.stats["agce"] = AGE.stats.get("agce", 0) + 1
                return
            d.n["py_ag"] += 1
        return o_ag(self, r, s, L, st)

    def p_moe(self, r, s, L):
        d = getattr(self, "_cxx", None)
        if d is not None:                     # AGCE copies the C++ ag piece issued into this rank: wait them first
            d.x.agce_join(r, torch.cuda.current_stream().cuda_stream)
        return o_moe(self, r, s, L)

    def p_rs(self, r, s, L, st):
        if s.get("moe") == "prod" and not cap() and not EM.C1_L2PF and "agrs_rsd_pending" not in s:
            d = self._cxx_drv()
            rsf = bool(s.get("agrs_rsf_pending", False))
            if (not rsf or "agrs_fl_rsf" in s) and d.rs(r, s, L, st, rsf):
                s.pop("agrs_rsf_pending", None)
                if agrs_on and d.rs_grid(s) != 132:
                    d.AGE.stats["sum528"] += 1
                return
            d.n["py_rs"] += 1
        return o_rs(self, r, s, L, st)

    def p_final(self, r, s):
        if s.get("moe") == "prod" and not cap():
            if not (EM.C1_L2PF and "pf_evs" in s) and self._cxx_drv().final(r, s):
                self.hooks.head_run(r, s)
                return
            self._cxx_drv().n["py_final"] += 1
        return o_final(self, r, s)

    if "router" in PIECES:
        E.p_router = p_router
    if "ag" in PIECES:
        E.p_ag, E.p_moe = p_ag, p_moe
    if "rs" in PIECES:
        E.p_rs = p_rs
    if "final" in PIECES:
        E.p_final = p_final
    E._cxx_installed = True

class _Span:
    """Stage 1d: per-rank program entries collected in Engine._issue order; flush() runs them on the driver threads."""

    def __init__(self, drv, st):
        self.d, self.st = drv, st
        self.prog = [[] for _ in st]
        self.refs = []                       # Python events the span waits (alive until it ran)
        self.n_runs = 0
        self.dq = False                      # pending entries dequantize into engine._DQ_DEV slots

    def add(self, r, kind, layer=-1, handle=-1, arg=0, joins=False):
        """Entries are added rank 0 first for each kind. A joining entry (bd / final / join / owner joins=True) first takes
        any Python boundary's pending s["mix_ev"] of every rank: a K_WAITEV entry (handle 0 = none) right before it, the
        point where Python's _join_mix would wait it."""
        if r == 0 and (joins or kind in (K_BD, K_FINAL, K_JOIN)):
            evs = [s.pop("mix_ev", None) for s in self.st]
            if any(e is not None for e in evs):
                for q, e in enumerate(evs):
                    if e is not None:
                        self.refs.append(e)
                    self.prog[q].append(ProgEntry(K_WAITEV, layer, int(e.cuda_event) if e is not None else 0, 0, 0))
        self.prog[r].append(ProgEntry(kind, layer, handle, arg, 0))

    def flush(self):
        if not any(self.prog):
            return
        d = self.d
        progs = [b"".join(bytes(e) for e in pr) for pr in self.prog]
        if self.dq:
            _dq_handover(self)
            self.dq = False
        t0 = time.perf_counter() if STAMP else 0.0
        rc = d.x.run_span(progs) if RUN != "serial" else d.x.run_span_serial(progs)
        if STAMP:
            us = int((time.perf_counter() - t0) * 1e6)
            d.n["span_us"] += us
            d.n["span_max_us"] = max(d.n["span_max_us"], us)
        if any(rc):
            raise RuntimeError("mixed_cxx: span failed: " + "; ".join(f"rank {r}: {c} {d.x.last_error(r)}"
                                                                     for r, c in enumerate(rc) if c))
        d.n["span"] += 1
        d.n["span_entries"] += sum(len(p) for p in self.prog)
        self.prog = [[] for _ in self.st]
        self.refs = []
        self.n_runs += 1

    def py(self, fn):
        """A Python piece for every rank (Engine._issue's each()), after the pending span. With ENG_FAST_EACH: each()'s fast switch (torch's C calls; the same device / stream state while fn runs and the same restore
        after it) instead of the two context managers (~30 us of Python per rank-piece)."""
        self.flush()
        E, EM = self.d.eng, self.d.EM
        t0 = time.perf_counter() if STAMP else 0.0
        if getattr(EM, "FAST_EACH", False):
            fx = self.__dict__.get("_fx")
            if fx is None:
                fx = self._fx = [(d.index if isinstance(d, torch.device) else int(d),
                                  (sm.stream_id, sm.device_index, sm.device_type)) for d, sm in zip(E.devs, E.streams)]
                assert all(dv == sm[1] for dv, sm in fx), "ENG_FAST_EACH: rank stream on another device"
            d0 = EM._c_get_dev()
            try:
                for r, s in enumerate(self.st):
                    dv, sm = fx[r]
                    EM._c_set_dev(dv)
                    prev = EM._c_get_stream(dv)
                    EM._c_set_stream(stream_id=sm[0], device_index=sm[1], device_type=sm[2])
                    try:
                        fn(r, s)
                    finally:
                        EM._c_set_stream(stream_id=prev[0], device_index=prev[1], device_type=prev[2])
            finally:
                EM._c_set_dev(d0)
        else:
            for r, s in enumerate(self.st):
                with torch.cuda.device(E.devs[r]), torch.cuda.stream(E.streams[r]):
                    fn(r, s)
        if STAMP:
            self.d.n["py_us"] += int((time.perf_counter() - t0) * 1e6)
            self.d.n["py_calls"] += 1


def _dq_handover(sp) -> None:
    """every pending side-stream dequant (slot.ready) waited on the rank stream, owner / last
    reset. The C++ pieces then overwrite the slots in the rank stream's order; Python's next _Dq.weight() dequantizes
    again (owner None) and its learned prefetch chain restarts (last None): its bookkeeping stays exact."""
    E, EM = sp.d.eng, sp.d.EM
    dd = getattr(EM, "_DQ_DEV", None) or {}
    for r in range(len(sp.st)):
        for sl in (dd.get(str(torch.device(E.devs[r]))) or {}).values():
            if sl.ready is not None:
                E.streams[r].wait_event(sl.ready)
                sl.ready = None
            sl.owner = sl.last = None
    sp.d.n["dq_handover"] += 1


def _issue_span(drv, st, layers, final, embed) -> bool:
    """Stage 1d (ENG_MIXED_CXX_RUN): Engine._issue for a prod bucket segment with every C++-covered piece in spans on the
    driver threads, the rest in Python between spans (same per-rank stream order; every decision as the 1b-i wrappers
    make it). False = not covered (the caller runs Engine._issue)."""
    E, EM, AGE = drv.eng, drv.EM, drv.AGE
    s0 = st[0]
    if (s0.get("moe") != "prod" or torch.cuda.is_current_stream_capturing() or EM.C1_L2PF or EM.NANCHECK or EM.DBG_SYNC
            or EM.PROF_PIECES or (getattr(E, "_cp", None) and s0.get("cp_on") and CP_SPAN == "0") or E._tp_on(s0)
            or E._mega_on(s0)
            or not E.cfg.router_pack3 or any(s.get("mm_rt") is not None for s in st) or not drv._mext_ok):
        return False
    agrs_on = bool(getattr(EM.Engine, "_agrs_installed", False))
    if agrs_on and AGE is not None and getattr(EM, "AGRS_AG32", False) and s0["R"] >= AGE.MIN_R:
        return False
    for r, s in enumerate(st):                                 # StepParams.chain_input = s["chain_input"] (BD / engram_bd)
        drv.x.set_chain_input(r, int(bool(s.get("chain_input"))))
    sp = _Span(drv, st)
    cp = E._cp if (CP_SPAN != "0" and getattr(E, "_cp", None) and s0.get("cp_on")) else None   # ENG_MIXED_CXX_CP

    def owner(kind, flag, L, joins=False):
        """an owner kind's entries for every rank (handle_<kind> >= 0 on all), else False (the caller runs Python)"""
        k = drv.x.owner_kind(kind) if flag in PIECES and kind in drv._handles else -1
        if k < 0:
            return False
        hs = [drv._handles[kind](drv, r, s, L) for r, s in enumerate(st)]
        if min(hs) < 0:
            return False
        for r, h in enumerate(hs):
            sp.add(r, k, -1 if L is None else L, h, 0, joins)
        drv.n[kind] += len(st)
        return True

    if embed and not owner("embed", "embed", None):
        sp.py(E.p_embed)
    E._tp_st = st
    moe_k = drv.x.owner_kind("moe") if "moe" in PIECES and "moe" in drv._handles else -1

    def bd(L, which):
        if "bd" not in PIECES:
            return False
        hs = []
        for r, s in enumerate(st):
            cache = _cache(drv, s)
            h = cache.get(("bd", L, which))
            if h is None:
                b = drv.build_bd(r, s, L, which)
                h = cache[("bd", L, which)] = drv.x.add_bd(b) if PieceBd.from_buffer_copy(b).path else -1
            hs.append(h)
        if "bd" not in PIECES or min(hs) < 0:
            return False
        for r, h in enumerate(hs):
            sp.add(r, K_BD, L, h)
        drv.n["bd"] += len(hs)
        return True

    ak = {k: (drv.x.owner_kind(k) if k in PIECES and k in drv._handles else -1) for k in ("attn_a", "idx", "attn_b")}

    def attn(L):
        """attn_a, idx (index layers with an indexer), attn_b."""
        kinds = ["attn_a"] + (["idx"] if (L in EM.INDEX_LAYERS and E.hooks.idx is not None) else []) + ["attn_b"]
        if min(ak[k] for k in kinds) < 0:
            return False
        hs = {k: [drv._handles[k](drv, r, s, L) for r, s in enumerate(st)] for k in kinds}
        if min(min(v) for v in hs.values()) < 0:
            return False
        for k in kinds:
            for r in range(len(st)):
                sp.add(r, ak[k], L, hs[k][r])
            drv.n[k] += len(st)
        dqh = drv.__dict__.get("_dq_handles")
        if dqh and any(h in dqh for h in hs["attn_b"]):
            sp.dq = True
            drv.n["attn_b_dq"] += len(st)
        return True

    def cp_attn(L):
        """ENG_MIXED_CXX_CP=2: Engine._cp_p1 / _cp_p2 / _cp_p3 of layer L with the attention calls as pieces (attn_a in P1,
        attn_bi in P2, attn_bo in P3; one attn_b handle), the indexer calls and the exchanges Python in between, in the
        phases' per-rank order, every phase's sends issued for all ranks before any receive (MIRROR of engine.py's
        _cp_p1 / _cp_p2 / _cp_p3: keep in sync). False = not covered (the caller runs the three Python phases)."""
        ka = drv.x.owner_kind("attn_a") if "attn_a" in PIECES and "attn_a" in drv._handles else -1
        kbi = drv.x.owner_kind("attn_bi") if "attn_b" in PIECES and "attn_bi" in drv._handles else -1
        kbo = drv.x.owner_kind("attn_bo") if "attn_b" in PIECES and "attn_bo" in drv._handles else -1
        if min(ka, kbi, kbo) < 0:
            return False
        kx = {k: drv.x.owner_kind(k) for k in ("cp_x1a_send", "cp_x1a_recv", "cp_x2_send", "cp_x2_recv")} \
            if (CP_SPAN == "3" and RUN == "1") else None
        if kx is not None and (min(kx.values()) < 0 or not sys.modules["mixed_cxx_attn"].cp_step(drv, E._cp, layers)):
            kx = None                                          # (mode 3 not covered: the exchanges stay Python)
        ha = [drv._handles["attn_a"](drv, r, s, L) for r, s in enumerate(st)]
        hb = [drv._handles["attn_b"](drv, r, s, L) for r, s in enumerate(st)] if min(ha) >= 0 else [-1]
        if min(ha) < 0 or min(hb) < 0:
            return False
        idx_l = E.hooks.idx is not None and L in EM.INDEX_LAYERS
        src_l = idx_l and L in EM.KV_SOURCE_LAYERS
        # the A-role indexer as pieces: idx_c (the compressor, P1) / idx_s (the scoring, P2), one A
        # handle per rank (mixed_cxx_attn.handle_idx_a); any rank without one: both halves Python on every rank
        hi = None
        if idx_l and "idx" in PIECES and "idx_c" in drv._handles and "idx_s" in drv._handles:
            kic, kis = drv.x.owner_kind("idx_c"), drv.x.owner_kind("idx_s")
            if min(kic, kis) >= 0:
                hi = [drv._handles["idx_c"](drv, r, s, L) for r, s in enumerate(st)]
                if min(hi) < 0:
                    hi = None

        def comp():                                            # P1: the compressor of every rank
            if hi is None:
                sp.py(lambda r, s: E.hooks.idx._compress_general(r, s, L))
            else:
                for r in range(len(st)):
                    sp.add(r, kic, L, hi[r])

        def score():                                           # P2: the scoring of every rank
            if hi is None:
                sp.py(lambda r, s: E.hooks.idx._score_general(r, s, L))
            else:
                for r in range(len(st)):
                    sp.add(r, kis, L, hi[r])
        for r in range(len(st)):                               # P1: attn_a; the compressor; X1a send
            sp.add(r, ka, L, ha[r])
        if idx_l and kx is not None:                           # mode 3: the exchanges as entries
            comp()
            if src_l:
                for r in range(len(st)):
                    sp.add(r, kx["cp_x1a_send"], L)
                for r in range(len(st)):                       # P2: X1a receive; the indexer's scoring
                    sp.add(r, kx["cp_x1a_recv"], L)
            score()
        elif idx_l:
            if hi is None:
                def p1b(r, s):
                    E.hooks.idx._compress_general(r, s, L)
                    if src_l:
                        E._cp.x1a_send(r, L)
                sp.py(p1b)

                def p2a(r, s):                                 # P2: X1a receive; the indexer's scoring
                    if src_l:
                        E._cp.x1a_recv(r, L)
                    E.hooks.idx._score_general(r, s, L)
                sp.py(p2a)
            else:                                              # (the same per-rank order with the halves as entries)
                comp()
                if src_l:
                    sp.py(lambda r, s: E._cp.x1a_send(r, L))
                    sp.py(lambda r, s: E._cp.x1a_recv(r, L))
                score()
        if hi is not None:
            drv.n["idx_c"] += len(st)
            drv.n["idx_s"] += len(st)
        dqh = drv.__dict__.get("_dq_handles")
        dq_ = bool(dqh) and any(h in dqh for h in hb)          # the handover before EACH flush of these entries
        for r in range(len(st)):                               # P2: q_b + RoPE + window insert; X2 send
            sp.add(r, kbi, L, hb[r])
        sp.dq = sp.dq or dq_
        if kx is not None:
            for r in range(len(st)):
                sp.add(r, kx["cp_x2_send"], L)
            for r in range(len(st)):                           # P3: X2 receive; FlashMLA .. wo_b
                sp.add(r, kx["cp_x2_recv"], L)
        else:
            sp.py(lambda r, s: E._cp.x2_send(r, L))
            sp.py(lambda r, s: E._cp.x2_recv(r, L))            # P3: X2 receive; FlashMLA .. wo_b
        for r in range(len(st)):
            sp.add(r, kbo, L, hb[r])
        sp.dq = sp.dq or dq_
        if dq_:
            drv.n["attn_b_dq"] += len(st)
        drv.n["cp_attn_layers"] += 1
        drv.n["attn_a"] += len(st)
        drv.n["attn_b"] += len(st)
        return True

    for L in layers:
        if L in EM.ENGRAM_LAYERS and E.hooks.engram is not None:
            if not owner("engram_bd", "engram", L, joins=True):
                sp.py(lambda r, s: E.p_attn_bd(r, s, L))
        elif not bd(L, 0):
            sp.py(lambda r, s: E.p_attn_bd(r, s, L))
        if cp is not None and L <= 20:     # Engine._issue's CP branch: every send of a phase issued for all ranks first
            if not (CP_SPAN in ("2", "3") and cp_attn(L)):
                sp.py(lambda r, s: E._cp_p1(r, s, L))
                sp.py(lambda r, s: E._cp_p2(r, s, L))
                sp.py(lambda r, s: E._cp_p3(r, s, L))
            drv.n["cp_layers"] += 1
        elif not attn(L):                                      # Engine.p_attn (prod, eager) = attn_a, indexer, attn_b
            sp.py(lambda r, s: E.p_attn(r, s, L))
        if not bd(L, 1):
            sp.py(lambda r, s: E.p_ffn_bd(r, s, L))
        # router + ag: both C++ or both Python (the AGCE flags stay consistent)
        ra = "router" in PIECES and "ag" in PIECES
        if ra:
            stl = getattr(E, "_agrs_st_last", None)
            hr, ha, ces = [], [], []
            for r, s in enumerate(st):
                ce = bool(agrs_on and AGE is not None and getattr(EM, "AGRS_AGCE", False) and s["R"] >= AGE.MIN_R
                          and stl is not None and stl[r] is s)
                h1 = drv._cached(s, ("router", L, ce), lambda: drv.build_router(r, s, L, stl if ce else None),
                                 drv.x.add_router)
                h2 = drv._cached(s, ("ag", L), lambda: drv.build_ag(r, s, L, st), drv.x.add_ag)
                hr.append(h1), ha.append(h2), ces.append(ce)
            ra = min(hr) >= 0 and min(ha) >= 0
        if ra:
            for r in range(len(st)):
                sp.add(r, K_ROUTER, L, hr[r], int(ces[r]))
            for r in range(len(st)):
                sp.add(r, K_AG, L, ha[r], int(ces[r]))
            if agrs_on:
                E._agrs_st = st
                E._agrs_st_last = st
                if any(ces):
                    AGE.stats["agce"] = AGE.stats.get("agce", 0) + sum(ces)
            drv.n["router_ce" if ces[0] else "router"] += len(st)
            drv.n["ag_ce" if ces[0] else "ag"] += len(st)
        else:
            sp.py(lambda r, s: E.p_router(r, s, L))
            sp.py(lambda r, s: E.p_ag(r, s, L, st))
        for r in range(len(st)):
            sp.add(r, K_AGCE_JOIN, L)                          # the receivers' AGCE waits, before the MoE
        # moe: an owner's C++ piece (handle_moe) or Python
        hm = [drv._handles["moe"](drv, r, s, L) for r, s in enumerate(st)] if (moe_k >= 0 and ra) else [-1]
        rsf = [False] * len(st)
        moe_cxx = min(hm) >= 0
        if moe_cxx:
            for r, s in enumerate(st):
                sp.add(r, moe_k, L, hm[r])
                rsf[r] = s.get("_cxx_moe_variant", -1) == 2     # PROD_RSF (mixed_plan.h): agrs's pending rs_sum
            drv.n["moe"] += len(st)
        else:
            sp.py(lambda r, s: E.p_moe(r, s, L))
            rsf = [bool(s.get("agrs_rsf_pending", False)) for s in st]
        # rs
        ok = "rs" in PIECES and all("agrs_rsd_pending" not in s and (not f or "agrs_fl_rsf" in s) for s, f in zip(st, rsf))
        hs = [drv._cached(s, ("rs", L, "agrs_fl_rsf" in s), lambda: drv.build_rs(r, s, L, st), drv.x.add_rs)
              for r, s in enumerate(st)] if ok else [-1]
        if min(hs) >= 0:
            for r, s in enumerate(st):
                sp.add(r, K_RS, L, hs[r], int(rsf[r]))
                s.pop("agrs_rsf_pending", None)
                if agrs_on and drv.rs_grid(s) != 132:
                    AGE.stats["sum528"] += 1
            drv.n["rs_rsf" if rsf[0] else f"rs_g{drv.rs_grid(s0)}"] += len(st)
        else:
            if moe_cxx:                                        # the C++ RSF layer pushed into rs: agrs's p_rs only sums
                for s, f in zip(st, rsf):
                    if f:
                        s["agrs_rsf_pending"] = True
            sp.py(lambda r, s: E.p_rs(r, s, L, st))
    if final:
        hf = [drv._cached(s, ("final",), lambda: drv.build_final(r, s), drv.x.add_final) for r, s in enumerate(st)] \
            if "final" in PIECES else [-1]
        if min(hf) >= 0:
            for r in range(len(st)):
                sp.add(r, K_FINAL, -1, hf[r])
            sp.flush()
            for r, s in enumerate(st):                         # p_final's tail: the HSD head stays Python
                with torch.cuda.device(E.devs[r]), torch.cuda.stream(E.streams[r]):
                    E.hooks.head_run(r, s)
            drv.n["final"] += len(st)
        else:
            sp.py(E.p_final)
    sp.py(lambda r, s: E._join_mix(s))
    drv.n["run_segments"] += 1
    if cp is not None:
        drv.n["cp_segments"] += 1
    return True
