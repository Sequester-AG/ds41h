"""mixed_cxx_engram: the Python side of pieces_engram.cpp.

Kinds (ENG_MIXED_CXX_PIECES): "embed" = Engine.p_embed (ids32 copy + EngramPhys.prepare + the token-embedding gather), "engram"
= Engine.p_attn_bd at the Engram layers (inject + the identity-mode boundary). Loaded by mixed_cxx.Driver: setup(drv) once,
piece_<kind>(drv, r, s, L) from the Engine hooks (1b-i), handle_<kind>(drv, r, s, L) from the span builder (1d).
Covered: prod buckets, the fixed-size prepare with the cuBLAS wkv (R > w8_max_rows or no gemm8: every bucket A of a mixed
step above 256 rows per rank); with ENG_CXX_EMBED_STAGE=1 also FPR_ENGRAM_STAGE's staged
wkv; R <= w8_max_rows with
gemm8: run_wkv's W8A16 GEMM through engram_w8a16_ext's own C ABI (w8a16.gemm, 128-row slices as run_wkv; w8a16.init at
setup), also under engram_stage's wrapper (it passes the W8A16 branch through). Not covered (-1 = Python): any other
run_wkv wrapper, prepare_prefill's dedup (kv_idx), images (mm_emb), inject_input (tests), the c1mega e0 prepare.
"""
from __future__ import annotations

import ctypes as C
import os
import sys

import torch

import mixed_cxx as M

P = C.c_void_p
VERSION = 1
PREP_VERSION = 2                      # pieces_engram.h FPR_ENGRAM_PREP_VERSION (v2: the W8A16 wkv fields)
W8_CHUNK = 32896                      # w8a16_cabi.h W8A16_CHUNK_BYTES
NL, NG, NC = 2, 4, 24
SCALAR = {torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3, torch.int64: 4}
EMBED_STAGE = os.environ.get("ENG_CXX_EMBED_STAGE", "0") == "1"   # (C4c) the embed piece takes FPR_ENGRAM_STAGE's staged wkv


class HashParams(C.Structure):        # engram_cabi.h FprEngramHashParams (filled by the C++ from ctx)
    _fields_ = [("mult", (C.c_int64 * NG) * NL), ("prime", (C.c_int64 * NC) * NL), ("off", (C.c_int64 * NC) * NL),
                ("pad_id", C.c_int32), ("vocab", C.c_int32)]


class HashArgs(C.Structure):
    _fields_ = [("c", M.FprCall), ("hp", P), ("ids", P), ("counts", P), ("s_row0", P), ("s_nrows", P), ("s_start", P),
                ("s_slot", P), ("s_look", P), ("tmap", P), ("ring", P), ("T", C.c_int32), ("out", P), ("keep", P),
                ("err", P)]


class GatherArgs(C.Structure):
    _fields_ = [("c", M.FprCall), ("hash", P), ("T", C.c_int32), ("nl", C.c_int32), ("tabs", P * NL), ("lidx", C.c_int32 * NL),
                ("outs", P * NL), ("lo", (C.c_int64 * NC) * NL), ("hi", (C.c_int64 * NC) * NL), ("ctas", C.c_int32),
                ("threads", C.c_int32), ("unroll", C.c_int32)]


class InjectArgs(C.Structure):
    _fields_ = [("c", M.FprCall), ("T", C.c_int32), ("x_in", P), ("y", P), ("rs", P), ("post", P), ("comb", P), ("kv", P),
                ("kv_stride", C.c_int64), ("qw", P), ("kw", P), ("keep", P), ("eps", C.c_double), ("clampv", C.c_double),
                ("out", P), ("gate_out", P), ("kv_idx", P)]


class Prep(C.Structure):              # pieces_engram.h FprPieceEngramPrep
    _fields_ = [("version", C.c_int32), ("R", C.c_int32), ("resolve", C.c_int32), ("src_dtype", C.c_int32), ("main", P),
                ("est", P), ("ctx", C.c_int64), ("b_start", P), ("S", C.c_int64), ("src", P), ("hp", HashParams),
                ("hash", HashArgs), ("gather", GatherArgs * 2),
                ("wx", P * 2), ("wx_st0", C.c_int64 * 2), ("wx_st1", C.c_int64 * 2),
                ("ww", P * 2), ("ww_st0", C.c_int64 * 2), ("ww_st1", C.c_int64 * 2),
                ("wy", P * 2), ("wy_st0", C.c_int64 * 2), ("wy_st1", C.c_int64 * 2),
                ("K", C.c_int32), ("N", C.c_int32), ("ev_l", P * 2), ("stage_buf", P), ("stage_chunks", C.c_int32),
                ("pad_", C.c_int32),
                ("w8", C.c_int32), ("pad2_", C.c_int32), ("w8_packed", P * 2), ("w8_nchunk", C.c_int64 * 2),   # (v2)
                ("w8_tiles", C.c_int64 * 2), ("w8_rpg", C.c_int64 * 2), ("w8_kxg", C.c_int64 * 2)]


class Embed(C.Structure):             # pieces_engram.h FprPieceEmbed
    _fields_ = [("version", C.c_int32), ("R", C.c_int32), ("has_engram", C.c_int32), ("tok_dtype", C.c_int32), ("main", P),
                ("tok", P), ("ids32", P), ("emb_tab", P), ("emb_rows", C.c_int64), ("emb_st0", C.c_int64), ("emb_tmp", P),
                ("res0", P), ("res0_st0", C.c_int64), ("res0_st1", C.c_int64), ("D", C.c_int32), ("HC", C.c_int32),
                ("prep", Prep)]


class EngramBd(C.Structure):          # pieces_engram.h FprPieceEngramBd
    _fields_ = [("version", C.c_int32), ("layer", C.c_int32), ("l_idx", C.c_int32), ("inject_join", C.c_int32),
                ("first_layer", C.c_int32), ("R", C.c_int32), ("main", P), ("ev_l", P), ("inj", InjectArgs * 2),
                ("bd", C.c_int64)]


def _mod(e):
    return sys.modules[type(e).__module__]


def setup(drv) -> None:
    """Once per driver: the engram table + per-rank init. Without Engram (hooks.engram None): inert."""
    E, x = drv.eng, drv.x
    drv._engram_ok = False
    drv._eg_n = dict(embed=0, engram_bd=0, python=0)
    for r, d in enumerate(E.devs):           # the embed piece's device + events (also without Engram)
        x.embed_init_rank(r, torch.device(d).index)
    if E.hooks.engram is None:
        E.log("mixed_cxx_engram: no Engram: the embed piece runs without prepare; engram_bd stays Python")
        return
    parts = x.sizeof_engram_parts()
    assert x.sizeof_embed() == C.sizeof(Embed) and x.sizeof_engram_bd() == C.sizeof(EngramBd), \
        (x.sizeof_embed(), C.sizeof(Embed), x.sizeof_engram_bd(), C.sizeof(EngramBd))
    assert (parts["prep"], parts["hash"], parts["gather"], parts["inject"], parts["hp"]) == (
        C.sizeof(Prep), C.sizeof(HashArgs), C.sizeof(GatherArgs), C.sizeof(InjectArgs), C.sizeof(HashParams)), parts
    lib = sys.modules.get("engram_phys_ext")
    if lib is None:   # torch.utils.cpp_extension.load never registers its modules in sys.modules: ask
        try:          # EngramPhys's own module (kernels/engram/engram.py) for its cached, already loaded extension
            lib = _mod(E.hooks.engram[0]).ext()
        except Exception as ex:  # noqa: BLE001
            E.log(f"mixed_cxx_engram: Engram extension not found ({type(ex).__name__}: {str(ex)[:120]}): the embed / engram "
                  f"pieces stay Python")
            return
    x.register_table(lib.__file__, "engram")
    for r, d in enumerate(E.devs):
        x.engram_init_rank(r, torch.device(d).index)
        e = E.hooks.engram[r]
        for l, ev in enumerate(e.ev):        # EngramPhys.ev[l]: created at its first record (torch); the C++ records them
            if not ev.cuda_event:
                with torch.cuda.device(d):
                    ev.record(E.estream[r])
    drv._engram_ok = True
    drv._eg_w8 = False                       # (v2) the W8A16 wkv: engram_w8a16_ext's table + w8a16.init on every device NOW
    g8 = getattr(E.hooks.engram[0], "gemm8", None)   # (setup, not lazily on a pool thread: the kernels load here)
    if g8:
        lib8 = getattr(g8[0].ext, "__file__", None)
        why = x.engram_w8_init(lib8) if lib8 else "gemm8 without an extension file"
        if not why:
            rcs = []
            for r, d in enumerate(E.devs):
                with torch.cuda.device(d):
                    rcs.append(x.engram_w8_init_rank(r, torch.device(d).index))
            why = "" if all(rc == 0 for rc in rcs) else f"w8a16.init returned {rcs}"
        drv._eg_w8 = not why
        E.log(f"mixed_cxx_engram: W8A16 wkv (R <= {E.hooks.engram[0].w8_max_rows}) " +
              (f"native ({lib8})" if not why else f"stays Python: {why}"))
        if drv._eg_w8 and W8_SELFTEST:
            _w8_selftest(drv)
    E.log(f"mixed_cxx_engram: embed / engram pieces up ({lib.__file__}; FprPieceEmbed {C.sizeof(Embed)} B, "
          f"FprPieceEngramBd {C.sizeof(EngramBd)} B)")


W8_SELFTEST = os.environ.get("ENG_CXX_EMBED_W8_SELFTEST", "0") == "1"   # the W8A16 wkv vs run_wkv


def _w8_selftest(drv) -> None:
    """(v2, ENG_CXX_EMBED_W8_SELFTEST=1; setup, before any step) the C++ W8A16 wkv (pieces_engram w8_wkv, the code prep_run
    runs) vs EngramPhys.run_wkv's "w8a16" loop (gemm8[l](x[r0:r1], y[r0:r1])) on the Engram stream, every rank and layer,
    T = 1 / 64 / 128 / 129 / 192 / 256 (<= w8_max_rows, <= max_rows): random bf16 x (seeded), y bitwise."""
    E, x = drv.eng, drv.x
    bad = n = 0
    for r, d in enumerate(E.devs):
        e = E.hooks.engram[r]
        with torch.cuda.device(d), torch.cuda.stream(E.estream[r]):
            g = torch.Generator(device=d).manual_seed(1234 + r)
            for l in range(2):
                g8 = e.gemm8[l]
                K, N = int(g8.K), int(g8.N)
                for T in (1, 64, 128, 129, 192, 256):
                    if T > e.w8_max_rows or T > e.max_rows:
                        continue
                    xt = (torch.randn(T, K, device=d, generator=g) * 0.5).to(torch.bfloat16)
                    yp = torch.full((T, N), -7.0, dtype=torch.bfloat16, device=d)
                    yc = yp.clone()
                    for r0 in range(0, T, 128):                    # = run_wkv's "w8a16" branch
                        r1 = min(T, r0 + 128)
                        g8(xt[r0:r1], yp[r0:r1])
                    rc = x.engram_w8_wkv_test(r, xt, yc, g8.packed, int(g8.rpg), int(g8.kxg))
                    torch.cuda.current_stream().synchronize()
                    nd = int((yp.view(torch.int16) != yc.view(torch.int16)).sum()) if rc == 0 else -1
                    n += 1
                    bad += nd != 0
                    E.log(f"mixed_cxx_engram: W8 SELFTEST rank {r} layer {l} T {T}: rc {rc} "
                          f"{'IDENTICAL' if nd == 0 else f'DIFFER {nd} elements'}")
    E.log(f"mixed_cxx_engram: W8 SELFTEST {'PASS' if bad == 0 and n > 0 else 'FAIL'} ({n} cases, {bad} bad)")


def _p(t):
    return M._p(t)


def build_prep(drv, r, s, e):
    """FprPieceEngramPrep for EngramPhys.prepare(R, stream=estream[r]) (fixed-size path, cuBLAS wkv); None = not covered."""
    E = drv.eng
    R = s["R"]
    staged = _staged(e)
    w8 = bool(e.gemm8) and R <= e.w8_max_rows          # (v2) run_wkv's "w8a16" impl
    if R > e.max_rows:
        return None
    if w8:   # engram_stage's wrapper passes the W8A16 branch to the original run_wkv; any other wrapper: Python
        wrap = e.__dict__.get("run_wkv")
        if not getattr(drv, "_eg_w8", False) or (
                wrap is not None and getattr(wrap, "__qualname__", "") != "install.<locals>.run_wkv"):
            return None
    elif e.wbf is None or ("run_wkv" in e.__dict__ and not staged):
        return None
    m = _mod(e)
    dev = torch.device(e.dev).index
    est = E.estream[r].cuda_stream
    p = Prep()
    p.version, p.R = PREP_VERSION, R
    p.main, p.est, p.ctx = P(E.streams[r].cuda_stream), P(est), int(e.hctx)
    src = getattr(e, "_start_src", None)
    if src is not None:
        if src.dtype not in SCALAR or not src.is_contiguous():
            return None
        p.resolve, p.src_dtype = 1, SCALAR[src.dtype]
        p.b_start, p.S, p.src = _p(e.b_start), e.b_start.numel(), _p(src)
    call = M.FprCall(P(est), dev, 0)
    p.hash = HashArgs(call, P(0), _p(e.b_ids), _p(e.b_counts), _p(e.b_row0), _p(e.b_nrows), _p(e.b_start), _p(e.b_slot),
                      _p(e.b_look), _p(e.tmap), _p(e.ring), R, _p(e.hash), _p(e.keep), _p(e.err))
    g = e.gather_cfg
    for l in range(2):
        rec0 = int(e.tables[m.LAYERS[l]].rec0)
        if rec0 % 16:
            return None
        ga = GatherArgs()
        ga.c, ga.hash, ga.T, ga.nl = call, _p(e.hash), R, 1
        ga.tabs[0], ga.lidx[0], ga.outs[0] = P(rec0), l, _p(e.rows[l])
        for c in range(NC):
            ga.lo[0][c], ga.hi[0][c] = e.lo[l * NC + c], e.hi[l * NC + c]
        ga.ctas, ga.threads, ga.unroll = g["ctas"], g["threads"], g["unroll"]
        p.gather[l] = ga
        xv = e.rows[l][:R].view(R, -1)
        y = e.kv[l][:R]
        if w8:   # (v2) gemm8[l](x[r0:r1], y[r0:r1]): the pybind gemm's checks (w8a16.cu gemm) on the same tensors
            g8 = e.gemm8[l]
            pk = g8.packed
            if not (xv.dtype == y.dtype == torch.bfloat16 and xv.stride(1) == 1 and y.stride(1) == 1 and xv.stride(0) % 8 == 0
                    and pk.dtype == torch.uint8 and pk.dim() == 3 and pk.size(2) == W8_CHUNK and pk.is_contiguous()
                    and int(g8.splits) == 1 and y.shape[1] >= pk.size(1) * 256 and pk.device == y.device):
                return None
            p.w8 = 1                            # (v2c) the C++ prep takes the W8A16 branch
            p.w8_packed[l], p.w8_nchunk[l], p.w8_tiles[l] = _p(pk), pk.size(0), pk.size(1)
            p.w8_rpg[l], p.w8_kxg[l] = int(g8.rpg), int(g8.kxg)
            p.wx[l], p.wx_st0[l], p.wx_st1[l] = _p(xv), xv.stride(0), xv.stride(1)
            p.wy[l], p.wy_st0[l], p.wy_st1[l] = _p(y), y.stride(0), y.stride(1)
            p.K, p.N = xv.shape[1], y.shape[1]
            p.ev_l[l] = P(e.ev[l].cuda_event)   # (v2b) the layer's event, as the cuBLAS branch (was skipped by the continue)
            continue
        w = e.wbf[l]
        if not (xv.dtype == w.dtype == y.dtype == torch.bfloat16 and w.shape[1] == xv.shape[1] and y.shape[1] == w.shape[0]):
            return None
        p.wx[l], p.wx_st0[l], p.wx_st1[l] = _p(xv), xv.stride(0), xv.stride(1)
        p.ww[l], p.ww_st0[l], p.ww_st1[l] = _p(w), w.stride(0), w.stride(1)
        p.wy[l], p.wy_st0[l], p.wy_st1[l] = _p(y), y.stride(0), y.stride(1)
        p.K, p.N = xv.shape[1], w.shape[0]
        p.ev_l[l] = P(e.ev[l].cuda_event)
    if staged:
        buf, Cn = e._stage_buf, int(e._stage_chunks)
        if not (buf.dtype == torch.bfloat16 and buf.is_contiguous() and buf.dim() == 2 and buf.shape[1] == p.K
                and buf.shape[0] >= -(-p.N // Cn) and buf.device == e.wbf[0].device):
            return None
        p.stage_buf, p.stage_chunks = _p(buf), Cn
    return p


def _staged(e) -> bool:
    """ENG_CXX_EMBED_STAGE and EngramPhys.run_wkv is engram_stage.install's wrapper (its cuBLAS branch = staged_mm)."""
    return bool(EMBED_STAGE and getattr(e, "_stage_buf", None) is not None and int(getattr(e, "_stage_chunks", 0) or 0) > 0
                and getattr(e.__dict__.get("run_wkv"), "__qualname__", "") == "install.<locals>.run_wkv")


def _prep_consistent(e, R, pr) -> str:
    """"" = the Prep's branch flag matches run_wkv's choice and the branch's
    pointers are set; else the reason. W8A16 (gemm8 and R <= w8_max_rows): w8 == 1, w8_packed / wx / wy non-NULL; cuBLAS:
    w8 == 0, wx / wy non-NULL and ww non-NULL (or, staged, stage_buf with stage_chunks > 0)."""
    want = bool(e.gemm8) and R <= e.w8_max_rows
    if bool(pr.w8) != want:
        return f"w8 flag {int(pr.w8)} but run_wkv's impl is {'w8a16' if want else 'cublas'}"
    for l in range(2):
        if not pr.wx[l] or not pr.wy[l] or not pr.ev_l[l]:
            return f"layer {l}: wx / wy / ev_l NULL"
        if want and not pr.w8_packed[l]:
            return f"layer {l}: w8_packed NULL"
        if not want and not pr.ww[l] and not (pr.stage_buf and pr.stage_chunks > 0):
            return f"layer {l}: ww NULL without a stage buffer"
    return ""


def build_embed(drv, r, s):
    E = drv.eng
    R, tok, res0, emb = s["R"], s["tok"], s["res0"], E.glob[r]["emb"]
    if not (tok.dim() == 1 and tok.numel() == R and tok.dtype in SCALAR and emb.dim() == 2 and emb.stride(1) == 1
            and emb.dtype == torch.bfloat16 and res0.dim() == 3 and res0.shape[0] == R and res0.stride(2) == 1):
        return None
    p = Embed()
    p.version, p.R, p.tok_dtype = VERSION, R, SCALAR[tok.dtype]
    p.main, p.tok = P(E.streams[r].cuda_stream), _p(tok)
    tmp = s.get("_cxx_emb_tmp")
    if tmp is None:
        tmp = s["_cxx_emb_tmp"] = torch.empty(R, emb.shape[1], dtype=emb.dtype, device=res0.device)
    p.emb_tab, p.emb_rows, p.emb_st0, p.emb_tmp = _p(emb), emb.shape[0], emb.stride(0), _p(tmp)
    p.res0, p.res0_st0, p.res0_st1, p.HC, p.D = _p(res0), res0.stride(0), res0.stride(1), res0.shape[1], res0.shape[2]
    if E.hooks.engram is not None:
        e = E.hooks.engram[r]
        pr = build_prep(drv, r, s, e)
        ids = E.hooks.ids32[r]
        if pr is None or ids.dtype != torch.int32 or ids.numel() < R:
            return None
        why = _prep_consistent(e, R, pr)
        if why:                                  # (v3) never hand the C++ a prep whose branch flag and pointers disagree
            n_ = drv.__dict__.setdefault("_eg_prep_bad", 0) + 1
            drv._eg_prep_bad = n_
            if n_ <= 3:
                E.log(f"mixed_cxx_engram: embed prep INCONSISTENT (R {R}, rank {r}): {why} -- the view's embed stays Python")
            return None
        p.has_engram, p.ids32, p.prep = 1, _p(ids), pr
    return bytes(p)


def _embed_ok(drv, s) -> bool:
    E = drv.eng
    if not getattr(drv, "_engram_ok", False) and E.hooks.engram is not None:
        return False
    return not (s.get("inject_input") or s.get("mm_emb") is not None or (E.hooks.engram is not None and E._c1m_e0(s) is not None))


def handle_embed(drv, r, s, L) -> int:
    if s.get("moe") != "prod" or not _embed_ok(drv, s):
        return -1
    E = drv.eng
    e = E.hooks.engram[r] if E.hooks.engram is not None else None
    src = getattr(e, "_start_src", None) if e is not None else None
    key = ("embed", None if src is None else src.data_ptr())
    h = drv._cached(s, key, lambda: build_embed(drv, r, s), drv.x.add_embed)
    if h >= 0 and e is not None:
        e.kv_idx = [None, None]              # as EngramPhys.prepare (the C++ prepare is the fixed-size path)
    return h


def piece_embed(drv, r, s, L) -> bool:
    h = handle_embed(drv, r, s, L)
    if h < 0:
        drv._eg_n["python"] += 1
        return False
    rc = drv.x.embed(r, h)
    assert rc == 0, f"mixed_cxx_engram: embed rank {r}: FPR error {rc}"
    drv._eg_n["embed"] += 1
    drv.n["embed"] += 1
    return True


def build_engram_bd(drv, r, s, L, ignore_kv_idx: bool = False):
    E, EM = drv.eng, drv.EM
    e = E.hooks.engram[r]
    li = EM.ENGRAM_LAYERS.index(L)
    if e.kv_idx[li] is not None and not ignore_kv_idx:
        return None                          # prepare_prefill's dedup rows: Python (ensure_view builds regardless: the
                                             # handle is step-independent; handle_engram_bd still refuses dedup steps)
    zs = (s["zy"], s["X1"], s["zpost"], s["eyec"])
    b = drv.build_bd(r, s, L, 0, engram_sets=[zs + (s["lag0"],), zs + (s["f_pre"],)])
    if not M.PieceBd.from_buffer_copy(b).path:
        return None
    hb = drv._cached(s, ("bd", L, "engram"), lambda: b, drv.x.add_bd)
    if hb < 0:
        return None
    R = s["R"]
    dev = torch.device(s["dev"]).index
    main = P(E.streams[r].cuda_stream)
    p = EngramBd()
    p.version, p.layer, p.l_idx, p.R = VERSION, L, li, R
    p.inject_join = int(bool(EM.BISECT["inject_join"]))
    p.first_layer = int(L == E.layers[0])
    p.main, p.ev_l, p.bd = main, P(e.ev[li].cuda_event), hb
    sets = [(s["x0"], s["res0"], s["post0"], s["comb0"]), (s["moe_out"], s["xm2"], s["f_post"], s["f_comb"])]
    for i, (y, rs, post, comb) in enumerate(sets):
        p.inj[i] = InjectArgs(M.FprCall(main, dev, 0), R, P(0), _p(y), _p(rs), _p(post), _p(comb), _p(e.kv[li]),
                              e.kv[li].stride(0), _p(e.q[li]), _p(e.k[li]), _p(e.keep), float(e.eps), float(e.clamp),
                              _p(s["X1"]), P(0), P(0))
    return bytes(p)


def handle_engram_bd(drv, r, s, L) -> int:
    E, EM = drv.eng, drv.EM
    if (s.get("moe") != "prod" or not getattr(drv, "_engram_ok", False) or L not in EM.ENGRAM_LAYERS
            or E.hooks.engram is None or E.hooks.engram[r].kv_idx[EM.ENGRAM_LAYERS.index(L)] is not None):
        return -1                            # (kv_idx: prepare_prefill's dedup rows, checked every call)
    return drv._cached(s, ("engram_bd", L), lambda: build_engram_bd(drv, r, s, L), drv.x.add_engram_bd)


def piece_engram_bd(drv, r, s, L) -> bool:
    h = handle_engram_bd(drv, r, s, L)
    if h < 0:
        return False
    drv._py_mix(s)
    drv.x.set_chain_input(r, int(bool(s.get("chain_input"))))   # engram_bd reads StepParams.chain_input
    rc = drv.x.engram_bd(r, h, L)
    assert rc == 0, f"mixed_cxx_engram: engram_bd rank {r} layer {L}: FPR error {rc}"
    drv._eg_n["engram_bd"] += 1
    drv.n["engram_bd"] += 1
    return True
