"""mixed_cxx_attn: the Python side of the attention pieces of the C++ mixed-step driver
Loaded by mixed_cxx.Driver.__init__: setup(drv) once, step_params(drv, r, s, sp) in every
set_step, piece_<kind>(drv, r, s, L) -> bool at the Python piece's call site (False: the Python piece runs).

Stage 1b-i: piece_<kind> builds (cached) and runs the piece now. Stage 1d (ENG_MIXED_CXX_RUN=1): handle_<kind> only
builds (cached) and returns the static handle (or -1 = the whole Python p_attn); the driver threads run attn_a_c /
attn_b_c (fpr_drv_register_piece) in the span. Building never launches on the rank stream (1d: the previous span may be
running there): the DeepGEMM warm-up uses scratch outputs of the same layout (keys hold no pointers) on a private stream.

Pieces (ENG_MIXED_CXX_PIECES=...,attn_a,...):
  attn_a   attn.attn_a's mixed branch: cuBLAS q_a | wkv (ATen at::mm_out) + attn_dense.norm_rope2
  attn_b   attn.attn_b's no-pf DENSE8 branch: w8a16 q_b + RoPE + insert, FlashMLA sparse decode (dispatcher; plans per
           (segment, key) in pieces_attn's rank state), ropeq64 + DeepGEMM einsum (wo_a), act_quant_128 + DeepGEMM gemm (wo_b)
  attn_pf  (stage 1c-1, opt-in on top of attn_b) bucket A with ENG_ATTN_PF: attn_b's FlashMLA decode on the decode rows
           (c["sched_dec"]), attn_prefill.attend for the prefill rows (gatherk gather, Triton-replayed combine, FlashMLA
           sparse prefill; the Plan's reuse rules mirrored), padding rows zeroed; then the same DENSE8 wo
Each piece mirrors the Python function's branch selection exactly and only takes the branch the served config uses in
mixed buckets; anything else returns False before building.
"""
from __future__ import annotations

import ctypes as C
import functools
import sys
import os
import time

import torch

P = C.c_void_p
I64 = C.c_int64
I32 = C.c_int32


class FprCall(C.Structure):
    _fields_ = [("stream", P), ("device", C.c_int), ("rank", C.c_int)]


class NormRope2Args(C.Structure):   # dense_cabi.h FprAttnDenseNormRope2Args
    _fields_ = [("call", FprCall), ("xq", P), ("ldxq", I64), ("wq", P), ("oq", P), ("ldoq", I64), ("dq", I64), ("xk", P),
                ("ldxk", I64), ("wk", P), ("ok", P), ("ldok", I64), ("dk", I64), ("rows", I64), ("eps", C.c_float),
                ("pos", P), ("cos_sin", P)]


class PieceAttnA(C.Structure):      # pieces_attn.h FprPieceAttnA
    _fields_ = [("version", I32), ("rank", I32), ("layer", I32), ("device", I32), ("R", I64), ("main", P), ("a_rows", P),
                ("a_ld", I64), ("wqkv", P), ("n", I64), ("k", I64), ("qakv", P), ("nr2", NormRope2Args)]


class AttnView(C.Structure):        # pieces_attn.h FprAttnView
    _fields_ = [("ptr", P), ("sizes", I64 * 4), ("strides", I64 * 4), ("ndim", I32), ("dtype", I32)]


class W8GemmArgs(C.Structure):      # w8a16_cabi.h FprW8a16GemmArgs
    _fields_ = [("call", FprCall), ("x", P), ("ntok", I64), ("kx", I64), ("x_ld", I64), ("packed", P), ("nchunk", I64),
                ("tiles", I64), ("y", P), ("ldy", I64), ("y_cols", I64), ("rows_per_group", I64), ("kx_per_group", I64),
                ("splits", I64), ("pos", P), ("cs", P), ("kv", P), ("slot", P), ("cache", P), ("page_stride", I64),
                ("shift", I64), ("inv_pos", P), ("ws", P), ("ws_floats", I64), ("cnt", P), ("cnt_n", I64), ("prof", P)]


class RopeQArgs(C.Structure):       # ropeq64_cabi.h
    _fields_ = [("call", FprCall), ("o", P), ("o_stride_t", I64), ("o_stride_h", I64), ("positions", P), ("cos_sin", P),
                ("cs_stride", I64), ("t", I64), ("n_groups", I64), ("out", P), ("scales", P)]


class DgView(C.Structure):          # dg_cabi.h FprDgView
    _fields_ = [("ptr", P), ("sizes", I64 * 4), ("strides", I64 * 4), ("ndim", I32), ("dtype", I32)]


class DgArgs(C.Structure):          # dg_cabi.h FprDgGemmArgs / FprDgEinsumArgs (same layout)
    _fields_ = [("call", FprCall), ("a", DgView), ("sfa", DgView), ("b", DgView), ("sfb", DgView), ("d", DgView),
                ("num_sms", I64)]


class ActQArgs(C.Structure):        # fp8_linear_cabi.h
    _fields_ = [("call", FprCall), ("x", P), ("ldx", I64), ("m", I64), ("k", I64), ("kp", I64), ("ue8m0", I32), ("q", P),
                ("scales", P)]


class InvRopeArgs(C.Structure):     # dense_cabi.h FprAttnDenseInvRopeArgs
    _fields_ = [("call", FprCall), ("o", P), ("pos", P), ("cos_sin", P), ("t", I64), ("nh", I64)]


class RopeQInsertArgs(C.Structure):  # dense_cabi.h FprAttnDenseRopeQInsertArgs
    _fields_ = [("call", FprCall), ("q", P), ("kv", P), ("pos", P), ("cos_sin", P), ("slot_mapping", P), ("cache", P),
                ("page_stride", I64), ("shift", I64), ("t", I64), ("nh", I64), ("do_insert", I32)]


class PieceAttnB(C.Structure):      # pieces_attn.h FprPieceAttnB
    _fields_ = [("version", I32), ("rank", I32), ("layer", I32), ("device", I32), ("R", I64), ("main", P), ("qb", W8GemmArgs),
                ("plan_key", I32), ("comp", I32), ("fq", AttnView), ("fk_swa", AttnView), ("fsi", AttnView),
                ("fsl", AttnView), ("fsink", AttnView), ("fk_comp", AttnView), ("fti", AttnView), ("ftl", AttnView),
                ("fo", AttnView), ("scale", C.c_double), ("rq", RopeQArgs), ("woa", DgArgs), ("aq", ActQArgs),
                ("wob", DgArgs), ("pf", I32), ("ratio", I32), ("src", I32), ("isrc", I32), ("pf_q", P), ("pf_o", P),
                ("pf_kswa", P), ("pf_kswa_ps", I64), ("pf_kcomp", P), ("pf_kcomp_ps", I64), ("pf_ti", P), ("pf_sink", P),
                ("wo", I32), ("pad_wo", I32), ("ir", InvRopeArgs), ("woa8", W8GemmArgs), ("wob8", W8GemmArgs),
                ("qdq", I32), ("pad_qdq", I32), ("dq_q", I64), ("dq_a", I64), ("dq_b", I64), ("slot_q", P), ("slot_a", P),
                ("slot_b", P), ("q_n", I64), ("q_k", I64), ("a_n", I64), ("a_k", I64), ("b_n", I64), ("b_k", I64),
                ("wo_o", P), ("wo_z", P), ("wo_y", P), ("z_ld", I64), ("y_ld", I64), ("rqi", RopeQInsertArgs)]


VERSION = 1
_AV = {torch.uint8: 1, torch.int32: 2, torch.float32: 3, torch.bfloat16: 4, torch.float8_e4m3fn: 5, torch.int64: 6}
_DG = {torch.float8_e4m3fn: 1, torch.float32: 2, torch.bfloat16: 3}


_WHY_SEEN: set = set()


def _why(drv, kind: str, reason: str, r=None, L=None) -> int:
    """why a piece stays Python: counts why_<kind> (drv.n: the harness' driver counters)
    and logs the first occurrence of each (kind, reason) (<= 64 distinct pairs). Returns -1 (a refused handle). Logging
    only: no decision depends on it."""
    drv.n[f"why_{kind}"] += 1
    if (kind, reason) not in _WHY_SEEN and len(_WHY_SEEN) < 64:
        _WHY_SEEN.add((kind, reason))
        drv.eng.log(f"mixed_cxx_attn: {kind} stays Python: {reason} (first at rank {r} layer {L})")
    return -1


def _view(t, cls=None, codes=None):
    v = (cls or AttnView)()
    if t is None:
        return v
    v.ptr = t.data_ptr()
    assert t.dim() <= 4, t.shape
    for i in range(t.dim()):
        v.sizes[i], v.strides[i] = t.size(i), t.stride(i)
    v.ndim = t.dim()
    v.dtype = (codes or _AV)[t.dtype]
    return v


def _p(t) -> P:
    return P(t.data_ptr()) if t is not None and t.numel() else P(0)


def _EA():
    import sys
    return sys.modules["engine.attn"] if "engine.attn" in sys.modules else __import__("attn")


def setup(drv) -> None:
    """Register the attention extensions' tables and resolve their functions per rank (Python thread)."""
    E = drv.eng
    drv._attn_ok = False
    drv._attn_b_ok = False
    kinds = {k for k in os.environ.get("ENG_MIXED_CXX_PIECES", "").split(",") if k}
    if not kinds & {"attn_a", "attn_b", "idx"}:   # inert unless an attention kind is enabled (no tables, no builds)
        return
    try:
        EA = _EA()
        kd = EA.AH._ext()
        x = drv.x
        if not hasattr(x, "add_attn_a"):
            E.log("mixed_cxx_attn: pieces_attn not compiled into mixed_cxx: attention pieces stay Python")
            return
        assert x.sizeof_attn_a() == C.sizeof(PieceAttnA), (x.sizeof_attn_a(), C.sizeof(PieceAttnA))
        import vllm.third_party.flashmla.flash_mla_interface  # noqa: F401  (registers _flashmla_C's dispatcher ops)
        x.register_table(kd.__file__, "attn_dense")
        _register_attn_b(drv)
        if "attn_pf" in kinds:
            try:
                import gather_k as GK                      # (kernels/attn_dense, on sys.path as attn_act8)
                x.register_table(GK.build().__file__, "gatherk")
            except Exception as e:
                E.log(f"mixed_cxx_attn: gatherk not available ({e}): bucket A prefill attention stays Python")
        idx_reg = bool(_register_idx(drv))
        for r, d in enumerate(E.devs):
            with torch.cuda.device(d):
                x.attn_init_rank(r, torch.device(d).index)
    except Exception as e:   # (an extension without its C ABI table: an older build) -> Python pieces
        E.log(f"mixed_cxx_attn: setup failed ({e}): attention pieces stay Python")
        return
    drv._attn_ok = True
    drv._attn_pf_ok = False
    if "attn_pf" in kinds:
        try:
            PS = _pf_step_type_cached()
            want = [PS.lay.offset, PS.rec.offset, PS.ws_ratio.offset, C.sizeof(PfGroup), C.sizeof(PfLayout)]
            drv._attn_pf_ok = (bool(x.attn_pf_ready()) and x.sizeof_pf_step() == C.sizeof(PS)
                               and list(x.offsets_pf_step()) == want)
            if not drv._attn_pf_ok:
                E.log(f"mixed_cxx_attn: attn_pf not ready (ready {x.attn_pf_ready()}, FprAttnPfStep {x.sizeof_pf_step()} vs "
                      f"{C.sizeof(PS)}, offsets {list(x.offsets_pf_step())} vs {want}): bucket A attention stays Python")
        except Exception as e:
            E.log(f"mixed_cxx_attn: attn_pf not ready ({e}): bucket A attention stays Python")
        if drv._attn_pf_ok and os.environ.get("ENG_COMB_INIT", "1") == "1":   # every combine key class, once per device
            try:
                nc_ = sum(record_comb_classes(drv, r_) for r_ in range(len(E.devs)))
                drv.n["comb_classes"] += nc_
                E.log(f"mixed_cxx_attn: combine key classes recorded at init: {nc_} over {len(E.devs)} devices")
            except Exception as e:
                E.log(f"mixed_cxx_attn: combine key classes not recorded at init ({e}): first-meet recording")
    try:
        drv._attn_idx_ok = idx_reg and bool(x.attn_idx_ready()) and x.sizeof_idx_step() == C.sizeof(IdxStep)
        if idx_reg and not drv._attn_idx_ok:
            E.log(f"mixed_cxx_attn: idx not ready (tables registered, attn_idx_ready {x.attn_idx_ready()}, FprAttnIdxStep "
                  f"{x.sizeof_idx_step()} vs {C.sizeof(IdxStep)}): idx stays Python")
    except Exception as e:
        E.log(f"mixed_cxx_attn: idx not ready ({e}): idx stays Python")
        drv._attn_idx_ok = False
    drv._idxa_ok = drv._idxa_pf4 = False   # (Stage B) A-role index layers: compressor + scoring
    if getattr(drv, "_attn_idx_ok", False) and os.environ.get("ENG_MIXED_CXX_IDXA", "1") == "1" and hasattr(x, "add_idx_a"):
        try:
            PT, ST = _piece_idxa_type(), IdxAStep
            want = [_kc_record_size(), PT.version.offset, PT.fscn.offset, PT.fscn_vals.offset, PT.rqi.offset, PT.iks.offset,
                    PT.wp.offset, PT.gt.offset, PT.gt_vals.offset, PT.vrow.offset, C.sizeof(_tr_record_type()), PT.pad3.offset,
                    PT.bs.offset, PT.sc_vals.offset, PT.bs_i.offset, PT.topk_ws.offset]
            wst = [ST.skip.offset, ST.max_len.offset, ST.nsl.offset, ST.sl.offset, ST.gst.offset, C.sizeof(IdxASlice),
                   ST.global_rows.offset, ST.max_seq_len.offset]
            drv._idxa_ok = (bool(x.idx_a_ready()) and x.sizeof_idx_a() == C.sizeof(PT) and list(x.offsets_idx_a()) == want
                            and x.sizeof_idxa_step() == C.sizeof(ST) and list(x.offsets_idxa_step()) == wst)
            drv._idxa_pf4 = bool(x.idx_a_pf4())
            if not drv._idxa_ok:
                E.log(f"mixed_cxx_attn: idx_a not ready (ready {x.idx_a_ready()}, FprPieceIdxA {x.sizeof_idx_a()} vs "
                      f"{C.sizeof(PT)}, offsets {list(x.offsets_idx_a())} vs {want}, step {x.sizeof_idxa_step()} vs "
                      f"{C.sizeof(ST)} {list(x.offsets_idxa_step())} vs {wst}): A-role index layers stay Python")
        except Exception as e:
            E.log(f"mixed_cxx_attn: idx_a not ready ({e}): A-role index layers stay Python")
            drv._idxa_ok = False
    try:   # DENSE8 handles need attn_b_ready(); the ENG_WO_W8A16 (wo8) variant needs attn_b_ready_wo8() only
        drv._attn_b_d8_ok = bool(drv._attn_b_tables) and bool(x.attn_b_ready())
        drv._attn_b_wo8_ok = bool(drv._attn_b_tables) and bool(getattr(x, "attn_b_ready_wo8", lambda: False)())
        drv._attn_b_ok = drv._attn_b_d8_ok or drv._attn_b_wo8_ok
        drv._attn_b_dq_ok = (drv._attn_b_wo8_ok and bool(getattr(x, "attn_b_ready_dq", lambda: False)())
                             and bool(getattr(drv.EM, "_DQ_DEV", None))
                             and os.environ.get("ENG_MIXED_CXX_ATTN_DQ", "1") == "1")
    except Exception as e:
        E.log(f"mixed_cxx_attn: attn_b not ready ({e}): attn_b stays Python")
        drv._attn_b_ok = False
    drv._attn_n = dict(attn_a=0, attn_a_py=0, attn_b=0, attn_b_py=0)
    if IDXA_STOCK and drv._idxa_ok and os.environ.get("ENG_MIXED_CXX_STOCK_WARM", "1") == "1":
        _stock_kwarm(drv)
    if os.environ.get("ENG_MIXED_CXX_OP_WARM", "1") == "1":
        _op_kwarm(drv)
    if os.environ.get("ENG_MIXED_CXX_FMLA_WARM", "1") == "1":
        _fmla_kwarm(drv)
    E.log(f"mixed_cxx_attn: attention pieces up (attn_dense {kd.__file__}; FprPieceAttnA {C.sizeof(PieceAttnA)} B)")


def _fmla_kwarm(drv) -> None:
    """(cp1e: rank-thread loads of sm90::prefill::sparse_fwd::KernelTemplate<512, true> from the attn_pf piece) every kernel of
    the FlashMLA extension (vllm/_flashmla_C, imported by setup above; outside the release root, so mixed_cxx_ltwarm's
    kwarm_all.warm_loaded does not see it) on every device, via kwarm_all (nm handles + cudaFuncGetAttributes, calling thread).
    ENG_MIXED_CXX_FMLA_WARM=0 skips."""
    E = drv.eng
    try:
        import kwarm_all as KA
        import vllm
        sos = [p for p in KA.loaded(os.path.dirname(os.path.realpath(vllm.__file__)))
               if os.path.basename(p).startswith("_flashmla")]   # (mapped by flash_mla_interface's import: never dlopened here)
        t0 = time.time()
        res = KA.warm(sos, [torch.device(d).index for d in E.devs])
        E.log(f"mixed_cxx_attn: FlashMLA kernel warm: {res} in {time.time() - t0:.2f} s")
    except Exception as e:
        E.log(f"mixed_cxx_attn: FlashMLA kernel warm failed ({e}): first launches stay lazy")


def _op_kwarm(drv) -> None:
    """the ATen ops pieces_attn issues on the driver's
    RANK threads, on the Python thread at setup, per device on its main stream, synced: attn_b_attend's bf16 zero_, the idx /
    idx_a pieces' int32 fill_(-1) / zero_ / index_select + index_copy_ (2-D [R, 512] and 1-D), and the CP exchanges (cp_send:
    uint8 rows index_select(0, int64) -> slice(1, o, o + n).contiguous(); cp_recv: uint8 slice(1, o, o + n).index_put_(
    (int64,), rows)), each with aligned and odd column offsets / widths (torch picks vectorized vs unrolled kernels by
    alignment and contiguity). ENG_MIXED_CXX_OP_WARM=0 skips."""
    E = drv.eng
    t0 = time.time()
    n = 0
    try:
        for r, d in enumerate(E.devs):
            dev = torch.device(d)
            with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
                i64 = dict(dtype=torch.int64, device=dev)
                torch.zeros(8, 64 * 512, dtype=torch.bfloat16, device=dev)[3:].zero_()
                for R in (64, 4096):
                    ti = torch.empty(R, 512, dtype=torch.int32, device=dev)
                    tl = torch.empty(R, dtype=torch.int32, device=dev)
                    ti.view(-1).fill_(-1)
                    tl.zero_()
                    ir = torch.arange(R // 2, **i64)
                    ti.index_copy_(0, ir, torch.zeros(R, 512, dtype=torch.int32, device=dev).index_select(0, ir))
                    tl.index_copy_(0, ir, torch.zeros(R, dtype=torch.int32, device=dev).index_select(0, ir))
                    n += 6
                for rows, ld in ((256, 584), (1024, 4096), (4096, 1168), (1024, 37376)):
                    pg = torch.zeros(rows, ld, dtype=torch.uint8, device=dev)
                    # (cp1e) index_select picks its small-index kernel for <= 16 indices, the large-index one (major / minor
                    # by shape) above: every message size class of the CP exchanges
                    for ni in (1, 2, 8, 16, 17, 64, rows // 3):
                        src = torch.arange(0, rows, max(1, rows // ni), **i64)[:ni]
                        for o_, w_ in ((0, ld), (0, 512), (64, 448), (3, 291)):
                            if o_ + w_ > ld:
                                continue
                            data = pg.index_select(0, src)[:, o_:o_ + w_].contiguous()
                            pg[:, o_:o_ + w_].index_put_((src,), data)
                            n += 3
            torch.cuda.synchronize(dev)
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        drv.n["op_kwarm"] += n
        E.log(f"mixed_cxx_attn: op warm: {n} ATen calls on {len(E.devs)} devices in {time.time() - t0:.2f} s "
              f"(zero_ / fill_ / index_select / index_copy_ / contiguous / index_put_)")
    except Exception as e:
        E.log(f"mixed_cxx_attn: op warm failed ({e}): first launches stay lazy")


def _stock_kwarm(drv) -> None:
    """Here, at setup on
    the Python thread, before any step: the same ops with the same argument stack on every device, over the regimes the
    branch can reach (cooperative <= 64 rows, persistent above; widths / max_seq_len from 16K to past LONG_ROW), so every
    kernel those calls can pick is loaded. Scratch only; synced. ENG_MIXED_CXX_STOCK_WARM=0 skips."""
    E = drv.eng
    t0 = time.time()
    try:
        try:
            from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE as RW
        except Exception:
            RW = 1024 * 1024
        op_p, op_c = torch.ops._C.persistent_topk, torch.ops._C.cooperative_topk
        n = 0
        for r, d in enumerate(E.devs):
            dev = torch.device(d)
            with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
                ws = torch.empty(RW, dtype=torch.uint8, device=dev)
                g = torch.Generator(device=dev).manual_seed(1234)
                for rows in (4, 68):
                    for w in (16384, 65536, 270336, 300032):
                        lg = torch.randn(rows, w, device=dev, dtype=torch.float32, generator=g)
                        sl = torch.full((rows // 4, 4), w, dtype=torch.int32, device=dev)
                        out = torch.empty(rows, 512, dtype=torch.int32, device=dev)
                        (op_c if rows <= 64 else op_p)(lg, sl, out, ws, 512, w)
                        n += 1
                        if rows == 4:          # the other op at the same width too (coop's rule also needs global_rows <= 64)
                            op_p(lg, sl, out, ws, 512, w)
                            n += 1
                        nb = (w + 7) // 8      # layer 20: at::topk(scores [rows, nb], min(2048, nb), -1, largest, sorted)
                        sc = torch.randn(rows, nb, device=dev, dtype=torch.float32, generator=g)
                        torch.topk(sc, min(2048, nb), -1, True, True)
                        big = torch.iinfo(torch.int32).max   # canon_topk: where / full_like / sort(1) / copy_
                        key = torch.where(out >= 0, out, torch.full_like(out, big))
                        srt = torch.sort(key, 1)[0]
                        out.copy_(torch.where(srt == big, torch.full_like(srt, -1), srt))
                        n += 2
            torch.cuda.synchronize(dev)
            del ws, g, lg, sl, out, sc, key, srt    # the ~80 MB warm scratch back to the device, not the cache
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        drv.n["stock_kwarm"] += n
        E.log(f"mixed_cxx_attn: stock top-k warm: {n} calls on {len(E.devs)} devices in {time.time() - t0:.2f} s "
              f"(persistent / cooperative_topk, topk, canon sort)")
    except Exception as e:
        E.log(f"mixed_cxx_attn: stock top-k warm failed ({e}): first launches stay lazy")


def _register_attn_b(drv) -> None:
    """attn_b's tables: w8a16 (the served q_b build), ropeq64, ds41h_fp8_linear, the DeepGEMM wrapper (dg_cabi)."""
    E, x = drv.eng, drv.x
    drv._attn_b_tables = False
    try:
        import attn_act8 as AA
        import dg_cabi as DGC
        from ds41h import linear as DL
        g = None
        for r in range(len(E.devs)):
            for L in E.layers:
                w8 = (E.lw[r][L].get("AW") or {}).get("w8") or {}
                if "wq_b" in w8:
                    g = w8["wq_b"]
                    break
            if g is not None:
                break
        if g is None:
            return
        x.register_table(g.ext.__file__, "w8a16")
        x.register_table(AA.ropeq().__file__, "ropeq64")
        x.register_table(DL.ext().__file__, "fp8_linear")
        drv._dg = DGC.build()
        x.register_table(drv._dg.__file__, "dg")
        drv._attn_b_tables = True
    except Exception as e:
        E.log(f"mixed_cxx_attn: attn_b tables not available ({e}): attn_b stays Python")


def _register_idx(drv) -> bool:
    """The idx piece's tables (idx2f_topk / fcand8 / cand: the served builds the hook loads)."""
    import idx2f as X
    try:
        for name, f in (("idx2f_topk", X.ext_topk), ("idx2f_fcand8", X.ext_fcand8), ("idx2f_cand", X.ext_cand)):
            drv.x.register_table(f().__file__, name)
    except Exception as e:
        drv.eng.log(f"mixed_cxx_attn: idx tables not available ({e}): idx stays Python")
        return False
    try:   # (Stage B, optional) the shared-key prefill logits of IDX2F_PF slices
        if getattr(X, "PF_EXT", "") == "idx2f_pf4" and hasattr(X, "ext_pf"):
            drv.x.register_table(X.ext_pf().__file__, "idx2f_pf4")
        return True
    except Exception as e:
        drv.eng.log(f"mixed_cxx_attn: idx tables not available ({e}): idx stays Python")
        return False


def step_params(drv, r, s, sp) -> None:
    """A bucket segment starts: its FlashMLA plans start over (_new_scheds makes fresh plans every mixed step); the idx
    piece's per-step tables (FprAttnIdxStep, pieces_attn's own state) for T-role buckets."""
    if getattr(drv, "_attn_b_ok", False):
        slots = list((s.get("_cxx_plan_ids") or {}).values())
        if slots:
            drv.x.attn_reset_plans(r, slots)
    if getattr(drv, "_attn_pf_ok", False):
        st = None
        if s.get("pf") is not None and _role(drv, s) == "A":
            try:
                st = _pf_step(drv, r, s)
            except Exception as e:
                drv.eng.log(f"mixed_cxx_attn: attn_pf rank {r}: step state failed ({e}): Python attends")
                st = None
        s["_cxx_pf_ok"] = st is not None
        if st is not None:
            drv.x.attn_set_pf_step(r, bytes(st))
        else:
            drv.x.attn_pf_clear(r)
    if getattr(drv, "_attn_idx_ok", False):
        st = _idx_step(drv, r, s) if _idx_ok(drv, s) else None
        s["_cxx_idx_step_ok"] = st is not None
        if st is not None:
            drv.x.attn_set_idx_step(r, bytes(st))
    if getattr(drv, "_idxa_ok", False) and _role(drv, s) == "A":
        try:
            st = _idxa_step(drv, r, s) if _idxa_ok(drv, s) else None
        except Exception as e:
            drv.eng.log(f"mixed_cxx_attn: idx_a rank {r}: step values failed ({e}): Python")
            st = None
        s["_cxx_idxa_step"] = st
        if st is not None:
            drv.x.attn_set_idxa_step(r, bytes(st))
        else:
            drv.x.attn_clear_idxa_step(r)


def _attn_a_why(EA, W, ws, R) -> str:
    """attn.attn_a's branches: "" only for the plain mixed branch (torch.mm + norm_rope2); else why not."""
    d8 = W.get("A8c")
    if d8 is not None and "qkv" in d8 and EA.d8_rows(ws):
        return "DENSE8_ADD qkv rows"                    # ENG_C128_DENSE8_ADD qkv
    if EA._a8_rows(W, ws, "qkv") is not None:
        return "PF_ACT8_DENSE qkv rows"                 # ENG_PF_ACT8_DENSE
    pk = W.get("pk8")
    if ws.get("pf_ab") is None and pk is not None and "wqkv_a" in pk and R <= 32 and ws.get("valid") is not None:
        return "skw8_dense (R <= 32)"                   # skw8_dense would run
    return ""


def _attn_a_eligible(EA, W, ws, R) -> bool:
    return not _attn_a_why(EA, W, ws, R)



_PRIV: dict = {}   # (device index, tag) -> a NON-POOLED torch.cuda.ExternalStream


def _priv_stream(dev, tag: str = "build"):
    """a stream no other code can be handed: torch.cuda.Stream() draws round-robin from a
    32-stream pool per device and priority, which the engine's side / Engram / AGRS comm streams also come from, so a
    "fresh" builder stream aliased an engine stream after ~32 creations (its waits / syncs then covered unrelated work).
    cudaStreamCreateWithFlags(cudaStreamNonBlocking) once per (device, tag), wrapped as an ExternalStream; never destroyed
    (process lifetime)."""
    import ctypes as _C
    d = torch.device(dev)
    d = torch.device("cuda", d.index if d.index is not None else torch.cuda.current_device())
    k = (d.index, tag)
    s = _PRIV.get(k)
    if s is None:
        rt = None
        for ln in open("/proc/self/maps"):
            p = ln.split()[-1]
            if "libcudart.so" in p:
                rt = _C.CDLL(p)
                break
        if rt is None:
            raise RuntimeError("_priv_stream: no libcudart mapped")
        h = _C.c_void_p()
        prev = _C.c_int()
        rt.cudaGetDevice(_C.byref(prev))
        rt.cudaSetDevice(_C.c_int(d.index))
        try:
            rc = rt.cudaStreamCreateWithFlags(_C.byref(h), _C.c_uint(1))   # cudaStreamNonBlocking
        finally:
            rt.cudaSetDevice(prev)
        if rc != 0 or not h.value:
            raise RuntimeError(f"_priv_stream: cudaStreamCreateWithFlags rc {rc}")
        s = _PRIV[k] = torch.cuda.ExternalStream(h.value, device=d)
    return s

def _build_sync(drv, r, ps=None) -> None:
    """called at every exit of a handle build
    BEFORE the builder's scratch is released. ENG_MIXED_CXX_ATTN_BUILD_SYNC=1 (its own knob; ENG_MIXED_CXX_BUILD_SYNC is mixed_cxx_ensure's _sync_all) =1 (default): a host wait on the builder's OWN private
    stream ps only (every builder launches its warm-ups / recordings / kernel_capture warms on ps; kernel_capture joins its
    pooled stream into the caller's current stream = ps; ps never waits on the rank's main stream), so the scratch is idle
    before it is freed, and the wait can never sit behind a step's cross-rank spin / exchange kernels queued on the device
    (a torch.cuda.synchronize(dev) inside ensure during a CP step can: rank r's thread blocks while device r waits for a peer
    launch this thread has not issued). Safe in-step too, so every build. A build that exits before creating ps launched
    nothing. =2: the old device-wide synchronize, ensure() builds only (diag); =0: off."""
    # =1 (default): ensure() builds a device-wide synchronize again
    # (the idx_a build path has async work outside ps, not yet found), in-step builds the ps wait (no device sync during a
    # step: the deadlock shape); =3: ps wait everywhere (the cp1f configuration, diag); =2: device sync at ensure only; =0 off.
    m = os.environ.get("ENG_MIXED_CXX_ATTN_BUILD_SYNC", "3")
    ens = getattr(drv, "_cxx_ensuring", False)
    if m in ("1", "2") and ens:
        torch.cuda.synchronize(torch.device(drv.eng.devs[r]))
        drv.n["build_sync_dev"] += 1
    elif m in ("1", "3") and ps is not None:
        ps.synchronize()
        drv.n["build_sync"] += 1


def build_piece_attn_a(drv, r, s, L) -> bytes:
    E = drv.eng
    AW, ws = E.lw[r][L]["AW"], s["ws"]
    R = int(ws["R"])
    a = s["a"]
    w = AW["wqkv_a"]
    q = ws["qakv"]
    assert a.dim() == 2 and a.stride(1) == 1 and a.dtype == torch.bfloat16 and w.is_contiguous() and q.is_contiguous()
    N, K = int(w.shape[0]), int(w.shape[1])
    assert q.shape == (R, N) and a.shape[1] == K and N == 1792, (q.shape, a.shape, w.shape)
    dev = torch.device(E.devs[r]).index
    st = PieceAttnA()
    st.version, st.rank, st.layer, st.device, st.R = VERSION, r, L, dev, R
    st.main = P(E.streams[r].cuda_stream)
    st.a_rows, st.a_ld = _p(a), a.stride(0)
    st.wqkv, st.n, st.k = _p(w), N, K
    st.qakv = _p(q)
    nr = st.nr2
    nr.call = FprCall(st.main, dev, r)
    pos = s["pos"]
    assert pos.dtype == torch.int64 and AW["cos_sin"].dtype == torch.float32
    nr.xq, nr.ldxq, nr.wq, nr.oq, nr.ldoq, nr.dq = _p(q), N, _p(AW["q_norm"]), _p(ws["qr"]), ws["qr"].stride(0), 1280
    nr.xk, nr.ldxk = P(q.data_ptr() + 1280 * q.element_size()), N
    nr.wk, nr.ok, nr.ldok, nr.dk = _p(AW["kv_norm"]), _p(ws["kv"]), ws["kv"].stride(0), N - 1280
    nr.rows, nr.eps, nr.pos, nr.cos_sin = R, float(AW["eps"]), _p(pos), _p(AW["cos_sin"])
    _build_sync(drv, r, locals().get("ps"))
    return bytes(st)


def piece_attn_a(drv, r, s, L) -> bool:
    if not getattr(drv, "_attn_ok", False):
        return False
    E = drv.eng
    EA = _EA()
    AW, ws = E.lw[r][L]["AW"], s["ws"]
    if not _attn_a_eligible(EA, AW, ws, int(ws["R"])):
        drv._attn_n["attn_a_py"] += 1
        return False
    cache = s.setdefault("_cxx", {})
    key = ("attn_a", r, L)
    h = cache.get(key)
    if h is None:
        h = cache[key] = drv.x.add_attn_a(build_piece_attn_a(drv, r, s, L))
    rc = drv.x.attn_a(r, h, L)
    assert rc == 0, f"mixed_cxx_attn: attn_a rank {r} layer {L}: FPR error {rc}"
    drv._attn_n["attn_a"] += 1
    if drv._attn_n["attn_a"] == 1:
        E.log(f"mixed_cxx_attn: first C++ attn_a piece: rank {r} layer {L} R {ws['R']}")
    return True


# ------------------------------------------------------------------------------------------------------------ attn_b
def _wo_mode(drv, EA, W, ws, s, c, R, spans: bool = True) -> int:
    """attn_b_attend takes its `full` branch (kd.inv_rope, then w8["wo_a"] / w8["wo_b"] =
    engine._WoW8; the inverse RoPE runs unless ENG_C128A's fused branch applies, refused in _attn_b_eligible).
    1 = both _WoW8 calls take their w8a16 path (rows < ENG_DQ_MINR or no ENG_WO_W8A16_DQ)
    0 = not this route."""
    if W.get("A8c") is not None or not getattr(drv, "_attn_b_wo8_ok", False):
        return 0
    w8 = W.get("w8") or {}
    EM = drv.EM
    WoW8 = getattr(EM, "_WoW8", None)
    if WoW8 is None or not all(isinstance(w8.get(k), WoW8) for k in ("wo_a", "wo_b")):
        return 0
    if not w8["wo_a"].grouped or w8["wo_b"].grouped:
        return 0
    o, z, y = ws.get("o"), ws.get("z"), ws.get("y")
    if not (o is not None and z is not None and y is not None and o.is_contiguous() and o.numel() == R * 32768
            and z.dim() == 2 and z.stride(1) == 1 and z.shape[1] == 8192 and y.dim() == 2 and y.stride(1) == 1
            and y.shape[1] == 5120):
        return 0
    dq = [w8[k].dq is not None and R >= EM.DQ_MINR for k in ("wo_a", "wo_b")]
    if not any(dq):
        return 1
    if all(dq) and spans and getattr(drv, "_attn_b_dq_ok", False):
        return 2
    return 0


def _qdq(drv, EA, W, R) -> bool:
    """attn_b_insert takes its QB_DQ branch."""
    w8 = W.get("w8")
    return bool(getattr(EA, "QB_DQ", False) and w8 is not None and "wq_b_dq" in w8 and R >= getattr(EA, "DQ_MINR", 384))


def _attn_b_why(drv, EA, W, ws, s, c, R, spans: bool = True, static: bool = False) -> str:
    """else why not."""
    AH = EA.AH
    capt = torch.cuda.is_current_stream_capturing()
    d8 = W.get("A8c")
    w8 = W.get("w8")
    if capt:
        return "capturing"
    if d8 is None or "wo_a" not in d8 or "wo_b" not in d8:
        if not _wo_mode(drv, EA, W, ws, s, c, R, spans):
            return f"wo mode 0 (no DENSE8 wo, no wo8 variant; R {R}, spans {spans})"
    elif not getattr(drv, "_attn_b_d8_ok", True):
        return "DENSE8 attn_b tables not ready"
    # (static = ensure time: the step-type globals below (QB16 / QB16D / SKW8_QB / C128A, ws["pf_ab"]) are re-checked at
    # handle time, which refuses a handle a step must not take; ensure only skips them)
    if not static and d8 is not None and "q_b" in d8 and EA.d8_rows(ws):
        return "DENSE8_ADD q_b rows"                    # DENSE8_ADD qb
    if _qdq(drv, EA, W, R) and not (spans and d8 is None and getattr(drv, "_attn_b_dq_ok", False)):
        return (f"QB_DQ q_b (R {R}: spans {spans}, DENSE8 {d8 is not None}, "
                f"attn_b_dq_ok {getattr(drv, '_attn_b_dq_ok', False)})")
    if not static and W.get("wq_b16") is not None and ((EA.QB16 and R >= 256) or
                                                         (EA.QB16D and R > 128 and ws.get("pf_ab") is None)):
        return f"q_b bf16 (QB16 / QB16D, R {R})"
    if not static and EA.SKW8_QB and w8 is not None and "wq_b" in w8 and R <= 32 and ws.get("pf_ab") is None and "valid" in ws:
        return "skw8 q_b (R <= 32)"
    if w8 is None or "wq_b" not in w8 or getattr(AH, "w8_qb", None) is None or getattr(AH, "QB_SLICE", False):
        return "no w8a16 q_b (w8 / w8_qb / QB_SLICE)"
    if s.get("pf") is not None and not static and not (getattr(drv, "_attn_pf_ok", False) and s.get("_cxx_pf_ok")
                                                        and c.get("sched_dec") is not None):
        return (f"pf rows: attn_pf_ok {getattr(drv, '_attn_pf_ok', False)}, step {bool(s.get('_cxx_pf_ok'))}, "
                f"sched_dec {c.get('sched_dec') is not None}")   # bucket A's prefill-attention ranks: attn_pf or Python
    comp = c.get("k_comp") is not None
    if not static and EA.C128A and comp and ws.get("c128a") and ws.get("pf_ab") is None:
        return "C128A"
    return ""


def _attn_b_eligible(drv, EA, W, ws, s, c, R, spans: bool = True, static: bool = False) -> bool:
    return not _attn_b_why(drv, EA, W, ws, s, c, R, spans, static)


def _role(drv, s) -> str:
    """The bucket's role ("A" all rows / "T" tail rows / "" decode) from Engine.bk's key, cached in the state."""
    if "_cxx_role" not in s:
        role = ""
        bk = s.get("bkey")
        if isinstance(bk, tuple) and len(bk) > 2 and bk[2] in ("A", "T"):
            s["_cxx_role"] = str(bk[2])
            return s["_cxx_role"]
        for k, v in drv.eng.bk.items():
            if isinstance(k, tuple) and any(x is s for x in v):
                role = str(k[2])
                break
        s["_cxx_role"] = role
    return s["_cxx_role"]


def _plan_key(drv, s, L, dec: bool = False) -> int:
    """FlashMLA plan slot of layer L in this bucket: engine._new_scheds(per_layer=True)'s grouping (mixed steps). Slots are
    fixed per bucket role (A: 0..31, T: 32..63), so a segment's reset never touches the slots of the other role's
    segment that may still run (1d), and evicted buckets need no bookkeeping (-1: more than 32 keys -> Python)."""
    E, EM = drv.eng, drv.EM
    ids = s.setdefault("_cxx_plan_ids", {})
    if E.cfg.mixed_plans == "source":
        k = (EM.RATIO[L], EM.index_source_of(L), bool(s.get("l20")) and L == 20)
    else:
        k = ("layer", L)
    if dec:
        k = k + ("dec",)                               # ENG_ATTN_PF: c["sched_dec"], the decode rows' own plan
    if k not in ids:
        half = drv.x.max_plans() // 2
        if len(ids) >= half:
            return -1
        ids[k] = (half if _role(drv, s) == "T" else 0) + len(ids)
        # slot numbers are shared by every view of a role; a view's first build takes a slot another view
        #  last used, whose FlashMLA plan (tile metadata + num_splits for THAT view's batch) step_params' reset did not
        #  clear (the slot was not in this view's ids yet): "num_splits must have shape (b+1)". Reset it when assigned.)
        r_ = next((i for i, d in enumerate(E.devs) if torch.device(d) == torch.device(s["dev"])), None)
        # ENG_MIXED_CXX_ENSURE_RESET=0
        # no reset when the slot is assigned inside ensure(): redundant there (the view's first segment start resets every
        # slot in its ids, step_params), and the reset drops the plan tensors of ANOTHER view of the role mid-serving
        if getattr(drv, "_cxx_ensuring", False) and os.environ.get("ENG_MIXED_CXX_ENSURE_RESET", "1") == "0":
            drv.n["ensure_reset_skipped"] += 1
            r_ = None
        if getattr(drv, "_cxx_slots_only", False):
            r_ = None
        if r_ is not None and os.environ.get("ENG_MIXED_CXX_PLAN_RESET", "1") == "1":
            drv.x.attn_reset_plans(r_, [ids[k]])
    return ids[k]


def _scr_base(drv, r, s):
    """The state whose piece scratch view state s shares: an ENG_MIX_ONE_BUCKET view's pinned base
    (engine.bk_view[s["bkey"]] = (base states, view states)), else s itself (a base or a plain bucket). The engine's own
    _view_state rule for scratch (rs, zb, ti / tl): the base's first elements re-viewed, fully rewritten before every read,
    one rank stream (one continuous use sequence). Before: one set per view state, i.e. sum over every view's rows (GBs
    per GPU with ENG_MIX_PRECREATE's ~60 views)."""
    if os.environ.get("ENG_MIXED_CXX_SCR_SHARE", "1") == "0":
        return s
    bv = getattr(drv.eng, "bk_view", None)
    hit = bv.get(s.get("bkey")) if isinstance(bv, dict) else None
    if hit is not None and len(hit[1]) > r and hit[1][r] is s and int(hit[0][r]["R"]) >= int(s["R"]):
        return hit[0][r]
    return s


def _scr_set(drv, r, s, key, R, mk):
    """The scratch set `key` for R rows: the base's (sized once for the base's R, never regrown, so cached handle pointers
    never go stale), or a private one of this state's when the base's cannot hold R rows."""
    bs = _scr_base(drv, r, s)
    f = bs.get(key)
    if f is None:
        Rb = int(bs["R"]) if bs is not s else R
        with torch.cuda.stream(drv.eng.streams[r]):   # (allocated on the stream that uses them)
            f = bs[key] = mk(max(Rb, R))
    if f["R"] < R:
        with torch.cuda.stream(drv.eng.streams[r]):
            f = mk(R)
    return f


def _bufs(drv, r, s, R, dev):
    """Per-call scratch of the C path's DENSE8 quants (the Python path allocates them per call): R-row prefixes of one set
    per (rank, onebk base) (_scr_set)."""
    b = s.get("_cxx_attn_buf")
    if b is None:
        def mk(n):
            t = (n + 3) // 4 * 4
            return dict(R=n, oq=torch.empty(8 * n * 4096, dtype=torch.float8_e4m3fn, device=dev),
                        osc=torch.empty(8 * 32 * t, device=dev), zq=torch.empty(n * 8192, dtype=torch.float8_e4m3fn, device=dev),
                        zs=torch.empty(8192 // 128 * t, device=dev))
        f = _scr_set(drv, r, s, "_cxx_attn_scr", R, mk)
        TA, MA = (R + 3) // 4 * 4, (R + 3) // 4 * 4
        b = s["_cxx_attn_buf"] = dict(oq=f["oq"][: 8 * R * 4096].view(8, R, 4096), osc=f["osc"][: 8 * 32 * TA],
                                      zq=f["zq"][: R * 8192].view(R, 8192), zs=f["zs"][: 8192 // 128 * MA])
    return b


def _dq_record(drv, r, kind, g) -> int:
    """Handle of the dequant record of weight g (w8a16.Gemm) into device r's `kind` slot: w8dq's dequant_packed(g.packed,
    slot view) as _Dq.weight() calls it, captured once per (device, kind, weight) (kernel_capture: nothing executes in the
    capture; the warm-up, once per device, runs on a tiny scratch on a private stream: never on a live slot)."""
    import kernel_capture as KC
    import w8dq as W8DQ
    E = drv.eng
    dev = torch.device(E.devs[r])
    sl = drv.EM._DQ_DEV[str(dev)][kind]
    key = (dev.index, kind, g.packed.data_ptr(), sl.buf.data_ptr())
    recs_ = drv.__dict__.setdefault("_dq_recs", {})
    if key in recs_:
        return recs_[key]
    N, K = g.packed.shape[1] * 256, g.packed.shape[0] * 128
    w = sl.buf[: N * K].view(N, K)
    ext = W8DQ.ext()
    warmed = drv.__dict__.setdefault("_dq_warmed", {})
    if dev.index not in warmed:
        with torch.cuda.device(dev):
            tp = torch.zeros(1, 1, g.packed.shape[2], dtype=torch.uint8, device=dev)
            to = torch.empty(256, 128, dtype=torch.bfloat16, device=dev)
        warmed[dev.index] = (tp, to)                     # (kept: the warm launch may still run on capture's stream)
        warm = lambda: ext.dequant_packed(tp, to)        # noqa: E731
    else:
        warm = lambda: None                              # noqa: E731
    recs = KC.capture(lambda: ext.dequant_packed(g.packed, w), dev, warm=warm, sync=False)
    h = int(drv.x.attn_add_dq_record(recs[0])) if len(recs) == 1 else -1
    recs_[key] = h
    return h


def _w8_gemm(a, call, g, x, y, R) -> None:
    """FprW8a16GemmArgs of engine._WoW8's w8a16 call g.ext.gemm(x, g.packed, y, g.rpg, g.kxg, S, None x 5, 0, 0, None,
    g.ws, g.cnt, None) with S = g.splits at <= 128 rows, else 1; filled as w8a16.cu's pybind gemm() fills it (ws / cnt
    only with split-K)."""
    assert x.dim() == 2 and x.stride(1) == 1 and y.dim() == 2 and y.stride(1) == 1 and x.shape[0] == R == y.shape[0]
    assert g.packed.is_contiguous() and y.shape[1] >= g.packed.shape[1] * 256
    S = g.splits if R <= 128 else 1
    a.call = call
    a.x, a.ntok, a.kx, a.x_ld = _p(x), R, x.shape[1], x.stride(0)
    a.packed, a.nchunk, a.tiles = _p(g.packed), g.packed.shape[0], g.packed.shape[1]
    a.y, a.ldy, a.y_cols = _p(y), y.stride(0), y.shape[1]
    a.rows_per_group, a.kx_per_group, a.splits = g.rpg, g.kxg, S
    if S > 1:
        a.ws, a.ws_floats, a.cnt, a.cnt_n = _p(g.ws), g.ws.numel(), _p(g.cnt), g.cnt.numel()


def build_piece_attn_b(drv, r, s, L) -> bytes | None:
    import attn_act8 as AA
    import dg_cabi as DGC
    from ds41h import linear as DL
    E = drv.eng
    EA = _EA()
    AW, ws, c = E.lw[r][L]["AW"], s["ws"], s["caches"][L]
    R = int(ws["R"])
    dev = torch.device(E.devs[r]).index
    main = P(E.streams[r].cuda_stream)
    st = PieceAttnB()
    st.version, st.rank, st.layer, st.device, st.R, st.main = VERSION, r, L, dev, R, main
    # q_b (Gemm.__call__ -> w8a16.gemm with the Gemm's packed / ws / cnt)
    g = AW["w8"]["wq_b"]
    ins = bool(E.cfg.insert)
    qr, q2 = ws["qr"], ws["q"].view(R, 32768)
    q = st.qb
    q.call = FprCall(main, dev, r)
    q.x, q.ntok, q.kx, q.x_ld = _p(qr), R, qr.shape[1], qr.stride(0)
    q.packed, q.nchunk, q.tiles = _p(g.packed), g.packed.shape[0], g.packed.shape[1]
    q.y, q.ldy, q.y_cols = _p(q2), q2.stride(0), q2.shape[1]
    q.rows_per_group, q.kx_per_group, q.splits = g.rpg, g.kxg, g.splits
    q.pos, q.cs = _p(s["pos"]), _p(AW["cos_sin"])
    if ins:
        q.kv, q.slot, q.cache = _p(ws["kv"]), _p(c["own_slot"]), _p(c["sbuf"])
    q.page_stride, q.shift = int(c["stride"]), int(EA.SWA_SHIFT)
    q.ws, q.ws_floats, q.cnt, q.cnt_n = _p(g.ws), g.ws.numel(), _p(g.cnt), g.cnt.numel()
    # FlashMLA (pf: the decode rows on c["sched_dec"]'s plan, the prefill rows through FprAttnPfStep)
    comp = c.get("k_comp") is not None
    pf = s.get("pf") is not None
    st.plan_key, st.comp = _plan_key(drv, s, L, dec=pf), int(comp)
    if pf:
        ratio = int(c["ratio"])
        st.pf, st.ratio = 1, ratio
        st.src = -1 if c.get("src") is None else int(c["src"])
        st.isrc = -1 if c.get("isrc") is None else int(c["isrc"])
        assert ws["q"].is_contiguous() and ws["o"].is_contiguous() and ws["q"].numel() == R * 32768
        st.pf_q, st.pf_o = _p(ws["q"]), _p(ws["o"])
        k3 = EA._ap()._view3(c["k_swa"])
        assert k3.shape[1] == 32 and k3.shape[2] == 584 and k3.stride(1) == 584 and k3.element_size() == 1
        st.pf_kswa, st.pf_kswa_ps = _p(k3), k3.stride(0)
        if ratio:
            kc3 = EA._ap()._view3(c["k_comp"])
            assert kc3.shape[1] == 64 // ratio and kc3.shape[2] == 584 and kc3.stride(1) == 584 and kc3.element_size() == 1
            st.pf_kcomp, st.pf_kcomp_ps = _p(kc3), kc3.stride(0)
            ti = c["ti"]
            assert ti.dtype == torch.int32 and ti.is_contiguous() and ti.numel() == R * 512
            st.pf_ti = _p(ti)
        sk = AW["sink"]
        assert sk.dtype == torch.float32 and sk.numel() == 64 and sk.is_contiguous()
        st.pf_sink = _p(sk)
    if st.plan_key < 0:
        E.log(f"mixed_cxx_attn: attn_b rank {r} layer {L}: no FlashMLA plan slot left: Python")
        _build_sync(drv, r, locals().get("ps"))
        return None
    st.fq, st.fk_swa, st.fsi, st.fsl, st.fsink = (_view(ws["q"]), _view(c["k_swa"]), _view(c["si"]), _view(c["sl"]),
                                                    _view(AW["sink"]))
    if comp:
        st.fk_comp, st.fti, st.ftl = _view(c["k_comp"]), _view(c["ti"]), _view(c["tl"])
    st.fo, st.scale = _view(ws["o"]), float(c["scale"])
    if AW.get("A8c") is None:   # the ENG_WO_W8A16 tail (_wo_mode; eligibility checked by the caller)
        w8 = AW["w8"]
        mode = _wo_mode(drv, EA, AW, ws, s, c, R, spans=True)
        assert mode in (1, 2), mode
        st.dq_q = st.dq_a = st.dq_b = -1
        o3 = ws["o"].view(R, 64, 512)
        ir = st.ir
        ir.call = FprCall(main, dev, r)
        ir.o, ir.pos, ir.cos_sin, ir.t, ir.nh = _p(o3), _p(s["pos"]), _p(AW["cos_sin"]), R, 64
        if mode == 1:
            _w8_gemm(st.woa8, FprCall(main, dev, r), w8["wo_a"].g, ws["o"].view(R, 32768), ws["z"], R)
            _w8_gemm(st.wob8, FprCall(main, dev, r), w8["wo_b"].g, ws["z"], ws["y"], R)
        else:                   # _WoW8's _Dq route (engine.py: torch.bmm / torch.mm on the slot's exact bf16)
            ga, gb = w8["wo_a"].g, w8["wo_b"].g
            st.dq_a, st.dq_b = _dq_record(drv, r, "a", ga), _dq_record(drv, r, "b", gb)
            if st.dq_a < 0 or st.dq_b < 0:
                _build_sync(drv, r, locals().get("ps"))
                return None
            sa, sb = (drv.EM._DQ_DEV[str(torch.device(E.devs[r]))][k] for k in ("a", "b"))
            st.slot_a, st.slot_b = _p(sa.buf), _p(sb.buf)
            st.a_n, st.a_k = ga.packed.shape[1] * 256, ga.packed.shape[0] * 128
            st.b_n, st.b_k = gb.packed.shape[1] * 256, gb.packed.shape[0] * 128
            assert (st.a_n, st.a_k, st.b_n, st.b_k) == (8192, 4096, 5120, 8192), (st.a_n, st.a_k, st.b_n, st.b_k)
            st.wo_o, st.wo_z, st.wo_y = _p(ws["o"]), _p(ws["z"]), _p(ws["y"])
            st.z_ld, st.y_ld = ws["z"].stride(0), ws["y"].stride(0)
        st.wo = mode
        if _qdq(drv, EA, AW, R):   # attn_b_insert's QB_DQ branch: mm on the q slot's exact bf16, then rope_q_insert
            gq = w8["wq_b_dq"].g
            st.dq_q = _dq_record(drv, r, "q", gq)
            if st.dq_q < 0:
                _build_sync(drv, r, locals().get("ps"))
                return None
            st.slot_q = _p(drv.EM._DQ_DEV[str(torch.device(E.devs[r]))]["q"].buf)
            st.q_n, st.q_k = gq.packed.shape[1] * 256, gq.packed.shape[0] * 128
            assert st.q_n == q2.shape[1] and st.q_k == qr.shape[1], (st.q_n, st.q_k, q2.shape, qr.shape)
            q3 = ws["q"].view(R, 64, 512)
            assert q3.is_contiguous() and ws["kv"].is_contiguous()
            ri = st.rqi
            ri.call = FprCall(main, dev, r)
            ri.q, ri.kv, ri.pos, ri.cos_sin = _p(q3), _p(ws["kv"]), _p(s["pos"]), _p(AW["cos_sin"])
            ri.slot_mapping, ri.cache = _p(c["own_slot"]), _p(c["sbuf"])
            ri.page_stride, ri.shift, ri.t, ri.nh, ri.do_insert = int(c["stride"]), int(EA.SWA_SHIFT), R, 64, int(ins)
            st.qdq = 1
        if st.wo == 2 or st.qdq:
            drv.__dict__.setdefault("_dq_structs", set()).add((r, L, R))
        _build_sync(drv, r, locals().get("ps"))
        return bytes(st)
    # DENSE8 wo_a: ropeq64 (o -> oq / osc), DeepGEMM einsum with the views the pybind path returns
    d8 = AW["A8c"]
    b = _bufs(drv, r, s, R, ws["o"].device)
    o3 = ws["o"].view(R, 64, 512)
    TA = (R + 3) // 4 * 4
    rq = st.rq
    rq.call = FprCall(main, dev, r)
    rq.o, rq.o_stride_t, rq.o_stride_h = _p(o3), o3.stride(0), o3.stride(1)
    rq.positions, rq.cos_sin, rq.cs_stride = _p(s["pos"]), _p(AW["cos_sin"]), AW["cos_sin"].stride(0)
    rq.t, rq.n_groups, rq.out, rq.scales = R, 8, _p(b["oq"]), _p(b["osc"])
    oqv = b["oq"].transpose(0, 1)                                           # [R, 8, 4096] (pybind: out.transpose(0, 1))
    oscv = b["osc"].as_strided((8, R, 32), (32 * TA, 1, TA)).transpose(0, 1)   # (pybind: sc.transpose(0, 1))
    wa, wsa = d8["wo_a"]
    z3 = ws["z"].view(R, 8, 1024)
    sms = DGC.num_sms_of(dev)
    a = st.woa
    a.call = FprCall(main, dev, r)
    a.a, a.sfa, a.b, a.sfb, a.d = (_view(t, DgView, _DG) for t in (oqv, oscv, wa, wsa, z3))
    a.num_sms = sms
    # DENSE8 wo_b: act_quant_128 (z -> zq / zs), DeepGEMM gemm
    fw = d8["wo_b"]
    z = ws["z"]
    assert z.is_contiguous() and z.shape == (R, fw.k) and fw.kp == 8192, (z.shape, fw.k, fw.kp)
    MA = (R + 3) // 4 * 4
    aq = st.aq
    aq.call = FprCall(main, dev, r)
    aq.x, aq.ldx, aq.m, aq.k, aq.kp, aq.ue8m0 = _p(z), z.stride(0), R, fw.k, fw.kp, int(bool(DL.UE8M0))
    aq.q, aq.scales = _p(b["zq"]), _p(b["zs"])
    zsv = b["zs"].as_strided((R, fw.kp // 128), (1, MA))
    wb = st.wob
    wb.call = FprCall(main, dev, r)
    wb.a, wb.sfa, wb.b, wb.sfb, wb.d = (_view(t, DgView, _DG) for t in (b["zq"], zsv, fw.w, fw.ws, ws["y"]))
    wb.num_sms = sms
    # DeepGEMM keys of this bucket warmed now: scratch outputs of the same layout (the keys hold no pointers) on a private
    # stream, so the warm never touches the bucket's live buffers (1d: a span may be running on the rank stream); an
    # allocating / failing key -> Python piece
    with torch.cuda.device(dev):
        ps = _priv_stream(dev, "build")
        with torch.cuda.stream(ps):
            oq_s, osc_s = torch.zeros_like(b["oq"]), torch.zeros_like(b["osc"])
            zq_s, zs_s = torch.zeros_like(b["zq"]), torch.zeros_like(b["zs"])
            r1 = drv._dg.warm_einsum(oq_s.transpose(0, 1), osc_s.as_strided((8, R, 32), (32 * TA, 1, TA)).transpose(0, 1),
                                     wa, wsa, torch.empty_like(z3), sms)
            r2 = drv._dg.warm_gemm(zq_s, zs_s.as_strided((R, fw.kp // 128), (1, MA)), fw.w, fw.ws, torch.empty_like(ws["y"]),
                                   sms)
    # (no ps.synchronize(): a build must never block the host; the keys are compiled host-side by the warm calls, their
    # scratch launches finish on ps in stream order)
    if r1 != 0 or r2 != 0:
        E.log(f"mixed_cxx_attn: attn_b rank {r} layer {L} R {R}: DeepGEMM key not warmable (einsum {r1}, gemm {r2}): Python")
        _build_sync(drv, r, locals().get("ps"))
        return None
    _build_sync(drv, r, locals().get("ps"))
    return bytes(st)


def piece_attn_b(drv, r, s, L) -> bool:
    if not getattr(drv, "_attn_b_ok", False):
        return False
    E = drv.eng
    EA = _EA()
    AW, ws = E.lw[r][L]["AW"], s["ws"]
    c = s["caches"][L]
    if not _attn_b_eligible(drv, EA, AW, ws, s, c, int(ws["R"]), spans=False):
        drv._attn_n["attn_b_py"] += 1
        return False
    cache = s.setdefault("_cxx", {})
    key = ("attn_b", r, L, s.get("pf") is not None)
    if key not in cache:
        b = build_piece_attn_b(drv, r, s, L)
        cache[key] = None if b is None else drv.x.add_attn_b(b)
    h = cache[key]
    if h is None:
        drv._attn_n["attn_b_py"] += 1
        return False
    _pf_handoff(drv, r, s)
    rc = drv.x.attn_b(r, h, L)
    assert rc == 0, f"mixed_cxx_attn: attn_b rank {r} layer {L}: FPR error {rc}"
    drv._attn_n["attn_b"] += 1
    if drv._attn_n["attn_b"] == 1:
        E.log(f"mixed_cxx_attn: first C++ attn_b piece: rank {r} layer {L} R {ws['R']} plan key {_plan_key(drv, s, L)}")
    return True


# ---------------------------------------------------------------------------------------------- stage 1d (spans)
def handle_attn_a(drv, r, s, L) -> int:
    """The cached static attn_a handle for this bucket / layer, or -1 (the whole Python p_attn). Build only."""
    if not getattr(drv, "_attn_ok", False):
        return -1
    E = drv.eng
    AW, ws = E.lw[r][L]["AW"], s["ws"]
    why = _attn_a_why(_EA(), AW, ws, int(ws["R"]))
    if why:
        return _why(drv, "attn_a", why, r, L)
    cache = s.setdefault("_cxx", {})
    key = ("attn_a", r, L)
    if key not in cache:
        cache[key] = drv.x.add_attn_a(build_piece_attn_a(drv, r, s, L))
    return int(cache[key])


def handle_attn_b(drv, r, s, L) -> int:
    """The cached static attn_b handle, or -1. Build only (the DeepGEMM warm-up on scratch outputs, private stream)."""
    if not getattr(drv, "_attn_b_ok", False):
        return -1
    E = drv.eng
    AW, ws = E.lw[r][L]["AW"], s["ws"]
    why = _attn_b_why(drv, _EA(), AW, ws, s, s["caches"][L], int(ws["R"]))
    if why:
        return _why(drv, "attn_b", why, r, L)
    cache = s.setdefault("_cxx", {})
    key = ("attn_b", r, L, s.get("pf") is not None)
    if key not in cache:
        b = build_piece_attn_b(drv, r, s, L)
        cache[key] = None if b is None else drv.x.add_attn_b(b)
    h = cache[key]
    if h is not None:
        _pf_handoff(drv, r, s)
        R_ = int(ws["R"])
        if _wo_mode(drv, _EA(), AW, ws, s, s["caches"][L], R_) == 2 or _qdq(drv, _EA(), AW, R_):
            drv.__dict__.setdefault("_dq_handles", set()).add(int(h))   # (mixed_cxx: slot handover at the span's flush)
    return -1 if h is None else int(h)


def _pf_handoff(drv, r, s) -> None:
    """Bucket A's prefill plan is shared by the Python and the C++ attends of a step: each side's reuse state (the
    resident compressed region, the combined lists) is only valid while the other side has not written the workspace /
    comb since. A Python attend since the last C++ one (Plan.stats moved) -> the C++ state starts over; and this C++ layer
    invalidates the Plan's own state for the next Python attend. (Both only force a recompute: same bits.) Runs at
    handle time, in issue order: a Python piece always flushes the pending span before it runs."""
    pl = s.get("pf")
    if pl is None or not s.get("_cxx_pf_ok"):
        return
    cur = sum(pl.stats.values())
    if cur != s.get("_cxx_pf_last"):
        drv.x.attn_pf_invalidate(r)
    pl.comp_src = pl.ws_ratio = None
    pl.comb_key.clear()
    s["_cxx_pf_last"] = cur


# ---------------------------------------------------------------------------------------------- attn_pf (stage 1c-1)
class PfGroup(C.Structure):          # pieces_attn.h FprAttnPfGroup
    _fields_ = [("a", I64), ("b", I64), ("N", I64), ("M", I64), ("kv", P), ("q0", I64), ("q1", I64), ("rec", I32),
                ("gy", I32), ("vals", C.c_uint64 * 16)]


class PfLayout(C.Structure):         # FprAttnPfLayout
    _fields_ = [("ratio", I32), ("ngroups", I32), ("disjoint", I32), ("pad", I32), ("g", PfGroup * 32)]


def _pf_step_type():
    import triton_replay as TR
    return type("PfStep", (C.Structure,), {"_fields_": [
        ("valid", I32), ("nlayouts", I32), ("nrecs", I32), ("pad0", I32), ("n", I64), ("row_lo", I64), ("row_hi", I64),
        ("seq_lens", P), ("gather_lens", P), ("bt", P), ("wbt", P), ("inv", P), ("loc", P), ("ld_bt", I64), ("ld_wbt", I64),
        ("n_inv", I64), ("seq_lens_r", P * 4), ("comb", P * 2), ("clen", P * 2), ("cw", I64 * 2), ("lay", PfLayout * 4),
        ("rec", TR.Record * 8), ("ws_ratio", I32), ("comp_src", I32), ("ck_valid", I32 * 2), ("ck_ratio", I32 * 2),
        ("ck_isrc", I32 * 2)]})


@functools.cache
def _pf_step_type_cached():          # (built on first use: the Record type comes from triton_replay)
    return _pf_step_type()


_COMB_RECS: dict = {}                # (device, ratio, top_k) -> [triton_replay.Rec]: the combine kernel's key classes


def _pf_step(drv, r, s):
    """FprAttnPfStep of this step's attn_prefill.Plan on rank r (Python thread, before the segment), or None (Python).
    Everything attend would derive per layer that is the same for all layers of the step: the group layouts per ratio
    (Plan.group_plan / group_ws, prealloc workspace: views only), comb / clen per width (created as attend would), the
    per-ratio seq_lens // ratio, and per group the combine's Triton record + this step's parameter values (a missing key
    class is recorded once with the real arguments on the rank stream: the launch only writes this step's comb / clen,
    which every first combine of the step rewrites)."""
    import triton_replay as TR
    from vllm.models.deepseek_v41.common.ops.cache_utils import combine_topk_swa_indices
    E = drv.eng
    EA = _EA()
    AP = EA._ap()
    pl = s["pf"]
    PS = _pf_step_type_cached()
    st = PS()
    ratios = sorted({int(c["ratio"]) for c in s["caches"].values()})
    if len(ratios) > 4 or pl.n <= 0:
        return None
    if any(len(pl.group_plan(ra)) > 32 for ra in ratios):
        return None

    def need(ratio):   # Plan.group_ws's allocation for this ratio (bf16 elements; its no-prealloc rule), without allocating
        areas = [(b - a) * M * 512 for a, b, N, M in pl.group_plan(ratio)]
        disjoint = len(areas) == 1 or sum(areas) * 2 <= AP.WS_TOTAL
        return sum(areas) if disjoint else max(areas)

    dev = torch.device(E.devs[r])
    f0 = dict(pl.stats_ws)
    lays = []
    # the layouts, largest first: without ENG_PF_WS_PREALLOC_MB the first call allocates Plan.buf once at the step's largest
    # need and every other ratio fits in it (no reallocation, so the views stay valid); and on the rank stream (with its
    # device current), as attend would allocate it
    with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
        for ratio in sorted(ratios, key=need, reverse=True):
            views, disjoint = pl.group_ws(ratio)
            lays.append((ratio, pl.group_plan(ratio), views, disjoint))
    if pl.stats_ws != f0 or any(pl.gws.get(ra) is None or pl.gws[ra][0] is not vw for ra, _, vw, _ in lays):
        return None                                   # (a workspace fallback / reallocation: Python decides per layer)
    keep = s["_cxx_pf_keep"] = []
    with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
        for cw in (640, 128):
            if cw not in pl.comb:
                pl.comb[cw] = torch.empty(pl.n, cw, dtype=torch.int32, device=dev)
                pl.clen[cw] = torch.empty(pl.n, dtype=torch.int32, device=dev)
        st.valid, st.nlayouts, st.n, st.row_lo, st.row_hi = 1, len(lays), pl.n, pl.row_lo, pl.row_hi
        st.seq_lens, st.gather_lens, st.bt, st.wbt = _p(pl.seq_lens), _p(pl.gather_lens), _p(pl.bt), _p(pl.wbt)
        st.inv, st.loc, st.n_inv = _p(pl.inv), _p(pl.loc), pl.inv.numel()
        st.ld_bt, st.ld_wbt = pl.bt.stride(0), pl.wbt.stride(0)
        for i, cw in enumerate((640, 128)):
            st.comb[i], st.clen[i], st.cw[i] = _p(pl.comb[cw]), _p(pl.clen[cw]), cw
        recs = []
        for li, (ratio, gp, views, disjoint) in enumerate(lays):
            if ratio:
                slr = pl.seq_lens // ratio
                keep.append(slr)
                st.seq_lens_r[li] = _p(slr)
            top_k = AP.TOPK if ratio else 0
            cw = -(-(top_k + AP.WIN) // 128) * 128
            comb, clen, loc = pl.comb[cw], pl.clen[cw], pl.loc
            Ly = st.lay[li]
            Ly.ratio, Ly.ngroups, Ly.disjoint = ratio, len(gp), int(disjoint)
            for gi, (a, b, N, M) in enumerate(gp):
                q0, q1 = pl.qsl_h[a], pl.qsl_h[b]
                G = Ly.g[gi]
                G.a, G.b, G.N, G.M, G.kv, G.q0, G.q1 = a, b, N, M, _p(views[gi]), q0, q1
                ov = dict(combined_indices_ptr=comb[q0:q1], combined_indices_stride=comb.stride(0),
                          combined_lens_ptr=clen[q0:q1], topk_indices_ptr=loc[q0:q1], topk_indices_stride=loc.stride(0),
                          query_start_loc_ptr=pl.qsl[a: b + 1], seq_lens_ptr=pl.seq_lens[a:b],
                          gather_lens_ptr=pl.gather_lens[a:b], M=M, N=N)
                ck = (dev.index, ratio, top_k)
                rec = None
                for rc_ in _COMB_RECS.get(ck, []):
                    vals = rc_.values(**ov)
                    if TR.ext().check(bytes(rc_.record), TR.Rec.values_bytes(vals)) == 0:
                        rec = rc_
                        break
                if rec is None:
                    with TR.recording() as trs:
                        combine_topk_swa_indices(loc[q0:q1], pl.qsl[a: b + 1], pl.seq_lens[a:b], pl.gather_lens[a:b],
                                                 AP.WIN, ratio, top_k, M, N, out=(comb[q0:q1], clen[q0:q1]))
                    if len(trs) != 1:
                        return None
                    rec = trs[0]
                    vals = rec.values(**ov)
                    if TR.ext().check(bytes(rec.record), TR.Rec.values_bytes(vals)) != 0:
                        return None
                    _COMB_RECS.setdefault(ck, []).append(rec)
                    drv.n["comb_late"] += 1
                if len(vals) > 16:
                    return None
                if rec not in recs:
                    if len(recs) >= 8:
                        return None
                    recs.append(rec)
                G.rec, G.gy = recs.index(rec), int(rec.grid[1])
                for i, v in enumerate(vals):
                    G.vals[i] = v
        st.nrecs = len(recs)
        for i, rc_ in enumerate(recs):
            C.memmove(C.addressof(st) + PS.rec.offset + i * C.sizeof(TR.Record), C.addressof(rc_.record), C.sizeof(TR.Record))
    s["_cxx_pf_last"] = sum(pl.stats.values())
    return st


# ------------------------------------------------------------------------------------------------------------------ idx
class IdxChunk(C.Structure):         # pieces_attn.h FprAttnIdxChunk
    _fields_ = [("b0", I64), ("b1", I64), ("ublk", P), ("uslot", P), ("nunion", P), ("cblk", P), ("clen", P), ("upk", P),
                ("comp", P), ("bm", P), ("ld_cblk", I64)]


class IdxStep(C.Structure):          # pieces_attn.h FprAttnIdxStep (pieces_attn's own per-rank state)
    _fields_ = [("valid", I32), ("idle", I32), ("use2", I32), ("single", I32), ("nchunks", I32), ("nwords", I32),
                ("B", I64), ("IR", I64), ("nv", I64), ("skip", I32 * 8), ("prep_done", I32), ("pad1", I32),
                ("ch", IdxChunk * 16)]


def _tr_record_type():
    import triton_replay as TR
    return TR.Record


def _kc_record_size():
    import kernel_capture as KC
    return KC.ext().record_size()


@functools.cache
def _piece_idx_type():
    TRR = _tr_record_type()
    return type("PieceAttnIdx", (C.Structure,), {"_fields_": [
        ("version", I32), ("rank", I32), ("layer", I32), ("device", I32), ("R", I64), ("main", P), ("lidx", I32), ("pad0", I32),
        ("x", P), ("wp", P), ("qr", P), ("wqb", P), ("wbuf", P), ("qbuf", P), ("x_ld", I64), ("kx", I64), ("qr_ld", I64),
        ("kq", I64), ("pad1", I64), ("qq", C.c_uint8 * _kc_record_size()), ("q8buf", P), ("wobuf", P), ("ir2row", P), ("q8i", P), ("woi", P),
        ("cand", P), ("seq_lens1", P), ("kv", P), ("bt", P), ("topk_out", P), ("slow", P), ("num_pages", I64),
        ("page_stride", I64), ("ld_bt", I64), ("gt", TRR), ("gt_vals", C.c_uint64 * 16), ("gbuf", P), ("lnbuf", P),
        ("ti", P), ("tl", P), ("vir", P), ("vrow", P), ("pad2", I64)]})


IDX_CONSUMERS = (24, 28, 32, 36)


def _idx_ok(drv, s) -> bool:
    return (getattr(drv, "_attn_idx_ok", False) and _role(drv, s) == "T" and isinstance(s.get("idx"), dict)
            and s["idx"].get("general"))


def _idx_step(drv, r, s):
    """This segment's FprAttnIdxStep from the bucket's indexer state (Python thread, before the span); None = Python."""
    import idx2f as X
    d = s["idx"]
    st = IdxStep()
    if d.get("idle"):
        st.valid, st.idle = 1, 1
        return st
    oi = d["oi"]
    ws = getattr(oi, "ws", None)
    if (ws is None or oi.mode != "fast" or getattr(oi, "fuse", False) or oi.ks[20].kvm is not None or X.LOGITS != "f8"
            or oi.cand.stride(0) != X.CAND_K):       # (the piece: fast mode, lean ws, unfused, no mirror, fcand8 logits)
        _why(drv, "idx_step", f"indexer state (lean ws {ws is not None}, mode {oi.mode}, fuse {getattr(oi, 'fuse', False)}, "
                              f"mirror {oi.ks[20].kvm is not None}, LOGITS {X.LOGITS}, cand ld {oi.cand.stride(0)} vs "
                              f"{X.CAND_K})", r)
        return None
    for i, L in enumerate(X.LAYERS):
        st.skip[i] = int(oi.max_len.get(L, 1) <= 0)
    ml = oi.max_len.get(24, 0)
    if any(oi.max_len.get(L, 0) != ml for L in IDX_CONSUMERS):
        _why(drv, "idx_step", "per-layer bounds differ", r)
        return None                                  # (per-layer bounds that differ: Python; ratio 1 makes them equal)
    use2 = bool(oi.cand_ascending) and ml <= X.LONG_ROW      # _consumer_ws's rule (fast mode)
    B, IR = int(d["B"]), int(d["IR"])
    nch = -(-B // ws.cb)
    if nch > 16:
        _why(drv, "idx_step", f"{nch} chunks > 16", r)
        return None
    st.valid, st.use2, st.single, st.nchunks = 1, int(use2), int(B <= ws.cb), nch
    st.nwords = -(-(-(-int(oi.max_len.get(24, 1)) // X.CAND_BLOCK)) // 32)
    st.B, st.IR, st.nv = B, IR, int(d["vir"].numel())
    for c, b0 in enumerate(range(0, B, ws.cb)):
        b1 = min(B, b0 + ws.cb)
        p = ws.prep_bufs(b1 - b0, cblk=not use2, upk=use2)
        k = st.ch[c]
        k.b0, k.b1 = b0, b1
        k.ublk, k.uslot, k.nunion, k.clen = _p(p.ublk), _p(p.uslot), _p(p.nunion), _p(p.clen)
        k.upk = _p(p.upk) if use2 else P(0)
        if use2:
            k.cblk, k.ld_cblk = P(oi.cand.data_ptr() + 4 * b0 * 4 * X.CAND_K), X.CAND_K
        else:
            k.cblk, k.ld_cblk = _p(p.cblk), p.cblk.stride(0)
        k.comp = _p(ws.logits(4 * (b1 - b0), X.CAND_K * X.CAND_BLOCK))
        k.bm = _p(ws.bm(4 * (b1 - b0), X.CAND_K))
    return st


def _idx_bufs(drv, r, s, R, dev, IRmax):
    """Per-call scratch of the idx / idx_a pieces: R-row prefixes of one set per (rank, onebk base) (_scr_set), its IR
    part sized for the base's bound 4 * ceil64(R) (GeneralTables: B <= rows, Bc = ceil64(B)), so it never regrows."""
    b = s.get("_cxx_idx_buf")
    if b is None or b["IRmax"] < IRmax:
        def mk(n, ir=None):
            ir = max(IRmax, 4 * (-(-n // 64) * 64)) if ir is None else ir
            return dict(R=n, IR=ir, wbuf=torch.empty(n, 32, dtype=torch.bfloat16, device=dev),
                        qbuf=torch.empty(n, 4096, dtype=torch.bfloat16, device=dev),
                        q8buf=torch.empty(n, 32, 128, dtype=torch.float8_e4m3fn, device=dev), wobuf=torch.empty(n, 32, device=dev),
                        q8i=torch.empty(ir, 32, 128, dtype=torch.float8_e4m3fn, device=dev), woi=torch.empty(ir, 32, device=dev),
                        gbuf=torch.empty(ir, 512, dtype=torch.int32, device=dev), lnbuf=torch.empty(ir, dtype=torch.int32, device=dev))
        f = _scr_set(drv, r, s, "_cxx_idx_scr", R, mk)
        if f["IR"] < IRmax:
            with torch.cuda.stream(drv.eng.streams[r]):
                f = mk(R, IRmax)
        b = s["_cxx_idx_buf"] = dict(IRmax=f["IR"], wbuf=f["wbuf"][:R], qbuf=f["qbuf"][:R], q8buf=f["q8buf"][:R],
                                     wobuf=f["wobuf"][:R], q8i=f["q8i"], woi=f["woi"], gbuf=f["gbuf"], lnbuf=f["lnbuf"])
    return b


def build_piece_attn_idx(drv, r, s, L):
    """FprPieceAttnIdx for (T bucket, consumer layer L); the CuTe capture / Triton recording on a private stream."""
    import idx2f as X
    import kernel_capture as KC
    import triton_replay as TR
    from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import _INDEXER_Q_FP8_KERNEL as QK
    from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens
    E = drv.eng
    H = E.hooks.idx
    W, d, ws = H.W[r][L], s["idx"], s["ws"]
    R = int(s["R"])
    dev = torch.device(E.devs[r])
    oi = d["oi"]
    gt = s.get("idx_gt")                             # GeneralTables: d's tables are prefix views of its 4 Bc rows
    IRmax = max(4 * int(gt.Bc) if gt is not None else 0, int(d["IR"]))
    b = _idx_bufs(drv, r, s, R, dev, IRmax)
    PT = _piece_idx_type()
    want = [PT.qq.offset, PT.q8buf.offset, PT.gt.offset, PT.gt_vals.offset, PT.vrow.offset]
    if C.sizeof(PT) != drv.x.sizeof_attn_idx() or list(drv.x.offsets_attn_idx()) != want:
        raise RuntimeError(f"FprPieceAttnIdx layout: ctypes {C.sizeof(PT)} {want} vs C++ {drv.x.sizeof_attn_idx()} "
                           f"{list(drv.x.offsets_attn_idx())}")
    st = PT()
    st.version, st.rank, st.layer, st.device, st.R = VERSION, r, L, dev.index, R
    st.main, st.lidx = P(E.streams[r].cuda_stream), X.LAYERS.index(L)
    x, qr = s["a"], ws["qr"]
    st.x, st.x_ld, st.kx = _p(x), x.stride(0), x.shape[1]
    st.qr, st.qr_ld, st.kq = _p(qr), qr.stride(0), qr.shape[1]
    st.wp, st.wqb = _p(W["idx_wp"]), _p(W["idx_wq_b"])
    assert W["idx_wp"].shape == (32, x.shape[1]) and W["idx_wq_b"].shape == (4096, qr.shape[1])
    st.wbuf, st.qbuf, st.q8buf, st.wobuf = _p(b["wbuf"]), _p(b["qbuf"]), _p(b["q8buf"]), _p(b["wobuf"])
    ps = _priv_stream(dev, "build")
    # NO host sync anywhere in a build (it runs while a span is built: the device may be running work that waits on work
    # not issued yet). The capture's warm-up runs on SCRATCH buffers of the same layout on a private stream (the live
    # q8buf / wobuf may still be read by an earlier span), the capture itself executes nothing.
    with torch.cuda.device(dev), torch.cuda.stream(ps):
        sq, sw = torch.empty_like(b["qbuf"]), torch.empty_like(b["wbuf"])
        s8, so = torch.empty_like(b["q8buf"]), torch.empty_like(b["wobuf"])

        spos = torch.zeros_like(s["pos"])   # the warm LAUNCHES: position 0 for every row, never the view's live pos
        #                                    (a new view's padding rows may hold out-of-range positions -> cos_sin OOB read)

        def mk(q_, w_, q8_, wo_, p_):
            return lambda: QK(positions=p_, q=q_.view(R, 32, 128), cos_sin_cache=W["cos_sin"], weights=w_,
                              weights_softmax_scale=128 ** -0.5, weights_head_scale=32 ** -0.5,
                              q_fp8=q8_.view(torch.uint8), weights_out=wo_)
        recs = KC.capture(mk(b["qbuf"], b["wbuf"], b["q8buf"], b["wobuf"], s["pos"]), dev, warm=mk(sq, sw, s8, so, spos),
                          sync=False)
        if len(recs) != 1:
            _build_sync(drv, r, locals().get("ps"))
            return None
        C.memmove(C.addressof(st) + PT.qq.offset, recs[0], len(recs[0]))
        st.ir2row, st.q8i, st.woi = _p(d["ir2row_c"]), _p(b["q8i"]), _p(b["woi"])
        k20 = oi.ks[20]
        st.cand, st.seq_lens1, st.kv, st.bt = _p(oi.cand), _p(d["seq_lens"][1]), _p(k20.kv), _p(k20.block_table)
        st.num_pages, st.page_stride, st.ld_bt = k20.kv.shape[0], k20.kv.stride(0), k20.block_table.stride(0)
        st.topk_out, st.slow = _p(oi.topk[L]), _p(oi.slow)
        # the top-k recording on scratch inputs (see build_piece_idx_a): no table read through garbage
        gk_s = {"topk": torch.full_like(oi.topk[L], -1), "t2r": torch.zeros_like(d["t2r_ir"]), "bt": torch.zeros_like(d["bt"]),
                "valid": torch.zeros_like(d["valid_ir"])}
        gk_real = {gk_s["topk"]: oi.topk[L], gk_s["t2r"]: d["t2r_ir"], gk_s["bt"]: d["bt"],
                   gk_s["valid"]: d["valid_ir"]}
        with TR.recording() as trecs:   # (runs on ps; its own outputs are dropped; no sync)
            compute_global_topk_indices_and_lens(gk_s["topk"], gk_s["t2r"], gk_s["bt"], H.RPP[20], gk_s["valid"])
    if len(trecs) != 1:
        _build_sync(drv, r, locals().get("ps"))
        return None
    tr = trecs[0]
    vals = _remap(tr, gk_real, dict(global_topk_indices_ptr=b["gbuf"], topk_lens_ptr=b["lnbuf"]))
    if len(vals) > 16:
        _build_sync(drv, r, locals().get("ps"))
        return None
    C.memmove(C.addressof(st) + PT.gt.offset, C.addressof(tr.record), C.sizeof(tr.record))
    for i, v in enumerate(vals):
        st.gt_vals[i] = v
    st.gbuf, st.lnbuf = _p(b["gbuf"]), _p(b["lnbuf"])
    st.ti, st.tl, st.vir, st.vrow = _p(s["ti"][L]), _p(s["tl"][L]), _p(d["vir"]), _p(d["vrow"])
    s.setdefault("_cxx_idx_ptrs", {})[L] = _idx_ptrs(s, L)
    _build_sync(drv, r, locals().get("ps"))
    return bytes(st)


# the signature fields of the idx / idx_a handles: ONE table drives the Python staleness keys (_idx_ptrs /
# _idxa_ptrs) and the export core2b checks per step (s["_cxx_attn_export"]["sig"], sig_hash = FNV-1a64 of the joined names)
def _scr(k):
    return lambda s, d, oi, L: (s.get("_cxx_idx_buf") or {}).get(k)


_SCR_FIELDS = tuple(("scr." + k, _scr(k)) for k in ("q8i", "woi", "gbuf", "lnbuf", "qbuf", "wbuf", "q8buf", "wobuf"))
_COMMON = (("a", lambda s, d, oi, L: s["a"]), ("qr", lambda s, d, oi, L: s["ws"]["qr"]), ("pos", lambda s, d, oi, L: s.get("pos")),
           ("ti[L]", lambda s, d, oi, L: s["ti"][L]), ("tl[L]", lambda s, d, oi, L: s["tl"][L]),
           ("ir2row_c", lambda s, d, oi, L: d["ir2row_c"]), ("seq_lens1", lambda s, d, oi, L: d["seq_lens"][1]),
           ("seq_lens2", lambda s, d, oi, L: d["seq_lens"][2]), ("bt", lambda s, d, oi, L: d["bt"]),
           ("t2r_ir", lambda s, d, oi, L: d["t2r_ir"]), ("valid_ir", lambda s, d, oi, L: d["valid_ir"]),
           ("vir", lambda s, d, oi, L: d["vir"]), ("vrow", lambda s, d, oi, L: d["vrow"]))
IDX_SIG = _SCR_FIELDS + _COMMON + (
    ("cand", lambda s, d, oi, L: oi.cand), ("slow", lambda s, d, oi, L: oi.slow), ("topk[L]", lambda s, d, oi, L: oi.topk[L]),
    ("ks20.kv", lambda s, d, oi, L: oi.ks[20].kv), ("ks20.bt", lambda s, d, oi, L: oi.ks[20].block_table))
IDXA_SIG = _SCR_FIELDS + (("kpre", lambda s, d, oi, L: s.get("_cxx_idxa_kpre")),) + _COMMON + (
    ("lat", lambda s, d, oi, L: d["lat"]), ("cslot1", lambda s, d, oi, L: d["cslot"][1]),
    ("cslot2", lambda s, d, oi, L: d["cslot"][2]), ("rslot", lambda s, d, oi, L: d["rslot"]),
    ("qsl", lambda s, d, oi, L: d["qsl"]), ("t2r", lambda s, d, oi, L: d["t2r"]), ("cand", lambda s, d, oi, L: oi.cand),
    ("topk[L]", lambda s, d, oi, L: oi.topk[L]), ("slow", lambda s, d, oi, L: oi.slow),
    ("nb_lens", lambda s, d, oi, L: oi.nb_lens), ("ws_f32", lambda s, d, oi, L: oi.ws.f32),
    ("ws_bmax", lambda s, d, oi, L: oi.ws.bmax), ("ksL.kv", lambda s, d, oi, L: oi.ks[L].kv),
    ("ksL.bt", lambda s, d, oi, L: oi.ks[L].block_table))
IDX_SIG_FIELDS = tuple(n for n, _ in IDX_SIG)
IDXA_SIG_FIELDS = tuple(n for n, _ in IDXA_SIG)
SIG_MAX = max(len(IDX_SIG), len(IDXA_SIG))


def _fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for ch in text.encode():
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


SIG_HASH = {"idx": _fnv1a64(",".join(IDX_SIG_FIELDS)), "idx_a": _fnv1a64(",".join(IDXA_SIG_FIELDS))}


def _sig(table, s, L) -> tuple:
    d = s["idx"]
    oi = d["oi"]
    out = []
    for _, f in table:
        t = f(s, d, oi, L)
        out.append(0 if t is None else int(t.data_ptr()))
    return tuple(out)


def _idx_ptrs(s, L):
    """the T idx handle's signature (IDX_SIG): every per-state pointer FprPieceAttnIdx bakes"""
    return _sig(IDX_SIG, s, L)


def _rb_log(drv, k, r, L) -> None:
    """server-log evidence of stale-key rebuilds: one line at counts 1, 2, 4, 8, ... (drv.n prints only at exit)"""
    c = drv.n[k]
    if c & (c - 1) == 0:
        print(f"mixed_cxx_attn: {k} {c} (rank {r} layer {L}: a baked pointer moved, handle rebuilt)", flush=True)


def _export_row(s, L, h, sig) -> None:
    """keep s["_cxx_attn_export"] current when a step rebuilds an idx / idx_a handle (core2b re-reads it on
    _cxx_sig_epoch)"""
    ex = s.get("_cxx_attn_export")
    if not ex or not (0 <= L - ex["layer0"] < ex["n"]) or os.environ.get("ENG_MIXED_CXX_EXPORT_IDX", "1") == "0":
        return
    i = L - ex["layer0"]
    ex["h"][i, 1] = -1 if h is None else int(h)
    ex["sig"][i, :] = 0
    if h is not None:
        ex["sig"][i, : len(sig)] = sig
    ex["sig_n"][i] = 0 if h is None else len(sig)


def handle_idx(drv, r, s, L) -> int:
    """The cached idx handle (T-role bucket, consumer layer; Stage B: A-role KV-source layer, a handle >= IDXA_BASE), or -1.
    Build only (capture / recording on a private stream)."""
    if L in IDXA_SOURCES and _role(drv, s) == "A":
        return handle_idx_a(drv, r, s, L)
    if L not in IDX_CONSUMERS:
        return -1
    if not _idx_ok(drv, s):
        d_ = s.get("idx")
        return _why(drv, "idx", f"idx not ok (attn_idx_ok {getattr(drv, '_attn_idx_ok', False)}, role {_role(drv, s)}, "
                                f"general {isinstance(d_, dict) and bool(d_.get('general'))})", r, L)
    if not s.get("_cxx_idx_step_ok", False):
        return _why(drv, "idx", "no step values (see the idx_step reason)", r, L)
    idle = bool(s["idx"].get("idle"))
    cache = s.setdefault("_cxx", {})
    key = ("idx", r, L)
    stale = not idle and s.get("_cxx_idx_ptrs", {}).get(L) not in (None, _idx_ptrs(s, L))
    if stale:
        drv.n["rebuild_idx"] += 1
        _rb_log(drv, "rebuild_idx", r, L)
        s["_cxx_sig_epoch"] = int(s.get("_cxx_sig_epoch", 0)) + 1   # (core2b re-reads the handles + signatures)
    if key not in cache or stale:
        if idle:
            return -1                                # (the first build needs a live step's tables)
        try:
            bts = build_piece_attn_idx(drv, r, s, L)
        except Exception as e:
            drv.eng.log(f"mixed_cxx_attn: idx rank {r} layer {L}: build failed ({e}): Python")
            bts = None
        cache[key] = None if bts is None else drv.x.add_attn_idx(bts)
        _export_row(s, L, cache[key], _idx_ptrs(s, L))
    h = cache[key]
    return -1 if h is None else int(h)


# CP prefill phases (mixed_cxx ENG_MIXED_CXX_CP=2): the two halves of the same attn_b handle (pieces_attn attn_bi / attn_bo)
handle_attn_bi = handle_attn_b
handle_attn_bo = handle_attn_b


# ---------------------------------------------------------------------------------------------- CP exchanges (mode 3)
# ------------------------------------------------------------------------------------------------- idx_a (Stage B)
# hooks_idx._compress_general + _score_general of an A-role mixed bucket's KV-source layer (pieces_attn.h FprPieceIdxA):
# kind "idx" (both halves, Engine.p_attn), kinds "idx_c" / "idx_s" (CP prefill P1 / P2). Served flags only: GeneralTables,
# lean OwnedIndexer on the EagerWorkspace, fast mode, idx2f dense logits, IDX2F_PF slices (idx2f_pf4), no mirror / IDX2F_DL;
# rows > LONG_ROW (stock select / top-k) stay Python for that layer and step. ENG_MIXED_CXX_IDXA=0: off.
IDXA_BASE = 1 << 20
IDXA_MAX_SLICES = 64
IDXA_MAXV = 32
IDXA_SOURCES = (2, 8, 14, 20)
# A-role index layers with rows > LONG_ROW run the idx_a piece's stock
# branch (vLLM select_candidate_blocks + _C top-k + canon_topk in C++, refuse = 2) instead of refusing to Python (refuse = 1)
IDXA_STOCK = os.environ.get("ENG_MIXED_CXX_IDXA_STOCK", "0") == "1"
_TOPK_WS: dict = {}                  # str(device) -> the stock top-k's private uint8 workspace (vLLM's RADIX size)


class IdxASlice(C.Structure):        # pieces_attn.h FprIdxASlice
    _fields_ = [("b0", I64), ("b1", I64), ("gst_off", I64), ("g", I64)]


class IdxAStep(C.Structure):         # pieces_attn.h FprIdxAStep
    _fields_ = [("valid", I32), ("idle", I32), ("logits", I32), ("global_rows", I32), ("B", I64), ("IR", I64), ("nv", I64),
                ("nreq", I64), ("skip", I32 * 4), ("refuse", I32 * 4), ("max_len", I64 * 4), ("width", I64 * 4),
                ("nsl", I32 * 2), ("pad1", I32 * 2), ("sl", (IdxASlice * IDXA_MAX_SLICES) * 2), ("gst", P),
                ("max_seq_len", I64)]


@functools.cache
def _piece_idxa_type():
    TRR = _tr_record_type()
    return type("PieceIdxA", (C.Structure,), {"_fields_": [
        ("qq", C.c_uint8 * _kc_record_size()), ("version", I32), ("rank", I32), ("layer", I32), ("device", I32), ("R", I64),
        ("main", P), ("lidx", I32), ("sidx", I32), ("ratio", I32), ("pad0", I32),
        ("x", P), ("comp_w", P), ("ks", P), ("lat", P), ("idx_wk", P), ("kpre", P), ("x_ld", I64), ("kx", I64), ("ncomp", I64),
        ("fscn", TRR), ("fscn_vals", C.c_uint64 * IDXA_MAXV), ("fscn_nreq", I32), ("fscn_nv", I32),
        ("rqi", TRR), ("rqi_vals", C.c_uint64 * IDXA_MAXV), ("iks", TRR), ("iks_vals", C.c_uint64 * IDXA_MAXV),
        ("rqi_nv", I32), ("iks_nv", I32),
        ("wp", P), ("qr", P), ("wqb", P), ("wbuf", P), ("qbuf", P), ("q8buf", P), ("wobuf", P), ("ir2row", P), ("q8i", P),
        ("woi", P), ("qr_ld", I64), ("kq", I64),
        ("kv", P), ("num_pages", I64), ("page_stride", I64), ("page_tok", I64), ("bt", P), ("ld_bt", I64), ("bt_cols", I64),
        ("seq_lens", P), ("nb_lens", P), ("cand", P), ("topk_out", P), ("slow", P), ("ws_f32", P), ("ws_bm", P),
        ("ws_f32_n", I64), ("ws_bm_n", I64),
        ("gt", TRR), ("gt_vals", C.c_uint64 * 16), ("gbuf", P), ("lnbuf", P), ("ti", P), ("tl", P), ("vir", P), ("vrow", P),
        ("bs", TRR), ("sc", TRR), ("bs_vals", C.c_uint64 * 16), ("sc_vals", C.c_uint64 * 16), ("bs_nv", I32), ("sc_nv", I32),
        ("bs_i", I32 * 6), ("sc_i", I32 * 4), ("canon", I32), ("stock_ok", I32), ("topk_ws", P), ("topk_ws_n", I64),
        ("pad4", I64), ("pad3", I64)]})


def _idxa_ok(drv, s) -> bool:
    return (getattr(drv, "_idxa_ok", False) and _role(drv, s) == "A" and isinstance(s.get("idx"), dict)
            and s["idx"].get("general"))


def _idxa_values(idle: bool, B: int, IR: int, nv: int, nreq: int, max_len: dict, width: dict, pf_groups, budget: int,
                 logits: int, LONG_ROW: int, NEXT_N: int = 4, MAX_CALL_B: int = 1024, stock: bool = False,
                 global_rows: int = 0, max_seq_len: int = 0):
    """The pure host part of _idxa_step (P1's C++ port fpr::p1::idxa_step mirrors it; golden: core/tests/p1_record_steps.py):
    FprIdxAStep's values and the pf4 slices' group starts (relative to each slice's b0), or None (> IDXA_MAX_SLICES)."""
    v = dict(valid=1, idle=int(bool(idle)), logits=int(logits), B=0, IR=0, nv=0, nreq=0, skip=[0] * 4, refuse=[0] * 4,
             max_len=[0] * 4, width=[0] * 4, nsl=[0, 0], sl=[[], []], global_rows=int(global_rows),
             max_seq_len=int(max_seq_len))
    if idle:
        return v, []
    v.update(B=int(B), IR=int(IR), nv=int(nv), nreq=int(nreq))
    for i, L in enumerate(IDXA_SOURCES):
        ml = int(max_len.get(L, 0))
        v["max_len"][i], v["width"][i] = ml, int(width.get(L, 256))
        v["skip"][i] = int(max_len.get(L, 1) <= 0)
        v["refuse"][i] = (2 if stock else 1) if ml > LONG_ROW else 0   # 2: the piece's stock branch
    gst = []
    for k, L in ((0, 2), (1, 20)):                 # (layers 2 / 8 / 14 share the ratio-2 bounds: plan_bounds)
        per_b = NEXT_N * int(width[L]) * 4
        bstep = min(max(1, budget // per_b), MAX_CALL_B)
        sls = []
        for b0 in range(0, B, bstep):
            b1 = min(B, b0 + bstep)
            off, g = -1, 0
            if pf_groups is not None:              # OwnedIndexer._pf_gstart's rule
                gs = sorted({b0, b1} | {x for x in pf_groups if b0 < x < b1})
                if 2 * (len(gs) - 1) <= b1 - b0:
                    off, g = len(gst), len(gs) - 1
                    gst.extend(x - b0 for x in gs)
            sls.append((b0, b1, off, g))
        if len(sls) > IDXA_MAX_SLICES:
            return None
        v["nsl"][k] = len(sls)
        v["sl"][k] = sls
    return v, gst


def _idxa_step(drv, r, s):
    """The A segment's FprIdxAStep (Python thread, before the span): the lean indexer's per-layer bounds, OwnedIndexer.
    _dense_sliced's slices per ratio set and _pf_gstart's group starts (uploaded to the bucket's own int32 buffer on the
    rank stream); None = Python (a branch the piece does not mirror)."""
    import idx2f as X
    d = s["idx"]
    if d.get("idle"):
        st = IdxAStep()
        st.valid, st.idle = 1, 1
        return st
    oi = d.get("oi")
    ws = getattr(oi, "ws", None)
    if (oi is None or ws is None or oi.mode != "fast" or getattr(oi, "dl", False) or oi.bmax is not None
            or getattr(oi, "dense_impl", "") != "idx2f" or d.get("mir") is not None or s.get("idx_gt") is None
            or any(getattr(oi.ks[L], "kvm", None) is not None for L in IDXA_SOURCES)):
        _why(drv, "idxa_step", f"indexer state (oi {oi is not None}, lean ws {ws is not None}, mode "
                               f"{getattr(oi, 'mode', None)}, dl {getattr(oi, 'dl', False)}, bmax {getattr(oi, 'bmax', None)}, "
                               f"impl {getattr(oi, 'dense_impl', '')}, mirror {d.get('mir') is not None}, "
                               f"gt {s.get('idx_gt') is not None})", r)
        return None
    lg = {"f8": 1, "cand": 2}.get(getattr(X, "LOGITS", ""))
    if lg is None:
        _why(drv, "idxa_step", f"LOGITS {getattr(X, 'LOGITS', '')}", r)
        return None
    pf = (bool(getattr(oi, "pf", False)) and oi.pf_groups is not None)
    if pf and not getattr(drv, "_idxa_pf4", False):
        _why(drv, "idxa_step", "IDX2F_PF slices without the pf4 table", r)
        return None
    r_ = _idxa_values(False, int(d["B"]), int(d["IR"]), int(d["vir"].numel()), int(d["qsl"].numel()) - 1, oi.max_len,
                      oi.width, oi.pf_groups if pf else None, ws.budget, lg, X.LONG_ROW, X.NEXT_N, X.MAX_CALL_B,
                      stock=IDXA_STOCK, global_rows=int(d.get("global_rows", 0)), max_seq_len=int(d.get("max_seq_len", 1)))
    if r_ is None:
        _why(drv, "idxa_step", "values refused (_idxa_values)", r)
        return None
    v, gst = r_
    st = IdxAStep()
    st.valid, st.idle, st.logits, st.B, st.IR, st.nv, st.nreq = 1, 0, v["logits"], v["B"], v["IR"], v["nv"], v["nreq"]
    st.global_rows, st.max_seq_len = v["global_rows"], v["max_seq_len"]
    for i in range(4):
        st.skip[i], st.refuse[i], st.max_len[i], st.width[i] = v["skip"][i], v["refuse"][i], v["max_len"][i], v["width"][i]
    for k in range(2):
        st.nsl[k] = v["nsl"][k]
        for j, (b0, b1, off, g) in enumerate(v["sl"][k]):
            e = st.sl[k][j]
            e.b0, e.b1, e.gst_off, e.g = b0, b1, off, g
    if gst:
        E = drv.eng
        dev = torch.device(E.devs[r])
        with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):   # (stream-ordered after the bucket's earlier users)
            buf = s.get("_cxx_idxa_gst")
            if buf is None or buf.numel() < len(gst):
                if s.get("_cxx_idxa_gst_fixed"):
                    return None                       # (ensure's buffer is exported to nd: never reallocated; Python runs idx_a)
                buf = s["_cxx_idxa_gst"] = torch.empty(max(4096, 2 * len(gst)), dtype=torch.int32, device=dev)
            buf[: len(gst)].copy_(torch.tensor(gst, dtype=torch.int32).pin_memory(), non_blocking=True)
        st.gst = _p(buf)
    return st


def _idxa_ptrs(s, L):
    """the Stage B idx_a handle's signature (IDXA_SIG): every per-state pointer FprPieceIdxA bakes"""
    return _sig(IDXA_SIG, s, L)


def _remap(rec, real: dict, override: dict | None = None) -> list:
    """rec.values() with every argument that points INTO a scratch tensor (its base or any offset: a wrapper may pass a
    slice / view) replaced by the same offset into its real tensor (scratch = zeros_like / full_like / empty_like of the
    real one: same strides), plus name overrides. keys are the scratch TENSORS (int keys: exact base only, as
    before). A value left pointing into a scratch range would dangle once the scratch is freed -> RuntimeError (the
    builder's caller refuses the handle)."""
    vals = rec.values(**(override or {}))
    rng = []
    for k, v in real.items():
        if isinstance(k, torch.Tensor):
            rng.append((int(k.data_ptr()), int(k.numel() * k.element_size()), int(v.data_ptr())))
        else:
            rng.append((int(k), 1, int(v.data_ptr())))
    out = []
    for x in vals:
        y = x
        if isinstance(x, int) and x > 0:
            for base, nb, rb in rng:
                if base <= x < base + nb:
                    y = rb + (x - base)
                    break
        out.append(y)
    for x in out:
        if isinstance(x, int) and x > 0 and any(base <= x < base + nb for base, nb, _ in rng):
            raise RuntimeError(f"_remap: a recorded pointer {x:#x} still points into builder scratch")
    return out

def build_piece_idx_a(drv, r, s, L):
    """FprPieceIdxA for (A bucket, KV-source layer L): the Triton recordings (scratch slot mappings of -1: the kernels write
    nothing), the CuTe q-quant capture and the global top-k recording, all on a private stream with no host sync."""
    import sys
    import idx2f as X
    import kernel_capture as KC
    import triton_replay as TR
    from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import _INDEXER_Q_FP8_KERNEL as QK
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store
    from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import fused_save_compress_norm, rope_quant_insert
    E = drv.eng
    H = E.hooks.idx
    EPS = sys.modules[type(H).__module__].EPS
    W, d, ws = H.W[r][L], s["idx"], s["ws"]
    v = H.views[r]
    R = int(s["R"])
    dev = torch.device(E.devs[r])
    oi = d["oi"]
    ratio = X.RATIO[L]
    gt = s.get("idx_gt")
    IRmax = max(4 * int(gt.Bc) if gt is not None else 0, int(d["IR"]))
    b = _idx_bufs(drv, r, s, R, dev, IRmax)
    kp = s.get("_cxx_idxa_kpre")
    if kp is None or kp.shape[0] < R:   # (the projected k of one call: scratch, an R-row prefix of the base's set)
        f = _scr_set(drv, r, s, "_cxx_idxa_kpre_scr", R,
                     lambda n: dict(R=n, k=torch.empty(n, 128, dtype=torch.bfloat16, device=dev)))
        kp = s["_cxx_idxa_kpre"] = f["k"][:R]
    ks = b["qbuf"].view(-1).view(torch.float32)[: R * 512 * ratio].view(R, 512 * ratio)   # (qbuf's bytes: the compressor
    #                                                    is done with ks before the scoring's q projection writes qbuf)
    PT = _piece_idxa_type()
    x, qr, pos = s["a"], ws["qr"], s["pos"]
    cw, wk = W["comp_w"], W["idx_wk"]
    assert cw.shape == (512 * ratio, x.shape[1]) and cw.is_contiguous() and wk.shape == (128, 512) and wk.is_contiguous()
    assert W["idx_wp"].shape == (32, x.shape[1]) and W["idx_wq_b"].shape == (4096, qr.shape[1])
    assert x.stride(1) == 1 and qr.stride(1) == 1 and d["lat"].shape == (R, 512) and d["lat"].is_contiguous()
    kv = oi.ks[L].kv
    assert kv.dim() == 4 and kv.dtype == torch.uint8 and kv.shape[2] == 1 and kv.shape[3] == 132
    st = PT()
    st.version, st.rank, st.layer, st.device, st.R = VERSION, r, L, dev.index, R
    st.main, st.lidx, st.sidx, st.ratio = P(E.streams[r].cuda_stream), IDXA_SOURCES.index(L), 0 if ratio == 2 else 1, ratio
    st.x, st.x_ld, st.kx = _p(x), x.stride(0), x.shape[1]
    st.comp_w, st.ks, st.lat, st.idx_wk, st.kpre, st.ncomp = _p(cw), _p(ks), _p(d["lat"]), _p(wk), _p(kp), 512 * ratio
    st.wp, st.qr, st.wqb, st.qr_ld, st.kq = _p(W["idx_wp"]), _p(qr), _p(W["idx_wq_b"]), qr.stride(0), qr.shape[1]
    st.wbuf, st.qbuf, st.q8buf, st.wobuf = _p(b["wbuf"]), _p(b["qbuf"]), _p(b["q8buf"]), _p(b["wobuf"])
    slot = d["rslot"] if ratio == 2 else d["cslot"][ratio]
    ps = _priv_stream(dev, "build")
    with torch.cuda.device(dev), torch.cuda.stream(ps):
        # 1. the compressor's Triton kernels, recorded with slot mappings of -1 (each kernel returns before any store) and
        #    scratch ks / latent / kpre; replays take the real pointers (_remap) and, at ratio 2, the step's num_reqs
        neg = torch.full((R,), -1, dtype=torch.int64, device=dev)
        sks = torch.zeros(R, 512 * ratio, dtype=torch.float32, device=dev)
        slat = torch.empty(R, 512, dtype=torch.bfloat16, device=dev)
        skp = torch.empty(R, 128, dtype=torch.bfloat16, device=dev)
        # recording LAUNCHES the kernels: at view creation the tables (qsl / t2r / topk / bt /
        # valid_ir) may hold garbage (fresh, or another view's), and the request program indexes slot_mapping through qsl and
        # the pair programs read t2r -> out-of-bounds reads on ps. Scratch copies with safe contents (qsl [0, 0]: one empty
        # request; t2r 0) are recorded instead and remapped to the real tables by address (_remap), like neg / sks / slat
        sqsl = torch.zeros_like(d["qsl"])
        st2r = torch.zeros_like(d["t2r"])
        spos = torch.zeros_like(pos)       # no recorded launch reads the live positions either
        with TR.recording() as crec:
            if ratio == 2:
                fused_save_compress_norm(sks, spos, v[f"ring{L}"], neg, sqsl, st2r, W["comp_norm"], EPS, ratio, slat)
            else:
                fused_save_compress_norm(sks, spos, None, neg, None, None, W["comp_norm"], EPS, ratio, slat)
            rope_quant_insert(slat, spos, W["cos_sin"], v[f"comp{L}"], neg, ratio)
            indexer_k_norm_rope_store(skp, spos, W["cos_sin"], W["idx_k_norm"], EPS, v[f"idxs{L}"], neg, ratio, False)
        if len(crec) != 3:
            _build_sync(drv, r, locals().get("ps"))
            return None
        real = {neg: slot, sks: ks, slat: d["lat"], skp: kp,
                sqsl: d["qsl"], st2r: d["t2r"], spos: pos}
        f_, r_, i_ = crec
        fv = _remap(f_, real)
        rv = _remap(r_, {neg: d["cslot"][ratio], slat: d["lat"], spos: pos})
        iv = _remap(i_, {neg: d["cslot"][ratio], skp: kp, spos: pos})
        if max(len(fv), len(rv), len(iv)) > IDXA_MAXV:
            _build_sync(drv, r, locals().get("ps"))
            return None
        st.fscn_nreq = f_.names.index("num_reqs") if (ratio == 2 and "num_reqs" in f_.names) else -1
        if ratio == 2 and st.fscn_nreq < 0:
            _build_sync(drv, r, locals().get("ps"))
            return None
        for name, rec, vals in (("fscn", f_, fv), ("rqi", r_, rv), ("iks", i_, iv)):
            C.memmove(C.addressof(st) + getattr(PT, name).offset, C.addressof(rec.record), C.sizeof(rec.record))
            arr = getattr(st, name + "_vals")
            for j, x_ in enumerate(vals):
                arr[j] = x_
        st.fscn_nv, st.rqi_nv, st.iks_nv = len(fv), len(rv), len(iv)
        # 2. the CuTe q-quant (= the T piece)
        sq, sw = torch.empty_like(b["qbuf"]), torch.empty_like(b["wbuf"])
        s8, so = torch.empty_like(b["q8buf"]), torch.empty_like(b["wobuf"])

        def mk(q_, w_, q8_, wo_, p_):
            return lambda: QK(positions=p_, q=q_.view(R, 32, 128), cos_sin_cache=W["cos_sin"], weights=w_,
                              weights_softmax_scale=128 ** -0.5, weights_head_scale=32 ** -0.5,
                              q_fp8=q8_.view(torch.uint8), weights_out=wo_)
        recs = KC.capture(mk(b["qbuf"], b["wbuf"], b["q8buf"], b["wobuf"], pos), dev, warm=mk(sq, sw, s8, so, spos), sync=False)
        if len(recs) != 1:
            _build_sync(drv, r, locals().get("ps"))
            return None
        C.memmove(C.addressof(st) + PT.qq.offset, recs[0], len(recs[0]))
        st.ir2row, st.q8i, st.woi = _p(d["ir2row_c"]), _p(b["q8i"]), _p(b["woi"])
        # 3. the dense layer's tables (GeneralTables / the lean indexer / the EagerWorkspace: persistent bases)
        bt = d["bt"]
        st.kv, st.num_pages, st.page_stride, st.page_tok = _p(kv), kv.shape[0], kv.stride(0), kv.shape[1]
        st.bt, st.ld_bt, st.bt_cols = _p(bt), bt.stride(0), bt.shape[1]
        st.seq_lens = _p(d["seq_lens"][ratio])
        st.nb_lens = _p(oi.nb_lens) if oi.nb_lens is not None else P(0)
        st.cand, st.topk_out, st.slow = _p(oi.cand), _p(oi.topk[L]), _p(oi.slow)
        st.ws_f32, st.ws_bm, st.ws_f32_n, st.ws_bm_n = _p(oi.ws.f32), _p(oi.ws.bmax), oi.ws.f32.numel(), oi.ws.bmax.numel()
        if L == 20 and oi.nb_lens is None:
            _build_sync(drv, r, locals().get("ps"))
            return None
        # 4. the global top-k (= the T piece; this source's records per page)
        # the top-k recording on scratch inputs (topk -1, t2r / bt 0, valid_ir 0: no table read through garbage)
        gk_s = {"topk": torch.full_like(oi.topk[L], -1), "t2r": torch.zeros_like(d["t2r_ir"]), "bt": torch.zeros_like(bt),
                "valid": torch.zeros_like(d["valid_ir"])}
        gk_real = {gk_s["topk"]: oi.topk[L], gk_s["t2r"]: d["t2r_ir"], gk_s["bt"]: bt,
                   gk_s["valid"]: d["valid_ir"]}
        with TR.recording() as trecs:   # (its own outputs are dropped; no sync)
            compute_global_topk_indices_and_lens(gk_s["topk"], gk_s["t2r"], gk_s["bt"], H.RPP[L], gk_s["valid"])
    if len(trecs) != 1:
        _build_sync(drv, r, locals().get("ps"))
        return None
    tr = trecs[0]
    vals = _remap(tr, gk_real, dict(global_topk_indices_ptr=b["gbuf"], topk_lens_ptr=b["lnbuf"]))
    if len(vals) > 16:
        _build_sync(drv, r, locals().get("ps"))
        return None
    C.memmove(C.addressof(st) + PT.gt.offset, C.addressof(tr.record), C.sizeof(tr.record))
    for i, x_ in enumerate(vals):
        st.gt_vals[i] = x_
    st.gbuf, st.lnbuf = _p(b["gbuf"]), _p(b["lnbuf"])
    st.ti, st.tl, st.vir, st.vrow = _p(s["ti"][L]), _p(s["tl"][L]), _p(d["vir"]), _p(d["vrow"])
    s.setdefault("_cxx_idxa_stock", {})[L] = _idxa_stock_fields(st, dev, oi, L, ps)
    s.setdefault("_cxx_idxa_ptrs", {})[L] = _idxa_ptrs(s, L)
    return bytes(st)   # (the scratch tensors were allocated on ps: their blocks are reused only by later ps work)


def _idxa_stock_fields(st, dev, oi, L, ps) -> bool:
    """FprPieceIdxA's stock branch: canon_topk (oi.canon_stock), the private top-k workspace (vLLM's
    RADIX_TOPK_WORKSPACE_SIZE; vLLM's own is a torch.empty that other features overwrite between calls, so the kernels keep
    nothing in it) and, for layer 20, the recordings of vLLM select_candidate_blocks' two Triton kernels on a dummy slice of
    the served key class (width 16,384: stride / width / nblocks multiples of 16, k = OUT_K = 2,048; long rows have width
    > 262,144 and nblocks > 32,768, the same class). False: the stock branch is not built (refuse 2 -> Python)."""
    import triton_replay as TR
    try:
        from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE as RW
    except Exception:
        RW = 1024 * 1024
    st.canon = int(bool(getattr(oi, "canon_stock", False)))
    k = str(dev)
    if k not in _TOPK_WS or _TOPK_WS[k].numel() < RW:
        _TOPK_WS[k] = torch.empty(RW, dtype=torch.uint8, device=dev)
    st.topk_ws, st.topk_ws_n = _p(_TOPK_WS[k]), int(_TOPK_WS[k].numel())
    if L != 20:
        st.stock_ok = 1
        return True
    try:
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks as scb
        stock = getattr(scb, "_ds41h_stock", scb)
        with torch.cuda.device(dev), torch.cuda.stream(ps):
            lgd = torch.zeros(4, 16384, dtype=torch.float32, device=dev)
            ked = torch.full((4,), 16384, dtype=torch.int32, device=dev)
            outd = torch.empty(4, 2048, dtype=torch.int32, device=dev)
            with TR.recording() as recs:
                stock(lgd, None, ked, 2048, 8, outd, 1)
        if len(recs) != 2:
            return False
        bs, sc = recs
        bn, sn = list(bs.names), list(sc.names)
        bv, sv = bs.values(), sc.values()
        if len(bv) > 16 or len(sv) > 16 or len(bn) != len(bv) or len(sn) != len(sv):
            return False
        bi = [bn.index(n) for n in ("logits", "ends", "scores", "stride_row", "width", "nblocks")]
        si = [sn.index(n) for n in ("values", "indices", "output", "k")]
        C.memmove(C.addressof(st) + type(st).bs.offset, C.addressof(bs.record), C.sizeof(bs.record))
        C.memmove(C.addressof(st) + type(st).sc.offset, C.addressof(sc.record), C.sizeof(sc.record))
        for j, x_ in enumerate(bv):
            st.bs_vals[j] = int(x_) if x_ is not None else 0
        for j, x_ in enumerate(sv):
            st.sc_vals[j] = int(x_) if x_ is not None else 0
        st.bs_nv, st.sc_nv = len(bv), len(sv)
        for j, x_ in enumerate(bi):
            st.bs_i[j] = x_
        for j, x_ in enumerate(si):
            st.sc_i[j] = x_
        st.stock_ok = 1
        return True
    except Exception:
        return False


def handle_idx_a(drv, r, s, L) -> int:
    """The cached A-role handle (>= IDXA_BASE) for KV-source layer L, or -1 (no step values / refused layer / build
    failed). The step values (_cxx_idxa_step, step_params) decide per segment; the struct is rebuilt only when a
    persistent table moved (GeneralTables growth)."""
    st = s.get("_cxx_idxa_step")
    if L not in IDXA_SOURCES:
        return -1
    if not _idxa_ok(drv, s):
        return _why(drv, "idx_a", f"idx_a not ok (idxa_ok {getattr(drv, '_idxa_ok', False)}, role {_role(drv, s)})", r, L)
    if st is None:
        return _why(drv, "idx_a", "no step values (see the idxa_step reason)", r, L)
    i = IDXA_SOURCES.index(L)
    if not st.idle and st.refuse[i] == 1:
        return _why(drv, "idx_a", "rows > LONG_ROW (refuse 1: ENG_MIXED_CXX_IDXA_STOCK off)", r, L)
    cache = s.setdefault("_cxx", {})
    key = ("idx_a", r, L)
    stale = not st.idle and s.get("_cxx_idxa_ptrs", {}).get(L) not in (None, _idxa_ptrs(s, L))
    if stale:
        drv.n["rebuild_idxa"] += 1
        _rb_log(drv, "rebuild_idxa", r, L)
        s["_cxx_sig_epoch"] = int(s.get("_cxx_sig_epoch", 0)) + 1
    if key not in cache or stale:
        if st.idle:
            return -1                                # (the first build needs a live step's tables)
        try:
            bts = build_piece_idx_a(drv, r, s, L)
        except Exception as e:
            drv.eng.log(f"mixed_cxx_attn: idx_a rank {r} layer {L}: build failed ({e}): Python")
            bts = None
        cache[key] = None if bts is None else int(drv.x.add_idx_a(bts))
        _export_row(s, L, cache[key], _idxa_ptrs(s, L))
    h = cache[key]
    if h is not None and not st.idle and st.refuse[i] == 2:
        if not s.get("_cxx_idxa_stock", {}).get(L):
            return -1
        drv.n["idxa_stock"] += 1
    return -1 if h is None else int(h)


# CP prefill (mixed_cxx ENG_MIXED_CXX_CP=2 / 3): the compressor (P1, before the X1a send) and the scoring (P2, after the
# X1a receive) of the same A handle
handle_idx_c = handle_idx_a
handle_idx_s = handle_idx_a


# ------------------------------------------------------------------------- combine templates
COMB_TPL_MAGIC, COMB_TPL_VERSION = 0x54435046, 1
_COMB_INIT: set = set()              # device indices whose key classes record_comb_classes has recorded


def record_comb_classes(drv, r) -> int:
    """every key class of the combine kernel (combine_topk_swa_indices) a served attn_pf step can meet
    recorded at init on rank r's device and appended to _COMB_RECS exactly as _pf_step does on a first meet, so
    comb_templates(dev) is complete before the first native prefill step (no "unrecorded key class" refusal in steady state).
    Triton specializes the call on the 16-B alignment of its pointers (comb / loc rows: always aligned; clen + q0, qsl /
    seq_lens / gather_lens + a: q0 % 4 and a % 4) and on the integers M / N (== 1, % 16 == 0, other): one synthetic group
    of two 4-row sequences per (ratio 0 / 1 / 2) x (a % 4 0 / 1) x (q0 % 4 0 / 1) x (M, N in 1 / 16 / 17). A candidate is
    launched only when no recorded class accepts its values (the check _pf_step runs), so only new classes compile. The
    launches write only the dummy tensors. Returns the number of classes added."""
    import triton_replay as TR
    from vllm.models.deepseek_v41.common.ops.cache_utils import combine_topk_swa_indices
    E = drv.eng
    dev = torch.device(E.devs[r])
    if dev.index in _COMB_INIT:
        return 0
    AP = _EA()._ap()
    i32 = dict(dtype=torch.int32, device=dev)
    added = 0
    with torch.cuda.device(dev), torch.cuda.stream(E.streams[r]):
        for ratio in (0, 1, 2):
            top_k = AP.TOPK if ratio else 0
            cw = -(-(top_k + AP.WIN) // 128) * 128
            ck = (dev.index, ratio, top_k)
            for am in (0, 1):
                for qm in (0, 1):
                    for M in (1, 16, 17):
                        for N in (1, 16, 17):
                            a, b, q0 = am, am + 2, qm
                            q1 = q0 + 8
                            loc = torch.full((q1 + 1, 512), -1, **i32)
                            comb = torch.zeros((q1 + 1, cw), **i32)
                            clen = torch.zeros((q1 + 1,), **i32)
                            qsl = torch.tensor([0] * a + [q0, q0 + 4, q1], **i32)
                            sl = torch.tensor([0] * a + [8, 8], **i32)
                            gl = torch.tensor([0] * a + [4, 4], **i32)
                            ov = dict(combined_indices_ptr=comb[q0:q1], combined_indices_stride=comb.stride(0),
                                      combined_lens_ptr=clen[q0:q1], topk_indices_ptr=loc[q0:q1],
                                      topk_indices_stride=loc.stride(0), query_start_loc_ptr=qsl[a: b + 1],
                                      seq_lens_ptr=sl[a:b], gather_lens_ptr=gl[a:b], M=M, N=N)
                            if any(TR.ext().check(bytes(rc_.record), TR.Rec.values_bytes(rc_.values(**ov))) == 0
                                   for rc_ in _COMB_RECS.get(ck, [])):
                                continue
                            with TR.recording() as trs:
                                combine_topk_swa_indices(loc[q0:q1], qsl[a: b + 1], sl[a:b], gl[a:b], AP.WIN, ratio, top_k, M,
                                                         N, out=(comb[q0:q1], clen[q0:q1]))
                            if len(trs) != 1:
                                continue
                            rec = trs[0]
                            if TR.ext().check(bytes(rec.record), TR.Rec.values_bytes(rec.values(**ov))) != 0:
                                continue
                            _COMB_RECS.setdefault(ck, []).append(rec)
                            added += 1
    _COMB_INIT.add(dev.index)
    return added


def comb_templates(dev_index: int) -> bytes:
    """The combine kernel's recorded key classes of one device (_COMB_RECS, append-only: _pf_step records a new class the
    first time an embedded step meets it), serialized for nd's pf_step (fpr::p1::parse_comb_templates; format "FPCT" v1,
    see steps_build.h). The exporter publishes it per rank as comb_tpl / comb_tpl_bytes and bumps comb_gen after a change
    (a list that grew since the last export: compare the count)."""
    import struct
    import triton_replay as TR
    items = [(k, rc) for k, lst in sorted(_COMB_RECS.items()) if k[0] == dev_index for rc in lst]
    rsz = C.sizeof(TR.Record)
    out = bytearray(struct.pack("<IIII", COMB_TPL_MAGIC, COMB_TPL_VERSION, len(items), rsz))
    for (dev, ratio, top_k), rc in items:
        names = list(rc.names)
        vals = rc.values()
        out += struct.pack("<iiii", int(ratio), int(top_k), int(rc.grid[1]), len(names))
        out += bytes(rc.record)
        for nm in names:
            b = nm.encode()
            out += struct.pack("<H", len(b)) + b
        out += b"\0" * ((8 - len(out) % 8) % 8)
        out += struct.pack(f"<{len(vals)}Q", *[int(v) & 0xFFFFFFFFFFFFFFFF for v in vals])
    return bytes(out)


# ------------------------------------------------------------------------------------------ ensure
def _gt_full_dict(drv, r, s):
    """GeneralTables.upload's dict at the view's full capacity (B = Bc, every row), from its PERSISTENT tensors: the same
    base pointers every per-step dict has (slices from 0), so handles built from it are the step's (the _idx_ptrs /
    _idxa_ptrs staleness tuples compare equal). None: no indexer / tables."""
    import idx2f as X
    H = drv.eng.hooks.idx
    if H is None or getattr(H, "tables", "device") == "host":
        return None
    gt, _core = H._general_tables(r, s)
    Bc, R = int(gt.Bc), int(gt.R)
    IR = 4 * Bc
    bt = gt.bt[:Bc]
    ks = {L: X.CompactK(gt.idx_views[L], bt, gt.rpp[L]) for L in X.KV_SOURCES}
    oi = gt.base.rows_view(Bc, ks)
    oi.nb_lens = gt.nb_lens[:IR]                       # (= plan_bounds' view)
    sl = {c: gt.sl[c][:IR].view(Bc, 4) for c in (1, 2)}
    return dict(general=True, B=Bc, IR=IR, bt=bt, t2r_ir=gt.t2r_ir[:IR], valid_ir=gt.valid_ir[:IR], ir2row_c=gt.ir2row_c[:IR],
                vir=gt.vir[:R], vrow=gt.vrow[:R], seq_lens=sl, cslot=gt.cs, rslot=gt.rs, qsl=gt.qsl[:2], t2r=gt.t2r, lat=gt.lat,
                row2ir_dev=gt.row2ir_d, global_rows=R, max_seq_len=1, oi=oi)


def plan_slots(drv, st) -> int:
    """assign the view's FlashMLA
    plan slot NUMBERS (the keys ensure()'s attn_b builds would assign: every role layer whose attn_b is statically eligible,
    + the pf variant on A views with ATTN_PF) and publish s["_cxx_attn_slots"] for nd's handover (v3.17 slice 5b), so the
    native executor does not refuse every mixed step. No reset (the view's first segment start resets every slot in its
    ids: step_params / nd's fpr_attn_reset_plans), no handle build, no allocation, no GPU work. Idempotent. Returns the
    number of slots assigned over the view's states."""
    import numpy as np
    if not getattr(drv, "_attn_b_ok", False):
        return 0
    E = drv.eng
    tot = 0
    drv._cxx_slots_only = True
    try:
        for r, s in enumerate(st):
            bk = s.get("bkey")
            role = str(bk[2]) if isinstance(bk, tuple) and len(bk) > 2 else _role(drv, s)
            l20 = bool(s.get("l20"))
            layers = [L for L in E.layers if (L <= 20 if role == "A" else (L > 20 or (l20 and L == 20)))]
            pf_on = role == "A" and bool(getattr(drv.EM, "ATTN_PF", False)) and getattr(drv, "_attn_pf_ok", False)
            for L in layers:
                for pf in ((False, True) if pf_on else (False,)):
                    old = s.get("pf")
                    s["pf"] = True if pf else None
                    try:
                        AW, ws = E.lw[r][L]["AW"], s["ws"]
                        if _attn_b_eligible(drv, _EA(), AW, ws, s, s["caches"][L], int(ws["R"]), static=True):
                            _plan_key(drv, s, L, dec=pf)
                    finally:
                        s["pf"] = old
            ids = s.get("_cxx_plan_ids") or {}
            s["_cxx_attn_slots"] = np.asarray(sorted(set(int(v) for v in ids.values() if v >= 0)), dtype=np.int64)
            tot += int(s["_cxx_attn_slots"].size)
    finally:
        drv._cxx_slots_only = False
    drv.n["plan_slots_views"] += 1
    return tot


def ensure(drv, st) -> dict:
    """_ensure_body with drv._cxx_ensuring set (ENG_MIXED_CXX_ENSURE_RESET reads it)"""
    drv._cxx_ensuring = True
    try:
        return _ensure_body(drv, st)
    finally:
        drv._cxx_ensuring = False


def _ensure_body(drv, st) -> dict:
    """FPR's mixed_cxx_ensure.ensure_view: every attention handle of a mixed
    view's states, built at creation, into the cache keys handle_attn_a / handle_attn_b (/ _bi / _bo) / handle_idx /
    handle_idx_a (/ _c / _s) read, so a step never builds one lazily:
      attn_a  every layer of the view's role (A: layers <= 20, T: > 20, + 20 with ENG_L20_TAIL)
      attn_b  the no-pf variant; A views also the pf variant (ENG_ATTN_PF: the per-step pf conditions stay at handle time)
      idx     T: consumer layers 24 .. 36
    Per-step values
    (FprAttnIdxStep / FprIdxAStep / FprAttnPfStep, CP messages) are not handles: step_params / P1's C++ builders.
    Returns counts per kind (built / refused)."""
    import numpy as np
    E = drv.eng
    n = {}
    if bool(getattr(drv.EM, "ATTN_PF", False)) and getattr(drv, "_attn_pf_ok", False):   # once per device
        for r_ in range(len(st)):
            try:
                n["comb_classes"] = n.get("comb_classes", 0) + record_comb_classes(drv, r_)
            except Exception as e:
                E.log(f"mixed_cxx_attn: ensure rank {r_}: combine key classes not recorded ({e}): first-meet recording")

    def cnt(k, h):
        kk = k + ("" if h >= 0 else "_refused")
        n[kk] = n.get(kk, 0) + 1
    for r, s in enumerate(st):
        # every
        # allocation ensure makes for rank r (GeneralTables via _gt_full_dict -> hooks_idx._new_gt, idx / pf scratch, gst, ...)
        # happens on rank r's DEVICE and STEP STREAM, as the lazy in-step path does: before, the GeneralTables core of an onebk
        # base was allocated on the device's CURRENT (default) stream at view creation, so the caching allocator could hand
        # its blocks out again (growth / _onebk_sweep frees) while E.streams[r] kernels were still reading them
        with torch.cuda.device(torch.device(E.devs[r])), torch.cuda.stream(E.streams[r]):
            bk = s.get("bkey")
            role = str(bk[2]) if isinstance(bk, tuple) and len(bk) > 2 else _role(drv, s)
            s["_cxx_role"] = role
            l20 = bool(s.get("l20"))
            layers = [L for L in E.layers if (L <= 20 if role == "A" else (L > 20 or (l20 and L == 20)))]
            layer0 = layers[0] if layers else 0
            nl = (layers[-1] - layer0 + 1) if layers else 0
            H = np.full((nl, 4), -1, dtype=np.int64)          # (attn_a, idx, attn_b, attn_b_pf) per role layer
            SIG = np.zeros((nl, SIG_MAX), dtype=np.int64)     # the idx / idx_a handle's signature per layer
            SN = np.zeros(nl, dtype=np.int32)                 #   (its field count: len(IDX_SIG) / len(IDXA_SIG) / 0)
            for L in layers:
                ha = handle_attn_a(drv, r, s, L)
                H[L - layer0, 0] = ha
                cnt("attn_a", ha)
                if getattr(drv, "_attn_b_ok", False):
                    pf_on = role == "A" and bool(getattr(drv.EM, "ATTN_PF", False)) and getattr(drv, "_attn_pf_ok", False)
                    for pf in ((False, True) if pf_on else (False,)):
                        old = s.get("pf")
                        s["pf"] = True if pf else None          # (build_piece_attn_b reads only "is not None")
                        try:
                            AW, ws = E.lw[r][L]["AW"], s["ws"]
                            h = -1
                            if _attn_b_eligible(drv, _EA(), AW, ws, s, s["caches"][L], int(ws["R"]), static=True):
                                cache = s.setdefault("_cxx", {})
                                key = ("attn_b", r, L, pf)
                                if key not in cache:
                                    b = build_piece_attn_b(drv, r, s, L)
                                    cache[key] = None if b is None else drv.x.add_attn_b(b)
                                h = -1 if cache[key] is None else int(cache[key])
                        finally:
                            s["pf"] = old
                        H[L - layer0, 3 if pf else 2] = h
                        cnt("attn_b_pf" if pf else "attn_b", h)
            d = None
            try:
                d = _gt_full_dict(drv, r, s) if E.hooks.idx is not None else None
            except Exception as e:
                E.log(f"mixed_cxx_attn: ensure rank {r}: GeneralTables ({e}): idx handles stay lazy")
            if d is not None:
                old = s.get("idx")
                s["idx"] = d
                try:
                    cache = s.setdefault("_cxx", {})
                    if role == "T" and getattr(drv, "_attn_idx_ok", False):
                        for L in IDX_CONSUMERS:
                            key = ("idx", r, L)
                            if key not in cache:
                                try:
                                    bts = build_piece_attn_idx(drv, r, s, L)
                                except Exception as e:
                                    E.log(f"mixed_cxx_attn: ensure idx rank {r} layer {L}: {e}")
                                    bts = None
                                cache[key] = None if bts is None else drv.x.add_attn_idx(bts)
                            cnt("idx", -1 if cache[key] is None else 0)
                            if cache[key] is not None and 0 <= L - layer0 < nl:
                                H[L - layer0, 1] = int(cache[key])
                                g = _idx_ptrs(s, L)
                                SIG[L - layer0, : len(g)], SN[L - layer0] = g, len(g)
                    elif role == "A" and getattr(drv, "_idxa_ok", False):
                        for L in IDXA_SOURCES:
                            key = ("idx_a", r, L)
                            if key not in cache:
                                try:
                                    bts = build_piece_idx_a(drv, r, s, L)
                                except Exception as e:
                                    E.log(f"mixed_cxx_attn: ensure idx_a rank {r} layer {L}: {e}")
                                    bts = None
                                cache[key] = None if bts is None else int(drv.x.add_idx_a(bts))
                            cnt("idx_a", -1 if cache[key] is None else 0)
                            if cache[key] is not None and 0 <= L - layer0 < nl:
                                H[L - layer0, 1] = int(cache[key])
                                g = _idxa_ptrs(s, L)
                                SIG[L - layer0, : len(g)], SN[L - layer0] = g, len(g)
                        if s.get("_cxx_idxa_gst") is None:   # FprIdxAStep.gst's buffer at its final size (handover idxa_gst):
                            cap = max(4096, 2 * int(d["B"]) + 256)   # <= 2 ratio sets x (B + slices) group starts
                            with torch.cuda.device(torch.device(E.devs[r])), torch.cuda.stream(E.streams[r]):
                                s["_cxx_idxa_gst"] = torch.empty(cap, dtype=torch.int32, device=torch.device(E.devs[r]))
                            s["_cxx_idxa_gst_fixed"] = True
                finally:
                    if old is None:
                        s.pop("idx", None)
                    else:
                        s["idx"] = old
            ids = s.get("_cxx_plan_ids") or {}
            s["_cxx_attn_slots"] = np.asarray(sorted(set(int(v) for v in ids.values() if v >= 0)), dtype=np.int64)
            if os.environ.get("ENG_MIXED_CXX_EXPORT_IDX", "1") == "0":   # native steps get no idx / idx_a handle
                H[:, 1], SIG[:], SN[:] = -1, 0, 0                         #  ensure-time handles bake per-step table pointers)
            s["_cxx_attn_export"] = dict(layer0=int(layer0), n=int(nl), h=H, sig=SIG, sig_n=SN, sig_hash=dict(SIG_HASH),
                                         sig_fields=dict(idx=IDX_SIG_FIELDS, idx_a=IDXA_SIG_FIELDS))
            s["_cxx_sig_epoch"] = int(s.get("_cxx_sig_epoch", 0)) + 1
    return n


class CpMsg(C.Structure):            # pieces_attn.h FprCpMsg
    _fields_ = [("src_rank", I32), ("dst_rank", I32), ("layer", I32), ("kind", I32), ("n", I64), ("len", I64), ("off", I64),
                ("src_idx", P), ("dst_idx", P), ("src_base", P), ("dst_base", P), ("src_rows", I64), ("src_ld", I64),
                ("dst_rows", I64), ("dst_ld", I64), ("src_dev", I32), ("dst_dev", I32), ("mail_stream", P)]


CP_X1A, CP_X2 = 1, 2


def cp_step(drv, cp, layers) -> bool:
    """ENG_MIXED_CXX_CP=3: this CP step's exchange messages (engine/cp.py CPPlan: the X1a sends of x1a_send / receives of
    x1a_recv, the X2 ones of x2_send / x2_recv, for every layer of the segment, in cp.py's loop order) into pieces_attn
    (attn_cp_step), once per plan; the plan's done() (after the step's sync) then frees them (attn_cp_done). False = not
    covered (the Python exchanges run): ENG_CP_AUDIT, the old mail allocation, or a layout this mirror does not know."""
    if getattr(cp, "_cxx_cp", None) is not None:
        return cp._cxx_cp
    import sys
    CPM = sys.modules[type(cp).__module__]
    E, EM = drv.eng, drv.EM
    ok = (not CPM.AUDIT and CPM.MAIL_STREAM and drv.x.sizeof_cp_msg() == C.sizeof(CpMsg)
          and getattr(drv, "_cp_x_ok", None) is not False)
    msgs = []
    if ok:
        def dev_of(r):
            return torch.device(E.devs[r]).index

        def add(r, q, L, kind, src, dst, sv, dv, off, ln):
            assert src.dtype == torch.int64 and dst.dtype == torch.int64 and src.numel() == dst.numel(), (src, dst)
            assert sv.dim() == 2 and sv.stride(1) == 1 and dv.dim() == 2 and dv.stride(1) == 1
            m = CpMsg()
            m.src_rank, m.dst_rank, m.layer, m.kind = r, q, L, kind
            m.n, m.len, m.off = src.numel(), ln, off
            m.src_idx, m.dst_idx, m.src_base, m.dst_base = _p(src), _p(dst), _p(sv), _p(dv)
            m.src_rows, m.src_ld, m.dst_rows, m.dst_ld = sv.shape[0], sv.stride(0), dv.shape[0], dv.stride(0)
            m.src_dev, m.dst_dev = dev_of(r), dev_of(q)
            m.mail_stream = P(cp._mail_stream(E.devs[q]).cuda_stream)
            msgs.append(m)
        idx_on = E.hooks.idx is not None
        for L in layers:
            if L > 20:
                continue
            if idx_on and L in EM.INDEX_LAYERS and L in EM.KV_SOURCE_LAYERS:   # x1a_send (r -> every later rank)
                for g, gd in cp.groups.items():
                    rk = gd["ranks"]
                    for i, r in enumerate(rk):
                        for q in rk[i + 1:]:
                            for k, (o, n) in enumerate(cp._sections(L)):
                                add(r, q, L, CP_X1A, gd["pages"][r][r], gd["pages"][q][r], cp._pg(r), cp._pg(q), o, n)
            for g, gd in cp.groups.items():                                    # x2_send (r -> the next rank)
                rk = gd["ranks"]
                for r in rk:
                    if r not in gd["halo_src"]:
                        continue
                    q = rk[rk.index(r) + 1]
                    assert q in gd["halo_dst"], (g, r, q)
                    add(r, q, L, CP_X2, gd["halo_src"][r], gd["halo_dst"][q], cp._wsec(r), cp._wsec(q), L * cp.WLB, cp.WLB)
        ok = len(msgs) <= 1024
    drv.n["cp_step_ok" if ok else "cp_step_python"] += 1
    k_ = drv.n["cp_step_ok"] + drv.n["cp_step_python"]
    if k_ <= 3 or k_ % 100 == 0:
        drv.eng.log(f"mixed_cxx_attn: CP step {k_}: {'pieces' if ok else 'Python exchanges'}, {len(msgs)} messages, "
                    f"layers {min(layers) if layers else -1}..{max(layers) if layers else -1} "
                    f"(pieces {drv.n['cp_step_ok']}, Python {drv.n['cp_step_python']})")
    if ok:
        drv.x.attn_cp_step(b"".join(bytes(m) for m in msgs), [int(sm.cuda_stream) for sm in E.streams])
        cp._cxx_keep = msgs                                  # (the index tensors are the plan's own: alive with it)
        orig = cp.done

        def done():
            try:
                orig()
            finally:
                drv.x.attn_cp_done()
        cp.done = done
        drv.n["cp_msgs"] += len(msgs)
    cp._cxx_cp = ok
    return ok


# ---- FPR_HANDLE_AUDIT=1: a deterministic dangling-pointer audit of every handle a builder returns. After the
# builder's frame is gone (its scratch locals freed), every device pointer the handle captured -- c_void_p fields, recursively
# through nested structs / arrays, plus the recorded kernel argument arrays (*_vals) -- is located in
# torch.cuda.memory_snapshot(): inside an ACTIVE block = ok; inside a freed / cached / pending-free block = DANGLING (logged
# with the field name, counted in drv.n["audit_bad"]); outside every torch segment = external (weights from C++ libs,
# pinned host, ...). Plain integers in *_vals never sit inside a freed block of the right device by accident often enough to
# matter; they are only reported when they do.
def _audit_ptrs(obj, prefix="", out=None):
    out = [] if out is None else out
    for name, ty in getattr(type(obj), "_fields_", []):
        v = getattr(obj, name)
        path = prefix + name
        if ty is C.c_void_p:
            if v:
                out.append((path, int(v), True))
        elif isinstance(v, C.Structure):
            _audit_ptrs(v, path + ".", out)
        elif isinstance(v, C.Array):
            if len(v) and isinstance(v[0], C.Structure):
                for i, e in enumerate(v):
                    _audit_ptrs(e, f"{path}[{i}].", out)
            elif name.endswith("_vals"):
                # argument slots the C++ replay OVERWRITES every launch are not captured pointers: the stock
                # branch's select / score kernels (pieces_attn: v[bs_i[0..2]] = logits / lens / scores, v[sc_i[0..2]] =
                # values / indices / output), recorded on the builder's own dummies
                skip = set()
                if name == "bs_vals" and hasattr(obj, "bs_i"):
                    skip = {int(obj.bs_i[j]) for j in range(3)}
                elif name == "sc_vals" and hasattr(obj, "sc_i"):
                    skip = {int(obj.sc_i[j]) for j in range(3)}
                for i, e in enumerate(v):
                    if i in skip:
                        continue
                    if isinstance(e, int) and e > (1 << 32):
                        out.append((f"{path}[{i}]", int(e), False))
    return out


def _audit_blocks(dev_index):
    segs = []
    for sg in torch.cuda.memory_snapshot():
        if int(sg.get("device", -1)) != dev_index:
            continue
        for bk in sg.get("blocks", []):
            a = int(bk["address"])
            segs.append((a, a + int(bk["size"]), str(bk.get("state", "?"))))
    segs.sort()
    return segs


def _audit(drv, r, L, kind, st):
    import bisect
    dev = torch.device(drv.eng.devs[r]).index
    torch.cuda.synchronize(torch.device("cuda", dev))   # (diag only: pending frees resolve; nothing races the snapshot)
    blocks = _audit_blocks(dev)
    starts = [b[0] for b in blocks]
    n_ok = n_ext = n_bad = 0
    for path, v, is_ptr in _audit_ptrs(st):
        i = bisect.bisect_right(starts, v) - 1
        if i >= 0 and blocks[i][0] <= v < blocks[i][1]:
            if blocks[i][2] == "active_allocated":
                n_ok += is_ptr
            else:
                n_bad += 1
                drv.n["audit_bad"] += 1
                drv.eng.log(f"mixed_cxx_attn: HANDLE AUDIT {kind} rank {r} layer {L}: {path} = {v:#x} in a {blocks[i][2]} "
                            f"block [{blocks[i][0]:#x}, +{blocks[i][1] - blocks[i][0]}): DANGLING")
        elif is_ptr:
            n_ext += 1
    drv.n["audit_builds"] += 1
    if n_bad or drv.n["audit_builds"] <= 8:
        drv.eng.log(f"mixed_cxx_attn: HANDLE AUDIT {kind} rank {r} layer {L}: {n_ok} active, {n_ext} external, {n_bad} DANGLING")


def _audited(kind, typ):
    def deco(fn):
        def w(drv, r, s, L):
            bl = os.environ.get("FPR_BUILD_LOG", "0") == "1"   # no sync, no snapshot: print only, stderr
            if bl:                                             # (the stream fpr_lt's WARM_SYNC lines use: they interleave)
                sys.stderr.write(f"mixed_cxx_attn: BUILD {kind} rank {r} layer {L} R={s.get('R')} start\n")
                sys.stderr.flush()
            b = fn(drv, r, s, L)
            if bl:
                sys.stderr.write(f"mixed_cxx_attn: BUILD {kind} rank {r} layer {L} R={s.get('R')} done "
                                 f"({'ok' if b is not None else 'refused'})\n")
                sys.stderr.flush()
            if b is not None and os.environ.get("FPR_HANDLE_AUDIT", "0") == "1":
                try:
                    _audit(drv, r, L, kind, typ().from_buffer_copy(b))
                except Exception as e:
                    drv.eng.log(f"mixed_cxx_attn: HANDLE AUDIT {kind} rank {r} layer {L}: audit failed ({e})")
            return b
        w.__wrapped__ = fn
        return w
    return deco


build_piece_attn_a = _audited("attn_a", lambda: PieceAttnA)(build_piece_attn_a)
build_piece_attn_b = _audited("attn_b", lambda: PieceAttnB)(build_piece_attn_b)
build_piece_attn_idx = _audited("idx", lambda: _piece_idx_type())(build_piece_attn_idx)
build_piece_idx_a = _audited("idx_a", lambda: _piece_idxa_type())(build_piece_idx_a)
