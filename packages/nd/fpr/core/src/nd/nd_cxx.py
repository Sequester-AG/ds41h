"""ENG_D13_NATIVE=1: D13._pre's device ops per rank
through the C++ of fpr/core/src/nd (module fpr_nd).

    install()    patches engine.d13.D13._pre

What moves to C++ (nd.cpp, per rank, same order on the same stream): the full path's _deltas (unless the index mirror is on
or nu > DELTA_CAP: then the Python body runs first), Engine._upload_v2(dev_pos=True)'s blob, hooks_idx.upload_d13's blob, the
owner's rslot base copy, RowExp.upload, _guard_host + graph T, the positions D2H, Engram's step args + upload (async decode
rows need no host lookback); the owner path's graph T' + positions D2H. What stays Python: every decision (_empty_*, _owner_sig), first-use allocations and graph captures (before the native
call; allocations only, no copies, so stream order is unchanged), the double-buffer bookkeeping (Python and the native path
share each set's alternation: the C++ waits the holder's event of the set and re-records a persistent event that then
becomes the holder), host-side fields, and every path ND-1 does not cover (ENG_D13_GT off,
CHECK / GT_CHECK / OWNER_CHECK, TFUSE, the index mirror): those run D13's own Python body.
Gate: the harness arms ENG_D13_NATIVE=1 vs 0 must be ALL IDENTICAL (every step's h_md5 / tok_md5, the streams).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.normpath(os.path.join(HERE, "..", ".."))
ON = os.environ.get("ENG_D13_NATIVE", "0") == "1"
_X = None
_ORIG = {}
STATS = {"full": 0, "owner": 0, "python_full": 0, "python_owner": 0}


def ext():
    global _X
    if _X is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "nd" + os.environ.get("ND_BUILD_SUFFIX", ""))
        os.makedirs(bd, exist_ok=True)
        srcs = [os.path.join(HERE, "nd_py.cpp"), os.path.join(HERE, "nd.cpp"), os.path.join(HERE, "nd_handover.cpp"), os.path.join(CORE, "src", "d13", "d13_host.cpp"),
                os.path.join(CORE, "src", "d13", "hsd_prep.cpp"), os.path.join(CORE, "src", "tables", "tables.cpp")]
        _X = load(name="fpr_nd", sources=srcs, extra_include_paths=[os.path.join(CORE, "include")], build_directory=bd,
                  extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"], with_cuda=True,
                  verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
    return _X


_X2 = None


def ext2():
    """ND-2's in-engine gate module fpr_nd2: nd::Engine (nd_engine.cpp + nd_hist.cpp) with ND-1 and the
    D13 host, bound as Engine2b (nd2_py.cpp); Separate from ext() (ND-1)."""
    global _X2
    if _X2 is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "nd2" + os.environ.get("ND_BUILD_SUFFIX", ""))
        os.makedirs(bd, exist_ok=True)
        srcs = [os.path.join(HERE, x) for x in ("nd2_py.cpp", "nd_engine.cpp", "nd_hist.cpp", "nd.cpp", "nd_mixed.cpp")] + \
            [os.path.join(CORE, "src", "d13", "d13_host.cpp"), os.path.join(CORE, "src", "d13", "hsd_prep.cpp"),
             os.path.join(CORE, "src", "tables", "tables.cpp"), os.path.join(CORE, "src", "p0", "head_build.cpp"),
             os.path.join(CORE, "src", "p0", "inputs_build.cpp")]   # (v3+: nd_mixed + the P0 builders it calls)
        _X2 = load(name="fpr_nd2", sources=srcs, extra_include_paths=[os.path.join(CORE, "include")], build_directory=bd,
                   extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"], with_cuda=True,
                   verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
    return _X2


def _cfg(d13, M) -> dict:   # = d13_cxx._cfg (stage 2a)
    eng = d13.eng
    return dict(graphs=bool(eng.cfg.graphs), largest_bucket=int(eng.cfg.buckets[-1]), graph_max_ctx=int(d13.E.GRAPH_MAX_CTX),
                k1=int(M.K1), delta_cap=int(M.DELTA_CAP), fb_reuse=bool(M.FB_REUSE), scratch=[int(x.scratch) for x in d13.rx])


def _nd(d13, M):
    x = getattr(d13, "_nd_x", None)
    if x is None:
        x = d13._nd_x = ext().Nd(_cfg(d13, M))
        d13.eng.log("engine D13: ENG_D13_NATIVE on (stage 2b ND-1: _pre's device ops through fpr_nd)")
    return x


def _ev(d13, key, stream):
    """A persistent event per (purpose, set, ...): recorded once at creation so its handle exists; the C++ re-records it."""
    t = d13.__dict__.setdefault("_nd_ev", {})
    e = t.get(key)
    if e is None:
        e = torch.cuda.Event()
        e.record(stream)
        t[key] = e
    return e


def _h(e) -> int:
    return 0 if e is None else int(e.cuda_event)


def _native_ok(d13, M) -> bool:
    H = d13.eng.hooks
    return (M.GT and not M.CHECK and not M.GT_CHECK and H.idx is not None and not getattr(H.idx, "mir", None)
            and hasattr(H.idx, "upload_d13_host"))


def _gt_ensure(d13, r, s, R, has_seqs):
    """_gt_run's capture half (capture on the side stream when the key is new or a bound buffer moved), no replay."""
    k = (R, r, has_seqs)
    ptrs = d13._gt_ptrs(r, s)
    ent = d13.gt.get(k)
    if ent is None or ent[1] != ptrs:
        if ent is not None:
            d13.stats["gt_recaptures"] += 1
            d13.eng.log(f"engine D13 GT: recapture graph T {k} (a bound buffer moved)")
        dev = d13.eng.devs[r]
        if d13.gt_pool[r] is None:
            with torch.cuda.device(dev):
                d13.gt_pool[r] = torch.cuda.graph_pool_handle()
                d13.gt_stream[r] = torch.cuda.Stream(device=dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.device(dev), torch.cuda.stream(d13.gt_stream[r]):
            g.capture_begin(pool=d13.gt_pool[r], capture_error_mode="thread_local")
            try:
                d13._gt_body(r, s, R, has_seqs)
            finally:
                g.capture_end()
        ent = d13.gt[k] = (g, ptrs)
        d13.stats["gt_captures"] += 1
        if d13.stats["gt_captures"] <= 64:
            d13.eng.log(f"engine D13 GT: graph T captured for (R, rank, seqs) = {k}")
    return ent[0]


def _gto_ensure(d13, r, s, R, n):
    """_gto_run's graph T' capture half (non-TFUSE), no replay."""
    H = d13.eng.hooks
    k = (R, r, "own", n)
    ptrs = d13._gt_ptrs(r, s) + (s["_own_rbase"].data_ptr(), s["_own_sbase"].data_ptr(), H.engram[r].b_start.data_ptr(),
                                 H._dt_d[r].data_ptr(), H.hsd.next_ids[r].data_ptr(), s["tok"].data_ptr())
    ent = d13.gt.get(k)
    if ent is None or ent[1] != ptrs:
        if ent is not None:
            d13.stats["gt_recaptures"] += 1
        dev = d13.eng.devs[r]
        if d13.gt_pool[r] is None:
            with torch.cuda.device(dev):
                d13.gt_pool[r] = torch.cuda.graph_pool_handle()
                d13.gt_stream[r] = torch.cuda.Stream(device=dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.device(dev), torch.cuda.stream(d13.gt_stream[r]):
            g.capture_begin(pool=d13.gt_pool[r], capture_error_mode="thread_local")
            try:
                d13._gto_body(r, s, R, n)
            finally:
                g.capture_end()
        d13.gt[k] = (g, ptrs)
        d13.stats["gt_captures"] += 1
        if d13.stats["gt_captures"] <= 64:
            d13.eng.log(f"engine D13 GT: graph T' {'re' if ent is not None else ''}captured for (R, rank, own, rows) = {k}")
        ent = d13.gt[k]
    return ent[0]


def _tab(M, sb, ct):
    n = sb.n_rows
    tok = torch.from_numpy(np.asarray(ct.tokens[:n], dtype=np.int64)) if n else torch.zeros(0, dtype=torch.int64)
    return M.NS(n=n, tokens=tok, sample_rows=list(np.asarray(ct.sample_idx).tolist()), seq={},
                seq_rows=[(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs])


def _common(d13, r, s, R, cur):
    ph, oh = d13.d2h[d13.d2h_i][r]
    ev = _ev(d13, ("pos", R, r, d13.d2h_i), cur)
    return dict(rank=r, device=torch.device(d13.eng.devs[r]).index, stream=int(cur.cuda_stream), R=R, ph=ph.data_ptr(),
                oh=oh.data_ptr(), pos_d=s["pos"].data_ptr(), own_d=s["own"].data_ptr(), pos_ev=_h(ev)), ev


def _owner_native(d13, M, r, s, sb, ct, R) -> bool:
    """D13._owner_skip with the device part (graph T' + positions D2H) in C++."""
    if M.OWNER_CHECK or M.TFUSE or M.TFUSE_CHECK:
        STATS["python_owner"] += 1
        return d13._owner_skip(r, s, sb, ct, R)
    eng, H = d13.eng, d13.eng.hooks
    rx = d13.rx[r]
    n = sb.n_rows
    with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
        cur = torch.cuda.current_stream()
        tab = _tab(M, sb, ct)
        s["n"] = n
        s["n_sample"] = len(tab.sample_rows)
        H.idx.upload_d13_host(r, s, sb)
        rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(int(s["R"]), M.SW), sl=s["sl"], paged=s["paged_d"])
        g = _gto_ensure(d13, r, s, R, n)
        d13.stats["gt_replays"] += 1
        c, ev = _common(d13, r, s, R, cur)
        c["gto_exec"] = int(g.raw_cuda_graph_exec())
        _nd(d13, M).pre_owner(c)
        s["d13_ev"] = ev
        s["tab"], s["batch"] = tab, sb
    d13.stats["owner_skips"] += 1
    STATS["owner"] += 1
    return True


def _full_native(d13, M, r, s, sb, ct, R, osig) -> None:
    """D13._pre's full path for rank r (ENG_D13_GT): the device ops in C++, everything else as the Python body."""
    eng, H = d13.eng, d13.eng.hooks
    EM = sys.modules[type(eng).__module__]          # engine.engine (_in_layout / _in_views)
    IM = sys.modules[type(H.idx).__module__]         # hooks_idx (_up_fields / _up_views)
    rx = d13.rx[r]
    with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
        cur = torch.cuda.current_stream()
        # _deltas: what the native path does not take runs the Python body first (same stream position: first)
        has_upd = ct.bt_upd_slot is not None and len(ct.bt_upd_slot)
        nu = int(np.asarray(ct.bt_upd_len, np.int64).sum()) if has_upd else 0
        py_dl = nu > M.DELTA_CAP
        if py_dl:
            d13._deltas(r, ct)
        n = sb.n_rows
        tab = _tab(M, sb, ct)
        # ---- Engine._upload_v2 (dev_pos): the bucket's lazy pinned sets (as Python creates them)
        if s.get("in_h") is None:
            nb = s["in_d"].numel()
            s["in_h"] = [[torch.zeros(nb, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
            s["in_hv"] = [EM._in_views(x[0], R, numpy=True) for x in s["in_h"]]
            s["in_i"] = 0
        i_in = s["in_i"]
        in_hb, in_hold = s["in_h"][i_in]
        lay, _ = EM._in_layout(int(s["R"]))   # rows; R may be a long KEY
        ev_in = _ev(d13, ("in", R, r, i_in), cur)
        # ---- hooks_idx.upload_d13: lazy pinned sets, the dev_bt check, host fields (upload_d13_host = the same host tail)
        d = s["idx"]
        B = d["B"]
        assert "up_d" in d, "D13 needs upload_v2 buckets (EngineConfig.upload_v2)"
        if d["up_h"] is None:
            nbytes = d["up_d"].numel()
            d["up_h"] = [[torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
            d["up_hv"] = [IM._up_views(x[0], R, B, numpy=True) for x in d["up_h"]]
            d["up_i"] = 0
        cap = d.get("dev_bt_captured")
        if cap is not None and cap is not True:
            raise RuntimeError(f"idx2f hook rank {r}: decode graph captured with dev_bt={cap}, step has dev_bt=True")
        H.idx.upload_d13_host(r, s, sb)
        uoff, off = {}, 0
        for nm, dt, cnt in IM._up_fields(int(s["R"]), B):
            uoff[nm] = off
            off += -(-(cnt * np.dtype(dt).itemsize) // 8) * 8
        i_up = d["up_i"]
        up_hb, up_hold = d["up_h"][i_up]
        ev_up = _ev(d13, ("up", R, r, i_up), cur)
        # ---- the owner's rslot base: first use allocates (Python's _keep clones = allocation + this copy)
        rslot = d["rslot"]
        rbase = None
        if osig is not None:
            buf = s.get("_own_rbase")
            if buf is None or buf.shape != rslot.shape or buf.dtype != rslot.dtype:
                s["_own_rbase"] = torch.empty_like(rslot)
            rbase = s["_own_rbase"]
            assert rslot.is_contiguous() and rbase.is_contiguous()
        # ---- RowExp.upload: the set (Python toggles _hi before its wait)
        i_rx = rx._hi
        rx._hi ^= 1
        rx_hold = rx._ev[i_rx]
        ev_rx = _ev(d13, ("rx", r, i_rx), cur)
        if "paged_d" not in s:
            s["paged_d"] = torch.full((int(s["R"]),), -1, dtype=torch.int64, device=s["dev"])
        rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(int(s["R"]), M.SW), sl=s["sl"], paged=s["paged_d"])
        # ---- guard + graph T
        S_ = len(sb.seqs)
        g = _gt_ensure(d13, r, s, R, S_ > 0)
        i = d13.d2h_i
        nb_hold = d13.nb_ev[i][r]
        ev_nb = _ev(d13, ("nb", i, r), cur)
        # ---- deltas set
        i_dl = d13.dl_i
        dl_hold = d13.dl_ev[i_dl][r]
        ev_dl = _ev(d13, ("dl", i_dl, r), cur)
        hr, hc, hp = d13.dl_h[i_dl][r]
        dr, dc, dp = d13.dl_d[r]
        dbt = rx.dbt
        # ---- Engram's step args + upload (async decode rows: no host lookback): its stage set (Python toggles before the
        # wait) and the set's persistent event (Python creates it on first use: created now if missing)
        eg = H.engram[r] if H.engram is not None else None
        egd = dict(engram=0)
        if eg is not None:
            EGM = sys.modules[type(eg).__module__]
            for j in range(2):
                if eg._stage_ev[j] is None:
                    eg._stage_ev[j] = torch.cuda.Event()
                    eg._stage_ev[j].record(cur)
            i_eg = eg._stage_i
            eg._stage_i ^= 1
            o_ = eg._pk_off
            egd = dict(engram=1, eg_stage_h=eg._stage[i_eg].data_ptr(), eg_ev=_h(eg._stage_ev[i_eg]), eg_pk_d=eg._pk.data_ptr(),
                       eg_pk_n=int(eg._pk_n), eg_off_row0=o_["row0"], eg_off_nrows=o_["nrows"], eg_off_start=o_["start"],
                       eg_off_slot=o_["slot"], eg_off_look=o_["look"], eg_off_counts=o_["counts"], eg_max_rows=int(eg.max_rows),
                       eg_max_seqs=int(eg.max_seqs), eg_ring_w=int(EGM.RING_W))
        c, ev_pos = _common(d13, r, s, R, cur)
        c.update(egd)
        c.update(hsd_on=int(H.hsd is not None), skip_deltas=int(py_dl),
                 dl_rows_h=hr.data_ptr(), dl_cols_h=hc.data_ptr(), dl_pages_h=hp.data_ptr(), dl_ev_wait=_h(dl_hold), dl_ev=_h(ev_dl),
                 dl_rows_d=dr.data_ptr(), dl_cols_d=dc.data_ptr(), dl_pages_d=dp.data_ptr(), delta_cap=int(M.DELTA_CAP),
                 dbt=dbt.data_ptr(), dbt_rows=dbt.shape[0], dbt_cols=dbt.shape[1], dbt_stride0=dbt.stride(0),
                 in_h=in_hb.data_ptr(), in_ev_wait=_h(in_hold), in_ev=_h(ev_in), in_d=s["in_d"].data_ptr(),
                 in_bytes=s["in_d"].numel(), in_off_tok=lay["tok"][0], in_off_valid=lay["valid"][0], in_off_samp=lay["samp"][0],
                 up_h=up_hb.data_ptr(), up_ev_wait=_h(up_hold), up_ev=_h(ev_up), up_d=d["up_d"].data_ptr(),
                 up_bytes=d["up_d"].numel(), up_off_valid=uoff["valid"], up_off_pslot=uoff["pslot"], up_off_rs=uoff["rs"], up_B=B,
                 pslot_fill=H.idx.views[r]["dbt"].shape[0] - 1,
                 rslot_d=rslot.data_ptr(), rbase_d=0 if rbase is None else rbase.data_ptr(),
                 rslot_bytes=rslot.numel() * rslot.element_size(),
                 rx_h32=rx._h[i_rx][0].data_ptr(), rx_h64=rx._h[i_rx][1].data_ptr(), rx_ev_wait=_h(rx_hold), rx_ev=_h(ev_rx),
                 rx_b32=rx._b32.data_ptr(), rx_b64=rx._b64.data_ptr(), S_max=rx.S_max, ops_max=rx.ops_max, R_max=rx.R_max,
                 nb_h=d13.nb_h[i][r].data_ptr(), nb_ev_wait=_h(nb_hold), nb_ev=_h(ev_nb), nb_d=d13.nb_d[r].data_ptr(),
                 gt_exec=int(g.raw_cuda_graph_exec()))
        _nd(d13, M).pre_full(c, sb, ct)
        # ---- Engine._upload_v2's tail: Hooks.device_tokens (HSD's next ids -> the decode rows' tok; creates H._dt_d on
        # first use, which graph T' bakes). Python runs it right after the in_d copy; here after pre_full's ops on the same
        # stream. Nothing in between reads or writes tok / _dt_d (upload_d13, rowexp upload, graph T, positions D2H,
        # Engram's step-args upload), so the same bytes reach the same buffers
        H.device_tokens(r, s, tab)
        # ---- the Python bookkeeping of what the C++ just did
        if not py_dl and nu > 0:
            d13.dl_ev[i_dl][r] = ev_dl
        s["in_h"][i_in][1] = ev_in
        s["in_i"] = i_in ^ 1
        s["n"] = n
        s["n_sample"] = len(tab.sample_rows) if n else 0
        d["up_h"][i_up][1] = ev_up
        d["up_i"] = i_up ^ 1
        rx._ev[i_rx] = ev_rx
        if S_:
            d13.nb_ev[i][r] = ev_nb
        d13.stats["gt_replays"] += 1
        s["d13_ev"] = ev_pos
        s["tab"], s["batch"] = tab, sb
    STATS["full"] += 1


def native_pre(self, st, batches, cts, R) -> None:
    """D13._pre (steps 1-3 of dispatch) with the device ops of the full and owner paths in C++."""
    M = sys.modules[type(self).__module__]
    if not _native_ok(self, M):
        STATS["python_full"] += 1
        return _ORIG["_pre"](self, st, batches, cts, R)
    eng = self.eng
    eng._batches = batches
    self.d2h_i ^= 1
    self.dl_i ^= 1
    tr = [] if M.PRE_T else None
    for r in range(M.W):
        if M.PRE_T:
            tr.append(time.perf_counter())
        s, sb, ct = st[r], batches[r], cts[r]
        el = None
        if M.EMPTY:
            el = self._empty_eligible(sb, ct)
            if el and self._r_empty[r] and self._b_empty.get((R, r)) == self._epoch:
                self._empty_skip(r, s, sb, R)
                continue
        osig = self._owner_sig(r, sb, ct) if M.OWNER else None
        if osig is not None and self._own_last[r] == (R, osig, self._epoch):
            if _owner_native(self, M, r, s, sb, ct, R):
                continue
        _full_native(self, M, r, s, sb, ct, R, osig)
        with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
            if M.EMPTY:
                self._empty_mark(r, s, R, el)
            if M.OWNER:
                self._owner_mark(r, s, R, osig)
    if M.PRE_T:
        tr.append(time.perf_counter())
        self._pre_r = [round((b_ - a_) * 1000, 3) for a_, b_ in zip(tr, tr[1:])]


def install() -> None:
    """Patch engine.d13.D13._pre (idempotent). Only under ENG_D13_NATIVE=1."""
    if not ON:
        return
    import engine.d13 as D
    if "_pre" not in _ORIG:
        _ORIG["_pre"] = D.D13._pre
        D.D13._pre = native_pre
        import atexit
        atexit.register(lambda: print(f"nd_cxx: steps per path {STATS}", flush=True))
    print("nd_cxx: ENG_D13_NATIVE=1: D13._pre patched (ND-1)", flush=True)
