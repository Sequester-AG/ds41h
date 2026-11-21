"""PHYS_CXX_D13=1: D13's host decisions and HSD's host prepare in
C++ (fpr/core/src/d13: d13_host.cpp, hsd_prep.cpp; binding d13_py.cpp = module fpr_d13_cxx) inside today's engine.

    install(d13)    called by D13.__init__ (engine/d13.py) under PHYS_CXX_D13=1, or by the CPU harness
On that D13 instance only (instance attributes shadow the class methods):
  _async_ok        -> Host.async_ok        (the same `why` strings)
  _owner_sig       -> Host.owner_sig       (+ engine.prev_of for the rank's rids; the same tuple, se.rid objects)
  _empty_eligible  -> Host.empty_eligible
  _delta_rows      -> Host.delta_rows      (_deltas' host half; mirror / staging / device copies stay Python)
  _fb_rows         -> Host.fb_rows         (_fallback's host half; dataclasses.replace of the device-resolved entries)
and on its HSD (hooks.hsd, the vectorized prep "1"):
  _build           -> Host.hsd_build: C++ writes HSD's own pinned host blobs (both sets, all ranks: bound by pointer) and
                      the bitmask staging (_bm_h); Python keeps hsd._hi (passed in, flipped by _upload as before), hsd.cur
                      (_Step rebuilt from the C++ record with the Python rid objects), the slot -> state-object references
                      (hsd._slot_key: they keep the identities the C++ compares alive) and _res.
Gate: fpr/core/tests/record_d13_cpu.py with PHYS_CXX_D13=1 vs 0 on the 6 CPU workloads -> identical d13.jsonl / hsd.jsonl
(fpr/core/tests/cmp_gtrace.py), plus the C++ replays.
"""
from __future__ import annotations

import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.normpath(os.path.join(HERE, "..", ".."))
_X = None


def ext():
    global _X
    if _X is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "d13_cxx")
        os.makedirs(bd, exist_ok=True)
        _X = load(name="fpr_d13_cxx", sources=[os.path.join(HERE, f) for f in ("d13_py.cpp", "d13_host.cpp", "hsd_prep.cpp")]
                  + [os.path.join(CORE, "src", "tables", "tables.cpp")], extra_include_paths=[os.path.join(CORE, "include")],
                  build_directory=bd, extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"], with_cuda=False,
                  verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
    return _X


def _cfg(d13) -> dict:
    M = sys.modules[type(d13).__module__]
    eng = d13.eng
    return dict(graphs=bool(eng.cfg.graphs), largest_bucket=int(eng.cfg.buckets[-1]), graph_max_ctx=int(d13.E.GRAPH_MAX_CTX),
                k1=int(M.K1), delta_cap=int(M.DELTA_CAP), fb_reuse=bool(M.FB_REUSE), scratch=[int(x.scratch) for x in d13.rx])


def _hcfg(hsd) -> dict:
    M = sys.modules[type(hsd).__module__]
    HS = M if hasattr(M, "_Step") else sys.modules["hsd"]
    return dict(k=int(hsd.k), max_seqs=int(hsd.max_seqs), seq_buckets=[int(x) for x in hsd.seq_buckets],
                row_buckets=[int(x) for x in hsd.row_buckets], nt_max=int(hsd.ranks0[0].b["aux"].shape[0]),
                pool=bool(hsd.pool), window=int(HS.DR.WINDOW), ring=int(HS.DR.RING))


def install(d13) -> None:
    if getattr(d13, "_cxx", None) is not None:
        return
    X = ext()
    hsd = d13.eng.hooks.hsd
    use_hsd = hsd is not None and getattr(hsd, "prep", "1") != "0" and hasattr(hsd, "_hblob")
    host = X.Host(_cfg(d13), _hcfg(hsd) if use_hsd else {})
    d13._cxx = host
    eng = d13.eng

    def _async_ok(batches, cts):
        return host.async_ok(batches, cts, bool(eng.cfg.graphs))

    def _owner_sig(r, sb, ct):
        prev = []
        for se in sb.seqs:
            p = eng.prev_of(se.rid)
            prev.append((str(se.rid), p is not None, int(p[0]) if p else -1, int(p[1]) if p else -1))
        s = host.owner_sig(r, sb, ct, prev)
        if s is None:
            return None
        n, seqs, sidx = s
        return (n, tuple((sb.seqs[k].rid, sl, r0, nr, nb, pi) for k, sl, r0, nr, nb, pi in seqs), tuple(sidx))

    def _fb_rows(batches, starts):
        out = []
        for sb, (upd, S2, pos, pg, ws) in zip(batches, host.fb_rows(batches, starts)):
            seqs = list(sb.seqs[:S2])
            for k, st_, nc, nb in upd:
                seqs[k] = dataclasses.replace(seqs[k], start=st_, num_computed=nc, n_blocks=nb)
            out.append(dataclasses.replace(sb, seqs=seqs, positions=pos, paged_slot=pg, window_slot=ws))
        return out

    d13._async_ok = _async_ok
    d13._owner_sig = _owner_sig
    d13._empty_eligible = host.empty_eligible
    d13._delta_rows = host.delta_rows
    d13._fb_rows = _fb_rows
    if use_hsd:
        install_hsd(hsd, host)
    print(f"d13_cxx: PHYS_CXX_D13 on (C++ D13 host{' + HSD prepare' if use_hsd else ''})", flush=True)


def install_hsd(hsd, host) -> None:
    import torch
    M = sys.modules[type(hsd).__module__]
    HS = M if hasattr(M, "_Step") else sys.modules["hsd"]
    lay = [(nm, off, hsd._dims[dim] * cols) for nm, (off, dt, dim, cols) in hsd._lay.items()]
    got = [tuple(x) for x in host.hsd_layout()]
    assert got == lay and host.hsd_bytes() == hsd._nbytes and host.hsd_shared_end() == hsd._shared_end, \
        f"d13_cxx: HSD blob layout differs: {got[:3]} vs {lay[:3]}"
    if hsd._bm_h is None:                  # (_build creates it on the first bitmask step; the C++ writes into it)
        hsd._bm_h = torch.empty(hsd._dims["T"], HS.BM_WORDS, dtype=torch.int32, pin_memory=torch.cuda.is_available())
    host.hsd_bind([[hsd._hblob[i][r].data_ptr() for r in range(HS.W)] for i in range(2)], hsd._bm_h.data_ptr())
    host.hsd_restore(hsd._temp_np, hsd._seed_np, [0 if x is None else id(x) for x in hsd._slot_key])
    MS1 = hsd.max_seqs + 1

    def _build(batches, states, bitmasks):
        st_l = []
        for b in batches:
            srl, ss = b.sample_rows, b.seq_slot
            dr = {ss[i] for i in srl} if srl else set()
            row = []
            for k, se in enumerate(b.seqs):
                if k in dr:
                    s = states[se.rid]
                    na = getattr(s, "num_acceptable_drafts", None)
                    row.append((float(getattr(s, "temperature", 1.0)), int(getattr(s, "seed", 0)),
                                bool(getattr(s, "drafts_valid", True)), None if na is None else int(na), id(s)))
                else:
                    row.append(None)
            st_l.append(row)
        bm = bitmasks if bitmasks is not None and any(x is not None for x in bitmasks) else None
        d = host.hsd_build(int(hsd._hi), batches, st_l, bm)
        if d is None:
            return None
        dmap = [[(int(si), int(qi)) for si, qi in ds] for ds in d["dseqs"]]
        rids = [None] * d["n_req"]
        for r, ds in enumerate(dmap):
            for si, qi in ds:
                se = batches[r].seqs[si]
                rids[qi] = se.rid
                hsd._slot_key[r * MS1 + se.state_slot] = states[se.rid]   # keeps the compared identity alive
        hsd.cur = HS._Step(d["n_s"], d["S_pad"], d["D_pad"], d["n_req"], d["T"], dmap, rids, d["n_seq"], d["n_t"], d["step"])
        hsd.cur.NT = d["NT"]
        hsd.bm_any = d["has_bm"]
        hsd._res = [None] * HS.W
        return (d["hi"], (d["S_pad"], d["D_pad"], d["NT"]), d["T"], d["has_bm"])

    hsd._build = _build
