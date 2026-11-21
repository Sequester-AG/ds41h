"""D13 steady-state decode, stage 1.

In a steady decode step (each rank's drafting requests the same as in the previous step and in the same order, all
decode, no grammar, the same HSD bucket key), HSD's host prepare changes only the position-dependent fields of its
per-rank blob (g_p0, pos, wbase, wpages, tpos, tslot). kernels/d13s/hsdprep.cu derives them on the device from the
previous step's num_sampled and rowexp's ring / pos / own.

ENG_D13S=shadow (debug, default off): the host prepare runs as always; in steady steps the device version is computed
into a snapshot of the previous step's blob and compared field by field (the bucket's prefixes) with the blob the host
built and uploaded: the kernel's fields AND every field the device path would leave untouched. Synchronizes; counts /
logs.

ENG_D13S=on: in a steady step the host does not wait for step N inside dispatch(N+1) and does not prepare HSD:
  - N's results (num_sampled / sampled of the ranks with drafting requests) are copied on the extraction stream after
    N's done event into pinned buffers (a ring of 4 sets); N's handle builds its StepResults from them later (the same
    formula as HSD.result, with N's own _Step snapshot), when vLLM merges it;
  - the step stream waits for those copies (graph H of N+1 rewrites num_sampled / sampled), then kernels/d13s/hsdprep.cu
    derives N+1's HSD fields in place in the live blob; HSD's host state: cur = a copy of N's _Step with N+1's step
    numbers, _res cleared; no _build, no blob upload, no positions D2H / wait.
  Every ENG_D13S_CHECK_EVERY-th steady step (default 0 = never) takes the shadow path instead: the host prepare runs and
  the kernel applied to the live (device-evolved) blob is compared with it. Steady = the previous step was an async step
  whose HSD prepare this module saw (graph H existed: no deferred capture), the same signature, all decode / 4 rows, no
  grammar / bitmask; anything else takes today's path (which also re-uploads the full blob).
"""
from __future__ import annotations

import copy
import os
import time

import torch

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODE = os.environ.get("ENG_D13S", "")
CHECK_EVERY = int(os.environ.get("ENG_D13S_CHECK_EVERY", "0"))
W = 4
NRING = 4                     # pinned result sets (a set is reused 4 steps later; extraction happens 1 step later)
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(PHYS, "engine/build/eng_d13s")
        os.makedirs(bd, exist_ok=True)
        _EXT = load(name="eng_d13s", sources=[os.path.join(PHYS, "kernels/d13s/hsdprep.cu")], build_directory=bd,
                    verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                      "-lineinfo"])
    return _EXT


class Steady:
    """Steady-step detection + the HSD-prep shadow check (ENG_D13S=shadow)."""

    FIELDS = ("cs", "ce", "req_idx", "g_dslot", "g_p0", "row_req", "row_local", "pos", "tok_slot", "tok_j", "temp", "seeds",
              "wbase", "wpages", "rstart", "lslot", "seq_of_d", "tpos", "tslot")

    def __init__(self, d13):
        self.d13, self.eng = d13, d13.eng
        self.prev_sig = None
        self.snap = None
        self.err = [None] * W
        self.stats = dict(steady=0, not_steady=0, checked=0, mismatch=0, live=0, deferred=0)
        self.d13.stats["d13s"] = self.stats
        self.live = MODE == "on"
        self.last_prep = None                # the StepHandle whose dispatch prepared HSD last (hsd.cur is its _Step)
        self.res_h = None                    # [NRING][W] (num_sampled int32 [MR], sampled int64 [MR, k + 1]) pinned
        self.res_i = 0
        self.res_owner = [None] * NRING
        self.n_live = 0
        self.window = _window(self.eng.hooks.hsd)
        ext()                                # load at engine init (never JIT mid-traffic)

    @staticmethod
    def signature(batches):
        """Per rank: the sequences' (rid, state slot, kind, rows, first row) in order."""
        return tuple(tuple((se.rid, se.state_slot, se.kind, se.n_rows, se.row0) for se in sb.seqs) for sb in batches)

    def before_prepare(self, st, batches) -> None:
        """Called right before HSD's prepare of step N+1 (step stream current per device): if this step is steady w.r.t.
        the previous one, snapshot every rank's device blob (still step N's), stream-ordered before the new upload."""
        H = self.eng.hooks
        hsd = H.hsd
        sig = self.signature(batches)
        prev, self.prev_sig = self.prev_sig, sig
        self.snap = None
        c = getattr(hsd, "cur", None)
        ok = (prev == sig and c is not None and getattr(hsd, "prep", "1") != "0" and getattr(hsd, "pool", False)
              and not H.bm_step and not getattr(hsd, "bm_any", False) and not H.grammar
              and all(se.kind == "decode" and se.n_rows == 4 for sb in batches for se in sb.seqs))
        if not ok:
            self.stats["not_steady"] += 1
            return
        self.stats["steady"] += 1
        key = (c.S_pad, c.D_pad, getattr(c, "NT", None))
        snap = []
        for r in range(W):
            with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                snap.append(hsd._dblob[r].clone())
        self.snap = (key, snap)

    def after_prepare(self, st, batches) -> None:
        """After HSD's prepare (its blob upload queued on the step streams), before graph H: in a steady step run the
        kernel on the snapshot and compare it with the uploaded blob (bucket prefixes)."""
        if self.snap is None:
            return
        key, snap = self.snap
        self.snap = None
        hsd = self.eng.hooks.hsd
        c = hsd.cur
        if (c.S_pad, c.D_pad, getattr(c, "NT", None)) != key:     # the bucket changed: not steady after all
            self.stats["steady"] -= 1
            self.stats["not_steady"] += 1
            return
        S_pad, D_pad, NT = key
        n_of = dict(T=W * S_pad, WD=W * D_pad, D=D_pad, NT=NT, MR=hsd.MR)
        X = ext()
        bad = {}
        for r in range(W):
            s = st[r]
            dev = self.eng.devs[r]
            with torch.cuda.device(dev), torch.cuda.stream(self.eng.streams[r]):
                if self.err[r] is None:
                    self.err[r] = torch.zeros(2, dtype=torch.int32, device=dev)
                v = hsd._views(snap[r])
                pre = {nm: v[nm][:n_of[dim]] for nm, (_, _, dim, _) in hsd._lay.items()}
                nt = c.n_t[r]
                X.hsdprep(pre["cs"], pre["ce"], pre["g_dslot"], pre["g_p0"], pre["pos"], hsd.st[r].num_sampled,
                          r * D_pad, D_pad, pre["wbase"], pre["wpages"], pre["lslot"], self.d13.rx[r].ring,
                          self.window, pre["tpos"], pre["tslot"], nt, s["pos"], s["own"], self.err[r])
                live = hsd._dv[r]
                for nm, (_, _, dim, _) in hsd._lay.items():
                    if not torch.equal(pre[nm], live[nm][:n_of[dim]]):
                        bad.setdefault(nm, []).append(r)
        self.stats["checked"] += 1
        if bad:
            self.stats["mismatch"] += 1
            if self.stats["mismatch"] <= 12:
                self.eng.log(f"engine D13S shadow: device HSD-prep differs from the host blob in {bad} (field: ranks), "
                             f"key {key}, n_t {list(c.n_t)}, seqs per rank {[len(sb.seqs) for sb in batches]}")
        if self.stats["checked"] % 1000 == 0:
            self.eng.log(f"engine D13S shadow: {self.stats['checked']} steady steps checked, {self.stats['mismatch']} "
                         f"differing; {self.stats['not_steady']} not steady")


class _Deferred:
    __slots__ = ("hcur", "bufs", "evs", "act")

    def __init__(self, hcur, bufs, evs, act):
        self.hcur, self.bufs, self.evs, self.act = hcur, bufs, evs, act


def _live_methods():
    """ENG_D13S=on: the steady path (methods of Steady)."""

    def is_steady(self, batches, inflight, R) -> bool:
        """dispatch(N+1), after on_ctx: may this step skip the wait for N and HSD's host prepare?"""
        H = self.eng.hooks
        hsd = H.hsd
        if inflight is None or inflight is not self.last_prep or inflight.cap_h is not None or inflight.R != R \
                or inflight._res is not None or getattr(inflight, "deferred", None) is not None:
            return False
        if H.bm_step or H.grammar or getattr(hsd, "bm_any", False) or getattr(hsd, "prep", "1") == "0" \
                or not getattr(hsd, "pool", False) or not H.hsd_on or hsd.cur is None:
            return False
        if not all(se.kind == "decode" and se.n_rows == 4 for sb in batches for se in sb.seqs):
            return False
        if getattr(self.d13.E, "NANCHECK", False) or any(se.rid in H.structured for sb in batches for se in sb.seqs):
            return False
        if self.signature(batches) != self.signature(inflight.batches):
            return False
        if CHECK_EVERY and (self.n_live + 1) % CHECK_EVERY == 0:      # this one goes through the shadow path
            self.n_live += 1
            return False
        return True

    def defer_extract(self, h) -> None:
        """Queue step N's result copies (ranks with drafting requests) on the extraction streams after N's done event;
        the step streams wait for them (graph H of N+1 rewrites num_sampled / sampled). No host wait."""
        hsd = self.eng.hooks.hsd
        c = hsd.cur
        n = c.n_req
        if self.res_h is None:
            pin = dict(pin_memory=True)
            self.res_h = [[(torch.zeros(hsd.MR, dtype=torch.int32, **pin), torch.zeros(hsd.MR, hsd.k + 1, dtype=torch.int64, **pin))
                           for _ in range(W)] for _ in range(NRING)]
        old = self.res_owner[self.res_i]
        if old is not None and old._res is None:  # (never in vLLM's loop: a step is merged before the next dispatch)
            self.d13._extract(old)
        self.res_owner[self.res_i] = h
        bufs = self.res_h[self.res_i]
        self.res_i = (self.res_i + 1) % NRING
        act = [r for r in range(W) if c.dseqs[r]]
        evs = []
        for r in act:
            dev = self.eng.devs[r]
            xs = self.d13.xstream[r]
            with torch.cuda.device(dev):
                xs.wait_event(h.ev[r])
                with torch.cuda.stream(xs):
                    st_ = hsd.st[r]
                    bufs[r][0][:n].copy_(st_.num_sampled[:n], non_blocking=True)
                    bufs[r][1][:n].copy_(st_.sampled[:n], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(xs)
                self.eng.streams[r].wait_event(ev)
            evs.append(ev)
        h.deferred = _Deferred(c, bufs, evs, act)
        self.stats["deferred"] += 1

    def extract_deferred(self, h) -> None:
        """StepHandle.result() of a deferred step: wait for its result copies, then the StepResults (HSD.result's
        formula with the step's own _Step) and head_result's host bookkeeping (gate fill, token history)."""
        import interfaces as IF
        eng, d13 = self.eng, self.d13
        H = eng.hooks
        t0 = time.perf_counter()
        dfr = h.deferred
        for ev in dfr.evs:
            ev.synchronize()
        t1 = time.perf_counter()
        c = dfr.hcur
        n = c.n_req
        st = eng.bucket(h.R)
        res = []
        for r in range(W):
            out = IF.StepResult(r, c.step[r])
            if r in dfr.act:
                ns = dfr.bufs[r][0][:n].tolist()
                smp = dfr.bufs[r][1][:n].tolist()
                acc, tok = out.accepted, out.tokens
                for _, qi in c.dseqs[r]:
                    rid = c.rids[qi]
                    a_ = ns[qi]
                    acc[rid] = a_
                    tok[rid] = smp[qi][:a_]
            if H.gate is not None:
                s_ = dict(st[r])
                s_["batch"], s_["tab"] = h.batches[r], h.tabs[r]
                with torch.cuda.device(s_["dev"]), torch.cuda.stream(d13.xstream[r]):
                    H.gate.fill(r, s_, s_["tab"], out)
            for rid, toks in out.tokens.items():
                if rid in eng.hist:
                    eng.hist[rid].extend(toks)
            res.append(out)
        h._res = res
        d13._post_extract(h)
        h.t.update(extract_wait=round((t1 - t0) * 1000, 3), extract=round((time.perf_counter() - t1) * 1000, 3))
        eng.stats["d13_last"] = dict(h.t)

    def prepare_device(self, st, batches) -> None:
        """HSD's prepare of a steady step N+1: host state (cur = N's _Step with N+1's step numbers, results cleared) and
        the device HSD-prep per rank on its step stream (after N's graph H and N's result copies, before N+1's graph H)."""
        H = self.eng.hooks
        hsd = H.hsd
        c = copy.copy(hsd.cur)                # keeps NT (and any subclass attributes such as gvar)
        c.step = [b.step for b in batches]
        hsd.cur = c
        hsd._res = [None] * W
        self.prev_sig = self.signature(batches)
        X = ext()
        D_pad = c.D_pad
        for r in range(W):
            s = st[r]
            dev = self.eng.devs[r]
            with torch.cuda.device(dev), torch.cuda.stream(self.eng.streams[r]):
                if self.err[r] is None:
                    self.err[r] = torch.zeros(2, dtype=torch.int32, device=dev)
                tb = hsd.tab[r]
                X.hsdprep(tb["cs"], tb["ce"], tb["g_dslot"], tb["g_p0"], tb["pos"], hsd.st[r].num_sampled, r * D_pad, D_pad,
                          tb["wbase"], tb["wpages"], tb["lslot"], self.d13.rx[r].ring, self.window, tb["tpos"], tb["tslot"],
                          c.n_t[r], s["pos"], s["own"], self.err[r])
        self.n_live += 1
        self.stats["live"] += 1

    return dict(is_steady=is_steady, defer_extract=defer_extract, extract_deferred=extract_deferred,
                prepare_device=prepare_device)


for _k, _v in _live_methods().items():
    setattr(Steady, _k, _v)


def _window(hsd=None) -> int:
    """drafter.WINDOW as HSD's module sees it (kernels/head/drafter.py, imported by hsd.py as DR)."""
    import sys
    for k in type(hsd).__mro__ if hsd is not None else ():
        m = sys.modules.get(k.__module__)
        if m is not None and hasattr(m, "DR"):
            return int(m.DR.WINDOW)
    raise RuntimeError("D13S: drafter.WINDOW not found through HSD's module")
