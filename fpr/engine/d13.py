"""D13: async decode steps whose positions only the device knows.

The host plans step N+1 while step N runs, so N+1's decode positions (num_computed after N's acceptance) are resolved
on the GPU by kernels/rowexp (k_ring_ops + k_expand from serve's CompactTables, k_advance after the head).

Engine.dispatch(batches, cts, ctx) -> StepHandle (Engine.step_compact is the same call: serve's worker contract).
dispatch(N+1), all on the step streams (ordered after N's work):
  1. D12 block-table deltas of N+1 -> the per-slot device table (shared by rowexp and the indexer);
  2. host-known inputs (tokens / valid / sample rows in the bucket's input blob; -1 rows from HSD's next ids), the
     indexer's host part (sequence slots, valid rows), Engram's compact args (decode starts resolved on the device);
  3. rowexp upload + begin(R) into the bucket's pos / own / si / sl (+ paged); the indexer's position tables derived on
     the device; one small D2H of pos / own;
  4. replay graph L (embed .. final, the layers) of the bucket;
  -- step N+1 is now queued behind N: the GPU does not idle --
  5. wait for step N, extract its results (HSD's result of N is read BEFORE HSD's prepare of N+1 overwrites it);
  6. wait for N+1's positions (D2H, right after N), HSD's host prepare of N+1 with them (overlaps graph L), replay
     graph H (HSD) and rowexp end(accepted) (k_advance from HSD's per-request emitted counts).
Steps that need host positions (prefill chunks / mixed steps, forced / score sequences, contexts beyond the decode
graphs' indexer bound, page copies, rows beyond the largest bucket) run the synchronous engine.step after draining:
rowexp resolves their starts (and keeps its ring / nc in step), a private TableBuilder builds full tables.

Exactness: the same rows / positions / slots as the synchronous tables (rowexp is bit-exact vs TableBuilder), the same
kernels and graphs; only where positions are computed and when the host prepares move.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

_ROUTE_STATS = os.environ.get("ENG_ROUTE_STATS", "")

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = 4
K1 = 4                    # rows per decode sequence (k + 1, k = 3)
SW = 128                  # sliding window (si columns)
CHECK = os.environ.get("ENG_D13_CHECK", "") == "1"     # debug: fallback steps compare rowexp's rows with the host tables
DELTA_CAP = 65_536        # block-table delta entries per rank and step through pinned staging (more: direct copies)
GT = os.environ.get("ENG_D13_GT", "0") == "1"   # "graph T": the device ops of _pre after the
                          # uploads (rowexp begin = ring ops + expand, the guard's own / paged fixes, the indexer's
                          # device_tables_d13: ~27 eager torch calls per rank, ~1.9 ms of host per C1 step) captured once
                          # per (bucket, rank, rank has sequences) into a small CUDA graph and replayed: the same ops on the
                          # same buffers, no per-step host values inside (R and the scratch slot are per bucket / rank)
ERR_EVERY = max(1, int(os.environ.get("ENG_D13_ERR_EVERY", "1")))   # read rowexp's cumulative ring-miss counters
                          # every N-th extraction (default every step: 4 sum kernels + 4 host syncs, ~0.29 ms per C1 step)
EMPTY = os.environ.get("ENG_D13_EMPTY", "0") == "1"   # C1 host work: a rank with no sequences, no ring ops and no
                          # block-table deltas whose device inputs already hold an empty step's state (its last prep of this
                          # bucket was empty, its rowexp descriptor / Engram block too, no synchronous step since) skips its
                          # uploads, rowexp / graph T and Engram: the full prep would write the same values (3 of 4 ranks at C1)
EMPTY_CHECK = os.environ.get("ENG_D13_EMPTY_CHECK", "0") == "1"   # debug: snapshot after every full empty prep, compare
                          # bitwise at every skip (synchronizes; counts / logs)
ADV_SLICE = os.environ.get("ENG_D13_ADV_SLICE", "0") == "1"   # _advance: when a rank's drafting sequences are its batch's
                          # sequences in order (HSD request index = rank * D_pad + j), the accepted counts are the contiguous
                          # slice num_sampled[q0 : q0 + S] (= the index_select the default path builds through an H2D copy)
RES_OWNER = os.environ.get("ENG_D13_RES_OWNER", "0") == "1"   # _extract: HSD.result's per-step D2H of the sampled ids /
                          # counts only from the ranks with drafting sequences (the others' results read none of it)
OWNER = os.environ.get("ENG_D13_OWNER", "0") == "1" and GT   # C1 host work (stage 1c): a rank WITH sequences whose step
                          # repeats its last full prep (same sequences / rows / allocated blocks / sample rows, every row a
                          # device-token row from this rank's previous step, no ring ops / block-table deltas, no synchronous
                          # step since) skips its uploads, rowexp descriptor, guard host half and Engram args: its constant
                          # inputs are already on the device. Graph T' resets the inputs the device rewrites in place
                          # (indexer rslot, Engram's resolved starts) from copies taken at that full prep, redoes the device
                          # tokens (next ids -> tok rows), then graph T's ops. Needs ENG_D13_GT.
OWNER_CHECK = os.environ.get("ENG_D13_OWNER_CHECK", "0") == "1"   # debug: every skip also runs the full path afterwards
                          # and compares every input buffer bitwise with what graph T' left (synchronizes)
PRE_T = os.environ.get("ENG_D13_PRE_T", "0") == "1"   # timing: per-rank host ms of _pre in StepHandle.t (PHYS_STEP_TIMES)
GT_CHECK = os.environ.get("ENG_D13_GT_CHECK", "0") == "1"   # debug: every step also runs the eager ops after the graph
                          # (rslot restored first) and compares every output buffer bitwise (synchronizes; counts / logs)
FB_REUSE = os.environ.get("ENG_D13_FB_REUSE", "1") == "1"   # fallback steps keep the StepBatch's
                          # own positions / paged / window slots for host-known sequences (start >= 0: build_step_batch
                          # computed them with the same formulas) and rebuild only the device-resolved decode rows. Same
                          # lists (CPU check on 9 CP steps x 4 ranks: 0 mismatches); 2.7 -> 0.6 ms of host per 17K-row step


TFUSE = os.environ.get("ENG_D13_TFUSE", "0") == "1"   # C1 host work: graph T' (the steady owner's step
                          # prep, ~45 tiny torch kernels, ~55 us on the critical path between graph H and the owner's graph L)
                          # as ONE kernel (fpr/kernels/d13tail: the same ops, order and buffers; integer only, bitwise; ~9 us).
                          # Default off. If the extension is unavailable the step falls back to graph T' (logged once).
TFUSE_CHECK = os.environ.get("ENG_D13_TFUSE_CHECK", "0") == "1"   # debug: every T' step runs the fused kernel, then graph T'
                          # as usual (the step uses graph T''s values), and compares every output bitwise (synchronizes)
_D13TAIL = None           # the d13tail module once loaded; False = unavailable (graph T' stays)
CXX = os.environ.get("PHYS_CXX_D13", "0") == "1"   # stage 2a: the host decisions in C++
LONG = os.environ.get("ENG_D13_LONG", "0") == "1"
                          # default off): a decode step with a row past GRAPH_MAX_CTX (262,144 positions), which _async_ok
                          # (Python or PHYS_CXX_D13's host: the same string, its last check) refuses with LONG_WHY, stays
                          # async on the bucket's long variant Rk = bucket_of(n, pad=False) + lvl * LVL_TAG under
                          # engine.step's rule (_long_level), instead of _fallback -> engine.step's sync _step_split(Rk).
                          # Keys (bucket, graphs L / H / T, skip / epoch state, steady) by Rk; sizes by st[r]["R"] (rows).
                          # A step whose level its device-resolved starts leave open (bound known + K1 vs known + 2 K1 on
                          # two levels) stays sync: async == sync exactly (no lvl 0 / 1 tie at the 262,144 edge).
LONG_WHY = "context beyond the decode graphs' indexer bound"
                          # (fpr/core/src/d13: _async_ok, _owner_sig, _empty_eligible, _delta_rows, _fb_rows, HSD's _build)
                          # through the fpr_d13_cxx binding (d13_cxx.install); device work, tables and objects stay Python


def _d13tail():
    """kernels/d13tail, imported on first use under ENG_D13_TFUSE(_CHECK); any failure -> False (graph T' is used)."""
    global _D13TAIL
    if _D13TAIL is None:
        try:
            p = os.path.join(PHYS, "kernels/d13tail")
            if p not in sys.path:
                sys.path.insert(0, p)
            import d13tail as DT
            DT.ext()
            _D13TAIL = DT
        except Exception as e:  # noqa: BLE001
            print(f"engine D13 TFUSE: kernels/d13tail unavailable ({type(e).__name__}: {str(e)[:200]}); graph T' stays",
                  flush=True)
            _D13TAIL = False
    return _D13TAIL


def _rx():
    """kernels/rowexp (read-only), its extension built as engine/build/eng_rowexp (rowexp_ext below)."""
    p = os.path.join(PHYS, "kernels/rowexp")
    if p not in sys.path:
        sys.path.insert(0, p)
    import rowexp as RX
    if getattr(RX, "_eng_ext", None) is None:
        RX._eng_ext = rowexp_ext
        RX.ext = rowexp_ext            # RowExp's methods call the module-level ext()
    return RX


_EXT = None


def rowexp_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(PHYS, "engine/build/eng_rowexp")
        os.makedirs(bd, exist_ok=True)
        _EXT = load(name="eng_rowexp", sources=[os.path.join(PHYS, "kernels/rowexp/rowexp.cu")], build_directory=bd,
                    verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                      "-lineinfo"])
    return _EXT


class StepHandle:
    """One dispatched step: result() -> [StepResult per rank] (waits for the step, extracts once). List-like, so callers
    that index / iterate the result of Engine.step get the same values (at the cost of waiting)."""

    def __init__(self, d13, step, results=None):
        self.d13, self.step, self._res = d13, step, results
        self.ev = None
        self.batches = self.tabs = None
        self.R = None
        self.cap_h = None
        self.t = {}

    def done(self) -> bool:
        return self._res is not None

    def result(self) -> list:
        if self._res is None:
            with self.d13.lock:
                if self._res is None:
                    self.d13._extract(self)
        return self._res

    def __iter__(self):
        return iter(self.result())

    def __len__(self):
        return len(self.result())

    def __getitem__(self, i):
        return self.result()[i]


class D13:
    def __init__(self, eng):
        from engine import engine as E
        self.E, self.eng = E, eng
        H = eng.hooks
        assert H.idx is not None and H.hsd is not None and H.engram is not None, \
            "D13 needs the full pieces (indexer idx2f, engram phys, head hsd)"
        assert eng.cfg.upload_v2, "D13 uses the bucket input blobs (EngineConfig.upload_v2)"
        RX = _rx()
        self.RX = RX
        self.lock = threading.RLock()
        self.inflight = None
        cfg = eng.cfg
        self.max_rows = max(cfg.buckets[-1], -(-cfg.max_mixed_rows // cfg.mixed_round) * cfg.mixed_round)
        self.max_seqs = cfg.max_seqs + 64
        self.rx, self.rx_own = [], []
        for r, dev in enumerate(eng.devs):
            dbt = H.idx.views[r]["dbt"]
            with torch.cuda.device(dev):
                x = RX.RowExp(dev, dbt.shape[0], self.max_seqs, self.max_rows,
                              scratch_slot=eng.pools[r].scratch_window_page * E.WINDOW_PAGE_TOKENS, dbt=dbt)
            self.rx.append(x)
            self.rx_own.append(dict(x.out))
            H.engram[r].bind_start(x.out["start"])      # Engram: decode starts -1 resolved from rowexp's (this step's)
        pin = dict(pin_memory=True)
        self.d2h = [[(torch.zeros(self.max_rows, dtype=torch.int64, **pin), torch.zeros(self.max_rows, dtype=torch.int64, **pin))
                     for _ in range(W)] for _ in range(2)]
        self.d2h_i = 0
        self.qidx_h = [[torch.zeros(self.max_seqs, dtype=torch.int64, **pin) for _ in range(W)] for _ in range(2)]
        self.qidx_ev = [[None] * W for _ in range(2)]
        self.qidx_d = [torch.zeros(self.max_seqs, dtype=torch.int64, device=d) for d in eng.devs]
        self.qi = 0
        self.dl_h = [[(torch.zeros(DELTA_CAP, dtype=torch.int64, **pin), torch.zeros(DELTA_CAP, dtype=torch.int64, **pin),
                       torch.zeros(DELTA_CAP, dtype=torch.int32, **pin)) for _ in range(W)] for _ in range(2)]
        self.dl_ev = [[None] * W for _ in range(2)]
        self.dl_d = [(torch.zeros(DELTA_CAP, dtype=torch.int64, device=d), torch.zeros(DELTA_CAP, dtype=torch.int64, device=d),
                      torch.zeros(DELTA_CAP, dtype=torch.int32, device=d)) for d in eng.devs]
        self.dl_i = 0
        self.acc_h = [torch.zeros(self.max_seqs, dtype=torch.int32, **pin) for _ in range(W)]
        self.nb_h = [[torch.zeros(self.max_seqs, dtype=torch.int64, **pin) for _ in range(W)] for _ in range(2)]
        self.nb_ev = [[None] * W for _ in range(2)]
        self.nb_d = [torch.zeros(self.max_seqs, dtype=torch.int64, device=d) for d in eng.devs]
        self.err_seen = [0] * W
        self._n_extract = 0
        # extraction stream per device: step N's result copies wait on N's done event only, not behind graph L(N + 1)
        # (on the engine stream they queued after it: extract 35 ms, H(N + 1) launched a step late)
        self.xstream = [torch.cuda.Stream(device=d) for d in eng.devs]
        from worker import TableBuilder
        self.tb = TableBuilder(stub_indexer=False)      # full tables of the synchronous fallback steps
        self._blas_threads = set()                       # threads whose cuBLAS / cuBLASLt handles exist (see _blas_ready)
        self._dispatch_tid = None                        # the thread that dispatches (served: the EngineCore loop)
        self._pending_caps = []                          # graph H captures of steps another thread extracted (_extract)
        self.stats = dict(async_steps=0, fallback_steps=0, first_steps=0)
        self._r_empty = [False] * W                      # ENG_D13_EMPTY: rank r's rowexp descriptor + Engram block hold an
        self._b_empty = {}                               # empty step's; (R, r) -> epoch of bucket R's empty state on rank r
        self._epoch = 0                                  # bumped by every synchronous (fallback / first) step
        self._esnap = {}                                 # ENG_D13_EMPTY_CHECK snapshots
        self._own_last = [None] * W                      # ENG_D13_OWNER: (R, signature, epoch) of rank r's last prep
        self._own_cmp = [None] * W                       # ENG_D13_OWNER_CHECK: graph T' outputs awaiting the full path
        if OWNER:
            self.stats.update(owner_skips=0, owner_full=0)
            if OWNER_CHECK:
                self.stats.update(owner_checked=0, owner_mismatch=0)
        if EMPTY:
            self.stats.update(empty_skips=0, empty_full=0)
            if EMPTY_CHECK:
                self.stats.update(empty_checked=0, empty_mismatch=0)
        self.steady = None                               # ENG_D13S (engine/d13s.py): steady-state decode, stage 1
        if os.environ.get("ENG_D13S", ""):
            from engine import d13s
            self.steady = d13s.Steady(self)
        self.gt = {}                                     # ENG_D13_GT: (R, rank, has seqs) -> (CUDAGraph, buffer ptrs)
        self.gt_pool = [None] * W                        # one private memory pool per rank for every graph T (they only
        self.gt_stream = [None] * W                      # run one after another on the step stream); a capture stream
        if GT:
            self.stats.update(gt_captures=0, gt_replays=0, gt_recaptures=0)
            if GT_CHECK:
                self.stats.update(gt_checked=0, gt_mismatch=0)
        if CXX:                               # PHYS_CXX_D13 (stage 2a): before the recorder, so traces record the C++ path
            p_ = os.path.join(PHYS, "core", "src", "d13")
            if p_ not in sys.path:
                sys.path.insert(0, p_)
            import d13_cxx
            d13_cxx.install(self)
        if os.environ.get("PHYS_GTRACE"):     # golden-trace recorder (fpr/core/GTRACE.md, stage 2; default off): wraps
            try:                              # THIS instance's host decisions and HSD's _build, passing everything through
                p_ = os.path.join(PHYS, "core", "src", "d13")      # (fpr/core/src/d13/gtrace_d13.py)
                if p_ not in sys.path:
                    sys.path.insert(0, p_)
                import gtrace_d13
                gtrace_d13.install(self)
                gtrace_d13.install_hsd(H.hsd)
            except Exception as e:  # noqa: BLE001
                print(f"engine D13: golden-trace recorder not installed ({type(e).__name__}: {str(e)[:200]})", flush=True)

    # ------------------------------------------------------------------------------------------ routing
    def _async_ok(self, batches, cts) -> str:
        eng, E = self.eng, self.E
        if not eng.cfg.graphs:
            return "graphs off"
        mx = 0
        for sb, ct in zip(batches, cts):
            if sb.copies:
                return "page copies"
            if sb.n_rows > eng.cfg.buckets[-1]:
                return "rows > largest bucket"
            S = len(sb.seqs)
            if not S:
                continue
            if np.any(ct.kind[:S] != 0) or np.any(ct.forced[:S] != 0) or np.any(ct.score[:S] != 0):
                return "prefill / forced / score rows"
            if np.any(ct.n_rows[:S] != K1) or np.any(ct.row0[:S] != K1 * np.arange(S)):
                return "decode layout"
            for se, st_ in zip(sb.seqs, ct.start[:S].tolist()):
                mx = max(mx, (st_ if st_ >= 0 else se.known + K1) + K1)
        if mx > E.GRAPH_MAX_CTX:
            return "context beyond the decode graphs' indexer bound"
        return ""

    def _long_level(self, batches, cts) -> tuple:
        """ENG_D13_LONG: (lvl, why) of a step _async_ok refused with LONG_WHY only (every other check passed).
        engine.step's rule (:3708-3717) on D13's bound: per sequence (start >= 0 ? start : known + K1) + K1, = engine.step's
        ctx_max for host-known starts (nd_long_gen.long_level); lvl = the first LONG_LEVEL_CTX entry >= it; ok = ENG_LONG_GRAPHS,
        split_hsd, HSD on, not ENG_SYNC_DEBUG, > 64 rows or FPR_LONG_ANY_ROWS (graphs: _async_ok). Not ok -> (0, LONG_WHY):
        the sync path, as today. A device-resolved start lies in (known, known + K1]: when the low bound (known + K1) and
        the high bound (known + 2 K1) fall on different levels the step stays sync (why = 'long level edge'). C1 lane /
        variable verify rows: sync (not gated on long variants)."""
        E, eng = self.E, self.eng
        hi = lo = 0
        for sb, ct in zip(batches, cts):
            S = len(sb.seqs)
            for se, st_ in zip(sb.seqs, ct.start[:S].tolist()):
                if st_ >= 0:
                    hi, lo = max(hi, st_ + K1), max(lo, st_ + K1)
                else:
                    hi, lo = max(hi, se.known + 2 * K1), max(lo, se.known + K1)

        def lv(m):
            return 0 if m <= E.GRAPH_MAX_CTX else next((i for i, c in enumerate(E.LONG_LEVEL_CTX) if m <= c), -1)
        lvl = lv(hi)
        if lv(lo) != lvl:
            self.stats["long_edge"] = self.stats.get("long_edge", 0) + 1
            return 0, "long level edge (device-resolved start)"
        ok = (E.LONG_GRAPHS and lvl > 0 and eng.cfg.split_hsd and eng.hooks.hsd is not None and not E.DBG_SYNC
              and (sum(sb.n_rows for sb in batches) > 64 or E.LONG_ANY_ROWS))
        if not ok:
            return 0, LONG_WHY
        if getattr(self, "lane", None) is not None or getattr(self, "vrows", None) is not None:
            return 0, LONG_WHY + " (C1 lane / variable rows: sync)"
        return lvl, ""

    # ------------------------------------------------------------------------------------------ dispatch
    def dispatch(self, batches, cts, ctx=None) -> StepHandle:
        with self.lock:
            eng = self.eng
            t0 = time.perf_counter()
            self._run_pending_caps()
            why = self._async_ok(batches, cts)
            lvl = 0
            if LONG and why == LONG_WHY:       # ENG_D13_LONG: the bucket's long variant, async (engine.step's rule)
                lvl, why = self._long_level(batches, cts)
            if why:
                return self._fallback(batches, cts, ctx, why)
            eng.hooks.on_ctx(ctx, None, None)
            if CHECK:
                self._remap_check(batches)
            n = max(sb.n_rows for sb in batches)
            if lvl:                            # (pad=False: ENG_DECODE_PRECREATE never pads a long-row step, as engine.step)
                R = eng.bucket_of(max(n, 1), pad=False) + lvl * self.E.LVL_TAG
                self.stats["long_steps"] = self.stats.get("long_steps", 0) + 1
                eng.stats["long_steps"] = eng.stats.get("long_steps", 0) + 1
            else:
                R = eng.bucket_of(max(n, 1))
            # ENG_C1_MEGA: C1 steps get their own graph L (ENG_C1_MEGA_AB: per arm). getattr: tolerate an engine.py older
            # than this file
            own = getattr(eng, "c1m_owner", None)
            mo = (own(eng.bucket(R), batches) if own else eng.c1m().owner_of(eng.bucket(R), batches)) if self.E.C1_MEGA else None
            abk = getattr(eng, "_ab_key", None)
            lk = (R, "LA") if mo is None else (R, "LA", "M", mo) + (abk() if abk else ())
            if lk not in eng.graphs:
                return self._first(batches, cts, R, mo)
            st = eng.bucket(R)
            eng.bump_serial(batches)           # (PREV_STEP_CHECK: this async step's serial, before its device-token reads)
            eng._tag_mega(st, mo)
            self._pre(st, batches, cts, R)
            t1 = time.perf_counter()
            eng.replay(lk)
            t2 = time.perf_counter()
            live = self.steady is not None and self.steady.live and self.steady.is_steady(batches, self.inflight, R)
            if live:                           # ENG_D13S=on: no wait for N, no positions D2H, HSD prepared on the device
                self.steady.defer_extract(self.inflight)
                t3 = t4 = time.perf_counter()
                self.steady.prepare_device(st, batches)
            else:
                if self.inflight is not None:
                    self._extract(self.inflight)
                t3 = time.perf_counter()
                pb = self._positions(st, batches)
                t4 = time.perf_counter()
                if self.steady is not None:
                    self.steady.before_prepare(st, batches)
                eng._prepare_head_on_streams(st, pb)
                if self.steady is not None:
                    self.steady.after_prepare(st, batches)
            self._set_prev_seq(batches)
            hk = eng._hkey(R)
            h = StepHandle(self, batches[0].step if batches else 0)
            if hk is not None and hk in eng.graphs:
                eng.replay(hk)
            else:
                eng.hooks.head_eager(st)      # new HSD key (graph H captured when this step is extracted) / grammar
                h.cap_h = hk
            self._advance(st, batches)
            h.ev = self._record()
            t5 = time.perf_counter()
            h.batches, h.tabs, h.R = list(batches), [st[r]["tab"] for r in range(W)], R
            h.grammar = dict(eng.hooks.grammar)
            ms = lambda a_, b_: round((b_ - a_) * 1000, 3)  # noqa: E731
            h.t = dict(prep=ms(t0, t1), launch_l=ms(t1, t2), wait_prev=ms(t2, t3), pos_wait=ms(t3, t4),
                       head=ms(t4, t5), host_total=ms(t0, t5) - ms(t2, t4))
            if mo is not None and abk and abk():   # ENG_C1_MEGA_AB: this C1 step's arm (PHYS_STEP_TIMES "d13" dict)
                h.t["ab"] = eng._ab_i
            if PRE_T:
                h.t["pre_r"] = self._pre_r
                h.t["seqs_r"] = [len(sb.seqs) for sb in batches]
            eng.last_st = st
            eng.stats["steps"] += 1
            eng.stats["graph_steps"] += 1
            self.stats["async_steps"] += 1
            if self.steady is not None:
                self.steady.last_prep = h
                if live:
                    h.t["d13s"] = 1
            if _ROUTE_STATS and eng.stats["steps"] % 256 == 0:   # ENG_ROUTE_STATS (debug, default off): see engine.py
                eng._dump_route_stats()
            self.inflight = h
            return h

    def _pre(self, st, batches, cts, R) -> None:
        """Steps 1-3 of dispatch (see the module doc) for every rank, on its step stream. R = the bucket key (Rk for a long
        variant: skip / epoch / graph T state); sizes come from the state's rows Rn = st[r]["R"] (= R at lvl 0)."""
        eng, H = self.eng, self.eng.hooks
        Rn = int(st[0]["R"])
        EG = H.EG
        eng._batches = batches
        self.d2h_i ^= 1
        self.dl_i ^= 1
        tr = [] if PRE_T else None
        for r in range(W):
            if PRE_T:
                tr.append(time.perf_counter())
            s, sb, ct, rx = st[r], batches[r], cts[r], self.rx[r]
            if EMPTY:
                el = self._empty_eligible(sb, ct)
                if el and self._r_empty[r] and self._b_empty.get((R, r)) == self._epoch:
                    self._empty_skip(r, s, sb, R)
                    continue
            osig = self._owner_sig(r, sb, ct) if OWNER else None
            if osig is not None and self._own_last[r] == (R, osig, self._epoch):
                if self._owner_skip(r, s, sb, ct, R):
                    continue
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
                cur = torch.cuda.current_stream()
                self._deltas(r, ct)
                n = sb.n_rows
                tok = torch.from_numpy(np.asarray(ct.tokens[:n], dtype=np.int64)) if n else torch.zeros(0, dtype=torch.int64)
                tab = NS(n=n, tokens=tok, sample_rows=list(np.asarray(ct.sample_idx).tolist()), seq={},
                         seq_rows=[(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs])
                eng._upload_v2(r, s, tab, dev_pos=True)      # tok (+ device next ids) / valid / samp
                H.idx.upload_d13(r, s, sb)
                if osig is not None:                         # ENG_D13_OWNER: the uploaded rslot base (before graph T), into
                    self._keep(s, "_own_rbase", s["idx"]["rslot"])   # a persistent buffer (graph T' bakes its address)
                rx.upload(self.RX.RowExp.desc_of_compact(ct), stream=cur)
                if "paged_d" not in s:
                    s["paged_d"] = torch.full((Rn,), -1, dtype=torch.int64, device=s["dev"])
                rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(Rn, SW), sl=s["sl"], paged=s["paged_d"])
                gt = GT and not CHECK
                if gt:       # ENG_D13_GT: the guard's host half (nb upload), then graph T = begin + guard + idx tables
                    S_ = len(sb.seqs)
                    if S_:
                        self._guard_host(r, sb)
                    self._gt_run(r, s, R, S_ > 0)
                else:
                    rx.begin(Rn)
                    if CHECK:
                        self._check_tables(r, s, sb, R)
                    self._guard(r, s, sb, R)
                ph, oh = self.d2h[self.d2h_i][r]
                ph[:Rn].copy_(s["pos"], non_blocking=True)
                oh[:Rn].copy_(s["own"], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(cur)
                s["d13_ev"] = ev
                if not gt:
                    H.idx.device_tables_d13(r, s, s["paged_d"])
                a = EG.step_args_from_compact(ct, H._lookback, pad_to=Rn)
                H.engram[r].upload(a, stream=cur)
                s["tab"], s["batch"] = tab, sb
                if EMPTY:
                    self._empty_mark(r, s, R, el)
                if OWNER:
                    self._owner_mark(r, s, R, osig)
        if PRE_T:                # ENG_D13_PRE_T: host ms of each rank's part of _pre (-> StepHandle.t["pre_r"])
            tr.append(time.perf_counter())
            self._pre_r = [round((b_ - a_) * 1000, 3) for a_, b_ in zip(tr, tr[1:])]

    def _remap_check(self, batches) -> None:
        """ENG_D13_CHECK: a window block whose page changed since the previous step, inside a sequence's read range
        [start - 127, start] (the KV written there by the previous step would be read from another page)."""
        snap = getattr(self, "_wsnap", {})
        new = {}
        for r, sb in enumerate(batches):
            for se in sb.seqs:
                new[(r, str(se.rid))] = dict(se.window_table)
                old = snap.get((r, str(se.rid)))
                if old is None:
                    continue
                ch = [(b, old[b], pg) for b, pg in se.window_table.items() if b in old and old[b] != pg]
                if ch:
                    self.eng.log(f"engine D13 CHECK rank {r} step {sb.step}: {se.rid} window blocks remapped since the last "
                                 f"step {ch[:4]} (start {se.start}, known {getattr(se, 'known', -1)})")
                    self.stats["window_remap"] = self.stats.get("window_remap", 0) + len(ch)
        self._wsnap = new

    def _check_tables(self, r, s, sb, R) -> None:
        """ENG_D13_CHECK: this step's expanded tables BEFORE any kernel reads them (synchronizes the step stream; before
        the guard): rowexp ring misses, live rows (sequence >= 0, position inside the allocation) whose own window slot is
        missing, window read slots inside a row's length that are negative or past the window pool, paged slots past the
        paged pool. Logs the first offending rows with their sequences (a CUDA fault later in the step then has a cause)."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        n = sb.n_rows
        if not n:
            return
        torch.cuda.current_stream().synchronize()
        rx, pool = self.rx[r], self.eng.pools[r]
        o = rx.out
        seq, pos, own = o["seq"][:n].long(), s["pos"][:n].long(), s["own"][:n].long()
        si, sl = s["si"].view(R, -1)[:n].long(), s["sl"][:n].long()
        pg = s["paged_d"][:n].long()
        nb = torch.tensor([se.n_blocks or len(se.block_table) for se in sb.seqs] + [0], dtype=torch.int64, device=seq.device)
        live = (seq >= 0) & (pos // 64 < nb.index_select(0, seq.clamp(min=0)))
        wmax = (pool.n_window + 1) * self.E.WINDOW_PAGE_TOKENS
        ar = torch.arange(si.shape[1], device=si.device)
        inlen = ar[None, :] < sl[:, None]
        bad = dict(own=live & ((own < 0) | (own >= wmax)),
                   si=live & (inlen & ((si < 0) | (si >= wmax))).any(1),
                   paged=live & ((pg < -1) | (pg >= pool.n_paged * 64)),
                   pos=(seq >= 0) & (pos < 0))
        e_ = int(rx.err.sum())
        out = {}
        for k_, m in bad.items():
            idx = m.nonzero().flatten()
            if idx.numel():
                i = int(idx[0])
                se = sb.seqs[int(seq[i])]
                out[k_] = (int(idx.numel()), i, str(se.rid), int(pos[i]), se.kind, int(se.known), int(sl[i]))
        if out or e_ != self.err_seen[r]:
            self.eng.log(f"engine D13 CHECK rank {r} step {sb.step}: tables before launch: bad {out} (count, row, rid, pos, "
                         f"kind, known, sl); rowexp err {self.err_seen[r]} -> {e_}")
            self.stats["table_bad"] = self.stats.get("table_bad", 0) + 1

    def _guard(self, r, s, sb, R) -> None:
        """Rows whose positions run past their sequence's allocation (in async mode: the one wasted step of a sequence that
        finished in the in-flight step): rowexp gives own = -1 for a missing window
        page (the q_b insert would write at slot -1: illegal address) and an unchecked paged slot from a stale per-slot
        table entry (a write into another sequence's page). Own -> the scratch window slot, paged -> -1 (no write) for
        rows at or beyond the sequence's allocated blocks (SeqEntry.n_blocks). Live rows are untouched."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        S = len(sb.seqs)
        own = s["own"]
        own.masked_fill_(own < 0, self.rx[r].scratch)
        if not S:
            return
        i = self.d2h_i
        nbh = self.nb_h[i][r]
        ev = self.nb_ev[i][r]
        if ev is not None:
            ev.synchronize()
        nbh.numpy()[:S] = [se.n_blocks or len(se.block_table) for se in sb.seqs]
        nbd = self.nb_d[r]
        nbd[:S].copy_(nbh[:S], non_blocking=True)
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        self.nb_ev[i][r] = e
        seq = self.rx[r].out["seq"][:R]
        lim = nbd.index_select(0, seq.clamp(min=0).long())
        s["paged_d"].masked_fill_((seq >= 0) & (s["pos"] // 64 >= lim), -1)

    def _res_owner(self) -> None:
        """ENG_D13_RES_OWNER: HSD.result's first-call copies (num_sampled / sampled of every rank -> pinned, one wait),
        restricted to the ranks with drafting sequences; the others get empty lists (their results index none). Same
        buffers, streams and values as HSD.result for every rank it reads. Skipped when HSD's result state is not the
        vectorized one or is already filled."""
        H = self.eng.hooks
        hsd = H.hsd
        if hsd is None or not H.hsd_on or getattr(hsd, "prep", "1") == "0" or hsd._res_h is None \
                or any(x is not None for x in hsd._res):
            return
        c = hsd.cur
        n = c.n_req
        act = [r for r in range(W) if c.dseqs[r]]
        evs = []
        for r2 in act:
            st_ = hsd.st[r2]
            with torch.cuda.device(hsd.devs[r2]):
                hsd._res_h[r2][0][:n].copy_(st_.num_sampled[:n], non_blocking=True)
                hsd._res_h[r2][1][:n].copy_(st_.sampled[:n], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                evs.append(ev)
        for ev in evs:
            ev.synchronize()
        for r2 in range(W):
            hsd._res[r2] = (hsd._res_h[r2][0][:n].tolist(), hsd._res_h[r2][1][:n].tolist()) if r2 in act else ([], [])
        self.stats["res_owner"] = self.stats.get("res_owner", 0) + 1

    # ------------------------------------------------------------------------------------------ ENG_D13_EMPTY
    @staticmethod
    def _empty_eligible(sb, ct) -> bool:
        """No sequences, no rows, no rowexp ring ops (window page sets / slot clears) and no block-table deltas."""
        return (not sb.seqs and not sb.n_rows and (ct.win_slot is None or not len(ct.win_slot))
                and (ct.bt_upd_slot is None or not len(ct.bt_upd_slot)))

    def _empty_mark(self, r, s, R, eligible: bool) -> None:
        """After a full prep of rank r (inside its device / step-stream context): record whether it left the empty state."""
        if not eligible:
            self._r_empty[r] = False
            self._b_empty.pop((R, r), None)
            return
        self._r_empty[r] = True
        self._b_empty[(R, r)] = self._epoch
        self.stats["empty_full"] += 1
        if EMPTY_CHECK:                   # stream-ordered clones: the state right after this prep, before graph L
            self._esnap[(R, r)] = [t.clone() for t in self._empty_set(r, s, R)]

    def _empty_set(self, r, s, R) -> list:
        """Every device input an empty step writes and graph L / H read (not the blobs' unused / host-stale bytes)."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        d, rx, eg = s["idx"], self.rx[r], self.eng.hooks.engram[r]
        return [s["tok"][:R], s["valid"], s["samp"], s["pos"][:R], s["own"][:R], s["si"], s["sl"][:R], s["paged_d"],
                d["valid"], d["preq_slot"], d["rslot"], d["seq_lens"][1], d["seq_lens"][2], d["cslot"][1], d["cslot"][2],
                rx.d["counts"], rx.out["seq"][:R], eg.b_counts[:2]]

    def _empty_skip(self, r, s, sb, R) -> None:
        """The host-side state of an empty rank's prep; its device inputs stay as the last (equal) empty prep left them.
        d13_ev stays the bucket's last (completed) position-copy event: _positions reads no positions for 0 rows."""
        s["n"] = 0
        s["n_sample"] = 0
        self.eng.hooks.idx.upload_d13_host(r, s, sb)
        s["tab"] = NS(n=0, tokens=torch.zeros(0, dtype=torch.int64), sample_rows=[], seq={}, seq_rows=[])
        s["batch"] = sb
        self.stats["empty_skips"] += 1
        if EMPTY_CHECK:
            snap = self._esnap.get((R, r))
            with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                cur_ = self._empty_set(r, s, R)
                bad = [i for i, (a_, b_) in enumerate(zip(snap or [], cur_)) if not torch.equal(a_, b_)]
            self.stats["empty_checked"] += 1
            if snap is None or bad:
                self.stats["empty_mismatch"] += 1
                if self.stats["empty_mismatch"] <= 8:
                    self.eng.log(f"engine D13 EMPTY CHECK rank {r} R {R}: skipped prep differs from the last empty prep in "
                                 f"buffers {bad if snap is not None else 'no snapshot'} (indices into _empty_set)")
            if self.stats["empty_checked"] % 1000 == 0:
                self.eng.log(f"engine D13 EMPTY CHECK: {self.stats['empty_checked']} skips checked, "
                             f"{self.stats['empty_mismatch']} differing; {self.stats['empty_full']} full empty preps")

    # ------------------------------------------------------------------------------------------ ENG_D13_OWNER
    def _owner_sig(self, r, sb, ct):
        """Everything rank r's constant inputs are built from, or None when the step is not of the reusable kind (no
        sequences, ring ops / block-table deltas, non-decode rows, host-given tokens, a sequence that was not on this rank
        in the previous step)."""
        if not sb.seqs or (ct.win_slot is not None and len(ct.win_slot)) or (ct.bt_upd_slot is not None and len(ct.bt_upd_slot)):
            return None
        n = sb.n_rows
        if not (np.asarray(ct.tokens[:n]) < 0).all():
            return None
        eng = self.eng
        out = []
        for se in sb.seqs:
            if se.kind != "decode" or se.n_rows != K1:
                return None
            p = eng.prev_of(se.rid)
            if p is None or p[0] != r:
                return None
            out.append((se.rid, se.state_slot, se.row0, se.n_rows, se.n_blocks or len(se.block_table), p[1]))
        return (n, tuple(out), tuple(np.asarray(ct.sample_idx).tolist()))

    def _owner_mark(self, r, s, R, osig) -> None:
        """After a full prep of rank r: remember its signature and copy Engram's uploaded starts (before graph L resolves
        them in place); the rslot base was copied right after the indexer upload."""
        if osig is None:
            self._own_last[r] = None
            return
        eg = self.eng.hooks.engram[r]
        self._keep(s, "_own_sbase", eg.b_start)
        self._own_last[r] = (R, osig, self._epoch)
        self.stats["owner_full"] += 1
        cmp_ = self._own_cmp[r]
        if cmp_ is not None:                    # ENG_D13_OWNER_CHECK: graph T' vs this full path
            self._own_cmp[r] = None
            S = len(s["batch"].seqs)
            bad = [i for i, (a_, b_) in enumerate(zip(cmp_, self._owner_set(r, s, R, S))) if not torch.equal(a_, b_)]
            self.stats["owner_checked"] += 1
            if bad:
                self.stats["owner_mismatch"] += 1
                if self.stats["owner_mismatch"] <= 8:
                    self.eng.log(f"engine D13 OWNER CHECK rank {r} R {R}: graph T' differs from the full prep in buffers "
                                 f"{bad} (indices into _owner_set)")
            if self.stats["owner_checked"] % 1000 == 0:
                self.eng.log(f"engine D13 OWNER CHECK: {self.stats['owner_checked']} steady owner preps checked, "
                             f"{self.stats['owner_mismatch']} differing")

    @staticmethod
    def _keep(s, key, src) -> None:
        """s[key] = a stream-ordered copy of src in a buffer allocated once (stable address for graph T')."""
        buf = s.get(key)
        if buf is None or buf.shape != src.shape or buf.dtype != src.dtype:
            s[key] = src.clone()
        else:
            buf.copy_(src)

    def _owner_set(self, r, s, R, S) -> list:
        eg = self.eng.hooks.engram[r]
        return self._empty_set(r, s, R) + [eg.b_start[:S], eg.b_row0[:S], eg.b_nrows[:S], eg.b_slot[:S], eg.b_look[:S]]

    def _owner_skip(self, r, s, sb, ct, R) -> bool:
        """A steady owner rank: host state as the full prep sets it, graph T' on the step stream, the positions D2H.
        Returns False under OWNER_CHECK (the full path then runs and is compared)."""
        eng, H = self.eng, self.eng.hooks
        rx = self.rx[r]
        n = sb.n_rows
        with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
            cur = torch.cuda.current_stream()
            tok = torch.from_numpy(np.asarray(ct.tokens[:n], dtype=np.int64))
            tab = NS(n=n, tokens=tok, sample_rows=list(np.asarray(ct.sample_idx).tolist()), seq={},
                     seq_rows=[(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs])
            s["n"] = n
            s["n_sample"] = len(tab.sample_rows)
            H.idx.upload_d13_host(r, s, sb)
            rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(int(s["R"]), SW), sl=s["sl"], paged=s["paged_d"])
            self._gto_run(r, s, R, n)
            if OWNER_CHECK:
                self._own_cmp[r] = [t.clone() for t in self._owner_set(r, s, R, len(sb.seqs))]
                return False
            ph, oh = self.d2h[self.d2h_i][r]
            ph[:int(s["R"])].copy_(s["pos"], non_blocking=True)
            oh[:int(s["R"])].copy_(s["own"], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(cur)
            s["d13_ev"] = ev
            s["tab"], s["batch"] = tab, sb
        self.stats["owner_skips"] += 1
        return True

    def _gto_body(self, r, s, R, n) -> None:
        """Graph T' = the in-place-rewritten inputs reset from the full prep's copies (indexer rslot base, Engram starts),
        the device tokens (as Hooks._device_tokens_v2: rows / next-id indices from the full prep's upload), then graph T."""
        H = self.eng.hooks
        d = s["idx"]
        d["rslot"].copy_(s["_own_rbase"])
        H.engram[r].b_start.copy_(s["_own_sbase"])
        dt = H._dt_d[r]
        s["tok"].index_copy_(0, dt[0, :n], H.hsd.next_ids[r].view(-1).index_select(0, dt[1, :n]))
        self._gt_body(r, s, R, True)

    def _gto_run(self, r, s, R, n) -> None:
        H = self.eng.hooks
        f_ = None
        if TFUSE or TFUSE_CHECK:     # ENG_D13_TFUSE: graph T' as one kernel (kernels/d13tail), graph T' if unavailable
            DT = _d13tail()
            if DT:
                try:
                    DT.run_owner(self, r, s, int(s["R"]), n)   # (rows; R = the key)
                except Exception as e:  # noqa: BLE001  (the ext's argument checks fail before any write: T' is safe)
                    global _D13TAIL
                    _D13TAIL = DT = False
                    self.eng.log(f"engine D13 TFUSE: run failed ({type(e).__name__}: {str(e)[:200]}); graph T' from now on")
            if DT:
                if not TFUSE_CHECK:
                    self.stats["tfuse"] = self.stats.get("tfuse", 0) + 1
                    return
                f_ = [t.clone() for t in self._tfuse_outs(r, s, R)]
        k = (R, r, "own", n)
        ptrs = self._gt_ptrs(r, s) + (s["_own_rbase"].data_ptr(), s["_own_sbase"].data_ptr(),
                                      H.engram[r].b_start.data_ptr(), H._dt_d[r].data_ptr(), H.hsd.next_ids[r].data_ptr(),
                                      s["tok"].data_ptr())
        ent = self.gt.get(k)
        if ent is None or ent[1] != ptrs:
            if ent is not None:
                self.stats["gt_recaptures"] += 1
            dev = self.eng.devs[r]
            if self.gt_pool[r] is None:
                with torch.cuda.device(dev):
                    self.gt_pool[r] = torch.cuda.graph_pool_handle()
                    self.gt_stream[r] = torch.cuda.Stream(device=dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(dev), torch.cuda.stream(self.gt_stream[r]):
                g.capture_begin(pool=self.gt_pool[r], capture_error_mode="thread_local")
                try:
                    self._gto_body(r, s, R, n)
                finally:
                    g.capture_end()
            self.gt[k] = (g, ptrs)
            self.stats["gt_captures"] += 1
            if self.stats["gt_captures"] <= 64:
                self.eng.log(f"engine D13 GT: graph T' {'re' if ent is not None else ''}captured for (R, rank, own, rows) = {k}")
            ent = self.gt[k]
        ent[0].replay()
        self.stats["gt_replays"] += 1
        if f_ is not None:
            self._tfuse_check(r, s, R, f_)

    def _tfuse_outs(self, r, s, R) -> list:
        """Every buffer graph T' writes (GT_CHECK's outputs + Engram's b_start + the device tokens; not rx.err)."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        rx, d = self.rx[r], s["idx"]
        return [rx.out["pos"][:R], rx.out["own"][:R], rx.out["si"][:R], rx.out["sl"][:R], s["paged_d"][:R], rx.out["start"],
                rx.out["seq"][:R], d["rslot"], d["seq_lens"][1], d["seq_lens"][2], d["cslot"][1], d["cslot"][2],
                self.eng.hooks.engram[r].b_start, s["tok"]]

    def _tfuse_check(self, r, s, R, f_) -> None:
        """ENG_D13_TFUSE_CHECK: the fused kernel's outputs (f_) vs graph T' replayed after it on the same inputs (T' resets
        rslot / b_start from its bases; ring ops and the device tokens are idempotent). Synchronizes the step stream."""
        names = ("pos", "own", "si", "sl", "paged", "start", "seq", "rslot", "seq_lens1", "seq_lens2", "cslot1", "cslot2",
                 "b_start", "tok")
        bad = [nm for nm, a, b in zip(names, f_, self._tfuse_outs(r, s, R)) if not torch.equal(a, b)]
        self.stats["tfuse_checked"] = self.stats.get("tfuse_checked", 0) + 1
        if bad:
            self.stats["tfuse_mismatch"] = self.stats.get("tfuse_mismatch", 0) + 1
            if self.stats["tfuse_mismatch"] <= 8:
                self.eng.log(f"engine D13 TFUSE CHECK rank {r} R {R}: fused tail differs from graph T' in {bad}")
        if self.stats["tfuse_checked"] % 1000 == 0:
            self.eng.log(f"engine D13 TFUSE CHECK: {self.stats['tfuse_checked']} owner steps checked, "
                         f"{self.stats.get('tfuse_mismatch', 0)} differing")

    # ------------------------------------------------------------------------------------------ ENG_D13_GT (graph T)
    def _guard_host(self, r, sb) -> None:
        """_guard's host half: the sequences' allocated block counts -> nb_d[r] (pinned set i, event-guarded), exactly
        as _guard does; its device half is in graph T."""
        S = len(sb.seqs)
        i = self.d2h_i
        nbh = self.nb_h[i][r]
        ev = self.nb_ev[i][r]
        if ev is not None:
            ev.synchronize()
        nbh.numpy()[:S] = [se.n_blocks or len(se.block_table) for se in sb.seqs]
        self.nb_d[r][:S].copy_(nbh[:S], non_blocking=True)
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        self.nb_ev[i][r] = e

    def _gt_body(self, r, s, R, has_seqs: bool) -> None:
        """Graph T's ops = the eager sequence of _pre: rowexp begin(R); _guard's device half (own < 0 -> scratch; rows of
        a sequence at or past its allocated blocks get paged -1: only with sequences on this rank, as in _guard); the
        indexer's device_tables_d13. Every operand is a per-bucket / per-rank buffer; R and the scratch slot are fixed."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        rx = self.rx[r]
        rx.begin(R)
        own = s["own"]
        own.masked_fill_(own < 0, rx.scratch)
        if has_seqs:
            seq = rx.out["seq"][:R]
            lim = self.nb_d[r].index_select(0, seq.clamp(min=0).long())
            s["paged_d"].masked_fill_((seq >= 0) & (s["pos"] // 64 >= lim), -1)
        self.eng.hooks.idx.device_tables_d13(r, s, s["paged_d"])

    def _gt_ptrs(self, r, s) -> tuple:
        """Addresses graph T bakes in (a different buffer behind any of them -> recapture)."""
        rx, d = self.rx[r], s["idx"]
        o = rx.out
        return (o["pos"].data_ptr(), o["own"].data_ptr(), o["si"].data_ptr(), o["sl"].data_ptr(), o["paged"].data_ptr(),
                o["start"].data_ptr(), o["seq"].data_ptr(), s["paged_d"].data_ptr(), self.nb_d[r].data_ptr(),
                d["valid"].data_ptr(), d["rslot"].data_ptr(), d["seq_lens"][1].data_ptr(), d["seq_lens"][2].data_ptr(),
                d["cslot"][1].data_ptr(), d["cslot"][2].data_ptr(), rx.d["counts"].data_ptr(), rx.ring.data_ptr(),
                rx.dbt.data_ptr(), rx.nc.data_ptr())

    def _gt_run(self, r, s, R, has_seqs: bool) -> None:
        """Replay graph T of (R, r, has_seqs) on the current (step) stream, capturing it first if needed (on a side
        stream: nothing executes during a capture, so the step stream's queued work is not involved)."""
        k = (R, r, has_seqs)
        ptrs = self._gt_ptrs(r, s)
        ent = self.gt.get(k)
        if ent is None or ent[1] != ptrs:
            if ent is not None:
                self.stats["gt_recaptures"] += 1
                self.eng.log(f"engine D13 GT: recapture graph T {k} (a bound buffer moved)")
            dev = self.eng.devs[r]
            if self.gt_pool[r] is None:
                with torch.cuda.device(dev):
                    self.gt_pool[r] = torch.cuda.graph_pool_handle()
                    self.gt_stream[r] = torch.cuda.Stream(device=dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(dev), torch.cuda.stream(self.gt_stream[r]):
                g.capture_begin(pool=self.gt_pool[r], capture_error_mode="thread_local")
                try:
                    self._gt_body(r, s, R, has_seqs)
                finally:
                    g.capture_end()
            ent = self.gt[k] = (g, ptrs)
            self.stats["gt_captures"] += 1
            if self.stats["gt_captures"] <= 64:
                self.eng.log(f"engine D13 GT: graph T captured for (R, rank, seqs) = {k}")
        if GT_CHECK:
            rs0 = s["idx"]["rslot"].clone()        # graph T adds pos % 8 to the uploaded rslot base: keep the base
        ent[0].replay()
        self.stats["gt_replays"] += 1
        if GT_CHECK:
            self._gt_check(r, s, R, has_seqs, rs0)

    def _gt_check(self, r, s, R, has_seqs, rs0) -> None:
        """ENG_D13_GT_CHECK: graph T's outputs vs the eager ops on the same inputs (rowexp's ring ops are idempotent;
        rslot restored to its uploaded base first); the step then uses the eager values. Synchronizes the step stream."""
        R = int(s["R"])                  # rows (ENG_D13_LONG: the caller's R is the bucket key Rk)
        rx, d = self.rx[r], s["idx"]
        outs = lambda: [rx.out["pos"][:R], rx.out["own"][:R], rx.out["si"][:R], rx.out["sl"][:R], s["paged_d"][:R],  # noqa: E731
                        rx.out["start"], rx.out["seq"][:R], d["rslot"], d["seq_lens"][1], d["seq_lens"][2],
                        d["cslot"][1], d["cslot"][2]]     # (not rx.err: k_expand counts ring misses again)
        g_ = [t.clone() for t in outs()]
        d["rslot"].copy_(rs0)
        self._gt_body(r, s, R, has_seqs)
        names = ("pos", "own", "si", "sl", "paged", "start", "seq", "rslot", "seq_lens1", "seq_lens2", "cslot1", "cslot2")
        bad = [n for n, a, b in zip(names, g_, outs()) if not torch.equal(a, b)]
        self.stats["gt_checked"] += 1
        if self.stats["gt_checked"] % 1000 == 0:
            self.eng.log(f"engine D13 GT CHECK: {self.stats['gt_checked']} rank-steps checked, {self.stats['gt_mismatch']} "
                         f"differing; {self.stats['gt_captures']} graphs T captured")
        if bad:
            self.stats["gt_mismatch"] += 1
            if self.stats["gt_mismatch"] <= 8:
                self.eng.log(f"engine D13 GT CHECK rank {r} R {R} seqs {has_seqs}: graph T differs from the eager ops in {bad}")

    def _positions(self, st, batches) -> list:
        """Wait for this step's expanded positions (D2H right behind the previous step) -> StepBatches with positions /
        window slots for HSD's host prepare (the rest of each batch is shared)."""
        out = []
        for r in range(W):
            st[r]["d13_ev"].synchronize()
            sb, n = batches[r], batches[r].n_rows
            ph, oh = self.d2h[self.d2h_i][r]
            out.append(dataclasses.replace(sb, positions=ph[:n].tolist(), window_slot=oh[:n].tolist()) if n else sb)
            if CHECK and n:     # HSD's drafter writes / reads window KV for [start - 127, start + 2k]: pages present?
                miss = []
                pos = out[-1].positions
                for se in sb.seqs:
                    s0 = pos[se.row0]
                    lo = max(se.replay_start, s0 - 127)
                    need = range(lo // 32, (s0 + 2 * 3) // 32 + 1)      # (window_end = start + 2k + 1, exclusive)
                    m = [b for b in need if b not in se.window_table]
                    if m:
                        miss.append((str(se.rid), s0, m))
                if miss:
                    self.eng.log(f"engine D13 CHECK rank {r} step {sb.step}: {len(miss)} sequences miss drafter-window "
                                 f"pages, e.g. {miss[:3]}")
                    self.stats["drafter_window_miss"] = self.stats.get("drafter_window_miss", 0) + len(miss)
        return out

    def _set_prev_seq(self, batches) -> None:
        """HSD's next-id rows of this step = its batch's sequence order (what head_result records in the sync loop; here
        at dispatch, after HSD's prepare, so the next step's device tokens find them before this step's extraction)."""
        if not self.eng.hooks.hsd_on:
            return
        ps = self.eng.prev_seq
        sn = self.eng.step_serial
        for r, sb in enumerate(batches):
            for si, se in enumerate(sb.seqs):
                ps[se.rid] = (r, si, sn)

    def _advance(self, st, batches) -> None:
        """k_advance: nc[slot] = start + emitted (HSD's per-request count, gathered into the step's sequence order)."""
        hsd = self.eng.hooks.hsd
        c = hsd.cur
        self.qi ^= 1
        for r in range(W):
            S = len(batches[r].seqs)
            if not S:
                continue
            if ADV_SLICE:
                q0 = r * c.D_pad
                ds = c.dseqs[r]
                if len(ds) == S and all(si == j and qi == q0 + j for j, (si, qi) in enumerate(ds)):
                    with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                        self.rx[r].end(hsd.st[r].num_sampled[q0:q0 + S].to(torch.int32))
                    self.stats["adv_slice"] = self.stats.get("adv_slice", 0) + 1
                    continue
            ev = self.qidx_ev[self.qi][r]
            if ev is not None:
                ev.synchronize()
            qh = self.qidx_h[self.qi][r]
            q = qh.numpy()
            q[:S] = 0
            for si, qi in c.dseqs[r]:             # (seq index in batch.seqs, request index)
                q[si] = qi
            with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                cur = torch.cuda.current_stream()
                qd = self.qidx_d[r]
                qd[:S].copy_(qh[:S], non_blocking=True)
                e2 = torch.cuda.Event()
                e2.record(cur)
                self.qidx_ev[self.qi][r] = e2
                acc = hsd.st[r].num_sampled.index_select(0, qd[:S]).to(torch.int32)
                self.rx[r].end(acc)

    def _record(self) -> list:
        evs = []
        for r in range(W):
            with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                e = torch.cuda.Event()
                e.record()
                evs.append(e)
        return evs

    def _deltas(self, r, ct) -> None:
        """D12 block-table deltas of a CompactTables into the per-slot device table (on the current stream)."""
        if ct.bt_upd_slot is None or not len(ct.bt_upd_slot):
            return
        nu, rows, cols, pages = self._delta_rows(ct)
        hx_ = self.eng.hooks.idx   # its map follows
        mir_ = getattr(hx_, "mirror_deltas", None) if hx_ is not None and getattr(hx_, "mir", None) else None   # the deltas
        if not nu:
            if mir_ is not None:   # (zero-length deltas: a new sequence with no pages yet in that slot = a reset for the mirror)
                e_ = np.zeros(0, np.int64)
                mir_(r, e_, e_, np.zeros(0, np.int32), ct.bt_upd_slot, ct.bt_upd_start)
            return
        if mir_ is not None:   # (before either dbt branch; device work on the current stream, as dbt's update)
            mir_(r, rows, cols, pages, ct.bt_upd_slot, ct.bt_upd_start)
        dbt = self.rx[r].dbt
        if nu > DELTA_CAP:
            dev = dbt.device
            dbt.index_put_((torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)), torch.from_numpy(pages).to(dev))
            return
        i = self.dl_i
        ev = self.dl_ev[i][r]
        if ev is not None:
            ev.synchronize()
        hr, hc, hp = self.dl_h[i][r]
        hr.numpy()[:nu], hc.numpy()[:nu], hp.numpy()[:nu] = rows, cols, pages
        dr, dc, dp = self.dl_d[r]
        dr[:nu].copy_(hr[:nu], non_blocking=True)
        dc[:nu].copy_(hc[:nu], non_blocking=True)
        dp[:nu].copy_(hp[:nu], non_blocking=True)
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        self.dl_ev[i][r] = e
        dbt.index_put_((dr[:nu], dc[:nu]), dp[:nu])

    @staticmethod
    def _delta_rows(ct):
        """_deltas' host half: (nu, rows, cols, pages) of a CompactTables' D12 deltas (nu = 0: zero-length, None x 3)."""
        ln = np.asarray(ct.bt_upd_len, np.int64)
        nu = int(ln.sum())
        if not nu:
            return 0, None, None, None
        rows = np.repeat(np.asarray(ct.bt_upd_slot, np.int64), ln)
        st0 = np.asarray(ct.bt_upd_start, np.int64)
        cols = (np.arange(nu, dtype=np.int64) - np.repeat(np.cumsum(ln) - ln, ln)) + np.repeat(st0, ln)
        pages = np.asarray(ct.bt_upd_pages[:nu], np.int32)
        return nu, rows, cols, pages

    # ------------------------------------------------------------------------------------------ results
    def _post_extract(self, h: StepHandle) -> None:
        """After a step's results: rowexp's miss counters (every ERR_EVERY-th extraction), the in-flight slot."""
        eng = self.eng
        self._n_extract += 1
        for r in (range(W) if self._n_extract % ERR_EVERY == 0 else ()):   # rowexp ring misses (a window page the host
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(self.xstream[r]):   # never sent: must stay 0 for
                e_ = int(self.rx[r].err.sum())                                          # live rows; a host sync each)
            if e_ != self.err_seen[r]:
                eng.log(f"engine D13: rank {r} rowexp window-page misses {e_ - self.err_seen[r]} at step {h.step}"
                        f"{'' if ERR_EVERY == 1 else f' (counted over the last {ERR_EVERY} steps)'} "
                        f"(rows past their allocation get the scratch slot; a live row here is a host-side bug)")
                self.stats["rowexp_miss"] = self.stats.get("rowexp_miss", 0) + e_ - self.err_seen[r]
                self.err_seen[r] = e_
        if self.inflight is h:
            self.inflight = None

    def _extract(self, h: StepHandle) -> None:
        if getattr(h, "deferred", None) is not None:      # ENG_D13S=on: a steady step's results (engine/d13s.py)
            return self.steady.extract_deferred(h)
        eng = self.eng
        t0 = time.perf_counter()
        for e in h.ev:
            e.synchronize()
        t1 = time.perf_counter()
        st = eng.bucket(h.R)
        res = []
        H = eng.hooks
        g_now = H.grammar
        H.grammar = getattr(h, "grammar", g_now)        # this step's grammar rows (on_ctx of the next step reset it)
        import contextlib
        try:
            with contextlib.ExitStack() as es:          # every device's current stream = its extraction stream
                for r in range(W):
                    self.xstream[r].wait_event(h.ev[r])
                    es.enter_context(torch.cuda.stream(self.xstream[r]))
                if RES_OWNER:
                    self._res_owner()
                for r in range(W):
                    s_ = dict(st[r])
                    s_["batch"], s_["tab"] = h.batches[r], h.tabs[r]
                    res.append(H.head_result(r, s_, batch=h.batches[r], async_=True))
        finally:
            H.grammar = g_now
        h._res = res
        self._post_extract(h)
        if h.cap_h is not None:
            if self._dispatch_tid in (None, threading.get_ident()):
                self._blas_ready()             # (this thread's cuBLAS handles, outside the capture)
                eng._capture_split(h.R, st, False, h.cap_h)
            else:
                # extracted on another thread (served D13: the adapter's merge thread). Capturing here would race the
                # dispatch thread's CUDA calls (global capture mode: its Worker.copies synchronize invalidates the capture)
                # and needs this thread's cuBLAS handles (cublasCreate inside the capture -> EngineCore dead). The next
                # dispatch captures it first, before any HSD prepare (HSD still holds this step's prepared state: a key's
                # capture must see that key's prepare).
                self._pending_caps.append((h.R, h.cap_h))
                self.stats["deferred_caps"] = self.stats.get("deferred_caps", 0) + 1
            h.cap_h = None
        h.t.update(extract_wait=round((t1 - t0) * 1000, 3), extract=round((time.perf_counter() - t1) * 1000, 3))
        eng.stats["d13_last"] = dict(h.t)

    def _blas_ready(self) -> None:
        """Before a graph H capture in _extract: that capture runs on whichever thread extracts the step first, in served
        D13 often the adapter's merge thread (handle.result()), not the dispatch thread. cuBLAS / cuBLASLt handles are
        per thread and device and their workspaces per stream; the first GEMM of a thread inside a capture calls
        cublasCreate, which is illegal there (CUBLAS_STATUS_NOT_INITIALIZED in HSD's
        drafter context_kv mm -> cudaErrorStreamCaptureInvalidated, EngineCore dead). So: one tiny bf16 + fp32 GEMM per
        device on every stream the capture uses (the step stream, HSD's side / side_ctx), outside the capture, once per
        thread."""
        tid = threading.get_ident()
        if tid in self._blas_threads:
            return
        hsd = self.eng.hooks.hsd
        for r, dev in enumerate(self.eng.devs):
            ss = [self.eng.streams[r]] + [x[r] for x in (getattr(hsd, "side", None), getattr(hsd, "side_ctx", None)) if x]
            with torch.cuda.device(dev):
                for s_ in ss:
                    with torch.cuda.stream(s_):
                        torch.cuda.current_blas_handle()
                        a = torch.ones(64, 64, device=dev, dtype=torch.bfloat16)
                        torch.mm(a, a)
                        torch.mm(a.float(), a.float())
                        torch.bmm(a.view(1, 64, 64), a.view(1, 64, 64))
        self._blas_threads.add(tid)
        self.stats["blas_threads"] = len(self._blas_threads)

    def _run_pending_caps(self) -> None:
        """Graph H captures deferred by _extract (another thread extracted), on the dispatch thread, before this step's
        HSD prepare. A key captured meanwhile (an eager step of the same key) is skipped."""
        self._dispatch_tid = threading.get_ident()
        while self._pending_caps:
            R, hk = self._pending_caps.pop(0)
            if hk in self.eng.graphs:
                continue
            self._blas_ready()
            self.eng._capture_split(R, self.eng.bucket(R), False, hk)

    def drain(self) -> None:
        with self.lock:
            if self.inflight is not None:
                self._extract(self.inflight)

    # ------------------------------------------------------------------------------------------ first step of a bucket
    def _first(self, batches, cts, R, mo=None) -> StepHandle:
        """The bucket's first async step: eager (after draining), then graph L (key (R, "LA")) and graph H captured."""
        eng, H = self.eng, self.eng.hooks
        self.drain()
        self._epoch += 1                       # ENG_D13_EMPTY: the eager first step of a bucket
        self._r_empty = [False] * W
        self._own_last = [None] * W            # ENG_D13_OWNER: the same
        if self.steady is not None:            # ENG_D13S: the next async step is not steady w.r.t. this one
            self.steady.prev_sig = None
        st = eng.bucket(R)
        eng.bump_serial(batches)
        eng._tag_mega(st, mo)              # ENG_C1_MEGA: mo = the C1 owner (None: the engine's pieces)
        self._pre(st, batches, cts, R)
        eng._new_scheds(st)
        eng._issue(st)
        pb = self._positions(st, batches)
        eng._prepare_head_on_streams(st, pb)
        self._set_prev_seq(batches)
        H.head_eager(st)
        self._advance(st, batches)
        h = StepHandle(self, batches[0].step if batches else 0)
        h.ev = self._record()
        h.batches, h.tabs, h.R = list(batches), [st[r]["tab"] for r in range(W)], R
        h.grammar = dict(H.grammar)
        hk = eng._hkey(R)
        self._extract(h)
        abk = getattr(eng, "_ab_key", None)   # (tolerates an older engine.py: see the dispatch)
        eng._capture_split(R, st, True, hk, lkey=(R, "LA") if mo is None else (R, "LA", "M", mo) + (abk() if abk else ()))
        eng.last_st = st
        eng.stats["steps"] += 1
        eng.stats["eager_steps"] += 1
        self.stats["first_steps"] += 1
        return h

    def _check_fallback(self, rb, tables) -> None:
        """ENG_D13_CHECK=1: rowexp's expansion of this step (its internal outputs, from the device ring / nc / per-slot
        table) vs the host tables the fallback built from the live KV tables, row by row: position, own window slot,
        window keys (si / sl), paged slot. A mismatch = the device ring and the host tables disagree."""
        for r in range(W):
            n = rb[r].n_rows
            if not n:
                continue
            o = self.rx[r].out
            with torch.cuda.device(self.eng.devs[r]), torch.cuda.stream(self.eng.streams[r]):
                pos, own = o["pos"][:n].cpu(), o["own"][:n].cpu()
                si, sl, pg = o["si"][:n].cpu(), o["sl"][:n].cpu(), o["paged"][:n].cpu()
            tab = tables[r]
            bad = {}
            for name, a_, b_ in (("pos", pos, tab.positions.long()), ("own", own, tab.own_slot.long()),
                                 ("sl", sl, tab.sl.long()), ("paged", pg, tab.paged_slot.long())):
                m = (a_ != b_).nonzero().flatten()
                if m.numel():
                    i = int(m[0])
                    bad[name] = (int(m.numel()), i, int(a_[i]), int(b_[i]))
            sim = (si.long() != tab.si.view(n, -1).long()).any(1).nonzero().flatten()
            if sim.numel():
                bad["si"] = (int(sim.numel()), int(sim[0]))
            if bad:
                seqs = rb[r].seqs
                sid = rb[r].seq_slot
                info = {k: v + ((str(seqs[sid[v[1]]].rid), seqs[sid[v[1]]].kind) if len(v) > 1 else ()) for k, v in bad.items()}
                self.eng.log(f"engine D13 CHECK rank {r} step {rb[r].step}: rowexp vs host tables differ {info}")
                self.stats["check_mismatch"] = self.stats.get("check_mismatch", 0) + 1

    # ------------------------------------------------------------------------------------------ synchronous fallback
    def _fb_rows(self, batches, starts) -> list:
        """_fallback's host half: the resolved StepBatches (device starts for the decode sequences whose start the device
        holds; every row's position / paged slot / window slot) the private TableBuilder builds full tables from."""
        from interfaces import PAD
        rb = []
        for r in range(W):
            sb = batches[r]
            seqs, pos, paged, win = [], [], [], []
            scratch = self.rx[r].scratch
            reuse = FB_REUSE and len(sb.positions) == sb.n_rows
            P0, G0, W0 = sb.positions, sb.paged_slot, sb.window_slot
            for se, s0 in zip(sb.seqs, starts[r]):
                if reuse and se.start >= 0:          # host-known: build_step_batch's rows (FB_REUSE)
                    seqs.append(se)
                    a_, b_ = se.row0, se.row0 + se.n_rows
                    pos.extend(P0[a_:b_])
                    paged.extend(G0[a_:b_])
                    win.extend(W0[a_:b_])
                    continue
                if se.start < 0:
                    se = dataclasses.replace(se, start=int(s0), num_computed=int(s0),
                                             n_blocks=min(-(-(int(s0) + se.n_rows) // 64), len(se.block_table)))
                seqs.append(se)
                pg, wt, cu = se.block_table, se.window_table, se.cached_upto
                nbk = se.n_blocks or len(pg)
                for p in range(se.start, se.start + se.n_rows):
                    pos.append(p)            # (rows past the allocation: a finished sequence's wasted step) no writes
                    paged.append(PAD if (p < cu or p // 64 >= min(nbk, len(pg))) else pg[p // 64] * 64 + p % 64)
                    w_ = wt.get(p // 32)
                    win.append(scratch if w_ is None else w_ * 32 + p % 32)
            rb.append(dataclasses.replace(sb, seqs=seqs, positions=pos, paged_slot=paged, window_slot=win))
        return rb

    def _fallback(self, batches, cts, ctx, why: str) -> StepHandle:
        """Drain, resolve every start on the device (rowexp: ring ops + expand, its internal outputs), D2H the starts,
        full tables from the resolved StepBatches (private TableBuilder: its block-table deltas rewrite the same pages),
        the synchronous engine.step, then k_advance with the step's accepted counts."""
        eng = self.eng
        t_fb0 = time.perf_counter()
        self._epoch += 1                       # ENG_D13_EMPTY: a synchronous step rewrites tables / descriptors
        self._r_empty = [False] * W
        self._own_last = [None] * W            # ENG_D13_OWNER: the same
        if self.steady is not None:            # ENG_D13S: the next async step is not steady w.r.t. this one
            self.steady.prev_sig = None
        self.drain()
        if CHECK:
            self._remap_check(batches)
        if os.environ.get("ENG_D13_DUMP_NEXT"):   # debug: HSD's next ids / per-slot store as the previous step left them
            hs = eng.hooks.hsd
            torch.save(dict(step=batches[0].step, next_ids=[x.cpu() for x in hs.next_ids], by_slot=[x.cpu() for x in hs.by_slot],
                            prev_seq=dict(eng.prev_seq)), os.environ["ENG_D13_DUMP_NEXT"] + f".s{batches[0].step}.pt")
        from interfaces import PAD
        tf = [time.perf_counter()]
        self.dl_i ^= 1
        starts = []
        for r in range(W):
            sb, ct, rx = batches[r], cts[r], self.rx[r]
            S = len(sb.seqs)
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
                self._deltas(r, ct)
                rx.bind_outputs(**self.rx_own[r])
                rx.upload(self.RX.RowExp.desc_of_compact(ct), stream=torch.cuda.current_stream())
                rx.begin(max(sb.n_rows, 1))
                starts.append(rx.out["start"][:max(S, 1)].to("cpu", non_blocking=False)[:S].tolist())
        tf.append(time.perf_counter())
        rb, tables = self._fb_rows(batches, starts), []
        t_tab = 0.0
        for r in range(W):
            sb, ct = batches[r], cts[r]
            t_ = time.perf_counter()
            tables.append(self.tb.build(rb[r], [int(x) for x in np.asarray(ct.tokens[:sb.n_rows]).tolist()]))
            t_tab += time.perf_counter() - t_
        tf.append(time.perf_counter())
        if CHECK:
            self._check_fallback(rb, tables)
        res = eng.step(rb, tables, ctx=ctx)
        tf.append(time.perf_counter())
        for r in range(W):
            sb, S = rb[r], len(rb[r].seqs)
            if not S:
                continue
            a = self.acc_h[r].numpy()
            for k, se in enumerate(sb.seqs):
                a[k] = int(res[r].accepted.get(se.rid, se.n_rows)) if se.kind == "decode" else se.n_rows
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
                self.rx[r].accepted[:S].copy_(self.acc_h[r][:S], non_blocking=True)
                self.rx[r].end()
            torch.cuda.synchronize(eng.devs[r])            # (acc_h is reused next fallback step)
        tf.append(time.perf_counter())
        ms = lambda a_, b_: round((b_ - a_) * 1000, 3)  # noqa: E731
        fb = dict(why=why, rows=[sb.n_rows for sb in batches], drain=ms(t_fb0, tf[0]), resolve=ms(tf[0], tf[1]),
                  rows_tables=ms(tf[1], tf[2]), tables=round(t_tab * 1000, 3), step=ms(tf[2], tf[3]), advance=ms(tf[3], tf[4]))
        eng.stats["d13_fb_last"] = fb
        acc_ = self.stats.setdefault("fb_ms", dict(n=0, drain=0.0, resolve=0.0, rows_tables=0.0, tables=0.0, step=0.0,
                                                   advance=0.0))
        acc_["n"] += 1
        for k_ in ("drain", "resolve", "rows_tables", "tables", "step", "advance"):
            acc_[k_] = round(acc_[k_] + fb[k_], 3)
        self.stats["fallback_steps"] += 1
        self.stats.setdefault("fallback_why", {})[why] = self.stats.get("fallback_why", {}).get(why, 0) + 1
        if not any(se.kind != "decode" for sb in batches for se in sb.seqs):   # decode-only fallbacks, per reason (the
            wd = self.stats.setdefault("fallback_why_dec", {})                    # engine's steplog line "fb dec ...")
            wd[why] = wd.get(why, 0) + 1
        return StepHandle(self, batches[0].step if batches else 0, results=res)
