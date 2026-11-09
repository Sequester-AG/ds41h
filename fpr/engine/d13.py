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

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = 4
K1 = 4                    # rows per decode sequence (k + 1, k = 3)
SW = 128                  # sliding window (si columns)
CHECK = os.environ.get("ENG_D13_CHECK", "") == "1"     # debug: fallback steps compare rowexp's rows with the host tables
DELTA_CAP = 65_536        # block-table delta entries per rank and step through pinned staging (more: direct copies)
FB_REUSE = os.environ.get("ENG_D13_FB_REUSE", "1") == "1"   # fallback steps keep the StepBatch's
                          # own positions / paged / window slots for host-known sequences (start >= 0: build_step_batch
                          # computed them with the same formulas) and rebuild only the device-resolved decode rows. Same
                          # lists (CPU check on 9 CP steps x 4 ranks: 0 mismatches); 2.7 -> 0.6 ms of host per 17K-row step


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
        # extraction stream per device: step N's result copies wait on N's done event only, not behind graph L(N + 1)
        # (on the engine stream they queued after it: extract 35 ms, H(N + 1) launched a step late)
        self.xstream = [torch.cuda.Stream(device=d) for d in eng.devs]
        from worker import TableBuilder
        self.tb = TableBuilder(stub_indexer=False)      # full tables of the synchronous fallback steps
        self._blas_threads = set()                       # threads whose cuBLAS / cuBLASLt handles exist (see _blas_ready)
        self._dispatch_tid = None                        # the thread that dispatches (served: the EngineCore loop)
        self._pending_caps = []                          # graph H captures of steps another thread extracted (_extract)
        self.stats = dict(async_steps=0, fallback_steps=0, first_steps=0)

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

    # ------------------------------------------------------------------------------------------ dispatch
    def dispatch(self, batches, cts, ctx=None) -> StepHandle:
        with self.lock:
            eng = self.eng
            t0 = time.perf_counter()
            self._run_pending_caps()
            why = self._async_ok(batches, cts)
            if why:
                return self._fallback(batches, cts, ctx, why)
            eng.hooks.on_ctx(ctx, None, None)
            if CHECK:
                self._remap_check(batches)
            n = max(sb.n_rows for sb in batches)
            R = eng.bucket_of(max(n, 1))
            lk = (R, "LA")
            if lk not in eng.graphs:
                return self._first(batches, cts, R)
            st = eng.bucket(R)
            eng.bump_serial(batches)           # (PREV_STEP_CHECK: this async step's serial, before its device-token reads)
            self._pre(st, batches, cts, R)
            t1 = time.perf_counter()
            eng.replay(lk)
            t2 = time.perf_counter()
            if self.inflight is not None:
                self._extract(self.inflight)
            t3 = time.perf_counter()
            pb = self._positions(st, batches)
            t4 = time.perf_counter()
            eng._prepare_head_on_streams(st, pb)
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
            eng.last_st = st
            eng.stats["steps"] += 1
            eng.stats["graph_steps"] += 1
            self.stats["async_steps"] += 1
            self.inflight = h
            return h

    def _pre(self, st, batches, cts, R) -> None:
        """Steps 1-3 of dispatch (see the module doc) for every rank, on its step stream."""
        eng, H = self.eng, self.eng.hooks
        EG = H.EG
        eng._batches = batches
        self.d2h_i ^= 1
        self.dl_i ^= 1
        for r in range(W):
            s, sb, ct, rx = st[r], batches[r], cts[r], self.rx[r]
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
                cur = torch.cuda.current_stream()
                self._deltas(r, ct)
                n = sb.n_rows
                tok = torch.from_numpy(np.asarray(ct.tokens[:n], dtype=np.int64)) if n else torch.zeros(0, dtype=torch.int64)
                tab = NS(n=n, tokens=tok, sample_rows=list(np.asarray(ct.sample_idx).tolist()), seq={},
                         seq_rows=[(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs])
                eng._upload_v2(r, s, tab, dev_pos=True)      # tok (+ device next ids) / valid / samp
                H.idx.upload_d13(r, s, sb)
                rx.upload(self.RX.RowExp.desc_of_compact(ct), stream=cur)
                if "paged_d" not in s:
                    s["paged_d"] = torch.full((R,), -1, dtype=torch.int64, device=s["dev"])
                rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(R, SW), sl=s["sl"], paged=s["paged_d"])
                rx.begin(R)
                if CHECK:
                    self._check_tables(r, s, sb, R)
                self._guard(r, s, sb, R)
                ph, oh = self.d2h[self.d2h_i][r]
                ph[:R].copy_(s["pos"], non_blocking=True)
                oh[:R].copy_(s["own"], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(cur)
                s["d13_ev"] = ev
                H.idx.device_tables_d13(r, s, s["paged_d"])
                a = EG.step_args_from_compact(ct, H._lookback, pad_to=R)
                H.engram[r].upload(a, stream=cur)
                s["tab"], s["batch"] = tab, sb

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
        ln = np.asarray(ct.bt_upd_len, np.int64)
        nu = int(ln.sum())
        if not nu:
            return
        rows = np.repeat(np.asarray(ct.bt_upd_slot, np.int64), ln)
        st0 = np.asarray(ct.bt_upd_start, np.int64)
        cols = (np.arange(nu, dtype=np.int64) - np.repeat(np.cumsum(ln) - ln, ln)) + np.repeat(st0, ln)
        pages = np.asarray(ct.bt_upd_pages[:nu], np.int32)
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

    # ------------------------------------------------------------------------------------------ results
    def _extract(self, h: StepHandle) -> None:
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
                for r in range(W):
                    s_ = dict(st[r])
                    s_["batch"], s_["tab"] = h.batches[r], h.tabs[r]
                    res.append(H.head_result(r, s_, batch=h.batches[r], async_=True))
        finally:
            H.grammar = g_now
        h._res = res
        for r in range(W):                 # rowexp ring misses (a window page the host never sent: must stay 0 for live rows)
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(self.xstream[r]):
                e_ = int(self.rx[r].err.sum())
            if e_ != self.err_seen[r]:
                eng.log(f"engine D13: rank {r} rowexp window-page misses {e_ - self.err_seen[r]} at step {h.step} "
                        f"(rows past their allocation get the scratch slot; a live row here is a host-side bug)")
                self.stats["rowexp_miss"] = self.stats.get("rowexp_miss", 0) + e_ - self.err_seen[r]
                self.err_seen[r] = e_
        if self.inflight is h:
            self.inflight = None
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
    def _first(self, batches, cts, R) -> StepHandle:
        """The bucket's first async step: eager (after draining), then graph L (key (R, "LA")) and graph H captured."""
        eng, H = self.eng, self.eng.hooks
        self.drain()
        st = eng.bucket(R)
        eng.bump_serial(batches)
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
        eng._capture_split(R, st, True, hk, lkey=(R, "LA"))
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
    def _fallback(self, batches, cts, ctx, why: str) -> StepHandle:
        """Drain, resolve every start on the device (rowexp: ring ops + expand, its internal outputs), D2H the starts,
        full tables from the resolved StepBatches (private TableBuilder: its block-table deltas rewrite the same pages),
        the synchronous engine.step, then k_advance with the step's accepted counts."""
        eng = self.eng
        t_fb0 = time.perf_counter()
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
        rb, tables = [], []
        t_tab = 0.0
        for r in range(W):
            sb, ct = batches[r], cts[r]
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
            sb2 = dataclasses.replace(sb, seqs=seqs, positions=pos, paged_slot=paged, window_slot=win)
            rb.append(sb2)
            t_ = time.perf_counter()
            tables.append(self.tb.build(sb2, [int(x) for x in np.asarray(ct.tokens[:sb.n_rows]).tolist()]))
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
        return StepHandle(self, batches[0].step if batches else 0, results=res)
