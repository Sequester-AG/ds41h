#!/usr/bin/env python3
"""D13 device-side row expansion: build, device state, host descriptor.

Per rank (one RowExp per GPU / rank):
    rx = RowExp(device, n_state_slots, max_seqs, max_rows, scratch_slot=pools.scratch_window_page * 32)
    host = HostDesc(rank)                       # host-side bookkeeping (who last advanced a slot, window pages sent)
  per step N+1 (the host may run this while step N is still on the GPU: nothing here depends on N's acceptance):
    d = host.build(sb)                          # StepBatch -> descriptor (decode starts -1 = device), window ops
    rx.upload(d)                                # H2D on the engine's stream (stream-ordered after step N's graph)
  inside step N+1's graph:
    rx.begin(R)                                 # k_ring_ops + k_expand -> rx.out (pos / own / si / sl / paged / start)
    ... model, head ...
    rx.end(accepted)                            # k_advance: nc[slot] = start + accepted (decode) / + n_rows (prefill)
Outputs land in rx.out (persistent tensors, the same layout as serve.worker.RankTables after engine._upload padding), or
in the engine's own bucket buffers via rx.bind_outputs(...).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SW, WT, BT, RB = 128, 32, 64, 512


@cache
def ext():
    from torch.utils.cpp_extension import load

    bd = os.path.join(HERE, "build_ext")
    os.makedirs(bd, exist_ok=True)
    return load(name="rowexp_ext", sources=[os.path.join(HERE, "rowexp.cu")], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"])


@dataclass
class Desc:
    """One rank's per-step descriptor (host, numpy). Sequences in row order."""
    S: int
    rows: int
    slot: np.ndarray       # [S] int32
    row0: np.ndarray       # [S] int32
    nrows: np.ndarray      # [S] int32
    kind: np.ndarray       # [S] int32 (0 decode, 1 prefill)
    start: np.ndarray      # [S] int64 (-1 = device nc[slot])
    replay: np.ndarray     # [S] int64
    cached: np.ndarray     # [S] int64
    ops: np.ndarray        # [n_ops, 3] int32 (slot, block, page); block -1 = clear the slot's ring


class HostDesc:
    """Host side of D13 for one rank. O(sequences + new window pages) per step; nothing depends on the previous step's
    acceptance. `device_start=False` sends every start from the host (the synchronous loop; same kernel)."""

    def __init__(self, rank: int, device_start: bool = True):
        self.rank = rank
        self.device_start = device_start
        self.owner = {}       # state slot -> uid whose step last advanced nc[slot] on the device
        self.sent = {}        # state slot -> (uid, {block: page}) window pages in the device ring

    def build(self, sb) -> Desc:
        S = len(sb.seqs)
        slot = np.empty(S, np.int32)
        row0 = np.empty(S, np.int32)
        nrows = np.empty(S, np.int32)
        kind = np.empty(S, np.int32)
        start = np.empty(S, np.int64)
        replay = np.empty(S, np.int64)
        cached = np.empty(S, np.int64)
        ops = []
        for k, se in enumerate(sb.seqs):
            # identity = (uid, inc): uids recycle FIFO and slots lowest-first, so a re-admitted or moved-back sequence can
            # land on the same uid AND slot with new pages (serve SeqKV.inc is unique per admit / move)
            sl_, uid = se.state_slot, (se.uid, getattr(se, "inc", 0))
            slot[k], row0[k], nrows[k] = sl_, se.row0, se.n_rows
            kind[k] = 0 if se.kind == "decode" else 1
            dev_ok = self.device_start and se.kind == "decode" and self.owner.get(sl_) == uid
            # async StepBatches carry decode start -1 and the host's settled count in se.known: the first step of a
            # sequence new to its slot (new uid / move / re-admission) must be explicit, = known
            host_start = se.start if se.start >= 0 else getattr(se, "known", -1)
            assert dev_ok or host_start >= 0, f"{se.rid}: no host start for a sequence new to slot {sl_}"
            start[k] = -1 if dev_ok else host_start
            replay[k], cached[k] = se.replay_start, se.cached_upto
            self.owner[sl_] = uid
            prev = self.sent.get(sl_)
            wt = se.window_table
            nb = se.n_blocks or len(se.block_table)
            wver = getattr(se, "wver", -1)
            # ring clear on the D12 bt_reset trigger: a new (uid, inc) in the slot, or its block table shrank
            if prev is None or prev[0] != uid or prev[2] > nb:
                ops.append((sl_, -1, -1))
                have, last_wver = {}, None
            else:
                have, last_wver = prev[1], prev[3]
            self.sent[sl_] = (uid, have, nb, wver)
            if wver < 0 or wver != last_wver:     # serve SeqKV.wver: the window table changed since the last diff
                for b, pg in wt.items():
                    if have.get(b) != pg:
                        ops.append((sl_, b, pg))
                        have[b] = pg
            if len(have) > 2 * len(wt) + 16:          # forget released blocks (host memory only)
                for b in [b for b in have if b not in wt]:
                    del have[b]
        return Desc(S, sb.n_rows, slot, row0, nrows, kind, start, replay, cached,
                    np.asarray(ops, np.int32).reshape(-1, 3))


class RowExp:
    def __init__(self, device, n_state_slots: int, max_seqs: int, max_rows: int, max_ops: int = 8192,
                 scratch_slot: int = 0, dbt: torch.Tensor | None = None, p_max: int = 655_360 // 64):
        self.dev = torch.device(device)
        dv = self.dev
        i32, i64 = dict(dtype=torch.int32, device=dv), dict(dtype=torch.int64, device=dv)
        self.S_max, self.R_max, self.ops_max = max_seqs, max_rows, max_ops
        self.scratch = int(scratch_slot)
        # device state
        self.nc = torch.zeros(n_state_slots, **i32)
        self.ring = torch.full((n_state_slots, RB, 2), -1, **i32)
        self.dbt = dbt if dbt is not None else torch.zeros(n_state_slots, p_max, **i32)   # the engine's D12 table
        # descriptor: two packed device blocks (int32: slot | row0 | nrows | kind | counts | ops; int64: start | replay |
        # cached) with views per field; the host fills numpy views of pinned staging and uploads 2 prefix copies
        S_, O_ = max_seqs, max_ops
        self._o32 = dict(slot=0, row0=S_, nrows=2 * S_, kind=3 * S_, counts=4 * S_, ops=4 * S_ + 4)
        n32, n64 = 4 * S_ + 4 + 3 * O_, 3 * S_
        self._b32 = torch.zeros(n32, **i32)
        self._b64 = torch.zeros(n64, **i64)
        o = self._o32
        self.d = dict(slot=self._b32[o["slot"]: o["slot"] + S_], row0=self._b32[o["row0"]: o["row0"] + S_],
                      nrows=self._b32[o["nrows"]: o["nrows"] + S_], kind=self._b32[o["kind"]: o["kind"] + S_],
                      counts=self._b32[o["counts"]: o["counts"] + 3], ops=self._b32[o["ops"]:].view(O_, 3),
                      start=self._b64[:S_], replay=self._b64[S_: 2 * S_], cached=self._b64[2 * S_:])
        # two pinned staging sets, alternating: the host may build step N+2's descriptor while N+1's copies still wait
        # behind step N's graph on the stream; a set is reused only after the upload that last used it has run
        self._h = [(torch.zeros(n32, dtype=torch.int32, pin_memory=True), torch.zeros(n64, dtype=torch.int64, pin_memory=True))
                   for _ in range(2)]
        self._hn = [(a.numpy(), b.numpy()) for a, b in self._h]
        self._ev = [None, None]
        self._hi = 0
        self.accepted = torch.zeros(max_seqs, **i32)
        # outputs
        self.out = dict(pos=torch.zeros(max_rows, **i64), own=torch.zeros(max_rows, **i64),
                        si=torch.full((max_rows, SW), -1, **i32), sl=torch.ones(max_rows, **i32),
                        paged=torch.full((max_rows, ), -1, **i64), start=torch.zeros(max_seqs, **i64),
                        seq=torch.full((max_rows, ), -1, **i32))
        self.err = torch.zeros(2, **i32)

    def bind_outputs(self, **bufs):
        """Write straight into the engine's buffers: pos / own (int64 [R]), si (int32 [R, 128]), sl (int32 [R]), paged
        (int64 [R]), start (int64 [S_max]); contiguous, on this device."""
        for k, t in bufs.items():
            assert k in self.out and t.dtype == self.out[k].dtype and t.is_contiguous() and t.device == self.dev, k
            self.out[k] = t

    def upload(self, d: Desc, stream: torch.cuda.Stream | None = None):
        """Descriptor -> pinned staging (numpy views) -> device: 2 non-blocking prefix copies on `stream` (default: the
        current one). The engine issues it on its step stream after step N's graph, so it is ordered after N's k_advance
        and before N+1's k_expand."""
        S, n_ops = d.S, len(d.ops)
        assert S <= self.S_max and d.rows <= self.R_max and n_ops <= self.ops_max, (S, d.rows, n_ops)
        i = self._hi
        self._hi ^= 1
        if self._ev[i] is not None:
            self._ev[i].synchronize()   # the copies of two uploads ago (normally long done)
        h32, h64 = self._hn[i]
        o, S_ = self._o32, self.S_max
        h32[o["slot"]: o["slot"] + S] = d.slot
        h32[o["row0"]: o["row0"] + S] = d.row0
        h32[o["nrows"]: o["nrows"] + S] = d.nrows
        h32[o["kind"]: o["kind"] + S] = d.kind
        h32[o["counts"]: o["counts"] + 3] = (S, d.rows, n_ops)
        if n_ops:
            h32[o["ops"]: o["ops"] + 3 * n_ops] = d.ops.reshape(-1)
        h64[:S] = d.start
        h64[S_: S_ + S] = d.replay
        h64[2 * S_: 2 * S_ + S] = d.cached
        n32 = o["ops"] + 3 * n_ops
        st = stream or torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(st):
            self._b32[:n32].copy_(self._h[i][0][:n32], non_blocking=True)
            self._b64.copy_(self._h[i][1], non_blocking=True)
        self._ev[i] = torch.cuda.Event()
        self._ev[i].record(st)

    def begin(self, R: int):
        """k_ring_ops + k_expand for R bucket rows (graph-capturable: all counts in device memory)."""
        E, d, o = ext(), self.d, self.out
        with torch.cuda.device(self.dev):
            E.ring_ops(self.ring, d["ops"], d["counts"])
            E.expand(d["slot"], d["row0"], d["nrows"], d["kind"], d["start"], d["replay"], d["cached"], d["counts"], self.ring,
                     self.dbt, self.nc, R, self.scratch, o["pos"], o["own"], o["si"], o["sl"], o["paged"], o["start"], o["seq"],
                     self.err)

    def end(self, accepted: torch.Tensor | None = None):
        """k_advance after the head: accepted int32 [S] in this step's sequence order (decode entries 1 .. k + 1; prefill
        entries ignored). Default: self.accepted."""
        a = self.accepted if accepted is None else accepted
        with torch.cuda.device(self.dev):
            ext().advance(self.d["slot"], self.d["kind"], self.d["nrows"], self.out["start"], a, self.d["counts"], self.nc,
                          self.S_max)

    def apply_compact_bt(self, ct) -> None:
        """serve rowexp_ref.CompactTables D12 deltas (bt_upd_slot / start / len / pages) into self.dbt (tests; the engine
        applies D12 itself)."""
        if ct.bt_upd_slot is None or not len(ct.bt_upd_slot):
            return
        rows, cols = [], []
        for s_, a_, n_ in zip(ct.bt_upd_slot.tolist(), ct.bt_upd_start.tolist(), ct.bt_upd_len.tolist()):
            rows += [int(s_)] * int(n_)
            cols += list(range(int(a_), int(a_) + int(n_)))
        if rows:
            self.dbt.index_put_((torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev)),
                                torch.as_tensor(np.asarray(ct.bt_upd_pages[: len(rows)]), dtype=torch.int32).to(self.dev))

    @staticmethod
    def desc_of_compact(ct) -> "Desc":
        """serve rowexp_ref.CompactTables (the D13 format) -> Desc."""
        S = len(ct.slot) if ct.slot is not None else 0
        ops = np.stack([np.asarray(ct.win_slot), np.asarray(ct.win_blk), np.asarray(ct.win_page)], 1).astype(np.int32) \
            if ct.win_slot is not None and len(ct.win_slot) else np.zeros((0, 3), np.int32)
        a = lambda x, dt: np.asarray(x, dt)  # noqa: E731
        return Desc(S, ct.n, a(ct.slot, np.int32), a(ct.row0, np.int32), a(ct.n_rows, np.int32), a(ct.kind, np.int32),
                    a(ct.start, np.int64), a(ct.replay_start, np.int64), a(ct.cached_upto, np.int64), ops)

    def apply_bt_deltas(self, tab) -> None:
        """serve D12 block-table deltas of a RankTables (seq bt_upd_*) into self.dbt (= engine/hooks_idx.apply_deltas)."""
        sq = tab.seq
        ks, s0, ln = sq["bt_upd_seq"].tolist(), sq["bt_upd_start"].tolist(), sq["bt_upd_len"].tolist()
        if not ks or not sum(ln):
            return
        slots = sq["state_slot"]
        rows, cols = [], []
        for k, a_, n_ in zip(ks, s0, ln):
            rows += [int(slots[k])] * n_
            cols += list(range(a_, a_ + n_))
        self.dbt.index_put_((torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev)),
                            sq["bt_upd_pages"].to(torch.int32).to(self.dev))


class RowExpCPU:
    """numpy emulation of the three kernels (same buffers / semantics; tests and a readable reference)."""

    def __init__(self, n_state_slots: int, max_seqs: int, max_rows: int, scratch_slot: int = 0, p_max: int = 655_360 // 64):
        self.nc = np.zeros(n_state_slots, np.int64)
        self.ring = np.full((n_state_slots, RB, 2), -1, np.int64)
        self.dbt = np.zeros((n_state_slots, p_max), np.int64)
        self.scratch = scratch_slot
        self.err = np.zeros(2, np.int64)
        self.d = None
        self.accepted = np.zeros(max_seqs, np.int64)

    def upload(self, d: Desc):
        self.d = d

    def apply_bt_deltas(self, tab):
        sq = tab.seq
        pages = sq["bt_upd_pages"].numpy()
        o = 0
        for k, a_, n_ in zip(sq["bt_upd_seq"].tolist(), sq["bt_upd_start"].tolist(), sq["bt_upd_len"].tolist()):
            self.dbt[int(sq["state_slot"][k]), a_ : a_ + n_] = pages[o : o + n_]
            o += n_

    def _page(self, slot, b, e):
        ent = self.ring[slot, b % RB]
        if ent[0] != b:
            self.err[e] += 1
            return -1
        return ent[1]

    def begin(self, R: int):
        d = self.d
        for sl_, b, pg in d.ops:
            if b < 0:
                self.ring[sl_] = -1
        for sl_, b, pg in d.ops:
            if b >= 0:
                self.ring[sl_, b % RB] = (b, pg)
        out = dict(pos=np.zeros(R, np.int64), own=np.full(R, self.scratch, np.int64), si=np.full((R, SW), -1, np.int64),
                   sl=np.ones(R, np.int64), paged=np.full(R, -1, np.int64), start=np.zeros(max(d.S, 1), np.int64))
        out["si"][:, 0] = self.scratch
        for s in range(d.S):
            sl_ = d.slot[s]
            start = d.start[s] if d.start[s] >= 0 else self.nc[sl_]
            out["start"][s] = start
            for i in range(d.nrows[s]):
                t, p = d.row0[s] + i, start + i
                lo = max(d.replay[s], p - (SW - 1))
                row = np.full(SW, -1, np.int64)
                for j, idx in enumerate(range(lo, p + 1)):
                    pg = self._page(sl_, idx // WT, 0)
                    row[j] = -1 if pg < 0 else pg * WT + idx % WT
                out["si"][t] = row
                out["pos"][t] = p
                pg = self._page(sl_, p // WT, 1)
                out["own"][t] = -1 if pg < 0 else pg * WT + p % WT
                out["sl"][t] = p - lo + 1
                out["paged"][t] = -1 if p < d.cached[s] else self.dbt[sl_, p // BT] * BT + p % BT
        self.out = out

    def end(self, accepted=None):
        a = self.accepted if accepted is None else accepted
        d = self.d
        for s in range(d.S):
            self.nc[d.slot[s]] = self.out["start"][s] + (a[s] if d.kind[s] == 0 else d.nrows[s])
