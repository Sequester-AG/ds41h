"""The indexer / compressor hook's per-step tables, derived on the device.

engine/hooks_idx.py built them on the host every step (numpy / torch CPU tensors, one pin_memory + H2D per tensor,
Python lists for the block-table deltas; mixed steps also a fresh OwnedIndexer and plan() with 8 device->host reads of
seq_lens.max()). Here, per call:
  host:   a handful of numpy copies of the worker's RankTables arrays (paged slots, per-sequence records, D12 block-table
          deltas) into ONE pinned int32 staging block (a ring of 4 per device, so a block is never rewritten while its
          copy is pending)
  device: one cudaMemcpyAsync of the block + 2 kernels (idx2f_tab.cu: delta scatter into the rank's per-slot table,
          then every table the hook's run() reads), on the current stream; row positions come from the engine's own
          device buffer s["pos"].
No host synchronization: host bounds (max position, pseudo-request counts) come from the host tables.

  DecodeTables(d, pos, dbt).upload(tab) -> True       decode buckets (hooks_idx.alloc's bucket dict d, in place)
  GeneralTables(...).upload(tab, global_rows) -> dict  mixed steps (the dict hooks_idx.upload_general built, same keys
                                                       and shapes; oi = a row view of a persistent OwnedIndexer)
Both return None when the tables carry no D12 deltas (tests without a worker): the hook then keeps its host path.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

import idx2f as X

RING = 4
CAP0 = 1 << 16                      # staging words per ring slot to start with (grows; see Stage._grow)
TAB_FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"]


@functools.cache
def ext_tab():
    return X._load("idx2f_tab", "idx2f_tab.cu", TAB_FLAGS, [str(X.HERE)])


def layout(n: int, S: int, U: int, P: int) -> tuple:
    """Word offsets of the staging block (idx2f_tab.cu layout(); keep in sync): ps, seqslot, sslot, row0, nrows, pr0,
    crow, uk, us, ul, uo, pages, total."""
    o, x = [], 0
    for size in (n, n, S, S, S, S + 1, S + 1, U, U, U, U, P):
        o.append(x)
        x += size
    o.append(x)
    return tuple(o)


class Stage:
    """Staging ring of one device (shared by every table set of the device): RING pinned int32 host blocks, RING device
    blocks, one event per slot recorded after the slot's kernels (so any stream may use it)."""

    _per_dev: dict = {}

    @classmethod
    def of(cls, dev) -> "Stage":
        dev = torch.device(dev)
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        st = cls._per_dev.get(dev.index)
        if st is None:
            st = cls._per_dev[dev.index] = cls(dev)
        return st

    def __init__(self, dev: torch.device, cap: int = CAP0):
        self.dev, self.k = dev, 0
        self.E = ext_tab()
        self.eid = self.E.ring(dev.index, RING)
        self._alloc(cap)

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        self.host = [torch.empty(cap, dtype=torch.int32, pin_memory=True) for _ in range(RING)]
        self.np = [h.numpy() for h in self.host]
        self.hptr = [h.data_ptr() for h in self.host]
        with torch.cuda.device(self.dev):
            self.dbuf = [torch.empty(cap, dtype=torch.int32, device=self.dev) for _ in range(RING)]
        self.dptr = [b.data_ptr() for b in self.dbuf]

    def _grow(self, total: int) -> None:
        # rare (capacity growth: first sight of many long block tables): every copy out of the old host blocks and
        # every kernel reading the old device blocks is done once the ring's events are (they are recorded after both)
        for i in range(RING):
            self.E.wait(self.eid, i)
        self._alloc(max(total, 2 * self.cap))

    def acquire(self, total: int):
        if total > self.cap:
            self._grow(total)
        i = self.k % RING
        self.k += 1
        self.E.wait(self.eid, i)          # returns at once unless that slot's copy (RING calls ago) is still pending
        return i, self.np[i]


def _deltas(sq: dict):
    """The tables' D12 block-table deltas as numpy (uk, us, ul int32 views, pages int64 view); None without them."""
    if "bt_upd_seq" not in sq:
        return None
    return sq["bt_upd_seq"].numpy(), sq["bt_upd_start"].numpy(), sq["bt_upd_len"].numpy(), sq["bt_upd_pages"].numpy()


def _fill(h: np.ndarray, o: tuple, ps, sslot, dl, seqslot=None, row0=None, nrows=None, pr0=None, crow=None) -> None:
    n, S = len(ps), len(sslot)
    h[o[0]: o[0] + n] = ps
    h[o[2]: o[2] + S] = sslot
    if seqslot is not None:
        h[o[1]: o[1] + n] = seqslot
        h[o[3]: o[3] + S] = row0
        h[o[4]: o[4] + S] = nrows
        h[o[5]: o[5] + S + 1] = pr0
        h[o[6]: o[6] + S + 1] = crow
    uk, us, ul, pages = dl
    U = len(uk)
    if U:
        h[o[7]: o[7] + U] = uk
        h[o[8]: o[8] + U] = us
        h[o[9]: o[9] + U] = ul
        uo = h[o[10]: o[10] + U]
        uo[0] = 0
        np.cumsum(ul[:-1], out=uo[1:])
        h[o[11]: o[11] + len(pages)] = pages


class DecodeTables:
    """Decode buckets (4 rows per sequence, sequence-contiguous; hooks_idx.alloc's dict d): writes d's seq_lens[1|2]
    [B, 4] int32, cslot[1|2] [R] int64, rslot [R] int64, valid [R] bool, preq_slot [B] int64 and applies the step's
    block-table deltas to dbt [n_state, P_MAX] int32. pos = the bucket's device positions (engine s["pos"], [R] int64)."""

    def __init__(self, d: dict, pos: torch.Tensor, dbt: torch.Tensor):
        self.stage = Stage.of(pos.device)
        self.R, self.B, self.n_state = pos.numel(), d["B"], dbt.shape[0]
        t = [pos, dbt, d["seq_lens"][1], d["seq_lens"][2], d["cslot"][1], d["cslot"][2], d["rslot"], d["valid"],
             d["preq_slot"]]
        self.hid = self.stage.E.create(t, 0)
        self._keep = t

    def upload(self, tab, max_ctx: int | None = None) -> bool | None:
        sq = tab.seq or {}
        dl = _deltas(sq)
        if dl is None:
            return None
        n = tab.n
        S = len(tab.seq_rows) if n else 0
        if n:
            nr = sq["n_rows"].numpy()
            assert n == 4 * S and bool((nr[:S] == 4).all()) and not bool(sq["kind"].numpy()[:S].any()) and \
                np.array_equal(tab.seq_slot.numpy(), np.repeat(np.arange(S, dtype=np.int32), 4)), \
                "idx2f tables: decode rows only (4 per sequence, sequence-contiguous)"
            if max_ctx is not None:
                mp = int(tab.positions.numpy().max()) + 1
                assert mp <= max_ctx, f"context {mp} > {max_ctx} (idx2f fast mode limit)"
        else:
            dl = tuple(a[:0] for a in dl)
        ps = tab.paged_slot.numpy() if n else np.zeros(0, np.int64)
        sslot = sq["state_slot"].numpy()[:S]
        U, P = len(dl[0]), len(dl[3])
        o = layout(n, S, U, P)
        st = self.stage
        i, h = st.acquire(o[-1])
        _fill(h, o, ps, sslot, dl)
        st.E.run(self.hid, st.eid, i, st.hptr[i], st.dptr[i], n, S, U, P, 0)
        return True


EAGER_LOGITS_BUDGET = 256 << 20   # fp32 dense logits per call in eager steps (lean OwnedIndexer)


class GeneralTables:
    """Mixed steps (any row layout; pseudo-requests of 4 consecutive rows of one sequence, padding rows past a
    sequence's rows): persistent device tables sized for the bucket (R rows, grows with the pseudo-request count), and
    per step the dict hooks_idx.upload_general built (same keys, shapes and values). Its OwnedIndexer is a row view of a
    persistent LEAN one (layer-20 candidates + one shared top-512 buffer, 10 KB per row) working in the
    device's fixed EagerWorkspace (dense-logit slices of <= logits_budget bytes, block maxima, consumer chunks, prep
    tables): no allocation that grows with the context."""

    def __init__(self, dev, R: int, pos: torch.Tensor, dbt: torch.Tensor, idx_views: dict, rpp: dict, mode: str,
                 dense_impl: str, logits_budget: int = EAGER_LOGITS_BUDGET):
        self.dev, self.R, self.pos, self.dbt = torch.device(dev), R, pos, dbt
        self.idx_views, self.rpp, self.mode, self.dense_impl = idx_views, rpp, mode, dense_impl
        self.logits_budget = logits_budget
        self.stage = Stage.of(self.dev)
        self.lat = torch.empty(R, 512, dtype=torch.bfloat16, device=self.dev)
        self.Bc = 0
        self._alloc(max(R // 4 + 64, 64))

    def _alloc(self, Bc: int) -> None:
        dev, R, i32, i64 = self.dev, self.R, torch.int32, torch.int64
        IRc = 4 * Bc
        with torch.cuda.device(dev):
            self.sl = {c: torch.zeros(IRc, dtype=i32, device=dev) for c in (1, 2)}
            self.cs = {c: torch.full((R,), -1, dtype=i64, device=dev) for c in (1, 2)}
            self.rs = torch.full((R,), -1, dtype=i64, device=dev)
            self.valid_ir = torch.zeros(IRc, dtype=torch.bool, device=dev)
            self.t2r = torch.zeros(R, dtype=i32, device=dev)
            self.qsl = torch.zeros(R + 2, dtype=i32, device=dev)
            self.ir2row_c = torch.zeros(IRc, dtype=i64, device=dev)
            self.vir = torch.zeros(R, dtype=i64, device=dev)
            self.vrow = torch.zeros(R, dtype=i64, device=dev)
            self.row2ir_d = torch.full((R,), -1, dtype=i64, device=dev)
            self.nb_lens = torch.zeros(IRc, dtype=i32, device=dev)
            self.bt = torch.zeros((Bc, self.dbt.shape[1]), dtype=i32, device=dev)
            self.t2r_ir = (torch.arange(IRc, device=dev) // 4).to(i32)
            ks = {L: X.CompactK(self.idx_views[L], self.bt, self.rpp[L]) for L in X.KV_SOURCES}
            self.base = X.OwnedIndexer(ks, Bc, dev, mode=self.mode, lean=True,
                                       ws=X.EagerWorkspace.of(dev, self.logits_budget))
            self.base.dense_impl = self.dense_impl
        t = [self.pos, self.dbt, self.sl[1], self.sl[2], self.cs[1], self.cs[2], self.rs, self.valid_ir, self.t2r,
             self.qsl, self.ir2row_c, self.vir, self.vrow, self.row2ir_d, self.nb_lens, self.bt]
        self.hid = self.stage.E.create(t, 1)
        self._keep = t
        self.Bc = Bc

    def upload(self, tab, global_rows: int, max_ctx: int | None = None) -> dict | None:
        sq = tab.seq or {}
        dl = _deltas(sq)
        if dl is None:
            return None
        n = tab.n
        S = len(tab.seq_rows) if n else 0
        if not n:
            return dict(general=True, idle=True, B=1, IR=4, global_rows=global_rows, max_seq_len=1, lat=self.lat,
                        row2ir=np.full(self.R, -1, np.int64))
        nr = sq["n_rows"].numpy()[:S].astype(np.int64)
        seqslot = tab.seq_slot.numpy()
        if int(nr.sum()) != n or not np.array_equal(seqslot, np.repeat(np.arange(S, dtype=seqslot.dtype), nr)):
            return None                                 # rows not sequence-contiguous in sequence order: host path
        crow = np.zeros(S + 1, np.int64)
        np.cumsum(nr, out=crow[1:])
        pr0 = np.zeros(S + 1, np.int64)
        np.cumsum((nr + 3) // 4, out=pr0[1:])
        B = int(pr0[S])
        IR = 4 * B
        maxp = int(tab.positions.numpy()[:n].max())
        if max_ctx is not None:
            assert maxp + 1 <= max_ctx, f"context {maxp + 1} > {max_ctx} (idx2f fast mode limit)"
        if B > self.Bc:
            self._alloc(-(-B // 64) * 64)
        U, P = len(dl[0]), len(dl[3])
        o = layout(n, S, U, P)
        st = self.stage
        i, h = st.acquire(o[-1])
        _fill(h, o, tab.paged_slot.numpy(), sq["state_slot"].numpy()[:S], dl, seqslot, crow[:S], nr, pr0, crow)
        st.E.run(self.hid, st.eid, i, st.hptr[i], st.dptr[i], n, S, U, P, B)
        # host-side row -> pseudo-request map (carry_candidates): row of sequence s at offset t -> 4 * pr0[s] + t
        row2ir = np.full(self.R, -1, np.int64)
        row2ir[:n] = np.repeat(4 * pr0[:S] - crow[:S], nr) + np.arange(n)
        bt = self.bt[:B]
        ks = {L: X.CompactK(self.idx_views[L], bt, self.rpp[L]) for L in X.KV_SOURCES}
        sl = {c: self.sl[c][:IR].view(B, 4) for c in (1, 2)}
        oi = self.base.rows_view(B, ks)
        # shared-key prefill logits (idx2f.OwnedIndexer.pf): groups of <= 4 pseudo-requests per sequence
        oi.pf_groups = X.pf_groups([(int(pr0[s]), int(pr0[s + 1])) for s in range(S)]) if getattr(oi, "pf", False) else None
        ins = {L: X.LayerIn(None, None, sl[X.RATIO[L]], maxp + 1, global_rows) for L in X.LAYERS}
        oi.plan_bounds(ins, maxp + 1, self.nb_lens[:IR])
        return dict(general=True, B=B, IR=IR, bt=bt, t2r_ir=self.t2r_ir[:IR], valid_ir=self.valid_ir[:IR],
                    ir2row_c=self.ir2row_c[:IR], vir=self.vir[:n], vrow=self.vrow[:n], seq_lens=sl,
                    cslot=self.cs, rslot=self.rs, qsl=self.qsl[: max(S, 1) + 1], t2r=self.t2r, lat=self.lat,
                    row2ir=row2ir, row2ir_dev=self.row2ir_d, global_rows=global_rows, max_seq_len=maxp + 1, oi=oi)


def carry_candidates_dev(dA: dict, dT: dict, ix: torch.Tensor) -> bool:
    """Layer 20's candidate blocks (bucket A, all rows) -> the tail bucket's pseudo-request rows, on the device (device
    tables: both dicts from GeneralTables). EVERY T row is written: a real row gets its own tail row's list, a padding row
    of a 4-row group (a sequence's tail rows % 4 != 0) its group's LAST real row's list -- padding rows carry the group's
    last length and join the group's candidate union (prep2 / candidate logits), so they must not hold stale or
    uninitialized block ids. ix: the tail rows as int64 on the device (T row t =
    A row ix[t]). One index_select into the T indexer's candidate rows; no host lists, no host sync. False: not device
    tables (the caller keeps its host path)."""
    if dA.get("idle") or dT.get("idle") or "row2ir_dev" not in dA or "ir2row_c" not in dT:
        return False
    B = dT["B"]
    g = dT["ir2row_c"].view(B, 4)                    # T rows of the T pseudo-request rows (padding rows: 0)
    v = dT["valid_ir"].view(B, 4)
    last = torch.where(v, g, torch.full_like(g, -1)).amax(1, keepdim=True)   # the group's last real T row
    fill = torch.where(v, g, last).reshape(-1)
    src = dA["row2ir_dev"].index_select(0, ix.index_select(0, fill))         # A pseudo-request rows
    torch.index_select(dA["oi"].cand, 0, src, out=dT["oi"].cand)
    return True
