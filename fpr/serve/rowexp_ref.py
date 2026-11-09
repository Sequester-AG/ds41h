"""D13 reference: device-side row expansion (the format the GPU kernels in kernels/rowexp/ implement), numpy. The
executable spec.

Per step the host sends, per rank, a CompactTables:
  * one record per sequence in row order: slot, row0, n_rows, kind (0 decode / 1 prefill), start (-1 = the device's
    nc[slot]), replay_start, cached_upto;
  * window ops (slot, block, page) -- clears (slot, -1, -1) first;
  * D12 block-table deltas (into the engine's persistent per-slot table);
  * host-known per-row data: tokens (-1 where the device supplies them), tail rows, sample rows, targets.
DeviceState keeps what the GPU keeps (nc, the window ring, the block table); expand() resolves every row (decode AND
prefill) into positions, own window slot, FlashMLA window keys, paged write slot; advance() moves nc after the head.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geometry import PAGE_TOKENS, SLIDING_WINDOW, WINDOW_PAGE_TOKENS

WRING = 512                  # window ring entries per slot, index block % 512 (a live window spans <= ~134 blocks)
PAD = -1


@dataclass
class CompactTables:
    rank: int
    step: int
    k1: int
    # per sequence [S], row order
    slot: np.ndarray = None
    row0: np.ndarray = None
    n_rows: np.ndarray = None
    kind: np.ndarray = None                         # 0 decode / 1 prefill
    start: np.ndarray = None                        # int64, -1 = nc[slot]
    replay_start: np.ndarray = None
    cached_upto: np.ndarray = None
    # window ops [V] (clears: blk = page = -1, listed first)
    win_slot: np.ndarray = None
    win_blk: np.ndarray = None
    win_page: np.ndarray = None
    # D12 block-table deltas [U]
    bt_upd_slot: np.ndarray = None
    bt_upd_start: np.ndarray = None
    bt_upd_len: np.ndarray = None
    bt_upd_pages: np.ndarray = None
    # host-known per row [n]
    tokens: np.ndarray = None                       # -1 = from the device (decode rows: last token / drafts)
    tail_rows: np.ndarray = None
    sample_idx: np.ndarray = None
    targets: np.ndarray = None
    # host bookkeeping (not uploaded)
    rids: list = field(default_factory=list)
    uid: np.ndarray = None
    last_chunk: np.ndarray = None
    score: np.ndarray = None
    forced: np.ndarray = None

    @property
    def n(self) -> int:
        return int(self.n_rows.sum()) if self.n_rows is not None and len(self.n_rows) else 0


class DeviceState:
    """Per-rank persistent state of the expansion (kernels/rowexp keeps it on the GPU)."""

    def __init__(self, n_slots: int, k: int = 3):
        self.nc = np.zeros(n_slots, dtype=np.int64)
        self.ring = np.full((n_slots, WRING, 2), -1, dtype=np.int64)    # (block, page)
        self.dbt = {}                                                   # slot -> list of pages
        self.last_token = np.zeros(n_slots, dtype=np.int64)            # (the head's; tokens are not the kernel's)
        self.drafts = np.zeros((n_slots, k), dtype=np.int64)
        self.resolved = None                                            # seq_start of the last expand

    def apply(self, ct: CompactTables) -> None:
        """k_ring_ops (clears, then sets) + the D12 deltas."""
        for s, b, p in zip(ct.win_slot, ct.win_blk, ct.win_page):
            if b < 0:
                self.ring[s, :, :] = -1
            else:
                self.ring[s, b % WRING] = (b, p)
        off = 0
        for s, st, ln in zip(ct.bt_upd_slot, ct.bt_upd_start, ct.bt_upd_len):
            t = self.dbt.setdefault(int(s), [])
            del t[int(st):]
            t.extend(int(x) for x in ct.bt_upd_pages[off:off + ln])
            off += int(ln)

    def expand(self, ct: CompactTables, strict: bool = True) -> dict:
        """k_expand: every row of every sequence -> positions, own window slot, window keys, paged write slot. A window
        key without a page in the ring is written -1 and counted in self.errors (as kernels/rowexp does); strict
        asserts there are none (the contract: 0 errors)."""
        WT, BT = WINDOW_PAGE_TOKENS, PAGE_TOKENS
        n, S = ct.n, len(ct.slot)
        pos = np.empty(n, dtype=np.int64)
        own = np.empty(n, dtype=np.int64)
        paged = np.empty(n, dtype=np.int64)
        si = np.full((n, SLIDING_WINDOW), PAD, dtype=np.int32)
        sl = np.empty(n, dtype=np.int32)
        seq_of = np.empty(n, dtype=np.int32)
        seq_start = np.empty(S, dtype=np.int64)
        self.errors = 0
        for k in range(S):
            s, r0, nr = int(ct.slot[k]), int(ct.row0[k]), int(ct.n_rows[k])
            base = int(ct.start[k]) if int(ct.start[k]) >= 0 else int(self.nc[s])
            seq_start[k] = base
            bt = self.dbt.get(s, [])
            rs, cu = int(ct.replay_start[k]), int(ct.cached_upto[k])

            def wslot(w):
                b = w // WT
                e = self.ring[s, b % WRING]
                if int(e[0]) != b:
                    self.errors += 1
                    return PAD
                return int(e[1]) * WT + w % WT

            for i in range(nr):
                p, r = base + i, r0 + i
                pos[r] = p
                own[r] = wslot(p)
                paged[r] = PAD if p < cu else bt[p // BT] * BT + p % BT
                lo = max(rs, p - (SLIDING_WINDOW - 1))
                sl[r] = p - lo + 1
                si[r, : sl[r]] = [wslot(w) for w in range(lo, p + 1)]
                seq_of[r] = k
        assert not strict or self.errors == 0, f"{self.errors} window keys without a page in the ring"
        self.resolved = seq_start
        return dict(positions=pos, own_slot=own, si=si, sl=sl, paged_slot=paged, seq_slot=seq_of, seq_start=seq_start,
                    tokens=ct.tokens, tail_rows=ct.tail_rows, sample_idx=ct.sample_idx, targets=ct.targets)

    def advance(self, ct: CompactTables, accepted: np.ndarray) -> None:
        """k_advance: nc[slot] = resolved start + (decode ? accepted : n_rows)."""
        for k, s in enumerate(ct.slot):
            self.nc[s] = int(self.resolved[k]) + (int(accepted[k]) if int(ct.kind[k]) == 0 else int(ct.n_rows[k]))
