"""FORCE dumps follow the forced-token order when a forced request changes ranks mid-decode (adapter/force.py).

Guards: ForcedReq collected each step's logit rows per rank and concatenated them rank by rank at flush, while the
dump's `idx` is in step order. A request rebalanced between ranks during decode (A -> B -> A) was written as
[A: early + late steps][B: middle steps], so from the first move on every row sat next to the wrong forced token
(kind "full", one rank returns the whole vocabulary). Kind "local" (every rank its vocabulary shard, every step) must
stay aligned too, whatever order the ranks' results arrive in.

CPU only: synthetic StepResults whose rows are one-hot at the forced token; serve/force_dump.write_item is captured
instead of writing files.
  python -m pytest -q adapter/tests/test_force_order.py
"""
import os
import random
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "serve"))

import force_dump  # noqa: E402

from adapter.force import ForcedReq  # noqa: E402

V, PL, W = 64, 100, 4          # vocabulary, prompt length, ranks (local shards of V / W)


def _forced(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(V) for _ in range(n)]


def _results(rid, js, forced, owner, local, rng):
    """One step's StepResults: rows at positions PL - 1 + j predict forced[j]."""
    pos = [PL - 1 + j for j in js]
    if not local:                              # kind "full": the owner rank only, the whole vocabulary
        t = torch.zeros(len(js), V)
        for i, j in enumerate(js):
            t[i, forced[j]] = 1.0
        return [NS(rank=owner, logits={rid: (pos, t, 0)})]
    w = V // W
    out = []
    for r in rng.sample(range(W), W):          # every rank its shard, in any order
        t = torch.zeros(len(js), w)
        for i, j in enumerate(js):
            if r * w <= forced[j] < (r + 1) * w:
                t[i, forced[j] - r * w] = 1.0
        out.append(NS(rank=r, logits={rid: (pos, t, r * w)}))
    return out


def _collect(monkeypatch, tmp_path, owners, rows, local, n_dump=None, seed=0, partial=False):
    """Run a forced request whose step s is owned by owners[s] and returns rows[s] verify rows; return
    (idx, full rows, forced)."""
    rng = random.Random(seed)
    total = sum(rows)
    n_dump = total if n_dump is None else n_dump
    forced = _forced(max(total, n_dump), seed)
    fr = ForcedReq("r0", dict(tag="t", forced=forced, n_dump=n_dump, dump_dir=str(tmp_path)), PL)
    got = {}
    monkeypatch.setattr(force_dump, "write_item",
                        lambda d, tag, tp, shards, idx, *a, **k: got.update(shards=shards, idx=idx))
    j0 = 0
    for owner, n in zip(owners, rows):
        fr.record(_results("r0", list(range(j0, j0 + n)), forced, owner, local, rng))
        j0 += n
    assert fr.done() or partial
    fr.flush(V, partial=partial)
    idx = got["idx"]
    full = torch.cat([got["shards"][vs] for vs in sorted(got["shards"])], 1)
    return idx, full, forced


def _misplaced(idx, full, forced):
    assert full.shape[0] == len(idx)
    return [i for i in range(len(idx)) if full[i].max() < 1 or int(full[i].argmax()) != forced[idx[i]]]


@pytest.mark.parametrize("local", [False, True], ids=["full", "local"])
@pytest.mark.parametrize("owners", [[3, 3, 3, 1, 1, 3, 3, 0, 3, 3], [3, 1, 3, 2, 3], [0, 1, 2, 3, 2, 1, 0], [2] * 6])
def test_rank_moves_fixed(monkeypatch, tmp_path, owners, local):
    idx, full, forced = _collect(monkeypatch, tmp_path, owners, [4] * len(owners), local)
    assert idx == list(range(len(idx)))
    bad = _misplaced(idx, full, forced)
    assert not bad, f"{len(bad)} of {len(idx)} rows misplaced, first {bad[:5]}"


@pytest.mark.parametrize("seed", range(40))
def test_rank_moves_random(monkeypatch, tmp_path, seed):
    """Synthetic permutations: random owner sequences (moves at random steps, returns to earlier ranks), 1-4 verify
    rows per step, n_dump cutting the last step, both dump kinds."""
    rng = random.Random(1000 + seed)
    steps = rng.randrange(2, 24)
    owners, cur = [], rng.randrange(W)
    for _ in range(steps):
        if rng.random() < 0.35:
            cur = rng.randrange(W)
        owners.append(cur)
    rows = [rng.randrange(1, 5) for _ in range(steps)]
    n_dump = sum(rows) - rng.randrange(0, rows[-1])
    for local in (False, True):
        idx, full, forced = _collect(monkeypatch, tmp_path, owners, rows, local, n_dump=n_dump, seed=seed)
        assert idx == list(range(n_dump)), (owners, rows, n_dump)
        bad = _misplaced(idx, full, forced)
        assert not bad, (local, owners, rows, bad[:5])


@pytest.mark.parametrize("local", [False, True], ids=["full", "local"])
def test_partial_flush_after_moves(monkeypatch, tmp_path, local):
    """A request that ends before n_dump rows (partial dump) after rank moves: the rows it has are aligned."""
    owners = [1, 1, 2, 2, 1, 0]
    rows = [4] * len(owners)
    idx, full, forced = _collect(monkeypatch, tmp_path, owners, rows, local, n_dump=sum(rows) + 10, partial=True)
    assert idx == list(range(sum(rows)))
    assert not _misplaced(idx, full, forced)
