"""Forced-decode dumps in the exact format of production's FORCE hook (ds41h/patch_force.py), so the comparison client
reads the fork's dumps unchanged:

    <dump_dir>/<tag>/rank{r}.pt    {"logits": [n, V or V/tp] (row j predicts forced[idx[j]]), "idx": int64 [n],
                                    "kind": "full" | "local", "vocab_start", "rank", "tp", "n", "n_dump", "prompt_len",
                                    "rows_per_step", "input_mismatch", "checked_rows", "overrides", "partial", "saved",
                                    "seconds", "shape", "dtype", "tag"}
    <dump_dir>/<tag>/rank{r}.json  the same metadata without the tensors, written last (completion marker)

kind "full": rank 0 saves the full-vocab rows, the other ranks write markers only (saved False), as production's
stock path does; kind "local": every rank saves its vocab shard [n, V/tp] at vocab_start = rank * V/tp.
"""
from __future__ import annotations

import json
import os
import time


def write_rank(dump_dir: str, tag: str, rank: int, tp: int, kind: str, vocab_start: int, logits, idx: list,
               n_dump: int, prompt_len: int, rows_per_step: list, input_mismatch: int = 0, checked_rows: int = 0,
               overrides: int = 0, partial: bool = False, t0: float | None = None) -> dict:
    import torch

    d = os.path.join(dump_dir, tag)
    os.makedirs(d, exist_ok=True)
    meta = {"tag": tag, "rank": rank, "tp": tp, "kind": kind, "vocab_start": vocab_start, "n": len(idx), "n_dump": n_dump,
            "prompt_len": prompt_len, "rows_per_step": list(rows_per_step), "input_mismatch": int(input_mismatch),
            "checked_rows": int(checked_rows), "overrides": int(overrides), "partial": bool(partial), "saved": False,
            "seconds": round(time.time() - (t0 or time.time()), 3)}
    save = logits is not None and (kind == "local" or rank == 0)
    if save:
        meta["shape"] = list(logits.shape)
        meta["dtype"] = str(logits.dtype)
        tmp = os.path.join(d, f".rank{rank}.pt.tmp")
        torch.save({**meta, "logits": logits, "idx": torch.tensor(idx, dtype=torch.int64)}, tmp)
        os.replace(tmp, os.path.join(d, f"rank{rank}.pt"))
        meta["saved"] = True
    tmp = os.path.join(d, f".rank{rank}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(d, f"rank{rank}.json"))
    return meta


def write_item(dump_dir: str, tag: str, tp: int, shards: dict, idx: list, n_dump: int, prompt_len: int,
               rows_per_step: list, t0: float | None = None, input_mismatch: int = 0) -> list:
    """shards: {vocab_start: tensor [n, w]} (one entry with vocab_start 0 and the full width = kind "full")."""
    metas = []
    full = len(shards) == 1 and 0 in shards
    if full:
        for r in range(tp):
            metas.append(write_rank(dump_dir, tag, r, tp, "full", 0, shards[0] if r == 0 else None, idx, n_dump,
                                    prompt_len, rows_per_step, input_mismatch, len(idx), 0, len(idx) < n_dump, t0))
        return metas
    for r, vs in enumerate(sorted(shards)):
        metas.append(write_rank(dump_dir, tag, r, tp, "local", vs, shards[vs], idx, n_dump, prompt_len, rows_per_step,
                                input_mismatch, len(idx), 0, len(idx) < n_dump, t0))
    return metas
