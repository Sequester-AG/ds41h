"""The FORCE hook of the reference vLLM deployment (ds41h/patch_force.py) for the served fork: teacher-forced decoding
through the real decode / verify path, with logit dumps in the reference format, so gate/client.py `collect` runs
unchanged against both.

Enabled by DS41H_FORCE_FILE=<path> (dumps: DS41H_FORCE_DUMP, default <dir of the force file>/dump; the force file's
"dump_dir" wins). Force file (JSON, written atomically by the client before each request):
    {"tag": str, "prompt_ids": [int, ...], "forced": [int, ...], "n_dump": int, "dump_dir": str (optional)}
It is re-read when its mtime / size changes. A new request whose prompt token ids equal "prompt_ids" becomes forced:
  * scheduler (PhysScheduler): Request.forced = True for the rank scheduler / worker (StepBatch SeqEntry.forced: accept
    every row, return the rows' raw logits); the k draft rows of each decode group carry the next forced tokens
    (row_tokens >= 0 override the device-side drafts); the step's emitted ids are the forced ones (ps.forced);
  * executor (PhysExecutor): the sampled ids of the forced request are replaced by the forced ones (k + 1 per verify
    step, 1 after the prefill), so vLLM's scheduler, the output and the next step all see fully accepted steps; the
    logits of every row that predicts forced[j], j < n_dump, are recorded (StepResult.logits: rid -> (positions,
    [rows, V or V/tp] fp32, vocab_start); row at position p predicts the token at p + 1);
  * when n_dump rows are recorded (or the request ends: partial) each rank that returned rows writes
        <dump>/<tag>/rank{r}.pt    {"logits", "idx", meta...}   (kind "local": every rank its vocab shard;
        <dump>/<tag>/rank{r}.json  meta, written last            kind "full": rank 0 only, tp = 1)
    with patch_force's meta keys (tag, rank, tp, kind, vocab_start, n, n_dump, prompt_len, rows_per_step,
    input_mismatch, checked_rows, overrides, partial, saved, seconds).
input_mismatch counts verify rows whose fed token (StepResult.fed[rid], if the worker reports it, else the row tokens the
host sent) differs from the forced token.
"""
from __future__ import annotations

import json
import os
import time

def force_file() -> str:
    return os.environ.get("DS41H_FORCE_FILE", "").strip()


def force_dump() -> str:
    ff = force_file()
    return os.environ.get("DS41H_FORCE_DUMP", "").strip() or (os.path.join(os.path.dirname(ff), "dump") if ff else "")


class ForceSpec:
    """Cached contents of the force file."""

    def __init__(self, path: str | None = None):
        self.path = force_file() if path is None else path
        self.key = None
        self.data = None

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def get(self):
        try:
            st = os.stat(self.path)
        except OSError:
            self.key, self.data = None, None
            return None
        key = (st.st_mtime_ns, st.st_size)
        if key != self.key:
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
                self.key = key
            except (OSError, ValueError):
                return None
        return self.data

    def match(self, prompt_ids) -> dict | None:
        d = self.get()
        if not d or list(d.get("prompt_ids", ())) != list(prompt_ids):
            return None
        return d


class ForcedReq:
    """Per forced request: the forced list and the dump being recorded."""

    def __init__(self, rid: str, spec: dict, prompt_len: int):
        self.rid = rid
        self.tag = str(spec["tag"])
        self.dump_dir = str(spec.get("dump_dir") or force_dump())
        self.forced = [int(x) for x in spec["forced"]]
        self.n_dump = min(int(spec["n_dump"]), len(self.forced))
        self.prompt_len = prompt_len
        self.parts: list = []           # (rank, idx list, tensor rows, vocab_start) in RECORD (= step) order
        self.idx: list = []             # forced indices recorded (rank-independent order)
        self.rows_per_step: list = []
        self.input_mismatch = 0
        self.checked_rows = 0
        self.overrides = 0
        self.flushed = False
        self.t0 = time.time()

    def spec(self) -> dict:
        """What a replay needs to rebuild this request's dump (adapter/replay.py)."""
        return dict(tag=self.tag, dump_dir=self.dump_dir, forced=list(self.forced), n_dump=self.n_dump,
                    prompt_len=self.prompt_len)

    @classmethod
    def from_spec(cls, rid: str, spec: dict, dump_dir: str | None = None) -> "ForcedReq":
        s = dict(spec)
        if dump_dir:
            s["dump_dir"] = dump_dir
        return cls(rid, s, int(s["prompt_len"]))

    def emit(self, n_out: int, n: int) -> list | None:
        """The forced ids a step that appends n tokens after n_out outputs must emit (None: the list ran out)."""
        if n_out + n > len(self.forced):
            return None
        return self.forced[n_out:n_out + n]

    # -- recording
    def record(self, results: list) -> None:
        got = False
        step_rows = 0
        for res in results:
            lg = (getattr(res, "logits", None) or {}).get(self.rid)
            if lg is None:
                continue
            positions, t, vstart = lg
            keep = [(i, p + 1 - self.prompt_len) for i, p in enumerate(positions)
                    if 0 <= p + 1 - self.prompt_len < self.n_dump]
            if not keep:
                continue
            import torch

            rows = t[torch.tensor([i for i, _ in keep], dtype=torch.long, device=t.device)].float().cpu()
            # one list in step order (grouping by rank wrote a request that left rank r and came back as
            # [r: early + late steps][other: middle steps]: rows no longer in idx order, misaligned with their tokens)
            self.parts.append((res.rank, [j for _, j in keep], rows, int(vstart)))
            if not got:
                self.idx.extend(j for _, j in keep)
                step_rows = len(positions)            # as production: the step's verify rows, incl. any past n_dump
                got = True
        if got:
            self.rows_per_step.append(step_rows)

    def done(self) -> bool:
        return len(self.idx) >= self.n_dump

    def flush(self, vocab: int, partial: bool = False) -> None:
        """Write the dump with serve's writer (serve/force_dump.write_item: patch_force's exact format, the same code
        OfflineEngine.force() uses): one shard at vocab_start 0 = kind "full" (rank 0's .pt + a marker per rank),
        several = kind "local" (every rank its shard)."""
        if self.flushed:
            return
        self.flushed = True
        import torch
        from force_dump import write_item  # serve/

        shards, sidx = {}, {}
        for _, js, rows, vs in self.parts:              # step order: kind "full" rows follow idx whatever rank owned
            shards.setdefault(vs, []).append(rows)      # them; kind "local": every shard in step order
            sidx.setdefault(vs, []).extend(js)
        shards = {vs: torch.cat(v, 0) for vs, v in shards.items()}
        self.reordered = 0
        for vs, js in sidx.items():                     # belt and braces: place every shard's rows by their forced index
            if js != list(self.idx):                    # (never raises: a dump must not fail a served step)
                pos = {j: k for k, j in enumerate(js)}
                sel = [pos[j] for j in self.idx if j in pos]
                shards[vs] = shards[vs][torch.tensor(sel, dtype=torch.long)]
                self.reordered += 1
        if not shards:
            shards = {0: torch.zeros(0, vocab)}
        write_item(self.dump_dir, self.tag, 4, shards, list(self.idx), self.n_dump, self.prompt_len,
                   list(self.rows_per_step), self.t0, self.input_mismatch)
        self.parts = []
