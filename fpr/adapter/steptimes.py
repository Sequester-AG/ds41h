"""Opt-in per-step host timing and a file-triggered profiler for the served path: measures the served path end to end,
host vs GPU (1-user decode ran at 0.54x production).

PHYS_STEP_TIMES=<file.jsonl>   one JSON line per engine step, written when vLLM's update_from_output of the step ends:
    times relative to the step's schedule() start, in ms (perf_counter, the EngineCore process):
      s1   schedule() end (PhysScheduler: rank scheduler + StepBatches + SchedulerOutput)
      x0 / x1   worker.step (sync: the whole engine step; ASYNC / D13: the dispatch, which also waits for the previous step)
      m0 / m1   merge_results (the 4 ranks' StepResults -> ModelRunnerOutput)
      u0 / u1   update_from_output (vLLM's request bookkeeping + the rank scheduler's update)
      gap  the previous step's u1 -> this step's schedule() start (EngineCore loop glue: output queue, input polling)
    rows / seqs / dec (decode sequences) / pf (prefill rows); worker (copies, tables, ctx, engine ms: serve's Worker);
    eng (Engine.stats["decode_last"]: the split decode step's own host split + GPU span from CUDA events, sync steps);
    d13 (StepHandle.t: prep, launch_l, wait_prev, pos_wait, head, host_total; ASYNC steps).
PHYS_PROFILE_DIR=<dir>         torch.profiler (CPU + CUDA) over N consecutive steps, started when <dir>/go exists (its
    content: N, default 40; the file is removed). Writes <dir>/prof_<step>.json (per device: GPU busy ms over the window,
    per step: wall and GPU busy per device; kernel categories via engine/profutil.py) and <dir>/trace_<step>.json.gz.
No token ids or text are recorded: row counts and times only.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import time

PATH = os.environ.get("PHYS_STEP_TIMES", "")
PROF_DIR = os.environ.get("PHYS_PROFILE_DIR", "")
pc = time.perf_counter


class _Times:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.f = open(path, "a", buffering=1 << 16)
        self.recs: dict = {}
        self.last_u1 = None
        self.n = 0
        self.t_flush = 0.0

    def start(self, step: int, t: float) -> dict:
        r = self.recs[step] = {"step": step, "t": round(t, 6), "_t0": t}
        if self.last_u1 is not None:
            r["gap"] = round((t - self.last_u1) * 1e3, 3)
        return r

    def get(self, step: int):
        return self.recs.get(step)

    def mark(self, step: int, key: str, t: float | None = None) -> None:
        r = self.recs.get(step)
        if r is not None:
            r[key] = round(((t if t is not None else pc()) - r["_t0"]) * 1e3, 3)

    def put(self, step: int, **kw) -> None:
        r = self.recs.get(step)
        if r is not None:
            r.update(kw)

    def finish(self, step: int, t: float) -> None:
        self.last_u1 = t
        r = self.recs.pop(step, None)
        if r is None:
            return
        r["u1"] = round((t - r.pop("_t0")) * 1e3, 3)
        self.f.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.n += 1
        if t - self.t_flush > 0.25:                  # a watcher may read the file live
            self.f.flush()
            self.t_flush = t
        if len(self.recs) > 64:                       # steps that never reached update_from_output (aborts / empty)
            for k in sorted(self.recs)[:-8]:
                self.recs.pop(k, None)

    def close(self) -> None:
        try:
            self.f.flush()
            self.f.close()
        except Exception:  # noqa: BLE001
            pass


T = _Times(PATH) if PATH else None


# ---------------------------------------------------------------------------------------------- profiler
class _Prof:
    """torch.profiler over N consecutive steps, armed by <dir>/go (checked every 8 steps)."""

    def __init__(self, d: str):
        os.makedirs(d, exist_ok=True)
        self.d = d
        self.prof = None
        self.left = 0
        self.k = 0
        self.first = None
        self.marks = []                              # (step, t_start_us, t_end_us) host wall of each profiled step

    def on_step_start(self, step: int) -> None:
        self.k += 1
        if self.prof is None:
            if self.k % 8:
                return
            go = os.path.join(self.d, "go")
            if not os.path.exists(go):
                return
            try:
                n = int((open(go).read().strip() or "40"))
            except Exception:  # noqa: BLE001
                n = 40
            os.remove(go)
            from torch.profiler import ProfilerActivity, profile
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.prof.start()
            self.left, self.first, self.marks = n, step, []
        self._t = time.time_ns() // 1000
        self._step = step

    def on_step_end(self) -> None:
        if self.prof is None:
            return
        self.marks.append((self._step, self._t, time.time_ns() // 1000))
        self.left -= 1
        if self.left > 0:
            return
        prof, self.prof = self.prof, None
        prof.stop()
        try:
            self._summarize(prof)
        except Exception as e:  # noqa: BLE001
            with open(os.path.join(self.d, f"prof_{self.first}.err"), "w") as f:
                f.write(repr(e))

    def _summarize(self, prof) -> None:
        import torch
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        # kineto times are in us on the profiler's clock; align the host marks via the first CPU event if needed
        dev = {}
        for e in evs:
            dev.setdefault(e.device_index, []).append((e.time_range.start, e.time_range.end))

        def busy(iv, a, b):
            tot, ce = 0.0, a
            for s, t in iv:
                s, t = max(s, ce), min(t, b)
                if t > s:
                    tot += t - s
                    ce = t
            return tot

        for v in dev.values():
            v.sort()
        cpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU]
        t_lo = min((e.time_range.start for e in cpu), default=0)
        t_hi = max((e.time_range.end for e in cpu), default=0)
        steps = []
        # the step ranges come from record_function("phys_step:<n>") (same clock as the kernels)
        rngs = sorted((e.time_range.start, e.time_range.end, e.name) for e in cpu if e.name.startswith("phys_step:"))
        for i, (a, b, nm) in enumerate(rngs):
            nxt = rngs[i + 1][0] if i + 1 < len(rngs) else b
            steps.append(dict(step=int(nm.split(":")[1]), wall_ms=round((nxt - a) / 1e3, 3), inside_ms=round((b - a) / 1e3, 3),
                              gpu_busy_ms={d: round(busy(iv, a, nxt) / 1e3, 3) for d, iv in sorted(dev.items())}))
        lo = rngs[0][0] if rngs else t_lo
        hi = rngs[-1][1] if rngs else t_hi
        out = dict(first_step=self.first, n_steps=len(steps), window_ms=round((hi - lo) / 1e3, 3),
                   gpu_busy_ms={d: round(busy(iv, lo, hi) / 1e3, 3) for d, iv in sorted(dev.items())}, steps=steps)
        try:
            from engine.profutil import categorize
            out["categories"] = categorize(prof.events())
        except Exception as e:  # noqa: BLE001
            out["categories_err"] = repr(e)
        # host-side ranges (record_function names in the adapter / engine), total ms over the window
        names = {}
        for e in cpu:
            if e.name.startswith(("phys:", "piece:")):
                names[e.name] = names.get(e.name, 0.0) + (e.time_range.end - e.time_range.start) / 1e3
        out["host_ranges_ms"] = {k: round(v, 3) for k, v in sorted(names.items(), key=lambda kv: -kv[1])}
        with open(os.path.join(self.d, f"prof_{self.first}.json"), "w") as f:
            json.dump(out, f, indent=1)
        tp = os.path.join(self.d, f"trace_{self.first}.json")
        prof.export_chrome_trace(tp)
        with open(tp, "rb") as a, gzip.open(tp + ".gz", "wb") as b:
            shutil.copyfileobj(a, b)
        os.remove(tp)


P = _Prof(PROF_DIR) if PROF_DIR else None
_RF = {}


def step_range(step: int):
    """record_function("phys_step:<n>") while the profiler runs (else a no-op context)."""
    if P is None or P.prof is None:
        return _NULL
    import torch
    return torch.profiler.record_function(f"phys_step:{step}")


def rf(name: str):
    if P is None or P.prof is None:
        return _NULL
    import torch
    return torch.profiler.record_function(name)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL = _Null()
