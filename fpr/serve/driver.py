"""Offline driver: RankScheduler + worker on a list of requests, no API server.

  run_offline(requests, sched, worker)   schedule -> StepBatch per rank -> worker.run_step -> sched.update, until done
  seed_decoding(sched, ctxs)             sequences already prefilled (KV allocated + committed, phase DECODE): for the
                                         decode-step smoke test, which has no prefill path yet

CPU (stub backend, no GPU):
  python3 driver.py --backend stub --requests 48 --steps 400
GPU smoke (allocation + one decode step on synthetic sequences, the sp_layer chain; 4 GPUs, after the prebuild):
  python3 driver.py --backend gpu --smoke --seqs 114
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from cost_model import CostModel
from geometry import Geometry
from interfaces import build_step_batch
from scheduler import DECODE, Request, RankScheduler, SchedConfig


def seed_decoding(sched: RankScheduler, ctxs: list, out_tokens: int = 1_000_000, tag: str = "s") -> list:
    """Create decoding sequences of the given contexts directly (no prefill steps), placed by the scheduler's own
    placement policy: admitted, paged KV allocated + committed, window pages for the last 128 positions only, phase
    DECODE with one output token -- as if a prefill had just finished."""
    from scheduler import PREFILL

    out = []
    for i, ctx in enumerate(ctxs):
        rid = f"{tag}{i}"
        q = Request(rid, ctx, out_tokens, 0.0, lambda b, rid=rid: (rid, b))
        sched.reqs[rid] = q
        q.prefill_end = ctx
        hashes = q.block_hashes(ctx)
        hits = sched.kv.lookup(hashes)
        res = None
        for r, src in sched._placement_order(q, hits):
            res = sched.kv.admit(rid, r, hashes, ctx, sched.step_id, copy_from=None)
            if res is not None:
                break
        assert res is not None, f"no rank can hold {rid} ({ctx} tokens)"
        q.rank, q.phase, q.admit_time = res.rank, PREFILL, 0.0
        sched.take_uid(q)
        sched.running.append(q)
        assert sched.kv.allocate(rid, ctx - 1, ctx, ctx + sched.g.draft_window_tokens), rid   # + drafter (window)
        sched.kv.commit(rid, ctx, q.hash_fn)
        sched.kv.shrink_reservation(rid)
        q.num_computed, q.phase, q.num_output = ctx, DECODE, 1
        out.append(q)
    return out


def chain_hashes(ids: list, upto: int, salt, cache_salt=None) -> list:
    """vLLM-style chained block hashes of the full 64-token blocks of ids[:upto] (blocks past `upto` get unique salted
    hashes so they never hit: a score request must recompute the rows it needs logprobs for). cache_salt (vLLM's
    cache_salt): mixed into the first block's parent, so only items with the same salt share cached blocks."""
    import hashlib

    out = []
    parent = b"" if cache_salt is None else hashlib.sha256(("cache_salt:" + repr(cache_salt)).encode()).digest()
    for b in range(len(ids) // 64):
        if (b + 1) * 64 <= upto:
            h = hashlib.sha256(parent + repr(tuple(ids[b * 64:(b + 1) * 64])).encode()).digest()
            out.append(h)
            parent = h
        else:
            out.append(("nocache", salt, b))
    return out


# the engine's pieces are chosen by $PHYS_ENGINE (engine/engine.py _env_kw; defaults stub / off / stub)
FULL_PIECES = "indexer=idx2f,engram=phys,head=hsd"
STUB_PIECES = "indexer=stub,engram=off,head=stub"


def set_pieces(spec: str | None = None) -> str:
    """$PHYS_ENGINE for the engine backend: "full" / "stub" / a raw "k=v,..." list; None or "" keeps $PHYS_ENGINE when
    set, else FULL_PIECES (the real model: exactness checks and end-to-end runs need the real indexer, Engram and
    head)."""
    spec = spec or os.environ.get("PHYS_ENGINE") or "full"
    spec = {"full": FULL_PIECES, "stub": STUB_PIECES}.get(spec, spec)
    os.environ["PHYS_ENGINE"] = spec
    # idx2f's extensions as engine/prebuild_ext.py --idx builds them; without it idx2f falls back to
    # kernels/indexer2f/build, which lacks idx2f_sel / idx2f_topk -> a compile on the GPU run's clock
    phys = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("IDX2F_BUILD", os.path.join(phys, "engine", "build", "idx2f"))
    return spec


class OfflineEngine:
    """Prompts in, tokens + logprobs out (the entry point of the exactness checks). No API server.

      eng = OfflineEngine(backend="stub" | "gpu")
      eng.generate([dict(id=..., prompt_ids=[...], max_tokens=64, salt=None)])      (salt = vLLM cache_salt, every API)
          -> [dict(id, tokens=[...], logprobs=[[(id, lp) x top-k] per generated token] or None)]
      eng.score([dict(id=..., ids=prompt + completion, start=len(prompt))], topk=20)
          -> [dict(id, positions=[start .. len - 1], topk=[[(id, lp) x top-k] per position])]
          topk[i] is the next-token distribution AT position positions[i] - 1, i.e. the one that scores ids[positions[i]]
          (teacher forcing, = vLLM prompt_logprobs). Prefix-cache hits are allowed only below `start`.
    Sampling is the engine's (greedy / seeded per request once kernels/head is wired); the stub returns fake numbers."""

    def __init__(self, backend: str = "stub", cfg=None, geom: Geometry | None = None, sched_cfg: SchedConfig | None = None,
                 max_rows: int = 320, log=print):
        from worker import Worker, WorkerConfig

        geom = geom or (Geometry(kv_bytes_per_rank=6_000_000_000) if backend == "stub" else Geometry())   # gpu/engine: resized
        if backend == "engine":
            self.pieces = set_pieces()                   # full pieces unless $PHYS_ENGINE says otherwise
        self.worker = Worker(cfg or WorkerConfig(), geom, backend, max_rows=max_rows, log=log)
        # the real engine's measured step costs (its eager mixed steps carry a ~41 ms fixed cost) unless $PHYS_COST
        # says otherwise; the stub / mock backends keep the cost model's defaults
        from cost_model import ENGINE_MEASURED_2026_11_07, params_from_env

        real = backend == "engine" and "mock" not in self.worker.cfg.engine_cls.lower()
        cp = params_from_env(ENGINE_MEASURED_2026_11_07 if real else None)
        self.sched = RankScheduler(self.worker.geom, CostModel(cp), sched_cfg or SchedConfig())
        self.items: dict = {}
        self.worker.tb.tokens_of = self._token_at
        self.n = 0

    def _token_at(self, rid, pos: int) -> int:
        it = self.items[rid]
        ids = it["ids"]
        if pos < len(ids):
            return ids[pos]
        k = pos - len(ids)
        if it.get("forced") is not None:                          # forced decode: the stream is prompt + forced
            return it["forced"][k] if k < len(it["forced"]) else -1
        return it["out"][k] if k < len(it["out"]) else -1        # -1: the engine fills drafts on the device

    def _submit(self, key, ids: list, max_new: int, score_from: int | None, forced: list | None = None, salt=None,
                sampling: dict | None = None):
        rid = f"g{self.n}"
        self.n += 1
        upto = len(ids) if score_from is None else max(0, score_from - 1)
        hs = chain_hashes(ids, upto, rid, salt)
        q = Request(rid, len(ids), max_new, 0.0, lambda b, hs=hs, rid=rid: hs[b] if b < len(hs) else ("out", rid, b),
                    score=score_from is not None, forced=forced is not None)
        self.items[rid] = dict(key=key, ids=list(ids), out=[], lps=[], score_from=score_from, topk=[], max_new=max_new,
                               forced=list(forced) if forced is not None else None, frows={}, steps=[], t0=time.time(),
                               sampling=sampling)
        if not self.sched.add_request(q):
            raise ValueError(f"{key}: {self.sched.rejected[rid]}")
        return rid

    def _run(self, max_steps: int = 1_000_000) -> None:
        def on_step(p, batches, results):
            for res in results:
                for rid, toks in res.tokens.items():
                    self.items[rid]["out"].extend(toks)
                for rid, rows in res.logprobs.items():
                    it = self.items[rid]
                    if it["score_from"] is not None:
                        it["topk"].extend(rows)
                    else:
                        it["lps"].extend(r[1] for r in rows)
                for rid, (pos, lg, vs) in res.logits.items():      # forced rows: per vocab shard
                    it = self.items[rid]
                    it["frows"].setdefault(vs, []).append((pos, lg.float().cpu()))
            for w in p.work:
                it = self.items.get(w.rid)
                if it is not None and it.get("forced") is not None and (w.kind == "decode" or w.last_chunk):
                    it["steps"].append(w.n if w.kind == "decode" else 1)
        def seq_init(rid):
            it = self.items[rid]
            return list(it["ids"]), list(it["out"]), dict(it.get("sampling") or {"temperature": 0.0})

        run_offline(self.sched, self.worker, max_steps, on_step, seq_init=seq_init, require_done=True)

    def generate(self, items: list) -> list:
        rids = [self._submit(it["id"], it["prompt_ids"], int(it.get("max_tokens", 16)), None, salt=it.get("salt"),
                             sampling={k: it[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed") if k in it}
                             or None)
                for it in items]
        self._run()
        out = []
        for rid in rids:
            it = self.items.pop(rid)
            n = self._max_new(it)
            out.append(dict(id=it["key"], tokens=it["out"][:n], logprobs=(it["lps"][:n] or None)))
        return out

    def _max_new(self, it) -> int:
        return it["max_new"]                 # the last step may accept more drafts than the request has left

    def force(self, items: list, dump_dir: str | None = None, tp: int = 4) -> list:
        """Forced decode through the DECODE / verify path (= production's patch_force): the last prefill row predicts
        forced[0], then every step verifies k + 1 rows fed [last forced token, next k forced tokens], all accepted.
        items: [dict(id, prompt_ids, forced, n_dump, tag?)] -> [dict(id, idx, positions, logits [n_dump, V] fp32 (shards
        assembled), vocab_starts, rows_per_step)]; with dump_dir also writes patch_force's per-rank dump files."""
        rids = []
        for it in items:
            forced = [int(t) for t in it["forced"]]
            n = min(int(it.get("n_dump", len(forced))), len(forced))
            rids.append((self._submit(it["id"], [int(t) for t in it["prompt_ids"]], n, None, forced=forced,
                                      salt=it.get("salt")), it, n))
        self._run()
        out = []
        import torch

        from force_dump import write_item

        for rid, src, n in rids:
            it = self.items.pop(rid)
            P = len(it["ids"])
            shards = {}
            idx = None
            for vs, rows in it["frows"].items():
                keep = sorted((p, i, lg) for p, lg in rows for i, p in enumerate(p) if 0 <= p + 1 - P < n)
                seen, ts, ix = set(), [], []
                for p_, i, lg in keep:
                    if p_ in seen:
                        continue
                    seen.add(p_)
                    ts.append(lg[i:i + 1])
                    ix.append(p_ + 1 - P)
                shards[vs] = torch.cat(ts, 0) if ts else torch.zeros(0, 0)
                idx = ix
            idx = idx or []
            full = torch.cat([shards[v] for v in sorted(shards)], 1) if shards else None
            o = dict(id=it["key"], idx=idx, positions=[P - 1 + j for j in idx], logits=full,
                     vocab_starts=sorted(shards), rows_per_step=it["steps"], n_dump=n, prompt_len=P)
            if dump_dir:
                write_item(dump_dir, str(src.get("tag", it["key"])), tp, shards, idx, n, P, it["steps"], it["t0"])
            out.append(o)
        return out

    def score(self, items: list, topk: int = 20) -> list:
        rids = [self._submit(it["id"], it["ids"], 1, int(it["start"]), salt=it.get("salt")) for it in items]
        self._run()
        out = []
        for rid in rids:
            it = self.items.pop(rid)
            start = it["score_from"]
            rows = {pos: tk for pos, tk in it["topk"]}               # row at position p scores the token at p + 1
            pos = list(range(start, len(it["ids"])))
            out.append(dict(id=it["key"], positions=pos, topk=[rows.get(p - 1) for p in pos]))
        return out


def smoke_engine(a, cfg) -> None:
    """4-GPU decode smoke THROUGH THE WORKER with engine/engine.py's Engine: Worker(backend="engine") loads the engine,
    sizes and binds the pools; scheduler-seeded decoding sequences; every step = schedule -> StepBatch -> worker.run_step
    (tables, deltas, engine graphs per bucket) -> scheduler.update, so positions and KV advance for real."""
    import dataclasses
    import faulthandler
    import statistics
    import sys

    import torch

    import worker as WK

    def arm(s):
        faulthandler.dump_traceback_later(s, repeat=False, file=sys.stderr, exit=True)

    arm(400)                                   # engine load: extensions + 84 GB of weights per GPU from the page cache
    t0 = time.time()
    pieces = set_pieces(a.pieces)              # the engine reads $PHYS_ENGINE (engine_kw would override it)
    cfg = dataclasses.replace(cfg, engine_kw={})
    if a.engine_cls:                           # CPU dry run of this path: --engine-cls mock_engine.MockEngine
        cfg = dataclasses.replace(cfg, engine_cls=a.engine_cls, engine_kw={},
                                  max_kv_bytes=int(a.max_kv_gb * 1e9), reserve_bytes=int(1e9))
    w = WK.Worker(cfg, Geometry(), "engine", max_rows=a.max_rows)
    print(f"smoke[engine]: worker + engine up in {time.time() - t0:.0f} s; PHYS_ENGINE={pieces}; geometry rows cap "
          f"{w.geom.rows_cap}, seats per rank {w.geom.max_seqs_per_rank}", flush=True)
    arm(120)
    for p in w.pools:                          # zeroed pools: every record any step reads is finite
        p.paged.zero_()
        p.window[: p.n_window * WK.WINDOW_PAGE_BYTES].zero_()
    s = RankScheduler(w.geom, CostModel(), SchedConfig(rebalance=False, steal=False))
    rng = random.Random(a.seed)
    ctxs = [int(min(194_000, max(14_000, rng.lognormvariate(10.9, 0.6)))) for _ in range(a.seqs)]
    seed_decoding(s, ctxs)
    tb_fill = WK.TableBuilder(stub_indexer=True)  # separate builder: the worker's delta state stays untouched
    p = s.schedule(0.0)
    batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
    nrec = [WK.fill_synthetic(w.pools[r], tb_fill.build(batches[r]), range(40), sources=(2, 8, 14, 20)) for r in range(4)]
    print(f"smoke[engine]: {a.seqs} sequences, rows per rank {[b.n_rows for b in batches]}, synthetic records {nrec}",
          flush=True)
    times, host = [], []
    for step in range(a.smoke_steps):
        arm(180 if step == 0 else 30)
        if step:
            p = s.schedule(0.0)
            batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
        t1 = time.time()
        res = w.run_step(batches)
        dt = time.time() - t1
        acc = {}
        for x in res:
            acc.update(x.accepted)
        s.update(p, acc, 0.0)
        times.append(dt)
        if step < 3 or step % 50 == 0:
            print(f"  step {step}: {dt * 1e3:.1f} ms wall (worker tables + engine), rows {[b.n_rows for b in batches]}, "
                  f"accepted {sum(acc.values())}", flush=True)
    faulthandler.cancel_dump_traceback_later()
    eng = w.backend
    steady = times[3:] or times
    print(json.dumps(dict(steps=len(times), first_ms=round(times[0] * 1e3, 1), steady_ms_median=round(statistics.median(steady) * 1e3, 2),
                          steady_ms_mean=round(statistics.mean(steady) * 1e3, 2), engine_stats=getattr(eng, "stats", None),
                          mem_alloc_gb=[round(torch.cuda.memory_allocated(d) / 1e9, 1) if str(d).startswith("cuda") else 0
                                        for d in eng.devices])), flush=True)
    s.kv.check()
    print(f"smoke[engine] done in {time.time() - t0:.0f} s")


def run_offline_async(sched: RankScheduler, worker, max_steps: int = 10_000, on_step=None, on_settled=None) -> dict:
    """D13 loop: step N+1 is scheduled (and its compact tables built) BEFORE step N's results are applied to the
    scheduler -- the host side of overlapping the CPU with the GPU step. With a real async engine the dispatch returns
    at once and the results of N are awaited after N+1 is planned; the mock computes at dispatch, the order of the
    scheduler's view is the same."""
    import gc

    assert sched.cfg.async_mode and worker.cfg.async_mode
    now, steps, finished, out_tok = 0.0, 0, [], 0
    pending = None                                   # (plan, accepted) of the step in flight
    gc.freeze()
    while (sched.running or sched.waiting or pending is not None) and steps < max_steps:
        if steps % 64 == 63:
            gc.freeze()
        p = sched.schedule(now) if (sched.running or sched.waiting) else None
        res = None
        if p is not None and p.work:
            batches = [build_step_batch(p, sched.kv, worker.geom, r) for r in range(worker.geom.n_ranks)]
            res = worker.run_step(batches)           # dispatch step N+1 (in flight)
            if on_step:
                on_step(p, batches, res)
        if pending is not None:                      # settle step N
            pp, pacc = pending
            now += pp.est_us / 1e6
            out_tok += sum(min(a, sched.reqs[rid].max_new_tokens - sched.reqs[rid].num_output)
                           for rid, a in pacc.items() if rid in sched.reqs)
            finished += sched.update(pp, pacc, now)
            pending = None
            if on_settled and p is not None and p.work:
                on_settled(p, batches)                # step N settled while N+1 (p) is still in flight
        if res is not None:
            acc = {}
            for x in res:
                acc.update(x.accepted)
            pending = (p, acc)
        elif p is not None:
            sched.update(p, {}, now)
        steps += 1
    return dict(steps=steps, done=len(finished), out_tokens=out_tok, sim_s=round(now, 3), sched=dict(sched.stats))


def run_offline(sched: RankScheduler, worker, max_steps: int = 10_000, on_step=None, seq_init=None,
                require_done: bool = False, stall_steps: int = 256) -> dict:
    """Drive scheduler + worker until nothing is left (or max_steps). Time = the cost model's estimate per step (stub) or
    the measured wall time (GPU). seq_init(rid) -> (prompt_ids, output_ids, sampling dict): with it, every step goes
    through Worker.step with the request lifecycle (D11: new sequences with their sampling params and token history,
    dropped = finished / preempted), as the served path does; without it, Worker.run_step."""
    import gc
    import types

    now, steps, finished, out_tok = 0.0, 0, [], 0
    t_wall = 0.0
    live, dropped = set(), set()
    idle = 0                       # consecutive steps with nothing scheduled while requests wait (no progress)
    gc.freeze()        # long-lived token lists / hashes out of the cyclic GC's reach (gen-2 collections of
    while (sched.running or sched.waiting) and steps < max_steps:   # millions of ids cost ~300 ms each at 256 x 60K)
        if steps % 64 == 63:
            gc.freeze()
        p = sched.schedule(now)
        if not p.work:
            sched.update(p, {}, now)
            steps += 1
            idle += 1
            if idle >= stall_steps:
                break
            continue
        idle = 0
        batches = [build_step_batch(p, sched.kv, worker.geom, r) for r in range(worker.geom.n_ranks)]
        t0 = time.time()
        if seq_init is None:
            results = worker.run_step(batches)
        else:
            for rid in p.preempted:                 # recomputed from scratch: re-announced as new
                live.discard(rid)
                dropped.add(rid)
            new = []
            for sb in batches:
                for se in sb.seqs:
                    if se.rid not in live:
                        pr, out, smp = seq_init(se.rid)
                        new.append(types.SimpleNamespace(rid=se.rid, prompt_token_ids=pr, output_token_ids=out,
                                                         sampling_params=smp, structured=False))
                        live.add(se.rid)
            ps = types.SimpleNamespace(step=p.step, batches=batches, row_tokens=None, new_reqs=new,
                                       finished=set(dropped), preempted=set(), prev_tokens={}, order=[])
            dropped.clear()
            results = worker.step(ps)
        dt = time.time() - t0
        t_wall += dt
        acc = {}
        for res in results:
            acc.update(res.accepted)
        now += p.est_us / 1e6 if worker.pools is None else dt
        out_tok += sum(min(a, sched.reqs[rid].max_new_tokens - sched.reqs[rid].num_output) for rid, a in acc.items())
        fin = sched.update(p, acc, now)
        finished += fin
        for q in fin:
            live.discard(q.rid)
            dropped.add(q.rid)
        steps += 1
        if on_step:
            on_step(p, batches, results)
    left = len(sched.running) + len(sched.waiting)
    if left and require_done:
        why = f"no progress for {idle} steps" if idle >= stall_steps else f"max_steps {max_steps} reached"
        head = sched.waiting[0] if sched.waiting else None
        raise RuntimeError(f"run_offline stopped with {left} unfinished requests ({why}; head of queue: "
                           f"{getattr(head, 'rid', None)} prompt {getattr(head, 'prompt_len', None)})")
    return dict(steps=steps, done=len(finished), out_tokens=out_tok, sim_s=round(now, 3), wall_s=round(t_wall, 3),
                sched=dict(sched.stats), unfinished=left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="stub", choices=["stub", "gpu", "engine"])
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="GPU: allocation + decode steps on synthetic decoding sequences")
    ap.add_argument("--seqs", type=int, default=114)
    ap.add_argument("--smoke-steps", type=int, default=2)
    ap.add_argument("--weights", default="reuse:21")
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-cls", default="", help="engine smoke: another engine class (CPU dry run: mock_engine.MockEngine)")
    ap.add_argument("--max-kv-gb", type=float, default=2.0, help="engine smoke with --engine-cls: KV pool cap per rank")
    ap.add_argument("--gate-in", default=None, help="JSONL requests: {id, prompt_ids, max_tokens} (generate) or "
                    "{id, ids, start} (score)")
    ap.add_argument("--gate-out", default=None)
    ap.add_argument("--mode", default="score", choices=["score", "generate", "force"])
    ap.add_argument("--dump-dir", default=None, help="force mode: write patch_force-format dumps here")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--pieces", default="", help="backend engine: $PHYS_ENGINE = full (idx2f + Engram + HSD) | stub | "
                    "k=v,...; default $PHYS_ENGINE if set, else full")
    a = ap.parse_args()
    if a.backend == "engine":
        set_pieces(a.pieces)
    from worker import Worker, WorkerConfig

    if a.gate_in:
        eng = OfflineEngine(backend=a.backend, cfg=WorkerConfig(weights=a.weights, seed=a.seed), max_rows=a.max_rows)
        items = [json.loads(x) for x in open(a.gate_in) if x.strip()]
        if a.mode == "force":
            res = eng.force(items, a.dump_dir)
            res = [dict({k: v for k, v in r.items() if k != "logits"}, logits_shape=list(r["logits"].shape)
                        if r["logits"] is not None else None) for r in res]
        else:
            res = eng.score(items, a.topk) if a.mode == "score" else eng.generate(items)
        with open(a.gate_out, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print(f"gate: {len(res)} results -> {a.gate_out}")
        return

    cfg = WorkerConfig(weights=a.weights, seed=a.seed)
    rng = random.Random(a.seed)
    if a.backend == "stub":
        geom = Geometry(kv_bytes_per_rank=6_000_000_000)
        w = Worker(cfg, geom, "stub")
        s = RankScheduler(w.geom, CostModel(), SchedConfig())
        for i in range(a.requests):
            s.add_request(Request(f"r{i}", rng.randint(2_000, 60_000), rng.randint(50, 800), 0.0,
                                  lambda b, i=i: (f"conv{i % 7}", b)))
        print(json.dumps(run_offline(s, w, a.steps)))
        return
    if a.backend == "engine":
        return smoke_engine(a, cfg)
    import faulthandler
    import sys

    import torch

    def arm(s=90):                         # hard watchdog: no phase may take longer than s seconds (dump + exit)
        faulthandler.dump_traceback_later(s, repeat=False, file=sys.stderr, exit=True)

    arm(120)                               # weights + extensions + pools
    t0 = time.time()
    w = Worker(cfg, Geometry(), "gpu", max_rows=a.max_rows)
    s = RankScheduler(w.geom, CostModel(), SchedConfig(rebalance=False, steal=False))
    # contexts like those of a real decode step (14K .. 194K)
    ctxs = [int(min(194_000, max(14_000, rng.lognormvariate(10.9, 0.6)))) for _ in range(a.seqs)]
    arm(60)
    seed_decoding(s, ctxs)
    print(f"smoke: {a.seqs} decoding sequences seeded, ranks {[len(s.on_rank(r)) for r in range(4)]}, "
          f"ctx sum per rank {[sum(q.num_tokens for q in s.on_rank(r)) for r in range(4)]}", flush=True)
    import worker as WK

    for step in range(a.smoke_steps):
        arm(60)
        p = s.schedule(0.0)
        batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
        tabs = [w.tb.build(sb) for sb in batches]
        print(f"  step {step}: tables built", flush=True)
        nrec = [WK.fill_synthetic(w.pools[r], tabs[r], cfg.layers) for r in range(4)]
        for d in w.backend.devs:
            torch.cuda.synchronize(d)
        print(f"  step {step}: synthetic records written {nrec}", flush=True)
        t1 = time.time()
        for r in range(4):
            w.backend.prepare(r, w.pools[r], tabs[r])
        for d in w.backend.devs:
            torch.cuda.synchronize(d)
        print(f"  step {step}: prepared (FlashMLA plans, caches, layer-0 input)", flush=True)
        t2 = time.time()
        w.backend.run_layers(trace=(lambda L, name: print(f"    issued L{L} {name}", flush=True)) if step == 0 else None)
        w.backend.sync()
        t3 = time.time()
        res = [WK.head_stub(tabs[r], r, p.step, w.k1) for r in range(4)]
        acc = {}
        for x in res:
            acc.update(x.accepted)
        s.update(p, acc, t3)
        print(json.dumps(dict(step=step, rows=[b.n_rows for b in batches], synthetic_records=nrec,
                              prepare_s=round(t2 - t1, 3), layers_s=round(t3 - t2, 4), est_ms=round(p.est_us / 1e3, 2),
                              finite=w.backend.outputs_finite(),
                              mem_alloc_gb=[round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in w.backend.devs])),
              flush=True)
    # the step as one CUDA graph per rank on the last tables, replayed for ~8 s (timing + sustained power)
    arm(60)
    t4 = time.time()
    w.backend.capture()
    print("  graphs captured", flush=True)
    w.backend.replay(3)
    one = max(w.backend.replay(5))
    reps = max(10, int(8e6 / one))
    e0 = None
    try:
        import pynvml
        pynvml.nvmlInit()
        hd = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in cfg.devices]
        e0 = [pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in hd]
    except Exception:  # noqa: BLE001
        pass
    tw = time.time()
    per = w.backend.replay(reps)
    el = time.time() - tw
    watts = None
    if e0 is not None:
        watts = round(sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e for h, e in zip(hd, e0)) / 1000.0 / el, 0)
    print(json.dumps(dict(graph_capture_s=round(time.time() - t4 - el, 2), replays=reps,
                          step_ms_per_rank=[round(x / 1e3, 2) for x in per], step_ms=round(max(per) / 1e3, 2),
                          us_per_layer=round(max(per) / len(cfg.layers), 1), watts_4gpu=watts,
                          finite_after_replays=w.backend.outputs_finite())), flush=True)
    faulthandler.cancel_dump_traceback_later()
    s.kv.check()
    print(f"smoke done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
