"""Test driver for the engine regression tests: serve's scheduler + the engine on a fixed synthetic workload.

A run seeds decoding sequences directly into the KV pools (no prefill), optionally lets prompts arrive mid-run (real
prefill, chunked, CP across ranks when enabled), and steps the engine in one of three modes:

  sync      Engine.step with full host tables (the synchronous serving path);
  async_np  Engine.dispatch (the pipelined decode path) with each step awaited before the next is planned;
  async     Engine.dispatch pipelined: step N+1 is dispatched before step N's result is taken.

Synthetic KV is TIE-FREE: every page a seeded sequence owns has its own random contents and every index key its own
scale (unique_pages). With a few repeating page templates (identical keys), or with one shared index-key scale (e4m3
dot products on a coarse grid), the indexer's top-k sees exact ties, whose winners are arbitrary run to run, and
bitwise comparisons fail for that reason alone.

Prompt and output token ids are a deterministic hash of (sequence, position); no captured traffic is used.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ENGINE = "indexer=idx2f,engram=phys,head=hsd,hsd_fold=1,hsd_mhc2=1"
W = 4
SCHED = "mixed_target_tokens=12288,mixed_target_fill=1,mixed_target_wait_s=1.0,backlog_drain_wait_s=20.0"

_ENGINE: dict = {}


def engine_once(key: str, extra: str = ""):
    """The process's one engine (PHYS_ENGINE = the release set + `extra`), built on first use. A different engine in the
    same process is refused with a skip: it would need a second copy of the weights in GPU memory."""
    import pytest
    if key in _ENGINE:
        return _ENGINE[key]
    if _ENGINE:
        pytest.skip(f"one engine per process: this process already built {sorted(_ENGINE)}; run this module alone")
    os.environ["PHYS_ENGINE"] = RELEASE_ENGINE + ("," + extra if extra else "")
    from engine import engine as E
    kw = dict(indexer="idx2f", engram="phys", head="hsd")
    kw.update(E._env_kw())
    eng = E.Engine(E.EngineConfig(**kw))
    eng.make_pools()
    eng._fpr_runs = 0
    _ENGINE[key] = eng
    return eng


def tok(rid, pos) -> int:
    """Deterministic synthetic token id of (sequence, position)."""
    return (zlib.crc32(f"{str(rid).split('#')[0]}:{pos}".encode()) % 120_000) + 1000


def device_ids_from_host(eng):
    """host_ids for run(): on every step after a sequence's first, the host gives exactly the ids the engine would read
    on the device (the drafter's next_ids row of the previous step)."""
    def f(k, rid, n, streams):
        p = eng.prev_seq.get(rid)                # (rank, batch index, step serial) of its last step with a head
        if p is None or p[2] != eng.step_serial:  # called before this step's serial bump: the last step = step_serial
            return None
        return eng.hooks.hsd.next_ids[p[0]][p[1], :n].tolist()
    return f


@dataclasses.dataclass
class Workload:
    ctxs: list                                   # seeded decoders' contexts (one sequence each)
    steps: int
    arrive: dict = dataclasses.field(default_factory=dict)    # step -> [prompt lengths] arriving before that step
    arrive_out: int = 24                         # output tokens per arriving prompt
    cp: bool = False                             # serve's CP prefill (a long prompt's chunk split over the 4 ranks)
    cp_min: int = 2048
    served_geom: bool = True                     # the scheduler's row caps as the serving worker sets them
    sched: str = SCHED                           # SchedConfig overrides "k=v,..."
    seed: int = 1234
    index_scales: str = "pow2"                   # synthetic index-key scales: "pow2" per record (unique_pages) or
                                                 # "shared" (synth's one 1/16 for every key; tie-prone)


@dataclasses.dataclass
class Result:
    streams: dict                                # rid -> emitted token ids, in order
    steps: list                                  # per step: dict(step, serial, rows, prefill, cp, tok_md5, h_md5, nonfinite,
                                                 # accepted); h_md5 / nonfinite are absent on pipelined steps
    dv: list                                     # per head step: (engine step serial, {rid: drafts_valid})
    stats: dict                                  # the engine's counters, numeric ones as this run's increments
    sched_stats: dict


def unique_pages(pools, top: list, seed: int, chunk: int = 2048, index_scales: str = "pow2") -> None:
    """Overwrite pages [0, top[r]) of rank r's paged pool with pages that are all different (random records in the
    layout of engine/synth.py's templates: e4m3 NoPE + bf16 RoPE records with unit ue8m0 scales, e4m3 index keys),
    generated on the device. Every index key gets its own random power-of-two scale (the model's index keys use
    ue8m0 scales): with one shared scale the e4m3 dot products fall on one coarse grid and exact ties at the top-512
    boundary are common over long contexts. Arbitrary fp32 scales would be out of distribution: the indexer's partial
    sums are then inexact and their reduction order shows. Every byte still decodes to a finite value."""
    import torch
    from geometry import KV_SOURCE_LAYERS, PAGE_BYTES, PAGE_OFFSETS, RECORDS_PER_PAGE
    for r, p in enumerate(pools):
        dev = p.paged.device
        g = torch.Generator(device=dev).manual_seed(seed * 7919 + r)
        pv = p.paged.view(-1, PAGE_BYTES)
        for a0 in range(0, top[r], chunk):
            n = min(chunk, top[r] - a0)
            pg = pv[a0:a0 + n]
            for s in KV_SOURCE_LAYERS:
                rpp = RECORDS_PER_PAGE[s]
                o = PAGE_OFFSETS[f"comp{s}"]
                nope = (torch.randn(n, rpp, 448, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
                rope = (torch.randn(n, rpp, 64, generator=g, device=dev) * 0.5).to(torch.bfloat16).view(torch.uint8)
                pg[:, o:o + rpp * 576] = torch.cat([nope, rope], 2).reshape(n, -1)
                sc = torch.full((rpp, 8), 127, dtype=torch.uint8, device=dev)
                sc[:, 7] = 0
                pg[:, o + rpp * 576:o + rpp * 584] = sc.reshape(1, -1)
                o = PAGE_OFFSETS[f"idx{s}"]
                pg[:, o:o + rpp * 128] = torch.randn(n, rpp * 128, generator=g, device=dev).to(torch.float8_e4m3fn).view(torch.uint8)
                ex = torch.randint(-8, -1, (n, rpp), generator=g, device=dev).float()     # 2^-8 .. 2^-2, one per record
                sc_i = torch.exp2(ex) if index_scales == "pow2" else torch.full_like(ex, 1.0 / 16)
                pg[:, o + rpp * 128:o + rpp * 132] = sc_i.view(torch.uint8).reshape(n, -1)
        torch.cuda.synchronize(dev)


def _install_dv_log(eng) -> None:
    hsd = eng.hooks.hsd
    if hsd is None or getattr(hsd, "_fpr_dv", None) is not None:
        return
    orig = hsd.prepare
    hsd._fpr_dv = []

    def prepare(batches, states, bms):
        hsd._fpr_dv.append((eng.step_serial, {str(k): bool(v.drafts_valid) for k, v in states.items()}))
        return orig(batches, states, bms)
    hsd.prepare = prepare


def _head_rows(eng):
    import torch
    out = []
    for r in range(W):
        s_ = eng.last_st[r]
        sr = s_["batch"].sample_rows
        if sr:
            out.append(s_["h"].index_select(0, torch.tensor(sr, device=s_["dev"])))
    return out


def head_md5(eng, per_rank: bool = False):
    """Digest of the step's sampled head inputs (bf16 bits), over all ranks or one per rank."""
    import torch
    out = []
    for r in range(W):
        s_ = eng.last_st[r]
        sr = s_["batch"].sample_rows
        hm = hashlib.md5()
        if sr:
            h = s_["h"].index_select(0, torch.tensor(sr, device=s_["dev"]))
            hm.update(h.contiguous().view(torch.int16).cpu().numpy().tobytes())
        out.append(hm.hexdigest()[:16])
    return out if per_rank else hashlib.md5("".join(out).encode()).hexdigest()[:16]


def nonfinite(eng) -> int:
    import torch
    return sum(int((~torch.isfinite(h.float()).all(1)).sum()) for h in _head_rows(eng))


def run(eng, wl: Workload, mode: str = "sync", graphs=None, on_plan=None, host_ids=None, token_fn=None,
        ctx_fn=None) -> Result:
    """One run of `wl` from a fresh engine state. on_plan(k, plan, eng): called after step k is planned (it may drop
    work items, e.g. to make a sequence miss a step). host_ids(k, rid, n, streams) -> list of n ids or None: host-given
    input ids of a decode sequence's rows at step k (default: -1 = the engine's own device ids); streams = the tokens
    emitted so far per sequence. token_fn(rid, pos): prompt token ids (default: tok). ctx_fn(k, batches): the step's
    StepContext (sync mode; default None)."""
    import numpy as np
    import torch
    from cost_model import ENGINE_MEASURED_2026_11_07, CostModel, params_from_env
    from driver import seed_decoding
    from engine import synth
    from interfaces import build_step_batch
    from scheduler import RankScheduler, Request, SchedConfig
    from worker import TableBuilder, build_compact

    assert mode in ("sync", "async_np", "async"), mode
    tokf = token_fn or tok
    if eng._fpr_runs:
        eng.reset_state()
    eng._fpr_runs += 1
    torch.manual_seed(wl.seed)
    np.random.seed(wl.seed)
    random.seed(wl.seed)
    _install_dv_log(eng)
    if eng.hooks.hsd is not None:
        eng.hooks.hsd._fpr_dv = []
    asyn = mode != "sync"
    g_sched, cm = eng.geom, CostModel()
    if wl.served_geom:
        g_ = eng.geom
        g_sched = dataclasses.replace(g_, max_rows_per_rank=eng.cfg.max_mixed_rows,
                                      max_seqs_per_rank=min(g_.max_seqs_per_rank, eng.max_rows // g_.rows_per_decode_seq),
                                      max_tail_rows_per_rank=eng.max_tail_rows)
        cm = CostModel(params_from_env(ENGINE_MEASURED_2026_11_07))
    scfg = SchedConfig(rebalance=False, steal=False, async_mode=asyn, cp_prefill=wl.cp, cp_min=wl.cp_min)
    for x in filter(None, (y.strip() for y in wl.sched.split(","))):
        k_, v_ = x.split("=", 1)
        cur = getattr(scfg, k_)
        setattr(scfg, k_, v_.lower() in ("1", "true", "yes", "on") if isinstance(cur, bool) else type(cur)(v_))
    sched = RankScheduler(g_sched, cm, scfg)
    seed_decoding(sched, list(wl.ctxs), tag="d")
    top = [0] * W                                # tie-free pools: one template per page in use
    for sk in sched.kv.seqs.values():
        if sk.pages:
            top[sk.rank] = max(top[sk.rank], max(sk.pages) + 1)
    synth.init_pools(eng.pools, seed=wl.seed)                  # every page valid ...
    unique_pages(eng.pools, top, wl.seed, index_scales=wl.index_scales)   # ... and every page in use different
    stats0 = {k: v for k, v in eng.stats.items() if isinstance(v, (int, float))}
    tb = TableBuilder(tokens_of=tokf)
    clock = 0.0
    streams, log = {}, []

    def settle(plan, res, rec):
        acc, toks = {}, {}
        for x in res:
            acc.update(x.accepted)
            toks.update({str(k_): [int(t) for t in v] for k_, v in x.tokens.items()})
        for rid, tl in toks.items():
            streams.setdefault(rid, []).extend(tl)
        rec["accepted"] = {str(k_): int(v) for k_, v in acc.items()}
        rec["tok_md5"] = hashlib.md5(json.dumps(sorted(toks.items())).encode()).hexdigest()[:16]
        sched.update(plan, acc, clock)
        log.append(rec)

    def row_tokens(sb, k):
        out = []
        for i in range(sb.n_rows):
            se = sb.seqs[sb.seq_slot[i]]
            out.append(-1 if se.kind == "decode" else tokf(se.rid, sb.positions[i]))
        if host_ids is not None:
            for se in sb.seqs:
                if se.kind == "decode":
                    ids = host_ids(k, str(se.rid), se.n_rows, streams)
                    if ids is not None:
                        out[se.row0:se.row0 + se.n_rows] = [int(t) for t in ids]
        return out

    pend = None
    for k in range(wl.steps):
        for j, n_ in enumerate(wl.arrive.get(k, [])):
            rid = f"a{k}_{j}"
            sched.add_request(Request(rid, n_, wl.arrive_out, 0.0, lambda b, rid=rid: (rid, b)))
        if not (sched.running or sched.waiting):
            break
        plan = sched.schedule(clock)
        clock += plan.est_us / 1e6
        if on_plan is not None:
            on_plan(k, plan, eng)
        if not asyn:                              # the synchronous worker allocates each step's rows (+ drafts)
            for w in plan.work:
                if w.kind == "decode":
                    assert sched.kv.allocate(w.rid, w.start, w.start + 2 * 3 + 1), w.rid
                elif w.last_chunk and w.kind == "prefill":
                    assert sched.kv.allocate(w.rid, w.start, w.start + w.n + 3), w.rid
        batches = [build_step_batch(plan, sched.kv, eng.geom, r) for r in range(W)]
        rec = dict(step=k, rows=[sb.n_rows for sb in batches],
                   prefill=any(se.kind != "decode" for sb in batches for se in sb.seqs),
                   cp=sum(1 for sb in batches for se in sb.seqs if getattr(se, "cp", None) is not None),
                   cp_groups=len({str(se.rid).split("#")[0] for sb in batches for se in sb.seqs
                                  if getattr(se, "cp", None) is not None}))
        if not asyn:
            res = eng.step(batches, tokens=[row_tokens(sb, k) for sb in batches], copies=True, graphs=graphs,
                           ctx=None if ctx_fn is None else ctx_fn(k, batches))
            rec["serial"] = eng.step_serial
            rec["h_md5"], rec["nonfinite"] = head_md5(eng), nonfinite(eng)
            rec["h_rank"] = head_md5(eng, per_rank=True)
            settle(plan, res, rec)
            continue
        cts = [build_compact(tb, sb, 4, row_tokens(sb, k)) for sb in batches]
        eng.copies(batches)                       # the step's page copies before dispatch, as the serving worker does
        h = eng.dispatch(batches, cts, None)
        rec["serial"] = eng.step_serial
        if mode == "async_np":
            res = h.result()
            rec["h_md5"], rec["nonfinite"] = head_md5(eng), nonfinite(eng)
            settle(plan, res, rec)
            continue
        if h.done():                              # a synchronous fallback step: its head input is the engine's last
            rec["h_md5"], rec["nonfinite"] = head_md5(eng), nonfinite(eng)
        if pend is not None:
            settle(pend[0], pend[1].result(), pend[2])
        pend = (plan, h, rec)
    if pend is not None:
        settle(pend[0], pend[1].result(), pend[2])
    dv = list(eng.hooks.hsd._fpr_dv) if eng.hooks.hsd is not None else []
    stats = {k: (v - stats0.get(k, 0) if isinstance(v, (int, float)) else v) for k, v in eng.stats.items()}
    return Result(streams=streams, steps=log, dv=dv, stats=stats, sched_stats=dict(sched.stats))
