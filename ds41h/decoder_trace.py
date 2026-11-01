"""Timeline trace of the ds41h persistent decoder kernel (decoder.cu built with TRACE=1).

  python -m ds41h.decoder_trace [--m 1 6 8] [--layer 3] [--random] [--json out.json]   # solo (1 GPU)
  torchrun --nproc-per-node 4 -m ds41h.decoder_trace --dist [--m 1 6 8]                # 4 GPUs, NVLink
  ... --bench-seconds 2     also time the untraced build (CUDA-graph replay for 2 s per M) while
                            sampling nvidia-smi SM clock / power every 50 ms
  DS41H_DECODER_DEFINES=ROUTED_MMA=1,DENSE_MMA=1 python -m ds41h.decoder_trace   # wgmma variants

Every CTA records, per job: the producer's claim time (job index obtained) and stage-0 issue time,
the routing-table time (first routed job), and the consumer's free / stage-0-landed / dependencies-
met / end times plus exchange sub-phases (posted, peers' flags seen, reduced) and the time spun in
dependency waits (decoder.cu TR_* words). The summary gives

  * kernel span, SM clock during the kernel (clock64 cycles / globaltimer ns per CTA);
  * per job type: count, busy (dep -> end), dependency wait (landed -> dep), data wait (free ->
    landed), first / last start, last end, the type's weight bytes and its effective GB/s;
  * an occupancy timeline (CTAs busy / waiting on dependencies / waiting on weights per bin, with
    the busiest job types);
  * the critical path: walking back from the last job through whatever gated each job's start
    (its dependency producers, the weight stream, or the CTA's previous job).

Weights: checkpoint layer (tests/test_decoder.py helpers; DS41H_MODEL_DIR, default
$DS41H_HOME/model) or --random synthetic weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

JOB_NAMES = ["WOA", "WOB", "XA", "OWNA", "XNORM", "GATE", "ROUTE", "SH1", "SH2", "MIX", "R1", "R2", "XF", "R1C", "R2C"]
(J_WOA, J_WOB, J_XA, J_OWNA, J_XNORM, J_GATE, J_ROUTE, J_SH1, J_SH2, J_MIX, J_R1, J_R2, J_XF, J_R1C, J_R2C) = range(15)
R1C_BLOCKS, R1C_STAGES = 18, 20
LC = False  # lossless coarse build (LOSSLESS=1: R1F / R2F in the J_R1C / J_R2C slots), set in main()
# record words (decoder.cu TR_*)
TR_TYPE, TR_AB, TR_FLAGS, TR_CLAIM, TR_ISSUE, TR_ROUTE, TR_FREE, TR_ACQ, TR_DEP, TR_END, TR_X0, TR_X1, TR_X2, TR_SPIN, TR_SPINS, TR_SWAIT, TR_G = range(17)
TR_RB, TR_RBE, TR_RAE = 17, 18, 19  # routing marks of gate jobs (route_block start / end, route_all done)
TR_S0, TR_S1, TR_S2, TR_S3 = 20, 21, 22, 23  # generic sub-phase marks (24-word records, decoder.cu TR_S*)
TF_FINISH = 1
OTOK, TILE, CHUNKS, SLICES, KS_WOB, KS_GATE = 32, 8, 40, 160, 16, 8
STAGE_BYTES = 8192


def job_bytes(t: int, m: int) -> int:
    """Weight bytes one job of type t streams (decoder.cu issue())."""
    if t == J_WOA:
        return (4 if m <= 16 else 2) * (STAGE_BYTES + 8)
    if t in (J_WOB, J_SH1):
        return 4 * (STAGE_BYTES + 8)
    if t == J_GATE:
        return 2 * STAGE_BYTES
    if t == J_SH2:
        return 5 * (STAGE_BYTES + 8)
    if t == J_OWNA:
        return 24 * 4 * 32 * 4
    if t == J_R1:
        return 4 * (STAGE_BYTES + 512)
    if t == J_R1C:  # K parts: stages = 20 / kp (kp unknown here: full K assumed, see summarize)
        if LC:  # R1F: 40 stages of 128 rows x 64 B + 10 scale groups of 2 KB (+ 2 KB f16 x slices from L2, not counted)
            return 40 * STAGE_BYTES + 10 * 2048
        return R1C_STAGES * (STAGE_BYTES + 512)
    if t in (J_R2, J_R2C):
        return 128 * 32 + 128 * 18 + 4 * STAGE_BYTES
    return 0


# ------------------------------------------------------------------------------- decode
class Trace:
    """One launch's records (absolute ns relative to the earliest CTA entry of that launch)."""

    def __init__(self, buf: np.ndarray, slot: int, hdr_words: int, cap: int, rec_words: int):
        b = buf[slot].astype(np.uint64)
        hdr = b[:, :hdr_words]
        self.ctas = b.shape[0]
        self.count = hdr[:, 0].astype(np.int64)
        self.attempted = hdr[:, 7].astype(np.int64)
        t0 = hdr[:, 2] | (hdr[:, 3] << np.uint64(32))
        self.abs0 = int(t0[self.count > 0].min()) if (self.count > 0).any() else 0
        self.cta_t0 = (t0.astype(np.int64) - self.abs0).astype(np.int64)
        self.cta_exit = self.cta_t0 + hdr[:, 4].astype(np.int64)
        self.epoch = int(hdr[0, 6])
        self.m = int(hdr[0, 8])
        self.entries = int(hdr[:, 9].max())
        cyc = (hdr[:, 10] | (hdr[:, 11] << np.uint64(32))).astype(np.float64)
        ns = np.maximum(hdr[:, 4].astype(np.float64), 1.0)
        self.mhz = (cyc / ns * 1e3)[self.count > 0]
        self.gt_res = int(hdr[0, 12])
        # last CTA's counter reset + epoch advance done (decoder.cu tail, TRACE builds), ns rel. to t0
        lastc = np.where(hdr[:, 17] == 0xFFFF)[0] if hdr.shape[1] > 18 else []
        self.reset_end = int(self.cta_t0[lastc[0]] + int(hdr[lastc[0], 18])) if len(lastc) else -1
        # PROUTE route_local timeline per CTA (entry / C_G complete seen / done), absolute like the records
        rt = hdr[:, [13, 14, 15, 16, 17] if hdr.shape[1] > 17 else [13, 14, 15]].astype(np.int64)
        self.rt = np.where(rt > 0, rt + self.cta_t0[:, None], -1)  # entry, C_G seen, done[, epoch read, top-k done]
        self.rt_ld = hdr[:, 18].astype(np.int64) if hdr.shape[1] > 19 else None  # durations (ns), not times
        self.rt_sc = hdr[:, 19].astype(np.int64) if hdr.shape[1] > 19 else None
        recs = b[:, hdr_words:].reshape(self.ctas, cap, rec_words)
        rows = []
        for cta in range(self.ctas):
            n = int(self.count[cta])
            if n:
                r = recs[cta, :n].astype(np.int64)
                rows.append(np.concatenate([np.full((n, 1), cta), np.arange(n)[:, None], r], axis=1))
        R = np.concatenate(rows) if rows else np.zeros((0, 2 + rec_words), np.int64)
        self.cta = R[:, 0]
        self.seq = R[:, 1]
        w = R[:, 2:]
        self.type = w[:, TR_TYPE] & 0xFF
        self.c = (w[:, TR_TYPE] >> 8) & 0xFF
        self.n = (w[:, TR_TYPE] >> 16) & 0xFFFF
        self.a = w[:, TR_AB] & 0xFFFF
        self.b = (w[:, TR_AB] >> 16) & 0xFFFF
        self.flags = w[:, TR_FLAGS]
        base = self.cta_t0[self.cta]

        def t(col, zero_is_none=False):
            v = w[:, col] + base
            return np.where(w[:, col] == 0, -1, v) if zero_is_none else v

        self.claim = t(TR_CLAIM)
        self.issue = t(TR_ISSUE)
        self.route = t(TR_ROUTE, True)
        self.free = t(TR_FREE)
        self.acq = t(TR_ACQ)
        self.dep = t(TR_DEP)
        self.end = t(TR_END)
        self.x0, self.x1, self.x2 = t(TR_X0, True), t(TR_X1, True), t(TR_X2, True)
        self.g = t(TR_G, True) if w.shape[1] > TR_G else np.full(len(w), -1)  # dense GEMM stage loop end
        none = np.full(len(w), -1)
        self.rb, self.rbe, self.rae = (t(k, True) if w.shape[1] > k else none for k in (TR_RB, TR_RBE, TR_RAE))
        self.s = [t(k, True) if w.shape[1] > k else none for k in (TR_S0, TR_S1, TR_S2, TR_S3)]
        self.spin = w[:, TR_SPIN]
        self.spins = w[:, TR_SPINS]
        self.swait = w[:, TR_SWAIT]  # ns waiting for the job's later stages to land
        self.N = len(self.type)
        # per-CTA previous record index
        self.prev = np.full(self.N, -1)
        order = np.lexsort((self.seq, self.cta))
        for i in range(1, len(order)):
            if self.cta[order[i]] == self.cta[order[i - 1]]:
                self.prev[order[i]] = order[i - 1]

    @property
    def span(self) -> int:
        return int(self.cta_exit.max())


def pick_slot(buf: np.ndarray, hdr_words: int) -> list[int]:
    """Slots ordered by epoch (oldest first), only those that hold records."""
    ep = buf[:, 0, 6].astype(np.int64)
    ok = buf[:, :, 0].sum(1) > 0
    return [int(s) for s in np.argsort(ep) if ok[s]]


# ------------------------------------------------------------------------------- analysis
def type_table(T: Trace) -> list[dict]:
    out = []
    for t in range(len(JOB_NAMES)):
        sel = T.type == t
        if not sel.any():
            continue
        busy = T.end[sel] - T.dep[sel]
        depw = T.dep[sel] - T.acq[sel]
        dataw = T.acq[sel] - T.free[sel]
        nbytes = job_bytes(t, T.m) * (int((T.c[sel] + 1).sum()) if t == J_R2C else int(sel.sum()))  # R2C: c = extra row blocks (R2P)
        first_issue = int(T.issue[sel].min())
        last_end = int(T.end[sel].max())
        row = dict(
            type=JOB_NAMES[t], count=int(sel.sum()), busy_us=busy.sum() / 1e3, busy_mean_us=busy.mean() / 1e3, busy_max_us=busy.max() / 1e3,
            depw_us=depw.sum() / 1e3, dataw_us=dataw.sum() / 1e3, spin_us=T.spin[sel].sum() / 1e3, swait_us=T.swait[sel].sum() / 1e3,
            first_dep_us=T.dep[sel].min() / 1e3, p50_dep_us=float(np.median(T.dep[sel])) / 1e3, last_dep_us=T.dep[sel].max() / 1e3,
            last_end_us=last_end / 1e3, first_issue_us=first_issue / 1e3, mbytes=nbytes / 1e6,
            gbps=nbytes / max(1, last_end - first_issue),
        )  # fmt: skip
        # producer side: claim after the previous job's consumer start, issue after claim, stage 0
        # landed after issue, consumer idle before landing (us means)
        pv = T.prev[sel]
        okp = pv >= 0
        if okp.any():
            idx = np.where(sel)[0][okp]
            pidx = pv[okp]
            claimed = T.claim[idx] > 0
            row["prod_us"] = [
                float(np.mean((T.claim[idx] - T.dep[pidx])[claimed])) / 1e3 if claimed.any() else float("nan"),
                float(np.mean(T.issue[idx] - T.claim[idx])) / 1e3,
                float(np.mean(T.acq[idx] - T.issue[idx])) / 1e3,
                float(np.mean(T.acq[idx] - np.maximum(T.free[idx], T.issue[idx]))) / 1e3,
            ]
        if t in (J_XA, J_XF):
            ok = (T.x0[sel] >= 0) & (T.x1[sel] >= 0)
            if ok.any():
                row["post_us"] = float(np.mean((T.x0[sel] - T.dep[sel])[ok])) / 1e3
                row["peer_wait_us"] = float(np.mean((T.x1[sel] - T.x0[sel])[ok])) / 1e3
                row["peer_wait_max_us"] = float(np.max((T.x1[sel] - T.x0[sel])[ok])) / 1e3
                row["reduce_us"] = float(np.mean((T.x2[sel] - T.x1[sel])[ok & (T.x2[sel] >= 0)])) / 1e3
        else:  # sub-phase marks (decoder.cu tmark): mean us of dep->X0, ->G (GEMM loop end), ->X1, ->X2, ->end
            marks = [T.dep[sel], T.x0[sel], T.g[sel], T.x1[sel], T.x2[sel], T.end[sel]]
            ph = []
            prev = marks[0].copy()
            for mk in marks[1:]:
                ok = mk >= 0
                ph.append(float(np.mean((mk - prev)[ok])) / 1e3 if ok.any() else float("nan"))
                prev = np.where(ok, mk, prev)
            row["phases_us"] = ph
        out.append(row)
    return out


def timeline(T: Trace, bin_ns: int) -> list[dict]:
    nb = T.span // bin_ns + 1
    out = []
    for i in range(nb):
        lo, hi = i * bin_ns, (i + 1) * bin_ns
        busy = (T.dep < hi) & (T.end > lo)
        depw = (T.acq < hi) & (T.dep > lo) & ~busy
        dataw = (T.free < hi) & (T.acq > lo) & ~busy & ~depw
        # overlap-weighted busy share per type
        ov = np.clip(np.minimum(T.end, hi) - np.maximum(T.dep, lo), 0, None)
        by = defaultdict(float)
        for t in np.unique(T.type[busy]):
            by[JOB_NAMES[t]] = float(ov[T.type == t].sum()) / bin_ns
        top = sorted(by.items(), key=lambda kv: -kv[1])[:3]
        out.append(dict(t_us=lo / 1e3, busy=float(ov.sum()) / bin_ns, depw=int(depw.sum()), dataw=int(dataw.sum()), top=top))
    return out


def dep_candidates(T: Trace, i: int) -> np.ndarray:
    """Records whose completion the job i waited for (mirror of decoder.cu deps())."""
    t, a, b, c = int(T.type[i]), int(T.a[i]), int(T.b[i]), int(T.c[i])
    fin = (T.flags & TF_FINISH) != 0
    split = T.m > 16
    if t == J_WOB:  # the finish_woa jobs of the z range, or (FUSED_Z) every wo_a job of it
        r = np.where((T.type == J_WOA) & (T.c == c) & (T.a // 8 == b) & fin)[0]
        return r if len(r) else np.where((T.type == J_WOA) & (T.c == c) & (T.a // 8 == b))[0]
    if t == J_XA:
        return np.where((T.type == J_WOB) & (T.a // 2 == a) & fin)[0]
    if t == J_OWNA:  # BRRR XA jobs, or (fused all-reduce #1) every WOB K part of the slice's row block
        xa = np.where((T.type == J_XA) & (T.a == a // 4))[0]
        return xa if len(xa) else np.where((T.type == J_WOB) & (T.a == a // 2))[0]
    if t in (J_XNORM, J_MIX):
        return np.where(T.type == J_OWNA)[0]
    if t == J_GATE:
        return np.where(T.type == (J_XNORM if split else J_OWNA))[0]
    if t == J_ROUTE:
        return np.where((T.type == J_GATE) & (T.c == a // OTOK))[0]
    if t == J_SH1:
        if split:
            return np.where((T.type == J_XNORM) & (T.a == b))[0]
        return np.where((T.type == J_GATE) & (T.a == 0) & (T.c == c) & (T.b * KS_GATE * 16 // 512 == b))[0]
    if t == J_SH2:
        return np.where((T.type == J_SH1) & (T.c == c) & fin)[0]
    if t in (J_R1, J_R1C):  # routing table: per-token ROUTE jobs (old builds) or the last gate job (route_all)
        r = np.where(T.type == J_ROUTE)[0]
        return r if len(r) else np.where((T.type == J_GATE) & fin)[0]
    if t == J_R2:
        return np.where((T.type == J_R1) & (T.a == a) & fin)[0]
    if t == J_R2C:  # every R1C block of the expert slot (K parts: the finishing ones)
        r = np.where((T.type == J_R1C) & (T.a == a))[0]
        return r
    if t == J_XF:
        return np.where(((T.type == J_SH2) & (T.a // 2 == a)) | (np.isin(T.type, (J_R2, J_R2C)) & (T.b == a)))[0]
    return np.zeros(0, np.int64)


def critical_path(T: Trace, eps: int = 300) -> tuple[list[dict], dict]:
    """Walk back from the last-ending job; at each job decide what gated its start."""
    i = int(np.argmax(T.end))
    steps, attrib = [], defaultdict(float)
    guard = 0
    while i >= 0 and guard < 10000:
        guard += 1
        name = JOB_NAMES[T.type[i]]
        ex = int(T.end[i] - T.dep[i])
        attrib[f"exec {name}"] += ex
        depw = int(T.dep[i] - T.acq[i])
        dataw = int(T.acq[i] - T.free[i])
        step = dict(job=name, a=int(T.a[i]), b=int(T.b[i]), c=int(T.c[i]), cta=int(T.cta[i]), free_us=T.free[i] / 1e3, acq_us=T.acq[i] / 1e3,
                    dep_us=T.dep[i] / 1e3, end_us=T.end[i] / 1e3)  # fmt: skip
        if T.type[i] in (J_XA, J_XF) and T.x0[i] >= 0 and T.x1[i] >= 0:
            attrib[f"peer wait {name}"] += int(T.x1[i] - T.x0[i])
            attrib[f"exec {name}"] -= int(T.x1[i] - T.x0[i])
        if T.type[i] in (J_R1, J_R1C) and T.route[i] >= 0 and dataw > eps and T.route[i] + eps >= T.issue[i]:
            # weights could not be issued before the routing table: jump to the routing
            cand = dep_candidates(T, i)
            if len(cand):
                d = int(cand[np.argmax(T.end[cand])])
                attrib["route->table->R1 data"] += int(T.dep[i] - T.end[d])
                step["gate"] = "route table"
                steps.append(step)
                i = d
                continue
        ready = max(int(T.acq[i]), int(T.free[i]))
        if depw > eps:
            cand = dep_candidates(T, i)
            cand = cand[(T.end[cand] <= T.dep[i] + eps) & (T.end[cand] > ready - eps)] if len(cand) else cand
            if len(cand):
                d = int(cand[np.argmax(T.end[cand])])
                attrib[f"dep latency ->{name}"] += max(0, int(T.dep[i] - max(T.end[d], T.acq[i])))
                attrib[f"dep wait {name} (overlapped)"] += 0
                step["gate"] = f"dep {JOB_NAMES[T.type[d]]}"
                steps.append(step)
                i = d
                continue
            attrib[f"dep wait {name} (unresolved)"] += depw
        if dataw > eps:
            attrib[f"data wait {name}"] += dataw
            step["gate"] = "weights"
        else:
            step["gate"] = "prev job"
        steps.append(step)
        p = int(T.prev[i])
        if p >= 0:
            attrib["gap (CTA bookkeeping)"] += max(0, int(T.free[i] - T.end[p]))
        else:
            attrib["kernel start -> first job"] += int(T.free[i])
        i = p
    steps.reverse()
    return steps, dict(sorted(((k, v / 1e3) for k, v in attrib.items() if v), key=lambda kv: -kv[1]))


def summarize(T: Trace, bin_us: float = 5.0, path_steps: int = 40) -> tuple[str, dict]:
    L = []
    span = T.span / 1e3
    L.append(f"launch epoch {T.epoch}: m={T.m} entries={T.entries} ctas={T.ctas} jobs={T.N} span {span:.1f} us "
             f"(first CTA entry -> last consumer exit); globaltimer step {T.gt_res} ns")  # fmt: skip
    if (T.attempted > T.count).any():
        L.append(f"  WARNING: {int((T.attempted > T.count).sum())} CTAs truncated (TRACE_CAP)")
    L.append(f"  SM clock during kernel (clock64/globaltimer): mean {T.mhz.mean():.0f} MHz, min {T.mhz.min():.0f}, max {T.mhz.max():.0f}")
    if T.reset_end >= 0:
        L.append(f"  kernel tail: last consumer exit {T.cta_exit.max() / 1e3:.1f} us, counter reset + epoch done {T.reset_end / 1e3:.1f} us")
    entry_spread = T.cta_t0.max() / 1e3
    L.append(f"  CTA entry spread {entry_spread:.1f} us; consumer exits {T.cta_exit.min() / 1e3:.1f}..{T.cta_exit.max() / 1e3:.1f} us")
    tt = type_table(T)
    L.append("  type    n   busy_us  mean  max   depw_us  dataw_us spin_us  first/p50/last dep us   last_end   MB     GB/s  stage-wait  [exchange | phases dep>X0>G>X1>X2>end]"
             "  prod: claim-after-prev-start/issue-after-claim/land-after-issue/land-after-max(free,issue)")
    for r in tt:
        ex = f"  sw {r['swait_us']:7.1f}"
        if "peer_wait_us" in r:
            ex += f"  post {r['post_us']:.1f} peer {r['peer_wait_us']:.1f} (max {r['peer_wait_max_us']:.1f}) reduce {r['reduce_us']:.1f}"
        elif "phases_us" in r:
            ex += "  ph " + "/".join("-" if v != v else f"{v:.2f}" for v in r["phases_us"])
        if "prod_us" in r:
            ex += "  prod " + "/".join("-" if v != v else f"{v:.2f}" for v in r["prod_us"])
        L.append(f"  {r['type']:<6}{r['count']:5d} {r['busy_us']:8.1f} {r['busy_mean_us']:5.2f} {r['busy_max_us']:5.1f} {r['depw_us']:8.1f} {r['dataw_us']:8.1f} "
                 f"{r['spin_us']:7.1f}  {r['first_dep_us']:6.1f} {r['p50_dep_us']:6.1f} {r['last_dep_us']:6.1f}  {r['last_end_us']:7.1f} {r['mbytes']:6.1f} {r['gbps']:7.0f}{ex}")  # fmt: skip
    total_busy = sum(r["busy_us"] for r in tt)
    L.append(f"  CTA-time: busy {total_busy:.0f} us of {T.ctas * span:.0f} ({100 * total_busy / max(1e-9, T.ctas * span):.0f}%), "
             f"dep wait {sum(r['depw_us'] for r in tt):.0f}, data wait {sum(r['dataw_us'] for r in tt):.0f}")  # fmt: skip
    tl = timeline(T, int(bin_us * 1e3))
    L.append(f"  occupancy ({bin_us:g} us bins; busy = CTA-equivalents executing, depw/dataw = CTAs waiting):")
    for r in tl:
        top = " ".join(f"{k}:{v:.0f}" for k, v in r["top"])
        bar = "#" * int(round(40 * r["busy"] / T.ctas))
        L.append(f"   {r['t_us']:6.1f} busy {r['busy']:6.1f} depw {r['depw']:4d} dataw {r['dataw']:4d} |{bar:<40}| {top}")
    # routing timeline: the row-block finishers (route_block) and the route_all job, then the routed weights
    rbj = np.where((T.type == J_GATE) & ((T.rb >= 0) | (T.rae >= 0)))[0]
    us = lambda v: f"{v / 1e3:6.1f}" if v >= 0 else "     -"  # noqa: E731
    gt = T.type == J_GATE
    if gt.any():
        L.append("  routing (us): gate jobs dep p50 %.1f, last GEMM end %.1f, last gate end %.1f" % (
            np.median(T.dep[gt]) / 1e3, T.g[gt].max() / 1e3, T.end[gt].max() / 1e3))
    for i in rbj[np.argsort(T.rb[rbj])]:
        ra = T.rae[i] >= 0
        sub = f" [loads {us(T.s[0][i])} select {us(T.s[1][i])} stored {us(T.s[2][i])}]" if T.s[0][i] >= 0 else ""
        L.append(f"   rb={T.a[i]} cta={T.cta[i]:<3} dep {us(T.dep[i])} gemm {us(T.g[i])} route_block {us(T.rb[i])}..{us(T.rbe[i])}{sub}"
                 + (f"  route_all {us(T.x0[i])} cand {us(T.s[3][i])} topk {us(T.x1[i])} table {us(T.x2[i])} done {us(T.rae[i])} end {us(T.end[i])}" if ra else ""))  # fmt: skip
    ow = np.where((T.type == J_OWNA) & (T.s[0] >= 0))[0]
    if len(ow):
        d = lambda a, b: np.median(a[ow] - b[ow]) / 1e3  # noqa: E731
        L.append(f"  OWNA sub-phases (median us): flags->inbox landed {d(T.s[0], T.x0):.2f}, ->mixes {d(T.s[1], T.s[0]):.2f}, ->lane sums "
                 f"{d(T.s[2], T.s[1]):.2f}, ->stores {d(T.s[3], T.s[2]):.2f}, ->end {d(T.end, T.s[3]):.2f}; dep->flags {d(T.x0, T.dep):.2f}")
    if (T.rt[:, 2] > 0).any():
        ok = T.rt[:, 2] > 0
        q = lambda v: f"{np.percentile(v, 10) / 1e3:.1f}/{np.median(v) / 1e3:.1f}/{np.percentile(v, 90) / 1e3:.1f}"  # noqa: E731
        L.append(f"   route_local (p10/p50/p90 us): entry {q(T.rt[ok, 0])}  C_G seen {q(T.rt[ok, 1])}  done {q(T.rt[ok, 2])}"
                 + (f"  epoch read {q(T.rt[ok, 3])}  top-k done {q(T.rt[ok, 4])}" if T.rt.shape[1] > 4 else "")
                 + (f"  [sum over pairs: loads {q(T.rt_ld[ok])} scores {q(T.rt_sc[ok])}]" if T.rt_ld is not None else ""))
    r1 = (T.type == J_R1C) | (T.type == J_R1)
    if r1.any() and (T.route[r1] >= 0).any():
        rt = T.route[r1 & (T.route >= 0)]
        L.append(f"   producers saw the table: first {rt.min() / 1e3:.1f} p50 {np.median(rt) / 1e3:.1f}; first R1 issue {T.issue[r1].min() / 1e3:.1f}, "
                 f"first R1 landed {T.acq[r1].min() / 1e3:.1f}, p50 landed {np.median(T.acq[r1]) / 1e3:.1f}")  # fmt: skip
    steps, attrib = critical_path(T)
    L.append("  critical path attribution (us): " + ", ".join(f"{k} {v:.1f}" for k, v in attrib.items() if v >= 0.05))
    L.append(f"  critical path (last {min(path_steps, len(steps))} of {len(steps)} steps):")
    for s in steps[-path_steps:]:
        L.append(f"   {s['job']:<5} a={s['a']:<4} b={s['b']:<3} c={s['c']:<2} cta={s['cta']:<3} free {s['free_us']:6.1f} landed {s['acq_us']:6.1f} "
                 f"dep {s['dep_us']:6.1f} end {s['end_us']:6.1f}  <- {s.get('gate', '')}")  # fmt: skip
    data = dict(epoch=T.epoch, m=T.m, entries=T.entries, span_us=span, mhz_mean=float(T.mhz.mean()), types=tt,
                timeline=tl, critical=attrib, path=steps)  # fmt: skip
    return "\n".join(L), data


def launch_gaps(buf: np.ndarray, hdr_words: int) -> list[float]:
    """Gap between consecutive launches: next launch's first CTA entry - previous last exit (us)."""
    slots = pick_slot(buf, hdr_words)
    out = []
    prev = None
    for s in slots:
        h = buf[s].astype(np.uint64)
        live = h[:, 0] > 0
        t0 = (h[:, 2] | (h[:, 3] << np.uint64(32)))[live].astype(np.int64)
        ex = t0 + h[live, 4].astype(np.int64)
        if prev is not None:
            out.append((t0.min() - prev) / 1e3)
        prev = ex.max()
    return out


# ------------------------------------------------------------------------------- running
def _test_module():
    spec = importlib.util.spec_from_file_location("ds41h_test_decoder", ROOT / "tests" / "test_decoder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SmiSampler:
    """nvidia-smi SM clock / power / util sampling (50 ms) of one GPU in the background."""

    def __init__(self, device: int):
        props = torch.cuda.get_device_properties(device)
        ident = None
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            ident = f"GPU-{uuid}"
        else:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ident = vis.split(",")[device] if vis else str(device)
        self.cmd = ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "50", "-i", ident]
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        time.sleep(0.3)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        rows = []
        for line in out.splitlines():
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass
        self.rows = np.array(rows) if rows else np.zeros((0, 3))

    def summary(self) -> str:
        r = self.rows[self.rows[:, 2] >= 50] if len(self.rows) else self.rows
        if not len(r):
            return "smi: no busy samples"
        clk, pw = r[:, 0], r[:, 1]
        hist = defaultdict(int)
        for c in clk:
            hist[int(c)] += 1
        h = " ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
        return f"smi ({len(r)} busy samples): SM clock mean {clk.mean():.0f} MHz (min {clk.min():.0f} max {clk.max():.0f}) power mean {pw.mean():.0f} W max {pw.max():.0f} W; clock hist {h}"


def graph_of(T, eng, idx, inp, launches: int = 8):
    """Graph of `launches` launches; idx = one layer or a list of layer copies used round-robin."""
    idxs = idx if isinstance(idx, (list, tuple)) else [idx]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for k in range(launches):
            T.run_kernel(eng, idxs[k % len(idxs)], inp)
    return g


def timed(g, launches: int, seconds: float) -> float:
    g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = 0
    start.record()
    while time.time() - t0 < seconds:
        for _ in range(20):
            g.replay()
        reps += 20
        if reps % 200 == 0:
            torch.cuda.synchronize()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / (reps * launches)


def analyze_dump(prefix: str, hdr: int = 20, cap: int = 192, rec: int = 0, full: int = 1) -> None:
    """In-situ dumps (decoder.py DS41H_DECODER_TRACE_DUMP=<prefix>, TRACE=1 build): <prefix>_r<rank>_<k>.npy rings of the
    last 8 layer launches each. Prints per dump and rank the launch spans / m / entries, a per-type phase table averaged
    over the launches, and the full summary of the median-span launch of the first `full` dumps (rank 0)."""
    import glob
    import re

    files = sorted(glob.glob(f"{prefix}_r*_*.npy"), key=lambda f: tuple(int(x) for x in re.findall(r"_r(\d+)_(\d+)\.npy$", f)[0])[::-1])
    if not files:
        print(f"no dumps {prefix}_r*_*.npy")
        return
    rows = defaultdict(list)
    shown = 0
    for f in files:
        rank, k = (int(x) for x in re.findall(r"_r(\d+)_(\d+)\.npy$", f)[0])
        buf = np.load(f).view(np.uint32)
        rw = rec or (buf.shape[2] - hdr) // cap  # record words: 20 (older builds) or 24
        slots = [sl for sl in pick_slot(buf, hdr)]
        trs = [Trace(buf, sl, hdr, cap, rw) for sl in slots]
        trs = [t for t in trs if t.N > 0 and t.count.max() < cap]
        if not trs:
            continue
        spans = [t.span / 1e3 for t in trs]
        print(f"[r{rank} dump {k}] launches {len(trs)}: m {sorted(set(t.m for t in trs))} entries {[t.entries for t in trs]} spans "
              f"{' '.join(f'{x:.0f}' for x in spans)} us, SM {np.mean([t.mhz.mean() for t in trs]):.0f} MHz")
        for t in trs:
            for r in type_table(t):
                rows[(rank, r["type"])].append(r)
        if rank == 0 and shown < full:
            med = trs[int(np.argsort(spans)[len(spans) // 2])]
            print(summarize(med)[0])
            shown += 1
    print("per rank / type, mean over all dumped launches: n, mean job us, first/p50/last dep, last end, phases dep>X0>G>X1>X2>end (us)")
    for (rank, ty), rs in sorted(rows.items()):
        def f(key):
            v = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean(v)) if v else float("nan")

        ph = np.nanmean(np.array([r["phases_us"] for r in rs if "phases_us" in r], dtype=float), 0) if any("phases_us" in r for r in rs) else []
        print(f"  r{rank} {ty:5s} n {f('count'):6.1f} job {f('busy_mean_us'):6.2f}  dep {f('first_dep_us'):6.1f} {f('p50_dep_us'):6.1f} "
              f"{f('last_dep_us'):6.1f}  end {f('last_end_us'):6.1f}  ph {'/'.join(f'{x:.2f}' for x in ph)}")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--analyze":  # offline: in-situ dumps of an end-to-end run
        return analyze_dump(sys.argv[2], full=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", action="store_true")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 6, 8])
    ap.add_argument("--layer", type=int, default=int(os.environ.get("DS41H_TEST_LAYER", "3")))
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--bin-us", type=float, default=5.0)
    ap.add_argument("--path-steps", type=int, default=30)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--bench-seconds", type=float, default=0.0, help="also time the untraced build with nvidia-smi sampling")
    ap.add_argument("--all-ranks", action="store_true", help="--dist: print every rank's summary (default rank 0 + short others)")
    ap.add_argument("--warm", type=float, default=1.5, help="seconds of back-to-back replay before the steady-state trace")
    ap.add_argument("--burst-detail", action="store_true", help="also print the full summary of the burst (boost-clock) launch")
    ap.add_argument("--layers", type=int, default=1, help="round-robin over N copies of the layer (separate memory: cold L2 like a real step)")
    ap.add_argument("--save", type=str, default="", help="path prefix: np.save the raw trace rings (load with Trace(np.load(f), slot, 16, 192, 16))")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    T = _test_module()
    from ds41h.decoder import DecoderEngine, load_raw_experts

    base_spec = os.environ.get("DS41H_DECODER_DEFINES", "")
    trace_spec = ",".join(filter(None, [base_spec, "TRACE=1"]))
    if args.dist:
        rank, dist, cpu = T._dist_setup()
        world = 4
    else:
        rank, dist, cpu, world = T.TEST_RANK, None, None, 1
        torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    eng = DecoderEngine(rank=rank, world=world, cpu_group=cpu, spec=trace_spec)
    assert int(eng.ext.TRACE) == 1
    beng = DecoderEngine(rank=rank, world=world, cpu_group=cpu, spec=base_spec) if args.bench_seconds > 0 else None
    if args.random:
        raw, experts = T.random_raw("cuda", seed=7)
        loader = lambda ids: None  # noqa: E731
    else:
        from ds41h.ref.weights import CheckpointReader

        reader = CheckpointReader(T.MODEL_DIR)
        raw = T.checkpoint_raw(reader, args.layer, rank, "cuda")
        experts = load_raw_experts(reader, args.layer, rank, "cuda", experts=range(0))
        loaded: set[int] = set()

        def loader(ids):
            if dist is not None:
                lst = [None] * world
                dist.all_gather_object(lst, sorted(ids), group=cpu)
                ids = sorted(set().union(*map(set, lst)))
            new = [e for e in ids if e not in loaded]
            if new:
                load_raw_experts(reader, args.layer, rank, "cuda", experts=new, out=experts)
                loaded.update(new)

    dense = T.dense_from_raw(raw)
    idx = eng.add_layer(dense, experts)
    bidx = beng.add_layer(dense, experts) if beng is not None else None
    copies = [tuple(t.clone() for t in experts) for _ in range(args.layers - 1)]
    idxs = [idx] + [eng.add_layer(T.dense_from_raw(raw), ex) for ex in copies]
    bidxs = [bidx] + [beng.add_layer(T.dense_from_raw(raw), ex) for ex in copies] if beng is not None else None
    W = T.RefWeights(raw, experts)
    variant = eng.variant
    global LC
    LC = bool(variant.get("coarse")) and bool(variant.get("lossless"))
    tag = f"{'4gpu' if world == 4 else 'solo'} r{rank} {'wg' + ('R' if variant['routed_mma'] else '') + ('D' if variant['dense_mma'] else '') + ('C' if variant.get('coarse') else '') + ('L' if LC else '') if any(variant.values()) else 'hmma'}"
    results = []
    for m in args.m:
        inp = T.make_inputs(m, "cuda", seed=300 + m, o_seed=300 + m + (1000 * rank if world == 4 else 0))
        T.fill_routed_experts(eng, idx, W, inp, loader)
        for ex in copies:
            for dst, src in zip(ex, experts):
                dst.copy_(src)
        for _ in range(4):
            T.run_kernel(eng, idx, inp)
        torch.cuda.synchronize()
        g = graph_of(T, eng, idxs, inp)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        H, CAP, REC = int(eng.ext.TRACE_HDR), int(eng.ext.TRACE_CAP), int(eng.ext.TRACE_REC)

        def capture(steady: bool):
            """burst: one graph replay right after idle (boost clocks); steady: the last 8 launches of
            `--warm` seconds of back-to-back replay (power-capped clocks), smi sampled meanwhile."""
            us, smi_txt = float("nan"), ""
            if steady:
                with SmiSampler(dev) as smi:
                    us = timed(g, 8, args.warm)
                smi_txt = smi.summary()
            else:
                eng.trace_dump(clear=True)
                g.replay()
                torch.cuda.synchronize()
            buf = eng.trace_dump().cpu().numpy().view(np.uint32)
            if args.save:
                np.save(f"{args.save}_m{m}_r{rank}_{'steady' if steady else 'burst'}.npy", buf)
            slots = pick_slot(buf, H)
            gaps = launch_gaps(buf, H)
            tr = Trace(buf, slots[len(slots) // 2], H, CAP, REC)
            spans = [Trace(buf, sl, H, CAP, REC).span / 1e3 for sl in slots]
            return us, smi_txt, gaps, tr, spans

        _, _, gaps_b, tr_b, spans_b = capture(False)
        text_b, data_b = summarize(tr_b, args.bin_us, args.path_steps)
        steady_us, smi_txt, gaps, tr, spans = capture(True)
        text, data = summarize(tr, args.bin_us, args.path_steps)
        head = (f"==== decoder trace [{tag}] layer {args.layer} x{args.layers} copies m={m}\n"
                f"  burst (1 graph of 8 after idle): spans {' '.join(f'{x:.0f}' for x in spans_b)} us, launch gaps {' '.join(f'{x:.1f}' for x in gaps_b)} us, "
                f"SM clock {data_b['mhz_mean']:.0f} MHz; critical: {', '.join(f'{k} {v:.1f}' for k, v in list(data_b['critical'].items())[:8])}\n"
                f"  steady ({args.warm:g} s replay, traced build): {steady_us:.1f} us/layer; spans {' '.join(f'{x:.0f}' for x in spans)} us; "
                f"launch gaps {' '.join(f'{x:.1f}' for x in gaps)} us; {smi_txt}")  # fmt: skip
        bench_line = ""
        if beng is not None:
            T.run_kernel(beng, bidx, inp)
            torch.cuda.synchronize()
            bg = graph_of(T, beng, bidxs, inp)
            bg.replay()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            bg.replay()
            t1.record()
            torch.cuda.synchronize()
            burst_us = t0.elapsed_time(t1) * 1e3 / 8
            with SmiSampler(dev) as smi:
                us = timed(bg, 8, args.bench_seconds)
            bench_line = f"  untraced build: burst {burst_us:.1f} us/layer, steady {us:.1f} us/layer over {args.bench_seconds:g} s; {smi.summary()}"
            data.update(bench_us=us, bench_burst_us=burst_us, smi_bench=smi.summary())
            del bg
        data.update(steady_us=steady_us, gaps_us=gaps, spans_us=spans, tag=tag, layer=args.layer, smi=smi_txt, burst=data_b)
        results.append(data)
        short = data["critical"]
        if rank == 0 or args.all_ranks or world == 1:
            print("\n".join(filter(None, [head, bench_line, "  -- steady-state launch:", text])), flush=True)
            if args.burst_detail:
                print("  -- burst launch:\n" + text_b, flush=True)
        else:
            print(f"[{tag}] m={m} steady span {data['span_us']:.1f} us, {steady_us:.1f} us/layer; "
                  f"critical: {', '.join(f'{k} {v:.1f}' for k, v in list(short.items())[:6])}" + (f"\n{bench_line}" if bench_line else ""), flush=True)  # fmt: skip
        del g
    if args.json:
        path = args.json if world == 1 else args.json.replace(".json", f".r{rank}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    if dist is not None:
        dist.barrier(group=cpu)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
