"""Per-phase Engram NVMe I/O + GPU stall statistics of one arm.

    python scripts/engram_nvme_stats.py <run dir>

Reads the poller's timestamped cumulative stats lines (DS41H_ENGRAM_NVME_STATS_S, server.log) and the arm's phase
markers (progress.log "== HH:MM:SS [tag] bench c1 ..." / "pf ..." / "tgd ..."), differences the counters over each phase
and prints per layer (summed over ranks): requests, row ids, reads, record-cache hits, GPU waits, % of waits that
stalled, mean stall per wait and per stalled wait (us), mean poller service time per request (us).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

LINE = re.compile(r"ds41h engram_nvme (\d\d):(\d\d):(\d\d)\.(\d+) r(\d+) L(\d+) \(cumulative\): (\d+) req, (\d+) ids "
                  r"\(([\d.]+)% unique, record cache hits ([\d.]+)%\), (\d+) reads.*GPU waits (\d+), stalled (\d+).*"
                  r"stall_ns (\d+) svc_ns (\d+) d_waits (\d+) d_spun (\d+) d_stall_ns (\d+)")
PHASE = re.compile(r"^== (\d\d):(\d\d):(\d\d) \[[^\]]+\] (.*)$")


def secs(h, m, s, ms="0"):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def main(out: str) -> int:
    samples = defaultdict(list)  # (rank, layer) -> [(t, dict)]
    for line in open(f"{out}/server.log", errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        g = m.groups()
        t = secs(*g[:4])
        req, ids, uniq_pct, hit_pct, reads, waits, spun, stall_ns, svc_ns, dw, ds_, dst = (
            int(g[6]), int(g[7]), float(g[8]), float(g[9]), int(g[10]), int(g[11]), int(g[12]), int(g[13]), int(g[14]),
            int(g[15]), int(g[16]), int(g[17]))
        uniq = ids * uniq_pct / 100
        samples[(int(g[4]), int(g[5]))].append((t, dict(req=req, ids=ids, uniq=uniq, hits=uniq * hit_pct / 100,
                                                      reads=reads, waits=waits, spun=spun, stall=stall_ns, svc=svc_ns,
                                                      dwaits=dw, dspun=ds_, dstall=dst)))
    marks = []
    for line in open(f"{out}/progress.log", errors="replace"):
        m = PHASE.match(line.strip())
        if m:
            marks.append((secs(*m.groups()[:3]), m.group(4)))
    phases = []  # phase = (previous marker time, this end marker time, label)
    allm = sorted(marks)
    for i, (t, x) in enumerate(allm):
        if re.match(r"(bench c\d+ rc|pf |tgd |long prefill)", x):
            t0 = allm[i - 1][0] if i else t
            phases.append((t0, t, x.split(" rc=")[0].split(":")[0]))

    def at(series, t):  # last sample at or before t
        best = None
        for ts, d in series:
            if ts <= t + 0.5:
                best = d
        return best

    layers = sorted({l for _, l in samples})
    print(f"{out}: {sum(len(v) for v in samples.values())} stats samples, layers {layers}")
    print(f"{'phase':<34} {'L':>3} {'req':>7} {'ids':>10} {'reads':>9} {'hit%':>5} {'waits':>6} {'stall%':>6} "
          f"{'stall/wait':>10} {'stall/stalled':>13} {'svc/req':>8} | {'dec waits':>9} {'dec stall%':>10} {'dec us/wait':>11}")
    for t0, t1, label in phases:
        for layer in layers:
            agg = defaultdict(float)
            for (rank, l), ser in samples.items():
                if l != layer:
                    continue
                a, b = at(ser, t0), at(ser, t1)
                if b is None:
                    continue
                a = a or {k: 0 for k in b}
                for k in b:
                    agg[k] += b[k] - a[k]
            if not agg or agg["waits"] <= 0:
                continue
            print(f"{label:<34} {layer:>3} {agg['req']:>7.0f} {agg['ids']:>10.0f} {agg['reads']:>9.0f} "
                  f"{100 * agg['hits'] / max(agg['uniq'], 1):>5.1f} {agg['waits']:>6.0f} "
                  f"{100 * agg['spun'] / agg['waits']:>6.1f} {agg['stall'] / agg['waits'] / 1e3:>10.1f} "
                  f"{agg['stall'] / max(agg['spun'], 1) / 1e3:>13.1f} {agg['svc'] / max(agg['req'], 1) / 1e3:>8.1f} | "
                  f"{agg['dwaits']:>9.0f} {100 * agg['dspun'] / max(agg['dwaits'], 1):>10.1f} "
                  f"{agg['dstall'] / max(agg['dwaits'], 1) / 1e3:>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
