"""Arm adapter/steptimes.py's profiler during a BRRR 1-user run (adapter/prof_1u.sh): read PHYS_STEP_TIMES live from
line n0; after the 2nd prefill step (BRRR: the PP pass, then the TG pass's cached-prefix tail) count decode-only steps
and, at N, write <go> with the window length.
  python st_trigger.py steps.jsonl n0 N go_path window_steps"""
import json
import os
import sys
import time

path, n0, N, go, win = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
seen, pf, dec, t0 = 0, 0, 0, time.time()
buf = ""
while time.time() - t0 < 600:
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except FileNotFoundError:
        time.sleep(0.2)
        continue
    full = lines[:-1]                      # the last element is a partial line (or "")
    for ln in full[max(seen, n0):]:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("pf", 0) > 0:
            pf += 1
        elif pf >= 2 and r.get("dec", 0) >= 1:
            dec += 1
    seen = max(seen, len(full))
    if dec >= N:
        with open(go + ".tmp", "w") as f:
            f.write(win)
        os.rename(go + ".tmp", go)
        print(f"armed after {dec} decode steps (line {seen}, {time.time() - t0:.1f} s)")
        sys.exit(0)
    time.sleep(0.1)
print(f"timeout: prefill steps {pf}, decode steps {dec}")
