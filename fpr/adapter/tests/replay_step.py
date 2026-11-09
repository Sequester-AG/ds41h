"""Replay dumped worker inputs (PHYS_DUMP_STEPS) through serve's Worker.step, without vLLM (adapter/replay.py).

From fpr/ with PYTHONPATH=<fpr>:<repo root>:
  python adapter/tests/replay_step.py DUMP_DIR [more dirs / step files] [--force-dump OUT] [--engine-cls ...]
    --force-dump OUT: the forced requests' logits as patch_force dumps under OUT/<tag>/ (the served dump's format and
    tag naming, so a comparison against offline dumps reads them unchanged). Without it, only the steps run (engine
    debugging).
Env as in the server: PHYS_ENGINE="indexer=idx2f,engram=phys,head=hsd", IDX2F_BUILD=..., DS41H_MOE_DEC=1 (production's
MoE env, as scripts/serve_fpr.sh exports), CUDA_LAUNCH_BLOCKING=1 to localize a kernel fault. Runs on the 4 GPUs
with the real engine, or on the CPU with --engine-cls mock_engine.MockEngine.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from adapter import replay as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="+", help="step{i}.pkl files or directories of them")
    ap.add_argument("--engine-cls", default="engine.engine.Engine")
    ap.add_argument("--max-rows", type=int, default=320)
    ap.add_argument("--max-seqs", type=int, default=int(os.environ.get("MAX_SEQS", "160")))
    ap.add_argument("--engine-kw", default="")
    ap.add_argument("--force-dump", default="", help="write the forced requests' dumps here (mode 700)")
    ap.add_argument("--async", dest="async_", default="auto", choices=("auto", "0", "1"),
                    help="ASYNC=1 dumps (start -1 decode groups): replay through the D13 dispatch model (auto: detect)")
    ap.add_argument("--sync-each", action="store_true", help="async replay: synchronize after every step (fault -> step)")
    a = ap.parse_args(argv)
    kw = {}
    for kv in filter(None, a.engine_kw.split(",")):
        k, v = kv.split("=", 1)
        kw[k] = int(v) if v.isdigit() else v
    os.environ.setdefault("DS41H_MOE_DEC", "1")          # production's MoE env (scripts/serve_fpr.sh exports it)
    os.environ.setdefault("DS41H_MOE_DEC_CFG", "0x55")
    files = R.step_files(a.pkl)
    am = R.is_async(files) if a.async_ == "auto" else a.async_ == "1"
    t0 = time.time()
    w = R.make_worker(a.engine_cls, kw, a.max_rows, a.max_seqs, async_mode=am)
    print(f"worker up in {time.time() - t0:.0f} s, {w.geom.pools()[0].num_pages} pages per rank, "
          f"{len(files)} dumps, {'async (D13 dispatch)' if am else 'sync'}", flush=True)
    if am:
        out = R.replay_async(files, w, sync_each=a.sync_each, log=lambda m: print(m, flush=True))
        print("replay FAULT" if out.get("fault") else "replay ok")
        return 1 if out.get("fault") else 0
    if a.force_dump:
        os.makedirs(a.force_dump, mode=0o700, exist_ok=True)
    R.replay(files, w, a.force_dump or None)
    print("replay ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
