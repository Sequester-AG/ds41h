"""Host time of the adapter per engine step at scale (CPU, fake worker): PhysScheduler.schedule (RankScheduler +
StepBatch build + SchedulerOutput), vLLM's update_from_output + rs.update, the executor's merge, and the engine loop
around them. The fake worker's own time is measured separately and excluded.

  CUDA_VISIBLE_DEVICES= python adapter/tests/host_overhead.py --seqs 256 --ctx 60000 --steps 200
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PHYS_GEOMETRY", "full")
os.environ["PHYS_ADAPTER_CHECK"] = "0"
import engine_util as U  # noqa: E402
from test_contract import prompt  # noqa: E402
from vllm import SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", type=int, default=256)
ap.add_argument("--ctx", type=int, default=60000)
ap.add_argument("--steps", type=int, default=200)
a = ap.parse_args()
eng = U.make_engine()
core = U.core_of(eng)
S, X = core.scheduler, core.model_executor
T = dict(schedule=0.0, update=0.0, worker=0.0, merge=0.0, n=0, rs_schedule=0.0, rs_build=0.0, rs_update=0.0,
         outproc=0.0, encode=0.0, enc_bytes=0, corestep=0.0)
from vllm.v1.serial_utils import MsgpackEncoder  # noqa: E402

_enc = MsgpackEncoder()
OP = eng.output_processor
o_op = OP.process_outputs
o_cs = core.step_fn


def w_op(*a, **k):
    t = time.perf_counter(); r = o_op(*a, **k); T["outproc"] += time.perf_counter() - t; return r


def w_cs(*a, **k):
    t = time.perf_counter()
    r = o_cs(*a, **k)
    T["corestep"] += time.perf_counter() - t
    outs = r[0] or {}
    t = time.perf_counter()
    for eco in outs.values():                     # what EngineCoreProc serializes to the API process every step
        T["enc_bytes"] += sum(len(b) for b in _enc.encode(eco))
    T["encode"] += time.perf_counter() - t
    return r
o_s, o_u, o_w, o_r = S.schedule, S.update_from_output, X.worker.step, X._run
import adapter.phys_scheduler as PSM  # noqa: E402
o_rs, o_ru, o_b = S.rs.schedule, S.rs.update, PSM.build_step_batch


def w_rs(*a, **k):
    t = time.perf_counter(); r = o_rs(*a, **k); T["rs_schedule"] += time.perf_counter() - t; return r


def w_ru(*a, **k):
    t = time.perf_counter(); r = o_ru(*a, **k); T["rs_update"] += time.perf_counter() - t; return r


def w_b(*a, **k):
    t = time.perf_counter(); r = o_b(*a, **k); T["rs_build"] += time.perf_counter() - t; return r



def w_s(*a, **k):
    t = time.perf_counter(); r = o_s(*a, **k); T["schedule"] += time.perf_counter() - t; return r


def w_u(*a, **k):
    t = time.perf_counter(); r = o_u(*a, **k); T["update"] += time.perf_counter() - t; return r


def w_w(*a, **k):
    t = time.perf_counter(); r = o_w(*a, **k); T["worker"] += time.perf_counter() - t; return r


def w_r(*a, **k):
    t = time.perf_counter(); r = o_r(*a, **k); T["merge"] += time.perf_counter() - t; return r


for i in range(a.seqs):
    eng.add_request(f"h{i}", {"prompt_token_ids": prompt(f"h{i}", a.ctx + 97 * i % 5000)},
                    SamplingParams(max_tokens=100000, ignore_eos=True))
t0 = time.time()
while len(S.rs.running) < a.seqs or any(q.phase != "decode" for q in S.rs.running):
    eng.step()
print(f"prefilled {a.seqs} x ~{a.ctx} tokens in {time.time() - t0:.0f} s ({S.phys_stats['steps']} steps)", flush=True)
S.schedule, S.update_from_output, X.worker.step, X._run = w_s, w_u, w_w, w_r
S.rs.schedule, S.rs.update, PSM.build_step_batch = w_rs, w_ru, w_b
OP.process_outputs = w_op
core.step_fn = w_cs
import gc  # noqa: E402

GC = {0: [0, 0.0], 1: [0, 0.0], 2: [0, 0.0]}
_gc_t = [0.0]


def _gc_cb(phase, info):
    if phase == "start":
        _gc_t[0] = time.perf_counter()
    else:
        g = GC[info["generation"]]
        g[0] += 1
        g[1] += time.perf_counter() - _gc_t[0]


if os.environ.get("HOST_GC_FREEZE") == "1":
    gc.collect()
    gc.freeze()
gc.callbacks.append(_gc_cb)
t0 = time.perf_counter()
for i in range(a.steps):
    eng.step()
wall = time.perf_counter() - t0
gc.callbacks.remove(_gc_cb)
print("gc per generation (count, ms total): " + ", ".join(f"gen{k} {v[0]} / {v[1] * 1e3:.0f}" for k, v in GC.items())
      + f"; frozen {gc.get_freeze_count()}")
t1 = time.perf_counter()
gc.unfreeze()
ncyc = gc.collect()
print(f"cyclic garbage after the run: {ncyc} objects ({(time.perf_counter() - t1) * 1e3:.0f} ms full collect), "
      f"{ncyc / max(1, a.steps):.1f} per step")
n = a.steps
merge = T["merge"] - T["worker"]
print(f"decode steps {n}: wall {wall / n * 1e3:.2f} ms/step = schedule {T['schedule'] / n * 1e3:.2f} + update "
      f"{T['update'] / n * 1e3:.2f} + executor merge {merge / n * 1e3:.2f} + fake worker {T['worker'] / n * 1e3:.2f} + rest "
      f"{(wall - T['schedule'] - T['update'] - T['merge']) / n * 1e3:.2f} (engine loop, output processor, detokenizer)")
serve = T["rs_schedule"] + T["rs_build"] + T["rs_update"]
print(f"  serve/ parts: rs.schedule {T['rs_schedule'] / n * 1e3:.2f} + build_step_batch {T['rs_build'] / n * 1e3:.2f} + rs.update "
      f"{T['rs_update'] / n * 1e3:.2f} = {serve / n * 1e3:.2f} ms; adapter + vLLM engine-core side (schedule + update - serve "
      f"+ merge) = {(T['schedule'] + T['update'] + merge - serve) / n * 1e3:.2f} ms/step")
core_other = T["corestep"] - T["schedule"] - T["update"] - T["merge"]
print(f"  EngineCore.step {T['corestep'] / n * 1e3:.2f} ms (of which other than schedule/execute/update: "
      f"{core_other / n * 1e3:.2f}); serialization of EngineCoreOutputs to the API process {T['encode'] / n * 1e3:.2f} ms "
      f"({T['enc_bytes'] / n / 1e3:.1f} KB/step); API-process output processor (detokenizer, RequestOutputs) "
      f"{T['outproc'] / n * 1e3:.2f} ms/step")
