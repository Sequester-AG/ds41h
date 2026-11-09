"""cProfile of the adapter's host path in decode steps (CPU, fake worker); top functions by own time."""
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PHYS_GEOMETRY", "full")
os.environ["PHYS_ADAPTER_CHECK"] = "0"
import engine_util as U  # noqa: E402
from test_contract import prompt  # noqa: E402
from vllm import SamplingParams  # noqa: E402

N, CTX, STEPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
eng = U.make_engine()
core = U.core_of(eng)
S = core.scheduler
for i in range(N):
    eng.add_request(f"h{i}", {"prompt_token_ids": prompt(f"h{i}", CTX)}, SamplingParams(max_tokens=100000, ignore_eos=True))
while len(S.rs.running) < N or any(q.phase != "decode" for q in S.rs.running):
    eng.step()
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for i in range(STEPS):
    eng.step()
pr.disable()
print(f"{(time.perf_counter() - t0) / STEPS * 1e3:.1f} ms/step under the profiler")
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(28)
st.sort_stats("cumulative").print_stats(40)
