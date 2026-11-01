"""Steady-state decode profile at concurrency N (the multi-stream sibling of profile_bench.py; run with the bench venv
python so the prompt is the bench's own code prompt):

  python scripts/conc_profile.py --conc 64 [--size 2048] [--after 300]
      [--energy-secs 3.0] [--secs 1.0] [--port 8100] [--temperature 1.0] [--no-profile]

N streams, each with its own copy of a calibrated code prompt of --size tokens (a per-stream first line keeps the KV of the
streams distinct), are prefilled with max_tokens=1 passes, then all N stream up to 2048 tokens at T=1 with ignore_eos, so no
stream finishes inside the windows. Once every stream has passed --after tokens: an un-profiled energy window of
--energy-secs (aggregate tok/s, per-stream tok/s, J/token, decode steps from the spec-decode counters, ms/step), then (unless
--no-profile) a torch-profiler window of --secs (server started with PROFILE=1). Analyze the traces with
scripts/step_profile.py --decode or scripts/cycle_profile.py.
"""

import argparse
import asyncio
import json
import time

import httpx
from brrr.bench import calibrated_prompt

from profile_bench import Energy, spec_counts


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--after", type=int, default=300)
    ap.add_argument("--energy-secs", type=float, default=3.0)
    ap.add_argument("--secs", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=f"http://localhost:{a.port}", timeout=None,
                                 limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as http:
        model = (await http.get("/v1/models")).json()["data"][0]["id"]

        def body(prompt, max_tokens, stream=True):
            b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": a.temperature,
                 "top_p": 1.0, "stream": stream, "max_tokens": max_tokens, "ignore_eos": True,
                 "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
            if stream:
                b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            return b

        base = await calibrated_prompt(http, model, a.size)
        prompts = [f"Session {i}.\n{base}" for i in range(a.conc)]
        rs = await asyncio.gather(*(http.post("/v1/chat/completions", json=body(p, 1, stream=False)) for p in prompts))
        print(f"prefill passes: {a.conc} x {rs[0].json()['usage']['prompt_tokens']} prompt tokens", flush=True)

        counts = [0] * a.conc
        done = asyncio.Event()

        async def stream(i):
            async with http.stream("POST", "/v1/chat/completions", json=body(prompts[i], 2048)) as resp:
                async for line in resp.aiter_lines():
                    if done.is_set():
                        break
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    u = json.loads(line[6:]).get("usage")
                    if u:
                        counts[i] = u.get("completion_tokens", 0)

        async def control():
            while min(counts) < a.after:
                await asyncio.sleep(0.05)
            energy = Energy()
            t0, n0, e0, s0 = time.perf_counter(), list(counts), energy.read(), await spec_counts(http)
            await asyncio.sleep(a.energy_secs)
            t1, n1, e1, s1 = time.perf_counter(), list(counts), energy.read(), await spec_counts(http)
            dt, dn = t1 - t0, sum(n1) - sum(n0)
            per = sorted((y - x) / dt for x, y in zip(n0, n1))
            print(f"c{a.conc} energy window {dt:.2f} s: {dn / dt:,.0f} tok/s aggregate, per stream p10/p50/p90 "
                  f"{per[len(per) // 10]:.0f}/{per[len(per) // 2]:.0f}/{per[9 * len(per) // 10]:.0f}", flush=True)
            if e0 and e1:
                de = [(y - x) / 1e3 for x, y in zip(e0, e1)]
                print(f"  avg power " + " ".join(f"g{i}:{x / dt:.0f}W" for i, x in enumerate(de))
                      + f"; {sum(de) / dn * 1e3:.1f} mJ/token (all GPUs)", flush=True)
            drafts, acc = s1[0] - s0[0], s1[1] - s0[1]
            if drafts > 0:
                steps = drafts / a.conc  # one draft per running request per step
                print(f"  {steps / dt:.1f} steps/s = {1e3 * dt / steps:.2f} ms/step, {1 + acc / drafts:.2f} tokens/step/stream"
                      + (f", {sum(de) / steps:.2f} J/step (all GPUs)" if e0 and e1 else ""), flush=True)
            if not a.no_profile:
                await http.post("/start_profile")
                p0, c0 = time.perf_counter(), sum(counts)
                await asyncio.sleep(a.secs)
                p1, c1 = time.perf_counter(), sum(counts)
                await http.post("/stop_profile")
                print(f"  profiled window: {c1 - c0} tokens in {p1 - p0:.2f} s = {(c1 - c0) / (p1 - p0):,.0f} tok/s "
                      "(profiler overhead included)", flush=True)
            done.set()

        tasks = [asyncio.create_task(stream(i)) for i in range(a.conc)]
        await control()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
