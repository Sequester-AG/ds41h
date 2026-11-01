"""Steady-state decode profile under BRRR-bench conditions (run with the bench venv python so the prompt is
the bench's own code prompt):

  python scripts/profile_bench.py [--size 2048] [--after 300] [--secs 2.0]
      [--port 8100] [--temperature 1.0] [--prefill16k]

The server must have been started with PROFILE=1 (torch profiler config). Like the bench: a calibrated code
prompt of --size tokens, a max_tokens=1 prefill pass, then the cached second pass streams up to 2048 tokens at
T=1, thinking off. The profiler is started once --after completion tokens have arrived (past the boost-clock
warm-up) and stopped --secs later; the request then runs to completion. Prints token rates of the profiled
window and of the whole stream. --prefill16k additionally profiles one uncached 16K prefill (separate trace
window before the decode window).
"""

import argparse
import asyncio
import ctypes
import json
import time

import httpx
from brrr.bench import calibrated_prompt


class Energy:
    """Total energy counters (mJ) of all GPUs via NVML (ctypes, no pynvml needed)."""

    def __init__(self):
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
            assert self.nvml.nvmlInit_v2() == 0
            n = ctypes.c_uint()
            self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(n))
            self.handles = []
            for i in range(n.value):
                h = ctypes.c_void_p()
                self.nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
                self.handles.append(h)
        except Exception:  # noqa: BLE001
            self.nvml = None

    def read(self):
        if self.nvml is None:
            return None
        out = []
        for h in self.handles:
            e = ctypes.c_ulonglong()
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e))
            out.append(e.value)
        return out


async def spec_counts(http) -> tuple[float, float]:
    """(num_drafts, num_accepted_tokens) from the server's Prometheus metrics (summed over labels)."""
    d = a = 0.0
    try:
        for line in (await http.get("/metrics")).text.splitlines():
            if line.startswith("vllm:spec_decode_num_drafts"):
                d += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens") and "per_pos" not in line:
                a += float(line.rsplit(" ", 1)[1])
    except Exception:  # noqa: BLE001
        pass
    return d, a


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--after", type=int, default=600)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prefill16k", action="store_true")
    ap.add_argument("--energy-pass", type=int, default=1,
                    help="1: a separate un-profiled stream first, energy window from token 200 to the stream end minus "
                         "100 tokens (~1300 tokens, ~3 s; NVML energy windows < 1 s read up to 10%% high)")
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=f"http://localhost:{a.port}", timeout=None) as http:
        model = (await http.get("/v1/models")).json()["data"][0]["id"]

        def body(prompt, max_tokens, stream=True):
            b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": a.temperature,
                 "top_p": 1.0, "stream": stream, "max_tokens": max_tokens,
                 "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
            if stream:  # stream_options is rejected for non-streaming requests
                b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            return b

        def usage(r):
            j = r.json()
            if "usage" not in j:
                raise RuntimeError(f"server error: {str(j)[:300]}")
            return j["usage"]

        if a.prefill16k:
            p16 = await calibrated_prompt(http, model, 16384)
            await http.post("/start_profile")
            t = time.perf_counter()
            r = await http.post("/v1/chat/completions", json=body(p16, 1, stream=False))
            print(f"prefill16k: {usage(r)['prompt_tokens']} tokens in {time.perf_counter() - t:.3f} s", flush=True)
            await http.post("/stop_profile")
        prompt = await calibrated_prompt(http, model, a.size)
        r = await http.post("/v1/chat/completions", json=body(prompt, 1, stream=False))
        print(f"prefill pass: {usage(r)['prompt_tokens']} prompt tokens", flush=True)
        if a.energy_pass:
            energy = Energy()
            marks = []  # (time, tokens, energy, spec counts) at every 100-token boundary from token 200 on
            async with http.stream("POST", "/v1/chat/completions", json=body(prompt, 2048)) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    u = json.loads(line[6:]).get("usage")
                    if not u:
                        continue
                    n = u.get("completion_tokens", 0)
                    if n >= 200 + 100 * len(marks):
                        marks.append((time.perf_counter(), n, energy.read(), await spec_counts(http)))
            if len(marks) >= 3 and marks[0][2]:
                m0, m1 = marks[0], marks[-2]  # drop the last 100 tokens (stream tail / EOS)
                dt, dn = m1[0] - m0[0], m1[1] - m0[1]
                de = [(y - x) / 1e3 for x, y in zip(m0[2], m1[2])]
                print(f"energy pass: tokens {m0[1]}..{m1[1]} in {dt:.3f} s = {dn / dt:.1f} tok/s; avg power "
                      + " ".join(f"g{i}:{x / dt:.0f}W" for i, x in enumerate(de))
                      + f"; {sum(de) / dn:.3f} J/token (all GPUs)", flush=True)
                steps = m1[3][0] - m0[3][0]
                if steps > 0:
                    acc = m1[3][1] - m0[3][1]
                    print(f"energy pass per decode step: {sum(de) / steps:.3f} J (all GPUs); {steps:.0f} steps, "
                          f"{1 + acc / steps:.2f} tokens/step, {1e3 * dt / steps:.2f} ms/step", flush=True)
            else:
                print("energy pass: stream too short or NVML unavailable", flush=True)
        arrivals = []
        prof = {"start": None, "stop": None, "tok0": None, "tok1": None}
        task = None
        energy = Energy()
        e_mark = {}  # energy window: from token a.after // 3 to token a.after (before the profiler starts)

        async def control():
            await http.post("/start_profile")
            prof["start"], prof["tok0"] = time.perf_counter(), arrivals[-1][1]
            await asyncio.sleep(a.secs)
            prof["stop"], prof["tok1"] = time.perf_counter(), arrivals[-1][1]
            await http.post("/stop_profile")

        t0 = time.perf_counter()
        async with http.stream("POST", "/v1/chat/completions", json=body(prompt, 2048)) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                usage = json.loads(line[6:]).get("usage")
                if not usage:
                    continue
                n = usage.get("completion_tokens", 0)
                arrivals.append((time.perf_counter(), n))
                if "e0" not in e_mark and n >= a.after // 3:
                    e_mark.update(e0=energy.read(), t0=arrivals[-1][0], n0=n, s0=await spec_counts(http))
                if task is None and n >= a.after:
                    e_mark.update(e1=energy.read(), t1=arrivals[-1][0], n1=n, s1=await spec_counts(http))
                    task = asyncio.create_task(control())
        if task is not None:
            await task
        tf, nf = arrivals[-1]
        ts, ns = arrivals[0]
        print(f"decode: {nf} tokens, TG {(nf - ns) / max(tf - ts, 1e-9):.1f} tok/s over the stream "
              f"(first arrival {ts - t0:.3f} s)", flush=True)
        if e_mark.get("e0") and e_mark.get("e1"):
            dt, dn = e_mark["t1"] - e_mark["t0"], e_mark["n1"] - e_mark["n0"]
            de = [(b - a) / 1e3 for a, b in zip(e_mark["e0"], e_mark["e1"])]  # J per GPU
            print(f"energy window: tokens {e_mark['n0']}..{e_mark['n1']} in {dt:.3f} s = {dn / dt:.1f} tok/s; "
                  f"avg power " + " ".join(f"g{i}:{x / dt:.0f}W" for i, x in enumerate(de)) +
                  f"; {sum(de) / dn:.3f} J/token (all GPUs), {sum(de) / dt / len(de) * (dt / dn) * 1e3:.1f} mJ/token/GPU")
            steps = e_mark["s1"][0] - e_mark["s0"][0]
            if steps > 0:
                acc = e_mark["s1"][1] - e_mark["s0"][1]
                print(f"energy per decode step: {sum(de) / steps:.3f} J (all GPUs) = {1e3 * sum(de) / steps / len(de):.0f} mJ/GPU; "
                      f"{steps:.0f} steps, {1 + acc / steps:.2f} tokens/step, {1e3 * dt / steps:.2f} ms/step")
        if prof["start"] is not None:
            win = [x for x in arrivals if prof["start"] <= x[0] <= prof["stop"]]
            if len(win) > 1:
                print(f"profiled window: {win[-1][1] - win[0][1]} tokens in {win[-1][0] - win[0][0]:.3f} s = "
                      f"{(win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0]):.1f} tok/s (profiler overhead included)")
            pre = [x for x in arrivals if x[0] < prof["start"]]
            if len(pre) > 1:
                print(f"before profiling: {(pre[-1][1] - pre[0][1]) / max(pre[-1][0] - pre[0][0], 1e-9):.1f} tok/s "
                      f"({pre[-1][1] - pre[0][1]} tokens)")
        else:
            print(f"stream ended before {a.after} tokens: no profile taken")


if __name__ == "__main__":
    asyncio.run(main())
