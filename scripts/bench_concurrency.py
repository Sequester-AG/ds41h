"""High-concurrency decode measure : same requests and window rule as scripts/conc_profile.py (ignore_eos, 2048 tokens, thinking off, continuous_usage_stats on, energy window once all streams pass --after tokens). No idle connection reuse, so a stream cannot inherit a closing warm-up connection. A run is valid only if every stream succeeds: no retries, a 255-stream run raises. Reports delivery rate and API server CPU every 5 s and exact final-token times."""
import argparse, asyncio, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx  # noqa: E402
from brrr.bench import calibrated_prompt  # noqa: E402
from profile_bench import Energy, spec_counts  # noqa: E402

HZ = os.sysconf("SC_CLK_TCK")


def cpu(p):
    f = open(f"/proc/{p}/stat").read().rsplit(") ", 1)[1].split()
    return (int(f[11]) + int(f[12])) / HZ


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--size", type=int, default=2048); ap.add_argument("--after", type=int, default=300); ap.add_argument("--energy-secs", type=float, default=8.0); ap.add_argument("--port", type=int, default=8100); ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    pid = int(subprocess.run(["pgrep", "-f", "bin/vllm serve"], capture_output=True, text=True).stdout.split()[0])
    async with httpx.AsyncClient(base_url=f"http://localhost:{a.port}", timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=0)) as http:
        model = (await http.get("/v1/models")).json()["data"][0]["id"]

        def body(prompt, max_tokens, stream=True):
            b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "top_p": 1.0, "stream": stream, "max_tokens": max_tokens, "ignore_eos": True, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
            if stream:
                b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            return b

        base = await calibrated_prompt(http, model, a.size)
        prompts = [f"Session {i}.\n{base}" for i in range(a.conc)]
        rs = await asyncio.gather(*(http.post("/v1/chat/completions", json=body(p, 1, stream=False)) for p in prompts))
        if any(r.status_code != 200 for r in rs):
            raise RuntimeError(f"prefill failed: {[r.status_code for r in rs if r.status_code != 200][:5]}")
        print(f"prefill passes: {a.conc}, {rs[0].json()['usage']['prompt_tokens']} prompt tokens", flush=True)
        counts = [0] * a.conc; status = {}; err = {}; ended = [False] * a.conc; last = [None] * a.conc; reasoning = [None] * a.conc

        async def stream(i):
            try:
                async with http.stream("POST", "/v1/chat/completions", json=body(prompts[i], a.max_tokens)) as resp:
                    status[i] = resp.status_code
                    if resp.status_code != 200:
                        err[i] = (await resp.aread()).decode("utf-8", "replace")[:300]
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        d = json.loads(line[6:])
                        if "error" in d:
                            err[i] = str(d["error"])[:300]
                        u = d.get("usage")
                        if u:
                            counts[i] = u.get("completion_tokens", 0); reasoning[i] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                        if d.get("choices"):
                            last[i] = time.perf_counter()
            except Exception as e:
                err[i] = f"{type(e).__name__}: {e}"[:300]
            finally:
                ended[i] = True

        t00 = time.perf_counter(); c00 = cpu(pid)
        tasks = [asyncio.create_task(stream(i)) for i in range(a.conc)]
        try:
            while sum(1 for c in counts if c >= a.after) < a.conc and not err and not any(ended) and time.perf_counter() - t00 < 200:
                await asyncio.sleep(0.05)
            print(f"window opens {time.perf_counter() - t00:.1f} s after the streams start; streams past {a.after} tokens: {sum(1 for c in counts if c >= a.after)} of {a.conc}", flush=True)
            if err or any(ended) or any(c < a.after for c in counts):
                raise RuntimeError(f"Invalid run: not all streams reached the measurement threshold; errors {list(err.items())[:3]}")
            energy = Energy()
            t0, n0, e0, s0 = time.perf_counter(), list(counts), energy.read(), await spec_counts(http)
            await asyncio.sleep(a.energy_secs)
            t1, n1, e1, s1 = time.perf_counter(), list(counts), energy.read(), await spec_counts(http)
            if err or any(ended) or any(y <= x for x, y in zip(n0, n1)):
                raise RuntimeError(f"Invalid window: a stream failed, finished, or did not advance; errors {list(err.items())[:3]}")
            dt, dn = t1 - t0, sum(n1) - sum(n0)
            print(f"c{a.conc} energy window {dt:.2f} s: {dn / dt:,.0f} tok/s aggregate, streams advancing {sum(1 for x, y in zip(n0, n1) if y > x)} of {a.conc}", flush=True)
            if e0 and e1:
                de = [(y - x) / 1e3 for x, y in zip(e0, e1)]
                print("  avg power " + " ".join(f"g{i}:{x / dt:.0f}W" for i, x in enumerate(de)) + f"; {sum(de) / dn * 1e3:.1f} mJ/token (all GPUs)", flush=True)
            drafts, acc = s1[0] - s0[0], s1[1] - s0[1]
            if drafts > 0:
                steps = drafts / a.conc
                print(f"  {steps / dt:.1f} steps/s = {1e3 * dt / steps:.2f} ms/step, {1 + acc / drafts:.2f} tokens/step/stream", flush=True)
            lastn, tl, cl, rates = sum(counts), time.perf_counter(), cpu(pid), []
            while not all(ended) and time.perf_counter() - t00 < 280:
                await asyncio.sleep(5)
                now, tn, cn = sum(counts), time.perf_counter(), cpu(pid)
                r = (now - lastn) / (tn - tl); rates.append(r)
                print(f"  t+{tn - t00:5.0f} s: {r:7,.0f} tok/s, API CPU {(cn - cl) / (tn - tl) * 100:4.0f} %, streams running {sum(1 for x in ended if not x)}, total {now}", flush=True)
                lastn, tl, cl = now, tn, cn
            failed = [i for i in range(a.conc) if status.get(i) != 200 or i in err or counts[i] != a.max_tokens]
            if failed:
                raise RuntimeError(f"Incomplete benchmark; failed stream indices: {failed[:12]}; errors {list(err.items())[:3]}")
            full = rates[:-1] or rates; n = len(full); third = max(1, n // 3)
            mean = lambda x: sum(x) / max(len(x), 1)
            lf, ll = min(last), max(last)
            print(f"final-token times: first +{lf - t00:.1f} s, median +{sorted(last)[a.conc // 2] - t00:.1f} s, last +{ll - t00:.1f} s; whole run {sum(counts) / (ll - t00):,.0f} tok/s")
            print(f"RESULT c{a.conc} @{a.size}: window {dn / dt:,.0f} tok/s; early/mid/late {mean(full[:third]):,.0f} / {mean(full[third:n - third]):,.0f} / {mean(full[n - third:]):,.0f}; whole run {sum(counts) / (ll - t00):,.0f}; last token +{ll - t00:.1f} s; API CPU {(cpu(pid) - c00) / (time.perf_counter() - t00) * 100:.0f} %; {a.conc} of {a.conc} streams complete, http 200 x {sum(1 for s in status.values() if s == 200)}; reasoning_tokens reported {sorted(set(reasoning))[:3]}", flush=True)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


asyncio.run(main())
