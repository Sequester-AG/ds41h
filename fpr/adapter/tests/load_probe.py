"""Short load probe of a served engine: C concurrent clients, each sending synthetic chat requests of a given prompt
length with max_tokens / ignore_eos; reports output tok/s, TTFT and per-request decode rate (aggregates only).

  python adapter/tests/load_probe.py --port 8100 --conc 32,128 --prompt 8000 --max-tokens 400 --seconds 60
"""
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import threading
import time
import urllib.request


def one(base, model, plen, mt, seed):
    rnd = random.Random(seed)
    words = " ".join(rnd.choice(("alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta")) + str(rnd.randint(0, 999))
                     for _ in range(plen // 3))
    body = dict(model=model, messages=[{"role": "user", "content": words}], max_tokens=mt, ignore_eos=True, stream=True,
                stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, n, usage = None, 0, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and ttft is None:
                    ttft = time.time() - t0
    t1 = time.time()
    out = (usage or {}).get("completion_tokens", 0)
    return dict(ttft=ttft or (t1 - t0), wall=t1 - t0, out=out, prompt=(usage or {}).get("prompt_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--conc", default="32")
    ap.add_argument("--prompt", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
    res = []
    for c in [int(x) for x in a.conc.split(",")]:
        stop = time.time() + a.seconds
        recs, lock = [], threading.Lock()

        def client(i):
            k = 0
            while time.time() < stop:
                r = one(base, model, a.prompt, a.max_tokens, seed=c * 100000 + i * 1000 + k)
                with lock:
                    recs.append(dict(r, t_end=time.time()))
                k += 1
        t0 = time.time()
        with cf.ThreadPoolExecutor(c) as ex:
            list(ex.map(client, range(c)))
        el = time.time() - t0
        out = sum(r["out"] for r in recs)
        dec = [(r["out"] - 1) / max(r["wall"] - r["ttft"], 1e-6) for r in recs if r["out"] > 1]
        res.append(dict(conc=c, calls=len(recs), wall_s=round(el, 1), out_tok_s=round(out / el, 1),
                        ttft_p50=round(statistics.median(r["ttft"] for r in recs), 3) if recs else None,
                        ttft_max=round(max(r["ttft"] for r in recs), 3) if recs else None,
                        decode_tok_s_per_req_p50=round(statistics.median(dec), 1) if dec else None,
                        prompt_tokens=recs[0]["prompt"] if recs else None))
        print(json.dumps(res[-1]), flush=True)


if __name__ == "__main__":
    main()
