"""Profile one 16K prefill and ~64 decode steps on our server (port 8100, started with PROFILE=1)."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8100"


def post(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


model = json.loads(urllib.request.urlopen(URL + "/v1/models").read())["data"][0]["id"]
words = ("def handler(request): return process(request.json()) " * 4000).split()
prompt16k = " ".join(words[:11000]) + f" nonce {time.time()}"
post("/start_profile")
t = time.time()
post("/v1/completions", {"model": model, "prompt": prompt16k, "max_tokens": 1})
print("prefill_s", time.time() - t, flush=True)
t = time.time()
out = json.loads(post("/v1/completions", {"model": model, "prompt": "Write a Python function that parses a CSV file and computes column statistics.", "max_tokens": 192, "temperature": 0}))
print("decode_s", time.time() - t, out["usage"], flush=True)
post("/stop_profile")
print("done", flush=True)
