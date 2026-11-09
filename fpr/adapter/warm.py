"""Warm the fork endpoint's prefill paths right after /health. The first mixed step of a server process costs
~2.0-2.4 s more than the cost model (step 0 of a 192-sequence replay; a 1-user bench's TTFT 2.1-2.4 s against
159-186 ms on its second run), and the first small mixed step ~0.15 s. The endpoint launcher sends these requests before
it reports "warm-up done", and its `start` waits for that line, so the first real request gets the warm path.

Requests (/v1/completions, synthetic token-id prompts, max_tokens 4, temperature 0, ignore_eos), one after another:
8,320 tokens (a CP chunk at cp_min 2,048), 320 tokens (a small mixed step), 24,000 tokens (a multi-chunk CP prompt),
then 4 x 1,200 tokens at once (a multi-sequence mixed step). Public mode: production's CA file and the first key of the
auth file, read at run time and never printed.
  python adapter/warm.py --url http://127.0.0.1:8100 [--cacert ca.pem --auth-file endpoint_auth.yaml]
"""
import argparse
import json
import ssl
import sys
import threading
import time
import urllib.request

SIZES = [8320, 320, 24000]
PAR = (4, 1200)


def prompt(n: int, salt: int) -> list:
    return [1000 + ((i + 1) * 7919 + salt * 104729) % 60000 for i in range(n)]


def post(url, body, headers, ctx, timeout=240) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        json.loads(r.read())
        return r.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="deepseek-v41-flash-h200")
    ap.add_argument("--cacert", default="")
    ap.add_argument("--auth-file", default="")
    a = ap.parse_args()
    headers = {"Content-Type": "application/json"}
    if a.auth_file:
        import yaml
        keys = yaml.safe_load(open(a.auth_file)).get("api-key") or []
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            print("warm: no api-key in the auth file", file=sys.stderr)
            return 2
        headers["Authorization"] = "Bearer " + str(keys[0])
    ctx = ssl.create_default_context(cafile=a.cacert) if a.cacert else None
    url = a.url.rstrip("/") + "/v1/completions"
    salt = int(time.time()) % 1000                     # a fresh prompt per start (no prefix hit from a previous one)
    out = []
    for k, n in enumerate(SIZES):
        t0 = time.time()
        st = post(url, dict(model=a.model, prompt=prompt(n, salt + k), max_tokens=4, temperature=0.0, ignore_eos=True),
                  headers, ctx)
        out.append(f"{n}:{time.time() - t0:.2f}s/{st}")
    errs = []

    def one(j):
        try:
            post(url, dict(model=a.model, prompt=prompt(PAR[1], salt + 10 + j), max_tokens=4, temperature=0.0,
                           ignore_eos=True), headers, ctx)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e)[:120])
    t0 = time.time()
    th = [threading.Thread(target=one, args=(j,)) for j in range(PAR[0])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    out.append(f"{PAR[0]}x{PAR[1]}:{time.time() - t0:.2f}s/{'ok' if not errs else errs[0]}")
    print("warm: " + " ".join(out))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
