"""Decode-path correctness A/B: greedy generations with top-20 logprobs from our server (port 8100).

  python kl_check.py collect OUT.json [--n 12] [--tokens 256]   # query the running server
  python kl_check.py compare STOCK.json DS41H.json              # report agreement / KL

Prompts mix code, prose and math. Generations are greedy (temperature 0), so the two runs follow the
same path until the first argmax flip; metrics are computed on the shared prefix:
  top1      fraction of positions where both runs pick the same token (until first divergence)
  kl20      mean KL(stock || ds41h) over the union of both top-20 sets (renormalized), nats
  diverge   position of first differing token (or length if identical)
"""

import json
import math
import sys
import urllib.request

URL = "http://127.0.0.1:8100"
PROMPTS = [
    "Write a Python function that parses a CSV file and returns per-column mean and standard deviation.",
    "Explain the difference between TCP and UDP to a high-school student.",
    "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
    "Prove that the square root of 2 is irrational.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a SQL query that returns the top 3 customers by total order value in 2023.",
    "Translate to French: The weather is lovely today, let's go for a walk by the river.",
    "Implement a thread-safe LRU cache in C++ with get and put in O(1).",
    "What is 17 * 23? Show your work step by step.",
    "Write a haiku about GPUs running out of power budget.",
    "Describe how a transformer attention layer works, with equations.",
    "Write a bash script that finds the 10 largest files under a directory.",
]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def collect(out, n=12, tokens=256):
    model = json.loads(urllib.request.urlopen(URL + "/v1/models").read())["data"][0]["id"]
    res = []
    for p in PROMPTS[:n]:
        r = post("/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": p}], "max_tokens": tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"thinking": False},
        })
        steps = r["choices"][0]["logprobs"]["content"]
        res.append([{"tok": s["token"], "lp": s["logprob"], "top": {t["token"]: t["logprob"] for t in s["top_logprobs"]}} for s in steps])
    json.dump(res, open(out, "w"))
    print(f"wrote {out}: {len(res)} prompts, {sum(map(len, res))} tokens")


def kl_union(a, b):
    keys = set(a) | set(b)
    floor = min(min(a.values()), min(b.values())) - 1.0  # unseen tokens get a small mass
    pa = {k: math.exp(a.get(k, floor)) for k in keys}
    pb = {k: math.exp(b.get(k, floor)) for k in keys}
    za, zb = sum(pa.values()), sum(pb.values())
    return sum(pa[k] / za * (math.log(pa[k] / za) - math.log(pb[k] / zb)) for k in keys)


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    agree = tot = 0
    kls, divs = [], []
    for sa, sb in zip(A, B):
        d = min(len(sa), len(sb))
        for i, (x, y) in enumerate(zip(sa, sb)):
            kls.append(kl_union(x["top"], y["top"]))
            tot += 1
            if x["tok"] != y["tok"]:
                d = i
                break
            agree += 1
        divs.append(d)
    print(f"positions compared {tot}, top1 agreement {agree / max(tot, 1):.4f}, mean kl20 {sum(kls) / max(len(kls), 1):.5f}, "
          f"max kl20 {max(kls) if kls else 0:.4f}")
    print("first divergence per prompt:", divs)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        args = sys.argv[2:]
        n = int(args[args.index("--n") + 1]) if "--n" in args else 12
        t = int(args[args.index("--tokens") + 1]) if "--tokens" in args else 256
        collect(args[0], n, t)
    else:
        compare(sys.argv[2], sys.argv[3])
