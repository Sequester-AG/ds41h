"""Many users decoding at a long context, one shared prefix (why was an ASYNC decode step ~5 ms slower than a sync one
under a 192-user long-context load, while it is 8 ms faster on the README bench's 8K contexts?).

Per phase (--ctx, e.g. 8192,65536): the prompt = the first CTX token ids of a corpus (fpr/docs/*.md + fpr/engine/*.py,
tokenized with the model's tokenizer; synthetic, no traffic data) + a per-user tail of --tail ids from a user-specific
corpus offset. One warm request caches the prefix, then --users streams (/v1/completions, token-id prompts,
ignore_eos, temperature 0, max_tokens --max-tokens) run concurrently. When every stream has its first token, wait
--settle s, then write <--go> with --prof-steps (adapter/steptimes.py's profiler trigger) and measure tokens in a
--window s window.
Output: one JSON line per phase (aggregate decode tok/s in the window, first-token spread, errors). Counts only.
From fpr/:
  python adapter/tests/lc_client.py --port 8100 --ctx 8192,65536 --users 96 --max-tokens 700 --go <prof_dir>/go
"""
import argparse
import asyncio
import glob
import json
import os
import time

import aiohttp


def corpus_ids(model_dir: str, need: int) -> list:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = ""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in sorted(glob.glob(os.path.join(root, "docs", "*.md"))) + sorted(glob.glob(os.path.join(root, "engine", "*.py"))):
        text += open(p, errors="ignore").read() + "\n"
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < need:
        ids = ids + ids
    return ids


async def one(sess, url, prompt, max_tokens, st):
    body = dict(model=st["model"], prompt=prompt, max_tokens=max_tokens, temperature=0.0, ignore_eos=True, stream=True)
    n, t_first = 0, None
    try:
        async with sess.post(url, json=body) as r:
            if r.status != 200:
                st["errors"] += 1
                return
            async for raw in r.content:
                ln = raw.decode(errors="ignore").strip()
                if not ln.startswith("data:") or ln == "data: [DONE]":
                    continue
                d = json.loads(ln[5:])
                ch = d.get("choices") or []
                if not ch:
                    continue
                k = len((ch[0].get("text") or "")) > 0 or ch[0].get("finish_reason") is None
                if t_first is None:
                    t_first = time.time()
                    st["first"].append(t_first)
                n += 1
                st["chunks"].append(time.time())
                _ = k
    except Exception:  # noqa: BLE001
        st["errors"] += 1


async def phase(a, ids, ctx):
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    st = dict(model=a.model, errors=0, first=[], chunks=[])
    to = aiohttp.ClientTimeout(total=1800)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=to, connector=conn) as sess:
        pre = ids[:ctx]
        t0 = time.time()
        await one(sess, url, pre + ids[ctx:ctx + 8], 1, dict(st, first=[], chunks=[]))
        warm = time.time() - t0
        prompts = [pre + ids[ctx + 1000 * (u + 1):ctx + 1000 * (u + 1) + a.tail] for u in range(a.users)]
        tasks = [asyncio.create_task(one(sess, url, p, a.max_tokens, st)) for p in prompts]
        t1 = time.time()
        while len(st["first"]) < a.users and any(not t.done() for t in tasks) and time.time() - t1 < 900:
            await asyncio.sleep(0.05)
        t_all = time.time()
        await asyncio.sleep(a.settle)
        if a.go:
            with open(a.go + ".tmp", "w") as f:
                f.write(str(a.prof_steps))
            os.rename(a.go + ".tmp", a.go)
        w0 = time.time() + 1.0                    # (the profiler starts within 8 steps)
        await asyncio.sleep(1.0 + a.window)
        w1 = w0 + a.window
        n_win = sum(1 for t in st["chunks"] if w0 <= t < w1)
        await asyncio.gather(*tasks)
        t_end = time.time()
    fs = sorted(st["first"])
    return dict(ctx=ctx, users=a.users, warm_s=round(warm, 2), first_token_spread_s=round(fs[-1] - fs[0], 2) if fs else None,
                all_first_s=round(t_all - t1, 2), window_s=a.window, window_chunks=n_win,
                window_chunk_rate=round(n_win / a.window, 1), total_s=round(t_end - t1, 2), errors=st["errors"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="deepseek-v41-flash-h200")
    ap.add_argument("--model-dir", default=os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
    ap.add_argument("--ctx", default="8192,65536")
    ap.add_argument("--users", type=int, default=96)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--go", default="")
    ap.add_argument("--prof-steps", type=int, default=30)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctx.split(",")]
    ids = corpus_ids(a.model_dir, max(ctxs) + 1000 * (a.users + 2) + a.tail)
    for c in ctxs:
        print(json.dumps(asyncio.run(phase(a, ids, c))), flush=True)


if __name__ == "__main__":
    main()
