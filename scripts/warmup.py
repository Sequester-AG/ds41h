"""Warm a fresh server before the BRRR bench (run with the bench venv python):

  python scripts/warmup.py [--port 8100] [--sizes 16384 2048]

One max_tokens=1 chat request per size with a fresh calibrated code prompt (unique nonce, no prefix-cache hit)
plus a short decode, so per-shape JIT / first-use costs (Humming / DeepGEMM / Triton configs of the chunk
remainders, allocator growth) are not charged to the bench's first measured round.
"""

import argparse
import asyncio
import time

import httpx
from brrr.bench import calibrated_prompt


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16384, 2048])
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=f"http://localhost:{a.port}", timeout=None) as http:
        model = (await http.get("/v1/models")).json()["data"][0]["id"]
        for size in a.sizes:
            for max_tokens in (1, 64):
                prompt = await calibrated_prompt(http, model, size)
                t = time.perf_counter()
                r = await http.post("/v1/chat/completions", json={
                    "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                    "temperature": 1.0, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
                u = r.json().get("usage", {})
                print(f"warmup size {size} max_tokens {max_tokens}: {u.get('prompt_tokens')} prompt tokens, "
                      f"{time.perf_counter() - t:.2f} s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
