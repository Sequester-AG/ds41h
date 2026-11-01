"""GSM8K accuracy of the server on port 8100 (exactness gate).

  python scripts/gsm8k_eval.py download                      # openai/gsm8k test split -> cache (HF datasets-server API)
  python scripts/gsm8k_eval.py run OUT.json [--n 500] [--concurrency 1] [--max-tokens 512] [--port 8100]
  python scripts/gsm8k_eval.py compare A.json B.json [C.json ...]   # accuracy + 95% CI, paired disagreements vs A

Data: the 1319-question test split of openai/gsm8k ("main" config), fetched as JSON rows from
https://datasets-server.huggingface.co/rows (100 rows per call) into $HF_HOME/gsm8k/test.jsonl.
`run` uses the first --n questions (default 500) in split order.

Request: chat completion, one user message = question + an instruction to end with "#### <number>", chat template
kwargs thinking=False, temperature 0 (greedy), max_tokens 512, no prefix-cache sharing between questions
(cache_salt). Default concurrency 1: on a k=5 DSpark server every decode step then has M = 6 rows, i.e. the ds41h
decode megakernel path (M <= 8); higher concurrency exercises the M > 8 stock-layer path instead.

Answer extraction (robust, in order): last "#### x"; last \\boxed{x}; last "answer is x" / "Answer: x"; last number
in the text. Numbers are normalized (commas, $, %, units, trailing dots removed; fractions / decimals compared
numerically with 1e-6 tolerance).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid

CACHE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "gsm8k", "test.jsonl")
API = "https://datasets-server.huggingface.co/rows"
INSTRUCTION = (
    "Solve the following math problem step by step. At the end, write the final answer as a single number on its own "
    "line in the form \"#### <number>\".\n\n"
)


# ----------------------------------------------------------------------------------------------- data
def download(path: str = CACHE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({"dataset": "openai/gsm8k", "config": "main", "split": "test", "offset": offset, "length": 100})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                    data = json.loads(r.read())
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"retry offset {offset}: {e!r}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
        total = int(data["num_rows_total"])
        got = [{"question": str(x["row"]["question"]), "answer": str(x["row"]["answer"])} for x in data["rows"]]
        if not got:
            break
        rows += got
        offset += len(got)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"idx": i, **r}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path}: {len(rows)} rows (total {total})")


def load(path: str = CACHE) -> list[dict]:
    if not os.path.exists(path):
        download(path)
    return [json.loads(line) for line in open(path)]


def gold_answer(answer: str) -> str:
    return normalize(answer.split("####")[-1])


# ----------------------------------------------------------------------------------------------- extraction
NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?"


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace("\\$", "").replace("$", "").replace(",", "").replace("\\!", "").replace("\\%", "").replace("%", "")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?", s)
    if not m:
        return None
    v = m[0].replace(" ", "").rstrip("/")
    return v


def to_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        if "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b) if float(b) != 0 else None
        return float(v)
    except ValueError:
        return None


def extract(text: str) -> str | None:
    t = text.replace("**", "")
    m = re.findall(r"####\s*([^\n]*)", t)
    for cand in reversed(m):
        v = normalize(cand)
        if v is not None:
            return v
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)
    for cand in reversed(m):
        v = normalize(cand)
        if v is not None:
            return v
    m = re.findall(r"(?:answer is|answer:|Answer:|ANSWER:)\s*([^\n]*)", t, flags=re.IGNORECASE)
    for cand in reversed(m):
        v = normalize(cand)
        if v is not None:
            return v
    m = re.findall(NUM, t)
    return normalize(m[-1]) if m else None


def correct(pred: str | None, gold: str | None) -> bool:
    a, b = to_float(pred), to_float(gold)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


# ----------------------------------------------------------------------------------------------- run
def _post(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(out: str, n: int = 500, concurrency: int = 1, max_tokens: int = 512, port: int = 8100) -> None:
    base = f"http://127.0.0.1:{port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
    rows = load()[:n]

    def one(r):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": INSTRUCTION + r["question"]}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
            "cache_salt": uuid.uuid4().hex,
        }
        t = time.time()
        res = _post(base + "/v1/chat/completions", body)
        ch = res["choices"][0]
        text = ch["message"].get("content") or ""
        pred, gold = extract(text), gold_answer(r["answer"])
        return {"idx": r["idx"], "pred": pred, "gold": gold, "correct": correct(pred, gold), "finish": ch.get("finish_reason"),
                "tokens": res.get("usage", {}).get("completion_tokens"), "seconds": round(time.time() - t, 3), "text": text}

    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for i, r in enumerate(ex.map(one, rows)):
            results.append(r)
            if (i + 1) % 50 == 0:
                acc = sum(x["correct"] for x in results) / len(results)
                print(f"{i + 1}/{len(rows)} acc {acc:.4f} ({time.time() - t0:.0f} s)", flush=True)
    acc = sum(x["correct"] for x in results) / max(1, len(results))
    summary = {"n": len(results), "correct": sum(x["correct"] for x in results), "accuracy": acc, "concurrency": concurrency,
               "max_tokens": max_tokens, "seconds": round(time.time() - t0, 1),
               "truncated": sum(x["finish"] == "length" for x in results), "no_answer": sum(x["pred"] is None for x in results),
               "completion_tokens": sum(x["tokens"] or 0 for x in results)}
    json.dump({"summary": summary, "results": results}, open(out, "w"), ensure_ascii=False)
    lo, hi = wilson(summary["correct"], summary["n"])
    print(f"GSM8K n={summary['n']} accuracy {acc:.4f} (95% CI {lo:.4f}-{hi:.4f}) truncated {summary['truncated']} "
          f"no_answer {summary['no_answer']} in {summary['seconds']} s -> {out}")


# ----------------------------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (A right, B wrong) vs c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def compare(paths: list[str]) -> None:
    runs = [(p, json.load(open(p))) for p in paths]
    ref_path, ref = runs[0]
    ref_by = {r["idx"]: r for r in ref["results"]}
    for p, d in runs:
        s = d["summary"]
        lo, hi = wilson(s["correct"], s["n"])
        line = f"{os.path.basename(os.path.dirname(p)) or p}: acc {s['accuracy']:.4f} [{lo:.4f}, {hi:.4f}] n={s['n']} trunc {s['truncated']} noans {s['no_answer']}"
        if p != ref_path:
            by = {r["idx"]: r for r in d["results"]}
            common = sorted(set(by) & set(ref_by))
            b = sum(ref_by[i]["correct"] and not by[i]["correct"] for i in common)
            c = sum(by[i]["correct"] and not ref_by[i]["correct"] for i in common)
            same_text = sum(ref_by[i]["text"] == by[i]["text"] for i in common)
            same_pred = sum(ref_by[i]["pred"] == by[i]["pred"] for i in common)
            line += (f" | vs ref: ref-only-correct {b}, this-only-correct {c} (McNemar p={mcnemar_exact(b, c):.3f}), "
                     f"same answer {same_pred}/{len(common)}, identical text {same_text}/{len(common)}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--port", type=int, default=8100)
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    t = sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "download":
        download()
    elif a.cmd == "run":
        run(a.out, a.n, a.concurrency, a.max_tokens, a.port)
    elif a.cmd == "compare":
        compare(a.paths)
    elif a.cmd == "selftest":
        cases = [("The answer is 42.", "42"), ("#### 1,234", "1234"), ("so $18 per day.\n#### $18", "18"), ("\\boxed{3/4}", "3/4"),
                 ("Total: 7.50 dollars", "7.50"), ("**#### 70000**", "70000"), ("Answer: -5", "-5"), ("x = 12 apples and 3 pears", "3")]
        for text, want in cases:
            got = extract(text)
            assert correct(got, want), (text, got, want)
        assert correct(gold_answer("blah\n#### 1,000"), "1000")
        print("selftest ok")


if __name__ == "__main__":
    main()
