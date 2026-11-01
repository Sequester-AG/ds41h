"""Exactness harness: teacher-forced full-vocab logit comparison through the REAL decode/verify path.

The server (port 8100) must run with DS41H=1 DS41H_FORCE_FILE=<file> DS41H_FORCE_DUMP=<dir> (ds41h/patch_force.py);
a stock-numerics server is DS41H=1 with every compute component off (scripts/exact_run.sh STOCK=1 sets them).

  python scripts/exact_eval.py gen SET.json [--k 5]                  # build prompts + greedy continuations (once)
  python scripts/exact_eval.py collect SET.json --force-file F --dump DIR   # teacher-forced run, dumps DIR/<tag>/
  python scripts/exact_eval.py apilp SET.json OUT.json               # normal-path greedy top-20 logprobs (hook check)
  python scripts/exact_eval.py hookcheck SET.json OUT.json DIR       # forced dump vs normal-path logprobs, same server
  python scripts/exact_eval.py compare REF_DIR TEST_DIR [--set SET.json] [--json OUT]   # KL / top-1 / |dlogp|

Prompt set (gen): short chat prompts (code, prose, math, German, Chinese, Japanese; 192 forced positions each)
and ~15K-token contexts (raw source-code continuation of real vLLM files, a chat question over ~15K tokens of
license texts, a 15K-token few-shot GSM8K chat prompt; 256 forced positions each, i.e. decode positions 15-16K
deep). Continuations: the serving model's own greedy output (chat prompts) or the document's true next tokens
(raw code). Every later run (stock or ds41h) is forced along the SAME token ids, so the logits of every position
are directly comparable. Forced runs send one request at a time with a fresh cache_salt (no prefix-cache reuse).

compare: per position p_ref = softmax(ref), p_test = softmax(test) over the full vocabulary (float64):
  KL(ref||test), top-1 agreement (argmax), |delta logprob| of the forced token and max over ref's top-20 tokens,
  reported per prompt, for short / long-context prompts and overall (decode rows; the single prefill row of each
  prompt is reported separately).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8100"
SHORT_D, LONG_D = 192, 256
MARGIN = 16  # forced tokens beyond n_dump (k + 1 bonus + next drafts) so the forcing never runs dry
LONG_TOKENS = 15000

SHORT = [
    ("s_code", "Write a Python module that implements a binary min-heap class with push, pop, peek and heapify methods. "
               "Include type hints, docstrings and a small unittest test suite at the end."),
    ("s_prose", "Write a detailed essay of at least 600 words about the invention of the printing press and how it changed "
                "European society, religion and science."),
    ("s_math", "A rectangular garden is 3 meters longer than it is wide. A path of uniform width 1 meter surrounds the garden, "
               "and the area of the path alone is 54 square meters. Find the dimensions of the garden. Explain every step of "
               "your reasoning carefully, then verify the result."),
    ("s_de", "Erkläre ausführlich auf Deutsch, wie ein Kühlschrank funktioniert. Gehe dabei auf den Kältekreislauf, den "
             "Kompressor, das Kältemittel und die Energieeffizienz ein und gib praktische Tipps zum Stromsparen."),
    ("s_zh", "请用中文详细介绍中国长城的历史，包括它的修建原因、主要朝代的贡献、建筑技术以及它在今天的文化意义。请至少写五段。"),
    ("s_ja", "日本の四季について、それぞれの季節の気候、行事、食べ物の特徴を詳しく説明してください。各季節について段落を分けて書いてください。"),
]


def post(path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id() -> str:
    return json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]


def tok_chat(model: str, text: str) -> list[int]:
    r = post("/tokenize", {"model": model, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True,
                           "chat_template_kwargs": {"thinking": False}})
    return list(r["tokens"])


def tok_raw(model: str, text: str, special: bool) -> list[int]:
    return list(post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": special})["tokens"])


def complete(model: str, ids: list[int], max_tokens: int, *, ignore_eos: bool, logprobs: int | None = None) -> dict:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": ignore_eos,
            "return_token_ids": True, "cache_salt": uuid.uuid4().hex, "skip_special_tokens": False}
    if logprobs:
        body.update(logprobs=logprobs, return_tokens_as_token_ids=True)
    return post("/v1/completions", body)["choices"][0]


# ----------------------------------------------------------------------------------------------- gen
def _long_texts() -> dict[str, str]:
    import vllm  # noqa: F401 - source files of the installed vLLM are the code corpus

    root = os.path.dirname(vllm.__file__)
    code = ""
    for rel in ("v1/worker/gpu/model_runner.py", "v1/worker/gpu/input_batch.py", "v1/worker/gpu/sample/sampler.py"):
        code += f"# ===== file: vllm/{rel}\n" + open(os.path.join(root, rel)).read() + "\n"
    lic = ""
    for name in ("GPL-3", "GFDL-1.3", "Apache-2.0", "MPL-2.0", "LGPL-3", "Artistic"):
        lic += f"===== {name} =====\n" + open(f"/usr/share/common-licenses/{name}").read() + "\n"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gsm8k_eval

    rows = gsm8k_eval.load()
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows[600:])  # never in the 500 eval questions
    return {"code": code, "lic": lic, "shots": shots, "final_q": rows[599]["question"]}


def _trim_to_tokens(model: str, text: str, n: int) -> str:
    """Cut `text` (at a line boundary) so that it tokenizes to about n tokens."""
    ids = tok_raw(model, text, False)
    if len(ids) <= n:
        return text
    frac = n / len(ids)
    cut = text[: int(len(text) * frac)]
    return cut[: cut.rfind("\n") + 1] or cut


def gen(path: str, k: int) -> None:
    model = model_id()
    items = []
    for tag, text in SHORT:
        items.append({"tag": tag, "kind": "short", "prompt_ids": tok_chat(model, text), "n_dump": SHORT_D})
    lt = _long_texts()
    code_ids = tok_raw(model, lt["code"], True)
    items.append({"tag": "l_code_raw", "kind": "long", "prompt_ids": code_ids[:LONG_TOKENS],
                  "forced": code_ids[LONG_TOKENS : LONG_TOKENS + LONG_D + MARGIN], "n_dump": LONG_D})
    lic = _trim_to_tokens(model, lt["lic"], LONG_TOKENS - 200)
    items.append({"tag": "l_lic_chat", "kind": "long", "n_dump": LONG_D, "prompt_ids": tok_chat(
        model, lic + "\n\nCompare the licenses above in detail: for each one, explain what it requires when you "
        "redistribute modified versions, how it treats patents, and whether it is compatible with the GPL-3.0.")})
    shots = _trim_to_tokens(model, lt["shots"], LONG_TOKENS - 300)
    items.append({"tag": "l_math_fewshot", "kind": "long", "n_dump": LONG_D, "prompt_ids": tok_chat(
        model, "Here are solved example problems.\n\n" + shots + "Now solve this new problem in the same style, step by step, "
        "and end with '#### <number>'.\nQuestion: " + lt["final_q"])})
    for it in items:
        if "forced" in it:
            continue
        need = it["n_dump"] + MARGIN
        ch = complete(model, it["prompt_ids"], need, ignore_eos=False)
        out = list(ch["token_ids"])
        if ch.get("finish_reason") == "stop" and out:
            out = out[:-1]  # drop the stop token: forcing it would end the request
        it["forced"] = out
        if len(out) < need:
            it["n_dump"] = max(0, len(out) - MARGIN)
            print(f"{it['tag']}: only {len(out)} tokens before EOS -> n_dump {it['n_dump']}")
        print(f"{it['tag']}: prompt {len(it['prompt_ids'])} tokens, forced {len(it['forced'])}, dump {it['n_dump']}: "
              f"{ch['text'][:80]!r}", flush=True)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "server_k": k, "positions": sum(i["n_dump"] for i in items)}
    json.dump({"meta": meta, "items": items}, open(path, "w"))
    print(f"wrote {path}: {len(items)} prompts, {meta['positions']} forced positions")


# ----------------------------------------------------------------------------------------------- collect
def _write_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- background load
# collect --bg N: N background streaming completions keep running while the forced prompts are collected, so every decode step of
# the forced request runs in a multi-request batch (N + 1 requests x (1 + k) rows > 8: the vLLM layer path with the routed MoE
# small-T / moe_dec kernels and, when enabled, the grouped multi-row decode attention) instead of the c1 decoder megakernel. The
# forced request is still identified by its prompt ids (ds41h/patch_force.py leaves every other request untouched).
BG_TOPICS = (
    "the history of the printing press", "how a steam engine works", "the water cycle", "medieval castle architecture",
    "the life of honey bees", "how compilers optimize loops", "the economics of shipping containers", "volcanoes and plate tectonics",
    "the invention of the telephone", "how vaccines train the immune system", "the history of chess", "how bridges carry loads",
    "the Silk Road trade network", "how solar panels make electricity", "the migration of birds", "the history of cartography",
    "how databases keep data consistent", "the Roman road system", "how coffee is grown and roasted", "the physics of sailing",
    "the history of clocks", "how the human heart works", "the rise of railways", "how glaciers shape valleys",
    "the history of mathematics in India", "how radio waves carry sound", "the ecology of coral reefs", "how paper is made",
    "the history of the bicycle", "how an operating system schedules tasks", "the origins of jazz", "how wind turbines are built",
)


def running_requests() -> int:
    """vllm:num_requests_running summed over the /metrics series (-1 if unavailable)."""
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return -1
    return int(sum(float(ln.rsplit(" ", 1)[1]) for ln in txt.splitlines() if ln.startswith("vllm:num_requests_running")))


class BgLoad:
    """N background streams (distinct prompts, ignore_eos, restarted when they finish; closing the connection aborts them)."""

    def __init__(self, model: str, n: int, tokens: int = 2048, temperature: float = 1.0) -> None:
        import threading

        self.model, self.n, self.tokens, self.temperature = model, n, tokens, temperature
        self.stop = threading.Event()
        self.first = [threading.Event() for _ in range(n)]
        self.threads = [threading.Thread(target=self._run, args=(i,), daemon=True) for i in range(n)]
        self.requests = self.errors = 0

    def _run(self, i: int) -> None:
        import http.client

        host, port = BASE.split("//", 1)[1].rsplit(":", 1)
        prompt = (f"Background stream {i}. Write a long, detailed essay about {BG_TOPICS[i % len(BG_TOPICS)]}, with many sections, "
                  "concrete examples, numbers and historical details.\n\n")
        while not self.stop.is_set():
            conn = None
            try:
                conn = http.client.HTTPConnection(host, int(port), timeout=900)
                body = {"model": self.model, "prompt": prompt, "max_tokens": self.tokens, "temperature": self.temperature,
                        "ignore_eos": True, "stream": True}
                conn.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    self.errors += 1
                    time.sleep(1.0)
                    continue
                self.requests += 1
                while not self.stop.is_set():
                    line = r.readline()
                    if not line:
                        break
                    if line.startswith(b"data: {"):
                        self.first[i].set()
            except Exception:  # noqa: BLE001
                self.errors += 1
                time.sleep(0.5)
            finally:
                if conn is not None:
                    conn.close()

    def __enter__(self) -> "BgLoad":
        t0 = time.time()
        for t in self.threads:
            t.start()
        for e in self.first:
            e.wait(max(1.0, 300.0 - (time.time() - t0)))
        print(f"background load: {sum(e.is_set() for e in self.first)}/{self.n} streams decoding after {time.time() - t0:.1f} s, "
              f"server running {running_requests()}", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=60)
        print(f"background load stopped: {self.requests} requests, {self.errors} errors", flush=True)


def collect(set_path: str, force_file: str, dump: str, only: list[str] | None = None, timeout: float = 180, bg: int = 0,
            bg_tokens: int = 2048) -> int:
    import contextlib

    model = model_id()
    with BgLoad(model, bg, bg_tokens) if bg > 0 else contextlib.nullcontext():
        return _collect(model, set_path, force_file, dump, only, timeout, bg)


def _collect(model: str, set_path: str, force_file: str, dump: str, only: list[str] | None, timeout: float, bg: int) -> int:
    data = json.load(open(set_path))
    bad = 0
    for it in data["items"]:
        if only and it["tag"] not in only:
            continue
        d = os.path.join(dump, it["tag"])
        for f in glob.glob(os.path.join(d, "rank*")):
            os.remove(f)
        _write_atomic(force_file, {"tag": it["tag"], "prompt_ids": it["prompt_ids"], "forced": it["forced"], "n_dump": it["n_dump"],
                                   "dump_dir": os.path.abspath(dump)})
        time.sleep(0.05)
        run0 = running_requests() if bg else None  # requests in the batch besides the forced one (--bg)
        t = time.time()
        ch = complete(model, it["prompt_ids"], it["n_dump"], ignore_eos=True)
        got = list(ch["token_ids"])
        ok_tokens = got == it["forced"][: len(got)] and len(got) == it["n_dump"]
        # wait for every rank's marker (rank 0 says how many ranks there are)
        metas = {}
        while time.time() - t < timeout:
            m0 = os.path.join(d, "rank0.json")
            if os.path.exists(m0):
                tp = json.load(open(m0))["tp"]
                files = [os.path.join(d, f"rank{r}.json") for r in range(tp)]
                if all(os.path.exists(f) for f in files):
                    metas = {r: json.load(open(f)) for r, f in enumerate(files)}
                    break
            time.sleep(0.2)
        m = metas.get(0, {})
        good = ok_tokens and m and not m.get("partial") and m.get("input_mismatch", 1) == 0 and m.get("n") == it["n_dump"]
        bad += not good
        steps = m.get("rows_per_step", [])
        hist = {}
        for x in steps:
            hist[x] = hist.get(x, 0) + 1
        print(f"{it['tag']}: {'OK ' if good else 'BAD'} tokens_match={ok_tokens} rows={m.get('n')}/{it['n_dump']} kind={m.get('kind')} "
              f"input_mismatch={m.get('input_mismatch')}/{m.get('checked_rows')} steps={hist} saved="
              f"{[metas[r].get('saved') for r in sorted(metas)]} {time.time() - t:.1f} s"
              + (f" bg_running_before={run0}" if bg else ""), flush=True)
    try:
        os.remove(force_file)
    except OSError:
        pass
    print(f"collect done: {bad} bad")
    return bad


# ----------------------------------------------------------------------------------------------- load / compare
def load_dump(d: str):
    """-> (logits float32 [n, V], idx [n]) assembled from rank files (full: rank 0; local: all shards)."""
    import torch

    m0 = json.load(open(os.path.join(d, "rank0.json")))
    if m0["kind"] == "full":
        x = torch.load(os.path.join(d, "rank0.pt"), map_location="cpu")
        return x["logits"].float(), x["idx"], m0
    parts = []
    idx = None
    for r in range(m0["tp"]):
        x = torch.load(os.path.join(d, f"rank{r}.pt"), map_location="cpu")
        parts.append((x["vocab_start"], x["logits"].float()))
        idx = x["idx"] if idx is None else idx
        assert torch.equal(idx, x["idx"]), f"{d}: rank {r} rows differ"
    parts.sort(key=lambda p: p[0])
    return torch.cat([p[1] for p in parts], dim=1), idx, m0


def _stats(vals: list[float]) -> dict:
    import statistics

    if not vals:
        return {}
    s = sorted(vals)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"n": len(s), "mean": statistics.fmean(s), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


def table(ref: str, tests: list[str], set_path: str | None) -> None:
    """One summary row per test dump dir vs ref (decode rows; long = ~15K-token contexts; prefill = last prompt row)."""
    import contextlib
    import io

    print(f"{'run':28s} {'n':>5s} {'meanKL':>9s} {'p99KL':>9s} {'maxKL':>9s} {'top1':>7s} {'flips':>5s} {'mx|dlp|tok':>10s} "
          f"{'mx|dlp|t20':>10s} {'dNLL':>9s} | {'longKL':>9s} {'longTop1':>8s} | {'prefKL':>9s}")
    for t in tests:
        with contextlib.redirect_stdout(io.StringIO()):
            r = compare_dirs(ref, t, set_path, None)
        a, lg, pf = r["all"], r["long"], r["prefill"]
        name = t.rstrip("/").split("/")[-2] if t.rstrip("/").endswith("dump") or t.rstrip("/").endswith("dump2") else t
        if t.rstrip("/").endswith("dump2"):
            name += "/2"
        print(f"{name:28s} {a['n']:5d} {a['kl_mean']:9.2e} {a['kl_p99']:9.2e} {a['kl_max']:9.2e} {a['top1']:7.4f} {a['top1_flips']:5d} "
              f"{a['dtok_max']:10.4f} {a['d20_max']:10.4f} {a['dnll_mean']:+9.5f} | {lg['kl_mean']:9.2e} {lg['top1']:8.4f} | "
              f"{pf.get('kl_mean', float('nan')):9.2e}")


def compare_dirs(ref: str, test: str, set_path: str | None, out_json: str | None) -> dict:
    import torch

    forced = {}
    kinds = {}
    if set_path:
        for it in json.load(open(set_path))["items"]:
            forced[it["tag"]] = it["forced"]
            kinds[it["tag"]] = it["kind"]
    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ref, "*")) if os.path.exists(os.path.join(p, "rank0.json")))
    rows = {}
    agg = {"all": [], "short": [], "long": [], "prefill": []}
    for tag in tags:
        if not os.path.exists(os.path.join(test, tag, "rank0.json")):
            print(f"{tag}: missing in {test}")
            continue
        a, ia, ma = load_dump(os.path.join(ref, tag))
        b, ib, mb = load_dump(os.path.join(test, tag))
        n = min(a.shape[0], b.shape[0])
        v = min(a.shape[1], b.shape[1])
        assert torch.equal(ia[:n], ib[:n]), f"{tag}: forced indices differ"
        a64, b64 = a[:n, :v].double(), b[:n, :v].double()
        la, lb = torch.log_softmax(a64, -1), torch.log_softmax(b64, -1)
        kl = (la.exp() * (la - lb)).sum(-1)
        top1 = (a64.argmax(-1) == b64.argmax(-1))
        top20 = la.topk(20, dim=-1).indices
        d20 = (la.gather(1, top20) - lb.gather(1, top20)).abs().max(-1).values
        if tag in forced:
            tok = torch.tensor([forced[tag][int(i)] for i in ia[:n]], dtype=torch.int64)
            dtok = (la.gather(1, tok[:, None]) - lb.gather(1, tok[:, None])).abs()[:, 0]
            nll_a, nll_b = -la.gather(1, tok[:, None])[:, 0], -lb.gather(1, tok[:, None])[:, 0]
        else:
            dtok = nll_a = nll_b = torch.zeros(n, dtype=torch.float64)
        kind = kinds.get(tag, "long" if tag.startswith("l_") else "short")
        per = []
        for j in range(n):
            rec = {"kl": float(kl[j]), "top1": bool(top1[j]), "dtok": float(dtok[j]), "d20": float(d20[j]), "nll_a": float(nll_a[j]),
                   "nll_b": float(nll_b[j])}
            per.append(rec)
            if int(ia[j]) == 0:
                agg["prefill"].append(rec)
            else:
                agg["all"].append(rec)
                agg[kind].append(rec)
        dec = [r for r, i in zip(per, ia[:n].tolist()) if i > 0]
        rows[tag] = summarize(dec)
        rows[tag].update(kind=kind, ref_kind=ma["kind"], test_kind=mb["kind"], prefill_kl=per[0]["kl"] if int(ia[0]) == 0 else None)
    res = {"ref": ref, "test": test, "per_prompt": rows, **{k: summarize(v) for k, v in agg.items()}}
    print(f"compare {ref} -> {test}")
    print(f"{'prompt':16s} {'n':>5s} {'meanKL':>10s} {'p99KL':>10s} {'maxKL':>10s} {'top1':>7s} {'max|dlp|tok':>11s} {'max|dlp|t20':>11s} "
          f"{'dNLL':>9s}")
    for tag, r in list(rows.items()) + [(k, res[k]) for k in ("short", "long", "all", "prefill")]:
        if not r.get("n"):
            continue
        print(f"{tag:16s} {r['n']:5d} {r['kl_mean']:10.3e} {r['kl_p99']:10.3e} {r['kl_max']:10.3e} {r['top1']:7.4f} {r['dtok_max']:11.4f} "
              f"{r['d20_max']:11.4f} {r['dnll_mean']:+9.5f}")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    return res


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    kl = _stats([r["kl"] for r in recs])
    return {"n": len(recs), "kl_mean": kl["mean"], "kl_p50": kl["p50"], "kl_p90": kl["p90"], "kl_p99": kl["p99"], "kl_max": kl["max"],
            "top1": sum(r["top1"] for r in recs) / len(recs), "top1_flips": sum(not r["top1"] for r in recs),
            "dtok_mean": sum(r["dtok"] for r in recs) / len(recs), "dtok_max": max(r["dtok"] for r in recs),
            "d20_max": max(r["d20"] for r in recs), "dnll_mean": sum(r["nll_b"] - r["nll_a"] for r in recs) / len(recs),
            "nll_ref_mean": sum(r["nll_a"] for r in recs) / len(recs)}


# ----------------------------------------------------------------------------------------------- hook check
def apilp(set_path: str, out: str) -> None:
    """Normal (unforced) greedy decode with top-20 raw logprobs of the same prompts (no force file needed)."""
    model = model_id()
    res = {}
    for it in json.load(open(set_path))["items"]:
        ch = complete(model, it["prompt_ids"], it["n_dump"], ignore_eos=True, logprobs=20)
        lps = ch["logprobs"]["top_logprobs"]
        res[it["tag"]] = {"token_ids": ch["token_ids"],
                          "top": [{int(t.split(":", 1)[1]): float(lp) for t, lp in (d or {}).items()} for d in lps]}
        print(f"{it['tag']}: {len(ch['token_ids'])} tokens", flush=True)
    json.dump(res, open(out, "w"))


def hookcheck(set_path: str, api_path: str, dump: str) -> None:
    """Forced-path dump vs normal-path API logprobs of the same server, on the shared greedy prefix."""
    import torch

    api = json.load(open(api_path))
    worst, n_tot = 0.0, 0
    for it in json.load(open(set_path))["items"]:
        tag = it["tag"]
        if tag not in api or not os.path.exists(os.path.join(dump, tag, "rank0.json")):
            continue
        logits, idx, _ = load_dump(os.path.join(dump, tag))
        lp = torch.log_softmax(logits.double(), -1)
        a = api[tag]
        same = 0
        while same < len(a["token_ids"]) and same < len(it["forced"]) and a["token_ids"][same] == it["forced"][same]:
            same += 1
        dmax, n = 0.0, 0
        for row, j in enumerate(idx.tolist()):
            if j >= same:
                break
            for t, v in a["top"][j].items():  # json keys are strings
                dmax = max(dmax, abs(float(lp[row, int(t)]) - float(v)))
            n += 1
        worst, n_tot = max(worst, dmax), n_tot + n
        print(f"{tag}: greedy==forced for {same}/{len(a['token_ids'])} tokens; forced-vs-normal max |dlogp| over top-20 = {dmax:.3e} ({n} positions)")
    print(f"hookcheck: {n_tot} positions, max |dlogp| {worst:.3e}")


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("set")
    g.add_argument("--k", type=int, default=5)
    c = sub.add_parser("collect")
    c.add_argument("set")
    c.add_argument("--force-file", required=True)
    c.add_argument("--dump", required=True)
    c.add_argument("--only", nargs="*")
    c.add_argument("--bg", type=int, default=0, help="background streams decoding during the collection (multi-request batches)")
    c.add_argument("--bg-tokens", type=int, default=2048, help="max_tokens of each background request (restarted when done)")
    a1 = sub.add_parser("apilp")
    a1.add_argument("set")
    a1.add_argument("out")
    h = sub.add_parser("hookcheck")
    h.add_argument("set")
    h.add_argument("api")
    h.add_argument("dump")
    tb = sub.add_parser("table")
    tb.add_argument("ref")
    tb.add_argument("tests", nargs="+")
    tb.add_argument("--set")
    m = sub.add_parser("compare")
    m.add_argument("ref")
    m.add_argument("test")
    m.add_argument("--set")
    m.add_argument("--json")
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"
    if a.cmd == "gen":
        gen(a.set, a.k)
    elif a.cmd == "collect":
        sys.exit(1 if collect(a.set, a.force_file, a.dump, a.only, bg=a.bg, bg_tokens=a.bg_tokens) else 0)
    elif a.cmd == "apilp":
        apilp(a.set, a.out)
    elif a.cmd == "hookcheck":
        hookcheck(a.set, a.api, a.dump)
    elif a.cmd == "table":
        table(a.ref, a.tests, a.set)
    elif a.cmd == "compare":
        compare_dirs(a.ref, a.test, a.set, a.json)


if __name__ == "__main__":
    main()
