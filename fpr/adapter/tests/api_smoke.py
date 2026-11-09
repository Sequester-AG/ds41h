"""API-layer smoke test of the served fork: OpenAI chat / completions through vLLM's unchanged API server (reasoning
parser deepseek_v41, DSML tool parser + dsml_rt_v2_2 plugin, strict-tool grammar, prompt tokens details), against a
server started with scripts/serve_fpr.sh (at the repo root). With the fake worker in script mode
(adapter/tests/api_script.json) the expected parse of every reply is known exactly; with the real engine the same
checks are structural (finish reasons, parsed fields present, valid tool-call JSON).

From fpr/:
  python adapter/tests/api_smoke.py --port 8191 --mode fake      (CPU)
  python adapter/tests/api_smoke.py --port 8100 --mode real      (GPU server)
Aggregate output only; no prompt or completion is logged beyond these synthetic cases.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# HTTPS + API key (the endpoint in public mode, scripts/serve_fpr.sh): set by main() from --base-url /
# --cacert / --api-key-file. The key is read from its file and only ever sent in the Authorization header.
_CTX = None
_HDR: dict = {}


def _open(req, timeout=600):
    for k, v in _HDR.items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=timeout, context=_CTX)

TOOLS = [{"type": "function", "function": {"name": "grep", "description": "search code", "parameters": {
    "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}]
STRICT = [{"type": "function", "function": {"name": "get_weather", "description": "weather", "strict": True,
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city", "days"], "additionalProperties": False}}}]


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with _open(req, timeout) as r:
        return json.loads(r.read())


def stream(base, body, timeout=600):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning, calls, finish, usage = "", "", {}, None, None
    with _open(req, timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    return dict(content=content, reasoning=reasoning, calls=[calls[i] for i in sorted(calls)], finish=finish, usage=usage)


def chat(base, model, user, tools=None, **kw):
    body = dict(model=model, messages=[{"role": "user", "content": user}], max_tokens=kw.pop("max_tokens", 400), **kw)
    if tools:
        body["tools"] = tools
    r = post(base, "/v1/chat/completions", body)
    m = r["choices"][0]["message"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
    return dict(content=m.get("content") or "", reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
                calls=calls, finish=r["choices"][0]["finish_reason"], usage=r.get("usage")), body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--mode", choices=["fake", "real"], default="fake")
    ap.add_argument("--base-url", default="", help="e.g. https://127.0.0.1:8100 (default http://127.0.0.1:PORT)")
    ap.add_argument("--cacert", default="", help="CA bundle for --base-url https (production: serve/tls/ca.pem)")
    ap.add_argument("--api-key-file", default="", help="a client key file (serve/keys/*.key); never printed")
    a = ap.parse_args()
    global _CTX, _HDR
    base = a.base_url.rstrip("/") or f"http://127.0.0.1:{a.port}"
    if a.cacert:
        import ssl
        _CTX = ssl.create_default_context(cafile=a.cacert)
    auth_checked = None
    if a.api_key_file:
        with open(a.api_key_file) as f:
            key = f.read().strip()
        # without the key the endpoint must refuse (401); with it, serve
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=60, context=_CTX)
            auth_checked = False
        except urllib.error.HTTPError as e:
            auth_checked = e.code == 401
        _HDR = {"Authorization": f"Bearer {key}"}
        del key
    model = json.loads(_open(urllib.request.Request(base + "/v1/models"), 60).read())["data"][0]["id"]
    fake = a.mode == "fake"
    res, fails = {}, []

    def check(name, cond, info=""):
        res[name] = bool(cond)
        if not cond:
            fails.append(f"{name}: {info}")

    for streaming in (False, True):
        sfx = "_stream" if streaming else ""
        run = (lambda u, t=None, **kw: stream(base, chat_body(model, u, t, **kw))) if streaming else \
            (lambda u, t=None, **kw: chat(base, model, u, t, **kw)[0])
        # 1. plain chat with reasoning
        r = run("[[phys:hello]] Hi there!")
        if fake:
            check("chat" + sfx, r["content"] == "Hello! How can I help you today?" and
                  r["reasoning"].strip() == "The user greets me; answer briefly." and r["finish"] == "stop", r)
        else:
            check("chat" + sfx, r["finish"] in ("stop", "length") and (r["content"] or r["reasoning"]), r)
        # 2. tool call (auto tool choice, DSML parser + dsml_rt plugin)
        r = run("[[phys:tool]] Find foo in the repo.", TOOLS)
        if fake:
            check("tool" + sfx, r["finish"] == "tool_calls" and len(r["calls"]) == 1 and r["calls"][0]["name"] == "grep"
                  and json.loads(r["calls"][0]["arguments"]) == {"pattern": "foo"}
                  and r["content"].strip() == "I will grep.", r)
        else:
            ok = r["finish"] in ("tool_calls", "stop", "length")
            for c in r["calls"]:
                ok &= c["name"] == "grep" and isinstance(json.loads(c["arguments"]), dict)
            check("tool" + sfx, ok, r)
        # 3. strict tool (grammar-constrained through vLLM's structural tag: DSML_STRICT=vllm)
        r = run("[[phys:strict]] Weather in Zurich for 3 days?", STRICT, tool_choice="required")
        okc = len(r["calls"]) >= 1
        args = {}
        if okc:
            try:
                args = json.loads(r["calls"][0]["arguments"])
            except ValueError:
                okc = False
        if fake:
            check("strict" + sfx, okc and r["calls"][0]["name"] == "get_weather" and args == {"city": "Zurich", "days": 3}, r)
        else:
            check("strict" + sfx, okc and r["calls"][0]["name"] == "get_weather" and isinstance(args.get("days"), int)
                  and isinstance(args.get("city"), str) and set(args) <= {"city", "days"}, r)
        # 4. reasoning effort mapping of the dsml_rt plugin (medium -> high: stock would reject it)
        r = run("[[phys:hello]] Hi again.", reasoning_effort="medium")
        check("effort_medium" + sfx, r["finish"] in ("stop", "length"), r)
    # 5. prefix cache: the same long prompt twice -> cached tokens reported on the second call
    long_user = "[[phys:long]] " + " ".join(f"word{i}" for i in range(3000))
    r1, _ = chat(base, model, long_user, max_tokens=60)
    time.sleep(0.2)
    r2, _ = chat(base, model, long_user, max_tokens=60)
    ct = ((r2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    check("prefix_cached_tokens", ct >= 2048, r2.get("usage"))
    if fake:
        check("prefix_same_output", r1["content"] == r2["content"], (r1["content"][:60], r2["content"][:60]))
    # 6. completions with token ids in and out (the call shape of a token-level comparison client)
    ids = list(range(1000, 1600))
    c = post(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 32, "temperature": 0,
                                       "ignore_eos": True, "return_token_ids": True, "skip_special_tokens": False})
    ch = c["choices"][0]
    check("completions_token_ids", len(ch.get("token_ids") or []) == 32 and c["usage"]["prompt_tokens"] == 600, c["usage"])
    # 7. concurrency: 64 parallel short chats
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(64) as ex:
        outs = list(ex.map(lambda i: chat(base, model, f"[[phys:hello]] parallel {i}", max_tokens=100)[0], range(64)))
    check("parallel64", all(o["finish"] in ("stop", "length") for o in outs), [o["finish"] for o in outs][:8])
    if auth_checked is not None:
        check("unauthorized_rejected", auth_checked, "a request without the key was not refused with 401")
    print(json.dumps(dict(mode=a.mode, passed=sum(res.values()), total=len(res), results=res), indent=1))
    for f in fails:
        print("FAIL", f[:600])
    sys.exit(0 if not fails else 1)


def chat_body(model, user, tools=None, **kw):
    body = dict(model=model, messages=[{"role": "user", "content": user}], max_tokens=kw.pop("max_tokens", 400), **kw)
    if tools:
        body["tools"] = tools
    return body


if __name__ == "__main__":
    main()
