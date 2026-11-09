"""The fork server's OpenAI API contract, against a running server (marked `server`; skipped without PHYS_SERVER_URL).

The fork replaces vLLM's scheduler, executor and engine but keeps vLLM's API server, parsers and chat template, so a
client must not see a difference. These tests are what the launcher's start check covers:
  * API smoke (11 checks): chat with reasoning, auto tool call, strict tool call, reasoning_effort "medium", each
    non-streaming and streaming; prefix-cache tokens reported; /v1/completions with token ids in and out; 64 parallel
    chats. With PHYS_API_KEY_FILE: a request without the key, or with a wrong one, is refused (401), on every path but
    /health (not only under /v1).
  * Re-render exactness (the DSML round trip): turn 1's prompt ids + generated ids must be the exact prefix of the
    prompt the server renders for turn 2 from the client-visible message (reasoning, content, parsed tool calls) plus
    tool results, non-streaming and streaming. Guards: any parser / template mismatch (a dropped or re-encoded DSML
    token, whitespace, argument JSON re-serialization) silently breaks prefix caching and changes what the model sees.
    With PHYS_MODEL and the model's own encoding/encoding.py: the server's parse equals the official parser's.
  * Strict tool calls under concurrency: every strict call is valid against its schema (required keys, types, enums,
    nested objects, arrays, no extra keys). Guards: the strict-tool grammar under the fork's scheduling (structured
    outputs held across ASYNC steps).

Environment: PHYS_SERVER_URL (http://host:port or https://...), PHYS_API_KEY_FILE (one client key; read, sent as a
Bearer token, never printed), PHYS_CA_FILE (CA bundle for https), PHYS_SERVER_FAKE=1 when the server runs the fake
worker in script mode (adapter/tests/api_script.json: replies are known exactly and checked exactly).
  PHYS_SERVER_URL=http://127.0.0.1:8100 python -m pytest -q adapter/tests/test_api_contract.py
Only synthetic prompts; nothing but pass / fail details is printed.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import os
import ssl
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.server

URL = os.environ.get("PHYS_SERVER_URL", "").rstrip("/")
FAKE = os.environ.get("PHYS_SERVER_FAKE", "0") == "1"
_CTX = ssl.create_default_context(cafile=os.environ["PHYS_CA_FILE"]) if os.environ.get("PHYS_CA_FILE") else None


def _key() -> str | None:
    p = os.environ.get("PHYS_API_KEY_FILE", "")
    if not p:
        return None
    with open(p) as f:
        return f.read().strip()


def _open(req, timeout=600, key: str | None | bool = True):
    k = _key() if key is True else (key or None)
    if k:
        req.add_header("Authorization", f"Bearer {k}")
    return urllib.request.urlopen(req, timeout=timeout, context=_CTX)


def get(path, **kw):
    with _open(urllib.request.Request(URL + path), 60, **kw) as r:
        return json.loads(r.read())


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with _open(req, timeout) as r:
        return json.loads(r.read())


def chat(body):
    """Non-streaming chat: the client-visible message, finish reason, usage, prompt / generated token ids."""
    r = post("/v1/chat/completions", body)
    ch = r["choices"][0]
    m = ch["message"]
    return dict(content=m.get("content") or "", reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
                calls=[{"id": c.get("id"), "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                       for c in m.get("tool_calls") or []],
                finish=ch.get("finish_reason"), usage=r.get("usage"), prompt_ids=r.get("prompt_token_ids"),
                gen_ids=ch.get("token_ids"))


def stream(body):
    """Streaming chat, accumulated as a client does (content / reasoning deltas, tool-call fragments by index)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning, calls, finish, usage, pids, gids = "", "", {}, None, None, None, []
    with _open(req, 600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            pids = ev.get("prompt_token_ids") or pids
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
                gids.extend(ch.get("token_ids") or [])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
    return dict(content=content, reasoning=reasoning, calls=[calls[i] for i in sorted(calls)], finish=finish,
                usage=usage, prompt_ids=pids, gen_ids=gids or None)


def run(streaming: bool, body: dict) -> dict:
    return stream(body) if streaming else chat(body)


@pytest.fixture(scope="module")
def model():
    return get("/v1/models")["data"][0]["id"]


def _body(model, user, tools=None, **kw):
    b = dict(model=model, messages=[{"role": "user", "content": user}], max_tokens=kw.pop("max_tokens", 400), **kw)
    if tools:
        b["tools"] = tools
    return b


TOOLS = [{"type": "function", "function": {"name": "grep", "description": "search code", "parameters": {
    "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}]
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather", "strict": True,
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city", "days"], "additionalProperties": False}}}


# ------------------------------------------------------------------------------------------------ minimal schema check
def _valid(v, s) -> bool:
    t = s.get("type")
    if "enum" in s and v not in s["enum"]:
        return False
    if t == "object":
        if not isinstance(v, dict) or any(k not in v for k in s.get("required", [])):
            return False
        props = s.get("properties", {})
        if s.get("additionalProperties") is False and any(k not in props for k in v):
            return False
        return all(_valid(v[k], props[k]) for k in v if k in props)
    if t == "array":
        return isinstance(v, list) and all(_valid(x, s.get("items", {})) for x in v)
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    return True


def _args(call):
    try:
        return json.loads(call["arguments"])
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------ API smoke (11 checks)
def test_unauthorized_rejected():
    if not os.environ.get("PHYS_API_KEY_FILE"):
        pytest.skip("PHYS_API_KEY_FILE is not set (a server without keys)")
    for bad in (False, "not-a-valid-key"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/v1/models", key=bad)
        assert e.value.code == 401, e.value.code


KEYED_PATHS = [("POST", "/invocations"), ("GET", "/metrics"), ("POST", "/tokenize"), ("POST", "/detokenize"),
               ("GET", "/version"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"),
               ("GET", "/load"), ("POST", "/pooling"), ("GET", "/no-such-path")]


def test_key_required_on_every_path():
    """Guards: stock vLLM 0.32's API-key middleware only checks /v1, /v2, /inference and /cohere, so e.g.
    /invocations runs the model without a key. A keyed server must refuse every path but /health without a key."""
    if not os.environ.get("PHYS_API_KEY_FILE"):
        pytest.skip("PHYS_API_KEY_FILE is not set (a server without keys)")
    open_ = []
    for method, path in KEYED_PATHS:
        body = json.dumps({"model": "m", "prompt": "x", "messages": [], "max_tokens": 1}).encode() if method == "POST" else None
        req = urllib.request.Request(URL + path, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60, context=_CTX).close()
            open_.append((path, 200))
        except urllib.error.HTTPError as e:
            if e.code != 401:
                open_.append((path, e.code))
    assert not open_, f"paths answered without a key: {open_}"
    with urllib.request.urlopen(URL + "/health", timeout=60, context=_CTX) as r:
        assert r.status == 200


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_chat_reasoning(model, streaming):
    r = run(streaming, _body(model, "[[phys:hello]] Hi there!"))
    if FAKE:
        assert r["content"] == "Hello! How can I help you today?", r
        assert r["reasoning"].strip() == "The user greets me; answer briefly." and r["finish"] == "stop", r
    else:
        assert r["finish"] in ("stop", "length") and (r["content"] or r["reasoning"]), r


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_tool_call(model, streaming):
    r = run(streaming, _body(model, "[[phys:tool]] Find foo in the repo.", TOOLS))
    if FAKE:
        assert r["finish"] == "tool_calls" and [c["name"] for c in r["calls"]] == ["grep"], r
        assert _args(r["calls"][0]) == {"pattern": "foo"} and r["content"].strip() == "I will grep.", r
    else:
        assert r["finish"] in ("tool_calls", "stop", "length"), r
        for c in r["calls"]:
            assert c["name"] == "grep" and isinstance(_args(c), dict), r


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_strict_tool(model, streaming):
    r = run(streaming, _body(model, "[[phys:strict]] Weather in Zurich for 3 days?", [WEATHER], tool_choice="required"))
    assert r["calls"] and r["calls"][0]["name"] == "get_weather", r
    a = _args(r["calls"][0])
    if FAKE:
        assert a == {"city": "Zurich", "days": 3}, r
    else:
        assert _valid(a, WEATHER["function"]["parameters"]), r


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_reasoning_effort_medium(model, streaming):
    """reasoning_effort "medium" is accepted (mapped by the chat-template plugin; stock rejects it)."""
    r = run(streaming, _body(model, "[[phys:hello]] Hi again.", reasoning_effort="medium"))
    assert r["finish"] in ("stop", "length"), r


def test_prefix_cached_tokens(model):
    user = "[[phys:long]] " + " ".join(f"word{i}" for i in range(3000))
    r1 = chat(_body(model, user, max_tokens=60))
    time.sleep(0.2)
    r2 = chat(_body(model, user, max_tokens=60))
    ct = ((r2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    assert ct >= 2048, r2.get("usage")
    if FAKE:
        assert r1["content"] == r2["content"]


def test_completions_token_ids(model):
    ids = list(range(1000, 1600))
    c = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 32, "temperature": 0, "ignore_eos": True,
                                 "return_token_ids": True, "skip_special_tokens": False})
    assert len(c["choices"][0].get("token_ids") or []) == 32 and c["usage"]["prompt_tokens"] == 600, c["usage"]


def test_parallel_64(model):
    with cf.ThreadPoolExecutor(64) as ex:
        outs = list(ex.map(lambda i: chat(_body(model, f"[[phys:hello]] parallel {i}", max_tokens=100)), range(64)))
    assert all(o["finish"] in ("stop", "length") for o in outs), [o["finish"] for o in outs]


# ------------------------------------------------------------------------------------------------ re-render exactness
RR_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read lines of a repository file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                                                     "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository with a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                     "case_sensitive": {"type": "boolean"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "write_note", "description": "Store a multi-line review note.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"},
                                                     "tags": {"type": "array", "items": {"type": "string"}},
                                                     "severity": {"type": "number"}, "meta": {"type": "object"}},
                    "required": ["title", "body"]}}},
]
RR_TASKS = [
    "Find where the cache size is configured: grep for `cache_size` (case sensitive), then read_file lines 1-40 of the hit.",
    "Write a note titled 'Cache: \"eviction\" notes' whose body has three bullet lines and a trailing newline, tags "
    "['cache', 'eviction'], severity 0.5 and meta {\"nested\": {\"a\": [1, 2.5, null]}, \"reviewed\": true}.",
]
CONTEXT = "\n".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}  # cache_size = {64 * i}" for i in range(120))
TOOL_RESULT = "\n".join(f"{i}: line {i} of the file, value = \"{i * 3}\"" for i in range(1, 41))


def _rr_cases():
    if FAKE:
        return [("[[phys:tool]] Find foo in the repo.", TOOLS)]
    return [(t, RR_TOOLS) for t in RR_TASKS]


def _turn2(model, msgs, tools, r1, effort):
    asst = {"role": "assistant", "content": r1["content"]}
    if r1["reasoning"]:
        asst["reasoning_content"] = r1["reasoning"]
    if r1["calls"]:
        asst["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                               "function": {"name": c["name"], "arguments": c["arguments"]}}
                              for i, c in enumerate(r1["calls"])]
    msgs2 = msgs + [asst]
    for tc in asst.get("tool_calls", []):
        msgs2.append({"role": "tool", "tool_call_id": tc["id"], "content": TOOL_RESULT})
    if not r1["calls"]:
        msgs2.append({"role": "user", "content": "Continue."})
    return chat(dict(model=model, messages=msgs2, tools=tools, tool_choice="auto", max_tokens=1, temperature=0,
                     reasoning_effort=effort, return_token_ids=True))


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_rerender_round_trip(model, streaming):
    effort = "high"
    for task, tools in _rr_cases():
        msgs = [{"role": "system", "content": "You are a code-review assistant. Always use the tools.\n\n" + CONTEXT},
                {"role": "user", "content": task}]
        r1 = run(streaming, dict(model=model, messages=msgs, tools=tools, tool_choice="auto", max_tokens=4000,
                                 temperature=0, reasoning_effort=effort, return_token_ids=True))
        if streaming and not (r1["prompt_ids"] and r1["gen_ids"]):
            pytest.skip("this server does not return token ids in streamed chat chunks")
        assert r1["prompt_ids"] and r1["gen_ids"], "the server returned no token ids (return_token_ids)"
        assert r1["finish"] in ("tool_calls", "stop"), ("turn 1 did not finish (raise max_tokens)", r1["finish"])
        r2 = _turn2(model, msgs, tools, r1, effort)
        ref, p2 = r1["prompt_ids"] + r1["gen_ids"], r2["prompt_ids"] or []
        k = next((i for i in range(min(len(ref), len(p2))) if ref[i] != p2[i]), None)
        assert len(p2) >= len(ref) and k is None, (
            f"turn 2's prompt is not turn 1's prompt + generation: first mismatch at {k} "
            f"({'in the generation, +' + str(k - len(r1['prompt_ids'])) if k is not None and k >= len(r1['prompt_ids']) else 'in the prompt'}"
            f"), {len(r1['calls'])} tool calls")


def _official():
    m = os.environ.get("PHYS_MODEL", "")
    p = os.path.join(m, "encoding", "encoding.py")
    if not m or not os.path.isfile(p) or importlib.util.find_spec("vllm") is None:
        return None, None
    spec = importlib.util.spec_from_file_location("official_encoding", p)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    from vllm.tokenizers import get_tokenizer

    return enc, get_tokenizer(m, tokenizer_mode="deepseek_v41")


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "stream"])
def test_parse_matches_official(model, streaming):
    """The server's reasoning / content / tool calls equal the model's official parser on the same generated ids."""
    enc, tok = _official()
    if enc is None:
        pytest.skip("needs PHYS_MODEL with the model's encoding/encoding.py and vLLM's tokenizer")
    task, tools = _rr_cases()[-1]
    msgs = [{"role": "user", "content": task}]
    r = run(streaming, dict(model=model, messages=msgs, tools=tools, tool_choice="auto", max_tokens=4000, temperature=0,
                            reasoning_effort="high", return_token_ids=True))
    if not r["gen_ids"]:
        pytest.skip("this server does not return token ids here")
    om = enc.parse_message_from_completion_text(tok.decode(r["gen_ids"], skip_special_tokens=False), "thinking")
    assert r["reasoning"] == om["reasoning_content"] and r["content"] == om["content"], (r["finish"],)
    o_calls = [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in om["tool_calls"]]
    assert [(c["name"], _args(c)) for c in r["calls"]] == o_calls


# ------------------------------------------------------------------------------------------------ strict tool calls
STRICT_CASES = [
    (WEATHER, "What is the weather in Lisbon for the next 5 days?"),
    ({"type": "function", "function": {"name": "create_ticket", "description": "Create a tracker ticket.", "strict": True,
      "parameters": {"type": "object", "properties": {
          "title": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
          "tags": {"type": "array", "items": {"type": "string"}}, "estimate_hours": {"type": "number"},
          "blocking": {"type": "boolean"}},
          "required": ["title", "priority", "tags", "estimate_hours", "blocking"], "additionalProperties": False}}},
     "File a high-priority ticket: the login page times out under load. Tag it backend and perf, about 6.5 hours, "
     "it blocks the release."),
    ({"type": "function", "function": {"name": "move_file", "description": "Move a file.", "strict": True,
      "parameters": {"type": "object", "properties": {
          "src": {"type": "string"}, "dst": {"type": "string"},
          "options": {"type": "object", "properties": {"overwrite": {"type": "boolean"}, "mode": {"type": "integer"}},
                      "required": ["overwrite", "mode"], "additionalProperties": False}},
          "required": ["src", "dst", "options"], "additionalProperties": False}}},
     "Move notes/a.txt to archive/a.txt, overwrite it if it exists, file mode 644."),
    ({"type": "function", "function": {"name": "set_points", "description": "Store polygon points.", "strict": True,
      "parameters": {"type": "object", "properties": {
          "points": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "integer"},
                                                                                 "y": {"type": "integer"}},
                                                "required": ["x", "y"], "additionalProperties": False}}},
          "required": ["points"], "additionalProperties": False}}},
     "Store the triangle (0, 0), (4, 0), (2, 3)."),
]


def test_strict_tool_calls_concurrent(model):
    cases = [(WEATHER, "[[phys:strict]] Weather in Zurich for 3 days?")] if FAKE else STRICT_CASES
    jobs = [(t, u) for _ in range(3) for t, u in cases]

    def one(job):
        tool, user = job
        return tool, chat(_body(model, user, [tool], tool_choice="required", max_tokens=2000))

    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    bad = []
    for tool, r in res:
        f = tool["function"]
        if not r["calls"] or any(c["name"] != f["name"] or not _valid(_args(c), f["parameters"]) for c in r["calls"]):
            bad.append((f["name"], r["finish"], [c["arguments"][:200] for c in r["calls"]]))
    assert not bad, f"{len(bad)} of {len(res)} strict calls invalid: {bad[:3]}"


# ------------------------------------------------------------------------------------------------ Responses store
STORE_MAX = int(os.environ.get("PHYS_SERVER_STORE_MAX", "0"))   # the server's PHYS_RESPONSES_STORE_MAX, if known


def _resp(model, text, **kw):
    return post("/v1/responses", dict(model=model, input=text, max_output_tokens=kw.pop("max_output_tokens", 200), **kw))


def _resp_text(r) -> str:
    return "".join(c.get("text", "") for item in r.get("output", []) for c in item.get("content") or []
                   if c.get("type") == "output_text")


def test_responses_store_and_follow_up(model):
    """Responses API with store: true: GET /v1/responses/{id} returns the stored response, and a follow-up naming it
    in previous_response_id is answered with turn 1 in its prompt. Guards the follow-up 404 (vLLM's store is off unless
    VLLM_ENABLE_RESPONSES_API_STORE=1)."""
    r1 = _resp(model, "Remember the word tangerine and reply OK.", store=True)
    assert r1.get("status") in ("completed", "incomplete"), r1.get("status")     # incomplete: max_output_tokens
    g = get(f"/v1/responses/{r1['id']}")
    assert g["id"] == r1["id"] and _resp_text(g) == _resp_text(r1)
    q = "Which word did I ask you to remember?"
    r2 = _resp(model, q, store=True, previous_response_id=r1["id"])
    assert r2.get("status") in ("completed", "incomplete"), r2.get("status")
    assert get(f"/v1/responses/{r2['id']}")["id"] == r2["id"]
    alone = _resp(model, q, store=False)
    assert r2["usage"]["input_tokens"] > alone["usage"]["input_tokens"] + 5, (r2["usage"], alone["usage"])


@pytest.mark.skipif(not 0 < STORE_MAX <= 64, reason="needs PHYS_SERVER_STORE_MAX = the server's store cap (<= 64)")
def test_responses_store_evicts(model):
    """The store is bounded (adapter/responses_store.py): after the cap's worth of newer stored responses the oldest
    answers 404, the newest still resolves."""
    first = _resp(model, "Say A.", store=True, max_output_tokens=16)
    for i in range(STORE_MAX):
        last = _resp(model, f"Say {i}.", store=True, max_output_tokens=16)
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"/v1/responses/{first['id']}")
    assert e.value.code == 404
    assert get(f"/v1/responses/{last['id']}")["id"] == last["id"]
