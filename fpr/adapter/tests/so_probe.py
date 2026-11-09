"""Structured-output requests alongside a decode load (checks ASYNC + PHYS_ASYNC_GRAMMAR=hold on the GPU). K client
loops for --seconds, alternating two grammar-constrained kinds, each output checked:
  json   response_format json_schema (an object with an "items" array of 10 to 30 {id:int, name:str, tags:[str]}
         objects, no extra keys): the content must parse and match the schema exactly;
  tool   a strict tool with tool_choice "required" (api_smoke.py's get_weather schema): one call to
         get_weather with {city: str, days: int} and no other keys.
Thinking off (chat_template_kwargs, as the README bench), synthetic prompts. Prints one JSON line: requests, valid,
invalid (+ up to 3 short reasons), errors, completion tokens, wall. Counts only; no output text is stored.
From fpr/:
  python adapter/tests/so_probe.py --port 8100 --conc 6 --seconds 60
"""
import argparse
import json
import threading
import time
import urllib.request

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
    "items": {"type": "array", "minItems": 10, "maxItems": 30, "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "name", "tags"], "properties": {
            "id": {"type": "integer"}, "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4}}}}}}
STRICT = [{"type": "function", "function": {"name": "get_weather", "description": "weather", "strict": True,
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city", "days"], "additionalProperties": False}}}]
CITIES = ["Zurich", "Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi"]


def check_json(txt: str) -> str:
    try:
        d = json.loads(txt)
    except ValueError as e:
        return f"json: {str(e)[:60]}"
    if not isinstance(d, dict) or set(d) != {"items"} or not isinstance(d["items"], list):
        return "json: top level"
    if not 10 <= len(d["items"]) <= 30:
        return f"json: {len(d['items'])} items"
    for it in d["items"]:
        if not isinstance(it, dict) or set(it) != {"id", "name", "tags"} or not isinstance(it["id"], int) \
                or isinstance(it["id"], bool) or not isinstance(it["name"], str) or not isinstance(it["tags"], list) \
                or len(it["tags"]) > 4 or not all(isinstance(t, str) for t in it["tags"]):
            return "json: item"
    return ""


def check_tool(msg: dict) -> str:
    calls = msg.get("tool_calls") or []
    if len(calls) < 1:
        return "tool: no call"
    c = calls[0]["function"]
    if c.get("name") != "get_weather":
        return "tool: name"
    try:
        a = json.loads(c.get("arguments") or "")
    except ValueError:
        return "tool: arguments json"
    if set(a) != {"city", "days"} or not isinstance(a["city"], str) or not isinstance(a["days"], int) \
            or isinstance(a["days"], bool):
        return "tool: arguments"
    return ""


def one(base, model, kind, k):
    if kind == "json":
        body = dict(model=model, max_tokens=1500, temperature=0.6, messages=[{"role": "user", "content":
                    f"Invent {12 + k % 15} fictional open-source projects (batch {k}). Give each an id, a name and 2-3 tags."}],
                    response_format={"type": "json_schema", "json_schema": {"name": "projects", "schema": SCHEMA, "strict": True}})
    else:
        body = dict(model=model, max_tokens=400, temperature=0.6, tools=STRICT, tool_choice="required", messages=[
                    {"role": "user", "content": f"Weather in {CITIES[k % len(CITIES)]} for {1 + k % 7} days?"}])
    body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    n = (d.get("usage") or {}).get("completion_tokens", 0)
    if kind == "json" and d["choices"][0].get("finish_reason") == "length":
        return "capped", n       # cut by max_tokens mid-object: not checkable, not a grammar failure
    why = check_json(msg.get("content") or "") if kind == "json" else check_tool(msg)
    return why, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--lenient", action="store_true", help="CPU fake worker: its unscripted outputs are not meaningful; "
                    "fail on request errors only")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=60).read())["data"][0]["id"]
    st = dict(requests=0, valid=0, invalid=0, errors=0, tokens=0, reasons=[])
    lock = threading.Lock()
    t_end = time.time() + a.seconds

    def loop(c):
        k = c * 1000
        while time.time() < t_end:
            kind = "json" if k % 2 == 0 else "tool"
            try:
                why, n = one(base, model, kind, k)
                with lock:
                    st["requests"] += 1
                    st["tokens"] += n
                    if why == "capped":
                        st["capped"] = st.get("capped", 0) + 1
                    elif why:
                        st["invalid"] += 1
                        if len(st["reasons"]) < 3:
                            st["reasons"].append(why)
                    else:
                        st["valid"] += 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    st["errors"] += 1
                    if len(st["reasons"]) < 3:
                        st["reasons"].append(repr(e)[:80])
            k += 1
    t0 = time.time()
    th = [threading.Thread(target=loop, args=(c,)) for c in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    st["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(st))
    ok = st["errors"] == 0 and st["requests"] > 0 and (a.lenient or st["invalid"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
