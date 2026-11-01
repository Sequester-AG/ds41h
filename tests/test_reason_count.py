"""CPU validation of the streaming reasoning-token count . Drives Parser.parse_delta and count_reasoning_tokens as chat_completion/serving.py does. Reference = a fresh full parse of the prefix after each chunk, from the state the prompt implies. Emitted deltas are hashed so old and new code can be compared byte for byte.
  python test_reason_count.py OUT.json"""
import hashlib, json, os, random, re, sys, time

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.tokenizers.registry import get_tokenizer

tok = get_tokenizer(os.environ.get("DS41H_MODEL_DIR", os.path.expanduser("~/.ds41h/model")), tokenizer_mode="deepseek_v41")
cls = ParserManager.get_parser(tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41", enable_auto_tools=True, model_name="m")
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], stream=True, tools=TOOLS)
E = lambda s: tok.encode(s, add_special_tokens=False)
ON = E("<｜begin▁of▁sentence｜><｜User｜>Say hi.<｜Assistant｜><think>")
OFF = E("<｜begin▁of▁sentence｜><｜User｜>Say hi.<｜Assistant｜></think>")
TOOL = lambda arg: f'<｜DSML｜ calls>\n<｜DSML｜ invoke name="read_file">\n<｜DSML｜ parameter name="path" string="true">{arg}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
BODY = " ".join(f"word{i} def f(x): return x+{i}." for i in range(70))
UNI = "日本語のテキスト 🐇🐛 emoji 👨‍👩‍👧‍👦 and accents éàü, 中文内容 " * 12
split = lambda *parts: [t for p in parts for t in E(p)]
CASES = {
    "off_plain": (OFF, E("Hello there. " + BODY)),
    "off_later_marker": (OFF, E("Intro. <think>later reasoning here " + BODY[:300] + "</think>Answer. " + BODY[:300])),
    "off_two_blocks": (OFF, E("A <think>one</think> B <think>two two two</think> C " + BODY[:200])),
    "off_open_never_closed": (OFF, E("Intro <think>open and never closed " + BODY[:300])),
    "off_split_delimiter": (OFF, split("Start <th", "ink>inner text ", BODY[:200], "</th", "ink>after ", BODY[:200])),
    "off_literal_end_marker": (OFF, E("Talking about </think> as text. " + BODY[:300])),
    "off_tool_marker_arg": (OFF, E("Calling. " + TOOL("/tmp/<think>x</think>y"))),
    "off_tool_plain": (OFF, E(TOOL("/etc/hostname"))),
    "off_unicode": (OFF, E(UNI)),
    "off_unicode_marker": (OFF, E(UNI[:80] + "<think>" + UNI[:120] + "</think>" + UNI[:80])),
    "on_answer": (ON, E("Reasoning here. " * 30 + "</think>The answer. " + BODY[:300])),
    "on_empty": (ON, E("</think>Hi there. " + BODY[:300])),
    "on_no_end": (ON, E("Still thinking, no end yet. " * 40)),
    "on_tool": (ON, E("Need a tool.</think>" + TOOL("/etc/hostname"))),
    "on_tool_marker_arg": (ON, E("Need a tool.</think>" + TOOL("/tmp/<think>x</think>y"))),
    "on_split_delimiter": (ON, split("Thinking ", BODY[:200], "</th", "ink>after ", BODY[:200])),
    "on_unicode": (ON, E(UNI[:300] + "</think>" + UNI[:300])),
}
for base in ("off_later_marker", "off_tool_marker_arg", "on_answer", "on_tool", "off_split_delimiter", "off_unicode_marker"):
    p, ids = CASES[base]
    for frac in (0.31, 0.5, 0.77):
        CASES[f"{base}__trunc{int(frac * 100)}"] = (p, ids[:max(2, int(len(ids) * frac))])
    CASES[f"{base}__trunc_last"] = (p, ids[:-1])


def reference(prompt, ids, ref):
    return ref._single_pass_parse(tok.decode(ids), ids, initial_state=ParserState.CONTENT if prompt is OFF else None) and None or ref.count_reasoning_tokens(ids)


def run(prompt, ids, chunk, seed, refcheck=True):
    parser = cls(tok, None, chat_template_kwargs={}, model_config=None) if False else cls(tok)
    ref_engine = parser._reasoning_parser._parser_engine_cls(tok)
    rnd = random.Random(seed); i, gen, first = 0, [], True
    h = hashlib.sha1(); diffs = []; cnt = []; t_cost = 0.0; repeat_ok = True
    while i < len(ids):
        n = chunk if chunk > 0 else rnd.randint(1, 8)
        d = ids[i:i + n]; i += n; gen.extend(d); done = i >= len(ids)
        m = parser.parse_delta(tok.decode(d), list(d), req, prompt_token_ids=prompt if first else None, finished=done); first = False
        h.update(re.sub(r'"id":"[^"]*"', '"id":"X"', m.model_dump_json() if m is not None else "-").encode())
        t = time.perf_counter(); c = parser.count_reasoning_tokens(tuple(gen)); t_cost += time.perf_counter() - t
        repeat_ok &= parser.count_reasoning_tokens(tuple(gen)) == c
        cnt.append(c)
        if refcheck:
            ref_engine._reset(); r = reference(prompt, gen, ref_engine); diffs.append(c - r)
    final_ref = diffs[-1] if diffs else None
    return dict(tokens=len(ids), final=cnt[-1], final_diff=final_ref, max_abs_mid_diff=max((abs(x) for x in diffs), default=0), mid_nonzero=sum(1 for x in diffs if x), chunks=len(cnt), emitted=h.hexdigest()[:16], repeat_ok=repeat_ok, ms=round(t_cost * 1e3, 2))


out = {}
for name, (prompt, ids) in CASES.items():
    for chunk in (1, 2, 3, 5, 8, 0):
        out[f"{name}/c{chunk or 'rnd'}"] = run(prompt, ids, chunk, seed=11)
# multiple choices: interleave two parsers, counts must equal running each alone
a_p, a_i = CASES["off_later_marker"]; b_p, b_i = CASES["off_plain"]
pa, pb = cls(tok), cls(tok); ga, gb, ca, cb = [], [], None, None
for j in range(max(len(a_i), len(b_i))):
    if j < len(a_i):
        ga.append(a_i[j]); pa.parse_delta(tok.decode([a_i[j]]), [a_i[j]], req, prompt_token_ids=a_p if j == 0 else None, finished=j == len(a_i) - 1); ca = pa.count_reasoning_tokens(tuple(ga))
    if j < len(b_i):
        gb.append(b_i[j]); pb.parse_delta(tok.decode([b_i[j]]), [b_i[j]], req, prompt_token_ids=b_p if j == 0 else None, finished=j == len(b_i) - 1); cb = pb.count_reasoning_tokens(tuple(gb))
iso = dict(a=ca, a_alone=out["off_later_marker/c1"]["final"], b=cb, b_alone=out["off_plain/c1"]["final"])
# request reset on one adapter: a shorter sequence must restart the count
pr = cls(tok); r1 = r2 = None
if hasattr(pr._reasoning_parser, "count_reasoning_tokens_after_prompt_end"):
    i2 = CASES["off_later_marker"][1]; ad = pr._reasoning_parser
    r1 = ad.count_reasoning_tokens_after_prompt_end(tuple(i2), finished=True); r2 = ad.count_reasoning_tokens_after_prompt_end(tuple(i2[:6]), finished=False)
# cost on a long stream, no reference
ln = E(" ".join(f"word{i} def f(x): return x+{i}." for i in range(1500)))
cost = {f"off_long_{len(ln)}/c{c}": run(OFF, ln, c, 3, refcheck=False)["ms"] for c in (4,)}
json.dump(dict(cases=out, isolation=iso, reset=[r1, r2], cost=cost), open(sys.argv[1], "w"), indent=0)
nz = {k: v for k, v in out.items() if v["final_diff"]}
print(f"cases {len(CASES)}, runs {len(out)}, chunks {sum(v['chunks'] for v in out.values())}")
print("final count differs from full parse:", len(nz), "of", len(out), "| repeat idempotent:", all(v["repeat_ok"] for v in out.values()))
for k, v in list(nz.items())[:14]:
    print(f"   {k:44s} final {v['final']:5d} diff {v['final_diff']:+d}")
mid = {k: v for k, v in out.items() if v["mid_nonzero"]}
print("runs with any mid-stream difference:", len(mid), "| worst difference:", max((v["max_abs_mid_diff"] for v in out.values()), default=0), "tokens | chunks differing:", sum(v["mid_nonzero"] for v in out.values()), "of", sum(v["chunks"] for v in out.values()))
for k, v in sorted(mid.items(), key=lambda kv: -kv[1]["max_abs_mid_diff"])[:8]:
    print(f"   {k:44s} worst {v['max_abs_mid_diff']:4d}, chunks {v['mid_nonzero']}/{v['chunks']}, final diff {v['final_diff']:+d}")
print("isolation", iso, "| reset", [r1, r2], "| cost ms", cost)
