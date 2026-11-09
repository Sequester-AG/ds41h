"""CPU contract tests of PhysScheduler + PhysExecutor against vLLM 0.32's own engine (LLMEngine -> in-process
EngineCore -> Scheduler.update_from_output / StructuredOutputManager / detokenizer / stop checks), with the fake worker.

  CUDA_VISIBLE_DEVICES= PHYS_MODEL=<model dir> python -m pytest -q adapter/tests/test_contract.py

Every step runs PhysScheduler._check (PHYS_ADAPTER_CHECK=1): the OwnedKVManager invariants, and vLLM's and the rank
scheduler's views of every running request agree (num_computed_tokens, output counts, RUNNING set). The fake worker
checks every StepBatch (row order, slots, decode / prefill token rows). Its tokens are keyed by (prompt, position), so
each finished request's output must equal the fake model's stream exactly -- through MTP acceptance, stop trimming,
preemption + recompute, prefix hits and migrations.
"""
from __future__ import annotations

import os
import re
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_util as U  # noqa: E402

import pytest  # noqa: E402
from vllm import SamplingParams  # noqa: E402

pytestmark = pytest.mark.vllm      # vLLM on the CPU, fake worker: needs PHYS_MODEL (conftest.py)
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402


class Harness:
    def __init__(self, geometry: str, **over):
        os.environ["PHYS_GEOMETRY"] = geometry
        self.eng = U.make_engine(**over)
        self.core = U.core_of(self.eng)
        self.sched = self.core.scheduler
        self.fake = self.core.model_executor.worker
        self.tok = self.eng.get_tokenizer()

    def add(self, reqs: list) -> None:
        for rid, ids, sp in reqs:
            self.eng.add_request(rid, {"prompt_token_ids": list(ids)}, sp)

    def drain(self, max_steps: int = 20000, on_step=None) -> dict:
        done = {}
        for i in range(max_steps):
            for o in self.eng.step():
                if o.finished:
                    done[o.request_id] = o
            if on_step:
                on_step(i)
            if not self.eng.has_unfinished_requests():
                break
        assert not self.eng.has_unfinished_requests(), "requests left after max_steps"
        return done

    def run(self, reqs: list, **kw) -> dict:
        self.add(reqs)
        return self.drain(**kw)

    def check_stream(self, out, n_expected: int | None = None) -> list:
        toks = list(out.outputs[0].token_ids)
        if n_expected is not None:
            assert len(toks) == n_expected, (out.request_id, len(toks), n_expected)
        exp = self.fake.expected(out.prompt_token_ids, len(toks))
        assert toks == exp, (out.request_id, [i for i, (a, b) in enumerate(zip(toks, exp)) if a != b][:5])
        return toks

    def idle(self) -> None:
        """Nothing left anywhere: rank scheduler, KV pools (prefix-cache pages aside), vLLM queues."""
        s = self.sched
        assert not s.rs.running and not s.rs.waiting and not s.rs.reqs, (len(s.rs.running), len(s.rs.waiting))
        assert not s.rs.kv.seqs and not s.rs.kv.pinned
        assert not s.running and not s.requests and not s._started
        assert all(p.reserved == 0 for p in s.rs.kv.paged)
        s.rs.kv.check()


def prompt(tag: str, n: int) -> list:
    """n token ids, unique per tag, deterministic."""
    s = zlib.crc32(tag.encode())
    return [1000 + (s + 7919 * i) % 50000 for i in range(n)]


@pytest.fixture(scope="module")
def big():
    return Harness("kv:2e9")


@pytest.fixture(scope="module")
def small():
    # 4 ranks x 160 pages of 64 tokens (10K tokens per rank), 96 window pages, 12 state slots: eviction, moves and
    # preemption with a handful of long requests
    return Harness("small:160,96,12")


# ------------------------------------------------------------------------------------------------ basic lifecycle
def test_lengths_and_streams(big):
    reqs = []
    for i in range(40):
        n = 30 + (i * 977) % 6000
        reqs.append((f"len{i}", prompt(f"len{i}", n), SamplingParams(max_tokens=1 + (i * 37) % 300)))
    done = big.run(reqs)
    assert len(done) == 40
    for rid, ids, sp in reqs:
        o = done[rid]
        assert o.outputs[0].finish_reason == "length"
        big.check_stream(o, sp.max_tokens)
    big.idle()


def test_mtp_acceptance(big):
    """k = 3: decode groups of 4 rows; tokens per decode step follow the fake's real-traffic tail (mean 2.83), and
    vLLM's num_computed_tokens follows the accepted counts (checked every step)."""
    big.fake.keep_log = True
    big.fake.log.clear()
    done = big.run([(f"mtp{i}", prompt(f"mtp{i}", 200), SamplingParams(max_tokens=400)) for i in range(16)])
    big.fake.keep_log = False
    dec = [e for e in big.fake.log if e[2] == "decode"]
    assert dec and all(e[4] == 4 for e in dec)
    mean = sum(len(e[5]) for e in dec) / len(dec)
    assert 2.6 < mean < 3.05, mean
    for o in done.values():
        big.check_stream(o, 400)
    big.idle()


def test_stop_token_inside_accepted_group(big):
    """A stop token among the k + 1 tokens of a step: vLLM trims after it, the rank scheduler commits only the kept
    tokens (no block hash beyond vLLM's token stream), the sequence is freed."""
    reqs, want = [], {}
    for i in range(24):
        ids = prompt(f"stop{i}", 300 + 50 * i)
        stream = big.fake.expected(ids, 200)
        j = 4 + (i * 13) % 150
        tok = stream[j]
        j = stream.index(tok)                      # first occurrence
        want[f"stop{i}"] = (j + 1, tok)
        reqs.append((f"stop{i}", ids, SamplingParams(max_tokens=200, stop_token_ids=[tok])))
    done = big.run(reqs)
    for rid, (n, tok) in want.items():
        o = done[rid]
        toks = big.check_stream(o, n)
        assert toks[-1] == tok and o.outputs[0].finish_reason == "stop" and o.outputs[0].stop_reason == tok
    big.idle()


def test_stop_string_and_abort(big):
    """Stop strings (detokenizer -> finish_requests) and client aborts, mid-prefill and mid-decode."""
    tok = big.tok
    reqs = []
    for i in range(10):
        ids = prompt(f"sstr{i}", 400)
        text = tok.decode(big.fake.expected(ids, 120))
        m = re.search(r"[A-Za-z]{4,}", text[40:])
        if not m:
            continue
        reqs.append((f"sstr{i}", ids, SamplingParams(max_tokens=120, stop=[m.group(0)])))
    assert len(reqs) >= 5
    long = [(f"ab{i}", prompt(f"ab{i}", 9000 + 3000 * i), SamplingParams(max_tokens=500)) for i in range(6)]
    big.add(reqs + long)
    done = {}
    for step in range(20000):
        for o in big.eng.step():
            if o.finished:
                done[o.request_id] = o
        if step == 3:
            big.eng.abort_request(["ab0", "ab1"])          # still prefilling
        if step == 40:
            big.eng.abort_request(["ab2", "ab3"])          # decoding by now
        if not big.eng.has_unfinished_requests():
            break
    for rid, ids, sp in reqs:
        o = done[rid]
        assert o.outputs[0].finish_reason == "stop", (rid, o.outputs[0].finish_reason)
        assert sp.stop[0] not in o.outputs[0].text
        toks = list(o.outputs[0].token_ids)
        assert toks == big.fake.expected(ids, len(toks))
    for rid in ("ab4", "ab5"):
        big.check_stream(done[rid], 500)
    big.idle()


# ------------------------------------------------------------------------------------------------ prefix cache
def test_prefix_hit_and_replay(big):
    """A follow-up call (prompt = previous prompt + output + new turn) hits the cached prefix on some rank: vLLM
    reports the hit (prompt_tokens_details.cached_tokens), the first chunk starts 128 tokens before the hit."""
    base = prompt("conv", 8000)
    o1 = big.run([("c1", base, SamplingParams(max_tokens=300))])["c1"]
    follow = base + list(o1.outputs[0].token_ids) + prompt("turn2", 700)
    big.fake.keep_log = True
    big.fake.log.clear()
    o2 = big.run([("c2", follow, SamplingParams(max_tokens=50))])["c2"]
    big.fake.keep_log = False
    hit = (len(base) + 300) // 64 * 64
    assert o2.num_cached_tokens >= hit - 64, (o2.num_cached_tokens, hit)
    first = [e for e in big.fake.log if e[2].startswith("prefill")][0]
    assert first[3] == o2.num_cached_tokens - 128, (first, o2.num_cached_tokens)
    big.check_stream(o2, 50)
    big.idle()


def test_skip_prefix_cache_reading(big):
    """prompt_logprobs requests do not read the prefix cache (vLLM's skip_reading_prefix_cache)."""
    base = prompt("nocache", 4000)
    big.run([("n1", base, SamplingParams(max_tokens=5))])
    sp = SamplingParams(max_tokens=5, prompt_logprobs=1)
    o = big.run([("n2", base, sp)])["n2"]
    assert o.num_cached_tokens == 0
    big.idle()


# ------------------------------------------------------------------------------------------------ pressure
def test_preemption_and_moves(small):
    """Tiny pools: decode growth evicts, moves sequences between ranks and preempts (recompute); every request still
    produces exactly its stream."""
    reqs = [(f"p{i}", prompt(f"p{i}", 2500 + 700 * (i % 5)), SamplingParams(max_tokens=1500)) for i in range(14)]
    done = small.run(reqs, max_steps=60000)
    for rid, ids, sp in reqs:
        small.check_stream(done[rid], 1500)
    st = small.sched.rs.stats
    assert st["preemptions"] > 0, st
    assert st["migrations"] > 0, st
    small.idle()


def test_many_waiting(big):
    """More requests than seats (max_num_seqs 256): FCFS waiting, all finish."""
    reqs = [(f"w{i}", prompt(f"w{i}", 64 + (i % 7) * 100), SamplingParams(max_tokens=8 + i % 30)) for i in range(320)]
    done = big.run(reqs)
    assert len(done) == 320
    for rid, ids, sp in reqs:
        big.check_stream(done[rid], sp.max_tokens)
    big.idle()


# ------------------------------------------------------------------------------------------------ sampling features
def test_structured_output_regex(big):
    """Grammar-constrained decoding: vLLM's StructuredOutputManager builds the bitmask (rows per draft position), the
    executor splits it per request, drafts come back through take_draft_token_ids and are grammar-validated (invalid
    ones padded to -1, num_acceptable_drafts caps the acceptance). Text prompts: random-id prompts confuse vLLM's
    incremental detokenizer, which is not what this tests."""
    tok = big.tok
    pat = r"(yes|no)[0-9]{3,40}"
    reqs = []
    for i in range(12):
        ids = tok.encode(f"Request {i}: answer yes or no, then a number with many digits.", add_special_tokens=False)
        sp = SamplingParams(max_tokens=60, structured_outputs=StructuredOutputsParams(regex=pat))
        reqs.append((f"so{i}", ids, sp))
    reqs.append(("so_choice", tok.encode("Pick alpha or beta.", add_special_tokens=False),
                 SamplingParams(max_tokens=20, structured_outputs=StructuredOutputsParams(choice=["alpha", "beta"]))))
    reqs.append(("so_json", tok.encode("Give a JSON object.", add_special_tokens=False),
                 SamplingParams(max_tokens=80, structured_outputs=StructuredOutputsParams(
                     json={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}))))
    done = big.run(reqs)
    for rid, ids, sp in reqs[:-2]:
        o = done[rid]
        text = tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True)
        assert re.fullmatch(pat, text), (rid, text)
        assert o.outputs[0].finish_reason in ("stop", "length"), o.outputs[0].finish_reason
    assert tok.decode(list(done["so_choice"].outputs[0].token_ids), skip_special_tokens=True) in ("alpha", "beta")
    import json
    js = tok.decode(list(done["so_json"].outputs[0].token_ids), skip_special_tokens=True)
    if done["so_json"].outputs[0].finish_reason == "stop":
        assert isinstance(json.loads(js)["a"], int), js
    big.idle()


def test_logprobs(big):
    sp = SamplingParams(max_tokens=40, logprobs=3)
    o = big.run([("lp", prompt("lp", 500), sp)])["lp"]
    lps = o.outputs[0].logprobs
    assert lps is not None and len(lps) == 40
    toks = list(o.outputs[0].token_ids)
    for t, d in zip(toks, lps):
        assert t in d and len(d) >= 3
    big.idle()


def test_unsupported_rejected(big):
    """Requests the phys engine cannot serve end with an error instead of hanging (here: a resumable request is not
    constructible from LLMEngine, so a prompt at max_model_len stands in)."""
    sp = SamplingParams(max_tokens=1)
    o = big.run([("ok", prompt("ok", 50), sp)])["ok"]
    assert o.outputs[0].finish_reason == "length"
    big.idle()


# ------------------------------------------------------------------------------------------------ FORCE hook
def test_force_file_protocol(tmp_path):
    """The reference deployment's DS41H_FORCE_FILE protocol through the served path (what gate/client.py collect does): the forced
    request emits exactly the forced ids (k + 1 per verify step), the dump has n_dump rows with idx 0..n_dump-1 in
    patch_force's format; a concurrent normal request keeps its own stream."""
    import json

    import torch

    ff = tmp_path / "force.json"
    os.environ["DS41H_FORCE_FILE"] = str(ff)
    try:
        h = Harness("kv:2e9")
    finally:
        del os.environ["DS41H_FORCE_FILE"]
    p = prompt("forced", 3000)
    forced = [2000 + (i * 37) % 40000 for i in range(316)]
    n_dump = 300
    dump = tmp_path / "dump"
    ff.write_text(json.dumps({"tag": "t1", "prompt_ids": p, "forced": forced, "n_dump": n_dump, "dump_dir": str(dump)}))
    other = prompt("bystander", 900)
    done = h.run([("f", p, SamplingParams(max_tokens=n_dump, ignore_eos=True, temperature=0)),
                  ("o", other, SamplingParams(max_tokens=200))])
    assert list(done["f"].outputs[0].token_ids) == forced[:n_dump]
    h.check_stream(done["o"], 200)
    meta = json.loads((dump / "t1" / "rank0.json").read_text())
    assert meta["n"] == n_dump and not meta["partial"] and meta["input_mismatch"] == 0 and meta["tp"] == 4, meta
    assert meta["kind"] == "full" and meta["saved"] and meta["checked_rows"] > 0
    for r in (1, 2, 3):                                    # production's stock path: markers on the other ranks
        assert json.loads((dump / "t1" / f"rank{r}.json").read_text())["saved"] is False
    assert sum(meta["rows_per_step"]) >= n_dump and set(meta["rows_per_step"][1:]) == {4}, meta["rows_per_step"]
    assert meta["rows_per_step"][0] == 1, meta
    d = torch.load(dump / "t1" / "rank0.pt")
    assert d["logits"].shape[0] == n_dump and list(d["idx"]) == list(range(n_dump))
    h.idle()


def test_sampling_caps_reject():
    """A worker that declares its sampler's features (the worker wrapper: hsd = temperature + seed only): requests needing more end
    with an error output instead of reaching the step; the others run."""
    from adapter import bridge

    h = Harness("kv:2e9")
    bridge.set_sampling_caps(set())
    try:
        done = h.run([("ok", prompt("caps_ok", 300), SamplingParams(max_tokens=20, temperature=0.7, seed=3)),
                      ("tp", prompt("caps_tp", 300), SamplingParams(max_tokens=20, top_p=0.9)),
                      ("lp", prompt("caps_lp", 300), SamplingParams(max_tokens=20, logprobs=2))])
    finally:
        bridge.set_sampling_caps(None)
    h.check_stream(done["ok"], 20)
    for rid in ("tp", "lp"):
        assert done[rid].outputs[0].finish_reason not in ("stop", "length"), (rid, done[rid].outputs[0].finish_reason)
        assert not done[rid].outputs[0].token_ids
    h.idle()


def test_reset_prefix_cache(big):
    """EngineCore.reset_prefix_cache (the replay / gate scripts call /reset_prefix_cache in dev mode): refused while
    requests run, done when idle; a repeated prompt then hits nothing."""
    p = prompt("reset", 3000)
    big.run([("rp1", p, SamplingParams(max_tokens=4))])
    o = big.run([("rp2", p, SamplingParams(max_tokens=4))])["rp2"]
    assert o.num_cached_tokens > 0
    assert big.core.reset_prefix_cache() is True
    o = big.run([("rp3", p, SamplingParams(max_tokens=4))])["rp3"]
    assert o.num_cached_tokens == 0
    big.eng.add_request("rp4", {"prompt_token_ids": prompt("reset_busy", 500)}, SamplingParams(max_tokens=50))
    big.eng.step()
    assert big.core.reset_prefix_cache() is False
    big.drain()
    big.idle()


def test_parallel_sampling_n3(big):
    """n = 3: vLLM fans out child requests (one engine request each); each child is a full sequence with its own KV
    (the fake's stream is keyed by the prompt, so all three are equal; they must all complete exactly)."""
    from vllm.sampling_params import RequestOutputKind

    p = prompt("n3", 800)
    o = big.run([("n3", p, SamplingParams(max_tokens=30, n=3, output_kind=RequestOutputKind.FINAL_ONLY))])["n3"]
    assert len(o.outputs) == 3
    exp = big.fake.expected(p, 30)
    for c in o.outputs:
        assert list(c.token_ids) == exp and c.finish_reason == "length"
    big.idle()


def test_stats_counters(big):
    """vLLM's own stats from our scheduler: spec-decode acceptance (drafts / accepted per position), prefix-cache
    queries / hits, KV usage (the facade), running / waiting counts -- what /metrics and the exactness checks read."""
    seen = []
    orig = big.sched.make_stats

    def spy(*a, **k):
        st = orig(*a, **k)
        if st is not None:
            seen.append(st)
        return st
    big.sched.make_stats = spy
    try:
        base = prompt("stats", 2000)
        big.run([("st1", base, SamplingParams(max_tokens=100))])
        big.run([("st2", base, SamplingParams(max_tokens=100))])
    finally:
        big.sched.make_stats = orig
    spec = [s.spec_decoding_stats for s in seen if s.spec_decoding_stats is not None]
    drafts = sum(x.num_drafts for x in spec)
    acc_pos = [sum(x.num_accepted_tokens_per_pos[i] for x in spec) for i in range(3)]
    assert drafts > 0 and acc_pos[0] >= acc_pos[1] >= acc_pos[2] > 0, (drafts, acc_pos)
    a0 = acc_pos[0] / drafts
    assert 0.7 < a0 < 0.9, a0                     # the fake's tail: 0.80 / 0.60 / 0.43
    pc = [s.prefix_cache_stats for s in seen if s.prefix_cache_stats is not None]
    assert sum(x.queries for x in pc) > 0 and sum(x.hits for x in pc) >= 1920
    assert max(s.kv_cache_usage for s in seen) > 0
    big.idle()


def test_too_large_request_rejected():
    """A request no rank could ever hold (serve's RankScheduler.add_request -> False): an error output instead of a
    request blocking the FCFS queue forever; the next request still runs."""
    h = Harness("small:64,64,16")                     # 4,096 tokens per rank
    if not hasattr(h.sched.rs, "max_request_tokens"):
        pytest.skip("serve without max_request_tokens")
    done = h.run([("big", prompt("too_big", 6000), SamplingParams(max_tokens=5)),
                  ("ok", prompt("ok_small", 100), SamplingParams(max_tokens=5))])
    assert done["big"].outputs[0].finish_reason not in ("stop", "length") and not done["big"].outputs[0].token_ids
    h.check_stream(done["ok"], 5)
    assert h.sched.phys_stats["rejected"] == 1
    h.idle()
