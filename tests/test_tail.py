"""GPU test for ds41h/patch_tail.py (DS41H_DRAFT_TAIL): the deferred combine_sampled_and_draft_tokens launch waits for
the draft-tail event and produces exactly the stock input_ids / logits_indices; a draft-tail graph captured apart from
its producer graph and replayed on a side stream gives the same result as one graph replayed on the main stream.

usage: python tests/test_tail.py   (1 GPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this tree's ds41h (not the live one)

import torch  # noqa: E402


def make_inputs(num_reqs: int, k: int, dev):
    g = torch.Generator(device="cpu").manual_seed(1234 + num_reqs)
    max_reqs = 16
    q = k + 1  # verify rows per request (bonus + k drafts)
    num_tokens = num_reqs * q
    idx_mapping = torch.randperm(max_reqs, generator=g)[:num_reqs].to(torch.int32)
    qsl = torch.arange(0, num_tokens + 1, q, dtype=torch.int32)
    prefill_len = torch.randint(10, 50, (max_reqs,), generator=g, dtype=torch.int32)
    seq_lens = torch.empty(num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        seq_lens[i] = prefill_len[idx_mapping[i]] + 5 + i * 3  # decoding (seq_len > prefill_len)
    if num_reqs > 2:
        seq_lens[1] = prefill_len[idx_mapping[1]] - 2  # one request still in (chunked) prefill
    cu = torch.arange(0, num_reqs * q + 1, q, dtype=torch.int32)
    return dict(
        input_ids=torch.randint(0, 1000, (num_tokens + 8,), generator=g, dtype=torch.int32).to(dev),
        idx_mapping=idx_mapping.to(dev),
        last_sampled_tokens=torch.randint(0, 129280, (max_reqs,), generator=g, dtype=torch.int32).to(dev),
        query_start_loc=qsl.to(dev),
        seq_lens=seq_lens.to(dev),
        prefill_len=prefill_len.to(dev),
        draft_tokens=torch.randint(0, 129280, (max_reqs, k), generator=g, dtype=torch.int64).to(dev),
        cu_num_logits=cu.to(dev),
        num_logits=num_reqs * q,
    )


def call(fn, d):
    return fn(d["input_ids"], d["idx_mapping"], d["last_sampled_tokens"], d["query_start_loc"], d["seq_lens"],
              d["prefill_len"], d["draft_tokens"], d["cu_num_logits"], d["num_logits"], 1)


def test_deferred_combine() -> None:
    from vllm.v1.worker.gpu import input_batch as ib
    from vllm.v1.worker.gpu import model_runner as mr

    from ds41h import patch_tail as T

    if not getattr(mr.combine_sampled_and_draft_tokens, "__wrapped__", None):
        T.patch_runner(mr)
    assert mr.combine_sampled_and_draft_tokens is not ib.combine_sampled_and_draft_tokens, "patch not applied"
    T.DEFER = True
    T._ST["defer_ok"] = True  # normally set per step by the execute_model wrapper
    dev = torch.device("cuda")
    tail = T.tail_stream(dev)
    for num_reqs in (1, 3, 8):
        for k in (5, 7):
            a = make_inputs(num_reqs, k, dev)
            b = {key: (v.clone() if isinstance(v, torch.Tensor) else v) for key, v in a.items()}
            new_drafts = torch.randint(0, 129280, a["draft_tokens"].shape, dtype=torch.int64, device=dev)
            # stock, on the final draft tokens
            a["draft_tokens"].copy_(new_drafts)
            li_a = call(ib.combine_sampled_and_draft_tokens, a)
            # deferred: the "tail" writes the draft tokens late on the side stream
            main = torch.cuda.current_stream()
            tail.wait_stream(main)
            with torch.cuda.stream(tail):
                torch.cuda._sleep(20_000_000)  # ~10 ms: a missing wait would read the old drafts
                b["draft_tokens"].copy_(new_drafts)
                ev = torch.cuda.Event()
                ev.record(tail)
            T._ST["event"] = ev
            li_b = call(mr.combine_sampled_and_draft_tokens, b)
            assert T._ST["deferred"] is not None, "combine was not deferred"
            T._flush()
            assert T._ST["deferred"] is None and T._ST["event"] is None
            torch.cuda.synchronize()
            assert torch.equal(a["input_ids"], b["input_ids"]), f"input_ids differ (reqs {num_reqs}, k {k})"
            assert torch.equal(li_a, li_b), f"logits_indices differ (reqs {num_reqs}, k {k})"
            # no pending event: stock path
            c = {key: (v.clone() if isinstance(v, torch.Tensor) else v) for key, v in a.items()}
            li_c = call(mr.combine_sampled_and_draft_tokens, c)
            assert T._ST["deferred"] is None
            torch.cuda.synchronize()
            assert torch.equal(li_c, li_a) and torch.equal(c["input_ids"], a["input_ids"])
    print("deferred combine: PASS (reqs 1/3/8 x k 5/7, bitwise vs stock, waits for the tail event)")


def test_split_graph() -> None:
    """Producer graph A (main stream) + consumer graph B (captured separately, own pool, replayed on a side stream)
    == the same work as one graph, including under a concurrent main-stream writer of unrelated buffers."""
    from ds41h import patch_tail as T

    dev = torch.device("cuda")
    w1 = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    w2 = torch.randn(4096, 512, device=dev, dtype=torch.bfloat16)
    x = torch.randn(8, 512, device=dev, dtype=torch.bfloat16)
    out_ref = torch.zeros(8, dtype=torch.int64, device=dev)
    out = torch.zeros(8, dtype=torch.int64, device=dev)

    def part_a():
        return torch.relu(x @ w1)

    def part_b(h, o):
        o.copy_((h @ w2.t()).float().argmax(-1))

    # warm-up
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        part_b(part_a(), out_ref)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        part_b(part_a(), out_ref)
    ga = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga):
        h = part_a()
    gb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb, pool=torch.cuda.graph_pool_handle()):
        part_b(h, out)
    tail = T.tail_stream(dev)
    other = torch.zeros(1 << 20, device=dev)
    for it in range(20):
        x.copy_(torch.randn(8, 512, device=dev, dtype=torch.bfloat16))
        g1.replay()
        ga.replay()
        main = torch.cuda.current_stream()
        tail.wait_stream(main)
        with torch.cuda.stream(tail):
            gb.replay()
            ev = torch.cuda.Event()
            ev.record(tail)
        other.add_(1.0)  # main-stream work concurrent with B
        main.wait_event(ev)
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref), f"iteration {it}: split graphs differ from the single graph"
    print("split draft-tail graph: PASS (20 replays)")


def test_draft_block_tables() -> None:
    """FULL tail (DS41H_DRAFT_TAIL=2): DraftBlockTables gives the drafter private slot mappings / draft-group block tables;
    sync() copies exactly the runner's gathered rows; the runner's later writes do not reach the private copies; every
    other attribute is the runner's."""
    from ds41h import patch_tail as T
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.block_table import BlockTables

    dev = torch.device("cuda")
    real = BlockTables(block_sizes=[64, 64, 256, 64], max_num_reqs=8, max_num_batched_tokens=256,
                       max_num_blocks_per_group=[40, 40, 12, 40], device=dev)
    for i, t in enumerate(real.input_block_tables):
        t.copy_(torch.randint(1, 1000, t.shape, device=dev, dtype=t.dtype))
    real.slot_mappings.copy_(torch.randint(0, 10**6, real.slot_mappings.shape, device=dev))
    d = T.DraftBlockTables(real, [1, 3])
    assert d.slot_mappings.data_ptr() != real.slot_mappings.data_ptr() and d.slot_mappings.shape == real.slot_mappings.shape
    assert d.input_block_tables[0] is real.input_block_tables[0] and d.input_block_tables[2] is real.input_block_tables[2]
    assert d.input_block_tables[1].data_ptr() != real.input_block_tables[1].data_ptr()
    assert d.block_sizes == real.block_sizes and d.cp_size == real.cp_size and d.num_kv_cache_groups == 4
    d.sync(3)
    for g in (1, 3):
        assert torch.equal(d.input_block_tables[g][:3], real.input_block_tables[g][:3]), "sync rows differ"
    ref1 = d.input_block_tables[1].clone()
    real.input_block_tables[1].add_(7)  # the runner's next-step gather while the drafter graph runs
    real.slot_mappings.add_(3)
    sm = d.get_dummy_slot_mappings(32)
    assert sm.data_ptr() == d.slot_mappings.data_ptr() and bool((d.slot_mappings == PAD_SLOT_ID).all())
    assert torch.equal(d.input_block_tables[1], ref1), "private table changed by a runner write"
    dummy = d.get_dummy_block_tables(2)
    assert bool((dummy[1] == 0).all()) and bool((dummy[3] == 0).all()) and dummy[0].data_ptr() == real.input_block_tables[0].data_ptr()
    assert not bool((real.input_block_tables[0][:2] == 0).all()), "runner table zeroed by the drafter's dummy tables"
    torch.cuda.synchronize()
    print("draft block tables (FULL tail): PASS")


def test_late_fork_event() -> None:
    """FULL tail late fork (DS41H_DRAFT_TAIL=2, patch_tail.LATE_FORK): an external event recorded INSIDE a captured graph (an
    event-record node) lets another stream resume exactly at that point of each replay: work before the node is visible, work
    after it is not yet done (the graph keeps running while the other stream proceeds)."""
    dev = torch.device("cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.zeros(1, device=dev)
    y = torch.zeros(1, device=dev)
    ev = torch.cuda.Event(external=True)
    cycles = 40_000_000  # ~20-25 ms at the 1.7-2 GHz boost clock
    g = torch.cuda.CUDAGraph()
    s1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        with torch.cuda.graph(g, stream=s1):
            torch.cuda._sleep(cycles)
            x.fill_(1.0)
            ev.record()
            torch.cuda._sleep(cycles)
            y.fill_(1.0)
    torch.cuda.synchronize()
    for it in range(3):
        x.zero_()
        y.zero_()
        z = torch.zeros(2, device=dev)
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            g.replay()
        s2.wait_event(ev)
        with torch.cuda.stream(s2):
            z[0].copy_(x[0])
            z[1].copy_(y[0])
        torch.cuda.synchronize()
        assert float(z[0]) == 1.0, f"replay {it}: the other stream ran before the fork node (x not written yet)"
        assert float(z[1]) == 0.0, f"replay {it}: the other stream waited for the whole graph instead of the fork node"
        assert float(y[0]) == 1.0
    print("late-fork external event inside a graph: PASS (3 replays)")


def test_external_wait_event() -> None:
    """DS41H_DRAFTER_EMBED_EAGER: a captured graph that waits on an external event (event-wait node) sees the eager work recorded
    on another stream BEFORE each replay (a slow producer: the graph's consumer must wait for it), while graph work placed
    before the wait node runs without waiting."""
    dev = torch.device("cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    buf = torch.zeros(4, device=dev)
    out = torch.zeros(4, device=dev)
    pre = torch.zeros(1, device=dev)
    ev = torch.cuda.Event(external=True)
    ev.record()  # creates the CUDA event: Event.wait on a never-recorded event is a no-op (no wait node would be captured)
    cycles = 40_000_000  # ~20-25 ms
    g = torch.cuda.CUDAGraph()
    s1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        with torch.cuda.graph(g, stream=s1):
            pre.add_(1.0)  # independent graph work before the wait node
            s1.wait_event(ev)  # external event-wait node
            out.copy_(buf)
    torch.cuda.synchronize()
    for it in range(3):
        out.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(s2):  # eager producer: slow, then writes buf, then records the event
            torch.cuda._sleep(cycles)
            buf.fill_(float(it + 1))
            ev.record(s2)
        with torch.cuda.stream(s1):
            g.replay()
        torch.cuda.synchronize()
        assert torch.all(out == float(it + 1)), f"replay {it}: the graph read the buffer before the eager producer's event"
    assert float(pre[0]) == 3.0
    print("external event-wait node inside a graph: PASS (3 replays, slow eager producer)")


if __name__ == "__main__":
    torch.cuda.init()
    test_deferred_combine()
    test_split_graph()
    test_draft_block_tables()
    test_late_fork_event()
    test_external_wait_event()
    print("ALL PASS")
    sys.exit(0)
