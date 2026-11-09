"""Draft validity and input ids of decode sequences across engine steps.

Bugs guarded:
1. Stale device ids after a missed step. A decode sequence's input ids come from the drafter's `next_ids` row of the
   previous engine step (in that step's batch order). A sequence that missed a step (a rank move, a skipped allocation)
   used to read its old row, which the missed step had given to another sequence: one wrong input token in the KV and
   drafts marked valid. Fix (engine.py, PREV_STEP_CHECK): every prev_seq entry carries the engine step serial; device
   ids and draft validity are used only if the sequence ran in the immediately previous step.
2. Host-given drafts rejected. A sequence resuming after a missed step with every row id given by the host (drafts the
   serving layer already validated, e.g. against a grammar) had its drafts treated as invalid, so it accepted one token
   per step. Fix (Hooks.prepare_head): drafts are valid with the previous step's device ids OR with host-given ids.
3. Pipelined decode (Engine.dispatch: step N+1 issued before step N is settled) must count steps exactly like the
   synchronous loop, so pipelined steps keep their device ids and drafts.
Acceptance with host-given drafts is compared with the device-ids path on the same workload (identical batches), so
the comparison is bit for bit.

The CPU tests drive the rules directly on stand-in objects; the GPU tests run the real engine.
"""
from __future__ import annotations

import types

import pytest


# ------------------------------------------------------------------------------------------------------------- CPU
def _engine_module():
    from engine import engine as E
    return E


def test_missed_step_keeps_host_ids():
    """A sequence that missed the previous step keeps its host t0 (no stale device row) and its drafts are invalid;
    a sequence that ran in the previous step reads its own device ids."""
    import torch
    E = _engine_module()
    old = (E.DEVTOK_V2, E.PREV_STEP_CHECK)
    E.DEVTOK_V2, E.PREV_STEP_CHECK = False, True          # the CPU-tensor variant of device_tokens; the fix on
    try:
        eng = types.SimpleNamespace(prev_seq={}, step_serial=0)
        eng.prev_of = types.MethodType(E.Engine.prev_of, eng)
        hsd = types.SimpleNamespace(next_ids=[torch.zeros(8, 4, dtype=torch.int64) for _ in range(4)])
        hooks = types.SimpleNamespace(eng=eng, hsd=hsd)
        eng.step_serial = 1                                          # step 1: rank 0 runs A (row 0) and B (row 1)
        hsd.next_ids[0][0] = torch.tensor([101, 102, 103, 104])
        hsd.next_ids[0][1] = torch.tensor([201, 202, 203, 204])
        eng.prev_seq["A"], eng.prev_seq["B"] = (0, 0, 1), (0, 1, 1)
        eng.step_serial = 2                                          # step 2: A misses it; B at row 0, C at row 1
        hsd.next_ids[0][0] = torch.tensor([211, 212, 213, 214])
        hsd.next_ids[0][1] = torch.tensor([301, 302, 303, 304])
        eng.prev_seq["B"], eng.prev_seq["C"] = (0, 0, 2), (0, 1, 2)
        eng.step_serial = 3                                          # step 3: A resumes on rank 1 with host t0 104
        got = {}
        for r, rid, t0 in ((1, "A", 104), (0, "B", 211)):
            s = dict(tok=torch.tensor([t0, 0, 0, 0], dtype=torch.int64), dev=torch.device("cpu"))
            tab = types.SimpleNamespace(n=4, tokens=torch.tensor([t0, -1, -1, -1]), seq_rows=[(rid, 0, 4, "decode", False)])
            E.Hooks.device_tokens(hooks, r, s, tab)
            got[rid] = s["tok"].tolist()
        assert got["A"][0] == 104, f"A read another sequence's ids {got['A']}"
        assert eng.prev_of("A") is None
        assert got["B"] == [211, 212, 213, 214] and eng.prev_of("B") == (0, 0)
    finally:
        E.DEVTOK_V2, E.PREV_STEP_CHECK = old


def test_drafts_valid_rule():
    """drafts_valid: True with the previous step's device ids, True with every row id given by the host, False for a
    sequence resuming with host t0 and no drafts."""
    import torch
    E = _engine_module()
    eng = types.SimpleNamespace(prev_seq={}, step_serial=3, stats={})
    eng.prev_of = types.MethodType(E.Engine.prev_of, eng)
    got = {}
    hsd = types.SimpleNamespace(prepare=lambda batches, states, bms: got.update({k: v.drafts_valid for k, v in states.items()}))
    hooks = types.SimpleNamespace(hsd=hsd, gate=None, sampling=None, samp_ctx={}, grammar={}, eng=eng)
    hooks._bitmasks = lambda b: None
    eng.prev_seq["A"] = (0, 0, 2)          # ran in serial 2 = the previous step
    eng.prev_seq["B"] = (0, 1, 1)          # missed the previous step, resumes with host t0 only
    eng.prev_seq["C"] = (0, 2, 1)          # missed it, resumes with host-given drafts
    SE = lambda rid, row0, kind="decode", n=4: types.SimpleNamespace(rid=rid, row0=row0, n_rows=n, kind=kind)  # noqa: E731
    sb = types.SimpleNamespace(seqs=[SE("A", 0), SE("B", 4), SE("C", 8), SE("P", 12, "prefill", 8)])
    tok = torch.tensor([-1, -1, -1, -1, 555, -1, -1, -1, 777, 11, 12, 13] + [9] * 8)
    E.Hooks.prepare_head(hooks, [dict(tab=types.SimpleNamespace(n=20, tokens=tok))], [sb])
    assert {k: got[k] for k in "ABC"} == dict(A=True, B=False, C=True)
    assert eng.stats["resume_last"] == (2, 1, 3)         # (resuming, of them with host drafts, decode sequences)


# ------------------------------------------------------------------------------------------------------------- GPU
CTXS = [8_192 + (i * 2_741) % 40_000 for i in range(48)]   # 48 decoders, 8K-48K contexts
STEPS = 12
SKIP_AT = 5                                                  # the step the skipped sequences miss


@pytest.fixture(scope="module")
def eng():
    import drive
    return drive.engine_once("d13", "d13=1,upload_v2=1")


def _wl():
    import drive
    return drive.Workload(ctxs=CTXS, steps=STEPS)


@pytest.fixture(scope="module")
def awaited(eng):
    import drive
    return drive.run(eng, _wl(), mode="async_np")


@pytest.fixture(scope="module")
def piped(eng, awaited):
    import drive
    return drive.run(eng, _wl(), mode="async")


@pytest.mark.gpu(n=4)
def test_pipelined_equals_awaited(awaited, piped):
    """Pipelined dispatch gives the same tokens and acceptance per step as dispatch with every step awaited."""
    assert len(piped.steps) == len(awaited.steps) == STEPS
    assert piped.streams == awaited.streams
    assert [s["accepted"] for s in piped.steps] == [s["accepted"] for s in awaited.steps]
    for a in (awaited, piped):
        assert all(s.get("nonfinite", 0) == 0 for s in a.steps)


@pytest.mark.gpu(n=4)
def test_pipelined_steps_keep_drafts(piped):
    """On every pipelined step after a sequence's first, its drafts (the previous step's device ids) are valid."""
    heads = [dv for _, dv in piped.dv if dv]
    assert len(heads) >= STEPS - 1
    first, rest = heads[0], heads[1:]
    assert not any(first.values())                      # seeded sequences start without drafts
    bad = [(i + 1, rid) for i, dv in enumerate(rest) for rid, ok in dv.items() if rid.startswith("d") and not ok]
    assert not bad, f"pipelined steps with drafts marked invalid: {bad[:8]} ({len(bad)} in all)"


@pytest.fixture(scope="module")
def ref(eng):
    import drive
    return drive.run(eng, _wl(), mode="sync")


@pytest.mark.gpu(n=4)
def test_host_given_drafts_equal_device_drafts(eng, ref):
    """The same workload with every decode row's ids given by the host (exactly the ids the engine would read on the
    device) gives the same tokens and the same acceptance at every step as the device-ids path."""
    import drive
    host = drive.run(eng, _wl(), mode="sync", host_ids=drive.device_ids_from_host(eng))
    assert [s["accepted"] for s in host.steps] == [s["accepted"] for s in ref.steps]
    assert host.streams == ref.streams
    assert [s["h_md5"] for s in host.steps] == [s["h_md5"] for s in ref.steps]


@pytest.mark.gpu(n=4)
def test_resume_after_a_missed_step(eng):
    """Sequences miss one step, then resume: with host-given drafts (the ids they would have read on the device) their
    drafts are valid; with host t0 only, their drafts are invalid and they accept exactly one token. Sequences that
    did not miss the step keep valid drafts."""
    import drive
    g_host = {f"d{i}" for i in range(0, 48, 6)}          # resume with host-given drafts
    g_t0 = {f"d{i}" for i in range(3, 48, 6)}            # resume with host t0 only
    saved = {}

    def on_plan(k, plan, eng_):
        if k == SKIP_AT:
            for rid in g_host:                           # the ids each would read now: its next_ids row of step k - 1
                r, si, serial = eng_.prev_seq[rid]
                assert serial == eng_.step_serial    # it ran in step k - 1
                saved[rid] = eng_.hooks.hsd.next_ids[r][si, :4].tolist()
            plan.work = [w for w in plan.work if not (w.kind == "decode" and str(w.rid) in g_host | g_t0)]

    def host_ids(k, rid, n, streams):
        if k != SKIP_AT + 1:
            return None
        if rid in g_host:
            return saved[rid][:n]
        if rid in g_t0:
            return [streams[rid][-1]] + [-1] * (n - 1)
        return None

    res = drive.run(eng, _wl(), mode="sync", on_plan=on_plan, host_ids=host_ids)
    serial = next(s["serial"] for s in res.steps if s["step"] == SKIP_AT + 1)
    dv = next(d for s_, d in res.dv if s_ == serial)
    assert all(dv[r] for r in g_host), "host-given drafts were marked invalid"
    assert not any(dv[r] for r in g_t0), "a resumed sequence without drafts had them marked valid"
    others = [r for r in dv if r.startswith("d") and r not in g_host | g_t0]
    assert others and all(dv[r] for r in others)
    acc = next(s["accepted"] for s in res.steps if s["step"] == SKIP_AT + 1)
    assert all(acc[r] == 1 for r in g_t0)
    assert all(s["nonfinite"] == 0 for s in res.steps)
