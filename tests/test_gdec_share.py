"""Shared grouped-decode work: bit-identity tests. GPU only (sm_90a).

  python -m pytest tests/test_gdec_share.py -x -q

S3a (DS41H_GDEC_SHARE): the layers of one index source get identical inputs to compute_global_topk_indices_and_lens, the
     key-set union and FlashMLA's split-KV scheduler; sharing their outputs is exact iff the three are pure functions of their
     inputs: repeated calls are bitwise equal (also with other work on a side stream and inside CUDA graphs) and attention with a
     reused (tile_md, num_splits) is bitwise equal to attention with a fresh schedule.
S3b (DS41H_GDEC_NOPAD): the stock fused qnorm / rope / KV-insert op with q_head_padded = 16 returns exactly the first 16 heads of
     its 64-head result and writes the same KV-cache bytes; the o-proj input op (fused_inv_rope_fp8_quant) gives identical
     outputs for the contiguous 16-head attention buffer and today's strided 16-of-64-head view.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import test_attn as ta  # noqa: E402

from ds41h import grouped_decode as gd  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

HAS = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _skip(cond, why):
    return pytest.mark.skipif(cond, reason=why) if pytest is not None else (lambda f: f)


def _param(n, v):
    return pytest.mark.parametrize(n, v) if pytest is not None else (lambda f: f)


gpu = _skip(not HAS, "needs sm_90a")

CASES = {
    "c64x4_r1": ([4000] * 64, [4] * 64, dict(ratio=1)),
    "c64x4_r2_holes": ([6000] * 64, [4] * 64, dict(ratio=2, holes=True)),
    "c16x8_r1": ([2048] * 16, [8] * 16, dict(ratio=1)),
    "short_ctx": ([40, 300, 41, 120], [8, 8, 8, 8], dict(ratio=2)),
    "mixed_reqs": ([700, 1200, 900], [6, 6, 4], dict(ratio=1)),
    "invalid_rows": ([800, 1000], [8, 8], dict(ratio=1, invalid_rows=(3, 9, 15))),
    "swa_only": ([300, 2000], [8, 8], dict(topk=0)),
}


def _unions(case):
    T = case["T"]
    sidx = case["swa_idx"].reshape(T, -1)
    s = gd.union(sidx, case["swa_lens"], gd._width(sidx.shape[1]), case["swa"].shape[0] * case["swa"].shape[1])
    e = None
    if "ex" in case:
        eidx = case["ex_glob"].reshape(T, -1)
        e = gd.union(eidx, case["ex_lens"], gd._width(eidx.shape[1]), case["ex"].shape[0] * case["ex"].shape[1])
    return s, e


def _decode(case, s, e, out, sched=None):
    T = case["T"]
    q16 = case["q"][:, : ta.HEADS].contiguous()
    su, sm, sl = s
    extra = None if e is None else (case["ex"].unsqueeze(-2), e[0], e[2], e[1])
    return gd.decode(q16, case["swa"].unsqueeze(-2), su, sl, sm, case["sink"], ta.SCALE, out, extra, sched)


def _eq_union(a, b):
    """(uidx, umask, ulen) equality where it is defined: umask / ulen everywhere, uidx where umask != 0 (the tail beyond ulen is
    torch.empty memory the union builder never writes; mask 0 = no row attends to it, the kernel treats it as invalid)."""
    ua, ma, la = a
    ub, mb, lb = b
    return torch.equal(ma, mb) and torch.equal(la, lb) and torch.equal(torch.where(ma != 0, ua, 0), torch.where(mb != 0, ub, 0))


def _eq(a, b):
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    if a.dtype == torch.float8_e4m3fn:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


@gpu
@_param("name", list(CASES))
def test_s3a_union_and_schedule_reuse(name):
    seq, ql, kw = CASES[name]
    case = ta.make_case(seq, ql, seed=41, use_stock_writer=False, **kw)
    T = case["T"]
    assert T % 4 == 0
    s0, e0 = _unions(case)
    side = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for i in range(6):  # repeated unions: bitwise equal (with and without concurrent side-stream work)
        if i % 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                a @ a
        s1, e1 = _unions(case)
        assert _eq_union(s0, s1), (name, i, "swa union")
        if e0 is not None:
            assert _eq_union(e0, e1), (name, i, "extra union")
        torch.cuda.current_stream().wait_stream(side)
    out_a = torch.empty(T, ta.HEADS, ta.DIM, dtype=torch.bfloat16, device="cuda")
    sched = _decode(case, s0, e0, out_a)  # fresh schedule (= today: every layer)
    out_b = torch.empty_like(out_a)
    sched_b = _decode(case, s0, e0, out_b, sched)  # reused schedule (= S3a: consumer layers of the index source)
    assert sched_b[0].data_ptr() == sched[0].data_ptr() and sched_b[1].data_ptr() == sched[1].data_ptr()
    assert _eq(out_a, out_b), name
    out_c = torch.empty_like(out_a)
    sched_c = _decode(case, *_unions(case), out_c)  # everything recomputed again
    # DecodingSchedMeta (FlashMLA csrc/params.h) = 7 ints + int _pad[1] that the scheduler never sets (stored from an
    # uninitialized local) and no kernel reads -> compare the 7 defined fields
    assert _eq(sched_c[0][:, :7], sched[0][:, :7]) and _eq(sched_c[1], sched[1]), name
    assert _eq(out_c, out_a), name
    # CUDA graph with one "source layer" (union + schedule) and two consumers reusing them, replayed
    outs = [torch.empty_like(out_a) for _ in range(3)]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        s, e = _unions(case)
        sc = _decode(case, s, e, outs[0])
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s, e = _unions(case)
        sc = _decode(case, s, e, outs[0])
        _decode(case, s, e, outs[1], sc)
        _decode(case, s, e, outs[2], sc)
    for _ in range(3):
        for o in outs:
            o.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert all(_eq(o, out_a) for o in outs), name


@gpu
def test_s3a_global_topk_mapping_is_pure():
    from vllm.models.deepseek_v41.nvidia.flashmla import compute_global_topk_indices_and_lens

    g = torch.Generator(device="cuda").manual_seed(7)
    T, K, R, NB, bs = 256, 512, 64, 300, 32
    local = torch.randint(0, NB * bs, (T, K), generator=g, device="cuda", dtype=torch.int32)
    local[torch.rand(T, K, generator=g, device="cuda") < 0.1] = -1
    t2r = torch.arange(T, device="cuda", dtype=torch.int32) // 4
    bt = torch.randint(0, 5000, (R, NB), generator=g, device="cuda", dtype=torch.int32)
    valid = torch.rand(T, generator=g, device="cuda") < 0.95
    ref = compute_global_topk_indices_and_lens(local, t2r, bt, bs, valid)
    for _ in range(4):
        got = compute_global_topk_indices_and_lens(local, t2r, bt, bs, valid)
        assert all(_eq(x, y) for x, y in zip(ref, got))


def _rope_cache(n, d, device):
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device, dtype=torch.float64) / d))
    t = torch.arange(n, device=device, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([t.cos(), t.sin()], dim=-1).float().contiguous()


@gpu
@_param("T", [12, 64, 68, 256, 384, 512])
def test_s3b_fused_insert_pad16_equals_pad64(T):
    import vllm._custom_ops  # noqa: F401 - registers torch.ops._C

    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    g = torch.Generator(device="cuda").manual_seed(100 + T)
    q = (torch.randn(T, 16, 512, generator=g, device="cuda") * 3).bfloat16()
    q[0, 3, :7] = torch.tensor([0.0, -0.0, 1e-30, -1e30, 65504.0, 3e38, -3e38], device="cuda").bfloat16()
    kv = (torch.randn(T, 512, generator=g, device="cuda") * 2).bfloat16()
    nb, bs = 64, 64
    cache = torch.randint(0, 256, (nb, bs, 584), generator=g, device="cuda", dtype=torch.uint8)
    slots = torch.randperm(nb * bs, generator=g, device="cuda")[:T].to(torch.int64)
    pos = torch.randint(0, 1 << 20, (T,), generator=g, device="cuda", dtype=torch.int64)
    cs = _rope_cache(1 << 20, 64, "cuda")
    ca, cb = cache.clone(), cache.clone()
    qa_in, qb_in = q.clone(), q.clone()
    qa = op(qa_in, kv.clone(), ca.view(nb, -1), slots, pos, cs, 16, 1e-20, bs, False, False, True, False)
    qb = op(qb_in, kv.clone(), cb.view(nb, -1), slots, pos, cs, 64, 1e-20, bs, False, False, True, False)
    torch.cuda.synchronize()
    assert tuple(qa.shape) == (T, 16, 512) and tuple(qb.shape) == (T, 64, 512) and qa.is_contiguous()
    assert _eq(qa, qb[:, :16].contiguous()), "q heads differ"
    assert not qb[:, 16:].float().abs().sum().item(), "padding heads not zero"
    assert torch.equal(ca, cb), "KV-cache bytes differ"
    assert _eq(qa_in, q) and _eq(qb_in, q), "q_in modified"


@gpu
@_param("quantize", [False, True])
@_param("T", [64, 256, 512])
def test_s3b_oproj_input_contiguous_equals_strided(quantize, T):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    g = torch.Generator(device="cuda").manual_seed(300 + T)
    o64 = (torch.randn(T, 64, 512, generator=g, device="cuda") * 2).bfloat16()
    view = o64[:, :16]
    contig = view.contiguous()
    pos = torch.randint(0, 1 << 20, (T,), generator=g, device="cuda", dtype=torch.int64)
    cs = _rope_cache(1 << 20, 64, "cuda")
    kw = dict(n_groups=2, heads_per_group=8, nope_dim=448, rope_dim=64, quant_group_size=128, tma_aligned_scales=False,
              quantize=quantize)
    a, sa = fused_inv_rope_fp8_quant(view, pos, cs, **kw)
    b, sb = fused_inv_rope_fp8_quant(contig, pos, cs, **kw)
    torch.cuda.synchronize()
    assert _eq(a.contiguous(), b.contiguous())
    if quantize:
        assert _eq(sa.contiguous(), sb.contiguous())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-x", "-q"]))
