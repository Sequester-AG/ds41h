"""CPU (gloo, 4 processes) logic test of ds41h/prefill_sp.py's row-split collectives: shard / reduce_scatter /
all_gather / _gather_rows / _cols_to_rows against single-process references, for T with and without tail rows.

    python tests/test_prefill_sp.py          (no GPU; spawns 4 gloo ranks)
"""

from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
W = 4


class _Comm:
    """pynccl-like shim on gloo (reduce_scatter via all_reduce + slice)."""

    disabled = False

    def reduce_scatter(self, out, inp):
        t = inp.clone()
        dist.all_reduce(t)
        c = out.shape[0]
        out.copy_(t[dist.get_rank() * c : (dist.get_rank() + 1) * c])

    def all_gather(self, out, inp):
        parts = list(out.chunk(W, dim=0))
        tmp = [torch.empty_like(inp) for _ in range(W)]
        dist.all_gather(tmp, inp.contiguous())
        for p, t in zip(parts, tmp):
            p.copy_(t)


def _install_fakes():
    import vllm.distributed as vd

    g = types.SimpleNamespace(world_size=W, rank_in_group=dist.get_rank(), device_group=dist.group.WORLD,
                              device_communicator=types.SimpleNamespace(pynccl_comm=_Comm()))
    vd.get_tp_group = lambda: g

    def ar(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    vd.tensor_model_parallel_all_reduce = ar


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=W)
    _install_fakes()
    from ds41h import prefill_sp as sp

    torch.manual_seed(0)
    for T in (16, 19, 4123, 37):
        for K in (1, 2, 3):
            rows = sp._Rows(T, K)
            full = torch.randn(T, 6)  # identical on all ranks (same seed)
            loc = rows.shard(full)
            ref = []
            for ch in rows.chunks:
                ref.append(list(range(ch.lo, ch.lo + ch.c)) + list(range(ch.body, ch.g1)))
            assert [len(r) for r in ref] == [t.shape[0] for t in loc], (T, K, "shard sizes")
            assert all(torch.equal(t, full[r]) for t, r in zip(loc, ref)), (T, K, "shard")
            assert sum(ch.g1 - ch.g0 for ch in rows.chunks) == T and rows.chunks[-1].g1 == T, (T, K, "chunks")
            assert torch.equal(rows.all_gather([t.contiguous() for t in loc]), full), (T, K, "all_gather")
            part = torch.randn(T, 5, generator=torch.Generator().manual_seed(100 + rank))  # per-rank partials
            tot = part.clone()
            dist.all_reduce(tot)
            for k in range(rows.K):
                rs, ev = rows.reduce_scatter(part, k)
                assert ev is None and torch.allclose(rs, tot[ref[k]], atol=1e-5), (T, K, k, "reduce_scatter")
                ch = rows.chunks[k]
                rs2, _ = sp._rs_rows(rows, part[ch.g0 : ch.g1].contiguous(), k)
                assert torch.allclose(rs2, tot[ref[k]], atol=1e-5), (T, K, k, "rs_rows")
            # _gather_rows: global rows -> assembled on every rank
            ridx = torch.tensor(sorted({0, T - 1, T // 2, max(0, T - 3), T // 3}), dtype=torch.long)
            a = torch.randn(T, 3, 2).to(torch.bfloat16)
            b = torch.randn(T, 4)
            ga, gb = sp._gather_rows(rows, ridx, ([t.contiguous() for t in rows.shard(a)], [t.contiguous() for t in rows.shard(b)]))
            assert torch.equal(ga, a[ridx]) and torch.equal(gb, b[ridx]), (T, K, "gather_rows")
            assert ga.is_contiguous() and gb.is_contiguous(), (T, K, "gather_rows contiguous")
            # several tensors of one dtype share an all-reduce buffer: every output must still be contiguous
            c = torch.randn(T, 4, 5).to(torch.bfloat16)
            ga2, gc2, gb2 = sp._gather_rows(rows, ridx, tuple([t.contiguous() for t in rows.shard(z)] for z in (a, c, b)))
            assert torch.equal(ga2, a[ridx]) and torch.equal(gc2, c[ridx]) and torch.equal(gb2, b[ridx]), (T, K, "gather_rows 3")
            assert all(t.is_contiguous() for t in (ga2, gc2, gb2)), (T, K, "gather_rows 3 contiguous")
            # _cols_to_rows: rank r holds columns [r N, (r + 1) N) of all rows
            N = 3
            fullc = torch.randn(T, W * N)
            mine = fullc[:, rank * N : (rank + 1) * N].contiguous()
            got = sp._cols_to_rows(mine, rows)
            assert all(torch.equal(g_, fullc[r]) for g_, r in zip(got, ref)), (T, K, "cols_to_rows")
    # _projections (attention input projections on local rows + gathers) against the same row-wise ops on all rows
    torch.manual_seed(1)
    H, NQ, NKV = 8, 5, 3
    Wq = torch.randn(H, NQ + NKV)
    Wc = torch.randn(4, H)
    Wi = torch.randn(H, 2)

    class _Mod:
        pass

    _mm = torch.mm
    torch.mm = lambda a, b, out_dtype=None: _mm(a, b).to(out_dtype or a.dtype)  # mm(out_dtype=) is CUDA-only
    attn = _Mod()
    attn.fused_wqa_wkv = _Mod()
    attn._fused_wqa_wkv_gemm = lambda x: x @ Wq
    attn._split_qkv_and_norm = lambda qk: ((qk[:, :NQ] * 2).contiguous(), None, (qk[:, NQ:] * 3).contiguous())
    attn.compressor = _Mod()
    attn.compressor.fused_wkv_wgate = _Mod()
    attn.compressor.fused_wkv_wgate.weight = Wc
    attn.indexer = _Mod()
    attn.indexer.weights_proj = lambda x: (x @ Wi, None)
    for T in (37, 4123):
        for K in (1, 2):
            rows = sp._Rows(T, K)
            x = torch.randn(T, H)
            for norm_local in (True, False):
                sp._NORM_LOCAL = norm_local
                qr, qs, kv, ks, iw = sp._projections(attn, [t.contiguous() for t in rows.shard(x)], rows)
                rq, _, rkv = attn._split_qkv_and_norm(x @ Wq)
                assert qs is None and torch.allclose(qr, rq) and torch.allclose(kv, rkv), (T, K, norm_local, "q/kv")
                assert torch.allclose(ks, x @ Wc.T) and torch.allclose(iw, x @ Wi), (T, K, norm_local, "score/idx")
    # _attention: wo_b per row chunk via the instance _wo_b_proj override, each chunk reduce-scattered
    class _WoB:
        reduce_results = True

        def __init__(self, W):
            self.W = W

        def __call__(self, z):
            assert self.reduce_results is False
            return z @ self.W

    Wb = torch.randn(NQ + NKV, 6, generator=torch.Generator().manual_seed(7 + rank))  # rank-specific TP partial
    attn.wo_b = _WoB(Wb)
    attn._alloc_attn_out = lambda T_, hs: torch.empty(T_, NQ + NKV)
    attn._prepare_and_attn_fn = lambda hs, qr, kv, qs, ks, iw, pos, out: out.copy_(torch.cat([qr, kv], 1))
    attn._o_proj = lambda out, pos: attn._wo_b_proj(out * 1.0)
    attn._wo_b_proj = None  # class-level stand-in; prefill_sp sets the instance attribute

    for T in (37, 4123):
        for K in (1, 2, 3):
            rows = sp._Rows(T, K)
            x = torch.randn(T, H)
            sp._NORM_LOCAL = True
            got = sp._attention(attn, [t.contiguous() for t in rows.shard(x)], torch.arange(T), rows)
            rq, _, rkv = attn._split_qkv_and_norm(x @ Wq)
            tot = torch.cat([rq, rkv], 1) @ Wb
            dist.all_reduce(tot)
            ref = rows.shard(tot)
            assert len(got) == rows.K and all(torch.allclose(g[0], r, atol=1e-4) for g, r in zip(got, ref)), (T, K, "attention")
            assert attn.wo_b.reduce_results is True and attn.__dict__.get("_wo_b_proj") is None
    if rank == 0:
        print("test_prefill_sp: all row-split collectives OK")
    dist.destroy_process_group()


def test_row_split_collectives():
    os.environ.setdefault("DS41H", "1")
    os.environ.setdefault("DS41H_PREFILL_SP", "1")
    mp.spawn(_worker, args=(29571,), nprocs=W, join=True)


def test_ctx_trim_matches_stock_math():
    """ds41h/prefill_ctx.py (DS41H_PREFILL_CTX_TRIM): the DSpark context work on the replay rows only gives the same
    main_x and the same inserted context KV as the stock math over all rows when the other rows' aux inputs are zero
    (integer-valued fp32 data: every sum is exact, so any summation order gives identical bits)."""
    from ds41h import prefill_ctx

    torch.manual_seed(0)
    H, Q, D, L, T = 8, 6, 4, 3, 40

    class RMS(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randint(-3, 4, (n,)).float())

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    def lin(i, o):
        m = torch.nn.Linear(i, o, bias=False)
        m.weight.data = torch.randint(-3, 4, (o, i)).float()
        return m

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lora_rank = Q
            self.wqa_wkv = lin(H, Q + D)
            self.kv_norm = RMS(D)

        def fused_wqa_wkv(self, x):
            return self.wqa_wkv(x), None

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    inserted: list = []

    def _insert_context_kv(attn, kv, positions, slot_mapping):
        inserted.append(kv.clone())

    class DSparkDeepseekV4Model(torch.nn.Module):  # the stock math of vllm/models/deepseek_v41/nvidia/dspark.py
        def __init__(self):
            super().__init__()
            self.main_proj = lin(3 * H, H)
            self.main_norm = RMS(H)
            self.layers = torch.nn.ModuleList(Layer() for _ in range(L))

        def combine_hidden_states(self, aux_hidden_states):
            return self.main_norm(self.main_proj(aux_hidden_states))

        def precompute_and_store_context_kv(self, main_x, context_positions, context_slot_mappings=None):
            for i, layer in enumerate(self.layers):
                slot_mapping = None if context_slot_mappings is None else context_slot_mappings[i]
                attn = layer.attn
                qr_kv, _ = attn.fused_wqa_wkv(main_x)
                kv = attn.kv_norm(qr_kv[..., attn.q_lora_rank :])
                if slot_mapping is None:
                    continue
                _insert_context_kv(attn, kv, context_positions, slot_mapping)

    mod = types.SimpleNamespace(DSparkDeepseekV4Model=DSparkDeepseekV4Model, _insert_context_kv=_insert_context_kv)
    m = DSparkDeepseekV4Model()
    rows = torch.tensor([3, 4, 5, 20, 37, 38, 39])
    aux = torch.zeros(T, 3 * H)
    aux[rows] = torch.randint(-4, 5, (rows.numel(), 3 * H)).float()
    pos, sm = torch.arange(T), [torch.arange(T)] * L
    start = 6  # DFlash propose with a FULL drafter graph: rows [start, T) eagerly; replay rows 3, 4, 5 lie before the slice
    with torch.no_grad():
        ref_x = m.combine_hidden_states(aux)  # stock math (before the patch)
        m.precompute_and_store_context_kv(ref_x, pos, sm)
        ref_kv, inserted[:] = list(inserted), []
        m.precompute_and_store_context_kv(ref_x[start:], pos[start:], sm)
        ref_kv_s, inserted[:] = list(inserted), []
        prefill_ctx.patch_dspark(mod)
        prefill_ctx.CTX_TRIM, prefill_ctx.MIN_T, prefill_ctx.INSTALL = True, 8, True
        target = types.SimpleNamespace(aux_hidden_state_layers=[20, 30, 42], decoder_replay_start=20)
        prefill_ctx.note_replay(target, rows, T)
        s0 = dict(prefill_ctx.STATS)
        x = m.combine_hidden_states(aux)
        m.precompute_and_store_context_kv(x, pos, sm)
        assert prefill_ctx.STATS["combine"] == s0["combine"] + 1 and prefill_ctx.STATS["kv"] == s0["kv"] + 1, "trim not used"
        assert torch.equal(x, ref_x), "main_x differs"
        assert len(inserted) == L and all(torch.equal(a, b) for a, b in zip(inserted, ref_kv)), "context KV differs"
        inserted.clear()
        m.precompute_and_store_context_kv(x[start:], pos[start:], sm)
        assert prefill_ctx.STATS["kv"] == s0["kv"] + 2, "suffix slice not trimmed"
        assert len(inserted) == L and all(torch.equal(a, b) for a, b in zip(inserted, ref_kv_s)), "suffix-slice context KV differs"
        # a stale early main_proj result of another step is dropped (as the drafter's combine does); a matching one defers
        m._ds41h_pending = {"rows": 3, "cols": 3 * H}
        assert torch.equal(m.combine_hidden_states(aux), ref_x) and "_ds41h_pending" not in m.__dict__
        assert prefill_ctx.STATS["combine"] == s0["combine"] + 2, "stale pending must not block the trim"
        m._ds41h_pending = {"rows": T, "cols": 3 * H}
        m.combine_hidden_states(aux)
        assert prefill_ctx.STATS["combine"] == s0["combine"] + 2, "matching pending must defer to the drafter's combine"
        m.__dict__.pop("_ds41h_pending", None)
        # not applied: wrong T, cleared note (next target forward), aux layer below the cut, switched off
        s1 = prefill_ctx.STATS["combine"]
        assert m.combine_hidden_states(aux[:-1]).shape[0] == T - 1 and prefill_ctx.STATS["combine"] == s1
        prefill_ctx.begin_forward()
        m.combine_hidden_states(aux)
        assert prefill_ctx.STATS["combine"] == s1
        bad = types.SimpleNamespace(aux_hidden_state_layers=[19, 30, 42], decoder_replay_start=20)
        prefill_ctx.note_replay(bad, rows, T)
        m.combine_hidden_states(aux)
        assert prefill_ctx.STATS["combine"] == s1
        prefill_ctx.note_replay(target, rows, T)
        prefill_ctx.CTX_TRIM = False
        m.combine_hidden_states(aux)
        assert prefill_ctx.STATS["combine"] == s1
    print("test_ctx_trim_matches_stock_math: OK")


if __name__ == "__main__":
    test_ctx_trim_matches_stock_math()
    test_row_split_collectives()
