"""GPU tests + microbenchmark for ds41h ENGRAM_HOST (ds41h/engram_host.py, kernels/engram_host.cu,
ds41h/patch_engram.py) on SYNTHETIC tables (the real ~200 GB tables need that much free in /dev/shm).

Run on ONE GPU (DS41H unset, so the vLLM functions compared against are stock):
    python tests/test_engram_host.py              # all tests (1 GPU)
    python tests/test_engram_host.py --bench [--gb 8] [--backends shm,thp,pinned,device]
    python tests/test_engram_host.py --bench4 [--gb 64]   # 4 GPUs share one table
    python tests/test_engram_host.py --tp4                # patched Engram at TP=4 vs stock
    DS41H_ENGRAM_REAL_GB=101 ... python tests/test_engram_host.py real              # whole real layer-1 table

Reference = stock vLLM 0.32.0 `_engram_lookup_kernel` (vllm/models/deepseek_v41/common/engram.py:639-721, non-packed,
unsorted: the sm90 path) run per TP rank on a GPU-resident copy of the same table, concatenated over the ranks like
`tensor_model_parallel_all_gather(rows, dim=1)` (engram.py:1673). Every comparison is BIT-EXACT (bf16 bits).
Tables live under DS41H_ENGRAM_HOST_DIR (default here: /dev/shm/ds41h_engram_test_<pid>, removed at the end).
"""

from __future__ import annotations

import ctypes
import mmap
import os
import shutil
import statistics
import sys
import time

import torch

from common import require_cuda, run_module_tests  # noqa: E402  (adds the repo to sys.path)

TEST_DIR = os.environ.setdefault("DS41H_ENGRAM_HOST_DIR", f"/dev/shm/ds41h_engram_test_{os.getpid()}")
TP = 4


def _eh():
    from ds41h import engram_host

    return engram_host


# ---------------------------------------------------------------------------------------- synthetic tables
def synthetic_layout(n_cols: int = 24, base: int = 40_000, seed: int = 0) -> list[int]:
    """Prime-sized head buckets like EngramLayout (smaller): 24 distinct primes above `base`."""
    from ds41h.ref.weights import _is_prime

    sizes, c = [], base + seed * 997
    while len(sizes) < n_cols:
        c += 1
        if _is_prime(c):
            sizes.append(c)
    return sizes


def synthetic_table(rows: int, device: str = "cuda", seed: int = 0, all_codes: bool = True):
    """(weight uint8 [rows, 256], scale uint8 [rows, 8]) on `device`. all_codes: every e4m3 byte (incl. NaN 0x7f/0xff,
    subnormals) appears in every column position and every ue8m0 byte 0..255 (0 -> scale 0, 255 -> inf, tiny scales
    -> fp32 subnormal products) appears as a scale."""
    g = torch.Generator(device=device).manual_seed(seed)
    if all_codes:
        r = torch.arange(rows, device=device, dtype=torch.int64)
        c = torch.arange(256, device=device, dtype=torch.int64)
        w = ((r[:, None] * 7 + c[None, :] * 13 + (r[:, None] // 256)) % 256).to(torch.uint8)
        j = torch.arange(8, device=device, dtype=torch.int64)
        s = ((r[:, None] + 37 * j[None, :] + (r[:, None] // 251)) % 256).to(torch.uint8)
    else:
        w = torch.randint(0, 256, (rows, 256), device=device, dtype=torch.uint8, generator=g)
        w[(w == 0x7F) | (w == 0xFF)] = 0x3C  # no NaN codes
        s = torch.randint(110, 140, (rows, 8), device=device, dtype=torch.uint8, generator=g)
    return w, s


def head_ids(T: int, sizes: list[int], device="cuda", seed: int = 0, n_layers: int = 2, layer: int = 0):
    """Random valid hash ids [T, n_layers, 24] (column h in head h's bucket range); returns the [T, 24] view of
    `layer` like model.py passes `gathered_hashes[:, layer]`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    offs = [sum(sizes[:h]) for h in range(len(sizes))]
    full = torch.empty(T, n_layers, len(sizes), dtype=torch.int32, device="cpu")
    for h, (o, n) in enumerate(zip(offs, sizes)):
        full[:, :, h] = torch.randint(0, n, (T, n_layers), generator=g, dtype=torch.int64, device="cpu").to(torch.int32) + o
    return full.to(device)[:, layer]


# ---------------------------------------------------------------------------------------- stock reference
def _stock():
    try:
        from vllm.models.deepseek_v41.common import engram as eg
    except Exception as e:  # noqa: BLE001
        from common import skip

        skip(f"vllm engram module not importable: {type(e).__name__}: {e}")
    return eg


def stock_rows(weight: torch.Tensor, scale: torch.Tensor, ids: torch.Tensor, sizes: list[int], tp: int = TP):
    """Stock lookup on every TP rank (GPU-resident shard) + all-gather over heads -> bf16 [T, n_cols, 256]."""
    eg = _stock()
    from vllm.triton_utils import triton

    T, n = ids.shape
    part = triton.cdiv(n, tp)
    sms = torch.cuda.get_device_properties(ids.device).multi_processor_count
    outs = []
    for r in range(tp):
        hs = r * part
        vs, ve = sum(sizes[:hs]), sum(sizes[: hs + part])
        w = weight[vs:ve].view(torch.float8_e4m3fn)
        s = scale[vs:ve]
        out = torch.empty(T, part, 256, dtype=torch.bfloat16, device=ids.device)
        rows = T * part
        grid = max(1, min(triton.cdiv(rows, 16), sms))
        eg._engram_lookup_kernel[(grid,)](
            w, s, ids, ids, out, vs, ve, rows, ids.stride(0), ids.stride(1),
            HEAD_START=hs, LOCAL_HEADS=part, TOTAL_HEADS=n, DIM=256, QUANT_BLOCK=32, BLOCK_R=16, GRID=grid,
            SORTED=False, PACKED=False, OUT_STRIDE=out.stride(0),
        )  # fmt: skip
        outs.append(out)
    return torch.cat(outs, 1)[:, :n]


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    ai, bi = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    bad = (ai != bi).nonzero()
    if bad.numel() == 0:
        return True, "bit-identical"
    i = tuple(bad[0].tolist())
    return False, f"{bad.shape[0]} differing elements, first at {i}: {int(ai[i]) & 0xffff:#06x} vs {int(bi[i]) & 0xffff:#06x}"


def make_table(name: str, rows: int, n_shards: int = TP, layout: str = "split", keep: bool = False):
    eh = _eh()
    key = f"test {name} rows={rows} pid={os.getpid()}"
    return eh.HostTable(eh.table_path(key, 0, layout), key, rows, n_shards=n_shards, layout=layout, keep=keep)


def _filled_table(name, w, s, layout="split"):
    t = make_table(name, w.shape[0], layout=layout)
    t.fill_from(w, s)
    for r in range(TP):
        t.set_state(r, "weight", 2)
        t.set_state(r, "scale", 2)
    t.register()
    return t


# ======================================================================================== tests
def test_bitexact_all_codes():
    """Every e4m3 code x every ue8m0 scale byte, all 24 columns: host gather == stock lookup + all-gather, bitwise."""
    require_cuda()
    eh = _eh()
    sizes = synthetic_layout()
    rows = sum(sizes) + 1000  # num_embeddings may exceed the bucket sum (stock asserts <=)
    w, s = synthetic_table(rows)
    for layout in ("split", "packed"):
        t = _filled_table(f"codes_{layout}", w, s, layout)
        lo, hi = eh.head_ranges(sizes, TP)
        for T in (1, 6, 64, 3000):
            ids = head_ids(T, sizes, seed=T)
            ref = stock_rows(w, s, ids, sizes)
            out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
            t.gather(ids, out, lo, hi)
            torch.cuda.synchronize()
            ok, msg = bits_equal(out, ref)
            assert ok, f"{layout} T={T}: {msg}"
        # sweep: every table row exactly once, in a column of its owning rank -> every (code, scale) pair goes
        # through the kernel and lands in the output (stock zeroes ids outside the owning rank's rows)
        part = -(-24 // TP)
        spans = [eh.shard_rows(sizes, TP, r) for r in range(TP)]
        T = max(-(-(ve - vs) // part) for vs, ve in spans)
        ids = torch.full((T, 24), -1, dtype=torch.int32)
        for r, (vs, ve) in enumerate(spans):
            block = torch.full((T * part,), -1, dtype=torch.int64)
            block[: ve - vs] = torch.arange(vs, ve)
            ids[:, r * part : (r + 1) * part] = block.view(T, part).to(torch.int32)
        ids = ids.cuda()
        out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
        ref = stock_rows(w, s, ids, sizes)
        t.gather(ids, out, lo, hi)
        torch.cuda.synchronize()
        ok, msg = bits_equal(out, ref)
        assert ok, f"{layout} sweep: {msg}"
        nan = torch.isnan(out.float()).sum().item()
        print(f"  {layout}: all-codes sweep {T * 24} rows bit-identical (incl. {nan} NaN, "
              f"{torch.isinf(out.float()).sum().item()} inf elements)")
        t.close()


def test_ownership_semantics():
    """Invalid / foreign ids give exactly what stock gives (zeros from the owning rank)."""
    require_cuda()
    eh = _eh()
    sizes = synthetic_layout(seed=1)
    rows = sum(sizes)
    w, s = synthetic_table(rows, all_codes=False, seed=3)
    t = _filled_table("own", w, s)
    lo, hi = eh.head_ranges(sizes, TP)
    T = 257
    ids = head_ids(T, sizes, seed=5).clone()
    g = torch.Generator(device="cpu").manual_seed(9)
    sel = torch.randint(0, 4, (T, 24), generator=g)
    junk = torch.stack([torch.full((T, 24), -1), torch.full((T, 24), rows + 5),
                        torch.randint(0, rows, (T, 24), generator=g)], 0)  # dead, out of table, any head's range
    ids_cpu = ids.cpu()
    for k in range(3):
        ids_cpu = torch.where(sel == k + 1, junk[k].to(torch.int32), ids_cpu)
    ids = ids_cpu.cuda()
    ref = stock_rows(w, s, ids, sizes)
    out = torch.full((T, 24, 256), 7.0, dtype=torch.bfloat16, device="cuda")
    t.gather(ids, out, lo, hi)
    torch.cuda.synchronize()
    ok, msg = bits_equal(out, ref)
    assert ok, msg
    zero_rows = (out.float().abs().sum(-1) == 0).sum().item()
    print(f"  {zero_rows} of {T * 24} rows zero (foreign / dead ids), all bit-identical to stock")
    t.close()


def test_launch_configs():
    """Decode and prefill shapes, every launch variant (ctas / threads / unroll / 8-byte loads) bit-identical."""
    require_cuda()
    eh = _eh()
    sizes = synthetic_layout(seed=2)
    rows = sum(sizes)
    w, s = synthetic_table(rows, all_codes=False, seed=4)
    t = _filled_table("cfg", w, s, "packed")
    lo, hi = eh.head_ranges(sizes, TP)
    g = t.geom
    for T in (1, 6, 8, 48, 1000, 16896):
        ids = head_ids(T, sizes, seed=T + 11)
        ref = stock_rows(w, s, ids, sizes)
        for ctas, threads, unroll, vec in ((None, None, None, None), (1, 1024, 8, 16), (3, 256, 2, 16),
                                           (132, 128, 1, 16), (7, 512, 4, 8)):
            out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
            eh.gather_rows(t.dev_base + g.w_off, t.dev_base + g.s_off, g.w_stride, g.s_stride, ids, out, lo, hi,
                           ctas=ctas, threads=threads, unroll=unroll, vec=vec)
            torch.cuda.synchronize()
            ok, msg = bits_equal(out, ref)
            assert ok, f"T={T} cfg={(ctas, threads, unroll, vec)}: {msg}"
    t.close()


def test_shard_gather():
    """Prefill mode (T > DS41H_ENGRAM_HOST_LOCAL_MAX): each rank gathers only its head shard from host memory; the
    shard must equal stock's per-rank lookup bit for bit (the stock TP all-gather then assembles identical rows)."""
    require_cuda()
    eh = _eh()
    from ds41h.patch_engram import HostLayer

    sizes = synthetic_layout(seed=5)
    rows = sum(sizes)
    w, s = synthetic_table(rows, all_codes=True)
    t = _filled_table("shard", w, s)
    lo, hi = eh.head_ranges(sizes, TP)
    T = 2000
    ids = head_ids(T, sizes, seed=17)
    ref = stock_rows(w, s, ids, sizes)  # [T, 24, 256] = cat over ranks of [T, 6, 256]
    part = 24 // TP
    for r in range(TP):
        host = HostLayer(t, lo, hi, 0, head_start=r * part, part=part)
        out = torch.empty(T, part, 256, dtype=torch.bfloat16, device="cuda")
        host.gather_shard(ids, out)
        torch.cuda.synchronize()
        ok, msg = bits_equal(out, ref[:, r * part : (r + 1) * part])
        assert ok, f"rank {r}: {msg}"
    t.close()


def test_graph_capture():
    """The patch's fork/join pattern (side stream + event) captured in a CUDA graph; replays with new ids."""
    require_cuda()
    eh = _eh()
    from ds41h.patch_engram import HostLayer

    sizes = synthetic_layout(seed=3)
    rows = sum(sizes)
    w, s = synthetic_table(rows, all_codes=False, seed=6)
    t = _filled_table("graph", w, s)
    lo, hi = eh.head_ranges(sizes, TP)
    host = HostLayer(t, lo, hi, 0)
    T = 6
    hashes = torch.zeros(T, 2, 24, dtype=torch.int32, device="cuda")  # static input buffer (like the model's)
    staged = torch.empty(64, 24, 256, dtype=torch.bfloat16, device="cuda")
    consumer = torch.empty(T, 24 * 256, dtype=torch.float32, device="cuda")
    side = torch.cuda.Stream(priority=-1)
    done = torch.cuda.Event()

    def step():
        ids = hashes[:, 1]
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            host.gather(ids, staged[:T])
        ids.record_stream(side)
        done.record(side)
        torch.cuda.current_stream().wait_event(done)  # Engram.embed
        consumer.copy_(staged[:T].flatten(-2).float() * 2)  # a downstream consumer (wkv stand-in)

    # warm up eagerly, then capture
    hashes.copy_(torch.stack([head_ids(T, sizes, seed=1, layer=k) for k in range(2)], 1))
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for it in range(5):
        new = torch.stack([head_ids(T, sizes, seed=100 + it, layer=k) for k in range(2)], 1)
        hashes.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        ref = stock_rows(w, s, new[:, 1].contiguous(), sizes)
        ok, msg = bits_equal(staged[:T], ref)
        assert ok, f"replay {it}: {msg}"
        assert torch.equal(consumer, ref.flatten(-2).float() * 2)
    t.close()


def _mp_worker(rank: int, n: int, name: str, rows: int, sizes: list[int], seed: int, q) -> None:
    try:
        os.environ["DS41H_ENGRAM_HOST_DIR"] = TEST_DIR
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        torch.cuda.set_device(0)
        eh = _eh()
        w, s = synthetic_table(rows, device="cpu", all_codes=False, seed=seed)
        key = f"test mp {name} rows={rows}"
        t = eh.HostTable(eh.table_path(key, 0, "split"), key, rows, n_shards=n, layout="split", keep=False)
        t.register()
        lo_r, hi_r = eh.shard_rows(sizes, n, rank)
        wv, sv = t.views(lo_r, hi_r)  # like the patched loader: the rank writes ITS rows into the shared mapping
        t.set_state(rank, "weight", 1)
        wv.copy_(w[lo_r:hi_r].view(torch.float8_e4m3fn))
        t.set_state(rank, "weight", 2)
        t.set_state(rank, "scale", 1)
        sv.copy_(s[lo_r:hi_r])
        t.set_state(rank, "scale", 2)
        t.wait_ready(timeout=120)
        lo, hi = eh.head_ranges(sizes, n)
        ids = head_ids(512, sizes, seed=rank + 1)
        out = torch.empty(512, 24, 256, dtype=torch.bfloat16, device="cuda")
        t.gather(ids, out, lo, hi)
        torch.cuda.synchronize()
        ref = stock_rows(w.cuda(), s.cuda(), ids, sizes, tp=n)
        ok, msg = bits_equal(out, ref)
        q.put((rank, ok, msg, str(t.path), t.created))
        t.close()
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", "", False))


def test_multiprocess_shared_table():
    """4 processes (TP-rank stand-ins, same GPU) share ONE file: each loads its shard, all read everything."""
    require_cuda()
    import torch.multiprocessing as mp

    sizes = synthetic_layout(seed=4)
    rows = sum(sizes)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_mp_worker, args=(r, TP, "shared", rows, sizes, 21, q)) for r in range(TP)]
    for p in procs:
        p.start()
    res = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join(60)
    res.sort()
    for r, ok, msg, path, created in res:
        print(f"  rank {r}: {msg} (file {os.path.basename(path)}, created={created})")
    assert all(ok for _, ok, *_ in res), [m for _, ok, m, *_ in res if not ok]
    assert len({p for *_, p, _ in res}) == 1, "ranks mapped different files"
    assert sum(c for *_, c in res) == 1, "exactly one rank must create the table"
    assert not os.path.exists(res[0][3]), "KEEP=0: the last user must unlink the table"


def test_patched_engram_module():
    """Stock Engram (vLLM 0.32.0, TP=1) with the ds41h host table vs the stock GPU table: identical forward output."""
    require_cuda()
    eg = _stock()
    from ds41h import patch_engram

    try:
        import tempfile

        from vllm.config import EngramConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps

        vcfg = VllmConfig()  # (engram_config without a model_config fails VllmConfig validation: set it afterwards)
        vcfg.engram_config = EngramConfig(cpu_offload=False, dp_shared_memory=False)
        vcfg.scheduler_config.max_num_batched_tokens = 512
        with set_current_vllm_config(vcfg):
            if not ps.model_parallel_is_initialized():
                port = 29500 + os.getpid() % 2000
                ps.init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                                distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
                ps.initialize_model_parallel(1, 1)
    except Exception as e:  # noqa: BLE001
        from common import skip

        skip(f"vLLM single-rank setup failed: {type(e).__name__}: {e}")
    patch_engram.patch_engram_module(eg)  # class-level patch (idempotent); per-instance host/stock choice below
    dtype0 = torch.get_default_dtype()
    try:
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)  # vLLM builds the model under the model dtype (wkv params bf16)
        _patched_engram_body(eg, vcfg)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(dtype0)


def _patched_engram_body(eg, vcfg):
    from vllm.config import set_current_vllm_config

    class Cfg:  # the Engram constructor reads these
        hidden_size = 512
        hc_mult = 4
        rms_norm_eps = 1e-6
        engram_layer_ids = [1, 14]
        engram_num_embeddings = [0, 0]
        engram_max_ngram_size = 4
        engram_n_heads = 8
        engram_head_dim = 256
        engram_compressed_vocab_size = 1000
        engram_pad_token_id = 2
        engram_vocab_size = 40_000

    cfg = Cfg()
    with set_current_vllm_config(vcfg):
        layout = eg.EngramLayout(cfg)
        sizes = [p for order in layout.primes[0] for p in order]
        cfg.engram_num_embeddings = [sum(sizes) + 17, sum(sizes) + 17]
        layout = eg.EngramLayout(cfg)
        torch.manual_seed(0)
        mods = {}
        for host in (False, True):
            if host:
                m = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
            else:  # stock storage: construct the table without the ds41h context (plain stock allocation)
                orig = eg.Engram._create_embedding
                eg.Engram._create_embedding = lambda self, lay, i: eg.ParallelEngramEmbedding(
                    lay.num_embeddings[i], lay.head_dim, tuple(p for o in lay.primes[i] for p in o))
                try:
                    m = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
                finally:
                    eg.Engram._create_embedding = orig
            mods[host] = m
        assert getattr(mods[True].embed_tokens, "_ds41h_host", None) is not None
        assert getattr(mods[False].embed_tokens, "_ds41h_host", None) is None
        assert not mods[True].embed_tokens.weight.is_cuda and mods[False].embed_tokens.weight.is_cuda
        rows = layout.num_embeddings[0]
        w, s = synthetic_table(rows, device="cpu", all_codes=True)
        wkv = (torch.randn(cfg.hidden_size * 5, 24 * 256, device="cpu") * 0.02).to(torch.bfloat16)
        q_w = torch.randn(4, cfg.hidden_size, device="cpu").to(torch.bfloat16)
        k_w = torch.randn(4, cfg.hidden_size, device="cpu").to(torch.bfloat16)
        for m in mods.values():
            et = m.embed_tokens
            et.weight.weight_loader(et.weight, w.view(torch.float8_e4m3fn))
            et.weight_scale_inv.weight_loader(et.weight_scale_inv, s.view(torch.float8_e8m0fnu))
            m.wkv.weight.data.copy_(wkv)
            m.q_weight.data.copy_(q_w)
            m.k_weight.data.copy_(k_w)
        T = 300
        hashes = torch.stack([head_ids(T, sizes, seed=7, layer=k) for k in range(2)], 1)
        X = (torch.randn(T, 4, cfg.hidden_size) * 0.5).to(torch.bfloat16)
        mask = torch.rand(T) > 0.1
        outs = {}
        for host, m in mods.items():
            m.prepare_embeddings(hashes[:, 0])
            outs[host] = m(X, hashes[:, 0], mask)
        torch.cuda.synchronize()
        # a second host module with the same checkpoint key (= a restart with DS41H_ENGRAM_HOST_KEEP=1, or another
        # process): it opens the existing file, finds its shard DONE and its loader must not copy (a poisoned tensor
        # would zero the table); its lookups read the shared table
        from ds41h import patch_engram

        early0 = patch_engram.EARLY_WKV
        patch_engram.EARLY_WKV = True  # the second module also exercises the side-stream wkv (opt-in)
        try:
            m2 = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
        finally:
            patch_engram.EARLY_WKV = early0
        assert getattr(m2.wkv, "_ds41h_early_installed", False)
        t2 = m2.embed_tokens._ds41h_host.table
        assert not t2.created and t2.shard_done(0), (t2.created, t2.shard_done(0))
        et = m2.embed_tokens
        et.weight.weight_loader(et.weight, torch.zeros_like(w).view(torch.float8_e4m3fn))
        et.weight_scale_inv.weight_loader(et.weight_scale_inv, torch.zeros_like(s).view(torch.float8_e8m0fnu))
        m2.wkv.weight.data.copy_(wkv)
        m2.q_weight.data.copy_(q_w)
        m2.k_weight.data.copy_(k_w)
        patch_engram.EARLY_WKV = True
        try:
            m2.prepare_embeddings(hashes[:, 0])
            assert m2._ds41h_early is not None, "early wkv not launched"
            out2 = m2(X, hashes[:, 0], mask)
            assert m2._ds41h_early is None, "early wkv product not consumed"
        finally:
            patch_engram.EARLY_WKV = early0
        torch.cuda.synchronize()
    ok, msg = bits_equal(outs[True], outs[False])
    assert ok, f"Engram.forward host vs stock: {msg}"
    ok2, msg2 = bits_equal(out2, outs[False])
    assert ok2, f"second opener (cached shard) vs stock: {msg2}"
    print(f"  Engram.forward [T={T}, 4, {cfg.hidden_size}]: host table == stock GPU table, {msg}; second opener "
          f"(shard cached, loader skipped, EARLY_WKV side-stream wkv): {msg2}")
    t2.close()
    mods[True].embed_tokens._ds41h_host.table.close()


def test_real_rows():
    """REAL checkpoint bytes (layer 1 table, e4m3 + ue8m0) through the host gather vs the stock kernel, bit-exact.

    DS41H_ENGRAM_REAL_GB (default 4): size of the window of real rows (host table + GPU copy). >= the whole table
    (~95 GiB) uses the real head-bucket layout and stock ownership ranges; a smaller window is
    treated as a table of its own (24 equal column buckets over the window) -- the ownership logic is covered by
    test_ownership_semantics, this test is about the real data."""
    require_cuda()
    eh = _eh()
    from common import reader

    from ds41h import ref

    rd = reader()
    name_w, name_s = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    iw, isc = rd.info(name_w), rd.info(name_s)
    total = iw.shape[0]
    gb = float(os.environ.get("DS41H_ENGRAM_REAL_GB", "4"))
    n = min(total, int(gb * 2**30 // 264))
    full = n == total
    if full:
        sizes = list(ref.engram_layout(rd.config).head_sizes(0))
    else:
        sizes = [n // 24] * 23 + [n - 23 * (n // 24)]
    avail = _eh().meminfo().get("MemAvailable", 0)
    if avail and avail < n * 264 + (12 << 30):
        from common import skip

        skip(f"MemAvailable {avail / 2**30:.0f} GiB too small for a {n * 264 / 2**30:.1f} GiB real-row window")
    a = 0 if full else int(os.environ.get("DS41H_ENGRAM_REAL_START", str(total // 3)))  # a window inside head 8
    t0 = time.perf_counter()
    t = make_table("real", n, layout=eh.default_layout())
    t.populate()
    t.register()
    wdev = torch.empty(n, 256, dtype=torch.uint8, device="cuda")
    sdev = torch.empty(n, 8, dtype=torch.uint8, device="cuda")
    chunk = 1 << 22  # 1 GiB of rows per read
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        w = rd.get_rows(name_w, a + r0, a + r1).view(torch.uint8)
        s = rd.get_rows(name_s, a + r0, a + r1).view(torch.uint8)
        t.fill_from(w, s, r0)
        wdev[r0:r1].copy_(w)
        sdev[r0:r1].copy_(s)
        del w, s
    for r in range(TP):
        t.set_state(r, "weight", 2)
        t.set_state(r, "scale", 2)
    t_load = time.perf_counter() - t0
    lo, hi = eh.head_ranges(sizes, TP)
    hist = torch.bincount(sdev[: min(n, 1 << 22)].flatten().long(), minlength=256)
    used = hist.nonzero().flatten().tolist()
    print(f"  real layer-1 rows [{a}, {a + n}) = {n * 264 / 2**30:.1f} GiB loaded in {t_load:.0f} s; scale bytes in use "
          f"{used[0]}..{used[-1]} ({len(used)} values), e4m3 NaN codes in window: "
          f"{int(((wdev[: 1 << 20] == 0x7F) | (wdev[: 1 << 20] == 0xFF)).sum())} (first 1M rows)")
    for T in (6, 16384):
        ids = head_ids(T, sizes, seed=T + 3)
        ref_rows = stock_rows(wdev, sdev, ids, sizes)
        out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
        t.gather(ids, out, lo, hi)
        torch.cuda.synchronize()
        ok, msg = bits_equal(out, ref_rows)
        assert ok, f"T={T}: {msg}"
        print(f"  T={T}: host gather == stock lookup on the GPU copy ({msg}); nonzero rows "
              f"{int((out.float().abs().sum(-1) > 0).sum())}/{T * 24}")
    del wdev, sdev
    t.close()


def test_register_memlock():
    """cudaHostRegister of a >= 1 GiB shared mapping works under the current RLIMIT_MEMLOCK (report it)."""
    require_cuda()
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    rows = (1 << 30) // 264 + 1
    t = make_table("memlock", rows)
    tp = t.populate()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    tr = t.register()
    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info()[0]
    gib = t.geom.total / 2**30
    lim = "unlimited" if soft == resource.RLIM_INFINITY else f"{soft / 2**20:.0f} MiB"
    print(f"  RLIMIT_MEMLOCK soft={lim}: registered {gib:.2f} GiB (populate {tp:.2f} s, cudaHostRegister {tr:.2f} s = "
          f"{tr / gib:.2f} s/GiB); device memory used by the mapping (GPU page tables): "
          f"{(free0 - free1) / 2**20:.1f} MiB = {(free0 - free1) / 2**20 / gib:.2f} MiB per GiB registered")
    t.close()


# ======================================================================================== microbenchmark
class _Raw:
    """A table with the HostTable geometry in other memory (THP anon mmap, cudaHostAlloc, device) for the bench."""

    def __init__(self, kind: str, rows: int, layout: str) -> None:
        eh = _eh()
        self.kind, self.geom = kind, eh.Geometry.make(rows, layout)
        n = self.geom.total
        self.mm = None
        t0 = time.perf_counter()
        if kind == "thp":
            self.mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            self.mm.madvise(mmap.MADV_HUGEPAGE)
            self.buf = torch.frombuffer(self.mm, dtype=torch.uint8)
            libc = ctypes.CDLL(None, use_errno=True)
            libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            libc.madvise(ctypes.c_void_p(self.buf.data_ptr()), n, eh.MADV_POPULATE_WRITE)
            self.t_alloc = time.perf_counter() - t0
            t0 = time.perf_counter()
            err, msg, self.dev = eh.ext().host_register(self.buf.data_ptr(), n, False)
            assert not err, msg
            self.t_reg = time.perf_counter() - t0
        elif kind == "pinned":
            self.buf = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            self.t_alloc, self.t_reg = time.perf_counter() - t0, 0.0
            self.dev = self.buf.data_ptr()
        elif kind == "device":
            self.buf = torch.empty(n, dtype=torch.uint8, device="cuda")
            self.t_alloc, self.t_reg = time.perf_counter() - t0, 0.0
            self.dev = self.buf.data_ptr()
        elif kind in ("vmm", "vmm_host"):  # cuMemCreate(HOST_NUMA 0 | HOST): driver-allocated, 2 MiB granularity
            self.buf = None
            loc = 0 if kind == "vmm" else 1
            err, ptr, size, gran, fd, host_access = eh.ext().vmm_host_alloc(n, loc, True)
            self.note = "fd-shareable"
            if err:
                err2, ptr, size, gran, fd, host_access = eh.ext().vmm_host_alloc(n, loc, False)
                self.note = f"NOT fd-shareable ({err})"
                if err2:
                    raise RuntimeError(f"{kind}: {err2}")
            self.dev, self.vmm = ptr, ptr
            self.note += f", granularity {gran >> 10} KiB, export fd {fd}, host access {bool(host_access)}"
            self.t_alloc, self.t_reg = time.perf_counter() - t0, 0.0
        else:
            raise ValueError(kind)

    def fill_rows(self, w, s, r0: int) -> None:
        g, n = self.geom, w.shape[0]
        if self.buf is not None:
            self.buf.as_strided((n, 256), (g.w_stride, 1), g.w_off + r0 * g.w_stride).copy_(w)
            self.buf.as_strided((n, 8), (g.s_stride, 1), g.s_off + r0 * g.s_stride).copy_(s)
        else:  # VMM: no CPU mapping needed, strided cudaMemcpy2D from the GPU source
            ext = _eh().ext()
            w, s = w.contiguous(), s.contiguous()
            ext.copy2d(self.dev + g.w_off + r0 * g.w_stride, g.w_stride, w.data_ptr(), 256, 256, n)
            ext.copy2d(self.dev + g.s_off + r0 * g.s_stride, g.s_stride, s.data_ptr(), 8, 8, n)

    def close(self):
        if getattr(self, "vmm", None):
            _eh().ext().vmm_host_free(self.vmm)
            self.vmm = None
        if self.kind == "thp":
            _eh().ext().host_unregister(self.buf.data_ptr())
        self.buf = None
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass

    def thp_coverage(self) -> str:
        if self.kind in ("vmm", "vmm_host"):
            return self.note
        if self.kind != "thp":
            return ""
        try:
            for line in open("/proc/self/smaps_rollup"):
                if line.startswith("AnonHugePages"):
                    return f"AnonHugePages {int(line.split()[1]) / 2**20:.1f} GiB"
        except OSError:
            pass
        return "?"


def _time_us(fn, iters: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def bench(argv: list[str]) -> int:
    import argparse

    require_cuda()
    eh = _eh()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0, help="synthetic table size per backend (GiB)")
    ap.add_argument("--backends", default="shm,thp,pinned,device")
    ap.add_argument("--layouts", default="split,packed")
    ap.add_argument("--quick", action="store_true", help="smaller prefill sweep")
    ap.add_argument("--diag", action="store_true", help="access-pattern diagnosis (sequential / windowed / random)")
    a = ap.parse_args(argv)
    dev = torch.cuda.current_device()
    print(f"GPU {dev} ({torch.cuda.get_device_name(dev)}), CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        import subprocess

        print(subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,power.limit,clocks.sm,pcie.link.gen.current,"
                              "pcie.link.width.current", "--format=csv"], capture_output=True, text=True).stdout)
    except Exception:  # noqa: BLE001
        pass
    rows = int(a.gb * 2**30 // 272)
    sizes_n = 24
    seg = rows // sizes_n
    sizes = [seg] * sizes_n  # bucket ranges (equal sizes are fine for timing)
    lo, hi = eh.head_ranges(sizes, TP)
    print(f"table: {rows:,} rows ({a.gb:.1f} GiB packed), 24 columns, ids uniform per column bucket")
    # fill source (GPU generated, copied to each backend)
    w = torch.randint(0, 256, (rows, 256), dtype=torch.uint8, device="cuda")
    s = torch.randint(115, 135, (rows, 8), dtype=torch.uint8, device="cuda")
    ref_cache: dict = {}

    def ids_for(T, seed=0):
        return head_ids(T, sizes, seed=seed)

    for layout in a.layouts.split(","):
        for kind in a.backends.split(","):
            torch.cuda.synchronize()
            free_before = torch.cuda.mem_get_info()[0]
            t0 = time.perf_counter()
            if kind == "shm":
                tab = make_table(f"bench_{layout}", rows, layout=layout)
                t_pop = tab.populate()
                t_reg = tab.register()
                base, geom = tab.dev_base, tab.geom
                cov = ""
            else:
                tab = _Raw(kind, rows, layout)
                t_pop, t_reg, base, geom = tab.t_alloc, tab.t_reg, tab.dev, tab.geom
                cov = tab.thp_coverage()
            # fill through the host mapping (chunks: avoid a full-size host staging copy)
            chunk = 1 << 22
            for r0 in range(0, rows, chunk):
                r1 = min(rows, r0 + chunk)
                if kind == "shm":
                    tab.fill_from(w[r0:r1], s[r0:r1], r0)
                else:
                    tab.fill_rows(w[r0:r1], s[r0:r1], r0)
            torch.cuda.synchronize()
            print(f"\n== {kind} / {layout}: {geom.total / 2**30:.2f} GiB, alloc+populate {t_pop:.2f} s, cudaHostRegister "
                  f"{t_reg:.2f} s ({t_reg / (geom.total / 2**30):.3f} s/GiB) {cov}; setup {time.perf_counter() - t0:.1f} s; "
                  f"device memory -{(free_before - torch.cuda.mem_get_info()[0]) / 2**20:.0f} MiB vs before")

            def run(ids, out, **kw):
                eh.gather_rows(base + geom.w_off, base + geom.s_off, geom.w_stride, geom.s_stride, ids, out, lo, hi, **kw)

            # correctness vs the stock kernel on the device table
            ids = ids_for(64, 3)
            out = torch.empty(64, 24, 256, dtype=torch.bfloat16, device="cuda")
            run(ids, out)
            torch.cuda.synchronize()
            ok, msg = bits_equal(out, stock_rows(w, s, ids, sizes))
            print(f"   check vs stock: {msg}")
            # decode latency (kernel alone; the patch overlaps it with layer 0)
            line = []
            for T in (1, 6, 8, 48, 64):
                ids = ids_for(T, T)
                out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
                us = _time_us(lambda: run(ids, out), iters=200)
                line.append(f"T={T} {us:.1f} us")
            print("   decode gather (24 rows/token): " + ", ".join(line))
            # prefill bandwidth
            T = 16384
            ids = ids_for(T, 77)
            out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
            nbytes = T * 24 * 264
            cfgs = [(c, th, u) for c in (1, 2, 4, 8, 16, 32, 66, 132) for th in (256, 1024) for u in (1, 4, 8)]
            if a.quick:
                cfgs = [(c, 1024, 4) for c in (2, 4, 8, 16, 32)] + [(132, 256, 1)]
            res = []
            for c, th, u in cfgs:
                us = _time_us(lambda: run(ids, out, ctas=c, threads=th, unroll=u), iters=10, warmup=2)
                res.append((nbytes / us / 1e3, c, th, u, us))
            res.sort(reverse=True)
            best = res[0]
            print(f"   prefill T=16384 ({nbytes / 1e6:.0f} MB of table bytes): best {best[0]:.1f} GB/s "
                  f"({best[4] / 1e3:.2f} ms) at ctas={best[1]} threads={best[2]} unroll={best[3]}")
            by_ctas = {}
            for gbs, c, th, u, us in res:
                if c not in by_ctas:
                    by_ctas[c] = (gbs, th, u)
            print("   best per CTA count: " + ", ".join(f"{c}: {v[0]:.1f} GB/s (t{v[1]} u{v[2]})"
                                                       for c, v in sorted(by_ctas.items())))
            dflt = eh.launch_config(T * 24)
            us = _time_us(lambda: run(ids, out), iters=10, warmup=2)
            print(f"   default config {dflt}: {nbytes / us / 1e3:.1f} GB/s ({us / 1e3:.2f} ms)")
            if a.diag:
                # access-pattern diagnosis: translation (GPU TLB / IOMMU IOTLB per 4 KiB page) vs PCIe request rate
                seq = torch.arange(T * 24, device="cuda", dtype=torch.int64)
                diag = {
                    "sequential rows": (seq % rows).to(torch.int32).view(T, 24),
                    "random in 2 MiB": (torch.randint(0, (2 << 20) // 272, (T, 24), device="cuda") ).to(torch.int32),
                    "random in 64 MiB": torch.randint(0, (64 << 20) // 272, (T, 24), device="cuda").to(torch.int32),
                    "random in 1 GiB": torch.randint(0, (1 << 30) // 272, (T, 24), device="cuda").to(torch.int32),
                    "random (full table)": torch.randint(0, rows, (T, 24), device="cuda").to(torch.int32),
                }
                full_lo, full_hi = [0] * 24, [rows] * 24
                for name, d_ids in diag.items():
                    for vec in (16, 8):
                        for c, th, u in ((16, 1024, 4), (132, 1024, 8)):
                            us = _time_us(lambda: eh.gather_rows(base + geom.w_off, base + geom.s_off, geom.w_stride,
                                                                 geom.s_stride, d_ids, out, full_lo, full_hi, ctas=c,
                                                                 threads=th, unroll=u, vec=vec), iters=8, warmup=2)
                            print(f"   diag {name:<20s} vec={vec:2d} ctas={c:3d} u={u}: {nbytes / us / 1e3:6.1f} GB/s "
                                  f"({T * 24 / us:7.1f} Mrows/s)")
            if kind not in ("device", "vmm", "vmm_host") and not a.quick:
                # interference: a bf16 GEMM stream on the main stream, gather loop on a side stream
                A = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
                B = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
                mm_us = _time_us(lambda: torch.mm(A, B), iters=20)
                side = torch.cuda.Stream(priority=-1)
                for c in (4, 16):
                    def both():
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            run(ids, out, ctas=c, threads=1024, unroll=4)
                        for _ in range(4):
                            torch.mm(A, B)
                        torch.cuda.current_stream().wait_stream(side)
                    us_both = _time_us(both, iters=10, warmup=2)
                    print(f"   overlap: 4x GEMM 8192^3 alone {4 * mm_us / 1e3:.2f} ms, with a concurrent 16K gather "
                          f"(ctas={c}) {us_both / 1e3:.2f} ms (gather alone ~{nbytes / 1e3 / max(1e-9, by_ctas.get(c, best)[0]) / 1e3:.2f} ms)")
                del A, B
            # stock Triton lookup on the device table (all 4 ranks' 6-head lookups back to back) for reference
            if kind == "device" and layout == a.layouts.split(",")[0]:
                for T in (6, 16384):
                    ids = ids_for(T, 5)
                    us = _time_us(lambda: stock_rows(w, s, ids, sizes), iters=20 if T > 100 else 100)
                    print(f"   stock _engram_lookup_kernel (4 shards, device table) T={T}: {us:.1f} us total")
            tab.close()
            del tab
            torch.cuda.empty_cache()
    return 0


def _tp4_worker(rank: int, n: int, port: int, q) -> None:
    """One TP rank: stock Engram with the stock head-sharded GPU table vs the ds41h host table (decode mode T=6 with and
    without EARLY_WKV, prefill/shard mode T=600 > DS41H_ENGRAM_HOST_LOCAL_MAX): Engram.forward outputs bit-identical."""
    try:
        torch.cuda.set_device(rank)
        eg = _stock()
        from vllm.config import EngramConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps

        from ds41h import patch_engram

        vcfg = VllmConfig()
        vcfg.engram_config = EngramConfig(cpu_offload=False, dp_shared_memory=False)
        vcfg.scheduler_config.max_num_batched_tokens = 1024
        with set_current_vllm_config(vcfg):
            ps.init_distributed_environment(world_size=n, rank=rank, local_rank=rank,
                                            distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ps.initialize_model_parallel(n, 1)
        patch_engram.patch_engram_module(eg)
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)

        class Cfg:
            hidden_size = 512
            hc_mult = 4
            rms_norm_eps = 1e-6
            engram_layer_ids = [1, 14]
            engram_num_embeddings = [0, 0]
            engram_max_ngram_size = 4
            engram_n_heads = 8
            engram_head_dim = 256
            engram_compressed_vocab_size = 1000
            engram_pad_token_id = 2
            engram_vocab_size = 40_000

        cfg = Cfg()
        res = []
        with set_current_vllm_config(vcfg):
            layout = eg.EngramLayout(cfg)
            sizes = [p for order in layout.primes[0] for p in order]
            cfg.engram_num_embeddings = [sum(sizes) + 17, sum(sizes) + 17]
            layout = eg.EngramLayout(cfg)
            w, s = synthetic_table(layout.num_embeddings[0], device="cpu", all_codes=True)
            g = torch.Generator(device="cpu").manual_seed(1)
            wkv = (torch.randn(cfg.hidden_size * 5, 24 * 256, generator=g, device="cpu") * 0.02).to(torch.bfloat16)
            q_w = torch.randn(4, cfg.hidden_size, generator=g, device="cpu").to(torch.bfloat16)
            k_w = torch.randn(4, cfg.hidden_size, generator=g, device="cpu").to(torch.bfloat16)

            def build(kind):
                if kind == "stock":
                    orig = eg.Engram._create_embedding
                    eg.Engram._create_embedding = lambda self, lay, i: eg.ParallelEngramEmbedding(
                        lay.num_embeddings[i], lay.head_dim, tuple(p for o in lay.primes[i] for p in o))
                    try:
                        m = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
                    finally:
                        eg.Engram._create_embedding = orig
                else:
                    early0 = patch_engram.EARLY_WKV
                    patch_engram.EARLY_WKV = kind == "host_early"
                    try:
                        m = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
                    finally:
                        patch_engram.EARLY_WKV = early0
                et = m.embed_tokens
                et.weight.weight_loader(et.weight, w.view(torch.float8_e4m3fn))
                et.weight_scale_inv.weight_loader(et.weight_scale_inv, s.view(torch.float8_e8m0fnu))
                m.wkv.weight.weight_loader(m.wkv.weight, wkv)
                m.q_weight.data.copy_(q_w)
                m.k_weight.data.copy_(k_w)
                return m

            mods = {k: build(k) for k in ("stock", "host", "host_early")}
            assert getattr(mods["host"].embed_tokens, "_ds41h_host", None) is not None
            for T in (6, 600):
                gh = torch.Generator(device="cpu").manual_seed(T)
                hashes = torch.stack([head_ids(T, sizes, seed=T, layer=k) for k in range(2)], 1)
                X = (torch.randn(T, 4, cfg.hidden_size, generator=gh, device="cpu") * 0.5).to(torch.bfloat16).cuda()
                mask = (torch.rand(T, generator=gh, device="cpu") > 0.1).cuda()
                outs = {}
                for k, m in mods.items():
                    early0 = patch_engram.EARLY_WKV
                    patch_engram.EARLY_WKV = k == "host_early"
                    try:
                        m.prepare_embeddings(hashes[:, 0])
                        outs[k] = m(X, hashes[:, 0], mask)
                    finally:
                        patch_engram.EARLY_WKV = early0
                torch.cuda.synchronize()
                mode = "shard" if T > patch_engram.LOCAL_MAX else "local"
                for k in ("host", "host_early"):
                    ok, msg = bits_equal(outs[k], outs["stock"])
                    res.append((T, mode, k, ok, msg))
        q.put((rank, res, ""))
        for m in mods.values():
            h = getattr(m.embed_tokens, "_ds41h_host", None)
            if h is not None:
                h.table.close()
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, [], f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def tp4(argv: list[str]) -> int:
    """TP=4 (4 GPUs, ~2 min, synthetic tables, no model): the patched Engram module across 4 processes -- cooperative
    shard loading into ONE host file, decode mode without the rows all-gather (+ EARLY_WKV), prefill shard mode with the
    stock all-gather -- vs stock vLLM's head-sharded GPU tables, bit-exact. Run: python
    tests/test_engram_host.py --tp4"""
    import torch.multiprocessing as mp

    n = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 2000
    ps = [ctx.Process(target=_tp4_worker, args=(r, n, port, q)) for r in range(n)]
    for p in ps:
        p.start()
    out = sorted(q.get(timeout=600) for _ in ps)
    for p in ps:
        p.join(120)
    bad = 0
    for rank, res, err in out:
        if err:
            print(f"  rank {rank}: ERROR {err}")
            bad += 1
        for T, mode, k, ok, msg in res:
            print(f"  rank {rank} T={T} ({mode} mode) {k}: {msg}")
            bad += not ok
    print(f"tp{n}: {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


def _bench4_worker(rank: int, n: int, gb: float, layout: str, T: int, iters: int, barrier, q) -> None:
    try:
        torch.cuda.set_device(rank)
        eh = _eh()
        rows = int(gb * 2**30 // 272)
        sizes = [rows // 24] * 24
        key = f"bench4 rows={rows} {layout}"
        t = eh.HostTable(eh.table_path(key, 0, layout), key, rows, n_shards=n, layout=layout, keep=False)
        if t.created:
            t.populate()
        t_reg = t.register()
        lo_r, hi_r = eh.shard_rows(sizes, n, rank)
        step = 1 << 22  # fill in 1 GiB chunks
        for r0 in range(lo_r, hi_r, step):
            r1 = min(hi_r, r0 + step)
            wv, sv = t.views(r0, r1)
            wv.view(torch.uint8).copy_(torch.randint(0, 256, (r1 - r0, 256), dtype=torch.uint8, device="cuda").cpu())
            sv.copy_(torch.randint(115, 135, (r1 - r0, 8), dtype=torch.uint8, device="cuda").cpu())
        del wv, sv
        t.set_state(rank, "weight", 2)
        t.set_state(rank, "scale", 2)
        t.wait_ready(600)
        lo, hi = eh.head_ranges(sizes, n)
        ids = head_ids(T, sizes, seed=rank)
        out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device="cuda")
        t.gather(ids, out, lo, hi)
        torch.cuda.synchronize()
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(iters):
            t.gather(ids, out, lo, hi)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        barrier.wait()
        q.put((rank, t_reg, dt, T * 24 * 264 / dt / 1e9))
        t.close()
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, -1.0, -1.0, f"{type(e).__name__}: {e} {traceback.format_exc()}"))


def bench4(argv: list[str]) -> int:
    """All GPUs gather from ONE shared table at the same time (the prefill situation): aggregate host/PCIe/IOMMU
    bandwidth. Needs all 4 GPUs and the RAM for the table."""
    import argparse

    import torch.multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0)
    ap.add_argument("--layout", default="split")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args(argv)
    n = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(n), ctx.Queue()
    ps = [ctx.Process(target=_bench4_worker, args=(r, n, a.gb, a.layout, a.tokens, a.iters, barrier, q)) for r in range(n)]
    for p in ps:
        p.start()
    res = sorted(q.get(timeout=1200) for _ in ps)
    for p in ps:
        p.join(60)
    for r, t_reg, dt, bw in res:
        print(f"  GPU {r}: cudaHostRegister {t_reg:.1f} s, T={a.tokens} gather {dt * 1e3:.2f} ms = {bw if isinstance(bw, str) else f'{bw:.1f} GB/s'}")
    tot = sum(b for *_, b in res if not isinstance(b, str))
    print(f"  aggregate {tot:.1f} GB/s over {n} GPUs reading one shared {a.gb:.0f} GiB {a.layout} table concurrently")
    return 0


if __name__ == "__main__":
    try:
        if "--tp4" in sys.argv:
            rc = tp4([x for x in sys.argv[1:] if x != "--tp4"])
        elif "--bench4" in sys.argv:
            rc = bench4([x for x in sys.argv[1:] if x != "--bench4"])
        elif "--bench" in sys.argv:
            rc = bench([x for x in sys.argv[1:] if x != "--bench"])
        else:
            rc = run_module_tests(globals())
    finally:
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    raise SystemExit(rc)
