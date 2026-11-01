"""Tests + CPU-only I/O benchmark for ds41h ENGRAM_NVME (ds41h/engram_nvme.py, kernels/engram_nvme.cu, NVMe mode of
ds41h/patch_engram.py).

Run with DS41H unset (the vLLM functions compared against are stock):
    CUDA_VISIBLE_DEVICES= python tests/test_engram_nvme.py cpu          # I/O engine vs the table files (no GPU)
    python tests/test_engram_nvme.py                  # all tests (one GPU)
    python tests/test_engram_nvme.py --tp4        # patched Engram at TP=4 vs stock (+ graphs)
    CUDA_VISIBLE_DEVICES= python tests/test_engram_nvme.py --io-bench [--procs 4] [--mode auto] ...   # no GPU

Reference = stock vLLM 0.32.0 `_engram_lookup_kernel` (vllm/models/deepseek_v41/common/engram.py:639-721, non-packed,
unsorted = the sm90 path) on a GPU-resident copy of the same rows, per TP rank (HEAD_START / LOCAL_HEADS / vocab range
of that rank). Every comparison is BIT-EXACT (bf16 bits). Real-row tests read the re-laid-out table files built by
`python -m ds41h.engram_nvme build` and the checkpoint (DS41H_MODEL_DIR, default $DS41H_HOME/model).
Synthetic tables are written to DS41H_ENGRAM_NVME_TEST_DIR (default ~/.ds41h/engram_nvme/test_<pid>, removed).
"""

from __future__ import annotations

import os
import shutil
import statistics
import struct
import sys
import time

import numpy as np
import torch

from common import require_cuda, run_module_tests, skip  # noqa: E402  (adds the repo to sys.path)

MODEL = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
GSM8K = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "gsm8k", "test.jsonl")
TEST_DIR = os.environ.setdefault("DS41H_ENGRAM_NVME_TEST_DIR",
                                 os.path.expanduser(f"~/.ds41h/engram_nvme/test_{os.getpid()}"))
TP = 4
PART = 6
NCOLS = 24
TIMEOUT_MS = 60_000


def _en():
    from ds41h import engram_nvme

    return engram_nvme


def _eh_tests():
    import test_engram_host as teh  # synthetic tables / stock reference helpers of the engram_host tests

    return teh


def new_reader(mode: str = "auto", cpu_only: bool = False, rank: int = 0, **env):
    en = _en()
    old = {k: os.environ.get(k) for k in ("DS41H_ENGRAM_NVME_IO", *env)}
    os.environ["DS41H_ENGRAM_NVME_IO"] = mode
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return en.Reader(rank, cpu_only=cpu_only)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def real_layout():
    from common import reader

    from ds41h import ref

    return ref.engram_layout(reader().config)


def real_table(layer: int):
    en = _en()
    try:
        return en.open_table(MODEL, layer, build_missing=False)
    except Exception as e:  # noqa: BLE001
        skip(f"no valid NVMe table for L{layer}: {e} (python -m ds41h.engram_nvme build)")


def write_table(path: str, w: np.ndarray, s: np.ndarray, layer: int = 0):
    """Synthetic table file in the NVMe format (header + 15 x 272-B records per 4 KiB block)."""
    en = _en()
    rows = w.shape[0]
    nb = en.n_blocks(rows)
    blk = np.zeros((nb, en.BLOCK), dtype=np.uint8)
    full = rows // en.PER_BLOCK
    rec = blk[:full, : en.PER_BLOCK * en.RECORD].reshape(full, en.PER_BLOCK, en.RECORD)
    rec[:, :, : en.DIM] = w[: full * en.PER_BLOCK].reshape(full, en.PER_BLOCK, en.DIM)
    rec[:, :, en.DIM : en.PAYLOAD] = s[: full * en.PER_BLOCK].reshape(full, en.PER_BLOCK, en.SDIM)
    rem = rows - full * en.PER_BLOCK
    if rem:
        r = blk[full, : rem * en.RECORD].reshape(rem, en.RECORD)
        r[:, : en.DIM] = w[full * en.PER_BLOCK :]
        r[:, en.DIM : en.PAYLOAD] = s[full * en.PER_BLOCK :]
    key = f"synthetic rows={rows} pid={os.getpid()} t={time.time()}"
    hdr = bytearray(en.DATA_OFF)
    kb = key.encode()
    en._HDR.pack_into(hdr, 0, en.MAGIC, en.VERSION, en.RECORD, en.PER_BLOCK, en.BLOCK, rows, en.DATA_OFF,
                      en.file_size(rows), layer, len(kb))
    hdr[en._OFF_KEY : en._OFF_KEY + len(kb)] = kb
    struct.pack_into("<Idd", hdr, en._OFF_DONE, en._DONE, 0.0, time.time())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(blk.tobytes())
    assert en.header_ok(__import__("pathlib").Path(path), key, rows)
    return en.Table(path, layer, rows, key)


def file_records(path: str, ids) -> list[bytes]:
    en = _en()
    out = []
    with open(path, "rb", buffering=0) as f:
        for r in ids:
            out.append(os.pread(f.fileno(), en.PAYLOAD, en.record_offset(int(r))))
    return out


def stock_lookup(weight: torch.Tensor, scale: torch.Tensor, ids: torch.Tensor, vstart: int, vend: int, head0: int,
                 part: int = PART, total: int = NCOLS) -> torch.Tensor:
    """Stock `_engram_lookup_kernel` (one TP rank): weight / scale = GPU rows [vstart, vend) (uint8 [W, 256] / [W, 8])."""
    from vllm.models.deepseek_v41.common import engram as eg
    from vllm.triton_utils import triton

    T = ids.shape[0]
    out = torch.empty(T, part, 256, dtype=torch.bfloat16, device=ids.device)
    rows = T * part
    if rows == 0:
        return out
    sms = torch.cuda.get_device_properties(ids.device).multi_processor_count
    grid = max(1, min(triton.cdiv(rows, 16), sms))
    eg._engram_lookup_kernel[(grid,)](
        weight.view(torch.float8_e4m3fn), scale, ids, ids, out, vstart, vend, rows, ids.stride(0), ids.stride(1),
        HEAD_START=head0, LOCAL_HEADS=part, TOTAL_HEADS=total, DIM=256, QUANT_BLOCK=32, BLOCK_R=16, GRID=grid,
        SORTED=False, PACKED=False, OUT_STRIDE=out.stride(0),
    )  # fmt: skip
    return out


def nvme_lookup(ch, ids: torch.Tensor, head0: int, part: int = PART, total: int = NCOLS, out=None) -> torch.Tensor:
    en = _en()
    T = ids.shape[0]
    if out is None:
        out = torch.empty(T, part, 256, dtype=torch.bfloat16, device=ids.device)
    en.ext().issue(ch.handle, ids, head0, part, total, ch.dstate)
    en.ext().wait_dequant(ch.handle, ch.dstate, out, T * part, TIMEOUT_MS)
    return out


def bits_equal(a: torch.Tensor, b: torch.Tensor):
    return _eh_tests().bits_equal(a, b)


def add_gpu_channel(rd, table, vstart, vend, max_rows):
    ch = rd.add_channel(table, vstart, vend, max_rows)
    ch.dstate = torch.zeros(16, dtype=torch.int64, device="cuda")
    return ch


# ======================================================================================== CPU tests (no GPU)
def _check_channel(en, ch, table, ids: np.ndarray, vstart: int, vend: int) -> int:
    """src / rows of the channel after a request with `ids` == the table file + checkpoint bytes. Returns #checked."""
    src, rows, _ = en.ext().channel_buffers(ch.handle)
    src = src[: len(ids)].numpy().copy()
    rows = rows.numpy()
    recs = {}
    owned = [int(i) for i in ids if vstart <= i < vend]
    for r, b in zip(owned, file_records(table.path, owned)):
        recs[r] = b
    for k, i in enumerate(ids.tolist()):
        if not (vstart <= i < vend):
            assert src[k] == -1, f"id {i} outside [{vstart}, {vend}) must give src -1, got {src[k]}"
            continue
        assert src[k] >= 0, f"owned id {i} got src {src[k]}"
        got = bytes(rows[src[k], : en.PAYLOAD])
        assert got == recs[i], f"k={k} id={i}: row bytes differ from the table file"
    return len(owned)


def test_cpu_engine_rows():
    """I/O engine (CPU-driven requests, both I/O modes, both real tables): every owned id -> its record bytes, ids
    outside the shard -> src -1 (stock zero rows), duplicates share one read; table bytes == checkpoint bytes."""
    en = _en()
    lay = real_layout()
    rng = np.random.default_rng(0)
    for li, layer in enumerate(lay.layer_ids):
        table = real_table(layer)
        src = en.tensor_source(MODEL, layer)
        sizes = lay.head_sizes(li)
        for mode in ("direct", "buffered"):
            rd = new_reader(mode, cpu_only=True)
            rank = 1 + li  # ranks 1 and 2: interior shards
            vstart, vend = sum(sizes[: rank * PART]), sum(sizes[: (rank + 1) * PART])
            ch = rd.add_channel(table, vstart, vend, 8192)
            rd.start()
            checked = 0
            for T in (1, 8, 300, 1365):
                ids = rng.integers(vstart, vend, T * PART).astype(np.int32)
                k = len(ids) // 3
                if k:
                    ids[-k:] = ids[:k]  # duplicates
                if T >= 8:
                    ids[:6] = [vstart - 1, vend, -1, 0, vend - 1, vstart]  # outside / edges
                ns = en.ext().cpu_request(ch.handle, torch.from_numpy(ids), 30.0)
                assert ns > 0, f"request timed out (mode {mode}, T {T})"
                checked += _check_channel(en, ch, table, ids, vstart, vend)
            st = en.stats_dict(en.ext().channel_stats(ch.handle))
            rd.stop()
            # the table file == checkpoint for the rows read
            sample = rng.integers(vstart, vend, 64)
            with open(src.file, "rb", buffering=0) as f:
                for r, rec in zip(sample.tolist(), file_records(table.path, sample)):
                    w = os.pread(f.fileno(), en.DIM, src.w_off + r * en.DIM)
                    s = os.pread(f.fileno(), en.SDIM, src.s_off + r * en.SDIM)
                    assert rec == w + s, f"L{layer} row {r}: table file != checkpoint"
            print(f"  L{layer} rank {rank} {mode}: {checked} owned ids checked ({st['requests']} requests, "
                  f"{st['unique']} unique reads of {st['owned']} owned, {st['direct_ios']} O_DIRECT)")


def test_cpu_engine_priority_and_stress():
    """Two channels (layer order) with back-to-back requests, 300 rounds of random sizes: every round correct."""
    en = _en()
    lay = real_layout()
    tables = [real_table(layer) for layer in lay.layer_ids]
    rng = np.random.default_rng(1)
    rd = new_reader("auto", cpu_only=True, DS41H_ENGRAM_NVME_AUTO_MAX="64")
    chans = []
    for li, t in enumerate(tables):
        sizes = lay.head_sizes(li)
        vs, ve = sum(sizes[:PART]), sum(sizes[: 2 * PART])
        chans.append((rd.add_channel(t, vs, ve, 4096), t, vs, ve))
    rd.start()
    t0 = time.perf_counter()
    for it in range(300):
        reqs = []
        for ch, t, vs, ve in chans:
            n = int(rng.integers(1, 4096))
            ids = rng.integers(vs - 100, ve + 100, n).astype(np.int32)
            reqs.append((ch, t, vs, ve, ids, en.ext().cpu_publish(ch.handle, torch.from_numpy(ids))))
        for ch, t, vs, ve, ids, g in reqs:
            assert en.ext().cpu_wait(ch.handle, g, 30.0) > 0, "timeout"
            if it % 25 == 0:
                _check_channel(en, ch, t, ids, vs, ve)
    rd.stop()
    st = [en.stats_dict(en.ext().channel_stats(c.handle)) for c, *_ in chans]
    print(f"  300 rounds x 2 layers in {time.perf_counter() - t0:.1f} s; reads {[s['ios'] for s in st]}, O_DIRECT "
          f"{[s['direct_ios'] for s in st]} (auto_max 64)")


# ======================================================================================== GPU tests
def _synthetic(rows: int):
    teh = _eh_tests()
    w, s = teh.synthetic_table(rows, device="cpu", all_codes=True)
    return w.numpy(), s.numpy()


def test_gpu_bitexact_all_codes():
    """Synthetic table with every e4m3 code x every ue8m0 byte: NVMe path of each of the 4 TP ranks == stock lookup of
    that rank (GPU-resident shard), bitwise; T from 1 (decode) to 2000 (> 4096 rows: the U=4 dequant variant)."""
    require_cuda()
    en = _en()
    teh = _eh_tests()
    sizes = teh.synthetic_layout(NCOLS, base=40_000)
    rows = sum(sizes) + 17
    w, s = _synthetic(rows)
    table = write_table(os.path.join(TEST_DIR, "allcodes.bin"), w, s)
    wg, sg = torch.from_numpy(w).cuda(), torch.from_numpy(s).cuda()
    for mode in ("direct", "buffered"):
        rd = new_reader(mode)
        chans = []
        for r in range(TP):
            vs, ve = sum(sizes[: r * PART]), sum(sizes[: (r + 1) * PART])
            chans.append((add_gpu_channel(rd, table, vs, ve, 4096 * PART), vs, ve))
        rd.start()
        for T in (1, 6, 8, 300, 2000):
            ids = teh.head_ids(T, sizes, seed=T + 11)
            for r, (ch, vs, ve) in enumerate(chans):
                ref = stock_lookup(wg[vs:ve], sg[vs:ve], ids, vs, ve, r * PART)
                got = nvme_lookup(ch, ids, r * PART)
                torch.cuda.synchronize()
                ok, msg = bits_equal(got, ref)
                assert ok, f"{mode} T={T} rank {r}: {msg}"
        rd.stop()
        print(f"  {mode}: 4 ranks x T in (1, 6, 8, 300, 2000): bit-identical to the stock lookup")


def test_gpu_ownership_and_dedupe():
    """ids outside the rank's rows (other shards, DEAD_ID -1, past the table) -> zero rows like stock; heavy duplicates."""
    require_cuda()
    teh = _eh_tests()
    sizes = teh.synthetic_layout(NCOLS, base=20_000, seed=3)
    rows = sum(sizes)
    w, s = _synthetic(rows)
    table = write_table(os.path.join(TEST_DIR, "own.bin"), w, s)
    wg, sg = torch.from_numpy(w).cuda(), torch.from_numpy(s).cuda()
    rd = new_reader("auto")
    r = 2
    vs, ve = sum(sizes[: r * PART]), sum(sizes[: (r + 1) * PART])
    ch = add_gpu_channel(rd, table, vs, ve, 1024 * PART)
    rd.start()
    T = 1000
    ids = teh.head_ids(T, sizes, seed=5).clone()
    ids[::3] = ids[0]  # duplicates of one row in every column
    ids[1::17] = -1
    ids[2::19] = rows + 5
    ids[3::23, :] = torch.tensor([sum(sizes[:h]) for h in range(NCOLS)], dtype=torch.int32, device="cuda")
    ref = stock_lookup(wg[vs:ve], sg[vs:ve], ids, vs, ve, r * PART)
    got = nvme_lookup(ch, ids, r * PART)
    torch.cuda.synchronize()
    ok, msg = bits_equal(got, ref)
    rd.stop()
    assert ok, msg
    st = _en().stats_dict(_en().ext().channel_stats(ch.handle))
    print(f"  T={T}: {msg}; {st['owned']} owned ids -> {st['unique']} reads")


def test_gpu_graph_replay():
    """issue + wait/dequant captured in CUDA graphs (T=8 and T=48, the FULL_DECODE_ONLY situation): 60 replays with new
    ids written into the static hash buffer, interleaved with eager calls of other sizes; every replay == stock."""
    require_cuda()
    teh = _eh_tests()
    sizes = teh.synthetic_layout(NCOLS, base=30_000, seed=1)
    rows = sum(sizes)
    w, s = _synthetic(rows)
    table = write_table(os.path.join(TEST_DIR, "graph.bin"), w, s)
    wg, sg = torch.from_numpy(w).cuda(), torch.from_numpy(s).cuda()
    rd = new_reader("auto")
    r = 1
    vs, ve = sum(sizes[: r * PART]), sum(sizes[: (r + 1) * PART])
    ch = add_gpu_channel(rd, table, vs, ve, 512 * PART)
    rd.start()
    graphs = {}
    stream = torch.cuda.Stream()
    for T in (8, 48):
        hash_all = torch.zeros(T, 2, NCOLS, dtype=torch.int32, device="cuda")  # stock [T, layers, cols] hash layout
        out = torch.empty(T, PART, 256, dtype=torch.bfloat16, device="cuda")
        hash_all[:, 0] = teh.head_ids(T, sizes, seed=1)
        with torch.cuda.stream(stream):  # warm up outside the capture
            nvme_lookup(ch, hash_all[:, 0], r * PART, out=out)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            nvme_lookup(ch, hash_all[:, 0], r * PART, out=out)
        graphs[T] = (g, hash_all, out)
    bad = 0
    for it in range(60):
        T = (8, 48)[it % 2]
        g, hash_all, out = graphs[T]
        ids = teh.head_ids(T, sizes, seed=100 + it)
        hash_all[:, 0].copy_(ids)
        g.replay()
        ref = stock_lookup(wg[vs:ve], sg[vs:ve], ids, vs, ve, r * PART)
        torch.cuda.synchronize()
        ok, msg = bits_equal(out, ref)
        bad += not ok
        if it % 7 == 3:  # eager request of another size between replays (shared generation counter)
            e_ids = teh.head_ids(33, sizes, seed=900 + it)
            ok2, msg2 = bits_equal(nvme_lookup(ch, e_ids, r * PART), stock_lookup(wg[vs:ve], sg[vs:ve], e_ids, vs, ve, r * PART))
            assert ok2, f"eager between replays: {msg2}"
    st = _en().stats_dict(_en().ext().channel_stats(ch.handle))
    rd.stop()
    assert not bad, f"{bad} of 60 replays differ"
    print(f"  60 graph replays (T=8 / 48) + eager calls: all bit-identical; GPU waits {st['gpu_waits']}, stalled "
          f"{st['gpu_spun']}, stall mean {st['gpu_stall_ns'] / max(st['gpu_spun'], 1) / 1e3:.1f} us")


def test_gpu_real_rows():
    """REAL table rows (both layers, re-laid-out files) through the NVMe path vs the stock kernel on a GPU copy of the
    checkpoint rows: a DS41H_ENGRAM_NVME_REAL_GB (default 2) window inside a real rank shard, decode and prefill sizes,
    both I/O modes. DS41H_ENGRAM_NVME_REAL_FULL=1: the whole 25 GB shard of rank 1 (real head ranges)."""
    require_cuda()
    en = _en()
    lay = real_layout()
    gb = float(os.environ.get("DS41H_ENGRAM_NVME_REAL_GB", "2"))
    full = os.environ.get("DS41H_ENGRAM_NVME_REAL_FULL", "0") == "1"
    rng = np.random.default_rng(7)
    for li, layer in enumerate(lay.layer_ids):
        table = real_table(layer)
        src = en.tensor_source(MODEL, layer)
        sizes = lay.head_sizes(li)
        rank = 1
        rs, re_ = sum(sizes[: rank * PART]), sum(sizes[: (rank + 1) * PART])
        if full:
            a, b = rs, re_
        else:
            n = int(gb * 2**30 // 264)
            a = int(rng.integers(rs, re_ - n))
            b = a + n
        t0 = time.perf_counter()
        wg = torch.empty(b - a, 256, dtype=torch.uint8, device="cuda")
        sg = torch.empty(b - a, 8, dtype=torch.uint8, device="cuda")
        with open(src.file, "rb", buffering=0) as f:
            step = 1 << 22
            for r0 in range(a, b, step):
                r1 = min(b, r0 + step)
                wv = os.pread(f.fileno(), (r1 - r0) * 256, src.w_off + r0 * 256)
                sv = os.pread(f.fileno(), (r1 - r0) * 8, src.s_off + r0 * 8)
                wg[r0 - a : r1 - a].copy_(torch.frombuffer(bytearray(wv), dtype=torch.uint8).view(r1 - r0, 256))
                sg[r0 - a : r1 - a].copy_(torch.frombuffer(bytearray(sv), dtype=torch.uint8).view(r1 - r0, 8))
        t_load = time.perf_counter() - t0
        for mode in ("auto", "direct", "buffered"):
            rd = new_reader(mode)
            ch = add_gpu_channel(rd, table, a, b, 16384 * PART)
            rd.start()
            for T in (8, 64, 16384):
                ids = torch.from_numpy(rng.integers(a, b, (T, NCOLS)).astype(np.int32)).cuda()
                ids[:, :3] = torch.from_numpy(rng.integers(0, src.rows, (T, 3)).astype(np.int32)).cuda()  # mostly not owned
                ref = stock_lookup(wg, sg, ids, a, b, rank * PART)
                got = nvme_lookup(ch, ids, rank * PART)
                torch.cuda.synchronize()
                ok, msg = bits_equal(got, ref)
                assert ok, f"L{layer} {mode} T={T}: {msg}"
            st = en.stats_dict(en.ext().channel_stats(ch.handle))
            rd.stop()
            print(f"  L{layer} rows [{a}, {b}) ({(b - a) * 264 / 2**30:.1f} GiB GPU copy loaded in {t_load:.0f} s), {mode}: "
                  f"T=8/64/16384 bit-identical; {st['unique']} reads ({st['direct_ios']} O_DIRECT), "
                  f"svc p50 {en.hist_quantile(st['svc_hist'], 0.5):.0f} us")
        del wg, sg
        torch.cuda.empty_cache()


def stock_hash(tokens, start: int, history, hp) -> np.ndarray:
    """Stock `_hash_ids_kernel` (V2 runner path: no slot cache, lookback ids) for one request chunk starting at
    position `start`, `history` = the tokens before it (only the last 3 are used) -> [T, 2, 24] int64."""
    from vllm.models.deepseek_v41.common import engram as eg
    from vllm.triton_utils import triton

    dev = "cuda"
    T = len(tokens)
    hist = list(history)[-3:]
    lb = [hist[len(hist) - 1 - c] if c < len(hist) else -1 for c in range(3)]  # col c = position start - 1 - c
    out = torch.empty(T, 2, NCOLS, dtype=torch.int32, device=dev)
    dummy = torch.zeros(1, 1, dtype=torch.int32, device=dev)
    eg._hash_ids_kernel[(triton.cdiv(T, 32), 2)](
        torch.tensor(np.asarray(tokens), dtype=torch.int32, device=dev), torch.tensor(hp.token_map, dtype=torch.int32, device=dev),
        torch.zeros(T, dtype=torch.bool, device=dev), torch.arange(start, start + T, dtype=torch.int64, device=dev), dummy,
        torch.tensor([0, T], dtype=torch.int32, device=dev), torch.tensor(hp.multipliers, device=dev),
        torch.tensor(hp.primes, device=dev), torch.tensor(hp.offsets, device=dev), None,
        torch.tensor([lb], dtype=torch.int32, device=dev), torch.zeros(1, 3, dtype=torch.bool, device=dev), out,
        T, 0, hp.pad_id,
        input_stride=1, mask_stride=1, position_stride=1, table_stride=0, table_col_stride=0, query_stride=1,
        num_query_rows=1, num_table_rows=0, max_blocks=0, cache_block_size=64, MAX_NGRAM=4, num_heads=8, BLOCK_T=32,
        BLOCK_H=8, dead_id=-1, lookback_depth=3, lookback_row_stride=3, lookback_col_stride=1,
        lookback_mask_row_stride=3, lookback_mask_col_stride=1, num_warps=4,
    )  # fmt: skip
    return out.cpu().numpy().astype(np.int64)


def test_gpu_cpu_hash_vs_stock():
    """The CPU n-gram hash used by the prefetcher (numpy here; the engine's C++ port is checked against it in
    test_cpu_prefetch) == stock `_hash_ids_kernel`: fresh prompts and chunk continuations with lookback ids."""
    require_cuda()
    hp = hash_params()
    rng = np.random.default_rng(11)
    docs = {"gsm8k": None, "random": rng.integers(1000, 120000, 4000)}
    from transformers import AutoTokenizer
    import json

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows_ = [json.loads(x) for x in open(GSM8K)][:60]
    docs["gsm8k"] = np.asarray(tok("\n\n".join(r["question"] + r["answer"] for r in rows_))["input_ids"][:4000])
    n = 0
    for name, toks in docs.items():
        for start, T in ((0, 1500), (1, 40), (2, 64), (1000, 2000), (3999 - 8, 8)):
            chunk = toks[start : start + T]
            ref = stock_hash(chunk, start, toks[:start], hp)
            got = _en().cpu_hash_ids(chunk, hp, history=toks[:start])
            assert ref.shape == got.shape and (ref == got).all(), f"{name} start={start} T={T}: CPU hash != stock"
            n += ref.size
    print(f"  {n} hash ids (gsm8k + random tokens, fresh prompts and chunk continuations): CPU hash == stock kernel")


# ---------------------------------------------------------------------------------------- patched stock Engram module
class _Cfg:  # the Engram constructor reads these
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


def _vllm_setup(world: int, rank: int, port: int, max_tokens: int):
    from vllm.config import EngramConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    vcfg = VllmConfig()
    vcfg.engram_config = EngramConfig(cpu_offload=False, dp_shared_memory=False)
    vcfg.scheduler_config.max_num_batched_tokens = max_tokens
    with set_current_vllm_config(vcfg):
        if not ps.model_parallel_is_initialized():
            ps.init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                            distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ps.initialize_model_parallel(world, 1)
    return vcfg


def real_window(rows: int, layer: int = 1, start: int | None = None):
    """(weight uint8 [rows, 256], scale uint8 [rows, 8]) numpy copies of REAL checkpoint rows [start, start + rows) of
    layer `layer`'s Engram table (default: a window in the middle of the table)."""
    en = _en()
    src = en.tensor_source(MODEL, layer)
    a = src.rows // 2 if start is None else start
    with open(src.file, "rb", buffering=0) as f:
        w = np.frombuffer(os.pread(f.fileno(), rows * en.DIM, src.w_off + a * en.DIM), dtype=np.uint8).reshape(rows, en.DIM)
        sc = np.frombuffer(os.pread(f.fileno(), rows * en.SDIM, src.s_off + a * en.SDIM), dtype=np.uint8).reshape(rows, en.SDIM)
    return w.copy(), sc.copy()


def _build_modules(eg, vcfg, table_path: str, seed: int = 0, source: str = "synthetic"):
    """(stock module with the stock GPU table, NVMe module reading `table_path`) with identical weights, at the current
    TP size. The class patch must be applied already; the stock module bypasses the ds41h allocation context.
    source: 'synthetic' (every e4m3 code x every ue8m0 byte) or 'real' (real checkpoint rows of layer 1)."""
    from vllm.config import set_current_vllm_config

    from ds41h import engram_nvme as en

    cfg = _Cfg()
    with set_current_vllm_config(vcfg):
        layout = eg.EngramLayout(cfg)
        sizes = [p for order in layout.primes[0] for p in order]
        cfg.engram_num_embeddings = [sum(sizes) + 17, sum(sizes) + 17]
        layout = eg.EngramLayout(cfg)
        rows = layout.num_embeddings[0]
        w, s = _synthetic(rows) if source == "synthetic" else real_window(rows)
        if not os.path.exists(table_path):
            write_table(table_path, w, s)
        tab = en.Table(table_path, 1, rows, en.read_header(__import__("pathlib").Path(table_path))["key"])
        open0 = en.open_table
        en.open_table = lambda model, layer, rows=None, build_missing=None: tab
        try:
            m_nv = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
        finally:
            en.open_table = open0
        orig = eg.Engram._create_embedding
        eg.Engram._create_embedding = lambda self, lay, i: eg.ParallelEngramEmbedding(
            lay.num_embeddings[i], lay.head_dim, tuple(p for o in lay.primes[i] for p in o))
        try:
            m_st = eg.Engram(cfg, None, layout, 0, use_sequence_parallel=False, prefix="layers.1.engram")
        finally:
            eg.Engram._create_embedding = orig
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        wkv = (torch.randn(cfg.hidden_size * 5, NCOLS * 256, generator=g, device="cpu") * 0.02).to(torch.bfloat16)
        q_w = torch.randn(4, cfg.hidden_size, generator=g, device="cpu").to(torch.bfloat16)
        k_w = torch.randn(4, cfg.hidden_size, generator=g, device="cpu").to(torch.bfloat16)
        for m in (m_st, m_nv):
            et = m.embed_tokens
            et.weight.weight_loader(et.weight, torch.from_numpy(w).view(torch.float8_e4m3fn))
            et.weight_scale_inv.weight_loader(et.weight_scale_inv, torch.from_numpy(s).view(torch.float8_e8m0fnu))
            m.wkv.weight.weight_loader(m.wkv.weight, wkv)
            m.q_weight.data.copy_(q_w)
            m.k_weight.data.copy_(k_w)
    assert getattr(m_nv.embed_tokens, "_ds41h_nvme", None) is not None, "NVMe storage not selected"
    assert getattr(m_st.embed_tokens, "_ds41h_nvme", None) is None
    assert m_nv.embed_tokens.weight.numel() == 0 and not m_nv.embed_tokens.weight.is_cuda, "NVMe table must not be allocated"
    return m_st, m_nv, sizes, cfg


def _module_cases(eg, vcfg, m_st, m_nv, sizes, cfg, Ts=(8, 300)) -> list[tuple[str, bool, str]]:
    """Engram.forward (prepare_embeddings + forward) of both modules, eager and (T=8) captured in a CUDA graph."""
    from vllm.config import set_current_vllm_config

    teh = _eh_tests()
    res = []
    with set_current_vllm_config(vcfg):
        for T in Ts:
            gh = torch.Generator(device="cpu").manual_seed(T)
            hashes = torch.stack([teh.head_ids(T, sizes, seed=T, layer=k) for k in range(2)], 1)
            X = (torch.randn(T, 4, cfg.hidden_size, generator=gh, device="cpu") * 0.5).to(torch.bfloat16).cuda()
            mask = (torch.rand(T, generator=gh, device="cpu") > 0.1).cuda()
            outs = []
            for m in (m_st, m_nv):
                m.prepare_embeddings(hashes[:, 0])
                outs.append(m(X, hashes[:, 0], mask))
            torch.cuda.synchronize()
            ok, msg = bits_equal(outs[1], outs[0])
            res.append((f"eager T={T}", ok, msg))
        # CUDA graph (decode size): capture prepare + forward of the NVMe module, replay with new hashes
        T = 8
        hashes = torch.stack([teh.head_ids(T, sizes, seed=1, layer=k) for k in range(2)], 1).contiguous()
        X = (torch.randn(T, 4, cfg.hidden_size) * 0.5).to(torch.bfloat16).cuda()
        mask = torch.ones(T, dtype=torch.bool, device="cuda")
        from vllm.distributed.parallel_state import graph_capture

        g = torch.cuda.CUDAGraph()
        with graph_capture(torch.device("cuda", torch.cuda.current_device())) as gc:  # vLLM's capture context (TP comms)
            with torch.cuda.stream(gc.stream):
                for _ in range(2):  # warm up on the capture stream
                    m_nv.prepare_embeddings(hashes[:, 0])
                    m_nv(X, hashes[:, 0], mask)
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=gc.stream):
                    m_nv.prepare_embeddings(hashes[:, 0])
                    out_g = m_nv(X, hashes[:, 0], mask)
        torch.cuda.synchronize()
        bad = 0
        for it in range(20):
            new = torch.stack([teh.head_ids(T, sizes, seed=500 + it, layer=k) for k in range(2)], 1)
            hashes.copy_(new)
            X.copy_((torch.randn(T, 4, cfg.hidden_size) * 0.5).to(torch.bfloat16))
            g.replay()
            m_st.prepare_embeddings(hashes[:, 0])
            ref = m_st(X, hashes[:, 0], mask)
            torch.cuda.synchronize()
            ok, msg = bits_equal(out_g, ref)
            bad += not ok
        res.append(("graph T=8 x20 replays", bad == 0, f"{20 - bad}/20 replays bit-identical"))
    return res


def test_patched_engram_module():
    """Stock vLLM Engram (TP=1) with the NVMe table vs the stock GPU table: Engram.forward bit-identical (eager T=8 /
    300 and 20 CUDA-graph replays), the NVMe table allocates nothing on the GPU."""
    require_cuda()
    from ds41h import patch_engram

    eg = _eh_tests()._stock()
    try:
        vcfg = _vllm_setup(1, 0, 29500 + os.getpid() % 2000, 512)
    except Exception as e:  # noqa: BLE001
        skip(f"vLLM single-rank setup failed: {type(e).__name__}: {e}")
    patch_engram.patch_engram_nvme_module(eg)
    dtype0 = torch.get_default_dtype()
    try:
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        m_st, m_nv, sizes, cfg = _build_modules(eg, vcfg, os.path.join(TEST_DIR, "module.bin"))
        res = _module_cases(eg, vcfg, m_st, m_nv, sizes, cfg)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(dtype0)
        _en().reader().stop()
    for name, ok, msg in res:
        print(f"  {name}: {msg}")
    assert all(ok for _, ok, _ in res), [m for n, ok, m in res if not ok]


def _tp4_worker(rank: int, n: int, port: int, path: str, q) -> None:
    try:
        torch.cuda.set_device(rank)
        os.environ["DS41H_ENGRAM_NVME_STATS_S"] = "0"
        from ds41h import patch_engram

        eg = _eh_tests()._stock()
        vcfg = _vllm_setup(n, rank, port, 1024)
        patch_engram.patch_engram_nvme_module(eg)
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        m_st, m_nv, sizes, cfg = _build_modules(eg, vcfg, path, source="real")
        res = _module_cases(eg, vcfg, m_st, m_nv, sizes, cfg, Ts=(8, 600))
        st = _en().reader().stats()
        q.put((rank, res, "", st))
        _en().reader().stop()
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, [], f"{type(e).__name__}: {e}\n{traceback.format_exc()}", []))


def tp4(argv: list[str]) -> int:
    """TP=4 (4 GPUs, one table file of REAL checkpoint rows -- a window of layer 1's table under the test module's small
    bucket layout -- shared by 4 processes): the patched stock Engram module at TP=4 -- every rank reads only its head
    shard from the file, the stock all-gather assembles the rows -- vs stock's head-sharded GPU tables holding the same
    real rows, Engram.forward bit-exact on every rank, eager T=8 / 600 and 20 CUDA-graph replays of T=8."""
    import torch.multiprocessing as mp

    os.makedirs(TEST_DIR, exist_ok=True)
    path = os.path.join(TEST_DIR, "tp4.bin")
    n = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 2000
    # the table file is written once up front (the workers only open it)
    cfg = _Cfg()
    eg = _eh_tests()._stock()
    from vllm.config import EngramConfig, VllmConfig, set_current_vllm_config

    vcfg = VllmConfig()
    vcfg.engram_config = EngramConfig(cpu_offload=False, dp_shared_memory=False)
    with set_current_vllm_config(vcfg):
        lay = eg.EngramLayout(cfg)
        sizes = [p for order in lay.primes[0] for p in order]
    w, s = real_window(sum(sizes) + 17)
    write_table(path, w, s)
    ps = [ctx.Process(target=_tp4_worker, args=(r, n, port, path, q)) for r in range(n)]
    for p in ps:
        p.start()
    out = sorted((q.get(timeout=900) for _ in ps), key=lambda x: x[0])
    for p in ps:
        p.join(120)
    bad = 0
    for rank, res, err, st in out:
        if err:
            print(f"  rank {rank}: ERROR {err}")
            bad += 1
        for name, ok, msg in res:
            print(f"  rank {rank} {name}: {msg}")
            bad += not ok
        for d in st:
            print(f"  rank {rank} L{d['layer']}: {d['requests']} requests, {d['unique']} reads of {d['owned']} owned ids")
    print(f"tp{n}: {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


_HP = None


def hash_params():
    global _HP
    if _HP is None:
        _HP = _en().HashParams.from_model(MODEL)
    return _HP


def text_ids(kind: str, n_tokens: int, seed: int = 0) -> np.ndarray:
    """[n_tokens, 2, 24] row ids (CPU n-gram hash) of a document: gsm8k (questions + answers), code (vLLM sources),
    random (uniform token ids)."""
    from transformers import AutoTokenizer

    rng = np.random.default_rng(seed)
    if kind == "random":
        toks = rng.integers(1000, 120000, n_tokens)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if kind == "gsm8k":
            import json

            rows = [json.loads(x) for x in open(GSM8K)]
            order = rng.permutation(len(rows))
            text = "\n\n".join(f"Question: {rows[i]['question']}\nAnswer: {rows[i]['answer']}" for i in order)
        else:
            import glob

            import vllm

            files = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "model_executor", "**", "*.py"),
                                     recursive=True))
            text = "\n".join(open(f).read() for f in files[:400])
        toks = np.asarray(tok(text[: n_tokens * 6])["input_ids"][:n_tokens])
        assert len(toks) == n_tokens, f"not enough {kind} text for {n_tokens} tokens"
    return _en().cpu_hash_ids(toks, hash_params())


def test_cpu_hash_matches_layout():
    """CPU n-gram hash sanity: every id inside its (layer, column) bucket; deterministic; history continuation ==
    one-shot hashing of the whole sequence (the chunked-prefill rule)."""
    hp = hash_params()
    ids = text_ids("gsm8k", 3000)
    lay = real_layout()
    for li in range(2):
        sizes = lay.head_sizes(li)
        offs = np.cumsum([0, *sizes[:-1]])
        assert (ids[:, li] >= offs).all() and (ids[:, li] < offs + np.asarray(sizes)).all(), "id outside its bucket"
    rng = np.random.default_rng(3)
    toks = rng.integers(1000, 120000, 500)
    full = _en().cpu_hash_ids(toks, hp)
    part = _en().cpu_hash_ids(toks[300:], hp, history=toks[:300])
    assert (full[300:] == part).all(), "history continuation differs from one-shot hashing"
    print(f"  3000 gsm8k tokens: ids in their buckets; unique ids per layer {[len(np.unique(ids[:, li])) for li in range(2)]} "
          f"of {ids.shape[0] * ids.shape[2]}")


def test_cpu_prefetch():
    """Prefetch (C++ port of the stock hash + low-priority reads into the record cache): the prefetched rows are exactly
    the CPU-hash ids of the document (this rank's columns), and a later request for them is served from the cache with
    the file's bytes (no reads)."""
    en = _en()
    lay = real_layout()
    hp = hash_params()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import json

    rows_ = [json.loads(x) for x in open(GSM8K)][:40]
    toks = tok("\n\n".join(r["question"] + r["answer"] for r in rows_))["input_ids"][:3000]
    rank = 2
    rd = new_reader("direct", cpu_only=True, rank=rank, DS41H_ENGRAM_NVME_PREFETCH="1")
    chans = []
    for li, layer in enumerate(lay.layer_ids):
        sizes = lay.head_sizes(li)
        vs, ve = sum(sizes[: rank * PART]), sum(sizes[: (rank + 1) * PART])
        chans.append((rd.add_channel(real_table(layer), vs, ve, 8192 * PART, li, rank * PART, PART), vs, ve, layer))
    rd.set_hash(hp)
    rd.start()
    rd.prefetch(toks, 0, 1800)  # two jobs: [0, 1800) and [1800, 3000) with history
    rd.prefetch(toks, 1800, len(toks))
    t0 = time.perf_counter()
    while True:
        st = rd.pf_stats()
        if st["queued"] == 0 and st["inflight"] == 0 and st["tokens"] >= len(toks) and time.perf_counter() - t0 > 0.2:
            break
        assert time.perf_counter() - t0 < 60, f"prefetch did not finish: {st}"
        time.sleep(0.05)
    ids = en.cpu_hash_ids(toks, hp)  # [T, 2, 24]
    for li, (ch, vs, ve, layer) in enumerate(chans):
        mine = np.unique(ids[:, li, rank * PART : (rank + 1) * PART].reshape(-1))
        mine = mine[(mine >= vs) & (mine < ve)]
        cached = en.ext().channel_cache_count(ch.handle, torch.from_numpy(mine.astype(np.int32)))
        assert cached >= 0.999 * len(mine), f"L{layer}: only {cached} of {len(mine)} prefetched rows cached"
        # a request for the document's ids is now served from the cache
        req = ids[:, li, rank * PART : (rank + 1) * PART].reshape(-1).astype(np.int32)
        before = en.stats_dict(en.ext().channel_stats(ch.handle))
        assert en.ext().cpu_request(ch.handle, torch.from_numpy(req), 30.0) > 0
        after = en.stats_dict(en.ext().channel_stats(ch.handle))
        _check_channel(en, ch, real_table(layer), req, vs, ve)
        print(f"  L{layer}: {len(mine)} unique rows prefetched ({cached} in cache); request of {len(req)} ids: "
              f"{after['cache_hits'] - before['cache_hits']} hits, {after['ios'] - before['ios']} reads")
    print(f"  prefetch stats: {rd.pf_stats()}")
    rd.stop()


def test_cpu_prefetch_concurrent():
    """A large prefetch (60K random tokens -> ~360K reads per layer) runs while decode-sized requests keep arriving:
    prefetch completions land inside the requests' I/O loops (routed by tag to the cache), requests are always
    correct, the prefetch finishes, nothing leaks (in-flight back to 0)."""
    en = _en()
    lay = real_layout()
    hp = hash_params()
    rng = np.random.default_rng(5)
    rank = 0
    rd = new_reader("direct", cpu_only=True, rank=rank, DS41H_ENGRAM_NVME_PREFETCH="1", DS41H_ENGRAM_NVME_CACHE_MB="1024")
    chans = []
    for li, layer in enumerate(lay.layer_ids):
        sizes = lay.head_sizes(li)
        vs, ve = sum(sizes[: rank * PART]), sum(sizes[: (rank + 1) * PART])
        chans.append((rd.add_channel(real_table(layer), vs, ve, 1024 * PART, li, rank * PART, PART), vs, ve, layer))
    rd.set_hash(hp)
    rd.start()
    toks = rng.integers(1000, 120000, 60000).tolist()
    rd.prefetch(toks, 0, len(toks))
    t0 = time.perf_counter()
    lat, checked, n_req = [], 0, 0
    while True:
        st = rd.pf_stats()
        for ch, vs, ve, layer in chans:
            ids = rng.integers(vs, ve, 8 * PART).astype(np.int32)
            ns = en.ext().cpu_request(ch.handle, torch.from_numpy(ids), 30.0)
            assert ns > 0, "request timed out during prefetch"
            lat.append(ns / 1e3)
            n_req += 1
            if n_req % 5 == 0:
                checked += _check_channel(en, ch, real_table(layer), ids, vs, ve)
        if st["queued"] == 0 and st["inflight"] == 0 and st["tokens"] >= len(toks) - 3 and time.perf_counter() - t0 > 1:
            break
        assert time.perf_counter() - t0 < 300, f"prefetch did not finish: {st}"
        time.sleep(0.002)
    st = rd.pf_stats()
    rd.stop()
    lat.sort()
    reads = [c["reads"] for c in st["channels"]]
    print(f"  prefetch of {len(toks)} tokens ({reads} reads) in {time.perf_counter() - t0:.1f} s alongside {n_req} decode-sized "
          f"requests ({checked} ids verified): request p50 {lat[len(lat) // 2]:.0f} / p99 {lat[int(len(lat) * 0.99)]:.0f} us; "
          f"errors {st['errors']}, in flight at the end {st['inflight']}")
    assert st["inflight"] == 0 and st["errors"] == 0


# ======================================================================================== CPU-only I/O benchmark
def _diskstats(dev: str = "nvme0n1") -> tuple[int, int]:
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == dev:
            return int(f[3]), int(f[5])  # reads completed, sectors read
    return 0, 0


def _io_worker(rank: int, nprocs: int, a: dict, barrier, q) -> None:
    try:
        en = _en()
        lay = real_layout()
        extra = {} if a["cache_mb"] is None else {"DS41H_ENGRAM_NVME_CACHE_MB": str(a["cache_mb"])}
        rd = new_reader(a["mode"], cpu_only=True, rank=rank, DS41H_ENGRAM_NVME_DEPTH=str(a["depth"]),
                        DS41H_ENGRAM_NVME_STATS_S="0", **extra)
        chans = []
        for li, layer in enumerate(lay.layer_ids):
            sizes = lay.head_sizes(li)
            head0 = rank * PART
            vs, ve = sum(sizes[:head0]), sum(sizes[: head0 + PART])
            ch = rd.add_channel(real_table(layer), vs, ve, max(a["tokens"], a["decode_tokens"]) * PART)
            offs = np.cumsum([0, *sizes[:-1]])
            chans.append((ch, offs[head0 : head0 + PART], np.array(sizes[head0 : head0 + PART])))
        rd.start()
        rng = np.random.default_rng(1000 * a["seed"] + rank)
        doc = None
        if a["text"] != "uniform":  # real text through the CPU n-gram hash: [T, 2, 24] row ids of one document
            doc = text_ids(a["text"], a["tokens"] + a["steps"] * a["decode_tokens"], a["seed"])
            doc_pos = [a["tokens"]]  # decode continues after the prefill part
        pools = [None, None]
        if a["pool"]:
            pools = [np.stack([rng.integers(0, sz, a["pool"]) for sz in c[2]], 1) for c in chans]

        def ids_for(li, T, phase="decode"):
            ch, offs, sz = chans[li]
            if doc is not None:
                head0 = rank * PART
                if phase == "prefill":
                    rows = doc[: a["tokens"], li, head0 : head0 + PART]
                else:
                    rows = doc[doc_pos[0] : doc_pos[0] + T, li, head0 : head0 + PART]
                return torch.from_numpy(np.ascontiguousarray(rows).astype(np.int32).reshape(-1))
            if pools[li] is not None:
                pick = pools[li][rng.integers(0, a["pool"], T)]
                return torch.from_numpy((pick + offs).astype(np.int32).reshape(-1))
            return torch.from_numpy((rng.integers(0, sz, (T, PART)) + offs).astype(np.int32).reshape(-1))

        out = {}
        phases = (("decode", a["decode_tokens"], a["steps"]), ("prefill", a["tokens"], a["prefill_steps"]))
        if doc is not None:  # a document: prefill its first `tokens`, then decode the continuation
            phases = phases[::-1]
        for phase, T, steps in phases:
            if not steps:
                continue
            lat = [[], []]
            d0 = _diskstats()
            barrier.wait()
            t0 = time.perf_counter()
            for _ in range(steps):
                ids = [ids_for(0, T, phase), ids_for(1, T, phase)]
                if doc is not None and phase == "decode":
                    doc_pos[0] += T
                barrier.wait()  # TP lockstep
                gens = [en.ext().cpu_publish(chans[li][0].handle, ids[li]) for li in (0, 1)]
                for li in (0, 1):
                    ns = en.ext().cpu_wait(chans[li][0].handle, gens[li], 120.0)
                    lat[li].append(ns / 1e3)
                if phase == "decode" and a["gap_ms"]:
                    time.sleep(a["gap_ms"] / 1e3)
            wall = time.perf_counter() - t0
            barrier.wait()
            d1 = _diskstats()
            out[phase] = (lat, wall, (d1[0] - d0[0], d1[1] - d0[1]))
        q.put((rank, out, rd.stats(), ""))
        rd.stop()
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, {}, [], f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def io_bench(argv: list[str]) -> int:
    """TP-like I/O benchmark without GPUs: `procs` processes (= TP ranks) each read their head shard of BOTH real
    tables through the engine, publishing layer-1 + layer-14 requests back to back per step like the issue kernels.
    decode: `decode_tokens` tokens per step (k=7: 8), `steps` steps, `gap_ms` between steps; prefill: `tokens` per
    step. Ids uniform in each head bucket (cold, worst case) or drawn from a `pool` of ids per head (repetitive text)."""
    import argparse

    import torch.multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gap-ms", type=float, default=8.0)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--prefill-steps", type=int, default=2)
    ap.add_argument("--pool", type=int, default=0, help="ids per head drawn from a pool of this size (0: uniform)")
    ap.add_argument("--text", default="uniform", choices=["uniform", "gsm8k", "code", "random"],
                    help="real row ids of a document (CPU n-gram hash): prefill its first --tokens, decode the rest")
    ap.add_argument("--cache-mb", type=float, default=None, help="DS41H_ENGRAM_NVME_CACHE_MB for the run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    a = vars(args)
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(args.procs), ctx.Queue()
    ps = [ctx.Process(target=_io_worker, args=(r, args.procs, a, barrier, q)) for r in range(args.procs)]
    for p in ps:
        p.start()
    res = sorted((q.get(timeout=1800) for _ in ps), key=lambda x: x[0])
    for p in ps:
        p.join(60)
    for rank, _, _, err in res:
        if err:
            print(f"rank {rank}: ERROR {err}")
            return 1
    print(f"io-bench: {args.procs} procs, mode {args.mode}, depth {args.depth}, ids {args.text} pool {args.pool or '-'}, "
          f"record cache {args.cache_mb if args.cache_mb is not None else 'default'} MB")
    for phase in ("decode", "prefill"):
        if phase not in res[0][1]:
            continue
        lat1 = [x for r in res for x in r[1][phase][0][0]]
        lat14 = [x for r in res for x in r[1][phase][0][1]]
        wall = max(r[1][phase][1] for r in res)
        dev_reads, dev_sect = res[0][1][phase][2]
        T = args.decode_tokens if phase == "decode" else args.tokens
        steps = args.steps if phase == "decode" else args.prefill_steps
        reads = steps * T * PART * 2 * args.procs

        def pct(v, p):
            v = sorted(v)
            return v[min(len(v) - 1, int(p * len(v)))]

        print(f"  {phase}: T={T} x {steps} steps: L1 ready p50 {pct(lat1, 0.5):.0f} / p90 {pct(lat1, 0.9):.0f} / p99 "
              f"{pct(lat1, 0.99):.0f} / max {max(lat1):.0f} us; L14 p50 {pct(lat14, 0.5):.0f} / p99 {pct(lat14, 0.99):.0f} us; "
              f"wall {wall:.2f} s; {reads} ids -> {reads / max(wall, 1e-9) / 1e3:.0f}k ids/s; device reads {dev_reads} "
              f"({dev_sect * 512 / max(dev_reads, 1):.0f} B avg) = {dev_reads / max(reads, 1):.2f} per id")
    for rank, _, st, _ in res[:1]:
        for d in st:
            print(f"  rank {rank} L{d['layer']}: {d['requests']} req, {d['owned']} owned ids, {d['unique']} unique, "
                  f"{d['cache_hits']} record-cache hits, {d['direct_ios']} O_DIRECT of {d['ios']} reads")
    return 0


if __name__ == "__main__":
    try:
        if "--tp4" in sys.argv:
            rc = tp4([x for x in sys.argv[1:] if x != "--tp4"])
        elif "--io-bench" in sys.argv:
            rc = io_bench([x for x in sys.argv[1:] if x != "--io-bench"])
        else:
            rc = run_module_tests(globals())
    finally:
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    raise SystemExit(rc)
