#!/usr/bin/env python3
"""ds41h-phys Engram on owned rows (layers 1 and 14): build, tables, weights, and the step API.

What Engram computes (DeepSeek-V4.1-Flash as production serves it: stock vLLM 0.32 common/engram.py + ds41h engram_host):
  per row (token at position p of a sequence) and Engram layer L in (1, 14):
    1. hash: the compressed ids (token_map, 129,280 -> 99,092 normalized ids) of positions p, p-1, p-2, p-3 (pad for p-s < 0,
       blocked after an image token) -> rolling XOR of id * multiplier[L][s] -> for n = 2, 3, 4 and 8 heads:
       id = rolling % prime[L][n][h] + offset[L][n][h]  (24 row ids per layer, each in its own prime-sized bucket)
    2. lookup: 24 table rows of 256 e4m3 + 8 ue8m0 (264 B; 272-B packed records in production's /dev/shm host copy,
       97.3 GiB per layer), dequantized exactly to bf16 -> e [24 x 256] = [6144]
    3. kv = e @ wkv^T (wkv fp8 [25600, 6144], 32x32 ue8m0 blocks; W8A16, fp32 accumulation, bf16 out) = 4 keys + 1 value
    4. on the hc stream x [4, 5120] after the previous layer's FFN hc_post (between that post and this layer's attention
       pre, model.py:531-561): gate_j = sigmoid(ssqrt(<x_j, q_j k_j key_j> rms(x_j) rms(key_j) / sqrt(5120))),
       x_j += gate_j * value  (rounded to bf16)
  Steps 1-3 depend only on token ids: they can run as soon as the step's input ids exist (start of the step), beside
  layer 0 (for layer 1) and layers 0..13 (for layer 14). Only step 4 sits on the critical path.

API for the step program (one EngramPhys per rank / device; rows = this rank's StepBatch rows):
    eng = EngramPhys(device, max_rows=..., n_state_slots=...)          # opens the host tables, loads + repacks weights
    args = eng.step_args(batch, lookback_fn)                            # host: int32 arrays of a serve.interfaces.StepBatch
    eng.upload(args)                                                    # into the persistent device buffers (graph-safe)
    eng.prepare(T, stream=side)                                         # hash + gather (both layers) + wkv (both layers)
    ...layer 0...
    eng.inject(0, T, y=y, rs=rs, post=post, comb=comb, out=x1)          # layer 1: hc_post + Engram on the main stream
    ...layers 1..13...
    eng.inject(1, T, ...)                                               # layer 14
`inject` waits for the side work of its layer (events), so the caller only orders the streams once per step.
"""
from __future__ import annotations

import fcntl
import json
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.dirname(HERE)
MODEL = os.environ.get("ENGRAM_MODEL", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
LAYERS = (1, 14)
NC, DIM, D, HC = 24, 256, 5120, 4
KVW = (HC + 1) * D
RING, RING_W = 16, 8
DEAD_TOKEN = 129264
PACKED_STRIDE = 272
FL = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-lineinfo"]


# ------------------------------------------------------------------------------------------------ build
@cache
def ext():
    from torch.utils.cpp_extension import load

    bd = os.path.join(HERE, "build_ext")
    os.makedirs(bd, exist_ok=True)
    return load(name="engram_phys_ext", sources=[os.path.join(HERE, "engram.cu")], build_directory=bd,
                extra_cuda_cflags=FL, verbose=False)


@cache
def w8a16_mod():
    """The fork's fused W8A16 GEMM (kernels/attn_dense/w8a16.cu, read-only source), built under an engram_ name."""
    import sys

    from torch.utils.cpp_extension import load

    ad = os.path.join(KD, "attn_dense")
    if ad not in sys.path:
        sys.path.insert(0, ad)
    import w8a16 as W  # noqa: E402  (python helpers: repack / Gemm)

    bd = os.path.join(HERE, "build_w8a16")
    os.makedirs(bd, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-DW8_ABL=0", "-DW8_PROF=0",
             f"-DW8_DQ={int(os.environ.get('W8A16_DQ', '1'))}"]
    e = load(name="engram_w8a16_ext", sources=[os.path.join(ad, "w8a16.cu")], extra_cuda_cflags=flags, extra_ldflags=["-lcuda"],
             extra_include_paths=[ad], build_directory=bd, verbose=False)
    return W, e


# ------------------------------------------------------------------------------------------------ layout + hash params
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass
class Layout:
    """= stock EngramLayout + compute_hash_multipliers (common/engram.py:189-252), from the checkpoint config."""
    layer_ids: tuple
    num_embeddings: tuple
    primes: np.ndarray       # [2, 24] int64 (column c = (n - 2) * 8 + head)
    offsets: np.ndarray      # [2, 24] int64
    mult: np.ndarray         # [2, 4] int64
    compressed_vocab: int
    pad_token_id: int
    lo: np.ndarray = field(default=None)   # [2, 24] production per-column ownership ranges (TP4 head shards)
    hi: np.ndarray = field(default=None)

    @staticmethod
    def from_config(cfg: dict, tp: int = 4) -> "Layout":
        ids = tuple(cfg["engram_layer_ids"])
        n_heads, max_ng = cfg["engram_n_heads"], cfg["engram_max_ngram_size"]
        seen, primes = set(), []
        for _ in ids:
            row = []
            for _ in range(max_ng - 1):
                cur = cfg["engram_vocab_size"] - 1
                for _ in range(n_heads):
                    cur += 1
                    while not _is_prime(cur) or cur in seen:
                        cur += 1
                    seen.add(cur)
                    row.append(cur)
            primes.append(row)
        primes = np.array(primes, dtype=np.int64)
        offsets = np.stack([np.cumsum([0, *p[:-1]]) for p in primes]).astype(np.int64)
        V = cfg["engram_compressed_vocab_size"]
        bound = max(1, (np.iinfo(np.int64).max // V) // 2)
        mult = np.stack([np.random.default_rng(10007 * lid).integers(0, bound, size=(max_ng,), dtype=np.int64) * 2 + 1
                         for lid in ids])
        lay = Layout(ids, tuple(cfg["engram_num_embeddings"]), primes, offsets, mult, V, cfg["engram_pad_token_id"])
        # production ownership (ds41h engram_host.head_ranges): column c belongs to the TP rank holding head c
        part = -(-NC // tp)
        lay.lo = np.zeros((len(ids), NC), np.int64)
        lay.hi = np.zeros((len(ids), NC), np.int64)
        for li in range(len(ids)):
            for c in range(NC):
                r = c // part
                lay.lo[li, c] = int(primes[li, : r * part].sum())
                lay.hi[li, c] = int(primes[li, : (r + 1) * part].sum())
        return lay


def model_config(model_dir: str = MODEL) -> dict:
    c = json.load(open(os.path.join(model_dir, "config.json")))
    return c.get("text_config", c)


def token_map(model_dir: str = MODEL, cache_dir: str | None = None) -> tuple[np.ndarray, int]:
    """stock build_compressed_token_map on the model's tokenizer (cached as .npy next to this file)."""
    cache_dir = cache_dir or os.path.join(HERE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, "token_map.npy")
    if os.path.exists(f):
        m = np.load(f)
        return m, int(m.max()) + 1
    from transformers import AutoTokenizer

    from vllm.models.deepseek_v41.common.engram import build_compressed_token_map

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    lookup, n = build_compressed_token_map(tok)
    m = np.array(lookup, dtype=np.int32)
    np.save(f, m)
    return m, n


def hash_ref_np(tokens: np.ndarray, start: int, lookback: list, lay: Layout, tmap: np.ndarray, pad_id: int) -> np.ndarray:
    """CPU reference of stock `_hash_ids_kernel` (V2 path) for ONE sequence's rows: tokens [n] at positions start..start+n-1,
    lookback = raw ids at start-1, start-2, start-3 (-1 unknown). Returns [n, 2, 24] int64. Python ints: exact."""
    n = len(tokens)
    out = np.zeros((n, len(lay.layer_ids), NC), dtype=np.int64)
    for i in range(n):
        p = start + i
        roll = [0] * len(lay.layer_ids)
        blocked = False
        for sh in range(4):
            lp = p - sh
            if lp >= start:
                tk = int(tokens[i - sh])
                src = -1 if tk == DEAD_TOKEN else int(tmap[tk])
            else:
                col = start - 1 - lp
                wt = int(lookback[col]) if 0 <= col < 3 else -1
                src = (-1 if wt == DEAD_TOKEN else int(tmap[wt])) if wt >= 0 else pad_id
            blocked = blocked or lp < 0 or src == -1
            v = pad_id if blocked else src
            for li in range(len(lay.layer_ids)):
                roll[li] ^= (v * int(lay.mult[li, sh])) & 0xFFFFFFFFFFFFFFFF
                if sh > 0:
                    for hd in range(8):
                        c = (sh - 1) * 8 + hd
                        out[i, li, c] = roll[li] % int(lay.primes[li, c]) + int(lay.offsets[li, c])
    return out


# ------------------------------------------------------------------------------------------------ host tables
MAGIC = b"DS41HENG"
_GEOM = struct.Struct("<IIQIIQQQQQ")


class HostTable:
    """Production's shared Engram table (ds41h engram_host packed layout), used read-only: LOCK_SH on the data file (marks
    it in use, so production's cleanup never unlinks it under us), MAP_SHARED mapping, cudaHostRegister. The host's driver
    refuses cudaHostRegisterReadOnly (801, operation not supported), so the mapping is PROT_READ | PROT_WRITE exactly as
    production maps it; nothing in this module writes to it (the kernels only issue ld.global.nc loads from it), and it is
    never unlinked by this process."""

    def __init__(self, layer: int, directory: str = "/dev/shm", rows: int | None = None):
        cands = sorted(Path(directory).glob(f"ds41h_engram_L{layer}_packed_*.bin"))
        if not cands:
            raise FileNotFoundError(f"no packed Engram table for layer {layer} in {directory}")
        self.path = cands[0]
        self.fd = os.open(self.path, os.O_RDWR)
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        hdr = self.mm[:4096]
        assert hdr[:8] == MAGIC, f"{self.path}: bad magic"
        ver, layout, nrows, dim, sdim, w_off, s_off, w_stride, s_stride, total = _GEOM.unpack_from(hdr, 8)
        assert layout == 1 and dim == DIM and sdim == 8 and w_stride == PACKED_STRIDE and s_off == w_off + DIM, (layout, dim)
        assert struct.unpack_from("<I", hdr, 120)[0] == 0xA5A5A5A5, "table header not ready"
        n_sh = struct.unpack_from("<I", hdr, 1024)[0]
        st = hdr[1028 : 1028 + 2 * n_sh]
        assert all(b == 2 for b in st), f"{self.path}: shards not complete ({list(st)})"
        if rows is not None:
            assert nrows == rows, (nrows, rows)
        self.rows, self.w_off, self.size = nrows, w_off, size
        self.key = hdr[132 : 132 + struct.unpack_from("<I", hdr, 128)[0]].decode(errors="replace")
        import ctypes

        self._anchor = ctypes.c_char.from_buffer(self.mm)  # host address of the mapping (keeps the buffer exported)
        self.base = ctypes.addressof(self._anchor)
        self.dev = 0
        self._chunks = []       # host addresses of this table's registrations (one, or the chunks of register_tables)

    def register(self, threads: int | None = None, chunk: int | None = None) -> float:
        """cudaHostRegister(Mapped | Portable) of the whole mapping; returns seconds. threads (default: env
        ENGRAM_REG_THREADS, else REG_THREADS) > 1: populate_tables, then page-aligned chunks from that many threads
        (register_tables). threads = 1: the original path (one call, no populate; the reference in init_bench.py)."""
        threads = reg_threads() if threads is None else threads
        if threads > 1:
            return register_tables([self], threads, chunk)["seconds"]
        t0 = time.perf_counter()
        err, msg, dptr = ext().host_register(self.base, self.size, False)
        if err:
            raise RuntimeError(f"cudaHostRegister of {self.path} failed: {msg} ({err})")
        self.dev = dptr
        self._chunks = [self.base]
        return time.perf_counter() - t0

    def unregister(self, threads: int = 1) -> float:
        t0 = time.perf_counter()
        if self.dev:
            unregister_tables([self], threads)
        return time.perf_counter() - t0

    @property
    def rec0(self) -> int:
        """Device address of record 0."""
        return self.dev + self.w_off

    def host_record(self, i: int) -> bytes:
        o = self.w_off + i * PACKED_STRIDE
        return self.mm[o : o + 264]

    def close(self):
        if self.dev:
            unregister_tables([self])
        self._anchor = None
        try:
            self.mm.close()
        except BufferError:
            pass
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


# ------------------------------------------------------------------------------------------------ host registration
# cudaHostRegister pins every 4 KiB page of the mapping (/dev/shm has no THP here) and maps it for every device of
# the process: in a fresh process it first faults in this process's 51M
# page-table entries (populate_tables does that beforehand from 32 threads, ~1 s); the driver's own part (~2.4 s per
# GPU for both tables) is serialized inside the driver whatever the thread count, and stalls every other CUDA call of
# the process while it runs. Two threads (one chunk per table) overlap one table's pinning with the other's mapping.
# Through ctypes (torch's own libcudart, by path), which drops the GIL for each call. UVA (cudaDevAttrCanUseHostPointer
# ForRegisteredMem): the device address of registered memory is its host address, so the chunks of one table form ONE
# device range (checked per chunk); without it a table is registered whole (one thread per table).
REG_THREADS = 2         # HostTable.register: populate + 2 chunks from 2 threads (1 GPU, fresh: 10.7 -> 7.6 s)
REG_ASYNC_THREADS = 2   # register_tables of both tables: one chunk per table (no gain beyond 2 threads)
POP_THREADS = 32        # populate_tables: madvise(MADV_POPULATE_WRITE) threads (both tables, fresh process)
POP_CHUNK_MB = 16       # MiB per madvise call (each holds the mmap lock): 1.0 s, a weight load beside it +0.17 s (1 GiB: 2.5 s, +2.7 s)
MADV_POPULATE_WRITE = 23
REG_ALIGN = 2 << 20
_CUDART = None


def reg_threads() -> int:
    return int(os.environ.get("ENGRAM_REG_THREADS", REG_THREADS))


def _cudart():
    global _CUDART
    if _CUDART is None:
        import ctypes as C

        torch.cuda.init()
        # the copy torch already loaded (by path: never a second runtime from the system library path)
        names = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "/libcudart.so" in ln})
        names += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        lib = None
        for name in names:
            try:
                lib = C.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libcudart not found")
        lib.cudaHostRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
        lib.cudaHostRegister.restype = C.c_int
        lib.cudaHostUnregister.argtypes = [C.c_void_p]
        lib.cudaHostUnregister.restype = C.c_int
        lib.cudaHostGetDevicePointer.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint]
        lib.cudaHostGetDevicePointer.restype = C.c_int
        lib.cudaSetDevice.argtypes = [C.c_int]
        lib.cudaSetDevice.restype = C.c_int
        lib.cudaDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        lib.cudaDeviceGetAttribute.restype = C.c_int
        lib.cudaGetErrorString.argtypes = [C.c_int]
        lib.cudaGetErrorString.restype = C.c_char_p
        lib.cudaGetLastError.restype = C.c_int
        lib.cudaFree.argtypes = [C.c_void_p]
        lib.cudaFree.restype = C.c_int
        _CUDART = lib
    return _CUDART


def _cuerr(lib, e: int) -> str:
    return (lib.cudaGetErrorString(e) or b"?").decode()


def host_ptr_is_device_ptr(device: int | None = None) -> bool:
    import ctypes as C

    lib = _cudart()
    v = C.c_int(0)
    d = torch.cuda.current_device() if device is None else device
    e = lib.cudaDeviceGetAttribute(C.byref(v), 91, d)  # cudaDevAttrCanUseHostPointerForRegisteredMem
    return e == 0 and v.value == 1


def populate_tables(tables, threads: int | None = None, chunk: int | None = None) -> float:
    """Fault in this process's page-table entries for the whole mappings (madvise MADV_POPULATE_WRITE, 1 GiB pieces
    from `threads` threads, GIL released; no data is read or written). A fresh process otherwise takes those 51M page
    faults inside cudaHostRegister, under the driver's lock (1 GPU: 9.9 s first registration vs 4.4 s with the entries
    present). Pure kernel work, no CUDA: it can run beside anything. Returns seconds; errors (kernel < 5.14) are
    ignored (the registration then faults the pages itself)."""
    import ctypes as C
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    libc = C.CDLL("libc.so.6", use_errno=True)
    libc.madvise.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
    libc.madvise.restype = C.c_int
    chunk = chunk or int(os.environ.get("ENGRAM_POP_CHUNK_MB", POP_CHUNK_MB)) << 20
    threads = threads or int(os.environ.get("ENGRAM_POP_THREADS", POP_THREADS))
    jobs = [(t.base + o, min(chunk, t.size - o)) for t in tables if not getattr(t, "_populated", False)
            for o in range(0, t.size, chunk)]
    threads = max(1, POP_THREADS if threads is None else threads)

    def work(job):
        return libc.madvise(C.c_void_p(job[0]), job[1], MADV_POPULATE_WRITE)

    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(jobs)))) as ex:
        bad = sum(1 for r in ex.map(work, jobs) if r)
    if not bad:
        for t in tables:
            t._populated = True
    return time.perf_counter() - t0


def register_tables(tables, threads: int | None = None, chunk: int | None = None, populate: bool = True) -> dict:
    """Register every not yet registered HostTable in `tables`, in page-aligned chunks from `threads` threads (GIL
    released). chunk: bytes per registration (default: the tables' total / threads, 2 MiB aligned). Sets t.dev (device
    address of the mapping) and t._chunks. Same flags as HostTable.register (Mapped | Portable): one registration serves
    every device of the process. populate: populate_tables first (parallel page faults outside the driver).
    Returns dict(seconds, populate_s, chunks, threads, chunk)."""
    import ctypes as C
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    lib = _cudart()
    dev = torch.cuda.current_device()
    threads = max(1, reg_threads() if threads is None else threads)
    todo = [t for t in tables if not t.dev]
    if not todo:
        return dict(seconds=0.0, populate_s=0.0, chunks=0, threads=threads, chunk=0)
    pop_s = populate_tables(todo) if populate else 0.0
    same = host_ptr_is_device_ptr(dev)
    total = sum(t.size for t in todo)
    if chunk is None:
        chunk = -(-total // threads)
    chunk = max(REG_ALIGN, -(-chunk // REG_ALIGN) * REG_ALIGN)
    jobs = []
    for t in todo:
        offs = range(0, t.size, chunk) if same else [0]
        jobs += [(t, o, (min(chunk, t.size - o) if same else t.size)) for o in offs]

    def work(job):
        t, o, n = job
        lib.cudaSetDevice(dev)
        hp = t.base + o
        e = lib.cudaHostRegister(C.c_void_p(hp), n, 3)  # cudaHostRegisterPortable | cudaHostRegisterMapped
        if e == 712:                                     # cudaErrorHostMemoryAlreadyRegistered
            lib.cudaGetLastError()
            e = -712
        if e > 0:
            lib.cudaGetLastError()
            return job, e, 0
        dp = C.c_void_p(0)
        e2 = lib.cudaHostGetDevicePointer(C.byref(dp), C.c_void_p(hp), 0)
        if e2:
            lib.cudaGetLastError()
            return job, e2, 0
        return job, e, int(dp.value or 0)

    with ThreadPoolExecutor(max_workers=min(threads, len(jobs))) as ex:
        res = list(ex.map(work, jobs))
    bad = [(j[0].path.name, j[1], e, _cuerr(lib, e)) for j, e, _ in res if e > 0]
    done = {}
    for (t, o, n), e, dp in res:
        if e <= 0:
            done.setdefault(id(t), []).append((o, t.base + o, dp, e == 0))
    if not bad:
        for t in todo:
            cs = sorted(done[id(t)])
            d0 = cs[0][2] - cs[0][0]
            if any(dp - o != d0 for o, _, dp, _ in cs):   # chunks must form one device range (UVA)
                bad.append((t.path.name, "device addresses not contiguous", 0, ""))
                break
            t.dev = d0
            t._chunks = [hp for _, hp, _, own in cs if own]
    if bad:
        for t in todo:
            for o, hp, dp, own in done.get(id(t), []):
                if own:
                    lib.cudaHostUnregister(C.c_void_p(hp))
            t.dev, t._chunks = 0, []
        raise RuntimeError(f"cudaHostRegister failed: {bad[:4]}")
    return dict(seconds=time.perf_counter() - t0, populate_s=pop_s, chunks=len(jobs), threads=threads, chunk=chunk)


class Registration:
    """Open tables, made ready in two parts (init_bench.py, 4 GPUs, fresh process):
      1. populate_tables (~1.0 s in 16 MiB pieces, kernel page faults, no CUDA): in a background thread from the start,
         behind whatever the caller does meanwhile (the engine: extensions + weight load; a weight load beside it ran
         0.88 s instead of 0.71 s);
      2. register_tables (9.8 s on 4 GPUs: the driver's part, serialized inside the driver, ~2.4 s per GPU): in wait(),
         in the caller's thread (mode "populate", the default). While it runs, every other CUDA call of the process waits
         (cudaMalloc, cudaHostAlloc, copies, syncs), so running it behind the weight load gains nothing:
         measured 16.2 s for load + registration together against 3.3 + 12.0 s one after the other.
         mode "register" still runs it in the background thread too (after creating every device's context there).
    wait() -> the tables dict (re-raises an error of the background part)."""

    def __init__(self, tables: dict, threads: int | None = None, chunk: int | None = None, mode: str = "populate"):
        import threading

        assert mode in ("populate", "register"), mode
        if mode == "register":
            _cudart()                   # CUDA init and the runtime handle in the caller's thread
        self.tables, self.info, self.err, self.mode = tables, None, None, mode
        self.threads, self.chunk = threads, chunk
        self.populate_s = self.register_s = 0.0
        self.waited = 0.0
        self.t0 = time.perf_counter()
        self.t_done = None
        self._th = threading.Thread(target=self._run, daemon=True, name="engram-tables")
        self._th.start()

    def _run(self):
        try:
            self.populate_s = populate_tables(list(self.tables.values()))
            if self.mode == "register":
                # every visible device's primary context first, so the Portable registration is made with all of them
                lib = _cudart()
                for i in range(torch.cuda.device_count()):
                    lib.cudaSetDevice(i)
                    lib.cudaFree(None)
                lib.cudaSetDevice(0)
                self.info = register_tables(list(self.tables.values()), self.threads, self.chunk)
                self.register_s = self.info["seconds"] - self.info["populate_s"]
        except BaseException as e:  # noqa: BLE001  (re-raised in wait)
            self.err = e
        self.t_done = time.perf_counter()

    def wait(self) -> dict:
        t = time.perf_counter()
        self._th.join()
        self.waited = time.perf_counter() - t
        if self.err is not None:
            raise self.err
        if self.mode == "populate" and not all(x.dev for x in self.tables.values()):
            self.info = register_tables(list(self.tables.values()), self.threads, self.chunk)   # populate: done
            self.register_s = self.info["seconds"]
        return self.tables


def open_tables_async(threads: int | None = None, mode: str = "populate") -> Registration:
    """open_tables() in two parts: start it early (the page-table populate runs in a background thread), call .wait()
    where the tables are needed (registers them there, mode "populate"). See Registration."""
    cfg = model_config()
    lay = Layout.from_config(cfg)
    return Registration({L: HostTable(L, rows=n) for L, n in zip(LAYERS, lay.num_embeddings)},
                        REG_ASYNC_THREADS if threads is None else threads, mode=mode)


def unregister_tables(tables, threads: int | None = None) -> float:
    """cudaHostUnregister of every registration of the tables (parallel threads); returns seconds."""
    import ctypes as C
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    lib = _cudart()
    dev = torch.cuda.current_device()
    ptrs = [hp for t in tables if t.dev for hp in t._chunks]

    def work(hp):
        lib.cudaSetDevice(dev)
        return lib.cudaHostUnregister(C.c_void_p(hp))

    threads = max(1, reg_threads() if threads is None else threads)
    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(ptrs)))) as ex:
        errs = [e for e in ex.map(work, ptrs) if e]
    for t in tables:
        t.dev, t._chunks = 0, []
    if errs:
        lib.cudaGetLastError()
    return time.perf_counter() - t0


# ------------------------------------------------------------------------------------------------ weights
def load_engram_weights(model_dir: str = MODEL) -> dict:
    """{layer: dict(wkv8 [25600, 6144] e4m3, wkv_s [800, 192] uint8 (ue8m0), q [4, 5120] bf16, k [4, 5120] bf16)} on CPU."""
    from safetensors import safe_open

    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for L in LAYERS:
        p = f"layers.{L}.engram."
        d = {}
        with safe_open(os.path.join(model_dir, wm[p + "wkv.weight"]), framework="pt") as f:
            d["wkv8"] = f.get_tensor(p + "wkv.weight")
            s = f.get_tensor(p + "wkv.scale")
            d["wkv_s"] = s.view(torch.uint8) if s.element_size() == 1 else s
            d["q"] = f.get_tensor(p + "q_weight").to(torch.bfloat16)
            d["k"] = f.get_tensor(p + "k_weight").to(torch.bfloat16)
        assert d["wkv8"].shape == (KVW, NC * DIM) and d["wkv_s"].shape == (KVW // 32, NC * DIM // 32), (d["wkv8"].shape, d["wkv_s"].shape)
        out[L] = d
    return out


def cached_weights(model_dir: str = MODEL) -> dict:
    """load_engram_weights + the W8A16 repack, built on the CPU once (tools/wcache.py, key versioned) and then loaded from
    the page cache (no conversion at load time)."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(KD), "tools"))
    from wcache import cached

    def build():
        W = load_engram_weights(model_dir)
        Wm, _ = w8a16_mod()
        for L in LAYERS:
            W[L]["packed"] = Wm.repack(W[L]["wkv8"], W[L]["wkv_s"])
        return W

    return cached(f"engram_weights_v1_{os.path.basename(model_dir.rstrip('/'))}", build)


def make_gemm8(Wm, e8, packed: torch.Tensor, w8_shape, s8: torch.Tensor):
    """w8a16.Gemm around an already repacked weight (same fields as Gemm.__init__, no repack on the device)."""
    g = Wm.Gemm.__new__(Wm.Gemm)
    g.ext = e8
    g.N, g.K = w8_shape
    g.scales = Wm.check_scales(s8)
    g.packed = packed
    g.splits, g.rpg, g.kxg = 1, int(g.N), 0
    T = g.packed.shape[1]
    g.ws = Wm._shared_ws(packed.device, 1)
    g.cnt = torch.zeros(T, dtype=torch.int32, device=packed.device)
    return g


def dequant_fp8_bf16(w8: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
    """Exact: e4m3 * 2^(s - 127) is representable in bf16 for the checkpoint's scale range (checked)."""
    N, K = w8.shape
    sc = torch.pow(2.0, s8.to(torch.float32) - 127.0)
    w = w8.view(torch.float8_e4m3fn).to(torch.float32).view(N // 32, 32, K // 32, 32) * sc.view(N // 32, 1, K // 32, 1)
    wb = w.view(N, K).to(torch.bfloat16)
    assert torch.equal(wb.float(), w.view(N, K)), "fp8 x ue8m0 not exact in bf16"
    return wb


# ------------------------------------------------------------------------------------------------ step args
@dataclass
class StepArgs:
    """Host-side int32 arrays of one rank's step (from serve.interfaces.StepBatch + the device input ids)."""
    T: int
    pos: np.ndarray | None # [T] (informational: the hash kernel derives positions from row0 / start)
    seq: np.ndarray | None # [T] (informational: index into the step's sequences, -1 = padding)
    row0: np.ndarray       # [S]
    nrows: np.ndarray      # [S]
    start: np.ndarray      # [S]
    slot: np.ndarray       # [S] state slot (ring row)
    look: np.ndarray       # [S, 3] raw ids at start-1..start-3 (-1 unknown, -2 = device ring)
    ids: np.ndarray | None = None  # [T] raw input ids (host copy; production: the head writes them on the device)
    n: int = -1            # real rows (rows n..T-1 are padding); -1 = T


def step_args_from_batch(batch, lookback_fn=None, ids=None, pad_to: int | None = None) -> StepArgs:
    """serve.interfaces.StepBatch -> StepArgs, O(sequences) (no per-row arrays: the hash kernel derives each row's
    sequence and position on the device). lookback_fn(rid, start) -> 3 raw ids at start-1, start-2, start-3 (-1 if < 0)
    for PREFILL chunks (the host knows every prompt token); decode sequences read the device ring (-2).
    pad_to: the graph's row count (rows n_rows..pad_to-1 are padding: inert)."""
    seqs = batch.seqs
    S = len(seqs)
    rec = np.array([(se.row0, se.n_rows, se.start, se.state_slot) for se in seqs], np.int32).reshape(S, 4)
    look = np.full((S, 3), -2, np.int32)
    for i, se in enumerate(seqs):
        if se.kind != "decode" or se.n_rows > RING_W:
            assert lookback_fn is not None, "prefill chunks need host lookback ids"
            lb = list(lookback_fn(se.rid, se.start))
            look[i] = [lb[c] if se.start - 1 - c >= 0 else -1 for c in range(3)]
    n = batch.n_rows
    T = max(n, pad_to or 0)
    idv = None
    if ids is not None:
        idv = np.zeros(T, np.int32)
        idv[:n] = np.asarray(ids, np.int32)[:n]
    return StepArgs(T, None, None, rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3], look, idv, n)


def step_args_from_tables(tab, lookback_fn=None, pad_to: int | None = None) -> StepArgs:
    """The same StepArgs from serve.worker.RankTables (what the engine already holds per rank): its per-sequence arrays
    (seq["start"], seq["n_rows"], seq["state_slot"], seq["kind"]) are numpy-backed, so this is a few vectorized ops;
    Python touches only the prefill sequences (host lookback). ~2-3 us per rank instead of ~20 us at 128 sequences."""
    sq = tab.seq
    n = tab.n
    S = len(tab.seq_rows) if n else 0
    npv = lambda t: t.numpy() if hasattr(t, "numpy") else np.asarray(t)  # noqa: E731  (torch CPU tensors: zero-copy)
    nr = npv(sq["n_rows"])[:S]
    row0 = np.zeros(S, np.int32)
    if S > 1:
        np.cumsum(nr[:-1], out=row0[1:])
    start = npv(sq["start"])[:S]
    slot = npv(sq["state_slot"])[:S]
    kind = npv(sq["kind"])[:S]
    look = np.full((S, 3), -2, np.int32)
    host = np.nonzero((kind != 0) | (nr > RING_W))[0]
    if len(host):
        assert lookback_fn is not None, "prefill chunks need host lookback ids"
        for i in host.tolist():
            rid, st_ = tab.seq_rows[i][0], int(start[i])
            lb = list(lookback_fn(rid, st_))
            look[i] = [lb[c] if st_ - 1 - c >= 0 else -1 for c in range(3)]
    T = max(n, pad_to or 0)
    return StepArgs(T, None, None, row0, nr, start, slot, look, None, n)


def step_args_from_compact(ct, lookback_fn=None, pad_to: int | None = None) -> StepArgs:
    """The same StepArgs from serve's D13 CompactTables (rowexp_ref.CompactTables, async steps): start -1 for decode
    sequences whose position only the device knows; EngramPhys.bind(start=rowexp.out["start"]) resolves those on the
    device. Prefill chunks carry host starts (host lookback ids)."""
    S = len(ct.slot) if ct.slot is not None else 0
    nr = np.asarray(ct.n_rows, np.int32)[:S]
    row0 = np.asarray(ct.row0, np.int32)[:S]
    start = np.asarray(ct.start, np.int64)[:S].astype(np.int32)
    slot = np.asarray(ct.slot, np.int32)[:S]
    kind = np.asarray(ct.kind)[:S]
    look = np.full((S, 3), -2, np.int32)
    host = np.nonzero((kind != 0) | (nr > RING_W))[0]
    if len(host):
        assert lookback_fn is not None, "prefill chunks need host lookback ids"
        for i in host.tolist():
            st_ = int(start[i])
            assert st_ >= 0, "a prefill / long sequence needs a host start"
            lb = list(lookback_fn(ct.rids[i], st_))
            look[i] = [lb[c] if st_ - 1 - c >= 0 else -1 for c in range(3)]
    n = ct.n
    T = max(n, pad_to or 0)
    return StepArgs(T, None, None, row0, nr, start, slot, look, None, n)


# ------------------------------------------------------------------------------------------------ the module
def open_tables(register: bool = True) -> dict:
    """Both layers' HostTables, registered (Portable: valid for every device of the process). In a worker process that
    drives several GPUs, open them ONCE and pass `tables=` to every EngramPhys."""
    cfg = model_config()
    lay = Layout.from_config(cfg)
    t = {L: HostTable(L, rows=n) for L, n in zip(LAYERS, lay.num_embeddings)}
    if register:
        for x in t.values():
            x.register()
    return t


class EngramPhys:
    """Engram for one rank's rows. Persistent device buffers sized for max_rows / max_seqs (CUDA-graph safe).

    wkv: "w8a16" (fork fused W8A16, fp8 weights replicated: 157 MB per layer; <= 128 rows per call) for T <= w8_max_rows,
    else cuBLAS bf16 on the exact dequant (bf16 copy: 315 MB per layer; prefill)."""

    def __init__(self, device, max_rows: int = 512, max_seqs: int = 160, n_state_slots: int = 4096, model_dir: str = MODEL,
                 tables: dict | None = None, weights: dict | None = None, w8_max_rows: int = 256, bf16_copy: bool = True,
                 register: bool = True):
        self.dev = torch.device(device)
        with torch.cuda.device(self.dev):
            self._init(model_dir, tables, weights, w8_max_rows, bf16_copy, register, max_rows, max_seqs, n_state_slots)

    def _init(self, model_dir, tables, weights, w8_max_rows, bf16_copy, register, max_rows, max_seqs, n_state_slots):
        cfg = model_config(model_dir)
        self.cfg = cfg
        self.lay = Layout.from_config(cfg)
        tm, nv = token_map(model_dir)
        assert nv == self.lay.compressed_vocab, (nv, self.lay.compressed_vocab)
        self.pad_id = int(tm[self.lay.pad_token_id])
        self.tmap = torch.from_numpy(tm.astype(np.int32)).to(self.dev)
        self.eps, self.clamp = float(cfg["rms_norm_eps"]), 1e-6
        E = ext()
        self.hctx = E.hash_setup(torch.from_numpy(self.lay.mult), torch.from_numpy(self.lay.primes),
                                 torch.from_numpy(self.lay.offsets), self.pad_id, len(tm))
        self.tables = tables if tables is not None else {L: HostTable(L, rows=n) for L, n in zip(LAYERS, self.lay.num_embeddings)}
        if register:
            for t in self.tables.values():
                if not t.dev:
                    t.register()
        W = weights if weights is not None else cached_weights(model_dir)
        self.q = [W[L]["q"].to(self.dev).contiguous() for L in LAYERS]
        self.k = [W[L]["k"].to(self.dev).contiguous() for L in LAYERS]
        self.w8_max_rows = w8_max_rows
        self.gemm8 = []
        if w8_max_rows > 0:
            Wm, e8 = w8a16_mod()
            for L in LAYERS:
                if "packed" in W[L]:
                    self.gemm8.append(make_gemm8(Wm, e8, W[L]["packed"].to(self.dev), tuple(W[L]["wkv8"].shape), W[L]["wkv_s"]))
                else:
                    self.gemm8.append(Wm.Gemm(e8, W[L]["wkv8"].to(self.dev), W[L]["wkv_s"].to(self.dev), splits=1))
        self.wbf = [dequant_fp8_bf16(W[L]["wkv8"].to(self.dev), W[L]["wkv_s"].to(self.dev)) for L in LAYERS] if bf16_copy else None
        # persistent buffers
        R, S = max_rows, max_seqs
        i32 = dict(dtype=torch.int32, device=self.dev)
        # per-step inputs: ONE packed int32 device block [ids R | row0 S | nrows S | start S | slot S | look 3S | counts 4]
        # (fixed addresses: graphs keep pointing at it) filled by ONE non-blocking H2D copy from pinned staging
        self._pk_off = dict(ids=0, row0=R, nrows=R + S, start=R + 2 * S, slot=R + 3 * S, look=R + 4 * S, counts=R + 7 * S)
        self._pk_n = R + 7 * S + 4
        self._pk = torch.zeros(self._pk_n, **i32)
        o_ = self._pk_off
        self.b_ids = self._pk[o_["ids"]: o_["ids"] + R]
        self.b_row0, self.b_nrows = self._pk[o_["row0"]: o_["row0"] + S], self._pk[o_["nrows"]: o_["nrows"] + S]
        self.b_start, self.b_slot = self._pk[o_["start"]: o_["start"] + S], self._pk[o_["slot"]: o_["slot"] + S]
        self.b_look = self._pk[o_["look"]: o_["look"] + 3 * S].view(S, 3)
        self.b_counts = self._pk[o_["counts"]: o_["counts"] + 4]
        self._stage = [torch.zeros(self._pk_n, dtype=torch.int32, pin_memory=True) for _ in range(2)]
        self._stage_np = [t.numpy() for t in self._stage]
        self._stage_ev = [None, None]
        self._stage_i = 0
        self.ring = torch.full((n_state_slots, RING, 2), -1, **i32)
        self.hash = torch.full((R, 2, NC), -1, **i32)
        self.keep = torch.zeros(R, dtype=torch.uint8, device=self.dev)
        self.err = torch.zeros(3, **i32)
        self.rows = [torch.zeros(R, NC, DIM, dtype=torch.bfloat16, device=self.dev) for _ in LAYERS]
        self.kv = [torch.zeros(R, KVW, dtype=torch.bfloat16, device=self.dev) for _ in LAYERS]
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in LAYERS]
        self.lo = [int(x) for x in self.lay.lo.reshape(-1)]
        self.hi = [int(x) for x in self.lay.hi.reshape(-1)]
        self.max_rows, self.max_seqs = R, S
        # 8 CTAs x 512 threads x 4 records in flight per 16-lane group: the same record rate as a full grid (translation-
        # bound: 25-34 M records/s from 4 CTAs up) at +19 us on a concurrent layer-0 GEMM instead of
        # +130 (16 CTAs) / +273 us (one CTA per 16 records)
        self.gather_cfg = dict(ctas=8, threads=512, unroll=4)
        self.kv_idx = [None, None]  # prefill row dedup: row -> unique kv row (None: identity)

    # -------------------------------------------------------------------------------- per-step inputs
    def bind(self, ids: torch.Tensor | None = None, pos: torch.Tensor | None = None, seq: torch.Tensor | None = None):
        """Read the engine's own persistent int32 input-id buffer (written by the head / host) instead of this module's
        copy. Contiguous int32, >= max rows. pos / seq are accepted for API compatibility and ignored: the hash kernel
        derives positions and sequence indices from the per-sequence arrays."""
        if ids is not None:
            assert ids.dtype == torch.int32 and ids.is_contiguous() and ids.device == self.dev and ids.numel() >= self.max_rows
            self.b_ids = ids
            self._bound = getattr(self, "_bound", set()) | {"b_ids"}

    def bind_start(self, start: torch.Tensor):
        """D13 (async steps): per-sequence starts resolved on the device (kernels/rowexp RowExp.out["start"], int64
        [>= max_seqs], this step's sequence order). Sequences uploaded with start -1 take it in prepare() (one device
        op, graph-capturable); host starts (prefill, forced, first step in a slot) are kept."""
        assert start.device == self.dev and start.numel() >= self.max_seqs, "start: >= max_seqs entries on this device"
        self._start_src = start

    def _resolve_starts(self):
        src = getattr(self, "_start_src", None)
        if src is not None:
            b = self.b_start
            torch.where(b < 0, src[: b.numel()].to(torch.int32), b, out=b)

    def upload(self, a: StepArgs, ids_dev: torch.Tensor | None = None, stream: torch.cuda.Stream | None = None):
        """StepArgs -> pinned staging (numpy, double-buffered) -> the packed device block in ONE non-blocking H2D copy on
        `stream` (default: the device's current stream). No host sync: the host returns at once even while the
        previous step still runs on that stream (the copy is stream-ordered before this step's graph). ids_dev (device)
        is copied stream-ordered too; bound ids are left alone."""
        T, S = a.T, len(a.row0)
        n = a.n if a.n >= 0 else T
        assert T <= self.max_rows and S <= self.max_seqs, (T, S)
        i = self._stage_i
        self._stage_i ^= 1
        if self._stage_ev[i] is not None:
            self._stage_ev[i].synchronize()       # the copy of two uploads ago (long done in a pipelined loop)
        h, o = self._stage_np[i], self._pk_off
        h[o["row0"]: o["row0"] + S] = a.row0
        h[o["nrows"]: o["nrows"] + S] = a.nrows
        h[o["start"]: o["start"] + S] = a.start
        h[o["slot"]: o["slot"] + S] = a.slot
        h[o["look"]: o["look"] + 3 * S] = a.look.reshape(-1)
        h[o["counts"]: o["counts"] + 2] = (S, n)
        bound = "b_ids" in getattr(self, "_bound", set())
        host_ids = a.ids is not None and ids_dev is None and not bound
        if host_ids:
            h[:T] = a.ids[:T]
        lo = 0 if host_ids else o["row0"]
        st = stream or torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(st):
            self._pk[lo:].copy_(self._stage[i][lo:], non_blocking=True)
            if ids_dev is not None and not bound:
                self.b_ids[:T].copy_(ids_dev[:T], non_blocking=True)
        if self._stage_ev[i] is None:
            self._stage_ev[i] = torch.cuda.Event()
        self._stage_ev[i].record(st)

    # -------------------------------------------------------------------------------- pieces
    def run_hash(self, T: int):
        with torch.cuda.device(self.dev):
            self._run_hash(T)

    def _run_hash(self, T: int):
        ext().hash(self.hctx, self.b_ids, self.b_counts, self.b_row0, self.b_nrows, self.b_start, self.b_slot,
                   self.b_look, self.tmap, self.ring, T, self.hash, self.keep, self.err)

    def run_gather(self, T: int, layers=(0, 1), ctas: int | None = None, threads: int | None = None, unroll: int | None = None):
        g = self.gather_cfg
        tabs = [self.tables[LAYERS[l]].rec0 for l in layers]
        lo = sum((self.lo[l * NC : (l + 1) * NC] for l in layers), [])
        hi = sum((self.hi[l * NC : (l + 1) * NC] for l in layers), [])
        ext().gather(self.hash, T, tabs, list(layers), [self.rows[l] for l in layers], lo, hi,
                     g["ctas"] if ctas is None else ctas, g["threads"] if threads is None else threads,
                     g["unroll"] if unroll is None else unroll)

    def run_wkv(self, l: int, T: int, impl: str | None = None):
        x = self.rows[l][:T].view(T, NC * DIM)
        y = self.kv[l][:T]
        impl = impl or ("w8a16" if (self.gemm8 and T <= self.w8_max_rows) else "cublas")
        with torch.cuda.device(self.dev):  # the W8A16 binding launches on the current device
            if impl == "w8a16":
                for r0 in range(0, T, 128):
                    r1 = min(T, r0 + 128)
                    self.gemm8[l](x[r0:r1], y[r0:r1])
            else:
                torch.mm(x, self.wbf[l].t(), out=y)
        return y

    def prepare(self, T: int, stream: torch.cuda.Stream | None = None, impl: str | None = None):
        """Side work of the step (token-id dependent only): hash, both gathers, both wkv. Records one event per layer.
        Fixed sizes, no host sync: CUDA-graph capturable per row bucket."""
        self.kv_idx = [None, None]
        st = stream or torch.cuda.current_stream(self.dev)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(st):
            self._resolve_starts()
            self.run_hash(T)
            for l in range(2):  # layer 1 first: its event (what layer 1's inject waits for) covers half the side work
                self.run_gather(T, (l,))
                self.run_wkv(l, T, impl)
                self.ev[l].record(st)

    def inject(self, l: int, T: int, x_in=None, y=None, rs=None, post=None, comb=None, out=None, gate_out=None,
               wait: bool = True):
        """Layer LAYERS[l]: out = Engram(x) with x = x_in or hc_post(y, rs, post, comb) (fused); out may alias x_in / rs.
        After prepare_prefill the kv rows are the deduplicated ones (kv_idx maps every row to its unique kv row)."""
        if wait:
            torch.cuda.current_stream(self.dev).wait_event(self.ev[l])
        if out is None:
            out = torch.empty(T, HC, D, dtype=torch.bfloat16, device=self.dev)
        ext().inject(T, x_in, y, rs, post, comb, self.kv[l], self.q[l], self.k[l], self.keep, self.eps, self.clamp, out,
                     gate_out, self.kv_idx[l])
        return out

    # -------------------------------------------------------------------------------- prefill (eager, deduplicated)
    def prepare_prefill(self, T: int, stream: torch.cuda.Stream | None = None, gather_cfg=(8, 512, 4)):
        """prepare() for large steps (eager; data-dependent sizes): per layer
          1. row dedup: rows with identical 24 ids have identical table rows and therefore identical kv rows (exact). Key
             = (4-gram head 0, head 1) ids, sorted; adjacent rows merge only if ALL 24 ids are equal (a key collision just
             leaves two kv rows, never a wrong one);
          2. record dedup over the unique rows: every distinct record is read once over PCIe;
          3. wkv on the unique rows only; inject reads kv through kv_idx.
        Captured real prompts, 16K chunks: 47 % unique records, 56 % unique rows."""
        st = stream or torch.cuda.current_stream(self.dev)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.dev))
        info = []
        with torch.cuda.stream(st):
            self._resolve_starts()
            self.run_hash(T)
            for l in range(2):
                ids = self.hash[:T, l]
                key = (ids[:, 16].to(torch.int64) << 32) | ids[:, 17].to(torch.int64)
                key_s, perm = torch.sort(key)
                ids_s = ids[perm]
                same = torch.zeros(T, dtype=torch.bool, device=self.dev)
                if T > 1:
                    same[1:] = (key_s[1:] == key_s[:-1]) & (ids_s[1:] == ids_s[:-1]).all(1)
                gid = torch.cumsum((~same).to(torch.int32), 0, dtype=torch.int32) - 1
                inv_row = torch.empty(T, dtype=torch.int32, device=self.dev)
                inv_row[perm] = gid
                urows = ids_s[~same]  # [U, 24]
                U = urows.shape[0]
                u, inv = torch.unique(urows.reshape(-1).to(torch.int64), return_inverse=True)
                ur = self._scratch(u.numel())
                g = gather_cfg
                ext().gather_ids(self.tables[LAYERS[l]].rec0, u, ur, g[0], g[1], g[2])
                ext().expand(ur, inv.to(torch.int32).view(U, 1, NC).contiguous(), U, 1, 0, self.rows[l])
                self.run_wkv(l, U, "cublas" if U > self.w8_max_rows or not self.gemm8 else None)
                self.kv_idx[l] = inv_row
                self.ev[l].record(st)
                info.append(dict(U=U, records=int(u.numel())))
        return info

    def _scratch(self, n: int) -> torch.Tensor:
        if getattr(self, "_ur", None) is None or self._ur.shape[0] < n:
            self._ur = torch.empty(n, DIM, dtype=torch.bfloat16, device=self.dev)
        return self._ur

    def clear_dedup(self):
        self.kv_idx = [None, None]

    # -------------------------------------------------------------------------------- token-history ring (decode lookback)
    def ring_copy(self, src_slot: int, dst_slot: int, dst: "EngramPhys | None" = None):
        """Sequence migration: the 16 (position, token) entries of a state slot follow the state slot (128 B; run with the
        planned page copies, before the step). dst = the destination rank's module (peer copy) or self."""
        (dst or self).ring[dst_slot].copy_(self.ring[src_slot], non_blocking=True)

    def ring_reset(self, slot: int):
        """A freed / reassigned state slot: its next sequence must bring host lookback ids for its first chunk anyway
        (prefill), so this only makes stale entries impossible to match (positions -1)."""
        self.ring[slot].fill_(-1)

    def check_errors(self) -> list:
        """[ring misses, token ids outside the map, ring reads of sequences with > 8 rows]: all must stay 0 (host sync)."""
        return self.err.tolist()
