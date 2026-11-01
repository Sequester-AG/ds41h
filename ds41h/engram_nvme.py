"""ENGRAM_NVME: the two Engram n-gram tables (layers 1 and 14) stay on the NVMe drive; each step reads the rows it needs
through io_uring instead of holding the tables in GPU HBM (or host RAM).

Ported from BRRR's NVMe Engram reader (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
LICENSES/BRRR-Apache-2.0.txt): src/brrr/engram_nvme.py (issue / wait kernels with host-mapped generation flags =
CUDA-graph-safe GPU -> host signalling, row dequant kernel) and src/brrr/kernels/engram_nvme.cpp (io_uring poller
thread, O_DIRECT / buffered descriptors, pinned staging slots). BRRR's b12x Engram hashing is NOT used: the row ids come
from stock vLLM 0.32.0's own `_hash_ids_kernel` (unchanged), so they are bit-identical to stock by construction.
Changes vs BRRR: raw io_uring syscalls (no liburing dependency), one poller thread per process serving both Engram
layers in layer order (layer 1 never queues behind layer 14), a re-laid-out table file with weight + scale in ONE
272-B record (one read per row instead of a read + 6 GB of resident scales), per-request de-duplication of row ids,
stock vLLM's head-shard ownership rule and bit-exact stock dequant (pinned PTX, shared with kernels/engram_host.cu),
multi-CTA publish, GPU-side timeout + stall statistics.

Why: stock (`--engram-config cpu_offload=false`) row-shards the two 101-GB tables over the TP4 GPUs = ~47 GiB of every
GPU's 124 GiB of weights (KV cache ~2.6 GiB = ~98K tokens); per token a layer reads only 24 rows (264 B each) whose
ids depend only on the token ids. Reading them from NVMe frees the 47 GiB for KV (~1.8M tokens); with enough host RAM,
DS41H_ENGRAM_HOST=1 (pinned host RAM) is the better home.

Data path per step and Engram layer (ds41h/patch_engram.py wires it into stock Engram):
  prepare_embeddings (before layer 0, right after stock's hash kernel)
      -> issue kernel: this rank's 6 hash columns of every token -> pinned `ids`, then (fence) the request generation
         -> pinned `req_gen` (st.release.sys). Generation counter in device memory => replayable in CUDA graphs.
  poller thread (C++, kernels/engram_nvme.cu): sees the new generation, de-duplicates the ids (stock ownership rule:
      ids outside this rank's rows -> zero rows), reads every unique record with io_uring (O_DIRECT: the 1-2 sectors
      holding the record into a staging slot; buffered: the 264 B straight into the pinned rows buffer), then
      publishes `done = gen` (mfence + release store).
  Engram.embed -> _ready_rows (right before layer 1 / 14 consume the rows)
      -> wait + dequant kernel: thread 0 of every CTA spins on `done` (ld.acquire.sys), then rows -> bf16 exactly like
         stock `_engram_lookup_kernel` into the stock staging buffer; the stock TP all-gather follows unchanged.
  The poller serves layer 1 before layer 14, so layer 14's reads overlap layers 1..13.

Table file (one per Engram layer, built once from the read-only checkpoint, `python -m ds41h.engram_nvme build`):
  DS41H_ENGRAM_NVME_DIR (default ~/.ds41h/engram_nvme)/
  ds41h_engram_nvme_L<layer>.bin: 4 KiB header + 4 KiB blocks of 15 records x 272 B (256 B e4m3 + 8 B ue8m0 + 8 B 0,
  16 B pad per block): a record never straddles a 4 KiB page -> one read per row; O_DIRECT reads are 512 B or 1024 B.
  104.9 GB per layer (checkpoint: 101.4). The header carries a path-independent checkpoint key (tensor offsets, size,
  sha1 of samples); a stale / partial file is rebuilt (under an flock) by the first server rank that opens it.

Environment (all opt-in; default path unchanged):
  DS41H_ENGRAM_NVME=1                 enable (ds41h/patch_engram.py); mutually exclusive with DS41H_ENGRAM_HOST
  DS41H_ENGRAM_NVME_DIR               table directory (default ~/.ds41h/engram_nvme)
  DS41H_ENGRAM_NVME_IO=direct|buffered|auto   direct (default): O_DIRECT reads (2x the cold read rate of buffered
                                      reads, measured); buffered: page cache; auto: O_DIRECT for requests of <=
                                      _AUTO_MAX missing rows, buffered above (BRRR's rule)
  DS41H_ENGRAM_NVME_AUTO_MAX=512      see above
  DS41H_ENGRAM_NVME_CACHE_MB=1024     record cache per process (all Engram layers): recently read 264-B records, exact
                                      row-id keys, 2-way set-associative; hits cost a memcpy instead of a read (repeated
                                      n-grams, prompt n-grams reappearing in decode). 0 = off
  DS41H_ENGRAM_NVME_DEPTH=512         io_uring depth per process (= reads in flight per rank)
  DS41H_ENGRAM_NVME_SPIN_MS=50        poller busy-spins this long after the last request, then polls every _IDLE_US
  DS41H_ENGRAM_NVME_IDLE_US=200
  DS41H_ENGRAM_NVME_TIMEOUT_S=300     the GPU traps (instead of hanging) if rows do not arrive within this time
  DS41H_ENGRAM_NVME_STATS_S=30        poller prints per-layer I/O statistics to stderr every N s while active (0: off)
  DS41H_ENGRAM_NVME_BUILD=1           build a missing / stale table file at server start (default 1; 0 = refuse)
  DS41H_ENGRAM_NVME_PREFETCH=0|1      prefill lookahead (default 0): when a request is admitted, its prompt tokens
                                      beyond this step's chunk are hashed on the CPU (stock hash, C++ port) and their
                                      records read into the record cache below every GPU request in priority, so later
                                      chunks / the next queued request find them cached. Numerics unaffected (the served
                                      lookups always use the GPU kernel's ids; a wrong prefetch only costs a read).
  DS41H_ENGRAM_NVME_PREFETCH_TOKENS=32768   lookahead horizon per request (tokens beyond the current chunk)
  DS41H_ENGRAM_NVME_PREFETCH_INFLIGHT=128   prefetch reads in flight per process (GPU requests keep the rest)
CLI: python -m ds41h.engram_nvme build|verify|status [--model DIR] [--layers 1,14]
Tests / CPU-only I/O benchmark: tests/test_engram_nvme.py [--io-bench ...] (the I/O engine part needs no GPU).
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import logging
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger("ds41h.engram_nvme")

MAGIC = b"DS41HENV"
VERSION = 1
DIM = 256  # engram_head_dim (e4m3 bytes per row)
SDIM = 8  # ue8m0 scale bytes per row (block 32)
PAYLOAD = DIM + SDIM  # 264 bytes read per row
RECORD = 272  # record stride in the file and in the pinned rows buffer (16-B aligned)
PER_BLOCK = 15  # records per 4 KiB block (15 x 272 = 4080 + 16 B pad): a record never straddles a page
BLOCK = 4096
DATA_OFF = 4096  # header block
SECTOR = 512
_HDR = struct.Struct("<8sIIIIQQQII")  # magic, version, record, per_block, block, rows, data_off, size, layer, key_len
_OFF_KEY = 64
_OFF_DONE = 3072  # u32 _DONE + f64 build seconds + f64 unix time (written last, after fsync of the data)
_DONE = 0x5A5A5A5A
MODES = {"direct": 0, "buffered": 1, "auto": 2}


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def table_dir() -> Path:
    return Path(os.environ.get("DS41H_ENGRAM_NVME_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "engram_nvme")))


def table_path(layer: int) -> Path:
    return table_dir() / f"ds41h_engram_nvme_L{layer}.bin"


def n_blocks(rows: int) -> int:
    return -(-rows // PER_BLOCK)


def file_size(rows: int) -> int:
    return DATA_OFF + n_blocks(rows) * BLOCK


def record_offset(r: int) -> int:
    return DATA_OFF + (r // PER_BLOCK) * BLOCK + (r % PER_BLOCK) * RECORD


@cache
def ext():
    from ds41h import cuext

    return cuext.load("ds41h_engram_nvme", "engram_nvme.cu")


# --------------------------------------------------------------------------------------------- checkpoint source
@dataclass(frozen=True)
class Source:
    """Where layer `layer`'s Engram table lives in the (read-only) safetensors checkpoint."""

    file: str
    layer: int
    rows: int
    w_off: int  # absolute byte offset of the e4m3 [rows, 256] data
    s_off: int  # absolute byte offset of the ue8m0 [rows, 8] data
    size: int


def tensor_source(model_dir: str | os.PathLike, layer: int) -> Source:
    model_dir = Path(model_dir)
    wmap = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wname, sname = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
    if wmap[wname] != wmap[sname]:
        raise ValueError(f"Engram L{layer}: weight and scale in different files ({wmap[wname]}, {wmap[sname]})")
    path = model_dir / wmap[wname]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    w, s = hdr[wname], hdr[sname]
    if w["dtype"] != "F8_E4M3" or s["dtype"] != "F8_E8M0" or w["shape"][1] != DIM or s["shape"][1] != SDIM:
        raise ValueError(f"Engram L{layer}: unexpected table format {w['dtype']} {w['shape']} / {s['dtype']} {s['shape']}")
    rows = int(w["shape"][0])
    if int(s["shape"][0]) != rows:
        raise ValueError(f"Engram L{layer}: weight / scale row counts differ")
    if w["data_offsets"][1] - w["data_offsets"][0] != rows * DIM or s["data_offsets"][1] - s["data_offsets"][0] != rows * SDIM:
        raise ValueError(f"Engram L{layer}: tensor byte sizes do not match the shapes")
    return Source(str(path.resolve()), layer, rows, base + int(w["data_offsets"][0]), base + int(s["data_offsets"][0]),
                  path.stat().st_size)


def source_key(src: Source) -> str:
    """Path-independent identity of the table contents (hardlinked / copied checkpoints give the same key)."""
    h = hashlib.sha1()
    with open(src.file, "rb") as f:
        for off in (src.w_off, src.w_off + (src.rows // 2) * DIM, src.w_off + (src.rows - 256) * DIM, src.s_off,
                    src.s_off + (src.rows // 2) * SDIM, src.s_off + (src.rows - 8192) * SDIM):
            f.seek(off)
            h.update(f.read(65536))
    return (f"ds41h-engram-nvme v{VERSION} L{src.layer} rows={src.rows} w={src.w_off} s={src.s_off} size={src.size} "
            f"record={RECORD}x{PER_BLOCK}/{BLOCK} sample={h.hexdigest()[:20]}")


# --------------------------------------------------------------------------------------------- table file
def read_header(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            raw = f.read(DATA_OFF)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return None
    if len(raw) < DATA_OFF:
        return None
    magic, ver, rec, per, blk, rows, doff, fsize, layer, klen = _HDR.unpack_from(raw, 0)
    if magic != MAGIC:
        return None
    key = raw[_OFF_KEY : _OFF_KEY + klen].decode("utf-8", "replace")
    done, secs, when = struct.unpack_from("<Idd", raw, _OFF_DONE)
    return dict(version=ver, record=rec, per_block=per, block=blk, rows=rows, data_off=doff, size=fsize, layer=layer,
                key=key, complete=done == _DONE, build_s=secs, built=when, actual_size=size)


def header_ok(path: Path, key: str, rows: int) -> bool:
    h = read_header(path)
    return bool(h and h["complete"] and h["version"] == VERSION and h["record"] == RECORD and h["per_block"] == PER_BLOCK
                and h["block"] == BLOCK and h["rows"] == rows and h["data_off"] == DATA_OFF
                and h["size"] == file_size(rows) == h["actual_size"] and h["key"] == key)


def _aligned(n: int) -> mmap.mmap:
    return mmap.mmap(-1, max(n, BLOCK))  # anonymous mappings are page aligned (O_DIRECT) and zero-filled


def _pread_direct(fd: int, off: int, n: int):
    """n bytes at `off` through O_DIRECT (aligned superset read). Returns (numpy uint8 view, owner mmap)."""
    import numpy as np

    a0 = off & ~(BLOCK - 1)
    a1 = (off + n + BLOCK - 1) & ~(BLOCK - 1)
    buf = _aligned(a1 - a0)
    got, pos = 0, 0
    while got < off + n - a0:
        view = memoryview(buf)[pos:]
        r = os.preadv(fd, [view], a0 + pos)
        if r <= 0:
            raise OSError(f"short O_DIRECT read at {a0 + pos} ({r})")
        got += r
        pos += r
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr[off - a0 : off - a0 + n], buf


def build(src: Source, path: Path, key: str, workers: int = 4, chunk_blocks: int = 1 << 16,
          progress: bool = True) -> float:
    """Write the re-laid-out table for `src` to `path` (via `path`.tmp + rename). O_DIRECT reads and writes keep the
    page cache (shared with running servers) clean. Returns seconds."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = src.rows
    nb = n_blocks(rows)
    size = file_size(rows)
    st = os.statvfs(path.parent)
    free = st.f_bavail * st.f_frsize
    old = path.stat().st_size if path.exists() else 0
    if free + old < size + (20 << 30):
        raise RuntimeError(f"ds41h engram_nvme: {path.parent} has {free / 1e9:.0f} GB free, the L{src.layer} table needs "
                           f"{size / 1e9:.1f} GB (+20 GB margin)")
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.perf_counter()
    fo = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    fi = os.open(src.file, os.O_RDONLY | os.O_DIRECT)
    try:
        os.posix_fallocate(fo, 0, size)
        chunks = [(b0, min(nb, b0 + chunk_blocks)) for b0 in range(0, nb, chunk_blocks)]
        done = [0]
        lock = threading.Lock()

        def work(c):
            b0, b1 = c
            r0, r1 = b0 * PER_BLOCK, min(rows, b1 * PER_BLOCK)
            n = r1 - r0
            w, wo = _pread_direct(fi, src.w_off + r0 * DIM, n * DIM)
            s, so = _pread_direct(fi, src.s_off + r0 * SDIM, n * SDIM)
            ob = _aligned((b1 - b0) * BLOCK)
            out = np.frombuffer(ob, dtype=np.uint8).reshape(b1 - b0, BLOCK)
            full = n // PER_BLOCK
            if full:
                rec = out[:full, : PER_BLOCK * RECORD].reshape(full, PER_BLOCK, RECORD)
                rec[:, :, :DIM] = w[: full * PER_BLOCK * DIM].reshape(full, PER_BLOCK, DIM)
                rec[:, :, DIM:PAYLOAD] = s[: full * PER_BLOCK * SDIM].reshape(full, PER_BLOCK, SDIM)
            rem = n - full * PER_BLOCK
            if rem:
                rec = out[full, : rem * RECORD].reshape(rem, RECORD)
                rec[:, :DIM] = w[full * PER_BLOCK * DIM :].reshape(rem, DIM)
                rec[:, DIM:PAYLOAD] = s[full * PER_BLOCK * SDIM :].reshape(rem, SDIM)
            del w, s, rec, out
            wo.close()
            so.close()
            off, pos, total = DATA_OFF + b0 * BLOCK, 0, (b1 - b0) * BLOCK
            while pos < total:
                r = os.pwritev(fo, [memoryview(ob)[pos:]], off + pos)
                if r <= 0:
                    raise OSError(f"short O_DIRECT write at {off + pos}")
                pos += r
            ob.close()
            with lock:
                done[0] += 1
                if progress and (done[0] % 16 == 0 or done[0] == len(chunks)):
                    el = time.perf_counter() - t0
                    gb = done[0] * chunk_blocks * BLOCK / 1e9
                    print(f"  L{src.layer}: {done[0]}/{len(chunks)} chunks, {gb:.1f} GB written, {el:.0f} s, "
                          f"{gb / max(el, 1e-9):.2f} GB/s", flush=True)

        with ThreadPoolExecutor(workers) as ex:
            for _ in ex.map(work, chunks):
                pass
        os.fsync(fo)
        secs = time.perf_counter() - t0
        hb = _aligned(DATA_OFF)
        kb = key.encode()[:2048]
        _HDR.pack_into(hb, 0, MAGIC, VERSION, RECORD, PER_BLOCK, BLOCK, rows, DATA_OFF, size, src.layer, len(kb))
        hb[_OFF_KEY : _OFF_KEY + len(kb)] = kb
        struct.pack_into("<Idd", hb, _OFF_DONE, _DONE, secs, time.time())
        if os.pwritev(fo, [hb], 0) != DATA_OFF:
            raise OSError("header write failed")
        os.fsync(fo)
        hb.close()
    except BaseException:
        os.close(fo)
        os.close(fi)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fo)
    os.close(fi)
    os.replace(tmp, path)
    return time.perf_counter() - t0


def verify(src: Source, path: Path, samples: int = 200_000, seed: int = 0) -> tuple[int, int]:
    """Compare `samples` random records (+ the first / last rows) with the checkpoint bytes. Returns (checked, bad)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = np.unique(np.concatenate([rng.integers(0, src.rows, samples), [0, 1, 14, 15, src.rows - 1]]))
    bad = 0
    with open(src.file, "rb", buffering=0) as fc, open(path, "rb", buffering=0) as ft:
        for r in rows.tolist():
            w = os.pread(fc.fileno(), DIM, src.w_off + r * DIM)
            s = os.pread(fc.fileno(), SDIM, src.s_off + r * SDIM)
            t = os.pread(ft.fileno(), RECORD, record_offset(r))
            if t[:DIM] != w or t[DIM:PAYLOAD] != s or any(t[PAYLOAD:]):
                bad += 1
                if bad <= 5:
                    logger.error("ds41h engram_nvme verify: L%d row %d differs", src.layer, r)
    return len(rows), bad


@dataclass(frozen=True)
class Table:
    path: str
    layer: int
    rows: int
    key: str


def open_table(model_dir: str | os.PathLike, layer: int, rows: int | None = None, build_missing: bool | None = None) -> Table:
    """Validated table file for `layer` of the checkpoint in `model_dir`; builds it (once, under an flock shared by all
    ranks / servers) when missing or stale and DS41H_ENGRAM_NVME_BUILD != 0."""
    src = tensor_source(model_dir, layer)
    if rows is not None and rows != src.rows:
        raise ValueError(f"ds41h engram_nvme: L{layer} model wants {rows} rows, checkpoint table has {src.rows}")
    key = source_key(src)
    path = table_path(layer)
    if not header_ok(path, key, src.rows):
        if build_missing is None:
            build_missing = env_flag("DS41H_ENGRAM_NVME_BUILD", "1")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not header_ok(path, key, src.rows):  # another rank / server may have built it meanwhile
                if not build_missing:
                    raise RuntimeError(f"ds41h engram_nvme: {path} missing or stale (key {key}); build it with "
                                       f"`python -m ds41h.engram_nvme build --model {model_dir}`")
                logger.warning("ds41h engram_nvme: building %s from %s (%.1f GB, one-time)", path, src.file,
                               file_size(src.rows) / 1e9)
                secs = build(src, path, key, progress=False)
                logger.warning("ds41h engram_nvme: built %s in %.0f s", path, secs)
            fcntl.flock(lk, fcntl.LOCK_UN)
    try:
        if os.stat(path).st_dev != os.stat(src.file).st_dev:
            logger.warning("ds41h engram_nvme: %s is not on the checkpoint's filesystem (keep it on the fast NVMe)", path)
    except OSError:
        pass
    return Table(str(path), layer, src.rows, key)


# --------------------------------------------------------------------------------------------- I/O engine (process)
@dataclass
class Channel:
    """One Engram layer of this process: the C++ channel id + its device-side state (generation counter, stats)."""

    handle: int
    table: Table
    vstart: int
    vend: int
    max_rows: int
    dstate: object = None  # torch int64 [16] CUDA tensor (counter, gen, arrive, stats)


class Reader:
    """Process-wide io_uring reader (one poller thread for all Engram layers of this process, served in layer order)."""

    def __init__(self, rank: int = 0, cpu_only: bool = False) -> None:
        self.rank = rank
        self.cpu_only = cpu_only
        self.mode = os.environ.get("DS41H_ENGRAM_NVME_IO", "direct")
        if self.mode not in MODES:
            raise ValueError(f"DS41H_ENGRAM_NVME_IO={self.mode!r}: expected one of {sorted(MODES)}")
        self.depth = int(os.environ.get("DS41H_ENGRAM_NVME_DEPTH", "512"))
        self.auto_max = int(os.environ.get("DS41H_ENGRAM_NVME_AUTO_MAX", "512"))
        self.spin_ms = float(os.environ.get("DS41H_ENGRAM_NVME_SPIN_MS", "50"))
        self.idle_us = float(os.environ.get("DS41H_ENGRAM_NVME_IDLE_US", "200"))
        self.stats_s = float(os.environ.get("DS41H_ENGRAM_NVME_STATS_S", "30"))
        self.timeout_ms = int(float(os.environ.get("DS41H_ENGRAM_NVME_TIMEOUT_S", "300")) * 1000)
        self.cache_mb = float(os.environ.get("DS41H_ENGRAM_NVME_CACHE_MB", "1024"))
        self.handle = ext().engine_new(self.depth, MODES[self.mode], self.auto_max, int(self.spin_ms * 1e6),
                                       int(self.idle_us * 1e3), int(self.stats_s * 1e9), rank, int(self.cache_mb * 2**20))
        self.channels: list[Channel] = []
        self.expected_channels = 2  # Engram layers per process (the record cache budget is split between them)
        self.prefetch_on = env_flag("DS41H_ENGRAM_NVME_PREFETCH", "0")
        self.pf_tokens = int(os.environ.get("DS41H_ENGRAM_NVME_PREFETCH_TOKENS", "32768"))
        self.pf_inflight = int(os.environ.get("DS41H_ENGRAM_NVME_PREFETCH_INFLIGHT", "128"))
        self.hash_set = False
        self.started = False
        self._lock = threading.Lock()

    def add_channel(self, table: Table, vstart: int, vend: int, max_rows: int, layer_index: int = -1, head0: int = 0,
                    part: int = 0) -> Channel:
        """layer_index / head0 / part: this channel's hash columns (prefetch only)."""
        with self._lock:
            if self.started:
                raise RuntimeError("ds41h engram_nvme: channels must be added before the reader starts")
            per_channel = int(self.cache_mb * 2**20 / max(1, self.expected_channels))
            h = ext().engine_add_channel(self.handle, table.layer, table.path, DATA_OFF, table.rows, PER_BLOCK, BLOCK,
                                         RECORD, vstart, vend, max_rows, not self.cpu_only, per_channel)
            ext().channel_set_cols(h, layer_index, head0, part)
            ch = Channel(h, table, vstart, vend, max_rows)
            self.channels.append(ch)
            return ch

    # ------------------------------------------------------------------------------------------- prefetch (opt-in)
    def set_hash(self, hp: HashParams) -> None:
        import torch

        ext().engine_set_hash(self.handle, torch.as_tensor(hp.token_map, dtype=torch.int32), int(hp.pad_id),
                              torch.as_tensor(hp.multipliers, dtype=torch.int64), torch.as_tensor(hp.primes, dtype=torch.int64),
                              torch.as_tensor(hp.offsets, dtype=torch.int64), self.pf_inflight, 1024)
        self.hash_set = True

    def prefetch(self, tokens, lo: int, hi: int) -> None:
        """Queue rows of prompt positions [lo, hi) of a request whose prompt token ids are `tokens` (list / array)."""
        import torch

        a = max(0, lo - 3)  # n-gram history (max_ngram - 1 = 3 tokens)
        ext().engine_prefetch(self.handle, torch.as_tensor(tokens[a:hi], dtype=torch.int32), lo - a)

    def pf_stats(self) -> dict:
        v = list(ext().engine_pf_stats(self.handle))
        d = dict(jobs=v[0], tokens=v[1], queued=v[2], inflight=v[3], errors=v[4], channels=[])
        for i in range(len(self.channels)):
            ids, reads, late = v[5 + 3 * i : 8 + 3 * i]
            d["channels"].append(dict(layer=self.channels[i].table.layer, ids=ids, reads=reads, late=late))
        return d

    def start(self) -> str:
        """Set up io_uring and start the poller thread (host only, no CUDA work). Returns the engine's config note."""
        if self.started:
            return ""
        with self._lock:
            if self.started:
                return ""
            note = ext().engine_start(self.handle)
            self.started = True
            atexit.register(self.stop)
            return note

    def stop(self) -> None:
        if self.started:
            try:
                ext().engine_stop(self.handle)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass
            self.started = False

    def stats(self) -> list[dict]:
        return [stats_dict(ext().channel_stats(c.handle)) | {"layer": c.table.layer} for c in self.channels]


STAT_FIELDS = ("requests", "ids", "owned", "unique", "ios", "direct_ios", "bytes", "svc_ns", "svc_max_ns", "io_ns",
               "gpu_waits", "gpu_spun", "gpu_stall_ns", "gpu_stall_max_ns", "cache_hits", "cache_records",
               "dec_waits", "dec_spun", "dec_stall_ns", "dec_stall_max_ns")


def stats_dict(raw) -> dict:
    vals, hist = raw
    d = dict(zip(STAT_FIELDS, vals))
    d["svc_hist"] = list(hist)
    return d


def hist_quantile(hist: list[int], q: float) -> float:
    """Quantile (us) from the engine's log histogram (4 buckets per octave of microseconds; see bucket_us)."""
    total = sum(hist)
    if not total:
        return 0.0
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= q * total:
            return bucket_us(i)
    return bucket_us(len(hist) - 1)


def bucket_us(i: int) -> float:
    """Upper edge (us) of histogram bucket i: bucket 4e + m holds [2^e (1 + m/4), 2^e (1 + (m+1)/4))."""
    e, m = divmod(i, 4)
    return float(2**e) * (1 + (m + 1) / 4)


_READER: Reader | None = None
_READER_LOCK = threading.Lock()


def reader(rank: int = 0) -> Reader:
    global _READER
    with _READER_LOCK:
        if _READER is None:
            _READER = Reader(rank)
        return _READER


# --------------------------------------------------------------------------------------------- CPU n-gram hash
@dataclass
class HashParams:
    """Inputs of stock `_hash_ids_kernel` (vllm/models/deepseek_v41/common/engram.py:288-420) on the CPU."""

    token_map: object  # numpy int64 [vocab]: token id -> compressed id
    pad_id: int
    multipliers: object  # numpy int64 [n_layers, max_ngram]
    primes: object  # numpy int64 [n_layers, max_ngram - 1, n_heads]
    offsets: object  # numpy int64 [n_layers, n_cols]
    layer_ids: tuple

    @staticmethod
    def from_model(model_dir: str | os.PathLike) -> HashParams:
        import types

        import numpy as np
        from transformers import AutoTokenizer
        from vllm.models.deepseek_v41.common import engram as eg

        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = cfg.get("text_config", cfg) if "engram_layer_ids" not in cfg else cfg
        layout = eg.EngramLayout(types.SimpleNamespace(**cfg))
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        tmap, vsize = eg.build_compressed_token_map(tok)
        if vsize != layout.compressed_vocab_size:
            raise ValueError(f"compressed vocab {vsize} != config {layout.compressed_vocab_size}")
        mult = eg.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vsize).numpy()
        return HashParams(np.asarray(tmap, dtype=np.int64), int(tmap[layout.pad_token_id]), mult.astype(np.int64),
                          np.asarray(layout.primes, dtype=np.int64), layout.offsets.numpy().astype(np.int64),
                          tuple(layout.layer_ids))


def cpu_hash_ids(tokens, p: HashParams, history=()):
    """Row ids [T, n_layers, n_cols] (int64) of `tokens` = stock `_hash_ids_kernel` for one request whose `history`
    (the tokens before this chunk, any length; only the last max_ngram - 1 matter) is known: n-grams that reach before
    the request's first token are cut with the pad id exactly like the kernel's `blocked` rule. Text only (no image
    DEAD ids). Used for row prefetch / realistic I/O benchmarks -- the served lookups always use the GPU kernel's ids."""
    import numpy as np

    n_layers, max_ngram = p.multipliers.shape
    n_heads = p.primes.shape[-1]
    hist = np.asarray(list(history)[-(max_ngram - 1) :], dtype=np.int64)
    seq = np.concatenate([hist, np.asarray(tokens, dtype=np.int64)])
    h0 = len(hist)
    T = len(seq) - h0
    tm = p.token_map[seq]
    out = np.empty((T, n_layers, (max_ngram - 1) * n_heads), dtype=np.int64)
    idx = np.arange(h0, h0 + T)
    for li in range(n_layers):
        rolling = np.zeros(T, dtype=np.int64)
        blocked = np.zeros(T, dtype=bool)
        for sh in range(max_ngram):
            j = idx - sh
            ok = j >= 0
            blocked |= ~ok
            value = np.where(blocked, p.pad_id, tm[np.maximum(j, 0)])
            rolling ^= value * p.multipliers[li, sh]
            if sh > 0:
                for h in range(n_heads):
                    col = (sh - 1) * n_heads + h
                    out[:, li, col] = rolling % p.primes[li, sh - 1, h] + p.offsets[li, col]
    return out


# --------------------------------------------------------------------------------------------- CLI
def _default_model() -> str:
    return os.environ.get("MODEL", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))


def cli_build(model: str, layers: list[int], workers: int, force: bool) -> int:
    rc = 0
    for layer in layers:
        src = tensor_source(model, layer)
        key = source_key(src)
        path = table_path(layer)
        if not force and header_ok(path, key, src.rows):
            print(f"L{layer}: {path} up to date ({file_size(src.rows) / 1e9:.1f} GB)")
            continue
        print(f"L{layer}: building {path} from {src.file} ({src.rows} rows, {file_size(src.rows) / 1e9:.1f} GB)", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            secs = build(src, path, key, workers=workers)
            fcntl.flock(lk, fcntl.LOCK_UN)
        n, bad = verify(src, path, samples=100_000)
        print(f"L{layer}: built in {secs:.0f} s ({file_size(src.rows) / 1e9 / secs:.2f} GB/s); verify {n} rows: "
              f"{'OK' if not bad else f'{bad} BAD'}", flush=True)
        rc |= bool(bad)
    return rc


def cli_status(model: str, layers: list[int]) -> int:
    for layer in layers:
        path = table_path(layer)
        h = read_header(path)
        ok = False
        try:
            src = tensor_source(model, layer)
            ok = header_ok(path, source_key(src), src.rows)
        except Exception as e:  # noqa: BLE001
            print(f"L{layer}: checkpoint source error: {e}")
        if h is None:
            print(f"L{layer}: {path}: missing")
        else:
            print(f"L{layer}: {path}: {h['actual_size'] / 1e9:.1f} GB, complete={h['complete']}, valid for {model}: {ok}, "
                  f"built {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(h['built']))} UTC in {h['build_s']:.0f} s\n"
                  f"    key={h['key']}")
    return 0


def cli_verify(model: str, layers: list[int], samples: int) -> int:
    rc = 0
    for layer in layers:
        src = tensor_source(model, layer)
        n, bad = verify(src, table_path(layer), samples=samples, seed=int(time.time()))
        print(f"L{layer}: verified {n} random rows against {src.file}: {'OK' if not bad else f'{bad} BAD'}")
        rc |= bool(bad)
    return rc


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m ds41h.engram_nvme")
    ap.add_argument("cmd", choices=["build", "status", "verify"])
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--layers", default="1,14")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", type=int, default=200_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    layers = [int(x) for x in args.layers.split(",") if x]
    if args.cmd == "build":
        return cli_build(args.model, layers, args.workers, args.force)
    if args.cmd == "status":
        return cli_status(args.model, layers)
    return cli_verify(args.model, layers, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
