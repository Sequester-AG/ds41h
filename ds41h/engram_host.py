"""Engram host offload: ONE shared pinned host copy of the Engram n-gram tables + a zero-copy CUDA gather.

Why: the two Engram tables (layers 1 and 14; e4m3 [~384M, 256] + ue8m0 [~384M, 8] each, 101.4 GB per layer, 203 GB in
total) are row-sharded over the 4 GPUs by stock vLLM (`--engram-config cpu_offload=false`): ~47 GiB of every GPU's
124 GiB of weights, leaving ~2.4 GiB of KV cache (~91K tokens). Per token a layer reads only 24 rows (264 B each), and
the row ids depend only on token ids, so the tables can live in host RAM and be read over PCIe (Gen5 x16 per GPU).

Design (docs/ENGRAM.md has the requirements):
* One file per Engram layer in DS41H_ENGRAM_HOST_DIR (default /dev/shm; a hugetlbfs or `huge=` tmpfs mount works the
  same and gives 2 MiB / 1 GiB pages), mapped MAP_SHARED by all 4 TP worker processes and registered with
  cudaHostRegister(Mapped|Portable) in each of them -> one physical copy, zero-copy device access through UVA.
  Layout: 4 KiB header (magic, geometry, checkpoint key, per-shard load states) + 2 MiB-aligned data:
    split          : e4m3 rows [N, 256] then ue8m0 rows [N, 8] (the checkpoint / stock layout);
    packed (default): [N, 272] records = 256 B e4m3 + 8 B ue8m0 + 8 B pad (one page / one TLB entry per row).
* Loading: each TP rank's stock weight loader (`_engram_head_shard_weight_loader`: narrow to the rank's head-shard rows
  + copy_) writes ITS rows straight into the shared mapping (4 ranks load in parallel, each ~25 GB per layer); the
  shard state bytes in the header mark (rank, weight|scale) complete; the first lookup waits until all shards are
  complete. With DS41H_ENGRAM_HOST_KEEP=1 the file outlives the server and a restart with the same checkpoint key skips
  the copy entirely (the loader becomes a no-op for complete shards).
* Lookup: every rank gathers ALL 24 hash columns of its tokens itself (kernels/engram_host.cu), so the stock TP
  all-gather of the looked-up rows disappears. Numerics are bit-identical to stock `_engram_lookup_kernel` + all-gather:
  bf16_rn(fp32(e4m3) * 2^(scale-127)) via pinned PTX (no ftz), stock's per-head shard-ownership rule (ids outside the
  owning rank's row range -> zeros).
Environment:
  DS41H_ENGRAM_HOST=1             enable (ds41h/patch_engram.py; opt-in, default 0)
  DS41H_ENGRAM_HOST_DIR           table directory (default /dev/shm)
  DS41H_ENGRAM_HOST_LAYOUT        packed | split (default packed: +15% random-row bandwidth, +3% RAM)
  DS41H_ENGRAM_HOST_KEEP=0|1      keep the table file after the last user exits (default 0: unlink when unused)
  DS41H_ENGRAM_HOST_CTAS / _THREADS / _UNROLL   large-batch gather grid (defaults 16 / 1024 / 4)
  DS41H_ENGRAM_HOST_POPULATE_THREADS            parallel prefault threads at creation (default 8)
CLI: python -m ds41h.engram_host status | drop [--all]
Tests + microbenchmark: tests/test_engram_host.py [--bench] (one GPU).
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import hashlib
import json
import logging
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

logger = logging.getLogger("ds41h.engram_host")

MAGIC = b"DS41HENG"
VERSION = 1
HEADER = 4096
ALIGN = 2 << 20  # data regions start on 2 MiB boundaries (huge-page friendly)
DIM = 256
QUANT = 32
SDIM = DIM // QUANT
LAYOUTS = {"split": 0, "packed": 1}
PACKED_STRIDE = 272  # 256 e4m3 + 8 ue8m0 + 8 pad (16-B aligned records)
MAX_SHARDS = 64
# header offsets
_OFF_GEOM = 8  # <IIQIIQQQQQ: version, layout, rows, dim, sdim, w_off, s_off, w_stride, s_stride, total
_GEOM = struct.Struct("<IIQIIQQQQQ")
_OFF_READY = 120  # u32 0xA5A5A5A5 once the header is complete
_OFF_KEY = 128  # u32 length + utf8 key (<= 880 B)
_OFF_SHARDS = 1024  # u32 n_shards, then 2 state bytes (weight, scale) per shard
_READY = 0xA5A5A5A5
EMPTY, LOADING, DONE = 0, 1, 2
MADV_POPULATE_WRITE = 23


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def host_dir() -> Path:
    return Path(os.environ.get("DS41H_ENGRAM_HOST_DIR", "/dev/shm"))


def default_layout() -> str:
    lay = os.environ.get("DS41H_ENGRAM_HOST_LAYOUT", "packed")
    if lay not in LAYOUTS:
        raise ValueError(f"DS41H_ENGRAM_HOST_LAYOUT={lay!r}: expected one of {sorted(LAYOUTS)}")
    return lay


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@cache
def ext():
    from ds41h import cuext

    return cuext.load("ds41h_engram_host", "engram_host.cu")


@cache
def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    return libc


@dataclass(frozen=True)
class Geometry:
    rows: int
    layout: str
    w_off: int
    s_off: int
    w_stride: int
    s_stride: int
    total: int

    @staticmethod
    def make(rows: int, layout: str, page: int = ALIGN) -> Geometry:
        page = max(page, ALIGN)
        if layout == "split":
            w_off = ALIGN
            s_off = _align(w_off + rows * DIM, ALIGN)
            total = _align(s_off + rows * SDIM, page)
            return Geometry(rows, layout, w_off, s_off, DIM, SDIM, total)
        if layout == "packed":
            w_off = ALIGN
            total = _align(w_off + rows * PACKED_STRIDE, page)
            return Geometry(rows, layout, w_off, w_off + DIM, PACKED_STRIDE, PACKED_STRIDE, total)
        raise ValueError(layout)


def meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


class HostTable:
    """One Engram layer table in a shared, CUDA-registered host mapping (see module docstring)."""

    def __init__(self, path: Path, key: str, rows: int, n_shards: int, layout: str | None = None,
                 keep: bool | None = None) -> None:
        self.path = Path(path)
        self.key = key
        self.n_shards = n_shards
        self.keep = env_flag("DS41H_ENGRAM_HOST_KEEP") if keep is None else keep
        layout = layout or default_layout()
        if n_shards > MAX_SHARDS:
            raise ValueError("too many shards")
        page = self._page_size(self.path.parent)
        self.geom = Geometry.make(rows, layout, page)
        self.created = False
        self.mm: mmap.mmap | None = None
        self.buf: torch.Tensor | None = None
        self.base = 0
        self.dev_base = 0
        self._registered = False
        self._ready = False
        self._use_fd = -1
        self._open()
        atexit.register(self.close)

    # ------------------------------------------------------------------------------------------------- mapping
    @staticmethod
    def _page_size(d: Path) -> int:
        try:
            return int(os.statvfs(d).f_bsize)
        except OSError:
            return 4096

    def _check_space(self) -> None:
        need = self.geom.total
        st = os.statvfs(self.path.parent)
        free = st.f_bavail * st.f_frsize
        if free < need:
            raise RuntimeError(
                f"ds41h engram_host: {self.path.parent} has {free / 2**30:.1f} GiB free, the table needs "
                f"{need / 2**30:.1f} GiB (both Engram layers: ~2x). Enlarge the mount (docs/"
                f"ENGRAM.md) or unset DS41H_ENGRAM_HOST.")
        avail = meminfo().get("MemAvailable", 0)
        if st.f_bsize <= 4096 and avail and avail < need + (8 << 30):  # tmpfs: the table is ordinary RAM
            raise RuntimeError(
                f"ds41h engram_host: MemAvailable {avail / 2**30:.1f} GiB < table {need / 2**30:.1f} GiB + 8 GiB "
                f"reserve; refusing to fill {self.path.parent} (unset DS41H_ENGRAM_HOST)")

    def _map(self, fd: int, size: int) -> None:
        self.mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.buf = torch.frombuffer(self.mm, dtype=torch.uint8)
        self.base = self.buf.data_ptr()

    def _header_ok(self) -> bool:
        mm = self.mm
        if mm is None or len(mm) < HEADER or mm[:8] != MAGIC:
            return False
        if struct.unpack_from("<I", mm, _OFF_READY)[0] != _READY:
            return False
        g = self.geom
        want = (VERSION, LAYOUTS[g.layout], g.rows, DIM, SDIM, g.w_off, g.s_off, g.w_stride, g.s_stride, g.total)
        if _GEOM.unpack_from(mm, _OFF_GEOM) != want:
            return False
        n = struct.unpack_from("<I", mm, _OFF_KEY)[0]
        if bytes(mm[_OFF_KEY + 4 : _OFF_KEY + 4 + n]).decode("utf-8", "replace") != self.key:
            return False
        return struct.unpack_from("<I", mm, _OFF_SHARDS)[0] == self.n_shards

    def _write_header(self) -> None:
        mm = self.mm
        g = self.geom
        key = self.key.encode()[:880]
        mm[:8] = MAGIC
        _GEOM.pack_into(mm, _OFF_GEOM, VERSION, LAYOUTS[g.layout], g.rows, DIM, SDIM, g.w_off, g.s_off, g.w_stride,
                        g.s_stride, g.total)
        struct.pack_into("<I", mm, _OFF_KEY, len(key))
        mm[_OFF_KEY + 4 : _OFF_KEY + 4 + len(key)] = key
        struct.pack_into("<I", mm, _OFF_SHARDS, self.n_shards)
        mm[_OFF_SHARDS + 4 : _OFF_SHARDS + 4 + 2 * MAX_SHARDS] = bytes(2 * MAX_SHARDS)
        struct.pack_into("<I", mm, _OFF_READY, _READY)

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fd = -1
            if self.path.exists():
                fd = os.open(self.path, os.O_RDWR)
                if os.fstat(fd).st_size == self.geom.total:
                    self._map(fd, self.geom.total)
                    if not self._header_ok():
                        self._unmap()
                if self.mm is None:  # stale (other checkpoint / geometry / torn create): replace, never truncate
                    os.close(fd)
                    fd = -1
                    logger.warning("ds41h engram_host: replacing stale table %s", self.path)
                    os.unlink(self.path)
            if self.mm is None:
                self._drop_unused_siblings()
                self._check_space()
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.ftruncate(fd, self.geom.total)
                self._map(fd, self.geom.total)
                self._advise_huge()
                self._write_header()
                self.created = True
            # lifetime marker: a shared lock on the data file while this process uses it (drop/unlink checks it)
            fcntl.flock(fd, fcntl.LOCK_SH)
            self._use_fd = fd
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)

    def _drop_unused_siblings(self) -> None:
        """Before creating a table, remove unused tables of the same layer under other keys / layouts (left by a killed
        server or an older checkpoint) so at most one copy per layer occupies RAM. Tables in use are kept."""
        stem = self.path.name.split("_")
        prefix = "_".join(stem[:3]) + "_"  # ds41h_engram_L<layer>_
        for p in self.path.parent.glob(prefix + "*.bin"):
            if p != self.path and not _in_use(p):
                logger.warning("ds41h engram_host: removing unused table %s", p)
                p.unlink(missing_ok=True)

    def _advise_huge(self) -> None:
        try:
            self.mm.madvise(mmap.MADV_HUGEPAGE)  # effective on huge=advise/within_size tmpfs; ignored otherwise
        except (OSError, AttributeError, ValueError):
            pass

    def _unmap(self) -> None:
        self.buf = None
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:  # live tensor views: leave the mapping to the GC
                pass
        self.mm = None
        self.base = 0

    def populate(self, threads: int | None = None) -> float:
        """Fault in (allocate) every page of the mapping with parallel MADV_POPULATE_WRITE; returns seconds."""
        threads = threads or int(os.environ.get("DS41H_ENGRAM_HOST_POPULATE_THREADS", "8"))
        total = self.geom.total
        chunk = _align(-(-total // threads), ALIGN)
        errs: list[int] = []
        t0 = time.perf_counter()

        def work(lo: int) -> None:
            n = min(chunk, total - lo)
            if n > 0 and _libc().madvise(ctypes.c_void_p(self.base + lo), n, MADV_POPULATE_WRITE) != 0:
                errs.append(ctypes.get_errno())

        ts = [threading.Thread(target=work, args=(i * chunk,)) for i in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            logger.warning("ds41h engram_host: MADV_POPULATE_WRITE failed (errno %s); pages fault on first touch", errs[0])
        return time.perf_counter() - t0

    def register(self) -> float:
        """cudaHostRegister the whole mapping in this process (current device); returns seconds."""
        if self._registered:
            return 0.0
        t0 = time.perf_counter()
        err, msg, dptr = ext().host_register(self.base, self.geom.total, False)
        self._chunks: list[int] = [self.base]
        if err and ext().host_ptr_is_device_ptr():
            # fallback: register in 16 GiB pieces (UVA: device address == host address, so one base still works)
            self._chunks, step, err2 = [], 16 << 30, 0
            for off in range(0, self.geom.total, step):
                e2, m2, _ = ext().host_register(self.base + off, min(step, self.geom.total - off), False)
                if e2:
                    err2, msg = e2, f"{msg}; chunked: {m2}"
                    break
                self._chunks.append(self.base + off)
            if err2:
                for c in self._chunks:
                    ext().host_unregister(c)
            else:
                logger.warning("ds41h engram_host: whole-range cudaHostRegister failed (%s); registered %d chunks",
                               msg, len(self._chunks))
                err, dptr = 0, self.base
        if err:
            raise RuntimeError(
                f"ds41h engram_host: cudaHostRegister({self.geom.total / 2**30:.1f} GiB of {self.path}) failed: "
                f"{msg} (code {err}); see docs/ENGRAM.md (memlock / IOMMU / hugetlbfs setup)")
        self.dev_base = dptr
        self._registered = True
        return time.perf_counter() - t0

    def close(self) -> None:
        """Unregister + unmap (if no tensor views are alive) and, unless KEEP, unlink the file when this process is its
        last user. Serialized with _open() on the .lock file, so a starting server never maps a file being unlinked."""
        if self._registered and self.base:
            try:
                for c in getattr(self, "_chunks", [self.base]):
                    ext().host_unregister(c)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass
            self._registered = False
        if self._use_fd >= 0:
            fd, self._use_fd = self._use_fd, -1
            lock = -1
            try:
                if not self.keep and self.path.parent.is_dir():
                    lock = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:  # no other shared lock -> last user; only unlink the inode we mapped
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if os.stat(self.path).st_ino == os.fstat(fd).st_ino:
                            os.unlink(self.path)
                    except OSError:
                        pass
            finally:
                os.close(fd)
                if lock >= 0:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    os.close(lock)
        self._unmap()

    # ------------------------------------------------------------------------------------------------- shards
    def _state_off(self, shard: int, kind: str) -> int:
        assert 0 <= shard < self.n_shards
        return _OFF_SHARDS + 4 + 2 * shard + (0 if kind == "weight" else 1)

    def state(self, shard: int, kind: str) -> int:
        return self.mm[self._state_off(shard, kind)]

    def set_state(self, shard: int, kind: str, value: int) -> None:
        self.mm[self._state_off(shard, kind)] = value

    def shard_done(self, shard: int) -> bool:
        return self.state(shard, "weight") == DONE and self.state(shard, "scale") == DONE

    def all_done(self) -> bool:
        return all(self.shard_done(s) for s in range(self.n_shards))

    def wait_ready(self, timeout: float = 3600.0) -> None:
        """Block (host side, no CUDA work) until every shard is loaded (other TP ranks write theirs)."""
        if self._ready:
            return
        t0 = time.monotonic()
        while not self.all_done():
            if time.monotonic() - t0 > timeout:
                st = [(self.state(s, "weight"), self.state(s, "scale")) for s in range(self.n_shards)]
                raise TimeoutError(f"ds41h engram_host: {self.path} shards not loaded after {timeout:.0f} s: {st}")
            time.sleep(0.02)
        self._ready = True

    # ------------------------------------------------------------------------------------------------- views
    def views(self, row_lo: int, row_hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU tensor views of rows [row_lo, row_hi): e4m3 [n, 256] and uint8 ue8m0 [n, 8] (param shapes of stock
        ParallelEngramEmbedding); in-place copies into them land in the shared mapping."""
        g, n = self.geom, row_hi - row_lo
        assert 0 <= row_lo <= row_hi <= g.rows
        w = self.buf.as_strided((n, DIM), (g.w_stride, 1), g.w_off + row_lo * g.w_stride)
        s = self.buf.as_strided((n, SDIM), (g.s_stride, 1), g.s_off + row_lo * g.s_stride)
        return w.view(torch.float8_e4m3fn), s

    def fill_from(self, weight: torch.Tensor, scale: torch.Tensor, row_lo: int = 0) -> None:
        """Copy table rows (e4m3/uint8 [n, 256] + uint8/e8m0 [n, 8], any device) into the mapping at row_lo."""
        n = weight.shape[0]
        w, s = self.views(row_lo, row_lo + n)
        w.copy_(weight.view(torch.float8_e4m3fn).cpu())
        s.copy_(scale.view(torch.uint8).cpu())

    # ------------------------------------------------------------------------------------------------- lookup
    def gather(self, ids: torch.Tensor, out: torch.Tensor, lo: list[int], hi: list[int], *, ctas: int | None = None,
               threads: int | None = None, unroll: int | None = None) -> None:
        """out[t, h] = stock-dequantized row ids[t, h] if lo[h] <= id < hi[h] else 0; on the current stream."""
        assert self._registered, "register() first"
        g = self.geom
        gather_rows(self.dev_base + g.w_off, self.dev_base + g.s_off, g.w_stride, g.s_stride, ids, out, lo, hi,
                    ctas=ctas, threads=threads, unroll=unroll)


def launch_config(rows: int, ctas: int | None = None, threads: int | None = None,
                  unroll: int | None = None) -> tuple[int, int, int]:
    """(ctas, threads, unroll) for `rows` table rows: small lookups (decode) put every row in flight at once on as
    many small CTAs as needed; large ones (prefill) use a few fat persistent CTAs (PCIe needs ~100s of KB in flight,
    not SMs) so the main stream keeps its SMs."""
    if rows <= 4096:
        t = threads or 256
        return ctas or 132, t, unroll or 1
    return (ctas or int(os.environ.get("DS41H_ENGRAM_HOST_CTAS", "16")),
            threads or int(os.environ.get("DS41H_ENGRAM_HOST_THREADS", "1024")),
            unroll or int(os.environ.get("DS41H_ENGRAM_HOST_UNROLL", "4")))


def gather_rows(w_ptr: int, s_ptr: int, w_stride: int, s_stride: int, ids: torch.Tensor, out: torch.Tensor,
                lo: list[int], hi: list[int], *, ctas: int | None = None, threads: int | None = None,
                unroll: int | None = None, vec: int | None = None) -> None:
    rows = ids.shape[0] * ids.shape[1]
    c, t, u = launch_config(rows, ctas, threads, unroll)
    if vec is None:
        vec = 16 if (w_ptr % 16 == 0 and w_stride % 16 == 0) else 8
    ext().gather(w_ptr, s_ptr, w_stride, s_stride, ids, out, [int(x) for x in lo], [int(x) for x in hi], c, t, u, vec)


# --------------------------------------------------------------------------------------------- layout helpers
def head_ranges(head_sizes: list[int] | tuple[int, ...], tp_size: int) -> tuple[list[int], list[int]]:
    """Per hash column h: the row range [lo, hi) of the TP rank that owns head h under stock sharding
    (ParallelEngramEmbedding.__init__: part = cdiv(n_cols, tp), rank r owns heads [r*part, (r+1)*part))."""
    n = len(head_sizes)
    part = -(-n // tp_size)
    lo, hi = [], []
    for h in range(n):
        r = h // part
        lo.append(sum(head_sizes[: r * part]))
        hi.append(sum(head_sizes[: (r + 1) * part]))
    return lo, hi


def shard_rows(head_sizes: list[int] | tuple[int, ...], tp_size: int, rank: int) -> tuple[int, int]:
    n = len(head_sizes)
    part = -(-n // tp_size)
    return sum(head_sizes[: rank * part]), sum(head_sizes[: (rank + 1) * part])


def checkpoint_key(model_path: str, layer_id: int, rows: int, load_format: str = "") -> str:
    """Identity of the table contents: model path + the safetensors file holding the table (size, mtime) + rows."""
    ident = ""
    try:
        idx = Path(model_path) / "model.safetensors.index.json"
        wm = json.loads(idx.read_text())["weight_map"]
        f = Path(model_path) / wm[f"layers.{layer_id}.engram.embed.weight"]
        st = f.stat()
        ident = f"{f.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    except Exception:  # noqa: BLE001
        ident = os.path.realpath(model_path) if model_path else "?"
    return f"ds41h-engram v{VERSION} L{layer_id} rows={rows} {ident} fmt={load_format}"


def table_path(key: str, layer_id: int, layout: str) -> Path:
    h = hashlib.sha1(f"{key}|{layout}".encode()).hexdigest()[:12]
    return host_dir() / f"ds41h_engram_L{layer_id}_{layout}_{h}.bin"


# --------------------------------------------------------------------------------------------- CLI
def _tables(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("ds41h_engram_*.bin"))


def _in_use(p: Path) -> bool:
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def cli_status(d: Path) -> None:
    for p in _tables(d):
        size = p.stat().st_size
        with open(p, "rb") as f:
            mm = mmap.mmap(f.fileno(), min(size, HEADER), prot=mmap.PROT_READ)
            ok = mm[:8] == MAGIC
            n = struct.unpack_from("<I", mm, _OFF_KEY)[0] if ok else 0
            key = bytes(mm[_OFF_KEY + 4 : _OFF_KEY + 4 + n]).decode("utf-8", "replace") if ok else "?"
            ns = struct.unpack_from("<I", mm, _OFF_SHARDS)[0] if ok else 0
            st = [tuple(mm[_OFF_SHARDS + 4 + 2 * s : _OFF_SHARDS + 6 + 2 * s]) for s in range(ns)]
            mm.close()
        print(f"{p}  {size / 2**30:.1f} GiB  in_use={_in_use(p)}  shards(w,s)={st}\n    key={key}")


def cli_drop(d: Path, force_all: bool) -> None:
    for p in _tables(d):
        if _in_use(p) and not force_all:
            print(f"in use, kept: {p}")
            continue
        p.unlink(missing_ok=True)
        Path(str(p) + ".lock").unlink(missing_ok=True)
        print(f"removed {p}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m ds41h.engram_host")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    dp = sub.add_parser("drop")
    dp.add_argument("--all", action="store_true", help="also remove tables in use (their users keep the mapping)")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        cli_status(host_dir())
    elif args.cmd == "drop":
        cli_drop(host_dir(), args.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
