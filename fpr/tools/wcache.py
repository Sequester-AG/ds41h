"""Converted-weight / input cache for GPU jobs.

Why: a 4-GPU A/B job spends ~13 s re-reading and converting ~16 GB of checkpoint weights at ~1.2 GB/s (CPU-bound)
for every variant, and ~3 s on the GPUs, while holding all four GPUs. Build once, then load it from the
page cache (no conversion) at ~5.6 GB/s and copy to the device at PCIe speed.

    import sys; sys.path.insert(0, "fpr/tools")
    from wcache import cached
    W = cached(f"L{layer}_r{r}_raw_experts_v1", lambda: load_raw_experts(layer, r, "cpu", model), device=dev)

Rules:
- The key must change whenever the builder's output changes (put layout / version / flag values in it).
- Build on the CPU if you can (no GPU needed, so it can run before the GPU work); a CUDA result is moved to the CPU to save.
- Objects: tensors, dicts, lists, tuples, namedtuples, plain Python values. Views keep their shared storage.
- Cache dir: $PHYS_WCACHE or $DS41H_HOME/cache/wcache (default ~/.ds41h; best on NVMe, stays in the page cache).
  Weights and derived inputs only: never cache request data here.
"""
import os
import threading
import time

import torch

ROOT = os.environ.get("PHYS_WCACHE", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "cache", "wcache"))
# Fast device path: a cache hit with a CUDA device reads each tensor's bytes straight
# from the page cache (os.preadv, GIL released) into two pinned 64 MB staging buffers per thread and DMAs them into a
# fresh device tensor on a side stream. Same result as torch.load + .to(device): one new contiguous device tensor per
# tensor (no aliasing between tensors, exactly like .to()); non-contiguous / Parameter / requires_grad entries take the
# plain path. Opt out with PHYS_WCACHE_FAST=0.
FAST = os.environ.get("PHYS_WCACHE_FAST", "1") != "0"
CHUNK = 64 << 20
_tl = threading.local()


def _map(x, f):
    if torch.is_tensor(x):
        return f(x)
    if isinstance(x, dict):
        return type(x)((k, _map(v, f)) for k, v in x.items())
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[_map(v, f) for v in x])
    if isinstance(x, (list, tuple)):
        return type(x)(_map(v, f) for v in x)
    return x


def path_of(key):
    return os.path.join(ROOT, key + ".pt")


def _tensors(x, out):
    if torch.is_tensor(x):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _tensors(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _tensors(v, out)
    return out


def _staging(dev):
    st = getattr(_tl, "st", None)
    if st is None:
        bufs = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        st = _tl.st = dict(bufs=bufs, views=[memoryview(b.numpy()) for b in bufs], streams={})
    if dev.index not in st["streams"]:
        st["streams"][dev.index] = torch.cuda.Stream(device=dev)
    return st


def _plan(p):
    """(obj, tensors, base) for the fast path, or None. obj's tensors are views of the file mapping; a tensor's bytes
    sit at file offset storage.data_ptr() - base + storage_offset * element_size (checked against the zip records)."""
    obj = torch.load(p, mmap=True, map_location="cpu", weights_only=False)   # pickle only; storages stay mapped
    ts = _tensors(obj, [])
    if not ts or any(type(t) is not torch.Tensor or t.requires_grad for t in ts):
        return None
    with open(p, "rb") as fh:
        rd = torch._C.PyTorchFileReader(fh)
        offs = sorted(rd.get_record_offset(r) for r in rd.get_all_records() if r.startswith("data/"))
    ptrs = sorted({t.untyped_storage().data_ptr() for t in ts if t.untyped_storage().nbytes() > 0})
    if not ptrs or not offs:
        return None
    base = ptrs[0] - offs[0]
    if not {q - base for q in ptrs} <= set(offs):
        return None                                   # mapping layout not as expected: plain path
    return obj, ts, base


def _load_fast(p, device):
    """Cache hit -> device without unpickling copies or pageable copies. Returns None to ask for the plain path."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    pl = _plan(p)
    if pl is None:
        return None
    obj, ts, base = pl
    st = _staging(dev)
    stream, bufs, views = st["streams"][dev.index], st["bufs"], st["views"]
    jobs, out = [], {}
    with torch.cuda.device(dev):
        for t in ts:
            if id(t) in out:
                continue
            if not t.is_contiguous():
                out[id(t)] = t.to(dev)                # rare: the plain per-tensor copy, same as .to()
                continue
            d = torch.empty(t.shape, dtype=t.dtype, device=dev)
            out[id(t)] = d
            n = t.numel() * t.element_size()
            if n:
                off = t.untyped_storage().data_ptr() - base + t.storage_offset() * t.element_size()
                jobs.append((off, n, d.reshape(-1).view(torch.uint8)))
    jobs.sort(key=lambda j: j[0])
    fd = os.open(p, os.O_RDONLY)
    ev = [None, None]
    i = 0
    try:
        with torch.cuda.stream(stream):
            for off, n, dst in jobs:
                pos = 0
                while pos < n:
                    m = min(CHUNK, n - pos)
                    if ev[i] is not None:
                        ev[i].synchronize()           # this staging buffer's previous DMA has finished
                    got = 0
                    while got < m:
                        k = os.preadv(fd, [views[i][got:m]], off + pos + got)
                        if k <= 0:
                            raise IOError(f"wcache: short read {p} at {off + pos + got}")
                        got += k
                    dst[pos:pos + m].copy_(bufs[i][:m], non_blocking=True)
                    e = torch.cuda.Event()
                    e.record(stream)
                    ev[i] = e
                    pos += m
                    i ^= 1
        stream.synchronize()
    finally:
        os.close(fd)
    # the device tensors were allocated on the caller's current stream and written only on `stream`, synced above
    return _map(obj, lambda t: out[id(t)])


def cached(key, build, device=None, verbose=True, mmap=False):
    """Return build()'s result, from the cache if present. device: move tensors there after loading (None = CPU).
    Measured (4 GB entry, page cache hot): plain load 0.72 s (5.6 GB/s); mmap=True returns at once but pays
    page faults on first touch (a full read took 3.0 s), so plain is the default. A hit with a CUDA device takes the
    fast path (_load_fast) unless PHYS_WCACHE_FAST=0."""
    p = path_of(key)
    t0 = time.time()
    if FAST and device is not None and not mmap and os.path.exists(p) and torch.device(device).type == "cuda":
        obj = _load_fast(p, device)
        if obj is not None:
            if verbose:
                print(f"wcache fast {key}: {time.time() - t0:.2f} s to {device}", flush=True)
            return obj
    if os.path.exists(p):
        obj = torch.load(p, mmap=mmap, map_location="cpu", weights_only=False)
        how = "hit"
    else:
        obj = _map(build(), lambda t: t.detach().cpu())
        os.makedirs(ROOT, exist_ok=True)
        tmp = f"{p}.tmp{os.getpid()}"
        torch.save(obj, tmp)
        os.replace(tmp, p)
        how = "built"
    t1 = time.time()
    if device is not None:
        obj = _map(obj, lambda t: t.to(device))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    if verbose:
        print(f"wcache {how} {key}: load {t1 - t0:.2f} s, to {device} {time.time() - t1:.2f} s", flush=True)
    return obj


def cached_many(items, threads=8, verbose=True, mmap=False):
    """Load many entries in parallel threads; items = [(key, build, device), ...], results in order.
    Measured (page cache hot, 4 x 3 GB entries): serial plain load 3.8 GB/s, 4 threads 14.8 GB/s.
    Build missing entries first (prebuild), since builders then run concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(cached, k, b, d, False, mmap) for k, b, d in items]
        out = [f.result() for f in futs]
    if verbose:
        print(f"wcache cached_many: {len(items)} entries in {time.time() - t0:.2f} s ({threads} threads)", flush=True)
    return out


def prebuild(key, build):
    """Build the cache entry without loading it to a device (CPU only: CUDA_VISIBLE_DEVICES= )."""
    if not os.path.exists(path_of(key)):
        cached(key, build, device=None)
