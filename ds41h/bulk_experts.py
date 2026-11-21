"""Bulk loading of the raw MXFP4 routed experts (DS41H_BULK_EXPERTS=1), for fast server start.

Stock loading goes through vLLM's per-expert RoutedExperts.weight_loader: ~9,000 small loader calls per layer, each
page-faulting a strided slice of the memory-mapped checkpoint (vl_model.load_weights also sorts all ~96K tensors by name
first, so the shards are touched in scattered order).

With the flag set, the raw MoE method's per-expert loaders become no-ops and process_weights_after_loading fills each
layer's four raw parameters directly from the safetensors byte offsets:
  * w13 / w13_scale (TP split along rows, dim 0): every expert's w1 and w3 row slice is one contiguous byte range,
    read with preadv straight into a pinned buffer laid out exactly like the parameter, then one H2D copy;
  * w2 / w2_scale (TP split along dim 1): full rows are read into a pinned buffer, the rank's columns are gathered on
    the CPU into a second pinned buffer, then one H2D copy (no GPU temporary, so the KV cache size is unchanged).
Slices come from ds41h.ref.weights.tp_range, the same TP rules as the stock loader (and load_raw_experts).

DS41H_BULK_THREADS (default 16): read threads per rank.
DS41H_PARAM_CHECKSUM=<dir>: write a per-layer checksum of the four expert parameters (stock or bulk path) to
<dir>/rank<r>.txt, to compare the two paths bit for bit.
"""
from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_LOCK = threading.Lock()
_HDR: dict[str, tuple[int, dict]] = {}
_FDS: dict[str, int] = {}
_STATE = {"index": None, "pool": None, "stage": {}, "calls": 0, "secs": 0.0, "bytes": 0}
_DRAFTER_FIRST_LAYER = 40


def enabled() -> bool:
    return os.environ.get("DS41H_BULK_EXPERTS", "0") not in ("0", "", "false", "False")


def checksum_dir() -> str | None:
    return os.environ.get("DS41H_PARAM_CHECKSUM") or None


_EXPERT_RE = re.compile(r"\.experts\.\d+\.")


def install_iterator_filter() -> None:
    """Skip every routed-expert tensor (weights and scales) in vLLM's safetensors iterator, before it is materialized.

    The iterator calls weight_utils.should_skip_weight(name, local_expert_ids) for each tensor (the EP weight filter);
    stock only skips `.weight` tensors of non-local experts under EP. In bulk mode all routed experts are filled by fill(), so iterating ~92K expert tensors per load pass is pure overhead.
    """
    from vllm.model_executor.model_loader import weight_utils

    cur = weight_utils.should_skip_weight
    if getattr(cur, "ds41h_bulk", False):
        return

    def should_skip_weight(weight_name, local_expert_ids):
        if _EXPERT_RE.search(weight_name):
            return True
        return cur(weight_name, local_expert_ids)

    should_skip_weight.ds41h_bulk = True
    weight_utils.should_skip_weight = should_skip_weight


def noop_loader(loader):
    """Stand-in for the per-expert weight loader in bulk mode (the data is filled after loading)."""

    def _skip(param, loaded_weight, *args, **kwargs):
        return True if kwargs.get("return_success") else None

    _skip.ds41h_bulk_skip = True
    _skip.ds41h_stock_loader = loader
    return _skip


# ------------------------------------------------------------------------------------------------ checkpoint index
def _model_dir() -> str:
    md = os.environ.get("DS41H_MODEL_DIR") or os.environ.get("MODEL")
    if not md:
        raise RuntimeError("ds41h bulk experts: DS41H_MODEL_DIR is not set")
    return md


def _index(md: str) -> dict:
    with _LOCK:
        if _STATE["index"] is None:
            with open(os.path.join(md, "model.safetensors.index.json")) as f:
                _STATE["index"] = json.load(f)["weight_map"]
        return _STATE["index"]


def _info(md: str, name: str) -> tuple[str, int, int, list]:
    """(path, absolute byte offset, nbytes, shape) of a checkpoint tensor."""
    fname = _index(md)[name]
    with _LOCK:
        if fname not in _HDR:
            with open(os.path.join(md, fname), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                _HDR[fname] = (8 + n, json.loads(f.read(n)))
        start, hdr = _HDR[fname]
    t = hdr[name]
    a, b = t["data_offsets"]
    return os.path.join(md, fname), start + a, b - a, t["shape"]


def _fd(path: str) -> int:
    with _LOCK:
        if path not in _FDS:
            _FDS[path] = os.open(path, os.O_RDONLY)
        return _FDS[path]


def _pool() -> ThreadPoolExecutor:
    with _LOCK:
        if _STATE["pool"] is None:
            _STATE["pool"] = ThreadPoolExecutor(int(os.environ.get("DS41H_BULK_THREADS", "16")), thread_name_prefix="ds41h-bulk")
        return _STATE["pool"]


def _pread_full(fd: int, mv: memoryview, off: int) -> None:
    pos, n = 0, len(mv)
    while pos < n:
        got = os.preadv(fd, [mv[pos:]], off + pos)
        if got <= 0:
            raise OSError(f"ds41h bulk experts: short read at offset {off + pos}")
        pos += got


def _stage(key: str, shape: tuple, torch):
    """Reusable pinned host buffer (one per role and shape)."""
    buf = _STATE["stage"].get(key)
    if buf is None or tuple(buf.shape) != tuple(shape):
        buf = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        _STATE["stage"][key] = buf
    return buf


def ckpt_prefix(layer_name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", layer_name)
    if not m:
        raise RuntimeError(f"ds41h bulk experts: cannot parse a layer index from {layer_name!r}")
    i = int(m.group(1))
    return f"layers.{i}.ffn.experts." if i < _DRAFTER_FIRST_LAYER else f"mtp.{i - _DRAFTER_FIRST_LAYER}.ffn.experts."


# ------------------------------------------------------------------------------------------------ fill
def fill(layer) -> None:
    """Read this rank's raw expert tensors of `layer` from the checkpoint into its four parameters."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    from ds41h.ref.weights import tp_range

    t0 = time.time()
    md = _model_dir()
    rank, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    pre = ckpt_prefix(str(getattr(layer, "layer_name", "")))
    w13, s13, w2, s2 = layer.w13_weight, layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale
    E, I2 = w13.shape[0], w13.shape[1]
    I = I2 // 2
    jobs = []  # (fd, memoryview, file offset)

    # w13 / s13: [E, 2I, K] with rows 0..I-1 = w1 (gate) slice, I..2I-1 = w3 (up) slice; dim-0 split = contiguous rows
    b13 = _stage("w13", tuple(w13.shape), torch)
    c13 = _stage("s13", tuple(s13.shape), torch)
    v13, vs13 = memoryview(b13.numpy()).cast("B"), memoryview(c13.numpy()).cast("B")
    for buf_mv, param, suffix in ((v13, w13, "weight"), (vs13, s13, "scale")):
        row_bytes = param.shape[2]
        for e in range(E):
            for half, w in enumerate(("w1", "w3")):
                name = f"{pre}{e}.{w}.{suffix}"
                path, off, nbytes, shape = _info(md, name)
                dim, start, stop = tp_range(name, tuple(shape), rank, tp)
                if dim != 0 or stop - start != I or nbytes != shape[0] * row_bytes:
                    raise RuntimeError(f"ds41h bulk experts: unexpected layout for {name}: dim {dim} [{start},{stop}) "
                                       f"shape {shape} bytes {nbytes} vs param {tuple(param.shape)}")
                dst0 = (e * I2 + half * I) * row_bytes
                jobs.append((_fd(path), buf_mv[dst0:dst0 + I * row_bytes], off + start * row_bytes))

    # w2 / s2: [E, H, K/tp]; dim-1 split -> read full rows [E, H, K], gather the rank's columns on the CPU
    full = {}
    for key, param, suffix in (("w2", w2, "weight"), ("s2", s2, "scale")):
        name0 = f"{pre}0.w2.{suffix}"
        _, _, nbytes0, shape0 = _info(md, name0)
        H, K = shape0[0], nbytes0 // shape0[0]
        fbuf = _stage(key + "full", (E, H, K), torch)
        fmv = memoryview(fbuf.numpy()).cast("B")
        for e in range(E):
            name = f"{pre}{e}.w2.{suffix}"
            path, off, nbytes, shape = _info(md, name)
            if nbytes != H * K:
                raise RuntimeError(f"ds41h bulk experts: {name} has {nbytes} bytes, expected {H}x{K}")
            jobs.append((_fd(path), fmv[e * H * K:(e + 1) * H * K], off))
        dim, start, stop = tp_range(name0, tuple(shape0), rank, tp)
        if dim != 1 or (stop - start) != param.shape[2] or tuple(param.shape[:2]) != (E, H):
            raise RuntimeError(f"ds41h bulk experts: unexpected layout for {name0}: dim {dim} [{start},{stop}) "
                               f"param {tuple(param.shape)}")
        full[key] = (fbuf, start, stop)

    nbytes_read = sum(len(mv) for _, mv, _ in jobs)
    list(_pool().map(lambda j: _pread_full(*j), jobs))
    t_read = time.time() - t0

    w13.data.copy_(b13, non_blocking=True)
    s13.data.copy_(c13, non_blocking=True)
    for key, param in (("w2", w2), ("s2", s2)):
        fbuf, start, stop = full[key]
        sl = _stage(key, tuple(param.shape), torch)
        sl.copy_(fbuf[:, :, start:stop])  # CPU column gather into a pinned buffer
        param.data.copy_(sl, non_blocking=True)
    torch.cuda.current_stream().synchronize()  # staging buffers are reused by the next layer
    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
    if m and int(m.group(1)) in (_DRAFTER_FIRST_LAYER - 1, _DRAFTER_FIRST_LAYER + 2):
        _STATE["stage"].clear()  # last main / last drafter MoE layer: release ~4 GB of pinned host memory per rank

    dt = time.time() - t0
    _STATE["calls"] += 1
    _STATE["secs"] += dt
    _STATE["bytes"] += nbytes_read
    if _STATE["calls"] in (1, 10, 20, 30, 40, 43) or os.environ.get("DS41H_BULK_VERBOSE"):
        from vllm.logger import init_logger

        init_logger("ds41h.bulk").info(
            "ds41h bulk experts: %s rank %d: %d experts, %.2f GB read in %.2f s, total %.2f s (%d layers, %.1f GB, %.1f s so far)",
            pre.rstrip("."), rank, E, nbytes_read / 1e9, t_read, dt, _STATE["calls"], _STATE["bytes"] / 1e9, _STATE["secs"])


# ------------------------------------------------------------------------------------------------ verification
def checksum(layer) -> None:
    """Append position-weighted int64 checksums of the four expert parameters to <dir>/rank<r>.txt."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    d = checksum_dir()
    os.makedirs(d, exist_ok=True)
    rank = get_tensor_model_parallel_rank()
    out = []
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        flat = getattr(layer, name).data.reshape(-1)
        s1 = torch.zeros((), dtype=torch.int64, device=flat.device)
        s2 = torch.zeros((), dtype=torch.int64, device=flat.device)
        step = 1 << 26
        w = (torch.arange(step, device=flat.device, dtype=torch.int64) * 2654435761) % 4294967291 + 1
        for i in range(0, flat.numel(), step):
            c = flat[i:i + step].to(torch.int64)
            s1 += c.sum()
            s2 += (c * w[:c.numel()]).sum()  # wraps mod 2^64: still deterministic and order-independent
        out.append(f"{name}={int(s1)}:{int(s2)}")
    with _LOCK, open(os.path.join(d, f"rank{rank}.txt"), "a") as f:
        f.write(f"{getattr(layer, 'layer_name', '?')} {' '.join(out)}\n")
