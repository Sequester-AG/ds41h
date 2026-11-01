"""KV-RAM tier helpers for the ds41h vLLM 0.32 server (v1.3: allocator cap, step trace).

Loaded into every vLLM worker with  --worker-extension-cls kvram_ext.KVRamWorkerExt  (this file's directory on PYTHONPATH).
Importing it (in the workers, before the model is built) applies the opt-in patches below; the class adds debug
collective-RPC methods (dev mode: POST /collective_rpc {"method": "kvram_info"}).

The tier itself is stock vLLM 0.32: --kv-transfer-config with the OffloadingConnector (CPU backend, shared /dev/shm region,
async loads, write-through stores of the prefix-cacheable group = DSv4.1 compressed KV + indexer K).

Env switches (read at import time in each worker):
  KVRAM_DEDUP=1   mark DeepSeek-V4.1's single-latent-head attention specs max_tp_shards=1 (they are computed by replicated
                  projections on every TP rank). vLLM then sees kv_tp_replicas == tp_size and uses its own single-copy
                  "replicated layout": ONE host copy, written by rank 0, read by all ranks (instead of 4 rank copies).
                  Valid only if the KV bytes are identical on all ranks -> check with kvram_slots_compare first
                  (KVRAM_DEDUP=0 keeps one slot per rank).
  KVRAM_CHECK=1   in-situ restore check: after every completed host->GPU load (and GPU->host store) each copy descriptor's
                  destination bytes are compared with its source bytes (synchronous, debug only). Totals via kvram_check.
  KVRAM_MEM_FRACTION=<f>  (v1.2) cap this worker's PyTorch caching allocator at f x the GPU's memory
                  (torch.cuda.set_per_process_memory_fraction, applied at the start of Worker.load_model, i.e. after
                  init_device and before the weights load). At the cap the allocator releases its unused cached blocks
                  and retries (counted in num_alloc_retries) instead of growing until the device is full, so memory
                  outside PyTorch (CUDA context, NCCL, module loads) keeps its room. Allocator-only: numerics unchanged.
                  Each rank logs "kvram_ext: memory cap rank R cuda:D: ... GiB"; kvram_mem_stats reports cap_bytes.
  KVRAM_MEM_GC=<t>  (v1.2, optional, needs KVRAM_MEM_FRACTION) PyTorch garbage_collection_threshold t: above t x the cap the
                  allocator frees old unused cached blocks before a new cudaMalloc (gradual, instead of a full release
                  + retry at the cap). Set after load_model (vLLM restores its allocator settings when the load ends).
  KVRAM_STEP_TRACE=1  (v1.3, measurement only) wrap Worker.execute_model: per step the start time, the call duration and
                  the scheduled token count go into in-memory lists (~3 us per step); every 10 steps the allocator
                  counters (num_alloc_retries, num_device_free, num_device_alloc, reserved bytes) are read, and a change is
                  recorded as an event with its step range. Step period = time between consecutive execute_model starts
                  (= one worker loop iteration incl. sample_tokens). RPCs: kvram_trace_reset, kvram_trace_dump(path_prefix)
                  -> <prefix>.r<rank>.json + a summary.
Unset = v1.1 behaviour.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading

logger = logging.getLogger("vllm.kvram_ext")

_DEDUP = os.environ.get("KVRAM_DEDUP", "0") == "1"
_CHECK = os.environ.get("KVRAM_CHECK", "0") == "1"
_MEMFRAC_ENV = os.environ.get("KVRAM_MEM_FRACTION", "").strip()
_MEMGC_ENV = os.environ.get("KVRAM_MEM_GC", "").strip()
_memcap: dict = {}  # applied cap of this worker (kvram_mem_stats)
_TRACE = os.environ.get("KVRAM_STEP_TRACE", "0") == "1"
_TRACE_EVERY = 10  # allocator counters are read every N steps
_tr: dict = {}
_tr_lock = threading.Lock()

_check_lock = threading.Lock()
_check_active = [_CHECK]  # runtime toggle (kvram_check_set); wrappers are installed only when KVRAM_CHECK=1
_check_totals = {"load_ops": 0, "load_bytes": 0, "load_mismatch_ops": 0, "store_ops": 0, "store_bytes": 0,
                 "store_mismatch_ops": 0, "loads": 0, "stores": 0}


# --------------------------------------------------------------------------------------------------------------------
# KVRAM_DEDUP: replicated-latent spec marking
# --------------------------------------------------------------------------------------------------------------------
def _mark_replicated(specs: dict) -> dict:
    from vllm.v1.kv_cache_interface import AttentionSpec

    n = 0
    out = {}
    for name, spec in specs.items():
        # dcp_sharded defaults to True for attention specs; vLLM's replica count already ignores it unless DCP > 1
        if isinstance(spec, AttentionSpec) and spec.max_tp_shards is None and spec.num_kv_heads == 1:
            spec = dataclasses.replace(spec, max_tp_shards=1)
            n += 1
        out[name] = spec
    logger.info("kvram_ext: KVRAM_DEDUP marked %d of %d KV cache specs max_tp_shards=1", n, len(specs))
    return out


def _install_dedup() -> None:
    from vllm.v1.worker import gpu_worker

    orig = gpu_worker.Worker.get_kv_cache_spec
    if getattr(orig, "_kvram_wrapped", False):
        return

    def get_kv_cache_spec(self):
        specs = orig(self)
        mt = getattr(self.vllm_config.model_config.hf_config, "model_type", "")
        if not str(mt).startswith("deepseek_v4"):
            logger.warning("kvram_ext: KVRAM_DEDUP ignored for model_type %s", mt)
            return specs
        return _mark_replicated(specs)

    get_kv_cache_spec._kvram_wrapped = True  # type: ignore[attr-defined]
    gpu_worker.Worker.get_kv_cache_spec = get_kv_cache_spec


# --------------------------------------------------------------------------------------------------------------------
# KVRAM_CHECK: in-situ copy verification
# --------------------------------------------------------------------------------------------------------------------
def _install_check() -> None:
    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker as cw

    handler_cls = cw.SingleDirectionOffloadingHandler
    if getattr(handler_cls.transfer_async, "_kvram_wrapped", False):
        return
    orig_transfer = handler_cls.transfer_async
    orig_finished = handler_cls.get_finished

    def transfer_async(self, job_id, src_spec, dst_spec):
        # capture the descriptors this transfer issues (src/dst byte pointers + sizes)
        captured = {}
        swap = self._swap_blocks_batch

        def capture(src, dst, sizes, **kw):
            captured["ops"] = (src.numpy().copy(), dst.numpy().copy(), sizes.numpy().copy())
            return swap(src, dst, sizes, **kw)

        self._swap_blocks_batch = capture
        try:
            ok = orig_transfer(self, job_id, src_spec, dst_spec)
        finally:
            self._swap_blocks_batch = swap
        if "ops" in captured:
            pend = self.__dict__.setdefault("_kvram_pending", {})
            pend[job_id] = captured["ops"]
        return ok

    def _views(self):
        v = self.__dict__.get("_kvram_views")
        if v is None:
            gpu = self.src_tensors[0] if self.gpu_to_cpu else self.dst_tensors[0]
            cpu = self.dst_tensors[0] if self.gpu_to_cpu else self.src_tensors[0]
            gst = gpu.untyped_storage()
            cst = cpu.untyped_storage()
            gflat = torch.tensor([], dtype=torch.int8, device=gpu.device).set_(gst, 0, (gst.nbytes(),), (1,))
            cflat = torch.tensor([], dtype=torch.int8).set_(cst, 0, (cst.nbytes(),), (1,))
            v = (gflat, gst.data_ptr(), cflat, cst.data_ptr())
            self.__dict__["_kvram_views"] = v
        return v

    def verify(self, ops) -> tuple[int, int, int]:
        """Compare every copy's destination with its source, batched: one gather + one D2H copy per <= 256 ops."""
        gflat, gbase, cflat, cbase = _views(self)
        src, dst, sizes = ops
        gp, cp = (src, dst) if self.gpu_to_cpu else (dst, src)
        go = (gp.astype("int64") - gbase).tolist()
        co = (cp.astype("int64") - cbase).tolist()
        nl = sizes.tolist()
        nmis = 0
        for b in range(0, len(nl), 256):
            sl = range(b, min(len(nl), b + 256))
            g = torch.cat([gflat[go[i]:go[i] + nl[i]] for i in sl]).cpu()
            c = torch.cat([cflat[co[i]:co[i] + nl[i]] for i in sl])
            if not torch.equal(g, c):
                off = 0
                for i in sl:
                    if not torch.equal(g[off:off + nl[i]], c[off:off + nl[i]]):
                        nmis += 1
                    off += nl[i]
        return len(nl), int(sum(nl)), nmis

    def get_finished(self):
        res = orig_finished(self)
        pend = self.__dict__.get("_kvram_pending", {})
        for r in res:
            ops = pend.pop(r.job_id, None)
            if ops is None or not _check_active[0]:
                continue
            nops, nbytes, nmis = verify(self, ops)
            kind = "store" if self.gpu_to_cpu else "load"
            with _check_lock:
                _check_totals[f"{kind}s"] += 1
                _check_totals[f"{kind}_ops"] += nops
                _check_totals[f"{kind}_bytes"] += nbytes
                _check_totals[f"{kind}_mismatch_ops"] += nmis
            if nmis:
                logger.error("kvram_ext CHECK: %s job %d: %d of %d copies DIFFER", kind, r.job_id, nmis, nops)
        return res

    transfer_async._kvram_wrapped = True  # type: ignore[attr-defined]
    handler_cls.transfer_async = transfer_async
    handler_cls.get_finished = get_finished
    logger.info("kvram_ext: KVRAM_CHECK in-situ copy verification installed")


# --------------------------------------------------------------------------------------------------------------------
# KVRAM_MEM_FRACTION / KVRAM_MEM_GC: per-process cap of the PyTorch caching allocator (v1.2)
# --------------------------------------------------------------------------------------------------------------------
def _env_unit(name: str, val: str, lo: float, hi: float) -> float | None:
    if not val:
        return None
    try:
        x = float(val)
    except ValueError:
        logger.warning("kvram_ext: %s=%r is not a number -> ignored", name, val)
        return None
    if not lo < x <= hi:
        logger.warning("kvram_ext: %s=%s outside (%s, %s] -> ignored", name, val, lo, hi)
        return None
    return x


def _worker_device(worker):
    import torch

    dev = getattr(worker, "device", None)
    if isinstance(dev, torch.device) and dev.type == "cuda" and dev.index is not None:
        return dev.index
    return torch.cuda.current_device()


def _worker_rank(worker):
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return getattr(worker, "rank", 0)


def _apply_memcap(worker, frac: float) -> None:
    import torch

    idx = _worker_device(worker)
    total = int(torch.cuda.get_device_properties(idx).total_memory)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    free, _ = torch.cuda.mem_get_info(idx)
    _memcap.update(mem_fraction=frac, cap_bytes=int(frac * total), device=idx)
    logger.info("kvram_ext: memory cap rank %s cuda:%d: KVRAM_MEM_FRACTION %.4f = %.2f GiB of %.2f GiB (PyTorch reserved "
                "now %.2f GiB, device free %.2f GiB)", _worker_rank(worker), idx, frac, frac * total / 2**30,
                total / 2**30, torch.cuda.memory_reserved(idx) / 2**30, free / 2**30)


def _apply_gc(worker, thr: float) -> None:
    import torch

    get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if get_settings is not None:
        cur = get_settings()
    else:
        cur = torch.cuda.memory._snapshot()["allocator_settings"]["PYTORCH_CUDA_ALLOC_CONF"]
    # the setter resets options that are not included -> keep the current ones (as vLLM's _scoped_allocator_max_split)
    keep = [x for x in cur.strip().rstrip(",").split(",") if x and not x.startswith("garbage_collection_threshold")]
    settings = ",".join(keep + ["garbage_collection_threshold:%g" % thr])
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is not None:
        setter(settings)
    else:
        torch.cuda.memory._set_allocator_settings(settings)
    _memcap.update(gc_threshold=thr)
    logger.info("kvram_ext: allocator GC rank %s cuda:%d: garbage_collection_threshold %g (allocator settings now %r)",
                _worker_rank(worker), _worker_device(worker), thr, settings)


def _install_memcap(frac: float, gc: float | None) -> None:
    import functools

    from vllm.v1.worker import gpu_worker

    orig = gpu_worker.Worker.load_model
    if getattr(orig, "_kvram_wrapped", False):
        return

    @functools.wraps(orig)
    def load_model(self, *args, **kwargs):
        _apply_memcap(self, frac)
        out = orig(self, *args, **kwargs)
        if gc is not None:
            _apply_gc(self, gc)
        return out

    load_model._kvram_wrapped = True  # type: ignore[attr-defined]
    gpu_worker.Worker.load_model = load_model
    logger.info("kvram_ext: KVRAM_MEM_FRACTION %g%s: allocator cap installed (applied at Worker.load_model)", frac,
                "" if gc is None else ", KVRAM_MEM_GC %g" % gc)


# --------------------------------------------------------------------------------------------------------------------
# KVRAM_STEP_TRACE: per-step timing + allocator-counter trace (v1.3, measurement only)
# --------------------------------------------------------------------------------------------------------------------
def _alloc_counters() -> tuple:
    import torch

    s = torch.cuda.memory_stats()
    return (int(s.get("num_alloc_retries", 0)), int(s.get("num_device_free", 0)), int(s.get("num_device_alloc", 0)),
            int(s.get("reserved_bytes.all.current", 0)))


def _trace_reset() -> None:
    _tr.clear()
    _tr.update(t=[], d=[], k=[], ev=[], reads=[], ctr=_alloc_counters(), ctr_i=-1)


def _trace_counters(n: int, t1: float) -> None:
    c = _alloc_counters()
    p = _tr["ctr"]
    _tr["reads"].append((n - 1, round(t1, 6), c[3]))
    if c[0] != p[0] or c[1] != p[1]:
        _tr["ev"].append({"i0": _tr["ctr_i"] + 1, "i1": n - 1, "retries": c[0] - p[0], "frees": c[1] - p[1],
                          "mallocs": c[2] - p[2], "reserved_before": p[3], "reserved_after": c[3]})
    _tr["ctr"], _tr["ctr_i"] = c, n - 1


def _trace_step(t0: float, t1: float, tok: int) -> None:
    with _tr_lock:
        if not _tr:
            _trace_reset()
        t = _tr["t"]
        t.append(t0)
        _tr["d"].append(t1 - t0)
        _tr["k"].append(tok)
        n = len(t)
        if n % _TRACE_EVERY == 0:
            _trace_counters(n, t1)


def _install_trace() -> None:
    import functools
    import time

    from vllm.v1.worker import gpu_worker

    orig = gpu_worker.Worker.execute_model
    if getattr(orig, "_kvram_wrapped", False):
        return
    perf = time.perf_counter

    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        t0 = perf()
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        finally:
            _trace_step(t0, perf(), int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0))

    execute_model._kvram_wrapped = True  # type: ignore[attr-defined]
    gpu_worker.Worker.execute_model = execute_model
    logger.info("kvram_ext: KVRAM_STEP_TRACE installed (Worker.execute_model start / duration / tokens per step, allocator "
                "counters every %d steps)", _TRACE_EVERY)


if _DEDUP:
    _install_dedup()
if _CHECK:
    _install_check()
if _TRACE:
    _install_trace()
_MEMFRAC = _env_unit("KVRAM_MEM_FRACTION", _MEMFRAC_ENV, 0.0, 1.0)
_MEMGC = _env_unit("KVRAM_MEM_GC", _MEMGC_ENV, 0.0, 0.999)
if _MEMGC is not None and _MEMFRAC is None:
    logger.warning("kvram_ext: KVRAM_MEM_GC needs KVRAM_MEM_FRACTION (PyTorch GC acts only with a set fraction) -> ignored")
    _MEMGC = None
if _MEMFRAC is not None:
    _install_memcap(_MEMFRAC, _MEMGC)


# --------------------------------------------------------------------------------------------------------------------
# collective-RPC debug methods (prefix kvram_ to avoid clashes with the Worker class)
# --------------------------------------------------------------------------------------------------------------------
def _connector_worker(worker):
    conn = getattr(worker.model_runner, "kv_connector", None)
    inner = getattr(conn, "kv_connector", None)
    cw = getattr(inner, "connector_worker", None)
    return inner, cw


class KVRamWorkerExt:
    def kvram_mem_stats(self) -> dict:
        """CUDA caching-allocator counters of this rank (k8: OOM retries / cache releases under high MAX_SEQS)."""
        import torch

        s = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        keys = ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free", "reserved_bytes.all.current",
                "reserved_bytes.all.peak", "allocated_bytes.all.current", "allocated_bytes.all.peak")
        out = {k: int(s.get(k, 0)) for k in keys}
        out.update(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, cuda_free=int(free),
                   cuda_total=int(total), cap_bytes=int(_memcap.get("cap_bytes", 0)),
                   mem_fraction=float(_memcap.get("mem_fraction", 0.0)), gc_threshold=float(_memcap.get("gc_threshold", 0.0)))
        return out

    def kvram_info(self) -> dict:
        import torch

        mr = self.model_runner
        cfg = mr.kv_cache_config
        groups = []
        for g in cfg.kv_cache_groups:
            spec = g.kv_cache_spec
            per = getattr(spec, "kv_cache_specs", None)
            first = next(iter(per.values())) if per else spec
            groups.append({"block_size": spec.block_size, "layers": len(g.layer_names), "spec": type(first).__name__,
                           "uniform": per is not None, "page_bytes_sum": int(spec.page_size_bytes) if per else
                           int(spec.page_size_bytes) * len(g.layer_names),
                           "prefix_cacheable": bool(spec.prefix_cacheable), "replay_tokens": int(spec.prefix_replay_tokens),
                           "max_tp_shards": getattr(first, "max_tp_shards", None), "eagle": bool(g.is_eagle_group),
                           "layer_names_head": g.layer_names[:2]})
        out = {"rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               "num_blocks": int(cfg.num_blocks), "kv_tensor_bytes": [int(t.size) for t in cfg.kv_cache_tensors][:4],
               "n_kv_tensors": len(cfg.kv_cache_tensors), "kv_tp_replicas": int(getattr(cfg, "kv_tp_replicas", 1)),
               "prefix_cacheable_group_ids": list(cfg.prefix_cacheable_group_ids), "groups": groups,
               "layout": getattr(cfg, "kv_cache_layout", None), "dedup_patch": _DEDUP, "check_patch": _CHECK}
        inner, cw = _connector_worker(self)
        if cw is not None and cw.worker is not None:
            spec = cw.spec
            w = cw.worker
            sh = w._store_handler
            out["connector"] = {
                "replicated_layout": bool(getattr(spec, "replicated_layout", False)), "is_store_writer": cw._is_store_writer,
                "num_chunks": int(spec.num_chunks), "blocks_per_chunk": int(spec.blocks_per_chunk),
                "kv_bytes_per_chunk": int(spec.kv_bytes_per_chunk), "cpu_page_size_per_worker": int(spec.cpu_page_size_per_worker),
                "gpu_tensors": [(list(t.shape), list(t.stride())) for t in sh.src_tensors],
                "cpu_tensors": [(list(t.shape), list(t.stride())) for t in sh.dst_tensors],
                "region_bytes": int(w._mmap_region.total_size_bytes) if w._mmap_region is not None else None,
                "region_pinned": bool(w._mmap_region.is_pinned) if w._mmap_region is not None else None,
                "swap_fn": getattr(sh._swap_blocks_batch, "__name__", str(sh._swap_blocks_batch)),
            }
        return out

    def kvram_slots_compare(self, nrows: str = "0") -> dict | None:
        """Rank 0 only: compare every rank's host slot with rank 0's slot over chunk rows [0, nrows) (0 = all rows).
        Only meaningful without KVRAM_DEDUP (one slot per rank)."""
        import torch

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return None
        _, cw = _connector_worker(self)
        if cw is None or cw.worker is None or cw.worker._mmap_region is None:
            return {"error": "no shared region"}
        spec = cw.spec
        if spec.replicated_layout:
            return {"error": "replicated layout: single slot"}
        reg = cw.worker._mmap_region
        world = spec.config.parallel.world_size
        page = int(spec.cpu_page_size_per_worker)
        rows = reg.num_chunks if int(nrows) <= 0 else min(int(nrows), reg.num_chunks)
        base = reg.base_tensor.view(reg.num_chunks, reg._row_stride)
        nonzero = [0] * world
        mismatch = [0] * world
        step = max(1, (256 << 20) // reg._row_stride)
        for r0 in range(0, rows, step):
            blk = base[r0:min(rows, r0 + step)]
            s0 = blk[:, 0:page]
            nz0 = s0.ne(0).any(dim=1)
            nonzero[0] += int(nz0.sum())
            for k in range(1, world):
                sk = blk[:, k * page:(k + 1) * page]
                nonzero[k] += int(sk.ne(0).any(dim=1).sum())
                mismatch[k] += int(sk.ne(s0).any(dim=1).sum())
        return {"rows_compared": rows, "slot_bytes": page, "nonzero_rows_per_rank": nonzero,
                "rows_differing_from_rank0": mismatch}

    def kvram_trace_reset(self) -> dict:
        """KVRAM_STEP_TRACE: drop the recorded steps and take a fresh allocator-counter baseline."""
        import torch

        if not _TRACE:
            return {"installed": False}
        with _tr_lock:
            _trace_reset()
            return {"installed": True, "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    "counters": list(_tr["ctr"])}

    def kvram_trace_dump(self, path_prefix: str = "") -> dict:
        """KVRAM_STEP_TRACE: summary of the steps since the last reset (+ <path_prefix>.r<rank>.json with the raw lists if a
        prefix is given); then reset."""
        import json

        import torch

        if not _TRACE:
            return {"installed": False}
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        with _tr_lock:
            if not _tr:
                _trace_reset()
            t, d, k = _tr["t"], _tr["d"], _tr["k"]
            if t and (len(t) - 1) != _tr["ctr_i"]:
                _trace_counters(len(t), t[-1] + d[-1])  # counter changes after the last periodic read
            per = sorted(b - a for a, b in zip(t, t[1:]) if b - a < 10.0)

            def q(p):
                return round(per[min(len(per) - 1, int(p * len(per)))] * 1000, 2) if per else None

            summ = {"rank": rank, "steps": len(t), "period_ms_p10": q(0.10), "period_ms_p50": q(0.50),
                    "period_ms_p90": q(0.90), "period_ms_p99": q(0.99), "period_ms_max": round(per[-1] * 1000, 2) if per else None,
                    "events": len(_tr["ev"]), "retries": sum(e["retries"] for e in _tr["ev"]),
                    "frees": sum(e["frees"] for e in _tr["ev"])}
            if path_prefix:
                with open("%s.r%d.json" % (path_prefix, rank), "w") as f:
                    json.dump({"rank": rank, "every": _TRACE_EVERY, "t": [round(x, 6) for x in t],
                               "d": [round(x, 6) for x in d], "k": k, "ev": _tr["ev"], "reads": _tr["reads"],
                               "summary": summ}, f)
            _trace_reset()
        return summ

    def kvram_check(self) -> dict:
        with _check_lock:
            return dict(_check_totals) | {"installed": _CHECK, "active": _check_active[0]}

    def kvram_check_set(self, on: str = "1") -> bool:
        _check_active[0] = _CHECK and on == "1"
        return _check_active[0]

    def kvram_gpu_digest(self, block_ids_csv: str) -> list[str]:
        """sha1 (16 hex) of the packed GPU KV block rows the connector transfers, for the given block ids."""
        import hashlib

        _, cw = _connector_worker(self)
        t = cw.worker._store_handler.src_tensors[0]
        out = []
        for b in (int(x) for x in block_ids_csv.split(",") if x):
            out.append(hashlib.sha1(t[b].cpu().numpy().tobytes()).hexdigest()[:16])
        return out
