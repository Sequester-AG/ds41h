"""Engram storage hooks (opt-in, default OFF; default path = stock vLLM's head-sharded GPU tables):

* DS41H_ENGRAM_NVME=1 (ds41h/engram_nvme.py + kernels/engram_nvme.cu): the tables stay on the NVMe
  drive; each step reads the rows it needs through io_uring (issue at prepare_embeddings, wait right before layers
  1 / 14). No table bytes on the GPUs or in host RAM (only the page cache in buffered mode). Same rows, same
  dequant, same downstream stock code -> lossless by construction. See patch_engram_nvme_module below.
* DS41H_ENGRAM_HOST=1 (below): ONE shared pinned host copy (ds41h/engram_host.py) -- needs ~200 GiB free in /dev/shm.
If both are set, NVME wins (logged).

ENGRAM_HOST hook: Engram n-gram tables in ONE shared pinned host copy (ds41h/engram_host.py) instead of GPU HBM.

Opt-in: DS41H=1 DS41H_ENGRAM_HOST=1 (keep the server's stock `--engram-config '{"cpu_offload":false,...}'`, as
scripts/serve.sh passes it: this hook replaces the table storage itself). Frees ~47 GiB of HBM per GPU (TP4) for KV
cache. Numerics: bit-identical lookup rows (stock dequant + stock head-shard ownership), everything downstream (wkv,
gate, mHC) is the stock code on identical inputs -> lossless by construction.

Patched symbols (vllm/models/deepseek_v41/common/engram.py, stock vLLM 0.32.0):
  * Engram._create_embedding (:1577)  -> passes (layout, layer index) to the table allocation below
  * ParallelEngramEmbedding._allocate_weights (:1035) -> this rank's param views INTO the shared host mapping (rows
    [vocab_start, vocab_end) of the full table; no HBM), the mapping cudaHostRegister'ed in this process, and a weight
    loader that runs the stock loader (narrow + copy_) unless the shard is already complete (KEEP=1 restart)
  * ParallelEngramEmbedding.__init__ (:953) -> + dummy_weight_value attrs (like stock cpu_offload)
  * ParallelEngramEmbedding.lookup / forward -> refuse for host tables (no device storage; not used by the model)
  * Engram._init_staging (:1591)      -> bf16 staging [max_num_batched_tokens, 24, 256] (ALL hash columns), a shared
    high-priority side stream and a per-layer event
  * Engram.prepare_embeddings (:1606) -> gather from host memory on the side stream, launched before the decoder
    layers (model.py:980-991), overlapped with layer 0 (resp. layers 0-13): steps with <= DS41H_ENGRAM_HOST_LOCAL_MAX
    tokens (decode) gather ALL 24 columns on every rank; larger steps (prefill) gather this rank's head shard only
  * Engram.embed (:1654)              -> join the side stream; decode: return the complete rows (no TP all-gather);
    prefill: the stock TP all-gather of the head shards (same call as stock)
  * opt-in DS41H_ENGRAM_HOST_EARLY_WKV=1: Engram.wkv.forward (per instance) returns the wkv product computed on the side
    stream right after the decode-mode gather (same quant_method.apply + gather_output all-gather)
Ineligible configurations (Engram DP groups / embedding_across_dp, missing CUDA) keep the stock storage and path.
All of it is CUDA-graph capturable (fork/join on events, fixed buffers, no host sync on the replay path).
"""

from __future__ import annotations

import contextvars
import logging
import os
import time

from ds41h import flags
from ds41h.patches import register

ENGRAM_HOST = flags.ENABLED and os.environ.get("DS41H_ENGRAM_HOST", "0") not in ("0", "", "false", "False")
ENGRAM_NVME = flags.ENABLED and os.environ.get("DS41H_ENGRAM_NVME", "0") not in ("0", "", "false", "False")
# opt-in lookahead for the NVMe reader (ds41h/engram_nvme.py DS41H_ENGRAM_NVME_PREFETCH; cache warm-up only, no numerics)
NVME_PREFETCH = ENGRAM_NVME and os.environ.get("DS41H_ENGRAM_NVME_PREFETCH", "0") not in ("0", "", "false", "False")
INLINE = os.environ.get("DS41H_ENGRAM_HOST_INLINE", "0") not in ("0", "", "false")
PRIORITY = int(os.environ.get("DS41H_ENGRAM_HOST_PRIORITY", "-1"))  # side stream priority (lower = higher)
# Steps with <= LOCAL_MAX tokens (decode, small batches): every rank gathers ALL hash columns from host memory (no TP
# all-gather). Larger steps (prefill): every rank gathers only its own head shard (1/TP of the PCIe bytes, hidden behind
# layer 0) and the stock TP all-gather assembles the rows, exactly like the stock path. Random 264-B row reads run at
# ~8 GB/s per GPU through the IOMMU (4 KiB pages, DMA-FQ; docs/ENGRAM.md), so a full 24-column gather
# would outlast layer 0 above ~200 tokens. 0 = always shard, a huge value = always local.
LOCAL_MAX = int(os.environ.get("DS41H_ENGRAM_HOST_LOCAL_MAX", "256"))
# Opt-in (default 0, to be measured on the RAM day): in local (decode) mode also run the Engram wkv GEMM
# (ColumnParallelLinear quant_method.apply, ~25 us per Engram layer at M=6 in b350_best) on the side stream right after
# the gather -- it depends only on the looked-up rows, not on the residual stream -- so only wkv's gather_output
# all-gather + the gate kernel stay on the critical path (~-50 us/step). Same apply() call on the same inputs -> same bits.
EARLY_WKV = os.environ.get("DS41H_ENGRAM_HOST_EARLY_WKV", "0") not in ("0", "", "false")

logger = logging.getLogger("ds41h")
_CTX: contextvars.ContextVar = contextvars.ContextVar("ds41h_engram_ctx", default=None)
_STREAMS: dict = {}


def side_stream(device):
    import torch

    key = device.index
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device=device, priority=PRIORITY)
    return _STREAMS[key]


class HostLayer:
    """Per-process handle of one Engram layer's shared host table: ownership ranges + gather."""

    def __init__(self, table, lo: list[int], hi: list[int], layer_id: int, head_start: int = 0,
                 part: int | None = None) -> None:
        self.table = table
        self.lo = lo
        self.hi = hi
        self.layer_id = layer_id
        self.head_start = head_start  # this rank's head shard [head_start, head_start + part) (stock TP sharding)
        self.part = len(lo) if part is None else part

    def wait_ready(self) -> None:
        self.table.wait_ready()

    def gather(self, ids, out) -> None:
        """All hash columns: out [T, n_cols, 256] = stock lookup + TP all-gather."""
        self.table.gather(ids, out, self.lo, self.hi)

    def gather_shard(self, ids, out) -> None:
        """This rank's head shard only: out [T, part, 256] = stock lookup on this rank (before the all-gather)."""
        h0, h1 = self.head_start, self.head_start + self.part
        self.table.gather(ids[:, h0:h1], out, self.lo[h0:h1], self.hi[h0:h1])


def _host_of(module):
    return getattr(module, "_ds41h_host", None)


def patch_engram_module(module) -> None:
    import torch

    from ds41h import engram_host as eh
    from ds41h.patch_core import mark_active

    PEE = module.ParallelEngramEmbedding
    Engram = module.Engram
    stock_loader = module._engram_head_shard_weight_loader
    if getattr(PEE, "_ds41h_engram_patched", False):
        return

    # ------------------------------------------------------------------------------- table allocation + loading
    base_create = Engram._create_embedding

    def _create_embedding(self, layout, layer_hash_index):
        token = _CTX.set((layout, layer_hash_index))
        try:
            return base_create(self, layout, layer_hash_index)
        finally:
            _CTX.reset(token)

    base_alloc = PEE._allocate_weights

    def _allocate_weights(self):
        ctx = _CTX.get()
        if ctx is None or self.dp_size != 1 or self.dp_shared_memory or self.dim != eh.DIM or self.block_size != eh.QUANT:
            if ctx is not None:
                logger.warning("ds41h engram_host: ineligible table (dp_size=%s dim=%s block=%s): stock storage",
                               self.dp_size, self.dim, self.block_size)
            return base_alloc(self)
        from vllm.config import get_current_vllm_config
        from vllm.distributed import get_tensor_model_parallel_rank

        layout, idx = ctx
        layer_id = int(layout.layer_ids[idx])
        head_sizes = [int(p) for order in layout.primes[idx] for p in order]
        rank = get_tensor_model_parallel_rank()
        vcfg = get_current_vllm_config()
        model_path = str(getattr(vcfg.model_config, "model", "") or "")
        load_format = str(getattr(getattr(vcfg, "load_config", None), "load_format", ""))
        key = eh.checkpoint_key(model_path, layer_id, self.num_embeddings, load_format)
        lay = eh.default_layout()
        path = eh.table_path(key, layer_id, lay)
        t0 = time.perf_counter()
        table = eh.HostTable(path, key, self.num_embeddings, n_shards=self.tp_size, layout=lay)
        t_pop = table.populate() if table.created else 0.0
        t_reg = table.register()
        dummy = "dummy" in load_format.lower()
        if dummy:  # dummy weights never reach a loader: nothing to wait for
            for s in range(self.tp_size):
                table.set_state(s, "weight", eh.DONE)
                table.set_state(s, "scale", eh.DONE)
        lo, hi = eh.head_ranges(head_sizes, self.tp_size)
        assert (lo[self.head_start], hi[self.head_start]) == (self.vocab_start_idx, self.vocab_end_idx)
        object.__setattr__(self, "_ds41h_host", HostLayer(table, lo, hi, layer_id, self.head_start,
                                                          self.part_n_hash_cols))
        cached = table.shard_done(rank)

        def weight_loader(param, loaded_weight, _table=table, _rank=rank):
            kind = "weight" if param.dtype == torch.float8_e4m3fn else "scale"
            if _table.state(_rank, kind) == eh.DONE:
                return  # shard already in the shared table (KEEP=1 restart or a concurrent server)
            _table.set_state(_rank, kind, eh.LOADING)
            stock_loader(param, loaded_weight)
            _table.set_state(_rank, kind, eh.DONE)

        self._weight_loader = weight_loader
        logger.info(
            "ds41h engram_host: layer %d rank %d: %s %.1f GiB (%s layout, %s%s), rows [%d, %d) of %d; populate %.1f s, "
            "cudaHostRegister %.1f s, total %.1f s", layer_id, rank, path, table.geom.total / 2**30, lay,
            "created" if table.created else "opened", ", shard cached" if cached else "", self.vocab_start_idx,
            self.vocab_end_idx, self.num_embeddings, t_pop, t_reg, time.perf_counter() - t0)
        return table.views(self.vocab_start_idx, self.vocab_end_idx)

    base_init = PEE.__init__

    def __init__(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        if _host_of(self) is not None and not self.cpu_offload:
            from vllm.model_executor.utils import set_weight_attrs

            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})

    base_lookup = PEE.lookup
    base_forward = PEE.forward

    def lookup(self, indices, out, background=False):
        if _host_of(self) is not None:
            raise RuntimeError("ds41h engram_host: host tables are read through Engram.prepare_embeddings/embed")
        return base_lookup(self, indices, out, background)

    def forward(self, indices):
        if _host_of(self) is not None:
            raise RuntimeError("ds41h engram_host: host tables are read through Engram.prepare_embeddings/embed")
        return base_forward(self, indices)

    # ------------------------------------------------------------------------------- staging + prefetch + join
    base_staging = Engram._init_staging

    def _init_staging(self, max_tokens, head_dim):
        if _host_of(self.embed_tokens) is None:
            return base_staging(self, max_tokens, head_dim)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.staged_rows = torch.empty(max_tokens, self.embed_tokens.n_hash_cols, head_dim, dtype=torch.bfloat16,
                                       device=dev)
        self._prefetch_stream = None if INLINE else side_stream(dev)
        self._prefetch_done = torch.cuda.Event()
        self._ds41h_early = None
        if EARLY_WKV:
            _install_early_wkv(self)

    def _install_early_wkv(self) -> None:
        """Per-instance wkv.forward: returns the side-stream product for exactly the staged rows of this step (same
        quant_method.apply + the stock gather_output all-gather = ColumnParallelLinear.forward), else stock forward."""
        import weakref

        wkv = self.wkv
        if getattr(wkv, "_ds41h_early_installed", False) or wkv.bias is not None or wkv.skip_bias_add:
            return
        base_fwd = wkv.forward
        owner = weakref.ref(self)

        def forward(input_):
            eng = owner()
            early = getattr(eng, "_ds41h_early", None) if eng is not None else None
            if early is not None and early[0] == input_.data_ptr() and early[1] == tuple(input_.shape):
                eng._ds41h_early = None
                out = early[2]
                if wkv.gather_output and wkv.tp_size > 1:
                    from vllm.distributed import tensor_model_parallel_all_gather

                    out = tensor_model_parallel_all_gather(out)
                return out if not wkv.return_bias else (out, None)
            return base_fwd(input_)

        wkv.forward = forward
        wkv._ds41h_early_installed = True

    base_prepare = Engram.prepare_embeddings

    def _sharded(self, n: int) -> bool:
        table = self.embed_tokens
        return (n > LOCAL_MAX and table.tp_size > 1 and not self.use_sequence_parallel
                and table.part_n_hash_cols * table.tp_size == table.n_hash_cols)

    def _shard_rows(self, n: int):
        table = self.embed_tokens
        return self.staged_rows.view(-1)[: n * table.part_n_hash_cols * table.dim].view(n, table.part_n_hash_cols,
                                                                                         table.dim)

    def prepare_embeddings(self, hash_ids):
        host = _host_of(self.embed_tokens)
        if host is None:
            return base_prepare(self, hash_ids)
        n = hash_ids.shape[0]
        assert self.staged_rows.shape[0] >= n, "engram staging buffer too small"
        host.wait_ready()  # host-side check (first call only), no CUDA work
        sharded = _sharded(self, n)  # same decision on every TP rank (same n): the all-gather stays collective-safe
        rows = _shard_rows(self, n) if sharded else self.staged_rows[:n]
        fn = host.gather_shard if sharded else host.gather
        cur = torch.cuda.current_stream()
        st = self._prefetch_stream
        if st is None:
            fn(hash_ids, rows)
            self._prefetch_done.record(cur)
            return
        st.wait_stream(cur)
        early = None
        with torch.cuda.stream(st):
            fn(hash_ids, rows)
            if EARLY_WKV and not sharded and getattr(self.wkv, "_ds41h_early_installed", False):
                x = rows.flatten(-2)
                early = (x.data_ptr(), tuple(x.shape), self.wkv.quant_method.apply(self.wkv, x, None))
        hash_ids.record_stream(st)
        if early is not None:
            early[2].record_stream(cur)  # consumed on the main stream (Engram.forward -> wkv) after the event join
        self._ds41h_early = early
        self._prefetch_done.record(st)

    base_embed = Engram.embed

    def embed(self, hash_ids):
        table = self.embed_tokens
        if _host_of(table) is None:
            return base_embed(self, hash_ids)
        torch.cuda.current_stream().wait_event(self._prefetch_done)
        n = hash_ids.shape[0]
        if _sharded(self, n):  # stock embed: TP all-gather of the head shards (engram.py:1673-1674)
            from vllm.distributed import tensor_model_parallel_all_gather

            return tensor_model_parallel_all_gather(_shard_rows(self, n), dim=1)[:, : table.n_hash_cols]
        rows = self.staged_rows[:n]
        if self.use_sequence_parallel and table.tp_size > 1:
            # stock: all-gather over tokens + _engram_select_rows of this rank's chunk (zeros past the last token)
            from vllm.distributed import get_tensor_model_parallel_rank

            chunk = -(-n // table.tp_size)
            start = get_tensor_model_parallel_rank() * chunk
            out = rows.new_zeros((chunk, *rows.shape[1:]))
            m = max(0, min(chunk, n - start))
            if m:
                out[:m].copy_(rows[start : start + m])
            return out
        return rows

    Engram._create_embedding = _create_embedding
    PEE._allocate_weights = _allocate_weights
    PEE.__init__ = __init__
    PEE.lookup = lookup
    PEE.forward = forward
    Engram._init_staging = _init_staging
    Engram.prepare_embeddings = prepare_embeddings
    Engram.embed = embed
    PEE._ds41h_engram_patched = True
    mark_active("engram_host", "common.engram.{ParallelEngramEmbedding storage,Engram.prepare_embeddings/embed}")


# ===================================================================================================== ENGRAM_NVME
class NvmeLayer:
    """Per-process handle of one Engram layer read from its NVMe table file (ds41h/engram_nvme.py)."""

    def __init__(self, table, layer_id: int, vstart: int, vend: int, head_start: int, part: int, n_cols: int,
                 rank: int, layer_index: int = -1) -> None:
        self.table = table
        self.layer_id = layer_id
        self.layer_index = layer_index  # layer_hash_index (row of the stock hash parameters)
        self.vstart, self.vend = vstart, vend  # this rank's rows (stock head shard)
        self.head_start, self.part, self.n_cols = head_start, part, n_cols
        self.rank = rank
        self.channel = None  # engram_nvme.Channel, created with the staging buffer (Engram._init_staging)
        self.pending = None  # tokens of the issued, not yet consumed request (host-side protocol check)


def _nvme_of(module):
    return getattr(module, "_ds41h_nvme", None)


def patch_engram_nvme_module(module) -> None:
    """DS41H_ENGRAM_NVME=1: stock Engram with the table rows read from NVMe at lookup time.

    Patched (vllm/models/deepseek_v41/common/engram.py, stock vLLM 0.32.0):
      * Engram._create_embedding (:1577)   -> passes (layout, layer index) to the allocation below
      * ParallelEngramEmbedding._allocate_weights (:1035) -> opens / validates (builds once) the layer's re-laid-out
        table file; returns EMPTY [0, 256] / [0, 8] CPU placeholders (nothing on the GPU, nothing in RAM) and a no-op
        weight loader (the checkpoint tensors are never read at load time: lazy safetensors views stay untouched)
      * ParallelEngramEmbedding.lookup / forward -> refuse (the rows only exist on disk)
      * Engram._init_staging (:1591)       -> stock staging [max_tokens, part, 256] bf16 + this layer's I/O channel
        (pinned ids / src / rows buffers sized max_tokens * part) + a device state tensor (generation counter, stats)
      * Engram.prepare_embeddings (:1606)  -> issue kernel (stream-ordered after stock's hash kernel, before layer 0):
        this rank's head-shard ids -> pinned buffer, generation -> the poller thread starts reading
      * Engram._ready_rows (:1645)         -> wait + dequant kernel into the stock staging rows right before the layer
        uses them; the stock embed (TP all-gather / SP select) and everything after is unchanged
    Ineligible configurations (stock cpu_offload, Engram DP groups, other geometry) keep the stock storage and path.
    Graph-capturable: two plain kernels, fixed pointers, generation counter in device memory, no host sync."""
    import torch

    from ds41h import engram_nvme as en
    from ds41h.patch_core import mark_active

    PEE = module.ParallelEngramEmbedding
    Engram = module.Engram
    if getattr(PEE, "_ds41h_engram_patched", False):
        return

    base_create = Engram._create_embedding

    def _create_embedding(self, layout, layer_hash_index):
        token = _CTX.set((layout, layer_hash_index))
        try:
            return base_create(self, layout, layer_hash_index)
        finally:
            _CTX.reset(token)

    base_alloc = PEE._allocate_weights

    def _allocate_weights(self):
        ctx = _CTX.get()
        if (ctx is None or self.cpu_offload or self.dp_size != 1 or self.dp_shared_memory or self.dim != en.DIM
                or self.block_size != 32):
            if ctx is not None:
                logger.warning("ds41h engram_nvme: ineligible table (cpu_offload=%s dp_size=%s dim=%s block=%s): stock "
                               "storage", self.cpu_offload, self.dp_size, self.dim, self.block_size)
            return base_alloc(self)
        from vllm.config import get_current_vllm_config
        from vllm.distributed import get_tensor_model_parallel_rank

        layout, idx = ctx
        layer_id = int(layout.layer_ids[idx])
        rank = get_tensor_model_parallel_rank()
        mc = getattr(get_current_vllm_config(), "model_config", None)
        model_path = str(getattr(mc, "model_weights", None) or getattr(mc, "model", None) or "")
        t0 = time.perf_counter()
        table = en.open_table(model_path, layer_id, rows=self.num_embeddings)
        object.__setattr__(self, "_ds41h_nvme", NvmeLayer(table, layer_id, self.vocab_start_idx, self.vocab_end_idx,
                                                          self.head_start, self.part_n_hash_cols, self.n_hash_cols, rank,
                                                          int(idx)))

        def weight_loader(param, loaded_weight):  # rows are read from the table file at lookup time
            return

        self._weight_loader = weight_loader
        logger.info("ds41h engram_nvme: layer %d rank %d: rows [%d, %d) of %d (%.1f GB of this rank's shard) read from %s "
                    "at lookup time; no GPU / host table (opened in %.1f s)", layer_id, rank, self.vocab_start_idx,
                    self.vocab_end_idx, self.num_embeddings, (self.vocab_end_idx - self.vocab_start_idx) * en.PAYLOAD / 1e9,
                    table.path, time.perf_counter() - t0)
        return (torch.empty(0, self.dim, dtype=torch.float8_e4m3fn, device="cpu"),
                torch.empty(0, self.dim // self.block_size, dtype=torch.uint8, device="cpu"))

    base_lookup = PEE.lookup
    base_forward = PEE.forward

    def lookup(self, indices, out, background=False):
        if _nvme_of(self) is not None:
            raise RuntimeError("ds41h engram_nvme: NVMe tables are read through Engram.prepare_embeddings/embed")
        return base_lookup(self, indices, out, background)

    def forward(self, indices):
        if _nvme_of(self) is not None:
            raise RuntimeError("ds41h engram_nvme: NVMe tables are read through Engram.prepare_embeddings/embed")
        return base_forward(self, indices)

    base_staging = Engram._init_staging

    def _init_staging(self, max_tokens, head_dim):
        base_staging(self, max_tokens, head_dim)  # stock: staged_rows [max_tokens, part, 256] bf16, no prefetch stream
        nv = _nvme_of(self.embed_tokens)
        if nv is None:
            return
        rd = en.reader(nv.rank)
        nv.channel = rd.add_channel(nv.table, nv.vstart, nv.vend, max_tokens * nv.part, nv.layer_index, nv.head_start,
                                    nv.part)
        nv.channel.dstate = torch.zeros(16, dtype=torch.int64, device=self.staged_rows.device)
        nv.timeout_ms = rd.timeout_ms

    base_prepare = Engram.prepare_embeddings

    def prepare_embeddings(self, hash_ids):
        nv = _nvme_of(self.embed_tokens)
        if nv is None:
            return base_prepare(self, hash_ids)
        n = hash_ids.shape[0]
        assert self.staged_rows.shape[0] >= n, "engram staging buffer too small"
        if n == 0:
            return
        rd = en.reader(nv.rank)
        if not rd.started:  # host-side only (ring + thread), no CUDA work: also safe inside a graph capture
            note = rd.start()
            logger.info("ds41h engram_nvme: rank %d reader started (%s, mode %s, auto_max %d)", nv.rank, note, rd.mode,
                        rd.auto_max)
        en.ext().issue(nv.channel.handle, hash_ids, nv.head_start, nv.part, nv.n_cols, nv.channel.dstate)
        nv.pending = n

    base_ready = Engram._ready_rows

    def _ready_rows(self, num_tokens):
        nv = _nvme_of(self.embed_tokens)
        if nv is None:
            return base_ready(self, num_tokens)
        rows = self.staged_rows[:num_tokens]
        if num_tokens:
            if nv.pending != num_tokens:  # the rows of this forward were never requested: refuse (stale rows otherwise)
                raise RuntimeError(f"ds41h engram_nvme: layer {nv.layer_id}: rows of {num_tokens} tokens consumed, but the "
                                   f"pending request has {nv.pending} (prepare_embeddings not called for this forward?)")
            en.ext().wait_dequant(nv.channel.handle, nv.channel.dstate, rows, num_tokens * nv.part, nv.timeout_ms)
        nv.pending = None
        return rows

    Engram._create_embedding = _create_embedding
    PEE._allocate_weights = _allocate_weights
    PEE.lookup = lookup
    PEE.forward = forward
    Engram._init_staging = _init_staging
    Engram.prepare_embeddings = prepare_embeddings
    Engram._ready_rows = _ready_rows
    PEE._ds41h_engram_patched = True
    mark_active("engram_nvme", "common.engram.{ParallelEngramEmbedding storage,Engram.prepare_embeddings/_ready_rows}")


def _find_hash_state(runner):
    """The model's stock NgramHashState (token map / multipliers / primes / offsets buffers), through wrappers."""
    seen, stack = set(), [getattr(runner, "model", None)]
    while stack:
        m = stack.pop()
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        hs = getattr(m, "engram_hash", None)
        if hs is not None and hasattr(hs, "token_map") and hasattr(hs, "multipliers"):
            return hs
        for name in ("model", "module", "runnable", "language_model", "_orig_mod"):
            stack.append(getattr(m, name, None))
    return None


def _prefetch_step(runner, so, state: dict) -> None:
    """Queue the rows of prompt tokens beyond this step's chunk (up to the lookahead horizon) of every request still in
    prefill; extend the window as the request's chunks get scheduled."""
    from ds41h import engram_nvme as en

    rd = en._READER
    if rd is None or not rd.prefetch_on or not rd.channels:
        return
    if not rd.hash_set:
        hs = _find_hash_state(runner)
        if hs is None:
            rd.prefetch_on = False
            logger.warning("ds41h engram_nvme: prefetch off (no NgramHashState found on the runner's model)")
            return
        import numpy as np

        rd.set_hash(en.HashParams(hs.token_map.cpu().numpy().astype(np.int64), int(hs.pad_id),
                                  hs.multipliers.cpu().numpy().astype(np.int64), hs.primes.cpu().numpy().astype(np.int64),
                                  hs.offsets.cpu().numpy().astype(np.int64), tuple(hs.layout.layer_ids)))
        logger.info("ds41h engram_nvme: prefetch on (horizon %d tokens, %d reads in flight)", rd.pf_tokens, rd.pf_inflight)
    for rid in so.finished_req_ids:
        state.pop(rid, None)
    sched = so.num_scheduled_tokens
    cur = {}
    for nr in so.scheduled_new_reqs:
        toks = nr.prompt_token_ids
        if not toks:
            continue
        end = nr.num_computed_tokens + sched.get(nr.req_id, 0)
        if end < len(toks):
            state[nr.req_id] = [toks, end]
            cur[nr.req_id] = end
    cr = so.scheduled_cached_reqs
    for rid, ncomp in zip(cr.req_ids, cr.num_computed_tokens):
        if rid in state:
            cur[rid] = ncomp + sched.get(rid, 0)
    for rid, end in cur.items():
        toks, done = state[rid]
        lo, hi = max(done, end), min(len(toks), end + rd.pf_tokens)
        if hi > lo:
            rd.prefetch(toks, lo, hi)
            state[rid][1] = hi
        if hi >= len(toks):
            state.pop(rid, None)


def patch_runner_prefetch(module) -> None:
    """DS41H_ENGRAM_NVME_PREFETCH=1: V2 GPUModelRunner.add_requests (vllm/v1/worker/gpu/model_runner.py:1181) also queues
    the NVMe reader's lookahead for the admitted requests (see _prefetch_step). Best effort: any error disables it."""
    GMR = getattr(module, "GPUModelRunner", None)
    if GMR is None or getattr(GMR, "_ds41h_pf_patched", False):
        return
    from ds41h.patch_core import mark_active

    base_add = GMR.add_requests
    state: dict = {}

    def add_requests(self, scheduler_output):
        base_add(self, scheduler_output)
        try:
            _prefetch_step(self, scheduler_output, state)
        except Exception:  # noqa: BLE001  (prefetch only warms a cache: never break serving)
            logger.exception("ds41h engram_nvme: prefetch disabled after an error")
            GMR.add_requests = base_add

    GMR.add_requests = add_requests
    GMR._ds41h_pf_patched = True
    mark_active("engram_nvme", "v1.worker.gpu.model_runner.GPUModelRunner.add_requests (prefetch)")


if ENGRAM_NVME:
    if ENGRAM_HOST:
        logger.warning("ds41h: DS41H_ENGRAM_NVME=1 and DS41H_ENGRAM_HOST=1 both set: using ENGRAM_NVME")
    register("vllm.models.deepseek_v41.common.engram")(patch_engram_nvme_module)
    if NVME_PREFETCH:
        register("vllm.v1.worker.gpu.model_runner")(patch_runner_prefetch)
elif ENGRAM_HOST:
    register("vllm.models.deepseek_v41.common.engram")(patch_engram_module)
