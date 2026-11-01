"""Checkpoint access, vLLM-0.32.0-exact TP-4 sharding rules and dequantization helpers.

``CheckpointReader`` is a dependency-free lazy safetensors reader: it parses the shard headers
itself and reads only the bytes it is asked for (whole tensors, dim-0 row ranges, or scattered
rows), so one layer -- or one expert, or a few Engram rows out of a 100 GB table -- can be loaded
without touching the rest. Tensors come back on the CPU in their exact checkpoint dtype:

  F8_E4M3 -> torch.float8_e4m3fn   F8_E8M0 -> torch.float8_e8m0fnu   I8 (packed e2m1) -> torch.int8
  BF16 -> torch.bfloat16           F32 -> torch.float32

Checkpoint facts: dense FP8 linears are e4m3 [N, K] with ue8m0 scales per
32x32 block, [N/32, K/32]; routed experts are I8 [N, K/2] (low nibble = even k) with ue8m0 per
1x32, [N, K/32]; wo_a is FP8 [8192, 4096] (model.py dequantizes it to bf16 at conversion).

``tp_slice`` reproduces exactly what each rank of stock vLLM 0.32.0 loads at TP=4 (rules below,
each with the vLLM source location it was read from). For block-scaled dense weights the slice
is returned in checkpoint block form [N_local/32, K_local/32]; stock vLLM's ModelOpt MXFP8 method
stores the same bytes row-expanded to [N_local, K_local/32] (``vllm_scale_layout``).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import torch

from .quant import as_u8, e4m3_to_float, e8m0_to_float, unpack_fp4

MODEL_DIR = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
TP = 4

_ST_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}
_ST_SIZES = {k: torch.empty((), dtype=v).element_size() for k, v in _ST_DTYPES.items()}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors dtype string
    shape: tuple[int, ...]
    path: str
    offset: int  # absolute byte offset of the tensor data in the file
    nbytes: int

    @property
    def torch_dtype(self) -> torch.dtype:
        return _ST_DTYPES[self.dtype]

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.shape[0] if self.shape and self.shape[0] else self.nbytes


def _read_into(path: str, offset: int, buf: torch.Tensor) -> None:
    mv = memoryview(buf.numpy()).cast("B")
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        pos = 0
        while pos < len(mv):
            n = f.readinto(mv[pos:])
            if not n:
                raise OSError(f"short read from {path} at {offset + pos}")
            pos += n


class LazyTensorDict(Mapping):
    """Read-only mapping of names (relative to ``prefix``) to tensors loaded on access."""

    def __init__(self, reader: "CheckpointReader", prefix: str, cache: bool = False):
        self.reader, self.prefix, self.cache = reader, prefix, cache
        self._keys = [n[len(prefix) :] for n in reader.names(prefix)]
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._cache:
            return self._cache[key]
        if key not in self._keys:
            raise KeyError(key)
        t = self.reader.get(self.prefix + key)
        if self.cache:
            self._cache[key] = t
        return t

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def info(self, key: str) -> TensorInfo:
        return self.reader.info(self.prefix + key)

    def tp(self, key: str, rank: int | None, tp: int = TP) -> torch.Tensor:
        return self.reader.get_tp(self.prefix + key, rank, tp)


class CheckpointReader:
    """Lazy reader over ``model.safetensors.index.json`` in ``model_dir``."""

    def __init__(self, model_dir: str | os.PathLike | None = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}
        self._lock = threading.Lock()

    # -- metadata ---------------------------------------------------------------------------
    @cached_property
    def hf_config(self) -> dict:
        with open(self.model_dir / "config.json") as f:
            return json.load(f)

    @cached_property
    def config(self) -> dict:
        """The text config (HF names), e.g. hidden_size, compress_ratios, engram_*."""
        cfg = self.hf_config
        return cfg.get("text_config", cfg)

    def _header(self, fname: str) -> tuple[int, dict]:
        with self._lock:
            if fname not in self._headers:
                path = self.model_dir / fname
                with open(path, "rb") as f:
                    n = int.from_bytes(f.read(8), "little")
                    hdr = json.loads(f.read(n))
                self._headers[fname] = (8 + n, hdr)
            return self._headers[fname]

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.weight_map if n.startswith(prefix))

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def info(self, name: str) -> TensorInfo:
        fname = self.weight_map[name]
        start, hdr = self._header(fname)
        h = hdr[name]
        b, e = h["data_offsets"]
        return TensorInfo(name, h["dtype"], tuple(h["shape"]), str(self.model_dir / fname), start + b, e - b)

    # -- reads ------------------------------------------------------------------------------
    def _raw(self, info: TensorInfo, byte_off: int, nbytes: int) -> torch.Tensor:
        buf = torch.empty(nbytes, dtype=torch.uint8)
        if nbytes:
            _read_into(info.path, info.offset + byte_off, buf)
        return buf

    def get(self, name: str) -> torch.Tensor:
        """Whole tensor, CPU, exact checkpoint dtype."""
        info = self.info(name)
        return self._raw(info, 0, info.nbytes).view(info.torch_dtype).reshape(info.shape)

    def get_rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Rows [start, stop) along dim 0, reading only those bytes."""
        info = self.info(name)
        start, stop = max(0, start), min(info.shape[0], stop)
        rb = info.row_bytes
        raw = self._raw(info, start * rb, (stop - start) * rb)
        return raw.view(info.torch_dtype).reshape((stop - start, *info.shape[1:]))

    def gather_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Arbitrary dim-0 rows (e.g. Engram table lookups); returns [*rows.shape, *row_shape]."""
        info = self.info(name)
        flat = rows.reshape(-1).tolist()
        rb = info.row_bytes
        out = torch.empty(len(flat), rb, dtype=torch.uint8)
        with open(info.path, "rb", buffering=0) as f:
            for i, r in enumerate(flat):
                if not 0 <= r < info.shape[0]:
                    raise IndexError(f"{name}: row {r} out of range {info.shape[0]}")
                f.seek(info.offset + r * rb)
                f.readinto(memoryview(out[i].numpy()))
        return out.view(info.torch_dtype).reshape(*rows.shape, *info.shape[1:])

    def get_tp(self, name: str, rank: int | None, tp: int = TP) -> torch.Tensor:
        """``tp_slice`` of a checkpoint tensor, reading only the needed rows for dim-0 splits."""
        if rank is None:
            return self.get(name)
        info = self.info(name)
        rule = tp_rule(name)
        if rule.kind == "rep":
            return self.get(name)
        dim, start, stop = tp_range(name, info.shape, rank, tp, cfg=self.config)
        if dim == 0:
            return self.get_rows(name, start, stop)
        return self.get(name).narrow(dim, start, stop - start).contiguous()

    def layer(self, i: int, cache: bool = False) -> LazyTensorDict:
        return LazyTensorDict(self, f"layers.{i}.", cache)

    def mtp(self, i: int, cache: bool = False) -> LazyTensorDict:
        return LazyTensorDict(self, f"mtp.{i}.", cache)


@lru_cache(maxsize=2)
def default_reader(model_dir: str | None = None) -> CheckpointReader:
    return CheckpointReader(model_dir)


# ===================================================================================== TP rules
#
# All line numbers refer to stock vLLM 0.32.0 (vllm/ in the wheel). "rep" = every rank loads the
# whole tensor; "split d" = rank r loads the r-th of TP equal chunks along dim d (for block-scaled
# weights the same chunking applies to the [N/32, K/32] scale; every split here is 32-aligned).


@dataclass(frozen=True)
class TPRule:
    kind: str  # "rep" | "split" | "engram_rows"
    dim: int | None
    source: str  # where in vLLM this was read from


_RULES: list[tuple[re.Pattern, TPRule]] = [
    (re.compile(r"^(embed|head)\.weight$"), TPRule("split", 0,
        "VocabParallelEmbedding/ParallelLMHead: nvidia/model.py:726, :1573; "
        "vocab_parallel_embedding.py:95-114 (129280 % 64 == 0 -> 32320 rows/rank)")),
    (re.compile(r"\.attn\.indexer\."), TPRule("rep", None,
        "DeepseekV4Indexer: wq_b/weights_proj/wk are ReplicatedLinear, attention.py:1313-1356")),
    (re.compile(r"\.attn\.compressor\."), TPRule("rep", None,
        "DeepseekCompressor.fused_wkv_wgate: MergedColumnParallelLinear(disable_tp=True), compressor.py:251-258")),
    (re.compile(r"\.attn\.attn_sink$"), TPRule("split", 0,
        "load_weights: loaded_weight[head_rank_start:head_rank_end], nvidia/model.py:1335-1342")),
    (re.compile(r"\.attn\.(wq_a|wkv)\.(weight|scale)$"), TPRule("rep", None,
        "fused_wqa_wkv: MergedColumnParallelLinear(disable_tp=True), attention.py:364-371; stacked at model.py:1246-1247")),
    (re.compile(r"\.attn\.wq_b\.(weight|scale)$"), TPRule("split", 0,
        "wq_b: ColumnParallelLinear (heads 16r..16r+15), attention.py:373-380")),
    (re.compile(r"\.attn\.wo_a\.(weight|scale)$"), TPRule("split", 0,
        "wo_a: ColumnParallelLinear, bmm over n_local_groups=2, attention.py:383-392")),
    (re.compile(r"\.attn\.wo_b\.(weight|scale)$"), TPRule("split", 1,
        "wo_b: RowParallelLinear (K-split), attention.py:396-403")),
    (re.compile(r"\.attn\.(q_norm|kv_norm)\.weight$"), TPRule("rep", None, "RMSNorm, attention.py:372, :382")),
    (re.compile(r"\.ffn\.experts\.\d+\.w[13]\.(weight|scale)$"), TPRule("split", 0,
        "FusedMoE _load_w13: loaded_per_rank = shape[0] // tp (576 rows), fused_moe/routed_experts.py:523-560")),
    (re.compile(r"\.ffn\.experts\.\d+\.w2\.(weight|scale)$"), TPRule("split", 1,
        "FusedMoE _load_w2: loaded_per_rank = shape[1] // tp (288 packed bytes / 18 scales), routed_experts.py:574-600")),
    (re.compile(r"\.ffn\.shared_experts\.w[13]\.(weight|scale)$"), TPRule("split", 0,
        "DeepseekV4MLP.gate_up_proj: MergedColumnParallelLinear, deepseek_v4/nvidia/model.py:155-167 "
        "(block-pad of v41 model.py:1369-1393 is a no-op at TP4: 72 blocks / 4)")),
    (re.compile(r"\.ffn\.shared_experts\.w2\.(weight|scale)$"), TPRule("split", 1,
        "DeepseekV4MLP.down_proj: RowParallelLinear, deepseek_v4/nvidia/model.py:168-176")),
    (re.compile(r"\.ffn\.gate\.(weight|bias|bias_vl)$"), TPRule("rep", None,
        "GateLinear(ReplicatedLinear), deepseek_v4/nvidia/model.py:724-773")),
    (re.compile(r"\.engram\.embed\.(weight|scale)$"), TPRule("engram_rows", 0,
        "ParallelEngramEmbedding: 6 of 24 hash heads per rank (complete prime buckets), "
        "common/engram.py:942-1035 + _engram_head_shard_weight_loader :604-619")),
    (re.compile(r"\.engram\.wkv\.(weight|scale)$"), TPRule("split", 0,
        "Engram.wkv: ColumnParallelLinear(gather_output=True), common/engram.py:1555-1564")),
    (re.compile(r"\.engram\.(q_weight|k_weight)$"), TPRule("rep", None, "common/engram.py:1565-1572")),
    (re.compile(r"^mtp\.\d+\.(main_proj|main_norm)\."), TPRule("rep", None,
        "DSpark main_proj: ReplicatedLinear, nvidia/dspark.py:103-110")),
    (re.compile(r"^mtp\.\d+\.markov_head\."), TPRule("rep", None,
        "DSparkMarkovHead: markov_w1 nn.Embedding, markov_w2 ParallelLMHead(disable_tp=True), "
        "model_executor/models/qwen3_dspark.py:56-80")),
    (re.compile(r"^mtp\.\d+\.confidence_head\."), TPRule("rep", None, "qwen3_dspark.py:176-183 (ReplicatedLinear)")),
    (re.compile(r"(^|\.)(hc_(attn|ffn)_(fn|base|scale)|attn_norm\.weight|ffn_norm\.weight|norm\.weight)$"),
        TPRule("rep", None, "plain parameters / RMSNorm, nvidia/model.py:328-452")),
]  # fmt: skip


def tp_rule(name: str) -> TPRule:
    name = name.removeprefix("model.")
    if name.startswith(("vision.", "aligner.", "image_")):
        raise ValueError(f"{name}: not loaded by the vLLM language model (model.py:1274-1277)")
    for pat, rule in _RULES:
        if pat.search(name):
            return rule
    raise KeyError(f"no TP rule for {name}")


# ----------------------------------------------------------------------------- Engram layout


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin (exact for n < 3.3e24)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class EngramLayout:
    """engram.py EngramLayout / vLLM common/engram.py:208-250 (primes drawn in order, never reused)."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][ngram-2][head]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    compressed_vocab_size: int
    pad_token_id: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def head_sizes(self, layer_idx: int) -> tuple[int, ...]:
        return tuple(p for per_ngram in self.primes[layer_idx] for p in per_ngram)

    def offsets(self, layer_idx: int) -> list[int]:
        sizes = self.head_sizes(layer_idx)
        out, acc = [], 0
        for s in sizes:
            out.append(acc)
            acc += s
        return out

    def rank_rows(self, layer_idx: int, rank: int, tp: int = TP) -> tuple[int, int]:
        """[vocab_start, vocab_end) rows of the table rank `rank` holds (engram.py:989-1006)."""
        sizes = self.head_sizes(layer_idx)
        part = -(-len(sizes) // tp)
        h0 = rank * part
        return sum(sizes[:h0]), sum(sizes[: h0 + part])


@lru_cache(maxsize=4)
def _engram_layout_cached(key: tuple) -> EngramLayout:
    layer_ids, num_emb, max_ng, n_heads, head_dim, vocab, cvs, pad = key
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ng - 1):
            sizes, cur = [], vocab - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not _is_prime(cand) or cand in seen:
                    cand += 1
                cur = cand
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return EngramLayout(layer_ids, num_emb, tuple(primes), max_ng, n_heads, head_dim, cvs, pad)


def engram_layout(cfg: Mapping) -> EngramLayout:
    """Layout from the HF text config (engram_* keys)."""
    key = (
        tuple(cfg["engram_layer_ids"]),
        tuple(cfg["engram_num_embeddings"]),
        int(cfg["engram_max_ngram_size"]),
        int(cfg["engram_n_heads"]),
        int(cfg["engram_head_dim"]),
        int(cfg["engram_vocab_size"]),
        int(cfg["engram_compressed_vocab_size"]),
        int(cfg.get("engram_pad_token_id", cfg.get("engram_pad_id", 2))),
    )
    return _engram_layout_cached(key)


def _layer_index(name: str) -> int:
    m = re.match(r"^(?:model\.)?layers\.(\d+)\.", name)
    if not m:
        raise ValueError(f"cannot find layer index in {name}")
    return int(m.group(1))


def tp_range(name: str, shape: tuple[int, ...], rank: int, tp: int = TP, cfg: Mapping | None = None):
    """(dim, start, stop) of rank `rank`'s slice, or None for replicated tensors."""
    rule = tp_rule(name)
    if rule.kind == "rep":
        return None
    if rule.kind == "engram_rows":
        cfg = cfg if cfg is not None else default_reader().config
        lay = engram_layout(cfg)
        start, stop = lay.rank_rows(lay.layer_ids.index(_layer_index(name)), rank, tp)
        return 0, start, stop
    n = shape[rule.dim]
    if n % tp:
        raise ValueError(f"{name}: dim {rule.dim} of size {n} not divisible by tp={tp}")
    part = n // tp
    return rule.dim, rank * part, (rank + 1) * part


def tp_shape(name: str, shape: tuple[int, ...], rank: int, tp: int = TP, cfg: Mapping | None = None) -> tuple[int, ...]:
    r = tp_range(name, shape, rank, tp, cfg)
    if r is None:
        return tuple(shape)
    dim, a, b = r
    out = list(shape)
    out[dim] = b - a
    return tuple(out)


def tp_slice(name: str, tensor: torch.Tensor, rank: int, tp: int = TP, cfg: Mapping | None = None) -> torch.Tensor:
    """The part of checkpoint tensor `name` that TP rank `rank` of stock vLLM 0.32.0 loads."""
    r = tp_range(name, tuple(tensor.shape), rank, tp, cfg)
    if r is None:
        return tensor
    dim, a, b = r
    return tensor.narrow(dim, a, b - a).contiguous()


def vllm_scale_layout(scale: torch.Tensor, block_rows: int = 32) -> torch.Tensor:
    """Checkpoint block scale [N/32, K/32] -> stock vLLM's MXFP8 weight_scale param [N, K/32] uint8
    (KMxfp8Static.get_scale_weight_loader repeat_interleave, quantization/modelopt.py:2276-2291)."""
    return as_u8(scale).repeat_interleave(block_rows, dim=0)


# ===================================================================================== dequant


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] x ue8m0 scale -> float32 [N, K].

    `s` may be the checkpoint block form [ceil(N/32), ceil(K/32)] or vLLM's row-expanded form
    [N, K/32]; the block size is inferred from the shapes.
    """
    n, k = w.shape[-2:]
    sf = e8m0_to_float(s)
    bn = -(-n // s.shape[-2])
    bk = -(-k // s.shape[-1])
    sf = sf.repeat_interleave(bn, dim=-2)[..., :n, :].repeat_interleave(bk, dim=-1)[..., :k]
    return e4m3_to_float(w) * sf


def dequant_fp4(w_packed: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Packed e2m1 [N, K/2] (I8/U8, low nibble = even k) x ue8m0 [N, K/32] -> float32 [N, K]."""
    w = unpack_fp4(w_packed)
    return w * e8m0_to_float(s).repeat_interleave(block, dim=-1)


def dequant_any(reader: CheckpointReader, name: str, rank: int | None = None, tp: int = TP,
                device=None) -> torch.Tensor:
    """Load `name` (weight) with its `.scale` sibling for `rank` and return float32 on `device`."""
    w = reader.get_tp(name, rank, tp)
    sname = name.removesuffix(".weight") + ".scale"
    if w.dtype == torch.float8_e4m3fn:
        s = reader.get_tp(sname, rank, tp)
        return dequant_fp8(w.to(device), s.to(device))
    if w.dtype == torch.int8 and sname in reader:
        s = reader.get_tp(sname, rank, tp)
        return dequant_fp4(w.to(device), s.to(device))
    return w.to(device).float()
