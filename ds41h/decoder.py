"""ds41h decoder: persistent post-attention decoder-layer kernel for decode (sm90a, TP4).

Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/decoder.py). Kernel: ds41h/kernels/decoder.cu.

One launch replaces everything stock vLLM 0.32.0 runs after the attention core of a
``DeepseekV4DecoderLayer`` (models/deepseek_v41/nvidia/model.py:591-625): inverse RoPE + wo_a +
wo_b + all-reduce #1 + FFN mHC (post/pre, Sinkhorn) + RMSNorm + gate + top-6 + shared and routed
experts + combine + all-reduce #2, for M <= ``MAXM`` decode tokens. It returns exactly the stock
tensors: x (reduced MoE output, bf16 [M, 5120]), residual (bf16 [M, 4, 5120]), post_mix
(fp32 [M, 4, 1]), res_mix (fp32 [M, 4, 4]), ffn_pre (fp32 [M, 4]).

Weights: dense FP8/bf16 weights are read from the checkpoint (stock Humming repacks the module
weights) into the kernel's fragment-major layout; routed experts use the raw TP-sliced layout
shared with the ds41h MoE method (``ds41h.moe.get_raw_experts``); the fp32/bf16 vectors (gate
bias, hc_ffn_*, ffn_norm) are the stock parameters themselves.

Public API
----------
extension(spec=None)                               compiled torch extension (JIT, cached); spec / env
                                                   DS41H_DECODER_DEFINES select compile knobs, e.g.
                                                   "ROUTED_MMA=1,DENSE_MMA=1" (RS-wgmma e4m3 GEMMs)
stage_fp8(w, perm8=True) -> Tensor                 fragment-major staging of a dense [N, K] 1-byte matrix
WG_PERM, wg_order(x), quant_e4m3_wg(x, scale)      RS-wgmma activation format (e4m3, in-block K order)
x_scale(norm) -> float                             static e4m3 scale of the FFN input x (wgmma paths)
interleave(w, block) -> Tensor                     [2R, K] -> 32-row interleave of two halves
load_dense(reader, layer, rank, device) -> DenseWeights
load_raw_experts(reader, layer, rank, device, experts=None, out=None) -> (w13, s13, w2, s2)  tests
DecoderEngine(rank, world=4, cpu_group=None, device=None, spec=None)
    .add_layer(dense, experts) -> int
    .update_experts(handle, experts)
    .run(handle, o, positions, cos_sin, residual, post_mix, res_mix, attn_pre) -> 5 tensors
    .debug_topk() -> (ids int32 [MAXM, 6], weights fp32 [MAXM, 6])
    .trace_dump(clear=False) -> int32 trace ring (TRACE=1 builds, see ds41h/decoder_trace.py)
    .variant -> {"routed_mma": 0|1, "dense_mma": 0|1}, .layer_xs(idx) -> float
vLLM glue (patch_decoder.py): engine(), prepare_model(lm), decode_step(layer, positions, limit),
    eligible(layer, x, positions), prologue_eligible(layer, x, residual, positions),
    raw_attention_output(attn), ffn(layer, o, positions, residual, post_mix, res_mix, attn_pre),
    attention_side (overridable hook, see patch_decoder.py).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
from dataclasses import dataclass, fields

import torch

from ds41h import cuext, flags

# Under the "vllm" hierarchy so vLLM's logging config shows it (a silent fallback to stock is the
# failure mode that matters here).
logger = logging.getLogger("vllm.ds41h.decoder")

# Model constants the kernel is compiled for (decoder.cu); checked against the config at load.
HID = 5120
HC = 4
MIX = 24
EXPERTS = 384
TOPK = 6
INTER = 576  # moe_intermediate_size 2304 / TP4
OA = 2048  # 2 local wo_a groups x o_lora_rank 1024
OAK = 4096  # 8 heads x 512 per group
HDIM = 512
LAYERS = 40
TP = 4
# In-8 K permutation shared by staged dense weights and f16 activations (decoder.cu header).
P8 = (0, 2, 4, 6, 1, 3, 5, 7)
# RS-wgmma activation order inside every 32-wide K block (decoder.cu wg_pos): position p holds
# physical k = WG_PERM[p]; k = 8t + j sits at 16 (j & 1) + 4t + (j >> 1). It matches the wgmma A
# fragment columns of both the FP4 bit-trick expansion (lo / hi nibbles of the ldmatrix word of
# k = 8t..8t+7) and the P8-staged dense fragments.
WG_PERM = tuple(8 * ((p & 15) >> 2) + 2 * (p & 3) + (p >> 4) for p in range(32))
H_SCALE = 0.25  # e4m3 scale of SwiGLU outputs on the wgmma paths (|h| <= 100 -> code <= 400)


# ------------------------------------------------------------------------------- extension
_KNOBS = (
    "STAGES", "PER_SM", "MAXM", "TWO_SHOT_MIN", "ROUTED_MMA", "DENSE_MMA", "TRACE", "COARSE", "R1C_NB", "FUSED_AR1", "FUSED_Z", "WOB1", "R1C_TMA",
    "DENSE_BULK", "HIER_ROUTE", "LOSSLESS", "R2F_PF", "LAZY_SIG", "R1F_PIPE", "SH2_EARLY", "PROUTE", "GDET", "SH1_GATE", "MBAR_SLEEP", "ABL", "FROUTE", "WOA_PRE", "POLLF", "POLL_NS", "PDL_EARLY", "SCL_DIRECT", "R1KP_DYN", "REL_DEP", "DENSE_TS1", "R2F_ROLL", "CTX_INLINE", "R1F_LOC", "SMEM_DIET", "R2F_LOC", "PROD_ARR", "DYN_FIRST", "WARM_ROUTE", "CLEAR_LATE", "ROUTE_LEAN", "R1SPLIT", "OWNA_PRE", "RESET_LEAN", "ROUTE1", "R2P", "R2T", "PFR1", "WOB3", "HOTDISP", "TAILFREE", "LLP", "R1SK", "GATE_KS", "TAGX", "GSUM2", "PXF",
)  # fmt: skip


def _defines(spec: str | None = None) -> tuple[str, ...]:
    """Compile-time knobs from `spec` or DS41H_DECODER_DEFINES, e.g. "STAGES=8,PER_SM=2,MAXM=32".

    ROUTED_MMA=1 / DENSE_MMA=1 select the RS-wgmma e4m3 variants of the routed-expert / dense FP8
    (wo_a, wo_b, shared expert) GEMMs (default 0 = FP16 HMMA, see decoder.cu header). TRACE=1 builds
    the per-job timeline recorder (ds41h/decoder_trace.py). COARSE=1 enables the coarse routed path
    for m <= 8 (decoder.cu DS41H_DECODER_COARSE, default 1); LOSSLESS=1 (default) makes it lossless (f16
    activations into every GEMM, R1F / R2F HMMA jobs); LOSSLESS=0 restores the e4m3 coarse path (opt-in,
    LOSSY: e4m3 x / h activations). ROUTED_MMA=1 / DENSE_MMA=1 are lossy as well."""
    if spec is None:
        spec = os.environ.get("DS41H_DECODER_DEFINES", "")
    out = []
    for item in filter(None, (s.strip() for s in spec.strip().split(","))):
        key, _, val = item.partition("=")
        if key not in _KNOBS or not val.isdigit():
            raise ValueError(f"DS41H_DECODER_DEFINES: bad entry {item!r}")
        out.append(f"-DDS41H_DECODER_{key}={val}")
    return tuple(out)


def extension(spec: str | None = None):
    """Build (or load the cached) decoder extension for sm_90a. `spec` overrides
    DS41H_DECODER_DEFINES (same "KEY=VAL,..." syntax), e.g. extension("ROUTED_MMA=1")."""
    defines = _defines(spec)
    tag = hashlib.sha1(" ".join(defines).encode()).hexdigest()[:8] if defines else "default"
    return cuext.load(f"ds41h_decoder_{tag}", "decoder.cu", extra_cuda_cflags=defines)


# ------------------------------------------------------------------------------- weight staging
def stage_fp8(w: torch.Tensor, perm8: bool = True) -> torch.Tensor:
    """Dense [N, K] 1-byte matrix -> the kernel's fragment-major stage layout (BRRR stages()).

    Per (64-row block, 128-byte K group): 16 fragments (K32 block q, 16-row tile rt) of 512 B;
    lane (g, t) holds row g's 8 bytes at K 32q + 8t.. then row g+8's, each 8-group permuted by
    P8 when ``perm8`` (FP8 weights feeding the f16 HMMA). bf16 matrices are passed as their
    uint8 view with ``perm8=False`` (the gate's bf16 HMMA consumes natural order). K is
    zero-padded to a multiple of 128 bytes.
    """
    w = w.contiguous().view(torch.uint8)
    n, k = w.shape
    if n % 64:
        raise ValueError(f"stage_fp8: rows {n} not a multiple of 64")
    if k % 128:
        w = torch.nn.functional.pad(w, (0, -k % 128))
    if perm8:
        idx = torch.tensor(P8, device=w.device)
        w = w.view(n, -1, 8).index_select(2, idx).reshape(n, -1)
    return w.reshape(n // 64, 4, 2, 8, -1, 4, 4, 8).permute(0, 4, 5, 1, 3, 6, 2, 7).contiguous().view(-1)


def wg_order(x: torch.Tensor) -> torch.Tensor:
    """[..., K] (K % 32 == 0) natural K order -> RS-wgmma activation order (WG_PERM per 32 block)."""
    idx = torch.tensor(WG_PERM, device=x.device)
    return x.unflatten(-1, (-1, 32)).index_select(-1, idx).flatten(-2)


def quant_e4m3_wg(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Activation rows -> the kernel's e4m3 bytes (x / scale, round-to-nearest-even, satfinite)
    in wg order (uint8 [..., K]). Dequantize with e4m3 value * scale."""
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    return wg_order(q).contiguous()


def x_scale(norm: torch.Tensor) -> float:
    """decoder.cu x_scale: power of two >= 1.01 max|ffn_norm| sqrt(HID) / 448 (RMSNorm output bound)."""
    bound = torch.tensor(float(norm.float().abs().max()), dtype=torch.float32) * torch.tensor(HID, dtype=torch.float32).sqrt() * 1.01
    return float(torch.exp2(torch.ceil(torch.log2(torch.clamp(bound, min=1e-30) / 448.0))))


def interleave(w: torch.Tensor, block: int) -> torch.Tensor:
    """[2R, ...] = [A; B] -> blocks of `block` rows alternating A, B (shared gate/up)."""
    return w.reshape(2, -1, block, *w.shape[1:]).transpose(0, 1).reshape(w.shape).contiguous()


def _u8(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {t.dtype}")
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


@dataclass
class DenseWeights:
    """One layer's dense decoder weights for one TP rank, staged for the kernel (all on GPU)."""

    woa: torch.Tensor  # staged e4m3 [2048 x 4096]
    woa_s: torch.Tensor  # ue8m0 [64, 128]
    wob: torch.Tensor  # staged e4m3 [5120 x 2048]
    wob_s: torch.Tensor  # ue8m0 [160, 64]
    gate: torch.Tensor  # staged bf16 [384 x 5120] (uint8 view)
    gate_b: torch.Tensor  # fp32 [384]
    sh13: torch.Tensor  # staged e4m3, 32-row interleaved w1/w3 [1152 x 5120]
    sh13_s: torch.Tensor  # ue8m0 [36, 160], interleaved like the rows
    sh2: torch.Tensor  # staged e4m3 [5120 x 640] (K padded)
    sh2_s: torch.Tensor  # ue8m0 [160, 20] (18 padded)
    fn: torch.Tensor  # hc_ffn_fn fp32 [24, 20480]
    hc_scale: torch.Tensor  # hc_ffn_scale fp32 [3]
    hc_base: torch.Tensor  # hc_ffn_base fp32 [24]
    norm: torch.Tensor  # ffn_norm.weight bf16 [5120]

    def nbytes(self) -> int:
        return sum(getattr(self, f.name).numel() * getattr(self, f.name).element_size() for f in fields(self))


def _get_tp(reader, name: str, rank: int) -> torch.Tensor:
    """Rank `rank`'s slice of a checkpoint tensor, exactly as stock vLLM loads it (ds41h.ref)."""
    if hasattr(reader, "get_tp"):
        return reader.get_tp(name, rank)
    from ds41h.ref.weights import tp_slice

    return tp_slice(name, reader.get(name), rank)


def _expect(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(t.shape) != shape:
        raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
    return t


def load_dense(reader, layer: int, rank: int, device, vectors: dict[str, torch.Tensor] | None = None) -> DenseWeights:
    """Load and stage layer `layer`'s dense decoder weights of TP rank `rank` from the checkpoint.

    `reader` is a ds41h.ref.weights.CheckpointReader. `vectors` may supply the fp32/bf16 vectors
    (gate_b, fn, hc_scale, hc_base, norm) from the live stock module instead of the checkpoint.
    TP slices (stock vLLM 0.32.0, see ds41h.ref.weights rules): wo_a rows (ColumnParallel bmm,
    attention.py:383-392), wo_b columns (RowParallel, :396-403), shared w1/w3 rows and w2 columns
    (deepseek_v4/nvidia/model.py:155-176), gate replicated.
    """
    p = f"layers.{layer}"

    def fp8(name: str, shape: tuple[int, ...], sshape: tuple[int, ...]):
        w = _expect(_get_tp(reader, f"{p}.{name}.weight", rank), name, shape)
        s = _expect(_get_tp(reader, f"{p}.{name}.scale", rank), name + ".scale", sshape)
        return _u8(w).to(device), _u8(s).to(device)

    woa, woa_s = fp8("attn.wo_a", (OA, OAK), (OA // 32, OAK // 32))
    wob, wob_s = fp8("attn.wo_b", (HID, OA), (HID // 32, OA // 32))
    w1, s1 = fp8("ffn.shared_experts.w1", (INTER, HID), (INTER // 32, HID // 32))
    w3, s3 = fp8("ffn.shared_experts.w3", (INTER, HID), (INTER // 32, HID // 32))
    w2, s2 = fp8("ffn.shared_experts.w2", (HID, INTER), (HID // 32, INTER // 32))
    gate = _expect(reader.get(f"{p}.ffn.gate.weight"), "ffn.gate.weight", (EXPERTS, HID))
    if gate.dtype != torch.bfloat16:
        raise TypeError(f"gate weight dtype {gate.dtype}, expected bfloat16")
    if vectors is None:
        vectors = {
            "gate_b": reader.get(f"{p}.ffn.gate.bias"),
            "fn": reader.get(f"{p}.hc_ffn_fn"),
            "hc_scale": reader.get(f"{p}.hc_ffn_scale"),
            "hc_base": reader.get(f"{p}.hc_ffn_base"),
            "norm": reader.get(f"{p}.ffn_norm.weight"),
        }
    vec = {
        "gate_b": (torch.float32, (EXPERTS,)),
        "fn": (torch.float32, (MIX, HC * HID)),
        "hc_scale": (torch.float32, (3,)),
        "hc_base": (torch.float32, (MIX,)),
        "norm": (torch.bfloat16, (HID,)),
    }
    v = {}
    for key, (dtype, shape) in vec.items():
        t = vectors[key]
        if t.dtype != dtype or tuple(t.shape) != shape:
            raise ValueError(f"{key}: {t.dtype} {tuple(t.shape)}, expected {dtype} {shape}")
        v[key] = t.to(device).contiguous()
    # x = RMSNorm(.) is bounded by sqrt(5120) * max|w| per element; f16 staging needs < 65504.
    bound = float(v["norm"].float().abs().max()) * HID**0.5
    if bound >= 32768.0:
        raise ValueError(f"layer {layer}: ffn_norm bound {bound:.0f} exceeds the f16 activation range")
    return DenseWeights(
        woa=stage_fp8(woa),
        woa_s=woa_s.contiguous(),
        wob=stage_fp8(wob),
        wob_s=wob_s.contiguous(),
        gate=stage_fp8(gate.to(device).view(torch.uint8), perm8=False),
        gate_b=v["gate_b"],
        sh13=stage_fp8(interleave(torch.cat([w1, w3]), 32)),
        sh13_s=interleave(torch.cat([s1, s3]), 1),
        sh2=stage_fp8(w2),
        sh2_s=torch.nn.functional.pad(s2, (0, 2)).contiguous(),
        fn=v["fn"],
        hc_scale=v["hc_scale"],
        hc_base=v["hc_base"],
        norm=v["norm"],
    )


def load_raw_experts(reader, layer: int, rank: int, device, experts=None, out=None):
    """Raw TP-sliced routed experts from the checkpoint (the layout ds41h.moe keeps):
    w13 [E, 1152, 2560], s13 [E, 1152, 160], w2 [E, 5120, 288], s2 [E, 5120, 18] uint8 on
    `device`. `experts` limits which experts are read (others stay zero / untouched), `out`
    fills existing tensors in place (tests load only the experts a step routes to)."""
    E = EXPERTS
    if out is None:
        out = (
            torch.zeros(E, 2 * INTER, HID // 2, dtype=torch.uint8, device=device),
            torch.zeros(E, 2 * INTER, HID // 32, dtype=torch.uint8, device=device),
            torch.zeros(E, HID, INTER // 2, dtype=torch.uint8, device=device),
            torch.zeros(E, HID, INTER // 32, dtype=torch.uint8, device=device),
        )
    w13, s13, w2, s2 = out
    for e in experts if experts is not None else range(E):
        q = f"layers.{layer}.ffn.experts.{e}"
        w13[e, :INTER] = _u8(_get_tp(reader, f"{q}.w1.weight", rank)).to(device)
        w13[e, INTER:] = _u8(_get_tp(reader, f"{q}.w3.weight", rank)).to(device)
        s13[e, :INTER] = _u8(_get_tp(reader, f"{q}.w1.scale", rank)).to(device)
        s13[e, INTER:] = _u8(_get_tp(reader, f"{q}.w3.scale", rank)).to(device)
        w2[e] = _u8(_get_tp(reader, f"{q}.w2.weight", rank)).to(device)
        s2[e] = _u8(_get_tp(reader, f"{q}.w2.scale", rank)).to(device)
    return out


def _check_experts(experts) -> tuple[torch.Tensor, ...]:
    if experts is None or len(experts) != 4:
        raise ValueError("experts must be (w13, s13, w2, s2)")
    shapes = ((EXPERTS, 2 * INTER, HID // 2), (EXPERTS, 2 * INTER, HID // 32), (EXPERTS, HID, INTER // 2), (EXPERTS, HID, INTER // 32))
    out = []
    for t, shape, name in zip(experts, shapes, ("w13", "s13", "w2", "s2")):
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        out.append(_u8(t))
    return tuple(out)


# ------------------------------------------------------------------------------- engine
class DecoderEngine:
    """Owns the per-rank scratch, the CUDA-IPC exchange buffers and the registered layers.

    world=4: buffers are exchanged over `cpu_group` (collective: every TP rank must construct
    its engine at the same point). world=1: solo mode, both all-reduces are local copies (each
    rank's partial sums), for single-GPU tests.
    """

    def __init__(self, rank: int, world: int = TP, cpu_group=None, device=None, spec: str | None = None):
        if world not in (1, TP):
            raise ValueError(f"world must be 1 or {TP}")
        self.ext = extension(spec)
        self.rank, self.world = rank, world
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.maxm = int(self.ext.MAXM)
        self._layers: list[tuple] = []
        with torch.cuda.device(self.device):
            mine = self.ext.alloc_ipc(self.ext.flag_bytes() + self.ext.inbox_bytes())
            ptrs = [0] * int(self.ext.RANKS)
            ptrs[rank] = mine[0]
            if world > 1:
                import torch.distributed as dist

                handles = cuext.exchange_ipc_handles(bytes(int(b) & 0xFF for b in mine[1:]), cpu_group)
                for r, h in enumerate(handles):
                    if r != rank:
                        ptrs[r] = self.ext.open_ipc(list(h))
            self.handle = self.ext.create(ptrs, rank, world)
            if world > 1:
                dist.barrier(group=cpu_group)
        self.set_coarse(int(os.environ.get("DS41H_DECODER_R1KP", "0")), int(os.environ.get("DS41H_DECODER_LAG", "0")))

    def set_coarse(self, r1kp: int = 0, lag: int = 0) -> None:
        """Coarse routed path (m <= 8) knobs: R1 K parts (0 = auto by m: the lossless path uses 5 / 2 / 1 K
        parts for m = 1 / 2-3 / 4-8; 1 = full K per job, 2, lossless also 5) and the expert lag of the R2 jobs
        behind their R1 jobs (0 = auto, about 1.5 grids of R1 jobs). Env defaults: DS41H_DECODER_R1KP,
        DS41H_DECODER_LAG."""
        self.ext.set_coarse(self.handle, r1kp, lag)

    def add_layer(self, dense: DenseWeights, experts) -> int:
        w13, s13, w2, s2 = _check_experts(experts)
        idx = self.ext.add_layer(
            self.handle, dense.woa, dense.woa_s, dense.wob, dense.wob_s, dense.gate, dense.gate_b, dense.sh13, dense.sh13_s,
            dense.sh2, dense.sh2_s, w13, s13, w2, s2, dense.fn, dense.hc_scale, dense.hc_base, dense.norm,
        )  # fmt: skip
        self._layers.append((dense, (w13, s13, w2, s2)))
        return idx

    def update_experts(self, idx: int, experts) -> None:
        w13, s13, w2, s2 = _check_experts(experts)
        self.ext.update_experts(self.handle, idx, w13, s13, w2, s2)
        self._layers[idx] = (self._layers[idx][0], (w13, s13, w2, s2))

    def set_prefetch(self, idx: int, tensors: list[torch.Tensor], dense: bool = False) -> None:
        """L2-prefetch these tensors' bytes during layer `idx`; [] = none. dense=False: in the OWNA phase
        (latency bound, HBM idle; evict_last), e.g. the next layer's prologue weights. dense=True: spread
        over the R1C jobs of the coarse routed phase (evict_normal), meant for the next layer's
        latency-critical dense weights (dense_prefetch_tensors)."""
        self.ext.set_prefetch(self.handle, idx, [t.contiguous() for t in tensors], dense)

    def dense_prefetch_tensors(self, idx: int) -> list[torch.Tensor]:
        """Layer `idx`'s weights on the dense critical path (wo_a, wo_b, gate, hc_ffn_fn), ~25 MB / rank."""
        d = self._layers[idx][0]
        return [d.woa, d.woa_s, d.wob, d.wob_s, d.gate, d.fn]

    def experts_of(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self._layers[idx][1]

    def run(self, idx: int, o, positions, cos_sin, residual, post_mix, res_mix, attn_pre, p1a=None):
        """Run layer `idx` on M = residual.shape[0] tokens. Shapes follow stock exactly:
        o bf16 [>=M, >=16, 512] (head stride 512, e.g. the stock [M, 64, 512] buffer), positions
        int64 [M], cos_sin fp32 [max_pos, 64], residual bf16 [M, 4, 5120], post_mix fp32
        [M, 4, 1], res_mix fp32 [M, 4, 4], attn_pre fp32 [M, 4].

        p1a = (resid_out bf16 [>=M, 4, 5120], xcb bf16 [>=M, 5120], xsq fp32 [>=M, 160]) or None: the kernel
        also writes the next layer's prologue phase P1a (post-mapped residual stream, ffn_pre collapse, per
        32-column slice square sums), bit-identical to prologue.cu p1_collapse (decoder.cu xf_p1a)."""
        m = residual.shape[0]
        dev = residual.device
        x = torch.empty((m, HID), dtype=torch.bfloat16, device=dev)
        resid1 = torch.empty_like(residual)
        post = torch.empty((m, HC, 1), dtype=torch.float32, device=dev)
        comb = torch.empty((m, HC, HC), dtype=torch.float32, device=dev)
        pre = torch.empty((m, HC), dtype=torch.float32, device=dev)
        if positions.dtype != torch.int64:
            positions = positions.long()
        p1r, p1x, p1s = p1a if p1a is not None else (None, None, None)
        self.ext.run(self.handle, idx, o, positions.contiguous(), cos_sin, residual, post_mix, res_mix, attn_pre, x, resid1, post, comb, pre, p1r, p1x, p1s)
        return x, resid1, post, comb, pre

    def trace_dump(self, clear: bool = False) -> torch.Tensor:
        """Trace builds (TRACE=1): int32 [TRACE_LAUNCHES, ctas, words] ring of per-CTA job records of
        the last launches (decoder_trace.py decodes it); empty otherwise."""
        return self.ext.trace_dump(self.handle, clear)

    def debug_topk(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids, w = self.ext.debug_topk(self.handle)
        return ids, w

    @property
    def variant(self) -> dict[str, int]:
        """Compiled MMA flavours (0 = HMMA, 1 = RS-wgmma e4m3), coarse path and its lossless mode, see decoder.cu."""
        return {
            "routed_mma": int(self.ext.ROUTED_MMA), "dense_mma": int(self.ext.DENSE_MMA), "coarse": int(self.ext.COARSE),
            "lossless": int(getattr(self.ext, "LOSSLESS", 0)),
        }  # fmt: skip

    @property
    def lossless(self) -> bool:
        """Whether every GEMM of this build gets 16-bit activations (no e4m3 activation path compiled in)."""
        v = self.variant
        return not v["routed_mma"] and not v["dense_mma"] and (not v["coarse"] or bool(v["lossless"]))

    def layer_xs(self, idx: int) -> float:
        """e4m3 scale of the FFN input x of layer `idx` (wgmma paths)."""
        return float(self.ext.layer_xs(self.handle, idx))


# ------------------------------------------------------------------------------- vLLM glue
_ENGINE: DecoderEngine | None = None
_PREPARED = False


def engine() -> DecoderEngine:
    """The process-wide engine for the vLLM TP group (collective on first call)."""
    global _ENGINE
    if _ENGINE is None:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        _ENGINE = DecoderEngine(group.rank_in_group, group.world_size, group.cpu_group)
        if os.environ.get("DS41H_DECODER_TRACE_DUMP"):
            _start_trace_dumper(_ENGINE, os.environ["DS41H_DECODER_TRACE_DUMP"], group.rank_in_group)
    return _ENGINE


def _start_trace_dumper(eng: DecoderEngine, prefix: str, rank: int) -> None:
    """In-situ decoder trace (diagnostics; needs a TRACE=1 build): a daemon thread waits for the file `<prefix>.go`
    (created by the job script once the server decodes, i.e. after graph capture: trace_dump synchronizes the
    device), then saves the ring of the last TRACE_LAUNCHES layer launches `<prefix>_r<rank>_<k>.npy` 4 times, 1.5 s
    apart (decoder_trace.analyze_dump reads them)."""
    import threading
    import time

    import numpy as np

    dev = torch.cuda.current_device()

    def run():
        torch.cuda.set_device(dev)
        while not os.path.exists(prefix + ".go"):
            time.sleep(0.25)
        stream = torch.cuda.Stream(dev)
        for k in range(4):
            time.sleep(1.5)
            try:
                with torch.cuda.stream(stream):
                    buf = eng.trace_dump(False).cpu().numpy()
                if buf.size:
                    np.save(f"{prefix}_r{rank}_{k}.npy", buf)
            except Exception as e:  # noqa: BLE001 (diagnostics must never take the server down)
                print(f"ds41h decoder trace dump failed: {e}", flush=True)
                return

    threading.Thread(target=run, name="ds41h-decoder-trace-dump", daemon=True).start()


@functools.cache
def max_tokens() -> int:
    return min(flags.MAX_DECODE_TOKENS, int(extension().MAXM))


def _config_ok(cfg) -> str | None:
    """Why the kernel's compiled constants do not match this model config (None = they do)."""
    expect = {
        "hidden_size": HID,
        "hc_mult": HC,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": INTER * TP,
        "num_attention_heads": 64,
        "head_dim": HDIM,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "hc_sinkhorn_iters": 20,
        "n_shared_experts": 1,
    }
    for key, val in expect.items():
        if getattr(cfg, key, None) != val:
            return f"{key}={getattr(cfg, key, None)} (kernel built for {val})"
    approx = {"hc_eps": 1e-6, "rms_norm_eps": 1e-20, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0}
    for key, val in approx.items():
        got = getattr(cfg, key, None)
        if got is None or abs(float(got) - val) > 1e-6 * abs(val):
            return f"{key}={got} (kernel built for {val})"
    if getattr(cfg, "scoring_func", "sqrtsoftplus") != "sqrtsoftplus" or not getattr(cfg, "norm_topk_prob", True):
        return "routing is not sqrtsoftplus + normalized top-k"
    return None


def _decoder_layers(lm) -> list:
    from vllm.model_executor.models.utils import PPMissingLayer

    model = lm.model
    out = []
    for layer in model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if getattr(layer.attn, "layer_id", LAYERS) >= LAYERS:
            continue
        out.append(layer)
    return out


def _layer_reason(layer) -> str | None:
    """Static reasons a decoder layer cannot use the kernel (None = eligible)."""
    if layer.use_sequence_parallel:
        return "sequence parallel"
    if layer.fuse_mhc_all_reduce:
        return "fused mHC all-reduce (forced off by ds41h when DECODER is on)"
    ffn = layer.ffn
    if getattr(ffn, "use_native_mega_moe", False):
        return "native MegaMoE"
    if getattr(ffn, "n_routed_experts", None) != EXPERTS or getattr(ffn.gate, "tid2eid", None) is not None:
        return "not a 384-expert noaux_tc MoE"
    if getattr(ffn.gate, "e_score_correction_bias", None) is None:
        return "no gate correction bias"
    if getattr(ffn, "shared_experts", None) is None:
        return "no shared expert"
    attn = layer.attn
    if (attn.n_local_heads, attn.n_local_groups, attn.head_dim, attn.rope_head_dim) != (16, 2, HDIM, 64):
        return "attention shape is not TP4 16 heads / 2 groups"
    # The fused path needs the raw [M, H, 512] attention buffer, which only the patched
    # DeepseekV4FlashMLAAttention._o_proj hands out (patch_decoder._patch_flashmla). Other backends
    # (FlashInfer sparse MLA, mega attention, ...) would return the projected [M, 5120] output.
    if not getattr(type(attn), "_ds41h_decoder_patched", False):
        return f"attention class {type(attn).__name__} does not expose the raw attention output"
    return None


def _raw_experts(get_raw_experts, layer):
    """ds41h.moe.get_raw_experts on the decoder layer (contract), tolerating implementations
    keyed on the MoE block or its FusedMoE module. Returns (experts | None, reason)."""
    why = "returned None"
    for obj in (layer, layer.ffn, getattr(layer.ffn, "experts", None)):
        if obj is None:
            continue
        try:
            ex = get_raw_experts(obj)
        except Exception as e:  # not finalized yet (first hook call) or keyed differently
            why = repr(e)
            continue
        if ex is not None:
            return ex, None
    return None, why


def prepare_model(lm) -> None:
    """Build the per-layer kernel state once weights are final (process_weights_after_loading).

    Called from DeepseekV41LLMForCausalLM.process_weights_after_loading (inside load_weights, i.e.
    BEFORE the per-module quant finalize that sets up the ds41h raw experts) and, via the patched
    multimodal wrapper DeepseekV41ForCausalLM (vl_model.py:332-338, whose own hook is the only one
    the loader runs after that finalize, model_loader/utils.py:202-204), once more afterwards.
    Preparation waits until ds41h.moe exposes the raw experts on every rank; later calls only
    re-bind moved expert tensors. Collective over the TP CPU group (all ranks reach this hook the
    same number of times).
    """
    global _PREPARED
    if not flags.DECODER:
        return
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    layers = _decoder_layers(lm)
    if not layers:
        return
    cfg = layers[0]._ds41h_cfg if hasattr(layers[0], "_ds41h_cfg") else lm.config
    group = get_tp_group()
    reason = _config_ok(cfg)
    if group.world_size != TP:
        reason = f"TP world size {group.world_size} (kernel built for {TP})"
    experts = {}
    if reason is None:
        try:
            from ds41h.moe import get_raw_experts
        except ImportError as e:  # moe component missing: stay on stock
            reason = f"ds41h.moe.get_raw_experts unavailable ({e})"
        else:
            for layer in layers:
                ex, why = _raw_experts(get_raw_experts, layer)
                if ex is None:
                    reason = f"raw experts of layer {layer.attn.layer_id} not available ({why})"
                    break
                try:
                    experts[layer.attn.layer_id] = _check_experts(ex)
                except (TypeError, ValueError) as e:  # stay on stock instead of failing the load
                    reason = f"raw experts of layer {layer.attn.layer_id}: {e}"
                    break
    # All ranks must agree before anything collective happens.
    ok = torch.tensor([1 if reason is None else 0], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if not int(ok.item()):
        if not _PREPARED:
            logger.info("ds41h decoder: not preparing yet: %s", reason or "another rank is not ready")
        return
    if _PREPARED:
        eng = engine()
        for layer in layers:
            idx = getattr(layer, "_ds41h_decoder", None)
            if idx is not None and any(a.data_ptr() != b.data_ptr() for a, b in zip(eng.experts_of(idx), experts[layer.attn.layer_id])):
                eng.update_experts(idx, experts[layer.attn.layer_id])
        return

    from ds41h.ref.weights import CheckpointReader

    eng = engine()
    rank = eng.rank
    model_dir = getattr(layers[0], "_ds41h_model_dir", None)
    if model_dir is None:
        raise RuntimeError("ds41h decoder: model directory unknown (DeepseekV4DecoderLayer.__init__ patch missing)")
    reader = CheckpointReader(model_dir)
    total = 0
    for layer in layers:
        lid = layer.attn.layer_id
        why = _layer_reason(layer)
        if why is not None:
            logger.warning("ds41h decoder: layer %d stays on the stock path: %s", lid, why)
            continue
        ffn = layer.ffn
        vectors = {
            "gate_b": ffn.gate.e_score_correction_bias.data.float(),
            "fn": layer.hc_ffn_fn.data,
            "hc_scale": layer.hc_ffn_scale.data,
            "hc_base": layer.hc_ffn_base.data,
            "norm": layer.ffn_norm.weight.data,
        }
        dense = load_dense(reader, lid, rank, eng.device, vectors)
        total += dense.nbytes()
        cos_sin = layer.attn.rotary_emb.cos_sin_cache
        if cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous() or cos_sin.device != eng.device:
            cos_sin = cos_sin.to(device=eng.device, dtype=torch.float32).contiguous()
        if cos_sin.dim() != 2 or cos_sin.shape[1] != 64:
            logger.warning("ds41h decoder: layer %d stays on the stock path: cos_sin_cache %s", lid, tuple(cos_sin.shape))
            continue
        layer._ds41h_cos_sin = cos_sin
        layer._ds41h_decoder = eng.add_layer(dense, experts[lid])
    pf = os.environ.get("DS41H_DECODER_PREFETCH", "0")
    if pf in ("1", "2"):  # 1: during the OWNA jobs, 2: during the routed R1 jobs (streaming phase, coarse m)
        _set_prologue_prefetch(eng, layers, phase=int(pf) - 1)
    if os.environ.get("DS41H_DECODER_DPF", "0") == "1":
        _set_dense_prefetch(eng, layers)
    torch.cuda.synchronize(eng.device)
    _PREPARED = True
    logger.info(
        "ds41h decoder: %d layers on the persistent decode kernel (M <= %d), %.2f GiB staged dense weights per rank",
        sum(hasattr(layer, "_ds41h_decoder") for layer in layers), max_tokens(), total / 2**30,
    )  # fmt: skip


def _set_prologue_prefetch(eng: DecoderEngine, layers: list, phase: int = 0) -> None:
    """Layer i's kernel L2-prefetches layer i+1's prologue weights (ds41h.prologue: fused wq_a|wkv,
    wq_b, hc_attn_fn; only when that layer runs the ds41h prologue), evict_last; phase 0 = during the OWNA jobs,
    1 = during the coarse R1 jobs (DS41H_DECODER_PREFETCH=2)."""
    if not flags.PROLOGUE:
        return
    try:
        from ds41h import prologue
    except ImportError:
        return
    n = 0
    for cur, nxt in zip(layers, layers[1:]):
        idx = getattr(cur, "_ds41h_decoder", None)
        if idx is None:
            continue
        try:
            info = prologue.prepare_layer(nxt)
        except Exception:  # noqa: BLE001 - prefetch is an optimization only
            info = None
        if info is None:
            continue
        eng.set_prefetch(idx, [info.wqa.w, info.wqa.s, info.wqb.w, info.wqb.s, info.fn])
        eng.ext.set_prefetch_phase(eng.handle, idx, phase)
        n += 1
    logger.info("ds41h decoder: L2 prefetch of the next layer's prologue weights on %d layers", n)


def _set_dense_prefetch(eng: DecoderEngine, layers: list) -> None:
    """Layer i's kernel L2-prefetches layer i+1's wo_a / wo_b / gate / fn during its routed phase (coarse
    path only: M <= COARSE_M)."""
    n = 0
    for cur, nxt in zip(layers, layers[1:]):
        idx, nidx = getattr(cur, "_ds41h_decoder", None), getattr(nxt, "_ds41h_decoder", None)
        if idx is None or nidx is None:
            continue
        eng.set_prefetch(idx, eng.dense_prefetch_tensors(nidx), dense=True)
        n += 1
    logger.info("ds41h decoder: L2 prefetch of the next layer's dense weights on %d layers", n)


def decode_step(layer, positions: torch.Tensor, limit: int) -> bool:
    """Decode-only step (uniform decode / DSpark verify) with 1..limit rows. Static per
    CUDA-graph shape: FULL_DECODE_ONLY graphs capture exactly these steps; padded rows of a
    captured graph are computed like stock does and ignored afterwards."""
    if not 1 <= positions.shape[0] <= limit:
        return False
    if layer.mhc_stream is not None or layer.use_sequence_parallel or layer.fuse_mhc_all_reduce:
        return False
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):  # profile / dummy runs (None) and microbatching (list)
        return False
    swa = md.get(layer.attn.swa_cache_layer.prefix)
    return swa is not None and getattr(swa, "num_prefills", 1) == 0


def eligible(layer, x, positions: torch.Tensor) -> bool:
    """Whether this forward call takes the fused decoder path (layer prepared + decode step)."""
    if not flags.DECODER or getattr(layer, "_ds41h_decoder", None) is None or not isinstance(x, torch.Tensor):
        return False
    return decode_step(layer, positions, max_tokens())


def prologue_eligible(layer, x, residual, positions: torch.Tensor) -> bool:
    """Whether the ds41h prologue (flag DS41H_PROLOGUE) replaces the attention-side mHC and input
    projections of this decode step (contract: ds41h/prologue.py eligible / run_prologue)."""
    if not flags.PROLOGUE or not decode_step(layer, positions, flags.MAX_DECODE_TOKENS):
        return False
    from ds41h import prologue

    return prologue.eligible(layer, x, residual, positions)


@contextlib.contextmanager
def raw_attention_output(attn):
    """Make attn._o_proj return the raw attention buffer [M, H, 512] (patched in patch_decoder)."""
    attn._ds41h_raw_o = True
    try:
        yield
    finally:
        attn._ds41h_raw_o = False


def ffn(layer, o: torch.Tensor, positions, residual, post_mix, res_mix, attn_pre, p1a=None):
    """Fused post-attention half of DeepseekV4DecoderLayer.forward (model.py:597-625). p1a = (resid_out, xcb, xsq):
    also write the next layer's prologue phase P1a (DecoderEngine.run, decoder.cu xf_p1a); None = off."""
    if not isinstance(o, torch.Tensor) or o.dtype != torch.bfloat16 or o.dim() != 3:
        raise RuntimeError(f"ds41h decoder: unexpected attention output {type(o)}")
    # The kernel needs dense row-major mixes / residual (the stock producers already return
    # contiguous tensors, so these are no-ops; they keep an unusual producer from failing the
    # extension's checks inside a captured graph).
    return engine().run(
        layer._ds41h_decoder, o, positions, layer._ds41h_cos_sin, residual.contiguous(), post_mix.contiguous(),
        res_mix.contiguous(), attn_pre.contiguous(), p1a=p1a,
    )  # fmt: skip


# Optional override of the attention half (e.g. the ds41h prologue component):
#   fn(layer, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes,
#      engram_mask, capture_previous_aux) -> (o, residual, post_mix, res_mix, attn_pre, previous_aux)
# where o is the raw attention output (pre-wo_a). None = stock replica in patch_decoder.py.
attention_side = None
