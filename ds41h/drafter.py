"""ds41h drafter: fused DSpark drafting kernels for TP=4 (kernels/drafter.cu).

MarkovFused  -- the whole sequential Markov sampling loop of DSparkSpeculator._sample_sequential (all n_spec
                steps) in one persistent launch with an in-kernel cross-rank winner exchange over CUDA IPC.
                Same math as ds41h.vp_sampling's per-step Triton kernels (which reproduce stock bitwise):
                draft tokens, cached draft logits, token logit and log-sum-exp per (request, step).
fp8_head     -- an e4m3 copy of the vocab-parallel lm_head shard (one fp32 scale per row) for the DRAFT head
                only (drafter.cu head_kernel, half the bytes of the bf16 head). Verification still uses
                the bf16 head (exact target distribution); drafting from a slightly different q is still an
                exact speculative sampler because the cached draft logits are the ones sampled from.

Environment
  DS41H_DRAFT_FUSED=1          use MarkovFused when eligible (default 1)
  DS41H_DRAFT_FUSED_MAX_ROWS=4 requests per draft step served by the fused kernel (more -> Triton path; kernel
                               limit 4). Graph bench (bench8): rows 1/2/4 fused 44/64/104 us vs Triton+LL 101/105/117
  DS41H_DRAFT_HEAD_FP8=1       FP8 draft head (default 1: end-to-end accept 5.16/5.22 vs 5.12/5.11 bf16, KL PASS,
                               59 us vs 106 us in-situ; changes the draft distribution only, verification stays exact)
  DS41H_VERIFY_FUSED=0         fused verification kernel when eligible (default 0: exact either way, but in end-to-end/dr4b
                               the fused kernel took 100 us vs 74 us for the Triton + LL-gather chain on the critical rank)
  DS41H_DRAFT_TIMEOUT_S=0      device-side spin timeout (debug)
  DS41H_DRAFTER_MOE2P=0        persistent moe2 (h fragments read once per CTA; bitwise-equal fused AR; slower in the
                               isolated bench: 67.4 vs 62.6 us at M=5 + shared)
MoEDecode.route (DS41H_DRAFTER_ROUTE1 in patch_drafter, default 0): one-launch router, 7.6 us vs stock 5.6 us isolated.

The extension is built WITHOUT --use_fast_math (accurate logf / expf, IEEE adds) so the Gumbel noise matches
the Triton/libdevice implementation bit for bit.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger("ds41h.drafter")

RANKS = 4
FUSED = os.environ.get("DS41H_DRAFT_FUSED", "1") not in ("0", "", "false")
FUSED_MAX_ROWS = int(os.environ.get("DS41H_DRAFT_FUSED_MAX_ROWS", "4"))  # bench8: fused wins at 1/2/4 rows
VERIFY_FUSED = os.environ.get("DS41H_VERIFY_FUSED", "0") not in ("0", "", "false")
HEAD_FP8 = os.environ.get("DS41H_DRAFT_HEAD_FP8", "1") not in ("0", "", "false")
TIMEOUT_NS = int(float(os.environ.get("DS41H_DRAFT_TIMEOUT_S", "0")) * 1e9)
PDL = os.environ.get("DS41H_DRAFT_PDL", "1") not in ("0", "", "false")
# DS41H_DRAFT_GROUPS=N (0 = rank-level last-CTA merge): the fused Markov loop's per-step exchange through N CTA groups (the
# last CTA of each group pushes its group candidate to every rank; no rank-wide 127-way arrival counter + merge)
DRAFT_GROUPS = int(os.environ.get("DS41H_DRAFT_GROUPS", "0") or 0)
MOE2P = os.environ.get("DS41H_DRAFTER_MOE2P", "0") not in ("0", "", "false")  # persistent moe2 (h fragments read once per CTA)
MOE2_VARIANT = int(os.environ.get("DS41H_DRAFTER_MOE2_VARIANT", "0") or 0)  # 1: slot-major moe2s, 2: persistent moe2r (drafter.cu reads it too)
CHUNK = 256
SLOT_ROWS = 64


@functools.cache
def ext():
    from pathlib import Path

    from torch.utils.cpp_extension import load

    kernels = Path(__file__).parent / "kernels"
    flags = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-O3"]
    mod = load(
        name="ds41h_drafter",
        sources=[str(kernels / "drafter.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        extra_include_paths=[str(kernels)],
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")),
    )
    if hasattr(mod, "preload") and torch.cuda.is_available() and torch.cuda.is_initialized():
        n = mod.preload()  # no lazy kernel loading inside the warmup / capture sequence (cross-GPU waiting kernels)
        logger.info("ds41h drafter: preloaded %d kernel functions", n)
    if hasattr(mod, "lin_set_stage"):
        # DS41H_DRAFTER_LIN_S1 (default 1): o-projection x staging in one pass (bitwise identical to the two-pass staging,
        # one global round trip after the PDL wait instead of a dependent inverse-RoPE chain); 0 = the old two-pass staging
        mod.lin_set_stage(1 if os.environ.get("DS41H_DRAFTER_LIN_S1", "1") not in ("0", "", "false") else 0)
    if hasattr(mod, "lin_set_early"):
        # DS41H_DRAFTER_LIN_EARLY=1: wo_a (one-pass staging) triggers wo_b's PDL launch right after its own PDL wait, so wo_b's
        # weight stream overlaps wo_a (same kernels and math; only the launch overlap changes)
        mod.lin_set_early(1 if os.environ.get("DS41H_DRAFTER_LIN_EARLY", "0") not in ("0", "", "false") else 0)
    if hasattr(mod, "head_set_rows"):
        # DS41H_DRAFT_HEAD_ROWS=1: FP8 draft head as the row-stream kernel (CTA-wide cp.async ring; same math, k-split across warps)
        mod.head_set_rows(1 if os.environ.get("DS41H_DRAFT_HEAD_ROWS", "0") not in ("0", "", "false") else 0)
    return mod


class DraftComm:
    """CUDA-IPC receive buffers + device state for the fused drafting kernel (collective constructor).

    Setup failures on any rank disable the instance on every rank (same agreement protocol as
    ds41h.allreduce.LLComm), so all ranks take the same (fused or Triton) path."""

    def __init__(self, cpu_group, device, max_ctas: int = 132, slot_rows: int = SLOT_ROWS, name: str = "draft") -> None:
        import torch.distributed as dist

        self.name = name
        self.world = dist.get_world_size(group=cpu_group)
        self.rank = dist.get_rank(group=cpu_group)
        self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        self.handle: int | None = None
        self.slot_rows = slot_rows
        self._own: int | None = None
        self._opened: list[int] = []
        self.disabled = self.world != RANKS
        if self.disabled:
            return
        e = None
        mine = None
        try:
            e = ext()
            with torch.cuda.device(self.device):
                mine = e.alloc_ipc(e.recv_bytes(slot_rows))
            self._own = mine[0]
        except Exception:  # noqa: BLE001 - agreed on below
            logger.exception("ds41h %s: buffer allocation failed on rank %d", name, self.rank)
        handles: list = [None] * self.world
        dist.all_gather_object(handles, None if mine is None else mine[1:], group=cpu_group)
        ok = mine is not None
        if all(h is not None for h in handles):
            try:
                with torch.cuda.device(self.device):
                    ptrs = []
                    for peer in range(self.world):
                        if peer == self.rank:
                            ptrs.append(mine[0])
                        else:
                            ptr = e.open_ipc(handles[peer])
                            self._opened.append(ptr)
                            ptrs.append(ptr)
                    self.handle = e.create(ptrs, self.rank, slot_rows, max_ctas)
            except Exception:  # noqa: BLE001
                logger.exception("ds41h %s: IPC setup failed on rank %d", name, self.rank)
                ok = False
        oks: list = [None] * self.world
        dist.all_gather_object(oks, ok, group=cpu_group)
        if not all(oks) or self.handle is None:
            logger.warning("ds41h %s: fused drafting disabled on all ranks (setup failed on ranks %s)", name,
                           [r for r, v in enumerate(oks) if not v])
            self._release(cpu_group)
            self.disabled = True
            return
        logger.info("ds41h %s: fused drafting comm rank %d/%d (%d slot rows)", name, self.rank, self.world, slot_rows)

    def _release(self, group=None) -> None:
        import torch.distributed as dist

        try:
            e = ext()
            with torch.cuda.device(self.device):
                if self.handle is not None:
                    e.destroy(self.handle)
                for ptr in self._opened:
                    e.close_ipc(ptr)
        except Exception:  # noqa: BLE001
            if group is None:
                raise
            logger.exception("ds41h %s: teardown failed on rank %d", self.name, self.rank)
        self.handle, self._opened = None, []
        if group is not None:
            dist.barrier(group=group)
        if self._own is not None:
            own, self._own = self._own, None
            with torch.cuda.device(self.device):
                ext().free_buffer(own)


class MarkovFused:
    """Fused sequential Markov drafting for one VPShard (see module docstring)."""

    def __init__(self, comm: DraftComm, shard) -> None:
        assert not comm.disabled and comm.handle is not None
        self.comm = comm
        self.shard = shard
        self.ctas = -(-shard.local // CHUNK)
        assert shard.w1.shape[1] == ext().markov_rank() and not shard.w2_fp8 and shard.w2.dtype == torch.bfloat16

    def eligible(self, num_reqs: int, n_spec: int) -> bool:
        return (
            0 < num_reqs <= min(FUSED_MAX_ROWS, 4)  # kernels/drafter.cu DRAFT_ROWS
            and n_spec <= ext().max_steps()
            and self.shard.logits.dtype == torch.bfloat16
        )

    def __call__(self, base, input_ids, anchor_stride: int, idx_map, sample_pos, temperature, seeds, draft_tokens, num_reqs: int, n_spec: int,
                 trace: torch.Tensor | None = None, req_draft: torch.Tensor | None = None):
        """base [num_reqs * n_spec, local] bf16 (row = r * n_spec + s); idx_map / sample_pos [num_reqs * n_spec] int64.
        req_draft: optional [max_reqs, >= n_spec] int64 (the runner's req_states.draft_tokens) also written at the
        request-state index (DS41H_DRAFT_TAIL_EARLY_COMBINE)."""
        s = self.shard
        ext().draft_markov(
            self.comm.handle,
            base,
            s.w1,
            s.w2,
            input_ids,
            anchor_stride,
            idx_map,
            sample_pos,
            temperature,
            seeds,
            draft_tokens,
            s.logits,
            s.token_logit,
            s.lse,
            num_reqs,
            n_spec,
            s.start,
            CHUNK,
            TIMEOUT_NS,
            PDL,
            trace,
            DRAFT_GROUPS,
            req_draft,
        )


class VerifyFused:
    """Fused vocab-parallel block verification (kernels/drafter.cu verify_kernel): one launch replacing
    vp_sampling.verify's 6 launches (stage, stats, gather, residual, gather, reject)."""

    MAX_ROWS = 64  # kernels/drafter.cu dver::MAXROWS
    MAX_REQS = 64

    def __init__(self, comm: DraftComm, shard) -> None:
        assert not comm.disabled and comm.handle is not None
        self.comm = comm
        self.shard = shard

    def eligible(self, logits: torch.Tensor, num_reqs: int) -> bool:
        return (
            logits.dtype == torch.bfloat16
            and logits.dim() == 2
            and logits.stride(1) == 1
            and logits.shape[1] == self.shard.local
            and 0 < logits.shape[0] <= min(self.MAX_ROWS, self.comm.slot_rows)
            and 0 < num_reqs <= self.MAX_REQS
            and self.shard.logits.dtype == torch.bfloat16
        )

    def __call__(self, logits, draft_sampled, pos, expanded_idx, expanded_local_pos, cu_num_logits, idx_mapping, num_reqs, temperature, seeds,
                 trace=None):
        s = self.shard

        def i32(t):
            return t if t.dtype == torch.int32 and t.is_contiguous() else t.to(torch.int32).contiguous()

        sampled = torch.empty((num_reqs, s.n_spec + 1), dtype=torch.int64, device=logits.device)
        num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
        ext().verify_fused(
            self.comm.handle,
            logits,
            i32(draft_sampled),
            pos if pos.dtype == torch.int64 and pos.is_contiguous() else pos.to(torch.int64).contiguous(),
            i32(expanded_idx),
            i32(expanded_local_pos),
            i32(cu_num_logits[: num_reqs + 1]),
            i32(idx_mapping[:num_reqs]),
            temperature,
            seeds,
            s.logits,
            s.token_logit,
            s.lse,
            sampled,
            num_sampled,
            s.start,
            s.n_spec,
            CHUNK,
            TIMEOUT_NS,
            trace,
        )
        return sampled, num_sampled


_COMMS: dict[str, DraftComm] = {}


def draft_comm(name: str = "draft") -> DraftComm:
    """Process-wide DraftComm per use ("draft", "verify") for the TP group (collective on first call)."""
    if name not in _COMMS:
        from ds41h.patch_core import tp_cpu_group, tp_device

        _COMMS[name] = DraftComm(tp_cpu_group(), tp_device(), name=name)
    return _COMMS[name]


# ------------------------------------------------------------------------------------------- FP8 draft head
class HeadFP8:
    """e4m3 copy of a bf16 lm_head shard [N, 5120] with one fp32 scale per row (amax / 448), in skinny
    fragment-major layout, for the DRAFT head only (kernels/drafter.cu head_kernel: <= 8 rows, one 16-byte
    coalesced load per lane per k32 block, HMMA accumulation over the full K). Half the bytes of the bf16 head."""

    MAX_ROWS = 8

    def __init__(self, weight: torch.Tensor) -> None:
        n, k = weight.shape
        assert n % 16 == 0 and k == 5120
        w = weight.detach()
        scale = torch.empty(n, dtype=torch.float32, device=w.device)
        packed = torch.empty(n * k, dtype=torch.uint8, device=w.device)
        step = 2048
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            blk = w[r0:r1].float()
            sc = blk.abs().amax(dim=1).clamp_min(1e-30) / 448.0
            q = (blk / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
            # fragment-major (ds41h/skinny.py pack): (tile, kb, g, t, half, 8 bytes)
            frag = q.view((r1 - r0) // 16, 2, 8, k // 32, 4, 8).permute(0, 3, 2, 4, 1, 5).contiguous().view(-1)
            packed[r0 * k : r1 * k] = frag
            scale[r0:r1] = sc
        self.w, self.scale, self.n, self.k = packed, scale, n, k

    def eligible(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= self.MAX_ROWS
            and x.shape[1] == self.k
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x: torch.Tensor, norm_weight: torch.Tensor | None = None, eps: float = 0.0) -> torch.Tensor:
        """y = head(x), or with norm_weight y = head(rms_norm(x)) with the RMSNorm fused into the kernel's staging
        (vLLM rms_norm rounding: bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w))."""
        y = torch.empty(x.shape[0], self.n, dtype=torch.bfloat16, device=x.device)
        ext().head_fp8(x, self.w, self.scale, y, PDL, norm_weight, float(eps))
        return y

    @staticmethod
    def norm_fusable(norm) -> bool:
        w = getattr(norm, "weight", None)
        return (
            isinstance(w, torch.Tensor)
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
            and w.numel() == 5120
            and w.data_ptr() % 16 == 0
            and isinstance(getattr(norm, "variance_epsilon", None), float)
            and not getattr(norm, "has_bias", False)
            and getattr(norm, "has_weight", True)
        )


def fp8_head(weight: torch.Tensor) -> HeadFP8:
    return HeadFP8(weight)


# ------------------------------------------------------------------------------------------- drafter MoE
class MoEDecode:
    """Routed-expert decode path for M <= 8 tokens on the raw TP-sliced expert layout (kernels/drafter.cu
    moe13 + moe2): returns this rank's routed partial sum [M, 5120] bf16 (no shared expert, no all-reduce),
    the same contract as ds41h.moe.moe_forward."""

    MAX_TOKENS = 8

    def __init__(self, device) -> None:
        e = ext()
        self.hb = torch.zeros(e.moe_hb_halves(), dtype=torch.float16, device=device)
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.moe2p = MOE2P and self.ctas * 3 >= 320
        self.glog = torch.zeros(8 * 512, dtype=torch.float32, device=device)  # fused-router logits scratch
        self.gbar = torch.zeros(1, dtype=torch.int32, device=device)  # fused-router grid barrier (monotonic)
        self.rscratch = torch.zeros(e.router_scratch_floats(), dtype=torch.float32, device=device)  # router: counter + logits
        # slot-major moe2s (DS41H_DRAFTER_MOE2_VARIANT=1): fp32 slot partials + per-chunk counters (zeroed: counters self-reset)
        self.m2s = torch.zeros(e.moe2s_scratch_floats(), dtype=torch.float32, device=device) if MOE2_VARIANT in (1, 2) else None

    @staticmethod
    def route_eligible(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor | None, topk: int) -> bool:
        """Shape/dtype-only (rank-invariant) eligibility of the one-launch router (kernels/drafter.cu drouter)."""
        return (
            bias is not None
            and gate_w.dtype == torch.bfloat16
            and gate_w.dim() == 2
            and gate_w.shape[1] == 5120
            and gate_w.shape[0] <= 512
            and gate_w.is_contiguous()
            and bias.dtype == torch.float32
            and bias.is_contiguous()
            and bias.numel() == gate_w.shape[0]
            and 0 < topk <= 16
            and x.dim() == 2
            and 0 < x.shape[0] <= MoEDecode.MAX_TOKENS
        )

    def route(self, x, gate_w, bias, topk: int, scale: float, renormalize: bool, bias_vl=None, input_ids=None, image_lo: int = 0):
        """Router logits (fp32, like the stock ll_bf16 GEMM output) + stock dsv4_topk routing in one launch.
        Returns (topk_weights fp32 [M, k], topk_ids int32 [M, k])."""
        x = self.aligned(x)
        m = x.shape[0]
        ids = torch.empty(m, topk, dtype=torch.int32, device=x.device)
        wts = torch.empty(m, topk, dtype=torch.float32, device=x.device)
        if bias_vl is not None and input_ids is not None and image_lo > 0:
            if bias_vl.dtype != torch.float32 or not bias_vl.is_contiguous():
                bias_vl = bias_vl.float().contiguous()
            input_ids = input_ids.reshape(-1)
            if input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_contiguous():
                input_ids = input_ids.to(torch.int64).contiguous()
        else:
            bias_vl, input_ids, image_lo = None, None, 0
        ext().router(x, gate_w, bias, bias_vl, input_ids, int(image_lo), float(scale), bool(renormalize), self.rscratch, ids, wts, PDL)
        return wts, ids

    @staticmethod
    def eligible(x: torch.Tensor, topk_ids: torch.Tensor) -> bool:
        """Shape / dtype only (identical on every TP rank: the fused-AR variant is a collective). Layout
        (strides, 16-byte alignment) is fixed up by `aligned` instead of falling back."""
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= MoEDecode.MAX_TOKENS
            and x.shape[1] == 5120
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and topk_ids.shape[0] == x.shape[0]
            and topk_ids.shape[1] <= 16
        )

    @staticmethod
    def aligned(x: torch.Tensor) -> torch.Tensor:
        """x itself if the kernels can read it (unit column stride, 8-element row stride, 16-byte base), else a
        contiguous copy (a fresh allocation is 512-byte aligned)."""
        if x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0:
            return x
        return x.contiguous() if not x.is_contiguous() else x.clone()

    def __call__(self, x, topk_weights, topk_ids, w13, s13, w2, s2, swiglu_limit, shared: "SharedPack | None" = None,
                 ar: "DraftComm | None" = None) -> torch.Tensor:
        """Routed partial sum (+ the shared expert's output when `shared` is given) of this rank, bf16 [M, 5120];
        with `ar`, the TP all-reduce is fused into moe2 (output already summed over the 4 ranks)."""
        limit = float(swiglu_limit) if swiglu_limit is not None and float(swiglu_limit) > 0 else float("inf")
        x = self.aligned(x)
        ids = topk_ids if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous() else topk_ids.to(torch.int32).contiguous()
        wts = topk_weights if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous() else topk_weights.float().contiguous()
        y = torch.empty(x.shape[0], 5120, dtype=torch.bfloat16, device=x.device)
        sh = (None, None, None, None) if shared is None else (shared.sh13, shared.ss13, shared.sh2, shared.ss2)
        ext().moe_decode(x, ids, wts, w13, s13, w2, s2, self.hb, y, limit, self.ctas, PDL, self.m2s, *sh, None, None, None, None, 1.0, True,
                         -1 if ar is None else ar.handle, TIMEOUT_NS, self.ctas if self.moe2p else 0)
        return y

    def routed(self, x, gate_w, gate_bias, topk: int, scale: float, renormalize: bool, w13, s13, w2, s2, swiglu_limit,
               shared: "SharedPack | None" = None, ar: "DraftComm | None" = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused router + experts: router logits (gate_w bf16 [E, 5120]), stock dsv4_topk routing (sqrtsoftplus + bias,
        top-k, renormalize * scale) computed inside moe13, then the routed (+ shared) experts.
        Returns (y bf16 [M, 5120], topk_weights fp32 [M, k], topk_ids int32 [M, k])."""
        limit = float(swiglu_limit) if swiglu_limit is not None and float(swiglu_limit) > 0 else float("inf")
        x = self.aligned(x)
        m = x.shape[0]
        ids = torch.empty(m, topk, dtype=torch.int32, device=x.device)
        wts = torch.empty(m, topk, dtype=torch.float32, device=x.device)
        y = torch.empty(m, 5120, dtype=torch.bfloat16, device=x.device)
        sh = (None, None, None, None) if shared is None else (shared.sh13, shared.ss13, shared.sh2, shared.ss2)
        ext().moe_decode(x, ids, wts, w13, s13, w2, s2, self.hb, y, limit, self.ctas, PDL, self.m2s, *sh, gate_w, gate_bias, self.glog, self.gbar,
                         float(scale), bool(renormalize), -1 if ar is None else ar.handle, TIMEOUT_NS, self.ctas if self.moe2p else 0)
        return y, wts, ids


P_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)  # k order inside each 8-group of a fragment (kernels/drafter.cu dmoe)


def _frag_pack(w: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes [N, K] -> fragment-major uint8 (16-row tile, k32 block, lane 4g+t: row g 8 bytes, row g+8 8 bytes)
    with the P order applied inside every 8-byte k group (the kernels' activation fragment order)."""
    n, k = w.shape
    idx = torch.tensor(P_ORDER, device=w.device)
    wp = w.view(n, k // 8, 8).index_select(2, idx).reshape(n, k)
    return wp.view(n // 16, 2, 8, k // 32, 4, 8).permute(0, 3, 2, 4, 1, 5).contiguous().view(-1)


class SharedPack:
    """The shared expert of one MoE layer (FP8 e4m3, 32x32 ue8m0 blocks, TP slice I = 576) in the layout of
    the drafter MoE kernels: w13 tiles pair gate rows 8p..8p+7 with up rows 8p..8p+7 (HMMA rows g / g+8), built
    from the skinny copies the prologue component captures (skinny.unpack gives the checkpoint bytes)."""

    def __init__(self, sw13, sw2) -> None:
        from ds41h import skinny

        w13, s13 = skinny.unpack(sw13)  # [1152, 5120] e4m3 bytes, [72, 160] per-16-row tile scales
        w2, s2 = skinny.unpack(sw2)  # [5120, 576], [320, 18]
        inter = w13.shape[0] // 2
        assert w13.shape == (2 * inter, 5120) and inter == 576 and w2.shape == (5120, inter), (w13.shape, w2.shape)
        dev = w13.device
        p = torch.arange(inter // 8, device=dev)
        gate_rows = p[:, None] * 8 + torch.arange(8, device=dev)
        rows = torch.cat([gate_rows, gate_rows + inter], 1).reshape(-1)
        self.sh13 = _frag_pack(w13.index_select(0, rows).contiguous())
        g_sc = s13.index_select(0, (8 * p) // 16)
        u_sc = s13.index_select(0, (inter + 8 * p) // 16)
        self.ss13 = torch.stack([g_sc, u_sc], -1).reshape(inter // 8, -1).contiguous()  # u16 = gate | up << 8
        self.sh2 = _frag_pack(w2.contiguous())
        self.ss2 = s2.contiguous()


_MOE: dict = {}


def moe_decode_op(device) -> MoEDecode:
    key = str(device)
    if key not in _MOE:
        _MOE[key] = MoEDecode(device)
    return _MOE[key]


# ------------------------------------------------------------------------------------------- o-projection
class OProj:
    """Drafter-layer o-projection for <= 8 rows (kernels/drafter.cu dlin::lin_kernel, two PDL launches): wo_a over the
    rank's 2 head groups with the inverse RoPE fused into the activation staging, then wo_b; both FP8 skinny-packed with
    exact 32x32 ue8m0 scales (stock: Triton inverse RoPE + bf16 bmm of the dequantized wo_a + skinny/Humming wo_b).
    Returns the rank's partial (the caller all-reduces)."""

    MAX_ROWS = 8

    def __init__(self, wa, wb, n_heads: int, groups: int, ar: "DraftComm | None" = None) -> None:
        assert wa.k == 4096 and wa.n == groups * 1024 and wb.k == groups * 1024 and wb.n == 5120, (wa.n, wa.k, wb.n, wb.k)
        assert n_heads * 512 == groups * 4096
        self.wa, self.wb, self.n_heads, self.groups = wa, wb, n_heads, groups
        self.ar = ar  # fused TP all-reduce in the wo_b launch (output already summed over the 4 ranks)

    def eligible(self, o: torch.Tensor, positions: torch.Tensor) -> bool:
        return (
            o.dim() == 3
            and o.dtype == torch.bfloat16
            and 0 < o.shape[0] <= self.MAX_ROWS
            and o.shape[1] >= self.n_heads
            and o.shape[2] == 512
            and o.stride(2) == 1
            and o.stride(1) == 512
            and o.stride(0) % 8 == 0
            and o.data_ptr() % 16 == 0
            and positions.dim() == 1
            and positions.shape[0] >= o.shape[0]
        )

    def __call__(self, o: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
        m = o.shape[0]
        e = ext()
        pos = positions if positions.dtype == torch.int64 and positions.is_contiguous() else positions.to(torch.int64).contiguous()
        z = torch.empty(m, self.wa.n, dtype=torch.bfloat16, device=o.device)
        e.lin_fp8(o, 4096, self.groups, self.wa.w, self.wa.s, self.wa.n, self.wa.k, z, pos, cos_sin_cache, PDL, -1, 0)
        y = torch.empty(m, self.wb.n, dtype=torch.bfloat16, device=o.device)
        e.lin_fp8(z, 0, 1, self.wb.w, self.wb.s, self.wb.n, self.wb.k, y, None, None, PDL, -1 if self.ar is None else self.ar.handle, TIMEOUT_NS)
        return y


# ------------------------------------------------------------------------------------------- main_proj
class MainProj:
    """DSpark main_proj (mtp.0.main_proj: e4m3 [5120, 15360], 32x32 ue8m0 blocks) for <= 8 rows: one launch of
    kernels/drafter.cu main_kernel (3 K slices x 40 tile groups, deterministic slice sum) instead of the stock
    Humming W8A16 GEMM. Weights are read from the checkpoint (stock repacks its copy for Humming)."""

    MAX_ROWS = 8

    def __init__(self, device, weight: torch.Tensor | None = None, scale: torch.Tensor | None = None) -> None:
        from ds41h import skinny

        if weight is None or scale is None:
            from ds41h.ref.weights import default_reader

            rd = default_reader()
            weight = rd.get("mtp.0.main_proj.weight")
            scale = rd.get("mtp.0.main_proj.scale")
        w = weight.to(device)
        sc = scale.view(torch.uint8).to(device) if scale.element_size() == 1 else scale.to(device)
        assert w.shape == (5120, 15360), w.shape
        sw = skinny.pack(w, sc)
        self.w, self.s = sw.w, sw.s
        self.work = torch.zeros(ext().main_part_floats(), dtype=torch.float32, device=device)

    @staticmethod
    def eligible(x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= MainProj.MAX_ROWS
            and x.shape[1] == 15360
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty(x.shape[0], 5120, dtype=torch.bfloat16, device=x.device)
        ext().main_proj(x, self.w, self.s, self.work, y, PDL)
        return y
