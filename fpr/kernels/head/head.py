"""Vocab-sharded LM heads of the step program (kernels/head).

VerifyHead  logits_r = h_all @ W_r^T [T, V/4] bf16: cuBLAS bf16, fp32 accumulation, one bf16 rounding -- the same GEMM
            (same shape per rank) production's compute_logits_local runs: bit-identical logits on captured steps.
DraftHead   base_r = hd_all @ W_r^T for the DSpark query rows: 'bf16' (= production at > 8 rows; bit-identical drafts on
            a capture) or 'e4m3' (W8A8: per-row e4m3 head + per-row e4m3 activations, fp32 scales, torch._scaled_mm rowwise;
            lossy, allowed for the drafter: verification stays exact; on a capture the slot-0 acceptance probability is
            0.8486 vs 0.8485 bf16).
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

E4M3_MAX = 448.0


@triton.jit
def _rowq_kernel(x_ptr, q_ptr, s_ptr, N, BLOCK: tl.constexpr):
    # per-row amax scale (fp32, amax / 448, floored at 1e-12) and e4m3(x / s), round-to-nearest-even, saturating
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + r * N + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / 448.0
    tl.store(s_ptr + r, s)
    y = tl.clamp(x / s, -448.0, 448.0)
    tl.store(q_ptr + r * N + offs, y.to(tl.float8e4nv), mask=m)


def quant_rows(x: torch.Tensor, q: torch.Tensor | None = None, s: torch.Tensor | None = None):
    M, N = x.shape
    q = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device) if q is None else q
    s = torch.empty(M, dtype=torch.float32, device=x.device) if s is None else s
    _rowq_kernel[(M,)](x, q, s, N, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return q, s


class VerifyHead:
    def __init__(self, w_slice: torch.Tensor, max_rows: int):
        self.w = w_slice.contiguous()
        self.out = torch.empty(max_rows, self.w.shape[0], dtype=torch.bfloat16, device=self.w.device)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        o = self.out[: h.shape[0]]
        torch.mm(h, self.w.t(), out=o)
        return o


class DraftHead:
    def __init__(self, w_slice: torch.Tensor, max_rows: int, mode: str = "bf16"):
        assert mode in ("bf16", "e4m3")
        self.mode = mode
        dev = w_slice.device
        self.out = torch.empty(max_rows, w_slice.shape[0], dtype=torch.bfloat16, device=dev)
        if mode == "bf16":
            self.w = w_slice.contiguous()
        else:
            ws = w_slice.float().abs().amax(-1).clamp_min(1e-12) / E4M3_MAX
            self.wq = (w_slice.float() / ws[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
            self.ws = ws[None, :].contiguous()
            self.xq = torch.empty(max_rows, w_slice.shape[1], dtype=torch.float8_e4m3fn, device=dev)
            self.xs = torch.empty(max_rows, dtype=torch.float32, device=dev)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        o = self.out[:M]
        if self.mode == "bf16":
            torch.mm(x, self.w.t(), out=o)
        else:
            q, s = quant_rows(x, self.xq[:M], self.xs[:M])
            torch._scaled_mm(q, self.wq.t(), scale_a=s[:, None], scale_b=self.ws, out_dtype=torch.bfloat16, out=o)
        return o
