"""Low-latency one-shot NVLink IPC all-reduce / all-gather for TP=4 (kernels in kernels/allreduce.cu).

Ideas adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/phased_all_reduce.py). See LICENSES/BRRR-Apache-2.0.txt.

`LLComm` owns one set of CUDA-IPC receive buffers and one device epoch. Every rank must issue the
same sequence of calls on one instance (same order, same shapes), on one stream at a time, exactly
like any other collective. Separate users (the patched CudaCommunicator, DSpark sampling gathers)
use separate instances so their call sequences never interleave.

All entry points are CUDA-graph safe: no host synchronization, no host-side state that changes
between replays (the epoch lives in device memory), and the only allocation is the output tensor
(served from the graph pool during capture, like stock vLLM collectives).
"""

from __future__ import annotations

import logging
import math
import os
from functools import cache

import torch
import torch.distributed as dist

from ds41h import cuext

logger = logging.getLogger("ds41h.allreduce")

RANKS = 4
THREADS = 512
LINE_BYTES = 8  # payload bytes per 16-byte LL line


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


# Per-rank input-size limits for routing stock collectives here (tune with tests/test_allreduce.py --bench).
AR_MAX_BYTES = _env_int("DS41H_AR_MAX_BYTES", 512 << 10)
AG_MAX_BYTES = _env_int("DS41H_AG_MAX_BYTES", 512 << 10)
# Receive-buffer capacity per (parity, source) slot in payload bytes: 2 * 4 * 2 * this bytes of IPC memory.
BUF_BYTES = _env_int("DS41H_AR_BUF_BYTES", 2 << 20)
CTAS_CAP = _env_int("DS41H_AR_CTAS", 64)  # upper bound on CTAs per call (each CTA is 512 threads)
LINES_PER_THREAD = _env_int("DS41H_AR_LINES_PER_THREAD", 1)
PDL = os.environ.get("DS41H_AR_PDL", "1") not in ("0", "", "false")
TIMEOUT_NS = int(float(os.environ.get("DS41H_AR_TIMEOUT_S", "0")) * 1e9)  # 0 = spin forever (like NCCL)


@cache
def ext():
    return cuext.load("ds41h_allreduce", "allreduce.cu")


def ctas_for(lines: int) -> int:
    """CTAs for `lines` LL packets: one packet per thread up to the cap, then several per thread."""
    per_cta = THREADS * max(1, LINES_PER_THREAD)
    return max(1, min(CTAS_CAP, ext().max_ctas(), -(-lines // per_cta)))


def inner_words(t: torch.Tensor, dim: int) -> int | None:
    """Words (4 bytes) of one contiguous run along `dim` (prod(shape[dim:]) * itemsize / 4)."""
    nbytes = math.prod(t.shape[dim:]) * t.element_size()
    return nbytes // 4 if nbytes % 4 == 0 and nbytes > 0 else None


class LLComm:
    """One-shot LL all-reduce / all-gather over CUDA IPC for a 4-rank group.

    Collective constructor: every rank of `cpu_group` must call it at the same point.
    """

    def __init__(self, cpu_group, device: torch.device | int, max_bytes: int = BUF_BYTES, name: str = "ds41h") -> None:
        self.name = name
        self.handle: int | None = None
        self.world = dist.get_world_size(group=cpu_group)
        self.rank = dist.get_rank(group=cpu_group)
        self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        self.disabled = self.world != RANKS
        self.max_bytes = 0
        self._own: int | None = None
        self._opened: list[int] = []
        if self.disabled:
            return
        self.slot_lines = max(1, -(-max_bytes // LINE_BYTES))
        self.max_bytes = self.slot_lines * LINE_BYTES
        # Every routing decision taken later must be identical on all ranks, so a local failure
        # (extension build/load, cudaMalloc, cudaIpcOpenMemHandle, ...) must disable the instance on
        # *every* rank. Each phase therefore catches its own error and the ranks agree through a
        # collective that every rank reaches whatever happened locally (a rank that raised out of
        # here before the exchange would leave its peers blocked in all_gather_object, and a rank
        # that silently fell back to stock while its peers use the LL kernels would deadlock).
        e = None
        mine: list[int] | None = None
        try:
            e = ext()
            assert e.ranks() == RANKS and e.threads() == THREADS
            with torch.cuda.device(self.device):
                mine = e.alloc_ipc(e.buffer_bytes(self.slot_lines))  # zeroed + device-synchronized
            self._own = mine[0]
        except Exception:  # noqa: BLE001 - reported, then agreed on below
            logger.exception("ds41h %s: LL comm buffer allocation failed on rank %d", name, self.rank)
        handles: list = [None] * self.world
        dist.all_gather_object(handles, None if mine is None else mine[1:], group=cpu_group)
        ok = mine is not None  # this rank's own setup status (reported in the agreement below)
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
                    self.handle = e.create(ptrs, self.rank, self.slot_lines)
            except Exception:  # noqa: BLE001
                logger.exception("ds41h %s: LL comm IPC setup failed on rank %d", name, self.rank)
                ok = False
        # Agreement + barrier: no rank pushes into a peer buffer before every peer finished setup
        # (buffers are zeroed before the handle exchange, the epoch state inside create()).
        oks: list = [None] * self.world
        dist.all_gather_object(oks, ok, group=cpu_group)
        if not all(oks) or self.handle is None:
            logger.warning("ds41h %s: LL comm disabled on all ranks (setup failed on ranks %s); using stock collectives",
                           name, [r for r, v in enumerate(oks) if not v])
            self._release(group=cpu_group)
            self.disabled = True
            return
        logger.info(
            "ds41h %s: LL comm rank %d/%d, %d KiB per slot (%.1f MiB IPC)",
            name,
            self.rank,
            self.world,
            self.max_bytes >> 10,
            e.buffer_bytes(self.slot_lines) / 2**20,
        )

    # ------------------------------------------------------------------ eligibility
    def can_all_reduce(self, t: torch.Tensor, max_bytes: int | None = None) -> bool:
        limit = min(self.max_bytes if not self.disabled else 0, AR_MAX_BYTES if max_bytes is None else max_bytes)
        return (
            not self.disabled
            and t.is_cuda
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            and 0 < t.nbytes <= limit
            and t.nbytes % LINE_BYTES == 0
            and t.device == self.device
        )  # no data_ptr test: routing must be identical on every rank (all_reduce copies a misaligned input)

    def can_all_gather(self, t: torch.Tensor, dim: int, max_bytes: int | None = None) -> bool:
        if self.disabled or not t.is_cuda or t.device != self.device or not t.is_contiguous() or t.dim() == 0:
            return False
        limit = min(self.max_bytes, AG_MAX_BYTES if max_bytes is None else max_bytes)
        dim %= t.dim()
        # no data_ptr test (rank-invariant routing): all_gather copies a misaligned input
        return 0 < t.nbytes <= limit and t.nbytes % 4 == 0 and inner_words(t, dim) is not None

    # ------------------------------------------------------------------ collectives
    def all_reduce(self, t: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Sum over ranks (bf16 in/out, fp32 accumulation in rank order 0..3). `out` may alias `t`."""
        assert self.handle is not None, f"LLComm {self.name} is disabled"
        if t.data_ptr() % LINE_BYTES:
            t = t.clone()  # fresh allocation: 512-byte aligned
        if out is None:
            out = torch.empty_like(t)
        if out.data_ptr() % LINE_BYTES:
            tmp = self.all_reduce(t)
            out.copy_(tmp)
            return out
        lines = t.nbytes // LINE_BYTES
        ext().all_reduce(self.handle, t, out, ctas_for(lines), PDL, TIMEOUT_NS)
        return out

    def all_gather(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenate the ranks' tensors along `dim` (stock CudaCommunicator.all_gather layout)."""
        assert self.handle is not None, f"LLComm {self.name} is disabled"
        dim %= t.dim()
        if t.data_ptr() % 4:
            t = t.clone()
        shape = list(t.shape)
        shape[dim] *= self.world
        out = torch.empty(shape, dtype=t.dtype, device=t.device)
        inner = inner_words(t, dim)
        assert inner is not None
        lines = -(-(t.nbytes // 4) // 2)
        ext().all_gather(self.handle, t, out, inner, ctas_for(lines), PDL, TIMEOUT_NS)
        return out

    def gather_words(self, t: torch.Tensor, out: torch.Tensor | None = None, early: bool = False) -> torch.Tensor:
        """Stack the ranks' tensors: returns [RANKS, *t.shape] (4-byte dtypes, small sizes). early: release the PDL dependent
        right after the kernel's own PDL wait (the next kernel must be PDL-launched and wait before reading `out`)."""
        assert self.handle is not None, f"LLComm {self.name} is disabled"
        assert t.element_size() == 4 and t.is_contiguous()
        if out is None:
            out = t.new_empty((self.world, *t.shape))
        words = t.numel()
        lines = -(-words // 2)
        e = ext()
        fn = e.all_gather_early if (early and PDL and hasattr(e, "all_gather_early")) else e.all_gather
        fn(self.handle, t, out, max(words, 1), ctas_for(lines), PDL, TIMEOUT_NS)
        return out

    def _release(self, group=None) -> None:
        """Free whatever part of the setup exists (also after a partial setup failure).

        With `group`, every rank first unmaps its peers' buffers and the ranks synchronize before any
        rank frees its own exported buffer (cudaFree of memory still opened by an importer is UB)."""
        try:
            if self.handle is not None or self._opened:
                e = ext()
                with torch.cuda.device(self.device):
                    if self.handle is not None:
                        e.destroy(self.handle)
                    for ptr in self._opened:
                        e.close_ipc(ptr)
        except Exception:  # noqa: BLE001 - must still reach the barrier below
            if group is None:
                raise
            logger.exception("ds41h %s: LL comm teardown failed on rank %d", self.name, self.rank)
        self.handle, self._opened = None, []
        if group is not None:
            dist.barrier(group=group)
        if self._own is not None:
            own, self._own = self._own, None
            with torch.cuda.device(self.device):
                ext().free_buffer(own)

    def close(self) -> None:
        self._release()
        self.disabled = True


_SAMPLING_COMM: LLComm | None = None


def sampling_comm() -> LLComm:
    """Process-wide LL instance for DSpark vocab-parallel word gathers (TP group). Collective on first call."""
    global _SAMPLING_COMM
    if _SAMPLING_COMM is None:
        from ds41h.patch_core import tp_cpu_group, tp_device

        _SAMPLING_COMM = LLComm(tp_cpu_group(), tp_device(), max_bytes=256 << 10, name="sampling")
    return _SAMPLING_COMM
