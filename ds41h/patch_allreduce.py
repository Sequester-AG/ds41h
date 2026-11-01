"""ALLREDUCE hook: route small TP all-reduces / all-gathers through ds41h.allreduce.LLComm.

Patched symbols (vllm/distributed/device_communicators/cuda_communicator.py):
  * CudaCommunicator.__init__   (:36)  - after stock init, builds an LLComm for the 4-rank "tp" group.
  * CudaCommunicator.all_reduce (:362) - bf16, contiguous, <= DS41H_AR_MAX_BYTES -> LLComm; else stock
    (stock order: FlashInfer trtllm IPC <= 2 MiB, vLLM custom AR, PyNCCL).
  * CudaCommunicator.all_gather (:460) - contiguous, <= DS41H_AG_MAX_BYTES per rank -> LLComm (writes the
    concatenated layout directly, no movedim copy); else stock.
  * CudaCommunicator.destroy             - releases the IPC buffers.
Group gating mirrors the stock custom-AR gating (:57-67): only the group whose unique_name starts with
"tp:" and only when it has exactly 4 ranks. Every routing decision depends only on tensor shape/dtype,
so all ranks take the same path (required for any collective).
"""

from __future__ import annotations

import logging

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.allreduce")
# DS41H_AR_DEBUG=N: log the first N routing decisions (all-reduce / all-gather) of the TP communicator on every rank
_DEBUG = int(__import__("os").environ.get("DS41H_AR_DEBUG", "0") or 0)
_DBG_N = [0]


def _debug(op: str, t, path: str) -> None:
    if _DBG_N[0] >= _DEBUG:
        return
    _DBG_N[0] += 1
    import torch

    try:
        from ds41h.patch_core import tp_rank

        rank = tp_rank()
    except Exception:  # noqa: BLE001
        rank = "?"
    logger.info("ds41h ar dbg rank=%s #%d %s %s %s -> %s capturing=%s", rank, _DBG_N[0], op, tuple(t.shape), t.dtype, path,
                torch.cuda.is_current_stream_capturing())


def patch_communicator(module) -> None:
    from ds41h.patch_core import mark_active

    comm_cls = module.CudaCommunicator
    base_init = comm_cls.__init__
    base_all_reduce = comm_cls.all_reduce
    base_all_gather = comm_cls.all_gather
    base_destroy = getattr(comm_cls, "destroy", None)

    def __init__(self, *args, **kwargs) -> None:
        base_init(self, *args, **kwargs)
        self.ds41h_comm = None
        name = getattr(self, "unique_name", "") or ""
        if name.split(":")[0] != "tp" or self.world_size != 4 or getattr(self.device, "type", None) != "cuda":
            return
        try:
            from ds41h.allreduce import LLComm

            self.ds41h_comm = LLComm(self.cpu_group, self.device, name=f"tp-allreduce({name})")
        except Exception:  # noqa: BLE001 - fall back to stock, but loudly
            logger.exception("ds41h: LL all-reduce init failed for group %s; using stock collectives", name)
            self.ds41h_comm = None

    def all_reduce(self, input_):
        comm = getattr(self, "ds41h_comm", None)
        if comm is not None and comm.can_all_reduce(input_):
            if _DEBUG:
                _debug("ar", input_, "ll")
            return comm.all_reduce(input_)
        if _DEBUG and comm is not None:
            _debug("ar", input_, "stock")
        return base_all_reduce(self, input_)

    def all_gather(self, input_, dim: int = -1):
        comm = getattr(self, "ds41h_comm", None)
        if comm is not None and comm.can_all_gather(input_, dim):
            if _DEBUG:
                _debug("ag", input_, "ll")
            return comm.all_gather(input_, dim)
        if _DEBUG and comm is not None:
            _debug("ag", input_, "stock")
        return base_all_gather(self, input_, dim)

    def destroy(self) -> None:
        comm = getattr(self, "ds41h_comm", None)
        self.ds41h_comm = None
        if comm is not None:
            try:
                comm.close()
            except Exception:  # noqa: BLE001
                logger.exception("ds41h: LL comm close failed")
        if base_destroy is not None:
            base_destroy(self)

    comm_cls.__init__ = __init__
    comm_cls.all_reduce = all_reduce
    comm_cls.all_gather = all_gather
    comm_cls.destroy = destroy
    mark_active("allreduce", "CudaCommunicator.{__init__,all_reduce,all_gather,destroy}")


if flags.ALLREDUCE:
    register("vllm.distributed.device_communicators.cuda_communicator")(patch_communicator)
