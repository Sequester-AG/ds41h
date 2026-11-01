"""Determinism probe (debugging only, default OFF): bit digests of every module input / output of one
eager target-model forward, to find the first op whose output differs between two runs of the same prompt.

DS41H_DETPROBE=<dir> installs the probe (ds41h/patch_prefill.py registers it); it is ARMED while <dir>/arm exists.
Every armed eager DeepseekV4Model.forward (not capturing, >= DS41H_DETPROBE_MIN_T tokens, default 16) records, in
call order, for every submodule of the target layers (+ the model's own inputs / output and the SP layer pieces of
ds41h/prefill_sp.py) a digest of each CUDA tensor in its inputs (pre-hook) and outputs (post-hook): per row (dim 0)
for tensors with >= 2 dims, else one value; digest = sum_j bits(x[i, j]) * w_j (int64, wrapping) with fixed odd
weights, i.e. any single-bit change changes it. At the end of the forward the records go to
<dir>/f<seq:05d>_r<rank>.pt (seq counts armed forwards per process). Digests are computed on the GPU in the
forward's stream (reads only; no effect on numerics), one host sync per armed forward.
Analysis: scripts/detprobe_cmp.py (first differing record per prompt / rep, per rank).
"""

from __future__ import annotations

import os

DIR = os.environ.get("DS41H_DETPROBE", "")
ENABLED = bool(DIR)
MIN_T = int(os.environ.get("DS41H_DETPROBE_MIN_T", "16"))
_MAX_ELEMS = 1 << 24  # int64 temporaries per digest chunk

_st: dict = {"active": False, "recs": None, "T": 0, "seq": 0}
_W: dict = {}


def armed() -> bool:
    return ENABLED and os.path.exists(os.path.join(DIR, "arm"))


def active() -> bool:
    return _st["active"]


def _weights(n: int, device):
    import torch

    key = (n, str(device))
    w = _W.get(key)
    if w is None:
        w = (torch.arange(n, device=device, dtype=torch.int64) * 2654435761 + 97) % 2147483647
        w = w * 2 + 1  # odd
        _W[key] = w
    return w


def _bits(t):
    import torch

    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    if t.dtype == torch.bool:
        t = t.view(torch.uint8)
    es = t.element_size()
    if es == 1:
        return t.view(torch.uint8)
    if es == 2:
        return t.view(torch.int16)
    if es == 4:
        return t.view(torch.int32)
    if es == 8:
        return t.view(torch.int64)
    return None


def digest(t):
    """-> int64 [rows] digest (rows = dim 0 for >= 2-dim tensors, else 1)."""
    import torch

    if t.numel() == 0:
        return torch.zeros(1, dtype=torch.int64, device=t.device)
    v = _bits(t)
    if v is None:
        return None
    v2 = v.reshape(v.shape[0], -1) if v.dim() >= 2 else v.reshape(1, -1)
    n = v2.shape[1]
    w = _weights(n, v2.device)
    out = torch.empty(v2.shape[0], dtype=torch.int64, device=v2.device)
    step = max(1, _MAX_ELEMS // max(n, 1))
    for i in range(0, v2.shape[0], step):
        blk = v2[i : i + step].to(torch.int64)
        out[i : i + step] = (blk * w).sum(1)
    return out


def _flatten(obj, out: list, depth: int = 0) -> None:
    import torch

    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif depth > 3 or obj is None:
        return
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _flatten(o, out, depth + 1)
    elif isinstance(obj, dict):
        for k in obj:
            _flatten(obj[k], out, depth + 1)
    elif hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            _flatten(getattr(obj, f, None), out, depth + 1)


def record(name: str, kind: str, objs) -> None:
    """Append one record (no-op unless an armed forward is running and the stream is not capturing)."""
    if not _st["active"]:
        return
    import torch

    if torch.cuda.is_current_stream_capturing():
        return
    ts: list = []
    _flatten(objs, ts)
    ds = []
    for t in ts:
        if not t.is_cuda:
            continue
        d = digest(t)
        if d is not None:
            ds.append((tuple(t.shape), str(t.dtype).replace("torch.", ""), d))
    _st["recs"].append((name, kind, ds))


def _hook_model(model) -> None:
    n_hooks = 0
    for name, m in model.layers.named_modules(prefix="layers"):
        m.register_forward_pre_hook(lambda mod, args, kwargs, n=name: record(n, "in", (args, kwargs)), with_kwargs=True)
        m.register_forward_hook(lambda mod, args, out, n=name: record(n, "out", out))
        n_hooks += 1
    norm = getattr(model, "norm", None)
    if norm is not None:
        norm.register_forward_hook(lambda mod, args, out: record("norm", "out", out))
    import logging

    logging.getLogger("vllm.ds41h").info("ds41h detprobe: hooked %d modules of the target model (dir %s)", n_hooks, DIR)


def _save() -> None:
    import torch

    rank = 0
    try:
        from vllm.distributed import get_tp_group

        rank = get_tp_group().rank_in_group
    except Exception:  # noqa: BLE001
        pass
    recs = [(n, k, [(s, dt, d.cpu()) for s, dt, d in ds]) for n, k, ds in _st["recs"]]
    seq = _st["seq"]
    _st["seq"] += 1
    path = os.path.join(DIR, f"f{seq:05d}_r{rank}.pt")
    torch.save({"T": _st["T"], "rank": rank, "seq": seq, "recs": recs}, path + ".tmp")
    os.replace(path + ".tmp", path)


def install(module) -> None:
    """Wrap vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.forward (idempotent)."""
    cls = module.DeepseekV4Model
    if getattr(cls, "_ds41h_detprobe", False):
        return
    base = cls.forward

    def forward(self, *args, **kwargs):
        import torch

        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if (
            positions is None
            or positions.shape[0] < MIN_T
            or not armed()
            or torch.cuda.is_current_stream_capturing()
        ):
            return base(self, *args, **kwargs)
        if not getattr(self, "_ds41h_detprobe_hooked", False):
            _hook_model(self)
            self._ds41h_detprobe_hooked = True
        _st.update(active=True, recs=[], T=int(positions.shape[0]))
        try:
            record("model", "in", (input_ids, positions))
            out = base(self, *args, **kwargs)
            record("model", "out", out)
        finally:
            _st["active"] = False
        _save()
        return out

    forward.__wrapped__ = base  # type: ignore[attr-defined]
    cls.forward = forward
    cls._ds41h_detprobe = True
