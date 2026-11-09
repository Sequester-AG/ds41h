"""Process / runtime state that can change the engine's numerics or graph behaviour, for diffing the served EngineCore
against an offline driver process (to locate a served-vs-offline numerical excess).

  PHYS_DUMP_RUNTIME=<file.jsonl>   PhysExecutor appends one record before the worker is built ("pre_worker"), after
                                   ("post_worker"), after warmup and at the first step (thread, current stream).
  python adapter/runtime_state.py --offline <file.jsonl>   the same records from a driver process (OfflineEngine)
  python adapter/runtime_state.py --diff A.jsonl B.jsonl    print the fields that differ (same-named records)

Records hold flags, env NAMES and values of numerics / CUDA / vLLM / phys variables (no keys: values of names that look
secret are masked), torch.cuda graph-object identities, dispatch / function-mode stacks, threads; never prompts.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

ENV_PREFIXES = ("VLLM_", "PYTORCH_", "TORCH", "CUDA", "CUBLAS", "CUDNN", "NCCL", "NVIDIA_", "DS41H", "PHYS_", "IDX2F",
                "OMP_", "MKL_", "TRITON", "CUTLASS", "LD_PRELOAD", "PYTHONPATH", "TOKENIZERS")
SECRET = ("KEY", "TOKEN", "SECRET", "PASS", "CERT")


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {str(e)[:80]}>" if default is None else default


def _ident(obj) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__name__)}@{id(obj):x}"


def collect(tag: str) -> dict:
    import torch

    tb = torch.backends
    r = dict(tag=tag, t=round(time.time(), 3), pid=os.getpid(), argv0=os.path.basename(sys.argv[0] if sys.argv else ""),
             thread=threading.current_thread().name, thread_main=threading.current_thread() is threading.main_thread(),
             threads=sorted(t.name for t in threading.enumerate()))
    r["torch"] = dict(
        version=torch.__version__,
        tf32=tb.cuda.matmul.allow_tf32,
        fp16_reduced=tb.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced=tb.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_accum=_safe(lambda: tb.cuda.matmul.allow_fp16_accumulation),
        f32_precision=torch.get_float32_matmul_precision(),
        cudnn_tf32=tb.cudnn.allow_tf32, cudnn_benchmark=tb.cudnn.benchmark, cudnn_deterministic=tb.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        default_dtype=str(torch.get_default_dtype()),
        default_device=_safe(lambda: str(torch.get_default_device())),
        num_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled(),
        autocast_gpu=_safe(lambda: torch.is_autocast_enabled("cuda")),
        sdp=_safe(lambda: dict(flash=tb.cuda.flash_sdp_enabled(), mem=tb.cuda.mem_efficient_sdp_enabled(),
                                math=tb.cuda.math_sdp_enabled())),
        blas=_safe(lambda: str(tb.cuda.preferred_blas_library())),
        dispatch_modes=_safe(lambda: torch._C._len_torch_dispatch_mode()),
        function_modes=_safe(lambda: torch._C._len_torch_function_stack()),
        dynamo_compiling=_safe(lambda: torch.compiler.is_compiling()),
    )
    import torch.cuda.graphs as G
    r["graphs"] = dict(
        cuda_graph=_ident(torch.cuda.graph), cuda_graph_is_orig=torch.cuda.graph is G.graph,
        CUDAGraph=_ident(torch.cuda.CUDAGraph), CUDAGraph_is_orig=torch.cuda.CUDAGraph is G.CUDAGraph,
        CUDAGraph_mro=[c.__qualname__ for c in torch.cuda.CUDAGraph.__mro__],
        replay=_ident(torch.cuda.CUDAGraph.replay), capture_begin=_ident(torch.cuda.CUDAGraph.capture_begin),
        pool_handle=_ident(torch.cuda.graph_pool_handle), pool_is_orig=torch.cuda.graph_pool_handle is G.graph_pool_handle,
        make_graphed=_ident(torch.cuda.make_graphed_callables),
        stream_ctx=_ident(torch.cuda.stream), set_stream=_ident(torch.cuda.set_stream),
    )
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        r["cuda"] = dict(
            device=torch.cuda.current_device(),
            current_stream=_safe(lambda: str(torch.cuda.current_stream())),
            default_stream=_safe(lambda: str(torch.cuda.default_stream())),
            is_default_stream=_safe(lambda: torch.cuda.current_stream() == torch.cuda.default_stream()),
            capturing=_safe(lambda: torch.cuda.is_current_stream_capturing()),
            allocator=_safe(lambda: torch._C._cuda_getAllocatorBackend()),
            mem_fraction=_safe(lambda: torch.cuda.get_per_process_memory_fraction()),
        )
    env = {}
    for k in sorted(os.environ):
        if k.startswith(ENV_PREFIXES):
            v = os.environ[k]
            env[k] = "<masked>" if any(s in k.upper() for s in SECRET) else v
    r["env"] = env
    mods = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None) or ""
        if "/ds41h/" in f or name.startswith(("ds41h", "sitecustomize", "usercustomize")) or "/runtime/" in f:
            mods[name] = f
    r["modules_deploy"] = dict(sorted(mods.items()))
    r["vllm_loaded"] = "vllm" in sys.modules
    r["plugins"] = _safe(lambda: sorted(__import__("vllm.plugins", fromlist=["x"]).plugins_loaded), [])
    r["ds41h_moe"] = moe_constants()
    if tag in ("first_step", "post_worker") or os.environ.get("PHYS_DUMP_RUNTIME_SYMBOLS") == "1":
        r["vllm_symbols"] = _safe(vllm_symbols, {})
    import gc
    r["gc"] = dict(enabled=gc.isenabled(), threshold=gc.get_threshold(), frozen=gc.get_freeze_count())
    return r


# the vLLM symbols the fork engine imports (engine/, kernels/ runtime modules). With DS41H=1 production's
# ds41h patches install import hooks that may replace some of them; the engine then gets the patched object.
VLLM_SYMBOLS = (
    ("vllm.models.deepseek_v41.common.ops.cache_utils", "compute_global_topk_indices_and_lens"),
    ("vllm.third_party.flashmla.flash_mla_interface", "flash_mla_with_kvcache"),
    ("vllm.third_party.flashmla.flash_mla_interface", "get_mla_metadata"),
    ("vllm.models.deepseek_v41.common.ops", "fused_indexer_q_rope_quant"),
    ("vllm.models.deepseek_v41.common.ops", "indexer_k_norm_rope_store"),
    ("vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache", "fused_save_compress_norm"),
    ("vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache", "rope_quant_insert"),
    ("vllm.v1.worker.gpu.sample.gumbel", "_log1p_neg_stable"),
    ("vllm.v1.worker.gpu.sample.gumbel", "gumbel_noised_argmax"),
    ("vllm.v1.worker.gpu.sample.gumbel", "murmur3_uniform32"),
    ("vllm.v1.worker.gpu.sample.gumbel", "tl_rand32"),
    ("vllm.utils.deep_gemm", "get_paged_mqa_logits_metadata"),
    ("vllm.utils.deep_gemm", "fp8_fp4_paged_mqa_logits"),
    ("vllm.model_executor.layers.indexer_topk", "get_indexer_topk"),
    ("vllm.model_executor.kernels.linear.mxfp8.humming", "HummingMxfp8LinearKernel"),
    ("vllm.model_executor.kernels.attention.dsa.candidate_blocks", "select_candidate_blocks"),
    ("vllm.model_executor.kernels.attention.dsa.candidate_blocks", "apply_candidate_mask"),
)


def _code_id(obj) -> str:
    """Where obj's code lives: function -> file:qualname, Triton JITFunction -> its python fn, else the type."""
    fn = getattr(obj, "fn", None) if not callable(getattr(obj, "__code__", None)) else None
    target = fn if fn is not None and hasattr(fn, "__code__") else obj
    code = getattr(target, "__code__", None)
    if code is not None:
        f = code.co_filename
        return f"{f.split('site-packages/')[-1].split('/fpr/')[-1]}:{getattr(target, '__qualname__', '?')}"
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def vllm_symbols() -> dict:
    """Identity of every vLLM symbol the engine uses, as an import in THIS process resolves it (imports them)."""
    import importlib

    out = {}
    for mod, name in VLLM_SYMBOLS:
        key = f"{mod.split('.')[-1]}.{name}"
        try:
            obj = getattr(importlib.import_module(mod), name)
        except Exception as e:  # noqa: BLE001
            out[key] = f"<{type(e).__name__}: {str(e)[:60]}>"
            continue
        rec = dict(code=_code_id(obj), ds41h=bool(getattr(obj, "_ds41h", False)),
                   stock=_code_id(obj._ds41h_stock) if hasattr(obj, "_ds41h_stock") else None)
        if isinstance(obj, type):
            rec["methods"] = {k: _code_id(v) for k, v in sorted(vars(obj).items()) if callable(v)}
        out[key] = rec
    return out


MOE_CONSTANTS = ("ENABLED", "MOE_DEC", "MOE_DEC_VARIANT", "SMALL_T", "SMALL_VARIANT", "SMALL_AUTO", "VARIANT", "HYBRID",
                 "HYBRID_ENGINE", "_CTL_PATH")


def moe_constants() -> dict | None:
    """production's ds41h.moe switches, read from the environment ONCE at its import (None: not imported yet)."""
    m = sys.modules.get("ds41h.moe")
    if m is None:
        return None
    out = {k: getattr(m, k, "<absent>") for k in MOE_CONSTANTS}
    out["file"] = getattr(m, "__file__", None)
    md = sys.modules.get("ds41h.moe_dec")
    if md is not None:
        out["moe_dec"] = {k: v for k, v in vars(md).items() if k.isupper() and isinstance(v, (int, str, float, bool))}
    flags = sys.modules.get("ds41h.flags")
    if flags is not None:
        out["flags"] = dict(ENABLED=getattr(flags, "ENABLED", None), LOSSY=getattr(flags, "LOSSY", None))
    return out


def check_moe_env(log, engine_default: str | None = "1") -> bool:
    """True if production's ds41h.moe (when imported) runs with the DS41H_MOE_DEC the fork engine asks for: the
    environment's value, else engine.prod_moe()'s setdefault (engine_default; None = no engine that calls it)."""
    m = sys.modules.get("ds41h.moe")
    if m is None or not hasattr(m, "MOE_DEC"):
        return True
    env = os.environ.get("DS41H_MOE_DEC")
    if env is None and engine_default is None:
        return True
    want = int((env if env is not None else engine_default) or 0)
    if int(m.MOE_DEC) != want:
        log(f"ds41h.moe was imported with MOE_DEC={m.MOE_DEC} but DS41H_MOE_DEC={want} now: its kernel choice no longer "
            f"follows the environment. Export DS41H_MOE_DEC before vLLM starts (scripts/serve_fpr.sh does).")
        return False
    return True


def dump(tag: str, path: str | None = None) -> None:
    path = path or os.environ.get("PHYS_DUMP_RUNTIME", "")
    if not path:
        return
    try:
        rec = collect(tag)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:  # noqa: BLE001  (a diagnostic never breaks the server)
        print(f"phys adapter: runtime dump failed: {e}", file=sys.stderr)


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("modules_deploy",):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def diff(a_path: str, b_path: str) -> None:
    def load(p):
        return {r["tag"]: r for r in map(json.loads, open(p))}
    A, B = load(a_path), load(b_path)
    print(f"A {a_path}: {sorted(A)}\nB {b_path}: {sorted(B)}")
    pairs = [(ta, tb) for ta in A for tb in B]
    for ta, tb in pairs:
        fa, fb = _flat(A[ta]), _flat(B[tb])
        skip = ("t", "pid", "argv0", "env.PYTHONPATH")
        keys = sorted(set(fa) | set(fb))
        d = [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in keys
             if not k.startswith(skip) and k not in skip and fa.get(k, "<absent>") != fb.get(k, "<absent>")
             and not (k.startswith("graphs.") and isinstance(fa.get(k), str) and isinstance(fb.get(k), str)
                      and fa[k].split("@")[0] == fb[k].split("@")[0])]
        print(f"\n--- {ta} (A) vs {tb} (B): {len(d)} differences")
        for k, va, vb in d:
            print(f"  {k}: A={str(va)[:160]}  B={str(vb)[:160]}")


def _offline(path: str, engine_cls: str) -> None:
    """A driver process as the offline fork sets one up (serve/ on sys.path, set_pieces, OfflineEngine engine)."""
    phys = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(phys, "serve"))
    sys.path.insert(0, os.path.join(phys, "adapter", "tests"))
    from driver import OfflineEngine, set_pieces
    import worker as W

    set_pieces("full")
    # what Engine.prod_moe() does on the GPU (the mock never calls it): the reference ds41h package + production's ds41h
    sys.path.insert(0, os.environ.get("DS41H_REF", os.path.dirname(phys)))
    for k, v in (("DS41H_MOE_DEC", "1"), ("DS41H_MOE_DEC_CFG", "0x55")):
        os.environ.setdefault(k, v)
    import ds41h
    dt = os.environ.get("DS41H_DEPLOY", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) + "/ds41h"
    if dt not in list(ds41h.__path__):
        ds41h.__path__.append(dt)
    from ds41h import moe  # noqa: F401
    dump("pre_worker", path)
    mock = any(x in engine_cls.lower() for x in ("mock", "shaped"))      # CPU stand-ins take a free-memory size
    kw = dict(free_bytes_per_rank=12_000_000_000) if mock else {}
    eng = OfflineEngine(backend="engine", cfg=W.WorkerConfig(engine_cls=engine_cls, engine_kw=kw))
    dump("post_worker", path)
    o_step = eng.worker.step
    seen = []

    def spy(ps, grammar=None):
        if not seen:
            seen.append(1)
            dump("first_step", path)
        return o_step(ps, grammar)
    eng.worker.step = spy
    eng.force([dict(id="x", prompt_ids=list(range(1000, 1600)), forced=list(range(2000, 2012)), n_dump=8)])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="OUT")
    ap.add_argument("--engine-cls", default="shaped_engine.ShapedEngine")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.offline:
        _offline(a.offline, a.engine_cls)
    if a.diff:
        diff(*a.diff)
