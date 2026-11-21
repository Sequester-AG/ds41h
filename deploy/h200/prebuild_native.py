#!/usr/bin/env python3
"""CPU-only build of the CUDA extensions that engine/prebuild_ext.py does not cover: each kernel's own loader, run with
the serving configuration's environment, so the build directories and names are the ones the engine asks for at its
first GPU start (no JIT on the GPUs later).
  cwd = the fpr/ tree; CUDA_VISIBLE_DEVICES= ; PHYS_ROOT, PYTHONPATH and the configuration's environment set by
  deploy/h200/build_native.sh.   python prebuild_native.py [name ...]   (no names: all)"""
import importlib
import os
import sys
import time

PHYS = os.path.abspath(os.environ.get("PHYS_ROOT", "."))
sys.path.insert(0, PHYS)
W8D = os.environ.get("PREBUILD_W8A16_DEFS", "")


def mod(path, name):
    d = os.path.join(PHYS, path)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(name)


def w8a16(defs):
    if defs:
        os.environ["W8A16_DEFS"] = defs
    else:
        os.environ.pop("W8A16_DEFS", None)
    m = mod("kernels/attn_dense", "w8a16")
    return m.build()


def w8dq(impl):
    os.environ["W8DQ_IMPL"] = impl
    return mod("kernels/attn_dense", "w8dq").ext()


def c1d_base():
    c = mod("kernels/hsdtail", "c1d")
    return c.ext(defs=c.KINDS["base"][1], src=c.KINDS["base"][0])


def c1d3():
    c = mod("kernels/hsdtail", "c1d3")
    return c.ext3(c.KINDS3["s8m_nox2"][0])


def engram_w8a16():
    return mod("kernels/engram", "engram").w8a16_mod()[1]


def fp8_linear():
    from ds41h import linear as L
    return L.ext()


def eng_comm_c1():
    from torch.utils.cpp_extension import load
    bd = os.path.join(PHYS, "engine/build/eng_comm_c1")
    os.makedirs(bd, exist_ok=True)
    return load(name="eng_comm_c1", sources=[os.path.join(PHYS, "kernels/sp_layer/comm_c1.cu")], build_directory=bd,
                verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])


T = [
    ("w8a16", lambda: w8a16(W8D)),
    ("w8a16_default", lambda: w8a16("")),
    ("ropeq64", lambda: mod("kernels/attn_dense", "attn_act8").ropeq()),
    ("fp8_linear", fp8_linear),
    ("eng_comm_c1", eng_comm_c1),
    ("agrs_pmws", lambda: mod("kernels/agrs", "agrs_pmws").ext()),
    ("d13tail", lambda: mod("kernels/d13tail", "d13tail").ext()),
    ("mixed_head", lambda: mod("engine/mixed_cxx", "mixed_cxx_head").ext()),
    ("idx2f_topk4", lambda: mod("kernels/indexer2f", "idx2f").ext_topk4()),
    ("w8dq", lambda: w8dq("1")),
    ("w8dq2", lambda: w8dq("2")),
    ("eng_d13s", lambda: mod("engine", "d13s").ext()),
    ("hsdt_rtag", lambda: mod("kernels/hsdtail", "hsdt").rtag_ext()),
    ("hsdt_xg", lambda: mod("kernels/hsdtail", "hsdt").xg_ext()),
    ("hsdt_dtail", lambda: mod("kernels/hsdtail", "hsdt").dtail_ext()),
    ("hsdt_ctx3", lambda: mod("kernels/hsdtail", "hsdt_c1e").ctx3_ext()),
    ("c1d", lambda: mod("kernels/hsdtail", "c1d").ext()),
    ("c1d_base", c1d_base),
    ("c1d3", c1d3),
    ("c1mega", lambda: mod(os.environ["C1M_PKG"], "c1mega_eng").ext()),
    ("c1idx_host", lambda: mod(os.environ["C1M_PKG"], "c1idx_host").ext()),
    ("d13_cxx", lambda: mod("core/src/d13", "d13_cxx").ext()),
    ("mixed_cxx_head2", lambda: mod("engine/mixed_cxx", "mixed_cxx_head2").ext()),
    ("kernel_capture", lambda: mod("engine/mixed_cxx", "kernel_capture").ext()),
    ("triton_replay", lambda: mod("engine/mixed_cxx", "triton_replay").ext()),
    ("engram", lambda: mod("kernels/engram", "engram").ext()),
    ("engram_w8a16", engram_w8a16),
    ("head_comm", lambda: mod("kernels/head", "comm_head").ext()),
    ("head_drafter_v4", lambda: mod("kernels/head", "drafter").v4_ext()),
    ("pm_fused", lambda: mod("kernels/moe_prefill", "pm_fused").ext()),
    ("pm_swiglu", lambda: mod("kernels/moe_prefill", "pm_fused").swiglu_ext()),
    ("pagecopy", lambda: mod("kernels/pagecopy", "pagecopy").ext()),
    ("idx2f_cand", lambda: mod("kernels/indexer2f", "idx2f").ext_cand()),
    ("idx2f_tab", lambda: mod("kernels/indexer2f", "tables2f").ext_tab()),
    ("p1k", lambda: mod("kernels/p1", "p1k").build()),
    ("tail_gather", lambda: mod("kernels/tail_gather", "tail_gather").ext()),
    ("agrs_commc", lambda: mod("kernels/agrs", "agrs_comm_cabi").ext()),
    ("agrs_pmws_c", lambda: mod("kernels/agrs", "agrs_pmws_cabi").ext()),
    ("dg_cabi", lambda: mod("kernels/attn_dense", "dg_cabi").build()),
    ("gatherk", lambda: mod("kernels/attn_dense", "gather_k").build()),
    ("v4_cabi", lambda: mod("kernels/moe_v4", "v4_cabi").load()),
    ("mixed_cxx_driver", lambda: mod("engine/mixed_cxx", "mixed_cxx").ext()),
]

only = set(sys.argv[1:])
bad = []
t00 = time.time()
for n, fn in T:
    if only and n not in only:
        continue
    t0 = time.time()
    try:
        r = fn()
        msg = f"ok {getattr(r, '__file__', '')}"
    except Exception as e:  # noqa: BLE001
        msg = f"FAIL {type(e).__name__}: {str(e)[-300:]}"
        bad.append(n)
    print(f"prebuild_native: {n:18s} {time.time() - t0:7.1f} s  {msg}", flush=True)
print(f"prebuild_native: {len(T) - len(bad)} / {len(T)} in {time.time() - t00:.0f} s; failed {bad}", flush=True)
sys.exit(1 if bad else 0)
