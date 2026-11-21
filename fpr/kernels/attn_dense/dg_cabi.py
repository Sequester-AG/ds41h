"""dg_cabi: build / warm-up of the DeepGEMM C ABI wrapper.

build()  -> the extension module (load with its __file__ for the driver's RTLD_NOLOAD dlopen; table fpr_dg_table).
warm_device(dev, wo_a, wo_b, m_max): every (op, M = 1 .. m_max) key of the served DENSE8 calls on device dev with today's
  per-device num_sms (attn_act8._dg_device: base - 2 * dev), on scratch buffers, Python thread (engine / bucket init).
  wo_a = (w, ws) of P["wo_a"] (einsum b [8, 1024, 4096]), wo_b = the FoldedWeight. Keys the op would allocate for are
  not warmed (the C path then returns FPR_E_SHAPE). Returns the number of warmed keys.
"""
from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT: dict = {}


def build():
    if "m" not in _EXT:
        from torch.utils.cpp_extension import load

        import vllm.third_party.deep_gemm  # noqa: F401  (registers the deep_gemm:: dispatcher ops)

        bdir = os.environ.get("DG_CABI_BUILD") or os.path.join(HERE, "build_dg_cabi")
        os.makedirs(bdir, exist_ok=True)
        _EXT["m"] = load(name="fpr_dg_cabi", sources=[os.path.join(HERE, "dg_cabi.cpp")], extra_cflags=["-O3", "-std=c++17"],
                         extra_include_paths=[HERE], with_cuda=True, build_directory=bdir, verbose=False)
    return _EXT["m"]


def num_sms_of(dev: int) -> int:
    """Today's per-device DeepGEMM SM count (attn_act8._dg_device)."""
    import attn_act8 as AA

    AA._dg()
    return int(AA._DG["base"]) - 2 * int(dev)


def warm_device(dev: int, wo_a=None, wo_b=None, m_max: int = 4416, ue8m0: bool = False) -> int:
    from ds41h import linear as DL

    m = build()
    n_ok = 0
    sms = num_sms_of(dev)
    with torch.cuda.device(dev):
        if wo_b is not None:
            x = torch.zeros(m_max, wo_b.k, dtype=torch.bfloat16, device=dev)
            y = torch.empty(m_max, wo_b.n, dtype=torch.bfloat16, device=dev)
            for M in range(1, m_max + 1):
                xq, xs = DL.act_quant(x[:M], wo_b.kp)
                n_ok += m.warm_gemm(xq, xs, wo_b.w, wo_b.ws, y[:M], sms) == 0
        if wo_a is not None:
            import attn_act8 as AA
            w, ws = wo_a
            o = torch.zeros(m_max, 64, 512, dtype=torch.bfloat16, device=dev)
            pos = torch.zeros(m_max, dtype=torch.int64, device=dev)
            cs = torch.zeros(8, 64, dtype=torch.float32, device=dev)
            z = torch.empty(m_max, 8, 1024, dtype=torch.bfloat16, device=dev)
            for M in range(1, m_max + 1):
                oq, osc = AA.ropeq().inv_rope_fp8_quant(o[:M], pos[:M], cs, 8)
                n_ok += m.warm_einsum(oq, osc, w, ws, z[:M], sms) == 0
    return n_ok
