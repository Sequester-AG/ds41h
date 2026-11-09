"""Fused streaming mHC boundary (mhcf.cu): drop-in for the engine's split boundary, bit-identical to mhc.cu's
k1a + k1b_out + k1b_mix (default flags) and k1a_x + k1b_out (the head).

    import sys; sys.path.insert(0, "kernels/mhc_fuse"); import mhcf
    F = mhcf.build()                       # CPU-buildable: CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=9.0a
    fnf = F.prep_f(fn_raw)                 # [24, 4 D] fp32 (finish_mhc's fa_raw / ff_raw) -> fp32 fragments (1.97 MB)

Weights at load (engine.py ~377): raw = Wt.load(f"mhc_L{L}_v1", ...); m = Wt.finish_mhc(raw, hext);
    m["fa_f"], m["ff_f"] = F.prep_f(raw["fa_raw"]), F.prep_f(raw["ff_raw"])      # +3.9 MB per layer per GPU

Engine's _split_bd, per boundary (scratch = s["part_mhc"]: >= R * (250 + 2560) fp32; part_mhc's 160 R 25 fits):
    F.fa(x, res, post, comb, lag, fnf, xn, scratch, 0)     # replaces k1a; takes lag (= the previous pre) as well
    ev.record(cur)
    F.fb_out(scratch, R, normw, norm_eps, out, ptok)       # replaces k1b_out(xn, lag, ...); ptok may be empty
    side stream: F.fb_mix(scratch, R, scale, base, norm_eps, hc_eps, iters, pre, post, comb)   # replaces k1b_mix
Head: F.fa_x(moe_out, xm2, f_post, f_comb, f_pre, Xf, scratch, 0); F.fb_out(scratch, R, norm_w, eps, h, empty)
Ordering: fa needs post / comb / lag from the previous boundary's fb_mix (the existing _join_mix before k1a covers it);
fa writes scratch, so the previous boundary's fb_mix (reads scratch's first R * 250 floats) must be joined before fa
(also the existing _join_mix). fa_x writes only the in part of scratch (from float ceil64(R * 250) on).
All tensors rows-contiguous ([R, ...] with stride(0) = row size; row slices of bucket buffers are fine).
"""
import os

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"]


def build(defs=None, verbose=False, build_directory=None):
    """The extension (name ds41h_phys_mhcf_<tag>); build_directory defaults to kernels/mhc_fuse/build/f_<tag>."""
    defs = defs or {}
    tag = "_".join(f"{k}{v}" for k, v in sorted(defs.items())) or "default"
    bd = build_directory or os.path.join(HERE, "build", "f_" + tag)
    os.makedirs(bd, exist_ok=True)
    return load(name=f"ds41h_phys_mhcf_{tag}", sources=[os.path.join(HERE, "mhcf.cu")], build_directory=bd,
                verbose=verbose, extra_cuda_cflags=FLAGS + [f"-D{k}={v}" for k, v in defs.items()])
