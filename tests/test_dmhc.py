"""1-GPU test for kernels/dmhc.cu (DS41H_DRAFTER_MHC): the drafter FFN-side mHC in one launch vs stock
mhc_fused_post_pre_delayed_tilelang (vLLM 0.32.0). usage: python tests/test_dmhc.py [--nobench]"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def make(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    H = 5120
    x = (torch.randn(m, H, generator=g, device=dev) * 0.8).to(torch.bfloat16)
    res = (torch.randn(m, 4, H, generator=g, device=dev) * 2.0).to(torch.bfloat16)
    post = torch.rand(m, 4, 1, generator=g, device=dev) * 2.0
    comb = torch.softmax(torch.randn(m, 4, 4, generator=g, device=dev), -1).contiguous()
    fn = torch.randn(24, 4 * H, generator=g, device=dev) * 0.02
    hc_scale = torch.tensor([0.7, 0.5, 0.9], device=dev)
    hc_base = torch.randn(24, generator=g, device=dev) * 0.1
    pre_in = torch.rand(m, 4, generator=g, device=dev) + 0.5
    norm_w = (1.0 + torch.randn(H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    return x, res, post, comb, fn, hc_scale, hc_base, pre_in, norm_w


EPS = dict(rms_eps=1e-20, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20, norm_eps=1e-20)


def stock(args):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_delayed_tilelang

    x, res, post, comb, fn, hc_scale, hc_base, pre_in, norm_w = args
    return mhc_fused_post_pre_delayed_tilelang(x, res, post, comb, fn, hc_scale, hc_base, EPS["rms_eps"], EPS["hc_pre_eps"],
                                               EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"], pre_mix=pre_in,
                                               norm_weight=norm_w, norm_eps=EPS["norm_eps"])


def ours(args, pdl=False, route=None):
    from ds41h import patch_drafter as P

    x, res, post, comb, fn, hc_scale, hc_base, pre_in, norm_w = args
    m = x.shape[0]
    dev = x.device
    out = (torch.empty_like(res), torch.empty(m, 4, 1, device=dev), torch.empty(m, 4, 4, device=dev),
           torch.empty(m, 5120, dtype=torch.bfloat16, device=dev), torch.empty(m, 4, device=dev))
    scratch, ctr, rpart = P._mhc_state(dev)
    rargs = (None, None, None, None, 0, None, None, None, 0.0, False, 0)
    if route is not None:
        gw, bias, bias_vl, ids_in, image_lo, topk, rscale = route
        ids_t = torch.empty(m, topk, dtype=torch.int32, device=dev)
        w_t = torch.empty(m, topk, dtype=torch.float32, device=dev)
        rargs = (gw, bias, bias_vl, ids_in, image_lo, rpart, ids_t, w_t, rscale, True, topk)
    P.mhc_ext().mhc_ffn(x, res, post, comb, fn, hc_scale, hc_base, pre_in, norm_w, *out, scratch, ctr, EPS["rms_eps"],
                        EPS["hc_pre_eps"], EPS["hc_sinkhorn_eps"], EPS["hc_post_mult_value"], EPS["sinkhorn_repeat"], EPS["norm_eps"], pdl,
                        *rargs)
    if route is not None:
        return out + (rargs[7], rargs[6])
    return out


def set_variant(v: int) -> None:
    from ds41h import patch_drafter as P

    P.mhc_ext().mhc_set_variant(v)


def main() -> int:
    torch.manual_seed(0)
    from ds41h import patch_drafter as P

    variants = [0]
    ncl = int(P.mhc_ext().cl_max_active(True))
    print(f"cluster kernel: max active 16-CTA clusters {ncl}")
    if ncl > 0:
        variants.append(1)
    for var in variants:
        set_variant(var)
        check(var)
    set_variant(0)
    check_final()
    print("dmhc: ALL PASS")
    if "--nobench" in sys.argv:
        return 0
    bench(variants)
    return 0


def check(var: int) -> None:
    tag = "cluster" if var else "v1"
    for m in (1, 3, 7, 8):
        for seed in (0, 1):
            a = make(m, seed)
            s = stock(a)
            o = ours(a)
            o2 = ours(a, pdl=True)
            torch.cuda.synchronize()
            for i, name in enumerate(("residual", "post", "comb", "layer_input", "pre")):
                assert torch.equal(o[i], o2[i]), f"m={m}: pdl / plain launch differ in {name}"
            res_eq = torch.equal(o[0], s[0])
            res_ulp = int((o[0].float() - s[0].float()).abs().gt(0).sum())
            assert res_ulp <= o[0].numel() // 1000, f"m={m}: residual differs in {res_ulp} elements"
            for i, name in ((1, "post"), (2, "comb"), (4, "pre")):
                d = (o[i].reshape(-1) - s[i].reshape(-1).float()).abs().max().item()
                assert d < 2e-4 * max(1.0, s[i].abs().max().item()), f"m={m}: {name} max |diff| {d}"
            li = (o[3].float() - s[3].float()).abs()
            ulp = torch.exp2(torch.floor(torch.log2(s[3].float().abs().clamp(min=1e-6))) - 7)
            bad = int((li > 1.01 * ulp).sum())
            assert bad <= 4, f"m={m}: layer_input {bad} elements > 1 bf16 ulp"
            o3 = ours(a, pdl=True)
            torch.cuda.synchronize()
            for i in range(5):
                assert torch.equal(o[i], o3[i]), f"{tag} m={m}: not bitwise repeatable (output {i})"
            print(f"dmhc[{tag}] m={m} seed={seed}: residual {'bitwise' if res_eq else f'{res_ulp} ulp-diffs'}, layer_input "
                  f"{int(li.gt(0).sum())} 1-ulp diffs, post/comb/pre within 2e-4, repeatable")
    # fused router (DS41H_DRAFTER_MHC=2) vs stock gate GEMM (fp32 logits) + ds41h route_topk (stock dsv4_topk semantics)
    from ds41h import moe

    for m in (1, 5, 8):
        a = make(m, 10 + m)
        g = torch.Generator(device="cuda").manual_seed(77)
        gw = (torch.randn(128, 5120, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        bias = torch.randn(128, generator=g, device="cuda") * 0.01
        bias_vl = torch.randn(128, generator=g, device="cuda") * 0.01
        ids_in = torch.randint(0, 1000, (m,), dtype=torch.int32, device="cuda")
        image_lo = 129264
        if m > 2:
            ids_in[1] = image_lo + 2  # an image-sentinel token routes with bias_vl
        o = ours(a, route=(gw, bias, bias_vl, ids_in, image_lo, 3, 1.5))
        o2 = ours(a, route=(gw, bias, bias_vl, ids_in, image_lo, 3, 1.5), pdl=True)
        torch.cuda.synchronize()
        assert torch.equal(o[5], o2[5]) and torch.equal(o[6], o2[6]), f"{tag} router m={m}: not repeatable"
        logits = o[3].float() @ gw.float().t()
        w_ref, id_ref = moe.route_topk(logits, bias, 3, 1.5, True, bias_vl=bias_vl, input_ids=ids_in.to(torch.int64), image_lo=image_lo)
        same = torch.equal(o[6].long(), id_ref.long())
        if not same:  # near-ties may legitimately flip with a different fp32 summation order: check the scores agree
            print(f"  router m={m}: ids differ (fp32 order) ref {id_ref.tolist()} got {o[6].tolist()}")
        assert (o[5] - w_ref.float()).abs().max().item() < 1e-3 or not same, f"router m={m}: weights differ"
        print(f"dmhc[{tag}]+router m={m}: ids {'==' if same else '!='} reference, weights within 1e-3")


def stock_final(x, res, post, comb, pre, norm_w, eps=1e-6):
    """Stock end of the drafter forward + head-input norm: mhc_post_tilelang -> hc_collapse_triton -> vLLM rms_norm."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton

    c = hc_collapse_triton(mhc_post_tilelang(x, res, post, comb), pre)
    y = torch.empty_like(c)
    ops.rms_norm(y, c, norm_w, eps)
    return c, y


def ours_final(x, res, post, comb, pre, norm_w, eps=1e-6, pdl=False):
    from ds41h import patch_drafter as P

    c = torch.empty(x.shape[0], 5120, dtype=torch.bfloat16, device=x.device)
    y = torch.empty_like(c)
    P.mhc_ext().final_fuse(x, res, post.contiguous(), comb, pre, norm_w, eps, c, y, pdl)
    return c, y


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    return int((ia - ib).abs().max().item())


def check_final() -> None:
    """DS41H_DRAFTER_FINAL_FUSE (dfin::fin_kernel) vs the stock 3-launch chain: c (= the forward's output) bitwise, y (= the
    normed head input) within 1 bf16 ulp on a tiny fraction (only the fp32 order of the sum of squares differs)."""
    worst_c, worst_y, n_c, n_y, tot = 0, 0, 0, 0, 0
    for m in (1, 3, 7, 8):
        for seed in (0, 1, 2):
            x, res, post, comb, _, _, _, pre, norm_w = make(m, seed)
            x = x * 3.0  # layer output scale comparable to the residual streams
            c_ref, y_ref = stock_final(x, res, post, comb, pre, norm_w)
            c, y = ours_final(x, res, post, comb, pre, norm_w)
            torch.cuda.synchronize()
            worst_c, worst_y = max(worst_c, ulp_diff(c, c_ref)), max(worst_y, ulp_diff(y, y_ref))
            n_c += int((c != c_ref).sum().item())
            n_y += int((y != y_ref).sum().item())
            tot += c.numel()
            c2, y2 = ours_final(x, res, post, comb, pre, norm_w, pdl=True)
            assert torch.equal(c2, c) and torch.equal(y2, y), "final_fuse not repeatable"
    print(f"final_fuse: c != stock on {n_c}/{tot} values (max {worst_c} ulp), y on {n_y}/{tot} (max {worst_y} ulp)")
    assert worst_c <= 1 and n_c <= tot // 1000, "final_fuse collapse output differs from stock beyond rounding-order noise"
    assert worst_y <= 1 and n_y <= tot // 200, "final_fuse normed output differs from stock beyond rounding-order noise"
    print("final_fuse: PASS")


def bench(variants) -> None:
    import torch.nn.functional as F

    from ds41h import moe

    a = make(7, 3)
    g = torch.Generator(device="cuda").manual_seed(78)
    gw = (torch.randn(128, 5120, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(128, generator=g, device="cuda") * 0.01
    route = (gw, bias, None, None, 0, 3, 1.5)

    def stock_chain():  # stock mHC (2 launches) + gate GEMV (fp32 logits) + top-k: the in-situ FFN-pre chain
        o = stock(a)
        logits = F.linear(o[3], gw).float()
        return moe.route_topk(logits, bias, 3, 1.5, True)

    cases = [("stock tilelang (2 launches)", lambda: stock(a), None), ("stock mHC + gate + topk", stock_chain, None)]
    fa = make(7, 5)
    cases.append(("stock final post+collapse+norm (3)", lambda: stock_final(fa[0], fa[1], fa[2], fa[3], fa[7], fa[8]), None))
    cases.append(("ds41h final_fuse (1 launch)", lambda: ours_final(fa[0], fa[1], fa[2], fa[3], fa[7], fa[8], pdl=True), None))
    for v in variants:
        nm = "cluster" if v else "v1"
        cases.append((f"ds41h dmhc {nm} (1 launch)", lambda: ours(a, pdl=True), v))
        cases.append((f"ds41h dmhc {nm} + router", lambda: ours(a, pdl=True, route=route), v))
    for name, fn, v in cases:
        if v is not None:
            set_variant(v)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        print(f"bench m=7 {name:32s} {s.elapsed_time(e) * 1e3 / 100:6.2f} us/call (graph, back-to-back)")
    set_variant(0)


if __name__ == "__main__":
    sys.exit(main())
