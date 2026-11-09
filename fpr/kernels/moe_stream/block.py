#!/usr/bin/env python3
"""The whole MoE block of one layer on one GPU.

  router (fp32, ref gate_ref) or synthetic skewed routing -> k_route (device routing tables, snake tiles, slot map)
  -> prepx (token prescale + images) -> GEMM1 -> GEMM2 with bf16 output per (token, slot) (no atomics)
  -> shared expert (bf16, exact fp8 -> bf16 weights, stock rounding points) -> k_combine -> bf16 rank partial.

Checks: device routing tables == the host builder (bench.build_tiles); output vs an fp64 reference with the kernel's
own roundings and stock's bf16 rounding of each expert output (errors in bf16 ulps). Timing per piece, W / MHz / J.

  python3 block.py --layer 21 --seqs 128 --cache cache/layer21_r0.pt [--routing router] [--out f]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

D, NE, TOPK, I_RANK = 5120, 384, 6, 576


def fp8_dequant_bf16(w, s):
    """e4m3 [R, C] with ue8m0 32x32 block scales [R/32, C/32] -> exact bf16 (dense scales 2^-13..2^-8)."""
    v = w.view(torch.float8_e4m3fn).float()
    sc = torch.pow(2.0, s.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)[: v.shape[0], : v.shape[1]]
    out = (v * sc).to(torch.bfloat16)
    assert torch.equal(out.float(), v * sc), "fp8 -> bf16 not exact"
    return out


def load_dense(ck, L, rank, dev):
    p = f"layers.{L}.ffn."
    g = lambda n: ck.get(p + n)  # noqa: E731
    g("gate.bias")  # opens the shard holding the router gate (Ckpt caches the handle)
    w1 = fp8_dequant_bf16(g("shared_experts.w1.weight"), g("shared_experts.w1.scale"))
    w3 = fp8_dequant_bf16(g("shared_experts.w3.weight"), g("shared_experts.w3.scale"))
    w2 = fp8_dequant_bf16(g("shared_experts.w2.weight"), g("shared_experts.w2.scale"))
    sl = slice(rank * I_RANK, (rank + 1) * I_RANK)
    typed = lambda n: ck.h[ck.idx[p + n]].get_tensor(p + n)  # noqa: E731  (Ckpt.get views everything as uint8)
    gate_w, gate_b = typed("gate.weight"), typed("gate.bias").float()  # gate weight stays bf16 (exact products)
    return (w1[sl].contiguous().to(dev), w3[sl].contiguous().to(dev), w2[:, sl].contiguous().to(dev),
            gate_w.to(dev), gate_b.to(dev))


def router(f, gate_w, gate_b, topk=TOPK, scale=1.5):
    """ref gate_ref: sqrtsoftplus scores in fp32, bias for selection only, weights / sum * 1.5."""
    scores = torch.nn.functional.softplus(f.float() @ gate_w.float().t()).sqrt()
    idx = (scores + gate_b).topk(topk, dim=-1).indices
    wts = scores.gather(-1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * scale
    return idx.int(), wts.float()


def shared_expert(f, w13, w2, limit=10.0):
    """Stock rounding points (ref expert_ffn_ref, VLLM_SM90): gate / up bf16, clamp, h bf16, out bf16.
    w13 = [w1; w3] concatenated: one GEMM for gate and up."""
    gu = f @ w13.t()
    gte, up = gu[:, : w13.shape[0] // 2], gu[:, w13.shape[0] // 2 :]
    up = up.clamp(-limit, limit)
    gte = gte.clamp(max=limit)
    h = (torch.nn.functional.silu(gte.float()) * up.float()).to(torch.bfloat16)
    return h @ w2.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--seqs", type=int, default=128)
    ap.add_argument("--routing", default="synth", choices=("synth", "router"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--energy-secs", type=float, default=2.0)
    ap.add_argument("--piece-energy", action="store_true", help="also W / MHz / J of every piece run alone in a loop")
    ap.add_argument("--profile", action="store_true", help="per-kernel device times of every piece (torch profiler)")
    ap.add_argument("--route2", action="store_true", help="device routing tables by route2.cu (fast, T <= 8192)")
    ap.add_argument("--n8", action="store_true", help="with --raw --route2: n_pad-8 ECs by the row-stream GEMM1 (gemm1r8)")
    ap.add_argument("--n8g2", action="store_true", help="with --n8: also GEMM2 of the n_pad-8 ECs by gemm2r8")
    ap.add_argument("--n8only", default="", help="timing: 'g1' / 'g2' runs only the n8 kernel (no tile stream) in that piece")
    ap.add_argument("--dup", type=int, default=1, help="with --cap: repeat the captured rows k times (mixed-step proxy)")
    ap.add_argument("--cap", default=None, help="Stage 2d capture file (rank 0, same layer): f rebuilt from the captured "
                    "layer input / x_mid (ds41h.ref, VLLM_SM90), T = the captured step's rows; implies --routing router")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--raw", action="store_true", help="moe_raw.cu: kernels on ds41h's raw expert layout (serving path)")
    ap.add_argument("--gather", type=int, default=0,
                    help="0: per-EC token images (default, fastest); GEMM1 rows from x16 without images: 1 TMA tensor "
                         "rows (GEMM1 3.4x slower), 2 cp.async by the producer warp (GEMM1 +87 us, prep -26 us)")
    a = ap.parse_args()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    ext = bench.build_extr() if a.raw else bench.build_ext3()
    print(f"extension built; route max EC {ext.route_max_ec()}")
    if a.build_only:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ck = bench.Ckpt(a.model)
    A13, S13, A2, S2, raw = bench.load_layer(ck, a.layer, dev, a.cache)
    P13, P2 = bench.make_p(S13), bench.make_p(S2)
    if a.raw:
        w13r, s13r, w2r, s2r = bench.raw_layout(raw, dev)
        a.gather = 0
    w1, w3, w2, gate_w, gate_b = load_dense(ck, a.layer, a.rank, dev)
    w13 = torch.cat([w1, w3], 0).contiguous()
    T = a.seqs * 4
    torch.manual_seed(0)
    f = torch.randn(T, D, device=dev).to(torch.bfloat16)
    if a.cap:  # real FFN input: X = hc_post(captured inputs), a_pre = attn mixes of X, f = RMSNorm(collapse(x_mid, a_pre))
        from ds41h.ref import layer as RL
        from ds41h.ref.weights import default_reader
        fs = []
        for cp in a.cap.split(","):
            rec = torch.load(cp, weights_only=False)
            assert rec["layer"] == a.layer and rec["rank"] == 0, "capture file must be rank 0 of --layer"
            if "ffn_args" in rec:  # captures with ffn_args: stock's own f
                fs.append(rec["ffn_args"][0].to(dev).to(torch.bfloat16))
                continue
            rd = default_reader(a.model)
            cfg = RL.ModelConfig.from_hf(rd.config)
            gw = lambda n: rd.get(f"layers.{a.layer}.{n}").to(dev).float()  # noqa: E731
            ci, co = rec["in"], rec["out"]
            with RL.strict_fp32():
                X = RL.hc_post(ci["x"].to(dev), ci["residual"].to(dev), ci["post_mix"].to(dev), ci["res_mix"].to(dev), RL.VLLM_SM90)
                a_pre = RL.hc_mixes(X, gw("hc_attn_fn"), gw("hc_attn_scale"), gw("hc_attn_base"), norm_eps=cfg.norm_eps,
                                    hc_eps=cfg.hc_eps, iters=cfg.hc_iters)[0]
                fs.append(RL.rmsnorm(RL.hc_collapse(co["residual"].to(dev), a_pre, RL.VLLM_SM90), gw("ffn_norm.weight"),
                                     cfg.norm_eps, RL.VLLM_SM90).to(torch.bfloat16))
        f = torch.cat(fs * a.dup).contiguous()
        T = f.shape[0]
        a.routing = "router"
        print(f"capture {a.cap}: T={T} ({len(fs)} file(s) x {a.dup}), |f| max {float(f.abs().max()):.3g}")
    if a.routing == "router":
        ids, wts = None, None  # set from the kernel router below
    else:
        i_, w_ = bench.synth_routing(a.seqs)
        ids, wts = i_.int().to(dev).contiguous(), w_.float().to(dev).contiguous()
    r2 = None
    if a.route2:
        from torch.utils.cpp_extension import load as _load
        r2 = _load(name="route2_ext", sources=[os.path.join(HERE, "route2.cu")], verbose=False,
                   extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"])
    max_ec = int(r2.route2_max_ec(T)) if r2 else int(ext.route_max_ec())
    EC_BYTES = 32 + 3 * 4 * bench.NMAX
    ecs = torch.zeros(max_ec * EC_BYTES, dtype=torch.uint8, device=dev)
    t1 = torch.zeros(max_ec * 9 * 24, dtype=torch.uint8, device=dev)
    t2 = torch.zeros(max_ec * 40 * 24, dtype=torch.uint8, device=dev)
    counts = torch.zeros(8, dtype=torch.int32, device=dev)
    ec8 = torch.zeros(max_ec, dtype=torch.int32, device=dev)
    logits = torch.empty(T, NE, dtype=torch.float32, device=dev)
    rids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
    rwts = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
    rts = torch.zeros(int(r2.route2_scratch_ints()) if r2 else int(ext.route_scratch_ints()), dtype=torch.int32, device=dev)
    gu = torch.empty(T, 2 * I_RANK, dtype=torch.bfloat16, device=dev)
    hsh = torch.empty(T, I_RANK, dtype=torch.bfloat16, device=dev)
    slotmap = torch.zeros(T * TOPK, dtype=torch.int32, device=dev)
    max_np = T * TOPK + 7 * max_ec
    img = torch.zeros(max_np * 80 * 128, dtype=torch.uint8, device=dev)
    himg = torch.zeros(max_np * 9 * 128, dtype=torch.uint8, device=dev)
    ptok = torch.zeros(T, dtype=torch.int32, device=dev)
    x16 = torch.zeros(T + 1, D, dtype=torch.float16, device=dev)  # row T stays zero (padding slots)
    empty = torch.zeros(0, dtype=torch.float32, device=dev)
    inv = torch.zeros(T, dtype=torch.float32, device=dev)
    Y = torch.zeros(T * TOPK, D, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(T, D, dtype=torch.bfloat16, device=dev)
    dummy = torch.zeros(1, dtype=torch.float32, device=dev)

    def r_route():
        if r2 is not None:
            if a.n8:
                r2.route2x(ids, wts, T, ecs, t1, t2, counts, slotmap, rts, sms, empty, ec8)
            else:
                r2.route2(ids, wts, T, ecs, t1, t2, counts, slotmap, rts, sms, empty)
            return
        ext.route(ids, wts, T, ecs, t1, t2, counts, slotmap, rts, sms * (1 if a.raw else int(ext.gemm_ctas())),
                  inv if a.gather else empty)

    def r_prep():  # gather: prescale + f16 copy only (before routing: the EC records take 2^-p from it)
        if a.raw:
            ext.prepr(f, ecs, counts, max_ec, img, ptok, inv)
        elif a.gather:
            ext.prescale16(f, ptok, inv, x16)
        else:
            ext.prepx(f, ecs, counts, max_ec, img, ptok, inv)

    def r_g1():
        if a.raw and a.n8:
            if a.n8only != "g1":
                ext.gemm1rs(w13r, s13r, ecs, t1, counts, img, himg, 0, 1)
            ext.gemm1r8(w13r, s13r, ecs, ec8, counts, img, himg, 0)
        elif a.raw:
            ext.gemm1r(w13r, s13r, ecs, t1, counts, img, himg, 0)
        elif a.gather:
            ext.gemm1g(A13, P13, ecs, t1, counts, x16, himg, inv, 0, a.gather)
        else:
            ext.gemm1x(A13, P13, ecs, t1, counts, img, himg, inv, dummy, 0)

    def r_g2():
        if a.raw and (a.n8g2 or a.n8):  # route2x tile lists hold only n_pad > 8 ECs: GEMM2 needs gemm2r8 too
            if a.n8only != "g2":
                ext.gemm2rs(w2r, s2r, ecs, t2, counts, himg, Y, 0, 1)
            ext.gemm2r8(w2r, s2r, ecs, ec8, counts, himg, Y, 0)
        elif a.raw:
            ext.gemm2r(w2r, s2r, ecs, t2, counts, himg, Y, 0)
        else:
            ext.gemm2x(A2, P2, ecs, t2, counts, himg, himg, inv, Y, 0)

    sh_out = [None]

    def r_sh():  # fused gate|up GEMM (cuBLAS) -> k_swiglu -> down GEMM (cuBLAS); stock rounding points
        torch.mm(f, w13.t(), out=gu)
        ext.swiglu(gu, hsh)
        sh_out[0] = hsh @ w2.t()

    def r_router():  # bf16 GEMM with fp32 output (cuBLAS) + k_router_top; synthetic runs keep their own ids
        torch.mm(f, gate_w.t(), out_dtype=torch.float32, out=logits)
        ext.router_top(logits, gate_b, rids, rwts, 1.5)

    def r_comb():
        ext.combine(Y, slotmap, sh_out[0], out)

    r_router()
    torch.cuda.synchronize()
    ids_t, wts_t = router(f, gate_w, gate_b)  # torch reference of the router
    router_ids_match = float((rids == ids_t).all(-1).float().mean())
    router_w_maxrel = float(((rwts - wts_t).abs() / wts_t.abs().clamp_min(1e-30)).max())
    if a.routing == "router":
        ids, wts = rids.clone(), rwts.clone()
    order = [("router", r_router), ("prep", r_prep), ("route", r_route)] if a.gather else \
        [("router", r_router), ("route", r_route), ("prep", r_prep)]
    pieces = order + [ ("gemm1", r_g1), ("gemm2", r_g2), ("shared", r_sh), ("combine", r_comb)]

    def run():
        for _, fn in pieces:
            fn()

    run()
    torch.cuda.synchronize()
    cnt = counts.cpu().tolist()
    himg_check = {}
    if a.n8:
        h_n8 = himg.clone()
        ext.gemm1r(w13r, s13r, ecs, t1, counts, img, himg, 0)  # tile stream for all ECs, same inputs
        torch.cuda.synchronize()
        hb = himg[: cnt[4]].view(torch.float16).float()
        hn = h_n8[: cnt[4]].view(torch.float16).float()
        d = (hb - hn).abs()
        ul = torch.pow(2.0, torch.floor(torch.log2(hb.abs().clamp_min(2.0 ** -14))) - 10)
        himg_check = dict(n8_ecs=cnt[6], h_frac_identical=float((d == 0).float().mean()),
                          h_max_ulp=float((d / ul).max()), h_frac_le1=float(((d / ul) <= 1).float().mean()))
        print("H image n8 vs tile stream:", himg_check)
        himg.copy_(h_n8)
    n_ec = cnt[0]
    # ---- routing tables vs the host builder (same expert token lists, in (token, slot) order)
    arr_h, *_ = bench.build_tiles(ids.cpu().long(), wts.cpu(), ec_inv=True)
    ec_np = np.frombuffer(ecs.cpu().numpy().tobytes(), dtype=bench.EC_DT, count=n_ec)
    ec_np_all = ec_np
    dev_lists, host_lists = {}, {}
    for r in ec_np:
        dev_lists.setdefault(int(r["e"]), []).extend(zip(r["tok"][: r["n_real"]].tolist(), r["wt"][: r["n_real"]].tolist()))
    for r in arr_h:
        host_lists.setdefault(int(r["e"]), []).extend(zip(r["tok"][: r["n_real"]].tolist(), r["wt"][: r["n_real"]].tolist()))
    routing_ok = dev_lists == host_lists and n_ec == len(arr_h)
    t1_np = t1[: cnt[1] * 24].cpu().numpy().view(np.int32).reshape(-1, 6)
    big = [x for x in range(n_ec) if not (a.n8 and int(ec_np_all[x]["n_pad"]) == 8)] if a.n8 else list(range(n_ec))
    tiles_ok = sorted(map(tuple, t1_np[:, :2].tolist())) == sorted((x, q) for x in big for q in range(9))
    # ---- fp64 reference: kernel roundings (x -> f16 with 2^p, h x 2^8 -> f16), each expert output -> bf16 (stock)
    p = ptok.double()
    xr = (f.double() * torch.pow(2.0, p).view(-1, 1)).half().double() / torch.pow(2.0, p).view(-1, 1)
    routed = torch.zeros(T, D, dtype=torch.float64, device=dev)
    for r in ec_np:
        e, n = int(r["e"]), int(r["n_real"])
        tok = torch.as_tensor(r["tok"][:n], device=dev).long()
        wt = torch.as_tensor(r["wt"][:n], device=dev).double()
        rw1, rw3, rw2, s1, s3, s2 = (z.to(dev) for z in raw[e])
        gt = (xr[tok] @ bench.dequant(rw1, s1).t()).clamp(max=10)
        up = (xr[tok] @ bench.dequant(rw3, s3).t()).clamp(-10, 10)
        h = gt * torch.sigmoid(gt) * up * wt.view(-1, 1)
        h = (h * 256).half().double() / 256
        y = (h @ bench.dequant(rw2, s2).t()).to(torch.bfloat16).double()
        routed.index_add_(0, tok, y)
    sh_ref = shared_expert(f, w13, w2)  # torch elementwise version: k_swiglu must reproduce it
    sh_match = float((sh_ref == sh_out[0]).float().mean())
    sh64 = sh_out[0].double()
    ref = (routed + sh64).to(torch.bfloat16)
    # bf16 ulps of the difference, relative to the reference value's ulp
    diff = (out.double() - ref.double()).abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(ref.double().abs().clamp_min(2.0 ** -126))) - 7)
    ulps = (diff / ulp).flatten()
    err = dict(max_ulp=float(ulps.max()), frac_exact=float((ulps == 0).double().mean()),
               frac_le1=float((ulps <= 1).double().mean()), max_abs=float(diff.max()),
               ref_max=float(ref.double().abs().max()))
    # ---- timing per piece
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(pieces) + 1)]
    tt = [0.0] * len(pieces)
    for _ in range(a.reps):
        ev[0].record()
        for i, (_, fn) in enumerate(pieces):
            fn()
            ev[i + 1].record()
        torch.cuda.synchronize()
        for i in range(len(pieces)):
            tt[i] += ev[i].elapsed_time(ev[i + 1]) * 1e3 / a.reps
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                run()
            torch.cuda.synchronize()
        print("per-kernel device time (us per call):")
        for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total):
            if ev.device_time_total > 0:
                print(f"  {ev.key[:70]:70s} {ev.device_time_total / ev.count:8.1f} x {ev.count // 10}")
    # ---- energy of the whole block
    W = J = mhz = None
    try:
        import pynvml
        pynvml.nvmlInit()
        pr = torch.cuda.get_device_properties(dev)
        hdl = pynvml.nvmlDeviceGetHandleByPciBusId(f"{pr.pci_domain_id:08x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0")
        e0, t0, n, clk = pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl), time.time(), 0, []
        while time.time() - t0 < a.energy_secs:
            for _ in range(8):
                run()
            n += 8
            clk.append(pynvml.nvmlDeviceGetClockInfo(hdl, pynvml.NVML_CLOCK_SM))
            torch.cuda.synchronize()
        el = time.time() - t0
        Jt = (pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl) - e0) / 1000.0
        W, J, mhz = Jt / el, Jt / n, sum(clk) / len(clk)
    except Exception as ex:
        print(f"energy skipped: {ex}")
    piece_e = {}
    if a.piece_energy:
        try:
            for name, fn in pieces:
                e0, t0, n, clk = pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl), time.time(), 0, []
                while time.time() - t0 < max(0.5, a.energy_secs / 2):
                    for _ in range(16):
                        fn()
                    n += 16
                    clk.append(pynvml.nvmlDeviceGetClockInfo(hdl, pynvml.NVML_CLOCK_SM))
                    torch.cuda.synchronize()
                el = time.time() - t0
                Jt = (pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl) - e0) / 1000.0
                piece_e[name] = dict(W=round(Jt / el, 1), mJ=round(Jt / n * 1e3, 2), us=round(el / n * 1e6, 1),
                                     mhz=round(sum(clk) / len(clk)))
        except Exception as ex:
            print(f"piece energy skipped: {ex}")
    b1 = n_ec * 9 * (80 * 4096 + 160 * 128)
    b2 = n_ec * 40 * (9 * 4096 + 18 * 128)
    res = dict(layer=a.layer, rank=a.rank, T=T, routing=a.routing, gather=a.gather, raw=a.raw, n_ec=n_ec, touched=len(dev_lists),
               routing_ok=routing_ok, tiles_ok=tiles_ok, **{f"{k}_us": v for (k, _), v in zip(pieces, tt)},
               block_us=sum(tt), moe_kernels_us=tt[3] + tt[4], gemm1_TBps=b1 / (tt[3] * 1e-6) / 1e12,
               gemm2_TBps=b2 / (tt[4] * 1e-6) / 1e12, W=W, J_per_block=J, mhz=mhz, shared_match=sh_match,
               router_ids_match=router_ids_match, router_w_maxrel=router_w_maxrel,
               cfg={k: os.environ[k] for k in os.environ if k.startswith("MS3_")}, piece_energy=piece_e, n8=bool(a.n8),
               **himg_check, **err)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
