#!/usr/bin/env python3
"""Decode-cycle wall-time budget (rank trace): every elementary interval of a cycle is split equally among the kernels
running in it ('idle' when none), attributed to (phase, category). Phases: target graph / eager sample+verify / drafter
graph / eager input prep (by host launch API + order). Shares add up to the cycle time.
  python cycle_budget.py TRACE.json.gz [--top-other N]"""
import gzip, json, re, sys
from collections import defaultdict
path = sys.argv[1]; topn = int(sys.argv[2]) if len(sys.argv) > 2 else 12
t = json.load(gzip.open(path)); ev = t["traceEvents"]
ks = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
rt = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
        c = e.get("args", {}).get("correlation")
        if c is not None: rt[c] = e.get("name", "?")
idx = [i for i, k in enumerate(ks) if "layer_kernel" in k["name"]]
runs, cur = [], [idx[0]]
for p, q in zip(idx, idx[1:]):
    if ks[q]["ts"] - (ks[p]["ts"] + ks[p]["dur"]) > 600: runs.append(cur); cur = [q]
    else: cur.append(q)
runs.append(cur); runs = [r for r in runs if len(r) >= 20]
CATS = [("decoder layer_kernel", r"layer_kernel"), ("prologue", r"prologue_kernel"), ("skinny (prologue path)", r"skinny_kernel"),
        ("sparse attention", r"sparse_decode_kernel"), ("LM head / verify GEMV", r"nvjet_sm90_tst_256x8"),
        ("drafter kernels", r"dmoe::|dhead::|dmain::|draft_markov|drouter|dctx|_ds41h_prepare_dflash"),
        ("ll all-reduce / all-gather", r"ll_all_reduce|ll_all_gather"),
        ("indexer / compressor / candidates", r"paged_mqa_logits|cooperative_topk|mask_candidates|select_blocks|indexer|compress|_rope_quant_insert|hash_ids|engram"),
        ("mHC (stock tilelang / tf32)", r"mhc_|hc_prenorm"), ("dense GEMM (humming / nvjet / cutlass)", r"humming|nvjet|cublasLt|cutlass|deep_gemm"),
        ("sampler / reject (vp_sampling)", r"_target_|_post_update|gather_stage|_sample|gumbel|reject"),
        ("input prep metadata", r"swa_indices|slot_mapping|block_tables|token_request|pos_seq_lens|combine_sampled|expand_idx|gather_lookback|ring_slot"),
        ("memcpy / elementwise / misc", r".")]
CATS = [(n, re.compile(p)) for n, p in CATS]
def cat(n):
    for c, p in CATS:
        if p.search(n): return c
def short(n, w=64):
    n = re.sub(r"\(anonymous namespace\)::", "", n); n = re.sub(r"void ", "", n)
    n = re.sub(r"<.*", "<..>", n) if len(n) > w else n
    return n[:w]
share = defaultdict(float); kshare = defaultdict(float); ncyc = 0; tot = 0.0
for r0, r1 in zip(runs, runs[1:]):
    g0 = ks[r0[0]].get("args", {}).get("correlation"); g1 = ks[r1[0]].get("args", {}).get("correlation")
    i0 = r0[0]
    while i0 > 0 and ks[i0 - 1].get("args", {}).get("correlation") == g0: i0 -= 1
    i1 = r1[0]
    while i1 > 0 and ks[i1 - 1].get("args", {}).get("correlation") == g1: i1 -= 1
    t0, t1 = ks[i0]["ts"], ks[i1]["ts"]
    if t1 - t0 > 100000: continue
    ncyc += 1; tot += t1 - t0
    # phase per kernel: 0 target graph, then by launch order
    phase_of = {}; ph = 0; graphs = 0; prev_kind = None
    for k in ks[i0:i1]:
        c = k.get("args", {}).get("correlation"); kind = "graph" if rt.get(c) == "cudaGraphLaunch" else "eager"
        key = (kind, c if kind == "graph" else None)
        if prev_kind is None: label = "target graph"
        elif key != prev_kind:
            if kind == "graph": graphs += 1; label = "drafter graph" if graphs == 1 else f"graph{graphs}"
            else: label = "eager sample/verify" if graphs == 0 else "eager input prep"
        phase_of[id(k)] = label; prev_kind = key
    pts = []
    for k in ks[i0:i1 + 400]:
        a, b = max(k["ts"], t0), min(k["ts"] + k["dur"], t1)
        if b > a:
            lab = (phase_of.get(id(k), "target graph"), cat(k["name"]), short(k["name"]))
            pts.append((a, 1, lab)); pts.append((b, -1, lab))
    pts.sort(key=lambda p: (p[0], p[1]))
    active = defaultdict(int); last = t0
    for x, d, lab in pts:
        if x > last:
            names = [m for m, cnt in active.items() if cnt > 0]
            if names:
                for m in names: share[(m[0], m[1])] += (x - last) / len(names); kshare[m] += (x - last) / len(names)
            else:
                share[("(idle)", "idle")] += x - last
            last = x
        active[lab] += d
    if t1 > last: share[("(idle)", "idle")] += t1 - last
print(f"{ncyc} cycles, cycle mean {tot / ncyc / 1000:.3f} ms; wall-time budget (us/cycle; overlaps split equally)")
byphase = defaultdict(float)
for (p, c), v in share.items(): byphase[p] += v
for p, pv in sorted(byphase.items(), key=lambda kv: -kv[1]):
    print(f"{pv / ncyc:8.1f}  {p}")
    for (p2, c), v in sorted(share.items(), key=lambda kv: -kv[1]):
        if p2 == p and p != "(idle)": print(f"      {v / ncyc:8.1f}  {c}")
print("top kernels in the target graph outside the ds41h decode kernels:")
oth = [(m, v) for m, v in kshare.items() if m[0] == "target graph" and m[1] not in ("decoder layer_kernel", "prologue", "sparse attention")]
for m, v in sorted(oth, key=lambda kv: -kv[1])[:topn]:
    print(f"      {v / ncyc:8.1f}  {m[2]}  [{m[1]}]")
