#!/usr/bin/env python3
"""Aggregate GPU kernel time of one vLLM prefill step by category from torch-profiler chrome traces.

Input: the per-rank Kineto traces vLLM writes with
    --profiler-config '{"profiler":"torch","torch_profiler_dir":DIR,"torch_profiler_use_gzip":true,...}'
(scripts/serve.sh PROFILE=1 -> $PROFILE_DIR/*.pt.trace.json.gz), started/stopped around a
request with POST /start_profile and /stop_profile.

Steps: the worker wraps every engine step in a record_function named
    execute_context_<nreq>(<ctx_tokens>)_generation_<nreq>(<gen_tokens>)            (default)
    execute_<tokens>_context_<nreq>(sq<ctx_tokens>sk...)_generation_<nreq>(sq<gen>sk...)  (detailed)
(vllm/v1/worker/gpu_worker.py annotate_profile). A kernel belongs to the step whose CPU annotation
contains the runtime call that launched it (kernel args.correlation -> cuda_runtime/cuda_driver event),
so async scheduling / multi-stream overlap is attributed correctly; GPU-side gpu_user_annotation ranges
or idle-gap splitting are used as fallbacks (--split gap).

Categories (first matching regex on the kernel name wins; override/extend with --rules JSON):
  allreduce   NCCL / custom / FlashInfer / ds41h all-reduce, all-gather, reduce-scatter
  indexer     sparse indexer: MQA logits, top-k per row, indexer q/k prep, candidates
  attention   FlashMLA sparse prefill/decode, KV gather/dequant, q/kv rope + cache insert, o-proj rope
  moe gemm    routed-expert GEMMs and MoE glue (routing top-k, permute, quant, combine, align)
  mhc         mHC pre/post (TileLang), Sinkhorn, hc prenorm GEMM
  engram      Engram hash / lookup / post-wkv kernels (the wkv GEMM itself counts as dense gemm)
  dense gemm  everything else that is a GEMM: Humming dense, DeepGEMM fp8_gemm/einsum, cuBLAS/nvjet,
              CUTLASS, ds41h skinny / act_quant_128
  other       norms, activations, copies, memsets, elementwise, sampling, ...

Usage:
  python scripts/prefill_profile.py DIR_OR_FILES... [--step largest|first|N] [--list-steps] [--top 8]
         [--split annotation|gpu_annotation|gap] [--gap-ms 3] [--json out.json] [--layers 20]
Example (4 ranks, the biggest prefill step, averaged):
  python scripts/prefill_profile.py $PROFILE_DIR --step largest --top 6
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_RULES: list[tuple[str, str]] = [
    ("allreduce", r"nccl|all_?reduce|allreduce|cross_device_reduce|one_?shot|two_?shot|lamport|AllReduce|all_?gather|AllGather"
                  r"|reduce_?scatter|ReduceScatter|SendRecv|Broadcast|custom_ar|ds41h_ar|ar_fusion|allreduce_fusion"),
    ("indexer", r"mqa_logits|indexer|top_?k_per_row|topk_per_row|radix_select|candidate|index_k|fused_indexer_q|sparse_indexer"
                r"|topk_transform|global_topk"),
    ("attention", r"flash_?mla|sparse_attn_fwd|sparse_decode|sparse_prefill|flash_fwd|fmha|FlashAttn|flashattn|mla_"
                  r"|dequantize_and_gather|gather_k_cache|combine_topk_swa|qnorm_rope|rope_quant_insert|kv_rope|kv_insert"
                  r"|inv_rope|fused_q_kv_rmsnorm|indices_and_lens|ds41h_attn|attn_fwd|splitkv_mla|get_mla"),
    ("moe gemm", r"moe|expert|humming.*(indexed|grouped)|topk_bias|dsv4_topk|routing|align_block|permute|unpermute|combine_slots"),
    ("mhc", r"mhc|sinkhorn|hc_pre|hc_post|prenorm|hc_head|hyper_?conn"),
    ("engram", r"engram|hash_ids"),
    ("dense gemm", r"gemm|humming|nvjet|cutlass|cublas|xmma|wgmma|matmul|bmm|einsum|sm90_fp8|Kernel2|ampere_|sm90_|skinny"
                   r"|act_quant_128|splitKreduce"),
]
CATEGORIES = ["moe gemm", "dense gemm", "attention", "indexer", "mhc", "allreduce", "engram", "other"]
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
STEP_RE = re.compile(
    r"execute_(?:(\d+)_)?context_(\d+)\((?:sq)?(\d+)[^)]*\)_generation_(\d+)\((?:sq)?(\d+)"
)


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def find_traces(args_paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, args_paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.name.endswith((".json", ".json.gz")) and "trace" in q.name)
        else:
            out.append(p)
    if not out:
        sys.exit("no trace files found")
    return out


def compile_rules(extra: str | None) -> list[tuple[str, re.Pattern]]:
    rules = list(DEFAULT_RULES)
    if extra:
        user = json.loads(Path(extra).read_text()) if Path(extra).exists() else json.loads(extra)
        rules = [(c, r) for c, r in user.items()] + rules  # user rules take precedence
    return [(c, re.compile(r, re.I)) for c, r in rules]


def categorize(name: str, rules) -> str:
    for cat, rx in rules:
        if rx.search(name):
            return cat
    return "other"


class Step:
    def __init__(self, name: str, t0: float, t1: float):
        self.name, self.t0, self.t1 = name, t0, t1
        m = STEP_RE.search(name)
        self.ctx_tokens = int(m.group(3)) if m else 0
        self.gen_tokens = int(m.group(5)) if m else 0
        self.kernels: list[dict] = []

    @property
    def tokens(self) -> int:
        return self.ctx_tokens + self.gen_tokens

    def label(self) -> str:
        return f"{self.name[:90]} (ctx {self.ctx_tokens}, gen {self.gen_tokens})"


def split_steps(trace: dict, mode: str, gap_ms: float) -> list[Step]:
    ev = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    kernels = [e for e in ev if e.get("cat") in GPU_CATS]
    kernels.sort(key=lambda e: e["ts"])
    steps: list[Step] = []
    if mode in ("annotation", "auto"):
        ann = [e for e in ev if e.get("cat") == "user_annotation" and STEP_RE.search(e.get("name", ""))]
        if ann:
            ann.sort(key=lambda e: e["ts"])
            steps = [Step(a["name"], a["ts"], a["ts"] + a["dur"]) for a in ann]
            launch_ts: dict[int, float] = {}
            for e in ev:
                if e.get("cat") in ("cuda_runtime", "cuda_driver"):
                    c = (e.get("args") or {}).get("correlation")
                    if c is not None:
                        launch_ts[c] = e["ts"]
            starts = [s.t0 for s in steps]
            orphans = 0
            for k in kernels:
                c = (k.get("args") or {}).get("correlation")
                t = launch_ts.get(c)
                if t is None:
                    orphans += 1
                    continue
                i = bisect.bisect_right(starts, t) - 1
                if i >= 0 and t <= steps[i].t1:
                    steps[i].kernels.append(k)
            if orphans:
                print(f"  note: {orphans} GPU events without a launch record were ignored", file=sys.stderr)
            return steps
        if mode == "annotation":
            sys.exit("no execute_* step annotations found (profile with the torch profiler CPU activity on)")
    if mode in ("gpu_annotation", "auto"):
        gann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and STEP_RE.search(e.get("name", ""))]
        if gann:
            gann.sort(key=lambda e: e["ts"])
            steps = [Step(a["name"], a["ts"], a["ts"] + a["dur"]) for a in gann]
            starts = [s.t0 for s in steps]
            for k in kernels:
                i = bisect.bisect_right(starts, k["ts"]) - 1
                if i >= 0 and k["ts"] <= steps[i].t1:
                    steps[i].kernels.append(k)
            return steps
    # idle-gap splitting on the GPU timeline
    cur: Step | None = None
    end = -1e30
    for k in kernels:
        if cur is None or k["ts"] - end > gap_ms * 1e3:
            cur = Step(f"gap-step@{k['ts'] / 1e3:.1f}ms", k["ts"], k["ts"])
            steps.append(cur)
        cur.kernels.append(k)
        end = max(end, k["ts"] + k.get("dur", 0))
        cur.t1 = end
    return steps


def pick_step(steps: list[Step], how: str) -> Step:
    prefill = [s for s in steps if s.ctx_tokens > 0 and s.kernels] or [s for s in steps if s.kernels]
    if not prefill:
        sys.exit("no step with GPU kernels")
    if how == "largest":
        return max(prefill, key=lambda s: (s.ctx_tokens, sum(k.get("dur", 0) for k in s.kernels)))
    if how == "first":
        return prefill[0]
    return prefill[int(how)]


def busy_union(kernels: list[dict]) -> float:
    iv = sorted((k["ts"], k["ts"] + k.get("dur", 0)) for k in kernels)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def aggregate(step: Step, rules) -> dict:
    per_cat: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    per_kernel: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k in step.kernels:
        name = k.get("name", "?")
        cat = "other" if k.get("cat") != "kernel" else categorize(name, rules)
        d = float(k.get("dur", 0))
        per_cat[cat] += d
        count[cat] += 1
        per_kernel[(cat, name)].append(d)
    t0 = min(k["ts"] for k in step.kernels)
    t1 = max(k["ts"] + k.get("dur", 0) for k in step.kernels)
    return {
        "step": step.label(),
        "ctx_tokens": step.ctx_tokens,
        "gen_tokens": step.gen_tokens,
        "gpu_span_ms": (t1 - t0) / 1e3,
        "gpu_busy_ms": busy_union(step.kernels) / 1e3,
        "kernel_sum_ms": sum(per_cat.values()) / 1e3,
        "n_kernels": len(step.kernels),
        "categories": {c: {"ms": per_cat.get(c, 0.0) / 1e3, "n": count.get(c, 0)} for c in CATEGORIES},
        "top": {
            c: sorted(((n, sum(v) / 1e3, len(v)) for (cc, n), v in per_kernel.items() if cc == c), key=lambda t: -t[1])
            for c in CATEGORIES
        },
    }


def short(name: str, n: int = 70) -> str:
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 3] + "..."


def print_report(rank_name: str, r: dict, top: int, layers: int | None) -> None:
    print(f"\n== {rank_name}: {r['step']}")
    tok = r["ctx_tokens"] + r["gen_tokens"]
    tps = f", {tok / (r['gpu_span_ms'] / 1e3):,.0f} tok/s over the GPU span" if tok else ""
    print(f"   GPU span {r['gpu_span_ms']:.2f} ms, busy (union) {r['gpu_busy_ms']:.2f} ms, kernel sum {r['kernel_sum_ms']:.2f} ms, "
          f"{r['n_kernels']} kernels{tps}")
    print(f"   {'category':<11s} {'ms':>9s} {'% sum':>7s} {'#':>6s}" + (f" {'ms/layer':>9s}" if layers else ""))
    for c in CATEGORIES:
        ms, n = r["categories"][c]["ms"], r["categories"][c]["n"]
        pct = 100 * ms / r["kernel_sum_ms"] if r["kernel_sum_ms"] else 0
        extra = f" {ms / layers:9.3f}" if layers else ""
        print(f"   {c:<11s} {ms:9.2f} {pct:6.1f}% {n:6d}{extra}")
    if top:
        for c in CATEGORIES:
            rows = r["top"][c][:top]
            if not rows:
                continue
            print(f"   -- {c}")
            for name, ms, n in rows:
                print(f"      {ms:8.2f} ms {n:5d}x  {short(name)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="trace files or directories (one trace per rank)")
    ap.add_argument("--step", default="largest", help="largest | first | <index among prefill steps>")
    ap.add_argument("--split", default="auto", choices=["auto", "annotation", "gpu_annotation", "gap"])
    ap.add_argument("--gap-ms", type=float, default=3.0, help="idle gap separating steps for --split gap")
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--top", type=int, default=5, help="top kernels per category (0 = none)")
    ap.add_argument("--layers", type=int, default=None, help="divide by this many layers (e.g. 20 full-row layers)")
    ap.add_argument("--rules", default=None, help='extra rules, JSON {"category": "regex"} or a file (checked first)')
    ap.add_argument("--json", default=None, help="write the per-rank results here")
    args = ap.parse_args()
    rules = compile_rules(args.rules)
    results = {}
    for path in find_traces(args.paths):
        steps = split_steps(load_trace(path), args.split, args.gap_ms)
        if args.list_steps:
            print(f"\n{path.name}: {len(steps)} steps")
            for i, s in enumerate(steps):
                kd = sum(k.get("dur", 0) for k in s.kernels) / 1e3
                print(f"  [{i:3d}] {s.label()}  kernels {len(s.kernels)}  sum {kd:.2f} ms")
            continue
        r = aggregate(pick_step(steps, args.step), rules)
        results[path.name] = r
        print_report(path.name, r, args.top, args.layers)
    if len(results) > 1:
        print(f"\n== mean over {len(results)} ranks")
        print(f"   {'category':<11s} {'ms':>9s} {'min':>8s} {'max':>8s}")
        for c in CATEGORIES:
            v = [r["categories"][c]["ms"] for r in results.values()]
            print(f"   {c:<11s} {statistics.mean(v):9.2f} {min(v):8.2f} {max(v):8.2f}")
        spans = [r["gpu_span_ms"] for r in results.values()]
        print(f"   GPU span mean {statistics.mean(spans):.2f} ms (min {min(spans):.2f}, max {max(spans):.2f})")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
