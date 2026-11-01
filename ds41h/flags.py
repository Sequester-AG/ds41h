"""Environment flags. DS41H=1 enables the hook at all; each component has its own switch so it
can be benchmarked and KL-checked in isolation (default on once DS41H=1, set to 0 to disable).

Numerics policy (docs/EXACTNESS.md): with DS41H=1 the TARGET model is LOSSLESS by
default -- every target computation at least as precise as stock vLLM 0.32.0 on sm90 (16-bit activations into
every target GEMM, exact weight scales, stock norm / softmax / Sinkhorn / routing math; only fp32 summation order
may differ). Approximations are allowed only in the DSpark drafter (rejection sampling keeps outputs exact).
Lossy fast paths stay available as explicit opt-ins:
  * DS41H_LOSSY=1 restores the phase-3 fast defaults as a group (end-to-end: FP8-activation prefill
    linears / wo_a, coarse e4m3 decoder, W4A8 MoE outside the decoder kernel);
  * or set an individual switch (e.g. DS41H_PREFILL_LINEAR=1, DS41H_DECODER_DEFINES=COARSE=1,
    DS41H_MOE_IMPL=w4a8): a switch present in the environment always wins over the policy.
The policy is applied by writing the lossless values as environment DEFAULTS (os.environ.setdefault) before the
component modules read them, so the component modules need no policy code and spawned workers inherit it.
"""

import os


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


ENABLED = _flag("DS41H", "0")

# ------------------------------------------------------------------------------------------ numerics policy
LOSSY = ENABLED and _flag("DS41H_LOSSY", "0")
# switch -> lossless value (comment: what the lossy value does)
LOSSLESS_ENV = {
    "DS41H_PREFILL_LINEAR": "0",  # 1: dense prefill linears W8A8 (e4m3 1x128 activations, 32x32 -> 128x128 scale fold)
    "DS41H_PREFILL_WOA": "0",  # 1: prefill wo_a as DeepGEMM FP8 einsum (e4m3 activations; stock: bf16 bmm)
    "DS41H_PREFILL_MOE_FP8ACT": "0",  # 1: stock Humming MoE with block-FP8 activations (only without DS41H_MOE)
    "DS41H_PROLOGUE_MMA": "hmma",  # wgmma: e4m3 activations in the decode prologue + skinny GEMMs
}
# decoder.cu compile-time knobs (DS41H_DECODER_DEFINES) forced unless the spec names them explicitly.
# COARSE=1 with LOSSLESS=0 (decoder-v2 coarse routed path, m <= 8) stages x / h / shared-expert inputs as e4m3
# (RS-wgmma, LOSSY); COARSE=0 is the HMMA path: f16 activations (exact for bf16 values), FP8 / FP4 weights with
# exact ue8m0 scales. The decoder's lossless coarse path (R1F / R2F HMMA f16, knob LOSSLESS=1) keeps COARSE:
# once decoder.cu's compiled-in default is DS41H_DECODER_LOSSLESS 1 (= validated), the policy forces
# LOSSLESS=1 instead of COARSE=0 (read from the source text next to this file, no import).
def _decoder_lossless_default() -> bool:
    import re

    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "decoder.cu")) as f:
            m = re.search(r"^#define DS41H_DECODER_LOSSLESS (\d+)", f.read(), re.M)
        return bool(m and int(m.group(1)))
    except OSError:
        return False


DECODER_LOSSLESS_COARSE = _decoder_lossless_default()
LOSSLESS_DECODER_DEFINES = (
    {"LOSSLESS": "1", "ROUTED_MMA": "0", "DENSE_MMA": "0"}
    if DECODER_LOSSLESS_COARSE
    else {"COARSE": "0", "ROUTED_MMA": "0", "DENSE_MMA": "0"}
)
# MoE forward on the ds41h raw MXFP4 experts outside the decoder kernel (prefill, decode with M > the decoder
# limit, c >= 2 at k=5): lossless default 'w4a16' (W4A16 grouped GEMM: bf16 activations, FP4
# weights expanded exactly to bf16, fp32 accumulation, stock clamped SwiGLU / combine; graph-capturable).
# 'ref_eager' = exact bf16 per-expert torch reference for steps with prefill tokens (w4a16 elsewhere; slow, for
# cross-checks); 'w4a8' = e4m3 activations (LOSSY opt-in, the phase-3 default). DSpark drafter layers use w4a8
# (approximations allowed there). See ds41h/moe.py (vLLM glue).
MOE_IMPL_LOSSLESS = "w4a16"
MOE_IMPL_LOSSY = "w4a8"

POLICY_OVERRIDES: dict[str, str] = {}  # switches the environment set explicitly against the lossless policy


def _apply_numerics_policy() -> None:
    if not ENABLED:
        return
    if LOSSY:
        # DS41H_LOSSY=1 = the phase-3 (i12) fast group: the decoder's coarse path with e4m3 activations. decoder.cu's
        # compiled default became LOSSLESS=1 (R1F / R2F), so the lossy group must name LOSSLESS=0.
        spec = os.environ.get("DS41H_DECODER_DEFINES", "")
        items = [s.strip() for s in spec.split(",") if s.strip()]
        if DECODER_LOSSLESS_COARSE and not any(s.partition("=")[0] == "LOSSLESS" for s in items):
            os.environ["DS41H_DECODER_DEFINES"] = ",".join(items + ["LOSSLESS=0"])
        return
    for key, val in LOSSLESS_ENV.items():
        cur = os.environ.get(key)
        if cur is None:
            os.environ[key] = val
        elif cur.strip().lower() != val:
            POLICY_OVERRIDES[key] = cur
    spec = os.environ.get("DS41H_DECODER_DEFINES", "")
    items = [s.strip() for s in spec.split(",") if s.strip()]
    given = {s.partition("=")[0]: s.partition("=")[2] for s in items}
    for key, val in LOSSLESS_DECODER_DEFINES.items():
        if key not in given:
            items.append(f"{key}={val}")
        elif given[key] != val:
            POLICY_OVERRIDES[f"DS41H_DECODER_DEFINES:{key}"] = given[key]
    os.environ["DS41H_DECODER_DEFINES"] = ",".join(items)
    moe = os.environ.get("DS41H_MOE_IMPL")
    if moe and moe != MOE_IMPL_LOSSLESS:
        POLICY_OVERRIDES["DS41H_MOE_IMPL"] = moe


_apply_numerics_policy()
# MoE impl the ds41h raw-MXFP4 vLLM method uses for target layers (ds41h/moe.py glue); DS41H_MOE_IMPL overrides.
MOE_IMPL = os.environ.get("DS41H_MOE_IMPL") or (MOE_IMPL_LOSSY if LOSSY else MOE_IMPL_LOSSLESS)


def numerics_summary() -> str:
    """One line for the startup log: policy + effective switches + explicit lossy overrides."""
    if not ENABLED:
        return "ds41h off"
    eff = {k: os.environ.get(k, "(module default)") for k in LOSSLESS_ENV}
    head = "LOSSY (DS41H_LOSSY=1: phase-3 fast defaults)" if LOSSY else "LOSSLESS target (default)"
    over = f"; explicit overrides: {POLICY_OVERRIDES}" if POLICY_OVERRIDES else ""
    # lossless-class switches owned by the component modules (informational: which exact variant runs)
    info = {k: os.environ[k] for k in INFO_ENV if k in os.environ}
    dec = ("" if LOSSY else " [lossless decoder variant: " + ("coarse R1F/R2F (decoder.cu LOSSLESS=1 default)]"
           if DECODER_LOSSLESS_COARSE else "HMMA COARSE=0 (decoder.cu LOSSLESS=0)]"))
    return (f"numerics {head}: {eff}, DS41H_DECODER_DEFINES={os.environ.get('DS41H_DECODER_DEFINES', '')!r}{dec}, "
            f"moe impl {MOE_IMPL}, set: {info}{over}")


# Exact (lossless-class) variant switches of other modules, logged by numerics_summary when set explicitly.
INFO_ENV = ("DS41H_PREFILL_LINEAR16", "DS41H_PREFILL_LINEAR16_GEMM", "DS41H_PREFILL_WOA16", "DS41H_PREFILL_WOA16_GEMM",
            "DS41H_PREFILL_MHC", "DS41H_PREFILL_MHC_IMPL", "DS41H_PREFILL_ATTN", "DS41H_PREFILL_ATTN_GROUPED",
            "DS41H_PREFILL_WQA_SHARD", "DS41H_PROLOGUE_FAST", "DS41H_ATTN_EARLY", "DS41H_MOE_VARIANT",
            "DS41H_MOE_W4A16_VARIANT", "NCCL_MIN_NCHANNELS", "NCCL_BUFFSIZE", "DS41H_FAST_TOKENIZE", "DS41H_PREP_FUSE", "DS41H_ATTN_SIDE",
            "DS41H_HUMMING_DET", "DS41H_TOPK_DET", "DS41H_MOE_SMALL_T", "DS41H_MOE_SMALL_VARIANT", "DS41H_MOE_HYBRID", "DS41H_PROLOGUE_INDEXER",
            "DS41H_DETPROBE", "DS41H_PREFILL_CTX_TRIM",
            # decoder / drafter / sampling opt-ins of the certification sets (logged for provenance only)
            "DS41H_PROLOGUE_FIRST", "DS41H_DRAFT_TAIL", "DS41H_META_BATCH", "DS41H_META_DRAFT", "DS41H_SAMPLING_POST_FUSE",
            "DS41H_SAMPLING_NOISE_EARLY", "DS41H_GRAPH_OUT_BATCH", "DS41H_DRAFTER_MAIN_DST", "DS41H_DRAFTER_CTXKV_EARLY",
            "DS41H_DRAFTER_OPROJ", "DS41H_DRAFTER_LIN_S1", "DS41H_DRAFTER_L0PRO", "DS41H_DRAFTER_EMBED_EARLY",
            "DS41H_DRAFTER_FINAL_FUSE", "DS41H_DRAFTER_LIN_EARLY", "DS41H_DRAFTER_INS_PAR",
            # next-round drafter / verify set + capacity mode
            "DS41H_VERIFY_HEAD", "DS41H_VHEAD_VARIANT", "DS41H_DRAFT_TAIL_EARLY_COMBINE", "DS41H_SAMPLING_EARLY_GLUE",
            "DS41H_DRAFTER_SKIPCAT", "DS41H_DRAFTER_PRECAT", "DS41H_DRAFTER_MAIN_NORM_DST", "DS41H_DRAFTER_EMBED_EAGER",
            "DS41H_SAMPLING_RESID_PDL", "DS41H_DRAFT_TAIL_MERGE", "DS41H_ENGRAM_NVME", "DS41H_ENGRAM_NVME_PREFETCH",
            "DS41H_MOE_SMALL_AUTO_T", "DS41H_MOE_DEC")


# ------------------------------------------------------------------------------------------ components
ALLREDUCE = ENABLED and _flag("DS41H_ALLREDUCE")  # custom NVLink IPC all-reduce for small tensors
CANDIDATES = ENABLED and _flag("DS41H_CANDIDATES")  # radix-select top-k for indexer/candidates
ATTN = ENABLED and _flag("DS41H_ATTN")  # sm90 sparse decode attention, 16 real heads (no pad to 64)
PROLOGUE = ENABLED and _flag("DS41H_PROLOGUE")  # fused mHC-pre + norm + q/kv projections for decode
DECODER = ENABLED and _flag("DS41H_DECODER")  # persistent post-attention layer kernel for decode
SAMPLING = ENABLED and _flag("DS41H_SAMPLING")  # vocab-parallel DSpark draft + block verification
PREFILL = ENABLED and _flag("DS41H_PREFILL")  # prefill-path kernels (MoE / linears / attention)
DEBUG_COMPARE = ENABLED and _flag("DS41H_DEBUG_COMPARE", "0")  # run stock path too and compare
# Prologue decode path: hand the ds41h 16-head attention q / output buffers of n_local_heads (16) instead of
# zero-padding q to padded_heads (64) with F.pad (a fill + a copy kernel per layer, ~6.7 us/layer at the
# power-capped clock). Stock FlashMLA fallbacks re-pad (ds41h/attn.py). Needs DS41H_ATTN.
# Default ON (validated end to end).
DECODE_NOPAD = ATTN and PROLOGUE and _flag("DS41H_DECODE_NOPAD", "1")
MAX_DECODE_TOKENS = int(os.environ.get("DS41H_MAX_DECODE_TOKENS", "8"))  # fused-path eligibility
# Serving path (ds41h/patch_serving.py): chunked parallel tokenization of long chat prompts in the API
# server (bit-identical ids, proof + check in patch_serving.py). Default ON (350 W reference
# r350c: in-server verify 6/6 identical, CPU test 1559 calls identical, TTFT16K -9 ms); DS41H_FAST_TOKENIZE=0 = stock.
FAST_TOKENIZE = ENABLED and _flag("DS41H_FAST_TOKENIZE", "1")
# Decode input prep (ds41h/patch_glue.py): DeepseekSparseSWAMetadataBuilder decode metadata (is_valid ge +
# tail fill + window indices) in ONE Triton launch per SWA group instead of three (identical integers). Default OFF
# until gated end-to-end at 350 W.
PREP_FUSE = ENABLED and _flag("DS41H_PREP_FUSE", "0")
# Prologue path of index/kv-source layers (ds41h/patch_decoder.py _prologue_attention_side): run the
# compressor (kv_score GEMM + compress) and the compressed-cache insert on the layer's stock aux stream, overlapped
# with the indexer's weights_proj / q path, exactly like stock _prepare_and_attn does (same kernels, same joins).
# Exactness gate passed at k=7 and k=5, -0.03..-0.04 ms/cycle. Opt-in here; scripts/serve_ds41h.sh turns it on.
ATTN_SIDE = ENABLED and _flag("DS41H_ATTN_SIDE", "0")
