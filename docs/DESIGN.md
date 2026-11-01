# ds41h design: DeepSeek-V4.1-Flash-1017 on 4x H200 NVL (sm90)

Original targets, on one machine with the GPUs power-capped at 200 W (the kernels were designed against this limit):
- prefill >= 20,000 tok/s (BRRR bench PP, 16K prompt; c1 per-stream and c16 aggregate)
- decode >= 600 tok/s single stream (BRRR bench TG p50 at c1, thinking off, DSpark on)

Stock vLLM 0.32.0 on this machine at that limit: prefill ~10.8k tok/s, decode ~100 tok/s (no DSpark).

## Ground rules
- Stock vLLM 0.32.0 PyPI wheel, no fork. ds41h patches stock modules at import time (`ds41h/hook.py`),
  activated by `DS41H=1` via `runtime/sitecustomize.py`. Every component has its own
  flag (`ds41h/flags.py`) and must fall back to the stock path when ineligible or disabled.
- TP=4 for everything (attention heads 16/rank, MoE intermediate 576/rank, vocab-parallel head). No EP.
- CUDA graphs: `cudagraph_mode=FULL_DECODE_ONLY` (prefill eager => CED trim active, avoids #62811).
  Model is not torch.compiled; kernels need not be torch custom ops, but must: run on the current stream,
  allocate nothing during capture (preallocate in `process_weights_after_loading` / first eager call),
  and have no host-dependent control flow that changes between replays.
- Build: `ds41h/cuext.py` (`torch.utils.cpp_extension.load`, `-gencode=arch=compute_90a,code=sm_90a`).
  Shared device helpers: `ds41h/kernels/{common,sm90,host}.cuh`. Keep each component in its own files.
- `source scripts/env.sh` before anything; MAX_JOBS=6 by default (kernel builds need several GB of RAM per unit).

## Hopper facts that drive the kernels
- No native FP8 mma.sync on sm90 (lowered to FP8->F16 unpack + HMMA). Native FP8 = wgmma only.
- Decode = memory bound at M<=8 tokens; per-step weight bytes ~8.5 GB/GPU at k=5 DSpark
  (routed experts ~5.5 GB, attention dense ~1.5 GB, shared 0.35, head 0.33, gate+HC 0.3, drafter ~0.45).
- At 200 W SM clocks may drop well below 1785 MHz, so ALU work per weight byte (FP4 dequant) and
  serial latency (syncs, Sinkhorn, reductions) matter. Minimize instructions per weight byte.
- FP4 e2m1 -> e4m3 bit trick is exact (x2^-6); FP4 -> f16 direct is exact (x2^-14). See sm90.cuh.

## Weight layouts (one copy only; routed experts are ~72 GB/GPU)
- Routed + shared experts: RAW checkpoint layout, TP4 slice, no repack:
  w13 [E, 2*576, 5120/2] uint8 (e2m1 pairs, low nibble = even k), s13 [E, 2*576, 5120/32] ue8m0;
  w2 [E, 5120, 576/2] uint8, s2 [E, 5120, 576/32] ue8m0. (Rows 0..575 of w13 = w1 (gate), 576..1151 = w3 (up).)
  The ds41h MoE method keeps this layout for both the decode megakernel and the prefill W4A8 grouped GEMM.
- Dense FP8 (attention, shared expert, indexer, engram wkv): checkpoint e4m3 [N, K] with ue8m0 [N/32, K/32].
  Stock Humming repacks module weights, so decode kernels read their own copy loaded from the checkpoint
  (~2 GB/GPU total), like BRRR's skinny path.

## Decode fast path (per decoder layer, M = 1 + k tokens per request, c1 => M <= 8)
Called from a patched `DeepseekV4DecoderLayer.forward` when the step is decode-only and M <= MAX_DECODE_TOKENS.
Interfaces at layer boundaries must match stock tensors exactly (dtype/shape) because stock layers
(Engram layers 1/14, any ineligible layer) interleave with fused ones in the same step.
1. PROLOGUE kernel: previous-FFN mHC post (x*post + comb^T*residual) -> attention mHC pre (RMS, GEMV with
   hc_attn_fn [24, 20480] f32, Sinkhorn 20 it) -> collapse with previous pre -> RMSNorm(attn_norm) -> `a`
   -> fused wq_a|wkv GEMM (1792x5120 FP8) -> q_norm (1280) / kv_norm (512) -> KV RoPE + FP8 quant + insert into
   the stock sm90 SWA cache record format -> wq_b (TP slice 8192x1280) + q RoPE -> q [M,16,512] bf16.
   Also emits `a` and `qr` for stock compressor/indexer calls on KV-source/index-source layers.
2. ATTN kernel: sparse decode over (128 SWA + <=512 top-k) keys for the 16 real local heads (stock pads
   to 64 heads), attention sink, reading stock sm90 cache records. Output o [M,16,512] bf16.
3. DECODER (layer) kernel: persistent kernel (port of BRRR layer_kernel): inverse RoPE on o -> wo_a
   (2 groups/rank) -> wo_b (K-sliced) -> in-kernel NVLink IPC all-reduce #1 -> attention mHC post + FFN mHC pre
   (Sinkhorn) -> RMSNorm(ffn_norm) -> gate (bf16 384x5120) -> sqrtsoftplus + bias top-6, *1.5/sum ->
   shared expert + routed experts (W4A16 or W4A8, raw layout) -> combine -> all-reduce #2.
   Returns exactly what stock forward returns (x reduced, residual, post_mix, res_mix, ffn_pre, aux).
4. SAMPLING: vocab-parallel DSpark drafting + block verification (port of BRRR vp_sampling.py).
5. CANDIDATES: radix-select top-k for indexer candidate blocks / top-512 (port of BRRR candidates.cu).
6. ALLREDUCE: low-latency NVLink IPC all-reduce for remaining small collectives (stock is FlashInfer trtllm).

## Prefill path
- MoE: ds41h W4A8 grouped GEMM on the raw layout: FP4->e4m3 dequant into smem (weight per-32 scales folded
  into the e4m3 values relative to a per-row-per-128 reference exponent: exact for spreads <= 2^13),
  activations e4m3 per token per 128, wgmma SS, promote every K=128. Routing/permutation/SwiGLU fused where
  possible.
- Dense linears: W8A8 FP8 (block activations) instead of stock W8A16.
- Attention prefill: avoid 16->64 head padding; indexer prefill on FP8 tensor cores.
- Everything measured first with the torch profiler; fix the biggest items first.

## Correctness
- `ds41h/ref/`: pure-torch reference of one decoder layer / DSpark built from the official
  `inference/model.py` semantics, on dequantized weights of one TP rank (or all ranks summed).
- Every kernel gets a GPU unit test in `tests/` comparing against the reference with real weights of a
  chosen layer loaded from safetensors (`DS41H_MODEL_DIR`), plus a random-input stress mode.
- End-to-end gate: full-vocab KL vs stock vLLM over 1,024 forced tokens (BRRR method) <= 0.012, top-1 >= 96%.
