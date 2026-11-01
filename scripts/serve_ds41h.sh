#!/bin/bash
# Tuned ds41h launcher: the validated configuration on top of scripts/serve.sh. Port 8100, localhost only, unless EXTRA overrides it.
#
#   MODEL=<dir> scripts/serve_ds41h.sh     LATENCY line (default): DSpark k=7 + moe_dec (DS41H_MOE_DEC=1: routed MoE of multi-request
#                                    decode, bit-identical to the small16 kernels, docs/EXACTNESS.md); Engram on NVMe.
#   THROUGHPUT=1 ...                 THROUGHPUT line: k=3 + moe_dec + DS41H_ATTN_GROUPED=1 (> 8 decode rows: 4 rows x 16 heads per
#                                    FlashMLA tile over the union of their keys; lossless) for concurrent long-context loads.
#   SPEC_K=<n> ...                   another draft length (k=5 was the line for the published high-concurrency numbers).
#   ENGRAM=nvme (default) | host | hbm   Engram n-gram tables on NVMe (PREFETCH=1 = prompt lookahead) | in ONE pinned host-RAM copy
#                                    (DS41H_ENGRAM_HOST=1: ~195 GiB in /dev/shm; bit-identical rows) | in HBM (stock; ~86K KV tokens).
#                                    docs/ENGRAM.md has the requirements of each mode.
#   CONC=0 ...                       CAPTURE=64 MAX_SEQS=16 (decode graphs for <= 8 streams; c1 benchmarks)
#   MAXLEN=<n> MAX_SEQS=<n> CAPTURE=<n> GPU_UTIL=<x> EXTRA="..."   passed through to serve.sh
#   VHEAD=0 ...                      plain cuBLAS verify head (frees 248 MB/GPU of KV; also lossless)
#   PREFETCH=1 ...                   Engram prompt lookahead (ENGRAM=nvme only; cold concurrent prefill +24 %; numerics-neutral)
#   DS41H_TORCH_EXT_DIR=<dir> GDEC_BUILD=<dir>   JIT cache / grouped-decode build;  DRYRUN=1: print the env only
#   KVRAM=<GiB>|auto ...             host-RAM KV tier, default OFF: stock vLLM 0.32 OffloadingConnector (CPU backend), ONE host copy
#                                    shared by the 4 TP ranks (KVRAM_DEDUP=1: plugins/kvram_ext.py marks V4.1's single-latent-head specs
#                                    max_tp_shards=1 -> vLLM's replicated layout; the 4 ranks' KV is byte-identical), write-through, LRU,
#                                    async restores that count as cached tokens. Forces expandable_segments off.
#                                    FAIL-SAFE SIZE: the region is a pre-faulted /dev/shm file; plugins/kvram_budget.py caps it at
#                                    min(tmpfs free, tmpfs quota left) - Engram table bytes still to allocate (ENGRAM=host) - 24 GiB; the
#                                    launcher starts WITHOUT the tier if that is < 64 GiB and prints the chosen size. 176 GiB ~ 100M tokens.
#                                    KVRAM=auto = the largest size the budget allows. KVRAM_MEM_FRACTION=<f> KVRAM_MEM_GC=<f>: allocator cap.
#   DSML_RT=1 (default) | 0          DeepSeek API parity plugin (--tool-parser-plugin plugins/dsml_roundtrip.py). Numerics unchanged.
#                                    (1) Round trip: content without the "\n\n" tool-call delimiter, reasoning verbatim = the official
#                                    parser -> a client sending the turn back reuses the previous OUTPUT from the prefix cache.
#                                    (2) Tool-call arguments = the official decoder byte for byte (no schema coercion / unwrapping).
#                                    (3) Requests: reasoning_effort minimal->low, medium->high, xhigh->high (API table; stock rejects
#                                    medium), thinking={"type": enabled|disabled} honoured. (4) tools rendered without "strict".
#   DSML_STRICT=vllm (default) | api strict-tool decoding: vllm = vLLM's grammar-constrained tool calls for tools with strict=true;
#                                    api = unconstrained like the DeepSeek API standard endpoint. Needs DSML_RT=1.
#
# Capacity depends on these settings and is logged at start; docs/ENGRAM.md has measured values.
# The first start JIT-builds the CUDA extensions (~10 min); to build them beforehand, on the CPU:
#   bash scripts/prebuild.sh <the DRYRUN VAR=VAL list>
# NVMe mode: tables under $DS41H_HOME/engram_nvme (built on first start if missing, ~2 min, 210 GB); host RAM ~4.2 GiB pinned + one
# busy poller thread per GPU worker; novel-text prefill is NVMe-read bound.
# Host mode: the table file is created + filled at server start (cooperative per-rank loading) and unlinked at exit unless
# DS41H_ENGRAM_HOST_KEEP=1; `python -m ds41h.engram_host status|drop` manages it.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DS41H_HOME=${DS41H_HOME:-$HOME/.ds41h}
K=7; [ "${THROUGHPUT:-0}" = "1" ] && K=3
K=${SPEC_K:-$K}
SRC=${DS41H_SRC:-$REPO}; EXT=${DS41H_TORCH_EXT_DIR:-$DS41H_HOME/cache/torch_ext}
ENGRAM=${ENGRAM:-nvme}
case "$ENGRAM" in nvme|host|hbm) ;; *) echo "serve_ds41h: ENGRAM must be nvme, host or hbm" >&2; exit 2;; esac
[ -d "$SRC/ds41h" ] || { echo "serve_ds41h: no source tree $SRC" >&2; exit 2; }
CERT=(
  SPEC_K=$K DS41H=1 BATCH_TOKENS=16896 DS41H_ATTN_EARLY=1 DS41H_PREFILL_WQA_SHARD=1 NCCL_MIN_NCHANNELS=64 NCCL_BUFFSIZE=2097152
  DS41H_FAST_TOKENIZE=1 DS41H_PROLOGUE_INDEXER=1 DS41H_ATTN_SIDE=1
  DS41H_DECODER_DEFINES=R1SPLIT=1,ROUTE1=1,OWNA_PRE=1,RESET_LEAN=1,DENSE_TS1=1,PXF=1,WOB3=1,LLP=1,TAILFREE=1,HOTDISP=1
  DS41H_DRAFT_TAIL=2 DS41H_META_BATCH=1 DS41H_META_DRAFT=1 DS41H_SAMPLING_POST_FUSE=1 DS41H_DRAFTER_MAIN_DST=1
  DS41H_DRAFTER_CTXKV_EARLY=1 DS41H_DRAFTER_OPROJ=1 DS41H_DRAFTER_L0PRO=1 DS41H_DRAFTER_EMBED_EARLY=1
  DS41H_DRAFT_TAIL_EARLY_COMBINE=1 DS41H_SAMPLING_EARLY_GLUE=1 DS41H_SAMPLING_NOISE_EARLY=1
  DS41H_DRAFTER_LIN_EARLY=1 DS41H_DRAFTER_SKIPCAT=1 DS41H_DRAFTER_PRECAT=1 DS41H_DRAFTER_MAIN_NORM_DST=1
)
[ "${VHEAD:-1}" = "1" ] && CERT+=(DS41H_VERIFY_HEAD=2 DS41H_VHEAD_VARIANT=1)
# moe_dec cfg 0x55 = the best measured in situ; bit-identical to small16 0x500 = small16k auto
CERT+=(DS41H_MOE_DEC=1 DS41H_MOE_DEC_CFG=0x55)
# the grouped extension lives in its own build dir (grouped_decode.py writes its generated sources there)
[ "${THROUGHPUT:-0}" = "1" ] && CERT+=(DS41H_ATTN_GROUPED=1 DS41H_GDEC_BUILD=${GDEC_BUILD:-$DS41H_HOME/build/flashmla_gdec})
[ "$ENGRAM" = "host" ] && CERT+=(DS41H_ENGRAM_HOST=1)
if [ "$ENGRAM" = "nvme" ]; then
  CERT+=(DS41H_ENGRAM_NVME=1)
  # PREFETCH=1: prompt lookahead into the record cache (numerics-neutral by construction -- served rows are the same exact-keyed
  # bytes -- but not covered by the exactness gate, whose prompts fit one chunk): +24 % on cold c16 16K prefill, decode / repetitive
  # prefill unchanged. Default off.
  [ "${PREFETCH:-0}" = "1" ] && CERT+=(DS41H_ENGRAM_NVME_PREFETCH=1)
fi
if [ "${CONC:-1}" = "1" ]; then
  # CUDA graphs up to 512 tokens per step (64 streams x 8) need expandable segments OFF (vLLM custom all-reduce otherwise
  # fails at capture with 'invalid argument')
  CERT+=(MAX_SEQS=${MAX_SEQS:-64} CAPTURE=${CAPTURE:-512} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False})
else
  CERT+=(MAX_SEQS=${MAX_SEQS:-16} CAPTURE=${CAPTURE:-64})
fi
# GPU_UTIL 0.93 (not 0.96): at 0.96 only ~0.6 GiB stays free and a kernel module loaded at runtime can fail with
# cuModuleLoadData CUDA_ERROR_OUT_OF_MEMORY; the ~4 GiB of KV this costs per GPU is irrelevant for serving.
CERT+=(MAXLEN=${MAXLEN:-36864} GPU_UTIL=${GPU_UTIL:-0.93})
XEXTRA=${EXTRA:-}  # serve.sh's EXTRA: the caller's + the KVRAM / DSML_RT flags below
if [ -n "${KVRAM:-}" ] && [ "${KVRAM}" != "0" ]; then
  KVRAM_SRC=${KVRAM_SRC:-$REPO/plugins}
  [ -f "$KVRAM_SRC/kvram_ext.py" ] || { echo "serve_ds41h: KVRAM needs $KVRAM_SRC/kvram_ext.py" >&2; exit 2; }
  KVWHYF=$(mktemp)
  KVCAP=$(python3 "$KVRAM_SRC/kvram_budget.py" --engram "$ENGRAM" --margin-gib 24 2>"$KVWHYF" || echo 0)
  KVWHY=$(cat "$KVWHYF" 2>/dev/null); rm -f "$KVWHYF"
  KVREQ=$KVRAM; [ "$KVRAM" = auto ] && KVREQ=1000000
  KVSIZE=$(( KVREQ < KVCAP ? KVREQ : KVCAP ))
  if [ "$KVSIZE" -lt 64 ]; then
    echo "serve_ds41h: KVRAM tier DISABLED (usable ${KVCAP} GiB < 64 GiB): ${KVWHY}"
  else
    echo "serve_ds41h: KVRAM tier ${KVSIZE} GiB (requested ${KVRAM}; ${KVWHY})"
    KVJSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$(( KVSIZE << 30 )),\"eviction_policy\":\"lru\"}}"
    CERT+=(KVRAM_DEDUP=${KVRAM_DEDUP:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False "PYTHONPATH=$KVRAM_SRC${PYTHONPATH:+:$PYTHONPATH}")
    XEXTRA+=" --kv-transfer-config $KVJSON --worker-extension-cls kvram_ext.KVRamWorkerExt"
  fi
fi
if [ "${DSML_RT:-1}" = "1" ]; then
  DSML_RT_SRC=${DSML_RT_SRC:-$REPO/plugins}
  [ -f "$DSML_RT_SRC/dsml_roundtrip.py" ] || { echo "serve_ds41h: DSML_RT needs $DSML_RT_SRC/dsml_roundtrip.py" >&2; exit 2; }
  XEXTRA+=" --tool-parser-plugin $DSML_RT_SRC/dsml_roundtrip.py"
  [ -n "${DSML_STRICT:-}" ] && CERT+=(DSML_STRICT=$DSML_STRICT)
fi
[ "$XEXTRA" != "${EXTRA:-}" ] && CERT+=("EXTRA=$XEXTRA")
[ "${DRYRUN:-0}" = "1" ] && { printf '%s\n' "${CERT[@]}" "DS41H_SRC=$SRC" "DS41H_TORCH_EXT_DIR=$EXT"; exit 0; }  # DRYRUN=1: print the env only
echo "serve_ds41h: src $SRC, JIT cache $EXT, power limits $(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
exec env "${CERT[@]}" DS41H_SRC="$SRC" DS41H_TORCH_EXT_DIR="$EXT" bash "$SRC/scripts/serve.sh"
