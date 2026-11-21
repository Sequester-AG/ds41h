// engram_cabi.h -- C ABI of kernels/engram/engram.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// Engram on owned rows (layers 1 and 14; EngramPhys in engram.py): the step's token-only side work (prepare: hash, per-layer
// gather from the pinned host tables; the wkv GEMM between gather and inject is w8a16's / cuBLAS, not this extension), the
// injection on the hc stream (inject: MODE 0 x_in / MODE 1 fused hc_post; inject_rs: MODE 2, RS round folded), hc_post
// alone, and the prefill dedup helpers (gather_ids, expand).
// No init entry: no per-device state (no dynamic SMEM, no attributes, no device buffers). hash_setup (host-only hash
// constants, one registry entry per EngramPhys), probe (debug) and host_register / host_unregister (init-time host memory
// registration) stay pybind only. The hash constants cross the boundary as a caller-owned FprEngramHashParams (host memory,
// read during the call only): "engram.hash_params" copies registry entry ctx (EngramPhys.hctx) into one, so the C path
// launches k_hash with exactly the pybind path's constants.
// Table: fpr_engram_table(); "engram.hash_params" (host copy, no launch), "engram.hash", "engram.gather", "engram.gather_ids",
// "engram.expand", "engram.inject", "engram.inject_rs", "engram.hc_post", "engram.ring_copy". All launches on c.stream; T == 0 (n == 0) -> FPR_OK
// without a launch, as the pybind path.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_ENGRAM_NL 2        // Engram layers (1, 14): eg::NL
#define FPR_ENGRAM_NG 4        // max n-gram size: eg::NG
#define FPR_ENGRAM_NC 24       // hash columns per layer: eg::NC
#define FPR_ENGRAM_DIM 256     // engram head dim (bf16 elements per looked-up record): eg::DIM
#define FPR_ENGRAM_D 5120      // hidden: eg::D
#define FPR_ENGRAM_KVW 25600   // kv row width (4 keys + value): eg::KVW

typedef struct {               // = eg::HashParams (engram.cu static_asserts the layout); stock compute_hash_multipliers + layout
  int64_t mult[FPR_ENGRAM_NL][FPR_ENGRAM_NG];
  int64_t prime[FPR_ENGRAM_NL][FPR_ENGRAM_NC];
  int64_t off[FPR_ENGRAM_NL][FPR_ENGRAM_NC];
  int32_t pad_id;              // token_map[engram_pad_token_id]
  int32_t vocab;               // token_map size
} FprEngramHashParams;

typedef struct {               // no pybind twin: the constants pybind hash(ctx, ...) launches with (hash_setup's entry ctx)
  FprCall c;                   // unused (host-only copy, no CUDA call)
  int64_t ctx;                 // hash_setup's return value (EngramPhys.hctx); out of range -> FPR_E_ARG
  FprEngramHashParams* out;    // host memory
} FprEngramHashParamsArgs;

typedef struct {               // = pybind hash(ctx, ids, counts, s_row0, s_nrows, s_start, s_slot, s_look, tmap, ring, T, out, keep, err)
  FprCall c;
  const FprEngramHashParams* hp;   // host memory, read during the call (pybind: hash_setup's entry ctx)
  const int32_t* ids;          // [>= T] raw input ids
  const int32_t* counts;       // [>= 2] device: (sequences S, real rows n)
  const int32_t* s_row0;       // [S]
  const int32_t* s_nrows;      // [S]
  const int32_t* s_start;      // [S]
  const int32_t* s_slot;       // [S] state slot (ring row), -1 = none
  const int32_t* s_look;       // [S, 3] lookback raw ids (-1 unknown, -2 device ring)
  const int32_t* tmap;         // [vocab] token map
  int32_t* ring;               // [slots, 16, 2] (position, token) ring, read + written
  int32_t T;                   // rows (grid ceil(8 T / 256) x 256)
  int32_t* out;                // [>= T, 2, 24] hash ids out
  uint8_t* keep;               // [>= T] out
  int32_t* err;                // [3] error counters (atomicAdd)
} FprEngramHashArgs;

typedef struct {               // = pybind gather(hash_ids, T, tabs, lidx, outs, lo, hi, ctas, threads, unroll)
  FprCall c;
  const int32_t* hash;         // [>= T, 2, 24]
  int32_t T;
  int32_t nl;                  // selected layers 1..2 (= len(tabs)); EngramPhys.prepare: 1 per call (layers=(l,))
  const void* tabs[FPR_ENGRAM_NL];     // device-visible address of record 0 per selected layer (HostTable.rec0), % 16 == 0
  int32_t lidx[FPR_ENGRAM_NL];         // hash-layer index per selected layer (0 = layer 1, 1 = layer 14)
  void* outs[FPR_ENGRAM_NL];           // bf16 [>= T, 24, 256] per selected layer (EngramPhys.rows[l])
  int64_t lo[FPR_ENGRAM_NL][FPR_ENGRAM_NC];   // per selected layer: ownership ranges [lo, hi) (pybind flat lo[l * 24 + c])
  int64_t hi[FPR_ENGRAM_NL][FPR_ENGRAM_NC];
  int32_t ctas;                // 0 = one record per 16-lane group (EngramPhys.gather_cfg: 8)
  int32_t threads;             // 32..512, % 32 (gather_cfg: 512)
  int32_t unroll;              // 1 / 2 / 4 / 8 (gather_cfg: 4)
} FprEngramGatherArgs;

typedef struct {               // = pybind gather_ids(tab, ids, out, ctas, threads, unroll) (prefill dedup)
  FprCall c;
  const void* tab;             // record 0 of one layer's table
  const int64_t* ids;          // [n] global row ids (< 0 -> zeros)
  int32_t n;                   // ids.numel()
  void* out;                   // bf16 [>= n, 256]
  int32_t ctas;                // 0 = one record per group
  int32_t threads;             // >= 16 (pybind: unchecked; the grid divides by threads / 16)
  int32_t unroll;              // 1 / 2 / 4 / 8
} FprEngramGatherIdsArgs;

typedef struct {               // = pybind expand(urows, inv, T, nl, l, out) (prefill dedup)
  FprCall c;
  const void* urows;           // bf16 [U, 256]
  const int32_t* inv;          // [T, nl, 24] -> urows row
  int32_t T;
  int32_t nl;
  int32_t l;
  void* out;                   // bf16 [T, 24, 256]
} FprEngramExpandArgs;

typedef struct {               // = pybind inject(T, x_in, y, rs, post, comb, kv, qw, kw, keep, eps, clampv, out, gate_out, kv_idx)
  FprCall c;
  int32_t T;                   // rows (grid T x 640)
  const void* x_in;            // bf16 [T, 4, 5120]: MODE 0; NULL: MODE 1 (x = hc_post(y, rs, post, comb), fused)
  const void* y;               // MODE 1: bf16 [T, 5120]
  const void* rs;              // MODE 1: bf16 [T, 4, 5120]
  const float* post;           // MODE 1: [T, 4]
  const float* comb;           // MODE 1: [T, 4, 4]
  const void* kv;              // bf16 rows [key_0 | .. | key_3 | value] (>= 25600 elements), row stride kv_stride
  int64_t kv_stride;           // elements (kv.stride(0); EngramPhys.kv[l]: 25600)
  const void* qw;              // bf16 [4, 5120] (EngramPhys.q[l])
  const void* kw;              // bf16 [4, 5120] (EngramPhys.k[l])
  const uint8_t* keep;         // [T] (EngramPhys.keep) or NULL (every row kept)
  double eps;                  // rms_norm_eps
  double clampv;               // EngramPhys.clamp (1e-6)
  void* out;                   // bf16 [T, 4, 5120] out (may alias x_in / rs); caller-provided (engram.py allocates if None)
  float* gate_out;             // [T, 4] or NULL
  const int32_t* kv_idx;       // [T] row -> kv row (prepare_prefill dedup) or NULL (identity)
} FprEngramInjectArgs;

typedef struct {               // = pybind inject_rs(T, recv, flag, epoch, signals, rs, post, comb, kv, qw, kw, keep, eps, clampv,
                               //                    out, gate_out, kv_idx): MODE 2, y = fixed-order sum of the RS receive slots
  FprCall c;
  int32_t T;
  const void* recv;            // bf16 [4, R, 5120] contiguous (slot = source rank)
  int32_t R;                   // recv.size(1) (>= T)
  const uint32_t* flag;        // RS flag (device word)
  const uint32_t* epoch;       // RS round counter (device word)
  uint32_t signals;            // per round
  const void* rs;              // bf16 [T, 4, 5120]
  const float* post;           // [T, 4]
  const float* comb;           // [T, 4, 4]
  const void* kv;
  int64_t kv_stride;
  const void* qw;
  const void* kw;
  const uint8_t* keep;         // or NULL
  double eps;
  double clampv;
  void* out;                   // bf16 [T, 4, 5120]
  float* gate_out;             // or NULL
  const int32_t* kv_idx;       // or NULL
} FprEngramInjectRsArgs;

typedef struct {               // = pybind hc_post(T, y, rs, post, comb, out)
  FprCall c;
  int32_t T;                   // rows (grid T x 640)
  const void* y;               // bf16 [T, 5120]
  const void* rs;              // bf16 [T, 4, 5120]
  const float* post;           // [T, 4]
  const float* comb;           // [T, 4, 4]
  void* out;                   // bf16 [T, 4, 5120]
} FprEngramHcPostArgs;

// "engram.ring_copy" = engine.copy_state_slots' EngramPhys.ring_copy(src_slot
// dst_slot, dst=eg[dr]) over the pairs of ONE (source rank, destination rank) group: the destination ring's slot gets the
// source ring's slot (16 (position, token) int32 pairs = 128 B), read over P2P / UVA. Launch on c.stream = the
// DESTINATION rank's step stream (copy_state_slots runs under torch.cuda.device(devs[dr])); c.device is the caller's
// (nothing here changes the current device). Pairs in Python's op order: a chained batch (a source slot that an earlier
// pair of the same ring wrote, or a repeated destination slot) runs one launch per pair in order. Ordering against the
// rings' other writers (the Engram hash on each rank's estream, slot reuse in the same step) is the CALLER's: nd's
// page_copies protocol (device syncs before, a destination event every step stream waits on); src_ready (optional) is a
// cudaEvent_t the destination stream waits on first, for other callers. Any slot outside [0, nring) or a ring base not
// 16-B aligned -> FPR_E_ARG with nothing issued. n == 0 -> FPR_OK without a launch. Nothing allocates (rule 2).
// PRECONDITION (P2P; torch's copy_ has none): src_ring must be mapped on the destination device, i.e.
// cudaDeviceEnablePeerAccess dst -> src and, for torch allocations under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
// (served), torch's peer registration of the pair (engine._torch_peer_maps, ENG_PEER_MAP=1); else the read faults (Xid 31
// / illegal address). Callers refuse the native path (Python copy_ stays) when ENG_PEER_MAP=0 or cudaDeviceCanAccessPeer
// is false for the pair. Cross-group order is the caller's: this entry orders the pairs of ONE group only (calls on
// different destination streams are unordered against each other; Python's cross-device copy_ is fully sequential).
#define FPR_ENGRAM_RING_COPY_MAX 64
typedef struct {
  FprCall c;                   // the destination rank's step stream
  const int32_t* src_ring;     // source rank's EngramPhys.ring [src_nring, 16, 2] (device; peer reads)
  int32_t* dst_ring;           // destination rank's EngramPhys.ring [dst_nring, 16, 2]
  int32_t src_nring, dst_nring;   // ring rows (EngramPhys.ring.shape[0], handover FprEhEngramRank.nring)
  void* src_ready;             // cudaEvent_t or NULL
  int32_t n;                   // pairs, 0..FPR_ENGRAM_RING_COPY_MAX (split larger groups)
  int32_t pad_;
  int32_t src_slot[FPR_ENGRAM_RING_COPY_MAX];
  int32_t dst_slot[FPR_ENGRAM_RING_COPY_MAX];
} FprEngramRingCopyArgs;

const FprExtTable* fpr_engram_table(void);

#ifdef __cplusplus
}
#endif
