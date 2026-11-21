// pieces_engram.h -- FPR's embed + Engram pieces of the C++ mixed-step driver: Engine.p_embed
// (ids32 copy, EngramPhys.prepare, the token-embedding gather into res0) and Engine.p_attn_bd at the Engram layers (1, 14:
// inject on the main stream, then the boundary in identity mode). Through engram's C ABI (engram_cabi.h, gate a 15 / 15) and
// ATen (copies, index_select, where, the cuBLAS wkv mm). Python side: mixed_cxx_engram.py.
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "pieces.h"
#include "../../kernels/engram/engram_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

// EngramPhys.prepare(T = R, stream = estream) on the fixed-size path with the cuBLAS wkv (rows > w8_max_rows or no gemm8):
//   est waits main (pre-created event); on est: [_resolve_starts: b_start = where(b_start < 0, src[:S].to(int32), b_start)];
//   hash
typedef struct {
  int32_t version, R, resolve, src_dtype;   // resolve: EngramPhys._start_src is bound
  cudaStream_t main, est;
  int64_t ctx;                              // EngramPhys.hctx: the driver fills hp from it (engram.hash_params) at add
  void* b_start; int64_t S; void* src;      // _resolve_starts
  FprEngramHashParams hp;
  FprEngramHashArgs hash;                   // hash.hp is pointed at hp by the driver at add
  FprEngramGatherArgs gather[2];
  void* wx[2]; int64_t wx_st0[2], wx_st1[2];   // bf16 [T, K]
  void* ww[2]; int64_t ww_st0[2], ww_st1[2];   // bf16 [N, K] (the mm takes its .t())
  void* wy[2]; int64_t wy_st0[2], wy_st1[2];   // bf16 [T, N]
  int32_t K, N;
  cudaEvent_t ev_l[2];                      // EngramPhys.ev[l] (cuda_event handles)
  void* stage_buf;                          // EngramPhys._stage_buf bf16 [>= ceil(N / C), K] (C4c); NULL with C = 0
  int32_t stage_chunks, pad_;               // C (0 = one mm on wbf)
  // the W8A16 wkv = EngramPhys.run_wkv's "w8a16" impl (gemm8 and T <= w8_max_rows; engram_stage's
  // wrapper passes it through unchanged): for r0 in range(0, T, 128): gemm8[l](x[r0:r1], y[r0:r1]) with x = wx[l] [T, K],
  // y = wy[l] [T, N], through engram_w8a16_ext's own C ABI "w8a16.gemm" (the build the Python path calls), splits 1,
  // rows_per_group / kx_per_group = gemm8[l].rpg / .kxg, no ws / cnt (the pybind path passes neither with splits 1).
  // w8 = 1: this branch (ww / stage_* unused); 0: the cuBLAS branch above, as v1.
  int32_t w8, pad2_;
  const void* w8_packed[2];                 // gemm8[l].packed uint8 [nchunk, tiles, W8A16_CHUNK_BYTES]
  int64_t w8_nchunk[2], w8_tiles[2], w8_rpg[2], w8_kxg[2];
} FprPieceEngramPrep;
#define FPR_ENGRAM_PREP_VERSION 2           // FprPieceEngramPrep.version (v1 = no w8 fields: a size mismatch, refused at add)

// Engine.p_embed for a prod bucket: [Engram: ids32[:R].copy_(tok) on main, prepare], then emb = index_select(table, 0, tok)
// (into a persistent [R, D] buffer) and res0.copy_(emb.unsqueeze(1).expand(R, HC, D)), main stream. No images (mm_emb).
typedef struct {
  int32_t version, R, has_engram, tok_dtype;
  cudaStream_t main;
  void* tok;                                // [R] (s["tok"])
  void* ids32;                              // int32 [>= R] (hooks.ids32[r]) or NULL without Engram
  void* emb_tab; int64_t emb_rows, emb_st0; // bf16 [V, D] (glob["emb"])
  void* emb_tmp;                            // bf16 [R, D]
  void* res0; int64_t res0_st0, res0_st1;   // bf16 [R, HC, D]
  int32_t D, HC;
  FprPieceEngramPrep prep;
} FprPieceEmbed;

// Engine.p_attn_bd at an Engram layer (prod): [inject_join: _join_mix], main waits ev[l], inject (MODE 1, fused hc_post) from
// the _inputs set into X1, then the boundary (FprPieceBd handle `bd`, built on the identity-mode inputs (zy, X1, zpost, eyec,
// lag of the set)), the set chosen as _inputs does (first layer and not chain_input -> 0, else 1).
typedef struct {
  int32_t version, layer, l_idx, inject_join;
  int32_t first_layer, R;
  cudaStream_t main;
  cudaEvent_t ev_l;
  FprEngramInjectArgs inj[2];
  int64_t bd;
} FprPieceEngramBd;

#ifdef __cplusplus
}
#endif
