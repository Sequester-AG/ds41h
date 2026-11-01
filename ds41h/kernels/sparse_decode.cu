// ds41h sparse MLA decode attention for DeepSeek-V4.1-Flash on sm90a (H200), 16 real local heads.
//
// Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/sparse_decode.cu); see LICENSES/BRRR-Apache-2.0.txt. Kept from BRRR: one
// thread-block cluster per query row splitting the key list, keys dequantized to bf16 into a swizzled
// smem tile, ldmatrix + mma.sync m16n8k16 for QK^T and PV (V == K, MQA latent), attention sink, partial
// softmax combine across the cluster. Changed for stock vLLM 0.32.0 on sm90:
//   * reads the stock sm90 "fp8_ds_mla" V4 record (584 B/token) for BOTH the SWA cache and the
//     compressed cache (BRRR read its fork's 528 B SWA / 288 B NVFP4 records);
//   * takes the stock FlashMLA decode metadata (decode_swa_indices/lens, top-k indices) and can fuse
//     compute_global_topk_indices_and_lens (block-table lookup + valid count) into the kernel;
//   * cluster combine through DSMEM (mapa + ld.shared::cluster) instead of global partial buffers;
//   * QK^T split over the head dimension (each warp owns 64 dims) so all 8 warps work for any key count;
//   * runtime cluster size 8 or 16 (non-portable) chosen per launch; any number of keys <= 640.
//
// ---------------------------------------------------------------------------------------------------
// Contract (all citations are vLLM 0.32.0 paths under vllm/):
//  q    [M, >=16, 512] bf16, heads contiguous (stride(1) == 512); only heads 0..15 are read. Stock passes
//       the padded [T, 64, 512] q (models/deepseek_v41/attention.py:1072-1087, nvidia/flashmla.py:202).
//  out  [M, >=16, 512] bf16, same strides; heads 0..15 written, padding heads untouched (o_proj reads
//       attn_out[:, :n_local_heads], models/deepseek_v41/nvidia/ops/o_proj.py:67).
//  KV record (V4, sm90; models/deepseek_v41/attention.py:122-130,512-522; common/ops/cache_utils.py:38-48,
//       78-118; third_party/flashmla/flash_mla_interface.py:111-117): cache tensor [num_blocks, bs, 584] u8,
//       page base = data_ptr + block * stride(0); token data row at base + pos*576 = 448 e4m3 NoPE values
//       followed by 64 bf16 RoPE values; scale row at base + bs*576 + pos*8 = 7 UE8M0 exponents (one per
//       64 NoPE dims) + 1 pad byte. Dequantized key = fp8 * 2^(e-127) (NoPE) | bf16 (RoPE); this is
//       exact in bf16, so the bf16 K tile is bit-identical to stock's dequant.
//  Index semantics (FlashMLA sparse decode, flash_mla_interface.py:90-95; ROCm reference
//       v1/attention/ops/rocm_aiter_mla_sparse.py:1951-1988): key j of a row is attended iff
//       j < len[row] and idx[row][j] >= 0 (and idx < num_blocks*bs, a bounds guard stock lacks).
//       SWA:   idx = decode_swa_indices[row, 0, j] (global slots, page size = swa cache bs), len =
//              decode_swa_lens[row] (v1/attention/backends/mla/sparse_swa.py:1042-1122 causal,
//              1242-1311 DSpark non-causal; invalid rows get len 0).
//       extra: global mode: idx/len given (what FlashMLA gets: compute_global_topk_indices_and_lens).
//              local mode: idx_local = topk_indices_buffer[row, j]; global = block_table[req, l / ebs] * ebs
//              + l % ebs for l >= 0; len = is_valid_token[row] ? count(idx_local >= 0 over the row) : 0
//              (models/deepseek_v41/common/ops/cache_utils.py:832-914, nvidia/flashmla.py:181-195).
//  Softmax: s = q.k * softmax_scale; out = sum_j e^(s_j - m) k_j / (sum_j e^(s_j - m) + e^(sink - m))
//       (sink in natural-log units, not scaled; flash_mla_interface.py:93); a row with no keys and a
//       finite sink outputs 0; sink = +inf outputs 0.
// Numerics follow FlashMLA: bf16 operands, fp32 accumulation, P rounded to bf16 for PV, l summed in fp32.
// ---------------------------------------------------------------------------------------------------

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>

#include "common.cuh"
#include "host.cuh"
#include "sm90.cuh"

namespace ds41h_attn {

constexpr int H = 16;           // real query heads per TP rank (64 heads / TP4)
constexpr int D = 512;          // latent head dim (448 NoPE + 64 RoPE); K == V
constexpr int NOPE = 448;
constexpr int REC_DATA = 576;   // bytes of the token data row (448 e4m3 + 64 bf16)
constexpr int REC_SCALE = 8;    // bytes of the token scale row (7 ue8m0 + pad)
constexpr int REC_BYTES = REC_DATA + REC_SCALE;  // 584, the per-token width the cache tensor publishes
constexpr int QK_GROUPS = 8;      // QK head-dim split: 8 groups of 64 dims (scores reduced over them in the softmax)
constexpr int MAX_THREADS = 512;  // NW = 8 or 16 warps per CTA (template)
constexpr int MAX_TOTAL = 640;  // 128 SWA + 512 top-k (DSpark drafter rows use 192 SWA only)
constexpr int MAX_EXTRA = 512;  // local-mode valid count: two values per thread (NW = 8), one (NW = 16)
constexpr float LOG2E = 1.4426950408889634f;

constexpr int round_up(int a, int b) { return (a + b - 1) / b * b; }

template <int CL, int NW>
struct Cfg {
  static_assert(CL == 8 || CL == 16, "cluster size");
  static_assert(NW == 8 || NW == 16, "warps per CTA");
  static constexpr int NT = NW * 32;                                  // threads
  static constexpr int TPH = NT / H;                                  // softmax / combine threads per head (16, 32)
  static constexpr int DW = D / NW;                                   // PV output dims per warp (64, 32)
  static constexpr int MAXK = round_up((MAX_TOTAL + CL - 1) / CL, 8);  // keys per CTA: 80 (CL 8), 40 (CL 16)
  static constexpr int KROWS = round_up(MAXK, 16);                    // K tile rows (PV k-steps of 16)
  static constexpr int TILES = MAXK / 8;                              // QK n-tiles of 8 keys
  static constexpr int KI = KROWS / NW;                               // K rows per warp in dequant
  static constexpr int SJ = KROWS / 16;                               // PV k-steps of 16 keys
  static constexpr int SK = (KROWS + TPH - 1) / TPH;                  // keys per softmax thread
  static constexpr int RS = MAXK + 4;                                 // partial-score row stride (f32)
  static constexpr int PS = KROWS + 8;                                // P row stride (bf16), PS*2/16 odd
  static constexpr int DPC = D / CL;                                  // output dims combined per CTA
  static constexpr int VPT = DPC * H / NT;                            // outputs per thread in combine
  // Partial exchange (push): CTA r sends owner o the chunk [H][CP] f32 (dims [o*DPC, (o+1)*DPC) of every head,
  // row pitch CP = DPC + 4 against bank conflicts) followed by m[H], l[H]; one bulk DSMEM copy per (r, o).
  static constexpr int CP = DPC + 4;
  static constexpr int CHUNK = H * CP + 2 * H;                        // f32 per chunk
  static constexpr int CHUNK_BYTES = CHUNK * 4;
  static_assert(KI * NW == KROWS && SJ * 16 == KROWS && (PS * 2 / 16) % 2 == 1, "layout");
  static_assert(VPT == 1 || VPT == 2 || VPT == 4, "combine vector width");
  static_assert(CHUNK_BYTES % 16 == 0, "bulk copy size");
};

template <int CL, int NW>
struct Smem {
  using C = Cfg<CL, NW>;
  union {
    struct {
      __nv_bfloat16 q[H * D];        // swizzled q tile (16 KB)
      float red[QK_GROUPS][H][C::RS];  // per-dim-group partial scores (head-dim split)
    } a;
    float send[CL][C::CHUNK];        // outgoing partial chunks, one per owner CTA (written after the softmax)
  } u;
  float recv[CL][C::CHUNK];          // incoming partial chunks, one per source CTA (bulk DSMEM copies)
  __nv_bfloat16 k[C::KROWS * D];     // swizzled dequantized keys (== values)
  __nv_bfloat16 p[H * C::PS];        // softmax numerators (bf16) for PV
  longlong2 off[C::MAXK];            // per key: {byte offset of data row (-1 = masked), of scale row}
  unsigned long long bar;            // mbarrier: all CL incoming chunks landed (tx bytes)
};

struct Params {
  const __nv_bfloat16* q;
  __nv_bfloat16* out;
  long long q_stride, out_stride;  // elements between query rows
  const float* sink;
  float scale_log2;  // softmax_scale * log2(e)
  int per_cta;       // keys per CTA (multiple of 8, <= Cfg::MAXK)
  // main (SWA) cache + indices
  const uint8_t* swa_cache;
  long long swa_page_stride;  // bytes
  int swa_bs;
  long long swa_slots;  // num_blocks * bs
  const int* swa_idx;
  long long swa_idx_stride;
  int swa_width;
  const int* swa_lens;
  // extra (compressed) cache + indices; ex_width == 0 -> no extra cache
  const uint8_t* ex_cache;
  long long ex_page_stride;
  int ex_bs;
  long long ex_slots;
  const int* ex_idx;
  long long ex_idx_stride;
  int ex_width;
  const int* ex_lens;       // global mode
  const int* token_to_req;  // local mode (block_table != nullptr)
  const int* block_table;
  long long bt_stride;
  int bt_width;
  const uint8_t* is_valid;  // bool
  // early mode: index resolution + gather of every key whose SWA slot is not written by this step (not in
  // fresh[0..nfresh)) run BEFORE griddepcontrol.wait, overlapping the predecessor (see sparse_decode()).
  const long long* fresh;
  int nfresh;
  int early;
  // flag mode (DS41H_ATTN_FLAG; the ds41h prologue signals aflag[0] once its q rows and this step's KV inserts are stored):
  // aflag[0] >= atarget replaces griddepcontrol.wait; every CTA past it bumps aflag[1], the last one (nctas) resets both;
  // griddepcontrol.wait moves to the end of the kernel (the attention never completes before its predecessor).
  unsigned* aflag;
  unsigned atarget, nctas;
  int await_mode;  // DS41H_ATTN_FLAG_WAIT: 0 relaxed poll + ld.acquire, 1 relaxed poll + fence.acq_rel.gpu, 2 ld.acquire poll
  // Key-offset table (DS41H_ATTN_OFFTAB, see ds41h/attn.py): offtab[row * MAX_TOTAL + j] = this kernel's resolved {data, scale}
  // offsets of key j of the row (FRESH_BIT included). offmode 1 (writer: the attention of an index-source layer, whose indexer just
  // wrote the top-k buffer) resolves as usual and also stores them; offmode 2 (reader: a later layer of the same index group, same
  // step metadata / top-k buffer / block tables / page geometry, checked by the caller) loads them instead of resolving, i.e.
  // one independent 16-B load per key instead of the index -> block-table load chain + valid-count barriers. Bit-identical.
  longlong2* offtab;
  int offmode;
  unsigned long long* prof;  // DS41H_ATTN_PROF builds only: [rows * CL][PROF_SLOTS] %globaltimer stamps
};
constexpr long long FRESH_BIT = 1LL << 62;  // in Smem::off[k].y: key k's SWA slot is written by this step
constexpr int MAX_FRESH = 64;

// ---------------------------------------------------------------- optional phase timestamps (profiling build)
// -DDS41H_ATTN_PROF: thread 0 of CTA (x, row) writes %globaltimer at phase boundaries into
// prof[(row * gridDim.x + x) * PROF_SLOTS + slot] and %clock64 into slot + 16 (exact intra-CTA durations).
constexpr int PROF_SLOTS = 32;
#ifdef DS41H_ATTN_PROF
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
#define APROF(slot)                                                                                              \
  do {                                                                                                           \
    if (threadIdx.x == 0 && p.prof) {                                                                            \
      p.prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_SLOTS + (slot)] = gtimer();                            \
      p.prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_SLOTS + 16 + (slot)] = (unsigned long long)clock64();   \
    }                                                                                                            \
  } while (0)
#define APROF_CLK(slot)                                                                                                         \
  do {                                                                                                                          \
    if (threadIdx.x == 0 && p.prof) p.prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_SLOTS + (slot)] = (unsigned long long)clock64(); \
  } while (0)
#else
#define APROF(slot) \
  do {              \
  } while (0)
#define APROF_CLK(slot) APROF(slot)
#endif

// ---------------------------------------------------------------- small device helpers
__device__ __forceinline__ int swz(int row, int col) {  // 16-byte chunk XOR swizzle of a [rows, 512] bf16 tile
  return row * D + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void st_shared16(void* p, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(smem_u32(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ unsigned cluster_ctarank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// Release/acquire cluster barrier: makes this CTA's smem writes visible to DSMEM readers in the cluster.
__device__ __forceinline__ void cluster_sync_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// Execution-only cluster barrier (exit guard): no release fence, so it does not wait for this CTA's
// outstanding global stores. Remote loads before it have completed (their values were consumed).
__device__ __forceinline__ void cluster_sync_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ unsigned dsmem_map(unsigned local_addr, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_addr), "r"(rank));
  return r;
}
__device__ __forceinline__ float ld_dsmem(unsigned addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}
template <int N>
__device__ __forceinline__ void ld_dsmem_vec(float (&v)[N], unsigned addr) {
  if constexpr (N == 4) {
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "r"(addr) : "memory");
  } else {
    static_assert(N == 2);
    asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(v[0]), "=f"(v[1]) : "r"(addr) : "memory");
  }
}
// Hang diagnosis (DS41H_ATTN_HANGCHECK, default 1): the flag / exchange waits check %globaltimer every 2048 polls and after
// 4 s print what they wait for and trap (a deadlock becomes a launch error with a message instead of a silent spin).
#ifndef DS41H_ATTN_HANGCHECK
#define DS41H_ATTN_HANGCHECK 1
#endif
struct HangCheck {
  unsigned long long t0 = 0;
  unsigned n = 0;
  __device__ __forceinline__ void tick(int where, unsigned value, unsigned target) {
#if DS41H_ATTN_HANGCHECK
    if ((++n & 2047u) != 0) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t0 == 0) {
      t0 = t;
    } else if (t - t0 > 4000000000ull) {
      printf("ds41h attn HANG: wait %d block (%d,%d) value %u target %u\n", where, int(blockIdx.x), int(blockIdx.y), value, target);
      __trap();
    }
#endif
  }
};

__device__ __forceinline__ unsigned ld_relaxed_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// Two e4m3 -> two bf16 scaled by 2^(e-127), exact. cvt gives the exact f16 (every e4m3 value is a normal
// f16 or zero, mantissa in the top 3 bits); reinterpreting f16 sign|exp|mant>>3 as bf16 gives value *
// 2^-112 exactly (zero stays zero), then one exact bf16x2 multiply by 2^(e-15). Valid for e <= 142.
__device__ __forceinline__ unsigned e4m3x2_bf16x2_fast(unsigned short v, unsigned scale2) {
  const unsigned h = e4m3x2_to_f16x2(v);
  return bf16x2_mul(((h & 0x7fff7fffu) >> 3) | (h & 0x80008000u), scale2);
}
// General path (any e): f32 multiply by 2^(e-127) (e == 0 -> 2^-127 as in OCP E8M0).
__device__ __forceinline__ unsigned e4m3x2_bf16x2_slow(unsigned short v, unsigned e) {
  const unsigned h = e4m3x2_to_f16x2(v);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const float s = e == 0 ? 5.877471754111438e-39f : __uint_as_float(e << 23);
  return pack_bf16(f.x * s, f.y * s);
}
// 16 e4m3 (one 16-byte chunk, byte i = dim i) with one UE8M0 exponent -> 16 bf16 as two 16-byte chunks.
__device__ __forceinline__ void dequant16(uint4 raw, unsigned e, uint4& lo, uint4& hi) {
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  unsigned o[8];
  if (e <= 142u) {
    const unsigned b = (e + 112u) << 7;  // bf16 bits of 2^(e-15)
    const unsigned s2 = b | (b << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[2 * j] = e4m3x2_bf16x2_fast(static_cast<unsigned short>(w[j] & 0xffffu), s2);
      o[2 * j + 1] = e4m3x2_bf16x2_fast(static_cast<unsigned short>(w[j] >> 16), s2);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[2 * j] = e4m3x2_bf16x2_slow(static_cast<unsigned short>(w[j] & 0xffffu), e);
      o[2 * j + 1] = e4m3x2_bf16x2_slow(static_cast<unsigned short>(w[j] >> 16), e);
    }
  }
  lo = make_uint4(o[0], o[1], o[2], o[3]);
  hi = make_uint4(o[4], o[5], o[6], o[7]);
}

// ---------------------------------------------------------------- kernel
// grid (CL, M), cluster (CL, 1, 1), 256 threads. CTA r of row m handles key positions
// [r*per, (r+1)*per) of the row's concatenated [SWA | extra] index list and combines output dims
// [r*512/CL, (r+1)*512/CL) for all 16 heads.
template <int CL, int NW>
__global__ void __launch_bounds__(NW * 32, 1) sparse_decode_kernel(const __grid_constant__ Params p) {
  using C = Cfg<CL, NW>;
  constexpr int THREADS = C::NT;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem<CL, NW>& sm = *reinterpret_cast<Smem<CL, NW>*>(smem_raw);

  const int crank = static_cast<int>(cluster_ctarank());
  const int row = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int per = p.per_cta, first = crank * per, krows = (per + 15) & ~15;
  const int sw = p.swa_width, ew = p.ex_width, total = sw + ew;

  APROF(0);
  APROF_CLK(13);
#ifdef DS41H_ATTN_PROF
  {  // slot 12: 1000000 + %smid (placement relative to the prologue / layer-kernel CTAs in in-situ traces)
    unsigned smid_;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));
    if (threadIdx.x == 0 && p.prof) p.prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_SLOTS + 12] = 1000000ull + smid_;
  }
#endif
  pdl_launch_dependents();
  // The partial exchange needs every peer's mbarrier initialized before the first bulk copy lands: init + expect the
  // CL incoming chunks now, release it to the cluster (relaxed arrive here, the matching wait right before sending).
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&sm.bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" ::"r"(smem_u32(&sm.bar)),
                 "r"(unsigned(CL * C::CHUNK_BYTES))
                 : "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  // ---- key slots for this CTA's positions -> sm.off (all threads: __syncthreads_count inside)
  auto resolve_offsets = [&]() {
    const int j = first + tid;  // position in the row's [SWA | extra] list handled by this thread
    if (p.offmode == 2) {  // reader: offsets resolved by this step's index-source attention (uniform branch)
      if (tid < per) sm.off[tid] = j < total ? __ldg(p.offtab + row * MAX_TOTAL + j) : make_longlong2(-1, 0);
      return;
    }
    int my_idx = -1;
    if (tid < per && j < total)
      my_idx = j < sw ? __ldg(p.swa_idx + row * p.swa_idx_stride + j) : __ldg(p.ex_idx + row * p.ex_idx_stride + (j - sw));
    const int swa_len = __ldg(p.swa_lens + row);
    int ex_len = 0, req = 0, blk = -1;
    if (ew > 0) {
      if (p.block_table != nullptr) {
        // compute_global_topk_indices_and_lens: len = is_valid ? #(local >= 0 over the whole row) : 0
        const int* irow = p.ex_idx + row * p.ex_idx_stride;
        const int v0 = tid < ew ? __ldg(irow + tid) : -1;
        const int v1 = THREADS < MAX_EXTRA && tid + THREADS < ew ? __ldg(irow + tid + THREADS) : -1;
        req = __ldg(p.token_to_req + row);
        const bool tok_ok = __ldg(p.is_valid + row) != 0;
        // the block-table lookup does not depend on the count: issue it before the counting barrier(s)
        if (tok_ok && tid < per && j >= sw && j < total && my_idx >= 0) {  // (padded rows: req is not meaningful)
          const int bi = my_idx / p.ex_bs;
          if (bi < p.bt_width) blk = __ldg(p.block_table + req * p.bt_stride + bi);
        }
        int cnt = __syncthreads_count(v0 >= 0);
        if constexpr (THREADS < MAX_EXTRA) cnt += __syncthreads_count(v1 >= 0);
        ex_len = tok_ok ? cnt : 0;
      } else {
        ex_len = __ldg(p.ex_lens + row);
      }
    }
    if (tid < per) {
      long long doff = -1, soff = 0;
      if (j < sw) {
        if (j < swa_len && my_idx >= 0 && my_idx < p.swa_slots) {
          const int pg = my_idx / p.swa_bs, ps = my_idx - pg * p.swa_bs;
          const long long base = static_cast<long long>(pg) * p.swa_page_stride;
          doff = base + static_cast<long long>(ps) * REC_DATA;
          soff = base + static_cast<long long>(p.swa_bs) * REC_DATA + ps * REC_SCALE;
          if (p.early) {  // written by this step's KV insert (the predecessor): gather after the wait
            bool fr = false;
            for (int f = 0; f < p.nfresh; ++f) fr |= __ldg(p.fresh + f) == static_cast<long long>(my_idx);
            if (fr) soff |= FRESH_BIT;
          }
        }
      } else if (j < total) {
        long long slot = -1;
        if (j - sw < ex_len && my_idx >= 0) {
          if (p.block_table != nullptr) {
            const int bi = my_idx / p.ex_bs;
            if (bi < p.bt_width && blk >= 0) slot = static_cast<long long>(blk) * p.ex_bs + (my_idx - bi * p.ex_bs);
          } else {
            slot = my_idx;
          }
        }
        if (slot >= 0 && slot < p.ex_slots) {
          const int pg = static_cast<int>(slot / p.ex_bs), ps = static_cast<int>(slot - static_cast<long long>(pg) * p.ex_bs);
          const long long base = static_cast<long long>(pg) * p.ex_page_stride;
          doff = base + static_cast<long long>(ps) * REC_DATA;
          soff = base + static_cast<long long>(p.ex_bs) * REC_DATA + ps * REC_SCALE;
        }
      }
      sm.off[tid] = make_longlong2(doff, soff);
      if (p.offmode == 1 && j < total) p.offtab[row * MAX_TOTAL + j] = make_longlong2(doff, soff);  // writer
    }
  };

  // ---- gather + dequant keys: warp w owns K rows w, w+NW, ...; lane l holds 16-byte chunk l of the
  // 576-byte data row (l < 28: 16 e4m3 NoPE dims, scale l/4; l >= 28: 8 bf16 RoPE dims) and lanes 0..3
  // also chunks 32..35 (the last 32 RoPE dims). Masked / padding rows become zeros (finite V for P = 0).
  // mode 0: every row; 1: every row except the fresh ones (early mode, before the wait); 2: fresh rows only.
  auto gather = [&](int mode) {
    uint4 raw[C::KI], rope[C::KI];
    unsigned sexp[C::KI];
    bool mine[C::KI];
#pragma unroll
    for (int i = 0; i < C::KI; ++i) {
      const int k = i * NW + warp;
      raw[i] = make_uint4(0u, 0u, 0u, 0u);
      rope[i] = raw[i];
      sexp[i] = 0u;
      const longlong2 off = k < per ? sm.off[k] : make_longlong2(-1, 0);
      const bool fresh = off.x >= 0 && (off.y & FRESH_BIT) != 0;
      mine[i] = mode == 0 || (mode == 1 ? !fresh : fresh);
      if (off.x >= 0 && mine[i]) {  // warp-uniform
        const uint8_t* base = first + k < sw ? p.swa_cache : p.ex_cache;
        const uint8_t* rec = base + off.x;
        raw[i] = ld_stream16(rec + 16 * lane);
        if (lane < 4) rope[i] = ld_stream16(rec + 512 + 16 * lane);
        if (lane < 28) sexp[i] = __ldg(base + (off.y & ~FRESH_BIT) + (lane >> 2));
      }
    }
#pragma unroll
    for (int i = 0; i < C::KI; ++i) {
      const int k = i * NW + warp;
      if (k >= krows) break;  // warp-uniform
      if (!mine[i]) continue;  // warp-uniform
      if (lane < 28) {
        uint4 lo, hi;
        dequant16(raw[i], sexp[i], lo, hi);
        st_shared16(sm.k + swz(k, 16 * lane), lo);
        st_shared16(sm.k + swz(k, 16 * lane + 8), hi);
      } else {
        st_shared16(sm.k + swz(k, NOPE + 8 * (lane - 28)), raw[i]);
      }
      if (lane < 4) st_shared16(sm.k + swz(k, NOPE + 32 + 8 * lane), rope[i]);
    }
  };

  if (p.early) {  // step metadata, index buffers and older KV records are not written by the predecessor
    resolve_offsets();
    __syncthreads();
    APROF(2);  // key offsets resolved (before the wait)
    gather(1);
    APROF(9);  // old keys gathered (before the wait)
  }
  if (p.aflag != nullptr) {
    // q / the fresh KV records are published by the ds41h prologue through aflag (see Params::aflag), so this CTA does not
    // wait for the whole prologue grid (its Sinkhorn tail) to complete and flush
    if (tid == 0) {
      HangCheck hc;
      unsigned v;
      if (p.await_mode == 2) {
        while ((v = ld_acquire_gpu(p.aflag)) < p.atarget) hc.tick(1, v, p.atarget);
      } else {
        while ((v = ld_relaxed_u32(p.aflag)) < p.atarget) hc.tick(1, v, p.atarget);
        if (p.await_mode == 1)
          asm volatile("fence.acq_rel.gpu;" ::: "memory");  // acquire pattern (strong read + fence): no 2nd round trip
        else
          (void)ld_acquire_gpu(p.aflag);
      }
    }
    __syncthreads();
  } else {
    pdl_wait();  // q, the fresh KV records (and, without early, the indices) are produced by upstream kernels
  }
  APROF(1);

  // ---- q (16 heads x 512 bf16) -> swizzled smem, async
  {
    const __nv_bfloat16* qrow = p.q + row * p.q_stride;
#pragma unroll
    for (int i = tid; i < H * D / 8; i += THREADS) {
      const int h = i >> 6, c = (i & 63) << 3;
      cp_async16(sm.u.a.q + swz(h, c), qrow + h * D + c);
    }
    cp_async_commit();
  }
  if (p.early) {
    gather(2);
  } else {
    resolve_offsets();
    __syncthreads();
    APROF(2);  // key offsets resolved
    gather(0);
  }
  cp_async_wait_all();
  __syncthreads();
  APROF(3);  // keys dequantized, q in smem

  // ---- S = Q K^T, head-dim split: warp w multiplies dims [64 dg, 64 dg + 64) (dg = w % 8) for the key tiles
  // t = w / 8 (mod NW / 8); red[dg] is written by exactly one warp per tile.
  {
    constexpr int KSPLIT = NW / QK_GROUPS;
    const int dg = warp % QK_GROUPS, kh = warp / QK_GROUPS;
    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) ldsm_x4(qa[ks], sm.u.a.q + swz(lane & 15, dg * 64 + ks * 16 + (lane >> 4) * 8));
    const int tiles = per >> 3;
    auto qk_mma = [&](int t, float (&s)[4]) {
#pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        uint32_t kb[4];
        ldsm_x4(kb, sm.k + swz(t * 8 + (lane & 7), dg * 64 + kp * 32 + (lane >> 3) * 8));
        hmma_bf16(s, qa[2 * kp][0], qa[2 * kp][1], qa[2 * kp][2], qa[2 * kp][3], kb[0], kb[1]);
        hmma_bf16(s, qa[2 * kp + 1][0], qa[2 * kp + 1][1], qa[2 * kp + 1][2], qa[2 * kp + 1][3], kb[2], kb[3]);
      }
    };
    auto qk_store = [&](int t, const float (&s)[4]) {
      *reinterpret_cast<float2*>(&sm.u.a.red[dg][g][t * 8 + 2 * tq]) = make_float2(s[0], s[1]);
      *reinterpret_cast<float2*>(&sm.u.a.red[dg][g + 8][t * 8 + 2 * tq]) = make_float2(s[2], s[3]);
    };
    auto qk_tile = [&](int t, float (&s)[4]) {
      qk_mma(t, s);
      qk_store(t, s);
    };
    constexpr int MYT = (C::TILES + KSPLIT - 1) / KSPLIT;  // tiles per warp (full rows)
    if (tiles == C::TILES) {
      // full CTA (every row with the maximum key count): straight-line code, so the independent per-tile HMMA chains
      // interleave instead of running one conditional block after the other (a warp past the last tile recomputes
      // it and only skips the store)
      float s[MYT][4];
#pragma unroll
      for (int i = 0; i < MYT; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
      for (int i = 0; i < MYT; ++i) qk_mma(min(i * KSPLIT + kh, C::TILES - 1), s[i]);
#pragma unroll
      for (int i = 0; i < MYT; ++i)
        if (C::TILES % KSPLIT == 0 || i * KSPLIT + kh < C::TILES) qk_store(i * KSPLIT + kh, s[i]);
    } else {
#pragma unroll
      for (int t = 0; t < C::TILES; ++t) {
        if (t % KSPLIT == kh && t < tiles) {  // warp-uniform
          float s[4] = {0.f, 0.f, 0.f, 0.f};
          qk_tile(t, s);
        }
      }
    }
  }
  __syncthreads();
  APROF(4);  // scores

  // ---- local softmax: thread (head = tid / TPH, key lane = tid % TPH) reduces the 8 partials of its keys;
  // the TPH threads of a head form one (half-)warp, so max/sum need only shuffles.
  {
    const int h = tid / C::TPH, kl = tid % C::TPH;
    float v[C::SK];
    float mx = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < C::SK; ++jj) {
      const int k = kl + C::TPH * jj;
      // partials loaded unconditionally (in-bounds smem; masked keys are discarded below), so the loads of all keys
      // issue together instead of one guarded block after the other
      const int kc = min(k, C::MAXK - 1);
      float part[QK_GROUPS];
#pragma unroll
      for (int w = 0; w < QK_GROUPS; ++w) part[w] = sm.u.a.red[w][h][kc];
      float acc = 0.f;
#pragma unroll
      for (int w = 0; w < QK_GROUPS; ++w) acc += part[w];
      const float x = (k < per && sm.off[kc].x >= 0) ? acc * p.scale_log2 : -INFINITY;
      v[jj] = x;
      mx = fmaxf(mx, x);
    }
#pragma unroll
    for (int o = C::TPH / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < C::SK; ++jj) {
      const int k = kl + C::TPH * jj;
      const float e = v[jj] == -INFINITY ? 0.f : exp2f(v[jj] - mx);
      sum += e;
      if (k < krows) sm.p[h * C::PS + k] = __float2bfloat16_rn(e);
    }
#pragma unroll
    for (int o = C::TPH / 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    // this CTA's m / l go to the tail of every outgoing chunk (u.send aliases u.a: written after the barrier below)
    __syncthreads();  // all reads of u.a.red done
    if (kl < CL) {
      sm.u.send[kl][H * C::CP + h] = mx;
      sm.u.send[kl][H * C::CP + H + h] = sum;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // read by the bulk copies (async proxy)
    }
  }
  __syncthreads();
  APROF(5);  // softmax
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // every peer's mbarrier is initialized (early arrive)

  // ---- O = P V (V == K): warp w owns output dims [DW w, DW (w + 1))
  {
    constexpr int NTL = C::DW / 8;  // n-tiles of 8 dims per warp
    float o[NTL][4];
#pragma unroll
    for (int nt = 0; nt < NTL; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    const int ksteps = krows >> 4;
#pragma unroll
    for (int ks = 0; ks < C::SJ; ++ks) {
      if (ks < ksteps) {  // block-uniform
        uint32_t pa[4];
        ldsm_x4(pa, sm.p + (lane & 15) * C::PS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int nt = 0; nt < NTL; nt += 2) {
          uint32_t vb[4];
          ldsm_x4_t(vb, sm.k + swz(ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, warp * C::DW + nt * 8 + (lane >> 4) * 8));
          hmma_bf16(o[nt], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
          hmma_bf16(o[nt + 1], pa[0], pa[1], pa[2], pa[3], vb[2], vb[3]);
        }
      }
    }
    // u.send aliases u.a (q, red), whose last reads precede the softmax barriers above.
#pragma unroll
    for (int nt = 0; nt < NTL; ++nt) {
      const int d = warp * C::DW + nt * 8 + 2 * tq, own = d / C::DPC, dd = d - own * C::DPC;
      *reinterpret_cast<float2*>(&sm.u.send[own][g * C::CP + dd]) = make_float2(o[nt][0], o[nt][1]);
      *reinterpret_cast<float2*>(&sm.u.send[own][(g + 8) * C::CP + dd]) = make_float2(o[nt][2], o[nt][3]);
    }
    if constexpr (C::DW % C::DPC == 0) {
      // this warp alone wrote the chunk(s) of owners [DW w / DPC, DW (w + 1) / DPC): send them now, no CTA barrier
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      constexpr int OPW = C::DW / C::DPC;  // owners per warp (1 or 2)
      if (lane < OPW) {
        const int own = warp * OPW + lane;
        const unsigned dst = dsmem_map(smem_u32(&sm.recv[crank][0]), unsigned(own));
        const unsigned bar = dsmem_map(smem_u32(&sm.bar), unsigned(own));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                     "r"(smem_u32(&sm.u.send[own][0])), "r"(unsigned(C::CHUNK_BYTES)), "r"(bar)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
  }

  // ---- partial exchange: CTA r pushes chunk o to owner o (one bulk shared::cta -> shared::cluster copy each,
  // completing on the owner's mbarrier), then reduces dims [r*DPC, (r+1)*DPC) of all heads from its own smem.
  const int ch = tid / C::TPH, seg = tid % C::TPH;
  const float sink2 = __ldg(p.sink + ch) * LOG2E;
  if constexpr (C::DW % C::DPC != 0) {  // chunks written by several warps: CTA barrier, then one thread per owner
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic smem writes -> bulk-copy (async proxy) reads
    __syncthreads();
  }
  APROF(6);  // partial o stored
  if (C::DW % C::DPC != 0 && tid < CL) {
    const unsigned dst = dsmem_map(smem_u32(&sm.recv[crank][0]), unsigned(tid));
    const unsigned bar = dsmem_map(smem_u32(&sm.bar), unsigned(tid));
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "r"(smem_u32(&sm.u.send[tid][0])), "r"(unsigned(C::CHUNK_BYTES)), "r"(bar)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  {
    HangCheck hc;
    for (;;) {
      unsigned r;
      asm volatile("{\n .reg .pred P;\n mbarrier.try_wait.parity.shared::cta.b64 P, [%1], 0;\n selp.u32 %0, 1, 0, P;\n}"
                   : "=r"(r)
                   : "r"(smem_u32(&sm.bar))
                   : "memory");
      if (r) break;
      hc.tick(2, 0u, 0u);
    }
  }
  APROF(7);  // all partials received
  {
    const int d0 = crank * C::DPC + seg * C::VPT;
    float mc[CL], lc[CL], vc[CL][C::VPT];
#pragma unroll
    for (int c = 0; c < CL; ++c) {
      mc[c] = sm.recv[c][H * C::CP + ch];
      lc[c] = sm.recv[c][H * C::CP + H + ch];
#pragma unroll
      for (int i = 0; i < C::VPT; ++i) vc[c][i] = sm.recv[c][ch * C::CP + seg * C::VPT + i];
    }
    float mt = sink2;
#pragma unroll
    for (int c = 0; c < CL; ++c) mt = fmaxf(mt, mc[c]);
    float acc[C::VPT];
#pragma unroll
    for (int i = 0; i < C::VPT; ++i) acc[i] = 0.f;
    float den = sink2 == -INFINITY ? 0.f : exp2f(sink2 - mt);
#pragma unroll
    for (int c = 0; c < CL; ++c) {
      const float w = mc[c] == -INFINITY ? 0.f : exp2f(mc[c] - mt);
      den += w * lc[c];
#pragma unroll
      for (int i = 0; i < C::VPT; ++i) acc[i] += w * vc[c][i];
    }
    // mt == +inf only for sink == +inf, which zeroes the row (FlashMLA semantics).
    const float inv = (den > 0.f && mt < INFINITY) ? 1.f / den : 0.f;
    __nv_bfloat16* orow = p.out + row * p.out_stride + ch * D + d0;
    if constexpr (C::VPT == 4) {
      *reinterpret_cast<uint2*>(orow) = make_uint2(pack_bf16(acc[0] * inv, acc[1] * inv), pack_bf16(acc[2] * inv, acc[3] * inv));
    } else if constexpr (C::VPT == 2) {
      *reinterpret_cast<unsigned*>(orow) = pack_bf16(acc[0] * inv, acc[1] * inv);
    } else {
      *orow = __float2bfloat16_rn(acc[0] * inv);
    }
  }
  APROF(8);  // output written
  // Our outgoing copies must have finished reading u.send before the CTA's smem goes away; incoming ones all
  // completed (mbarrier), so no cluster-wide barrier is needed before exit.
  if (C::DW % C::DPC == 0 ? lane < C::DW / C::DPC : tid < CL) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  if (p.aflag != nullptr && tid == 0) {
    // every CTA past its wait bumps aflag[1] here (off the critical path); the last one resets both for the next launch
    if (atomicAdd(p.aflag + 1, 1u) == p.nctas - 1) {
      p.aflag[1] = 0u;
      p.aflag[0] = 0u;
    }
    pdl_wait();  // flag mode: complete only after the prologue grid (stream order downstream)
  }
  APROF(15);
  APROF_CLK(14);
}

// ---------------------------------------------------------------- host side
template <int CL, int NW>
constexpr size_t smem_bytes() {
  return sizeof(Smem<CL, NW>);
}
constexpr size_t SPREAD_SMEM = 120 * 1024;  // > half of the 228 KB SM smem: forces one CTA per SM

template <int CL, int NW>
size_t request_bytes(bool spread) {
  return spread ? std::max(smem_bytes<CL, NW>(), SPREAD_SMEM) : smem_bytes<CL, NW>();
}

// Every CUDA runtime failure on the host side is cleared with cudaGetLastError() before we report it: these
// errors are not sticky, and a left-over "last error" would be picked up by the NEXT torch kernel's
// C10_CUDA_KERNEL_LAUNCH_CHECK (e.g. the stock FlashMLA fallback that ds41h/attn.py runs after a failure here)
// and misreported as that kernel's failure.
template <int CL, int NW>
bool ensure_attributes(int device) {
  static std::once_flag flags[64];
  static bool ok[64] = {};
  TORCH_CHECK(device >= 0 && device < 64, "sparse_decode: device index ", device, " out of range");
  std::call_once(flags[device], [device] {
    auto* kernel = sparse_decode_kernel<CL, NW>;
    const int bytes = static_cast<int>(std::max(smem_bytes<CL, NW>(), SPREAD_SMEM));
    bool good = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess;
    if (good && CL > 8) good = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
    if (!good) (void)cudaGetLastError();
    ok[device] = good;
  });
  return ok[device];
}

// Max co-resident clusters of CL CTAs on an idle device; -1 if none can ever be resident (or the attributes
// were refused), in which case a launch with this cluster size would fail.
template <int CL, int NW>
int max_active_clusters_impl(int device, bool spread) {
  static std::mutex mu;
  static std::array<std::array<int, 2>, 64> cache{};  // 0 = unknown
  TORCH_CHECK(device >= 0 && device < 64, "sparse_decode: device index ", device, " out of range");
  std::lock_guard<std::mutex> lock(mu);
  int& slot = cache[device][spread ? 1 : 0];
  if (slot == 0) {
    int n = 0;
    if (ensure_attributes<CL, NW>(device)) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = CL;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(CL, 1024);
      cfg.blockDim = dim3(NW * 32);
      cfg.dynamicSmemBytes = request_bytes<CL, NW>(spread);
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      if (cudaOccupancyMaxActiveClusters(&n, sparse_decode_kernel<CL, NW>, &cfg) != cudaSuccess) {
        (void)cudaGetLastError();
        n = 0;
      }
    }
    slot = n > 0 ? n : -1;
  }
  return slot;
}

template <int CL, int NW>
void launch(const Params& p, int rows, bool spread, bool pdl, int device) {
  TORCH_CHECK((ensure_attributes<CL, NW>(device)), "sparse_decode: cannot opt in to ", (std::max(smem_bytes<CL, NW>(), SPREAD_SMEM)),
              " B smem", CL > 8 ? " / non-portable cluster size" : "");
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = CL;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL, rows);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = request_bytes<CL, NW>(spread);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cfg.attrs = attrs;
  cfg.numAttrs = pdl ? 2 : 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, sparse_decode_kernel<CL, NW>, p);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(false, "sparse_decode launch failed (cluster ", CL, ", rows ", rows, "): ", cudaGetErrorString(err));
  }
}

// Warps per CTA: DS41H_ATTN_WARPS = 8 or 16 (read once; default 16).
static int warps_per_cta() {
  static const int nw = [] {
    const char* e = std::getenv("DS41H_ATTN_WARPS");
    const int v = e != nullptr ? std::atoi(e) : 16;
    return v == 8 ? 8 : 16;
  }();
  return nw;
}

// Resets the prologue -> attention handoff counter (aflag[0], aflag[1]) when the attention of a flag-mode prologue ran on the
// stock path: a regular launch, so it runs after the prologue (and everything else queued before it) completed.
__global__ void aflag_reset_kernel(unsigned* aflag) {
  aflag[0] = 0u;
  aflag[1] = 0u;
}
void aflag_reset(torch::Tensor aflag) {
  TORCH_CHECK(aflag.is_cuda() && aflag.scalar_type() == at::kInt && aflag.is_contiguous() && aflag.numel() >= 2, "aflag: int32 [2]");
  const c10::cuda::OptionalCUDAGuard guard(aflag.device());
  aflag_reset_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned*>(aflag.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int flag_wait_mode() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_ATTN_FLAG_WAIT");
    const int m = e != nullptr && e[0] != '\0' ? std::atoi(e) : 1;
    return m < 0 || m > 2 ? 1 : m;
  }();
  return v;
}

int64_t max_active_clusters(int64_t cluster, bool spread) {
  int device = 0;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess);
  const bool w16 = warps_per_cta() == 16;
  if (cluster == 16) return w16 ? max_active_clusters_impl<16, 16>(device, spread) : max_active_clusters_impl<16, 8>(device, spread);
  TORCH_CHECK(cluster == 8, "cluster must be 8 or 16");
  return w16 ? max_active_clusters_impl<8, 16>(device, spread) : max_active_clusters_impl<8, 8>(device, spread);
}

// 2-D [rows, width] int32 index view of a [rows, width] or [rows, 1, width] tensor.
static void index_view(const torch::Tensor& t, const char* name, int64_t rows, long long& stride, int& width) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kInt, name, " must be a CUDA int32 tensor");
  TORCH_CHECK(t.dim() == 2 || (t.dim() == 3 && t.size(1) == 1), name, " must be [rows, width] or [rows, 1, width]");
  TORCH_CHECK(t.size(0) >= rows && t.stride(-1) == 1, name, ": too few rows or non-unit last stride");
  stride = t.stride(0);
  width = static_cast<int>(t.size(-1));
}

static void cache_view(const torch::Tensor& t, const char* name, long long& page_stride, int& bs, long long& slots) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte && t.dim() == 3, name, " must be a CUDA uint8 [num_blocks, bs, 584] tensor");
  TORCH_CHECK(t.size(2) == REC_BYTES, name, ": per-token width ", t.size(2), " is not the V4 fp8_ds_mla record (584 B)");
  TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == REC_BYTES, name, ": page must be contiguous");
  TORCH_CHECK(t.stride(0) >= t.size(1) * REC_BYTES && t.stride(0) % 16 == 0, name, ": bad page stride ", t.stride(0));
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": base must be 16-byte aligned");
  page_stride = t.stride(0);
  bs = static_cast<int>(t.size(1));
  slots = t.size(0) * t.size(1);
}

static void qo_view(const torch::Tensor& t, const char* name, int64_t rows) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 3, name, " must be a CUDA bf16 [rows, heads, 512] tensor");
  TORCH_CHECK(t.size(0) >= rows && t.size(1) >= H && t.size(2) == D, name, " must have >= 16 heads of 512");
  TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == D && t.stride(0) % 8 == 0, name, ": heads must be contiguous 512-element rows");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

static unsigned long long* g_prof = nullptr;  // profiling build: stamp buffer of the next launches
void set_prof(c10::optional<torch::Tensor> buf) {
  if (!buf.has_value()) {
    g_prof = nullptr;
    return;
  }
  TORCH_CHECK(buf->is_cuda() && buf->is_contiguous() && buf->scalar_type() == at::kLong, "sparse_decode: prof buffer must be int64");
  g_prof = reinterpret_cast<unsigned long long*>(buf->data_ptr<int64_t>());
}

// Launches the kernel on the current stream. Returns the cluster size used.
int64_t sparse_decode(torch::Tensor q, torch::Tensor out, torch::Tensor swa_cache, torch::Tensor swa_indices, torch::Tensor swa_lens,
                      torch::Tensor sink, double scale, c10::optional<torch::Tensor> extra_cache, c10::optional<torch::Tensor> extra_indices,
                      c10::optional<torch::Tensor> extra_lens, c10::optional<torch::Tensor> token_to_req,
                      c10::optional<torch::Tensor> block_table, c10::optional<torch::Tensor> is_valid, int64_t cluster, bool spread, bool pdl,
                      c10::optional<torch::Tensor> fresh, c10::optional<torch::Tensor> aflag, int64_t atarget,
                      c10::optional<torch::Tensor> offtab, int64_t offmode) {
  const int64_t rows = q.size(0);
  qo_view(q, "q", rows);
  qo_view(out, "out", rows);
  if (rows == 0) return 0;
  TORCH_CHECK(rows <= 65535, "sparse_decode: too many rows");
  const auto device = q.device();
  const c10::cuda::OptionalCUDAGuard guard(device);

  Params p{};
  p.q = data_as<const __nv_bfloat16>(q);
  p.out = data_as<__nv_bfloat16>(out);
  p.q_stride = q.stride(0);
  p.out_stride = out.stride(0);
  TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kFloat && sink.is_contiguous() && sink.numel() >= H, "sink must be f32 [>=16]");
  p.sink = sink.data_ptr<float>();
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.prof = g_prof;

  cache_view(swa_cache, "swa_cache", p.swa_page_stride, p.swa_bs, p.swa_slots);
  p.swa_cache = swa_cache.data_ptr<uint8_t>();
  index_view(swa_indices, "swa_indices", rows, p.swa_idx_stride, p.swa_width);
  p.swa_idx = swa_indices.data_ptr<int>();
  TORCH_CHECK(swa_lens.is_cuda() && swa_lens.scalar_type() == at::kInt && swa_lens.is_contiguous() && swa_lens.numel() >= rows,
              "swa_lens must be contiguous int32 [>=rows]");
  p.swa_lens = swa_lens.data_ptr<int>();

  if (extra_cache.has_value()) {
    TORCH_CHECK(extra_indices.has_value(), "extra_cache needs extra_indices");
    cache_view(*extra_cache, "extra_cache", p.ex_page_stride, p.ex_bs, p.ex_slots);
    p.ex_cache = extra_cache->data_ptr<uint8_t>();
    index_view(*extra_indices, "extra_indices", rows, p.ex_idx_stride, p.ex_width);
    p.ex_idx = extra_indices->data_ptr<int>();
    if (block_table.has_value()) {
      TORCH_CHECK(!extra_lens.has_value(), "pass extra_lens (global indices) or block_table (local indices), not both");
      TORCH_CHECK(token_to_req.has_value() && is_valid.has_value(), "local mode needs token_to_req and is_valid");
      TORCH_CHECK(p.ex_width <= MAX_EXTRA, "local mode supports at most ", MAX_EXTRA, " top-k entries");
      const auto& bt = *block_table;
      TORCH_CHECK(bt.is_cuda() && bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1, "block_table must be int32 [reqs, blocks]");
      p.block_table = bt.data_ptr<int>();
      p.bt_stride = bt.stride(0);
      p.bt_width = static_cast<int>(bt.size(1));
      TORCH_CHECK(token_to_req->is_cuda() && token_to_req->scalar_type() == at::kInt && token_to_req->is_contiguous() && token_to_req->numel() >= rows,
                  "token_to_req must be contiguous int32 [>=rows]");
      p.token_to_req = token_to_req->data_ptr<int>();
      TORCH_CHECK(is_valid->is_cuda() && is_valid->scalar_type() == at::kBool && is_valid->is_contiguous() && is_valid->numel() >= rows,
                  "is_valid must be contiguous bool [>=rows]");
      p.is_valid = reinterpret_cast<const uint8_t*>(is_valid->data_ptr<bool>());
    } else {
      TORCH_CHECK(extra_lens.has_value(), "global mode needs extra_lens");
      TORCH_CHECK(extra_lens->is_cuda() && extra_lens->scalar_type() == at::kInt && extra_lens->is_contiguous() && extra_lens->numel() >= rows,
                  "extra_lens must be contiguous int32 [>=rows]");
      p.ex_lens = extra_lens->data_ptr<int>();
    }
  }
  if (fresh.has_value()) {
    // early mode: the caller guarantees that, apart from the SWA records of the slots in `fresh` (this step's
    // KV insert), nothing the kernel reads before griddepcontrol.wait (indices, lens, block table, token
    // metadata, KV records) is written by a kernel that may still be running (a PDL-overlapped predecessor).
    TORCH_CHECK(fresh->is_cuda() && fresh->device() == device && fresh->scalar_type() == at::kLong && fresh->is_contiguous() &&
                    fresh->numel() <= MAX_FRESH,
                "fresh must be a contiguous int64 CUDA tensor of <= ", MAX_FRESH, " slots");
    p.fresh = data_as<const long long>(*fresh);
    p.nfresh = static_cast<int>(fresh->numel());
    p.early = 1;
  }
  if (aflag.has_value()) {
    TORCH_CHECK(aflag->is_cuda() && aflag->device() == device && aflag->scalar_type() == at::kInt && aflag->is_contiguous() &&
                    aflag->numel() >= 2 && (reinterpret_cast<uintptr_t>(aflag->data_ptr()) & 7) == 0 && atarget > 0,
                "aflag must be the int32 [2] handoff counter of the ds41h prologue scratch, atarget > 0");
    p.aflag = reinterpret_cast<unsigned*>(aflag->data_ptr<int>());
    p.atarget = static_cast<unsigned>(atarget);
    p.await_mode = flag_wait_mode();
  }
  const int total = p.swa_width + p.ex_width;
  TORCH_CHECK(total > 0 && total <= MAX_TOTAL, "sparse_decode supports 1..", MAX_TOTAL, " keys per row, got ", total);
  if (offmode != 0) {
    TORCH_CHECK(offmode == 1 || offmode == 2, "offmode must be 0, 1 (write) or 2 (read)");
    TORCH_CHECK(offtab.has_value() && offtab->is_cuda() && offtab->device() == device && offtab->scalar_type() == at::kLong &&
                    offtab->is_contiguous() && offtab->numel() >= rows * MAX_TOTAL * 2 && (reinterpret_cast<uintptr_t>(offtab->data_ptr()) & 15) == 0,
                "offtab must be a contiguous 16-byte aligned int64 CUDA tensor of >= rows * ", MAX_TOTAL, " * 2 entries");
    p.offtab = reinterpret_cast<longlong2*>(offtab->data_ptr<int64_t>());
    p.offmode = static_cast<int>(offmode);
  }
  for (const torch::Tensor* t : {&out, &swa_cache, &swa_indices, &swa_lens, &sink})
    TORCH_CHECK(t->device() == device, "all tensors must be on ", device);

  const int dev = device.index();
  int cl = static_cast<int>(cluster);
  // auto: 16-CTA clusters when the row has enough keys to split (>= 32 per CTA) and all rows fit in one
  // wave of 16-CTA clusters; otherwise 8 (SWA-only rows, large batches).
  const bool w16 = warps_per_cta() == 16;
  const int max16 = w16 ? max_active_clusters_impl<16, 16>(dev, spread) : max_active_clusters_impl<16, 8>(dev, spread);
  if (cl == 0) cl = (total > 256 && rows <= max16) ? 16 : 8;
  TORCH_CHECK(cl == 8 || cl == 16, "cluster must be 0 (auto), 8 or 16");
  // A 16-CTA cluster needs 16 SMs of one GPC (non-portable size); if the device can never co-schedule one
  // (occupancy 0, attribute refused: MIG / MPS partitions, GPCs with < 16 usable SMs), an explicit
  // DS41H_ATTN_CLUSTER=16 would fail every launch, so degrade to 8, which is always schedulable.
  if (cl == 16 && max16 <= 0) cl = 8;
  p.per_cta = round_up((total + cl - 1) / cl, 8);
  p.nctas = static_cast<unsigned>(rows * cl);
  if (cl == 16) {
    TORCH_CHECK((p.per_cta <= Cfg<16, 8>::MAXK));
    if (w16)
      launch<16, 16>(p, static_cast<int>(rows), spread, pdl, dev);
    else
      launch<16, 8>(p, static_cast<int>(rows), spread, pdl, dev);
  } else {
    TORCH_CHECK((p.per_cta <= Cfg<8, 8>::MAXK));
    if (w16)
      launch<8, 16>(p, static_cast<int>(rows), spread, pdl, dev);
    else
      launch<8, 8>(p, static_cast<int>(rows), spread, pdl, dev);
  }
  return cl;
}

}  // namespace ds41h_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("H") = ds41h_attn::H;
  m.attr("D") = ds41h_attn::D;
  m.attr("REC_BYTES") = ds41h_attn::REC_BYTES;
  m.attr("MAX_TOTAL") = ds41h_attn::MAX_TOTAL;
  m.attr("MAX_EXTRA") = ds41h_attn::MAX_EXTRA;
  m.attr("SMEM_CL8") = static_cast<int64_t>(ds41h_attn::smem_bytes<8, 8>());
  m.attr("SMEM_CL16") = static_cast<int64_t>(ds41h_attn::smem_bytes<16, 8>());
  m.def("warps_per_cta", &ds41h_attn::warps_per_cta);
  m.def("sparse_decode", &ds41h_attn::sparse_decode, "sm90a sparse MLA decode (16 heads, V4 fp8_ds_mla records)", py::arg("q"),
        py::arg("out"), py::arg("swa_cache"), py::arg("swa_indices"), py::arg("swa_lens"), py::arg("sink"), py::arg("scale"),
        py::arg("extra_cache") = py::none(), py::arg("extra_indices") = py::none(), py::arg("extra_lens") = py::none(),
        py::arg("token_to_req") = py::none(), py::arg("block_table") = py::none(), py::arg("is_valid") = py::none(),
        py::arg("cluster") = 0, py::arg("spread") = true, py::arg("pdl") = true, py::arg("fresh") = py::none(), py::arg("aflag") = py::none(),
        py::arg("atarget") = 0, py::arg("offtab") = py::none(), py::arg("offmode") = 0);
  m.def("aflag_reset", &ds41h_attn::aflag_reset);
  m.attr("MAX_FRESH") = ds41h_attn::MAX_FRESH;
  m.def("max_active_clusters", &ds41h_attn::max_active_clusters, py::arg("cluster"), py::arg("spread") = true);
  m.def("set_prof", &ds41h_attn::set_prof, py::arg("buf") = py::none());
  m.attr("PROF_SLOTS") = ds41h_attn::PROF_SLOTS;
}
