// w8a16.cu: fused W8A16 dense GEMM for the sequence-parallel layout (sm_90a).
//
//   y[t, n] = sum_k x[t, k] * W[n, k],   W = e4m3 [N, K] with one ue8m0 scale per 32x32 block (checkpoint), x / y bf16,
//   t < ntok own rows (<= 128 per CTA; more rows run in ONE launch with grid z = 128-row token slices, splits == 1
//   only, bit-identical to 128-row calls). Used for q_b (N 32768, K 1280), the 8-group wo_a (N 8192 = 8 x 1024, K 4096
//   per group: group g reads x[:, 4096 g : 4096 g + 4096]) and wo_b (N 5120, K 8192).
//
// Numerics (lossless class): every weight is dequantized EXACTLY to bf16 (e4m3 -> f16 cvt, exponent shift, bf16 multiply
// by 2^(s-15): e4m3 * 2^(s-127) is a bf16 value), bf16 x bf16 products are exact, accumulation is fp32 (wgmma), split-K
// partials are summed in fp32 in a fixed order (split 0, 1, ...), one bf16 rounding at the end. With RoPE (q_b):
// q = bf16(acc), then stock's interleaved RoPE in fp32 without FMA contraction, bf16 (same as dense.cu / the reference).
//
// Work split: one CTA = 256 weight rows x one K range; grid = (split-K S, 256-row tiles). Weights are repacked once
// (repack in w8a16.py) so that every (tile, 128-k chunk) is ONE contiguous 32,896-byte block made of two 16,448-byte
// halves, one per consumer warpgroup (its 128 rows): 16 KB of fragments + 16 B of scales + pad. Inside a half, k32
// block kb, 64-row sub-tile m and warp w own 512 bytes; lane l = 4 g + t holds its RS-wgmma A fragments for two k16
// steps (rows g / g + 8, k 2t, 2t+1, 2t+8, 2t+9 and +16) in 16 contiguous bytes -> one 16-byte LDS per lane per k32 per
// sub-tile.
//
// v3 (256 threads, no producer warps): 2 consumer warpgroups (warpgroup c owns sub-tiles 2c, 2c+1 = 128 rows;
// accumulators 2 x 64 x NT fp32). With a third (producer) warpgroup ptxas budgets 168 registers per thread in its wgmma
// pipeline analysis even under setmaxnreg, and serialized every wgmma (C7512 / C7513: ~111 cycles per m64n120k16 = 54 %
// of tensor peak); with 2 warpgroups the budget is 255 and nothing is serialized (a device printf in the watchdog also
// serialized them, C7510: it now stores a code and traps). So the consumers issue their own loads: each warpgroup
// streams its own W half blocks into its own ring (2-4 slots as SMEM allows; 2 suffice) with bulk copies (L2
// evict-first) and refills a slot as soon as its 4 warps have LDS'd it; the x ring (W8_NXS slots of two TMA tiles of
// 64 k x NT rows, 128B swizzle, rows >= ntok zero-filled) is shared and refilled by whichever warpgroup finishes a chunk
// second (shared-memory counter, no waiting). Warp 1 loads the tokens' cos | sin rows (RoPE / inverse RoPE); for q_b
// warp 7 does the window KV record insert of the CTA's token under the first loads. Mainloop per k32: wait for the
// wgmma group of k32 - 2, dequantize 2 x 16 bytes into the A registers (double buffered; W8_DQ 1 = integer bit
// placement, no cvt), wgmma.fence, 4 x RS wgmma m64 n NT k16 (B = x from SMEM, K-major SW128 descriptor), commit; at the
// end of each chunk the warpgroup drains its groups (none may cross the loop back-edge). NT = ntok rounded up to 8
// (96..128). The next chunk's barrier waits and first k32 dequant run under the chunk's last wgmma group. (A ping-pong
// between the warpgroups, named-barrier turns, measured 2-3 us slower per GEMM and was removed.)
// v4 (consumer side): the raw fp8 bytes are LDS'd one k32 step ahead of their dequant; the next chunk's barrier waits
// and first k32 dequant run under the chunk's last wgmma group; the plain GEMMs get the deepest x ring that fits next to
// 2 W slots per warpgroup (5 at NT 120 / 128); the fused KV insert loads at kernel start and stores after the mainloop;
// PDL (W8_PDL): barrier init + the weight stream start before griddepcontrol.wait, launch_dependents after the mainloop.
// Measured and kept off: W from global straight into registers (W8_WLDG; register-capacity bound), LDS 3 steps ahead
// (W8_LA3), ping-pong (W8_PP), a per-tile rotated chunk order, cos | sin loaded after the mainloop.
// wo_a: the inverse RoPE of x is applied in SMEM to the chunk holding dims 448..511 of a head before its MMAs (inv_pos;
// attn_half keeps the dense.cu inverse RoPE instead, which is faster with PDL).
// Epilogue: fp32 accumulators -> SMEM [NT][256 + 4] (conflict-free). S > 1: the staging image is bulk-stored (TMA,
// W8_BST) into a workspace shared by all GEMMs of the device (L2-resident), a monotonic per-tile arrival counter (the
// grid is one resident wave), then each CTA of the tile reduces a token slice over the S partials in the fixed order
// 0..S-1 (all S loads of two items in flight); optional RoPE (cos / sin from SMEM); coalesced 16-byte bf16 stores.
#include <atomic>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

#include "w8a16_wgmma.cuh"
#include "w8a16_wgmma_wide.cuh"
#include "w8a16_cabi.h"

#ifndef W8_PROF
#define W8_PROF 0  // clock64 profile (separate build): prof[cta][wg][16]: mainloop parts, prologue, epilogue, globaltimer
#endif
#ifndef W8_DQ
#define W8_DQ 1  // dequant: 0 = e4m3 -> f16 cvt + shift / mask + bf16 multiply; 1 = integer bit placement (no cvt):
                 // LOP3 + PRMT on the integer pipe, IMAD + bf16 multiply on the FMA pipe (default; W8A16_DQ=0 builds the cvt path)
#endif
#ifndef W8_NXS
#define W8_NXS 0  // x ring stages of the plain GEMMs; 0 = the deepest ring (<= 6) that fits next to 2 W slots per
                  // warpgroup (5 at NT 120 / 128): deeper x was better at every step (2 / 3 / 4 / 5 slots, M = 116:
                  // chain 109.6 / 106.2 / 102.6 us with 3-4 W slots; 4 -> 5 with 2 W slots: 95.2 -> 92.5 us)
#endif
#ifndef W8_BST
#define W8_BST 1  // split-K partials: 1 = TMA bulk stores straight from the fp32 staging (workspace at the staging pitch),
                  // 0 = 16-byte register stores (store-path bound: ~2.3 TB/s aggregate, ~7 us for wo_a)
#endif
#ifndef W8_PP
#define W8_PP 0  // 1: the two warpgroups take turns issuing their wgmma groups (named barriers 5 / 6), so their dequants
                 // (integer pipe) do not coincide
#endif
#ifndef W8_RIF
#define W8_RIF 2  // split-K reduce: token items per thread whose S partial loads are all in flight at once
#endif
#ifndef W8_SMAX
#define W8_SMAX 8  // most K-splits (16: a second batch of 8 partials in the reduce; separate build, e.g. q_a | wkv)
#endif
#ifndef W8_WLDG
#define W8_WLDG 0  // W path: 1 = no SMEM: the W stream is bulk-prefetched into L2 (W8_NPF chunks ahead) and every lane
                   // loads its own fragment bytes with 16-byte global loads 3 k32 steps ahead (registers); the SMEM
                   // freed from the W rings deepens the x ring. 0 = per-warpgroup W rings in SMEM (bulk copies + LDS).
#endif
#ifndef W8_LA3
#define W8_LA3 0   // SMEM W path: 1 = LDS the raw bytes 3 k32 steps ahead (4-slot raw ring) instead of 1 (no gain:
                   // the LDS cost is MIO throughput next to the HGMMAs, not latency)
#endif
#ifndef W8_SC1
#define W8_SC1 1   // SMEM W path (1-step): the chunk's 16 scale bytes in ONE LDS.128 at its first k32 instead of 2 LDS.U8
                   // per k32 (fewer MIO ops next to the HGMMAs)
#endif
#ifndef W8_NPF
#define W8_NPF 4   // W8_WLDG: L2 prefetch distance in chunks
#endif
#ifndef W8_NXS_CS
#define W8_NXS_CS 4  // x ring stages of the RoPE / inverse-RoPE variants (their cos | sin rows take 30 KB of SMEM;
                     // 2 / 3 / 4 measured, 4 best)
#endif
#ifndef W8_PDL
#define W8_PDL 1  // programmatic dependent launch: barrier init + the weight stream start before griddepcontrol.wait
                  // (they do not depend on the previous kernel); x / kv / the workspace only after it;
                  // launch_dependents after the mainloop lets the next kernel launch during this one's epilogue
#endif
#ifndef W8_WIDE
#define W8_WIDE 1  // > 128 token rows (splits 1, no RoPE / KV insert): the wide kernel; 0 = served z slices
#endif
#ifndef W8_ABL
#define W8_ABL 0  // timing ablations, separate builds only: 1 = no dequant, 2 = no wgmma, 4 = no x loads,
                  // 8 = no loads and no waits at all (compute on whatever SMEM holds)
#endif

namespace w8 {

constexpr int ABL = W8_ABL;
constexpr int TR = 256;             // weight rows per CTA
constexpr int KC = 128;             // k per chunk
constexpr int HW = TR / 2 * KC;     // fragment bytes of one warpgroup's 128 rows per chunk
constexpr int HB = HW + 64;         // one warpgroup's half block: fragments + 16 B of scales (4 k32 x 4 row blocks) + pad
constexpr int CHUNK = 2 * HB;       // packed bytes per (tile, chunk)
constexpr int WSTR = 16512;         // per-warpgroup W ring slot stride (HB rounded up to 128 B)
constexpr int NXS = W8_NXS;         // x ring stages (0 = adaptive, see W8_NXS), shared by both warpgroups
constexpr int NTHR = 256;           // 2 consumer warpgroups, no producer warps
constexpr int YP = TR + 4;          // fp32 staging pitch (floats): bank-conflict-free accumulator stores
constexpr int SMEM_MAX = 232448;

template <int NT, bool CS>  // CS: the kernel does RoPE / inverse RoPE (cos | sin rows in SMEM); else that SMEM goes to W / x
struct Cfg {
  static constexpr int XB = NT * 128;                     // one 64-k atom of x (NT rows x 128 B), 1 KB multiple
  static constexpr int XS = 2 * XB;                       // x bytes per chunk
  static constexpr int CSB = CS ? NT * 64 * 4 : 0;        // cos | sin rows of the tokens (RoPE / inverse RoPE)
  static constexpr int YS = NT * YP * 4;
  static constexpr int NXF = (SMEM_MAX - CSB - 1280) / XS;                // x slots that fit without W rings
  static constexpr int NXA = (SMEM_MAX - CSB - 1280 - 4 * WSTR) / XS;      // x slots that fit next to 2 W slots per WG
  static constexpr int NX = W8_WLDG ? (NXF < 8 ? NXF : 8)
                                    : CS ? W8_NXS_CS : NXS > 0 ? NXS : (NXA < 6 ? NXA : 6);  // x ring stages
  static constexpr int fixed = NX * XS + CSB + 256 + 1024;
  static constexpr int NWS = W8_WLDG ? 1 : (fixed + 8 * WSTR <= SMEM_MAX) ? 4 : (fixed + 6 * WSTR <= SMEM_MAX) ? 3 : 2;
  static constexpr int RING = NX * XS + (W8_WLDG ? 0 : 2 * NWS * WSTR);
  static constexpr int BODY = RING > YS ? RING : YS;
  static constexpr int CSOFF = BODY;
  static constexpr int BAR = CSOFF + CSB;
  static constexpr int SMEM = BAR + 256 + 1024;
  static_assert(SMEM <= SMEM_MAX, "smem");
};

struct Params {
  const uint8_t* w;          // packed [nchunk][tiles][CHUNK] (chunk-major)
  __nv_bfloat16* y;          // [ntok, ldy]; column = global weight row
  int ldy, ntok, nchunk, rows_per_group, kx_per_group;
  const long long* pos;      // RoPE of y (q_b): positions [ntok] (nullptr = none)
  const float* cs;           // stock cos | sin cache [P, 64]
  const long long* inv_pos;  // inverse RoPE of x (wo_a: x = attention output, dims 448..511 of every 512): positions
  // window KV insert (q_b): kv [ntok, 512] bf16, slot [ntok] int64, cache pages
  const __nv_bfloat16* kv;
  const long long* slot;
  uint8_t* cache;
  long long page_stride;
  int shift;
  // split-K (gridDim.x > 1): fp32 partials [tiles][S][NT][256] + one arrival counter per tile (monotonic)
  float* ws;
  unsigned* cnt;
  int abl;
  unsigned long long* prof;  // W8_PROF builds only
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t a, unsigned parity) {
  unsigned ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok)
               : "r"(a), "r"(parity)
               : "memory");
  return ok != 0;
}
// watchdog: a lost transaction (bad tensor map / descriptor) traps after ~2^26 polls instead of hanging the GPU.
// No device print: a call (vprintf) inside the consumer loop makes ptxas serialize every wgmma (C7510). The stuck site
// is left in w8_watchdog (block.x | block.y << 8 | thread << 16 | mbarrier offset << 32 | parity << 63) before the trap.
__device__ unsigned long long w8_watchdog;
__device__ __forceinline__ void watchdog_trap(unsigned long long info) {
  w8_watchdog = info;
  __threadfence_system();
  __trap();
}
__device__ __forceinline__ void mbar_wait(uint32_t a, unsigned parity) {
  unsigned n = 0;
  while (!mbar_try(a, parity)) {
    if (++n > (1u << 26))
      watchdog_trap(uint64_t(blockIdx.x & 0xff) | uint64_t(blockIdx.y & 0xff) << 8 | uint64_t(threadIdx.x & 0xffff) << 16 |
                    uint64_t(a & 0x7fffffffu) << 32 | uint64_t(parity & 1u) << 63);
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int x, int y, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(tm)), "r"(x), "r"(y), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ uint4 ldg_w(const uint8_t* a, uint64_t pol) {  // read-once weight bytes: no L1, L2 evict-first
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a), "l"(pol));
  return v;
}
__device__ __forceinline__ void prefetch_l2(const void* a, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
// K-major, 128-B rows, 128B swizzle, 8-row groups 1 KB apart.
__device__ __forceinline__ uint64_t desc_k(uint32_t a) {
  return ((uint64_t(a) & 0x3FFFFull) >> 4) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned short v) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
  return h;
}
// f16x2 holding two e4m3 values -> bf16x2 of value * 2^-112 (exact, normal), times m2 = 2^(s-15) x2 -> exact weights
__device__ __forceinline__ unsigned f16x2_to_w(unsigned h, unsigned m2) {
  return bf16x2_mul(((h >> 3) & 0x0fff0fffu) | (h & 0x80008000u), m2);
}
// 16 packed bytes = 4 words -> A fragments of two k16 steps: word 2s -> a[s][0] (row g), a[s][1] (row g+8);
// word 2s+1 -> a[s][2], a[s][3]
__device__ __forceinline__ void dequant_frag(const uint4 raw, unsigned m2, unsigned (&a0)[4], unsigned (&a1)[4]) {
  a0[0] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.x & 0xffffu)), m2);
  a0[1] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.x >> 16)), m2);
  a0[2] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.y & 0xffffu)), m2);
  a0[3] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.y >> 16)), m2);
  a1[0] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.z & 0xffffu)), m2);
  a1[1] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.z >> 16)), m2);
  a1[2] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.w & 0xffffu)), m2);
  a1[3] = f16x2_to_w(e4m3x2_to_f16x2(static_cast<unsigned short>(raw.w >> 16)), m2);
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// W8_DQ 1: the e4m3 bits s eeee mmm placed as bf16 bits (sign -> bit 15, eeee mmm -> bits 4..10) ARE the value * 2^-120,
// exactly, subnormals included (e4m3 exponent field 0 -> bf16 subnormal 0.mmm x 2^-126). 4 bytes -> 2 bf16x2 words:
// magnitudes / signs split once (2 LOP3), spread per word (2 PRMT), shift + combine in one IMAD, then x m2 = 2^(s-7).
__device__ __forceinline__ void e4m3x4_to_w(unsigned v, unsigned m2, unsigned& lo, unsigned& hi) {
  const unsigned mg = v & 0x7f7f7f7fu, sg = v & 0x80808080u;
  lo = bf16x2_mul(prmt(mg, 0u, 0x4140u) * 16u + prmt(sg, 0u, 0x1404u), m2);
  hi = bf16x2_mul(prmt(mg, 0u, 0x4342u) * 16u + prmt(sg, 0u, 0x3424u), m2);
}
__device__ __forceinline__ void dequant_frag_int(const uint4 raw, unsigned m2, unsigned (&a0)[4], unsigned (&a1)[4]) {
  e4m3x4_to_w(raw.x, m2, a0[0], a0[1]);
  e4m3x4_to_w(raw.y, m2, a0[2], a0[3]);
  e4m3x4_to_w(raw.z, m2, a1[0], a1[1]);
  e4m3x4_to_w(raw.w, m2, a1[2], a1[3]);
}
__device__ __forceinline__ unsigned scale_m2(unsigned s) {
  const unsigned b = (s + (W8_DQ ? 120u : 112u)) << 7;  // bf16 2^(s-15) (cvt path) or 2^(s-7) (integer path)
  return b | (b << 16);
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, unsigned rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ float4 ld_cluster_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ float4 ld_local_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// window KV record insert of one token by one warp (stock sm90 fp8_ds_mla; identical math to dense.cu): the loads
// (kv_insert_load, at kernel start) and the quantize + record stores (kv_insert_store, after the mainloop) are split
// so the load latency hides under the mainloop instead of delaying warpgroup 1's first wgmma
__device__ __forceinline__ long long kv_insert_load(const Params& p, int t, int lane, unsigned (&raw)[8]) {
  const long long slot = __ldg(p.slot + t);
  const unsigned* kt = reinterpret_cast<const unsigned*>(p.kv + static_cast<long long>(t) * 512);
#pragma unroll
  for (int w = 0; w < 8; ++w) raw[w] = __ldg(kt + 32 * w + lane);  // group w: bf16 pair (64 w + 2 lane, +1); w = 7: RoPE part
  return slot;
}
__device__ __forceinline__ void kv_insert_store(const Params& p, long long slot, int lane, const unsigned (&raw)[8]) {
  if (slot < 0) return;
  const long long pg = slot >> p.shift, ps = slot & ((1LL << p.shift) - 1);
  uint8_t* base = p.cache + pg * p.page_stride;
  uint8_t* data = base + ps * 576;
  uint8_t* scl = base + (576LL << p.shift) + ps * 8;
#pragma unroll
  for (int w = 0; w < 7; ++w) {
    const __nv_bfloat162 hv = *reinterpret_cast<const __nv_bfloat162*>(&raw[w]);
    const float x0 = __low2float(hv), x1 = __high2float(hv);
    float am = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    am = fmaxf(am, 1e-4f);
    const float v = __fmul_rn(am, 1.0f / 448.0f);
    const unsigned bits = __float_as_uint(v);
    const int ex = static_cast<int>(bits >> 23) - ((bits & 0x7fffffu) == 0 ? 127 : 126);
    const float inv = __uint_as_float(static_cast<unsigned>(127 - ex) << 23);
    const __nv_fp8x2_storage_t qv = __nv_cvt_float2_to_fp8x2(make_float2(x0 * inv, x1 * inv), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<__nv_fp8x2_storage_t*>(data + 64 * w + 2 * lane) = qv;
    if (lane == 0) scl[w] = static_cast<uint8_t>(ex + 127);
  }
  reinterpret_cast<unsigned*>(data + 448)[lane] = raw[7];
  if (lane == 0) scl[7] = 0;
}
__device__ __forceinline__ void kv_insert_warp(const Params& p, int t, int lane) {
  unsigned raw[8];
  const long long slot = kv_insert_load(p, t, lane, raw);
  kv_insert_store(p, slot, lane, raw);
}

template <int NT, bool CS>
__global__ void __launch_bounds__(NTHR, 1) w8a16_kernel(const __grid_constant__ CUtensorMap tmx, const Params p) {
  using C = Cfg<NT, CS>;
  constexpr int NWS = C::NWS, NX = C::NX;
  constexpr bool WLDG = W8_WLDG != 0;
  constexpr bool LA3 = !WLDG && W8_LA3 != 0;
  const long long t_k0 = W8_PROF ? clock64() : 0;
  unsigned long long g_k0 = 0;
  if constexpr (W8_PROF) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_k0));
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  // warp index made provably warp-uniform (shfl from lane 0), as CUTLASS does
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int wg = warp >> 2, w = warp & 3;
  const int S = gridDim.x, split = blockIdx.x, tile = blockIdx.y, tiles = gridDim.y;
  // token slice (more than 128 rows, splits == 1 only): blockIdx.z covers token rows [128 z, 128 z + 128)
  const int tok0 = static_cast<int>(blockIdx.z) * 128, ntok = min(p.ntok - tok0, 128);
  const int c0 = p.nchunk * split / S, c1 = p.nchunk * (split + 1) / S, nc = c1 - c0;
  auto chunk = [&](int i) { return c0 + i; };  // global chunk of step i (a per-tile rotated order measured slower)
  const int row0 = tile * TR;
  const int kx0 = (row0 / p.rows_per_group) * p.kx_per_group;
  const uint32_t sbase = smem_u32(smem);
  // cos | sin rows: CS (inverse RoPE in the mainloop): their own region, loaded at the start. RoPE of y only (q_b):
  // no reserved region; they are bulk-loaded into x slot 0 as soon as its last chunk is done (a few chunks before the
  // end), so the plain-GEMM SMEM layout (deeper rings) applies. The fp32 staging image sits at the top of the ring
  // region, clear of x slot 0.
  const bool late_cs = !CS && p.pos != nullptr;
  const uint32_t xring = sbase, wring = sbase + NX * C::XS + wg * NWS * WSTR, cst = CS ? sbase + C::CSOFF : sbase;
  const uint32_t stg = sbase + (C::BODY - C::YS);
  static_assert(CS || C::BODY - C::YS >= C::XS, "the staging image must clear x slot 0 (late cos | sin rows)");
  static_assert(C::XS >= NT * 256, "cos | sin rows must fit in one x slot");
  // barriers: full_w[2][NWS] (one W ring per warpgroup; none with WLDG), full_x[NX], cs_full; x slot release counters
  constexpr int NWB = WLDG ? 0 : 2 * NWS;
  const uint32_t bfw = sbase + C::BAR + wg * 8 * NWS, bfx = sbase + C::BAR + 8 * NWB, bcs = bfx + 8 * NX, xcnt = bcs + 8;
  const bool need_cs = CS && (p.pos != nullptr || p.inv_pos != nullptr);
  const long long* cs_pos = p.pos != nullptr ? p.pos : p.inv_pos;
  const uint64_t pol = policy_evict_first();
  // this warpgroup's W half block of step i's chunk
  auto wblk = [&](int i) { return p.w + (static_cast<size_t>(chunk(i)) * tiles + tile) * CHUNK + wg * HB; };
  // W half-block of chunk c for this warpgroup (each weight byte is read once per call: L2 evict-first)
  auto issue_w = [&](int i) {
    const int s = i % NWS;
    mbar_expect_tx(bfw + 8 * s, HB);
    bulk_g2s_hint(wring + s * WSTR, p.w + (static_cast<size_t>(chunk(i)) * tiles + tile) * CHUNK + wg * HB, HB, bfw + 8 * s, pol);
  };
  auto issue_x = [&](int i) {
    const int s = i % NX;
    const uint32_t fb = bfx + 8 * s, st = xring + s * C::XS;
    mbar_expect_tx(fb, (ABL & 4) ? 0 : C::XS);
    if constexpr (!(ABL & 4)) {
      const int kx = kx0 + chunk(i) * KC;
      tma2d(st, &tmx, kx, tok0, fb);
      tma2d(st + C::XB, &tmx, kx + 64, tok0, fb);
    }
  };
  if (tid == 0) {
    for (int i = 0; i < NWB; ++i) mbar_init(sbase + C::BAR + 8 * i, 1);
    for (int i = 0; i < NX; ++i) mbar_init(bfx + 8 * i, 1);
    mbar_init(bcs, 1);
    for (int i = 0; i < NX; ++i) asm volatile("st.shared.u32 [%0], 0;" ::"r"(xcnt + 4 * i) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // ------------------------------------------------------------------ prologue: first ring fills, cos | sin, KV insert
  // No producer warps: a third warpgroup would cap every thread at 168 registers in ptxas's wgmma pipeline analysis
  // (setmaxnreg does not relax it), which serializes all wgmmas (C7512). The consumers issue their own loads.
  if (!(ABL & 8)) {
    if (w == 0 && lane == 0) {
      if constexpr (WLDG) {
        for (int i = 0; i < (nc < W8_NPF ? nc : W8_NPF); ++i) prefetch_l2(wblk(i), HB);
      } else {
        for (int i = 0; i < (nc < NWS ? nc : NWS); ++i) issue_w(i);
      }
    }
    if (tid == 0) asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
  }
  // everything above reads only constants (weights, the tensor map); everything below may read the previous kernel's
  // outputs (x, kv, positions) or reuse the shared split-K workspace
  if constexpr (W8_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (!(ABL & 8)) {
    if (tid == 0) {
      for (int i = 0; i < (nc < NX ? nc : NX); ++i) issue_x(i);
    }
    if (need_cs) {
      // stock cos | sin rows (256 B per token) gathered with plain 16-byte loads + stores by all threads: 116 small bulk
      // copies would occupy the TMA unit for ~5 us ahead of the weight stream. Read after a barrier (first rotation).
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int t = (tid >> 4) + 16 * j;
        if (t < ntok) {
          const float4 v = __ldg(reinterpret_cast<const float4*>(p.cs + __ldg(cs_pos + tok0 + t) * 64) + (tid & 15));
          asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(cst + t * 256 + (tid & 15) * 16), "f"(v.x), "f"(v.y), "f"(v.z),
                       "f"(v.w)
                       : "memory");
        }
      }
    }
  }
  // q_b only: window KV insert, one token per CTA round-robin; warp 7 loads its first token now, stores after the mainloop
  const int kv_nct = gridDim.x * gridDim.y, kv_me = tile * gridDim.x + split;
  unsigned kv_raw[8];
  long long kv_slot = -1;
  if (warp == 7 && p.kv != nullptr && kv_me < ntok) kv_slot = kv_insert_load(p, tok0 + kv_me, lane, kv_raw);
  long long cpos[8];  // late_cs: positions of tokens (tid >> 4) + 16 j, for the epilogue's cos | sin gather
#pragma unroll
  for (int j = 0; j < 8; ++j) cpos[j] = (late_cs && (tid >> 4) + 16 * j < ntok) ? __ldg(p.pos + tok0 + (tid >> 4) + 16 * j) : 0;
  // ------------------------------------------------------------------ mainloop
  bool cs_ready = false;  // the cos | sin rows are waited for at the first inverse-RoPE chunk, not before the mainloop
  float acc[2][NT / 2];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < NT / 2; ++j) acc[m][j] = 0.f;
  unsigned A[2][2][2][4];  // [buffer][sub-tile][k16 step][reg]
  unsigned long long pt[7] = {0, 0, 0, 0, 0, 0, 0};  // full_x wait, full_w wait, wgmma wait, LDS, issue, loop, dequant
  long long t_loop = W8_PROF ? clock64() : 0;
  // chunk i's x and W have landed (+ the inverse RoPE of its x atom 1 if it holds dims 448..511 of a head)
  auto chunk_ready = [&](int i) {
    const int sw = i % NWS, sx = i % NX;
    long long tq = W8_PROF ? clock64() : 0;
    if constexpr (WLDG) {  // keep the L2 prefetch W8_NPF chunks ahead of the step being made ready
      if (!(ABL & 8) && w == 0 && lane == 0 && i > 0 && i + W8_NPF - 1 < nc) prefetch_l2(wblk(i + W8_NPF - 1), HB);
    }
    if constexpr (!(ABL & 8)) mbar_wait(bfx + 8 * sx, (i / NX) & 1);
    if constexpr (W8_PROF) { const long long t = clock64(); pt[0] += t - tq; tq = t; }
    if constexpr (CS) {
      if (p.inv_pos != nullptr && (((kx0 + chunk(i) * KC) & 511) == 384)) {
        if (!cs_ready) {  // the cos | sin rows stored by all threads in the prologue are visible after this barrier
          __syncthreads();
          cs_ready = true;
        }
        // inverse RoPE in place (stock fp32 formula, bf16 rounding), cos / sin from SMEM; then make the generic-proxy
        // writes visible to wgmma (async proxy) before either warpgroup reads them. No wgmma reads this slot yet.
        const uint32_t xa = xring + sx * C::XS + C::XB;
        const int r = tid >> 1, h = tid & 1;  // row r, pairs 16h .. 16h + 15 (16-byte chunks 4h .. 4h + 3 of the row)
        if (r < ntok) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t ad = xa + r * 128 + (((4 * h + q) ^ (r & 7)) << 4);
            const float4 cv = ld_local_f4(cst + r * 256 + (16 * h + 4 * q) * 4);
            const float4 sv = ld_local_f4(cst + r * 256 + 128 + (16 * h + 4 * q) * 4);
            uint4 u = lds128(ad);
            unsigned* uw = reinterpret_cast<unsigned*>(&u);
            const float c4[4] = {cv.x, cv.y, cv.z, cv.w}, s4[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const __nv_bfloat162 hv = *reinterpret_cast<const __nv_bfloat162*>(&uw[j]);
              const float e = __low2float(hv), o = __high2float(hv);
              const __nv_bfloat162 hr = __floats2bfloat162_rn(__fadd_rn(__fmul_rn(e, c4[j]), __fmul_rn(o, s4[j])),
                                                              __fsub_rn(__fmul_rn(o, c4[j]), __fmul_rn(e, s4[j])));
              uw[j] = *reinterpret_cast<const unsigned*>(&hr);
            }
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ad), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
      }
    }
    if constexpr (!WLDG && !LA3) {
      if constexpr (W8_PROF) tq = clock64();
      if constexpr (!(ABL & 8)) mbar_wait(bfw + 8 * sw, (i / NWS) & 1);
      if constexpr (W8_PROF) pt[1] += clock64() - tq;
    }
  };
  auto w_ready = [&](int i) {  // LA3: chunk i's W slot has landed (waited before its first LDS, apart from x)
    long long tq = W8_PROF ? clock64() : 0;
    if constexpr (!(ABL & 8)) mbar_wait(bfw + 8 * (i % NWS), (i / NWS) & 1);
    if constexpr (W8_PROF) pt[1] += clock64() - tq;
  };
  // raw fp8 bytes + scale bytes of one k32 block (both sub-tiles), loaded one step ahead of their dequant so the LDS
  // latency (long under the wgmma's SMEM traffic) is off the critical path
  constexpr int NR = (WLDG || LA3) ? 4 : 2;  // raw slots: 3 steps ahead (slot = kb), or 1 step (slot = kb & 1)
  uint4 R[NR][2];
  unsigned SE[NR][2];
  auto ldg_kb = [&](int i, int kb) {  // WLDG: this lane's fragment bytes + the scale bytes, straight from global / L2
    long long tq = W8_PROF ? clock64() : 0;
    if constexpr (!(ABL & 8)) {
      const uint8_t* b = wblk(i);
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        R[kb][m] = ldg_w(b + ((kb * 2 + m) * 4 + w) * 512 + lane * 16, pol);
        SE[kb][m] = __ldg(b + HW + kb * 4 + m * 2 + (w >> 1));
      }
    }
    if constexpr (W8_PROF) pt[3] += clock64() - tq;
  };
  constexpr bool SC1 = !WLDG && !LA3 && W8_SC1 != 0;
  uint4 SCn = make_uint4(0, 0, 0, 0), SCc = SCn;  // SC1: scale bytes of the next / current chunk ([kb][4 row blocks])
  auto lds_kb = [&](int i, int kb, int rb) {
    long long tq = W8_PROF ? clock64() : 0;
    const uint32_t wf = wring + (i % NWS) * WSTR, sc = wf + HW;
    if constexpr (SC1) {
      if (kb == 0) SCn = lds128(sc);
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      R[rb][m] = lds128(wf + ((kb * 2 + m) * 4 + w) * 512 + lane * 16);
      if constexpr (!SC1) asm volatile("ld.shared.u8 %0, [%1];" : "=r"(SE[rb][m]) : "r"(sc + kb * 4 + m * 2 + (w >> 1)));
    }
    if (kb == 3) {  // all of this chunk's weight bytes of the warpgroup are loaded: refill the slot now
      // named barriers are warp-uniform: warp 0 of the warpgroup waits for the other three warps (their LDS are
      // performed once the barrier completes), then refills
      if (w == 0) {
        asm volatile("bar.sync %0, 128;" ::"r"(3 + wg) : "memory");
        if (lane == 0 && !(ABL & 8) && i + NWS < nc) issue_w(i + NWS);
      } else {
        asm volatile("bar.arrive %0, 128;" ::"r"(3 + wg) : "memory");
      }
    }
    if constexpr (W8_PROF) pt[3] += clock64() - tq;
  };
  auto deq_kb = [&](int rb, int b, int kb) {
    long long tq = W8_PROF ? clock64() : 0;
    if constexpr (SC1) {
      const unsigned wsc = kb == 0 ? SCc.x : kb == 1 ? SCc.y : kb == 2 ? SCc.z : SCc.w;
#pragma unroll
      for (int m = 0; m < 2; ++m) SE[rb][m] = (wsc >> (8 * (m * 2 + (w >> 1)))) & 0xffu;
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const uint4 raw = R[rb][m];
      if constexpr ((ABL & 1) != 0) {
        const unsigned se = SE[rb][m];
        A[b][m][0][0] = raw.x ^ se; A[b][m][0][1] = raw.x; A[b][m][0][2] = raw.y; A[b][m][0][3] = raw.y;
        A[b][m][1][0] = raw.z; A[b][m][1][1] = raw.z; A[b][m][1][2] = raw.w; A[b][m][1][3] = raw.w;
      } else {
        if constexpr (W8_DQ != 0)
          dequant_frag_int(raw, scale_m2(SE[rb][m]), A[b][m][0], A[b][m][1]);
        else
          dequant_frag(raw, scale_m2(SE[rb][m]), A[b][m][0], A[b][m][1]);
      }
    }
    if constexpr (W8_PROF) pt[6] += clock64() - tq;
  };
  auto wait1 = [&]() {
    long long tq = W8_PROF ? clock64() : 0;
    wg_wait<1>();
    if constexpr (W8_PROF) pt[2] += clock64() - tq;
  };
  if (nc > 0) {
    chunk_ready(0);
    if constexpr (WLDG) {
      ldg_kb(0, 0);
      ldg_kb(0, 1);
      ldg_kb(0, 2);
    } else if constexpr (LA3) {
      w_ready(0);
      lds_kb(0, 0, 0);
      lds_kb(0, 1, 1);
      lds_kb(0, 2, 2);
    } else {
      lds_kb(0, 0, 0);
    }
    SCc = SCn;
    deq_kb(0, 0, 0);
    if constexpr (W8_PP) {
      if (wg == 1) asm volatile("bar.arrive 5, 256;" ::: "memory");  // warpgroup 0 issues first
    }
  }
#pragma unroll 1
  for (int i = 0; i < nc; ++i) {
    const int sx = i % NX;
    const uint32_t xs = xring + sx * C::XS;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int b = kb & 1;
      if (kb > 0) {  // the group of kb - 2 is done (A[b] free); kb - 1 stays in flight during this dequant
        wait1();
        deq_kb((WLDG || LA3) ? kb : b, b, kb);
      }
      if constexpr (W8_PP) asm volatile("bar.sync %0, 256;" ::"r"(5 + wg) : "memory");  // my turn
      long long tq = W8_PROF ? clock64() : 0;
      wg_fence();
      if constexpr (!(ABL & 2)) {
#pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
          const uint64_t db = desc_k(xs + (kb >> 1) * C::XB + (kb & 1) * 64 + k16 * 32);
#pragma unroll
          for (int m = 0; m < 2; ++m) WgRS<NT>::mma(acc[m], A[b][m][k16], db, 1);
        }
      } else {
#pragma unroll
        for (int m = 0; m < 2; ++m) acc[m][kb] += __uint_as_float(A[b][m][0][0] ^ A[b][m][1][3]);  // keep the dequant live
      }
      wg_commit();
      if constexpr (W8_PP) {  // the other warpgroup's turn (warpgroup 1 skips its very last arrive: counts balance)
        if (wg == 0 || i + 1 < nc || kb < 3) asm volatile("bar.arrive %0, 256;" ::"r"(6 - wg) : "memory");
      }
      if constexpr (W8_PROF) pt[4] += clock64() - tq;
      if constexpr (WLDG) {  // raw bytes 3 steps ahead: (i, 3) or (i + 1, kb - 1); under kb 3 the next chunk's x
        if (kb == 0)
          ldg_kb(i, 3);
        else if (i + 1 < nc)
          ldg_kb(i + 1, kb - 1);
        if (kb == 3 && i + 1 < nc) chunk_ready(i + 1);
      } else if constexpr (LA3) {  // raw bytes 3 steps ahead from SMEM; the next chunk's W slot is waited at kb 1
        if (kb == 0) {
          lds_kb(i, 3, 3);  // last LDS of chunk i: its W slot is refilled inside
        } else if (i + 1 < nc) {
          if (kb == 1) w_ready(i + 1);
          lds_kb(i + 1, kb - 1, kb - 1);
        }
        if (kb == 3 && i + 1 < nc) chunk_ready(i + 1);
      } else {  // raw bytes of the next step: kb + 1 of this chunk, or (under kb 3) the next chunk's barriers + kb 0
        if (kb < 3) {
          lds_kb(i, kb + 1, (kb + 1) & 1);
        } else if (i + 1 < nc) {
          chunk_ready(i + 1);
          lds_kb(i + 1, 0, 0);
        }
      }
    }
    // the next chunk's first k32 dequant runs under this chunk's last group (kb 2 done: A[0] free); then drain, so no
    // wgmma group stays in flight across the loop back-edge (else ptxas serializes all wgmmas, C7513)
    if (i + 1 < nc) {
      wait1();
      SCc = SCn;
      deq_kb(0, 0, 0);
    }
    wg_wait<0>();
    if (w == 0 && !(ABL & 8)) {
      // this warpgroup is done with x slot sx; the second of the two warpgroups to get here refills it (no waiting)
      unsigned old = 0;
      if (lane == 0) asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(xcnt + 4 * sx) : "memory");
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old & 1u) {
        if (i + NX < nc) {
          if (lane == 0) issue_x(i + NX);
        }
      }
    }
  }
  if constexpr (W8_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (warp == 7 && p.kv != nullptr && kv_me < ntok) {
    kv_insert_store(p, kv_slot, lane, kv_raw);
    for (int t = kv_me + kv_nct; t < ntok; t += kv_nct) kv_insert_warp(p, tok0 + t, lane);
  }
  const long long t_le = W8_PROF ? clock64() : 0;
  long long t_b0 = t_le, t_b1 = t_le;  // split-K barrier enter / leave (thread 0, profile builds)
  if constexpr (W8_PROF) pt[5] = t_le - t_loop;
  __syncthreads();  // both warpgroups are out of the rings (every issued copy was consumed)
  // ------------------------------------------------------------------ epilogue: accumulators -> staging Ys[token][row]
  float4 csv[8];  // late_cs: this thread's share of the cos | sin rows (16 B of token (tid >> 4) + 16 j), into x slot 0
  if (late_cs) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if ((tid >> 4) + 16 * j < ntok) csv[j] = __ldg(reinterpret_cast<const float4*>(p.cs + cpos[j] * 64) + (tid & 15));
  }
  {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int r = (2 * wg + m) * 64 + 16 * w + g;
#pragma unroll
      for (int j = 0; j < NT / 8; ++j) {
        const int tk = 8 * j + 2 * t;
        const uint32_t a0 = stg + (tk * YP + r) * 4;
        asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0), "f"(acc[m][4 * j + 0]) : "memory");
        asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + YP * 4), "f"(acc[m][4 * j + 1]) : "memory");
        asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + 32), "f"(acc[m][4 * j + 2]) : "memory");
        asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + YP * 4 + 32), "f"(acc[m][4 * j + 3]) : "memory");
      }
    }
  }
  if (late_cs) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int t = (tid >> 4) + 16 * j;
      if (t < ntok)
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(cst + t * 256 + (tid & 15) * 16), "f"(csv[j].x), "f"(csv[j].y),
                     "f"(csv[j].z), "f"(csv[j].w)
                     : "memory");
    }
  }
  __syncthreads();
  const long long t_st = W8_PROF ? clock64() : 0;  // staging done
  long long t_pw = t_st;                           // partial writes issued
  const int items_all = ntok * (TR / 8);
  if (S > 1) {
    // ------------------------------------------------------------------ split-K: partial -> L2 workspace, counter barrier
    if constexpr (W8_BST) {
      // workspace block (tile, split) = the staging image [NT][YP] fp32: each warp bulk-stores its share of token rows
      float* wsp = p.ws + (static_cast<size_t>(tile) * S + split) * NT * YP;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // staging (generic writes) -> bulk copy (async reads)
      __syncthreads();
      if (lane == 0) {
        const int rpw = (ntok + 7) / 8, r0 = warp * rpw, r1 = min(ntok, r0 + rpw);
        if (r1 > r0) {
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(wsp + r0 * YP),
                       "r"(stg + r0 * YP * 4), "r"((r1 - r0) * YP * 4)
                       : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
    } else {
      float* wsp = p.ws + (static_cast<size_t>(tile) * S + split) * NT * YP;
      for (int it = tid; it < items_all; it += NTHR) {
        const int tk = it / (TR / 8), rg = it % (TR / 8);
        const uint32_t la = stg + (tk * YP + 8 * rg) * 4;
        float4* dst = reinterpret_cast<float4*>(wsp + tk * YP + 8 * rg);
        dst[0] = ld_local_f4(la);
        dst[1] = ld_local_f4(la + 16);
      }
    }
    if constexpr (W8_PROF) t_pw = clock64();
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      if constexpr (W8_PROF) t_b0 = clock64();
      const unsigned old = atomicAdd(p.cnt + tile, 1u);
      const unsigned target = (old / S + 1) * S;
      volatile unsigned* vc = p.cnt + tile;
      unsigned n = 0;
      while (*vc < target) {
        if (++n > (1u << 28)) watchdog_trap(0xffffull << 16 | uint64_t(tile & 0xffff));  // thread 0xffff = split-K barrier
      }
      __threadfence();
      if constexpr (W8_PROF) t_b1 = clock64();
    }
    __syncthreads();
  }
  // ------------------------------------------------------------------ reduce (fixed order 0..S-1) + RoPE + store
  // cos | sin rows: CS since the prologue (+ a barrier), late_cs stored before the staging barrier above
  const int t_lo = ntok * split / S, t_hi = ntok * (split + 1) / S;
  const int items = (t_hi - t_lo) * (TR / 8);
  const float* wst = p.ws + static_cast<size_t>(tile) * S * NT * YP;
  constexpr int RIF = W8_RIF;
  for (int it0 = tid; it0 < items; it0 += RIF * NTHR) {
    // RIF items per thread, all S x 2 partial loads issued before any use (the reduce is L2-latency bound otherwise)
    float4 pa[RIF][8], pb[RIF][8];
#pragma unroll
    for (int u = 0; u < RIF; ++u) {
      const int it = it0 + u * NTHR;
      if (it < items && S > 1) {
        const int tk = t_lo + it / (TR / 8), rg = it % (TR / 8);
#pragma unroll
        for (int q = 0; q < 8; ++q)
          if (q < S) {
            const float4* src = reinterpret_cast<const float4*>(wst + (static_cast<size_t>(q) * NT + tk) * YP + 8 * rg);
            pa[u][q] = __ldcg(src);
            pb[u][q] = __ldcg(src + 1);
          }
      }
    }
#pragma unroll
    for (int u = 0; u < RIF; ++u) {
      const int it = it0 + u * NTHR;
      if (it >= items) break;
      const int tk = t_lo + it / (TR / 8), rg = it % (TR / 8);
      float v[8];
      if (S == 1) {
        const uint32_t la = stg + (tk * YP + 8 * rg) * 4;
        const float4 a = ld_local_f4(la), b = ld_local_f4(la + 16);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
      } else {
        v[0] = pa[u][0].x; v[1] = pa[u][0].y; v[2] = pa[u][0].z; v[3] = pa[u][0].w;
        v[4] = pb[u][0].x; v[5] = pb[u][0].y; v[6] = pb[u][0].z; v[7] = pb[u][0].w;
#pragma unroll
        for (int q = 1; q < 8; ++q)  // fixed order 0..S-1
          if (q < S) {
            v[0] += pa[u][q].x; v[1] += pa[u][q].y; v[2] += pa[u][q].z; v[3] += pa[u][q].w;
            v[4] += pb[u][q].x; v[5] += pb[u][q].y; v[6] += pb[u][q].z; v[7] += pb[u][q].w;
          }
        if (W8_SMAX > 8 && S > 8) {  // partials 8..S-1 (S <= 16: small-N GEMMs such as q_a | wkv), loads in flight together, same order
          float4 qa[8], qb[8];
#pragma unroll
          for (int q = 8; q < 16; ++q)
            if (q < S) {
              const float4* src = reinterpret_cast<const float4*>(wst + (static_cast<size_t>(q) * NT + tk) * YP + 8 * rg);
              qa[q - 8] = __ldcg(src);
              qb[q - 8] = __ldcg(src + 1);
            }
#pragma unroll
          for (int q = 8; q < 16; ++q)
            if (q < S) {
              v[0] += qa[q - 8].x; v[1] += qa[q - 8].y; v[2] += qa[q - 8].z; v[3] += qa[q - 8].w;
              v[4] += qb[q - 8].x; v[5] += qb[q - 8].y; v[6] += qb[q - 8].z; v[7] += qb[q - 8].w;
            }
        }
      }
      const int col = row0 + 8 * rg;
      if (p.pos != nullptr && (col & 511) >= 448) {
        const int i0 = ((col & 511) - 448) >> 1;
        const float4 cv = ld_local_f4(cst + tk * 256 + i0 * 4), sv = ld_local_f4(cst + tk * 256 + 128 + i0 * 4);
        const float c4[4] = {cv.x, cv.y, cv.z, cv.w}, s4[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float e = bfr(v[2 * j]), o = bfr(v[2 * j + 1]);
          v[2 * j] = __fsub_rn(__fmul_rn(e, c4[j]), __fmul_rn(o, s4[j]));
          v[2 * j + 1] = __fadd_rn(__fmul_rn(e, s4[j]), __fmul_rn(o, c4[j]));
        }
      }
      uint4 o;
      __nv_bfloat162 h;
      h = __floats2bfloat162_rn(v[0], v[1]); o.x = *reinterpret_cast<unsigned*>(&h);
      h = __floats2bfloat162_rn(v[2], v[3]); o.y = *reinterpret_cast<unsigned*>(&h);
      h = __floats2bfloat162_rn(v[4], v[5]); o.z = *reinterpret_cast<unsigned*>(&h);
      h = __floats2bfloat162_rn(v[6], v[7]); o.w = *reinterpret_cast<unsigned*>(&h);
      *reinterpret_cast<uint4*>(p.y + static_cast<long long>(tok0 + tk) * p.ldy + col) = o;
    }
  }
  if constexpr (W8_PROF) {
    if ((tid & 127) == 0 && p.prof != nullptr) {
      unsigned long long g_end;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_end));
      unsigned long long* pp = p.prof + ((static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 2 + wg) * 16;
      for (int k = 0; k < 6; ++k) pp[k] = pt[k];
      pp[6] = nc;
      pp[7] = t_loop - t_k0;      // prologue: kernel start -> mainloop
      pp[8] = clock64() - t_le;   // epilogue: staging, split-K partials + barrier, reduce, stores
      pp[9] = g_k0;
      pp[10] = g_end;
      pp[11] = t_b0 - t_le;       // staging + split-K partial write (thread 0 only)
      pp[12] = t_b1 - t_b0;       // split-K counter barrier wait (thread 0 only)
      pp[13] = pt[6];             // dequant
      pp[14] = t_st - t_le;       // other warpgroup + accumulator staging
      pp[15] = t_pw - t_st;       // split-K partial write loop (issue); t_b0 - t_pw = __threadfence + sync
    }
  }
}


// Here one CTA = 128 weight rows (half h of a 256-row tile = one packed half block of every chunk) x NW <= 256 token
// rows: warpgroup c owns 64-row sub-tile c of the half, dequantizes it once per k32 and issues ONE RS wgmma m64 nNW k16 per k16
// step over all tokens (accumulators 64 x NW fp32: 128 registers at NW 256, as the served 2 x 64 x 128). Per output element the
// weights (same bytes, same scale), x and the k16 order are the served splits-1 path's -> bitwise equal. Stage s = one x chunk
// (two TMA tiles of 64 k x NW rows) + the W half block, one mbarrier; the second warpgroup done with a stage refills it.
template <int NW>
struct CfgW {
  static constexpr int XB = NW * 128;
  static constexpr int XS = 2 * XB;
  static constexpr int ST = XS + WSTR;
  static constexpr int YPW = 128 + 4;  // staging pitch (floats) for the CTA's 128 columns
  static constexpr int YS = NW * YPW * 4;
  static constexpr int NS0 = (SMEM_MAX - 256 - 1024) / ST;
  static constexpr int NS = NS0 < 4 ? NS0 : 4;
  static constexpr int RING = NS * ST;
  static constexpr int BODY = RING > YS ? RING : YS;
  static constexpr int BAR = BODY;
  static constexpr int SMEM = BAR + 256 + 1024;
  static_assert(NS >= 2, "two stages");
  static_assert(SMEM <= SMEM_MAX, "smem");
  static_assert(XS % 1024 == 0, "x tiles 1 KB aligned (SW128)");
};

template <int NW>
__global__ void __launch_bounds__(NTHR, 1) w8a16w_kernel(const __grid_constant__ CUtensorMap tmx, const Params p) {
  using C = CfgW<NW>;
  constexpr int NS = C::NS;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int wg = warp >> 2, w = warp & 3;
  const int tile = static_cast<int>(blockIdx.y) >> 1, h = static_cast<int>(blockIdx.y) & 1, tiles = static_cast<int>(gridDim.y) >> 1;
  const int tok0 = static_cast<int>(blockIdx.z) * NW, ntok = min(p.ntok - tok0, NW);
  const int nc = p.nchunk;
  const int row0 = tile * TR + h * 128;
  const int kx0 = (tile * TR / p.rows_per_group) * p.kx_per_group;
  const uint32_t sbase = smem_u32(smem);
  const uint32_t xring = sbase, wring = sbase + NS * C::XS, stg = sbase;
  const uint32_t bfs = sbase + C::BAR, scnt = bfs + 8 * NS;
  const uint64_t pol = policy_evict_first();
  auto issue_w = [&](int i) {  // the stage's W half block (expect_tx of the whole stage first)
    const int s = i % NS;
    mbar_expect_tx(bfs + 8 * s, C::XS + HB);
    bulk_g2s_hint(wring + s * WSTR, p.w + (static_cast<size_t>(i) * tiles + tile) * CHUNK + h * HB, HB, bfs + 8 * s, pol);
  };
  auto issue_x = [&](int i) {
    const int s = i % NS;
    const uint32_t st = xring + s * C::XS;
    const int kx = kx0 + i * KC;
    tma2d(st, &tmx, kx, tok0, bfs + 8 * s);
    tma2d(st + C::XB, &tmx, kx + 64, tok0, bfs + 8 * s);
  };
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) mbar_init(bfs + 8 * i, 1);
    for (int i = 0; i < NS; ++i) asm volatile("st.shared.u32 [%0], 0;" ::"r"(scnt + 4 * i) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    for (int i = 0; i < (nc < NS ? nc : NS); ++i) issue_w(i);  // weights: constants, before the PDL wait
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
  }
  if constexpr (W8_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) {
    for (int i = 0; i < (nc < NS ? nc : NS); ++i) issue_x(i);
  }
  float acc[NW / 2];
#pragma unroll
  for (int j = 0; j < NW / 2; ++j) acc[j] = 0.f;
  unsigned A[2][2][4];  // [buffer][k16 step][reg]
  uint4 R[2];
  uint4 SCn = make_uint4(0, 0, 0, 0), SCc = SCn;
  auto stage_ready = [&](int i) { mbar_wait(bfs + 8 * (i % NS), (i / NS) & 1); };
  auto lds_kb = [&](int i, int kb, int rb) {
    const uint32_t wf = wring + (i % NS) * WSTR;
    if (kb == 0) SCn = lds128(wf + HW);
    R[rb] = lds128(wf + ((kb * 2 + wg) * 4 + w) * 512 + lane * 16);
  };
  auto deq_kb = [&](int rb, int b, int kb) {
    const unsigned wsc = kb == 0 ? SCc.x : kb == 1 ? SCc.y : kb == 2 ? SCc.z : SCc.w;
    const unsigned se = (wsc >> (8 * (wg * 2 + (w >> 1)))) & 0xffu;
    if constexpr (W8_DQ != 0)
      dequant_frag_int(R[rb], scale_m2(se), A[b][0], A[b][1]);
    else
      dequant_frag(R[rb], scale_m2(se), A[b][0], A[b][1]);
  };
  if (nc > 0) {
    stage_ready(0);
    lds_kb(0, 0, 0);
    SCc = SCn;
    deq_kb(0, 0, 0);
  }
#pragma unroll 1
  for (int i = 0; i < nc; ++i) {
    const int s = i % NS;
    const uint32_t xs = xring + s * C::XS;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int b = kb & 1;
      if (kb > 0) {
        wg_wait<1>();
        deq_kb(b, b, kb);
      }
      wg_fence();
#pragma unroll
      for (int k16 = 0; k16 < 2; ++k16) {
        const uint64_t db = desc_k(xs + (kb >> 1) * C::XB + (kb & 1) * 64 + k16 * 32);
        WgRS<NW>::mma(acc, A[b][k16], db, 1);
      }
      wg_commit();
      if (kb < 3) {
        lds_kb(i, kb + 1, (kb + 1) & 1);
      } else if (i + 1 < nc) {
        stage_ready(i + 1);
        lds_kb(i + 1, 0, 0);
      }
    }
    if (i + 1 < nc) {
      wg_wait<1>();
      SCc = SCn;
      deq_kb(0, 0, 0);
    }
    wg_wait<0>();
    // stage s: this warpgroup's wgmmas (x) are complete and its 4 warps have consumed their W LDS; the second warpgroup
    // here refills the stage with chunk i + NS (no waiting)
    if (w == 0) {
      asm volatile("bar.sync %0, 128;" ::"r"(3 + wg) : "memory");
      unsigned old = 0;
      if (lane == 0) asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(scnt + 4 * s) : "memory");
      old = __shfl_sync(0xffffffffu, old, 0);
      if ((old & 1u) && i + NS < nc && lane == 0) {
        issue_w(i + NS);
        issue_x(i + NS);
      }
    } else {
      asm volatile("bar.arrive %0, 128;" ::"r"(3 + wg) : "memory");
    }
  }
  if constexpr (W8_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();  // every issued stage was consumed; the ring region becomes the staging image
  {
    const int g = lane >> 2, t = lane & 3;
    const int r = wg * 64 + 16 * w + g;
#pragma unroll
    for (int j = 0; j < NW / 8; ++j) {
      const int tk = 8 * j + 2 * t;
      const uint32_t a0 = stg + (tk * C::YPW + r) * 4;
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0), "f"(acc[4 * j + 0]) : "memory");
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + C::YPW * 4), "f"(acc[4 * j + 1]) : "memory");
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + 32), "f"(acc[4 * j + 2]) : "memory");
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(a0 + C::YPW * 4 + 32), "f"(acc[4 * j + 3]) : "memory");
    }
  }
  __syncthreads();
  const int items = ntok * 16;
  for (int it = tid; it < items; it += NTHR) {
    const int tk = it >> 4, rg = it & 15;
    const uint32_t la = stg + (tk * C::YPW + 8 * rg) * 4;
    const float4 a = ld_local_f4(la), b = ld_local_f4(la + 16);
    uint4 o;
    __nv_bfloat162 hv;
    hv = __floats2bfloat162_rn(a.x, a.y); o.x = *reinterpret_cast<unsigned*>(&hv);
    hv = __floats2bfloat162_rn(a.z, a.w); o.y = *reinterpret_cast<unsigned*>(&hv);
    hv = __floats2bfloat162_rn(b.x, b.y); o.z = *reinterpret_cast<unsigned*>(&hv);
    hv = __floats2bfloat162_rn(b.z, b.w); o.w = *reinterpret_cast<unsigned*>(&hv);
    *reinterpret_cast<uint4*>(p.y + static_cast<long long>(tok0 + tk) * p.ldy + row0 + 8 * rg) = o;
  }
}

}  // namespace w8

// ------------------------------------------------------------------------------------------------ host
// C ABI: run() below is the one launch path of
// both the pybind gemm (its checks and messages unchanged) and the C entry c_gemm; kernels unchanged. One-time state: the per-device dynamic-SMEM attribute (pybind: lazily on first use, as before; C: init).
static_assert(W8A16_CHUNK_BYTES == w8::CHUNK, "w8a16_cabi.h W8A16_CHUNK_BYTES");

#ifndef W8_NTMIN
#define W8_NTMIN 96  // 32 = token tiles of 32 / 64 for buckets of <= 32 / 64 rows
#endif

// failures of run(), mapped to the pybind path's messages / the C ABI's codes
enum RunErr { RE_OK = 0, RE_TMAP, RE_SMEM, RE_LAUNCH_EX, RE_LAUNCH, RE_TOKENS, RE_INIT };
struct RunRes {
  RunErr e;
  int detail;  // CUresult / cudaError_t
};

// per (instantiation, device): the SMEM attribute is set (pybind: on first use, as the old function-local static of each
// launch1<NT, CS>; C ABI: w8a16.init sets every instantiation of this build)
template <int NT, bool CS>
static bool* init_dev() {
  static bool f[64] = {};
  return f;
}

static RunRes make_tmap(const void* base, int64_t rows, int64_t cols, int64_t ld, int nt, CUtensorMap* m) {
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(ld) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(nt)};
  const cuuint32_t es[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box,
                                            es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return {r == CUDA_SUCCESS ? RE_OK : RE_TMAP, static_cast<int>(r)};
}

template <int NT, bool CS>
static cudaError_t set_smem() {
  return cudaFuncSetAttribute(w8::w8a16_kernel<NT, CS>, cudaFuncAttributeMaxDynamicSharedMemorySize, w8::Cfg<NT, CS>::SMEM);
}
template <int NT, bool CS>
static cudaError_t init_one(int dev) {
  const cudaError_t e = set_smem<NT, CS>();
  if (e == cudaSuccess) init_dev<NT, CS>()[dev & 63] = true;
  return e;
}

static cudaError_t prime_dev(int dev);  // below: every instantiation of the device at once
template <int NT, bool CS>
static RunRes launch1(const CUtensorMap& tm, const w8::Params& p, int S, int tiles, int nslices, cudaStream_t st, bool lazy) {
  auto kern = w8::w8a16_kernel<NT, CS>;
  const int smem = w8::Cfg<NT, CS>::SMEM;
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!init_dev<NT, CS>()[cur_dev & 63]) {
    if (!lazy) return {RE_INIT, 0};
    if (prime_dev(cur_dev) != cudaSuccess || !init_dev<NT, CS>()[cur_dev & 63]) return {RE_SMEM, 0};
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(S, tiles, nslices);
  cfg.blockDim = dim3(w8::NTHR, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = W8_PDL ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, kern, tm, p) != cudaSuccess) return {RE_LAUNCH_EX, 0};
  const cudaError_t e = cudaGetLastError();
  return {e == cudaSuccess ? RE_OK : RE_LAUNCH, static_cast<int>(e)};
}
template <int NT>
static RunRes launch(const CUtensorMap& tm, const w8::Params& p, int S, int tiles, int nslices, cudaStream_t st, bool lazy) {
  if (p.inv_pos != nullptr)  // the RoPE of y (q_b) needs no cos | sin region: its rows land in x slot 0 late (late_cs)
    return launch1<NT, true>(tm, p, S, tiles, nslices, st, lazy);
  return launch1<NT, false>(tm, p, S, tiles, nslices, st, lazy);
}

template <int NW>
static bool* init_dev_w() {
  static bool f[64] = {};
  return f;
}
template <int NW>
static cudaError_t init_one_w(int dev) {
  const cudaError_t e = cudaFuncSetAttribute(w8::w8a16w_kernel<NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, w8::CfgW<NW>::SMEM);
  if (e == cudaSuccess) init_dev_w<NW>()[dev & 63] = true;
  return e;
}
template <int NW>
static RunRes launch_w(const CUtensorMap& tm, const w8::Params& p, int tiles, int nz, cudaStream_t st, bool lazy) {
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!init_dev_w<NW>()[cur_dev & 63]) {
    if (!lazy) return {RE_INIT, 0};
    if (prime_dev(cur_dev) != cudaSuccess || !init_dev_w<NW>()[cur_dev & 63]) return {RE_SMEM, 0};
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, 2 * tiles, nz);
  cfg.blockDim = dim3(w8::NTHR, 1, 1);
  cfg.dynamicSmemBytes = w8::CfgW<NW>::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = W8_PDL ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, w8::w8a16w_kernel<NW>, tm, p) != cudaSuccess) return {RE_LAUNCH_EX, 0};
  const cudaError_t e = cudaGetLastError();
  return {e == cudaSuccess ? RE_OK : RE_LAUNCH, static_cast<int>(e)};
}
// prime: set the SMEM attribute of EVERY instantiation (served + wide) on device dev, so
// no module load / attribute call happens at a first call inside a CUDA graph capture (CUDA_MODULE_LOADING=LAZY). Called by the
// pybind prime() (w8a16.py build() after load: all devices), by c_init, and by any lazy first init (all at once, not per NW).
// wide path per device: set by prime / c_init; a failed wide init leaves the device on the served z-slice path
static bool g_wide_ok[64] = {};
static int g_wide_err[64] = {};                     // the cudaError of a failed wide init, 0 = ok / not tried
static std::atomic<long long> g_wide_launches{0};   // (obs) host launches / graph captures on the wide kernel
static std::atomic<long long> g_wide_fallbacks{0};  // (obs) wide-eligible calls that ran the served kernel (wide not ok)
static cudaError_t prime_dev(int dev) {
  cudaError_t e = cudaSuccess;
#define W8_PRIME(NT)                                                \
  if (e == cudaSuccess) e = init_one<NT, true>(dev);                \
  if (e == cudaSuccess) e = init_one<NT, false>(dev);
#if W8_NTMIN <= 32
  W8_PRIME(32)
#endif
#if W8_NTMIN <= 64
  W8_PRIME(64)
#endif
  W8_PRIME(96)
  W8_PRIME(104)
  W8_PRIME(112)
  W8_PRIME(120)
  W8_PRIME(128)
#undef W8_PRIME
  if (e != cudaSuccess) return e;  // served path failed: fatal (as before)
#if W8_WIDE
  cudaError_t ew = cudaSuccess;
  if (ew == cudaSuccess) ew = init_one_w<64>(dev);
  if (ew == cudaSuccess) ew = init_one_w<72>(dev);
  if (ew == cudaSuccess) ew = init_one_w<80>(dev);
  if (ew == cudaSuccess) ew = init_one_w<88>(dev);
  if (ew == cudaSuccess) ew = init_one_w<96>(dev);
  if (ew == cudaSuccess) ew = init_one_w<104>(dev);
  if (ew == cudaSuccess) ew = init_one_w<112>(dev);
  if (ew == cudaSuccess) ew = init_one_w<120>(dev);
  if (ew == cudaSuccess) ew = init_one_w<128>(dev);
  if (ew == cudaSuccess) ew = init_one_w<136>(dev);
  if (ew == cudaSuccess) ew = init_one_w<144>(dev);
  if (ew == cudaSuccess) ew = init_one_w<152>(dev);
  if (ew == cudaSuccess) ew = init_one_w<160>(dev);
  if (ew == cudaSuccess) ew = init_one_w<168>(dev);
  if (ew == cudaSuccess) ew = init_one_w<176>(dev);
  if (ew == cudaSuccess) ew = init_one_w<184>(dev);
  if (ew == cudaSuccess) ew = init_one_w<192>(dev);
  if (ew == cudaSuccess) ew = init_one_w<200>(dev);
  if (ew == cudaSuccess) ew = init_one_w<208>(dev);
  if (ew == cudaSuccess) ew = init_one_w<216>(dev);
  if (ew == cudaSuccess) ew = init_one_w<224>(dev);
  if (ew == cudaSuccess) ew = init_one_w<232>(dev);
  if (ew == cudaSuccess) ew = init_one_w<240>(dev);
  if (ew == cudaSuccess) ew = init_one_w<248>(dev);
  if (ew == cudaSuccess) ew = init_one_w<256>(dev);
  g_wide_ok[dev & 63] = ew == cudaSuccess;  // wide failure: not fatal, the device keeps the served path
  g_wide_err[dev & 63] = static_cast<int>(ew);
  if (ew != cudaSuccess) (void)cudaGetLastError();
#endif
  return cudaSuccess;
}

// wide token tiling for ntok > 128
// nz = max(2, ceil(ntok / 256)) z slices of NW = ceil(ntok / nz) rounded up to a multiple of 8 (>= 64) token rows
static void wide_tiling(int ntok, int* nw, int* nz) {
  int z = (ntok + 255) / 256;
  if (z < 2) z = 2;
  const int per = (ntok + z - 1) / z;
  *nw = per < 64 ? 64 : (per + 7) / 8 * 8;
  *nz = (ntok + *nw - 1) / *nw;
}
static bool use_wide(const FprW8a16GemmArgs& a) {
  // decode buckets only; larger calls (mixed wo up to max_mixed_rows) keep the served path
  return W8_WIDE && a.ntok > 128 && a.ntok <= 320 && a.splits == 1 && a.pos == nullptr && a.inv_pos == nullptr && a.kv == nullptr &&
         a.prof == nullptr && g_wide_ok[a.call.device & 63];
}

// the launch (both paths): Params from the argument struct, the token tile, the TMA map, one launch
static RunRes run(const FprW8a16GemmArgs& a, bool lazy) {
  w8::Params p{};
  p.w = a.packed;
  p.y = static_cast<__nv_bfloat16*>(a.y);
  p.ldy = static_cast<int>(a.ldy);
  p.ntok = static_cast<int>(a.ntok);
  p.nchunk = static_cast<int>(a.nchunk);
  p.rows_per_group = static_cast<int>(a.rows_per_group);
  p.kx_per_group = static_cast<int>(a.kx_per_group);
  if (a.prof != nullptr) p.prof = reinterpret_cast<unsigned long long*>(a.prof);
  if (a.splits > 1) {
    p.ws = a.ws;
    p.cnt = reinterpret_cast<unsigned*>(a.cnt);
  }
  if (a.pos != nullptr) {
    p.pos = reinterpret_cast<const long long*>(a.pos);
    p.cs = a.cs;
  }
  if (a.inv_pos != nullptr) {
    p.inv_pos = reinterpret_cast<const long long*>(a.inv_pos);
    p.cs = a.cs;
  }
  if (a.kv != nullptr) {
    p.kv = static_cast<const __nv_bfloat16*>(a.kv);
    p.slot = reinterpret_cast<const long long*>(a.slot);
    p.cache = a.cache;
    p.page_stride = a.page_stride;
    p.shift = static_cast<int>(a.shift);
  }
  const int ntok = static_cast<int>(a.ntok);
  const int nslices = (ntok + 127) / 128;
  if (W8_WIDE && ntok > 128 && ntok <= 320 && a.splits == 1 && a.pos == nullptr && a.inv_pos == nullptr && a.kv == nullptr &&
      a.prof == nullptr && !g_wide_ok[a.call.device & 63])
    g_wide_fallbacks.fetch_add(1, std::memory_order_relaxed);   // (obs) eligible, but this device's wide init failed
  const int tiles = static_cast<int>(a.tiles);
  if (use_wide(a)) {
    g_wide_launches.fetch_add(1, std::memory_order_relaxed);
    int nw = 0, nz = 0;
    wide_tiling(ntok, &nw, &nz);
    CUtensorMap tmw;
    const RunRes rw = make_tmap(a.x, a.ntok, a.kx, a.x_ld, nw, &tmw);
    if (rw.e != RE_OK) return rw;
    switch (nw) {
      case 64: return launch_w<64>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 72: return launch_w<72>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 80: return launch_w<80>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 88: return launch_w<88>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 96: return launch_w<96>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 104: return launch_w<104>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 112: return launch_w<112>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 120: return launch_w<120>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 128: return launch_w<128>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 136: return launch_w<136>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 144: return launch_w<144>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 152: return launch_w<152>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 160: return launch_w<160>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 168: return launch_w<168>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 176: return launch_w<176>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 184: return launch_w<184>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 192: return launch_w<192>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 200: return launch_w<200>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 208: return launch_w<208>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 216: return launch_w<216>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 224: return launch_w<224>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 232: return launch_w<232>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 240: return launch_w<240>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 248: return launch_w<248>(tmw, p, tiles, nz, a.call.stream, lazy);
      case 256: return launch_w<256>(tmw, p, tiles, nz, a.call.stream, lazy);
      default: return {RE_TOKENS, 0};
    }
  }
  const int nt = nslices > 1 ? 128 : (ntok + 7) / 8 * 8;
  const int ntx = (W8_NTMIN <= 32 && nt <= 32) ? 32 : (W8_NTMIN <= 64 && nt <= 64) ? 64 : nt < 96 ? 96 : nt;
  CUtensorMap tm;
  const RunRes r = make_tmap(a.x, a.ntok, a.kx, a.x_ld, ntx, &tm);
  if (r.e != RE_OK) return r;
  cudaStream_t st = a.call.stream;
  const int S = static_cast<int>(a.splits);
  switch (ntx) {
#if W8_NTMIN <= 32
    case 32: return launch<32>(tm, p, S, tiles, nslices, st, lazy);
#endif
#if W8_NTMIN <= 64
    case 64: return launch<64>(tm, p, S, tiles, nslices, st, lazy);
#endif
    case 96: return launch<96>(tm, p, S, tiles, nslices, st, lazy);
    case 104: return launch<104>(tm, p, S, tiles, nslices, st, lazy);
    case 112: return launch<112>(tm, p, S, tiles, nslices, st, lazy);
    case 120: return launch<120>(tm, p, S, tiles, nslices, st, lazy);
    case 128: return launch<128>(tm, p, S, tiles, nslices, st, lazy);
    default: return {RE_TOKENS, 0};
  }
}

// ---- C ABI
static int c_init(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev != static_cast<const FprW8a16InitArgs*>(p)->call.device) return FPR_E_ARG;
  cudaError_t e = cudaSuccess;
#define W8_SET(NT)                                                  \
  if (e == cudaSuccess) e = init_one<NT, true>(dev);                \
  if (e == cudaSuccess) e = init_one<NT, false>(dev);
#if W8_NTMIN <= 32
  W8_SET(32)
#endif
#if W8_NTMIN <= 64
  W8_SET(64)
#endif
  W8_SET(96)
  W8_SET(104)
  W8_SET(112)
  W8_SET(120)
  W8_SET(128)
#undef W8_SET
#if W8_WIDE
  if (e == cudaSuccess) e = prime_dev(dev);  // wide instantiations: non-fatal (g_wide_ok), served already set
#endif
  return e == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

static int c_gemm(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprW8a16GemmArgs*>(p);
#ifdef FPR_CABI_DEBUG
  int d = -1;
  if (cudaGetDevice(&d) != cudaSuccess || d != a.call.device) return FPR_E_ARG;
#endif
  // the pybind path's value checks (its dtype / layout checks are the caller's contract here)
  if (a.x == nullptr || a.packed == nullptr || a.y == nullptr || a.ntok < 1 || a.x_ld % 8) return FPR_E_ARG;
  const int64_t nslices = (a.ntok + 127) / 128;
  if ((nslices != 1 && a.splits != 1) || nslices > 65535) return FPR_E_ARG;
  if (a.y_cols < a.tiles * w8::TR || a.splits < 1 || a.splits > W8_SMAX || a.splits > a.nchunk) return FPR_E_ARG;
  if (a.splits > 1) {
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    if (a.splits * a.tiles > nsm || a.ws == nullptr || a.cnt == nullptr || a.ws_floats < a.splits * a.tiles * 128 * w8::YP ||
        a.cnt_n < a.tiles)
      return FPR_E_ARG;
  }
  if ((a.pos != nullptr || a.inv_pos != nullptr) && a.cs == nullptr) return FPR_E_ARG;
  if (a.pos != nullptr && a.inv_pos != nullptr) return FPR_E_ARG;
  if (a.kv != nullptr && (a.slot == nullptr || a.cache == nullptr)) return FPR_E_ARG;
  const RunRes r = run(a, false);
  switch (r.e) {
    case RE_OK: return FPR_OK;
    case RE_INIT: return FPR_E_INIT;
    case RE_TOKENS: return FPR_E_SHAPE;
    default: return FPR_E_CUDA;
  }
}

extern "C" __attribute__((visibility("default"))) int fpr_w8a16_wide_ok(int dev) { return g_wide_ok[dev & 63] ? 1 : 0; }   // (obs)
extern "C" __attribute__((visibility("default"))) int fpr_w8a16_wide_err(int dev) { return g_wide_err[dev & 63]; }
extern "C" __attribute__((visibility("default"))) long long fpr_w8a16_wide_launches(void) { return g_wide_launches.load(); }
extern "C" __attribute__((visibility("default"))) long long fpr_w8a16_wide_fallbacks(void) { return g_wide_fallbacks.load(); }
extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_w8a16_table(void) {
  static const FprFnEntry fns[] = {
      {"w8a16.init", c_init, nullptr, static_cast<int64_t>(sizeof(FprW8a16InitArgs))},
      {"w8a16.gemm", c_gemm, nullptr, static_cast<int64_t>(sizeof(FprW8a16GemmArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "w8a16", "0ccd41d2149fb4d2d4cb9e13c194ee13", 2, fns};
  return &t;
}

// ---- pybind (signature, checks, messages and behaviour unchanged; the launch is run() above)
// x [ntok, Kx] bf16 (row stride multiple of 8), packed [nchunk, tiles, CHUNK] uint8 (chunk-major), y [ntok, >= N] bf16.
void gemm(torch::Tensor x, torch::Tensor packed, torch::Tensor y, int64_t rows_per_group, int64_t kx_per_group, int64_t splits,
          c10::optional<torch::Tensor> pos, c10::optional<torch::Tensor> cs, c10::optional<torch::Tensor> kv,
          c10::optional<torch::Tensor> slot, c10::optional<torch::Tensor> cache, int64_t page_stride, int64_t shift,
          c10::optional<torch::Tensor> inv_pos, c10::optional<torch::Tensor> ws, c10::optional<torch::Tensor> cnt,
          c10::optional<torch::Tensor> prof) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x");
  TORCH_CHECK(packed.dtype() == torch::kUInt8 && packed.dim() == 3 && packed.size(2) == w8::CHUNK && packed.is_contiguous(), "packed");
  TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.dim() == 2 && y.stride(1) == 1 && y.size(0) == x.size(0), "y");
  const int ntok = static_cast<int>(x.size(0));
  TORCH_CHECK(ntok >= 1, "ntok must be >= 1");
  const int nslices = (ntok + 127) / 128;  // > 1: one launch, grid z = 128-row token slices (splits == 1 only)
  TORCH_CHECK(nslices == 1 || splits == 1, "more than 128 token rows needs splits == 1");
  TORCH_CHECK(nslices <= 65535, "too many token rows");
  const int tiles = static_cast<int>(packed.size(1));
  TORCH_CHECK(y.size(1) >= static_cast<long long>(tiles) * w8::TR, "y too narrow");
  TORCH_CHECK(splits >= 1 && splits <= W8_SMAX && splits <= packed.size(0), "splits");
  FprW8a16GemmArgs a{};
  a.call.stream = at::cuda::getCurrentCUDAStream();
  a.call.device = x.get_device();
  a.call.rank = 0;
  a.x = x.data_ptr();
  a.ntok = x.size(0);
  a.kx = x.size(1);
  a.x_ld = x.stride(0);
  a.packed = packed.data_ptr<uint8_t>();
  a.nchunk = packed.size(0);
  a.tiles = packed.size(1);
  a.y = y.data_ptr();
  a.ldy = y.stride(0);
  a.y_cols = y.size(1);
  a.rows_per_group = rows_per_group;
  a.kx_per_group = kx_per_group;
  a.splits = splits;
  if (prof.has_value()) a.prof = prof->data_ptr<int64_t>();
  if (splits > 1) {
    // split-K: the CTAs of a tile meet at a counter barrier, so the whole grid must be resident (1 CTA per SM)
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    TORCH_CHECK(splits * packed.size(1) <= nsm, "split-K grid must fit on the SMs (one wave)");
    TORCH_CHECK(ws.has_value() && cnt.has_value(), "split-K needs ws / cnt");
    TORCH_CHECK(ws->dtype() == torch::kFloat32 && ws->numel() >= splits * packed.size(1) * 128 * w8::YP, "ws too small");
    TORCH_CHECK(cnt->dtype() == torch::kInt32 && cnt->numel() >= packed.size(1), "cnt");
    a.ws = ws->data_ptr<float>();
    a.ws_floats = ws->numel();
    a.cnt = cnt->data_ptr<int>();
    a.cnt_n = cnt->numel();
  }
  if (pos.has_value()) {
    TORCH_CHECK(cs.has_value() && cs->dtype() == torch::kFloat32 && cs->size(1) == 64, "cos_sin");
    a.pos = pos->data_ptr<int64_t>();
    a.cs = cs->data_ptr<float>();
  }
  if (inv_pos.has_value()) {
    TORCH_CHECK(cs.has_value() && cs->dtype() == torch::kFloat32 && cs->size(1) == 64, "cos_sin");
    TORCH_CHECK(!pos.has_value(), "RoPE of y and inverse RoPE of x are exclusive");
    a.inv_pos = inv_pos->data_ptr<int64_t>();
    a.cs = cs->data_ptr<float>();
  }
  if (kv.has_value()) {
    TORCH_CHECK(kv->dtype() == torch::kBFloat16 && kv->is_contiguous() && kv->size(0) == ntok && kv->size(1) == 512, "kv");
    TORCH_CHECK(slot.has_value() && cache.has_value(), "slot / cache");
    a.kv = kv->data_ptr();
    a.slot = slot->data_ptr<int64_t>();
    a.cache = cache->data_ptr<uint8_t>();
    a.page_stride = page_stride;
    a.shift = shift;
  }
  const RunRes r = run(a, true);
  switch (r.e) {
    case RE_OK: return;
    case RE_TMAP: TORCH_CHECK(false, "cuTensorMapEncodeTiled failed: ", r.detail);
    case RE_SMEM: TORCH_CHECK(false, "smem attr");
    case RE_LAUNCH_EX: TORCH_CHECK(false, "w8a16 launch (ex)");
    case RE_LAUNCH: TORCH_CHECK(false, "w8a16 launch: ", cudaGetErrorString(static_cast<cudaError_t>(r.detail)));
    default: TORCH_CHECK(false, "token count");
  }
}

int64_t chunk_bytes() { return w8::CHUNK; }

// prime(devs): every instantiation's SMEM attribute on the given devices, so no first-use module load / attribute call falls inside
// a CUDA graph capture. w8a16.py build(prime_devs=...) calls it right after loading (the engine passes its devices). Returns the
// number of devices primed (0 = no device visible, e.g. a login-node build); raises if the served path cannot be set.
int64_t prime(std::vector<int64_t> devs) {
  int n = 0, cur = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
    (void)cudaGetLastError();
    return 0;
  }
  cudaGetDevice(&cur);
  if (devs.empty()) devs.push_back(cur);
  for (const int64_t d : devs) {
    TORCH_CHECK(d >= 0 && d < n, "w8a16 prime: no device ", d);
    cudaSetDevice(static_cast<int>(d));
    TORCH_CHECK(prime_dev(static_cast<int>(d)) == cudaSuccess, "w8a16 prime: device ", d);
  }
  cudaSetDevice(cur);
  return static_cast<int64_t>(devs.size());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, "fused W8A16 GEMM (exact fp8 weights, bf16 MMA, fp32 accumulation; optional RoPE + KV insert)");
  m.def("chunk_bytes", &chunk_bytes);
  m.def("prime", &prime, "set every kernel instantiation's SMEM attribute on the given devices (empty: current) -- no lazy load at first use",
        pybind11::arg("devs") = std::vector<int64_t>{});
  m.def("wide_ok", [](int64_t dev) { return g_wide_ok[dev & 63]; }, "(obs) wide init status of a device");
  m.def("wide_err", [](int64_t dev) { return g_wide_err[dev & 63]; }, "(obs) the cudaError of a failed wide init (0: ok / not tried)");
  m.def("wide_launches", []() { return g_wide_launches.load(); }, "(obs) host launches / captures on the wide kernel");
  m.def("wide_fallbacks", []() { return g_wide_fallbacks.load(); }, "(obs) wide-eligible calls served by the narrow kernel");
}
