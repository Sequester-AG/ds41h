// ds41h W4A8 grouped GEMM for the DeepSeek-V4.1 routed experts on Hopper (sm_90a).
//
// Weights stay in the raw checkpoint layout (TP-4 slice, see ds41h/moe.py):
//   w13 [E, 2I, H/2] u8 (e2m1 pairs, low nibble = even k; rows 0..I-1 = w1/gate, I..2I-1 = w3/up)
//   s13 [E, 2I, H/32] ue8m0,  w2 [E, H, I/2] u8,  s2 [E, H, I/32] ue8m0      (H=5120, I=576 at TP4)
//
// One persistent, warp-specialized kernel template serves both GEMMs (MODE 1: w13 + SwiGLU,
// MODE 2: w2). Orientation is "weights as the wgmma M dimension": a CTA tile is 128 weight rows x
// N routed rows ("tokens") of one expert, N = 16..BNMAX in steps of 16 chosen per tile (the last
// token tile of an expert is only as wide as needed, so padding is < 16 rows per expert instead of
// up to BN-1). 384 threads:
//   WG2       producer (setmaxnreg Regs::PRODUCER): fetches tiles from a global atomic counter
//             (dynamic scheduling: tiles have very different N), decodes them, publishes per-tile
//             metadata in a small smem ring (tile, epilogue scale tables, per-row reference
//             exponents), then cp.async-streams the FP4 tile (128 rows x 64 B), the scale bytes and
//             the gathered token rows (N x 128 B, 128B-swizzled) into an NSTAGE ring (full/empty
//             mbarriers).
//   WG0, WG1  consumers (setmaxnreg Regs::CONSUMER): each owns 64 weight rows. Per 128-wide K block
//             a thread reads one 16-byte FP4 word per row from smem and expands e2m1 -> e4m3 with
//             the per-32 ue8m0 scale folded in relative to a per-row reference exponent e_ref = the
//             max ue8m0 byte of the whole row (computed by the producer per tile; exact for
//             e_ref - e <= 14, the checkpoint's per-row spread is <= 4, see fp4_e4m3_lut), then
//             issues RS-wgmma m64nNk32 x4 (A = weights in registers, B = tokens from smem). Because
//             the reference is constant along K, the accumulator needs no per-block promotion:
//             WIN = 0 accumulates the whole K in the wgmma accumulator (FP22-ish tensor-core
//             accumulation), WIN = w adds the accumulator into fp32 registers every w K blocks
//             (FADD, like DeepGEMM's promotion, needs a second N/2 register array -> BNMAX <= 128).
//             The expansion of block k+1 overlaps the wgmma group of block k (wait_group 1).
// Activations are e4m3 with one dynamic scale per row (per token for GEMM1, applied in its epilogue;
// per routed row for GEMM2, applied by moe_combine); they are stored in "fragment order" (see
// moe_route.cu).
// MODE 1 epilogue: g, u (rows j and I+j are interleaved so each thread holds both), SwiGLU with
// clamp (gate <= limit, |up| <= limit; vLLM SiluAndMulWithClamp / model.py Expert), times the
// routing weight -> bf16 h [R, I] in expert order. MODE 2 epilogue: times the weight row scale -> bf16
// slot buffer row t*K+k ([T*K, H]); moe_combine sums the K rows of each token. Both epilogues
// transpose through smem with stmatrix and store 16-byte chunks.
//
// Tiles: per expert e with n_e routed rows, ceil(n_e/BNMAX) token tiles x NB weight-row blocks
// (NB = I/64 for MODE 1: 64 gate + 64 up rows; H/128 for MODE 2), ordered expert-major, then
// weight block, then token tile. All metadata is on the device (CUDA-graph safe, no host sync);
// the tile counter resets itself at the end of every launch.
#include "common.cuh"
#ifndef DS41H_DEVICE_ONLY
#include "host.cuh"
#endif
#include "sm90.cuh"

#include <cuda.h>

#include <type_traits>
#ifndef DS41H_DEVICE_ONLY
#include <cudaTypedefs.h>

#include <cstdio>
#include <cstdlib>
#endif

namespace {

constexpr int NTHREADS = 384;
constexpr int MAX_E = 1024;
constexpr int A_STAGE = 128 * 64;  // FP4 bytes per stage (128 rows x 128 K values)
constexpr int S_STAGE = 128 * 8;   // scale bytes per stage (8-byte aligned window per row)
constexpr int RING = 3;            // tile metadata ring depth (producer -> consumers; the two consumer WGs may be a tile apart)
// setmaxnreg moves registers inside the CTA's launch allocation (168 x 384 = 64512): the increase
// of the consumers must be covered by the producer's decrease, 2 * CONSUMER + PRODUCER <= 504
// (otherwise setmaxnreg.inc blocks forever), as CUTLASS's 232 / 40 split.
// 256-token tiles need ~230 consumer registers (128-register accumulator); 128-token tiles fit 216.
template <int BNMAX>
struct Regs {
  static constexpr int CONSUMER = BNMAX > 128 ? 232 : 216;
  static constexpr int PRODUCER = 3 * 168 - 2 * CONSUMER;  // 40 / 72
  static_assert(PRODUCER >= 24 && PRODUCER % 8 == 0, "setmaxnreg range");
};
constexpr int META_CTR = 8;        // counters in meta (zeroed by moe_permute): [0..1] GEMM1, [2..3] GEMM2
constexpr int NLOAD = 96;          // cp.async loader threads (producer warps 9..11)
// WG drift for the asynchronous-epilogue variants (WG 1 starts NSTAGE - 1 K blocks behind WG 0): measured
// neutral for GEMM1 and slightly negative for GEMM2 (k13 vs k12, 250 W) -> off
constexpr bool WG_DRIFT = false;
constexpr int ESCRATCH = 128 * 34 + 16;  // MODE 2 scale-row block of one tile (I <= 1088 -> rows <= 34 B)

// ------------------------------------------------------------------------------------------ PTX
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void reg_fence(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R));
}
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R));
}
__device__ __forceinline__ void consumer_bar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
// per-consumer-warpgroup barrier (ids 3, 4): the two consumer WGs run their epilogues independently,
// so one WG's epilogue overlaps the other WG's wgmmas (which then get the whole tensor core)
__device__ __forceinline__ void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" ::"r"(3 + wg) : "memory"); }
__device__ __forceinline__ void producer_bar() { asm volatile("bar.sync 2, 128;" ::: "memory"); }
// Make cp.async (generic proxy) smem writes observed through the full barrier visible to the
// wgmma operand reads (async proxy).
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// cp.async with zero fill: reads src_bytes (<= size) and zero-fills the rest of the destination.
__device__ __forceinline__ void cp_async16_zfill(void* dst, const void* src, unsigned src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async16_s(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async4_zfill(void* dst, const void* src, unsigned src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// stmatrix: 8x8 b16 matrices, transposed on the way to smem. Lane 8m + r gives the smem address of
// (stored) row r of matrix m; register m holds this thread's fragment (row lane/4, cols 2(lane%4)+{0,1}).
__device__ __forceinline__ void stsm_x4_trans(uint32_t a, unsigned r0, unsigned r1, unsigned r2, unsigned r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}
__device__ __forceinline__ void stsm_x2_trans(uint32_t a, unsigned r0, unsigned r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};" ::"r"(a), "r"(r0), "r"(r1) : "memory");
}

// wgmma shared-memory descriptor: K-major operand, rows of 128 bytes, 128B swizzle, tile base
// 1024-byte aligned. LBO unused (0), SBO = 1024 B between 8-row groups (DeepGEMM make_smem_desc /
// cute GmmaDescriptor). Advancing K by 32 e4m3 = +32 bytes = +2 in the address field.
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return ((uint64_t(saddr) & 0x3FFFFull) >> 4) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}

// 32-bit shared-window address variants (the consumer keeps every smem location as a u32 to save
// registers: it runs at the edge of the 216-register budget with a 128-register accumulator).
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ float2 lds_f2(uint32_t a) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ int lds_s32(uint32_t a) {
  int v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ int lds_u8(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a));
  return int(v);
}
// mbarrier wait with a watchdog: a phase that does not complete within ~20 s (a protocol bug, never
// legitimate) traps instead of spinning forever and holding the GPU.
__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .u64 t0, t1;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t0, %%globaltimer;\n"
      "LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t1, %%globaltimer;\n"
      " sub.u64 t1, t1, t0;\n"
      " setp.lt.u64 p, t1, 20000000000;\n"
      " @p bra.uni LAB_WAIT;\n"
      " trap;\n"
      "LAB_DONE:\n}" ::"r"(bar), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_s(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}

#ifdef DS41H_MOE_TRACE
// Debug timeline (build with -DDS41H_MOE_TRACE; read with the extension's `trace` function): per
// (CTA, consumer WG) up to TR_N records of 8 u64 (see the kernels for the fields).
constexpr int TR_N = 4096;
__device__ unsigned long long g_mtrace[160 * 4 * TR_N * 8];  // per CTA: consumer WG 0, 1, storer warp 0, 1
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#endif

// TMA / bulk-copy helpers (the CTA is its own cluster; shared::cta addresses are valid shared::cluster ones)
__device__ __forceinline__ void mbar_expect_tx_s(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void bulk_store(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

// Thread-block-cluster helpers (W4A16 multicast variants, CS > 1)
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa_rank(uint32_t addr, uint32_t rank) {  // shared::cta address -> shared::cluster address in CTA `rank`
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void st_remote_u32(uint32_t cluster_addr, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(cluster_addr), "r"(v) : "memory");
}
// wait with cluster-scope acquire (barriers completed by remote CTAs' release.cluster arrives)
// (watchdog as mbar_wait_s: trap after ~20 s instead of hanging the GPU on a protocol bug)
__device__ __forceinline__ void mbar_wait_cl(uint32_t bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .u64 t0, t1;\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t0, %%globaltimer;\n"
      "LAB_WAIT:\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t1, %%globaltimer;\n"
      " sub.u64 t1, t1, t0;\n"
      " setp.lt.u64 p, t1, 20000000000;\n"
      " @p bra.uni LAB_WAIT;\n"
      " trap;\n"
      "LAB_DONE:\n}" ::"r"(bar), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// TMA 2D load multicast to the CTAs of `mask` (same smem offsets / mbarrier offset in each)
__device__ __forceinline__ void tma_load_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar), "h"(mask)
      : "memory");
}

// ---------------------------------------------------------------------- RS wgmma (generated)
// D[64xN] (+)= A[64x32] (e4m3, registers) * B[Nx32]^T (e4m3, smem K-major SW128); scale_d = 0 overwrites D.
template <int N>
struct WgmmaRS;
template <>
struct WgmmaRS<16> {
  __device__ __forceinline__ static void mma(float (&d)[8], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7"
        "}, {%8, %9, %10, %11}, %12, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<32> {
  __device__ __forceinline__ static void mma(float (&d)[16], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
        "}, {%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<48> {
  __device__ __forceinline__ static void mma(float (&d)[24], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23"
        "}, {%24, %25, %26, %27}, %28, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<64> {
  __device__ __forceinline__ static void mma(float (&d)[32], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
        "}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<80> {
  __device__ __forceinline__ static void mma(float (&d)[40], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39"
        "}, {%40, %41, %42, %43}, %44, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<96> {
  __device__ __forceinline__ static void mma(float (&d)[48], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47"
        "}, {%48, %49, %50, %51}, %52, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<112> {
  __device__ __forceinline__ static void mma(float (&d)[56], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55"
        "}, {%56, %57, %58, %59}, %60, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<128> {
  __device__ __forceinline__ static void mma(float (&d)[64], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
        "}, {%64, %65, %66, %67}, %68, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<144> {
  __device__ __forceinline__ static void mma(float (&d)[72], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %77, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n144k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71"
        "}, {%72, %73, %74, %75}, %76, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<160> {
  __device__ __forceinline__ static void mma(float (&d)[80], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %85, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n160k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79"
        "}, {%80, %81, %82, %83}, %84, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<176> {
  __device__ __forceinline__ static void mma(float (&d)[88], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %93, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n176k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87"
        "}, {%88, %89, %90, %91}, %92, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<192> {
  __device__ __forceinline__ static void mma(float (&d)[96], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"
        "}, {%96, %97, %98, %99}, %100, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<208> {
  __device__ __forceinline__ static void mma(float (&d)[104], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %109, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n208k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103"
        "}, {%104, %105, %106, %107}, %108, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<224> {
  __device__ __forceinline__ static void mma(float (&d)[112], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %117, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n224k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111"
        "}, {%112, %113, %114, %115}, %116, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<240> {
  __device__ __forceinline__ static void mma(float (&d)[120], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %125, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n240k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119"
        "}, {%120, %121, %122, %123}, %124, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS<256> {
  __device__ __forceinline__ static void mma(float (&d)[128], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
        "}, {%128, %129, %130, %131}, %132, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

// ---------------------------------------------------------------------- RS wgmma, bf16 (generated)
// D[64xN] (+)= A[64x16] (bf16, registers) * B[Nx16]^T (bf16, smem K-major SW128, no transpose); scale_d = 0 overwrites D.
template <int N>
struct WgmmaRS16;
template <>
struct WgmmaRS16<16> {
  __device__ __forceinline__ static void mma(float (&d)[8], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7"
        "}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<32> {
  __device__ __forceinline__ static void mma(float (&d)[16], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
        "}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<48> {
  __device__ __forceinline__ static void mma(float (&d)[24], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23"
        "}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<64> {
  __device__ __forceinline__ static void mma(float (&d)[32], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
        "}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<80> {
  __device__ __forceinline__ static void mma(float (&d)[40], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39"
        "}, {%40, %41, %42, %43}, %44, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<96> {
  __device__ __forceinline__ static void mma(float (&d)[48], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47"
        "}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<112> {
  __device__ __forceinline__ static void mma(float (&d)[56], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55"
        "}, {%56, %57, %58, %59}, %60, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<128> {
  __device__ __forceinline__ static void mma(float (&d)[64], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
        "}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<144> {
  __device__ __forceinline__ static void mma(float (&d)[72], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %77, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n144k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71"
        "}, {%72, %73, %74, %75}, %76, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<160> {
  __device__ __forceinline__ static void mma(float (&d)[80], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %85, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n160k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79"
        "}, {%80, %81, %82, %83}, %84, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<176> {
  __device__ __forceinline__ static void mma(float (&d)[88], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %93, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n176k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87"
        "}, {%88, %89, %90, %91}, %92, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<192> {
  __device__ __forceinline__ static void mma(float (&d)[96], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"
        "}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<208> {
  __device__ __forceinline__ static void mma(float (&d)[104], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %109, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n208k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103"
        "}, {%104, %105, %106, %107}, %108, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<224> {
  __device__ __forceinline__ static void mma(float (&d)[112], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %117, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n224k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111"
        "}, {%112, %113, %114, %115}, %116, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<240> {
  __device__ __forceinline__ static void mma(float (&d)[120], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %125, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n240k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119"
        "}, {%120, %121, %122, %123}, %124, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};
template <>
struct WgmmaRS16<256> {
  __device__ __forceinline__ static void mma(float (&d)[128], const unsigned (&a)[4], uint64_t desc, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
        "}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
  }
};

// ------------------------------------------------- FP4 -> bf16, exact, scale folded (W4A16 path, LD 4)
// e2m1 (s e1 e0 m0) placed at bf16 bits (s << 15) | (e1 e0 m0) << 6 is value * 2^-126 exactly for all 16 codes
// (0.5 -> the bf16 subnormal 2^-127, 0 -> +-0). One mul.rn.bf16x2 by 2^(S-1) (S = ue8m0 byte; S <= 128 keeps the
// multiplier finite, checked on the host side by ds41h.moe) gives value * 2^(S-127) exactly: the product of a
// power of two and a 2-significant-bit value stays exact (bf16 has fp32's exponent range; subnormals are not
// flushed by the bf16 mul). So every dequantized weight equals the reference dequant bit for bit.
//
// K order of the W4A16 path. Lane quad t owns the physical range [16t, 16t+16) of each 64-wide K block (FP4 bytes
// [8t, 8t+8) of a row = words j = 0, 1); both words are rotated right by 4t bits. k16 step s = 2j + h of the block
// uses word j: fragment register R0 (cols 2t, 2t+1) = rotated nibbles (2h, 2h+4), R2 (cols 2t+8, 2t+9) = rotated
// nibbles (2h+1, 2h+5) of row g (R1 / R3: row g+8). Hence logical column L of a 64-block (what the B operand
// holds there) is physical column a16_phys(L) = 16t + 8j + ((n + t) & 7) with s = L >> 4, j = s >> 1, t =
// (L >> 1) & 3, n = 2 (s & 1) + ((L >> 3) & 1) + 4 (L & 1). act_perm16 (moe_route.cu) writes x and GEMM1 writes h in
// this logical order (GEMM1's accumulator row for h column c reads weight row a16_phys(c)); the dot product over a
// block is permutation invariant, so this is exact. The rotation makes the 8 weight rows a GEMM1 warp reads with
// one LDS (a16_phys of 8 consecutive c) fall into 4 distinct 32-byte bank groups (conflict-free).
template <int P>  // nibble pair (P, P+4) of a rotated word -> raw bf16x2 (value * 2^-126)
__device__ __forceinline__ unsigned fp4_pair_bf16_raw(unsigned w) {
  constexpr int MS = 6 - 4 * P, SS = 12 - 4 * P;
  const unsigned mag = (MS >= 0 ? (w << MS) : (w >> (-MS))) & 0x01C001C0u;
  const unsigned sgn = (w << SS) & 0x80008000u;
  return mag | sgn;
}
__device__ __forceinline__ unsigned bf16x2_mul_rn(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// {2^(S-1), 2^(S-1)} as bf16x2 for a ue8m0 byte S in [0, 128]
__device__ __forceinline__ unsigned fp4_bf16_mult(int S) { return unsigned(S + 126) * 0x00800080u; }
__host__ __device__ __forceinline__ constexpr int a16_phys(int L) {
  return 16 * ((L >> 1) & 3) + 8 * (L >> 5) + ((2 * ((L >> 4) & 1) + ((L >> 3) & 1) + 4 * (L & 1) + ((L >> 1) & 3)) & 7);
}
// accurate expf (libdevice) for the stock SwiGLU numerics: the extension is built with --use_fast_math, under which
// expf would become the approximate __expf
extern "C" __device__ float __nv_expf(float);

// ---------------------------------------------------------------------------------- arguments
struct GemmArgs {
  // TMA loader (LD = 1) only: B = activation rows (routed-row order) box {128 B, 64 rows} SW128;
  // W = FP4 weights box {64 B, 64 (MODE 1) / 128 (MODE 2) rows}; S = MODE 1 scales box {16 B, 64 rows}
  CUtensorMap tm_b, tm_w, tm_s;
  const uint8_t* w;        // expert weights (raw FP4, row-major [E * rows_per_expert, K/2])
  const uint8_t* s;        // ue8m0 scales [E * rows_per_expert, K/32]
  const uint8_t* act;      // e4m3 activations, fragment order: MODE 1 xq [T, K]; MODE 2 hq [R, K]
  const float* act_scale;  // MODE 1 xs [T]; MODE 2 hs [R]
  const int* off;          // [E+1] routed-row offsets per expert
  const int* perm_tok;     // [R] token of routed row
  const int* perm_slot;    // [R] t*K + k of routed row
  const float* perm_w;     // [R] routing weight of routed row
  int* ctr;                // [2] tile counter, finished-CTA counter (zero at launch, reset at exit)
  __nv_bfloat16* out;      // MODE 1 h [R, I]; MODE 2 slot [T*K, H]
  long w_estride;          // bytes per expert (weights)
  long s_estride;          // bytes per expert (scales)
  long s_total;            // bytes of the whole scale tensor
  int w_rstride;           // bytes per weight row (K/2)
  int s_rstride;           // bytes per scale row (K/32)
  int K;                   // reduction length (multiple of 64)
  int E;                   // experts
  int NB;                  // weight-row blocks per expert
  int I;                   // MODE 1: row offset of w3 (up) inside w13
  int out_stride;          // elements per output row
  float limit;             // MODE 1: swiglu limit
  int brows;               // LD 1: rows of a tm_b box (min(64, activation rows))
  // Hybrid remainder split (W4A16; 0 = off): with tsplit = S > 0 the LD 4 kernel skips each expert's
  // last partial token tile when it holds <= S rows (n % BNMAX in (0, S]) and moe_gemm16_kernel processes exactly those
  // remainders (rows [tsplit_bn * floor(n / tsplit_bn), n), one tile per expert): narrow tiles cost a full k16 step of
  // the 128-row kernel, the 256-row kernel streams twice the weight rows per step there.
  int tsplit;
  int tsplit_bn;
  // small16k unit order (0 = small16's (expert, chunk, weight block) with the weight block fastest):
  // 1 = the token chunks of one (expert, weight block) are consecutive units (chunk fastest) -> they run on neighbouring CTAs at
  // the same time and share the weight stream through L2 (hot experts with > 16 rows in multi-request decode)
  int s16_cf;
};

// token tiles of an expert with n routed rows in the LD 4 kernel (hybrid: remainders <= tsplit go to gemm16)
template <int BNMAX>
__device__ __forceinline__ int ld_ntiles(int n, int tsplit) {
  if (tsplit <= 0) return (n + BNMAX - 1) / BNMAX;
  const int r = n % BNMAX;
  return n / BNMAX + (r > tsplit ? 1 : 0);
}

// LD: 0 = cp.async gather loader; 1 = TMA loader (activations in routed-row order) + asynchronous epilogue;
// 2 = 1 with the bf16-reduction GEMM2 epilogue; 3 = 1 with one-row TMA boxes of the token rows (GEMM1);
// 4 = W4A16: bf16 activations (routed-row order, K-block logical order), FP4 expanded exactly to bf16, bf16
//     RS-wgmma m64nNk16, 64-wide K blocks; otherwise as 1 (GEMM2 writes unscaled bf16 slot rows).
// 6 = 4 for GEMM1 with the B rows gathered by the loader warp ("x-gather"): cp.async 16-byte chunks
//     with the 128B swizzle applied by hand, straight from x in token order (logical K order, act_log16) -- no
//     routed-row copy of x (act_perm16 wrote 6 x [T, H] per layer). Same products and wgmma order as LD 4.
constexpr bool ld_a16(int LD) { return LD == 4 || LD == 6; }
template <int MODE, int BNMAX, int WIN, int LD = 0>
struct Cfg {
  static_assert(BNMAX % 16 == 0 && BNMAX <= 256, "BNMAX");
  static_assert(WIN == 0 || BNMAX <= 128, "fp32 promotion needs a second N/2 register array");
  static_assert(!ld_a16(LD) || WIN == 0, "W4A16 accumulates in fp32 wgmma accumulators (no promotion window)");
  static_assert(LD != 6 || MODE == 1, "gathered B rows (LD 6): GEMM1 only");
  static constexpr bool A16 = ld_a16(LD);
  static constexpr int B_STAGE = BNMAX * 128;  // 128-byte activation rows: 128 e4m3 or 64 bf16 K values
  static constexpr int A_BYTES = A16 ? 128 * 32 : A_STAGE;  // FP4 bytes per stage (128 rows x K/2 of the block)
  // scales per stage: LD 0 = an 8-byte window per row (cp.async); LD 1 = MODE 1: the 16-byte group of 4
  // K blocks of the 128 rows (2 TMA boxes), MODE 2: the tile's whole 128 x I/32 scale block (one bulk copy);
  // LD 4: as LD 1 (MODE 1: the 16-byte group covers 8 blocks of 64)
#ifdef DS41H_MOE_G1_5STAGE  // W4A16 GEMM1: per-mode scale bytes (2 KB) + one staging buffer (its storer is ~97% idle) -> 5 stages
  static constexpr bool G1S5 = ld_a16(LD) && MODE == 1;
#else
  static constexpr bool G1S5 = false;
#endif
  static constexpr int S_BYTES = G1S5 ? 2048 : (LD ? 3072 : S_STAGE);
  static constexpr int STAGE = B_STAGE + A_BYTES + S_BYTES;  // multiple of 1024
  static constexpr int EPI_TOK = BNMAX < 128 ? BNMAX : 128;  // tokens per epilogue staging pass
  static constexpr int PITCH = MODE == 1 ? 144 : 272;        // 64 / 128 bf16 + 16 B pad (conflict-free stmatrix)
  static constexpr int EPI = EPI_TOK * PITCH;
  // LD >= 1: asynchronous epilogue -- the consumers stage a pass into one of two buffers per WG
  // (column halves of a shared [token][cols] tile) and producer warps 10 / 11 store it
  static constexpr int EPI_BUFS = (LD && !G1S5) ? 2 : 1;
  // ring slot: int hdr[4] (e, nb, pos0, nvalid<0 = end) | f32 tab0[BNMAX] | f32 tab1[BNMAX] | u8 erow[128] |
  // int rows[BNMAX] (B rows to gather; producer-only)
  static constexpr int SLOT_TAB0 = 16, SLOT_TAB1 = 16 + 4 * BNMAX, SLOT_EROW = 16 + 8 * BNMAX, SLOT_ROWS = SLOT_EROW + 128;
  static constexpr int SLOT = SLOT_ROWS + 4 * BNMAX;
  static constexpr int SCRATCH = MODE == 2 ? ESCRATCH : 0;  // scheduler-warp copy of a tile's scale rows (MODE 2)
  static constexpr int FIXED = EPI_BUFS * EPI + RING * SLOT + SCRATCH + 2 * (MAX_E + 1) * 4 + 512 + 1024;  // + s_off/s_tp, barriers, align slack
  static constexpr int BUDGET = 227 * 1024;
  static constexpr int NSTAGE = (BUDGET - FIXED) / STAGE > 8 ? 8 : (BUDGET - FIXED) / STAGE;
  static constexpr int SMEM = NSTAGE * STAGE + FIXED;
  static_assert(NSTAGE >= 3, "pipeline too shallow");
  // byte offset of the first pipeline barrier (full[0]) from the 1024-aligned smem base (the
  // consumers derive the stage / staging addresses from it to save registers)
  static constexpr int BAR_OFF = (NSTAGE * STAGE + EPI_BUFS * EPI + RING * SLOT + SCRATCH + 2 * (MAX_E + 1) * 4 + 7) / 8 * 8;
  // cluster variants (CS > 1), offsets from full[0], after sbars[0..16]: cempty[8] | tbar[RING] | tfree[RING] | int tidx[RING]
  static constexpr int CL_EMPTY = 8 * (2 * NSTAGE + 2 * RING) + 32 + 8 * 17;
  static constexpr int CL_TBAR = CL_EMPTY + 8 * 8, CL_TFREE = CL_TBAR + 8 * RING, CL_TIDX = CL_TFREE + 8 * RING;
  static_assert(CL_TIDX + 4 * RING <= 512, "barrier region");
  static_assert(STAGE % 1024 == 0, "stages must stay 1024-byte aligned for the 128B swizzle");
  static_assert(SLOT % 16 == 0 && EPI % 16 == 0, "alignment");
};

// Global weight row (inside the expert) of tile-local A row `lr` (0..127).
template <int MODE>
__device__ __forceinline__ int weight_row(int lr, int nb, int I) {
  if constexpr (MODE == 1)
    return lr < 64 ? 64 * nb + lr : I + 64 * nb + (lr - 64);
  else
    return 128 * nb + lr;
}

struct Consumer {
  // shared address of full[0]; empty[s] = full[s] + 8 * NSTAGE; stage 0 at bar - C::BAR_OFF, staging
  // at stage 0 + NSTAGE * STAGE (derived, not kept in registers)
  uint32_t bar;
  uint32_t slot;  // shared address of the current ring slot
  int stage;
  unsigned phase;
  int tid, sr0, sr1;
  // current tile
  int e, nb, pos0, nvalid, erow0, erow1;
  // asynchronous epilogue (LD >= 1): staging passes posted so far by this WG
  int epass;
#ifdef DS41H_MOE_TRACE
  long long twait;  // clock64 cycles spent in full-barrier waits of the current tile
#endif
};

// Asynchronous epilogue hand-off (LD >= 1). The staging barriers follow the pipeline / ring barriers:
// sfull[wg][b] = sbar + 8 (2 wg + b), sempty = sfull + 32, headers int[4] at sbar + 64 + 16 (2 wg + b),
// with sbar = full[0] + 8 (2 NSTAGE + 2 RING) + 32 (after s_bcast[8]); kept out of the Consumer
// registers (the BN 256 mainloop is at the edge of the register budget).
template <int NSTAGE>
__device__ __forceinline__ uint32_t epi_sbar(const Consumer& c) { return c.bar + 8 * (2 * NSTAGE + 2 * RING) + 32; }
// wait until staging buffer b of this WG is free (the first NB passes find their buffers free)
template <int NSTAGE, int NB = 2>
__device__ __forceinline__ uint32_t epi_acquire(const Consumer& c, int wg) {
  const int b = c.epass % NB;
  mbar_wait_s(epi_sbar<NSTAGE>(c) + 32 + 8 * (2 * wg + b), ((c.epass / NB) & 1) ^ 1);
  return uint32_t(b);
}
// publish the staged pass: header {ring slot, first token, tokens, flags (1 = last pass of the tile, 2 = end)}
template <int NSTAGE, int NB = 2>
__device__ __forceinline__ void epi_post(Consumer& c, int wg, int rslot, int c0, int np, int flags) {
  const int b = c.epass % NB;
  const uint32_t sbar = epi_sbar<NSTAGE>(c);
  if ((c.tid & 127) == 0) {
    int* h = reinterpret_cast<int*>(__cvta_shared_to_generic(sbar + 64 + 16 * (2 * wg + b)));
    h[0] = rslot, h[1] = c0, h[2] = np, h[3] = flags;
  }
  __syncwarp();
  if ((c.tid & 31) == 0) mbar_arrive_s(sbar + 8 * (2 * wg + b));
  ++c.epass;
}

// ------------------------------------------------------------------- consumer: epilogue pass
// One staging pass of up to EPI_TOK tokens starting at token P * EPI_TOK: scale (and SwiGLU), pack to
// bf16, stmatrix-transpose into [token][cols] smem, then 16-byte coalesced global stores.
template <int MODE, int BNMAX, int WIN, int LD, int P>
__device__ __forceinline__ void epi_pass(Consumer& c, const GemmArgs& a, const float (&v)[BNMAX / 2], int nch, bool last) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  constexpr int c0 = P * C::EPI_TOK;
  constexpr int NCH = C::EPI_TOK / 8;  // 8-token chunks of a full pass; nch (even) = valid ones of this pass
  const int lane = c.tid & 31, t = lane & 3, m8 = lane >> 3, r8 = lane & 7;
  const int wg = c.tid >> 7, wq = (c.tid >> 5) & 3, wtid = c.tid & 127;
  const float ws0 = fp4_ref_scale(c.erow0), ws1 = fp4_ref_scale(c.erow1);
  const uint32_t tab0 = c.slot + C::SLOT_TAB0 + 8 * t, tab1 = c.slot + C::SLOT_TAB1 + 8 * t;  // + 32 j: tokens 8j+2t, +1
  // Each consumer WG owns a disjoint column half of the staging tile. LD 0: the WG stores it itself;
  // LD >= 1: it is handed to the WG's storer warp (warp 10 + wg) through a double-buffered staging tile.
  const uint32_t epi0 = c.bar - C::BAR_OFF + C::NSTAGE * C::STAGE;
  uint32_t ebase = epi0;
  if constexpr (LD >= 1)
    ebase += epi_acquire<C::NSTAGE, C::EPI_BUFS>(c, wg) * C::EPI;
  else
    wg_bar(wg);  // this WG's staging columns free (its previous pass / tile stored)
  const int np = min(C::EPI_TOK, c.nvalid - c0);
  if constexpr (MODE == 1) {
#pragma unroll
    for (int i = 0; i < NCH; i += 2) {
      if (i >= nch) break;
      unsigned rr[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int j = c0 / 8 + i + q;
        float h[2];
        if constexpr (ld_a16(LD)) {
          // W4A16: stock vLLM clamped SiLU (humming/activation.py _CLAMPED_SILU_IMPL, alpha 1, beta 0) on the fp32
          // accumulators (stock rounds gate/up to bf16 first: this is at least as precise); routing weight NOT
          // applied here (stock applies it in the final weighted sum, moe_combine)
#pragma unroll
          for (int hv = 0; hv < 2; ++hv) {
            const float gt = fminf(v[4 * j + hv], a.limit);
            const float up = fmaxf(fminf(v[4 * j + 2 + hv], a.limit), -a.limit);
#ifdef DS41H_MOE_ABL_FASTSILU  // timing experiment only: approximate exp / division (not the stock numerics)
            h[hv] = gt / (1.f + __expf(-gt)) * up;
#else
            h[hv] = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
#endif
          }
        } else {
          const float2 sx = lds_f2(tab0 + 32 * j);
          const float2 rw = lds_f2(tab1 + 32 * j);
#pragma unroll
          for (int hv = 0; hv < 2; ++hv) {
            const float sxv = hv ? sx.y : sx.x;
            float gt = v[4 * j + hv] * (sxv * ws0), up = v[4 * j + 2 + hv] * (sxv * ws1);
            gt = fminf(gt, a.limit);
            up = fminf(fmaxf(up, -a.limit), a.limit);
            h[hv] = gt / (1.f + __expf(-gt)) * up * (hv ? rw.y : rw.x);
          }
        }
        rr[q] = pack_bf16(h[0], h[1]);
      }
      const int tok = 8 * (i + (m8 & 1)) + r8;
      stsm_x2_trans(ebase + tok * C::PITCH + (32 * wg + 8 * wq) * 2, rr[0], rr[1]);
    }
    if constexpr (LD >= 1) {
      epi_post<C::NSTAGE, C::EPI_BUFS>(c, wg, int((c.slot - (epi0 + C::EPI_BUFS * C::EPI)) / C::SLOT), c0, np, last ? 1 : 0);
      return;
    }
    wg_bar(wg);
    for (int idx = wtid; idx < np * 4; idx += 128) {  // this WG's 32 columns = 4 x 16 B per token
      const int cc = idx >> 2, chn = 4 * wg + (idx & 3);
      const uint4 val = lds128(epi0 + cc * C::PITCH + chn * 16);
      stg128(a.out + long(c.pos0 + c0 + cc) * a.out_stride + 64 * c.nb + chn * 8, val);
    }
  } else {
#pragma unroll
    for (int i = 0; i < NCH; i += 2) {
      if (i >= nch) break;
      // per-routed-row activation scales: applied here for the reduction epilogue (LD 2), otherwise by moe_combine
      const int j0 = c0 / 8 + i, j1 = j0 + 1;
      float2 h0 = make_float2(1.f, 1.f), h1 = make_float2(1.f, 1.f);
      if constexpr (LD == 2) h0 = lds_f2(tab0 + 32 * j0), h1 = lds_f2(tab0 + 32 * j1);
      unsigned r0, r1, r2, r3;
      if constexpr (ld_a16(LD)) {  // W4A16: y unscaled (exact weights); moe_combine applies the routing weights
        r0 = pack_bf16(v[4 * j0 + 0], v[4 * j0 + 1]);
        r1 = pack_bf16(v[4 * j0 + 2], v[4 * j0 + 3]);
        r2 = pack_bf16(v[4 * j1 + 0], v[4 * j1 + 1]);
        r3 = pack_bf16(v[4 * j1 + 2], v[4 * j1 + 3]);
      } else {
        r0 = pack_bf16(v[4 * j0 + 0] * (ws0 * h0.x), v[4 * j0 + 1] * (ws0 * h0.y));
        r1 = pack_bf16(v[4 * j0 + 2] * (ws1 * h0.x), v[4 * j0 + 3] * (ws1 * h0.y));
        r2 = pack_bf16(v[4 * j1 + 0] * (ws0 * h1.x), v[4 * j1 + 1] * (ws0 * h1.y));
        r3 = pack_bf16(v[4 * j1 + 2] * (ws1 * h1.x), v[4 * j1 + 3] * (ws1 * h1.y));
      }
      const int tok = 8 * (i + (m8 >> 1)) + r8;
      stsm_x4_trans(ebase + tok * C::PITCH + (64 * wg + 16 * wq + 8 * (m8 & 1)) * 2, r0, r1, r2, r3);
    }
    if constexpr (LD >= 1) {
      epi_post<C::NSTAGE, C::EPI_BUFS>(c, wg, int((c.slot - (epi0 + C::EPI_BUFS * C::EPI)) / C::SLOT), c0, np, last ? 1 : 0);
      return;
    }
    wg_bar(wg);
    for (int idx = wtid; idx < np * 8; idx += 128) {  // this WG's 64 columns = 8 x 16 B per token
      const int cc = idx >> 3, chn = 8 * wg + (idx & 7);
      const uint4 val = lds128(epi0 + cc * C::PITCH + chn * 16);
      const long row = lds_s32(c.slot + C::SLOT_TAB1 + 4 * (c0 + cc));
      if constexpr (LD == 2)
        asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(a.out + row * a.out_stride + 128 * c.nb + chn * 8), "r"(val.x),
                     "r"(val.y), "r"(val.z), "r"(val.w)
                     : "memory");
      else
        stg128(a.out + row * a.out_stride + 128 * c.nb + chn * 8, val);
    }
  }
}

// ---------------------------------------------------------------------------- consumer: one tile
// Software pipeline at k32-step granularity: wgmma step s (own commit group) runs while the FP4
// word of step s+1 is expanded into the other half of a 2-step register buffer (wait_group 1
// before overwriting it). At the last step of a block the next block's FP4 words / scales are
// read from the next stage and its step 0 expanded. Registers: acc N/2 + 8 (steps) + 8 (FP4 words).
template <int MODE, int BNMAX, int WIN, int LD, int N>
__device__ __forceinline__ void run_mainloop(Consumer& c, const GemmArgs& a, float (&acc_all)[BNMAX / 2], float (&fin_all)[WIN ? BNMAX / 2 : 1]) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  constexpr int NSTAGE = C::NSTAGE;
  // All N variants share the same accumulator registers (ptxas otherwise serializes the wgmmas).
  float(&acc)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc_all[0]);
  float(&fin)[WIN ? N / 2 : 1] = *reinterpret_cast<float(*)[WIN ? N / 2 : 1]>(&fin_all[0]);
  if constexpr (WIN) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) fin[i] = 0.f;
  }
  const int nkb = (a.K + 127) / 128;  // last block may be 64 wide (MODE 2 only)
  const int nfull = a.K / 128;
  const int t = c.tid & 3;
  // byte offset of this row's block scales inside the 8-byte staging window (constant per tile)
  const int soff0 = int((long(c.e) * a.s_estride + long(weight_row<MODE>(c.sr0, c.nb, a.I)) * a.s_rstride) & 3);
  const int soff1 = int((long(c.e) * a.s_estride + long(weight_row<MODE>(c.sr1, c.nb, a.I)) * a.s_rstride) & 3);

  unsigned v0[4], v1[4];         // FP4 words of the current block (rows sr0, sr1), one per k32 step
  unsigned l00, l01, l10, l11;   // e2m1 -> e4m3 LUTs of the current block (scale folded in)
  unsigned Ab[2][4];             // A fragments of two consecutive k32 steps

  // FP4 words + scales of block kb from stage c.stage (after the full wait + proxy fence)
  auto load_block = [&](int kb) {
    const uint32_t sa = c.bar - C::BAR_OFF + c.stage * C::STAGE + C::B_STAGE, ss = sa + A_STAGE;
    const bool fullblk = MODE == 1 || (kb + 1) * 128 <= a.K;  // MODE 1: K % 128 == 0 (host check)
    if (fullblk) {
      const uint4 q0 = lds128(sa + c.sr0 * 64 + 16 * t), q1 = lds128(sa + c.sr1 * 64 + 16 * t);
      v0[0] = q0.x, v0[1] = q0.y, v0[2] = q0.z, v0[3] = q0.w;
      v1[0] = q1.x, v1[1] = q1.y, v1[2] = q1.z, v1[3] = q1.w;
    } else {  // 64-wide tail: thread t owns k [16t, 16t+16) = FP4 bytes [8t, 8t+8)
      const uint2 q0 = lds64(sa + c.sr0 * 64 + 8 * t), q1 = lds64(sa + c.sr1 * 64 + 8 * t);
      v0[0] = q0.x, v0[1] = q0.y, v0[2] = 0u, v0[3] = 0u;
      v1[0] = q1.x, v1[1] = q1.y, v1[2] = 0u, v1[3] = 0u;
    }
    const int sub = fullblk ? t : (t >> 1);  // 32-wide scale block of this thread's K range
    int e0, e1;
    if constexpr (LD == 0) {
      e0 = lds_u8(ss + c.sr0 * 8 + soff0 + sub), e1 = lds_u8(ss + c.sr1 * 8 + soff1 + sub);
    } else if constexpr (MODE == 1) {
      e0 = lds_u8(ss + c.sr0 * 16 + 4 * (kb & 3) + sub), e1 = lds_u8(ss + c.sr1 * 16 + 4 * (kb & 3) + sub);
    } else {
      e0 = lds_u8(ss + c.sr0 * a.s_rstride + 4 * kb + sub), e1 = lds_u8(ss + c.sr1 * a.s_rstride + 4 * kb + sub);
    }
    fp4_e4m3_lut(e0 - c.erow0, l00, l01);
    fp4_e4m3_lut(e1 - c.erow1, l10, l11);
  };
  // fragment regs: a0 = (row g, k 4t..), a1 = (row g+8, k 4t..), a2 = (row g, k 16+4t..), a3 = (row g+8, k 16+4t..)
  auto expand = [&](int s, unsigned (&A)[4]) {
    fp4x8_to_e4m3_lut(v0[s], l00, l01, A[0], A[2]);
    fp4x8_to_e4m3_lut(v1[s], l10, l11, A[1], A[3]);
  };
  // one K block of NK k32 steps (4, or 2 for a 64-wide tail)
  auto block = [&](auto nk_c, int kb) {
    constexpr int NK = decltype(nk_c)::value;
    const uint64_t desc = desc_sw128(c.bar - C::BAR_OFF + c.stage * C::STAGE);
    const bool first = WIN ? (kb % WIN == 0) : (kb == 0);
#pragma unroll
    for (int s = 0; s < NK; ++s) {
#pragma unroll
      for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
      wgmma_fence();
      WgmmaRS<N>::mma(acc, Ab[s & 1], desc + 2 * s, (s > 0 || !first) ? 1 : 0);
      wgmma_commit();
      wgmma_wait<1>();  // previous step retired: Ab[(s+1)&1] free (and, at s == 0, block kb-1's stage)
#pragma unroll
      for (int j = 0; j < 4; ++j) reg_fence(Ab[(s + 1) & 1][j]);
      if (s == 0 && kb > 0 && (c.tid & 31) == 0) mbar_arrive_s(c.bar + 8 * (NSTAGE + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));
      if constexpr (LD >= 1 && WG_DRIFT) {  // WG drift: WG 0 lets WG 1 start once it is NSTAGE - 1 K blocks into its first tile
        if (s == 0 && kb == min(NSTAGE - 1, nkb - 1) && c.epass == 0 && c.tid < 128 && (c.tid & 31) == 0)
          mbar_arrive_s(epi_sbar<NSTAGE>(c) + 128);
      }
      if (s + 1 < NK) {
        expand(s + 1, Ab[(s + 1) & 1]);
      } else {
        if (++c.stage == NSTAGE) c.stage = 0, c.phase ^= 1;
        if (kb + 1 < nkb) {
#ifdef DS41H_MOE_TRACE
          const long long tw0 = clock64();
#endif
          mbar_wait_s(c.bar + 8 * c.stage, c.phase);
#ifdef DS41H_MOE_TRACE
          c.twait += clock64() - tw0;
#endif
          if constexpr (LD == 0) fence_proxy_async();  // cp.async (generic proxy) -> wgmma reads (async proxy)
          load_block(kb + 1);
          expand(0, Ab[0]);
        }
      }
    }
    if constexpr (WIN != 0) {
      if ((kb + 1) % WIN == 0 || kb + 1 == nkb) {
#pragma unroll
        for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
        wgmma_wait<0>();
#pragma unroll
        for (int i = 0; i < N / 2; ++i) fin[i] += acc[i];
      }
    }
  };

  {
    mbar_wait_s(c.bar + 8 * c.stage, c.phase);
    if constexpr (LD == 0) fence_proxy_async();
    load_block(0);
    expand(0, Ab[0]);
    const auto k4 = std::integral_constant<int, 4>{};
    for (int kb = 0; kb < nfull; ++kb) block(k4, kb);
    if constexpr (MODE == 2) {
      if (nfull < nkb) block(std::integral_constant<int, 2>{}, nfull);  // 64-wide tail block
    }
#pragma unroll
    for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
    if ((c.tid & 31) == 0) mbar_arrive_s(c.bar + 8 * (NSTAGE + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));  // last block's stage
  }

}

// W4A16 mainloop (LD 4): per 64-wide K block one 8-byte FP4 load per row (rows sr0, sr1), one scale byte per row,
// 4 k16 steps of bf16 RS-wgmma; the bf16 expansion of step s+1 overlaps the wgmma of step s (wait_group 1),
// the next block's words / scales are read right after its full barrier at the last step. fp32 accumulation
// over the whole K in the wgmma accumulators (bf16 wgmma accumulates in full fp32).
// A16_DEPTH = wgmmas a warpgroup keeps in flight (1: expand step s+1 while step s runs; 3: up to 3 queued steps,
// 4 fragment buffers indexed by the step inside the block)
#ifndef A16_DEPTH
#define A16_DEPTH 1
#endif
template <int MODE, int BNMAX, int N, int CS = 1, bool GATHER = false>
__device__ __forceinline__ void run_mainloop16(Consumer& c, const GemmArgs& a, float (&acc_all)[BNMAX / 2]) {
  using C = Cfg<MODE, BNMAX, 0, 4>;  // (LD 6 has the LD 4 layout)
  // stage release: local empty (this CTA's weights / scales) and, CS > 1, the leader's cluster-empty (multicast B)
  auto release = [&](int st) {
    if ((c.tid & 31) == 0) {
      mbar_arrive_s(c.bar + 8 * (C::NSTAGE + st));
      if constexpr (CS > 1) mbar_arrive_remote(mapa_rank(c.bar + C::CL_EMPTY + 8 * st, 0));
    }
  };
  constexpr int D = A16_DEPTH;
  static_assert(D == 1 || D == 3, "A16_DEPTH");
  constexpr int NSTAGE = C::NSTAGE;
  float(&acc)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc_all[0]);
  const int nkb = a.K / 64;
  const int t = c.tid & 3;
  const unsigned rot = 4u * unsigned(t);
  // per-row byte offsets inside a stage: FP4 words (A rows of 32 bytes) and the scale byte of this thread's k32 block
  const uint32_t aoff0 = c.sr0 * 32 + 8 * t, aoff1 = c.sr1 * 32 + 8 * t;
  const uint32_t soff0 = MODE == 1 ? c.sr0 * 16 + (t >> 1) : c.sr0 * a.s_rstride + (t >> 1);
  const uint32_t soff1 = MODE == 1 ? c.sr1 * 16 + (t >> 1) : c.sr1 * a.s_rstride + (t >> 1);
  unsigned w0[2], w1[2];  // rotated FP4 words j = 0, 1 of rows sr0 / sr1
  unsigned m0, m1;        // bf16x2 scale multipliers of rows sr0 / sr1
  unsigned Ab[D + 1][4];  // A fragments of D + 1 consecutive k16 steps

  auto load_block = [&](int kb) {
    const uint32_t sa = c.bar - C::BAR_OFF + c.stage * C::STAGE + C::B_STAGE, ss = sa + C::A_BYTES;
    const uint2 q0 = lds64(sa + aoff0), q1 = lds64(sa + aoff1);
    w0[0] = __funnelshift_r(q0.x, q0.x, rot), w0[1] = __funnelshift_r(q0.y, q0.y, rot);
    w1[0] = __funnelshift_r(q1.x, q1.x, rot), w1[1] = __funnelshift_r(q1.y, q1.y, rot);
    const uint32_t kofs = MODE == 1 ? 2 * (kb & 7) : 2 * kb;
    m0 = fp4_bf16_mult(lds_u8(ss + soff0 + kofs));
    m1 = fp4_bf16_mult(lds_u8(ss + soff1 + kofs));
  };
  auto expand = [&](int s, unsigned (&A)[4]) {
    const unsigned x0 = w0[s >> 1], x1 = w1[s >> 1];
#ifdef DS41H_MOE_ABL_NOCVT  // timing experiment only (wrong numerics): raw FP4 words as bf16 fragments
    A[0] = x0 ^ m0, A[2] = x0 + m0, A[1] = x1 ^ m1, A[3] = x1 + m1;
    return;
#endif
    if ((s & 1) == 0) {
      A[0] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x0), m0), A[2] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x0), m0);
      A[1] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x1), m1), A[3] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x1), m1);
    } else {
      A[0] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x0), m0), A[2] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x0), m0);
      A[1] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x1), m1), A[3] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x1), m1);
    }
  };
  auto block = [&](int kb) {
    const uint64_t desc = desc_sw128(c.bar - C::BAR_OFF + c.stage * C::STAGE);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
      for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
      wgmma_fence();
#ifndef DS41H_MOE_ABL_NOMMA  // timing experiment only: no tensor-core work
      WgmmaRS16<N>::mma(acc, Ab[s % (D + 1)], desc + 2 * s, (s > 0 || kb > 0) ? 1 : 0);
#else
      if (Ab[s % (D + 1)][0] == 0x12345u && Ab[s % (D + 1)][3] == 0x777u) acc[0] += 1.f;
#endif
      wgmma_commit();
      wgmma_wait<D>();  // steps <= this one - D retired: Ab[(s+1) % (D+1)] free (and, at s == D-1, block kb-1's stage)
#pragma unroll
      for (int j = 0; j < 4; ++j) reg_fence(Ab[(s + 1) % (D + 1)][j]);
      if (s == D - 1 && kb > 0) release(c.stage == 0 ? NSTAGE - 1 : c.stage - 1);
      if (s < 3) {
        expand(s + 1, Ab[(s + 1) % (D + 1)]);
      } else {
        if (++c.stage == NSTAGE) c.stage = 0, c.phase ^= 1;
        if (kb + 1 < nkb) {
#ifdef DS41H_MOE_TRACE
          const long long tw0 = clock64();
#endif
          mbar_wait_s(c.bar + 8 * c.stage, c.phase);
#ifdef DS41H_MOE_TRACE
          c.twait += clock64() - tw0;
#endif
          if constexpr (GATHER) fence_proxy_async();  // cp.async (generic proxy) B rows -> wgmma reads (async proxy)
          load_block(kb + 1);
          expand(0, Ab[0]);
        }
      }
    }
  };

  mbar_wait_s(c.bar + 8 * c.stage, c.phase);
  if constexpr (GATHER) fence_proxy_async();
  load_block(0);
  expand(0, Ab[0]);
  for (int kb = 0; kb < nkb; ++kb) block(kb);
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc[i]);
  release(c.stage == 0 ? NSTAGE - 1 : c.stage - 1);  // last block's stage
}

template <int MODE, int BNMAX, int WIN, int LD, int CS = 1, int NS = BNMAX / 16>
__device__ __forceinline__ void dispatch_tile(int nsel, Consumer& c, const GemmArgs& a, float (&acc)[BNMAX / 2],
                                              float (&fin)[WIN ? BNMAX / 2 : 1]) {
  if constexpr (NS >= 1) {
#ifdef ONLY_ONE
    run_mainloop<MODE, BNMAX, WIN, LD, BNMAX>(c, a, acc, fin);
    return;
#endif
    if (nsel == NS) {
      if constexpr (ld_a16(LD))
        run_mainloop16<MODE, BNMAX, 16 * NS, CS, LD == 6>(c, a, acc);
      else
        run_mainloop<MODE, BNMAX, WIN, LD, 16 * NS>(c, a, acc, fin);
      return;
    }
    dispatch_tile<MODE, BNMAX, WIN, LD, CS, NS - 1>(nsel, c, a, acc, fin);
  }
}

// Storer warp of the asynchronous epilogue (LD >= 1): global stores (MODE 1: h rows; MODE 2: slot rows or,
// LD 2, bf16 reductions into out) of the passes staged by consumer WG w, overlapping that WG's next
// mainloop. Not inlined: inside the kernel body its registers made ptxas serialize the consumers'
// wgmma pipeline (C7512) for BNMAX 256.
template <int MODE, int BNMAX, int WIN, int LD>
__device__ __noinline__ void storer_loop(const GemmArgs& a, int w, int lane, uint32_t sb0, uint32_t epi_s, uint32_t ring_s) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  for (int pass = 0;; ++pass) {
    const int b = pass % C::EPI_BUFS;
#ifdef DS41H_MOE_TRACE
    const unsigned long long ts0 = gtimer();
#endif
    mbar_wait_s(sb0 + 8 * (2 * w + b), (pass / C::EPI_BUFS) & 1);
#ifdef DS41H_MOE_TRACE
    const unsigned long long ts1 = gtimer();
#endif
    const uint32_t h = sb0 + 64 + 16 * (2 * w + b);
    const int rs = lds_s32(h), c0 = lds_s32(h + 4), np = lds_s32(h + 8), flags = lds_s32(h + 12);
    if (flags & 2) break;
    const uint32_t slot = ring_s + rs * C::SLOT;
    const int nb = lds_s32(slot + 4);
    const uint32_t buf = epi_s + b * C::EPI;
    // (loads of 4 iterations are issued before their stores: one shared-memory round trip per 4 stores)
    if constexpr (MODE == 1) {
      const int pos0 = lds_s32(slot + 8);
      const int n = np * 4;  // WG w's 32 h columns = 4 x 16 B per token
      for (int i0 = lane; i0 < n; i0 += 128) {
        uint4 val[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int idx = i0 + 32 * q;
          if (idx < n) val[q] = lds128(buf + (idx >> 2) * C::PITCH + (4 * w + (idx & 3)) * 16);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int idx = i0 + 32 * q;
          if (idx < n) stg128(a.out + long(pos0 + c0 + (idx >> 2)) * a.out_stride + 64 * nb + (4 * w + (idx & 3)) * 8, val[q]);
        }
      }
    } else {
      const uint32_t tab1 = slot + C::SLOT_TAB1;
      const int n = np * 8;  // WG w's 64 columns = 8 x 16 B per token
      for (int i0 = lane; i0 < n; i0 += 128) {
        uint4 val[4];
        int row[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int idx = i0 + 32 * q;
          if (idx < n) {
            val[q] = lds128(buf + (idx >> 3) * C::PITCH + (8 * w + (idx & 7)) * 16);
            row[q] = lds_s32(tab1 + 4 * (c0 + (idx >> 3)));
          }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int idx = i0 + 32 * q;
          if (idx < n) {
            __nv_bfloat16* dst = a.out + long(row[q]) * a.out_stride + 128 * nb + (8 * w + (idx & 7)) * 8;
#ifdef DS41H_MOE_ABL_NOSTORE  // timing experiment: no global output
            if (val[q].x == 0x12345678u && row[q] < 0) stg128(dst, val[q]);
#else
            if constexpr (LD == 2)
              asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(val[q].x), "r"(val[q].y), "r"(val[q].z),
                           "r"(val[q].w)
                           : "memory");
            else
              stg128(dst, val[q]);
#endif
          }
        }
      }
    }
    __syncwarp();
#ifdef DS41H_MOE_TRACE
    // storer record: t_wait0, t_staged, t_stored, -, np, -, -, tag 3
    if (lane == 0 && pass < TR_N) {
      unsigned long long* r = g_mtrace + ((blockIdx.x * 4 + 2 + w) * TR_N + pass) * 8;
      r[0] = ts0, r[1] = ts1, r[2] = gtimer(), r[3] = r[2], r[4] = unsigned(np), r[5] = 0, r[6] = 0, r[7] = 3;
    }
#endif
    if (lane == 0) {
      mbar_arrive_s(sb0 + 32 + 8 * (2 * w + b));                    // staging buffer free
      if (flags & 1) mbar_arrive_s(sb0 - 32 - 8 * RING + 8 * rs);  // last pass: ring slot released
    }                                                               // (rempty = s_bcast - 8 RING, sbars = s_bcast + 32)
  }
}

// ------------------------------------------------------------------------------------- kernel
// CS > 1 (LD 4 only): thread-block clusters of CS CTAs work on the same (expert, token tile) with weight blocks
// nb = CS * nbg + cluster rank; the cluster leader (rank 0) loads the shared activation (B) boxes once and multicasts
// them to all CTAs (L2 -> SM traffic of B / CS). Protocol: the leader's scheduler fetches tile groups and broadcasts the
// index (tidx / tbar, freed via tfree); every consumer warp releases a stage locally (own weights / scales) and on the
// leader's cluster-empty barrier (count CS * 8) before the leader multicasts the next B into it; cluster barriers at the
// start (barrier init) and at the end (no CTA exits while remote arrives / multicast writes to it may be pending).
template <int MODE, int BNMAX, int WIN, int LD, int CS = 1>
__global__ void __launch_bounds__(NTHREADS, 1) moe_gemm_kernel(const __grid_constant__ GemmArgs a) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  constexpr int NSTAGE = C::NSTAGE;
  static_assert(CS == 1 || LD == 4, "cluster multicast: W4A16 (LD 4) only");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* epi = smem + NSTAGE * C::STAGE;
  uint8_t* ring = epi + C::EPI_BUFS * C::EPI;
  uint8_t* escratch = ring + RING * C::SLOT;
  int* s_off = reinterpret_cast<int*>(escratch + C::SCRATCH);
  int* s_tp = s_off + (MAX_E + 1);
  unsigned long long* full =
      reinterpret_cast<unsigned long long*>((reinterpret_cast<uintptr_t>(s_tp + (MAX_E + 1)) + 7) & ~uintptr_t(7));
  unsigned long long* empty = full + NSTAGE;
  unsigned long long* rfull = empty + NSTAGE;
  // (full == smem + C::BAR_OFF: all partitions above are multiples of 4 bytes and smem is 1024-aligned)
  unsigned long long* rempty = rfull + RING;
  int* s_bcast = reinterpret_cast<int*>(rempty + RING);
  // asynchronous epilogue (LD >= 1): sfull[2 WG][2 buf], sempty[2][2], staging headers int[2][2][4]
  unsigned long long* sbars = reinterpret_cast<unsigned long long*>(s_bcast + 8);

  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) {
      // LD 0: one cp.async.mbarrier.arrive.noinc per loader thread; LD 1..4: lane 0's arrive.expect_tx;
      // LD 6: lane 0's arrive.expect_tx (weights / scales) + the 32 loader lanes' cp.async noinc arrives (B rows)
      mbar_init(&full[s], LD == 6 ? 33 : (LD ? 1 : NLOAD));
      mbar_init(&empty[s], 8);     // one arrive per consumer warp
    }
    for (int s = 0; s < RING; ++s) {
      mbar_init(&rfull[s], 32);  // every scheduler lane publishes its part of the slot
      mbar_init(&rempty[s], LD ? 10 : 8);  // every consumer warp after its epilogue (+ the 2 storer warps, LD >= 1)
    }
    if constexpr (LD >= 1) {
      for (int i = 0; i < 4; ++i) {
        mbar_init(&sbars[i], 4);      // sfull: the 4 warps of the staging WG
        mbar_init(&sbars[4 + i], 1);  // sempty: the storer warp
      }
      mbar_init(&sbars[16], 4);  // drift: WG 0's warps, once (see the consumer loop)
    }
    if constexpr (CS > 1) {
      unsigned long long* cl = reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(full) + C::CL_EMPTY);
      for (int s = 0; s < NSTAGE; ++s) mbar_init(&cl[s], 8 * CS);  // cempty: every consumer warp of the cluster
      for (int s = 0; s < RING; ++s) mbar_init(&cl[8 + s], 1), mbar_init(&cl[8 + RING + s], CS);  // tbar, tfree
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (CS > 1)
    cluster_sync_all();  // barriers of every CTA initialized before any remote arrive / multicast
  else
    __syncthreads();
  pdl_wait();  // off/activations/counters come from the preceding kernels
  if constexpr (LD >= 1) asm volatile("fence.proxy.async.global;" ::: "memory");  // generic writes -> TMA reads

  // Warp-uniform role dispatch (shfl-broadcast warp index, as cutlass canonical_warp_idx_sync):
  // ptxas must see the wgmma pipeline in convergent code or it serializes every wgmma.
  const int warp_idx = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (warp_idx >= 8) {
    // ================================================================ producer warpgroup
    // warp 8 = tile scheduler (dynamic tile fetch, metadata, per-row reference exponents; runs up to
    // RING tiles ahead so its global-memory latency is hidden), warps 9..11 = cp.async loaders.
    setmaxnreg_dec<Regs<BNMAX>::PRODUCER>();
    const int pt = tid - 256;
    const int E = a.E;
    // expert row offsets and exclusive prefix of token tiles ceil(n_e / BNMAX)
    {
      const int per = (E + 127) / 128;  // experts per thread (<= 8)
      int loc = 0;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          const int o0 = a.off[e], o1 = a.off[e + 1];
          s_off[e] = o0;
          loc += ld_ntiles<BNMAX>(o1 - o0, a.tsplit);
        }
      }
      if (pt == 0) s_off[E] = a.off[E];
      // block-wide exclusive scan of loc over the 128 producer threads
      const int lane = pt & 31, w = pt >> 5;
      int inc = loc;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (lane == 31) s_bcast[4 + w] = inc;
      producer_bar();
      int base = 0;
      for (int k = 0; k < w; ++k) base += s_bcast[4 + k];
      int run = base + inc - loc;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          s_tp[e] = run;
          run += ld_ntiles<BNMAX>(s_off[e + 1] - s_off[e], a.tsplit);  // all s_off written before the barrier
        }
      }
      if (pt == 127) s_tp[E] = base + inc;
    }
    producer_bar();

    if (warp_idx == 8) {
      // ---------------------------------------------------------------- scheduler warp
      const int lane = pt;
      const int NBG = a.NB / CS;  // weight-block groups (CS > 1: the cluster's CTAs take nb = CS * nbg + rank)
      const int total = s_tp[E] * NBG;
      const uint32_t crank = CS > 1 ? cluster_rank() : 0u;
      const uint32_t fb0 = smem_u32(full);
      int rslot = 0, gslot = 0;
      unsigned rphase = 0, gphase = 0;
      (void)fb0, (void)gslot, (void)gphase;
      for (;;) {
        int tile = 0;
        if constexpr (CS > 1) {
          if (lane == 0) {
            if (crank == 0) {  // leader: fetch a tile group and broadcast its index to the cluster
              const int g = atomicAdd(&a.ctr[0], 1);
              mbar_wait_cl(fb0 + C::CL_TFREE + 8 * gslot, gphase ^ 1);  // every CTA read the previous index of this slot
              for (int r = 0; r < CS; ++r) {
                st_remote_u32(mapa_rank(fb0 + C::CL_TIDX + 4 * gslot, r), uint32_t(g));
                mbar_arrive_remote(mapa_rank(fb0 + C::CL_TBAR + 8 * gslot, r));
              }
            }
            mbar_wait_cl(fb0 + C::CL_TBAR + 8 * gslot, gphase);
            tile = lds_s32(fb0 + C::CL_TIDX + 4 * gslot);
            mbar_arrive_remote(mapa_rank(fb0 + C::CL_TFREE + 8 * gslot, 0));
          }
          if (++gslot == RING) gslot = 0, gphase ^= 1;
        } else {
          if (lane == 0) tile = atomicAdd(&a.ctr[0], 1);
        }
        tile = __shfl_sync(0xffffffffu, tile, 0);
        mbar_wait_s(smem_u32(&rempty[rslot]), rphase ^ 1);
        uint8_t* slot = ring + rslot * C::SLOT;
        int* hdr = reinterpret_cast<int*>(slot);
        if (tile >= total) {
          if (lane == 0) hdr[3] = -1;
          __syncwarp();
          mbar_arrive(&rfull[rslot]);
          break;
        }
        // decode: largest e with s_tp[e] * NB <= tile (empty experts have s_tp[e] == s_tp[e+1])
        int lo = 0, hi = E;
        while (hi - lo > 1) {
          const int mid = (lo + hi) >> 1;
          if (s_tp[mid] * NBG <= tile)
            lo = mid;
          else
            hi = mid;
        }
        const int e = lo;
        const int nt = s_tp[e + 1] - s_tp[e];
        const int local = tile - s_tp[e] * NBG;
        const int nbg = local / nt;
        const int nb = nbg * CS + int(crank);
        const int pos0 = s_off[e] + (local - nbg * nt) * BNMAX;
        const int nvalid = min(BNMAX, s_off[e + 1] - pos0);
        if (lane == 0) hdr[0] = e, hdr[1] = nb, hdr[2] = pos0, hdr[3] = nvalid;
        // epilogue tables + gather rows (4 independent loads in flight per lane)
        float* tab0 = reinterpret_cast<float*>(slot + C::SLOT_TAB0);
        float* tab1 = reinterpret_cast<float*>(slot + C::SLOT_TAB1);
        int* rows = reinterpret_cast<int*>(slot + C::SLOT_ROWS);
        for (int c0 = 0; c0 < BNMAX; c0 += 128) {
          int rw[4];
          float v1[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int cc = c0 + lane + 32 * q, pos = pos0 + cc;
            const bool ok = cc < nvalid;
            if constexpr (MODE == 1) {
              rw[q] = ok ? a.perm_tok[pos] : -1;
              v1[q] = ok ? a.perm_w[pos] : 0.f;
            } else if constexpr (LD == 2) {  // reduction epilogue: token row and per-routed-row scale
              rw[q] = ok ? pos : -1;
              v1[q] = ok ? __int_as_float(a.perm_tok[pos]) : 0.f;
            } else {
              rw[q] = ok ? pos : -1;
              v1[q] = ok ? __int_as_float(a.perm_slot[pos]) : 0.f;
            }
          }
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int cc = c0 + lane + 32 * q;
            if (cc < BNMAX) {
              // MODE 1: per-token activation scale; MODE 2: hs of the routed row for the reduction epilogue
              // (LD 2), otherwise unused (moe_combine applies it)
              tab0[cc] = rw[q] >= 0 ? ((MODE == 1 && !ld_a16(LD)) || LD == 2 ? a.act_scale[rw[q]] : 0.f) : 0.f;
              tab1[cc] = v1[q];
              rows[cc] = rw[q];
            }
          }
        }
        // per-row reference exponent = max ue8m0 byte of the whole scale row (MODE 2 with the TMA
        // loader: computed by the consumers from the staged scale block instead)
        uint8_t* erow = slot + C::SLOT_EROW;
        if (ld_a16(LD) || (MODE == 2 && LD >= 1)) {  // (LD 4 / 6: exact bf16 expansion, no reference exponent)
        } else if (a.s_rstride % 16 == 0) {
          // 16-byte aligned rows (w13: 160 B): lane handles rows lane + 32 q, one chunk of all four at a time
          long srow[4];
          unsigned m[4] = {0u, 0u, 0u, 0u};
#pragma unroll
          for (int q = 0; q < 4; ++q)
            srow[q] = long(e) * a.s_estride + long(weight_row<MODE>(lane + 32 * q, nb, a.I)) * a.s_rstride;
          for (int j = 0; j < a.s_rstride; j += 16) {
            uint4 v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = __ldg(reinterpret_cast<const uint4*>(a.s + srow[q] + j));
#pragma unroll
            for (int q = 0; q < 4; ++q) m[q] = __vmaxu4(m[q], __vmaxu4(__vmaxu4(v[q].x, v[q].y), __vmaxu4(v[q].z, v[q].w)));
          }
#pragma unroll
          for (int q = 0; q < 4; ++q)
            erow[lane + 32 * q] = uint8_t(max(max(m[q] & 0xFFu, (m[q] >> 8) & 0xFFu), max((m[q] >> 16) & 0xFFu, m[q] >> 24)));
        } else {
          // MODE 2 (w2: 18-byte rows): the tile's 128 rows are contiguous -> copy the block (aligned
          // 4-byte words) into the scratch, then reduce each row from smem
          const long b0 = long(e) * a.s_estride + long(weight_row<MODE>(0, nb, a.I)) * a.s_rstride;
          const long w0 = b0 & ~3L;
          const int nw = int((b0 + 128L * a.s_rstride - w0 + 3) >> 2);
          unsigned* sw = reinterpret_cast<unsigned*>(escratch);
          for (int i0 = 0; i0 < nw; i0 += 32 * 8) {
            unsigned v[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const int i = i0 + lane + 32 * q;
              const long ad = w0 + 4L * i;
              v[q] = (i < nw && ad + 4 <= a.s_total) ? __ldg(reinterpret_cast<const unsigned*>(a.s + ad)) : 0u;
            }
#pragma unroll
            for (int q = 0; q < 8; ++q)
              if (i0 + lane + 32 * q < nw) sw[i0 + lane + 32 * q] = v[q];
          }
          __syncwarp();
          const int sb = int(b0 - w0);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int lr = lane + 32 * q;
            int mx = 0;
            for (int k = 0; k < a.s_rstride; ++k) mx = max(mx, int(escratch[sb + lr * a.s_rstride + k]));
            erow[lr] = uint8_t(mx);
          }
          __syncwarp();  // scratch reads done before the next tile overwrites it
        }
        __syncwarp();
        mbar_arrive(&rfull[rslot]);
        if (++rslot == RING) rslot = 0, rphase ^= 1;
      }
      // all CTAs (CS > 1: cluster leaders) past their last tile fetch -> the last one resets the counters for the next launch
      if (lane == 0 && crank == 0) {
        __threadfence();
        if (atomicAdd(&a.ctr[1], 1) == int(gridDim.x) / CS - 1) {
          __threadfence();
          a.ctr[0] = 0;
          a.ctr[1] = 0;
        }
      }
      if constexpr (CS > 1) cluster_sync_all();
      return;
    }

    if constexpr (LD >= 1) {
      // -------------------------------------------------------------- TMA loader (warp 9)
      // lane 0 issues the weight / scale boxes and (LD 1, 2) the activation boxes of routed rows
      // [pos0, pos0 + nvalid); LD 3 (MODE 1): every lane issues one-row boxes of the gathered token
      // rows (activations in token order, no act_quant_perm)
      if (warp_idx == 9) {
        const int lane = tid & 31;
        const int nkb = ld_a16(LD) ? a.K / 64 : (a.K + 127) / 128;
        const int rows_e = int(a.w_estride / a.w_rstride);  // weight rows per expert
        int stage = 0, rslot = 0;
        unsigned phase = 0, rphase = 0;
        for (;;) {
          mbar_wait_s(smem_u32(&rfull[rslot]), rphase);
          const int* hdr = reinterpret_cast<const int*>(ring + rslot * C::SLOT);
          const int nvalid = hdr[3];
          if (nvalid < 0) break;
          const int e = hdr[0], nb = hdr[1], pos0 = hdr[2];
          const uint32_t rows_s = smem_u32(ring + rslot * C::SLOT + C::SLOT_ROWS);
          if (++rslot == RING) rslot = 0, rphase ^= 1;
          const int nbox = (LD == 3 || LD == 6) ? 0 : (nvalid + a.brows - 1) / a.brows;  // B boxes of brows (64) routed rows
          [[maybe_unused]] const bool leader = CS == 1 || cluster_rank() == 0;
          const int wrow = e * rows_e + (MODE == 1 ? 64 * nb : 128 * nb);
          const uint32_t sbytes = MODE == 1 ? 2048u : uint32_t(128 * a.s_rstride);
          for (int kb = 0; kb < nkb; ++kb) {
            mbar_wait_s(smem_u32(&empty[stage]), phase ^ 1);
            const uint32_t st = smem_u32(smem + stage * C::STAGE), fb = smem_u32(&full[stage]);
            if constexpr (CS > 1) {  // leader: B multicast only after every CTA's consumers released this stage
              if (leader) mbar_wait_cl(smem_u32(full) + C::CL_EMPTY + 8 * stage, phase ^ 1);
            }
            if (lane == 0) {
              const uint32_t bbytes = LD == 3 ? uint32_t(nvalid * 128) : (LD == 6 ? 0u : uint32_t(nbox * a.brows * 128));
#ifdef DS41H_MOE_ABL_BX2  // energy experiment only: every B box loaded twice (same data, same destination)
              constexpr int BREP = 2;
#else
              constexpr int BREP = 1;
#endif
#ifdef DS41H_MOE_ABL_WX2  // energy experiment only: every weight box loaded twice
              constexpr int WREP = 2;
#else
              constexpr int WREP = 1;
#endif
              mbar_expect_tx_s(fb, BREP * bbytes + WREP * uint32_t(C::A_BYTES) + sbytes);
              // (B coordinates are bytes: 128 e4m3 or 64 bf16 activations per K block)
              for (int rep = 0; rep < BREP; ++rep) {
                if constexpr (CS > 1) {  // the leader multicasts B to every CTA of the cluster (same offsets / barrier)
                  if (leader)
                    for (int j = 0; j < nbox; ++j)
                      tma_load_2d_mc(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb, uint16_t((1u << CS) - 1));
                } else {
                  for (int j = 0; j < nbox; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
                }
              }
              if constexpr (ld_a16(LD)) {  // 64-wide K blocks: 32 FP4 bytes per row; scales: 16-byte group of 8 blocks
                if constexpr (MODE == 1) {
                  for (int rep = 0; rep < WREP; ++rep) {
                    tma_load_2d(st + C::B_STAGE, &a.tm_w, kb * 32, wrow, fb);               // gate rows
                    tma_load_2d(st + C::B_STAGE + 2048, &a.tm_w, kb * 32, wrow + a.I, fb);  // up rows
                  }
                  tma_load_2d(st + C::B_STAGE + C::A_BYTES, &a.tm_s, 16 * (kb >> 3), wrow, fb);
                  tma_load_2d(st + C::B_STAGE + C::A_BYTES + 1024, &a.tm_s, 16 * (kb >> 3), wrow + a.I, fb);
                } else {
                  for (int rep = 0; rep < WREP; ++rep) tma_load_2d(st + C::B_STAGE, &a.tm_w, kb * 32, wrow, fb);  // 128 rows
                  bulk_load(st + C::B_STAGE + C::A_BYTES, a.s + long(wrow) * a.s_rstride, sbytes, fb);
                }
              } else if constexpr (MODE == 1) {
                tma_load_2d(st + C::B_STAGE, &a.tm_w, kb * 64, wrow, fb);               // gate rows
                tma_load_2d(st + C::B_STAGE + 4096, &a.tm_w, kb * 64, wrow + a.I, fb);  // up rows
                tma_load_2d(st + C::B_STAGE + A_STAGE, &a.tm_s, 16 * (kb >> 2), wrow, fb);
                tma_load_2d(st + C::B_STAGE + A_STAGE + 1024, &a.tm_s, 16 * (kb >> 2), wrow + a.I, fb);
              } else {
                tma_load_2d(st + C::B_STAGE, &a.tm_w, kb * 64, wrow, fb);  // 128 rows (tail block: OOB bytes -> 0)
                bulk_load(st + C::B_STAGE + A_STAGE, a.s + long(wrow) * a.s_rstride, sbytes, fb);
              }
            }
            if constexpr (LD == 3) {
              __syncwarp();
              for (int r = lane; r < nvalid; r += 32) tma_load_2d(st + r * 128, &a.tm_b, kb * 128, lds_s32(rows_s + 4 * r), fb);
            }
            if constexpr (LD == 6) {
              // gathered B (rows_s = token rows, MODE 1): 8 lanes per row (16-byte chunk ch = lane & 7), 4 rows per
              // instruction; chunk ch of tile row r -> 16-byte slot ch ^ (r & 7) (SWIZZLE_128B of the 1024-aligned
              // stage, as the LD 0 loader). Rows >= nvalid are not loaded (their accumulator columns are discarded).
              const int ch = lane & 7;
              const uint8_t* src = a.act + kb * 128 + ch * 16;  // xl: bf16 [T, K], 128 bytes per 64-wide K block
              const long rstride = 2L * a.K;
              for (int r = lane >> 3; r < nvalid; r += 4)
                cp_async16_s(st + r * 128 + ((ch ^ (r & 7)) << 4), src + long(lds_s32(rows_s + 4 * r)) * rstride);
              asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(fb) : "memory");
            }
            if (++stage == NSTAGE) stage = 0, phase ^= 1;
          }
        }
      } else if (warp_idx >= 10) {
        // storer warps (warp 10 + wg): global stores of the passes staged by consumer WG wg
        storer_loop<MODE, BNMAX, WIN, LD>(a, warp_idx - 10, tid & 31, smem_u32(sbars), smem_u32(epi), smem_u32(ring));
      }
      if constexpr (CS > 1) cluster_sync_all();
      return;
    }
    // ------------------------------------------------------------------ loader warps (96 threads)
    const int lt = pt - 32;
    const int nkb = (a.K + 127) / 128;
    int stage = 0, rslot = 0;
    unsigned phase = 0, rphase = 0;
    for (;;) {
      mbar_wait_s(smem_u32(&rfull[rslot]), rphase);
      const uint8_t* slot = ring + rslot * C::SLOT;
      const int* hdr = reinterpret_cast<const int*>(slot);
      const int nvalid = hdr[3];
      if (nvalid < 0) break;
      const int e = hdr[0], nb = hdr[1];
      const uint32_t rows_s = smem_u32(slot + C::SLOT_ROWS);
      if (++rslot == RING) rslot = 0, rphase ^= 1;
      const long wexp = long(e) * (a.w_estride / a.w_rstride);  // first weight row of the expert
      for (int kb = 0; kb < nkb; ++kb) {
        mbar_wait_s(smem_u32(&empty[stage]), phase ^ 1);
        uint8_t* st = smem + stage * C::STAGE;
        const int kbytes = min(128, a.K - kb * 128);  // 128 or 64
        // B: gathered activation rows (rows >= nvalid are not loaded: their accumulator columns are
        // discarded by the epilogue); thread owns 16-byte chunk ch = lt & 7 of rows (lt >> 3) + 12 j,
        // 128B swizzle
        {
          const int ch = lt & 7;
          if (ch * 16 < kbytes) {
            const uint8_t* src = a.act + kb * 128 + ch * 16;
            for (int r = lt >> 3; r < nvalid; r += NLOAD / 8)
              cp_async16_zfill(st + r * 128 + ((ch ^ (r & 7)) << 4), src + long(lds_s32(rows_s + 4 * r)) * a.K, 16u);
          }
        }
        // A: FP4 bytes [kb*64, kb*64 + 64) of the 128 weight rows: thread owns piece lt & 3 of rows
        // (lt >> 2) + 24 j (tail blocks are zero filled)
        {
          const int piece = lt & 3;
          const bool ok = piece * 16 < (kbytes >> 1);
          for (int r = lt >> 2; r < 128; r += NLOAD / 4) {
            const uint8_t* wsrc = a.w + (wexp + weight_row<MODE>(r, nb, a.I)) * a.w_rstride + kb * 64 + piece * 16;
            cp_async16_zfill(st + C::B_STAGE + r * 64 + piece * 16, ok ? wsrc : a.w, ok ? 16u : 0u);
          }
        }
        // scales: 8 bytes from the 4-aligned address at or below row r's block-kb scale bytes (scale
        // rows are 18 bytes for w2, so the 4 bytes of a block can straddle a word); thread owns word
        // lt & 1 of rows (lt >> 1) + 48 j
        {
          const int hh = lt & 1;
          for (int r = lt >> 1; r < 128; r += NLOAD / 2) {
            const long srow = long(e) * a.s_estride + long(weight_row<MODE>(r, nb, a.I)) * a.s_rstride;
            const long addr = ((srow + kb * 4) & ~3L) + 4 * hh;
            const long rem = a.s_total - addr;
            const unsigned nbytes = rem >= 4 ? 4u : (rem > 0 ? unsigned(rem) : 0u);
            cp_async4_zfill(st + C::B_STAGE + A_STAGE + r * 8 + 4 * hh, nbytes ? a.s + addr : a.s, nbytes);
          }
        }
        cp_async_mbar_arrive(&full[stage]);
        if (++stage == NSTAGE) stage = 0, phase ^= 1;
      }
    }
    cp_async_wait_all();
    return;
  }

  // ================================================================== consumer warpgroups
  setmaxnreg_inc<Regs<BNMAX>::CONSUMER>();
  Consumer c;
  c.bar = smem_u32(full);
  c.stage = 0;
  c.phase = 0;
  c.tid = tid;
  c.epass = 0;
  {
    // Tile-local A rows of this thread's accumulator rows g and g+8 (wgmma D fragment). MODE 1
    // pairs gate row j with up row j: accumulator row g <-> gate j, g+8 <-> up j.
    const int wg = warp_idx >> 2, wq = warp_idx & 3, g = (tid & 31) >> 2;
    c.sr0 = MODE == 1 ? 32 * wg + 8 * wq + g : 64 * wg + 16 * wq + g;
    // W4A16 GEMM1: accumulator row of (logical) h column c = 32 wg + 8 wq + g computes physical h column
    // a16_phys(c), so h is written in the logical K order GEMM2 consumes (see a16_phys)
    if constexpr (MODE == 1 && ld_a16(LD)) c.sr0 = a16_phys(c.sr0);
    c.sr1 = MODE == 1 ? 64 + c.sr0 : c.sr0 + 8;
  }
  const uint32_t ring_s = smem_u32(ring), rbar = smem_u32(rfull);  // rempty[s] = rfull[s] + 8 * RING
  int rslot = 0;
  unsigned rphase = 0;
  if constexpr (LD >= 1 && WG_DRIFT) {
    // WG drift: both WGs work on the same tiles (64 weight rows each) and would otherwise run their
    // epilogues (drain + SwiGLU / scaling + stmatrix) at the same time, leaving the tensor core idle.
    // WG 1 starts NSTAGE - 1 K blocks behind WG 0; the shared stage ring keeps it there.
    if (tid >= 128) mbar_wait_s(epi_sbar<NSTAGE>(c) + 128, 0);
  }
  float acc[BNMAX / 2];
  float fin[WIN ? BNMAX / 2 : 1];
#ifdef DS41H_MOE_TRACE
  int ntr = 0;
#endif
  for (;;) {
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr_ring = gtimer();
#endif
    mbar_wait_s(rbar + 8 * rslot, rphase);
    c.slot = ring_s + rslot * C::SLOT;
    c.nvalid = lds_s32(c.slot + 12);
    if (c.nvalid < 0) break;
    c.e = lds_s32(c.slot);
    c.nb = lds_s32(c.slot + 4);
    c.pos0 = lds_s32(c.slot + 8);
    if constexpr (ld_a16(LD)) {
      c.erow0 = c.erow1 = 0;  // exact bf16 expansion: no reference exponent
    } else if constexpr (MODE == 2 && LD >= 1) {
      // reference exponents from the tile's scale block, which every stage of the tile holds (wait
      // for the first stage here, outside the register-heavy mainloop; its own wait then passes)
      mbar_wait_s(c.bar + 8 * c.stage, c.phase);
      const uint32_t ss = c.bar - C::BAR_OFF + c.stage * C::STAGE + C::B_STAGE + A_STAGE;
      const int t = tid & 3;
      int m0 = 0, m1 = 0;
      for (int j = t; j < a.s_rstride; j += 4)
        m0 = max(m0, lds_u8(ss + c.sr0 * a.s_rstride + j)), m1 = max(m1, lds_u8(ss + c.sr1 * a.s_rstride + j));
      m0 = max(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      c.erow0 = max(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      c.erow1 = max(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    } else {
      c.erow0 = lds_u8(c.slot + C::SLOT_EROW + c.sr0);
      c.erow1 = lds_u8(c.slot + C::SLOT_EROW + c.sr1);
    }
    const int nsel = (c.nvalid + 15) >> 4;  // tile width N = 16 nsel
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr1 = gtimer();
    const long long tc0 = clock64();
    c.twait = 0;
#endif
    dispatch_tile<MODE, BNMAX, WIN, LD, CS>(nsel, c, a, acc, fin);
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr2 = gtimer();
    const long long tc1 = clock64();
#endif
    // epilogue (shared by all N; chunks beyond N are skipped at run time)
    constexpr int EPC = C::EPI_TOK / 8;
    if constexpr (WIN != 0) {
      epi_pass<MODE, BNMAX, WIN, LD, 0>(c, a, fin, min(2 * nsel, EPC), 2 * nsel <= EPC);
      if constexpr (BNMAX > C::EPI_TOK)
        if (2 * nsel > EPC) epi_pass<MODE, BNMAX, WIN, LD, 1>(c, a, fin, 2 * nsel - EPC, true);
    } else {
      epi_pass<MODE, BNMAX, WIN, LD, 0>(c, a, acc, min(2 * nsel, EPC), 2 * nsel <= EPC);
      if constexpr (BNMAX > C::EPI_TOK)
        if (2 * nsel > EPC) epi_pass<MODE, BNMAX, WIN, LD, 1>(c, a, acc, 2 * nsel - EPC, true);
    }
    __syncwarp();
#ifdef DS41H_MOE_TRACE
    // record: t_ring, t_mainloop_start, t_mainloop_end, t_epilogue_end, nsel | nb << 8, full-wait clk, mainloop clk, tag 2
    if ((tid & 127) == 0 && ntr < TR_N) {
      unsigned long long* r = g_mtrace + ((blockIdx.x * 4 + (tid >> 7)) * TR_N + ntr) * 8;
      r[0] = tr_ring, r[1] = tr1, r[2] = tr2, r[3] = gtimer(), r[4] = unsigned(nsel) | (unsigned(c.nb) << 8), r[5] = c.twait, r[6] = tc1 - tc0, r[7] = 2;
    }
    ++ntr;
#endif
    if ((tid & 31) == 0) mbar_arrive_s(rbar + 8 * (RING + rslot));  // tables/erow of this slot no longer read
    if (++rslot == RING) rslot = 0, rphase ^= 1;
  }
  if constexpr (LD >= 1) {  // tell this WG's storer warp to exit
    if (WG_DRIFT && tid < 128 && c.epass == 0 && (tid & 31) == 0) mbar_arrive_s(epi_sbar<NSTAGE>(c) + 128);  // no tile: release WG 1
    epi_acquire<NSTAGE, C::EPI_BUFS>(c, tid >> 7);
    epi_post<NSTAGE, C::EPI_BUFS>(c, tid >> 7, 0, 0, 0, 2);
  }
  if constexpr (CS > 1) cluster_sync_all();  // no CTA leaves while cluster peers may still signal / write into it
}

// ============================================================================================
// GEMM2 "B-stationary" kernel (MODE 2, variant 3).
//
// A work item is (expert e, token tile of <= 128 routed rows [pos0, pos0 + 128), range [nb0, nb1) of
// 128-row w2 blocks). The item's whole B operand (hq rows x I, one 128B-swizzled tile per 128-wide K
// block) is loaded once by TMA and stays in shared memory while the CTA streams the expert's w2
// blocks, so activations are no longer re-gathered for every weight block (the s1 kernel moved
// ~1.1 GB of hq through L2 per layer for this). Consumer WG w owns rows 64w..64w+63 of every block
// and has its own 2-stage weight ring, filled with two bulk copies per block (its 64 rows x I/2 FP4
// bytes and their I/32 scale bytes are both contiguous in the raw layout). The per-row reference
// exponent (max ue8m0 byte of the row) is reduced from the staged scale bytes by each lane quad.
// The whole K range of a block is staged, so the MMA loop has no barrier waits.
// Epilogue: stmatrix into a per-WG staging tile, then one 128-byte bulk store per routed row (slot
// row perm_slot[pos], columns 128 nb + 64 w ..), asynchronous to the next block's MMAs.
// Item order: all full 128-row tiles, then the partial tiles by decreasing width (longest processing
// time first; items are fetched dynamically from a global counter), each tile split into CHUNKS nb
// ranges (CHUNKS = ceil(target_items / tiles), so small batches still spread over all SMs).
namespace g2 {
constexpr int BN = 128;
constexpr int MAXKB = 5;                     // I <= 640
constexpr int BKB = BN * 128;                // B bytes per 128-wide K block
constexpr int NWS = 2;                       // weight stages per consumer WG
constexpr int WSTAGE = 64 * 320 + 64 * 20;   // 64 rows x (I/2 + I/32) bytes for I <= 640 (21760, 128-multiple)
constexpr int PITCH = 144;                   // staging row: 64 bf16 + 16 B pad (conflict-free stmatrix)
constexpr int STG = BN * PITCH;
constexpr int IRING = 4;
constexpr int SLOT = 32 + 8 * BN;            // int hdr[8] (e, pos0, nvalid, nb0, nb1, -, -, end) | int tab[BN] | f32 tabf[BN]
constexpr int OFF_B = 0;                     // 1024-aligned (128B swizzle)
constexpr int OFF_W = OFF_B + MAXKB * BKB;
constexpr int OFF_STG = OFF_W + 2 * NWS * WSTAGE;
constexpr int OFF_RING = OFF_STG + 2 * STG;
constexpr int OFF_ARR = OFF_RING + IRING * SLOT;                // s_off[E+1] | s_fp[E+1] | s_plist[E+1] | s_cnt[32]
constexpr int OFF_BAR = (OFF_ARR + (3 * (MAX_E + 1) + 32) * 4 + 7) / 8 * 8;
// barriers: rfull[IRING] rempty[IRING] bfull bempty wfull[2][NWS] wempty[2][NWS]
constexpr int B_RFULL = 0, B_REMPTY = IRING, B_BFULL = 2 * IRING, B_BEMPTY = 2 * IRING + 1, B_WFULL = 2 * IRING + 2,
              B_WEMPTY = 2 * IRING + 2 + 2 * NWS, NBAR = 2 * IRING + 2 + 4 * NWS;
constexpr int SMEM = 1024 + OFF_BAR + 8 * NBAR;
static_assert(SMEM <= 227 * 1024, "g2 smem");
static_assert(OFF_W % 1024 == 0 && OFF_STG % 16 == 0 && OFF_RING % 16 == 0, "g2 alignment");
constexpr int CONSUMER_REGS = 232, PRODUCER_REGS = 40;  // 2 * 232 + 40 = 504 (the 168 x 384 launch allocation)
}  // namespace g2


struct G2Args {
  CUtensorMap tm_b;          // hq [R, I] u8: dims {I, R}, box {128, 128}, SWIZZLE_128B (OOB -> 0)
  const uint8_t* w;          // w2 [E, H, I/2]
  const uint8_t* s;          // s2 [E, H, I/32]
  const int* off;            // [E+1] routed-row offsets per expert
  const int* perm_slot;      // [R] slot row t*K + k of routed row
  int* ctr;                  // [2] item counter, finished-CTA counter (zero at launch, reset at exit)
  const int* perm_tok;       // [R] token of routed row (EPI_RED)
  const float* hs;           // [R] per-routed-row activation scale, routed-row order (EPI_RED)
  __nv_bfloat16* out;        // EPI_BULK/EPI_STG: slot [R, H], unscaled (moe_combine applies hs);
                             // EPI_RED: out [T, H] (zeroed), += hs * y of every routed row
  int E, H, I, NB;           // NB = H / 128
  int target_items;          // >= this many items per launch (CHUNKS = ceil(target / tiles), <= NB)
  int brows;                 // TMA box rows of tm_b (min(128, R))
};


// One 128-row w2 block of one consumer WG: 64 rows (local lr0 = 16 wq + g, lr1 = lr0 + 8) x N tokens,
// whole K from the staged block `wst` (rows I/2 B, then scales I/32 B per row) and the stationary B.
// RS-wgmma has a ~200-cycle issue-to-completion latency at the capped clock; with one accumulator
// chain per warpgroup a k32 step costs that latency whatever N is (measured 205-235 clk/step for
// N = 16..128), so the tensor core idles for N < ~256. Here consecutive K blocks go to two
// independent accumulators (acc0: even blocks, acc1: odd blocks; the caller sums them) and the
// k32 steps of a block pair are issued interleaved (a0 b0 a1 b1 ...): each issue only waits for the
// issue before the previous one (wait_group 1), so two chains per warpgroup are in flight with a
// single double-buffered A fragment (8 registers; 2 x N/2 accumulators must fit the 168-register
// allocation). The FP4 word of a step is loaded from shared memory just before its expansion.
template <int N>
__device__ __forceinline__ void g2_mainloop(float (&acc0_all)[g2::BN / 2], float (&acc1_all)[g2::BN / 2], uint32_t wst, uint32_t bbase, int wrs,
                                            int srs, int nfull, int nkb, int lr0, int lr1, int erow0, int erow1, int t) {
  float(&acc0)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc0_all[0]);
  float(&acc1)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc1_all[0]);
  const uint32_t soff = wst + 64 * wrs;
  const uint32_t f0 = wst + lr0 * wrs + 16 * t, f1 = wst + lr1 * wrs + 16 * t;  // full block kb, step s: + 64 kb + 4 s
  const uint32_t q0 = wst + lr0 * wrs + 8 * t, q1 = wst + lr1 * wrs + 8 * t;    // 64-wide tail: + 64 kb + 4 s (s < 2)
  const uint32_t e0a = soff + lr0 * srs, e1a = soff + lr1 * srs;
  unsigned A[2][4];
  unsigned La[4], Lb[4];
  auto lut = [&](int kb, bool full, unsigned (&L)[4]) {
    const int sub = full ? t : (t >> 1);
    fp4_e4m3_lut(lds_u8(e0a + 4 * kb + sub) - erow0, L[0], L[1]);
    fp4_e4m3_lut(lds_u8(e1a + 4 * kb + sub) - erow1, L[2], L[3]);
  };
  auto prep = [&](int kb, int s, bool full, const unsigned (&L)[4], unsigned (&F)[4]) {
    const unsigned w0 = unsigned(lds_s32((full ? f0 : q0) + 64 * kb + 4 * s));
    const unsigned w1 = unsigned(lds_s32((full ? f1 : q1) + 64 * kb + 4 * s));
    fp4x8_to_e4m3_lut(w0, L[0], L[1], F[0], F[2]);
    fp4x8_to_e4m3_lut(w1, L[2], L[3], F[1], F[3]);
  };
  // (no per-issue accumulator fences: two chains are in flight, and ptxas serializes the pipeline
  // if a non-wgmma instruction touches an in-flight accumulator; acc is only accessed by the wgmma
  // asm statements between the fences at the start and the end)
  auto mma = [&](float (&acc)[N / 2], unsigned (&F)[4], uint64_t desc, bool accumulate) {
    wgmma_fence();
    WgmmaRS<N>::mma(acc, F, desc, accumulate ? 1 : 0);
    wgmma_commit();
  };
  auto retire = [&](unsigned (&F)[4]) {  // the issue before the previous one is done: F is free
    wgmma_wait<1>();
#pragma unroll
    for (int j = 0; j < 4; ++j) reg_fence(F[j]);
  };
  auto desc = [&](int kb) { return desc_sw128(bbase + kb * g2::BKB); };
  // first fragment of the sequence starting at block kb (a pair if two full blocks remain)
  auto head = [&](int kb) {
    if (kb + 1 < nfull) {
      lut(kb, true, La);
      lut(kb + 1, true, Lb);
      prep(kb, 0, true, La, A[0]);
    } else if (kb < nkb) {
      const bool full = kb < nfull;
      lut(kb, full, La);
      prep(kb, 0, full, La, A[0]);
    }
  };
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  head(0);
  int kb = 0;
#pragma unroll 1
  for (; kb + 1 < nfull; kb += 2) {
    const uint64_t da = desc(kb), db = desc(kb + 1);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      mma(acc0, A[0], da + 2 * s, kb > 0 || s > 0);
      retire(A[1]);
      prep(kb + 1, s, true, Lb, A[1]);
      mma(acc1, A[1], db + 2 * s, kb > 0 || s > 0);
      retire(A[0]);
      if (s < 3) prep(kb, s + 1, true, La, A[0]);
    }
    head(kb + 2);
  }
  // remaining single blocks (an odd full block and / or the 64-wide tail): acc0, then acc1
  auto single = [&](float (&acc)[N / 2], int kb, bool fresh) {
    const bool full = kb < nfull;
    const int ns = full ? 4 : 2;
    const uint64_t d = desc(kb);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      if (s < ns) {
        mma(acc, A[s & 1], d + 2 * s, !(fresh && s == 0));
        retire(A[(s + 1) & 1]);
        if (s + 1 < ns) prep(kb, s + 1, full, La, A[(s + 1) & 1]);
      }
    }
    // ns is even: the sequence ended on A[1], the next block's first fragment goes to A[0]
    if (kb + 1 < nkb) head(kb + 1);
  };
  if (kb < nkb) single(acc0, kb, kb == 0), ++kb;
  if (kb < nkb) single(acc1, kb, kb == 1), ++kb;
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  if (nkb == 1) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc1[i] = 0.f;
  }
}

template <int NS = g2::BN / 16>
__device__ __forceinline__ void g2_dispatch(int nsel, float (&acc0)[g2::BN / 2], float (&acc1)[g2::BN / 2], uint32_t wst, uint32_t bbase, int wrs,
                                            int srs, int nfull, int nkb, int lr0, int lr1, int erow0, int erow1, int t) {
  if constexpr (NS >= 1) {
    if (nsel == NS) {
      g2_mainloop<16 * NS>(acc0, acc1, wst, bbase, wrs, srs, nfull, nkb, lr0, lr1, erow0, erow1, t);
      return;
    }
    g2_dispatch<NS - 1>(nsel, acc0, acc1, wst, bbase, wrs, srs, nfull, nkb, lr0, lr1, erow0, erow1, t);
  }
}

// GEMM2-BS epilogue modes
constexpr int EPI_BULK = 0;  // slot rows (unscaled), one 128-byte bulk store per routed row
constexpr int EPI_STG = 1;   // slot rows (unscaled), cooperative 16-byte stores
constexpr int EPI_RED = 2;   // out[token] += hs * y (bf16x2 vector reductions; out zeroed beforehand)
constexpr int EPI_NONE = 3;  // no output (timing experiments)

template <int EPI>
__global__ void __launch_bounds__(NTHREADS, 1) moe_gemm2_bs_kernel(const __grid_constant__ G2Args a) {
  using namespace g2;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sb = smem_u32(sm);
  int* s_off = reinterpret_cast<int*>(sm + OFF_ARR);
  int* s_fp = s_off + (MAX_E + 1);
  int* s_plist = s_fp + (MAX_E + 1);
  int* s_cnt = s_plist + (MAX_E + 1);  // [0..7] partial tiles per bucket, [8..15] fill pointers, [16] F, [17] P, [20..23] scan
  const uint32_t bar = sb + OFF_BAR;
  auto B = [&](int i) { return bar + 8 * i; };
  const int tid = threadIdx.x;
  if (tid == 0) {
    unsigned long long* b = reinterpret_cast<unsigned long long*>(sm + OFF_BAR);
    for (int r = 0; r < IRING; ++r) {
      mbar_init(&b[B_RFULL + r], 32);  // every scheduler lane publishes its part of the slot
      mbar_init(&b[B_REMPTY + r], 8);  // every consumer warp releases it at the end of the item
    }
    mbar_init(&b[B_BFULL], 1);
    mbar_init(&b[B_BEMPTY], 8);
    for (int i = 0; i < 2 * NWS; ++i) {
      mbar_init(&b[B_WFULL + i], 1);
      mbar_init(&b[B_WEMPTY + i], 4);  // the 4 warps of the owning WG
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  pdl_wait();  // off / hq / perm_slot / counters come from the preceding kernels
  // order the preceding kernels' generic-proxy writes (hq, and the output buffer) before this
  // kernel's async-proxy (TMA / bulk copy) reads and writes
  asm volatile("fence.proxy.async.global;" ::: "memory");
  const int warp_idx = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int lane = tid & 31;
  const int wrs = a.I / 2, srs = a.I / 32;
  const int nfull = a.I / 128, nkb = (a.I + 127) / 128;

  if (warp_idx >= 8) {
    // ============================================================== producer warpgroup
    setmaxnreg_dec<PRODUCER_REGS>();
    const int pt = tid - 256;
    const int E = a.E;
    if (pt < 32) s_cnt[pt] = 0;
    producer_bar();
    // per expert: offsets, full tiles (exclusive prefix s_fp), partial-tile width buckets
    {
      const int per = (E + 127) / 128;
      int loc = 0;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          const int o0 = a.off[e], n = a.off[e + 1] - o0;
          s_off[e] = o0;
          loc += n / BN;
          if (n % BN) atomicAdd(&s_cnt[8 - ((n % BN) + 15) / 16], 1);  // bucket 0 = widest
        }
      }
      if (pt == 0) s_off[E] = a.off[E];
      const int w = pt >> 5;
      int inc = loc;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (lane == 31) s_cnt[20 + w] = inc;
      producer_bar();
      int base = 0;
      for (int k = 0; k < w; ++k) base += s_cnt[20 + k];
      int run = base + inc - loc;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          s_fp[e] = run;
          run += (s_off[e + 1] - s_off[e]) / BN;
        }
      }
      if (pt == 127) s_fp[E] = s_cnt[16] = base + inc;
      if (pt == 0) {
        int acc = 0;
        for (int b = 0; b < 8; ++b) s_cnt[8 + b] = acc, acc += s_cnt[b];
        s_cnt[17] = acc;
      }
      producer_bar();
    }
    const int F = s_cnt[16], P = s_cnt[17], tiles = F + P;
    const int chunks = tiles > 0 ? max(1, min(a.NB, (a.target_items + tiles - 1) / tiles)) : 1;
    const int total = tiles * chunks;

    if (warp_idx == 8) {
      // ---------------------------------------------------------- scheduler warp
      // Partial tiles by bucket (widest first), experts in increasing order inside a bucket: the
      // item -> tile mapping must be identical in every CTA (items come from one global counter).
      {
        int fill[8];
#pragma unroll
        for (int b = 0; b < 8; ++b) fill[b] = s_cnt[8 + b];
        for (int e0 = 0; e0 < E; e0 += 32) {
          const int e = e0 + lane;
          const int r = e < E ? (s_off[e + 1] - s_off[e]) % BN : 0;
          const int bk = r ? 8 - (r + 15) / 16 : -1;
#pragma unroll
          for (int b = 0; b < 8; ++b) {
            const unsigned m = __ballot_sync(0xffffffffu, bk == b);
            if (bk == b) s_plist[fill[b] + __popc(m & ((1u << lane) - 1u))] = e;
            fill[b] += __popc(m);
          }
        }
        __syncwarp();
      }
      int rslot = 0;
      unsigned rphase = 0;
      for (;;) {
        int item = 0;
        if (lane == 0) item = atomicAdd(&a.ctr[0], 1);
        item = __shfl_sync(0xffffffffu, item, 0);
        mbar_wait_s(B(B_REMPTY + rslot), rphase ^ 1);
        int* hdr = reinterpret_cast<int*>(sm + OFF_RING + rslot * SLOT);
        int* tab = hdr + 8;
        if (item >= total) {
          if (lane == 0) hdr[7] = 1;
          __syncwarp();
          mbar_arrive_s(B(B_RFULL + rslot));
          break;
        }
        const int tile = item / chunks, ch = item - tile * chunks;
        int e, pos0, nvalid;
        if (tile < F) {
          int lo = 0, hi = E;
          while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (s_fp[mid] <= tile)
              lo = mid;
            else
              hi = mid;
          }
          e = lo;
          pos0 = s_off[e] + (tile - s_fp[e]) * BN;
          nvalid = BN;
        } else {
          e = s_plist[tile - F];
          const int n = s_off[e + 1] - s_off[e];
          pos0 = s_off[e] + (n / BN) * BN;
          nvalid = n - (n / BN) * BN;
        }
        int sl[BN / 32];
        float sf[BN / 32];
#pragma unroll
        for (int q = 0; q < BN / 32; ++q) {
          const int cc = lane + 32 * q;
          const bool ok = cc < nvalid;
          sl[q] = ok ? (EPI == EPI_RED ? a.perm_tok[pos0 + cc] : a.perm_slot[pos0 + cc]) : 0;
          sf[q] = (EPI == EPI_RED && ok) ? a.hs[pos0 + cc] : 0.f;
        }
        float* tabf = reinterpret_cast<float*>(tab + BN);
#pragma unroll
        for (int q = 0; q < BN / 32; ++q) tab[lane + 32 * q] = sl[q], tabf[lane + 32 * q] = sf[q];
        if (lane == 0) {
          hdr[0] = e, hdr[1] = pos0, hdr[2] = nvalid;
          hdr[3] = ch * a.NB / chunks, hdr[4] = (ch + 1) * a.NB / chunks;
          hdr[7] = 0;
        }
        __syncwarp();
        mbar_arrive_s(B(B_RFULL + rslot));
        if (++rslot == IRING) rslot = 0, rphase ^= 1;
      }
      if (lane == 0) {
        __threadfence();
        if (atomicAdd(&a.ctr[1], 1) == int(gridDim.x) - 1) {
          __threadfence();
          a.ctr[0] = 0;
          a.ctr[1] = 0;
        }
      }
      return;
    }
    if (warp_idx == 9 || warp_idx == 10) {
      // ---------------------------------------------------------- loader warps (lane 0): w = 0 also loads B
      const int w = warp_idx - 9;
      if (lane == 0) {
        int rslot = 0, ws = 0;
        unsigned rphase = 0, wph = 0, bph = 0;
        const uint32_t wbytes = 64u * uint32_t(wrs), sbytes = 64u * uint32_t(srs);
        // B of an item is loaded after the item's first weight stage is issued (that stage does not
        // depend on the previous item releasing B)
        auto load_b = [&](int pos0) {
          mbar_wait_s(B(B_BEMPTY), bph ^ 1);
          bph ^= 1;
          mbar_expect_tx_s(B(B_BFULL), uint32_t(nkb * a.brows * 128));
          for (int kb = 0; kb < nkb; ++kb) tma_load_2d(sb + OFF_B + kb * BKB, &a.tm_b, kb * 128, pos0, B(B_BFULL));
        };
        for (;;) {
          mbar_wait_s(B(B_RFULL + rslot), rphase);
          const int* hdr = reinterpret_cast<const int*>(sm + OFF_RING + rslot * SLOT);
          const int end = hdr[7], e = hdr[0], pos0 = hdr[1], nb0 = hdr[3], nb1 = hdr[4];
          if (++rslot == IRING) rslot = 0, rphase ^= 1;
          if (end) break;
          for (int nb = nb0; nb < nb1; ++nb) {
            if (w == 0 && nb == nb0 + 1) load_b(pos0);
            mbar_wait_s(B(B_WEMPTY + w * NWS + ws), wph ^ 1);
            const uint32_t dst = sb + OFF_W + (w * NWS + ws) * WSTAGE;
            const long row0 = long(e) * a.H + 128 * nb + 64 * w;
            mbar_expect_tx_s(B(B_WFULL + w * NWS + ws), wbytes + sbytes);
            bulk_load(dst, a.w + row0 * wrs, wbytes, B(B_WFULL + w * NWS + ws));
            bulk_load(dst + wbytes, a.s + row0 * srs, sbytes, B(B_WFULL + w * NWS + ws));
            if (++ws == NWS) ws = 0, wph ^= 1;
          }
          if (w == 0 && nb1 - nb0 == 1) load_b(pos0);
        }
      }
      __syncwarp();
    }
    return;
  }

  // ================================================================ consumer warpgroups
  setmaxnreg_inc<CONSUMER_REGS>();
  const int w = warp_idx >> 2, wq = warp_idx & 3, g = lane >> 2, t = lane & 3;
  const int wtid = tid & 127;
  const int lr0 = 16 * wq + g, lr1 = lr0 + 8;
  const uint32_t stg = sb + OFF_STG + w * STG;
  int rslot = 0, ws = 0;
  unsigned rphase = 0, wph = 0, bph = 0;
  float acc0[BN / 2], acc1[BN / 2];
#ifdef DS41H_MOE_TRACE
  int ntr = 0;
#endif
  for (;;) {
    mbar_wait_s(B(B_RFULL + rslot), rphase);
    const uint32_t slot = sb + OFF_RING + rslot * SLOT;
    if (lds_s32(slot + 28)) break;
    const int nvalid = lds_s32(slot + 8), nb0 = lds_s32(slot + 12), nb1 = lds_s32(slot + 16);
    const int nsel = (nvalid + 15) >> 4;
#ifdef DS41H_MOE_TRACE
    const unsigned long long tb0 = gtimer();
#endif
    mbar_wait_s(B(B_BFULL), bph);
    bph ^= 1;
#ifdef DS41H_MOE_TRACE
    unsigned long long tbw = gtimer() - tb0;
#endif
    for (int nb = nb0; nb < nb1; ++nb) {
      const uint32_t wst = sb + OFF_W + (w * NWS + ws) * WSTAGE;
#ifdef DS41H_MOE_TRACE
      const unsigned long long tw0 = gtimer();
#endif
      mbar_wait_s(B(B_WFULL + w * NWS + ws), wph);
#ifdef DS41H_MOE_TRACE
      const unsigned long long tw1 = gtimer();
      const long long c0 = clock64();
#endif
      // reference exponents of rows lr0 / lr1: max of their scale bytes (lane quad splits the row)
      int m0 = 0, m1 = 0;
      {
        const uint32_t s0 = wst + 64 * wrs + lr0 * srs, s1 = wst + 64 * wrs + lr1 * srs;
        for (int j = t; j < srs; j += 4) m0 = max(m0, lds_u8(s0 + j)), m1 = max(m1, lds_u8(s1 + j));
        m0 = max(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m1 = max(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m0 = max(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = max(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      }
      g2_dispatch(nsel, acc0, acc1, wst, sb + OFF_B, wrs, srs, nfull, nkb, lr0, lr1, m0, m1, t);
#ifdef DS41H_MOE_TRACE
      const unsigned long long tw2 = gtimer();
      const long long c1 = clock64();
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive_s(B(B_WEMPTY + w * NWS + ws));  // stage read (lds only; A lives in registers)
      if (++ws == NWS) ws = 0, wph ^= 1;
      // ------------------------------------------------ epilogue: scale, bf16, stmatrix, stores
      const float ws0 = fp4_ref_scale(m0), ws1 = fp4_ref_scale(m1);
      if constexpr (EPI == EPI_BULK) bulk_wait_read0();  // this lane's previous bulk store has read the staging tile
      wg_bar(w);
      const int m8 = lane >> 3, r8 = lane & 7;
      const uint32_t tabf = slot + 32 + 4 * BN + 8 * t;  // + 32 j: per-token scales of tokens 8j + 2t, +1
#pragma unroll
      for (int i = 0; i < BN / 8; i += 2) {
        if (i >= 2 * nsel) break;
        const int j0 = i, j1 = i + 1;
        float2 h0 = make_float2(1.f, 1.f), h1 = make_float2(1.f, 1.f);
        if constexpr (EPI == EPI_RED) h0 = lds_f2(tabf + 32 * j0), h1 = lds_f2(tabf + 32 * j1);
        const unsigned r0 = pack_bf16((acc0[4 * j0 + 0] + acc1[4 * j0 + 0]) * (ws0 * h0.x), (acc0[4 * j0 + 1] + acc1[4 * j0 + 1]) * (ws0 * h0.y));
        const unsigned r1 = pack_bf16((acc0[4 * j0 + 2] + acc1[4 * j0 + 2]) * (ws1 * h0.x), (acc0[4 * j0 + 3] + acc1[4 * j0 + 3]) * (ws1 * h0.y));
        const unsigned r2 = pack_bf16((acc0[4 * j1 + 0] + acc1[4 * j1 + 0]) * (ws0 * h1.x), (acc0[4 * j1 + 1] + acc1[4 * j1 + 1]) * (ws0 * h1.y));
        const unsigned r3 = pack_bf16((acc0[4 * j1 + 2] + acc1[4 * j1 + 2]) * (ws1 * h1.x), (acc0[4 * j1 + 3] + acc1[4 * j1 + 3]) * (ws1 * h1.y));
        const int tok = 8 * (i + (m8 >> 1)) + r8;
        stsm_x4_trans(stg + tok * PITCH + (16 * wq + 8 * (m8 & 1)) * 2, r0, r1, r2, r3);
      }
      if constexpr (EPI == EPI_BULK) {
        fence_proxy_async();  // stmatrix (generic proxy) -> bulk store reads (async proxy)
        wg_bar(w);
        if (wtid < nvalid) {
          const int row = lds_s32(slot + 32 + 4 * wtid);
          bulk_store(a.out + (long(row) * a.H + 128 * nb + 64 * w), stg + wtid * PITCH, 128u);
          bulk_commit();
        }
      } else if constexpr (EPI == EPI_STG || EPI == EPI_RED) {
        wg_bar(w);
        // 8 lanes per row (16 B each): a warp covers 4 full 128-byte row segments per instruction
        for (int idx = wtid; idx < nvalid * 8; idx += 128) {
          const int rr = idx >> 3, ch = idx & 7;
          const uint4 v = lds128(stg + rr * PITCH + 16 * ch);
          const int row = lds_s32(slot + 32 + 4 * rr);
          __nv_bfloat16* dst = a.out + (long(row) * a.H + 128 * nb + 64 * w + 8 * ch);
          if constexpr (EPI == EPI_STG)
            stg128(dst, v);
          else
            asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
      }
#ifdef DS41H_MOE_TRACE
      if (wtid == 0 && ntr < TR_N) {
        unsigned long long* r = g_mtrace + ((blockIdx.x * 4 + w) * TR_N + ntr) * 8;
        r[0] = tw0, r[1] = tw1, r[2] = tw2, r[3] = gtimer(), r[4] = unsigned(nsel) | (unsigned(nb) << 8), r[5] = tbw, r[6] = c1 - c0, r[7] = 1;
        tbw = 0;
      }
      ++ntr;
#endif
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive_s(B(B_BEMPTY));       // all wgmmas of the item retired (wait<0> in the mainloop)
      mbar_arrive_s(B(B_REMPTY + rslot));  // header / slot table no longer read
    }
    if (++rslot == IRING) rslot = 0, rphase ^= 1;
  }
  if constexpr (EPI == EPI_BULK) bulk_wait0();
}

// ============================================================================================
// W4A16 grouped GEMM, 256-row tiles (GEMM1 variant 11, GEMM2 variant 13) -- moe_gemm16_kernel.
//
// Numerics identical to LD 4 (exact FP4 -> bf16 expansion with the ue8m0 scale folded in, bf16 RS-wgmma m64nNk16,
// fp32 accumulation over the whole K, stock clamped SwiGLU on the fp32 accumulators, unscaled bf16 y in slot order;
// deterministic). Restructured for the bf16 operand traffic and the wgmma issue latency:
// * A tile is 256 weight rows x <= 128 tokens. Each consumer WG owns two m64 row blocks = two independent accumulator
//   chains (2 x 64 registers) that share the token (B) operand of every k16 step, so the bf16 token tile crosses
//   L2 -> SM once per 256 weight rows (LD 4: per 128) and every k16 step issues two independent wgmmas.
//   MODE 1: WG w computes h block hb = 2 nb + w (64 columns): chain 0 = its 64 gate rows, chain 1 = the 64 up rows
//   (accumulator row of logical h column c reads weight row a16_phys(c), as LD 4), so SwiGLU pairs acc0[i] with acc1[i].
//   When I % 128 == 64 the last weight block is a half tile: WG 1 only passes the pipeline stages through.
//   MODE 2: rows [256 nb, 256 nb + 256); chain c of WG w = rows 64 (2 w + c) .. + 63.
// * Pipeline stages carry B (128 rows x 64 bf16, SW128, 16 KB) and A (256 rows x 32 FP4 bytes, 8 KB) only; the ue8m0
//   scales go through a 2-slot ring filled once per 8 K blocks (MODE 1: 16-byte column groups, 64-row TMA boxes; a half
//   tile loads only its 64 + 64 rows) or once per tile (MODE 2: the tile's whole 256 x I/32 block, one bulk copy)
//   -> 7 stages.
// * W16_DEPTH (1 or 3): k16-step wgmma groups (2 wgmmas each) a WG keeps in flight while expanding the next step.
// * Asynchronous epilogue: 2 staging buffers of 64 tokens x 128 columns (WG w: columns 64 w ..); storer warp 10 + w
//   stores 128 B per token and pass (MODE 1: h row segment; MODE 2: slot row segment of one chain).
#ifndef W16_DEPTH
#define W16_DEPTH 1
#endif
#ifndef W16_BN
#define W16_BN 128
#endif
namespace w16 {
constexpr int BN = W16_BN;                   // tokens per tile (max; multiple of 16, <= 128)
static_assert(BN % 16 == 0 && BN <= 128, "W16_BN");
constexpr int W_BSTAGE = BN * 128;           // 16 KB
constexpr int W_ASTAGE = 256 * 32;           // 8 KB
constexpr int STAGE = W_BSTAGE + W_ASTAGE;   // 24 KB (1024-multiple: B stays SW128-aligned)
constexpr int MAXSR = 18;                    // MODE 2 scale-row bytes (I <= 576)
constexpr int S_SLOT = 256 * MAXSR;          // 4608 (MODE 1 uses 256 x 16)
constexpr int NSR = 2;
constexpr int EPI_TOK = 64;
constexpr int PITCH = 272;                   // 128 bf16 + 16 B pad (conflict-free stmatrix)
constexpr int EPI = EPI_TOK * PITCH;         // 17408
constexpr int WRING = 3;
constexpr int SLOT = 32 + 4 * BN;            // int hdr[8] (e, nb, pos0, nvalid<0 = end) | int rows[BN] (MODE 2 slot rows)
constexpr int CONSUMER_REGS = 232, PRODUCER_REGS = 40;
constexpr int FIXED_NOSTAGE = NSR * S_SLOT + 2 * EPI + WRING * SLOT + 2 * (MAX_E + 1) * 4 + 8 * 64 + 160 + 1024;
constexpr int NSTAGE = (227 * 1024 - FIXED_NOSTAGE) / STAGE > 8 ? 8 : (227 * 1024 - FIXED_NOSTAGE) / STAGE;
constexpr int OFF_SR = NSTAGE * STAGE;
constexpr int OFF_EPI = OFF_SR + NSR * S_SLOT;
constexpr int OFF_RING = OFF_EPI + 2 * EPI;
constexpr int OFF_ARR = OFF_RING + WRING * SLOT;  // s_off[E+1] | s_tp[E+1]
constexpr int OFF_BAR = (OFF_ARR + 2 * (MAX_E + 1) * 4 + 7) / 8 * 8;
// barriers: full[NSTAGE] empty[NSTAGE] sfull[NSR] sempty[NSR] rfull[WRING] rempty[WRING] efull[2 wg][2 buf] eempty[2][2]
constexpr int B_FULL = 0, B_EMPTY = NSTAGE, B_SFULL = 2 * NSTAGE, B_SEMPTY = 2 * NSTAGE + NSR, B_RFULL = 2 * NSTAGE + 2 * NSR,
              B_REMPTY = B_RFULL + WRING, B_EFULL = B_REMPTY + WRING, B_EEMPTY = B_EFULL + 4, NBAR = B_EEMPTY + 4;
constexpr int OFF_HDR = OFF_BAR + 8 * NBAR;  // epilogue pass headers int[2 wg][2 buf][8]
constexpr int OFF_BC = OFF_HDR + 128;        // s_bcast int[8]
constexpr int SMEM = 1024 + OFF_BC + 32;
static_assert(SMEM <= 227 * 1024, "w16 smem");
static_assert(NSTAGE >= 4, "w16 pipeline too shallow");
static_assert(OFF_SR % 1024 == 0 && OFF_EPI % 128 == 0 && OFF_RING % 16 == 0 && S_SLOT % 128 == 0, "w16 alignment");
static_assert(2 * CONSUMER_REGS + PRODUCER_REGS <= 504, "setmaxnreg budget");
}  // namespace w16

struct W16Cons {
  uint32_t sb;    // 1024-aligned smem base
  int stage, ss;  // pipeline stage / scale slot
  unsigned phase, sph;
  // byte offsets (inside an A stage / a scale slot) of chain 0's rows for accumulator rows g and g + 8; chain 1 is at
  // a fixed row offset (MODE 1: + 128 = the up rows; MODE 2: + 64)
  uint32_t aoff0, aoff1, soff0, soff1;
  int t;
  int epass;
#ifdef DS41H_MOE_TRACE
  long long twait;
#endif
};

// One tile's mainloop (both chains), N tokens (16..128).
template <int MODE, int N>
__device__ __forceinline__ void w16_mainloop(W16Cons& c, const GemmArgs& a, float (&acc0_all)[w16::BN / 2], float (&acc1_all)[w16::BN / 2]) {
  using namespace w16;
  constexpr int D = W16_DEPTH;
  static_assert(D == 1 || D == 3, "W16_DEPTH");
  constexpr int NBUF = D + 1;
  constexpr int C1 = MODE == 1 ? 128 : 64;  // row offset of chain 1
  float(&acc0)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc0_all[0]);
  float(&acc1)[N / 2] = *reinterpret_cast<float(*)[N / 2]>(&acc1_all[0]);
  const int nkb = a.K / 64;
  // scale-ring slot groups: MODE 1 8 K blocks, MODE 2 the whole tile (no runtime modulo in the loop)
  auto sgroup_first = [&](int kb) { return MODE == 1 ? (kb & 7) == 0 : kb == 0; };
  auto sgroup_last = [&](int kb) { return (MODE == 1 && (kb & 7) == 7) || kb == nkb - 1; };
  const unsigned rot = 4u * unsigned(c.t);
  unsigned w[2][2][2];  // rotated FP4 words [chain][row g / g+8][word j]
  unsigned m[2][2];     // bf16x2 scale multipliers [chain][row]
  unsigned F[NBUF][2][4];
  auto bar = [&](int i) { return c.sb + OFF_BAR + 8 * i; };
  auto acquire = [&](int kb) {
#ifdef DS41H_MOE_TRACE
    const long long tw0 = clock64();
#endif
    mbar_wait_s(bar(B_FULL + c.stage), c.phase);
    if (sgroup_first(kb)) mbar_wait_s(bar(B_SFULL + c.ss), c.sph);
#ifdef DS41H_MOE_TRACE
    c.twait += clock64() - tw0;
#endif
  };
  auto load_block = [&](int kb) {
    const uint32_t sa = c.sb + c.stage * STAGE + W_BSTAGE;
    const uint32_t ss = c.sb + OFF_SR + c.ss * S_SLOT + (MODE == 1 ? 2 * (kb & 7) : 2 * kb);
    const int rs = MODE == 1 ? 16 : a.s_rstride;
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const uint2 q0 = lds64(sa + c.aoff0 + ch * C1 * 32), q1 = lds64(sa + c.aoff1 + ch * C1 * 32);
      w[ch][0][0] = __funnelshift_r(q0.x, q0.x, rot), w[ch][0][1] = __funnelshift_r(q0.y, q0.y, rot);
      w[ch][1][0] = __funnelshift_r(q1.x, q1.x, rot), w[ch][1][1] = __funnelshift_r(q1.y, q1.y, rot);
      m[ch][0] = fp4_bf16_mult(lds_u8(ss + c.soff0 + ch * C1 * rs));
      m[ch][1] = fp4_bf16_mult(lds_u8(ss + c.soff1 + ch * C1 * rs));
    }
    // the scale slot is only read here (into registers): release it after the group's last block
    if (sgroup_last(kb)) {
      fence_proxy_async();  // generic scale reads before the TMA refill of the slot (cross-proxy WAR, see small16)
      __syncwarp();
      if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_SEMPTY + c.ss));
      if (++c.ss == NSR) c.ss = 0, c.sph ^= 1;
    }
  };
  auto expand = [&](int s, unsigned (&Fb)[2][4]) {
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const unsigned x0 = w[ch][0][s >> 1], x1 = w[ch][1][s >> 1];
#ifdef DS41H_MOE_ABL_NOCVT  // timing experiment only (wrong numerics)
      Fb[ch][0] = x0 ^ m[ch][0], Fb[ch][2] = x0 + m[ch][0], Fb[ch][1] = x1 ^ m[ch][1], Fb[ch][3] = x1 + m[ch][1];
      continue;
#endif
      if ((s & 1) == 0) {
        Fb[ch][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x0), m[ch][0]), Fb[ch][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x0), m[ch][0]);
        Fb[ch][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x1), m[ch][1]), Fb[ch][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x1), m[ch][1]);
      } else {
        Fb[ch][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x0), m[ch][0]), Fb[ch][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x0), m[ch][0]);
        Fb[ch][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x1), m[ch][1]), Fb[ch][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x1), m[ch][1]);
      }
    }
  };
  auto release_prev_stage = [&]() {  // the block before the current stage is fully retired
    if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_EMPTY + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));
  };
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  acquire(0);
  load_block(0);
  expand(0, F[0]);
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    const uint64_t desc = desc_sw128(c.sb + c.stage * STAGE);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      wgmma_fence();
#ifndef DS41H_MOE_ABL_NOMMA
      WgmmaRS16<N>::mma(acc0, F[s % NBUF][0], desc + 2 * s, (s > 0 || kb > 0) ? 1 : 0);
      WgmmaRS16<N>::mma(acc1, F[s % NBUF][1], desc + 2 * s, (s > 0 || kb > 0) ? 1 : 0);
#endif
      wgmma_commit();
      wgmma_wait<D>();  // groups <= this step - D retired: F[(s + 1) % NBUF] free; at s == D - 1 block kb - 1 retired
#pragma unroll
      for (int ch = 0; ch < 2; ++ch)
#pragma unroll
        for (int j = 0; j < 4; ++j) reg_fence(F[(s + 1) % NBUF][ch][j]);
      if (s == D - 1 && kb > 0) release_prev_stage();
      if (s < 3) {
        expand(s + 1, F[(s + 1) % NBUF]);
      } else {
        if (++c.stage == NSTAGE) c.stage = 0, c.phase ^= 1;
        if (kb + 1 < nkb) {
          acquire(kb + 1);
          load_block(kb + 1);
          expand(0, F[0]);
        }
      }
    }
  }
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  release_prev_stage();  // last block's stage
}

template <int MODE, int NS = w16::BN / 16>
__device__ __forceinline__ void w16_dispatch(int nsel, W16Cons& c, const GemmArgs& a, float (&acc0)[w16::BN / 2], float (&acc1)[w16::BN / 2]) {
  if constexpr (NS >= 1) {
    if (nsel == NS) {
      w16_mainloop<MODE, 16 * NS>(c, a, acc0, acc1);
      return;
    }
    w16_dispatch<MODE, NS - 1>(nsel, c, a, acc0, acc1);
  }
}

// MODE 1 half tile, WG 1: pass the tile's pipeline stages / scale slots through without computing
__device__ __forceinline__ void w16_skip_tile(W16Cons& c, const GemmArgs& a) {
  using namespace w16;
  const int nkb = a.K / 64;
  for (int kb = 0; kb < nkb; ++kb) {
    mbar_wait_s(c.sb + OFF_BAR + 8 * (B_FULL + c.stage), c.phase);
    if (kb % 8 == 0) mbar_wait_s(c.sb + OFF_BAR + 8 * (B_SFULL + c.ss), c.sph);
    __syncwarp();
    if ((threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_EMPTY + c.stage));
    if (kb % 8 == 7 || kb == nkb - 1) {
      if ((threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_SEMPTY + c.ss));
      if (++c.ss == NSR) c.ss = 0, c.sph ^= 1;
    }
    if (++c.stage == NSTAGE) c.stage = 0, c.phase ^= 1;
  }
}

// staging buffer of this WG's next epilogue pass (waits until its storer released it)
__device__ __forceinline__ uint32_t w16_epi_acquire(const W16Cons& c, int wg) {
  using namespace w16;
  const int b = c.epass & 1;
  mbar_wait_s(c.sb + OFF_BAR + 8 * (B_EEMPTY + 2 * wg + b), ((c.epass >> 1) & 1) ^ 1);
  return c.sb + OFF_EPI + b * EPI;
}
// publish a staged pass: hdr {ring slot, first token, tokens, flags (1 = last pass of the tile, 2 = end), nb, pos0, chain}
__device__ __forceinline__ void w16_epi_post(W16Cons& c, int wg, int rslot, int c0, int np, int flags, int nb, int pos0, int chain) {
  using namespace w16;
  const int b = c.epass & 1;
  if ((threadIdx.x & 127) == 0) {
    int* h = reinterpret_cast<int*>(__cvta_shared_to_generic(c.sb + OFF_HDR + 32 * (2 * wg + b)));
    h[0] = rslot, h[1] = c0, h[2] = np, h[3] = flags, h[4] = nb, h[5] = pos0, h[6] = chain;
  }
  __syncwarp();
  if ((threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_EFULL + 2 * wg + b));
  ++c.epass;
}

// Epilogue of one tile: passes of 64 tokens (MODE 2: one pass per chain and 64 tokens). Each WG stages its 64
// columns (accumulator rows) of the pass as [token][64 w + col] with stmatrix (transposed).
template <int MODE>
__device__ __forceinline__ void w16_epilogue(W16Cons& c, const GemmArgs& a, const float (&acc0)[w16::BN / 2], const float (&acc1)[w16::BN / 2],
                                             int nvalid, int nb, int pos0, int rslot) {
  using namespace w16;
  const int lane = threadIdx.x & 31, m8 = lane >> 3, r8 = lane & 7;
  const int wg = (threadIdx.x >> 7) & 1, wq = (threadIdx.x >> 5) & 3;
  const int npass = (nvalid + EPI_TOK - 1) / EPI_TOK;
#pragma unroll
  for (int p = 0; p < (BN + EPI_TOK - 1) / EPI_TOK; ++p) {
    if (p >= npass) break;
    const int c0 = p * EPI_TOK, np = min(EPI_TOK, nvalid - c0);
    const int nch = (np + 7) >> 3;  // valid 8-token chunks of this pass
#pragma unroll
    for (int ch = 0; ch < (MODE == 1 ? 1 : 2); ++ch) {
      const float(&v)[BN / 2] = ch == 0 ? acc0 : acc1;
      const uint32_t ebase = w16_epi_acquire(c, wg);
#pragma unroll
      for (int i = 0; i < EPI_TOK / 8; i += 2) {
        if (i >= nch || c0 / 8 + i >= BN / 8) break;
#ifdef DS41H_MOE_ABL_NOEPI  // timing experiment only: no epilogue math / staging
        break;
#endif
        const int j0 = min(c0 / 8 + i, BN / 8 - 2), j1 = j0 + 1;  // (clamp: compile-time bound for BN < 128)
        unsigned r[4];  // (row g, chunk j0), (row g+8, j0), (row g, j1), (row g+8, j1); each 2 tokens
        if constexpr (MODE == 1) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int idx = 4 * (q < 2 ? j0 : j1) + 2 * (q & 1);
            float h[2];
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {  // stock clamped SiLU (see epi_pass, LD 4): gate = acc0, up = acc1
              const float gt = fminf(acc0[idx + hv], a.limit);
              const float up = fmaxf(fminf(acc1[idx + hv], a.limit), -a.limit);
#ifdef DS41H_MOE_ABL_FASTSILU
              h[hv] = gt / (1.f + __expf(-gt)) * up;
#else
              h[hv] = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
#endif
            }
            r[q] = pack_bf16(h[0], h[1]);
          }
        } else {
          r[0] = pack_bf16(v[4 * j0 + 0], v[4 * j0 + 1]), r[1] = pack_bf16(v[4 * j0 + 2], v[4 * j0 + 3]);
          r[2] = pack_bf16(v[4 * j1 + 0], v[4 * j1 + 1]), r[3] = pack_bf16(v[4 * j1 + 2], v[4 * j1 + 3]);
        }
        const int tok = 8 * (i + (m8 >> 1)) + r8;
        stsm_x4_trans(ebase + tok * PITCH + (64 * wg + 16 * wq + 8 * (m8 & 1)) * 2, r[0], r[1], r[2], r[3]);
      }
      w16_epi_post(c, wg, rslot, c0, np, (p == npass - 1 && ch == (MODE == 1 ? 0 : 1)) ? 1 : 0, nb, pos0, ch);
    }
  }
}

// Storer warp of WG w: 16-byte global stores of the staged passes, 128 B per token (4 loads in flight per lane).
template <int MODE>
__device__ __noinline__ void w16_storer(const GemmArgs& a, int w, int lane, uint32_t sb) {
  using namespace w16;
  for (int pass = 0;; ++pass) {
    const int b = pass & 1;
#ifdef DS41H_MOE_TRACE
    const unsigned long long ts0 = gtimer();
#endif
    mbar_wait_s(sb + OFF_BAR + 8 * (B_EFULL + 2 * w + b), (pass >> 1) & 1);
#ifdef DS41H_MOE_TRACE
    const unsigned long long ts1 = gtimer();
#endif
    const uint32_t h = sb + OFF_HDR + 32 * (2 * w + b);
    const int rs = lds_s32(h), c0 = lds_s32(h + 4), np = lds_s32(h + 8), flags = lds_s32(h + 12);
    if (flags & 2) break;
    const int nb = lds_s32(h + 16), pos0 = lds_s32(h + 20), chain = lds_s32(h + 24);
    const uint32_t buf = sb + OFF_EPI + b * EPI + 128 * w;
    const uint32_t rows = sb + OFF_RING + rs * SLOT + 32;
    // MODE 1: h row pos0 + c0 + r, columns 64 (2 nb + w) ..; MODE 2: slot row rows[c0 + r], columns 256 nb + 64 (2 w + chain) ..
    __nv_bfloat16* const base = a.out + (MODE == 1 ? long(pos0 + c0) * a.out_stride + 64 * (2 * nb + w) : 256 * nb + 64 * (2 * w + chain));
    const int n = np * 8;
    for (int i0 = lane; i0 < n; i0 += 128) {
      uint4 val[4];
      long ro[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int idx = i0 + 32 * q;
        if (idx < n) {
          val[q] = lds128(buf + (idx >> 3) * PITCH + 16 * (idx & 7));
          ro[q] = MODE == 1 ? long(idx >> 3) * a.out_stride : long(lds_s32(rows + 4 * (c0 + (idx >> 3)))) * a.out_stride;
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int idx = i0 + 32 * q;
        if (idx < n) {
#ifdef DS41H_MOE_ABL_NOSTORE
          if (val[q].x == 0x12345678u && ro[q] < 0)
#endif
            stg128(base + ro[q] + 8 * (idx & 7), val[q]);
        }
      }
    }
    __syncwarp();
#ifdef DS41H_MOE_TRACE
    if (lane == 0 && pass < TR_N) {
      unsigned long long* r = g_mtrace + ((blockIdx.x * 4 + 2 + w) * TR_N + pass) * 8;
      r[0] = ts0, r[1] = ts1, r[2] = gtimer(), r[3] = r[2], r[4] = unsigned(np), r[5] = 0, r[6] = 0, r[7] = 3;
    }
#endif
    if (lane == 0) {
      mbar_arrive_s(sb + OFF_BAR + 8 * (B_EEMPTY + 2 * w + b));          // staging buffer free
      if (flags & 1) mbar_arrive_s(sb + OFF_BAR + 8 * (B_REMPTY + rs));  // last pass of the tile: ring slot released
    }
  }
}

// hybrid remainder mode (a.tsplit > 0): only each expert's last partial chunk of tsplit_bn rows, if it has <= tsplit rows
__device__ __forceinline__ int w16_ntiles(int n, const GemmArgs& a) {
  if (a.tsplit <= 0) return (n + w16::BN - 1) / w16::BN;
  const int r = n % a.tsplit_bn;
  return (r > 0 && r <= a.tsplit) ? 1 : 0;
}
__device__ __forceinline__ int w16_base(int off, int n, const GemmArgs& a) {
  return a.tsplit <= 0 ? off : off + (n / a.tsplit_bn) * a.tsplit_bn;
}

template <int MODE>
__global__ void __launch_bounds__(NTHREADS, 1) moe_gemm16_kernel(const __grid_constant__ GemmArgs a) {
  using namespace w16;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sb = smem_u32(sm);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + OFF_BAR);
  auto bar = [&](int i) { return sb + OFF_BAR + 8 * i; };
  int* s_off = reinterpret_cast<int*>(sm + OFF_ARR);
  int* s_tp = s_off + (MAX_E + 1);
  int* s_bcast = reinterpret_cast<int*>(sm + OFF_BC);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) mbar_init(&bars[B_FULL + s], 1), mbar_init(&bars[B_EMPTY + s], 8);
    for (int s = 0; s < NSR; ++s) mbar_init(&bars[B_SFULL + s], 1), mbar_init(&bars[B_SEMPTY + s], 8);
    for (int s = 0; s < WRING; ++s) mbar_init(&bars[B_RFULL + s], 32), mbar_init(&bars[B_REMPTY + s], 10);  // 8 consumer warps + 2 storers
    for (int i = 0; i < 4; ++i) mbar_init(&bars[B_EFULL + i], 4), mbar_init(&bars[B_EEMPTY + i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  pdl_wait();
  asm volatile("fence.proxy.async.global;" ::: "memory");  // generic writes of the previous kernels -> TMA reads
  const int warp_idx = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int nkb = a.K / 64;
  if (warp_idx >= 8) {
    // ================================================================ producer warpgroup
    setmaxnreg_dec<PRODUCER_REGS>();
    const int pt = tid - 256;
    const int E = a.E;
    {  // expert row offsets and exclusive prefix of token tiles ceil(n_e / BN) (as moe_gemm_kernel)
      const int per = (E + 127) / 128;
      int loc = 0;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          const int o0 = a.off[e], o1 = a.off[e + 1];
          s_off[e] = o0;
          loc += w16_ntiles(o1 - o0, a);
        }
      }
      if (pt == 0) s_off[E] = a.off[E];
      const int lane = pt & 31, w = pt >> 5;
      int inc = loc;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (lane == 31) s_bcast[4 + w] = inc;
      producer_bar();
      int base = 0;
      for (int k = 0; k < w; ++k) base += s_bcast[4 + k];
      int run = base + inc - loc;
      for (int k = 0; k < per; ++k) {
        const int e = pt * per + k;
        if (e < E) {
          s_tp[e] = run;
          run += w16_ntiles(s_off[e + 1] - s_off[e], a);
        }
      }
      if (pt == 127) s_tp[E] = base + inc;
    }
    producer_bar();
    if (warp_idx == 8) {
      // ---------------------------------------------------------------- scheduler warp
      const int lane = pt;
      const int total = s_tp[E] * a.NB;
      int rslot = 0;
      unsigned rphase = 0;
      for (;;) {
        int tile = 0;
        if (lane == 0) tile = atomicAdd(&a.ctr[0], 1);
        tile = __shfl_sync(0xffffffffu, tile, 0);
        mbar_wait_s(bar(B_REMPTY + rslot), rphase ^ 1);
        int* hdr = reinterpret_cast<int*>(sm + OFF_RING + rslot * SLOT);
        if (tile >= total) {
          if (lane == 0) hdr[3] = -1;
          __syncwarp();
          mbar_arrive(&bars[B_RFULL + rslot]);
          break;
        }
        int lo = 0, hi = E;
        while (hi - lo > 1) {
          const int mid = (lo + hi) >> 1;
          if (s_tp[mid] * a.NB <= tile)
            lo = mid;
          else
            hi = mid;
        }
        const int e = lo;
        const int nt = s_tp[e + 1] - s_tp[e];
        const int local = tile - s_tp[e] * a.NB;
        const int nb = local / nt;
        const int pos0 = w16_base(s_off[e], s_off[e + 1] - s_off[e], a) + (local - nb * nt) * BN;
        const int nvalid = min(BN, s_off[e + 1] - pos0);
        if (lane == 0) hdr[0] = e, hdr[1] = nb, hdr[2] = pos0, hdr[3] = nvalid;
        if constexpr (MODE == 2) {
          int* rows = hdr + 8;
#pragma unroll
          for (int q = 0; q < (BN + 31) / 32; ++q) {
            const int cc = lane + 32 * q;
            if (cc < BN) rows[cc] = cc < nvalid ? a.perm_slot[pos0 + cc] : 0;
          }
        }
        __syncwarp();
        mbar_arrive(&bars[B_RFULL + rslot]);
        if (++rslot == WRING) rslot = 0, rphase ^= 1;
      }
      if (lane == 0) {  // all CTAs past their last tile fetch -> the last one resets the counters for the next launch
        __threadfence();
        if (atomicAdd(&a.ctr[1], 1) == int(gridDim.x) - 1) {
          __threadfence();
          a.ctr[0] = 0;
          a.ctr[1] = 0;
        }
      }
      return;
    }
    if (warp_idx == 9) {
      // -------------------------------------------------------------- TMA warp (lane 0)
      if ((tid & 31) != 0) return;
      const int rows_e = int(a.w_estride / a.w_rstride);
      const uint32_t sbytes = MODE == 1 ? 4096u : uint32_t(256 * a.s_rstride);
      int stage = 0, rslot = 0, ss = 0;
      unsigned phase = 0, rphase = 0, sph = 0;
      for (;;) {
        mbar_wait_s(bar(B_RFULL + rslot), rphase);
        const int* hdr = reinterpret_cast<const int*>(sm + OFF_RING + rslot * SLOT);
        const int nvalid = hdr[3];
        if (nvalid < 0) break;
        const int e = hdr[0], nb = hdr[1], pos0 = hdr[2];
        if (++rslot == WRING) rslot = 0, rphase ^= 1;
        const int nbox = (nvalid + a.brows - 1) / a.brows;
        const int w0 = e * rows_e + (MODE == 1 ? 128 * nb : 256 * nb), w1 = w0 + a.I;
        const uint32_t bbytes = uint32_t(nbox * a.brows * 128);
        const int nbw = MODE == 1 && 128 * nb + 128 > a.I ? 1 : 2;  // MODE 1: 64-row weight / scale boxes (half tile: 1)
        for (int kb = 0; kb < nkb; ++kb) {
          if (MODE == 1 ? (kb & 7) == 0 : kb == 0) {
            mbar_wait_s(bar(B_SEMPTY + ss), sph ^ 1);
            const uint32_t sd = sb + OFF_SR + ss * S_SLOT, sf = bar(B_SFULL + ss);
            if constexpr (MODE == 1) {
#ifdef DS41H_MOE_ABL_NOLOADW  // timing experiment only: no weight / scale traffic
              mbar_expect_tx_s(sf, 0u);
              for (int j = 0; j < 0; ++j) {
#else
              mbar_expect_tx_s(sf, 2048u * nbw);
              for (int j = 0; j < nbw; ++j) {
#endif
                tma_load_2d(sd + 1024 * j, &a.tm_s, 16 * (kb >> 3), w0 + 64 * j, sf);         // gate rows: 16-byte group of 8 K blocks
                tma_load_2d(sd + 2048 + 1024 * j, &a.tm_s, 16 * (kb >> 3), w1 + 64 * j, sf);  // up rows
              }
            } else {
#ifdef DS41H_MOE_ABL_NOLOADW
              mbar_expect_tx_s(sf, 0u);
#else
              mbar_expect_tx_s(sf, sbytes);
              bulk_load(sd, a.s + long(w0) * a.s_rstride, sbytes, sf);  // the tile's 256 scale rows (contiguous)
#endif
            }
            if (++ss == NSR) ss = 0, sph ^= 1;
          }
          mbar_wait_s(bar(B_EMPTY + stage), phase ^ 1);
          const uint32_t st = sb + stage * STAGE, fb = bar(B_FULL + stage);
#if defined(DS41H_MOE_ABL_NOLOADB) || defined(DS41H_MOE_ABL_NOLOADW)  // timing experiments only
#ifdef DS41H_MOE_ABL_NOLOADB
          constexpr bool LB = false;
#else
          constexpr bool LB = true;
#endif
#ifdef DS41H_MOE_ABL_NOLOADW
          constexpr bool LW = false;
#else
          constexpr bool LW = true;
#endif
          mbar_expect_tx_s(fb, (LB ? bbytes : 0u) + (LW ? uint32_t(MODE == 1 ? 4096 * nbw : W_ASTAGE) : 0u));
          if (LB)
            for (int j = 0; j < nbox; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
          if (!LW) {
          } else if constexpr (MODE == 1) {
#else
          mbar_expect_tx_s(fb, bbytes + uint32_t(MODE == 1 ? 4096 * nbw : W_ASTAGE));
          for (int j = 0; j < nbox; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
          if constexpr (MODE == 1) {
#endif
            for (int j = 0; j < nbw; ++j) {
              tma_load_2d(st + W_BSTAGE + 2048 * j, &a.tm_w, kb * 32, w0 + 64 * j, fb);         // 64 gate rows
              tma_load_2d(st + W_BSTAGE + 4096 + 2048 * j, &a.tm_w, kb * 32, w1 + 64 * j, fb);  // 64 up rows
            }
          } else {
            tma_load_2d(st + W_BSTAGE, &a.tm_w, kb * 32, w0, fb);  // 256 rows
          }
          if (++stage == NSTAGE) stage = 0, phase ^= 1;
        }
      }
      return;
    }
    // storer warps 10, 11
    w16_storer<MODE>(a, warp_idx - 10, tid & 31, sb);
    return;
  }

  // ================================================================== consumer warpgroups
  setmaxnreg_inc<CONSUMER_REGS>();
  W16Cons c;
  c.sb = sb;
  c.stage = 0, c.ss = 0, c.phase = 0, c.sph = 0, c.epass = 0;
  c.t = tid & 3;
  const int wg = warp_idx >> 2;
  {
    const int wq = warp_idx & 3, g = (tid & 31) >> 2;
    int r0, r1;
    if constexpr (MODE == 1) {  // gate rows of logical h columns 16 wq + g (+ 8) of h block 2 nb + wg
      r0 = 64 * wg + a16_phys(16 * wq + g), r1 = 64 * wg + a16_phys(16 * wq + 8 + g);
    } else {
      r0 = 128 * wg + 16 * wq + g, r1 = r0 + 8;
    }
    const int rs = MODE == 1 ? 16 : a.s_rstride;
    c.aoff0 = uint32_t(r0 * 32 + 8 * c.t), c.aoff1 = uint32_t(r1 * 32 + 8 * c.t);
    c.soff0 = uint32_t(r0 * rs + (c.t >> 1)), c.soff1 = uint32_t(r1 * rs + (c.t >> 1));
  }
  int rslot = 0;
  unsigned rphase = 0;
  float acc0[BN / 2], acc1[BN / 2];
#ifdef DS41H_MOE_TRACE
  int ntr = 0;
#endif
  for (;;) {
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr_ring = gtimer();
#endif
    mbar_wait_s(bar(B_RFULL + rslot), rphase);
    const uint32_t slot = sb + OFF_RING + rslot * SLOT;
    const int nvalid = lds_s32(slot + 12);
    if (nvalid < 0) break;
    const int nb = lds_s32(slot + 4), pos0 = lds_s32(slot + 8);
    const int nsel = (nvalid + 15) >> 4;
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr1 = gtimer();
    const long long tc0 = clock64();
    c.twait = 0;
#endif
    // MODE 1 half tile (I % 128 == 64, last block): h block 2 nb + 1 does not exist -> WG 1 idles
    const bool idle = MODE == 1 && wg == 1 && 128 * nb + 128 > a.I;
    if (idle) {
      w16_skip_tile(c, a);
    } else {
      w16_dispatch<MODE>(nsel, c, a, acc0, acc1);
    }
#ifdef DS41H_MOE_TRACE
    const unsigned long long tr2 = gtimer();
    const long long tc1 = clock64();
#endif
    if (idle) {  // empty last pass: the storer releases the ring slot
      w16_epi_acquire(c, wg);
      w16_epi_post(c, wg, rslot, 0, 0, 1, nb, pos0, 0);
    } else {
      w16_epilogue<MODE>(c, a, acc0, acc1, nvalid, nb, pos0, rslot);
    }
    __syncwarp();
#ifdef DS41H_MOE_TRACE
    if ((tid & 127) == 0 && ntr < TR_N) {  // record as moe_gemm_kernel (nsel | nb << 8 | idle << 16)
      unsigned long long* r = g_mtrace + ((blockIdx.x * 4 + (tid >> 7)) * TR_N + ntr) * 8;
      r[0] = tr_ring, r[1] = tr1, r[2] = tr2, r[3] = gtimer(), r[4] = unsigned(nsel) | (unsigned(nb) << 8) | (unsigned(idle) << 16),
      r[5] = c.twait, r[6] = tc1 - tc0, r[7] = 2;
    }
    ++ntr;
#endif
    if ((tid & 31) == 0) mbar_arrive_s(bar(B_REMPTY + rslot));
    if (++rslot == WRING) rslot = 0, rphase ^= 1;
  }
  // tell this WG's storer warp to exit
  w16_epi_acquire(c, wg);
  w16_epi_post(c, wg, 0, 0, 0, 2, 0, 0, 0);
}

// ------------------------------------------------------------------------------------------------------------------
// W4A16 small-batch grouped GEMM ("small16"; GEMM1 / GEMM2 variant 16). For experts with few routed
// rows (the decoder-replay layers: ~2 rows per expert at 128 rows per request; the narrow remainder tiles of the full
// prefill layers in hybrid mode) the wgmma kernels are bound by a fixed ~190-clock cost per k16 step and tile
// (dependent RS-wgmma + FP4 expansion chain), i.e. they stream the weights at ~1.1-1.5 TB/s. small16 streams the
// weights with plain warp-level mma.sync m16n8k16 instead: 4 consumer warps x 2 m16 row tiles = 128 weight rows per
// unit, <= 64 tokens (8 n8 groups) per unit, independent warps (no warpgroup-wide wgmma dependency), 2 CTAs per SM.
//
// Numerics = LD 4: the same exact FP4 -> bf16 expansion with the ue8m0 scale folded in (load_block / expand code and
// fragment layout of run_mainloop16; an RS-wgmma warp fragment IS the m16n8k16 A fragment), bf16 x bf16 products
// accumulated in fp32 over the whole K, stock clamped SwiGLU on the fp32 accumulators (accurate expf), unscaled bf16 y
// in slot order (moe_combine applies the routing weights). Only the fp32 summation order inside a k16 step may differ
// (mma.sync vs wgmma) -> within 1 bf16 ulp of LD 4 (measured: bit-identical in situ and in the unit tests). Deterministic (static
// unit striding, no atomics) since the cross-proxy fence before every stage / scale-slot release (s16_mainloop).
//
// Unit = (expert e, token chunk of <= 64 routed rows, weight block nb): MODE 1: h columns [64 nb, 64 nb + 64) (warp w,
// tile i: 8 logical columns L = 16 w + 8 i + g; A rows g = gate row a16_phys(L), g + 8 = up row I + a16_phys(L), as
// LD 4's sr0 / sr1 = 64 + sr0, so each thread holds gate and up of the same column); MODE 2: y rows [128 nb, + 128)
// (warp w, tile i: rows 32 w + 16 i + g, + 8). Consecutive units share the token chunk (nb fastest) -> L2 reuse.
// Pipeline (warp 4 lane 0 = TMA producer): per 64-wide K block one stage = 128 weight rows x 32 FP4 bytes (4 KB) +
// ceil(n / 16) token boxes of 16 rows x 64 bf16 (SW128, 2 KB each; token rows in routed-row order); ue8m0 scales in a
// 2-slot ring (MODE 1: 16-byte column group of 8 K blocks for the 128 rows; MODE 2: the unit's 128 x I/32 block).
// Hybrid (a.tsplit > 0): only each expert's remainder rows [tsplit_bn * floor(n / tsplit_bn), n) if n % tsplit_bn is in
// (0, tsplit] (the LD 4 kernel skips exactly those, see ld_ntiles).
// NT = tokens per unit (max): 64 (2 CTAs / SM) or 16 (tiny batches: 16 accumulators per thread, 4 CTAs / SM for latency
// hiding, 6 KB stages; experts with > 16 rows take several units).
namespace sm16 {
constexpr int NCW = 4;                   // consumer warps
constexpr int NTHR = 32 * (NCW + 1);     // + 1 producer warp
constexpr int W_ST = 128 * 32;           // FP4 bytes of a stage
constexpr int S_SLOT = 128 * 18;         // MODE 2: 128 rows x I/32 (I <= 576); MODE 1: 128 x 16
constexpr int NSR = 2;
constexpr int PITCH1 = 144, PITCH2 = 272;  // staging rows: 64 / 128 bf16 + 16 B
}  // namespace sm16
template <int NT>
struct S16Cfg {
  static_assert(NT == 16 || NT == 64, "small16 NT");
  static constexpr int X_ST = NT * 128;                    // token rows of a stage (64 bf16 each)
  static constexpr int STAGE = sm16::W_ST + X_ST;          // 12 / 6 KB (1024-multiple: the token boxes stay SW128-aligned)
  static constexpr int NST = 6;
  static constexpr int CTAS = NT == 16 ? 4 : 2;            // CTAs per SM
  static constexpr int EPI = NT * sm16::PITCH2;
  static constexpr int OFF_SR = NST * STAGE;
  static constexpr int OFF_EPI = OFF_SR + sm16::NSR * sm16::S_SLOT;
  static constexpr int OFF_ARR = OFF_EPI + EPI;           // s_off[E+1] | s_tp[E+1]
  static constexpr int OFF_BAR = (OFF_ARR + 2 * (MAX_E + 1) * 4 + 7) / 8 * 8;
  static constexpr int B_FULL = 0, B_EMPTY = NST, B_SFULL = 2 * NST, B_SEMPTY = 2 * NST + sm16::NSR, NBAR = 2 * NST + 2 * sm16::NSR;
  static constexpr int SMEM = 1024 + OFF_BAR + 8 * NBAR;
  static_assert(CTAS * (SMEM + 1024) <= 228 * 1024, "small16: CTAs per SM");
  static_assert(OFF_SR % 1024 == 0 && OFF_EPI % 16 == 0 && sm16::S_SLOT % 128 == 0, "small16 alignment");
};

__device__ __forceinline__ void mma16816_bf16(float (&d)[4], const unsigned (&A)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t a, unsigned& r0, unsigned& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}

// chunks (units per weight block) of an expert with n routed rows
template <int NT>
__device__ __forceinline__ int s16_nchunks(int n, const GemmArgs& a) {
  if (a.tsplit <= 0) return (n + NT - 1) / NT;
  const int r = n % a.tsplit_bn;
  return (r > 0 && r <= a.tsplit) ? (r + NT - 1) / NT : 0;
}
struct S16Unit {
  int e, nb, pos0, n;
};
template <int NT>
__device__ __forceinline__ S16Unit s16_decode(int u, const int* s_off, const int* s_tp, const GemmArgs& a) {
  int lo = 0, hi = a.E;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (s_tp[mid] * a.NB <= u)
      lo = mid;
    else
      hi = mid;
  }
  S16Unit r;
  r.e = lo;
  const int local = u - s_tp[lo] * a.NB, chunk = local / a.NB;
  r.nb = local - chunk * a.NB;
  const int o0 = s_off[lo], n = s_off[lo + 1] - o0;
  const int base = a.tsplit <= 0 ? o0 : o0 + (n / a.tsplit_bn) * a.tsplit_bn;
  r.pos0 = base + NT * chunk;
  r.n = min(NT, s_off[lo + 1] - r.pos0);
  return r;
}

struct S16Cons {
  uint32_t sb;
  int stage, ss;
  unsigned phase, sph;
  uint32_t aoff[2][2], soff[2][2];  // [tile][row g / g + 8]: byte offsets inside a stage / scale slot
  int t;
};

// One unit's K loop with NG n8 token groups; acc[tile][group][4]
template <int MODE, int NT, int NG>
__device__ __forceinline__ void s16_mainloop(S16Cons& c, const GemmArgs& a, float (&acc)[2][NT / 8][4]) {
  using namespace sm16;
  using C = S16Cfg<NT>;
  const int nkb = a.K / 64;
  const int lane = threadIdx.x & 31;
  const unsigned rot = 4u * unsigned(c.t);
  auto bar = [&](int i) { return c.sb + C::OFF_BAR + 8 * i; };
  auto sgroup_first = [&](int kb) { return MODE == 1 ? (kb & 7) == 0 : kb == 0; };
  auto sgroup_last = [&](int kb) { return (MODE == 1 && (kb & 7) == 7) || kb == nkb - 1; };
  // ldmatrix row address of this lane for token group pair gp, k16 step s (matrix mi = lane >> 3: tokens 16 gp + 8 (mi >> 1)
  // + (lane & 7), 16-byte chunk 2 s + (mi & 1), SW128: chunk ^ (row & 7) with row & 7 == lane & 7)
  const int mi = lane >> 3, r7 = lane & 7;
  const uint32_t xrow = uint32_t((8 * (mi >> 1) + r7) * 128);
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int q = 0; q < NT / 8; ++q)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][q][v] = 0.f;
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    mbar_wait_s(bar(C::B_FULL + c.stage), c.phase);
    if (sgroup_first(kb)) mbar_wait_s(bar(C::B_SFULL + c.ss), c.sph);
    const uint32_t st = c.sb + c.stage * C::STAGE, xs = st + W_ST;
    const uint32_t ss = c.sb + C::OFF_SR + c.ss * S_SLOT + (MODE == 1 ? 2 * (kb & 7) : 2 * kb);
    unsigned w[2][2][2], m[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint2 q0 = lds64(st + c.aoff[i][0]), q1 = lds64(st + c.aoff[i][1]);
      w[i][0][0] = __funnelshift_r(q0.x, q0.x, rot), w[i][0][1] = __funnelshift_r(q0.y, q0.y, rot);
      w[i][1][0] = __funnelshift_r(q1.x, q1.x, rot), w[i][1][1] = __funnelshift_r(q1.y, q1.y, rot);
      m[i][0] = fp4_bf16_mult(lds_u8(ss + c.soff[i][0]));
      m[i][1] = fp4_bf16_mult(lds_u8(ss + c.soff[i][1]));
    }
    if (sgroup_last(kb)) {  // the scale slot is only read here (into registers)
      fence_proxy_async();  // generic smem reads before the producer's TMA (async proxy) rewrites the slot (see below)
      __syncwarp();
      if (lane == 0) mbar_arrive_s(bar(C::B_SEMPTY + c.ss));
      if (++c.ss == NSR) c.ss = 0, c.sph ^= 1;
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      unsigned A[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const unsigned x0 = w[i][0][s >> 1], x1 = w[i][1][s >> 1];
        if ((s & 1) == 0) {
          A[i][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x0), m[i][0]), A[i][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x0), m[i][0]);
          A[i][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x1), m[i][1]), A[i][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x1), m[i][1]);
        } else {
          A[i][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x0), m[i][0]), A[i][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x0), m[i][0]);
          A[i][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x1), m[i][1]), A[i][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x1), m[i][1]);
        }
      }
      const uint32_t xa = xs + xrow + uint32_t(((2 * s + (mi & 1)) ^ r7) << 4);
#pragma unroll
      for (int gp = 0; gp < (NG + 1) / 2; ++gp) {
        unsigned b0, b1, b2 = 0, b3 = 0;
        if (2 * gp + 1 < NG)
          ldsm_x4(xa + gp * 16 * 128, b0, b1, b2, b3);
        else
          ldsm_x2(xa + gp * 16 * 128, b0, b1);  // (lanes 16..31 addresses unused)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          mma16816_bf16(acc[i][2 * gp], A[i], b0, b1);
          if (2 * gp + 1 < NG) mma16816_bf16(acc[i][2 * gp + 1], A[i], b2, b3);
        }
      }
    }
    // Cross-proxy WAR fence: the stage was read with GENERIC loads (lds / ldmatrix) and is refilled
    // by TMA (ASYNC proxy) as soon as all warps arrived; without fence.proxy.async the async write could overtake the generic
    // reads -> the consumer saw the next round's weights / tokens. Measured (T = 1000 / 2000,
    // concurrent side-stream GEMM load): 29 bad of 1200 calls without the fence, 0 of 1200 with it (same session).
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) mbar_arrive_s(bar(C::B_EMPTY + c.stage));
    if (++c.stage == C::NST) c.stage = 0, c.phase ^= 1;
  }
}

template <int MODE, int NT, int NG = NT / 8>
__device__ __forceinline__ void s16_dispatch(int ng, S16Cons& c, const GemmArgs& a, float (&acc)[2][NT / 8][4]) {
  if constexpr (NG >= 1) {
    if (ng == NG) {
      s16_mainloop<MODE, NT, NG>(c, a, acc);
      return;
    }
    s16_dispatch<MODE, NT, NG - 1>(ng, c, a, acc);
  }
}

template <int MODE, int NT>
__global__ void __launch_bounds__(sm16::NTHR, S16Cfg<NT>::CTAS) moe_small16_kernel(const __grid_constant__ GemmArgs a) {
  using namespace sm16;
  using C = S16Cfg<NT>;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sb = smem_u32(sm);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + C::OFF_BAR);
  auto bar = [&](int i) { return sb + C::OFF_BAR + 8 * i; };
  int* s_off = reinterpret_cast<int*>(sm + C::OFF_ARR);
  int* s_tp = s_off + (MAX_E + 1);
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (tid == 0) {
    for (int s = 0; s < C::NST; ++s) mbar_init(&bars[C::B_FULL + s], 1), mbar_init(&bars[C::B_EMPTY + s], NCW);
    for (int s = 0; s < NSR; ++s) mbar_init(&bars[C::B_SFULL + s], 1), mbar_init(&bars[C::B_SEMPTY + s], NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  pdl_wait();  // off / activations come from the preceding kernels
  asm volatile("fence.proxy.async.global;" ::: "memory");
  const int E = a.E;
  // per-expert row offsets and the exclusive prefix of units per weight block (chunks), every CTA the same
  for (int e = tid; e <= E; e += NTHR) s_off[e] = a.off[e];
  __syncthreads();
  if (warp == 0) {
    int carry = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane;
      const int v = e < E ? s16_nchunks<NT>(s_off[e + 1] - s_off[e], a) : 0;
      int inc = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (e < E) s_tp[e] = carry + inc - v;
      carry += __shfl_sync(0xffffffffu, inc, 31);
    }
    if (lane == 0) s_tp[E] = carry;
  }
  __syncthreads();
  const int total = s_tp[E] * a.NB;
  const int nkb = a.K / 64;
  if (warp == NCW) {
    // ------------------------------------------------------------------ TMA producer (lane 0)
    if (lane != 0) return;
    const int rows_e = int(a.w_estride / a.w_rstride);
    int stage = 0, ss = 0;
    unsigned phase = 0, sph = 0;
    for (int u = blockIdx.x; u < total; u += gridDim.x) {
      const S16Unit un = s16_decode<NT>(u, s_off, s_tp, a);
      const int w0 = un.e * rows_e + (MODE == 1 ? 64 * un.nb : 128 * un.nb);
      const int br = a.brows;  // token rows per box: 16 (fewer only if the activation tensor has < 16 rows: one box)
      const int nbx = (un.n + br - 1) / br;
      for (int kb = 0; kb < nkb; ++kb) {
        if (MODE == 1 ? (kb & 7) == 0 : kb == 0) {
          mbar_wait_s(bar(C::B_SEMPTY + ss), sph ^ 1);
          const uint32_t sd = sb + C::OFF_SR + ss * S_SLOT, sf = bar(C::B_SFULL + ss);
          if constexpr (MODE == 1) {
            mbar_expect_tx_s(sf, 2048u);
            tma_load_2d(sd, &a.tm_s, 16 * (kb >> 3), w0, sf);                // gate rows: 16-byte group of 8 K blocks
            tma_load_2d(sd + 1024, &a.tm_s, 16 * (kb >> 3), w0 + a.I, sf);  // up rows
          } else {
            const uint32_t sbytes = uint32_t(128 * a.s_rstride);
            mbar_expect_tx_s(sf, sbytes);
            bulk_load(sd, a.s + long(w0) * a.s_rstride, sbytes, sf);  // the unit's 128 scale rows (contiguous)
          }
          if (++ss == NSR) ss = 0, sph ^= 1;
        }
        mbar_wait_s(bar(C::B_EMPTY + stage), phase ^ 1);
        const uint32_t st = sb + stage * C::STAGE, fb = bar(C::B_FULL + stage);
        mbar_expect_tx_s(fb, uint32_t(W_ST + nbx * br * 128));
        if constexpr (MODE == 1) {
          tma_load_2d(st, &a.tm_w, kb * 32, w0, fb);                // 64 gate rows
          tma_load_2d(st + 2048, &a.tm_w, kb * 32, w0 + a.I, fb);  // 64 up rows
        } else {
          tma_load_2d(st, &a.tm_w, kb * 32, w0, fb);  // 128 rows
        }
        for (int j = 0; j < nbx; ++j) tma_load_2d(st + W_ST + br * 128 * j, &a.tm_b, kb * 128, un.pos0 + br * j, fb);
        if (++stage == C::NST) stage = 0, phase ^= 1;
      }
    }
    return;
  }
  // -------------------------------------------------------------------- consumer warps
  S16Cons c;
  c.sb = sb, c.stage = 0, c.ss = 0, c.phase = 0, c.sph = 0;
  const int g = lane >> 2, t = lane & 3;
  c.t = t;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int r0, r1;  // stage rows of A rows g / g + 8
    if constexpr (MODE == 1) {
      r0 = a16_phys(16 * warp + 8 * i + g);  // gate row of logical column L (stage rows 0..63)
      r1 = 64 + r0;                          // its up row (stage rows 64..127)
    } else {
      r0 = 32 * warp + 16 * i + g;
      r1 = r0 + 8;
    }
    c.aoff[i][0] = uint32_t(r0 * 32 + 8 * t), c.aoff[i][1] = uint32_t(r1 * 32 + 8 * t);
    const int rs = MODE == 1 ? 16 : a.s_rstride;
    c.soff[i][0] = uint32_t(r0 * rs + (t >> 1)), c.soff[i][1] = uint32_t(r1 * rs + (t >> 1));
  }
  const uint32_t epi = sb + C::OFF_EPI;
  for (int u = blockIdx.x; u < total; u += gridDim.x) {
    const S16Unit un = s16_decode<NT>(u, s_off, s_tp, a);
    float acc[2][NT / 8][4];
    s16_dispatch<MODE, NT>((un.n + 7) >> 3, c, a, acc);
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile free (previous unit copied out)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int q = 0; q < NT / 8; ++q) {
        if (8 * q >= un.n) break;
        const int tok = 8 * q + 2 * t;
        if constexpr (MODE == 1) {
          const int L = 16 * warp + 8 * i + g;
#pragma unroll
          for (int hv = 0; hv < 2; ++hv) {  // stock clamped SiLU (LD 4 epilogue): row g = gate, row g + 8 = up
            const float gt = fminf(acc[i][q][hv], a.limit);
            const float up = fmaxf(fminf(acc[i][q][2 + hv], a.limit), -a.limit);
            const float h = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
            const __nv_bfloat16 hb = __float2bfloat16_rn(h);
            asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH1 + 2 * L)),
                         "h"(*reinterpret_cast<const unsigned short*>(&hb)) : "memory");
          }
        } else {
          const int col = 32 * warp + 16 * i + g;
#pragma unroll
          for (int hv = 0; hv < 2; ++hv)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
              const __nv_bfloat16 yb = __float2bfloat16_rn(acc[i][q][2 * rr + hv]);
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH2 + 2 * (col + 8 * rr))),
                           "h"(*reinterpret_cast<const unsigned short*>(&yb)) : "memory");
            }
        }
      }
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile complete
    if constexpr (MODE == 1) {  // h rows pos0 + r, columns [64 nb, 64 nb + 64): 8 x 16 B per row
      for (int idx = tid; idx < un.n * 8; idx += 128) {
        const int r = idx >> 3, ch = idx & 7;
        stg128(a.out + long(un.pos0 + r) * a.out_stride + 64 * un.nb + 8 * ch, lds128(epi + r * PITCH1 + 16 * ch));
      }
    } else {  // slot rows perm_slot[pos0 + r], columns [128 nb, 128 nb + 128): 16 x 16 B per row
      for (int idx = tid; idx < un.n * 16; idx += 128) {
        const int r = idx >> 4, ch = idx & 15;
        const long row = a.perm_slot[un.pos0 + r];
        stg128(a.out + row * a.out_stride + 128 * un.nb + 8 * ch, lds128(epi + r * PITCH2 + 16 * ch));
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------
// small16 with WIDE WEIGHT BOXES / WIDER UNITS ("small16k"; GEMM1 / GEMM2 variants 18..23,
// + 8 = chunk-fastest unit order 26..31; forward flag 0x500 | sub << 11 [| 0x4000]). The per-K-block math and the fp32
// accumulation order of every output element are exactly those of moe_small16_kernel (same A fragments, same mma.sync per
// token group in the same K order) -> bit-identical outputs; only data movement / scheduling change:
//  * KPS = 1 / 2 / 4 consecutive 64-wide K blocks per pipeline stage = ONE weight box of 128 rows x 32 KPS bytes (64- / 128-byte
//    row segments instead of 32: fewer TMA row requests, mbarrier handoffs and fences per weight byte) + KPS token tiles. Wide
//    boxes are stored with the TMA 64B / 128B swizzle so the consumer's 8-byte row loads stay bank-conflict-free: 16-byte chunk c
//    of stage row r sits at chunk c ^ key(r), key = (r >> 1) & 3 (64-byte rows) or r & 7 (128-byte rows) (cute Swizzle<2,4,3> /
//    <3,4,3> on the byte offset). A K tail (GEMM2: K = 576 = 9 blocks) loads the full box (TMA zero-fills the bytes beyond the
//    row; the transaction count is the full box) and the consumer skips the missing blocks (no extra products).
//  * NT = 16 / 32 tokens per unit: decode batches route 8 correlated tokens per stream -> hot experts get > 16 rows at c32+;
//    32-token units expand (FP4 -> bf16 + scale: ~70% of small16's instructions; the kernel is issue-bound, not HBM-bound) each
//    weight once per 32 instead of 16 routed rows.
//  * a.s16_cf: the token chunks of one (expert, weight block) are consecutive units (chunk fastest) -> neighbouring CTAs at the
//    same time -> they share the weight stream through L2.
// Same cross-proxy rule as small16: fence.proxy.async before every stage / scale-slot release.
template <int NT_, int KPS_, int NST_, int CTAS_>
struct S16K {
  static constexpr int NT = NT_;
  static constexpr int KPS = KPS_;
  static constexpr int RB = 32 * KPS;     // weight bytes per row per stage
  static constexpr int W_ST = 128 * RB;   // 4 / 8 / 16 KB
  static constexpr int X_KB = NT * 128;   // token tile of one K block: NT rows x 64 bf16 (SW128)
  static constexpr int X_ST = KPS * X_KB;
  static constexpr int STAGE = W_ST + X_ST;
  static constexpr int NST = NST_;
  static constexpr int CTAS = CTAS_;
  static constexpr int EPI = NT * sm16::PITCH2;
  static constexpr int OFF_SR = NST * STAGE;
  static constexpr int OFF_EPI = OFF_SR + sm16::NSR * sm16::S_SLOT;
  static constexpr int OFF_ARR = OFF_EPI + EPI;  // s_off[E+1] | s_tp[E+1]
  static constexpr int OFF_BAR = (OFF_ARR + 2 * (MAX_E + 1) * 4 + 7) / 8 * 8;
  static constexpr int B_FULL = 0, B_EMPTY = NST, B_SFULL = 2 * NST, B_SEMPTY = 2 * NST + sm16::NSR, NBAR = 2 * NST + 2 * sm16::NSR;
  static constexpr int SMEM = 1024 + OFF_BAR + 8 * NBAR;
  static_assert(NT == 16 || NT == 32, "small16k NT");
  static_assert(KPS == 1 || KPS == 2 || KPS == 4, "small16k: KPS 1, 2 or 4 (divides the 8-block scale groups)");
  static_assert(NST >= 2, "small16k stages");
  static_assert(CTAS * (SMEM + 1024) <= 228 * 1024, "small16k: CTAs per SM");
  static_assert(STAGE % 1024 == 0 && OFF_SR % 1024 == 0 && OFF_EPI % 16 == 0, "small16k alignment");
};

struct S16KCons {
  uint32_t sb;
  int stage, ss;
  unsigned phase, sph;
  uint32_t abase[2][2];  // [tile][row g / g + 8]: row r * RB + 8 (t & 1) inside the stage's weight box
  uint32_t akey[2][2];   // ((t >> 1) ^ key(r)) << 4: chunk of K block j = 2 j ^ ((t >> 1) ^ key) -> offset (32 j) ^ akey
  uint32_t soff[2][2];   // byte offsets inside a scale slot (as small16)
  int t;
};

// One unit's K loop (NG n8 token groups); acc[tile][group][4]
template <int MODE, class C, int NG>
__device__ __forceinline__ void s16k_mainloop(S16KCons& c, const GemmArgs& a, float (&acc)[2][C::NT / 8][4]) {
  using namespace sm16;
  constexpr int KPS = C::KPS;
  const int nkb = a.K / 64, nks = (nkb + KPS - 1) / KPS;
  const int lane = threadIdx.x & 31;
  const unsigned rot = 4u * unsigned(c.t);
  auto bar = [&](int i) { return c.sb + C::OFF_BAR + 8 * i; };
  auto sgroup_first = [&](int kb) { return MODE == 1 ? (kb & 7) == 0 : kb == 0; };
  auto sgroup_last = [&](int kb) { return (MODE == 1 && (kb & 7) == 7) || kb == nkb - 1; };
  const int mi = lane >> 3, r7 = lane & 7;
  const uint32_t xrow = uint32_t((8 * (mi >> 1) + r7) * 128);
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int q = 0; q < C::NT / 8; ++q)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][q][v] = 0.f;
#pragma unroll 1
  for (int ks = 0; ks < nks; ++ks) {
    mbar_wait_s(bar(C::B_FULL + c.stage), c.phase);
    const uint32_t st = c.sb + c.stage * C::STAGE;
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
      const int kb = ks * KPS + j;
      if (KPS > 1 && kb >= nkb) break;  // K tail (GEMM2): the box's zero-filled rest is not read
      if (sgroup_first(kb)) mbar_wait_s(bar(C::B_SFULL + c.ss), c.sph);
      const uint32_t xs = st + C::W_ST + j * C::X_KB;
      const uint32_t ss = c.sb + C::OFF_SR + c.ss * S_SLOT + (MODE == 1 ? 2 * (kb & 7) : 2 * kb);
      unsigned w[2][2][2], m[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint2 q0 = lds64(st + c.abase[i][0] + ((32u * j) ^ c.akey[i][0]));
        const uint2 q1 = lds64(st + c.abase[i][1] + ((32u * j) ^ c.akey[i][1]));
        w[i][0][0] = __funnelshift_r(q0.x, q0.x, rot), w[i][0][1] = __funnelshift_r(q0.y, q0.y, rot);
        w[i][1][0] = __funnelshift_r(q1.x, q1.x, rot), w[i][1][1] = __funnelshift_r(q1.y, q1.y, rot);
        m[i][0] = fp4_bf16_mult(lds_u8(ss + c.soff[i][0]));
        m[i][1] = fp4_bf16_mult(lds_u8(ss + c.soff[i][1]));
      }
      if (sgroup_last(kb)) {  // the scale slot is only read here (into registers)
        fence_proxy_async();  // generic smem reads before the producer's TMA (async proxy) rewrites the slot
        __syncwarp();
        if (lane == 0) mbar_arrive_s(bar(C::B_SEMPTY + c.ss));
        if (++c.ss == NSR) c.ss = 0, c.sph ^= 1;
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        unsigned A[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned x0 = w[i][0][s >> 1], x1 = w[i][1][s >> 1];
          if ((s & 1) == 0) {
            A[i][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x0), m[i][0]), A[i][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x0), m[i][0]);
            A[i][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x1), m[i][1]), A[i][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x1), m[i][1]);
          } else {
            A[i][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x0), m[i][0]), A[i][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x0), m[i][0]);
            A[i][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x1), m[i][1]), A[i][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x1), m[i][1]);
          }
        }
        const uint32_t xa = xs + xrow + uint32_t(((2 * s + (mi & 1)) ^ r7) << 4);
#pragma unroll
        for (int gp = 0; gp < (NG + 1) / 2; ++gp) {
          unsigned b0, b1, b2 = 0, b3 = 0;
          if (2 * gp + 1 < NG)
            ldsm_x4(xa + gp * 16 * 128, b0, b1, b2, b3);
          else
            ldsm_x2(xa + gp * 16 * 128, b0, b1);  // (lanes 16..31 addresses unused)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma16816_bf16(acc[i][2 * gp], A[i], b0, b1);
            if (2 * gp + 1 < NG) mma16816_bf16(acc[i][2 * gp + 1], A[i], b2, b3);
          }
        }
      }
    }
    // cross-proxy WAR fence before the stage goes back to the TMA producer (see s16_mainloop)
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) mbar_arrive_s(bar(C::B_EMPTY + c.stage));
    if (++c.stage == C::NST) c.stage = 0, c.phase ^= 1;
  }
}

template <int MODE, class C, int NG = C::NT / 8>
__device__ __forceinline__ void s16k_dispatch(int ng, S16KCons& c, const GemmArgs& a, float (&acc)[2][C::NT / 8][4]) {
  if constexpr (NG >= 1) {
    if (ng == NG) {
      s16k_mainloop<MODE, C, NG>(c, a, acc);
      return;
    }
    s16k_dispatch<MODE, C, NG - 1>(ng, c, a, acc);
  }
}

// small16k unit decode: small16's order, or (a.s16_cf) the chunks of one (expert, weight block) consecutive. Either order visits
// every (expert, chunk, block) unit exactly once and every unit writes its own output tile -> identical results.
template <int NT>
__device__ __forceinline__ S16Unit s16k_decode(int u, const int* s_off, const int* s_tp, const GemmArgs& a) {
  if (!a.s16_cf) return s16_decode<NT>(u, s_off, s_tp, a);
  int lo = 0, hi = a.E;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (s_tp[mid] * a.NB <= u)
      lo = mid;
    else
      hi = mid;
  }
  S16Unit r;
  r.e = lo;
  const int nch = s_tp[lo + 1] - s_tp[lo];  // > 0: this expert owns unit u
  const int local = u - s_tp[lo] * a.NB, nb = local / nch, chunk = local - nb * nch;
  r.nb = nb;
  const int o0 = s_off[lo], n = s_off[lo + 1] - o0;
  const int base = a.tsplit <= 0 ? o0 : o0 + (n / a.tsplit_bn) * a.tsplit_bn;
  r.pos0 = base + NT * chunk;
  r.n = min(NT, s_off[lo + 1] - r.pos0);
  return r;
}

template <int MODE, int NT, int KPS, int NST, int CTAS>
__global__ void __launch_bounds__(sm16::NTHR, CTAS) moe_small16k_kernel(const __grid_constant__ GemmArgs a) {
  using namespace sm16;
  using C = S16K<NT, KPS, NST, CTAS>;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sb = smem_u32(sm);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + C::OFF_BAR);
  auto bar = [&](int i) { return sb + C::OFF_BAR + 8 * i; };
  int* s_off = reinterpret_cast<int*>(sm + C::OFF_ARR);
  int* s_tp = s_off + (MAX_E + 1);
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (tid == 0) {
    for (int s = 0; s < C::NST; ++s) mbar_init(&bars[C::B_FULL + s], 1), mbar_init(&bars[C::B_EMPTY + s], NCW);
    for (int s = 0; s < NSR; ++s) mbar_init(&bars[C::B_SFULL + s], 1), mbar_init(&bars[C::B_SEMPTY + s], NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  pdl_wait();  // off / activations come from the preceding kernels
  asm volatile("fence.proxy.async.global;" ::: "memory");
  const int E = a.E;
  for (int e = tid; e <= E; e += NTHR) s_off[e] = a.off[e];
  __syncthreads();
  if (warp == 0) {
    int carry = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane;
      const int v = e < E ? s16_nchunks<NT>(s_off[e + 1] - s_off[e], a) : 0;
      int inc = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (e < E) s_tp[e] = carry + inc - v;
      carry += __shfl_sync(0xffffffffu, inc, 31);
    }
    if (lane == 0) s_tp[E] = carry;
  }
  __syncthreads();
  const int total = s_tp[E] * a.NB;
  const int nkb = a.K / 64, nks = (nkb + KPS - 1) / KPS;
  if (warp == NCW) {
    // ------------------------------------------------------------------ TMA producer (lane 0)
    if (lane != 0) return;
    const int rows_e = int(a.w_estride / a.w_rstride);
    int stage = 0, ss = 0;
    unsigned phase = 0, sph = 0;
    for (int u = blockIdx.x; u < total; u += gridDim.x) {
      const S16Unit un = s16k_decode<NT>(u, s_off, s_tp, a);
      const int w0 = un.e * rows_e + (MODE == 1 ? 64 * un.nb : 128 * un.nb);
      const int br = a.brows;  // token rows per box: 16 (fewer only if the activation tensor has < 16 rows: one box)
      const int nbx = (un.n + br - 1) / br;
      for (int ks = 0; ks < nks; ++ks) {
        const int kb0 = ks * KPS;
        if (MODE == 1 ? (kb0 & 7) == 0 : ks == 0) {
          mbar_wait_s(bar(C::B_SEMPTY + ss), sph ^ 1);
          const uint32_t sd = sb + C::OFF_SR + ss * S_SLOT, sf = bar(C::B_SFULL + ss);
          if constexpr (MODE == 1) {
            mbar_expect_tx_s(sf, 2048u);
            tma_load_2d(sd, &a.tm_s, 16 * (kb0 >> 3), w0, sf);                // gate rows: 16-byte group of 8 K blocks
            tma_load_2d(sd + 1024, &a.tm_s, 16 * (kb0 >> 3), w0 + a.I, sf);  // up rows
          } else {
            const uint32_t sbytes = uint32_t(128 * a.s_rstride);
            mbar_expect_tx_s(sf, sbytes);
            bulk_load(sd, a.s + long(w0) * a.s_rstride, sbytes, sf);  // the unit's 128 scale rows (contiguous)
          }
          if (++ss == NSR) ss = 0, sph ^= 1;
        }
        const int nv = nkb - kb0 < KPS ? nkb - kb0 : KPS;  // valid K blocks of this stage
        mbar_wait_s(bar(C::B_EMPTY + stage), phase ^ 1);
        const uint32_t st = sb + stage * C::STAGE, fb = bar(C::B_FULL + stage);
        mbar_expect_tx_s(fb, uint32_t(C::W_ST + nv * nbx * br * 128));
        if constexpr (MODE == 1) {
          tma_load_2d(st, &a.tm_w, kb0 * 32, w0, fb);                     // 64 gate rows x 32 KPS bytes
          tma_load_2d(st + 64 * C::RB, &a.tm_w, kb0 * 32, w0 + a.I, fb);  // 64 up rows
        } else {
          tma_load_2d(st, &a.tm_w, kb0 * 32, w0, fb);  // 128 rows (tail stage: bytes beyond the row -> 0)
        }
        for (int j = 0; j < nv; ++j)
          for (int jj = 0; jj < nbx; ++jj)
            tma_load_2d(st + C::W_ST + j * C::X_KB + br * 128 * jj, &a.tm_b, (kb0 + j) * 128, un.pos0 + br * jj, fb);
        if (++stage == C::NST) stage = 0, phase ^= 1;
      }
    }
    return;
  }
  // -------------------------------------------------------------------- consumer warps
  S16KCons c;
  c.sb = sb, c.stage = 0, c.ss = 0, c.phase = 0, c.sph = 0;
  const int g = lane >> 2, t = lane & 3;
  c.t = t;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int r[2];  // stage rows of A rows g / g + 8 (as small16)
    if constexpr (MODE == 1) {
      r[0] = a16_phys(16 * warp + 8 * i + g);  // gate row of logical column L (stage rows 0..63)
      r[1] = 64 + r[0];                        // its up row (stage rows 64..127)
    } else {
      r[0] = 32 * warp + 16 * i + g;
      r[1] = r[0] + 8;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int key = KPS == 1 ? 0 : KPS == 2 ? ((r[h] >> 1) & 3) : (r[h] & 7);
      c.abase[i][h] = uint32_t(r[h] * C::RB + 8 * (t & 1));
      c.akey[i][h] = uint32_t(((t >> 1) ^ key) << 4);
      const int rs = MODE == 1 ? 16 : a.s_rstride;
      c.soff[i][h] = uint32_t(r[h] * rs + (t >> 1));
    }
  }
  const uint32_t epi = sb + C::OFF_EPI;
  for (int u = blockIdx.x; u < total; u += gridDim.x) {
    const S16Unit un = s16k_decode<NT>(u, s_off, s_tp, a);
    float acc[2][NT / 8][4];
    s16k_dispatch<MODE, C>((un.n + 7) >> 3, c, a, acc);
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile free (previous unit copied out)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int q = 0; q < NT / 8; ++q) {
        if (8 * q >= un.n) break;
        const int tok = 8 * q + 2 * t;
        if constexpr (MODE == 1) {
          const int L = 16 * warp + 8 * i + g;
#pragma unroll
          for (int hv = 0; hv < 2; ++hv) {  // stock clamped SiLU (LD 4 epilogue): row g = gate, row g + 8 = up
            const float gt = fminf(acc[i][q][hv], a.limit);
            const float up = fmaxf(fminf(acc[i][q][2 + hv], a.limit), -a.limit);
            const float h = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
            const __nv_bfloat16 hb = __float2bfloat16_rn(h);
            asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH1 + 2 * L)),
                         "h"(*reinterpret_cast<const unsigned short*>(&hb)) : "memory");
          }
        } else {
          const int col = 32 * warp + 16 * i + g;
#pragma unroll
          for (int hv = 0; hv < 2; ++hv)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
              const __nv_bfloat16 yb = __float2bfloat16_rn(acc[i][q][2 * rr + hv]);
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH2 + 2 * (col + 8 * rr))),
                           "h"(*reinterpret_cast<const unsigned short*>(&yb)) : "memory");
            }
        }
      }
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile complete
    if constexpr (MODE == 1) {  // h rows pos0 + r, columns [64 nb, 64 nb + 64): 8 x 16 B per row
      for (int idx = tid; idx < un.n * 8; idx += 128) {
        const int r = idx >> 3, ch = idx & 7;
        stg128(a.out + long(un.pos0 + r) * a.out_stride + 64 * un.nb + 8 * ch, lds128(epi + r * PITCH1 + 16 * ch));
      }
    } else {  // slot rows perm_slot[pos0 + r], columns [128 nb, 128 nb + 128): 16 x 16 B per row
      for (int idx = tid; idx < un.n * 16; idx += 128) {
        const int r = idx >> 4, ch = idx & 15;
        const long row = a.perm_slot[un.pos0 + r];
        stg128(a.out + row * a.out_stride + 128 * un.nb + 8 * ch, lds128(epi + r * PITCH2 + 16 * ch));
      }
    }
  }
}

#ifndef DS41H_DEVICE_ONLY
// ---------------------------------------------------------------------------------------- host
int num_sms() {
  static const int n = [] {
    int dev = 0, v = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    return v;
  }();
  return n;
}

template <int MODE, int BNMAX, int WIN, int LD = 0>
void launch_gemm(const GemmArgs& args) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  opt_in_smem<moe_gemm_kernel<MODE, BNMAX, WIN, LD>>(C::SMEM);
  launch_pdl("moe_gemm", moe_gemm_kernel<MODE, BNMAX, WIN, LD>, dim3(num_sms()), dim3(NTHREADS), C::SMEM, args);
}

// W4A16 cluster-multicast variants (moe_gemm_kernel CS > 1): grid = co-resident clusters x CS
template <int MODE, int BNMAX, int WIN, int LD, int CS>
void launch_gemm_cl(const GemmArgs& args) {
  using C = Cfg<MODE, BNMAX, WIN, LD>;
  opt_in_smem<moe_gemm_kernel<MODE, BNMAX, WIN, LD, CS>>(C::SMEM);
  static const int nclusters = [] {
    cudaLaunchAttribute at{};
    at.id = cudaLaunchAttributeClusterDimension;
    at.val.clusterDim.x = CS, at.val.clusterDim.y = 1, at.val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_sms() / CS * CS);
    cfg.blockDim = dim3(NTHREADS);
    cfg.dynamicSmemBytes = C::SMEM;
    cfg.attrs = &at;
    cfg.numAttrs = 1;
    int n = 0;
    const cudaError_t e = cudaOccupancyMaxActiveClusters(&n, moe_gemm_kernel<MODE, BNMAX, WIN, LD, CS>, &cfg);
    TORCH_CHECK(e == cudaSuccess && n > 0, "moe_gemm cluster variant: no co-resident cluster of ", CS, " CTAs (", cudaGetErrorString(e), ")");
    if (std::getenv("DS41H_MOE_VERBOSE")) std::fprintf(stderr, "ds41h moe_gemm MODE %d: %d co-resident clusters of %d CTAs\n", MODE, n, CS);
    return n;
  }();
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = CS, attrs[0].val.clusterDim.y = 1, attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(nclusters * CS);
  config.blockDim = dim3(NTHREADS);
  config.dynamicSmemBytes = C::SMEM;
  config.stream = c10::cuda::getCurrentCUDAStream();
  config.attrs = attrs;
  config.numAttrs = 2;
  const cudaError_t err = cudaLaunchKernelEx(&config, moe_gemm_kernel<MODE, BNMAX, WIN, LD, CS>, args);
  TORCH_CHECK(err == cudaSuccess, "moe_gemm cluster launch failed: ", cudaGetErrorString(err));
}

PFN_cuTensorMapEncodeTiled_v12000 tmap_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q) == cudaSuccess && ptr,
                "cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

void encode_u8_2d(CUtensorMap* map, const void* base, uint64_t inner, uint64_t rows, uint32_t box_inner, uint32_t box_rows, bool sw128,
                  const char* name) {
  const cuuint64_t dims[2] = {cuuint64_t(inner), cuuint64_t(rows)}, strides[1] = {cuuint64_t(inner)};
  const cuuint32_t box[2] = {box_inner, box_rows}, es[2] = {1u, 1u};
  const CUresult r = tmap_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, sw128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "moe_gemm: cuTensorMapEncodeTiled(", name, ") failed: ", int(r));
}
// any swizzle mode (small16k weight boxes: 64B / 128B swizzle)
void encode_u8_2d_swz(CUtensorMap* map, const void* base, uint64_t inner, uint64_t rows, uint32_t box_inner, uint32_t box_rows,
                      CUtensorMapSwizzle swz, const char* name) {
  const cuuint64_t dims[2] = {cuuint64_t(inner), cuuint64_t(rows)}, strides[1] = {cuuint64_t(inner)};
  const cuuint32_t box[2] = {box_inner, box_rows}, es[2] = {1u, 1u};
  const CUresult r = tmap_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "moe_gemm: cuTensorMapEncodeTiled(", name, ") failed: ", int(r));
}

// tensor maps of the TMA loader (LD = 1); act_rows = rows of the activation operand (routed-row order)
// a16 (W4A16, LD 4): bf16 activations (2K bytes per row, 64 per box row), 32-byte FP4 boxes (64-wide K blocks)
void make_tma_maps(int mode, GemmArgs& a, int act_rows, bool row_boxes = false, bool a16 = false) {
  const int rows_total = int(a.s_total / a.s_rstride);  // E * weight rows per expert
  a.brows = row_boxes ? 1 : (act_rows < 64 ? act_rows : 64);
  encode_u8_2d(&a.tm_b, a.act, uint64_t(a16 ? 2 * a.K : a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
  encode_u8_2d(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), a16 ? 32u : 64u, mode == 1 ? 64u : 128u, false, "w");
  if (mode == 1) encode_u8_2d(&a.tm_s, a.s, uint64_t(a.s_rstride), uint64_t(rows_total), 16u, 64u, false, "s");
}

// W4A16 256-row-tile kernel (moe_gemm16_kernel; GEMM1 variant 11, GEMM2 variant 13). MODE 1 needs I % 64 == 0;
// MODE 2 needs H % 256 == 0 and scale rows of <= 18 bytes (I <= 576); otherwise the LD 4 kernel (9 / 11) runs.
bool gemm16_ok(int mode, const GemmArgs& a) {
  const int rows_e = int(a.w_estride / a.w_rstride);
  if (a.K % 64 != 0 || a.E > MAX_E || (reinterpret_cast<uintptr_t>(a.s) & 15) != 0) return false;
  if (mode == 1) return rows_e % 128 == 0 && (rows_e / 2) % 64 == 0;
  return rows_e % 256 == 0 && a.s_rstride <= w16::MAXSR;
}

template <int MODE>
void launch_gemm16(GemmArgs a, int act_rows) {
  const int rows_total = int(a.s_total / a.s_rstride);
  const int rows_e = int(a.w_estride / a.w_rstride);
  a.brows = act_rows < 64 ? act_rows : 64;
  encode_u8_2d(&a.tm_b, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
  encode_u8_2d(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), 32u, MODE == 1 ? 64u : 256u, false, "w");
  if (MODE == 1) encode_u8_2d(&a.tm_s, a.s, uint64_t(a.s_rstride), uint64_t(rows_total), 16u, 64u, false, "s");
  a.NB = MODE == 1 ? (rows_e / 2 + 127) / 128 : rows_e / 256;
  opt_in_smem<moe_gemm16_kernel<MODE>>(w16::SMEM);
  launch_pdl("moe_gemm16", moe_gemm16_kernel<MODE>, dim3(num_sms()), dim3(NTHREADS), w16::SMEM, a);
}

// W4A16 small-batch kernel (moe_small16_kernel; GEMM1 / GEMM2 variant 16, hybrid engine 1). MODE 1: I % 64 == 0;
// MODE 2: H % 128 == 0, 128 scale rows <= sm16::S_SLOT bytes (I <= 576); 16-byte aligned scales.
bool small16_ok(int mode, const GemmArgs& a) {
  const int rows_e = int(a.w_estride / a.w_rstride);
  if (a.K % 64 != 0 || a.E > MAX_E || (reinterpret_cast<uintptr_t>(a.s) & 15) != 0) return false;
  if (mode == 1) return rows_e % 128 == 0 && (rows_e / 2) % 64 == 0;
  return rows_e % 128 == 0 && 128 * a.s_rstride <= sm16::S_SLOT && (128 * a.s_rstride) % 16 == 0;
}

template <int MODE, int NT = 64>
void launch_small16(GemmArgs a, int act_rows) {
  const int rows_total = int(a.s_total / a.s_rstride);
  const int rows_e = int(a.w_estride / a.w_rstride);
  a.brows = act_rows < 16 ? act_rows : 16;
  encode_u8_2d(&a.tm_b, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
  encode_u8_2d(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), 32u, MODE == 1 ? 64u : 128u, false, "w");
  if (MODE == 1) encode_u8_2d(&a.tm_s, a.s, uint64_t(a.s_rstride), uint64_t(rows_total), 16u, 64u, false, "s");
  a.NB = MODE == 1 ? (rows_e / 2) / 64 : rows_e / 128;
  using C = S16Cfg<NT>;
  opt_in_smem<moe_small16_kernel<MODE, NT>>(C::SMEM);
  // debug knobs (read once): DS41H_S16_GRIDX = CTAs per SM in the grid (default
  // C::CTAS), DS41H_S16_NOPDL=1 = plain launch (no programmatic stream serialization)
  static const int gridx = [] { const char* v = std::getenv("DS41H_S16_GRIDX"); return v && *v ? std::atoi(v) : 0; }();
  static const bool nopdl = [] { const char* v = std::getenv("DS41H_S16_NOPDL"); return v && *v && *v != '0'; }();
  const dim3 grid((gridx > 0 ? gridx : C::CTAS) * num_sms());
  if (nopdl) {
    moe_small16_kernel<MODE, NT><<<grid, dim3(sm16::NTHR), C::SMEM, c10::cuda::getCurrentCUDAStream()>>>(a);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_small16 launch failed");
    return;
  }
  launch_pdl("moe_small16", moe_small16_kernel<MODE, NT>, grid, dim3(sm16::NTHR), C::SMEM, a);
}

// small16k (GEMM1 / GEMM2 variants 18..23, chunk-fastest 26..31): small16 units with wide weight boxes / 32-token units;
// same shape conditions as small16 (small16_ok), bit-identical outputs (tests/test_moe.py test_small16k_bit_identical_to_0x500).
template <int MODE, int NT, int KPS, int NST, int CTAS>
void launch_small16k(GemmArgs a, int act_rows) {
  using C = S16K<NT, KPS, NST, CTAS>;
  const int rows_total = int(a.s_total / a.s_rstride);
  const int rows_e = int(a.w_estride / a.w_rstride);
  a.brows = act_rows < 16 ? act_rows : 16;
  encode_u8_2d(&a.tm_b, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
  encode_u8_2d_swz(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), uint32_t(C::RB), MODE == 1 ? 64u : 128u,
                   KPS == 1 ? CU_TENSOR_MAP_SWIZZLE_NONE : KPS == 2 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B, "w");
  if (MODE == 1) encode_u8_2d(&a.tm_s, a.s, uint64_t(a.s_rstride), uint64_t(rows_total), 16u, 64u, false, "s");
  a.NB = MODE == 1 ? (rows_e / 2) / 64 : rows_e / 128;
  opt_in_smem<moe_small16k_kernel<MODE, NT, KPS, NST, CTAS>>(C::SMEM);
  launch_pdl("moe_small16k", moe_small16k_kernel<MODE, NT, KPS, NST, CTAS>, dim3(CTAS * num_sms()), dim3(sm16::NTHR), C::SMEM, a);
}
// config sub = 1..6 (forward flag bits 11..13; GEMM variant 17 + sub, + 8 = chunk-fastest unit order)
inline bool small16k_variant(int v) { return (v >= 18 && v <= 23) || (v >= 26 && v <= 31); }
template <int MODE>
void launch_small16k_v(int v, GemmArgs a, int act_rows) {
  a.s16_cf = v >= 26 ? 1 : 0;
  switch (v >= 26 ? v - 8 : v) {
    case 18: launch_small16k<MODE, 16, 2, 3, 4>(a, act_rows); break;  // 16-token units, 64-byte rows, 3 x 12 KB stages, 4 CTAs / SM
    case 19: launch_small16k<MODE, 16, 2, 4, 3>(a, act_rows); break;  // 16-token units, 64-byte rows, 4 x 12 KB stages, 3 CTAs / SM
    case 20: launch_small16k<MODE, 16, 4, 3, 2>(a, act_rows); break;  // 16-token units, 128-byte rows, 3 x 24 KB stages, 2 CTAs / SM
    case 21: launch_small16k<MODE, 16, 4, 2, 3>(a, act_rows); break;  // 16-token units, 128-byte rows, 2 x 24 KB stages, 3 CTAs / SM
    case 22: launch_small16k<MODE, 32, 1, 6, 3>(a, act_rows); break;  // 32-token units, 32-byte rows, 6 x 8 KB stages, 3 CTAs / SM
    case 23: launch_small16k<MODE, 32, 2, 3, 3>(a, act_rows); break;  // 32-token units, 64-byte rows, 3 x 16 KB stages, 3 CTAs / SM
    default: TORCH_CHECK(false, "small16k: unknown variant ", v);
  }
}

// GEMM2 variant 3 is usable for I % 64 == 0, I <= 640 (B tile of <= 5 K blocks), H % 128 == 0.
bool gemm2_bs_ok(const GemmArgs& a) { return a.K % 64 == 0 && a.K <= 128 * g2::MAXKB && a.E <= MAX_E; }

void launch_gemm2_bs(const GemmArgs& ga, int R, int epi) {
  G2Args a{};
  const int I = ga.K;
  const cuuint64_t dims[2] = {cuuint64_t(I), cuuint64_t(R)}, strides[1] = {cuuint64_t(I)};
  const int brows = R < 128 ? R : 128;
  const cuuint32_t box[2] = {128u, cuuint32_t(brows)}, es[2] = {1u, 1u};
  const CUresult r = tmap_encode_fn()(&a.tm_b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(ga.act), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "moe_gemm2_bs: cuTensorMapEncodeTiled(hq) failed: ", int(r));
  a.w = ga.w;
  a.s = ga.s;
  a.off = ga.off;
  a.perm_slot = ga.perm_slot;
  a.perm_tok = ga.perm_tok;
  a.hs = ga.act_scale;
  a.ctr = ga.ctr;
  a.out = ga.out;
  a.E = ga.E;
  a.H = ga.NB * 128;
  a.I = I;
  a.NB = ga.NB;
  static const int target = [] {
    const char* v = std::getenv("DS41H_MOE_G2_ITEMS");
    return v ? std::atoi(v) : 8 * num_sms();
  }();
  a.target_items = target;
  a.brows = brows;
  auto go = [&](auto kern) {
    opt_in_smem<decltype(kern)::value>(g2::SMEM);
    launch_pdl("moe_gemm2_bs", decltype(kern)::value, dim3(num_sms()), dim3(NTHREADS), g2::SMEM, a);
  };
  switch (epi) {
    case EPI_BULK: go(std::integral_constant<decltype(&moe_gemm2_bs_kernel<EPI_BULK>), &moe_gemm2_bs_kernel<EPI_BULK>>{}); break;
    case EPI_STG: go(std::integral_constant<decltype(&moe_gemm2_bs_kernel<EPI_STG>), &moe_gemm2_bs_kernel<EPI_STG>>{}); break;
    case EPI_RED: go(std::integral_constant<decltype(&moe_gemm2_bs_kernel<EPI_RED>), &moe_gemm2_bs_kernel<EPI_RED>>{}); break;
    default: go(std::integral_constant<decltype(&moe_gemm2_bs_kernel<EPI_NONE>), &moe_gemm2_bs_kernel<EPI_NONE>>{}); break;
  }
}

// GEMM variants. MODE 1: 0/1 = BNMAX 256, whole-K tensor-core accumulation; 2/3/4 = BNMAX 128 with
// fp32 promotion every 4/2/1 K blocks (DeepGEMM-style accumulation precision). MODE 2 (K = 576):
// 1 = BNMAX 256, 2 = BNMAX 128 (both whole-K accumulation), 3 = B-stationary kernel (moe_gemm2_bs,
// falls back to 2 for unsupported shapes).
// Defaults (validated on layer 10 / T = 8192 and by tests/test_moe.py): TMA-fed GEMM1 (activations in
// routed-row order) and TMA-fed GEMM2 with the bf16-reduction epilogue (no slot buffer, no combine).
// W4A16 (lossless vs stock W4A16): GEMM1 9 / 10 = BNMAX 256 / 128, GEMM2 11 / 12 = BNMAX 256 / 128 (slot rows +
// moe_combine with the routing weights, deterministic); GEMM1 11 / GEMM2 13 = moe_gemm16_kernel (256-row tiles,
// 2 accumulator chains per consumer WG, separate scale ring; same numerics). The default since the lossless rule;
// the W4A8 variants (FP8_V1 / FP8_V2 = 0x95) stay available explicitly (ds41h.moe DS41H_MOE_PRECISION=fp8).
constexpr int DEFAULT_V1 = 9;
constexpr int DEFAULT_V2 = 11;
constexpr int FP8_V1 = 5;
constexpr int FP8_V2 = 9;
// GEMM1 13 / 14: LD 4 (BNMAX 256 / 128) with the B rows gathered from x in token order (LD 6, "x-gather": input
// act_log16(x) [T, H] instead of act_perm16's routed-row copy [T*K, H]); same numerics as 9 / 10
// GEMM1 / GEMM2 16: moe_small16_kernel (small-batch W4A16, mma.sync; forward variant flag 0x100)
inline bool is_a16(int mode, int v) { return mode == 1 ? ((v >= 9 && v <= 14) || v == 16 || v == 17 || small16k_variant(v)) : ((v >= 11 && v <= 17) || small16k_variant(v)); }
inline bool is_gather16(int mode, int v) { return mode == 1 && (v == 13 || v == 14); }

static int g_hybrid_split = 0;  // set_hybrid(): 0 = off, S > 0 = remainder tiles <= S rows of the default W4A16 GEMMs to gemm16
static int g_hybrid_engine = 0;  // set_hybrid(S, engine): remainder kernel 0 = moe_gemm16_kernel, 1 = moe_small16_kernel

void dispatch_gemm(int mode, int variant, const GemmArgs& args_in, int R) {
  GemmArgs args = args_in;
  int v = variant ? variant : (mode == 1 ? DEFAULT_V1 : DEFAULT_V2);
  if (g_hybrid_split > 0 && ((mode == 1 && v == 9) || (mode == 2 && v == 11)) && gemm16_ok(mode, args)) {
    // full (and > S row) token tiles: LD 4 kernel, BNMAX 256; the <= S row remainders: moe_gemm16_kernel (same numerics)
    GemmArgs a1 = args, a2 = args;
    a1.tsplit = g_hybrid_split, a1.tsplit_bn = 256;
    a2.tsplit = g_hybrid_split, a2.tsplit_bn = 256;
    make_tma_maps(mode, a1, R, false, true);
    if (mode == 1)
      launch_gemm<1, 256, 0, 4>(a1);
    else
      launch_gemm<2, 256, 0, 4>(a1);
    if (g_hybrid_engine == 1 && small16_ok(mode, a2)) {
      if (mode == 1)
        launch_small16<1>(a2, R);
      else
        launch_small16<2>(a2, R);
    } else if (mode == 1) {
      launch_gemm16<1>(a2, R);
    } else {
      launch_gemm16<2>(a2, R);
    }
    return;
  }
  if (small16k_variant(v)) {  // small16k: wide weight boxes / 32-token units / chunk-fastest order
    if (small16_ok(mode, args)) {
      if (mode == 1)
        launch_small16k_v<1>(v, args, R);
      else
        launch_small16k_v<2>(v, args, R);
      return;
    }
    v = mode == 1 ? 9 : 11;  // unsupported shape: LD 4 kernel
  }
  if (v == 16 || v == 17) {  // W4A16 small-batch kernel (all rows); 17 = tiny batches (16-token units, 4 CTAs / SM)
    if (small16_ok(mode, args)) {
      if (v == 17)
        mode == 1 ? launch_small16<1, 16>(args, R) : launch_small16<2, 16>(args, R);
      else
        mode == 1 ? launch_small16<1, 64>(args, R) : launch_small16<2, 64>(args, R);
      return;
    }
    v = mode == 1 ? 9 : 11;  // unsupported shape: LD 4 kernel
  }
  if ((mode == 1 && v == 11) || (mode == 2 && v == 13)) {  // W4A16, 256-row tiles
    if (gemm16_ok(mode, args)) {
      if (mode == 1)
        launch_gemm16<1>(args, R);
      else
        launch_gemm16<2>(args, R);
      return;
    }
    v = mode == 1 ? 9 : 11;  // unsupported shape: LD 4 kernel
  }
  // W4A16 cluster-multicast variants (activation boxes loaded once per cluster): GEMM1 12 = clusters of 3 (NB = 9),
  // GEMM2 14 / 15 = clusters of 4 / 2; NB must be a multiple of the cluster size, else the plain LD 4 kernel runs
  if ((mode == 1 && v == 12) || (mode == 2 && (v == 14 || v == 15))) {
    const int cs = mode == 1 ? 3 : (v == 14 ? 4 : 2);
    if (args.NB % cs == 0) {
      make_tma_maps(mode, args, R, false, true);
      if (mode == 1)
        launch_gemm_cl<1, 256, 0, 4, 3>(args);
      else if (v == 14)
        launch_gemm_cl<2, 256, 0, 4, 4>(args);
      else
        launch_gemm_cl<2, 256, 0, 4, 2>(args);
      return;
    }
    v = mode == 1 ? 9 : 11;
  }
  if (is_a16(mode, v))
    make_tma_maps(mode, args, R, false, true);
  else if ((mode == 1 && v >= 5 && v <= 8) || (mode == 2 && v >= 7 && v <= 10))
    make_tma_maps(mode, args, R, mode == 1 && v >= 7);
  if (mode == 1) {
    switch (v) {
      case 9: launch_gemm<1, 256, 0, 4>(args); break;  // W4A16, activations bf16 in routed-row / logical order
      case 10: launch_gemm<1, 128, 0, 4>(args); break;
      case 13: launch_gemm<1, 256, 0, 6>(args); break;  // W4A16 x-gather: B rows gathered from xl [T, H] (token order)
      case 14: launch_gemm<1, 128, 0, 6>(args); break;
      case 1: launch_gemm<1, 256, 0>(args); break;
      case 2: launch_gemm<1, 128, 4>(args); break;
      case 3: launch_gemm<1, 128, 2>(args); break;
      case 4: launch_gemm<1, 128, 1>(args); break;
      case 5: launch_gemm<1, 256, 0, 1>(args); break;  // TMA loader; act = activations in routed-row order (act_quant_perm)
      case 6: launch_gemm<1, 128, 0, 1>(args); break;
      case 7: launch_gemm<1, 256, 0, 3>(args); break;  // TMA loader with one-row boxes of the token rows (act_quant order)
      case 8: launch_gemm<1, 128, 0, 3>(args); break;
      default: TORCH_CHECK(false, "moe_gemm mode 1: variant must be 0..14, got ", variant);
    }
  } else {
    switch (v) {
      case 1: launch_gemm<2, 256, 0>(args); break;
      case 2: launch_gemm<2, 128, 0>(args); break;
      case 7: launch_gemm<2, 256, 0, 1>(args); break;  // TMA loader
      case 8: launch_gemm<2, 128, 0, 1>(args); break;
      case 9: launch_gemm<2, 256, 0, 2>(args); break;  // TMA loader, out[T, H] += hs * y (bf16 reductions)
      case 10: launch_gemm<2, 128, 0, 2>(args); break;
      case 11: launch_gemm<2, 256, 0, 4>(args); break;  // W4A16: h bf16 (logical order) -> unscaled bf16 slot rows
      case 12: launch_gemm<2, 128, 0, 4>(args); break;
      case 3:  // B-stationary, slot output via bulk stores
      case 4:  // B-stationary, slot output via 16-byte stores
        if (gemm2_bs_ok(args))
          launch_gemm2_bs(args, R, v == 3 ? EPI_BULK : EPI_STG);
        else
          launch_gemm<2, 128, 0>(args);
        break;
      case 5:  // B-stationary, out[T, H] += hs * y (bf16 reductions; hs in routed-row order, out zeroed)
        TORCH_CHECK(gemm2_bs_ok(args), "moe_gemm mode 2 variant 5: unsupported shape");
        launch_gemm2_bs(args, R, EPI_RED);
        break;
      case 6:  // B-stationary, no output (timing only)
        TORCH_CHECK(gemm2_bs_ok(args), "moe_gemm mode 2 variant 6: unsupported shape");
        launch_gemm2_bs(args, R, EPI_NONE);
        break;
      default: TORCH_CHECK(false, "moe_gemm mode 2: variant must be 0..15, got ", variant);
    }
  }
}

#endif  // DS41H_DEVICE_ONLY
}  // namespace

#ifndef DS41H_DEVICE_ONLY
// Route-side kernels (moe_route.cu)
void ds41h_moe_topk(torch::Tensor logits, torch::Tensor bias, c10::optional<torch::Tensor> bias_vl, c10::optional<torch::Tensor> input_ids,
                    int64_t image_lo, double scale, bool renormalize, torch::Tensor out_w, torch::Tensor out_ids);
void ds41h_moe_permute(torch::Tensor topk_ids, torch::Tensor topk_w, int64_t E, torch::Tensor meta, torch::Tensor perm_w);
void ds41h_act_quant(torch::Tensor x, torch::Tensor xq, torch::Tensor xs);
void ds41h_act_quant_perm(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xq, torch::Tensor xs);
void ds41h_act_perm16(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xp);
void ds41h_act_log16(torch::Tensor x, torch::Tensor xl);
void ds41h_h_quant(torch::Tensor h, torch::Tensor nrows, torch::Tensor hq, torch::Tensor hs, c10::optional<torch::Tensor> hs_idx);
void ds41h_moe_combine(torch::Tensor slot, torch::Tensor topk_ids, int64_t E, torch::Tensor out, c10::optional<torch::Tensor> sscale,
                       c10::optional<torch::Tensor> addend);
int64_t ds41h_moe_meta_size(int64_t E, int64_t R);

static void check_u8(const torch::Tensor& t, std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kUInt8, name, ": expected contiguous CUDA uint8");
  TORCH_CHECK(t.sizes() == c10::IntArrayRef(shape), name, ": shape ", t.sizes(), " != ", c10::IntArrayRef(shape));
}

// Grouped GEMM entry for one MODE. meta/perm_w are the moe_permute outputs.
//   MODE 1: act = xq [T, H] u8, act_scale = xs [T], w = w13 [E, 2I, H/2], s = s13, out = h [R, I] bf16
//   MODE 2: act = hq [R, I] u8, act_scale unused, w = w2 [E, H, I/2], s = s2, out = slot [T*K, H] bf16 *without*
//           the per-routed-row activation scale (moe_combine applies it, h_quant writes it in slot order)
void ds41h_moe_gemm(int64_t mode, torch::Tensor act, torch::Tensor act_scale, torch::Tensor w, torch::Tensor s, torch::Tensor meta,
                    torch::Tensor perm_w, int64_t R, int64_t variant, double limit, torch::Tensor out) {
  TORCH_CHECK(mode == 1 || mode == 2);
  TORCH_CHECK(w.dim() == 3 && s.dim() == 3);
  const int E = w.size(0), rows = w.size(1), K = w.size(2) * 2;
  check_u8(w, {E, rows, K / 2}, "w");
  check_u8(s, {E, rows, K / 32}, "s");
  TORCH_CHECK(K % 64 == 0 && E <= MAX_E, "moe_gemm: K % 64 == 0, E <= 1024");
  variant = variant ? variant : (mode == 1 ? DEFAULT_V1 : DEFAULT_V2);
  // W4A16 variants: act = bf16 [R, K] in routed-row order with the logical K order (act_perm16 / GEMM1 output),
  // act_scale unused; GEMM2 writes unscaled bf16 slot rows (moe_combine applies the routing weights)
  const bool a16 = is_a16(int(mode), int(variant));
  TORCH_CHECK(act.is_cuda() && act.is_contiguous() && act.dtype() == (a16 ? torch::kBFloat16 : torch::kUInt8) && act.numel() % K == 0,
              "moe_gemm: activations must be contiguous ", a16 ? "bf16" : "uint8 (e4m3)", " for variant ", variant);
  // (mode 1 TMA variants 5/6 take the activations in routed-row order [R, H] with per-token scales [T])
  const bool act_routed = (mode == 1 && (variant == 5 || variant == 6)) || (a16 && !is_gather16(int(mode), int(variant)));
  TORCH_CHECK(act_scale.dtype() == torch::kFloat32 && (act_routed || act_scale.numel() >= act.numel() / K));
  if (act_routed) TORCH_CHECK(act.numel() >= R * K, "moe_gemm mode 1 variants 5/6/9/10: activations must be [R, H] in routed-row order");
  TORCH_CHECK(meta.dtype() == torch::kInt32 && meta.numel() >= ds41h_moe_meta_size(E, R));
  TORCH_CHECK(perm_w.dtype() == torch::kFloat32 && perm_w.numel() >= R);
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous());
  GemmArgs args{};
  int* m = mptr<int>(meta);
  args.w = cptr<uint8_t>(w);
  args.s = cptr<uint8_t>(s);
  args.act = reinterpret_cast<const uint8_t*>(act.data_ptr());  // e4m3 bytes or bf16 (W4A16), contiguous (checked)
  args.act_scale = cptr<float>(act_scale);
  args.off = m + (E + 1);
  args.ctr = m + 2 * (E + 1) + (mode == 1 ? 0 : 2);
  args.perm_tok = m + 2 * (E + 1) + META_CTR;
  args.perm_slot = args.perm_tok + R;
  args.perm_w = cptr<float>(perm_w);
  args.out = mptr<__nv_bfloat16>(out);
  args.w_estride = long(rows) * (K / 2);
  args.s_estride = long(rows) * (K / 32);
  args.s_total = long(E) * rows * (K / 32);
  args.w_rstride = K / 2;
  args.s_rstride = K / 32;
  args.K = K;
  args.E = E;
  args.limit = float(limit);
  if (mode == 1) {
    TORCH_CHECK(rows % 128 == 0, "w13 rows (2I) must be a multiple of 128");
    TORCH_CHECK(K % 128 == 0, "GEMM1 reduction length (H) must be a multiple of 128");
    args.I = rows / 2;
    args.NB = rows / 128;
    args.out_stride = rows / 2;
    TORCH_CHECK(out.numel() >= R * (rows / 2));
  } else {
    TORCH_CHECK(rows % 128 == 0, "w2 rows (H) must be a multiple of 128");
    args.I = 0;
    args.NB = rows / 128;
    args.out_stride = rows;
    // routed rows (B operand / act_scale) and slot rows (perm_slot = t*K + k < R) are both < R
    TORCH_CHECK(act.numel() >= R * K && (a16 || act_scale.numel() >= R), "moe_gemm mode 2: hq/hs must hold R rows");
    if (variant == 5 || variant == 9 || variant == 10) {  // out [T, H] indexed by perm_tok (< T <= R)
      TORCH_CHECK(out.numel() % rows == 0 && out.numel() >= rows, "moe_gemm mode 2 variant 5: out must be [T, H]");
    } else {
      TORCH_CHECK(out.numel() >= R * rows, "moe_gemm mode 2: slot buffer must hold R rows of H");
    }
  }
  if (R == 0) return;
  dispatch_gemm(int(mode), int(variant), args, int(act.numel() / K));
}

// Full routed-expert forward on the raw layout. Returns the (TP-partial) routed output [T, H]
// bf16: the caller (vLLM MoERunner) adds the shared expert and all-reduces, as for every other
// FusedMoE method. topk_w must already include renormalization and routed_scaling_factor.
// variant = v1 + 16 * v2 (GEMM1 / GEMM2 variants of dispatch_gemm, 0 = default).
// addend (optional, bf16 [T, H]): added to the routed sum inside moe_combine (fp32, one rounding) -- the shared-expert
// output of the same rank, so the caller needs no separate add.
torch::Tensor ds41h_moe_forward(torch::Tensor x, torch::Tensor topk_w, torch::Tensor topk_ids, torch::Tensor w13, torch::Tensor s13,
                                torch::Tensor w2, torch::Tensor s2, double limit, int64_t variant, c10::optional<torch::Tensor> addend) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == x.size(0) && topk_w.sizes() == topk_ids.sizes(),
              "moe_forward: topk_ids/topk_weights must be [T, K] for x [T, H]");
  x = x.contiguous();
  const int64_t T = x.size(0), H = x.size(1), E = w13.size(0), I = w13.size(1) / 2, K = topk_ids.size(1), R = T * K;
  check_u8(w13, {E, 2 * I, H / 2}, "w13");
  check_u8(s13, {E, 2 * I, H / 32}, "s13");
  check_u8(w2, {E, H, I / 2}, "w2");
  check_u8(s2, {E, H, I / 32}, "s2");
  auto out = torch::empty({T, H}, x.options());
  if (T == 0) return out;
  auto ids = topk_ids.to(torch::kInt32).contiguous();
  auto tw = topk_w.to(torch::kFloat32).contiguous();
  auto i32 = x.options().dtype(torch::kInt32);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto u8 = x.options().dtype(torch::kUInt8);
  auto meta = torch::empty({ds41h_moe_meta_size(E, R)}, i32);
  auto perm_w = torch::empty({R}, f32);
  ds41h_moe_permute(ids, tw, E, meta, perm_w);
  // resolve the defaults here: the activation layout and the GEMM2 output path depend on the variants
  int v1 = int(variant & 15), v2 = int((variant >> 4) & 15);
  if (variant & 0x100) {  // small-batch W4A16 kernel for both GEMMs: 16-token units for tiny batches (<= 1536 routed rows);
    // 0x400: 16-token units at every size (the 64-token-unit kernel showed run-to-run differences
    // under a concurrent side-stream load at T >= 1000, tests/test_moe.py test_bit_repeatable_stress; NT16 is clean)
    v1 = v2 = ((variant & 0x400) || (R <= 1536 && !(variant & 0x200))) ? 17 : 16;
    // small16k: 0x500 | sub << 11 with sub 1..6 -> GEMM variants 17 + sub (wide weight boxes /
    // 32-token units, same math), + 0x4000 = the chunk-fastest unit order (+ 8)
    const int sub = int((variant >> 11) & 7);
    if (sub >= 1 && sub <= 6 && v1 == 17) v1 = v2 = 17 + sub + ((variant & 0x4000) ? 8 : 0);
  } else if (!v1 && !v2)
    v1 = DEFAULT_V1, v2 = DEFAULT_V2;
  else if (!v1)
    v1 = is_a16(2, v2) ? DEFAULT_V1 : FP8_V1;
  else if (!v2)
    v2 = is_a16(1, v1) ? DEFAULT_V2 : FP8_V2;
  TORCH_CHECK(is_a16(1, v1) == is_a16(2, v2), "moe_forward: W4A16 (GEMM1 9..14, GEMM2 11..15) and W4A8 variants cannot be mixed");
  const int64_t t0 = 2 * (E + 1) + META_CTR;
  if (is_a16(1, v1)) {
    // W4A16 (lossless vs stock vLLM W4A16): bf16 x in routed-row / logical K order -> GEMM1 (exact bf16 weights,
    // fp32 accumulation, stock clamped SwiGLU) -> h bf16 -> GEMM2 -> unscaled bf16 y per routed row in slot order
    // (stock down_output) -> out[t] = bf16(sum_k topk_w[t, k] * y[t*K + k]) in fp32 (stock moe_unpermute / mul_sum)
    auto h = torch::empty({R, I}, x.options());
    if (is_gather16(1, v1)) {  // x-gather GEMM1: x only reordered along K (logical order), rows gathered by the kernel
      auto xl = torch::empty({T, H}, x.options());
      ds41h_act_log16(x, xl);
      ds41h_moe_gemm(1, xl, perm_w, w13, s13, meta, perm_w, R, v1, limit, h);
    } else {
      auto xp = torch::empty({R, H}, x.options());  // freed before the slot buffer is allocated (same size)
      ds41h_act_perm16(x, meta.narrow(0, t0 + 2 * R, R), K, xp);
      ds41h_moe_gemm(1, xp, perm_w, w13, s13, meta, perm_w, R, v1, limit, h);
    }
    auto slot = torch::empty({R, H}, x.options());
    ds41h_moe_gemm(2, h, perm_w, w2, s2, meta, perm_w, R, v2, limit, slot);
    ds41h_moe_combine(slot, ids, E, out, tw.view({-1}), addend);
    return out;
  }
  const bool perm_act = v1 == 5 || v1 == 6;  // TMA-fed GEMM1: activations in routed-row order
  auto xq = torch::empty({perm_act ? R : T, H}, u8);
  auto xs = torch::empty({T}, f32);
  if (perm_act)
    ds41h_act_quant_perm(x, meta.narrow(0, 2 * (E + 1) + META_CTR + 2 * R, R), K, xq, xs);
  else
    ds41h_act_quant(x, xq, xs);
  auto h = torch::empty({R, I}, x.options());
  ds41h_moe_gemm(1, xq, xs, w13, s13, meta, perm_w, R, v1, limit, h);
  auto hq = torch::empty({R, I}, u8);
  auto hs = torch::empty({R}, f32);
  if (v2 == 5 || v2 == 9 || v2 == 10) {
    // GEMM2 accumulates hs * y of every routed row into out (bf16 reductions): no slot buffer, no combine
    ds41h_h_quant(h, meta.narrow(0, E, 1), hq, hs, c10::nullopt);  // hs in routed-row order
    if (addend.has_value() && addend->defined())
      out.copy_(*addend);
    else
      out.zero_();
    ds41h_moe_gemm(2, hq, hs, w2, s2, meta, perm_w, R, v2, limit, out);
    return out;
  }
  // slot order: GEMM2 writes unscaled rows, moe_combine applies hs
  ds41h_h_quant(h, meta.narrow(0, E, 1), hq, hs, meta.narrow(0, t0 + R, R));  // counts[E] = valid routed rows; perm_slot
  auto slot = torch::empty({R, H}, x.options());
  ds41h_moe_gemm(2, hq, hs, w2, s2, meta, perm_w, R, v2, limit, slot);
  ds41h_moe_combine(slot, ids, E, out, hs, addend);
  return out;
}

#ifdef DS41H_MOE_TRACE
torch::Tensor ds41h_moe_trace(bool clear) {
  auto t = torch::empty({long(sizeof(g_mtrace) / 8)}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  TORCH_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), g_mtrace, sizeof(g_mtrace)) == cudaSuccess);
  if (clear) {
    void* p = nullptr;
    TORCH_CHECK(cudaGetSymbolAddress(&p, g_mtrace) == cudaSuccess);
    TORCH_CHECK(cudaMemset(p, 0, sizeof(g_mtrace)) == cudaSuccess);
  }
  return t;
}
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef DS41H_MOE_TRACE
  m.def("trace", &ds41h_moe_trace, "dump (and clear) the debug timeline buffer");
#endif
  m.def("moe_forward", &ds41h_moe_forward, "routed experts forward on the raw MXFP4 layout (W4A16 default / W4A8 grouped GEMM)",
        py::arg("x"), py::arg("topk_w"), py::arg("topk_ids"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"),
        py::arg("limit"), py::arg("variant"), py::arg("addend") = py::none());
  m.def("default_variant", [] { return int64_t(DEFAULT_V1 + 16 * DEFAULT_V2); }, "GEMM variants used for variant 0 (v1 + 16 v2)");
  m.def("set_hybrid", [](int64_t split, int64_t engine) { g_hybrid_split = int(split), g_hybrid_engine = int(engine); },
        "default W4A16 GEMMs: token-tile remainders <= split rows run on moe_gemm16_kernel (engine 0) or moe_small16_kernel "
        "(engine 1); split 0 = off", py::arg("split"), py::arg("engine") = 0);
  m.def("fp8_variant", [] { return int64_t(FP8_V1 + 16 * FP8_V2); }, "default W4A8 GEMM variants (v1 + 16 v2)");
  m.def("act_log16", &ds41h_act_log16, "bf16 x -> same rows, W4A16 logical K order within 64-blocks (x-gather GEMM1 input)");
  m.def("act_perm16", &ds41h_act_perm16, "bf16 x -> routed-row order, W4A16 logical K order within 64-blocks");
  m.def("moe_gemm", &ds41h_moe_gemm, "one grouped GEMM (mode 1: w13+SwiGLU, mode 2: w2)");
  m.def("moe_topk", &ds41h_moe_topk, "fused sqrtsoftplus + bias top-k routing (dsv4_topk semantics)");
  m.def("moe_permute", &ds41h_moe_permute, "expert permutation metadata");
  m.def("meta_size", &ds41h_moe_meta_size, "int32 elements of the moe_permute meta buffer for (E, R)");
  m.def("act_quant", &ds41h_act_quant, "bf16 -> e4m3 per-row scale, fragment order (K % 512 == 0)");
  m.def("act_quant_perm", &ds41h_act_quant_perm, "act_quant written to the routed rows of each token (routed-row order)");
  m.def("h_quant", &ds41h_h_quant, "bf16 -> e4m3 per-row scale, fragment order (I % 64 == 0)", py::arg("h"), py::arg("nrows"), py::arg("hq"),
        py::arg("hs"), py::arg("hs_idx") = py::none());
  m.def("moe_combine", &ds41h_moe_combine, "sum top-k slot rows per token (times sscale[slot])", py::arg("slot"), py::arg("topk_ids"),
        py::arg("E"), py::arg("out"), py::arg("sscale") = py::none(), py::arg("addend") = py::none());
}
#endif  // DS41H_DEVICE_ONLY
