// c1e0k.cuh: E0 as its own launch. The SAME c1e0.cuh device code as c1mega's layer 0
// with C1IDX_E0=1 (hooks 2 / 1 / 3: c1e0::e0_push_x, e0_produce, e0_consume), in a 132-CTA kernel of its own: 8 consumer
// warps (c1e0::CTH, named barrier 1) + 1 producer warp, a local ring of NS stages (the half tile's 16,448 B + the NR x rows),
// per-CTA epochs E.tick[c] advanced exactly as layer 0 does. So: the same units (CTA c < 100 = (layer c / 50, half tile
// c % 50) of this GPU), the same k order, the same per-(row, k32) prescales and accumulation order -> the same kv bits on
// every GPU; the same kvcnt arrivals (132 CTAs x 4 GPUs per layer per call) -> layers 1 / 14 (local mode, c1eng.cuh) wait
// for kvcnt[li] >= 528 x e0tick[c] as before. c1mega's layer 0 then has no E0 (Args::eg = 0): a regular layer.
// Launch (c1idx_host.C1Idx.e0_launch): rank r's Engram stream, after e0_prepare's gathers there, joined into rank r's stream
// before its c1mega launch. No intra-GPU co-residency is needed (no CTA waits on another CTA of the same GPU); the peers'
// producers wait for the owner's x push (xcnt), which the owner's launch issues without waiting on anything.
#pragma once
#include "c1e0.cuh"

namespace c1e0k {
constexpr int NS = 8;                         // ring stages (e0_produce keeps the stage of chunk k in stg[k % 8]: NS <= 8)
constexpr int STG = 17536;                    // stage stride: >= HALF + NRM x XST, a 128-B multiple
static_assert(STG >= c1e0::STAGE_BYTES && STG % 128 == 0 && NS <= 8, "E0K ring");
constexpr int NCW = c1e0::CTH / 32;           // 8 consumer warps (the ring's empty count)
constexpr int THR = c1e0::CTH + 32;           // + the producer warp
constexpr int OB = c1e0::NRM * 128 * 2;       // e0_consume's ob scratch: [NRM][128] bf16
constexpr int SMEM = NS * STG + OB + 2 * NS * 8 + 128;   // (+ 128: base alignment)

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint32_t a, unsigned b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t a, uint32_t par) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_W%=;\n}" ::"r"(a),
      "r"(par)
      : "memory");
}
__device__ __forceinline__ uint64_t pol_first() {   // the weights: streamed once per step
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_last() {    // the x rows: read by the 50 units of their layer
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}

// grid c1e0::NCTA (= c1mega's CTA count: e0tick / kvcnt are per c1mega CTA), THR threads, SMEM dynamic bytes
__global__ void __launch_bounds__(THR, 1) k_e0(const c1e0::E0Args* __restrict__ ep, int rank) {
  extern __shared__ unsigned char sm_[];
  const uint32_t s0 = su32(sm_), base = (s0 + 127u) & ~127u;
  const uint32_t ring = base, obs = base + NS * STG, full = obs + OB, empty = full + 8 * NS;
  void* ob = sm_ + (obs - s0);
  const int tid = threadIdx.x, c = blockIdx.x;
  const c1e0::E0Args& E = *ep;
  const unsigned call = E.tick[c] + 1u;   // (every thread reads it here; thread 0 advances it after its last arrival)
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) mb_init(full + 8 * i, 1), mb_init(empty + 8 * i, NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid >= c1e0::CTH) {   // producer warp: lane 0 streams this CTA's unit (CTAs c >= 100: nothing)
    if (tid == c1e0::CTH) {
      const uint64_t pw = pol_first(), px = pol_last();
      int ss = 0;
      unsigned ph = 0;
      c1e0::e0_produce(
          E, rank, c, call, NS,
          [&](unsigned bytes) -> int {
            const int s = ss;
            mb_wait(empty + 8 * s, ph ^ 1u);
            mb_expect(full + 8 * s, bytes);
            if (++ss == NS) ss = 0, ph ^= 1u;
            return s;
          },
          [&](int s, int off, const void* src, unsigned bytes) {
            bulk(ring + s * STG + off, src, bytes, full + 8 * s, off == 0 ? pw : px);
          });
    }
    return;   // (no CTA-wide barrier follows: the consumers use named barrier 1 over their 256 threads only)
  }
  c1e0::e0_push_x(E, rank, c, tid);   // owner: x (both layers) + keep to the peers, one xcnt arrival per GPU
  const int lane = tid & 31;
  int s = 0;
  unsigned ph = 0;
  c1e0::e0_consume(
      E, rank, c, tid, call, ob,
      [&]() -> uint32_t {
        mb_wait(full + 8 * s, ph);
        return ring + s * STG;
      },
      [&]() {
        __syncwarp();
        if (lane == 0) mb_arrive(empty + 8 * s);
        if (++s == NS) s = 0, ph ^= 1u;
      });
  if (tid == 0) E.tick[c] = call;
}
}  // namespace c1e0k
