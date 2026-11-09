// route2: multi-CTA routing-table builder for moe_raw.cu (namespace msr), T = 0..8192.
//
// Produces exactly msr::route's buffers (same layout, same values, byte for byte): EC records (e, n_real, n_pad, pad =
// first Y slot, img_off, h_off, tok / wt / inv of the 64 slots, -1 / 0 / 0 past n_real), GEMM1 tiles t1 (9 per EC) and
// GEMM2 tiles t2 (40 per EC) in the snake order (heaviest n_pad first, stable, every other round of `grid` reversed),
// counts = {n_ec, ntiles1, ntiles2, img bytes, h bytes, P = 6T} and slotmap[p] = sorted position (-1 for ids outside
// [0, 384)). The pair order inside an expert is the STABLE (token, slot) order, as CUB's stable radix sort gives it, so
// the expert chunks (and with them the GEMM tile shape and MMA path each token lands in) are identical too.
//
// No host sync, graph-capturable, deterministic (no atomics on outputs, no inter-CTA waiting). Two paths:
//  T <= 1365 (route2_one_max_t): k_one, one kernel. Every CTA loads all 6T ids and counts them per 256-pair segment
//               and expert in SMEM (atomics: counts do not depend on the order), runs the 384-expert scan itself, ranks
//               its own segment's pairs exactly (ballot match + exclusive scan over its warps) and scatters them; the EC
//               items are split evenly over all CTAs.
//  T > 1365:    two kernels.
//   k_rank<IPT> G = ceil(6T / (1024 IPT)) CTAs x 1024 threads (route2_ipt(T)). Each warp ranks its 32 IPT contiguous
//               pairs (9-ballot key match into a warp-private SMEM histogram, empty rounds skipped); an exclusive scan
//               over the 32 warps gives every pair its stable rank inside the CTA's run of its expert, stored as
//               e | rank << 9 (-1: id outside [0, 384)); the CTA histogram goes to global.
//   k_fill      <= 1 CTA per SM x 512 threads, programmatic dependent launch (its CTAs become resident while k_rank
//               runs and start with griddepcontrol.wait). Every CTA rebuilds the global picture in SMEM from the G
//               histograms (exclusive prefix over the k_rank CTAs per expert, the 384-expert scan), then takes an even
//               share of the pairs (EC slot from the stable rank, weight, 2^-p, slot map) and of the EC items.
//  Shared pieces: the 384-expert scan (count | padded count | chunks | 64-chunks in one u64, 7 last-chunk buckets of 9
//  bits in another, warp shuffles + warp totals); EC items (header + padding slots, 9 GEMM1 + 40 GEMM2 tiles) need no
//  global loads: expert from an SMEM EC -> expert map, tile rank in closed form (a 64-chunk: F[e] + k; a last chunk of
//  n_pad 8b: n64 + sum_{b' > b} nb[b'] + its rank among the experts whose last chunk is in bucket b), which equals
//  k_route's "stable rank by n_pad descending"; CTA 0 writes counts. PDL on the first kernel too (route2_config).
// Buffers: route2_max_ec(T) = 384 + ceil(6T / 64) EC records / 9x / 40x tiles; route2_scratch_ints() int32 scratch (no
// initialisation needed; one scratch per stream).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#include <vector>

#ifndef R2_MATCH
#define R2_MATCH 0  // 1: __match_any_sync for the warp's equal-expert lanes; 0: 9 ballots on the key bits
#endif
#ifndef R2_DBG
#define R2_DBG 0  // 1: phase timestamps in g_dbg (route2_dbg / route2_dbg_reset), for tuning only
#endif

namespace msr2 {

constexpr int NEXP = 384, NMAX = 64, TOPK = 6, MAXT = 8192, MAXP = MAXT * TOPK;
constexpr int R1_THREADS = 1024, R1_WARPS = R1_THREADS / 32, IPT_MIN = 1, IPT_MAX = 16;
constexpr int MAXG = (MAXP + R1_THREADS * IPT_MIN - 1) / (R1_THREADS * IPT_MIN);  // 48 CTA histograms at most
constexpr int F_THREADS = 512;
constexpr int NWS = NEXP / 32;  // 12 warps scan the experts
// packed scan fields: count 16 b, padded count 16 b (sum <= 49152 + 7 * 384), chunks 11 b (<= 1152), 64-chunks 11 b
static_assert(MAXP + 7 * NEXP < 65536 && NEXP + MAXP / NMAX < 2048, "packed scan fields");
static_assert(R1_THREADS * IPT_MAX <= 65535 && NEXP % 32 == 0 && F_THREADS > NEXP, "uint16 prefixes; rank < 2^23");
static_assert(((NEXP + MAXP / NMAX) * 2) % 16 == 0 && (NEXP * 2) % 16 == 0, "SMEM layouts");
__host__ __device__ constexpr int max_ec(int P) { return NEXP + (P + NMAX - 1) / NMAX; }

struct EC {  // = msr::EC
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
  float inv[NMAX];
};
static_assert(sizeof(EC) == 32 + 3 * 4 * NMAX, "EC layout must match moe_raw.cu");
struct Tile {  // = msr::Tile
  int ec, q, e, npad;
  long long boff;
};
static_assert(sizeof(Tile) == 24, "Tile layout must match moe_raw.cu");

// int32 scratch offsets
constexpr int RS_PK = 0;               // [MAXP] e | (rank inside the CTA's run of e) << 9; -1: id outside [0, 384)
constexpr int RS_HIST = RS_PK + MAXP;  // [MAXG][NEXP] per-CTA expert counts
constexpr int RS_INTS = RS_HIST + MAXG * NEXP;

#if R2_DBG
__device__ long long g_dbg[32];  // 0..15 clock64 of CTA 0 (k_rank 0..3, k_fill 8..11); 20..24 globaltimer (ns)
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DBG_CLK(i) \
  if (threadIdx.x == 0 && blockIdx.x == 0) g_dbg[i] = clock64();
#define DBG_GMIN(i) \
  if (threadIdx.x == 0) atomicMin((unsigned long long*)&g_dbg[i], (unsigned long long)gtime());
#define DBG_GMAX(i) \
  if (threadIdx.x == 0) atomicMax((unsigned long long*)&g_dbg[i], (unsigned long long)gtime());
#define DBG_SYNC() __syncthreads();
#else
#define DBG_CLK(i)
#define DBG_GMIN(i)
#define DBG_GMAX(i)
#define DBG_SYNC()
#endif

// lanes of this warp with the same 9-bit key (all 32 lanes active)
__device__ __forceinline__ unsigned peers_of(int key) {
#if R2_MATCH
  return __match_any_sync(0xffffffffu, key);
#else
  unsigned m = 0xffffffffu;
#pragma unroll
  for (int b = 0; b < 9; ++b) {
    const unsigned bal = __ballot_sync(0xffffffffu, (key >> b) & 1);
    m &= ((key >> b) & 1) ? bal : ~bal;
  }
  return m;
#endif
}

__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <int IPT>
__global__ void __launch_bounds__(R1_THREADS, 1) k_rank(const int* __restrict__ ids, int P, int* __restrict__ rs) {
  constexpr int TILE = R1_THREADS * IPT;
  __shared__ __align__(16) unsigned short wh[R1_WARPS][NEXP];  // warp-private counts -> exclusive prefix over warps
  const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31;
  DBG_GMIN(20)
  DBG_CLK(0)
  const int wbase = blockIdx.x * TILE + wp * 32 * IPT;  // this warp's 32 IPT contiguous pairs (stable order)
  for (int i = lane; i < NEXP / 8; i += 32) reinterpret_cast<int4*>(wh[wp])[i] = make_int4(0, 0, 0, 0);
  griddep_wait();  // no-op without PDL
  int ex[IPT];
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    const int p = wbase + 32 * j + lane;
    const int id = p < P ? __ldg(ids + p) : -1;
    ex[j] = (unsigned)id < (unsigned)NEXP ? id : -1;
  }
  griddep_launch();  // k_fill (PDL) may become resident now; it waits for this grid's completion
  __syncwarp();
  const unsigned lt = (1u << lane) - 1u;
  int rk[IPT];
#pragma unroll
  for (int j = 0; j < IPT; ++j) {  // rank inside the warp's run of expert e: earlier rounds + lower lanes
    rk[j] = 0;
    if (wbase + 32 * j < P) {  // warp-uniform: rounds past P are skipped
      const int e = ex[j];
      const unsigned peers = peers_of(e & 511);  // invalid (-1) -> key 511, no expert has it
      const int old = e >= 0 ? (int)wh[wp][e] : 0;
      __syncwarp();
      if (e >= 0 && (peers & lt) == 0) wh[wp][e] = (unsigned short)(old + __popc(peers));
      __syncwarp();
      rk[j] = old + __popc(peers & lt);
    }
  }
  __syncthreads();
  DBG_CLK(1)
  if (tid < NEXP / 4) {  // exclusive prefix over the warps, 4 experts per thread in packed uint16 (sums <= TILE: no
                         // carries between the fields); CTA histogram
    unsigned long long v[R1_WARPS];
#pragma unroll
    for (int w = 0; w < R1_WARPS; ++w) v[w] = reinterpret_cast<const unsigned long long*>(wh[w])[tid];
    unsigned long long run = 0;
#pragma unroll
    for (int w = 0; w < R1_WARPS; ++w) {
      reinterpret_cast<unsigned long long*>(wh[w])[tid] = run;
      run += v[w];
    }
    reinterpret_cast<int4*>(rs + RS_HIST + blockIdx.x * NEXP)[tid] =
        make_int4((int)(run & 0xffff), (int)((run >> 16) & 0xffff), (int)((run >> 32) & 0xffff), (int)(run >> 48));
  }
  __syncthreads();
  DBG_CLK(2)
#pragma unroll
  for (int j = 0; j < IPT; ++j) {
    const int p = wbase + 32 * j + lane, e = ex[j];
    if (p < P) rs[RS_PK + p] = e >= 0 ? (e | ((wh[wp][e] + rk[j]) << 9)) : -1;
  }
  DBG_SYNC()
  DBG_CLK(3)
  DBG_GMAX(21)
}

constexpr int MAXEC = NEXP + MAXP / NMAX;  // 1152
struct FSmem {
  int cnt[NEXP], seg[NEXP], ecb[NEXP], pseg[NEXP], f64[NEXP], brk[NEXP], lb[NEXP];
  unsigned short ecmap[MAXEC];  // EC -> expert
  unsigned long long wtot[NWS][2];
  int bbase[8];
  int n_ec;
};

__device__ __forceinline__ int ec_expert(const FSmem& sm, int x) { return sm.ecmap[x]; }

__device__ __forceinline__ void fill_pair(const FSmem& sm, const unsigned short* offE, int tshift, int j, int pk,
                                          float wj, float ivj, EC* __restrict__ ecs, int* __restrict__ slotmap) {
  if (pk < 0) {
    slotmap[j] = -1;
    return;
  }
  const int e = pk & 511, re = offE[(j >> tshift) * NEXP + e] + (pk >> 9);  // stable rank inside expert e
  slotmap[j] = sm.seg[e] + re;
  EC* ec = ecs + sm.ecb[e] + (re >> 6);
  const int n = re & (NMAX - 1);
  ec->tok[n] = j / TOPK;
  ec->wt[n] = wj;
  ec->inv[n] = ivj;
}

// 384-expert scan (call with every thread; tid < NEXP passes its expert's total c): SMEM tables seg / pseg / ecb / f64
// / brk / lb / cnt / ecmap / bbase / n_ec; CTA 0 writes counts. Two __syncthreads.
// A = count | padded count << 16 | chunks << 32 | 64-chunks << 43; Fl = one 9-bit field per last-chunk bucket.
__device__ __forceinline__ void expert_scan(FSmem& sm, int c, int P, int* __restrict__ counts, const int* ec8 = nullptr) {
  const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31;
  unsigned long long A = 0, Fl = 0;
  int lbv = 0;
  if (tid < NEXP) {
    const int r = c & (NMAX - 1);
    const int nc = (c + NMAX - 1) / NMAX, pad = (c + 7) & ~7, f = c / NMAX + (r > 56);
    lbv = (r != 0 && r <= 56) ? (r + 7) >> 3 : 0;
    sm.cnt[tid] = c;
    sm.lb[tid] = lbv;
    A = (unsigned long long)c | ((unsigned long long)pad << 16) | ((unsigned long long)nc << 32) |
        ((unsigned long long)f << 43);
    Fl = lbv ? 1ull << (9 * (lbv - 1)) : 0ull;
  }
  const unsigned long long A0 = A, F0 = Fl;
  if (wp < NWS) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long a = __shfl_up_sync(0xffffffffu, A, o), fl = __shfl_up_sync(0xffffffffu, Fl, o);
      if (lane >= o) A += a, Fl += fl;
    }
    if (lane == 31) sm.wtot[wp][0] = A, sm.wtot[wp][1] = Fl;
  }
  DBG_CLK(12)
  __syncthreads();
  DBG_CLK(13)
  if (tid < NEXP) {
    unsigned long long pa = 0, pf = 0;
#pragma unroll
    for (int ww = 0; ww < NWS; ++ww) {  // fixed trip count: the 24 SMEM loads issue back to back
      const unsigned long long a = sm.wtot[ww][0], f = sm.wtot[ww][1];
      if (ww < wp) pa += a, pf += f;
    }
    const unsigned long long xa = A + pa - A0, xf = Fl + pf - F0;
    sm.seg[tid] = (int)(xa & 0xffffu), sm.pseg[tid] = (int)((xa >> 16) & 0xffffu);
    const int ecb = (int)((xa >> 32) & 0x7ffu), nc = (int)((A0 >> 32) & 0x7ffu);
    sm.ecb[tid] = ecb, sm.f64[tid] = (int)((xa >> 43) & 0x7ffu);
    sm.brk[tid] = lbv ? (int)((xf >> (9 * (lbv - 1))) & 511u) : 0;
    for (int k = 0; k < nc; ++k) sm.ecmap[ecb + k] = (unsigned short)tid;
  } else if (tid == NEXP) {  // totals
    unsigned long long ta = 0, tf = 0;
#pragma unroll
    for (int ww = 0; ww < NWS; ++ww) ta += sm.wtot[ww][0], tf += sm.wtot[ww][1];
    const int n_ec = (int)((ta >> 32) & 0x7ffu), n64 = (int)((ta >> 43) & 0x7ffu);
    const long long ptot = (long long)((ta >> 16) & 0xffffu);
    int above = n64;  // ECs with a larger n_pad than bucket b
#pragma unroll
    for (int b = 7; b >= 1; --b) {
      sm.bbase[b] = above;
      above += (int)((tf >> (9 * (b - 1))) & 511u);
    }
    sm.bbase[0] = 0;
    sm.n_ec = n_ec;
    if (blockIdx.x == 0) {
      const int nbig = n_ec - (ec8 ? (int)(tf & 511u) : 0);  // route2x: the tile lists hold only n_pad > 8 ECs
      counts[0] = n_ec, counts[1] = 9 * nbig, counts[2] = 40 * nbig;
      counts[3] = (int)(ptot * 80 * 128), counts[4] = (int)(ptot * 9 * 128), counts[5] = P;
      if (ec8) counts[6] = (int)(tf & 511u);  // ECs with n_pad 8 (bucket 1): the row-stream GEMM1's work list length
    }
  }
  DBG_CLK(14)
  __syncthreads();
}

// pair p with stable rank re inside expert e: EC slot (token, weight, 2^-p) and slot map
__device__ __forceinline__ void put_pair(const FSmem& sm, int p, int e, int re, float wp_, float iv, EC* __restrict__ ecs,
                                         int* __restrict__ slotmap) {
  slotmap[p] = sm.seg[e] + re;
  EC* ec = ecs + sm.ecb[e] + (re >> 6);
  const int n = re & (NMAX - 1);
  ec->tok[n] = p / TOPK;
  ec->wt[n] = wp_;
  ec->inv[n] = iv;
}

// EC item jj in [0, mec x 113): jj < mec x 64: EC x = jj / 64 slot n (header at n = 0, padding past n_real); else tile
// (9 GEMM1 + 40 GEMM2 per EC, heaviest first, alternate rounds of `grid` reversed, as k_route_fill)
__device__ __forceinline__ void ec_item(const FSmem& sm, int jj, int mec, int n_ec, int grid, EC* __restrict__ ecs,
                                        Tile* __restrict__ t1, Tile* __restrict__ t2, int* __restrict__ ec8 = nullptr) {
  if (jj < mec * NMAX) {
    const int x = jj >> 6, n = jj & (NMAX - 1);
    if (x >= n_ec) return;
    const int e = sm.ecmap[x], k = x - sm.ecb[e], nr = min(NMAX, sm.cnt[e] - NMAX * k);
    EC* ec = ecs + x;
    if (n >= nr) {
      ec->tok[n] = -1;
      ec->wt[n] = 0.f;
      ec->inv[n] = 0.f;
    }
    if (n == 0) {
      const long long pre = sm.pseg[e] + NMAX * k;
      reinterpret_cast<int4*>(ec)[0] = make_int4(e, nr, (nr + 7) & ~7, sm.seg[e] + NMAX * k);
      reinterpret_cast<longlong2*>(ec)[1] = make_longlong2(pre * 80 * 128, pre * 9 * 128);
      if (ec8 && nr <= 8) ec8[sm.brk[e]] = x;  // n_pad 8 = the expert's last chunk, bucket 1, rank brk
    }
  } else {
    const int jt = jj - mec * NMAX, x = jt / 49, q = jt - 49 * x;
    if (x >= n_ec) return;
    const int e = sm.ecmap[x], k = x - sm.ecb[e], nr = min(NMAX, sm.cnt[e] - NMAX * k), np = (nr + 7) & ~7;
    const int pos0 = np == NMAX ? sm.f64[e] + k : sm.bbase[sm.lb[e]] + sm.brk[e];
    const long long pre = sm.pseg[e] + NMAX * k;
    if (ec8 && np == 8) return;  // route2x: n_pad-8 ECs run in the row-stream kernels (ec8 list), not as tiles
    const int nbig = ec8 ? sm.bbase[1] : n_ec;  // bbase[1] = n_ec - (number of n_pad-8 ECs): they rank last
    if (q < 9) {
      const int u = pos0 * 9 + q, n1 = 9 * nbig, r = u / grid, pos = u - r * grid, L = min(grid, n1 - r * grid);
      t1[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, q, e, np, pre * 80 * 128};
    } else {
      const int qq = q - 9, u = pos0 * 40 + qq, n2 = 40 * nbig, r = u / grid, pos = u - r * grid,
                L = min(grid, n2 - r * grid);
      t2[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, qq, e, np, pre * 9 * 128};
    }
  }
}

// Items: P pairs and mec x 113 EC items (mec = max_ec(P), known on the host), both split evenly over the CTAs.
__global__ void __launch_bounds__(F_THREADS) k_fill(const int* __restrict__ rs, const float* __restrict__ w,
                                                    const float* __restrict__ inv, int P, int G, int mec, int tshift,
                                                    EC* __restrict__ ecs, Tile* __restrict__ t1, Tile* __restrict__ t2,
                                                    int* __restrict__ counts, int* __restrict__ slotmap, int grid,
                                                    int* __restrict__ ec8) {
  __shared__ __align__(16) FSmem sm;
  extern __shared__ __align__(16) unsigned short offE[];  // [G][NEXP] exclusive prefix over the k_rank CTAs
  const int tid = threadIdx.x;
  DBG_GMIN(22)
  griddep_wait();  // no-op without PDL
  griddep_launch();  // the next kernel may be scheduled; a PDL-launched k_rank touches global memory only after its wait
  DBG_GMIN(23)
  DBG_CLK(8)
  // the first pair item's loads stay in flight during the prologue
  const int pch = (P + gridDim.x - 1) / gridDim.x, j0p = blockIdx.x * pch + tid;
  const bool has0 = tid < pch && j0p < P;
  const int pk0 = has0 ? rs[RS_PK + j0p] : -1;
  const float w0 = has0 ? w[j0p] : 0.f, iv0 = (inv && has0) ? inv[j0p / TOPK] : 0.f;
  int c = 0;  // per-expert exclusive prefix over the k_rank CTAs, total
  if (tid < NEXP) {
    for (int g0 = 0; g0 < G; g0 += 8) {
      int v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = g0 + u < G ? rs[RS_HIST + (g0 + u) * NEXP + tid] : 0;
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (g0 + u < G) {
          offE[(g0 + u) * NEXP + tid] = (unsigned short)c;
          c += v[u];
        }
    }
  }
  DBG_CLK(9)
  expert_scan(sm, c, P, counts, ec8);
  DBG_CLK(10)
  const int n_ec = sm.n_ec;
  // CTA b's items: its pair chunk [b pch, + pch) (pch = ceil(P / grid): the scattered EC stores are spread over all
  // SMs), then its chunk of the EC items [b ech, + ech) (ech = ceil(mec x 113 / grid)); consecutive threads take
  // consecutive items
  const int nE = mec * (NMAX + 49), ech = (nE + gridDim.x - 1) / gridDim.x;
  const int pb = blockIdx.x * pch, np_ = max(0, min(pch, P - pb));
  const int eb = blockIdx.x * ech, ne_ = max(0, min(ech, nE - eb));
  for (int i = tid; i < np_ + ne_; i += F_THREADS) {
    if (i < np_) {
      const int j = pb + i;
      const int pk = i == tid ? pk0 : rs[RS_PK + j];
      const float wj = i == tid ? w0 : w[j], ivj = i == tid ? iv0 : (inv ? inv[j / TOPK] : 0.f);
      if (pk < 0) slotmap[j] = -1;
      else put_pair(sm, j, pk & 511, offE[(j >> tshift) * NEXP + (pk & 511)] + (pk >> 9), wj, ivj, ecs, slotmap);
    } else {
      ec_item(sm, eb + i - np_, mec, n_ec, grid, ecs, t1, t2, ec8);
    }
  }
  DBG_SYNC()
  DBG_CLK(11)
  DBG_GMAX(24)
}

// k_one: the whole table build in one kernel for small T (S = ceil(6T / 256) <= ONE_MAXS segments of 256 pairs).
// Every CTA loads all 6T ids and counts them per segment and expert (SMEM atomics: the counts do not depend on the
// order), so it knows the global picture without a second kernel; CTA b < S also ranks segment b's pairs exactly (one
// ballot round in each of its first 8 warps + an exclusive scan over those warps) and scatters them; the EC items are
// split evenly over all CTAs.
constexpr int ONE_SEG = 256, ONE_MAXS = 32;  // P <= 8192 (T <= 1365)
constexpr int O_RW = ONE_SEG / 32, O_LOADS = ONE_MAXS * ONE_SEG / F_THREADS;  // 8 ranking warps, 16 ids per thread
__global__ void __launch_bounds__(F_THREADS) k_one(const int* __restrict__ ids, const float* __restrict__ w,
                                                   const float* __restrict__ inv, int P, int S, int mec,
                                                   EC* __restrict__ ecs, Tile* __restrict__ t1, Tile* __restrict__ t2,
                                                   int* __restrict__ counts, int* __restrict__ slotmap, int grid,
                                                   int* __restrict__ ec8, int rstride = 0) {
  // rstride > 0 (route2r): ids / w are read straight from the 64-B routing rows (int32 ids at row * rstride + k, fp32 w
  // at the same index of w = rows + 6): no ids / wts copies; values identical
  auto px = [rstride](int p) { return rstride ? (p / TOPK) * rstride + p % TOPK : p; };
  __shared__ __align__(16) FSmem sm;
  __shared__ __align__(16) unsigned short wh[O_RW][NEXP];  // segment b: per-warp counts -> prefix over the warps
  __shared__ int offB[NEXP];                                // segment b: pairs of e in the segments before b
  extern __shared__ __align__(16) int cntS[];               // [S][NEXP] pairs per segment and expert
  const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, b = blockIdx.x;
  DBG_GMIN(20)
  for (int i = tid; i < S * NEXP / 4; i += F_THREADS) reinterpret_cast<int4*>(cntS)[i] = make_int4(0, 0, 0, 0);
  if (wp < O_RW)
    for (int i = lane; i < NEXP / 8; i += 32) reinterpret_cast<int4*>(wh[wp])[i] = make_int4(0, 0, 0, 0);
  griddep_wait();  // no-op without PDL
  griddep_launch();
  DBG_CLK(0)
  int idv[O_LOADS];  // pair tid + 512 i: segment 2 i + (tid >= 256)
#pragma unroll
  for (int i = 0; i < O_LOADS; ++i) {
    const int p = tid + F_THREADS * i;
    const int id = p < P ? __ldg(ids + px(p)) : -1;
    idv[i] = (unsigned)id < (unsigned)NEXP ? id : -1;
  }
  const int pown = ONE_SEG * b + tid;
  const bool has_own = b < S && tid < ONE_SEG && pown < P;
  int eown = -1;
  if (b < S && tid < ONE_SEG) {
    const int id = pown < P ? __ldg(ids + px(pown)) : -1;
    eown = (unsigned)id < (unsigned)NEXP ? id : -1;
  }
  const float wown = has_own ? w[px(pown)] : 0.f, ivown = (has_own && inv) ? inv[pown / TOPK] : 0.f;
  __syncthreads();  // cntS / wh zeroed
  const int sh = tid >= ONE_SEG ? 1 : 0;
#pragma unroll
  for (int i = 0; i < O_LOADS; ++i)
    if (idv[i] >= 0) atomicAdd(&cntS[(2 * i + sh) * NEXP + idv[i]], 1);
  int rk = 0;
  if (b < S && wp < O_RW) {  // warp-uniform: rank of the own pair inside its warp (one round), warp counts
    const unsigned peers = peers_of(eown & 511), lt = (1u << lane) - 1u;
    rk = __popc(peers & lt);
    if (eown >= 0 && (peers & lt) == 0) wh[wp][eown] = (unsigned short)__popc(peers);
  }
  __syncthreads();
  DBG_CLK(1)
  int c = 0;
  if (tid < NEXP) {  // totals and the prefix over the segments before b
    int ob = 0;
#pragma unroll
    for (int s = 0; s < ONE_MAXS; ++s)
      if (s < S) {
        const int v = cntS[s * NEXP + tid];
        ob += s < b ? v : 0;
        c += v;
      }
    offB[tid] = ob;
  } else if (tid < NEXP + NEXP / 4 && b < S) {  // warps 12..14: exclusive prefix over the 8 warps, packed uint16 x 4
    const int t4 = tid - NEXP;
    unsigned long long v[O_RW];
#pragma unroll
    for (int ww = 0; ww < O_RW; ++ww) v[ww] = reinterpret_cast<const unsigned long long*>(wh[ww])[t4];
    unsigned long long run = 0;
#pragma unroll
    for (int ww = 0; ww < O_RW; ++ww) {
      reinterpret_cast<unsigned long long*>(wh[ww])[t4] = run;
      run += v[ww];
    }
  }
  expert_scan(sm, c, P, counts, ec8);  // its first __syncthreads also publishes offB / the warp prefix
  DBG_CLK(2)
  const int n_ec = sm.n_ec, nE = mec * (NMAX + 49), ech = (nE + gridDim.x - 1) / gridDim.x;
  const int np_ = b < S ? min(ONE_SEG, P - ONE_SEG * b) : 0;
  const int eb = b * ech, ne_ = max(0, min(ech, nE - eb));
  for (int i = tid; i < np_ + ne_; i += F_THREADS) {
    if (i < np_) {  // i == tid: the own pair
      if (eown < 0) slotmap[pown] = -1;
      else put_pair(sm, pown, eown, offB[eown] + wh[wp][eown] + rk, wown, ivown, ecs, slotmap);
    } else {
      ec_item(sm, eb + i - np_, mec, n_ec, grid, ecs, t1, t2, ec8);
    }
  }
  DBG_SYNC()
  DBG_CLK(3)
  DBG_GMAX(21)
}

}  // namespace msr2

// ------------------------------------------------------------------------------------------------ bindings
using namespace msr2;
static int g_sms = 0;
static int g_pdl = 3;   // bit 0: k_fill with programmatic dependent launch, bit 1: k_rank too
static int g_attr_dev = -1;
static int g_fill = 1;  // k_fill CTAs (512 threads) per SM, cap (sweep: 1 >= 2 = 4 > fewer)
static int g_ipt = 0;   // k_rank pairs per thread; 0 = auto
static int g_one_max_t = 1365;  // T <= this: single-kernel k_one (<= 1365); 0 = always the two-kernel path

int64_t route2_ipt(int64_t T) {  // auto rule: 1 pair per thread up to T = 4096 (G <= 24), then 2
  if (g_ipt) return g_ipt;
  return T <= 4096 ? 1 : 2;
}

// Drop-in for moe_raw's route(...): same arguments, buffers sized by route2_max_ec(T) / route2_scratch_ints().
// route2x = route2 + ec8 (int32 >= route2_max_ec(T), or empty): the EC indices with n_pad 8 in their snake rank order,
// their count in counts[6] (needs counts.numel() >= 7); for moe_raw's row-stream GEMM1 (gemm1r8).
void route2x(torch::Tensor ids, torch::Tensor w, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2,
             torch::Tensor counts, torch::Tensor slotmap, torch::Tensor rts, int64_t grid, torch::Tensor inv,
             torch::Tensor ec8t);
void route2(torch::Tensor ids, torch::Tensor w, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2,
            torch::Tensor counts, torch::Tensor slotmap, torch::Tensor rts, int64_t grid, torch::Tensor inv) {
  route2x(ids, w, T, ecs, t1, t2, counts, slotmap, rts, grid, inv, torch::Tensor());
}
void route2x(torch::Tensor ids, torch::Tensor w, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2,
             torch::Tensor counts, torch::Tensor slotmap, torch::Tensor rts, int64_t grid, torch::Tensor inv,
             torch::Tensor ec8t) {
  TORCH_CHECK(T >= 0 && T <= MAXT, "route2: T must be in 0..", MAXT);
  const int P = (int)T * TOPK, mec = max_ec(P);
  TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.numel() >= P,
              "route2: ids int32 [T, 6] contiguous");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() && w.numel() >= P, "route2: w fp32 [T, 6]");
  TORCH_CHECK(ecs.nbytes() >= (size_t)mec * sizeof(EC) && t1.nbytes() >= (size_t)mec * 9 * sizeof(Tile) &&
                  t2.nbytes() >= (size_t)mec * 40 * sizeof(Tile),
              "route2: ecs / t1 / t2 smaller than route2_max_ec(T) records");
  TORCH_CHECK(counts.scalar_type() == torch::kInt32 && counts.numel() >= 6, "route2: counts int32 >= 6");
  TORCH_CHECK(slotmap.scalar_type() == torch::kInt32 && slotmap.numel() >= P, "route2: slotmap int32 >= 6T");
  TORCH_CHECK(rts.scalar_type() == torch::kInt32 && rts.numel() >= RS_INTS, "route2: scratch smaller than route2_scratch_ints()");
  TORCH_CHECK(((uintptr_t)ecs.data_ptr() & 15) == 0 && ((uintptr_t)rts.data_ptr() & 15) == 0 &&
                  ((uintptr_t)t1.data_ptr() & 7) == 0 && ((uintptr_t)t2.data_ptr() & 7) == 0,
              "route2: ecs / scratch 16-B, tiles 8-B aligned");
  TORCH_CHECK(inv.numel() == 0 || (inv.scalar_type() == torch::kFloat32 && inv.numel() >= T), "route2: inv fp32 [T] or empty");
  int* ec8 = nullptr;
  if (ec8t.defined() && ec8t.numel() > 0) {
    TORCH_CHECK(ec8t.scalar_type() == torch::kInt32 && ec8t.numel() >= mec && counts.numel() >= 7,
                "route2x: ec8 int32 >= route2_max_ec(T), counts >= 7");
    ec8 = ec8t.data_ptr<int>();
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, ids.get_device());
  if (g_attr_dev != ids.get_device()) {  // k_fill: up to MAXG x 384 uint16 of dynamic SMEM (+ ~14 KB static)
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_fill, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)(MAXG * NEXP * sizeof(unsigned short))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_one, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)(ONE_MAXS * NEXP * sizeof(int))));
    g_attr_dev = ids.get_device();
  }
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  const float* invp = inv.numel() ? inv.data_ptr<float>() : nullptr;
  const int sg = grid > 0 ? (int)grid : g_sms;
  if (T > 0 && T <= g_one_max_t) {  // ---- one kernel
    const int S = (P + ONE_SEG - 1) / ONE_SEG, nE = mec * (NMAX + 49);
    cudaLaunchConfig_t c0 = {};
    c0.gridDim = dim3(std::max(S, std::min((nE + F_THREADS - 1) / F_THREADS, g_sms)));
    c0.blockDim = dim3(F_THREADS);
    c0.dynamicSmemBytes = (size_t)S * NEXP * sizeof(int);
    c0.stream = stream;
    c0.attrs = at;
    c0.numAttrs = (g_pdl & 2) ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&c0, k_one, (const int*)ids.data_ptr<int>(), (const float*)w.data_ptr<float>(),
                                      invp, P, S, mec, (EC*)ecs.data_ptr(), (Tile*)t1.data_ptr(), (Tile*)t2.data_ptr(),
                                      counts.data_ptr<int>(), slotmap.data_ptr<int>(), sg, ec8, 0));
    return;
  }
  const int ipt = (int)route2_ipt(T), tshift = 10 + __builtin_ctz(ipt), tile = R1_THREADS * ipt;
  const int G = P > 0 ? (P + tile - 1) / tile : 1;
  cudaLaunchConfig_t c1 = {};
  c1.gridDim = dim3(G);
  c1.blockDim = dim3(R1_THREADS);
  c1.stream = stream;
  c1.attrs = at;
  c1.numAttrs = (g_pdl & 2) ? 1 : 0;
  auto* kr = ipt == 1 ? k_rank<1> : ipt == 2 ? k_rank<2> : ipt == 4 ? k_rank<4> : ipt == 8 ? k_rank<8> : k_rank<16>;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c1, kr, (const int*)ids.data_ptr<int>(), P, rts.data_ptr<int>()));
  const int items = P + mec * (NMAX + 49);
  const int g2 = std::max(1, std::min((items + F_THREADS - 1) / F_THREADS, g_fill > 0 ? g_fill * g_sms : -g_fill));
  cudaLaunchConfig_t c2 = {};
  c2.gridDim = dim3(g2);
  c2.blockDim = dim3(F_THREADS);
  c2.dynamicSmemBytes = (size_t)G * NEXP * sizeof(unsigned short);
  c2.stream = stream;
  c2.attrs = at;
  c2.numAttrs = (g_pdl & 1) ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c2, k_fill, (const int*)rts.data_ptr<int>(), (const float*)w.data_ptr<float>(),
                                    invp, P, G, mec, tshift, (EC*)ecs.data_ptr(), (Tile*)t1.data_ptr(),
                                    (Tile*)t2.data_ptr(), counts.data_ptr<int>(), slotmap.data_ptr<int>(), sg, ec8));
}
// route2 reading ids / weights straight from the routing rows [T, 64 B] (int32 ids at bytes 0..23, fp32 weights at
// 24..47; sp_layer's gather buffer gr): no .contiguous() copies. One-kernel path only (T <= the k_one limit).
void route2r(torch::Tensor rows, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2, torch::Tensor counts,
             torch::Tensor slotmap, torch::Tensor rts, int64_t grid) {
  TORCH_CHECK(T > 0 && T <= g_one_max_t && rows.is_cuda() && rows.nbytes() >= (size_t)T * 64, "route2r: rows [T, 64 B], T <= k_one limit");
  const int P = (int)T * TOPK, mec = max_ec(P);
  TORCH_CHECK(ecs.nbytes() >= (size_t)mec * sizeof(EC) && t1.nbytes() >= (size_t)mec * 9 * sizeof(Tile) &&
                  t2.nbytes() >= (size_t)mec * 40 * sizeof(Tile) && counts.numel() >= 6 && slotmap.numel() >= P &&
                  rts.numel() >= RS_INTS,
              "route2r: buffers");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, rows.get_device());
  if (g_attr_dev != rows.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_fill, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)(MAXG * NEXP * sizeof(unsigned short))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_one, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)(ONE_MAXS * NEXP * sizeof(int))));
    g_attr_dev = rows.get_device();
  }
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  const int sg = grid > 0 ? (int)grid : g_sms;
  const int S = (P + ONE_SEG - 1) / ONE_SEG, nE = mec * (NMAX + 49);
  cudaLaunchConfig_t c0 = {};
  c0.gridDim = dim3(std::max(S, std::min((nE + F_THREADS - 1) / F_THREADS, g_sms)));
  c0.blockDim = dim3(F_THREADS);
  c0.dynamicSmemBytes = (size_t)S * NEXP * sizeof(int);
  c0.stream = stream;
  c0.attrs = at;
  c0.numAttrs = (g_pdl & 2) ? 1 : 0;
  const int* r32 = (const int*)rows.data_ptr();
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c0, k_one, r32, (const float*)(r32 + 6), (const float*)nullptr, P, S, mec,
                                    (EC*)ecs.data_ptr(), (Tile*)t1.data_ptr(), (Tile*)t2.data_ptr(), counts.data_ptr<int>(),
                                    slotmap.data_ptr<int>(), sg, (int*)nullptr, 16));
}
int64_t route2_max_ec(int64_t T) { return max_ec((int)T * TOPK); }
int64_t route2_scratch_ints() { return RS_INTS; }
int64_t route2_max_t() { return MAXT; }
// tuning knobs (defaults 3, 1, 0, 1365): pdl bit 0 = k_fill PDL, bit 1 = k_rank / k_one PDL; k_fill CTAs per SM (< 0:
// absolute cap); ipt 0 = auto; one_max_t: T up to which the single-kernel path runs (0..1365)
void route2_config(int64_t pdl, int64_t fill_ctas_per_sm, int64_t ipt, int64_t one_max_t) {
  TORCH_CHECK(ipt == 0 || ipt == 1 || ipt == 2 || ipt == 4 || ipt == 8 || ipt == 16, "route2_config: ipt");
  TORCH_CHECK(one_max_t >= 0 && one_max_t * TOPK <= ONE_MAXS * ONE_SEG, "route2_config: one_max_t <= 1365");
  g_pdl = (int)pdl;
  if (fill_ctas_per_sm != 0) g_fill = (int)fill_ctas_per_sm;  // < 0: absolute k_fill CTA cap
  g_ipt = (int)ipt;
  g_one_max_t = (int)one_max_t;
}
int64_t route2_one_max_t() { return g_one_max_t; }

#if R2_DBG
__global__ void k_null1() {}
__global__ void k_null2() { griddep_wait(); }
void route2_null(int64_t nk, int64_t pdl, int64_t g1, int64_t g2) {  // launch floor: 1 or 2 empty kernels
  auto stream = at::cuda::getCurrentCUDAStream();
  k_null1<<<(unsigned)g1, R1_THREADS, 0, stream>>>();
  if (nk < 2) return;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)g2);
  cfg.blockDim = dim3(256);
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_null2));
}
std::vector<int64_t> route2_dbg() {  // synchronizes
  long long h[32];
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(h, g_dbg, sizeof(h)));
  return std::vector<int64_t>(h, h + 32);
}
void route2_dbg_reset() {
  long long h[32] = {0};
  for (int i : {20, 22, 23}) h[i] = 0x7fffffffffffffffll;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(g_dbg, h, sizeof(h)));
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route2", &route2);
  m.def("route2x", &route2x);
  m.def("route2_max_ec", &route2_max_ec);
  m.def("route2r", &route2r);
  m.def("route2_scratch_ints", &route2_scratch_ints);
  m.def("route2_max_t", &route2_max_t);
  m.def("route2_config", &route2_config);
  m.def("route2_ipt", &route2_ipt);
  m.def("route2_one_max_t", &route2_one_max_t);
#if R2_DBG
  m.def("route2_null", &route2_null);
  m.def("route2_dbg", &route2_dbg);
  m.def("route2_dbg_reset", &route2_dbg_reset);
#endif
}
