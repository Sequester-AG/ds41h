// k_gather6 (moe_v4): the decode MoE's token-image gather (moe_raw.cu prepr5 / k_gather5), same image
// bytes (real and padding rows) and the same ec.inv, with each token row brought into SMEM once by a TMA bulk copy.
// k_gather5 (persistent, 640 threads = one slot row, 4 rows in flight) loads every 64-B source block 4 times (the 4 threads
// that emit its 4 output chunks) and waits one dependent round trip per 4 rows: 45 us for a 75-MB image at T = 1,024
// (1.7 TB/s). Here: per block a producer warp resolves 32 slots at a time (k_gather5's binary search over the EC slot
// bases), writes ec.inv, and bulk-copies each slot's 10-KB token row into a G6_NSLOT-entry ring; 20 consumer warps emit
// the 80 x 8 16-B chunks of the row (k_gather5's emit8 / swz, verbatim) and store them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef G6_NSLOT
#define G6_NSLOT 8
#endif
#ifndef G6_B
#define G6_B 1  // blocks per SM
#endif
namespace g6 {
constexpr int D = 5120, NMAX = 64, ROW = D * 2, NW = 20, THR = (NW + 1) * 32, MAXE = 512, NSLOT = G6_NSLOT;
struct EC {  // moe_raw.cu's EC (route2's records)
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
  float inv[NMAX];
};
static_assert(sizeof(EC) == 32 + 3 * 4 * NMAX, "EC layout");
// ---- moe_raw.cu, verbatim
__host__ __device__ __forceinline__ int kperm(int L) {
  const int s = L >> 4, kk = L & 15;
  return 8 * ((kk & 7) >> 1) + 4 * (kk >> 3) + 2 * (kk & 1) + s;
}
__device__ __forceinline__ unsigned swz(int n, int b) {
  return (n >> 3) * 1024 + (n & 7) * 128 + ((((b >> 4) ^ (n & 7)) & 7) << 4) + (b & 15);
}
__device__ __forceinline__ float bf_at(const uint4 (&v)[4], int p) {
  const int w = p >> 1, q = w >> 2, c = w & 3;
  const unsigned u = c == 0 ? v[q].x : c == 1 ? v[q].y : c == 2 ? v[q].z : v[q].w;
  return __uint_as_float((p & 1) ? (u & 0xffff0000u) : (u << 16));
}
template <int C>
__device__ __forceinline__ uint4 emit8(const uint4 (&v)[4], float sc) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 h = __halves2half2(__float2half_rn(bf_at(v, kperm(8 * C + 2 * i)) * sc),
                                     __float2half_rn(bf_at(v, kperm(8 * C + 2 * i + 1)) * sc));
    w[i] = *(const unsigned*)&h;
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
// ---- TMA / mbarrier
__device__ __forceinline__ uint32_t s_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(s_u32(b)),
               "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s_u32(dst)),
               "l"(src), "r"(bytes), "r"(s_u32(b)) : "memory");
}

struct Meta {
  long long soff;  // image byte offset of the EC's slab 0
  int sstr, n, tok;
  float sc;
};

__global__ void __launch_bounds__(THR, G6_B) k_gather6(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                       const int* __restrict__ ptok, char* __restrict__ img,
                                                       const int* __restrict__ counts, int pstride) {
  extern __shared__ __align__(128) unsigned char sm[];
  unsigned char* ring = sm;                                     // [NSLOT][ROW]
  Meta* meta = (Meta*)(sm + NSLOT * ROW);                       // [NSLOT]
  int* sbase = (int*)(meta + NSLOT);                            // [MAXE + 1]
  uint64_t* full = (uint64_t*)(sbase + MAXE + 8);               // [NSLOT]
  uint64_t* empty = full + NSLOT;                               // [NSLOT]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nec = counts[0], S = counts[3] / (80 * 128), G = gridDim.x;
  for (int i = tid; i < nec; i += blockDim.x) sbase[i] = (int)(ecs[i].img_off / (80 * 128));
  if (tid == 0) {
    sbase[nec] = S;
    for (int k = 0; k < NSLOT; ++k) mb_init(&full[k], 1), mb_init(&empty[k], NW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nmine = (int)blockIdx.x < S ? (S - 1 - (int)blockIdx.x) / G + 1 : 0;
  if (warp == NW) {  // ---------------- producer warp
    for (int i0 = 0; i0 < nmine; i0 += 32) {
      // resolve 32 slots in parallel: slot j -> (EC lo, n, token, scale); ec.inv as k_gather5
      const int i = i0 + lane;
      long long soff = 0;
      int sstr = 0, n = 0, tok = -1;
      float sc = 0.f;
      if (i < nmine) {
        const int j = (int)blockIdx.x + G * i;
        int lo = 0, hi = nec - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (sbase[mid] <= j) lo = mid; else hi = mid - 1;
        }
        n = j - sbase[lo];
        const int np = sbase[lo + 1] - sbase[lo];
        tok = ecs[lo].tok[n];
        float iv = 0.f;
        if (tok >= 0) {
          const int p = ptok[(size_t)tok * pstride];
          sc = exp2f((float)p), iv = exp2f((float)-p);
        }
        ecs[lo].inv[n] = iv;
        soff = (long long)sbase[lo] * (80 * 128);
        sstr = np * 128;
      }
      const int cnt = min(32, nmine - i0);
      for (int l = 0; l < cnt; ++l) {
        const int ii = i0 + l, k = ii % NSLOT;
        const long long so = __shfl_sync(~0u, soff, l);
        const int st = __shfl_sync(~0u, sstr, l), nn = __shfl_sync(~0u, n, l), tk = __shfl_sync(~0u, tok, l);
        const float s = __shfl_sync(~0u, sc, l);
        if (lane == 0) {
          if (ii >= NSLOT) mb_wait(&empty[k], ((ii / NSLOT) - 1) & 1);
          meta[k] = Meta{so, st, nn, tk, s};
          if (tk >= 0) {
            mb_expect(&full[k], ROW);
            bulk_g2s(ring + (size_t)k * ROW, x + (size_t)tk * D, ROW, &full[k]);
          } else {
            mb_arrive(&full[k]);  // padding slot: zeros, nothing to load
          }
        }
        __syncwarp();
      }
    }
  } else {  // ---------------- consumers: thread = (slab, cc), k_gather5's mapping
    const int slab = tid >> 3, cc = tid & 7, blk = cc >> 2, c = cc & 3;
    for (int i = 0; i < nmine; ++i) {
      const int k = i % NSLOT;
      mb_wait(&full[k], (i / NSLOT) & 1);
      const Meta m = meta[k];
      uint4 o = make_uint4(0, 0, 0, 0);
      if (m.tok >= 0) {
        const uint4* src = (const uint4*)(ring + (size_t)k * ROW + slab * 128 + blk * 64);
        uint4 v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = src[q];
        switch (c) {
          case 0: o = emit8<0>(v, m.sc); break;
          case 1: o = emit8<1>(v, m.sc); break;
          case 2: o = emit8<2>(v, m.sc); break;
          default: o = emit8<3>(v, m.sc); break;
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&empty[k]);  // this warp is done reading slot k's row and meta
      *(uint4*)(img + m.soff + (size_t)slab * m.sstr + swz(m.n, cc * 16)) = o;
    }
  }
}
constexpr int SMEM = NSLOT * ROW + NSLOT * (int)sizeof(Meta) + (MAXE + 8) * 4 + 2 * NSLOT * 8;
}  // namespace g6

// = moe_raw.cu prepr5(x, ecs, counts, img, rows, sms, ctr) (MSR_G5NOPAD 0): routing rows [T, 64 B] with the prescale p
// at byte 48 (int 12)
static unsigned long long g_attr = 0;
void gather6(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img, torch::Tensor rows, int64_t sms) {
  using namespace g6;
  const c10::cuda::CUDAGuard dg(x.device());
  TORCH_CHECK(rows.numel() >= x.size(0) * 64, "gather6: routing rows [T, 64]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.size(1) == D && x.stride(0) == D && ((uintptr_t)x.data_ptr() % 16) == 0,
              "gather6: x [T, 5120] bf16 rows contiguous");
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!(g_attr >> dev & 1ull)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_gather6, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    g_attr |= 1ull << dev;
  }
  k_gather6<<<(unsigned)(G6_B * sms), THR, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(), (const int*)rows.data_ptr() + 12, (char*)img.data_ptr(),
      counts.data_ptr<int>(), 16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gather6", &gather6); }
