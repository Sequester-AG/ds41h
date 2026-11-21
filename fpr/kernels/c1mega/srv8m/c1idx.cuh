// c1idx.cuh: index-layer phases for c1mega. Self-contained (namespace c1ix, own PTX helpers): c1mega.cu includes it at file scope
// and calls the phase functions from k_layer
//
// Phases here (owner GPU of the C1 sequence; 256 consumer threads = 2 warpgroups, named barrier 1 as c1mega's cbar):
//   X4 logits_tile: one 64-key tile of an index layer on one warpgroup, DeepGEMM sm90_fp8_paged_mqa_logits arithmetic
//      exactly as kernels/indexer2f/idx2f_fcand8.cu (NEXT_N 4: A = the key tile, B = Q of the 4 rows, m64n128k32 e4m3 SS
//      x 4 k-steps, relu * weight with 4 intra-thread sums, * kv scale, 2 xor shuffles), then the logits of the rows'
//      positions < n and the 8-key block maxima (= idx2f block_max: the largest order-preserving key, positions < n).
//   X5 select_row: the exact top-512 of a row (canonical order: key desc, lowest position on ties; = idx2f exact_topk)
//      in ascending position order, n <= 512 -> 0 .. n-1, then FlashMLA's global slots bt[p / bs] * bs + p % bs and the
//      length (= ftopk's GlobalOut). Block threshold tau (24 high bits of the 512-th largest block key) -> the elements
//      >= tau of the blocks >= tau in position order -> 4-pass radix select; too many such elements: the same select
//      streamed over the row.
//   X5 cand_row (layer 20): the candidate blocks of the consumer layers (= idx2f select_blocks from lens): the newest
//      block pinned to +inf, the top-2048 block keys (lowest id on ties), -inf blocks dropped, ascending, -1 padded.
// Bit-exactness: given the same q8 / weights / keys, logits and block maxima are bit-identical to fcand8 + block_max
// (same instruction, operand layouts and epilogue statements; c1mega's flags = DeepGEMM's: -O3, no fast math), and the
// selected sets are the unique canonical top-512 / top-2048, so ti / tl / cand are identical.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace c1ix {
constexpr int NRM = 4, HEADS = 32, HD = 128, TILE = 64, BLK = 8, TOPK = 512, CAND_K = 2048;
constexpr int CTH = 256;                       // consumer threads (8 warps, 2 warpgroups)
constexpr int QIMG = NRM * HEADS * HD;         // Q image bytes (SW128, 128 rows x 128 B)
constexpr uint32_t NEG_INF_KEY = 0x007fffffu;  // ukey(-inf)
constexpr uint32_t POS_INF_KEY = 0xff800000u;  // ukey(+inf)
// selection scratch (shared memory, 16-B aligned): the histograms (per-warp 256-bin or one 4,096-bin), scan words, the
// candidates (keys + 16-bit positions; first the staged keys of the row or its block maxima), the compaction list (also
// the selected blocks), the row's 512 positions
constexpr int RS_BINS = 2048;                  // 11-bit radix digits (windows below the keys' highest differing bit)
constexpr int CMAX = 5952;                     // candidates (rows up to 5,952 positions: every element)
// staged keys / candidates are stored one padding word per 32 (index i at i + i / 32): thread-contiguous chunks of 16 or 32
// items then read conflict-free (32-way / 16-way bank conflicts otherwise)
__device__ __forceinline__ int pidx(int i) { return i + (i >> 5); }
// Shared memory across the __noinline__ phase functions (v9): a pointer parameter is generic (LD.E / ST.E / ATOM.E), and
// the test kernels' 1024-B alignment (via uintptr_t) hides the space even where the compiler could infer it: v8's selection
// ran 1.3-2x slower than v7's inlined code. The noinline phases take byte offsets into the dynamic shared memory and
// rebuild their pointers from this array (every extern __shared__ array aliases the dynamic base): LDS / STS / ATOMS.
extern __shared__ __align__(16) char c1ix_dsm[];
__device__ __forceinline__ int soff(const void* p) {
  return (int)(reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(c1ix_dsm));
}
template <typename T = char>
__device__ __forceinline__ T* sptr(int o) {
  return reinterpret_cast<T*>(c1ix_dsm + o);
}
constexpr int LCAP = 2048;                     // compaction list (keys) / selected blocks
constexpr int SBMAX = LCAP;                     // (16-bit block ids in the first half of the list region)
constexpr int SC_HIST = 0, SC_WS = SC_HIST + 2 * RS_BINS * 4, SC_MISC = SC_WS + 32 * 4, SC_CKEY = SC_MISC + 16 * 4,
              SC_CPOS = SC_CKEY + (CMAX + CMAX / 32) * 4, SC_LKEY = SC_CPOS + CMAX * 2, SC_POS = SC_LKEY + LCAP * 4,
              SC_BYTES = SC_POS + TOPK * 4;
// counters (unsigned [IX_N], monotonic, targets 132 x the index-layer epoch `icall` of the CTA: every CTA arrives once
// per index layer on each, owner GPU only): X1 done (ks / weights_proj), X3 done (q), X2 done (new keys), logits, top-k,
// (v14) the distributed selection's candidate lists complete
enum { IX_LG = 0, IX_SEL = 1, IX_KS = 2, IX_Q = 3, IX_K = 4, IX_CL = 5, IX_N = 8 };
// v14 distributed selection (see ds_stage1): digits = the top 11 bits of the order-preserving key; global scratch IdxArgs
// dsc (u32 words, zero at allocation; each user zeroes what it used): DS_H element-digit histograms [NRM][2048], DS_HB
// block-max-digit histograms (layer 20), DS_H2 (v14b) the level-2 histograms of the 8 rows (the next 11 bits of the keys in
// the boundary digit), DS_C list counters ([0, NRM) select rows, [NRM, 2 NRM) candidate rows), DS_LK /
// DS_LP the select rows' lists (keys / positions) [NRM][DCAP], DS_BK / DS_BP the candidate rows' (keys / block ids)
constexpr int DBITS = 11, DSH = 32 - DBITS, DBINS = 1 << DBITS, DCAP = 16384, NCTA_IX = 132;   // (DCAP: v14f 16K; the lists
                                                                                                // live in global memory)
constexpr int DCB = 64;   // (v14d) coarse bins per histogram row (32 digits each): DS_CO + fine row index x 64, the fine rows
                          // being DS_H's 4, DS_HB's 4, DS_H2's 8 (stage 1 / 2 read 64 + 32 words per row, not 2,048)
constexpr int DS_H = 0, DS_HB = DS_H + NRM * DBINS, DS_H2 = DS_HB + NRM * DBINS, DS_CO = DS_H2 + 2 * NRM * DBINS,
              DS_C = DS_CO + 4 * NRM * DCB,
              DS_DBG = DS_C + 16,   // (tests: [L / 4][8 rows] list length | boundary group << 16 of index layer L)
              DS_UMC = DS_DBG + 80, UMC_W = 336,   // (v14f) per-CTA union-map cache [132][8 x 8 x 5 map, T, epoch]
              DS_LK = DS_UMC + NCTA_IX * UMC_W, DS_LP = DS_LK + NRM * DCAP, DS_BK = DS_LP + NRM * DCAP,
              DS_BP = DS_BK + NRM * DCAP,
              DS_WORDS = DS_BP + NRM * DCAP, DS_ZW = DS_C + 8;   // (v14b: [0, DS_ZW) zero between index layers; DS_C + 8 + q
                                                                  // tests' debug: list length | boundary group << 16)

struct IdxArgs {
  int L, ratio, bs;          // layer, compression ratio, positions (records) per page of the KV source (32 / 64)
  const uint8_t* q8;         // [NRM][32][128] e4m3
  const float* wo;           // [NRM][32] fp32
  const int* lens;           // [NRM] compressed row lengths n = (p + 1) / ratio
  const uint8_t* kpages;     // the source's index-K page 0 (paged pool + PAGE_OFFSETS["idx<src>"]): bs x 128 B, bs x fp32
  long long kstride;         // page stride (116,928)
  const int* bt;             // the sequence's block table (packed page ids; position p -> bt[p / bs])
  float* logits;             // [NRM][ldl] fp32
  int ldl;
  float* bmax;               // [NRM][ldb] fp32
  int ldb;
  int* cand;                 // [NRM][CAND_K] (layer 20) or null
  int* gout;                 // [NRM][ldg] global slots (ti)
  int ldg;
  int* glen;                 // [NRM] (tl)
  unsigned* cnt;             // [IX_N]
  const CUtensorMap* tmk;    // index-K pages: 3D {128, bs, pages} strides {128, kstride}, box {128, bs, 1}, SW128
  int* slow;                 // [1] rows that took the streamed select
  int* cand_n;               // layer 20: [NRM] valid entries of each candidate row (written with `cand`)
  const int* cblk;           // consumer layers 24 .. 36: layer 20's candidate rows [NRM][CAND_K] (ascending, -1 padded)
  const int* cblk_n;         // .. and their valid lengths [NRM]
  const CUtensorMap* tmk8;   // consumer layers: the same pages with box {128, 8, 1} (one candidate block = one SW128 atom)
  // bt == null: the row of the persistent per-slot table, bt = dbt + preq[0] x dbt_ld (the engine's dev_bt mode: its
  // per-step index_select folded into the kernel; one sequence per C1 step)
  const int* dbt;
  long long dbt_ld;
  const long long* preq;
  // v14: the distributed selection's global scratch [DS_WORDS] (null: v13's one CTA per row), its row threshold (rows with
  // more than max(dthr, 512) keys; candidate rows past 2,048 blocks), union tiles for the consumer layers' candidate mode
  unsigned* dsc;
  int dthr, uni;
};

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t ukey(float f) {
  const uint32_t b = __float_as_uint(f);
  if ((b & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;  // NaN: highest (torch radix twiddle)
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {   // = DeepGEMM make_smem_desc(p, SWIZZLE_128B, 0, 1024)
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void wg_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n128k32_e4m3(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, "
      "%47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}

// ------------------------------------------------------------------------------------------------ X4 (logits)
// Q of the 4 rows -> the SW128 image (row i = r * 32 + h, 16-B chunk j at j ^ (i & 7)) + the weights; all consumers.
__device__ __forceinline__ void load_q(const IdxArgs& a, char* qimg, float* sw, int tid) {
#pragma unroll
  for (int k = 0; k < QIMG / 16 / CTH; ++k) {
    const int q = tid + CTH * k, i = q >> 3, j = q & 7;
    const uint4 v = __ldcg(reinterpret_cast<const uint4*>(a.q8) + q);
    *reinterpret_cast<uint4*>(qimg + i * 128 + ((j ^ (i & 7)) << 4)) = v;
  }
  if (tid < NRM * HEADS) sw[tid] = __ldcg(a.wo + tid);
}

// One 64-key tile (positions t0 .. t0 + 63) on the calling warpgroup (wq = warp in the group). kv: smem address of the
// SW128 values tile, sc: its 64 fp32 scales, qimg: smem address of the Q image, sw: weights [4][32] (smem), n: the rows'
// lengths (rows >= NR: 0).
__device__ __noinline__ void logits_tile(const IdxArgs& a, uint32_t kv, int sco, uint32_t qimg, int swo, int t0,
                                         const int (&n)[NRM], int wq, int lane, int only_row = -1, int pb0 = -1,
                                         int pb1 = -1, int umo = -1, int hso = -1, int hbo = -1) {
  // (v14: umo >= 0 union mode, the tile's 8 map entries {block, compact index of rows 0..3 or -1} at smem offset umo;
  // hso / hbo >= 0: stage 0 of the distributed selection, the stored keys' digits / the block maxima's digits counted in
  // the CTA's shared histograms, u16- / u8-packed)
  const float* sc = sptr<const float>(sco);   // (offsets: see soff)
  const float* sw = sptr<const float>(swo);
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) wg_fence(acc[i]);
  wg_arrive();
#pragma unroll
  for (int k = 0; k < HD / 32; ++k) wgmma_m64n128k32_e4m3(acc, desc_sw128(kv + 32 * k), desc_sw128(qimg + 32 * k), k);
  wg_commit();
#pragma unroll
  for (int i = 0; i < 64; ++i) wg_fence(acc[i]);
  const int g = lane >> 2, t4 = lane & 3;
  const float scale_kv_0 = sc[wq * 16 + g], scale_kv_1 = sc[wq * 16 + g + 8];
  wg_wait0();
#pragma unroll
  for (int i = 0; i < 64; ++i) wg_fence(acc[i]);
  float ov0[NRM], ov1[NRM];
#pragma unroll
  for (int r = 0; r < NRM; ++r) {   // (verbatim DeepGEMM / fcand8 epilogue)
    float weights[HEADS / 4];
#pragma unroll
    for (int j = 0; j < HEADS / 4; ++j) weights[j] = sw[r * HEADS + (j / 2) * 8 + (j & 1) + t4 * 2];
    const float* shifted_accum = acc + r * (HEADS / 2);
    const auto transform = [&](const int& j) { return fmaxf(shifted_accum[j], 0) * weights[(j / 4) * 2 + (j & 1)]; };
    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
#pragma unroll
    for (int j = 1; j < HEADS / 8; ++j) {
#pragma unroll
      for (int k = 0; k < 4; k++) sum[k] += transform(j * 4 + k);
    }
    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
    float v_1 = (sum[2] + sum[3]) * scale_kv_1;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int offset = 1 << j;
      v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
      v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
    }
    ov0[r] = v_0, ov1[r] = v_1;
  }
  // lane t4 stores row t4 (the quad's lanes hold the same reduced values). Dense: positions p0 = t0 + 16 wq + g, p1 =
  // p0 + 8, stored at the position. Compact (only_row = r, lanes t4 = r): the warp's candidate blocks start at positions
  // pb0 / pb1, stored at compact index t0 + 16 wq + g (+ 8): row r's candidate tile t0 / 64.
  const int rr = t4;
  const float w0 = rr == 0 ? ov0[0] : rr == 1 ? ov0[1] : rr == 2 ? ov0[2] : ov0[3];
  const float w1 = rr == 0 ? ov1[0] : rr == 1 ? ov1[1] : rr == 2 ? ov1[2] : ov1[3];
  const int nr = rr == 0 ? n[0] : rr == 1 ? n[1] : rr == 2 ? n[2] : n[3];
  int i0 = t0 + 16 * wq + g, i1 = i0 + 8;   // store indices
  bool v0, v1;
  int ub0 = -1, ub1 = -1, uc0 = -1, uc1 = -1;   // (union mode: the warp's 2 blocks, row rr's compact block indices)
  if (umo >= 0) {
    const int* um = sptr<const int>(umo);
    ub0 = um[10 * wq], ub1 = um[10 * wq + 5];
    uc0 = ub0 >= 0 ? um[10 * wq + 1 + rr] : -1, uc1 = ub1 >= 0 ? um[10 * wq + 6 + rr] : -1;
    v0 = uc0 >= 0 && 8 * ub0 + g < nr, v1 = uc1 >= 0 && 8 * ub1 + g < nr;
    i0 = 8 * uc0 + g, i1 = 8 * uc1 + g;
  } else if (only_row < 0) {
    v0 = i0 < nr, v1 = i1 < nr;
  } else {
    v0 = rr == only_row && pb0 >= 0 && pb0 + g < nr;
    v1 = rr == only_row && pb1 >= 0 && pb1 + g < nr;
  }
  float* lrow = a.logits + (size_t)rr * a.ldl;
  if (v0) lrow[i0] = w0;
  if (v1) lrow[i1] = w1;
  // 8-key block maxima of row rr: blocks i0 / 8 (tokens g) and + 1 (tokens g + 8); the first max key wins
  uint32_t k0 = v0 ? ukey(w0) : 0u, k1 = v1 ? ukey(w1) : 0u;
  if (hso >= 0) {   // v14 stage 0: the stored keys' digits (row rr: 1,024 words of 2 u16 bins), warp-aggregated (lanes of
                    // one bin share one atomic: same-word shared atomics serialize)
    unsigned* h = sptr<unsigned>(hso);
    const unsigned e0 = v0 ? (unsigned)(rr * DBINS) + (k0 >> DSH) : 0xffffffffu;
    const unsigned e1 = v1 ? (unsigned)(rr * DBINS) + (k1 >> DSH) : 0xffffffffu;
    const unsigned m0 = __match_any_sync(0xffffffffu, e0), m1 = __match_any_sync(0xffffffffu, e1);
    if (v0 && lane == __ffs(m0) - 1) atomicAdd(h + (e0 >> 1), (unsigned)__popc(m0) << (16 * (e0 & 1u)));
    if (v1 && lane == __ffs(m1) - 1) atomicAdd(h + (e1 >> 1), (unsigned)__popc(m1) << (16 * (e1 & 1u)));
  }
  float b0 = w0, b1 = w1;
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    const uint32_t ko0 = __shfl_xor_sync(0xffffffffu, k0, o), ko1 = __shfl_xor_sync(0xffffffffu, k1, o);
    const float bo0 = __shfl_xor_sync(0xffffffffu, b0, o), bo1 = __shfl_xor_sync(0xffffffffu, b1, o);
    if (ko0 > k0) k0 = ko0, b0 = bo0;   // equal keys: equal values (non-NaN)
    if (ko1 > k1) k1 = ko1, b1 = bo1;
  }
  if (g == 0) {
    const int bb = (t0 + 16 * wq) >> 3;
    if (umo >= 0) {
      if (uc0 >= 0 && 8 * ub0 < nr) a.bmax[(size_t)rr * a.ldb + uc0] = b0;
      if (uc1 >= 0 && 8 * ub1 < nr) a.bmax[(size_t)rr * a.ldb + uc1] = b1;
    } else if (only_row < 0) {
      if (8 * bb < nr) a.bmax[(size_t)rr * a.ldb + bb] = b0;
      if (8 * bb + 8 < nr) a.bmax[(size_t)rr * a.ldb + bb + 1] = b1;
      if (hbo >= 0) {   // v14 stage 0 (layer 20): the block maxima's digits, the row's newest block pinned to +inf (u8 bins)
        unsigned* h = sptr<unsigned>(hbo) + rr * (DBINS / 4);
        const int nbl = (nr + BLK - 1) / BLK - 1;
        const uint32_t q0 = bb == nbl ? max(k0, POS_INF_KEY) : k0, q1 = bb + 1 == nbl ? max(k1, POS_INF_KEY) : k1;
        if (8 * bb < nr) atomicAdd(h + (q0 >> (DSH + 2)), 1u << (8 * ((q0 >> DSH) & 3u)));
        if (8 * bb + 8 < nr) atomicAdd(h + (q1 >> (DSH + 2)), 1u << (8 * ((q1 >> DSH) & 3u)));
      }
    } else if (rr == only_row) {
      if (pb0 >= 0 && pb0 < nr) a.bmax[(size_t)rr * a.ldb + bb] = b0;
      if (pb1 >= 0 && pb1 < nr) a.bmax[(size_t)rr * a.ldb + bb + 1] = b1;
    }
  }
}

// ------------------------------------------------------------------------------------------------ X5 (selection)
// block-wide exclusive scan over the 256 consumer threads (ws: 8 + 1 ints of scratch); returns the exclusive prefix, total
__device__ __forceinline__ int bscan(int v, int& total, int* ws, int tid) {
  const int lane = tid & 31, w = tid >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) ws[w] = x;
  cbar();
  if (w == 0) {
    int t = lane < CTH / 32 ? ws[lane] : 0;
#pragma unroll
    for (int o = 1; o < CTH / 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    if (lane < CTH / 32) ws[8 + lane] = t;   // inclusive warp-total prefix
  }
  cbar();
  total = ws[8 + CTH / 32 - 1];
  const int base = w ? ws[8 + w - 1] : 0;
  cbar();   // ws reusable
  return base + x - v;
}

// v8 (code size: c1mega pays for every instruction of the module on every layer): ONE instance of each selection routine.
// A key source is a runtime descriptor: fp32 values in global memory (g; L2, ukey on load; pin >= 0: that index reads
// max(key, +inf), cand_row's newest block), or (g null) shared keys at byte offset so (padded index pidx(i) or plain).
// v9: the loops branch on the source once, outside (two copies of each body), and load clamped indices unconditionally:
// v8's per-key conditional loads each sat in their own branch region, one memory latency per key.
struct KSrc {
  const float* g;
  int so, pad, pin;
};
constexpr int LB = 4, LBG = 8;   // keys in flight per thread: shared / global sources
__device__ __forceinline__ uint32_t kpin(const KSrc& k, int i, uint32_t u) { return i == k.pin ? max(u, POS_INF_KEY) : u; }
// unordered loops: f(u) for every key, i = tid + CTH j (coalesced in L2, conflict-free in shared memory)
template <typename F>
__device__ __forceinline__ void strided_each(const KSrc& src, int N, F f, int tid) {
  if (src.g != nullptr) {
#pragma unroll 1   // (code size: c1mega charges every layer for it; the LBG / LB loads per trip are the batch)
    for (int i0 = tid; i0 < N; i0 += LBG * CTH) {
      float v[LBG];
#pragma unroll
      for (int k = 0; k < LBG; ++k) v[k] = __ldcg(src.g + min(i0 + k * CTH, N - 1));
#pragma unroll
      for (int k = 0; k < LBG; ++k)
        if (i0 + k * CTH < N) f(kpin(src, i0 + k * CTH, ukey(v[k])));
    }
  } else {
    const uint32_t* s = sptr<const uint32_t>(src.so);
    const int pad = src.pad;
#pragma unroll 1
    for (int i0 = tid; i0 < N; i0 += LB * CTH) {
      uint32_t u[LB];
#pragma unroll
      for (int k = 0; k < LB; ++k) {
        const int i = min(i0 + k * CTH, N - 1);
        u[k] = s[pad ? pidx(i) : i];
      }
#pragma unroll
      for (int k = 0; k < LB; ++k)
        if (i0 + k * CTH < N) f(u[k]);
    }
  }
}

__device__ __forceinline__ void sstamp(unsigned long long* st, int k, int tid) {   // tests: sub-phase timestamps
  if (st && tid == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    st[k] = t;
  }
}

// Radix select over a source of N keys (get(i): shared memory or L2), v5. The keys' min / max first, so the digit windows
// start at the highest bit where they differ (11 bits per pass, 2,048 bins, plain shared atomics). Two barriers per
// pass: the histogram (double-buffered: each pass zeroes the next one's), then warp 0 alone finds the digit (warp scans
// over the bins). After the first pass the items of the chosen digit are compacted (chunks + one scan) into a shared
// list when they fit; later passes run on the list.
// warp 0: the digit of hist (2,048 bins) where the count from the top reaches need -> misc[0] = digit, misc[1] = #items in
// higher digits, misc[2] = #items with that digit
__device__ __forceinline__ void rs_find(const int* hist, int need, int* misc, int lane) {
  const int b0 = RS_BINS - 64 * (lane + 1);   // lane l: bins b0 .. b0 + 63 (lane 0 = the top 64)
  const int4* h4 = reinterpret_cast<const int4*>(hist + b0);
  int s = 0;
#pragma unroll
  for (int k = 0; k < 16; ++k) {   // rotated by lane: the lanes' rows are 256 B apart (same banks) otherwise
    const int4 v = h4[(k + lane) & 15];
    s += v.x + v.y + v.z + v.w;
  }
  int incl = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  const int excl = incl - s;
  const unsigned hit = __ballot_sync(0xffffffffu, excl < need && need <= incl);
  const int ls = __ffs(hit) - 1;   // the lane whose 64 bins hold the crossing
  const int base = __shfl_sync(0xffffffffu, excl, ls);
  const int top = RS_BINS - 1 - 64 * ls;   // its highest bin; lane k: bins top - 2k, top - 2k - 1
  const int c0 = hist[top - 2 * lane], c1 = hist[top - 2 * lane - 1];
  int inc2 = c0 + c1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc2, o);
    if (lane >= o) inc2 += y;
  }
  const int ex2 = base + inc2 - c0 - c1;
  if (ex2 < need && need <= ex2 + c0) misc[0] = top - 2 * lane, misc[1] = ex2, misc[2] = c0;
  else if (ex2 + c0 < need && need <= ex2 + c0 + c1) misc[0] = top - 2 * lane - 1, misc[1] = ex2 + c0, misc[2] = c1;
}

// Radix select (v8): the keys' min / max, then 11-bit digit passes below their highest differing bit (2,048 bins, shared
// atomics; double-buffered histograms; warp 0 finds the digit). After any pass whose chosen digit holds <= RK_MAX items,
// those items are compacted in any order (warp scan + one shared atomic per warp) and ranked exactly (thread t: item t
// against all of them): the exact K-th key, no further passes. A larger digit (<= lcap) is compacted into a shared list
// for the next passes.
constexpr int RK_MAX = 256;
// exact K-th largest key over N keys (1 <= K <= N); *gtp = #keys above it. lowbits > 0: the radix may stop once only the
// bits below lowbits remain (a lower bound with those bits 0); the rank step returns the exact key (a tighter bound).
__device__ __noinline__ uint32_t kth_key(int so, int N, int K, KSrc src, int* gtp, int lowbits, int tid, int lofs, int lcap,
                                         unsigned long long* st) {
  char* scr = sptr<char>(so);
  int* misc = reinterpret_cast<int*>(scr + SC_MISC);
  int* ws = reinterpret_cast<int*>(scr + SC_WS);
  uint32_t* lk = reinterpret_cast<uint32_t*>(scr + SC_LKEY + lofs);   // the pass list (lcap keys)
  uint32_t* rk = reinterpret_cast<uint32_t*>(scr + SC_POS);           // the rank list (free until the caller's emit)
  const int lane = tid & 31, w = tid >> 5;
  int4* h0 = reinterpret_cast<int4*>(scr + SC_HIST);
  h0[tid] = make_int4(0, 0, 0, 0);
  h0[tid + CTH] = make_int4(0, 0, 0, 0);
  if (tid == 0) misc[3] = 0, misc[6] = 0;   // (compaction counters: the rank list, the pass list)
  uint32_t mn = 0xffffffffu, mx = 0u;
  strided_each(src, N, [&](uint32_t u) { mn = min(mn, u), mx = max(mx, u); }, tid);
  mn = __reduce_min_sync(0xffffffffu, mn);
  mx = __reduce_max_sync(0xffffffffu, mx);
  if (lane == 0) ws[16 + w] = mn, ws[24 + w] = mx;
  cbar();
  sstamp(st, 5, tid);
  mn = 0xffffffffu, mx = 0u;
#pragma unroll
  for (int k = 0; k < CTH / 32; ++k) mn = min(mn, (uint32_t)ws[16 + k]), mx = max(mx, (uint32_t)ws[24 + k]);
  if (mn == mx) {   // every key equal
    *gtp = 0;
    return mn;
  }
  int hb = 31 - __clz(mn ^ mx);   // the keys share the bits above hb
  uint32_t pmask = hb >= 31 ? 0u : ~((2u << hb) - 1u), prefix = mn & pmask;
  int need = K, above = 0, buf = 0, nc = N;
  KSrc cur = src;
  bool listed = false;
  while (hb >= lowbits) {
    const int nb = min(11, hb + 1), sh = hb + 1 - nb;
    {   // the histogram of the digit below the prefix; the next buffer zeroed for the next pass
      int* hist = reinterpret_cast<int*>(scr + SC_HIST) + buf * RS_BINS;
      int4* nxt = reinterpret_cast<int4*>(reinterpret_cast<int*>(scr + SC_HIST) + (buf ^ 1) * RS_BINS);
      nxt[tid] = make_int4(0, 0, 0, 0);
      nxt[tid + CTH] = make_int4(0, 0, 0, 0);
      const uint32_t dm = (1u << nb) - 1u, pf = prefix, pm = pmask;
      strided_each(cur, nc, [&](uint32_t u) {
        if ((u & pm) == pf) atomicAdd(&hist[(u >> sh) & dm], 1);
      }, tid);
      cbar();
      sstamp(st, 6, tid);
      if (tid < 32) rs_find(hist, need, misc, tid);
      cbar();
      sstamp(st, 7, tid);
      st = nullptr;   // (tests: the first pass only)
    }
    buf ^= 1;
    const int d = misc[0], ab = misc[1], cnt = misc[2];   // (rewritten only after the next pass's histogram barrier)
    prefix |= uint32_t(d) << sh;
    pmask |= ((1u << nb) - 1u) << sh;
    need -= ab;
    above += ab;
    hb = sh - 1;
    if (hb < lowbits) break;
    const bool small = cnt <= RK_MAX;
    if (!small && (listed || cnt > lcap)) continue;   // another pass over the same source
    {   // the digit's items, any order: strided counts, a warp scan, one shared atomic per warp
      uint32_t* dst = small ? rk : lk;
      int* ctr = small ? misc + 3 : misc + 6;
      const uint32_t pf = prefix, pm = pmask;
      int m = 0;
      strided_each(cur, nc, [&](uint32_t u) { m += (u & pm) == pf; }, tid);
      int incl = m;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      int base = 0;
      if (lane == 31 && incl) base = atomicAdd(ctr, incl);
      base = __shfl_sync(0xffffffffu, base, 31) + incl - m;
      if (m) strided_each(cur, nc, [&](uint32_t u) {
          if ((u & pm) == pf) dst[base++] = u;
        }, tid);
      if (small && tid < 4) rk[cnt + tid] = 0u;   // padding (below every key)
      cbar();
    }
    if (!small) {   // the next passes run on the list
      cur = KSrc{nullptr, so + SC_LKEY + lofs, 0, -1};
      nc = cnt;
      listed = true;
      continue;
    }
    if (tid < cnt) {   // rank: the need-th largest of the digit's cnt keys
      const uint32_t u = rk[tid];
      const uint2* r2 = reinterpret_cast<const uint2*>(rk);   // (SC_POS is 8-B aligned)
      int cg = 0, ce = 0;
      for (int j = 0; j < (cnt + 3) >> 2; ++j) {
        const uint2 v = r2[2 * j], v2 = r2[2 * j + 1];
        cg += (v.x > u) + (v.y > u) + (v2.x > u) + (v2.y > u);
        ce += (v.x >= u) + (v.y >= u) + (v2.x >= u) + (v2.y >= u);
      }
      if (cg < need && need <= ce) misc[4] = (int)u, misc[5] = cg;   // (ties: the same values)
    }
    cbar();
    *gtp = above + misc[5];
    const uint32_t kth = (uint32_t)misc[4];
    cbar();   // misc / rk reusable
    return kth;
  }
  *gtp = above;
  return prefix;
}

// Ordered selection over the N keys of src in index order: item i is selected iff key > kth, or key == kth among the
// first need_eq such ties; out[rank] = val(i), rank = #items selected before i. val(i) = i, or with cpo >= 0 (shared 16-bit
// positions at byte offset cpo, block ids at sbo): 8 sblk[cpos[i] >> 3] + (cpos[i] & 7). Output: gout (global ints) if not
// null, else shared memory at byte offset oo (16-bit if o16, else ints). Warp w takes the 32-aligned range [w ch, w ch +
// ch) (coalesced in L2, conflict-free in shared memory), one ballot per 32 items; one exchange of the warps' (above, tie)
// counts. Returns the number selected; nothing is written if it exceeds cap.
// the warp's range [i0, i1) in steps of 32 x NB keys (clamped indices, every load in flight; f(i, key) checks i < i1)
template <bool G, typename F>
__device__ __forceinline__ void warp_each(const KSrc& src, int i0, int i1, int lane, F f) {
  constexpr int NB = G ? LBG : LB;
  const uint32_t* s = sptr<const uint32_t>(src.so);
  const int pad = src.pad;
#pragma unroll 1
  for (int j = i0; j < i1; j += 32 * NB) {
    uint32_t u[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) {
      const int i = min(j + 32 * k + lane, i1 - 1);
      if (G) u[k] = kpin(src, i, ukey(__ldcg(src.g + i)));
      else u[k] = s[pad ? pidx(i) : i];
    }
#pragma unroll
    for (int k = 0; k < NB; ++k) f(j + 32 * k + lane, u[k]);
  }
}
__device__ __noinline__ int emit_ordered(int so, int N, KSrc src, uint32_t kth, int need_eq, int cpo, int sbo, int* gout,
                                         int oo, int o16, int cap, int tid) {
  int* ws = sptr<int>(so + SC_WS);
  const bool hasc = cpo >= 0;
  const unsigned short* cpos = sptr<const unsigned short>(hasc ? cpo : 0);
  const unsigned short* sblk = sptr<const unsigned short>(sbo);
  int* out32 = sptr<int>(oo);
  unsigned short* out16 = sptr<unsigned short>(oo);
  const int lane = tid & 31, w = tid >> 5;
  const int ch = (((N + CTH / 32 - 1) / (CTH / 32)) + 31) & ~31;
  const int i0 = min(N, w * ch), i1 = min(N, i0 + ch);
  const unsigned lt = (1u << lane) - 1u;
  int g = 0, e = 0;
  {   // counts
    const auto f = [&](int i, uint32_t u) {
      const bool ok = i < i1;
      g += __popc(__ballot_sync(0xffffffffu, ok && u > kth));
      e += __popc(__ballot_sync(0xffffffffu, ok && u == kth));
    };
    if (src.g != nullptr) warp_each<true>(src, i0, i1, lane, f);
    else warp_each<false>(src, i0, i1, lane, f);
  }
  if (lane == 0) ws[w] = g, ws[8 + w] = e;
  cbar();
  int gb = 0, eb = 0, tg = 0, te = 0;
#pragma unroll
  for (int v = 0; v < CTH / 32; ++v) {
    const int a = ws[v], b = ws[8 + v];
    gb += v < w ? a : 0, eb += v < w ? b : 0, tg += a, te += b;
  }
  cbar();   // ws reusable
  const int total = tg + min(te, need_eq);
  if (total > cap) return total;
  {   // writes
    const auto f = [&](int i, uint32_t u) {
      const bool ok = i < i1, isg = ok && u > kth, ise = ok && u == kth;
      const unsigned gm = __ballot_sync(0xffffffffu, isg), em = __ballot_sync(0xffffffffu, ise);
      const int g0 = gb + __popc(gm & lt), e0 = eb + __popc(em & lt);
      int idx = -1;
      if (isg) idx = g0 + min(e0, need_eq);
      else if (ise && e0 < need_eq) idx = g0 + e0;
      if (idx >= 0) {
        int val = i;
        if (hasc) {
          const int v = cpos[i];
          val = (sblk[v >> 3] << 3) | (v & 7);
        }
        if (gout != nullptr) gout[idx] = val;
        else if (o16) out16[idx] = (unsigned short)val;
        else out32[idx] = val;
      }
      gb += __popc(gm), eb += __popc(em);
    };
    if (src.g != nullptr) warp_each<true>(src, i0, i1, lane, f);
    else warp_each<false>(src, i0, i1, lane, f);
  }
  cbar();
  return total;
}

// the slot tables of a direct row (block table entries [0, npg), compact rows: the candidate list [0, ncb)) -> shared
// memory by 4-B cp.async (no registers, issued before the keys' loads, waited with them): the slot conversion after the
// emit then needs no dependent L2 rounds (cb, then bt)
constexpr int NBT = 928, NCB = CAND_K;   // (the CPOS region: NCB + NBT ints = 11,904 B)
static_assert((NBT + NCB) * 4 <= CMAX * 2 && SC_CPOS % 4 == 0 && SC_POS % 8 == 0, "slot tables / rank list");
struct SlotTabs {
  int* sbt;          // [NBT] shared
  int* scb;          // [NCB] shared
  const int* bt;
  const int* cb;
  int npg, ncb;      // 0: nothing to stage
};
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void tabs_issue(const SlotTabs& t, int tid) {   // (then cp.async.wait_all + a barrier)
  for (int i = tid; i < t.npg; i += CTH) cp_async4(t.sbt + i, t.bt + i);
  for (int i = tid; i < t.ncb; i += CTH) cp_async4(t.scb + i, t.cb + i);
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
// keys of src[0 .. n) -> dst (shared memory), 8 loads in flight per thread
__device__ __forceinline__ void stage_keys(uint32_t* dst, const float* src, int n, int tid) {
#pragma unroll 1
  for (int i0 = 0; i0 < n; i0 += 8 * CTH) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = i0 + k * CTH + tid;
      v[k] = i < n ? __ldcg(src + i) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = i0 + k * CTH + tid;
      if (i < n) dst[pidx(i)] = ukey(v[k]);
    }
  }
}

// X5: the exact top-512 of row r -> FlashMLA's global slots / length. scr: SC_BYTES of shared memory. valid: row is real.
// cb (compact rows of the consumer layers): index t -> position 8 cb[t >> 3] + (t & 7), n_c the compact length.
// n <= CMAX: every element a candidate (keys staged in shared memory). Else: tau = the 512-th largest block key (block
// keys staged when they fit), the blocks >= tau (ascending), their elements >= tau in position order (<= CMAX; positions
// as 16-bit (block-list index, offset)), the exact select on them; more candidates: the select streams over the whole row
// from L2 (counted in a.slow). bt: the block table (null: a.bt).
__device__ __noinline__ void select_row(const IdxArgs& a, int r, bool valid, int so, int tid, const int* cb, int n_c,
                                        unsigned long long* st, const int* btp) {
  char* scr = sptr<char>(so);
  const int* bt = btp != nullptr ? btp : a.bt;
  sstamp(st, 0, tid);
  const int n = !valid ? 0 : cb ? n_c : __ldcg(a.lens + r);
  int* pos = reinterpret_cast<int*>(scr + SC_POS);
  int* ws = reinterpret_cast<int*>(scr + SC_WS);
  const float* lrow = a.logits + (size_t)r * a.ldl;
  const float* brow = a.bmax + (size_t)r * a.ldb;
  uint32_t* ckey = reinterpret_cast<uint32_t*>(scr + SC_CKEY);
  unsigned short* cpos = reinterpret_cast<unsigned short*>(scr + SC_CPOS);
  unsigned short* sblk = reinterpret_cast<unsigned short*>(scr + SC_LKEY);   // (block ids < 65,536; the candidate select's
                                                                             // pass list then uses the second half)
  // direct rows (n <= CMAX; the CPOS region unused): the slot tables staged with the keys
  const int nfull = !valid ? 0 : cb ? __ldcg(a.lens + r) : n;   // positions < nfull
  const int npg = (nfull + a.bs - 1) / a.bs;
  const bool stg = n <= CMAX && npg <= NBT;
  const SlotTabs tabs{reinterpret_cast<int*>(scr + SC_CPOS) + NCB, reinterpret_cast<int*>(scr + SC_CPOS), bt, cb,
                      stg ? npg : 0, stg && cb ? (n + BLK - 1) / BLK : 0};
  if (stg) tabs_issue(tabs, tid);
  const KSrc staged{nullptr, so + SC_CKEY, 1, -1};
  int gt;
  if (n <= TOPK) {
    for (int i = tid; i < TOPK; i += CTH) pos[i] = i < n ? i : -1;
    cp_wait_all();
    cbar();
  } else if (n <= CMAX) {
    stage_keys(ckey, lrow, n, tid);
    cp_wait_all();
    cbar();
    sstamp(st, 1, tid);
    const uint32_t kth = kth_key(so, n, TOPK, staged, &gt, 0, tid, 0, LCAP, st ? st - 8 : nullptr);
    sstamp(st, 2, tid);
    emit_ordered(so, n, staged, kth, TOPK - gt, -1, 0, nullptr, so + SC_POS, 0, TOPK, tid);
    sstamp(st, 3, tid);
  } else {
    const int nb = (n + BLK - 1) / BLK;
    const bool bsm = nb <= CMAX;
    if (bsm) {
      stage_keys(ckey, brow, nb, tid);
      cbar();
    }
    const KSrc bsrc = bsm ? staged : KSrc{brow, 0, 0, -1};
    const uint32_t tau = kth_key(so, nb, TOPK, bsrc, &gt, 8, tid, 0, LCAP, nullptr);
    sstamp(st, 1, tid);
    // the selected blocks (key >= tau), ascending -> sblk (<= SBMAX)
    const int S = emit_ordered(so, nb, bsrc, tau ? tau - 1u : 0u, tau ? 0 : 0x7fffffff, -1, 0, nullptr, so + SC_LKEY, 1,
                               SBMAX, tid);
    bool overflow = S > SBMAX;
    int C = 0;
    if (!overflow) {   // the candidates overwrite the staged block keys (no longer needed)
      for (int r0 = 0; r0 < S; r0 += CTH) {
        const int i = r0 + tid;
        uint32_t u[8];
        int b = 0, m = 0, cc = 0;
        if (i < S) {
          b = sblk[i];
          m = min(BLK, n - BLK * b);
          const float4* p = reinterpret_cast<const float4*>(lrow + (size_t)BLK * b);
          const float4 x0 = __ldcg(p), x1 = __ldcg(p + 1);
          const float v[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            u[j] = ukey(v[j]);
            cc += (j < m) && u[j] >= tau;
          }
        }
        int tot;
        int o2 = bscan(cc, tot, ws, tid);
        if (C + tot > CMAX) {   // (uniform)
          overflow = true;
          break;
        }
        if (i < S) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if ((j < m) && u[j] >= tau) ckey[pidx(C + o2)] = u[j], cpos[C + o2] = (unsigned short)((i << 3) | j), ++o2;
        }
        C += tot;
      }
    }
    cbar();
    sstamp(st, 2, tid);
    if (!overflow) {
      const uint32_t kth = kth_key(so, C, TOPK, staged, &gt, 0, tid, LCAP * 2, LCAP / 2, nullptr);
      sstamp(st, 3, tid);
      emit_ordered(so, C, staged, kth, TOPK - gt, so + SC_CPOS, so + SC_LKEY, nullptr, so + SC_POS, 0, TOPK, tid);
    } else {
      if (tid == 0 && a.slow) atomicAdd(a.slow, 1);
      const KSrc row{lrow, 0, 0, -1};
      const uint32_t kth = kth_key(so, n, TOPK, row, &gt, 0, tid, 0, LCAP, nullptr);
      emit_ordered(so, n, row, kth, TOPK - gt, -1, 0, nullptr, so + SC_POS, 0, TOPK, tid);
    }
  }
  // global slots + length
  sstamp(st, 4, tid);
#pragma unroll
  for (int k = 0; k < TOPK / CTH; ++k) {
    int p = pos[tid + CTH * k];
    if (cb && p >= 0) p = ((stg ? tabs.scb[p >> 3] : __ldcg(cb + (p >> 3))) << 3) | (p & 7);
    const int pg = p >= 0 ? (stg ? tabs.sbt[p / a.bs] : __ldg(bt + p / a.bs)) : 0;
    a.gout[(size_t)r * a.ldg + tid + CTH * k] = p >= 0 ? pg * a.bs + p % a.bs : -1;
  }
  if (tid == 0) a.glen[r] = valid ? min(n, TOPK) : 0;   // (every row with n > 512 selects 512; n <= 512: 0 .. n - 1)
  sstamp(st, 5, tid);
}

// X5 (layer 20): the candidate blocks of row r (= idx2f select_blocks(bmax, lens, 2048, out, from_lens=1)): the newest
// block pinned to +inf, the top-2048 block keys (lowest id on ties), -inf blocks dropped, ascending, -1 padded
__device__ __noinline__ void cand_row(const IdxArgs& a, int r, bool valid, int so, int tid) {
  char* scr = sptr<char>(so);
  const int n = valid ? __ldcg(a.lens + r) : 0;
  const int nb = (n + BLK - 1) / BLK;
  int* out = a.cand + (size_t)r * CAND_K;
  int written = 0;
  if (nb > 0) {
    const float* brow = a.bmax + (size_t)r * a.ldb;
    uint32_t* ck = reinterpret_cast<uint32_t*>(scr + SC_CKEY);
    const bool sm = nb <= CMAX;   // block keys staged in shared memory, else from L2 at every pass
    if (sm) {
      stage_keys(ck, brow, nb, tid);
      cbar();
      if (tid == 0) ck[pidx(nb - 1)] = max(ck[pidx(nb - 1)], POS_INF_KEY);   // (stock: scores[(end - 1) // 8] = inf)
      cbar();
    }
    const KSrc key = sm ? KSrc{nullptr, so + SC_CKEY, 1, -1} : KSrc{brow, 0, 0, nb - 1};
    uint32_t kth = NEG_INF_KEY;   // (no cut: every block above -inf)
    int need_eq = 0;
    if (nb > CAND_K) {
      int gt;
      kth = kth_key(so, nb, CAND_K, key, &gt, 0, tid, 0, LCAP, nullptr);
      need_eq = kth == NEG_INF_KEY ? 0 : CAND_K - gt;   // (-inf blocks are never selected)
    }
    written = emit_ordered(so, nb, key, kth, need_eq, -1, 0, out, 0, 0, CAND_K, tid);
  }
  for (int i = written + tid; i < CAND_K; i += CTH) out[i] = -1;
  if (tid == 0 && a.cand_n) a.cand_n[r] = written;
}

// ================================================================================================ v14: distributed select
// The exact top-512 of a row (and layer 20's top-2048 blocks) with the whole grid. v13's one CTA per row
// took 14-23 us in situ at 8-20K and 30-54 us standalone at 64-128K (passes over the row from L2, barrier-bound) while
// 128 CTAs idled at IX_SEL.
//  stage 0 (the logits CTAs, fused into logits_tile): per-row histograms of the 11-bit digits (the top bits of the order-
//          preserving key) of every stored key and (layer 20) every block maximum (the newest block pinned to +inf), in
//          shared memory, added to global ones before IX_LG (integer adds: exact, order-free);
//  stage 1 (every CTA, after IX_LG): the digit d1 holding the K-th largest (K = 512 / 2048) from the global histogram; a
//          strided 1/132 of the row's keys from L2; those with digit >= d1 appended (key, position) to the row's global
//          list (one atomic per warp, row and batch); then IX_CL;
//  stage 2 (row CTA r / candidate CTA 4 + r, after IX_CL): the list staged (more than DCAP entries: v13's single-CTA
//          routine), the exact K-th key (kth_key), the keys above it + the lowest ids among its ties, ascending (rank by
//          count / a bitmap), then the slots as select_row. The CTA then zeroes its row's histogram and counter (every
//          reader is past IX_CL; the next writers come after IX_SEL).
// Exactness: the list holds every key with digit >= d1, a superset of the top K, so the result is the canonical top-K (key
// desc, lowest position on ties): the same set as select_row / cand_row / idx2f.
__device__ __forceinline__ void red_add(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void ds_zero(int o, int words, int tid) {   // shared words [o, o + 4 words), 16-B aligned
  uint4* p = sptr<uint4>(o);
  for (int j = tid; j < words / 4; j += CTH) p[j] = make_uint4(0u, 0u, 0u, 0u);
}
// stage 0 -> global: the CTA's shared histograms (element digits u16-packed at hso, block-max digits u8-packed at hbo) added
// to the rows of smask / bmask (other rows' bins are dropped: those histograms are neither read nor zeroed)
__device__ __forceinline__ void ds_flush(int hso, int hbo, unsigned* dsc, unsigned smask, unsigned bmask, int tid) {
  if (smask) {
    const unsigned* h = sptr<const unsigned>(hso);
#pragma unroll 4
    for (int j = tid; j < NRM * DBINS / 2; j += CTH) {
      const unsigned v = h[j];
      if (v != 0u && ((smask >> (j / (DBINS / 2))) & 1u)) {
        unsigned* gp = dsc + DS_H + 2 * j;   // (row j / 1024's bins 2 (j % 1024), + 1)
        if (v & 0xffffu) red_add(gp, v & 0xffffu);
        if (v >> 16) red_add(gp + 1, v >> 16);
      }
    }
  }
  if (bmask) {
    const unsigned* h = sptr<const unsigned>(hbo);
#pragma unroll 2
    for (int j = tid; j < NRM * DBINS / 4; j += CTH) {
      const unsigned v = h[j];
      if (v != 0u && ((bmask >> (j / (DBINS / 4))) & 1u)) {
        unsigned* gp = dsc + DS_HB + 4 * j;
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if ((v >> (8 * k)) & 255u) red_add(gp + k, (v >> (8 * k)) & 255u);
      }
    }
  }
  {   // the coarse rows: thread t = row t / 64, coarse bin t % 64 (32 digits: 16 u16-pair / 8 u8-quad words, rotated by t)
    const int r = tid >> 6, cbn = tid & 63;
    if ((smask >> r) & 1u) {
      const unsigned* h = sptr<const unsigned>(hso) + r * (DBINS / 2) + cbn * 16;
      unsigned sum = 0u;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const unsigned v = h[(k + tid) & 15];
        sum += (v & 0xffffu) + (v >> 16);
      }
      if (sum) red_add(dsc + DS_CO + r * DCB + cbn, sum);
    }
    if ((bmask >> r) & 1u) {
      const unsigned* h = sptr<const unsigned>(hbo) + r * (DBINS / 4) + cbn * 8;
      unsigned sum = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const unsigned v = h[(k + tid) & 7];
        sum += (v & 255u) + ((v >> 8) & 255u) + ((v >> 16) & 255u) + (v >> 24);
      }
      if (sum) red_add(dsc + DS_CO + (NRM + r) * DCB + cbn, sum);
    }
  }
}
// tests (logits / block maxima given, no X4): stage 0 from the rows in global memory, a strided 1/132 per CTA
__device__ __noinline__ void ds_hist_given(const IdxArgs& a, int hso, int hbo, const int (&ne)[NRM], const int (&nbk)[NRM],
                                           unsigned smask, unsigned bmask, int c, int tid) {
  unsigned* h = sptr<unsigned>(hso);
  unsigned* hb = sptr<unsigned>(hbo);
  for (int r = 0; r < NRM; ++r) {
    if ((smask >> r) & 1u)
      for (int i = c * CTH + tid; i < ne[r]; i += NCTA_IX * CTH) {
        const uint32_t k = ukey(__ldcg(a.logits + (size_t)r * a.ldl + i));
        atomicAdd(h + r * (DBINS / 2) + (k >> (DSH + 1)), ((k >> DSH) & 1u) ? 0x10000u : 1u);
      }
    if ((bmask >> r) & 1u)
      for (int b = c * CTH + tid; b < nbk[r]; b += NCTA_IX * CTH) {
        uint32_t k = ukey(__ldcg(a.bmax + (size_t)r * a.ldb + b));
        if (b == nbk[r] - 1) k = max(k, POS_INF_KEY);
        atomicAdd(hb + r * (DBINS / 4) + (k >> (DSH + 2)), 1u << (8 * ((k >> DSH) & 3u)));
      }
  }
}
// one warp: from a global histogram row (fine: 2,048 digits, coarse: 64 bins of 32) the digit where the count from the top
// reaches need, the number above it and the number in it (not found: cn = 0x7fffffff, the callers fall back). v14d: two
// dependent rounds of 2 + 1 words per lane (v14c read the whole 8 KB row per warp: with all 132 CTAs reading the same
// rows that took 4-9 us in stage 1)
struct GF {
  int d, ab, cn;
};
#ifdef C1IDX_V14_PICKT
#define C1IDX_GFQ __forceinline__
#else
#define C1IDX_GFQ __noinline__   // (v14h: one copy)
#endif
__device__ C1IDX_GFQ GF gfind(const unsigned* fine, const unsigned* coarse, int need, int lane) {
  const int c_hi = (int)__ldcg(coarse + DCB - 1 - 2 * lane), c_lo = (int)__ldcg(coarse + DCB - 2 - 2 * lane);
  const int s = c_hi + c_lo;
  int incl = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  const int excl = incl - s;
  const unsigned hit = __ballot_sync(0xffffffffu, excl < need && need <= incl);
  if (hit == 0u) return GF{0, 0, 0x7fffffff};
  const int ls = __ffs(hit) - 1;
  const bool top = excl + c_hi >= need;
  const int cb = __shfl_sync(0xffffffffu, top ? DCB - 1 - 2 * lane : DCB - 2 - 2 * lane, ls);
  const int above = __shfl_sync(0xffffffffu, top ? excl : excl + c_hi, ls);
  const int f = (int)__ldcg(fine + cb * 32 + 31 - lane);   // lane 0 = the coarse bin's top digit
  int inc2 = f;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc2, o);
    if (lane >= o) inc2 += y;
  }
  const int ex2 = above + inc2 - f;
  const unsigned h2 = __ballot_sync(0xffffffffu, ex2 < need && need <= ex2 + f);
  if (h2 == 0u) return GF{0, 0, 0x7fffffff};
  const int l2 = __ffs(h2) - 1;
  return GF{cb * 32 + 31 - l2, __shfl_sync(0xffffffffu, ex2, l2), __shfl_sync(0xffffffffu, f, l2)};
}
__device__ __forceinline__ const unsigned* ds_co(const unsigned* dsc, int fine_off) {   // the coarse row of a fine row
  return dsc + DS_CO + ((fine_off - DS_H) / DBINS) * DCB;
}
// predicated L2 load (no load at all when !pr: no clamped duplicates hammering one line)
__device__ __forceinline__ unsigned ldcg_if(const void* p, bool pr) {
  unsigned v = 0u;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q ld.global.cg.u32 %0, [%1];\n}" : "+r"(v) : "l"(p), "r"((int)pr));
  return v;
}
// stage 1 / 2 shared words after the selection scratch (SC_DS, ints): d1 / #above d1 of the 8 rows (select 0..3, candidate
// 4..7), the warps' totals / bases [8][8], stage 2's words; the level-2 histograms (stage 1, shared, u16-packed [8][1024])
constexpr int SC_DS = SC_BYTES, SD_D1 = 0, SD_AB1 = 8, SD_WT = 16, SD_S2 = 80, SD_N = 96, GMAX = CTH;
constexpr int S1K = 8;   // stage 1: at most 8 chunks of 256 keys per CTA and row (rows up to 270,336 keys: >= LMAX)
static_assert(SC_CKEY + 2 * GMAX * 4 <= SC_CPOS && (3 * CTH + TOPK) * 4 <= SC_WS, "v14d scratch");
#ifdef C1IDX_V14_S1T
// v14c stage 1 (every CTA, after IX_LG): select rows of smask (ne[r] keys: logits positions < n, or compact indices < n_c)
// and candidate rows of bmask (nbk[r] block maxima, the newest pinned). The active rows' chunks of this CTA (c, c + 132, ..;
// KC per row: the code runs only the chunks that exist, v14b ran all 64 slots predicated: latency-bound at 8 warps per SM)
// in flight at once; the digit d1 of each row (global histogram); keys with digit >= d1 -> the row's global list (ONE
// atomic per CTA and row); keys with digit == d1 -> the row's level-2 histogram (the next 11 bits; global reds, a few
// keys per CTA), so stage 2 knows the K-th key's top 22 bits.
template <int KC>
__device__ __forceinline__ void ds_s1(const IdxArgs& a, int* sd, const int (&nq)[2 * NRM], unsigned qm, int c, int tid,
                                      unsigned long long* st) {
  const int lane = tid & 31, w = tid >> 5;
  uint32_t u[2 * NRM][KC];
#pragma unroll
  for (int q = 0; q < 2 * NRM; ++q) {
    if (!((qm >> q) & 1u)) continue;
    const float* src = q < NRM ? a.logits + (size_t)q * a.ldl : a.bmax + (size_t)(q - NRM) * a.ldb;
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      const int i = (c + NCTA_IX * k) * CTH + tid;
      u[q][k] = ldcg_if(src + (i < nq[q] ? i : 0), i < nq[q]);
    }
  }
  if ((qm >> w) & 1u) {   // warp w: row w's digit (w < 4: select, K = 512; else candidate, K = 2,048)
    const int fo = w < NRM ? DS_H + w * DBINS : DS_HB + (w - NRM) * DBINS;
    const GF f = gfind(a.dsc + fo, ds_co(a.dsc, fo), w < NRM ? TOPK : CAND_K, lane);
    if (lane == 0) sd[SD_D1 + w] = f.d, sd[SD_AB1 + w] = f.ab;
  }
  cbar();
  sstamp(st, 0, tid);
  unsigned okm[2 * NRM];
  int mq[2 * NRM], iq[2 * NRM];
#pragma unroll
  for (int q = 0; q < 2 * NRM; ++q) {
    okm[q] = 0u, mq[q] = 0, iq[q] = 0;
    if (!((qm >> q) & 1u)) continue;
    const int N = nq[q];
    const uint32_t d1 = (uint32_t)sd[SD_D1 + q];
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      const int i = (c + NCTA_IX * k) * CTH + tid;
      uint32_t x = ukey(__uint_as_float(u[q][k]));
      if (q >= NRM && i == N - 1) x = max(x, POS_INF_KEY);
      u[q][k] = x;
      if (i < N && (x >> DSH) >= d1) {
        okm[q] |= 1u << k;
        if ((x >> DSH) == d1) {   // level 2: fine + coarse
          const uint32_t b2 = (x >> (DSH - DBITS)) & (DBINS - 1);
          red_add(a.dsc + DS_H2 + q * DBINS + b2, 1u);
          red_add(a.dsc + DS_CO + (2 * NRM + q) * DCB + (b2 >> 5), 1u);
        }
      }
    }
    mq[q] = __popc(okm[q]);
    int x = mq[q];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    iq[q] = x;
    if (lane == 31) sd[SD_WT + q * 8 + w] = x;
  }
  cbar();
  sstamp(st, 1, tid);
  if (tid < 2 * NRM && ((qm >> tid) & 1u)) {   // one atomic per CTA and row: the warps' bases
    int run = 0;
#pragma unroll
    for (int v = 0; v < CTH / 32; ++v) {
      const int t = sd[SD_WT + tid * 8 + v];
      sd[SD_WT + tid * 8 + v] = run;
      run += t;
    }
    const int base = run ? (int)atomicAdd(a.dsc + DS_C + tid, (unsigned)run) : 0;
#pragma unroll
    for (int v = 0; v < CTH / 32; ++v) sd[SD_WT + tid * 8 + v] += base;
  }
  cbar();
  sstamp(st, 2, tid);
#pragma unroll
  for (int q = 0; q < 2 * NRM; ++q) {
    if (!okm[q]) continue;
    unsigned* lk = a.dsc + (q < NRM ? DS_LK + q * DCAP : DS_BK + (q - NRM) * DCAP);
    int* lp = reinterpret_cast<int*>(a.dsc + (q < NRM ? DS_LP + q * DCAP : DS_BP + (q - NRM) * DCAP));
    int b = sd[SD_WT + q * 8 + w] + iq[q] - mq[q];
#pragma unroll
    for (int k = 0; k < KC; ++k)
      if ((okm[q] >> k) & 1u) {
        if (b < DCAP) lk[b] = u[q][k], lp[b] = (c + NCTA_IX * k) * CTH + tid;
        ++b;
      }
  }
  sstamp(st, 3, tid);
}
__device__ __noinline__ void ds_stage1(const IdxArgs& a, int so, const int (&ne)[NRM], const int (&nbk)[NRM], unsigned smask,
                                       unsigned bmask, int c, int tid, unsigned long long* st = nullptr) {
  int* sd = sptr<int>(so + SC_DS);
  const unsigned qm = smask | (bmask << NRM);
  int nq[2 * NRM];
  int nmx = 0;
#pragma unroll
  for (int q = 0; q < 2 * NRM; ++q) {
    nq[q] = ((qm >> q) & 1u) ? (q < NRM ? ne[q] : nbk[q - NRM]) : 0;
    nmx = max(nmx, nq[q]);
  }
  const int kc = (nmx + NCTA_IX * CTH - 1) / (NCTA_IX * CTH);   // chunks per row per CTA (uniform)
  if (kc <= 1) ds_s1<1>(a, sd, nq, qm, c, tid, st);
  else if (kc <= 2) ds_s1<2>(a, sd, nq, qm, c, tid, st);
  else if (kc <= 4) ds_s1<4>(a, sd, nq, qm, c, tid, st);
  else ds_s1<8>(a, sd, nq, qm, c, tid, st);
}
#else
// v14c stage 1 (every CTA, after IX_LG): select rows of smask (ne[r] keys: logits positions < n, or compact indices < n_c)
// and candidate rows of bmask (nbk[r] block maxima, the newest pinned). The active rows' chunks of this CTA (c, c + 132, ..;
// KC per row: the code runs only the chunks that exist, v14b ran all 64 slots predicated: latency-bound at 8 warps per SM)
// in flight at once; the digit d1 of each row (global histogram); keys with digit >= d1 -> the row's global list (ONE
// atomic per CTA and row); keys with digit == d1 -> the row's level-2 histogram (the next 11 bits; global reds, a few
// keys per CTA), so stage 2 knows the K-th key's top 22 bits.
// (v14g) ONE body, a runtime loop over batches of S1B chunks per row (v14c-f: 4 template copies, 10.3K SASS of cold code
// in c1mega's module; rows up to 132 x 256 x S1B keys take one batch, LMAX two)
constexpr int S1B = 4;
__device__ __noinline__ void ds_stage1(const IdxArgs& a, int so, const int (&ne)[NRM], const int (&nbk)[NRM], unsigned smask,
                                       unsigned bmask, int c, int tid, unsigned long long* st = nullptr) {
  int* sd = sptr<int>(so + SC_DS);
  const int lane = tid & 31, w = tid >> 5;
  const unsigned qm = smask | (bmask << NRM);
  int nq[2 * NRM];
  int nmx = 0;
#pragma unroll
  for (int q = 0; q < 2 * NRM; ++q) {
    nq[q] = ((qm >> q) & 1u) ? (q < NRM ? ne[q] : nbk[q - NRM]) : 0;
    nmx = max(nmx, nq[q]);
  }
  uint32_t u[2 * NRM][S1B];
  auto loads = [&](int k0) {   // this CTA's chunks c + 132 (k0 + k) of every distributed row, all in flight
#pragma unroll
    for (int q = 0; q < 2 * NRM; ++q) {
      if (!((qm >> q) & 1u)) continue;
      const float* src = q < NRM ? a.logits + (size_t)q * a.ldl : a.bmax + (size_t)(q - NRM) * a.ldb;
#pragma unroll
      for (int k = 0; k < S1B; ++k) {
        const int i = (c + NCTA_IX * (k0 + k)) * CTH + tid;
        u[q][k] = ldcg_if(src + (i < nq[q] ? i : 0), i < nq[q]);
      }
    }
  };
  loads(0);
  if ((qm >> w) & 1u) {   // warp w: row w's digit (w < 4: select, K = 512; else candidate, K = 2,048)
    const int fo = w < NRM ? DS_H + w * DBINS : DS_HB + (w - NRM) * DBINS;
    const GF f = gfind(a.dsc + fo, ds_co(a.dsc, fo), w < NRM ? TOPK : CAND_K, lane);
    if (lane == 0) sd[SD_D1 + w] = f.d, sd[SD_AB1 + w] = f.ab;
  }
  cbar();
  sstamp(st, 0, tid);
#pragma unroll 1
  for (int k0 = 0; k0 * NCTA_IX * CTH < nmx; k0 += S1B) {
    if (k0) loads(k0);
    unsigned okm[2 * NRM];
    int mq[2 * NRM], iq[2 * NRM];
#pragma unroll
    for (int q = 0; q < 2 * NRM; ++q) {
      okm[q] = 0u, mq[q] = 0, iq[q] = 0;
      if (!((qm >> q) & 1u)) continue;
      const int N = nq[q];
      const uint32_t d1 = (uint32_t)sd[SD_D1 + q];
#pragma unroll
      for (int k = 0; k < S1B; ++k) {
        const int i = (c + NCTA_IX * (k0 + k)) * CTH + tid;
        uint32_t x = ukey(__uint_as_float(u[q][k]));
        if (q >= NRM && i == N - 1) x = max(x, POS_INF_KEY);
        u[q][k] = x;
        if (i < N && (x >> DSH) >= d1) {
          okm[q] |= 1u << k;
          if ((x >> DSH) == d1) {   // level 2: fine + coarse
            const uint32_t b2 = (x >> (DSH - DBITS)) & (DBINS - 1);
            red_add(a.dsc + DS_H2 + q * DBINS + b2, 1u);
            red_add(a.dsc + DS_CO + (2 * NRM + q) * DCB + (b2 >> 5), 1u);
          }
        }
      }
      mq[q] = __popc(okm[q]);
      int x = mq[q];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      iq[q] = x;
      if (lane == 31) sd[SD_WT + q * 8 + w] = x;
    }
    cbar();
    sstamp(st, 1, tid);
    if (tid < 2 * NRM && ((qm >> tid) & 1u)) {   // one atomic per CTA, row and batch: the warps' bases
      int run = 0;
#pragma unroll
      for (int v = 0; v < CTH / 32; ++v) {
        const int t = sd[SD_WT + tid * 8 + v];
        sd[SD_WT + tid * 8 + v] = run;
        run += t;
      }
      const int base = run ? (int)atomicAdd(a.dsc + DS_C + tid, (unsigned)run) : 0;
#pragma unroll
      for (int v = 0; v < CTH / 32; ++v) sd[SD_WT + tid * 8 + v] += base;
    }
    cbar();
    sstamp(st, 2, tid);
#pragma unroll
    for (int q = 0; q < 2 * NRM; ++q) {
      if (!okm[q]) continue;
      unsigned* lk = a.dsc + (q < NRM ? DS_LK + q * DCAP : DS_BK + (q - NRM) * DCAP);
      int* lp = reinterpret_cast<int*>(a.dsc + (q < NRM ? DS_LP + q * DCAP : DS_BP + (q - NRM) * DCAP));
      int b = sd[SD_WT + q * 8 + w] + iq[q] - mq[q];
#pragma unroll
      for (int k = 0; k < S1B; ++k)
        if ((okm[q] >> k) & 1u) {
          if (b < DCAP) lk[b] = u[q][k], lp[b] = (c + NCTA_IX * (k0 + k)) * CTH + tid;
          ++b;
        }
    }
    cbar();   // (the bases are read before the next batch rewrites SD_WT)
  }
  sstamp(st, 3, tid);
}
#endif
#ifdef C1IDX_V14_PICKT
// the K-th key's top 22 bits from the
// two histograms (P22; A keys above it, G keys with it); the list's keys above P22 are selected, the boundary group (<= 256
// keys) ranked exactly by (key desc, id asc) for the K - A still needed; f(id) per selected id. drop_ninf: -inf keys are
// never selected (cand_row's rule). Returns false (v13's routine instead) on a list overflow or a boundary group > 256.
template <typename F>
__device__ __forceinline__ bool ds_pick2(const IdxArgs& a, int so, int q, int K, bool drop_ninf, F f, int tid,
                                        unsigned long long* st = nullptr) {
  int* sd = sptr<int>(so + SC_DS);
  uint32_t* bk = sptr<uint32_t>(so + SC_CKEY);
  int* bp = sptr<int>(so + SC_CKEY + GMAX * 4);
  const int lane = tid & 31, w = tid >> 5;
  const int ab1 = sd[SD_AB1 + q];
  if (w == 0) {
    const GF f2 = gfind(a.dsc + DS_H2 + q * DBINS, ds_co(a.dsc, DS_H2 + q * DBINS), K - ab1, lane);
    if (lane == 0) sd[SD_S2] = f2.d, sd[SD_S2 + 1] = f2.ab, sd[SD_S2 + 2] = f2.cn;
  }
  if (tid == 32) sd[SD_S2 + 3] = (int)__ldcg(a.dsc + DS_C + q), sd[SD_S2 + 4] = 0, sd[SD_S2 + 5] = 0, sd[SD_S2 + 6] = 0;
  cbar();
  sstamp(st, 1, tid);
  const int C = sd[SD_S2 + 3], A = ab1 + sd[SD_S2 + 1], G = sd[SD_S2 + 2];
  if (tid == 0) {   // (tests: list / group of the last call and per layer)
    const unsigned dv = (unsigned)min(C, 0xffff) | ((unsigned)min(G, 0xffff) << 16);
    a.dsc[DS_C + 8 + q] = dv;
    if (a.L >= 0 && a.L < 40) a.dsc[DS_DBG + (a.L >> 2) * 8 + q] = dv;
  }
  if (C > DCAP || C < K || G > GMAX) return false;
  const uint32_t P22 = ((uint32_t)sd[SD_D1 + q] << DBITS) | (uint32_t)sd[SD_S2];
  const unsigned* lk = a.dsc + (q < NRM ? DS_LK + q * DCAP : DS_BK + (q - NRM) * DCAP);
  const int* lp = reinterpret_cast<const int*>(a.dsc + (q < NRM ? DS_LP + q * DCAP : DS_BP + (q - NRM) * DCAP));
#pragma unroll 1
  for (int i0 = 0; i0 < C; i0 += 8 * CTH) {   // keys + ids, 16 loads in flight per thread
    uint32_t kk[8];
    int pp[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = i0 + k * CTH + tid;
      kk[k] = ldcg_if(lk + (i < C ? i : 0), i < C);
      pp[k] = (int)ldcg_if(lp + (i < C ? i : 0), i < C);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (i0 + k * CTH + tid < C) {
        const uint32_t t22 = kk[k] >> (32 - 2 * DBITS);
        if (t22 > P22) {
          f(pp[k]);
        } else if (t22 == P22) {
          const int s = atomicAdd(sd + SD_S2 + 4, 1);
          if (s < GMAX) bk[s] = kk[k], bp[s] = pp[k];
        }
      }
  }
  cbar();
  sstamp(st, 2, tid);
  if (sd[SD_S2 + 4] != G) {   // (cannot happen: the list holds the whole group; a safety net, flagged)
    if (tid == 0) sd[SD_S2 + 6] = 1;
    cbar();
    return false;
  }
  if (tid < G) {   // the boundary group: rank by (key desc, id asc); the first K - A selected
    const uint32_t x = bk[tid];
    const int p = bp[tid];
    int rk = 0;
    for (int j = 0; j < G; ++j) {
      const uint32_t y = bk[j];
      rk += (y > x) || (y == x && bp[j] < p);
    }
    if (rk < K - A && !(drop_ninf && x == NEG_INF_KEY)) f(p);
  }
  cbar();
  return true;
}
// v14b stage 2, select row r (row CTA r, after IX_CL): -> ti row r / tl[r] as select_row
__device__ __noinline__ void ds_select_row(const IdxArgs& a, int r, bool valid, int so, int tid, const int* cb, int n_c,
                                           const int* btp, unsigned long long* st = nullptr) {
  sstamp(st, 0, tid);
  char* scr = sptr<char>(so);
  const int* bt = btp != nullptr ? btp : a.bt;
  int* sd = sptr<int>(so + SC_DS);
  const int n = cb ? n_c : __ldcg(a.lens + r);
  const int nfull = cb ? __ldcg(a.lens + r) : n;
  const int npg = (nfull + a.bs - 1) / a.bs;
  const bool stg = npg <= NBT;
  const SlotTabs tabs{reinterpret_cast<int*>(scr + SC_CPOS) + NCB, reinterpret_cast<int*>(scr + SC_CPOS), bt, cb,
                      stg ? npg : 0, cb ? (n + BLK - 1) / BLK : 0};
  tabs_issue(tabs, tid);
  int* sp = reinterpret_cast<int*>(scr + SC_LKEY);   // [512] the selected positions (any order)
  int* pos = reinterpret_cast<int*>(scr + SC_POS);
  // (v14d) ascending by a counting sort on 256 buckets of 2^sh positions (one per thread): counts during the pick, a scan,
  // a scatter, then each item's rank inside its bucket (v14c's bitmap scan took 1-5 us at 80-131K)
  int* bc = reinterpret_cast<int*>(scr + SC_HIST);   // [256] bucket counts, [256] starts, [256] cursors, [512] scattered
  int* bs0 = bc + CTH;
  int* bcur = bs0 + CTH;
  int* tmp = bcur + CTH;
  const int sh = max(0, 32 - __clz(max(n - 1, 1)) - 8);
  bc[tid] = 0, bcur[tid] = 0;   // (ds_pick2's first barrier orders it)
  const bool pk = ds_pick2(a, so, r, TOPK, false, [&](int p) {
        sp[atomicAdd(sd + SD_S2 + 5, 1)] = p;
        atomicAdd(bc + (p >> sh), 1);
      }, tid, st);
  sstamp(st, 3, tid);
  const bool bad = sd[SD_S2 + 6] != 0 || (pk && sd[SD_S2 + 5] != TOPK);   // (cannot happen: safety nets, counted apart)
  cp_wait_all();
  cbar();
  if (!pk || bad) {
    if (tid == 0 && a.slow) atomicAdd(a.slow, bad ? 0x1000000 : 0x10000);
    select_row(a, r, valid, so, tid, cb, n_c, nullptr, btp);
  } else {
    {
      int tot;
      bs0[tid] = bscan(bc[tid], tot, sptr<int>(so + SC_WS), tid);   // (its barriers publish bs0 before the scatter reads)
      cbar();
#pragma unroll
      for (int k = 0; k < TOPK / CTH; ++k) {   // scatter (any order inside a bucket)
        const int p = sp[tid + CTH * k], b = p >> sh;
        tmp[bs0[b] + atomicAdd(bcur + b, 1)] = p;
      }
      cbar();
#pragma unroll
      for (int k = 0; k < TOPK / CTH; ++k) {   // rank inside the bucket (positions unique)
        const int i = tid + CTH * k, p = tmp[i], b = p >> sh, b0 = bs0[b], b1 = b0 + bc[b];
        int rk = 0;
        for (int j = b0; j < b1; ++j) rk += tmp[j] < p;
        pos[b0 + rk] = p;
      }
    }
    cbar();
    sstamp(st, 4, tid);
#pragma unroll
    for (int k = 0; k < TOPK / CTH; ++k) {
      int p = pos[tid + CTH * k];
      if (cb) p = (tabs.scb[p >> 3] << 3) | (p & 7);
      const int pg = stg ? tabs.sbt[p / a.bs] : __ldg(bt + p / a.bs);
      a.gout[(size_t)r * a.ldg + tid + CTH * k] = pg * a.bs + p % a.bs;
    }
    if (tid == 0) a.glen[r] = valid ? min(n, TOPK) : 0;
  }
  sstamp(st, 5, tid);
  cbar();   // the row's histograms and counter -> 0 for the next index layer (every reader is past IX_CL)
  uint4* hz = reinterpret_cast<uint4*>(a.dsc + DS_H + r * DBINS);
  uint4* hz2 = reinterpret_cast<uint4*>(a.dsc + DS_H2 + r * DBINS);
  for (int j = tid; j < DBINS / 4; j += CTH) hz[j] = make_uint4(0u, 0u, 0u, 0u), hz2[j] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < DCB) a.dsc[DS_CO + r * DCB + tid] = 0u, a.dsc[DS_CO + (2 * NRM + r) * DCB + tid] = 0u;
  if (tid == 0) a.dsc[DS_C + r] = 0u;
}
// v14b stage 2, candidate row r (CTA 4 + r, layer 20, after IX_CL): -> cand row r / cand_n[r] as cand_row
__device__ __noinline__ void ds_cand_row(const IdxArgs& a, int r, bool valid, int so, int tid) {
  char* scr = sptr<char>(so);
  const int n = valid ? __ldcg(a.lens + r) : 0, nb = (n + BLK - 1) / BLK, NW = (nb + 31) >> 5;
  unsigned* bm = reinterpret_cast<unsigned*>(scr + SC_LKEY);   // the selected blocks' bitmap (NW <= 2,048 words)
  for (int j = tid; j < NW; j += CTH) bm[j] = 0u;   // (ds_pick2's first barrier orders it)
  const bool ok = ds_pick2(a, so, NRM + r, CAND_K, true, [&](int id) { atomicOr(bm + (id >> 5), 1u << (id & 31)); }, tid);
  if (!ok) {
    if (tid == 0 && a.slow) atomicAdd(a.slow, sptr<int>(so + SC_DS)[SD_S2 + 6] ? 0x1000000 : 0x10000);
    cand_row(a, r, valid, so, tid);
  } else {
    int* out = a.cand + (size_t)r * CAND_K;
    const int q = (NW + CTH - 1) / CTH, w0 = tid * q;   // words w0 .. w0 + q - 1 of this thread, ascending
    int cw = 0;
    for (int j = 0; j < q; ++j)
      if (w0 + j < NW) cw += __popc(bm[w0 + j]);
    int tot;
    int off = bscan(cw, tot, sptr<int>(so + SC_WS), tid);
    for (int j = 0; j < q; ++j)
      if (w0 + j < NW)
        for (unsigned m = bm[w0 + j]; m; m &= m - 1u) out[off++] = 32 * (w0 + j) + __ffs(m) - 1;
    for (int i = tot + tid; i < CAND_K; i += CTH) out[i] = -1;
    if (tid == 0 && a.cand_n) a.cand_n[r] = tot;
  }
  cbar();
  uint4* hz = reinterpret_cast<uint4*>(a.dsc + DS_HB + r * DBINS);
  uint4* hz2 = reinterpret_cast<uint4*>(a.dsc + DS_H2 + (NRM + r) * DBINS);
  for (int j = tid; j < DBINS / 4; j += CTH) hz[j] = make_uint4(0u, 0u, 0u, 0u), hz2[j] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < DCB) a.dsc[DS_CO + (NRM + r) * DCB + tid] = 0u, a.dsc[DS_CO + (3 * NRM + r) * DCB + tid] = 0u;
  if (tid == 0) a.dsc[DS_C + NRM + r] = 0u;
}

#else
// the K-th key's top 22 bits from the
// two histograms (P22; A keys above it, G keys with it); the list's keys above P22 are selected, the boundary group (<= 256
// keys) ranked exactly by (key desc, id asc) for the K - A still needed; f(id) per selected id. drop_ninf: -inf keys are
// never selected (cand_row's rule). Returns false (v13's routine instead) on a list overflow or a boundary group > 256.
// (v14h) ONE noinline copy for both callers: mode 0 (select row) appends the position to the list at SC_LKEY and counts its
// counting-sort bucket (SC_HIST, 2^sh positions per bucket); mode 1 (candidate row, the -inf rule) sets the block's bit in
// the bitmap at SC_LKEY. (v14b-g: a template on a lambda, inlined into both callers.)
__device__ __noinline__ bool ds_pick2(const IdxArgs& a, int so, int q, int K, int mode, int sh, int tid,
                                      unsigned long long* st = nullptr) {
  const bool drop_ninf = mode == 1;
  auto f = [&](int p) {
    if (mode == 0) {
      sptr<int>(so + SC_LKEY)[atomicAdd(sptr<int>(so + SC_DS) + SD_S2 + 5, 1)] = p;
      atomicAdd(sptr<int>(so + SC_HIST) + (p >> sh), 1);
    } else {
      atomicOr(sptr<unsigned>(so + SC_LKEY) + (p >> 5), 1u << (p & 31));
    }
  };
  int* sd = sptr<int>(so + SC_DS);
  uint32_t* bk = sptr<uint32_t>(so + SC_CKEY);
  int* bp = sptr<int>(so + SC_CKEY + GMAX * 4);
  const int lane = tid & 31, w = tid >> 5;
  const int ab1 = sd[SD_AB1 + q];
  if (w == 0) {
    const GF f2 = gfind(a.dsc + DS_H2 + q * DBINS, ds_co(a.dsc, DS_H2 + q * DBINS), K - ab1, lane);
    if (lane == 0) sd[SD_S2] = f2.d, sd[SD_S2 + 1] = f2.ab, sd[SD_S2 + 2] = f2.cn;
  }
  if (tid == 32) sd[SD_S2 + 3] = (int)__ldcg(a.dsc + DS_C + q), sd[SD_S2 + 4] = 0, sd[SD_S2 + 5] = 0, sd[SD_S2 + 6] = 0;
  cbar();
  sstamp(st, 1, tid);
  const int C = sd[SD_S2 + 3], A = ab1 + sd[SD_S2 + 1], G = sd[SD_S2 + 2];
  if (tid == 0) {   // (tests: list / group of the last call and per layer)
    const unsigned dv = (unsigned)min(C, 0xffff) | ((unsigned)min(G, 0xffff) << 16);
    a.dsc[DS_C + 8 + q] = dv;
    if (a.L >= 0 && a.L < 40) a.dsc[DS_DBG + (a.L >> 2) * 8 + q] = dv;
  }
  if (C > DCAP || C < K || G > GMAX) return false;
  const uint32_t P22 = ((uint32_t)sd[SD_D1 + q] << DBITS) | (uint32_t)sd[SD_S2];
  const unsigned* lk = a.dsc + (q < NRM ? DS_LK + q * DCAP : DS_BK + (q - NRM) * DCAP);
  const int* lp = reinterpret_cast<const int*>(a.dsc + (q < NRM ? DS_LP + q * DCAP : DS_BP + (q - NRM) * DCAP));
#pragma unroll 1
  for (int i0 = 0; i0 < C; i0 += 8 * CTH) {   // keys + ids, 16 loads in flight per thread
    uint32_t kk[8];
    int pp[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = i0 + k * CTH + tid;
      kk[k] = ldcg_if(lk + (i < C ? i : 0), i < C);
      pp[k] = (int)ldcg_if(lp + (i < C ? i : 0), i < C);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (i0 + k * CTH + tid < C) {
        const uint32_t t22 = kk[k] >> (32 - 2 * DBITS);
        if (t22 > P22) {
          f(pp[k]);
        } else if (t22 == P22) {
          const int s = atomicAdd(sd + SD_S2 + 4, 1);
          if (s < GMAX) bk[s] = kk[k], bp[s] = pp[k];
        }
      }
  }
  cbar();
  sstamp(st, 2, tid);
  if (sd[SD_S2 + 4] != G) {   // (cannot happen: the list holds the whole group; a safety net, flagged)
    if (tid == 0) sd[SD_S2 + 6] = 1;
    cbar();
    return false;
  }
  if (tid < G) {   // the boundary group: rank by (key desc, id asc); the first K - A selected
    const uint32_t x = bk[tid];
    const int p = bp[tid];
    int rk = 0;
    for (int j = 0; j < G; ++j) {
      const uint32_t y = bk[j];
      rk += (y > x) || (y == x && bp[j] < p);
    }
    if (rk < K - A && !(drop_ninf && x == NEG_INF_KEY)) f(p);
  }
  cbar();
  return true;
}
// v14b stage 2, select row r (row CTA r, after IX_CL): -> ti row r / tl[r] as select_row
__device__ __noinline__ void ds_select_row(const IdxArgs& a, int r, bool valid, int so, int tid, const int* cb, int n_c,
                                           const int* btp, unsigned long long* st = nullptr) {
  sstamp(st, 0, tid);
  char* scr = sptr<char>(so);
  const int* bt = btp != nullptr ? btp : a.bt;
  int* sd = sptr<int>(so + SC_DS);
  const int n = cb ? n_c : __ldcg(a.lens + r);
  const int nfull = cb ? __ldcg(a.lens + r) : n;
  const int npg = (nfull + a.bs - 1) / a.bs;
  const bool stg = npg <= NBT;
  const SlotTabs tabs{reinterpret_cast<int*>(scr + SC_CPOS) + NCB, reinterpret_cast<int*>(scr + SC_CPOS), bt, cb,
                      stg ? npg : 0, cb ? (n + BLK - 1) / BLK : 0};
  tabs_issue(tabs, tid);
  int* sp = reinterpret_cast<int*>(scr + SC_LKEY);   // [512] the selected positions (any order)
  int* pos = reinterpret_cast<int*>(scr + SC_POS);
  // (v14d) ascending by a counting sort on 256 buckets of 2^sh positions (one per thread): counts during the pick, a scan,
  // a scatter, then each item's rank inside its bucket (v14c's bitmap scan took 1-5 us at 80-131K)
  int* bc = reinterpret_cast<int*>(scr + SC_HIST);   // [256] bucket counts, [256] starts, [256] cursors, [512] scattered
  int* bs0 = bc + CTH;
  int* bcur = bs0 + CTH;
  int* tmp = bcur + CTH;
  const int sh = max(0, 32 - __clz(max(n - 1, 1)) - 8);
  bc[tid] = 0, bcur[tid] = 0;   // (ds_pick2's first barrier orders it)
  const bool pk = ds_pick2(a, so, r, TOPK, 0, sh, tid, st);
  sstamp(st, 3, tid);
  const bool bad = sd[SD_S2 + 6] != 0 || (pk && sd[SD_S2 + 5] != TOPK);   // (cannot happen: safety nets, counted apart)
  cp_wait_all();
  cbar();
  if (!pk || bad) {
    if (tid == 0 && a.slow) atomicAdd(a.slow, bad ? 0x1000000 : 0x10000);
    select_row(a, r, valid, so, tid, cb, n_c, nullptr, btp);
  } else {
    {
      int tot;
      bs0[tid] = bscan(bc[tid], tot, sptr<int>(so + SC_WS), tid);   // (its barriers publish bs0 before the scatter reads)
      cbar();
#pragma unroll
      for (int k = 0; k < TOPK / CTH; ++k) {   // scatter (any order inside a bucket)
        const int p = sp[tid + CTH * k], b = p >> sh;
        tmp[bs0[b] + atomicAdd(bcur + b, 1)] = p;
      }
      cbar();
#pragma unroll
      for (int k = 0; k < TOPK / CTH; ++k) {   // rank inside the bucket (positions unique)
        const int i = tid + CTH * k, p = tmp[i], b = p >> sh, b0 = bs0[b], b1 = b0 + bc[b];
        int rk = 0;
        for (int j = b0; j < b1; ++j) rk += tmp[j] < p;
        pos[b0 + rk] = p;
      }
    }
    cbar();
    sstamp(st, 4, tid);
#pragma unroll
    for (int k = 0; k < TOPK / CTH; ++k) {
      int p = pos[tid + CTH * k];
      if (cb) p = (tabs.scb[p >> 3] << 3) | (p & 7);
      const int pg = stg ? tabs.sbt[p / a.bs] : __ldg(bt + p / a.bs);
      a.gout[(size_t)r * a.ldg + tid + CTH * k] = pg * a.bs + p % a.bs;
    }
    if (tid == 0) a.glen[r] = valid ? min(n, TOPK) : 0;
  }
  sstamp(st, 5, tid);
  cbar();   // the row's histograms and counter -> 0 for the next index layer (every reader is past IX_CL)
  uint4* hz = reinterpret_cast<uint4*>(a.dsc + DS_H + r * DBINS);
  uint4* hz2 = reinterpret_cast<uint4*>(a.dsc + DS_H2 + r * DBINS);
  for (int j = tid; j < DBINS / 4; j += CTH) hz[j] = make_uint4(0u, 0u, 0u, 0u), hz2[j] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < DCB) a.dsc[DS_CO + r * DCB + tid] = 0u, a.dsc[DS_CO + (2 * NRM + r) * DCB + tid] = 0u;
  if (tid == 0) a.dsc[DS_C + r] = 0u;
}
// v14b stage 2, candidate row r (CTA 4 + r, layer 20, after IX_CL): -> cand row r / cand_n[r] as cand_row
__device__ __noinline__ void ds_cand_row(const IdxArgs& a, int r, bool valid, int so, int tid) {
  char* scr = sptr<char>(so);
  const int n = valid ? __ldcg(a.lens + r) : 0, nb = (n + BLK - 1) / BLK, NW = (nb + 31) >> 5;
  unsigned* bm = reinterpret_cast<unsigned*>(scr + SC_LKEY);   // the selected blocks' bitmap (NW <= 2,048 words)
  for (int j = tid; j < NW; j += CTH) bm[j] = 0u;   // (ds_pick2's first barrier orders it)
  const bool ok = ds_pick2(a, so, NRM + r, CAND_K, 1, 0, tid);
  if (!ok) {
    if (tid == 0 && a.slow) atomicAdd(a.slow, sptr<int>(so + SC_DS)[SD_S2 + 6] ? 0x1000000 : 0x10000);
    cand_row(a, r, valid, so, tid);
  } else {
    int* out = a.cand + (size_t)r * CAND_K;
    const int q = (NW + CTH - 1) / CTH, w0 = tid * q;   // words w0 .. w0 + q - 1 of this thread, ascending
    int cw = 0;
    for (int j = 0; j < q; ++j)
      if (w0 + j < NW) cw += __popc(bm[w0 + j]);
    int tot;
    int off = bscan(cw, tot, sptr<int>(so + SC_WS), tid);
    for (int j = 0; j < q; ++j)
      if (w0 + j < NW)
        for (unsigned m = bm[w0 + j]; m; m &= m - 1u) out[off++] = 32 * (w0 + j) + __ffs(m) - 1;
    for (int i = tot + tid; i < CAND_K; i += CTH) out[i] = -1;
    if (tid == 0 && a.cand_n) a.cand_n[r] = tot;
  }
  cbar();
  uint4* hz = reinterpret_cast<uint4*>(a.dsc + DS_HB + r * DBINS);
  uint4* hz2 = reinterpret_cast<uint4*>(a.dsc + DS_H2 + (NRM + r) * DBINS);
  for (int j = tid; j < DBINS / 4; j += CTH) hz[j] = make_uint4(0u, 0u, 0u, 0u), hz2[j] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < DCB) a.dsc[DS_CO + (NRM + r) * DCB + tid] = 0u, a.dsc[DS_CO + (3 * NRM + r) * DCB + tid] = 0u;
  if (tid == 0) a.dsc[DS_C + NRM + r] = 0u;
}

#endif
// ================================================================================================ index-layer front
// The rest of an index layer on the owner GPU (c1mega hooks: producer hook 1 streams the X1 / X3 weights; hook 3 runs
// x1_consume, x3_consume and, on the service CTAs 128 .. 131, x2_row; hook 4 runs x45). Checkpoint dtypes: compressor
// wkv | wgate bf16 [512 ratio][5120] (KV sources), weights_proj bf16 [32][5120], wq_b e4m3 [4096][1280] + ue8m0
// [128][40], wk bf16 [128][512], norms bf16. Same math as the engine's pieces (hooks_idx.run): bf16 x bf16 / e4m3 x f16
// MMAs with fp32 accumulation for the GEMVs (other summation order than cuBLAS), the stock ops' statements elsewhere.
constexpr int DM = 5120, QL = 1280, QN = HEADS * HD;
constexpr int NWK = 128;                       // c1mega's worker CTAs (X3: 32 rows of wq_b each)
constexpr int X3_S = 8192;                     // X3 stage: [2 kblocks][32 rows][128 B], then the row block's 48 scale bytes
constexpr int IMS = DM * 2 + 64;               // c1mega's activation image row stride (bytes)
// X4 scratch layout (shared memory, >= XS_BYTES, 1024-aligned): 2 key tiles + scales, the Q image, the weights, an mbarrier
constexpr int XS_T0 = 0, XS_SC = 16384, XS_Q = 17408, XS_W = XS_Q + QIMG, XS_MB = 63736, XS_BYTES = 63744;
// the q quant's inputs, staged before the key tiles arrive: q rows 0, 1 (bf16, 8 KB each) at XS_T0, rows 2, 3 at XS_QS,
// weights_proj [4][32] bf16 at XS_WB, the rows' cos / sin (4 x 64 fp32) at XS_CS
constexpr int XS_QS = XS_W + 512, XS_WB = XS_QS + 16384, XS_CS = XS_WB + 256;
static_assert(XS_W + 512 <= XS_MB && SC_BYTES <= XS_MB && XS_CS + 1024 <= XS_MB, "x4 scratch");

struct IxLayer {
  IdxArgs s;                  // X4 / X5 (s.q8 / s.wo: optional debug copies of the Q image (unswizzled) and the weights)
  int kind;                   // 1: indexer only (layers 24 .. 36); 2: + compressor (KV sources 2, 8, 14, 20)
  int NR, ratio;
  const __nv_bfloat16* ain;   // [NR][5120] the attention input a (natural bf16)
  const long long* pos;       // [NR]
  const float* cos_sin;       // compressor / indexer RoPE table [pos][64] (cos 0 .. 31, sin 32 .. 63)
  int nt1, ntc;               // X1 tiles of 16 rows: ntc compressor tiles (64 / 32 / 0), then 2 weights_proj tiles
  float* ks;                  // [NRM][1024] fp32 (compressor kv | score)
  __nv_bfloat16* wbuf;        // [NRM][32] weights_proj(a), bf16
  __nv_bfloat16* q;           // [NRM][4096] wq_b(qr), bf16
  const uint8_t* wq_s;        // [128][48] ue8m0 (40 used)
  const __nv_bfloat16* wk;    // [128][512]
  const __nv_bfloat16* cnorm; // [512]
  const __nv_bfloat16* knorm; // [128]
  float eps;                  // both RMSNorms (the engine: 1e-20)
  float* ring;                // ratio 2: this layer's ring (state slot b: ring + b * ring_stride, 8 rows of 1024 fp32)
  long long ring_stride;      // floats per state slot (3 x 8 x 1024)
  const long long* rslot;     // [NR] ring slots (state slot * 8 + p % 8), -1 = none
  const long long* cslot;     // [NR] compressed slots (page * rpp + record in page), -1 = none
  uint8_t* comp;              // comp<L> page 0: rpp x 576 data, then rpp x 8 scales
  uint8_t* kidx;              // idx<L> page 0: rpp x 128 values, then rpp x fp32 scales
  long long pstride;          // page stride (116,928)
  const CUtensorMap* tm;      // [3]: compressor {64, 512 ratio, 80} / weights_proj {64, 32, 80}: box {64, 16, 8};
                              // wq_b {128, 4096, 10} box {128, 32, 2}; 128-B swizzle, global memory, 64-B aligned
  int skip;                   // tests: bit 0 = X1 inputs given (ks / wbuf), bit 1 = X3 given (q): no stream, no compute;
                              // bit 3 (C1IDX_SEL2=1): the row CTAs run their select twice (the 2nd warm, stamp 8 + 7)
  unsigned* itick;            // c1mega: [132] per-CTA index-layer epoch (the counters' targets are 132 x (itick + 1))
  unsigned* selflag;          // c1mega: [132] the layer epoch (ncall) once this layer's top-k is final (the push warp waits)
  unsigned long long* tim;    // c1mega tests (C1IDX_TIM=1): [132][32] globaltimer sub-stamps of the glue (c1hooks_ix.inc), or null
  unsigned* rowsel;           // c1mega: [NRM] the layer epoch (ncall) once row r's top-k is final (ti row r, tl[r], the new
                              // compressed records; hook 6: the push warp's per-row wait), or null
};

// X1 tile i on CTA X1C0 + i; -1 = no tile (nt1 <= 66: 64 compressor + 2 weights_proj)
constexpr int X1C0 = 62;
__device__ __forceinline__ int x1_tile_of(int c) { return c >= X1C0 && c < X1C0 + 66 ? X1C0 + 65 - c : -1; }   // (v13: from
// CTA 127 down: layer 20's 34 tiles on 94..127, clear of the logits CTAs 0..64 of an 8K row)
// the caller's SMEM weight ring (c1mega's: STAGE-byte stages, full / empty mbarrier arrays, empty count = 8 warps), by value:
// the phases return the advanced `it`
constexpr int RING_STAGE = 20480;
struct Ring {
  uint32_t base, full, empty;   // shared addresses: stage 0, full[0], empty[0] (8-B mbarriers)
  int ns;
  long it;                      // chunks consumed so far (returned to the caller)
  int s;                        // it % ns
  unsigned ph;                  // (it / ns) & 1
};
__device__ __forceinline__ Ring mk_ring(uint32_t base, uint32_t full, uint32_t empty, int ns, long it) {
  return Ring{base, full, empty, ns, it, (int)((unsigned)it % (unsigned)ns), ((unsigned)it / (unsigned)ns) & 1u};   // (32-bit: no div_s64)
}
__device__ __forceinline__ uint32_t rwait(const Ring& r) {
  asm volatile("{\n .reg .pred p;\n LAB_RW%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_RW%=;\n}" ::"r"(
                   r.full + 8 * r.s),
               "r"(r.ph)
               : "memory");
  return r.base + r.s * RING_STAGE;
}
__device__ __forceinline__ void rrel(Ring& r, int lane) {
  __syncwarp();
#ifdef C1M_XN   // the ring's empty count is NCWX = 8 + C1M_XN: base warp w arrives with c1mega_xn's xrc_of(w)
  if (lane == 0) {
    const unsigned xw = threadIdx.x >> 5, xc = 1u + (unsigned)(C1M_XN / 8) + (xw < (unsigned)(C1M_XN % 8) ? 1u : 0u);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(r.empty + 8 * r.s), "r"(xc) : "memory");
  }
#else
  if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(r.empty + 8 * r.s) : "memory");
#endif
  ++r.it;
  if (++r.s == r.ns) r.s = 0, r.ph ^= 1u;
}
// own counters (IX_*): arrive = consumer barrier, fence, thread 0 red.release; wait = thread 0 spins, consumer barrier
__device__ __forceinline__ void ix_arr(unsigned* cnt, int k, int tid) {
  cbar();
  if (tid == 0) {
    __threadfence();
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(cnt + k), "r"(1u) : "memory");
  }
}
// waits poll with RELAXED loads and acquire ONCE: an acquire load compiles to LDG.STRONG + CCTL.IVALL (the whole
// SM's L1 invalidated per poll), which in situ stalls the other warps' LSU traffic (smem included) on the same SM
__device__ __forceinline__ unsigned ld_rlx_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_acq() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
// waits where the whole CTA
// sits at the barrier keep acquire polls (their per-poll L1 invalidation hits no computing warp); relaxed polls + one fence
// only where a warp polls NEXT TO computing warps (the push warp's pushwait / rowwait, E0's xready)
__device__ __forceinline__ void ix_spin(const unsigned* cnt, int k, unsigned target, int tid) {
  if (tid == 0) {
    unsigned v;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt + k) : "memory");
    } while ((int)(v - target) < 0);
  }
  cbar();
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void ldm_x4(uint32_t a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void hmma_bf16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                          unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cvt_e4m3x4(unsigned v, unsigned& lo2, unsigned& hi2) {
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, h;\n}"
      : "=r"(lo2), "=r"(hi2)
      : "r"(v));
}
__device__ __forceinline__ float pow2i(int p) { return __int_as_float((127 + p) << 23); }
__device__ __forceinline__ int ceil_log2(float v) {   // exact ceil(log2(v)), v > 0 normal (= stock's ceil(log2) on these inputs)
  const unsigned b = __float_as_uint(v);
  return (int)(b >> 23) - ((b & 0x7fffffu) == 0 ? 127 : 126);
}
__device__ __forceinline__ float ex2a(float x) {   // Triton tl.exp (fp32): ex2.approx(x * log2 e)
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(x, 1.4426950408889634f)));
  return y;
}
__device__ __forceinline__ float divf(float a, float b) {   // Triton fdiv (ieee_rounding False): div.full
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float rsqa(float x) {   // Triton tl.rsqrt (its PTX: rsqrt.approx.ftz)
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ int tceil_log2(float v) {   // Triton ceil(log2(v)): libdevice log2f, then ceil (not always the
  return (int)ceilf(log2f(v));                          // exact ceil(log2): v one ulp above a power of 2 gives log2 == k)
}
// Triton's RoPE statements as its PTX contracts them: e c - o s = fma(e, c, -(o s)); o c + e s = fma(o, c, e s)
__device__ __forceinline__ float rope_e(float e, float o, float c, float s) { return fmaf(e, c, -__fmul_rn(o, s)); }
__device__ __forceinline__ float rope_o(float e, float o, float c, float s) { return fmaf(o, c, __fmul_rn(e, s)); }
// Triton's 4-element sum of squares per thread (its PTX: x1 x1 first, then fma x0, x2, x3)
__device__ __forceinline__ float sq4(float x0, float x1, float x2, float x3) {
  return fmaf(x3, x3, fmaf(x2, x2, fmaf(x0, x0, __fmul_rn(x1, x1))));
}
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ unsigned f2e4m3x2(float a, float b) {   // satfinite RN (Triton .to(float8e4nv))
  return (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
// cross-warp sum of acc[NT][4] (warp order 0 .. 7) into warp 0's registers; red: 8 x NT x 32 x 4 floats
template <int NT>
__device__ __forceinline__ void wsum(float (&acc)[NT][4], float* red, int w, int lane) {
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[((w * NT + t) * 32 + lane) * 4 + i] = acc[t][i];
  cbar();
  if (w < NT) {   // warp t finalizes tile t
    const int t = w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float a = red[((0 * NT + t) * 32 + lane) * 4 + i];
      for (int ww = 1; ww < CTH / 32; ++ww) a += red[((ww * NT + t) * 32 + lane) * 4 + i];
      acc[t][i] = a;
    }
  }
}

// ---- X1 (hook 3, owner): one 16-row tile of [compressor | weights_proj] x a: 10 ring chunks of 512 K ([8 kblocks of 64]
// [16 rows][128 B], 128-B swizzle); B = the natural a rows from global (L2), one chunk ahead. cwait() -> stage address.
__device__ __forceinline__ void x1_consume(const IxLayer& L, Ring& R, float* red, int c, int w, int lane,
                                           unsigned long long* st = nullptr, uint32_t bimg = 0u, uint32_t btail = 0u) {
  // B of k >= 1280 from the CTA's own A2 image at hook 3 (natural bf16 rows at btail + n IMS
  // A4's qr image overwrote only bytes 0 .. 2559 of each row), so only k < 1280 (20 of the 80 words per lane) come from L2
  // (st: tests' per-chunk stamps 23 .. 25; bimg != 0: B from c1mega's A2 image in SMEM (natural bf16 rows at bimg + n IMS,
  // the same bits as L.ain), the X1-early placement of hook 7)
  const int i = x1_tile_of(c);
  if ((L.skip & 1) || i < 0 || i >= L.nt1) return;
  const int g = lane >> 2, t = lane & 3, lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
  const bool has = g < L.NR;
  const unsigned* arow = reinterpret_cast<const unsigned*>(L.ain + (size_t)(has ? g : 0) * DM);
  float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
  // The B operand of all 10 chunks (this lane's 80 words of the 4 rows' a)
  // goes into registers before the first stage wait: unconditional loads (lanes without a row read row 0's words, then
  // select 0), so the loop below waits for ring stages only, not one L2 round per chunk
  unsigned b[10][4][2];
  if (bimg != 0u) {
    const uint32_t rb = bimg + (uint32_t)((has ? g : 0) * IMS);
#pragma unroll
    for (int kc = 0; kc < 10; ++kc)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = 512 * kc + 16 * (w + 8 * j) + 2 * t;
        b[kc][j][0] = lds32(rb + 2 * k);
        b[kc][j][1] = lds32(rb + 2 * k + 16);
      }
  } else {
#pragma unroll
    for (int kc = 0; kc < 10; ++kc)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = 512 * kc + 16 * (w + 8 * j) + 2 * t;   // element index (even): u32 index k / 2
#ifdef C1IDX_X1IMG
        if (512 * kc + 128 * j >= 1280) continue;   // (compile-time split: from the image below; the caller passes btail)
#endif
        b[kc][j][0] = __ldcg(arow + (k >> 1));
        b[kc][j][1] = __ldcg(arow + (k >> 1) + 4);
      }
#ifdef C1IDX_X1IMG
    {
      const uint32_t rb = btail + (uint32_t)((has ? g : 0) * IMS);
#pragma unroll
      for (int kc = 2; kc < 10; ++kc)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (512 * kc + 128 * j < 1280) continue;
          const int k = 512 * kc + 16 * (w + 8 * j) + 2 * t;
          b[kc][j][0] = lds32(rb + 2 * k);
          b[kc][j][1] = lds32(rb + 2 * k + 16);
        }
    }
#else
    (void)btail;
#endif
  }
#pragma unroll
  for (int kc = 0; kc < 10; ++kc) {
    const uint32_t sa = rwait(R);
    if (kc == 0 || kc == 6 || kc == 9) sstamp(st, kc == 0 ? 23 : kc == 6 ? 24 : 25, 32 * w + lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = w + 8 * j;
      const uint32_t off = (kk >> 2) * 2048 + lr * 128 + (((((kk & 3) << 1) | lc) ^ (lr & 7)) << 4);
      unsigned a0, a1, a2, a3;
      ldm_x4(sa + off, a0, a1, a2, a3);
      hmma_bf16(acc[0], a0, a1, a2, a3, has ? b[kc][j][0] : 0u, has ? b[kc][j][1] : 0u);
    }
    rrel(R, lane);
  }
  wsum<1>(acc, red, w, lane);
  if (w == 0)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = g + 8 * (e >> 1), n = 2 * t + (e & 1);
      if (n >= L.NR) continue;
      if (i < L.ntc) L.ks[(size_t)n * 1024 + 16 * i + row] = acc[0][e];
      else L.wbuf[n * HEADS + 16 * (i - L.ntc) + row] = __float2bfloat16_rn(acc[0][e]);
    }
  cbar();   // red reusable
}

// ---- X3 (hook 3, owner, CTAs c < 128): q rows 32 c .. 32 c + 31 = wq_b . qr over the f16 P8 qr image (row n at img + n *
// IMS, f16 x 2^pv[n]; A5's statements): 5 chunks of 256 K, warp w = k32 block w of each chunk for both 16-row tiles
__device__ __forceinline__ void x3_consume(const IxLayer& L, Ring& R, uint32_t img, const int* pv, float* red, int c, int w,
                           int lane, unsigned long long* tst = nullptr) {   // (tst: tests' per-chunk stamps 26, 27)
  if ((L.skip & 2) || c >= NWK) return;
  const int g = lane >> 2, t4 = lane & 3;
  float acc[2][4];
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[t][e] = 0.f;
  for (int kc = 0; kc < 5; ++kc) {
    const uint32_t st = rwait(R);
    if (kc == 0 || kc == 4) sstamp(tst, kc == 0 ? 26 : 27, 32 * w + lane);
    const int kb = w;
    uint4 bv = make_uint4(0u, 0u, 0u, 0u);
    if (g < L.NR) bv = lds128(img + g * IMS + (256 * kc + 32 * kb) * 2 + 16 * t4);
    const uint32_t off = (kb >> 2) * 4096 + g * 128 + (((((kb & 3) << 1) | (t4 >> 1)) ^ g) << 4) + ((t4 & 1) << 3);
    unsigned sb;
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + X3_S + 8 * kc + kb));
    const float sc = __uint_as_float(sb << 23);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const uint32_t ra = st + t * 2048 + off;
      const uint2 wg = lds64(ra), wh = lds64(ra + 1024);
      const unsigned pg = __byte_perm(wg.x, wg.y, 0x6420), qg = __byte_perm(wg.x, wg.y, 0x7531);
      const unsigned ph = __byte_perm(wh.x, wh.y, 0x6420), qh = __byte_perm(wh.x, wh.y, 0x7531);
      unsigned g02, g46, h02, h46, g13, g57, h13, h57;
      cvt_e4m3x4(pg, g02, g46);
      cvt_e4m3x4(ph, h02, h46);
      cvt_e4m3x4(qg, g13, g57);
      cvt_e4m3x4(qh, h13, h57);
      float tmp[4] = {0.f, 0.f, 0.f, 0.f};
      mma16816(tmp, g02, h02, g46, h46, bv.x, bv.y);
      mma16816(tmp, g13, h13, g57, h57, bv.z, bv.w);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[t][e] = fmaf(tmp[e], sc, acc[t][e]);
    }
    rrel(R, lane);
  }
  wsum<2>(acc, red, w, lane);
  if (w < 2)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = g + 8 * (e >> 1), n = 2 * t4 + (e & 1);
      if (n < L.NR) L.q[(size_t)n * QN + 32 * c + 16 * w + row] = __float2bfloat16_rn(acc[w][e] * pow2i(-pv[n]));
    }
  cbar();
}

// ---- X2 (hook 3, owner, service CTA 128 + n, after IX_KS): row n of the compressor: ring store (ratio 2), the pooled +
// normalized latent (fused_save_compress_norm), its 584-B record (rope_quant_insert), kpre = wk . latent and the 132-B
// index-K entry (indexer_k_norm_rope_store). sm: >= 1,024 floats of shared scratch.
// does row n write a compressed record this step (the producer streams X2's chunks iff so: both sides use this)
__device__ __forceinline__ bool x2_has_record(const IxLayer& L, int n) {
  const long long p = __ldg(L.pos + n), cs = __ldg(L.cslot + n);
  const bool lat_ok = L.ratio == 1 ? cs >= 0 : __ldg(L.rslot + n) >= 0;   // (row 0: the request program's state_slots[start])
  return (p + 1) % L.ratio == 0 && cs >= 0 && lat_ok;
}
// X2's ring chunks (hook 1 streams them after the X1 tile on the service CTA of a row with a record): 8 stages of wk rows
// 16 k .. 16 k + 15 (16 KB); stage 0 also carries cnorm (1 KB, at X2_XO), knorm (256 B), the cos / sin row of the record's
// position (256 B)
constexpr int X2_XO = 16384, X2_XB = 1024 + 256 + 256;
__device__ __forceinline__ const float* x2_csrow(const IxLayer& L, int n) {
  const long long p = __ldg(L.pos + n);
  return L.cos_sin + (size_t)((p / L.ratio) * L.ratio) * 64;
}
// X2 (service CTA 128 + n, kind 2): waits for X1 (IX_KS target kst; thread 0's), then the compressor tail of row n. Its
// constant inputs come through the ring (staged while X1 streams): one L2 round (ks) after the wait instead of ~8.
__device__ void x2_row(const IxLayer& L, int n, int smo, int tid, Ring& R, unsigned kst, unsigned long long* st = nullptr) {
  float* sm = sptr<float>(smo);   // (byte offset: see soff)
  const int lane = tid & 31, w = tid >> 5;
  sstamp(st, 0, tid);
  const long long p = __ldg(L.pos + n);
  const int ratio = L.ratio;
  const float* ksr = L.ks + (size_t)n * 1024;
  const long long rs = ratio == 2 ? __ldg(L.rslot + n) : -1;
  const long long cs = __ldg(L.cslot + n);
  const bool rec = x2_has_record(L, n);
  float* xb = sm + 1280;   // cnorm (512) | knorm (128) | cos (32) | sin (32), as floats
  if (rec) {
    const uint32_t sa = rwait(R);   // (stage 0 stays held: its wk rows are kpre's chunk 0)
    for (int i = tid; i < 704; i += CTH) {
      unsigned v;
      if (i < 640) {
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(sa + X2_XO + 2 * i));
        xb[i] = __uint_as_float(v << 16);
      } else {
        xb[i] = __uint_as_float(lds32(sa + X2_XO + 1280 + 4 * (i - 640)));
      }
    }
  }
  sstamp(st, 1, tid);
  ix_spin(L.s.cnt, IX_KS, kst, tid);   // X1 done: ks (and the previous row's) final
  sstamp(st, 2, tid);
  float kq[4], pq[4];                  // ksr[tid + 256 j]; the previous row's / ring row's
#pragma unroll
  for (int j = 0; j < 4; ++j) kq[j] = __ldcg(ksr + tid + CTH * j);
  if (rec && ratio == 2) {
    const float* prv = n == 0 ? L.ring + (__ldg(L.rslot) >> 3) * L.ring_stride + ((p - 1) & 7) * 1024
                              : L.ks + (size_t)(n - 1) * 1024;
#pragma unroll
    for (int j = 0; j < 4; ++j) pq[j] = __ldcg(prv + tid + CTH * j);
  }
  if (ratio == 2 && rs >= 0) {   // ring tail store (the chunk's rows: <= 4 < 8, so no row of this chunk is read after its store)
    float* dst = L.ring + (rs >> 3) * L.ring_stride + (rs & 7) * 1024;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[tid + CTH * j] = kq[j];
  }
  if (!rec) return;   // (uniform)
  // pooled (sm[0 .. 511]) -> variance -> latent (bf16-valued floats, sm[512 .. 1023])
  float pl[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int d = tid + CTH * h;
    if (ratio == 1) {
      pl[h] = kq[h];
    } else {
      const float kv0 = pq[h], s0 = pq[h + 2], kv1 = kq[h], s1 = kq[h + 2];
      const float m = fmaxf(s0, s1);
      const float e0 = ex2a(__fsub_rn(s0, m)), e1 = ex2a(__fsub_rn(s1, m));
      const float den = __fadd_rn(e0, e1);
      pl[h] = fmaf(divf(e0, den), kv0, __fmul_rn(divf(e1, den), kv1));   // (Triton's contraction)
    }
    sm[d] = pl[h];
  }
  cbar();
  sstamp(st, 3, tid);
  {   // variance: Triton's 4-warp layout (thread j < 128 holds 4 j .. 4 j + 3), butterflies, then the 4 warp sums
    float v = 0.f;
    if (tid < 128) {
      const float4 x = *reinterpret_cast<const float4*>(sm + 4 * tid);
      v = sq4(x.x, x.y, x.z, x.w);
#pragma unroll
      for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    if (tid < 128 && lane == 0) sm[1024 + w] = v;
    cbar();
    const float s4 = __fadd_rn(__fadd_rn(sm[1024], sm[1026]), __fadd_rn(sm[1025], sm[1027]));   // (butterflies 2, 1)
    const float r = rsqa(__fadd_rn(divf(s4, 512.f), L.eps));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int d = tid + CTH * h;
      sm[512 + d] = bfr(__fmul_rn(__fmul_rn(pl[h], r), xb[d]));   // (x r) w
    }
  }
  cbar();
  sstamp(st, 4, tid);
  const float* lat = sm + 512;
  const long long pg = cs / L.s.bs, ps = cs % L.s.bs;
  const float* csr = xb + 640;
  {   // rope_quant_insert: 7 x 64 e4m3 with ue8m0 scales (warps 0 .. 6), RoPE of 448 .. 511 in bf16 (warp 7)
    uint8_t* rec_ = L.comp + pg * L.pstride + ps * 576;
    uint8_t* scl = L.comp + pg * L.pstride + (long long)L.s.bs * 576 + ps * 8;
    if (w < 7) {
      const float y0 = lat[64 * w + 2 * lane], y1 = lat[64 * w + 2 * lane + 1];
      float am = fmaxf(fabsf(y0), fabsf(y1));
#pragma unroll
      for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
      const int ex = tceil_log2(__fmul_rn(fmaxf(am, 1e-4f), 1.0f / 448.0f));
      const float inv = pow2i(-ex);
      *reinterpret_cast<unsigned short*>(rec_ + 64 * w + 2 * lane) =
          (unsigned short)f2e4m3x2(__fmul_rn(y0, inv), __fmul_rn(y1, inv));
      if (lane == 0) scl[w] = (uint8_t)min(max(ex + 127, 0), 255);
    } else {
      const float ev = lat[448 + 2 * lane], od = lat[449 + 2 * lane], cc = csr[lane], sn = csr[32 + lane];
      const __nv_bfloat16 r0 = __float2bfloat16_rn(rope_e(ev, od, cc, sn));
      const __nv_bfloat16 r1 = __float2bfloat16_rn(rope_o(ev, od, cc, sn));
      reinterpret_cast<unsigned*>(rec_ + 448)[lane] =
          (unsigned)__bfloat16_as_ushort(r0) | ((unsigned)__bfloat16_as_ushort(r1) << 16);
      if (lane == 0) scl[7] = 0;
    }
  }
  // kpre = wk . latent (bf16), wk from the ring: chunk k (outputs 16 k .. 16 k + 15), warp w the outputs 16 k + 2 w, + 1,
  // lane l the K slice 16 l .. 16 l + 15 (the earlier version's per-output arithmetic: same FMA order, same xor tree)
  float* kp = sm + 1024 + 16;
  {
    float lv[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) lv[j] = lat[16 * lane + j];
    for (int k = 0; k < 8; ++k) {
      const uint32_t sa = rwait(R);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t ra = sa + (2 * w + q) * 1024 + 32 * lane;
        const uint4 u0 = lds128(ra), u1 = lds128(ra + 16);
        const unsigned uu[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          a = fmaf(__uint_as_float(uu[j] << 16), lv[2 * j], a);
          a = fmaf(__uint_as_float(uu[j] & 0xffff0000u), lv[2 * j + 1], a);
        }
#pragma unroll
        for (int o2 = 16; o2; o2 >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o2);
        if (lane == 0) kp[16 * k + 2 * w + q] = bfr(a);
      }
      rrel(R, lane);
    }
  }
  cbar();
  sstamp(st, 5, tid);
  if (w == 0) {   // indexer_k_norm_rope_store (num_warps 1: lane l holds 4 l .. 4 l + 3)
    float k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) k[j] = kp[4 * lane + j];
    float v = sq4(k[0], k[1], k[2], k[3]);
#pragma unroll
    for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    const float r = rsqa(__fadd_rn(divf(v, 128.f), L.eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) k[j] = bfr(__fmul_rn(__fmul_rn(r, k[j]), xb[512 + 4 * lane + j]));
    if (lane >= 16) {   // RoPE pairs i = 2 lane - 32 + {0, 1} of dims 64 .. 127 (bf16 round trip)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int ip = 2 * lane - 32 + h;
        const float cc = csr[ip], sn = csr[32 + ip], ev = k[2 * h], od = k[2 * h + 1];
        k[2 * h] = bfr(rope_e(ev, od, cc, sn));
        k[2 * h + 1] = bfr(rope_o(ev, od, cc, sn));
      }
    }
    float am = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])), fmaxf(fabsf(k[2]), fabsf(k[3])));
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const int ex = tceil_log2(__fmul_rn(fmaxf(am, 1e-4f), 1.0f / 448.0f));
    const float inv = pow2i(-ex);
    uint8_t* kb = L.kidx + pg * L.pstride;
    const unsigned lo = f2e4m3x2(__fmul_rn(k[0], inv), __fmul_rn(k[1], inv)), hi = f2e4m3x2(__fmul_rn(k[2], inv), __fmul_rn(k[3], inv));
    *reinterpret_cast<unsigned*>(kb + ps * 128 + 4 * lane) = (lo & 0xffffu) | (hi << 16);
    if (lane == 0) *reinterpret_cast<float*>(kb + (long long)L.s.bs * 128 + ps * 4) = pow2i(ex);
  }
}

// ---- X3b (inside x45): q (bf16) -> RoPE (dims 64 .. 127, bf16 round trip) -> per-head ue8m0 e4m3 + the folded weights
// (fused_indexer_q_rope_quant) -> the SW128 Q image + weights in shared memory. Inputs staged in shared memory first
// (x3b_stage: cp.async, every 16-B piece in flight at once; the cos / sin rows by x45 before its waits): the earlier
// version's global loads went out one or two at a time (LDG.STRONG.GPU inside the loop, cos / sin in local memory),
// ~10 serial L2 round trips = 5 us. Warp w: (row, head) pairs w + 8 (2 it + hw), it < 8, hw = lane / 16 -> row it / 2.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ uint32_t xs_qrow(int r) { return r < 2 ? XS_T0 + r * 8192 : XS_QS + (r - 2) * 8192; }
__device__ __forceinline__ void x3b_stage(const IxLayer& L, uint32_t sb, uint32_t mb, int tid) {
  // thread 0: the NR q rows (8 KB each) + weights_proj (256 B) by cp.async.bulk on the X4 mbarrier (its phase 0; written
  // by other CTAs' generic stores: proxy fence after the acquire); every thread waits
  if (tid == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"((unsigned)(L.NR * 8192 + 256)) : "memory");
    for (int r = 0; r < L.NR; ++r)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 8192, [%2];" ::"r"(sb + xs_qrow(r)),
                   "l"(L.q + (size_t)r * QN), "r"(mb)
                   : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" ::"r"(sb + XS_WB),
                 "l"(L.wbuf), "r"(mb)
                 : "memory");
  }
  asm volatile("{\n .reg .pred p;\n LAB_XS%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra LAB_XS%=;\n}" ::"r"(mb)
               : "memory");
}
__device__ void x3b_quant(const IxLayer& L, int so, int tid) {
  char* scr = sptr<char>(so);
  // half-warp per (row, head): lane l (of 16) holds dims 8 l .. 8 l + 7 (16 B), RoPE pairs 4 l - 32 .. 4 l - 29 (l >= 8)
  const int lane = tid & 31, w = tid >> 5, hw = lane >> 4, l = lane & 15;
  char* qimg = scr + XS_Q;
  float* sw = reinterpret_cast<float*>(scr + XS_W);
  const float* css = reinterpret_cast<const float*>(scr + XS_CS);
  const __nv_bfloat16* wbs = reinterpret_cast<const __nv_bfloat16*>(scr + XS_WB);
  // two groups of 4 independent (row, head) items: all loads of a group first, then the 4
  // chains (RoPE, half-warp max, quant, stores) unrolled so they interleave; the same statements per item
#pragma unroll 1
  for (int it0 = 0; it0 < 8; it0 += 4) {
  uint4 qva[4];
  float4 c4a[4], s4a[4];
#pragma unroll
  for (int iv = 0; iv < 4; ++iv) {
    const int it = it0 + iv, r = it >> 1, h = w + 16 * (it & 1) + 8 * hw;
    qva[iv] = *reinterpret_cast<const uint4*>(scr + xs_qrow(r) + h * 256 + l * 16);
    const int lc = l >= 8 ? l : 8;   // (lanes < 8: no RoPE; a valid address, unused)
    c4a[iv] = *reinterpret_cast<const float4*>(css + r * 64 + 4 * lc - 32);
    s4a[iv] = *reinterpret_cast<const float4*>(css + r * 64 + 4 * lc);
  }
#pragma unroll
  for (int iv = 0; iv < 4; ++iv) {
    const int it = it0 + iv, r = it >> 1, h = w + 16 * (it & 1) + 8 * hw, ph = 32 * r + h;
    const uint4 qv = qva[iv];
    float x[8];
    const unsigned u[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) x[2 * k] = __uint_as_float(u[k] << 16), x[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
    const bool ok = r < L.NR;
    if (l >= 8) {
      const float4 c4 = c4a[iv];
      const float4 s4 = s4a[iv];
      const float cc[4] = {c4.x, c4.y, c4.z, c4.w}, sn[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float ev = x[2 * k], od = x[2 * k + 1];
        x[2 * k] = bfr(rope_e(ev, od, cc[k], sn[k]));
        x[2 * k + 1] = bfr(rope_o(ev, od, cc[k], sn[k]));
      }
    }
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) am = fmaxf(am, fabsf(x[k]));
#pragma unroll
    for (int o = 8; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));   // (within the half-warp)
    // (the engine's CuTe IndexerQFp8Kernel: scale = 2^ceil(log2(max(amax, 1e-4) * (1 / 448))) by bit manipulation)
    const unsigned e = ((__float_as_uint(__fmul_rn(fmaxf(am, 1e-4f), 1.0f / 448.0f)) + 0x7FFFFFu) >> 23) & 0xFFu;
    const float sc = __uint_as_float(e << 23), inv = __uint_as_float((254u - e) << 23);
    unsigned pk[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) pk[k] = ok ? f2e4m3x2(__fmul_rn(x[2 * k], inv), __fmul_rn(x[2 * k + 1], inv)) : 0u;
    const uint2 v = make_uint2((pk[0] & 0xffffu) | (pk[1] << 16), (pk[2] & 0xffffu) | (pk[3] << 16));
    const int j = l >> 1;   // 16-B chunk j of the 128-B row, this lane's half of it
    *reinterpret_cast<uint2*>(qimg + ph * 128 + ((j ^ (ph & 7)) << 4) + (l & 1) * 8) = v;
    if (L.s.q8) reinterpret_cast<uint2*>(const_cast<uint8_t*>(L.s.q8))[ph * 16 + l] = v;
    if (l == 0) {
      // (w * fp32(softmax_scale * head_scale) = 2^-6) * q_scale (the CuTe kernel's statement)
      const float wv = ok ? __fmul_rn(__fmul_rn(__bfloat162float(wbs[r * HEADS + h]), 0.015625f), sc) : 0.f;
      sw[ph] = wv;
      if (L.s.wo) const_cast<float*>(L.s.wo)[ph] = wv;
    }
  }
  }
}

// ---- v14 union tiles (consumer layers in candidate mode). v13 scored each row's 2,048 candidate blocks in tiles of its
// own (4 x 256 tiles: 4 rounds of 2 per CTA, ~20 us at 20K), the MMA's other 3 rows wasted. Here a tile = 8 blocks of the
// UNION of the rows' candidate blocks (layer 20's ascending lists), all 4 rows scored at once, each row's logits / block
// maxima stored at its compact index (the same values at the same places: the per-element arithmetic does not depend on
// the tile). Built by every CTA at hook-4 entry (idle: before the IX_KS / IX_Q waits; the X4 scratch is free until the q
// staging): the rows' bitmaps, word prefix counts (rows 0..3, union), then this CTA's tiles' entries {block, the rows'
// compact block indices or -1} -> XS_UM ([8 local tiles: round 2 j + g][8 blocks][5] ints).
constexpr int UNW = 1024;                                      // rows up to 32,768 blocks (LMAX positions)
constexpr int XS_UB = 0, XS_UP = XS_UB + NRM * UNW * 4;        // bitmaps [NRM][UNW], prefixes [5][UNW + 1] (entry only)
constexpr int XS_UM = XS_CS + 1024, UM_T = 8;                  // map: 8 local tiles (T <= 1,024: <= 4 rounds)
constexpr int XS_HE = XS_QS, XS_HBK = XS_CS + 1024;            // v14 stage-0 histograms: elements (16 KB, after the q
                                                               // quant), block maxima (8 KB, layer 20 only)
static_assert(SC_DS + SD_N * 4 <= XS_MB && SC_LKEY % 8 == 0 && XS_HE % 16 == 0 && XS_HBK % 16 == 0 && XS_UP + 5 * (UNW + 1) * 4 <= XS_CS && XS_UM + UM_T * 8 * 5 * 4 <= XS_MB && XS_HE + 16384 <= XS_WB &&
                  XS_HBK + 8192 <= XS_MB && UM_T * 8 * 5 + 2 <= UMC_W,
              "v14 scratch");
__device__ __noinline__ int ub_build(const IdxArgs& a, int so, int NR, int nbmax, int c, int tid) {
  char* scr = sptr<char>(so);
  unsigned* bm = reinterpret_cast<unsigned*>(scr + XS_UB);
  int* pre = reinterpret_cast<int*>(scr + XS_UP);
  int* um = reinterpret_cast<int*>(scr + XS_UM);
  const int NW = (nbmax + 31) >> 5, lane = tid & 31, w = tid >> 5;
  for (int j = tid; j < NRM * NW; j += CTH) bm[(j / NW) * UNW + j % NW] = 0u;
  cbar();
  {
    int b[NRM][8];
#pragma unroll
    for (int r = 0; r < NRM; ++r) {
      const int m = r < NR ? __ldcg(a.cblk_n + r) : 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int i = tid + CTH * k;
        b[r][k] = i < m ? __ldcg(a.cblk + r * CAND_K + i) : -1;
      }
    }
#pragma unroll
    for (int r = 0; r < NRM; ++r)
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (b[r][k] >= 0) atomicOr(bm + r * UNW + (b[r][k] >> 5), 1u << (b[r][k] & 31));
  }
  cbar();
  if (w < 5) {   // warp k: exclusive word prefixes of array k (rows 0..3, 4 = the union); lane l: words l q .. l q + q - 1
    const int q = (NW + 31) >> 5, j0 = lane * q;
    int* pw = pre + w * (UNW + 1);
    auto word = [&](int j) -> unsigned {
      return w < 4 ? bm[w * UNW + j] : (bm[j] | bm[UNW + j] | bm[2 * UNW + j] | bm[3 * UNW + j]);
    };
    int s = 0;
    for (int j = 0; j < q; ++j)
      if (j0 + j < NW) s += __popc(word(j0 + j));
    int incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    int run = incl - s;
    for (int j = 0; j < q; ++j)
      if (j0 + j < NW) pw[j0 + j] = run, run += __popc(word(j0 + j));
    if (lane == 31) pw[NW] = incl;
  }
  cbar();
  const int* up = pre + 4 * (UNW + 1);
  const int U = up[NW];
  if (tid < UM_T * 8) {
    const int lt = tid >> 3, i = tid & 7, t = 2 * c + (lt & 1) + 2 * NCTA_IX * (lt >> 1), k = 8 * t + i;
    int e0 = -1, e1 = -1, e2 = -1, e3 = -1, e4 = -1;
    if (k < U) {
      int lo = 0, hi = NW;   // the last word whose prefix <= k (it holds the k-th union block)
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (up[mid] <= k) lo = mid;
        else hi = mid;
      }
      unsigned uw = bm[lo] | bm[UNW + lo] | bm[2 * UNW + lo] | bm[3 * UNW + lo];
      for (int q2 = k - up[lo]; q2 > 0; --q2) uw &= uw - 1u;
      const int bit = __ffs(uw) - 1;
      const unsigned below = (1u << bit) - 1u;
      auto ci = [&](int r) {
        const unsigned rw = bm[r * UNW + lo];
        return ((rw >> bit) & 1u) ? pre[r * (UNW + 1) + lo] + __popc(rw & below) : -1;
      };
      e0 = 32 * lo + bit, e1 = ci(0), e2 = ci(1), e3 = ci(2), e4 = ci(3);
    }
    int* d = um + tid * 5;
    d[0] = e0, d[1] = e1, d[2] = e2, d[3] = e3, d[4] = e4;
  }
  cbar();
  return (U + 7) >> 3;
}

// ---- X4 + X5 (hook 4, owner, all consumer threads of every CTA). scr: XS_BYTES of shared memory (1024-aligned), free for
// the hook. Waits for IX_KS / IX_Q (and IX_K for KV sources), builds the Q image, loads its key tiles by TMA (thread 0,
// 2 per round: tiles 2c + g + 264 k), the logits; CTAs 0 .. 3 the top-512 of their row, 4 .. 7 (layer 20) the candidate
// blocks; then every CTA waits for IX_SEL (the attention reads ti / tl next). v14 (IdxArgs dsc / dthr / uni): rows past
// dthr keys select with the whole grid (ds_stage1 .. ds_cand_row), consumer layers in candidate mode score union tiles.
__device__ void x45(const IxLayer& L, int so, unsigned icall, int c, int tid, unsigned long long* st = nullptr,
                    unsigned ncall = 0u) {
  char* scr = sptr<char>(so);   // (byte offset: see soff)
  const IdxArgs& a = L.s;
  auto spin = [&](int k, unsigned target) { ix_spin(a.cnt, k, target, tid); };
  auto arr = [&](int k) { ix_arr(a.cnt, k, tid); };
  const int w = tid >> 5, lane = tid & 31, wg = w >> 2, wq = w & 3;
  int n[NRM];
  int nmax = 0;
#pragma unroll
  for (int r = 0; r < NRM; ++r) {
    n[r] = r < L.NR ? __ldg(a.lens + r) : 0;
    nmax = max(nmax, n[r]);
  }
  int T = (nmax + TILE - 1) / TILE;
  const int Tdense = T;
  const int P = a.bs;
  const int* bt = a.bt != nullptr ? a.bt : a.dbt + (size_t)__ldg(a.preq) * a.dbt_ld;
  // consumer layers whose rows pass 2,048 blocks: layer 20's candidate blocks, one compact row per row (tiles of 8
  // candidate blocks; row r's tiles tr0[r] .. tr0[r + 1] - 1, or v14's union tiles); otherwise every block is a candidate
  int cnt[NRM], tr0[NRM + 1], nec[NRM], ne[NRM], nbk[NRM];
  bool cmode = false;
  if (L.kind == 1 && a.cblk != nullptr) cmode = (nmax + BLK - 1) / BLK > CAND_K;
#pragma unroll
  for (int r = 0; r < NRM; ++r) nec[r] = 0, nbk[r] = (n[r] + BLK - 1) / BLK;
  if (cmode) {
    tr0[0] = 0;
#pragma unroll
    for (int r = 0; r < NRM; ++r) {
      cnt[r] = r < L.NR ? __ldcg(a.cblk_n + r) : 0;
      tr0[r + 1] = tr0[r] + (cnt[r] + 7) / 8;
      nec[r] = cnt[r] > 0 ? 8 * cnt[r] - (8 * nbk[r] - n[r]) : 0;   // (the newest block, pinned, is the last candidate)
    }
    T = tr0[NRM];
  }
  // (C1IDX_NO_V14: v14's union tiles and distributed selection compiled out: v13's paths and code, the same results)
#ifdef C1IDX_NO_V14
  constexpr bool v14 = false;
#else
  constexpr bool v14 = true;
#endif
#if defined(C1IDX_NO_V14) || defined(C1IDX_NO_DS)
  constexpr bool v14ds = false;   // (C1IDX_NO_DS: union tiles only, v13's one CTA per row for the selection)
#else
  constexpr bool v14ds = true;
#endif
  const bool umode = v14 && cmode && a.uni != 0 && ((nmax + BLK - 1) / BLK + 31) / 32 <= UNW;
  // v14 distributed selection: the rows / candidate rows past the thresholds (the same decision on every CTA)
  unsigned smask = 0u, bmask = 0u;
#pragma unroll
  for (int r = 0; r < NRM; ++r) {
    ne[r] = cmode ? nec[r] : n[r];
    if (v14ds && a.dsc != nullptr && a.dthr > 0 && r < L.NR) {
      if (ne[r] > max(a.dthr, TOPK) && ne[r] <= S1K * NCTA_IX * CTH) smask |= 1u << r;
      if (a.cand != nullptr && nbk[r] > CAND_K && Tdense <= 15 * 2 * NCTA_IX) bmask |= 1u << r;   // (u8 bins: <= 15 rounds)
    }
  }
  const int hso = smask ? so + XS_HE : -1, hbo = bmask ? so + XS_HBK : -1;
  if (umode) {   // (v14f) layers 24 .. 36 share layer 20's candidate rows: the CTA's map is built at the first consumer
                 // layer of the step (24) and cached in global memory with its index-layer epoch; 28 / 32 / 36 load it
    unsigned* uc = a.dsc + DS_UMC + c * UMC_W;
    const int li = (a.L - 24) >> 2;   // consumer layers since the first one of the step
    int* um = sptr<int>(so + XS_UM);
    // (icall is valid in thread 0 only (c1hooks_ix back): thread 0 decides, the CTA follows; a per-thread decision split
    // the CTA between the two paths' barriers and hung, i-11l / i-11m)
    int* hf = sptr<int>(so + SC_DS) + SD_S2 + 7;   // (free at entry: stage 1 writes the SC_DS words later)
    if (tid == 0) *hf = (a.dsc != nullptr && li > 0 && __ldcg(uc + UM_T * 8 * 5 + 1) == icall - (unsigned)li) ? 1 : 0;
    cbar();
    const bool hit = *hf != 0;
    if (hit) {
      for (int j = tid; j < UM_T * 8 * 5; j += CTH) um[j] = (int)__ldcg(uc + j);
      T = (int)__ldcg(uc + UM_T * 8 * 5);
      cbar();
    } else {
      T = ub_build(a, so, L.NR, (nmax + BLK - 1) / BLK, c, tid);
      if (a.dsc != nullptr) {
        for (int j = tid; j < UM_T * 8 * 5; j += CTH) uc[j] = (unsigned)um[j];
        if (tid == 0) uc[UM_T * 8 * 5] = (unsigned)T, uc[UM_T * 8 * 5 + 1] = icall - (unsigned)max(li, 0);
      }
    }
  }
  auto tile_row = [&](int t, int& j0) {   // compact mode: tile t -> (row, first candidate index)
    int r = 0;
#pragma unroll
    for (int k = 1; k < NRM; ++k) r += t >= tr0[k];
    j0 = 8 * (t - tr0[r]);
    return r;
  };
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(scr), mb = sbase + XS_MB;
  if (2 * c < T) {
    if (tid == 0) {   // the X4 mbarrier: phase 0 = the q staging, then one phase per key-tile round
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (tid < 4 * 16) {   // the rows' cos / sin (inputs of the launch) while the X1 / X3 counters fill
      const int r = tid >> 4, k = tid & 15;
      float4 v = make_float4(1.f, 1.f, 1.f, 1.f);
      if (r < L.NR) v = __ldg(reinterpret_cast<const float4*>(L.cos_sin + (size_t)__ldg(L.pos + r) * 64) + k);
      reinterpret_cast<float4*>(scr + XS_CS)[tid] = v;
    }
    if (bmask) ds_zero(so + XS_HBK, NRM * DBINS / 4, tid);   // (not touched by the q staging / quant)
    spin(IX_KS, 132u * icall);
    spin(IX_Q, 132u * icall);
    sstamp(st, 0, tid);
    x3b_stage(L, sbase, mb, tid);
    sstamp(st, 6, tid);
    x3b_quant(L, so, tid);
    sstamp(st, 1, tid);
    if (L.kind == 2) spin(IX_K, 132u * icall);
    sstamp(st, 2, tid);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the Q image (generic stores) -> wgmma
    cbar();
    if (smask) {   // (XS_HE = the q staging rows 2, 3: free once every thread is past the quant)
      ds_zero(hso, NRM * DBINS / 2, tid);
      cbar();
    }
    int round = 0;
    for (int t2 = 2 * c; t2 < T; t2 += 2 * 132, ++round) {
      if (w == 0) {   // one box (+ its scales) per lane of warp 0: the boxes' table lookups run in parallel
        asm volatile("fence.proxy.async.global;" ::: "memory");   // keys written by X2 (generic) -> TMA reads
        bool ok = false;
        int pg = 0, tk = 0;
        uint32_t dv = 0, dsc = 0, sb = 0;
        if (cmode && lane < 16) {   // 8 candidate blocks per tile: boxes of [8 keys x 128 B]
          const int g = lane >> 3, i = lane & 7, t = t2 + g;
          if (t < T) {
            int pb = -1;
            if (umode) {
              const int b = sptr<const int>(so + XS_UM)[((2 * round + g) * 8 + i) * 5];
              if (b >= 0) pb = 8 * b;
            } else {
              int j0;
              const int r = tile_row(t, j0);
              if (j0 + i < cnt[r]) pb = 8 * __ldcg(a.cblk + r * CAND_K + j0 + i);
            }
            if (pb >= 0) {
              pg = __ldg(bt + pb / P), tk = pb % P, ok = true;
              dv = sbase + XS_T0 + g * 8192 + i * 1024, dsc = sbase + XS_SC + g * 256 + i * 32, sb = 32;
            }
          }
        } else if (!cmode && lane < 2 * (TILE / P)) {   // whole pages of P keys
          const int g = lane / (TILE / P), k = lane % (TILE / P), t = t2 + g, p0 = TILE * t + P * k;
          if (t < T && p0 < nmax) {
            pg = __ldg(bt + p0 / P), ok = true;
            dv = sbase + XS_T0 + g * 8192 + k * P * HD, dsc = sbase + XS_SC + g * 256 + k * P * 4, sb = P * 4;
          }
        }
        const unsigned bytes = __reduce_add_sync(0xffffffffu, ok ? (cmode ? BLK * HD : P * HD) + sb : 0u);
        if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
        __syncwarp();
        if (ok) {
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dv),
              "l"(cmode ? a.tmk8 : a.tmk), "r"(0), "r"(tk), "r"(pg), "r"(mb)
              : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dsc),
                       "l"(a.kpages + (long long)pg * a.kstride + P * HD + tk * 4), "r"(sb), "r"(mb)
                       : "memory");
        }
      }
      asm volatile("{\n .reg .pred p;\n LAB_X4%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_X4%=;\n}" ::"r"(mb),
                   "r"((unsigned)((round + 1) & 1))
                   : "memory");
      if (t2 + wg < T && !cmode)
        logits_tile(a, sbase + XS_T0 + wg * 8192, so + XS_SC + wg * 256, sbase + XS_Q, so + XS_W, TILE * (t2 + wg), n, wq,
                    lane, -1, -1, -1, -1, hso, hbo);
      if (t2 + wg < T && umode)
        logits_tile(a, sbase + XS_T0 + wg * 8192, so + XS_SC + wg * 256, sbase + XS_Q, so + XS_W, 0, n, wq, lane, -1, -1, -1,
                    so + XS_UM + (2 * round + wg) * 8 * 5 * 4, hso, -1);
      if (t2 + wg < T && cmode && !umode) {
        int j0;
        const int r = tile_row(t2 + wg, j0), jb = j0 + 2 * wq;
        const int pb0 = jb < cnt[r] ? 8 * __ldcg(a.cblk + r * CAND_K + jb) : -1;
        const int pb1 = jb + 1 < cnt[r] ? 8 * __ldcg(a.cblk + r * CAND_K + jb + 1) : -1;
        logits_tile(a, sbase + XS_T0 + wg * 8192, so + XS_SC + wg * 256, sbase + XS_Q, so + XS_W, 8 * j0, n, wq, lane, r,
                    pb0, pb1, -1, hso, -1);
      }
      cbar();   // the tiles are read before the next round's copies
    }
    if (tid == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb) : "memory");
    if (smask | bmask) ds_flush(hso, hbo, a.dsc, smask, bmask, tid);   // (the loop's last barrier: every count is in)
    sstamp(st, 3, tid);
  }
  arr(IX_LG);
  const bool dsany = (smask | bmask) != 0u;
  if (dsany) {   // stage 1 on every CTA: the global histograms are complete at IX_LG
    spin(IX_LG, 132u * icall);
    ds_stage1(a, so, ne, nbk, smask, bmask, c, tid);
    sstamp(st, 7, tid);   // (tests: stage 1 done)
  }
  arr(IX_CL);   // (every index layer: the counter's epoch)
  const bool selc = c < NRM, candc = a.cand != nullptr && c >= NRM && c < 2 * NRM;
  if (selc || candc) {
    const int r = selc ? c : c - NRM;
    const bool dr = (((selc ? smask : bmask) >> r) & 1u) != 0u;
    if (dr) spin(IX_CL, 132u * icall);
    else if (!dsany) spin(IX_LG, 132u * icall);
    sstamp(st, 4, tid);
    const int* cbr = cmode ? a.cblk + r * CAND_K : nullptr;
    if (selc && dr) ds_select_row(a, r, r < L.NR, so, tid, cbr, nec[r], bt);
    else if (selc) select_row(a, r, r < L.NR, so, tid, cbr, nec[r], nullptr, bt);
    else if (dr) ds_cand_row(a, r, r < L.NR, so, tid);
    else cand_row(a, r, r < L.NR, so, tid);
    sstamp(st, 5, tid);
    if (selc && !dr && (L.skip & 8)) {   // tests (C1IDX_SEL2=1): the select once more, warm (same inputs / outputs): stamp 8 + 7
      cbar();
      select_row(a, r, r < L.NR, so, tid, cbr, nec[r], nullptr, bt);
      sstamp(st, 7, tid);
    }
    if (selc && L.rowsel != nullptr) {   // row c's top-k is final: the push warps' per-row flag (hook 6)
      cbar();
      if (tid == 0) {
        __threadfence();
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(L.rowsel + c), "r"(ncall) : "memory");
      }
    }
  }
  arr(IX_SEL);
  spin(IX_SEL, 132u * icall);
}
}  // namespace c1ix
