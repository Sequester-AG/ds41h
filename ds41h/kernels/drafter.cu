// ds41h drafter kernels (sm90a, TP=4 over NVLink IPC).
//
// draft_markov_kernel: the whole DSpark sequential Markov sampling loop (all n_spec draft steps) in ONE
// persistent launch, replacing per step {_draft_kernel, ll_all_gather, _draft_combine} (ds41h/vp_sampling.py,
// 3 launches x n_spec + an index_select).
//
// Vocab-parallel layout (as vp_sampling.py): rank r owns global token ids [vocab_start, vocab_start + local).
// CTA c owns local rows [c * chunk, c * chunk + chunk) and keeps its slice of markov_w2 ([chunk][R] bf16) in
// shared memory for all steps. Per step s, per request row r:
//   emb      = w1[prev_r]                                   (R = 256 bf16; prev = anchor token or step s-1 winner)
//   bias_v   = bf16_rn(sum_k emb[k] * w2[v][k])             (fp32 accumulation; stock: bf16 F.linear)
//   logit_v  = bf16_rn(base[r, s, v] + bias_v)               (stock speculator.py:184-185 rounding)
//   cache[req, s, v] = logit_v                               (pre-temperature draft logits for verification)
//   value_v  = logit_v / T + Gumbel(seed, pos + SALT, global id)   (stock gumbel_noised_argmax, bitwise:
//              div.full.f32, murmur3 uniform, libdevice-equivalent logf; T == 0 -> plain argmax)
//   stats    = max / sum-exp of logit_v / (T > 0 ? T : 1)
// Reduction: each CTA writes a 5-word candidate (value, token, logit, max, sum-exp) per row; the last CTA to
// arrive (atomic counter) merges the G candidates (lowest token wins ties) and pushes the rank's candidate to
// all 4 ranks (LL words: 32-bit payload | 32-bit tag in one 64-bit store). Every CTA polls the 4 rank words of
// each row in its own (local) receive buffer, picks the global winner (lowest rank = lowest token on ties), and
// continues with it. CTA 0 writes draft_tokens / token_logit / lse.
//
// Graph safety: no host arguments change between replays; the epoch lives in device memory (state[0]); the
// last CTA of the final step advances it. Slots are indexed by (epoch parity, step, src rank, row), so a rank
// that runs ahead into the next launch can never overwrite a slot that is still being read (it must first
// complete this launch, which needs every CTA's last-step candidate, i.e. every CTA past its reads of the
// earlier steps; the next launch uses the other parity). Tags are never 0 (zeroed buffers never match).
// Co-residency: all G CTAs of every rank must eventually run concurrently (G <= #SMs, 1 CTA/SM); waiting CTAs
// only depend on other CTAs of this launch.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>

#include "common.cuh"
#include "host.cuh"
#include "sm90.cuh"

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int R = 256;           // dspark_markov_rank
constexpr int MAXCHUNK = 256;    // local vocab rows per CTA (one per thread in the elementwise phase)
constexpr int MAXS = 8;          // draft steps per launch
constexpr int WORDS = 6;         // 64-bit LL words per (row, src) slot (5 used)
constexpr int MAXCTAS = 132;     // drafting slot stride over source CTAs
constexpr int W2PITCH = 528;     // smem row pitch of the markov_w2 slice (512 B + 16: conflict-free ldmatrix)
constexpr int DRAFT_ROWS = 4;    // rows per fused drafting launch (slot capacity)
constexpr long long SALT = 1ll << 30;  // gumbel.py _DRAFT_NOISE_SALT
constexpr int MAXGRP = 16;       // grouped exchange: CTA groups per rank (state counters [1 + s * MAXGRP + group])

struct DraftArgs {
  unsigned long long* recv[RANKS];  // receive buffer of each rank (peer-mapped), 64-bit words
  unsigned* state;                  // [0] epoch, [1 + s] per-step arrival counters
  float* cand;                      // [MAXS][max_rows][G][5] per-CTA candidates (local)
  const __nv_bfloat16* base;        // [rows * nspec, base_stride] local draft-head logits, row = r * nspec + s
  long base_stride;
  const __nv_bfloat16* w1;          // [vocab, R] replicated Markov embedding
  const __nv_bfloat16* w2;          // [local, R] this rank's markov_w2 slice
  const int* input_ids;             // anchor token of row r = input_ids[r * anchor_stride] (int32 InputBuffers)
  long anchor_stride;
  const long long* idx_map;         // [rows * nspec] request-state index (-1 = padding row)
  const long long* sample_pos;      // [rows * nspec] predicted positions (draws keyed by pos - 1)
  const float* temp;                // [max_reqs]
  const long long* seeds;           // [max_reqs]
  long long* draft_tokens;          // [>= rows, dt_stride]
  long dt_stride;
  long long* req_draft;             // optional [max_reqs, req_stride]: the runner's per-request draft tokens (req_states.draft_tokens),
  long req_stride;                  // written with the same tokens at the request-state index (early combine), or null
  __nv_bfloat16* cache;             // [max_reqs, cache_s0 / cache_s1 ...] draft logit cache
  long cache_s0, cache_s1;
  float* token_logit;               // [max_reqs + 1, vp_stride]
  float* lse;                       // [max_reqs + 1, vp_stride]
  long vp_stride;
  long sink;                        // row index of the sink slot (max_reqs)
  long slot_rows;                   // rows capacity of the receive slots / candidate buffer
  unsigned long long timeout;       // ns, 0 = spin forever
  unsigned long long* trace;        // optional %globaltimer trace (CTA 0: [0, 64), last-CTA merge windows [64, 96))
  int rank, rows, nspec, local, vocab_start, chunk, vocab;
  int groups;                       // 0: rank-level last-CTA merge (original); NG > 0: NG CTA groups push to the ranks directly
};

// ------------------------------------------------------------------ exact replicas of the Triton sampling math
__device__ __forceinline__ float div_full(float a, float b) {  // Triton fp32 '/' lowers to div.full.f32
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ unsigned rotl32(unsigned v, int s) { return (v << s) | (v >> (32 - s)); }
__device__ __forceinline__ unsigned mm3_mix(unsigned h, unsigned key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ unsigned mm3_fmix(unsigned h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
// gumbel.py murmur3_uniform32(seed, pos, offset)
__device__ __forceinline__ float mm3_uniform(long long seed, long long pos, unsigned offset) {
  unsigned h = 0u;
  h = mm3_mix(h, unsigned(seed & 0xFFFFFFFFll));
  h = mm3_mix(h, unsigned((seed >> 32) & 0xFFFFFFFFll));
  h = mm3_mix(h, unsigned(pos & 0xFFFFFFFFll));
  h = mm3_mix(h, offset);
  const unsigned r = mm3_fmix(h ^ 16u);
  const float hi = float(r >> 16), lo = float(r & 0xFFFFu);
  return __fmaf_rn(hi, 1.52587890625e-05f, __fmul_rn(__fadd_rn(lo, 0.5f), 2.3283064365386963e-10f));
}
// gumbel.py _log1p_neg_stable(u) = log(1 - u)
__device__ __forceinline__ float log1p_neg_stable(float u) {
  float p = __fmaf_rn(u, 0.125f, __uint_as_float(0x3E124925u));  // 1/7
  p = __fmaf_rn(u, p, __uint_as_float(0x3E2AAAABu));             // 1/6
  p = __fmaf_rn(u, p, __uint_as_float(0x3E4CCCCDu));             // 1/5
  p = __fmaf_rn(u, p, 0.25f);
  p = __fmaf_rn(u, p, __uint_as_float(0x3EAAAAABu));             // 1/3
  p = __fmaf_rn(u, p, 0.5f);
  p = __fmaf_rn(u, p, 1.0f);
  const float series = __fmul_rn(__fsub_rn(0.0f, u), p);
  const float direct = logf(fmaxf(__fsub_rn(1.0f, u), 5.960464477539063e-08f));
  return u < 0.25f ? series : direct;
}
// value of a (pre-temperature) logit under Gumbel-max at temperature t != 0 (gumbel_noised_argmax)
__device__ __forceinline__ float gumbel_noise(long long seed, long long pos, unsigned key) {
  const float u = mm3_uniform(seed, pos, key);
  return __fsub_rn(0.0f, logf(__fsub_rn(0.0f, log1p_neg_stable(u))));
}
// verification residual noise (APPLY_TEMPERATURE=False, IS_DRAFTING=False): same formula, pos without the draft salt
__device__ __forceinline__ float gumbel_noise_raw(long long seed, long long pos, unsigned key) { return gumbel_noise(seed, pos, key); }
__device__ __forceinline__ float gumbel_value(float logit, float t, long long seed, long long pos, unsigned key) {
  return __fadd_rn(div_full(logit, t), gumbel_noise(seed, pos, key));
}

// (value, token) order: larger value wins, lower token on ties; NaN never beats a number.
__device__ __forceinline__ bool better(float v, int tok, float bv, int btok) { return v > bv || (v == bv && tok < btok); }

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void st_word(unsigned long long* p, unsigned long long v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_word(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
// Poll one LL word until its tag matches. After a short hot spin the poller backs off with __nanosleep (a waiting
// rank burns less power under the cap; wake-up latency <= ~0.1 us). site: 0 draft_markov, 1 moe2 all-reduce, 2 verify.
__device__ __forceinline__ unsigned poll_word(const unsigned long long* p, unsigned tag, unsigned long long timeout, int site = 0) {
  unsigned long long v = ld_word(p);
  if (unsigned(v >> 32) != tag) {
    const unsigned long long t0 = timeout ? globaltimer() : 0ull;
    unsigned spins = 0;
    do {
      if (++spins > 32u) __nanosleep(64);
      v = ld_word(p);
      if (timeout && (spins & 255u) == 0 && globaltimer() - t0 > timeout) {
        printf("ds41h %s: timeout waiting for tag %u at %p (saw %u)\n", site == 1 ? "moe2-allreduce" : site == 2 ? "verify" : "draft_markov",
               tag, p, unsigned(v >> 32));
        __trap();
      }
    } while (unsigned(v >> 32) != tag);
  }
  return unsigned(v);
}
__device__ __forceinline__ unsigned long long* recv_of(const DraftArgs& a, int p) {
  unsigned long long* r = a.recv[0];
#pragma unroll
  for (int q = 1; q < RANKS; ++q) r = p == q ? a.recv[q] : r;
  return r;
}
// word offset of slot (par, step, src, row)
__device__ __forceinline__ long slot_off(const DraftArgs& a, int par, int s, int src, int r) {
  return ((((long(par) * MAXS + s) * RANKS + src) * a.slot_rows + r) * WORDS);
}
// drafting slots: (par, step, src rank, src CTA, row) -- every CTA pushes its own candidate to every rank
__device__ __forceinline__ long dslot(const DraftArgs& a, int par, int s, int src, int cta, int r) {
  return (((((long(par) * MAXS + s) * RANKS + src) * MAXCTAS + cta) * DRAFT_ROWS + r) * WORDS);
}

struct Cand {
  float val, logit, m, s;
  int tok;
};

// Merge (m, s) pairs: total = sum s_i * exp(m_i - top) over finite m_i.
__device__ __forceinline__ void merge_ms(float& m, float& s, float m2, float s2) {
  if (!(m2 > -INFINITY)) return;
  if (!(m > -INFINITY)) {
    m = m2, s = s2;
    return;
  }
  if (m2 > m) {
    s = s * expf(m - m2) + s2;
    m = m2;
  } else {
    s = s + s2 * expf(m2 - m);
  }
}

// Block-wide reduction of per-thread candidates (THREADS threads). Result valid in all threads.
__device__ Cand block_reduce(Cand c, Cand* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    Cand d;
    d.val = __shfl_xor_sync(0xffffffffu, c.val, o);
    d.tok = __shfl_xor_sync(0xffffffffu, c.tok, o);
    d.logit = __shfl_xor_sync(0xffffffffu, c.logit, o);
    d.m = __shfl_xor_sync(0xffffffffu, c.m, o);
    d.s = __shfl_xor_sync(0xffffffffu, c.s, o);
    if (better(d.val, d.tok, c.val, c.tok)) c.val = d.val, c.tok = d.tok, c.logit = d.logit;
    merge_ms(c.m, c.s, d.m, d.s);
  }
  __syncthreads();  // red reuse
  if (lane == 0) red[warp] = c;
  __syncthreads();
  c = red[0];
  for (int w = 1; w < WARPS; ++w) {
    const Cand d = red[w];
    if (better(d.val, d.tok, c.val, c.tok)) c.val = d.val, c.tok = d.tok, c.logit = d.logit;
    merge_ms(c.m, c.s, d.m, d.s);
  }
  return c;
}

// Block reduction of one-element candidates (thread: val/tok/logit of its token, m = logit / T, s = 1 if finite):
// pass 1 argmax (val, lowest tok) + max m, pass 2 sum exp(m_i - M). One expf per thread instead of one per merge
// level (the per-step critical path runs at the power-capped SM clock). Result valid in all threads.
__device__ Cand block_reduce1(Cand c, Cand* red, float* red2) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float m = c.m;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float dv = __shfl_xor_sync(0xffffffffu, c.val, o);
    const int dt = __shfl_xor_sync(0xffffffffu, c.tok, o);
    const float dl = __shfl_xor_sync(0xffffffffu, c.logit, o);
    if (better(dv, dt, c.val, c.tok)) c.val = dv, c.tok = dt, c.logit = dl;
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  }
  __syncthreads();  // red reuse
  if (lane == 0) {  // c.m stays this thread's own element (pass 2 below)
    Cand w = c;
    w.m = m;
    red[warp] = w;
  }
  __syncthreads();
  Cand r = red[0];
#pragma unroll
  for (int w = 1; w < WARPS; ++w) {
    const Cand d = red[w];
    if (better(d.val, d.tok, r.val, r.tok)) r.val = d.val, r.tok = d.tok, r.logit = d.logit;
    r.m = fmaxf(r.m, d.m);
  }
  float e = c.s > 0.0f && r.m > -INFINITY ? expf(c.m - r.m) : 0.0f;  // this thread's own element
  e = warp_sum(e);
  if (lane == 0) red2[warp] = e;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) tot += red2[w];
  r.s = tot;
  return r;
}

// Warp-wide merge of CTA candidates (result in every lane): argmax (lowest token on ties) and the (m, s) sum-exp pair
// in two passes (max, then one expf per lane and a sum) instead of an expf per butterfly level.
__device__ __forceinline__ void warp_merge_cand(Cand& c) {
  float top = c.m;
#pragma unroll
  for (int ofs = 16; ofs > 0; ofs >>= 1) {
    const float dv = __shfl_xor_sync(0xffffffffu, c.val, ofs);
    const int dt = __shfl_xor_sync(0xffffffffu, c.tok, ofs);
    const float dl = __shfl_xor_sync(0xffffffffu, c.logit, ofs);
    if (better(dv, dt, c.val, c.tok)) c.val = dv, c.tok = dt, c.logit = dl;
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, ofs));
  }
  float e = c.m > -INFINITY ? c.s * expf(c.m - top) : 0.0f;
#pragma unroll
  for (int ofs = 16; ofs > 0; ofs >>= 1) e += __shfl_xor_sync(0xffffffffu, e, ofs);
  c.m = top;
  c.s = top > -INFINITY ? e : 0.0f;
}

__global__ void __launch_bounds__(THREADS, 1) draft_markov_kernel(const DraftArgs a) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* w2b = smem;                                           // [MAXCHUNK rows][W2PITCH] bf16 rows
  float* acc = reinterpret_cast<float*>(smem + MAXCHUNK * W2PITCH);    // [MAXCHUNK]
  uint4* bases = reinterpret_cast<uint4*>(smem + MAXCHUNK * W2PITCH + MAXCHUNK * 4);  // [rows * nspec][MAXCHUNK] bf16
  __shared__ __align__(16) __nv_bfloat16 embh[R];
  __shared__ float red2[WARPS];
  __shared__ Cand red[WARPS];
  __shared__ int prev[64];
  __shared__ int last_flag;
  __shared__ Cand mred[DRAFT_ROWS][WARPS];  // last-CTA merge: per-warp partials
  // per (row, step) sampling state, read once (temperature / seeds may be host-mapped: no per-step PCIe round trips)
  __shared__ long long dm_req[64], dm_seed[64], dm_pos[64];
  __shared__ float dm_t[64];
  // Gumbel noise of this thread's token for the next step (depends only on (seed, pos, token): computed while the
  // previous step's candidates are in flight, off the per-step dependency chain). Column tid is private to thread tid.
  __shared__ float gnz[DRAFT_ROWS][MAXCHUNK];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cta = blockIdx.x, G = gridDim.x;
  const int v0 = cta * a.chunk;
  const int nv = min(a.chunk, a.local - v0);
  auto fill_noise = [&](int s) {
    if (tid < nv) {
      const unsigned tok = unsigned(a.vocab_start + v0 + tid);
      for (int r = 0; r < a.rows; ++r) {
        const int rs = r * a.nspec + s;
        if (dm_t[rs] != 0.0f) gnz[r][tid] = gumbel_noise(dm_seed[rs], dm_pos[rs], tok);
      }
    }
  };

  // markov_w2 slice -> smem, row-major with a 528-byte row pitch (conflict-free ldmatrix), coalesced cp.async (constant
  // weights: may run before the upstream kernel finished under PDL). Rows past nv are zeroed.
  auto trace = [&](int slot) {
    if (a.trace != nullptr && tid == 0 && cta == 0) a.trace[slot] = globaltimer();
  };
  trace(0);
  {
    const uint4* src = reinterpret_cast<const uint4*>(a.w2 + long(v0) * R);
    const int n16 = nv * (R / 8);
    for (int i = tid; i < n16; i += THREADS) cp_async16(w2b + (i >> 5) * W2PITCH + (i & 31) * 16, src + i);
    for (int i = n16 + tid; i < MAXCHUNK * (R / 8); i += THREADS)
      *reinterpret_cast<uint4*>(w2b + (i >> 5) * W2PITCH + (i & 31) * 16) = make_uint4(0u, 0u, 0u, 0u);
    cp_async_commit();
  }
  pdl_wait();
  trace(1);
  const unsigned epoch = ld_volatile(a.state);
  const int par = int(epoch & 1u);
  for (int r = tid; r < a.rows; r += THREADS) prev[r] = a.input_ids[long(r) * a.anchor_stride];
  for (int rs = tid; rs < a.rows * a.nspec; rs += THREADS) {
    const long long req = a.idx_map[rs];
    dm_req[rs] = req;
    dm_t[rs] = req >= 0 ? a.temp[req] : 0.0f;
    dm_seed[rs] = req >= 0 ? a.seeds[req] : 0ll;
    dm_pos[rs] = a.sample_pos[rs] - 1 + SALT;
  }
  // this CTA's slice of the draft-head logits for every (row, step) -> smem (one round trip instead of one per step)
  {
    const int per = a.chunk / 8;  // 16-byte pieces per (row, step)
    const int n = a.rows * a.nspec * per;
    for (int i = tid; i < n; i += THREADS) {
      const int rs = i / per, c = i % per;
      if (8 * c < nv) cp_async16(bases + (long(rs) * MAXCHUNK + 8 * c) / 8, a.base + long(rs) * a.base_stride + v0 + 8 * c);
    }
    cp_async_commit();
  }
  cp_async_wait_all();
  __syncthreads();
  fill_noise(0);
  trace(2);

  for (int s = 0; s < a.nspec; ++s) {
    const unsigned tag = ((epoch << 4) | unsigned(s + 1));
    for (int r = 0; r < a.rows; ++r) {
      // ---- Markov embedding of the previous token (bf16 row, 512 B)
      {
        const long tok = min(max(prev[r], 0), a.vocab - 1);  // padding rows may hold garbage anchors
        if (tid < R / 8) reinterpret_cast<uint4*>(embh)[tid] = __ldg(reinterpret_cast<const uint4*>(a.w1 + tok * R) + tid);
      }
      __syncthreads();
      // ---- bias = w2 . emb on the tensor cores: warp w owns rows [32w, 32w + 32) (2 m16 tiles) x K = 256 (16 k16 steps);
      // B = emb in column 0 of the n8 operand (lanes g == 0), fp32 accumulation of exact bf16 products
      {
        const int g = lane >> 2, t4 = lane & 3;
        float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lk = (lane >> 4) * 8;
        const unsigned char* a0p = w2b + (warp * 32 + lrow) * W2PITCH + lk * 2;
        const unsigned char* a1p = a0p + 16 * W2PITCH;
#pragma unroll
        for (int ks = 0; ks < R / 16; ++ks) {
          unsigned x0, x1, x2, x3, y0, y1, y2, y3;
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                       : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(smem_u32(a0p + ks * 32)));
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                       : "=r"(y0), "=r"(y1), "=r"(y2), "=r"(y3) : "r"(smem_u32(a1p + ks * 32)));
          unsigned b0 = 0u, b1 = 0u;
          if (g == 0) {
            b0 = *reinterpret_cast<const unsigned*>(embh + ks * 16 + 2 * t4);
            b1 = *reinterpret_cast<const unsigned*>(embh + ks * 16 + 2 * t4 + 8);
          }
          hmma_bf16(c0, x0, x1, x2, x3, b0, b1);
          hmma_bf16(c1, y0, y1, y2, y3, b0, b1);
        }
        if (t4 == 0) {  // column 0 = (row g, token 0) in c[0], (row g + 8) in c[2]
          acc[warp * 32 + g] = c0[0];
          acc[warp * 32 + g + 8] = c0[2];
          acc[warp * 32 + 16 + g] = c1[0];
          acc[warp * 32 + 16 + g + 8] = c1[2];
        }
      }
      __syncthreads();
      // ---- elementwise: thread tid owns local row v0 + tid
      const long rs = long(r) * a.nspec + s;
      const long long req = dm_req[rs];
      const bool valid = req >= 0;
      const float t = dm_t[rs];
      Cand c{-INFINITY, 0.0f, -INFINITY, 0.0f, 0x7fffffff};
      if (tid < nv) {
        const int v = v0 + tid;
        const float bias = bf16_round(acc[tid]);
        const float base = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(bases)[rs * MAXCHUNK + tid]);
        const float logit = bf16_round(__fadd_rn(base, bias));
        if (valid) a.cache[req * a.cache_s0 + long(s) * a.cache_s1 + v] = __float2bfloat16_rn(logit);
        const int tok = a.vocab_start + v;
        c.val = t != 0.0f ? __fadd_rn(div_full(logit, t), gnz[r][tid]) : logit;  // = gumbel_value(logit, t, ...)
        if (c.val != c.val) c.val = -INFINITY;  // NaN never wins
        c.tok = tok;
        c.logit = logit;
        c.m = div_full(logit, t > 0.0f ? t : 1.0f);
        c.s = c.m > -INFINITY ? 1.0f : 0.0f;
      }
      c = block_reduce1(c, red, red2);
      // the global winner is one of the CTA candidates: warm L2 with this candidate's Markov embedding row so the
      // next step's emb fetch is an L2 hit whichever candidate wins
      if (tid >= 32 && tid < 36 && c.tok >= 0 && c.tok < a.vocab) {
        const char* line = reinterpret_cast<const char*>(a.w1 + long(c.tok) * R) + 128 * (tid - 32);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
      }
      if (tid == 0) {
        float* o = a.cand + ((long(s) * a.slot_rows + r) * G + cta) * 5;
        o[0] = c.val;
        o[1] = __int_as_float(c.tok);
        o[2] = c.logit;
        o[3] = c.m;
        o[4] = c.s;
      }
    }
    // ---- arrival; the last CTA of this rank merges the G candidates and pushes the rank candidate to every rank.
    // (Measured: every CTA pushing to every rank and merging 4 x G is slower: L2 polling.)
    // Grouped (a.groups = NG > 0): CTAs form NG groups of gsz; the last CTA of each GROUP merges its <= gsz candidates and
    // pushes the group candidate to every rank (slots dslot(par, s, rank, group, r)); every CTA then merges 4 x NG.
    trace(3 + 4 * s);
    const int NG = a.groups;
    const int gsz = NG > 0 ? (G + NG - 1) / NG : G;
    const int grp = NG > 0 ? cta / gsz : 0;
    const int gbeg = grp * gsz, gend = min(G, gbeg + gsz);
    if (tid == 0) {
      __threadfence();
      unsigned* ctr = NG > 0 ? a.state + 1 + s * MAXGRP + grp : a.state + 1 + s;
      const bool last = atomicAdd(ctr, 1u) == unsigned((NG > 0 ? gend - gbeg : G) - 1);
      if (last) __threadfence();  // acquire side: the candidates of every CTA are visible to this CTA's loads below
      last_flag = last;
    }
    __syncthreads();
    trace(4 + 4 * s);
    if (last_flag && NG > 0) {
      // group merge: warp 0, lane i < members loads member gbeg + i (one L2 round trip), warp merge, push 5 words x 4 ranks
      if (warp == 0) {
        for (int r = 0; r < a.rows; ++r) {
          Cand c{-INFINITY, 0.0f, -INFINITY, 0.0f, 0x7fffffff};
          if (lane < gend - gbeg) {
            const float* q = a.cand + ((long(s) * a.slot_rows + r) * G + gbeg + lane) * 5;
            c.val = __ldcg(q);
            c.tok = __float_as_int(__ldcg(q + 1));
            c.logit = __ldcg(q + 2);
            c.m = __ldcg(q + 3);
            c.s = __ldcg(q + 4);
          }
          warp_merge_cand(c);
          if (lane < RANKS * 5) {
            const int p = lane / 5, w = lane % 5;
            const float x = w == 0 ? c.val : w == 1 ? __int_as_float(c.tok) : w == 2 ? c.logit : w == 3 ? c.m : c.s;
            st_word(recv_of(a, p) + dslot(a, par, s, a.rank, grp, r) + w, (static_cast<unsigned long long>(tag) << 32) | __float_as_uint(x));
          }
        }
        if (lane == 0) a.state[1 + s * MAXGRP + grp] = 0u;
      }
    } else if (last_flag) {
      // every thread loads one CTA candidate per row (G <= THREADS: one L2 round trip, the writers fenced before their
      // arrival atomics), warp-level merge, then warp 0 merges the WARPS partials and pushes. (Was: warp 0 alone
      // looping over G candidates -- 4.3 us cold / ~17 us at the power-capped clock per step.)
      for (int r = 0; r < a.rows; ++r) {
        Cand c{-INFINITY, 0.0f, -INFINITY, 0.0f, 0x7fffffff};
        if (tid < G) {
          const float* q = a.cand + ((long(s) * a.slot_rows + r) * G + tid) * 5;
          c.val = __ldcg(q);
          c.tok = __float_as_int(__ldcg(q + 1));
          c.logit = __ldcg(q + 2);
          c.m = __ldcg(q + 3);
          c.s = __ldcg(q + 4);
        }
        warp_merge_cand(c);
        if (lane == 0) mred[r][warp] = c;
      }
      __syncthreads();  // last_flag is block-uniform
      if (warp == 0) {
        for (int r = 0; r < a.rows; ++r) {
          Cand c{-INFINITY, 0.0f, -INFINITY, 0.0f, 0x7fffffff};
          if (lane < WARPS) c = mred[r][lane];
          warp_merge_cand(c);
          if (lane < RANKS * 5) {
            const int p = lane / 5, w = lane % 5;
            const float x = w == 0 ? c.val : w == 1 ? __int_as_float(c.tok) : w == 2 ? c.logit : w == 3 ? c.m : c.s;
            st_word(recv_of(a, p) + slot_off(a, par, s, a.rank, r) + w, (static_cast<unsigned long long>(tag) << 32) | __float_as_uint(x));
          }
        }
        if (lane == 0) a.state[1 + s] = 0u;  // counter reused by the next launch only (stream order)
      }
    }
    if (s + 1 < a.nspec) fill_noise(s + 1);  // overlaps the cross-rank exchange
    trace(5 + 4 * s);
    // ---- every CTA: global winner per row from the 4 rank candidates (lowest rank = lowest token on ties)
    const unsigned long long* box = recv_of(a, a.rank);
    if (NG > 0) {
      // 4 x NG group candidates per row: lane i < 2 * 32 / ... -> each lane polls up to 2 slots (64 slots max), warp merge;
      // merge order is a deterministic function of (rank, group) values: the argmax ties resolve by token, the sum-exp
      // pair merge is the same two-pass reduction as warp_merge_cand
      const int nslots = RANKS * NG;
      for (int r = warp; r < a.rows; r += WARPS) {
        Cand c{-INFINITY, 0.0f, -INFINITY, 0.0f, 0x7fffffff};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = lane + 32 * h;
          if (i < nslots) {
            const int src = i / NG, gi = i % NG;
            if (gi * gsz < G) {  // groups past the last CTA never push
              const unsigned long long* q = box + dslot(a, par, s, src, gi, r);
              Cand d;
              d.val = __uint_as_float(poll_word(q, tag, a.timeout));
              d.tok = int(poll_word(q + 1, tag, a.timeout));
              d.logit = __uint_as_float(poll_word(q + 2, tag, a.timeout));
              d.m = __uint_as_float(poll_word(q + 3, tag, a.timeout));
              d.s = __uint_as_float(poll_word(q + 4, tag, a.timeout));
              if (h == 0) {
                c = d;
              } else {
                if (better(d.val, d.tok, c.val, c.tok)) c.val = d.val, c.tok = d.tok, c.logit = d.logit;
                merge_ms(c.m, c.s, d.m, d.s);
              }
            }
          }
        }
        warp_merge_cand(c);
        if (lane == 0) {
          prev[r] = c.tok;
          if (cta == 0) {
            const long rs = long(r) * a.nspec + s;
            const long long req = dm_req[rs];
            const long slot = req >= 0 ? long(req) : a.sink;
            a.draft_tokens[long(r) * a.dt_stride + s] = c.tok;
            if (a.req_draft != nullptr && req >= 0) a.req_draft[long(req) * a.req_stride + s] = c.tok;
            a.token_logit[slot * a.vp_stride + s] = c.logit;
            a.lse[slot * a.vp_stride + s] = c.m + logf(c.s);
          }
        }
      }
    }
    for (int r = warp; NG == 0 && r < a.rows; r += WARPS) {
      float x = 0.0f;
      if (lane < RANKS * 5) x = __uint_as_float(poll_word(box + slot_off(a, par, s, lane / 5, r) + lane % 5, tag, a.timeout));
      float val[RANKS], logit[RANKS], m[RANKS], sm[RANKS];
      int tok[RANKS];
#pragma unroll
      for (int p = 0; p < RANKS; ++p) {
        val[p] = __shfl_sync(0xffffffffu, x, 5 * p);
        tok[p] = __float_as_int(__shfl_sync(0xffffffffu, x, 5 * p + 1));
        logit[p] = __shfl_sync(0xffffffffu, x, 5 * p + 2);
        m[p] = __shfl_sync(0xffffffffu, x, 5 * p + 3);
        sm[p] = __shfl_sync(0xffffffffu, x, 5 * p + 4);
      }
      int w = 0;
#pragma unroll
      for (int p = 1; p < RANKS; ++p)
        if (better(val[p], tok[p], val[w], tok[w])) w = p;
      float top = m[0];
#pragma unroll
      for (int p = 1; p < RANKS; ++p) top = fmaxf(top, m[p]);
      float total = 0.0f;
#pragma unroll
      for (int p = 0; p < RANKS; ++p) total += m[p] > -INFINITY ? sm[p] * expf(m[p] - top) : 0.0f;
      int win_tok = tok[0];
      float win_logit = logit[0];
#pragma unroll
      for (int p = 1; p < RANKS; ++p) win_tok = w == p ? tok[p] : win_tok, win_logit = w == p ? logit[p] : win_logit;
      if (lane == 0) {
        prev[r] = win_tok;
        if (cta == 0) {
          const long rs = long(r) * a.nspec + s;
          const long long req = dm_req[rs];
          const long slot = req >= 0 ? long(req) : a.sink;
          a.draft_tokens[long(r) * a.dt_stride + s] = win_tok;
          if (a.req_draft != nullptr && req >= 0) a.req_draft[long(req) * a.req_stride + s] = win_tok;
          a.token_logit[slot * a.vp_stride + s] = win_logit;
          a.lse[slot * a.vp_stride + s] = top + logf(total);
        }
      }
    }
    if (s == a.nspec - 1 && cta == 0 && tid == 0) {
      // CTA 0 saw the last step's candidates of every rank, i.e. every CTA of every rank contributed (and so had read
      // the epoch before step 0): advance it once for the next launch
      atomicExch(a.state, epoch + 1u == 0u ? 1u : epoch + 1u);
    }
    __syncthreads();
    trace(6 + 4 * s);
  }
  pdl_launch_dependents();
}


// ================================================================== shared x staging for the decode GEMVs
__device__ __forceinline__ float amax8(uint4 v) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = unpack_bf16(u[i]);
    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  return m;
}
// Stage x [m <= 8, K] bf16 (row stride ldx, 16-byte aligned rows) as f16 m16n8k16 B fragments xs[kb * 32 + lane]
// (lane 4g + t: token g, k = 32 kb + 8t .. + 8; P order {0,2,4,6,1,3,5,7} when PORDER, else natural) with one
// power-of-two scale per token in xsc[t] (max |x_t| * 2^-e < 2^15; xsc = 2^e). Every pass issues all of its global
// loads before using them (latency-bound otherwise). Ends with __syncthreads().
template <int K, int NT, bool PORDER>
__device__ __forceinline__ void stage_x(const __nv_bfloat16* x, long ldx, int m, uint4* xs, float* xsc) {
  constexpr int KB = K / 32, V = K / 8, NW = NT / 32, PER = (V + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int tok = warp; tok < 8; tok += NW) {
    float mx = 0.0f;
    if (tok < m) {
      const uint4* p = reinterpret_cast<const uint4*>(x + long(tok) * ldx);
      uint4 v[PER];
#pragma unroll
      for (int i = 0; i < PER; ++i) v[i] = lane + 32 * i < V ? __ldg(p + lane + 32 * i) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int i = 0; i < PER; ++i) mx = fmaxf(mx, amax8(v[i]));
    }
    mx = warp_max(mx);
    if (lane == 0) xsc[tok] = ldexpf(1.0f, mx > 0.0f ? (int((__float_as_uint(mx) >> 23) & 0xff) - 127) - 14 : 0);
  }
  __syncthreads();
  constexpr int E = KB * 32;
  static_assert(E % NT == 0, "staging entries must divide over the block");
  constexpr int PER2 = E / NT, BATCH = PER2 < 10 ? PER2 : 10;
  static_assert(PER2 % BATCH == 0, "batching");
#pragma unroll 1
  for (int b0 = 0; b0 < PER2; b0 += BATCH) {
    uint4 raw[BATCH];
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
      const int i = tid + (b0 + j) * NT, kb = i >> 5, l = i & 31, gg = l >> 2, tt = l & 3;
      raw[j] = gg < m ? __ldg(reinterpret_cast<const uint4*>(x + long(gg) * ldx + kb * 32 + 8 * tt)) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
      const int i = tid + (b0 + j) * NT, gg = (i & 31) >> 2;
      const float inv = gg < m ? 1.0f / xsc[gg] : 0.0f;
      const float2 f01 = unpack_bf16(raw[j].x), f23 = unpack_bf16(raw[j].y), f45 = unpack_bf16(raw[j].z), f67 = unpack_bf16(raw[j].w);
      uint4 v;
      if (PORDER) {  // P = {0, 4, 1, 5, 2, 6, 3, 7}
        v = make_uint4(pack_f16(f01.x * inv, f45.x * inv), pack_f16(f01.y * inv, f45.y * inv), pack_f16(f23.x * inv, f67.x * inv),
                       pack_f16(f23.y * inv, f67.y * inv));
      } else {
        v = make_uint4(pack_f16(f01.x * inv, f01.y * inv), pack_f16(f23.x * inv, f23.y * inv), pack_f16(f45.x * inv, f45.y * inv),
                       pack_f16(f67.x * inv, f67.y * inv));
      }
      xs[i] = v;
    }
  }
  __syncthreads();
}

// stage_x with the RMSNorm fused in front (natural k order): y = bf16(bf16(x * r) * w), r = rsqrt(mean(x^2) + eps), the
// vLLM rms_norm kernel's rounding sequence; then the same per-token power-of-two f16 staging as stage_x.
template <int K, int NT>
__device__ __forceinline__ void stage_x_norm(const __nv_bfloat16* x, long ldx, int m, const __nv_bfloat16* nw, float eps, uint4* xs, float* xsc,
                                             float* rs) {
  constexpr int KB = K / 32, V = K / 8, NW = NT / 32, PER = (V + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  auto bfr = [](float v) { return __bfloat162float(__float2bfloat16_rn(v)); };
  for (int tok = warp; tok < 8; tok += NW) {
    float mx = 0.0f, r = 0.0f;
    if (tok < m) {
      // two passes over the (L1/L2-resident) row instead of holding it in registers: the head kernel keeps its first
      // weight chunk in flight across this staging
      const uint4* p = reinterpret_cast<const uint4*>(x + long(tok) * ldx);
      float ss = 0.0f;
#pragma unroll 4
      for (int i = 0; i < PER; ++i) {
        if (lane + 32 * i < V) {
          const uint4 v = __ldg(p + lane + 32 * i);
          const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = unpack_bf16(u[q]);
            ss = __fmaf_rn(f.x, f.x, ss);
            ss = __fmaf_rn(f.y, f.y, ss);
          }
        }
      }
      ss = warp_sum(ss);
      r = rsqrtf(ss / float(K) + eps);
      const uint4* wp = reinterpret_cast<const uint4*>(nw);
#pragma unroll 4
      for (int i = 0; i < PER; ++i) {
        if (lane + 32 * i < V) {
          const uint4 v = __ldg(p + lane + 32 * i), wv = __ldg(wp + lane + 32 * i);
          const unsigned u[4] = {v.x, v.y, v.z, v.w}, w4[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = unpack_bf16(u[q]), wf = unpack_bf16(w4[q]);
            mx = fmaxf(mx, fabsf(bfr(bfr(f.x * r) * wf.x)));
            mx = fmaxf(mx, fabsf(bfr(bfr(f.y * r) * wf.y)));
          }
        }
      }
    }
    mx = warp_max(mx);
    if (lane == 0) {
      xsc[tok] = ldexpf(1.0f, mx > 0.0f ? (int((__float_as_uint(mx) >> 23) & 0xff) - 127) - 14 : 0);
      rs[tok] = r;
    }
  }
  __syncthreads();
  constexpr int E = KB * 32;
  static_assert(E % NT == 0, "staging entries must divide over the block");
  constexpr int PER2 = E / NT, BATCH = PER2 < 5 ? PER2 : 5;
  static_assert(PER2 % BATCH == 0, "batching");
#pragma unroll 1
  for (int b0 = 0; b0 < PER2; b0 += BATCH) {
    uint4 raw[BATCH], wr[BATCH];
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
      const int i = tid + (b0 + j) * NT, kb = i >> 5, l = i & 31, gg = l >> 2, tt = l & 3;
      raw[j] = gg < m ? __ldg(reinterpret_cast<const uint4*>(x + long(gg) * ldx + kb * 32 + 8 * tt)) : make_uint4(0u, 0u, 0u, 0u);
      wr[j] = __ldg(reinterpret_cast<const uint4*>(nw + kb * 32 + 8 * tt));
    }
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
      const int i = tid + (b0 + j) * NT, gg = (i & 31) >> 2;
      const float inv = gg < m ? 1.0f / xsc[gg] : 0.0f, r = gg < m ? rs[gg] : 0.0f;
      const unsigned u[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w}, w4[4] = {wr[j].x, wr[j].y, wr[j].z, wr[j].w};
      unsigned o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = unpack_bf16(u[q]), wf = unpack_bf16(w4[q]);
        o[q] = pack_f16(bfr(bfr(f.x * r) * wf.x) * inv, bfr(bfr(f.y * r) * wf.y) * inv);
      }
      xs[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  __syncthreads();
}

// ================================================================== routed-expert decode GEMMs (drafter MoE)
// y[t] = sum_j wts[t, j] * W2_e(clampSwiGLU(W1_e x_t, W3_e x_t)),  e = ids[t, j],  t < M <= 8, raw TP-sliced layout
// (docs/DESIGN.md): w13 [E, 2I, H/2] e2m1 (rows 0..I-1 = w1 gate, I..2I-1 = w3 up) + s13 [E, 2I, H/32] ue8m0,
// w2 [E, H, I/2] + s2 [E, H, I/32], I = 576, H = 5120. Ids outside [0, E) contribute nothing.
// Two launches, both deriving the distinct-expert slot table from ids on device (graph safe):
//  * moe13: unit = (slot, 8 intermediate rows) = 8 gate + 8 up weight rows (HMMA rows g / g+8) x full K, split in
//    two K halves over a warp pair (smem reduce); x staged once per CTA as f16 B fragments (per-token power-of-two
//    scale); weights FP4 -> e4m3 bit trick -> f16 (exact, x 2^-6), fresh m16n8k16 accumulator per k32 block
//    post-scaled by the ue8m0 row scale; epilogue SwiGLU (limit clamp) -> h as f16 B fragments of moe2.
//  * moe2: CTA = 16 output rows, warps over slots; per-slot partials x routing weight summed in fixed slot order
//    (deterministic) -> y bf16.
// Fragment k order inside a k32 block (both kernels): lane (g, t) owns physical k [8t, 8t + 8) as the word at byte
// 4t of the row's 16-byte block; nibble pairs (0,4) (1,5) -> HMMA #1 slots, (2,6) (3,7) -> HMMA #2 (sm90.cuh
// fp4_pair_f16x2 order); B fragments hold the activations in the same order: positions P = {0, 4, 1, 5, 2, 6, 3, 7}.
namespace dmoe {
constexpr int H = 5120, I = 576, KB1 = H / 32, KB2 = I / 32, UPE = I / 8;
constexpr int MAXM = 8, MAXSLOTS = 64, MAXE = 512;
constexpr int T = 512, W = T / 32;      // moe13 CTA
constexpr int T2 = 256, W2 = T2 / 32;   // moe2 CTA (3 per SM: 320 CTAs in one wave)
constexpr int UNITS_PER_CTA = W / 2;  // warp pair per unit (K halves)
constexpr int HALF_KB = KB1 / 2;      // 80
constexpr int HB_SLOT = KB2 * 32 * 8;  // halves per slot in the h fragment buffer
constexpr int WPITCH = 144;            // moe13 staged weight row pitch (128 B + 16 B: conflict-free LDS.32)
constexpr int WCHUNK = 16 * WPITCH;    // one 8-kb chunk of 16 rows

struct Args {
  const __nv_bfloat16* x;  // [m, ldx]
  long ldx;
  const int* ids;          // [m, k]
  const float* wts;        // [m, k]
  const uint8_t* w13;
  const uint8_t* s13;
  const uint8_t* w2;
  const uint8_t* s2;
  __half* hb;              // [MAXSLOTS][KB2][32][8]
  __nv_bfloat16* y;        // [m, ldy]
  long ldy;
  // optional shared expert (FP8 e4m3, 32x32 ue8m0 blocks), packed by ds41h/drafter.py SharedPack:
  const uint4* sh13;       // [UPE tiles (8 gate + 8 up rows)][KB1][32 lanes] fragment-major, k P-permuted per 8
  const uint4* ss13;       // [UPE][KB1] u16 (gate byte | up byte << 8), read 8 kb (16 B) at a time
  const uint4* sh2;        // [H / 16 tiles][KB2][32 lanes]
  const uint8_t* ss2;      // [H / 16][KB2]
  // optional fused router (moe13 computes the gate GEMV + sqrtsoftplus/bias top-k and writes ids / wts for moe2):
  const __nv_bfloat16* gate_w;  // [E, H] bf16, or null (ids / wts are inputs)
  const float* gate_bias;       // [E] e_score_correction_bias
  float* glog;                  // [MAXM][E] fp32 scratch (router logits)
  unsigned* gbar;               // grid-barrier counter (monotonic, never reset)
  int* ids_out;                 // [m, k]
  float* wts_out;               // [m, k]
  float route_scale;
  int renormalize;
  // optional fused TP all-reduce of y in moe2 (LL words over CUDA IPC, fixed rank order = ds41h LL all-reduce numerics)
  unsigned long long* ar_recv[RANKS];  // receive buffers (peer-mapped), null = no fused all-reduce
  unsigned* ar_state;                  // [0] epoch, [1] finished-CTA counter
  int ar_rank;
  unsigned long long timeout;
  float limit;
  int m, k, E, shared;
  int dbg;  // DS41H_DRAFTER_MOE_DBG (benchmark diagnostics only): 1 = moe13 skips the FP4 math, 2 = moe13 skips the weight loads,
            // 4 = moe2 not launched
  float* part;     // moe2s: [MAXSLOTS][MAXM][H] fp32 slot partials (x routing weight)
  unsigned* cnt;   // moe2s: [H / M2S_ROWS] arrival counters (self-resetting)
};

struct Slots {
  int n;
  int e[MAXSLOTS];
  float rw[MAXSLOTS][MAXM];
};

// Distinct experts of ids (ascending) and routing weights per (slot, token). All threads participate.
__device__ void build_slots(const Args& a, Slots& sl, unsigned* bits, const int* ids, const float* wts) {
  const int tid = threadIdx.x;
  for (int i = tid; i < MAXE / 32; i += blockDim.x) bits[i] = 0u;
  for (int i = tid; i < MAXSLOTS * MAXM; i += blockDim.x) (&sl.rw[0][0])[i] = 0.0f;
  __syncthreads();
  const int n = a.m * a.k;
  int e = -1;
  if (tid < n) {
    e = ids[tid];
    if (e >= 0 && e < a.E) atomicOr(bits + (e >> 5), 1u << (e & 31));
    else e = -1;
  }
  __syncthreads();
  if (tid < n && e >= 0) {
    int slot = __popc(bits[e >> 5] & ((1u << (e & 31)) - 1u));
    for (int w = 0; w < (e >> 5); ++w) slot += __popc(bits[w]);
    if (slot < MAXSLOTS - 1) {
      sl.e[slot] = e;
      sl.rw[slot][tid / a.k] = wts[tid];
    }
  }
  if (tid == 0) {
    int c = 0;
    for (int w = 0; w < (a.E + 31) / 32; ++w) c += __popc(bits[w]);
    sl.n = min(c, MAXSLOTS - 1);  // routed slots; the shared expert (if any) is slot sl.n with weight 1
  }
  __syncthreads();
  if (a.shared && tid < MAXM) {
    sl.e[sl.n] = -1;
    sl.rw[sl.n][tid] = tid < a.m ? 1.0f : 0.0f;
  }
  __syncthreads();
}

// FP4 word (8 nibbles) -> 4 f16x2 in fragment order (n0,n4) (n1,n5) (n2,n6) (n3,n7), values x 2^-14, with the
// sm90.cuh bit trick (integer ops only: at the power-capped clock the F2FP conversion rate was the limit).
__device__ __forceinline__ void fp4w_to_f16(unsigned w, unsigned (&p)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) p[j] = fp4_pair_f16x2(w, j);
}

__device__ __forceinline__ void hmma_zero(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}
__device__ __forceinline__ void hmma_acc(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One k32 block: rows (g | g+8) words wa / wb, scale factors fa / fb, B fragment xb.
__device__ __forceinline__ void kblock(float (&acc)[4], unsigned wa, unsigned wb, float fa, float fb, uint4 xb) {
  unsigned pa[4], pb[4];
  fp4w_to_f16(wa, pa);
  fp4w_to_f16(wb, pb);
  float d[4];
  hmma_zero(d, pa[0], pb[0], pa[1], pb[1], xb.x, xb.y);
  hmma_acc(d, pa[2], pb[2], pa[3], pb[3], xb.z, xb.w);
  acc[0] = __fmaf_rn(d[0], fa, acc[0]);
  acc[1] = __fmaf_rn(d[1], fa, acc[1]);
  acc[2] = __fmaf_rn(d[2], fb, acc[2]);
  acc[3] = __fmaf_rn(d[3], fb, acc[3]);
}
__device__ __forceinline__ float sc(unsigned byte) { return fp4_block_scale<14>(byte & 0xffu); }
__device__ __forceinline__ float sc8(unsigned byte) { return fp4_block_scale<0>(byte & 0xffu); }

// One k32 block of FP8 rows: wv = {row g: 8 e4m3 in P order, row g+8: 8 e4m3}, same fragment order as kblock.
__device__ __forceinline__ void kblock8(float (&acc)[4], uint4 wv, float fa, float fb, uint4 xb) {
  const unsigned a0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x & 0xffffu)), a1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x >> 16));
  const unsigned a2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y & 0xffffu)), a3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y >> 16));
  const unsigned b0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z & 0xffffu)), b1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z >> 16));
  const unsigned b2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w & 0xffffu)), b3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w >> 16));
  float d[4];
  hmma_zero(d, a0, b0, a1, b1, xb.x, xb.y);
  hmma_acc(d, a2, b2, a3, b3, xb.z, xb.w);
  acc[0] = __fmaf_rn(d[0], fa, acc[0]);
  acc[1] = __fmaf_rn(d[1], fa, acc[1]);
  acc[2] = __fmaf_rn(d[2], fb, acc[2]);
  acc[3] = __fmaf_rn(d[3], fb, acc[3]);
}

// Fused router for moe13 (graph safe): CTA c computes router logits of experts c, c + G, ... for all tokens from the
// staged x fragments (fp32 accumulation of exact bf16 x f16 products), a grid barrier (all G CTAs co-resident, 1 per
// SM), then every CTA runs the stock dsv4_topk semantics (ds41h moe_route.cu moe_topk): score = sqrt(softplus(l)),
// top-k on score + bias (lowest expert id on ties), weights = score * scale / sum (renormalize). CTA 0 writes the
// routing for moe2.
__device__ void route(const Args& a, const uint4* xs, const float* xsc, int* rids, float* rwts) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float part[W][MAXM];
  constexpr int P[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  for (int e = blockIdx.x; e < a.E; e += gridDim.x) {
    float acc[MAXM];
#pragma unroll
    for (int t = 0; t < MAXM; ++t) acc[t] = 0.0f;
    const uint4* wrow = reinterpret_cast<const uint4*>(a.gate_w + long(e) * H);
    for (int j = tid; j < H / 8; j += T) {  // k group [8j, 8j + 8) = (kb = j / 4, t' = j % 4)
      const uint4 wv = __ldg(wrow + j);
      float w8[8];
      {
        const float2 f0 = unpack_bf16(wv.x), f1 = unpack_bf16(wv.y), f2 = unpack_bf16(wv.z), f3 = unpack_bf16(wv.w);
        w8[0] = f0.x, w8[1] = f0.y, w8[2] = f1.x, w8[3] = f1.y, w8[4] = f2.x, w8[5] = f2.y, w8[6] = f3.x, w8[7] = f3.y;
      }
      const int kb = j >> 2, tt = j & 3;
#pragma unroll
      for (int t = 0; t < MAXM; ++t) {
        if (t < a.m) {
          const uint4 xv = xs[kb * 32 + t * 4 + tt];
          const unsigned u[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u[q]));
            acc[t] = __fmaf_rn(f.x, w8[P[2 * q]], acc[t]);
            acc[t] = __fmaf_rn(f.y, w8[P[2 * q + 1]], acc[t]);
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < MAXM; ++t) {
      const float v = warp_sum(acc[t]);
      if (lane == 0) part[warp][t] = v;
    }
    __syncthreads();
    if (tid < a.m) {
      float v = 0.0f;
      for (int w = 0; w < W; ++w) v += part[w][tid];
      a.glog[long(tid) * a.E + e] = v * xsc[tid];
    }
    __syncthreads();
  }
  // grid barrier: monotonic counter, target = next multiple of G
  if (tid == 0) {
    __threadfence();
    const unsigned old = atomicAdd(a.gbar, 1u);
    const unsigned target = (old / gridDim.x + 1u) * gridDim.x;
    while (int(ld_acquire_gpu(a.gbar) - target) < 0) {
    }
  }
  __syncthreads();
  // top-k per token: warp t
  if (warp < a.m) {
    const int row = warp;
    const float* lrow = a.glog + long(row) * a.E;
    auto score = [](float x) { return sqrtf(x > 20.f ? x : logf(1.f + expf(x))); };
    constexpr int VPL = MAXE / 32;
    float cur[VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      const int e = v * 32 + lane;
      if (e < a.E) {
        const float c = score(__ldcg(lrow + e)) + a.gate_bias[e];
        cur[v] = (c == c) ? c : -1e30f;
      } else {
        cur[v] = -INFINITY;
      }
    }
    float sel_w = 0.f, sum = 0.f;
    int sel_id = 0;
    for (int k = 0; k < a.k; ++k) {
      float best = -INFINITY;
      int best_e = 0x7fffffff;
#pragma unroll
      for (int v = 0; v < VPL; ++v) {
        const int e = v * 32 + lane;
        if (cur[v] > best || (cur[v] == best && e < best_e)) best = cur[v], best_e = e;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
        if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
      }
#pragma unroll
      for (int v = 0; v < VPL; ++v)
        if (v * 32 + lane == best_e) cur[v] = -INFINITY;
      const float wk = score(__ldcg(lrow + best_e));
      sum += wk;
      if (lane == k) sel_w = wk, sel_id = best_e;
    }
    const float mul = a.renormalize ? a.route_scale / (sum > 0.f ? sum : 1.f) : a.route_scale;
    if (lane < a.k) {
      rids[row * a.k + lane] = sel_id;
      rwts[row * a.k + lane] = sel_w * mul;
      if (blockIdx.x == 0) {
        a.ids_out[row * a.k + lane] = sel_id;
        a.wts_out[row * a.k + lane] = sel_w * mul;
      }
    }
  }
  __syncthreads();
}

// NST: weight-chunk pipeline depth per warp (2 = original double buffering; 3 keeps two 8-kb chunks in flight while one
// computes: at 350 W the per-warp chunk latency (~1-1.5 us) exceeds the 8-kb compute time (~0.9 us at 16 warps/SM)).
__device__ __forceinline__ void cp_async_wait_n(int n) {
  if (n <= 0) cp_async_wait<0>();
  else if (n == 1) cp_async_wait<1>();
  else if (n == 2) cp_async_wait<2>();
  else cp_async_wait<3>();
}
template <int NST, int CKB>
__global__ void __launch_bounds__(T, 1) moe13_kernel(const Args a) {
  constexpr int WCH = 16 * (CKB * 16 + 16);  // staged chunk: 16 rows x (CKB * 16 B + 16 B pad)
  constexpr int PITCH = CKB * 16 + 16;
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB1][32] B fragments
  unsigned char* wsm = smem + KB1 * 32 * 16;    // [W warps][2 buffers][WCHUNK] staged weight chunks
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float xsc[MAXM];
  __shared__ float red[UNITS_PER_CTA][32][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  __shared__ int rids[MAXM * 16];
  __shared__ float rwts[MAXM * 16];
  pdl_wait();
  stage_x<H, T, true>(a.x, a.ldx, a.m, xs, xsc);
  if (a.gate_w != nullptr) {
    route(a, xs, xsc, rids, rwts);  // gate GEMV + grid barrier + top-k (every CTA), CTA 0 publishes ids / wts
    build_slots(a, sl, bits, rids, rwts);
  } else {
    build_slots(a, sl, bits, a.ids, a.wts);
  }
  const int units = (sl.n + (a.shared ? 1 : 0)) * UPE;
  const int j = warp >> 1, half = warp & 1;
  for (int base = blockIdx.x * UNITS_PER_CTA; base < units; base += gridDim.x * UNITS_PER_CTA) {
    // Work order: the shared expert's FP8 units (twice the bytes of an FP4 unit) first, so that a second pass (units >
    // gridDim * UNITS_PER_CTA, e.g. 14 routed + shared = 1080 > 1056) only holds short FP4 units. u = permuted unit index
    // (slot = u / UPE in the hb buffer's numbering: shared = slot sl.n).
    const int uw = base + j;
    const int u = a.shared ? (uw < UPE ? sl.n * UPE + uw : uw - UPE) : uw;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (uw < units && u / UPE == sl.n) {  // shared expert unit (FP8)
      const int p = u % UPE;
      const uint4* wp = a.sh13 + long(p) * KB1 * 32 + lane;
      const uint4* sp = a.ss13 + long(p) * (KB1 / 8);
      const int kb0 = half * HALF_KB;
      // double-buffered 8-kb chunks: the next chunk's loads are in flight while this one computes
      uint4 wv0[8], wv1[8], sv0, sv1;
      auto load = [&](uint4 (&wv)[8], uint4& sv, int kb) {
#pragma unroll
        for (int q = 0; q < 8; ++q) wv[q] = ld_stream16(wp + (kb + q) * 32);
        sv = __ldg(sp + kb / 8);
      };
      auto compute = [&](const uint4 (&wv)[8], const uint4 sv, int kb) {
        const unsigned sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const unsigned pair = sw[q >> 1] >> (16 * (q & 1));
          kblock8(acc, wv[q], sc8(pair), sc8(pair >> 8), xs[(kb + q) * 32 + lane]);
        }
      };
      load(wv0, sv0, kb0);
#pragma unroll 1
      for (int kb = kb0; kb < kb0 + HALF_KB; kb += 16) {
        load(wv1, sv1, kb + 8);
        compute(wv0, sv0, kb);
        if (kb + 16 < kb0 + HALF_KB) load(wv0, sv0, kb + 16);
        compute(wv1, sv1, kb + 8);
      }
    } else if (uw < units) {
      const int slot = u / UPE, i0 = (u % UPE) * 8;
      const long e = sl.e[slot];
      const uint8_t* we = a.w13 + e * (2L * I * (H / 2));
      const uint8_t* se = a.s13 + e * (2L * I * KB1);
      const uint8_t* sg = se + long(i0 + g) * KB1;
      const uint8_t* su = se + long(I + i0 + g) * KB1;
      const int kb0 = half * HALF_KB;
      // Weight chunks (8 k32 blocks = 128 B per row x 8 gate + 8 up rows) staged through this warp's shared buffers
      // with coalesced 16-byte cp.async (whole 128-byte lines; per-lane 4-byte global loads of 16-byte row segments
      // wasted half of every sector), row pitch 144 B (conflict-free LDS.32: bank = 4g + t), double buffered.
      if constexpr (NST == 2 && CKB == 8) {  // the original double-buffered loop (unroll 1: small code, in-situ i-cache)
        unsigned char* wbuf = wsm + warp * (2 * WCHUNK);
        const uint8_t* rowp[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int pce = lane + 32 * i, row = pce >> 3;  // 16 rows x 8 pieces
          rowp[i] = we + long(row < 8 ? i0 + row : I + i0 + row - 8) * (H / 2) + (pce & 7) * 16;
        }
        auto load = [&](int buf, int kb) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int pce = lane + 32 * i;
            cp_async16(wbuf + buf * WCHUNK + (pce >> 3) * WPITCH + (pce & 7) * 16, rowp[i] + kb * 16);
          }
          cp_async_commit();
        };
        uint2 sa0, sb0, sa1, sb1;
        auto compute = [&](int buf, const uint2 sa, const uint2 sb, int kb) {
          const unsigned char* b = wbuf + buf * WCHUNK + 4 * t;
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const unsigned wa = *reinterpret_cast<const unsigned*>(b + g * WPITCH + q * 16);
            const unsigned wb = *reinterpret_cast<const unsigned*>(b + (8 + g) * WPITCH + q * 16);
            const unsigned ba = (q < 4 ? sa.x : sa.y) >> (8 * (q & 3));
            const unsigned bb = (q < 4 ? sb.x : sb.y) >> (8 * (q & 3));
            kblock(acc, wa, wb, sc(ba), sc(bb), xs[(kb + q) * 32 + lane]);
          }
        };
        load(0, kb0);
        sa0 = __ldg(reinterpret_cast<const uint2*>(sg + kb0));
        sb0 = __ldg(reinterpret_cast<const uint2*>(su + kb0));
#pragma unroll 1
        for (int kb = kb0; kb < kb0 + HALF_KB; kb += 16) {
          load(1, kb + 8);
          sa1 = __ldg(reinterpret_cast<const uint2*>(sg + kb + 8));
          sb1 = __ldg(reinterpret_cast<const uint2*>(su + kb + 8));
          cp_async_wait<1>();
          __syncwarp();
          compute(0, sa0, sb0, kb);
          __syncwarp();
          if (kb + 16 < kb0 + HALF_KB) {
            load(0, kb + 16);
            sa0 = __ldg(reinterpret_cast<const uint2*>(sg + kb + 16));
            sb0 = __ldg(reinterpret_cast<const uint2*>(su + kb + 16));
            cp_async_wait<1>();
          } else {
            cp_async_wait<0>();
          }
          __syncwarp();
          compute(1, sa1, sb1, kb + 8);
          __syncwarp();
        }
          } else {
        unsigned char* wbuf = wsm + warp * (NST * WCH);
        constexpr int PIECES = CKB;  // 16-byte pieces per row per chunk
        constexpr int PER_LANE = 16 * PIECES / 32;
        const uint8_t* rowp[PER_LANE];
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
          const int pce = lane + 32 * i, row = pce / PIECES;  // 16 rows x PIECES pieces
          rowp[i] = we + long(row < 8 ? i0 + row : I + i0 + row - 8) * (H / 2) + (pce % PIECES) * 16;
        }
        constexpr int NCH = HALF_KB / CKB;  // chunks per K half
        static_assert(HALF_KB % CKB == 0 && CKB % 8 == 0, "chunking");
        uint2 sa[NST][CKB / 8], sb[NST][CKB / 8];
        auto load = [&](int c) {  // chunk c -> buffer c % NST (+ its gate / up row scales)
          const int buf = c % NST, kb = kb0 + CKB * c;
          if (!(a.dbg & 2)) {
#pragma unroll
            for (int i = 0; i < PER_LANE; ++i) {
              const int pce = lane + 32 * i;
              cp_async16(wbuf + buf * WCH + (pce / PIECES) * PITCH + (pce % PIECES) * 16, rowp[i] + kb * 16);
            }
          }
          cp_async_commit();
#pragma unroll
          for (int h8 = 0; h8 < CKB / 8; ++h8) {
            sa[buf][h8] = __ldg(reinterpret_cast<const uint2*>(sg + kb + 8 * h8));
            sb[buf][h8] = __ldg(reinterpret_cast<const uint2*>(su + kb + 8 * h8));
          }
        };
        auto compute = [&](int buf, int kb) {
          if (a.dbg & 1) return;
          const unsigned char* b = wbuf + buf * WCH + 4 * t;
#pragma unroll
          for (int q = 0; q < CKB; ++q) {
            const unsigned wa = *reinterpret_cast<const unsigned*>(b + g * PITCH + q * 16);
            const unsigned wb = *reinterpret_cast<const unsigned*>(b + (8 + g) * PITCH + q * 16);
            const uint2 sa_ = sa[buf][q / 8], sb_ = sb[buf][q / 8];
            const unsigned ba = ((q & 7) < 4 ? sa_.x : sa_.y) >> (8 * (q & 3));
            const unsigned bb = ((q & 7) < 4 ? sb_.x : sb_.y) >> (8 * (q & 3));
            kblock(acc, wa, wb, sc(ba), sc(bb), xs[(kb + q) * 32 + lane]);
          }
        };
#pragma unroll
        for (int c = 0; c < NST - 1; ++c) load(c);
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
          if (c + NST - 1 < NCH) load(c + NST - 1);  // into the buffer computed in iteration c - 1 (warp-synced below)
          cp_async_wait_n(min(NST - 1, NCH - 1 - c));  // chunk c landed (later chunks may still be in flight)
          __syncwarp();
          compute(c % NST, kb0 + CKB * c);
          __syncwarp();
        }
          }
    }
    if (half) {
#pragma unroll
      for (int q = 0; q < 4; ++q) red[j][lane][q] = acc[q];
    }
    __syncthreads();
    if (!half && uw < units) {
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[q] += red[j][lane][q];
      const int slot = u / UPE, i0 = (u % UPE) * 8;
      const int ii = i0 + g, kb2 = ii >> 5, t2 = (ii & 31) >> 3;
      const int pos = (g & 3) * 2 + (g >> 2);  // Pinv[g] for P = {0, 4, 1, 5, 2, 6, 3, 7}
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * t + c;
        const float gate = fminf(acc[c] * xsc[tok], a.limit);
        const float up = fminf(fmaxf(acc[2 + c] * xsc[tok], -a.limit), a.limit);
        float h = gate / (1.0f + expf(-gate)) * up;
        h = fminf(fmaxf(h, -65504.0f), 65504.0f);
        a.hb[(long(slot) * KB2 * 32 + kb2 * 32 + tok * 4 + t2) * 8 + pos] = __float2half_rn(h);
      }
    }
    __syncthreads();
  }
  pdl_launch_dependents();
}

// moe13q (DS41H_DRAFTER_MOE13_VARIANT=3): the same math as moe13 with BALANCED work: unit u (shared FP8 units first) runs
// on CTA u % G, and every unit is split into 4 K quarters (40 kb) that the CTA's 16 warps take round robin (batches of
// QB units: QB * 4 items = 2 rounds), quarter partials summed in fixed order in smem. moe13 runs a whole second pass of
// K halves on a few CTAs when units > 8 G (k=7: up to 22 slots x 72 = 1584 units vs 1056); here max over CTAs of
// ceil(units / G) * 4 quarter items / 16 warps rounds.
constexpr int QB = 8;  // units per CTA batch
constexpr int QKB = KB1 / 4;  // kb per quarter
__global__ void __launch_bounds__(T, 1) moe13q_kernel(const Args a) {
  constexpr int CKB = 8, NST = 2, WCH = 16 * (CKB * 16 + 16), PITCH = CKB * 16 + 16, NCH = QKB / CKB;
  static_assert(QKB % CKB == 0 && QKB % 8 == 0, "quarters");
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB1][32] B fragments
  unsigned char* wsm = smem + KB1 * 32 * 16;    // [W warps][NST][WCH]
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float xsc[MAXM];
  __shared__ float qred[QB][4][32][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  __shared__ int rids[MAXM * 16];
  __shared__ float rwts[MAXM * 16];
  pdl_wait();
  stage_x<H, T, true>(a.x, a.ldx, a.m, xs, xsc);
  if (a.gate_w != nullptr) {
    route(a, xs, xsc, rids, rwts);
    build_slots(a, sl, bits, rids, rwts);
  } else {
    build_slots(a, sl, bits, a.ids, a.wts);
  }
  const int units = (sl.n + (a.shared ? 1 : 0)) * UPE;
  const int G = gridDim.x;
  const int my_units = units > int(blockIdx.x) ? (units - 1 - int(blockIdx.x)) / G + 1 : 0;
  unsigned char* wbuf = wsm + warp * (NST * WCH);
  for (int b0 = 0; b0 < my_units; b0 += QB) {
    const int nb = min(QB, my_units - b0);
    for (int it = warp; it < nb * 4; it += W) {
      const int ul = it >> 2, q4 = it & 3;
      const int uw = int(blockIdx.x) + (b0 + ul) * G;  // work-order unit index
      const int u = a.shared ? (uw < UPE ? sl.n * UPE + uw : uw - UPE) : uw;
      const int kb0 = q4 * QKB;
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (u / UPE == sl.n) {  // shared expert unit (FP8), quarter [kb0, kb0 + QKB)
        const int p = u % UPE;
        const uint4* wp = a.sh13 + long(p) * KB1 * 32 + lane;
        const uint4* sp = a.ss13 + long(p) * (KB1 / 8);
        uint4 wv0[8], wv1[8], sv0, sv1;
        auto load = [&](uint4 (&wv)[8], uint4& sv, int kb) {
#pragma unroll
          for (int q = 0; q < 8; ++q) wv[q] = ld_stream16(wp + (kb + q) * 32);
          sv = __ldg(sp + kb / 8);
        };
        auto compute = [&](const uint4 (&wv)[8], const uint4 sv, int kb) {
          const unsigned sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const unsigned pair = sw[q >> 1] >> (16 * (q & 1));
            kblock8(acc, wv[q], sc8(pair), sc8(pair >> 8), xs[(kb + q) * 32 + lane]);
          }
        };
        load(wv0, sv0, kb0);
#pragma unroll 1
        for (int kb = kb0; kb < kb0 + QKB; kb += 16) {  // 5 chunks of 8 kb: (0,1) (2,3) (4)
          const bool has1 = kb + 8 < kb0 + QKB;
          if (has1) load(wv1, sv1, kb + 8);
          compute(wv0, sv0, kb);
          if (kb + 16 < kb0 + QKB) load(wv0, sv0, kb + 16);
          if (has1) compute(wv1, sv1, kb + 8);
        }
      } else {
        const int slot = u / UPE, i0 = (u % UPE) * 8;
        const long e = sl.e[slot];
        const uint8_t* we = a.w13 + e * (2L * I * (H / 2));
        const uint8_t* se = a.s13 + e * (2L * I * KB1);
        const uint8_t* sg = se + long(i0 + g) * KB1;
        const uint8_t* su = se + long(I + i0 + g) * KB1;
        constexpr int PER_LANE = 16 * CKB / 32;
        const uint8_t* rowp[PER_LANE];
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
          const int pce = lane + 32 * i, row = pce / CKB;
          rowp[i] = we + long(row < 8 ? i0 + row : I + i0 + row - 8) * (H / 2) + (pce % CKB) * 16;
        }
        uint2 sa[NST], sb[NST];
        auto load = [&](int buf, int c) {  // chunk c -> buffer buf (constant at every call site)
          const int kb = kb0 + CKB * c;
#pragma unroll
          for (int i = 0; i < PER_LANE; ++i) {
            const int pce = lane + 32 * i;
            cp_async16(wbuf + buf * WCH + (pce / CKB) * PITCH + (pce % CKB) * 16, rowp[i] + kb * 16);
          }
          cp_async_commit();
          sa[buf] = __ldg(reinterpret_cast<const uint2*>(sg + kb));
          sb[buf] = __ldg(reinterpret_cast<const uint2*>(su + kb));
        };
        auto compute = [&](int buf, int kb) {
          const unsigned char* bp = wbuf + buf * WCH + 4 * t;
#pragma unroll
          for (int q = 0; q < CKB; ++q) {
            const unsigned wa = *reinterpret_cast<const unsigned*>(bp + g * PITCH + q * 16);
            const unsigned wb = *reinterpret_cast<const unsigned*>(bp + (8 + g) * PITCH + q * 16);
            const unsigned ba = (q < 4 ? sa[buf].x : sa[buf].y) >> (8 * (q & 3));
            const unsigned bb = (q < 4 ? sb[buf].x : sb[buf].y) >> (8 * (q & 3));
            kblock(acc, wa, wb, sc(ba), sc(bb), xs[(kb + q) * 32 + lane]);
          }
        };
        load(0, 0);
#pragma unroll 1
        for (int c = 0; c < NCH; c += 2) {  // chunk pairs (buffers 0 / 1), unroll 1 (small code)
          const bool has1 = c + 1 < NCH;
          if (has1) {
            load(1, c + 1);
            cp_async_wait<1>();
          } else {
            cp_async_wait<0>();
          }
          __syncwarp();
          compute(0, kb0 + CKB * c);
          __syncwarp();
          if (has1) {
            if (c + 2 < NCH) {
              load(0, c + 2);
              cp_async_wait<1>();
            } else {
              cp_async_wait<0>();
            }
            __syncwarp();
            compute(1, kb0 + CKB * (c + 1));
            __syncwarp();
          }
        }
      }
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) qred[ul][q4][lane][qq] = acc[qq];
    }
    __syncthreads();
    for (int ul = warp; ul < nb; ul += W) {  // epilogue: unit ul = 4 quarters in fixed order -> SwiGLU -> h fragments
      const int uw = int(blockIdx.x) + (b0 + ul) * G;
      const int u = a.shared ? (uw < UPE ? sl.n * UPE + uw : uw - UPE) : uw;
      float v[4];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) v[qq] = ((qred[ul][0][lane][qq] + qred[ul][1][lane][qq]) + qred[ul][2][lane][qq]) + qred[ul][3][lane][qq];
      const int slot = u / UPE, i0 = (u % UPE) * 8;
      const int ii = i0 + g, kb2 = ii >> 5, t2 = (ii & 31) >> 3;
      const int pos = (g & 3) * 2 + (g >> 2);  // Pinv[g] for P = {0, 4, 1, 5, 2, 6, 3, 7}
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * t + c;
        const float gate = fminf(v[c] * xsc[tok], a.limit);
        const float up = fminf(fmaxf(v[2 + c] * xsc[tok], -a.limit), a.limit);
        float h = gate / (1.0f + expf(-gate)) * up;
        h = fminf(fmaxf(h, -65504.0f), 65504.0f);
        a.hb[(long(slot) * KB2 * 32 + kb2 * 32 + tok * 4 + t2) * 8 + pos] = __float2half_rn(h);
      }
    }
    __syncthreads();
  }
  pdl_launch_dependents();
}

// moe13r (DS41H_DRAFTER_MOE13_VARIANT=4): moe13q's balanced quarter schedule, routed FP4 weights through a per-warp
// NST-deep cp.async ring (runtime buffer index: one compute call site, small code) with the 16 rows' scale bytes copied
// into the same ring slot, so NST - 1 chunks (2 KB each) stay in flight per warp instead of one. Same math as moe13q.
constexpr int R_CKB = 8, R_PITCH = R_CKB * 16 + 16, R_WCH = 16 * R_PITCH + 16 * 8;  // 16 rows x (128 + 16) B + 16 x 8 scale B
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
template <int NST>
__global__ void __launch_bounds__(T, 1) moe13r_kernel(const Args a) {
  constexpr int CKB = R_CKB, PITCH = R_PITCH, WCH = R_WCH, NCH = QKB / CKB;
  static_assert(QKB % CKB == 0, "quarters");
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB1][32] B fragments
  unsigned char* wsm = smem + KB1 * 32 * 16;    // [W warps][NST][WCH]
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float xsc[MAXM];
  __shared__ float qred[QB][4][32][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  __shared__ int rids[MAXM * 16];
  __shared__ float rwts[MAXM * 16];
  pdl_wait();
  stage_x<H, T, true>(a.x, a.ldx, a.m, xs, xsc);
  if (a.gate_w != nullptr) {
    route(a, xs, xsc, rids, rwts);
    build_slots(a, sl, bits, rids, rwts);
  } else {
    build_slots(a, sl, bits, a.ids, a.wts);
  }
  const int units = (sl.n + (a.shared ? 1 : 0)) * UPE;
  const int G = gridDim.x;
  const int my_units = units > int(blockIdx.x) ? (units - 1 - int(blockIdx.x)) / G + 1 : 0;
  unsigned char* ring = wsm + warp * (NST * WCH);
  for (int b0 = 0; b0 < my_units; b0 += QB) {
    const int nb = min(QB, my_units - b0);
    for (int it = warp; it < nb * 4; it += W) {
      const int ul = it >> 2, q4 = it & 3;
      const int uw = int(blockIdx.x) + (b0 + ul) * G;  // work-order unit index
      const int u = a.shared ? (uw < UPE ? sl.n * UPE + uw : uw - UPE) : uw;
      const int kb0 = q4 * QKB;
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (u / UPE == sl.n) {  // shared expert unit (FP8), quarter [kb0, kb0 + QKB): direct fragment-major loads
        const int p = u % UPE;
        const uint4* wp = a.sh13 + long(p) * KB1 * 32 + lane;
        const uint4* sp = a.ss13 + long(p) * (KB1 / 8);
        uint4 wv0[8], wv1[8], sv0, sv1;
        auto load = [&](uint4 (&wv)[8], uint4& sv, int kb) {
#pragma unroll
          for (int q = 0; q < 8; ++q) wv[q] = ld_stream16(wp + (kb + q) * 32);
          sv = __ldg(sp + kb / 8);
        };
        auto compute = [&](const uint4 (&wv)[8], const uint4 sv, int kb) {
          const unsigned sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const unsigned pair = sw[q >> 1] >> (16 * (q & 1));
            kblock8(acc, wv[q], sc8(pair), sc8(pair >> 8), xs[(kb + q) * 32 + lane]);
          }
        };
        load(wv0, sv0, kb0);
#pragma unroll 1
        for (int kb = kb0; kb < kb0 + QKB; kb += 16) {
          const bool has1 = kb + 8 < kb0 + QKB;
          if (has1) load(wv1, sv1, kb + 8);
          compute(wv0, sv0, kb);
          if (kb + 16 < kb0 + QKB) load(wv0, sv0, kb + 16);
          if (has1) compute(wv1, sv1, kb + 8);
        }
      } else {
        const int slot = u / UPE, i0 = (u % UPE) * 8;
        const long e = sl.e[slot];
        const uint8_t* we = a.w13 + e * (2L * I * (H / 2));
        const uint8_t* se = a.s13 + e * (2L * I * KB1);
        constexpr int PER_LANE = 16 * CKB / 32;
        const uint8_t* rowp[PER_LANE];
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
          const int pce = lane + 32 * i, row = pce / CKB;
          rowp[i] = we + long(row < 8 ? i0 + row : I + i0 + row - 8) * (H / 2) + (pce % CKB) * 16;
        }
        const uint8_t* srow = se + long(lane < 8 ? i0 + lane : I + i0 + (lane & 7)) * KB1;  // lanes 0..15 copy scales
        auto load = [&](int c) {  // chunk c -> ring slot c % NST (weights + the 16 rows' 8 scale bytes)
          unsigned char* dst = ring + (c % NST) * WCH;
          const int kb = kb0 + CKB * c;
#pragma unroll
          for (int i = 0; i < PER_LANE; ++i) {
            const int pce = lane + 32 * i;
            cp_async16(dst + (pce / CKB) * PITCH + (pce % CKB) * 16, rowp[i] + kb * 16);
          }
          if (lane < 16) cp_async8(dst + 16 * PITCH + lane * 8, srow + kb);
        };
#pragma unroll
        for (int c = 0; c < NST - 1; ++c) {
          if (c < NCH) load(c);
          cp_async_commit();
        }
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          if (c + NST - 1 < NCH) load(c + NST - 1);
          cp_async_commit();
          cp_async_wait<NST - 1>();
          __syncwarp();
          const unsigned char* bp = ring + (c % NST) * WCH;
          const uint2 sa = *reinterpret_cast<const uint2*>(bp + 16 * PITCH + g * 8);
          const uint2 sb = *reinterpret_cast<const uint2*>(bp + 16 * PITCH + (8 + g) * 8);
          const int kb = kb0 + CKB * c;
#pragma unroll
          for (int q = 0; q < CKB; ++q) {
            const unsigned wa = *reinterpret_cast<const unsigned*>(bp + 4 * t + g * PITCH + q * 16);
            const unsigned wb = *reinterpret_cast<const unsigned*>(bp + 4 * t + (8 + g) * PITCH + q * 16);
            const unsigned ba = (q < 4 ? sa.x : sa.y) >> (8 * (q & 3));
            const unsigned bb = (q < 4 ? sb.x : sb.y) >> (8 * (q & 3));
            kblock(acc, wa, wb, sc(ba), sc(bb), xs[(kb + q) * 32 + lane]);
          }
          __syncwarp();
        }
      }
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) qred[ul][q4][lane][qq] = acc[qq];
    }
    __syncthreads();
    for (int ul = warp; ul < nb; ul += W) {  // epilogue (moe13q): 4 quarters in fixed order -> SwiGLU -> h fragments
      const int uw = int(blockIdx.x) + (b0 + ul) * G;
      const int u = a.shared ? (uw < UPE ? sl.n * UPE + uw : uw - UPE) : uw;
      float v[4];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) v[qq] = ((qred[ul][0][lane][qq] + qred[ul][1][lane][qq]) + qred[ul][2][lane][qq]) + qred[ul][3][lane][qq];
      const int slot = u / UPE, i0 = (u % UPE) * 8;
      const int ii = i0 + g, kb2 = ii >> 5, t2 = (ii & 31) >> 3;
      const int pos = (g & 3) * 2 + (g >> 2);
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * t + c;
        const float gate = fminf(v[c] * xsc[tok], a.limit);
        const float up = fminf(fmaxf(v[2 + c] * xsc[tok], -a.limit), a.limit);
        float h = gate / (1.0f + expf(-gate)) * up;
        h = fminf(fmaxf(h, -65504.0f), 65504.0f);
        a.hb[(long(slot) * KB2 * 32 + kb2 * 32 + tok * 4 + t2) * 8 + pos] = __float2half_rn(h);
      }
    }
    __syncthreads();
  }
  pdl_launch_dependents();
}

// moe13v (DS41H_DRAFTER_MOE13_VARIANT=5): the same math as moe13 (FP4 / FP8 kblocks on f16 x fragments, fp32 per-kb scaled
// accumulation, clamped SwiGLU -> f16 hb) with the vrh row-stream structure: every CTA streams its OWN contiguous run of whole
// units through ONE CTA-wide cp.async ring of 8-KB stages (a stage = 16 rows x 32 kb of one FP4 unit or 16 kb of one FP8 shared
// tile; all 512 threads issue it, 16 B each), and every warp takes a fixed k-slice of each stage. Work balance: shared tile c runs
// on CTA c (c < 72), the routed units (5 stages each) are dealt to the CTAs by stage budget (shared CTAs get 10 stages less), so
// every CTA streams ~TS / G stages with all its bytes in flight across a D-deep ring (moe13: per-warp unit streams with 2 chunks
// in flight + a second pass on a few CTAs when units > 8 G). Unit partials of the 16 warps are summed in warp order (smem,
// double-buffered by unit parity) by warps 0-1 at the next stage: deterministic.
constexpr int V_STB = 16 * (512 + 16);       // FP4 stage: 16 rows x 512 B (+16 B pitch pad); FP8 stage uses 8 KB of it
constexpr int V_SCB = 16 * 32;               // per-stage scale bytes (FP4: 16 rows x 32 kb; FP8: 16 kb x u16)
constexpr int V_PITCH = 512 + 16;
template <int D>
struct V13 {
  static constexpr int RING = D * (V_STB + V_SCB);
  static constexpr int RED = 2 * W * 32 * 4 * 4;
  static constexpr int SMEM = KB1 * 32 * 16 + RING + RED;
};
template <int D>
__global__ void __launch_bounds__(T, 1) moe13v_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);                     // [KB1][32] B fragments (P order)
  unsigned char* ring = smem + KB1 * 32 * 16;                     // [D][V_STB weights | V_SCB scales]
  float* red = reinterpret_cast<float*>(ring + D * (V_STB + V_SCB));  // [2][W][32][4]
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float xsc[MAXM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x, c = blockIdx.x;
  pdl_wait();
  build_slots(a, sl, bits, a.ids, a.wts);
  // ---- this CTA's work: [shared tile c] + routed units [u0, u1) (unit = slot * UPE + group)
  const int nsh = a.shared ? UPE : 0;
  const int nrt = sl.n * UPE;
  const long ts = long(nsh) * 10 + long(nrt) * 5;                 // total stages
  auto budget_prefix = [&](int cc) -> long {                      // routed stages of CTAs [0, cc)
    const long tot = (ts * cc) / G;                               // all stages of CTAs [0, cc)
    const long shd = 10L * (cc < nsh ? cc : nsh);                 // their pre-assigned shared stages
    return tot > shd ? tot - shd : 0;
  };
  long p0 = budget_prefix(c), p1 = budget_prefix(c + 1);
  int u0 = int((p0 + 2) / 5), u1 = c == G - 1 ? nrt : int((p1 + 2) / 5);
  if (u0 > nrt) u0 = nrt;
  if (u1 > nrt) u1 = nrt;
  if (u1 < u0) u1 = u0;
  const bool has_sh = c < nsh;
  const int nst = (has_sh ? 10 : 0) + (u1 - u0) * 5;             // stages of this CTA
  // stage j -> (kind, unit, kstage)
  auto stage_unit = [&](int j, int& fp8, int& unit, int& q) {
    if (has_sh && j < 10) { fp8 = 1; unit = c; q = j; return; }
    const int r = j - (has_sh ? 10 : 0);
    fp8 = 0; unit = u0 + r / 5; q = r % 5;
  };
  auto issue = [&](int j) {
    if (j < nst) {
      int fp8, unit, q;
      stage_unit(j, fp8, unit, q);
      unsigned char* st = ring + (j % D) * (V_STB + V_SCB);
      unsigned char* sc_ = st + V_STB;
      if (fp8) {
        // shared tile `unit`: fragment-major [kb][32 lanes] uint4, 16 kb = 8 KB contiguous; scales [kb] u16
        const uint4* src = a.sh13 + (long(unit) * KB1 + q * 16) * 32;
        cp_async16(st + tid * 16, src + tid);
        if (tid < 2) cp_async16(sc_ + tid * 16, reinterpret_cast<const unsigned char*>(a.ss13) + (long(unit) * KB1 + q * 16) * 2 + tid * 16);
      } else {
        const int slot = unit / UPE, i0 = (unit % UPE) * 8;
        const long e = sl.e[slot];
        const int row = tid >> 5, pce = tid & 31;                 // 16 rows x 32 pieces of 16 B (512 B per row)
        const long grow = row < 8 ? i0 + row : I + i0 + row - 8;
        const uint8_t* src = a.w13 + e * (2L * I * (H / 2)) + grow * (H / 2) + q * 512 + pce * 16;
        cp_async16(st + row * V_PITCH + pce * 16, src);
        if (tid < 32) {                                           // scales: 16 rows x 32 kb (2 x 16 B per row)
          const int rr = tid >> 1, hh = tid & 1;
          const long gr = rr < 8 ? i0 + rr : I + i0 + rr - 8;
          cp_async16(sc_ + rr * 32 + hh * 16, a.s13 + e * (2L * I * KB1) + gr * KB1 + q * 32 + hh * 16);
        }
      }
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int j = 0; j < D - 1; ++j) issue(j);
  stage_x<H, T, true>(a.x, a.ldx, a.m, xs, xsc);  // overlaps the first stages' loads; ends with __syncthreads
  // unit-end reduction (warps 0-1, one stage later): thread (lane l, token half c2) sums gate (reg c2) / up (reg 2 + c2) of
  // the 16 warp partials in warp order, then the clamped SwiGLU -> hb
  auto reduce_unit = [&](int fp8, int unit, int par) {
    const int rl = tid & 31, c2 = tid >> 5;                       // tid < 64
    const int rg = rl >> 2, rt = rl & 3;
    float gs = 0.0f, us = 0.0f;
    const float* rp = red + par * (W * 32 * 4);
#pragma unroll
    for (int w = 0; w < W; ++w) gs += rp[(w * 32 + rl) * 4 + c2], us += rp[(w * 32 + rl) * 4 + 2 + c2];
    const int slot = fp8 ? sl.n : unit / UPE;
    const int i0 = fp8 ? unit * 8 : (unit % UPE) * 8;
    const int ii = i0 + rg, kb2 = ii >> 5, t2 = (ii & 31) >> 3;
    const int pos = (rg & 3) * 2 + (rg >> 2);  // Pinv[g] for P = {0, 4, 1, 5, 2, 6, 3, 7}
    const int tok = 2 * rt + c2;
    const float gate = fminf(gs * xsc[tok], a.limit);
    const float up = fminf(fmaxf(us * xsc[tok], -a.limit), a.limit);
    float h = gate / (1.0f + expf(-gate)) * up;
    h = fminf(fmaxf(h, -65504.0f), 65504.0f);
    a.hb[(long(slot) * KB2 * 32 + kb2 * 32 + tok * 4 + t2) * 8 + pos] = __float2half_rn(h);
  };
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int pend_fp8 = 0, pend_unit = -1, pend_par = 0, uidx = 0;
#pragma unroll 1
  for (int j = 0; j < nst; ++j) {
    cp_async_wait<D - 2>();
    __syncthreads();  // stage j landed everywhere; stage j - 1 consumed by every warp (its slot is refilled below)
    if (pend_unit >= 0) {
      if (tid < 64) reduce_unit(pend_fp8, pend_unit, pend_par);
      pend_unit = -1;
    }
    issue(j + D - 1);
    int fp8, unit, q;
    stage_unit(j, fp8, unit, q);
    const unsigned char* st = ring + (j % D) * (V_STB + V_SCB);
    const unsigned char* sc_ = st + V_STB;
    if (fp8) {
      const int kbl = warp;                                       // 16 kb per stage, one per warp
      const uint4 wv = reinterpret_cast<const uint4*>(st)[kbl * 32 + lane];
      const unsigned short pr = reinterpret_cast<const unsigned short*>(sc_)[kbl];
      kblock8(acc, wv, sc8(pr & 0xffu), sc8(pr >> 8), xs[(q * 16 + kbl) * 32 + lane]);
    } else {
#pragma unroll
      for (int z = 0; z < 2; ++z) {                               // 32 kb per stage, two per warp
        const int kbl = 2 * warp + z;
        const unsigned wa = *reinterpret_cast<const unsigned*>(st + g * V_PITCH + kbl * 16 + 4 * t);
        const unsigned wb = *reinterpret_cast<const unsigned*>(st + (8 + g) * V_PITCH + kbl * 16 + 4 * t);
        kblock(acc, wa, wb, sc(sc_[g * 32 + kbl]), sc(sc_[(8 + g) * 32 + kbl]), xs[(q * 32 + kbl) * 32 + lane]);
      }
    }
    const bool last = fp8 ? q == 9 : q == 4;
    if (last) {
      float* rp = red + (uidx & 1) * (W * 32 * 4) + (warp * 32 + lane) * 4;
      *reinterpret_cast<float4*>(rp) = make_float4(acc[0], acc[1], acc[2], acc[3]);
      acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
      pend_fp8 = fp8, pend_unit = unit, pend_par = uidx & 1;
      ++uidx;
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  if (pend_unit >= 0 && tid < 64) reduce_unit(pend_fp8, pend_unit, pend_par);
  pdl_launch_dependents();
}

// moe2v (DS41H_DRAFTER_MOE2_VARIANT=3): moe2's math (FP4 / FP8 kblocks on the f16 hb fragments, per-slot routing weight, slot
// order fixed) with the row-stream structure: CTA c owns output tiles [c 320 / G, (c+1) 320 / G) (2-3 x 16 rows) and streams,
// per slot, its rows of w2 (contiguous: <= 48 rows x 288 B) + scales + the slot's h fragments (9.2 KB, L2) as ONE stage of a
// CTA-wide D-deep cp.async ring; warp w takes (tile w / 5, k32 blocks w % 5 + 5 i) of every stage. moe2 (320 CTAs x 16 rows)
// re-read every slot's h per 16 rows and walked the slots of a warp with ~2 round trips each.
// The shared expert (FP8 w2: 2x the bytes of an FP4 routed row) is streamed as TWO stages (kb halves [0, 9) and [9, 18)) so
// its <= 3 tiles x 16 rows x 288 B per half fit the weight region of a stage (one stage of 576-B rows overflowed it).
constexpr int V2_ROWS = 48, V2_WB = V2_ROWS * (I / 2), V2_SB = V2_ROWS * KB2, V2_HB = KB2 * 32 * 16, V2_SHK = KB2 / 2;
static_assert(KB2 % 2 == 0 && 3 * V2_SHK * 32 * 16 <= V2_WB, "moe2v: a shared-expert half stage fits the weight region");
constexpr int V2_STB = V2_WB + ((V2_SB + 15) / 16) * 16 + V2_HB;  // 13824 + 864 + 9216 = 23904 B
template <int D>
struct V2 {
  static constexpr int SMEM = D * V2_STB + 3 * 5 * 32 * 4 * 4;
};
template <int D>
__global__ void __launch_bounds__(T, 1) moe2v_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem + D * V2_STB);  // [3 tiles][5 warps][32][4]
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x, c = blockIdx.x;
  const int t0 = (c * (H / 16)) / G, t1 = ((c + 1) * (H / 16)) / G, ntl = t1 - t0;  // 16-row tiles of this CTA (<= 3)
  pdl_wait();
  build_slots(a, sl, bits, a.gate_w != nullptr ? a.ids_out : a.ids, a.gate_w != nullptr ? a.wts_out : a.wts);
  const int nsl = sl.n + (a.shared ? 2 : 0);  // the shared expert = stages sl.n (kb [0, 9)) and sl.n + 1 (kb [9, 18))
  constexpr int SBP = ((V2_SB + 15) / 16) * 16;
  auto issue = [&](int s) {
    if (s < nsl) {
      unsigned char* st = smem + (s % D) * V2_STB;
      const bool shd = a.shared && s >= sl.n;
      if (shd) {  // shared expert half: fragment-major [tile][KB2][32] uint4 -> stage [tile][V2_SHK][32]; scales [tile][V2_SHK]
        const int k0 = (s - sl.n) * V2_SHK;
        for (int i = tid; i < ntl * V2_SHK * 32; i += T) {
          const int tl = i / (V2_SHK * 32), r = i - tl * (V2_SHK * 32);
          cp_async16(st + i * 16, a.sh2 + (long(t0 + tl) * KB2 + k0) * 32 + r);
        }
        for (int i = tid; i < ntl * V2_SHK; i += T) {
          const int tl = i / V2_SHK, kb = i - tl * V2_SHK;
          st[V2_WB + i] = a.ss2[long(t0 + tl) * KB2 + k0 + kb];  // 1 B scales (not 4 B aligned)
        }
      } else {   // routed: rows [16 t0, 16 t1) of w2[e] (288 B each, contiguous) + scales s2[e][rows][KB2]
        const long e = sl.e[s];
        const uint8_t* wsrc = a.w2 + e * (long(H) * (I / 2)) + long(16 * t0) * (I / 2);
        const int wbytes = ntl * 16 * (I / 2);
        for (int i = tid; i < wbytes / 16; i += T) cp_async16(st + i * 16, wsrc + i * 16);
        const uint8_t* ssrc = a.s2 + e * (long(H) * KB2) + long(16 * t0) * KB2;
        const int sbytes = ntl * 16 * KB2;
        for (int i = tid; i < sbytes / 4; i += T) cp_async4(st + V2_WB + i * 4, ssrc + i * 4);
      }
      const int hs = shd ? sl.n : s;  // h slot (the shared expert's h is slot sl.n, both halves)
      const uint4* hsrc = reinterpret_cast<const uint4*>(a.hb) + long(hs) * KB2 * 32;  // h fragments (moe13 output, L2)
      for (int i = tid; i < KB2 * 32; i += T) cp_async16(st + V2_WB + SBP + i * 16, hsrc + i);
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int s = 0; s < D - 1; ++s) issue(s);
  const int tl = warp / 5, kq = warp % 5;   // warp -> (tile, k32 blocks kq, kq + 5, kq + 10, kq + 15 < 18)
  const bool live = warp < 15 && tl < ntl;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
  for (int s = 0; s < nsl; ++s) {
    cp_async_wait<D - 2>();
    __syncthreads();
    issue(s + D - 1);
    if (live) {
      const unsigned char* st = smem + (s % D) * V2_STB;
      const uint4* hf = reinterpret_cast<const uint4*>(st + V2_WB + SBP);
      const bool shd = a.shared && s >= sl.n;
      float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (shd) {
        const int k0 = (s - sl.n) * V2_SHK;
        const uint4* wf = reinterpret_cast<const uint4*>(st) + long(tl) * V2_SHK * 32;
        const uint8_t* sf = st + V2_WB + tl * V2_SHK;
        for (int kb = kq; kb < V2_SHK; kb += 5) {
          const float f = sc8(sf[kb]);
          const uint4 xb = g < a.m ? hf[(k0 + kb) * 32 + lane] : make_uint4(0u, 0u, 0u, 0u);
          kblock8(d, wf[kb * 32 + lane], f, f, xb);
        }
      } else {
        const unsigned char* wr = st + (tl * 16) * (I / 2) + 4 * t;
        const uint8_t* sr = st + V2_WB + (tl * 16) * KB2;
        for (int kb = kq; kb < KB2; kb += 5) {
          const unsigned wa = *reinterpret_cast<const unsigned*>(wr + g * (I / 2) + kb * 16);
          const unsigned wb = *reinterpret_cast<const unsigned*>(wr + (g + 8) * (I / 2) + kb * 16);
          const uint4 xb = g < a.m ? hf[kb * 32 + lane] : make_uint4(0u, 0u, 0u, 0u);
          kblock(d, wa, wb, sc(sr[g * KB2 + kb]), sc(sr[(g + 8) * KB2 + kb]), xb);
        }
      }
      // routing weight per token column (moe2: pp = acc * rw[slot][tok]), slots summed in order (shared halves: weight 1)
      const int ws = shd ? sl.n : s;
      acc[0] = __fmaf_rn(d[0], sl.rw[ws][2 * t], acc[0]);
      acc[1] = __fmaf_rn(d[1], sl.rw[ws][2 * t + 1], acc[1]);
      acc[2] = __fmaf_rn(d[2], sl.rw[ws][2 * t], acc[2]);
      acc[3] = __fmaf_rn(d[3], sl.rw[ws][2 * t + 1], acc[3]);
    }
  }
  cp_async_wait<0>();
  if (live) *reinterpret_cast<float4*>(red + ((tl * 5 + kq) * 32 + lane) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (tid < ntl * 128) {  // (tile, row, token) per thread; the 5 k-slice partials in warp order
    const int tt = tid >> 7, e2 = tid & 127, rl = e2 >> 2, r4 = e2 & 3;  // lane rl, register r4
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < 5; ++q) v += red[((tt * 5 + q) * 32 + rl) * 4 + r4];
    const int rg = rl >> 2, rtt = rl & 3;
    const int row = 16 * (t0 + tt) + rg + 8 * (r4 >> 1), tok = 2 * rtt + (r4 & 1);
    if (tok < a.m) a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v);
  }
  pdl_launch_dependents();
}

__global__ void __launch_bounds__(T2, 3) moe2_kernel(const Args a) {
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float part[MAXSLOTS][16 * MAXM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  pdl_wait();
  build_slots(a, sl, bits, a.gate_w != nullptr ? a.ids_out : a.ids, a.gate_w != nullptr ? a.wts_out : a.wts);
  const int nsl = sl.n + (a.shared ? 1 : 0);
  if (a.shared && warp == W2 - 1) {  // shared expert slot (FP8 rows, one 32x32 scale per (tile, kb))
    const int s = sl.n;
    const uint4* wp = a.sh2 + long(blockIdx.x) * KB2 * 32 + lane;
    const uint8_t* sp = a.ss2 + long(blockIdx.x) * KB2;
    const uint4* hb = reinterpret_cast<const uint4*>(a.hb) + long(s) * KB2 * 32 + lane;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr int HK = KB2 / 2;
#pragma unroll
    for (int h0 = 0; h0 < KB2; h0 += HK) {
      uint4 wv[HK], hv[HK];
      unsigned char sv[HK];
#pragma unroll
      for (int q = 0; q < HK; ++q) {
        wv[q] = ld_stream16(wp + (h0 + q) * 32);
        hv[q] = g < a.m ? __ldcg(hb + (h0 + q) * 32) : make_uint4(0u, 0u, 0u, 0u);  // token columns >= m are zero
        sv[q] = __ldg(sp + h0 + q);
      }
#pragma unroll
      for (int q = 0; q < HK; ++q) {
        const float f = sc8(sv[q]);
        kblock8(acc, wv[q], f, f, hv[q]);
      }
    }
    float* pp = part[s];
    pp[g * MAXM + 2 * t] = acc[0] * sl.rw[s][2 * t];
    pp[g * MAXM + 2 * t + 1] = acc[1] * sl.rw[s][2 * t + 1];
    pp[(g + 8) * MAXM + 2 * t] = acc[2] * sl.rw[s][2 * t];
    pp[(g + 8) * MAXM + 2 * t + 1] = acc[3] * sl.rw[s][2 * t + 1];
  }
  const int rw_warps = a.shared ? W2 - 1 : W2;  // warps serving routed slots
  for (int s = warp; warp < rw_warps && s < sl.n; s += rw_warps) {
    const long e = sl.e[s];
    const uint8_t* we = a.w2 + e * (long(H) * (I / 2));
    const uint8_t* se = a.s2 + e * (long(H) * KB2);
    const uint8_t* wa_p = we + long(r0 + g) * (I / 2) + 4 * t;
    const uint8_t* wb_p = we + long(r0 + g + 8) * (I / 2) + 4 * t;
    const uint8_t* sa_p = se + long(r0 + g) * KB2;
    const uint8_t* sb_p = se + long(r0 + g + 8) * KB2;
    const uint4* hb = reinterpret_cast<const uint4*>(a.hb) + long(s) * KB2 * 32 + lane;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr int HK = KB2 / 2;  // two halves of 9 k32 blocks; every load of a half is issued before its math
#pragma unroll
    for (int h0 = 0; h0 < KB2; h0 += HK) {
      unsigned wa[HK], wb[HK];
      uint4 hv[HK];
      unsigned short sa[HK], sb[HK];
#pragma unroll
      for (int q = 0; q < HK; ++q) {
        wa[q] = __ldg(reinterpret_cast<const unsigned*>(wa_p + (h0 + q) * 16));
        wb[q] = __ldg(reinterpret_cast<const unsigned*>(wb_p + (h0 + q) * 16));
        hv[q] = g < a.m ? __ldcg(hb + (h0 + q) * 32) : make_uint4(0u, 0u, 0u, 0u);  // token columns >= m are zero
        sa[q] = sa_p[h0 + q];
        sb[q] = sb_p[h0 + q];
      }
#pragma unroll
      for (int q = 0; q < HK; ++q) kblock(acc, wa[q], wb[q], sc(sa[q]), sc(sb[q]), hv[q]);
    }
    float* pp = part[s];
    pp[g * MAXM + 2 * t] = acc[0] * sl.rw[s][2 * t];
    pp[g * MAXM + 2 * t + 1] = acc[1] * sl.rw[s][2 * t + 1];
    pp[(g + 8) * MAXM + 2 * t] = acc[2] * sl.rw[s][2 * t];
    pp[(g + 8) * MAXM + 2 * t + 1] = acc[3] * sl.rw[s][2 * t + 1];
  }
  __syncthreads();
  if (a.ar_recv[0] == nullptr) {
    if (tid < 16 * MAXM) {
      const int row = tid / MAXM, tok = tid % MAXM;
      if (tok < a.m) {
        float v = 0.0f;
        for (int s = 0; s < nsl; ++s) v += part[s][tid];
        a.y[long(tok) * a.ldy + r0 + row] = __float2bfloat16_rn(v);
      }
    }
  } else {
    // fused all-reduce of this 16-row tile: bf16-round the rank partial (as the unfused path writes y), push it to the
    // 3 peers as 64 LL words (2 bf16 each), then sum the 4 rank partials in rank order in fp32 (ds41h LL all-reduce
    // numerics: bit-identical on every rank). Slots [parity][tile][src][64].
    const unsigned epoch = ld_volatile(a.ar_state);
    const unsigned tag = epoch + 1u == 0u ? 2u : epoch + 1u;
    const int par = int(tag & 1u);
    __shared__ unsigned mine[64];
    if (tid < 64) {  // word w = (row, token pair)
      const int row = tid >> 2, tp = (tid & 3) * 2;
      float v0 = 0.0f, v1 = 0.0f;
      for (int s = 0; s < nsl; ++s) v0 += part[s][row * MAXM + tp], v1 += part[s][row * MAXM + tp + 1];
      const unsigned w = pack_bf16(v0, v1);
      mine[tid] = w;
#pragma unroll
      for (int q = 1; q < RANKS; ++q) {
        const int p = (a.ar_rank + q) % RANKS;
        unsigned long long* dst = a.ar_recv[0];
#pragma unroll
        for (int z = 1; z < RANKS; ++z) dst = p == z ? a.ar_recv[z] : dst;
        st_word(dst + ((long(par) * (H / 16) + blockIdx.x) * RANKS + a.ar_rank) * 64 + tid, (static_cast<unsigned long long>(tag) << 32) | w);
      }
    }
    __syncthreads();
    if (tid < 64) {
      const int row = tid >> 2, tp = (tid & 3) * 2;
      const unsigned long long* box = a.ar_recv[0];
#pragma unroll
      for (int z = 1; z < RANKS; ++z) box = a.ar_rank == z ? a.ar_recv[z] : box;
      float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
      for (int src = 0; src < RANKS; ++src) {
        unsigned w;
        if (src == a.ar_rank) {
          w = mine[tid];
        } else {
          unsigned long long* q = const_cast<unsigned long long*>(box) + ((long(par) * (H / 16) + blockIdx.x) * RANKS + src) * 64 + tid;
          w = poll_word(q, tag, a.timeout, 1);
          st_word(q, 0ull);
        }
        const float2 f = unpack_bf16(w);
        if (src == 0) acc0 = f.x, acc1 = f.y;
        else acc0 = __fadd_rn(acc0, f.x), acc1 = __fadd_rn(acc1, f.y);
      }
      if (tp < a.m) a.y[long(tp) * a.ldy + r0 + row] = __float2bfloat16_rn(acc0);
      if (tp + 1 < a.m) a.y[long(tp + 1) * a.ldy + r0 + row] = __float2bfloat16_rn(acc1);
    }
    // the last CTA to finish advances the epoch (every CTA read it before pushing)
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      if (atomicAdd(a.ar_state + 1, 1u) == gridDim.x - 1) {
        atomicExch(a.ar_state + 1, 0u);
        __threadfence();
        atomicExch(a.ar_state, tag);
      }
    }
  }
  pdl_launch_dependents();
}

// Slot-major moe2 (DS41H_DRAFTER_MOE2_VARIANT=1): CTA (row chunk c of M2S_ROWS rows, slot s): stages h_s (9 KB) in smem ONCE,
// warp w computes 16-row tile w of the chunk for expert s over all 18 kb (every load of the tile issued up front), writes
// part[s][tok][row] = acc * rw; the last CTA of the chunk to arrive (counter) sums the slots in fixed slot order ->
// y bf16 (moe2's reduction order: bitwise identical output). moe2 re-reads every slot's 9 KB h per 16-row CTA
// (320 x nsl x 9 KB of L2 traffic, ~60 MB at k=7) -- here ~nsl x 20 x 9 KB.
constexpr int M2S_ROWS = 256;  // rows per CTA (16 warps x 16)
__global__ void __launch_bounds__(T, 1) moe2s_kernel(const Args a) {
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ uint4 hsm[KB2 * 32];
  __shared__ int last_flag;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int c = blockIdx.x, s = blockIdx.y;
  pdl_wait();
  build_slots(a, sl, bits, a.gate_w != nullptr ? a.ids_out : a.ids, a.gate_w != nullptr ? a.wts_out : a.wts);
  const int nsl = sl.n + (a.shared ? 1 : 0);
  if (s >= nsl) return;  // block-uniform
  for (int i = tid; i < KB2 * 32; i += T) {
    const int l = i & 31;
    hsm[i] = (l >> 2) < a.m ? __ldcg(reinterpret_cast<const uint4*>(a.hb) + long(s) * KB2 * 32 + i) : make_uint4(0u, 0u, 0u, 0u);
  }
  const int tile = c * (M2S_ROWS / 16) + warp;
  const int r0 = tile * 16;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (a.shared && s == sl.n) {  // shared expert (FP8 fragment-major, one scale per (tile, kb))
    const uint4* wp = a.sh2 + long(tile) * KB2 * 32 + lane;
    const uint8_t* sp = a.ss2 + long(tile) * KB2;
    constexpr int HK = KB2 / 2;
    uint4 wv[KB2];
    unsigned char sv[KB2];
#pragma unroll
    for (int q = 0; q < KB2; ++q) wv[q] = ld_stream16(wp + q * 32), sv[q] = __ldg(sp + q);
    __syncthreads();
#pragma unroll
    for (int q = 0; q < KB2; ++q) {
      const float f = sc8(sv[q]);
      kblock8(acc, wv[q], f, f, hsm[q * 32 + lane]);
    }
    (void)HK;
  } else {
    const long e = sl.e[s];
    const uint8_t* we = a.w2 + e * (long(H) * (I / 2));
    const uint8_t* se = a.s2 + e * (long(H) * KB2);
    const uint8_t* wa_p = we + long(r0 + g) * (I / 2) + 4 * t;
    const uint8_t* wb_p = we + long(r0 + g + 8) * (I / 2) + 4 * t;
    const uint8_t* sa_p = se + long(r0 + g) * KB2;
    const uint8_t* sb_p = se + long(r0 + g + 8) * KB2;
    unsigned wa[KB2], wb[KB2];
    unsigned short sa[KB2], sb[KB2];
#pragma unroll
    for (int q = 0; q < KB2; ++q) {
      wa[q] = __ldg(reinterpret_cast<const unsigned*>(wa_p + q * 16));
      wb[q] = __ldg(reinterpret_cast<const unsigned*>(wb_p + q * 16));
      sa[q] = sa_p[q];
      sb[q] = sb_p[q];
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < KB2; ++q) kblock(acc, wa[q], wb[q], sc(sa[q]), sc(sb[q]), hsm[q * 32 + lane]);
  }
  // partials (x routing weight, as moe2's part[s]): [s][tok][row]
  float* ps = a.part + long(s) * MAXM * H;
#pragma unroll
  for (int cc = 0; cc < 2; ++cc) {
    const int tok = 2 * t + cc;
    if (tok < a.m) {
      ps[long(tok) * H + r0 + g] = acc[cc] * sl.rw[s][tok];
      ps[long(tok) * H + r0 + g + 8] = acc[2 + cc] * sl.rw[s][tok];
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const bool last = atomicAdd(a.cnt + c, 1u) == unsigned(nsl - 1);
    if (last) {
      a.cnt[c] = 0u;  // next launch (stream order)
      __threadfence();
    }
    last_flag = last;
  }
  __syncthreads();
  if (last_flag) {
    for (int i = tid; i < a.m * M2S_ROWS; i += T) {
      const int tok = i / M2S_ROWS, row = c * M2S_ROWS + i % M2S_ROWS;
      float v = 0.0f;
      for (int z = 0; z < nsl; ++z) v += __ldcg(a.part + (long(z) * MAXM + tok) * H + row);
      a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v);
    }
  }
  pdl_launch_dependents();
}

// Persistent slot-major moe2 (DS41H_DRAFTER_MOE2_VARIANT=2): G = #SMs CTAs, each CTA = two half-CTAs of 8 warps; work
// items (slot s, 128-row chunk c) are dealt round robin to the 2 G half-CTAs. Per item the half issues its weight loads
// (warp w: tile 8c + w, all 18 kb) BEFORE staging h_s (9 KB, live token lanes only) in its own smem buffer, then
// computes and writes part[s][tok][row] = acc * rw; the last item of a chunk (counter) sums the slots in slot order
// -> y (moe2's order: bitwise == moe2). h L2 traffic = items x 9 KB (moe2: 320 tiles x nsl x 9 KB); no wave quantization.
constexpr int M2R_TILES = 8, M2R_ROWS = M2R_TILES * 16, M2R_CHUNKS = H / M2R_ROWS;
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__global__ void __launch_bounds__(T, 1) moe2r_kernel(const Args a) {
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ uint4 hsm[2][KB2 * 32];
  __shared__ int last_flag[2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int half = warp >> 3, wh = warp & 7, htid = tid & 255;
  pdl_wait();
  build_slots(a, sl, bits, a.gate_w != nullptr ? a.ids_out : a.ids, a.gate_w != nullptr ? a.wts_out : a.wts);
  const int nsl = sl.n + (a.shared ? 1 : 0);
  const int items = nsl * M2R_CHUNKS;
  const int nh = 2 * gridDim.x;
  for (int it = 2 * int(blockIdx.x) + half; it < items; it += nh) {  // half-uniform loop
    const int s = it / M2R_CHUNKS, c = it % M2R_CHUNKS;
    const int tile = c * M2R_TILES + wh;
    const int r0 = tile * 16;
    const bool sh = a.shared && s == sl.n;
    auto stage_h = [&]() {  // h_s -> this half's smem (live token lanes only), then the half's barrier
      for (int i = htid; i < KB2 * 32; i += 256) {
        const int l = i & 31;
        hsm[half][i] = (l >> 2) < a.m ? __ldcg(reinterpret_cast<const uint4*>(a.hb) + long(s) * KB2 * 32 + i) : make_uint4(0u, 0u, 0u, 0u);
      }
      bar_named(1 + half, 256);
    };
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (sh) {  // half-uniform branch: weight loads first, then h staging, then the math
      const uint4* wp = a.sh2 + long(tile) * KB2 * 32 + lane;
      const uint8_t* sp = a.ss2 + long(tile) * KB2;
      uint4 wv8[KB2 / 2];
      unsigned char s8[KB2];
#pragma unroll
      for (int q = 0; q < KB2 / 2; ++q) wv8[q] = ld_stream16(wp + q * 32);
#pragma unroll
      for (int q = 0; q < KB2; ++q) s8[q] = __ldg(sp + q);
      stage_h();
#pragma unroll
      for (int q = 0; q < KB2 / 2; ++q) {
        const float f = sc8(s8[q]);
        kblock8(acc, wv8[q], f, f, hsm[half][q * 32 + lane]);
      }
#pragma unroll
      for (int q = KB2 / 2; q < KB2; ++q) wv8[q - KB2 / 2] = ld_stream16(wp + q * 32);
#pragma unroll
      for (int q = KB2 / 2; q < KB2; ++q) {
        const float f = sc8(s8[q]);
        kblock8(acc, wv8[q - KB2 / 2], f, f, hsm[half][q * 32 + lane]);
      }
    } else {
      const long e = sl.e[s];
      const uint8_t* we = a.w2 + e * (long(H) * (I / 2));
      const uint8_t* se = a.s2 + e * (long(H) * KB2);
      const uint8_t* wa_p = we + long(r0 + g) * (I / 2) + 4 * t;
      const uint8_t* wb_p = we + long(r0 + g + 8) * (I / 2) + 4 * t;
      const uint8_t* sa_p = se + long(r0 + g) * KB2;
      const uint8_t* sb_p = se + long(r0 + g + 8) * KB2;
      unsigned wa[KB2], wb[KB2];
      unsigned short sa[KB2], sb[KB2];
#pragma unroll
      for (int q = 0; q < KB2; ++q) {
        wa[q] = __ldg(reinterpret_cast<const unsigned*>(wa_p + q * 16));
        wb[q] = __ldg(reinterpret_cast<const unsigned*>(wb_p + q * 16));
        sa[q] = sa_p[q];
        sb[q] = sb_p[q];
      }
      stage_h();
#pragma unroll
      for (int q = 0; q < KB2; ++q) kblock(acc, wa[q], wb[q], sc(sa[q]), sc(sb[q]), hsm[half][q * 32 + lane]);
    }
    float* ps = a.part + long(s) * MAXM * H;
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int tok = 2 * t + cc;
      if (tok < a.m) {
        ps[long(tok) * H + r0 + g] = acc[cc] * sl.rw[s][tok];
        ps[long(tok) * H + r0 + g + 8] = acc[2 + cc] * sl.rw[s][tok];
      }
    }
    __threadfence();
    bar_named(1 + half, 256);  // the half's partials written + fenced; hsm[half] free for the next item
    if (htid == 0) {
      const bool last = atomicAdd(a.cnt + c, 1u) == unsigned(nsl - 1);
      if (last) {
        a.cnt[c] = 0u;
        __threadfence();
      }
      last_flag[half] = last;
    }
    bar_named(1 + half, 256);
    if (last_flag[half]) {
      for (int i = htid; i < a.m * M2R_ROWS; i += 256) {
        const int tok = i / M2R_ROWS, row = c * M2R_ROWS + i % M2R_ROWS;
        float v = 0.0f;
        for (int z = 0; z < nsl; ++z) v += __ldcg(a.part + (long(z) * MAXM + tok) * H + row);
        a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v);
      }
    }
  }
  pdl_launch_dependents();
}

// Persistent moe2 (DS41H_DRAFTER_MOE2P): G CTAs (<= #SMs), CTA c owns the 16-row output tiles c, c + G, c + 2G (< 320).
// Each warp loads a slot's h fragments ONCE per k half and applies them to all of its CTA's tiles (moe2 re-reads the
// whole h buffer per 16-row CTA: 320 x nsl x 9 KB ~ 44 MB of L2 traffic per call at nsl = 15; here G x nsl x 9 KB).
// Per warp all tiles' weight loads of a k half are in flight together. Slot partial sums are accumulated per warp in
// registers (slot order fixed) and reduced over the 8 warps in fixed order: deterministic. Same fused all-reduce
// protocol / numerics as moe2 (per tile: bf16 rank partial, 4 rank words summed in rank order).
constexpr int PT = 3;  // max tiles per CTA (320 tiles over >= 107 CTAs)
__global__ void __launch_bounds__(T2, 1) moe2p_kernel(const Args a) {
  __shared__ Slots sl;
  __shared__ unsigned bits[MAXE / 32];
  __shared__ float part[W2][PT][16 * MAXM];
  __shared__ unsigned mine[PT][64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x;
  int nt = 0;
  for (int tt = 0; tt < PT; ++tt) nt += (blockIdx.x + tt * G < H / 16) ? 1 : 0;
  pdl_wait();
  build_slots(a, sl, bits, a.gate_w != nullptr ? a.ids_out : a.ids, a.gate_w != nullptr ? a.wts_out : a.wts);
  const int nsl = sl.n + (a.shared ? 1 : 0);
  float tot[PT][4];
#pragma unroll
  for (int tt = 0; tt < PT; ++tt) tot[tt][0] = tot[tt][1] = tot[tt][2] = tot[tt][3] = 0.0f;
  constexpr int HK = KB2 / 3;  // 3 k batches of 6 kb (PT tiles x 6 kb of weights + scales in flight per batch)
  for (int s = warp; s < nsl; s += W2) {
    float acc[PT][4];
#pragma unroll
    for (int tt = 0; tt < PT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.0f;
    const uint4* hb = reinterpret_cast<const uint4*>(a.hb) + long(s) * KB2 * 32 + lane;
    if (a.shared && s == sl.n) {  // shared expert (FP8 rows, one 32x32 scale per (tile, kb))
#pragma unroll
      for (int h0 = 0; h0 < KB2; h0 += HK) {
        uint4 hv[HK], wv[PT][HK];
        unsigned char sv[PT][HK];
#pragma unroll
        for (int q = 0; q < HK; ++q) hv[q] = g < a.m ? __ldcg(hb + (h0 + q) * 32) : make_uint4(0u, 0u, 0u, 0u);  // token columns >= m are zero
#pragma unroll
        for (int tt = 0; tt < PT; ++tt) {
          if (tt < nt) {
            const long tile = blockIdx.x + tt * G;
            const uint4* wp = a.sh2 + tile * KB2 * 32 + lane;
            const uint8_t* sp = a.ss2 + tile * KB2;
#pragma unroll
            for (int q = 0; q < HK; ++q) {
              wv[tt][q] = ld_stream16(wp + (h0 + q) * 32);
              sv[tt][q] = __ldg(sp + h0 + q);
            }
          }
        }
#pragma unroll
        for (int tt = 0; tt < PT; ++tt) {
          if (tt < nt) {
#pragma unroll
            for (int q = 0; q < HK; ++q) {
              const float f = sc8(sv[tt][q]);
              kblock8(acc[tt], wv[tt][q], f, f, hv[q]);
            }
          }
        }
      }
    } else {
      const long e = sl.e[s];
      const uint8_t* we = a.w2 + e * (long(H) * (I / 2));
      const uint8_t* se = a.s2 + e * (long(H) * KB2);
#pragma unroll
      for (int h0 = 0; h0 < KB2; h0 += HK) {
        uint4 hv[HK];
        unsigned wa[PT][HK], wb[PT][HK];
        unsigned short sa[PT][HK], sb[PT][HK];
#pragma unroll
        for (int q = 0; q < HK; ++q) hv[q] = g < a.m ? __ldcg(hb + (h0 + q) * 32) : make_uint4(0u, 0u, 0u, 0u);  // token columns >= m are zero
#pragma unroll
        for (int tt = 0; tt < PT; ++tt) {
          if (tt < nt) {
            const long r0 = long(blockIdx.x + tt * G) * 16;
            const uint8_t* wa_p = we + (r0 + g) * (I / 2) + 4 * t;
            const uint8_t* wb_p = we + (r0 + g + 8) * (I / 2) + 4 * t;
            const uint8_t* sa_p = se + (r0 + g) * KB2;
            const uint8_t* sb_p = se + (r0 + g + 8) * KB2;
#pragma unroll
            for (int q = 0; q < HK; ++q) {
              wa[tt][q] = __ldg(reinterpret_cast<const unsigned*>(wa_p + (h0 + q) * 16));
              wb[tt][q] = __ldg(reinterpret_cast<const unsigned*>(wb_p + (h0 + q) * 16));
              sa[tt][q] = sa_p[h0 + q];
              sb[tt][q] = sb_p[h0 + q];
            }
          }
        }
#pragma unroll
        for (int tt = 0; tt < PT; ++tt) {
          if (tt < nt) {
#pragma unroll
            for (int q = 0; q < HK; ++q) kblock(acc[tt], wa[tt][q], wb[tt][q], sc(sa[tt][q]), sc(sb[tt][q]), hv[q]);
          }
        }
      }
    }
    const float w0 = sl.rw[s][2 * t], w1 = sl.rw[s][2 * t + 1];
#pragma unroll
    for (int tt = 0; tt < PT; ++tt) {
      tot[tt][0] = __fmaf_rn(acc[tt][0], w0, tot[tt][0]);
      tot[tt][1] = __fmaf_rn(acc[tt][1], w1, tot[tt][1]);
      tot[tt][2] = __fmaf_rn(acc[tt][2], w0, tot[tt][2]);
      tot[tt][3] = __fmaf_rn(acc[tt][3], w1, tot[tt][3]);
    }
  }
#pragma unroll
  for (int tt = 0; tt < PT; ++tt) {
    float* pp = part[warp][tt];
    pp[g * MAXM + 2 * t] = tot[tt][0];
    pp[g * MAXM + 2 * t + 1] = tot[tt][1];
    pp[(g + 8) * MAXM + 2 * t] = tot[tt][2];
    pp[(g + 8) * MAXM + 2 * t + 1] = tot[tt][3];
  }
  __syncthreads();
  if (a.ar_recv[0] == nullptr) {
    for (int i = tid; i < nt * 16 * MAXM; i += T2) {
      const int tt = i / (16 * MAXM), rr = i % (16 * MAXM), row = rr / MAXM, tok = rr % MAXM;
      if (tok < a.m) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < W2; ++w) v += part[w][tt][rr];
        a.y[long(tok) * a.ldy + (long(blockIdx.x + tt * G) * 16) + row] = __float2bfloat16_rn(v);
      }
    }
  } else {
    const unsigned epoch = ld_volatile(a.ar_state);
    const unsigned tag = epoch + 1u == 0u ? 2u : epoch + 1u;
    const int par = int(tag & 1u);
    if (tid < nt * 64) {  // word (tile tt, row, token pair)
      const int tt = tid >> 6, wd = tid & 63, row = wd >> 2, tp = (wd & 3) * 2;
      const long tile = blockIdx.x + tt * G;
      float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
      for (int w = 0; w < W2; ++w) v0 += part[w][tt][row * MAXM + tp], v1 += part[w][tt][row * MAXM + tp + 1];
      const unsigned wv = pack_bf16(v0, v1);
      mine[tt][wd] = wv;
#pragma unroll
      for (int q = 1; q < RANKS; ++q) {
        const int p = (a.ar_rank + q) % RANKS;
        unsigned long long* dst = a.ar_recv[0];
#pragma unroll
        for (int z = 1; z < RANKS; ++z) dst = p == z ? a.ar_recv[z] : dst;
        st_word(dst + ((long(par) * (H / 16) + tile) * RANKS + a.ar_rank) * 64 + wd, (static_cast<unsigned long long>(tag) << 32) | wv);
      }
    }
    __syncthreads();
    if (tid < nt * 64) {
      const int tt = tid >> 6, wd = tid & 63, row = wd >> 2, tp = (wd & 3) * 2;
      const long tile = blockIdx.x + tt * G;
      const unsigned long long* box = a.ar_recv[0];
#pragma unroll
      for (int z = 1; z < RANKS; ++z) box = a.ar_rank == z ? a.ar_recv[z] : box;
      float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
      for (int src = 0; src < RANKS; ++src) {
        unsigned w;
        if (src == a.ar_rank) {
          w = mine[tt][wd];
        } else {
          unsigned long long* q = const_cast<unsigned long long*>(box) + ((long(par) * (H / 16) + tile) * RANKS + src) * 64 + wd;
          w = poll_word(q, tag, a.timeout, 1);
          st_word(q, 0ull);
        }
        const float2 f = unpack_bf16(w);
        if (src == 0) acc0 = f.x, acc1 = f.y;
        else acc0 = __fadd_rn(acc0, f.x), acc1 = __fadd_rn(acc1, f.y);
      }
      const long r = tile * 16 + row;
      if (tp < a.m) a.y[long(tp) * a.ldy + r] = __float2bfloat16_rn(acc0);
      if (tp + 1 < a.m) a.y[long(tp + 1) * a.ldy + r] = __float2bfloat16_rn(acc1);
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      if (atomicAdd(a.ar_state + 1, 1u) == gridDim.x - 1) {
        atomicExch(a.ar_state + 1, 0u);
        __threadfence();
        atomicExch(a.ar_state, tag);
      }
    }
  }
  pdl_launch_dependents();
}
}  // namespace dmoe


// ================================================================== FP8 draft head (M <= 8 rows, per-row scales)
// y[t, n] = bf16(scale[n] * sum_k w[n, k] x[t, k]),  w e4m3 [N, K] with one fp32 scale per row (the bf16 lm_head shard
// quantized per row for DRAFTING only). Weights in skinny fragment-major layout (16-row tile rt, k32 block kb ->
// 512 contiguous bytes, lane l = 4g + t holds W[16rt+g][32kb+8t..+8] then W[16rt+g+8][32kb+8t..+8]), so every lane
// issues one coalesced 16-byte load per k32 block. One warp per 16-row tile over the full K (HMMA accumulates
// across all K: the weight scale is per row and x is staged with one power-of-two scale per token).
namespace dhead {
constexpr int K = 5120, KB = K / 32, T = 512, W = T / 32, MAXM = 8;

struct Args {
  const __nv_bfloat16* x;  // [m, ldx]
  long ldx;
  const uint4* w;          // fragment-major e4m3
  const float* scale;      // [n]
  __nv_bfloat16* y;        // [m, ldy]
  long ldy;
  const __nv_bfloat16* nw;  // optional fused RMSNorm weight [K] (x is then the un-normalized hidden state)
  float eps;
  int m, n;
};

__global__ void __launch_bounds__(T, 1) head_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB][32] B fragments (natural k order)
  __shared__ float xsc[MAXM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tiles = a.n / 16;
  const int tile = blockIdx.x * W + warp;
  // weights do not depend on upstream kernels: start the first loads before the PDL wait
  const uint4* wp = a.w + long(tile < tiles ? tile : 0) * KB * 32 + lane;
  uint4 wv0[8], wv1[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) wv0[q] = ld_stream16(wp + q * 32);
#pragma unroll
  for (int q = 0; q < 8; ++q) wv1[q] = ld_stream16(wp + (8 + q) * 32);  // 16 kb in flight across the PDL wait + x staging
  pdl_wait();
  if (a.nw != nullptr) {
    __shared__ float rs[MAXM];
    stage_x_norm<K, T>(a.x, a.ldx, a.m, a.nw, a.eps, xs, xsc, rs);
  } else {
    stage_x<K, T, false>(a.x, a.ldx, a.m, xs, xsc);
  }
  if (tile < tiles) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto step = [&](const uint4 wv, int kb) {
      const uint4 xb = xs[kb * 32 + lane];
      // A: rows g (wv.x, wv.y) and g+8 (wv.z, wv.w); physical k 8t..8t+7 = HMMA#1 slots (k0..k3), HMMA#2 (k4..k7)
      const unsigned a0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x & 0xffffu));
      const unsigned a2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x >> 16));
      const unsigned a1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z & 0xffffu));
      const unsigned a3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z >> 16));
      const unsigned c0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y & 0xffffu));
      const unsigned c2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y >> 16));
      const unsigned c1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w & 0xffffu));
      const unsigned c3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w >> 16));
      dmoe::hmma_acc(acc, a0, a1, a2, a3, xb.x, xb.y);
      dmoe::hmma_acc(acc, c0, c1, c2, c3, xb.z, xb.w);
    };
#pragma unroll 1
    for (int kb = 0; kb < KB; kb += 16) {  // double-buffered 8-kb chunks (KB % 16 == 0); both preloaded at entry
#pragma unroll
      for (int q = 0; q < 8; ++q) step(wv0[q], kb + q);
      if (kb + 16 < KB) {
#pragma unroll
        for (int q = 0; q < 8; ++q) wv0[q] = ld_stream16(wp + (kb + 16 + q) * 32);
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) step(wv1[q], kb + 8 + q);
      if (kb + 24 < KB) {
#pragma unroll
        for (int q = 0; q < 8; ++q) wv1[q] = ld_stream16(wp + (kb + 24 + q) * 32);
      }
    }
    const int r0 = tile * 16;
    const float s0 = a.scale[r0 + g], s1 = a.scale[r0 + g + 8];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = 2 * t + c;
      if (tok < a.m) {
        a.y[long(tok) * a.ldy + r0 + g] = __float2bfloat16_rn(acc[c] * xsc[tok] * s0);
        a.y[long(tok) * a.ldy + r0 + g + 8] = __float2bfloat16_rn(acc[2 + c] * xsc[tok] * s1);
      }
    }
  }
  pdl_launch_dependents();
}

// head_rows_kernel (DS41H_DRAFT_HEAD_ROWS=1, drafter.ext().head_set_rows): the same FP8 draft head math with the vrh row-stream
// structure: CTA c owns tiles [c tiles / G, (c+1) tiles / G) (contiguous 80-KB fragment-major tiles), streamed as 16-KB stages
// (32 kb of one tile) through a CTA-wide D-deep cp.async ring; warp w takes k32 blocks 2w, 2w + 1 of every stage; the 16 warp
// partials of a tile are summed in warp order (double-buffered smem) one stage later by warps 0-3. Deterministic.
constexpr int R_STAGE = 32 * 32 * 16;  // 32 kb x 32 lanes x 16 B
template <int D>
__global__ void __launch_bounds__(T, 1) head_rows_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);                         // [KB][32]
  unsigned char* ring = smem + KB * 32 * 16;                          // [D][R_STAGE]
  float* red = reinterpret_cast<float*>(ring + D * R_STAGE);          // [2][W][32][4]
  __shared__ float xsc[MAXM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x, c = blockIdx.x;
  const int tiles = a.n / 16;
  const int tl0 = int((long(c) * tiles) / G), tl1 = int((long(c + 1) * tiles) / G);
  const int nst = (tl1 - tl0) * (KB / 32);  // 5 stages per tile
  auto issue = [&](int j) {
    if (j < nst) {
      const int tile = tl0 + j / (KB / 32), q = j % (KB / 32);
      const uint4* src = a.w + (long(tile) * KB + q * 32) * 32;
      unsigned char* dst = ring + (j % D) * R_STAGE;
      cp_async16(dst + tid * 16, src + tid);
      cp_async16(dst + (tid + T) * 16, src + tid + T);
    }
    cp_async_commit();
  };
#pragma unroll 1
  for (int j = 0; j < D - 1; ++j) issue(j);  // weights do not depend on upstream kernels: in flight across the PDL wait
  pdl_wait();
  if (a.nw != nullptr) {
    __shared__ float rs[MAXM];
    stage_x_norm<K, T>(a.x, a.ldx, a.m, a.nw, a.eps, xs, xsc, rs);
  } else {
    stage_x<K, T, false>(a.x, a.ldx, a.m, xs, xsc);
  }
  auto finish = [&](int tile, int par) {  // tid < 128: element tid of the 16 warp partials (lane tid / 4, register tid % 4)
    const float* rp = red + par * (W * 128) + tid;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) v += rp[w * 128];
    const int rl = tid >> 2, r4 = tid & 3, rg = rl >> 2, rt = rl & 3;
    const int row = tile * 16 + rg + 8 * (r4 >> 1), tok = 2 * rt + (r4 & 1);
    if (tok < a.m) a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v * xsc[tok] * a.scale[row]);
  };
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int pend = -1, ppar = 0, tcount = 0;
#pragma unroll 1
  for (int j = 0; j < nst; ++j) {
    cp_async_wait<D - 2>();
    __syncthreads();
    if (pend >= 0) {
      if (tid < 128) finish(pend, ppar);
      pend = -1;
    }
    issue(j + D - 1);
    const int tile = tl0 + j / (KB / 32), q = j % (KB / 32);
    const uint4* st = reinterpret_cast<const uint4*>(ring + (j % D) * R_STAGE);
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int kbl = 2 * warp + z, kb = q * 32 + kbl;
      const uint4 wv = st[kbl * 32 + lane];
      const uint4 xb = xs[kb * 32 + lane];
      const unsigned a0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x & 0xffffu));
      const unsigned a2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x >> 16));
      const unsigned a1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z & 0xffffu));
      const unsigned a3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z >> 16));
      const unsigned c0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y & 0xffffu));
      const unsigned c2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y >> 16));
      const unsigned c1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w & 0xffffu));
      const unsigned c3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w >> 16));
      dmoe::hmma_acc(acc, a0, a1, a2, a3, xb.x, xb.y);
      dmoe::hmma_acc(acc, c0, c1, c2, c3, xb.z, xb.w);
    }
    if (q == KB / 32 - 1) {
      *reinterpret_cast<float4*>(red + (tcount & 1) * (W * 128) + (warp * 32 + lane) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
      acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
      pend = tile, ppar = tcount & 1;
      ++tcount;
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  if (pend >= 0 && tid < 128) finish(pend, ppar);
  pdl_launch_dependents();
}
}  // namespace dhead


// ================================================================== fused vocab-parallel block verification
// One persistent launch replacing vp_sampling.verify (stage, _target_stats_kernel, LL gather, _target_residual_kernel,
// LL gather, _target_reject_kernel). Same math (stock rejection_sampler_utils.py block verification + residual
// resampling, gumbel.py / Philox RNG), same tie rules (lowest token wins, NaN -> -inf in argmaxes); fp32 reduction
// trees differ (log-sum-exp / residual-mass merge order). Two exchange rounds through LL slots (same buffer geometry
// as draft_markov; phase 0: per-row target stats, phase 1: per-row residual mass + residual Gumbel argmax), then CTA 0
// runs the per-request rejection decision (one thread per request).
namespace dver {
constexpr int T = 256, W = T / 32;
constexpr int MAXROWS = 64, MAXREQS = 64;
constexpr int VR = 8;  // rows per block (batched per-row reductions held in registers)

struct Args {
  unsigned long long* recv[RANKS];
  unsigned* state;  // [0] epoch, [1] phase-0 counter, [2] phase-1 counter
  float* cand;      // [2][slot_rows][G][3]
  long slot_rows;
  const __nv_bfloat16* logits;  // [rows, ld] this rank's target logits (unprocessed)
  long ld;
  const int* draft_sampled;     // [rows] (-1 placeholders)
  const long long* pos;         // [rows]
  const int* exp_idx;           // [rows]
  const int* local_pos;         // [rows]
  const int* cu;                // [reqs + 1]
  const int* idx_mapping;       // [reqs]
  const float* temp;            // [max_reqs] (may be host-mapped)
  const long long* seeds;       // [max_reqs]
  const __nv_bfloat16* cache;   // draft logit cache [max_reqs][cs0 / cs1][local]
  long cs0, cs1;
  const float* vp_logit;        // [max_reqs + 1, vp_stride]
  const float* vp_lse;
  long vp_stride;
  long long* sampled;           // [reqs, sstride]
  long sstride;
  int* num_sampled;             // [reqs]
  unsigned long long timeout;
  unsigned long long* trace;    // optional %globaltimer trace: CTA 0 slots [0, 16), last-CTA merge [16, 20)
  int rank, rows, reqs, local, vocab_start, chunk, nspec;
};

__device__ __forceinline__ float texp(float x) {  // Triton tl.exp (fp32): ex2.approx(x * log2 e)
  float y;
  const float z = __fmul_rn(x, 1.44269502162933349609375f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(z));
  return y;
}
__device__ __forceinline__ float nan_inf(float v) { return v != v ? -INFINITY : v; }
// triton.language.random.rand (Philox 4x32-10, c0 = offset lo, c1 = offset hi), then tl_rand32(includes_zero=False)
__device__ __forceinline__ float tl_rand32(long long seed, long long offset) {
  unsigned k0 = unsigned(static_cast<unsigned long long>(seed) & 0xffffffffull), k1 = unsigned(static_cast<unsigned long long>(seed) >> 32);
  unsigned c0 = unsigned(offset), c1 = unsigned(offset >> 32), c2 = 0u, c3 = 0u;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned _c0 = c0, _c2 = c2;
    c0 = __umulhi(0xCD9E8D57u, _c2) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53u, _c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * _c2;
    c3 = 0xD2511F53u * _c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  int x = int(c0);
  x = x < 0 ? ~x : x;
  const float u = __fmul_rn(float(x), 4.6566127342e-10f);
  return fmaxf(u, 4.6566127342e-10f);
}

struct RowMeta {
  int req, step, dnext, next_pos;
  float temp;
  long long seed, pos;
};
struct Stat {  // phase 0 (greedy row: m = max value, s = argmax token) / phase 1 (m = mass, s = value, o = token)
  float m, s, o;
};

// (m, s) log-sum-exp pair merge with Triton exp; greedy (value, token) merge; owned = max
__device__ __forceinline__ void ms_merge(float& m, float& s, float m2, float s2) {
  if (!(m2 > -INFINITY)) return;
  if (!(m > -INFINITY)) {
    m = m2, s = s2;
  } else if (m2 > m) {
    s = s * texp(m - m2) + s2;
    m = m2;
  } else {
    s = s + s2 * texp(m2 - m);
  }
}
__device__ __forceinline__ void p0_merge(bool greedy, Stat& a, const Stat& b) {
  if (greedy) {
    const float bm = nan_inf(b.m);
    if (bm > a.m || (bm == a.m && b.s < a.s)) a.m = bm, a.s = b.s;
  } else {
    ms_merge(a.m, a.s, b.m, b.s);
  }
  a.o = fmaxf(a.o, b.o);
}
__device__ __forceinline__ void p1_merge(Stat& a, const Stat& b) {
  a.m += b.m;
  const float bv = nan_inf(b.s);
  if (bv > a.s || (bv == a.s && b.o < a.o)) a.s = bv, a.o = b.o;
}
__device__ __forceinline__ Stat shfl_stat(const Stat& x, int o) {
  return Stat{__shfl_xor_sync(0xffffffffu, x.m, o), __shfl_xor_sync(0xffffffffu, x.s, o), __shfl_xor_sync(0xffffffffu, x.o, o)};
}
__device__ __forceinline__ Stat p0_init(bool greedy) { return Stat{-INFINITY, greedy ? 3.0e38f : 0.0f, -INFINITY}; }
__device__ __forceinline__ Stat p1_init() { return Stat{0.0f, -INFINITY, 3.0e38f}; }

__device__ __forceinline__ unsigned long long* vrecv(const Args& a, int p) {
  unsigned long long* r = a.recv[0];
#pragma unroll
  for (int q = 1; q < RANKS; ++q) r = p == q ? a.recv[q] : r;
  return r;
}
__device__ __forceinline__ long vslot(const Args& a, int par, int phase, int src, int r) {
  return ((((long(par) * 2 + phase) * RANKS + src) * a.slot_rows + r) * WORDS);
}

// log p(x) - log q(x) of the draft token predicted by row q (request rows start .. ; drafted index q - start)
__device__ __forceinline__ float ratio_term(const Args& a, const Stat* g1, int req, int start, int q, float temp) {
  const float lse = g1[q].m + logf(g1[q].s);
  const long vi = long(req) * a.vp_stride + (q - start);
  const float d_lp = __fsub_rn(div_full(a.vp_logit[vi], temp), a.vp_lse[vi]);
  return g1[q].o - lse - d_lp;
}

// Block reduction of one Stat per thread with merge f; result in thread 0.
template <class F>
__device__ __forceinline__ Stat block_stat(Stat x, Stat* red, F f) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) f(x, shfl_stat(x, o));
  __syncthreads();
  if (lane == 0) red[warp] = x;
  __syncthreads();
  if (threadIdx.x == 0)
    for (int w = 1; w < W; ++w) f(x, red[w]);
  return x;
}

__global__ void __launch_bounds__(T, 1) verify_kernel(const Args a) {
  __shared__ RowMeta meta[MAXROWS];
  __shared__ Stat g1[MAXROWS];
  __shared__ Stat g2[MAXROWS];
  __shared__ Stat red2[W][VR];
  __shared__ float rowmax[VR];
  __shared__ float rowsum[W][VR];
  __shared__ float logp[MAXROWS];
  __shared__ float term[MAXROWS];
  __shared__ float rlse[MAXROWS];    // phase 1: target lse of row r (sampled rows)
  __shared__ float rdlse[MAXROWS];   // phase 1: draft lse of row r's drafted step
  __shared__ float urnd[MAXROWS];    // CTA 0: acceptance uniform of row r (Philox, computed while the residual exchange is in flight)
  __shared__ int last_flag;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cta = blockIdx.x, G = gridDim.x;
  const int v0 = cta * a.chunk, nv = min(a.chunk, a.local - v0);
  const int v = v0 + tid;
  const bool live = tid < nv;
  auto trace = [&](int slot) {
    if (a.trace != nullptr && tid == 0 && cta == 0) a.trace[slot] = globaltimer();
  };
  // phase-1 Gumbel noise of this thread's token for the rows of the first row block (depends only on (seed, pos,
  // token): computed while the phase-0 exchange is in flight)
  float gz[VR];
  pdl_wait();
  trace(0);
  const unsigned epoch = ld_volatile(a.state);
  const int par = int(epoch & 1u);
  for (int r = tid; r < a.rows; r += T) {
    RowMeta mt;
    mt.req = a.exp_idx[r];
    mt.step = a.local_pos[r];
    mt.dnext = r + 1 < a.rows ? a.draft_sampled[r + 1] : -1;
    mt.next_pos = r + 1 < a.rows ? a.local_pos[r + 1] : 0;
    mt.temp = a.temp[mt.req];
    mt.seed = a.seeds[mt.req];
    mt.pos = a.pos[r];
    meta[r] = mt;
  }
  __syncthreads();
  trace(1);

  for (int phase = 0; phase < 2; ++phase) {
    const unsigned tag = (epoch << 4) | unsigned(phase + 1);
    if (phase == 1) {
      // ratio term of every row's drafted token (row r scores the draft at r + 1), all rows in parallel (one round
      // trip), then the per-row prefix log p = cumulative min(0, sum of the request's earlier terms) from smem
      for (int r = tid; r < a.rows; r += T) {
        const RowMeta mt = meta[r];
        float tm = 0.0f;
        const float lse = mt.temp != 0.0f ? g1[r].m + logf(g1[r].s) : 0.0f;
        rlse[r] = lse;
        rdlse[r] = mt.step < a.nspec && mt.req >= 0 ? a.vp_lse[long(mt.req) * a.vp_stride + mt.step] : 0.0f;
        if (mt.temp != 0.0f && mt.dnext >= 0 && r + 1 < a.rows && mt.next_pos != 0 && mt.step < a.nspec) {
          const long vi = long(mt.req) * a.vp_stride + mt.step;
          tm = g1[r].o - lse - __fsub_rn(div_full(a.vp_logit[vi], mt.temp), a.vp_lse[vi]);
        }
        term[r] = tm;
      }
      __syncthreads();
      for (int r = tid; r < a.rows; r += T) {
        const RowMeta mt = meta[r];
        const int start = r - mt.step;
        float log_p = 0.0f;
        if (mt.temp != 0.0f)
          for (int s = 0; s < mt.step; ++s)
            if (meta[start + s].dnext >= 0) log_p = fminf(log_p + term[start + s], 0.0f);
        logp[r] = log_p;
      }
      __syncthreads();
    }
    // ---------------- local candidates of this CTA's vocab slice, VR rows at a time: all loads of a row block in
    // flight at once, one batched block reduction per row block
    for (int rb = 0; rb < a.rows; rb += VR) {
    float xr[VR], dc[VR];
#pragma unroll
    for (int i = 0; i < VR; ++i) {
      const int r = rb + i;
      xr[i] = -INFINITY;
      dc[i] = -INFINITY;
      if (r < a.rows && live) {
        const RowMeta mt = meta[r];
        xr[i] = __bfloat162float(a.logits[long(r) * a.ld + v]);
        if (phase == 1 && mt.step < a.nspec && mt.req >= 0)
          dc[i] = __bfloat162float(a.cache[long(mt.req) * a.cs0 + long(mt.step) * a.cs1 + v]);
      }
    }
    Stat c[VR];
#pragma unroll
    for (int i = 0; i < VR; ++i) {
      const int r = rb + i;
      if (r >= a.rows) {
        c[i] = p1_init();
        continue;
      }
      const RowMeta mt = meta[r];
      const bool greedy = mt.temp == 0.0f;
      const float t = mt.temp > 0.0f ? mt.temp : 1.0f;
      if (phase == 0) {
        c[i] = p0_init(greedy);
        if (live) {
          const float x = div_full(xr[i], t);
          if (greedy) c[i].m = nan_inf(x), c[i].s = float(a.vocab_start + v);
          else c[i].m = x, c[i].s = x > -INFINITY ? 1.0f : 0.0f;
          if (mt.dnext - a.vocab_start == v) c[i].o = x;  // owned draft-token logit / T
        }
      } else {
        c[i] = p1_init();
        const bool has_next = r + 1 < a.rows;
        const bool is_bonus = !has_next || mt.next_pos == 0;
        const bool not_bonus = has_next && mt.next_pos != 0;
        const float log_p = logp[r];
        if (live) {
          const float x = div_full(xr[i], t);
          const bool use_draft = !greedy && not_bonus && mt.dnext >= 0;
          const float lse = rlse[r];  // = greedy ? 0 : g1[r].m + logf(g1[r].s)
          const float t_lp = x - lse;
          float d_lp = -INFINITY;
          if (use_draft) d_lp = __fsub_rn(div_full(dc[i], mt.temp), rdlse[r]);
          if (use_draft && mt.step >= 1 && mt.step < a.nspec) c[i].m = fmaxf(texp(log_p) * texp(t_lp) - texp(d_lp), 0.0f);
          const int tok = a.vocab_start + v;
          c[i].o = float(tok);
          if (!greedy || is_bonus) {
            float residual = x;
            if (use_draft) {
              const float shifted = t_lp + log_p;
              const float ratio = texp(d_lp - shifted);
              residual = ratio < 1.0f ? shifted + log1pf(-ratio) : -INFINITY;
            }
            float val = residual;
            if (!greedy) {  // APPLY_TEMPERATURE=False, IS_DRAFTING=False: residual + Gumbel(seed, pos, token)
              const float noise = rb == 0 ? gz[i] : gumbel_noise_raw(mt.seed, mt.pos, unsigned(tok));
              val = __fadd_rn(residual, noise);
            }
            c[i].s = nan_inf(val);
          }
        }
      }
    }
    // pass 1: argmax / max reductions only (no exp in the shuffle chains; the window runs at the capped SM clock)
#pragma unroll
    for (int i = 0; i < VR; ++i) {
      const int r = rb + i;
      if (r >= a.rows) continue;  // block-uniform
      const bool greedy = meta[r].temp == 0.0f;
      Stat x = c[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const Stat y = shfl_stat(x, o);
        if (phase == 1) p1_merge(x, y);
        else if (greedy) p0_merge(true, x, y);
        else x.m = fmaxf(x.m, y.m), x.o = fmaxf(x.o, y.o);  // sum-exp in pass 2
      }
      if (lane == 0) red2[warp][i] = x;
    }
    __syncthreads();
    if (phase == 0) {
      // pass 2 (sampled rows): CTA max M, then sum exp(x - M) with one texp per thread and row
      for (int i = tid; i < VR && rb + i < a.rows; i += T) {
        float mx = red2[0][i].m;
        for (int w = 1; w < W; ++w) mx = fmaxf(mx, red2[w][i].m);
        rowmax[i] = mx;
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < VR; ++i) {
        const int r = rb + i;
        if (r >= a.rows) continue;
        if (meta[r].temp == 0.0f) continue;
        float e = (c[i].m > -INFINITY) ? texp(c[i].m - rowmax[i]) : 0.0f;
        e = warp_sum(e);
        if (lane == 0) rowsum[warp][i] = e;
      }
      __syncthreads();
    }
    for (int i = tid; i < VR && rb + i < a.rows; i += T) {
      const int r = rb + i;
      const bool greedy = meta[r].temp == 0.0f;
      Stat x = red2[0][i];
      for (int w = 1; w < W; ++w) {
        if (phase == 1) p1_merge(x, red2[w][i]);
        else if (greedy) p0_merge(true, x, red2[w][i]);
        else x.o = fmaxf(x.o, red2[w][i].o);
      }
      if (phase == 0 && !greedy) {
        x.m = rowmax[i];
        float tot = 0.0f;
        for (int w = 0; w < W; ++w) tot += rowsum[w][i];
        x.s = x.m > -INFINITY ? tot : 0.0f;
      }
      float* o = a.cand + ((long(phase) * a.slot_rows + r) * G + cta) * 3;
      o[0] = x.m;
      o[1] = x.s;
      o[2] = x.o;
    }
    __syncthreads();
    }  // row blocks
    // ---------------- arrival; the last CTA merges this rank's candidates and pushes them to every rank
    trace(2 + 4 * phase);
    if (tid == 0) {
      __threadfence();
      last_flag = atomicAdd(a.state + 1 + phase, 1u) == unsigned(G - 1);
    }
    __syncthreads();
    trace(3 + 4 * phase);
    if (last_flag) {
      if (a.trace != nullptr && tid == 0) a.trace[16 + 2 * phase] = globaltimer();
      __threadfence();
      for (int r = warp; r < a.rows; r += W) {
        const bool greedy = meta[r].temp == 0.0f;
        const float* base = a.cand + (long(phase) * a.slot_rows + r) * G * 3;
        Stat x = phase == 0 ? p0_init(greedy) : p1_init();
        for (int gi = lane; gi < G; gi += 32) {
          const Stat y{__ldcg(base + gi * 3), __ldcg(base + gi * 3 + 1), __ldcg(base + gi * 3 + 2)};
          if (phase == 0) p0_merge(greedy, x, y);
          else p1_merge(x, y);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          const Stat y = shfl_stat(x, o);
          if (phase == 0) p0_merge(greedy, x, y);
          else p1_merge(x, y);
        }
        if (lane < RANKS * 3) {
          const int p = lane / 3, wd = lane % 3;
          const float val = wd == 0 ? x.m : (wd == 1 ? x.s : x.o);
          st_word(vrecv(a, p) + vslot(a, par, phase, a.rank, r) + wd, (static_cast<unsigned long long>(tag) << 32) | __float_as_uint(val));
        }
      }
      if (tid == 0) {
        a.state[1 + phase] = 0u;
        if (phase == 1) {
          __threadfence();
          atomicExch(a.state, epoch + 1u == 0u ? 1u : epoch + 1u);  // every CTA read the epoch before phase 0
        }
        if (a.trace != nullptr) a.trace[17 + 2 * phase] = globaltimer();
      }
    }
    if (phase == 1 && cta != 0) break;  // only CTA 0 needs the residual words
    // off the critical path while the exchange is in flight: phase-1 noise (phase 0) / acceptance uniforms (phase 1)
    if (phase == 0) {
#pragma unroll
      for (int i = 0; i < VR; ++i) {
        gz[i] = 0.0f;
        if (i < a.rows && live && meta[i].temp != 0.0f) gz[i] = gumbel_noise_raw(meta[i].seed, meta[i].pos, unsigned(a.vocab_start + v));
      }
    } else {
      for (int r = tid; r < a.rows; r += T) urnd[r] = tl_rand32(meta[r].seed, meta[r].pos);
    }
    // ---------------- poll the 4 rank words of every row and merge them in rank order
    const unsigned long long* box = vrecv(a, a.rank);
    for (int r = warp; r < a.rows; r += W) {
      const bool greedy = meta[r].temp == 0.0f;
      float x = 0.0f;
      if (lane < RANKS * 3) x = __uint_as_float(poll_word(box + vslot(a, par, phase, lane / 3, r) + lane % 3, tag, a.timeout, 2));
      Stat acc = phase == 0 ? p0_init(greedy) : p1_init();
      if (phase == 0 && !greedy) {
        // vp_sampling._logsumexp over ranks: top + log(sum s_p exp(m_p - top)) (stored as (top, total))
        float m[RANKS], sm[RANKS], top = -INFINITY;
#pragma unroll
        for (int p = 0; p < RANKS; ++p) {
          m[p] = __shfl_sync(0xffffffffu, x, 3 * p), sm[p] = __shfl_sync(0xffffffffu, x, 3 * p + 1);
          acc.o = fmaxf(acc.o, __shfl_sync(0xffffffffu, x, 3 * p + 2));
          top = fmaxf(top, m[p]);
        }
        float total = 0.0f;
#pragma unroll
        for (int p = 0; p < RANKS; ++p) total += m[p] > -INFINITY ? sm[p] * texp(m[p] - top) : 0.0f;
        acc.m = top, acc.s = total;
      } else {
#pragma unroll
        for (int p = 0; p < RANKS; ++p) {
          const Stat y{__shfl_sync(0xffffffffu, x, 3 * p), __shfl_sync(0xffffffffu, x, 3 * p + 1), __shfl_sync(0xffffffffu, x, 3 * p + 2)};
          if (phase == 0) p0_merge(greedy, acc, y);
          else p1_merge(acc, y);
        }
      }
      if (lane == 0) (phase == 0 ? g1 : g2)[r] = acc;
    }
    __syncthreads();
    trace(4 + 4 * phase);
  }
  if (cta != 0) return;
  // ---------------- per-request rejection (_target_reject_kernel semantics), one thread per request
  for (int rq = tid; rq < a.reqs; rq += T) {
    const int start = a.cu[rq], end = a.cu[rq + 1];
    const int num_draft = end - start - 1;
    const float temp = meta[start].temp;  // = temp[idx_mapping[rq]] (rows of a request carry its state index)
    const bool greedy = temp == 0.0f;
    long long* out = a.sampled + long(rq) * a.sstride;
    int accepted = 0;
    bool verifying = num_draft >= 0;
    float log_p = 0.0f;
    for (int i = 0; i < num_draft; ++i) {
      const int raw = meta[start + i].dnext;  // draft_sampled[start + i + 1]
      const bool valid = raw >= 0;
      const int token = max(raw, 0);
      if (greedy) {
        if (verifying) {
          const int top = int(g1[start + i].s);
          const bool ok = top == token && valid;
          verifying = ok;
          accepted += ok;
          out[i] = ok ? token : top;
        }
      } else {
        if (valid) log_p = fminf(log_p + term[start + i], 0.0f);
        verifying = verifying && valid;
        if (verifying) {
          const float prefix = texp(log_p);
          const int following = i < num_draft - 1 ? meta[start + i + 1].dnext : -1;
          float h;
          if (following >= 0) {
            const float weight = g2[start + i + 1].m;
            const float denom = weight + 1.0f - prefix;
            h = denom > 0.0f ? div_full(weight, denom) : 1.0f;
          } else {
            h = prefix;
          }
          if (urnd[start + i] <= h) accepted = i + 1;  // = tl_rand32(seed, pos) (rows of a request share the seed)
          out[i] = token;
        }
      }
    }
    a.num_sampled[rq] = accepted + 1;
    const int row = start + accepted;
    if (!greedy || row == end - 1) out[accepted] = (long long)(g2[row].o);
  }
  __syncthreads();
  trace(10);
}
}  // namespace dver


// ================================================================== drafter o-projection (wo_a with inverse RoPE, wo_b)
// Split-K FP8 GEMV for <= 8 rows on skinny-packed weights (ds41h/skinny.py pack: 16-row tile x k32 block = 512 B,
// lane 4g+t holds W[16rt+g][32kb+8t..+8] then W[16rt+g+8][..]; one ue8m0 scale per (tile, kb)), exact FP8 -> f16
// expansion, fresh HMMA accumulator per kb post-scaled in fp32 (the skinny / stock W8A16 numerics class), x staged as
// f16 fragments with one power-of-two scale per token. Grouped: tiles of group g read x + g * xg (wo_a: 2 groups of 8
// heads x 512 per rank). Optional inverse RoPE of the staged x (stock fused_inv_rope_fp8_quant with quantize=False:
// interleaved pairs of dims 448..511 of every 512-dim head, cos / sin = cos_sin_cache[pos][j] / [32 + j], result rounded
// to bf16). Grid: (#tiles / (W / KS)) CTAs of W warps; warp w -> tile w / KS, K slice w % KS; the slices of a tile are
// summed in fixed order in smem (deterministic). Weights + scales are loaded before griddepcontrol.wait (PDL).
namespace dlin {
constexpr int T = 512, W = T / 32, MAXM = 8, NOPE = 448, HALF_ROPE = 32, HEAD = 512;

struct Args {
  const __nv_bfloat16* x;  // [m, ...] row stride ldx; group g at x + g * xg
  long ldx, xg;
  const uint4* w;          // skinny fragment-major e4m3
  const uint8_t* s;        // [tiles][kbp] ue8m0
  int kbp;
  __nv_bfloat16* y;        // [m, ldy]
  long ldy;
  const long long* pos;    // [m] positions (inverse RoPE), or null
  const void* cs;          // cos_sin_cache [*, cs_stride] (fp32 or bf16)
  long cs_stride;
  int cs_bf16;
  int m, tiles_per_group;
  // optional fused TP all-reduce of y (DS41H_DRAFTER_OPROJ_AR; the moe2 fused-AR protocol / LL all-reduce numerics:
  // bf16 rank partial per 16-row tile as 64 LL words (row, token pair), fp32 sum in rank order, one bf16 rounding)
  unsigned long long* ar_recv[RANKS];
  unsigned* ar_state;
  int ar_rank;
  unsigned long long timeout;
  int early_dep;  // S1 wo_a: trigger the dependent (wo_b) right after the PDL wait (lin_set_early)
};

// rotate one 8-element (4-pair) bf16 chunk starting at head dim d0 (multiple of 8, >= NOPE) by the inverse RoPE of row tok
__device__ __forceinline__ uint4 inv_rope8(uint4 v, int d0, const Args& a, int tok) {
  const long long p = a.pos[tok];
  const int j0 = (d0 - NOPE) >> 1;
  float c[4], sn[4];
  if (a.cs_bf16) {
    const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(a.cs) + p * a.cs_stride;
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = __bfloat162float(row[j0 + i]), sn[i] = __bfloat162float(row[HALF_ROPE + j0 + i]);
  } else {
    const float* row = reinterpret_cast<const float*>(a.cs) + p * a.cs_stride;
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = row[j0 + i], sn[i] = row[HALF_ROPE + j0 + i];
  }
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  unsigned o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = unpack_bf16(u[i]);  // (even, odd) pair
    const float e = __fmaf_rn(f.y, sn[i], f.x * c[i]);   // x * cos + partner * sin
    const float od = __fmaf_rn(-f.x, sn[i], f.y * c[i]); // x * cos - partner * sin
    o[i] = pack_bf16(e, od);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// stage_x for [m <= 8, K] with an optional inverse RoPE per 512-dim head (natural k order)
template <int K>
__device__ __forceinline__ void stage_x_lin(const __nv_bfloat16* x, const Args& a, uint4* xs, float* xsc) {
  constexpr int KB = K / 32, V = K / 8, NW = T / 32, PER = (V + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool rope = a.cs != nullptr;
  for (int tok = warp; tok < 8; tok += NW) {
    float mx = 0.0f;
    if (tok < a.m) {
      const uint4* p = reinterpret_cast<const uint4*>(x + long(tok) * a.ldx);
#pragma unroll 4
      for (int i = 0; i < PER; ++i) {
        const int c = lane + 32 * i;
        if (c < V) {
          uint4 v = __ldg(p + c);
          const int d0 = (c * 8) % HEAD;
          if (rope && d0 >= NOPE) v = inv_rope8(v, d0, a, tok);
          mx = fmaxf(mx, amax8(v));
        }
      }
    }
    mx = warp_max(mx);
    if (lane == 0) xsc[tok] = ldexpf(1.0f, mx > 0.0f ? (int((__float_as_uint(mx) >> 23) & 0xff) - 127) - 14 : 0);
  }
  __syncthreads();
  constexpr int E = KB * 32;
  static_assert(E % T == 0, "staging entries must divide over the block");
  constexpr int PER2 = E / T;
#pragma unroll
  for (int j = 0; j < PER2; ++j) {
    const int i = tid + j * T, kb = i >> 5, l = i & 31, gg = l >> 2, tt = l & 3;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (gg < a.m) {
      raw = __ldg(reinterpret_cast<const uint4*>(x + long(gg) * a.ldx + kb * 32 + 8 * tt));
      const int d0 = (kb * 32 + 8 * tt) % HEAD;
      if (rope && d0 >= NOPE) raw = inv_rope8(raw, d0, a, gg);
    }
    const float inv = gg < a.m ? 1.0f / xsc[gg] : 0.0f;
    const float2 f01 = unpack_bf16(raw.x), f23 = unpack_bf16(raw.y), f45 = unpack_bf16(raw.z), f67 = unpack_bf16(raw.w);
    xs[i] = make_uint4(pack_f16(f01.x * inv, f01.y * inv), pack_f16(f23.x * inv, f23.y * inv), pack_f16(f45.x * inv, f45.y * inv),
                       pack_f16(f67.x * inv, f67.y * inv));
  }
  __syncthreads();
}

// inverse RoPE of one 8-element chunk with the token's cos / sin row preloaded into smem (cs[0..31] cos, cs[32..63] sin):
// the same float math as inv_rope8 (bitwise identical results)
__device__ __forceinline__ uint4 inv_rope8_s(uint4 v, int d0, const float* cs) {
  const int j0 = (d0 - NOPE) >> 1;
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  unsigned o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float c = cs[j0 + i], sn = cs[HALF_ROPE + j0 + i];
    const float2 f = unpack_bf16(u[i]);
    const float e = __fmaf_rn(f.y, sn, f.x * c);
    const float od = __fmaf_rn(-f.x, sn, f.y * c);
    o[i] = pack_bf16(e, od);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// One-pass staging (default; lin_set_stage(0) = the two-pass stage_x_lin): entry i = tid + j T of the [KB][32] fragment
// layout always belongs to token (tid & 31) >> 2, so every thread loads all its PER2 entries in one batch, applies the
// inverse RoPE from the smem cos / sin rows (loaded before the PDL wait), and the per-token amax is reduced over the quad
// (shuffles) and the 16 warps (smem): ONE global round trip after the PDL wait instead of the two passes with a dependent
// pos -> cos_sin_cache chain per RoPE chunk (~10 us of wo_a's tail in situ). Same amax set, same scale, same per-element
// math -> bitwise identical xs / xsc.
template <int K>
__device__ __forceinline__ void stage_x_lin1(const __nv_bfloat16* x, const Args& a, const float (*csm)[64], float* ampart, uint4* xs,
                                             float* xsc) {
  constexpr int KB = K / 32, E = KB * 32, NW = T / 32, PER2 = E / T;
  static_assert(E % T == 0, "staging entries must divide over the block");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gg = lane >> 2, tt = lane & 3;
  const bool live = gg < a.m;
  const bool rope = a.cs != nullptr;
  uint4 raw[PER2];
#pragma unroll
  for (int j = 0; j < PER2; ++j) {
    const int kb = warp + j * NW;
    raw[j] = live ? __ldg(reinterpret_cast<const uint4*>(x + long(gg) * a.ldx + kb * 32 + 8 * tt)) : make_uint4(0u, 0u, 0u, 0u);
  }
  float mx = 0.0f;
#pragma unroll
  for (int j = 0; j < PER2; ++j) {
    const int d0 = ((warp + j * NW) * 32 + 8 * tt) % HEAD;
    if (rope && live && d0 >= NOPE) raw[j] = inv_rope8_s(raw[j], d0, csm[gg]);
    mx = fmaxf(mx, amax8(raw[j]));
  }
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  if (tt == 0) ampart[warp * 8 + gg] = mx;
  __syncthreads();
  float tm = 0.0f;
#pragma unroll
  for (int w = 0; w < NW; ++w) tm = fmaxf(tm, ampart[w * 8 + gg]);
  const float sc = ldexpf(1.0f, tm > 0.0f ? (int((__float_as_uint(tm) >> 23) & 0xff) - 127) - 14 : 0);
  if (warp == 0 && tt == 0) xsc[gg] = sc;
  const float inv = live ? 1.0f / sc : 0.0f;
#pragma unroll
  for (int j = 0; j < PER2; ++j) {
    const float2 f01 = unpack_bf16(raw[j].x), f23 = unpack_bf16(raw[j].y), f45 = unpack_bf16(raw[j].z), f67 = unpack_bf16(raw[j].w);
    xs[tid + j * T] = make_uint4(pack_f16(f01.x * inv, f01.y * inv), pack_f16(f23.x * inv, f23.y * inv), pack_f16(f45.x * inv, f45.y * inv),
                                 pack_f16(f67.x * inv, f67.y * inv));
  }
  __syncthreads();
}

template <int K, int KS, bool S1>
__global__ void __launch_bounds__(T, 1) lin_kernel(const Args a) {
  constexpr int KB = K / 32, KBW = KB / KS, TPC = W / KS;
  static_assert(KB % KS == 0 && W % KS == 0 && KBW % 16 == 0, "split");
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB][32]
  __shared__ float xsc[MAXM];
  __shared__ float red[W][32][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile0 = blockIdx.x * TPC;
  const int tile = tile0 + warp / KS, ks = warp % KS, kb0 = ks * KBW;
  // weights + scales first (independent of the upstream kernel)
  uint4 wv[KBW];
  const uint4* wp = a.w + (long(tile) * KB + kb0) * 32 + lane;
#pragma unroll
  for (int q = 0; q < KBW; ++q) wv[q] = ld_stream16(wp + q * 32);
  uint4 sv[KBW / 16];
#pragma unroll
  for (int q = 0; q < KBW / 16; ++q) sv[q] = __ldg(reinterpret_cast<const uint4*>(a.s + long(tile) * a.kbp + kb0) + q);
  __shared__ float csm[MAXM][64];  // S1: per-token inverse-RoPE cos / sin rows (positions + cache are ready before the graph)
  __shared__ float ampart[W * 8];
  if (S1) {
    if (a.cs != nullptr) {
      const int tok = tid >> 6, q = tid & 63;  // T = 512 = 8 tokens x 64
      float v = 0.0f;
      if (tok < a.m) {
        const long long p = a.pos[tok];
        v = a.cs_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(a.cs)[p * a.cs_stride + q])
                      : reinterpret_cast<const float*>(a.cs)[p * a.cs_stride + q];
      }
      csm[tok][q] = v;
    }
    __syncthreads();
  }
  pdl_wait();
  // early_dep (S1 wo_a, lin_set_early(1)): release the dependent (wo_b: 80 CTAs, its weights / scales are loaded before its
  // own PDL wait) as soon as the upstream attention is complete, so wo_b's CTAs take the 68 SMs wo_a (64 CTAs) leaves free
  // and stream their 10.5 MB of weights while wo_a stages and computes. Safe: wo_b only waits (griddepcontrol.wait) for
  // wo_a's completion and every wo_a CTA is already resident when the trigger fires (no co-residency requirement, no
  // cross-CTA wait in wo_a); the dependent's own trigger stays at its end. Results are unchanged (same kernels, same math).
  if (S1 && K == 4096 && a.early_dep) pdl_launch_dependents();
  const int grp = tile0 / a.tiles_per_group;
  if (S1) {
    stage_x_lin1<K>(a.x + grp * a.xg, a, csm, ampart, xs, xsc);
  } else {
    stage_x_lin<K>(a.x + grp * a.xg, a, xs, xsc);
  }
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int q = 0; q < KBW; ++q) {
    const unsigned sw4 = (&sv[q / 16].x)[(q % 16) / 4];
    const float f = dmoe::sc8(sw4 >> (8 * (q % 4)));
    const uint4 xb = xs[(kb0 + q) * 32 + lane];
    const uint4 w = wv[q];
    // lane: row g bytes (w.x, w.y) = k 8t..8t+7, row g+8 (w.z, w.w): HMMA#1 k 8t..8t+3 (a0 / a1), HMMA#2 k 8t+4.. (a2 / a3)
    const unsigned a0 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.x & 0xffffu));
    const unsigned a2 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.x >> 16));
    const unsigned a1 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.z & 0xffffu));
    const unsigned a3 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.z >> 16));
    const unsigned c0 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.y & 0xffffu));
    const unsigned c2 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.y >> 16));
    const unsigned c1 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.w & 0xffffu));
    const unsigned c3 = e4m3x2_to_f16x2(static_cast<unsigned short>(w.w >> 16));
    float d[4];
    dmoe::hmma_zero(d, a0, a1, a2, a3, xb.x, xb.y);
    dmoe::hmma_acc(d, c0, c1, c2, c3, xb.z, xb.w);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = __fmaf_rn(d[i], f, acc[i]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
  __syncthreads();
  __shared__ unsigned yb[TPC][64];  // fused AR: bf16 rank partials, word (row, token pair)
  if (ks == 0) {
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int z = 0; z < KS; ++z)
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] += red[warp + z][lane][i];
    const int r0 = tile * 16;
    if (a.ar_recv[0] == nullptr) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * t + c;
        if (tok < a.m) {
          a.y[long(tok) * a.ldy + r0 + g] = __float2bfloat16_rn(v[c] * xsc[tok]);
          a.y[long(tok) * a.ldy + r0 + g + 8] = __float2bfloat16_rn(v[2 + c] * xsc[tok]);
        }
      }
    } else {
      // lane (g, t) holds rows g / g + 8, tokens 2t / 2t + 1 = word (row, token pair t)
      const int tl = warp / KS;
      yb[tl][g * 4 + t] = pack_bf16(v[0] * xsc[2 * t], v[1] * xsc[2 * t + 1]);
      yb[tl][(g + 8) * 4 + t] = pack_bf16(v[2] * xsc[2 * t], v[3] * xsc[2 * t + 1]);
    }
  }
  if (a.ar_recv[0] != nullptr) {
    __syncthreads();
    const unsigned epoch = ld_volatile(a.ar_state);
    const unsigned tag = epoch + 1u == 0u ? 2u : epoch + 1u;
    const int par = int(tag & 1u);
    constexpr int NT = 5120 / 16;  // tile slots per parity (the moe2 fused-AR layout)
    if (tid < TPC * 64) {
      const int tl = tid >> 6, wd = tid & 63;
      const long gt = long(tile0 + tl);
      const unsigned w = yb[tl][wd];
#pragma unroll
      for (int q = 1; q < RANKS; ++q) {
        const int p = (a.ar_rank + q) % RANKS;
        unsigned long long* dst = a.ar_recv[0];
#pragma unroll
        for (int z = 1; z < RANKS; ++z) dst = p == z ? a.ar_recv[z] : dst;
        st_word(dst + ((long(par) * NT + gt) * RANKS + a.ar_rank) * 64 + wd, (static_cast<unsigned long long>(tag) << 32) | w);
      }
    }
    if (tid < TPC * 64) {
      const int tl = tid >> 6, wd = tid & 63, row = wd >> 2, tp = (wd & 3) * 2;
      const long gt = long(tile0 + tl);
      const unsigned long long* box = a.ar_recv[0];
#pragma unroll
      for (int z = 1; z < RANKS; ++z) box = a.ar_rank == z ? a.ar_recv[z] : box;
      float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
      for (int src = 0; src < RANKS; ++src) {
        unsigned w;
        if (src == a.ar_rank) {
          w = yb[tl][wd];
        } else {
          unsigned long long* q = const_cast<unsigned long long*>(box) + ((long(par) * NT + gt) * RANKS + src) * 64 + wd;
          w = poll_word(q, tag, a.timeout, 1);
          st_word(q, 0ull);
        }
        const float2 f = unpack_bf16(w);
        if (src == 0) acc0 = f.x, acc1 = f.y;
        else acc0 = __fadd_rn(acc0, f.x), acc1 = __fadd_rn(acc1, f.y);
      }
      const long r = gt * 16 + row;
      if (tp < a.m) a.y[long(tp) * a.ldy + r] = __float2bfloat16_rn(acc0);
      if (tp + 1 < a.m) a.y[long(tp + 1) * a.ldy + r] = __float2bfloat16_rn(acc1);
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      if (atomicAdd(a.ar_state + 1, 1u) == gridDim.x - 1) {
        atomicExch(a.ar_state + 1, 0u);
        __threadfence();
        atomicExch(a.ar_state, tag);
      }
    }
  }
  pdl_launch_dependents();
}
}  // namespace dlin

// ================================================================== bf16 verification head GEMV (M <= 8 rows, exact class)
// y[t, n] = bf16(sum_k x[t, k] W[n, k]) over the row-major bf16 lm_head shard W [n, ldw] with fp32 accumulation over all of
// K on bf16 HMMA (stock: cuBLAS bf16 GEMM, fp32 accumulation; only the summation order differs). Lane (g, t) of a warp
// loads W[16 tile + g][32 kb + 8t .. +8] and W[16 tile + g + 8][..] (16 B each, every row's 64 B per kb coalesced over
// the quad) and x is staged in the same physical-k order, so no layout transform is needed. Optional row gather
// x_row(t) = x[idx[t]] (stock: hidden_states[logits_indices] before compute_logits). One 16-row tile per warp.
namespace dvh {
constexpr int K = 5120, KB = K / 32, T = 512, W = T / 32, MAXM = 8, CK = 4;

struct Args {
  const __nv_bfloat16* x;
  long ldx;
  const long long* idx;  // [m] row indices of x, or null
  const __nv_bfloat16* w;
  long ldw;
  __nv_bfloat16* y;
  long ldy;
  int m, n;
};

__device__ __forceinline__ void hmma_bf16_acc(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(T, 1) head_bf16_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB][32]: lane (gg, tt) = x[gg][32 kb + 8 tt .. +8] (bf16)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tiles = a.n / 16;
  const int tile = blockIdx.x * W + warp;
  const bool live = tile < tiles;
  const __nv_bfloat16* ra = a.w + long(live ? tile : 0) * 16 * a.ldw + long(g) * a.ldw + 8 * t;
  const __nv_bfloat16* rb = ra + 8 * a.ldw;
  uint4 va0[CK], vb0[CK], va1[CK], vb1[CK];
#pragma unroll
  for (int q = 0; q < CK; ++q) va0[q] = ld_stream16(ra + q * 32), vb0[q] = ld_stream16(rb + q * 32);  // before the PDL wait
#pragma unroll
  for (int q = 0; q < CK; ++q) va1[q] = ld_stream16(ra + (CK + q) * 32), vb1[q] = ld_stream16(rb + (CK + q) * 32);
  pdl_wait();
  for (int i = tid; i < KB * 32; i += T) {
    const int kb = i >> 5, l = i & 31, gg = l >> 2, tt = l & 3;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gg < a.m) {
      const long row = a.idx != nullptr ? long(a.idx[gg]) : long(gg);
      v = __ldg(reinterpret_cast<const uint4*>(a.x + row * a.ldx + kb * 32 + 8 * tt));
    }
    xs[i] = v;
  }
  __syncthreads();
  if (live) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto step = [&](const uint4 va, const uint4 vb, int kb) {
      const uint4 xb = xs[kb * 32 + lane];
      hmma_bf16_acc(acc, va.x, vb.x, va.y, vb.y, xb.x, xb.y);
      hmma_bf16_acc(acc, va.z, vb.z, va.w, vb.w, xb.z, xb.w);
    };
#pragma unroll 1
    for (int kb = 0; kb < KB; kb += 2 * CK) {  // both CK-kb chunks preloaded at entry
#pragma unroll
      for (int q = 0; q < CK; ++q) step(va0[q], vb0[q], kb + q);
      if (kb + 2 * CK < KB) {
#pragma unroll
        for (int q = 0; q < CK; ++q) va0[q] = ld_stream16(ra + (kb + 2 * CK + q) * 32), vb0[q] = ld_stream16(rb + (kb + 2 * CK + q) * 32);
      }
#pragma unroll
      for (int q = 0; q < CK; ++q) step(va1[q], vb1[q], kb + CK + q);
      if (kb + 3 * CK < KB) {
#pragma unroll
        for (int q = 0; q < CK; ++q) va1[q] = ld_stream16(ra + (kb + 3 * CK + q) * 32), vb1[q] = ld_stream16(rb + (kb + 3 * CK + q) * 32);
      }
    }
    const int r0 = tile * 16;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = 2 * t + c;
      if (tok < a.m) {
        a.y[long(tok) * a.ldy + r0 + g] = __float2bfloat16_rn(acc[c]);
        a.y[long(tok) * a.ldy + r0 + g + 8] = __float2bfloat16_rn(acc[2 + c]);
      }
    }
  }
  pdl_launch_dependents();
}
}  // namespace dvh

// ================================================================== DSpark main_proj GEMV (M <= 8)
// y[t, n] = bf16(sum_k W[n, k] x[t, k]), W e4m3 [5120, 15360] with ue8m0 tile scales [N/16, KBP] in skinny
// fragment-major layout (ds41h/skinny.py pack; natural k order). Grid (40 tile groups x 3 K slices of 5120): each CTA
// stages its x slice once (f16, power-of-two scale per token), warp w owns tile 8*gx + w over the slice (fresh HMMA
// accumulator per k32 block, post-scaled by the tile scale); slice partials go to global and the last of the three CTAs
// of a tile group sums them in slice order (deterministic) and writes bf16.
namespace dmain {
constexpr int N = 5120, KS = 5120, KBS = KS / 32, SLICES = 3, KBT = KBS * SLICES, T = 256, W = T / 32, MAXM = 8;
constexpr int GROUPS = N / 16 / W;  // 40

struct Args {
  const __nv_bfloat16* x;  // [m, ldx], K = 15360
  long ldx;
  const uint4* w;          // fragment-major
  const uint8_t* s;        // [N/16, kbp]
  int kbp;
  float* part;             // [SLICES][MAXM][N]
  unsigned* ctr;           // [GROUPS], zero, self-resetting
  __nv_bfloat16* y;        // [m, ldy]
  long ldy;
  int m;
};

__global__ void __launch_bounds__(T, 2) main_kernel(const Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KBS][32]
  __shared__ float xsc[MAXM];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int gx = blockIdx.x, slice = blockIdx.y;
  const int tile = gx * W + warp;
  const uint4* wp = a.w + (long(tile) * KBT + slice * KBS) * 32 + lane;
  const uint8_t* sp = a.s + long(tile) * a.kbp + slice * KBS;
  uint4 wv0[8], wv1[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) wv0[q] = ld_stream16(wp + q * 32);
  pdl_wait();
  stage_x<KS, T, false>(a.x + slice * KS, a.ldx, a.m, xs, xsc);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  auto step = [&](const uint4 wv, int kb, unsigned sbyte) {
    const uint4 xb = xs[kb * 32 + lane];
    const unsigned a0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x & 0xffffu));
    const unsigned a2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.x >> 16));
    const unsigned a1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z & 0xffffu));
    const unsigned a3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.z >> 16));
    const unsigned c0 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y & 0xffffu));
    const unsigned c2 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.y >> 16));
    const unsigned c1 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w & 0xffffu));
    const unsigned c3 = e4m3x2_to_f16x2(static_cast<unsigned short>(wv.w >> 16));
    float d[4];
    dmoe::hmma_zero(d, a0, a1, a2, a3, xb.x, xb.y);
    dmoe::hmma_acc(d, c0, c1, c2, c3, xb.z, xb.w);
    const float f = fp4_block_scale<0>(sbyte & 0xffu);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = __fmaf_rn(d[i], f, acc[i]);
  };
  uint2 sc0 = __ldg(reinterpret_cast<const uint2*>(sp)), sc1;
#pragma unroll 1
  for (int kb = 0; kb < KBS; kb += 16) {  // double-buffered 8-kb chunks (weights + their 8 scale bytes)
#pragma unroll
    for (int q = 0; q < 8; ++q) wv1[q] = ld_stream16(wp + (kb + 8 + q) * 32);
    sc1 = __ldg(reinterpret_cast<const uint2*>(sp + kb + 8));
#pragma unroll
    for (int q = 0; q < 8; ++q) step(wv0[q], kb + q, (q < 4 ? sc0.x : sc0.y) >> (8 * (q & 3)));
    if (kb + 16 < KBS) {
#pragma unroll
      for (int q = 0; q < 8; ++q) wv0[q] = ld_stream16(wp + (kb + 16 + q) * 32);
      sc0 = __ldg(reinterpret_cast<const uint2*>(sp + kb + 16));
    }
#pragma unroll
    for (int q = 0; q < 8; ++q) step(wv1[q], kb + 8 + q, (q < 4 ? sc1.x : sc1.y) >> (8 * (q & 3)));
  }
  const int r0 = tile * 16;
  float* pp = a.part + long(slice) * MAXM * N;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int tok = 2 * t + c;
    pp[long(tok) * N + r0 + g] = acc[c] * xsc[tok];
    pp[long(tok) * N + r0 + g + 8] = acc[2 + c] * xsc[tok];
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(a.ctr + gx, 1u) == SLICES - 1;
  }
  __syncthreads();
  if (last) {
    __threadfence();
    for (int i = tid; i < a.m * W * 16; i += T) {
      const int tok = i / (W * 16), row = gx * W * 16 + i % (W * 16);
      float v = 0.0f;
#pragma unroll
      for (int sl = 0; sl < SLICES; ++sl) v += __ldcg(a.part + (long(sl) * MAXM + tok) * N + row);
      a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v);
    }
    if (tid == 0) a.ctr[gx] = 0u;
  }
  pdl_launch_dependents();
}
}  // namespace dmain

// ================================================================== drafter router (gate GEMV + dsv4 top-k, one launch)
// Replaces stock GateLinear (cute_dsl ll_bf16 GEMM, bf16 logits) + logits.float() copy + ds41h moe_topk (3 launches) for the
// DRAFTER FFN (M <= 8 tokens). CTA c, warp w computes expert e = 4c + w for all tokens (fp32 accumulation and fp32
// logits like the stock GEMM); the last CTA to arrive (atomic counter, reset by itself: graph safe) runs moe_topk's
// semantics: score = sqrt(softplus(l)), choice = score + bias (bias_vl for image-sentinel tokens), NaN -> -1e30, top-k with
// ties to the lowest expert id, weights = score * scale / sum when renormalize.
namespace drouter {
// One CTA per expert (E <= 512 CTAs, one wave for E = 128), T = 320 threads: thread i owns k elements [16 i, 16 i + 16)
// of the expert row and of every token (2 x 16-byte loads each, all issued at once), per-token fp32 sums reduced over
// the CTA (warp shuffles + smem, fixed order). The last CTA to arrive runs the top-k with one warp per token.
constexpr int H = 5120, T = 320, W = T / 32, MAXM = 8, MAXE = 512;
static_assert(H / 8 == 2 * T, "two uint4 per thread");

struct Args {
  const __nv_bfloat16* x;
  long ldx;
  const __nv_bfloat16* gw;  // [E, H]
  const float* bias;        // [E]
  const float* bias_vl;     // [E] or null
  const void* ids_in;       // input ids (int32 or int64) or null
  int ids64;
  long image_lo;
  float* logits;            // [MAXM, E] scratch
  unsigned* counter;        // arrival counter (0 between launches)
  int* out_ids;             // [m, k]
  float* out_w;             // [m, k]
  float scale;
  int renormalize, m, k, E;
};

__device__ __forceinline__ float dot8(uint4 xv, uint4 wv) {
  const unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w}, ww[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.0f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 xf = unpack_bf16(xw[q]), wf = unpack_bf16(ww[q]);
    acc = __fmaf_rn(xf.x, wf.x, acc);
    acc = __fmaf_rn(xf.y, wf.y, acc);
  }
  return acc;
}

template <int VPL>
__global__ void __launch_bounds__(T) router_kernel(const Args a) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int e = blockIdx.x;
  __shared__ float red[W][MAXM];
  __shared__ bool last;
  // gate weights do not depend on upstream kernels: issue their loads before the PDL wait
  const uint4* wr = reinterpret_cast<const uint4*>(a.gw + long(e) * H);
  const uint4 w0 = __ldg(wr + tid), w1 = __ldg(wr + T + tid);
  pdl_wait();
  uint4 xv0[MAXM], xv1[MAXM];
#pragma unroll
  for (int t = 0; t < MAXM; ++t) {
    if (t < a.m) {
      const uint4* xr = reinterpret_cast<const uint4*>(a.x + long(t) * a.ldx);
      xv0[t] = __ldg(xr + tid);
      xv1[t] = __ldg(xr + T + tid);
    }
  }
#pragma unroll
  for (int t = 0; t < MAXM; ++t) {
    if (t < a.m) {
      const float v = warp_sum(dot8(xv0[t], w0) + dot8(xv1[t], w1));
      if (lane == 0) red[warp][t] = v;
    }
  }
  __syncthreads();
  if (tid < a.m) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) v += red[w][tid];
    a.logits[long(tid) * a.E + e] = v;  // fp32 logits (stock ll_bf16_gemm output_dtype = float32)
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned prev = atomicAdd(a.counter, 1u);
    last = prev == gridDim.x - 1;
    if (last) atomicExch(a.counter, 0u);
  }
  __syncthreads();
  if (!last) {
    pdl_launch_dependents();
    return;
  }
  __threadfence();
  auto score = [](float v) { return sqrtf(v > 20.f ? v : logf(1.f + expf(v))); };
  for (int row = warp; row < a.m; row += W) {
    bool image = false;
    if (a.bias_vl != nullptr && a.ids_in != nullptr && a.image_lo > 0) {
      const long id = a.ids64 ? static_cast<const long long*>(a.ids_in)[row] : long(static_cast<const int*>(a.ids_in)[row]);
      image = id >= a.image_lo && id < a.image_lo + 5;
    }
    const float* b = image ? a.bias_vl : a.bias;
    const float* lrow = a.logits + long(row) * a.E;
    float cur[VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      const int ex = v * 32 + lane;
      if (ex < a.E) {
        const float c = score(__ldcg(lrow + ex)) + (b != nullptr ? b[ex] : 0.f);
        cur[v] = (c == c) ? c : -1e30f;
      } else {
        cur[v] = -INFINITY;
      }
    }
    float sel_w = 0.f, sum = 0.f;
    int sel_id = 0;
    for (int kk = 0; kk < a.k; ++kk) {
      float best = -INFINITY;
      int best_e = 0x7fffffff;
#pragma unroll
      for (int v = 0; v < VPL; ++v) {
        const int ex = v * 32 + lane;
        if (cur[v] > best || (cur[v] == best && ex < best_e)) best = cur[v], best_e = ex;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
        if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
      }
#pragma unroll
      for (int v = 0; v < VPL; ++v)
        if (v * 32 + lane == best_e) cur[v] = -INFINITY;
      const float wk = score(__ldcg(lrow + best_e));
      sum += wk;
      if (lane == kk) sel_w = wk, sel_id = best_e;
    }
    const float mul = a.renormalize ? a.scale / (sum > 0.f ? sum : 1.f) : a.scale;
    if (lane < a.k) {
      a.out_w[row * a.k + lane] = sel_w * mul;
      a.out_ids[row * a.k + lane] = sel_id;
    }
  }
  pdl_launch_dependents();
}
}  // namespace drouter

// ------------------------------------------------------------------ host
struct Comm {
  unsigned long long* recv[RANKS];
  unsigned* state;
  float* cand;
  long slot_rows;
  int rank;
  int max_ctas;
};
std::vector<Comm*> comms;

Comm& get(int64_t h) {
  TORCH_CHECK(h >= 0 && h < int64_t(comms.size()) && comms[h] != nullptr, "invalid ds41h drafter handle");
  return *comms[h];
}

}  // namespace

int64_t recv_bytes(int64_t slot_rows) {
  return std::max(2LL * MAXS * RANKS * slot_rows * WORDS * 8, 2LL * MAXS * RANKS * MAXCTAS * DRAFT_ROWS * WORDS * 8);
}
int64_t max_steps() { return MAXS; }
int64_t max_chunk() { return MAXCHUNK; }
int64_t markov_rank() { return R; }

int64_t create(std::vector<int64_t> buffers, int64_t rank, int64_t slot_rows, int64_t max_ctas) {
  TORCH_CHECK(int64_t(buffers.size()) == RANKS && rank >= 0 && rank < RANKS && slot_rows > 0 && slot_rows <= 64);
  auto* c = new Comm{};
  for (int p = 0; p < RANKS; ++p) c->recv[p] = reinterpret_cast<unsigned long long*>(buffers[p]);
  c->state = zeroed<unsigned>(1 + MAXS * MAXGRP);
  c->cand = zeroed<float>(long(MAXS) * slot_rows * max_ctas * 5);
  c->slot_rows = slot_rows;
  c->rank = int(rank);
  c->max_ctas = int(max_ctas);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  comms.push_back(c);
  return int64_t(comms.size()) - 1;
}

void destroy(int64_t h) {
  Comm& c = get(h);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  cudaFree(c.state);
  cudaFree(c.cand);
  delete comms[h];
  comms[h] = nullptr;
}

void draft_markov(int64_t h, torch::Tensor base, torch::Tensor w1, torch::Tensor w2, torch::Tensor input_ids, int64_t anchor_stride,
                  torch::Tensor idx_map, torch::Tensor sample_pos, torch::Tensor temp, torch::Tensor seeds, torch::Tensor draft_tokens,
                  torch::Tensor cache, torch::Tensor token_logit, torch::Tensor lse, int64_t rows, int64_t nspec, int64_t vocab_start,
                  int64_t chunk, int64_t timeout_ns, bool pdl, std::optional<torch::Tensor> trace, int64_t groups,
                  std::optional<torch::Tensor> req_draft) {
  const Comm& c = get(h);
  TORCH_CHECK(rows >= 1 && rows <= DRAFT_ROWS && nspec >= 1 && nspec <= MAXS, "rows / nspec out of range");
  TORCH_CHECK(groups >= 0 && groups <= MAXGRP && RANKS * groups <= 64, "groups: 0..16");
  TORCH_CHECK(base.scalar_type() == at::kBFloat16 && w1.scalar_type() == at::kBFloat16 && w2.scalar_type() == at::kBFloat16 &&
                  cache.scalar_type() == at::kBFloat16,
              "bf16 base / w1 / w2 / cache required");
  TORCH_CHECK(base.dim() == 2 && base.stride(1) == 1 && base.size(0) >= rows * nspec, "base [rows * nspec, local] unit column stride");
  TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous() && w1.size(1) == R && w2.size(1) == R, "markov rank 256, contiguous");
  const int local = int(w2.size(0));
  TORCH_CHECK(base.size(1) >= local, "base narrower than the local vocab");
  TORCH_CHECK(chunk >= 1 && chunk <= MAXCHUNK, "chunk");
  const int G = int((local + chunk - 1) / chunk);
  TORCH_CHECK(G <= c.max_ctas && G <= MAXCTAS, "too many CTAs for the candidate buffer");
  TORCH_CHECK(input_ids.scalar_type() == at::kInt && idx_map.scalar_type() == at::kLong && sample_pos.scalar_type() == at::kLong &&
                  seeds.scalar_type() == at::kLong && draft_tokens.scalar_type() == at::kLong && temp.scalar_type() == at::kFloat,
              "index dtypes");
  TORCH_CHECK(idx_map.is_contiguous() && sample_pos.is_contiguous() && idx_map.numel() >= rows * nspec && sample_pos.numel() >= rows * nspec);
  TORCH_CHECK(token_logit.scalar_type() == at::kFloat && lse.scalar_type() == at::kFloat && token_logit.stride(0) == lse.stride(0) &&
              token_logit.stride(1) == 1 && lse.stride(1) == 1);
  TORCH_CHECK(cache.dim() == 3 && cache.stride(2) == 1 && cache.size(1) >= nspec && cache.size(2) >= local);
  TORCH_CHECK(draft_tokens.dim() == 2 && draft_tokens.stride(1) == 1 && draft_tokens.size(1) >= nspec);
  DraftArgs a{};
  for (int p = 0; p < RANKS; ++p) a.recv[p] = c.recv[p];
  a.state = c.state;
  a.cand = c.cand;
  a.base = reinterpret_cast<const __nv_bfloat16*>(base.data_ptr());
  a.base_stride = base.stride(0);
  a.w1 = reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr());
  a.w2 = reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr());
  a.input_ids = input_ids.data_ptr<int>();
  a.anchor_stride = anchor_stride * input_ids.stride(0);
  a.idx_map = reinterpret_cast<const long long*>(idx_map.data_ptr<int64_t>());
  a.sample_pos = reinterpret_cast<const long long*>(sample_pos.data_ptr<int64_t>());
  a.temp = temp.data_ptr<float>();
  a.seeds = reinterpret_cast<const long long*>(seeds.data_ptr<int64_t>());
  a.draft_tokens = reinterpret_cast<long long*>(draft_tokens.data_ptr<int64_t>());
  a.dt_stride = draft_tokens.stride(0);
  if (req_draft.has_value()) {
    TORCH_CHECK(req_draft->scalar_type() == at::kLong && req_draft->dim() == 2 && req_draft->stride(1) == 1 && req_draft->size(1) >= nspec &&
                    req_draft->size(0) >= token_logit.size(0) - 1,
                "req_draft: [max_reqs, >= nspec] int64, unit column stride");
    a.req_draft = reinterpret_cast<long long*>(req_draft->data_ptr<int64_t>());
    a.req_stride = req_draft->stride(0);
  }
  a.cache = reinterpret_cast<__nv_bfloat16*>(cache.data_ptr());
  a.cache_s0 = cache.stride(0);
  a.cache_s1 = cache.stride(1);
  a.token_logit = token_logit.data_ptr<float>();
  a.lse = lse.data_ptr<float>();
  a.vp_stride = token_logit.stride(0);
  a.sink = token_logit.size(0) - 1;
  a.slot_rows = c.slot_rows;
  a.timeout = static_cast<unsigned long long>(timeout_ns);
  a.rank = c.rank;
  a.rows = int(rows);
  a.nspec = int(nspec);
  a.local = local;
  a.vocab_start = int(vocab_start);
  a.chunk = int(chunk);
  a.vocab = int(w1.size(0));
  a.trace = trace.has_value() ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  a.groups = int(groups);
  TORCH_CHECK(rows * nspec <= 64, "rows * nspec <= 64 (base staging)");
  TORCH_CHECK(chunk % 8 == 0 && base.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(base.data_ptr()) % 16 == 0 && vocab_start % 8 == 0,
              "16-byte aligned base rows");
  const int smem = MAXCHUNK * W2PITCH + MAXCHUNK * 4 + 64 * MAXCHUNK * 2;
  opt_in_smem<draft_markov_kernel>(smem);
  const c10::cuda::CUDAGuard guard(base.device());
  if (pdl) {
    launch_pdl("ds41h draft_markov", draft_markov_kernel, dim3(G), dim3(THREADS), smem, a);
  } else {
    draft_markov_kernel<<<G, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


// ------------------------------------------------------------------ drafter MoE host
int64_t moe_hb_halves() { return int64_t(dmoe::MAXSLOTS) * dmoe::HB_SLOT; }
int64_t moe2s_scratch_floats() { return int64_t(dmoe::MAXSLOTS) * dmoe::MAXM * dmoe::H + dmoe::M2R_CHUNKS; }  // + counters (moe2s: 20, moe2r: 40)
// moe13 variant (DS41H_DRAFTER_MOE13_VARIANT, read once): 0 = 2-stage 8-kb chunks (default), 1 = 3-stage 8-kb chunks
// (slower, 64.9 vs 63.7 us at m=5), 2 = 2-stage 16-kb chunks (256 B contiguous per weight row per request)
int moe13_variant() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_DRAFTER_MOE13_VARIANT");
    const int x = e ? std::atoi(e) : 0;
    return x >= 1 && x <= 5 ? x : 0;
  }();
  return v;
}
// DS41H_DRAFTER_MOE2_VARIANT=2: persistent half-CTA moe2r instead of moe2s (both need the scratch; read once)
bool moe2v_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("DS41H_DRAFTER_MOE2_VARIANT");
    return e && std::atoi(e) == 3;
  }();
  return v;
}
bool moe2r_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("DS41H_DRAFTER_MOE2_VARIANT");
    return e && std::atoi(e) == 2;
  }();
  return v;
}
int moe_dbg() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_DRAFTER_MOE_DBG");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

void moe_decode(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor w13, torch::Tensor s13, torch::Tensor w2,
                torch::Tensor s2, torch::Tensor hb, torch::Tensor y, double limit, int64_t ctas, bool pdl, std::optional<torch::Tensor> scratch,
                std::optional<torch::Tensor> sh13,
                std::optional<torch::Tensor> ss13, std::optional<torch::Tensor> sh2, std::optional<torch::Tensor> ss2,
                std::optional<torch::Tensor> gate_w, std::optional<torch::Tensor> gate_bias, std::optional<torch::Tensor> glog,
                std::optional<torch::Tensor> gbar, double route_scale, bool renormalize, int64_t ar_handle, int64_t timeout_ns,
                int64_t moe2_ctas) {
  using namespace dmoe;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == H && x.stride(0) % 8 == 0, "x [m, 5120] bf16");
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "1 <= m <= 8");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m, "ids int32 [m, k]");
  TORCH_CHECK(wts.scalar_type() == at::kFloat && wts.is_contiguous() && wts.sizes() == ids.sizes(), "wts fp32 [m, k]");
  const int k = int(ids.size(1));
  TORCH_CHECK(m * k <= T2 && k <= 16, "m * k too large");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  TORCH_CHECK(w13.scalar_type() == at::kByte && w13.is_contiguous() && w13.dim() == 3 && w13.size(1) == 2 * I && w13.size(2) == H / 2, "w13");
  const int E = int(w13.size(0));
  TORCH_CHECK(E <= MAXE, "too many experts");
  TORCH_CHECK(s13.is_contiguous() && s13.numel() == long(E) * 2 * I * KB1, "s13");
  TORCH_CHECK(w2.is_contiguous() && w2.numel() == long(E) * H * (I / 2), "w2");
  TORCH_CHECK(s2.is_contiguous() && s2.numel() == long(E) * H * KB2, "s2");
  TORCH_CHECK(hb.scalar_type() == at::kHalf && hb.is_contiguous() && hb.numel() >= moe_hb_halves(), "hb buffer");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == m && y.size(1) == H && y.stride(1) == 1, "y [m, 5120] bf16");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.ids = ids.data_ptr<int>();
  a.wts = wts.data_ptr<float>();
  a.w13 = w13.data_ptr<uint8_t>();
  a.s13 = reinterpret_cast<const uint8_t*>(s13.data_ptr());
  a.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  a.s2 = reinterpret_cast<const uint8_t*>(s2.data_ptr());
  a.hb = reinterpret_cast<__half*>(hb.data_ptr());
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.limit = float(limit);
  a.m = m;
  a.k = k;
  a.E = E;
  a.shared = sh13.has_value() ? 1 : 0;
  if (gate_w.has_value()) {  // fused router: ids / wts are outputs written by moe13 (CTA 0)
    TORCH_CHECK(gate_bias.has_value() && glog.has_value() && gbar.has_value(), "fused router: gate_bias / glog / gbar required");
    TORCH_CHECK(gate_w->scalar_type() == at::kBFloat16 && gate_w->is_contiguous() && gate_w->size(0) == E && gate_w->size(1) == H, "gate_w [E, 5120] bf16");
    TORCH_CHECK(gate_bias->scalar_type() == at::kFloat && gate_bias->is_contiguous() && gate_bias->numel() == E, "gate bias fp32 [E]");
    TORCH_CHECK(glog->scalar_type() == at::kFloat && glog->numel() >= long(MAXM) * E && gbar->numel() >= 1, "router scratch");
    TORCH_CHECK(k <= 16 && ctas <= 1024, "router topk <= 16");
    a.gate_w = reinterpret_cast<const __nv_bfloat16*>(gate_w->data_ptr());
    a.gate_bias = gate_bias->data_ptr<float>();
    a.glog = glog->data_ptr<float>();
    a.gbar = reinterpret_cast<unsigned*>(gbar->data_ptr());
    a.ids_out = ids.data_ptr<int>();
    a.wts_out = wts.data_ptr<float>();
    a.route_scale = float(route_scale);
    a.renormalize = renormalize ? 1 : 0;
  }
  if (a.shared) {
    TORCH_CHECK(ss13.has_value() && sh2.has_value() && ss2.has_value(), "shared expert: all four tensors required");
    TORCH_CHECK(sh13->is_contiguous() && sh13->numel() * sh13->element_size() == long(UPE) * KB1 * 512, "sh13 packed size");
    TORCH_CHECK(ss13->is_contiguous() && ss13->numel() * ss13->element_size() == long(UPE) * KB1 * 2, "ss13 size");
    TORCH_CHECK(sh2->is_contiguous() && sh2->numel() * sh2->element_size() == long(H / 16) * KB2 * 512, "sh2 packed size");
    TORCH_CHECK(ss2->is_contiguous() && ss2->numel() * ss2->element_size() == long(H / 16) * KB2, "ss2 size");
    a.sh13 = reinterpret_cast<const uint4*>(sh13->data_ptr());
    a.ss13 = reinterpret_cast<const uint4*>(ss13->data_ptr());
    a.sh2 = reinterpret_cast<const uint4*>(sh2->data_ptr());
    a.ss2 = reinterpret_cast<const uint8_t*>(ss2->data_ptr());
  }
  if (ar_handle >= 0) {  // fused TP all-reduce in moe2 (DraftComm buffers: [2][320 tiles][4][64] LL words)
    const Comm& c = get(ar_handle);
    TORCH_CHECK(2L * (H / 16) * RANKS * 64 * 8 <= recv_bytes(c.slot_rows), "moe all-reduce slots exceed the comm buffer");
    for (int p = 0; p < RANKS; ++p) a.ar_recv[p] = c.recv[p];
    a.ar_state = c.state;
    a.ar_rank = c.rank;
  }
  a.timeout = static_cast<unsigned long long>(timeout_ns);
  const int var = moe13_variant();  // 0: <2 stages, 8 kb>, 1: <3, 8>, 2: <2, 16>, 3: balanced quarters (moe13q)
  const int smem13 = var == 5 ? V13<8>::SMEM
                              : KB1 * 32 * 16 + W * (var == 4 ? 3 * R_WCH : var == 1 ? 3 * 16 * (8 * 16 + 16) : var == 2 ? 2 * 16 * (16 * 16 + 16) : 2 * 16 * (8 * 16 + 16));
  a.dbg = moe_dbg();
  if (var == 5) opt_in_smem<moe13v_kernel<8>>(smem13);
  else if (var == 1) opt_in_smem<moe13_kernel<3, 8>>(smem13);
  else if (var == 2) opt_in_smem<moe13_kernel<2, 16>>(smem13);
  else if (var == 3) opt_in_smem<moe13q_kernel>(smem13);
  else if (var == 4) opt_in_smem<moe13r_kernel<3>>(smem13);
  else opt_in_smem<moe13_kernel<2, 8>>(smem13);
  const c10::cuda::CUDAGuard guard(x.device());
  const int g13 = int(ctas);
  TORCH_CHECK(var != 5 || g13 >= UPE, "moe13v: needs >= 72 CTAs (one per shared-expert tile)");
  TORCH_CHECK(var != 5 || a.gate_w == nullptr, "moe13v: the fused router (DS41H_DRAFTER_ROUTER) is not supported");
  // moe2_ctas > 0: persistent moe2p with that many CTAs (each <= PT tiles), else the classic 320-CTA moe2
  const int g2 = moe2_ctas > 0 ? int(std::min<int64_t>(moe2_ctas, H / 16)) : 0;
  TORCH_CHECK(g2 == 0 || g2 * PT >= H / 16, "moe2p: need >= 107 CTAs (3 tiles each)");
  // slot-major moe2s: needs the scratch tensor (fp32 partials + counters), no fused all-reduce
  const bool m2s = scratch.has_value() && ar_handle < 0 && g2 == 0;  // python passes scratch iff DS41H_DRAFTER_MOE2_VARIANT >= 1
  const bool m2r = m2s && moe2r_enabled();
  if (m2s) {
    TORCH_CHECK(scratch->scalar_type() == at::kFloat && scratch->is_contiguous() &&
                    scratch->numel() >= long(MAXSLOTS) * MAXM * H + M2R_CHUNKS, "moe2s scratch");
    a.part = scratch->data_ptr<float>();
    a.cnt = reinterpret_cast<unsigned*>(scratch->data_ptr<float>() + long(MAXSLOTS) * MAXM * H);
  }
  const int ns2 = std::min(m * k + (a.shared ? 1 : 0), MAXSLOTS);
  if (pdl) {
    if (var == 5) launch_pdl("ds41h moe13v", moe13v_kernel<8>, dim3(g13), dim3(T), smem13, a);
    else if (var == 1) launch_pdl("ds41h moe13", moe13_kernel<3, 8>, dim3(g13), dim3(T), smem13, a);
    else if (var == 2) launch_pdl("ds41h moe13", moe13_kernel<2, 16>, dim3(g13), dim3(T), smem13, a);
    else if (var == 3) launch_pdl("ds41h moe13q", moe13q_kernel, dim3(g13), dim3(T), smem13, a);
    else if (var == 4) launch_pdl("ds41h moe13r", moe13r_kernel<3>, dim3(g13), dim3(T), smem13, a);
    else launch_pdl("ds41h moe13", moe13_kernel<2, 8>, dim3(g13), dim3(T), smem13, a);
    if (a.dbg & 4) return;
    if (moe2v_enabled() && ar_handle < 0) {
      opt_in_smem<moe2v_kernel<6>>(V2<6>::SMEM);
      launch_pdl("ds41h moe2v", moe2v_kernel<6>, dim3(g13), dim3(T), V2<6>::SMEM, a);
    } else if (g2 > 0) launch_pdl("ds41h moe2p", moe2p_kernel, dim3(g2), dim3(T2), 0, a);
    else if (m2r) launch_pdl("ds41h moe2r", moe2r_kernel, dim3(g13), dim3(T), 0, a);
    else if (m2s) launch_pdl("ds41h moe2s", moe2s_kernel, dim3(H / M2S_ROWS, ns2), dim3(T), 0, a);
    else launch_pdl("ds41h moe2", moe2_kernel, dim3(H / 16), dim3(T2), 0, a);
  } else {
    auto st = c10::cuda::getCurrentCUDAStream();
    if (var == 5) moe13v_kernel<8><<<g13, T, smem13, st>>>(a);
    else if (var == 1) moe13_kernel<3, 8><<<g13, T, smem13, st>>>(a);
    else if (var == 2) moe13_kernel<2, 16><<<g13, T, smem13, st>>>(a);
    else if (var == 3) moe13q_kernel<<<g13, T, smem13, st>>>(a);
    else if (var == 4) moe13r_kernel<3><<<g13, T, smem13, st>>>(a);
    else moe13_kernel<2, 8><<<g13, T, smem13, st>>>(a);
    if (a.dbg & 4) return;
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (moe2v_enabled() && ar_handle < 0) {
      opt_in_smem<moe2v_kernel<6>>(V2<6>::SMEM);
      moe2v_kernel<6><<<g13, T, V2<6>::SMEM, st>>>(a);
    } else if (g2 > 0) moe2p_kernel<<<g2, T2, 0, st>>>(a);
    else if (m2r) moe2r_kernel<<<g13, T, 0, st>>>(a);
    else if (m2s) moe2s_kernel<<<dim3(H / M2S_ROWS, ns2), T, 0, st>>>(a);
    else moe2_kernel<<<H / 16, T2, 0, st>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


// ------------------------------------------------------------------ FP8 draft head host
static int g_head_rows = 0;  // DS41H_DRAFT_HEAD_ROWS: 1 = row-stream head_rows_kernel
void head_set_rows(int64_t v) { g_head_rows = int(v); }

void head_fp8(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor y, bool pdl, std::optional<torch::Tensor> norm_w, double eps) {
  using namespace dhead;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == K && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x [m, 5120] bf16, 16-byte aligned rows");
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "1 <= m <= 8");
  const int n = int(scale.size(0));
  TORCH_CHECK(n % 16 == 0 && scale.scalar_type() == at::kFloat && scale.is_contiguous(), "scale fp32 [n], n % 16 == 0");
  TORCH_CHECK(w.is_contiguous() && w.numel() * w.element_size() == long(n) * K, "w: n x 5120 e4m3 bytes (fragment-major)");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == m && y.size(1) == n && y.stride(1) == 1, "y [m, n] bf16");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.w = reinterpret_cast<const uint4*>(w.data_ptr());
  a.scale = scale.data_ptr<float>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  a.n = n;
  if (norm_w.has_value()) {
    TORCH_CHECK(norm_w->scalar_type() == at::kBFloat16 && norm_w->is_contiguous() && norm_w->numel() == K &&
                    reinterpret_cast<uintptr_t>(norm_w->data_ptr()) % 16 == 0,
                "norm weight bf16 [5120]");
    a.nw = reinterpret_cast<const __nv_bfloat16*>(norm_w->data_ptr());
    a.eps = float(eps);
  }
  const c10::cuda::CUDAGuard guard(x.device());
  if (g_head_rows) {
    constexpr int smem_r = KB * 32 * 16 + 7 * R_STAGE + 2 * W * 128 * 4;
    opt_in_smem<head_rows_kernel<7>>(smem_r);
    int nsm = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, x.device().index()) == cudaSuccess && nsm > 0);
    if (pdl) {
      launch_pdl("ds41h head_rows", head_rows_kernel<7>, dim3(nsm), dim3(T), smem_r, a);
    } else {
      head_rows_kernel<7><<<nsm, T, smem_r, c10::cuda::getCurrentCUDAStream()>>>(a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return;
  }
  const int smem = KB * 32 * 16;
  opt_in_smem<head_kernel>(smem);
  const int grid = (n / 16 + W - 1) / W;
  if (pdl) {
    launch_pdl("ds41h head_fp8", head_kernel, dim3(grid), dim3(T), smem, a);
  } else {
    head_kernel<<<grid, T, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


// ------------------------------------------------------------------ fused verification host
void verify_fused(int64_t h, torch::Tensor logits, torch::Tensor draft_sampled, torch::Tensor pos, torch::Tensor exp_idx, torch::Tensor local_pos,
                  torch::Tensor cu, torch::Tensor idx_mapping, torch::Tensor temp, torch::Tensor seeds, torch::Tensor cache, torch::Tensor vp_logit,
                  torch::Tensor vp_lse, torch::Tensor sampled, torch::Tensor num_sampled, int64_t vocab_start, int64_t nspec, int64_t chunk,
                  int64_t timeout_ns, std::optional<torch::Tensor> trace) {
  using namespace dver;
  const Comm& c = get(h);
  const int rows = int(logits.size(0)), reqs = int(idx_mapping.size(0));
  TORCH_CHECK(rows >= 1 && rows <= MAXROWS && rows <= c.slot_rows && reqs >= 1 && reqs <= MAXREQS, "rows / reqs out of range");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2 && logits.stride(1) == 1, "logits bf16 [rows, local]");
  const int local = int(logits.size(1));
  const int G = int((local + chunk - 1) / chunk);
  TORCH_CHECK(chunk == T && G <= c.max_ctas && 2L * rows * G * 3 <= long(MAXS) * c.slot_rows * c.max_ctas * 5, "chunk / ctas");
  for (auto* t : {&draft_sampled, &exp_idx, &local_pos, &cu, &idx_mapping}) TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous(), "int32 index tensors");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.is_contiguous() && seeds.scalar_type() == at::kLong && temp.scalar_type() == at::kFloat, "pos / seeds / temp");
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.dim() == 3 && cache.stride(2) == 1, "draft cache bf16");
  TORCH_CHECK(vp_logit.stride(1) == 1 && vp_lse.stride(1) == 1 && vp_logit.stride(0) == vp_lse.stride(0), "vp stats");
  TORCH_CHECK(sampled.scalar_type() == at::kLong && sampled.stride(1) == 1 && num_sampled.scalar_type() == at::kInt, "outputs");
  Args a{};
  for (int p = 0; p < RANKS; ++p) a.recv[p] = c.recv[p];
  a.state = c.state;
  a.cand = c.cand;
  a.slot_rows = c.slot_rows;
  a.logits = reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr());
  a.ld = logits.stride(0);
  a.draft_sampled = draft_sampled.data_ptr<int>();
  a.pos = reinterpret_cast<const long long*>(pos.data_ptr<int64_t>());
  a.exp_idx = exp_idx.data_ptr<int>();
  a.local_pos = local_pos.data_ptr<int>();
  a.cu = cu.data_ptr<int>();
  a.idx_mapping = idx_mapping.data_ptr<int>();
  a.temp = temp.data_ptr<float>();
  a.seeds = reinterpret_cast<const long long*>(seeds.data_ptr<int64_t>());
  a.cache = reinterpret_cast<const __nv_bfloat16*>(cache.data_ptr());
  a.cs0 = cache.stride(0);
  a.cs1 = cache.stride(1);
  a.vp_logit = vp_logit.data_ptr<float>();
  a.vp_lse = vp_lse.data_ptr<float>();
  a.vp_stride = vp_logit.stride(0);
  a.sampled = reinterpret_cast<long long*>(sampled.data_ptr<int64_t>());
  a.sstride = sampled.stride(0);
  a.num_sampled = num_sampled.data_ptr<int>();
  a.timeout = static_cast<unsigned long long>(timeout_ns);
  a.trace = trace.has_value() ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  a.rank = c.rank;
  a.rows = rows;
  a.reqs = reqs;
  a.local = local;
  a.vocab_start = int(vocab_start);
  a.chunk = int(chunk);
  a.nspec = int(nspec);
  const c10::cuda::CUDAGuard guard(logits.device());
  verify_kernel<<<G, T, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ------------------------------------------------------------------ main_proj host
int64_t main_part_floats() { return int64_t(dmain::SLICES) * dmain::MAXM * dmain::N + dmain::GROUPS; }

void main_proj(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor work, torch::Tensor y, bool pdl) {
  using namespace dmain;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == KS * SLICES && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x [m, 15360] bf16, 16-byte aligned rows");
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "1 <= m <= 8");
  TORCH_CHECK(w.is_contiguous() && w.numel() * w.element_size() == long(N) * KS * SLICES, "w: fragment-major 5120 x 15360 e4m3");
  TORCH_CHECK(s.dim() == 2 && s.size(0) == N / 16 && s.size(1) >= KBT && s.is_contiguous() && s.element_size() == 1, "tile scales [320, >=480]");
  TORCH_CHECK(work.scalar_type() == at::kFloat && work.numel() >= main_part_floats() && work.is_contiguous(), "work buffer");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == m && y.size(1) == N && y.stride(1) == 1, "y [m, 5120] bf16");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.w = reinterpret_cast<const uint4*>(w.data_ptr());
  a.s = reinterpret_cast<const uint8_t*>(s.data_ptr());
  a.kbp = int(s.size(1));
  a.part = work.data_ptr<float>();
  a.ctr = reinterpret_cast<unsigned*>(work.data_ptr<float>() + long(SLICES) * MAXM * N);
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  const int smem = KBS * 32 * 16;
  opt_in_smem<main_kernel>(smem);
  const c10::cuda::CUDAGuard guard(x.device());
  if (pdl) {
    launch_pdl("ds41h main_proj", main_kernel, dim3(GROUPS, SLICES), dim3(T), smem, a);
  } else {
    main_kernel<<<dim3(GROUPS, SLICES), T, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void close_ipc(int64_t ptr) { TORCH_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)) == cudaSuccess); }
void free_buffer(int64_t ptr) { TORCH_CHECK(cudaFree(reinterpret_cast<void*>(ptr)) == cudaSuccess); }

// ------------------------------------------------------------------ drafter router host
int64_t router_scratch_floats() { return int64_t(drouter::MAXM) * drouter::MAXE + 4; }

void router(torch::Tensor x, torch::Tensor gate_w, torch::Tensor bias, std::optional<torch::Tensor> bias_vl, std::optional<torch::Tensor> input_ids,
            int64_t image_lo, double scale, bool renormalize, torch::Tensor scratch, torch::Tensor ids, torch::Tensor wts, bool pdl) {
  using namespace drouter;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == H && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "router: x [m, 5120] bf16, 16-byte aligned rows");
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "router: 1 <= m <= 8");
  TORCH_CHECK(gate_w.scalar_type() == at::kBFloat16 && gate_w.is_contiguous() && gate_w.dim() == 2 && gate_w.size(1) == H, "router: gate_w [E, 5120]");
  const int E = int(gate_w.size(0));
  TORCH_CHECK(E >= 1 && E <= MAXE, "router: E <= 512");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == E, "router: bias fp32 [E]");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m && ids.size(1) <= 32, "router: ids");
  TORCH_CHECK(wts.scalar_type() == at::kFloat && wts.is_contiguous() && wts.sizes() == ids.sizes(), "router: wts");
  TORCH_CHECK(scratch.scalar_type() == at::kFloat && scratch.is_contiguous() && scratch.numel() >= router_scratch_floats(), "router: scratch");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.gw = reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr());
  a.bias = bias.data_ptr<float>();
  if (bias_vl.has_value() && input_ids.has_value() && image_lo > 0) {
    TORCH_CHECK(bias_vl->scalar_type() == at::kFloat && bias_vl->is_contiguous() && bias_vl->numel() == E, "router: bias_vl fp32 [E]");
    TORCH_CHECK(input_ids->is_contiguous() && input_ids->numel() >= m &&
                    (input_ids->scalar_type() == at::kInt || input_ids->scalar_type() == at::kLong),
                "router: input_ids int32/int64 [>= m]");
    a.bias_vl = bias_vl->data_ptr<float>();
    a.ids_in = input_ids->data_ptr();
    a.ids64 = input_ids->scalar_type() == at::kLong ? 1 : 0;
    a.image_lo = long(image_lo);
  }
  float* sp = scratch.data_ptr<float>();
  a.logits = sp + 4;
  a.counter = reinterpret_cast<unsigned*>(sp);
  a.out_ids = ids.data_ptr<int>();
  a.out_w = wts.data_ptr<float>();
  a.scale = float(scale);
  a.renormalize = renormalize ? 1 : 0;
  a.m = m;
  a.k = int(ids.size(1));
  a.E = E;
  const c10::cuda::CUDAGuard guard(x.device());
  const int vpl = (E + 31) / 32;
  auto go = [&](auto kernel) {
    if (pdl) {
      launch_pdl("ds41h router", kernel, dim3(E), dim3(T), 0, a);
    } else {
      kernel<<<E, T, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (vpl <= 4) go(router_kernel<4>);
  else if (vpl <= 8) go(router_kernel<8>);
  else go(router_kernel<16>);
}

// ------------------------------------------------------------------ drafter o-projection host
static int g_lin_stage = 1;  // 1: one-pass x staging (default), 0: the two-pass stage_x_lin (A/B, tests)
void lin_set_stage(int64_t v) { g_lin_stage = int(v); }
static int g_lin_early = 0;  // 1: S1 wo_a triggers wo_b's launch right after its PDL wait (wo_b weight prefetch overlaps wo_a)
void lin_set_early(int64_t v) { g_lin_early = int(v); }

void lin_fp8(torch::Tensor x, int64_t xg, int64_t groups, torch::Tensor w, torch::Tensor s, int64_t n, int64_t k, torch::Tensor y,
             std::optional<torch::Tensor> pos, std::optional<torch::Tensor> cs, bool pdl, int64_t ar_handle, int64_t timeout_ns) {
  using namespace dlin;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(-1) == 1, "x bf16, unit inner stride");
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "1 <= m <= 8");
  TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && xg % 8 == 0, "16-byte aligned x rows / groups");
  TORCH_CHECK(n % (16 * groups) == 0 && k % 32 == 0, "n / k");
  TORCH_CHECK(w.is_contiguous() && w.numel() * w.element_size() == n * k, "w: skinny-packed n x k e4m3");
  TORCH_CHECK(s.dim() == 2 && s.size(0) == n / 16 && s.element_size() == 1 && s.is_contiguous() && s.size(1) >= k / 32 && s.size(1) % 16 == 0,
              "s: [n/16, kbp] ue8m0, kbp % 16 == 0");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == m && y.size(1) == n && y.stride(1) == 1, "y [m, n] bf16");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.xg = xg;
  a.w = reinterpret_cast<const uint4*>(w.data_ptr());
  a.s = reinterpret_cast<const uint8_t*>(s.data_ptr());
  a.kbp = int(s.size(1));
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  a.tiles_per_group = int(n / 16 / groups);
  a.early_dep = (k == 4096 && g_lin_stage && pdl) ? g_lin_early : 0;
  if (cs.has_value()) {
    TORCH_CHECK(pos.has_value() && pos->scalar_type() == at::kLong && pos->is_contiguous() && pos->numel() >= m, "positions int64");
    TORCH_CHECK((cs->scalar_type() == at::kFloat || cs->scalar_type() == at::kBFloat16) && cs->stride(1) == 1 && cs->size(1) >= 2 * HALF_ROPE,
                "cos_sin_cache [*, 64] fp32/bf16");
    a.pos = reinterpret_cast<const long long*>(pos->data_ptr<int64_t>());
    a.cs = cs->data_ptr();
    a.cs_stride = cs->stride(0);
    a.cs_bf16 = cs->scalar_type() == at::kBFloat16 ? 1 : 0;
  }
  if (ar_handle >= 0) {  // fused TP all-reduce (wo_b, N = 5120 only: the 320-tile slot layout of the moe2 fused AR)
    TORCH_CHECK(n == 5120 && groups == 1 && m <= MAXM, "fused all-reduce: wo_b [5120, K] only");
    const Comm& c = get(ar_handle);
    TORCH_CHECK(2L * (5120 / 16) * RANKS * 64 * 8 <= recv_bytes(c.slot_rows), "o-proj all-reduce slots exceed the comm buffer");
    for (int p = 0; p < RANKS; ++p) a.ar_recv[p] = c.recv[p];
    a.ar_state = c.state;
    a.ar_rank = c.rank;
    a.timeout = static_cast<unsigned long long>(timeout_ns);
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto go = [&](auto kernel, int K, int KS) {
    const int smem = (K / 32) * 32 * 16;
    TORCH_CHECK(a.tiles_per_group % (W / KS) == 0, "tiles per group must be a multiple of the tiles per CTA");
    const int grid = int(n / 16) / (W / KS);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
    if (pdl) {
      launch_pdl("ds41h lin_fp8", kernel, dim3(grid), dim3(T), smem, a);
    } else {
      kernel<<<grid, T, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (k == 4096) {
    if (g_lin_stage) go(lin_kernel<4096, 8, true>, 4096, 8);
    else go(lin_kernel<4096, 8, false>, 4096, 8);
  } else if (k == 2048) {
    if (g_lin_stage) go(lin_kernel<2048, 4, true>, 2048, 4);
    else go(lin_kernel<2048, 4, false>, 2048, 4);
  } else {
    TORCH_CHECK(false, "lin_fp8: K must be 4096 (wo_a) or 2048 (wo_b)");
  }
}

// ------------------------------------------------------------------ bf16 verification head host
void head_bf16(torch::Tensor x, std::optional<torch::Tensor> idx, torch::Tensor w, torch::Tensor y, bool pdl) {
  using namespace dvh;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == K && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "head_bf16: x [*, 5120] bf16, 16-byte aligned rows");
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.stride(1) == 1 && w.size(1) == K && w.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && w.size(0) % 16 == 0,
              "head_bf16: w [n, 5120] bf16, n % 16 == 0");
  const int m = int(y.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM && y.scalar_type() == at::kBFloat16 && y.size(1) == w.size(0) && y.stride(1) == 1, "head_bf16: y [m <= 8, n]");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  if (idx.has_value()) {
    TORCH_CHECK(idx->scalar_type() == at::kLong && idx->is_contiguous() && idx->numel() >= m, "head_bf16: idx int64 [m]");
    a.idx = reinterpret_cast<const long long*>(idx->data_ptr<int64_t>());
  } else {
    TORCH_CHECK(x.size(0) >= m, "head_bf16: x rows");
  }
  a.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  a.ldw = w.stride(0);
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  a.n = int(w.size(0));
  const int smem = KB * 32 * 16;
  opt_in_smem<head_bf16_kernel>(smem);
  const int grid = (a.n / 16 + W - 1) / W;
  const c10::cuda::CUDAGuard guard(x.device());
  if (pdl) {
    launch_pdl("ds41h head_bf16", head_bf16_kernel, dim3(grid), dim3(T), smem, a);
  } else {
    head_bf16_kernel<<<grid, T, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Force-load every kernel of this module now (CUDA lazy loading would otherwise load each function at its first launch,
// which may synchronize the context while a cross-GPU waiting kernel of this or another module is running: a hazard
// for the in-kernel collectives (draft_markov, verify, moe2 fused all-reduce) during the warmup / capture sequence).
int64_t preload() {
  int n = 0;
  auto touch = [&](const void* f) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, f) == cudaSuccess) ++n;
  };
  touch(reinterpret_cast<const void*>(draft_markov_kernel));
  touch(reinterpret_cast<const void*>(dmoe::moe13_kernel<2, 8>));
  touch(reinterpret_cast<const void*>(dmoe::moe13_kernel<3, 8>));
  touch(reinterpret_cast<const void*>(dmoe::moe13_kernel<2, 16>));
  touch(reinterpret_cast<const void*>(dmoe::moe13q_kernel));
  touch(reinterpret_cast<const void*>(dmoe::moe13r_kernel<3>));
  touch(reinterpret_cast<const void*>(dmoe::moe2s_kernel));
  touch(reinterpret_cast<const void*>(dmoe::moe2r_kernel));
  touch(reinterpret_cast<const void*>(dmoe::moe2_kernel));
  touch(reinterpret_cast<const void*>(dmoe::moe13v_kernel<8>));
  touch(reinterpret_cast<const void*>(dmoe::moe2v_kernel<6>));
  touch(reinterpret_cast<const void*>(dhead::head_rows_kernel<7>));
  touch(reinterpret_cast<const void*>(dmoe::moe2p_kernel));
  touch(reinterpret_cast<const void*>(dhead::head_kernel));
  touch(reinterpret_cast<const void*>(dver::verify_kernel));
  touch(reinterpret_cast<const void*>(dmain::main_kernel));
  touch(reinterpret_cast<const void*>(drouter::router_kernel<4>));
  touch(reinterpret_cast<const void*>(drouter::router_kernel<8>));
  touch(reinterpret_cast<const void*>(drouter::router_kernel<16>));
  touch(reinterpret_cast<const void*>(dlin::lin_kernel<4096, 8, true>));
  touch(reinterpret_cast<const void*>(dlin::lin_kernel<4096, 8, false>));
  touch(reinterpret_cast<const void*>(dvh::head_bf16_kernel));
  touch(reinterpret_cast<const void*>(dlin::lin_kernel<2048, 4, true>));
  touch(reinterpret_cast<const void*>(dlin::lin_kernel<2048, 4, false>));
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preload", &preload, "load every kernel function of this module now (no lazy loading at first launch)");
  m.def("recv_bytes", &recv_bytes);
  m.def("max_steps", &max_steps);
  m.def("max_chunk", &max_chunk);
  m.def("markov_rank", &markov_rank);
  m.def("alloc_ipc", &alloc_ipc);
  m.def("open_ipc", &open_ipc);
  m.def("close_ipc", &close_ipc);
  m.def("free_buffer", &free_buffer);
  m.def("create", &create);
  m.def("destroy", &destroy);
  m.def("draft_markov", &draft_markov, "fused DSpark Markov drafting over all steps (vocab-parallel, TP=4)");
  m.def("moe_hb_halves", &moe_hb_halves);
  m.def("moe2s_scratch_floats", &moe2s_scratch_floats);
  m.def("verify_fused", &verify_fused, "fused vocab-parallel block verification (TP=4)");
  m.def("main_part_floats", &main_part_floats);
  m.def("main_proj", &main_proj, "DSpark main_proj GEMV (FP8 32x32 blocks, M <= 8)");
  m.def("head_fp8", &head_fp8, "FP8 (per-row scale) draft head GEMM for <= 8 rows");
  m.def("router_scratch_floats", &router_scratch_floats);
  m.def("router", &router, "drafter router: gate GEMV + dsv4 top-k (M <= 8, one launch)");
  m.def("head_bf16", &head_bf16, "bf16 verification head GEMV (M <= 8, fp32 accumulation, optional row gather)");
  m.def("lin_fp8", &lin_fp8, "drafter o-projection: grouped split-K FP8 GEMV (skinny-packed) with optional inverse RoPE, M <= 8");
  m.def("lin_set_stage", &lin_set_stage, "o-projection x staging: 1 = one-pass (default), 0 = two-pass");
  m.def("lin_set_early", &lin_set_early, "o-projection: 1 = wo_a releases wo_b right after its PDL wait (weight prefetch overlap)");
  m.def("head_set_rows", &head_set_rows, "FP8 draft head kernel: 1 = row-stream head_rows_kernel, 0 = head_kernel");
  m.def("moe_decode", &moe_decode, "routed-expert decode GEMMs (M <= 8, raw TP layout): w13+SwiGLU, w2+combine");
}
