// c1hooks_ix.inc: the index / compressor and Engram phases at c1mega's hook points.
// Build with -DC1M_HOOKS="c1hooks_ix.inc" (variant source c1mega_ix.cu; c1mega.cu's default c1hooks.inc stays empty).
// Every hook site is ONE branch + ONE call of a __noinline__ glue function with scalar arguments (no lambdas, no
// address-taken locals of k_layer: m-51 measured +8.4 us per regular layer for the earlier version that passed c1mega's
// lambdas into the phases). The consumer ring counter `it` goes in and comes back by value. Shared memory goes in as byte
// offsets (c1ix::soff / sptr): a pointer parameter would make every access through it generic.
// Phases: c1idx.cuh (owner GPU: X1 / X3 GEMVs, X2 compressor tail, X4 logits, X5 top-512), c1eng.cuh (all GPUs: the
// Engram injection). Per-layer structs: Args::ix -> c1ix::IxLayer, Args::eg -> c1eg::EngArgs (null on other layers).
// Counters: the phases' own arrays + per-CTA epochs (IxLayer::itick, EngArgs::tick); the push warp waits on
// IxLayer::selflag[c] >= ncall (hook 5: every row) or IxLayer::rowsel[n] >= ncall (hook 6: row n).
#if C1M_HOOK == 0
// FILE SCOPE (inside namespace c1m, after Args / Smem / the helpers, before k_layer)
#define C1M_HAS_PHASES 1   // c1mega: index / Engram layers (al[l].ix || al[l].eg) run cons_ix with hooks 2-4
#include "c1idx.cuh"
#include "c1eng.cuh"
#include "c1e0.cuh"
namespace c1ixg {
// hook 1 (producer, owner, index layers): the X1 tile of this CTA (10 chunks of [8 kblocks][16 rows][128 B] bf16), then
// its X3 chunks (5 x [2 kblocks][32 rows][128 B] e4m3 + the row block's 48 scale bytes), in hook 3's consumption order
// part: 0 = X1, X3, X2 (hook 1); 1 = X1 only (hook 8, X1 early); 2 = X3, X2 only (hook 1 with X1 early)
__device__ __noinline__ long produce(const Args* ap, int c, long it, uint32_t ring_u, int so, int NS, uint64_t pol, int part = 0) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1ix::IxLayer& X = *static_cast<const c1ix::IxLayer*>(ap->ix);
  int s = (int)((unsigned)it % (unsigned)NS);   // (32-bit: it < 2^32 stages per launch; then incremental)
  unsigned ph = ((unsigned)it / (unsigned)NS) & 1u;
  const int i = c1ix::x1_tile_of(c), n = c - c1ix::NWK;
  const int n1 = (part != 2 && !(X.skip & 1) && i >= 0 && i < X.nt1) ? 10 : 0;
  const int n3 = (part != 1 && !(X.skip & 2) && c < c1ix::NWK) ? 5 : 0;
  const int n2 = (part != 1 && X.kind == 2 && n >= 0 && n < X.NR && c1ix::x2_has_record(X, n)) ? 8 : 0;
  const CUtensorMap* m1 = i < X.ntc ? X.tm : X.tm + 1;
  const int r1 = 16 * (i < X.ntc ? i : i - X.ntc);
  // one loop over this CTA's chunks: X1 tile (10 x 16 KB), X3 (5 x [8 KB + 48 B]), X2 (8 x 16 KB, the first + 1.5 KB)
  for (int k = 0; k < n1 + n3 + n2; ++k, ++it) {
    const bool x1 = k < n1, x3 = !x1 && k < n1 + n3;
    const int kk = x1 ? k : x3 ? k - n1 : k - n1 - n3;
    const unsigned bytes = x1 ? 16384u : x3 ? (unsigned)(c1ix::X3_S + 48) : (kk ? 16384u : 16384u + c1ix::X2_XB);
    mb_wait(su32(&S->empty[s]), ph ^ 1u);
    mb_expect(su32(&S->full[s]), bytes);
    const uint32_t b = ring_u + s * STAGE, f = su32(&S->full[s]);
    if (x1) {
      tma3d(b, m1, 0, r1, 8 * kk, f, pol);
    } else if (x3) {
      tma3d(b, X.tm + 2, 0, 32 * c, 2 * kk, f, pol);
      bulk(b + c1ix::X3_S, X.wq_s + (size_t)c * 48, 48, f, pol);
    } else {
      bulk(b, X.wk + (size_t)kk * 16 * 512, 16384, f, pol);
      if (kk == 0) {
        bulk(b + c1ix::X2_XO, X.cnorm, 1024, f, pol);
        bulk(b + c1ix::X2_XO + 1024, X.knorm, 256, f, pol);
        bulk(b + c1ix::X2_XO + 1280, c1ix::x2_csrow(X, n), 256, f, pol);
      }
    }
    if (++s == NS) s = 0, ph ^= 1u;
    if (X.tim != nullptr && (k == 0 || k == n1 - 1 || k == n1 + n3 - 1))   // (tests: X1 chunk 0 / last X1 / last X3 issued)
      c1ix::sstamp(X.tim + (size_t)c * 32, k == 0 ? 28 : k == n1 - 1 ? 29 : 30, 0);
  }
  return it;
}
// hook 1 (producer, all GPUs, layer 0 with E0): this GPU's wkv tiles of both Engram layers
__device__ __noinline__ long eg_produce(const Args* ap, int rank, int c, long it, uint32_t ring_u, int so, int NS, uint64_t pol) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1eg::EngArgs& G = *static_cast<const c1eg::EngArgs*>(ap->eg);
  if (G.li >= 0) return it;
  const c1e0::E0Args& E = *static_cast<const c1e0::E0Args*>(G.e0);
  int ss = (int)((unsigned)it % (unsigned)NS);
  unsigned ph = ((unsigned)it / (unsigned)NS) & 1u;
  c1e0::e0_produce(
      E, rank, c, E.tick[c] + 1u, NS,
      [&](unsigned bytes) -> int {
        const int s = ss;
        mb_wait(su32(&S->empty[s]), ph ^ 1u);
        mb_expect(su32(&S->full[s]), bytes);
        ++it;
        if (++ss == NS) ss = 0, ph ^= 1u;
        return s;
      },
      [&](int s, int off, const void* src, unsigned bytes) { bulk(ring_u + s * STAGE + off, src, bytes, su32(&S->full[s]), pol); });
  return it;
}
// hook 3 (consumers, all GPUs, layer 0 with E0): the wkv GEMV of this CTA's tiles -> kv on every GPU
__device__ __noinline__ long eg_front(int so, int rank, int c, int tid, long it, uint32_t ring_u, int NS) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1eg::EngArgs& G = *static_cast<const c1eg::EngArgs*>(S->args.eg);
  if (G.li >= 0) return it;
  const c1e0::E0Args& E = *static_cast<const c1e0::E0Args*>(G.e0);
  const int lane = tid & 31;
  c1ix::Ring R = c1ix::mk_ring(ring_u, su32(&S->full[0]), su32(&S->empty[0]), NS, it);
  const unsigned call = E.tick[c] + 1u;
  c1e0::e0_consume(E, rank, c, tid, call, S->red, [&]() { return c1ix::rwait(R); }, [&]() { c1ix::rrel(R, lane); });
  if (tid == 0) E.tick[c] = call;
  return R.it;
}
// hook 2 (consumers, all GPUs, Engram layers): the injection on this CTA's columns, then A1 in identity mode on X1'.
// Layer 0 with E0 (li < 0): the owner pushes the gathered rows to the peers; the layer runs on unchanged.
__device__ __noinline__ void engram(int so, int rank, int NR, int c, int col0, int ncols, int tid) {
  Smem* S = c1ix::sptr<Smem>(so);
  const Args& a = S->args;
  const c1eg::EngArgs& E = *static_cast<const c1eg::EngArgs*>(a.eg);
  if (E.li < 0) {
    c1e0::e0_push_x(*static_cast<const c1e0::E0Args*>(E.e0), rank, c, tid);
    return;
  }
  double* red = reinterpret_cast<double*>(S->red);
  c1eg::inject(E, rank, a.owner, NR, c, col0, ncols, a.xin, a.rsin, S->post, S->comb, red, red + 512, tid);
  if (tid == 0) S->args.xin = E.zrow, S->args.rsin = E.x1;
  if (tid < NR * 16) {
    const int n = tid >> 4, k = tid & 15;
    S->comb[n][k] = (k % 5 == 0) ? 1.f : 0.f;
    if (k < 4) S->post[n][k] = 0.f;
  }
  cbar();
}
// X1 early: hook 8 streams this CTA's X1 tile right after A1's
// slice, hook 7 runs it right after A2 with B from the image in SMEM (no L2 loads) and arrives IX_KS; hooks 1 / 3 skip X1.
// On index layers the attention waits for the select, so q_a starting X1's ~1-2 us later costs nothing there.
__device__ __forceinline__ bool x1early(const Args& a) {
#ifdef C1IDX_X1EARLY
  return B16(a, wqkv16, 0);
#else
  (void)a;
  return false;
#endif
}
// hook 7 (consumers, owner, index layers, X1 early): X1 on the A2 image -> IX_KS
__device__ __noinline__ long x1front(int so, int c, int tid, long it, uint32_t ring_u, uint32_t img, int NS) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1ix::IxLayer& X = *static_cast<const c1ix::IxLayer*>(S->args.ix);
  const int w = tid >> 5, lane = tid & 31;
  c1ix::Ring R = c1ix::mk_ring(ring_u, su32(&S->full[0]), su32(&S->empty[0]), NS, it);
  unsigned long long* tm = X.tim != nullptr ? X.tim + (size_t)c * 32 : nullptr;   // (tests: 22 hook-7 entry, 1 X1 done)
  c1ix::sstamp(tm, 22, tid);
  c1ix::x1_consume(X, R, S->red, c, w, lane, tm, img);
  c1ix::sstamp(tm, 1, tid);
  c1ix::ix_arr(X.s.cnt, c1ix::IX_KS, tid);
  return R.it;
}
#ifdef C1IDX_ARRM
// several counter arrivals behind one barrier + ONE fence (fence + relaxed adds = a release pattern per add)
__device__ __forceinline__ void arr_m(unsigned* cnt, int tid, bool ks, bool q, bool k) {
  cbar();
  if (tid == 0) {
    __threadfence();
    if (ks) asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(cnt + c1ix::IX_KS), "r"(1u) : "memory");
    if (q) asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(cnt + c1ix::IX_Q), "r"(1u) : "memory");
    if (k) asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(cnt + c1ix::IX_K), "r"(1u) : "memory");
  }
}
#endif
// hook 3 (consumers, owner, index layers): X1 -> IX_KS (unless X1 early), X3 -> IX_Q, the compressor tail on the service
// CTAs -> IX_K
__device__ __noinline__ long front(int so, int c, int tid, int NR, long it, uint32_t ring_u, uint32_t img, int NS,
                                   bool early = false) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1ix::IxLayer& X = *static_cast<const c1ix::IxLayer*>(S->args.ix);
  const int w = tid >> 5, lane = tid & 31;
  c1ix::Ring R = c1ix::mk_ring(ring_u, su32(&S->full[0]), su32(&S->empty[0]), NS, it);
  unsigned long long* tm = X.tim != nullptr ? X.tim + (size_t)c * 32 : nullptr;   // (tests: sub-stamps 0..3, X2 16..21)
  c1ix::sstamp(tm, 0, tid);
#ifdef C1IDX_ARRM
  // the same one arrival per counter per CTA and call, merged behind ONE fence where nothing waits
  // for it: a tile CTA arrives IX_KS right after its X1 (the X2 chain waits for it), everything else at its hook-3 end (an
  // X2 CTA: IX_KS + IX_Q before its x2_row, which waits for IX_KS); each fence waits for the CTA's in-flight ring loads
  // (~1.1 us in situ), and the X3-only CTAs paid three of them in series
  const int i1_ = c1ix::x1_tile_of(c);
  const bool has1 = !early && !(X.skip & 1) && i1_ >= 0 && i1_ < X.nt1;
  const bool x2c = X.kind == 2 && c >= c1ix::NWK && c - c1ix::NWK < NR;
#endif
  if (!early) {
#ifdef C1IDX_X1IMG   // B of k >= 1280 from the A2 image (natural bf16 only: q_a | wkv from the engine's bf16, fixed)
#if !defined(C1M_FIX_B16) || !((C1M_FIX_B16) & 1)
#error "C1IDX_X1IMG: needs the natural bf16 A2 image (C1M_FIX_B16 with bit 0, q_a | wkv from the engine's bf16)"
#endif
    c1ix::x1_consume(X, R, S->red, c, w, lane, tm, 0u, img);
#else
    c1ix::x1_consume(X, R, S->red, c, w, lane, tm);
#endif
    c1ix::sstamp(tm, 1, tid);
#ifdef C1IDX_ARRM
    if (has1) c1ix::ix_arr(X.s.cnt, c1ix::IX_KS, tid);
#else
    c1ix::ix_arr(X.s.cnt, c1ix::IX_KS, tid);
#endif
  }
  c1ix::x3_consume(X, R, img, S->pv, S->red, c, w, lane, tm);
  c1ix::sstamp(tm, 2, tid);
#ifdef C1IDX_ARRM
  const bool ksp = !early && !has1;   // IX_KS still to arrive
  if (x2c) {
    arr_m(X.s.cnt, tid, ksp, true, false);
    c1ix::x2_row(X, c - c1ix::NWK, so + (int)offsetof(Smem, red), tid, R, tid == 0 ? 132u * (X.itick[c] + 1u) : 0u,
                 tm != nullptr ? tm + 16 : nullptr);
    c1ix::ix_arr(X.s.cnt, c1ix::IX_K, tid);
  } else {
    arr_m(X.s.cnt, tid, ksp, true, true);
  }
#else
  c1ix::ix_arr(X.s.cnt, c1ix::IX_Q, tid);
  if (X.kind == 2 && c >= c1ix::NWK && c - c1ix::NWK < NR)   // (waits for IX_KS itself, after its ring inputs)
    c1ix::x2_row(X, c - c1ix::NWK, so + (int)offsetof(Smem, red), tid, R, tid == 0 ? 132u * (X.itick[c] + 1u) : 0u,
                 tm != nullptr ? tm + 16 : nullptr);
  c1ix::ix_arr(X.s.cnt, c1ix::IX_K, tid);
#endif
  c1ix::sstamp(tm, 3, tid);
  return R.it;
}
// hook 4 (consumers, owner, index layers): q quant + logits, top-512 / layer-20 candidates, every CTA waits for IX_SEL;
// scratch = imgp .. + XS_BYTES (free until the attention); then the push warp's flag
__device__ __noinline__ void back(int so, int imgo, int c, int tid, unsigned ncall) {
  Smem* S = c1ix::sptr<Smem>(so);
  const c1ix::IxLayer& X = *static_cast<const c1ix::IxLayer*>(S->args.ix);
  const unsigned ic = tid == 0 ? X.itick[c] + 1u : 0u;   // (only thread 0 spins)
  unsigned long long* tm = X.tim != nullptr ? X.tim + (size_t)c * 32 : nullptr;   // (tests: 4, 5; x45's 8..14)
  c1ix::sstamp(tm, 4, tid);
  c1ix::x45(X, imgo, ic, c, tid, tm != nullptr ? tm + 8 : nullptr, ncall);
  c1ix::sstamp(tm, 5, tid);
  if (tid == 0) {
    X.itick[c] = ic;
    st_rel_sys(X.selflag + c, ncall);
  }
}
// hook 5 (push warp, owner): this index layer's top-k (incl. a new compressed record) is final
__device__ __noinline__ void pushwait(const void* ix, int c, unsigned ncall) {
  const unsigned* f = static_cast<const c1ix::IxLayer*>(ix)->selflag + c;
  while ((int)(c1ix::ld_rlx_sys(f) - ncall) < 0) {   // (relaxed polls, one acquire: no L1 invalidation per poll)
  }
  c1ix::fence_acq_sys();
  __syncwarp();
}
// hook 6 (push warp, owner, index layers): row n's top-k is final (ti row n, tl[n], this layer's new compressed records;
// IxLayer::rowsel[n] >= ncall, released by CTA n's consumers right after its select). Any lane(s); no __syncwarp inside
__device__ __forceinline__ void rowwait(const void* ix, int n, unsigned ncall) {
  const unsigned* f = static_cast<const c1ix::IxLayer*>(ix)->rowsel + n;
  while ((int)(c1ix::ld_rlx_sys(f) - ncall) < 0) {
  }
  c1ix::fence_acq_sys();
}
}  // namespace c1ixg
#elif C1M_HOOK == 1
if (a.ix != nullptr && rank == a.owner) it = c1ixg::produce(&a, c, it, ring_u, c1ix::soff(&S), NS, pol, c1ixg::x1early(a) ? 2 : 0);
else if (a.eg != nullptr) it = c1ixg::eg_produce(&a, rank, c, it, ring_u, c1ix::soff(&S), NS, pol);
#elif C1M_HOOK == 2
if (a.eg != nullptr) c1ixg::engram(c1ix::soff(&S), rank, NR, c, col0, ncols, tid);
#elif C1M_HOOK == 3
if (a.ix != nullptr && rank == a.owner) it = c1ixg::front(c1ix::soff(&S), c, tid, NR, it, ring_u, img, NS, c1ixg::x1early(a));
else if (a.eg != nullptr) it = c1ixg::eg_front(c1ix::soff(&S), rank, c, tid, it, ring_u, NS);
#elif C1M_HOOK == 4
if (a.ix != nullptr && rank == a.owner) c1ixg::back(c1ix::soff(&S), c1ix::soff(imgp), c, tid, ncall);
#elif C1M_HOOK == 5
if (A.ix != nullptr) c1ixg::pushwait(A.ix, c, ncall);
#elif C1M_HOOK == 6
if (A.ix != nullptr) c1ixg::rowwait(A.ix, n, ncall);
#elif C1M_HOOK == 7
if (a.ix != nullptr && rank == a.owner && c1ixg::x1early(a)) it = c1ixg::x1front(c1ix::soff(&S), c, tid, it, ring_u, img, NS);
#elif C1M_HOOK == 8
if (a.ix != nullptr && rank == a.owner && c1ixg::x1early(a)) it = c1ixg::produce(&a, c, it, ring_u, c1ix::soff(&S), NS, pol, 1);
#endif
