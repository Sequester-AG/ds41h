"""PMV_NT: the narrow-tile consumer for production's moe_gemm16 (hybrid remainder kernel), as source patches of the pmv copy
(build/pmv_src, see pmv.py). Every patch is gated by #ifdef PMV_NT: without the define the copy is production's code.

What changes (same per-element math: the same expand / HGMMA sequence in the same K order -> bit-identical by design):
1. w16n_mainloop<MODE, N>: production's w16_mainloop with the block boundary taken off the critical path. The next k64
   block's stage / scale-slot barrier wait and its FP4 words + scale multipliers (lds64 x 4, funnelshift x 8, lds_u8 x 4)
   are loaded at step 1 of the current block into a second register set (N <= PMV_NT_LOOKN; wider tiles keep
   production's order), and the register fences that order F against the async wgmma carry no "memory" clobber (so
   the compiler may hoist the shared-memory loads).
2. w16n_dispatch: mainloop + epilogue specialized per tile size N on views of the kernel-scope accumulators (N / 2 per
   chain; the rest dead in that path), so narrow tiles need fewer live registers (production: 168 registers, spills).
   (A first version with arrays local to each N put all of them on the stack: 2.1 KB per thread.)
3. (separate switch PMV_NT_BOX16) Token boxes in 16-row granularity: floor(n / 64) boxes of 64 rows + ceil((n % 64) / 16) boxes of 16 rows (a second
   tensor map tm_b16), instead of ceil(n / 64) boxes of 64 rows (a 16-token tile loaded 64 rows).
"""
from __future__ import annotations

LOOKN_DEFAULT = 96

# ---------------------------------------------------------------- GemmArgs: the 16-row token-box tensor map
P_ARGS = ("""  CUtensorMap tm_b, tm_w, tm_s;
  const uint8_t* w;        // expert weights""", """  CUtensorMap tm_b, tm_w, tm_s;
#ifdef PMV_NT_BOX16
  CUtensorMap tm_b16;      // PMV_NT (moe_gemm16): token rows, box {128 B, 16 rows} SW128
#endif
#ifdef PMV_NT_BOXN
  CUtensorMap tm_bn[8];    // PMV_NT_BOXN (moe_gemm16): token rows, box {128 B, 16 (j + 1) rows} SW128 (one TMA op per stage)
  int bn_rows[8];          //   their box rows (min(16 (j + 1), activation rows))
#endif
  const uint8_t* w;        // expert weights""")

# ---------------------------------------------------------------- host: encode tm_b16 in launch_gemm16
P_HOST = ("""  a.brows = act_rows < 64 ? act_rows : 64;
  encode_u8_2d(&a.tm_b, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
  encode_u8_2d(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), 32u, MODE == 1 ? 64u : 256u, false, "w");""",
          """  a.brows = act_rows < 64 ? act_rows : 64;
  encode_u8_2d(&a.tm_b, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), true, "act");
#ifdef PMV_NT_BOXN
  for (int j = 0; j < 8; ++j) {
    a.bn_rows[j] = act_rows < 16 * (j + 1) ? act_rows : 16 * (j + 1);
    encode_u8_2d(&a.tm_bn[j], a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.bn_rows[j]), true, "actn");
  }
#endif
#ifdef PMV_NT_BOX16
  if (act_rows >= 64) encode_u8_2d(&a.tm_b16, a.act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, 16u, true, "act16");
#endif
  encode_u8_2d(&a.tm_w, a.w, uint64_t(a.w_rstride), uint64_t(rows_total), 32u, MODE == 1 ? 64u : 256u, false, "w");""")

# ---------------------------------------------------------------- TMA warp: 64-row + 16-row token boxes
P_TMA1 = ("""        const int nbox = (nvalid + a.brows - 1) / a.brows;
        const int w0 = e * rows_e + (MODE == 1 ? 128 * nb : 256 * nb), w1 = w0 + a.I;
        const uint32_t bbytes = uint32_t(nbox * a.brows * 128);""", """        const int nbox = (nvalid + a.brows - 1) / a.brows;
        const int w0 = e * rows_e + (MODE == 1 ? 128 * nb : 256 * nb), w1 = w0 + a.I;
#if defined(PMV_NT_BOXN)
        // one token box of the tile's size, rounded up to 16 rows (map nsel - 1)
        const int nsel_b = (nvalid + 15) / 16;
        const uint32_t bbytes = uint32_t(a.bn_rows[nsel_b - 1] * 128);
#elif defined(PMV_NT_BOX16)
        // 64-row boxes for the full 64-row chunks, 16-row boxes for the rest (a.brows < 64: tiny calls keep one map)
        const int n64 = a.brows == 64 ? nvalid / 64 : nbox, n16 = a.brows == 64 ? (nvalid - 64 * n64 + 15) / 16 : 0;
        const uint32_t bbytes = uint32_t((n64 * a.brows + 16 * n16) * 128);
#else
        const uint32_t bbytes = uint32_t(nbox * a.brows * 128);
#endif""")
P_TMA2 = ("""          mbar_expect_tx_s(fb, bbytes + uint32_t(MODE == 1 ? 4096 * nbw : W_ASTAGE));
          for (int j = 0; j < nbox; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
          if constexpr (MODE == 1) {
#endif""", """          mbar_expect_tx_s(fb, bbytes + uint32_t(MODE == 1 ? 4096 * nbw : W_ASTAGE));
#if defined(PMV_NT_BOXN)
          tma_load_2d(st, &a.tm_bn[nsel_b - 1], kb * 128, pos0, fb);
#elif defined(PMV_NT_BOX16)
          for (int j = 0; j < n64; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
          for (int j = 0; j < n16; ++j)
            tma_load_2d(st + (n64 * 64 + 16 * j) * 128, &a.tm_b16, kb * 128, pos0 + 64 * n64 + 16 * j, fb);
#else
          for (int j = 0; j < nbox; ++j) tma_load_2d(st + j * a.brows * 128, &a.tm_b, kb * 128, pos0 + a.brows * j, fb);
#endif
          if constexpr (MODE == 1) {
#endif""")

# ---------------------------------------------------------------- consumer: dispatch the whole tile (mainloop + epilogue)
P_CONS1 = ("""      w16_dispatch<MODE>(nsel, c, a, acc0, acc1);
    }""", """#ifdef PMV_NT
      w16n_dispatch<MODE>(nsel, c, a, acc0, acc1, nvalid, nb, pos0, rslot);
#else
      w16_dispatch<MODE>(nsel, c, a, acc0, acc1);
#endif
    }""")
P_CONS2 = ("""      w16_epilogue<MODE>(c, a, acc0, acc1, nvalid, nb, pos0, rslot);""", """#ifndef PMV_NT
      w16_epilogue<MODE>(c, a, acc0, acc1, nvalid, nb, pos0, rslot);
#endif""")

# ---------------------------------------------------------------- new device code, inserted before moe_gemm16_kernel
NEW_CODE = r'''
#ifdef PMV_NT
#ifndef PMV_NT_LOOKN
#define PMV_NT_LOOKN 96
#endif
__device__ __forceinline__ void reg_fence_r(unsigned& r) { asm volatile("" : "+r"(r)); }  // register order only (no memory clobber)
// non-blocking probe of an mbarrier phase (PMV_NT_TRY: prefetch the next block only if its stage has already landed)
__device__ __forceinline__ bool mbar_test_s(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}

// production's w16_mainloop (same expand / HGMMA sequence and K order) with the next k64 block's barrier wait and FP4 words /
// scale multipliers loaded at step 1 of the current block (LOOK), accumulators of N / 2 per chain
template <int MODE, int N>
__device__ __forceinline__ void w16n_mainloop(W16Cons& c, const GemmArgs& a, float (&acc0)[N / 2], float (&acc1)[N / 2]) {
  using namespace w16;
  constexpr bool LOOK = N <= PMV_NT_LOOKN;
  constexpr int D = 1, NBUF = 2;
  constexpr int C1 = MODE == 1 ? 128 : 64;
  const int nkb = a.K / 64;
  auto sgroup_first = [&](int kb) { return MODE == 1 ? (kb & 7) == 0 : kb == 0; };
  auto sgroup_last = [&](int kb) { return (MODE == 1 && (kb & 7) == 7) || kb == nkb - 1; };
  const unsigned rot = 4u * unsigned(c.t);
  unsigned w[2][2][2], m[2][2];      // this block's rotated FP4 words / bf16x2 scale multipliers [chain][row][word]
  unsigned wn[2][2][2], mn[2][2];    // the next block's (LOOK)
  unsigned F[NBUF][2][4];
  auto bar = [&](int i) { return c.sb + OFF_BAR + 8 * i; };
  auto acquire_at = [&](int kb, int stage, unsigned phase) {
#ifdef DS41H_MOE_TRACE
    const long long tw0 = clock64();
#endif
    mbar_wait_s(bar(B_FULL + stage), phase);
    if (sgroup_first(kb)) mbar_wait_s(bar(B_SFULL + c.ss), c.sph);
#ifdef DS41H_MOE_TRACE
    c.twait += clock64() - tw0;
#endif
  };
  auto load_at = [&](int kb, int stage, unsigned (&ww)[2][2][2], unsigned (&mm)[2][2]) {
    const uint32_t sa = c.sb + stage * STAGE + W_BSTAGE;
    const uint32_t ss = c.sb + OFF_SR + c.ss * S_SLOT + (MODE == 1 ? 2 * (kb & 7) : 2 * kb);
    const int rs = MODE == 1 ? 16 : a.s_rstride;
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const uint2 q0 = lds64(sa + c.aoff0 + ch * C1 * 32), q1 = lds64(sa + c.aoff1 + ch * C1 * 32);
      ww[ch][0][0] = __funnelshift_r(q0.x, q0.x, rot), ww[ch][0][1] = __funnelshift_r(q0.y, q0.y, rot);
      ww[ch][1][0] = __funnelshift_r(q1.x, q1.x, rot), ww[ch][1][1] = __funnelshift_r(q1.y, q1.y, rot);
      mm[ch][0] = fp4_bf16_mult(lds_u8(ss + c.soff0 + ch * C1 * rs));
      mm[ch][1] = fp4_bf16_mult(lds_u8(ss + c.soff1 + ch * C1 * rs));
    }
    if (sgroup_last(kb)) {
      fence_proxy_async();
      __syncwarp();
      if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_SEMPTY + c.ss));
      if (++c.ss == NSR) c.ss = 0, c.sph ^= 1;
    }
  };
  auto expand = [&](int s, unsigned (&Fb)[2][4]) {
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const unsigned x0 = w[ch][0][s >> 1], x1 = w[ch][1][s >> 1];
      if ((s & 1) == 0) {
        Fb[ch][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x0), m[ch][0]), Fb[ch][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x0), m[ch][0]);
        Fb[ch][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<0>(x1), m[ch][1]), Fb[ch][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<1>(x1), m[ch][1]);
      } else {
        Fb[ch][0] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x0), m[ch][0]), Fb[ch][2] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x0), m[ch][0]);
        Fb[ch][1] = bf16x2_mul_rn(fp4_pair_bf16_raw<2>(x1), m[ch][1]), Fb[ch][3] = bf16x2_mul_rn(fp4_pair_bf16_raw<3>(x1), m[ch][1]);
      }
    }
  };
  auto release_prev_stage = [&]() {
    if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_EMPTY + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));
  };
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  acquire_at(0, c.stage, c.phase);
  load_at(0, c.stage, w, m);
  expand(0, F[0]);
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    const uint64_t desc = desc_sw128(c.sb + c.stage * STAGE);
    const int nst = c.stage + 1 == NSTAGE ? 0 : c.stage + 1;
    const unsigned nph = c.stage + 1 == NSTAGE ? (c.phase ^ 1u) : c.phase;
    bool pre = false;  // the next block's words / scales are in wn / mn
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      wgmma_fence();
      WgmmaRS16<N>::mma(acc0, F[s % NBUF][0], desc + 2 * s, (s > 0 || kb > 0) ? 1 : 0);
      WgmmaRS16<N>::mma(acc1, F[s % NBUF][1], desc + 2 * s, (s > 0 || kb > 0) ? 1 : 0);
      wgmma_commit();
      wgmma_wait<D>();  // groups <= this step - 1 retired: F[(s + 1) % 2] free; at s == 0 block kb - 1 retired
#pragma unroll
      for (int ch = 0; ch < 2; ++ch)
#pragma unroll
        for (int j = 0; j < 4; ++j) reg_fence_r(F[(s + 1) % NBUF][ch][j]);
      if (s == 0 && kb > 0) release_prev_stage();
      if (LOOK && s == 1 && kb + 1 < nkb) {
#ifdef PMV_NT_TRY
        // only if the stage (and, at a group start, the scale slot) has already landed: never stall here
        pre = mbar_test_s(bar(B_FULL + nst), nph) && (!sgroup_first(kb + 1) || mbar_test_s(bar(B_SFULL + c.ss), c.sph));
        if (pre) load_at(kb + 1, nst, wn, mn);
#else
        acquire_at(kb + 1, nst, nph);
        load_at(kb + 1, nst, wn, mn);
        pre = true;
#endif
      }
      if (s < 3) {
        expand(s + 1, F[(s + 1) % NBUF]);
      } else {
        c.stage = nst, c.phase = nph;
        if (kb + 1 < nkb) {
          if (LOOK && pre) {
#pragma unroll
            for (int ch = 0; ch < 2; ++ch)
#pragma unroll
              for (int r = 0; r < 2; ++r) w[ch][r][0] = wn[ch][r][0], w[ch][r][1] = wn[ch][r][1], m[ch][r] = mn[ch][r];
          } else {
            acquire_at(kb + 1, c.stage, c.phase);
            load_at(kb + 1, c.stage, w, m);
          }
          expand(0, F[0]);
        }
      }
    }
  }
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) reg_fence(acc0[i]), reg_fence(acc1[i]);
  release_prev_stage();
}

// production's w16_epilogue for accumulators of N / 2 per chain (BN -> N; same math, staging and pass headers)
template <int MODE, int N>
__device__ __forceinline__ void w16n_epilogue(W16Cons& c, const GemmArgs& a, const float (&acc0)[N / 2], const float (&acc1)[N / 2],
                                              int nvalid, int nb, int pos0, int rslot) {
  using namespace w16;
  const int lane = threadIdx.x & 31, m8 = lane >> 3, r8 = lane & 7;
  const int wg = (threadIdx.x >> 7) & 1, wq = (threadIdx.x >> 5) & 3;
  const int npass = (nvalid + EPI_TOK - 1) / EPI_TOK;
#pragma unroll
  for (int p = 0; p < (N + EPI_TOK - 1) / EPI_TOK; ++p) {
    if (p >= npass) break;
    const int c0 = p * EPI_TOK, np = min(EPI_TOK, nvalid - c0);
    const int nch = (np + 7) >> 3;
#pragma unroll
    for (int ch = 0; ch < (MODE == 1 ? 1 : 2); ++ch) {
      const float(&v)[N / 2] = ch == 0 ? acc0 : acc1;
      const uint32_t ebase = w16_epi_acquire(c, wg);
#pragma unroll
      for (int i = 0; i < EPI_TOK / 8; i += 2) {
        if (i >= nch || c0 / 8 + i >= N / 8) break;
        const int j0 = min(c0 / 8 + i, N / 8 - 2), j1 = j0 + 1;
        unsigned r[4];
        if constexpr (MODE == 1) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int idx = 4 * (q < 2 ? j0 : j1) + 2 * (q & 1);
            float h[2];
#pragma unroll
            for (int hv = 0; hv < 2; ++hv) {
              const float gt = fminf(acc0[idx + hv], a.limit);
              const float up = fmaxf(fminf(acc1[idx + hv], a.limit), -a.limit);
              h[hv] = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
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

// the kernel-scope accumulator arrays (as production), viewed as N / 2 per chain by each tile size: elements beyond N / 2
// are dead in that path (the first HGMMA of a tile overwrites with scale_d 0), so narrow paths need no more registers
template <int MODE, int NS = w16::BN / 16>
__device__ __forceinline__ void w16n_dispatch(int nsel, W16Cons& c, const GemmArgs& a, float (&acc0_all)[w16::BN / 2],
                                              float (&acc1_all)[w16::BN / 2], int nvalid, int nb, int pos0, int rslot) {
  if constexpr (NS >= 1) {
    if (nsel == NS) {
      float(&acc0)[8 * NS] = *reinterpret_cast<float(*)[8 * NS]>(&acc0_all[0]);
      float(&acc1)[8 * NS] = *reinterpret_cast<float(*)[8 * NS]>(&acc1_all[0]);
      w16n_mainloop<MODE, 16 * NS>(c, a, acc0, acc1);
      w16n_epilogue<MODE, 16 * NS>(c, a, acc0, acc1, nvalid, nb, pos0, rslot);
      return;
    }
    w16n_dispatch<MODE, NS - 1>(nsel, c, a, acc0_all, acc1_all, nvalid, nb, pos0, rslot);
  }
}
#endif  // PMV_NT

'''
P_INSERT = ("""template <int MODE>
__global__ void __launch_bounds__(NTHREADS, 1) moe_gemm16_kernel(const __grid_constant__ GemmArgs a) {""",
            NEW_CODE + """template <int MODE>
__global__ void __launch_bounds__(NTHREADS, 1) moe_gemm16_kernel(const __grid_constant__ GemmArgs a) {""")

PATCHES = [P_ARGS, P_HOST, P_TMA1, P_TMA2, P_CONS1, P_CONS2, P_INSERT]


# ---------------------------------------------------------------- PMV_ARV: every consumer thread releases the stage
# (rs_ubench MODE 6 + F 32: a lane-0 mbarrier.arrive per warp per k64 block costs ~60-80 cycles per k16 step; with all 256
# consumer threads arriving (EMPTY / SEMPTY counts 256) the protocol costs ~20). Applied to every EMPTY / SEMPTY release of
# the moe_gemm16 section (production's w16_mainloop, w16_skip_tile and PMV_NT's w16n_mainloop) and the two inits.
ARV_REGION = ("// W4A16 grouped GEMM, 256-row tiles", "// W4A16 small-batch grouped GEMM")
ARV_SUBS = [
    ("if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_SEMPTY + c.ss));", "if (W16_ARV_ALL || (threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_SEMPTY + c.ss));"),
    ("if ((threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_EMPTY + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));",
     "if (W16_ARV_ALL || (threadIdx.x & 31) == 0) mbar_arrive_s(bar(B_EMPTY + (c.stage == 0 ? NSTAGE - 1 : c.stage - 1)));"),
    ("if ((threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_EMPTY + c.stage));",
     "if (W16_ARV_ALL || (threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_EMPTY + c.stage));"),
    ("if ((threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_SEMPTY + c.ss));",
     "if (W16_ARV_ALL || (threadIdx.x & 31) == 0) mbar_arrive_s(c.sb + OFF_BAR + 8 * (B_SEMPTY + c.ss));"),
    ("for (int s = 0; s < NSTAGE; ++s) mbar_init(&bars[B_FULL + s], 1), mbar_init(&bars[B_EMPTY + s], 8);",
     "for (int s = 0; s < NSTAGE; ++s) mbar_init(&bars[B_FULL + s], 1), mbar_init(&bars[B_EMPTY + s], W16_ARV_ALL ? 256 : 8);"),
    ("for (int s = 0; s < NSR; ++s) mbar_init(&bars[B_SFULL + s], 1), mbar_init(&bars[B_SEMPTY + s], 8);",
     "for (int s = 0; s < NSR; ++s) mbar_init(&bars[B_SFULL + s], 1), mbar_init(&bars[B_SEMPTY + s], W16_ARV_ALL ? 256 : 8);"),
]
ARV_DEF = """#ifdef PMV_ARV
#define W16_ARV_ALL true
#else
#define W16_ARV_ALL false
#endif
"""


def arv_patch(txt: str) -> str:
    """PMV_ARV substitutions inside the moe_gemm16 section (after the other patches inserted PMV_NT's code there)."""
    a, b = txt.index(ARV_REGION[0]), txt.index(ARV_REGION[1])
    reg = txt[a:b]
    for o, n in ARV_SUBS:
        assert reg.count(o) >= 1, o[:60]
        reg = reg.replace(o, n)
    return txt[:a] + ARV_DEF + reg + txt[b:]


# ---------------------------------------------------------------- PMV_WAITFAST / PMV_WAITWD: cheaper mbarrier waits
# PMV_WAITWD: the same 20-s watchdog with %globaltimer read every 4,096 failed polls (the safe variant)
# production's mbar_wait_s re-reads %globaltimer on every failed try_wait (20-s watchdog -> trap); PMV_WAITFAST spins on try_wait
# only (timing experiment: does the polling granularity cost the consumers?)
P_WAIT = ("""__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {
  asm volatile(""", """__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {
#ifdef PMV_WAITWD
  // production's 20-s watchdog, but %globaltimer read only every 4,096 failed polls (production: every poll)
  asm volatile(
      "{\\n .reg .pred p;\\n .reg .u64 t0, t1;\\n .reg .u32 n;\\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"
      " @p bra.uni LAB_DONE;\\n"
      " mov.u64 t0, %%globaltimer;\\n"
      " mov.u32 n, 0;\\n"
      "LAB_WAIT:\\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"
      " @p bra.uni LAB_DONE;\\n"
      " add.u32 n, n, 1;\\n"
      " and.b32 n, n, 4095;\\n"
      " setp.ne.u32 p, n, 0;\\n"
      " @p bra.uni LAB_WAIT;\\n"
      " mov.u64 t1, %%globaltimer;\\n"
      " sub.u64 t1, t1, t0;\\n"
      " setp.lt.u64 p, t1, 20000000000;\\n"
      " @p bra.uni LAB_WAIT;\\n"
      " trap;\\n"
      "LAB_DONE:\\n}" ::"r"(bar), "r"(parity)
      : "memory");
  return;
#endif
#ifdef PMV_WAITCNT2
  // PMV_WAITCNT with the give-up record inside the asm (predicated store on the loop exit; no C++ branch after the wait)
  asm volatile("{\\n .reg .pred p, q;\\n .reg .u32 n;\\n mov.u32 n, 0;\\nLAB_WC2:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"
               " @p bra.uni LAB_WC2_D;\\n add.u32 n, n, 1;\\n setp.lt.u32 p, n, 1073741824;\\n @p bra.uni LAB_WC2;\\n"
               " setp.ge.u32 q, n, 1073741824;\\n @q st.global.u32 [%2], %0;\\nLAB_WC2_D:\\n}\\n" ::"r"(bar), "r"(parity),
               "l"(&pmv_wait_err)
               : "memory");
  return;
#endif
#ifdef PMV_WAITCNT
  // one counted spin loop with a single exit: after 2^30 failed polls (~minutes) it records the stuck barrier in pmv_wait_err
  // (host-readable via the extension's wait_err()) and falls through instead of hanging (no %globaltimer, no trap)
  unsigned n_;
  asm volatile("{\\n .reg .pred p;\\n mov.u32 %0, 0;\\nLAB_WC:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n @p bra.uni LAB_WC_D;\\n"
               " add.u32 %0, %0, 1;\\n setp.lt.u32 p, %0, 1073741824;\\n @p bra.uni LAB_WC;\\nLAB_WC_D:\\n}\\n"
               : "=r"(n_) : "r"(bar), "r"(parity) : "memory");
  if (n_ >= 1073741824u) pmv_wait_err = bar | 0x80000000u;
  return;
#endif
#ifdef PMV_WAITFAST
  asm volatile("{\\n .reg .pred p;\\nLAB_WF:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra.uni LAB_WF;\\n}\\n" ::"r"(bar),
               "r"(parity)
               : "memory");
  return;
#endif
  asm volatile(""")
PATCHES.append(P_WAIT)


P_WAITERR = ("""__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {""",
             """#if defined(PMV_WAITCNT) || defined(PMV_WAITCNT2)
__device__ unsigned pmv_wait_err = 0;  // PMV_WAITCNT: a barrier whose wait gave up (bit 31 set), else 0
#endif
__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {""")
PATCHES.insert(PATCHES.index(P_WAIT), P_WAITERR)
