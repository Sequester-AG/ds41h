// x2t: x2n's consumers + header-4 stages in the routed loop: shared
// GEMM2 (bf16 w2) tile ti = val / 2 on expert warp (ti + X2T_SHW0) % 10, which takes half 0 and then half 1 (the next stage,
// streamed right after it) into one accumulator with sh2_16 in the post-END code's order (k32 blocks 0..9, then 10..17);
// the other warps release both stages. The post-END shared section then runs only for the fp8 shared path. S.shbf written
// before XEXIT's barrier (warps 10 / 11 can own a tile), read by the combine after the cbar. Bitwise vs x2n.
#ifndef X2T_SHW0
#define X2T_SHW0 7
#endif
// warp wx takes k32 blocks 4 wx .. 4 wx + 3 of every chunk
//    the scale bytes 40 kc + 4 wx .. +3 of the item's [2][16][160 B] scale rows (x2j's double buffer); the item's 10
//    partials go through ONE slot (10 KB) to the epilogue warp rcnt % 10, summed in warp order;
//  - GEMM2 routed tiles rotated over the 10 warps; every stage of the routed loop released with crel1() by all 10.
// Outside (base warps only; crel() arrives with xrc_of(w): warps 0 / 1 count 2): the shared GEMM1 halves before XENTER,
// the shared GEMM2 after XEXIT, as x2j.
// Same operands and per-element products as x2j; NOT bitwise (a warp's partial covers 16 k32 blocks of K in 4 runs of 4
// instead of 20 in 5 runs of 4, and 10 partials are summed): same math, fp32 order changed.
// The routed loop on 16 consumer warps (wx = 0..15)
//  - GEMM1: warp wx takes k32 blocks 2 wx, 2 wx + 1 of every chunk; the item's 16 partials go through ONE 16 KB slot
//    (mbarriers full 16 / free 1) to the epilogue warp rcnt % 16;
//  - GEMM2 routed tiles rotated over the 16 warps; every stage of the loop released with crel1() by all 16;
//  - x2j's stage headers off the ring + GEMM1 scale double buffer, x2i's epilogue fences, x2h's B rows.
// Outside (base warps only, crel() = count 2): the shared GEMM1 halves before XENTER, the shared GEMM2 after XEXIT, as x2j.
// Same operands and per-element products, but NOT bitwise vs x2i: a warp's partial covers 2 k32 blocks instead of 4,
// so the fp32 summation order changes (accept on moe_out / xm2 diffs vs the engine of x2i's size).
#ifndef C1M_XN
#error "c1moe_x2n_consume.inc needs c1mega_xn.cu with C1M_XN 2 (the 10-warp expert phase)"
#endif
#ifndef C1M_XFRAG
#error "c1moe_x2n_consume.inc: define C1M_XFRAG in the wrapper (before c1mega_xn.cu), else warps 10-11 exit at kernel start"
#endif
#if C1M_XROLE == 0
// x2j: x2i's consumers + x2f's
// stage headers off the ring (attention scratch) and the GEMM1 scale double buffer (attention scratch / S.red, item & 1).
// Same math: bitwise-identical outputs.
// --- x2i:
// x2i: x2h's
// consumers without the two sequentially-consistent fences per routed item / shared half: (1) the item epilogue's __threadfence before __syncwarp + red.release.gpu(C_H + j) goes (the release RED is
// cumulative over the warp's himg stores, ordered by bar.warp.sync; the producer's ld.acquire(C_H) + fence.proxy.async.global
// before the bulk copy of himg stays as in x2b); (2) the shared halves' ticket: __threadfence + relaxed atomicAdd + the
// reader's __threadfence -> one atom.acq_rel.gpu. Same math: bitwise-identical outputs.
// --- x2h:
// x2h: x2b's
// consumers with the routed GEMM1 / GEMM2 B-fragment loads unconditional from row min(g, NR - 1) and the GEMM1
// accumulators reset once per item instead of a conditional zeroing per chunk (see below).
// Outputs bitwise identical.
// --- x2b:
// c1moe_x2b_consume.inc
// the x2a change (below) + a header-driven stage loop: routed GEMM1 chunks (items grabbed by the producer) and
// GEMM2 routed stages (interleaved as soon as an expert's h is complete) in whatever order the producer streamed
// them, until the END stage; shared GEMM1 halves before, GEMM2 shared after, as today. Per-element math and
// summation order unchanged: bitwise-identical outputs.
// x2a: Change vs today: in FULL builds (attention scratch present) a routed
// GEMM1 item no longer ends in a CTA barrier. Each warp writes its partials into one of two slots in the attention
// scratch (free between top-6 and the combine) and arrives on the slot's mbarrier; only the item's epilogue warp
// waits, sums the 8 partials in warp order (the same order: bitwise-identical h) and frees the slot. So the
// epilogue (SwiGLU, himg stores, __threadfence, C_H arrival) leaves the other warps' critical path.
// c1moe_consume.inc: the expert phase (CONSUMER warps 0-7), textually included inside k_layer's consumer layer loop
// (fpr/kernels/c1mega/c1mega.cu), between top-6 (stamp 10) and the combine (stamp 12). Contract owner: the C1 persistent-
// kernel session
//  in:  the f image (imgp / img: NR rows x D, f16 x 2^S.pv[n], P8 order, row stride IMS), S.pv[n], the expert list
//       S.nexp, S.xid[j] (expert id), S.xw[j][n] (routing weight of row n, 0 if not chosen), S.xk[j][n] (top-k slot of
//       row n or -1); this CTA's hidden columns col0 .. col0 + ncols (nt tiles of 16); Args a (w13 / s13 / w2 / s2 FP4 raw,
//       sw13 / sw13_s / sw2 / sw2_s fp8, hsh / hmax / himg / shp scratch, cnt counters C_H / C_SH / C_SHT).
//  out: S.Ybf[n][k][j] (bf16 routed output of row n's k-th expert, routing weight applied, columns j < ncols) and
//       S.shbf[n][j] (bf16 shared-expert output); the include is followed by cbar() + the combine.
//  ring: cwait() / crel() per stage in the producer's order; grid counters arrive() / waitc() / red_rel with epochs
//       k * ncall, the same arrivals on every rank; stamp(11) = GEMM1 done (timeline).
  // ---------------------------------------------------------------------------------------------- GEMM1 shared halves
  {
    for (int gq = g1first(c); gq < SH_NH; gq += NWK) {   // shared half (the default fragment's code, verbatim)
      float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const int u = gq >> 1, hf = gq & 1;
        if (B16(a, sw13_16, 3)) {   // the engine's bf16 w1 | w3: 10 chunks of 256 K
          for (int kc = 0; kc < 10; ++kc) {
            const uint32_t st = cwait();
            sh16_chunk(acc, st, img, 2560 * hf + 256 * kc, NR, w, lane);
            crel();
          }
        } else {
          for (int kc = 0; kc < 5; ++kc) {
            const uint32_t st = cwait();
            dense_chunk<2>(acc, st, 8192, 16, st + L_SH1_S, 16, img, 2560 * hf + 512 * kc, NR, w, lane);
            crel();
          }
        }
        reduce_warps<2>(acc, S.red, w, lane);
        if (w == 0)
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int row = 16 * ti + g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
              if (n < NR) a.shp[((size_t)gq * 32 + row) * NRM + n] = acc[ti][i] * pow2i(-S.pv[n]);
            }
        cbar();
        if (tid == 0) {   // x2i: one acq_rel RMW (release: this CTA's shp rows, ordered by the cbar; acquire: the other half's)
          unsigned tk;
          asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(tk) : "l"(a.cnt + C_SHT + u) : "memory");
          S.ticket = (int)(tk & 1u);
        }
        cbar();
        if (S.ticket) {  // second half done: sum the halves in order, the shared expert's gu / SwiGLU (bf16 rounding points)
          if (tid < 16 * NRM) {   // warps 0, 1 whole (shuffles); lanes of rows >= NR idle
            const int i = tid & 15, n = tid >> 4;
            float m = 0.f;
            if (n < NR) {
              const float g0 = __ldcg(a.shp + ((size_t)(2 * u) * 32 + i) * NRM + n);
              const float g1 = __ldcg(a.shp + ((size_t)(2 * u + 1) * 32 + i) * NRM + n);
              const float u0 = __ldcg(a.shp + ((size_t)(2 * u) * 32 + 16 + i) * NRM + n);
              const float u1 = __ldcg(a.shp + ((size_t)(2 * u + 1) * 32 + 16 + i) * NRM + n);
              const float gf = bfr(g0 + g1), uf = bfr(u0 + u1);
              const float ga = fminf(gf, 10.f), ua = fminf(fmaxf(uf, -10.f), 10.f);
              const bf16 hb = __float2bfloat16_rn(ga / (1.f + expf(-ga)) * ua);
              a.hsh[(size_t)n * IR + 16 * u + i] = hb;
              m = fabsf(__bfloat162float(hb));
            }
#pragma unroll
            for (int o = 8; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
            if (i == 0 && n < NR) a.hmax[u * NRM + n] = m;
          }
          arrive(C_SH);
        }
    }
  }
#endif
  // ---------------------------------------------------------------------------------------------- routed (10 warps)
  constexpr int XM_HDR = STAGE - 16;
  // x2m: ONE partial slot for 16 warps [16][2][32][4] floats (16 KB) at the attention scratch start, mbarriers full (16
  // arrivals) / free (1 arrival) after it; the x2j stage headers + GEMM1 scale buffer after those (as x2j's producer)
  float* const xred = reinterpret_cast<float*>(imgp + NRM * IMS);
  uint64_t* const xmb = reinterpret_cast<uint64_t*>(imgp + NRM * IMS + 16384);
  constexpr int XSC_OFS = 16384 + 64;
  static_assert(XSC_OFS == 2 * NCW * 2 * 32 * 4 * 4 + 64, "x2m: the scale / header area where x2j's producer puts it");
  static_assert(NCWX * 2 * 32 * 4 * 4 <= 16384, "x2n: the partial slot fits before the mbarriers");
  constexpr bool XSC = ATTX >= XSC_OFS + 5120 + NS * 8 && sizeof(S.red) >= 5120;
#if C1M_XROLE == 0
  const int wx = w;   // the base warps are expert-phase warps 0..7 (the extra warps' wx = w - 2 comes from xcons)
  if constexpr (XSC)
    if (tid == 0) {   // re-initialized every layer (the attention reuses the region); C1M_XENTER's xbar orders it for all 16
      mb_init(su32(&xmb[0]), NCWX), mb_init(su32(&xmb[1]), 1);
    }
#endif
  C1M_XENTER();
  if constexpr (!XSC) {   // x2m needs the attention scratch (FULL builds): the non-FULL instantiation is never launched with it
    __trap();
  } else {
    static_assert(NCWX == 10, "x2n: 4 k32 blocks per warp and 1280-K GEMM1 chunk");
    int rcnt = 0;   // routed items done by this CTA (all 10 warps see every item): item k's epilogue runs on warp k % 10
    // x2h: accumulators zero at every item start (reset after the 5th chunk); B fragment rows min(g, NR - 1)
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, acc2[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const int gc = g < NR ? g : NR - 1;
    const uint32_t xs_u = su32(imgp + NRM * IMS + XSC_OFS);
    for (;;) {
      const uint32_t st = cwait();
      const uint2 hv = lds64(xs_u + 5120 + 8 * (uint32_t)(it % NS));   // x2j: headers off the ring
      const int typ = (int)hv.x, val = (int)hv.y;
      (void)XM_HDR;
      if (typ == 3) {   // END
        crel1();
        break;
      }
      if (typ == 2) {   // GEMM2 routed, expert slot j: tiles of the CTA's columns rotated over the 16 warps
        const int j = val;
        for (int ti = 0; ti < nt; ++ti) {
          if ((j * nt + ti) % NCWX != wx) continue;
          float gacc[4] = {0.f, 0.f, 0.f, 0.f}, gacc2[4] = {0.f, 0.f, 0.f, 0.f};
          const int r0 = 16 * ti + g;
#pragma unroll 3
          for (int kp = 0; kp < 9; ++kp) {   // k32 blocks 2 kp, 2 kp + 1
            unsigned short bgs, bhs;
            asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bgs) : "r"(st + L_G2_S + r0 * 18 + 2 * kp));
            asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bhs) : "r"(st + L_G2_S + (r0 + 8) * 18 + 2 * kp));
            const unsigned u = bias4((unsigned)bgs | ((unsigned)bhs << 16));
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              const int kb = 2 * kp + q;
              const unsigned wg = lds32(st + r0 * 288 + 16 * kb + 4 * t4), wh = lds32(st + (r0 + 8) * 288 + 16 * kb + 4 * t4);
              const uint4 bv = lds128(st + L_G2_H + gc * L_G2_HS + 64 * kb + 16 * t4);
              unsigned f0[4], f1[4];
              dequant_pair(f0, f1, wg, wh, __byte_perm(u, 0, q ? 0x1414 : 0x0404), __byte_perm(u, 0, q ? 0x3434 : 0x2424));
              mma16816(gacc, f0[0], f0[1], f0[2], f0[3], bv.x, bv.y);
              mma16816(gacc2, f1[0], f1[1], f1[2], f1[3], bv.z, bv.w);
            }
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
            if (n < NR && S.xk[j][n] >= 0) S.Ybf[n][S.xk[j][n]][16 * ti + row] = __float2bfloat16_rn((gacc[i] + gacc2[i]) * (1.f / 256.f));
          }
        }
        crel1();
        continue;
      }
      if (typ == 4) {   // x2t: shared GEMM2 (bf16 w2) tile ti = val / 2, half 0 (the producer streams half 1 right after it):
        const int ti = val >> 1;   // the owner warp takes both halves into one accumulator, in the post-END code's k order
        if ((val & 1) == 0 && wx == (ti + X2T_SHW0) % NCWX) {
          float sacc[4] = {0.f, 0.f, 0.f, 0.f};
          sh2_16(sacc, st, 0, 10, NR, lane);
          crel1();
          const uint32_t st2 = cwait();   // half 1 of the same tile (header 4, value 2 ti + 1)
          sh2_16(sacc, st2, 10, 8, NR, lane);
          crel1();
#pragma unroll
          for (int i2 = 0; i2 < 4; ++i2) {
            const int row = g + 8 * (i2 >> 1), n = 2 * t4 + (i2 & 1);
            if (n < NR) S.shbf[n][16 * ti + row] = __float2bfloat16_rn(sacc[i2]);
          }
          continue;
        }
        crel1();   // not this warp's tile (or half 1 seen by a non-owner)
        continue;
      }
      // typ 1: routed GEMM1 unit q (slot j, 16 intermediates), chunk kc of 4 (1280 K of FP4 = 40 k32 blocks): warp wx
      // takes k32 blocks 4 wx .. 4 wx + 3 (scale bytes 40 kc + 4 wx .. + 3, one aligned word per row)
      const int q = val >> 3, kc = val & 7, j = q / SH_NU, i0 = 16 * (q % SH_NU);
      {
        unsigned sgw[2], shw[2];
        const uint32_t xb = (rcnt & 1) ? su32(S.red) : xs_u;   // x2j: item rcnt's scales, buffer rcnt & 1
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
          sgw[ti] = bias4(lds32(xb + ti * 2560 + g * 160 + 40 * kc + 4 * wx)),
          shw[ti] = bias4(lds32(xb + ti * 2560 + (g + 8) * 160 + 40 * kc + 4 * wx));
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
          const int kb = 4 * wx + kq;
          const uint4 bv = lds128(img + gc * IMS + (1280 * kc + 32 * kb) * 2 + 16 * t4);
#pragma unroll
          for (int ti = 0; ti < 2; ++ti) {
            const uint32_t ra = st + ti * 10240 + (kb >> 3) * 2048 + g * 128 + (((kb & 7) ^ g) << 4) + 4 * t4;
            const unsigned wg = lds32(ra), wh = lds32(ra + 1024);
            const unsigned mg = __byte_perm(sgw[ti], 0, 0x0404 + 0x1010 * kq), mh = __byte_perm(shw[ti], 0, 0x0404 + 0x1010 * kq);
            unsigned f0[4], f1[4];
            dequant_pair(f0, f1, wg, wh, mg, mh);
            mma16816(acc[ti], f0[0], f0[1], f0[2], f0[3], bv.x, bv.y);
            mma16816(acc2[ti], f1[0], f1[1], f1[2], f1[3], bv.z, bv.w);
          }
        }
      }
      crel1();
      if (kc != 3) continue;
      float r[2][4];
#pragma unroll
      for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int i = 0; i < 4; ++i) r[ti][i] = acc[ti][i] + acc2[ti][i], acc[ti][i] = 0.f, acc2[ti][i] = 0.f;
      // cross-warp sums over the 10 warps (warp order) in the single slot; the epilogue on warp e
      const int e = rcnt % NCWX;
      if (rcnt > 0) mb_wait(su32(&xmb[1]), (unsigned)((rcnt - 1) & 1));   // the slot read by item rcnt - 1's epilogue warp
#pragma unroll
      for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int i = 0; i < 4; ++i) xred[((wx * 2 + ti) * 32 + lane) * 4 + i] = r[ti][i];
      __syncwarp();
      if (lane == 0) mb_arrive(su32(&xmb[0]));
      const unsigned use = (unsigned)rcnt;
      ++rcnt;
      if (wx == e) {
        mb_wait(su32(&xmb[0]), use & 1u);
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float sm_ = xred[((0 * 2 + ti) * 32 + lane) * 4 + i];
            for (int ww = 1; ww < NCWX; ++ww) sm_ += xred[((ww * 2 + ti) * 32 + lane) * 4 + i];
            r[ti][i] = sm_;
          }
        __syncwarp();
        if (lane == 0) mb_arrive(su32(&xmb[1]));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          if (n < NR) {
            float h = 0.f;
            const float wt = S.xw[j][n];
            if (S.xk[j][n] >= 0) {
              const float iv = pow2i(-S.pv[n]);
              const float gt = fminf(r[0][i] * iv, 10.f);
              const float up = fminf(fmaxf(r[1][i] * iv, -10.f), 10.f);
              h = gt / (1.f + expf(-gt)) * up * wt * 256.f;
            }
            a.himg[((size_t)j * NRM + n) * 608 + p8pos(i0 + row)] = __float2half_rn(h);
          }
        }
        __syncwarp();   // (x2i) the release RED is cumulative over the lanes' himg stores the warp barrier ordered
        if (lane == 0) red_rel(a.cnt + C_H + j, 1u);
      }
    }
  }
  C1M_XEXIT();
#if C1M_XROLE == 0
  stamp(11);
  // ---------------------------------------------------------------------------------------------- GEMM2 shared
  // one tile (16 rows x 576, whole K) per stage; tile ti on warp ti % NCW; B = hsh x 2^p (f16, P8 order) built per k32 block
  // (x2t: the bf16 path ran inside the routed loop, header 4; only the fp8 path is left here)
  for (int ti = 0; ti < nt && !B16(a, sw2_16, 3); ++ti) {
    if (B16(a, sw2_16, 3)) {   // the engine's bf16 w2: 2 stages per tile (k32 blocks 0..9, 10..17), B = hsh bf16
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int hb = 0; hb < 2; ++hb) {
        const uint32_t st = cwait();
        if (w == ti % NCW) sh2_16(acc, st, 10 * hb, hb ? 8 : 10, NR, lane);
        crel();
      }
      if (w == ti % NCW)
#pragma unroll
        for (int i2 = 0; i2 < 4; ++i2) {
          const int row = g + 8 * (i2 >> 1), n = 2 * t4 + (i2 & 1);
          if (n < NR) S.shbf[n][16 * ti + row] = __float2bfloat16_rn(acc[i2]);
        }
      continue;
    }
    const uint32_t st = cwait();
    if (w == ti % NCW) {
      const char* sb_ = ring + (st - ring_u);
      int ph[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int n = 2 * t4 + q;
        float m = 0.f;
        if (n < NR)
          for (int u = 0; u < SH_NU; ++u) m = fmaxf(m, *(const float*)(sb_ + L_S2_HM + (u * NRM + n) * 4));
        ph[q] = p_of(m);
      }
      float m = 0.f;
      if (g < NR)
        for (int u = 0; u < SH_NU; ++u) m = fmaxf(m, *(const float*)(sb_ + L_S2_HM + (u * NRM + g) * 4));
      const float scg = pow2i(p_of(m));
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kb = 0; kb < 18; ++kb) {
        uint4 bv = make_uint4(0u, 0u, 0u, 0u);
        if (g < NR) bv = p8_f16(*(const uint4*)(sb_ + L_S2_HS + g * IR * 2 + (32 * kb + 8 * t4) * 2), scg);
        const uint32_t ra = st + g * IR + kb * 32 + 8 * t4;
        const uint2 wg = lds64(ra), wh = lds64(ra + 8 * IR);
        const unsigned pg_ = __byte_perm(wg.x, wg.y, 0x6420), qg = __byte_perm(wg.x, wg.y, 0x7531);
        const unsigned ph_ = __byte_perm(wh.x, wh.y, 0x6420), qh = __byte_perm(wh.x, wh.y, 0x7531);
        unsigned g02, g46, h02, h46, g13, g57, h13, h57;
        cvt_e4m3x4(pg_, g02, g46);
        cvt_e4m3x4(ph_, h02, h46);
        cvt_e4m3x4(qg, g13, g57);
        cvt_e4m3x4(qh, h13, h57);
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(tmp, g02, h02, g46, h46, bv.x, bv.y);
        mma16816(tmp, g13, h13, g57, h57, bv.z, bv.w);
        unsigned sbb;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sbb) : "r"(st + L_S2_S + kb));
        const float sc = ue8m0(sbb);
#pragma unroll
        for (int i2 = 0; i2 < 4; ++i2) acc[i2] = fmaf(tmp[i2], sc, acc[i2]);
      }
#pragma unroll
      for (int i2 = 0; i2 < 4; ++i2) {
        const int row = g + 8 * (i2 >> 1), n = 2 * t4 + (i2 & 1);
        if (n < NR) S.shbf[n][16 * ti + row] = __float2bfloat16_rn(acc[i2] * pow2i(-ph[i2 & 1]));
      }
    }
    crel();
  }
#endif
