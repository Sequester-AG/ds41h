// Other paths: c1dense_f2_u.inc verbatim
// c1dense_f2_u.inc: c1dense_f2.inc with the bf16 chunk loop 2 stages per wait
// loads hoisted (dense16_c1u). Same values
// c1dense_f2.inc: F2 wo_b: every CTA, its nt tiles of 16 output columns (ftb + ti).
//  C1M_DPART 0 (producer): per tile 4 bf16 stages (TM_WOB16, B16 bit 2) or 2 fp8 stages (TM_WOB + scales).
//  C1M_DPART 1 (consumers, after the z image + cbar()): in: img (the z image); out: S.yp[n][16 ti + row] fp32 (descaled), one cbar()
//    per tile as today; the AR #1 that follows reads S.yp (C1 core).
// Default = the code as it was in c1mega.cu, verbatim. Variants: a wrapper .cu that defines C1M_DENSE_F2 to
// another file, then includes the deployment wrapper (see c1mega_x2hq29.cu).
#if C1M_DPART == 0
    // F2 wo_b
    for (int ti = 0; ti < nt && B16(a, wob8, 6); ++ti) {   // (srv8x) the engine's packed wo_b: rows r0 = 16 (ftb + ti) of 5120
      const int r0 = 16 * (ftb + ti);
      for (int kc = 0; kc < 4; ++kc) {
        const uint32_t s = slot(L_D8S + 64), b = sbase(s), fb = su32(&S.full[s]);
        tma5d(b, a.tm + TM_WOB8, 0, 2 * ((r0 & 127) >> 4), 0, r0 >> 7, 4 * kc, fb, pol);
        tma3d(b + L_D8S, a.tm + TM_WOB8S, 0, r0 >> 7, 4 * kc, fb, pol);
        ++it;
      }
    }
    for (int ti = 0; ti < nt && B16(a, wob16, 2); ++ti)   // the engine's bf16 wo_b: 4 chunks of 512 K of this rank's K quarter
      for (int kc = 0; kc < 4; ++kc) {
        const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOB16, 0, 16 * (ftb + ti), 32 * rank + 8 * kc, fb, pol);
        ++it;
      }
    for (int ti = 0; ti < nt && !B16(a, wob16, 2) && !B16(a, wob8, 6); ++ti)
      for (int kc = 0; kc < 2; ++kc) {
        const int r0 = 16 * (ftb + ti);
        const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOB, 0, r0, a.wob_k0 + 8 * kc, fb, pol);
        bulk(b + L_D_S, a.wob_s + (size_t)(r0 / 32) * a.wob_sst + a.wob_sc0 + 32 * kc, 32, fb, pol);
        ++it;
      }
#else
    const bool w8 = B16(a, wob8, 6), b16 = B16(a, wob16, 2) || w8;   // (srv8x: the packed path's image is natural bf16 too)
    for (int ti = 0; ti < nt; ++ti) {
      float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      if (w8) {   // (srv8x) two stages per wait / release, as C1M_DU; the same k order (dense8_c1u)
        const int rb = ((16 * (ftb + ti)) & 127) >> 5;
        for (int kc = 0; kc < 4; kc += 2) {
          const uint32_t st0 = cwait(), st1 = cwait1();
          dense8_c1u(acc[0], st0, img, 512 * kc, NR, rb, w, lane);
          dense8_c1u(acc[0], st1, img, 512 * (kc + 1), NR, rb, w, lane);
          crel();
          crel();
        }
      } else
#ifdef C1M_DU
      if (b16)   // (C1M_DU: two stages per wait / release, loads hoisted; the same MMA order)
        for (int kc = 0; kc < 4; kc += 2) {
          const uint32_t st0 = cwait(), st1 = cwait1();
          dense16_c1u(acc[0], st0, img, 512 * kc, NR, w, lane);
          dense16_c1u(acc[0], st1, img, 512 * (kc + 1), NR, w, lane);
          crel();
          crel();
        }
      else
#endif
      for (int kc = 0; kc < (b16 ? 4 : 2); ++kc) {
        const uint32_t st = cwait();
        if (b16) dense16_chunk<1>(acc, st, 0, 32, img, 512 * kc, NR, w, lane);
        else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
        crel();
      }
      reduce_warps<1>(acc, S.red, w, lane);
      if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          if (n < NR) S.yp[n][16 * ti + row] = b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]);
        }
      }
      cbar();
    }
#endif
