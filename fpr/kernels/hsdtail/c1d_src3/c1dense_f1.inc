// c1dense_f1.inc: F1 wo_a: CTA c < 128 (gi = c >> 6 group, tr = (c & 63) * 16 rows).
//  C1M_DPART 0 (producer): 8 bf16 stages (TM_WOA16, two boxes each; B16 bit 1) or 4 fp8 stages (TM_WOA + scales).
//  C1M_DPART 1 (consumers, after the o image + cbar() + stamp(26), followed by arrive(C_Z)): in: img (the o image: natural bf16 or
//    f16 P8 x 2^S.pv[n]); out: a.z[n * 2048 + 1024 gi + tr + row] bf16.
// Default = the code as it was in c1mega.cu, verbatim. Variants: a wrapper .cu that defines C1M_DENSE_F1 to
// another file, then includes the deployment wrapper (see c1mega_x2hq29.cu).
#if C1M_DPART == 0
    // F1 wo_a
    if (c < 128 && B16(a, woaT16, 1)) {   // the engine's bf16 wo_a^T: 8 chunks of 512 K, each two boxes {16 columns, 256 K}
      const int gi = c >> 6, tr = (c & 63) * 16;
      for (int kc = 0; kc < 8; ++kc) {
        const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOA16, tr, 512 * kc, 2 * rank + gi, fb, pol);
        tma3d(b + 8192, a.tm + TM_WOA16, tr, 512 * kc + 256, 2 * rank + gi, fb, pol);
        ++it;
      }
    } else if (c < 128) {
      const int gi = c >> 6, tr = (c & 63) * 16;
      for (int kc = 0; kc < 4; ++kc) {
        const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOA, 0, 1024 * gi + tr, 8 * kc, fb, pol);
        bulk(b + L_D_S, a.woa_s + (size_t)((1024 * gi + tr) / 32) * 128 + 32 * kc, 32, fb, pol);
        ++it;
      }
    }
#else
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    const bool b16 = B16(a, woaT16, 1);
    for (int kc = 0; kc < (b16 ? 8 : 4); ++kc) {
      const uint32_t st = cwait();
      if (b16) dense16t_chunk(acc[0], st, 32, img, 512 * kc, NR, w, lane);
      else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
      crel();
    }
    reduce_warps<1>(acc, S.red, w, lane);
    if (w == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
        if (n < NR) a.z[(size_t)n * 2048 + 1024 * gi + tr + row] = __float2bfloat16_rn(b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]));
      }
    }
#endif
