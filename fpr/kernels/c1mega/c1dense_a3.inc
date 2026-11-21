// c1dense_a3.inc: A3 q_a | wkv (replicated on every GPU): CTA c < 112 owns output columns 16c .. 16c + 15 of [1280 q_a | 512 kv].
//  C1M_DPART 0 (producer, warp NCW lane 0, inside the layer loop): a (this layer's Args), c, slot(bytes) / sbase(s) / S.full[s],
//    tma3d / bulk, pol, ++it per stage. Today: 10 bf16 stages (TM_WQKV16, B16 bit 0) or 5 fp8 stages (TM_WQKV + 32 B scales).
//  C1M_DPART 1 (consumers, inside `if (c < 112) {`, followed by arrive(C_QA)): in: img (the A2 image: f16 P8 x 2^S.pv[n], or natural
//    bf16 when B16(a, wqkv16, 0)), NR, w, lane, g, t4, cwait() / crel(), S.red; out: a.qakv[n * 1792 + col] bf16 (descaled).
// Default = the code as it was in c1mega.cu, verbatim. Variants: a wrapper .cu that defines C1M_DENSE_A3 to
// another file, then includes the deployment wrapper (see c1mega_x2hq29.cu).
#if C1M_DPART == 0
      // A3: q_a | wkv tile c (16 rows x 5120): 5 chunks of 1024 K (fp8) or 10 chunks of 512 K (the engine's bf16)
      if (c < 112 && B16(a, wqkv16, 0))
        for (int kc = 0; kc < 10; ++kc) {
          const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_WQKV16, 0, 16 * c, 8 * kc, fb, pol);
          ++it;
        }
      else if (c < 112)
        for (int kc = 0; kc < 5; ++kc) {
          const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_WQKV, 0, 16 * c, 8 * kc, fb, pol);
          bulk(b + L_D_S, a.wqkv_s + (size_t)((16 * c) / 32) * 160 + 32 * kc, 32, fb, pol);
          ++it;
        }
#else
      float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      const bool b16 = B16(a, wqkv16, 0);
      for (int kc = 0; kc < (b16 ? 10 : 5); ++kc) {
        const uint32_t st = cwait();
        if (b16) dense16_chunk<1>(acc, st, 0, 32, img, 512 * kc, NR, w, lane);
        else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
        crel();
      }
      reduce_warps<1>(acc, S.red, w, lane);
      if (w == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          if (n < NR) a.qakv[(size_t)n * 1792 + 16 * c + row] = __float2bfloat16_rn(b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]));
        }
#endif
