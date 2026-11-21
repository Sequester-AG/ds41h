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
  // ---------------------------------------------------------------------------------------------- GEMM1 items
  {
    const int nit = SH_NH + SH_NU * nexp;
    int rcnt = 0;   // routed items done by this CTA: item k's epilogue runs on warp k % NCW
    for (int gq = g1first(c); gq < nit; gq += NWK) {
      float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      if (gq < SH_NH) {  // shared half: 5 chunks of 512 K (dense fp8)
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
        if (tid == 0) {
          __threadfence();
          S.ticket = (int)(atomicAdd(a.cnt + C_SHT + u, 1u) & 1u);
        }
        cbar();
        if (S.ticket) {  // second half done: sum the halves in order, the shared expert's gu / SwiGLU (bf16 rounding points)
          __threadfence();
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
      } else {  // routed unit (slot j, 16 intermediates): 5 chunks of 1024 K (FP4)
        const int q = gq - SH_NH, j = q / SH_NU, i0 = 16 * (q % SH_NU);
        float acc2[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        static_assert(NCW == 8, "GEMM1: 4 k32 blocks per warp and chunk");
        for (int kc = 0; kc < 5; ++kc) {
          const uint32_t st = cwait();
          if (!(a.mode & 1)) {
            unsigned sgw[2], shw[2];   // biased scale bytes of rows g / g + 8, blocks 4w .. 4w + 3 (moe_v4's multipliers)
#pragma unroll
            for (int ti = 0; ti < 2; ++ti)
              sgw[ti] = bias4(lds32(st + L_G1_S + ti * 512 + g * 32 + 4 * w)),
              shw[ti] = bias4(lds32(st + L_G1_S + ti * 512 + (g + 8) * 32 + 4 * w));
#pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
              const int kb = 4 * w + kq;
              uint4 bv = make_uint4(0u, 0u, 0u, 0u);
              if (g < NR) bv = lds128(img + g * IMS + (1024 * kc + 32 * kb) * 2 + 16 * t4);
#pragma unroll
              for (int ti = 0; ti < 2; ++ti) {
                const uint32_t ra = st + ti * 8192 + (kb >> 3) * 2048 + g * 128 + (((kb & 7) ^ g) << 4) + 4 * t4;
                const unsigned wg = lds32(ra), wh = lds32(ra + 1024);
                const unsigned mg = __byte_perm(sgw[ti], 0, 0x0404 + 0x1010 * kq), mh = __byte_perm(shw[ti], 0, 0x0404 + 0x1010 * kq);
                unsigned f0[4], f1[4];
                dequant_pair(f0, f1, wg, wh, mg, mh);
                mma16816(acc[ti], f0[0], f0[1], f0[2], f0[3], bv.x, bv.y);
                mma16816(acc2[ti], f1[0], f1[1], f1[2], f1[3], bv.z, bv.w);
              }
            }
          }
          crel();
        }
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[ti][i] += acc2[ti][i];
        // cross-warp sums (reduce_warps' order) and the epilogue on warp e; the other warps go on to the next item as soon
        // as warp e has read the partials
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
#pragma unroll
          for (int i = 0; i < 4; ++i) S.red[((w * 2 + ti) * 32 + lane) * 4 + i] = acc[ti][i];
        cbar();
        const int e = rcnt % NCW;
        ++rcnt;
        if (w != e) {
          nb2_sync();
        } else {
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              float sm_ = S.red[((0 * 2 + ti) * 32 + lane) * 4 + i];
              for (int ww = 1; ww < NCW; ++ww) sm_ += S.red[((ww * 2 + ti) * 32 + lane) * 4 + i];
              acc[ti][i] = sm_;
            }
          nb2_arrive();
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
            if (n < NR) {
              float h = 0.f;
              const float wt = S.xw[j][n];
              if (S.xk[j][n] >= 0) {
                const float iv = pow2i(-S.pv[n]);
                const float gt = fminf(acc[0][i] * iv, 10.f);
                const float up = fminf(fmaxf(acc[1][i] * iv, -10.f), 10.f);
                h = gt / (1.f + expf(-gt)) * up * wt * 256.f;
              }
              a.himg[((size_t)j * NRM + n) * 608 + p8pos(i0 + row)] = __float2half_rn(h);
            }
          }
          __threadfence();
          __syncwarp();
          if (lane == 0) red_rel(a.cnt + C_H + j, 1u);
        }
      }
    }
  }
  stamp(11);
  // ---------------------------------------------------------------------------------------------- GEMM2 routed
  for (int j = 0; j < nexp; ++j) {
    const uint32_t st = cwait();
    for (int ti = 0; ti < nt; ++ti) {
      if ((j * nt + ti) % NCW != w || (a.mode & 1)) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
      const int r0 = 16 * ti + g;
#pragma unroll 3
      for (int kp = 0; kp < 9; ++kp) {   // k32 blocks 2 kp, 2 kp + 1
        unsigned short bgs, bhs;
        asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bgs) : "r"(st + L_G2_S + r0 * 18 + 2 * kp));
        asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bhs) : "r"(st + L_G2_S + (r0 + 8) * 18 + 2 * kp));
        const unsigned u = bias4((unsigned)bgs | ((unsigned)bhs << 16));   // bytes: g blk0, g blk1, g+8 blk0, g+8 blk1
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int kb = 2 * kp + q;
          const unsigned wg = lds32(st + r0 * 288 + 16 * kb + 4 * t4), wh = lds32(st + (r0 + 8) * 288 + 16 * kb + 4 * t4);
          uint4 bv = make_uint4(0u, 0u, 0u, 0u);
          if (g < NR) bv = lds128(st + L_G2_H + g * L_G2_HS + 64 * kb + 16 * t4);
          unsigned f0[4], f1[4];
          dequant_pair(f0, f1, wg, wh, __byte_perm(u, 0, q ? 0x1414 : 0x0404), __byte_perm(u, 0, q ? 0x3434 : 0x2424));
          mma16816(acc, f0[0], f0[1], f0[2], f0[3], bv.x, bv.y);
          mma16816(acc2, f1[0], f1[1], f1[2], f1[3], bv.z, bv.w);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
        if (n < NR && S.xk[j][n] >= 0) S.Ybf[n][S.xk[j][n]][16 * ti + row] = __float2bfloat16_rn((acc[i] + acc2[i]) * (1.f / 256.f));
      }
    }
    crel();
  }
  // ---------------------------------------------------------------------------------------------- GEMM2 shared
  // one tile (16 rows x 576, whole K) per stage; tile ti on warp ti % NCW; B = hsh x 2^p (f16, P8 order) built per k32 block
  for (int ti = 0; ti < nt; ++ti) {
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
