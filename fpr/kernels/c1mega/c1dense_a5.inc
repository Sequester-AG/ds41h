// c1dense_a5.inc: A5 q_b (this GPU's 16 heads): worker CTA c < NWK owns tiles tq = 4c + t (16 rows each, rows 64c ..), RoPE on dims
//  448..511 in the epilogue.
//  C1M_DPART 0 (producer): 5 stages per worker: qb8 (B16(a, wqb8, 4): the engine's W8A16 packed wq_b by TM_WQB 5-D + TM_WQB8S, its
//    layout fixed) or the fp8 copy (TM_WQB + 96 B scales).
//  C1M_DPART 1 (consumers, inside `if (c < NWK) {`): in: img (A4's q image, f16 P8 x 2^S.pv[n]), S.pv, a.cos_sin / a.pos;
//    out: S.qt[n][64] bf16 (RoPE applied), ending with stamp(36) + cbar() (everything after: the q out, C1 core).
// Default = the code as it was in c1mega.cu, verbatim. Variants: a wrapper .cu that defines C1M_DENSE_A5 to
// another file, then includes the deployment wrapper (see c1mega_x2hq29.cu).
#if C1M_DPART == 0
      // A5: q_b tiles 4c .. 4c + 3 (rows 64c .., 64 x 1280): 5 chunks of 256 K (one 64-row box + the 2 scale rows)
      if (c < NWK) {
        if (B16(a, wqb8, 4)) {   // the engine's packed wq_b: rows 64c .. = half tile c / 2, 64-row sub-tile c & 1
          for (int kc = 0; kc < 5; ++kc) {   // [2 chunks][4 k32 blocks][4 x 16 rows][32 lanes][16 B] + [2 chunks][16 B]
            const uint32_t s = slot(16384 + 32), b = sbase(s), fb = su32(&S.full[s]);
            tma5d(b, a.tm + TM_WQB, 0, 8 * (c & 1), 0, c >> 1, 2 * kc, fb, pol);
            tma3d(b + L_QB_S, a.tm + TM_WQB8S, 0, c >> 1, 2 * kc, fb, pol);
            ++it;
          }
        } else {
          for (int kc = 0; kc < 5; ++kc) {
            const uint32_t s = slot(64 * 256 + 96), b = sbase(s), fb = su32(&S.full[s]);
            tma3d(b, a.tm + TM_WQB, 0, 64 * c, 2 * kc, fb, pol);
            bulk(b + L_QB_S, a.wqb_s + (size_t)(2 * c) * 48, 96, fb, pol);
            ++it;
          }
        }
      }
#else
      static_assert(512 == 4 * NWK && NCW == 8, "q_b: 4 tiles per worker, 8 k32 blocks per chunk");
      float rc[4] = {1.f, 1.f, 1.f, 1.f}, rsn[4] = {0.f, 0.f, 0.f, 0.f};   // RoPE factors of this lane's 4 values
      if (w < 4) {
        const int tq = 4 * c + w;
        if ((tq & 31) >= 28)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1), d = 16 * (tq & 31) + row;
            if (n < NR) {
              const float* cs = a.cos_sin + __ldg(a.pos + n) * 64;
              const int ip = (d - 448) >> 1;
              rc[i] = __ldg(cs + ip), rsn[i] = __ldg(cs + 32 + ip);
            }
          }
      }
      float acc[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;
      if (B16(a, wqb8, 4)) {   // the engine's packed wq_b: lane-ready A fragments (k 2t.., 2t + 8.. of rows g, g + 8)
        // B: lane t4 needs k (2 t4, 2 t4 + 1) of each 8-group of the k32 block = P8 image positions t4 and 4 + t4 (f16 words
        // t4 >> 1 and 2 + (t4 >> 1), half t4 & 1); lanes g >= NR feed only the unused columns n >= NR (row NR - 1 read)
        const int gq = g < NR ? g : NR - 1;
        const unsigned psel = (t4 & 1) ? 0x7632u : 0x5410u;
        const uint32_t ab = (uint32_t)(w >> 2) * 8192u + (uint32_t)(w & 3) * 2048u + (uint32_t)lane * 16u;   // chunk, k32 block
        const uint32_t sbo = L_QB_S + (w >> 2) * 16 + (w & 3) * 4 + 2 * (c & 1);   // + (t >> 1): the tile's 32-row block
        for (int kc = 0; kc < 5; ++kc) {
          const uint32_t st = cwait();
          const uint32_t ib = img + gq * IMS + (256 * kc + 32 * w) * 2 + 4 * (t4 >> 1);
          unsigned bq[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) bq[q] = __byte_perm(lds32(ib + 16 * q), lds32(ib + 16 * q + 8), psel);
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint4 v = lds128(st + ab + t * 512);
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            cvt_e4m3x4(v.x, a0, a1);
            cvt_e4m3x4(v.y, a2, a3);
            cvt_e4m3x4(v.z, b0, b1);
            cvt_e4m3x4(v.w, b2, b3);
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
            mma16816(tmp, a0, a1, a2, a3, bq[0], bq[1]);
            mma16816(tmp, b0, b1, b2, b3, bq[2], bq[3]);
            unsigned sb;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + sbo + (t >> 1)));
            const float sc = ue8m0(sb);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[t][i] = fmaf(tmp[i], sc, acc[t][i]);
          }
          crel();
        }
      } else
      for (int kc = 0; kc < 5; ++kc) {
        const uint32_t st = cwait();
        const int kb = w;
        uint4 bv = make_uint4(0u, 0u, 0u, 0u);
        if (g < NR) bv = lds128(img + g * IMS + (256 * kc + 32 * kb) * 2 + 16 * t4);
        const uint32_t off = (kb >> 2) * 8192 + g * 128 + (((((kb & 3) << 1) | (t4 >> 1)) ^ g) << 4) + ((t4 & 1) << 3);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
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
          unsigned sb;
          asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + L_QB_S + (t >> 1) * 48 + 8 * kc + kb));
          const float sc = ue8m0(sb);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[t][i] = fmaf(tmp[i], sc, acc[t][i]);
        }
        crel();
      }
      stamp(35);
      cbar();   // every warp is done with the image: partials go there
      float* red = (float*)imgp;
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[((w * 4 + t) * 32 + lane) * 4 + i] = acc[t][i];
      cbar();
      if (w < 4) {
        const int t = w, tq = 4 * c + t;
        const bool rope = (tq & 31) >= 28;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float sm_ = red[((0 * 4 + t) * 32 + lane) * 4 + i];
          for (int ww = 1; ww < NCW; ++ww) sm_ += red[((ww * 4 + t) * 32 + lane) * 4 + i];
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          const float v = __bfloat162float(__float2bfloat16_rn(sm_ * pow2i(-S.pv[n < NR ? n : 0])));
          const float pv = __shfl_xor_sync(~0u, v, 4);  // the pair partner: row g ^ 1
          float r = v;
          if (rope && n < NR) {
            const float e = (g & 1) ? pv : v, od = (g & 1) ? v : pv, cc = rc[i], sn = rsn[i];
            r = (g & 1) ? __fadd_rn(__fmul_rn(e, sn), __fmul_rn(od, cc)) : __fsub_rn(__fmul_rn(e, cc), __fmul_rn(od, sn));
          }
          if (n < NR) S.qt[n][16 * t + row] = __float2bfloat16_rn(r);
        }
      }
      stamp(36);
      cbar();
#endif
