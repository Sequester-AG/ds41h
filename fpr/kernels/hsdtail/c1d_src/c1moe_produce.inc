// c1moe_produce.inc: the expert phase's weight stream (PRODUCER warp, lane 0), textually included inside k_layer's
// producer layer loop (fpr/kernels/c1mega/c1mega.cu). Owner of the contract: the C1 persistent-kernel session
// In scope (see c1mega.cu at the include): Args a (= al[l]), l, c (CTA)
// rank, NR, nt / ncols / col0 / ftb (this CTA's hidden-column tiles), ncall / par, it, pol, the lambdas slot(bytes) /
// sbase(s), S (Smem: S.route mbarrier, S.xid, S.nexp, S.full[], S.empty[]), NS, constants (D, IR, NE, TOPK, SH_NU,
// SH_NH, NWK, L_*), helpers (bulk, bulk_nc, tma3d, tma4d, spin_ge, ld_acq, mb_wait, mb_arrive, g1first ...).
// Must stream exactly the chunks c1moe_consume.inc consumes, in the same order.
    // H shared GEMM1 halves (items g = c + 132 i < 72)
    for (int g = g1first(c); g < SH_NH; g += NWK) {
      const int u = g >> 1, hf = g & 1, i0 = 16 * u;
      if (B16(a, sw13_16, 3)) {   // the engine's bf16 w1 | w3: 10 chunks of 256 K, [2][4 k-blocks of 64][16][128 B]
        for (int kc = 0; kc < 10; ++kc) {
          const uint32_t s = slot(2 * 4 * 16 * 128), b = sbase(s), fb = su32(&S.full[s]);
          tma4d(b, a.tm + TM_SW13_16, 0, i0, (2560 * hf + 256 * kc) / 64, 0, fb, pol);
          ++it;
        }
        continue;
      }
      for (int kc = 0; kc < 5; ++kc) {
        const int k0 = 2560 * hf + 512 * kc;
        const uint32_t s = slot(32 * 512 + 32), b = sbase(s), fb = su32(&S.full[s]);
        tma4d(b, a.tm + TM_SW13, 0, i0, k0 / 128, 0, fb, pol);   // gate and up halves: [2][4 k-blocks][16 rows][128 B]
        bulk(b + L_SH1_S, a.sw13_s + (size_t)(i0 / 32) * 160 + k0 / 32, 16, fb, pol);
        bulk(b + L_SH1_S + 16, a.sw13_s + (size_t)(18 + i0 / 32) * 160 + k0 / 32, 16, fb, pol);
        ++it;
      }
    }
    // the routing of this layer (consumers' top-6)
    mb_wait(su32(&S.route), (unsigned)(l & 1));
    const int nexp = S.nexp;
    const int nit = SH_NH + SH_NU * nexp;
    for (int g = g1first(c); g < nit; g += NWK) {
      if (g < SH_NH) continue;
      const int q = g - SH_NH, j = q / SH_NU, i0 = 16 * (q % SH_NU), e = S.xid[j];
      for (int kc = 0; kc < 5; ++kc) {
        if (a.mode & 2) {   // ablation: no copy
          const int s0 = (int)(it % NS);
          mb_wait(su32(&S.empty[s0]), (unsigned)(((it / NS) & 1) ^ 1));
          mb_arrive(su32(&S.full[s0]));
          ++it;
          continue;
        }
        const uint32_t s = slot(32 * 512 + 32 * 32), b = sbase(s), fb = su32(&S.full[s]);
        tma4d(b, a.tm + TM_W13, 0, i0, 4 * kc, 2 * e, fb, pol);          // [2 halves][4 k-blocks][16 rows][128 B]
        tma3d(b + L_G1_S, a.tm + TM_S13, 32 * kc, i0, 2 * e, fb, pol);   // [2 halves][16 rows][32 B]
        ++it;
      }
    }
    // I GEMM2 routed: per expert slot one stage of this CTA's w2 rows + scales + the expert's h image. The w2 / s2 copies
    // go out as soon as a stage is free; the h copy (it needs the expert's 36 GEMM1 units) is deferred in FIFO order and
    // issued when ready (before the producer would block on a free stage, pending h copies are flushed: the consumers of
    // the full stages need them)
    {
      int pj[NSMAX], ps_[NSMAX], ph = 0, pt = 0;   // FIFO of (expert slot, stage) with the h copy outstanding
      auto issue_h = [&](bool block) -> bool {
        if (ph == pt) return false;
        const int j = pj[ph % NSMAX];
        if (!block && (int)(ld_acq(a.cnt + C_H + j) - SH_NU * ncall) < 0) return false;
        spin_ge(a.cnt + C_H + j, SH_NU * ncall);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        bulk_nc(sbase((uint32_t)ps_[ph % NSMAX]) + L_G2_H, a.himg + (size_t)j * NRM * 608, NR * 1216,
                su32(&S.full[ps_[ph % NSMAX]]));
        ++ph;
        return true;
      };
      for (int j = 0; j < nexp; ++j) {
        const int e = S.xid[j];
        if (a.mode & 2) {   // ablation: no copy
          const int s0 = (int)(it % NS);
          mb_wait(su32(&S.empty[s0]), (unsigned)(((it / NS) & 1) ^ 1));
          mb_arrive(su32(&S.full[s0]));
          ++it;
          continue;
        }
        const int s0 = (int)(it % NS);
        const unsigned epar = (unsigned)(((it / NS) & 1) ^ 1);
        while (!mb_test(su32(&S.empty[s0]), epar)) {
          if (!issue_h(true)) {
            mb_wait(su32(&S.empty[s0]), epar);
            break;
          }
        }
        mb_expect(su32(&S.full[s0]), ncols * 288 + ncols * 18 + NR * 1216);
        const uint32_t b = sbase((uint32_t)s0), fb = su32(&S.full[s0]);
        bulk(b, a.w2 + ((size_t)e * D + col0) * 288, ncols * 288, fb, pol);
        bulk(b + L_G2_S, a.s2 + ((size_t)e * D + col0) * 18, ncols * 18, fb, pol);
        pj[pt % NSMAX] = j, ps_[pt % NSMAX] = s0, ++pt;
        while (issue_h(false)) {
        }
        ++it;
      }
      while (issue_h(true)) {
      }
    }
    // I GEMM2 shared: 2 K halves
    spin_ge(a.cnt + C_SH, SH_NU * ncall);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    for (int ti = 0; ti < nt; ++ti) {   // one tile (16 rows x 576, whole K) per stage
      if (B16(a, sw2_16, 3)) {   // the engine's bf16 w2: 2 stages per tile (k-blocks of 64 0..4, 5..8 + one
        for (int hb = 0; hb < 2; ++hb) {   // out-of-bounds zero block), each with the hsh rows
          const uint32_t s = slot(L_S2B_H + NR * IR * 2), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_SW2_16, 0, col0 + 16 * ti, 5 * hb, fb, pol);
          bulk_nc(b + L_S2B_H, a.hsh, NR * IR * 2, fb);
          ++it;
        }
        continue;
      }
      const uint32_t s = slot(16 * IR + 32 + NR * IR * 2 + SH_NU * NRM * 4), b = sbase(s), fb = su32(&S.full[s]);
      bulk(b, a.sw2 + (size_t)(col0 + 16 * ti) * IR, 16 * IR, fb, pol);
      bulk(b + L_S2_S, a.sw2_s + (size_t)((col0 + 16 * ti) / 32) * 32, 32, fb, pol);
      bulk_nc(b + L_S2_HS, a.hsh, NR * IR * 2, fb);
      bulk_nc(b + L_S2_HM, a.hmax, SH_NU * NRM * 4, fb);
      ++it;
    }
