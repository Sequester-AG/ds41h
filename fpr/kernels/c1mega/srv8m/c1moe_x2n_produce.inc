// x2n: x2y's producer (GEMM2 backlog X2Y_B, default 2 here) with the routed GEMM1 items in 4 chunks of 1280 K
// (W13 box [2][5][16][128 B] = 20,480 B, coordinate 5 kc; chunk 0 also carries the item's 2 scale copies, 5,120 B, into
// x2j's double buffer). Everything else as x2y / x2j.
#ifndef X2Y_B
#define X2Y_B 2
#endif
// --- x2y:
// x2y
// x2j's producer with a GEMM2 BACKLOG. the routed phase ends with ~10 us of GEMM2 after the last GEMM1 chunk
// mostly waiting for the last experts' h (every CTA needs every expert's h: column-split GEMM2), while x2j stages each
// completed expert's GEMM2 as soon as its h is seen. x2y keeps up to X2Y_B completed experts UNSTAGED while this CTA
// still grabs GEMM1 items (it stages the oldest only when more than X2Y_B are waiting), so after its last item the CTA
// streams that backlog while the last experts' items finish elsewhere. The early relaxed h loads cover X2Y_B + 4 experts.
// GEMM2 stages stay in ascending expert order per CTA and the consumers are unchanged: bitwise-identical outputs.
// --- x2j:
// x2j: x2f's producer (GEMM1 scales in 2 bulk copies per
// item into a double SMEM buffer, stage headers off the ring) with NON-BLOCKING h polls: the next 4 experts' C_H counters
// are loaded (ld.relaxed.gpu) before an item's chunks and checked after them; complete ones (in order) get their GEMM2
// stage after one fence.acq_rel.gpu (the acquire pattern for the relaxed loads), and only if all 4 were complete does the
// blocking poll continue. blocking polls cost the producer ~300 ns each, ~29 per CTA (~9 us of a 42 us skeleton).
// The ticket stays where it was (x2e's early ticket cost balance). Same math.
// --- x2f:
// x2f: x2d's GEMM1 changes
// (scales in 2 bulk copies per item into a double SMEM buffer, stage headers off the ring, no per-stage proxy fence) on
// x2b's GEMM2 routed stages (w2 + s2 + h image in the stage, as x2b): the producer cuts WITHOUT x2c's consumer-side
// global loads. Same math.
// --- x2d:
// x2d: x2c with
// fewer producer issues per routed GEMM1 item. In FULL builds (attention scratch present; XSC):
//  - the item's GEMM1 scales (gate and up rows: 2 contiguous runs of 16 x 160 B in s13) come in 2 bulk copies with its
//    chunk 0 instead of one [2][16][32 B] TMA box per chunk (7 copies per item instead of 10), into one of two buffers:
//    even items (per CTA and layer) the attention scratch after x2a's partial slots, odd items S.red (8 KB; FULL builds
//    use it in the routed phase only for the shared GEMM1 halves, before the first routed stage). Consumers read them per
//    chunk as before. Before an item's scale copies the producer waits for the release of the previous item's chunk 0
//    (all warps: the item before it is done with the same buffer; for item 1, the shared halves' S.red reductions are
//    done), unless that stage is NS or more back (the slot wait covers it); item 1 then fences S.red generic -> async.
//  - the stage headers {type, value} live in the attention scratch too ([NS][2] ints, never written by the async proxy),
//    so the per-stage fence.proxy.async after the header store goes (one fence per layer after the routing wait orders
//    the consumers' generic writes of the scratch before the async scale copies).
// X2D_EARLY (x2e): the next item's ticket (atomicAdd) and the next expert's h counter (ld.relaxed, then fence.acq_rel
// on success) are issued before the current item's chunks, so their L2 round trips overlap the stream.
// Same per-element math and order as x2b / x2c: bitwise-identical outputs.
// --- x2c:
// x2c: x2b with GEMM2-routed stages that carry ONLY the w2 rows (one copy per expert
// instead of three): the consumers read the expert's s2 scale words (__ldg, read-only weights) and its h image (__ldcg,
// L2 only: written this layer by other CTAs; ordered by the producer's ld.acquire of C_H before the stage's mbarrier
// arrive, which the consumers acquire) straight from global memory. ~48 fewer producer copy issues per CTA per layer
// Same per-element math and order as x2b: bitwise-identical outputs.
// c1moe_x2b_produce.inc. The expert phase's weight stream (PRODUCER warp, lane 0). Changes vs today
//  - routed GEMM1 items are GRABBED from a per-GPU counter (a.cnt + C_X2G) instead of dealt round-robin: CTAs that are
//    ahead take more items (balance; the 3-tile CTAs' extra GEMM2 work is absorbed). Ids q = 0 .. 36 nexp - 1 in the same
//    expert-major order as today, so experts' h complete in order. The 128 worker CTAs make one failing grab each; the
//    last one (value 36 nexp + 127) resets the counter to 0 for the next layer.
//  - GEMM2 routed stages are INTERLEAVED: between two GEMM1 items, every expert whose h is complete (C_H + j at 36 per
//    call) gets its stage (w2 / s2 rows of this CTA's columns + the h image), in expert order; after the last item the
//    rest follow (blocking), then an END stage (no data).
//  - every stage of this phase carries a header {type, value} at HDR (16 B at the end of the stage): 1 = GEMM1 routed
//    chunk (value q * 8 + kc), 2 = GEMM2 routed (value j), 3 = END. The shared GEMM1 halves (before the routing) and the
//    GEMM2 shared stages (after END) are streamed as today.
// The ablation bits of a.mode are not supported here (mode 0 only).
    constexpr int HDR = STAGE - 16, C_X2G = 44;
    static_assert(L_G2_H + NRM * 1216 <= HDR && L_G1_S + 32 * 32 <= HDR, "x2b stage header");
    constexpr int XSC_OFS = 2 * NCW * 2 * 32 * 4 * 4 + 64;           // x2d: after x2a's partial slots + mbarriers
    constexpr bool XSC = ATTX >= XSC_OFS + 5120 + NS * 8 && sizeof(S.red) >= 5120;   // scales [2][16][160] + headers [NS][2]
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
    {
      const int nexp = S.nexp;
      const int nrt = SH_NU * nexp;   // routed GEMM1 items of this GPU
      char* const xs = imgp + NRM * IMS + XSC_OFS;                            // x2d: the current item's scales
      volatile int* const xh = reinterpret_cast<volatile int*>(xs + 5120);   // x2d: stage headers [NS][2]
      if constexpr (XSC) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // scratch: generic writes -> async
      // a stage with header {type, value}: waits for the stage to be free, writes the header, expect_tx(bytes) (0: arrive)
      auto hslot = [&](int type, int value, unsigned bytes) -> uint32_t {
        const int s = (int)(it % NS);
        mb_wait(su32(&S.empty[s]), (unsigned)(((it / NS) & 1) ^ 1));
        if constexpr (XSC) {
          xh[2 * s] = type, xh[2 * s + 1] = value;   // (generic only: no proxy fence)
        } else {
          volatile int* const hp = reinterpret_cast<volatile int*>(ring + (size_t)s * STAGE + HDR);
          hp[0] = type, hp[1] = value;
          // generic-proxy store, then this slot's later async-proxy (TMA / bulk) writes (a full 20,480-B router stage of
          // the next layer covers these bytes): order them across proxies (C1 session's review)
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (bytes)
          mb_expect(su32(&S.full[s]), bytes);
        else
          mb_arrive(su32(&S.full[s]));
        return (uint32_t)s;
      };
      int g2n = 0;   // next expert slot for GEMM2
      auto g2_stage = [&](int j) {   // h of slot j is complete (acquire seen)
        const int e = S.xid[j];
        asm volatile("fence.proxy.async.global;" ::: "memory");
        const uint32_t s = hslot(2, j, ncols * 288 + ncols * 18 + NR * 1216), b = sbase(s), fb = su32(&S.full[s]);
        bulk(b, a.w2 + ((size_t)e * D + col0) * 288, ncols * 288, fb, pol);
        bulk(b + L_G2_S, a.s2 + ((size_t)e * D + col0) * 18, ncols * 18, fb, pol);
        bulk_nc(b + L_G2_H, a.himg + (size_t)j * NRM * 608, NR * 1216, fb);
        ++it;
      };
      auto g2_ready = [&]() {   // non-blocking: every expert (in order) whose h is complete
        while (g2n < nexp && (int)(ld_acq(a.cnt + C_H + g2n) - SH_NU * ncall) >= 0) g2_stage(g2n++);
      };
      if (c < NWK) {   // worker CTAs grab routed GEMM1 items
        long sc_it = -1;   // x2d: stage index of the previous item's chunk 0
        int ni = 0;        // x2d: items of this CTA in this layer (scale buffer ni & 1)
#if X2D_EARLY
        int qn = (int)atomicAdd(a.cnt + C_X2G, 1u);
#endif
        for (;;) {
#if X2D_EARLY
          const int q = qn;
#else
          const int q = (int)atomicAdd(a.cnt + C_X2G, 1u);
#endif
          if (q >= nrt) {
            if (q == nrt + NWK - 1) atomicExch(a.cnt + C_X2G, 0u);   // the last failing grab: reset for the next layer
            break;
          }
#if X2D_EARLY
          qn = (int)atomicAdd(a.cnt + C_X2G, 1u);   // the next ticket, in flight while this item streams
          unsigned pv = 0u;
          if (g2n < nexp) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(pv) : "l"(a.cnt + C_H + g2n) : "memory");
#endif
          // x2j: the next experts' h counters as relaxed loads, in flight while this item's chunks are issued
          // (x2y: X2Y_B + 4 of them, from the first unstaged expert)
          constexpr int XP = X2Y_B + 4;
          unsigned xpv[XP];
          const int xpj = g2n;
#pragma unroll
          for (int k = 0; k < XP; ++k) {
            xpv[k] = 0u;
            if (xpj + k < nexp) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(xpv[k]) : "l"(a.cnt + C_H + xpj + k) : "memory");
          }
          const int j = q / SH_NU, i0 = 16 * (q % SH_NU), e = S.xid[j];
          static_assert(C1M_W13KB == 5, "x2n: the W13 box must hold 5 k-blocks of 128 B (1280 K): C1M_W13KB 5");
          for (int kc = 0; kc < 4; ++kc) {   // x2n: 4 chunks of 1280 K (20,480 B each)
            if constexpr (XSC) {
              if (kc == 0) {
                if (sc_it >= 0 && it - sc_it < NS)   // the previous item's chunk 0 released by every warp
                  mb_wait(su32(&S.empty[(int)(sc_it % NS)]), (unsigned)((sc_it / NS) & 1));
                sc_it = it;
                const uint32_t s = hslot(1, q * 8, 20480 + 5120), b = sbase(s), fb = su32(&S.full[s]);
                const uint32_t xb = (ni & 1) ? su32(S.red) : su32(xs);
                if (ni == 1) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // S.red: generic -> async
                tma4d(b, a.tm + TM_W13, 0, i0, 0, 2 * e, fb, pol);
                bulk(xb, a.s13 + ((size_t)e * 1152 + i0) * 160, 2560, fb, pol);                // gate rows i0 .. +16
                bulk(xb + 2560, a.s13 + ((size_t)e * 1152 + 576 + i0) * 160, 2560, fb, pol);   // up rows
              } else {
                const uint32_t s = hslot(1, q * 8 + kc, 20480), b = sbase(s), fb = su32(&S.full[s]);
                tma4d(b, a.tm + TM_W13, 0, i0, 5 * kc, 2 * e, fb, pol);
              }
            } else {
              __trap();   // x2n: FULL builds only (the scales live off the ring: a 1280-K chunk fills the stage)
            }
            ++it;
          }
          ++ni;
#if X2D_EARLY
          if (g2n < nexp && (int)(pv - SH_NU * ncall) >= 0) {   // the early poll saw expert g2n complete: acquire, stage
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            g2_stage(g2n++);
          }
#endif
          // x2j: stage the experts the early loads saw complete (in order), one acquire fence for all of them; only if all
          // XP were complete, poll on (blocking) for more
          int xk = 0;
#pragma unroll
          for (int k = 0; k < XP; ++k)
            if (xk == k && xpj + k < nexp && (int)(xpv[k] - SH_NU * ncall) >= 0) ++xk;
          if (xk > X2Y_B) {   // x2y: stage only the experts beyond the backlog of X2Y_B complete ones (in order)
            asm volatile("fence.acq_rel.gpu;" ::: "memory");   // acquire for the relaxed loads that saw the counts
            for (int k = 0; k < xk - X2Y_B; ++k) g2_stage(g2n++);
          }
        }
      }
      for (; g2n < nexp; ++g2n) {   // the rest of GEMM2 routed, in expert order (blocking)
        spin_ge(a.cnt + C_H + g2n, SH_NU * ncall);
        g2_stage(g2n);
      }
      hslot(3, 0, 0u);   // END
      ++it;
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
