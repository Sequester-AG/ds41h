// c1hooks.inc: hook points of k_layer (fpr/kernels/c1mega/c1mega.cu) for the index / compressor and Engram layer phases.
// Textually included at 5 places with C1M_HOOK = 1..5; the default sections are empty. The phases' owner replaces this file (or points -DC1M_HOOKS at another one)
//
// Rules: the producer and the consumers must stream / consume the same chunks in the same order (hooks 1 + 2..4); grid
// counters: your own array + your own per-CTA tick for owner-only phases (self-consistent per GPU), epoch-aligned across
// GPUs for all-GPU phases; never scattered remote loads; Args a.ix / a.eg point to your per-layer structs (null on other
// layers): branch on them, so regular layers cost nothing.
#if C1M_HOOK == 1
// PRODUCER (warp 8 lane 0), FULL front half, after the q_a | wkv chunks, before the q_b chunks. In scope: const Args& a
// (= al[l]), l, c, rank, ncall, it, pol, slot(bytes), sbase(s), S, NS. E.g. if (a.ix) { idx_produce(...); }
#elif C1M_HOOK == 2
// CONSUMERS (warps 0-7), before the A1 attention boundary (Engram inject). In scope: const Args& a (= S.args), l, c, tid,
// w, lane, g, t4, rank, NR, ncols, col0, nt, ncall, par, img / imgp, S, cwait / crel / arrive / waitc / stamp / cbar.
#elif C1M_HOOK == 3
// CONSUMERS, after A4 (q norm -> the q image; kv norm / RoPE / window insert on the owner), before A5 q_b. The natural
// bf16 `a` rows: a.ain (global); qr: a.qrout; the q image (f16 P8) in imgp. Index layers: compressor / indexer stage 1.
#elif C1M_HOOK == 4
// CONSUMERS, after A5 q_b (C_QB arrived), before the attention waits for C_QB. Index layers: indexer stage 2 / top-k;
// the attention reads a.ti / a.tl afterwards (mode 3: the push warp re-pushes the top-k records when a.push_topk).
#elif C1M_HOOK == 5
// PUSH WARP (warp 9, mode 3 owner), before layer l's records are pushed (const Args& A = al[l], l, c, lane, ncall, rank).
// Index layers: wait here until this layer's top-k (incl. a new compressed record) is final.
#endif
