#!/usr/bin/env python3
"""Offline lossless repack of routed-expert FP4 weights into per-thread wgmma A fragments (prototype).

wgmma RS A fragment, m64nNk16 f16, per warp w of the warpgroup (rows 16w..16w+15), lane = 4g + t:
  a0 = (row g,   k 2t, 2t+1)   a1 = (row g+8, k 2t, 2t+1)   a2 = (row g, k 2t+8, 2t+9)   a3 = (row g+8, k 2t+8, 2t+9)
One 32-bit word holds a thread's 8 nibbles of one k16 step, ordered for ds41h's exact FP4 -> e4m3 -> f16 path
(sm90.cuh fp4_lo_e4m3 / fp4_hi_e4m3 + e4m3x2 -> f16x2):
  lo nibbles of bytes 0..3 -> e4m3 bytes -> f16x2 (n0, n2), (n4, n6); hi nibbles -> (n1, n3), (n5, n7)
  so  n0 = a0.x  n2 = a0.y  n1 = a1.x  n3 = a1.y  n4 = a2.x  n6 = a2.y  n5 = a3.x  n7 = a3.y
GEMM1 tiles interleave gate and up rows so SwiGLU happens in the thread that holds both accumulators:
  tile q (64 intermediates), warpgroup c, warp w, g:  i = 64q + 32c + 8w + g;  row g = w1[i], row g+8 = w3[i].
GEMM2 tiles are 128 consecutive output rows: row (c, w, g) = 128 rt + 64c + 16w + g (+8).

Layouts (uint32 words, uint8 scales), one TP rank, per expert e:
  w13: [e][q 9][kb 80][c 2][thread 128][j 4]      scales13: [e][q][b 160][R 128]   (R = 64c + 16w + g (+8))
  w2 : [e][rt 40][kb 9][c 2][thread 128][j 4]     scales2 : [e][rt][b 18][R 128]
`python3 repack.py` runs the CPU self-test on random and (if the checkpoint is present) real weights.
"""
from __future__ import annotations

import sys

import torch

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
# nibble j of the word -> (fragment register, row offset 0/8, k offset)
NIB = {0: (0, 0, 0), 2: (0, 0, 1), 1: (1, 8, 0), 3: (1, 8, 1), 4: (2, 0, 8), 6: (2, 0, 9), 5: (3, 8, 8), 7: (3, 8, 9)}


def nibbles(w: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 (low nibble = even k) -> [..., K] int64 codes."""
    w = w.to(torch.int64)
    return torch.stack([w & 15, w >> 4], -1).flatten(-2)


def _frag_words(W: torch.Tensor, rows_g: torch.Tensor, rows_g8: torch.Tensor) -> torch.Tensor:
    """W [R, K] codes; rows_g/rows_g8 [128] row index per thread (for g and g+8) -> words [K/64, 128, 4] uint32.
    Thread t = 32w + 4g + tq; word j (k16 step) of k64 block kb uses k0 = 64kb + 16j."""
    K = W.shape[1]
    tq = torch.arange(128) % 4
    out = torch.zeros(K // 64, 128, 4, dtype=torch.int64)
    for j in range(4):
        k0 = torch.arange(K // 64).view(-1, 1) * 64 + 16 * j  # [kb, 1]
        word = torch.zeros(K // 64, 128, dtype=torch.int64)
        for n, (_, roff, koff) in NIB.items():
            rows = rows_g if roff == 0 else rows_g8
            kk = k0 + 2 * tq.view(1, -1) + (koff % 2) + (8 if koff >= 8 else 0)  # [kb, 128]
            word |= W[rows.view(1, -1).expand_as(kk), kk] << (4 * n)
        out[:, :, j] = word
    return out


def thread_rows():
    t = torch.arange(128)
    w, g = t // 32, (t % 32) // 4
    return w, g


def repack_w13(w1: torch.Tensor, w3: torch.Tensor, s1: torch.Tensor, s3: torch.Tensor):
    """One expert, one rank: w1/w3 [576, 2560] uint8, s1/s3 [576, 160] uint8 -> (words [9, 80, 2, 128, 4] int64,
    scales [9, 160, 128] uint8)."""
    I = w1.shape[0]
    W = torch.cat([nibbles(w1), nibbles(w3)], 0)  # rows 0..I-1 gate, I..2I-1 up
    S = torch.cat([s1, s3], 0).to(torch.int64)
    w, g = thread_rows()
    words = torch.zeros(I // 64, W.shape[1] // 64, 2, 128, 4, dtype=torch.int64)
    scales = torch.zeros(I // 64, S.shape[1], 128, dtype=torch.int64)
    for q in range(I // 64):
        for c in range(2):
            i = 64 * q + 32 * c + 8 * w + g  # [128] intermediate per thread
            words[q, :, c] = _frag_words(W, i, I + i)
            R = 64 * c + 16 * w + g
            # scales are per tile row R: (R) gate row, (R + 8) up row; threads with the same (w, g) share them
            scales[q, :, R] = S[i].t()
            scales[q, :, R + 8] = S[I + i].t()
    return words, scales.to(torch.uint8)


def repack_w2(w2: torch.Tensor, s2: torch.Tensor):
    """One expert, one rank: w2 [5120, 288] uint8 (K = 576), s2 [5120, 18] -> (words [40, 9, 2, 128, 4],
    scales [40, 18, 128])."""
    W = nibbles(w2)
    S = s2.to(torch.int64)
    D = W.shape[0]
    w, g = thread_rows()
    words = torch.zeros(D // 128, W.shape[1] // 64, 2, 128, 4, dtype=torch.int64)
    scales = torch.zeros(D // 128, S.shape[1], 128, dtype=torch.int64)
    for rt in range(D // 128):
        for c in range(2):
            r = 128 * rt + 64 * c + 16 * w + g
            words[rt, :, c] = _frag_words(W, r, r + 8)
            R = 64 * c + 16 * w + g
            scales[rt, :, R] = S[r].t()
            scales[rt, :, R + 8] = S[r + 8].t()
    return words, scales.to(torch.uint8)


def to_u32(words: torch.Tensor) -> torch.Tensor:
    """int64 words (< 2^32) -> int32 tensor with the same bits (for upload)."""
    return (words & 0xFFFFFFFF).to(torch.int64).where(words < 2**31, words - 2**32).to(torch.int32)


# --------------------------------------------------------------------------------------------- CPU emulation
def emulate_fragment(word: int, scale_g: int, scale_g8: int):
    """Exactly what the kernel computes for one thread's word: returns {(reg, half): value} with the weight value
    e2m1 * 2^(s - 127) (f16 path: e4m3 trick gives x 2^-6, the HMUL2 multiplier is 2^(s - 121))."""
    out = {}
    for n, (reg, roff, koff) in NIB.items():
        code = (word >> (4 * n)) & 15
        s = scale_g if roff == 0 else scale_g8
        out[(reg, koff % 2)] = float(E2M1[code]) * 2.0 ** (s - 127)
    return out


def check_w13(w1, w3, s1, s3, words, scales, samples=4000, seed=0):
    I = w1.shape[0]
    W = torch.cat([nibbles(w1), nibbles(w3)], 0)
    S = torch.cat([s1, s3], 0).to(torch.int64)
    gen = torch.Generator().manual_seed(seed)
    bad = 0
    for _ in range(samples):
        q, kb, c, t, j = (int(torch.randint(n, (1,), generator=gen)) for n in (I // 64, W.shape[1] // 64, 2, 128, 4))
        wv, g, tq = t // 32, (t % 32) // 4, t % 4
        i = 64 * q + 32 * c + 8 * wv + g
        k0 = 64 * kb + 16 * j
        b = k0 // 32
        R = 64 * c + 16 * wv + g
        assert int(scales[q, b, R]) == int(S[i, b]) and int(scales[q, b, R + 8]) == int(S[I + i, b])
        frag = emulate_fragment(int(words[q, kb, c, t, j]), int(scales[q, b, R]), int(scales[q, b, R + 8]))
        for reg, (row, kof) in enumerate(((i, 0), (I + i, 0), (i, 8), (I + i, 8))):
            for half in (0, 1):
                k = k0 + 2 * tq + kof + half
                want = float(E2M1[W[row, k]]) * 2.0 ** (int(S[row, k // 32]) - 127)
                bad += frag[(reg, half)] != want
    return bad


def check_w2(w2, s2, words, scales, samples=4000, seed=1):
    W = nibbles(w2)
    S = s2.to(torch.int64)
    gen = torch.Generator().manual_seed(seed)
    bad = 0
    for _ in range(samples):
        rt, kb, c, t, j = (int(torch.randint(n, (1,), generator=gen)) for n in (W.shape[0] // 128, W.shape[1] // 64, 2, 128, 4))
        wv, g, tq = t // 32, (t % 32) // 4, t % 4
        r = 128 * rt + 64 * c + 16 * wv + g
        k0 = 64 * kb + 16 * j
        R = 64 * c + 16 * wv + g
        frag = emulate_fragment(int(words[rt, kb, c, t, j]), int(scales[rt, k0 // 32, R]), int(scales[rt, k0 // 32, R + 8]))
        for reg, (row, kof) in enumerate(((r, 0), (r + 8, 0), (r, 8), (r + 8, 8))):
            for half in (0, 1):
                k = k0 + 2 * tq + kof + half
                bad += frag[(reg, half)] != float(E2M1[W[row, k]]) * 2.0 ** (int(S[row, k // 32]) - 127)
    return bad


def main():
    gen = torch.Generator().manual_seed(42)
    I, D = 576, 5120
    w1 = torch.randint(0, 256, (I, D // 2), dtype=torch.uint8, generator=gen)
    w3 = torch.randint(0, 256, (I, D // 2), dtype=torch.uint8, generator=gen)
    s1 = torch.randint(110, 130, (I, D // 32), dtype=torch.uint8, generator=gen)
    s3 = torch.randint(110, 130, (I, D // 32), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (D, I // 2), dtype=torch.uint8, generator=gen)
    s2 = torch.randint(110, 130, (D, I // 32), dtype=torch.uint8, generator=gen)
    a, sa = repack_w13(w1, w3, s1, s3)
    b, sb = repack_w2(w2, s2)
    print(f"random: w13 words {tuple(a.shape)} scales {tuple(sa.shape)}; w2 words {tuple(b.shape)} scales {tuple(sb.shape)}")
    # every nibble lands exactly once (bijection): compare multisets of codes
    assert sorted(nibbles(torch.cat([w1, w3])).flatten().tolist()) == sorted(
        ((a.flatten().view(-1, 1) >> (4 * torch.arange(8))) & 15).flatten().tolist())
    print(f"random: w13 mismatches {check_w13(w1, w3, s1, s3, a, sa)}, w2 mismatches {check_w2(w2, s2, b, sb)} "
          "(4,000 sampled words x 8 values each)")
    if len(sys.argv) > 1:  # real expert from the checkpoint: python3 repack.py <model_dir> [layer] [expert]
        import json, os
        from safetensors import safe_open
        d = sys.argv[1]
        L = int(sys.argv[2]) if len(sys.argv) > 2 else 21
        e = int(sys.argv[3]) if len(sys.argv) > 3 else 7
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

        def get(k):
            with safe_open(os.path.join(d, idx[k]), "pt") as f:
                return f.get_tensor(k).view(torch.uint8)
        p = f"layers.{L}.ffn.experts.{e}."
        rw1, rw3, rw2 = get(p + "w1.weight")[:I], get(p + "w3.weight")[:I], get(p + "w2.weight")[:, : I // 2]
        rs1, rs3, rs2 = get(p + "w1.scale")[:I], get(p + "w3.scale")[:I], get(p + "w2.scale")[:, : I // 32]
        a, sa = repack_w13(rw1, rw3, rs1, rs3)
        b, sb = repack_w2(rw2, rs2)
        print(f"real layer {L} expert {e} rank 0: w13 mismatches {check_w13(rw1, rw3, rs1, rs3, a, sa)}, "
              f"w2 mismatches {check_w2(rw2, rs2, b, sb)}")


if __name__ == "__main__":
    main()
