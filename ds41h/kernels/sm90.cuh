// sm90 (H200) low-precision helpers for decode-path kernels.
//
// Key facts (verified with nvcc 13.0 SASS on this box):
//  * Hopper has NO native FP8 mma.sync: `mma.sync...e4m3.e4m3` lowers to F2FP.F16.E4M3.UNPACK_B
//    (A and B) + 2x HMMA.16816. Native FP8 tensor-core math on sm_90a only comes from wgmma (QGMMA).
//  * So decode (M <= 8 tokens, memory bound) uses either
//      (a) FP16 HMMA m16n8k16 with weights expanded FP4 -> e4m3 (bit trick) -> f16 (hardware cvt),
//          activations staged once as f16; or
//      (b) RS-wgmma m64n8k32 e4m3 with weights expanded FP4 -> e4m3 in registers (A operand) and
//          tokens as the N=8 B operand from smem (activations quantized to e4m3 per token).
//    Which is faster at 200 W is an open benchmark question (ALU issue rate vs clocks); keep both.
//
// FP4 (e2m1) packing in the checkpoint: I8 [rows, K/2], low nibble = even k (convert.py FP4_TABLE),
// scales F8_E8M0 [rows, K/32] (1x32 along K).
#pragma once

#include "common.cuh"

// ---------------------------------------------------------------- FP4 -> e4m3 (exact, x 2^-6)
// e2m1 (s e1 e0 m0) -> e4m3 byte (s<<7) | (e1 e0 m0)<<2 encodes value * 2^-6 exactly, including the
// 0.5 subnormal (e4m3 subnormal 0.5*2^-6) and zero. Fold 2^6 into the scale (ue8m0 byte + 6).
// fp4_lo takes the even (low) nibble of each byte, fp4_hi the odd (high) nibble. (BRRR decoder.cu)
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }

// ---------------------------------------------------------------- e4m3 -> f16 (hardware, 2 per instr)
__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned short v) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
  return h;
}

// ---------------------------------------------------------------- FP4 -> f16 directly (exact, x 2^-14)
// f16 bits ((v&7)<<9) | ((v&8)<<12) == e2m1 value * 2^-14 for all 16 codes (incl. 0.5 -> f16 subnormal).
// Pairs nibble j (low half) with nibble j+4 (high half) of the 32-bit word.
__device__ __forceinline__ unsigned fp4_pair_f16x2(unsigned w, int j) {
  const unsigned x = w >> (4 * j);
  return ((x & 0x00070007u) << 9) | ((x & 0x00080008u) << 12);
}

// ---------------------------------------------------------------- HMMA m16n8k16 (f16 in, f32 acc)
// Fragment map (g = lane>>2, t = lane&3):
//   A (16x16, row-major): a0=(g, 2t..2t+1) a1=(g+8, 2t..2t+1) a2=(g, 2t+8..2t+9) a3=(g+8, 2t+8..2t+9)
//   B (16x8, col-major):  b0=(2t..2t+1, g)  b1=(2t+8..2t+9, g)
//   C (16x8):             c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)
// Weights are A (rows = output features), tokens are B/N (<= 8 per tile).
// Any permutation of k inside a 32-wide scale block is legal as long as the activation (B) staging
// applies the same permutation, so weights can be consumed in checkpoint order.
__device__ __forceinline__ void hmma_f16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma_bf16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Combined per-(row, k32) dequant factor for an FP4 block expanded via e4m3 (x 2^-6 -> +6) or
// directly to f16 (x 2^-14 -> +14): returns 2^(ue8m0 - 127 + bias) as float.
template <int BIAS>
__device__ __forceinline__ float fp4_block_scale(unsigned ue8m0) { return __uint_as_float((ue8m0 + BIAS) << 23); }

// ---------------------------------------------------------------- FP4 -> e4m3 with the per-32 scale folded in
// (added by the moe component, used by moe_prefill.cu; reusable by decode kernels)
// Each 32-wide FP4 block with ue8m0 exponent e is expanded relative to a reference exponent
// e_ref >= e of its row (e.g. the max over the 4 blocks of a 128-wide K block): e2m1 magnitude
// code c (0..7) -> e4m3 byte of value(c) * 2^(6 + d), d = e - e_ref <= 0. Exact for d >= -14
// (6 * 2^6 = 384 <= 448 and 0.5 * 2^-8 = 2^-9 is the smallest e4m3 subnormal); rn/satfinite
// rounded beyond. The dequantized weight is (e4m3 value) * fp4_ref_scale(e_ref). The checkpoint's
// per-row spreads are <= 4 binades (measured on layers 0/10/20/39), so this is exact in practice.
__device__ __forceinline__ unsigned prmt_b32(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// 8-entry byte LUT (prmt operands {l0, l1}) for magnitude codes 0..7 at relative exponent d.
__device__ __forceinline__ void fp4_e4m3_lut(int d, unsigned& l0, unsigned& l1) {
  if (d >= -11) {  // all codes normal in e4m3: byte(c) = (c << 2) + ((12 + d) << 3), code 1 -> (12+d) << 3
    const unsigned k = unsigned(12 + d) << 3;
    l0 = k * 0x01010100u + 0x0C080000u;  // {0, k, k+8, k+12}
    l1 = k * 0x01010101u + 0x1C181410u;  // {k+16, k+20, k+24, k+28}
  } else {
    const float sc = ldexpf(1.0f, 6 + d);
    l0 = e4m3x2(0.0f, 0.5f * sc) | (e4m3x2(sc, 1.5f * sc) << 16);
    l1 = e4m3x2(2.0f * sc, 3.0f * sc) | (e4m3x2(4.0f * sc, 6.0f * sc) << 16);
  }
}
// 8 packed e2m1 values (nibble i = element i, low nibble first) -> 8 e4m3 bytes in element order:
// lo = elements 0..3, hi = elements 4..7. 9 integer ops (2 prmt for magnitudes, 2 prmt in
// sign-replicate mode for the signs).
__device__ __forceinline__ void fp4x8_to_e4m3_lut(unsigned w, unsigned l0, unsigned l1, unsigned& lo, unsigned& hi) {
  const unsigned m = w & 0x77777777u;
  const unsigned w4 = w << 4;                       // byte i msb = sign of element 2i
  const unsigned mag_lo = prmt_b32(l0, l1, m);       // selector nibbles = codes of elements 0..3
  const unsigned mag_hi = prmt_b32(l0, l1, m >> 16);  // elements 4..7
  const unsigned sgn_lo = prmt_b32(w, w4, 0x9D8Cu);   // replicate sign of elements 0,1,2,3
  const unsigned sgn_hi = prmt_b32(w, w4, 0xBFAEu);   // replicate sign of elements 4,5,6,7
  lo = (sgn_lo & 0x80808080u) | mag_lo;
  hi = (sgn_hi & 0x80808080u) | mag_hi;
}
// 2^(e_ref - 127 - 6): the factor that undoes the +6 placement of fp4_e4m3_lut.
__device__ __forceinline__ float fp4_ref_scale(int e_ref) {
  return e_ref >= 7 ? __uint_as_float(unsigned(e_ref - 6) << 23) : ldexpf(1.0f, e_ref - 133);
}
