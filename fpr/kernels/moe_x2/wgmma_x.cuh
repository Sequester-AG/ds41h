// generated (moe_x): wgmma.mma_async m64nNk16 f32 += f16 (A registers) x f16 (B SMEM, K-major, 128B swizzle, SBO 1024)
// with the B descriptor given as its low 32-bit word; the high word (SBO 64 x 16 B, swizzle 128B) is built inside
// the asm from an immediate, so ptxas can keep it in a uniform register instead of rebuilding it per wgmma.
#pragma once
template <int N> struct WgX;
template <> struct WgX<8> {
  __device__ __forceinline__ static void mma(float (&d)[4], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%8, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<16> {
  __device__ __forceinline__ static void mma(float (&d)[8], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%12, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<24> {
  __device__ __forceinline__ static void mma(float (&d)[12], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%16, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<32> {
  __device__ __forceinline__ static void mma(float (&d)[16], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%20, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<40> {
  __device__ __forceinline__ static void mma(float (&d)[20], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%24, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<48> {
  __device__ __forceinline__ static void mma(float (&d)[24], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%28, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<56> {
  __device__ __forceinline__ static void mma(float (&d)[28], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%32, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
template <> struct WgX<64> {
  __device__ __forceinline__ static void mma(float (&d)[32], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, 1073741888;\nmov.b64 d, {%36, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, d, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo));
  }
};
