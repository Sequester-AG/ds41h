// generated (moe_x2): wgmma.mma_async m64nNk32 f32 (+)= e4m3 (A registers, 4 x b32) x e4m3 (B SMEM, K-major), the B
// descriptor given as its low 32-bit word; the high word HI comes from an immediate (SW128 rows of 128 B: 0x40000040 =
// SBO 1024; SW64 rows of 64 B: 0x80000020 = SBO 512). mma: fresh accumulator (scale-d = 0); mmac: accumulate.
#pragma once
constexpr unsigned F8_SW128 = 0x40000040u, F8_SW64 = 0x80000020u;
template <int N> struct WgF8;
template <> struct WgF8<8> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[4], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %9;\nmov.b64 d, {%8, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[4], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %9;\nmov.b64 d, {%8, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<16> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[8], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %13;\nmov.b64 d, {%12, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[8], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %13;\nmov.b64 d, {%12, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<24> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[12], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %17;\nmov.b64 d, {%16, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[12], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %17;\nmov.b64 d, {%16, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<32> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[16], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %21;\nmov.b64 d, {%20, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[16], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %21;\nmov.b64 d, {%20, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<40> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[20], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %25;\nmov.b64 d, {%24, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[20], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %25;\nmov.b64 d, {%24, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<48> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[24], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %29;\nmov.b64 d, {%28, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[24], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %29;\nmov.b64 d, {%28, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<56> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[28], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %33;\nmov.b64 d, {%32, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[28], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %33;\nmov.b64 d, {%32, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
template <> struct WgF8<64> {
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mma(float (&d)[32], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %37;\nmov.b64 d, {%36, h};\n.reg .pred p;\nsetp.ne.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
  template <unsigned HI = F8_SW128>
  __device__ __forceinline__ static void mmac(float (&d)[32], const unsigned (&a)[4], unsigned dlo) {
    asm volatile("{\n.reg .b64 d;\n.reg .b32 h;\nmov.b32 h, %37;\nmov.b64 d, {%36, h};\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, d, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(dlo), "n"(HI));
  }
};
