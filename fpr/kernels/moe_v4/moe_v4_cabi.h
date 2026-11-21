// moe_v4_cabi.h -- C ABI of moe_v4's fork-MoE launches on the mixed path. Two extensions, as production loads two (v4.load)
//   moe_v4c_api   (moe_v4_cabi.cu: moe_v4.cu #included UNCHANGED + this ABI; production's defines MS3_S2=2 MS4_G2P=1
//                 MS4_SPOL=1): table fpr_moe_v4_table, ext "moe_v4": moe_v4.init, moe_v4.gemm1r, moe_v4.gemm2r.
//   moe_v4c_route: table fpr_moe_v4r_table, ext
//                 "moe_v4r": moe_v4r.init, moe_v4r.route2r.
// Each call = the served pybind function's statements (moe_v4.cu gemm1r / gemm2r + launch<MODE>, route2.cu route2r) on
// Args.stream. The served host state becomes init state or arguments:
//   kernel smem attributes (served: set on a device's first launch), the SM count (served: g_sms, the first device's)
//     -> <ext>.init per device;
//   route2.cu's g_pdl / g_one_max_t (route2_config, served defaults 3 / 1365) -> MoeV4Route2rArgs.pdl / one_max_t.
// No allocation, no host sync, no event, no device change. No scratch (ws_bytes NULL): the buffers are the bucket's.
#pragma once
#include <stdint.h>

#include "fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t device;          // the CURRENT device (the init sets its kernels' attributes there)
  int32_t pad_;
} MoeV4InitArgs;           // moe_v4.init and moe_v4r.init

typedef struct {
  FprCall call;
  const uint8_t* w13;      // this rank's raw experts [E, 1152, 2560] u8 (MXFP4)
  const uint8_t* s13;      // [E, 1152, 160] u8 (ue8m0)
  int64_t E;               // 384
  const void* ecs;         // route2r's EC records
  const void* tiles;       // route2r's t1 (GEMM1 items)
  const int32_t* counts;   // route2r's counts (device; counts[1] = GEMM1 items)
  const void* img;         // the token image (moe_raw prepr5 / gather6)
  void* himg;              // the H image (out)
  int64_t grid;            // CTAs; 0 = the device's SM count (served call: 0)
} MoeV4Gemm1rArgs;

typedef struct {
  FprCall call;
  const uint8_t* w2;       // [E, 5120, 288] u8
  const uint8_t* s2;       // [E, 5120, 18] u8
  int64_t E;
  const void* ecs;
  const void* tiles;       // route2r's t2 (GEMM2 items)
  const int32_t* counts;   // counts[2] = GEMM2 items
  const void* himg;        // the H image (in)
  void* yout;              // bf16 Y [slot rows, 5120] (out)
  int64_t grid;            // 0 = SM count (served call: 0)
} MoeV4Gemm2rArgs;

typedef struct {
  FprCall call;
  const void* rows;        // u8 [>= T, 64] route rows (ids int32 x 6 | weights f32 x 6 | ...): the bucket's gr
  int64_t T;               // rows routed (Tall); 1 <= T <= one_max_t
  void* ecs;               // EC records, >= route2_max_ec(T) records (moe_v4r_max_ec)
  void* t1;                // GEMM1 items, >= max_ec x 9 tiles
  void* t2;                // GEMM2 items, >= max_ec x 40 tiles
  int32_t* counts;         // >= 6 ints
  int32_t* slotmap;        // >= 6 T ints
  int64_t ecs_bytes, t1_bytes, t2_bytes, counts_n, slotmap_n;   // capacities, checked as route2r checks them
  int64_t grid;            // slot-group CTAs (engine: the bucket's sms); 0 = SM count
  int32_t pdl;             // route2_config pdl (served default 3: bit 1 = k_one with programmatic dependent launch)
  int32_t one_max_t;       // route2_config one_max_t (served default 1365)
} MoeV4Route2rArgs;

// = the served swiglu / prepr6 / combine, one launch each
// (prepr6: two), on Args.stream; no init state, no scratch.
typedef struct {
  FprCall call;
  const void* gu;          // bf16 [T, 2 I]
  void* h;                 // bf16 [T, I] (out)
  int32_t T;               // grid (h.size(0))
  int32_t I;               // h.size(1), % 8 == 0 (else FPR_E_ARG)
} MoeV4SwigluArgs;         // = swiglu(gu, h): k_swiglu<<<T, I / 8>>>

typedef struct {
  FprCall call;
  const void* x;           // bf16 [T, 5120]
  void* ecs;               // EC records (route2r)
  const int32_t* counts;   // route2r's counts
  void* img;               // token image (out)
  int32_t* ptok;           // int32 [>= T] per-token prescale (out of k_prescale2, read by k_gather5)
  float* inv;              // fp32 [>= T] (out of k_prescale2)
  int32_t T;               // x.size(0)
  int32_t sms;             // k_gather5 grid = MSR_G5B x sms (the served call's sms argument)
} MoeV4Prepr6Args;         // = prepr6(x, ecs, counts, img, ptok, inv, sms)

typedef struct {
  FprCall call;
  const void* Y;           // bf16 [slots, 5120]
  const int32_t* slotmap;  // int32 [T, 6]
  const void* sh;          // bf16 [T, 5120] shared-expert addend, or NULL (served: sh.numel() == 0)
  void* out;               // bf16 [T, 5120] (out)
  int32_t T;               // grid (out.size(0))
  int32_t pad_;
} MoeV4CombineArgs;        // = combine(Y, slotmap, sh, out): k_combine<<<T, 640>>>

int moe_v4_init(const void* args);       // MoeV4InitArgs
int moe_v4_gemm1r(const void* args);     // MoeV4Gemm1rArgs
int moe_v4_gemm2r(const void* args);     // MoeV4Gemm2rArgs
int moe_v4_swiglu(const void* args);     // MoeV4SwigluArgs
int moe_v4_prepr6(const void* args);     // MoeV4Prepr6Args
int moe_v4_combine(const void* args);    // MoeV4CombineArgs
const FprExtTable* fpr_moe_v4_table(void);

int moe_v4r_init(const void* args);      // MoeV4InitArgs
int moe_v4r_route2r(const void* args);   // MoeV4Route2rArgs
int64_t moe_v4r_max_ec(int64_t T);       // route2_max_ec(T): EC records for T rows (buffer sizing)
const FprExtTable* fpr_moe_v4r_table(void);

#ifdef __cplusplus
}
#endif
