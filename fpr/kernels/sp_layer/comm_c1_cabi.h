// comm_c1_cabi.h -- C ABI of kernels/sp_layer/comm_c1.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The decode buckets' real-row collectives (Engine.p_ag / p_rs / _tp_wo1 / p_tpwo2 / p_tpwo3; drafter.py HSD_TPWO):
// ag_push2v (+ _wait: the last CTA waits), ag_rt (router top-k folded into the AG push), rs_pushv / rs_cpushv (+ combine)
// / rs_sumv, the TP-wo trio tp_scatter / tp_rs / tp_rs_sum, and the lamport one-kernel reduces tp_rsl / rs_cpushl.
// Peers: one device pointer per rank (W = 4) for every gather / receive buffer and flag; peer access must be on (comm.cu's
// enable_peers stays Python). Flags / epochs / tickets / ticks: uint32 counters, zero at allocation, never reset (the
// caller owns them: one ticket per call site, one tick per CTA of the lamport kernels). c.rank = this rank.
// Every kernel is launched with cudaLaunchKernelEx and the programmatic-stream-serialization attribute = ENG_C1_PDL, read
// ONCE per process: "comm_c1.init" latches it (call it before the driver threads launch; every other entry returns
// FPR_E_INIT until it, or a pybind launch, has). No other state: no dynamic SMEM, no device buffers.
// Table: fpr_comm_c1_table(); "comm_c1.init", "comm_c1.ag_push2v", "comm_c1.ag_push2v_wait", "comm_c1.ag_rt",
// "comm_c1.rs_cpushv", "comm_c1.rs_pushv", "comm_c1.rs_sumv", "comm_c1.tp_scatter", "comm_c1.tp_rs", "comm_c1.tp_rs_sum",
// "comm_c1.tp_rsl", "comm_c1.rs_cpushl". (pybind arl: not in the table, the engine does not call it.)
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_COMM_C1_W 4

typedef struct { FprCall c; } FprCommC1InitArgs;   // latches ENG_C1_PDL (idempotent; any device)

typedef struct {                       // = pybind ag_push2v(src, src2, rank, bufs, flags, bufs2, epoch, ncta)
  FprCall c;                           // c.rank: this rank (its slot in every gather buffer)
  const void* src;                     // [R, RB bytes] f rows (pushed only where src2's id 0 >= 0)
  const void* src2;                    // [R, 64 B] routing rows (always pushed)
  int32_t R;
  int32_t RB;                          // % 16 == 0
  int32_t RB2;                         // == 64
  int32_t ncta;                        // CTAs per destination (grid W * ncta)
  void* bufs[FPR_COMM_C1_W];           // f gather buffer [W * R, RB] on each rank
  void* bufs2[FPR_COMM_C1_W];          // routing gather buffer [W * R, 64] on each rank
  uint32_t* flags[FPR_COMM_C1_W];      // f flag on each rank (one signal per CTA covers both row sets)
  uint32_t* epoch;                     // this rank's AG round counter (bumped by the kernel)
} FprCommC1AgPush2vArgs;

typedef struct {                       // = pybind ag_push2v_wait(src, src2, rank, bufs, flags, bufs2, epoch, ncta, myflag,
  FprCall c;                           //                         signals, ticket)
  const void* src;
  const void* src2;
  int32_t R;
  int32_t RB;                          // % 16 == 0
  int32_t RB2;                         // == 64
  int32_t ncta;
  void* bufs[FPR_COMM_C1_W];
  void* bufs2[FPR_COMM_C1_W];
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;
  const uint32_t* myflag;              // this rank's f flag (= flags[c.rank]): the last CTA waits for it
  uint32_t* ticket;                    // >= 1 uint32, monotonic (one per call site)
  uint32_t signals;                    // per round (W * ncta)
} FprCommC1AgPush2vWaitArgs;

typedef struct {                       // = pybind ag_rt(src, logits, bias, route_scale, valid, ptok, ids, wts, rrow, rank, bufs,
  FprCall c;                           //                flags, bufs2, epoch, ncta, myflag, signals, ticket)
  const void* src;                     // [R, RB bytes] f rows (pushed where valid[row] != 0)
  const float* logits;                 // [>= R, 384] fp32, contiguous
  const float* bias;                   // [384]
  const uint8_t* valid;                // [R] row mask
  const int32_t* ptok;                 // [R] prescale exponents (routing row bytes 48..51)
  int32_t* ids;                        // [R, 6] out (this rank)
  float* wts;                          // [R, 6] out (this rank)
  void* rrow;                          // [R, 64 B] this rank's routing rows out
  float route_scale;
  int32_t R;
  int32_t RB;                          // % 16 == 0
  int32_t ncta;
  void* bufs[FPR_COMM_C1_W];
  void* bufs2[FPR_COMM_C1_W];          // routing gather buffers [W * R, 64]
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;
  const uint32_t* myflag;
  uint32_t* ticket;
  uint32_t signals;
} FprCommC1AgRtArgs;

typedef struct {                       // = pybind rs_cpushv(Y, slotmap, sh, R, rank, bufs, flags, epoch, ncta, gr)
  FprCall c;
  const void* Y;                       // bf16 [slots, H] expert outputs
  const int32_t* slotmap;              // [W * R, 6]
  const void* sh;                      // bf16 [W * R, H] shared expert, or NULL
  const void* gr;                      // [W * R, 64 B] this rank's gathered routing rows
  int32_t R;                           // rows per rank
  int32_t H;                           // % 8 == 0
  int32_t ncta;                        // CTAs per destination (grid W * ncta)
  void* bufs[FPR_COMM_C1_W];           // receive buffer bf16 [W, R, H] on each rank
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;                     // this rank's RS round counter (bumped by the kernel)
} FprCommC1RsCpushvArgs;

typedef struct {                       // = pybind rs_pushv(part, rank, bufs, flags, epoch, ncta, gr)
  FprCall c;
  const void* part;                    // bf16 [W * R, H] (all rows)
  const void* gr;                      // [W * R, 64 B]
  int32_t R;                           // part.size(0) / W
  int32_t H;
  int32_t ncta;
  void* bufs[FPR_COMM_C1_W];
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;
} FprCommC1RsPushvArgs;

typedef struct {                       // = pybind rs_sumv(recv, flag, epoch, signals, out, ncta, gr, rank)
  FprCall c;                           // c.rank: this rank (its real rows = gathered routing rows c.rank * R + i)
  const void* recv;                    // bf16 [W, R, H] (slot = source rank)
  const uint32_t* flag;
  const uint32_t* epoch;
  void* out;                           // bf16 [R, H]
  const void* gr;                      // [W * R, 64 B]
  int32_t R;
  int32_t H;
  uint32_t signals;
  int32_t ncta;                        // grid
} FprCommC1RsSumvArgs;

typedef struct {                       // = pybind tp_scatter(o, valid, rank, bufs, flags, vbufs, epoch, ncta, myflag, signals,
  FprCall c;                           //                     ticket, pos, cos_sin)
  const void* o;                       // bf16 [R, 32768] attention output, contiguous
  const uint8_t* valid;                // [R]
  const int64_t* pos;                  // [R] positions, or NULL (no inverse RoPE on the way)
  const float* cos_sin;                // [*, 64] (read only when pos != NULL; the pybind path passes NULL with pos)
  int32_t R;                           // <= 512
  int32_t ncta;
  void* bufs[FPR_COMM_C1_W];           // head-group slices bf16 [W * R, 8192] on each rank
  void* vbufs[FPR_COMM_C1_W];          // gathered row masks uint8 [W * R] on each rank
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;
  const uint32_t* myflag;
  uint32_t* ticket;
  uint32_t signals;
} FprCommC1TpScatterArgs;

typedef struct {                       // = pybind tp_rs(part, vm, rank, bufs, flags, epoch, ncta)
  FprCall c;
  const float* part;                   // fp32 [W * R, 5120], contiguous
  const uint8_t* vm;                   // [W * R] gathered row masks
  int32_t R;                           // part.size(0) / W
  int32_t ncta;
  void* bufs[FPR_COMM_C1_W];           // receive buffer fp32 [W, R, 5120] on each rank
  uint32_t* flags[FPR_COMM_C1_W];
  uint32_t* epoch;
} FprCommC1TpRsArgs;

typedef struct {                       // = pybind tp_rs_sum(recv, flag, epoch, signals, y, ncta, valid)
  FprCall c;
  const float* recv;                   // fp32 [W, R, 5120]
  const uint32_t* flag;
  const uint32_t* epoch;
  void* y;                             // bf16 [R, 5120] out (real rows)
  const uint8_t* valid;                // [R]
  int32_t R;
  uint32_t signals;
  int32_t ncta;                        // grid
} FprCommC1TpRsSumArgs;

typedef struct {                       // = pybind tp_rsl(part, vm, rank, lbufs, tick, y, valid, grid)
  FprCall c;
  const float* part;                   // fp32 [W * R, 5120], contiguous
  const uint8_t* vm;                   // [W * R]
  void* y;                             // bf16 [R, 5120] out
  const uint8_t* valid;                // [R]
  uint32_t* tick;                      // [>= grid] per-CTA call counters (slot parity)
  int32_t R;
  int32_t grid;
  float* lbufs[FPR_COMM_C1_W];         // lamport slots fp32 [2][W][R][5120] on each rank, filled with -0.0 at allocation
} FprCommC1TpRslArgs;

typedef struct {                       // = pybind rs_cpushl(Y, slotmap, sh, R, rank, lbufs, tick, out, gr, grid)
  FprCall c;
  const void* Y;                       // bf16 [slots, H]
  const int32_t* slotmap;              // [W * R, 6]
  const void* sh;                      // bf16 [W * R, H] or NULL
  void* out;                           // bf16 [R, H] out
  const void* gr;                      // [W * R, 64 B]
  uint32_t* tick;                      // [>= grid]
  int32_t R;
  int32_t H;                           // % 8 == 0
  int32_t grid;
  void* lbufs[FPR_COMM_C1_W];          // lamport slots bf16 [2][W][R][H] on each rank, filled with -0.0 at allocation
} FprCommC1RsCpushlArgs;

const FprExtTable* fpr_comm_c1_table(void);

#ifdef __cplusplus
}
#endif
