// c1d_base.cu: the drafter's c1mega instance (fpr/kernels/hsdtail/c1d.py, kind "base") =
// fpr/kernels/c1mega/c1mega.cu with the drafter's shape knobs and the expert fragments PINNED to c1moe_produce /
// consume.inc. C1M_WIN 131 = the ring's keys.
// quoted
// includes inside it resolve in c1d_src.
#define C1M_NE 128
#define C1M_TOPK 3
#define C1M_WIN 131
#define C1M_TKA 0
#define C1M_MOE_PRODUCE "c1moe_produce.inc"
#define C1M_MOE_CONSUME "c1moe_consume.inc"
#include "c1d_src/c1mega.cu"
