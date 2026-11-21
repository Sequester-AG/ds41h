#define C1M_UNIT_EARLY
// Bitwise vs srv8x on rows with finite keys.
// Fragments
// c1dense_f1_l8.inc / c1dense_f2_u8.inc
// C1S_CLS 2: 66 clusters x 2 CTAs
// via cudaLaunchKernelEx, per-device co-residency TORCH_CHECK (max active clusters >= 66); C1S_OGMC: the pair's rank-0 CTA
// waits C_O and bulk-multicasts the og rows into both CTAs' image rows.
// Bitwise expected
// cl-02: graph L -0.064 ms per step.
#define C1S_CLS 2
#define C1S_OGMC
#define C1M_FN_FRAG2
#define C1M_DENSE_A5 "c1dense_a5_e8.inc"
#define C1M_DEP_IX
#define C1M_HOOKS "c1hooks_ix.inc"
#define C1M_ONLY_FULL
#define C1M_DEP_B16 121   // (srv8x: 1 wqkv | 8 sh | 16 qb8 | 32 woa8 | 64 wob8)
#define C1M_XN 2
#define C1M_XFRAG
#define C1M_W13KB 5
#define X2Y_B 2
#define X2T_TAILK 56
#define C1M_MOE_PRODUCE "c1moe_x2r_produce.inc"
#define C1M_MOE_CONSUME "c1moe_x2t_consume.inc"
#define C1M_RTL
#define C1M_RTF
#define C1M_FONE_H
#define C1M_DU
#define C1M_FIMG
#define C1M_RNP
#define C1M_NWP
#define C1M_OWNSTG
#define C1M_STF
#define C1M_LBM
#define C1IDX_NO_DS
#define C1IDX_X1IMG
#define C1IDX_ARRM
#define C1M_DENSE_A3 "c1dense_a3_l.inc"
#define C1M_DENSE_F1 "c1dense_f1_l8.inc"
#define C1M_DENSE_F2 "c1dense_f2_u8.inc"
#define C1S_RTS2
#define C1S_RT1T
#define C1S_GPF
#define C1S_T6W
#define C1S_SVC
#include "c1mega_depn.cu"
