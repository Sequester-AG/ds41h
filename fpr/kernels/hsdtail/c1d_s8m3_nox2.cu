// graph H = 606-632 us/step of which the drafter's c1d k_layer is 323 us. Source: src/ = a frozen copy of fpr/kernels/c1mega/srv8m (SOURCE.md5).
// Shape knobs = c1d's (NE 128, TOPK 3, WIN 131 = 128 + 3 keys, TKA 0) + C1M_INS_OLD.
// Matrices from c1d's own fp8 copies: C1M_DEP_B16 0 and the DEFAULT dense fragments (not srv8x's W8A16-packed _l / _u8 ones,
// which read the engine's packed views). Whole-layer kernel only, no index / Engram phases (no C1M_DEP_IX / ix hooks / C1IDX_*).
// Not applicable at the drafter's shape (compile-time asserts): C1M_CML (ANCH 17..32; drafter 5), C1M_STF (KVR % 32; 131),
// C1S_RT1T (NE == 3 x 128; 128), C1M_OWNSTG + C1M_UNIT_EARLY (the new insert path, WIN <= 128), C1M_FN_FRAG2 (engine fragments).
// Also: src/c1mega.cu's counter-layout assert is scoped to TOPK 6.
// srv8m's levers kept: (a) C1M_UNIT_EARLY, (c) C1M_CML, cluster-2 og multicast, x2r56 / x2t experts,
// the a2r H chain (C1S_RTS2 RT1T GPF T6W SVC), RTL / RTF / FONE_H / DU / FIMG / RNP / NWP / OWNSTG / STF / LBM, FN_FRAG2.
#define C1M_NE 128
#define C1M_TOPK 3
#define C1M_WIN 131
#define C1M_TKA 0
#define C1M_INS_OLD
// (ANCH 17..32 only; drafter ANCH 5) #define C1M_CML
// (needs OWNSTG) #define C1M_UNIT_EARLY
#define C1S_CLS 2
#define C1S_OGMC
// (reads the engine's mhcf prep_f fragments; the drafter keeps c1d's fn slices) #define C1M_FN_FRAG2
#define C1M_DENSE_A5 "c1dense_a5_e8.inc"
#define C1M_ONLY_FULL
#define C1M_DEP_B16 0
// (bisect nox2) #define C1M_XN 2
// (bisect nox2) #define C1M_XFRAG
// (bisect nox2) #define C1M_W13KB 5
// (bisect nox2) #define X2Y_B 2
// (bisect nox2) #define X2T_TAILK 56
// (bisect nox2) #define C1M_MOE_PRODUCE "c1moe_x2r_produce.inc"
// (bisect nox2) #define C1M_MOE_CONSUME "c1moe_x2t_consume.inc"
#define C1M_RTL
#define C1M_RTF
#define C1M_FONE_H
#define C1M_DU
#define C1M_FIMG
#define C1M_RNP
#define C1M_NWP
// (new insert path only: WIN <= 128) #define C1M_OWNSTG
// (KVR % 32; drafter KVR 131) #define C1M_STF
#define C1M_LBM
#define C1S_RTS2
// (NE == 3 x 128; drafter NE 128) #define C1S_RT1T
#define C1S_GPF
#define C1S_T6W
#define C1S_SVC
#define C1M_MOE_PRODUCE "c1moe_x2j_produce.inc"
#define C1M_MOE_CONSUME "c1moe_x2j_consume.inc"
#include "c1d_src3/c1mega_depn.cu"
