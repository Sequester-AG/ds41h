// c1mega_dep.cu: the deployment-config build (C1M_SRC=c1mega_dep.cu): attention
// mode 3, the bf16 set fixed at compile time (C1M_DEP_B16, default 15 = q_a|wkv, wo_a, wo_b and the shared expert all from
// the engine's bf16 tensors), timeline kept (the tests stamp); the other paths compiled out (less code per launch)
#define C1M_FIX_AMODE 3
#ifndef C1M_DEP_B16
#define C1M_DEP_B16 15
#endif
#define C1M_FIX_B16 C1M_DEP_B16
#ifndef C1M_DEP_IX   // (no in-kernel index layers yet: their A2 output compiled out)
#define C1M_NO_IX
#endif
#include "c1mega.cu"
