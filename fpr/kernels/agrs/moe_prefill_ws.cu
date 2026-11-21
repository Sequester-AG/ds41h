// moe_prefill_ws.cu: production's moe_prefill.cu (the private copy
// in build/pmc_src, module block renamed to ds41h_prefill_bind by pm_fused.sync_sources) #included UNCHANGED as its own
// translation unit, as the served builds compile it, plus HOST code only:
//   agrs_pmws_set_hybrid: the setter of its file-scope set_hybrid statics for agrs_pmws_init (the pybind set_hybrid lambda
//     writes the same two ints).
#include "moe_prefill.cu"

void agrs_pmws_set_hybrid(int64_t split, int64_t engine) {
  g_hybrid_split = int(split);
  g_hybrid_engine = int(engine);
}
