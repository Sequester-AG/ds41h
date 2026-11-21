// moe_decide.cpp -- see moe_decide.h. Line by line = mixed_cxx_moe.decide(drv, s).
#include "moe_decide.h"

namespace fpr {
namespace p0 {

int32_t moe_decide(const MoeSnap& m, const MoeBucket& b, int64_t vcount, int64_t ndec) {
  if (!b.prod) return MOE_PY;                                            // s.get("moe") != "prod"
  if (vcount > m.prefill_moe_switch || !b.has_ecs) {                     // Engine.p_moe -> p_moe_prod
    if (!m.prod_ok || !m.pm_fused || vcount <= m.small_T) return MOE_PY;  // unfused prod: Python
    if (m.pf_act8 && ndec >= 0) return MOE_PY;                           // ENG_PF_ACT8's layer8: Python
    if (m.nt) return MOE_PY;                                             // narrow-tile kernels: Python
    const int64_t cap = (b.device >= 0 && b.device < kMoeMaxDev) ? m.ws_cap[b.device] : 0;
    if (!m.ws || vcount > cap) return MOE_PY;                            // above the workspace's rows: the Python fallback
    if (m.agrs_rsf && b.R >= m.min_r) return MOE_PROD_RSF;
    return m.cxx_fused ? MOE_PROD_FUSED : MOE_PY;
  }
  if (!m.fork_ok || m.dump_moe || b.rs_lb128) return MOE_PY;
  const int64_t Tall = (int64_t)m.W * b.R;
  if (Tall <= 1365 && m.moe_v4 && m.v4_mixed) return MOE_FORK_V4;
  return MOE_FORK_RAW;
}

}  // namespace p0
}  // namespace fpr
