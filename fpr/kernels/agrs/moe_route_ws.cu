// moe_route_ws.cu
// production's moe_route.cu (the private copy in build/pmc_src) #included UNCHANGED as its own translation unit, as the
// served builds compile it (agrs_pm_rs / mpf_pm_fused), plus HOST code only:
//   agrs_pmws_moe_permute: ds41h_moe_permute's statements with the hist [G E] i32 from the caller's workspace instead of
//     torch::empty (its moe_count / moe_scatter kernels are in moe_route.cu's anonymous namespace, so the launches live in
//     this TU). Same kernels, same launch_pdl configs, same arguments.
//   agrs_pmws_hist_ints: G E for R routed rows (the workspace size of that hist).
// (A first build put moe_route.cu + moe_prefill.cu + pm_rs.cu into ONE TU: moe_count_kernel and moe_gemm2_bs_kernel<0>
// compiled to different SASS there, so each production source keeps its own TU.)
#include "moe_route.cu"

int64_t agrs_pmws_hist_ints(int64_t R, int64_t E) { return int64_t(R > 0 ? cdiv(int(R), PERM_CHUNK) : 1) * E; }

// returns 0, or -1 for an E production's permute rejects (E < 1 or > MAX_EXPERTS); nothing is launched then
int agrs_pmws_moe_permute(const int* topk_ids, const float* topk_w, int R, int K, int64_t E, int* meta, float* perm_w, int* hist) {
  if (E < 1 || E > MAX_EXPERTS) return -1;
  const int G = R > 0 ? cdiv(R, PERM_CHUNK) : 1;
  int* m = meta;
  int* pos_of = m + 2 * (E + 1) + META_CTR + 2 * R;
  launch_pdl("moe_count", moe_count_kernel, dim3(G), dim3(512), 0, topk_ids, R, int(E), hist, pos_of);
  launch_pdl("moe_scatter", moe_scatter_kernel, dim3(G), dim3(1024), 0, topk_ids, topk_w, R, K, int(E), G,
             static_cast<const int*>(hist), m, m + (E + 1), m + 2 * (E + 1), m + 2 * (E + 1) + META_CTR, m + 2 * (E + 1) + META_CTR + R,
             perm_w, pos_of);
  return 0;
}
