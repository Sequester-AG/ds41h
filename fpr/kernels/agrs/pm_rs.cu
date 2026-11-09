// agrs: RSF = the RS push of a full-load mixed step fused into the production MoE's combine.
// Generated from kernels/moe_prefill/pm_fused.cu (read-only) by adding ONE kernel and ONE entry point; all other code is
// that file's, verbatim:
//   k_pm_combine_push: k_pm_combine's statements element for element (same arithmetic, same order, one bf16 rounding),
//     over a persistent grid split into W destination groups (all links at once); output row p = (rank q, row i) is
//     stored into owner q's RS receive slot `rank`
//     (rs_q + (rank R + i) H, comm.cu's k_rs_push layout) instead of part[p]; padding rows get zeros as in part.
//     Then comm.cu's protocol: fence.acq_rel.sys per thread, __syncthreads, one red.release.sys per CTA on every owner's
//     flag (an owner counts W * grid signals per round), block 0 bumps this rank's epoch first.
//   layer_rs(...): pm_layer with the combine replaced by k_pm_combine_push (part is not written).
// The owner then runs comm.cu's k_rs_sum on its receive buffer (its own flags / epoch), as after k_rs_push.
// pm_fused: the engine's production-MoE layer (engine.py p_moe_prod: shared expert slice + ds41h.moe.moe_forward_w4a16
// W4A16 0xB9 + hybrid remainders) as ONE host call per layer per rank, with the gather / scatter glue fused into two
// small kernels. Built (pm_fused.py) together with private, unmodified copies of production's moe_route.cu and
// moe_prefill.cu (ds41h/kernels/; host.cuh with engine/prod_moe_ext.py's per-device smem patch; moe_prefill.cu's
// PYBIND11_MODULE renamed to ds41h_prefill_bind) and production's NVCC flags. The GEMMs, routing, act_perm16 and the
// combine arithmetic are production's; the shared-expert swiglu is the fork's k_swiglu (pm_swiglu.cu, built without
// fast math like moe_raw.cu), reached through a launcher pointer.
//
// Row layout (engine bucket): Tall = W * R rows, rank q's rows at [q R, (q + 1) R), its valid rows the prefix
// [q R, q R + nv[q]) (engine _set_vidx); the route rows gr [Tall, 64] u8 (ids int32 x6 | weights f32 x6 | ...).
//   compact (padded = false): the valid rows only, T = vcount = sum nv (the current path's call shapes: bit-identical
//            to index_select + moe_forward_w4a16 + zero_ + index_copy_ by construction). Eager (T is a host value).
//   padded  (padded = true):  all Tall rows, T = Tall; padding rows get ids -1 (no routed rows; production drops ids
//            outside [0, E)) and a zero output row. Static shapes per bucket, per-step data only through nv / fall / gr:
//            CUDA-graph capturable. The shared expert's cuBLAS GEMMs then run on M = Tall rows (the fork MoE's choice).
// Both write part [Tall, H] in full: valid rows = bf16(shared + sum_k w_k y_k) (fp32, one rounding), padding rows 0.
#include "common.cuh"
#include "host.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

// production host entry points (moe_route.cu / moe_prefill.cu, compiled into this extension unchanged)
void ds41h_moe_permute(torch::Tensor topk_ids, torch::Tensor topk_w, int64_t E, torch::Tensor meta, torch::Tensor perm_w);
void ds41h_act_perm16(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xp);
int64_t ds41h_moe_meta_size(int64_t E, int64_t R);
void ds41h_moe_gemm(int64_t mode, torch::Tensor act, torch::Tensor act_scale, torch::Tensor w, torch::Tensor s, torch::Tensor meta,
                    torch::Tensor perm_w, int64_t R, int64_t variant, double limit, torch::Tensor out);
void ds41h_prefill_bind(pybind11::module_& m);

namespace {
constexpr int KTOP = 6;
constexpr int MAXW = 8;
constexpr int META_CTR_PM = 8;  // == moe_route.cu META_CTR (meta layout: counts | off | ctr | perm_tok | perm_slot | pos_of)

using swiglu_fn = void (*)(const void* gu, void* h, int T, int I, cudaStream_t stream);
swiglu_fn g_swiglu = nullptr;

// k_pm_prep: one CTA per call row t. compact: t -> (rank q, row i) by the prefix of nv, copies fall row q R + i to
// fv[t] and its route ids / weights; padded: row t = q R + i itself, ids / weights only (-1 / 0 for padding rows).
__global__ void __launch_bounds__(128) k_pm_prep(const uint4* __restrict__ fall, const uint8_t* __restrict__ gr, const int* __restrict__ nv,
                                                 int W, int R, int H8, int compact, uint4* __restrict__ fv, int* __restrict__ ids,
                                                 float* __restrict__ tw) {
  const int t = blockIdx.x;
  int q = 0, i = 0;
  if (compact) {
    int acc = 0;
    while (q < W - 1 && t >= acc + nv[q]) acc += nv[q++];
    i = t - acc;
  } else {
    q = t / R;
    i = t - q * R;
  }
  const int p = q * R + i;
  const bool valid = i < nv[q];
  if (threadIdx.x < KTOP) {
    const uint8_t* row = gr + size_t(p) * 64;
    ids[size_t(t) * KTOP + threadIdx.x] = valid ? reinterpret_cast<const int*>(row)[threadIdx.x] : -1;
    tw[size_t(t) * KTOP + threadIdx.x] = valid ? reinterpret_cast<const float*>(row + 24)[threadIdx.x] : 0.f;
  }
  if (compact)
    for (int c = threadIdx.x; c < H8; c += blockDim.x) fv[size_t(t) * H8 + c] = __ldg(fall + size_t(p) * H8 + c);
}

// k_pm_combine: production's moe_combine_kernel (moe_route.cu, sscale + addend path: acc = addend, then
// acc = fmaf(y, w, acc) over the valid ids in k order, one bf16 rounding) over all Tall output rows: output row p of
// rank q reads call row t (compact: prefix(q) + i, padded: p); padding rows (i >= nv[q]) are written as zeros.
__global__ void __launch_bounds__(256) k_pm_combine(const __nv_bfloat16* __restrict__ slot, const int* __restrict__ ids, int K, int H, int E,
                                                    const float* __restrict__ sscale, const __nv_bfloat16* __restrict__ addend,
                                                    __nv_bfloat16* __restrict__ out, const int* __restrict__ nv, int W, int R, int compact) {
  const int per_row = H / 8;
  const long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(W) * R * per_row) return;
  const int p = int(idx / per_row), c = int(idx % per_row) * 8;
  const int q = p / R, i = p - q * R;
  if (i >= nv[q]) {
    *reinterpret_cast<uint4*>(out + size_t(p) * H + c) = make_uint4(0u, 0u, 0u, 0u);
    return;
  }
  int t = p;
  if (compact) {
    t = i;
    for (int j = 0; j < q; ++j) t += nv[j];
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (addend) add8_bf16(acc, *reinterpret_cast<const uint4*>(addend + size_t(t) * H + c));
  for (int k = 0; k < K; ++k) {
    const int e = ids[size_t(t) * K + k];
    if (unsigned(e) >= unsigned(E)) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(slot + (size_t(t) * K + k) * H + c);
    if (sscale) {
      const float sc = sscale[size_t(t) * K + k];
      const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        const float2 f = unpack_bf16(u[ii]);
        acc[2 * ii] = fmaf(f.x, sc, acc[2 * ii]);
        acc[2 * ii + 1] = fmaf(f.y, sc, acc[2 * ii + 1]);
      }
    } else {
      add8_bf16(acc, v);
    }
  }
  *reinterpret_cast<uint4*>(out + size_t(p) * H + c) = pack8_bf16(acc);
}


__device__ __forceinline__ void agrs_red_release_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void agrs_fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
struct RsPeers {
  __nv_bfloat16* buf[MAXW];  // owner q's RS receive buffer [W, R, H] bf16
  unsigned* flag[MAXW];      // owner q's flag word (this path's own)
};

// k_pm_combine_push (agrs): k_pm_combine's per-element statements, verbatim, in a grid-stride loop; row p's output goes
// to owner q's receive slot `rank`. Grid-stride order only changes which thread computes an element, not its arithmetic.
__global__ void __launch_bounds__(256) k_pm_combine_push(const __nv_bfloat16* __restrict__ slot, const int* __restrict__ ids, int K, int H, int E,
                                                         const float* __restrict__ sscale, const __nv_bfloat16* __restrict__ addend,
                                                         RsPeers rp, const int* __restrict__ nv, int W, int R, int compact, int rank,
                                                         unsigned* __restrict__ epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per_row = H / 8;
  // destination-major split (comm.cu k_rs_push's layout): CTA group g = blockIdx / nc serves owner
  // q = (rank + 1 + g) % W, so the three NVLink peers and the local slot are written at the same time. (A flat
  // grid-stride over all W R rows sweeps the owners one after another: one link at a time, 2.2x slower.)
  const int nc = gridDim.x / W;
  const int q = (rank + 1 + int(blockIdx.x) / nc) % W, b = int(blockIdx.x) % nc;
  const long nq = (int(blockIdx.x) < nc * W) ? long(R) * per_row : 0;
  for (long j = long(b) * blockDim.x + threadIdx.x; j < nq; j += long(nc) * blockDim.x) {
    const int i = int(j / per_row), c = int(j % per_row) * 8;
    const int p = q * R + i;
    __nv_bfloat16* out = rp.buf[q] + (size_t(rank) * R + i) * H + c;
    if (i >= nv[q]) {
      *reinterpret_cast<uint4*>(out) = make_uint4(0u, 0u, 0u, 0u);
      continue;
    }
    int t = p;
    if (compact) {
      t = i;
      for (int j = 0; j < q; ++j) t += nv[j];
    }
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (addend) add8_bf16(acc, *reinterpret_cast<const uint4*>(addend + size_t(t) * H + c));
    for (int k = 0; k < K; ++k) {
      const int e = ids[size_t(t) * K + k];
      if (unsigned(e) >= unsigned(E)) continue;
      const uint4 v = *reinterpret_cast<const uint4*>(slot + (size_t(t) * K + k) * H + c);
      if (sscale) {
        const float sc = sscale[size_t(t) * K + k];
        const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
          const float2 f = unpack_bf16(u[ii]);
          acc[2 * ii] = fmaf(f.x, sc, acc[2 * ii]);
          acc[2 * ii + 1] = fmaf(f.y, sc, acc[2 * ii + 1]);
        }
      } else {
        add8_bf16(acc, v);
      }
    }
    *reinterpret_cast<uint4*>(out) = pack8_bf16(acc);
  }
  agrs_fence_sys();
  __syncthreads();
  if (threadIdx.x < W) agrs_red_release_sys(rp.flag[threadIdx.x], 1u);
}

void check_bf16_rows(const torch::Tensor& t, int64_t rows, int64_t cols, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kBFloat16 && t.dim() == 2 && t.size(0) >= rows && t.size(1) == cols,
              name, ": expected contiguous CUDA bf16 [>= ", rows, ", ", cols, "], got ", t.sizes());
}
}  // namespace

void pm_set_swiglu(int64_t fn) { g_swiglu = reinterpret_cast<swiglu_fn>(fn); }

// One layer of one rank: part [Tall, H] <- shared(fall) + routed(fall) on the valid rows (see the file comment).
// fall bf16 [Tall, H]; gr u8 [Tall, 64]; nv int32 [W] (device: valid rows per rank); vcount = sum nv (host; compact);
// raw experts (w13, s13, w2, s2); shared slice sw13 bf16 [2 Is, H], sw2 [H, Is]; gu / hsh / sh: the bucket's shared
// expert scratch ([>= T, 2 Is] / [>= T, Is] / [>= T, H]); variant: production's W4A16 variant (0xB9);
// fvbuf (optional, compact): [>= vcount, H] bf16 for the gathered rows (else from the caching allocator).
static void pm_layer_impl(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
              torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
              torch::Tensor sh, torch::Tensor part, double limit, int64_t variant, bool padded, c10::optional<torch::Tensor> fvbuf,
                          const RsPeers* rp, int rank, unsigned* epoch, int grid) {
  TORCH_CHECK(g_swiglu != nullptr, "pm_layer: set_swiglu(pm_swiglu launcher) first");
  const c10::cuda::CUDAGuard guard(fall.device());
  const int W = int(nv.numel());
  TORCH_CHECK(W >= 1 && W <= MAXW && nv.is_cuda() && nv.dtype() == torch::kInt32 && nv.is_contiguous(), "pm_layer: nv int32 [W] on device");
  const int64_t Tall = int64_t(W) * R, H = fall.size(1), E = w13.size(0), I = w13.size(1) / 2, Is = sw2.size(1);
  TORCH_CHECK(H % 8 == 0 && Is % 8 == 0 && sw13.size(0) == 2 * Is && sw13.size(1) == H && sw2.size(0) == H, "pm_layer: shapes");
  check_bf16_rows(fall, Tall, H, "fall");
  TORCH_CHECK(fall.size(0) == Tall, "pm_layer: fall must hold W R rows");
  TORCH_CHECK(gr.is_cuda() && gr.is_contiguous() && gr.dtype() == torch::kUInt8 && gr.dim() == 2 && gr.size(0) >= Tall && gr.size(1) == 64,
              "pm_layer: gr u8 [Tall, 64]");
  TORCH_CHECK(part.is_cuda() && part.is_contiguous() && part.dtype() == torch::kBFloat16 && part.size(0) == Tall && part.size(1) == H,
              "pm_layer: part bf16 [Tall, H]");
  const int64_t T = padded ? Tall : vcount;
  TORCH_CHECK(T >= 1 && T <= Tall, "pm_layer: 1 <= T <= Tall");
  check_bf16_rows(gu, T, 2 * Is, "gu");
  check_bf16_rows(hsh, T, Is, "hsh");
  check_bf16_rows(sh, T, H, "sh");
  // production's ds41h_moe_forward, W4A16 act_perm16 branch only (0xB9: GEMM1 9 = LD 4 BNMAX 256, GEMM2 11; hybrid via
  // set_hybrid); the x-gather (13 / 14), small-batch (0x100) and W4A8 variants are not handled here
  const int v1 = int(variant & 15), v2 = int((variant >> 4) & 15);
  TORCH_CHECK(!(variant & ~int64_t(0xff)) && (v1 == 9 || v1 == 10 || v1 == 11 || v1 == 12) && v2 >= 11 && v2 <= 15,
              "pm_layer: variant ", variant, " is not a W4A16 act_perm16 variant");
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto opt = fall.options();
  auto i32 = opt.dtype(torch::kInt32), f32 = opt.dtype(torch::kFloat32);
  auto ids = torch::empty({T, KTOP}, i32);
  auto tw = torch::empty({T, KTOP}, f32);
  torch::Tensor x = fall;
  if (!padded) {
    if (fvbuf.has_value() && fvbuf->defined()) {
      check_bf16_rows(*fvbuf, T, H, "fvbuf");
      x = fvbuf->narrow(0, 0, T);
    } else {
      x = torch::empty({T, H}, opt);
    }
  }
  k_pm_prep<<<unsigned(T), 128, 0, stream>>>(reinterpret_cast<const uint4*>(fall.data_ptr()), gr.data_ptr<uint8_t>(), nv.data_ptr<int>(), W,
                                             int(R), int(H / 8), padded ? 0 : 1, padded ? nullptr : reinterpret_cast<uint4*>(x.data_ptr()),
                                             ids.data_ptr<int>(), tw.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_prep launch failed");
  // shared expert slice (engine p_moe_prod: torch.mm, the fork's swiglu, torch.mm; stock rounding points)
  auto guv = gu.narrow(0, 0, T), hv = hsh.narrow(0, 0, T), shv = sh.narrow(0, 0, T);
  at::mm_out(guv, x, sw13.t());
  g_swiglu(guv.data_ptr(), hv.data_ptr(), int(T), int(Is), stream.stream());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "swiglu launch failed");
  at::mm_out(shv, hv, sw2.t());
  // routed experts (= ds41h_moe_forward's W4A16 path)
  const int64_t RR = T * KTOP;
  auto meta = torch::empty({ds41h_moe_meta_size(E, RR)}, i32);
  auto perm_w = torch::empty({RR}, f32);
  ds41h_moe_permute(ids, tw, E, meta, perm_w);
  const int64_t t0 = 2 * (E + 1) + META_CTR_PM;
  auto h = torch::empty({RR, I}, opt);
  {
    auto xp = torch::empty({RR, H}, opt);  // freed before the slot buffer is allocated (same size), as production
    ds41h_act_perm16(x, meta.narrow(0, t0 + 2 * RR, RR), KTOP, xp);
    ds41h_moe_gemm(1, xp, perm_w, w13, s13, meta, perm_w, RR, v1, limit, h);
  }
  auto slot = torch::empty({RR, H}, opt);
  ds41h_moe_gemm(2, h, perm_w, w2, s2, meta, perm_w, RR, v2, limit, slot);
  if (rp != nullptr) {  // agrs RSF: the combine stores straight into the owners' RS receive slots
    k_pm_combine_push<<<unsigned(grid), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids.data_ptr<int>(),
                                                         KTOP, int(H), int(E), tw.data_ptr<float>(),
                                                         reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()), *rp,
                                                         nv.data_ptr<int>(), W, int(R), padded ? 0 : 1, rank, epoch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_combine_push launch failed");
    return;
  }
  const long n = long(Tall) * (H / 8);
  k_pm_combine<<<unsigned((n + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids.data_ptr<int>(),
                                                             KTOP, int(H), int(E), tw.data_ptr<float>(),
                                                             reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()),
                                                             reinterpret_cast<__nv_bfloat16*>(part.data_ptr()), nv.data_ptr<int>(), W, int(R),
                                                             padded ? 0 : 1);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_combine launch failed");
}

void pm_layer(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
              torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
              torch::Tensor sh, torch::Tensor part, double limit, int64_t variant, bool padded, c10::optional<torch::Tensor> fvbuf) {
  pm_layer_impl(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, variant, padded, fvbuf, nullptr, 0,
                nullptr, 0);
}

// agrs RSF entry: bufs = every owner's RS receive buffer (bf16, >= W R H elements, rank order), flags = every owner's
// int32 flag word of this path, epoch = this rank's int32 epoch of this path, grid = CTAs (owners count W * grid signals).
void pm_layer_rs(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
                 torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
                 torch::Tensor sh, torch::Tensor part, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
                 torch::Tensor epoch, int64_t rank, int64_t grid, double limit, int64_t variant, bool padded,
                 c10::optional<torch::Tensor> fvbuf) {
  const int W = int(nv.numel());
  TORCH_CHECK(int(bufs.size()) == W && int(flags.size()) == W, "layer_rs: one receive buffer and one flag per rank");
  TORCH_CHECK(rank >= 0 && rank < W && grid >= W && grid % W == 0 && grid <= 65535, "layer_rs: rank / grid (a multiple of W)");
  RsPeers rp;
  for (int q = 0; q < W; ++q) {
    TORCH_CHECK(bufs[q].is_cuda() && bufs[q].is_contiguous() && bufs[q].dtype() == torch::kBFloat16 &&
                    bufs[q].numel() >= int64_t(W) * R * fall.size(1),
                "layer_rs: receive buffer ", q, " must be contiguous bf16 with >= W R H elements");
    TORCH_CHECK(flags[q].is_cuda() && flags[q].dtype() == torch::kInt32, "layer_rs: flag ", q, " int32 on device");
    rp.buf[q] = reinterpret_cast<__nv_bfloat16*>(bufs[q].data_ptr());
    rp.flag[q] = reinterpret_cast<unsigned*>(flags[q].data_ptr());
  }
  TORCH_CHECK(epoch.is_cuda() && epoch.dtype() == torch::kInt32 && epoch.device() == fall.device(), "layer_rs: epoch int32 on fall's device");
  pm_layer_impl(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, variant, padded, fvbuf, &rp, int(rank),
                reinterpret_cast<unsigned*>(epoch.data_ptr()), int(grid));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ds41h_prefill_bind(m);  // production's moe_prefill.cu bindings (moe_forward, set_hybrid, moe_gemm, moe_permute, ...)
  m.def("set_swiglu", &pm_set_swiglu, "the shared-expert swiglu launcher (pm_swiglu.launcher())");
  m.def("layer_rs", &pm_layer_rs, "agrs RSF: pm_layer with the combine storing into the owners' RS receive slots");
  m.def("layer", &pm_layer, "one rank's production-MoE layer: part <- shared + routed of the valid rows", py::arg("fall"), py::arg("gr"),
        py::arg("nv"), py::arg("R"), py::arg("vcount"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"), py::arg("sw13"),
        py::arg("sw2"), py::arg("gu"), py::arg("hsh"), py::arg("sh"), py::arg("part"), py::arg("limit") = 10.0, py::arg("variant") = 0xB9,
        py::arg("padded") = false, py::arg("fvbuf") = py::none());
}
