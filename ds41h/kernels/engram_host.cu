// ds41h Engram host offload: zero-copy gather of FP8 Engram table rows (+ ue8m0 scales) from pinned host
// memory over PCIe, dequantized to bf16 EXACTLY like stock vLLM 0.32.0 `_engram_lookup_kernel`
// (vllm/models/deepseek_v41/common/engram.py:639-721, non-packed branch):
//     out = bf16_rn( fp32(e4m3 value) * fp32_bits(scale_byte << 23) )
// with stock's TP head-shard ownership rule replicated per hash column (a row id outside the head's shard range
// [lo[h], hi[h]) gives zeros, like the rank that owns head h would write before the all-gather).
// Arithmetic is pinned with inline PTX (cvt.rn.f16x2.e4m3x2, cvt.f32.f16, mul.rn.f32 WITHOUT .ftz,
// cvt.rn.bf16x2.f32), so --use_fast_math (-ftz=true) of the ds41h build cannot change a single bit.
//
// Thread mapping: one table row (256 B of e4m3 + 8 B of scales) per group of LANES = 256/VEC lanes; every lane
// loads VEC bytes (VEC = 16: ld.v4, needs 16-B aligned rows; VEC = 8: ld.v2 for 8-B aligned rows such as the raw
// safetensors layout of layer 1). Each group keeps U rows in flight (all U loads issued before the first use) and
// the grid strides over rows, so a few fat CTAs saturate PCIe without taking many SMs from the main stream.
// Graph-capturable: no allocation, no host sync, fixed pointers, grid depends only on the token count.
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kDim = 256;    // engram_head_dim
constexpr int kQuant = 32;   // ue8m0 block (scale columns = kDim / kQuant = 8)
constexpr int kMaxCols = 32; // hash columns per layer ((max_ngram - 1) * n_heads = 24)

struct Params {
  const uint8_t* w;          // device-visible address of table row 0 (e4m3 values)
  const uint8_t* s;          // device-visible address of the scale bytes of row 0
  long long w_stride;        // bytes between consecutive rows (256 split layout, 272 packed)
  long long s_stride;        // bytes between consecutive scale rows (8 split, 272 packed)
  const int* ids;            // [T, ncols] int32 global row ids (arbitrary strides)
  long long ids_st, ids_sh;  // element strides (token, column)
  __nv_bfloat16* out;        // [T, ncols, kDim] contiguous
  int rows;                  // T * ncols
  int ncols;
  long long lo[kMaxCols];    // per hash column: owned id range [lo, hi) (stock TP head-shard rule)
  long long hi[kMaxCols];
};

template <int VEC>
struct Raw;
template <>
struct Raw<16> {
  uint32_t x[4];
};
template <>
struct Raw<8> {
  uint32_t x[2];
};

template <int VEC>
__device__ __forceinline__ void ld_row(Raw<VEC>& r, const uint8_t* p, uint64_t pol) {
  if constexpr (VEC == 16) {
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x[0]), "=r"(r.x[1]), "=r"(r.x[2]), "=r"(r.x[3])
        : "l"(p), "l"(pol));
  } else {
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
        : "=r"(r.x[0]), "=r"(r.x[1])
        : "l"(p), "l"(pol));
  }
}

__device__ __forceinline__ uint32_t ld_u8(const uint8_t* p, uint64_t pol) {
  uint16_t v;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.u8 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
  return v;
}

// two e4m3 (low byte = element 0) times an exact power-of-two fp32 scale -> packed bf16x2 (element 0 low).
__device__ __forceinline__ uint32_t dequant2(uint32_t pair16, float scale) {
  uint32_t h2;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(pair16));
  float f0, f1;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
      : "=f"(f0), "=f"(f1)
      : "r"(h2));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f0) : "f"(scale));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f1) : "f"(scale));
  uint32_t b2;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b2) : "f"(f1), "f"(f0));  // d = {hi: f1, lo: f0}
  return b2;
}

template <int VEC, int U>
__global__ void __launch_bounds__(1024) gather_kernel(const __grid_constant__ Params p) {
  constexpr int LANES = kDim / VEC;  // lanes per row
  static_assert(VEC == 16 || VEC == 8);
  static_assert((VEC * LANES) == kDim && (kQuant % VEC) == 0);
  const int lane = threadIdx.x % LANES;
  const int grp = blockIdx.x * (blockDim.x / LANES) + threadIdx.x / LANES;
  const int ngrp = gridDim.x * (blockDim.x / LANES);
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

  for (int base = 0; base < p.rows; base += ngrp * U) {
    long long idx[U];
    bool own[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int row = base + u * ngrp + grp;
      own[u] = false;
      idx[u] = 0;
      if (row < p.rows) {
        const int t = row / p.ncols, h = row - t * p.ncols;
        const long long i = __ldg(p.ids + t * p.ids_st + h * p.ids_sh);
        own[u] = i >= p.lo[h] && i < p.hi[h];
        idx[u] = i;
      }
    }
    Raw<VEC> v[U];
    uint32_t sb[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (own[u]) {
        ld_row<VEC>(v[u], p.w + idx[u] * p.w_stride + lane * VEC, pol);
        sb[u] = ld_u8(p.s + idx[u] * p.s_stride + (lane * VEC) / kQuant, pol);
      } else {
#pragma unroll
        for (int k = 0; k < VEC / 4; ++k) v[u].x[k] = 0;
        sb[u] = 0;
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int row = base + u * ngrp + grp;
      if (row >= p.rows) continue;
      uint32_t o[VEC / 2];
      if (own[u]) {
        const float scale = __uint_as_float(sb[u] << 23);
#pragma unroll
        for (int k = 0; k < VEC / 4; ++k) {
          o[2 * k] = dequant2(v[u].x[k] & 0xffffu, scale);
          o[2 * k + 1] = dequant2(v[u].x[k] >> 16, scale);
        }
      } else {
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) o[k] = 0;  // stock: rows of other shards / invalid ids -> 0
      }
      uint4* dst = reinterpret_cast<uint4*>(p.out + (long long)row * kDim + lane * VEC);
#pragma unroll
      for (int k = 0; k < VEC / 8; ++k) dst[k] = make_uint4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    }
  }
}

template <int VEC>
void launch_vec(const Params& p, int ctas, int threads, int unroll, cudaStream_t st) {
  switch (unroll) {
    case 1: gather_kernel<VEC, 1><<<ctas, threads, 0, st>>>(p); break;
    case 2: gather_kernel<VEC, 2><<<ctas, threads, 0, st>>>(p); break;
    case 4: gather_kernel<VEC, 4><<<ctas, threads, 0, st>>>(p); break;
    case 8: gather_kernel<VEC, 8><<<ctas, threads, 0, st>>>(p); break;
    default: TORCH_CHECK(false, "engram_host: unroll must be 1, 2, 4 or 8");
  }
}

// ids: int32 [T, ncols] CUDA tensor (any strides); out: bf16 [>=T, ncols, 256] contiguous CUDA tensor.
// w_ptr / s_ptr: device-visible addresses (cudaHostGetDevicePointer of a registered mapping, or device memory).
void gather(int64_t w_ptr, int64_t s_ptr, int64_t w_stride, int64_t s_stride, torch::Tensor ids, torch::Tensor out,
            std::vector<int64_t> lo, std::vector<int64_t> hi, int64_t ctas, int64_t threads, int64_t unroll,
            int64_t vec) {
  TORCH_CHECK(ids.is_cuda() && ids.dtype() == torch::kInt32 && ids.dim() == 2, "ids: int32 CUDA [T, ncols]");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.dim() == 3,
              "out: contiguous bf16 CUDA [T, ncols, 256]");
  const int64_t T = ids.size(0), ncols = ids.size(1);
  TORCH_CHECK(out.size(0) >= T && out.size(1) == ncols && out.size(2) == kDim, "out shape");
  TORCH_CHECK(ncols >= 1 && ncols <= kMaxCols && (int64_t)lo.size() == ncols && (int64_t)hi.size() == ncols,
              "ncols / ranges");
  TORCH_CHECK(T * ncols < (1ll << 31), "too many rows");
  TORCH_CHECK(vec == 16 || vec == 8, "vec must be 16 or 8");
  TORCH_CHECK(w_ptr % vec == 0 && w_stride % vec == 0, "weight rows must be vec-aligned");
  TORCH_CHECK(threads >= 32 && threads <= 1024 && threads % 32 == 0, "threads");
  if (T == 0) return;
  Params p{};
  p.w = reinterpret_cast<const uint8_t*>(w_ptr);
  p.s = reinterpret_cast<const uint8_t*>(s_ptr);
  p.w_stride = w_stride;
  p.s_stride = s_stride;
  p.ids = ids.data_ptr<int>();
  p.ids_st = ids.stride(0);
  p.ids_sh = ids.stride(1);
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.rows = static_cast<int>(T * ncols);
  p.ncols = static_cast<int>(ncols);
  for (int64_t h = 0; h < ncols; ++h) {
    p.lo[h] = lo[h];
    p.hi[h] = hi[h];
  }
  const int lanes = kDim / static_cast<int>(vec);
  const int64_t groups_per_cta = threads / lanes;
  const int64_t need = (p.rows + groups_per_cta * unroll - 1) / (groups_per_cta * unroll);
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(ctas, need)));
  auto st = c10::cuda::getCurrentCUDAStream(out.device().index()).stream();
  if (vec == 16) {
    launch_vec<16>(p, grid, static_cast<int>(threads), static_cast<int>(unroll), st);
  } else {
    launch_vec<8>(p, grid, static_cast<int>(threads), static_cast<int>(unroll), st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cudaHostRegister on an existing mapping (e.g. a MAP_SHARED mmap of a /dev/shm or hugetlbfs file) of the
// current process. Returns (cuda error code, message, device pointer). Error 0 = registered (or already registered
// by this process: cudaErrorHostMemoryAlreadyRegistered is mapped to success with the existing device pointer).
std::tuple<int64_t, std::string, int64_t> host_register(int64_t ptr, int64_t bytes, bool read_only) {
  unsigned flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
  if (read_only) flags |= cudaHostRegisterReadOnly;
  void* hp = reinterpret_cast<void*>(ptr);
  cudaError_t e = cudaHostRegister(hp, static_cast<size_t>(bytes), flags);
  if (e == cudaErrorHostMemoryAlreadyRegistered) {
    (void)cudaGetLastError();
    e = cudaSuccess;
  }
  if (e != cudaSuccess) {
    std::string msg = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return {static_cast<int64_t>(e), msg, 0};
  }
  void* dp = nullptr;
  e = cudaHostGetDevicePointer(&dp, hp, 0);
  if (e != cudaSuccess) {
    std::string msg = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return {static_cast<int64_t>(e), msg, 0};
  }
  return {0, "", reinterpret_cast<int64_t>(dp)};
}

std::tuple<int64_t, std::string> host_unregister(int64_t ptr) {
  cudaError_t e = cudaHostUnregister(reinterpret_cast<void*>(ptr));
  if (e != cudaSuccess) {
    std::string msg = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return {static_cast<int64_t>(e), msg};
  }
  return {0, ""};
}

// 1 if the GPU accesses registered host memory at its host address (UVA, no separate device pointer).
int64_t host_ptr_is_device_ptr() {
  int dev = 0, v = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&v, cudaDevAttrCanUseHostPointerForRegisteredMem, dev);
  return v;
}

// ---------------------------------------------------------------------------------------------- VMM host experiment
// Driver-allocated host memory through the VMM API (cuMemCreate with a HOST_NUMA / HOST location, 2 MiB granularity):
// does the GPU then map it with large PTEs (better random-row bandwidth than cudaHostRegister'ed 4 KiB mappings)?
// Bench-only for now (tests/test_engram_host.py --bench --backends vmm). Driver entry points are fetched at runtime
// (no -lcuda link dependency).
template <class F>
F drv(const char* name) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q{};
  const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &q);
  TORCH_CHECK(e == cudaSuccess && fn != nullptr && q == cudaDriverEntryPointSuccess, "driver symbol ", name, " unavailable");
  return reinterpret_cast<F>(fn);
}

struct VmmAlloc {
  CUmemGenericAllocationHandle handle;
  size_t size;
};
std::map<int64_t, VmmAlloc> vmm_allocs;

std::string cu_err(CUresult r) {
  const char* s = nullptr;
  auto get = drv<CUresult (*)(CUresult, const char**)>("cuGetErrorString");
  get(r, &s);
  return std::string(s ? s : "?") + " (" + std::to_string(int(r)) + ")";
}

// location: 0 = HOST_NUMA id 0, 1 = HOST; want_fd: request a POSIX-fd shareable handle. Returns (error or "", device
// address, mapped size, granularity, fd or -1, host_access 0/1).
std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t> vmm_host_alloc(int64_t bytes, int64_t location,
                                                                                       bool want_fd) {
  using Create = CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  using Gran = CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
  using Reserve = CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
  using Map = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
  using Access = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
  using Export = CUresult (*)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
  int dev = 0;
  cudaGetDevice(&dev);
  cudaFree(nullptr);  // make sure the primary context exists
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = location == 0 ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_HOST;
  prop.location.id = 0;
  prop.requestedHandleTypes = want_fd ? CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR : CU_MEM_HANDLE_TYPE_NONE;
  size_t gran = 0;
  CUresult r = drv<Gran>("cuMemGetAllocationGranularity")(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (r != CUDA_SUCCESS) return {"granularity: " + cu_err(r), 0, 0, 0, -1, 0};
  const size_t size = (size_t(bytes) + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h{};
  r = drv<Create>("cuMemCreate")(&h, size, &prop, 0);
  if (r != CUDA_SUCCESS) return {"cuMemCreate: " + cu_err(r), 0, 0, int64_t(gran), -1, 0};
  CUdeviceptr p = 0;
  r = drv<Reserve>("cuMemAddressReserve")(&p, size, std::max<size_t>(gran, 1ull << 30), 0, 0);
  if (r != CUDA_SUCCESS) return {"cuMemAddressReserve: " + cu_err(r), 0, 0, int64_t(gran), -1, 0};
  r = drv<Map>("cuMemMap")(p, size, 0, h, 0);
  if (r != CUDA_SUCCESS) return {"cuMemMap: " + cu_err(r), 0, 0, int64_t(gran), -1, 0};
  CUmemAccessDesc acc[2]{};
  acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc[0].location.id = dev;
  acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  acc[1].location.type = prop.location.type;
  acc[1].location.id = 0;
  acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  int64_t host_access = 1;
  r = drv<Access>("cuMemSetAccess")(p, size, acc, 2);
  if (r != CUDA_SUCCESS) {
    host_access = 0;
    r = drv<Access>("cuMemSetAccess")(p, size, acc, 1);
    if (r != CUDA_SUCCESS) return {"cuMemSetAccess: " + cu_err(r), 0, 0, int64_t(gran), -1, 0};
  }
  int64_t fd = -1;
  if (want_fd) {
    int f = -1;
    r = drv<Export>("cuMemExportToShareableHandle")(&f, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    fd = r == CUDA_SUCCESS ? f : -2;
  }
  vmm_allocs[int64_t(p)] = {h, size};
  return {"", int64_t(p), int64_t(size), int64_t(gran), fd, host_access};
}

void vmm_host_free(int64_t ptr) {
  auto it = vmm_allocs.find(ptr);
  if (it == vmm_allocs.end()) return;
  using Unmap = CUresult (*)(CUdeviceptr, size_t);
  using Release = CUresult (*)(CUmemGenericAllocationHandle);
  using Free = CUresult (*)(CUdeviceptr, size_t);
  cudaDeviceSynchronize();
  drv<Unmap>("cuMemUnmap")(CUdeviceptr(ptr), it->second.size);
  drv<Release>("cuMemRelease")(it->second.handle);
  drv<Free>("cuMemAddressFree")(CUdeviceptr(ptr), it->second.size);
  vmm_allocs.erase(it);
}

// strided copy between any two UVA addresses (cudaMemcpyDefault), on the current stream, synchronous.
void copy2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch, int64_t width, int64_t height) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void*>(dst), dpitch, reinterpret_cast<const void*>(src), spitch, width,
                                height, cudaMemcpyDefault, st) == cudaSuccess, "copy2d failed");
  TORCH_CHECK(cudaStreamSynchronize(st) == cudaSuccess);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vmm_host_alloc", &vmm_host_alloc, "cuMemCreate(HOST_NUMA|HOST) + map + access (bench experiment)");
  m.def("vmm_host_free", &vmm_host_free);
  m.def("copy2d", &copy2d);
  m.def("gather", &gather,
        "gather + dequantize Engram rows (stock _engram_lookup_kernel numerics) from host/device memory");
  m.def("host_register", &host_register, "cudaHostRegister(Mapped|Portable[|ReadOnly]) -> (err, msg, dev_ptr)");
  m.def("host_unregister", &host_unregister, "cudaHostUnregister -> (err, msg)");
  m.def("host_ptr_is_device_ptr", &host_ptr_is_device_ptr);
}
