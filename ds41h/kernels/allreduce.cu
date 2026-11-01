// ds41h ALLREDUCE: low-latency one-shot NVLink all-reduce / all-gather for TP=4 on H200 NVL
// (NV6 all-to-all between every pair, no NVSwitch / multicast).
//
// Ideas adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/phased_all_reduce.cu): raw cudaMalloc buffers shared via CUDA IPC over the TP CPU
// group, a device-resident epoch so CUDA-graph replays need no host arguments, and LL (flag-in-data)
// packets so a receiver needs no separate flag/fence round trip. The PCIe-motivated 3-step staggering and
// grid barrier of BRRR are dropped: every rank pushes to all 3 peers at once.
//
// Protocol (one call = one epoch, identical sequence of calls on every rank):
//  * Receive buffer of rank p: recv_p[parity 2][src RANKS][slot_lines] lines of 16 bytes.
//  * A line carries 8 payload bytes as two 64-bit words {payload_lo32 | tag << 32, payload_hi32 | tag << 32}.
//    Each 64-bit word is written / read with a single 64-bit access (st/ld.volatile.v2.u64 = two 64-bit
//    single-copy-atomic accesses), so a reader that sees tag == epoch in a word also sees its payload.
//  * Writer s stores line i of its input into recv_p[parity][s][i] of every peer p (push, NVLink stores).
//    Reader r polls its own recv_r[parity][s][i] (local L2) until both tags match, then clears the line
//    to zero. Tags are never 0, so cleared/stale lines can never match, including after 2^32 epochs.
//  * Parity double buffering: a rank can only write call n+2 (same parity as n) into peer p after it has
//    completed call n+1, which needs p's call n+1 pushes, which p issues only after its call-n kernel
//    (reads + clears) has completed (stream order; with PDL the griddepcontrol.wait below).
//  * Reduction order is fixed (src rank 0, 1, 2, 3 in fp32, one bf16 rounding), so every rank produces
//    bit-identical outputs (TP-consistent), unlike the stock FlashInfer path.
//  * The epoch is read by every CTA at kernel start and advanced once by the last CTA to finish; the next
//    call reads it only after this grid completed (stream order / griddepcontrol.wait).
//
// No CTA waits on another CTA of the same grid, so there is no co-residency requirement.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "common.cuh"
#include "host.cuh"

#include <vector>

namespace {

constexpr int THREADS = 512;
constexpr int MAX_CTAS = 132;

struct Comm {
  ulonglong2* recv[RANKS];  // base of each rank's receive buffer (own + peer-mapped)
  unsigned* state;          // [0] epoch, [1] CTAs arrived (local device memory)
  long slot_lines;          // capacity of one (parity, src) slot in lines
  int rank;
};

std::vector<Comm*> comms;

struct Args {
  ulonglong2* recv[RANKS];
  unsigned* state;
  long slot_lines;
  int rank;
  long lines;                   // packets to exchange (8 payload bytes each)
  unsigned long long timeout;   // ns, 0 = wait forever
  int early_dep;                // all_gather_early: trigger the PDL dependent right after the PDL wait
};

__device__ __forceinline__ void st_line(ulonglong2* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ ulonglong2 ld_line(const ulonglong2* p) {
  ulonglong2 v;
  asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long pack_word(unsigned payload, unsigned tag) {
  return (static_cast<unsigned long long>(tag) << 32) | payload;
}

// Spin until both words of the line carry `tag`; returns the two payload words.
__device__ __forceinline__ uint2 poll_line(const ulonglong2* p, unsigned tag, unsigned long long timeout) {
  ulonglong2 v = ld_line(p);
  if (unsigned(v.x >> 32) != tag || unsigned(v.y >> 32) != tag) {
    const unsigned long long t0 = timeout ? globaltimer() : 0ull;
    unsigned spins = 0;
    do {
      if (++spins > 32u) __nanosleep(64);  // back off after a short hot spin (less power while waiting for a slow rank)
      v = ld_line(p);
      if (timeout && (spins & 255u) == 0 && globaltimer() - t0 > timeout) {
        printf("ds41h allreduce: timeout waiting for tag %u at %p (saw %u/%u)\n", tag, p, unsigned(v.x >> 32), unsigned(v.y >> 32));
        __trap();
      }
    } while (unsigned(v.x >> 32) != tag || unsigned(v.y >> 32) != tag);
  }
  return make_uint2(unsigned(v.x), unsigned(v.y));
}

// Exact IEEE arithmetic independent of --use_fast_math (which would add .ftz and flush bf16 subnormals).
__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ unsigned pack_bf16_rn(float lo, float hi) {
  unsigned d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));  // d = {hi:upper, lo:lower}
  return d;
}

// a.recv[p] for a runtime p without dynamically indexing the parameter array (which would go through
// a local-memory copy): static selects only.
__device__ __forceinline__ ulonglong2* recv_of(const Args& a, int p) {
  ulonglong2* r = a.recv[0];
#pragma unroll
  for (int q = 1; q < RANKS; ++q) r = p == q ? a.recv[q] : r;
  return r;
}

__device__ __forceinline__ unsigned begin_epoch(const Args& a) {
  const unsigned e = ld_volatile(a.state);
  const unsigned tag = e + 1u;
  return tag == 0u ? 2u : tag;  // never 0 (cleared lines), parity keeps alternating across the wrap
}

__device__ __forceinline__ void finish_epoch(const Args& a, unsigned tag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(a.state + 1, 1u) == gridDim.x - 1) {
      atomicExch(a.state + 1, 0u);
      __threadfence();
      atomicExch(a.state, tag);
    }
  }
}

// out = sum over ranks of in (bf16, fixed rank order). in/out: `lines` x 8 bytes (4 bf16), may alias.
__global__ void __launch_bounds__(THREADS) ll_all_reduce_kernel(const Args a, const uint2* in, uint2* out) {
  pdl_wait();
  const unsigned tag = begin_epoch(a);
  const long par = tag & 1u;
  const long gtid = long(blockIdx.x) * THREADS + threadIdx.x, gstride = long(gridDim.x) * THREADS;
  ulonglong2* dst[RANKS - 1];  // peers rank+1, rank+2, rank+3 (static indices only)
#pragma unroll
  for (int s = 1; s < RANKS; ++s) dst[s - 1] = recv_of(a, (a.rank + s) % RANKS) + (par * RANKS + a.rank) * a.slot_lines;
  for (long i = gtid; i < a.lines; i += gstride) {
    const uint2 v = in[i];
    const unsigned long long w0 = pack_word(v.x, tag), w1 = pack_word(v.y, tag);
#pragma unroll
    for (int s = 1; s < RANKS; ++s) st_line(dst[s - 1] + i, w0, w1);
  }
  ulonglong2* box = recv_of(a, a.rank) + par * RANKS * a.slot_lines;
  for (long i = gtid; i < a.lines; i += gstride) {
    // ((x0 + x1) + x2) + x3 in fp32, seeded with x0 (not +0.0, which would turn an all -0.0 sum into +0.0).
    float acc[4];
#pragma unroll
    for (int r = 0; r < RANKS; ++r) {
      uint2 v;
      if (r == a.rank) {
        v = in[i];
      } else {
        ulonglong2* line = box + r * a.slot_lines + i;
        v = poll_line(line, tag, a.timeout);
        st_line(line, 0ull, 0ull);
      }
      const float2 lo = unpack_bf16(v.x), hi = unpack_bf16(v.y);
      if (r == 0) {
        acc[0] = lo.x, acc[1] = lo.y, acc[2] = hi.x, acc[3] = hi.y;
      } else {
        acc[0] = add_rn(acc[0], lo.x), acc[1] = add_rn(acc[1], lo.y), acc[2] = add_rn(acc[2], hi.x), acc[3] = add_rn(acc[3], hi.y);
      }
    }
    out[i] = make_uint2(pack_bf16_rn(acc[0], acc[1]), pack_bf16_rn(acc[2], acc[3]));
  }
  finish_epoch(a, tag);
}

// All-gather of 32-bit words: out[(j / inner) * RANKS * inner + r * inner + j % inner] = in_r[j].
// in/out must not alias.
__global__ void __launch_bounds__(THREADS)
    ll_all_gather_kernel(const Args a, const unsigned* __restrict__ in, unsigned* __restrict__ out, long words, long inner) {
  pdl_wait();
  // all_gather_early (verify chain): the dependent (PDL-launched Triton residual / reject kernel) is released now so it
  // prefetches its gather-independent inputs while the exchange runs; it reads `out` only after its own griddepcontrol.wait
  // (= this grid's completion). Every CTA of this grid is resident when the trigger fires (no co-residency hazard).
  if (a.early_dep) pdl_launch_dependents();
  const unsigned tag = begin_epoch(a);
  const long par = tag & 1u;
  const long gtid = long(blockIdx.x) * THREADS + threadIdx.x, gstride = long(gridDim.x) * THREADS;
  ulonglong2* dst[RANKS - 1];  // peers rank+1, rank+2, rank+3 (static indices only)
#pragma unroll
  for (int s = 1; s < RANKS; ++s) dst[s - 1] = recv_of(a, (a.rank + s) % RANKS) + (par * RANKS + a.rank) * a.slot_lines;
  auto out_index = [&](int r, long j) { return (j / inner) * RANKS * inner + r * inner + j % inner; };
  for (long i = gtid; i < a.lines; i += gstride) {
    const long j = 2 * i;
    const unsigned x = in[j], y = j + 1 < words ? in[j + 1] : 0u;
    const unsigned long long w0 = pack_word(x, tag), w1 = pack_word(y, tag);
#pragma unroll
    for (int s = 1; s < RANKS; ++s) st_line(dst[s - 1] + i, w0, w1);
    out[out_index(a.rank, j)] = x;
    if (j + 1 < words) out[out_index(a.rank, j + 1)] = y;
  }
  ulonglong2* box = recv_of(a, a.rank) + par * RANKS * a.slot_lines;
  for (long i = gtid; i < a.lines; i += gstride) {
    const long j = 2 * i;
#pragma unroll
    for (int s = 1; s < RANKS; ++s) {
      const int r = (a.rank + s) % RANKS;
      ulonglong2* line = box + r * a.slot_lines + i;
      const uint2 v = poll_line(line, tag, a.timeout);
      st_line(line, 0ull, 0ull);
      out[out_index(r, j)] = v.x;
      if (j + 1 < words) out[out_index(r, j + 1)] = v.y;
    }
  }
  finish_epoch(a, tag);
}

Args make_args(const Comm& c, long lines, int64_t timeout_ns) {
  Args a{};
  for (int p = 0; p < RANKS; ++p) a.recv[p] = c.recv[p];
  a.state = c.state;
  a.slot_lines = c.slot_lines;
  a.rank = c.rank;
  a.lines = lines;
  a.timeout = static_cast<unsigned long long>(timeout_ns);
  return a;
}

template <class... Params, class... Ts>
void launch(bool pdl, void (*kernel)(Params...), int ctas, Ts... args) {
  if (pdl) {
    launch_pdl("ds41h_allreduce", kernel, dim3(ctas), dim3(THREADS), 0, args...);
  } else {
    kernel<<<ctas, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

Comm& get(int64_t handle) {
  TORCH_CHECK(handle >= 0 && handle < int64_t(comms.size()) && comms[handle] != nullptr, "invalid ds41h allreduce handle");
  return *comms[handle];
}

}  // namespace

int64_t ranks() { return RANKS; }
int64_t threads() { return THREADS; }
int64_t max_ctas() { return MAX_CTAS; }
int64_t buffer_bytes(int64_t slot_lines) { return 2LL * RANKS * slot_lines * 16; }

// buffers: base pointer of every rank's receive buffer as mapped in this process (own one included).
int64_t create(std::vector<int64_t> buffers, int64_t rank, int64_t slot_lines) {
  TORCH_CHECK(int64_t(buffers.size()) == RANKS && rank >= 0 && rank < RANKS && slot_lines > 0);
  auto* c = new Comm{};
  for (int p = 0; p < RANKS; ++p) {
    c->recv[p] = reinterpret_cast<ulonglong2*>(buffers[p]);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(c->recv[p]) % 16 == 0, "receive buffers must be 16-byte aligned");
  }
  c->state = zeroed<unsigned>(2);
  c->slot_lines = slot_lines;
  c->rank = int(rank);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  comms.push_back(c);
  return int64_t(comms.size()) - 1;
}

void destroy(int64_t handle) {
  Comm& c = get(handle);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  TORCH_CHECK(cudaFree(c.state) == cudaSuccess);
  delete comms[handle];
  comms[handle] = nullptr;
}

void close_ipc(int64_t ptr) { TORCH_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)) == cudaSuccess); }
void free_buffer(int64_t ptr) { TORCH_CHECK(cudaFree(reinterpret_cast<void*>(ptr)) == cudaSuccess); }

void all_reduce(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t ctas, bool pdl, int64_t timeout_ns) {
  const Comm& c = get(handle);
  TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.is_contiguous() && out.is_contiguous(), "contiguous CUDA tensors required");
  TORCH_CHECK(in.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && in.numel() == out.numel(), "bf16 in/out of equal size");
  const long bytes = in.numel() * 2;
  TORCH_CHECK(bytes % 8 == 0 && bytes / 8 <= c.slot_lines, "size must be a multiple of 8 bytes and fit the buffer");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0, "8-byte alignment");
  TORCH_CHECK(ctas >= 1 && ctas <= MAX_CTAS);
  if (bytes == 0) return;
  const c10::cuda::CUDAGuard guard(in.device());
  launch(pdl, ll_all_reduce_kernel, int(ctas), make_args(c, bytes / 8, timeout_ns), static_cast<const uint2*>(in.data_ptr()),
         static_cast<uint2*>(out.data_ptr()));
}

static void all_gather_impl(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t inner_words, int64_t ctas, bool pdl,
                            int64_t timeout_ns, bool early) {
  const Comm& c = get(handle);
  TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.is_contiguous() && out.is_contiguous(), "contiguous CUDA tensors required");
  TORCH_CHECK(out.nbytes() == size_t(RANKS) * in.nbytes(), "out must hold RANKS x in");
  const long bytes = long(in.nbytes());
  TORCH_CHECK(bytes % 4 == 0 && inner_words > 0 && (bytes / 4) % inner_words == 0, "word-granular sizes required");
  const long words = bytes / 4, lines = (words + 1) / 2;
  TORCH_CHECK(lines <= c.slot_lines, "all_gather input does not fit the buffer");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0, "4-byte alignment");
  TORCH_CHECK(ctas >= 1 && ctas <= MAX_CTAS);
  if (words == 0) return;
  const c10::cuda::CUDAGuard guard(in.device());
  Args args = make_args(c, lines, timeout_ns);
  args.early_dep = (early && pdl) ? 1 : 0;
  launch(pdl, ll_all_gather_kernel, int(ctas), args, static_cast<const unsigned*>(in.data_ptr()), static_cast<unsigned*>(out.data_ptr()),
         words, long(inner_words));
}

void all_gather(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t inner_words, int64_t ctas, bool pdl, int64_t timeout_ns) {
  all_gather_impl(handle, in, out, inner_words, ctas, pdl, timeout_ns, false);
}

// all_gather with the early PDL trigger (the next kernel must be PDL-launched and griddepcontrol.wait before reading `out`)
void all_gather_early(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t inner_words, int64_t ctas, bool pdl, int64_t timeout_ns) {
  all_gather_impl(handle, in, out, inner_words, ctas, pdl, timeout_ns, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ranks", &ranks);
  m.def("threads", &threads);
  m.def("max_ctas", &max_ctas);
  m.def("buffer_bytes", &buffer_bytes);
  m.def("alloc_ipc", &alloc_ipc, "cudaMalloc + zero a buffer; returns [ptr, cudaIpcMemHandle bytes...]");
  m.def("open_ipc", &open_ipc, "open a peer cudaIpcMemHandle; returns the mapped pointer");
  m.def("close_ipc", &close_ipc);
  m.def("free_buffer", &free_buffer);
  m.def("create", &create);
  m.def("destroy", &destroy);
  m.def("all_reduce", &all_reduce, "one-shot LL all-reduce of a bf16 tensor (fixed rank order)");
  m.def("all_gather", &all_gather, "one-shot LL all-gather of 32-bit words into the concatenated layout");
  m.def("all_gather_early", &all_gather_early, "all_gather releasing its PDL dependent right after its own PDL wait");
}
