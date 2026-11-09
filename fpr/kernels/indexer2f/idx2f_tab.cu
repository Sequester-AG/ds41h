// indexer2f: device-side per-step tables of the indexer / compressor hook (engine/hooks_idx.py; tables2f.py).
// The host writes one small int32 staging block per step (paged slots, per-sequence records, block-table deltas) into a
// pinned ring buffer; run() issues ONE cudaMemcpyAsync of it and two kernels on the current stream:
//   scatter_kernel  block-table deltas -> the rank's persistent per-state-slot table dbt [n_state, ld]
//   derive_kernel   decode: seq_lens (ratio 1 / 2), compressed-record slots (ratio 1 / 2), ring slots, valid,
//                   per-request state slot; general (mixed steps, pseudo-requests of 4 rows of one sequence): the same
//                   per row / pseudo-request plus ir2row, valid_ir, vir / vrow, row2ir, qsl, t2r, nb_lens, and the
//                   pseudo-requests' block-table rows gathered from dbt.
// Row positions come from the engine's device buffer (s["pos"], int64 [R], uploaded by its own _upload), so the
// tables need no host copy of them. No host synchronization anywhere: the event of a ring slot (host block + device
// block, recorded after the kernels) is waited on only before that slot is rewritten (RING calls later).
//
// Staging layout (int32 words, offsets in this order; tables2f.layout mirrors it):
//   ps [n] paged slot (page * 64 + pos % 64, -1 = no write) | seqslot [n] | sslot [S] state slot | row0 [S] |
//   nrows [S] | pr0 [S + 1] first pseudo-request of each sequence | crow [S + 1] rows before each sequence (seq order) |
//   uk [U] delta's sequence | us [U] first block | ul [U] blocks | uo [U] offset into pages | pages [P]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>

namespace idx2f_tab {

struct Ptrs {
  const int64_t* pos;
  int32_t* dbt;
  int64_t ld;          // dbt / bt row length (P_MAX)
  int n_state;
  int R;
  // decode + general
  int32_t* sl1;
  int32_t* sl2;
  int64_t* cs1;
  int64_t* cs2;
  int64_t* rs;
  bool* valid;         // decode: valid [R]; general: valid_ir [IR]
  int64_t* preq;       // decode: preq_slot [B]
  // general only
  int32_t* t2r;
  int32_t* qsl;
  int64_t* ir2row_c;
  int64_t* vir;
  int64_t* vrow;
  int64_t* row2ir;
  int32_t* nb_lens;
  int32_t* bt;
  int general;
};

struct Off {
  int ps, seqslot, sslot, row0, nrows, pr0, crow, uk, us, ul, uo, pages, total;
};

static Off layout(int n, int S, int U, int P) {
  Off o;
  int x = 0;
  o.ps = x; x += n;
  o.seqslot = x; x += n;
  o.sslot = x; x += S;
  o.row0 = x; x += S;
  o.nrows = x; x += S;
  o.pr0 = x; x += S + 1;
  o.crow = x; x += S + 1;
  o.uk = x; x += U;
  o.us = x; x += U;
  o.ul = x; x += U;
  o.uo = x; x += U;
  o.pages = x; x += P;
  o.total = x;
  return o;
}

struct Handle {
  Ptrs p;
  int device;
};

struct Ring {               // the staging ring of one device: one event per pinned host block
  int device;
  std::vector<cudaEvent_t> ev;
};

static std::vector<Handle>& handles() {
  static std::vector<Handle> h;
  return h;
}

static std::vector<Ring>& rings() {
  static std::vector<Ring> r;
  return r;
}

__global__ void scatter_kernel(Ptrs p, const int32_t* __restrict__ st, Off o) {
  const int u = blockIdx.x;
  const int k = st[o.uk + u];
  const int64_t row = st[o.sslot + k];
  const int64_t b0 = st[o.us + u];
  const int len = st[o.ul + u];
  const int off = st[o.uo + u];
  if (row < 0 || row >= p.n_state) return;
  int32_t* dst = p.dbt + row * p.ld + b0;
  const int32_t* src = st + o.pages + off;
  for (int i = threadIdx.x; i < len; i += blockDim.x)
    if (b0 + i < p.ld) dst[i] = src[i];
}

__device__ __forceinline__ int seq_of(const int32_t* __restrict__ pr0, int S, int b) {
  // largest s in [0, S) with pr0[s] <= b (pr0 ascending, pr0[S] = B)
  int lo = 0, hi = S - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (pr0[mid] <= b) lo = mid; else hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ void slots(int64_t ps, int64_t& c1, int64_t& c2) {
  if (ps >= 0) {
    const int64_t pg = ps >> 6, off = ps & 63;
    c1 = pg * 64 + off;
    c2 = pg * 32 + (off >> 1);
  } else {
    c1 = -1;
    c2 = -1;
  }
}

// decode: rows r in [0, R) (4 rows per sequence, sequence-contiguous; B = R / 4)
__global__ void decode_kernel(Ptrs p, const int32_t* __restrict__ st, Off o, int n, int S) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= p.R) return;
  if (r < n) {
    const int64_t pos = p.pos[r];
    p.sl1[r] = (int32_t)(pos + 1);
    p.sl2[r] = (int32_t)((pos + 1) / 2);
    int64_t c1, c2;
    slots(st[o.ps + r], c1, c2);
    p.cs1[r] = c1;
    p.cs2[r] = c2;
    p.rs[r] = (int64_t)st[o.sslot + r / 4] * 8 + pos % 8;
    p.valid[r] = true;
  } else {
    p.sl1[r] = 0;
    p.sl2[r] = 0;
    p.cs1[r] = -1;
    p.cs2[r] = -1;
    p.rs[r] = -1;
    p.valid[r] = false;
  }
  if ((r & 3) == 0) {
    const int b = r >> 2;
    p.preq[b] = b < S ? (int64_t)st[o.sslot + b] : (int64_t)(p.n_state - 1);
  }
}

// general: blocks [0, nb_ir) pseudo-request rows, [nb_ir, nb_ir + nb_row) bucket rows, then 1 block qsl, then B blocks
// of block-table rows
__global__ void general_kernel(Ptrs p, const int32_t* __restrict__ st, Off o, int n, int S, int B, int nb_ir,
                               int nb_row) {
  const int IR = 4 * B;
  const int32_t* pr0 = st + o.pr0;
  int blk = blockIdx.x;
  if (blk < nb_ir) {
    const int ir = blk * blockDim.x + threadIdx.x;
    if (ir >= IR) return;
    const int b = ir >> 2, j = ir & 3;
    const int s = seq_of(pr0, S, b);
    const int c0 = (b - pr0[s]) * 4, m = st[o.nrows + s], r0 = st[o.row0 + s];
    const int t = c0 + j;
    int64_t pos;
    if (t < m) {
      const int row = r0 + t;
      pos = p.pos[row];
      p.ir2row_c[ir] = row;
      p.valid[ir] = true;
      const int k = st[o.crow + s] + t;
      p.vir[k] = ir;
      p.vrow[k] = row;
      p.row2ir[row] = ir;
    } else {   // padding row of the group: the group's last real row's length (DeepGEMM next_n = 4 semantics)
      pos = p.pos[r0 + min(c0 + 4, m) - 1];
      p.ir2row_c[ir] = 0;
      p.valid[ir] = false;
    }
    const int32_t l1 = (int32_t)(pos + 1);
    p.sl1[ir] = l1;
    p.sl2[ir] = (int32_t)((pos + 1) / 2);
    p.nb_lens[ir] = (l1 + 7) / 8;
    return;
  }
  blk -= nb_ir;
  if (blk < nb_row) {
    const int row = blk * blockDim.x + threadIdx.x;
    if (row >= p.R) return;
    if (row < n) {
      const int s = st[o.seqslot + row];
      const int64_t pos = p.pos[row];
      int64_t c1, c2;
      slots(st[o.ps + row], c1, c2);
      p.cs1[row] = c1;
      p.cs2[row] = c2;
      p.rs[row] = (int64_t)st[o.sslot + s] * 8 + pos % 8;
      p.t2r[row] = s;
    } else {
      p.cs1[row] = -1;
      p.cs2[row] = -1;
      p.rs[row] = -1;
      p.t2r[row] = S > 0 ? S - 1 : 0;
      p.row2ir[row] = -1;
    }
    return;
  }
  blk -= nb_row;
  if (blk == 0) {
    if (threadIdx.x == 0) {
      p.qsl[0] = 0;
      if (S == 0) p.qsl[1] = 0;
    }
    for (int s = threadIdx.x; s < S; s += blockDim.x) p.qsl[s + 1] = st[o.row0 + s] + st[o.nrows + s];
    return;
  }
  const int b = blk - 1;
  if (b >= B) return;
  const int s = seq_of(pr0, S, b);
  const int64_t slot = st[o.sslot + s];
  const int4* src = reinterpret_cast<const int4*>(p.dbt + slot * p.ld);
  int4* dst = reinterpret_cast<int4*>(p.bt + (int64_t)b * p.ld);
  for (int64_t i = threadIdx.x; i < p.ld / 4; i += blockDim.x) dst[i] = src[i];
}

template <typename T>
static T* ptr(const torch::Tensor& t) {
  return t.defined() && t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}

// tensors (decode): pos, dbt, sl1, sl2, cs1, cs2, rs, valid, preq
// tensors (general): pos, dbt, sl1, sl2, cs1, cs2, rs, valid_ir, t2r, qsl, ir2row_c, vir, vrow, row2ir, nb_lens, bt
int64_t create(std::vector<torch::Tensor> t, int64_t general) {
  Ptrs p{};
  TORCH_CHECK(t[0].scalar_type() == torch::kInt64 && t[1].scalar_type() == torch::kInt32 && t[1].dim() == 2);
  p.pos = ptr<int64_t>(t[0]);
  p.dbt = ptr<int32_t>(t[1]);
  p.ld = t[1].size(1);
  p.n_state = (int)t[1].size(0);
  p.R = (int)t[0].numel();
  p.sl1 = ptr<int32_t>(t[2]);
  p.sl2 = ptr<int32_t>(t[3]);
  p.cs1 = ptr<int64_t>(t[4]);
  p.cs2 = ptr<int64_t>(t[5]);
  p.rs = ptr<int64_t>(t[6]);
  p.valid = ptr<bool>(t[7]);
  p.general = (int)general;
  for (auto& x : t) TORCH_CHECK(!x.defined() || x.is_contiguous(), "idx2f_tab: contiguous tensors only");
  if (general) {
    TORCH_CHECK(t.size() == 16);
    p.t2r = ptr<int32_t>(t[8]);
    p.qsl = ptr<int32_t>(t[9]);
    p.ir2row_c = ptr<int64_t>(t[10]);
    p.vir = ptr<int64_t>(t[11]);
    p.vrow = ptr<int64_t>(t[12]);
    p.row2ir = ptr<int64_t>(t[13]);
    p.nb_lens = ptr<int32_t>(t[14]);
    p.bt = ptr<int32_t>(t[15]);
    TORCH_CHECK(t[15].size(1) == p.ld && p.ld % 4 == 0);
  } else {
    TORCH_CHECK(t.size() == 9);
    p.preq = ptr<int64_t>(t[8]);
  }
  Handle h;
  h.p = p;
  h.device = t[0].get_device();
  handles().push_back(h);
  return (int64_t)handles().size() - 1;
}

int64_t ring(int64_t device, int64_t k) {
  Ring r;
  r.device = (int)device;
  c10::cuda::CUDAGuard g(r.device);
  r.ev.resize(k);
  for (auto& e : r.ev) {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(e, at::cuda::getCurrentCUDAStream()));
  }
  rings().push_back(r);
  return (int64_t)rings().size() - 1;
}

void wait(int64_t rid, int64_t slot) {
  C10_CUDA_CHECK(cudaEventSynchronize(rings()[rid].ev[slot]));   // returns at once unless the slot's copy is pending
}

// one H2D of the staging words + scatter + derive on the current stream; records the ring slot's event after them
void run(int64_t hid, int64_t rid, int64_t slot, int64_t host_ptr, int64_t dev_ptr, int64_t n, int64_t S, int64_t U,
         int64_t P, int64_t B) {
  Handle& h = handles()[hid];
  TORCH_CHECK(rings()[rid].device == h.device, "idx2f_tab: ring of another device");
  c10::cuda::CUDAGuard g(h.device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Off o = layout((int)n, (int)S, (int)U, (int)P);
  int32_t* dst = reinterpret_cast<int32_t*>(dev_ptr);
  if (o.total)
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(host_ptr), (size_t)o.total * 4,
                                   cudaMemcpyHostToDevice, stream));
  if (U > 0) scatter_kernel<<<(unsigned)U, 256, 0, stream>>>(h.p, dst, o);
  const int R = h.p.R;
  if (!h.p.general) {
    decode_kernel<<<(R + 255) / 256, 256, 0, stream>>>(h.p, dst, o, (int)n, (int)S);
  } else {
    const int nb_ir = (int)((4 * B + 255) / 256), nb_row = (R + 255) / 256;
    general_kernel<<<(unsigned)(nb_ir + nb_row + 1 + B), 256, 0, stream>>>(h.p, dst, o, (int)n, (int)S, (int)B, nb_ir,
                                                                            nb_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // after the kernels: the slot's host block AND device block are free again once this event completes
  C10_CUDA_CHECK(cudaEventRecord(rings()[rid].ev[slot], stream));
}

}  // namespace idx2f_tab

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create", &idx2f_tab::create);
  m.def("ring", &idx2f_tab::ring);
  m.def("wait", &idx2f_tab::wait);
  m.def("run", &idx2f_tab::run);
}
