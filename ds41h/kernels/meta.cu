// ds41h meta: batched attention-metadata kernels for the decode steps (sm90a).
//
// swa_batch_kernel: the decode part of vLLM 0.32.0 DeepseekSparseSWAMetadataBuilder for SEVERAL KV-cache groups in one
// launch. Per group (job) it reproduces, bit for bit,
//   build() (causal, decode-only; v1/attention/backends/mla/sparse_swa.py:575-680):
//     torch.ge(slot_mapping, 0, out=is_valid_token)                 (is_valid_token[: slot_mapping.shape[0]])
//     decode_swa_lens[num_decode_tokens:] = 0                       (the whole tail of the builder buffer)
//     _compute_swa_indices_and_lens_kernel(HAS_IMAGE=False, token_offset=0) over num_decode_tokens tokens
//       (window_size, index_width = decode_swa_indices.shape[-1], replay_start)
//   update_draft_decode_metadata() (DSpark non-causal draft block, :829-874):
//     torch.ge(metadata.slot_mapping, 0, out=metadata.is_valid_token)
//     compute_token_to_req_indices(query_start_loc, token_to_req_indices, num_tokens, num_tokens)  (binary search,
//       v1/attention/ops/metadata.py:10-30)
//     ComputeDSparkNoncausalSWAIndicesKernel over num_tokens tokens (index_width = noncausal width)
// Stock runs 3 launches per group (8 target SWA groups per decode step: 24 launches, ~66 us at 350 W incl. gaps).
//
// Grid: (row blocks + fill blocks, jobs). Row block t < ntok of job j: token t's lens / index row (+ token->request
// map for non-causal jobs). Fill blocks: is_valid_token[i] = slot[i] >= 0 for i < n_slot and (causal jobs) the
// lens tail zeroing. The token rows compute their own validity from slot_mapping (same value the fill blocks
// store), so no intra-kernel ordering is needed. Jobs must not alias each other's outputs (checked on the host;
// the non-causal jobs of one draft step share token_to_req_indices and write identical values into it).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int MAXJOBS = 16;
constexpr int T = 256;
constexpr int FILL_PER_BLOCK = T * 4;

struct Job {
  const long long* slot;   // slot_mapping int64 [n_slot]
  bool* valid;             // is_valid_token [>= max(n_slot, ntok)]
  int* lens;               // decode_swa_lens [lens_len]
  int* idx;                // swa indices rows [ntok][idx_stride] (int32)
  const int* qsl;          // query_start_loc int32 [num_reqs + 1]
  const int* seq_lens;     // int32 [num_reqs]
  int* t2r;                // token_to_req_indices int32 (causal: read; non-causal: written)
  const int* bt;           // block table int32 [*, bt_stride]
  const int* replay;       // replay_start int32 [num_reqs] (causal only)
  long long idx_stride, bt_stride;
  int n_slot, ntok, lens_len, width, window, block_size, num_reqs, noncausal, fill_blocks;
};

struct Args {
  Job job[MAXJOBS];
  int njobs, max_ntok;
};

// v1/attention/ops/metadata.py _token_request: binary search over query_start_loc (same probing sequence)
__device__ __forceinline__ int token_request(const int* qsl, int token, int num_reqs) {
  int lo = 0, hi = num_reqs;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const int end = mid < num_reqs ? qsl[mid + 1] : 0x7FFFFFFF;
    const bool right = end <= token;
    if (right) lo = mid + 1;
    else hi = mid;
  }
  return min(lo, num_reqs - 1);
}

__global__ void __launch_bounds__(T) swa_batch_kernel(const __grid_constant__ Args a) {
  const int j = blockIdx.y;
  if (j >= a.njobs) return;
  const Job& b = a.job[j];
  const int x = blockIdx.x;
  if (x < a.max_ntok) {
    const int pid = x;
    if (pid >= b.ntok) return;
    __shared__ int sh[4];  // start_pos, swa_len, req, valid
    if (threadIdx.x == 0) {
      const bool valid = pid < b.n_slot ? b.slot[pid] >= 0 : b.valid[pid];
      int req = 0, start = 0, len = 0;
      if (b.noncausal) {
        req = token_request(b.qsl, pid, b.num_reqs);
        b.t2r[pid] = req;
      }
      if (valid) {
        if (!b.noncausal) req = b.t2r[pid];
        const int qs = b.qsl[req], qe = b.qsl[req + 1];
        const int qlen = qe - qs;
        const int seq_len = b.seq_lens[req];
        const int prefix = seq_len - qlen;
        int end;
        if (b.noncausal) {
          start = max(prefix - b.window, 0);
          end = seq_len;
        } else {
          const int pos = prefix + pid - qs;
          start = max(pos - (b.window - 1), 0);  // left_add = 0 (no image)
          start = max(start, b.replay[req]);
          end = pos + 1;
        }
        len = end - start;
      }
      b.lens[pid] = valid ? len : 0;
      sh[0] = start, sh[1] = len, sh[2] = req, sh[3] = valid ? 1 : 0;
    }
    __syncthreads();
    const int start = sh[0], len = sh[1], req = sh[2], valid = sh[3];
    int* row = b.idx + (long long)pid * b.idx_stride;
    const int* btr = b.bt + (long long)req * b.bt_stride;
    for (int o = threadIdx.x; o < b.width; o += T) {
      int v = -1;
      if (valid && o < len) {
        const int p = start + o;
        v = btr[p / b.block_size] * b.block_size + p % b.block_size;
      }
      row[o] = v;
    }
    return;
  }
  const int f = x - a.max_ntok;
  if (f >= b.fill_blocks) return;
  const int base = f * FILL_PER_BLOCK;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int i = base + u * T + threadIdx.x;
    if (i < b.n_slot) b.valid[i] = b.slot[i] >= 0;
    if (!b.noncausal && i >= b.ntok && i < b.lens_len) b.lens[i] = 0;
  }
}

}  // namespace

// jobs: list of (tensors..., ints...) packed by ds41h/patch_meta.py
void swa_batch(std::vector<torch::Tensor> slot, std::vector<torch::Tensor> valid, std::vector<torch::Tensor> lens,
               std::vector<torch::Tensor> idx, std::vector<torch::Tensor> qsl, std::vector<torch::Tensor> seq_lens,
               std::vector<torch::Tensor> t2r, std::vector<torch::Tensor> bt, std::vector<std::optional<torch::Tensor>> replay,
               std::vector<int64_t> ntok, std::vector<int64_t> width, std::vector<int64_t> window, std::vector<int64_t> block_size,
               std::vector<int64_t> noncausal) {
  const int n = int(slot.size());
  TORCH_CHECK(n >= 1 && n <= MAXJOBS, "1..16 jobs");
  TORCH_CHECK(int(valid.size()) == n && int(lens.size()) == n && int(idx.size()) == n && int(qsl.size()) == n &&
                  int(seq_lens.size()) == n && int(t2r.size()) == n && int(bt.size()) == n && int(replay.size()) == n &&
                  int(ntok.size()) == n && int(width.size()) == n && int(window.size()) == n && int(block_size.size()) == n &&
                  int(noncausal.size()) == n,
              "job list lengths");
  Args a{};
  a.njobs = n;
  int max_fill = 0;
  for (int j = 0; j < n; ++j) {
    Job& b = a.job[j];
    TORCH_CHECK(slot[j].scalar_type() == at::kLong && slot[j].dim() == 1 && slot[j].stride(0) == 1, "slot_mapping int64 [n]");
    TORCH_CHECK(valid[j].scalar_type() == at::kBool && valid[j].is_contiguous(), "is_valid_token bool");
    TORCH_CHECK(lens[j].scalar_type() == at::kInt && lens[j].is_contiguous(), "swa lens int32");
    TORCH_CHECK(idx[j].scalar_type() == at::kInt && idx[j].stride(-1) == 1 && idx[j].dim() == 3, "swa indices int32 [T, 1, W]");
    TORCH_CHECK(qsl[j].scalar_type() == at::kInt && qsl[j].is_contiguous() && qsl[j].numel() >= 2, "query_start_loc int32");
    TORCH_CHECK(seq_lens[j].scalar_type() == at::kInt && seq_lens[j].is_contiguous(), "seq_lens int32");
    TORCH_CHECK(t2r[j].scalar_type() == at::kInt && t2r[j].is_contiguous(), "token_to_req int32");
    TORCH_CHECK(bt[j].scalar_type() == at::kInt && bt[j].dim() == 2 && bt[j].stride(1) == 1, "block table int32 [R, B]");
    b.slot = reinterpret_cast<const long long*>(slot[j].data_ptr<int64_t>());
    b.valid = valid[j].data_ptr<bool>();
    b.lens = lens[j].data_ptr<int>();
    b.idx = idx[j].data_ptr<int>();
    b.qsl = qsl[j].data_ptr<int>();
    b.seq_lens = seq_lens[j].data_ptr<int>();
    b.t2r = t2r[j].data_ptr<int>();
    b.bt = bt[j].data_ptr<int>();
    b.idx_stride = idx[j].stride(0);
    b.bt_stride = bt[j].stride(0);
    b.n_slot = int(slot[j].numel());
    b.ntok = int(ntok[j]);
    b.lens_len = int(lens[j].numel());
    b.width = int(width[j]);
    b.window = int(window[j]);
    b.block_size = int(block_size[j]);
    b.num_reqs = int(qsl[j].numel()) - 1;
    b.noncausal = noncausal[j] ? 1 : 0;
    TORCH_CHECK(b.ntok >= 0 && b.ntok <= idx[j].size(0) && b.width <= idx[j].size(-1) && b.ntok <= b.lens_len, "ntok / width");
    TORCH_CHECK(b.ntok <= t2r[j].numel() && valid[j].numel() >= std::max(b.n_slot, b.ntok), "buffer sizes");
    TORCH_CHECK(b.block_size > 0 && b.window > 0, "block_size / window");
    if (!b.noncausal) {
      TORCH_CHECK(replay[j].has_value() && replay[j]->scalar_type() == at::kInt && replay[j]->is_contiguous(), "causal: replay_start int32");
      b.replay = replay[j]->data_ptr<int>();
    }
    const int fill_len = std::max(b.n_slot, b.noncausal ? 0 : b.lens_len);
    b.fill_blocks = (fill_len + FILL_PER_BLOCK - 1) / FILL_PER_BLOCK;
    a.max_ntok = std::max(a.max_ntok, b.ntok);
    max_fill = std::max(max_fill, b.fill_blocks);
  }
  if (a.max_ntok + max_fill == 0) return;
  const c10::cuda::CUDAGuard guard(slot[0].device());
  swa_batch_kernel<<<dim3(a.max_ntok + max_fill, n), T, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------- multi_copy
// Up to 8 contiguous src -> dst copies (16-byte aligned, sizes multiple of 16 B) in ONE launch with many CTAs: 4 KB per CTA
// (256 threads x 16 B), CTA c finds its pair by the prefix chunk counts. Used by DS41H_GRAPH_OUT_BATCH for the decode graph's
// output copies (torch._foreach_copy_'s multi_tensor_apply ran these 4 x 80 KB copies on very few CTAs: 9.2 us in situ, more
// than the 4 stock memcpy nodes).
namespace mcp {
constexpr int MAXP = 8, T = 256, CHUNK = T * 16;
struct Args {
  const uint4* src[MAXP];
  uint4* dst[MAXP];
  long n16[MAXP];      // 16-byte words per pair
  int first[MAXP + 1]; // prefix CTA counts
  int npair;
};
__global__ void __launch_bounds__(T) multi_copy_kernel(const Args a) {
  const int c = blockIdx.x;
  int p = 0;
#pragma unroll
  for (int i = 1; i < MAXP; ++i) p += (i < a.npair && c >= a.first[i]) ? 1 : 0;
  const long i = long(c - a.first[p]) * T + threadIdx.x;
  if (i < a.n16[p]) a.dst[p][i] = a.src[p][i];
}
}  // namespace mcp

void multi_copy(std::vector<torch::Tensor> dst, std::vector<torch::Tensor> src) {
  TORCH_CHECK(dst.size() == src.size() && !dst.empty() && dst.size() <= size_t(mcp::MAXP), "multi_copy: 1..8 pairs");
  mcp::Args a{};
  int ctas = 0;
  for (size_t k = 0; k < dst.size(); ++k) {
    const auto& d = dst[k];
    const auto& s_ = src[k];
    TORCH_CHECK(d.is_cuda() && s_.is_cuda() && d.is_contiguous() && s_.is_contiguous() && d.nbytes() == s_.nbytes() &&
                    d.scalar_type() == s_.scalar_type() && d.device() == s_.device() && d.device() == dst[0].device(),
                "multi_copy: contiguous same-size same-dtype CUDA pairs on one device");
    TORCH_CHECK(d.nbytes() % 16 == 0 && reinterpret_cast<uintptr_t>(d.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(s_.data_ptr()) % 16 == 0,
                "multi_copy: 16-byte aligned, multiple of 16 bytes");
    a.src[k] = reinterpret_cast<const uint4*>(s_.data_ptr());
    a.dst[k] = reinterpret_cast<uint4*>(d.data_ptr());
    a.n16[k] = long(d.nbytes() / 16);
    a.first[k] = ctas;
    ctas += int((a.n16[k] + mcp::T - 1) / mcp::T);
  }
  a.npair = int(dst.size());
  for (int k = a.npair; k <= mcp::MAXP; ++k) a.first[k] = ctas;
  if (ctas == 0) return;
  const c10::cuda::CUDAGuard guard(dst[0].device());
  mcp::multi_copy_kernel<<<ctas, mcp::T, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swa_batch", &swa_batch, "batched DeepseekSparseSWA decode metadata (several KV-cache groups, one launch)");
  m.def("multi_copy", &multi_copy, "up to 8 contiguous src -> dst copies in one launch (4 KB per CTA)");
}
