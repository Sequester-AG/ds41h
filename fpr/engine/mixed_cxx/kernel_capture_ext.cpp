// kernel_capture_ext.cpp -- capture a Python call's kernels into FprKcRecord structs, and the test launch entry
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "kernel_capture.h"

#define CU_OK(x)                                                                    \
  do {                                                                              \
    CUresult r_ = (x);                                                              \
    TORCH_CHECK(r_ == CUDA_SUCCESS, "kernel_capture: ", #x, " failed: ", (int)r_); \
  } while (0)

static void begin(int64_t stream) {
  C10_CUDA_CHECK(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(stream), cudaStreamCaptureModeThreadLocal));
}

// end the capture; every node must be a kernel (anything else: an error, the piece stays Python); records in
// dependency order (a topological order of the graph)
static std::vector<py::bytes> end(int64_t stream, int64_t device) {
  cudaGraph_t g = nullptr;
  C10_CUDA_CHECK(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream), &g));
  CUgraph cg = reinterpret_cast<CUgraph>(g);
  size_t n = 0;
  CU_OK(cuGraphGetNodes(cg, nullptr, &n));
  std::vector<CUgraphNode> nodes(n);
  if (n) CU_OK(cuGraphGetNodes(cg, nodes.data(), &n));
  // topological order (Kahn) over the node dependencies
  std::map<CUgraphNode, int> idx;
  for (size_t i = 0; i < n; ++i) idx[nodes[i]] = (int)i;
  std::vector<std::vector<int>> deps(n);
  for (size_t i = 0; i < n; ++i) {
    size_t nd = 0;
    CU_OK(cuGraphNodeGetDependencies(nodes[i], nullptr, nullptr, &nd));   // (CUDA 13: + edge data)
    std::vector<CUgraphNode> d(nd);
    if (nd) CU_OK(cuGraphNodeGetDependencies(nodes[i], d.data(), nullptr, &nd));
    for (auto x : d) deps[i].push_back(idx[x]);
  }
  std::vector<int> order, done(n, 0);
  for (size_t round = 0; round < n; ++round)
    for (size_t i = 0; i < n; ++i)
      if (!done[i] && std::all_of(deps[i].begin(), deps[i].end(), [&](int j) { return done[j] != 0; })) {
        done[i] = 1;
        order.push_back((int)i);
      }
  TORCH_CHECK(order.size() == n, "kernel_capture: cyclic graph?");
  std::vector<py::bytes> out;
  for (int i : order) {
    CUgraphNodeType t;
    CU_OK(cuGraphNodeGetType(nodes[i], &t));
    TORCH_CHECK(t == CU_GRAPH_NODE_TYPE_KERNEL, "kernel_capture: a non-kernel node (type ", (int)t, ") in the capture");
    CUDA_KERNEL_NODE_PARAMS p;
    std::memset(&p, 0, sizeof(p));
    CU_OK(cuGraphKernelNodeGetParams(nodes[i], &p));
    TORCH_CHECK(p.extra == nullptr, "kernel_capture: kernel launched with 'extra' parameters (not supported)");
    FprKcRecord r;
    std::memset(&r, 0, sizeof(r));
    CUfunction f = p.func;
    if (f == nullptr && p.kern != nullptr) CU_OK(cuKernelGetFunction(&f, p.kern));
    TORCH_CHECK(f != nullptr, "kernel_capture: no function in the kernel node");
    r.function = f;
    r.device = (int32_t)device;
    r.grid[0] = p.gridDimX, r.grid[1] = p.gridDimY, r.grid[2] = p.gridDimZ;
    r.block[0] = p.blockDimX, r.block[1] = p.blockDimY, r.block[2] = p.blockDimZ;
    r.shared = p.sharedMemBytes;
    CUlaunchAttributeValue v;
    std::memset(&v, 0, sizeof(v));
    if (cuGraphKernelNodeGetAttribute(nodes[i], CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &v) == CUDA_SUCCESS && v.clusterDim.x > 1) {
      r.cluster[0] = v.clusterDim.x, r.cluster[1] = v.clusterDim.y, r.cluster[2] = v.clusterDim.z;
    }
    int np = 0, used = 0;
    for (;; ++np) {
      size_t off = 0, sz = 0;
      if (cuFuncGetParamInfo(f, np, &off, &sz) != CUDA_SUCCESS) break;
      TORCH_CHECK(np < FPR_KC_MAX_PARAMS && off + sz <= FPR_KC_PARAM_BYTES, "kernel_capture: too many / too large parameters");
      std::memcpy(r.bytes + off, p.kernelParams[np], sz);
      r.param_off[np] = (int32_t)off;
      r.param_size[np] = (int32_t)sz;
      used = std::max(used, (int)(off + sz));
    }
    r.n_params = np;
    r.bytes_used = used;
    const char* nm = nullptr;
    if (cuFuncGetName(&nm, f) == CUDA_SUCCESS && nm) std::strncpy(r.name, nm, sizeof(r.name) - 1);
    out.emplace_back(reinterpret_cast<const char*>(&r), sizeof(r));
  }
  cudaGraphDestroy(g);
  return out;
}

static int64_t launch(py::bytes rec, int64_t stream) {
  const std::string s = rec;
  TORCH_CHECK(s.size() == sizeof(FprKcRecord), "kernel_capture: record size");
  return fpr_kc_launch(reinterpret_cast<const FprKcRecord*>(s.data()), reinterpret_cast<CUstream>(stream));
}

static int64_t record_size() { return sizeof(FprKcRecord); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("begin", &begin, "begin a thread-local stream capture on stream (int)");
  m.def("end", &end, "end it: one FprKcRecord (bytes) per kernel node, dependency order");
  m.def("launch", &launch, "fpr_kc_launch(record, stream)");
  m.def("record_size", &record_size);
}
