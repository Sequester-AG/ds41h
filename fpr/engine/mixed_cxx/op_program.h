// op_program.h -- the op-program format of the C++ driver.
//
// A Program is one rank's issue sequence of a piece for one fixed configuration (e.g. the HSD head for one HSD key): the
// ops in issue order, each on one of the program's streams, with the cross-stream ordering as explicit event record /
// wait ops. Interim (stage 4 (A)): programs are RECORDED from a real Python issue of the piece (mixed_cxx_head2.py: under
// a throwaway CUDA-graph capture, so nothing runs while recording):
//   - C ABI entries: their Args bytes through the rule-9 hooks (fpr_<ext>_set_record), or the owners' packers where a
//     served module has no hook
//
//   - dispatcher ops (ATen and custom ops such as _flashmla_C::sparse_decode_fwd), seen by a TorchDispatchMode: in-place /
//     out= ops as called, functional ops through their out= overload into the recorded result (held), views and empty
//     allocations not issued;
//   - torch.cuda.Event record / wait and Stream.wait_event / wait_stream.
// End state (B): the same format built by C++ logic.
//
// Rules: run() never synchronizes the host and launches only on the program's streams. It allocates only where the
// recorded op does (a dispatcher op's own outputs that the program never reads, e.g. FlashMLA's lse: the recorder aborts
// when a later op reads such an output). Events are created by the program (cudaEventDisableTiming). Tensors referenced
// by ops are held (Op::stack / Program::keep), so their addresses are fixed for the program's life; buffers referenced
// only by raw pointers (C ABI Args, Triton values) were allocated in the engine's graph pool during the recording, which
// no eager allocation reuses (the graph-pool sharing rule of graph H applies: programs and graphs of one pool never run
// concurrently). run() sets the grad mode the recording ran under (Program::grad).
#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "fpr_cabi.h"
#include "triton_replay.h"

namespace fpr_op {

enum Kind : int32_t {
  CABI = 1,        // fn(args): a C ABI entry (fpr_cabi.h) with its Args bytes (the stream inside: args' FprCall)
  TRITON = 2,      // fpr_tr_launch(rec, vals, grid, stream, 0, 0) (records with scratch are refused when recorded)
  BOXED = 3,       // op.callBoxed(copy of stack) under a CUDAStreamGuard of streams[stream]; returns dropped
  EV_RECORD = 4,   // cudaEventRecord(events[ev], streams[stream])
  EV_WAIT = 5,     // cudaStreamWaitEvent(streams[stream], events[ev]) (the event is recorded earlier in the program)
};

enum Grad : int32_t { INFERENCE = 0, NO_GRAD = 1, GRAD = 2 };

struct Op {
  int32_t kind = 0;
  int32_t stream = 0;            // index into Program::streams (0 = the rank's main stream)
  std::string name;              // "<ext>.<fn>" / the Triton key / "aten::mm.out" ... (diagnostics only)
  // CABI
  FprFn fn = nullptr;
  std::vector<uint64_t> args;    // the Args bytes (8-byte aligned storage)
  // TRITON
  FprTrRecord rec{};
  std::vector<uint64_t> vals;
  uint32_t grid[3] = {1, 1, 1};
  // BOXED
  c10::optional<c10::OperatorHandle> op;
  std::vector<c10::IValue> stack;
  // EV_*
  int32_t ev = -1;
};

struct Program {
  int device = -1;
  int32_t grad = INFERENCE;
  std::vector<cudaStream_t> streams;   // [0] = main
  std::vector<cudaEvent_t> events;
  std::vector<Op> ops;
  std::vector<at::Tensor> keep;        // buffers the ops reference that nothing else holds (recorded results)
  std::vector<int64_t> segs;           // op counts at the piece's stage boundaries (HSDT._steps' yields: a collective
                                       // follows): one thread issuing all ranks interleaves them there, as run_all does
  std::string label;
};

// Issue ops [b, e) in order on device `p.device` (sets it). Returns FPR_OK or the first error, with *at = its op index.
// No host sync.
int run_range(const Program& p, int64_t b, int64_t e, int64_t* at = nullptr);
inline int run(const Program& p, int64_t* at = nullptr) { return run_range(p, 0, (int64_t)p.ops.size(), at); }

}  // namespace fpr_op
