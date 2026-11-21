// nd_py.cpp -- module fpr_nd: NativeDecode ND-1 inside today's engine (nd_cxx.py installs it
// into D13 under ENG_D13_NATIVE=1). The engine passes its buffers / current staging sets / persistent events as integers;
// StepBatch / CompactTables go through the 2a converters (d13/py_conv.h).
#include <c10/cuda/CUDAGuard.h>

#include "../d13/py_conv.h"
#include "nd.h"

namespace py = pybind11;
void nd_handover_register(py::module_& m);   // nd_handover.cpp (engine_handover.h v0.2, the engine side)
using namespace fpr;
using namespace fpr::d13;
using namespace fpr::d13::pyc;

namespace {

int64_t gi(const py::dict& d, const char* k) {   // missing / None -> 0
  if (!d.contains(k)) return 0;
  py::object v = d[k];
  return v.is_none() ? 0 : py::cast<int64_t>(v);
}
template <class T>
T* gp(const py::dict& d, const char* k) { return reinterpret_cast<T*>((intptr_t)gi(d, k)); }

void common_of(const py::dict& d, nd::PreCommon* c) {
  c->rank = (int)gi(d, "rank");
  c->device = (int)gi(d, "device");
  c->stream = gp<CUstream_st>(d, "stream");
  c->R = gi(d, "R");
  c->ph = gp<int64_t>(d, "ph");
  c->oh = gp<int64_t>(d, "oh");
  c->pos_d = gp<const int64_t>(d, "pos_d");
  c->own_d = gp<const int64_t>(d, "own_d");
  c->pos_ev = gp<CUevent_st>(d, "pos_ev");
}

class Nd {
 public:
  explicit Nd(py::dict cfg) : h_(cfg_of(cfg), hcfg_of(py::dict())) {}

  void pre_full(py::dict d, py::handle sb, py::handle ct) {
    nd::PreFull p;
    common_of(d, &p.c);
    p.hsd_on = gi(d, "hsd_on") != 0;
    p.skip_deltas = gi(d, "skip_deltas") != 0;
    p.dl_rows_h = gp<int64_t>(d, "dl_rows_h"); p.dl_cols_h = gp<int64_t>(d, "dl_cols_h"); p.dl_pages_h = gp<int32_t>(d, "dl_pages_h");
    p.dl_ev_wait = gp<CUevent_st>(d, "dl_ev_wait"); p.dl_ev = gp<CUevent_st>(d, "dl_ev");
    p.dl_rows_d = gp<int64_t>(d, "dl_rows_d"); p.dl_cols_d = gp<int64_t>(d, "dl_cols_d"); p.dl_pages_d = gp<int32_t>(d, "dl_pages_d");
    p.delta_cap = gi(d, "delta_cap");
    p.dbt = gp<int32_t>(d, "dbt"); p.dbt_rows = gi(d, "dbt_rows"); p.dbt_cols = gi(d, "dbt_cols"); p.dbt_stride0 = gi(d, "dbt_stride0");
    p.in_h = gp<uint8_t>(d, "in_h"); p.in_ev_wait = gp<CUevent_st>(d, "in_ev_wait"); p.in_ev = gp<CUevent_st>(d, "in_ev");
    p.in_d = gp<void>(d, "in_d"); p.in_bytes = gi(d, "in_bytes");
    p.in_off_tok = gi(d, "in_off_tok"); p.in_off_valid = gi(d, "in_off_valid"); p.in_off_samp = gi(d, "in_off_samp");
    p.up_h = gp<uint8_t>(d, "up_h"); p.up_ev_wait = gp<CUevent_st>(d, "up_ev_wait"); p.up_ev = gp<CUevent_st>(d, "up_ev");
    p.up_d = gp<void>(d, "up_d"); p.up_bytes = gi(d, "up_bytes");
    p.up_off_valid = gi(d, "up_off_valid"); p.up_off_pslot = gi(d, "up_off_pslot"); p.up_off_rs = gi(d, "up_off_rs");
    p.up_B = gi(d, "up_B"); p.pslot_fill = gi(d, "pslot_fill");
    p.rslot_d = gp<const void>(d, "rslot_d"); p.rbase_d = gp<void>(d, "rbase_d"); p.rslot_bytes = gi(d, "rslot_bytes");
    p.rx_h32 = gp<int32_t>(d, "rx_h32"); p.rx_h64 = gp<int64_t>(d, "rx_h64");
    p.rx_ev_wait = gp<CUevent_st>(d, "rx_ev_wait"); p.rx_ev = gp<CUevent_st>(d, "rx_ev");
    p.rx_b32 = gp<int32_t>(d, "rx_b32"); p.rx_b64 = gp<int64_t>(d, "rx_b64");
    p.S_max = gi(d, "S_max"); p.ops_max = gi(d, "ops_max"); p.R_max = gi(d, "R_max");
    p.nb_h = gp<int64_t>(d, "nb_h"); p.nb_ev_wait = gp<CUevent_st>(d, "nb_ev_wait"); p.nb_ev = gp<CUevent_st>(d, "nb_ev");
    p.nb_d = gp<int64_t>(d, "nb_d");
    p.gt_exec = gp<CUgraphExec_st>(d, "gt_exec");
    p.engram = gi(d, "engram") != 0;
    p.eg_stage_h = gp<int32_t>(d, "eg_stage_h"); p.eg_ev = gp<CUevent_st>(d, "eg_ev"); p.eg_pk_d = gp<int32_t>(d, "eg_pk_d");
    p.eg_pk_n = gi(d, "eg_pk_n"); p.eg_off_row0 = gi(d, "eg_off_row0"); p.eg_off_nrows = gi(d, "eg_off_nrows");
    p.eg_off_start = gi(d, "eg_off_start"); p.eg_off_slot = gi(d, "eg_off_slot"); p.eg_off_look = gi(d, "eg_off_look");
    p.eg_off_counts = gi(d, "eg_off_counts"); p.eg_max_rows = gi(d, "eg_max_rows"); p.eg_max_seqs = gi(d, "eg_max_seqs");
    p.eg_ring_w = gi(d, "eg_ring_w");
    const StepBatch b = batch_of(sb, 0);
    const CompactTables c = ct_of(ct);
    std::string err;
    const c10::cuda::CUDAGuard g((c10::DeviceIndex)p.c.device);
    if (nd::pre_full(p, b, c, h_, &err) != Status::OK) throw std::runtime_error(err);
  }

  void pre_owner(py::dict d) {
    nd::PreOwner p;
    common_of(d, &p.c);
    p.gto_exec = gp<CUgraphExec_st>(d, "gto_exec");
    std::string err;
    const c10::cuda::CUDAGuard g((c10::DeviceIndex)p.c.device);
    if (nd::pre_owner(p, &err) != Status::OK) throw std::runtime_error(err);
  }

 private:
  D13Host h_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Nd>(m, "Nd")
      .def(py::init<py::dict>())
      .def("pre_full", &Nd::pre_full, "D13._pre's full path (ENG_D13_GT) for one rank: deltas, input / indexer blobs, rbase keep, "
                                     "rowexp descriptor, guard + graph T, positions D2H, Engram step args + upload")
      .def("pre_owner", &Nd::pre_owner, "D13._owner_skip's device part: graph T', positions D2H");
  nd_handover_register(m);
}
