// d13_py.cpp
// Module fpr_d13_cxx, built and installed by d13_cxx.py under PHYS_CXX_D13=1.
// Converts the engine's Python objects (serve StepBatch / SeqEntry, rowexp_ref CompactTables, HSD's sampling states) to
// the core's structs per call and returns Python values the existing code consumes. Decisions are C++; device work and
// the objects' identities stay Python.
#include <torch/extension.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "d13_host.h"
#include "hsd_prep.h"
#include "py_conv.h"

namespace py = pybind11;
using namespace fpr;
using namespace fpr::d13;

namespace {
using namespace fpr::d13::pyc;

class Host {
 public:
  Host(py::dict cfg, py::dict hcfg) : h_(cfg_of(cfg), hcfg_of(hcfg)) {}

  std::string async_ok(py::sequence batches, py::sequence cts, bool graphs) {
    std::vector<StepBatch> b;
    std::vector<CompactTables> c;
    for (auto x : batches) b.push_back(batch_of(x, 0));
    for (auto x : cts) c.push_back(ct_of(x));
    h_.set_graphs(graphs);
    return h_.async_ok(b, c);
  }

  py::tuple delta_rows(py::handle ct) {   // d13._delta_rows: (nu, rows, cols, pages) (nu == 0: None x 3)
    Deltas d;
    h_.deltas(0, ct_of(ct), &d);
    if (!d.nu) return py::make_tuple(0, py::none(), py::none(), py::none());
    return py::make_tuple(d.nu, arr(d.rows), arr(d.cols), arr(d.pages));
  }

  py::object owner_sig(int r, py::handle sb, py::handle ct, py::sequence prev) {
    std::vector<PrevOf> pv;
    for (auto p : prev) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(p);
      pv.push_back(PrevOf{py::cast<std::string>(t[0]), py::cast<bool>(t[1]), py::cast<int64_t>(t[2]), py::cast<int64_t>(t[3])});
    }
    StepBatch b = batch_of(sb, 0);
    auto s = h_.owner_sig(r, b, ct_of(ct), pv);
    if (!s) return py::none();
    py::list seqs;   // (seq index, state slot, row0, n_rows, n_blocks, prev index): d13_cxx maps the index to se.rid
    for (size_t k = 0; k < s->seqs.size(); ++k) {
      const auto& q = s->seqs[k];
      seqs.append(py::make_tuple((int64_t)k, q.state_slot, q.row0, q.n_rows, q.n_blocks, q.prev_index));
    }
    return py::make_tuple(s->n, seqs, py::tuple(py::cast(s->sample_idx)));
  }

  bool empty_eligible(py::handle sb, py::handle ct) { return h_.empty_eligible(batch_of(sb, 0), ct_of(ct)); }

  py::list fb_rows(py::sequence batches, py::sequence starts) {   // d13._fb_rows's host half
    std::vector<StepBatch> b, rb;
    for (auto x : batches) b.push_back(batch_of(x, WANT_BT | WANT_WT | WANT_ROWS));
    std::vector<std::vector<int64_t>> st;
    for (auto x : starts) {
      st.emplace_back();
      ivec(x, &st.back());
    }
    std::string err;
    if (h_.fallback_resolve(b, st, &rb, &err) != Status::OK) raise_py(err);
    py::list out;
    for (size_t r = 0; r < rb.size(); ++r) {
      py::list upd;   // (k, start, num_computed, n_blocks) of the sequences replaced (device-resolved starts)
      for (size_t k = 0; k < rb[r].seqs.size(); ++k)
        if (b[r].seqs[k].start < 0)
          upd.append(py::make_tuple((int64_t)k, rb[r].seqs[k].start, rb[r].seqs[k].num_computed, rb[r].seqs[k].n_blocks));
      out.append(py::make_tuple(upd, (int64_t)rb[r].seqs.size(), py::cast(rb[r].positions), py::cast(rb[r].paged_slot),
                                py::cast(rb[r].window_slot)));
    }
    return out;
  }

  // ---- HSD prepare
  void hsd_bind(py::sequence blobs, int64_t bm_ptr) {
    HsdPrep& p = h_.hsd();
    for (int i = 0; i < 2; ++i) {
      py::sequence row = py::reinterpret_borrow<py::sequence>(blobs[i]);
      for (int r = 0; r < W; ++r) p.bind(i, r, (uint8_t*)py::cast<int64_t>(row[r]));
    }
    if (bm_ptr) p.bind_bm((int32_t*)bm_ptr);
  }
  void hsd_restore(py::array_t<float, py::array::c_style | py::array::forcecast> temp,
                   py::array_t<int64_t, py::array::c_style | py::array::forcecast> seeds, py::sequence keys) {
    std::vector<float> t(temp.data(), temp.data() + temp.size());
    std::vector<int64_t> s(seeds.data(), seeds.data() + seeds.size());
    std::vector<uint64_t> k;
    for (auto x : keys) k.push_back(py::cast<uint64_t>(x));
    h_.hsd().restore(t, s, k);
  }
  py::list hsd_layout() {
    py::list o;
    for (const auto& f : h_.hsd().layout()) o.append(py::make_tuple(f.name, f.off, f.count));
    return o;
  }
  int64_t hsd_bytes() { return h_.hsd_blob_bytes(); }
  int64_t hsd_shared_end() { return h_.hsd().shared_end(); }

  // states[r][k] = (temperature, seed, drafts_valid, num_acceptable or None, id(state)) or None (not a drafting row)
  py::object hsd_build(int hi, py::sequence batches, py::sequence states, py::object bitmasks) {
    std::vector<StepBatch> b;
    for (auto x : batches) b.push_back(batch_of(x, WANT_WT | WANT_ROWS));
    std::vector<std::vector<HsdState>> sv(b.size());
    for (size_t r = 0; r < b.size(); ++r) {
      py::sequence sr = states[r];
      sv[r].resize(b[r].seqs.size());
      for (size_t k = 0; k < sv[r].size() && k < (size_t)py::len(sr); ++k) {
        py::object t = sr[k];
        if (t.is_none()) continue;
        py::tuple q = py::reinterpret_borrow<py::tuple>(t);
        HsdState& s = sv[r][k];
        s.temperature = py::cast<double>(q[0]);
        s.seed = py::cast<int64_t>(q[1]);
        s.drafts_valid = py::cast<bool>(q[2]);
        s.has_num_acceptable = !q[3].is_none();
        if (s.has_num_acceptable) s.num_acceptable = py::cast<int64_t>(q[3]);
        s.ident = py::cast<uint64_t>(q[4]);
      }
    }
    std::vector<HsdBitmask> bms;
    std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>> keep;
    const bool has_bm = !bitmasks.is_none();
    if (has_bm)
      for (auto x : bitmasks) {
        HsdBitmask m;
        if (!x.is_none()) {
          py::object a = py::hasattr(x, "numpy") ? x.attr("numpy")() : py::reinterpret_borrow<py::object>(x);
          keep.push_back(py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(a));
          m.n_s = keep.back().shape(0);
          m.words = keep.back().shape(1);
          m.rows = keep.back().data();
        }
        bms.push_back(m);
      }
    HsdStep cur;
    std::string err;
    h_.hsd().set_hi(hi);
    if (h_.hsd_prepare(b, sv, has_bm ? &bms : nullptr, &cur, &err) != Status::OK) raise_py(err);
    if (cur.v0) return py::none();
    py::list dseqs;
    for (const auto& x : cur.dseqs) dseqs.append(py::cast(x));
    py::dict d;
    d["hi"] = cur.hi; d["S_pad"] = cur.S_pad; d["D_pad"] = cur.D_pad; d["NT"] = cur.NT; d["n_req"] = cur.n_req;
    d["T"] = cur.T; d["n_s"] = cur.n_s; d["n_seq"] = cur.n_seq; d["n_t"] = cur.n_t; d["step"] = cur.step;
    d["dseqs"] = dseqs; d["has_bm"] = cur.has_bm;
    return d;
  }

 private:
  D13Host h_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Host>(m, "Host")
      .def(py::init<py::dict, py::dict>())
      .def("async_ok", &Host::async_ok)
      .def("delta_rows", &Host::delta_rows)
      .def("owner_sig", &Host::owner_sig)
      .def("empty_eligible", &Host::empty_eligible)
      .def("fb_rows", &Host::fb_rows)
      .def("hsd_bind", &Host::hsd_bind)
      .def("hsd_restore", &Host::hsd_restore)
      .def("hsd_layout", &Host::hsd_layout)
      .def("hsd_bytes", &Host::hsd_bytes)
      .def("hsd_shared_end", &Host::hsd_shared_end)
      .def("hsd_build", &Host::hsd_build);
}
