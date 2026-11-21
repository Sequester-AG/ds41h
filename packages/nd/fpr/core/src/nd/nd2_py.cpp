// nd2_py.cpp
// Every
// step nd's async_ok accepts runs natively (dispatch / wait); every other step runs in the Python engine between
// embedded_begin / embedded_end, as 2b's NativeWorker routes them. This proves nd::Engine bitwise against Python's D13 on
// one engine before 2b's C++ loop calls it directly. run_python = a direct call (this is the engine's main thread; the
// shim's GIL acquire nests). Results come back as (rank, step, [(rid, accepted)], [(rid, tokens)]) per rank.
#include <torch/extension.h>

#include <map>
#include <stdexcept>

#include "../d13/py_conv.h"
#include "nd_engine.h"

namespace py = pybind11;
using namespace fpr;
using namespace fpr::d13::pyc;

namespace {

std::vector<StepBatch> batches_of(py::list b) {
  std::vector<StepBatch> out;
  for (auto x : b) out.push_back(batch_of(x, WANT_BT | WANT_WT | WANT_ROWS));
  return out;
}
std::vector<CompactTables> cts_of(py::list c) {
  std::vector<CompactTables> out;
  for (auto x : c) out.push_back(ct_of(x));
  return out;
}
// host = dict(fresh=[(rid, serial, prompt, output, temperature, seed, structured)], dropped=[rid], grammar=[rid])
nd::StepHost host_of(py::dict d) {
  nd::StepHost h;
  if (d.contains("fresh"))
    for (auto x : py::list(d["fresh"])) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(x);
      nd::NewSeq f;
      f.rid = py::cast<std::string>(t[0]);
      f.serial = py::cast<int64_t>(t[1]);
      f.prompt = py::cast<std::vector<int32_t>>(t[2]);
      f.output = py::cast<std::vector<int32_t>>(t[3]);
      f.sampling.temperature = py::cast<float>(t[4]);
      f.sampling.seed = py::cast<int64_t>(t[5]);
      f.sampling.has_seed = true;
      f.structured = py::cast<bool>(t[6]);
      h.fresh.push_back(std::move(f));
    }
  if (d.contains("dropped"))
    for (auto x : py::list(d["dropped"])) h.dropped.push_back(py::cast<std::string>(x));
  if (d.contains("grammar"))
    for (auto x : py::list(d["grammar"])) {
      nd::GrammarRows g;
      g.rid = py::cast<std::string>(x);
      h.grammar.push_back(std::move(g));
    }
  return h;
}

class Engine2b {
 public:
  void init(int64_t handover) {
    std::string err;
    auto rp = [](const std::function<void()>& f) { f(); return true; };   // (the engine's main thread already)
    if (e_.init(reinterpret_cast<const FprEngineHandover*>((intptr_t)handover), rp, &err) != Status::OK)
      throw std::runtime_error(err);
  }
  std::string async_ok(py::list b, py::list c, py::dict host) { return e_.async_ok(batches_of(b), cts_of(c), host_of(host)); }
  int64_t dispatch(py::list b, py::list c, py::dict host) {
    std::string err;
    nd::Handle* h = nullptr;
    if (e_.dispatch(batches_of(b), cts_of(c), host_of(host), &h, &err) != Status::OK) throw std::runtime_error(err);
    hs_[++next_] = h;
    return next_;
  }
  py::list wait(int64_t id) {
    auto it = hs_.find(id);
    if (it == hs_.end()) throw std::runtime_error("Engine2b.wait: unknown handle");
    std::vector<StepResult> out;
    std::string err;
    if (e_.wait(it->second, &out, &err) != Status::OK) throw std::runtime_error(err);
    hs_.erase(it);
    py::list res;
    for (const auto& o : out) {
      py::list acc, tok;
      for (const auto& a : o.accepted) acc.append(py::make_tuple(a.first, a.second));
      for (const auto& t : o.tokens) tok.append(py::make_tuple(t.first, py::cast(t.second)));
      res.append(py::make_tuple(o.rank, o.step, acc, tok));
    }
    return res;
  }
  void drain() {
    std::string err;
    if (e_.drain(&err) != Status::OK) throw std::runtime_error(err);
  }
  void embedded_begin(py::dict host) {
    std::string err;
    if (e_.embedded_begin(host_of(host), &err) != Status::OK) throw std::runtime_error(err);
  }
  void embedded_end() {
    std::string err;
    if (e_.embedded_end(&err) != Status::OK) throw std::runtime_error(err);
  }
  std::string stats() const { return e_.stats_json(); }
  // (v3.3, ENG_ND_T) Engine._upload_v2 natively, for nd_t.py's _mixed_upload_T override: tab = the
  // T RankTables (CPU tensors), prev = [(rid, rank, seq index)] of tab's decode sequences with a prev-of (Python's
  // eng.prev_of: exact inside embedded steps). ValueError = refused, nothing issued (the override falls back).
  py::tuple upload_inputs(int r, int64_t R, int64_t scratch, bool hsd_on, int64_t in_d, int64_t in_d_bytes, int64_t host,
                          int64_t host_bytes, py::object tab, py::list prev) {
    p0::InputsSnap sn;
    sn.R = R, sn.scratch = scratch, sn.hsd = hsd_on;
    for (auto x : prev) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(x);
      sn.prev[rid_str(t[0])] = p0::PrevOf{py::cast<int32_t>(t[1]), py::cast<int32_t>(t[2])};
    }
    RankTables t;
    t.n = py::cast<int32_t>(tab.attr("n"));
    if (t.n) {
      auto np_of = [](py::object o) { return o.attr("numpy")(); };
      ivec(np_of(tab.attr("tokens")), &t.tokens);
      ivec(np_of(tab.attr("positions")), &t.positions);
      ivec(np_of(tab.attr("own_slot")), &t.own_slot);
      ivec_t(np_of(tab.attr("si")).attr("reshape")(-1), &t.si);
      ivec_t(np_of(tab.attr("sl")), &t.sl);
      ivec(tab.attr("sample_rows"), &t.sample_rows);
      for (auto q : py::list(tab.attr("seq_rows"))) {
        py::tuple u = py::reinterpret_borrow<py::tuple>(q);
        t.seq_rows.push_back(RankSeqRow{rid_str(u[0]), py::cast<int32_t>(u[1]), py::cast<int32_t>(u[2]),
                                        py::cast<std::string>(u[3]), py::cast<bool>(u[4])});
      }
    }
    int64_t n = 0, ns = 0, np_ = 0;
    std::string err;
    const Status st = e_.upload_inputs(r, sn, (void*)(intptr_t)in_d, in_d_bytes, (void*)(intptr_t)host, host_bytes, t, &n,
                                       &ns, &np_, &err);
    if (st == Status::BAD_ARG) throw py::value_error(err);
    if (st != Status::OK) throw std::runtime_error(err);
    return py::make_tuple(n, ns, np_);
  }
  // None when nd does not know the rid
  py::object history(const std::string& rid) const {
    const auto* h = e_.history(Rid(rid));
    return h == nullptr ? py::object(py::none()) : py::object(py::cast(*h));
  }
  // 2b: the engine as its IEngine2b, NON-owning. The C++ side calls init(handover, run_python)
  // itself; this Python object must outlive it (core2b_proc keeps it for the process lifetime)
  nd::IEngine2b* ptr() { return &e_; }

 private:
  nd::Engine e_;
  std::map<int64_t, nd::Handle*> hs_;
  int64_t next_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Engine2b>(m, "Engine2b")
      .def(py::init<>())
      .def("init", &Engine2b::init, "init(handover pointer: fpr_nd.Handover.ptr())")
      .def("async_ok", &Engine2b::async_ok)
      .def("dispatch", &Engine2b::dispatch, "-> handle id")
      .def("wait", &Engine2b::wait, "-> [(rank, step, [(rid, accepted)], [(rid, tokens)])] per rank")
      .def("drain", &Engine2b::drain)
      .def("embedded_begin", &Engine2b::embedded_begin)
      .def("embedded_end", &Engine2b::embedded_end)
      .def("stats", &Engine2b::stats)
      .def("upload_inputs", &Engine2b::upload_inputs,
           "(v3.3, ENG_ND_T) upload_inputs(r, R, scratch, hsd_on, in_d ptr, in_d bytes, set ptr, set bytes, tab, prev) -> "
           "(n, n_sample, n_pairs): Engine._upload_v2 natively (nd_t.py)")
      .def("history", &Engine2b::history, "nd's Req.hist of rid (prompt + output + every token since), or None")
      .def("iengine", [](Engine2b& e) { return py::capsule((void*)e.ptr(), "fpr.IEngine2b"); },
           "a non-owning capsule of nd::IEngine2b* (named \"fpr.IEngine2b\") for fpr_core2b; keep this object alive");
}
