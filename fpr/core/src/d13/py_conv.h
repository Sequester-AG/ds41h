// py_conv.h -- the engine's Python objects -> the core's structs: serve StepBatch / SeqEntry
// rowexp_ref CompactTables, the D13 / HSD configs. Shared by d13_py.cpp (stage 2a, module fpr_d13_cxx) and nd_py.cpp
// (stage 2b NativeDecode, module fpr_nd). Moved verbatim from d13_py.cpp.
#pragma once
#include <torch/extension.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "d13_host.h"
#include "hsd_prep.h"

namespace fpr {
namespace d13 {
namespace pyc {
namespace py = pybind11;

inline void ivec(py::handle o, std::vector<int64_t>* v) {   // None / list / tuple / numpy -> int64
  v->clear();
  if (!o || o.is_none()) return;
  if (py::isinstance<py::array>(o)) {
    auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(o);
    if (!a) throw std::runtime_error("fpr_d13_cxx: not an integer array");
    const int64_t* p = a.data();
    v->assign(p, p + a.size());
    return;
  }
  for (auto x : o) v->push_back(py::cast<int64_t>(x));
}
template <class T>
void ivec_t(py::handle o, std::vector<T>* v) {
  std::vector<int64_t> t;
  ivec(o, &t);
  v->assign(t.begin(), t.end());
}
inline int64_t iattr(py::handle o, const char* k, int64_t dflt) {
  if (!py::hasattr(o, k)) return dflt;
  py::object x = o.attr(k);
  if (x.is_none()) return dflt;
  return py::cast<int64_t>(x);
}
inline std::string rid_str(py::handle x) { return py::str(x); }

enum : int { WANT_BT = 1, WANT_WT = 2, WANT_ROWS = 4 };

inline SeqEntry seq_of(py::handle se, int want) {
  SeqEntry e;
  e.rid = rid_str(se.attr("rid"));
  e.kind = py::cast<std::string>(se.attr("kind"));
  e.row0 = (int32_t)iattr(se, "row0", 0);
  e.n_rows = (int32_t)iattr(se, "n_rows", 0);
  e.start = iattr(se, "start", 0);
  e.num_computed = iattr(se, "num_computed", 0);
  e.cached_upto = iattr(se, "cached_upto", 0);
  e.replay_start = iattr(se, "replay_start", 0);
  e.last_chunk = py::cast<bool>(se.attr("last_chunk"));
  e.state_slot = (int32_t)iattr(se, "state_slot", -1);
  e.tail_row0 = (int32_t)iattr(se, "tail_row0", 0);
  e.n_blocks = iattr(se, "n_blocks", 0);
  e.score = py::cast<bool>(se.attr("score"));
  e.forced = py::cast<bool>(se.attr("forced"));
  e.uid = (int32_t)iattr(se, "uid", -1);
  e.known = iattr(se, "known", -1);
  e.wver = iattr(se, "wver", -1);
  e.inc = iattr(se, "inc", 0);
  if (want & WANT_BT) {
    auto bt = std::make_shared<std::vector<int64_t>>();
    ivec(se.attr("block_table"), bt.get());
    e.block_table = bt;
  } else {   // len() only (n_blocks fallback): a vector of the right size is enough for se.n_blocks or len(bt)
    py::object b = se.attr("block_table");
    e.block_table = std::make_shared<std::vector<int64_t>>(b.is_none() ? 0 : py::len(b), 0);
  }
  if (want & WANT_WT) {
    auto wt = std::make_shared<WindowTable>();
    py::object w = se.attr("window_table");
    if (w.is_none()) {
    } else if (py::isinstance<py::dict>(w)) {   // the Python KV manager: {block: page}
      for (auto kv : py::reinterpret_borrow<py::dict>(w)) wt->items.push_back({py::cast<int64_t>(kv.first), py::cast<int64_t>(kv.second)});
    } else {   // fpr_kvtables.Window, items() = [(block, page)] in its table order
      for (auto it : w.attr("items")()) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(it);
        wt->items.push_back({py::cast<int64_t>(t[0]), py::cast<int64_t>(t[1])});
      }
    }
    e.window_table = wt;
  }
  return e;
}

inline StepBatch batch_of(py::handle sb, int want) {
  StepBatch b;
  b.rank = (int32_t)iattr(sb, "rank", 0);
  b.step = iattr(sb, "step", 0);
  b.n_rows = (int32_t)iattr(sb, "n_rows", 0);
  if (want & WANT_ROWS) {
    ivec(sb.attr("positions"), &b.positions);
    ivec_t(sb.attr("seq_slot"), &b.seq_slot);
    ivec(sb.attr("paged_slot"), &b.paged_slot);
    ivec(sb.attr("window_slot"), &b.window_slot);
    ivec_t(sb.attr("tail_rows"), &b.tail_rows);
    ivec_t(sb.attr("sample_rows"), &b.sample_rows);
  }
  py::object cp = sb.attr("copies");
  b.copies.resize(cp.is_none() ? 0 : py::len(cp));
  for (auto se : sb.attr("seqs")) b.seqs.push_back(seq_of(se, want));
  return b;
}

inline CompactTables ct_of(py::handle ct) {
  CompactTables c;
  c.rank = (int32_t)iattr(ct, "rank", 0);
  c.step = iattr(ct, "step", 0);
  c.k1 = (int32_t)iattr(ct, "k1", 0);
  ivec(ct.attr("slot"), &c.slot); ivec(ct.attr("row0"), &c.row0); ivec(ct.attr("n_rows"), &c.n_rows);
  ivec(ct.attr("kind"), &c.kind); ivec(ct.attr("start"), &c.start); ivec(ct.attr("replay_start"), &c.replay_start);
  ivec(ct.attr("cached_upto"), &c.cached_upto);
  ivec(ct.attr("win_slot"), &c.win_slot); ivec(ct.attr("win_blk"), &c.win_blk); ivec(ct.attr("win_page"), &c.win_page);
  ivec(ct.attr("bt_upd_slot"), &c.bt_upd_slot); ivec(ct.attr("bt_upd_start"), &c.bt_upd_start);
  ivec(ct.attr("bt_upd_len"), &c.bt_upd_len); ivec(ct.attr("bt_upd_pages"), &c.bt_upd_pages);
  ivec(ct.attr("tokens"), &c.tokens); ivec(ct.attr("sample_idx"), &c.sample_idx);
  ivec(ct.attr("uid"), &c.uid); ivec(ct.attr("last_chunk"), &c.last_chunk); ivec(ct.attr("score"), &c.score);
  ivec(ct.attr("forced"), &c.forced);
  int64_t n = 0;
  for (auto x : c.n_rows) n += x;
  c.n = n;
  return c;
}

template <class T>
py::array_t<T> arr(const std::vector<T>& v) {
  py::array_t<T> a((py::ssize_t)v.size());
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

[[noreturn]] inline void raise_py(const std::string& err) {   // the Python exception types _build raises
  if (err.find("exceeds the largest bucket") != std::string::npos) throw py::value_error(err);
  PyErr_SetString(PyExc_AssertionError, err.c_str());
  throw py::error_already_set();
}


inline Config cfg_of(py::dict c) {
  Config x;
  x.graphs = py::cast<bool>(c["graphs"]);
  x.largest_bucket = py::cast<int64_t>(c["largest_bucket"]);
  x.graph_max_ctx = py::cast<int64_t>(c["graph_max_ctx"]);
  x.k1 = py::cast<int64_t>(c["k1"]);
  x.delta_cap = py::cast<int64_t>(c["delta_cap"]);
  x.fb_reuse = py::cast<bool>(c["fb_reuse"]);
  py::sequence sc = c["scratch"];
  for (int r = 0; r < W && r < (int)py::len(sc); ++r) x.scratch[r] = py::cast<int64_t>(sc[r]);
  return x;
}
inline HsdConfig hcfg_of(py::dict c) {
  HsdConfig h;
  if (c.empty()) {
    h.seq_buckets = {8};
    h.row_buckets = {32};
    return h;
  }
  h.k = py::cast<int64_t>(c["k"]);
  h.max_seqs = py::cast<int64_t>(c["max_seqs"]);
  h.seq_buckets = py::cast<std::vector<int64_t>>(c["seq_buckets"]);
  h.row_buckets = py::cast<std::vector<int64_t>>(c["row_buckets"]);
  h.nt_max = py::cast<int64_t>(c["nt_max"]);
  h.pool = py::cast<bool>(c["pool"]);
  h.window = py::cast<int64_t>(c["window"]);
  h.ring = py::cast<int64_t>(c["ring"]);
  return h;
}

}  // namespace pyc
}  // namespace d13
}  // namespace fpr
