// kvtables_py.cpp -- pybind11 module `fpr_kvtables`: the C++ KV manager (kv.h / src/kv) and tables (tables.h / src/tables)
// for stage 2a. The Python surface serve/ uses is the wrapper fpr/core/src/py/cxx_kv.py; this
// module only converts. Keys arrive as C++ Rid strings, block hashes as 16-byte bytes (cxx_kv.py's h128); pages / window
// tables are exposed BY REFERENCE (Pages / Window views over the SeqKV's shared_ptrs), as Python shares its list / dict.
// Build: tests/kvtables_py.mk (g++-13 -O2 -std=c++17 -ffp-contract=off -shared -fPIC, pybind11 + numpy from the venv).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "../../include/tables.h"
#include "../kv/kv_manager.h"

namespace py = pybind11;
using fpr::Hash;
using fpr::Rid;
using fpr::kvimpl::OwnedKVManager;

namespace {

[[noreturn]] void raise(PyObject* type, const std::string& msg) {
  PyErr_SetString(type, msg.c_str());
  throw py::error_already_set();
}

Hash hash_of(py::handle b) {
  char* p = nullptr;
  Py_ssize_t n = 0;
  if (b.is_none()) return Hash{0, 0};
  if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0 || n < 16) raise(PyExc_TypeError, "block hash: 16+ bytes expected");
  uint64_t hi = 0, lo = 0;
  for (int i = 0; i < 8; ++i) hi = (hi << 8) | (uint8_t)p[i];
  for (int i = 8; i < 16; ++i) lo = (lo << 8) | (uint8_t)p[i];
  return Hash{hi, lo};
}
py::bytes bytes_of(const Hash& h) {
  char b[16];
  for (int i = 0; i < 8; ++i) b[i] = (char)(h.hi >> (56 - 8 * i));
  for (int i = 0; i < 8; ++i) b[8 + i] = (char)(h.lo >> (56 - 8 * i));
  return py::bytes(b, 16);
}
std::vector<Hash> hashes_of(const py::list& l) {
  std::vector<Hash> v;
  v.reserve(l.size());
  for (auto h : l) v.push_back(hash_of(h));
  return v;
}
py::tuple copy_t(const fpr::CopyOp& c) {
  return py::make_tuple(fpr::pool_name(c.pool), c.src_rank, c.src_page, c.dst_rank, c.dst_page, c.step);
}
py::list copies_t(const std::vector<fpr::CopyOp>& v) {
  py::list l;
  for (auto& c : v) l.append(copy_t(c));
  return l;
}

// ---- by-reference views
struct Pages { std::shared_ptr<std::vector<int64_t>> v; };
struct Window { std::shared_ptr<fpr::WindowTable> w; };

int64_t norm_index(int64_t i, int64_t n) {
  if (i < 0) i += n;
  if (i < 0 || i >= n) raise(PyExc_IndexError, "list index out of range");
  return i;
}

template <class T> py::array_t<T> arr(const std::vector<T>& v) {
  py::array_t<T> a((py::ssize_t)v.size());
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}
template <class T> py::array_t<T> arr2(const std::vector<T>& v, py::ssize_t r, py::ssize_t c) {
  py::array_t<T> a({r, c});
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

// a Python StepBatch, flattened by cxx_kv.py, -> fpr::StepBatch (the tables read no copies / poison)
fpr::StepBatch sb_of(const py::tuple& t) {
  fpr::StepBatch sb;
  sb.rank = t[0].cast<int32_t>();
  sb.step = t[1].cast<int64_t>();
  sb.n_rows = t[2].cast<int32_t>();
  sb.positions = t[3].cast<std::vector<int64_t>>();
  sb.seq_slot = t[4].cast<std::vector<int32_t>>();
  sb.paged_slot = t[5].cast<std::vector<int64_t>>();
  sb.window_slot = t[6].cast<std::vector<int64_t>>();
  sb.tail_rows = t[7].cast<std::vector<int32_t>>();
  sb.sample_rows = t[8].cast<std::vector<int32_t>>();
  for (auto h : t[9].cast<py::list>()) {
    py::tuple s = h.cast<py::tuple>();
    fpr::SeqEntry se;
    se.rid = s[0].cast<std::string>(); se.kind = s[1].cast<std::string>(); se.row0 = s[2].cast<int32_t>();
    se.n_rows = s[3].cast<int32_t>(); se.start = s[4].cast<int64_t>(); se.num_computed = s[5].cast<int64_t>();
    se.cached_upto = s[6].cast<int64_t>(); se.replay_start = s[7].cast<int64_t>(); se.last_chunk = s[8].cast<bool>();
    if (py::isinstance<Pages>(s[9])) se.block_table = s[9].cast<Pages&>().v;
    else se.block_table = std::make_shared<std::vector<int64_t>>(s[9].cast<std::vector<int64_t>>());
    if (py::isinstance<Window>(s[10])) se.window_table = s[10].cast<Window&>().w;
    else {
      auto w = std::make_shared<fpr::WindowTable>();
      for (auto it : s[10].cast<py::list>()) { py::tuple bp = it.cast<py::tuple>(); w->items.emplace_back(bp[0].cast<int64_t>(), bp[1].cast<int64_t>()); }
      se.window_table = w;
    }
    se.state_slot = s[11].cast<int32_t>(); se.tail_row0 = s[12].cast<int32_t>(); se.n_blocks = s[13].cast<int64_t>();
    se.score = s[14].cast<bool>(); se.forced = s[15].cast<bool>(); se.uid = s[16].cast<int32_t>();
    se.known = s[17].cast<int64_t>(); se.wver = s[18].cast<int64_t>(); se.inc = s[19].cast<int64_t>();
    sb.seqs.push_back(std::move(se));
  }
  return sb;
}

fpr::TokensOf tokens_fn(py::object fn, const py::dict& rids) {
  if (fn.is_none()) return nullptr;
  return [fn, rids](const Rid& r, int64_t pos) -> int64_t {
    py::object key = rids.contains(py::str(r)) ? py::object(rids[py::str(r)]) : py::object(py::str(r));
    return fn(key, pos).cast<int64_t>();
  };
}

}  // namespace

// the ITables ABI this extension implements. 1 = build / build_compact; 2 = + save_state /
// restore_state (appended virtuals). nd's exporter checks it (fpr_tables_abi via dlsym, or fpr_kvtables.tables_abi())
// before calling save_state / restore_state through an ITables* from this .so: a stale .so must refuse, not crash.
extern "C" __attribute__((visibility("default"))) int fpr_tables_abi(void) { return FPR_TABLES_ABI; }

// TableBuilder.build's RankTables -> dict, one function (also _rt_dict's)
static py::dict rt_dict(const fpr::RankTables& t) {
        py::dict d;
        const py::ssize_t n = t.n;
        d["n"] = t.n;
        d["positions"] = arr(t.positions); d["tokens"] = arr(t.tokens); d["own_slot"] = arr(t.own_slot);
        d["si"] = arr2(t.si, n, fpr::SLIDING_WINDOW); d["sl"] = arr(t.sl);
        py::dict ti, tl;
        for (size_t s = 0; s < t.ti.size(); ++s) { ti[py::int_(t.ti_sources[s])] = arr2(t.ti[s], n, 512); tl[py::int_(t.ti_sources[s])] = arr(t.tl[s]); }
        d["ti"] = ti; d["tl"] = tl;
        d["seq_slot"] = arr(t.seq_slot); d["paged_slot"] = arr(t.paged_slot); d["tail_rows"] = arr(t.tail_rows);
        d["sample_idx"] = arr(t.sample_idx); d["targets"] = arr(t.targets);
        const auto& q = t.seq;
        py::dict sq;
        sq["state_slot"] = arr(q.state_slot); sq["uid"] = arr(q.uid); sq["start"] = arr(q.start); sq["n_rows"] = arr(q.n_rows);
        sq["kind"] = arr(q.kind); sq["last_chunk"] = arr(q.last_chunk); sq["score"] = arr(q.score); sq["forced"] = arr(q.forced);
        sq["replay_start"] = arr(q.replay_start); sq["cached_upto"] = arr(q.cached_upto); sq["n_blocks"] = arr(q.n_blocks);
        sq["bt_reset"] = arr(q.bt_reset); sq["bt_upd_seq"] = arr(q.bt_upd_seq); sq["bt_upd_start"] = arr(q.bt_upd_start);
        sq["bt_upd_len"] = arr(q.bt_upd_len); sq["bt_upd_pages"] = arr(q.bt_upd_pages);
        if (q.has_block_table) sq["block_table"] = arr2(q.block_table, (py::ssize_t)q.bt_rows, (py::ssize_t)q.bt_cols);
        d["seq"] = sq;
        return d;
}

PYBIND11_MODULE(fpr_kvtables, m) {
  m.doc() = "C++ KV manager + tables; use through fpr/core/src/py/cxx_kv.py";

  py::class_<Pages>(m, "Pages")
      .def("__len__", [](const Pages& p) { return p.v->size(); })
      .def("__getitem__", [](const Pages& p, int64_t i) { return (*p.v)[(size_t)norm_index(i, (int64_t)p.v->size())]; })
      .def("__getitem__", [](const Pages& p, py::slice s) {
        size_t start, stop, step, len;
        if (!s.compute(p.v->size(), &start, &stop, &step, &len)) throw py::error_already_set();
        py::list l;
        for (size_t i = 0; i < len; ++i) l.append((*p.v)[start + i * step]);
        return l;
      })
      .def("__iter__", [](const Pages& p) { return py::iter(py::cast(*p.v)); })
      .def("__contains__", [](const Pages& p, int64_t x) { return std::find(p.v->begin(), p.v->end(), x) != p.v->end(); })
      .def("__eq__", [](const Pages& p, py::object o) {
        if (py::isinstance<Pages>(o)) return *p.v == *o.cast<Pages&>().v;
        try { return *p.v == o.cast<std::vector<int64_t>>(); } catch (...) { return false; }
      })
      .def("index", [](const Pages& p, int64_t x) {
        auto it = std::find(p.v->begin(), p.v->end(), x);
        if (it == p.v->end()) raise(PyExc_ValueError, std::to_string(x) + " is not in list");
        return (int64_t)(it - p.v->begin());
      })
      .def("count", [](const Pages& p, int64_t x) { return (int64_t)std::count(p.v->begin(), p.v->end(), x); })
      .def("__reversed__", [](const Pages& p) { std::vector<int64_t> r(p.v->rbegin(), p.v->rend()); return py::iter(py::cast(r)); })
      .def("copy", [](const Pages& p) { return *p.v; })
      .def("__bool__", [](const Pages& p) { return !p.v->empty(); })
      .def("tolist", [](const Pages& p) { return *p.v; })
      .def("ptr", [](const Pages& p) { return (uintptr_t)p.v.get(); })
      .def("use_count", [](const Pages& p) { return (int64_t)p.v.use_count(); })
      .def("__repr__", [](const Pages& p) { return "Pages(" + std::to_string(p.v->size()) + ")"; })
      .def("__reduce__", [](const Pages& p) {
        return py::make_tuple(py::module_::import("builtins").attr("list"), py::make_tuple(py::cast(*p.v)));
      });

  py::class_<Window>(m, "Window")
      .def("__len__", [](const Window& w) { return w.w->size(); })
      .def("__contains__", [](const Window& w, py::object b) { return py::isinstance<py::int_>(b) && w.w->contains(b.cast<int64_t>()); })
      .def("__getitem__", [](const Window& w, int64_t b) {
        int64_t i = w.w->find(b);
        if (i < 0) raise(PyExc_KeyError, std::to_string(b));
        return w.w->items[(size_t)i].second;
      })
      .def("get", [](const Window& w, int64_t b, py::object d) -> py::object {
        int64_t i = w.w->find(b);
        return i < 0 ? d : py::object(py::int_(w.w->items[(size_t)i].second));
      }, py::arg("b"), py::arg("default") = py::none())
      .def("keys", [](const Window& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.first); return k; })
      .def("values", [](const Window& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.second); return k; })
      .def("items", [](const Window& w) { return w.w->items; })
      .def("__iter__", [](const Window& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.first); return py::iter(py::cast(k)); })
      .def("__bool__", [](const Window& w) { return !w.w->empty(); })
      .def("copy", [](const Window& w) { py::dict d; for (auto& e : w.w->items) d[py::int_(e.first)] = e.second; return d; })
      .def("__eq__", [](const Window& w, py::object o) {
        if (py::isinstance<Window>(o)) return w.w->items == o.cast<Window&>().w->items;
        if (!py::isinstance<py::dict>(o)) return false;
        py::dict d = o.cast<py::dict>();
        if (d.size() != w.w->size()) return false;
        for (auto& e : w.w->items) { py::int_ k(e.first); if (!d.contains(k) || d[k].cast<int64_t>() != e.second) return false; }
        return true;
      })
      .def("ptr", [](const Window& w) { return (uintptr_t)w.w.get(); })
      .def("use_count", [](const Window& w) { return (int64_t)w.w.use_count(); })
      .def("__repr__", [](const Window& w) { return "Window(" + std::to_string(w.w->size()) + ")"; })
      // (pickle / copy: a plain dict snapshot {block: page}, insertion order kept)
      .def("__reduce__", [](const Window& w) {
        py::list it;
        for (auto& e : w.w->items) it.append(py::make_tuple(e.first, e.second));
        return py::make_tuple(py::module_::import("builtins").attr("dict"), py::make_tuple(it));
      });

  py::class_<fpr::SeqKV, std::shared_ptr<fpr::SeqKV>>(m, "SeqKV")
      .def_readonly("seq_key", &fpr::SeqKV::seq_id)
      .def_readonly("rank", &fpr::SeqKV::rank)
      .def_readonly("num_registered", &fpr::SeqKV::num_registered)
      .def_property_readonly("state_slot", [](const fpr::SeqKV& s) -> py::object { return s.state_slot < 0 ? py::none() : py::object(py::int_(s.state_slot)); })
      .def_readonly("reserved", &fpr::SeqKV::reserved)
      .def_readonly("cached_tokens", &fpr::SeqKV::cached_tokens)
      .def_readonly("replay_start", &fpr::SeqKV::replay_start)
      .def_readonly("wver", &fpr::SeqKV::wver)
      .def_readonly("shadow", &fpr::SeqKV::shadow)
      .def_readwrite("cp_have", &fpr::SeqKV::cp_have)
      .def_readonly("inc", &fpr::SeqKV::inc)
      .def("pages_view", [](const fpr::SeqKV& s) { return Pages{s.pages}; })
      .def("window_view", [](const fpr::SeqKV& s) { return Window{s.window}; })
      .def("pages_ptr", [](const fpr::SeqKV& s) { return (uintptr_t)s.pages.get(); })
      .def("window_ptr", [](const fpr::SeqKV& s) { return (uintptr_t)s.window.get(); });

  py::class_<OwnedKVManager, std::shared_ptr<OwnedKVManager>>(m, "KV")
      .def(py::init([](int32_t n_ranks, int64_t page_tokens, int64_t window, int64_t wpt, int64_t replay_tokens, double rw,
                       py::list pools, int64_t hbm_pages, std::string poison_mode) {
        fpr::KVGeometry g;
        g.n_ranks = n_ranks; g.page_tokens = page_tokens; g.window = window; g.window_page_tokens = wpt;
        g.replay_tokens = replay_tokens; g.reserve_watermark = rw; g.hbm_pages = hbm_pages; g.poison_mode = poison_mode;
        fpr::PoolSpec* dst[3] = {&g.paged, &g.window_pool, &g.state};
        for (size_t i = 0; i < 3; ++i) {
          py::tuple t = pools[i].cast<py::tuple>();
          *dst[i] = fpr::PoolSpec{t[0].cast<std::string>(), t[1].cast<std::string>(), t[2].cast<int64_t>(), t[3].cast<int64_t>(), t[4].cast<int64_t>()};
        }
        return std::make_shared<OwnedKVManager>(g);
      }))
      .def("lookup", [](const OwnedKVManager& k, py::list h) { return k.lookup(hashes_of(h)); })
      .def("free_pages", &OwnedKVManager::free_pages)
      .def("usage", [](const OwnedKVManager& k, int r) { auto u = k.usage(r); return py::make_tuple(u.used, u.cached_free, u.free, u.reserved, u.total, u.window_used, u.state_used); })
      .def("pages_for", &OwnedKVManager::pages_for)
      .def("hit_after_cap", [](const OwnedKVManager& k, int64_t h, int64_t p) { int64_t a, b, c; k.hit_after_cap(h, p, &a, &b, &c); return py::make_tuple(a, b, c); })
      .def("admit", [](OwnedKVManager& k, const Rid& id, int rank, py::list h, int64_t plen, int64_t step, int cf) -> py::object {
        if (k.seq(id)) raise(PyExc_AssertionError, id);
        fpr::AdmitResult r{};
        fpr::Status st = k.admit(id, rank, hashes_of(h), plen, step, cf, &r);
        if (st == fpr::Status::NO_ROOM) return py::none();
        if (st != fpr::Status::OK) raise(PyExc_AssertionError, k.last_error());
        return py::make_tuple(r.rank, r.hit_blocks, r.cached_tokens, r.replay_start, r.copied_blocks, copies_t(r.copies));
      })
      .def("can_allocate", [](const OwnedKVManager& k, const Rid& id, int64_t s, int64_t e, int64_t we) {
        if (!k.seq(id)) raise(PyExc_KeyError, id);
        return k.can_allocate(id, s, e, we);
      })
      .def("allocate", [](OwnedKVManager& k, const Rid& id, int64_t s, int64_t e, int64_t we) {
        if (!k.seq(id)) raise(PyExc_KeyError, id);
        return k.allocate(id, s, e, we);
      })
      .def("commit", [](OwnedKVManager& k, const Rid& id, int64_t nc, py::list h) {
        if (!k.seq(id)) raise(PyExc_KeyError, id);
        k.commit(id, nc, hashes_of(h));
      })
      .def("shrink_reservation", [](OwnedKVManager& k, const Rid& id) { if (!k.seq(id)) raise(PyExc_KeyError, id); k.shrink_reservation(id); })
      .def("open_shadow", [](OwnedKVManager& k, const Rid& key, int rank, int64_t rs) {
        if (k.seq(key)) raise(PyExc_AssertionError, key);
        return k.open_shadow(key, rank, rs);
      })
      .def("trim_window", [](OwnedKVManager& k, const Rid& id, int64_t lo) { if (!k.seq(id)) raise(PyExc_KeyError, id); k.trim_window(id, lo); })
      .def("free_now", [](OwnedKVManager& k, const Rid& id) { if (!k.seq(id)) raise(PyExc_KeyError, id); k.free_now(id); })
      .def("free_deferred", [](OwnedKVManager& k, const Rid& id, int64_t s) { if (!k.seq(id)) raise(PyExc_KeyError, id); k.free_deferred(id, s); })
      .def("flush_deferred", &OwnedKVManager::flush_deferred)
      .def("migrate", [](OwnedKVManager& k, const Rid& id, int dst, int64_t step, int64_t nc) -> py::object {
        if (!k.seq(id)) raise(PyExc_KeyError, id);
        std::vector<fpr::CopyOp> ops;
        fpr::Status st = k.migrate(id, dst, step, nc, &ops);
        if (st == fpr::Status::NO_ROOM) return py::none();
        return copies_t(ops);
      })
      .def("tier_copies", [](OwnedKVManager& k, int64_t step, int64_t mp) { return copies_t(k.tier_copies(step, mp)); })
      .def("release_copies", &OwnedKVManager::release_copies)
      .def("take_poison", [](OwnedKVManager& k, int r) {
        py::list l;
        for (auto& p : k.take_poison(r)) l.append(py::make_tuple(fpr::pool_name(p.first), p.second));
        return l;
      })
      .def("check", [](const OwnedKVManager& k) { if (k.check() != fpr::Status::OK) raise(PyExc_AssertionError, k.last_error()); })
      .def("seq", [](const OwnedKVManager& k, const Rid& key) { return k.seq_ptr(key); })
      .def("seq_keys", &OwnedKVManager::seq_keys)
      .def("num_seqs", &OwnedKVManager::num_seqs)
      .def("stats", [](const OwnedKVManager& k) {
        const auto& s = k.stats();
        py::dict d; d["copies"] = s.copies; d["copy_bytes"] = s.copy_bytes; d["evictions"] = s.evictions;
        d["promoted"] = s.promoted; d["demoted"] = s.demoted; return d;
      })
      .def("num_pages", &OwnedKVManager::num_pages)
      .def("num_free", &OwnedKVManager::num_free)
      .def("num_used", &OwnedKVManager::num_used)
      .def("reserved", [](const OwnedKVManager& k, int32_t pool, int r) { return k.pool(pool, r).reserved; })
      .def("ref", [](const OwnedKVManager& k, int32_t pool, int r, int64_t p) { const auto& pl = k.pool(pool, r); return pl.ref[(size_t)norm_index(p, pl.num_pages())]; })
      .def("refs", [](const OwnedKVManager& k, int32_t pool, int r) { const auto& pl = k.pool(pool, r); return std::vector<int64_t>(pl.ref.begin(), pl.ref.end()); })
      .def("lru", &OwnedKVManager::lru_list)
      .def("cached_get", [](const OwnedKVManager& k, int32_t pool, int r, py::object h) -> int64_t {
        int64_t p; return k.pool(pool, r).cached_get(hash_of(h), &p) ? p : -1;
      })
      .def("cached_items", [](const OwnedKVManager& k, int32_t pool, int r) {
        py::list l; for (auto& e : k.pool(pool, r).cached) l.append(py::make_tuple(bytes_of(e.first), e.second)); return l;
      })
      .def("hash_of", [](const OwnedKVManager& k, int32_t pool, int r, int64_t p) -> py::object {
        const auto& pl = k.pool(pool, r); p = norm_index(p, pl.num_pages());
        return pl.has_hash[(size_t)p] ? py::object(bytes_of(pl.hash_of[(size_t)p])) : py::none();
      })
      .def("nfree_uncached", [](const OwnedKVManager& k, int32_t pool, int r) { return k.pool(pool, r).free_uncached.size(); })
      .def("n_hbm", [](const OwnedKVManager& k, int32_t pool, int r) { return k.pool(pool, r).n_hbm; })
      .def("poison_on", [](const OwnedKVManager& k, int32_t pool, int r) { return k.pool(pool, r).poison_on; })
      .def("poison_list", [](const OwnedKVManager& k, int32_t pool, int r) { return k.pool(pool, r).poison; })
      .def("poison_append", &OwnedKVManager::poison_append)
      .def("pinned", [](const OwnedKVManager& k) {
        py::dict d;
        for (auto& e : k.pinned_list()) {
          py::list l; for (auto& p : e.second) l.append(py::make_tuple(fpr::pool_name(p.pool), p.rank, p.page));
          d[py::int_(e.first)] = l;
        }
        return d;
      })
      .def("pending_steps", &OwnedKVManager::pending_steps)
      .def("deferred", [](const OwnedKVManager& k) {
        py::dict d; for (auto& e : k.deferred_list()) d[py::int_(e.first)] = e.second; return d;
      })
      .def("ikv_capsule", [](const std::shared_ptr<OwnedKVManager>& k) {   // NON-owning: the Python object owns it
        return py::capsule(static_cast<void*>(static_cast<fpr::IKVManager*>(k.get())), "fpr.IKVManager");
      })
      .def("cxx_capsule", [](const std::shared_ptr<OwnedKVManager>& k) {
        auto* sp = new std::shared_ptr<fpr::IKVManager>(k);
        return py::capsule(sp, "fpr::IKVManager::shared_ptr", [](PyObject* c) {
          delete static_cast<std::shared_ptr<fpr::IKVManager>*>(PyCapsule_GetPointer(c, "fpr::IKVManager::shared_ptr"));
        });
      });

  // ---- build_step_batch: items = the rank's work, flattened by cxx_kv.py (rid string, kind, start, n, cached_upto,
  // replay_start, last_chunk, score, forced, uid, known, has_cp, cp subs [(rank, start, n)], cp rid string)
  m.def("build_step_batch", [](std::shared_ptr<OwnedKVManager> kv, int64_t step, py::list items, int rank) {
    fpr::StepPlan plan;
    plan.step = step;
    for (auto h : items) {
      py::tuple t = h.cast<py::tuple>();
      fpr::SeqWork w;
      w.rid = t[0].cast<std::string>(); w.rank = rank; w.kind = t[1].cast<std::string>(); w.start = t[2].cast<int64_t>();
      w.n = t[3].cast<int64_t>(); w.cached_upto = t[4].cast<int64_t>(); w.replay_start = t[5].cast<int64_t>();
      w.last_chunk = t[6].cast<bool>(); w.score = t[7].cast<bool>(); w.forced = t[8].cast<bool>();
      w.uid = t[9].cast<int32_t>(); w.known = t[10].cast<int64_t>(); w.has_cp = t[11].cast<bool>();
      if (w.has_cp) {
        for (auto s : t[12].cast<py::list>()) { py::tuple x = s.cast<py::tuple>(); w.cp.subs.emplace_back(x[0].cast<int32_t>(), x[1].cast<int64_t>(), x[2].cast<int64_t>()); }
        w.cp.rid = t[13].cast<std::string>();
      }
      plan.work.push_back(std::move(w));
    }
    fpr::StepBatch sb;
    std::string err;
    fpr::KVViews v;
    v.seq = [&kv](const Rid& id) -> const fpr::SeqKV* { return static_cast<const OwnedKVManager&>(*kv).seq(id); };
    if (fpr::build_step_batch(plan, v, kv->geom().page_tokens, kv->geom().window_page_tokens, rank, &sb, &err) != fpr::Status::OK)
      raise(PyExc_KeyError, err);
    // the entries' order = Python's stable sort (decode first); per entry: index into `items` + the fields C++ computed
    std::vector<size_t> order(plan.work.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return (plan.work[a].kind != "decode") < (plan.work[b].kind != "decode"); });
    py::list seqs;
    for (size_t k = 0; k < sb.seqs.size(); ++k) {
      const auto& se = sb.seqs[k];
      seqs.append(py::make_tuple(order[k], se.row0, se.n_blocks, se.tail_row0, se.wver, se.inc, se.state_slot,
                                 Pages{std::const_pointer_cast<std::vector<int64_t>>(se.block_table)},
                                 Window{std::const_pointer_cast<fpr::WindowTable>(se.window_table)},
                                 (uintptr_t)se.block_table.get(), (uintptr_t)se.window_table.get()));
    }
    return py::make_tuple(sb.n_rows, sb.positions, sb.seq_slot, sb.paged_slot, sb.window_slot, sb.tail_rows, sb.sample_rows, seqs);
  });

  // ---- TableBuilder
  m.def("tables_abi", []() { return FPR_TABLES_ABI; });
  // StepBatch / RankTables behind capsules, so another extension can call
  // ITables::build through itables_ptr() (the cross-module golden; production nd builds its own StepBatch)
  m.def("_sb_capsule", [](py::tuple sbt) {
    return py::capsule(new fpr::StepBatch(sb_of(sbt)), "fpr::StepBatch",
                       [](PyObject* c) { delete static_cast<fpr::StepBatch*>(PyCapsule_GetPointer(c, "fpr::StepBatch")); });
  });
  m.def("_rt_capsule", []() {
    return py::capsule(new fpr::RankTables(), "fpr::RankTables",
                       [](PyObject* c) { delete static_cast<fpr::RankTables*>(PyCapsule_GetPointer(c, "fpr::RankTables")); });
  });
  m.def("_rt_dict", [](py::capsule c) {
    return rt_dict(*static_cast<fpr::RankTables*>(PyCapsule_GetPointer(c.ptr(), "fpr::RankTables")));
  });
  // an opaque saved delta state of a TableBuilder (save_state / restore_state)
  struct TbState { std::shared_ptr<const void> s; };
  py::class_<TbState>(m, "TableBuilderState");
  py::class_<fpr::TableBuilder>(m, "TableBuilder")
      .def(py::init<>())
      .def_readwrite("sources", &fpr::TableBuilder::sources)
      .def_readwrite("stub_indexer", &fpr::TableBuilder::stub_indexer)
      .def_readwrite("full_block_tables", &fpr::TableBuilder::full_block_tables)
      // the address of this builder as fpr::ITables*, for nd / p0::build_mixed (the
      // engine's ONE tb_tail / D13.tb instance shared with Python: one D12 `sent_` state). Borrowed: the Python object owns it.
      .def("save_state", [](const fpr::TableBuilder& tb) { return TbState{tb.save_state()}; })
      .def("restore_state", [](fpr::TableBuilder& tb, const TbState& st) {
        try {
          tb.restore_state(st.s);
        } catch (const std::invalid_argument& e) {
          raise(PyExc_ValueError, e.what());
        }
      })
      .def("itables_ptr", [](fpr::TableBuilder& tb) {
        return (int64_t)reinterpret_cast<uintptr_t>(static_cast<fpr::ITables*>(&tb));
      })
      .def("build", [](fpr::TableBuilder& tb, py::tuple sbt, py::object row_tokens, py::object tokfn, py::dict rids) {
        fpr::StepBatch sb = sb_of(sbt);
        std::vector<int64_t> rt;
        const bool has_rt = !row_tokens.is_none();
        if (has_rt) rt = row_tokens.cast<std::vector<int64_t>>();
        tb.tokens_of = tokens_fn(tokfn, rids);
        fpr::RankTables t;
        std::string err;
        fpr::Status st = tb.build(sb, has_rt ? &rt : nullptr, &t, &err);
        tb.tokens_of = nullptr;
        if (st != fpr::Status::OK) raise(PyExc_AssertionError, err);
        return rt_dict(t);
      })
      .def("build_compact", [](fpr::TableBuilder& tb, py::tuple sbt, int32_t k1, py::object row_tokens, py::object tokfn, py::dict rids) {
        fpr::StepBatch sb = sb_of(sbt);
        std::vector<int64_t> rt;
        const bool has_rt = !row_tokens.is_none();
        if (has_rt) rt = row_tokens.cast<std::vector<int64_t>>();
        tb.tokens_of = tokens_fn(tokfn, rids);
        fpr::CompactTables ct;
        std::string err;
        fpr::Status st = tb.build_compact(sb, k1, has_rt ? &rt : nullptr, &ct, &err);
        tb.tokens_of = nullptr;
        if (st != fpr::Status::OK) raise(PyExc_AssertionError, err);
        py::dict d;
#define F(x) d[#x] = arr(ct.x)
        F(slot); F(row0); F(n_rows); F(kind); F(start); F(replay_start); F(cached_upto); F(win_slot); F(win_blk); F(win_page);
        F(bt_upd_slot); F(bt_upd_start); F(bt_upd_len); F(bt_upd_pages); F(tokens); F(tail_rows); F(sample_idx); F(targets);
        F(uid); F(last_chunk); F(score); F(forced);
#undef F
        return d;
      });
}
