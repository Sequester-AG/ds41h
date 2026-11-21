// fpr_h1scan: nd_export.h1_guard_scan's per-slot check in C++.
// check(ents) -> list[int]: indices of the entries mixed_cxx_head.guard_check([e]) MAY reject. Conservative by construction:
// an entry is reported OK only when it passes exactly guard_check's rule on exact builtin containers / plain __dict__ objects;
// Entry layout = guard_record's: (par, kind, key, ch, ptr, sptr, snb, path).
//   kind 0: par.get(key, MISS)   kind 1: par[key] if key < len(par) else MISS   kind 2: par.__dict__.get(key, MISS)
//   OK  iff  cur is ch and (ptr is None or ch.data_ptr() == ptr)
//        or  ptr is not None and cur is a tensor and cur.data_ptr() == ptr and its storage (data_ptr, nbytes) == (sptr, snb)
#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>

namespace {

// 1 = OK, 0 = report (Python decides)
int one(PyObject* e) {
    if (!PyTuple_CheckExact(e) || PyTuple_GET_SIZE(e) < 7) return 0;
    PyObject* par = PyTuple_GET_ITEM(e, 0);
    PyObject* kindo = PyTuple_GET_ITEM(e, 1);
    PyObject* key = PyTuple_GET_ITEM(e, 2);
    PyObject* ch = PyTuple_GET_ITEM(e, 3);
    PyObject* ptro = PyTuple_GET_ITEM(e, 4);
    if (!PyLong_CheckExact(kindo)) return 0;
    long kind = PyLong_AsLong(kindo);
    PyObject* cur = nullptr;     // borrowed; nullptr = MISS
    if (kind == 0) {
        if (!PyDict_CheckExact(par)) return 0;
        cur = PyDict_GetItemWithError(par, key);
        if (cur == nullptr && PyErr_Occurred()) { PyErr_Clear(); return 0; }
    } else if (kind == 1) {
        if (!PyLong_CheckExact(key)) return 0;
        Py_ssize_t i = PyLong_AsSsize_t(key);
        if (i == -1 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
        if (i < 0) return 0;     // (Python would index from the end: leave it to Python)
        if (PyList_CheckExact(par)) cur = i < PyList_GET_SIZE(par) ? PyList_GET_ITEM(par, i) : nullptr;
        else if (PyTuple_CheckExact(par)) cur = i < PyTuple_GET_SIZE(par) ? PyTuple_GET_ITEM(par, i) : nullptr;
        else return 0;
    } else if (kind == 2) {
        PyObject** dp = _PyObject_GetDictPtr(par);
        if (dp == nullptr || *dp == nullptr || !PyDict_CheckExact(*dp)) return 0;
        cur = PyDict_GetItemWithError(*dp, key);
        if (cur == nullptr && PyErr_Occurred()) { PyErr_Clear(); return 0; }
    } else {
        return 0;
    }
    if (cur == nullptr) return 0;     // MISS: guard_check rejects (ch is never MISS)
    if (ptro != Py_None && !PyLong_CheckExact(ptro)) return 0;
    if (cur == ch) {
        if (ptro == Py_None) return 1;
        if (!THPVariable_Check(ch)) return 0;
        const at::Tensor& t = THPVariable_Unpack(ch);
        unsigned long long p = PyLong_AsUnsignedLongLong(ptro);
        if (PyErr_Occurred()) { PyErr_Clear(); return 0; }
        return (unsigned long long)(uintptr_t)t.data_ptr() == p ? 1 : 0;
    }
    if (ptro == Py_None || !THPVariable_Check(cur)) return 0;
    PyObject* sptro = PyTuple_GET_ITEM(e, 5);
    PyObject* snbo = PyTuple_GET_ITEM(e, 6);
    if (!PyLong_CheckExact(sptro) || !PyLong_CheckExact(snbo)) return 0;
    unsigned long long p = PyLong_AsUnsignedLongLong(ptro);
    unsigned long long sp = PyLong_AsUnsignedLongLong(sptro);
    unsigned long long sn = PyLong_AsUnsignedLongLong(snbo);
    if (PyErr_Occurred()) { PyErr_Clear(); return 0; }
    const at::Tensor& t = THPVariable_Unpack(cur);
    if ((unsigned long long)(uintptr_t)t.data_ptr() != p) return 0;
    if (!t.has_storage()) return 0;
    const c10::Storage& st = t.storage();
    if ((unsigned long long)(uintptr_t)st.data() != sp) return 0;
    if ((unsigned long long)st.nbytes() != sn) return 0;
    return 1;
}

py::list check(py::list ents) {
    py::list bad;
    PyObject* L = ents.ptr();
    Py_ssize_t n = PyList_GET_SIZE(L);
    for (Py_ssize_t i = 0; i < n; ++i) {
        int ok = 0;
        try {
            ok = one(PyList_GET_ITEM(L, i));
        } catch (...) {
            ok = 0;
        }
        if (PyErr_Occurred()) PyErr_Clear();
        if (!ok) bad.append(py::int_(i));
    }
    return bad;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check", &check, "indices of guard entries the Python guard_check must re-check");
    m.attr("abi") = 1;
}
