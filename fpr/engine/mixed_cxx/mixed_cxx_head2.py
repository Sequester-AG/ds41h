"""ENG_MIXED_HEAD_KIND=prog with ENG_MIXED_CXX_HEAD=1: the HSD head of a mixed
step as RECORDED OP PROGRAMS issued from C++, phase H2. H1 (mixed_cxx_head.py) unchanged otherwise: same eligibility, key
admission cap, per-step staging (set_step: the sample-row gather + the aux copy) and engine calls; MixedHead2 replaces
the captured graph of HSDT.run(r, None, None) by a program of the same issue sequence:

  after_sync (the key's eager step done): per rank, under a THROWAWAY capture into eng.gpool[r] (nothing runs, as H1's
  capture), HSDT.run(r, None, None) with every recording source on:
    - C ABI entries: the rule-9 hooks (fpr_<ext>_set_record) of each loaded extension that exports one

    - dispatcher ops (TorchDispatchMode): in-place / out= ops and custom ops (_flashmla_C::sparse_decode_fwd) as called,
      functional ATen ops through their out= overload into the recorded result (held), views / empty allocations not
      issued;
    - torch.cuda.Event record / wait, Stream.wait_event (wait_stream goes through them).
  Safety net (any hit aborts the key; it stays on the Python head):
    - sys.setprofile: a call into a loaded C extension of ours (not torch / triton / vllm) must be a hooked one that made
      a record, or a packer-wrapped moe_v4 call; a dispatcher op inside a hooked call after its record is refused;
    - no host tensor operands, no .item() / torch.tensor(...); an EV_WAIT must follow its EV_RECORD in the program;
    - a raw pointer (C ABI Args word, Triton pointer value) or a tensor operand inside an output a custom op allocated
      during the recording (a buffer the replay does not reproduce) is refused.
  The throwaway graphs are kept (never destroyed: the engine's rule for graphs sharing its pool).
Status: not yet GPU-gated.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mixed_cxx_head as H1  # noqa: E402

W = H1.W
_X = None
HELPERS = {n for n in os.environ.get("ENG_MIXED_HEAD2_HELPERS", "").split(",") if n}   # non-launching ext calls (names)
# H2g: each recorded program stream-captured into a CUDA graph by C++ (head_h2 capture(): into the
# engine's graph pool, thread-local mode, nothing runs); head() / fpr_head2_c then launch the graph. A failed capture
# leaves that key on its program (logged); captured graphs are never destroyed
GRAPH = os.environ.get("ENG_MIXED_HEAD2_GRAPH", "0") == "1"
_SKIP_MOD = ("torch", "triton", "__triton", "cuda_utils", "vllm", "numpy", "_C")
_SAFE_PYBIND = ("torch", "triton", "vllm", "numpy", "flash_mla", "_flashmla", "fpr_mixed_head2")   # pybind modules not ours
_SYS_DIRS = ("/usr/", "/lib", "/opt/", "/etc/", "/user-environment/", "/capstor/", "/proc/")


def ext():
    global _X
    if _X is None:
        from torch.utils.cpp_extension import load
        bd = os.environ.get("ENG_MIXED_HEAD2_BUILD") or os.path.join(HERE, "build", "head_h2")
        os.makedirs(bd, exist_ok=True)
        # triton_replay.h: through fpr_cu_launch.h (cudart's driver entry point, no -lcuda) or cuLaunchKernelEx (-lcuda)
        ld = ["-ldl"] + ([] if "fpr_cu_launch.h" in open(os.path.join(HERE, "triton_replay.h")).read() else ["-lcuda"])
        _X = load(name="fpr_mixed_head2", sources=[os.path.join(HERE, "head_h2.cpp")], extra_include_paths=[HERE],
                  build_directory=bd, extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"],
                  extra_ldflags=ld, with_cuda=True, verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
    return _X


def _loaded_exts() -> dict:
    """module name -> .so path of every loaded shared object that is not the interpreter's / pip's / the system's /
    Triton's: torch.utils.cpp_extension.load() does not put its modules in sys.modules, so /proc/self/maps is the list.
    A pybind module's functions carry __module__ = its name = the .so's basename (load(name=...))."""
    sysdirs = tuple(os.path.realpath(d) + os.sep for d in {sys.prefix, sys.base_prefix, sys.exec_prefix} if d)
    tcache = os.path.realpath(os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton")) + os.sep
    out, dup = {}, set()
    try:
        lines = open("/proc/self/maps").read().splitlines()
    except OSError:
        return out
    for ln in lines:
        f = ln.split()
        if len(f) < 6 or not f[5].endswith(".so"):
            continue
        rf = os.path.realpath(f[5])
        if "-packages/" in rf or rf.startswith(sysdirs) or rf.startswith(tcache) or rf.startswith(_SYS_DIRS):
            continue
        name = os.path.basename(rf)[:-3].split(".")[0]
        if name.startswith(_SKIP_MOD) or name == "fpr_mixed_head2":
            continue
        if name in out and out[name] != rf:
            dup.add(name)
        out[name] = rf
    for n in dup:
        out[n] = None   # two different .so files under one module name: no hook can be attributed (calls refused)
    return out


def _is_pybind(fn) -> bool:
    """A pybind11 function: its __self__ is the function record (pybind11 >= 3: pybind11_builtins.
    pybind11_detail_function_record_*; older: a PyCapsule)."""
    t = type(getattr(fn, "__self__", None))
    return t.__name__ == "PyCapsule" or (t.__module__ == "pybind11_builtins" and "function_record" in t.__name__)


def _dyn_syms(so: str) -> set:
    try:
        out = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return set()
    return {ln.split()[-1] for ln in out.splitlines() if ln.split()}


def _tensors(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _tensors(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _tensors(y, out)
    return out


def _span(t):
    """[lo, hi) bytes a tensor's elements touch (None if empty)."""
    if t.numel() == 0:
        return None
    lo = t.data_ptr()
    return lo, lo + (sum((t.size(i) - 1) * abs(t.stride(i)) for i in range(t.dim())) + 1) * t.element_size()


def _out_variant(func):
    """The out= overload of a functional ATen op: (OpOverload, out arg name, its kwarg names) or None."""
    pk = func.overloadpacket
    ov = func._overloadname
    names = pk.overloads()
    for cand in ([ov + "_out"] if ov != "default" else []) + ["out"]:
        if cand not in names:
            continue
        o = getattr(pk, cand)
        outs = [a.name for a in o._schema.arguments if a.is_out]
        if len(outs) == 1:
            return o, outs[0], {a.name for a in o._schema.arguments}
    return None


_NOOP = {"aten::detach", "aten::alias", "aten::_unsafe_view", "aten::lift_fresh"}   # lift_fresh: refused below (host data)
_ALLOC = {"aten::empty", "aten::empty_strided", "aten::empty_like", "aten::new_empty", "aten::new_empty_strided"}
_REFUSE = {"aten::_local_scalar_dense": "a host read (.item())", "aten::lift_fresh": "a host tensor (torch.tensor(...))",
           "aten::lift_fresh_copy": "a host tensor (torch.tensor(...))", "aten::record_stream": "record_stream"}


class _Rec:
    """One rank's recording (into program handle h)."""

    def __init__(self, x, dev, main_stream, label):
        self.x, self.dev = x, dev
        grad = 0 if torch.is_inference_mode_enabled() else (1 if not torch.is_grad_enabled() else 2)
        self.h = x.prog_new(torch.device(dev).index, main_stream.cuda_stream, grad, label)
        self.err = None
        self.errs = []
        self.evs = {}            # id(torch event) -> (program event index, the event: held, no id reuse)
        self.ev_rec = set()      # program events recorded so far
        self.vol = []            # [lo, hi, op index, name, StorageWeakRef, op count at expiry]: outputs custom ops
                                 # allocated while recording (a hazard only while that storage lives)
        self.suspend = 0         # inside a packer-recorded call (its own dispatcher ops are part of the C entry)
        self.in_ext = None       # (module, fn, rec_seq at entry) while inside a hooked C extension call
        self.n_boxed = self.n_tr = self.n_pk = 0

    def fail(self, msg):
        if self.err is None:
            self.err = msg
        if msg not in self.errs and len(self.errs) < 64:   # the census: every distinct reason (one run lists them all)
            self.errs.append(msg)

    def cur(self) -> int:
        return torch.cuda.current_stream(self.dev).cuda_stream

    def n_ops(self) -> int:
        return len(self.x.ops(self.h))

    def ev(self, e) -> int:
        k = id(e)
        if k not in self.evs:
            self.evs[k] = (self.x.new_event(self.h), e)
        return self.evs[k][0]

    def tick(self):
        """Retire the custom-op outputs whose storage was freed (a later overlap is then the allocator reusing the
        block, not a read of the output)."""
        if not self.vol:
            return
        n = None
        for v in self.vol:
            if v[5] is None and v[4].expired():
                if n is None:
                    n = self.n_ops()
                v[5] = n

    def check_vol(self, ts, what):
        if not self.vol:
            return
        self.tick()
        for t in ts:
            s = _span(t)
            if s is None:
                continue
            for lo, hi, _, nm, _, exp in self.vol:
                if exp is None and s[0] < hi and lo < s[1]:
                    self.fail(f"{what} reads an output of {nm} allocated while recording (not reproduced by the replay)")

    def finish(self):
        """Program-level checks once the rank's recording is complete: a raw pointer (C ABI Args word, Triton value)
        into a custom op's output while that output lived."""
        if not self.vol:
            return
        ops = self.x.ops(self.h)
        for i, (kind, _, name) in enumerate(ops):
            if kind not in (1, 2):
                continue
            words = self.x.op_words(self.h, i)
            for lo, hi, j, nm, _, exp in self.vol:
                if j < i and (exp is None or i < exp) and any(lo <= w < hi for w in words):
                    self.fail(f"{name} (op {i}) points into an output of {nm} allocated while recording")


class _Mode(TorchDispatchMode):
    def __init__(self, R):
        super().__init__()
        self.R = R

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        R = self.R
        if R.suspend:          # (after a failure too: the census keeps checking)
            return out
        R.tick()
        try:
            self._rec(func, args, kwargs, out)
        except Exception as e:  # noqa: BLE001
            R.fail(f"{func}: {type(e).__name__}: {e}")
        return out

    def _rec(self, func, args, kwargs, out):
        R = self.R
        qn = func._schema.name
        if qn in _REFUSE:
            return R.fail(f"{qn}: {_REFUSE[qn]}")
        if func.is_view or qn in _NOOP:
            return
        if qn in _ALLOC:
            if isinstance(out, torch.Tensor):
                R.x.keep(R.h, out)
            return
        if R.in_ext is not None and R.x.rec_seq() > R.in_ext[2]:
            return R.fail(f"{qn} inside {R.in_ext[0]}.{R.in_ext[1]} after its C ABI record")
        ins = _tensors(args, []) + _tensors(kwargs, [])
        for t in ins:
            if t.device.type == "cpu" and t.dim() > 0:
                return R.fail(f"{qn}: a host tensor operand {tuple(t.shape)}")
        R.check_vol(ins, qn)
        ns = qn.split("::")[0]
        if ns != "aten" or func._schema.is_mutable:
            R.x.add_boxed(R.h, R.cur(), qn, func._overloadname if func._overloadname != "default" else "", tuple(args),
                          dict(kwargs))
            R.n_boxed += 1
            if ns != "aten":
                ptrs = {t.untyped_storage().data_ptr() for t in ins}
                for t in _tensors(out, []):
                    s = _span(t)
                    if s is not None and t.untyped_storage().data_ptr() not in ptrs:
                        R.vol.append([s[0], s[1], R.n_ops(), qn, StorageWeakRef(t.untyped_storage()), None])
            return
        if not isinstance(out, torch.Tensor):
            return R.fail(f"{qn}: functional op with {type(out).__name__} outputs")
        v = _out_variant(func)
        if v is None:
            return R.fail(f"{qn}.{func._overloadname}: functional op without an out= overload")
        o, oname, allowed = v
        kw = {k: x for k, x in kwargs.items() if k in allowed}
        kw[oname] = out
        R.x.add_boxed(R.h, R.cur(), qn, o._overloadname, tuple(args), kw)
        R.x.keep(R.h, out)
        R.n_boxed += 1


class _Sources:
    """The loaded C extensions of ours: their rule-9 hooks, the packer-wrapped moe_v4 modules (built once)."""

    def __init__(self, eng, x):
        self.x = x
        self.mods = {}           # module name -> (so, [hooked ext])
        self.cabi = []           # (so, ext) for rec_begin
        unhooked = []
        for name, f in sorted(_loaded_exts().items()):
            if f is None:
                self.mods[name] = (None, [])
                unhooked.append(name + " (2 builds)")
                continue
            syms = _dyn_syms(f)
            exts = sorted(s[4:-6] for s in syms if s.startswith("fpr_") and s.endswith("_table"))
            hooked = [e for e in exts if f"fpr_{e}_set_record" in syms]
            self.mods[name] = (f, hooked)
            self.cabi += [(f, e) for e in hooked]
            if not hooked:
                unhooked.append(name)
        self.moe = None
        try:
            sys.path.insert(0, os.path.join(PHYS, "kernels", "moe_v4"))
            import v4_cabi as V4C
            DR = next((m for m in list(sys.modules.values()) if m is not None and hasattr(m, "v4_ext")
                       and getattr(m, "__name__", "").endswith("drafter")), None)
            assert DR is not None, "the engine's drafter module (v4_ext) is not loaded"
            api, route = V4C.load()
            e4, r4 = DR.v4_ext()
            names = {nm for so in (api.__file__, route.__file__) for t in ("moe_v4", "moe_v4r")
                     for nm, _ in self._names(so, t)}
            for d in eng.devs:   # moe_v4.init / moe_v4r.init on each device (their SMEM attributes), before any replay
                with torch.cuda.device(d):
                    di = torch.device(d).index
                    assert x.call_cabi(api.__file__, "moe_v4", "moe_v4.init", V4C.pack_init(di)) == 0
                    assert x.call_cabi(route.__file__, "moe_v4r", "moe_v4r.init", V4C.pack_init(di)) == 0
            need = {"moe_v4.gemm1r", "moe_v4.gemm2r", "moe_v4.swiglu", "moe_v4.prepr6", "moe_v4.combine", "moe_v4r.route2r"}
            assert need <= names, f"moe_v4c tables lack {sorted(need - names)}"
            self.moe = dict(V4C=V4C, api=api.__file__, route=route.__file__, e4=e4, r4=r4)
        except Exception as e:  # noqa: BLE001
            eng.log(f"mixed_cxx_head2: moe_v4 C ABI unavailable ({type(e).__name__}: {e}); keys whose head reaches the "
                    f"drafter MoE abort their recording")
        eng.log(f"mixed_cxx_head2: rule-9 hooks {[e for _, e in self.cabi]}; C extensions without one {unhooked}")

    def _names(self, so, ext_):
        try:
            return self.x.table_names(so, ext_)
        except Exception:  # noqa: BLE001
            return []


class _Recording:
    """Every recording source on for one rank's HSDT.run (all restored on exit)."""

    def __init__(self, R, S):
        self.R, self.S = R, S

    def __enter__(self):
        self.o_ev = self.J = self.mode = None
        self.pm, self.hooks_on, self.prof_on = [], False, False
        try:
            return self._enter()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _enter(self):
        R, S, X = self.R, self.S, self.R.x
        E, St = torch.cuda.Event, torch.cuda.Stream
        self.o_ev = (E.record, E.wait, St.wait_event)
        o_rec, o_wait, o_swe = self.o_ev

        def ev_record(ev, stream=None):
            s = stream if stream is not None else torch.cuda.current_stream(R.dev)
            i = R.ev(ev)
            X.add_ev(R.h, s.cuda_stream, i, False)
            R.ev_rec.add(i)
            return o_rec(ev, stream)

        def ev_wait(ev, stream=None):
            s = stream if stream is not None else torch.cuda.current_stream(R.dev)
            i = R.ev(ev)
            if i not in R.ev_rec:
                R.fail("an event wait on an event not recorded in the head (an external dependency)")
            X.add_ev(R.h, s.cuda_stream, i, True)
            return o_wait(ev, stream)

        def st_wait_event(st, event):
            i = R.ev(event)
            if i not in R.ev_rec:
                R.fail("a stream wait on an event not recorded in the head (an external dependency)")
            X.add_ev(R.h, st.cuda_stream, i, True)
            return o_swe(st, event)
        E.record, E.wait, St.wait_event = ev_record, ev_wait, st_wait_event
        # ---- Triton (triton_replay's recording, appended to the program in call order)
        from triton import knobs
        from triton.runtime import driver
        from triton.runtime import jit as J
        import triton_replay as TR
        self.J, self.o_run = J, J.JITFunction.run
        o_run = self.o_run

        def run(fn, *args, grid, warmup, **kwargs):
            kernel = o_run(fn, *args, grid=grid, warmup=warmup, **kwargs)
            if warmup or kernel is None or R.suspend:
                return kernel
            try:
                kw = dict(kwargs)
                kw["debug"] = kw.get("debug", fn.debug) or knobs.runtime.debug
                kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
                device = driver.active.get_current_device()
                bound, specialization, _ = fn.device_caches[device][4](*args, **kw)
                g = grid(bound) if callable(grid) else grid
                rec = TR.Rec(fn, kernel, device, bound, specialization, g)
                vals = rec.values()
                ptrs = [t for t in bound.values() if isinstance(t, torch.Tensor)]
                R.check_vol(ptrs, f"triton {fn.__name__}")
                X.add_triton(R.h, R.cur(), bytes(rec.record), vals, *rec.grid)
                R.n_tr += 1
            except Exception as e:  # noqa: BLE001
                R.fail(f"triton {getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}")
            return kernel
        J.JITFunction.run = run
        self.pm = []
        if S.moe is not None:
            M = S.moe
            V4C = M["V4C"]
            for mod, fn, so, ext_, pack in ((M["e4"], "gemm1r", M["api"], "moe_v4", V4C.pack_gemm1r),
                                            (M["e4"], "gemm2r", M["api"], "moe_v4", V4C.pack_gemm2r),
                                            (M["e4"], "swiglu", M["api"], "moe_v4", V4C.pack_swiglu),
                                            (M["e4"], "prepr6", M["api"], "moe_v4", V4C.pack_prepr6),
                                            (M["e4"], "combine", M["api"], "moe_v4", V4C.pack_combine),
                                            (M["r4"], "route2r", M["route"], "moe_v4r", V4C.pack_route2r_served)):
                orig = getattr(mod, fn)

                def wrap(*a, _o=orig, _so=so, _ext=ext_, _nm=f"{ext_}.{fn}", _pack=pack, **k):
                    if True:   # (after a failure too: the census)
                        try:
                            R.check_vol(_tensors(a, []), _nm)
                            X.add_cabi(R.h, _so, _ext, _nm, bytes(_pack(*a, **k)))
                            R.n_pk += 1
                        except Exception as e:  # noqa: BLE001
                            R.fail(f"{_nm} packer: {type(e).__name__}: {e}")
                    R.suspend += 1
                    try:
                        return _o(*a, **k)
                    finally:
                        R.suspend -= 1
                setattr(mod, fn, wrap)
                self.pm.append((mod, fn, orig))
        self.packed = {m.__name__ for m, _, _ in self.pm}
        # ---- the rule-9 hooks, the dispatcher mode, the C extension call check
        X.rec_begin(R.h, S.cabi)
        self.hooks_on = True
        mode = _Mode(R)
        mode.__enter__()
        self.mode = mode
        self.o_prof = sys.getprofile()
        mods = S.mods
        packed = self.packed

        def prof(frame, event, arg):
            if event == "c_call":
                mod = getattr(arg, "__module__", None)
                if mod not in mods:
                    if isinstance(mod, str) and not mod.startswith(_SAFE_PYBIND) and _is_pybind(arg):
                        R.fail(f"{mod}.{getattr(arg, '__name__', '?')}: a pybind call into a module of no loaded .so "
                               f"(its launches cannot be attributed to a rule-9 hook)")
                    return
                if R.vol:
                    R.tick()
                nm = getattr(arg, "__name__", "?")
                if mod in packed and R.suspend:
                    return
                if nm in HELPERS:
                    return
                if not mods[mod][1]:
                    R.fail(f"{mod}.{nm}: a C extension call without a rule-9 hook (its launches would be lost)")
                elif R.in_ext is None:
                    R.in_ext = (mod, nm, X.rec_seq())
            elif event in ("c_return", "c_exception"):
                mod = getattr(arg, "__module__", None)
                ie = R.in_ext
                if ie is None or mod != ie[0] or getattr(arg, "__name__", "?") != ie[1]:
                    return
                if X.rec_seq() == ie[2]:
                    R.fail(f"{ie[0]}.{ie[1]}: a hooked C extension call that made no record")
                R.in_ext = None
        sys.setprofile(prof)
        self.prof_on = True
        return self

    def __exit__(self, *exc):
        R, X = self.R, self.R.x
        if self.prof_on:
            sys.setprofile(self.o_prof)
        try:
            if self.mode is not None:
                self.mode.__exit__(*exc)
        finally:
            if self.hooks_on:
                err = X.rec_end()
                if err:
                    R.fail(err)
            if self.o_ev is not None:
                E, St = torch.cuda.Event, torch.cuda.Stream
                E.record, E.wait, St.wait_event = self.o_ev
            if self.J is not None:
                self.J.JITFunction.run = self.o_run
            for mod, fn, orig in self.pm:
                setattr(mod, fn, orig)
        return False


class MixedHead2(H1.MixedHead):
    """H1 with programs instead of graphs (head_eager inherited: set_step + head(rank, handle) of fpr_mixed_head2)."""

    def __init__(self, hooks):
        super().__init__(hooks, x=ext())
        self.bad = set()
        self.src = None
        self.kept = []           # throwaway capture graphs (never destroyed: they share the engine's pool)
        self.stats["kind_prog"] = 1
        self.eng.log("mixed_cxx_head2: H2 on (recorded op programs instead of captured graphs)")

    def after_sync(self, st, batches) -> None:
        k, self.pending = self.pending, None
        if k is None or k in self.graphs or k in self.bad or len(self.graphs) >= H1.MAX_KEYS or self._key() != k:
            return
        if self._rows(st) is None:
            return
        hsd, eng = self.hsd, self.eng
        tabs = [hsd._tables(r, k) for r in range(W)]
        if any(hsd.tab[r] is not tabs[r] or tabs[r].get("use_bm") for r in range(W)):
            return
        if self.src is None:
            self.src = _Sources(eng, self.x)
        mem0 = [torch.cuda.memory_reserved(d) for d in self.devs]
        t0 = time.perf_counter()
        recs, gs = [], []
        try:
            for r in range(W):
                dev = self.devs[r]
                with torch.cuda.device(dev), torch.cuda.stream(eng.streams[r]):
                    R = _Rec(self.x, dev, eng.streams[r], f"head {tuple(k)} r{r}")
                    recs.append(R)
                    g = torch.cuda.CUDAGraph()
                    g.capture_begin(pool=eng.gpool[r], capture_error_mode="thread_local")
                    own = "_steps" in hsd.__dict__
                    o_steps = hsd._steps

                    def steps(*a, _o=o_steps, _h=R.h, **kw):   # a stage boundary after each yield's collective
                        for req in _o(*a, **kw):
                            yield req
                            self.x.mark_seg(_h)
                    hsd._steps = steps
                    try:
                        with _Recording(R, self.src):
                            hsd.run(r, None, None)
                    finally:
                        if own:
                            hsd._steps = o_steps
                        else:
                            del hsd._steps
                        g.capture_end()
                        gs.append(g)
                    R.finish()
                    if R.err:
                        raise RuntimeError(f"rank {r}: {R.err}")
            for d in self.devs:
                torch.cuda.synchronize(d)
        except Exception as e:  # noqa: BLE001  (this key stays on the Python head)
            self.kept += gs
            for R in recs:
                self.x.prog_free(R.h)
            self.bad.add(k)
            self.seen[k] = -10 ** 9
            self.stats["record_aborted"] += 1
            eng.log(f"mixed_cxx_head2: recording of key {k} aborted ({type(e).__name__}: {str(e)[:300]}); Python head stays")
            for R in recs:
                for m in R.errs:
                    eng.log(f"mixed_cxx_head2:   census key {k} {R.dev}: {m[:300]}")
            eng.stats.setdefault("mixed_head2_census", []).append(dict(key=list(k), errs=[R.errs for R in recs]))
            return
        self.kept += gs
        self.graphs[k] = (recs, [R.h for R in recs])
        self.stats["recorded"] += 1
        ms = (time.perf_counter() - t0) * 1e3
        mem1 = [torch.cuda.memory_reserved(d) for d in self.devs]
        n = [len(self.x.ops(R.h)) for R in recs]
        cap = dict(key=list(k), ms=round(ms, 1), ops=n, segs=[len(self.x.segs(R.h)) for R in recs],
                   streams=[self.x.n_streams(R.h) for R in recs],
                   boxed=[R.n_boxed for R in recs], triton=[R.n_tr for R in recs], packed=[R.n_pk for R in recs],
                   reserved_delta_MB=[round((b - a) / 1e6, 1) for a, b in zip(mem0, mem1)])
        if GRAPH:
            self._capture(k, recs, cap, mem1)
        eng.stats.setdefault("mixed_head_caps", []).append(cap)
        eng.log(f"mixed_cxx_head2: recorded key {k} ({len(self.graphs)} / {H1.MAX_KEYS}) in {ms:.0f} ms: ops per rank {n} "
                f"(boxed {[R.n_boxed for R in recs]}, triton {[R.n_tr for R in recs]}, moe packed {[R.n_pk for R in recs]}; "
                f"stages {[len(self.x.segs(R.h)) for R in recs]})")

    def _capture(self, k, recs, cap, mem1) -> None:
        """H2g: the key's 4 programs into CUDA graphs (C++ capture into eng.gpool[r]); on failure the programs stay."""
        eng = self.eng
        gms = []
        try:
            for r, R in enumerate(recs):
                with torch.cuda.device(self.devs[r]):
                    pa, pb = (int(v) for v in eng.gpool[r])
                    gms.append(round(self.x.capture(R.h, pa, pb), 1))
            for d in self.devs:
                torch.cuda.synchronize(d)
        except Exception as e:  # noqa: BLE001
            self.stats["graph_capture_failed"] += 1
            cap["graph_failed"] = f"{type(e).__name__}: {str(e)[:300]}"
            eng.log(f"mixed_cxx_head2: graph capture of key {k} failed ({cap['graph_failed']}); "
                    f"{'its captured ranks launch graphs, the others their programs' if gms else 'programs stay'}")
            return
        mem2 = [torch.cuda.memory_reserved(d) for d in self.devs]
        self.stats["captures"] += 1
        cap.update(graph_ms=gms, graph_reserved_delta_MB=[round((b - a) / 1e6, 1) for a, b in zip(mem1, mem2)])
        eng.log(f"mixed_cxx_head2: captured key {k}'s programs as graphs: ms per rank {gms}; reserved +MB "
                f"{cap['graph_reserved_delta_MB']}; reserved GB {[round(b / 1e9, 2) for b in mem2]}")
