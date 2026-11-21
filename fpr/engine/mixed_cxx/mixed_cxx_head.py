"""ENG_MIXED_CXX_HEAD: the HSD head of a MIXED step from C++, phase H1

Engine.step_mixed ends with Hooks.head_eager(stT, batches): per rank the sample rows of bucket T's final hidden
(index_select), then ONE hsd.run_all (hsdt_c1e.HSDT, stage-interleaved over the ranks; device-side collectives only, no
host sync). H1 issues the same work per rank from C++ (head_h1.cpp, module fpr_mixed_head): the gather into hsd.h_own[r],
the copy of the aux rows into the drafter's aux buffer, then the rank's captured graph of HSDT.run(r, None, None) for the
step's HSD key (S_pad, D_pad, NT): what graph H holds for decode keys. 4 calls instead of Python's ~400-600 launches.

    head_eager(hooks, st, batches) -> bool    top of Hooks.head_eager (mixed steps: batches is not None): True = issued
    after_sync(hooks, st, batches)            right after the mixed step's engine.sync(): captures a pending key
Engine side: both calls guarded by ENG_MIXED_CXX_HEAD; nothing else changes.

Policy:
  - eligible: HSD on, no grammar / bitmask, HSD's key without a graph variant (gvar == (): c1d is off in mixed steps),
    every rank's sample-row count == HSD's n_s and its aux rows contiguous. Anything else: today's Python head.
  - capture: after a key's ENG_MIXED_HEAD_MIN_SEEN-th eager step (default 2), at most ENG_MIXED_HEAD_GRAPHS keys (default
    8), into the engine's graph pool eng.gpool[r] on the rank's engine stream (as graph H; no new pool). Captured graphs
    are never destroyed (engine rule: graphs sharing a pool are retired, not freed): the cap bounds admission. Each
    capture logs its time and the pool's / device's reserved bytes.
  - guard: at capture, every CUDA tensor reachable from the HSD object (and every
    container on the way: dicts / lists / tuples / plain objects' attributes) is recorded with its slot and data_ptr;
    before EACH replay the slots are re-read (identity first, then data_ptr / storage). A changed slot means the graph
    holds a pointer the eager head no longer uses (possibly freed memory): log "H1 STALE key=.. arg=.. old=.. new=.."
    once per (key, slot), retire the graph (never replayed, never freed: the pool rule), run this step's Python head, and
    re-capture the key after MIN_SEEN eager steps at most ENG_MIXED_HEAD_GUARD_RECAP times (default 1), then Python.
    The records hold STRONG references to every recorded tensor and container ON PURPOSE: a buffer the
    eager head replaced stays allocated while a graph may still point at it, so its memory cannot be reused under a
    replay (a would-be IMA becomes a detected STALE). Do not make them weak references. Cost: only replaced tensors
    linger, until the key's next check retires its record. A slot that is legitimately re-created per prepare and NOT read
    by the graph goes into _GUARD_SKIP_ATTR, with a comment saying why the graph does not read it.
Bitwise: the gather and the aux copy move bits; the graph replays the eager run's kernels with the same arguments
(decode serving's graph H mechanism).
"""
from __future__ import annotations

import collections
import functools
import ctypes
import os
import time
import types

import torch
try:   # (stage 4a: graph capture timers; log-only)
    from engine import caplog as _CAPLOG
except Exception:   # noqa: BLE001
    _CAPLOG = None
_CAPLOG_PRE = None   # (stage 4a) the H1 precapture list: ENG_MIXED_HEAD_PRECAPTURE_KEYS + keys given to precapture()


def _caplog_listed(k) -> bool:
    """(stage 4a) k is on the precapture list: its lazy capture is a real miss (else lazy by design)."""
    global _CAPLOG_PRE
    try:
        if _CAPLOG_PRE is None:
            _CAPLOG_PRE = set()
            fn = os.environ.get("ENG_MIXED_HEAD_PRECAPTURE_KEYS", "")
            if fn and os.path.exists(fn):
                for ln in open(fn):
                    ln = ln.split("#", 1)[0].strip()   # (inline "# count" comments allowed)
                    if ln and not ln.startswith("#"):
                        _CAPLOG_PRE.add(tuple(int(x) for x in ln.split("|")[:3]))
        return len(k) == 3 and tuple(int(x) for x in k) in _CAPLOG_PRE   # (grammar 'bm' variants: never listed)
    except Exception:   # noqa: BLE001
        return False

HERE = os.path.dirname(os.path.abspath(__file__))
W = 4
ON = os.environ.get("ENG_MIXED_CXX_HEAD", "0") == "1"
MAX_KEYS = int(os.environ.get("ENG_MIXED_HEAD_GRAPHS", "8"))
MIN_SEEN = max(1, int(os.environ.get("ENG_MIXED_HEAD_MIN_SEEN", "2")))
MAX_NT = int(os.environ.get("ENG_MIXED_HEAD_MAX_NT", "0") or 0)   # keys with more tail
                          # rows (NT) are not captured: they run the eager head, as every uncaptured key (bitwise). The
                          # shared graph pool grows to the largest key's transients: (4, 1, 388) alone added 63 MB (soak)
POOLSTAT = os.environ.get("ENG_MIXED_HEAD_POOLSTAT", "1") == "1"
GUARD = os.environ.get("ENG_MIXED_HEAD_GUARD", "1") == "1"          # pointer-stability guard (docstring)
MAX_BM_KEYS = int(os.environ.get("ENG_MIXED_HEAD_MAX_BM_KEYS", "4"))   # the grammar ("bm") keys' own cap
                          # INSIDE MAX_KEYS: each key holds pool transients; JSON traffic must not crowd the plain keys out.
                          # Over it, a grammar step's head runs in Python
BM = os.environ.get("ENG_HEAD_BM_GRAPH", "0") == "1"   # grammar steps: HSD's bitmask
                          # variant (use_bm over its persistent _bm_d) as its own key (S_pad, D_pad, NT, "bm"); grammar rows
                          # without a sampled mask (bm_step False) run the plain variant under the plain key. Off: today's
                          # rule (any grammar -> the Python head)
GUARD_RECAP = int(os.environ.get("ENG_MIXED_HEAD_GUARD_RECAP", "1"))   # re-captures of a key after a STALE, then Python
LAZY = os.environ.get("ENG_MIXED_HEAD_LAZY", "1") == "1"
                          # 0 = never capture at serve time (after_sync admits nothing): with ENG_MIXED_HEAD_PRECAPTURE_KEYS=all
                          # every reachable key is captured at init; an uncaptured key then runs the Python head (logged)
MEMLOG_S = float(os.environ.get("ENG_MIXED_HEAD_MEMLOG_S", "0") or 0)
                          # slack) > 0: guard_ok logs torch reserved / allocated per GPU at most every MEMLOG_S s (log only)
_MEMLOG_T = [0.0]
# the epoch trip test
# right after the ENG_MIXED_HEAD_TRIP_AT-th successful mark_verified that follows a precapture (the moment the native fast path
# starts trusting the epoch), rebind hsd.h_own (per-rank tensors every captured H1 graph reads; recorded by the guard) to clones. TRIP_MODE=bump: through
# the bumping __setattr__ (expect: epoch moves, no fast skip, the Python guard finds STALE and retires); nobump: via
# object.__setattr__ (expect: VERIFY counts WOULD-SKIP-BUT-STALE >= 1 = the verify path catches a real hole). The old tensors
# stay referenced (_TRIP_KEEP) so a stale replay reads valid memory; HSD only drafts (verification is exact): tokens unchanged
TRIP_AT = int(os.environ.get("ENG_MIXED_HEAD_TRIP_AT", "0") or 0)
TRIP_MODE = os.environ.get("ENG_MIXED_HEAD_TRIP_MODE", "bump")
# > 0: trip at the first successful mark_verified at least TRIP_AFTER_S s after the
# precapture (instead of the TRIP_AT-th verify)
TRIP_AFTER_S = float(os.environ.get("ENG_MIXED_HEAD_TRIP_AFTER_S", "0") or 0)
_PRECAP_T = [0.0]
_TRIP_KEEP = []
_TRIP_N = [-1]                                  # successful mark_verified calls since the last precapture
_HSD_REF = []                                  # (the HSD object, set by MixedHead.__init__: mark_verified has no hooks arg)
_ENG_REF = []


def _trip() -> None:
    hsd, eng = (_HSD_REF or [None])[0], (_ENG_REF or [None])[0]
    if hsd is None or _TRIP_KEEP:
        return
    old = hsd.h_own
    _TRIP_KEEP.append(old)
    new = [t.clone() for t in old]
    e0 = int(_EP[0])
    if TRIP_MODE == "nobump":
        object.__setattr__(hsd, "h_own", new)
    else:
        hsd.h_own = new
    if eng is not None:
        eng.log(f"mixed_cxx_head: EPOCH TRIP ({TRIP_MODE}) at verify {_TRIP_N[0]}: hsd.h_own rebound to clones; epoch "
                f"{e0} -> {int(_EP[0])}, verified {int(_EP[1])}; t={time.time():.1f} ({time.time() - _PRECAP_T[0]:.0f} s after "
                f"the precapture)")
HIST_TOP = int(os.environ.get("ENG_MIXED_HEAD_HIST_TOP", "48"))   # uncaptured keys listed per summary line (0 = off)
GUARD_DEPTH = 7
_EP = (ctypes.c_uint64 * 4)()
# ENG_MIXED_HEAD_EPOCH=0: no magic -> NativeWorker.set_h1_epoch ignores the block (today's Python guard path)
_EP[3] = 0x48314550 if os.environ.get("ENG_MIXED_HEAD_EPOCH", "1") == "1" else 0


def epoch_addr() -> int:
    return ctypes.addressof(_EP)


def epoch() -> int:
    return int(_EP[0])


def bump() -> None:
    _EP[0] += 1


def mark_verified(ep: int, buckets_epoch: int) -> bool:
    """the exporter, after h1_guard_scan(mh) retired nothing and with ep = epoch() read BEFORE the scan: every published key
    is guard-checked as of ep. Refused (False) when anything bumped the epoch since ep."""
    if int(_EP[0]) != int(ep) or int(ep) == 0:
        return False
    _EP[2] = int(buckets_epoch)
    _EP[1] = int(ep)
    if (TRIP_AT > 0 or TRIP_AFTER_S > 0) and _TRIP_N[0] >= 0:
        _TRIP_N[0] += 1
        if (TRIP_AFTER_S > 0 and time.time() - _PRECAP_T[0] >= TRIP_AFTER_S) or (TRIP_AFTER_S <= 0 and _TRIP_N[0] == TRIP_AT):
            _trip()
    return True


# per-step HOST attributes of HSD (no device pointer a captured graph reads; already guard-skipped or plain ints / lists /
# numpy): rebinding them does not bump (nd_export sets _hi / _slot_key per published step; prepare sets cur / _res / bm_any)
_EP_HOST_ATTRS = frozenset(("cur", "_res", "_hi", "_slot_key", "bm_any", "_temp_np", "_seed_np", "_h1ep_wrapped", "tab"))


class _BumpList(list):
    """a list whose item writes bump the epoch"""

    def __setitem__(self, i, v):
        _EP[0] += 1
        super().__setitem__(i, v)


def _bump_setattr(self, name, value):
    if name not in _EP_HOST_ATTRS:
        _EP[0] += 1
        if name == "_bm_d" and isinstance(value, list) and not isinstance(value, _BumpList):
            value = _BumpList(value)
    object.__setattr__(self, name, value)


def scan_needed(buckets_epoch: int) -> bool:
    """False = the
    exporter's full H1 guard scan cannot find anything new: the epoch block is live (magic), nothing rebound an HSD slot since
    the last clean scan (epoch == verified != 0) and the handover's buckets epoch is the one recorded at that scan. True in
    every other case (no magic = ENG_MIXED_HEAD_EPOCH=0). The HONEST answer also under verify runs: nd's
    ND_H1_SCAN_VERIFY runs the full scan anyway on a would-skip and counts h1_scan_skip_miss if it retires anything."""
    if int(_EP[3]) != 0x48314550:
        return True
    e, v = int(_EP[0]), int(_EP[1])
    return not (v != 0 and e == v and int(_EP[2]) == int(buckets_epoch))


def _wrap_hsd(hsd) -> None:
    """instance-level wrappers that bump the epoch before HSD's slot-writing methods (functions are skipped by the guard walk),
    + (v5b) ANY attribute rebind on the HSD object (a per-instance subclass with a bumping __setattr__, host attributes
    excepted) and item writes of its _bm_d list (_BumpList): writers outside this file (nd_export's set_hsd_bm _bm_h /
    _bm_d[r]) move the epoch too."""
    if hsd is None or hsd.__dict__.get("_h1ep_wrapped"):
        return
    cls = type(hsd)
    if not getattr(cls, "_h1ep_cls", False):
        import sys
        sub = type(cls.__name__, (cls,), {"__setattr__": _bump_setattr, "_h1ep_cls": True, "__module__": cls.__module__,
                                          "__qualname__": cls.__qualname__})
        mod0 = sys.modules.get(cls.__module__)
        try:
            hsd.__class__ = sub
        except TypeError:                      # (a slotted / builtin layout: method wrappers only; logged by the caller)
            pass
        # served code resolves HSD's module from its class (d13_cxx L51 / L103, nd_export L904, gtrace_d13)
        # the swap must not change it
        assert type(hsd).__module__ == cls.__module__ and sys.modules.get(type(hsd).__module__) is mod0, \
            "mixed_cxx_head: HSD class swap changed its module"
    bd = hsd.__dict__.get("_bm_d")
    if isinstance(bd, list) and not isinstance(bd, _BumpList):
        hsd.__dict__["_bm_d"] = _BumpList(bd)
    # prepare / run / run_all: always (per-step writers).
    new_ = dict(draft_rank=lambda a: (a[0], a[1]) not in hsd.drk,
                _tables=lambda a: (a[0], tuple(a[1])) not in getattr(hsd, "_tab_cache", {}) or getattr(hsd, "prep", "1") == "0")
    for nm in ("prepare", "draft_rank", "_tables", "run", "run_all"):
        f = getattr(hsd, nm, None)
        if f is None:
            continue
        chk = new_.get(nm)

        def w(*a, _f=f, _chk=chk, **kw):
            if _chk is None or len(a) < 2 or _chk(a):
                _EP[0] += 1
            return _f(*a, **kw)
        w.__name__ = nm
        setattr(hsd, nm, w)
    hsd.__dict__["_h1ep_wrapped"] = True
SUMMARY_EVERY = int(os.environ.get("ENG_MIXED_HEAD_SUMMARY", "500") or 0)   # one counters line every N mixed
                          # heads (0 = off): C++ / Python heads by reason, captures, guard checks / us / stale
_GUARD_SKIP_ATTR = frozenset(("cur", "eng", "hooks", "H", "stats", "_mixed_head", "lane", "_dmask", "_rtag"))
                          # per-step host records / back references (hsd.cur is a new HsdStep every prepare: not graph
                          # memory); DraftRank._dmask / _rtag (hsdt.py _steps): launch-argument tuples rebuilt at the start
                          # of EVERY run from the running key's tables (hsd.tab[r]), so they hold the previous step's key at
                          # check time; a replay uses its own captured pointers, whose tables the hsd.tab / _tab_cache slots
                          # cover
_GUARD_SKIP_ROOT = frozenset(("tab",))   # HSD.tab[r] = the running step's key tables (set by Python's HSD.prepare; stale
                          # every key's tables dict is walked under hsd._tab_cache
_GUARD_SKIP_T = (torch.cuda.Stream, torch.cuda.Event, torch.nn.Module, type, type(os), str, bytes, int, float, bool,
                 complex, type(None))
_MISS = object()
_FN_T = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType, functools.partial)
KIND = os.environ.get("ENG_MIXED_HEAD_KIND", "graph")   # "graph" (H1) / "prog" (H2: mixed_cxx_head2.MixedHead2)
assert KIND in ("graph", "prog"), f"ENG_MIXED_HEAD_KIND={KIND!r}"
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "head_h1")
        os.makedirs(bd, exist_ok=True)
        _EXT = load(name="fpr_mixed_head", sources=[os.path.join(HERE, "head_h1.cpp")], build_directory=bd,
                    extra_cflags=["-O2", "-std=c++17", "-ffp-contract=off"], with_cuda=True,
                    verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
    return _EXT


def _children(o):
    """(kind, key, child) slots of a container: 0 dict, 1 list / tuple, 2 a plain object's attributes; None: a leaf."""
    if isinstance(o, dict):
        return [(0, k, v) for k, v in o.items()]
    if isinstance(o, (list, tuple)):
        if len(o) > 4096 and not (torch.is_tensor(o[0]) or _children(o[0])):   # (host id / position lists: no tensors)
            return None
        return [(1, i, v) for i, v in enumerate(o)]
    d = getattr(o, "__dict__", None)
    if isinstance(d, dict) and not isinstance(o, _FN_T):
        return [(2, k, v) for k, v in d.items() if k not in _GUARD_SKIP_ATTR]
    return None


def _slot(par, kind, key):
    if kind == 0:
        return par.get(key, _MISS)
    if kind == 1:
        return par[key] if key < len(par) else _MISS
    return par.__dict__.get(key, _MISS)


def guard_record(root, name="hsd"):
    """The slots of every CUDA tensor reachable from root, parent first: [(parent, kind, key, child, ptr, sptr, snb,
    path)]; ptr None for a container. A container reached again (an alias) records its slot, not its children."""
    out, seen = [], set()

    def walk(o, path, depth):
        kids = _children(o) if depth < GUARD_DEPTH else None
        if not kids:
            return []
        ents = []
        for kind, key, v in kids:
            p = f"{path}.{key}" if kind == 2 else f"{path}[{key!r}]"
            if torch.is_tensor(v):
                if v.is_cuda:
                    st = v.untyped_storage()
                    ents.append((o, kind, key, v, v.data_ptr(), st.data_ptr(), st.nbytes(), p))
                continue
            if isinstance(v, _GUARD_SKIP_T + _FN_T) or type(v).__name__ in ("Engine", "Hooks", "D13", "MixedHead",
                                                                                     "ndarray"):
                continue
            if depth == 0 and kind == 2 and key in _GUARD_SKIP_ROOT:
                continue
            if id(v) in seen:
                if _children(v):
                    ents.append((o, kind, key, v, None, None, None, p))
                continue
            seen.add(id(v))
            sub = walk(v, p, depth + 1)
            if sub:
                ents.append((o, kind, key, v, None, None, None, p))
                ents.extend(sub)
        return ents

    seen.add(id(root))
    out = walk(root, name, 0)
    return out


def guard_check(ents):
    """None = every recorded slot holds what the graph captured; else (path, old, new) of the first changed slot."""
    for par, kind, key, ch, ptr, sptr, snb, path in ents:
        cur = _slot(par, kind, key)
        if cur is ch:
            if ptr is None or ch.data_ptr() == ptr:
                continue
            return path, f"{ptr:#x}", f"{ch.data_ptr():#x} (same tensor, storage moved)"
        if ptr is not None and torch.is_tensor(cur) and cur.data_ptr() == ptr:
            st = cur.untyped_storage()
            if st.data_ptr() == sptr and st.nbytes() == snb:
                continue                                   # (a new view of the same storage at the same address)
        if ptr is None:
            return path, f"{type(ch).__name__}@{id(ch):#x}", ("missing" if cur is _MISS else f"{type(cur).__name__}@{id(cur):#x}")
        new = ("missing" if cur is _MISS else f"{cur.data_ptr():#x}/{cur.untyped_storage().nbytes()}" if torch.is_tensor(cur)
               else type(cur).__name__)
        return path, f"{ptr:#x}/{snb}", new
    return None


# _stale's pointer guard through nd's C++ slot pre-check (nd_export._h1x() = fpr_h1scan
# the module nd's h1_guard_scan already uses). check(ents) returns the indices of every entry guard_check MIGHT reject; it
# reports an entry OK only when it passes exactly guard_check's rule (identity -> ptr; or a tensor at the same ptr with the
# same storage (data_ptr, nbytes)) on exact dict / list / tuple / __dict__ parents, so an empty list implies guard_check(ents)
# is None.
# Not loadable (nd_export not imported yet / ND_H1_CXX=0 / build missing) -> today's Python guard_check. ENG_MIXED_HEAD_GUARD_CXX=0
# forces the Python path; =2 runs both and counts guard_cxx_miss (C++ passed, Python rejects) as a check.
GUARD_CXX = os.environ.get("ENG_MIXED_HEAD_GUARD_CXX", "1")
# "family" = once per (S_pad, D_pad, grammar) family, "key" = before every key, "nt" (v5k3) = once
# per (S_pad, D_pad, grammar) family AND once per (NT, grammar)
# "0" (default) = off.
# Only inside precapture (not first-use captures at serve time). A failure logs once and leaves the capture as before.
PRELOAD = os.environ.get("ENG_MIXED_HEAD_PRECAPTURE_PRELOAD", "0")
_PRELOADED = set()
_HP = [None, False]


def _head_preload():
    if not _HP[1]:
        _HP[1] = True
        import importlib.util as ilu_
        import sys as sys_
        hp = sys_.modules.get("head_preload")
        if hp is None:
            sp_ = ilu_.spec_from_file_location("head_preload", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                              "head_preload.py"))
            hp = ilu_.module_from_spec(sp_)
            sp_.loader.exec_module(hp)
            sys_.modules["head_preload"] = hp
        _HP[0] = hp
    return _HP[0]


def _drop_capture_events(hsd):
    """events a capture recorded into persistent drafter state (DraftRank.mix_ev) are capture-state events: drop them after a capture that never replays (a failed one, or head_preload's throwaway probe)"""
    for d_ in list(getattr(hsd, "ranks0", ())) + list(getattr(hsd, "drk", {}).values()):
        if getattr(d_, "mix_ev", None) is not None:
            d_.mix_ev = None
_GCX = [None, False]                           # [fpr_h1scan module or None, resolved]


def _guard_cxx():
    if _GCX[1]:
        return _GCX[0]
    if GUARD_CXX == "0":
        _GCX[1] = True
        return None
    import sys as _sys
    ne = _sys.modules.get("nd_export")
    if ne is None:                             # (nd not imported yet: retry at the next guard; resolved once nd is in)
        return None
    _GCX[1] = True
    try:
        f = getattr(ne, "_h1x", None)
        _GCX[0] = f() if callable(f) else None
    except Exception as ex:                    # (any failure: the Python guard_check, as before)
        _GCX[0] = None
        print(f"mixed_cxx_head: guard C++ pre-check unavailable ({type(ex).__name__}: {ex}); Python guard_check", flush=True)
    print(f"mixed_cxx_head: guard C++ pre-check {'ON (fpr_h1scan via nd_export)' if _GCX[0] is not None else 'OFF (Python guard_check)'}"
          f"{'; ENG_MIXED_HEAD_GUARD_CXX=2: Python re-checks every guard' if GUARD_CXX == '2' and _GCX[0] is not None else ''}",
          flush=True)
    return _GCX[0]


def _guard_check_fast(ents, stats):
    """"""
    X = _guard_cxx()
    if X is None or type(ents) is not list:
        return guard_check(ents)
    try:
        fl = X.check(ents)
    except Exception:
        stats["guard_cxx_err"] += 1
        return guard_check(ents)
    stats["guard_cxx"] += 1
    if fl:
        stats["guard_cxx_flagged"] += 1
        return guard_check(ents)
    if GUARD_CXX == "2":
        bad = guard_check(ents)
        if bad is not None:
            stats["guard_cxx_miss"] += 1
            print(f"mixed_cxx_head: GUARD CXX MISS {bad[0]}: C++ passed, guard_check rejects", flush=True)
        return bad
    return None


class MixedHead:
    def __init__(self, hooks, x=None):
        self.H, self.eng, self.hsd = hooks, hooks.eng, hooks.hsd
        self.x = ext() if x is None else x     # (H2: mixed_cxx_head2's fpr_mixed_head2, the same set_step / head API)
        self.devs = list(self.eng.devs)
        self.graphs = {}                       # HSD key -> ([CUDAGraph per rank], [handle per rank])
        self.seen = collections.Counter()      # eager mixed-head steps per eligible key
        self.gkeys = collections.Counter()     # guard_ok calls per key (nd's native mixed steps' H1 keys)
        self.gref = collections.Counter()      # ... of them refused (key not captured / STALE): the worker's 'H' refusals
        self.pending = None                    # key of this step's eager head (capture after its sync)
        n = self.hsd.h_own[0].shape[0]         # HSD's S_max rows per rank
        pin = dict(pin_memory=True)
        self.idx_d = [torch.zeros(n, dtype=torch.int64, device=d) for d in self.devs]
        self.idx_h = [[torch.zeros(n, dtype=torch.int64, **pin) for _ in range(W)] for _ in range(2)]
        self.idx_ev = [[None] * W for _ in range(2)]
        self.ii = 0
        self.stats = collections.Counter()
        self.eng.stats["mixed_head"] = self.stats
        self.guard = {}                        # key -> guard_record(hsd) at its capture
        _wrap_hsd(self.hsd)
        _HSD_REF[:] = [self.hsd]
        _ENG_REF[:] = [self.eng]
        bump()
        self._defer_guard = False              # precapture batch: one shared record after the batch
        self.retired = []                      # graphs of STALE keys (never replayed; kept: pool rule)
        self.recaps = collections.Counter()
        self.stale_logged = set()
        self.eng.log(f"mixed_cxx_head: on (max {MAX_KEYS} captured keys, capture after {MIN_SEEN} eager steps"
                     f"{'' if MAX_NT <= 0 else f', keys with NT <= {MAX_NT} only'}"
                     f"{f'; grammar bitmask keys (ENG_HEAD_BM_GRAPH) <= {MAX_BM_KEYS} of them' if BM else ''}"
                     f"{f'; pointer guard on, {GUARD_RECAP} re-capture(s) per key' if GUARD else '; pointer guard OFF'})")

    # ---------------------------------------------------------------------------------------------- eligibility
    def _key(self):
        H, hsd = self.H, self.hsd
        if hsd is None or not H.hsd_on or getattr(hsd, "prep", "1") == "0" or getattr(hsd, "cur", None) is None:
            return None
        bm = bool(H.bm_step or getattr(hsd, "bm_any", False))
        if (bm or H.grammar) and not BM:       # (ENG_HEAD_BM_GRAPH off: grammar steps stay on the Python head)
            return None
        k = H.hsd_key()
        if k is None or len(k) != 3:           # (S_pad, D_pad, NT) only: no graph variant (c1d owner)
            return None
        return k + ("bm",) if bm else k

    def _rows(self, st):
        """Per rank (n_s, sample rows, n_t, aux rows view, drafter aux) or None when the layout is not the eager one's."""
        hsd = self.hsd
        c = hsd.cur
        out = []
        for r in range(W):
            s = st[r]
            sr = list(s["batch"].sample_rows)
            n_s, n_t = int(c.n_s[r]), int(c.n_t[r])
            if len(sr) < n_s:                  # (head_eager's h_rows[:n_s] would not exist)
                return None
            h = s["h"]
            aux = s["aux"]
            if not (h.is_contiguous() and aux.is_contiguous() and h.dtype == torch.bfloat16):
                return None
            a2 = aux.view(aux.shape[0], -1)
            da = hsd.draft_rank(r, c.D_pad).b["aux"]
            if n_t > a2.shape[0] or a2.shape[1] != da.shape[1] or a2.dtype != da.dtype:
                return None
            out.append((n_s, sr[:n_s], n_t, a2, da))
        return out

    # ---------------------------------------------------------------------------------------------- the head
    def head_eager(self, st, batches) -> bool:
        bump()                                 # (a Python head step: HSD state may change)
        if MEMLOG_S > 0 and time.time() - _MEMLOG_T[0] >= MEMLOG_S:
            _MEMLOG_T[0] = time.time()
            memlog(self.eng)
        self.stats["calls"] += 1
        if SUMMARY_EVERY and self.stats["calls"] % SUMMARY_EVERY == 0:
            c = self.stats
            n_ = max(1, c["guard_checks"])
            self.eng.log(f"mixed_cxx_head: {c['calls']} mixed heads: C++ {c['cxx']}, Python uncaptured {c['py_uncaptured']} / "
                         f"layout {c['py_layout']} / ineligible {c['py_ineligible']} / stale {c['py_stale']}; keys captured "
                         f"{len(self.graphs)} / {MAX_KEYS} (seen {len(self.seen)}, retired {len(self.retired)}); guard checks "
                         f"{c['guard_checks']}, us avg {c['guard_us'] / n_:.1f} max {c['guard_max_us']:.1f}, stale {c['stale']}")
            # the uncaptured keys' eager sightings, most frequent first: the live
            # histogram for ENG_MIXED_HEAD_PRECAPTURE_KEYS / ENG_MIXED_HEAD_GRAPHS (keys "S_pad|D_pad|NT[|bm]": count)
            top = [(kk, n) for kk, n in self.seen.most_common() if n > 0][:HIST_TOP]
            if top and HIST_TOP > 0:
                self.eng.log("mixed_cxx_head: uncaptured keys (eager steps): " + ", ".join(
                    "|".join(str(x) for x in kk) + f": {n}" for kk, n in top))
        k = self._key()
        if k is None:
            self.stats["py_ineligible"] += 1
            self.pending = None
            return False
        ent = self.graphs.get(k)
        if ent is None:
            self.seen[k] += 1
            self.pending = k if (LAZY and self.seen[k] >= MIN_SEEN and len(self.graphs) < MAX_KEYS and self._bm_room(k)
                                 and (MAX_NT <= 0 or int(k[2]) <= MAX_NT)) else None
            self.stats["py_uncaptured"] += 1
            return False
        if GUARD and self._stale(k):
            return False
        rows = self._rows(st)
        if rows is None:
            self.stats["py_layout"] += 1
            self.pending = None
            return False
        hsd, eng = self.hsd, self.eng
        i = self.ii
        self.ii ^= 1
        for r in range(W):
            n_s, sr, n_t, a2, da = rows[r]
            s = st[r]
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(eng.streams[r]):
                if n_s:
                    ev = self.idx_ev[i][r]
                    if ev is not None:
                        ev.synchronize()       # the H2D of two steps ago (long done)
                    hb = self.idx_h[i][r]
                    hb.numpy()[:n_s] = sr
                    self.idx_d[r][:n_s].copy_(hb[:n_s], non_blocking=True)
                    ev = self.idx_ev[i][r] = torch.cuda.Event()
                    ev.record()
                h = s["h"]
                self.x.set_step(r, torch.device(self.devs[r]).index, eng.streams[r].cuda_stream, h.data_ptr(), h.shape[0],
                                h.shape[1], self.idx_d[r].data_ptr(), n_s, hsd.h_own[r].data_ptr(), a2.data_ptr(),
                                da.data_ptr(), n_t, a2.shape[1] * a2.element_size())
        for r in range(W):
            with torch.cuda.device(self.devs[r]):
                rc = self.x.head(r, ent[1][r])
            if rc != 0:
                raise RuntimeError(f"mixed_cxx_head: head rank {r} key {k}: FPR error {rc}")
        self.stats["cxx"] += 1
        self.pending = None
        return True

    def _bm_room(self, k) -> bool:
        """a grammar ("bm") key may be captured only while fewer than MAX_BM_KEYS bm keys are (plain keys: always)"""
        return len(k) != 4 or sum(1 for x in self.graphs if len(x) == 4) < MAX_BM_KEYS

    def _stale(self, k) -> bool:
        """The pointer guard before a replay of key k: True = a recorded slot changed (the graph retired, the Python head
        runs this step)."""
        ents = self.guard.get(k)
        if ents is None:
            return False
        t0 = time.perf_counter()
        bad = _guard_check_fast(ents, self.stats)
        us = (time.perf_counter() - t0) * 1e6
        self.stats["guard_checks"] += 1
        self.stats["guard_us"] += us
        if us > self.stats["guard_max_us"]:
            self.stats["guard_max_us"] = us
        if bad is None:
            return False
        path, old, new = bad
        self.stats["stale"] += 1
        if int(_EP[1]) != 0 and int(_EP[0]) == int(_EP[1]):   # (v5b) stale while the epoch says "verified, unchanged":
            self.stats["epoch_hole"] += 1                       # a writer that does not bump (the native fast path would
            self.eng.log(f"H1 EPOCH HOLE t={time.time():.1f} key={k} arg={path}: stale with epoch == verified {int(_EP[0])} "   # have replayed it)
                         f"(an unbumped writer; the native fast path would have replayed this graph)")
        if (k, path) not in self.stale_logged:
            self.stale_logged.add((k, path))
            self.eng.log(f"H1 STALE t={time.time():.1f} key={k} arg={path} old={old} new={new} (graph retired; Python head this step"
                         f"{'; re-capture after ' + str(MIN_SEEN) + ' eager steps' if self.recaps[k] < GUARD_RECAP else '; Python head from now on'})")
        self.retired.append(self.graphs.pop(k))
        bump()
        self.guard.pop(k, None)
        self.recaps[k] += 1
        self.seen[k] = 0 if self.recaps[k] <= GUARD_RECAP else -10 ** 9
        self.stats["py_stale"] += 1
        self.pending = None
        return True

    # ---------------------------------------------------------------------------------------------- capture
    def _pool_bytes(self):
        if not POOLSTAT:
            return None
        try:
            snap = torch.cuda.memory_snapshot()
        except Exception:  # noqa: BLE001
            return None
        out = []
        for r, d in enumerate(self.devs):
            di, pid = torch.device(d).index, tuple(self.eng.gpool[r])
            out.append(sum(sg["total_size"] for sg in snap
                           if sg.get("device") == di and tuple(sg.get("segment_pool_id", ())) == pid))
        return out

    def after_sync(self, st, batches) -> None:
        k, self.pending = self.pending, None
        if k is None or k in self.graphs or len(self.graphs) >= MAX_KEYS or self._key() != k or not self._bm_room(k) \
                or (MAX_NT > 0 and int(k[2]) > MAX_NT):
            return
        if self._rows(st) is None:
            return
        self._capture(k)

    def _capture(self, k, why="") -> bool:
        """Capture HSD.run of key k (HSD's current tables must be k's) per rank into the engine's graph pool; True = admitted.
        after_sync (an eager head of k just ran) and precapture (a synthetic prepare of k, nothing executed)."""
        hsd, eng = self.hsd, self.eng
        bump()
        tabs = [hsd._tables(r, k[:3]) for r in range(W)]
        if any(hsd.tab[r] is not tabs[r] or bool(tabs[r].get("use_bm")) != (len(k) == 4) for r in range(W)):
            return False                       # HSD's tables of another key / the other variant
        if PRELOAD in ("family", "key", "nt") and self._defer_guard:   # (v5k: precapture only)
            fams = [k] if PRELOAD == "key" else [("f", int(k[0]), int(k[1]), len(k) == 4)] + \
                ([("nt", int(k[2]), len(k) == 4)] if PRELOAD == "nt" else [])
            if any(f_ not in _PRELOADED for f_ in fams):
                _PRELOADED.update(fams)
                tp = time.perf_counter()
                try:
                    # the REAL capture's pool + mode
                    # roots: HSD's slots / scalars around it
                    res = _head_preload().preload(eng, lambda r: hsd.run(r, None, None), tag=f"h1 {k}", log=eng.log,
                                                  pools=[eng.gpool[r] for r in range(W)], roots=[hsd],
                                                  capture_error_mode="thread_local")
                    if res.get("new_slots") or res.get("rebound_slots") or res.get("scalar_changes"):
                        self.stats["preload_side_effects"] += 1
                    self.stats["preload_calls"] += 1
                    self.stats["preload_triton"] += int(res.get("triton", 0))
                    self.stats["preload_mm"] += int(res.get("mm", 0))
                except Exception as e:  # noqa: BLE001  (the capture below runs as before)
                    self.stats["preload_failed"] += 1
                    if self.stats["preload_failed"] == 1:
                        eng.log(f"mixed_cxx_head: head_preload FAILED at key {k} ({type(e).__name__}: {str(e)[:200]}); "
                                f"captures continue without it")
                finally:
                    _drop_capture_events(hsd)
                    self.stats["preload_ms"] += (time.perf_counter() - tp) * 1e3
                bump()
                tabs = [hsd._tables(r, k[:3]) for r in range(W)]
                if any(hsd.tab[r] is not tabs[r] or bool(tabs[r].get("use_bm")) != (len(k) == 4) for r in range(W)):
                    return False               # (the probe left HSD's tables on another key: cannot happen, checked)
        mem0 = [torch.cuda.memory_reserved(d) for d in self.devs]
        t0 = time.perf_counter()
        gs = []
        try:
            for r in range(W):
                with torch.cuda.device(self.devs[r]), torch.cuda.stream(eng.streams[r]):
                    g = torch.cuda.CUDAGraph()
                    g.capture_begin(pool=eng.gpool[r], capture_error_mode="thread_local")
                    try:
                        hsd.run(r, None, None)
                    finally:
                        g.capture_end()
                    gs.append(g)
            for d in self.devs:
                torch.cuda.synchronize(d)
        except Exception as e:  # noqa: BLE001  (a capture failure: this key stays on the Python head)
            # events the failed capture recorded into persistent state are capture-state events: the
            # next EAGER head's wait on them is cudaErrorInvalidValue (drafter.py _join_mix: DraftRank.mix_ev, set by every
            # mHC boundary for the next one). Nothing of the failed capture ever runs, so there is nothing to wait for:
            # drop them (the step's own eager head ran before this capture; the next one starts from scratch)
            _drop_capture_events(hsd)
            self.stats["capture_failed"] += 1
            eng.log(f"mixed_cxx_head: capture of key {k} failed ({type(e).__name__}: {str(e)[:200]}); Python head stays")
            self.seen[k] = -10 ** 9
            return False
        hs = [self.x.add_graph(g.raw_cuda_graph_exec(), torch.device(self.devs[r]).index) for r, g in enumerate(gs)]
        self.graphs[k] = (gs, hs)
        gtxt = ""
        if GUARD and self._defer_guard:        # (precapture batch: the shared record is assigned when the batch ends)
            gtxt = "; guard deferred (batch)"
        elif GUARD:
            tg = time.perf_counter()
            self.guard[k] = ents = guard_record(hsd)
            gtxt = (f"; guard {sum(1 for e in ents if e[4] is not None)} tensors / {len(ents)} slots recorded in "
                    f"{(time.perf_counter() - tg) * 1e3:.1f} ms")
        self.stats["captures"] += 1
        ms = (time.perf_counter() - t0) * 1e3
        if _CAPLOG is not None:
            _CAPLOG.record("H1", k, ms, expected=not _caplog_listed(k))
        mem1 = [torch.cuda.memory_reserved(d) for d in self.devs]
        pb = None if self._defer_guard else self._pool_bytes()   # (batch: memory_snapshot once, after the batch)
        eng.stats.setdefault("mixed_head_caps", []).append(dict(   # per admitted key: what it costs (sizes the cap)
            key=list(k), ms=round(ms, 1), reserved_delta_MB=[round((b - a) / 1e6, 1) for a, b in zip(mem0, mem1)],
            reserved_GB=[round(b / 1e9, 3) for b in mem1], pool_GB=[round(b / 1e9, 4) for b in pb] if pb else None))
        eng.log(f"mixed_cxx_head: captured key {k}{why} ({len(self.graphs)} / {MAX_KEYS}) in {ms:.0f} ms; reserved GB per GPU "
                f"{[round(b / 1e9, 2) for b in mem1]} (+MB {[round((b - a) / 1e6, 1) for a, b in zip(mem0, mem1)]})"
                + (f"; graph pool GB {[round(b / 1e9, 3) for b in pb]}" if pb else "") + gtxt)
        return True


# ---------------------------------------------------------------------------------------------- precapture
class _AnyPage(dict):
    """a window table that maps every page to one window page (the rank's scratch page): synthetic sequences only"""

    def __init__(self, page, blocks=128):
        super().__init__()
        self._p = int(page)
        # the served C++ D13 HSD prepare (d13_cxx host.hsd_build) reads window_table as a MAP
        # (iterates / looks up the blocks a row needs: "window page for position 1000 missing"), not through .get():
        # pre-populate every 32-token block up to blocks * 32 positions (decode rows at 1000 + 2k + 1, tail rows <= 2368)
        self.update({i: self._p for i in range(int(blocks))})

    def get(self, k, d=None):
        return self._p


def synth_layout(key, k, row_buckets, seq_buckets, tail_buckets):
    """(d, p, n_t) for HSD key = (S_pad, D_pad, NT): d decode sequences (k + 1 sample rows each) + p final-prefill sequences
    (1 sample row each) on one rank, n_t tail rows, such that hsd._build derives exactly key (S_pad = bucket(n_s), D_pad =
    bucket(n_d), NT = bucket(n_t, tail_buckets(S_pad))); None when no such step exists (most decode rows first)."""
    S_pad, D_pad, NT = (int(x) for x in key[:3])

    def bucket(n, bs):
        for b in bs:
            if n <= b:
                return b
        return None
    K1 = k + 1
    tb = tail_buckets(S_pad)
    for d in range(D_pad, -1, -1):
        for p in range(0, D_pad - d + 1):
            n_s, n_d = K1 * d + p, d + p
            if n_s < 1 or n_d < 1 or bucket(n_s, row_buckets) != S_pad or bucket(n_d, seq_buckets) != D_pad:
                continue
            n_t = max(NT, n_s)
            if bucket(n_t, tb) == NT:          # (rows beyond the sampled ones: one non-sampled filler prefill sequence)
                return d, p, n_t
    return None


def _synth(hsd, key, scratch_page):
    """synthetic StepBatches + states of HSD key `key` (rank 0 only; the other ranks empty): positions / slots arbitrary but
    valid (nothing executes: only HSD's tables / blob are written, and the step in progress rewrites them)"""
    from types import SimpleNamespace as NS
    lay = synth_layout(key, hsd.k, hsd.row_buckets, hsd.seq_buckets, hsd.tail_buckets)
    if lay is None:
        return None
    d, p, n_t = lay
    K1 = hsd.k + 1
    wt = _AnyPage(scratch_page)
    seqs, positions, seq_slot, sample_rows, states = [], [], [], [], {}
    row = 0
    import sys
    IF = sys.modules.get("interfaces")       # the served path reads every StepBatch / SeqEntry field
    PAD = int(getattr(IF, "PAD", -1))         # (paged_slot, block_table, cached_upto, ...): real dataclasses when loaded

    def SE(**kw):
        kw["start"] = kw.pop("start_")
        full = dict(cached_upto=0, last_chunk=False, block_table=[], tail_row0=kw.get("row0", 0), n_blocks=0)
        full.update(kw)
        full["num_computed"] = full["start"]
        if IF is not None and hasattr(IF, "SeqEntry"):
            import dataclasses
            names = {f.name for f in dataclasses.fields(IF.SeqEntry)}
            return IF.SeqEntry(**{k: v for k, v in full.items() if k in names})
        return NS(**full)
    for i in range(d):                         # decode sequences: k + 1 rows, all sampled
        rid = f"__precap_d{i}"
        seqs.append(SE(rid=rid, kind="decode", row0=row, n_rows=K1, start_=1000, state_slot=i, window_table=wt,
                       replay_start=0))
        for j in range(K1):
            positions.append(1000 + j)
            seq_slot.append(len(seqs) - 1)
            sample_rows.append(row + j)
        row += K1
    for i in range(p):                         # final prefill chunks: one row each, sampled
        rid = f"__precap_p{i}"
        seqs.append(SE(rid=rid, kind="prefill", row0=row, n_rows=1, start_=0, last_chunk=True, state_slot=d + i,
                       window_table=wt, replay_start=0))
        positions.append(0)
        seq_slot.append(len(seqs) - 1)
        sample_rows.append(row)
        row += 1
    if n_t > row:                              # a non-final prefill chunk: the remaining tail rows, none sampled
        n = n_t - row
        seqs.append(SE(rid="__precap_f", kind="prefill", row0=row, n_rows=n, start_=0, state_slot=d + p, window_table=wt,
                       replay_start=0, tail_row0=max(row, row + n - 128)))
        for j in range(n):
            positions.append(j)
            seq_slot.append(len(seqs) - 1)
        row += n
    for s in seqs:
        states[s.rid] = NS(temperature=1.0, seed=0, drafts_valid=False, num_acceptable_drafts=None)
    def mk(rank, **kw):
        kw.setdefault("paged_slot", [PAD] * len(kw["positions"]))      # PAD: no paged KV write
        if IF is not None and hasattr(IF, "StepBatch"):
            return IF.StepBatch(rank, 0, len(kw["positions"]), **kw)
        return NS(rank=rank, step=0, n_rows=len(kw["positions"]), copies=[], poison=[], **kw)
    b0 = mk(0, positions=positions, seq_slot=seq_slot, sample_rows=sample_rows, tail_rows=list(range(row)),
            window_slot=[0] * row, seqs=seqs)
    empty = lambda r_: mk(r_, positions=[], seq_slot=[], sample_rows=[], tail_rows=[], window_slot=[], seqs=[])   # noqa: E731
    return [b0] + [empty(r_) for r_ in range(1, W)], states


_HSD_STEP_ATTRS = ("cur", "bm_any", "_res", "_hi")          # reassigned by HSD.prepare (the old objects stay untouched)
_HSD_STEP_INPLACE = ("tab", "_slot_key", "_temp_np", "_seed_np")   # mutated in place by HSD.prepare (copied, written back)


def hsd_save(hsd) -> dict:
    """HSD's host-side per-step state before precapture's synthetic prepares: whatever reads
    it before the step's own prepare sees the pre-precapture values (device tables cannot be restored: the step's prepare
    rewrites them)"""
    sv = {a: getattr(hsd, a) for a in _HSD_STEP_ATTRS if hasattr(hsd, a)}
    for a in _HSD_STEP_INPLACE:
        x = getattr(hsd, a, None)
        if x is not None:
            sv[a] = x.copy() if hasattr(x, "copy") else list(x)
    return sv


def hsd_restore(hsd, sv) -> None:
    for a in _HSD_STEP_ATTRS:
        if a in sv:
            setattr(hsd, a, sv[a])
    for a in _HSD_STEP_INPLACE:
        if a in sv:
            getattr(hsd, a)[:] = sv[a]


def precapture(eng, keys=None, log=None, max_new=None) -> dict:
    """capture H1 graphs for mixed-head keys BEFORE their first use, at ENG_DECODE_PRECREATE's done point
    per key a synthetic HSD.prepare (writes only HSD's
    per-step tables / blob; the step in progress rewrites them) + the key's drafter buckets (draft_rank + plan + PeerComm,
    as prepare's _upload) + the capture (nothing executes: no next_ids / h_all / drafter-ring writes). keys: (S_pad, D_pad,
    NT) tuples, default the file ENG_MIXED_HEAD_PRECAPTURE_KEYS (one "S_pad|D_pad|NT" per line, most frequent first);
    MAX_KEYS / MAX_NT / max_new cap it. Skipped (with the reason): keys whose drafter state a first EAGER run would create
    (DraftRank._skws: HSD_SKW8 with Rd <= 32 and no workspace yet: its zeroed counters would be re-zeroed by every replay),
    keys no synthetic step maps to, captured keys. ENG_MIXED_CXX_HEAD off / grammar variants ("bm"): nothing.
    Returns dict(captured=[...], skipped={key: why})."""
    lg = log or eng.log
    out = dict(captured=[], skipped={})
    if os.environ.get("ENG_DECODE_HEAD_PRECAPTURE", "0") == "1":   # + the decode graph H keys
        try:
            import importlib.util as ilu_
            import sys as sys_
            dhp = sys_.modules.get("decode_head_precap")
            if dhp is None:
                sp_ = ilu_.spec_from_file_location("decode_head_precap", os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "decode_head_precap.py"))
                dhp = ilu_.module_from_spec(sp_)
                sp_.loader.exec_module(dhp)
                sys_.modules["decode_head_precap"] = dhp
            out["decode"] = dhp.precapture(eng, hsd_save, hsd_restore, _AnyPage, log=lg)
        except Exception as e_:  # noqa: BLE001  (a failed decode precapture leaves those keys first-use, as before)
            lg(f"mixed_cxx_head.precapture: decode_head_precap FAILED ({type(e_).__name__}: {str(e_)[:200]})")
        bump()                                 # its synthetic HSD prepares may precede _wrap_hsd: move the epoch
    if not ON or KIND != "graph" or getattr(eng.hooks, "hsd", None) is None:
        return out
    m = _get(eng.hooks)
    hsd = m.hsd
    if keys is None:
        fn = os.environ.get("ENG_MIXED_HEAD_PRECAPTURE_KEYS", "")
        keys = []
        if fn == "all":                        # every reachable key (+ the grammar twins with ENG_HEAD_BM_GRAPH)
            keys = grid_keys(hsd)
            if len(keys) > MAX_KEYS:
                lg(f"mixed_cxx_head.precapture: WARNING the key grid has {len(keys)} keys > ENG_MIXED_HEAD_GRAPHS {MAX_KEYS}")
        elif fn and os.path.exists(fn):
            for ln in open(fn):
                ln = ln.split("#", 1)[0].strip()   # (inline "# count" comments allowed)
                if ln and not ln.startswith("#"):
                    p_ = ln.split("#")[0].strip().split("|")
                    keys.append(tuple(int(x) for x in p_[:3]) + (("bm",) if len(p_) > 3 and p_[3] == "bm" else ()))
    if _CAPLOG is not None and isinstance(keys, (list, tuple)):   # (stage 4a) the keys asked for here count as listed
        try:
            _caplog_listed(())
            _CAPLOG_PRE.update(tuple(int(x) for x in k_[:3]) for k_ in keys)
        except Exception:   # noqa: BLE001
            pass
    import sys
    skw8 = bool(getattr(sys.modules.get("drafter"), "HSD_SKW8", False))   # (kernels/head/drafter.py, as hsd.py imports it)
    sv = hsd_save(hsd)
    t0_ = time.perf_counter()
    s0 = guard_record(hsd) if GUARD else None  # the slots that exist before the batch
    m._defer_guard = True
    bump()
    if MEMLOG_S > 0:
        memlog(eng, "precapture-start")
    try:
        _precapture_keys(eng, m, hsd, keys, max_new, skw8, out)
    finally:
        m._defer_guard = False
        hsd_restore(hsd, sv)
        if GUARD:
            # ONE record after the batch for every key it captured: precapture's synthetic prepares only ADD slots (new key
            # tables in _tab_cache, new D_pad drafters / PeerComm / _skws) or rebind skipped ones (tab, cur, _res, mix_ev),
            # so each key's own slots are unchanged since its capture; checked: every slot recorded BEFORE the batch
            bad = guard_check(s0)
            if bad is not None:
                # the batching argument does not hold: a shared record would carry a
                # stale guard for the batch's keys -> retire EVERY key of this batch (never
                # replayed; kept: pool rule), loudly; they run the Python head (visible as 'H' / uncaptured)
                for k in out["captured"]:
                    ent = m.graphs.pop(k, None)
                    if ent is not None:
                        m.retired.append(ent)
                    m.guard.pop(k, None)
                    out["skipped"][k] = "batch guard check FAILED (retired)"
                m.stats["precapture_guard_failed"] += 1
                lg(f"mixed_cxx_head.precapture: ERROR a slot recorded BEFORE the batch changed during it: {bad}; "
                   f"retired all {len(out['captured'])} keys of this batch (Python head for them)")
                out["captured"] = []
            else:
                ents = guard_record(hsd)
                for k in out["captured"]:
                    m.guard[k] = ents
            out["guard"] = dict(pre_ok=bad is None, slots_before=len(s0),
                                tensors=(sum(1 for e in ents if e[4] is not None) if bad is None else None))
        if PRELOAD in ("family", "key", "nt"):
            out["preload"] = dict(mode=PRELOAD, calls=m.stats["preload_calls"], triton=m.stats["preload_triton"],
                                  mm=m.stats["preload_mm"], failed=m.stats["preload_failed"],
                                  side_effects=m.stats["preload_side_effects"],
                                  s=round(m.stats["preload_ms"] / 1e3, 1))
        pb = m._pool_bytes()
        out["pool_GB"] = [round(b / 1e9, 3) for b in pb] if pb else None
        out["wall_s"] = round(time.perf_counter() - t0_, 1)
        bump()
        if MEMLOG_S > 0:
            memlog(eng, "precapture-end")
        _TRIP_N[0] = 0                         # (trip counting starts after a precapture)
        _PRECAP_T[0] = time.time()                   # (cur / tab / bm_any / _res / _hi / skey / temp / seed as before precapture)
    lg(f"mixed_cxx_head.precapture: {len(out['captured'])} H1 keys captured {out['captured'][:64]}; skipped "
       f"{len(out['skipped'])}: {dict(list(out['skipped'].items())[:6])}; skip reasons "
       f"{dict(collections.Counter(str(v)[:60] for v in out['skipped'].values()))}; {out.get('wall_s')} s, guard "
       f"{out.get('guard')}, graph pool GB {out.get('pool_GB')}" + (f", head_preload {out['preload']}" if "preload" in out else ""))
    return out


def grid_keys(hsd) -> list:
    """every HSD key a mixed step can reach: S_pad in row_buckets x D_pad in seq_buckets x NT in
    tail_buckets(S_pad) with a synthetic layout (synth_layout != None); then each key's grammar twin (k + ("bm",)) when
    ENG_HEAD_BM_GRAPH is on. Plain keys first (a MAX_KEYS cap cuts the twins first), each sorted by (S_pad, D_pad, NT)."""
    out = []
    for S in sorted({int(x) for x in hsd.row_buckets}):
        for D in sorted({int(x) for x in hsd.seq_buckets}):
            for NT in hsd.tail_buckets(S):
                k = (S, D, int(NT))
                if synth_layout(k, hsd.k, hsd.row_buckets, hsd.seq_buckets, hsd.tail_buckets) is not None:
                    out.append(k)
    return out + ([k + ("bm",) for k in out] if BM else [])


def _synth_bm(hsd, batches):
    """per rank the synthetic step's bitmask rows (all tokens allowed, as HSD's own padding rows): rank 0's sample rows"""
    import sys
    bw = None                                  # (hk-03b: hsd.py's module is not in sys.modules under its class's __module__)
    for c in type(hsd).__mro__:                # the defining module's globals via a method of the class, then sys.modules
        fn = c.__dict__.get("prepare") or c.__dict__.get("_build")
        bw = getattr(fn, "__globals__", {}).get("BM_WORDS") if fn is not None else None
        if bw is None:
            bw = getattr(sys.modules.get(c.__module__), "BM_WORDS", None)
        if bw is not None:
            break
    if bw is None:
        bw = getattr(sys.modules.get("hsd"), "BM_WORDS", None)
    if bw is None:
        raise RuntimeError(f"no BM_WORDS in the modules of {[c.__module__ for c in type(hsd).__mro__]} / hsd")
    return [torch.full((len(b.sample_rows), bw), -1, dtype=torch.int32) if b.sample_rows else None for b in batches]


_SKWS_SRC = "self._skws = (EA.skw8(), torch.empty(8 * 32 * 32768, dtype=torch.float32, device=x.device),"


def _skws_create(eng, ds):
    """DraftRank._sk's lazy HSD_SKW8 workspace (fn, fp32 ws, zeroed per-weight counters, empty) created exactly as its first
    call does, eagerly and outside any capture (in a capture the zeros would become a replayed memset). Refuses ('' = done)
    when drafter.py's _sk no longer has the expected allocation (then the key stays skipped, as before)."""
    import importlib
    import inspect
    try:
        src = inspect.getsource(type(ds[0])._sk)
    except Exception as e:  # noqa: BLE001
        return f"cannot read DraftRank._sk ({type(e).__name__})"
    if _SKWS_SRC not in src or '("wq_b", "wo_a", "wo_b")' not in src:
        return "DraftRank._sk allocation changed (not pre-created)"
    EA = importlib.import_module("engine.attn")
    for r, d in enumerate(ds):
        if getattr(d, "_skws", None) is not None:
            continue
        dv = torch.device(eng.devs[r])
        with torch.cuda.device(dv):
            d._skws = (EA.skw8(), torch.empty(8 * 32 * 32768, dtype=torch.float32, device=dv),
                       {n: torch.zeros(512, dtype=torch.int32, device=dv) for n in ("wq_b", "wo_a", "wo_b")},
                       torch.empty(0, dtype=torch.uint8, device=dv))
    for d_ in eng.devs:
        torch.cuda.synchronize(d_)
    return ""


def _precapture_keys(eng, m, hsd, keys, max_new, skw8, out) -> None:
    for k in keys:
        bm = len(k) > 3 and k[3] == "bm"
        k = tuple(int(x) for x in k[:3]) + (("bm",) if bm else ())
        if bm and not BM:
            out["skipped"][k] = "bm key without ENG_HEAD_BM_GRAPH"
            continue
        if bm and not m._bm_room(k):
            out["skipped"][k] = "MAX_BM_KEYS"
            continue
        if max_new is not None and len(out["captured"]) >= max_new:
            break
        if k in m.graphs:
            out["skipped"][k] = "captured"
            continue
        if len(m.graphs) >= MAX_KEYS:
            out["skipped"][k] = "MAX_KEYS"
            break
        if MAX_NT > 0 and k[2] > MAX_NT:
            out["skipped"][k] = "NT > MAX_NT"
            continue
        sp = [int(eng.pools[r].scratch_window_page) for r in range(W)]
        sy = _synth(hsd, k, sp[0])
        if sy is None:
            out["skipped"][k] = "no synthetic step maps to this key"
            continue
        ds = [hsd.draft_rank(r, k[1]) for r in range(W)]
        if skw8 and ds[0].Rd <= 32 and any(getattr(d, "_skws", None) is None for d in ds):
            why_ = _skws_create(eng, ds)       # the workspace eagerly, OUTSIDE the capture (as _sk's first call)
            if why_:
                out["skipped"][k] = f"HSD_SKW8 workspace: {why_}"
                continue
        batches, states = sy
        for r in range(1, W):                  # (the other ranks' synthetic window tables: their own scratch page)
            for s in batches[r].seqs:
                s.window_table = _AnyPage(sp[r])
        for d in list(hsd.ranks0) + list(hsd.drk.values()):
            d.mix_ev = None                    # (a previous call's mixes: done, the GPUs are synced; not a capture input)
        try:
            import contextlib
            with contextlib.ExitStack() as es:
                for r in range(W):
                    es.enter_context(torch.cuda.stream(eng.streams[r]))
                hsd.prepare(batches, states, _synth_bm(hsd, batches) if bm else None)
            for d_ in m.devs:
                torch.cuda.synchronize(d_)
        except Exception as e:  # noqa: BLE001
            import traceback
            fr = traceback.extract_tb(e.__traceback__)[-1] if e.__traceback__ is not None else None
            at = f" at {os.path.basename(fr.filename)}:{fr.lineno} {fr.name}" if fr is not None else ""
            out["skipped"][k] = f"synthetic prepare: {type(e).__name__}: {str(e)[:120]}{at}"
            continue
        if bm and not getattr(hsd, "bm_any", False):
            out["skipped"][k] = "synthetic prepare without the bitmask variant"
            continue
        if hsd.cur is None or (hsd.cur.S_pad, hsd.cur.D_pad, getattr(hsd.cur, "NT", None)) != k[:3]:
            out["skipped"][k] = f"synthetic prepare gave key {(hsd.cur.S_pad, hsd.cur.D_pad, getattr(hsd.cur, 'NT', None))}"
            continue
        if m._capture(k, why=" (precapture)"):
            out["captured"].append(k)
        else:
            out["skipped"][k] = "capture refused / failed"


# ---------------------------------------------------------------------------------------------- the engine's two calls
def _get(hooks):
    m = hooks.__dict__.get("_mixed_head")
    if m is None:
        if KIND == "prog":                    # H2: recorded op programs (mixed_cxx_head2.py, head_h2.cpp)
            import mixed_cxx_head2
            m = hooks._mixed_head = mixed_cxx_head2.MixedHead2(hooks)
        else:
            m = hooks._mixed_head = MixedHead(hooks)
    return m


def head_eager(hooks, st, batches) -> bool:
    """Hooks.head_eager(st, batches) of a mixed step: True = the head was issued here (skip the Python head)."""
    if not ON or batches is None or hooks.hsd is None:
        return False
    return _get(hooks).head_eager(st, batches)


def memlog(eng, why="") -> None:
    """(log only) "mixed_cxx_head: mem t=<epoch s> reserved / allocated / slack GiB per GPU" (slack = reserved - allocated:
    torch's own cache, reusable before an OOM)"""
    try:
        rs = [torch.cuda.memory_reserved(d) / 2 ** 30 for d in eng.devs]
        al = [torch.cuda.memory_allocated(d) / 2 ** 30 for d in eng.devs]
        eng.log(f"mixed_cxx_head: mem t={time.time():.1f}{(' ' + why) if why else ''} reserved {[round(x, 3) for x in rs]} "
                f"allocated {[round(x, 3) for x in al]} slack {[round(x - y, 3) for x, y in zip(rs, al)]}")
    except Exception:  # noqa: BLE001
        pass


def guard_ok(hooks, key_text: str) -> bool:
    """False = not captured, or STALE now
    (the graph is retired here, as in head_eager; the exporter's next refresh_graphs unpublishes the key): the worker runs the
    step embedded. Guard off: True for any captured key."""
    m = hooks.__dict__.get("_mixed_head") if hooks is not None else None
    if m is None or not key_text:
        return False
    try:
        parts = key_text.split("|")
        bm = parts[-1] == "bm"
        k = tuple(int(x) for x in (parts[:-1] if bm else parts)) + (("bm",) if bm else ())
    except ValueError:
        return False
    m.gkeys[k] += 1
    m.stats["guard_keys_calls"] += 1
    if MEMLOG_S > 0 and time.time() - _MEMLOG_T[0] >= MEMLOG_S:
        _MEMLOG_T[0] = time.time()
        memlog(m.eng)
    if SUMMARY_EVERY and HIST_TOP > 0 and m.stats["guard_keys_calls"] % SUMMARY_EVERY == 0:   # the native keys' histogram
        n_, r_ = m.stats["guard_keys_calls"], sum(m.gref.values())         # (log only; one line every N guard_ok calls)
        m.eng.log(f"mixed_cxx_head: guard keys: {n_} calls, {len(m.gkeys)} distinct, refused {r_} ({len(m.gref)} keys), "
                  f"captured {len(m.graphs)} / {MAX_KEYS}: " + ", ".join(
                      "|".join(str(x) for x in kk) + f": {c}" + (f" (refused {m.gref[kk]})" if m.gref[kk] else "")
                      for kk, c in m.gkeys.most_common(HIST_TOP)))
    if k not in m.graphs:
        m.gref[k] += 1
        return False
    m.stats["guard_ok_calls"] += 1
    if GUARD and m._stale(k):
        m.stats["guard_ok_refused"] += 1
        m.gref[k] += 1
        return False
    return True


def after_sync(hooks, st, batches) -> None:
    """Right after the mixed step's engine.sync(): capture the key whose eager head this step ran (if admitted)."""
    if not ON or hooks.__dict__.get("_mixed_head") is None:
        return
    hooks._mixed_head.after_sync(st, batches)
