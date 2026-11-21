"""nd_export.py -- the ENGINE side of fpr/core/src/core2b/engine_handover.h v0.2: fills the
C shim (nd_handover.cpp: fpr_nd.Handover; header v0.4) from the live Python Engine inside the embedded interpreter of the stage-2b
process, and implements its GIL callbacks.

    ex = Exporter(eng)          # once at init, after the engine is built (decode buckets are ensured lazily or up front)
    ptr = ex.ptr()              # FprEngineHandover* for the C++ loop (valid while `ex` lives)
    ex.ensure_bucket(R)         # == the C callback; pre-creates every lazy buffer + graph T (has_seqs 0 / 1), fills views

Rules kept here (v0.2 header):
  - views only; the engine keeps every tensor alive. Every graph ever published stays referenced (self._keep): Python's
    D13 graph-T recapture replaces its dict entry, which would otherwise destroy a published exec (F).
  - generations: a decode bucket slot is bumped when the engine's state object of that key changed (re-created).
  - the set handoff (C): every double-buffered set has ONE persistent engine event per set index. push_state makes it
    the holder of each Python set; pull_state records it on the rank's step stream (after the embedded step's own
    uploads) and makes it the holder again (M), so nd's "wait ev[i], fill, copy, re-record ev[i]" is exact on both sides.
v0.3: K dl as three tensors / sets (one alternation dl_i, the same events), L estream, I prev-of + step serial in
push / pull (FprEhPrevOf.serial = the PRODUCING STEP's eng.step_serial, as eng.prev_seq[rid][2]), J host token history
appended in push, M events. OPEN: N extraction pinned targets.
v0.5
  - rank state: h1_h_own (hsd.h_own[r]), rx_own_* (d13.rx_own[r]: D13._fallback's binding), pools[] (CopyOp.pool names);
  - H1 graph entries (mixed_cxx_head's captured keys, bare "S_pad|D_pad|NT", aux_dst = the (rank, D_pad) drafter aux);
  - mixed A / T views (rule (b)): ensure_mixed(keys) takes the step's whole key set at once, then
    fills the views. The engine's bucket dicts (eng.bk, eng.bk_view) are wrapped (_NotifyDict) so EVERY create / evict /
    onebk view rebuild / max-A re-carve bumps that key's generation + buckets_epoch when it happens (inside ensure_mixed
    or an embedded step), without the exporter holding a reference to any state (an evicted bucket's memory must free);
  - pull_state returns the embedded step's eng.hist DELTAS (base = len at push, or len at on_ctx's admission inside the
    step: eng.hist is wrapped too, _HistDict);
  the views' ptrs / bytes / fields_hash are empty (n_ptrs 0, fields_hash 0: nd
    refuses such a view and runs the step embedded). The mixed indexer / attn_pf views are v0.6.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import time
import itertools
import collections
import collections.abc

PULL_PROF = os.environ.get("ND_PULL_PROF", "0") == "1"   # (v3.28jP) pull_state per-phase timer
PULL_PROF_EVERY = int(os.environ.get("ND_PULL_PROF_EVERY", "64") or 64)
# refresh_graphs right after pull_state in the same boundary (core2b
# embedded_end -> post) reuses pull_state's h1_guard_scan instead of scanning again: no Python runs between them that could
# rebind a guarded slot. The mark is (MixedHead identity, step serial), set by pull_state's scan and consumed (cleared) by
# the next refresh_graphs; push_state / ensure_bucket / ensure_mixed / verify_all clear it (Python may follow them).
H1_REUSE = os.environ.get("ND_H1_REUSE", "1") != "0"
# refresh_graphs skips ONLY its three
# graph walks (eng.graphs, d13.gt, MixedHead.graphs) when the graph containers' fingerprint (container / value / per-rank
# element ids + the owners' monotonic capture / recapture / retire / stale counters) is unchanged since its last run. Every
# derived-source setter (span kinds, libs / MoeSnap, P1, Engram binds), the H1 pointer guard and the H1 unpublish run on
# every call as before: their sources can change in any Python crossing, so only the walks are skippable. core2b runs the
# post after every Python crossing (no native skip). BOUNDARY: mixed VIEW contents are ensure_mixed's (its _fill_sig
# compare), not refresh_graphs'. ND_PUB_FAST=0: always walk. ND_PUB_VERIFY=1: walk anyway when the fingerprint
# says skip; a graph published then counts as pub_missed (a fingerprint blind spot). ND_PUB_PROF=1: per-part host us.
PUB_ON = os.environ.get("ND_PUB_FAST", "1") != "0"
PUB_VERIFY = os.environ.get("ND_PUB_VERIFY", "0") == "1"
# ND_H1_SCAN_SKIP=1: _h1_guard skips h1_guard_scan when mixed_cxx_head.scan_needed(buckets epoch) says
# nothing can have changed since the last clean scan (epoch == verified != 0, same buckets epoch).
# ND_H1_SCAN_VERIFY=1: never skip; scan anyway and count h1_scan_skip_miss when a would-be skip retires something.
H1_SCAN_SKIP = os.environ.get("ND_H1_SCAN_SKIP", "0") == "1"
H1_SCAN_VERIFY = os.environ.get("ND_H1_SCAN_VERIFY", "0") == "1"
PUB_PROF = os.environ.get("ND_PUB_PROF", "0") == "1"
PUB_LOG_EVERY = int(os.environ.get("ND_PUB_LOG_EVERY", "4096") or 4096)
# precapture_all(): every decode bucket (graph T has_seqs 0 / 1 + views) at the native done
# point; ND_PRECAP_LONG=1 also the ND_LONG variants R + lvl * LVL_TAG. mark_native_done(): later publishes (graph LA / H /
# T / H1, a decode bucket first met) count as LATE by kind, logged with their key.
PRECAP_LONG = os.environ.get("ND_PRECAP_LONG", "0") == "1"
LATE_LOG_MAX = int(os.environ.get("ND_LATE_LOG_MAX", "64") or 64)
# ND_PRECAP_AUTO: "pc" = at the first
# refresh_graphs after the engine's decode precreate finished (eng._pc None), "n:<N>" = at the N-th refresh_graphs call.
# Logs precapture_all's counts and every GPU's free memory before -> after (GiB), then marks native done. Unset = off.
PRECAP_AUTO = os.environ.get("ND_PRECAP_AUTO", "")
# the handover's fixed sizes: generation slots
# and graph entries (4 per H1 / decode-H key + L / LA / T). Storage never moves, so both are sized once here. Host cost
# ~103 KB per slot (8 ranks x (FprEhBucketRank 1,520 B + FprEhMixedBucketRank 11,312 B)), 152 B per graph entry: 4096 slots
# ~0.42 GB host, 32768 entries ~5 MB. ND_N_GENS / ND_GRAPH_CAPACITY override (ints >= 64 / >= 1024, else the default).
def _env_int(name: str, default: int, lo: int) -> int:
    try:
        v = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return v if v >= lo else default


N_GENS = _env_int("ND_N_GENS", 4096, 64)
GRAPH_CAPACITY = _env_int("ND_GRAPH_CAPACITY", 32768, 1024)

PAGE_TOKENS, WINDOW_PAGE_TOKENS = 64, 32      # geometry.py (d13_host.h BT / WT)
_DT = {torch.uint8: "u8", torch.bool: "bool", torch.int32: "i32", torch.int64: "i64", torch.float32: "f32",
       torch.bfloat16: "bf16", torch.float16: "f16"}


def T(t):
    """FprEhTensor: (ptr, dtype, shape, stride, device | -1 pinned host)"""
    if t is None:
        return None
    assert t.dim() <= 4, t.shape
    return (t.data_ptr(), _DT[t.dtype], list(t.shape), list(t.stride()), t.device.index if t.is_cuda else -1)


def B(t):
    """FprEhBuf: (ptr, bytes, device)"""
    if t is None:
        return None
    return (t.data_ptr(), t.numel() * t.element_size(), t.device.index if t.is_cuda else -1)


class _NotifyDict(dict):
    """eng.bk / eng.bk_view with change notices (v0.5 rule (b)). The engine's only mutations are d[k] = v, del d[k],
    d.pop(k) (LRU: popped and re-inserted at once) and clear(). A pop is resolved at the next set of the same key (the
    same object: no change) or at flush() (a removal). Lookups stay dict's own C paths. No references to values are kept."""

    def __init__(self, d, cb):
        super().__init__(d)
        self._cb = cb
        self._popped = {}      # key -> id(value) popped and not re-inserted yet

    def __setitem__(self, k, v):
        p = self._popped.pop(k, None)
        old = dict.get(self, k, None)
        dict.__setitem__(self, k, v)
        if (p is not None and p == id(v)) or (old is not None and old is v):
            return
        self._cb(k)

    def __delitem__(self, k):
        dict.__delitem__(self, k)
        self._popped.pop(k, None)
        self._cb(k)

    def pop(self, k, *a):
        had = dict.__contains__(self, k)
        v = dict.pop(self, k, *a)
        if had:
            self._popped[k] = id(v)
        return v

    def popitem(self):
        k, v = dict.popitem(self)
        self._cb(k)
        return k, v

    def clear(self):
        ks = list(self)
        dict.clear(self)
        self._popped.clear()
        for k in ks:
            self._cb(k)

    def flush(self):
        """pops never re-inserted are removals"""
        ks, self._popped = list(self._popped), {}
        for k in ks:
            self._cb(k)


class _DirtyDict(dict):
    """MixedHead.graphs with a record of the keys written (_set) and removed (_rm) since nd last read them, both
    insertion-ordered (a key written again moves to the end, as a re-inserted key does in the dict). Every mutating dict
    method is routed (set / del / pop / popitem / clear / setdefault / update / |=); lookups, iteration and len stay
    dict's own C paths. No values are kept here."""

    def __init__(self, d):
        super().__init__(d)
        self._set = {}
        self._rm = {}

    def _w(self, k):
        self._set.pop(k, None)
        self._set[k] = None

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, v)
        self._w(k)

    def __delitem__(self, k):
        dict.__delitem__(self, k)
        self._rm[k] = None

    def pop(self, k, *a):
        had = dict.__contains__(self, k)
        v = dict.pop(self, k, *a)
        if had:
            self._rm[k] = None
        return v

    def popitem(self):
        k, v = dict.popitem(self)
        self._rm[k] = None
        return k, v

    def clear(self):
        for k in dict.keys(self):
            self._rm[k] = None
        dict.clear(self)

    def setdefault(self, k, d=None):
        if not dict.__contains__(self, k):
            self[k] = d
        return dict.__getitem__(self, k)

    def update(self, *a, **kw):
        for k, v in dict(*a, **kw).items():
            self[k] = v

    def __ior__(self, o):
        self.update(o)
        return self


class _HistDict(dict):
    """eng.hist with the admission notice (v0.5 pull deltas): on_ctx's `eng.hist[rid] = prompt + output` inside an
    embedded step records the request's base length, so pull returns exactly what the step appended after it."""

    def __init__(self, d, ex):
        super().__init__(d)
        self._ex = ex

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, v)
        b = self._ex._hist_base
        if b is not None:
            b[k] = len(v)


def _max_model_len(eng) -> tuple:
    """the served max model length and its source, first hit of: EngineConfig / Engine
    attribute max_model_len, env FPR_MAX_MODEL_LEN, env MAXLEN, hooks_idx.P_MAX x 64 (the block
    table width production's indexer assumes), 655,360 (production's --max-model-len)."""
    for src, v in (("cfg.max_model_len", getattr(getattr(eng, "cfg", None), "max_model_len", None)),
                   ("eng.max_model_len", getattr(eng, "max_model_len", None)),
                   ("env FPR_MAX_MODEL_LEN", os.environ.get("FPR_MAX_MODEL_LEN")),
                   ("env MAXLEN", os.environ.get("MAXLEN"))):
        try:
            n = int(v or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return n, src
    HI = next((m for k, m in list(sys.modules.items()) if k.endswith("hooks_idx") and hasattr(m, "P_MAX")), None)
    if HI is not None and int(getattr(HI, "P_MAX", 0) or 0) > 0:
        return int(HI.P_MAX) * 64, "hooks_idx.P_MAX x 64"
    return 655_360, "default 655,360"


def _key_text(key) -> str:
    out = []

    def walk(x):
        if isinstance(x, (tuple, list)):
            for y in x:
                walk(y)
        else:
            out.append(str(x))
    walk(key)
    return "|".join(out)


# ENG_H1_SCAN_ON_ENSURE=1: ensure_mixed (a Python boundary that can
# create / evict buckets) ends with the same h1_guard_scan + unpublish as refresh_graphs. Every key published after a scan is
# guard-clean, and only Python changes the guarded objects, so NativeWorker may treat the return of refresh_graphs (post) and
# of ensure_mixed as scan points: a key nd can see, with no Python since the last scan point, needs no guard_ok call.
H1_SCAN_ON_ENSURE = os.environ.get("ENG_H1_SCAN_ON_ENSURE", "0") == "1"

# the per-slot check of h1_guard_scan in C++ (fpr_h1scan, h1scan.cpp
# prebuilt in build/h1scan): it returns the slots it cannot pass by guard_check's exact rule
# So a slot is bad iff Python's guard_check says so, as before
# C++ only skips the Python call for slots that hold what was captured. ND_H1_CXX_CHECK=1: also run the Python check on
# every slot C++ passed and count disagreements (stat h1_cxx_miss; must stay 0; logged).
H1_CXX = os.environ.get("ND_H1_CXX", "1") != "0"
H1_CXX_CHECK = os.environ.get("ND_H1_CXX_CHECK", "0") == "1"
_H1X = [None, False]     # [module, tried]
H1X_STATS = dict(cxx_scans=0, cxx_slots=0, cxx_flagged=0, cxx_miss=0, py_scans=0, rebuilds=0, extends=0, sig_miss=0, extend_miss=0)
# MixedHead's monotonic counters that move whenever a guard record is assigned or popped without the dict's length
# changing (a pop + a capture inside one interval); see h1_guard_scan
_H1_SIG_CTRS = ("captures", "stale", "py_stale", "capture_failed", "precapture_guard_failed")


def _h1x():
    if not H1_CXX:
        return None
    if not _H1X[1]:
        _H1X[1] = True
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            bd = os.path.join(here, "build", "h1scan" + os.environ.get("ND_BUILD_SUFFIX", ""))
            if os.environ.get("PHYS_CXX_ND_PREBUILT", "0") == "1":   # (as core2b_proc's fpr_nd / fpr_nd2: the .so AS BUILT, no JIT)
                import glob
                import importlib.util as ilu
                sos = sorted(glob.glob(os.path.join(bd, "fpr_h1scan*.so")))
                if not sos:
                    raise FileNotFoundError(f"no prebuilt fpr_h1scan in {bd}")
                spec = ilu.spec_from_file_location("fpr_h1scan", sos[0])
                m_ = ilu.module_from_spec(spec)
                spec.loader.exec_module(m_)
                _H1X[0] = m_
            else:
                from torch.utils.cpp_extension import load
                os.makedirs(bd, exist_ok=True)
                _H1X[0] = load(name="fpr_h1scan", sources=[os.path.join(here, "h1scan.cpp")], build_directory=bd,
                               extra_cflags=["-O2", "-std=c++17"], with_cuda=False,
                               verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")))
            print(f"nd_export: h1 guard scan C++ ON (fpr_h1scan abi {getattr(_H1X[0], 'abi', '?')}, {bd})"
                  f"{'; ND_H1_CXX_CHECK=1: Python re-checks every slot' if H1_CXX_CHECK else ''}", flush=True)
        except Exception as ex:      # (no extension: the Python loop, as jR)
            _H1X[0] = None
            print(f"nd_export: h1 guard scan C++ OFF (fpr_h1scan failed to load: {type(ex).__name__}: {ex}); Python scan",
                  flush=True)
    return _H1X[0]


def _h1_ctrs_ok(mh, mst) -> bool:
    """every _H1_SIG_CTRS counter readable from MixedHead.stats; else logged once per MixedHead.
    MixedHead.stats is a collections.Counter: a counter never incremented is absent and reads
    0 (_h1_ctr), so for a Counter only a non-int value counts as missing; a plain mapping must hold every name."""
    if not isinstance(mst, collections.abc.Mapping):
        miss = list(_H1_SIG_CTRS)
    else:
        cnt = isinstance(mst, collections.Counter)
        miss = [n_ for n_ in _H1_SIG_CTRS if not isinstance(mst.get(n_, 0) if cnt else mst.get(n_), int)]
    if miss and not mh.__dict__.get("_nd_ctr_logged"):
        mh.__dict__["_nd_ctr_logged"] = True
        print(f"nd_export: H1 SIG COUNTERS MISSING {miss} in MixedHead.stats: no O(1) signature (scan cache rebuilt and "
              f"MixedHead.graphs walked on every call)", flush=True)
    return not miss


def _h1_build(groups, gix, slots, items) -> None:
    """h1_guard_scan's cache build over (key, record) items, extending groups / gix / slots in place: records grouped
    by identity (a batch's keys share one), each NEW group's entries walked once; a slot lists its groups in order."""
    g0 = len(groups)
    for k, ents in items:
        gi = gix.get(id(ents))
        if gi is None:
            gix[id(ents)] = gi = len(groups)
            groups.append((ents, []))
        groups[gi][1].append(k)
    for gi in range(g0, len(groups)):
        for e in groups[gi][0]:
            ident = (id(e[0]), e[1], e[2], id(e[3]), e[4])
            v = slots.get(ident)
            if v is None:
                slots[ident] = v = [e, []]
            if not v[1] or v[1][-1] != gi:
                v[1].append(gi)


def h1_guard_scan(mh) -> dict:
    """mixed_cxx_head's guard over EVERY captured H1 key at a
    step boundary, so nd only ever finds guarded keys published and never needs a Python call per native step. The keys' guard records share most slots (HSD's tables / blobs, the drafter ranks of a
    D_pad): each distinct slot (parent identity, kind, key, recorded child identity, recorded pointer) is checked ONCE with
    mixed_cxx_head.guard_check([entry]) (the same per-entry rule); a key holding a changed slot goes through the head's own
    MixedHead._stale(k) (re-checks, logs "H1 STALE", retires the graph, re-capture policy), and the exporter's unpublish
    loop then bumps its generation. Returns dict(keys, slots, retired=[keys], us)."""
    import time
    t0 = time.perf_counter()
    out = dict(keys=0, slots=0, retired=[], us=0.0)
    MXH = sys.modules.get(type(mh).__module__) if mh is not None else None
    if mh is None or MXH is None or not getattr(MXH, "GUARD", False):
        return out
    guard = getattr(mh, "guard", {})
    # The distinct slots (parent identity, kind, key, recorded child identity, recorded pointer) and the keys holding each
    # are cached per MixedHead and rebuilt only when the guard dict changes (a key captured / retired: MixedHead assigns
    # guard[k] = guard_record(hsd) whole and pops it; never edits a record in place). Per scan: one guard_check per distinct
    # slot, then bad = the live keys (in mh.graphs) holding a changed slot, _stale in guard order.
    # slot stats count every distinct slot.
    # mixed_cxx_head's
    # batched precapture assigns ONE shared guard_record (every HSD tensor slot, so its length grows with the key count) to
    # every key of the batch; the jQ rebuild walked keys x entries and kept a per-slot list of every key: O(N x E(N)) in ONE
    # call. Now records are grouped by identity: each distinct record's entries are walked once, a slot lists the RECORDS
    # (group indices) holding it, a group lists its keys.
    # The sig is O(1): MixedHead assigns guard[k] whole only at a capture (captures++) or
    # a batch end (len grows) and pops it only with its graph (stale / batch failure: len or the counters move);
    # ND_PUB_VERIFY=1 also checks the jQ per-key sig every scan (a miss counts in H1X_STATS["sig_miss"] + rebuilds).
    # mixed_cxx_head binds mh.guard / mh.graphs once; the cache still pins the dict it was built from ([6], compared with
    # `is`), so a rebound dict can never reuse its id. A missing counter (renamed in MixedHead) = no O(1) sig: logged once,
    # and every scan rebuilds.
    mst = getattr(mh, "stats", None)
    if not _h1_ctrs_ok(mh, mst):
        sig = (object(),)
    else:
        sig = (id(guard), len(guard)) + tuple(int(mst.get(n_, 0)) for n_ in _H1_SIG_CTRS)
    cache = mh.__dict__.get("_nd_h1_scan")
    if cache is not None and cache[6] is not guard:
        cache = None
    if PUB_VERIFY:
        full = tuple((k, id(ents), len(ents)) for k, ents in guard.items())
        if cache is not None and cache[0] == sig and cache[4] != full:
            H1X_STATS["sig_miss"] += 1
            print(f"nd_export: H1 SIG MISS: guard changed at an unchanged O(1) sig (total {H1X_STATS['sig_miss']})",
                  flush=True)
            cache = None
    else:
        full = None
    if cache is None or cache[0] != sig:
        # additions only since the cached sig (same dict, every counter but captures unchanged, len grown by exactly
        # the captures' delta): the new keys are the dict's last `dn` items (insertion order; MixedHead never re-inserts
        # without a pop, which moves stale / the failure counters) -> extend the cached groups / slots with them only.
        # The result equals a full rebuild over the whole dict (same first-seen slot order, same group order).
        # Anything else = full rebuild.
        dn = sig[1] - cache[0][1] if cache is not None and len(sig) > 1 and len(cache[0]) > 1 else -1
        inc = (cache is not None and len(sig) > 1 and len(cache[0]) > 1 and sig[0] == cache[0][0] and dn > 0
               and sig[2] - cache[0][2] == dn and sig[3:] == cache[0][3:])
        if inc:
            groups, gix, slots = cache[5]
            _h1_build(groups, gix, slots, list(itertools.islice(reversed(guard.items()), dn))[::-1])
        else:
            groups, gix, slots = [], {}, {}
            _h1_build(groups, gix, slots, guard.items())
        vals = list(slots.values())
        if inc and PUB_VERIFY:     # the extend must equal a full rebuild: same slot entries in order, same key groups
            g2, x2, s2 = [], {}, {}
            _h1_build(g2, x2, s2, guard.items())
            if [id(v[0]) for v in s2.values()] != [id(v[0]) for v in vals] or [v[1] for v in s2.values()] != \
                    [v[1] for v in vals] or [g_[1] for g_ in g2] != [g_[1] for g_ in groups]:
                H1X_STATS["extend_miss"] += 1
                print(f"nd_export: H1 EXTEND MISS: the incremental scan cache differs from a full rebuild (total "
                      f"{H1X_STATS['extend_miss']}); using the rebuild", flush=True)
                groups, gix, slots, inc = g2, x2, s2, False
                vals = list(slots.values())
        # (v3.28jS) [2] = the entries, in slot order, for fpr_h1scan.check; [3] = the groups' key lists; [4] = jQ sig
        # (VERIFY); [5] = the build state for an incremental extend (the group records stay referenced: ids stay unique)
        cache = (sig, vals, [v[0] for v in vals], [g_[1] for g_ in groups],
                 full if full is not None else (tuple((k, id(ents), len(ents)) for k, ents in guard.items())
                                                if PUB_VERIFY else None), (groups, gix, slots), guard)
        mh.__dict__["_nd_h1_scan"] = cache
        out["rebuilt"] = 1
        H1X_STATS["extends" if inc else "rebuilds"] += 1
    badk = set()
    gk = cache[3]

    def mark(gis):
        for gi in gis:
            badk.update(gk[gi])
    X = _h1x()
    if X is not None:
        flagged = X.check(cache[2])
        H1X_STATS["cxx_scans"] += 1
        H1X_STATS["cxx_slots"] += len(cache[2])
        H1X_STATS["cxx_flagged"] += len(flagged)
        for i in flagged:
            e, gis = cache[1][i]
            if MXH.guard_check([e]) is not None:
                mark(gis)
        if H1_CXX_CHECK:
            fl = set(flagged)
            for i, (e, gis) in enumerate(cache[1]):
                if i not in fl and MXH.guard_check([e]) is not None:
                    H1X_STATS["cxx_miss"] += 1
                    mark(gis)
                    print(f"nd_export: H1 CXX MISS slot {i} {e[7] if len(e) > 7 else '?'}: C++ passed, guard_check rejects "
                          f"(total {H1X_STATS['cxx_miss']})", flush=True)
        out["cxx"] = len(flagged)
    else:
        H1X_STATS["py_scans"] += 1
        for e, gis in cache[1]:
            if MXH.guard_check([e]) is not None:
                mark(gis)
    out["keys"] = sum(1 for k in guard if k in mh.graphs)
    out["slots"] = len(cache[1])
    bad = [k for k in list(guard) if k in badk and k in mh.graphs] if badk else []   # no O(N) pass when clean
    for k in bad:
        if mh._stale(k):
            out["retired"].append(k)
    out["us"] = (time.perf_counter() - t0) * 1e6
    return out


def _stages() -> dict:
    """tables2f.Stage._per_dev (device index -> the indexer's staging ring), {} when tables2f is not loaded (never imported
    or created from here)"""
    TF = sys.modules.get("tables2f")
    S = getattr(TF, "Stage", None) if TF is not None else None
    return dict(getattr(S, "_per_dev", None) or {})


def _c1m_facts(eng, EM) -> dict:
    """the c1mega adapter facts nd's owner_of needs: runs configured (owner_of's `if not
    self.runs`), NRM (the adapter module's), ENG_C1_MEGA_AB (per-arm keys: nd refuses), C1M_IX + C1IDX_E0 (ix.e0_on:
    e0_prepare on mega steps: nd refuses). ENG_C1_MEGA off: zeros. The adapter is built at engine init (C1_MEGA)."""
    if not bool(getattr(EM, "C1_MEGA", False)):
        return dict(c1_runs=False, c1_nrm=4, c1_ab=False, c1_e0=False)
    a = eng.c1m()
    am = sys.modules.get(type(a).__module__)
    ix = getattr(a, "ix", None)
    return dict(c1_runs=bool(getattr(a, "runs", None)), c1_nrm=int(getattr(am, "NRM", 4)),
                c1_ab=bool(getattr(EM, "C1_MEGA_AB", None)), c1_e0=bool(ix is not None and getattr(ix, "e0_on", False)))


class Exporter:
    def __init__(self, eng, n_gens: int = 0, graph_capacity: int = 0):
        n_gens = int(n_gens) or N_GENS              # 0 = the module default (env-overridable)
        graph_capacity = int(graph_capacity) or GRAPH_CAPACITY
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nd_cxx
        self.eng = eng
        self.d13 = eng._d13()
        self.M = sys.modules[type(self.d13).__module__]
        H = eng.hooks
        self.mml, self.mml_src = _max_model_len(eng)
        eng.log(f"nd_export: max_model_len {self.mml} ({self.mml_src}): handover caps + P1 pf buffer widths")
        eng.log(f"nd_export: handover generation slots {n_gens} (ND_N_GENS), graph entries {graph_capacity} (ND_GRAPH_CAPACITY): "
                f"~{n_gens * 103 / 1024:.0f} MB + {graph_capacity * 152 / 2 ** 20:.1f} MB host")
        self.x = nd_cxx.ext().Handover(len(eng.devs), int(getattr(H.hsd, "k", 3) if H.hsd is not None else 3),
                                       PAGE_TOKENS, WINDOW_PAGE_TOKENS,
                                       int(getattr(eng, "vocab_size", 129280)), int(self.mml),
                                       int(eng.cfg.buckets[-1]), n_gens, graph_capacity)
        self._ev = {}          # (purpose, ..., set) -> the persistent torch Event (engine-owned)
        self._keep = []        # every published CUDAGraph (F)
        self._pub = {}         # (key, rank, piece) -> the published CUDAGraph object
        self._keep_s4 = {}     # (v0.7 slice 4) (view key, rank, what) -> the Python object a published pointer points into
        self._state_of = {}    # R -> the bucket's state list identity (re-creation -> generation bump)
        self._rids = {}        # rid text -> the engine's rid object (prev_seq / hist keys)
        # v0.5: mixed views. _mix_ser[key text] counts the engine's changes of that bucket (NotifyDict); a view is
        # refilled when its serial moved since the fill (_mix_filled) or (v3.11b) when a published field's source changed
        # without a dict notify (_mix_sig, see _fill_sig). No state references are kept here.
        self._mix_ser, self._mix_filled, self._mix_sig = {}, {}, {}
        self._hist_base = None     # v0.5: rid -> len(eng.hist[rid]) between push and pull (None outside)
        self._p1_sig, self._gst_h, self._comb_n, self._comb_keep = {}, {}, {}, []
        EM = sys.modules[type(eng).__module__]
        self._onebk = bool(getattr(EM, "MIX_ONE_BUCKET", False))
        self.fields_hash = 0       # OPEN: md5 over mixed_fields.h's lines; 0 = no field table (nd refuses)
        self.stats = dict(mixed_bumps=0, decode_bumps=0, ensure_mixed=0, ensure_mixed_refused=0, mixed_fills=0, mixed_refills_sig=0, h1_unpublished=0,
                          h1_guard_scans=0, h1_guard_retired=0, h1_guard_slots_max=0, h1_guard_us_sum=0.0, h1_guard_us_max=0.0,
                          pub_calls=0, pub_walks_skipped=0, pub_walks=0, pub_verify=0, pub_missed=0)
        self._pub_fp = None        # the graph containers' fingerprint after the last refresh_graphs
        self._pub_us = {}          # part -> host us sum
        self._native_done = False  # mark_native_done(): later publishes count as late
        self._h1_dd = None         # the _DirtyDict nd installed as MixedHead.graphs
        self._h1_idents = {}       # H1 key -> {published ident: None}
        self._h1_unpub_full = True # the next _h1_unpublish is a full pass (set at every (re)wrap)
        self._late = {}
        if not isinstance(eng.bk, _NotifyDict):
            eng.bk = _NotifyDict(eng.bk, self._on_bk)
        if hasattr(eng, "bk_view") and not isinstance(eng.bk_view, _NotifyDict):
            eng.bk_view = _NotifyDict(eng.bk_view, self._on_view)
        if not isinstance(eng.hist, _HistDict):
            eng.hist = _HistDict(eng.hist, self)
        self.x.set_config(self.config_json())
        for r in range(len(eng.devs)):
            self.x.set_rank(r, self._rank(r))
        self.x.set_callbacks(self.ensure_bucket, self.push_state, self.pull_state, self.ensure_mixed)
        self._set_itables()
        self._set_engram()
        self._set_pagecopy()
        self._set_span_kinds()
        self._set_libs_snap()
        self._alloc_nd_bufs()
        self._set_hsd_bm()
        self._set_p1()
        self.refresh_graphs()
        if bool(getattr(EM, "C1_MEGA", False)) and os.environ.get("FPR_CRUMBS", "0") not in ("", "0"):
            eng.log("nd_export: native C1: crumbs n/a (FPR_CRUMBS wraps Python pieces only: steps nd runs natively, C1 mega "
                    "decode included, leave no crumbs; missing crumbs are not a hang)")

    def _set_itables(self) -> None:
        """(v0.7) the engine's C++ TableBuilders as fpr::ITables* for nd: ONE
        instance per builder for Python and nd (one D12 delta state). 0 (NULL) when a builder is Python's / in check mode
        or cxx_kv has no itables_ptr yet: nd then uses its private builder / refuses native TB_TAIL."""
        def ptr(tb):
            mod = sys.modules.get(type(tb).__module__) if tb is not None else None
            f = getattr(mod, "itables_ptr", None) if mod is not None else None
            if f is None:
                return 0
            try:
                return int(f(tb) or 0)
            except Exception:  # noqa: BLE001  (a Python / check builder: no C++ instance)
                return 0
        a, b = ptr(getattr(self.eng, "tb_tail", None)), ptr(getattr(self.d13, "tb", None))
        if hasattr(self.x, "set_tables"):
            self.x.set_tables(a, b)
        self.stats["itables_tail"], self.stats["itables_d13"] = int(bool(a)), int(bool(b))
        # 0 = no getter
        abi = 0
        for tb in (getattr(self.eng, "tb_tail", None), getattr(self.d13, "tb", None)):
            mod = sys.modules.get(type(tb).__module__) if tb is not None else None
            if mod is not None and hasattr(mod, "TABLES_ABI"):
                abi = int(mod.TABLES_ABI)
                break
        else:   # (no C++ builder: the loaded cxx_kv if any; never imported from here: no fpr_kvtables load / sys.path change)
            CK = sys.modules.get("cxx_kv")
            abi = int(getattr(CK, "TABLES_ABI", 0) or 0) if CK is not None else 0
        if hasattr(self.x, "set_tables_abi"):
            self.x.set_tables_abi(abi)
        self.stats["tables_abi"] = abi

    def _engram_rank(self, r) -> dict:
        """hooks.engram[r] (EngramPhys) as engine_handover.h's
        FprEhEngramRank; {} = no Engram on this rank (nd refuses PREPARE_ENGRAM). Events are created up front (Python makes
        _stage_ev lazily, and the per-layer ev get their CUDA event at their first record), so both sides use the same
        handles; ws_ev = one persistent spare event of the rank."""
        eng, H = self.eng, self.eng.hooks
        eg = H.engram[r] if getattr(H, "engram", None) is not None else None
        if eg is None or getattr(eg, "_pk", None) is None:
            return {}
        p = lambda t: int(t.data_ptr()) if t is not None else 0  # noqa: E731
        with torch.cuda.device(eng.devs[r]):
            for i in range(2):
                if eg._stage_ev[i] is None:
                    eg._stage_ev[i] = torch.cuda.Event()
                    eg._stage_ev[i].record(eng.streams[r])
            for e in eg.ev:
                if not int(e.cuda_event):
                    e.record(eng.streams[r])
        EGM = sys.modules[type(eg).__module__]
        LAY = tuple(getattr(EGM, "LAYERS", (1, 14)))
        g = eg.gather_cfg
        wbf = list(eg.wbf) if getattr(eg, "wbf", None) is not None else [None, None]
        ss = getattr(eg, "_start_src", None)
        return dict(device=torch.device(eng.devs[r]).index, pk_d=p(eg._pk), stage=[p(x) for x in eg._stage],
                    stage_ev=[int(e.cuda_event) for e in eg._stage_ev], hctx=int(eg.hctx), ids=p(eg.b_ids),
                    tmap=p(eg.tmap), ring=p(eg.ring), hash=p(eg.hash), keep=p(eg.keep), err=p(eg.err),
                    rows=[p(x) for x in eg.rows], kv=[p(x) for x in eg.kv], tabs=[int(eg.tables[L].rec0) for L in LAY],
                    lo=[int(x) for x in eg.lo], hi=[int(x) for x in eg.hi], g_ctas=int(g["ctas"]),
                    g_threads=int(g["threads"]), g_unroll=int(g["unroll"]), wbf=[p(x) for x in wbf],
                    wbf_N=int(wbf[0].shape[0]) if wbf[0] is not None else 0,
                    wbf_K=int(wbf[0].shape[1]) if wbf[0] is not None else 0,
                    stage_buf=p(getattr(eg, "_stage_buf", None)), stage_chunks=int(getattr(eg, "_stage_chunks", 0) or 0),
                    start_src_i64=int(ss is not None and ss.dtype == torch.int64), start_src=p(ss),
                    ev=[int(e.cuda_event) for e in eg.ev], ws_ev=int(self.ev(("ws", r), r).cuda_event),
                    w8_max_rows=int(eg.w8_max_rows) if getattr(eg, "gemm8", None) else 0,   # (no gemm8: every T on cuBLAS)
                    guard=int(bool(getattr(EGM, "GUARD", False))),
                    nring=int(eg.ring.shape[0]))

    def _slice4_view(self, t, r, s) -> dict:
        """0 = not available (nd refuses)."""
        import ctypes
        out = dict(tkeys=0, tkeys_bytes=0, moe_bucket=0, moe_bucket_bytes=0, plan_slots=0, n_plan_slots=0)
        ps = s.get("_cxx_attn_slots")    # the view's FlashMLA plan slots, int64 numpy
        if ps is not None:               # (published in place: no copy, so a non-int64 / non-contiguous array is refused)
            if isinstance(ps, np.ndarray) and ps.dtype == np.int64 and ps.ndim == 1 and ps.flags["C_CONTIGUOUS"]:
                self._keep_s4[(t, r, "ps")] = ps
                out.update(plan_slots=int(ps.ctypes.data) if ps.size else 0, n_plan_slots=int(ps.size))
            else:
                self.stats["plan_slots_refused"] = self.stats.get("plan_slots_refused", 0) + 1
        kb = s.get("_tg_kb")
        if kb is not None and isinstance(kb[1], (bytes, bytearray)) and len(kb[1]):
            b = kb[1]
            self._keep_s4[(t, r, "tk")] = b           # (the published pointer points into this object)
            if isinstance(b, bytearray):              # (mutable: its own buffer, no copy)
                addr = ctypes.addressof((ctypes.c_char * len(b)).from_buffer(b))
            else:                                     # (bytes: c_char_p points at the object's data, no copy)
                addr = ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value or 0
            out.update(tkeys=int(addr), tkeys_bytes=len(b))
        ME = sys.modules.get("mixed_cxx_moe_ensure")
        if ME is not None and hasattr(ME, "bucket_of"):
            try:
                mb = ME.bucket_of(s)
                self._keep_s4[(t, r, "mb")] = mb
                out.update(moe_bucket=int(ctypes.addressof(mb)), moe_bucket_bytes=int(ctypes.sizeof(mb)))
            except Exception:  # noqa: BLE001
                pass
        return out

    def _p1_view(self, t, r, s, ro) -> dict:
        """a mixed view's P1 fields: idx = the view's tables2f.GeneralTables s["idx_gt"] (plain
        or an onebk GtView over the core): Bc, the idx2f_tab registration hid of its 16 tensors IN CREATION ORDER (pos, dbt,
        sl[1], sl[2], cs[1], cs[2], rs, valid_ir, t2r, qsl, ir2row_c, vir, vrow, row2ir_d, nb_lens, bt), t2r_ir, lat and the
        lean OwnedIndexer's cand / topk (one tensor shared by the layers) / slow.
        A re-created GeneralTables (growth, onebk sweep) changes
        _fill_sig: refill + generation bump at the next ensure_mixed. pf: not exported."""
        out = {}
        gt = s.get("idx_gt")
        if gt is not None and getattr(gt, "hid", None) is not None and getattr(gt, "base", None) is not None:
            try:
                t16 = [gt.pos, gt.dbt, gt.sl[1], gt.sl[2], gt.cs[1], gt.cs[2], gt.rs, gt.valid_ir, gt.t2r, gt.qsl,
                       gt.ir2row_c, gt.vir, gt.vrow, gt.row2ir_d, gt.nb_lens, gt.bt]
                kp = getattr(gt, "_keep", None)
                if kp is not None and (len(kp) != 16 or any(a is not b for a, b in zip(kp, t16))):
                    raise ValueError("GeneralTables._keep is not the 16-tensor creation list")
                base = gt.base
                tk = base.topk
                if isinstance(tk, dict):
                    tk = next(iter(tk.values())) if tk else None
                out["idx"] = dict(Bc=int(gt.Bc), tab_hid=int(gt.hid), t16=[T(x) for x in t16], t2r_ir=T(gt.t2r_ir),
                                  lat=T(gt.lat), cand=T(base.cand), topk=T(tk), slow=T(base.slow))
            except Exception as e:  # noqa: BLE001  (an unexpected tables layout: not exported, owner_p1 refuses)
                k = "p1_idx_refused"
                if not self.stats.get(k):
                    self.eng.log(f"nd_export: P1 idx view of {t} rank {r} not exported ({e})")
                self.stats[k] = self.stats.get(k, 0) + 1
        pf = (self.__dict__.get("_pf_view") or {}).get(r)
        if ro == "A" and pf is not None:           # (v3.19) the rank's ONE pf set, shared by every A view
            out["pf"] = pf
        g = s.get("_cxx_idxa_gst")
        if ro == "A" and g is not None and s.get("_cxx_idxa_gst_fixed"):
            nb = int(g.numel() * g.element_size())
            hs = self._gst_h.get((t, r))
            if hs is None or hs[0].numel() != nb:
                hs = self._gst_h[(t, r)] = [torch.zeros(nb, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            out.update(idxa_gst=T(g), idxa_gst_up=self._sets(hs[0], hs[1], ("gst", t, r), r))
        return out

    def _p1_ews(self, r):
        """the EagerWorkspace the rank's idx pieces read: the filled views' GeneralTables.base.ws (one per device and
        budget); none filled yet -> the device's only EagerWorkspace, if exactly one exists; ambiguous -> None"""
        wss = {}
        for t in list(self._mix_filled):
            R, moe, ro = t.split("|")
            st = self._mixed_state((int(R), moe, ro))
            gt = st[r].get("idx_gt") if st is not None else None
            ws = getattr(getattr(gt, "base", None), "ws", None)
            if ws is not None:
                wss[id(ws)] = ws
        if len(wss) == 1:
            return next(iter(wss.values()))
        if len(wss) > 1:
            self.stats["p1_ews_ambiguous"] = self.stats.get("p1_ews_ambiguous", 0) + 1
            return None
        X = sys.modules.get("idx2f")
        EW = getattr(X, "EagerWorkspace", None) if X is not None else None
        dev = torch.device(self.eng.devs[r]).index
        c = [w for k, w in (getattr(EW, "_per", None) or {}).items() if k[0] == dev]
        return c[0] if len(c) == 1 else None

    def _p1_rank(self, r) -> dict:
        """(v3.18) the rank's P1 fields: the indexer's tables2f.Stage ring of the device (SHARED with Python, never created
        here), the EagerWorkspace, Engine.pf_ws[r]. pf_inv: not exported. Missing = zero."""
        out = {}
        st = _stages().get(torch.device(self.eng.devs[r]).index)
        if st is not None and 0 < len(st.hptr) <= 4 and len(st.dptr) == len(st.hptr):
            out.update(idx_ring_id=int(st.eid), idx_ring_n=len(st.hptr), idx_ring_cap=int(st.cap),
                       idx_ring_host=[int(x) for x in st.hptr], idx_ring_dev=[int(x) for x in st.dptr])
        ws = self._p1_ews(r)
        if ws is not None:
            out.update(ews_f32=T(ws.f32), ews_bmax=T(ws.bmax), ews_prep=T(ws.prep))
        pw = getattr(self.eng, "pf_ws", None)
        if pw:
            out["pf_ws"] = T(pw[r])
        nb = (getattr(self.eng, "_nd_bufs", None) or {}).get(r) or {}
        if nb.get("pf_inv") is not None:
            out["pf_inv"] = T(nb["pf_inv"])
        return out

    def _set_hsd_bm(self) -> None:
        """HSD's grammar mask buffers, pre-created here exactly as HSD._build /
        _upload create them at the first grammar step (_bm_h: pinned int32 [T_max, BM_WORDS]; _bm_d[r]: device), so their
        addresses are fixed for graph H's "|bm" variant and nd's mask H2D; exported as hsd_bm_h / hsd_bm_d[r]. Only with the
        engine's bitmask graph variant on (ENG_HEAD_BM_GRAPH): else nothing (nd keeps refusing grammar masks)."""
        eng = self.eng
        EM = sys.modules[type(eng).__module__]
        hsd = getattr(eng.hooks, "hsd", None)
        if not bool(getattr(EM, "HEAD_BM_GRAPH", False)) or hsd is None or not hasattr(self.x, "set_hsd_bm"):
            return
        bmw = None
        for klass in type(hsd).__mro__:
            m = sys.modules.get(klass.__module__)
            if m is not None and hasattr(m, "BM_WORDS"):
                bmw = int(m.BM_WORDS)
        if bmw is None or not hasattr(hsd, "_dims"):
            eng.log("nd_export: HSD has no BM_WORDS / _dims: grammar masks not exported")
            return
        T_ = int(hsd._dims["T"])
        MXH = sys.modules.get("mixed_cxx_head")   # a (re)allocation of what bm graphs read bumps v5's H1 epoch
        if getattr(hsd, "_bm_h", None) is None:
            hsd._bm_h = torch.empty(T_, bmw, dtype=torch.int32, pin_memory=torch.cuda.is_available())
            if MXH is not None and hasattr(MXH, "bump"):
                MXH.bump()
        for r, dv in enumerate(hsd.devs):
            if hsd._bm_d[r] is None:
                with torch.cuda.device(dv):
                    hsd._bm_d[r] = torch.empty(T_, bmw, dtype=torch.int32, device=dv)
                if MXH is not None and hasattr(MXH, "bump"):
                    MXH.bump()
        self.x.set_hsd_bm(T(hsd._bm_h), [T(x) for x in hsd._bm_d])
        eng.log(f"nd_export: HSD grammar masks exported: [{T_}, {bmw}] int32 = {T_ * bmw * 4 / 2 ** 20:.1f} MiB pinned + "
                f"the same per GPU (HSD's own _bm_h / _bm_d, created now instead of at the first grammar step)")
        self.stats["hsd_bm"] = 1

    def _alloc_nd_bufs(self) -> None:
        """the P1 buffers nd needs that Python does not keep, allocated ONCE here, with torch on
        each engine device, under ONE engine attribute eng._nd_bufs[r] (memstats / the soak tools / mpsite see them):
          cp (cfg.cp only): cp_idx i64 [1 << 20] + 2 pinned sets [1 << 20] i64 (FprEhSets, persistent events); the CP mail
             streams through cp.py's own CPPlan._mail_stream (one definition: it creates eng._cp_mail_streams if absent)
          pf (ENG_ATTN_PF only): ONE set per rank, shared by every A view (one A view is live per step; native rewrites are
             stream-ordered behind the previous step on the rank stream): blob i32 [4 S_max + 1] (seq_lens | gather_lens |
             qsl [S_max + 1] | seq_lens // 2), bt i32 [S_max, bt_cols_max], wbt i32 [S_max, wbt_cols_max], loc i32 [R_A, 512],
             comb640 i32 [R_A, 640] + clen640 [R_A], comb128 [R_A, 128] + clen128 [R_A], 2 pinned up sets of
             max(4 (4 S_max + 1), 4 S_max bt_cols_max) bytes; pf_inv i32 [n_paged + 1] per rank
          S_max = cfg.max_seqs + 32 (CP helper shadows hold slots too, as Engram's rows), R_A = the largest A bucket
          (config_json's rnd(max_mixed_rows)), bt_cols_max = ceil(max_model_len / 64), wbt_cols_max = max_model_len / 32 + 1.
        A step beyond these sizes is refused natively (p1::pf_host / cp_host) and runs embedded. One log line per group."""
        import importlib
        import types as _ty
        eng = self.eng
        EM = sys.modules[type(eng).__module__]
        cfg = eng.cfg
        W = len(eng.devs)
        bufs = {r: {} for r in range(W)}
        self._cp_view, self._pf_view = {}, {}
        MB = 1 << 20
        if bool(getattr(cfg, "cp", False)):
            pkg = getattr(EM, "__package__", None)
            CPM = importlib.import_module(".cp", pkg) if pkg else importlib.import_module("cp")
            cap = 1 << 20
            for r, d in enumerate(eng.devs):
                with torch.cuda.device(d):
                    b = bufs[r]
                    b["cp_idx"] = torch.zeros(cap, dtype=torch.int64, device=d)
                    b["cp_up"] = [torch.zeros(cap * 8, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
                    ms = CPM.CPPlan._mail_stream(_ty.SimpleNamespace(eng=eng), d)
                self._cp_view[r] = dict(cp_idx=T(b["cp_idx"]), cp_idx_up=self._sets(b["cp_up"][0], b["cp_up"][1], ("cp", r), r),
                                        cp_mail_stream=int(ms.cuda_stream), cap=cap)
            eng.log(f"nd_export: P1 cp buffers per GPU: {cap * 8 / MB:.1f} MB device + {2 * cap * 8 / MB:.1f} MB pinned "
                    f"(cp_idx [{cap}] i64 + 2 up sets; CP mail streams via cp.py)")
        mml = int(getattr(self, "mml", 0) or _max_model_len(eng)[0])   # (v3.28i) was cfg.max_model_len: 0 on every engine
        if bool(getattr(EM, "ATTN_PF", False)) and mml > 0:
            mr_ = int(getattr(cfg, "mixed_round", 64))
            RA = max(mr_, -(-int(getattr(cfg, "max_mixed_rows", 0)) // mr_) * mr_)
            Sm = int(getattr(cfg, "max_seqs", 80)) + 32
            btc, wbc = -(-mml // 64), mml // 32 + 1
            upb = max(4 * (4 * Sm + 1), 4 * Sm * btc)
            i32 = torch.int32
            dev_b = pin_b = 0
            for r, d in enumerate(eng.devs):
                with torch.cuda.device(d):
                    b = bufs[r]
                    b["pf"] = dict(blob=torch.zeros(4 * Sm + 1, dtype=i32, device=d), bt=torch.zeros(Sm, btc, dtype=i32, device=d),
                                   wbt=torch.zeros(Sm, wbc, dtype=i32, device=d), loc=torch.zeros(RA, 512, dtype=i32, device=d),
                                   comb640=torch.zeros(RA, 640, dtype=i32, device=d), clen640=torch.zeros(RA, dtype=i32, device=d),
                                   comb128=torch.zeros(RA, 128, dtype=i32, device=d), clen128=torch.zeros(RA, dtype=i32, device=d),
                                   up=[torch.zeros(upb, dtype=torch.uint8, pin_memory=True) for _ in range(2)])
                    npg = int(getattr(eng.pools[r], "n_paged", 0) or 0)
                    b["pf_inv"] = torch.full((npg + 1,), -1, dtype=i32, device=d)
                p_ = b["pf"]
                self._pf_view[r] = dict(S_max=Sm, bt_cols_max=btc, wbt_cols_max=wbc, blob=B(p_["blob"]),
                                        up=self._sets(p_["up"][0], p_["up"][1], ("pfup", r), r),
                                        **{k: T(p_[k]) for k in ("bt", "wbt", "loc", "comb640", "clen640", "comb128", "clen128")})
                dev_b = sum(x.numel() * x.element_size() for k, x in p_.items() if k != "up") + b["pf_inv"].numel() * 4
                pin_b = 2 * upb
            eng.log(f"nd_export: P1 pf buffers per GPU: {dev_b / MB:.1f} MB device + {pin_b / MB:.1f} MB pinned (one set per "
                    f"rank shared by every A view: S_max {Sm}, R_A {RA}, bt cols {btc}, wbt cols {wbc}; pf_inv incl.)")
        eng._nd_bufs = bufs
        for r in range(W):
            if r in self._cp_view and hasattr(self.x, "set_cp"):
                self.x.set_cp(r, self._cp_view[r])
        self.stats["nd_bufs_cp"], self.stats["nd_bufs_pf"] = int(bool(self._cp_view)), int(bool(self._pf_view))

    def _set_p1(self) -> None:
        """(v3.18) the ranks' P1 fields + the combine templates, re-exported when they moved. Runs at init, refresh_graphs,
        the end of every ensure_mixed and pull_state: Stage._grow (a Python step meeting a bigger table) re-allocates the
        ring, freeing the old blocks, so nd must see the new pointers before its next native step (nothing of nd is in
        flight at these points). The C side (owner_p1) reads h->ranks[r].idx_ring_* per step, never caches them."""
        if hasattr(self.x, "set_rank_p1"):
            for r in range(len(self.eng.devs)):
                d = self._p1_rank(r)
                sig = repr(sorted(d.items()))
                if self._p1_sig.get(r) != sig:
                    self.x.set_rank_p1(r, d)
                    self._p1_sig[r] = sig
                    self.stats["p1_rank_exports"] = self.stats.get("p1_rank_exports", 0) + 1
        MA = sys.modules.get("mixed_cxx_attn")
        if MA is None or not hasattr(MA, "comb_templates") or not hasattr(self.x, "set_comb"):
            return
        import ctypes
        recs = getattr(MA, "_COMB_RECS", None) or {}
        grew = False
        for r, d in enumerate(self.eng.devs):
            di = torch.device(d).index
            n = sum(len(v) for k, v in recs.items() if k[0] == di)
            if n and n != self._comb_n.get(r):          # (append-only: the count grew)
                b = MA.comb_templates(di)
                self._comb_keep.append(b)               # the parsed records point INTO it: kept for the exporter's life
                self.x.set_comb(r, int(ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value or 0), len(b))
                self._comb_n[r] = n
                grew = True
        if grew:
            self.x.bump_comb_gen()                      # (release, after every rank's pointer is in place)
            self.stats["comb_exports"] = self.stats.get("comb_exports", 0) + 1

    def _set_libs_snap(self) -> None:
        """p1 / idx2f_tab follow when their loaders exist."""
        eng = self.eng
        libs = {}
        drv = getattr(eng, "_cxx", None)
        if drv is not None and getattr(getattr(drv, "x", None), "__file__", None):
            libs["mixed_cxx"] = str(drv.x.__file__)
        tg = getattr(eng, "_tg_mod", None)       # Engine._tg() caches its module here; never build it from the exporter)
        if tg is not None and hasattr(tg, "lib_path"):
            try:
                libs["tail_gather"] = str(tg.lib_path())
            except Exception:  # noqa: BLE001
                pass
        PC = sys.modules.get("pagecopy")
        if PC is not None:
            try:
                libs["pagecopy"] = str(getattr(PC.ext(), "__file__", "") or "")
            except Exception:  # noqa: BLE001
                pass
        H = eng.hooks
        if getattr(H, "engram", None) is not None:
            try:
                libs["engram"] = str(getattr(sys.modules[type(H.engram[0]).__module__].ext(), "__file__", "") or "")
            except Exception:  # noqa: BLE001
                pass
        P1 = sys.modules.get("p1k")        # only when p1k.build() loaded them in-process
        m1 = getattr(P1, "_EXT", {}).get("m") if P1 is not None else None
        if m1 is not None:
            libs["p1"] = str(getattr(m1, "__file__", "") or "")
        st0 = next(iter(_stages().values()), None)   # (v3.18) idx2f_tab = the loaded ext of the indexer's Stage ring
        if st0 is not None:
            libs["idx2f_tab"] = str(getattr(st0.E, "__file__", "") or "")
        libs = {k: v for k, v in libs.items() if v}
        if libs != getattr(self, "_libs", None) and hasattr(self.x, "set_libs"):
            self.x.set_libs(sorted(libs.items()))
            self._libs = libs
            self.stats["libs"] = sorted(libs)
        ME = sys.modules.get("mixed_cxx_moe_ensure")
        if self.__dict__.get("_moe_snap") is None and drv is not None and ME is not None and hasattr(ME, "snapshot") \
                and hasattr(self.x, "set_moe_snap"):
            import ctypes
            try:
                sn = ME.snapshot(drv)
                self._moe_snap = sn
                self.x.set_moe_snap(int(ctypes.addressof(sn)), int(ctypes.sizeof(sn)))
                self.stats["moe_snap"] = 1
            except Exception:  # noqa: BLE001
                pass

    def _set_pagecopy(self) -> None:
        """eng._pcs[r] = dict(d = i64 [2, cap] device, h = [[pinned, persistent Event] x 2]
        cap, i); None / absent = not allocated (nd's own staging is used by page_copies anyway; this only publishes)."""
        if not hasattr(self.x, "set_pagecopy"):
            return
        pcs = getattr(self.eng, "_pcs", None)
        for r in range(len(self.eng.devs)):
            p = pcs[r] if pcs is not None and r < len(pcs) else None
            self.x.set_pagecopy(r, dict(ids=T(p["d"]), up=self._sets_ev(p["h"]), cap=int(p["cap"])) if p else {})

    _SPAN_KINDS = ("embed", "engram_bd", "attn_a", "idx", "attn_b", "moe")
    _SPAN_KINDS_CP = ("attn_bi", "attn_bo", "idx_c", "idx_s", "cp_x1a_send", "cp_x1a_recv", "cp_x2_send", "cp_x2_recv")

    def _set_span_kinds(self) -> None:
        """-1 when the driver or the kind is
        absent (nd / NativeWorker refuse those segments: <= 0 = not registered). The driver loads lazily (eng._cxx) and an
        owner registers its kinds when it loads: re-run at init, refresh_graphs and every ensure_mixed, published only
        when a value changed (14 owner_kind calls)."""
        if not hasattr(self.x, "set_span_kinds") or self.__dict__.get("_span_done"):
            return
        drv = getattr(self.eng, "_cxx", None)
        f = getattr(getattr(drv, "x", None), "owner_kind", None)

        def kinds(names):
            out = []
            for nm in names:
                try:
                    out.append(int(f(nm)) if f is not None else -1)
                except Exception:  # noqa: BLE001
                    out.append(-1)
            return out
        cp_off = os.environ.get("ND_SPAN_CP", "1") == "0"   # CP span kinds -1: CP segments refuse
        ks = kinds(self._SPAN_KINDS)                          # natively and run embedded (the lazy cuBLASLt load vs cp_recv hang)
        kc = [-1] * len(self._SPAN_KINDS_CP) if cp_off else kinds(self._SPAN_KINDS_CP)
        if cp_off and not self.__dict__.get("_span_cp_logged"):
            self._span_cp_logged = True
            self.eng.log("nd_export: ND_SPAN_CP=0: CP span kinds not published (-1): CP segments refuse natively, run embedded")
        self._span_done = all(k >= 0 for k in ks) and (cp_off or all(k >= 0 for k in kc))
        if (ks, kc) == self.__dict__.get("_span_k"):
            return
        self._span_k = (ks, kc)
        self.x.set_span_kinds(ks)
        if hasattr(self.x, "set_span_kinds_cp"):
            self.x.set_span_kinds_cp(kc)
        self.stats["span_kinds"], self.stats["span_kinds_cp"] = ks, kc

    def _engram_rebind_check(self) -> None:
        """re-export the Engram views when a rank's bound start source / ids moved since the last export
        (bind_start / bind after the exporter's init: never in serving today, cheap to check at every refresh)."""
        H = self.eng.hooks
        if getattr(H, "engram", None) is None:
            return
        now = [(int(getattr(e, "_start_src", None).data_ptr()) if getattr(e, "_start_src", None) is not None else 0,
                int(e.b_ids.data_ptr())) for e in H.engram]
        if now != getattr(self, "_eg_bound", None):
            if getattr(self, "_eg_bound", None) is not None:
                self.stats["engram_reexports"] = self.stats.get("engram_reexports", 0) + 1
            self._set_engram()

    def _set_engram(self) -> None:
        """(v0.7 slice 2) engram_lib + every rank's FprEhEngramRank (re-run after a bind_start / bind change)."""
        H = self.eng.hooks
        if getattr(H, "engram", None) is None or not hasattr(self.x, "set_engram"):
            return
        eg0 = H.engram[0]
        try:
            lib = str(getattr(sys.modules[type(eg0).__module__].ext(), "__file__", "") or "")
        except Exception:  # noqa: BLE001
            lib = ""
        self.x.set_engram_lib(lib)
        for r in range(len(self.eng.devs)):
            self.x.set_engram(r, self._engram_rank(r))
        self.stats["engram_exported"] = int(bool(lib))
        self._eg_bound = [(int(getattr(e, "_start_src", None).data_ptr()) if getattr(e, "_start_src", None) is not None else 0,
                           int(e.b_ids.data_ptr())) for e in H.engram]

    def ptr(self) -> int:
        return int(self.x.ptr())

    def config_json(self) -> str:
        """v0.4 config_json: the engine's decode configuration nd::Engine::init reads (d13_cxx._cfg / _hcfg + the D13 /
        D13S / HSDT c1d / C1 mega flags + the buckets)."""
        import json
        eng, d13, M, H = self.eng, self.d13, self.M, self.eng.hooks
        EM = sys.modules[type(eng).__module__]
        hsd = H.hsd
        c = dict(graphs=bool(eng.cfg.graphs), largest_bucket=int(eng.cfg.buckets[-1]), graph_max_ctx=int(d13.E.GRAPH_MAX_CTX),
                 k1=int(M.K1), delta_cap=int(M.DELTA_CAP), fb_reuse=bool(M.FB_REUSE), scratch=[int(x.scratch) for x in d13.rx],
                 buckets=[int(b) for b in eng.cfg.buckets],
                 gt=bool(M.GT), owner=bool(M.OWNER), empty=bool(M.EMPTY), adv_slice=bool(M.ADV_SLICE), tfuse=bool(M.TFUSE),
                 d13s=bool(getattr(d13, "steady", None) is not None and d13.steady.live),
                 check=bool(M.CHECK or M.GT_CHECK or M.OWNER_CHECK or M.EMPTY_CHECK or M.TFUSE_CHECK
                            or int(os.environ.get("ENG_D13S_CHECK_EVERY", "0") or 0) > 0
                            or os.environ.get("ENG_D13S", "") == "shadow"),
                 c1_mega=bool(getattr(EM, "C1_MEGA", False)), **_c1m_facts(eng, EM),
                 head_bm=bool(getattr(EM, "HEAD_BM_GRAPH", False)),   # graph H's "|bm" variant is captured
                 peer_map=os.environ.get("ENG_PEER_MAP", "1") == "1",   # engine._torch_peer_maps ran (ring copy P2P)
                 # (v3.27, P4) decode steps with Worker.copies / migrated sequences natively (0 = embedded, as before)
                 decode_copies=os.environ.get("ND_DECODE_COPIES", "1") == "1",
                 decode_mig=os.environ.get("ND_DECODE_MIG", "0") == "1",   # (v3.27: migrated device tokens, not gated)
                 # (v3.30) long-context decode: engine.step's long-variant rule (nd_hist.h long_level_of)
                 long_level_ctx=[int(x) for x in getattr(EM, "LONG_LEVEL_CTX", ())],
                 long_graphs=bool(getattr(EM, "LONG_GRAPHS", False)), long_any_rows=bool(getattr(EM, "LONG_ANY_ROWS", False)),
                 split_hsd=bool(getattr(eng.cfg, "split_hsd", False)), dbg_sync=bool(getattr(EM, "DBG_SYNC", False)),
                 # (v3.33, slice 2) the async long variant natively (needs FPR's d13 step 1: Python captures (Rk, "LA"))
                 long_async=os.environ.get("ND_LONG", "0") == "1",
                 # (v3.34, = D13._long_level) a C1 lane HSD (FPR_C1_KMAX > 3) or variable verify rows: long steps stay sync
                 long_c1_lane=bool(getattr(d13, "lane", None) is not None or getattr(d13, "vrows", None) is not None))
        mr_ = int(getattr(eng.cfg, "mixed_round", 64))
        rA_ = max(mr_, -(-int(getattr(eng.cfg, "max_mixed_rows", 0)) // mr_) * mr_)
        c["mixed_max_rows"] = max(rA_, int(getattr(eng, "max_tail_rows", 0) or 0))
        # ND-2: the extensions nd calls through their C tables (the engine's own loaded builds: dlopen NOLOAD)
        # what native decode does not do (async_ok refuses), RES_OWNER / ERR_EVERY
        ext = {}
        try:
            ext["rowexp"] = str(getattr(M.rowexp_ext(), "__file__", "") or "")
        except Exception:  # noqa: BLE001  (nd's init then fails loudly: "no extension path")
            ext["rowexp"] = ""
        try:   # (v3.10, optional) the pagecopy extension (its fpr_pagecopy_table: nd's native page copies)
            import importlib
            PC = sys.modules.get("pagecopy")
            if PC is None:
                pd_ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(EM.__file__))), "kernels", "pagecopy")
                if pd_ not in sys.path:
                    sys.path.insert(0, pd_)
                PC = importlib.import_module("pagecopy")
            ext["pagecopy"] = str(getattr(PC.ext(), "__file__", "") or "")
        except Exception:  # noqa: BLE001  (no build: native page copies refused)
            ext["pagecopy"] = ""
        if getattr(d13, "steady", None) is not None:
            try:
                DS = sys.modules.get("engine.d13s") or sys.modules.get(type(d13.steady).__module__)
                ext["hsdprep"] = str(getattr(DS.ext(), "__file__", "") or "")
            except Exception:  # noqa: BLE001
                ext["hsdprep"] = ""
        c.update(ext=ext, gate=bool(getattr(H, "gate", None) is not None), nancheck=bool(getattr(EM, "NANCHECK", False)),
                 vrows=bool(getattr(d13, "vrows", None) is not None), res_owner=bool(getattr(M, "RES_OWNER", False)),
                 err_every=int(getattr(M, "ERR_EVERY", 1)))
        if hsd is not None:
            HM = sys.modules[type(hsd).__module__]
            HS = HM if hasattr(HM, "_Step") else sys.modules.get("hsd", HM)
            c.update(k=int(hsd.k), hsd_max_seqs=int(hsd.max_seqs), hsd_seq_buckets=[int(x) for x in hsd.seq_buckets],
                     hsd_row_buckets=[int(x) for x in hsd.row_buckets], hsd_nt_max=int(hsd.ranks0[0].b["aux"].shape[0]),
                     hsd_pool=bool(hsd.pool), hsd_window=int(HS.DR.WINDOW), hsd_ring=int(HS.DR.RING),
                     c1d=bool(getattr(hsd, "_c1d_cfg", False)), c1d_smax=int(getattr(HM, "SMAX", 0) or 0))
        return json.dumps(c, sort_keys=True)

    # ------------------------------------------------------------------ events
    def ev(self, key, r):
        e = self._ev.get(key)
        if e is None:
            eng = self.eng
            with torch.cuda.device(eng.devs[r]):
                e = torch.cuda.Event()
                e.record(eng.streams[r])
            self._ev[key] = e
        return e

    def _sets(self, h0, h1, key, r):
        return (B(h0), B(h1), int(self.ev(key + (0,), r).cuda_event), int(self.ev(key + (1,), r).cuda_event))

    # ------------------------------------------------------------------ per rank (A)
    def _rank(self, r) -> dict:
        eng, d13, H = self.eng, self.d13, self.eng.hooks
        rx = d13.rx[r]
        hsd = H.hsd
        eg = H.engram[r] if H.engram is not None else None
        if hsd is not None and getattr(H, "_dt_d", None) is None:   # Hooks._device_tokens_v2's buffers, created
            nmax = 4096                                            # now as its first call would (n <= 4,096): nd writes the
            H._dt = [[(torch.zeros(2, nmax, dtype=torch.int64, pin_memory=True), None) for _ in range(2)]   # same dt_d
                     for _ in range(len(eng.devs))]
            H._dt_d = [torch.zeros(2, nmax, dtype=torch.int64, device=d) for d in eng.devs]
            H._dt_i = [0] * len(eng.devs)
        if hsd is not None:     # HSD's blob events are persistent already (_hev[hi][r], created on first upload): create now
            for i in range(2):
                if hsd._hev[i][r] is None:
                    with torch.cuda.device(eng.devs[r]):
                        hsd._hev[i][r] = torch.cuda.Event()
                        hsd._hev[i][r].record(eng.streams[r])
        if eg is not None:      # Engram's stage events are persistent too (_stage_ev[i])
            for i in range(2):
                if eg._stage_ev[i] is None:
                    with torch.cuda.device(eng.devs[r]):
                        eg._stage_ev[i] = torch.cuda.Event()
                        eg._stage_ev[i].record(eng.streams[r])
        o = rx._o32
        dr, dc, dp = d13.dl_d[r]
        d = dict(
            device=torch.device(eng.devs[r]).index, step_stream=int(eng.streams[r].cuda_stream),
            xstream=int(d13.xstream[r].cuda_stream), side_stream=int(eng.side[r].cuda_stream),
            estream=int(eng.estream[r].cuda_stream),
            step_event0=int(self.ev(("step", r, 0), r).cuda_event), step_event1=int(self.ev(("step", r, 1), r).cuda_event),
            rx_b32_d=B(rx._b32), rx_b64_d=B(rx._b64),
            rx_h32=self._sets(rx._h[0][0], rx._h[1][0], ("rx", r), r),
            rx_h64=self._sets(rx._h[0][1], rx._h[1][1], ("rx", r), r),   # one alternation (rx._hi), the same events
            rx_o32=[o["slot"], o["row0"], o["nrows"], o["kind"], o["counts"], o["ops"]],
            rx_ring=T(rx.ring), rx_dbt=T(rx.dbt), rx_nc=T(rx.nc), rx_err=T(rx.err), rx_counts=T(rx.d["counts"]),
            rx_start=T(rx.out["start"]), rx_seq=T(rx.out["seq"]),
            rx_scratch_slot=int(rx.scratch), rx_S_max=int(rx.S_max), rx_R_max=int(rx.R_max), rx_ops_max=int(rx.ops_max),
            dl_rows_d=T(dr), dl_cols_d=T(dc), dl_pages_d=T(dp),           # K: three buffers, one alternation, the same events
            dl_rows=self._sets(d13.dl_h[0][r][0], d13.dl_h[1][r][0], ("dl", r), r),
            dl_cols=self._sets(d13.dl_h[0][r][1], d13.dl_h[1][r][1], ("dl", r), r),
            dl_pages=self._sets(d13.dl_h[0][r][2], d13.dl_h[1][r][2], ("dl", r), r), delta_cap=int(self.M.DELTA_CAP),
            nb_d=B(d13.nb_d[r]), nb=self._sets(d13.nb_h[0][r], d13.nb_h[1][r], ("nb", r), r),
            d2h_pos0=T(d13.d2h[0][r][0]), d2h_pos1=T(d13.d2h[1][r][0]),
            d2h_own0=T(d13.d2h[0][r][1]), d2h_own1=T(d13.d2h[1][r][1]),
            qidx_d=B(d13.qidx_d[r]), qidx=self._sets(d13.qidx_h[0][r], d13.qidx_h[1][r], ("qidx", r), r),
            num_sampled=T(hsd.st[r].num_sampled) if hsd is not None else None,
            engram_pk_d=B(eg._pk) if eg is not None else None,
            engram_stage=((B(eg._stage[0]), B(eg._stage[1]), int(eg._stage_ev[0].cuda_event), int(eg._stage_ev[1].cuda_event))
                          if eg is not None else None),
            engram_pk_off=([eg._pk_off[k] for k in ("ids", "row0", "nrows", "start", "slot", "look", "counts")]
                           if eg is not None else []),
            engram_b_start=T(eg.b_start) if eg is not None else None,
            engram_b_ids=T(eg.b_ids) if eg is not None else None,
            hsd_dblob=B(hsd._dblob[r]) if hsd is not None else None,
            hsd_hblob=((B(hsd._hblob[0][r]), B(hsd._hblob[1][r]), int(hsd._hev[0][r].cuda_event), int(hsd._hev[1][r].cuda_event))
                       if hsd is not None else None),
            next_ids=T(hsd.next_ids[r]) if hsd is not None else None,
            num_emitted=T(hsd.num_emitted[r]) if hsd is not None else None,
            sampled=T(hsd.st[r].sampled) if hsd is not None else None,
            next_ids_h=None, num_emitted_h=None, sampled_h=None,
            dt_d=T(H._dt_d[r]) if getattr(H, "_dt_d", None) is not None else None,
            idx_dbt=T(H.idx.views[r]["dbt"]) if H.idx is not None else None)
        # v0.5
        ro = d13.rx_own[r] if getattr(d13, "rx_own", None) else {}
        d.update(h1_h_own=T(hsd.h_own[r]) if hsd is not None else None,
                 rx_own_pos=T(ro.get("pos")), rx_own_own=T(ro.get("own")), rx_own_si=T(ro.get("si")),
                 rx_own_sl=T(ro.get("sl")), rx_own_paged=T(ro.get("paged")), pools=self._pools(r))
        # v0.6: the tail gather index per rank, Engine._tix[r] = dict(d = i64
        # [max_tail_rows], h = [[pinned, persistent Event] x 2]); None when the flag is off (nd refuses the T stage)
        tix = getattr(self.eng, "_tix", None)
        tx = tix[r] if tix is not None and r < len(tix) else None
        d.update(tail_ix=T(tx["d"]) if tx is not None else None,
                 tail_ix_up=self._sets_ev(tx["h"]) if tx is not None else None)
        return d

    @staticmethod
    def _sets_ev(h) -> tuple:
        """(v0.6) an engine-owned staging pair [[pinned, persistent Event] x 2] as an FprEhSets: the ENGINE's events (not the exporter's), so either side may pick
        either set (wait-then-re-record on the same events)."""
        return (B(h[0][0]), B(h[1][0]), int(h[0][1].cuda_event), int(h[1][1].cuda_event))

    def _vidx_view(self, s) -> dict:
        """(v0.6) ENG_MIX_PERSIST's per-view Engine._set_vidx buffers: vidx_d (u8 blob, the view's own R layout), the
        full views (vidx, nv, nd) = s["_vidx_v"], vidx_up = s["vidx_h"] (persistent events). None = flag off (nd refuses
        SET_VIDX: vidx_d NULL)."""
        if s.get("vidx_d") is None or s.get("_vidx_v") is None or s.get("vidx_h") is None:
            return dict(vidx_d=None, vidx=None, nv=None, nd=None, vidx_up=None)
        vv, nvv, ndv = s["_vidx_v"]
        return dict(vidx_d=B(s["vidx_d"]), vidx=T(vv), nv=T(nvv), nd=T(ndv), vidx_up=self._sets_ev(s["vidx_h"]))

    def _pools(self, r) -> list:
        """v0.5 (3): the rank's KV pools by CopyOp.pool name (serve/worker.py Pools; geometry.py page sizes): base,
        bytes per page, pages (window: incl. the scratch page). Allocated once at init, never moved."""
        P = (getattr(self.eng, "pools", None) or [None] * (r + 1))[r]
        if P is None:
            return []
        import geometry as G      # (serve/ is on sys.path: engine.py and worker.py import it as `geometry`)
        out = []
        for name, pb, n in (("paged", G.PAGE_BYTES, getattr(P, "n_paged", 0)),
                            ("window", G.WINDOW_PAGE_BYTES, getattr(P, "n_window", 0) + 1),
                            ("state", G.STATE_SLOT_BYTES, getattr(P, "n_state", 0))):
            t = getattr(P, name, None)
            if t is not None:
                assert t.numel() * t.element_size() >= pb * n, (name, t.numel(), pb, n)
                out.append((name, int(t.data_ptr()), int(pb), int(n)))
        return out

    # ------------------------------------------------------------------ per (bucket, rank): ensure (rule (a)(i)) + fill
    def ensure_bucket(self, R: int) -> None:
        """Loop thread, step boundary, nothing enqueued: the decode bucket's state, every lazy buffer, graph T for
        has_seqs 0 / 1 on every rank; then its views (and a generation bump if the engine re-created the bucket)."""
        self._h1_fresh = None          # (v3.28jR) Python may follow: the next refresh_graphs scans
        eng, d13, H, M = self.eng, self.d13, self.eng.hooks, self.M
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nd_cxx
        EM = sys.modules[type(eng).__module__]
        IM = sys.modules[type(H.idx).__module__]
        if R not in self._state_of:
            self._late_pub("bucket", int(R))   # a decode bucket first met at serve time
        st = eng.bucket(int(R))
        Rr = int(st[0].get("R", R))   # rows; R = the bucket KEY (Rk = R + lvl * 2^20 for a long variant: keys only)
        if R in self._state_of and self._state_of[R] is not st:
            self.x.bump(int(R))
        self._state_of[R] = st
        for r in range(len(eng.devs)):
            s, rx = st[r], d13.rx[r]
            d = s["idx"]
            Bn = d["B"]
            with torch.cuda.device(eng.devs[r]), torch.cuda.stream(eng.streams[r]):
                if s.get("in_h") is None:
                    nb = s["in_d"].numel()
                    s["in_h"] = [[torch.zeros(nb, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
                    s["in_hv"] = [EM._in_views(x[0], Rr, numpy=True) for x in s["in_h"]]
                    s["in_i"] = 0
                if d["up_h"] is None:
                    nbytes = d["up_d"].numel()
                    d["up_h"] = [[torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
                    d["up_hv"] = [IM._up_views(x[0], Rr, Bn, numpy=True) for x in d["up_h"]]
                    d["up_i"] = 0
                if "paged_d" not in s:
                    s["paged_d"] = torch.full((Rr,), -1, dtype=torch.int64, device=s["dev"])
                if s.get("_own_rbase") is None:
                    s["_own_rbase"] = torch.empty_like(d["rslot"])
                eg = H.engram[r] if H.engram is not None else None
                if eg is not None and s.get("_own_sbase") is None:
                    s["_own_sbase"] = torch.empty_like(eg.b_start)
                rx.bind_outputs(pos=s["pos"], own=s["own"], si=s["si"].view(Rr, M.SW), sl=s["sl"], paged=s["paged_d"])
                for has in (False, True):
                    nd_cxx._gt_ensure(d13, r, s, R, has)
                lay, _ = EM._in_layout(Rr)
                uoff, off = {}, 0
                for nm, dt, cnt in IM._up_fields(Rr, Bn):
                    uoff[nm] = off
                    off += -(-(cnt * np.dtype(dt).itemsize) // 8) * 8
                v = dict(moe_kind=0, in_d=B(s["in_d"]), **{"in": self._sets(s["in_h"][0][0], s["in_h"][1][0], ("in", R, r), r)},
                         in_off_tok=lay["tok"][0], in_off_valid=lay["valid"][0], in_off_samp=lay["samp"][0],
                         up_d=B(d["up_d"]), up=self._sets(d["up_h"][0][0], d["up_h"][1][0], ("up", R, r), r),
                         up_off_valid=uoff["valid"], up_off_pslot=uoff["pslot"], up_off_rs=uoff["rs"], up_B=int(Bn),
                         pos=T(s["pos"]), own=T(s["own"]), si=T(s["si"]), sl=T(s["sl"]), paged=T(s["paged_d"]), tok=T(s["tok"]),
                         idx_rslot=T(d["rslot"]), idx_valid=T(d["valid"]),
                         idx_seq_lens1=T(d["seq_lens"][1]), idx_seq_lens2=T(d["seq_lens"][2]),
                         idx_cslot1=T(d["cslot"][1]), idx_cslot2=T(d["cslot"][2]),
                         own_rbase=T(s["_own_rbase"]), own_sbase=T(s.get("_own_sbase")),
                         pos_ev0=int(self.ev(("pos", R, r, 0), r).cuda_event), pos_ev1=int(self.ev(("pos", R, r, 1), r).cuda_event))
            self.x.set_bucket(int(R), r, v)
        self.refresh_graphs()

    # ------------------------------------------------------------------ graphs (append-only, F)
    # ------------------------------------------------------------------ the publication fast path
    def _graphs_fp(self) -> tuple:
        """what refresh_graphs' three walks read: container ids / sizes, value ids, per-rank element ids, and the owners'
        monotonic capture / recapture / retire / stale counters"""
        def fpd(d):
            if not isinstance(d, dict):
                return (id(d),)
            vs = list(d.values())
            inner = []
            for v in vs:
                if isinstance(v, (list, tuple)):
                    inner.extend(map(id, v))
                    v0 = v[0] if v else None
                    if isinstance(v0, list):            # MixedHead.graphs: ([CUDAGraph per rank], handles)
                        inner.extend(map(id, v0))
            return (id(d), len(vs), hash(tuple(map(id, vs))), hash(tuple(inner)))

        def ctr(o):
            st_ = getattr(o, "stats", None)
            if not isinstance(st_, dict):
                return ()
            return tuple(sorted((k, v) for k, v in st_.items() if isinstance(v, int) and not isinstance(v, bool)
                                and ("capt" in k or "retir" in k or "stale" in k)))
        eng = self.eng
        mh = getattr(eng.hooks, "__dict__", {}).get("_mixed_head")
        # MixedHead.graphs (thousands of H1 keys at the full grid) by identity + length + its counters, O(1)
        # MixedHead inserts a key only at a capture of a key NOT in graphs (captures++) and removes one only by pop (stale /
        # batch failure: len, stale / precapture_guard_failed move); it never replaces a value in place. eng.graphs / d13.gt
        # (tens of decode keys) keep the per-value walk.
        mg = getattr(mh, "graphs", None) if mh is not None else None
        if mh is None:
            mfp = (0,)
        elif _h1_ctrs_ok(mh, getattr(mh, "stats", None)):
            # the dict whose id is in the fingerprint stays pinned (_fp_pin): no id reuse after a rebind
            mfp = (id(mg), len(mg) if isinstance(mg, dict) else -1) + tuple(int(mh.stats.get(n_, 0)) for n_ in _H1_SIG_CTRS)
            self._fp_pin = mg
        else:
            mfp = ("walk",) + fpd(mg)
        fast = (fpd(eng.graphs), fpd(self.d13.gt), id(mh), mfp, ctr(self.d13), ctr(mh), ctr(eng))
        if not PUB_VERIFY:
            return fast
        return fast + (fpd(mg) if mh is not None else (0,),)

    def _pub_t(self, part, t0):
        if PUB_PROF:
            t1 = time.perf_counter()
            self._pub_us[part] = self._pub_us.get(part, 0.0) + (t1 - t0) * 1e6
            return t1
        return t0

    def _pub_log(self) -> None:
        st = self.stats
        n = st["pub_calls"]
        if n in (1, 64) or n % PUB_LOG_EVERY == 0:
            us = " ".join(f"{k} {v / n:.0f}" for k, v in sorted(self._pub_us.items())) if PUB_PROF else "(ND_PUB_PROF=0)"
            self.eng.log(f"nd_export: publication calls {n} walks skipped {st['pub_walks_skipped']} walked {st['pub_walks']} "
                         f"verify {st['pub_verify']} missed {st['pub_missed']} fp_miss {st.get('pub_fp_miss', 0)} h1sig_miss "
                         f"{H1X_STATS['sig_miss']} extend_miss {H1X_STATS['extend_miss']} h1 rebuilds {H1X_STATS['rebuilds']} extends {H1X_STATS['extends']} | h1 dirty walks {st.get('h1_dirty_walks', 0)} full {st.get('h1_full_walks', 0)} rewraps "
                         f"{st.get('h1_rewraps', 0)} dirty_verify {st.get('dirty_verify', 0)} dirty_miss {st.get('dirty_miss', 0)} | us/call {us}"
                         + (f" | late {dict(sorted(self._late.items()))}" if self._native_done else " | native done not marked"))

    def refresh_graphs(self) -> int:
        eng, d13 = self.eng, self.d13
        n0 = self.x.graph_count()
        added = 0
        st_ = self.stats
        st_["pub_calls"] += 1
        tq = time.perf_counter() if PUB_PROF else 0.0
        fp0 = self._graphs_fp() if PUB_ON else None
        would_skip = PUB_ON and self._pub_fp is not None and fp0[:7] == self._pub_fp[:7]   # [7:] = VERIFY's full fp
        if would_skip and PUB_VERIFY and fp0[7:] != self._pub_fp[7:]:
            st_["pub_fp_miss"] = st_.get("pub_fp_miss", 0) + 1
            if st_["pub_fp_miss"] <= 16:
                eng.log("nd_export: PUB FP MISS: MixedHead.graphs changed at an unchanged O(1) fingerprint")
        skip = would_skip and not PUB_VERIFY       # only the three graph walks are skipped
        st_["pub_walks_skipped" if skip else "pub_walks"] += 1
        tq = self._pub_t("fp", tq)
        for key, gs in ([] if skip else list(eng.graphs.items())):
            if not (isinstance(key, tuple) and len(key) >= 2 and key[1] in ("LA", "H")):
                continue
            kind = 0 if key[1] == "LA" else 1
            n = int(key[2][1]) if kind == 1 and isinstance(key[2], tuple) and len(key[2]) > 1 else 0
            for r, gr in enumerate(gs):
                pieces = gr if isinstance(gr, tuple) else (gr,)
                for p, g in enumerate(pieces):
                    ident = (key, r, p)
                    if self._pub.get(ident) is g:
                        continue
                    self.x.add_graph(kind, r, int(key[0]), n, p, int(g.raw_cuda_graph_exec()), _key_text(key))
                    if r == 0 and p == 0:
                        self._late_pub("LA" if kind == 0 else "H", key)
                    self._pub[ident] = g
                    self._keep.append(g)
                    added += 1
        for k, (g, _ptrs) in ([] if skip else list(d13.gt.items())):
            if len(k) != 3:            # (R, r, "own", n): graph T' is not handed over in α (E)
                continue
            ident = (("T",) + tuple(k), int(k[1]), 0)
            if self._pub.get(ident) is g:
                continue
            self.x.add_graph(2, int(k[1]), int(k[0]), int(bool(k[2])), 0, int(g.raw_cuda_graph_exec()), _key_text(("T",) + tuple(k)))
            self._late_pub("T", ("T",) + tuple(k))   # per (R, rank, has_seqs), as D13's gt_captures
            self._pub[ident] = g
            self._keep.append(g)
            added += 1
        # v0.5 (2): H1 (mixed_cxx_head.MixedHead: (S_pad, D_pad, NT) -> ([CUDAGraph per rank], handles)), kind 4
        tq = self._pub_t("walk_la_h_t", tq)
        mh = getattr(eng.hooks, "__dict__", {}).get("_mixed_head")
        hsd = eng.hooks.hsd
        self._engram_rebind_check()
        self._set_libs_snap()                 # (slice 4: libs / MoeSnap appear when their lazy owners load)
        self._set_span_kinds()                # (v3.20: the driver / owners load lazily)
        self._set_p1()                        # (v3.18)
        if H1_REUSE and mh is not None and self.__dict__.get("_h1_fresh") == (id(mh), int(eng.step_serial)):
            self.stats["h1_guard_reused"] = self.stats.get("h1_guard_reused", 0) + 1   # (v3.28jR) pull_state's scan, same boundary
        else:
            self._h1_guard(mh, publish=False)     # (v3.4) guarded keys only: a STALE one is retired before it is published
        self._h1_fresh = None
        tq = self._pub_t("derived_guard", tq)
        if would_skip and self._graphs_fp()[:7] != fp0[:7]:   # the guard retired something: walk the H1 keys after
            skip = would_skip = False
        if mh is not None and hsd is not None and not skip:
            added += self._walk_h1(mh, hsd)   # dirty keys only once MixedHead.graphs is wrapped
        tq = self._pub_t("walk_h1", tq)
        # unpublish only after a walk: with the fingerprint unchanged since the last call (which ran it) MixedHead.graphs
        # lost no key, so there is nothing to retire; a guard retirement above already turned skip off
        nun = 0 if skip else self._h1_unpublish(mh)
        if PUB_VERIFY and not skip and mh is not None and hsd is not None and self._h1_dd is not None:
            # VERIFY: the full walk + full unpublish after the dirty path; anything they still do is a dirty miss
            dm = self._walk_h1(mh, hsd, full=True) + self._h1_unpublish(mh, full=True)
            if dm:
                st_["dirty_miss"] = st_.get("dirty_miss", 0) + dm
                added += dm
                if st_["dirty_miss"] <= 16:
                    eng.log(f"nd_export: DIRTY MISS: {dm} H1 publish / unpublish actions the dirty path missed")
            st_["dirty_verify"] = st_.get("dirty_verify", 0) + 1
        if added:
            self.x.publish()
        tq = self._pub_t("unpublish_publish", tq)
        if PUB_ON:
            if would_skip and (added or nun) and PUB_VERIFY:  # a graph the fingerprint did not see: a blind spot
                st_["pub_missed"] += added + nun
                if st_["pub_missed"] <= 16:
                    eng.log(f"nd_export: PUB MISSED: {added} graph(s) published at an unchanged fingerprint")
            if would_skip and PUB_VERIFY:
                st_["pub_verify"] += 1
            self._pub_fp = self._graphs_fp()
            self._pub_t("fp_end", tq)
        self._pub_log()
        if PRECAP_AUTO and not self._native_done and not self.__dict__.get("_precap_busy"):
            self._precap_auto()
        return int(self.x.graph_count()) - int(n0)

    def _free_gib(self) -> list:
        out = []
        for d in self.eng.devs:
            try:
                out.append(round(torch.cuda.mem_get_info(torch.device(d))[0] / 2 ** 30, 3))
            except Exception:  # noqa: BLE001
                out.append(None)
        return out

    def _precap_auto(self) -> None:
        """ND_PRECAP_AUTO's trigger (module comment); never re-entered (precapture_all's ensure_bucket refreshes)"""
        a, eng = PRECAP_AUTO, self.eng
        if a == "pc":
            due = hasattr(eng, "_pc") and eng._pc is None
        elif a.startswith("n:"):
            due = self.stats["pub_calls"] >= int(a[2:] or "0")
        else:
            due = False
        if not due:
            return
        self._precap_busy = True
        try:
            for d in eng.devs:
                torch.cuda.synchronize(torch.device(d))
            f0 = self._free_gib()
            out = self.precapture_all()
            for d in eng.devs:
                torch.cuda.synchronize(torch.device(d))
            f1 = self._free_gib()
            dl = [None if x is None or y is None else round(y - x, 3) for x, y in zip(f0, f1)]
            self.stats["precapture_free_gib"] = dict(before=f0, after=f1, delta=dl)
            eng.log(f"nd_export: ND_PRECAP_AUTO={a}: precapture {out} | free GiB per GPU before {f0} after {f1} delta {dl}")
            self.mark_native_done()
        finally:
            self._precap_busy = False

    # ------------------------------------------------------------------ precapture + late publishes
    def precapture_all(self) -> dict:
        """ensure every decode bucket now (graph T has_seqs 0 / 1 on every rank + its views), so nd
        never meets a new bucket while serving. Once, at the native done point (engine main thread, nothing of nd in flight:
        the ensure_bucket rule). ND_PRECAP_LONG=1: the ND_LONG variants too. Returns counts + seconds."""
        eng = self.eng
        EM = sys.modules[type(eng).__module__]
        keys = [int(R) for R in eng.cfg.buckets]
        tag = int(getattr(EM, "LVL_TAG", 1 << 20))
        nlv = len(getattr(EM, "LONG_LEVEL_CTX", ()) or ())
        if PRECAP_LONG and nlv:
            keys += [int(R) + lvl * tag for lvl in range(1, nlv + 1) for R in eng.cfg.buckets]
        t0 = time.perf_counter()
        g0 = int(self.d13.stats.get("gt_captures", 0)) if isinstance(getattr(self.d13, "stats", None), dict) else -1
        done = 0
        for R in keys:
            if R not in self._state_of:
                self.ensure_bucket(R)
                done += 1
        out = dict(keys=len(keys), ensured=done, gt_captures=(int(self.d13.stats.get("gt_captures", 0)) - g0) if g0 >= 0 else -1,
                   s=round(time.perf_counter() - t0, 2), long=bool(PRECAP_LONG and nlv))
        self.stats["precapture"] = out
        eng.log(f"nd_export: precapture_all {out}")
        return out

    def mark_native_done(self) -> None:
        """from now on a publish is LATE (serve time): counted by kind, logged with its key"""
        self._native_done = True
        self.eng.log("nd_export: native done mark: later graph / bucket publishes count as late")

    def _late_pub(self, kind: str, key) -> None:
        if not self._native_done:
            return
        n = self._late[kind] = self._late.get(kind, 0) + 1
        self.stats["late_" + kind] = n
        if sum(self._late.values()) <= LATE_LOG_MAX:
            self.eng.log(f"nd_export: LATE publish {kind} {_key_text(key) if isinstance(key, tuple) else key} (n {n})")

    def _h1_unpublish(self, mh, full: bool = False) -> int:
        """H1 keys published before and no longer captured: mixed_cxx_head's pointer guard retired
        them (its graphs may point at buffers the eager head replaced). Bump the key's generation slot so nd skips its
        entries (F); a later re-capture publishes new entries at the new generation (the idents are dropped here).
        Once MixedHead.graphs is nd's _DirtyDict: only the keys removed since the last call that are still absent
        (the same rule: a key is unpublished iff it is not live now), through the key -> idents index; full=True or an
        unwrapped dict = the whole _pub."""
        mg = getattr(mh, "graphs", None) if mh is not None else None
        if not full and not self._h1_unpub_full and mg is not None and mg is self._h1_dd and type(mg) is _DirtyDict:
            rm, mg._rm = mg._rm, {}
            n = 0
            for k in rm:
                if dict.__contains__(mg, k):
                    continue
                for ident in self._h1_idents.pop(tuple(k), {}):
                    if ident[1] == 0:
                        self.x.bump_h1(_key_text(ident[0][1:]))
                        self.stats["h1_unpublished"] += 1
                        n += 1
                    self._pub.pop(ident, None)
            return n
        live = set(tuple(k) for k in (mg if mg is not None else {}))
        if mg is not None and mg is self._h1_dd and type(mg) is _DirtyDict:
            mg._rm = {}                     # (a full pass covers every pending removal)
            self._h1_unpub_full = False     # (the wrapper's record is complete from here on)
        n = 0
        for ident in [i for i in self._pub if isinstance(i[0], tuple) and i[0][:1] == ("H1",) and i[0][1:] not in live]:
            k = ident[0][1:]
            if ident[1] == 0:
                self.x.bump_h1(_key_text(k))
                self.stats["h1_unpublished"] += 1
                n += 1
            del self._pub[ident]
            d_ = self._h1_idents.get(k)
            if d_ is not None:
                d_.pop(ident, None)
                if not d_:
                    del self._h1_idents[k]
        return n

    def _walk_h1(self, mh, hsd, full: bool = False) -> int:
        """v0.5 (2): H1 (mixed_cxx_head.MixedHead: (S_pad, D_pad, NT) -> ([CUDAGraph per rank], handles)), kind 4.
        MixedHead.graphs is replaced by
        nd's _DirtyDict (a dict subclass recording every written / removed key, as eng.bk's _NotifyDict); then only the
        keys written since the last walk are visited, in the dict's order, with the same per-key body. The first walk, a
        rebound / foreign dict, or full=True walk every key and (re)wrap."""
        mg = mh.graphs
        if not full and mg is self._h1_dd and type(mg) is _DirtyDict:
            ks, mg._set = mg._set, {}
            items = [(k, dict.__getitem__(mg, k)) for k in ks if dict.__contains__(mg, k)]
            self.stats["h1_dirty_walks"] = self.stats.get("h1_dirty_walks", 0) + 1
        else:
            if not (mg is self._h1_dd and type(mg) is _DirtyDict):
                if mg is not self._h1_dd and self._h1_dd is not None:
                    self.stats["h1_rewraps"] = self.stats.get("h1_rewraps", 0) + 1
                mg = _DirtyDict(mg)
                mh.graphs = mg              # (MixedHead binds graphs once, in __init__; nd's wrapper replaces it like eng.bk)
                self._h1_dd = mg
                self._h1_unpub_full = True  # (removals recorded before this wrap, or in a replaced dict, are unknown:
                                            # the next unpublish is a full pass)
            else:
                mg._set = {}                # (a full walk covers every pending write)
            items = list(mg.items())
            self.stats["h1_full_walks"] = self.stats.get("h1_full_walks", 0) + 1
        added = 0
        for k, ent in items:
            gs = ent[0] if isinstance(ent, tuple) else None
            if not (isinstance(k, tuple) and (len(k) == 3 or (len(k) == 4 and k[3] == "bm")) and isinstance(gs, list)
                    and all(isinstance(g, torch.cuda.CUDAGraph) for g in gs)):
                continue                      # (H2's recorded programs are not graphs; (v3.26) "bm" = the bitmask variant)
            tk = tuple(k)
            for r, g in enumerate(gs):
                ident = (("H1",) + tk, r, 0)
                if self._pub.get(ident) is g:
                    continue
                aux = hsd.draft_rank(r, int(k[1])).b["aux"]
                self.x.add_graph(4, r, -1, int(k[2]), 0, int(g.raw_cuda_graph_exec()), _key_text(k), int(aux.data_ptr()))
                if r == 0:
                    self._late_pub("H1", k)
                self._pub[ident] = g
                self._h1_idents.setdefault(tk, {})[ident] = None
                self._keep.append(g)
                added += 1
        return added

    def _h1_guard(self, mh, publish: bool = True) -> None:
        """(v3.4) the native H1 path's pointer guard at a step boundary (h1_guard_scan) + unpublishing what it retired."""
        if mh is None:
            return
        MXH = sys.modules.get(type(mh).__module__)
        st = self.stats
        wskip = False
        if (H1_SCAN_SKIP or H1_SCAN_VERIFY) and MXH is not None and hasattr(MXH, "scan_needed"):
            wskip = not MXH.scan_needed(int(self.x.epoch()))   # no side effects; honest under VERIFY
            if wskip and not H1_SCAN_VERIFY:
                st["h1_guard_skipped"] = st.get("h1_guard_skipped", 0) + 1   # no scan, no mark_verified, cache untouched
                return
        ep0 = MXH.epoch() if MXH is not None and hasattr(MXH, "epoch") else None   # BEFORE the scan
        g = h1_guard_scan(mh)
        if wskip:
            st["h1_scan_skip_verify"] = st.get("h1_scan_skip_verify", 0) + 1
            if g["retired"]:
                st["h1_scan_skip_miss"] = st.get("h1_scan_skip_miss", 0) + 1
                self.eng.log(f"nd_export: H1 SCAN SKIP MISS: a skippable scan retired {len(g['retired'])} keys "
                             f"(epoch {MXH.epoch()}, buckets epoch {int(self.x.epoch())})")
        st["h1_guard_scans"] += 1
        st["h1_guard_retired"] += len(g["retired"])
        st["h1_guard_slots_max"] = max(st["h1_guard_slots_max"], g["slots"])
        st["h1_guard_us_sum"] += g["us"]
        st["h1_guard_us_max"] = max(st["h1_guard_us_max"], g["us"])
        if st["h1_guard_scans"] in (1, 64) or st["h1_guard_scans"] % 4096 == 0:
            self.eng.log(f"nd_export: h1 guard scans {st['h1_guard_scans']} slots_max {st['h1_guard_slots_max']} "
                         f"us/scan {st['h1_guard_us_sum'] / st['h1_guard_scans']:.1f} max {st['h1_guard_us_max']:.0f} "
                         f"retired {st['h1_guard_retired']} | C++ scans {H1X_STATS['cxx_scans']} flagged "
                         f"{H1X_STATS['cxx_flagged']} miss {H1X_STATS['cxx_miss']} py scans {H1X_STATS['py_scans']} | "
                         f"skipped {st.get('h1_guard_skipped', 0)} skip_verify {st.get('h1_scan_skip_verify', 0)} "
                         f"skip_miss {st.get('h1_scan_skip_miss', 0)}")
        if publish and g["retired"]:
            self._h1_unpublish(mh)
        if ep0 is not None and not g["retired"] and hasattr(MXH, "mark_verified"):
            # verified at ep0 against the handover's own buckets epoch (self.x.epoch() =
            # the counter fpr_eh_buckets_epoch(h_) reads); mark_verified refuses if mixed_cxx_head's epoch moved since ep0
            ok = MXH.mark_verified(ep0, int(self.x.epoch()))
            st["h1_marked"] = st.get("h1_marked", 0) + (1 if ok else 0)
            st["h1_mark_refused"] = st.get("h1_mark_refused", 0) + (0 if ok else 1)

    # ------------------------------------------------------------------ v0.5 (1): mixed buckets, rule (b)
    def _on_bk(self, k) -> None:
        """eng.bk changed key k (create / evict / reset): a decode bucket -> its slot's generation; a mixed one -> its
        view's generation + the epoch. ENG_MIX_ONE_BUCKET: a change of the max A bucket (the only prod A key in eng.bk)
        re-carves the base of every A view: all of them bump."""
        if isinstance(k, int):
            self.x.bump(int(k))
            self.stats["decode_bumps"] += 1
            return
        if not isinstance(k, tuple):
            return
        self._mix_changed(k)
        if self._onebk and len(k) == 3 and k[1] == "prod" and k[2] == "A":
            for t in self.x.mixed_keys():
                if t.endswith("|prod|A") and t != _key_text(k):
                    self._mix_changed_text(t)

    def _on_view(self, k) -> None:
        """eng.bk_view changed key k: a view built or rebuilt over a new base"""
        if isinstance(k, tuple):
            self._mix_changed(k)

    def _mix_changed(self, k) -> None:
        self._mix_changed_text(_key_text(k))

    def _mix_changed_text(self, t) -> None:
        self._mix_ser[t] = self._mix_ser.get(t, 0) + 1
        self.x.bump_mixed(t)         # (+ buckets_epoch; a key nd never saw only moves the epoch)
        self.stats["mixed_bumps"] += 1

    def _flush(self) -> None:
        for d in (self.eng.bk, getattr(self.eng, "bk_view", None)):
            if isinstance(d, _NotifyDict):
                d.flush()

    def _mixed_state(self, k):
        """the CURRENT per-rank state list of mixed key k (eng.bk or a onebk view), None if the engine has none"""
        eng = self.eng
        st = dict.get(eng.bk, k)
        if st is None and getattr(eng, "bk_view", None) is not None:
            hit = dict.get(eng.bk_view, k)
            st = hit[1] if hit is not None else None
        return st

    def ensure_mixed(self, keys) -> int:
        """== the C callback (rule (b) point (i)): loop thread, step boundary, nothing of nd in flight, via run_python.
        keys = [(R, role 0 A / 1 T, key text)] = the step's WHOLE set: all of it is marked busy first, so no bucket of the
        set evicts another, then each bucket is ensured in order and its views (re)filled. 0 = OK
        1 = the engine refused (out of memory: nd runs the step embedded)."""
        self._h1_fresh = None          # (v3.28jR) Python may follow: the next refresh_graphs scans
        eng = self.eng
        want = []
        for R, role, text in keys:
            k = (int(R), "prod", "A" if int(role) == 0 else "T")
            if _key_text(k) != str(text):
                raise ValueError(f"ensure_mixed: key text {text!r} is not {_key_text(k)!r}")
            want.append(k)
        self.stats["ensure_mixed"] += 1
        self._set_span_kinds()
        self._set_libs_snap()   # a lazily loaded owner ext (tail_gather: Engine._tg() at
        #                         the first Python T stage) is in libs[] before this step's phase-1a lookup; cached, cheap
        eng._busy = set(want)
        try:
            sts = [eng.bucket(R, moe, ro) for R, moe, ro in want]
        except torch.cuda.OutOfMemoryError:
            self._flush()
            self.stats["ensure_mixed_refused"] += 1
            return 1
        self._flush()
        for k, st in zip(want, sts):
            t = _key_text(k)
            if self._mix_filled.get(t) != self._mix_ser.get(t, 0) or self.x.mixed_view(k[0], 0 if k[2] == "A" else 1, t, 0)["rc"] != 0:
                self._fill_mixed(k, st)
            elif self._mix_sig.get(t) != self._fill_sig(st):
                self.stats["mixed_refills_sig"] += 1
                self._mix_changed_text(t)
                self._fill_mixed(k, st)
        self._set_p1()                      # (v3.18) after the fills: the ranks' ring / workspace / templates of this step
        if H1_SCAN_ON_ENSURE:               # (v3.31) the H1 pointer guard over every key here too (a Python boundary that moves
            self._h1_guard(getattr(eng.hooks, "__dict__", {}).get("_mixed_head"), publish=True)   # the buckets epoch): a key
            self.stats["h1_scan_on_ensure"] = self.stats.get("h1_scan_on_ensure", 0) + 1         # published after it is clean
        return 0

    def verify_all(self) -> int:
        """the sweep after a Python BODY, which can re-allocate what CURRENT views point at without a dict notify (the shared onebk GT under
        ENG_IDX_ONEBK_SHARE, Stage._grow rings via _set_p1, _cxx_idxa_gst, _cxx_attn_slots, a late _cxx_moe_export):
        - every filled mixed view whose serial is current (a moved serial = a stale generation nd already refuses) and whose
          bucket the engine still holds: _fill_sig re-checked; a changed one -> generation bumped + refilled (ensure_mixed's
          rule, for every view instead of the step's set);
        - _set_span_kinds / _set_libs_snap / _set_p1 (ensure_mixed's non-bucket refreshes);
        - the H1 pointer guard over every captured head key + unpublish (h1_guard_scan, as refresh_graphs): the guarded
          objects are HSD's own (replaced inside hsd.py), so a scan is the only way to see a replacement; after a 0 return
          every published H1 key is clean until the next Python body.
        Loop thread, step boundary, nothing of nd in flight (as ensure_mixed). Returns 0 = every current view and every
        published H1 key verified; 1 = a refill failed (logged): the caller keeps calling Python."""
        self._h1_fresh = None          # (v3.28jR) Python may follow: the next refresh_graphs scans
        eng = self.eng
        self._flush()
        st_ = self.stats
        st_["verify_all"] = st_.get("verify_all", 0) + 1
        try:
            self._set_span_kinds()
            self._set_libs_snap()
            n_ref = 0
            for t in list(self._mix_filled):
                if self._mix_filled.get(t) != self._mix_ser.get(t, 0):
                    continue                       # (stale generation: nd refuses it; the next ensure refills it)
                R, moe, ro = t.split("|")
                k = (int(R), moe, ro)
                st = self._mixed_state(k)
                if st is None:
                    continue                       # (evicted: its notify moved the serial)
                if self._mix_sig.get(t) != self._fill_sig(st):
                    self._mix_changed_text(t)
                    self._fill_mixed(k, st)
                    n_ref += 1
            self._set_p1()
            self._h1_guard(getattr(eng.hooks, "__dict__", {}).get("_mixed_head"), publish=True)
        except Exception as e:  # noqa: BLE001  (the worker keeps its Python lookups)
            st_["verify_all_failed"] = st_.get("verify_all_failed", 0) + 1
            eng.log(f"nd_export: verify_all FAILED ({type(e).__name__}: {str(e)[:200]}): views stay on the Python lookup")
            return 1
        st_["verify_all_refills"] = st_.get("verify_all_refills", 0) + n_ref
        return 0

    @staticmethod
    def _fill_sig(st) -> tuple:
        """per rank, where a view's published fields come from, cheap: views are filled
        here (ensure_mixed, after eng.bucket() returned: Engine._ensure_new's MIX_PERSIST buffers + MIX_ENSURE hooks done),
        never at the _on_bk / _on_view notify (that only bumps the serial). But some sources change later WITHOUT a dict
        notify: Engine._tg_keys REPLACES s["_tg_kb"] when si20 appears after the fill; with ENG_MIX_PERSIST off, vidx_d /
        _vidx_v / vidx_h appear at the first Python step; an owner's ensure may export later. Any change here refills the
        view at the next ensure_mixed. No state references: data pointers, presence, the tkeys bytes' id (the exporter
        holds the published bytes in _keep_s4, so that id cannot be reused while it is compared)."""
        def p(x):
            return int(x.data_ptr()) if x is not None and hasattr(x, "data_ptr") else 0
        out = []
        for s in st:
            kb = s.get("_tg_kb")
            sp = s.get("_cxx_span")
            ps = s.get("_cxx_attn_slots")
            gt = s.get("idx_gt")
            gs = (int(getattr(gt, "hid", -1) or -1), int(getattr(gt, "Bc", -1) or -1), p(getattr(gt, "lat", None)),
                  p(getattr(gt, "bt", None)), p(getattr(gt, "pos", None))) if gt is not None else None
            out.append((gs, bool(s.get("_cxx_idxa_gst_fixed")), p(s.get("_cxx_idxa_gst")),
                        id(ps) if ps is not None else 0, len(ps) if ps is not None else -1, p(s.get("in_d")), s.get("in_h") is not None, p(s.get("vidx_d")), s.get("_vidx_v") is not None,
                        s.get("vidx_h") is not None, kb[0] if kb else -1, id(kb[1]) if kb else 0, len(kb[1]) if kb else 0,
                        int(sp["built"]) if sp else -1, s.get("_cxx_moe_export") is not None, p(s.get("h")), p(s.get("aux"))))
        return tuple(out)

    def _fill_mixed(self, k, st) -> None:
        """the views of mixed bucket k = (R, moe, role) on every rank (pre-creates its upload sets, as _upload_v2 would)"""
        eng = self.eng
        R, _moe, ro = k
        t = _key_text(k)
        EM = sys.modules[type(eng).__module__]
        lay, _ = EM._in_layout(R)
        for r in range(len(eng.devs)):
            s = st[r]
            if s.get("in_d") is not None and s.get("in_h") is None:
                with torch.cuda.device(eng.devs[r]):
                    nb = s["in_d"].numel()
                    s["in_h"] = [[torch.zeros(nb, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
                    s["in_hv"] = [EM._in_views(x[0], R, numpy=True) for x in s["in_h"]]
                    s["in_i"] = 0
            ins = self._sets(s["in_h"][0][0], s["in_h"][1][0], ("min", t, r), r) if s.get("in_h") is not None else None
            v = {"moe_kind": 0, "fields_hash": int(self.fields_hash), "ptrs": [], "bytes": [],
                 "in_d": B(s.get("in_d")), "in": ins,
                 "in_off_tok": lay["tok"][0], "in_off_valid": lay["valid"][0], "in_off_samp": lay["samp"][0],
                 "h": T(s.get("h")) if ro == "T" else None, "aux": T(s.get("aux")) if ro == "T" else None}
            v.update(self._vidx_view(s))
            sp = s.get("_cxx_span")   # v0.7 slice 3 (C2): FPR's mixed_cxx_ensure span table (ENG_MIX_ENSURE), None = not built
            v["span"] = (dict(built=int(sp["built"]), embed=int(sp["embed"]), final=int(sp["final"]), fl_rsf=int(sp["fl_rsf"]),
                              rows=[[int(x) for x in row[:15]] for row in sp["t"]]) if sp else None)   # (slice 4: 15 cols)
            v.update(self._slice4_view(t, r, s))
            v.update(self._p1_view(t, r, s, ro))   # (v3.18)
            self.x.set_mixed(t, int(R), 0 if ro == "A" else 1, r, v)
        self._mix_filled[t] = self._mix_ser.get(t, 0)
        self._mix_sig[t] = self._fill_sig(st)
        self.stats["mixed_fills"] += 1

    # ------------------------------------------------------------------ the set handoff (C)
    def _holders(self):
        """(python holder setter, persistent event, rank) for every double-buffered set the native path shares."""
        eng, d13, H = self.eng, self.d13, self.eng.hooks
        out = []
        for R, st in self._state_of.items():
            for r, s in enumerate(st):
                for i in range(2):
                    if s.get("in_h") is not None:
                        out.append((lambda e, s=s, i=i: s["in_h"][i].__setitem__(1, e), self.ev(("in", R, r, i), r), r))
                    if s["idx"].get("up_h") is not None:
                        out.append((lambda e, d=s["idx"], i=i: d["up_h"][i].__setitem__(1, e), self.ev(("up", R, r, i), r), r))
        for t in list(self._mix_filled):   # v0.5: the mixed views' upload sets (the engine's CURRENT states only)
            R, moe, ro = t.split("|")
            st = self._mixed_state((int(R), moe, ro))
            if st is None:
                continue
            for r, s in enumerate(st):
                if s.get("in_h") is None:
                    continue
                for i in range(2):
                    out.append((lambda e, s=s, i=i: s["in_h"][i].__setitem__(1, e), self.ev(("min", t, r, i), r), r))
        for r in range(len(eng.devs)):
            rx = d13.rx[r]
            for i in range(2):
                out.append((lambda e, rx=rx, i=i: rx._ev.__setitem__(i, e), self.ev(("rx", r, i), r), r))
                out.append((lambda e, r=r, i=i: d13.dl_ev[i].__setitem__(r, e), self.ev(("dl", r, i), r), r))
                out.append((lambda e, r=r, i=i: d13.nb_ev[i].__setitem__(r, e), self.ev(("nb", r, i), r), r))
                out.append((lambda e, r=r, i=i: d13.qidx_ev[i].__setitem__(r, e), self.ev(("qidx", r, i), r), r))
        return out

    def push_state(self, st: dict) -> None:
        """nd -> engine (after nd::drain(), before an embedded step): the set indices, every holder = its persistent event,
        HSD's _hi, and HSD's per-slot cache invalidated (the native side keys slots differently)."""
        self._h1_fresh = None          # (v3.28jR) Python may follow: the next refresh_graphs scans
        eng, d13, H = self.eng, self.d13, self.eng.hooks
        pc = time.perf_counter
        q0 = pc()
        # native steps rewrote the step inputs Python's skip paths assume unchanged since ITS last prep: the
        # embedded step must not take an OWNER / EMPTY skip or a D13S steady step against them (= D13._fallback's
        # bookkeeping); nor a variable-rows cut against a tok_j base nd overwrote
        d13._epoch += 1
        d13._r_empty = [False] * len(eng.devs)
        d13._own_last = [None] * len(eng.devs)
        if getattr(d13, "steady", None) is not None:
            d13.steady.prev_sig = None
        if getattr(d13, "vrows", None) is not None:
            d13.vrows.invalidate()
        for b, R in enumerate(st["bucket_R"]):
            s_l = self._state_of.get(R)
            if s_l is None:
                continue
            for r, s in enumerate(s_l):
                s["in_i"] = int(st["in_i"][b][r])
                s["idx"]["up_i"] = int(st["up_i"][b][r])
        for r in range(len(eng.devs)):
            d13.rx[r]._hi = int(st["rx_hi"][r])
            if H.engram is not None:
                H.engram[r]._stage_i = int(st["engram_i"][r])
                if "engram_gen" in st and hasattr(H.engram[r], "_gen"):   # (v0.7: ENGRAM_GUARD v2's generation)
                    H.engram[r]._gen = int(st["engram_gen"][r])
        d13.dl_i, d13.qi, d13.d2h_i = int(st["dl_i"]), int(st["qi"]), int(st["d2h_i"])
        q1 = pc()
        for set_, e, _r in self._holders():
            set_(e)
        q2 = pc()
        if H.hsd is not None:
            H.hsd._hi = int(st["hsd_hi"])
            H.hsd._slot_key = [None] * len(H.hsd._slot_key)
        # I: nd's prev-of map + step serial (eng.prev_seq[rid] = (rank, seq index, the producing step's serial))
        eng.prev_seq.clear()
        for rid, rank, index, serial in st["prev"]:
            eng.prev_seq[self._rid(rid)] = (int(rank), int(index), int(serial))
        eng.step_serial = int(st["step_serial"])
        q3 = pc()
        # J: the tokens nd extracted natively since the last handoff -> eng.hist (Engram's lookback of later chunks);
        # toks None = the request was dropped on a native step: its engine-side state goes as
        # hooks.on_ctx's `dropped` would have dropped it (Python never saw that step's context)
        for rid, toks in st["hist"]:
            rid_ = self._rid(rid)
            if toks is None:
                eng.hist.pop(rid_, None)
                eng.prev_seq.pop(rid_, None)
                H.samp_ctx.pop(rid_, None)
                H.structured.discard(rid_)
                continue
            h_ = eng.hist.get(rid_)
            if h_ is not None:
                h_.extend(int(t) for t in toks)
            elif toks:   # (v3.29) a rid Python never admitted: its tokens would vanish silently; counted + logged, never an error
                n_ = self.stats["push_hist_unknown_rid"] = self.stats.get("push_hist_unknown_rid", 0) + 1
                if n_ <= 3 or n_ & (n_ - 1) == 0:
                    eng.log(f"nd_export: push_state: J deltas for {rid_!r} ({len(toks)} tokens) but eng.hist has no entry "
                            f"(Python never saw its admission: an on_ctx flush missing?) [{n_} so far]")
        q4 = pc()
        # v0.5: the embedded step's deltas are measured from here (or from on_ctx's admission: _HistDict)
        self._hist_base = {rid_: len(h_) for rid_, h_ in eng.hist.items()}
        q5 = pc()
        if PULL_PROF:                  # (v3.28jP) push_state's phases too
            a_ = self.__dict__.setdefault("_ps", dict(n=0, sets=0.0, holders=0.0, prev=0.0, hist=0.0, base=0.0, nprev=0,
                                                      nhist=0, ntok=0))
            a_["n"] += 1
            for k_, v_ in (("sets", q1 - q0), ("holders", q2 - q1), ("prev", q3 - q2), ("hist", q4 - q3), ("base", q5 - q4)):
                a_[k_] += v_ * 1e3
            a_["nprev"] += len(st["prev"])
            a_["nhist"] += len(st["hist"])
            a_["ntok"] += sum(len(t) for _, t in st["hist"] if t)
            if a_["n"] % PULL_PROF_EVERY == 0:
                E_ = PULL_PROF_EVERY
                eng.log("nd_export: ND_PULL_PROF push_state mean ms over %d: " % E_ +
                        " ".join(f"{k_} {a_[k_] / E_:.3f}" for k_ in ("sets", "holders", "prev", "hist", "base")) +
                        f" | prev {a_['nprev'] / E_:.0f} hist rids {a_['nhist'] / E_:.0f} toks {a_['ntok'] / E_:.0f} "
                        f"| eng.hist {len(eng.hist)} | pushes {a_['n']}")
                for k_ in a_:
                    if k_ != "n":
                        a_[k_] = 0 if k_ in ("nprev", "nhist", "ntok") else 0.0

    def pull_state(self) -> dict:
        """engine -> nd (after an embedded step): record every persistent set event on its rank's step stream (after the
        step's own uploads), make it the holder again (M), and return the set indices. nd invalidates its HsdPrep slot cache."""
        eng, d13, H = self.eng, self.d13, self.eng.hooks
        pc = time.perf_counter
        t0 = pc()
        # the embedded step may be an ASYNC D13 step: extract it now (its handle keeps the results for the
        # worker's merge), else nd's next graph H overwrites HSD's num_sampled / sampled before Python reads them
        d13.drain()
        t1 = pc()
        self._flush()                  # v0.5: the embedded step's mixed creates / evictions are bumped already; pops resolve
        t2 = pc()
        hs = self._holders()
        t3 = pc()
        # the same records on the same streams, grouped per rank under
        # ONE device context (was one torch.cuda.device enter / exit per holder); per rank in the list's order
        by_r = {}
        for set_, e, r in hs:
            by_r.setdefault(r, []).append((set_, e))
        for r, lst in by_r.items():
            st_ = eng.streams[r]
            with torch.cuda.device(eng.devs[r]):
                for set_, e in lst:
                    e.record(st_)
                    set_(e)
        t4 = pc()
        mh_ = getattr(eng.hooks, "__dict__", {}).get("_mixed_head")
        self._h1_guard(mh_)            # (v3.4) after the embedded step's own changes
        self._h1_fresh = (id(mh_), int(eng.step_serial)) if (H1_REUSE and mh_ is not None) else None   # (v3.28jR)
        t5 = pc()
        self._set_libs_snap()          # (v3.15) the embedded step may have loaded an owner's ext (tail_gather, ...)
        t6 = pc()
        self._set_p1()                 # (v3.18) ... grown the indexer ring (Stage._grow) or recorded combine templates
        t7 = pc()
        if PULL_PROF:                  # (v3.28jP) ND_PULL_PROF=1: per-phase host ms, logged every PULL_PROF_EVERY pulls
            a_ = self.__dict__.setdefault("_pp", dict(n=0, drain=0.0, flush=0.0, holders=0.0, records=0.0, h1=0.0, libs=0.0,
                                                      p1=0.0, nrec=0, tail=0.0, nprev=0))
            a_["n"] += 1
            for k_, v_ in (("drain", t1 - t0), ("flush", t2 - t1), ("holders", t3 - t2), ("records", t4 - t3),
                           ("h1", t5 - t4), ("libs", t6 - t5), ("p1", t7 - t6)):
                a_[k_] += v_ * 1e3
            a_["nrec"] += len(hs)
            if a_["n"] % PULL_PROF_EVERY == 0:
                n_ = a_["n"]
                eng.log("nd_export: ND_PULL_PROF pull_state mean ms over %d: " % PULL_PROF_EVERY +
                        " ".join(f"{k_} {a_[k_] / PULL_PROF_EVERY:.3f}" for k_ in ("drain", "flush", "holders", "records",
                                                                                   "h1", "libs", "p1", "tail")) +
                        f" | holders/call {a_['nrec'] / PULL_PROF_EVERY:.0f} prev/call {a_['nprev'] / PULL_PROF_EVERY:.0f}"
                        f" | pulls {n_}")
                for k_ in a_:
                    if k_ != "n":
                        a_[k_] = 0 if k_ in ("nrec", "nprev") else 0.0
        t8 = pc()
        Rs = list(self._state_of)
        in_i = [[int(s.get("in_i", 0)) for s in self._state_of[R]] for R in Rs]
        up_i = [[int(s["idx"].get("up_i", 0)) for s in self._state_of[R]] for R in Rs]
        W = len(eng.devs)
        prev = [(str(rid), int(p[0]), int(p[1]), int(p[2]) if len(p) > 2 else -1) for rid, p in eng.prev_seq.items()]
        hist = []                      # v0.5 J deltas: what the embedded step appended (rids live at pull)
        base, self._hist_base = self._hist_base or {}, None
        for rid_, b in base.items():
            h_ = eng.hist.get(rid_)
            if h_ is not None and len(h_) > b:
                hist.append((str(rid_), [int(t) for t in h_[b:]]))
        out_ = dict(bucket_R=Rs, in_i=in_i, up_i=up_i, rx_hi=[int(d13.rx[r]._hi) for r in range(W)],
                    engram_i=[int(H.engram[r]._stage_i) if H.engram is not None else 0 for r in range(W)],
                    engram_gen=[int(getattr(H.engram[r], "_gen", 0)) if H.engram is not None else 0 for r in range(W)],
                    dl_i=int(d13.dl_i), qi=int(d13.qi), hsd_hi=int(H.hsd._hi) if H.hsd is not None else 0,
                    d2h_i=int(d13.d2h_i), prev=prev, step_serial=int(eng.step_serial), hist=hist)
        if PULL_PROF:                  # (v3.28jP) the tail: set indices, prev list over ALL eng.prev_seq, J deltas
            a_ = self._pp
            a_["tail"] = a_.get("tail", 0.0) + (pc() - t8) * 1e3
            a_["nprev"] = a_.get("nprev", 0) + len(prev)
        return out_

    def _rid(self, rid_text):
        """the engine's rid object for nd's rid text (prev_seq / hist are keyed by the engine's rid objects)"""
        return self._rids.get(rid_text, rid_text)

    def note_rid(self, rid) -> None:
        """the adapter registers each request's engine rid object at admission (rid text -> object)"""
        self._rids[str(rid)] = rid
