"""CxxRankScheduler: serve/scheduler.py RankScheduler's Python surface on the C++ scheduler.

serve/scheduler.py constructs this instead of its own RankScheduler when PHYS_CXX_SCHED=1 (default off; see make()).
Every decision is the C++ scheduler's (golden-trace identical to the Python one); this file only converts:
  * requests: the caller's Request object stays THE request object. After add_request its scheduler-state fields
    (phase, num_computed, num_output, rank, skip_until_step, ...) become live views of the C++ request (reads and
    writes go through; the adapter's hold / sync writes land in C++), via a per-class subclass swap. Rejected requests
    stay plain objects (phase "finished", max_new_tokens capped, as in Python);
  * plans: StepPlan / SeqWork / CPRole / CopyOp / RankLoad objects with the callers' own rid objects (CP helper
    entries: ("cp", rid, rank) tuples, as Python);
  * queues and maps: reqs / running / waiting / prefilling are rebuilt from C++ after each call (cached in between);
    assigning s.waiting / s.running replaces the C++ queue (the adapter's PAUSED_NEW hide / restore);

  * cfg / cost.p mutations by callers are pushed to C++ before each call.
Not available (Python-scheduler internals): private methods other than _slack / _decode_load / _prefill_sorted,
take_uid / free_uid, in-place mutation of the returned containers.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
_SERVE = os.path.normpath(os.path.join(_CORE, "..", "serve"))
for _p in (_SERVE, _CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POOLS = ("paged", "window", "state")
_SEP = "\x1f"
# the Request fields that live in C++ (everything the scheduler reads or writes)
LIVE = ("phase", "rank", "num_computed", "num_output", "prefill_end", "cached_tokens", "replay_start",
        "skip_until_step", "inflight", "first_pending", "max_new_tokens", "priority", "score", "forced", "arrival",
        "admit_time", "pf_progress_time", "first_chunk_time", "first_token_time", "finish_time", "last_pf_step",
        "cp_forced", "uid", "home_rank", "n_preempt", "n_migrate", "admit_steps_waited", "hit_tokens_total",
        "copied_tokens_total")
SCALARS = ("step_id", "rr", "imbalance_steps", "pf_credit", "dec_since_mixed", "lam")
_MOD = None
_VIEWS: dict = {}


def module():
    """fpr_sched_py: from $PHYS_CXX_SCHED_DIR (a build dir of `make -f sched.mk pymod`), else sys.path."""
    global _MOD
    if _MOD is None:
        d = os.environ.get("PHYS_CXX_SCHED_DIR", "")
        if d and d not in sys.path:
            sys.path.insert(0, d)
        import fpr_sched_py
        _MOD = fpr_sched_py
    return _MOD


def enc_rid(x) -> str:
    """a CP shadow ("cp", rid, <int rank>) -> kv.h shadow_key(enc_rid(rid), rank)
    (what the C++ scheduler builds); anything else -> "#r:" + repr(key)."""
    if type(x) is str:
        return x
    if type(x) is tuple and len(x) == 3 and x[0] == "cp" and type(x[2]) is int:   # any rid
        return "cp%s%s%s%d" % (_SEP, enc_rid(x[1]), _SEP, x[2])
    return "#r:" + repr(x)


def h128(key) -> bytes:
    """bytes of 16+ bytes (vLLM's BlockHash) -> the first
    16; anything else -> blake2b-128 of repr(key). Equal keys map equal (repr), distinct ones collide ~2^-128."""
    if type(key) is bytes and len(key) >= 16:
        return key[:16]
    return hashlib.blake2b(repr(key).encode(), digest_size=16).digest()


def _ident(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else "#" + repr(x)


def _view_cls(base):
    """A subclass of the caller's Request class whose LIVE fields read / write the C++ request (handle _cx_h)."""
    v = _VIEWS.get(base)
    if v is not None:
        return v
    ns = {"__slots__": ()} if hasattr(base, "__slots__") and "__dict__" not in dir(base) else {}

    def prop(f):
        return property(lambda self: self._cx_h.get(f), lambda self, x: self._cx_h.set(f, x))
    for f in LIVE:
        ns[f] = prop(f)

    def cp_shadows(self):
        rid = self.rid
        return {ri: (("cp", rid, ri), uid) for ri, uid in self._cx_h.get("cp_shadow_ranks")}
    ns["cp_shadows"] = property(cp_shadows)
    ns["cp_prev_subs"] = property(lambda self: [tuple(s) for s in self._cx_h.get("cp_prev_subs")])
    ns["_cx_live"] = True
    v = type(base.__name__, (base,), ns)
    v.__qualname__ = base.__qualname__
    v.__module__ = base.__module__
    _VIEWS[base] = v
    return v


class _ExtRequest:
    """External-driver mode (CxxRankScheduler.external_views): a request the C++ side holds that was NOT added through
    this wrapper: its rid (the caller's rid object when known, CP
    shadows ("cp", rid, rank), else the C++ rid) and its C++ handle. The LIVE fields, prompt_len and the CP fields read
    through the handle; LIVE writes go through. Equal (and hashed) by rid, so `q in rs.running` holds across reads."""
    __slots__ = ("rid", "_cx_h")
    _cx_live = True
    _FIELDS = frozenset(LIVE + ("prompt_len",))

    def __init__(self, rid, h):
        object.__setattr__(self, "rid", rid)
        object.__setattr__(self, "_cx_h", h)

    def __getattr__(self, f):
        h = object.__getattribute__(self, "_cx_h")
        if h is not None:
            if f in _ExtRequest._FIELDS:
                return h.get(f)
            if f == "cp_shadows":
                rid = object.__getattribute__(self, "rid")
                return {ri: (("cp", rid, ri), uid) for ri, uid in h.get("cp_shadow_ranks")}
            if f == "cp_prev_subs":
                return [tuple(s) for s in h.get("cp_prev_subs")]
        raise AttributeError(f"PHYS_CXX_SCHED external view of {object.__getattribute__(self, 'rid')!r}: no field {f}")

    def __setattr__(self, f, v):
        h = object.__getattribute__(self, "_cx_h")
        if f in LIVE and h is not None:
            h.set(f, v)
            return
        raise AttributeError(f"PHYS_CXX_SCHED external view: {f} is read-only")

    def __eq__(self, o):
        return isinstance(o, _ExtRequest) and o.rid == self.rid

    def __hash__(self):
        return hash(("_ExtRequest", repr(self.rid)))

    def __repr__(self):
        h = object.__getattribute__(self, "_cx_h")
        return f"<CxxRequest {self.rid!r} {h.get('phase') if h is not None else '?'}>"


class _PrivateKV:
    """the scheduler's private C++ OwnedKVManager (only check())."""

    def __init__(self, sched):
        self._s = sched

    def check(self):
        why = self._s._cx.kv_check()
        assert why is None, why

    def __getattr__(self, name):
        raise AttributeError(f"PHYS_CXX_SCHED: kv.{name} needs the C++ KV binding (fpr_kv_py); this scheduler runs "
                             "on a private C++ OwnedKVManager")


def _make_kv(geom):
    """None if not built."""
    d = os.path.normpath(os.path.join(_CORE, "src", "py"))
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import cxx_kv
    except ImportError:
        return None
    import kv_manager
    cxx_kv.use_types(kv_manager.CopyOp, kv_manager.AdmitResult)
    return cxx_kv.CxxKVManager(geom)


def make(geom, cost=None, cfg=None):
    """serve/scheduler.py's PHYS_CXX_SCHED=1 path: the C++ scheduler, or None (-> the Python one) for a cost model the
    C++ port does not have (ProdCostModel)."""
    from cost_model import CostModel
    cost = cost if cost is not None else CostModel()
    if type(cost) is not CostModel:
        return None
    return CxxRankScheduler(geom, cost, cfg)


class CxxRankScheduler:
    def __init__(self, geom, cost, cfg=None):
        from scheduler import SchedConfig
        self.g, self.cost, self.cfg = geom, cost, cfg or SchedConfig()
        S = module()
        self._objs: dict = {}               # C++ rid -> the caller's Request object (live)
        self._keys: dict = {}               # C++ rid -> the caller's rid object
        self.rejected: dict = {}
        kvo = _make_kv(geom)
        from kv_manager import hbm_pages
        pools = [(p.name, p.kind, p.tokens_per_page, p.bytes_per_page, p.num_pages) for p in geom.pools()]
        self._cx = S.Scheduler(dataclasses.asdict(geom), pools, dict(vars(cost.p)), dataclasses.asdict(self.cfg), kvo,
                               self._hash_cb, hbm_pages(), os.environ.get("PHYS_POISON_ON_FREE", ""))
        self._kv = kvo if kvo is not None else _PrivateKV(self)
        self._kv_real = self._kv
        self._cfg_seen = dict(vars(self.cfg))
        self._cost_seen = dict(vars(cost.p))
        self._cache = None
        self._ext = False                   # external_views(): the C++ core drives the scheduler (views read through)
        if os.environ.get("PHYS_GTRACE"):
            sys.path.insert(0, _HERE) if _HERE not in sys.path else None
            import gtrace_sched
            gtrace_sched.install(self)

    # ------------------------------------------------------------------ hashes
    def _hash_cb(self, crid, first, n):
        """C++ asks for the FINAL keys of blocks [first, first + n) of a request: its hash_fn (ImageBlockKeys-wrapped
        for image prompts, as Python's add_request leaves it), through h128"""
        q = self._objs.get(crid)
        if q is None:
            return None
        hf = q.hash_fn
        # PHYS_CXX_KV_REVERSE=1 (tests): convert through the KV wrapper so its h128 -> key map knows the scheduler's keys
        conv = self._kv_real._h if getattr(self._kv_real, "_rev", None) is not None else h128
        out = []
        for i in range(first, first + n):
            try:
                out.append(conv(hf(i)))
            except Exception:  # noqa: BLE001  (a block hash not computed yet: the C++ side asks again later)
                break
        return b"".join(out)

    # ------------------------------------------------------------------ bookkeeping
    def _rid(self, crid):
        """C++ rid -> the caller's rid object (CP shadows: ("cp", rid, rank))."""
        r = self._keys.get(crid)
        if r is not None or crid in self._keys:
            return r
        if crid.startswith("cp" + _SEP):
            _, owner, rank = crid.split(_SEP)
            return ("cp", self._keys.get(owner, owner), int(rank))
        return crid

    def _sync(self) -> None:
        """Before a C++ call: push the callers' cfg / cost.p changes."""
        d = vars(self.cfg)
        if d != self._cfg_seen:
            for k, v in d.items():
                if self._cfg_seen.get(k, object()) != v:
                    self._cx.set_cfg(k, v)
            self._cfg_seen = dict(d)
        d = vars(self.cost.p)
        if d != self._cost_seen:
            for k, v in d.items():
                if self._cost_seen.get(k, object()) != v:
                    self._cx.set_cost(k, v)
            self._cost_seen = dict(d)

    def external_views(self, on: bool = True) -> None:
        """no wrapper method runs
        between its calls, so the _dirty() cache would never refresh, and its requests never pass through add_request.
        On: every read of reqs / running / waiting / prefilling / on_rank rebuilds from C++, and requests this wrapper
        did not add (the core's, CP shadows) appear as _ExtRequest read-through views (rid, LIVE fields, prompt_len).
        Reads only (tests / idle checks): no per-step cost."""
        self._ext = bool(on)
        self._cache = None

    def _obj(self, crid, h=None):
        """The caller's request object for a C++ rid; in external mode a read-through view for one it does not know."""
        q = self._objs.get(crid)
        if q is not None or not self._ext:
            return q if q is not None else self._objs[crid]   # (wrapper mode: an unknown rid is a bug: KeyError)
        return _ExtRequest(self._rid(crid), h if h is not None else self._cx.handle(crid))

    def _views(self):
        c = self._cache
        if c is None or self._ext:
            ob = self._obj
            reqs = {}
            for crid, h in self._cx.reqs_t():
                q = self._objs.get(crid)
                if q is None and self._ext:
                    q = ob(crid, h)
                if q is not None:
                    reqs[q.rid] = q
            running = [ob(c_) for c_ in self._cx.running()]
            waiting = collections.deque(ob(c_) for c_ in self._cx.waiting_t())
            prefilling = {ob(c_): None for c_ in self._cx.prefilling_t()}
            c = (reqs, running, waiting, prefilling)
            if not self._ext:
                self._cache = c
        return c

    def _dirty(self) -> None:
        self._cache = None

    # ------------------------------------------------------------------ the RankScheduler surface
    @property
    def reqs(self):
        return self._views()[0]

    @property
    def running(self):
        return self._views()[1]

    @running.setter
    def running(self, v):
        self._cx.set_running_t([enc_rid(q.rid) for q in v])
        self._dirty()

    @property
    def waiting(self):
        return self._views()[2]

    @waiting.setter
    def waiting(self, v):
        self._cx.set_waiting_t([enc_rid(q.rid) for q in v])
        self._dirty()

    @property
    def prefilling(self):
        return self._views()[3]

    @property
    def stats(self):
        return dict(self._cx.stats_t())

    @property
    def free_uids(self):
        return collections.deque(self._cx.free_uids_t())

    @property
    def free_shadow_uids(self):
        return collections.deque(self._cx.free_shadow_uids_t())

    @property
    def kv(self):
        return self._kv

    @kv.setter
    def kv(self, v):
        real = getattr(v, "_gt_real", v)             # gtrace_sched's recording proxy wraps the same manager
        if real is not self._kv_real:
            self._cx.reset_kv_obj(real)              # the adapter's reset_prefix_cache: a new manager
            self._kv_real = real
        self._kv = v

    def max_new_cap(self, prompt_len: int) -> int:
        return self._cx.max_new_cap(prompt_len)

    def max_request_tokens(self) -> int:
        return self._cx.max_request_tokens()

    def seats_per_rank(self) -> int:
        return self._cx.seats_per_rank()

    def on_rank(self, r: int) -> list:
        return [self._obj(c) for c in self._cx.on_rank_t(r)]

    def rank_loads(self) -> list:
        return list(self._cx.rank_loads_t())

    def _slack(self, base_us, backlog_us, frac=None):
        return self._cx.slack_t(base_us, backlog_us, frac)

    def _decode_load(self, q):
        return self._cx.decode_load_t(enc_rid(q.rid))

    def _prefill_sorted(self, qs, step, now=0.0):
        return [self._obj(c) for c in self._cx.prefill_sorted_t([enc_rid(q.rid) for q in qs], step, now)]

    def add_request(self, req) -> bool:
        from scheduler import FINISHED, ImageBlockKeys, check_images
        self._sync()
        ims = tuple(req.images or ())
        if ims and any(len(im) != 3 or type(im[0]) is not int or type(im[1]) is not int for im in ims):
            # a malformed entry C++ cannot even represent: Python's order of checks, here
            req.max_new_tokens = min(req.max_new_tokens, self.max_new_cap(req.prompt_len))
            cap = self.max_request_tokens()
            why = None
            if req.prompt_len + req.max_new_tokens > cap:
                why = (f"prompt {req.prompt_len} + max_tokens {req.max_new_tokens} tokens > {cap} "
                       f"(one rank's KV pool / max_model_len)")
            else:
                why = check_images(ims, req.prompt_len)
            self.rejected[req.rid] = why
            req.phase = FINISHED
            return False
        crid = enc_rid(req.rid)
        ok, why, mx, h = self._cx.add_t(crid, req.prompt_len, req.max_new_tokens, req.arrival, req.priority,
                                        bool(req.score), bool(req.forced), [(o, n, _ident(c)) for o, n, c in ims],
                                        True)    # final keys from hash_fn (ImageBlockKeys applied in Python)
        self._dirty()
        req.max_new_tokens = mx
        if not ok:
            self.rejected[req.rid] = why
            req.phase = FINISHED
            return False
        if ims and not isinstance(req.hash_fn, ImageBlockKeys):    # as Python's add_request (its keys feed C++)
            req.hash_fn = ImageBlockKeys(req.hash_fn, ims, self.g.page_tokens)
        req._cx_h = h
        if not getattr(type(req), "_cx_live", False):
            req.__class__ = _view_cls(type(req))
        self._objs[crid] = req
        self._keys[crid] = req.rid
        return True

    def abort(self, rid) -> None:
        self._sync()
        crid = enc_rid(rid)
        err = self._cx.abort_t(crid)
        self._dirty()
        if err:
            raise KeyError(rid)
        self._objs.pop(crid, None)
        self._keys.pop(crid, None)

    def schedule(self, now: float = 0.0):
        from cost_model import RankLoad
        from interfaces import CPRole
        from kv_manager import CopyOp
        from scheduler import SeqWork, StepPlan
        self._sync()
        step, work, copies, pre, mig, loads, est, idle, err = self._cx.schedule_t(now)
        self._dirty()
        if err:
            raise AssertionError(err)
        p = StepPlan(step)
        rid = self._rid
        objs = self._objs
        out = p.work
        for (wr, rank, kind, start, n, cu, rs, last, score, forced, uid, known, cp, ims) in work:
            role = None
            if cp is not None:
                role = CPRole(cp[0], rid(cp[1]), cp[2], [tuple(s) for s in cp[3]], cp[4])
            images = ()
            if ims:
                owner = objs.get(cp[1] if cp is not None else wr)
                images = owner.images_in(start, start + n) if owner is not None else tuple(tuple(x) for x in ims)
            out.append(SeqWork(rid(wr), rank, kind, start, n, cu, rs, last, score, forced, uid, known, role, images))
        p.copies = [CopyOp(POOLS[c[0]], c[1], c[2], c[3], c[4], c[5]) for c in copies]
        p.preempted = [rid(r) for r in pre]
        p.migrated = [rid(r) for r in mig]
        p.loads = [RankLoad(*ld) for ld in loads]
        p.est_us, p.idle_frac = est, idle
        return p

    def update(self, plan, accepted: dict, now: float = 0.0) -> list:
        self._sync()
        fin, err = self._cx.update_t(plan.step, [(enc_rid(r), int(a)) for r, a in accepted.items()], now)
        self._dirty()
        if err:
            raise (KeyError if "KeyError" in err else AssertionError)(err)
        out = []
        for c in fin:
            q = self._objs.pop(c, None)
            self._keys.pop(c, None)
            if q is not None:
                out.append(q)
        return out

    def __getattr__(self, name):
        if name in SCALARS:
            return self._cx.sched_get(name)
        raise AttributeError(f"PHYS_CXX_SCHED: RankScheduler.{name} is a Python-scheduler internal the C++ scheduler "
                             f"does not expose")

    def __setattr__(self, name, value):
        if name in SCALARS:
            self._cx.sched_set(name, value)
            return
        object.__setattr__(self, name, value)
