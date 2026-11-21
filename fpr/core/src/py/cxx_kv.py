"""Stage 2a: the C++ KV manager and tables behind serve/'s Python surface.

  PHYS_CXX_KV=1      serve/kv_manager.py's OwnedKVManager = CxxKVManager (C++ OwnedKVManager, include/kv.h)
  PHYS_CXX_TABLES=1  serve/worker.py's TableBuilder = CxxTableBuilder, worker.build_compact / interfaces.build_step_batch
                     dispatch to C++ when given a CxxTableBuilder / CxxKVManager (else the Python functions run)
Both default off; the hooks are a few guarded lines at the end of those modules. The extension `fpr_kvtables` (pybind11,
src/py/kvtables_py.cpp) is loaded from $PHYS_CXX_BUILD (default fpr/core/build_py); build: make -f tests/kvtables_py.mk.

Exactness against the Python manager:
- keys: str rids as themselves; CP shadows ("cp", rid, rank) -> fpr::shadow_key(kstr(rid), rank); anything else ->
  "#r:" + repr(key);
- block hashes: h128() maps each key to 16 bytes:
  - bytes of 16+ bytes (vLLM's BlockHash) -> its first 16 bytes;
  - anything else -> blake2b-128 of repr(key).
  Equal keys map equal and distinct keys collide with probability ~2^-128, so the C++ manager decides exactly as
  Python's dicts do.
- commit's block_hash(i) is asked for the same indices as Python asks: [num_registered, min(nc // 64, len(pages))).

Views (by reference, as Python shares its list / dict objects):
- kv.seqs[key].pages / .window are live C++ views (Pages / Window): allocate's appends are seen and migrate's new
  objects are not.
- kv.paged[r] etc. expose spec / num_pages / num_free() / num_used() / reserved / ref / cached / hash_of / lru / poison
  for the tests.
- cached / hash_of give the original Python keys only with PHYS_CXX_KV_REVERSE=1 (an h128 -> key dict: tests;
  unbounded, never in serving).
"""
from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
BUILD = os.environ.get("PHYS_CXX_BUILD") or os.path.join(_CORE, "build_py")
if BUILD not in sys.path:
    sys.path.insert(0, BUILD)
import fpr_kvtables as X  # noqa: E402

# the ITables ABI of the loaded fpr_kvtables (1: build / build_compact; 2: + save_state / restore_state); 1 if it predates it
TABLES_ABI = int(X.tables_abi()) if hasattr(X, "tables_abi") else 1

REVERSE = os.environ.get("PHYS_CXX_KV_REVERSE", "0") == "1"
POOLS = ("paged", "window", "state")
_KEYS: dict = {}                      # C++ key string -> the Python key (CP shadow tuples, "#r:" keys)
CopyOp = AdmitResult = None           # serve/kv_manager.py's dataclasses (use_types)


def use_types(copy_op, admit_result) -> None:
    global CopyOp, AdmitResult
    CopyOp, AdmitResult = copy_op, admit_result


def kstr(key) -> str:
    if type(key) is str:
        return key
    if type(key) is tuple and len(key) == 3 and key[0] == "cp" and type(key[2]) is int:
        s = "cp\x1f%s\x1f%d" % (kstr(key[1]), key[2])
    else:
        s = "#r:" + repr(key)
    _KEYS[s] = key
    return s


def kobj(s: str):
    k = _KEYS.get(s)
    if k is not None:
        return k
    if s.startswith("cp\x1f"):                   # a shadow key made in C++: Python's tuple
        _, rid, rank = s.split("\x1f")
        return ("cp", rid, int(rank))
    return s


def h128(key) -> bytes:
    if type(key) is bytes and len(key) >= 16:
        return key[:16]
    return hashlib.blake2b(repr(key).encode(), digest_size=16).digest()


def _real(kv):
    for _ in range(8):
        r = getattr(kv, "_gt_real", None)
        if r is None:
            return kv
        kv = r
    return kv


# ---------------------------------------------------------------------------------------------- SeqKV view
# View identity as Python's: ONE Python view object per live C++ page list / window table (kv.seqs[k].pages is the same
# object at every access, and the same object as a batch's se.block_table), a new one after migrate.
# A cached view keeps its C++
# object alive, so the address cannot be reused while cached; entries only the cache still references are dropped.
_PV: dict = {}
_WV: dict = {}
_NACC = [0]


def _prune() -> None:
    for d in (_PV, _WV):
        for k in [k for k, v in d.items() if v.use_count() <= 1]:
            del d[k]


def _view(cache: dict, ptr: int, make):
    v = cache.get(ptr)
    if v is None:
        v = cache[ptr] = make()
    _NACC[0] += 1
    if _NACC[0] % 4096 == 0:
        _prune()
    return v


def pages_of(sk):
    return _view(_PV, sk.pages_ptr(), sk.pages_view)


def window_of(sk):
    return _view(_WV, sk.window_ptr(), sk.window_view)


# the extension's SeqKV (shared_ptr) gets the Python SeqKV's attribute names
X.SeqKV.pages = property(pages_of)
X.SeqKV.window = property(window_of)
X.SeqKV.seq_id = property(lambda s: kobj(s.seq_key))
X.SeqKV.capacity_tokens = property(lambda s: len(pages_of(s)))


class _Seqs:
    """kv.seqs: get / [] / in / len / bool / iteration (keys) / values / items."""

    def __init__(self, kv):
        self._kv = kv

    def get(self, key, default=None):
        sk = self._kv._k.seq(kstr(key))
        return default if sk is None else sk

    def __getitem__(self, key):
        sk = self._kv._k.seq(kstr(key))
        if sk is None:
            raise KeyError(key)
        return sk

    def __contains__(self, key):
        return self._kv._k.seq(kstr(key)) is not None

    def __len__(self):
        return self._kv._k.num_seqs()

    def __bool__(self):
        return self._kv._k.num_seqs() > 0

    def keys(self):
        return [kobj(s) for s in self._kv._k.seq_keys()]

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [self._kv._k.seq(s) for s in self._kv._k.seq_keys()]

    def items(self):
        return [(kobj(s), self._kv._k.seq(s)) for s in self._kv._k.seq_keys()]


class _Ref:
    def __init__(self, kv, pool, rank):
        self._k, self._p, self._r = kv._k, pool, rank

    def __getitem__(self, p):
        return self._k.ref(self._p, self._r, p)

    def __len__(self):
        return self._k.num_pages(self._p, self._r)

    def __iter__(self):
        return iter(self._k.refs(self._p, self._r))

    def __eq__(self, other):
        return list(self) == list(other)


class _Cached:
    """pool.cached: hash key -> page (keys are the original Python keys with PHYS_CXX_KV_REVERSE=1, else 16-byte h128s)."""

    def __init__(self, kv, pool, rank):
        self._kv, self._p, self._r = kv, pool, rank

    def get(self, h, default=None):
        p = self._kv._k.cached_get(self._p, self._r, h128(h))
        return default if p < 0 else p

    def __getitem__(self, h):
        p = self._kv._k.cached_get(self._p, self._r, h128(h))
        if p < 0:
            raise KeyError(h)
        return p

    def __contains__(self, h):
        return self._kv._k.cached_get(self._p, self._r, h128(h)) >= 0

    def items(self):
        rev = self._kv._rev
        return [((rev.get(b, b) if rev is not None else b), p) for b, p in self._kv._k.cached_items(self._p, self._r)]

    def keys(self):
        return [k for k, _ in self.items()]

    def values(self):
        return [p for _, p in self.items()]

    def __len__(self):
        return len(self._kv._k.cached_items(self._p, self._r))

    def __iter__(self):
        return iter(self.keys())


class _HashOf:
    def __init__(self, kv, pool, rank):
        self._kv, self._p, self._r = kv, pool, rank

    def __getitem__(self, p):
        b = self._kv._k.hash_of(self._p, self._r, p)
        if b is None or self._kv._rev is None:
            return b
        return self._kv._rev.get(b, b)

    def __len__(self):
        return self._kv._k.num_pages(self._p, self._r)


class _Poison:
    def __init__(self, kv, pool, rank):
        self._k, self._p, self._r = kv._k, pool, rank

    def append(self, page):
        self._k.poison_append(self._p, self._r, int(page))

    def __iter__(self):
        return iter(self._k.poison_list(self._p, self._r))

    def __len__(self):
        return len(self._k.poison_list(self._p, self._r))

    def __bool__(self):
        return len(self) > 0


class _FreeLen:
    def __init__(self, kv, pool, rank):
        self._k, self._p, self._r = kv._k, pool, rank

    def __len__(self):
        return self._k.nfree_uncached(self._p, self._r)


class _Pool:
    """kv.paged[r] / kv.window[r] / kv.state[r] (PagePool's attributes that callers and tests read)."""

    def __init__(self, kv, pool, rank, spec):
        self._kv, self._p, self.rank, self.spec = kv, pool, rank, spec
        self.ref = _Ref(kv, pool, rank)
        self.cached = _Cached(kv, pool, rank)
        self.hash_of = _HashOf(kv, pool, rank)
        self.poison = _Poison(kv, pool, rank)
        self.free_uncached = _FreeLen(kv, pool, rank)
        self.low_first = spec.kind == "state"

    @property
    def num_pages(self):
        return self._kv._k.num_pages(self._p, self.rank)

    def num_free(self):
        return self._kv._k.num_free(self._p, self.rank)

    def num_used(self):
        return self._kv._k.num_used(self._p, self.rank)

    @property
    def reserved(self):
        return self._kv._k.reserved(self._p, self.rank)

    @property
    def lru(self):
        return {p: None for p in self._kv._k.lru(self._p, self.rank)}

    @property
    def n_hbm(self):
        return self._kv._k.n_hbm(self._p, self.rank)

    @property
    def poison_on(self):
        return self._kv._k.poison_on(self._p, self.rank)


# ---------------------------------------------------------------------------------------------- the manager
class CxxKVManager:
    """serve/kv_manager.py OwnedKVManager's surface over the C++ manager (same arguments, same returns)."""

    __cxx__ = True

    def __init__(self, geom):
        from kv_manager import hbm_pages
        self.geom = geom
        specs = {p.name: p for p in geom.pools()}
        mode = os.environ.get("PHYS_POISON_ON_FREE", "")
        pm = mode if mode in ("1", "pages") else ""
        self._k = X.KV(geom.n_ranks, geom.page_tokens, geom.window, geom.window_page_tokens, geom.replay_tokens,
                       float(geom.reserve_watermark),
                       [(s.name, s.kind, s.tokens_per_page, s.bytes_per_page, s.num_pages) for s in
                        (specs["paged"], specs["window"], specs["state"])], int(hbm_pages()), pm)
        self._rev = {} if REVERSE else None
        self.seqs = _Seqs(self)
        self.paged = [_Pool(self, 0, r, specs["paged"]) for r in range(geom.n_ranks)]
        self.window = [_Pool(self, 1, r, specs["window"]) for r in range(geom.n_ranks)]
        self.state = [_Pool(self, 2, r, specs["state"]) for r in range(geom.n_ranks)]

    # -- forms
    def _h(self, h) -> bytes:
        b = h128(h)
        if self._rev is not None:
            self._rev.setdefault(b, h)
        return b

    def _ops(self, ops):
        return [CopyOp(*o) for o in ops]

    def _ikv(self):
        """PyCapsule "fpr.IKVManager" holding the fpr::IKVManager* (NON-owning: keep this object alive while it is used)."""
        return self._k.ikv_capsule()

    def cxx_capsule(self):
        """"""
        return self._k.cxx_capsule()

    # -- state views
    @property
    def stats(self) -> dict:
        return self._k.stats()

    @property
    def pinned(self) -> dict:
        return self._k.pinned()

    @property
    def pending(self) -> dict:
        return {s: {} for s in self._k.pending_steps()}

    @property
    def deferred(self) -> dict:
        return self._k.deferred()

    def take_poison(self, rank: int) -> list:
        return self._k.take_poison(rank)

    # -- queries
    def lookup(self, block_hashes) -> list:
        return self._k.lookup([self._h(h) for h in block_hashes])

    def free_pages(self, rank: int) -> int:
        return self._k.free_pages(rank)

    def usage(self, rank: int) -> dict:
        u = self._k.usage(rank)
        return dict(used=u[0], cached_free=u[1], free=u[2], reserved=u[3], total=u[4], window_used=u[5], state_used=u[6])

    def block_table(self, seq_id):
        return self.seqs[seq_id].pages

    def window_table(self, seq_id):
        return self.seqs[seq_id].window

    def pages_for(self, tokens: int) -> int:
        return self._k.pages_for(tokens)

    def hit_after_cap(self, hit_blocks: int, prompt_len: int):
        return self._k.hit_after_cap(hit_blocks, prompt_len)

    # -- admission / growth
    def admit(self, seq_id, rank: int, block_hashes, prompt_len: int, step: int = 0, copy_from=None):
        r = self._k.admit(kstr(seq_id), rank, [self._h(h) for h in block_hashes], prompt_len, step,
                          -1 if copy_from is None else int(copy_from))
        if r is None:
            return None
        return AdmitResult(r[0], r[1], r[2], r[3], r[4], self._ops(r[5]))

    def can_allocate(self, seq_id, start: int, end: int, window_end=None) -> bool:
        return self._k.can_allocate(kstr(seq_id), start, end, end if window_end is None else window_end)

    def allocate(self, seq_id, start: int, end: int, window_end=None) -> bool:
        return self._k.allocate(kstr(seq_id), start, end, end if window_end is None else window_end)

    def commit(self, seq_id, num_computed: int, block_hash) -> None:
        k = kstr(seq_id)
        sk = self._k.seq(k)
        if sk is None:
            raise KeyError(seq_id)
        nreg = sk.num_registered
        full = min(num_computed // self.geom.page_tokens, len(pages_of(sk)))
        hs = [None] * nreg + [self._h(block_hash(i)) for i in range(nreg, full)]
        self._k.commit(k, num_computed, hs)

    def shrink_reservation(self, seq_id) -> None:
        self._k.shrink_reservation(kstr(seq_id))

    # -- CP shadows
    def open_shadow(self, key, rank: int, replay_start: int = 0) -> bool:
        return self._k.open_shadow(kstr(key), rank, replay_start)

    def trim_window(self, seq_id, lo_pos: int) -> None:
        self._k.trim_window(kstr(seq_id), lo_pos)

    # -- release
    def free(self, seq_id, defer_until=None) -> None:
        if defer_until is None:
            self._k.free_now(kstr(seq_id))
        else:
            self._k.free_deferred(kstr(seq_id), int(defer_until))

    def flush_deferred(self, upto_step: int) -> None:
        self._k.flush_deferred(upto_step)

    # -- moves and copies
    def migrate(self, seq_id, dst: int, step: int, num_computed: int):
        ops = self._k.migrate(kstr(seq_id), dst, step, num_computed)
        return None if ops is None else self._ops(ops)

    def tier_copies(self, step: int, max_pages: int = 4096) -> list:
        return self._ops(self._k.tier_copies(step, max_pages))

    def release_copies(self, step: int) -> None:
        self._k.release_copies(step)

    def _pool(self, name: str, rank: int):
        return {"paged": self.paged, "window": self.window, "state": self.state}[name][rank]

    # -- invariants
    def check(self) -> None:
        self._k.check()


# ---------------------------------------------------------------------------------------------- tables
def _sb_tuple(sb, rids: dict) -> tuple:
    seqs = []
    for se in sb.seqs:
        k = kstr(se.rid)
        rids[k] = se.rid
        bt = se.block_table
        wt = se.window_table
        seqs.append((k, se.kind, se.row0, se.n_rows, se.start, se.num_computed, se.cached_upto, se.replay_start,
                     bool(se.last_chunk), bt if isinstance(bt, X.Pages) else list(bt),
                     wt if isinstance(wt, X.Window) else list(wt.items()),
                     -1 if se.state_slot is None else int(se.state_slot), se.tail_row0, se.n_blocks, bool(se.score),
                     bool(se.forced), int(se.uid), int(se.known), int(se.wver), int(se.inc)))
    return (sb.rank, sb.step, sb.n_rows, list(sb.positions), list(sb.seq_slot), list(sb.paged_slot), list(sb.window_slot),
            list(sb.tail_rows), list(sb.sample_rows), seqs)


class CxxTableBuilder:
    """serve/worker.py TableBuilder's surface over the C++ TableBuilder (its sent / cstate caches live in C++)."""

    __cxx__ = True

    def __init__(self, sources=(2, 8, 14, 20), tokens_of=None, stub_indexer: bool = False, full_block_tables: bool = False):
        self._t = X.TableBuilder()
        self.sources = sources
        self.tokens_of = tokens_of or (lambda rid, pos: (hash((str(rid), pos)) % 120_000) + 1000)
        self.stub_indexer = stub_indexer
        self.full_block_tables = full_block_tables

    def _sync(self):
        self._t.sources = [int(s) for s in self.sources]
        self._t.stub_indexer = bool(self.stub_indexer)
        self._t.full_block_tables = bool(self.full_block_tables)

    def build(self, sb, row_tokens=None):
        import torch
        from worker import RankTables

        self._sync()
        rids: dict = {}
        d = self._t.build(_sb_tuple(sb, rids), None if row_tokens is None else [int(x) for x in row_tokens],
                          self.tokens_of if row_tokens is None else None, rids)
        return _rt_of(d, sb)

    def save_state(self):
        """an opaque copy of this builder's delta state (D12 sent + build_compact's cstate), for restore_state: the native
        step saves before its build and restores when the step falls back to Python"""
        return self._t.save_state()

    def restore_state(self, state) -> None:
        self._t.restore_state(state)

    def itables_ptr(self) -> int:
        """the C++ builder as fpr::ITables* (an int address) for nd / p0::build_mixed: the SAME instance (one D12 sent_
        state) as this object's build(); borrowed: keep this object alive while native code uses the pointer. The settings
        (sources / stub_indexer / full_block_tables) are pushed to the C++ object here (once): a later Python-side change
        reaches C++ only at the next Python build (settings are fixed after engine init today; an exporter refresh that may
        follow a change calls this again). Native callers pass row_tokens (tokens_of is a per-call Python callback) and check
        TABLES_ABI >= 2 before save_state / restore_state through the pointer."""
        self._sync()
        return int(self._t.itables_ptr())

    def build_compact(self, sb, k1: int, row_tokens=None):
        from rowexp_ref import CompactTables

        self._sync()
        rids: dict = {}
        d = self._t.build_compact(_sb_tuple(sb, rids), int(k1), None if row_tokens is None else [int(x) for x in row_tokens],
                                  self.tokens_of if row_tokens is None else None, rids)
        ct = CompactTables(sb.rank, sb.step, k1)
        for f, v in d.items():
            setattr(ct, f, v)
        ct.rids = [se.rid for se in sb.seqs]
        return ct


def _rt_of(d: dict, sb):
    """the C++ build's dict -> worker.RankTables (CxxTableBuilder.build's conversion; also the native golden's)"""
    import torch
    from worker import RankTables

    T = torch.from_numpy
    seq = {k: T(v) for k, v in d["seq"].items()}
    ti = {int(s): T(v) for s, v in d["ti"].items()}
    tl = {int(s): T(v) for s, v in d["tl"].items()}
    return RankTables(d["n"], T(d["positions"]), T(d["tokens"]), T(d["own_slot"]), T(d["si"]), T(d["sl"]), ti, tl,
                      list(sb.sample_rows), [(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs],
                      seq_slot=T(d["seq_slot"]), paged_slot=T(d["paged_slot"]), tail_rows=T(d["tail_rows"]),
                      sample_idx=T(d["sample_idx"]), seq=seq, targets=T(d["targets"]))


def itables_ptr(tb) -> int:
    """native prefill #4: the fpr::ITables* address of an engine-private builder for nd's handover (tb_tail / D13.tb), or
    0 = not shareable (the Python builder; ENG_CXX_TB=check, whose two builders keep separate states) -> nd refuses"""
    if not getattr(tb, "__cxx__", False):
        return 0
    return tb.itables_ptr()


def wrap_build_compact(py_build_compact):
    """worker.build_compact: C++ for a CxxTableBuilder, the Python function otherwise."""
    def build_compact(tb, sb, k1, row_tokens=None):
        if getattr(tb, "__cxx__", False):
            return tb.build_compact(sb, k1, row_tokens)
        return py_build_compact(tb, sb, k1, row_tokens)
    build_compact.__wrapped__ = py_build_compact
    return build_compact


def wrap_build_step_batch(py_bsb, StepBatch, SeqEntry):
    """interfaces.build_step_batch: C++ when the KV manager is a CxxKVManager, the Python function otherwise."""
    def build_step_batch(plan, kv, geom, rank):
        real = _real(kv)
        if not getattr(real, "__cxx__", False):
            return py_bsb(plan, kv, geom, rank)
        items, works = [], []
        for w in plan.work:
            if w.rank != rank:
                continue
            cp = getattr(w, "cp", None)
            items.append((kstr(w.rid), w.kind, w.start, w.n, w.cached_upto, w.replay_start, bool(w.last_chunk),
                          bool(getattr(w, "score", False)), bool(getattr(w, "forced", False)), int(getattr(w, "uid", -1)),
                          int(getattr(w, "known", -1)), cp is not None,
                          [tuple(int(x) for x in s) for s in cp.subs] if cp is not None else [],
                          kstr(cp.rid) if cp is not None else ""))
            works.append(w)
        n_rows, pos, seq_slot, paged, win, tail, samp, seqs = X.build_step_batch(real._k, plan.step, items, rank)
        sb = StepBatch(rank, plan.step, n_rows)
        sb.positions, sb.seq_slot, sb.paged_slot, sb.window_slot, sb.tail_rows, sb.sample_rows = \
            pos, seq_slot, paged, win, tail, samp
        for i, row0, n_blocks, tail_row0, wver, inc, slot, pages, window, bp, wp in seqs:
            pages = _view(_PV, bp, lambda pages=pages: pages)      # the same object as kv.seqs[k].pages
            window = _view(_WV, wp, lambda window=window: window)
            w = works[i]
            helper = w.kind == "cp"
            cp = getattr(w, "cp", None)
            se = SeqEntry(cp.rid if helper else w.rid, "prefill" if helper else w.kind, row0, w.n, w.start, w.start,
                          0 if helper else w.cached_upto, w.replay_start, False if helper else w.last_chunk, pages, window,
                          None if slot < 0 else slot)
            se.cp = cp
            se.n_blocks = n_blocks
            se.known = int(getattr(w, "known", -1))
            se.images = tuple(getattr(w, "images", ()))
            se.score = bool(getattr(w, "score", False))
            se.forced = bool(getattr(w, "forced", False))
            se.uid = int(getattr(w, "uid", -1))
            se.wver, se.inc = wver, inc
            se.tail_row0 = tail_row0
            sb.seqs.append(se)
        sb.copies = [c for c in plan.copies if rank in (c.src_rank, c.dst_rank)]
        take = getattr(kv, "take_poison", None)
        if take is not None:
            sb.poison = take(rank)
        return sb
    build_step_batch.__wrapped__ = py_bsb
    return build_step_batch


# each is idempotent and swaps module globals only, so every later `from <module> import <name>` sees the C++ version.
def install_kv(km) -> bool:
    """PHYS_CXX_KV=1: kv_manager.OwnedKVManager = CxxKVManager (the Python class stays as PyOwnedKVManager)."""
    if getattr(km, "__cxx_kv__", False):
        return True
    use_types(km.CopyOp, km.AdmitResult)
    km.PyOwnedKVManager = km.OwnedKVManager
    km.OwnedKVManager = CxxKVManager
    km.__cxx_kv__ = True
    return True


def install_interfaces(ifc) -> bool:
    """PHYS_CXX_TABLES=1: interfaces.build_step_batch dispatches to C++ for a CxxKVManager."""
    if getattr(ifc, "__cxx_tables__", False):
        return True
    ifc.py_build_step_batch = ifc.build_step_batch
    ifc.build_step_batch = wrap_build_step_batch(ifc.build_step_batch, ifc.StepBatch, ifc.SeqEntry)
    ifc.__cxx_tables__ = True
    return True


def _cmp(a, b, path: str, out: list) -> None:
    """exact comparison of RankTables pieces: tensors (dtype, shape, values), dicts (keys + values), sequences, scalars"""
    import torch

    if len(out) >= 8:
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            out.append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
        elif a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            out.append(f"{path}: {a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}")
        elif not torch.equal(a, b):
            out.append(f"{path}: values differ")
    elif isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            out.append(f"{path}: keys {sorted(a) if isinstance(a, dict) else a!r} vs {sorted(b) if isinstance(b, dict) else b!r}")
        else:
            for k in a:
                _cmp(a[k], b[k], f"{path}[{k!r}]", out)
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            out.append(f"{path}: {a!r} vs {b!r}"[:300])
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}"[:300])


RT_FIELDS = ("n", "positions", "tokens", "own_slot", "si", "sl", "ti", "tl", "sample_rows", "seq_rows", "seq_slot",
             "paged_slot", "tail_rows", "sample_idx", "seq", "targets", "tensors")
CHECK_STATS = dict(calls=0, mismatches=0)


def table_builder_class(py_cls, check: bool = False):
    """engine/engine.py tb_class (ENG_CXX_TB): CxxTableBuilder, or (check) a builder running both, compared per call"""
    if not check:
        return CxxTableBuilder

    class CheckTableBuilder:
        """ENG_CXX_TB=check: py_cls and CxxTableBuilder side by side (each its own sent / cstate caches); every build's
        RankTables compared field by field; the Python result is returned"""

        def __init__(self, *a, **k):
            self.py, self.cx = py_cls(*a, **k), CxxTableBuilder(*a, **k)

        def __getattr__(self, name):          # attributes (stub_indexer, sources, ...) read from the Python builder
            return getattr(self.__dict__["py"], name)

        def __setattr__(self, name, v):
            if name in ("py", "cx"):
                object.__setattr__(self, name, v)
            else:                              # settings go to both
                setattr(self.py, name, v)
                setattr(self.cx, name, v)

        def build(self, sb, row_tokens=None):
            p = self.py.build(sb, row_tokens)
            c = self.cx.build(sb, row_tokens)
            diffs: list = []
            for f in RT_FIELDS:
                _cmp(getattr(p, f, None), getattr(c, f, None), f, diffs)
            CHECK_STATS["calls"] += 1
            if diffs:
                CHECK_STATS["mismatches"] += 1
                if CHECK_STATS["mismatches"] <= 20:
                    print(f"cxx_kv ENG_CXX_TB=check MISMATCH (call {CHECK_STATS['calls']}, step {getattr(sb, 'step', '?')}): "
                          + "; ".join(diffs), flush=True)
            if CHECK_STATS["calls"] in (1, 10, 100, 1000) or CHECK_STATS["calls"] % 5000 == 0:
                print(f"cxx_kv ENG_CXX_TB=check: {CHECK_STATS['calls']} builds compared, {CHECK_STATS['mismatches']} "
                      "mismatching", flush=True)
            return p

    return CheckTableBuilder


def install_worker(wk) -> bool:
    """PHYS_CXX_TABLES=1: worker.TableBuilder = CxxTableBuilder; worker.build_compact dispatches on the builder."""
    if getattr(wk, "__cxx_tables__", False):
        return True
    wk.PyTableBuilder = wk.TableBuilder
    wk.TableBuilder = CxxTableBuilder
    wk.build_compact = wrap_build_compact(wk.build_compact)
    wk.__cxx_tables__ = True
    return True
