"""mixed_cxx_ensure: every FPR driver piece handle of a mixed
bucket VIEW built at ensure time (its creation: Engine._ensure_new -> Engine._ensure_hooks, ENG_MIX_ENSURE=1) instead of
lazily inside its first step, plus the owners' ensure fns, plus the per (view, rank) export the handover publishes.

ensure_view(drv, st)   st = the new view's 4 states (s["bkey"] = (R, "prod", "A" | "T")). Builds into the SAME s["_cxx"]
  cache keys the step's lookups use (mixed_cxx.Driver.bd / _issue_span / mixed_cxx_engram), so neither the Python chain
  nor the span program builds lazily afterwards (_lazy_check counts any that do):
    A: embed (mixed_cxx_engram, key ("embed", start_src)); layers <= 20: engram_bd (Engram layers, Engram on) or
       bd(L, 0); bd(L, 1); router (AGCE off and, where possible, on); ag; rs
    T: layers > 20: bd(L, 0), bd(L, 1), router, ag, rs; final
  Per-step state stays per step: the Engram dedup-rows case (e.kv_idx set) is not built here (that step runs Python, as
  today); AGRS's RSF flag words are created here under the rule agrs_engine.p_moe_prod applies at its first RSF layer
  (AGRS_RSF + PM_FUSED + R >= MIN_R), so the rs handle's key is the one every later step looks up.
  Owners: mixed_cxx_moe_ensure.ensure_moe(drv, st, nv=ENG_MIX_PERSIST's persistent nv) when present; any owner
  module's ensure(drv, st) (mixed_cxx_attn / _engram / ...) when defined.
export: s["_cxx_export"] per rank = dict(role, layer0, n, kinds, h (numpy int64 [n][len(KINDS)], -1 = none, alive with the
  state), embed, final, fl_rsf): FPR's own handles.

the first 13 columns are FprEhSpanLayer's fields in
  order, moe_rsf / attn_b_pf are proposed additions (the variants set_step / ws["pf_ab"] choose per step).
"""
from __future__ import annotations

import numpy as np

import mixed_cxx as MX

KINDS = ("bd_attn", "bd_ffn", "engram_bd", "router", "router_ce", "ag", "rs")
SPAN_COLS = ("layer", "bd0", "engram_bd", "attn_a", "idx", "attn_b", "bd1", "router", "router_ce", "ag", "moe", "rs", "rs_fl",
             "moe_rsf", "attn_b_pf")


# ENG_MIX_ENSURE_SKIP: comma list of ensure-time builds to SKIP (they then
# happen lazily in the step, as with ENSURE=0): handles, rsf (AGRS RSF flag words: AGE._ensure "rsf"), moe
# attn / engram / head / c1mega (that owner module's ensure). Empty (default) = today.
# ENG_MIXED_CXX_BUILD_SYNC: every device synchronized after FPR's
# own ensure-time handle builds and after each owner's ensure, so no build's async work outlives scratch the allocator re-hands
# (ltg1p: IMA 2 / 2 without any sync between builds; every arm with a sync between builds clean). View creation only (~50 / run).
BUILD_SYNC = __import__("os").environ.get("ENG_MIXED_CXX_BUILD_SYNC", "0") == "1"


def _sync_all(E, drv, what):
    if not BUILD_SYNC:
        return
    import torch
    for d_ in E.devs:
        torch.cuda.synchronize(d_)
    drv.n["build_sync_" + what] += 1


SKIP = frozenset(x.strip() for x in __import__("os").environ.get("ENG_MIX_ENSURE_SKIP", "").split(",") if x.strip())


def _role(s) -> str:
    k = s.get("bkey")
    return k[2] if isinstance(k, tuple) and len(k) == 3 else ""


def _embed_build_ok(drv, s) -> bool:
    """the ensure-time form of mixed_cxx_engram._embed_ok: only its
    STATIC conditions (Engram's pieces up when Engram is on; not a c1mega E0 step). The per-step ones stay with the step
    (handle_embed / piece_embed still call _embed_ok): s["mm_emb"] is this step's image rows, and native mixed steps never
    run with images (core2b has_image_rows -> embedded, design 5a); s["inject_input"] is a test hook. An onebk view
    inherits its base state's last mm_emb tuple, so _embed_ok refused the view's embed handle for the view's lifetime
    build_embed reads none of them (tok / res0 / emb / prep)."""
    E = drv.eng
    if E.hooks.engram is not None and not getattr(drv, "_engram_ok", False):
        return False
    return not (E.hooks.engram is not None and E._c1m_e0(s) is not None)


def _bd(drv, r, s, L, which) -> int:
    """mixed_cxx.Driver.bd's / _issue_span's bd lookup, key ("bd", L, which)"""
    c = MX._cache(drv, s)
    h = c.get(("bd", L, which))
    if h is None:
        b = drv.build_bd(r, s, L, which)
        h = c[("bd", L, which)] = drv.x.add_bd(b) if MX.PieceBd.from_buffer_copy(b).path else -1
    return h


def ensure_view(drv, st) -> None:
    E, EM, AGE = drv.eng, drv.EM, drv.AGE
    s0 = st[0]
    role = _role(s0)
    if s0.get("moe") != "prod" or role not in ("A", "T") or s0.get("_cxx_export") is not None:
        return
    P = MX.PIECES
    R = s0["R"]
    layers = [L for L in E.layers if (L <= 20) == (role == "A")]
    agrs_on = bool(getattr(EM.Engine, "_agrs_installed", False)) and AGE is not None
    ce_ok = agrs_on and getattr(EM, "AGRS_AGCE", False) and R >= AGE.MIN_R
    if ("rs" in P and agrs_on and getattr(EM, "AGRS_RSF", False) and getattr(EM, "PM_FUSED", False) and R >= AGE.MIN_R):
        if "rsf" not in SKIP:
            AGE._ensure(E, st, "rsf")                 # (agrs_engine.p_moe_prod's first RSF layer would; one device sync)
    eng_on = E.hooks.engram is not None and "engram" in P and "engram_bd" in drv._handles
    eg = None
    if eng_on or ("embed" in P and "embed" in drv._handles):
        import mixed_cxx_engram as EG
        eg = EG
    for r, s in enumerate(st):
        H = np.full((len(layers), len(KINDS)), -1, dtype=np.int64)
        for i, L in enumerate(layers if "handles" not in SKIP else ()):
            if eng_on and L in EM.ENGRAM_LAYERS:
                e = E.hooks.engram[r]
                # built whatever this step's kv_idx is: with PRECREATE=0 a view can be ensured
                # inside an Engram dedup step, and the old "kv_idx is None" condition then left engram_bd -1 in the span for
                # good ("bd0 L1" refusals). The handle holds only persistent buffers (kv / q / k / keep), identity-mode inject;
                # dedup steps never use it (Python: handle_engram_bd checks kv_idx every call; native: C++ prepare, no dedup).
                if getattr(drv, "_engram_ok", False):
                    H[i, 2] = drv._cached(s, ("engram_bd", L), lambda: eg.build_engram_bd(drv, r, s, L, ignore_kv_idx=True),
                                          drv.x.add_engram_bd)
            elif "bd" in P:
                H[i, 0] = _bd(drv, r, s, L, 0)
            if "bd" in P:
                H[i, 1] = _bd(drv, r, s, L, 1)
            if "router" in P and "ag" in P and drv._mext_ok:
                H[i, 3] = drv._cached(s, ("router", L, False), lambda: drv.build_router(r, s, L, None), drv.x.add_router)
                if ce_ok:
                    H[i, 4] = drv._cached(s, ("router", L, True), lambda: drv.build_router(r, s, L, st), drv.x.add_router)
                H[i, 5] = drv._cached(s, ("ag", L), lambda: drv.build_ag(r, s, L, st), drv.x.add_ag)
            if "rs" in P:
                H[i, 6] = drv._cached(s, ("rs", L, "agrs_fl_rsf" in s), lambda: drv.build_rs(r, s, L, st), drv.x.add_rs)
        h_embed = h_final = -1
        if role == "A" and eg is not None and "embed" in P and "handles" not in SKIP and _embed_build_ok(drv, s):
            e = E.hooks.engram[r] if E.hooks.engram is not None else None
            src = getattr(e, "_start_src", None) if e is not None else None
            h_embed = drv._cached(s, ("embed", None if src is None else src.data_ptr()), lambda: eg.build_embed(drv, r, s),
                                  drv.x.add_embed)
        if role == "T" and "final" in P and "handles" not in SKIP:
            h_final = drv._cached(s, ("final",), lambda: drv.build_final(r, s), drv.x.add_final)
        s["_cxx_export"] = dict(role=role, layer0=layers[0] if layers else -1, n=len(layers), kinds=KINDS, h=H,
                                embed=int(h_embed), final=int(h_final), fl_rsf=bool("agrs_fl_rsf" in s), layers=layers)
    _sync_all(E, drv, "fpr")
    # owners
    try:
        import mixed_cxx_moe_ensure as ME
    except ImportError:
        ME = None
    if ME is not None and "moe" in P and "moe" not in SKIP:
        nv = [s["_vidx_v"][1] for s in st] if all("_vidx_v" in s for s in st) else None
        ME.ensure_moe(drv, st, nv=nv)
        _sync_all(E, drv, "moe")
    for m in _owner_mods():
        f = getattr(m, "ensure", None)
        if f is not None and m.__name__.replace("mixed_cxx_", "") not in SKIP:
            f(drv, st)
            _sync_all(E, drv, m.__name__.replace("mixed_cxx_", ""))
    if SKIP:
        drv.n["ensure_skip_" + "_".join(sorted(SKIP))] += 1
    for s in st:
        s["_cxx_span"] = span_table(s)
    drv.n["ensure_views"] += 1
    drv.n[f"ensure_handles_{role}"] += sum(int((x["_cxx_export"]["h"] >= 0).sum()) for x in st)


def _owner_h(s, key, L, j):
    """an owner's exported handle of layer L, column j (-1 = none / not exported)"""
    ex = s.get(key)
    if not ex:
        return -1
    i = L - int(ex.get("layer0", 0))
    h = ex.get("h")
    if h is None or i < 0 or i >= int(ex.get("n", 0)):
        return -1
    v = h[i][j]
    return int(v) if v is not None else -1


def span_table(s) -> dict:
    """FprEhSpanTable's content for one rank's view (SPAN_COLS; see the module doc)."""
    ex = s["_cxx_export"]
    H, Ls, fl = ex["h"], ex["layers"], ex["fl_rsf"]
    t = np.full((len(Ls), len(SPAN_COLS)), -1, dtype=np.int64)
    for i, L in enumerate(Ls):
        t[i] = (L, H[i, 0], H[i, 2], _owner_h(s, "_cxx_attn_export", L, 0), _owner_h(s, "_cxx_attn_export", L, 1),
                _owner_h(s, "_cxx_attn_export", L, 2), H[i, 1], H[i, 3], H[i, 4], H[i, 5], _owner_h(s, "_cxx_moe_export", L, 0),
                -1 if fl else H[i, 6], H[i, 6] if fl else -1, _owner_h(s, "_cxx_moe_export", L, 1),
                _owner_h(s, "_cxx_attn_export", L, 3))
    return dict(built=1, n_layers=len(Ls), embed=ex["embed"], final=ex["final"], fl_rsf=int(fl), cols=SPAN_COLS, t=t)


def _owner_mods():
    import sys
    return [sys.modules[n] for n in ("mixed_cxx_attn", "mixed_cxx_engram", "mixed_cxx_head", "mixed_cxx_c1mega")
            if n in sys.modules]


def lazy_count(drv, st) -> int:
    """Handles a view's caches hold now that its ensure did not export (= built lazily in a step after ensure): 0 expected.
    other owners' keys are not counted here."""
    col_kind = {"bd_attn": "bd", "bd_ffn": "bd", "engram_bd": "engram_bd", "router": "router", "router_ce": "router",
                "ag": "ag", "rs": "rs"}
    n = 0
    for s in st:
        ex = s.get("_cxx_export")
        c = s.get("_cxx") or {}
        if ex is None:
            continue
        have = {(col_kind[k], int(v)) for j, k in enumerate(ex["kinds"]) for v in ex["h"][:, j] if v >= 0}
        have |= {("embed", ex["embed"]), ("final", ex["final"])}
        mx = s.get("_cxx_moe_export")
        if mx and mx.get("h") is not None:
            have |= {("moe", int(v)) for row in mx["h"] for v in row if v is not None and int(v) >= 0}
        for k, v in c.items():
            if not (isinstance(k, tuple) and k and isinstance(v, int) and v >= 0):
                continue
            if k[0] == "bd" and len(k) == 3 and k[2] == "engram":   # (engram_bd's inner boundary handle)
                continue
            if k[0] in ("bd", "router", "ag", "rs", "final", "embed", "engram_bd") or (k[0] == "moe" and mx):
                if (k[0], v) not in have:
                    n += 1
    return n

def export(st) -> list:
    """the 4 per-rank export dicts"""
    return [s.get("_cxx_export") for s in st]

