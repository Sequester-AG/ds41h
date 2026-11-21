"""agrs WS: ENG_AGRS_WS=1 serves the production MoE layer's per-call buffers of a mixed
step from ONE persistent workspace per GPU (pm_rs_ws.cu via agrs_pmws.FusedRSWS) instead of the caching allocator.

Those requests are the layer's xp / slot [6T, 5120] bf16; with x, h, meta etc. one call
needs 78,736 B per call row (T = all ranks' valid rows).

install(E) (engine._install_agrs, after agrs_engine.install, when ENG_AGRS_WS != 0) wraps three Engine methods:
  __init__    after the original init (weights loaded; the worker sizes the pools only after Engine(...) returns): one
              uint8 workspace per GPU for T = W x rows call rows, bound in the extension. The RSF object is seeded as
              agrs_engine._frs["0"], so agrs_engine.p_moe_prod's layer_rs calls run on it unchanged.
              Log: "engine: ENG_AGRS_WS: production-MoE workspace ... MiB per GPU ...".
  pmf         pm_fused's .layer (the non-RSF mixed path, R < AGRS_MIN_R; PM_NT 0) goes to FusedRSWS.layer (pm_rs.cu's
              layer = pm_fused.cu's statements); layer8 and every other attribute stay pm_fused's.
  step_mixed  every ENG_AGRS_WS_LOG_S (10) s with new calls: "engine: ENG_AGRS_WS stats: ws_calls ... fallbacks size ...
              capture ... no_ws ... stream_switches ... max_T ... of <T>". A fallback (T above the workspace rows,
              a capturing stream) runs the allocating layer, as today; its first occurrence per GPU is also printed by
              the extension.
Rows per rank: ENG_AGRS_WS_ROWS, else rnd64(PHYS_GEOM max_prefill_tokens_per_rank + (spec_k + 1) x max_seqs_per_rank)
(Geometry defaults 4096 / 3 / 80), at most rnd64(cfg.max_mixed_rows) (the engine's own bound, 4,416):
  cap 2048 -> 2,368 rows -> 9,472 call rows -> ~746 MB per GPU; 3072 -> 3,392 / 13,568 / ~1,068 MB; 4096 -> 4,416 / 17,664 /
  ~1,391 MB.
Bitwise: same kernels, same launch shapes and operands as the allocating layer (tests/test_agrs_ws.py).
"""
from __future__ import annotations

import os
import time

LOG_S = float(os.environ.get("ENG_AGRS_WS_LOG_S", "10") or 0)
# ENG_AGRS_WS=1: agrs_pmws (pm_rs_ws.cu, the workspace bound inside the extension). =2: the same
# workspace through agrs_pmws_cabi.FusedRSC (pm_rs_cabi.cu: the C ABI the driver's moe piece calls), one Python-visible ws
# tensor per device (_state["ws"][device index]) shared by the Python path and the piece; same kernels (SASS-gated).
MODE = os.environ.get("ENG_AGRS_WS", "0").strip()
_state = dict(fr=None, T=0, rows=0, bytes=0, t_log=0.0, last_key=None, pmnt_warned=False, ws={}, mode=MODE)


def _geom_field(name: str, default: int) -> int:
    for kv in os.environ.get("PHYS_GEOM", "").split(","):
        kv = kv.strip()
        if kv.startswith(name + "="):
            return int(float(kv.split("=", 1)[1]))
    return int(default)


def rows_per_rank(cfg) -> tuple[int, str]:
    """(rows per rank, where the number came from) for the workspace (see the module doc)."""
    rnd = lambda n: -(-int(n) // cfg.mixed_round) * cfg.mixed_round  # noqa: E731
    env = os.environ.get("ENG_AGRS_WS_ROWS", "").strip()
    if env:
        return int(env), "ENG_AGRS_WS_ROWS"
    try:
        from geometry import Geometry   # serve/geometry.py (on the engine's path)
        g = Geometry()
        d_cap, d_seats, d_k = g.max_prefill_tokens_per_rank, g.max_seqs_per_rank, g.spec_k
    except Exception:  # noqa: BLE001
        d_cap, d_seats, d_k = 4096, 80, 3
    cap = _geom_field("max_prefill_tokens_per_rank", d_cap)
    seats = _geom_field("max_seqs_per_rank", d_seats)
    k = _geom_field("spec_k", d_k)
    want, bound = rnd(cap + (k + 1) * seats), rnd(cfg.max_mixed_rows)
    return min(want, bound), (f"prefill cap {cap} + {(k + 1) * seats} decode rows" +
                              (f", capped at max_mixed_rows {bound}" if want > bound else ""))


def stats() -> dict:
    if _state["mode"] == "2":
        fr = _state["fr"]
        return dict(fr.st) if fr is not None else dict(ws_calls=0, fb_size=0, fb_capture=0, fb_no_ws=0, stream_switches=0, max_T=0)
    import agrs_pmws
    return agrs_pmws.stats()


def stats_line(st: dict | None = None) -> str:
    st = stats() if st is None else st
    return (f"ws_calls {st['ws_calls']} fallbacks size {st['fb_size']} capture {st['fb_capture']} no_ws {st['fb_no_ws']} "
            f"stream_switches {st['stream_switches']} max_T {st['max_T']} of {_state['T']}")


class _PmfWS:
    """pm_fused.Fused with .layer on the workspace; everything else (layer8, variant, e, ...) is the original's."""

    def __init__(self, base, fr):
        self._base, self._fr = base, fr

    def layer(self, *a, **k):
        return self._fr.layer(*a, **k)

    def __getattr__(self, n):
        return getattr(self._base, n)


def _reserve(eng, E) -> None:
    import torch

    import agrs_engine as AE
    import agrs_pmws
    L0 = eng.layers[0]
    w13 = eng.lw[0][L0]["w13"]      # raw experts [E, 2 I, H / 2] u8 (every MoE layer has the same shape)
    Ex, I, H = int(w13.shape[0]), int(w13.shape[1]) // 2, int(w13.shape[2]) * 2
    rows, src = rows_per_rank(eng.cfg)
    T = len(eng.devs) * rows
    if _state["mode"] == "2":
        import agrs_pmws_cabi as APC
        assert (H, Ex, I) == (APC.H_, APC.E_, APC.I_), (H, Ex, I)
        fr = _state["fr"] or APC.FusedRSC(eng.prod_moe(), devices=[torch.device(d).index for d in eng.devs])
        nbytes = fr.reserve(eng.devs, T)
        _state["ws"] = dict(fr.ws)
    else:
        fr = _state["fr"] or agrs_pmws.FusedRSWS(eng.prod_moe())
        nbytes = agrs_pmws.reserve(eng.devs, T, H, Ex, I)
    for d in eng.devs:
        torch.cuda.synchronize(d)
    AE._frs["0"] = fr
    _state.update(fr=fr, T=T, rows=rows, bytes=nbytes)
    pmnt = str(getattr(E, "PM_NT", "0"))
    tag = "" if _state["mode"] == "1" else "=" + _state["mode"]   # (mode 1's line unchanged: the soak pieces grep it)
    eng.log(f"engine: ENG_AGRS_WS{tag}: production-MoE workspace {nbytes / 2**20:,.1f} MiB ({nbytes:,} B) per GPU: {rows} rows per "
            f"rank x {len(eng.devs)} = {T} call rows ({src}); H {H}, E {Ex}, I {I}; serves agrs RSF layer_rs and pm_fused "
            f"layer (PM_NT {pmnt}{'' if pmnt == '0' else ': NOT USED, the narrow-tile kernels allocate as before'}); "
            f"stats every {LOG_S:g} s")


def install(E) -> None:
    if getattr(E.Engine, "_agrs_ws_installed", False):
        return
    import agrs_engine  # noqa: F401  (installed first by engine._install_agrs: its _frs is seeded at engine init)
    orig_init, orig_pmf, orig_step_mixed = E.Engine.__init__, E.Engine.pmf, E.Engine.step_mixed

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        _reserve(self, E)

    def pmf(self):
        base = orig_pmf(self)
        fr = _state["fr"]
        if fr is None or str(getattr(E, "PM_NT", "0")) != "0":
            return base
        px = self.__dict__.get("_agrs_ws_pmf")
        if px is None or px._base is not base:
            px = self._agrs_ws_pmf = _PmfWS(base, fr)
        return px

    def step_mixed(self, *a, **k):
        out = orig_step_mixed(self, *a, **k)
        if LOG_S > 0:
            now = time.monotonic()
            if now - _state["t_log"] >= LOG_S:
                _state["t_log"] = now
                st = stats()
                key = (st["ws_calls"], st["fb_size"], st["fb_capture"], st["fb_no_ws"])
                if key != _state["last_key"]:
                    _state["last_key"] = key
                    self.log("engine: ENG_AGRS_WS stats: " + stats_line(st))
        return out

    E.Engine.__init__, E.Engine.pmf, E.Engine.step_mixed = __init__, pmf, step_mixed
    E.Engine._agrs_ws_installed = True
