"""Engine hook: indexer + compressor of kernels/indexer2f on the engine's pools and StepBatch tables. Decode rows
(next_n = 4 per sequence, sequence-contiguous) in run(); mixed steps in _run_general.

Per index layer L, between the attention half's q / kv norms and FlashMLA (engine.p_attn -> Hooks.indexer_run):
  kv sources (2 / 8 / 14 / 20), compressor (stock ops, owned rows):
      ks = x @ comp_w^T (fp32) -> fused_save_compress_norm (ring = the rank's state pool, layer view [slots, 8, 1024],
      ring slot = state_slot * 8 + p % 8; ratio 1: no ring) -> latent -> rope_quant_insert into the paged pool's comp{L}
      records (slot = page * (64 // ratio) + (p % 64) // ratio from StepBatch.paged_slot; -1 = no write: padded and
      prefix-cached rows) -> wk(latent) -> indexer_k_norm_rope_store into the paged pool's idx{L} entries (same slots)
  every index layer, q path (default: exactly dequantized bf16 weights, the stock quant op):
      w = x @ idx_wp^T, q = qr @ idx_wq_b^T, (q8, wo) = fused_indexer_q_rope_quant(pos, q, cos_sin, w, 128^-.5, 32^-.5)
  scoring + selection: idx2f.OwnedIndexer(ks = this rank's idx pages of the 4 sources, block tables [B, 10240]).layer
      -> request-local top-512 -> compute_global_topk_indices_and_lens (stock op) -> the bucket's ti[L] / tl[L], which
      the FlashMLA calls of layers L .. next index layer read.

CUDA graphs: OwnedIndexer.plan() is host-side; for a graph it is replaced by fixed bounds (widths and max lengths of the
largest row the engine admits, MAX_CTX) and the DeepGEMM schedules + layer-20 block counts are recomputed INSIDE the
graph from the step's device seq_lens (dg_schedule runs inside the graph).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("IDX2F_CANON_STOCK", "1")   # (engine.py sets it too) canonical stock top-k order
IDX = os.path.join(PHYS, "kernels/indexer2f")
if IDX not in sys.path:
    sys.path.insert(0, IDX)

LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
KV_SOURCES = (2, 8, 14, 20)
RATIO = {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}
K_SOURCE = {2: 2, 8: 8, 14: 14, 20: 20, 24: 20, 28: 20, 32: 20, 36: 20}
RING_LAYERS = (2, 8, 14)          # state pool slot = 3 x [8, 1024] fp32, in this order (geometry.STATE_SLOT_BYTES)
IDLE_DEV_BT = os.environ.get("ENG_IDLE_DEVBT", "1") == "1"      # idle ranks use the device block-table path too
POP_IDX_FIRST = os.environ.get("ENG_POP_IDX", "1") == "1"        # free the previous step's indexer state first
USE_PLAN_BOUNDS = os.environ.get("ENG_PLAN_BOUNDS", "1") == "1"   # eager host path: plan_bounds + persistent bmax
TOPK_WS_PER_DEV = os.environ.get("ENG_TOPK_WS_PER_DEV", "1") == "1"   # CORRECTNESS (default on): vLLM's
                                 # stock top-k (idx2f.stock_topk: long rows past 262,144 positions and the non-fast modes,
                                 # graphed long variants and eager) takes its 1 MB radix workspace from vLLM's ONE global
                                 # WorkspaceManager, which X._workspace() created on the FIRST device: every rank's persistent /
                                 # cooperative top-k ran on that one device-0 buffer at once (persistent kernels coordinate their
                                 # CTAs through it). One manager per device instead: workspace allocated here, outside any
                                 # capture, then locked (a growth request asserts instead of freeing a buffer graphs point at).
CAND_PAD_FIX = os.environ.get("ENG_CAND_PAD_FIX", "1") == "1"   # tail bucket: padding IR rows get their group's cand
LEVEL_CTX = (262_144, 524_288, 655_360)   # decode-graph bound per long-row level: lvl 0 = idx2f fast
                                          # mode everywhere; lvl 1 = ratio-2 layers fast, layer 20 + its consumers stock
                                          # select / top-k; lvl 2 = all stock (max_model_len). Same per-layer rule as eager.
MAX_CTX = 262_144                 # idx2f fast top-k / select limit (rows up to 262,144 compressed positions); the engine
                                  # refuses longer sequences in this mode
P_MAX = 655_360 // 64             # block-table width (production's max_model_len / 64)
NU_MAX = 4096                     # upload v2: block-table delta entries per rank and decode step in the blob (more: per tensor)


def _up_fields(R: int, B: int):
    """upload v2: the decode bucket's per-step indexer inputs in one blob (8-byte aligned fields)."""
    return (("sl1", "int32", B * 4), ("sl2", "int32", B * 4), ("cs1", "int64", R), ("cs2", "int64", R),
            ("rs", "int64", R), ("valid", "bool", R), ("pslot", "int64", B), ("du_rows", "int64", NU_MAX),
            ("du_cols", "int64", NU_MAX), ("du_pages", "int32", NU_MAX))


def _up_bytes(R: int, B: int) -> int:
    import numpy as np
    return sum(-(-(n * np.dtype(dt).itemsize) // 8) * 8 for _, dt, n in _up_fields(R, B))


def _up_views(blob, R: int, B: int, numpy: bool = False) -> dict:
    import numpy as np
    out, off = {}, 0
    tdt = dict(int64=torch.int64, int32=torch.int32, bool=torch.bool)
    src = blob.numpy() if numpy else blob
    for nm, dt, n in _up_fields(R, B):
        nb = n * np.dtype(dt).itemsize
        x = src[off: off + nb]
        out[nm] = x.view(dt) if numpy else x.view(tdt[dt])
        off += -(-nb // 8) * 8
    out["_bytes"] = off
    return out
EPS = 1e-20
# Indexer decode-step cuts, each bitwise == the unfused chain (top-512 == production), all default off; module flags
# read per bucket alloc / per step (tests switch them per arm):
#  IDX_FUSE: decode buckets (level 0): the top-k writes FlashMLA's global slots / lengths (oi.fuse + GlobalOut): no
#            compute_global_topk_indices_and_lens / copies / nb_lens kernels; layers off the fast path fall back inside
#            OwnedIndexer.layer (global_topk)
#  IDX_SIDE: decode buckets: the kv-source compressor chain + weights_proj on a side branch (fork / join before the q
#            RoPE / quant; torch.mm(x, wp.t(), out) == F.linear bitwise), next to the wq_b GEMM
#  IDX_F8:   idx2f's logits kernel family "f8" (idx2f_fcand8.cu: dense + candidate logits, same DeepGEMM arithmetic);
#            the engine sets idx2f.LOGITS per step (default: IDX2F_LOGITS=f8 in the env)
IDX_FUSE = os.environ.get("ENG_IDX_FUSE", "0") == "1"
IDX_SIDE = os.environ.get("ENG_IDX_SIDE", "0") == "1"
IDX_F8 = os.environ.get("ENG_IDX_F8", "0") == "1" or os.environ.get("IDX2F_LOGITS", "") == "f8"


def patch_multidevice(X) -> None:
    """idx2f_cand.cu raises its kernels' dynamic-smem limit once per PROCESS (static bool), so devices 1..3 of a one-process
    4-GPU engine launch with the default limit (cudaErrorInvalidValue). Until idx2f_cand.cu makes it per-device: build
    a copy of its CURRENT source with only that guard per device (eng_idx2f_cand, engine/build) and point
    idx2f.ext_cand at it. Kernels and launch arguments are unchanged."""
    import functools
    from pathlib import Path
    src = (Path(X.HERE) / "idx2f_cand.cu").read_text()
    old = "static bool set = false;"
    if src.count(old) == 0:
        return                                     # fixed upstream
    new = "static bool set_dev_[64] = {}; int dev_ = 0; cudaGetDevice(&dev_); bool& set = set_dev_[dev_];"
    out = Path(PHYS) / "engine" / "build" / "eng_idx2f_cand.cu"
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = "// GENERATED by engine/hooks_idx.py patch_multidevice from kernels/indexer2f/idx2f_cand.cu (per-device smem\n" \
          "// attribute guard only). Do not edit.\n" + src.replace(old, new)
    if not out.exists() or out.read_text() != txt:
        out.write_text(txt)
    X.ext_cand = functools.cache(lambda: X._load("eng_idx2f_cand", str(out), X.DG_FLAGS,
                                                 [str(X.deep_gemm_include()), str(X.HERE)]))


def build_idx_weights(L: int):
    """CPU builder (wcache key eng_idxw_L{L}_v1): comp2f.load_weights without the fp32 copies and the RoPE table."""
    import comp2f
    W = comp2f.load_weights(L, "cpu")
    W.pop("f32", None)
    W.pop("cos_sin", None)
    return W


class Idx2f:
    """Owned-sequence indexer + compressor for one engine (all ranks)."""

    def __init__(self, eng, mode: str = "fast", logits: str | None = None):
        import idx2f as X
        sys.path.insert(0, os.path.join(PHYS, "tools"))
        from wcache import cached
        self.X, self.eng, self.mode = X, eng, mode
        lg = getattr(X, "LOGITS", None)
        self._logits0 = None if lg is None else ("cand" if lg == "f8" else lg)   # IDX_F8 off: idx2f's import-time family
        # dense index logits: DeepGEMM works on one device of a process only; idx2f's own exact kernel
        # (OwnedIndexer.dense_impl = "idx2f", bit-identical to DeepGEMM) on every device; engine Triton = fallback
        has_idx2f_dense = "dense_impl" in getattr(X.OwnedIndexer, "__dataclass_fields__", {})
        self.logits_impl = logits if logits else (("idx2f" if has_idx2f_dense else "triton") if len(eng.devs) > 1
                                                  else "deepgemm")
        if len(eng.devs) > 1:
            patch_multidevice(X)
        if self.logits_impl == "triton":
            # DeepGEMM's JIT kernels are bound to the first device's context in a multi-GPU process (idx_logits.py doc)
            from . import idx_logits as IL

            def _logits(q, ck, weights, seq_lens, width, sched=None):
                return IL.paged_mqa_logits(q, ck.kv, weights, seq_lens, ck.block_table, ck.block_size, width)
            X.dg_logits = _logits
            X.dg_schedule = lambda seq_lens, block_size: None
        self.dbg = {}     # tests: ("ks" | "q8" | "wo", L) -> captured tensors [R, ...] replacing our GEMM outputs (replay)
        self._wrap_dense_logits(X)
        self._wrap_stock_topk(X, eng)
        # mixed-step tables: "host" (upload_general below), "device" (kernels/indexer2f tables2f.GeneralTables: the same dict
        # derived on the device from one staging copy, no host sync, persistent OwnedIndexer), "check" (both, compared
        # on every step, the host dict used)
        self.lean = os.environ.get("ENG_IDX_LEAN", "1") == "1"   # eager host path: idx2f lean indexer + EagerWorkspace
        self.tables = os.environ.get("ENG_IDX_TABLES", "device")   # == host tables bitwise
        assert self.tables in ("host", "device", "check"), self.tables
        self.checked = dict(steps=0, mismatch=0, fallback=0)
        self.W = []
        for r, dev in enumerate(eng.devs):
            with torch.cuda.device(dev):
                X._workspace(dev)
                per = {}
                for L in LAYERS:
                    if L not in eng.layers:
                        continue
                    w = cached(f"eng_idxw_L{L}_v1", lambda L=L: build_idx_weights(L), device=dev, verbose=False)
                    w["cos_sin"] = eng.glob[r]["cos_sin"][RATIO[L]]
                    per[L] = w
                self.W.append(per)
        from geometry import PAGE_BYTES, PAGE_OFFSETS, RECORDS_PER_PAGE, STATE_SLOT_BYTES
        self.PB, self.PO, self.RPP, self.SSB = PAGE_BYTES, PAGE_OFFSETS, RECORDS_PER_PAGE, STATE_SLOT_BYTES
        self.views = []
        if getattr(eng, "pools", None) is not None:
            self.attach(eng.pools)

    def _wrap_stock_topk(self, X, eng) -> None:
        """TOPK_WS_PER_DEV: idx2f.stock_topk with vLLM's workspace manager of the logits' own device (see the flag)."""
        if not TOPK_WS_PER_DEV or getattr(X, "_eng_topk_wrapped", False) or not hasattr(X, "stock_topk"):
            return
        from vllm.model_executor.layers import indexer_topk as IT
        from vllm.v1.worker import workspace as WS
        for dev in eng.devs:                 # the process-global manager exists before the canary below (idempotent)
            with torch.cuda.device(torch.device(dev)):
                X._workspace(torch.device(dev))
        mgrs = {}
        for dev in eng.devs:
            d = torch.device(dev)
            with torch.cuda.device(d):
                m = WS.WorkspaceManager(d)
                m.get_simultaneous(((IT.RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8))
                m.lock()
            mgrs[d.index] = m
        orig = X.stock_topk

        def stock_topk(logits, seq_lens, out, max_seq_len, backend="persistent"):
            m = mgrs.get(logits.device.index)
            if m is None:
                return orig(logits, seq_lens, out, max_seq_len, backend)
            prev = WS._manager
            WS._manager = m
            try:
                return orig(logits, seq_lens, out, max_seq_len, backend)
            finally:
                WS._manager = prev
        X.stock_topk = stock_topk
        X._eng_topk_wrapped = True
        self.topk_ws = mgrs
        # canary: the process-global manager (device 0) refuses requests from another current device, so any vLLM path
        # that still reaches it from rank 1..3 fails loudly instead of silently sharing device 0's scratch
        g = getattr(WS, "_manager", None)
        if g is not None and not getattr(g, "_eng_canary", False):
            gs = g.get_simultaneous

            def guarded(*a, **k):
                cur = torch.cuda.current_device()
                if g._device.index is not None and cur != g._device.index:
                    raise AssertionError(f"vLLM workspace manager of {g._device} used from cuda:{cur} ("
                                         f"TOPK_WS_PER_DEV canary: a cross-device workspace)")
                return gs(*a, **k)
            g.get_simultaneous = guarded
            g._eng_canary = True

    def _wrap_dense_logits(self, X) -> None:
        """Eager steps: idx2f's dense logits [rows, width] fp32 go into ONE workspace per device (idx2f slices each call to
        its logits_budget, 1 GB) instead of a fresh tensor per call, whose width grows with every chunk of a long prompt
        (the caching allocator then never reuses the old blocks: tens of GB cached on the rank with the long prompts).
        The logits of a call are consumed (block max / top-k / select) before the next call on the same stream; nothing
        keeps them (OwnedIndexer.keep_logits False). Graph captures keep the original (graph-pool) allocation."""
        if getattr(X, "_eng_ws_wrapped", False) or not hasattr(X, "dense_logits") or \
                (os.environ.get("ENG_IDX_LEAN", "1") == "1" and hasattr(X, "EagerWorkspace")):
            return
        orig = X.dense_logits
        ws = {}

        def dense_logits(q, ck, weights, seq_lens, width, out=None, **kw):
            rows = q.shape[0] * q.shape[1]
            need = rows * int(width)
            if out is not None or kw or not getattr(X, "_eng_ws_on", True) or torch.cuda.is_current_stream_capturing() or \
                    getattr(X.OwnedIndexer, "keep_logits", False) is True:
                return orig(q, ck, weights, seq_lens, width, **({"out": out} if out is not None else {}), **kw)
            dev = q.device.index
            buf = ws.get(dev)
            if buf is None or buf.numel() < need:
                ws.pop(dev, None)
                budget = int(getattr(X.OwnedIndexer, "logits_budget", 1 << 30))
                buf = ws[dev] = torch.empty(max(need, budget // 4 + (1 << 20)), dtype=torch.float32, device=q.device)
            out = buf[:need].view(rows, int(width))
            X.ext_cand().logits_dense(q, ck.kv, weights, seq_lens.reshape(-1), ck.block_table, out)
            return out
        X.dense_logits = dense_logits
        X._eng_ws_wrapped = True
        self.logits_ws = ws

    def attach(self, pools) -> None:
        """Views of the rank's pools: comp / idx pages of the 4 sources (packed page), the 3 rings (state pool)."""
        self.views = []
        for r, p in enumerate(pools):
            v = {}
            for s in KV_SOURCES:
                rpp = self.RPP[s]
                v[f"comp{s}"] = torch.as_strided(p.paged, (p.n_paged, rpp, 584), (self.PB, 584, 1), self.PO[f"comp{s}"])
                v[f"idx{s}"] = torch.as_strided(p.paged, (p.n_paged, rpp, 1, 132), (self.PB, 132, 132, 1), self.PO[f"idx{s}"])
                v[f"idxs{s}"] = torch.as_strided(p.paged, (p.n_paged, rpp, 132), (self.PB, 132, 1), self.PO[f"idx{s}"])
            st32 = p.state.view(torch.float32).view(p.n_state, len(RING_LAYERS), 8, 1024)
            for i, s in enumerate(RING_LAYERS):
                v[f"ring{s}"] = st32[:, i]
            # serve D12: persistent per-state-slot block table, updated from the tables' deltas every step
            v["dbt"] = torch.zeros(p.n_state, P_MAX, dtype=torch.int32, device=p.paged.device)
            self.views.append(v)

    def apply_deltas(self, r: int, tab) -> bool:
        """RankTables block-table deltas (seq bt_upd_*) -> the rank's device table. False: tables without deltas."""
        sq = getattr(tab, "seq", None) or {}
        if "bt_upd_seq" not in sq:
            return False
        if not tab.n:
            # an idle rank with delta tables still takes the device-table path: a decode graph captured on an idle rank
            # must contain the per-slot table gather (its `dev_bt` branch is fixed at capture time), else its later
            # replays with real rows read a never-refreshed (zero) block table (symptom: the first sequence of a process
            # right, every later one on a rank idle at capture wrong)
            return IDLE_DEV_BT
        ks, s0, ln = sq["bt_upd_seq"].tolist(), sq["bt_upd_start"].tolist(), sq["bt_upd_len"].tolist()
        pages = sq["bt_upd_pages"]
        if not ks or not sum(ln):
            return True
        slots = sq["state_slot"]
        rows, cols = [], []
        for k, a_, n_ in zip(ks, s0, ln):
            rows += [int(slots[k])] * n_
            cols += list(range(a_, a_ + n_))
        dbt = self.views[r]["dbt"]
        dev = dbt.device
        dbt.index_put_((torch.tensor(rows).to(dev, non_blocking=True), torch.tensor(cols).to(dev, non_blocking=True)),
                       pages.to(torch.int32).pin_memory().to(dev, non_blocking=True))
        return True

    # ---- per bucket device state
    def alloc(self, r: int, s: dict) -> None:
        X, dev, R = self.X, s["dev"], s["R"]
        assert R % 4 == 0, "idx2f hook: decode buckets only (R = 4 x sequences)"
        B = R // 4
        i32, i64 = torch.int32, torch.int64
        with torch.cuda.device(dev):
            if self.eng.cfg.upload_v2:   # the per-step inputs as views of one device blob (one H2D copy per rank)
                up_d = torch.zeros(_up_bytes(R, B), dtype=torch.uint8, device=dev)
                uv = _up_views(up_d, R, B)
                uv["cs1"].fill_(-1)
                uv["cs2"].fill_(-1)
                uv["rs"].fill_(-1)
                d = dict(B=B, bt=torch.zeros(B, P_MAX, dtype=i32, device=dev), preq_slot=uv["pslot"],
                         t2r=(torch.arange(R, device=dev) // 4).to(i32), qsl=torch.arange(0, R + 1, 4, dtype=i32, device=dev),
                         valid=uv["valid"], lat=torch.empty(R, 512, dtype=torch.bfloat16, device=dev),
                         seq_lens={1: uv["sl1"].view(B, 4), 2: uv["sl2"].view(B, 4)},
                         cslot={1: uv["cs1"], 2: uv["cs2"]}, rslot=uv["rs"], up_d=up_d, up_dv=uv, up_h=None)
            else:
                d = dict(B=B, bt=torch.zeros(B, P_MAX, dtype=i32, device=dev), preq_slot=torch.zeros(B, dtype=i64, device=dev),
                         t2r=(torch.arange(R, device=dev) // 4).to(i32), qsl=torch.arange(0, R + 1, 4, dtype=i32, device=dev),
                         valid=torch.zeros(R, dtype=torch.bool, device=dev),
                         lat=torch.empty(R, 512, dtype=torch.bfloat16, device=dev),
                         seq_lens={c: torch.zeros(B, 4, dtype=i32, device=dev) for c in (1, 2)},
                         cslot={c: torch.full((R,), -1, dtype=i64, device=dev) for c in (1, 2)},
                         rslot=torch.full((R,), -1, dtype=i64, device=dev))
            ks = {L: X.CompactK(self.views[r][f"idx{L}"], d["bt"], self.RPP[L]) for L in KV_SOURCES}
            oi = X.OwnedIndexer(ks, B, dev, mode=self.mode)
            if self.logits_impl == "idx2f":
                oi.dense_impl = "idx2f"
            # graph bounds: the largest row the bucket admits (MAX_CTX; long-row levels: LEVEL_CTX[s["lvl"]], idx2f then
            # takes the stock select / top-k statically on layers whose max_len passes its fast-mode limit)
            bound = LEVEL_CTX[s.get("lvl", 0)]
            d["bound"] = bound
            for L in LAYERS:
                oi.max_len[L] = bound // RATIO[L]
            for L in KV_SOURCES:
                oi.width[L] = -(-oi.max_len[L] // 256) * 256
            if self.mode in ("fast", "hybrid"):
                if s.get("lvl", 0):      # long-row levels: [R, max(ceil(fast max_len / CAND_BLOCK), 2048)]
                    fast = max([m for m in oi.max_len.values() if m <= MAX_CTX] or [0])
                    oi.bmax = torch.empty((R, max(-(-fast // X.CAND_BLOCK), 2048)), dtype=torch.float32, device=dev)
                else:
                    oi.bmax = torch.empty((R, -(-MAX_CTX // X.CAND_BLOCK)), dtype=torch.float32, device=dev)
                oi.nb_lens = torch.zeros(R, dtype=i32, device=dev)
            if IDX_FUSE and self.mode == "fast" and self.logits_impl == "idx2f" and hasattr(X, "GlobalOut") \
                    and not s.get("lvl", 0):
                oi.fuse = True
            if IDX_SIDE:
                d["side"] = torch.cuda.Stream(device=dev)
                d["wbuf"] = torch.empty((R, 32), dtype=torch.bfloat16, device=dev)
            if s.get("lvl", 0) and hasattr(oi, "logits_scratch"):   # one shared fp32 scratch per device for the long variants
                oi.logits_scratch = self._long_scratch(dev, R, bound)
            d["oi"] = oi
            d["global_rows"] = 0
            E = sys.modules.get("engine.engine")
            if E is not None and E.POISON:   # debug: the decode bucket's persistent indexer state (never rewritten wholesale)
                E.poison_(d["lat"], oi.cand, oi.compact, oi.bmax, *list(oi.topk.values()))
        s["idx"] = d

    def _long_scratch(self, dev, R: int, bound: int):
        """ONE fp32 logits scratch per device shared by every long-row variant, sized once for the largest
        (largest decode bucket x LEVEL_CTX[-1]: 839 MB at 320 rows), so no variant's graph ever points at a freed one."""
        cur = self.__dict__.setdefault("_lscr", {}).get(str(dev))
        if cur is None:
            cur = torch.empty(max(self.eng.cfg.buckets) * LEVEL_CTX[-1], dtype=torch.float32, device=dev)
            self._lscr[str(dev)] = cur
        assert R * bound <= cur.numel()
        return cur[: R * bound].view(R, bound)

    # ---- host side of a step
    def _set_logits(self) -> None:
        """IDX_F8: idx2f's logits kernel family for this step's calls (ext_logits() reads idx2f.LOGITS per call; graphs keep
        the family they were captured with, and tests recapture per arm)."""
        if self._logits0 is not None:
            want = "f8" if IDX_F8 else self._logits0
            if self.X.LOGITS != want:
                self.X.LOGITS = want

    def upload(self, r: int, s: dict, sb, tab=None) -> None:
        """Decode bucket tables, vectorized (numpy, O(rows)); block tables from the per-slot device table (D12 deltas)."""
        self._set_logits()
        if self.eng.cfg.upload_v2:
            return self._upload_v2(r, s, sb, tab)
        import numpy as np
        d = s["idx"]
        R, B = s["R"], d["B"]
        n = sb.n_rows
        S = len(sb.seqs)
        assert n == 4 * S and all(se.kind == "decode" and se.n_rows == 4 and se.row0 == 4 * b for b, se in enumerate(sb.seqs)), \
            "idx2f hook: decode rows only (4 per sequence, sequence-contiguous)"
        d["dev_bt"] = tab is not None and self.apply_deltas(r, tab)
        cap = d.get("dev_bt_captured")
        if cap is not None and cap != d["dev_bt"]:
            # run()'s block-table branch is fixed in the bucket's captured graph: a replay with the other table mode would
            # read a stale block table. Never silently.
            raise RuntimeError(f"idx2f hook rank {r}: decode graph captured with dev_bt={cap}, step has dev_bt={d['dev_bt']}")
        i64 = np.int64
        pos = np.asarray(sb.positions, dtype=i64)
        if n:
            assert int(pos.max()) + 1 <= d.get("bound", MAX_CTX), \
                f"context {int(pos.max()) + 1} > {d.get('bound', MAX_CTX)} (the decode bucket's indexer bound)"
        ps = np.asarray(sb.paged_slot, dtype=i64)
        sslot = np.asarray([se.state_slot for se in sb.seqs], dtype=i64)
        sl1 = np.zeros((B, 4), np.int32)
        sl2 = np.zeros((B, 4), np.int32)
        sl1.reshape(-1)[:n] = pos + 1
        sl2.reshape(-1)[:n] = (pos + 1) // 2
        cs1 = np.full(R, -1, i64)
        cs2 = np.full(R, -1, i64)
        okp = ps >= 0
        pg, off = ps // 64, ps % 64
        cs1[:n] = np.where(okp, pg * 64 + off, -1)
        cs2[:n] = np.where(okp, pg * 32 + off // 2, -1)
        rs = np.full(R, -1, i64)
        rs[:n] = np.repeat(sslot, 4) * 8 + pos % 8
        valid = np.zeros(R, bool)
        valid[:n] = True
        pslot = np.full(B, self.views[r]["dbt"].shape[0] - 1, i64)
        pslot[:S] = sslot
        ups = [(sl1, d["seq_lens"][1]), (sl2, d["seq_lens"][2]), (cs1, d["cslot"][1]), (cs2, d["cslot"][2]), (rs, d["rslot"]),
               (valid, d["valid"]), (pslot, d["preq_slot"])]
        if not d["dev_bt"]:                      # tests without deltas: the full table on the host
            bt = np.zeros((B, P_MAX), np.int32)
            for b, se in enumerate(sb.seqs):
                nb = se.n_blocks if hasattr(se, "n_blocks") else len(se.block_table)   # never the live length (serve)
                bt[b, :nb] = se.block_table[:nb]
            ups.append((bt, d["bt"]))
        for h, t in ups:
            t.copy_(torch.from_numpy(h).pin_memory(), non_blocking=True)
        d["global_rows"] = sum(x.n_rows for x in self.eng._batches) if getattr(self.eng, "_batches", None) else n
        d["max_seq_len"] = d.get("bound", MAX_CTX)

    def _upload_v2(self, r: int, s: dict, sb, tab=None) -> None:
        """upload() with the same device result: every per-step array (and the block-table deltas, up to NU_MAX entries)
        filled into one pinned host blob of the bucket (two sets, a set refilled only after its copy from two steps ago
        completed) -> ONE H2D copy into the bucket's blob (the indexer's input tensors are its views)."""
        import numpy as np
        d = s["idx"]
        R, B = s["R"], d["B"]
        n = sb.n_rows
        S = len(sb.seqs)
        assert n == 4 * S and all(se.kind == "decode" and se.n_rows == 4 and se.row0 == 4 * b for b, se in enumerate(sb.seqs)), \
            "idx2f hook: decode rows only (4 per sequence, sequence-contiguous)"
        if d["up_h"] is None:
            nbytes = d["up_d"].numel()
            d["up_h"] = [[torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
            d["up_hv"] = [_up_views(x[0], R, B, numpy=True) for x in d["up_h"]]
            d["up_i"] = 0
        i = d["up_i"]
        hb, ev = d["up_h"][i]
        if ev is not None:
            ev.synchronize()
        v = d["up_hv"][i]
        # block-table deltas (apply_deltas, same entries in the same order)
        sq = (getattr(tab, "seq", None) or {}) if tab is not None else {}
        nu, big = 0, None
        if tab is None or "bt_upd_seq" not in sq:
            dev_bt = False
        elif not tab.n:
            dev_bt = IDLE_DEV_BT
        else:
            dev_bt = True
            ks, s0, ln = sq["bt_upd_seq"].tolist(), sq["bt_upd_start"].tolist(), sq["bt_upd_len"].tolist()
            nu = int(sum(ln)) if ks else 0
            if nu:
                slots = sq["state_slot"]
                rows = np.repeat(np.asarray([int(slots[k]) for k in ks], np.int64), ln)
                cols = np.concatenate([np.arange(a_, a_ + n_, dtype=np.int64) for a_, n_ in zip(s0, ln)])
                pages = sq["bt_upd_pages"].numpy().astype(np.int32, copy=False)
                if nu <= NU_MAX:
                    v["du_rows"][:nu], v["du_cols"][:nu], v["du_pages"][:nu] = rows, cols, pages
                else:                            # (rare: more new pages than the blob holds) per tensor, as v1
                    big = (rows, cols, pages)
                    nu = 0
        d["dev_bt"] = dev_bt
        cap = d.get("dev_bt_captured")
        if cap is not None and cap != d["dev_bt"]:
            raise RuntimeError(f"idx2f hook rank {r}: decode graph captured with dev_bt={cap}, step has dev_bt={d['dev_bt']}")
        i64 = np.int64
        pos = np.asarray(sb.positions, dtype=i64)
        if n:
            assert int(pos.max()) + 1 <= d.get("bound", MAX_CTX), \
                f"context {int(pos.max()) + 1} > {d.get('bound', MAX_CTX)} (the decode bucket's indexer bound)"
        ps = np.asarray(sb.paged_slot, dtype=i64)
        sslot = np.asarray([se.state_slot for se in sb.seqs], dtype=i64)
        sl1, sl2 = v["sl1"], v["sl2"]
        sl1[:] = 0
        sl2[:] = 0
        sl1[:n] = pos + 1
        sl2[:n] = (pos + 1) // 2
        cs1, cs2, rs = v["cs1"], v["cs2"], v["rs"]
        cs1[:] = -1
        cs2[:] = -1
        okp = ps >= 0
        pg, off = ps // 64, ps % 64
        cs1[:n] = np.where(okp, pg * 64 + off, -1)
        cs2[:n] = np.where(okp, pg * 32 + off // 2, -1)
        rs[:] = -1
        rs[:n] = np.repeat(sslot, 4) * 8 + pos % 8
        v["valid"][:] = False
        v["valid"][:n] = True
        v["pslot"][:] = self.views[r]["dbt"].shape[0] - 1
        v["pslot"][:S] = sslot
        d["up_d"].copy_(hb, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(s["dev"]))
        d["up_h"][i][1] = ev
        d["up_i"] = i ^ 1
        dbt = self.views[r]["dbt"]
        if nu:
            uv = d["up_dv"]
            dbt.index_put_((uv["du_rows"][:nu], uv["du_cols"][:nu]), uv["du_pages"][:nu])
        elif big is not None:
            dev = dbt.device
            dbt.index_put_((torch.from_numpy(big[0]).to(dev), torch.from_numpy(big[1]).to(dev)),
                           torch.from_numpy(big[2]).to(dev))
        if not d["dev_bt"]:                      # tests without deltas: the full table on the host
            bt = np.zeros((B, P_MAX), np.int32)
            for b, se in enumerate(sb.seqs):
                nb = se.n_blocks if hasattr(se, "n_blocks") else len(se.block_table)   # never the live length (serve)
                bt[b, :nb] = se.block_table[:nb]
            d["bt"].copy_(torch.from_numpy(bt).pin_memory(), non_blocking=True)
        d["global_rows"] = sum(x.n_rows for x in self.eng._batches) if getattr(self.eng, "_batches", None) else n
        d["max_seq_len"] = d.get("bound", MAX_CTX)

    def upload_d13(self, r: int, s: dict, sb) -> None:
        """D13 async decode: the host-known part of upload() (sequence slots, valid rows, the per-row state-slot base of
        rslot) into the bucket's blob; positions / paged slots come from rowexp (device_tables_d13 after its expand).
        Block tables: always the per-slot device table (D13 applies the step's deltas itself)."""
        self._set_logits()
        import numpy as np
        d = s["idx"]
        R, B = s["R"], d["B"]
        n = sb.n_rows
        S = len(sb.seqs)
        assert n == 4 * S, "idx2f hook (D13): decode rows only (4 per sequence, sequence-contiguous)"
        assert "up_d" in d, "D13 needs upload_v2 buckets (EngineConfig.upload_v2)"
        if d["up_h"] is None:
            nbytes = d["up_d"].numel()
            d["up_h"] = [[torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
            d["up_hv"] = [_up_views(x[0], R, B, numpy=True) for x in d["up_h"]]
            d["up_i"] = 0
        d["dev_bt"] = True
        cap = d.get("dev_bt_captured")
        if cap is not None and cap != d["dev_bt"]:
            raise RuntimeError(f"idx2f hook rank {r}: decode graph captured with dev_bt={cap}, step has dev_bt=True")
        i = d["up_i"]
        hb, ev = d["up_h"][i]
        if ev is not None:
            ev.synchronize()
        v = d["up_hv"][i]
        sslot = np.asarray([se.state_slot for se in sb.seqs], dtype=np.int64)
        v["valid"][:] = False
        v["valid"][:n] = True
        v["pslot"][:] = self.views[r]["dbt"].shape[0] - 1
        v["pslot"][:S] = sslot
        v["rs"][:] = -1
        v["rs"][:n] = np.repeat(sslot, 4) * 8         # + pos % 8 on the device
        d["up_d"].copy_(hb, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(s["dev"]))
        d["up_h"][i][1] = ev
        d["up_i"] = i ^ 1
        d["global_rows"] = sum(x.n_rows for x in self.eng._batches) if getattr(self.eng, "_batches", None) else n
        d["max_seq_len"] = d.get("bound", MAX_CTX)

    def device_tables_d13(self, r: int, s: dict, paged: torch.Tensor) -> None:
        """D13: upload()'s position-derived tables on the device from rowexp's pos / paged slots (same values as the
        host formulas: seq_lens = pos + 1 / (pos + 1) // 2, compressed slots of the paged slot, rslot = slot * 8 + pos % 8;
        padding rows 0 / -1). On the current (step) stream, after rowexp's expand."""
        d = s["idx"]
        R = s["R"]
        pos, v = s["pos"][:R], d["valid"]
        p1 = pos + 1
        d["seq_lens"][1].view(-1).copy_(torch.where(v, p1, 0))
        d["seq_lens"][2].view(-1).copy_(torch.where(v, p1 // 2, 0))
        ok = v & (paged >= 0)
        d["cslot"][1].copy_(torch.where(ok, paged, -1))
        d["cslot"][2].copy_(torch.where(ok, (paged // 64) * 32 + (paged % 64) // 2, -1))
        d["rslot"].copy_(torch.where(v, d["rslot"] + pos % 8, -1))

    # ---- mixed steps (eager): any row layout (decode rows + prefill chunks + padding)
    @staticmethod
    def layout(sb):
        """Indexer rows: pseudo-requests of 4 consecutive rows of ONE sequence (DeepGEMM next_n = 4 shares a block table;
        rows of a pseudo-request past the sequence's rows are padding). Decode sequences: their own 4 rows."""
        ir2row, preq_seq = [], []
        for si, se in enumerate(sb.seqs):
            for c0 in range(0, se.n_rows, 4):
                preq_seq.append(si)
                ir2row += [se.row0 + c0 + j if c0 + j < se.n_rows else -1 for j in range(4)]
        return ir2row, preq_seq

    def upload_general(self, r: int, s: dict, sb, tab=None) -> None:
        """Tables of an eager mixed step: ENG_IDX_TABLES selects the device tables (tables2f.GeneralTables), the host
        path (_upload_general_host), or both compared."""
        self._set_logits()
        if self.tables == "host" or tab is None:
            return self._upload_general_host(r, s, sb, tab)
        gt = s.get("idx_gt")
        if gt is None:
            import tables2f as T2
            v = self.views[r]
            gt = s["idx_gt"] = T2.GeneralTables(s["dev"], s["R"], s["pos"], v["dbt"],
                                                {L: v[f"idx{L}"] for L in KV_SOURCES}, self.RPP, self.mode,
                                                "idx2f" if self.logits_impl == "idx2f" else "deepgemm")
        grows = sum(x.n_rows for x in self.eng._batches) if getattr(self.eng, "_batches", None) else sb.n_rows
        d = gt.upload(tab, grows, None)     # eager: rows > 262,144 positions take idx2f's stock fallback
        if d is None:                       # tables without D12 deltas or rows out of sequence order: host path
            self.checked["fallback"] += 1
            return self._upload_general_host(r, s, sb, tab)
        if self.tables == "device":
            s["idx"] = d
            return
        self._upload_general_host(r, s, sb, tab)
        self._compare(s["idx"], d, sb.n_rows)

    def _compare(self, h: dict, d: dict, n: int) -> None:
        """ENG_IDX_TABLES=check: the device tables == the host tables on every key the hook reads."""
        bad = []
        self.checked["steps"] += 1
        if h.get("idle") or d.get("idle"):
            if bool(h.get("idle")) != bool(d.get("idle")):
                bad.append("idle")
        else:
            for k in ("B", "IR", "global_rows", "max_seq_len"):
                if int(h[k]) != int(d[k]):
                    bad.append(f"{k} {h[k]} != {d[k]}")
            IR, B = h["IR"], h["B"]
            pairs = [("t2r_ir", IR), ("valid_ir", IR), ("ir2row_c", IR), ("vir", n), ("vrow", n), ("rslot", None),
                     ("qsl", None), ("t2r", n)]
            for k, m in pairs:
                a_, b_ = h[k], d[k]
                if m is not None:
                    a_, b_ = a_[:m], b_[:m]
                if a_.shape != b_.shape or not bool((a_.to(b_.dtype) == b_).all()):
                    bad.append(k)
            for c in (1, 2):
                if not bool((h["seq_lens"][c] == d["seq_lens"][c]).all()):
                    bad.append(f"seq_lens{c}")
                if not bool((h["cslot"][c] == d["cslot"][c]).all()):      # all R rows: padding rows must be -1
                    bad.append(f"cslot{c}")
            P = min(h["bt"].shape[1], d["bt"].shape[1])
            if not bool((h["bt"][:B, :P] == d["bt"][:B, :P]).all()):
                bad.append("bt")
            rh = {int(k): int(v) for k, v in h["row2ir"].items()}
            rd = {i: int(x) for i, x in enumerate(d["row2ir"][:n])}
            if rh != rd:
                bad.append("row2ir")
            oh, od = h["oi"], d["oi"]
            for L in (2, 8, 14, 20, 24, 28, 32, 36):
                if oh.max_len.get(L) != od.max_len.get(L):
                    bad.append(f"max_len{L} {oh.max_len.get(L)} != {od.max_len.get(L)}")
            for L in (2, 8, 14, 20):
                if oh.width.get(L) != od.width.get(L):
                    bad.append(f"width{L}")
        if bad:
            self.checked["mismatch"] += 1
            print(f"ENG_IDX_TABLES check: device != host tables: {bad[:8]}", flush=True)

    def _upload_general_host(self, r: int, s: dict, sb, tab=None) -> None:
        """Host tables of an eager mixed step (bucket rows = sb's rows, padded to s['R']); a fresh OwnedIndexer sized for
        this step's pseudo-requests, planned on the host bounds (OwnedIndexer.plan)."""
        X, dev, R = self.X, s["dev"], s["R"]
        if POP_IDX_FIRST:
            s.pop("idx", None)          # the previous step's indexer state (finished: steps end synchronized)
        ir2row, preq_seq = self.layout(sb)
        if not ir2row:
            ir2row, preq_seq = [-1] * 4, [0]
        IR, B = len(ir2row), len(preq_seq)
        i32, i64 = torch.int32, torch.int64
        seqs = sb.seqs
        nbs = [se.n_blocks if hasattr(se, "n_blocks") else len(se.block_table) for se in seqs]
        dev_bt = tab is not None and self.apply_deltas(r, tab)
        if dev_bt:
            pq = torch.tensor([seqs[si].state_slot for si in preq_seq] if seqs else [0], dtype=i64).to(dev, non_blocking=True)
            bt = None
        else:
            P = max(nbs + [1])
            P = -(-P // 64) * 64
            bt = torch.zeros(B, P, dtype=i32)
            for b, si in enumerate(preq_seq):
                if seqs:
                    tb = seqs[si].block_table[: nbs[si]]
                    bt[b, : len(tb)] = torch.tensor(tb, dtype=i32)
        pos = torch.tensor(sb.positions + [0] * (R - sb.n_rows), dtype=i64)
        irr = torch.tensor(ir2row, dtype=i64)
        ok = irr >= 0
        pir = torch.where(ok, pos[irr.clamp(min=0)], torch.zeros_like(irr))
        sl = {2: torch.where(ok, (pir + 1) // 2, 0).to(i32).view(B, 4), 1: torch.where(ok, pir + 1, 0).to(i32).view(B, 4)}
        # padding rows inside a group carry the group's last real row's length (DeepGEMM next_n = 4 scans the
        # pages of the group's last row; a 0 there makes the whole group scan nothing); they stay invalid (valid_ir)
        okv = ok.view(B, 4)
        for c in (1, 2):
            v = sl[c]
            last = torch.where(okv, v, torch.zeros_like(v)).max(1, keepdim=True).values
            sl[c] = torch.where(okv, v, last).contiguous()
        # no MAX_CTX limit here: in eager steps idx2f's plan routes layers with rows > 262,144 compressed positions to
        # the stock select / top-k; only the decode graphs keep MAX_CTX
        cs = {c: torch.full((R,), -1, dtype=i64) for c in (1, 2)}
        rs = torch.full((R,), -1, dtype=i64)
        qsl = torch.zeros(max(len(seqs), 1) + 1, dtype=i32)
        t2r = torch.full((R,), max(len(seqs) - 1, 0), dtype=i32)
        for si, se in enumerate(seqs):
            qsl[si + 1] = se.row0 + se.n_rows
            for j in range(se.n_rows):
                row, p = se.row0 + j, se.start + j
                t2r[row] = si
                ps = sb.paged_slot[row]
                if ps >= 0:
                    pg, off = ps // 64, ps % 64
                    cs[2][row], cs[1][row] = pg * 32 + off // 2, pg * 64 + off
                rs[row] = se.state_slot * 8 + p % 8
        vir = ok.nonzero().flatten()
        to = lambda x: x.pin_memory().to(dev, non_blocking=True)  # noqa: E731
        d = dict(general=True, B=B, IR=IR, bt=(self.views[r]["dbt"].index_select(0, pq) if dev_bt else to(bt)), t2r_ir=to(torch.arange(IR, dtype=i32) // 4), valid_ir=to(ok),
                 ir2row_c=to(irr.clamp(min=0)), vir=to(vir), vrow=to(irr[vir]), seq_lens={c: to(sl[c]) for c in (1, 2)},
                 cslot={c: to(cs[c]) for c in (1, 2)}, rslot=to(rs), qsl=to(qsl), t2r=to(t2r),
                 lat=torch.empty(R, 512, dtype=torch.bfloat16, device=dev), row2ir={int(x): i for i, x in enumerate(ir2row) if x >= 0},
                 ir2row_h=list(ir2row),
                 global_rows=sum(x.n_rows for x in self.eng._batches) if getattr(self.eng, "_batches", None) else sb.n_rows,
                 max_seq_len=max(int(pir.max()) + 1, 1))
        if not sb.n_rows or not int(ok.sum()):
            d["idle"] = True            # an idle rank (no rows this step): the hook does nothing on it
            s["idx"] = d
            return
        ks = {L: X.CompactK(self.views[r][f"idx{L}"], d["bt"], self.RPP[L]) for L in KV_SOURCES}
        lean = self.lean and hasattr(X, "EagerWorkspace")
        if lean:   # idx2f's lean eager indexer: every per-call buffer is a view of one fixed ~406 MB workspace per device
            oi = X.OwnedIndexer(ks, B, dev, mode=self.mode, lean=True, ws=X.EagerWorkspace.of(dev))
        else:
            oi = X.OwnedIndexer(ks, B, dev, mode=self.mode)
        if self.logits_impl == "idx2f":
            oi.dense_impl = "idx2f"
        ins = {L: X.LayerIn(None, None, d["seq_lens"][RATIO[L]], d["max_seq_len"], d["global_rows"]) for L in LAYERS}
        if lean:
            oi.plan_bounds(ins, d["max_seq_len"])
        elif hasattr(oi, "plan_bounds") and USE_PLAN_BOUNDS:
            # host bounds (no device reads; = plan(): max_len[L] = (max position + 1) // ratio) and a PERSISTENT block-max
            # buffer per bucket: plan() allocated a fresh [IR, context / 8] fp32 one every step, whose size grows with
            # every chunk of a long prompt, so the caching allocator never reused it (memory growth on long prompts)
            nbw = max(-(-min(d["max_seq_len"], X.LONG_ROW) // X.CAND_BLOCK), X.CAND_K)
            buf = s.get("idx_bmax")
            if buf is None or buf.numel() < IR * nbw:
                s["idx_bmax"] = buf = None
                cap = max(IR, s["R"] + 512) * max(nbw, -(-X.LONG_ROW // X.CAND_BLOCK))
                buf = s["idx_bmax"] = torch.empty(cap, dtype=torch.float32, device=dev)
            oi.plan_bounds(ins, d["max_seq_len"], None, buf)
        else:
            oi.plan(ins)
        if not lean and self.mode in ("fast", "hybrid") and oi.bmax.shape[1] < X.CAND_K:   # consumers: 16,384 positions
            oi.bmax = torch.empty((IR, X.CAND_K), dtype=torch.float32, device=dev)
        d["oi"] = oi
        s["idx"] = d
        E = sys.modules.get("engine.engine")
        if E is not None and E.POISON:      # debug: the fresh per-step indexer state, poisoned (see engine.poison_), on
            E.poison_(d["lat"], oi.cand, oi.compact, getattr(oi, "bmax", None), *list(oi.topk.values()))   # the current
            # stream: upload_general runs inside the rank's engine-stream context (prepare / step_mixed)

    def carry_candidates(self, sA: dict, sT: dict, tail_rows: list, ix=None) -> None:
        """Layer 20's candidate blocks of the tail rows (computed on all rows) -> the tail bucket's indexer (24..36).
        ix = the tail rows as a device int64 tensor (device tables: device-side maps, no host lists)."""
        dA, dT = sA["idx"], sT["idx"]
        if dA.get("idle") or dT.get("idle") or not tail_rows:
            return
        if ix is not None and "row2ir_dev" in dA and "row2ir_dev" in dT:     # ENG_IDX_TABLES=device
            IR = int(dT["IR"])
            dev = sA["dev"]
            if CAND_PAD_FIX:
                # every T indexer row, padding rows of a 4-row group included: they take the group's last real row (the
                # real rows of a group come first; GeneralTables keeps cand across steps, so padding rows held stale ids)
                ar = torch.arange(IR, device=dev)
                last = torch.where(dT["valid_ir"][:IR], ar, torch.full_like(ar, -1)).view(-1, 4).cummax(1).values.view(-1)
                trow = dT["ir2row_c"][:IR].index_select(0, last.clamp(min=0))          # IR row -> its T row
                src = dA["row2ir_dev"].index_select(0, ix.index_select(0, trow))
                dT["oi"].cand.index_copy_(0, ar, dA["oi"].cand.index_select(0, src))
            else:
                m = len(tail_rows)
                dT["oi"].cand.index_copy_(0, dT["row2ir_dev"][:m],
                                          dA["oi"].cand.index_select(0, dA["row2ir_dev"].index_select(0, ix)))
            return
        src = [dA["row2ir"][row] for row in tail_rows]
        dst = [dT["row2ir"][t] for t in range(len(tail_rows))]
        irT = dT.get("ir2row_h")
        if CAND_PAD_FIX and irT is not None:
            # the tail bucket's indexer is fresh each step (torch.empty cand): padding rows of a 4-row group (a sequence's
            # tail rows % 4 != 0) take part in the group's candidate union (they carry the group's last real length), so
            # they get the group's last real row's candidates instead of uninitialized block ids
            src, dst = [], []
            for g in range(0, len(irT), 4):
                grp = irT[g:g + 4]
                real = [t for t in grp if t >= 0]
                if not real:
                    continue
                for j, t in enumerate(grp):
                    dst.append(g + j)
                    src.append(dA["row2ir"][tail_rows[t if t >= 0 else real[-1]]])
        dev = sA["dev"]
        dT["oi"].cand.index_copy_(0, torch.tensor(dst, device=dev), dA["oi"].cand.index_select(0, torch.tensor(src, device=dev)))

    def _run_general(self, r: int, s: dict, L: int) -> None:
        """Mixed steps: the compressor half then the scoring half (CP prefill exchanges index pages between them)."""
        self._compress_general(r, s, L)
        self._score_general(r, s, L)

    def _compress_general(self, r: int, s: dict, L: int) -> None:
        """KV-source layers: the compressed / index records of this step's rows (kv_score GEMM, pooling, inserts)."""
        from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import fused_save_compress_norm, rope_quant_insert

        W, d, v = self.W[r][L], s["idx"], self.views[r]
        ratio = RATIO[L]
        x, pos = s["a"], s["pos"]
        if d.get("idle") or L not in KV_SOURCES:
            return
        if True:
            ks = self.dbg.get(("ks", L))
            if ks is None:
                ks = torch.mm(x, W["comp_w"].T, out_dtype=torch.float32)
            cslot = d["cslot"][ratio]
            if ratio == 2:
                fused_save_compress_norm(ks, pos, v[f"ring{L}"], d["rslot"], d["qsl"], d["t2r"], W["comp_norm"], EPS, ratio,
                                         d["lat"])
            else:
                fused_save_compress_norm(ks, pos, None, cslot, None, None, W["comp_norm"], EPS, ratio, d["lat"])
            rope_quant_insert(d["lat"], pos, W["cos_sin"], v[f"comp{L}"], cslot, ratio)
            kpre = F.linear(d["lat"], W["idx_wk"])
            indexer_k_norm_rope_store(kpre, pos, W["cos_sin"], W["idx_k_norm"], EPS, v[f"idxs{L}"], cslot, ratio, False)

    def _score_general(self, r: int, s: dict, L: int) -> None:
        """Index layers: scoring over the context's index keys -> the rows' top-512 (ti / tl)."""
        from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
        from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens

        X, W, d = self.X, self.W[r][L], s["idx"]
        R, ratio = s["R"], RATIO[L]
        x, qr, pos = s["a"], s["ws"]["qr"], s["pos"]
        if d.get("idle"):        # no rows on this rank this step: nothing to compress or score; FlashMLA reads no keys
            s["ti"][L].fill_(-1)
            s["tl"][L].zero_()
            return
        oi = d["oi"]
        if oi.max_len.get(L, 1) <= 0:
            # no compressed position yet on any row (e.g. a 1-token prompt at ratio 2: (0 + 1) // 2 = 0): the top-512 is
            # empty (production: zero-length rows); idx2f has no zero-width logits call
            s["ti"][L].fill_(-1)
            s["tl"][L].zero_()
            return
        w = F.linear(x, W["idx_wp"])
        q = F.linear(qr, W["idx_wq_b"])
        q8, wo = fused_indexer_q_rope_quant(pos, q.view(R, 32, 128), W["cos_sin"], w, 128 ** -0.5, 32 ** -0.5)
        q8i = q8.index_select(0, d["ir2row_c"]).view(d["B"], 4, 32, 128)
        woi = wo.index_select(0, d["ir2row_c"]).contiguous()
        top = oi.layer(L, X.LayerIn(q8i, woi, d["seq_lens"][ratio], d["max_seq_len"], d["global_rows"]))
        oi.logits.pop(L, None)       # eager: free the [IR, width] fp32 logits before the next layer
        g, ln = compute_global_topk_indices_and_lens(top, d["t2r_ir"], d["bt"], self.RPP[K_SOURCE[L]], d["valid_ir"])
        s["ti"][L].fill_(-1)     # rows of this bucket that are not this step's indexer rows: no keys (no stale lists)
        s["tl"][L].zero_()
        s["ti"][L].view(R, 512).index_copy_(0, d["vrow"], g.index_select(0, d["vir"]))
        s["tl"][L].index_copy_(0, d["vrow"], ln.index_select(0, d["vir"]))

    # ---- device side, inside the layer (graph-capturable)
    def run(self, r: int, s: dict, L: int) -> None:
        if s["idx"].get("general"):
            return self._run_general(r, s, L)
        from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant, indexer_k_norm_rope_store
        from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import fused_save_compress_norm, rope_quant_insert

        X, W, d, v = self.X, self.W[r][L], s["idx"], self.views[r]
        R, ratio = s["R"], RATIO[L]
        x, qr, pos = s["a"], s["ws"]["qr"], s["pos"]
        oi = d["oi"]
        if L == min(l for l in LAYERS if l in self.eng.layers):   # once per step: DeepGEMM schedules from device lengths
            if torch.cuda.is_current_stream_capturing():
                d["dev_bt_captured"] = bool(d.get("dev_bt"))     # upload() asserts every later step matches
            if d.get("dev_bt"):    # the step's block tables = the persistent per-slot table's rows (in the graph)
                torch.index_select(v["dbt"], 0, d["preq_slot"], out=d["bt"])
            for Ls in KV_SOURCES:
                if Ls in self.eng.layers and self.logits_impl == "deepgemm":
                    oi.sched[Ls] = X.dg_schedule(d["seq_lens"][RATIO[Ls]], self.RPP[Ls])
            if self.mode in ("fast", "hybrid") and not getattr(oi, "fuse", False):   # fused: select reads the lengths
                oi.nb_lens.copy_(((d["seq_lens"][1].reshape(-1) + X.CAND_BLOCK - 1) // X.CAND_BLOCK).to(torch.int32))
        side = d.get("side")
        if side is not None:     # IDX_SIDE: fork; the compressor (+ weights_proj) next to the q GEMM
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            sctx = torch.cuda.stream(side)
            sctx.__enter__()
        if L in KV_SOURCES:
            ks = self.dbg.get(("ks", L))
            if ks is None:
                ks = torch.mm(x, W["comp_w"].T, out_dtype=torch.float32)
            cslot = d["cslot"][ratio]
            if ratio == 2:
                fused_save_compress_norm(ks, pos, v[f"ring{L}"], d["rslot"], d["qsl"], d["t2r"], W["comp_norm"], EPS, ratio,
                                         d["lat"])
            else:
                fused_save_compress_norm(ks, pos, None, cslot, None, None, W["comp_norm"], EPS, ratio, d["lat"])
            rope_quant_insert(d["lat"], pos, W["cos_sin"], v[f"comp{L}"], cslot, ratio)
            kpre = F.linear(d["lat"], W["idx_wk"])
            indexer_k_norm_rope_store(kpre, pos, W["cos_sin"], W["idx_k_norm"], EPS, v[f"idxs{L}"], cslot, ratio, False)
        if side is not None:
            torch.mm(x, W["idx_wp"].t(), out=d["wbuf"])     # == F.linear(x, idx_wp) bitwise
            sctx.__exit__(None, None, None)
            q = F.linear(qr, W["idx_wq_b"])
            cur.wait_stream(side)                            # join: the indexer reads this layer's new index-K entries
            q8, wo = fused_indexer_q_rope_quant(pos, q.view(R, 32, 128), W["cos_sin"], d["wbuf"], 128 ** -0.5, 32 ** -0.5)
        else:
            w = F.linear(x, W["idx_wp"])
            q = F.linear(qr, W["idx_wq_b"])
            q8, wo = fused_indexer_q_rope_quant(pos, q.view(R, 32, 128), W["cos_sin"], w, 128 ** -0.5, 32 ** -0.5)
        if ("q8", L) in self.dbg:
            q8, wo = self.dbg[("q8", L)], self.dbg[("wo", L)]
        lin = X.LayerIn(q8.view(d["B"], 4, 32, 128), wo, d["seq_lens"][ratio], d["max_seq_len"], d["global_rows"])
        if getattr(oi, "fuse", False):   # IDX_FUSE: the top-k writes FlashMLA's slots / lengths itself
            oi.layer(L, lin, glob=X.GlobalOut(s["ti"][L].view(R, 512), s["tl"][L], d["t2r"], d["bt"],
                                              self.RPP[K_SOURCE[L]], d["valid"]))
            return
        top = oi.layer(L, lin)
        g, ln = compute_global_topk_indices_and_lens(top, d["t2r"], d["bt"], self.RPP[K_SOURCE[L]], d["valid"])
        s["ti"][L].view(R, 512).copy_(g)
        s["tl"][L].copy_(ln)
