"""The real worker behind PhysExecutor: serve/worker.py's Worker (pools, copies, tables) over a GPU backend.

PHYS_WORKER_BACKEND = engine (default: serve's Worker over engine/engine.py's Engine) | engine-direct (the Engine
                      driven directly: it owns pools, copies and tables; Engine.step(batches, tokens,
                      sampling_state)) | splayer (kernels/sp_layer chain with the stub head) | stub (CPU)
PHYS_ENGINE_CLS     = the engine class for backend engine (default engine.engine.Engine; mock_engine.MockEngine = serve's
                      contract mock without a model, CPU)
PHYS_ENGINE_KW      = 'field=value,...' Engine kwargs for backend engine (serve WorkerConfig.engine_kw: indexer=idx2f,
                      engram=phys, head=hsd, graphs=0, ...)
PHYS_ENGINE_CFG     = 'field=value,...' EngineConfig overrides for engine-direct
PHYS_WORKER_MAX_ROWS = per-rank row buffer of the step program (default 4,416 = 4,096 prefill + 80 x 4 decode)
DS41H_MODEL_DIR / MODEL: checkpoint (default: the model vLLM serves)

If the Worker grows a native `step(ps, grammar)` (the protocol of adapter/fake_worker.py), it is called directly.
Otherwise this wraps `Worker.run_step(batches)`: its TableBuilder reads the input token of row (rid, pos) through
`tokens_of`, which here looks the step's row tokens up (PhysStep.row_tokens: prompt / last-sampled tokens, -1 for
device-side drafts, forced drafts >= 0), and the token each row predicts (forced requests) from the force list.
New requests' sampling params and frees are handed to optional hooks `on_new_requests(list[ReqInit])` /
`on_free(set[rid])` when the Worker has them.

Images (PHYS_MM=1, adapter/mm.py): only with an engine that applies span rows (its `supports_images`). One VisionTower
per GPU (engine/vision.py) and one EncoderCache per rank. Each step's PhysStep.mm is applied before the step: drop,
encode (on the engine's step stream of the rank), gather. The rank's (idx int32 [M], emb bf16 [M, hidden]) go to the
engine through serve's Worker.step(ps, grammar, mm=...) as StepContext.mm (engine-direct: sampling_state["mm"]).
"""
from __future__ import annotations

import os
import time

from adapter import bridge  # noqa: F401
from vllm.logger import init_logger

logger = init_logger("vllm.phys_adapter")


def _vision_weight_bytes(model_dir: str) -> int:
    """Bytes of one vision tower + aligner on a GPU in bfloat16, from the checkpoint's safetensors headers."""
    import glob
    import json
    import struct

    size = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8}
    n = 0
    for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with open(f, "rb") as fh:
            h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in h.items():
            if k.startswith(("vision.", "aligner.")):
                b = v["data_offsets"][1] - v["data_offsets"][0]
                n += b // size.get(v["dtype"], 2) * 2
    return n


class Stage4Worker:
    def __init__(self, vllm_config):
        import geometry as G
        import worker as W

        mc = vllm_config.model_config
        backend = os.environ.get("PHYS_WORKER_BACKEND", "engine")
        # engine: max_rows = the largest decode-graph bucket (320 rows per rank); the worker sizes the step rows from the
        # engine's mixed-step capacity itself. splayer: its buffer rows (128 = the smoke-tested size).
        max_rows = int(os.environ.get("PHYS_WORKER_MAX_ROWS", "320" if backend == "engine" else "128"))
        cfg = W.WorkerConfig()
        model = os.environ.get("DS41H_MODEL_DIR") or mc.model
        cfg = W.dataclasses.replace(cfg, model=model)
        # the engine class: engine/engine.py imports its siblings relatively, so it must be imported as the package
        # module engine.engine (serve's default "engine.Engine" resolves to the package when the fork root is on
        # sys.path, as in the server, and fails with "module 'engine' has no attribute 'Engine'")
        cfg = W.dataclasses.replace(cfg, engine_cls=os.environ.get("PHYS_ENGINE_CLS", "engine.engine.Engine"))
        # serve's OfflineEngine rule for the scheduler's step costs: the real engine's measured ones, else the defaults
        self._real_engine = backend in ("engine", "engine-direct") and "mock" not in cfg.engine_cls.lower()
        self.async_mode = bool(vllm_config.scheduler_config.async_scheduling)
        if self.async_mode:                                     # D13: compact tables, engine.step_compact
            assert hasattr(cfg, "async_mode"), "serve/worker.py has no WorkerConfig.async_mode (D13)"
            cfg = W.dataclasses.replace(cfg, async_mode=True)
        kw = _kv_dict(os.environ.get("PHYS_ENGINE_KW", ""))     # e.g. indexer=idx2f,engram=phys,head=hsd
        if kw:
            cfg = W.dataclasses.replace(cfg, engine_kw=dict(getattr(cfg, "engine_kw", {}) or {}, **kw))
        if os.environ.get("PHYS_WORKER_WEIGHTS"):
            cfg = W.dataclasses.replace(cfg, weights=os.environ["PHYS_WORKER_WEIGHTS"])
        geom = G.Geometry(max_model_len=mc.max_model_len, spec_k=vllm_config.num_speculative_tokens,
                          max_seqs=vllm_config.scheduler_config.max_num_seqs)
        t0 = time.time()
        self.direct = backend == "engine-direct"
        self.sampling: dict = {}
        self._pending_drop: set = set()
        self._row_tok: dict = {}
        self._forced: dict = {}
        if self.direct:
            import sys

            if bridge.ROOT not in sys.path:
                sys.path.insert(0, bridge.ROOT)
            from engine.engine import Engine, EngineConfig

            ecfg = _override(EngineConfig(model=model), os.environ.get("PHYS_ENGINE_CFG", ""))
            self.w = Engine(ecfg, geom, log=logger.info)
            self.geom = self.w.geom
            self.native = False
            logger.info("phys adapter: engine-direct up in %.0f s (%s), %d paged pages per rank, buckets %s",
                        time.time() - t0, ecfg, self.geom.pools()[0].num_pages, ecfg.buckets)
            self._setup_images(vllm_config, model)
            return
        if os.environ.get("PHYS_MM", "0") == "1":
            # The vision towers and the encoder caches are created after the worker sizes the KV pools, so their
            # memory must be reserved here. Otherwise it comes out of the headroom that the mixed steps' MoE
            # workspaces need, and a large prompt arriving next to many decoding sequences runs out of memory.
            extra = (_vision_weight_bytes(model) + int(float(os.environ.get("PHYS_MM_CACHE_MB", "512")) * 2 ** 20)
                     + int(float(os.environ.get("PHYS_MM_MARGIN_MB", "1024")) * 2 ** 20))
            cfg = W.dataclasses.replace(cfg, reserve_bytes=cfg.reserve_bytes + extra)
            logger.info("phys adapter: images on: %.2f GB more left free after the KV pools (towers, encoder "
                        "cache, margin)", extra / 1e9)
        self.w = W.Worker(cfg, geom, backend, max_rows=max_rows, log=logger.info)
        self.geom = self.w.geom
        self.native = callable(getattr(self.w, "step", None))
        self._row_tok: dict = {}
        self._forced: dict = {}
        self._pending_drop: set = set()
        if not self.native:
            self.w.tb.tokens_of = self._tokens_of
        logger.info("phys adapter: worker (%s backend, %s API) up in %.0f s, %d paged pages per rank",
                    backend, "native step()" if self.native else "run_step wrapper", time.time() - t0,
                    self.geom.pools()[0].num_pages)
        self._setup_images(vllm_config, model)

    # ------------------------------------------------------------------------------------------ images
    def _setup_images(self, vllm_config, model: str) -> None:
        self.supports_images = False
        self.mm_caches: list = []
        if os.environ.get("PHYS_MM", "0") != "1":
            return
        eng = self.w if self.direct else getattr(self.w, "backend", None)
        if not getattr(eng, "supports_images", False):
            logger.warning("phys adapter: PHYS_MM=1, but the engine does not apply image rows (no supports_images): "
                           "images stay off")
            return
        if not self.direct:
            import inspect

            if not self.native or "mm" not in inspect.signature(self.w.step).parameters:
                # serve's Worker.step builds the engine's StepContext: without its `mm` argument the span rows would
                # never reach the engine, and image positions would silently run on the image token's embedding
                logger.warning("phys adapter: PHYS_MM=1, but serve's Worker.step takes no `mm`: images stay off")
                return
        import torch

        from adapter import mm as MM
        from engine import vision

        t0 = time.time()
        devs = [torch.device("cuda", r) for r in range(self.geom.n_ranks)]
        towers = [vision.VisionTower(model, d, vllm_config=vllm_config) for d in devs]
        streams = getattr(eng, "streams", None)
        self.mm_caches = [MM.EncoderCache(r, self._encoder(towers[r], streams[r] if streams else None), devs[r])
                          for r in range(len(devs))]
        self.supports_images = True
        logger.info("phys adapter: vision towers up on %d GPUs in %.0f s", len(devs), time.time() - t0)

    @staticmethod
    def _encoder(tower, stream):
        """adapter/mm.py MMItems -> one span per item, encoded on the rank's step stream (before the step's kernels)."""
        def run(items):
            return [tower.encode_one(it.data["patches"], it.data["vit_grid"], it.data["llm_grid"], it.data["types"])
                    for it in items]

        def encode(items):
            if stream is None:
                return run(items)
            import torch

            with torch.cuda.stream(stream):
                return run(items)
        return encode

    def _mm_apply(self, ps):
        """The step's image work on every rank (drop, encode, gather): ps.mm_dev[r] = (idx, emb) or None."""
        mm = getattr(ps, "mm", None)
        if mm is None or not self.mm_caches:
            return ps
        import copy

        ps = copy.copy(ps)
        ps.mm_dev = [c.apply(mm) for c in self.mm_caches]
        return ps

    def _native_step(self, ps, grammar):
        ps2 = ps if self._device_drafts() else self._fill_drafts(ps)
        mm_dev = getattr(ps, "mm_dev", None)
        if mm_dev is not None:
            return self.w.step(ps2, grammar, mm=mm_dev)
        return self.w.step(ps2, grammar)

    def _flag(self, name: str) -> bool:
        for o in (self.w, getattr(self.w, "backend", None)):
            if o is not None and getattr(o, name, None) is not None:
                return bool(getattr(o, name))
        return False

    @property
    def cost_default(self):
        """Step-cost parameters for the scheduler (bridge.set_cost_default), as serve's OfflineEngine picks them."""
        if not self._real_engine:
            return None
        from cost_model import ENGINE_MEASURED_2026_11_07
        return dict(ENGINE_MEASURED_2026_11_07)

    @property
    def last_ms(self):
        """serve's Worker: (copies, tables, ctx, engine.step) ms of the last step, if it records them."""
        return getattr(self.w, "last_ms", None)

    @property
    def device_positions(self) -> bool:
        """Async scheduling needs the engine to expand decode rows on the device (D13 step_compact + row expansion)."""
        return self._flag("device_positions") and callable(getattr(getattr(self.w, "backend", None), "step_compact", None))

    @property
    def main_thread_dispatch(self) -> bool:
        """D13: in compact / async mode serve's Worker.step ends in engine.step_compact, which
        IS dispatch: it queues this step, waits for the previous one, extracts its results into its StepHandle, and
        returns this step's handle. PhysExecutor then calls step() on the engine loop's thread and merges
        handle.result() on its pool thread."""
        return bool(self.async_mode and self.device_positions)

    @property
    def signals_prepared(self) -> bool:
        """The worker sets ps.prepared once its host tables are built (value copies), so the engine loop may plan the
        next step while the GPU runs this one."""
        return self._flag("signals_prepared")

    @property
    def sampling_caps(self):
        """What the engine's head / sampler implements: the worker's or engine's own `sampling_caps`, else
        PHYS_SAMPLING_CAPS (comma list of adapter.bridge.SAMPLING_FEATURES, 'all' = no check). Default: none of them --
        kernels/head/hsd.py supports temperature + seed only and asserts on anything else; a request
        that needs more is rejected with an error instead of failing the shared step."""
        for o in (self.w, getattr(self.w, "backend", None)):
            c = getattr(o, "sampling_caps", None) if o is not None else None
            if c is not None:
                return c
        env = os.environ.get("PHYS_SAMPLING_CAPS", "")
        if env == "all":
            return None
        return {x.strip() for x in env.split(",") if x.strip()}

    def _tokens_of(self, rid, pos: int) -> int:
        t = self._row_tok.get((rid, pos))
        if t is not None:
            return t
        f = self._forced.get(rid)
        if f is not None:
            prompt_len, forced = f
            j = pos - prompt_len
            if 0 <= j < len(forced):
                return forced[j]
        return -1

    def _device_drafts(self) -> bool:
        """True once the engine's head writes the next step's input ids (last token + drafts) on the device (contract D6).
        Until then (stub head, interim sp_layer chain) the -1 draft rows are filled on the host."""
        for o in (self.w, getattr(self.w, "backend", None)):
            if o is not None and getattr(o, "device_drafts", None) is not None:
                return bool(o.device_drafts)
        # engine/engine.py's Engine: with HSD (PHYS_ENGINE head=hsd) a decode group with any -1 row is fed from
        # hsd.next_ids on the device; an all >= 0 group counts as host-given (engine.py, D6) -- so never fill the draft
        # rows then
        return getattr(getattr(self.w, "backend", None), "hsd", None) is not None

    @staticmethod
    def _fill_drafts(ps):
        import copy

        ps2 = copy.copy(ps)
        rows = []
        for sb, toks in zip(ps.batches, ps.row_tokens):
            t = list(toks)
            for se in sb.seqs:
                if se.kind == "decode":
                    for i in range(1, se.n_rows):
                        if t[se.row0 + i] < 0:
                            t[se.row0 + i] = t[se.row0]
            rows.append(t)
        ps2.row_tokens = rows
        return ps2

    def step(self, ps, grammar):
        from interfaces import StepResult

        # a step without rows only carries frees: do not run the engine on empty batches; hand the drops to the next
        # step that has rows (rids are unique; D5: a reused uid re-initializes anyway)
        if not ps.batches or not any(sb.n_rows for sb in ps.batches):
            self._pending_drop |= set(ps.finished) | set(ps.preempted)
            return [StepResult(r, ps.step) for r in range(len(ps.batches))]
        if self._pending_drop:
            import copy

            ps = copy.copy(ps)
            ps.finished = set(ps.finished) | self._pending_drop
            self._pending_drop = set()
        ps = self._mm_apply(ps)
        if self.native:
            return self._native_step(ps, grammar)
        if self.direct:
            return self._step_direct(ps, grammar)
        if ps.new_reqs and hasattr(self.w, "on_new_requests"):
            self.w.on_new_requests(ps.new_reqs)
        gone = ps.finished | ps.preempted
        if gone and hasattr(self.w, "on_free"):
            self.w.on_free(gone)
        for rid in gone:
            self._forced.pop(rid, None)
        for rid, fr in (getattr(ps, "forced_reqs", None) or {}).items():
            self._forced[rid] = (fr.prompt_len, fr.forced)
        if not ps.batches or not any(sb.n_rows for sb in ps.batches):
            from interfaces import StepResult

            return [StepResult(r, ps.step) for r in range(len(ps.batches))]
        m = {}
        for sb, toks in zip(ps.batches, ps.row_tokens):
            for se in sb.seqs:
                for i in range(se.n_rows):
                    t = toks[se.row0 + i]
                    if t >= 0:
                        m[(se.rid, se.start + i)] = t
        self._row_tok = m
        return self.w.run_step(ps.batches)

    def _step_direct(self, ps, grammar):
        from interfaces import StepResult

        for rid in ps.finished | ps.preempted:
            self.sampling.pop(rid, None)
        for ri in ps.new_reqs:
            self.sampling[ri.rid] = ri.sampling_params
        if not ps.batches or not any(sb.n_rows for sb in ps.batches):
            return [StepResult(r, ps.step) for r in range(len(ps.batches))]
        tokens = (ps if self._device_drafts() else self._fill_drafts(ps)).row_tokens
        state = dict(sampling=self.sampling, grammar=grammar, new=ps.new_reqs, prev_tokens=ps.prev_tokens,
                     forced=getattr(ps, "forced", {}), mm=getattr(ps, "mm_dev", None))
        return self.w.step(ps.batches, tokens=tokens, sampling_state=state)

    def warmup(self) -> None:
        fn = getattr(self.w, "warmup", None) or getattr(getattr(self.w, "backend", None), "warmup", None)
        if fn:
            fn()

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        fn = getattr(self.w, "shutdown", None)
        if fn:
            fn()


def _override(cfg, spec: str):
    import dataclasses

    kw = {}
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            kw[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, tuple):
            kw[k] = tuple(int(x) for x in v.split(":"))
        else:
            kw[k] = type(cur)(v)
    return dataclasses.replace(cfg, **kw) if kw else cfg


def _kv_dict(spec: str) -> dict:
    """'a=1,b=x,c=true' -> {'a': 1, 'b': 'x', 'c': True} (ints / floats / bools parsed, else strings)."""
    out = {}
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        for t in (int, float):
            try:
                out[k] = t(v)
                break
            except ValueError:
                pass
        else:
            out[k] = v
    return out
