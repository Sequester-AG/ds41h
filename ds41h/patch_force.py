"""FORCE hook (debug / exactness harness): teacher-forced decoding through the REAL decode/verify path.

Enabled by DS41H=1 and DS41H_FORCE_FILE=<path> (independent of every compute component flag, so a server with
all of DS41H_{ALLREDUCE,CANDIDATES,ATTN,PROLOGUE,DECODER,SAMPLING,PREFILL,DRAFTER}=0 runs stock kernels with only
this hook on). Dumps go to DS41H_FORCE_DUMP=<dir> (default: <dir of the force file>/dump).

Force file (JSON, written atomically by scripts/exact_eval.py before each request):
    {"tag": str, "prompt_ids": [int, ...], "forced": [int, ...], "n_dump": int, "dump_dir": str (optional)}
Every worker re-reads it (on mtime change) when new requests arrive. A new request whose prompt token ids equal
"prompt_ids" becomes the forced request (all other requests are untouched):
  * the token sampled from the last prefill chunk is replaced by forced[0];
  * the DSpark drafter's proposals are replaced by the next forced tokens (DSparkSpeculator.propose output, which
    the runner stores in req_states.draft_tokens and feeds to the next verify step);
  * the verify / sample step is made to accept all of them: sampled_token_ids[i, :m] = forced[n:n+m],
    num_sampled = m (= 1 + k: k drafts + bonus = next forced token), num_rejected = 0, so postprocess_sampled,
    the scheduler output and the next proposal all see a fully accepted step;
  * SPEC_K=0 (no speculator): plain M=1 decode, the sampled token is replaced by the next forced token.
The target logits of every verified row (M = 1 + k rows per step; 1 row for the prefill step) are copied right
after the model's compute_logits / compute_logits_local returns (before any sampler code can touch them):
full-vocab rows on the stock path (all-gathered; only TP rank 0 writes them), the rank-local vocab shard on the
ds41h vocab-parallel path (DS41H_SAMPLING; every rank writes its shard). Row j of a step predicts forced[n + j].
When n_dump positions are recorded the worker writes
    <dump>/<tag>/rank{r}.pt   {"logits": [n_dump, V or V/tp], "idx": forced indices, "kind": "full"|"local",
                               "vocab_start", "rank", "tp", "rows_per_step", "input_mismatch", ...}
    <dump>/<tag>/rank{r}.json  completion marker (written last; the client waits for all ranks' markers)
The forced input ids of every decode step are checked against input_batch.input_ids (input_mismatch counts rows
whose fed token differs from the forced one: must be 0, i.e. the drafter override reached the verify step).

Numerics: the hook only reads logits and overwrites integer token ids / counts after the target forward; it adds
host syncs (timing only). Placement: instance-level wrappers installed after GPUModelRunner.load_model, so they sit
outside every class-level patch (e.g. ds41h.patch_sampling's vocab-parallel GPUModelRunner.sample).
"""

from __future__ import annotations

import json
import logging
import os
import time

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.force")

FORCE_FILE = os.environ.get("DS41H_FORCE_FILE", "").strip()
FORCE_DUMP = os.environ.get("DS41H_FORCE_DUMP", "").strip() or (os.path.join(os.path.dirname(FORCE_FILE), "dump") if FORCE_FILE else "")
ENABLED = flags.ENABLED and bool(FORCE_FILE)


class _Spec:
    """Cached contents of the force file (re-read when its mtime/size changes)."""

    def __init__(self) -> None:
        self.key = None
        self.data = None

    def get(self):
        try:
            st = os.stat(FORCE_FILE)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        if key != self.key:
            try:
                with open(FORCE_FILE) as f:
                    data = json.load(f)
                data["prompt_ids"] = [int(t) for t in data["prompt_ids"]]
                data["forced"] = [int(t) for t in data["forced"]]
                data["n_dump"] = int(data.get("n_dump", len(data["forced"])))
                self.data, self.key = data, key
            except Exception as e:  # noqa: BLE001 - a half-written file is retried on the next request
                logger.warning("ds41h force: cannot read %s: %r", FORCE_FILE, e)
                return None
        return self.data


_SPEC = _Spec()


class _Req:
    """Per forced request state (one per worker process)."""

    def __init__(self, req_id: str, spec: dict, prompt_len: int, rank: int, tp: int) -> None:
        self.req_id = req_id
        self.tag = str(spec["tag"])
        self.dump_dir = str(spec.get("dump_dir") or FORCE_DUMP)
        self.forced = list(spec["forced"])
        self.n_dump = min(int(spec["n_dump"]), len(self.forced))
        self.prompt_len = prompt_len
        self.rank, self.tp = rank, tp
        self.n = 0  # forced tokens emitted so far (row j of the next sampling step predicts forced[n + j])
        self.rows = []  # (kind, CPU tensor [m, V or V/tp]) per step (only rows with forced index < n_dump)
        self.idx = []  # forced index of every recorded row
        self.rows_per_step = []
        self.kind = None
        self.vocab_start = 0
        self.input_mismatch = 0
        self.checked_rows = 0
        self.overrides = 0
        self.flushed = False
        self.stopped = False  # ran out of forced tokens: the model samples freely from here on
        self.t0 = time.time()

    def out_dir(self) -> str:
        return os.path.join(self.dump_dir, self.tag)

    def flush(self, partial: bool = False) -> None:
        if self.flushed:
            return
        self.flushed = True
        import torch

        d = self.out_dir()
        os.makedirs(d, exist_ok=True)
        # The prefill step (no drafts) samples on the stock path (full vocab) while verify steps may use the
        # vocab-parallel path (local shard): store everything as this rank's shard if any row is local.
        widths = {t.shape[-1] for k, t in self.rows if k == "local"}
        if widths:
            w = widths.pop()
            self.kind, self.vocab_start = "local", self.rank * w
            tensors = [t if k == "local" else t[:, self.rank * w : (self.rank + 1) * w] for k, t in self.rows]
        else:
            self.kind, self.vocab_start = "full", 0
            tensors = [t for _, t in self.rows]
        save = self.kind == "local" or self.rank == 0
        meta = {
            "tag": self.tag,
            "rank": self.rank,
            "tp": self.tp,
            "kind": self.kind,
            "vocab_start": self.vocab_start,
            "n": len(self.idx),
            "n_dump": self.n_dump,
            "prompt_len": self.prompt_len,
            "rows_per_step": self.rows_per_step,
            "input_mismatch": self.input_mismatch,
            "checked_rows": self.checked_rows,
            "overrides": self.overrides,
            "partial": partial,
            "saved": False,
            "seconds": round(time.time() - self.t0, 3),
        }
        try:
            if save and self.rows:
                logits = torch.cat(tensors, 0)
                meta["shape"] = list(logits.shape)
                meta["dtype"] = str(logits.dtype)
                tmp = os.path.join(d, f".rank{self.rank}.pt.tmp")
                torch.save({**meta, "logits": logits, "idx": torch.tensor(self.idx, dtype=torch.int64)}, tmp)
                os.replace(tmp, os.path.join(d, f"rank{self.rank}.pt"))
                meta["saved"] = True
        except Exception as e:  # noqa: BLE001
            meta["error"] = repr(e)
            logger.exception("ds41h force: dump of %s failed", self.tag)
        tmp = os.path.join(d, f".rank{self.rank}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(d, f"rank{self.rank}.json"))
        self.rows = []
        logger.info("ds41h force: %s rank %d wrote %d rows (%s, partial=%s, input_mismatch=%d, steps %s)", self.tag, self.rank, meta["n"], self.kind,
                    partial, self.input_mismatch, _hist(self.rows_per_step))


def _hist(xs) -> str:
    out: dict[int, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return " ".join(f"{k}x{v}" for k, v in sorted(out.items()))


def _tp() -> tuple[int, int]:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        return int(g.rank_in_group), int(g.world_size)
    except Exception:  # noqa: BLE001
        return 0, 1


def _install(runner) -> None:
    """Instance-level wrappers (outside every class-level patch)."""
    import torch

    if getattr(runner, "_ds41h_force_installed", False):
        return
    runner._ds41h_force_installed = True
    reqs: dict[str, _Req] = {}
    runner._ds41h_force_reqs = reqs
    runner._ds41h_force_cap = None
    rank, tp = _tp()

    # ---- logits capture (stock compute_logits = full vocab; compute_logits_local = this rank's vocab shard)
    model = runner.model
    for name, kind in (("compute_logits", "full"), ("compute_logits_local", "local")):
        base = getattr(model, name, None)
        if base is None:
            continue

        def make(base=base, kind=kind):
            def wrapped(hidden_states, *args, **kwargs):
                out = base(hidden_states, *args, **kwargs)
                cap = runner._ds41h_force_cap
                if cap is not None and out is not None and "rows" not in cap:
                    # copy the forced requests' rows now: samplers may modify logits in place later
                    cap["kind"] = kind
                    cap["width"] = int(out.shape[-1])
                    cap["rows"] = {i: out[lo:hi].detach().clone() for i, lo, hi in cap["want"]}
                return out

            return wrapped

        setattr(model, name, make())

    # ---- request registration (after the stock add_requests: it calls _remove_request(req_id) first)
    base_add = runner.add_requests

    def add_requests(scheduler_output):
        base_add(scheduler_output)
        new = getattr(scheduler_output, "scheduled_new_reqs", None) or []
        if not new:
            return
        spec = _SPEC.get()
        if spec is None:
            return
        for nr in new:
            ids = nr.prompt_token_ids
            if ids is None and nr.prefill_token_ids is not None:
                ids = nr.prefill_token_ids
            if ids is None or len(ids) != len(spec["prompt_ids"]) or list(ids) != spec["prompt_ids"]:
                continue
            reqs[nr.req_id] = _Req(nr.req_id, spec, len(ids), rank, tp)
            logger.info("ds41h force: request %s matched tag %s (prompt %d tokens, %d forced, dump %d) rank %d", nr.req_id, spec["tag"], len(ids),
                        len(spec["forced"]), spec["n_dump"], rank)

    runner.add_requests = add_requests

    base_remove = runner._remove_request

    def _remove_request(req_id):
        st = reqs.pop(req_id, None)
        if st is not None and not st.flushed:
            st.flush(partial=True)
        return base_remove(req_id)

    runner._remove_request = _remove_request

    # ---- sampling: capture rows, then force the outcome
    base_sample = runner.sample

    def sample(hidden_states, input_batch, grammar_output):
        live = []
        if reqs:
            n_req = int(input_batch.num_reqs)
            for i, rid in enumerate(list(input_batch.req_ids)[:n_req]):
                st = reqs.get(rid)
                if st is not None and not st.stopped:
                    live.append((i, st))
        if not live:
            return base_sample(hidden_states, input_batch, grammar_output)
        if getattr(runner, "batch_sharder", None) is not None:
            logger.warning("ds41h force: batch-sharded sampling is not supported; request not forced")
            for _, st in live:
                st.stopped = True
            return base_sample(hidden_states, input_batch, grammar_output)
        cu = input_batch.cu_num_logits_np
        runner._ds41h_force_cap = cap = {"want": [(i, int(cu[i]), int(cu[i + 1])) for i, _ in live]}
        try:
            out, num_sampled, num_rejected = base_sample(hidden_states, input_batch, grammar_output)
        finally:
            runner._ds41h_force_cap = None
        ns = num_sampled.to("cpu", non_blocking=False)
        qsl = getattr(input_batch, "query_start_loc_np", None)
        for i, st in live:
            m = int(cu[i + 1]) - int(cu[i])
            if int(ns[i]) == 0 or m <= 0:  # chunked prefill (not the last chunk): nothing sampled
                continue
            toks = st.forced[st.n : st.n + m]
            if len(toks) < m:
                st.stopped = True
                logger.info("ds41h force: %s out of forced tokens at n=%d (m=%d): sampling freely", st.tag, st.n, m)
                if not st.flushed:
                    st.flush(partial=len(st.idx) < st.n_dump)
                continue
            # the fed tokens of a decode/verify step must be [forced[n-1], forced[n .. n+m-2]]
            if st.n > 0 and qsl is not None:
                q0, q1 = int(qsl[i]), int(qsl[i + 1])
                if q1 - q0 == m:
                    fed = input_batch.input_ids[q0:q1].to("cpu").tolist()
                    exp = st.forced[st.n - 1 : st.n - 1 + m]
                    st.input_mismatch += sum(int(a != b) for a, b in zip(fed, exp))
                    st.checked_rows += m
            rows = cap.get("rows", {}).get(i)
            if rows is not None and len(st.idx) < st.n_dump:
                keep = min(m, st.n_dump - len(st.idx))
                st.rows.append((cap.get("kind"), rows[:keep].to("cpu")))
                st.idx.extend(range(st.n, st.n + keep))
            elif rows is None:
                logger.warning("ds41h force: %s step n=%d: logits were not captured", st.tag, st.n)
            st.rows_per_step.append(m)
            dev = out.sampled_token_ids.device
            out.sampled_token_ids[i, :m] = torch.tensor(toks, dtype=out.sampled_token_ids.dtype, device=dev)
            num_sampled[i] = m
            num_rejected[i] = 0
            st.overrides += 1
            st.n += m
            if len(st.idx) >= st.n_dump and not st.flushed:
                st.flush()
        return out, num_sampled, num_rejected

    runner.sample = sample

    # ---- drafter: propose exactly the next forced tokens
    spec_obj = getattr(runner, "speculator", None)
    if spec_obj is not None and hasattr(spec_obj, "propose"):
        base_propose = spec_obj.propose

        def propose(*args, **kwargs):
            draft = base_propose(*args, **kwargs)
            if not reqs or draft is None:
                return draft
            input_batch = args[0] if args else kwargs.get("input_batch")
            if input_batch is None:
                return draft
            n_req = int(input_batch.num_reqs)
            cloned = False
            for i, rid in enumerate(list(input_batch.req_ids)[:n_req]):
                st = reqs.get(rid)
                if st is None or st.stopped or st.n == 0:
                    continue
                k = int(draft.shape[1])
                toks = st.forced[st.n : st.n + k]
                if len(toks) < k:
                    toks = toks + [st.forced[-1]] * (k - len(toks))  # past the end: request stops before using them
                if not cloned:
                    draft = draft.clone()
                    cloned = True
                draft[i, :k] = torch.tensor(toks, dtype=draft.dtype, device=draft.device)
            return draft

        spec_obj.propose = propose
    logger.info("ds41h force hook installed on rank %d/%d (file %s, dump %s, speculator %s)", rank, tp, FORCE_FILE, FORCE_DUMP,
                type(spec_obj).__name__ if spec_obj is not None else None)


def patch_model_runner(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.GPUModelRunner
    base_load = cls.load_model

    def load_model(self, *args, **kwargs):
        result = base_load(self, *args, **kwargs)
        try:
            _install(self)
        except Exception:  # noqa: BLE001 - never break model loading
            logger.exception("ds41h force: hook installation failed")
        return result

    cls.load_model = load_model
    mark_active("force", "GPUModelRunner.load_model(force hook)")


if ENABLED:
    register("vllm.v1.worker.gpu.model_runner")(patch_model_runner)
