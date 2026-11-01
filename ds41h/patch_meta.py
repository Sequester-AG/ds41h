"""META hook: batched attention-metadata launches for decode steps. Lossless: integer metadata only,
bit-identical buffers (tests/test_drafter.py --meta, in-situ self-check DS41H_META_CHECK=N).

Patched stock vLLM 0.32.0 symbols
  * vllm.v1.attention.backends.mla.sparse_swa.DeepseekSparseSWAMetadataBuilder.build  (DS41H_META_BATCH)
      inside a batching scope, a causal decode-only build records its three launches (is_valid_token = slot_mapping
      >= 0, decode_swa_lens tail zeroing, _compute_swa_indices_and_lens_kernel) as one job and returns the same
      metadata object (views of the builder buffers); everything else runs the stock build.
  * vllm.v1.worker.gpu.model_states.default.build_attn_metadata (module global used by DefaultModelState.prepare_attn,
      i.e. the target's per-step metadata build): opens the batching scope and launches all recorded jobs in ONE
      kernel (kernels/meta.cu swa_batch_kernel) before returning. The 8 target SWA KV-cache groups cost 24 eager
      launches (~66 us at 350 W incl. launch gaps) -> 1.
  * vllm.v1.worker.gpu.spec_decode.speculator.DraftModelSpeculator._update_draft_decode_metadata  (DS41H_META_DRAFT)
      the DSpark drafter graph's per-group update (ge + token->request map + non-causal SWA indices, 3 kernels per
      draft KV-cache group) as one captured launch for all non-causal SWA groups; other groups run stock in order.
Flags: DS41H_META_BATCH (default 0 until end-to-end-validated), DS41H_META_DRAFT (default 0), DS41H_META_CHECK=N (run the
stock launches too for the first N batches and compare the buffers bitwise; logs mismatches; debug only).
"""

from __future__ import annotations

import functools
import logging
import os

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.meta")

BATCH = flags.ENABLED and flags._flag("DS41H_META_BATCH", "0")
DRAFT = flags.ENABLED and flags._flag("DS41H_META_DRAFT", "0")
CHECK = int(os.environ.get("DS41H_META_CHECK", "0") or 0)
# DS41H_META_CTL=<json file> (debug / A/B on ONE server): {"batch": 0|1} switches the target-side batching at runtime
# (re-read when the file's mtime changes). DS41H_META_SHADOW=1 (debug): every recorded job keeps a copy of its inputs
# taken at record time; the flush compares them with the inputs at flush time on the GPU (no sync) and the count of
# differing jobs is logged every 200 batches (proves the deferred launch reads exactly what the stock launch read).
CTL = os.environ.get("DS41H_META_CTL", "")
SHADOW = flags._flag("DS41H_META_SHADOW", "0")
MAXJOBS = 16

_PENDING: list | None = None  # open batching scope (list of jobs) or None
_STATS = {"batches": 0, "jobs": 0, "checked": 0, "mismatch": 0}
_CTL = {"mtime": None, "batch": 1, "shadow": int(SHADOW)}
_SHADOW = {"bad": None, "n": 0}


def _ctl_batch() -> bool:
    if not CTL:
        return True
    try:
        m = os.stat(CTL).st_mtime_ns
    except OSError:
        return True
    if m != _CTL["mtime"]:
        try:
            import json

            with open(CTL) as f:
                d = json.load(f)
            _CTL["batch"] = int(d.get("batch", 1))
            _CTL["shadow"] = int(d.get("shadow", int(SHADOW)))
            _CTL["mtime"] = m
            logger.info("ds41h meta: ctl batch=%d shadow=%d", _CTL["batch"], _CTL["shadow"])
        except (OSError, ValueError):
            return True
    return bool(_CTL["batch"])


_IN_KEYS = ("slot", "qsl", "seq_lens", "t2r", "bt", "replay")


def _shadow_record(job: dict) -> None:
    job["_in"] = {k: job[k].clone() for k in _IN_KEYS if job[k] is not None}


def _shadow_check(jobs: list[dict]) -> None:
    import torch

    if _SHADOW["bad"] is None:
        _SHADOW["bad"] = torch.zeros((), dtype=torch.int64, device=jobs[0]["slot"].device)
    for j in jobs:
        ref = j.get("_in")
        if not ref:
            continue
        diff = None
        for k, v in ref.items():
            d = (j[k] != v).any()
            diff = d if diff is None else (diff | d)
        _SHADOW["bad"] += diff.to(torch.int64)
    _SHADOW["n"] += 1
    if _SHADOW["n"] % 100 == 0:
        logger.warning("ds41h meta: SHADOW %d batches, jobs whose inputs changed between record and flush: %d",
                       _SHADOW["n"], int(_SHADOW["bad"]))


@functools.cache
def ext():
    from pathlib import Path

    from torch.utils.cpp_extension import load

    kernels = Path(__file__).parent / "kernels"
    return load(
        name="ds41h_meta",
        sources=[str(kernels / "meta.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")),
    )


# ------------------------------------------------------------------------------------------------ jobs
def make_job(slot, valid, lens, idx, qsl, seq_lens, t2r, bt, replay, ntok: int, width: int, window: int, block_size: int,
             noncausal: bool) -> dict | None:
    """A swa_batch job, or None if a tensor does not have the layout the kernel reads (caller falls back to stock)."""
    import torch

    ok = (
        slot.dtype == torch.int64 and slot.dim() == 1 and (slot.numel() <= 1 or slot.stride(0) == 1)
        and valid.dtype == torch.bool and valid.is_contiguous()
        and lens.dtype == torch.int32 and lens.is_contiguous()
        and idx.dtype == torch.int32 and idx.dim() == 3 and idx.stride(-1) == 1
        and qsl.dtype == torch.int32 and qsl.is_contiguous() and qsl.numel() >= 2
        and seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
        and t2r.dtype == torch.int32 and t2r.is_contiguous()
        and bt.dtype == torch.int32 and bt.dim() == 2 and bt.stride(1) == 1
        and (noncausal or (replay is not None and replay.dtype == torch.int32 and replay.is_contiguous()))
        and 0 < ntok <= idx.shape[0] and width <= idx.shape[-1] and ntok <= lens.numel() and ntok <= t2r.numel()
        and valid.numel() >= max(slot.numel(), ntok)
        and block_size > 0 and window > 0
    )
    if not ok:
        return None
    return dict(slot=slot, valid=valid, lens=lens, idx=idx, qsl=qsl, seq_lens=seq_lens, t2r=t2r, bt=bt, replay=replay,
                ntok=int(ntok), width=int(width), window=int(window), block_size=int(block_size), noncausal=bool(noncausal))


def _aliases(job: dict, jobs: list[dict]) -> bool:
    keys = ("valid", "lens", "idx") if not job["noncausal"] else ("valid", "lens", "idx")
    mine = {job[k].data_ptr() for k in keys}
    return any(j[k].data_ptr() in mine for j in jobs for k in keys)


def launch(jobs: list[dict]) -> None:
    """One swa_batch launch for up to MAXJOBS jobs (more: several launches, in order)."""
    for i in range(0, len(jobs), MAXJOBS):
        part = jobs[i : i + MAXJOBS]
        ext().swa_batch(
            [j["slot"] for j in part],
            [j["valid"] for j in part],
            [j["lens"] for j in part],
            [j["idx"] for j in part],
            [j["qsl"] for j in part],
            [j["seq_lens"] for j in part],
            [j["t2r"] for j in part],
            [j["bt"] for j in part],
            [j["replay"] for j in part],
            [j["ntok"] for j in part],
            [j["width"] for j in part],
            [j["window"] for j in part],
            [j["block_size"] for j in part],
            [int(j["noncausal"]) for j in part],
        )


def stock_launch(job: dict) -> None:
    """The stock launch sequence of one job (reference for tests and the in-situ check)."""
    import torch
    from vllm.v1.attention.backends.mla import sparse_swa as m
    from vllm.v1.attention.ops.metadata import compute_token_to_req_indices

    n_slot = job["slot"].numel()
    torch.ge(job["slot"], 0, out=job["valid"][:n_slot])
    if job["noncausal"]:
        compute_token_to_req_indices(job["qsl"], job["t2r"], job["ntok"], job["ntok"])
        m._COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(
            job["idx"], job["lens"], job["window"], job["width"], job["qsl"], job["seq_lens"], job["t2r"], job["valid"],
            job["bt"], job["block_size"], num_tokens=job["ntok"], token_offset=0,
        )  # fmt: skip
    else:
        job["lens"][job["ntok"]:] = 0
        m._COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
            job["idx"], job["lens"], job["window"], job["width"], job["lens"], job["lens"], job["qsl"], job["seq_lens"],
            job["t2r"], job["valid"], job["bt"], job["block_size"], job["replay"], num_tokens=job["ntok"], token_offset=0,
        )  # fmt: skip


def _outputs(job: dict) -> list:
    n = job["ntok"]
    out = [job["valid"][: max(job["slot"].numel(), n)], job["lens"] if not job["noncausal"] else job["lens"][:n], job["idx"][:n, :, : job["width"]]]
    if job["noncausal"]:
        out.append(job["t2r"][:n])
    return out


def flush(jobs: list[dict]) -> None:
    if not jobs:
        return
    if _CTL["shadow"]:
        _shadow_check(jobs)
    _STATS["batches"] += 1
    _STATS["jobs"] += len(jobs)
    if CHECK and _STATS["checked"] < CHECK:
        import torch

        if not torch.cuda.is_current_stream_capturing():
            _STATS["checked"] += 1
            for j in jobs:
                stock_launch(j)
            ref = [[t.clone() for t in _outputs(j)] for j in jobs]
            launch(jobs)
            bad = [i for i, j in enumerate(jobs) if not all(torch.equal(a, b) for a, b in zip(ref[i], _outputs(j)))]
            if bad:
                _STATS["mismatch"] += 1
                logger.error("ds41h meta: batched SWA metadata MISMATCH vs stock in jobs %s (batch %d)", bad, _STATS["batches"])
            elif _STATS["checked"] in (1, CHECK):
                logger.info("ds41h meta: batched SWA metadata == stock (check %d/%d, %d jobs)", _STATS["checked"], CHECK, len(jobs))
            return
    launch(jobs)
    if _STATS["batches"] == 1:
        logger.info("ds41h meta: first batched SWA metadata launch (%d jobs)", len(jobs))


# ------------------------------------------------------------------------------------------------ target build
def _swa_build_jobbed(self, cam, replay_start, base_build, common_prefix_len, fast_build):
    """Causal decode-only DeepseekSparseSWAMetadataBuilder.build with the three launches recorded as a job."""
    from vllm.v1.attention.backends.mla import sparse_swa as m

    pend = _PENDING
    causal = cam.causal
    if pend is None or not (causal is True) or getattr(cam, "mm_req_doc_ranges", None):
        return None
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = m.split_decodes_and_prefills(cam, decode_threshold=self.decode_threshold)
    if num_prefill_tokens != 0 or num_decode_tokens <= 0 or num_prefills != 0:
        return None
    seq_lens = cam.seq_lens
    seq_lens_cpu = cam.seq_lens_cpu_upper_bound
    query_start_loc = cam.query_start_loc
    query_start_loc_cpu = cam.query_start_loc_cpu
    block_table = cam.block_table_tensor
    slot_mapping = cam.slot_mapping
    token_to_req_indices = cam.token_to_req_indices(self.token_to_req_indices)
    if replay_start is None:
        replay_start = self.no_replay_start
    decode_swa_indices = self.decode_swa_indices
    job = make_job(slot_mapping, self.is_valid_token, self.decode_swa_lens, decode_swa_indices, query_start_loc, seq_lens,
                   token_to_req_indices, block_table, replay_start, num_decode_tokens, decode_swa_indices.shape[-1], self.window_size,
                   self.block_size, False)
    if job is None or slot_mapping.shape[0] > self.is_valid_token.shape[0]:
        return None
    if len(pend) >= MAXJOBS or _aliases(job, pend):
        flush(pend)
        pend.clear()
    if _CTL["shadow"]:
        _shadow_record(job)
    pend.append(job)
    is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
    deepseek_v4_fields = self._build_deepseek_v4_metadata(num_decodes, num_prefills, seq_lens, seq_lens_cpu, query_start_loc,
                                                          query_start_loc_cpu, replay_start)
    tile_sched = self.build_tile_scheduler(num_decode_tokens)
    return m.DeepseekSparseSWAMetadata(
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        block_table=block_table,
        slot_mapping=slot_mapping,
        is_valid_token=is_valid_token,
        token_to_req_indices=token_to_req_indices,
        decode_swa_indices=decode_swa_indices[:num_decode_tokens],
        decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
        decode_swa_width=self.window_size,
        prefill_swa_indices=None,
        prefill_swa_lens=None,
        prefill_left_visible=None,
        prefill_right_visible=None,
        replay_start=replay_start,
        block_size=self.block_size,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        max_decode_query_len=min(cam.max_query_len, self.decode_threshold),
        tile_sched_swaonly=tile_sched[m._LAYER_TYPE_SWAONLY],
        tile_sched_c4a=tile_sched[m._LAYER_TYPE_C4A],
        tile_sched_c128a=tile_sched[m._LAYER_TYPE_C128A],
        tile_sched_c1a=tile_sched[m._LAYER_TYPE_C1A],
        tile_sched_c2a=tile_sched[m._LAYER_TYPE_C2A],
        **deepseek_v4_fields,
    )


# Source fingerprint of the stock functions this module re-implements (vLLM 0.32.0). A different source disables the
# patch instead of silently diverging.
_BUILD_SHA = "39b96e5fed72c389"  # DeepseekSparseSWAMetadataBuilder.build
_SWA_UPDATE_SHA = "913db877f75f21bd"  # DeepseekSparseSWAMetadataBuilder.update_draft_decode_metadata
_SPEC_UPDATE_SHA = "bceb1384b9fc9a74"  # DraftModelSpeculator._update_draft_decode_metadata


def _sha(fn) -> str:
    import hashlib
    import inspect

    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def patch_sparse_swa(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.DeepseekSparseSWAMetadataBuilder
    if getattr(cls, "_ds41h_meta_patched", False):
        return
    base_build = cls.build
    sha = _sha(base_build)
    if sha != _BUILD_SHA:
        logger.warning("ds41h meta: stock DeepseekSparseSWAMetadataBuilder.build changed (%s); batching disabled", sha)
        return

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, replay_start=None):
        if _PENDING is not None:
            md = _swa_build_jobbed(self, common_attn_metadata, replay_start, base_build, common_prefix_len, fast_build)
            if md is not None:
                return md
        return base_build(self, common_prefix_len, common_attn_metadata, fast_build=fast_build, replay_start=replay_start)

    cls.build = build
    cls._ds41h_meta_patched = True
    mark_active("meta", f"DeepseekSparseSWAMetadataBuilder.build(batched decode; stock sha {sha})")


def patch_default_state(module) -> None:
    from ds41h.patch_core import mark_active

    base = module.build_attn_metadata
    if getattr(base, "_ds41h_meta", False):
        return

    def build_attn_metadata(*args, **kwargs):
        global _PENDING
        import torch

        if (_PENDING is not None or kwargs.get("for_cudagraph_capture", False) or torch.cuda.is_current_stream_capturing()
                or not _ctl_batch()):
            return base(*args, **kwargs)
        _PENDING = jobs = []
        try:
            out = base(*args, **kwargs)
        finally:
            _PENDING = None
        flush(jobs)
        return out

    build_attn_metadata._ds41h_meta = True
    module.build_attn_metadata = build_attn_metadata
    mark_active("meta", "model_states.default.build_attn_metadata(batching scope)")


# ------------------------------------------------------------------------------------------------ drafter update
def patch_speculator(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.DraftModelSpeculator
    base = cls._update_draft_decode_metadata
    if getattr(base, "_ds41h_meta", False):
        return
    from vllm.v1.attention.backends.mla import sparse_swa as m

    shas = (_sha(base), _sha(m.DeepseekSparseSWAMetadataBuilder.update_draft_decode_metadata))
    if shas != (_SPEC_UPDATE_SHA, _SWA_UPDATE_SHA):
        logger.warning("ds41h meta: stock draft metadata update changed %s; DSpark batching disabled", shas)
        return

    def _update_draft_decode_metadata(self, attn_metadata, num_reqs):
        if self.block_tables.cp_size > 1:
            return base(self, attn_metadata, num_reqs)
        from vllm.v1.attention.backends.mla import sparse_swa as m

        swa_update = m.DeepseekSparseSWAMetadataBuilder.update_draft_decode_metadata
        jobs: list[tuple] = []

        def run(pending):
            if pending:
                flush([j for _, _, j in pending])
                for builder, md, _ in pending:  # stock CPU tail (sparse_swa.py:890-897)
                    tile_sched = builder.build_tile_scheduler(md.num_decode_tokens)
                    md.tile_sched_swaonly = tile_sched[m._LAYER_TYPE_SWAONLY]
                    md.tile_sched_c4a = tile_sched[m._LAYER_TYPE_C4A]
                    md.tile_sched_c128a = tile_sched[m._LAYER_TYPE_C128A]
                    md.tile_sched_c1a = tile_sched[m._LAYER_TYPE_C1A]
                    md.tile_sched_c2a = tile_sched[m._LAYER_TYPE_C2A]
                    md.flashinfer_sparse_index_cache.clear()
                pending.clear()

        for groups in self.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                md = attn_metadata[group.layer_names[0]]
                job = None
                nc = getattr(builder, "decode_swa_indices_noncausal", None)
                if (
                    isinstance(builder, m.DeepseekSparseSWAMetadataBuilder)
                    and type(builder).update_draft_decode_metadata is swa_update
                    and isinstance(md, m.DeepseekSparseSWAMetadata)
                    and md.num_decode_tokens > 0
                    and nc is not None
                    and md.decode_swa_indices is not None
                    and md.decode_swa_indices.data_ptr() == nc.data_ptr()
                    and md.is_valid_token is not None
                    and md.token_to_req_indices is not None
                    and md.slot_mapping.shape[0] == md.is_valid_token.shape[0]
                ):
                    job = make_job(md.slot_mapping, md.is_valid_token, md.decode_swa_lens, md.decode_swa_indices.view(
                        md.decode_swa_indices.shape[0], 1, -1) if md.decode_swa_indices.dim() == 2 else md.decode_swa_indices,
                        md.query_start_loc, md.seq_lens, md.token_to_req_indices, md.block_table, None, md.num_decode_tokens,
                        builder.noncausal_index_width, builder.window_size, builder.block_size, True)
                    if job is not None and any(j["valid"].data_ptr() == job["valid"].data_ptr() or j["idx"].data_ptr() == job["idx"].data_ptr()
                                               or j["lens"].data_ptr() == job["lens"].data_ptr() for _, _, j in jobs):
                        run(jobs)
                if job is None:
                    run(jobs)  # keep the stock order around a group we do not batch
                    group.update_draft_decode_metadata(attn_metadata)
                else:
                    jobs.append((builder, md, job))
        run(jobs)

    _update_draft_decode_metadata._ds41h_meta = True
    cls._update_draft_decode_metadata = _update_draft_decode_metadata
    mark_active("meta", "DraftModelSpeculator._update_draft_decode_metadata(batched DSpark SWA)")


def _register(name: str, fn) -> None:
    import sys

    register(name)(fn)
    if name in sys.modules:  # imported before the hook saw it
        fn(sys.modules[name])


if BATCH and flags._flag("DS41H_PREP_FUSE", "0"):
    logger.warning("ds41h meta: DS41H_PREP_FUSE (patch_glue, per-group fusion) is set; DS41H_META_BATCH stays off (both rewrite "
                   "DeepseekSparseSWAMetadataBuilder.build)")
    BATCH = False
if BATCH:
    _register("vllm.v1.attention.backends.mla.sparse_swa", patch_sparse_swa)
    _register("vllm.v1.worker.gpu.model_states.default", patch_default_state)
if DRAFT:
    _register("vllm.v1.worker.gpu.spec_decode.speculator", patch_speculator)
