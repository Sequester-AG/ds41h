/* engine_handover.h
 * slice 2: Engram's native prepare views per rank + engram_lib + the Engram
 * set state (engram_gen, appended to FprEhSetState); the page-copy id staging per rank (B7); slice 3: the span
 * table per mixed view and rank (checklist C2: every driver piece handle built at ENSURE, core2b/span_program.h's input)
 * + the engine's span kind ids; slice 4: FprEhSpanLayer + moe_rsf / attn_b_pf
 * Engram guard / nring, per mixed view: T_STAGE's key table, the FlashMLA plan slots (NEW_SCHEDS = fpr_attn_reset_plans), idxa_gst + set
 * the moe decide bucket; per rank: CP index staging + the CP mail stream; engine: the libraries
 * table, the moe decide snapshot, the combine templates; slice 5: tables_abi (the ITables
 * vtable of tb_tail / tb_d13's creator: save_state / restore_state only when it matches); slice 5b: plan_slots is
 * const int64_t*; slice 6: span_kinds_cp[8] (the CP span pieces' kinds).
 * Every field is zero until exported: nd
 * refuses the pieces that need it.
 * engine_handover.h -- DRAFT v0.6: ADDITIVE over v0.5 (struct
 * layouts grow; both sides rebuild together). GeneralTables + idx2f_tab hid +
 * OwnedIndexer; FprEhMixedPf: attn_pf Plan tables; the rank's idx2f_tab ring / EagerWorkspace / pf_inv / pf_ws) + the
 * native step's persistent per-step buffers (per mixed view: vidx_d / vidx / nv / nd + vidx_up; per rank: tail_ix +
 * tail_ix_up).
 * v0.6 refusal rules: a native mixed step needs idx.tab_hid != 0 for both
 * buckets on every rank (and the other new handles non-zero for the ops that use them: vidx_d.ptr for SET_VIDX, tail_ix.data
 * for T_STAGE / carry, pf.blob.ptr + pf_ws.data for PF_PLAN, idx_ring_n > 0 for UPLOAD_GENERAL); else the step runs
 * embedded. An exporter / nd FPR_EH_VERSION or struct_size mismatch refuses at INIT (never a misread).
 * Ownership (v0.6): the mixed buckets' step-input set indices (s["in_i"], FprEhSets alternation) stay PYTHON-owned under
 * ENG_ND_T (nd's upload_inputs reads and flips them through the engine); the executor owning them would add a set-state
 * field (not in v0.6).
 * engine_handover.h
 * Consumers: fpr/core/src/core2b, nd.
 * Rules
 *  - The engine OWNS everything here: every pointer is a view into tensors / pinned buffers / events / graphs the Python
 *    Engine keeps alive. Nothing here transfers ownership; the C++ never frees.
 *  - Anything the engine can evict or re-carve carries a GENERATION, checked before EVERY use (re-fetch on mismatch).
 *  - Graph execs: append-only table + version. Published execs are RETIRED, never freed (F).
 *  - Per-step tensors (vidx / nv / nd, staging contents) are not handover: they are per-step params.
 *  - Streams and events are explicit; the C++ never relies on a "current stream".
 *  - C ABI: struct_size + version first, fixed-width types, (ptr, count) arrays, no C++ types.
 * INVALIDATION RULE (a), single owner. The engine invalidates (evicts / re-carves / frees) memory behind
 * a handed-over view ONLY at one of these points, both on the loop thread with no native step in flight:
 *   (i)  inside ensure_bucket(): only the loop thread calls it, at a step boundary, with nothing enqueued;
 *   (ii) inside a step of the embedded Python engine (2b-α: mixed / synchronous-fallback / grammar steps). The loop runs
 *        these only after nd::drain(): every inflight step extracted AND every rank's step / extraction streams
 *        synchronized.
 * The generation check only detects that (i) / (ii) invalidated a view cached earlier. MIXED-step buckets are never
 * handed over in 2b-α.
 * RULE (b), v0.5: mixed A / T buckets ARE handed over (FprEhMixedBucketRank). Every create /
 * evict of a mixed bucket (today inside Engine.bucket() during step_mixed, incl. the mixed-round churn and its
 * empty_cache) happens only in ensure_mixed() (point (i): loop thread, step boundary, nothing of nd in flight, via
 * run_python) or inside an embedded step (point (ii)). It bumps that bucket's generation AND buckets_epoch. nd re-fetches
 * the mixed views of a native mixed step whenever buckets_epoch moved since its last fill (cheap), and checks
 * fpr_eh_current() on each view before building the step's plans.
 * fill_mixed then returns the per-R view of the pinned bucket. Whatever caches a GPU program over
 * a bucket (the driver's programs and piece handles, H1 / H2 and T graphs, agrs's bucket set, the moe Args) keys it by
 * (bucket, generation) and drops it on the eviction notice BEFORE the free / empty_cache; fpr_driver_run also asserts the
 * plan's generation. The device half of the A -> T tail gather is FPR's driver op FPR_OP_TAIL_GATHER (no handover view);
 * native page copies use pools[] through FPR's fpr_pagecopy C ABI.
 *  - the eviction's empty_cache + all-device sync stay inside ensure_mixed (the boundary);
 *  - the generation bumps on create, evict AND onebk re-carve (a new max-A base invalidates every view over it); onebk
 *    view creation goes through ensure_mixed too (cheap, no eviction);
 *  - ZERO lazily added per-state tensors during a native step: every pointer the pieces use exists at ensure time
 * STORAGE NEVER MOVES: gens[] and the graph entry array are allocated once with a fixed capacity, never reallocated; an
 * overflow is an error.
 * SET HANDOFF (C): in 2b-α the embedded engine's steps use the SAME pinned staging sets, events and HSD state as nd.
 *   - Around every embedded step: nd::drain(); push_state() (nd's set-alternation indices -> the engine); the embedded
 *     step; pull_state() (the engine's indices -> nd).
 *   - At each handoff, nd invalidates HsdPrep's per-slot temp / seed / key caches: the two sides key slots differently
 *     (id(obj) vs request serial), and a rewrite produces the same bytes.
 */
#ifndef FPR_ENGINE_HANDOVER_H
#define FPR_ENGINE_HANDOVER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define FPR_EH_VERSION 7          /* 0 = v0 .. 7 = v0.7 (drafts); first reviewed version = 10 */
#define FPR_EH_IDX_RING_MAX 4     /* v0.6: tables2f.Stage RING slots per device (idx2f_tab ring) */
#define FPR_EH_MAX_LAYERS 64      /* v0.7: layers per span table (the model has 40) */
#define FPR_EH_ENGRAM_GUARD_FIELDS 1   /* v0.7 slice 4: FprEhEngramRank.guard / nring (were pads; layout unchanged) */
#define FPR_EH_MAX_LIBS 16        /* v0.7 slice 4: loaded C ABI libraries by name */
#define FPR_EH_MAX_POOLS 8        /* KV pools per rank (paged, window, state, index-key mirror, ...) */
#define FPR_EH_MAX_RANKS 8
#define FPR_EH_KEY 96             /* canonical text of a Python cache key, e.g. "320|LA|M|2|ab" (NUL-terminated) */

/* ------------------------------------------------------------------ views */
typedef enum { FPR_EH_U8 = 0, FPR_EH_I32 = 1, FPR_EH_I64 = 2, FPR_EH_F32 = 3, FPR_EH_BF16 = 4, FPR_EH_F16 = 5 } FprEhDtype;
typedef struct {                  /* a tensor view the engine owns: device or pinned host */
  void* data;
  int32_t dtype;                  /* FprEhDtype */
  int32_t ndim;
  int64_t shape[4];
  int64_t stride[4];              /* in elements */
  int32_t device;                 /* CUDA device, -1 = pinned host */
  int32_t pad_;
} FprEhTensor;
typedef struct { void* ptr; int64_t bytes; int32_t device; int32_t pad_; } FprEhBuf;   /* raw bytes (blobs) */
/* a double-buffered staging set: 2 pinned host sets + their persistent events (B). Reuse rule: wait ev[i] (the set's
 * previous H2D), fill h[i], enqueue the copy, re-record ev[i]. The engine owns the events */
typedef struct { FprEhBuf h[2]; void* ev[2]; } FprEhSets;

/* ------------------------------------------------------------------ per RANK: fixed for the engine's lifetime (A) */
typedef struct {
  int32_t struct_size;
  int32_t rank, device;
  int32_t pad_;
  void* step_stream;              /* cudaStream_t: Engine.streams[r]; every NativeDecode row is issued here */
  void* xstream;                  /* extraction stream (hsd.result D2H) */
  void* side_stream;              /* side[r]: only where the engine uses it today */
  void* estream;                  /* eng.estream[r]: Engram's token-only side work (its per-step prepare) (L) */
  void* step_event[2];            /* _record (row 17), alternating */
  /* rowexp (rows 6 / 8 / 16): descriptor blobs, ring state, bound start / seq outputs, limits */
  FprEhBuf rx_b32_d, rx_b64_d;
  FprEhSets rx_h32, rx_h64;       /* the h32 / h64 pinned sets share one alternation index (rx_hi) */
  int64_t rx_o32[8];
  FprEhTensor rx_ring, rx_dbt, rx_nc, rx_err, rx_counts, rx_start, rx_seq;
  int64_t rx_scratch_slot, rx_S_max, rx_R_max, rx_ops_max;
  /* row 3: D12 deltas: THREE buffers per set (K): d13.dl_d[r] = (rows int64, cols int64, pages int32), DELTA_CAP each;
   * dl_h[i][r] the same three pinned. The three FprEhSets share the GLOBAL dl_i */
  FprEhTensor dl_rows_d, dl_cols_d, dl_pages_d;
  FprEhSets dl_rows, dl_cols, dl_pages;
  int64_t delta_cap;
  /* row 7: guard */
  FprEhBuf nb_d;
  FprEhSets nb;
  /* row 9: positions D2H targets d2h[i] = (ph, oh), alternating with d2h_i */
  FprEhTensor d2h_pos[2], d2h_own[2];
  /* row 16: advance */
  FprEhBuf qidx_d;
  FprEhSets qidx;
  FprEhTensor num_sampled;        /* hsd.st[r].num_sampled */
  /* row 10: Engram (pk + stage sets + _pk_off; b_start; b_ids binding) */
  FprEhBuf engram_pk_d;
  FprEhSets engram_stage;
  int64_t engram_pk_off[8];
  FprEhTensor engram_b_start, engram_b_ids;
  /* rows 14 / 15: HSD blobs (device + 2 pinned sets). Bitmask rows: not in v1 (H: grammar steps have no graph H ->
   * embedded fallback in α) */
  FprEhBuf hsd_dblob;
  FprEhSets hsd_hblob;
  /* §2 extraction: D13's output tensors exactly as head_result reads them (dtype / shape), + pinned D2H targets.
   * hsd.result's RES_OWNER D2H uses its own buffers today */
  FprEhTensor next_ids, num_emitted, sampled;
  FprEhTensor next_ids_h, num_emitted_h, sampled_h;
  /* TFUSE / T' device-token index (H._dt_d); the indexer's dbt rows (pslot fill) */
  FprEhTensor dt_d, idx_dbt;
  /* v0.5 (2): the H1 mixed-step head. Per rank, the replay is (a) h_own[:n_s] =
   * T.h[sample_rows[:n_s]] (an index_select into hsd.h_own[r]: h1_h_own; nd owns the small index staging), (b)
   * drafter_aux[:n_t] = T.aux[:n_t] (a D2D copy from row 0, skipped when src == dst; the destination is per (rank, D_pad):
   * FprEhGraph.aux_dst of the H1 entry), (c) cudaGraphLaunch of the key's exec */
  FprEhTensor h1_h_own;
  /* v0.5: rowexp's internal outputs that D13._fallback binds before begin */
  FprEhTensor rx_own_pos, rx_own_own, rx_own_si, rx_own_sl, rx_own_paged;
  int64_t idx_ring_id;            /* tables2f.Stage.eid */
  int32_t idx_ring_n;             /* <= FPR_EH_IDX_RING_MAX */
  int32_t pad6_;
  int64_t idx_ring_cap;           /* int32 words per slot (sized at init for S_max x max pages: no _grow mid-step) */
  void* idx_ring_host[FPR_EH_IDX_RING_MAX];   /* pinned int32 blocks (Stage.hptr) */
  void* idx_ring_dev[FPR_EH_IDX_RING_MAX];    /* device int32 blocks (Stage.dptr) */
  FprEhTensor ews_f32, ews_bmax, ews_prep;    /* EagerWorkspace.of(dev, 256 MB) */
  FprEhTensor pf_inv;             /* i32 [n_pool_pages + 1] (attn_prefill.Plan.inv) */
  FprEhTensor pf_ws;              /* Engine.pf_ws[r] (ENG_PF_WS_PREALLOC_MB); data NULL = per-step workspace: p1::pf_host refuses */
  /* v0.6: the mixed step's tail gather index, ONE device copy per step shared by FPR_OP_TAIL_GATHER (#7) and
   * p1 carry_issue: plan.T[r].stage.ix (A rows) staged through tail_ix_up (FprEhSets rule) */
  FprEhTensor tail_ix;            /* i64 [max_tail_rows]; data NULL = not allocated (flag
                                     off): nd refuses the T stage */
  FprEhSets tail_ix_up;           /* 2 pinned sets of [max_tail_rows] i64 */
  /* v0.5 (3): the rank's KV pools for native page copies (Worker.copies / FPR_PAGECOPY's C ABI): CopyOp.pool indexes
   * pools[] in the engine's POOLS order (kv_manager); pool memory is allocated once at init and never moves */
  int32_t n_pools;
  int32_t pad5_;
  struct { char name[16]; void* base; int64_t page_bytes, n_pages; } pools[FPR_EH_MAX_POOLS];
} FprEhRankState;

/* ------------------------------------------------------------------ per (decode BUCKET R, rank): generation-checked (A) */
typedef enum { FPR_EH_ROLE_DECODE = 0, FPR_EH_ROLE_MIXED = 1 } FprEhRole;
typedef struct {
  int32_t struct_size;
  int32_t rank;
  /* the bucket's identity: a re-fetch fills the view of exactly this bucket, never another */
  int64_t R;                      /* the bucket's rows */
  int32_t role;                   /* FprEhRole (2b-α: DECODE only) */
  int32_t moe_kind;               /* the bucket's MoE variant as the engine's enum; 0 = default */
  uint64_t generation;            /* gens[gen_slot] when this view was filled */
  int32_t gen_slot;               /* index into FprEngineHandover.gens */
  int32_t pad_;
  /* row 4: step inputs blob (s["in_d"] + 2 pinned sets + events, _in_views offsets in bytes) */
  FprEhBuf in_d;
  FprEhSets in;
  int64_t in_off_tok, in_off_valid, in_off_samp;
  /* row 5: indexer blob (up_d + sets + events, _up_views offsets, B) */
  FprEhBuf up_d;
  FprEhSets up;
  int64_t up_off_valid, up_off_pslot, up_off_rs, up_B;
  /* step views (pos / own / si / sl are views of in_d), the paged table, tokens */
  FprEhTensor pos, own, si, sl, paged, tok;
  /* indexer per-bucket tensors */
  FprEhTensor idx_rslot, idx_valid, idx_seq_lens[2], idx_cslot[2];
  /* owner path bases (tensors, A) */
  FprEhTensor own_rbase, own_sbase;
  void* pos_ev[2];                /* the positions D2H event per set (d13_ev), persistent (B) */
} FprEhBucketRank;

/* ------------------------------------------------------------------ v0.6: the mixed bucket's indexer / attn_pf views */
/* tables2f.GeneralTables of one (mixed bucket, rank), A and T: created at ensure time with Bc = ceil64(R / 4
 * + S_max) and never regrown natively (p1::gt_host refuses B > Bc; an embedded regrowth re-creates the tensors -> the
 * bucket's generation moves -> nd re-fetches). tab_hid 0 = not exported: nd refuses native mixed steps */
typedef struct {
  int64_t Bc;                     /* pseudo-request capacity */
  int64_t tab_hid;                /* idx2f_tab registration (create(..., general=1)) of the 16 tensors below */
  FprEhTensor pos, dbt, sl1, sl2, cs1, cs2, rs, valid_ir, t2r, qsl, ir2row_c, vir, vrow, row2ir_d, nb_lens, bt;
  FprEhTensor t2r_ir;             /* i32 [4 Bc] */
  FprEhTensor lat;                /* bf16 [R, 512]: the compressor latent */
  FprEhTensor cand, topk, slow;   /* the base OwnedIndexer: cand i32 [4 Bc, 2048], the shared top-512 [4 Bc, 512], slow [1] */
} FprEhMixedIdx;
/* bucket A only (ENG_ATTN_PF): attn_prefill.Plan's device tables, persistent for native (per step in Python) */
typedef struct {
  int64_t S_max, bt_cols_max, wbt_cols_max;   /* S_max x max_model_len / 64, S_max x (max_model_len / 32 + 1) */
  FprEhBuf blob;                  /* device i32: seq_lens [S_max] | gather_lens [S_max] | qsl [S_max + 1] */
  FprEhSets up;                   /* pinned sets for blob + bt (FprEhSets rule) */
  FprEhTensor bt, wbt;            /* i32 [S_max, bt_cols_max], [S_max, wbt_cols_max] */
  FprEhTensor loc;                /* i32 [R_A, 512] (to_local) */
  FprEhTensor comb640, clen640, comb128, clen128;   /* combine_topk_swa_indices per index width */
} FprEhMixedPf;

/* v0.7 slice 3 (checklist C1 / C2): one layer's driver piece handles of a mixed view on one rank, built at ENSURE time
 * (mixed_cxx's lazily cached handles today: drv.build_bd / build_router / build_ag / build_rs / build_final + the owners'
 * _handles[kind]); -1 = absent (the span program refuses the segment: embedded). = core2b::SpanLayerHandles */
typedef struct {
  int32_t layer, pad_;
  int64_t bd0, engram_bd, attn_a, idx, attn_b, bd1;
  int64_t router, router_ce, ag, moe, rs, rs_fl;   /* router_ce: the AGCE variant; rs_fl: the agrs_fl_rsf variant */
  int64_t moe_rsf;                /* (slice 4) the moe piece's PROD_RSF handle (moe = the plain one): set_step's variant picks */
  int64_t attn_b_pf;              /* (slice 4) A views under ENG_ATTN_PF: attn_b of a rank with prefill rows (ws["pf_ab"]) */
} FprEhSpanLayer;
typedef struct {                  /* = core2b::SpanRankHandles + the state's static flags */
  int32_t built;                  /* 1 = every handle of the view built at ensure (0 = not exported: refuse) */
  int32_t n_layers;               /* the bucket role's segment layers (A: 0..20, T: 21..39) */
  int64_t embed, final_;          /* -1 = absent (embed: A only; final: T only) */
  uint8_t fl_rsf;                 /* the state has agrs_fl_rsf (rs takes rs_fl) */
  uint8_t pad2_[7];
  FprEhSpanLayer layers[FPR_EH_MAX_LAYERS];
} FprEhSpanTable;

/* ------------------------------------------------------------------ v0.5 (1): per (MIXED bucket, rank), rule (b) */
typedef enum { FPR_EH_MIXED_A = 0, FPR_EH_MIXED_T = 1 } FprEhMixedRole;   /* Engine.bucket(R, moe, "A" | "T") */
typedef struct { int64_t R; int32_t role; int32_t pad_; char key[FPR_EH_KEY]; } FprEhMixedKey;   /* ensure_mixed's set */
typedef struct {
  int32_t struct_size;
  int32_t rank;
  int64_t R;                      /* rows per rank of the bucket */
  int32_t role;                   /* FprEhMixedRole */
  int32_t moe_kind;               /* the engine's MoE kind of the bucket ("prod" = 0); the key text is authoritative */
  uint64_t generation;            /* gens[gen_slot] when this view was filled */
  int32_t gen_slot;
  int32_t pad_;
  char key[FPR_EH_KEY];           /* canonical bucket key, e.g. "704|prod|A" */
  /* the bucket state as the stage-1 driver sees it (mixed_plan.h FprBucketState): pointers by FprStateField id + sizes.
   * mixed_fields.h is generated from Engine._alloc's key set (onebk's s["_alloc_keys"]), nested keys flattened
   * (ws.q, caches.20.si, ti.21 / tl.21, ...), one line per field (name, dtype, shape rule in R / W, per role), stable
   * order; fields_hash = md5 (first 8 bytes) over those lines. The engine recomputes it from its live _alloc and asserts
   * it equals the compiled header's before exporting; nd refuses a view whose fields_hash is not its own */
  uint64_t fields_hash;
  int32_t n_ptrs;
  int32_t pad2_;
  const void* const* ptrs;
  const int64_t* bytes;
  /* the step-input upload of this bucket (Engine._upload: in_d + 2 pinned sets + events, offsets) */
  FprEhBuf in_d;
  FprEhSets in;
  int64_t in_off_tok, in_off_valid, in_off_samp;
  /* T buckets: the H1 head's sources: h (s["h"], gathered by sample_rows into FprEhRankState.h1_h_own) and aux (copied
   * rows 0 .. n_t - 1 into the H1 entry's aux_dst) */
  FprEhTensor h, aux;
  /* si / sl for p1::pf_issue are views
   * of in_d at p0::in_layout(R).off[IN_SI / IN_SL] (inputs_build.h: the one layout source), not separate fields */
  FprEhMixedIdx idx;
  FprEhMixedPf pf;
  /* ONE device blob per (bucket, rank) in that view's own R layout, one H2D per step
   *   vidx i64 [W * R] at 0 | nv i32 [W] at 8 W R | nd i32 [W] at 8 W R + ceil8(4 W)   (the step's values: vidx[:vcount])
   * the same values on every rank (p0::build_vidx). vidx_d.ptr NULL = not allocated (flag off): nd refuses that piece */
  FprEhBuf vidx_d;
  FprEhTensor vidx;               /* view: i64 [W * R] */
  FprEhTensor nv, nd;             /* views: i32 [W] (nd: PF_ACT8 only) */
  FprEhSets vidx_up;              /* 2 pinned sets of vidx_d's layout */
  /* v0.7 slice 3 (C2): the view's span table (ensure-time handles); zero = not built */
  FprEhSpanTable span;
  /* v0.7 slice 4 */
  const void* tkeys;              /* FPR's FprTgKeys of this view (kernels/tail_gather/tail_gather_cabi.h; engine-owned bytes,
                                     s["_tg_kb"][1], valid for the view's generation): T_STAGE = fpr_t_stage_run(&A.tkeys,
                                     &T.tkeys, tail_ix, n, l20, clear); NULL = not exported: nd refuses T_STAGE */
  int64_t tkeys_bytes;            /* sizeof(FprTgKeys) as the engine built it (the ABI drift guard) */
  const int64_t* plan_slots;
  int32_t n_plan_slots, pad4_;
  FprEhTensor idxa_gst;           /* FprIdxAStep.gst's device int32 [cap] (Python: s["_cxx_idxa_gst"]) */
  FprEhSets idxa_gst_up;
  const void* moe_bucket;         /* p0::MoeBucket of this view (mixed_cxx_moe_ensure.bucket_of(s)); NULL = refuse */
  int64_t moe_bucket_bytes;
} FprEhMixedBucketRank;

/* ------------------------------------------------------------------ graph table (append-only, versioned) */
typedef enum {
  FPR_EH_GRAPH_L = 0,             /* (R, "LA"[, "M", owner, ab]); with ENG_LAUNCH_SPLIT one entry per piece (G) */
  FPR_EH_GRAPH_H = 1,             /* (R, "H", key): n = D_pad */
  FPR_EH_GRAPH_T = 2,             /* (R, r, has_seqs): n = has_seqs (0 / 1), both pre-captured by ensure_bucket (D) */
  FPR_EH_GRAPH_TOWN_RESERVED = 3, /* graph T' (R, r, "own", n): NOT used in α (E): owner steps use TFUSE (d13tail) */
  FPR_EH_GRAPH_H1 = 4             /* v0.5 (2): the mixed-step head (H1) per rank, keyed by _key_text(hsd_key()) of the
                                     BARE 3-tuple "S_pad|D_pad|NT" (H1 never captures gvar keys; distinct from graph H's
                                     "R|H|S|D|NT"); R = -1 (bucket-independent); v3.1: ONE gen_slot per key (the shim's
                                     h1_slot_of), bumped by bump_h1 when refresh_graphs unpublishes a key retired from
                                     mh.graphs (nd's graph() skips gen-mismatched entries); aux_dst = the drafter's
                                     aux buffer hsd.draft_rank(r, D_pad).b["aux"]. Captured by the Python engine (after_sync
                                     of an embedded mixed step), published by refresh_graphs: an unpublished key -> embedded */
} FprEhGraphKind;
typedef struct {
  int32_t kind;                   /* FprEhGraphKind */
  int32_t rank;
  int64_t R;                      /* bucket rows */
  int64_t n;                      /* see FprEhGraphKind */
  int32_t piece;                  /* ENG_LAUNCH_SPLIT piece index. Launch order is PIECE-MAJOR (G): piece p on every
                                     rank before piece p + 1 on any rank. 0 if unsplit */
  int32_t gen_slot;               /* the bucket's generation slot */
  uint64_t generation;            /* the bucket's generation at capture: an entry whose generation != gens[gen_slot],
                                     or that a later entry with the same key supersedes, is skipped (F) */
  void* exec;                     /* cudaGraphExec_t (torch.cuda.CUDAGraph.raw_cuda_graph_exec). The exporter RETIRES a
                                     published graph on recapture (keeps the CUDAGraph object alive), never frees it (F) */
  void* aux_dst;                  /* v0.5: H1 entries only, the (rank, D_pad) drafter aux buffer; NULL for other kinds */
  char key[FPR_EH_KEY];           /* canonical key text: the lookup key */
} FprEhGraph;
typedef struct {
  uint64_t version;               /* the engine writes new entries fully, THEN bumps version (release); a reader loads
                                     version (acquire), then reads entries [0, count) */
  int64_t count;                  /* entries [0, count) are immutable once published */
  int64_t capacity;               /* fixed at init; never reallocated (a full table fails the capture loudly) */
  const FprEhGraph* entries;      /* engine-owned array of `capacity` entries: the pointer never changes */
} FprEhGraphTable;

/* ------------------------------------------------------------------ set-alternation state handoff (C, I, J, M) */
typedef struct {                  /* one eng.prev_seq entry (I): the device-token provenance of a request */
  const char* rid;                /* NUL-terminated request id */
  int64_t rank, index;            /* where its last tokens were produced (rank, sequence index) */
  int64_t serial;                 /* eng.step_serial of the step that PRODUCED these tokens: Engine.prev_of
                                     keeps the entry only if serial == step_serial - 1 (PREV_STEP_CHECK). nd stores its
                                     own producing-step serial, keyed by rid text; the exporter maps rid text back to the
                                     engine's rid objects */
} FprEhPrevOf;
typedef struct {                  /* tokens appended to eng.hist[rid] since the last handoff (J) */
  const char* rid;
  const int32_t* tokens;
  int64_t n;                      /* n = -1, tokens NULL: rid was DROPPED on a native step ->
                                     the exporter pops eng.hist / prev_seq / samp_ctx / structured (on_ctx's `dropped`) */
} FprEhHist;
typedef struct {
  int32_t struct_size;
  int32_t n_buckets;
  const int64_t* bucket_R;        /* [n_buckets]: the decode buckets the indices below refer to */
  uint8_t* in_i;                  /* [n_buckets * n_ranks]: s["in_h"] set index per (bucket, rank) */
  uint8_t* up_i;                  /* [n_buckets * n_ranks]: up_h set index */
  /* per rank (each rank toggles only when it uploads: owner / empty skips diverge the ranks) */
  uint8_t rx_hi[FPR_EH_MAX_RANKS];      /* RowExp._hi */
  uint8_t engram_i[FPR_EH_MAX_RANKS];   /* engram[r]._stage_i */
  /* global: one value toggled once per step for all ranks */
  uint8_t dl_i;                   /* D13._pre: dl_i ^= 1; sets dl_h[i][r] */
  uint8_t d2h_i;                  /* D13._pre: the positions D2H set */
  uint8_t qi;                     /* D13._advance */
  uint8_t hsd_hi;                 /* HSD._upload: _hi ^= 1 after the per-rank loop */
  uint8_t pad_[4];
  /* HSD's Python-side state (the engine's _hi etc.) travels inside push / pull: nd's HsdPrep.hi_ <-> hsd._hi */
  /* the prev-of map + step serial (I): nd owns them in 2b. push writes nd's map into eng.prev_seq / eng.step_serial;
   * pull reads them back (the embedded step's drafts_valid, head_result's non-async rows, _set_prev_seq) */
  const FprEhPrevOf* prev;
  int64_t n_prev;
  int64_t step_serial;
  /* the host token history (J): push hands the tokens nd extracted natively since the last handoff, so eng.hist (which
   * Engram's lookback reads for later prefill chunks) is complete before the embedded step.
   * v0.5: pull returns the DELTAS the embedded step appended (n_hist > 0): the exporter records
   * len(eng.hist[rid]) at push and returns eng.hist[rid][old_len:] at pull, for rids live on both sides. nd keeps the
   * full per-request history (StepHost.fresh prompt + output, its native extractions, these deltas): native mixed steps
   * build Engram's step args WITH host lookback (EG.step_args_from_compact(ct, H._lookback) reads it) */
  const FprEhHist* hist;
  int64_t n_hist;
  /* storage: on push the arrays are nd's (valid during the call); on pull the engine fills them with engine-owned
   * storage valid until the next push / pull */
  /* v0.7: EngramPhys._gen per rank (ENGRAM_GUARD v2's counts[2] / log generation), synced like engram_i */
  uint64_t engram_gen[FPR_EH_MAX_RANKS];
} FprEhSetState;

/* ------------------------------------------------------------------ the handover */
/* v0.7: one rank's EngramPhys
 * (hooks.engram[r]). The exporter CREATES EngramPhys._stage_ev[0 / 1] up front (Python creates them lazily) so both sides
 * wait on the same two events; ws_ev is one spare event of the rank. NULL pk_d = not exported: nd refuses PREPARE_ENGRAM */
typedef struct {
  int32_t struct_size;
  int32_t device, rank;
  int32_t guard;                  /* (slice 4, was pad_) 1 = engram.py's ENGRAM_GUARD v2 is on (counts[2] = gen, host slot check) */
  int32_t* pk_d;                  /* EngramPhys._pk (device) */
  int32_t* stage[2];              /* EngramPhys._stage[2] (pinned) */
  void* stage_ev[2];              /* cudaEvent_t EngramPhys._stage_ev[0 / 1] */
  int64_t hctx;                   /* EngramPhys.hctx (hash_setup's entry) */
  const int32_t* ids;             /* the bound input ids (hooks.ids32[r]: EngramPhys.b_ids) */
  const int32_t* tmap;            /* EngramPhys.tmap */
  int32_t* ring;                  /* EngramPhys.ring [n_state_slots, 16, 2] */
  int32_t* hash;                  /* [max_rows, 2, 24] */
  uint8_t* keep;                  /* [max_rows] */
  int32_t* err;                   /* [3] */
  void* rows[2];                  /* bf16 [max_rows, 24, 256] */
  void* kv[2];                    /* bf16 [max_rows, 25600] */
  const void* tabs[2];            /* HostTable.rec0 of layers 1, 14 */
  int64_t lo[2][24], hi[2][24];   /* EngramPhys.lo / hi (per layer, per column) */
  int32_t g_ctas, g_threads, g_unroll;   /* EngramPhys.gather_cfg */
  int32_t pad2_;
  const void* wbf[2];             /* bf16 [N, K] exact dequant of wkv (host-mapped under FPR_HOSTMAP=engram) */
  int64_t wbf_N, wbf_K;
  void* stage_buf;                /* EngramPhys._stage_buf (FPR_ENGRAM_STAGE): bf16 [ceil(N / C), K] */
  int32_t stage_chunks;           /* C (0 = no staging: one mm on wbf) */
  int32_t start_src_i64;          /* 1: start_src is int64, 0: int32 */
  void* ev[2];                    /* cudaEvent_t EngramPhys.ev[l]: inject L1 / L14 wait on them */
  void* ws_ev;                    /* a spare cudaEvent_t of this rank (estream.wait_stream(step stream)) */
  const void* start_src;          /* EngramPhys._start_src (D13 rowexp start); NULL = unbound */
  int32_t w8_max_rows;            /* EngramPhys.w8_max_rows (T <= it: W8A16 in Python -> nd refuses natively; 0 = no gemm8) */
  int32_t nring;                  /* (slice 4, was pad3_) EngramPhys.ring.shape[0] (the guard's host slot check) */
} FprEhEngramRank;
/* v0.7 (B7, native Worker.copies via the pagecopy C ABI): one rank's page-id staging, allocated at engine init:
 * ids = device int64 [2, cap] (row 0 source page ids, row 1 destination page ids of one copy group), up = 2 pinned sets of
 * the same layout (FprEhSets rule). ids.data NULL = not allocated: nd refuses native page copies (embedded Worker.copies) */
typedef struct {
  FprEhTensor ids;
  FprEhSets up;
  int64_t cap;
} FprEhPageCopyRank;

/* v0.7 slice 4: the CP step's index blob (p1::cp_host's int64 lists, cap entries) + its pinned
 * sets, and the CP-only mail stream (cp.py creates it lazily today: the engine creates it at init). NULL = refuse CP steps */
typedef struct {
  FprEhTensor cp_idx;             /* i64 [cap] device */
  FprEhSets cp_idx_up;            /* 2 pinned sets of [cap] i64 */
  void* cp_mail_stream;           /* cudaStream_t */
  int64_t cap;
} FprEhCpRank;
/* v0.7 slice 4: a loaded C ABI library by name ("mixed_cxx", "tail_gather", "p1", "idx2f_tab", "pagecopy", ...): nd / core2b
 * dlopen(path, RTLD_NOLOAD) + the extension's fpr_<ext>_table (one copy of each extension in the process) */
typedef struct { char name[24]; const char* path; } FprEhLib;

typedef struct FprEngineHandover {
  int32_t struct_size;            /* sizeof(FprEngineHandover) as the engine compiled it */
  int32_t version;                /* FPR_EH_VERSION */
  int32_t n_ranks;
  int32_t k;                      /* speculative tokens (k + 1 rows per decode sequence) */
  int64_t page_tokens, window_page_tokens, vocab_size, max_model_len;
  int64_t largest_bucket;         /* eng.cfg.buckets[-1] */
  /* the engine's decode configuration nd::Engine::init needs, as JSON (engine-owned, NUL-terminated, valid for the
   * handover's lifetime). the D13 module
   * flags (GT / OWNER / EMPTY / ADV_SLICE / D13S / TFUSE / CHECK modes, K1, DELTA_CAP, ERR_EVERY, RES_OWNER), D13Host's
   * Config (graphs, largest_bucket, graph_max_ctx, fb_reuse, scratch[]), HsdConfig (k, max_seqs, seq / row buckets,
   * nt_max, pool, window, ring), the HSDT c1d switch + HSDT_C1D_SMAX, the bucket list. JSON keeps the C layout stable
   * while that set grows (ND-2 / ND-3) */
  const char* config_json;
  FprEhRankState ranks[FPR_EH_MAX_RANKS];
  void* bm_h_reserved;            /* HSD's ONE pinned bitmask buffer shared by all ranks: v1 leaves it to the fallback */
  /* generations: one engine-owned counter per bucket slot, allocated ONCE at init with headroom for lazily created
   * buckets (n_gens = fixed capacity, never reallocated). The engine bumps a slot (release) when it invalidates its
   * memory (only at (i) / (ii)); the C++ loads it (acquire) before every use of a view of that slot */
  uint64_t* gens;
  int32_t n_gens;
  int32_t pad_;
  /* v0.5 (4): the eviction notice. Bumped (release) on every mixed-bucket create / evict, together with that bucket's
   * generation; nd loads it (acquire) at the start of every native mixed step and re-fetches its mixed views if it moved */
  uint64_t* buckets_epoch;
  /* the engine's callbacks (ctx = its C shim). 0 = OK, else an error code; the text from last_error.
   *  fill_bucket / graphs: cheap, no GIL, callable from any native thread (graphs(): acquire on version).
   *  ensure_bucket: may take the GIL and may invalidate (rule (i)): called by the loop at a step boundary with nothing
   *    enqueued, never from T_head, and ALWAYS via run_python, i.e. on the engine's main thread (rank streams current,
   *    allocator stream tags right; the shim's GIL acquire nests harmlessly). It pre-creates EVERY lazy per-bucket buffer (in_h / up_h sets + events, paged_d,
   *    own_rbase / own_sbase, pos_ev) and captures graph T for has_seqs 0 and 1 (D): nd never allocates or captures.
   *  push_state / pull_state: the set handoff (C), called by the loop after nd::drain(), around every embedded step,
   *    ALWAYS via run_python (the engine's main thread).
   *    Events (M): Python's own uploads may leave fresh torch Events as set holders. pull_state records every
   *    persistent set event on its rank's step stream (it completes after the embedded step's copies) and makes it the
   *    holder again; push_state sets every holder to the persistent event. Then nd's wait-then-re-record rule is exact
   *    on both sides */
  void* ctx;
  int32_t (*fill_bucket)(void* ctx, int64_t R, int32_t rank, FprEhBucketRank* out);
  int32_t (*graphs)(void* ctx, FprEhGraphTable* out);
  int32_t (*ensure_bucket)(void* ctx, int64_t R);
  int32_t (*push_state)(void* ctx, const FprEhSetState* s);
  int32_t (*pull_state)(void* ctx, FprEhSetState* s);
  const char* (*last_error)(void* ctx);
  /* v0.5 (1): mixed buckets. fill_mixed: cheap, no GIL (as fill_bucket). ensure_mixed: rule (b) point (i): may create,
   * evict others (mixed-round churn, empty_cache) and bumps generations + buckets_epoch; called by nd at a step boundary
   * with nothing in flight, ALWAYS via run_python. Status 1 = the engine refused (no room): nd runs the step embedded */
  int32_t (*fill_mixed)(void* ctx, int64_t R, int32_t role, const char* key, int32_t rank, FprEhMixedBucketRank* out);
  /* the step's WHOLE set at once */
  int32_t (*ensure_mixed)(void* ctx, int32_t n, const FprEhMixedKey* keys);
  /* v0.7: the engine's C++ TableBuilders, fpr::ITables* (tables.h; kvtables'
   * TableBuilder.itables_ptr() / cxx_kv.itables_ptr(tb)), engine lifetime, OWNED by the Python objects:
   *  tb_tail = Engine.tb_tail, the mixed steps' T tables: ONE instance for Python and nd (one D12 `sent` state, no re-send);
   *  tb_d13  = D13.tb, the synchronous fallback's builder (nd's mixed_begin tables).
   * NULL = Python builders (ENG_CXX_TB off, or "check" mode: split state): nd refuses native TB_TAIL / mixed_begin tables
   * (embedded). Calls go through the vtable in fpr_kvtables.so (both sides: tables.h + g++-13). Never used concurrently with
   * a Python build (nd's step runs between Python's; an embedded step builds through the same object) */
  void* tb_tail;
  void* tb_d13;
  /* v0.7 slice 2: Engram's native prepare. engram_lib = the loaded engram extension's path
   * (EG.ext().__file__: dlopen RTLD_NOLOAD + fpr_engram_table); NULL = no Engram / not exported */
  const char* engram_lib;
  FprEhEngramRank engram[FPR_EH_MAX_RANKS];
  /* v0.7 slice 2 (B7): the page-copy id staging per rank */
  FprEhPageCopyRank pagecopy[FPR_EH_MAX_RANKS];
  /* v0.7 slice 3 (C2): the driver's owner kind ids (drv.x.owner_kind; mixed_cxx's builtin K_* are fixed: bd 1, router 2,
   * ag 3, agce_join 4, rs 5, final 6, join 7, waitev 8): embed, engram_bd, attn_a, idx, attn_b, moe; -1 = no owner kind */
  int32_t span_kinds[8];
  /* v0.7 slice 4 */
  int32_t n_libs, pad7_;
  FprEhLib libs[FPR_EH_MAX_LIBS];
  FprEhCpRank cp[FPR_EH_MAX_RANKS];
  const void* moe_snap;           /* p0::MoeSnap (mixed_cxx_moe_ensure.snapshot(drv)); NULL = moe decide unavailable */
  int64_t moe_snap_bytes;
  const void* comb_tpl[FPR_EH_MAX_RANKS];
  int64_t comb_tpl_bytes[FPR_EH_MAX_RANKS];
  uint64_t* comb_gen;             /* acquire before reading comb_tpl (release by the exporter after an append) */
  /* v0.7 slice 5: the ITables ABI of tb_tail / tb_d13's creator (fpr_kvtables.so): its compiled
   * FPR_TABLES_ABI (tables.h; 2 = save_state / restore_state appended to the vtable). 0 = unknown / older: a caller compiled
   * against a newer tables.h must not call the appended virtuals (past the creator's vtable). The exporter copies the value
   * from the creating module (cxx_kv / kvtables .tables_abi()), never from its own header */
  int32_t tables_abi, pad8_;
  /* v0.7 slice 6: the CP pieces' driver kind ids
   * (drv.x.owner_kind / fpr_drv_owner_kind): attn_bi, attn_bo, idx_c, idx_s, cp_x1a_send, cp_x1a_recv, cp_x2_send,
   * cp_x2_recv. <= 0 = not registered (zero until exported): a CP step's segment refuses (embedded) */
  int32_t span_kinds_cp[8];
  /* v0.7 slice 7: HSD's bitmask buffers, pre-created by the exporter at init.
   * hsd_bm_h = HSD._bm_h, pinned int32 [T_max, BM_WORDS = ceil(V / 32)]: the one host staging HSD._build writes (nd's
   * HsdPrep binds it as bind_bm). hsd_bm_d[r] = HSD._bm_d[r], device int32 [T_max, BM_WORDS]: nd H2Ds rows [:T] into it on
   * rank r's step stream after graph L and before the "|bm" graph H that reads it. A per-rank ARRAY here, not a
   * FprEhRankState field (growing ranks[]'s element would move every later offset). data NULL = not exported: nd keeps
   * refusing "grammar rows" */
  FprEhTensor hsd_bm_h;
  FprEhTensor hsd_bm_d[FPR_EH_MAX_RANKS];
} FprEngineHandover;

/* the C++ check before every use of a bucket view (inline, lock-free) */
static inline int fpr_eh_current(const FprEngineHandover* h, const FprEhBucketRank* b) {
  return __atomic_load_n(&h->gens[b->gen_slot], __ATOMIC_ACQUIRE) == b->generation;
}
/* v0.5: the same for a mixed view, and the eviction notice */
static inline int fpr_eh_mixed_current(const FprEngineHandover* h, const FprEhMixedBucketRank* b) {
  return __atomic_load_n(&h->gens[b->gen_slot], __ATOMIC_ACQUIRE) == b->generation;
}
static inline uint64_t fpr_eh_buckets_epoch(const FprEngineHandover* h) {
  return h->buckets_epoch ? __atomic_load_n(h->buckets_epoch, __ATOMIC_ACQUIRE) : 0;
}

#ifdef __cplusplus
}
#endif
#endif /* FPR_ENGINE_HANDOVER_H */
