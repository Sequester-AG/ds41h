// nd_engine.h -- NativeDecode ND-2 / ND-3: the nd::IEngine2b implementation (core2b/nd_api.h)
// over the engine handover (core2b/engine_handover.h).
// One async decode step = engine/d13.py D13.dispatch's async branch (+ engine/d13s.py's steady path)
// decision for decision, in the same stream order; the rest of the step kinds go to the embedded engine (the caller).
// v0: the head (positions -> HSD prepare -> graph H -> advance) runs on the calling thread, as Python's dispatch; T_head is
// a later, separately gated step.
#pragma once
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core2b/engine_handover.h"
#include "../core2b/nd_api.h"
#include "../d13/d13_host.h"
#include "../p0/inputs_build.h"   // (v3.3) p0::InputsSnap / build_inputs
#include "nd.h"
#include "nd_hist.h"

namespace fpr {
namespace nd {

constexpr int NW = 4;   // ranks

struct Cfg {            // config_json (the engine's D13 / HSD / HSDT settings, filled by nd_export.py)
  bool gt = true, owner = true, empty = true, adv_slice = true, d13s = true, tfuse = false;
  bool check = false;   // any *_CHECK mode on -> async_ok refuses (they compare against Python)
  int64_t k1 = 4, k = 3, delta_cap = 65536;
  std::vector<int64_t> buckets;
  bool c1d = false; int64_t c1d_smax = 0;
  bool c1_mega = false;
  // (v3.22, L1) c1mega adapter facts (nd_export config_json): runs configured, NRM (rows of a C1 step), ENG_C1_MEGA_AB (per-arm
  // graph keys: refused), C1M_IX + C1IDX_E0 (informational: e0_prepare is inside graph L, nd only uploads)
  bool c1_runs = false, c1_ab = false, c1_e0 = false;
  bool head_bm = false;   // the engine captures graph H's bitmask variant (key + "|bm"; ENG_HEAD_BM_GRAPH)
  bool peer_map = false;  // (v3.26) engine._torch_peer_maps ran (ENG_PEER_MAP=1): P2P reads of torch memory are mapped
  // (v3.27, P4 / L5) decode steps with Worker.copies (page copies + the Engram ring follow) and migrated sequences (device
  // tokens on another rank) natively (ND_DECODE_COPIES, exporter default 1; 0 = refused as before: embedded)
  bool decode_copies = false;
  // (v3.27) migrated sequences' device tokens natively (ND_DECODE_MIG, default 0 = refused as before): unreachable on the
  // served async scheduler (a moved sequence skips a step, then resumes on host ids); not GPU-gated yet
  bool decode_mig = false;
  LongCfg lng;   // (v3.30) long-context decode: the long-row level rule's inputs (nd_hist.h long_level_of)
  // (v3.33, slice 2; ND_LONG, default off) a long-context decode step (lvl > 0) natively on D13's ASYNC long variant: keys and
  // views by Rk = R + lvl * LVL_TAG ((Rk, "LA"), (Rk, "H", key), ("T", Rk, r, b), the bucket view, the set indices), sizes by
  // the bucket rows R. Needs FPR's d13 step 1 (Python captures (Rk, "LA")) and the worker's Rk ensure; else refused as v3.30
  bool long_async = false;
  int64_t c1_nrm = 4;
  int64_t mixed_max_rows = 0;   // (v3.4) the largest mixed bucket R (A / T): nd's own mixed upload staging (0 = none)
  // ND-2: the extensions' C tables (dlopen of the engine's own builds)
  // what native decode cannot do (gate head, NANCHECK, VROWS -> async_ok refuses), D13's ERR_EVERY / RES_OWNER
  std::string ext_rowexp, ext_hsdprep;
  std::string ext_pagecopy;   // (v3.10) the pagecopy extension's path (its fpr_pagecopy_table: native page copies)
  bool gate = false, nancheck = false, vrows = false, res_owner = true;
  int64_t err_every = 1;
  d13::Config d13;
  d13::HsdConfig hsd;
};

// Req / PrevOf: nd_hist.h (the CUDA-free request bookkeeping + J history, replay-tested on CPU)

struct Handle {         // one native in-flight step (d13.py StepHandle). nd OWNS handles: valid until KEEP dispatches after
  int64_t step = 0, R = 0;            // its wait() (the 2b loop: dispatch(N+1), then wait(N))
  int32_t mega = -1;                  // (v3.22) the C1 mega owner of this step (graph L "R|LA|M|owner"), -1 = none
  std::vector<GrammarRows> grammar;   // this step's grammar rows (Hooks.grammar of the step: masks, acceptable
  bool bm = false;                    //  the drafts readback set); bm = some rank has a mask (graph H's "|bm" variant)
  std::vector<StepBatch> batches;
  d13::HsdStep cur;                   // the HSD step record (dseqs / rids / n_req / step numbers) of THIS step
  std::array<cudaEvent_t, NW> ev{};   // _record (the handover's persistent step events, alternating)
  int ev_slot = -1;                   // which of step_event[2]
  bool deferred = false;              // D13S steady: result copies queued on the extraction streams
  int res_set = -1;                   // pinned result set (D2H targets) of this step's deferred copies
  std::vector<int> act;               // ranks with drafting requests (deferred copies)
  bool done = false;                  // results extracted (h._res is not None)
  bool consumed = false;              // wait() returned them
  std::vector<StepResult> res;
};

class Engine : public IEngine2b, public IMixed2b {
 public:
  Engine();
  ~Engine() override;
  Status init(const FprEngineHandover* h, RunPython run_python, std::string* err) override;
  std::string async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host) override;
  // per-call host time of the loop's nd calls: stats nd_<hook>_us / _n / _max_us
  void nd_tick(const char* hook, double us);
  Status dispatch(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host, Handle** h,
                  std::string* err) override;
  Status wait(Handle* h, std::vector<StepResult>* out, std::string* err) override;
  Status drain(std::string* err) override;
  Status embedded_begin(const StepHost& host, std::string* err) override;
  Status embedded_end(std::string* err) override;
  // (v0.5) the request's full host history (= eng.hist[rid]) for native mixed steps' Engram lookback; nullptr = unknown
  const std::vector<int32_t>* history(const Rid& rid) const {
    auto it = reqs_.find(rid);
    return it == reqs_.end() ? nullptr : &it->second.hist;
  }
  std::string stats_json() const override;
  void replay_view(const std::vector<Rid>& rids, std::vector<ReplayEntry>* out) const override;   // (jZr)

  // Every "native
  // impossible" condition is checked by can_head / begin's own refusals BEFORE any device op or state change; after begin
  // returned OK, a non-OK status is INTERNAL (fatal).
  IMixed2b* mixed() override { return this; }
  bool can_head(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                std::string* why) override;
  Status begin(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host, Ctx** c,
               std::string* err) override;
  std::string h1_key(const std::vector<StepBatch>& b) const override;
  std::string h1_key_for(const std::vector<StepBatch>& b, const StepHost& host) const override;   // (v3.26)
  const std::vector<StepBatch>& resolved(const Ctx* c) const override;
  const std::vector<RankTables>& tables(const Ctx* c) const override;
  Status prepare_head(Ctx* c, std::string* err) override;
  Status h1_launch(Ctx* c, int r, const FprEhMixedBucketRank& t, const StepBatch& tail_b, std::string* err) override;
  Status devtok(Ctx* c, int r, const FprEhMixedBucketRank& b, const std::vector<int64_t>& rows,
                const std::vector<int64_t>& src, const std::vector<std::array<int64_t, 4>>& mig, std::string* err) override;
  Status upload(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::InputsOut& io, std::string* err) override;
  Status set_vidx(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::VidxOut& v, std::string* err) override;
  Status set_tail_ix(Ctx* c, int r, const std::vector<int64_t>& ix, std::string* err) override;
  bool prev_of(const Ctx* c, const Rid& rid, int32_t* rank, int64_t* seq) const override;
  Status sync(Ctx* c, std::string* err) override;
  bool page_copies_ok(const std::vector<PageCopyBatch>& groups, std::string* why) const override;
  Status page_copies(Ctx* c, const std::vector<PageCopyBatch>& groups, std::string* err) override;
  Status cabi(Ctx* c, int r, int (*fn)(const void*), void* args, std::string* err) override;                  // (v3.11)
  Status h2d(Ctx* c, int r, const FprEhSets* sets, void* dst, const void* src, int64_t bytes, std::string* err) override;
  bool lookback(const Ctx* c, const Rid& rid, int64_t start, int64_t out[3]) const override;                // (v3.16)
  void set_lazy_push(bool on) override { lazy_push_ = on; }                                                  // (v3.23)
  Status push_now(Ctx* c, std::string* err) override;
  Status finish(Ctx* c, std::vector<StepResult>* out, std::string* err) override;

  struct MixedBegin {                    // D13._fallback's resolved inputs of the step
    std::vector<StepBatch> rb;           // _fb_rows: device-resolved starts / positions / paged / window slots per rank
    std::vector<RankTables> tables;      // the private TableBuilder's full tables of rb (D13.tb.build)
  };
  struct MixedHead {                     // prepare_head's output
    d13::HsdStep cur;                    // HSD's record of the step (hsd.cur): bucket key, dseqs, rids, n_req, n_s / n_t
    std::string h1_key;                  // FPR_EH_GRAPH_H1 key text "S_pad|D_pad|NT"
  };

 private:
  // host bookkeeping (d13.py / engine.py state nd owns in 2b)
  void observe(const StepHost& host, bool native);   // native: drops queued for push_state (J)
  bool prev_of(const Rid& rid, int64_t* rank, int64_t* index) const;   // engine.prev_of (PREV_STEP_CHECK)
  void set_prev_seq(const std::vector<StepBatch>& b);                  // D13._set_prev_seq
  bool bucket_view(int64_t R, int r, FprEhBucketRank* v, std::string* err);   // handover view, generation-checked
  const FprEhGraph* graph(const std::string& key, int r, int piece);   // the newest live entry of a key
  std::string l_key(int64_t R, int32_t mega = -1) const;   // (v3.22) mega >= 0: (R, "LA", "M", owner)
  int32_t mega_owner(const std::vector<StepBatch>& b, int64_t R) const;   // (v3.22) engine.c1m_owner (-1 = not a C1 step)
  std::string h_key(int64_t R, const d13::HsdStep& c, bool bm = false) const;   // (v3.25) bm: + "|bm"
  bool bm_ready_() const;             // (v3.25) cfg head_bm + HSD's _bm_h / _bm_d[r] exported (handover slice 7)
  Status structured_drafts_(Handle* h, std::string* err);   // (v3.25) Hooks.head_result's D7 / D11 drafts
  Status drafts_into_(const std::vector<StepBatch>& b, const std::vector<GrammarRows>& g, std::vector<StepResult>* res,
                      std::string* err);   // (v3.26) the shared body (decode extraction / mixed finish)
  // HSD's bucket key (S_pad, D_pad, NT + the c1d variant's owner) from host-known batch shapes (HSD._build's n_s /
  // first-row runs / n_t; positions only change the blob): async_ok checks graph H BEFORE nd commits to a step
  bool hsd_key_of(const std::vector<StepBatch>& b, d13::HsdStep* k, std::string* why) const;
  // (v3.33) Rk = the bucket KEY (views, set indices, graph T); R = its rows (sizes). Rk == R below lvl 1
  Status pre(int64_t Rk, int64_t R, const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, std::string* err);
  int long_lvl_(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct) const;   // 0 unless ND_LONG
  Status head(Handle* h, const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, int64_t R,
              std::string* err);                                            // extract N, positions, HSD's host prepare
  Status steady_head(Handle* h, const std::vector<StepBatch>& b, int64_t R, std::string* err);   // defer + device prep
  Status tail(Handle* h, const std::vector<StepBatch>& b, int64_t R, std::string* err);   // prev-seq, graph H, advance, record
  Status advance(const std::vector<StepBatch>& b, std::string* err);        // D13._advance
  Status device_tokens(int r, const StepBatch& sb, const CompactTables& ct, const FprEhBucketRank& v,
                       std::string* err);                                // Hooks._device_tokens_v2 (after pre_full)
  // (v3.27, P4) a decode sequence's device tokens live on another rank (Python: one cross-device copy): *r0 / *si0 = its
  // previous step's (rank, seq index); false = not migrated (host ids, no previous step, or the same rank)
  bool migrated_(int r, const SeqEntry& se, const StepBatch& sb, const CompactTables& ct, int64_t* r0, int64_t* si0) const;
  // (v3.27, P4) Worker.copies' groups of a decode step (tables.h dedup_copies + page_copy_batches over its batches)
  static void copy_groups_(const std::vector<StepBatch>& b, std::vector<PageCopyBatch>* g);
  // (v3.27) the page copies body shared by the mixed step (c = its Ctx: stats "mixed_*", ring copies through cabi(c)) and the
  // decode dispatch (c = nullptr: stats "decode_*", ring copies through cabi_raw_): page_copies_ok, device syncs of the
  // touched ranks, pool copies, Engram ring copies, the done events every rank's step stream waits for
  Status issue_page_copies_(Ctx* c, const std::vector<PageCopyBatch>& groups, std::string* err);
  Status cabi_raw_(int r, int (*fn)(const void*), void* args, std::string* err);   // cabi without a mixed step
 public:
  // Hooks._device_tokens_v2's DEVICE half on rank r's step
  // stream, for any bucket view: the migrated copies first (tok[row0 : row0 + n] = next_ids[rank][seq, :n], seq_rows order;
  // peer copies from that rank's next_ids), then the pairs' H2D into Hooks._dt_d (set ping-pong) and
  // tok.index_copy_(0, rows, next_ids[r].flat[src]). tok = the bucket's tok (decode: FprEhBucketRank.tok; mixed:
  // in_d + in_off_tok), tok_len its rows. The pairs / copies come from the host builders (p0::build_inputs' rows / src /
  // mig, build_mixed's InUpload.devtok). Requires the previous step complete (a migrated source lives on another rank).
  struct DevtokMig { int64_t row0, n; int32_t rank; int64_t seq; };
  Status devtok_issue(int r, int64_t* tok, int64_t tok_len, const std::vector<int64_t>& rows,
                      const std::vector<int64_t>& src, const std::vector<DevtokMig>& mig, std::string* err);
  // devtok_issue's checks alone (sizes, Hooks._dt_d capacity, every pair / copy in range): BAD_ARG, no device op
  Status devtok_check(int r, int64_t tok_len, const std::vector<int64_t>& rows, const std::vector<int64_t>& src,
                      const std::vector<DevtokMig>& mig, std::string* err) const;
  // (v3.3, ENG_ND_T: FPR's step_mixed seam Engine._mixed_upload_T, the override in nd_t.py) Engine._upload_v2 (dev_pos
  // False) of one (bucket, rank) natively, on rank r's step stream: p0::build_inputs(snap, tab) -> the blob into `host`
  // (the bucket's pinned set the CALLER chose and waited for) -> its H2D into `in_d` -> devtok_issue into in_d's tok
  // (migrated copies, pairs, index_copy). snap carries Python's prev-of (exact inside embedded steps too), the scratch slot
  // and hsd on; k1 is filled from next_ids here. The caller keeps the set bookkeeping (holder event, in_i) and sets
  // s["n"] / s["n_sample"] from *n / *n_sample (*n_pairs: the same-rank device-token pairs; Python flips
  // Hooks._dt_i[r] when > 0, as _device_tokens_v2 does). BAD_ARG = nothing issued: every check (the builder, sizes,
  // devtok_check) runs before the first device op.
  // bucket b's
  // Engine._set_vidx values into b's v0.6 vidx_d on rank r's step stream: the whole layout (vidx[:vcount], zeros up to W R,
  // nv, nd or zeros) through nd's own pinned set, ONE H2D. BAD_ARG = refused before any device op (no vidx_d / layout or
  // size mismatch / not current), INTERNAL = a CUDA error after issue.
  Status vidx_issue(int r, const FprEhMixedBucketRank& b, const p0::VidxOut& v, std::string* err);
  Status upload_inputs(int r, p0::InputsSnap snap, void* in_d, int64_t in_d_bytes, void* host, int64_t host_bytes,
                       const RankTables& tab, int64_t* n, int64_t* n_sample, int64_t* n_pairs, std::string* err);
 private:
  bool is_steady(const std::vector<StepBatch>& b, int64_t R, const StepHost& host) const;
  Status extract(Handle* h, std::string* err);                              // D13._extract / Steady.extract_deferred
  Status defer_extract(Handle* n, std::string* err);                        // Steady.defer_extract(N)
  void results_from(Handle* h, const int32_t* const* ns, const int64_t* const* smp,
                    const std::vector<StepBatch>* b = nullptr);   // HSD.result's formula + hist (J); (jY) b: mixed
  Status post_extract(Handle* h, std::string* err);                         // rowexp miss counters (ERR_EVERY)
  Status ext_init(std::string* err);                                        // C tables, result sets, hsdprep err words
  void keep(std::unique_ptr<Handle> h);                                     // retention of finished handles

  const FprEngineHandover* h_ = nullptr;
  RunPython run_python_;
  Cfg cfg_;
  std::unique_ptr<d13::D13Host> host_;
  std::map<Rid, Req> reqs_;
  std::map<Rid, PrevOf> prev_;
  std::vector<Rid> native_drops_;   // requests dropped on native steps since the last push (J: n = -1)
  int64_t step_serial_ = 0;
  // D13 toggles + skip state (d13.py: d2h_i, dl_i, qi, _r_empty, _b_empty, _own_last, _epoch)
  uint8_t d2h_i_ = 0, dl_i_ = 0, qi_ = 0;
  std::array<uint8_t, NW> rx_hi_{}, eg_i_{};
  std::array<uint64_t, NW> eg_gen_{};   // (v0.7) EngramPhys._gen: pulled, pushed back
  std::map<std::pair<int64_t, int>, std::array<uint8_t, 2>> in_up_i_;   // (R, rank) -> (in_i, up_i)
  int64_t epoch_ = 0;
  std::array<bool, NW> r_empty_{};
  std::map<std::pair<int64_t, int>, int64_t> b_empty_;                  // (R, rank) -> epoch
  std::array<std::string, NW> own_last_;                               // serialized (R, owner signature, epoch)
  // steady path (d13s.py)
  std::string prev_sig_;
  Handle* inflight_ = nullptr;
  Handle* last_prep_ = nullptr;
  // graph table snapshot
  uint64_t gversion_ = 0;
  std::map<std::string, std::vector<const FprEhGraph*>> gindex_;      // key|rank|piece -> entries (newest last)
  // ND-2: HSD's step record (hsd.cur), the C tables, nd's own pinned result sets + events, hsdprep's error words
  d13::HsdStep cur_;
  bool have_cur_ = false;
  int (*rx_advance_)(const void*) = nullptr;
  int (*hsdprep_)(const void*) = nullptr;
  static constexpr int NRING = 4;                                       // d13s.NRING (+ one set for synchronous extraction)
  struct ResSet { int32_t* ns[NW] = {}; int64_t* smp[NW] = {}; cudaEvent_t ev[NW] = {}; Handle* owner = nullptr; };
  std::array<ResSet, NRING + 1> res_{};
  int res_i_ = 0;
  std::array<int32_t*, NW> prep_err_{};
  std::array<std::array<int64_t*, 2>, NW> dt_h_{};                     // device-token (rows, src) staging [2, dt_nmax_]
  std::array<std::array<cudaEvent_t, 2>, NW> dt_ev_{};
  std::array<uint8_t, NW> dt_i_{};
  int64_t dt_nmax_ = 0;
  // (v3.27, P4) migrated decode sequences: mig_src_ev_[r0] = rank r0's step stream at pre()'s start (step N's next_ids
  // written: RAW), mig_done_ev_[r] = rank r's step stream after its migrated copies (r0's later work waits: WAR). Python
  // gets both from copy_'s two-way cross-device barrier (in its embedded, drained step)
  std::array<cudaEvent_t, NW> mig_src_ev_{}, mig_done_ev_{};
  bool mig_step_ = false;
  uint8_t ev_i_ = 0;
  std::array<Handle*, 2> ev_owner_{};
  int64_t n_extract_ = 0;
  std::array<int64_t, NW> err_seen_{};
  bool ext_ready_ = false;
  std::string ext_fail_;   // (v3.40, jX) can_head's ext_init failure, latched (refusal "nd init failed")
  static constexpr size_t KEEP = 8;
  std::vector<std::unique_ptr<Handle>> handles_;
  int (*rx_ring_ops_)(const void*) = nullptr;
  int (*rx_expand_)(const void*) = nullptr;
  std::string mixed_off_;                                               // non-empty: native mixed steps unavailable (why)
  std::array<int64_t*, NW> mx_start_h_{};                               // starts D2H [S_max]
  std::array<int32_t*, NW> mx_acc_h_{}, mx_acc_d_{};                     // accepted counts [S_max] (D13.acc_h / rx.accepted)
  std::array<std::array<int64_t*, 2>, NW> mx_idx_h_{};                  // H1 sample-row index staging [n_max], 2 sets
  std::array<int64_t*, NW> mx_idx_d_{};
  std::array<std::array<cudaEvent_t, 2>, NW> mx_idx_ev_{};
  std::array<uint8_t, NW> mx_idx_i_{};
  int64_t mx_idx_nmax_ = 0;
  // (v3.4) IMixed2b::upload's own pinned blob sets per (role A / T, rank): 2 sets of in_layout(mixed_max_rows) bytes + events
  std::array<std::array<std::array<uint8_t*, 2>, NW>, 2> mxu_h_{};
  std::array<std::array<std::array<cudaEvent_t, 2>, NW>, 2> mxu_ev_{};
  std::array<std::array<uint8_t, NW>, 2> mxu_i_{};
  int64_t mxu_cap_ = 0;
  // (v3.4) set_vidx's / set_tail_ix's own pinned sets: per (role, rank) the vidx_d layout at mixed_max_rows; per rank tail_ix
  std::array<std::array<std::array<uint8_t*, 2>, NW>, 2> mxv_h_{};
  std::array<std::array<std::array<cudaEvent_t, 2>, NW>, 2> mxv_ev_{};
  std::array<std::array<uint8_t, NW>, 2> mxv_i_{};
  std::array<std::array<int64_t*, 2>, NW> mxx_h_{};
  std::array<std::array<cudaEvent_t, 2>, NW> mxx_ev_{};
  std::array<uint8_t, NW> mxx_i_{};
  int64_t mxv_cap_ = 0, mxx_cap_ = 0;
  // (v3.10) native page copies: pagecopy.copy, per rank nd's id staging (2 pinned sets [2, cap] + events, device [2, cap])
  // and one "copies done" event (the cross-rank waits that replace Worker.copies' device syncs)
  int (*pc_copy_)(const void*) = nullptr;
  int (*eg_ring_copy_)(const void*) = nullptr;   // engram's "engram.ring_copy"
                                                 // the state pool's Engram ring follow; nullptr = state groups refused
  std::array<std::array<bool, NW>, NW> peer_ok_{};   // (v3.26) cudaDeviceCanAccessPeer(rank d -> rank s)
  std::array<std::array<int64_t*, 2>, NW> pch_{};
  std::array<std::array<cudaEvent_t, 2>, NW> pcev_{};
  std::array<int64_t*, NW> pcd_{};
  std::array<cudaEvent_t, NW> pcdone_{};
  std::array<uint8_t, NW> pc_i_{};
  std::unordered_map<const FprEhSets*, uint8_t> h2d_i_;                // (v3.11) h2d's alternation index per sets
  int64_t pc_cap_ = 0;
  // (v3: the internal steps behind IMixed2b; v2's public hooks)
  // (1) begin (nd_mixed.cpp) does D13._fallback's front itself (refusals, drain, push, resets, deltas, rowexp, tables)
  // (3) Hooks.prepare_head = the P0 builder (sampling states, drafts_valid from prev_of / host-given ids) + HSD._build
  //     (hsd_prep) into the next blob set + its H2D per rank. toks[r] = bucket T's host ids of rank r (nullptr: none).
  Status mixed_prepare_head(const std::vector<StepBatch>& b, const std::vector<const std::vector<int64_t>*>& toks,
                            MixedHead* out, std::string* err, const std::vector<GrammarRows>* g = nullptr);
  // (4) mixed_cxx_head.head_eager for rank r (h_own gather, aux copy, the H1 graph)
  Status mixed_h1_launch(const MixedHead& mh, int r, const FprEhMixedBucketRank& t, const StepBatch& tail_b,
                         std::string* err);
  // (5) head_result per rank + J, prev_seq on b (CP helpers excluded), D13._fallback's tail (rowexp advance, device sync)
  Status mixed_finish(const MixedHead& mh, const std::vector<StepBatch>& b, const MixedBegin& mb,
                      std::vector<StepResult>* out, std::string* err);
  // embedded_begin's handoff (C): nd's set indices, prev-of + serial (I), the natively extracted tokens + drops (J) ->
  // the engine (push_state through run_python); clears what it pushed
  // (v3.23, L3a) the scalars push_state hands over, as of the push point: a lazy push (push_now, after begin's dl / rx
  // flips and serial bump) sends begin's snapshot, so Python sees exactly what the eager push would have sent
  struct PushScalars {
    std::array<uint8_t, NW> rx_hi{}, eg_i{};
    std::array<uint64_t, NW> eg_gen{};
    uint8_t dl_i = 0, d2h_i = 0, qi = 0, hsd_hi = 0;
    int64_t step_serial = 0;
  };
  PushScalars push_scalars_() const;
  Status push_state_(std::string* err, const PushScalars* sc = nullptr);
  std::unique_ptr<Ctx> mx_ctx_;                                         // the open native mixed step (begin .. finish)
  bool lazy_push_ = false;      // (v3.23, L3a) begin skips push_state; push_now pushes once per step
  bool pushed_step_ = false;    // this mixed step's state was pushed (begin when not lazy, or push_now)
  PushScalars push_sc_;         // (v3.23) begin's snapshot at its push point (lazy mode)
  Status mixed_ext_init(std::string* err);
  void mixed_free();
  void mixed_tick(const char* hook, double us);                         // per-hook host-us counters (sum / n / max)
  // (v3.21, L0) async_ok's body; the public async_ok counts its refusals by reason (stats refuse:<reason>, digits -> #)
  std::string async_ok_(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host);
  // ND_EVPROF=1 (default off: no event is created or recorded): per-rank TIMING events on the step
  // stream at the native decode step's boundaries, E0 dispatch / pre / E1 graph L / E2 head / E3 graph H / E4 advance + _record /
  // E5, and the GPU idle gap E5(N) -> E0(N + 1) between consecutive native decode steps. Harvested in order at the next
  // dispatch (cudaEventQuery: never blocks), summed into stats evp_* (ns): rank 0 and the max over ranks per step.
  struct EvpSet {
    std::array<std::array<cudaEvent_t, 6>, NW> e{};
    int64_t seq = -1;
    bool rec = false, harv = false;
  };
  std::array<EvpSet, 8> evp_{};
  int evp_on_ = -1, evp_cur_ = -1, evp_prev_ = -1;
  int64_t evp_seq_ = 0;
  bool evp_init_();
  void evp_rec_(int k);
  void evp_harvest_();
  void evp_break_() { evp_seq_ += 2; }
  // ND_ACCSTAT=1 (default off): drafter acceptance counts at every native result extraction
  // per drafting request: accepted drafts by position (0..k-1) x context-length bin (start position: < 8K / 8-64K /
  // 64-262K / >= 262K) x requests-per-step bin (1 / 2-8 / 9-32 / > 32). Counts only (no ids, no tokens). A stderr line every
  // ND_ACCSTAT_EVERY (4096) counted requests: tokens per step (1 + accepted / request) and per-position rates per bin.
  int acc_on_ = -1;
  int64_t acc_every_ = 4096, acc_last_ = 0, acc_nopos_ = 0;   // (jY) acc_nopos_: requests without a context position
  int64_t acc_hist_ = 0;                                       // (jZ) requests whose context came from reqs_[rid].hist
  std::array<std::array<std::array<int64_t, 8>, 4>, 4> acc_pos_{};   // [ctx][nreq][pos]
  std::array<std::array<int64_t, 4>, 4> acc_req_{};                  // [ctx][nreq]
  void acc_count_(const Handle* h, const std::vector<StepBatch>& b, const int32_t* const* ns);   // other GPU work (embedded / mixed) between two native decode steps: no gap
  // stats
  std::map<std::string, int64_t> stats_;
};

struct IMixed2b::Ctx {                   // (nd_api.h) one native mixed step, begin .. finish
  std::vector<StepBatch> b;              // the step's batches as the worker gave them (prev-of follows their seq order)
  Engine::MixedBegin mb;
  Engine::MixedHead mh;
  bool headed = false;                   // prepare_head ran
  std::vector<GrammarRows> grammar;      // the step's host grammar rows (masks, acceptable, drafts set)
  bool bm = false;                       //   the step has sampled masks: H1's "|bm" key
  int64_t launched = 0;                  // h1_launch calls
};

}  // namespace nd
}  // namespace fpr
