// nd_handover.cpp -- the ENGINE side of fpr/core/src/core2b/engine_handover.h (v0.5)
// v0.5: rank state h1_h_own / rx_own_* / pools[]; graph entries carry aux_dst (H1: one gen slot per key, v3.1); mixed
// A / T views in their own table (generation slots per key text, shared gens[]), buckets_epoch (the eviction notice),
// fill_mixed (no GIL) / ensure_mixed (GIL, exporter); pull returns the embedded step's history deltas.
// v0.3: the C shim that owns the handover's memory, filled by the Python exporter (nd_export.py) inside the embedded engine.
//   fill_bucket / graphs   no GIL, any native thread: copies of the shim's own tables
//   ensure_bucket / push_state / pull_state   take the GIL and call the exporter (loop thread only, rule (a))
// Storage never moves (rule): gens[], the bucket-view slots and the graph entries are sized once at construction; an
// overflow is an error. Graph entries are written fully, then `count` / `version` are published with release order.
#include <torch/extension.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../core2b/engine_handover.h"

static_assert(FPR_EH_VERSION >= 5, "nd_handover.cpp is the v0.5 engine side: build against engine_handover.h v0.5");

namespace py = pybind11;

namespace {

int dt_of(const std::string& s) {
  if (s == "u8" || s == "bool") return FPR_EH_U8;
  if (s == "i32") return FPR_EH_I32;
  if (s == "i64") return FPR_EH_I64;
  if (s == "f32") return FPR_EH_F32;
  if (s == "bf16") return FPR_EH_BF16;
  if (s == "f16") return FPR_EH_F16;
  throw std::runtime_error("nd_handover: dtype " + s);
}

// exporter encodings: tensor = (ptr, dtype, shape, stride, device); buf = (ptr, bytes, device);
// sets = ((ptr, bytes, device), (ptr, bytes, device), ev0, ev1)
FprEhTensor tensor_of(py::handle o) {
  FprEhTensor t{};
  if (o.is_none()) return t;
  py::tuple q = py::reinterpret_borrow<py::tuple>(o);
  t.data = reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(q[0]));
  t.dtype = dt_of(py::cast<std::string>(q[1]));
  auto sh = py::cast<std::vector<int64_t>>(q[2]);
  auto st = py::cast<std::vector<int64_t>>(q[3]);
  if (sh.size() > 4 || sh.size() != st.size()) throw std::runtime_error("nd_handover: tensor rank > 4");
  t.ndim = (int32_t)sh.size();
  for (size_t i = 0; i < sh.size(); ++i) {
    t.shape[i] = sh[i];
    t.stride[i] = st[i];
  }
  t.device = py::cast<int32_t>(q[4]);
  return t;
}
FprEhBuf buf_of(py::handle o) {
  FprEhBuf b{};
  if (o.is_none()) return b;
  py::tuple q = py::reinterpret_borrow<py::tuple>(o);
  b.ptr = reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(q[0]));
  b.bytes = py::cast<int64_t>(q[1]);
  b.device = py::cast<int32_t>(q[2]);
  return b;
}
FprEhSets sets_of(py::handle o) {
  FprEhSets s{};
  if (o.is_none()) return s;
  py::tuple q = py::reinterpret_borrow<py::tuple>(o);
  s.h[0] = buf_of(q[0]);
  s.h[1] = buf_of(q[1]);
  s.ev[0] = reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(q[2]));
  s.ev[1] = reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(q[3]));
  return s;
}
void* ptr_of(py::handle o) { return o.is_none() ? nullptr : reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(o)); }

struct D {                              // a dict with required keys (a missing key is an exporter bug: loud)
  py::dict d;
  py::object operator[](const char* k) const {
    if (!d.contains(k)) throw std::runtime_error(std::string("nd_handover: exporter dict lacks ") + k);
    return d[k];
  }
  int64_t i(const char* k) const { return py::cast<int64_t>((*this)[k]); }
};

class Shim {
 public:
  Shim(int n_ranks, int k, int64_t page_tokens, int64_t window_page_tokens, int64_t vocab, int64_t max_model_len,
       int64_t largest_bucket, int n_gens, int64_t graph_capacity)
      : gens_(n_gens, 0), views_((size_t)n_gens * FPR_EH_MAX_RANKS), have_((size_t)n_gens * FPR_EH_MAX_RANKS, 0),
        mviews_((size_t)n_gens * FPR_EH_MAX_RANKS), mhave_((size_t)n_gens * FPR_EH_MAX_RANKS, 0),
        mptrs_((size_t)n_gens * FPR_EH_MAX_RANKS), mbytes_((size_t)n_gens * FPR_EH_MAX_RANKS),
        graphs_(graph_capacity) {
    if (n_ranks < 1 || n_ranks > FPR_EH_MAX_RANKS) throw std::runtime_error("nd_handover: n_ranks");
    std::memset(&h_, 0, sizeof(h_));
    h_.struct_size = (int32_t)sizeof(FprEngineHandover);
    h_.version = FPR_EH_VERSION;
    h_.n_ranks = n_ranks;
    h_.k = k;
    h_.page_tokens = page_tokens;
    h_.window_page_tokens = window_page_tokens;
    h_.vocab_size = vocab;
    h_.max_model_len = max_model_len;
    h_.largest_bucket = largest_bucket;
    cfg_ = "{}";
    h_.config_json = cfg_.c_str();
    h_.gens = gens_.data();
    h_.n_gens = n_gens;
    h_.ctx = this;
    h_.fill_bucket = &Shim::c_fill_bucket;
    h_.graphs = &Shim::c_graphs;
    h_.ensure_bucket = &Shim::c_ensure_bucket;
    h_.push_state = &Shim::c_push_state;
    h_.pull_state = &Shim::c_pull_state;
    h_.last_error = &Shim::c_last_error;
    h_.buckets_epoch = &epoch_;
    h_.comb_gen = &comb_gen_;   // 0 until the first export
    h_.fill_mixed = &Shim::c_fill_mixed;
    h_.ensure_mixed = &Shim::c_ensure_mixed;
  }

  int64_t ptr() { return (int64_t)(intptr_t)&h_; }
  // 0 = NULL (a Python builder)
  void set_tables(int64_t tb_tail, int64_t tb_d13) {
    h_.tb_tail = reinterpret_cast<void*>((intptr_t)tb_tail);
    h_.tb_d13 = reinterpret_cast<void*>((intptr_t)tb_d13);
  }
  // one rank's indexer ring (tables2f.Stage: eid, n, cap, host / device blocks), EagerWorkspace
  // (f32 / bmax / prep) and Engine.pf_ws; every key optional (absent = zero). Called at step boundaries only (nothing of
  // nd in flight); rewrites ONLY these fields (set_rank's other fields stay).
  void set_rank_p1(int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    FprEhRankState& x = h_.ranks[r];
    x.idx_ring_id = 0;
    x.idx_ring_n = 0;
    x.idx_ring_cap = 0;
    for (int i = 0; i < FPR_EH_IDX_RING_MAX; ++i) x.idx_ring_host[i] = x.idx_ring_dev[i] = nullptr;
    x.ews_f32 = x.ews_bmax = x.ews_prep = x.pf_ws = x.pf_inv = FprEhTensor{};
    if (dd.contains("idx_ring_n")) {
      auto hp = py::cast<std::vector<int64_t>>(dd["idx_ring_host"]);
      auto dp = py::cast<std::vector<int64_t>>(dd["idx_ring_dev"]);
      const int n = py::cast<int>(dd["idx_ring_n"]);
      if (n < 1 || n > FPR_EH_IDX_RING_MAX || (int)hp.size() != n || (int)dp.size() != n)
        throw std::runtime_error("nd_handover: idx ring slots");
      for (int i = 0; i < n; ++i) {
        x.idx_ring_host[i] = reinterpret_cast<void*>((intptr_t)hp[i]);
        x.idx_ring_dev[i] = reinterpret_cast<void*>((intptr_t)dp[i]);
      }
      x.idx_ring_id = py::cast<int64_t>(dd["idx_ring_id"]);
      x.idx_ring_cap = py::cast<int64_t>(dd["idx_ring_cap"]);
      x.idx_ring_n = n;
    }
    if (dd.contains("ews_f32")) {
      x.ews_f32 = tensor_of(dd["ews_f32"]);
      x.ews_bmax = tensor_of(dd["ews_bmax"]);
      x.ews_prep = tensor_of(dd["ews_prep"]);
    }
    if (dd.contains("pf_ws")) x.pf_ws = tensor_of(dd["pf_ws"]);
    if (dd.contains("pf_inv")) x.pf_inv = tensor_of(dd["pf_inv"]);   // (v3.19) eng._nd_bufs[r]["pf_inv"]
  }
  // HSD's grammar mask buffers: _bm_h (pinned) + _bm_d[r] (device), set at init
  void set_hsd_bm(py::object hb, py::list bd) {
    h_.hsd_bm_h = tensor_of(hb);
    if ((int)py::len(bd) > FPR_EH_MAX_RANKS) throw std::runtime_error("nd_handover: hsd_bm_d ranks");
    for (int r = 0; r < FPR_EH_MAX_RANKS; ++r) h_.hsd_bm_d[r] = r < (int)py::len(bd) ? tensor_of(bd[r]) : FprEhTensor{};
  }
  // (v3.19) rank r's CP views (eng._nd_bufs[r]: cp_idx + 2 pinned sets, cp.py's mail stream of the device), set at init
  void set_cp(int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    FprEhCpRank& c = h_.cp[r];
    c.cp_idx = tensor_of(dd["cp_idx"]);
    c.cp_idx_up = sets_of(dd["cp_idx_up"]);
    c.cp_mail_stream = reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(dd["cp_mail_stream"]));
    c.cap = py::cast<int64_t>(dd["cap"]);
  }
  // (v3.18) the combine templates of rank r
  void set_comb(int r, int64_t ptr, int64_t bytes) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    h_.comb_tpl[r] = reinterpret_cast<const void*>((intptr_t)ptr);
    h_.comb_tpl_bytes[r] = bytes;
  }
  void bump_comb_gen() { __atomic_add_fetch(&comb_gen_, 1, __ATOMIC_RELEASE); }
  // (v0.7 slice 5) the ITables ABI of tb_tail / tb_d13's creator (the exporter: cxx_kv.TABLES_ABI; 0 = unknown)
  void set_tables_abi(int32_t v) { h_.tables_abi = v; }
  // one rank's EngramPhys for native PREPARE_ENGRAM (exporter _engram_rank: pointers as ints
  // events as cudaEvent_t ints, lo / hi as 48 ints each = [2][24]); an empty dict = not exported (pk_d NULL: nd refuses)
  void set_engram(int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    FprEhEngramRank& x = h_.engram[r];
    std::memset(&x, 0, sizeof(x));
    x.struct_size = (int32_t)sizeof(FprEhEngramRank);
    x.rank = r;
    if (dd.size() == 0) return;
    const D d{dd};
    auto P = [&](const char* k) { return reinterpret_cast<void*>((intptr_t)d.i(k)); };
    auto P2 = [&](const char* k, int j) { return reinterpret_cast<void*>((intptr_t)py::cast<int64_t>(py::list(d[k])[j])); };
    x.device = (int32_t)d.i("device");
    x.pk_d = static_cast<int32_t*>(P("pk_d"));
    for (int j = 0; j < 2; ++j) {
      x.stage[j] = static_cast<int32_t*>(P2("stage", j));
      x.stage_ev[j] = P2("stage_ev", j);
      x.rows[j] = P2("rows", j);
      x.kv[j] = P2("kv", j);
      x.tabs[j] = P2("tabs", j);
      x.wbf[j] = P2("wbf", j);
      x.ev[j] = P2("ev", j);
    }
    x.hctx = d.i("hctx");
    x.ids = static_cast<const int32_t*>(P("ids"));
    x.tmap = static_cast<const int32_t*>(P("tmap"));
    x.ring = static_cast<int32_t*>(P("ring"));
    x.hash = static_cast<int32_t*>(P("hash"));
    x.keep = static_cast<uint8_t*>(P("keep"));
    x.err = static_cast<int32_t*>(P("err"));
    auto lo = py::cast<std::vector<int64_t>>(d["lo"]), hi = py::cast<std::vector<int64_t>>(d["hi"]);
    if (lo.size() != 48 || hi.size() != 48) throw std::runtime_error("nd_handover: engram lo / hi need 2 x 24");
    for (int l = 0; l < 2; ++l)
      for (int c = 0; c < 24; ++c) x.lo[l][c] = lo[(size_t)l * 24 + c], x.hi[l][c] = hi[(size_t)l * 24 + c];
    x.g_ctas = (int32_t)d.i("g_ctas");
    x.g_threads = (int32_t)d.i("g_threads");
    x.g_unroll = (int32_t)d.i("g_unroll");
    x.wbf_N = d.i("wbf_N");
    x.wbf_K = d.i("wbf_K");
    x.stage_buf = P("stage_buf");
    x.stage_chunks = (int32_t)d.i("stage_chunks");
    x.start_src_i64 = (int32_t)d.i("start_src_i64");
    x.ws_ev = P("ws_ev");
    x.start_src = P("start_src");
    x.w8_max_rows = (int32_t)d.i("w8_max_rows");
#ifdef FPR_EH_ENGRAM_GUARD_FIELDS
    x.guard = (int32_t)d.i("guard");
    x.nring = (int32_t)d.i("nring");
#endif
  }
  // (v0.7 slice 2, B7) one rank's page-copy id staging; {} = not allocated (ids NULL: nd refuses)
  void set_pagecopy(int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    FprEhPageCopyRank& x = h_.pagecopy[r];
    std::memset(&x, 0, sizeof(x));
    if (dd.size() == 0) return;
    const D d{dd};
    x.ids = tensor_of(d["ids"]);
    x.up = sets_of(d["up"]);
    x.cap = d.i("cap");
  }
  // (v0.7 slice 3) the driver's owner kind ids (embed, engram_bd, attn_a, idx, attn_b, moe, -1 = none), at most 8
  void set_span_kinds(const std::vector<int32_t>& k) {
    for (int i = 0; i < 8; ++i) h_.span_kinds[i] = i < (int)k.size() ? k[(size_t)i] : -1;
  }
  // -1 = not registered
  void set_span_kinds_cp(const std::vector<int32_t>& k) {
    for (int i = 0; i < 8; ++i) h_.span_kinds_cp[i] = i < (int)k.size() ? k[(size_t)i] : -1;
  }
  // (v0.7 slice 4) the loaded C ABI libraries by name (FPR_EH_MAX_LIBS); the strings live in the shim
  void set_libs(const std::vector<std::pair<std::string, std::string>>& libs) {
    if (libs.size() > (size_t)FPR_EH_MAX_LIBS) throw std::runtime_error("nd_handover: more libs than FPR_EH_MAX_LIBS");
    lib_paths_.clear();
    lib_paths_.reserve(libs.size());
    std::memset(h_.libs, 0, sizeof(h_.libs));
    for (size_t i = 0; i < libs.size(); ++i) {
      if (libs[i].first.size() >= sizeof(h_.libs[i].name)) throw std::runtime_error("nd_handover: lib name too long");
      std::memcpy(h_.libs[i].name, libs[i].first.c_str(), libs[i].first.size() + 1);
      lib_paths_.push_back(libs[i].second);
    }
    for (size_t i = 0; i < libs.size(); ++i) h_.libs[i].path = lib_paths_[i].c_str();
    h_.n_libs = (int32_t)libs.size();
  }
  // p0::MoeSnap bytes (the exporter keeps the ctypes struct alive); 0 = unavailable
  void set_moe_snap(int64_t ptr, int64_t bytes) {
    h_.moe_snap = reinterpret_cast<const void*>((intptr_t)ptr);
    h_.moe_snap_bytes = bytes;
  }
  void set_engram_lib(const std::string& path) {   // EG.ext().__file__ ("" = no Engram)
    engram_lib_ = path;
    h_.engram_lib = engram_lib_.empty() ? nullptr : engram_lib_.c_str();
  }
  void set_config(const std::string& json) {
    cfg_ = json;
    h_.config_json = cfg_.c_str();
  }
  int64_t sizeof_handover() { return (int64_t)sizeof(FprEngineHandover); }

  void set_rank(int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    const D d{dd};
    FprEhRankState& x = h_.ranks[r];
    std::memset(&x, 0, sizeof(x));
    x.struct_size = (int32_t)sizeof(FprEhRankState);
    x.rank = r;
    x.device = (int32_t)d.i("device");
    x.step_stream = ptr_of(d["step_stream"]);
    x.xstream = ptr_of(d["xstream"]);
    x.side_stream = ptr_of(d["side_stream"]);
    x.estream = ptr_of(d["estream"]);
    x.step_event[0] = ptr_of(d["step_event0"]);
    x.step_event[1] = ptr_of(d["step_event1"]);
    x.rx_b32_d = buf_of(d["rx_b32_d"]);
    x.rx_b64_d = buf_of(d["rx_b64_d"]);
    x.rx_h32 = sets_of(d["rx_h32"]);
    x.rx_h64 = sets_of(d["rx_h64"]);
    auto o32 = py::cast<std::vector<int64_t>>(d["rx_o32"]);
    for (size_t i = 0; i < o32.size() && i < 8; ++i) x.rx_o32[i] = o32[i];
    x.rx_ring = tensor_of(d["rx_ring"]); x.rx_dbt = tensor_of(d["rx_dbt"]); x.rx_nc = tensor_of(d["rx_nc"]);
    x.rx_err = tensor_of(d["rx_err"]); x.rx_counts = tensor_of(d["rx_counts"]); x.rx_start = tensor_of(d["rx_start"]);
    x.rx_seq = tensor_of(d["rx_seq"]);
    x.rx_scratch_slot = d.i("rx_scratch_slot"); x.rx_S_max = d.i("rx_S_max"); x.rx_R_max = d.i("rx_R_max");
    x.rx_ops_max = d.i("rx_ops_max");
    x.dl_rows_d = tensor_of(d["dl_rows_d"]); x.dl_cols_d = tensor_of(d["dl_cols_d"]); x.dl_pages_d = tensor_of(d["dl_pages_d"]);
    x.dl_rows = sets_of(d["dl_rows"]); x.dl_cols = sets_of(d["dl_cols"]); x.dl_pages = sets_of(d["dl_pages"]);
    x.delta_cap = d.i("delta_cap");
    x.nb_d = buf_of(d["nb_d"]); x.nb = sets_of(d["nb"]);
    x.d2h_pos[0] = tensor_of(d["d2h_pos0"]); x.d2h_pos[1] = tensor_of(d["d2h_pos1"]);
    x.d2h_own[0] = tensor_of(d["d2h_own0"]); x.d2h_own[1] = tensor_of(d["d2h_own1"]);
    x.qidx_d = buf_of(d["qidx_d"]); x.qidx = sets_of(d["qidx"]); x.num_sampled = tensor_of(d["num_sampled"]);
    x.engram_pk_d = buf_of(d["engram_pk_d"]); x.engram_stage = sets_of(d["engram_stage"]);
    auto po = py::cast<std::vector<int64_t>>(d["engram_pk_off"]);
    for (size_t i = 0; i < po.size() && i < 8; ++i) x.engram_pk_off[i] = po[i];
    x.engram_b_start = tensor_of(d["engram_b_start"]); x.engram_b_ids = tensor_of(d["engram_b_ids"]);
    x.hsd_dblob = buf_of(d["hsd_dblob"]); x.hsd_hblob = sets_of(d["hsd_hblob"]);
    x.next_ids = tensor_of(d["next_ids"]); x.num_emitted = tensor_of(d["num_emitted"]); x.sampled = tensor_of(d["sampled"]);
    x.next_ids_h = tensor_of(d["next_ids_h"]); x.num_emitted_h = tensor_of(d["num_emitted_h"]);
    x.sampled_h = tensor_of(d["sampled_h"]);
    x.dt_d = tensor_of(d["dt_d"]); x.idx_dbt = tensor_of(d["idx_dbt"]);
    // v0.5
    x.h1_h_own = tensor_of(d["h1_h_own"]);
    if (dd.contains("tail_ix")) {   // v0.6 (ENG_MIX_PERSIST): the tail gather index (None = off)
      x.tail_ix = tensor_of(dd["tail_ix"]);
      if (!py::object(dd["tail_ix_up"]).is_none()) x.tail_ix_up = sets_of(dd["tail_ix_up"]);
    }
    x.rx_own_pos = tensor_of(d["rx_own_pos"]); x.rx_own_own = tensor_of(d["rx_own_own"]);
    x.rx_own_si = tensor_of(d["rx_own_si"]); x.rx_own_sl = tensor_of(d["rx_own_sl"]);
    x.rx_own_paged = tensor_of(d["rx_own_paged"]);
    py::list pl = d["pools"];   // [(name, base, page_bytes, n_pages)] in CopyOp.pool order
    if ((int)py::len(pl) > FPR_EH_MAX_POOLS) throw std::runtime_error("nd_handover: more than FPR_EH_MAX_POOLS pools");
    x.n_pools = (int32_t)py::len(pl);
    for (int i = 0; i < x.n_pools; ++i) {
      py::tuple q = py::reinterpret_borrow<py::tuple>(pl[i]);
      const std::string nm = py::cast<std::string>(q[0]);
      if (nm.size() >= sizeof(x.pools[i].name)) throw std::runtime_error("nd_handover: pool name too long: " + nm);
      std::memcpy(x.pools[i].name, nm.data(), nm.size());
      x.pools[i].base = ptr_of(q[1]);
      x.pools[i].page_bytes = py::cast<int64_t>(q[2]);
      x.pools[i].n_pages = py::cast<int64_t>(q[3]);
    }
  }

  int new_slot() {
    if (next_slot_ >= h_.n_gens) throw std::runtime_error("nd_handover: generation slots full (n_gens fixed)");
    return next_slot_++;
  }
  int h1_slot_of(const std::string& key) {   // (v3.1) an H1 key's generation slot (created on its first entry)
    auto it = h1slot_.find(key);
    if (it != h1slot_.end()) return it->second;
    const int s = new_slot();
    h1slot_[key] = s;
    return s;
  }
  // (v3.1) mixed_cxx_head retired H1 key `key` (its graphs hold pointers the eager head no longer uses): nd skips every
  // entry of it published so far (release). No-op for a key never published.
  void bump_h1(const std::string& key) {
    auto it = h1slot_.find(key);
    if (it == h1slot_.end()) return;
    __atomic_add_fetch(&gens_[it->second], 1, __ATOMIC_RELEASE);
  }
  int slot_of(int64_t R, bool create) {
    auto it = slot_.find(R);
    if (it != slot_.end()) return it->second;
    if (!create) return -1;
    const int s = new_slot();
    slot_[R] = s;
    return s;
  }
  int mslot_of(const std::string& key, bool create) {   // v0.5: a mixed bucket's slot, by its canonical key text
    auto it = mslot_.find(key);
    if (it != mslot_.end()) return it->second;
    if (!create) return -1;
    const int s = new_slot();
    mslot_[key] = s;
    return s;
  }

  // v0.5: a mixed view (exporter, under the GIL, inside ensure_mixed / pull_state: nothing of nd in flight)
  void set_mixed(const std::string& key, int64_t R, int role, int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    if (key.size() >= FPR_EH_KEY) throw std::runtime_error("nd_handover: mixed key too long: " + key);
    const D d{dd};
    const int s = mslot_of(key, true);
    const size_t i = (size_t)s * FPR_EH_MAX_RANKS + r;
    FprEhMixedBucketRank& v = mviews_[i];
    std::memset(&v, 0, sizeof(v));
    v.struct_size = (int32_t)sizeof(FprEhMixedBucketRank);
    v.rank = r;
    v.R = R;
    v.role = role;
    v.moe_kind = (int32_t)d.i("moe_kind");
    v.gen_slot = s;
    v.generation = __atomic_load_n(&gens_[s], __ATOMIC_ACQUIRE);
    std::memcpy(v.key, key.data(), key.size());
    v.fields_hash = (uint64_t)py::cast<uint64_t>(d["fields_hash"]);
    auto& P = mptrs_[i];
    auto& Bv = mbytes_[i];
    P.clear();
    Bv.clear();
    for (auto x : py::list(d["ptrs"])) P.push_back(ptr_of(x));
    for (auto x : py::list(d["bytes"])) Bv.push_back(py::cast<int64_t>(x));
    if (P.size() != Bv.size()) throw std::runtime_error("nd_handover: mixed view ptrs / bytes differ in length");
    v.n_ptrs = (int32_t)P.size();
    v.ptrs = P.data();
    v.bytes = Bv.data();
    v.in_d = buf_of(d["in_d"]); v.in = sets_of(d["in"]);
    v.in_off_tok = d.i("in_off_tok"); v.in_off_valid = d.i("in_off_valid"); v.in_off_samp = d.i("in_off_samp");
    v.h = tensor_of(d["h"]); v.aux = tensor_of(d["aux"]);
    if (dd.contains("span") && !py::object(dd["span"]).is_none()) {   // v0.7 slice 3 (C2): FPR's s["_cxx_span"]
      py::dict sp = py::reinterpret_borrow<py::dict>(dd["span"]);
      auto rows = py::cast<std::vector<std::vector<int64_t>>>(sp["rows"]);   // [n][13]: FprEhSpanLayer's fields in order
      if (rows.size() > (size_t)FPR_EH_MAX_LAYERS) throw std::runtime_error("nd_handover: span table > FPR_EH_MAX_LAYERS");
      v.span.built = py::cast<int32_t>(sp["built"]);
      v.span.n_layers = (int32_t)rows.size();
      v.span.embed = py::cast<int64_t>(sp["embed"]);
      v.span.final_ = py::cast<int64_t>(sp["final"]);
      v.span.fl_rsf = (uint8_t)py::cast<int>(sp["fl_rsf"]);
      for (size_t i = 0; i < rows.size(); ++i) {
        const auto& q = rows[i];
        if (q.size() < 13) throw std::runtime_error("nd_handover: span row < 13 columns");   // (13: before slice 4)
        FprEhSpanLayer& L = v.span.layers[i];
        L.layer = (int32_t)q[0];
        L.bd0 = q[1]; L.engram_bd = q[2]; L.attn_a = q[3]; L.idx = q[4]; L.attn_b = q[5]; L.bd1 = q[6];
        L.router = q[7]; L.router_ce = q[8]; L.ag = q[9]; L.moe = q[10]; L.rs = q[11]; L.rs_fl = q[12];
        L.moe_rsf = q.size() > 13 ? q[13] : -1;
        L.attn_b_pf = q.size() > 14 ? q[14] : -1;
      }
    }
    if (dd.contains("tkeys")) {   // v0.7 slice 4: FPR's FprTgKeys bytes (engine-owned) + the moe bucket
      v.tkeys = reinterpret_cast<const void*>((intptr_t)py::cast<int64_t>(dd["tkeys"]));
      v.tkeys_bytes = py::cast<int64_t>(dd["tkeys_bytes"]);
      v.moe_bucket = reinterpret_cast<const void*>((intptr_t)py::cast<int64_t>(dd["moe_bucket"]));
      v.moe_bucket_bytes = py::cast<int64_t>(dd["moe_bucket_bytes"]);
    }
    if (dd.contains("plan_slots")) {   // (slice 5b) s["_cxx_attn_slots"]: int64 numpy, engine-owned (the exporter keeps it)
      v.plan_slots = reinterpret_cast<const int64_t*>((intptr_t)py::cast<int64_t>(dd["plan_slots"]));
      v.n_plan_slots = py::cast<int32_t>(dd["n_plan_slots"]);
    }
    if (dd.contains("vidx_d")) {   // v0.6 (ENG_MIX_PERSIST): Engine._set_vidx's persistent buffers (None = off)
      v.vidx_d = buf_of(dd["vidx_d"]);
      v.vidx = tensor_of(dd["vidx"]); v.nv = tensor_of(dd["nv"]); v.nd = tensor_of(dd["nd"]);
      if (!py::object(dd["vidx_up"]).is_none()) v.vidx_up = sets_of(dd["vidx_up"]);
    }
    // the view's GeneralTables: Bc, its idx2f_tab hid, the 16 registered tensors in creation order
    // t2r_ir, lat and the lean OwnedIndexer's cand / topk / slow. Absent / None = zero (owner_p1 refuses). pf: not exported.
    if (dd.contains("idx") && !py::object(dd["idx"]).is_none()) {
      py::dict q = py::reinterpret_borrow<py::dict>(dd["idx"]);
      FprEhMixedIdx& x = v.idx;
      x.Bc = py::cast<int64_t>(q["Bc"]);
      x.tab_hid = py::cast<int64_t>(q["tab_hid"]);
      py::list t16 = q["t16"];
      if (py::len(t16) != 16) throw std::runtime_error("nd_handover: idx needs the 16 GeneralTables tensors");
      FprEhTensor* dst[16] = {&x.pos, &x.dbt, &x.sl1, &x.sl2, &x.cs1, &x.cs2, &x.rs, &x.valid_ir, &x.t2r, &x.qsl,
                              &x.ir2row_c, &x.vir, &x.vrow, &x.row2ir_d, &x.nb_lens, &x.bt};
      for (int k = 0; k < 16; ++k) *dst[k] = tensor_of(t16[k]);
      x.t2r_ir = tensor_of(q["t2r_ir"]);
      x.lat = tensor_of(q["lat"]);
      x.cand = tensor_of(q["cand"]);
      x.topk = tensor_of(q["topk"]);
      x.slow = tensor_of(q["slow"]);
    }
    if (dd.contains("pf") && !py::object(dd["pf"]).is_none()) {   // (v3.19) A: the rank's ONE pf set (eng._nd_bufs[r]["pf"])
      py::dict q = py::reinterpret_borrow<py::dict>(dd["pf"]);
      FprEhMixedPf& f = v.pf;
      f.S_max = py::cast<int64_t>(q["S_max"]);
      f.bt_cols_max = py::cast<int64_t>(q["bt_cols_max"]);
      f.wbt_cols_max = py::cast<int64_t>(q["wbt_cols_max"]);
      f.blob = buf_of(q["blob"]);
      f.up = sets_of(q["up"]);
      f.bt = tensor_of(q["bt"]); f.wbt = tensor_of(q["wbt"]); f.loc = tensor_of(q["loc"]);
      f.comb640 = tensor_of(q["comb640"]); f.clen640 = tensor_of(q["clen640"]);
      f.comb128 = tensor_of(q["comb128"]); f.clen128 = tensor_of(q["clen128"]);
    }
    if (dd.contains("idxa_gst") && !py::object(dd["idxa_gst"]).is_none()) {
      v.idxa_gst = tensor_of(dd["idxa_gst"]);
      v.idxa_gst_up = sets_of(dd["idxa_gst_up"]);
    }
    mhave_[i] = 1;
  }
  // v0.5: the engine created / evicted / re-carved mixed bucket `key`: bump its generation (if nd ever saw it) + the epoch
  void bump_mixed(const std::string& key) {
    const int s = mslot_of(key, false);
    if (s >= 0) {
      __atomic_add_fetch(&gens_[s], 1, __ATOMIC_RELEASE);
      for (int r = 0; r < FPR_EH_MAX_RANKS; ++r) mhave_[(size_t)s * FPR_EH_MAX_RANKS + r] = 0;
    }
    __atomic_add_fetch(&epoch_, 1, __ATOMIC_RELEASE);
  }
  void bump_epoch() { __atomic_add_fetch(&epoch_, 1, __ATOMIC_RELEASE); }
  uint64_t epoch() { return __atomic_load_n(&epoch_, __ATOMIC_ACQUIRE); }
  std::vector<std::string> mixed_keys() {
    std::vector<std::string> o;
    for (auto& kv : mslot_) o.push_back(kv.first);
    return o;
  }

  // a bucket view (exporter, under the GIL, at ensure_bucket): the slot's current generation goes into the view
  void set_bucket(int64_t R, int r, py::dict dd) {
    if (r < 0 || r >= h_.n_ranks) throw std::runtime_error("nd_handover: rank");
    const D d{dd};
    const int s = slot_of(R, true);
    FprEhBucketRank& v = views_[(size_t)s * FPR_EH_MAX_RANKS + r];
    std::memset(&v, 0, sizeof(v));
    v.struct_size = (int32_t)sizeof(FprEhBucketRank);
    v.rank = r;
    v.R = R;
    v.role = FPR_EH_ROLE_DECODE;
    v.moe_kind = (int32_t)d.i("moe_kind");
    v.gen_slot = s;
    v.generation = __atomic_load_n(&gens_[s], __ATOMIC_ACQUIRE);
    v.in_d = buf_of(d["in_d"]); v.in = sets_of(d["in"]);
    v.in_off_tok = d.i("in_off_tok"); v.in_off_valid = d.i("in_off_valid"); v.in_off_samp = d.i("in_off_samp");
    v.up_d = buf_of(d["up_d"]); v.up = sets_of(d["up"]);
    v.up_off_valid = d.i("up_off_valid"); v.up_off_pslot = d.i("up_off_pslot"); v.up_off_rs = d.i("up_off_rs");
    v.up_B = d.i("up_B");
    v.pos = tensor_of(d["pos"]); v.own = tensor_of(d["own"]); v.si = tensor_of(d["si"]); v.sl = tensor_of(d["sl"]);
    v.paged = tensor_of(d["paged"]); v.tok = tensor_of(d["tok"]);
    v.idx_rslot = tensor_of(d["idx_rslot"]); v.idx_valid = tensor_of(d["idx_valid"]);
    v.idx_seq_lens[0] = tensor_of(d["idx_seq_lens1"]); v.idx_seq_lens[1] = tensor_of(d["idx_seq_lens2"]);
    v.idx_cslot[0] = tensor_of(d["idx_cslot1"]); v.idx_cslot[1] = tensor_of(d["idx_cslot2"]);
    v.own_rbase = tensor_of(d["own_rbase"]); v.own_sbase = tensor_of(d["own_sbase"]);
    v.pos_ev[0] = ptr_of(d["pos_ev0"]); v.pos_ev[1] = ptr_of(d["pos_ev1"]);
    have_[(size_t)s * FPR_EH_MAX_RANKS + r] = 1;
  }

  // the engine invalidated bucket R's memory (rule (a) (i) / (ii)): bump its generation (release)
  void bump(int64_t R) {
    const int s = slot_of(R, false);
    if (s < 0) return;
    __atomic_add_fetch(&gens_[s], 1, __ATOMIC_RELEASE);
    for (int r = 0; r < FPR_EH_MAX_RANKS; ++r) have_[(size_t)s * FPR_EH_MAX_RANKS + r] = 0;
  }

  // a graph entry, written fully into the next free slot; visible only after publish()
  void add_graph(int kind, int r, int64_t R, int64_t n, int piece, int64_t exec, const std::string& key, int64_t aux_dst) {
    if (pending_ >= (int64_t)graphs_.size()) throw std::runtime_error("nd_handover: graph table full (capacity fixed)");
    if (key.size() >= FPR_EH_KEY) throw std::runtime_error("nd_handover: graph key too long: " + key);
    FprEhGraph& g = graphs_[pending_];
    std::memset(&g, 0, sizeof(g));
    g.kind = kind;
    g.rank = r;
    g.R = R;
    g.n = n;
    g.piece = piece;
    if (kind == FPR_EH_GRAPH_H1) {   // v0.5: over HSD's own buffers, bucket-independent: R = -1
      if (aux_dst == 0) throw std::runtime_error("nd_handover: H1 entry without aux_dst: " + key);
      // one generation slot per H1 key: the exporter bumps it when mixed_cxx_head's pointer guard
      // RETIRES the key (bump_h1), so nd skips the entry (F) until a re-capture publishes a new one at the new generation
      const int s = h1_slot_of(key);
      g.gen_slot = s;
      g.generation = __atomic_load_n(&gens_[s], __ATOMIC_ACQUIRE);
    } else {
      const int s = slot_of(R, true);
      g.gen_slot = s;
      g.generation = __atomic_load_n(&gens_[s], __ATOMIC_ACQUIRE);
    }
    g.exec = reinterpret_cast<void*>((intptr_t)exec);
    g.aux_dst = reinterpret_cast<void*>((intptr_t)aux_dst);
    std::memcpy(g.key, key.data(), key.size());
    ++pending_;
  }
  void publish() {
    __atomic_store_n(&count_, pending_, __ATOMIC_RELEASE);
    __atomic_add_fetch(&version_, 1, __ATOMIC_RELEASE);
  }
  int64_t graph_count() { return __atomic_load_n(&count_, __ATOMIC_ACQUIRE); }

  void set_callbacks(py::object ensure, py::object push, py::object pull, py::object ensure_mixed) {
    ensure_ = ensure;
    push_ = push;
    pull_ = pull;
    ensure_mixed_ = ensure_mixed;
  }

  // test / diagnostics: the C view as the loop would get it
  py::dict view(int64_t R, int r) {
    FprEhBucketRank v;
    py::dict o;
    const int rc = c_fill_bucket(this, R, r, &v);
    o["rc"] = rc;
    if (rc) return o;
    o["R"] = v.R; o["gen_slot"] = v.gen_slot; o["generation"] = v.generation; o["current"] = fpr_eh_current(&h_, &v);
    o["in_d"] = (int64_t)(intptr_t)v.in_d.ptr; o["in_bytes"] = v.in_d.bytes; o["pos"] = (int64_t)(intptr_t)v.pos.data;
    o["up_B"] = v.up_B; o["own_rbase"] = (int64_t)(intptr_t)v.own_rbase.data;
    return o;
  }
  py::dict mixed_view(int64_t R, int role, const std::string& key, int r) {
    FprEhMixedBucketRank v;
    py::dict o;
    const int rc = c_fill_mixed(this, R, role, key.c_str(), r, &v);
    o["rc"] = rc;
    if (rc) {
      o["err"] = err_;
      return o;
    }
    o["R"] = v.R; o["role"] = v.role; o["gen_slot"] = v.gen_slot; o["generation"] = v.generation;
    o["current"] = fpr_eh_mixed_current(&h_, &v); o["key"] = std::string(v.key); o["n_ptrs"] = v.n_ptrs;
    o["fields_hash"] = v.fields_hash; o["in_d"] = (int64_t)(intptr_t)v.in_d.ptr; o["in_bytes"] = v.in_d.bytes;
    o["h"] = (int64_t)(intptr_t)v.h.data; o["aux"] = (int64_t)(intptr_t)v.aux.data;
    return o;
  }
  py::list graph_entries() {   // diagnostics: every published entry (kind, rank, R, n, piece, gen_slot, key, aux_dst)
    py::list o;
    const int64_t n = graph_count();
    for (int64_t i = 0; i < n; ++i) {
      const FprEhGraph& g = graphs_[i];
      o.append(py::make_tuple(g.kind, g.rank, g.R, g.n, g.piece, g.gen_slot, std::string(g.key),
                              (int64_t)(intptr_t)g.aux_dst));
    }
    return o;
  }

 private:
  static int32_t c_fill_bucket(void* ctx, int64_t R, int32_t rank, FprEhBucketRank* out) {
    Shim* s = static_cast<Shim*>(ctx);
    if (rank < 0 || rank >= s->h_.n_ranks || out == nullptr) return s->fail("fill_bucket: bad rank / out");
    auto it = s->slot_.find(R);   // (slots are added only under ensure_bucket: loop thread, nothing in flight)
    if (it == s->slot_.end() || !s->have_[(size_t)it->second * FPR_EH_MAX_RANKS + rank])
      return s->fail("fill_bucket: bucket " + std::to_string(R) + " not ensured (call ensure_bucket on the loop thread)");
    *out = s->views_[(size_t)it->second * FPR_EH_MAX_RANKS + rank];
    return 0;
  }
  static int32_t c_fill_mixed(void* ctx, int64_t R, int32_t role, const char* key, int32_t rank, FprEhMixedBucketRank* out) {
    Shim* s = static_cast<Shim*>(ctx);
    if (rank < 0 || rank >= s->h_.n_ranks || out == nullptr || key == nullptr) return s->fail("fill_mixed: bad rank / out / key");
    auto it = s->mslot_.find(key);   // (slots are added only under ensure_mixed / pull_state: nothing of nd in flight)
    const size_t i = it == s->mslot_.end() ? 0 : (size_t)it->second * FPR_EH_MAX_RANKS + rank;
    if (it == s->mslot_.end() || !s->mhave_[i])
      return s->fail(std::string("fill_mixed: bucket ") + key + " not ensured (call ensure_mixed on the loop thread)");
    const FprEhMixedBucketRank& v = s->mviews_[i];
    if (v.R != R || v.role != role)
      return s->fail(std::string("fill_mixed: key ") + key + " is R " + std::to_string(v.R) + " role " + std::to_string(v.role));
    *out = v;
    return 0;
  }
  static int32_t c_ensure_mixed(void* ctx, int32_t n, const FprEhMixedKey* keys) {
    Shim* s = static_cast<Shim*>(ctx);
    py::gil_scoped_acquire g;
    try {
      py::list ks;
      for (int32_t i = 0; i < n; ++i) ks.append(py::make_tuple(keys[i].R, keys[i].role, std::string(keys[i].key)));
      const int rc = py::cast<int>(s->ensure_mixed_(ks));
      if (rc == 1) {
        s->err_ = "ensure_mixed: the engine refused (no room): run the step embedded";
        return 1;
      }
      if (rc != 0) return s->fail("ensure_mixed: status " + std::to_string(rc));
    } catch (const std::exception& e) {
      return s->fail(std::string("ensure_mixed: ") + e.what());
    }
    return 0;
  }
  static int32_t c_graphs(void* ctx, FprEhGraphTable* out) {
    Shim* s = static_cast<Shim*>(ctx);
    out->version = __atomic_load_n(&s->version_, __ATOMIC_ACQUIRE);
    out->count = __atomic_load_n(&s->count_, __ATOMIC_ACQUIRE);
    out->capacity = (int64_t)s->graphs_.size();
    out->entries = s->graphs_.data();
    return 0;
  }
  static int32_t c_ensure_bucket(void* ctx, int64_t R) {
    Shim* s = static_cast<Shim*>(ctx);
    py::gil_scoped_acquire g;
    try {
      s->ensure_(R);
    } catch (const std::exception& e) {
      return s->fail(std::string("ensure_bucket: ") + e.what());
    }
    return 0;
  }
  static int32_t c_push_state(void* ctx, const FprEhSetState* st) {
    Shim* s = static_cast<Shim*>(ctx);
    py::gil_scoped_acquire g;
    try {
      s->push_(state_dict(s->h_, st));
    } catch (const std::exception& e) {
      return s->fail(std::string("push_state: ") + e.what());
    }
    return 0;
  }
  static int32_t c_pull_state(void* ctx, FprEhSetState* st) {
    Shim* s = static_cast<Shim*>(ctx);
    py::gil_scoped_acquire g;
    try {
      py::dict d = s->pull_().cast<py::dict>();
      s->state_from(d, st);
    } catch (const std::exception& e) {
      return s->fail(std::string("pull_state: ") + e.what());
    }
    return 0;
  }
  static const char* c_last_error(void* ctx) { return static_cast<Shim*>(ctx)->err_.c_str(); }

  static py::dict state_dict(const FprEngineHandover& h, const FprEhSetState* st) {
    py::dict d;
    py::list R, in_i, up_i;
    for (int b = 0; b < st->n_buckets; ++b) {
      R.append(st->bucket_R[b]);
      py::list a, u;
      for (int r = 0; r < h.n_ranks; ++r) {
        a.append((int)st->in_i[(size_t)b * h.n_ranks + r]);
        u.append((int)st->up_i[(size_t)b * h.n_ranks + r]);
      }
      in_i.append(a);
      up_i.append(u);
    }
    d["bucket_R"] = R; d["in_i"] = in_i; d["up_i"] = up_i;
    py::list rx, eg;
    for (int r = 0; r < h.n_ranks; ++r) {
      rx.append((int)st->rx_hi[r]);
      eg.append((int)st->engram_i[r]);
    }
    d["rx_hi"] = rx; d["engram_i"] = eg;
    py::list egg;                  // (v0.7) EngramPhys._gen
    for (int r = 0; r < h.n_ranks; ++r) egg.append(py::int_(st->engram_gen[r]));
    d["engram_gen"] = egg;
    d["dl_i"] = (int)st->dl_i; d["d2h_i"] = (int)st->d2h_i; d["qi"] = (int)st->qi; d["hsd_hi"] = (int)st->hsd_hi;
    py::list prev;   // (rid, rank, index, producing step serial)
    for (int64_t i = 0; i < st->n_prev; ++i)
      prev.append(py::make_tuple(std::string(st->prev[i].rid), st->prev[i].rank, st->prev[i].index, st->prev[i].serial));
    d["prev"] = prev;
    d["step_serial"] = st->step_serial;
    py::list hist;   // (rid, [tokens]); (rid, None) = dropped on a native step (n < 0, nd_hist.h: on_ctx's `dropped`)
    for (int64_t i = 0; i < st->n_hist; ++i) {
      if (st->hist[i].n < 0) {
        hist.append(py::make_tuple(std::string(st->hist[i].rid), py::none()));
        continue;
      }
      py::list t;
      for (int64_t k = 0; k < st->hist[i].n; ++k) t.append(st->hist[i].tokens[k]);
      hist.append(py::make_tuple(std::string(st->hist[i].rid), t));
    }
    d["hist"] = hist;
    return d;
  }
  // pull: every array points into this shim's storage (valid until the next push / pull, v0.3 header)
  void state_from(py::dict d, FprEhSetState* st) {
    const int W = h_.n_ranks;
    pull_R_ = py::cast<std::vector<int64_t>>(d["bucket_R"]);
    auto in_i = py::cast<std::vector<std::vector<int>>>(d["in_i"]);
    auto up_i = py::cast<std::vector<std::vector<int>>>(d["up_i"]);
    pull_in_.assign(pull_R_.size() * W, 0);
    pull_up_.assign(pull_R_.size() * W, 0);
    for (size_t b = 0; b < pull_R_.size(); ++b)
      for (int r = 0; r < W; ++r) {
        pull_in_[b * W + r] = (uint8_t)in_i[b][r];
        pull_up_[b * W + r] = (uint8_t)up_i[b][r];
      }
    st->struct_size = (int32_t)sizeof(FprEhSetState);
    st->n_buckets = (int32_t)pull_R_.size();
    st->bucket_R = pull_R_.data();
    st->in_i = pull_in_.data();
    st->up_i = pull_up_.data();
    auto rx = py::cast<std::vector<int>>(d["rx_hi"]);
    auto eg = py::cast<std::vector<int>>(d["engram_i"]);
    for (int r = 0; r < W; ++r) {
      st->rx_hi[r] = (uint8_t)rx[r];
      st->engram_i[r] = (uint8_t)eg[r];
    }
    if (d.contains("engram_gen")) {   // (v0.7) EngramPhys._gen (0 without ENGRAM_GUARD v2)
      auto gg = py::cast<std::vector<uint64_t>>(d["engram_gen"]);
      for (int r = 0; r < W && r < (int)gg.size(); ++r) st->engram_gen[r] = gg[r];
    }
    st->dl_i = (uint8_t)py::cast<int>(d["dl_i"]);
    st->d2h_i = (uint8_t)py::cast<int>(d["d2h_i"]);
    st->qi = (uint8_t)py::cast<int>(d["qi"]);
    st->hsd_hi = (uint8_t)py::cast<int>(d["hsd_hi"]);
    pull_rids_.clear();
    pull_prev_.clear();
    py::list prev = d["prev"];
    pull_rids_.reserve(py::len(prev));   // (c_str pointers stay valid: no reallocation after reserve)
    for (auto x : prev) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(x);
      pull_rids_.push_back(py::cast<std::string>(t[0]));
    }
    size_t i = 0;
    for (auto x : prev) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(x);
      pull_prev_.push_back(FprEhPrevOf{pull_rids_[i++].c_str(), py::cast<int64_t>(t[1]), py::cast<int64_t>(t[2]), py::cast<int64_t>(t[3])});
    }
    st->prev = pull_prev_.data();
    st->n_prev = (int64_t)pull_prev_.size();
    st->step_serial = py::cast<int64_t>(d["step_serial"]);
    // v0.5: the tokens the embedded step appended to eng.hist, per rid live on both sides (engine-owned until next call)
    py::list hl = d["hist"];
    pull_hrids_.clear();
    pull_htoks_.clear();
    pull_hist_.clear();
    pull_hrids_.reserve(py::len(hl));
    pull_htoks_.reserve(py::len(hl));
    for (auto x : hl) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(x);
      pull_hrids_.push_back(py::cast<std::string>(t[0]));
      pull_htoks_.push_back(py::cast<std::vector<int32_t>>(t[1]));
    }
    for (size_t k = 0; k < pull_hrids_.size(); ++k)
      pull_hist_.push_back(FprEhHist{pull_hrids_[k].c_str(), pull_htoks_[k].data(), (int64_t)pull_htoks_[k].size()});
    st->hist = pull_hist_.empty() ? nullptr : pull_hist_.data();
    st->n_hist = (int64_t)pull_hist_.size();
  }

  int32_t fail(const std::string& e) {
    err_ = e;
    return -1;
  }

  FprEngineHandover h_;
  std::vector<uint64_t> gens_;
  std::vector<FprEhBucketRank> views_;
  std::vector<uint8_t> have_;
  std::vector<FprEhMixedBucketRank> mviews_;   // v0.5: per (mixed slot, rank)
  std::vector<uint8_t> mhave_;
  std::vector<std::vector<const void*>> mptrs_;
  std::vector<std::vector<int64_t>> mbytes_;
  std::vector<FprEhGraph> graphs_;
  std::map<int64_t, int> slot_;
  std::map<std::string, int> h1slot_;                      // (v3.1) H1 key text -> generation slot
  std::map<std::string, int> mslot_;
  int next_slot_ = 0;
  uint64_t epoch_ = 0;
  uint64_t comb_gen_ = 0;   // (v3.18) bumped (release) by the exporter after comb_tpl[] changed
  int64_t pending_ = 0, count_ = 0;
  uint64_t version_ = 0;
  py::object ensure_, push_, pull_, ensure_mixed_;
  std::string err_;
  std::string cfg_;
  std::string engram_lib_;
  std::vector<std::string> lib_paths_;
  std::vector<int64_t> pull_R_;
  std::vector<uint8_t> pull_in_, pull_up_;
  std::vector<std::string> pull_rids_;
  std::vector<FprEhPrevOf> pull_prev_;
  std::vector<std::string> pull_hrids_;
  std::vector<std::vector<int32_t>> pull_htoks_;
  std::vector<FprEhHist> pull_hist_;
};

}  // namespace

void nd_handover_register(py::module_& m) {
  py::class_<Shim>(m, "Handover")
      .def(py::init<int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int, int64_t>())
      .def("ptr", &Shim::ptr, "the FprEngineHandover* the C++ loop reads (lives as long as this object)")
      .def("sizeof_handover", &Shim::sizeof_handover)
      .def("set_config", &Shim::set_config)
      .def("set_engram", &Shim::set_engram, "(v0.7) set_engram(rank, dict): one rank's EngramPhys ({} = not exported)")
      .def("set_engram_lib", &Shim::set_engram_lib)
      .def("set_pagecopy", &Shim::set_pagecopy, "(v0.7) set_pagecopy(rank, dict(ids, up, cap)) ({} = not allocated)")
      .def("set_libs", &Shim::set_libs, "(v0.7) set_libs([(name, path), ...])")
      .def("set_moe_snap", &Shim::set_moe_snap)
      .def("set_span_kinds", &Shim::set_span_kinds)
      .def("set_span_kinds_cp", &Shim::set_span_kinds_cp, "(slice 6) span_kinds_cp[8]")
      .def("set_tables", &Shim::set_tables, "(v0.7) set_tables(tb_tail ITables* or 0, tb_d13 ITables* or 0)")
      .def("set_tables_abi", &Shim::set_tables_abi, "(v0.7 slice 5) the creator's FPR_TABLES_ABI (0 = unknown)")
      .def("set_rank_p1", &Shim::set_rank_p1, "(v3.18) a rank's P1 fields: idx ring, EagerWorkspace, pf_ws")
      .def("set_comb", &Shim::set_comb, "(v3.18) comb_tpl[r] / comb_tpl_bytes[r]")
      .def("set_cp", &Shim::set_cp, "(v3.19) cp[r]: cp_idx, cp_idx_up, cp_mail_stream, cap")
      .def("set_hsd_bm", &Shim::set_hsd_bm, "(v3.25) hsd_bm_h + hsd_bm_d[r] (HSD's grammar masks)")
      .def("bump_comb_gen", &Shim::bump_comb_gen, "(v3.18) *comb_gen += 1 (release)")
      .def("set_rank", &Shim::set_rank)
      .def("set_bucket", &Shim::set_bucket)
      .def("bump", &Shim::bump)
      .def("add_graph", &Shim::add_graph, py::arg("kind"), py::arg("r"), py::arg("R"), py::arg("n"), py::arg("piece"),
           py::arg("exec"), py::arg("key"), py::arg("aux_dst") = 0)
      .def("set_mixed", &Shim::set_mixed)
      .def("bump_mixed", &Shim::bump_mixed)
      .def("bump_h1", &Shim::bump_h1)
      .def("bump_epoch", &Shim::bump_epoch)
      .def("epoch", &Shim::epoch)
      .def("mixed_keys", &Shim::mixed_keys)
      .def("mixed_view", &Shim::mixed_view)
      .def("graph_entries", &Shim::graph_entries)
      .def("publish", &Shim::publish)
      .def("graph_count", &Shim::graph_count)
      .def("set_callbacks", &Shim::set_callbacks)
      .def("view", &Shim::view);
}
