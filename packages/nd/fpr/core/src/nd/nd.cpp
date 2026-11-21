// nd.cpp -- NativeDecode ND-1: D13._pre's device ops per rank (see nd.h). Each op is the
// Python line it replaces, in the same order on the same stream; comments name it.
#include "nd.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace fpr {
namespace nd {
namespace {

bool ck(cudaError_t e, const char* what, std::string* err) {
  if (e == cudaSuccess) return true;
  if (err) *err = std::string("nd: ") + what + ": " + cudaGetErrorString(e);
  return false;
}
bool wait_ev(cudaEvent_t e, const char* what, std::string* err) {   // ev.synchronize() of the set's previous upload
  return e == nullptr || ck(cudaEventSynchronize(e), what, err);
}
bool h2d(void* dst, const void* src, int64_t bytes, cudaStream_t st, const char* what, std::string* err) {
  return bytes <= 0 || ck(cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyHostToDevice, st), what, err);
}
bool rec(cudaEvent_t e, cudaStream_t st, const char* what, std::string* err) {
  return e == nullptr || ck(cudaEventRecord(e, st), what, err);
}

Status positions_d2h(const PreCommon& c, std::string* err) {
  // ph[:R].copy_(s["pos"], non_blocking=True); oh[:R].copy_(s["own"], non_blocking=True); ev.record(cur)
  const size_t nb = (size_t)c.R * sizeof(int64_t);
  if (c.R > 0) {
    if (!ck(cudaMemcpyAsync(c.ph, c.pos_d, nb, cudaMemcpyDeviceToHost, c.stream), "positions D2H", err)) return Status::INTERNAL;
    if (!ck(cudaMemcpyAsync(c.oh, c.own_d, nb, cudaMemcpyDeviceToHost, c.stream), "own D2H", err)) return Status::INTERNAL;
  }
  return rec(c.pos_ev, c.stream, "d13_ev", err) ? Status::OK : Status::INTERNAL;
}

}  // namespace

Status pre_full(const PreFull& p, const StepBatch& sb, const CompactTables& ct, const d13::D13Host& h, std::string* err) {
  const PreCommon& c = p.c;
  const cudaStream_t st = c.stream;
  const int64_t R = c.R, n = sb.n_rows, S = (int64_t)sb.seqs.size();
  if (n > R || n < 0) {
    if (err) *err = "nd: rows " + std::to_string(n) + " > bucket " + std::to_string(R);
    return Status::BAD_ARG;
  }
  // ---- self._deltas(r, ct): host half (D13Host::deltas = _delta_rows), pinned set dl_i -> dl_d, dbt.index_put_
  if (!p.skip_deltas) {
    d13::Deltas d;
    h.deltas(c.rank, ct, &d);
    if (d.any && d.nu > 0) {
      if (d.direct || d.nu > p.delta_cap) {
        // D13._deltas' direct branch natively (it was a BAD_ARG after async_ok had accepted the step: a
        // fatal dispatch in the 2b loop): dbt.index_put_((torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)),
        // torch.from_numpy(pages).to(dev)) on the step stream: pageable copies (synchronous, as Python's .to(dev)), no pinned
        // set, no dl_ev
        try {
          const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(st, (c10::DeviceIndex)c.device));
          const at::Device dev(at::kCUDA, (c10::DeviceIndex)c.device);
          at::Tensor dbt = at::for_blob(p.dbt, {p.dbt_rows, p.dbt_cols}).strides({p.dbt_stride0, 1}).target_device(dev)
                               .options(at::TensorOptions().dtype(at::kInt).device(dev)).make_tensor();
          const auto cpu64 = at::TensorOptions().dtype(at::kLong), cpu32 = at::TensorOptions().dtype(at::kInt);
          at::Tensor hr = at::from_blob(d.rows.data(), {(int64_t)d.rows.size()}, cpu64);
          at::Tensor hc = at::from_blob(d.cols.data(), {(int64_t)d.cols.size()}, cpu64);
          at::Tensor hp = at::from_blob(d.pages.data(), {(int64_t)d.pages.size()}, cpu32);
          c10::List<std::optional<at::Tensor>> idx;
          idx.push_back(hr.to(dev));
          idx.push_back(hc.to(dev));
          dbt.index_put_(idx, hp.to(dev));
        } catch (const std::exception& e) {
          if (err) *err = std::string("nd: dbt.index_put_ (direct, above DELTA_CAP): ") + e.what();
          return Status::INTERNAL;
        }
        d.any = false;   // (done: skip the staged branch below)
        if (p.n_direct) ++*p.n_direct;
      }
    }
    if (d.any && d.nu > 0) {
      const int64_t nu = d.nu;
      if (!wait_ev(p.dl_ev_wait, "dl_ev", err)) return Status::INTERNAL;
      std::memcpy(p.dl_rows_h, d.rows.data(), nu * sizeof(int64_t));
      std::memcpy(p.dl_cols_h, d.cols.data(), nu * sizeof(int64_t));
      std::memcpy(p.dl_pages_h, d.pages.data(), nu * sizeof(int32_t));
      if (!h2d(p.dl_rows_d, p.dl_rows_h, nu * 8, st, "dl rows", err) || !h2d(p.dl_cols_d, p.dl_cols_h, nu * 8, st, "dl cols", err) ||
          !h2d(p.dl_pages_d, p.dl_pages_h, nu * 4, st, "dl pages", err) || !rec(p.dl_ev, st, "dl_ev", err))
        return Status::INTERNAL;
      try {   // dbt.index_put_((dr[:nu], dc[:nu]), dp[:nu]) on the step stream
        const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(st, (c10::DeviceIndex)c.device));
        const at::Device dev(at::kCUDA, (c10::DeviceIndex)c.device);
        auto mk = [&](void* ptr, at::IntArrayRef sz, at::IntArrayRef strd, at::ScalarType t) {
          return at::for_blob(ptr, sz).strides(strd).target_device(dev).options(at::TensorOptions().dtype(t).device(dev)).make_tensor();
        };
        at::Tensor dbt = mk(p.dbt, {p.dbt_rows, p.dbt_cols}, {p.dbt_stride0, 1}, at::kInt);
        at::Tensor dr = mk(p.dl_rows_d, {nu}, {1}, at::kLong), dc = mk(p.dl_cols_d, {nu}, {1}, at::kLong);
        at::Tensor dp = mk(p.dl_pages_d, {nu}, {1}, at::kInt);
        c10::List<std::optional<at::Tensor>> idx;
        idx.push_back(dr);
        idx.push_back(dc);
        dbt.index_put_(idx, dp);
      } catch (const std::exception& e) {
        if (err) *err = std::string("nd: dbt.index_put_: ") + e.what();
        return Status::INTERNAL;
      }
    }
  }
  // ---- eng._upload_v2(r, s, tab, dev_pos=True): tok / valid / samp into pinned set s["in_i"], ONE copy into s["in_d"]
  {
    if (!wait_ev(p.in_ev_wait, "in_h ev", err)) return Status::INTERNAL;
    int64_t* tok = reinterpret_cast<int64_t*>(p.in_h + p.in_off_tok);
    bool* valid = reinterpret_cast<bool*>(p.in_h + p.in_off_valid);
    int64_t* samp = reinterpret_cast<int64_t*>(p.in_h + p.in_off_samp);
    for (int64_t i = 0; i < R; ++i) tok[i] = 0;                    // v["tok"][:] = 0
    for (int64_t i = 0; i < R; ++i) valid[i] = false;              // v["valid"][:] = False
    for (int64_t i = 0; i < R; ++i) samp[i] = 0;                   // v["samp"][:] = 0
    if (n) {
      if ((int64_t)ct.tokens.size() < n) {
        if (err) *err = "nd: CompactTables.tokens shorter than the batch's rows";
        return Status::BAD_ARG;
      }
      const int64_t neg = p.hsd_on ? 0 : 1000;                      // decode rows: -1 = the device next ids (HSD)
      for (int64_t i = 0; i < n; ++i) tok[i] = ct.tokens[i] < 0 ? neg : ct.tokens[i];
      for (int64_t i = 0; i < n; ++i) valid[i] = true;
      const int64_t ns = (int64_t)ct.sample_idx.size();
      if (ns > R) {
        if (err) *err = "nd: more sample rows than bucket rows";
        return Status::BAD_ARG;
      }
      for (int64_t i = 0; i < ns; ++i) samp[i] = ct.sample_idx[i];
    }
    if (!h2d(p.in_d, p.in_h, p.in_bytes, st, "in_d", err) || !rec(p.in_ev, st, "in_h ev", err)) return Status::INTERNAL;
  }
  // ---- H.idx.upload_d13(r, s, sb): valid / pslot / rs into pinned set d["up_i"], ONE copy into d["up_d"]
  {
    if (n != 4 * S) {
      if (err) *err = "nd: idx2f hook (D13): decode rows only (4 per sequence, sequence-contiguous)";
      return Status::BAD_ARG;
    }
    if (!wait_ev(p.up_ev_wait, "up_h ev", err)) return Status::INTERNAL;
    bool* valid = reinterpret_cast<bool*>(p.up_h + p.up_off_valid);
    int64_t* pslot = reinterpret_cast<int64_t*>(p.up_h + p.up_off_pslot);
    int64_t* rs = reinterpret_cast<int64_t*>(p.up_h + p.up_off_rs);
    for (int64_t i = 0; i < R; ++i) valid[i] = i < n;              // v["valid"][:] = False; [:n] = True
    for (int64_t i = 0; i < p.up_B; ++i) pslot[i] = p.pslot_fill;  // v["pslot"][:] = dbt rows - 1
    for (int64_t k = 0; k < S; ++k) pslot[k] = sb.seqs[k].state_slot;
    for (int64_t i = 0; i < R; ++i) rs[i] = -1;                    // v["rs"][:] = -1; [:n] = repeat(sslot, 4) * 8
    for (int64_t k = 0; k < S; ++k)
      for (int j = 0; j < 4; ++j) rs[4 * k + j] = (int64_t)sb.seqs[k].state_slot * 8;
    if (!h2d(p.up_d, p.up_h, p.up_bytes, st, "up_d", err) || !rec(p.up_ev, st, "up_h ev", err)) return Status::INTERNAL;
  }
  // ---- ENG_D13_OWNER: self._keep(s, "_own_rbase", s["idx"]["rslot"]) (buffer exists: a stream-ordered copy)
  if (p.rbase_d != nullptr &&
      !ck(cudaMemcpyAsync(p.rbase_d, p.rslot_d, (size_t)p.rslot_bytes, cudaMemcpyDeviceToDevice, st), "rbase keep", err))
    return Status::INTERNAL;
  // ---- rx.upload(RowExp.desc_of_compact(ct), stream=cur): pinned set rx._hi -> 2 prefix copies
  {
    d13::Desc d;
    d13::D13Host::desc_of_compact(ct, &d);
    if (!wait_ev(p.rx_ev_wait, "rowexp ev", err)) return Status::INTERNAL;
    int64_t n32 = 0;
    if (d13::DescStaging::pack_into(d, p.S_max, p.ops_max, p.R_max, p.rx_h32, p.rx_h64, &n32, err) != Status::OK)
      return Status::BAD_ARG;
    if (!h2d(p.rx_b32, p.rx_h32, n32 * 4, st, "rowexp b32", err) || !h2d(p.rx_b64, p.rx_h64, 3 * p.S_max * 8, st, "rowexp b64", err) ||
        !rec(p.rx_ev, st, "rowexp ev", err))
      return Status::INTERNAL;
  }
  // ---- ENG_D13_GT: _guard_host (sequences only) + graph T (R, r, has_seqs)
  if (S) {
    if (!wait_ev(p.nb_ev_wait, "nb_ev", err)) return Status::INTERNAL;
    const std::vector<int64_t> nb = h.guard_blocks(sb);
    std::memcpy(p.nb_h, nb.data(), S * sizeof(int64_t));
    if (!h2d(p.nb_d, p.nb_h, S * 8, st, "nb_d", err) || !rec(p.nb_ev, st, "nb_ev", err)) return Status::INTERNAL;
  }
  if (p.gt_exec == nullptr) {
    if (err) *err = "nd: no graph T exec";
    return Status::BAD_ARG;
  }
  if (!ck(cudaGraphLaunch(p.gt_exec, st), "graph T", err)) return Status::INTERNAL;
  // ---- positions D2H + s["d13_ev"]
  const Status ps = positions_d2h(c, err);
  if (ps != Status::OK || !p.engram) return ps;
  // ---- a = EG.step_args_from_compact(ct, H._lookback, pad_to=R); H.engram[r].upload(a, stream=cur)
  {
    const int64_t Sx = (int64_t)ct.slot.size(), T = std::max<int64_t>(ct.n, R);
    if (T > p.eg_max_rows || Sx > p.eg_max_seqs) {
      if (err) *err = "nd: Engram step args exceed EngramPhys max_rows / max_seqs";
      return Status::BAD_ARG;
    }
    for (int64_t k = 0; k < Sx; ++k)
      if (ct.kind[k] != 0 || ct.n_rows[k] > p.eg_ring_w) {   // host lookback rows never reach the async path
        if (err) *err = "nd: Engram row needs a host lookback (prefill / long row) on the native path";
        return Status::BAD_ARG;
      }
    if (!wait_ev(p.eg_ev, "engram stage ev", err)) return Status::INTERNAL;   // the set's previous upload
    int32_t* h = p.eg_stage_h;
    for (int64_t k = 0; k < Sx; ++k) {
      h[p.eg_off_row0 + k] = (int32_t)ct.row0[k];
      h[p.eg_off_nrows + k] = (int32_t)ct.n_rows[k];
      h[p.eg_off_start + k] = (int32_t)ct.start[k];          // np.int64 -> int32 (astype)
      h[p.eg_off_slot + k] = (int32_t)ct.slot[k];
    }
    for (int64_t k = 0; k < 3 * Sx; ++k) h[p.eg_off_look + k] = -2;   // device ring
    h[p.eg_off_counts] = (int32_t)Sx;
    h[p.eg_off_counts + 1] = (int32_t)ct.n;
    const int64_t lo = p.eg_off_row0;                         // no host ids: the copy starts at row0
    if (!h2d(p.eg_pk_d + lo, h + lo, (p.eg_pk_n - lo) * 4, st, "engram pk", err) || !rec(p.eg_ev, st, "engram stage ev", err))
      return Status::INTERNAL;
  }
  return Status::OK;
}

Status pre_owner(const PreOwner& p, std::string* err) {
  if (p.gto_exec != nullptr) {
    if (!ck(cudaGraphLaunch(p.gto_exec, p.c.stream), "graph T'", err)) return Status::INTERNAL;
  } else if (p.tfuse_fn != nullptr && p.tfuse_args != nullptr) {
    const int rc = p.tfuse_fn(p.tfuse_args);
    if (rc != 0) {
      if (err) *err = "nd: d13tail.tail: FPR status " + std::to_string(rc);
      return Status::INTERNAL;
    }
  } else {
    if (err) *err = "nd: owner step without graph T' or fused tail";
    return Status::BAD_ARG;
  }
  return positions_d2h(p.c, err);
}

}  // namespace nd
}  // namespace fpr
