// owner_engram.cpp
// = Hooks.prepare_engram (engine.py): per rank, EG.step_args_from_tables(s["tab"], self._lookback, pad_to=s["R"]) on the A
// bucket, then EngramPhys.upload(a) on the rank's current (= step) stream. The Engram DEVICE prepare (hash / gather / wkv on
// the Engram stream) is NOT this op: Python runs it inside p_embed, after ids32[r][:R].copy_(s["tok"]) (the embed ISSUE);
// natively FPR's embed piece runs it.
// ENGRAM_MARK (FPR_ENGRAM_MARK debug words): prep_run writes NO Engram-stream marks 1..5 (hash / gather / wkv progress), so
// an ENGRAM_MARK dump of a NATIVE step shows only the main-stream marks (6 / 7, after each inject). Missing marks 1..5 on a
// native step do NOT mean Engram never ran; debug an IMA there with the step's native path in mind.
//   phase 1a (prepare_plan): every refusal: handover fields (v0.7 FprEhEngramRank + row 10's engram_pk_d / engram_stage /
//            engram_pk_off), the packed layout, nd's host lookback (FPR_ND_LOOKBACK), the step args of every rank from nd's
//            full tables (fpr::p0::eg_args_from_tables, byte-identical port: tests/eg_check.py), T / S within the layout.
//   phase 1b (prepare): nothing (A's tables come from begin, not from the T build).
//   phase 2  (op PREPARE_ENGRAM): per rank fpr::p0::eg_pack (gen, ENGRAM_GUARD v2's host checks, the staging writes) into
//            this rank's persistent block image, then ONE h2d of [lo, pk_n) through row 10's engram_stage sets into
//            engram_pk_d (nd's FprEhSets rule on the step stream = upload's single non-blocking copy, BISECT engram_stream
//            off). Host-check lines go to stderr as engram.py's _guard_log ("ENGRAM OOB ... rank=r").
// Differences from Python that no kernel reads: the image is one buffer per rank (Python alternates two staging sets with
// their own stale tails beyond S; the kernels read [0, S) per array + counts only); EngramPhys._gen (counts[2], the guard's
// log generation) is this owner's own counter, not synced with Python's (log correlation only).
// Engram off (no engram ranks exported) is a refusal, not a no-op: an unexported view must never skip an upload silently.
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../p0/engram_host.h"
#include "mixed_owner.h"

namespace fpr {
namespace core2b {

namespace {

struct EngramOwnerState {
  int W = 0;
  p0::EgLayout L;
  std::vector<p0::EgHostState> hs;            // per rank: gen + guard / nring (stage_i unused: nd alternates the sets)
  std::vector<std::vector<int32_t>> img;      // per rank: the packed block (persistent across steps)
  std::vector<p0::EgArgs> args;               // this step's, per rank (phase 1a)
};

// row 10's engram_pk_off = EngramPhys._pk_off (ids, row0, nrows, start, slot, look, counts) + pk_n
std::string layout_of(const FprEhRankState& rk, p0::EgLayout* L) {
  const int64_t* o = rk.engram_pk_off;
  const int64_t R = o[1] - o[0], S = o[2] - o[1];
  if (o[0] != 0 || R <= 0 || S <= 0) return "engram_pk_off: no layout (ids 0, row0 R, nrows R + S expected)";
  *L = p0::eg_layout((int32_t)R, (int32_t)S);
  if (o[3] != L->start || o[4] != L->slot || o[5] != L->look || o[6] != L->counts)
    return "engram_pk_off: not EngramPhys's layout [ids R | row0 S | nrows S | start S | slot S | look 3S | counts 4]";
  const int64_t pk_n = o[7] > 0 ? o[7] : rk.engram_pk_d.bytes / (int64_t)sizeof(int32_t);
  if (pk_n != L->pk_n || rk.engram_pk_d.bytes < L->pk_n * (int64_t)sizeof(int32_t)) return "engram_pk_d: size != the layout's pk_n";
  return "";
}

void reg(MixedOwnerRegistry& Rg) {
  auto st = std::make_shared<EngramOwnerState>();

  Rg.prepare_plan("engram", [st](const MixedOwnerCtx& x, const p0::MixedPlan& P) -> std::string {
    const FprEngineHandover* h = x.h;
    if (h == nullptr || x.mx == nullptr || x.c == nullptr) return "no handover / open step";
#ifndef FPR_ND_LOOKBACK
    return "nd has no host lookback (IMixed2b::lookback, FPR_ND_LOOKBACK)";
#else
    if (h->engram_lib == nullptr) return "engram not exported (engram_lib NULL)";
    const int W = (int)P.A.size();
    if (W <= 0 || W > FPR_EH_MAX_RANKS) return "no bucket A ranks";
    const std::vector<RankTables>& tabs = x.mx->tables(x.c);
    if ((int)tabs.size() != W) return "nd's full tables: rank count != bucket A's";
    p0::EgLayout L0;
    for (int r = 0; r < W; ++r) {
      const FprEhEngramRank& e = h->engram[r];
      const FprEhRankState& rk = h->ranks[r];
      if (e.struct_size != (int32_t)sizeof(FprEhEngramRank) || e.pk_d == nullptr) return "engram rank " + std::to_string(r) + " not exported";
      if (rk.engram_pk_d.ptr == nullptr || rk.engram_pk_d.ptr != (void*)e.pk_d) return "row 10 engram_pk_d != engram[r].pk_d";
      if (!rk.engram_stage.h[0].ptr || !rk.engram_stage.h[1].ptr || !rk.engram_stage.ev[0] || !rk.engram_stage.ev[1])
        return "row 10 engram_stage sets / events not exported";
#ifdef FPR_EH_ENGRAM_GUARD_FIELDS
      if (e.guard != 0 && e.nring <= 0) return "engram guard on with nring <= 0";
#else
      return "handover without engram guard / nring (FPR_EH_ENGRAM_GUARD_FIELDS)";
#endif
      p0::EgLayout L;
      const std::string why = layout_of(rk, &L);
      if (!why.empty()) return why;
      if (r == 0) L0 = L;
      else if (L.pk_n != L0.pk_n) return "engram layouts differ across ranks";
    }
    if (st->W != W || st->L.pk_n != L0.pk_n) {   // first step (or a layout change): fresh images, as EngramPhys._stage (zeros)
      st->W = W;
      st->L = L0;
      st->hs.assign((size_t)W, p0::EgHostState());
      st->img.assign((size_t)W, std::vector<int32_t>((size_t)L0.pk_n, 0));
    }
    st->args.assign((size_t)W, p0::EgArgs());
    std::string miss;
    nd::IMixed2b* mx = x.mx;
    const nd::IMixed2b::Ctx* c = x.c;
    p0::EgLookback lb = [mx, c, &miss](const Rid& rid, int64_t start) {   // Hooks._lookback natively (nd's history)
      std::array<int64_t, 3> out{-1, -1, -1};
      int64_t o3[3] = {-1, -1, -1};
      if (!mx->lookback(c, rid, start, o3)) {
        if (miss.empty()) miss = rid + "@" + std::to_string(start);
      } else {
        out = {o3[0], o3[1], o3[2]};
      }
      return out;
    };
    for (int r = 0; r < W; ++r) {
      p0::EgHostState& hs = st->hs[(size_t)r];
#ifdef FPR_EH_ENGRAM_GUARD_FIELDS
      hs.guard = h->engram[r].guard != 0;
      hs.nring = h->engram[r].nring;
#endif
      std::string err;
      if (p0::eg_args_from_tables(tabs[(size_t)r], P.A[(size_t)r].R, lb, &st->args[(size_t)r], &err) != Status::OK)
        return "step args rank " + std::to_string(r) + ": " + err;
      if (!miss.empty()) return "no host history for the lookback of " + miss;
      const p0::EgArgs& a = st->args[(size_t)r];
      if (a.T > st->L.max_rows || (int64_t)a.row0.size() > st->L.max_seqs)
        return "rank " + std::to_string(r) + ": T / S beyond EngramPhys's max_rows / max_seqs";
    }
    return "";
#endif
  });

  Rg.prepare("engram", [](const MixedOwnerCtx&, const p0::MixedPlan&) -> std::string { return ""; });

  Rg.op(p0::MixedOpKind::PREPARE_ENGRAM, [st](const MixedOwnerCtx& x, const p0::MixedOp&, const p0::MixedPlan&,
                                              std::string* err) -> Status {
    for (int r = 0; r < st->W; ++r) {
      std::vector<int32_t>& im = st->img[(size_t)r];
      int32_t* sets[2] = {im.data(), im.data()};
      p0::EgPack pk;
      std::string e;
      if (p0::eg_pack(&st->args[(size_t)r], st->L, &st->hs[(size_t)r], sets, nullptr, &pk, &e) != Status::OK) {
        if (err) *err = "engram: eg_pack rank " + std::to_string(r) + ": " + e;   // phase 1a checked T / S: not expected
        return Status::INTERNAL;
      }
      for (const std::string& ln : pk.logs) std::fprintf(stderr, "ENGRAM OOB %s rank=%d\n", ln.c_str(), r);
      const FprEhRankState& rk = x.h->ranks[r];
      int32_t* dst = static_cast<int32_t*>(rk.engram_pk_d.ptr) + pk.lo;
      const Status s = x.mx->h2d(x.c, r, &rk.engram_stage, dst, im.data() + pk.lo, pk.n_copy * (int64_t)sizeof(int32_t), err);
      if (s != Status::OK) return s;
    }
    return Status::OK;
  });
}

const bool registered = add_mixed_owner("engram", &reg);

}  // namespace
}  // namespace core2b
}  // namespace fpr
