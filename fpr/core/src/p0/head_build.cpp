// head_build.cpp -- see head_build.h
#include "head_build.h"

#include <algorithm>

namespace fpr {
namespace p0 {

const char* head_status_name(HeadStatus s) {
  switch (s) {
    case HeadStatus::OK: return "ok";
    case HeadStatus::SKIP_GATE: return "skip: gate-only";
    case HeadStatus::REFUSED_BITMASK: return "refused: bitmask";
  }
  return "?";
}

const char* head_src_name(HeadSrc s) {
  switch (s) {
    case HeadSrc::SAMPLING: return "sampling";
    case HeadSrc::CTX: return "ctx";
    case HeadSrc::DEFAULT: return "default";
  }
  return "?";
}

HeadOut build_prepare_head(const HeadSnap& snap, const std::vector<StepBatch>& b,
                           const std::vector<const std::vector<int64_t>*>& tokens) {
  HeadOut out;
  if (!snap.hsd || snap.gate_only) {
    out.status = HeadStatus::SKIP_GATE;
    return out;
  }
  if (snap.grammar_any && !snap.bm_ok) {   // (= Hooks._bitmasks: a rank with sample rows holds a grammar sequence)
    for (const StepBatch& sb : b) {
      if (sb.sample_rows.empty()) continue;
      for (const SeqEntry& se : sb.seqs) {
        auto it = snap.rids.find(se.rid);
        if (it != snap.rids.end() && it->second.has_grammar) {
          out.status = HeadStatus::REFUSED_BITMASK;
          return out;
        }
      }
    }
  }
  for (size_t r = 0; r < b.size(); ++r) {
    const std::vector<int64_t>* tl = r < tokens.size() ? tokens[r] : nullptr;
    for (const SeqEntry& se : b[r].seqs) {
      const HeadRid& e = snap.rids.at(se.rid);
      HeadState st;
      if (e.has_sampling) {
        st.v = e.sampling;
        st.src = HeadSrc::SAMPLING;
      } else if (e.has_ctx) {
        st.v = e.ctx;
        st.src = HeadSrc::CTX;
      } else {
        st.v = HeadAttrs{1.0, 0, false, false, 0};
        st.src = HeadSrc::DEFAULT;
      }
      if (!e.has_sampling) {
        const bool dev_ok = e.dev_prev;
        bool host_ok = false;
        if (!dev_ok && se.kind == "decode" && se.n_rows > 1 && tl != nullptr &&
            (int64_t)se.row0 + se.n_rows <= (int64_t)tl->size()) {
          host_ok = *std::min_element(tl->begin() + se.row0, tl->begin() + se.row0 + se.n_rows) >= 0;
        }
        st.v.drafts_valid = dev_ok || host_ok;
        if (se.kind == "decode") {
          out.n_dec += 1;
          out.n_resume += !dev_ok;
          out.n_host += host_ok;
        }
      }
      st.v.has_num_acceptable = e.has_grammar;
      st.v.num_acceptable = e.has_grammar ? e.grammar : 0;
      out.states[se.rid] = st;
    }
  }
  return out;
}

std::vector<std::vector<d13::HsdState>> to_hsd_states(const HeadOut& out, const std::vector<StepBatch>& b,
                                                      uint64_t (*ident)(const Rid&, HeadSrc)) {
  std::vector<std::vector<d13::HsdState>> s(b.size());
  for (size_t r = 0; r < b.size(); ++r) {
    for (const SeqEntry& se : b[r].seqs) {
      const HeadState& h = out.states.at(se.rid);
      d13::HsdState x;
      x.temperature = h.v.temperature;
      x.seed = h.v.seed;
      x.drafts_valid = h.v.drafts_valid;
      x.has_num_acceptable = h.v.has_num_acceptable;
      x.num_acceptable = h.v.num_acceptable;
      x.ident = (ident != nullptr && h.src != HeadSrc::DEFAULT) ? ident(se.rid, h.src) : 0;
      s[r].push_back(x);
    }
  }
  return s;
}

std::vector<HeadRecord> head_records(int64_t blob_bytes, int hi, int n_ranks) {
  std::vector<HeadRecord> v;
  for (int r = 0; r < n_ranks; ++r) {
    HeadRecord x;
    x.op = "h2d";
    x.rank = r;
    x.src = "hsd._hblob[" + std::to_string(hi) + "][" + std::to_string(r) + "]";
    x.dst = "hsd._dblob[" + std::to_string(r) + "]";
    x.bytes = blob_bytes;
    v.push_back(x);
  }
  return v;
}

}  // namespace p0
}  // namespace fpr
