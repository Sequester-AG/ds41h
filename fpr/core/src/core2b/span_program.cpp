// span_program.cpp -- see span_program.h. Build: -O2 -std=c++17.
// Reference: engine/mixed_cxx/mixed_cxx.py _issue_span, restricted to the all-C++ case.
#include "span_program.h"

#include <algorithm>

namespace fpr {
namespace core2b {

namespace {
bool has(const std::vector<int32_t>& v, int32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }
}  // namespace

std::string build_span(const SpanSegment& s, const std::vector<SpanRankHandles>& h, const SpanKinds& k,
                       std::vector<std::vector<SpanEntry>>* prog) {
  const int W = (int)h.size();
  prog->assign((size_t)W, {});
  // _issue_span's "not covered" conditions (the caller would run Engine._issue in Python)
  if (!s.moe_prod) return "segment: moe not prod";
  if (s.capturing || s.l2pf || s.nancheck || s.dbg_sync || s.prof_pieces) return "segment: debug / capture / L2PF switch";
  if (s.cp_on && (k.attn_bi < 0 || k.attn_bo < 0 || k.idx_c < 0 || k.idx_s < 0 || k.cp_x1a_send < 0 || k.cp_x1a_recv < 0 ||
                  k.cp_x2_send < 0 || k.cp_x2_recv < 0))
    return "segment: CP kinds not registered (CP pieces)";
  if (s.tp_on || s.mega_on) return "segment: TP / c1mega";
  if (!s.router_pack3) return "segment: router_pack3 off";
  if (s.mm_rt) return "segment: image routing rows";
  if (!s.mext_ok) return "segment: moe ext not ready";
  if (s.agrs_on && s.agrs_ag32 && s.R >= s.agrs_min_r) return "segment: AGRS_AG32";
  if ((int)s.ce.size() != W || (int)s.rsf.size() != W || (int)s.fl_rsf.size() != W) return "segment: per-rank flags";
  auto pf = [&](int r) { return !s.pf.empty() && s.pf[(size_t)r]; };
  for (const SpanRankHandles& rh : h)
    if (rh.layers.size() != s.layers.size()) return "segment: handles per layer";
  auto add_all = [&](int32_t kind, int32_t layer, auto handle_of, auto arg_of, bool joins) -> std::string {
    for (int r = 0; r < W; ++r) {
      const int64_t hd = handle_of(r);
      if (hd < 0) return "handle missing";
      (*prog)[(size_t)r].push_back(SpanEntry{r, kind, layer, hd, arg_of(r), joins});
    }
    return "";
  };
  auto zero = [](int) { return (int64_t)0; };
  auto add_fixed = [&](int32_t kind, int32_t layer) {   // a handle-less entry (cp send / recv), every rank
    for (int r = 0; r < W; ++r) (*prog)[(size_t)r].push_back(SpanEntry{r, kind, layer, -1, 0, false});
  };
  std::string e;
  if (s.embed) {
    if (k.embed < 0) return "embed: no owner kind";
    if (!(e = add_all(k.embed, -1, [&](int r) { return h[(size_t)r].embed; }, zero, false)).empty()) return "embed: " + e;
  }
  for (size_t i = 0; i < s.layers.size(); ++i) {
    const int32_t L = s.layers[i];
    const std::string at = " L" + std::to_string(L);
    auto H = [&](int r) -> const SpanLayerHandles& { return h[(size_t)r].layers[i]; };
    if (has(s.engram_layers, L) && s.engram_on) {
      if (k.engram_bd < 0) return "engram_bd: no owner kind" + at;
      if (!(e = add_all(k.engram_bd, L, [&](int r) { return H(r).engram_bd; }, zero, true)).empty()) return "engram_bd" + at + ": " + e;
    } else if (!(e = add_all(k.bd, L, [&](int r) { return H(r).bd0; }, zero, false)).empty()) {
      return "bd0" + at + ": " + e;
    }
    const bool with_idx = has(s.index_layers, L) && s.idx_on;
    if (k.attn_a < 0 || k.attn_b < 0 || (with_idx && k.idx < 0)) return "attn: no owner kind" + at;
    if (!(e = add_all(k.attn_a, L, [&](int r) { return H(r).attn_a; }, zero, false)).empty()) return "attn_a" + at + ": " + e;
    auto attn_b_h = [&](int r) { return pf(r) ? H(r).attn_b_pf : H(r).attn_b; };
    if (s.cp_on && L <= 20) {           // CP layout
      if (with_idx) {
        if (!(e = add_all(k.idx_c, L, [&](int r) { return H(r).idx; }, zero, false)).empty()) return "idx_c" + at + ": " + e;
        add_fixed(k.cp_x1a_send, L);
        add_fixed(k.cp_x1a_recv, L);
        if (!(e = add_all(k.idx_s, L, [&](int r) { return H(r).idx; }, zero, false)).empty()) return "idx_s" + at + ": " + e;
      }
      if (!(e = add_all(k.attn_bi, L, attn_b_h, zero, false)).empty()) return "attn_bi" + at + ": " + e;
      add_fixed(k.cp_x2_send, L);
      add_fixed(k.cp_x2_recv, L);
      if (!(e = add_all(k.attn_bo, L, attn_b_h, zero, false)).empty()) return "attn_bo" + at + ": " + e;
    } else {
      if (with_idx && !(e = add_all(k.idx, L, [&](int r) { return H(r).idx; }, zero, false)).empty()) return "idx" + at + ": " + e;
      if (!(e = add_all(k.attn_b, L, attn_b_h, zero, false)).empty()) return "attn_b" + at + ": " + e;
    }
    if (!(e = add_all(k.bd, L, [&](int r) { return H(r).bd1; }, zero, false)).empty()) return "bd1" + at + ": " + e;
    // AGCE per LAYER, as mixed_cxx._issue_span: layer 0 of the segment takes the segment's ce (the previous AG was on this
    // bucket: s.ce); every later layer follows an AG on this same bucket (Python sets E._agrs_st_last = st after each
    // layer's router + ag), so ce = agrs_on && agce && R >= min_r
    const bool ce_rest = s.agrs_on && s.agrs_agce && s.R >= s.agrs_min_r;
    auto ce_of = [&](int r) { return i == 0 ? (bool)s.ce[(size_t)r] : ce_rest; };
    auto ce = [&](int r) { return (int64_t)ce_of(r); };
    if (!(e = add_all(k.router, L, [&](int r) { return ce_of(r) ? H(r).router_ce : H(r).router; }, ce, false)).empty())
      return "router" + at + ": " + e;
    if (!(e = add_all(k.ag, L, [&](int r) { return H(r).ag; }, ce, false)).empty()) return "ag" + at + ": " + e;
    for (int r = 0; r < W; ++r) (*prog)[(size_t)r].push_back(SpanEntry{r, k.agce_join, L, -1, 0, false});
    if (k.moe < 0) return "moe: no owner kind" + at;
    if (!(e = add_all(k.moe, L, [&](int r) { return s.rsf[(size_t)r] ? H(r).moe_rsf : H(r).moe; }, zero, false)).empty())
      return "moe" + at + ": " + e;
    if (!(e = add_all(k.rs, L, [&](int r) { return s.fl_rsf[(size_t)r] ? H(r).rs_fl : H(r).rs; },
                      [&](int r) { return (int64_t)s.rsf[(size_t)r]; }, false)).empty())
      return "rs" + at + ": " + e;
  }
  if (s.final_ && !(e = add_all(k.final_, -1, [&](int r) { return h[(size_t)r].final_; }, zero, false)).empty())
    return "final: " + e;
  for (int r = 0; r < W; ++r) (*prog)[(size_t)r].push_back(SpanEntry{r, k.join, -1, -1, 0, false});
  return "";
}

}  // namespace core2b
}  // namespace fpr
