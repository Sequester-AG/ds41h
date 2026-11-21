// cost_model.h -- serve/cost_model.py (RankLoad, StepCost, CostModel). Part of fpr_core.h (umbrella).
// Bitwise parity with Python's floats: every expression keeps Python's operand order and association (left to right),
// ints convert to double where Python mixes int and float, and the build MUST use -ffp-contract=off (no FMA fusion).
// Python max() / min() keep the FIRST extreme element (std::max / std::min of two values do the same).
// ProdCostModel (the one-rank production reference of the simulation) is not ported: no scheduler path of the engine
// uses it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "common.h"
#include "cost_params.h"
#include "geometry.h"

namespace fpr {

// PySum (Python 3.12's float sum()): common.h.

struct RankLoad {                     // cost_model.RankLoad: what one rank runs in a step (same fields, same types)
  int64_t dec_rows = 0;               // decode rows (k + 1 per decoding sequence)
  double dec_idx_pos = 0.0;           // sum over decoding seqs of FULL_POS_PER_CTX * ctx
  double dec_cand_pos = 0.0;          // sum over decoding seqs of 4 * min(ctx, CAND_SCORED)
  int64_t n_dec_seqs = 0;
  int64_t max_dec_ctx = 0;
  int64_t pf_rows = 0;                // prefill rows (layers 0..20)
  int64_t pf_tail_rows = 0;           // prefill rows that also run layers 21..39 (last 128 of each chunk)
  double pf_idx_pos = 0.0;            // sum over prefill rows of FULL_POS_PER_CTX * ctx_at_row
  double pf_cand_pos = 0.0;           // sum over tail rows of 4 * min(ctx, CAND_SCORED)

  void add_decode(int64_t rows, int64_t ctx) {
    dec_rows += rows;
    n_dec_seqs += 1;
    max_dec_ctx = std::max(max_dec_ctx, ctx);
    dec_idx_pos += FULL_POS_PER_CTX * (double)ctx;
    dec_cand_pos += (double)(N_CAND_LAYERS * std::min(ctx, CAND_SCORED));
  }
  void add_prefill(int64_t start, int64_t n) {   // a chunk of n prefill rows at positions [start, start + n)
    pf_rows += n;
    pf_idx_pos += FULL_POS_PER_CTX * (double)n * ((double)start + 0.5 * (double)n);
    int64_t tail = std::min<int64_t>(n, SLIDING_WINDOW);
    pf_tail_rows += tail;
    pf_cand_pos += (double)(N_CAND_LAYERS * tail * std::min<int64_t>(start + n, CAND_SCORED));
  }
  // every field with its Python name (golden traces: the LOAD_FIELDS order of gtrace_sched.py)
  template <class C, class V> static void visit(C& c, V&& v) {
    v("dec_rows", c.dec_rows); v("dec_idx_pos", c.dec_idx_pos); v("dec_cand_pos", c.dec_cand_pos);
    v("n_dec_seqs", c.n_dec_seqs); v("max_dec_ctx", c.max_dec_ctx); v("pf_rows", c.pf_rows);
    v("pf_tail_rows", c.pf_tail_rows); v("pf_idx_pos", c.pf_idx_pos); v("pf_cand_pos", c.pf_cand_pos);
  }
};

struct StepCost {                     // cost_model.StepCost
  double total_us = 0.0;
  std::vector<double> rank_a_us, rank_b_us;
  double shared_us = 0.0, head_us = 0.0;

  double idle_fraction() const {
    size_t n = rank_a_us.size();
    if (!n || total_us <= 0) return 0.0;
    double ma = rank_a_us[0], mb = rank_b_us[0];
    for (double a : rank_a_us) ma = std::max(ma, a);
    for (double b : rank_b_us) mb = std::max(mb, b);
    PySum la, lb;                     // sum(ma - a for a in ...) + sum(mb - b for b in ...)
    for (double a : rank_a_us) la.add(ma - a);
    for (double b : rank_b_us) lb.add(mb - b);
    return (la.value() + lb.value()) / ((double)n * total_us);
  }
};

class CostModel {
 public:
  CostParams p;
  CostModel() = default;
  explicit CostModel(const CostParams& p_) : p(p_) {}

  void rank_sides(const RankLoad& ld, double* a_out, double* b_out) const {
    double a = (double)COST_N_A * (p.a0 + p.a_row * (double)ld.dec_rows + p.a_pf_row * (double)ld.pf_rows);
    a += p.idx_dec_us_per_pos * ld.dec_idx_pos + p.idx_dec_fixed_us * (double)ld.n_dec_seqs;
    a += p.idx_pf_us_per_pos * ld.pf_idx_pos;
    double b = (double)COST_N_B * (p.a0 + p.a_row * (double)(ld.dec_rows + ld.pf_tail_rows));
    b += p.idx_dec_us_per_pos * ld.dec_cand_pos + p.idx_pf_us_per_pos * ld.pf_cand_pos;
    *a_out = a;
    *b_out = b;
  }
  double rank_sides_sum(const RankLoad& ld) const {   // Python sum(self.cost.rank_sides(ld))
    double a, b;
    rank_sides(ld, &a, &b);
    PySum s;
    s.add(a);
    s.add(b);
    return s.value();
  }
  struct LayerSides { double base_a, idx_a, base_b, idx_b; };
  LayerSides layer_sides(const RankLoad& ld) const {
    LayerSides s;
    s.base_a = p.a0 + p.a_row * (double)ld.dec_rows + p.a_pf_row * (double)ld.pf_rows;
    s.idx_a = p.idx_dec_us_per_pos * ld.dec_idx_pos + p.idx_dec_fixed_us * (double)ld.n_dec_seqs +
              p.idx_pf_us_per_pos * ld.pf_idx_pos;
    s.base_b = p.a0 + p.a_row * (double)(ld.dec_rows + ld.pf_tail_rows);
    s.idx_b = p.idx_dec_us_per_pos * ld.dec_cand_pos + p.idx_pf_us_per_pos * ld.pf_cand_pos;
    return s;
  }

  template <class LoadAt>             // LoadAt(i) -> const RankLoad& for i in [0, n)
  StepCost step_of(size_t n, LoadAt at, int64_t copy_bytes = 0) const {
    StepCost c;
    c.rank_a_us.resize(n);
    c.rank_b_us.resize(n);
    for (size_t i = 0; i < n; ++i) rank_sides(at(i), &c.rank_a_us[i], &c.rank_b_us[i]);
    double att;
    if (p.ideal_balance != 0.0) {
      PySum sa, sb;
      for (double a : c.rank_a_us) sa.add(a);
      for (double b : c.rank_b_us) sb.add(b);
      double ma = sa.value() / (double)n, mb = sb.value() / (double)n;
      std::fill(c.rank_a_us.begin(), c.rank_a_us.end(), ma);
      std::fill(c.rank_b_us.begin(), c.rank_b_us.end(), mb);
      att = ma + mb;
    } else {
      const int ni = N_KV_SOURCE_LAYERS;
      double m0 = 0, m1 = 0, m2 = 0, m3 = 0;
      for (size_t i = 0; i < n; ++i) {
        LayerSides x = layer_sides(at(i));
        double v0 = x.base_a, v1 = x.base_a + x.idx_a / (double)ni, v2 = x.base_b,
               v3 = x.base_b + x.idx_b / (double)N_CAND_LAYERS;
        if (i == 0) { m0 = v0; m1 = v1; m2 = v2; m3 = v3; }
        else { m0 = std::max(m0, v0); m1 = std::max(m1, v1); m2 = std::max(m2, v2); m3 = std::max(m3, v3); }
      }
      att = (double)(COST_N_A - ni) * m0 + (double)ni * m1;
      att += (double)(COST_N_B - N_CAND_LAYERS) * m2 + (double)N_CAND_LAYERS * m3;
    }
    int64_t dec = 0, pf = 0, pft = 0, max_rows_a = 0, max_rows_b = 0, sum_a = 0, sum_b = 0, max_dec = 0, max_ctx = 0;
    for (size_t i = 0; i < n; ++i) {
      const RankLoad& ld = at(i);
      dec += ld.dec_rows;
      pf += ld.pf_rows;
      pft += ld.pf_tail_rows;
      int64_t ra = ld.dec_rows + (p.pf_tok_us != 0.0 ? 0 : ld.pf_rows), rb = ld.dec_rows + ld.pf_tail_rows;
      max_rows_a = i ? std::max(max_rows_a, ra) : ra;
      max_rows_b = i ? std::max(max_rows_b, rb) : rb;
      sum_a += ra;
      sum_b += rb;
      max_dec = i ? std::max(max_dec, ld.dec_rows) : ld.dec_rows;
      max_ctx = i ? std::max(max_ctx, ld.max_dec_ctx) : ld.max_dec_ctx;
    }
    double rmax_a, rmax_b;
    if (p.coll_real_rows != 0.0 || p.ideal_balance != 0.0) {
      rmax_a = (double)sum_a / (double)n;
      rmax_b = (double)sum_b / (double)n;
    } else {
      rmax_a = (double)max_rows_a;
      rmax_b = (double)max_rows_b;
    }
    if (dec + pf == 0) {
      c.total_us = 0.0;
      return c;
    }
    double shared;
    if (pf && p.moe_mixed0 > 0.0) {
      shared = (double)COST_N_A * (p.c0 + p.c_row * rmax_a + p.moe_mixed0 + p.moe_mixed_row * (double)(dec + pf));
      shared += (double)COST_N_B * (p.c0 + p.c_row * rmax_b + p.moe_mixed0 + p.moe_mixed_row * (double)(dec + pft));
    } else {
      shared = (double)COST_N_A * (p.c0 + p.c_row * rmax_a + p.moe0 + p.moe_dec_row * (double)dec + p.moe_pf_row * (double)pf);
      shared += (double)COST_N_B * (p.c0 + p.c_row * rmax_b + p.moe0 + p.moe_dec_row * (double)dec + p.moe_pf_row * (double)pft);
    }
    double head = dec ? p.head0 + p.head_row * (double)max_dec : 0.0;
    shared += p.pf_tok_us * (double)pf;
    double total = att + shared + head + (pf ? p.pf_overhead : 0.0) + p.copy_us_per_mb * (double)copy_bytes / 1e6;
    total += p.host_us + (!pf ? p.dec_only_extra_us : 0.0);
    if (!pf && p.eager_ctx_above && max_ctx > p.eager_ctx_above) total += p.eager_extra_us;
    c.total_us = total;
    c.shared_us = shared;
    c.head_us = head;
    return c;
  }
  StepCost step(const std::vector<RankLoad>& loads, int64_t copy_bytes = 0) const {
    return step_of(loads.size(), [&](size_t i) -> const RankLoad& { return loads[i]; }, copy_bytes);
  }

  double mixed_fixed_us() const {
    double extra = p.moe_mixed0 > 0.0 ? (double)N_LAYERS * std::max(0.0, p.moe_mixed0 - p.moe0) : 0.0;
    return p.pf_overhead + extra;
  }
  double decode_seq_rank_us(int64_t ctx, int64_t rows) const {
    return (double)N_LAYERS * p.a_row * (double)rows +
           p.idx_dec_us_per_pos * (FULL_POS_PER_CTX * (double)ctx + (double)(N_CAND_LAYERS * std::min(ctx, CAND_SCORED))) +
           p.idx_dec_fixed_us;
  }
  double prefill_rank_us(int64_t start, int64_t n) const {
    RankLoad ld;
    ld.add_prefill(start, n);
    double a, b;
    rank_sides(ld, &a, &b);
    return a + b - (double)N_LAYERS * p.a0;
  }
  double prefill_shared_us(int64_t n) const {
    return (double)COST_N_A * p.moe_pf_row * (double)n +
           (double)COST_N_B * p.moe_pf_row * (double)std::min<int64_t>(n, SLIDING_WINDOW) + p.pf_tok_us * (double)n;
  }
};

}  // namespace fpr
