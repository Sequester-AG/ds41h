// native_mixed.cc -- see native_mixed.h. Build: -O2 -std=c++17 -ffp-contract=off.
#include "native_mixed.h"

#include <algorithm>
#include <unordered_map>

namespace fpr {
namespace core2b {

Status tail_batch(const StepBatch& sb, bool tail_bound, int64_t sliding_window, StepBatch* out, std::string* err) {
  StepBatch tb;                                     // StepBatch(sb.rank, sb.step, 0)
  tb.rank = sb.rank;
  tb.step = sb.step;
  tb.n_rows = 0;
  std::unordered_map<int32_t, int32_t> new_of;      // {row: i for i, row in enumerate(sb.tail_rows)}
  new_of.reserve(sb.tail_rows.size() * 2 + 1);
  for (size_t i = 0; i < sb.tail_rows.size(); ++i) new_of[sb.tail_rows[i]] = (int32_t)i;   // a repeated row: the last i wins
  for (const SeqEntry& se : sb.seqs) {
    std::vector<int32_t> rows;                      // [x for x in range(row0, row0 + n_rows) if x in new_of]
    for (int32_t x = se.row0; x < se.row0 + se.n_rows; ++x)
      if (new_of.count(x)) rows.push_back(x);
    if (rows.empty()) continue;
    if ((size_t)rows[0] >= sb.positions.size()) {
      if (err) *err = "tail_batch: row " + std::to_string(rows[0]) + " beyond positions";
      return Status::BAD_ARG;
    }
    SeqEntry ne = se;                               // dataclasses.replace(se, row0=..., n_rows=..., start=...)
    ne.row0 = new_of[rows[0]];
    ne.n_rows = (int32_t)rows.size();
    ne.start = sb.positions[rows[0]];
    if (tail_bound && se.kind != "decode" && !se.score && (int32_t)rows.size() < se.n_rows)
      ne.replay_start = std::max(se.replay_start, se.start + (int64_t)se.n_rows - sliding_window);
    ne.tail_row0 = ne.row0;
    tb.seqs.push_back(std::move(ne));
    int32_t slot = (int32_t)tb.seqs.size() - 1;
    for (int32_t x : rows) {
      if ((size_t)x >= sb.positions.size() || (size_t)x >= sb.paged_slot.size() || (size_t)x >= sb.window_slot.size()) {
        if (err) *err = "tail_batch: row " + std::to_string(x) + " beyond the row arrays";
        return Status::BAD_ARG;
      }
      tb.positions.push_back(sb.positions[x]);
      tb.seq_slot.push_back(slot);
      tb.paged_slot.push_back(sb.paged_slot[x]);
      tb.window_slot.push_back(sb.window_slot[x]);
      tb.tail_rows.push_back(new_of[x]);
    }
  }
  for (int32_t x : sb.sample_rows) {                // [new_of[x] for x in sb.sample_rows] (KeyError -> BAD_ARG)
    auto it = new_of.find(x);
    if (it == new_of.end()) {
      if (err) *err = "tail_batch: sample row " + std::to_string(x) + " is not a tail row";
      return Status::BAD_ARG;
    }
    tb.sample_rows.push_back(it->second);
  }
  tb.n_rows = (int32_t)sb.tail_rows.size();
  tb.copies.clear();
  tb.poison.clear();                                // (Python: StepBatch()'s default, not carried by tail_batch)
  *out = std::move(tb);
  return Status::OK;
}

std::vector<int32_t> tail_tokens(const StepBatch& sb, const std::vector<int32_t>& tokens) {
  std::vector<int32_t> t;
  t.reserve(sb.tail_rows.size());
  for (int32_t i : sb.tail_rows) t.push_back((size_t)i < tokens.size() ? tokens[i] : -1);
  return t;
}

}  // namespace core2b
}  // namespace fpr
