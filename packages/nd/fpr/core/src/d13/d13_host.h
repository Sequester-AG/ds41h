// d13_host.h -- stage 2: D13's host side in C++: fpr::ID13Host (include/d13.h), decision
// for decision with
//   fpr/engine/d13.py              _async_ok, _deltas, _empty_eligible, _owner_sig, _guard_host, _positions,
//                                  _fallback (start resolution, accepted counts), _advance (HSD request index)
//   fpr/engine/d13s.py             Steady.signature + is_steady's batch-only conditions
//   fpr/kernels/rowexp/rowexp.py   RowExp.desc_of_compact, HostDesc.build, RowExp.upload's pinned staging layout
//   fpr/kernels/head/hsd.py        HSD._build (hsd_prep.h)
// Golden traces: fpr/core/GTRACE.md; recorder src/d13/gtrace_d13.py; replays tests/replay_d13.cpp, tests/replay_hsd.cpp.
// Python integer semantics throughout (floor // and %, slices clamp, numpy int32 casts wrap).
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../include/d13.h"
#include "hsd_prep.h"

namespace fpr {
namespace d13 {

constexpr int64_t PAD = -1;
constexpr int64_t SW = 128;            // sliding window (si columns)
constexpr int64_t WT = 32;             // window page tokens
constexpr int64_t BT = 64;             // paged page tokens

// Python // and % (floor semantics, as numpy / Python ints)
inline int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int64_t fmod(int64_t a, int64_t b) { int64_t m = a % b; return (m != 0 && ((m < 0) != (b < 0))) ? m + b : m; }

class WindowIndex {                    // dict.get on a window table (linear when small, an index when large)
 public:
  explicit WindowIndex(const WindowTable* wt);
  bool get(int64_t blk, int64_t* page) const;
  int64_t get_or(int64_t blk, int64_t dflt) const { int64_t p; return get(blk, &p) ? p : dflt; }
 private:
  const WindowTable* wt_;
  std::map<int64_t, int64_t> idx_;
};

class HostDesc {                       // rowexp.HostDesc: StepBatch -> Desc with per-slot ownership / ring bookkeeping
 public:
  explicit HostDesc(int rank, bool device_start = true) : rank_(rank), device_start_(device_start) {}
  Status build(const StepBatch& sb, Desc* out, std::string* err);
 private:
  struct Sent { std::pair<int64_t, int64_t> uid; std::map<int64_t, int64_t> have; int64_t nb; int64_t wver; };
  int rank_; bool device_start_;   // (rank_: as HostDesc.rank; no decision reads it)
  std::map<int64_t, std::pair<int64_t, int64_t>> owner_;   // state slot -> (uid, inc) that last advanced nc[slot]
  std::map<int64_t, Sent> sent_;                          // state slot -> window pages in the device ring
};

class DescStaging {                    // RowExp.upload's host half: two pinned sets (int32 block | int64 block), alternating
 public:
  DescStaging(int64_t max_seqs, int64_t max_ops);
  int64_t n32_total() const { return 4 * S_ + 4 + 3 * O_; }
  int64_t n64_total() const { return 3 * S_; }
  // packs into the next set, *n32 = the int32 prefix the upload copies (the int64 block is copied whole)
  Status pack(const Desc& d, int64_t max_rows, int64_t* n32, std::string* err);
  // the same layout into caller-owned buffers (RowExp._h[i]: h32 [4 S + 4 + 3 O], h64 [3 S]); stage 2b NativeDecode
  static Status pack_into(const Desc& d, int64_t S_max, int64_t O_max, int64_t max_rows, int32_t* h32, int64_t* h64,
                          int64_t* n32, std::string* err);
  const std::vector<int32_t>& h32(int i) const { return h32_[i]; }
  const std::vector<int64_t>& h64(int i) const { return h64_[i]; }
  int last_set() const { return last_; }
 private:
  int64_t S_, O_;
  std::vector<int32_t> h32_[2];
  std::vector<int64_t> h64_[2];
  int hi_ = 0, last_ = -1;
};

class D13Host : public ID13Host {
 public:
  D13Host(const Config& c, const HsdConfig& h) : cfg_(c), hsd_(h) {}
  const Config& cfg() const { return cfg_; }
  void set_graphs(bool g) { cfg_.graphs = g; }   // eng.cfg.graphs is read per dispatch (the replay feeds it per call)

  std::string async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct) const override;
  void deltas(int rank, const CompactTables& ct, Deltas* out, MirrorFn mirror = nullptr,
              void* user = nullptr) const override;
  void desc(const CompactTables& ct, Desc* out) const override { desc_of_compact(ct, out); }
  static void desc_of_compact(const CompactTables& ct, Desc* d);
  bool empty_eligible(const StepBatch& sb, const CompactTables& ct) const override;
  std::optional<OwnerSig> owner_sig(int rank, const StepBatch& sb, const CompactTables& ct,
                                    const std::vector<PrevOf>& prev) const override;
  std::vector<int64_t> guard_blocks(const StepBatch& sb) const override;
  void positions(const StepBatch& sb, const std::vector<int64_t>& pos, const std::vector<int64_t>& own,
                 StepBatch* out) const override;
  Status fallback_resolve(const std::vector<StepBatch>& b, const std::vector<std::vector<int64_t>>& starts,
                          std::vector<StepBatch>* rb, std::string* err) const override;
  Status fallback_tables(const std::vector<StepBatch>& rb, const std::vector<CompactTables>& ct,
                         std::vector<RankTables>* out, std::string* err) override;
  std::vector<int64_t> fallback_accepted(const StepBatch& rb,
                                         const std::vector<std::pair<Rid, int32_t>>& accepted) const override;
  std::vector<int64_t> advance_index(int rank, int64_t S, int64_t D_pad,
                                     const std::vector<std::pair<int64_t, int64_t>>& dseqs, bool* slice) const override;
  bool steady_batches(const std::vector<StepBatch>& b, const std::vector<StepBatch>& prev,
                      const std::vector<Rid>& structured) const override;
  Status hsd_prepare(const std::vector<StepBatch>& b, const std::vector<std::vector<HsdState>>& states,
                     const std::vector<HsdBitmask>* bitmasks, HsdStep* cur, std::string* err) override {
    return hsd_.build(b, states, bitmasks, cur, err);
  }
  void hsd_uploaded() override { hsd_.uploaded(); }
  uint8_t* hsd_blob(int set, int rank) override { return hsd_.blob(set, rank); }
  int64_t hsd_blob_bytes() const override { return hsd_.blob_bytes(); }

  HsdPrep& hsd() { return hsd_; }
  TableBuilder& tables() { return tb_; }   // D13's private TableBuilder (stub_indexer=False): the fallback steps' tables
  // the engine's OWN D13.tb as a C++ ITables
  // fallback_tables builds through it, so Python's embedded fallback steps and nd's native mixed steps share ONE D12 delta
  // state. nullptr = the private tb_ (ENG_CXX_TB off / check mode). PRECONDITION: Python and nd never
  // build through the shared instance concurrently (it has no locking): nd builds only between Python's steps.
  void set_tables(ITables* shared) { shared_tb_ = shared; }
  bool shared_tables() const { return shared_tb_ != nullptr; }

 private:
  Config cfg_;
  HsdPrep hsd_;
  TableBuilder tb_;
  ITables* shared_tb_ = nullptr;
};

}  // namespace d13
}  // namespace fpr
