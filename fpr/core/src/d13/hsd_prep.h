// hsd_prep.h -- kernels/head/hsd.py HSD._build in C++: the host blob HSD's prepare uploads
// byte for byte (layout HSD._SPEC: 64-byte aligned sections, the SHARED part identical on every rank, double-buffered
// sets: stale bytes beyond a bucket's prefixes are whatever that set held, exactly as in Python), plus the step record
// (_Step) and the grammar bitmask staging.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../../include/d13.h"

namespace fpr {
namespace d13 {

class HsdPrep {
 public:
  enum Kind : int { I32 = 0, I64 = 1, F32 = 2 };
  struct Field { std::string name; int kind; char dim; int64_t cols; int64_t off; int64_t count; };   // count = dim * cols
  static constexpr int64_t BM_WORDS = (129280 + 31) / 32;

  explicit HsdPrep(const HsdConfig& c);
  const HsdConfig& cfg() const { return c_; }
  const std::vector<Field>& layout() const { return lay_; }
  int64_t blob_bytes() const { return nbytes_; }
  int64_t shared_end() const { return shared_end_; }
  int64_t MR() const { return W * (c_.max_seqs + 1); }
  int64_t dim(char d) const;
  uint8_t* blob(int set, int rank) { return set_[set][rank]; }
  // serve: point a set's blobs at pinned memory (copies the current content there first)
  void bind(int set, int rank, uint8_t* p);
  int hi() const { return hi_; }
  void set_hi(int h) { hi_ = h; }
  std::vector<int64_t> tail_buckets(int64_t S_pad) const;   // HSD.tail_buckets
  // HSD._build: fills blob set hi(); cur->v0 = rows not in the vectorized layout (Python returns None, nothing written)
  Status build(const std::vector<StepBatch>& b, const std::vector<std::vector<HsdState>>& states,
               const std::vector<HsdBitmask>* bitmasks, HsdStep* cur, std::string* err);
  void uploaded() { hi_ ^= 1; }                              // HSD._upload's `self._hi ^= 1`
  const int32_t* bm() const { return bm_; }                  // _bm_h ([T_max, BM_WORDS] int32; rows [:T] are the step's)
  void bind_bm(int32_t* p) { bm_ = p; }                      // 2a: HSD's own pinned _bm_h (else a private buffer)
  // replays: the per-slot temperature / seed / state identity HSD holds (HSD._temp_np / _seed_np / _slot_key)
  void restore(const std::vector<float>& temp, const std::vector<int64_t>& seed, const std::vector<uint64_t>& keys);
  // a byte offset -> "field[index]" (replay diagnostics)
  std::string where(int64_t byte_off) const;

 private:
  HsdConfig c_;
  std::vector<Field> lay_;
  int64_t nbytes_ = 0, shared_end_ = 0;
  std::vector<uint8_t> own_[2][W];
  uint8_t* set_[2][W];
  int hi_ = 0;
  std::vector<float> temp_;                                  // HSD._temp_np
  std::vector<int64_t> seed_;                                // HSD._seed_np
  std::vector<uint64_t> key_;                                // HSD._slot_key (identity of the slot's state object)
  std::vector<bool> key_set_;
  std::vector<int32_t> bm_own_;
  int32_t* bm_ = nullptr;
  int fidx(const char* name) const;
  template <class T> T* f(uint8_t* blob, int i) const { return reinterpret_cast<T*>(blob + lay_[i].off); }
};

}  // namespace d13
}  // namespace fpr
