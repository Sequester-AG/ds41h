// common.h -- shared types of the C++ engine core.
#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fpr {

using Rid = std::string;            // vLLM request id (external string)
struct Hash {                       // 128-bit block hash: 2a = the first 16 bytes of vLLM's block-hash bytes (the
  uint64_t hi = 0, lo = 0;           // adapter derives it at the boundary; never truncated to 64 bits: a collision
  bool operator==(const Hash& o) const { return hi == o.hi && lo == o.lo; }   // = a wrong prefix reuse); traces
  bool operator!=(const Hash& o) const { return !(*this == o); }              // intern hashes to ids (exact).
};
struct HashHasher { size_t operator()(const Hash& h) const { return (size_t)(h.lo ^ (h.hi * 0x9E3779B97F4A7C15ull)); } };
constexpr int W = 4;                // ranks
constexpr int PAGE_TOKENS = 64, WINDOW_PAGE_TOKENS = 32, SLIDING_WINDOW = 128;

// ---------------------------------------------------------------- float parity with the served Python (3.12.15)
// Python's built-in sum() over floats, CPython >= 3.12: the start value
// is the int 0 (so the first item enters as 0 + x), then Neumaier-compensated summation, the compensation added once
// at the end if it is non-zero and finite (bltinmodule.c builtin_sum_impl; verified against CPython 3.12.15 on 200K
// random float lists: plain left-to-right addition differs in ~30 % of them, last 1-2 bits). RULES for every port:
//  - Python `sum(<floats>)` -> PySum; a Python `x += y` loop stays plain addition (in Python's order).
//  - Other reductions are different again and NOT PySum: math.fsum / statistics.fmean (exact, correctly rounded),
//    statistics.mean (exact rationals), numpy .sum() / np.sum / np.mean (pairwise, 8-way unrolled blocks), torch.sum.
//    Grep your Python for them before porting; emulate exactly or keep the value out of decisions.
//  - A Python int item in the middle of a float sum() is NOT covered here (CPython adds it on a separate path): convert
//    such sums only after checking 3.12.15's bltinmodule.c.
//  - Record golden traces with the served interpreter (venv Python 3.12), never a 3.9 / 3.10 system python.
struct PySum {
  double f = 0.0, c = 0.0;
  bool any = false;
  void add(double x) {
    if (!any) { f = 0.0 + x; any = true; return; }
    double t = f + x;
    if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
    else c += (x - t) + f;
    f = t;
  }
  double value() const { return (c != 0.0 && std::isfinite(c)) ? f + c : f; }
};

enum class Status : int { OK = 0, REJECTED, NO_ROOM, BAD_ARG, INTERNAL };
enum class Phase : int { WAITING = 0, PREFILL, DECODE, FINISHED };
enum class Finish : int { NONE = -1, STOP = 0, LENGTH = 1, ABORT = 2, ERROR = 3 };   // = vLLM FinishReason values

// ---------------------------------------------------------------- requests (serve/scheduler.py Request + SamplingParams)
struct Sampling {                     // the subset the engine uses (sampling_dict(SamplingParams))
  float temperature = 1.0f, top_p = 1.0f; int32_t top_k = -1; int64_t seed = 0; bool has_seed = false;
  int32_t logprobs = -1;              // -1 = none; else top-k logprobs per sampled token
  bool prompt_logprobs = false;       // -> score mode (every prompt row sampled)
  std::vector<int32_t> stop_token_ids; bool ignore_eos = false; int32_t eos_token_id = -1;
  int64_t max_tokens = 0;             // min(client, L - prompt - (k - 1)) (RankScheduler::max_new_cap)
};
struct Image { int64_t offset, length; std::string key; };       // (offset, length, key) in the prompt
struct Request {
  Rid rid;
  std::vector<int32_t> prompt_ids;    // prompt_len = prompt_ids.size()
  std::vector<Hash> block_hashes;     // the prompt's FULL blocks (vLLM's in 2a); outputs' blocks: IScheduler::extend_block_hashes
  bool forced = false;                // the adapter's ForceSpec matched (forced decode: accept every row)
  Sampling sampling;
  double arrival = 0.0;               // seconds, the core's clock (as Python's wall0-relative arrival)
  int32_t priority = 0;
  bool structured = false;            // grammar-constrained (ICore owns the grammar matcher)
  std::string grammar_spec;           // xgrammar input (json schema / ebnf / tool-call structural tag)
  std::vector<Image> images;          // keys + positions; embeddings arrive via the IEngine's mm channel
  std::string cache_salt;
};

}  // namespace fpr
