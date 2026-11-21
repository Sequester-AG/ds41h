// blockhash.h -- the core's prefix-cache block keys, used when the C++ lifecycle drives the C++
// rank scheduler directly. vLLM's own keys (request_block_hasher: sha256 over (parent
// tokens, extra keys)) never leave the engine, so the core only needs the same SEMANTICS, not the same bits:
//   * key[i] = SHA-256(key[i-1] (16 bytes, zeros for i = 0) || extra_i || tokens of block i as int32 LE), first 16 bytes;
//   * extra_0 = "salt:" + cache_salt when the request has one (vLLM: cache_salt in the first block's extra keys);
//   * a request that must not share cache entries (skip_reading_prefix_cache, e.g. prompt logprobs) gets
//     "private:" + request id in EVERY block (the adapter's _HashFn private key space);
//   * full blocks only (PAGE_TOKENS = 64), computed incrementally as tokens are appended.
// SHA-256, never a weak hash: equal keys = shared KV, so a collision would hand one request another's cache.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace fpr {
namespace core {

void sha256(const uint8_t* data, size_t n, uint8_t out[32]);

struct BlockKeyCtx {
  std::string salt;                    // cache_salt ("" = none)
  std::string private_rid;             // non-empty: per-request key space
  int32_t block_tokens = PAGE_TOKENS;
};
// appends the keys of the full blocks of `tokens` beyond keys->size()
void extend_block_keys(const BlockKeyCtx& c, const std::vector<int32_t>& tokens, std::vector<Hash>* keys);

}  // namespace core
}  // namespace fpr
