// Hashes of component IMAGE:
//  - BLAKE3 (portable reference algorithm): vLLM MultiModalHasher, mm_hasher_algorithm "blake3" (the default)
//  - the MultiModalHasher framing for an image item (vllm/multimodal/hasher.py)
// - BLAKE2b-128: the phys adapter's content key
//  - SHA-256: test support
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "image/image_internal.h"

namespace fe::img {
namespace {

const char* kHex = "0123456789abcdef";
std::string hex(const uint8_t* p, size_t n) {
  std::string s(2 * n, '0');
  for (size_t i = 0; i < n; i++) s[2 * i] = kHex[p[i] >> 4], s[2 * i + 1] = kHex[p[i] & 15];
  return s;
}

// ---------------------------------------------------------------- BLAKE3
constexpr uint32_t kIV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
constexpr int kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
constexpr uint32_t kChunkStart = 1, kChunkEnd = 2, kParent = 4, kRoot = 8;
constexpr size_t kBlockLen = 64, kChunkLen = 1024;

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 7);
}

struct Schedule {
  int s[7][16];
  Schedule() {
    for (int i = 0; i < 16; i++) s[0][i] = i;
    for (int r = 1; r < 7; r++)
      for (int i = 0; i < 16; i++) s[r][i] = s[r - 1][kPerm[i]];  // m'[i] = m[perm[i]] after every round
  }
};
const Schedule kSched;

void compress(const uint32_t cv[8], const uint32_t m[16], uint64_t counter, uint32_t block_len, uint32_t flags,
              uint32_t out[16]) {
  uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    kIV[0], kIV[1], kIV[2], kIV[3], uint32_t(counter), uint32_t(counter >> 32), block_len, flags};
  for (int r = 0; r < 7; r++) {
    const int* q = kSched.s[r];
    g(s, 0, 4, 8, 12, m[q[0]], m[q[1]]);
    g(s, 1, 5, 9, 13, m[q[2]], m[q[3]]);
    g(s, 2, 6, 10, 14, m[q[4]], m[q[5]]);
    g(s, 3, 7, 11, 15, m[q[6]], m[q[7]]);
    g(s, 0, 5, 10, 15, m[q[8]], m[q[9]]);
    g(s, 1, 6, 11, 12, m[q[10]], m[q[11]]);
    g(s, 2, 7, 8, 13, m[q[12]], m[q[13]]);
    g(s, 3, 4, 9, 14, m[q[14]], m[q[15]]);
  }
  for (int i = 0; i < 8; i++) {
    s[i] ^= s[i + 8];
    s[i + 8] ^= cv[i];
  }
  std::memcpy(out, s, 64);
}

#if defined(__aarch64__)
// Four whole chunks at once (lane i = chunk counter + i): the chaining values of non-root chunks.
template <int N>
inline uint32x4_t vrotr(uint32x4_t x) {
  return vorrq_u32(vshrq_n_u32(x, N), vshlq_n_u32(x, 32 - N));
}
inline void g4(uint32x4_t* v, int a, int b, int c, int d, uint32x4_t mx, uint32x4_t my) {
  v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), mx);
  v[d] = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(veorq_u32(v[d], v[a]))));  // rotr 16
  v[c] = vaddq_u32(v[c], v[d]);
  v[b] = vrotr<12>(veorq_u32(v[b], v[c]));
  v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), my);
  v[d] = vrotr<8>(veorq_u32(v[d], v[a]));
  v[c] = vaddq_u32(v[c], v[d]);
  v[b] = vrotr<7>(veorq_u32(v[b], v[c]));
}
void hash4_chunks(const uint8_t* in, uint64_t counter, uint32_t cvs[4][8]) {
  uint32x4_t h[8];
  for (int i = 0; i < 8; i++) h[i] = vdupq_n_u32(kIV[i]);
  const uint32_t clo[4] = {uint32_t(counter), uint32_t(counter + 1), uint32_t(counter + 2), uint32_t(counter + 3)};
  const uint32_t chi[4] = {uint32_t((counter) >> 32), uint32_t((counter + 1) >> 32), uint32_t((counter + 2) >> 32),
                           uint32_t((counter + 3) >> 32)};
  const uint32x4_t vlo = vld1q_u32(clo), vhi = vld1q_u32(chi);
  for (int blk = 0; blk < 16; blk++) {
    uint32x4_t m[16];
    for (int k = 0; k < 4; k++) {
      const uint32x4_t a = vld1q_u32(reinterpret_cast<const uint32_t*>(in + 0 * 1024 + blk * 64 + 16 * k));
      const uint32x4_t b = vld1q_u32(reinterpret_cast<const uint32_t*>(in + 1 * 1024 + blk * 64 + 16 * k));
      const uint32x4_t c = vld1q_u32(reinterpret_cast<const uint32_t*>(in + 2 * 1024 + blk * 64 + 16 * k));
      const uint32x4_t d = vld1q_u32(reinterpret_cast<const uint32_t*>(in + 3 * 1024 + blk * 64 + 16 * k));
      const uint32x4_t t0 = vtrn1q_u32(a, b), t1 = vtrn2q_u32(a, b), t2 = vtrn1q_u32(c, d), t3 = vtrn2q_u32(c, d);
      m[4 * k + 0] = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
      m[4 * k + 1] = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
      m[4 * k + 2] = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
      m[4 * k + 3] = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    }
    const uint32_t flags = (blk == 0 ? kChunkStart : 0) | (blk == 15 ? kChunkEnd : 0);
    uint32x4_t v[16] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
                        vdupq_n_u32(kIV[0]), vdupq_n_u32(kIV[1]), vdupq_n_u32(kIV[2]), vdupq_n_u32(kIV[3]),
                        vlo, vhi, vdupq_n_u32(64), vdupq_n_u32(flags)};
    for (int r = 0; r < 7; r++) {
      const int* q = kSched.s[r];
      g4(v, 0, 4, 8, 12, m[q[0]], m[q[1]]);
      g4(v, 1, 5, 9, 13, m[q[2]], m[q[3]]);
      g4(v, 2, 6, 10, 14, m[q[4]], m[q[5]]);
      g4(v, 3, 7, 11, 15, m[q[6]], m[q[7]]);
      g4(v, 0, 5, 10, 15, m[q[8]], m[q[9]]);
      g4(v, 1, 6, 11, 12, m[q[10]], m[q[11]]);
      g4(v, 2, 7, 8, 13, m[q[12]], m[q[13]]);
      g4(v, 3, 4, 9, 14, m[q[14]], m[q[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] = veorq_u32(v[i], v[i + 8]);
  }
  uint32_t tmp[8][4];
  for (int i = 0; i < 8; i++) vst1q_u32(tmp[i], h[i]);
  for (int c = 0; c < 4; c++)
    for (int i = 0; i < 8; i++) cvs[c][i] = tmp[i][c];
}
#endif

void words_from_le(const uint8_t* b, uint32_t* w, size_t nwords) {
  for (size_t i = 0; i < nwords; i++)
    w[i] = uint32_t(b[4 * i]) | (uint32_t(b[4 * i + 1]) << 8) | (uint32_t(b[4 * i + 2]) << 16) |
           (uint32_t(b[4 * i + 3]) << 24);
}

struct Output {
  uint32_t cv[8];
  uint32_t block[16];
  uint64_t counter;
  uint32_t block_len, flags;
  void chaining_value(uint32_t out[8]) const {
    uint32_t s[16];
    compress(cv, block, counter, block_len, flags, s);
    std::memcpy(out, s, 32);
  }
  void root_bytes(uint8_t* out, size_t n) const {
    uint64_t oc = 0;
    while (n) {
      uint32_t s[16];
      compress(cv, block, oc++, block_len, flags | kRoot, s);
      for (int i = 0; i < 16 && n; i++)
        for (int k = 0; k < 4 && n; k++, n--) *out++ = uint8_t(s[i] >> (8 * k));
    }
  }
};

struct Chunk {
  uint32_t cv[8];
  uint64_t counter = 0;
  uint8_t block[kBlockLen] = {};
  size_t block_len = 0;
  size_t blocks_compressed = 0;
  void init(const uint32_t key[8], uint64_t c) {
    std::memcpy(cv, key, 32);
    counter = c;
    std::memset(block, 0, sizeof block);
    block_len = 0;
    blocks_compressed = 0;
  }
  size_t len() const { return kBlockLen * blocks_compressed + block_len; }
  uint32_t start_flag() const { return blocks_compressed == 0 ? kChunkStart : 0; }
  void update(const uint8_t* p, size_t n) {
    while (n) {
      if (block_len == kBlockLen) {
        uint32_t w[16], s[16];
        words_from_le(block, w, 16);
        compress(cv, w, counter, kBlockLen, start_flag(), s);
        std::memcpy(cv, s, 32);
        blocks_compressed++;
        std::memset(block, 0, sizeof block);
        block_len = 0;
      }
      size_t take = std::min(kBlockLen - block_len, n);
      std::memcpy(block + block_len, p, take);
      block_len += take;
      p += take;
      n -= take;
    }
  }
  Output output() const {
    Output o;
    std::memcpy(o.cv, cv, 32);
    words_from_le(block, o.block, 16);
    o.counter = counter;
    o.block_len = static_cast<uint32_t>(block_len);
    o.flags = start_flag() | kChunkEnd;
    return o;
  }
};

Output parent_output(const uint32_t l[8], const uint32_t r[8]) {
  Output o;
  std::memcpy(o.cv, kIV, 32);
  std::memcpy(o.block, l, 32);
  std::memcpy(o.block + 8, r, 32);
  o.counter = 0;
  o.block_len = kBlockLen;
  o.flags = kParent;
  return o;
}

}  // namespace

struct Blake3::State {
  Chunk chunk;
  std::vector<std::array<uint32_t, 8>> stack;
};

Blake3::Blake3() : s_(new State) { s_->chunk.init(kIV, 0); }
Blake3::~Blake3() { delete s_; }

void Blake3::push_cv(const uint32_t cv[8], uint64_t total) {
  // add_chunk_chaining_value: merge while the total chunk count has trailing zero bits
  std::array<uint32_t, 8> nc;
  std::memcpy(nc.data(), cv, 32);
  while ((total & 1) == 0) {
    uint32_t m[8];
    parent_output(s_->stack.back().data(), nc.data()).chaining_value(m);
    s_->stack.pop_back();
    std::memcpy(nc.data(), m, 32);
    total >>= 1;
  }
  s_->stack.push_back(nc);
}

void Blake3::update(const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n) {
    if (s_->chunk.len() == kChunkLen) {
      uint32_t cv[8];
      s_->chunk.output().chaining_value(cv);
      const uint64_t total = s_->chunk.counter + 1;
      push_cv(cv, total);
      s_->chunk.init(kIV, total);
    }
#if defined(__aarch64__)
    // whole chunks, four at a time, while more input follows them (the last chunk stays in chunk for the root)
    if (s_->chunk.len() == 0 && n > 4 * kChunkLen) {
      while (n > 4 * kChunkLen) {
        uint32_t cvs[4][8];
        const uint64_t c0 = s_->chunk.counter;
        hash4_chunks(p, c0, cvs);
        for (int c = 0; c < 4; c++) push_cv(cvs[c], c0 + c + 1);
        s_->chunk.init(kIV, c0 + 4);
        p += 4 * kChunkLen;
        n -= 4 * kChunkLen;
      }
      continue;
    }
#endif
    size_t take = std::min(kChunkLen - s_->chunk.len(), n);
    s_->chunk.update(p, take);
    p += take;
    n -= take;
  }
}

std::string Blake3::hexdigest() {
  Output o = s_->chunk.output();
  for (size_t i = s_->stack.size(); i-- > 0;) {
    uint32_t cv[8];
    o.chaining_value(cv);
    o = parent_output(s_->stack[i].data(), cv);
  }
  uint8_t out[32];
  o.root_bytes(out, 32);
  return hex(out, 32);
}

std::string blake3_hex(std::span<const uint8_t> data) {
  Blake3 h;
  h.update(data.data(), data.size());
  return h.hexdigest();
}

// ---------------------------------------------------------------- MultiModalHasher framing
namespace {
void put_len(Blake3& h, uint64_t v) {
  uint8_t b[8];
  for (int i = 0; i < 8; i++) b[i] = uint8_t(v >> (8 * i));
  h.update(b, 8);
}
void framed(Blake3& h, const void* p, size_t n) {
  put_len(h, n);
  h.update(p, n);
}
void framed_str(Blake3& h, std::string_view s) { framed(h, s.data(), s.size()); }
const uint8_t kTagSequence = 0x01, kTagMapping = 0x02, kTagLeaf = 0x03;
void tag(Blake3& h, uint8_t t) { h.update(&t, 1); }
void leaf_int(Blake3& h, int64_t v) {  // _TAG_LEAF + _framed(np.array(v).tobytes())  (int64, little endian)
  tag(h, kTagLeaf);
  uint8_t b[8];
  for (int i = 0; i < 8; i++) b[i] = uint8_t(uint64_t(v) >> (8 * i));
  framed(h, b, 8);
}
}  // namespace

std::string mm_hash(std::span<const uint8_t> original_bytes, bool converted, const std::string& model_id) {
  // hash_kwargs("blake3", model_id=..., image=MediaWithBytes): keys sorted -> "image", "model_id"
  Blake3 h;
  framed_str(h, "image");
  tag(h, kTagLeaf);  // iter_value_to_bytes(MediaWithBytes) -> leaf -> serialize_item
  framed_str(h, "image");
  if (converted) {
    // {"io_config": {"image_mode": "RGB", "rgba_background_color": (255, 255, 255)}, "data": original_bytes}
    tag(h, kTagMapping);
    put_len(h, 2);
    framed_str(h, "io_config");
    tag(h, kTagMapping);
    put_len(h, 2);
    framed_str(h, "image_mode");
    tag(h, kTagLeaf);
    framed_str(h, "RGB");
    framed_str(h, "rgba_background_color");
    tag(h, kTagSequence);
    put_len(h, 3);
    for (int i = 0; i < 3; i++) leaf_int(h, 255);
    framed_str(h, "data");
    tag(h, kTagLeaf);
    framed(h, original_bytes.data(), original_bytes.size());
  } else {
    tag(h, kTagLeaf);
    framed(h, original_bytes.data(), original_bytes.size());
  }
  framed_str(h, "model_id");
  tag(h, kTagLeaf);
  framed_str(h, model_id);
  return h.hexdigest();
}

// ---------------------------------------------------------------- BLAKE2b
namespace {
constexpr uint64_t kB2IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
                               0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                               0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
constexpr uint8_t kSigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
inline uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

struct B2 {
  uint64_t h[8];
  uint64_t t[2] = {0, 0};
  uint8_t buf[128];
  size_t buflen = 0;
  size_t outlen;
  explicit B2(size_t ol) : outlen(ol) {
    std::memcpy(h, kB2IV, sizeof h);
    h[0] ^= 0x01010000ULL ^ ol;
  }
  void comp(const uint8_t* blk, bool last) {
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) {
      uint64_t x = 0;
      for (int k = 7; k >= 0; k--) x = (x << 8) | blk[8 * i + k];
      m[i] = x;
    }
    for (int i = 0; i < 8; i++) v[i] = h[i], v[i + 8] = kB2IV[i];
    v[12] ^= t[0];
    v[13] ^= t[1];
    if (last) v[14] = ~v[14];
    auto G = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
      v[a] = v[a] + v[b] + x;
      v[d] = rotr64(v[d] ^ v[a], 32);
      v[c] = v[c] + v[d];
      v[b] = rotr64(v[b] ^ v[c], 24);
      v[a] = v[a] + v[b] + y;
      v[d] = rotr64(v[d] ^ v[a], 16);
      v[c] = v[c] + v[d];
      v[b] = rotr64(v[b] ^ v[c], 63);
    };
    for (int r = 0; r < 12; r++) {
      const uint8_t* s = kSigma[r];
      G(0, 4, 8, 12, m[s[0]], m[s[1]]);
      G(1, 5, 9, 13, m[s[2]], m[s[3]]);
      G(2, 6, 10, 14, m[s[4]], m[s[5]]);
      G(3, 7, 11, 15, m[s[6]], m[s[7]]);
      G(0, 5, 10, 15, m[s[8]], m[s[9]]);
      G(1, 6, 11, 12, m[s[10]], m[s[11]]);
      G(2, 7, 8, 13, m[s[12]], m[s[13]]);
      G(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
  }
  void inc(size_t n) {
    t[0] += n;
    if (t[0] < n) t[1]++;
  }
  void update(const uint8_t* p, size_t n) {
    while (n) {
      if (buflen == 128) {  // a full buffered block is compressed only once more data follows
        inc(128);
        comp(buf, false);
        buflen = 0;
      }
      size_t take = std::min<size_t>(128 - buflen, n);
      std::memcpy(buf + buflen, p, take);
      buflen += take;
      p += take;
      n -= take;
    }
  }
  std::string final_hex() {
    inc(buflen);
    std::memset(buf + buflen, 0, 128 - buflen);
    comp(buf, true);
    uint8_t out[64];
    for (int i = 0; i < 8; i++)
      for (int k = 0; k < 8; k++) out[8 * i + k] = uint8_t(h[i] >> (8 * k));
    return hex(out, outlen);
  }
};
}  // namespace

std::string blake2b128_hex(std::span<const std::span<const uint8_t>> parts) {
  B2 b(16);
  for (auto& p : parts) b.update(p.data(), p.size());
  return b.final_hex();
}

std::string content_key(const ImageFeatures& f) {
  // f"{k}:{x.dtype}:{tuple(x.shape)};" + raw bytes, for patches, vit_grid, llm_grid, types (per-item tensors)
  const std::string pk = "patches:torch.bfloat16:(" + std::to_string(f.n_patches) + ", 3, 14, 14);";
  const std::string vk = "vit_grid:torch.int64:(2,);";
  const std::string lk = "llm_grid:torch.int64:(2,);";
  const std::string tk = "types:torch.int64:(" + std::to_string(f.types.size()) + ",);";
  auto bytes = [](const void* p, size_t n) { return std::span<const uint8_t>(static_cast<const uint8_t*>(p), n); };
  // (little-endian host: the tensors' memory is the bytes hashed)
  std::span<const uint8_t> parts[] = {
      bytes(pk.data(), pk.size()), bytes(f.patches_bf16.data(), f.patches_bf16.size() * 2),
      bytes(vk.data(), vk.size()), bytes(f.vit_grid, 16),
      bytes(lk.data(), lk.size()), bytes(f.llm_grid, 16),
      bytes(tk.data(), tk.size()), bytes(f.types.data(), f.types.size() * 8)};
  return "img-" + blake2b128_hex(parts);
}

// ---------------------------------------------------------------- SHA-256
std::array<uint8_t, 32> sha256(std::span<const uint8_t> data) {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t H[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto block = [&](const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g2 = H[6], h = H[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g2);
      uint32_t t1 = h + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      h = g2, g2 = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    H[0] += a, H[1] += b, H[2] += c, H[3] += d, H[4] += e, H[5] += f, H[6] += g2, H[7] += h;
  };
  size_t n = data.size();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) block(data.data() + i);
  uint8_t tail[128] = {};
  size_t r = n - i;
  std::memcpy(tail, data.data() + i, r);
  tail[r] = 0x80;
  size_t tl = r + 1 + 8 <= 64 ? 64 : 128;
  uint64_t bits = uint64_t(n) * 8;
  for (int k = 0; k < 8; k++) tail[tl - 1 - k] = uint8_t(bits >> (8 * k));
  block(tail);
  if (tl == 128) block(tail + 64);
  std::array<uint8_t, 32> out;
  for (int k = 0; k < 8; k++)
    for (int j = 0; j < 4; j++) out[4 * k + j] = uint8_t(H[k] >> (24 - 8 * j));
  return out;
}

std::string sha256_hex(std::span<const uint8_t> data) {
  auto d = sha256(data);
  return hex(d.data(), d.size());
}

}  // namespace fe::img
