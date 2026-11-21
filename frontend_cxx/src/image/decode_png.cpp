// PNG: a port of Pillow 13 PngImagePlugin (chunk walk, CRC rules, info / transparency / text / eXIf handling,
// load_read + load_end) and libImaging/ZipDecode.c plus the
// Unpack.c rawmodes of PngImagePlugin._MODES. Pillow does not use libpng for decoding, so neither do we: libpng's
// stricter checks (IDAT CRCs, palette indices, trailing data) would reject files Pillow accepts.
#include <zlib.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "image/image_internal.h"

namespace fe::img {
namespace {

[[noreturn]] void unidentified() {
  throw ImageError(ImageError::kUnidentified, "Failed to load image: cannot identify image file");
}
[[noreturn]] void truncated(const char* m = "image file is truncated") {
  throw ImageError(ImageError::kTruncated, std::string("Failed to load image: ") + m);
}
[[noreturn]] void broken(const char* m) { throw ImageError(ImageError::kDecode, std::string("Failed to load image: ") + m); }

uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }
int be16(const uint8_t* p) { return (p[0] << 8) | p[1]; }

bool is_cid(const uint8_t* c) {  // re.compile(rb"\w\w\w\w").match
  for (int i = 0; i < 4; i++) {
    uint8_t ch = c[i];
    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')) return false;
  }
  return true;
}

struct RawMode {
  const char* mode;
  const char* raw;
  int bits;  // bits per pixel of the rawmode
};

const RawMode* png_mode(int bitdepth, int ctype) {
  static const struct {
    int bd, ct;
    RawMode m;
  } T[] = {
      {1, 0, {"1", "1", 1}},        {2, 0, {"L", "L;2", 2}},           {4, 0, {"L", "L;4", 4}},
      {8, 0, {"L", "L", 8}},        {16, 0, {"I;16", "I;16B", 16}},    {8, 2, {"RGB", "RGB", 24}},
      {16, 2, {"RGB", "RGB;16B", 48}}, {1, 3, {"P", "P;1", 1}},        {2, 3, {"P", "P;2", 2}},
      {4, 3, {"P", "P;4", 4}},      {8, 3, {"P", "P", 8}},             {8, 4, {"LA", "LA", 16}},
      {16, 4, {"RGBA", "LA;16B", 32}}, {8, 6, {"RGBA", "RGBA", 32}},   {16, 6, {"RGBA", "RGBA;16B", 64}},
  };
  for (auto& t : T)
    if (t.bd == bitdepth && t.ct == ctype) return &t.m;
  return nullptr;
}

int mode_bytes(const std::string& m) {
  if (m == "1" || m == "L" || m == "P") return 1;
  if (m == "LA" || m == "I;16") return 2;
  if (m == "RGB") return 3;
  return 4;
}

// _safe_zlib_decompress(s): one decompressobj().decompress(s, MAX_TEXT_CHUNK); returns false on zlib.error
// (or "too large", which Pillow raises as ValueError: treated the same, a corner we document).
bool zlib_text(std::span<const uint8_t> in, std::vector<uint8_t>& out) {
  z_stream z{};
  if (inflateInit(&z) != Z_OK) return false;
  out.clear();
  const size_t kMax = 1 << 20;  // ImageFile.SAFEBLOCK
  std::vector<uint8_t> buf(64 * 1024);
  z.next_in = const_cast<uint8_t*>(in.data());
  z.avail_in = static_cast<uInt>(in.size());
  int rc = Z_OK;
  while (true) {
    z.next_out = buf.data();
    z.avail_out = static_cast<uInt>(buf.size());
    rc = inflate(&z, Z_NO_FLUSH);
    out.insert(out.end(), buf.data(), buf.data() + (buf.size() - z.avail_out));
    if (rc == Z_STREAM_END || rc == Z_BUF_ERROR) break;
    if (rc != Z_OK) {
      inflateEnd(&z);
      return false;
    }
    if (out.size() > kMax) break;
    if (z.avail_in == 0 && z.avail_out != 0) break;
  }
  inflateEnd(&z);
  if (out.size() > kMax) out.resize(kMax);
  return true;
}

std::string latin1_to_utf8(std::span<const uint8_t> s) {
  std::string o;
  for (uint8_t c : s) {
    if (c < 0x80) {
      o.push_back(char(c));
    } else {
      o.push_back(char(0xC0 | (c >> 6)));
      o.push_back(char(0x80 | (c & 0x3F)));
    }
  }
  return o;
}

bool valid_utf8(std::span<const uint8_t> s) {  // bytes.decode("utf-8", "strict") succeeds
  size_t i = 0;
  while (i < s.size()) {
    const uint8_t c = s[i];
    if (c < 0x80) {
      i++;
      continue;
    }
    int n;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0) {
      n = 1, cp = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
      n = 2, cp = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
      n = 3, cp = c & 0x07;
    } else {
      return false;
    }
    if (i + n >= s.size()) return false;
    for (int k = 1; k <= n; k++) {
      if ((s[i + k] & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && (cp < 0x10000 || cp > 0x10FFFF)) ||
        (cp >= 0xD800 && cp <= 0xDFFF))
      return false;
    i += n + 1;
  }
  return true;
}

struct Png {
  std::span<const uint8_t> b;
  Native& n;
  const RawMode* rm = nullptr;
  std::string im_mode;  // PngStream.im_mode (post-IDAT IHDR changes it, not the image)
  int w = 0, h = 0;
  bool interlace = false;
  bool ihdr_ok = false;
  std::optional<std::vector<uint8_t>> plte;  // im_palette
  std::optional<int> n_frames;               // APNG acTL
  std::optional<int> seq_num;
  bool has_bbox = false;
  int bbox[4] = {0, 0, 0, 0};
  bool default_image = false;

  explicit Png(std::span<const uint8_t> bb, Native& nn) : b(bb), n(nn) {}

  // ---- chunk handlers shared by the pre-IDAT walk (_open) and the post-IDAT walk (load_end).
  // Return false for "EOFError" (IDAT / IEND), throw for the exceptions Pillow raises.
  enum Kind { kHandled, kUnknown, kEof };

  Kind handle(const uint8_t* cid, std::span<const uint8_t> s, bool post) {
    auto is = [&](const char* t) { return std::memcmp(cid, t, 4) == 0; };
    if (is("IHDR")) {
      if (s.size() < 13) throw ImageError(ImageError::kDecode, "Failed to load image: Truncated IHDR chunk");
      w = static_cast<int>(be32(&s[0]));
      h = static_cast<int>(be32(&s[4]));
      if (be32(&s[0]) > 0x7FFFFFFF || be32(&s[4]) > 0x7FFFFFFF) unidentified();
      if (auto m = png_mode(s[8], s[9])) {
        if (!post) rm = m;
        im_mode = m->mode;
      }
      if (s[12]) interlace = true;
      if (s[11]) unidentified();  // "unknown filter category"
      ihdr_ok = true;
      return kHandled;
    }
    if (is("IDAT") || is("IEND")) return kEof;
    if (is("PLTE")) {
      if (im_mode == "P") plte = std::vector<uint8_t>(s.begin(), s.end());
      return kHandled;
    }
    if (is("tRNS")) {
      const std::string& m = im_mode;
      if (m == "P") {
        // _simple_palette = re.compile(b"^\xff*\x00\xff*$").match (note: "$" also matches before a final "\n")
        auto simple = [](std::span<const uint8_t> t) {
          size_t zeros = 0;
          for (uint8_t v : t) {
            if (v == 0) {
              zeros++;
            } else if (v != 0xFF) {
              return false;
            }
          }
          return zeros == 1;
        };
        const bool is_simple = simple(s) || (!s.empty() && s.back() == '\n' && simple(s.first(s.size() - 1)));
        if (is_simple) {
          n.trns = Native::kInt;
          n.trns_int = static_cast<int>(std::find(s.begin(), s.end(), uint8_t{0}) - s.begin());
        } else {
          n.trns = Native::kBytes;
          n.trns_bytes.assign(s.begin(), s.end());
        }
      } else if (m == "1" || m == "L" || m == "I;16" || m == "RGB") {
        if (s.size() < (m == "RGB" ? 6u : 2u))
          throw ImageError(ImageError::kDecode, "Failed to load image: Truncated tRNS chunk");
        if (m == "1") {
          n.trns = Native::kInt;
          n.trns_int = be16(&s[0]) ? 255 : 0;
        } else if (m == "RGB") {
          n.trns = Native::kTriple;
          n.trns_rgb[0] = be16(&s[0]);
          n.trns_rgb[1] = be16(&s[2]);
          n.trns_rgb[2] = be16(&s[4]);
        } else {
          n.trns = Native::kInt;
          n.trns_int = be16(&s[0]);
        }
      }
      return kHandled;
    }
    if (is("gAMA") || is("cHRM") || is("sRGB") || is("pHYs")) {
      size_t need = is("gAMA") ? 4 : is("cHRM") ? 32 : is("sRGB") ? 1 : 9;
      if (s.size() < need) throw ImageError(ImageError::kDecode, "Failed to load image: Truncated chunk");
      return kHandled;
    }
    if (is("iCCP")) {
      auto nul = std::find(s.begin(), s.end(), uint8_t{0});
      long i = nul == s.end() ? -1 : long(nul - s.begin());
      // comp_method = s[i + 1]  (i == -1 -> s[0])
      size_t ci = static_cast<size_t>(i + 1);
      if (ci >= s.size()) unidentified();  // IndexError
      if (s[ci] != 0) unidentified();      // "Unknown compression method"
      return kHandled;
    }
    if (is("tEXt")) {
      auto nul = std::find(s.begin(), s.end(), uint8_t{0});
      std::span<const uint8_t> k(s.begin(), nul);
      std::span<const uint8_t> v = nul == s.end() ? std::span<const uint8_t>() : std::span<const uint8_t>(nul + 1, s.end());
      if (!k.empty()) set_text(k, v, /*bytes_exif=*/true);
      return kHandled;
    }
    if (is("zTXt")) {
      auto nul = std::find(s.begin(), s.end(), uint8_t{0});
      std::span<const uint8_t> k(s.begin(), nul);
      std::span<const uint8_t> v = nul == s.end() ? std::span<const uint8_t>() : std::span<const uint8_t>(nul + 1, s.end());
      int comp = v.empty() ? 0 : v[0];
      if (comp != 0) unidentified();  // SyntaxError
      std::vector<uint8_t> plain;
      if (!zlib_text(v.empty() ? v : v.subspan(1), plain)) plain.clear();
      if (!k.empty()) set_text(k, plain, /*bytes_exif=*/false);
      return kHandled;
    }
    if (is("iTXt")) {
      auto nul = std::find(s.begin(), s.end(), uint8_t{0});
      if (nul == s.end()) return kHandled;
      std::span<const uint8_t> k(s.begin(), nul);
      std::span<const uint8_t> r(nul + 1, s.end());
      if (r.size() < 2) return kHandled;
      int cf = r[0], cm = r[1];
      r = r.subspan(2);
      auto n1 = std::find(r.begin(), r.end(), uint8_t{0});
      if (n1 == r.end()) return kHandled;
      auto n2 = std::find(n1 + 1, r.end(), uint8_t{0});
      if (n2 == r.end()) return kHandled;
      std::span<const uint8_t> lang(r.begin(), n1), tk(n1 + 1, n2), v(n2 + 1, r.end());
      std::vector<uint8_t> plain(v.begin(), v.end());
      if (cf != 0) {
        if (cm != 0) return kHandled;
        if (!zlib_text(v, plain)) return kHandled;
      }
      bool is_xmp = k.size() == 17 && std::memcmp(k.data(), "XML:com.adobe.xmp", 17) == 0;
      if (is_xmp) n.xmp_bytes = plain;
      if (!valid_utf8(lang) || !valid_utf8(tk) || !valid_utf8(plain)) return kHandled;
      std::string key(k.begin(), k.end());
      std::string val(plain.begin(), plain.end());
      if (is_xmp) n.xmp_str = val;
      if (key == "Raw profile type exif") n.raw_profile_exif = val;
      if (key == "exif") n.exif_unsupported_str = true, n.exif.reset();
      return kHandled;
    }
    if (is("eXIf")) {
      std::vector<uint8_t> e = {'E', 'x', 'i', 'f', 0, 0};
      e.insert(e.end(), s.begin(), s.end());
      n.exif = std::move(e);
      n.exif_unsupported_str = false;
      return kHandled;
    }
    if (is("acTL")) {
      if (s.size() < 8) throw ImageError(ImageError::kDecode, "Failed to load image: APNG contains truncated acTL chunk");
      if (n_frames) {
        n_frames.reset();
        return kHandled;
      }
      uint32_t nf = be32(&s[0]);
      if (nf == 0 || nf > 0x80000000u) return kHandled;
      n_frames = static_cast<int>(nf);
      return kHandled;
    }
    if (is("fcTL")) {
      if (s.size() < 26) throw ImageError(ImageError::kDecode, "Failed to load image: APNG contains truncated fcTL chunk");
      int seq = static_cast<int>(be32(&s[0]));
      if ((!seq_num && seq != 0) || (seq_num && *seq_num != seq - 1)) {
        if (!post) unidentified();
        throw ImageError(ImageError::kDecode, "Failed to load image: APNG contains frame sequence errors");
      }
      seq_num = seq;
      uint32_t fw = be32(&s[4]), fh = be32(&s[8]), px = be32(&s[12]), py = be32(&s[16]);
      if (uint64_t(px) + fw > uint64_t(w) || uint64_t(py) + fh > uint64_t(h)) {
        if (!post) unidentified();
        throw ImageError(ImageError::kDecode, "Failed to load image: APNG contains invalid frames");
      }
      if (!post) {
        has_bbox = true;
        bbox[0] = int(px), bbox[1] = int(py), bbox[2] = int(px + fw), bbox[3] = int(py + fh);
      }
      return kHandled;
    }
    if (is("fdAT")) {
      // only reachable before the first IDAT in malformed files (APNG data without a default image)
      throw ImageError(ImageError::kUnsupported, "Failed to load image: APNG fdAT before IDAT is not supported");
    }
    return kUnknown;
  }

  void set_text(std::span<const uint8_t> k, std::span<const uint8_t> v, bool bytes_exif) {
    std::string key(k.begin(), k.end());
    if (key == "exif") {
      if (bytes_exif) {  // tEXt "exif": info["exif"] = v (bytes)
        n.exif = std::vector<uint8_t>(v.begin(), v.end());
        n.exif_unsupported_str = false;
      } else {  // zTXt "exif": a str -> Exif.load raises TypeError -> no transpose
        n.exif.reset();
        n.exif_unsupported_str = true;
      }
      return;
    }
    std::string val = latin1_to_utf8(v);  // v.decode("latin-1")
    if (key == "Raw profile type exif") n.raw_profile_exif = val;
    if (key == "XML:com.adobe.xmp") n.xmp_str = val;
  }
};

// ---- ZipDecode.c port
const int kStartRow[7] = {0, 0, 4, 0, 2, 0, 1};
const int kStartCol[7] = {0, 4, 0, 2, 0, 1, 0};
const int kRowInc[7] = {8, 8, 8, 4, 4, 2, 2};
const int kColInc[7] = {8, 8, 4, 4, 2, 2, 1};

struct Zip {
  int xsize, ysize, bits;  // tile size, rawmode bits per pixel
  bool interlaced;
  int pass = 0, y = 0;
  int last_output = 0;
  size_t full_row;  // bytes per full-width row
  std::vector<uint8_t> buffer, previous;
  std::vector<uint8_t>& raw;  // decoded raw rows of the full tile, packed at rawmode depth
  z_stream z{};
  bool init = false, done = false;

  Zip(int xs, int ys, int b, bool il, std::vector<uint8_t>& out) : xsize(xs), ysize(ys), bits(b), interlaced(il), raw(out) {
    full_row = (static_cast<size_t>(xsize) * bits + 7) / 8;
    raw.assign(full_row * ysize, 0);
  }
  ~Zip() {
    if (init) inflateEnd(&z);
  }
  int row_len_for(int p) const {
    long cols = (static_cast<long>(xsize) + (kColInc[p] - 1 - kStartCol[p])) / kColInc[p];
    return static_cast<int>((cols * bits + 7) / 8);
  }
  int row_len() const { return interlaced ? row_len_for(pass) : static_cast<int>(full_row); }

  // place one decoded (unfiltered) pass row at row y
  void put_row(const uint8_t* src) {
    if (!interlaced) {
      std::memcpy(raw.data() + full_row * y, src, full_row);
      return;
    }
    const int c0 = kStartCol[pass], ci = kColInc[pass];
    uint8_t* dst = raw.data() + full_row * y;
    if (bits >= 8) {
      const int bpp = bits / 8;
      for (int i = 0, x = c0; x < xsize; i++, x += ci) std::memcpy(dst + size_t(x) * bpp, src + size_t(i) * bpp, bpp);
    } else {
      const int per = 8 / bits, mask = (1 << bits) - 1;
      for (int i = 0, x = c0; x < xsize; i++, x += ci) {
        int v = (src[i / per] >> (8 - bits * (i % per + 1))) & mask;
        uint8_t& d = dst[x / per];
        int sh = 8 - bits * (x % per + 1);
        d = static_cast<uint8_t>((d & ~(mask << sh)) | (v << sh));
      }
    }
  }

  void next_pass_start() {
    // skip empty passes: row length 0 or no rows
    while (pass < 7 && (row_len_for(pass) == 0 || ysize <= kStartRow[pass])) pass++;
    if (pass < 7) y = kStartRow[pass];
  }

  // returns true when the image is complete (Pillow returns -1 with errcode 0)
  bool feed(const uint8_t* buf, size_t bytes) {
    if (!init) {
      if (inflateInit(&z) != Z_OK) broken("zlib init");
      init = true;
      buffer.assign(full_row + 1, 0);
      previous.assign(full_row + 1, 0);
      if (interlaced) {
        pass = 0;
        y = kStartRow[0];
        // Pillow starts at pass 0 and only skips empty passes after a pass ends; an empty pass 0 (row_len 0)
        // is skipped up front the same way.
        next_pass_start();
      }
      if (interlaced && pass >= 7) {
        done = true;
        return true;
      }
    }
    int rl = row_len();
    z.next_in = const_cast<uint8_t*>(buf);
    z.avail_in = static_cast<uInt>(bytes);
    while (z.avail_in > 0) {
      z.next_out = buffer.data() + last_output;
      z.avail_out = static_cast<uInt>(rl + 1 - last_output);
      int err = inflate(&z, Z_NO_FLUSH);
      if (err < 0) broken("broken data stream when reading image file");
      int got = rl + 1 - static_cast<int>(z.avail_out);
      if (got < rl + 1) {
        last_output = got;
        break;  // need more input
      }
      // predictor
      uint8_t* B = buffer.data();
      const uint8_t* P = previous.data();
      const int bpp = (bits + 7) / 8;
      switch (B[0]) {
        case 0:
          break;
        case 1:
          for (int i = bpp + 1; i <= rl; i++) B[i] = uint8_t(B[i] + B[i - bpp]);
          break;
        case 2:
          for (int i = 1; i <= rl; i++) B[i] = uint8_t(B[i] + P[i]);
          break;
        case 3: {
          int i = 1;
          for (; i <= bpp && i <= rl; i++) B[i] = uint8_t(B[i] + P[i] / 2);
          for (; i <= rl; i++) B[i] = uint8_t(B[i] + (B[i - bpp] + P[i]) / 2);
          break;
        }
        case 4: {
          int i = 1;
          for (; i <= bpp && i <= rl; i++) B[i] = uint8_t(B[i] + P[i]);
          for (; i <= rl; i++) {
            int a = B[i - bpp], bb = P[i], c = P[i - bpp];
            int pa = std::abs(bb - c), pb = std::abs(a - c), pc = std::abs(a + bb - 2 * c);
            B[i] = uint8_t(B[i] + ((pa <= pb && pa <= pc) ? a : (pb <= pc) ? bb : c));
          }
          break;
        }
        default:
          throw ImageError(ImageError::kDecode, "Failed to load image: unknown PNG filter type");
      }
      put_row(B + 1);
      last_output = 0;
      if (interlaced) {
        y += kRowInc[pass];
        if (y >= ysize) {
          // next pass: the previous row buffer is cleared
          pass++;
          next_pass_start();
          if (pass >= 7) {
            done = true;
            return true;
          }
          rl = row_len();
          std::fill(buffer.begin(), buffer.end(), 0);
          std::fill(previous.begin(), previous.end(), 0);
          continue;  // (buffer/previous both zero: no swap needed)
        }
      } else {
        y++;
        if (y >= ysize) {
          done = true;
          return true;
        }
      }
      std::swap(buffer, previous);
    }
    return false;
  }
};

// Unpack.c for the PNG rawmodes -> Pillow mode samples
void unpack(const RawMode* rm, const std::vector<uint8_t>& raw, int w, int h, size_t row_bytes, std::vector<uint8_t>& px,
            int out_w, int x0, int y0) {
  const std::string r = rm->raw;
  const int ob = mode_bytes(rm->mode);
  for (int y = 0; y < h; y++) {
    const uint8_t* s = raw.data() + row_bytes * y;
    uint8_t* d = px.data() + (static_cast<size_t>(y + y0) * out_w + x0) * ob;
    if (r == "1" || r == "L;2" || r == "L;4" || r == "P;1" || r == "P;2" || r == "P;4") {
      const int bits = rm->bits, per = 8 / bits, mask = (1 << bits) - 1;
      const int mul = r == "1" ? 255 : r == "L;2" ? 0x55 : r == "L;4" ? 0x11 : 1;
      for (int x = 0; x < w; x++) {
        int v = (s[x / per] >> (8 - bits * (x % per + 1))) & mask;
        d[x] = static_cast<uint8_t>(r == "1" ? (v ? 255 : 0) : v * mul);
      }
    } else if (r == "L" || r == "P" || r == "RGB" || r == "LA" || r == "RGBA") {
      std::memcpy(d, s, static_cast<size_t>(w) * ob);
    } else if (r == "I;16B") {
      for (int x = 0; x < w; x++) {
        d[2 * x] = s[2 * x + 1];
        d[2 * x + 1] = s[2 * x];
      }
    } else if (r == "RGB;16B") {
      for (int x = 0; x < w * 3; x++) d[x] = s[2 * x];
    } else if (r == "RGBA;16B") {
      for (int x = 0; x < w * 4; x++) d[x] = s[2 * x];
    } else if (r == "LA;16B") {  // unpackRGBALA16B: L MSB replicated, A MSB
      for (int x = 0; x < w; x++) {
        d[4 * x] = d[4 * x + 1] = d[4 * x + 2] = s[4 * x];
        d[4 * x + 3] = s[4 * x + 2];
      }
    }
  }
}

}  // namespace

Native decode_png(std::span<const uint8_t> b, const DecodeOpts& opts) {
  Native n;
  n.format = "PNG";
  static const uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  if (b.size() < 8 || std::memcmp(b.data(), kSig, 8) != 0) unidentified();
  Png png(b, n);
  size_t pos = 8;
  size_t idat_pos = 0, idat_len = 0;
  // ---- _open: chunks up to the first IDAT, every one CRC-checked
  while (true) {
    if (b.size() - pos < 8) unidentified();  // struct.error / SyntaxError("broken PNG file")
    const uint32_t length = be32(&b[pos]);
    const uint8_t* cid = &b[pos + 4];
    if (!is_cid(cid)) unidentified();
    const size_t dpos = pos + 8;
    auto kind_probe = std::memcmp(cid, "IDAT", 4) == 0 || std::memcmp(cid, "IEND", 4) == 0;
    if (kind_probe) {
      if (std::memcmp(cid, "IEND", 4) == 0) {
        // EOFError before any IDAT: no tile -> ImageFile.load "cannot load this image"
        if (!png.ihdr_ok || png.im_mode.empty() || png.w <= 0 || png.h <= 0) unidentified();
        throw ImageError(ImageError::kDecode, "Failed to load image: cannot load this image");
      }
      idat_pos = dpos;
      idat_len = length;
      break;
    }
    if (b.size() - dpos < length) truncated("Truncated File Read");  // _safe_read
    std::span<const uint8_t> s(&b[dpos], length);
    png.handle(cid, s, /*post=*/false);
    // crc
    if (b.size() - (dpos + length) < 4) unidentified();
    uint32_t crc = static_cast<uint32_t>(crc32(crc32(0, cid, 4), s.data(), static_cast<uInt>(s.size())));
    if (crc != be32(&b[dpos + length])) unidentified();  // "broken PNG file (bad header checksum)"
    pos = dpos + length + 4;
  }
  // ImageFile.__init__ checks
  n.mode = png.im_mode;
  if (n.mode.empty() || png.w <= 0 || png.h <= 0 || !png.rm) unidentified();
  check_size(png.w, png.h, opts);
  if (png.n_frames && !png.has_bbox) png.default_image = true;
  n.w = png.w;
  n.h = png.h;
  if (png.plte) {
    if (png.plte->size() / 3 > 256)
      throw ImageError(ImageError::kDecode, "Failed to load image: invalid palette size");  // putpalette ValueError
    n.has_palette = true;
    n.palette.assign(png.plte->begin(), png.plte->begin() + (png.plte->size() / 3) * 3);
  }
  // ---- decode: ImageFile.load + PngImageFile.load_read (65536-byte reads that never cross a chunk)
  int x0 = 0, y0 = 0, tw = png.w, th = png.h;
  if (png.has_bbox) {
    x0 = png.bbox[0], y0 = png.bbox[1], tw = png.bbox[2] - png.bbox[0], th = png.bbox[3] - png.bbox[1];
  }
  const int ob = mode_bytes(n.mode);
  n.px.assign(static_cast<size_t>(n.w) * n.h * ob, 0);
  std::vector<uint8_t> raw;
  bool complete = false;
  size_t fp = idat_pos;                // file position
  size_t idat_remaining = idat_len;    // PngImageFile.__idat
  if (tw > 0 && th > 0) {
    Zip zip(tw, th, png.rm->bits, png.interlace, raw);
    while (true) {
      // load_read(65536)
      bool eof = false;
      while (idat_remaining == 0) {
        if (b.size() - fp < 4) {
          fp = b.size();
          eof = true;
          break;
        }
        fp += 4;  // CRC
        if (b.size() - fp < 8) truncated();  // png.read(): struct.error -> "image file is truncated"
        uint32_t len = be32(&b[fp]);
        const uint8_t* cid = &b[fp + 4];
        if (!is_cid(cid)) {  // SyntaxError out of load_read: raised inside ImageFile.load's loop (hard)
          throw ImageError(ImageError::kTruncated, "Failed to load image: broken PNG file");
        }
        bool data_chunk = std::memcmp(cid, "IDAT", 4) == 0 || std::memcmp(cid, "DDAT", 4) == 0 ||
                          std::memcmp(cid, "fdAT", 4) == 0;
        if (!data_chunk) {
          eof = true;  // pushed back; returns b""
          break;
        }
        fp += 8;
        if (std::memcmp(cid, "fdAT", 4) == 0) {
          if (len < 4) truncated();
          fp += 4;
          idat_remaining = len - 4;
        } else {
          idat_remaining = len;
        }
      }
      if (eof) truncated();
      size_t want = std::min<size_t>(65536, idat_remaining);
      size_t got = std::min(want, b.size() - fp);
      idat_remaining -= want;
      if (got == 0) truncated();
      const uint8_t* blk = &b[fp];
      fp += got;
      // A decoder error (zlib data error, unknown filter type) is soft: ImageFile.load raises it after clearing
      // the tile and running load_end, so vLLM keeps the rows decoded so far (and skips the EXIF transpose).
      try {
        if (zip.feed(blk, got)) {
          complete = true;
          break;
        }
      } catch (const ImageError& e) {
        if (e.kind != ImageError::kDecode) throw;
        n.load_error = true;
        break;
      }
      if (got < want) {
        // fp.read hit EOF: the next read returns b"" -> truncated
        truncated();
      }
    }
    unpack(png.rm, raw, tw, th, zip.full_row, n.px, n.w, x0, y0);
  } else {
    complete = true;
  }
  (void)complete;
  // ---- load_end: chunks after the image data, no CRC checks. An exception here propagates out of the first
  // image.load() (inside ImageOps.exif_transpose, where vLLM's normalize_image suppresses it), so the EXIF
  // transpose is skipped.
  bool load_end_raised = false;
  try {
    size_t p = fp + 4;  // self.fp.read(4)  # CRC  (assumes the last read ended at a chunk end)
    if (p > b.size()) p = b.size();
    const bool animated = png.n_frames && (*png.n_frames + (png.default_image ? 1 : 0)) > 1;
    while (true) {
      if (b.size() - p < 8) break;  // struct.error -> break
      uint32_t length = be32(&b[p]);
      const uint8_t* cid = &b[p + 4];
      if (!is_cid(cid)) break;  // SyntaxError -> break
      size_t dpos = p + 8;
      if (std::memcmp(cid, "IEND", 4) == 0) break;
      if (std::memcmp(cid, "fcTL", 4) == 0 && animated) break;
      if (std::memcmp(cid, "IDAT", 4) == 0 || std::memcmp(cid, "fdAT", 4) == 0) {
        // EOFError -> _safe_read(length) (fdAT: length - 4 after the handler consumed the sequence number)
        size_t skip = length;
        if (std::memcmp(cid, "fdAT", 4) == 0) {
          // chunk_fdAT: ValueError when < 4 bytes, reads the sequence number, SyntaxError on sequence errors
          if (length < 4) throw ImageError(ImageError::kDecode, "APNG contains truncated fDAT chunk");
          if (b.size() - dpos < 4) throw ImageError(ImageError::kTruncated, "Truncated File Read");
          int seq = static_cast<int>(be32(&b[dpos]));
          if (!png.seq_num || *png.seq_num != seq - 1)
            throw ImageError(ImageError::kDecode, "APNG contains frame sequence errors");
          png.seq_num = seq;
          dpos += 4;
          skip = length >= 4 ? length - 4 : 0;
        }
        if (b.size() - dpos < skip) throw ImageError(ImageError::kTruncated, "Truncated File Read");
        p = dpos + skip + 4;
        continue;
      }
      if (b.size() - dpos < length) throw ImageError(ImageError::kTruncated, "Truncated File Read");
      std::span<const uint8_t> s(&b[dpos], length);
      png.handle(cid, s, /*post=*/true);
      p = dpos + length + 4;
    }
  } catch (const ImageError&) {
    load_end_raised = true;
  }
  if (load_end_raised) n.load_error = true;  // the first image.load() raised: no EXIF transpose
  return n;
}

}  // namespace fe::img
