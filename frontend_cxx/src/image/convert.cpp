// vLLM ImageMediaIO._convert_image_mode / convert_image_mode / rgba_to_rgb on top of Pillow 13's Convert.c and
// Paste.c formulas, and ImageOps.exif_transpose (Image.getexif + TiffImagePlugin.ImageFileDirectory_v2 rules).
#include <cmath>
#include <cstring>
#include <optional>
#include <algorithm>
#include <string>
#include <vector>

#include "image/image_internal.h"

namespace fe::img {
namespace {

inline uint8_t div255(int a) {  // DIV255 / MULDIV255 of ImagingUtils.h
  int t = a + 128;
  return static_cast<uint8_t>(((t >> 8) + t) >> 8);
}

// Paste.c paste_mask_L: out = BLEND(mask, out, in) on a white (255) canvas
inline uint8_t over_white(uint8_t c, uint8_t a) { return div255(255 * (255 - a) + c * a); }

// palette entries as the C image holds them after putpalette("RGB", data): ImagingPaletteNew zero-fills
// (0, 0, 0, 255) and size = len(data) // 3 entries are copied in.
struct Pal {
  uint8_t rgba[256][4];
};

Pal make_palette(const Native& n) {
  Pal p;
  for (int i = 0; i < 256; i++) p.rgba[i][0] = p.rgba[i][1] = p.rgba[i][2] = 0, p.rgba[i][3] = 255;
  size_t entries = std::min<size_t>(256, n.palette.size() / 3);
  for (size_t i = 0; i < entries; i++) {
    p.rgba[i][0] = n.palette[3 * i];
    p.rgba[i][1] = n.palette[3 * i + 1];
    p.rgba[i][2] = n.palette[3 * i + 2];
  }
  return p;
}

}  // namespace

Rgb to_rgb(Native& n, bool* converted) {
  Rgb o;
  o.w = n.w;
  o.h = n.h;
  const size_t np = static_cast<size_t>(n.w) * n.h;
  const std::string& m = n.mode;
  *converted = m != "RGB";
  if (m == "RGB") {
    o.px = std::move(n.px);  // transparency info is ignored for RGB (image.mode == image_mode)
    return o;
  }
  o.px.resize(np * 3);
  uint8_t* d = o.px.data();
  const uint8_t* s = n.px.data();
  if (m == "RGBA") {  // rgba_to_rgb
    for (size_t i = 0; i < np; i++) {
      const uint8_t a = s[4 * i + 3];
      d[3 * i] = over_white(s[4 * i], a);
      d[3 * i + 1] = over_white(s[4 * i + 1], a);
      d[3 * i + 2] = over_white(s[4 * i + 2], a);
    }
    return o;
  }
  const bool has_trns = m == "LA" || m == "PA" || n.trns != Native::kNone;
  if (m == "LA") {  // convert("RGBA") (la2rgba) then rgba_to_rgb
    for (size_t i = 0; i < np; i++) {
      uint8_t v = over_white(s[2 * i], s[2 * i + 1]);
      d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = v;
    }
    return o;
  }
  if (m == "P") {
    Pal p = make_palette(n);
    if (has_trns) {
      // Image.convert("RGBA"): putpalettealphas(bytes) / putpalettealpha(index, 0), then p2rgba
      if (n.trns == Native::kBytes) {
        if (n.trns_bytes.size() > 256)
          throw ImageError(ImageError::kDecode, "Failed to load image: invalid palette size");  // ValueError
        for (size_t i = 0; i < n.trns_bytes.size(); i++) p.rgba[i][3] = n.trns_bytes[i];
      } else if (n.trns == Native::kInt) {
        if (n.trns_int < 0 || n.trns_int >= 256)
          throw ImageError(ImageError::kDecode, "Failed to load image: palette index out of range");
        p.rgba[n.trns_int][3] = 0;
      }
      for (size_t i = 0; i < np; i++) {
        const uint8_t* e = p.rgba[s[i]];
        d[3 * i] = over_white(e[0], e[3]);
        d[3 * i + 1] = over_white(e[1], e[3]);
        d[3 * i + 2] = over_white(e[2], e[3]);
      }
    } else {
      for (size_t i = 0; i < np; i++) {
        const uint8_t* e = p.rgba[s[i]];
        d[3 * i] = e[0], d[3 * i + 1] = e[1], d[3 * i + 2] = e[2];
      }
    }
    return o;
  }
  if (m == "1" || m == "L" || m == "I;16") {
    // base conversion: bit2rgb / l2rgb / I16_RGB (clip to 255, Pillow does not scale)
    auto base = [&](size_t i) -> uint8_t {
      if (m == "1") return s[i] ? 255 : 0;
      if (m == "L") return s[i];
      return s[2 * i + 1] == 0 ? s[2 * i] : 255;
    };
    if (has_trns) {
      // convert_transparent("RGBA", trns): rgbT2rgba compares with (trns & 0xff) replicated, alpha 0 on a match
      const int t = (n.trns == Native::kInt ? n.trns_int : 0) & 0xFF;
      for (size_t i = 0; i < np; i++) {
        uint8_t v = base(i);
        uint8_t r = v == t ? 255 : v;  // alpha 0 -> the white background, alpha 255 -> the pixel
        d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = r;
      }
    } else {
      for (size_t i = 0; i < np; i++) {
        uint8_t v = base(i);
        d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = v;
      }
    }
    return o;
  }
  if (m == "CMYK") {  // cmyk2rgb
    for (size_t i = 0; i < np; i++) {
      const uint8_t* c = s + 4 * i;
      const int nk = 255 - c[3];
      for (int k = 0; k < 3; k++) {
        int v = nk - div255(c[k] * nk);
        d[3 * i + k] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
      }
    }
    return o;
  }
  throw ImageError(ImageError::kInternal, "unsupported mode " + m);
}

// ---------------------------------------------------------------------------------------------------------------
namespace {

struct Tiff {
  std::span<const uint8_t> d;
  bool be = false;
  uint32_t u32(size_t p) const {
    return be ? (uint32_t(d[p]) << 24) | (uint32_t(d[p + 1]) << 16) | (uint32_t(d[p + 2]) << 8) | d[p + 3]
              : (uint32_t(d[p + 3]) << 24) | (uint32_t(d[p + 2]) << 16) | (uint32_t(d[p + 1]) << 8) | d[p];
  }
  uint16_t u16(size_t p) const { return be ? uint16_t((d[p] << 8) | d[p + 1]) : uint16_t((d[p + 1] << 8) | d[p]); }
  uint64_t u64(size_t p) const {
    uint64_t a = u32(p), b = u32(p + 4);
    return be ? (a << 32) | b : (b << 32) | a;
  }
};

struct ExifFailed {};  // getexif() raised: exif_transpose raises, normalize_image returns the image untouched

// The value Exif.get(Orientation) compares against the method table, as an exact integer when Python's
// dict lookup would match an int key, std::nullopt otherwise (bytes, str, non-integral numbers, NaN).
struct Orient {
  bool present = false;           // tag in IFD0 (any type): suppresses the XMP fallback
  std::optional<int64_t> value;   // integral value
};

Orient parse_exif(std::vector<uint8_t> data) {
  // Exif.load: re.sub(b"^(Exif\x00\x00){1,2}", b"", data)
  size_t off = 0;
  for (int k = 0; k < 2 && data.size() - off >= 6 && std::memcmp(data.data() + off, "Exif\0\0", 6) == 0; k++) off += 6;
  Orient o;
  if (off >= data.size()) return o;  // self._info = None
  std::span<const uint8_t> d(data.data() + off, data.size() - off);
  // head = fp.read(8); ImageFileDirectory_v2(head)
  std::span<const uint8_t> head = d.first(std::min<size_t>(8, d.size()));
  static const char* kPrefixes[] = {"MM\x00\x2a", "II\x2a\x00", "MM\x2a\x00", "II\x00\x2a", "MM\x00\x2b", "II\x2b\x00"};
  bool ok = false;
  if (head.size() >= 4)
    for (auto p : kPrefixes)
      if (std::memcmp(head.data(), p, 4) == 0) ok = true;
  if (!ok) throw ExifFailed{};  // SyntaxError
  Tiff t{d, head[0] == 'M'};
  const bool bigtiff = head[2] == 43;
  if (bigtiff) throw ExifFailed{};  // unpack("Q", ifh[8:]) on an 8-byte head: struct.error
  if (head.size() < 8) throw ExifFailed{};
  const uint64_t next = t.u32(4);
  // ImageFileDirectory_v2.load(fp) at offset `next`; OSError (short reads) ends it, keeping the tags so far
  struct Tag {
    int typ;
    std::vector<uint8_t> data;
  };
  std::optional<Tag> orient;
  auto rd = [&](uint64_t p, uint64_t n) -> std::optional<std::span<const uint8_t>> {
    if (p > d.size() || d.size() - p < n) return std::nullopt;
    return d.subspan(static_cast<size_t>(p), static_cast<size_t>(n));
  };
  static const int kUnit[19] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 0, 0, 8, 0, 0};
  // Pillow 13 ImageFileDirectory_v2.load: the first stored occurrence of a tag wins; out-of-line values count
  // towards a running total that may not exceed the stream size (else the tag is skipped with a warning); a
  // value that cannot be read in full ends the load (OSError), keeping the tags read so far.
  const uint64_t limit = d.size();
  uint64_t total_size = 0;
  std::vector<bool> stored(65536, false);
  do {
    uint64_t p = next;
    auto cnt = rd(p, 2);
    if (!cnt) break;
    const int count_tags = t.u16(static_cast<size_t>(p));
    p += 2;
    bool stop = false;
    for (int i = 0; i < count_tags && !stop; i++) {
      auto e = rd(p, 12);
      if (!e) break;
      const size_t ep = static_cast<size_t>(p);
      p += 12;
      const int tag = t.u16(ep), typ = t.u16(ep + 2);
      const uint64_t count = t.u32(ep + 4);
      if (stored[tag]) continue;  // `if tag in self._tagdata: continue`
      if (typ <= 0 || typ > 18 || kUnit[typ] == 0) continue;  // unsupported type
      const uint64_t size = count * static_cast<uint64_t>(kUnit[typ]);
      std::vector<uint8_t> val;
      if (size > 4) {
        const uint64_t vo = t.u32(ep + 8);
        total_size += size;
        if (total_size > limit) continue;  // "Tag data larger than file. Skipping tag"
        auto v = rd(vo, size);
        if (!v) {  // _safe_read: OSError("Truncated File Read") -> load() returns
          stop = true;
          break;
        }
        val.assign(v->begin(), v->end());
      } else {
        val.assign(d.begin() + ep + 8, d.begin() + ep + 8 + size);
      }
      if (val.empty()) continue;
      stored[tag] = true;
      if (tag == 0x0112) orient = Tag{typ, std::move(val)};
    }
  } while (false);
  if (!orient) return o;
  o.present = true;
  // first value, by type (TiffImagePlugin load dispatch + _setitem)
  const auto& v = orient->data;
  Tiff tv{std::span<const uint8_t>(v), t.be};
  switch (orient->typ) {
    case 3:
      o.value = tv.u16(0);
      break;
    case 8:
      o.value = static_cast<int16_t>(tv.u16(0));
      break;
    case 4:
    case 13:
      o.value = tv.u32(0);
      break;
    case 9:
      o.value = static_cast<int32_t>(tv.u32(0));
      break;
    case 6:
      o.value = static_cast<int8_t>(v[0]);
      break;
    case 16:
      if (v.size() >= 8) o.value = static_cast<int64_t>(tv.u64(0));
      break;
    case 5:
    case 10: {
      if (v.size() < 8) break;
      int64_t num = orient->typ == 5 ? int64_t(tv.u32(0)) : int64_t(int32_t(tv.u32(0)));
      int64_t den = orient->typ == 5 ? int64_t(tv.u32(4)) : int64_t(int32_t(tv.u32(4)));
      if (den != 0 && num % den == 0) o.value = num / den;  // Fraction == int
      break;
    }
    case 11: {
      uint32_t u = tv.u32(0);
      float f;
      std::memcpy(&f, &u, 4);
      if (std::isfinite(f) && f == std::floor(f) && std::fabs(f) < 1e9) o.value = static_cast<int64_t>(f);
      break;
    }
    case 12: {
      uint64_t u = tv.u64(0);
      double f;
      std::memcpy(&f, &u, 8);
      if (std::isfinite(f) && f == std::floor(f) && std::fabs(f) < 1e15) o.value = static_cast<int64_t>(f);
      break;
    }
    default:  // BYTE / ASCII / UNDEFINED: bytes or str, never equal to an int key
      break;
  }
  return o;
}

bool hexval(char c, int* v) {
  if (c >= '0' && c <= '9') return *v = c - '0', true;
  if (c >= 'a' && c <= 'f') return *v = c - 'a' + 10, true;
  if (c >= 'A' && c <= 'F') return *v = c - 'A' + 10, true;
  return false;
}

// bytes.fromhex("".join(s.split("\n")[3:]))
std::vector<uint8_t> raw_profile_bytes(const std::string& s) {
  size_t p = 0;
  for (int k = 0; k < 3; k++) {
    size_t q = s.find('\n', p);
    if (q == std::string::npos) return {};
    p = q + 1;
  }
  std::string h;
  for (size_t i = p; i < s.size(); i++)
    if (s[i] != '\n') h.push_back(s[i]);
  std::vector<uint8_t> out;
  size_t i = 0;
  auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
  while (i < h.size()) {
    if (ws(h[i])) {
      i++;
      continue;
    }
    int a, b;
    if (i + 1 >= h.size() || !hexval(h[i], &a) || !hexval(h[i + 1], &b)) throw ExifFailed{};  // ValueError
    out.push_back(static_cast<uint8_t>(a * 16 + b));
    i += 2;
  }
  return out;
}

std::optional<int> xmp_orientation(std::string_view x) {
  // re.search(r'tiff:Orientation(="|>)([0-9])', xmp)
  const std::string_view key = "tiff:Orientation";
  size_t p = 0;
  while ((p = x.find(key, p)) != std::string_view::npos) {
    size_t q = p + key.size();
    if (q + 1 < x.size() + 0 && q < x.size()) {
      if (x[q] == '=' && q + 2 < x.size() + 1 && q + 1 < x.size() && x[q + 1] == '"' && q + 2 < x.size() &&
          x[q + 2] >= '0' && x[q + 2] <= '9')
        return x[q + 2] - '0';
      if (x[q] == '>' && q + 1 < x.size() && x[q + 1] >= '0' && x[q + 1] <= '9') return x[q + 1] - '0';
    }
    p++;
  }
  return std::nullopt;
}

}  // namespace

int exif_transpose_method(const Native& n) {
  if (n.exif_unsupported_str || n.load_error) return 0;
  try {
    std::optional<std::vector<uint8_t>> exif_info = n.exif;
    if (!exif_info && n.raw_profile_exif) exif_info = raw_profile_bytes(*n.raw_profile_exif);
    Orient o;
    if (exif_info) o = parse_exif(*exif_info);
    std::optional<int64_t> orientation = o.value;
    if (!o.present) {
      std::optional<int> x;
      if (n.xmp_str && !n.xmp_str->empty()) {
        x = xmp_orientation(*n.xmp_str);
      } else if (n.xmp_bytes && !n.xmp_bytes->empty()) {
        x = xmp_orientation(std::string_view(reinterpret_cast<const char*>(n.xmp_bytes->data()), n.xmp_bytes->size()));
      }
      orientation = x ? std::optional<int64_t>(*x) : std::optional<int64_t>(1);
    }
    if (!orientation) return 0;
    switch (*orientation) {
      case 2:
        return 1;  // FLIP_LEFT_RIGHT
      case 3:
        return 4;  // ROTATE_180
      case 4:
        return 2;  // FLIP_TOP_BOTTOM
      case 5:
        return 6;  // TRANSPOSE
      case 6:
        return 5;  // ROTATE_270
      case 7:
        return 7;  // TRANSVERSE
      case 8:
        return 3;  // ROTATE_90
      default:
        return 0;
    }
  } catch (const ExifFailed&) {
    return 0;
  }
}

Rgb transpose(Rgb in, int method) {
  if (method == 0) return in;
  const int w = in.w, h = in.h;
  Rgb o;
  const bool swap = method == 3 || method == 5 || method == 6 || method == 7;
  o.w = swap ? h : w;
  o.h = swap ? w : h;
  o.px.resize(in.px.size());
  const uint8_t* s = in.px.data();
  uint8_t* d = o.px.data();
  for (int y = 0; y < o.h; y++) {
    for (int x = 0; x < o.w; x++) {
      int sx, sy;
      switch (method) {
        case 1: sx = w - 1 - x, sy = y; break;              // FLIP_LEFT_RIGHT
        case 2: sx = x, sy = h - 1 - y; break;              // FLIP_TOP_BOTTOM
        case 3: sx = w - 1 - y, sy = x; break;              // ROTATE_90 (counter-clockwise)
        case 4: sx = w - 1 - x, sy = h - 1 - y; break;      // ROTATE_180
        case 5: sx = y, sy = h - 1 - x; break;              // ROTATE_270
        case 6: sx = y, sy = x; break;                      // TRANSPOSE
        default: sx = w - 1 - y, sy = h - 1 - x; break;     // TRANSVERSE
      }
      const uint8_t* p = s + (static_cast<size_t>(sy) * w + sx) * 3;
      uint8_t* q = d + (static_cast<size_t>(y) * o.w + x) * 3;
      q[0] = p[0], q[1] = p[1], q[2] = p[2];
    }
  }
  return o;
}

}  // namespace fe::img
