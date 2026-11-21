// Component IMAGE: the top of the Python image path.
//   ImageMediaIO.load_bytes (vllm/multimodal/media/image.py:79-102):
//     open_image (Pillow 13, ALLOWED_IMAGE_FORMATS) -> VLLM_MAX_IMAGE_PIXELS -> normalize_image (exif_transpose)
//     -> load() -> _convert_image_mode("RGB")
//   DeepseekV4VLProcessor (vllm/models/deepseek_v41/common/mm_preprocess.py:121-246)
//   MultiModalHasher.hash_kwargs("blake3", model_id=..., image=MediaWithBytes)
#include <cmath>
#include <cstring>
#include <mutex>

#include "image/image.h"
#include "image/image_internal.h"

namespace fe {
namespace img {

void check_bomb(int64_t w, int64_t h) {
  const int64_t kMax = 1024LL * 1024 * 1024 / 4 / 3;  // PIL.Image.MAX_IMAGE_PIXELS = 89478485
  const int64_t pixels = w * h;
  if (pixels > 2 * kMax)
    throw ImageError(ImageError::kDecompressionBomb,
                     "Image size (" + std::to_string(pixels) + " pixels) exceeds limit of " + std::to_string(2 * kMax) +
                         " pixels, could be decompression bomb DOS attack.");
}

void check_size(int64_t w, int64_t h, const DecodeOpts& opts) {
  check_bomb(w, h);
  if (opts.vllm_max_pixels > 0 && w * h > opts.vllm_max_pixels)
    throw ImageError(ImageError::kTooManyPixels,
                     "Image dimensions " + std::to_string(w) + "x" + std::to_string(h) + " (" + std::to_string(w * h) +
                         " pixels) exceed the maximum of " + std::to_string(opts.vllm_max_pixels) +
                         " pixels. Set VLLM_MAX_IMAGE_PIXELS to increase this limit.");
}

std::string sniff_format(std::span<const uint8_t> b) {
  // The plugins tried before these four in ALLOWED_IMAGE_FORMATS order cannot accept their signatures
  // (checked against Pillow 13's _accept functions; IM / IMT / TGA reject them in _open).
  if (b.size() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return "JPEG";
  static const uint8_t kPng[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  if (b.size() >= 8 && std::memcmp(b.data(), kPng, 8) == 0) return "PNG";
  if (b.size() >= 6 && (std::memcmp(b.data(), "GIF87a", 6) == 0 || std::memcmp(b.data(), "GIF89a", 6) == 0))
    return "GIF";
  if (b.size() >= 16 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0 &&
      (std::memcmp(b.data() + 12, "VP8 ", 4) == 0 || std::memcmp(b.data() + 12, "VP8X", 4) == 0 ||
       std::memcmp(b.data() + 12, "VP8L", 4) == 0))
    return "WEBP";
  return "";
}

bool sniff_other_pillow_format(std::span<const uint8_t> b) {
  // Best-effort magic numbers of the other ALLOWED_IMAGE_FORMATS plugins (only the error kind depends on it).
  auto has = [&](size_t off, const char* m, size_t n) { return b.size() >= off + n && std::memcmp(b.data() + off, m, n) == 0; };
  if (has(0, "BM", 2) || has(0, "BLP1", 4) || has(0, "BLP2", 4) || has(0, "DDS ", 4) || has(0, "FTEX", 4) ||
      has(0, "icns", 4) || has(0, "8BPS", 4) || has(0, "qoif", 4) || has(0, "SIMPLE", 6) || has(0, "DanM", 4) ||
      has(0, "LinS", 4) || has(0, "/* XPM */", 9) || has(0, "P7 332", 6) || has(0, "\x80\xe8\x00\x00", 4) ||
      has(0, "\xff\x4f\xff\x51", 4) || has(0, "\x00\x00\x00\x0cjP  \r\n\x87\n", 12) || has(0, "\0\0\1\0", 4) ||
      has(0, "\0\0\2\0", 4) || has(0, "MM\x00\x2a", 4) || has(0, "II\x2a\x00", 4) || has(0, "MM\x00\x2b", 4) ||
      has(0, "II\x2b\x00", 4) || has(0, "\xb1\x68\xde\x3a", 4) || has(0, "\x59\xa6\x6a\x95", 4) ||
      has(0, "\x01\xda", 2) || has(4, "ftyp", 4))
    return true;
  if (b.size() >= 2 && b[0] == 'P' && std::strchr("0123456fy", b[1]) && b[1]) return true;  // PPM / PFM
  if (b.size() >= 2 && b[0] == 10 && (b[1] == 0 || b[1] == 2 || b[1] == 3 || b[1] == 5)) return true;  // PCX
  if (b.size() >= 4) {  // DIB header sizes
    uint32_t v = b[0] | (b[1] << 8) | (b[2] << 16) | (uint32_t(b[3]) << 24);
    if (v == 12 || v == 40 || v == 52 || v == 56 || v == 64 || v == 108 || v == 124) return true;
  }
  if (b.size() >= 18) {  // TGA: TgaImageFile._open header validation
    int cmt = b[1], it = b[2], depth = b[16];
    int w = b[12] | (b[13] << 8), h = b[14] | (b[15] << 8);
    if ((cmt == 0 || cmt == 1) && w > 0 && h > 0 && (depth == 1 || depth == 8 || depth == 16 || depth == 24 || depth == 32) &&
        (it == 1 || it == 2 || it == 3 || it == 9 || it == 10 || it == 11))
      return true;
  }
  {  // XBM
    size_t i = 0;
    while (i < b.size() && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r' || b[i] == '\v' || b[i] == '\f')) i++;
    if (has(i, "#define", 7)) return true;
  }
  return false;
}

// pybase64.b64decode(s, validate=True): accepts what binascii.a2b_base64(s, strict_mode=True) accepts; str input must be ASCII
std::vector<uint8_t> b64decode_validate(std::string_view s) {
  // pybase64's C decoder (the venv's) reports every validate=True failure as "Non-base64 digit found",
  // whatever binascii would say; the structure below stays binascii's.
  auto err = [](const char*) -> std::vector<uint8_t> {
    throw ImageError(ImageError::kBase64, "Non-base64 digit found");
  };
  for (unsigned char c : s)
    if (c >= 0x80) return err("string argument should contain only ASCII characters");
  static int8_t T[256];
  static std::once_flag once;
  std::call_once(once, [] {
    std::memset(T, -1, sizeof T);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) T[static_cast<uint8_t>(a[i])] = static_cast<int8_t>(i);
  });
  std::vector<uint8_t> out;
  out.reserve(s.size() / 4 * 3 + 3);
  if (!s.empty() && s[0] == '=') return err("Leading padding not allowed");
  int quad = 0, pads = 0;
  bool padding_started = false;
  unsigned left = 0;
  for (size_t i = 0; i < s.size(); i++) {
    unsigned char ch = static_cast<unsigned char>(s[i]);
    if (ch == '=') {
      padding_started = true;
      if (quad == 0) return err("Excess padding not allowed");
      if (quad >= 2 && quad + ++pads >= 4) {
        if (i + 1 < s.size()) return err("Excess data after padding");
        return out;
      }
      continue;
    }
    int v = T[ch];
    if (v < 0) return err("Only base64 data is allowed");
    if (padding_started) return err("Discontinuous padding not allowed");
    pads = 0;
    switch (quad) {
      case 0:
        quad = 1, left = static_cast<unsigned>(v);
        break;
      case 1:
        quad = 2, out.push_back(static_cast<uint8_t>((left << 2) | (v >> 4))), left = v & 0x0F;
        break;
      case 2:
        quad = 3, out.push_back(static_cast<uint8_t>((left << 4) | (v >> 2))), left = v & 0x03;
        break;
      default:
        quad = 0, out.push_back(static_cast<uint8_t>((left << 6) | v)), left = 0;
        break;
    }
  }
  if (quad == 1) return err("Invalid base64-encoded string: number of data characters cannot be 1 more than a multiple of 4");
  if (quad != 0) return err("Incorrect padding");
  return out;
}

namespace {

const uint16_t* bf16_lut() {
  // torch: float32(v) / 255 ; (x - 0.5) / 0.5 ; .to(torch.bfloat16) (round to nearest even)
  static uint16_t lut[256];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int v = 0; v < 256; v++) {
      volatile float x = static_cast<float>(v) / 255.0f;
      volatile float y = (x - 0.5f) / 0.5f;
      float yy = y;
      uint32_t u;
      std::memcpy(&u, &yy, 4);
      lut[v] = static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
  });
  return lut;
}

}  // namespace
}  // namespace img

bool is_data_url(std::string_view url) {
  if (url.size() < 5) return false;
  const char* d = "data:";
  for (int i = 0; i < 5; i++) {
    char c = url[i];
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (c != d[i]) return false;
  }
  return true;
}

std::vector<uint8_t> decode_data_url(std::string_view url) {
  if (!is_data_url(url)) throw ImageError(ImageError::kBadDataUrl, "The URL must be either a HTTP, data or file URL.");
  std::string_view rest = url.substr(5);
  const size_t comma = rest.find(',');
  if (comma == std::string_view::npos)
    throw ImageError(ImageError::kBadDataUrl, "Invalid data URL: missing ',' separator.");
  std::string_view spec = rest.substr(0, comma), data = rest.substr(comma + 1);
  const size_t semi = spec.rfind(';');
  if (semi == std::string_view::npos || spec.substr(semi + 1) != "base64")
    throw ImageError(ImageError::kBadDataUrl, "Only base64 data URLs are supported for now.");
  return img::b64decode_validate(data);
}

const std::string& image_identifier(const ImageFeatures& f, const std::optional<std::string>& client_uuid) {
  return client_uuid ? *client_uuid : f.mm_hash;
}

std::vector<int64_t> expand_image_placeholders(std::vector<int32_t>& ids, const std::vector<ImageFeatures>& images,
                                               int32_t image_token_id) {
  std::vector<int64_t> offsets;
  std::vector<int32_t> out;
  out.reserve(ids.size());
  size_t k = 0;
  for (int32_t t : ids) {
    if (t == image_token_id && k < images.size()) {
      offsets.push_back(static_cast<int64_t>(out.size()));
      out.insert(out.end(), static_cast<size_t>(images[k].num_tokens), image_token_id);
      k++;
    } else {
      out.push_back(t);
    }
  }
  if (k < images.size())
    throw ImageError(ImageError::kInternal, "Expected there to be " + std::to_string(images.size()) +
                                                " prompt placeholders for the image items, found " + std::to_string(k));
  ids = std::move(out);
  return offsets;
}

bool webp_available(const VisionConfig& vc) { return img::webp_load(vc.webp_lib_dir); }

ImageFeatures preprocess_image(std::span<const uint8_t> bytes, const VisionConfig& vc) {
  using namespace img;
  DecodeOpts opts;
  opts.vllm_max_pixels = vc.max_image_pixels;
  opts.webp_lib_dir = vc.webp_lib_dir;
  const std::string fmt = sniff_format(bytes);
  Native n;
  if (fmt == "JPEG") {
    n = decode_jpeg(bytes, opts);
  } else if (fmt == "PNG") {
    n = decode_png(bytes, opts);
  } else if (fmt == "GIF") {
    n = decode_gif(bytes, opts);
  } else if (fmt == "WEBP") {
    n = decode_webp(bytes, opts);
  } else if (sniff_other_pillow_format(bytes)) {
    throw ImageError(ImageError::kUnsupported, "Failed to load image: image format not supported by frontend_cxx");
  } else {
    throw ImageError(ImageError::kUnidentified, "Failed to load image: cannot identify image file");
  }
  const int method = exif_transpose_method(n);
  ImageFeatures f;
  Rgb rgb = to_rgb(n, &f.converted);
  f.format = n.format;
  f.mode = n.mode;
  n.px.clear();
  n.px.shrink_to_fit();
  rgb = transpose(std::move(rgb), method);
  f.width = rgb.w;
  f.height = rgb.h;
  const Plan pl = plan_resize(rgb.w, rgb.h, vc);
  const Rgb t = transform(rgb, pl);
  const int p = vc.patch_size;
  f.vit_grid[0] = pl.n_vit_h;
  f.vit_grid[1] = pl.n_vit_w;
  f.llm_grid[0] = pl.n_llm_h;
  f.llm_grid[1] = pl.n_llm_w;
  f.n_patches = static_cast<int64_t>(pl.n_vit_h) * pl.n_vit_w;
  // patches: x (3, H, W) -> reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4) -> (nh * nw, 3, p, p)
  const uint16_t* lut = bf16_lut();
  const size_t pp = static_cast<size_t>(p) * p;
  f.patches_bf16.resize(static_cast<size_t>(f.n_patches) * 3 * pp);
  uint16_t* out = f.patches_bf16.data();
  const int W = t.w;
  for (int y = 0; y < t.h; y++) {
    const int i = y / p, py = y % p;
    const uint8_t* row = t.px.data() + static_cast<size_t>(y) * W * 3;
    for (int j = 0; j < pl.n_vit_w; j++) {
      uint16_t* base = out + (static_cast<size_t>(i) * pl.n_vit_w + j) * 3 * pp + static_cast<size_t>(py) * p;
      const uint8_t* s = row + static_cast<size_t>(j) * p * 3;
      for (int px = 0; px < p; px++) {
        base[px] = lut[s[3 * px]];
        base[pp + px] = lut[s[3 * px + 1]];
        base[2 * pp + px] = lut[s[3 * px + 2]];
      }
    }
  }
  // types: [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
  f.types.reserve(static_cast<size_t>(pl.n_llm_h) * (pl.n_llm_w + 1) + 2);
  f.types.push_back(0);
  for (int r = 0; r < pl.n_llm_h; r++) {
    f.types.insert(f.types.end(), static_cast<size_t>(pl.n_llm_w), 1);
    f.types.push_back(2);
  }
  f.types.push_back(3);
  f.num_tokens = static_cast<int64_t>(f.types.size());
  f.mm_hash = mm_hash(bytes, f.converted, vc.model_id);
  if (vc.compute_content_key) f.content_key = content_key(f);
  return f;
}

}  // namespace fe
