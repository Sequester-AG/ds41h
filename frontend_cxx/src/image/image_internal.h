// Internal API of component IMAGE; used by src/image/*.cpp and tests/image.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "image/image.h"

namespace fe::img {

// Packed 8-bit RGB, row-major.
struct Rgb {
  int w = 0, h = 0;
  std::vector<uint8_t> px;
};

// An image as Pillow holds it after Image.open + load(), before vLLM's mode conversion.
struct Native {
  std::string format;  // "JPEG", "PNG", "GIF", "WEBP"
  std::string mode;    // "1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I;16"
  int w = 0, h = 0;
  // samples per mode: 1/L/P: 1 byte ("1" holds 0 or 255), LA: 2 (L, A), RGB: 3, RGBA/CMYK: 4,
  // I;16: 2 bytes little-endian
  std::vector<uint8_t> px;
  std::vector<uint8_t> palette;  // P: RGB triples as stored in the file (Pillow palette.size = entries)
  bool has_palette = false;
  // info["transparency"]
  enum Trns { kNone, kInt, kBytes, kTriple } trns = kNone;
  int trns_int = 0;
  std::vector<uint8_t> trns_bytes;
  int trns_rgb[3] = {0, 0, 0};
  // metadata that Image.getexif() reads
  std::optional<std::vector<uint8_t>> exif;            // info["exif"]
  std::optional<std::string> raw_profile_exif;         // info["Raw profile type exif"] (str)
  std::optional<std::string> xmp_str;                  // info["XML:com.adobe.xmp"] (str)
  std::optional<std::vector<uint8_t>> xmp_bytes;       // info["xmp"] (bytes)
  bool exif_unsupported_str = false;                   // info["exif"] would be a str (zTXt/iTXt "exif")
  // The first image.load() raised (a decoder error, which ImageFile.load raises after clearing self.tile, or an
  // exception in load_end). It raises inside ImageOps.exif_transpose, which vLLM's normalize_image suppresses, so
  // there is no transpose; vLLM's own image.load() then succeeds with the partially decoded pixels.
  bool load_error = false;
};

// Format sniffing in Pillow's plugin order for the formats this port decodes. Returns "" when none matches.
std::string sniff_format(std::span<const uint8_t> b);
bool sniff_other_pillow_format(std::span<const uint8_t> b);  // a format Pillow opens that this port does not

struct DecodeOpts {
  int64_t vllm_max_pixels = 178956970;  // VLLM_MAX_IMAGE_PIXELS (<= 0: off)
  std::string webp_lib_dir;
};
// Decoders. They throw ImageError with Python-equivalent kinds. Errors come in Python's order: header errors
// (Image.open), then check_size() on the header size (Pillow's DecompressionBombError in Image.open, then vLLM's
// VLLM_MAX_IMAGE_PIXELS check in load_bytes), then decoding errors (image.load()).
Native decode_jpeg(std::span<const uint8_t> b, const DecodeOpts& opts);
Native decode_png(std::span<const uint8_t> b, const DecodeOpts& opts);
Native decode_gif(std::span<const uint8_t> b, const DecodeOpts& opts);
Native decode_webp(std::span<const uint8_t> b, const DecodeOpts& opts);
bool webp_load(const std::string& lib_dir);
void check_bomb(int64_t w, int64_t h);                          // Image._decompression_bomb_check
void check_size(int64_t w, int64_t h, const DecodeOpts& opts);  // bomb check + VLLM_MAX_IMAGE_PIXELS

// vLLM ImageMediaIO._convert_image_mode(image) with image_mode="RGB", background (255,255,255).
// Sets *converted when vLLM records an io_config (mode != "RGB").
// May take over n.px (mode "RGB").
Rgb to_rgb(Native& n, bool* converted);

// ImageOps.exif_transpose: the Transpose method (0 = none, else Pillow's Image.Transpose value + 1:
// 1 FLIP_LEFT_RIGHT, 2 FLIP_TOP_BOTTOM, 3 ROTATE_90, 4 ROTATE_180, 5 ROTATE_270, 6 TRANSPOSE, 7 TRANSVERSE).
int exif_transpose_method(const Native& n);
Rgb transpose(Rgb in, int method);

// Pillow 13 Image.resize(size, BICUBIC) of an RGB image (libImaging/Resample.c, 8 bpc fixed point).
Rgb resize_bicubic(const Rgb& in, int w, int h);
// ImageOps.pad(image, size, BICUBIC, color=(fill,fill,fill), centering=(0.5,0.5)).
Rgb pad(const Rgb& in, int w, int h, uint8_t fill);

// mm_preprocess.load_image size math
struct Plan {
  int best_w = 0, best_h = 0, n_vit_h = 0, n_vit_w = 0, n_llm_h = 0, n_llm_w = 0;
  bool stretch = false;  // max_wh_ratio path: image.resize instead of ImageOps.pad
};
Plan plan_resize(int width, int height, const VisionConfig& vc);
// the uint8 image load_image feeds to numpy (pad or stretch)
Rgb transform(const Rgb& rgb, const Plan& p);

// hashes
std::string blake3_hex(std::span<const uint8_t> data);
class Blake3 {
 public:
  Blake3();
  ~Blake3();
  void update(const void* p, size_t n);
  std::string hexdigest();  // 32 bytes
 private:
  void push_cv(const uint32_t cv[8], uint64_t total_chunks);
  struct State;
  State* s_;
};
std::string blake2b128_hex(std::span<const std::span<const uint8_t>> parts);
std::array<uint8_t, 32> sha256(std::span<const uint8_t> data);
std::string sha256_hex(std::span<const uint8_t> data);
std::string mm_hash(std::span<const uint8_t> original_bytes, bool converted, const std::string& model_id);
std::string content_key(const ImageFeatures& f);

// pybase64.b64decode(s, validate=True)
std::vector<uint8_t> b64decode_validate(std::string_view s);

}  // namespace fe::img
