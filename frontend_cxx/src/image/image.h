// frontend_cxx component IMAGE: bit-exact port of the Python image path of vLLM 0.32.0 for
// DeepSeek-V4.1 (see DESIGN.md "Interfaces" and tools/inventory.md 1.5).
//
//   data: URL  -> decode_data_url()                       == MediaConnector._load_data_url + pybase64 validate=True
//   file bytes -> preprocess_image()                      == ImageMediaIO().load_bytes(...)          (Pillow 13)
//                                                           + DeepseekV4VLProcessor (mm_preprocess.load_image)
//                                                           + MultiModalHasher.hash_kwargs("blake3", ...)
//   prompt ids -> expand_image_placeholders()            == the PromptReplacement of mm_preprocess.py:353-376
//
// Formats: JPEG (libjpeg-turbo 3.2.0, the version Pillow 13 bundles), PNG, GIF (first frame), WebP (libwebp via
// dlopen, optional). Everything else Pillow would open (BMP, TIFF, ICO, ...) raises ImageError(kUnsupported).
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fe {

struct VisionConfig {
  // $M/config.json vision_config (DeepseekV41Config.vision_*)
  int patch_size = 14;
  int downsample_ratio = 3;
  int max_n_token = 1024;              // max_image_tokens
  int64_t min_pixels = 295936;
  std::optional<double> max_wh_ratio;  // null in the checkpoint
  // VLLM_MAX_IMAGE_PIXELS (vllm/envs.py default 178956970); <= 0 disables vLLM's own check. Pillow's
  // DecompressionBombError (> 2 * Image.MAX_IMAGE_PIXELS = 178956970 pixels) applies regardless.
  int64_t max_image_pixels = 178956970;
  // ModelConfig.model: the served model tag (args-json "model_tag"). Part of the mm_hash input.
  std::string model_id;
  int32_t image_token_id = 129264;     // <｜deepseek_image｜>
  // Optional directory holding libwebp.so.7 + libwebpdemux.so.2 (e.g. Pillow's pillow.libs is NOT used by
  // default); empty = the system libraries. WebP is unavailable (kUnsupported) when they cannot be loaded.
  std::string webp_lib_dir;
  // Fill ImageFeatures::content_key (the phys adapter key; the engine computes it itself, so the serving path does
  // not need it: ~6-14 ms of BLAKE2b over the patches).
  bool compute_content_key = false;
};

struct ImageFeatures {
  std::vector<uint16_t> patches_bf16;  // [n_patches, 3, patch, patch] bf16 bit patterns (torch layout)
  int64_t n_patches = 0;
  int64_t vit_grid[2] = {0, 0};        // [n_vit_h, n_vit_w]
  int64_t llm_grid[2] = {0, 0};        // [n_llm_h, n_llm_w]
  std::vector<int64_t> types;          // IMAGE_START=0, IMAGE=1, IMAGE_NEW_LINE=2, IMAGE_END=3
  std::string mm_hash;                 // vLLM MultiModalHasher (blake3 hex); see image_identifier() for uuids
  int64_t num_tokens = 0;              // == types.size(): the placeholder expands to this many 129264 ids
  // ---- additions
  std::string content_key;             // phys adapter mm.content_key (only with VisionConfig::compute_content_key)
  int32_t width = 0, height = 0;       // image size after EXIF transpose (what load_image sees)
  std::string format;                  // Pillow format name: "JPEG", "PNG", "GIF", "WEBP"
  std::string mode;                    // Pillow mode as opened ("RGB", "L", "P", "RGBA", "CMYK", "I;16", ...)
  bool converted = false;              // vLLM io_config was set (mode != "RGB"): it is part of the mm_hash
};

class ImageError : public std::runtime_error {
 public:
  enum Kind {
    kBadDataUrl,         // ValueError / NotImplementedError from _load_data_url
    kBase64,             // binascii.Error from pybase64.b64decode(validate=True)
    kUnidentified,       // PIL.UnidentifiedImageError -> ValueError("Failed to load image: cannot identify ...")
    kUnsupported,        // a format Pillow would open but this port does not decode (BMP, TIFF, ...)
    kDecode,             // OSError while decoding (broken data stream, ...) -> ValueError("Failed to load image")
    kTruncated,          // OSError("image file is truncated ...") -> ValueError("Failed to load image: ...")
    kDecompressionBomb,  // PIL.Image.DecompressionBombError (not wrapped by vLLM)
    kTooManyPixels,      // ValueError from the VLLM_MAX_IMAGE_PIXELS check
    kInternal,
  };
  ImageError(Kind k, const std::string& m) : std::runtime_error(m), kind(k) {}
  Kind kind;
};

// Full Python path for one image: decode + EXIF transpose + RGB conversion + DeepSeek-V4.1 transform + hashes.
ImageFeatures preprocess_image(std::span<const uint8_t> file_bytes, const VisionConfig&);

// data:[<mediatype>][;<param>=<value>]*;base64,<data>  -> the decoded bytes (vLLM _load_data_url semantics,
// case-insensitive "data:" scheme, base64 decoded with pybase64 validate=True). Throws kBadDataUrl / kBase64.
std::vector<uint8_t> decode_data_url(std::string_view url);
bool is_data_url(std::string_view url);

// The vLLM identifier of an image item. Stock vLLM 0.32 (the running venv, without the mm-content-identity
// patch) uses a client-supplied uuid verbatim when there are no hash factors; otherwise the content hash.
const std::string& image_identifier(const ImageFeatures&, const std::optional<std::string>& client_uuid);

// Replace the first images.size() occurrences of image_token_id in prompt_ids by num_tokens copies each
// (PromptReplacement target=[129264]). Returns the offsets of the expanded spans (PlaceholderRange.offset; the
// length is num_tokens and every position is an embed position). Throws kInternal if there are fewer
// placeholders than images.
std::vector<int64_t> expand_image_placeholders(std::vector<int32_t>& prompt_ids,
                                               const std::vector<ImageFeatures>& images, int32_t image_token_id);

// Library availability (WebP is loaded lazily).
bool webp_available(const VisionConfig&);

}  // namespace fe
