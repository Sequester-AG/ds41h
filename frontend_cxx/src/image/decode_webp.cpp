// WebP: Pillow 13 WebPImagePlugin + _webp.c. Pillow decodes every WebP (still or animated) through
// WebPAnimDecoderNew(data, NULL) (MODE_RGBA, no threads) and reads frame 0 of the canvas; the mode is "RGBA"
// unless WebPGetFeatures says the bitstream has no alpha ("RGBX" raw -> mode "RGB"). info["exif"] / ["xmp"] are
// the first EXIF / "XMP " chunks of the demuxer when non-empty.
#include <dirent.h>
#include <algorithm>
#include <optional>
#include <dlfcn.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "image/image_internal.h"

namespace fe::img {
namespace {

struct WebPData {
  const uint8_t* bytes;
  size_t size;
};
struct WebPBitstreamFeatures {
  int width, height, has_alpha, has_animation, format;
  uint32_t pad[5];
};
struct WebPAnimInfo {
  uint32_t canvas_width, canvas_height, loop_count, bgcolor, frame_count;
  uint32_t pad[4];
};
struct WebPChunkIterator {
  int chunk_num, num_chunks;
  WebPData chunk;
  uint32_t pad[6];
  void* private_;
};
constexpr int kDecoderAbi = 0x0209;  // WEBP_DECODER_ABI_VERSION (major 0x02 since libwebp 0.5)
constexpr int kDemuxAbi = 0x0107;    // WEBP_DEMUX_ABI_VERSION (major 0x01)
constexpr int kVp8StatusOk = 0;

struct Api {
  bool ok = false;
  std::string error;
  int (*GetFeaturesInternal)(const uint8_t*, size_t, WebPBitstreamFeatures*, int) = nullptr;
  void* (*AnimDecoderNewInternal)(const WebPData*, const void*, int) = nullptr;
  int (*AnimDecoderGetInfo)(const void*, WebPAnimInfo*) = nullptr;
  int (*AnimDecoderGetNext)(void*, uint8_t**, int*) = nullptr;
  const void* (*AnimDecoderGetDemuxer)(const void*) = nullptr;
  void (*AnimDecoderDelete)(void*) = nullptr;
  int (*DemuxGetChunk)(const void*, const char*, int, WebPChunkIterator*) = nullptr;
  void (*DemuxReleaseChunkIterator)(WebPChunkIterator*) = nullptr;
};

std::mutex g_mu;
std::map<std::string, Api> g_apis;

std::vector<std::string> libs_in(const std::string& dir) {
  // load order: sharpyuv (dependency of libwebp >= 1.3), webp, webpdemux. Matches plain and auditwheel names.
  std::vector<std::string> names[3];
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string f = e->d_name;
      if (f.find(".so") == std::string::npos) continue;
      if (f.rfind("libsharpyuv", 0) == 0) names[0].push_back(dir + "/" + f);
      else if (f.rfind("libwebpdemux", 0) == 0) names[2].push_back(dir + "/" + f);
      else if (f.rfind("libwebp.", 0) == 0 || f.rfind("libwebp-", 0) == 0) names[1].push_back(dir + "/" + f);
    }
    closedir(d);
  }
  std::vector<std::string> out;
  for (auto& v : names) {
    std::sort(v.begin(), v.end());
    if (!v.empty()) out.push_back(v.back());
  }
  return out;
}

const Api& api(const std::string& dir) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_apis.find(dir);
  if (it != g_apis.end()) return it->second;
  Api a;
  void* hw = nullptr;
  void* hd = nullptr;
  if (dir.empty()) {
    hw = dlopen("libwebp.so.7", RTLD_NOW | RTLD_GLOBAL);
    hd = dlopen("libwebpdemux.so.2", RTLD_NOW | RTLD_GLOBAL);
  } else {
    for (const auto& p : libs_in(dir)) {
      void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!h) continue;
      if (p.find("libwebpdemux") != std::string::npos) hd = h;
      else if (p.find("libsharpyuv") == std::string::npos) hw = h;
    }
  }
  if (hw && hd) {
    a.GetFeaturesInternal = reinterpret_cast<decltype(a.GetFeaturesInternal)>(dlsym(hw, "WebPGetFeaturesInternal"));
    a.AnimDecoderNewInternal = reinterpret_cast<decltype(a.AnimDecoderNewInternal)>(dlsym(hd, "WebPAnimDecoderNewInternal"));
    a.AnimDecoderGetInfo = reinterpret_cast<decltype(a.AnimDecoderGetInfo)>(dlsym(hd, "WebPAnimDecoderGetInfo"));
    a.AnimDecoderGetNext = reinterpret_cast<decltype(a.AnimDecoderGetNext)>(dlsym(hd, "WebPAnimDecoderGetNext"));
    a.AnimDecoderGetDemuxer = reinterpret_cast<decltype(a.AnimDecoderGetDemuxer)>(dlsym(hd, "WebPAnimDecoderGetDemuxer"));
    a.AnimDecoderDelete = reinterpret_cast<decltype(a.AnimDecoderDelete)>(dlsym(hd, "WebPAnimDecoderDelete"));
    a.DemuxGetChunk = reinterpret_cast<decltype(a.DemuxGetChunk)>(dlsym(hd, "WebPDemuxGetChunk"));
    a.DemuxReleaseChunkIterator =
        reinterpret_cast<decltype(a.DemuxReleaseChunkIterator)>(dlsym(hd, "WebPDemuxReleaseChunkIterator"));
    a.ok = a.GetFeaturesInternal && a.AnimDecoderNewInternal && a.AnimDecoderGetInfo && a.AnimDecoderGetNext &&
           a.AnimDecoderGetDemuxer && a.AnimDecoderDelete && a.DemuxGetChunk && a.DemuxReleaseChunkIterator;
  }
  if (!a.ok) a.error = "libwebp / libwebpdemux not available";
  return g_apis.emplace(dir, a).first->second;
}

uint32_t le24(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16); }

}  // namespace

bool webp_load(const std::string& lib_dir) { return api(lib_dir).ok; }

Native decode_webp(std::span<const uint8_t> b, const DecodeOpts& opts) {
  const Api& a = api(opts.webp_lib_dir);
  if (!a.ok) throw ImageError(ImageError::kUnsupported, "Failed to load image: WebP support not available");
  Native n;
  n.format = "WEBP";
  // _get_size(s) on the first 30 bytes -> _decompression_bomb_check(size, error_only=True)
  if (b.size() >= 30) {
    const uint8_t* s = b.data();
    int64_t w = -1, h = -1;
    if (std::memcmp(s + 12, "VP8 ", 4) == 0 && s[23] == 0x9D && s[24] == 0x01 && s[25] == 0x2A) {
      w = (s[26] | (s[27] << 8)) & 0x3FFF;
      h = (s[28] | (s[29] << 8)) & 0x3FFF;
    } else if (std::memcmp(s + 12, "VP8X", 4) == 0) {
      w = int64_t(le24(s + 24)) + 1;
      h = int64_t(le24(s + 27)) + 1;
    } else if (std::memcmp(s + 12, "VP8L", 4) == 0 && b.size() >= 25 && s[20] == 0x2F) {
      uint32_t bits = s[21] | (s[22] << 8) | (s[23] << 16) | (uint32_t(s[24]) << 24);
      w = (bits & 0x3FFF) + 1;
      h = ((bits >> 14) & 0x3FFF) + 1;
    }
    if (w >= 0) check_bomb(w, h);
  }
  WebPBitstreamFeatures feat{};
  bool has_alpha = true;  // mode = "RGBA" unless WebPGetFeatures succeeds and reports no alpha
  if (a.GetFeaturesInternal(b.data(), b.size(), &feat, kDecoderAbi) == kVp8StatusOk) has_alpha = feat.has_alpha != 0;
  WebPData data{b.data(), b.size()};
  void* dec = a.AnimDecoderNewInternal(&data, nullptr, kDemuxAbi);
  WebPAnimInfo info{};
  if (!dec || !a.AnimDecoderGetInfo(dec, &info)) {
    if (dec) a.AnimDecoderDelete(dec);
    throw ImageError(ImageError::kDecode, "Failed to load image: could not create decoder object");
  }
  struct Guard {
    const Api& a;
    void* d;
    ~Guard() { a.AnimDecoderDelete(d); }
  } guard{a, dec};
  n.w = static_cast<int>(info.canvas_width);
  n.h = static_cast<int>(info.canvas_height);
  n.mode = has_alpha ? "RGBA" : "RGB";
  if (n.w <= 0 || n.h <= 0) throw ImageError(ImageError::kUnidentified, "Failed to load image: cannot identify image file");
  check_size(n.w, n.h, opts);
  const void* demux = a.AnimDecoderGetDemuxer(dec);
  auto chunk = [&](const char* fourcc) -> std::optional<std::vector<uint8_t>> {
    WebPChunkIterator it{};
    if (!a.DemuxGetChunk(demux, fourcc, 1, &it)) return std::nullopt;
    std::vector<uint8_t> v(it.chunk.bytes, it.chunk.bytes + it.chunk.size);
    a.DemuxReleaseChunkIterator(&it);
    if (v.empty()) return std::nullopt;
    return v;
  };
  n.exif = chunk("EXIF");
  n.xmp_bytes = chunk("XMP ");
  uint8_t* buf = nullptr;
  int ts = 0;
  if (!a.AnimDecoderGetNext(dec, &buf, &ts) || !buf)
    throw ImageError(ImageError::kDecode, "Failed to load image: failed to read next frame");
  const size_t np = static_cast<size_t>(n.w) * n.h;
  if (has_alpha) {
    n.px.assign(buf, buf + np * 4);
  } else {
    n.px.resize(np * 3);
    for (size_t i = 0; i < np; i++) {
      n.px[3 * i] = buf[4 * i];
      n.px[3 * i + 1] = buf[4 * i + 1];
      n.px[3 * i + 2] = buf[4 * i + 2];
    }
  }
  return n;
}

}  // namespace fe::img
