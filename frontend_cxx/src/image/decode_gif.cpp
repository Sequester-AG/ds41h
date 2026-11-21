// GIF, first frame: a port of Pillow 13 GifImagePlugin (_open, _seek(0), load_prepare) and libImaging/GifDecode.c.
// Frame 0 is "P" (global or local palette) or "L" (no palette, or an identity grey palette, _is_palette_needed),
// drawn into a canvas of the logical screen size (enlarged when the frame extends past it) that starts as zeros,
// or as the transparency index when the frame has one; info["transparency"] = that index.
#include <algorithm>
#include <cstring>
#include <optional>
#include <vector>

#include "image/image_internal.h"

namespace fe::img {
namespace {

[[noreturn]] void unidentified() {
  throw ImageError(ImageError::kUnidentified, "Failed to load image: cannot identify image file");
}

struct Rd {
  std::span<const uint8_t> b;
  size_t pos = 0;
  std::span<const uint8_t> read(size_t n) {
    size_t k = std::min(n, b.size() - pos);
    auto r = b.subspan(pos, k);
    pos += k;
    return r;
  }
  // GifImageFile.data(): one sub-block, or None for a zero / missing length byte
  bool data(std::span<const uint8_t>* out) {
    auto s = read(1);
    if (!s.empty() && s[0]) {
      *out = read(s[0]);
      return true;
    }
    return false;
  }
};

bool palette_needed(std::span<const uint8_t> p) {
  // for i in range(0, len(p), 3): if not (i // 3 == p[i] == p[i + 1] == p[i + 2]): return True
  // (chained comparison: short-circuits, so a short table only raises IndexError when the prefix matched)
  for (size_t i = 0; i < p.size(); i += 3) {
    if (int(i / 3) != p[i]) return true;
    if (i + 1 >= p.size()) unidentified();  // IndexError
    if (p[i] != p[i + 1]) return true;
    if (i + 2 >= p.size()) unidentified();
    if (p[i + 1] != p[i + 2]) return true;
  }
  return false;
}

// libImaging/GifDecode.c, fed with everything from the tile offset to EOF (Pillow feeds 64 KiB reads; the
// decoder keeps its state across calls and only starts a sub-block it has completely, so the result is the
// same). Returns: 1 = frame complete (IMAGING_CODEC_END or the non-interlaced "return -1"), 0 = data exhausted
// (ImageFile.load then raises "image file is truncated"), throws on broken data.
constexpr int kGifBits = 12;
constexpr int kGifTable = 1 << kGifBits;

int gif_decode(std::vector<uint8_t>& im, int im_w, int xoff, int yoff, int xsize, int ysize, int bits,
               bool interlace_flag, std::span<const uint8_t> data) {
  if (bits < 0 || bits > 12) throw ImageError(ImageError::kDecode, "Failed to load image: bits out of range");
  const int clear = 1 << bits, end = clear + 1;
  int interlace = interlace_flag ? 1 : 0;
  int step = interlace ? 8 : 1;
  [[maybe_unused]] int repeat = interlace ? 8 : 0;
  int x = 0, y = 0;
  int state = 1;
  int next = 0, codesize = 0, codemask = 0;
  int bufferindex = kGifTable;
  int lastdata = 0, lastcode = 0;
  int blocksize = 0;
  uint32_t bitbuffer = 0;
  int bitcount = 0;
  std::vector<uint8_t> buffer(kGifTable, 0), tdata(kGifTable, 0);
  std::vector<int> link(kGifTable, 0);
  uint8_t one = 0;  // the first byte of a decoded string (GifDecode's &context->lastdata)
  size_t ptr = 0;
  size_t bytes = data.size();
  size_t subblock_end = 0;  // end offset of the sub-block being read
  auto broken = []() -> int { throw ImageError(ImageError::kDecode, "Failed to load image: broken data stream"); };
  auto row = [&](int yy) { return im.data() + static_cast<size_t>(yy + yoff) * im_w + xoff; };
  uint8_t* out = row(0);

  // NEWLINE macro: returns 1 when the frame ends through the "default: return -1" branch
  auto newline = [&]() -> bool {
    x = 0;
    y += step;
    while (y >= ysize) {
      switch (interlace) {
        case 1:
          repeat = y = 4;
          interlace = 2;
          break;
        case 2:
          step = 4;
          repeat = y = 2;
          interlace = 3;
          break;
        case 3:
          step = 2;
          repeat = y = 1;
          interlace = 0;
          break;
        default:
          return true;
      }
    }
    if (y < ysize) out = row(y);
    return false;
  };

  for (;;) {
    if (state == 1) {
      next = clear + 2;
      codesize = bits + 1;
      codemask = (1 << codesize) - 1;
      bufferindex = kGifTable;
      state = 2;
    }
    const uint8_t* p;
    int i;
    if (bufferindex < kGifTable) {
      i = kGifTable - bufferindex;
      p = &buffer[bufferindex];
      bufferindex = kGifTable;
    } else {
      while (bitcount < codesize) {
        if (blocksize > 0) {
          int c = data[ptr++];
          bytes--;
          blocksize--;
          bitbuffer |= static_cast<uint32_t>(c) << bitcount;
          bitcount += 8;
        } else {
          if (bytes < 1) return 0;
          int c = data[ptr];
          if (bytes < static_cast<size_t>(c) + 1) return 0;
          blocksize = c;
          ptr++;
          bytes--;
          subblock_end = ptr + static_cast<size_t>(c);
        }
      }
      int c = static_cast<int>(bitbuffer & static_cast<uint32_t>(codemask));
      bitbuffer >>= codesize;
      bitcount -= codesize;
      if (c == clear) {
        if (state != 2) state = 1;
        continue;
      }
      if (c == end) {
        // GifDecode breaks out and returns the bytes consumed. ImageFile.load then reads the next 64 KiB block:
        // at EOF that raises "image file is truncated"; otherwise the next decode() call resumes reading codes
        // after the end code. The call that saw the end code had data up to the first 64 KiB boundary at or
        // after the end of the current sub-block.
        const size_t kBlock = 65536;
        const size_t call_end = std::min(data.size(), (subblock_end + kBlock - 1) / kBlock * kBlock);
        if (call_end >= data.size()) return 0;
        continue;
      }
      i = 1;
      if (state == 2) {
        if (c > clear) return broken();
        lastdata = lastcode = c;
        state = 3;
      } else {
        int thiscode = c;
        if (c > next) return broken();
        if (c == next) {
          if (bufferindex <= 0) return broken();
          buffer[--bufferindex] = static_cast<uint8_t>(lastdata);
          c = lastcode;
        }
        while (c >= clear) {
          if (bufferindex <= 0 || c >= kGifTable) return broken();
          buffer[--bufferindex] = tdata[c];
          c = link[c];
        }
        lastdata = c;
        if (next < kGifTable) {
          tdata[next] = static_cast<uint8_t>(c);
          link[next] = lastcode;
          if (next == codemask && codesize < kGifBits) {
            codesize++;
            codemask = (1 << codesize) - 1;
          }
          next++;
        }
        lastcode = thiscode;
      }
      one = static_cast<uint8_t>(lastdata);
      p = &one;
    }
    // copy the bytes into the image
    if (y >= ysize) throw ImageError(ImageError::kDecode, "Failed to load image: buffer overrun when reading image file");
    // (frame 0 has transparency == -1: the shortcut paths and the slow path write the same bytes)
    for (int k = 0; k < i; k++) {
      *out++ = p[k];
      if (++x >= xsize) {
        if (newline()) return 1;
        if (y >= ysize) return 1;  // IMAGING_CODEC_END
      }
    }
  }
}

}  // namespace

Native decode_gif(std::span<const uint8_t> b, const DecodeOpts& opts) {
  Native n;
  n.format = "GIF";
  if (b.size() < 6 || (std::memcmp(b.data(), "GIF87a", 6) != 0 && std::memcmp(b.data(), "GIF89a", 6) != 0))
    unidentified();
  Rd r{b, 0};
  auto s = r.read(13);
  if (s.size() < 13) unidentified();  // i16(s, 6) ... struct.error
  int sw = s[6] | (s[7] << 8), sh = s[8] | (s[9] << 8);
  int flags = s[10];
  int bits = (flags & 7) + 1;
  std::vector<uint8_t> global;
  bool has_global = false;
  if (flags & 128) {
    auto p = r.read(static_cast<size_t>(3) << bits);
    if (palette_needed(p)) {
      global.assign(p.begin(), p.end());
      has_global = true;
    }
  }
  // _seek(0)
  std::optional<int> frame_transparency;
  int interlace = -1;  // None
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  int frame_bits = 0;
  std::optional<std::vector<uint8_t>> local;  // palette (False -> empty optional with local_false)
  bool local_false = false;
  std::vector<uint8_t> sb;
  auto c = r.read(1);
  if (c.empty() || c[0] == ';') {
    // EOFError("no more images in GIF file") propagates out of Image.open (not caught there)
    throw ImageError(ImageError::kDecode, "Failed to load image: no more images in GIF file");
  }
  std::span<const uint8_t> cur = c;
  while (true) {
    if (cur.empty()) cur = r.read(1);
    if (cur.empty() || cur[0] == ';') break;
    if (cur[0] == '!') {
      auto lab = r.read(1);
      if (lab.empty()) unidentified();  // s[0] IndexError
      std::span<const uint8_t> block;
      bool has_block = r.data(&block);
      if (lab[0] == 249 && has_block) {
        if (block.size() < 3) unidentified();  // i16(block, 1): struct.error
        int f = block[0];
        if (f & 1) {
          if (block.size() < 4) unidentified();  // block[3]: IndexError
          frame_transparency = block[3];
        }
      } else if (lab[0] == 254) {
        while (has_block && !block.empty()) has_block = r.data(&block);
        cur = {};
        continue;
      } else if (lab[0] == 255 && has_block) {
        if (block.size() >= 11 && std::memcmp(block.data(), "NETSCAPE2.0", 11) == 0) {
          std::span<const uint8_t> b2;
          r.data(&b2);
          // i16(block, 1) only when len >= 3 and block[0] == 1: no error path
        }
      }
      std::span<const uint8_t> skip;
      while (r.data(&skip)) {
      }
    } else if (cur[0] == ',') {
      auto d = r.read(9);
      if (d.size() < 9) unidentified();  // struct.error / IndexError
      x0 = d[0] | (d[1] << 8);
      y0 = d[2] | (d[3] << 8);
      x1 = x0 + (d[4] | (d[5] << 8));
      y1 = y0 + (d[6] | (d[7] << 8));
      if (x1 > sw || y1 > sh) {
        int nw = std::max(x1, sw), nh = std::max(y1, sh);
        check_bomb(nw, nh);
        sw = nw;
        sh = nh;
      }
      int lf = d[8];
      interlace = (lf & 64) != 0;
      if (lf & 128) {
        int lb = (lf & 7) + 1;
        auto p = r.read(static_cast<size_t>(3) << lb);
        if (palette_needed(p)) {
          local = std::vector<uint8_t>(p.begin(), p.end());
        } else {
          local_false = true;
        }
      }
      auto bb = r.read(1);
      if (bb.empty()) unidentified();  // IndexError
      frame_bits = bb[0];
      break;
    }
    cur = {};
  }
  if (interlace < 0) {
    // EOFError("image not found in GIF frame"): not caught by Image.open
    throw ImageError(ImageError::kDecode, "Failed to load image: image not found in GIF frame");
  }
  const size_t tile_offset = r.pos;
  // frame palette: local (if any; False for an identity grey local table) else the global one
  const std::vector<uint8_t>* frame_pal = nullptr;
  if (local) {
    frame_pal = &*local;
  } else if (!local_false && has_global) {
    frame_pal = &global;
  }
  n.mode = frame_pal ? "P" : "L";
  if (frame_pal) {
    n.has_palette = true;
    n.palette = *frame_pal;
  }
  if (frame_transparency) {
    n.trns = Native::kInt;
    n.trns_int = *frame_transparency;
  }
  if (sw <= 0 || sh <= 0) unidentified();  // ImageFile: "not identified by this driver"
  check_size(sw, sh, opts);
  n.w = sw;
  n.h = sh;
  n.px.assign(static_cast<size_t>(sw) * sh, frame_transparency ? static_cast<uint8_t>(*frame_transparency) : 0);
  const int xs = x1 - x0, ys = y1 - y0;
  if (xs <= 0 || ys <= 0) {
    // decoder.setimage: ValueError("tile cannot extend outside image")
    throw ImageError(ImageError::kDecode, "Failed to load image: tile cannot extend outside image");
  }
  // Decoder errors (codec configuration, broken data, overrun) are soft: ImageFile.load raises them after clearing
  // the tile, so vLLM keeps the canvas as decoded so far. Running out of data is hard ("image file is truncated").
  int rc = 1;
  try {
    rc = gif_decode(n.px, sw, x0, y0, xs, ys, frame_bits, interlace == 1, b.subspan(tile_offset));
  } catch (const ImageError& e) {
    if (e.kind != ImageError::kDecode) throw;
    n.load_error = true;
  }
  if (rc == 0) throw ImageError(ImageError::kTruncated, "Failed to load image: image file is truncated");
  return n;
}

}  // namespace fe::img
