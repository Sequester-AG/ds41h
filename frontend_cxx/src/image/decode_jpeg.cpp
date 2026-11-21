// JPEG: Pillow 13 JpegImagePlugin._open (the Python marker walk that decides mode, size and info) + the libjpeg
// decode of libImaging/JpegDecode.c (out_color_space from the rawmode, libjpeg defaults otherwise: JDCT_ISLOW,
// fancy upsampling) with libjpeg-turbo 3.2.0, the version Pillow 13 bundles.
#include <csetjmp>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <string>
#include <vector>

extern "C" {
#include <jpeglib.h>
#include <jerror.h>
}

#include "image/image_internal.h"

namespace fe::img {
namespace {

[[noreturn]] void unidentified() {
  throw ImageError(ImageError::kUnidentified, "Failed to load image: cannot identify image file");
}

struct Reader {
  std::span<const uint8_t> b;
  size_t pos = 0;
  // fp.read(n): up to n bytes
  std::span<const uint8_t> read(size_t n) {
    size_t k = std::min(n, b.size() - pos);
    auto r = b.subspan(pos, k);
    pos += k;
    return r;
  }
  // i16(fp.read(2)) -> struct.error when short -> Image.open fails -> UnidentifiedImageError
  int read_i16() {
    auto s = read(2);
    if (s.size() < 2) unidentified();
    return (s[0] << 8) | s[1];
  }
  // ImageFile._safe_read: OSError("Truncated File Read") when short (propagates out of Image.open)
  std::span<const uint8_t> safe_read(long n) {
    if (n <= 0) return {};
    if (b.size() - pos < static_cast<size_t>(n))
      throw ImageError(ImageError::kTruncated, "Failed to load image: Truncated File Read");
    return read(static_cast<size_t>(n));
  }
};

bool starts_with(std::span<const uint8_t> s, const char* lit, size_t n) {
  return s.size() >= n && std::memcmp(s.data(), lit, n) == 0;
}

struct JpegInfo {
  std::string mode;
  int w = 0, h = 0;
  bool sof = false, sos = false;
};

// JpegImagePlugin._open: returns the mode / size and fills the metadata Image.getexif() uses.
JpegInfo walk_markers(std::span<const uint8_t> b, Native& n) {
  JpegInfo ji;
  if (b.size() < 3 || b[0] != 0xFF || b[1] != 0xD8 || b[2] != 0xFF) unidentified();
  Reader r{b, 3};
  std::vector<uint8_t> s = {0xFF};
  std::vector<std::vector<uint8_t>> icclist;
  while (!s.empty()) {
    int i = s[0];
    if (i == 0xFF) {
      auto c = r.read(1);
      s.insert(s.end(), c.begin(), c.end());
      if (s.size() < 2) break;
      i = (s[0] << 8) | s[1];
    } else {
      auto c = r.read(1);
      s.assign(c.begin(), c.end());
      continue;
    }
    if (i >= 0xFFC0 && i <= 0xFFFE) {  // MARKER table
      const bool is_skip = i == 0xFFC4 || i == 0xFFCC || i == 0xFFDA || i == 0xFFDC || i == 0xFFDD || i == 0xFFDF;
      const bool is_sof = (i >= 0xFFC0 && i <= 0xFFCF && !is_skip && i != 0xFFC8) || i == 0xFFDE;
      const bool is_app = i >= 0xFFE0 && i <= 0xFFEF;
      const bool is_dqt = i == 0xFFDB;
      const bool is_com = i == 0xFFFE;
      if (is_skip) {
        long len = r.read_i16() - 2;
        r.safe_read(len);
      } else if (is_app || is_com) {
        long len = r.read_i16() - 2;
        auto seg = r.safe_read(len);
        if (i == 0xFFE0 && starts_with(seg, "JFIF", 4)) {
          if (seg.size() < 7) unidentified();  // i16(s, 5): struct.error
        } else if (i == 0xFFE1 && starts_with(seg, "Exif\0\0", 6)) {
          if (n.exif) {
            n.exif->insert(n.exif->end(), seg.begin() + 6, seg.end());
          } else {
            n.exif = std::vector<uint8_t>(seg.begin(), seg.end());
          }
        } else if (i == 0xFFE1 && starts_with(seg, "http://ns.adobe.com/xap/1.0/\0", 29)) {
          // s.split(b"\x00", 1)[1]
          auto nul = std::find(seg.begin(), seg.end(), uint8_t{0});
          n.xmp_bytes = std::vector<uint8_t>(nul + 1, seg.end());
        } else if (i == 0xFFE2 && starts_with(seg, "ICC_PROFILE\0", 12)) {
          icclist.emplace_back(seg.begin(), seg.end());
        } else if (i == 0xFFEE && starts_with(seg, "Adobe", 5)) {
          if (seg.size() < 7) unidentified();  // i16(s, 5)
        }
      } else if (is_sof) {
        long len = r.read_i16() - 2;
        auto seg = r.safe_read(len);
        if (seg.size() < 5) unidentified();  // i16(s, 3): struct.error
        ji.w = (seg[3] << 8) | seg[4];
        ji.h = (seg[1] << 8) | seg[2];
        if (seg[0] != 8) unidentified();     // "cannot handle %d-bit layers" (SyntaxError)
        if (seg.size() < 6) unidentified();  // s[5]: IndexError
        int layers = seg[5];
        if (layers == 1) {
          ji.mode = "L";
        } else if (layers == 3) {
          ji.mode = "RGB";
        } else if (layers == 4) {
          ji.mode = "CMYK";
        } else {
          unidentified();
        }
        if (!icclist.empty()) {
          std::sort(icclist.begin(), icclist.end());
          if (icclist[0].size() < 14) unidentified();  // icclist[0][13]: IndexError
          icclist.clear();
        }
        if ((seg.size() - 6) % 3 != 0) unidentified();  // short component triple: IndexError
        ji.sof = true;
      } else if (is_dqt) {
        long len = r.read_i16() - 2;
        auto seg = r.safe_read(len);
        size_t p = 0;
        while (p < seg.size()) {
          int v = seg[p];
          size_t qt = 1 + ((v >> 4) == 0 ? 1 : 2) * 64;
          if (seg.size() - p < qt) unidentified();  // "bad quantization table marker"
          p += qt;
        }
      }
      if (i == 0xFFDA) {
        ji.sos = true;
        break;
      }
      auto c = r.read(1);
      s.assign(c.begin(), c.end());
    } else if (i == 0xFFFF) {
      s = {0xFF};
    } else if (i == 0xFF00) {
      auto c = r.read(1);
      s.assign(c.begin(), c.end());
    } else {
      unidentified();  // "no marker found"
    }
  }
  if (ji.mode.empty() || ji.w <= 0 || ji.h <= 0) unidentified();  // "not identified by this driver"
  return ji;
}

struct ErrMgr {
  jpeg_error_mgr pub;
  std::jmp_buf jb;
  volatile bool eof = false;
};

void on_error_exit(j_common_ptr cinfo) {
  auto* e = reinterpret_cast<ErrMgr*>(cinfo->err);
  std::longjmp(e->jb, 1);
}

void on_emit_message(j_common_ptr cinfo, int level) {
  if (level < 0) {
    auto* e = reinterpret_cast<ErrMgr*>(cinfo->err);
    // jpeg_mem_src inserts a fake EOI at the end of the data; Pillow's suspending source would wait for more
    // data instead and ImageFile.load raises "image file is truncated".
    if (cinfo->err->msg_code == JWRN_JPEG_EOF) e->eof = true;
    cinfo->err->num_warnings++;
  }
}

void on_output_message(j_common_ptr) {}

// Decodes into px (w x h x comps, zero-filled). Returns the number of rows written. A libjpeg error is a
// "soft" error (*load_error): Pillow's decoder returns IMAGING_CODEC_BROKEN and ImageFile.load raises it only after
// clearing the tile, so vLLM ends up with the rows decoded so far. Running out of data is a hard error (Pillow's
// suspending source waits for more data and ImageFile.load raises "image file is truncated" inside its loop).
int run_libjpeg(std::span<const uint8_t> b, uint8_t* px, int w, int h, int comps, bool* load_error) {
  jpeg_decompress_struct cinfo;
  ErrMgr err;
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = on_error_exit;
  err.pub.emit_message = on_emit_message;
  err.pub.output_message = on_output_message;
  volatile int rows_done = 0;
  const size_t stride = static_cast<size_t>(w) * comps;
  if (setjmp(err.jb)) {
    jpeg_destroy_decompress(&cinfo);
    if (err.eof) throw ImageError(ImageError::kTruncated, "Failed to load image: image file is truncated");
    *load_error = true;
    return rows_done;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, b.data(), static_cast<unsigned long>(b.size()));
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = comps == 1 ? JCS_GRAYSCALE : comps == 3 ? JCS_RGB : JCS_CMYK;
  jpeg_start_decompress(&cinfo);
  if (err.eof) {
    jpeg_destroy_decompress(&cinfo);
    throw ImageError(ImageError::kTruncated, "Failed to load image: image file is truncated");
  }
  if (static_cast<int>(cinfo.output_width) != w || static_cast<int>(cinfo.output_components) != comps) {
    // Pillow's shuffle would write rows of its own size: not reachable for files libjpeg and Pillow agree on
    jpeg_destroy_decompress(&cinfo);
    *load_error = true;
    return 0;
  }
  const int limit = std::min<int>(h, static_cast<int>(cinfo.output_height));
  while (static_cast<int>(cinfo.output_scanline) < limit) {
    JSAMPROW row = px + stride * cinfo.output_scanline;
    if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) break;
    if (err.eof) break;
    rows_done = rows_done + 1;
  }
  const bool eof = err.eof;
  const int done = rows_done;
  jpeg_destroy_decompress(&cinfo);  // no jpeg_finish_decompress: Pillow does not need the EOI after the rows
  if (eof || done < h) throw ImageError(ImageError::kTruncated, "Failed to load image: image file is truncated");
  return done;
}

}  // namespace

Native decode_jpeg(std::span<const uint8_t> b, const DecodeOpts& opts) {
  Native n;
  n.format = "JPEG";
  JpegInfo ji = walk_markers(b, n);
  if (!ji.sos) {
    // no tile: ImageFile.load raises OSError("cannot load this image")
    throw ImageError(ImageError::kDecode, "Failed to load image: cannot load this image");
  }
  check_size(ji.w, ji.h, opts);
  n.mode = ji.mode;
  n.w = ji.w;
  n.h = ji.h;
  const int comps = ji.mode == "L" ? 1 : ji.mode == "RGB" ? 3 : 4;
  n.px.assign(static_cast<size_t>(n.w) * n.h * comps, 0);
  const int rows = run_libjpeg(b, n.px.data(), n.w, n.h, comps, &n.load_error);
  if (n.mode == "CMYK") {  // rawmode "CMYK;I" (JpegImagePlugin assumes Adobe conventions), decoded rows only
    const size_t m = static_cast<size_t>(rows) * n.w * 4;
    for (size_t i = 0; i < m; i++) n.px[i] = static_cast<uint8_t>(255 - n.px[i]);
  }
  return n;
}

}  // namespace fe::img
