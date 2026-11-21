// Pillow 13 libImaging/Resample.c for RGB images with the BICUBIC filter (a = -0.5, support 2): double
// coefficients from precompute_coeffs, normalised to 22-bit fixed point (PRECISION_BITS = 32 - 8 - 2), two
// passes (each only when that size changes; order as in resize_bicubic) through an 8-bit intermediate, each output
// rounded with 1 << 21 and clipped like clip8(). Integer sums are order independent, so the loops below are
// arranged for vectorisation; the coefficients and bounds are computed exactly as Pillow does.
//
// Also ImageOps.contain / ImageOps.pad and the mm_preprocess.load_image size math.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "image/image_internal.h"

namespace fe::img {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

double bicubic_filter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return std::fma(std::fma(x, 1.5, -2.5) * x, x, 1.0);   // ((a + 2) x - (a + 3)) x x + 1
  if (x < 2.0) return std::fma(std::fma(x - 5.0, x, 8.0), x, -4.0) * -0.5;  // (((x - 5) x + 8) x - 4) a
  return 0.0;
}

// precompute_coeffs + normalize_coeffs_8bpc. in0/in1 are the float box edges (0 and the input size).
int precompute_coeffs(int in_size, float in0, float in1, int out_size, std::vector<int>& bounds,
                      std::vector<int32_t>& kk) {
  const double support0 = 2.0;  // BICUBIC.support
  double scale, filterscale;
  filterscale = scale = static_cast<double>(in1 - in0) / out_size;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = support0 * filterscale;
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  std::vector<double> pre(static_cast<size_t>(out_size) * ksize, 0.0);
  bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  for (int xx = 0; xx < out_size; xx++) {
    const double center = std::fma(xx + 0.5, scale, static_cast<double>(in0));  // fmadd (Resample.c:277)
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &pre[static_cast<size_t>(xx) * ksize];
    int x;
    for (x = 0; x < xmax; x++) {
      const double w = bicubic_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++) {
      if (ww != 0.0) k[x] /= ww;
    }
    for (; x < ksize; x++) k[x] = 0;
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(pre.size());
  for (size_t i = 0; i < pre.size(); i++) {
    kk[i] = pre[i] < 0 ? static_cast<int32_t>(-0.5 + pre[i] * (1 << kPrecisionBits))
                       : static_cast<int32_t>(0.5 + pre[i] * (1 << kPrecisionBits));
  }
  return ksize;
}

inline uint8_t clip8(int32_t in) {
  const int32_t v = in >> kPrecisionBits;
  return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

// Both passes are exact integer arithmetic (u8 x int32 coefficient sums < 2^31), so the NEON kernels below give
// the same bytes as Pillow's scalar loops: clip8(v) == saturate_u8(saturate_u16(v >> 22)).

// horizontal pass over rows [y0, y0 + rows) of in -> out (rows x out_w)
void resample_h(const Rgb& in, int y0, int rows, int out_w, int ksize, const std::vector<int>& bounds,
                const std::vector<int32_t>& kk, std::vector<uint8_t>& out) {
  out.resize(static_cast<size_t>(rows) * out_w * 3);
  const int32_t half = 1 << (kPrecisionBits - 1);
#if defined(__aarch64__)
  // Widen each source row once to (R, G, B, 0) int32 lanes (zero-padded by ksize pixels), then run 4 output
  // pixels at a time over all ksize taps: precompute_coeffs zero-fills the taps past xmax, so the padded taps add
  // exactly 0 and the 4 independent accumulator chains hide the multiply-accumulate latency.
  std::vector<int32_t> wide((static_cast<size_t>(in.w) + ksize + 1) * 4, 0);
  for (int yy = 0; yy < rows; yy++) {
    const uint8_t* src = in.px.data() + static_cast<size_t>(yy + y0) * in.w * 3;
    int32_t* wp = wide.data();
    for (int x = 0; x < in.w; x++) {
      wp[4 * x] = src[3 * x];
      wp[4 * x + 1] = src[3 * x + 1];
      wp[4 * x + 2] = src[3 * x + 2];
    }
    uint8_t* dst = out.data() + static_cast<size_t>(yy) * out_w * 3;
    auto store = [&](int xx, int32x4_t a) {
      const uint16x4_t n16 = vqmovun_s32(vshrq_n_s32(a, kPrecisionBits));
      const uint8x8_t n8 = vqmovn_u16(vcombine_u16(n16, n16));
      dst[3 * xx] = vget_lane_u8(n8, 0);
      dst[3 * xx + 1] = vget_lane_u8(n8, 1);
      dst[3 * xx + 2] = vget_lane_u8(n8, 2);
    };
    int xx = 0;
    for (; xx + 4 <= out_w; xx += 4) {
      const int32_t* k0 = &kk[static_cast<size_t>(xx) * ksize];
      const int32_t* k1 = k0 + ksize;
      const int32_t* k2 = k1 + ksize;
      const int32_t* k3 = k2 + ksize;
      const int32_t* s0 = wp + static_cast<size_t>(bounds[xx * 2]) * 4;
      const int32_t* s1 = wp + static_cast<size_t>(bounds[xx * 2 + 2]) * 4;
      const int32_t* s2 = wp + static_cast<size_t>(bounds[xx * 2 + 4]) * 4;
      const int32_t* s3 = wp + static_cast<size_t>(bounds[xx * 2 + 6]) * 4;
      int32x4_t a0 = vdupq_n_s32(half), a1 = a0, a2 = a0, a3 = a0;
      for (int x = 0; x < ksize; x++) {
        a0 = vmlaq_n_s32(a0, vld1q_s32(s0 + 4 * x), k0[x]);
        a1 = vmlaq_n_s32(a1, vld1q_s32(s1 + 4 * x), k1[x]);
        a2 = vmlaq_n_s32(a2, vld1q_s32(s2 + 4 * x), k2[x]);
        a3 = vmlaq_n_s32(a3, vld1q_s32(s3 + 4 * x), k3[x]);
      }
      store(xx, a0), store(xx + 1, a1), store(xx + 2, a2), store(xx + 3, a3);
    }
    for (; xx < out_w; xx++) {
      const int32_t* k = &kk[static_cast<size_t>(xx) * ksize];
      const int32_t* sp = wp + static_cast<size_t>(bounds[xx * 2]) * 4;
      int32x4_t a = vdupq_n_s32(half);
      for (int x = 0; x < ksize; x++) a = vmlaq_n_s32(a, vld1q_s32(sp + 4 * x), k[x]);
      store(xx, a);
    }
  }
#else
  for (int yy = 0; yy < rows; yy++) {
    const uint8_t* src = in.px.data() + static_cast<size_t>(yy + y0) * in.w * 3;
    uint8_t* dst = out.data() + static_cast<size_t>(yy) * out_w * 3;
    for (int xx = 0; xx < out_w; xx++) {
      const int xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
      const int32_t* k = &kk[static_cast<size_t>(xx) * ksize];
      const uint8_t* s = src + static_cast<size_t>(xmin) * 3;
      int32_t s0 = half, s1 = half, s2 = half;
      for (int x = 0; x < xmax; x++) {
        const int32_t c = k[x];
        s0 += s[3 * x] * c;
        s1 += s[3 * x + 1] * c;
        s2 += s[3 * x + 2] * c;
      }
      dst[3 * xx] = clip8(s0);
      dst[3 * xx + 1] = clip8(s1);
      dst[3 * xx + 2] = clip8(s2);
    }
  }
#endif
}

// vertical pass: in_rows rows of width w (row r of `src` is source row r + row_offset of the bounds)
void resample_v(const uint8_t* src, int w, int out_h, int ksize, const std::vector<int>& bounds, int row_offset,
                const std::vector<int32_t>& kk, uint8_t* dst) {
  const size_t n = static_cast<size_t>(w) * 3;
  const int32_t half = 1 << (kPrecisionBits - 1);
  for (int yy = 0; yy < out_h; yy++) {
    const int ymin = bounds[yy * 2] - row_offset, ymax = bounds[yy * 2 + 1];
    const int32_t* k = &kk[static_cast<size_t>(yy) * ksize];
    const uint8_t* base = src + static_cast<size_t>(ymin) * n;
    uint8_t* d = dst + static_cast<size_t>(yy) * n;
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
      int32x4_t a0 = vdupq_n_s32(half), a1 = a0, a2 = a0, a3 = a0;
      const uint8_t* r = base + i;
      for (int y = 0; y < ymax; y++, r += n) {
        const int32_t c = k[y];
        const uint8x16_t v = vld1q_u8(r);
        const uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_u8(vget_high_u8(v));
        a0 = vmlaq_n_s32(a0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))), c);
        a1 = vmlaq_n_s32(a1, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo))), c);
        a2 = vmlaq_n_s32(a2, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))), c);
        a3 = vmlaq_n_s32(a3, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi))), c);
      }
      const uint16x8_t n0 = vcombine_u16(vqmovun_s32(vshrq_n_s32(a0, kPrecisionBits)),
                                         vqmovun_s32(vshrq_n_s32(a1, kPrecisionBits)));
      const uint16x8_t n1 = vcombine_u16(vqmovun_s32(vshrq_n_s32(a2, kPrecisionBits)),
                                         vqmovun_s32(vshrq_n_s32(a3, kPrecisionBits)));
      vst1q_u8(d + i, vcombine_u8(vqmovn_u16(n0), vqmovn_u16(n1)));
    }
#endif
    for (; i < n; i++) {
      int32_t a = half;
      const uint8_t* r = base + i;
      for (int y = 0; y < ymax; y++, r += n) a += *r * k[y];
      d[i] = clip8(a);
    }
  }
}

// Python round() of a double: round half to even
inline long long pyround(double v) { return static_cast<long long>(std::nearbyint(v)); }

}  // namespace

Rgb resize_bicubic(const Rgb& in, int w, int h) {
  if (in.w == w && in.h == h) return in;  // Image.resize: self.copy()
  if (w < 1 || h < 1) throw ImageError(ImageError::kInternal, "height and width must be > 0");
  // _resize: a box of the full image whose size equals the request is a crop (only when both sizes match,
  // handled above); otherwise ImagingResample.
  const bool need_h = w != in.w;
  const bool need_v = h != in.h;
  std::vector<int> bh, bv;
  std::vector<int32_t> kh, kv;
  const int ksh = precompute_coeffs(in.w, 0.0f, static_cast<float>(in.w), w, bh, kh);
  const int ksv = precompute_coeffs(in.h, 0.0f, static_cast<float>(in.h), h, bv, kv);
  const int ybox_first = bv[0];
  const int ybox_last = bv[h * 2 - 2] + bv[h * 2 - 1];
  Rgb out;
  out.w = w;
  out.h = h;
  // Pillow 13 ImagingResampleInner (Resample.c:865-867, read from the wheel's DWARF line table + disassembly and
  // verified on 1367 random cases): when both passes are needed, the vertical pass runs first iff it removes more
  // than twice as many pixels as the horizontal one: (in_h - out_h) > 0 && (in_h - out_h) > 2 * (in_w - out_w).
  const int dy = in.h - h, dx = in.w - w;
  const bool vertical_first = need_h && need_v && dy > 0 && dy > 2 * dx;
  if (vertical_first) {
    Rgb tmp;
    tmp.w = in.w;
    tmp.h = h;
    tmp.px.resize(static_cast<size_t>(in.w) * h * 3);
    resample_v(in.px.data(), in.w, h, ksv, bv, 0, kv, tmp.px.data());
    resample_h(tmp, 0, h, w, ksh, bh, kh, out.px);
  } else if (need_h) {
    std::vector<uint8_t> tmp;
    resample_h(in, ybox_first, ybox_last - ybox_first, w, ksh, bh, kh, tmp);
    if (need_v) {
      out.px.resize(static_cast<size_t>(w) * h * 3);
      resample_v(tmp.data(), w, h, ksv, bv, ybox_first, kv, out.px.data());
    } else {
      out.px = std::move(tmp);  // ybox covers every row when the height is unchanged
    }
  } else {
    out.px.resize(static_cast<size_t>(w) * h * 3);
    resample_v(in.px.data(), w, h, ksv, bv, 0, kv, out.px.data());
  }
  return out;
}

Rgb pad(const Rgb& in, int W, int H, uint8_t fill) {
  // ImageOps.contain(image, size, BICUBIC)
  int nw = W, nh = H;
  const double im_ratio = static_cast<double>(in.w) / in.h;
  const double dest_ratio = static_cast<double>(W) / H;
  if (im_ratio != dest_ratio) {
    if (im_ratio > dest_ratio) {
      const long long new_h = pyround(static_cast<double>(in.h) / in.w * W);
      if (new_h != H) nh = static_cast<int>(new_h);
    } else {
      const long long new_w = pyround(static_cast<double>(in.w) / in.h * H);
      if (new_w != W) nw = static_cast<int>(new_w);
    }
  }
  Rgb resized;
  if (std::min(nw, nh) == 0) {
    resized.w = nw, resized.h = nh;  // Image.new(mode, size): empty
  } else {
    resized = resize_bicubic(in, nw, nh);
  }
  if (nw == W && nh == H) return resized;
  Rgb out;
  out.w = W, out.h = H;
  out.px.assign(static_cast<size_t>(W) * H * 3, fill);
  int x = 0, y = 0;
  if (nw != W) {
    x = static_cast<int>(pyround((W - nw) * 0.5));
  } else {
    y = static_cast<int>(pyround((H - nh) * 0.5));
  }
  for (int r = 0; r < resized.h; r++) {
    const int dy = r + y;
    if (dy < 0 || dy >= H) continue;
    const int x0 = std::max(0, x), x1 = std::min(W, x + resized.w);
    if (x1 <= x0) continue;
    std::memcpy(out.px.data() + (static_cast<size_t>(dy) * W + x0) * 3,
                resized.px.data() + (static_cast<size_t>(r) * resized.w + (x0 - x)) * 3,
                static_cast<size_t>(x1 - x0) * 3);
  }
  return out;
}

// mm_preprocess.load_image:135-153 (+ safe_resize / solve_resize_ratio / llm_grid), Python float semantics
Plan plan_resize(int img_w, int img_h, const VisionConfig& vc) {
  const int p = vc.patch_size;
  const int ds = vc.downsample_ratio;
  const int max_n = vc.max_n_token;
  double width = img_w, height = img_h;
  if (vc.max_wh_ratio && width > height * *vc.max_wh_ratio) width = height * *vc.max_wh_ratio;
  const double area = width * height;
  if (0 < area && area < static_cast<double>(vc.min_pixels)) {
    const double ratio = std::pow(static_cast<double>(vc.min_pixels) / area, 0.5);
    width = std::trunc(width * ratio);   // int()
    height = std::trunc(height * ratio);
  }
  auto ceil_div_f = [](double a, int b) { return static_cast<long long>(std::ceil(a / b)); };
  long long best_w = ceil_div_f(width, p) * p;
  long long best_h = ceil_div_f(height, p) * p;
  auto grid = [&](long long bh, long long bw, long long* gh, long long* gw) {
    *gh = static_cast<long long>(std::ceil(static_cast<double>(bh / p) / ds));
    *gw = static_cast<long long>(std::ceil(static_cast<double>(bw / p) / ds));
  };
  long long nlh, nlw;
  grid(best_h, best_w, &nlh, &nlw);
  if (nlh * (nlw + 1) + 2 > max_n) {
    // solve_resize_ratio(height, width, ...)
    const double r = height / width;
    const double max_w_float = std::sqrt((max_n - 2) / r + 0.25) - 0.5;
    const double max_h_float = max_w_float * r;
    const int cell = p * ds;
    if (max_w_float < 1.0) {
      best_h = static_cast<long long>((max_n - 2) / 2) * cell;
      best_w = cell;
    } else if (max_h_float < 1.0) {
      best_h = cell;
      best_w = static_cast<long long>(max_n - 3) * cell;
    } else {
      const double b1 = static_cast<double>(static_cast<long long>(std::floor(max_w_float)) * cell) / width;
      const double b2 = static_cast<double>(static_cast<long long>(std::floor(max_h_float)) * cell) / height;
      const double beta = b2 < b1 ? b2 : b1;  // min(b1, b2)
      best_h = static_cast<long long>(std::floor(height * beta / p)) * p;
      best_w = static_cast<long long>(std::floor(width * beta / p)) * p;
    }
    grid(best_h, best_w, &nlh, &nlw);
    if (nlh * (nlw + 1) + 2 > max_n) throw ImageError(ImageError::kInternal, "image token budget assertion");
  }
  Plan pl;
  pl.best_w = static_cast<int>(best_w);
  pl.best_h = static_cast<int>(best_h);
  pl.n_vit_h = static_cast<int>(best_h / p);
  pl.n_vit_w = static_cast<int>(best_w / p);
  pl.n_llm_h = static_cast<int>(nlh);
  pl.n_llm_w = static_cast<int>(nlw);
  pl.stretch = vc.max_wh_ratio && static_cast<double>(img_w) >= *vc.max_wh_ratio * img_h;
  return pl;
}

Rgb transform(const Rgb& rgb, const Plan& pl) {
  if (pl.stretch) return resize_bicubic(rgb, pl.best_w, pl.best_h);  // image.resize((bw, bh)), BICUBIC default
  return pad(rgb, pl.best_w, pl.best_h, 127);
}

}  // namespace fe::img
