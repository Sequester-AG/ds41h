// c1idx_host.cu: host helpers for the c1idx / c1eng phases. Built from the same
// headers the kernel includes, so the struct layouts match.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "c1e0.cuh"
#include "c1eng.cuh"
#include "c1idx.cuh"

namespace c1ixh {
static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "cuTensorMapEncodeTiled");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static void enc3(CUtensorMap* m, CUtensorMapDataType ty, void* p, cuuint64_t d0, cuuint64_t d1, cuuint64_t d2,
                 cuuint64_t s1, cuuint64_t s2, cuuint32_t b0, cuuint32_t b1, cuuint32_t b2) {
  const cuuint64_t d[3] = {d0, d1, d2}, st[2] = {s1, s2};
  const cuuint32_t b[3] = {b0, b1, b2}, es[3] = {1, 1, 1};
  const CUresult r = encode_fn()(m, ty, 3, p, d, st, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(r));
}

// one index layer's tensor maps -> CPU bytes [5 x 128]: 0 compressor wkv | wgate [rows][5120] bf16 (absent: zeros),
// 1 weights_proj [32][5120] bf16, 2 wq_b [4096][1280] e4m3, 3 / 4 the source's index-K pages (box bs / 8 positions)
// at kpages (16-B aligned), page stride kstride, npages pages
torch::Tensor ix_maps(int64_t comp_w, int64_t comp_rows, int64_t wp, int64_t wq, int64_t kpages, int64_t bs, int64_t kstride,
                      int64_t npages) {
  static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap");
  auto out = torch::zeros({5 * 128}, torch::kUInt8);
  CUtensorMap* m = reinterpret_cast<CUtensorMap*>(out.data_ptr());
  const auto U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16, U8 = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (comp_w) enc3(m, U16, (void*)comp_w, 64, (cuuint64_t)comp_rows, 80, 10240, 128, 64, 16, 8);
  enc3(m + 1, U16, (void*)wp, 64, 32, 80, 10240, 128, 64, 16, 8);
  enc3(m + 2, U8, (void*)wq, 128, 4096, 10, 1280, 128, 128, 32, 2);
  TORCH_CHECK(kpages % 16 == 0 && kstride % 16 == 0, "index-K pages: 16-B alignment");
  enc3(m + 3, U8, (void*)kpages, 128, (cuuint64_t)bs, (cuuint64_t)npages, 128, (cuuint64_t)kstride, 128, (cuuint32_t)bs, 1);
  enc3(m + 4, U8, (void*)kpages, 128, (cuuint64_t)bs, (cuuint64_t)npages, 128, (cuuint64_t)kstride, 128, 8, 1);
  return out;
}

// IxLayer -> CPU bytes. p (device addresses, 0 = null), in this order:
//  0 q8dbg 1 wodbg 2 lens 3 kpages 4 bt 5 logits 6 bmax 7 cand 8 gout 9 glen 10 cnt 11 tmk 12 slow 13 cand_n 14 cblk
//  15 cblk_n 16 tmk8 17 ain 18 pos 19 cos_sin 20 ks 21 wbuf 22 q 23 wq_s 24 wk 25 cnorm 26 knorm 27 ring 28 rslot
//  29 cslot 30 comp 31 kidx 32 tm 33 itick 34 selflag [35 dbt 36 preq: bt = dbt + preq[0] x dbt_ld when bt is null]
//  [37 tim 38 rowsel 39 dsc (v14: the distributed selection's scratch, DS_WORDS u32, zeroed)]
// v: 0 L 1 ratio 2 bs 3 kstride 4 ldl 5 ldb 6 ldg 7 kind 8 NR 9 ntc 10 ring_stride 11 pstride 12 skip [13 dbt_ld
//    [14 dthr 15 uni (v14)]]; f: 0 eps
torch::Tensor pack_ix(std::vector<int64_t> p, std::vector<int64_t> v, std::vector<double> f) {
  TORCH_CHECK(((p.size() == 35 && v.size() == 13) || (p.size() >= 37 && p.size() <= 39 && v.size() == 14) ||
               (p.size() == 40 && v.size() == 16)) && f.size() == 1,
              "pack_ix: 35 / 37 / 38 / 39 / 40 pointers, 13 / 14 / 16 ints, 1 float");
  c1ix::IxLayer L{};
  c1ix::IdxArgs& a = L.s;
  auto P = [&](int i) { return (void*)(uintptr_t)p[i]; };
  a.L = int(v[0]), a.ratio = int(v[1]), a.bs = int(v[2]);
  a.q8 = (const uint8_t*)P(0), a.wo = (const float*)P(1), a.lens = (const int*)P(2), a.kpages = (const uint8_t*)P(3);
  a.kstride = v[3];
  a.bt = (const int*)P(4), a.logits = (float*)P(5), a.ldl = int(v[4]), a.bmax = (float*)P(6), a.ldb = int(v[5]);
  a.cand = (int*)P(7), a.gout = (int*)P(8), a.ldg = int(v[6]), a.glen = (int*)P(9), a.cnt = (unsigned*)P(10);
  a.tmk = (const CUtensorMap*)P(11), a.slow = (int*)P(12), a.cand_n = (int*)P(13), a.cblk = (const int*)P(14);
  a.cblk_n = (const int*)P(15), a.tmk8 = (const CUtensorMap*)P(16);
  L.kind = int(v[7]), L.NR = int(v[8]), L.ratio = int(v[1]);
  L.ain = (const __nv_bfloat16*)P(17), L.pos = (const long long*)P(18), L.cos_sin = (const float*)P(19);
  L.ntc = int(v[9]), L.nt1 = int(v[9]) + 2;
  L.ks = (float*)P(20), L.wbuf = (__nv_bfloat16*)P(21), L.q = (__nv_bfloat16*)P(22), L.wq_s = (const uint8_t*)P(23);
  L.wk = (const __nv_bfloat16*)P(24), L.cnorm = (const __nv_bfloat16*)P(25), L.knorm = (const __nv_bfloat16*)P(26);
  L.eps = float(f[0]);
  L.ring = (float*)P(27), L.ring_stride = v[10], L.rslot = (const long long*)P(28), L.cslot = (const long long*)P(29);
  L.comp = (uint8_t*)P(30), L.kidx = (uint8_t*)P(31), L.pstride = v[11], L.tm = (const CUtensorMap*)P(32);
  L.skip = int(v[12]), L.itick = (unsigned*)P(33), L.selflag = (unsigned*)P(34);
  if (p.size() >= 37) a.dbt = (const int*)P(35), a.preq = (const long long*)P(36), a.dbt_ld = v[13];
  if (p.size() >= 38) L.tim = (unsigned long long*)P(37);   // (tests: the glue's sub-stamps)
  if (p.size() >= 39) L.rowsel = (unsigned*)P(38);         // (hook 6)
  if (p.size() >= 40) a.dsc = (unsigned*)P(39), a.dthr = int(v[14]), a.uni = int(v[15]);   // (v14)
  TORCH_CHECK(a.dsc == nullptr || (reinterpret_cast<uintptr_t>(a.dsc) & 15) == 0, "pack_ix: dsc 16-B aligned");
  TORCH_CHECK(a.bt != nullptr || (a.dbt != nullptr && a.preq != nullptr), "pack_ix: bt or dbt + preq");
  // 16-B alignment of what the kernels copy with cp.async / cp.async.bulk: q, wbuf (q quant staging), wk / cnorm /
  // knorm / cos_sin rows (X2's ring chunks, kind 2)
  auto al16 = [](const void* q) { return (reinterpret_cast<uintptr_t>(q) & 15) == 0; };
  TORCH_CHECK(al16(L.q) && al16(L.wbuf) && al16(L.cos_sin), "pack_ix: q / wbuf / cos_sin must be 16-B aligned");
  TORCH_CHECK(L.kind != 2 || (al16(L.wk) && al16(L.cnorm) && al16(L.knorm)), "pack_ix: wk / cnorm / knorm 16-B aligned");
  auto out = torch::empty({(int64_t)sizeof(L)}, torch::kUInt8);
  memcpy(out.data_ptr(), &L, sizeof(L));
  return out;
}

// EngArgs -> CPU bytes. p: 0 kv 1 keep 2..5 kvl[4] 6..9 keepl[4] 10..13 kflag[4] 14 qw 15 kw 16 part 17 x1 18 zrow 19 cnt
// 20 tick [21 e0 (layer 0: the E0Args; li = -1) 22 kvloc 23 keeploc 24 kvcnt 25 e0tick (layers 1 / 14, local mode)];
// v: 0 li 1 kv_stride; f: 0 eps 1 clamp
torch::Tensor pack_eg(std::vector<int64_t> p, std::vector<int64_t> v, std::vector<double> f) {
  TORCH_CHECK((p.size() == 21 || p.size() == 26) && v.size() == 2 && f.size() == 2, "pack_eg: 21 / 26 pointers, 2 ints, 2 floats");
  c1eg::EngArgs e{};
  auto P = [&](int i) { return (void*)(uintptr_t)p[i]; };
  e.li = int(v[0]), e.kv_stride = v[1];
  e.kv = (const __nv_bfloat16*)P(0), e.keep = (const unsigned char*)P(1);
  for (int s = 0; s < c1eg::W; ++s) {
    e.kvl[s] = (__nv_bfloat16*)P(2 + s), e.keepl[s] = (unsigned char*)P(6 + s), e.kflag[s] = (unsigned*)P(10 + s);
  }
  e.qw = (const __nv_bfloat16*)P(14), e.kw = (const __nv_bfloat16*)P(15);
  e.eps = f[0], e.clampv = f[1];
  e.part = (double*)P(16), e.x1 = (__nv_bfloat16*)P(17), e.zrow = (const __nv_bfloat16*)P(18);
  e.cnt = (unsigned*)P(19), e.tick = (unsigned*)P(20);
  if (p.size() == 26) {
    e.e0 = P(21), e.kvloc = (const __nv_bfloat16*)P(22), e.keeploc = (const unsigned char*)P(23);
    e.kvcnt = (const unsigned*)P(24), e.e0tick = (const unsigned*)P(25);
  }
  auto out = torch::empty({(int64_t)sizeof(e)}, torch::kUInt8);
  memcpy(out.data_ptr(), &e, sizeof(e));
  return out;
}
// E0Args -> CPU bytes. p: 0 packed0 1 packed1 (EngramPhys.gemm8[l].packed of this GPU) 2 (unused) 3 xo0 4 xo1 5 keepo,
// 6 + 2 s + li xl[s][li], 14 + s keepl[s], 18 + s xcnt[s], 22 + 2 s + li kvd[s][li], 30 + s kvcnt[s], 34 tick;
// v: 0 NR 1 owner 2 cstride (bytes between the packed k-chunks)
torch::Tensor pack_e0(std::vector<int64_t> p, std::vector<int64_t> v) {
  TORCH_CHECK(p.size() == 35 && v.size() == 3, "pack_e0: 35 pointers, 3 ints");
  c1e0::E0Args e{};
  auto P = [&](int i) { return (void*)(uintptr_t)p[i]; };
  e.NR = int(v[0]), e.owner = int(v[1]), e.cstride = v[2];
  e.wp[0] = (const uint8_t*)P(0), e.wp[1] = (const uint8_t*)P(1);
  TORCH_CHECK(e.wp[0] && e.wp[1] && (p[0] % 16) == 0 && (p[1] % 16) == 0 && v[2] % 16 == 0, "pack_e0: packed weights");
  e.xo[0] = (const __nv_bfloat16*)P(3), e.xo[1] = (const __nv_bfloat16*)P(4), e.keepo = (const unsigned char*)P(5);
  for (int s = 0; s < c1e0::W; ++s) {
    for (int li = 0; li < c1e0::NL; ++li) {
      e.xl[s][li] = (__nv_bfloat16*)P(6 + 2 * s + li);
      e.kvd[s][li] = (__nv_bfloat16*)P(22 + 2 * s + li);
    }
    e.keepl[s] = (unsigned char*)P(14 + s), e.xcnt[s] = (unsigned*)P(18 + s), e.kvcnt[s] = (unsigned*)P(30 + s);
  }
  e.tick = (unsigned*)P(34);
  auto out = torch::empty({(int64_t)sizeof(e)}, torch::kUInt8);
  memcpy(out.data_ptr(), &e, sizeof(e));
  return out;
}
int64_t e0_bytes() { return (int64_t)sizeof(c1e0::E0Args); }
int64_t ix_bytes() { return (int64_t)sizeof(c1ix::IxLayer); }
int64_t eg_bytes() { return (int64_t)sizeof(c1eg::EngArgs); }
int64_t xs_bytes() { return (int64_t)c1ix::XS_BYTES; }
int64_t ds_words() { return (int64_t)c1ix::DS_WORDS; }
int64_t ds_dbg() { return (int64_t)c1ix::DS_DBG; }
}  // namespace c1ixh

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ix_maps", &c1ixh::ix_maps);
  m.def("pack_ix", &c1ixh::pack_ix);
  m.def("pack_eg", &c1ixh::pack_eg);
  m.def("pack_e0", &c1ixh::pack_e0);
  m.def("e0_bytes", &c1ixh::e0_bytes);
  m.def("ix_bytes", &c1ixh::ix_bytes);
  m.def("eg_bytes", &c1ixh::eg_bytes);
  m.def("xs_bytes", &c1ixh::xs_bytes);
  m.def("ds_words", &c1ixh::ds_words);
  m.def("ds_dbg", &c1ixh::ds_dbg);
}
