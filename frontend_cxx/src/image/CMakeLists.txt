# component IMAGE: image decode + DeepSeek-V4.1 preprocessing, bit-exact with vLLM 0.32 / Pillow 13.
# Links the vendored static libjpeg-turbo 3.2.0 (Pillow 13's version) and system zlib; libwebp is dlopen'ed.
# (No libpng: Pillow decodes PNG with its own zlib-based decoder, which this component ports.)
find_package(ZLIB REQUIRED)
add_library(fe_image STATIC
  preprocess.cpp
  decode_jpeg.cpp
  decode_png.cpp
  decode_gif.cpp
  decode_webp.cpp
  convert.cpp
  resample.cpp
  hash.cpp)
target_include_directories(fe_image PUBLIC ${CMAKE_SOURCE_DIR}/src PRIVATE ${TP}/include)
# Python / Pillow float semantics: no fused multiply-add contraction in the size math and filter weights
target_compile_options(fe_image PRIVATE -ffp-contract=off -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(fe_image PUBLIC ${TP}/lib64/libjpeg.a ZLIB::ZLIB ${CMAKE_DL_LIBS})
