# fe_render: DeepSeek V4.1 chat template + vLLM request validation + dsml_roundtrip request/prompt patches.
add_library(fe_render STATIC render.cpp encoding.cpp pyd.cpp pyval.cpp)
target_include_directories(fe_render PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(fe_render PUBLIC fe_common)
target_compile_options(fe_render PRIVATE -Wall -Wextra -O2)
