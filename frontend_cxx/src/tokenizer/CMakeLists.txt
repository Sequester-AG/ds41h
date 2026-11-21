# fe_tokenizer: byte-level BPE + pre-tokenizer + added tokens + fast-tokenize chunking + DecodeStream.
add_library(fe_tokenizer STATIC tokenizer.cpp decode_stream.cpp)
target_include_directories(fe_tokenizer PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/install/include)
target_link_libraries(fe_tokenizer PUBLIC ${CMAKE_SOURCE_DIR}/third_party/install/lib/libonig.a pthread)
target_compile_options(fe_tokenizer PRIVATE -Wall -Wextra)
