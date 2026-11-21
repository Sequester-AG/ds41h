# fe_common: JSON (simdjson parse -> ordered DOM, Python-compatible writers) + msgpack (msgspec-compatible).
# Owner: A0 (COMMON). Consumers link `fe_common`; it carries the simdjson implementation.
set(FE_COMMON_SOURCES json.cpp json_dump.cpp)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/msgpack.cpp)
  list(APPEND FE_COMMON_SOURCES msgpack.cpp)
endif()
add_library(fe_common STATIC ${FE_COMMON_SOURCES} ${TP}/include/simdjson/simdjson.cpp)
target_include_directories(fe_common PUBLIC ${CMAKE_SOURCE_DIR}/src ${TP}/include)
target_compile_options(fe_common PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
set_source_files_properties(${TP}/include/simdjson/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-O3;-w;-g0")
set_source_files_properties(json.cpp json_dump.cpp PROPERTIES COMPILE_OPTIONS "-O3")

# The top-level CMakeLists adds tests/<component> itself (FE_COMPONENTS); only add ours if it does not.
if(NOT DEFINED FE_COMPONENTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/common/CMakeLists.txt)
  add_subdirectory(${CMAKE_SOURCE_DIR}/tests/common ${CMAKE_BINARY_DIR}/tests/common)
endif()
