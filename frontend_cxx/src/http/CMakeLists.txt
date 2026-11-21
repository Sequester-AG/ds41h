# fe_http: HTTP/1.1 server
add_library(fe_http STATIC http_server.cpp)
target_include_directories(fe_http PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(fe_http PUBLIC pthread)

# TLS (HTTPS, vLLM's --ssl-* args) via OpenSSL >= 3.0, compiled in when it is found. Builds without
# system headers point CMake at them, e.g.
#   -DOPENSSL_INCLUDE_DIR=<dir with openssl/> -DOPENSSL_SSL_LIBRARY=/usr/lib/x86_64-linux-gnu/libssl.so.3
#   -DOPENSSL_CRYPTO_LIBRARY=/usr/lib/x86_64-linux-gnu/libcrypto.so.3
option(FE_WITH_TLS "fe_http: TLS via OpenSSL when it is found" ON)
set(FE_HTTP_TLS OFF PARENT_SCOPE)
if(FE_WITH_TLS)
  find_package(OpenSSL 3.0 COMPONENTS SSL Crypto)
  if(OPENSSL_FOUND)
    # visible to the binary and tests/http, which live outside this directory
    set_target_properties(OpenSSL::SSL OpenSSL::Crypto PROPERTIES IMPORTED_GLOBAL TRUE)
    target_compile_definitions(fe_http PUBLIC FE_HTTP_TLS=1)
    target_link_libraries(fe_http PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    set(FE_HTTP_TLS ON PARENT_SCOPE)
    message(STATUS "fe_http: TLS ON (OpenSSL ${OPENSSL_VERSION}: ${OPENSSL_SSL_LIBRARY})")
  else()
    message(STATUS "fe_http: OpenSSL not found, TLS OFF (an --ssl-certfile start fails)")
  endif()
else()
  message(STATUS "fe_http: TLS OFF (FE_WITH_TLS=OFF)")
endif()
