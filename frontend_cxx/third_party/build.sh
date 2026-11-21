#!/bin/bash
# Builds the vendored third-party libs (static, -fPIC) into third_party/install (CC / CXX: default gcc / g++).
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); P=$T/install; B=$T/build; mkdir -p $P $B
export CC=${CC:-gcc} CXX=${CXX:-g++}
cd $B
tar -xzf $T/dl/zeromq-4.3.5.tar.gz; tar -xzf $T/dl/onig-6.9.10.tar.gz; tar -xzf $T/dl/libjpeg-turbo-3.2.0.tar.gz
cmake -S zeromq-4.3.5 -B zmq-b -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DBUILD_SHARED=OFF -DBUILD_STATIC=ON -DZMQ_BUILD_TESTS=OFF -DWITH_LIBSODIUM=OFF -DENABLE_CURVE=OFF -DWITH_DOCS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON > zmq.cfg.log 2>&1
cmake --build zmq-b -j 16 > zmq.build.log 2>&1 && cmake --install zmq-b > /dev/null
(cd onig-6.9.10 && ./configure --prefix=$P --disable-shared --enable-static --with-pic CC=$CC > ../onig.cfg.log 2>&1 && make -j 16 > ../onig.build.log 2>&1 && make install > /dev/null)
cmake -S libjpeg-turbo-3.2.0 -B ljt-b -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_TURBOJPEG=OFF > ljt.cfg.log 2>&1
cmake --build ljt-b -j 16 > ljt.build.log 2>&1 && cmake --install ljt-b > /dev/null
mkdir -p $P/include/simdjson && cp $T/dl/simdjson.h $T/dl/simdjson.cpp $P/include/simdjson/
ls $P/lib* $P/include
echo THIRD_PARTY_BUILD_OK
