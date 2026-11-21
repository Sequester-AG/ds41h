# fpr/core/tests/core.mk
# -O2 -std=c++17 -ffp-contract=off.
CXX ?= g++
OUT ?= build
XGR ?= xgrammar
XGRLIB ?= $(OUT)/libxgrammar.a
SRC := ..
INC ?= $(SRC)/include      # the component headers (the build snapshot's, see README_core.md)
CXXFLAGS := -O2 -std=c++17 -ffp-contract=off -fPIC -Wall -Wextra -Wno-unused-parameter -pthread
XINC := -I$(XGR)/include -isystem $(XGR)/3rdparty/picojson -isystem $(XGR)/3rdparty/dlpack/include -DXGRAMMAR_ENABLE_CPPTRACE=0

WIRE := $(OUT)/v1wire.o $(OUT)/pyfloat.o
GRAM := $(OUT)/grammar.o
CORE := $(OUT)/lifecycle.o $(OUT)/core.o

PY ?= python3
PYINC ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND ?= $(shell $(PY) -c "import pybind11; print(pybind11.get_include())")
PYEXT ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

all: pymod core2b

# stage 2a: the pybind module (make -f core.mk pymod PY=$DS41H_HOME/venv/bin/python)
pymod: $(OUT)/fpr_core_py$(PYEXT)
$(OUT)/fpr_core_py$(PYEXT): $(SRC)/src/core/py_core.cc $(SRC)/src/core/py_cfg.h $(OUT)/lifecycle.o $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(WIRE) $(GRAM)
	$(CXX) $(CXXFLAGS) -shared -I$(INC) -I$(SRC)/src/core -isystem $(PYINC) -isystem $(PYBIND) $(XINC) $< $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(OUT)/lifecycle.o $(GRAM) $(WIRE) $(XGRLIB) -o $@.new && mv $@.new $@   # (a running server keeps its mapped copy)

# stage 2b: the engine process module fpr_core2b (src/core2b: V1 socket IO + the ICore loop), libzmq = pyzmq's bundled
# copy (the one the engine process already loaded), zmq.h from frontend_cxx's third-party build (both 4.3.5)
#   make -f core.mk core2b PY=$DS41H_HOME/venv/bin/python OUT=... XGR=... XGRLIB=... CXX=...
ZMQ_INC ?= $(SRC)/../../frontend_cxx/third_party/build/zeromq-4.3.5/include
ZMQ_LIB ?= $(shell $(PY) -c "import glob, os, zmq; print(sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(zmq.__file__)), 'pyzmq.libs', 'libzmq*')))[0])")
core2b: $(OUT)/fpr_core2b$(PYEXT)
$(OUT)/fpr_core2b$(PYEXT): $(SRC)/src/core2b/py_core2b.cc $(SRC)/src/core2b/server.cc $(SRC)/src/core2b/server.h $(SRC)/src/core2b/zio.h $(SRC)/src/core2b/native_worker.cc $(SRC)/src/core2b/native_worker.h $(SRC)/src/core2b/native_mixed.cc $(SRC)/src/core2b/native_mixed.h $(SRC)/src/core2b/mixed_exec.cpp $(SRC)/src/core2b/mixed_exec.h $(SRC)/src/core2b/mixed_issue_nd.cpp $(SRC)/src/core2b/mixed_issue_nd.h $(SRC)/src/core2b/span_program.cpp $(SRC)/src/core2b/span_program.h $(SRC)/src/core2b/driver_bridge.cpp $(SRC)/src/core2b/driver_bridge.h $(SRC)/src/core2b/cabi_libs.cpp $(SRC)/src/core2b/cabi_libs.h $(SRC)/src/core2b/mixed_owner.cpp $(SRC)/src/core2b/mixed_owner.h $(SRC)/src/core2b/owner_p1.cpp $(SRC)/src/core2b/owner_engram.cpp $(SRC)/src/p1/gt_build.cpp $(SRC)/src/p1/pf_build.cpp $(SRC)/src/p1/steps_build.cpp $(SRC)/src/p1/cp_build.cpp $(SRC)/src/p1/p1_cmirror.h $(SRC)/src/p0/mixed_build.cpp $(SRC)/src/p0/mixed_build.h $(SRC)/src/p0/inputs_build.cpp $(SRC)/src/p0/inputs_build.h $(SRC)/src/p0/moe_decide.cpp $(SRC)/src/p0/moe_decide.h $(SRC)/src/p0/engram_host.cpp $(SRC)/src/p0/engram_host.h $(SRC)/src/core2b/nd_api.h $(SRC)/src/core2b/engine_handover.h $(SRC)/src/core2b/fake_nd.cc $(SRC)/src/core2b/fake_nd.h $(SRC)/src/core/py_cfg.h $(OUT)/lifecycle.o $(OUT)/core.o $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(WIRE) $(GRAM)
	$(CXX) $(CXXFLAGS) -shared -I$(INC) -I$(SRC)/src/core -I$(SRC)/src/core2b -I$(ZMQ_INC) -isystem $(PYINC) -isystem $(PYBIND) $(XINC) $< $(SRC)/src/core2b/server.cc $(SRC)/src/core2b/native_worker.cc $(SRC)/src/core2b/native_mixed.cc $(SRC)/src/core2b/mixed_exec.cpp $(SRC)/src/core2b/mixed_issue_nd.cpp $(SRC)/src/core2b/span_program.cpp $(SRC)/src/core2b/driver_bridge.cpp $(SRC)/src/core2b/cabi_libs.cpp $(SRC)/src/core2b/mixed_owner.cpp $(SRC)/src/core2b/owner_p1.cpp $(SRC)/src/core2b/owner_engram.cpp $(SRC)/src/p1/gt_build.cpp $(SRC)/src/p1/pf_build.cpp $(SRC)/src/p1/steps_build.cpp $(SRC)/src/p1/cp_build.cpp $(SRC)/src/p0/mixed_build.cpp $(SRC)/src/p0/inputs_build.cpp $(SRC)/src/p0/moe_decide.cpp $(SRC)/src/p0/engram_host.cpp $(SRC)/src/core2b/fake_nd.cc $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(OUT)/lifecycle.o $(OUT)/core.o $(GRAM) $(WIRE) $(XGRLIB) $(ZMQ_LIB) -ldl -Wl,-rpath,$(dir $(ZMQ_LIB)) -o $@.new && mv $@.new $@

$(OUT):
	mkdir -p $(OUT)

$(OUT)/v1wire.o: $(SRC)/src/v1wire/v1wire.cc $(SRC)/src/v1wire/v1wire.h $(SRC)/src/v1wire/msgpack.h | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/pyfloat.o: $(SRC)/src/v1wire/pyfloat.cc | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/grammar.o: $(SRC)/src/grammar/grammar.cc $(SRC)/src/grammar/grammar.h | $(OUT)
	$(CXX) $(CXXFLAGS) $(XINC) -c $< -o $@
$(OUT)/lifecycle.o: $(SRC)/src/core/lifecycle.cc $(SRC)/src/core/lifecycle.h $(SRC)/src/grammar/grammar.h | $(OUT)
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@
$(OUT)/core.o: $(SRC)/src/core/core.cc $(SRC)/include/core.h $(SRC)/src/core/lifecycle.h | $(OUT)
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OUT)/libfprcore.a: $(WIRE) $(GRAM) $(CORE)
	rm -f $@ && ar rcs $@ $^


.PHONY: all pymod core2b
