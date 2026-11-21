# kvtables_py.mk
#   make -f tests/kvtables_py.mk PY=$DS41H_HOME/venv/bin/python CXX=g++-13 [B=build_py]     (from fpr/core)
# The .so is written as <name>.new and moved over the target (a running process keeps its mapped copy).
PY ?= python3
CXX ?= g++
B ?= build_py
PYINC := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PBINC := $(shell $(PY) -c "import pybind11; print(pybind11.get_include())")
EXT := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SRC := src/py/kvtables_py.cpp src/kv/kv_manager.cpp src/tables/tables.cpp
HDR := include/common.h include/geometry.h include/kv.h include/tables.h include/scheduler.h src/kv/kv_manager.h
$(B)/fpr_kvtables$(EXT): $(SRC) $(HDR)
	@mkdir -p $(B)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -shared -fPIC -fvisibility=hidden -Wall -Wno-unused-parameter \
	  -I$(PYINC) -I$(PBINC) -Iinclude -o $@.new $(SRC) && mv $@.new $@
clean:
	rm -f $(B)/fpr_kvtables*
