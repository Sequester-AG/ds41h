# fpr/core/tests/sched.mk (make -f sched.mk) -- torch-free build of the stage-2 golden-trace replays. scheduler.
# Build rule for every component: -O2 -std=c++17 -ffp-contract=off (float parity with Python, no FMA contraction).
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
INC = -I../include -I../src/scheduler
BUILD ?= build

SCHED_SRC = ../src/scheduler/rank_scheduler.cpp
SCHED_HDR = ../include/common.h ../include/kv.h ../include/geometry.h ../include/cost_model.h ../include/cost_params.h \
            ../include/sched_config.h ../include/scheduler.h ../src/scheduler/rank_scheduler.h

clean:
	rm -rf $(BUILD)

.PHONY: clean

PY ?= python3
PYINC ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND ?= $(shell $(PY) -c "import pybind11; print(pybind11.get_include())")
PYEXT ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYLDFLAGS ?= -shared

pymod: $(BUILD)/fpr_sched_py$(PYEXT)

$(BUILD)/fpr_sched_py$(PYEXT): ../src/scheduler/py_sched.cc ../src/kv/kv_manager.cpp ../src/kv/kv_manager.h $(SCHED_SRC) $(SCHED_HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC $(PYLDFLAGS) $(INC) -isystem $(PYINC) -isystem $(PYBIND) -o $@ ../src/scheduler/py_sched.cc $(SCHED_SRC) ../src/kv/kv_manager.cpp

.PHONY: pymod
