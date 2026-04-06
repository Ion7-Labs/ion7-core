# Copyright (C) 2026 Ion7 Project Contributors
# SPDX-License-Identifier: MIT
#
# This file is part of ion7-core.
#
# ion7-core is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ion7-core is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ion7-core. If not, see <https://www.gnu.org/licenses/>.


#
# Usage:
#   make LIB_DIR=/path/to/llama.cpp/build/bin
#   make LIB_DIR=... COMMON_LIB_DIR=/path/to/llama.cpp/build/common
#
# Bridge v2.0 — C++ build with libcommon integration.
# Requires: libllama.so (LIB_DIR) + libcommon.a (COMMON_LIB_DIR)
#
# Source layout (split from monolithic ion7_bridge.cpp):
#   bridge_core.cpp     — llama.h layer (model, context, KV, state, samplers)
#   bridge_common.cpp   — libcommon layer (templates, csampler, speculative, chat parse)
#   bridge_training.cpp — llama_opt training pipeline
#   bridge_utils.cpp    — standalone utilities (utf8, json, regex, cvec, numa, ...)

CXX        ?= g++
LIB_DIR    ?= /usr/local/lib
BRIDGE_SO  := ion7_bridge.so
BRIDGE_SRCS := bridge_core.cpp bridge_common.cpp bridge_training.cpp bridge_utils.cpp

# Include path resolution.
#
# Default: infer from LIB_DIR - works when source and build share the same tree
#   e.g. LIB_DIR=/path/to/llama.cpp/build/bin → root is ../../
#
# Override: set LLAMA_SRC when source and build directories are separate
#   e.g. CI: make LIB_DIR=/tmp/llama-build/bin LLAMA_SRC=/tmp/llama-src
LLAMA_SRC  ?=
LLAMA_ROOT := $(if $(LLAMA_SRC),$(abspath $(LLAMA_SRC)),$(abspath $(LIB_DIR)/../..))
LLAMA_INC  := $(LLAMA_ROOT)/include
GGML_INC   := $(LLAMA_ROOT)/ggml/include
COMMON_INC := $(LLAMA_ROOT)/common

# libcommon.a is typically in build/common/ next to build/bin/
# Override with COMMON_LIB_DIR if it lives elsewhere.
COMMON_LIB_DIR ?= $(abspath $(LIB_DIR)/../common)

VENDOR_INC := $(LLAMA_ROOT)/vendor

CXXFLAGS := -O2 -fPIC -std=c++17 -Wall -Wextra \
            -isystem $(LLAMA_INC) -isystem $(GGML_INC) \
            -isystem $(COMMON_INC) -isystem $(VENDOR_INC)

# On macOS, libllama.dylib does not re-export ggml symbols (strict two-level
# namespace). The bridge uses ggml_threadpool_* which live in libggml.dylib.
# Since the Lua loader loads libllama first (pulling in libggml transitively),
# those symbols are already in the process by the time the bridge is dlopen'd.
# -undefined dynamic_lookup defers symbol resolution to load time — the
# idiomatic macOS solution for plugins/bridges loaded into a host process.
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    LDFLAGS := -shared -L$(LIB_DIR) -lllama \
               $(COMMON_LIB_DIR)/libcommon.a \
               -Wl,-rpath,$(LIB_DIR) \
               -Wl,-undefined,dynamic_lookup
else
    LDFLAGS := -shared -L$(LIB_DIR) -lllama \
               $(COMMON_LIB_DIR)/libcommon.a \
               -Wl,-rpath,$(LIB_DIR)
endif

all: $(BRIDGE_SO)

BRIDGE_OBJS := $(BRIDGE_SRCS:.cpp=.o)

# Compile each translation unit separately for faster incremental rebuilds
%.o: %.cpp ion7_bridge.h bridge_internal.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BRIDGE_SO): $(BRIDGE_OBJS)
	@echo "[ion7-core] linking bridge (C++17 + libcommon)..."
	$(CXX) $(LDFLAGS) -o $(BRIDGE_SO) $(BRIDGE_OBJS)
	@echo "[ion7-core] bridge ready → $(BRIDGE_SO)"

clean:
	rm -f $(BRIDGE_SO) $(BRIDGE_OBJS)

.PHONY: all clean
