# Copyright (C) 2026 Ion7 Project Contributors
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


#
# Usage:
#   make LIB_DIR=/path/to/llama.cpp/build/bin
#   make LIB_DIR=... COMMON_LIB_DIR=/path/to/llama.cpp/build/common
#
# Bridge v1.1 - C++ build with libcommon integration.
# Requires: libllama.so (LIB_DIR) + libcommon.a (COMMON_LIB_DIR)
#
# Source layout (split from monolithic ion7_bridge.cpp):
#   bridge_core.cpp     - llama.h layer (model, context, KV, state, samplers)
#   bridge_common.cpp   - libcommon layer (templates, csampler, speculative, chat parse)
#   bridge_training.cpp - llama_opt training pipeline
#   bridge_utils.cpp    - standalone utilities (utf8, json, regex, cvec, numa, ...)

CXX        ?= g++
# Default to the vendored llama.cpp build (relative from bridge/).
# Overridden by the top-level Makefile with an absolute path.
LIB_DIR    ?= $(abspath ../vendor/llama.cpp/build/bin)
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

# Optional: set MARCH=native (or any -march target) for CPU-specific tuning.
# Example: make LIB_DIR=... MARCH=native
MARCH ?=

CXXFLAGS := -O2 -fPIC -std=c++17 -Wall -Wextra -flto=auto \
            $(if $(MARCH),-march=$(MARCH),) \
            -isystem $(LLAMA_INC) -isystem $(GGML_INC) \
            -isystem $(COMMON_INC) -isystem $(VENDOR_INC)

# Link flags split into two groups so that object files are scanned BEFORE
# the static archive.  GNU ld resolves symbols from archives lazily (left to
# right), so BRIDGE_OBJS must appear between LDFLAGSF and LDFLAGSL:
#   $(CXX) $(LDFLAGSF) -o $(BRIDGE_SO) $(BRIDGE_OBJS) $(LDFLAGSL)
#
# On macOS, -undefined dynamic_lookup defers resolution of ggml symbols to
# load time (they arrive transitively via libllama.dylib at runtime).
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    LDFLAGSF := -shared -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR) \
                -Wl,-undefined,dynamic_lookup
    LDFLAGSL := -lllama \
                $(COMMON_LIB_DIR)/libcommon.a
else
    LDFLAGSF := -shared -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR)
    LDFLAGSL := -lllama \
                $(COMMON_LIB_DIR)/libcommon.a
endif

all: $(BRIDGE_SO)

BRIDGE_OBJS := $(BRIDGE_SRCS:.cpp=.o)

# Compile each translation unit separately for faster incremental rebuilds
%.o: %.cpp ion7_bridge.h bridge_internal.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BRIDGE_SO): $(BRIDGE_OBJS)
	@echo "[ion7-core] linking bridge (C++17 + libcommon)..."
	$(CXX) $(LDFLAGSF) -o $(BRIDGE_SO) $(BRIDGE_OBJS) $(LDFLAGSL)
	@echo "[ion7-core] bridge ready → $(BRIDGE_SO)"

clean:
	rm -f $(BRIDGE_SO) $(BRIDGE_OBJS)

.PHONY: all clean
