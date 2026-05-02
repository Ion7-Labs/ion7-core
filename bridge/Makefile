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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

# ------------------------------------------------------------------------------
# ion7-core bridge v2 - libcommon-only build.
#
# Direct llama.cpp / ggml symbols are no longer wrapped by the bridge —
# they are consumed by Lua via the auto-generated FFI bindings under
# `src/ion7/core/ffi/`. The bridge now exists ONLY to provide a flat C
# ABI over the C++ helpers in `llama.cpp/common/` (chat templates,
# common_sampler, common_speculative, json_schema_to_grammar, regex,
# common_fit_params, common_opt_dataset_init).
#
# Source layout:
#   bridge_chat.cpp     - common_chat_templates / parse / reasoning_budget
#   bridge_sampling.cpp - common_sampler + common_speculative
#   bridge_text.cpp     - regex_partial + json_schema_to_grammar + json utils
#   bridge_training.cpp - llama_opt + common_opt_dataset_init
#   bridge_fit.cpp      - common_fit_params (VRAM auto-fit) + version
#
# Usage:
#   make LIB_DIR=/path/to/llama.cpp/build/bin
#   make LIB_DIR=... COMMON_LIB_DIR=/path/to/llama.cpp/build/common
#
# Required dependencies (resolved at link time):
#   - libllama         (-L LIB_DIR -lllama)
#   - libcommon.a      (COMMON_LIB_DIR/libcommon.a)
#
# llama.cpp source layout assumed:
#   <LLAMA_ROOT>/include/         -> public llama.h (extern "C")
#   <LLAMA_ROOT>/ggml/include/    -> public ggml*.h (extern "C")
#   <LLAMA_ROOT>/common/          -> libcommon C++ headers
#   <LLAMA_ROOT>/vendor/          -> nlohmann/json.hpp + base64.hpp
# ------------------------------------------------------------------------------

CXX        ?= g++

# Default to the vendored llama.cpp build relative to bridge/.
# The top-level Makefile may override LIB_DIR with an absolute path.
LIB_DIR    ?= $(abspath ../vendor/llama.cpp/build/bin)
BRIDGE_SO  := ion7_bridge.so

BRIDGE_SRCS := bridge_chat.cpp \
               bridge_sampling.cpp \
               bridge_text.cpp \
               bridge_training.cpp \
               bridge_fit.cpp

# Include path resolution.
#
# Default: infer from LIB_DIR — works when source and build share the
# same tree, e.g. LIB_DIR=/path/to/llama.cpp/build/bin → root is ../../
#
# Override: set LLAMA_SRC when the source and build dirs are separate
# (CI: make LIB_DIR=/tmp/llama-build/bin LLAMA_SRC=/tmp/llama-src).
LLAMA_SRC  ?=
LLAMA_ROOT := $(if $(LLAMA_SRC),$(abspath $(LLAMA_SRC)),$(abspath $(LIB_DIR)/../..))
LLAMA_INC  := $(LLAMA_ROOT)/include
GGML_INC   := $(LLAMA_ROOT)/ggml/include
COMMON_INC := $(LLAMA_ROOT)/common
VENDOR_INC := $(LLAMA_ROOT)/vendor

# libcommon used to ship as `libcommon.a` next to `<build>/bin/`. Recent
# llama.cpp builds it as a SHARED library `libllama-common.so` placed
# directly in `<build>/bin/`. We prefer the shared variant when present
# (linked via `-lllama-common`) and fall back to the static archive
# otherwise. Override `COMMON_LIB_DIR` if your layout differs.
COMMON_LIB_DIR ?= $(abspath $(LIB_DIR)/../common)
COMMON_SHARED  := $(wildcard $(LIB_DIR)/libllama-common.so $(LIB_DIR)/libllama-common.dylib)
COMMON_STATIC  := $(wildcard $(COMMON_LIB_DIR)/libcommon.a)

# Optional: pass MARCH=native (or any -march target) for CPU tuning.
MARCH ?=

CXXFLAGS := -O2 -fPIC -std=c++17 -Wall -Wextra -flto=auto \
            $(if $(MARCH),-march=$(MARCH),) \
            -isystem $(LLAMA_INC) -isystem $(GGML_INC) \
            -isystem $(COMMON_INC) -isystem $(VENDOR_INC)

# Link order matters : GNU ld scans archives left-to-right, so the
# bridge object files must appear BEFORE libcommon.a so the latter can
# resolve their references.
#
# On macOS, -undefined dynamic_lookup defers resolution of ggml symbols
# to load time ; they arrive transitively via libllama.dylib at runtime.
# `LIBCOMMON_LINK` resolves to `-lllama-common` when libcommon is built
# as a shared library, or to the absolute path of `libcommon.a` when
# only the static archive is on disk.
ifneq ($(COMMON_SHARED),)
    LIBCOMMON_LINK := -lllama-common
else ifneq ($(COMMON_STATIC),)
    LIBCOMMON_LINK := $(COMMON_STATIC)
else
    $(error Cannot find libllama-common.so in $(LIB_DIR) or libcommon.a in $(COMMON_LIB_DIR). Build llama.cpp with -DBUILD_SHARED_LIBS=ON first.)
endif

UNAME := $(shell uname -s)

# RPATH selection :
#   ION7_RELEASE=1 makes the .so load its sibling libllama from the
#                  same directory it lives in (`$ORIGIN` on Linux,
#                  `@loader_path` on macOS). Use this for distributable
#                  tarballs where every consumer extracts the bundle
#                  to a different directory.
#   default        bakes the build-time LIB_DIR as the rpath. Best for
#                  local development : the dev rebuilds llama.cpp in
#                  place and the bridge picks it up without any env
#                  fiddling.
ifdef ION7_RELEASE
    ifeq ($(UNAME), Darwin)
        RPATH := @loader_path
    else
        # `$$` produces a literal `$` after `make` expansion. Single-
        # quoting the flag in the recipe below stops the shell from
        # interpreting `$ORIGIN` as one of its own variables.
        RPATH := $$ORIGIN
    endif
else
    RPATH := $(LIB_DIR)
endif

ifeq ($(UNAME), Darwin)
    LDFLAGSF := -shared -L$(LIB_DIR) '-Wl,-rpath,$(RPATH)' \
                -Wl,-undefined,dynamic_lookup
    LDFLAGSL := -lllama $(LIBCOMMON_LINK)
else
    LDFLAGSF := -shared -L$(LIB_DIR) '-Wl,-rpath,$(RPATH)'
    LDFLAGSL := -lllama $(LIBCOMMON_LINK)
endif

all: $(BRIDGE_SO)

BRIDGE_OBJS := $(BRIDGE_SRCS:.cpp=.o)

# Compile each translation unit separately for faster incremental builds.
%.o: %.cpp ion7_bridge.h bridge_internal.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BRIDGE_SO): $(BRIDGE_OBJS)
	@echo "[ion7-core] linking bridge v2.0 (libcommon-only)..."
	$(CXX) $(LDFLAGSF) -o $(BRIDGE_SO) $(BRIDGE_OBJS) $(LDFLAGSL)
	@echo "[ion7-core] bridge ready -> $(BRIDGE_SO)"

clean:
	rm -f $(BRIDGE_SO) $(BRIDGE_OBJS)

.PHONY: all clean
