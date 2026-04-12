# ion7-core top-level Makefile
#
# Usage:
#   make build               Build with vendored llama.cpp (auto-configures + builds it)
#   make build LIB_DIR=PATH  Build using an external llama.cpp installation
#   make test  ION7_MODEL=/path/to/model.gguf
#   make bench ION7_MODEL=/path/to/model.gguf
#   make all   ION7_MODEL=...

VENDOR_LLAMA  := vendor/llama.cpp
LLAMA_BUILD   := $(VENDOR_LLAMA)/build

# By default, use the vendored llama.cpp.
# Override with LIB_DIR to point at an external build.
LIB_DIR       ?= $(abspath $(LLAMA_BUILD)/bin)
COMMON_LIB_DIR ?= $(abspath $(LLAMA_BUILD)/common)

ION7_MODEL    ?=
ION7_EMBED    ?=
ION7_LIB_DIR  ?=
LLAMA_LIB     ?=

# CUDA architecture for the vendor build.
# 86 = Ampere (RTX 30xx)   89 = Ada Lovelace (RTX 40xx)   80 = A100
CUDA_ARCH     ?= 86

# Pass extra cmake flags if needed:
#   make build LLAMA_CMAKE_EXTRA="-DGGML_VULKAN=ON"
LLAMA_CMAKE_EXTRA ?=

# Auto-build the vendor only when LIB_DIR comes from its default value.
# If the user sets LIB_DIR on the command line we skip the vendor build.
ifeq ($(origin LIB_DIR),default)
_BUILD_PREREQS := llama
else
_BUILD_PREREQS :=
endif

.PHONY: build test bench bench-embed compare jitter stability charts clean all help llama

help:
	@echo "ion7-core build targets:"
	@echo "  make llama               Configure + build vendor/llama.cpp"
	@echo "  make build               Build ion7_bridge.so (auto-builds llama.cpp)"
	@echo "  make build LIB_DIR=PATH  Build using an external llama.cpp"
	@echo "  make test      ION7_MODEL=/path/to/model.gguf"
	@echo "  make bench     ION7_MODEL=/path/to/model.gguf  (JSON, 14 sections)"
	@echo "  make compare   ION7_MODEL=/path/to/model.gguf  (Lua vs Python side-by-side)"
	@echo "  make stability ION7_MODEL=/path/to/model.gguf  (memory leak detection)"
	@echo "  make all       ION7_MODEL=..."
	@echo ""
	@echo "Vendor build config:"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)           86=RTX30xx  89=RTX40xx  80=A100"
	@echo "  LLAMA_CMAKE_EXTRA=...       extra cmake flags for llama.cpp"
	@echo ""
	@echo "Extra bench args:  make bench BENCH_ARGS='--n-gpu-layers 20 --stress'"

# ── Vendor build ──────────────────────────────────────────────────────────────
llama:
	@if [ ! -f $(LLAMA_BUILD)/CMakeCache.txt ]; then \
	  echo "[ion7-core] configuring llama.cpp (CUDA_ARCH=$(CUDA_ARCH))..."; \
	  cmake -B $(LLAMA_BUILD) -S $(VENDOR_LLAMA) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DGGML_CUDA=ON \
	    -DBUILD_SHARED_LIBS=ON \
	    -DLLAMA_BUILD_TESTS=OFF \
	    -DLLAMA_BUILD_EXAMPLES=OFF \
	    -DLLAMA_BUILD_SERVER=OFF \
	    -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
	    $(LLAMA_CMAKE_EXTRA); \
	fi
	@echo "[ion7-core] building llama.cpp..."
	@cmake --build $(LLAMA_BUILD) --config Release -j$(shell nproc)

# ── Bridge ────────────────────────────────────────────────────────────────────
build: $(_BUILD_PREREQS)
	$(MAKE) -C bridge \
		LIB_DIR=$(LIB_DIR) \
		COMMON_LIB_DIR=$(COMMON_LIB_DIR)

test: build
	@echo ""
	@ION7_MODEL=$(ION7_MODEL) \
	 ION7_EMBED=$(ION7_EMBED) \
	 bash tests/run_all.sh

bench:
	@ION7_MODEL=$(ION7_MODEL) ION7_LIB_DIR=$(ION7_LIB_DIR) LLAMA_LIB=$(LLAMA_LIB) luajit benchmark/bench_lua.lua $(BENCH_ARGS) | python3 -m json.tool

bench-embed:
	@ION7_EMBED=$(ION7_EMBED) LLAMA_LIB=$(LLAMA_LIB) luajit benchmark/bench_embed.lua

compare:
	@ION7_MODEL=$(ION7_MODEL) ION7_LIB_DIR=$(ION7_LIB_DIR) LLAMA_LIB=$(LLAMA_LIB) bash benchmark/compare.sh $(BENCH_ARGS)
	@$(MAKE) --no-print-directory charts

charts:
	@python3 benchmark/gen_charts.py

jitter:
	@ION7_MODEL=$(ION7_MODEL) ION7_LIB_DIR=$(ION7_LIB_DIR) LLAMA_LIB=$(LLAMA_LIB) bash benchmark/compare_jitter.sh $(BENCH_ARGS)

stability:
	@ION7_MODEL=$(ION7_MODEL) \
	 ION7_LIB_DIR=$(ION7_LIB_DIR) \
	 LLAMA_LIB=$(LLAMA_LIB) \
	 ION7_TOKENS=$(or $(ION7_TOKENS),100000) \
	 luajit benchmark/bench_stability.lua

all: build test bench

clean:
	$(MAKE) -C bridge clean
