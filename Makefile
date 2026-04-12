# ion7-core top-level Makefile
#
# Usage:
#   make build               Build with vendored llama.cpp (auto-configures + builds it)
#   make build LIB_DIR=PATH  Build using an external llama.cpp installation
#   make test  ION7_MODEL=/path/to/model.gguf
#   make docs                Generate HTML docs via ion7-doc (requires ../ion7-doc)

# ── Local overrides ───────────────────────────────────────────────────────────
# Copy local.mk.example → local.mk and set LIB_DIR / COMMON_LIB_DIR to your
# llama.cpp build.  local.mk is gitignored - safe to store absolute paths.
-include local.mk

VENDOR_LLAMA  := vendor/llama.cpp
LLAMA_BUILD   := $(VENDOR_LLAMA)/build

# By default, use the vendored llama.cpp.
# Override with LIB_DIR to point at an external build (or via local.mk).
LIB_DIR        ?= $(abspath $(LLAMA_BUILD)/bin)
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
# If the user sets LIB_DIR on the command line (or via local.mk) we skip the vendor build.
ifeq ($(origin LIB_DIR),default)
_BUILD_PREREQS := llama
else
_BUILD_PREREQS :=
endif

# Path to ion7-doc (sibling repo by default)
ION7_DOC ?= $(abspath ../ion7-doc)

# Output directory for generated docs.
# Override to publish directly to the github.io repo:
#   make docs DOCS_OUT=/path/to/ion7-labs.github.io
DOCS_OUT ?= $(ION7_DOC)/docs

.PHONY: build test docs pages clean all help llama

help:
	@echo "ion7-core build targets:"
	@echo "  make llama               Configure + build vendor/llama.cpp"
	@echo "  make build               Build ion7_bridge.so (auto-builds llama.cpp)"
	@echo "  make build LIB_DIR=PATH  Build using an external llama.cpp"
	@echo "  make test  ION7_MODEL=/path/to/model.gguf"
	@echo "  make docs                Generate HTML docs → $(DOCS_OUT)/"
	@echo "  make docs DOCS_OUT=PATH  Generate docs to a custom directory"
	@echo "  make pages PAGES_DIR=PATH  Generate docs directly to github.io repo"
	@echo ""
	@echo "Vendor build config:"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)  86=RTX30xx  89=RTX40xx  80=A100"
	@echo "  LLAMA_CMAKE_EXTRA=...   extra cmake flags for llama.cpp"
	@echo ""
	@echo "Tip: copy local.mk.example → local.mk to persist your LIB_DIR."

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

# ── Documentation ─────────────────────────────────────────────────────────────
docs:
	@test -f $(ION7_DOC)/bin/gendoc.lua || \
	  (echo "[ion7-core] ion7-doc not found at $(ION7_DOC)" && exit 1)
	@mkdir -p $(DOCS_OUT)
	@luajit $(ION7_DOC)/bin/gendoc.lua \
	  $(abspath src/ion7/core) \
	  $(DOCS_OUT) \
	  $(abspath README.md)

pages:
	@test -d "$(PAGES_DIR)" || \
	  (echo "[ion7-core] PAGES_DIR not set or not found: $(PAGES_DIR)" && exit 1)
	@$(MAKE) docs DOCS_OUT=$(PAGES_DIR)

all: build test

clean:
	$(MAKE) -C bridge clean
