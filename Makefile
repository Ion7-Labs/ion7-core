# ion7-core top-level Makefile
#
# Usage:
#   make build LIB_DIR=/path/to/llama.cpp/build/bin
#   make test  ION7_MODEL=/path/to/model.gguf
#   make bench ION7_MODEL=/path/to/model.gguf
#   make all   LIB_DIR=... ION7_MODEL=...

LIB_DIR      ?= /usr/local/lib
ION7_MODEL   ?=
ION7_EMBED   ?=
ION7_LIB_DIR ?=
LLAMA_LIB    ?=


.PHONY: build test bench bench-embed compare jitter stability clean all help

help:
	@echo "ion7-core build targets:"
	@echo "  make build     LIB_DIR=/path/to/llama.cpp/build/bin"
	@echo "  make test      ION7_MODEL=/path/to/model.gguf"
	@echo "  make bench     ION7_MODEL=/path/to/model.gguf  (JSON, 14 sections)"
	@echo "  make compare   ION7_MODEL=/path/to/model.gguf  (Lua vs Python side-by-side)"
	@echo "  make stability ION7_MODEL=/path/to/model.gguf  (memory leak detection)"
	@echo "  make all       LIB_DIR=... ION7_MODEL=..."
	@echo ""
	@echo "Extra args via BENCH_ARGS: make bench BENCH_ARGS='--n-gpu-layers 20 --stress'"

build:
	$(MAKE) -C bridge \
		LIB_DIR=$(LIB_DIR)

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
