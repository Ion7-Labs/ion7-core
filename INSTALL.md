# ion7-core - Installation & Build Guide

This document covers everything needed to build ion7-core and its dependencies from scratch.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Build llama.cpp](#2-build-llamacpp)
3. [Build the ion7-core bridge](#3-build-the-ion7-core-bridge)
4. [Library path configuration](#4-library-path-configuration)
5. [LuaJIT setup](#5-luajit-setup)
6. [Running tests and benchmarks](#6-running-tests-and-benchmarks)
7. [Build flags reference](#7-build-flags-reference)
8. [Platform-specific notes](#8-platform-specific-notes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| GCC or Clang | GCC 11 / Clang 14 | **C++17** support required (bridge is C++17) |
| CMake | 3.21 | For building llama.cpp |
| LuaJIT | 2.1 | Must be 2.1 - LuaJIT 2.0 lacks required FFI features |
| Git | Any | For cloning |
| CUDA Toolkit | 12.0+ | Optional, for NVIDIA GPU offload |

### Linux (Fedora/RHEL)

```bash
sudo dnf install gcc gcc-c++ cmake git luajit luajit-devel
# For CUDA:
sudo dnf install cuda-toolkit
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install gcc g++ cmake git luajit libluajit-5.1-dev
# For CUDA:
# Follow https://developer.nvidia.com/cuda-downloads for your distro
```

### macOS

```bash
brew install cmake luajit
# Xcode command line tools provide clang:
xcode-select --install
```

---

## 2. Build llama.cpp

ion7-core links against `libllama.so` (Linux) or `libllama.dylib` (macOS). You need to build llama.cpp as a shared library.

### Clone llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### CPU-only build

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j$(nproc)
```

### CUDA build (NVIDIA GPU)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j$(nproc)
```

### ROCm build (AMD GPU)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_HIPBLAS=ON \
    -DLLAMA_BUILD_TESTS=OFF

cmake --build build --config Release -j$(nproc)
```

### Metal build (Apple Silicon)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_METAL=ON

cmake --build build --config Release -j$(nproc)
```

### Vulkan build

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_VULKAN=ON

cmake --build build --config Release -j$(nproc)
```

After a successful build, the libraries are in `llama.cpp/build/bin/`:

```
llama.cpp/build/bin/
├── libllama.so        ← main library
├── libggml.so
├── libggml-cpu.so
└── libggml-cuda.so    ← if CUDA was enabled
```

> **Note:** The `build/bin` path is used as `LIB_DIR` throughout this document. Some CMake versions may place libraries in `build/src/` or `build/` - adjust accordingly.

---

## 3. Build the ion7-core bridge

```bash
git clone https://github.com/ion7-labs/ion7-core
cd ion7-core

make build LIB_DIR=/path/to/llama.cpp/build/bin
```

This compiles `bridge/ion7_bridge.so` from four focused translation units (`bridge_core.cpp`, `bridge_common.cpp`, `bridge_training.cpp`, `bridge_utils.cpp`) and links against `libllama.so` and `libcommon.a`.

`libcommon.a` is expected at `$(LIB_DIR)/../common/libcommon.a` (the default CMake output location). Override with `COMMON_LIB_DIR` if yours is elsewhere:

```bash
make build LIB_DIR=/path/to/llama.cpp/build/bin \
           COMMON_LIB_DIR=/path/to/llama.cpp/build/common
```

If your llama.cpp source and build trees are in different directories (e.g. CI):

```bash
make build LIB_DIR=/tmp/llama-build/bin \
           LLAMA_SRC=/tmp/llama-src
```

### Verify the build

```bash
ls -lh bridge/ion7_bridge.so
# Should output something like: -rwxr-xr-x 1 user user 400K Apr 2026 bridge/ion7_bridge.so
```

---

## 4. Library path configuration

ion7-core needs to find `libllama.so` and `ion7_bridge.so` at runtime. There are three ways to configure this.

### Option A - Environment variables (recommended for development)

```bash
export LLAMA_LIB=/path/to/llama.cpp/build/bin/libllama.so
export ION7_BRIDGE=/path/to/ion7-core/bridge/ion7_bridge.so
```

Add these to your `.bashrc` or `.zshrc` for persistence.

### Option B - Pass paths to ion7.init()

```lua
local ion7 = require "ion7.core"
ion7.init({
    llama_path  = "/path/to/llama.cpp/build/bin/libllama.so",
    bridge_path = "/path/to/ion7-core/bridge/ion7_bridge.so",
})
```

### Option C - System install

Install libllama.so to a system path and the loader will find it automatically:

```bash
# From your llama.cpp build directory:
sudo cp build/bin/libllama.so /usr/local/lib/
sudo cp build/bin/libggml*.so /usr/local/lib/
sudo ldconfig

# ion7.init() will find libllama.so automatically
```

For `ion7_bridge.so`, you still need to set `ION7_BRIDGE` or pass `bridge_path` - it is project-specific and not meant to be installed system-wide.

### Runtime library loading (RPATH)

The bridge is compiled with `-Wl,-rpath,$(LIB_DIR)` pointing to your llama.cpp build directory. If you move the llama.cpp build, recompile the bridge with the new `LIB_DIR`.

---

## 5. LuaJIT setup

### Verify your LuaJIT version

```bash
luajit -v
# Should output: LuaJIT 2.1.0-beta3 or later
```

### Add ion7-core to your Lua path

```bash
# In your shell config:
export LUA_PATH="/path/to/ion7-core/src/?.lua;/path/to/ion7-core/src/?/init.lua;$LUA_PATH"
```

Or in your Lua code before the first require:

```lua
package.path = "/path/to/ion7-core/src/?.lua;" ..
               "/path/to/ion7-core/src/?/init.lua;" ..
               package.path
```

### Quick smoke test

```bash
luajit -e "
local ion7 = require 'ion7.core'
ion7.init({ log_level = 0 })
local caps = ion7.capabilities()
print('bridge:', caps.bridge_ver)
print('gpu offload:', caps.gpu_offload)
ion7.shutdown()
"
```

---

## 6. Running tests and benchmarks

### Tests

```bash
make test ION7_MODEL=/path/to/model.gguf

# With an embedding model too:
make test ION7_MODEL=/path/to/chat.gguf ION7_EMBED=/path/to/embed.gguf
```

Tests are LuaJIT scripts using the built-in `tests/framework.lua`. Each test prints `[OK]`, `[FAIL]` (with reason), or `[SKIP]`. The runner `tests/run_all.sh` orchestrates both suites and exits non-zero on any failure.

### Benchmarks

```bash
# Lua benchmark - 14 sections, JSON output
make bench ION7_MODEL=/path/to/model.gguf BENCH_ARGS='--n-gpu-layers 32'

# Lua vs Python side-by-side comparison
make compare ION7_MODEL=/path/to/model.gguf BENCH_ARGS='--n-gpu-layers 32'

# Stability (timing variance over N iterations)
make jitter ION7_MODEL=/path/to/model.gguf BENCH_ARGS='--n-gpu-layers 32 --n-iter 500'

# Embedding benchmark
make bench-embed ION7_EMBED=/path/to/embed.gguf

# Long-running memory stability test (default: 100k tokens)
make stability ION7_MODEL=/path/to/model.gguf
make stability ION7_MODEL=/path/to/model.gguf ION7_TOKENS=500000
```

`make bench` outputs JSON. Pipe through `jq` to pretty-print or extract specific metrics:

```bash
make bench ION7_MODEL=... | jq '.benchmarks.generation'
```

---

## 7. Platform-specific notes

### Linux + NVIDIA (CUDA)

After building llama.cpp with `-DGGML_CUDA=ON`, make sure the CUDA runtime is on your library path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

If `libggml-cuda.so` is not found at runtime:

```bash
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

### Linux + AMD (ROCm)

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### macOS (Apple Silicon)

On macOS, llama.cpp produces `.dylib` files. The loader checks for `/usr/local/lib/libllama.dylib` automatically. Set `LLAMA_LIB` to the full `.dylib` path if it is elsewhere.

Metal is enabled by default on Apple Silicon builds - no extra configuration needed.

### Fedora / SELinux

If you get `Permission denied` loading the `.so` files, SELinux may be blocking them:

```bash
# Allow the shared libraries (development only):
sudo chcon -t lib_t bridge/ion7_bridge.so
sudo chcon -t lib_t /path/to/llama.cpp/build/bin/libllama.so
```

Or build with the system install path where SELinux contexts are already set.

---

## 8. Troubleshooting

### `[ion7-core] cannot load libllama.so`

The loader cannot find `libllama.so`. Set the `LLAMA_LIB` environment variable to the full path of the file, or pass `llama_path` to `ion7.init()`.

### `[ion7-core] cannot load ion7_bridge.so`

The bridge `.so` was not found. Set `ION7_BRIDGE` to the full path of `bridge/ion7_bridge.so`, or pass `bridge_path` to `ion7.init()`.

### `undefined symbol: llama_chat_apply_template` or wrong parameter count

Signature mismatch between bridge and your llama.cpp version. Try:
```bash
make build LIB_DIR=... ION7_CHAT_TMPL_NO_MODEL=1
```

### LuaJIT segfault inside ffi.cast callback

Lua errors must not propagate across FFI callback boundaries in LuaJIT. If you write a `CustomSampler`, all logic inside `apply()`, `accept()`, and `reset()` must be wrapped in `pcall`. The bridge wraps them for you - but errors thrown from within a `pcall`'d function are silently discarded. Check your sampler logic for Lua errors first.

### `KV cache is full`

Increase `n_ctx` when creating the context, or call `ctx:kv_clear()` between conversations. If you are in a multi-sequence scenario, use `ctx:kv_seq_rm()` to free individual sequences.

### Tests fail with `model pointer is NULL`

The model failed to load. Common causes:
- `ION7_MODEL` points to a file that doesn't exist or isn't a valid GGUF
- Not enough RAM/VRAM to load the model
- `n_gpu_layers` is set too high for your GPU's VRAM

Try loading with `n_gpu_layers = 0` first to rule out VRAM issues.

---

## Getting a GGUF model

If you don't have a GGUF model yet:

```bash
# Using llama.cpp's built-in downloader:
/path/to/llama.cpp/build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF --list-models

# Or download directly from Hugging Face:
# https://huggingface.co/models?library=gguf&sort=trending
```

Recommended models to start with:
- `Qwen2.5-7B-Instruct-Q4_K_M` - good balance of speed and quality
- `Qwen2.5-1.5B-Instruct-Q8_0` - fast, fits in 4GB VRAM
- `Llama-3.2-3B-Instruct-Q4_K_M` - solid general purpose, small footprint
