# Installing ion7-core

Three ways to get a working install, in order of effort :

1. **[Use a release tarball](#1--use-a-release-tarball)** — zero compile time, drop-in for CPU / Vulkan / Metal targets.
2. **[Build from source with the vendored `llama.cpp`](#2--build-from-source)** — single `make build`, picks the right backend from your flags.
3. **[Build against an external `llama.cpp`](#3--build-against-an-external-llamacpp)** — re-use a build you already have.

The remainder covers backend selection, runtime configuration, the test/example harness and platform-specific quirks.

---

## 1 — Use a release tarball

Each release ships a per-platform bundle on the [Releases page](https://github.com/Ion7-Labs/ion7-core/releases).
A tarball contains :

```
ion7-core-<version>-<target>/
├── lib/
│   ├── ion7_bridge.so          # rpath = $ORIGIN, finds its siblings
│   ├── libllama.so.<N>         # llama.cpp + ggml + libcommon
│   ├── libggml*.so.<N>
│   └── libllama-common.so.<N>
├── src/ion7/                   # the Lua runtime
├── bin/
│   └── ion7-load.lua           # preamble : sets package.path + env vars
├── README.md   INSTALL.md   LICENSE
```

Extract anywhere and call the preamble before your first `require` :

```bash
curl -L -o ion7.tgz \
  https://github.com/Ion7-Labs/ion7-core/releases/latest/download/ion7-core-linux-x86_64-cpu.tar.gz
tar xzf ion7.tgz -C /opt
```

```lua
dofile("/opt/ion7-core-<version>-linux-x86_64-cpu/bin/ion7-load.lua")
local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })
```

The preamble wires three things :

- `package.path` so `require "ion7.core"` resolves to the tarballed copy ;
- `ION7_LIBLLAMA_PATH` / `ION7_LIBGGML_PATH` so the FFI loader picks the bundled `libllama` / `libggml` ;
- `ION7_BRIDGE_PATH` so the bridge module loads `ion7_bridge.so` from the same `lib/` dir.

**Available targets**

| Target                       | Build flags                | GPU usable on |
|------------------------------|----------------------------|---------------|
| `linux-x86_64-cpu`           | AVX2                       | none          |
| `linux-x86_64-vulkan`        | `-DGGML_VULKAN=ON`         | NVIDIA · AMD · Intel · MoltenVK |
| `linux-aarch64-cpu`          | NEON                       | none          |
| `macos-arm64-metal`          | `-DGGML_METAL=ON`          | Apple Silicon |
| `macos-x86_64-cpu`           | AVX2                       | none          |

CUDA and ROCm tarballs are intentionally not shipped — they require a driver/toolkit version that matches the user's machine. Build locally instead (see §2).
Windows binaries are not yet shipped ; the `Makefile` targets POSIX shell. A CMake-based Windows build is on the roadmap.

---

## 2 — Build from source

Prerequisites : a C++17-capable host compiler, CMake 3.21+, LuaJIT 2.1, and Git (for the submodule).

```bash
# Fedora / RHEL
sudo dnf install -y gcc gcc-c++ cmake git luajit luajit-devel

# Debian / Ubuntu
sudo apt install -y gcc g++ cmake git luajit libluajit-5.1-dev

# macOS
brew install cmake luajit
xcode-select --install
```

Clone with the vendored `llama.cpp` :

```bash
git clone --recurse-submodules https://github.com/Ion7-Labs/ion7-core
cd ion7-core
```

Then either let the top-level `Makefile` build everything or hand it your own `llama.cpp` build directory.

### Auto-build (vendored `llama.cpp`)

The vendor build is triggered automatically when `LIB_DIR` is left at its default :

```bash
make build                                # CPU only
make build LLAMA_CMAKE_EXTRA="-DGGML_CUDA=ON"     CUDA_ARCH=86   # NVIDIA Ampere
make build LLAMA_CMAKE_EXTRA="-DGGML_VULKAN=ON"                   # any Vulkan GPU
make build LLAMA_CMAKE_EXTRA="-DGGML_HIPBLAS=ON"                  # AMD ROCm
make build LLAMA_CMAKE_EXTRA="-DGGML_METAL=ON"                    # Apple Silicon
```

`CUDA_ARCH` accepts the same values as `CMAKE_CUDA_ARCHITECTURES` (`80` = A100, `86` = RTX 30xx, `89` = RTX 40xx, `90` = H100, `120` = RTX 50xx).

If your host GCC is too new for the bundled CUDA toolkit, point `nvcc` at `clang++` :

```bash
make build LLAMA_CMAKE_EXTRA="-DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++"
```

### Persisting your knobs in `local.mk`

Copy the example file (gitignored) and fill in the bits that don't belong in version control :

```bash
cp local.mk.example local.mk
$EDITOR local.mk
```

Two recommended forms — pick one :

```make
# Anchored on $(HOME) :
LIB_DIR        = $(HOME)/llama.cpp/build/bin
COMMON_LIB_DIR = $(HOME)/llama.cpp/build/common

# Anchored on an env var :
LIB_DIR        = $(LLAMA_CPP_DIR)/build/bin
COMMON_LIB_DIR = $(LLAMA_CPP_DIR)/build/common
```

---

## 3 — Build against an external `llama.cpp`

If you already have a `llama.cpp` build you want to re-use, override `LIB_DIR` (and optionally `COMMON_LIB_DIR`) :

```bash
make build \
    LIB_DIR=/path/to/llama.cpp/build/bin \
    COMMON_LIB_DIR=/path/to/llama.cpp/build/common
```

When the source and build trees live in different places (typical in CI), pass `LLAMA_SRC` so the bridge picks up the right headers :

```bash
make build \
    LIB_DIR=/tmp/llama-build/bin \
    LLAMA_SRC=/tmp/llama-src
```

### Building a release-mode `ion7_bridge.so`

A locally-built bridge bakes the build-time `LIB_DIR` into its rpath, which is right for development but breaks the moment you copy the file elsewhere. For a redistributable build (relocatable rpath) :

```bash
ION7_RELEASE=1 make build
```

The resulting `ion7_bridge.so` sets its rpath to `$ORIGIN` (Linux) / `@loader_path` (macOS), so it loads its siblings (`libllama`, `libggml`) from whatever directory it ends up in. This is exactly what `make tarball` and the CI release workflow do.

---

## Runtime configuration

`require "ion7.core"` only needs to find three `.so` files at runtime — `libllama`, `libggml`, and `ion7_bridge.so`. The FFI loader probes a list of locations in this order :

| Order | Source |
|------:|--------|
| 1     | env var `ION7_LIBLLAMA_PATH` / `ION7_LIBGGML_PATH` / `ION7_BRIDGE_PATH` |
| 2     | `vendor/llama.cpp/build/bin/lib*.so` (relative to CWD) |
| 3     | `vendor/llama.cpp/build/lib/lib*.so` |
| 4     | `/usr/local/lib/lib*.so` |
| 5     | `lib*.so` (system loader path : `LD_LIBRARY_PATH`, `ldconfig`) |

For a tarball install, the supplied `bin/ion7-load.lua` sets the env vars from §1 — call it once before your first `require` and you're done.

For a system-wide install :

```bash
sudo cp vendor/llama.cpp/build/bin/lib*.so /usr/local/lib/
sudo cp bridge/ion7_bridge.so              /usr/local/lib/
sudo ldconfig
```

For a per-shell setup :

```bash
export ION7_LIBLLAMA_PATH=/path/to/llama.cpp/build/bin/libllama.so
export ION7_LIBGGML_PATH=/path/to/llama.cpp/build/bin/libggml.so
export ION7_BRIDGE_PATH=/path/to/ion7-core/bridge/ion7_bridge.so
```

---

## LuaJIT path

`require "ion7.core"` finds the runtime through `package.path`. From the project root the test / example scripts prepend it themselves ; for arbitrary scripts pick whichever fits :

```bash
# In your shell rc :
export LUA_PATH="/path/to/ion7-core/src/?.lua;/path/to/ion7-core/src/?/init.lua;$LUA_PATH"

# Or inside the script, before the first require :
package.path = "/path/to/ion7-core/src/?.lua;" ..
               "/path/to/ion7-core/src/?/init.lua;" .. package.path
```

---

## Verification

Once everything is in place :

```bash
luajit -e '
local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })
local caps = ion7.capabilities()
print("bridge :",      caps.bridge_ver)
print("gpu offload :", caps.gpu_offload)
print("info :",        caps.llama_info:sub(1, 60))
ion7.shutdown()
'
```

Or run the full test suite :

```bash
make test ION7_MODEL=/path/to/your-model.gguf
```

The runner is `tests/run_all.sh` — it picks up every `tests/[0-9][0-9]_*.lua` and reports `[OK]`, `[SKIP]` or `[FAIL]` per test. Files starting with `0` need no model ; `1` needs `ION7_MODEL` ; `2` may need extras (`ION7_DRAFT`, `ION7_EMBED`, `ION7_TRAIN`).

---

## Examples

Nine standalone scripts under [`examples/`](examples/) cover the canonical patterns — minimal pipeline, multi-turn chat with KV-delta, streaming, JSON Schema, embeddings, KV reuse, custom samplers, threadpools, speculative decoding. Each one runs from the project root with no extra setup beyond `ION7_MODEL` :

```bash
ION7_MODEL=/path/to/model.gguf luajit examples/01_hello.lua
```

See [`examples/README.md`](examples/README.md) for the full tour.

---

## Platform notes

### Linux + NVIDIA (CUDA)

After `make build LLAMA_CMAKE_EXTRA="-DGGML_CUDA=ON"`, ensure the CUDA runtime is on your library path :

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

If `nvcc` rejects your host compiler ("`unsupported GNU version`") use `clang++` :

```bash
sudo dnf install -y clang
make build LLAMA_CMAKE_EXTRA="-DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++"
```

### Linux + AMD (ROCm)

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
make build LLAMA_CMAKE_EXTRA="-DGGML_HIPBLAS=ON"
```

### macOS

`llama.cpp` produces `.dylib` files. The auto-fit / Metal path is on by default for Apple Silicon ; no extra knob.

### Fedora / SELinux

If loading `.so` files fails with `Permission denied` :

```bash
sudo chcon -t lib_t bridge/ion7_bridge.so
sudo chcon -t lib_t /path/to/llama.cpp/build/bin/lib*.so
```

Or install to `/usr/local/lib/` where the SELinux label is already correct.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `[ion7-core] libllama not found. Candidates : ...` | None of the candidates resolved. Set `ION7_LIBLLAMA_PATH` to the absolute file, or run from the project root after `make build`. |
| `ion7_bridge.so : libllama.so.0 : cannot open shared object file` | The bridge's rpath points at a moved or missing `libllama.so`. Rebuild with `ION7_RELEASE=1` for a relocatable rpath, or set `LD_LIBRARY_PATH`. |
| `decode : KV cache is full` | The prompt + cache exceeds `n_ctx_seq` (`n_ctx / n_seq_max`). Increase `n_ctx`, lower `n_seq_max`, or call `ctx:kv_clear()` between conversations. |
| `decode_single returned -1` after a sequence-fork loop | The Lua-side `_n_past` mirror is shared across sequences. Save `prefill_pos` before forking and call `ctx:set_n_past(prefill_pos)` before sampling each branch. |
| `llama_sampler_init_grammar_lazy_patterns : wrong number of arguments` | You're on a pre-fix build. Pull the latest `src/ion7/core/sampler.lua` — the wrapper now routes between the words / patterns variants automatically. |
| Threadpool segfault on first decode | The pool size must equal both `n_threads` AND `n_threads_batch` of the attached context. |
| `GGML_ASSERT(...)` aborts the process during `ion7_opt_init` / `ion7_opt_epoch` | Training requires `n_ctx % n_batch == 0`, F16 KV, mmap=false, and an architecture whose ops have a backward path in ggml. Some recent fused-attention models (Gated Delta Net, ...) are not yet trainable upstream. |

---

## Getting a model

If you don't have a `.gguf` yet, anything from [Hugging Face GGUF tag](https://huggingface.co/models?library=gguf&sort=trending) works. Solid starting points :

- **`Qwen2.5-7B-Instruct-Q4_K_M`** — balanced, ~4 GB on disk.
- **`Llama-3.2-3B-Instruct-Q8_0`** — fast, fits in 4 GB VRAM.
- **`Ministral-3-3B-Instruct-2512-UD-Q8_K_XL`** — what the test suite uses by default.
