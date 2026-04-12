# ion7-core

[![CI](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml)
[![Release](https://github.com/Ion7-Labs/ion7-core/actions/workflows/release.yml/badge.svg)](https://github.com/Ion7-Labs/ion7-core/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![LuaJIT](https://img.shields.io/badge/LuaJIT-2.1-orange.svg)](https://luajit.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-master-red.svg)](https://github.com/ggml-org/llama.cpp)
[![Version](https://img.shields.io/badge/version-1.2.0-brightgreen.svg)](https://github.com/Ion7-Labs/ion7-core/releases)

**LuaJIT bindings for llama.cpp — zero-overhead local LLM inference, natively in Lua.**

ion7-core bridges LuaJIT and llama.cpp via a thin C++ shim (`ion7_bridge.so`) that absorbs llama.cpp API churn. Load a GGUF model, decode tokens, manage KV cache, write custom samplers in pure Lua — at native speed, no Python, no HTTP.

```lua
local ion7    = require "ion7.core"
ion7.init({ log_level = 0 })

local model   = ion7.Model.load("qwen2.5-7b-instruct-q4_k_m.gguf", { n_gpu_layers = 35 })
local ctx     = model:context({ n_ctx = 8192, kv_type = "q8_0" })
local vocab   = model:vocab()
local sampler = ion7.Sampler.chain():top_k(40):top_p(0.95):temp(0.8):dist():build()

local prompt  = vocab:apply_template({{ role = "user", content = "Explain RoPE embeddings." }})
local tokens, n = vocab:tokenize(prompt, true, true)
ctx:decode(tokens, n, 0, 0)

local out = {}
for _ = 1, 256 do
    local token = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(token) then break end
    out[#out + 1] = vocab:piece(token)
    ctx:decode_single(token, 0)
end

print(table.concat(out))
ion7.shutdown()
```

**[Full API documentation →](https://ion7-labs.github.io)**

---

## Quick start

**Prerequisites:** LuaJIT 2.1, GCC/Clang with C++17, a compiled llama.cpp.

```bash
git clone https://github.com/ion7-labs/ion7-core
cd ion7-core

# External llama.cpp build
make build LIB_DIR=/path/to/llama.cpp/build/bin COMMON_LIB_DIR=/path/to/llama.cpp/build/common

# Or persist paths via local.mk (gitignored)
cp local.mk.example local.mk
# edit LIB_DIR and COMMON_LIB_DIR, then:
make build

# Bundled llama.cpp with CUDA (auto-build)
make build CUDA_ARCH=86   # 86=RTX30xx  89=RTX40xx  80=A100
```

```bash
# Run tests
make test ION7_MODEL=/path/to/your-model.gguf
```

---

## Compatibility

| Component | Requirement |
|---|---|
| LuaJIT | 2.1+ |
| llama.cpp | b8600+ (April 2026) |
| OS | Linux, macOS |
| GPU | CUDA, ROCm, Metal, Vulkan |
| CPU | x86_64, ARM64 |

---

## License

MIT — free to use in any project, open source or commercial.

ion7-core is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
