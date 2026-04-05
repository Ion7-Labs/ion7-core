# ion7-core

[![CI](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml)
[![Release](https://github.com/Ion7-Labs/ion7-core/actions/workflows/release.yml/badge.svg)](https://github.com/Ion7-Labs/ion7-core/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-green.svg)](#-licensing)
[![LuaJIT](https://img.shields.io/badge/LuaJIT-2.1-orange.svg)](https://luajit.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-master-red.svg)](https://github.com/ggml-org/llama.cpp)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/Ion7-Labs/ion7-core/releases)

**LuaJIT bindings for llama.cpp - zero-overhead local LLM inference, natively in Lua.**

ion7-core is the missing bridge between LuaJIT and llama.cpp. Load a GGUF model, decode tokens, sample, manage KV cache, write custom samplers in pure Lua - all at native speed, with no Python, no HTTP, no overhead.

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

---

## Why ion7-core?

llama.cpp has bindings for Python, Go, Rust, Java, Swift - but nothing serious for Lua. If you're building LLM features inside a Lua runtime (game engines, OpenResty, Lapis, Eluna, Love2D, embedded scripting hosts), your options until now were:

- Call the llama-server HTTP API and eat the latency
- Shell out to a Python subprocess
- Write your own FFI bindings from scratch

ion7-core is the third option, done properly.

### The bridge architecture

Calling llama.cpp directly from LuaJIT FFI is fragile - llama.cpp breaks its internal API frequently as it evolves. ion7-core solves this with a thin C shim (`ion7_bridge.so`) that sits between LuaJIT and `libllama.so`:

```
Your Lua code
     │
     ▼
ion7.core  (LuaJIT FFI)
     │
     ├──► ion7_bridge.so   ← stable API, absorbs llama.cpp churn
     │         │
     │         ▼
     │    libllama.so      ← llama.cpp native library
     │
     └──► libllama.so      ← direct FFI for stable structs (batch, logits...)
```

When llama.cpp changes its structs or param signatures, only the bridge needs recompiling. Your Lua code stays untouched.

---

## Features

**Model management**
- Load GGUF models from disk, file descriptor, or multi-shard paths
- Full metadata introspection (architecture, dimensions, GGUF key-value pairs)
- Auto-fit VRAM: probe available GPU memory and compute optimal `n_gpu_layers` + `n_ctx`
- In-place quantization (Q2K → Q8_0, BF16, etc.)

**Inference**
- Chunked batch decode with automatic KV accumulation
- Single-token decode loop with pre-allocated batch (zero malloc per token)
- Encoder-decoder support (T5, Whisper)
- Per-position logits and embeddings access

**KV cache**
- Full cache lifecycle: clear, remove, copy, keep, shift
- Per-sequence operations for parallel decoding
- State snapshot/restore (in-memory or file-backed)
- Per-sequence state persistence

**Sampler chains**
- Fluent builder API: `Sampler.chain():top_k(40):temp(0.8):dist():build()`
- All llama.cpp samplers: greedy, dist, top-k, top-p, min-p, typical, XTC, mirostat v1/v2, DRY, penalties, grammar (GBNF), logit bias, adaptive-p, top-n-sigma
- **Custom Lua samplers**: write sampling logic in pure Lua, inject it into the chain via `ffi.cast` callbacks

**System**
- Shared CPU threadpool across contexts
- LoRA adapter loading and hot-swapping
- Activation steering (control vectors)
- Full performance counters
- Embedding contexts with pooling (mean, cls, last, rank)
- Chat template rendering (Jinja-compatible, model-embedded or custom)

---

## Benchmarks

Tested on **Ryzen 9 9950X + RTX 3060 12GB**, Fedora 43, llama.cpp b8600, LuaJIT 2.1.  
Model: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` — n_ctx 8192, 35 GPU layers.

| Metric | ion7-core | vs llama-cpp-python |
|---|---|---|
| Token generation | **67 tok/s** | 1.05× faster |
| Prompt prefill (avg) | **1 170 tok/s** | 1.32× faster |
| Context creation | **17 ms** | 23× faster |
| Detokenization | **0.001 ms** | 9× faster |
| malloc/free per token | **0** | — |
| RSS growth over 100k tokens | **0 MB** | — |

→ [Full results, charts and stability report](benchmark/RESULTS.md)

> Run `make bench ION7_MODEL=your.gguf` for your own baseline.

---

## Quick start

### Prerequisites

- LuaJIT 2.1
- A compiled llama.cpp (see [INSTALL.md](INSTALL.md) for build instructions)
- GCC or Clang

### Build the bridge

```bash
git clone https://github.com/ion7-labs/ion7-core
cd ion7-core

make build LIB_DIR=/path/to/llama.cpp/build/bin
```

This compiles `bridge/ion7_bridge.so` against your local llama.cpp install.

### Run the tests

```bash
make test ION7_MODEL=/path/to/your-model.gguf
```

### Run the benchmarks

```bash
make bench ION7_MODEL=/path/to/your-model.gguf
```

See [INSTALL.md](INSTALL.md) for detailed build options, CUDA/ROCm/Metal support, LoRA API compatibility flags, and library path configuration.

---

## Usage

### Loading a model

```lua
local ion7 = require "ion7.core"
ion7.init({ log_level = 1 })  -- 0=silent, 1=error, 2=warn, 3=info, 4=debug

-- Basic load
local model = ion7.Model.load("model.gguf", {
    n_gpu_layers = 35,    -- layers to offload to GPU (0 = CPU only, -1 = all)
    use_mmap     = true,  -- default: true
})

-- Auto-fit VRAM
local fit = ion7.Model.fit_params("model.gguf", { n_ctx = 32768 })
if fit then
    local model = ion7.Model.load("model.gguf", { n_gpu_layers = fit.n_gpu_layers })
    local ctx   = model:context({ n_ctx = fit.n_ctx })
end
```

### Creating a context

```lua
local ctx = model:context({
    n_ctx        = 8192,    -- context window
    kv_type      = "q8_0", -- KV cache quantization: "f16", "q8_0", "q4_0", ...
    flash_attn   = true,
    offload_kqv  = true,
    n_threads    = 8,
})
```

### Tokenization and chat templates

```lua
local vocab = model:vocab()

-- Chat template (uses the model's embedded template)
local prompt = vocab:apply_template({
    { role = "system",    content = "You are a helpful assistant." },
    { role = "user",      content = "What is LuaJIT?" },
}, true)  -- true = append assistant prefix

local tokens, n = vocab:tokenize(prompt, true, true)
```

### Generation loop

```lua
local sampler = ion7.Sampler.chain()
    :penalties(64, 1.1, 0.0, 0.0)
    :top_k(40)
    :top_p(0.95)
    :temp(0.8)
    :dist()
    :build()

ctx:decode(tokens, n, 0, 0)

local out = {}
for _ = 1, 512 do
    local token = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(token) then break end
    out[#out + 1] = vocab:piece(token)
    ctx:decode_single(token, 0)
end

print(table.concat(out))
```

### Custom Lua sampler

```lua
local cs = ion7.CustomSampler.new("my_sampler", {
    apply = function(candidates, n)
        -- candidates: cdata llama_token_data* (.id, .logit, .p)
        -- return the index of the token you want to select (0-based)
        local best_i, best = 0, -math.huge
        for i = 0, n - 1 do
            if candidates[i].logit > best then
                best   = candidates[i].logit
                best_i = i
            end
        end
        return best_i
    end,
    accept = function(token_id) end, -- optional
    reset  = function() end,         -- optional
})

local sampler = ion7.Sampler.chain():custom(cs):build()
```

### Embeddings

```lua
local embed_ctx = model:embedding_context({
    n_ctx    = 512,
    pooling  = "last",   -- "none", "mean", "cls", "last", "rank"
    n_threads = 4,
})

local tokens, n = vocab:tokenize("Embed this text.", false, false)
embed_ctx:decode(tokens, n, 0, 0)

local vec = embed_ctx:embedding(0, model:n_embd())
-- vec is a Lua array of floats
```

---

## Module structure

```
ion7-core
├── bridge/
│   ├── ion7_bridge.c     - C shim (compile once, insulates from llama.cpp churn)
│   └── ion7_bridge.h     - stable public C API
└── src/ion7/core/
    ├── init.lua           - ion7.init(), ion7.shutdown(), ion7.capabilities()
    ├── model.lua          - Model.load(), model:context(), model:vocab()
    ├── context.lua        - decode, KV cache, logits, state persistence
    ├── vocab.lua          - tokenize, detokenize, chat templates, special tokens
    ├── sampler.lua        - fluent sampler chain builder
    ├── custom_sampler.lua - Lua-native custom samplers via ffi.cast
    ├── threadpool.lua     - shared CPU threadpool
    └── ffi/
        ├── loader.lua     - library resolution and backend init
        └── types.lua      - all llama.cpp cdef declarations
```

The public API contract (stable across 1.x minor versions) is documented in [`spec/PUBLIC_API.md`](spec/PUBLIC_API.md).

---

## Compatibility

| Component | Requirement |
|---|---|
| LuaJIT | 2.1+ |
| llama.cpp | b8600+ (April 2026) |
| OS | Linux, macOS |
| GPU | CUDA (NVIDIA), ROCm (AMD), Metal (Apple Silicon), Vulkan |
| CPU | x86_64, ARM64 |

> **llama.cpp compatibility note:** ion7-core tracks llama.cpp master. The bridge absorbs most API changes, but a recompile of `ion7_bridge.so` is required when llama.cpp makes breaking C API changes. The bridge version (`ion7.capabilities().bridge_ver`) is independent of the ion7-core version.

---

## 📄 Licensing

ion7-core is dual-licensed.

### Open Source - AGPLv3

Free to use under the [GNU Affero General Public License v3](LICENSE). This means: if you use ion7-core in a network-accessible service or product, you must release your source code under the same license.

### Commercial License

If you want to integrate ion7-core into a proprietary product, closed-source game, or SaaS without AGPLv3 obligations, a commercial license is available.

**→ [Contact for commercial licensing](mailto:contact@ion7.dev)**

Commercial licenses cover: single project use, team use, OEM embedding. Pricing on request.

---

## Contributing

Issues and pull requests are welcome. Before opening a PR for a significant feature, open an issue first to discuss the design - ion7-core has a strict stability contract for the 1.x API surface.

Please read [`spec/PUBLIC_API.md`](spec/PUBLIC_API.md) before contributing to understand what is and isn't in scope for ion7-core.

---

## Acknowledgements

ion7-core is built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors - the project that made local LLM inference possible for everyone.
