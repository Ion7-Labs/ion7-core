<div align="center">

# ion7-core

**LuaJIT bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — local LLM inference, natively in Lua, at silicon speed.**

[![CI](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Ion7-Labs/ion7-core/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Ion7-Labs/ion7-core?include_prereleases&sort=semver)](https://github.com/Ion7-Labs/ion7-core/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![LuaJIT 2.1](https://img.shields.io/badge/LuaJIT-2.1-orange.svg)](https://luajit.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-vendored-red.svg)](https://github.com/ggml-org/llama.cpp)
[![Tests: 202](https://img.shields.io/badge/tests-202%20passing-brightgreen)](tests/)

<img src="assets/architecture.svg" alt="ion7-core architecture" width="640"/>

</div>

---

`ion7-core` is the silicon-level layer of the [Ion7 ecosystem](https://github.com/Ion7-Labs).
It puts every primitive of `llama.cpp` directly under Lua control — no Python middleman, no HTTP loop, no subprocess. Two consumption paths cohabit by design :

- **direct FFI** (`src/ion7/core/ffi/llama/*`) for everything `llama.h` and `ggml*.h` already expose as a stable C ABI ;
- **a thin C++ shim** (`bridge/ion7_bridge.so`, ~34 symbols) for the libcommon helpers that only exist in C++ — chat templates, advanced samplers, JSON-Schema → GBNF, speculative decoding, training.

Everything else — KV cache management, sampler chains, custom samplers, UTF-8 streaming, threadpool sharing, metric snapshots — runs in pure LuaJIT, JIT-compiled.

## Quick taste

```lua
local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local model   = ion7.Model.load("qwen2.5-7b-instruct-q4_k_m.gguf",
                                { n_gpu_layers = 35 })
local ctx     = model:context({ n_ctx = 8192, kv_type = "q8_0" })
local vocab   = model:vocab()
local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95):temperature(0.8):dist():build()

local prompt    = vocab:apply_template(
    { { role = "user", content = "Explain RoPE embeddings in two sentences." } },
    true, -1)
local toks, n   = vocab:tokenize(prompt, false, true)
ctx:decode(toks, n)

for _ = 1, 256 do
    local tok = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(tok) then break end
    io.write(vocab:piece(tok)) ; io.flush()
    ctx:decode_single(tok)
end

ion7.shutdown()
```

Nine progressive examples in [`examples/`](examples/) walk through every layer — from a 30-line hello-world to multi-turn chat with KV-delta prefilling, JSON-Schema-constrained output, embeddings, KV snapshots, custom samplers, threadpool sharing and speculative decoding.

## What's covered

| Surface | Status | Where to look |
|---------|:------:|---------------|
| Model loading, GGUF metadata, LoRA, quantization | ✅ | `Model`, [`10_model.lua`](tests/10_model.lua) |
| Tokenize / detokenize, special tokens, Jinja2 templates | ✅ | `Vocab`, [`11_vocab.lua`](tests/11_vocab.lua) |
| Context, decode / encode, warmup, perf counters | ✅ | `Context`, [`12_context.lua`](tests/12_context.lua) |
| KV cache : clear, seq_rm, seq_cp, shift, div | ✅ | [`13_kv.lua`](tests/13_kv.lua) |
| State : snapshot / restore / save / load + per-seq | ✅ | [`14_state.lua`](tests/14_state.lua) |
| Logits, logprob, entropy, embeddings, sampled_* | ✅ | [`15_logits.lua`](tests/15_logits.lua) |
| Sampler chain : 14 samplers + factories + clone | ✅ | `Sampler`, [`16_sampler.lua`](tests/16_sampler.lua) |
| Lua-implemented `CustomSampler` (ffi.cast) | ✅ | [`17_custom_sampler.lua`](tests/17_custom_sampler.lua) |
| Shared CPU `Threadpool` across multiple contexts | ✅ | [`18_threadpool.lua`](tests/18_threadpool.lua) |
| Speculative decoding (n-gram cache, draft model, EAGLE3) | ✅ | [`19_speculative.lua`](tests/19_speculative.lua) |
| Chat templates + chat parse + reasoning_budget | ✅ | [`20_chat_templates.lua`](tests/20_chat_templates.lua) |
| Grammar tooling (GBNF, JSON Schema, partial regex) | ✅ | [`21_grammar.lua`](tests/21_grammar.lua) |
| Training (`llama_opt`) | ⚠️ | gated on `ION7_TRAIN=1` ; arch-dependent upstream |
| Multimodal (`mtmd` / vision-language) | ❌ | not yet wrapped |
| Distributed RPC backend | ❌ | capability detection only |

## Install

### From a release tarball (recommended)

Pre-built bundles ship with [`ion7-core` releases](https://github.com/Ion7-Labs/ion7-core/releases). Each tarball contains the bridge, every `llama.cpp` shared library it transitively needs, the Lua runtime and a `bin/ion7-load.lua` preamble that wires `package.path` and the loader env vars :

```bash
curl -L -o ion7.tgz \
  https://github.com/Ion7-Labs/ion7-core/releases/latest/download/ion7-core-linux-x86_64-cpu.tar.gz
tar xzf ion7.tgz
cd ion7-core-*/

luajit -e 'dofile("bin/ion7-load.lua") ; require "ion7.core".init()'
```

Available targets : Linux x86_64 (CPU, Vulkan), Linux aarch64 (CPU), macOS arm64 (Metal), macOS x86_64 (CPU). Windows, CUDA and ROCm are user-built ; see [`INSTALL.md`](INSTALL.md).

### From source (vendored llama.cpp)

```bash
git clone --recurse-submodules https://github.com/Ion7-Labs/ion7-core
cd ion7-core
make build              # auto-builds vendor/llama.cpp + bridge
make test ION7_MODEL=/path/to/model.gguf
```

For external `llama.cpp`, custom CUDA / ROCm / Vulkan flags, distributed builds, environment variables and platform-specific notes, see **[`INSTALL.md`](INSTALL.md)**.

## Compatibility

| Component   | Requirement              |
|-------------|--------------------------|
| LuaJIT      | 2.1 (any post-2017 build) |
| C++ host    | GCC 11+ or Clang 14+ (C++17) |
| CMake       | 3.21+ for the vendor build |
| OS          | Linux (glibc), macOS 12+ |
| CPU         | x86_64 (AVX2), aarch64 (NEON) |
| GPU         | CUDA, ROCm, Metal, Vulkan |
| `llama.cpp` | vendored as a submodule (pinned per release) |

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layered design, module tree, lifetime model, hot path, bridge anatomy, build pipeline (six annotated diagrams)
- [`INSTALL.md`](INSTALL.md) — build paths, runtime configuration, troubleshooting
- [`examples/README.md`](examples/README.md) — guided tour of the nine example scripts
- [Online API reference](https://ion7-labs.github.io) — generated from the `--- @module` LuaDoc blocks

## License

[MIT](LICENSE). `ion7-core` builds on `llama.cpp` by Georgi Gerganov and contributors.
