# ion7-core examples

Nine standalone scripts that walk through every layer of the public
API. Each one is runnable from the project root with no extra setup
beyond a model path.

## Setup

Set the env vars your scripts need :

```bash
export ION7_MODEL=/path/to/model.gguf       # required for examples 01-04, 06-09
export ION7_EMBED=/path/to/embed.gguf       # required for example 05
export ION7_DRAFT=/path/to/draft.gguf       # optional, makes example 09 faster
export ION7_GPU_LAYERS=99                   # optional, override n_gpu_layers
```

Run from the **project root** :

```bash
luajit examples/01_hello.lua
```

## What each example shows

| File | Topic | Concepts demonstrated |
|------|-------|----------------------|
| [01_hello.lua](01_hello.lua) | minimal pipeline | load → tokenize → decode → sample loop |
| [02_chat.lua](02_chat.lua) | multi-turn chat | KV-delta prefill (re-decode only the new tokens) |
| [03_streaming.lua](03_streaming.lua) | per-token streaming | UTF-8-safe output, early stop, perf metrics |
| [04_grammar.lua](04_grammar.lua) | constrained generation | inline GBNF, JSON Schema → GBNF, nested grammar |
| [05_embeddings.lua](05_embeddings.lua) | semantic similarity | embedding context, cosine similarity matrix |
| [06_kv_reuse.lua](06_kv_reuse.lua) | KV cache tricks | snapshot / restore, sequence forking, sliding window |
| [07_custom_sampler.lua](07_custom_sampler.lua) | sampling in Lua | argmax, contrastive, stateful no-repeat |
| [08_threadpool.lua](08_threadpool.lua) | shared CPU workers | single pool across two contexts, pause / resume |
| [09_speculative.lua](09_speculative.lua) | draft-verify decoding | n-gram cache, draft model |

## Reading order

The progression is incremental — each file assumes the patterns of
the previous one without re-explaining. If you only have time for
three, read **01**, **03** and **04** : they cover the typical
chat / streaming / structured-output happy path.

## Contracts every example follows

- Reads paths from env vars only ; no hardcoded fallbacks.
- Calls `ion7.init({ log_level = 0 })` / `ion7.shutdown()` at the
  top and bottom.
- Frees its `Sampler`, `Context`, `Model` before exit.
- Defaults `n_gpu_layers = 0` so it runs on CPU-only laptops, but
  honours `ION7_GPU_LAYERS` when set.
- Uses the model's embedded chat template via `vocab:apply_template`
  rather than hand-rolling instruct formatting.
