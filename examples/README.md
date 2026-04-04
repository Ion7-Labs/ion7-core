# ion7-core examples

Progressive examples from minimal to advanced. Each runs standalone.

## Setup

```bash
export ION7_MODEL=/path/to/model.gguf
export ION7_EMBED=/path/to/embed.gguf   # for example 05
export ION7_LIB_DIR=/path/to/llama.cpp/build/bin
```

Run from the **project root**:

```bash
luajit examples/01_hello.lua
```

## Examples

| File | Concepts | Difficulty |
|------|----------|------------|
| [01_hello.lua](01_hello.lua) | Load model, tokenize, generate | ★☆☆☆ |
| [02_chat.lua](02_chat.lua) | Multi-turn chat, chat template, KV cache | ★★☆☆ |
| [03_streaming.lua](03_streaming.lua) | Token streaming, callbacks, early stop | ★★☆☆ |
| [04_grammar.lua](04_grammar.lua) | GBNF constraints, JSON output, lazy grammar | ★★★☆ |
| [05_embeddings.lua](05_embeddings.lua) | Embedding model, cosine similarity, RAG foundation | ★★★☆ |
| [06_kv_reuse.lua](06_kv_reuse.lua) | Snapshot/restore, sequence forking, sliding window | ★★★★ |
| [07_custom_sampler.lua](07_custom_sampler.lua) | Custom sampling in Lua, stateful callbacks | ★★★★ |
| [08_threadpool.lua](08_threadpool.lua) | Shared CPU threadpool, pause/resume | ★★★★ |

## What's not here yet

These will come with the higher-level modules:

- `ion7-llm` - OpenAI-compatible API, automatic chat management
- `ion7-rag` - vector store, chunking, retrieval pipeline
- `ion7-vision` - multimodal (LLaVA, Qwen-VL)
