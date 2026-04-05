# ion7-core — Benchmark Results

**Model:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf`  
**Hardware:** Ryzen 9 9950X · RTX 3060 12GB · 64 GB DDR5 · Fedora 43  
**Stack:** llama.cpp b8600 · LuaJIT 2.1 · CUDA 13.2

---

## Reproduce

```bash
# Absolute performance (14 sections, JSON output)
make bench ION7_MODEL=/path/to/model.gguf \
           LLAMA_LIB=/path/to/llama.cpp/build/bin/libllama.so \
           BENCH_ARGS='--n-gpu-layers 35 --n-ctx 8192 --n-gen 100'

# Side-by-side vs llama-cpp-python
make compare ION7_MODEL=/path/to/model.gguf \
             LLAMA_LIB=/path/to/llama.cpp/build/bin/libllama.so \
             BENCH_ARGS='--n-gpu-layers 35 --n-ctx 8192'

# Long-run stability (memory leak detection, throughput drift)
make stability ION7_MODEL=/path/to/model.gguf \
               LLAMA_LIB=/path/to/llama.cpp/build/bin/libllama.so \
               BENCH_ARGS='--n-gpu-layers 35 --n-ctx 8192'
```

---

## 1. ion7-core vs llama-cpp-python 0.3.20

n_ctx 8192 · 35 GPU layers · n_gen 128 · n_repeat 3

### Speedup ratios

![speedup](charts/speedup.svg)

### Performance comparison

![compare](charts/compare.svg)

### Full comparison table

| Section | ion7-core | llama-cpp-python | ratio |
|---|---|---|---|
| **Model load** | 378 ms | 499 ms | **1.32×** |
| **Prompt prefill** (avg, 3 prompts) | 1 170 tok/s | 883 tok/s | **1.32×** |
| Prefill — case 1 (15 tokens) | 692 tok/s | 181 tok/s | **3.82×** |
| Prefill — case 2 (26 tokens) | 1 014 tok/s | 770 tok/s | 1.32× |
| Prefill — case 3 (81 tokens) | 1 803 tok/s | 1 699 tok/s | 1.06× |
| **Token generation** | 67.3 tok/s | 63.9 tok/s | **1.05×** |
| Generation latency | 14.85 ms/tok | 15.65 ms/tok | 1.05× |
| Tokenization (avg) | 609 k tok/s | 602 k tok/s | 1.01× |
| **Detokenization** | 0.001 ms | 0.009 ms | **9×** |
| **Context creation** (KV f16) | 17 ms | 395 ms | **23×** |
| Grammar-constrained gen | 48.9 ms | 51.3 ms | 1.05× |
| KV snapshot save (in-memory) | 0.47 ms | — | — |
| KV snapshot restore (in-memory) | 2.01 ms | — | — |

> **Note on context creation:** Python's overhead comes from ctypes boxing and `__init__` plumbing; ion7-core hits the C API directly via JIT-compiled FFI.

### ion7-core internals (no Python equivalent)

| Metric | Value |
|---|---|
| Sampler chain — greedy / minimal / standard | 0.061 / 0.060 / 0.065 ms/sample |
| Custom Lua sampler overhead vs native | ~10% |
| `bos()` calls/s | 785 M/s |
| `is_eog()` calls/s | 182 M/s |
| `kv_seq_rm()` calls/s | 321 k/s |
| `kv_seq_cp()` calls/s | 310 k/s |
| State save (file) | 0.29 ms |
| State load (file) | 1.64 ms |
| malloc/free per generated token | **0** |

---

## 2. Stability — sustained generation

n_ctx 8192 · 35 GPU layers · target 100 000 tokens  
Checkpoint every 5 000 tokens, KV reset when context fills.

![stability](charts/stability.svg)

### Stability table

| Checkpoint | tok/s | RSS (MB) | RSS Δ from start | KV resets |
|---|---|---|---|---|
| 5 000 | 66.2 | 708.3 | +60.6 | 49 |
| 10 000 | 66.2 | 708.3 | +60.6 | 97 |
| 15 000 | 66.2 | 708.3 | +60.6 | 145 |
| 20 000 | 66.2 | 708.3 | +60.6 | 193 |
| 25 000 | 66.2 | 708.3 | +60.6 | 241 |
| 30 000 | 66.2 | 708.3 | +60.6 | 289 |
| 35 000 | 66.2 | 708.3 | +60.6 | 338 |
| 40 000 | 66.2 | 708.3 | +60.6 | 386 |
| 45 000 | 66.2 | 708.3 | +60.6 | 434 |
| 50 000 | 66.2 | 708.3 | +60.6 | 483 |
| 55 000 | 66.2 | 708.3 | +60.6 | 531 |
| 60 000 | 66.2 | 708.3 | +60.6 | 579 |
| 65 000 | 66.2 | 708.3 | +60.6 | 627 |
| 70 000 | 66.2 | 708.5 | +60.7 | 675 |
| 75 000 | 66.2 | 708.5 | +60.7 | 723 |
| 80 000 | 66.2 | 708.5 | +60.7 | 771 |
| 85 000 | 66.2 | 708.5 | +60.7 | 820 |
| 90 000 | 66.2 | 708.5 | +60.7 | 868 |
| 95 000 | 66.2 | 708.5 | +60.7 | 916 |
| 100 000 | 66.2 | 708.5 | +60.7 | 964 |

**avg throughput: 66.15 tok/s · total time: 1511.7s · post-warmup RSS variation: +0.2 MB over 964 KV resets.**

---

## 3. Future comparisons

Planned: go-llama · llama-cpp-rs · llama.cpp direct C baseline

---

## Methodology

- **n_repeat 3** — median taken, not mean, to minimize OS scheduling noise
- **KV snapshot** — ion7-core uses in-memory blobs (`snapshot()`/`restore()`); no file I/O involved
- **Pre-allocated batch** — `decode_single()` reuses a single pre-allocated `llama_batch`; zero malloc per token in the hot path
- **Custom sampler overhead** — measured as `(lua_greedy_time - native_greedy_time) / native_greedy_time`; FFI callback round-trip only
- **Context creation** — ion7-core measures only the `ion7_context_create()` call; Python includes ctypes boxing and Python object construction
- **Stability** — RSS sampled via `/proc/self/status VmRSS`; tok/s computed over each 5k-token window independently
