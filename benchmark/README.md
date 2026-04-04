# ion7-core Benchmark Suite

Comprehensive performance comparison between **ion7-core (LuaJIT FFI)** and
**llama-cpp-python (Python ctypes)**. Same llama.cpp shared library under the
hood - different language runtimes and binding overhead.

## Quick start

```bash
# Full Lua vs Python comparison (14 sections)
ION7_MODEL=/path/to/model.gguf bash benchmark/compare.sh

# With options
ION7_MODEL=/path/to/model.gguf bash benchmark/compare.sh \
  --n-gpu-layers 20 --n-gen 128 --n-repeat 5 --stress

# Lua only (JSON output)
ION7_MODEL=/path/to/model.gguf luajit benchmark/bench_lua.lua | jq .

# Python only
ION7_MODEL=/path/to/model.gguf python3 benchmark/bench_python.py | jq .

# Via Makefile
make bench     ION7_MODEL=/path/to/model.gguf
make compare   ION7_MODEL=/path/to/model.gguf
make stability ION7_MODEL=/path/to/model.gguf
```

## Benchmarks

### Core (1-8) - direct comparison

| # | Benchmark | Measures |
|---|-----------|----------|
| 1 | Model load | Cold load time from disk |
| 2 | Tokenization | Throughput for short/medium/long inputs |
| 3 | Prompt prefill | Decode speed for prompt tokens (TTFT) |
| 4 | Generation | Token generation throughput (tok/s) |
| 5 | Grammar constraint | GBNF-constrained generation - speed + correctness |
| 6 | KV snapshot | Save/restore KV cache state |
| 7 | Detokenization | Token-to-text speed |
| 8 | Sampler overhead | Per-sample cost for different sampler profiles |

### Deep (9-14) - ion7-core internals

| # | Benchmark | Measures |
|---|-----------|----------|
| 9 | Context creation | Create+free with f16/q8_0/q4_0 KV cache quantization |
| 10 | decode_single | Single-token decode hot path (with/without KV cache) |
| 11 | KV operations | kv_clear, kv_seq_rm, kv_seq_cp throughput |
| 12 | State persistence | File I/O save/load round-trip |
| 13 | Custom Lua sampler | FFI callback overhead vs native greedy |
| 14 | Vocab operations | bos(), is_eog(), piece() micro-benchmarks |

### Stress (S1-S4) - `--stress` flag

| # | Benchmark | Measures |
|---|-----------|----------|
| S1 | Sustained generation | Throughput stability over 512 tokens (32-token windows) |
| S2 | Back-to-back sessions | 10 consecutive sessions with KV clear |
| S3 | Context fill pressure | Generation at 75% context fill |
| S4 | Sampler throughput | 1000 raw sample() calls |

### Standalone

| File | Purpose |
|------|---------|
| `bench_embed.lua` | Embedding model: sequential, batch, throughput at scale |
| `bench_stability.lua` | Long-running generation (100k tokens) with RSS monitoring |

## Install llama-cpp-python

```bash
# CPU only
pip install llama-cpp-python

# CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# Optional memory tracking
pip install psutil
```

## CLI options

All flags are forwarded to both Lua and Python benchmarks:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-gpu-layers N` | auto-fit | GPU layers to offload |
| `--n-ctx N` | 2048 | Context size |
| `--n-gen N` | 128 | Tokens to generate per benchmark |
| `--n-repeat N` | 3 | Repetitions per measurement |
| `--seed N` | 42 | RNG seed |
| `--n-batch N` | auto | Batch size |
| `--n-ubatch N` | auto | Physical batch size |
| `--flash` | off | Enable flash attention |
| `--stress` | off | Run stress tests S1-S4 |

## Expected differences

**ion7-core advantages:**
- LuaJIT FFI has ~zero marshalling overhead - direct C pointer access
- In-memory KV snapshots (no file I/O)
- Per-sample overhead measurable at microsecond scale
- Custom Lua samplers via FFI callbacks
- Context creation with KV cache quantization control

**llama-cpp-python advantages:**
- Larger ecosystem, battle-tested
- Python async/await integration
- Easier to install

**Same regardless of binding:**
- Actual CUDA/CPU compute (kernel execution time)
- Model load from disk (dominated by I/O)
- KV cache memory layout

## Output

```
benchmark/last_results_lua.json     # Raw ion7-core results
benchmark/last_results_python.json  # Raw llama-cpp-python results
```

Both follow the same JSON schema for automated comparison.
