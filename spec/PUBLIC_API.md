# ion7-core - Public API Contract v1.0

This document is the **stability contract** for `ion7-core 1.x`.

Every symbol listed here is guaranteed stable across minor versions.
Breaking changes require a major version bump (2.0).
Downstream modules (`ion7-llm`, `ion7-embed`, ...) depend on this.

---

## Scope

ion7-core exposes **runtime primitives only**:
- Model loading and introspection
- Context lifecycle and decode operations
- KV cache management
- State persistence
- Tokenization and vocabulary access
- Sampler chain construction
- Custom Lua samplers
- Threadpool management
- Performance monitoring

**NOT in scope:** chat loops, stop strings, streaming, RAG, grammar,
embeddings, template engines. Those belong in downstream modules.

---

## ion7.core (init.lua)

```lua
local ion7 = require "ion7.core"
```

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ion7.init(opts?)` | `opts.{log_level, llama_path, bridge_path}` | Must call before any model load |
| `ion7.shutdown()` | `→ void` | Call at process exit. Safe to call multiple times. |
| `ion7.capabilities()` | `→ table` | See table below |
| `ion7.time_us()` | `→ number` | Microsecond timestamp from llama.cpp clock |
| `ion7.numa_init(strategy?)` | `→ void` | Call BEFORE `ion7.init()` on NUMA systems |
| `ion7.NUMA_DISABLED` | `= 0` | No NUMA optimization |
| `ion7.NUMA_DISTRIBUTE` | `= 1` | Distribute threads across NUMA nodes (default) |
| `ion7.NUMA_ISOLATE` | `= 2` | Isolate to specific node |
| `ion7.NUMA_NUMACTL` | `= 3` | Use numactl settings |
| `ion7.NUMA_MIRROR` | `= 4` | Mirror across nodes |

### ion7.capabilities() return table

| Field | Type | Description |
|-------|------|-------------|
| `mmap` | bool | Memory-mapped loading supported |
| `mlock` | bool | Page locking supported |
| `gpu_offload` | bool | GPU offload available (CUDA/Metal/Vulkan) |
| `rpc` | bool | RPC backend available |
| `max_devices` | number | Maximum GPU device count |
| `max_parallel_seqs` | number | Maximum parallel sequences |
| `bridge_ver` | string | ion7_bridge version, e.g. `"1.0.0"` |
| `llama_info` | string | llama.cpp system info string |

---

## ion7.core.Model

```lua
local Model = require "ion7.core.model"
-- or: ion7.Model  (lazy-loaded)
```

### Class methods (constructors)

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Model.load(path, opts?)` | `→ Model` | Load single GGUF. `opts.n_gpu_layers`, `.use_mmap`, `.use_mlock`, `.vocab_only` |
| `Model.load_splits(paths, opts?)` | `→ Model` | Multi-shard GGUF. `paths` is a table of file paths in order |
| `Model.load_fd(fd, ngl?)` | `→ Model` | Load from open file descriptor. Caller owns the fd |
| `Model.fit_params(path, opts?)` | `→ table\|nil` | Auto-fit VRAM. Returns `{n_gpu_layers, n_ctx}` or `nil` |
| `Model.quantize(inp, out, opts?)` | `→ number` | Quantize GGUF file. Returns 0 on success |

#### Model.fit_params opts

| Field | Default | Description |
|-------|---------|-------------|
| `n_ctx` | `0` | Desired context size (0 = maximize) |
| `n_ctx_min` | `512` | Minimum acceptable context size |

#### Model.quantize opts

| Field | Default | Description |
|-------|---------|-------------|
| `ftype` | `"q4_k_m"` | Target format: `"f32"`, `"f16"`, `"bf16"`, `"q4_0"`, `"q4_1"`, `"q5_0"`, `"q5_1"`, `"q8_0"`, `"q2_k"`, `"q3_k_s"`, `"q3_k_m"`, `"q3_k_l"`, `"q4_k_s"`, `"q4_k_m"`, `"q5_k_s"`, `"q5_k_m"`, `"q6_k"`, `"iq4_xs"`, `"copy"` |
| `nthread` | `0` | Worker threads (0 = hardware concurrency) |
| `pure` | `false` | Quantize ALL tensors (no mixed precision) |
| `allow_requantize` | `false` | Allow re-quantizing already-quantized tensors |
| `quantize_output` | `false` | Quantize the output tensor too |
| `keep_split` | `false` | Preserve shard count in output |
| `dry_run` | `false` | Compute plan without writing output |

### Instance methods - Context creation

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `model:context(opts?)` | `→ Context` | Create inference context |
| `model:embedding_context(opts?)` | `→ Context` | Create embedding context |
| `model:vocab()` | `→ Vocab` | Get vocabulary handle (cached) |

#### model:context opts

| Field | Default | Description |
|-------|---------|-------------|
| `n_ctx` | `4096` | Context window size |
| `n_batch` | `2048` | Logical max batch size |
| `n_ubatch` | auto | Physical micro-batch (auto: 512 GPU, 256 CPU) |
| `n_seq_max` | `1` | Max parallel sequences |
| `n_gpu_layers` | `0` | Layers to offload (0=CPU, -1=all) |
| `n_threads` | `4` | Generation thread count |
| `n_threads_batch` | auto | Batch processing threads |
| `kv_type` | `"f16"` | KV cache type for K and V: `"f16"`, `"bf16"`, `"q8_0"`, `"q4_0"`, `"q4_1"`, `"q5_0"`, `"q5_1"`, `"iq4_nl"`. Use `kv_type_k` / `kv_type_v` to set them independently. |
| `flash_attn` | `false` | Enable Flash Attention |
| `offload_kqv` | `true` | Offload KQV ops to GPU |
| `no_perf` | `false` | Disable performance counters |

#### model:embedding_context opts

| Field | Default | Description |
|-------|---------|-------------|
| `n_ctx` | `512` | Context window |
| `n_seq_max` | `1` | Parallel sequences (>1 enables batch embedding) |
| `n_threads` | `4` | Thread count |
| `pooling` | `"last"` | Strategy: `"none"`, `"mean"`, `"cls"`, `"last"`, `"rank"` |

### Instance methods - Introspection

| Symbol | Returns | Notes |
|--------|---------|-------|
| `model:info()` | `table` | Full metadata: `n_params, n_layer, n_embd, n_head, n_head_kv, n_ctx_train, n_embd_inp, n_embd_out, n_swa, size, rope_type, has_encoder, has_decoder, is_recurrent, is_hybrid, is_diffusion` |
| `model:n_params()` | `number` | Parameter count |
| `model:n_layer()` | `number` | Transformer layer count |
| `model:n_embd()` | `number` | Main embedding dimension |
| `model:n_embd_inp()` | `number` | Input embedding dimension |
| `model:n_embd_out()` | `number` | Output embedding dimension |
| `model:n_head()` | `number` | Attention head count |
| `model:n_head_kv()` | `number` | KV head count (GQA models: < n_head) |
| `model:n_swa()` | `number` | Sliding window attention size |
| `model:n_ctx_train()` | `number` | Training context length |
| `model:n_cls_out()` | `number` | Classifier output classes (0 if not a classifier) |
| `model:size()` | `number` | Total tensor size in bytes |
| `model:rope_type()` | `string` | `"none"\|"norm"\|"neox"\|"mrope"\|"imrope"\|"vision"\|"unknown"` |
| `model:rope_freq_scale_train()` | `number` | RoPE frequency scale from training |
| `model:decoder_start_token()` | `number` | Decoder start token ID (-1 if N/A) |
| `model:cls_label(i)` | `string?` | Classifier label at index i (0-based) |
| `model:has_encoder()` | `bool` | Has encoder (T5, Whisper, ...) |
| `model:has_decoder()` | `bool` | Has decoder (most LLMs) |
| `model:is_recurrent()` | `bool` | Recurrent (Mamba, RWKV, ...) |
| `model:is_hybrid()` | `bool` | Hybrid attention+SSM (Jamba, ...) |
| `model:is_diffusion()` | `bool` | Diffusion-based (LLaDA, Dream, ...) |
| `model:meta_count()` | `number` | GGUF key-value pair count |
| `model:meta_val(key)` | `string?` | Metadata value by key name |
| `model:meta_key_at(i)` | `string?` | Key name at index i (0-based) |
| `model:meta_val_at(i)` | `string?` | Value at index i (0-based) |
| `model:chat_template(name?)` | `string?` | Embedded Jinja template (nil if absent) |
| `model:save(path)` | `void` | Save to GGUF file |
| `model:lora_load(path)` | `LoraAdapter` | Load a LoRA adapter |

---

## ion7.core.Context

```lua
local Context = require "ion7.core.context"
```

### Decode

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:decode(tokens, n, seq_id?, pos_offset?)` | `→ void` | Chunked decode with KV accumulation. `seq_id` defaults to 0, `pos_offset` defaults to `n_past`. |
| `ctx:decode_single(token, seq_id?)` | `→ void` | Single-token decode using pre-allocated batch |
| `ctx:encode(tokens, n)` | `→ void` | Encoder pass (encoder-decoder models only) |
| `ctx:ptr()` | `→ cdata` | Raw `llama_context*` - pass to `sampler:sample()` |

### Dimensions

| Symbol | Returns | Notes |
|--------|---------|-------|
| `ctx:n_ctx()` | `number` | Context window size |
| `ctx:n_ctx_seq()` | `number` | Per-sequence context window |
| `ctx:n_batch()` | `number` | Logical max batch size |
| `ctx:n_ubatch()` | `number` | Physical micro-batch size |
| `ctx:n_seq_max()` | `number` | Max parallel sequences |
| `ctx:n_past()` | `number` | Current KV fill position |
| `ctx:n_threads()` | `number` | Current generation thread count |
| `ctx:pooling_type()` | `string` | `"none"\|"mean"\|"cls"\|"last"\|"rank"\|"unspecified"` |

### Thread control

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:set_n_threads(n, n_batch?)` | `→ void` | Dynamically change thread count |
| `ctx:set_embeddings(on)` | `→ void` | Toggle embedding extraction mode |
| `ctx:set_causal_attn(on)` | `→ void` | Toggle causal (true) vs bidirectional attention |
| `ctx:set_warmup(on)` | `→ void` | Enable/disable warmup mode |
| `ctx:synchronize()` | `→ void` | Wait for all GPU operations to complete |

### KV cache

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:kv_clear()` | `→ void` | Clear entire KV cache (all sequences) |
| `ctx:kv_seq_rm(seq, p0, p1)` | `→ bool` | Remove KV in range [p0, p1). -1 = infinity |
| `ctx:kv_seq_cp(src, dst, p0, p1)` | `→ void` | Copy KV range between sequences |
| `ctx:kv_seq_keep(seq)` | `→ void` | Remove all KV NOT in this sequence |
| `ctx:kv_seq_shift(seq_id, delta, p0?, p1?)` | `→ void` | Shift positions (sliding window). `p0` default 0, `p1` default -1 (end). |
| `ctx:kv_can_shift()` | `→ bool` | True if positions can be shifted |
| `ctx:kv_seq_pos_min(seq)` | `→ number` | Minimum position present (-1 if empty) |
| `ctx:kv_seq_pos_max(seq)` | `→ number` | Maximum position present (-1 if empty) |

### State persistence

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:snapshot()` | `→ string` | Serialize KV + logits to Lua string |
| `ctx:restore(blob)` | `→ bool` | Deserialize from snapshot blob |
| `ctx:save_state(path)` | `→ void` | Persist full state to file |
| `ctx:load_state(path)` | `→ void` | Load full state from file |
| `ctx:seq_state_size(seq_id?)` | `→ number` | Per-sequence state size in bytes |
| `ctx:seq_save_state(path, seq_id?)` | `→ bool` | Save one sequence's KV to file |
| `ctx:seq_load_state(path, seq_id?)` | `→ bool` | Load one sequence's KV from file |

### Logits + Embeddings (raw access)

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:logits(i?)` | `→ cdata` | `float*` logit array for position i (or all) |
| `ctx:embedding(seq_id?, dim?)` | `→ table?` | Pooled embedding for seq_id as Lua float array |
| `ctx:embeddings_seq(seq_id)` | `→ cdata` | Pooled embedding for sequence seq_id |
| `ctx:sampled_token(i)` | `→ number` | Token sampled at batch position i |

### Adapters

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:lora_apply(adapter, scale)` | `→ bool` | Apply LoRA (scale 0.0–1.0) |
| `ctx:lora_remove(adapter)` | `→ bool` | Remove LoRA adapter |
| `ctx:set_control_vector(data, n_embd, il_start, il_end)` | `→ bool` | Activation steering |
| `ctx:clear_control_vector()` | `→ void` | Remove control vector |
| `ctx:attach_threadpool(tp, tp_batch?)` | `→ void` | Attach shared threadpool |
| `ctx:detach_threadpool()` | `→ void` | Detach threadpool |
| `ctx:set_sampler(seq_id, smpl)` | `→ bool` | Per-sequence sampler (experimental) |

### Performance

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ctx:perf()` | `→ table` | `{t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval, n_reused, tokens_per_s}` |
| `ctx:perf_reset()` | `→ void` | Reset all performance counters |
| `ctx:perf_print()` | `→ void` | Print perf summary to stderr |
| `ctx:memory_breakdown()` | `→ void` | Print per-device memory usage to stderr |

---

## ion7.core.Vocab

```lua
local Vocab = require "ion7.core.vocab"
```

### Tokenization

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `vocab:tokenize(text, add_special?, parse_special?)` | `→ tokens, n` | `tokens` is `cdata int32_t*`, `n` is count |
| `vocab:detokenize(tokens, n, remove_special?, unparse_special?)` | `→ string` | Inverse of tokenize |
| `vocab:piece(token, special?)` | `→ string` | Single token → string piece |
| `vocab:n_tokens()` | `→ number` | Vocabulary size |
| `vocab:type()` | `→ string` | `"spm"\|"bpe"\|"wpm"\|"ugm"\|"rwkv"\|"plamo2"\|"none"` |

### Special tokens

| Symbol | Returns | Notes |
|--------|---------|-------|
| `vocab:bos()` | `number` | Beginning-of-sentence |
| `vocab:eos()` | `number` | End-of-sentence |
| `vocab:eot()` | `number` | End-of-turn |
| `vocab:sep()` | `number` | Sentence separator |
| `vocab:pad()` | `number` | Padding |
| `vocab:nl()` | `number` | Newline |
| `vocab:mask()` | `number` | Mask (BERT-style) |
| `vocab:cls()` | `number` | CLS classification token |
| `vocab:fim_pre()` | `number` | Fill-in-Middle prefix |
| `vocab:fim_suf()` | `number` | Fill-in-Middle suffix |
| `vocab:fim_mid()` | `number` | Fill-in-Middle middle |
| `vocab:fim_pad()` | `number` | FIM padding |
| `vocab:fim_rep()` | `number` | FIM repository |
| `vocab:fim_sep()` | `number` | FIM separator |
| `vocab:get_add_bos()` | `bool` | Model adds BOS automatically |
| `vocab:get_add_eos()` | `bool` | Model adds EOS automatically |
| `vocab:get_add_sep()` | `bool` | Model adds SEP automatically |

### Token classification

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `vocab:is_eog(token)` | `→ bool` | End-of-generation token (EOS, EOT, ...) |
| `vocab:is_control(token)` | `→ bool` | Control token |
| `vocab:score(token)` | `→ number` | Unigram probability score (SPM models) |
| `vocab:attr(token)` | `→ number` | Attribute bitmask (NORMAL, CONTROL, BYTE, ...) |
| `vocab:text(token)` | `→ string` | Raw token text from vocabulary |

### Chat templates

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `vocab:apply_template(messages, add_ass?, template?)` | `→ string` | Apply Jinja-like chat template. `messages` is `{{role,content}}` |
| `vocab:builtin_templates()` | `→ table` | List of built-in template names |

---

## ion7.core.Sampler

```lua
local Sampler = require "ion7.core.sampler"
```

### Builder (SamplerBuilder)

Create with `Sampler.chain()`, chain methods, finalize with `:build(vocab)`.

| Method | Signature | Algorithm |
|--------|-----------|-----------|
| `:greedy()` | | Always pick highest-logit token |
| `:dist(seed?)` | | Multinomial sampling from probabilities |
| `:top_k(k)` | | Keep top k tokens by logit |
| `:top_p(p, min_keep?)` | | Nucleus: keep tokens summing to probability p |
| `:min_p(p, min_keep?)` | | Keep tokens with p >= p * max_p |
| `:temp(t)` | | Divide logits by t (t=0 → greedy) |
| `:temp_dynamic(base, range, exp?)` | | Dynamic temperature (entropy-based) |
| `:typical(p, min_keep?)` | | Locally typical sampling |
| `:top_n_sigma(n)` | | Keep tokens within n standard deviations |
| `:xtc(p, t, min_keep?, seed?)` | | Exclude Top Choices |
| `:mirostat(n_vocab, seed?, tau?, eta?, m?)` | | Mirostat v1 |
| `:mirostat_v2(seed?, tau?, eta?)` | | Mirostat v2 |
| `:adaptive_p(target?, decay?, seed?)` | | Adaptive probability threshold |
| `:penalties(last_n?, repeat?, freq?, present?)` | | Repetition penalties |
| `:dry(vocab, n_ctx_train, mult?, base?, len?, last_n?, breakers?)` | | DRY repetition penalty |
| `:grammar(gbnf, root?, vocab?)` | | GBNF grammar-constrained sampling |
| `:grammar_lazy(gbnf, root?, vocab?, trigger_words?, trigger_tokens?, trigger_patterns?)` | | Lazy grammar (activates on trigger) |
| `:logit_bias(biases)` | `biases = {{token, bias}}` | Add fixed bias to specific tokens |
| `:infill(vocab)` | | Fill-in-Middle infill sampler |
| `:custom(cs)` | `cs: CustomSampler` | Lua-defined sampling logic |
| `:build(vocab)` | `→ Sampler` | Finalize chain |

### Sampler instance

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `sampler:sample(ctx_ptr, idx?)` | `→ token` | Sample + accept. `idx` is batch position (-1=last) |
| `sampler:accept(token)` | `→ void` | Update state without sampling |
| `sampler:reset()` | `→ void` | Reset all state (new conversation) |
| `sampler:clone()` | `→ Sampler` | Independent deep copy |
| `sampler:n()` | `→ number` | Number of samplers in chain |
| `sampler:get(i)` | `→ Sampler` | Sampler at 0-based index i |
| `sampler:name()` | `→ string` | Chain display name |
| `sampler:seed()` | `→ number` | Current RNG seed |
| `sampler:perf()` | `→ table` | `{t_sample_ms, n_sample}` |
| `sampler:perf_reset()` | `→ void` | Reset sampling perf counters |
| `sampler:perf_print()` | `→ void` | Print perf to stderr |

---

## ion7.core.CustomSampler

```lua
local CustomSampler = require "ion7.core.custom_sampler"
```

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `CustomSampler.new(name, callbacks)` | `→ CustomSampler` | `callbacks.apply` required |
| `cs:ptr()` | `→ cdata` | Raw `llama_sampler*` (NO GC finalizer - chain owns it) |
| `cs:name()` | `→ string` | Display name |

### Callback signatures

```lua
-- Required:
callbacks.apply = function(candidates, n)
    -- candidates: cdata llama_token_data* (fields: .id, .logit, .p)
    -- n: number of candidates
    -- Set candidates.selected = index  OR  return index
    -- Return value: selected index (0-based)
end

-- Optional:
callbacks.accept = function(token_id) end   -- called after acceptance
callbacks.reset  = function() end           -- called on chain reset
```

---

## ion7.core.Threadpool

```lua
local Threadpool = require "ion7.core.threadpool"
```

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Threadpool.new(n_threads)` | `→ Threadpool` | Create CPU thread pool. n_threads must be > 0 |
| `tp:n_threads()` | `→ number` | Thread count (stored at creation) |
| `tp:pause()` | `→ void` | Pause all worker threads |
| `tp:resume()` | `→ void` | Resume paused workers |
| `tp:ptr()` | `→ cdata` | Raw `ggml_threadpool_t` |
| `tp:free()` | `→ void` | Manual free. GC also handles. Idempotent. **Prefer explicit free in production.** |

Use `ctx:attach_threadpool(tp)` / `ctx:detach_threadpool()` to attach.

---

## LoRA Adapter

Returned by `model:lora_load(path)`.

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `adapter:meta_val(key)` | `→ string?` | Metadata value by key |
| `adapter:meta_count()` | `→ number` | Metadata key-value count |

Use `ctx:lora_apply(adapter, scale)` and `ctx:lora_remove(adapter)`.

---

## Bridge C API (ion7_bridge.h)

All `ion7_*` C functions declared in `bridge/ion7_bridge.h` are stable.
Direct `llama_*` calls via `ffi/types.lua` cdef are also stable but
not versioned here - they track llama.h master directly.

### Coverage summary

| Category | Functions |
|----------|-----------|
| Backend | `ion7_backend_init/free`, `ion7_set_log_level` |
| Capabilities | `ion7_supports_mmap/mlock/gpu_offload/rpc`, `ion7_max_devices/parallel_sequences` |
| Model | `ion7_model_load/load_splits/load_fd/free/save/quantize` + 20 introspection fns |
| Context | `ion7_context_create/embedding_context_create/free` |
| KV | `ion7_kv_clear/seq_rm/cp/keep/shift` |
| State | `ion7_state_size/get/set/save_file/load_file` + seq variants |
| LoRA | `ion7_lora_load/free/apply/remove/meta_val` |
| Perf | `ion7_perf_print/reset/get` |
| Chat | `ion7_chat_apply_template` (version-stable) |
| Threadpool | `ion7_threadpool_create/free/pause/resume/attach/detach` |
| Custom sampler | `ion7_sampler_create` + callback typedefs |
| VRAM fit | `ion7_params_fit` |

**llama.h coverage: 213/224 functions (95%)**  
Excluded: `llama_opt_*` (training), `llama_model_init_from_user` (needs `gguf_context*`).

---

## Memory management contract

ion7-core objects wrapping C resources expose `:free()` for explicit release:

| Object | Method | What it releases |
|--------|--------|-----------------|
| `Model` | `model:free()` | Model weights from RAM/VRAM |
| `Context` | `ctx:free()` | KV cache from VRAM, decode state |
| `Threadpool` | `tp:free()` | Worker threads |

**LuaJIT's GC is blind to C-side memory.** A context holding 4GB of KV cache
appears as ~24 bytes to Lua. In production loops, always call `:free()`
explicitly rather than relying on GC finalization.

`collectgarbage("collect")` is called automatically inside `Model.load()` and
`Context.new()` as a defensive measure, but this is not a substitute for
explicit resource management.

## What downstream modules may depend on

Downstream modules (`ion7-llm`, `ion7-embed`, etc.) MUST:
- Only use methods listed in this document
- Use `ion7.capabilities()` to detect optional features at runtime
- Never call `llama_*` or `ion7_*` functions directly (use the Lua objects)

Downstream modules MUST NOT:
- Add methods to `ion7.core.*` metatables (use composition)
- Depend on internal fields (`_ptr`, `_lib`, `_bridge`)
- Assume specific bridge version beyond `1.x`
