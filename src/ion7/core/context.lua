--- @module ion7.core.context
--- SPDX-License-Identifier: MIT
--- Inference context: decoding, KV cache management, state persistence.
---
--- A Context wraps a llama_context* created by ion7_context_create().
--- It is the central object for all inference operations.
---
--- Resource management: the underlying llama_context* is freed automatically
--- when the Context object is garbage-collected.
---
--- A single llama_model may host multiple simultaneous Context instances.
--- Contexts are NOT thread-safe -- use one Context per thread.
---
--- @usage
---   local ctx = model:context({ n_ctx = 65536, kv_type = "q8_0" })
---
---   -- Decode a batch of tokens
---   ctx:decode(tokens, n_tokens, seq_id)
---
---   -- Sample next token
---   local token = sampler:sample(ctx:ptr(), 0)

local Loader = require "ion7.core.ffi.loader"

-- ── Context ───────────────────────────────────────────────────────────────────

--- @class Context
--- @field _ptr     cdata  llama_context* (freed on GC).
--- @field _lib     cdata  libllama.so namespace.
--- @field _bridge  cdata  ion7_bridge.so namespace.
--- @field _ffi     cdata  ffi module.
--- @field _n_past  number Current KV cache fill position.
local Context = {}
Context.__index = Context

--- Wrap a raw llama_context pointer.
--- Prefer model:context() over calling this directly.
---
--- @param  lib     cdata  libllama.so FFI namespace.
--- @param  bridge  cdata  ion7_bridge.so namespace.
--- @param  ptr     cdata  llama_context* (will be freed on GC).
--- @--- Attach a custom threadpool to this context.
--- The threadpool is used for all subsequent decodes.
--- Create one with llama.Threadpool.new(n_threads).
---
--- @param  tp        cdata  ggml_threadpool_t
--- @param  tp_batch  cdata? Separate pool for batch processing (defaults to tp).
function Context:attach_threadpool(tp, tp_batch)
    -- tp can be a Threadpool object (with ._ptr) or a raw cdata pointer
    local p  = type(tp) == "table" and tp._ptr or tp
    local pb = tp_batch and (type(tp_batch) == "table" and tp_batch._ptr or tp_batch) or p
    self._bridge.ion7_threadpool_attach(self._ptr, p, pb)
end

--- Detach the current threadpool from this context.
--- The context reverts to its internal auto-managed pool.
function Context:detach_threadpool()
    self._bridge.ion7_threadpool_detach(self._ptr)
end

--- Assign a specific sampler to a sequence (experimental).
--- Allows different sequences to use different sampling strategies
--- within the same context (useful with MultiSeq).
---
--- @param  seq_id  number   Sequence ID.
--- @param  smpl    cdata    llama_sampler* (from Sampler:ptr() or Sampler:build()).
--- @return bool


--- Print a detailed memory breakdown to stderr (debugging).
function Context:memory_breakdown()
    self._lib.llama_memory_breakdown_print(self._ptr)
end

--- Apply a control vector (activation steering) to this context.
--- @param  data     table   float[] of size n_embd * n_layers.
--- @param  n_embd   number  Embedding dimension.
--- @param  il_start number  Start layer (inclusive).
--- @param  il_end   number  End layer (inclusive).
--- @return bool
function Context:set_control_vector(data, n_embd, il_start, il_end)
    local ffi = self._ffi
    local n   = #data
    local buf = ffi.new("float[?]", n)
    for i, v in ipairs(data) do buf[i - 1] = v end
    return self._lib.llama_set_adapter_cvec(
        self._ptr, buf, n, n_embd, il_start, il_end) == 0
end

--- Attach a sampler to a specific sequence in this context. [EXPERIMENTAL]
--- When set, llama_decode() will automatically sample using this sampler
--- for the given sequence ID. Useful with MultiSeq for per-sequence sampling.
---
--- @param  seq_id  number  Sequence ID.
--- @param  sampler Sampler Sampler chain (must remain alive).
--- @return bool    true on success.
function Context:set_sampler(seq_id, sampler)
    local chain = sampler._chain or sampler._ptr
    assert(chain ~= nil, "[ion7.core.context] set_sampler: invalid sampler")
    return self._lib.llama_set_sampler(self._ptr, seq_id, chain)
end

--- Clear any active control vector.
function Context:clear_control_vector()
    self._lib.llama_set_adapter_cvec(self._ptr, nil, 0, 0, 0, 0)
end

function Context.new(lib, bridge, ptr)
    -- Hint GC before acquiring potentially GB-scale C resources
    collectgarbage("collect")
    assert(ptr ~= nil, "[ion7.core.context] context pointer is NULL")
    local L   = Loader.instance()
    local ffi = L.ffi
    -- Pre-allocate a single-token batch for decode_single().
    -- Avoids one malloc+free per generated token (100+ per response).
    local db = lib.llama_batch_init(1, 0, 1)

    return setmetatable({
        _ptr          = ffi.gc(ptr, bridge.ion7_context_free),
        _lib          = lib,
        _bridge       = bridge,
        _ffi          = ffi,
        _n_past       = 0,
        _decode_batch = db,  -- reused by decode_single()
    }, Context)
end

--- Return the raw llama_context* (for passing to sampler:sample() etc.).
--- @return cdata
--- Explicitly free the context and release VRAM immediately.
--- Normally handled by GC, but call this in tight loops to avoid OOM.
function Context:free()
    if self._ptr then
        self._bridge.ion7_context_free(self._ptr)
        -- Disarm the GC finalizer to prevent double-free
        self._ptr = self._ffi.gc(self._ptr, nil)
        self._ptr = nil
    end
end

function Context:ptr()
    return self._ptr
end

--- @return number  Context window size in tokens.
function Context:n_ctx()
    return tonumber(self._lib.llama_n_ctx(self._ptr))
end

--- @return number  Batch size.
function Context:n_batch()
    return tonumber(self._lib.llama_n_batch(self._ptr))
end

function Context:n_ubatch()
    return tonumber(self._lib.llama_n_ubatch(self._ptr))
end

--- @return number  Current KV cache fill position.
function Context:n_past()
    return self._n_past
end

--- @return number  Context window per sequence.
function Context:n_ctx_seq()
    return tonumber(self._lib.llama_n_ctx_seq(self._ptr))
end

--- @return number  Maximum number of sequences.
function Context:n_seq_max()
    return tonumber(self._lib.llama_n_seq_max(self._ptr))
end

--- @return number  Current thread count for generation.
function Context:n_threads()
    return tonumber(self._lib.llama_n_threads(self._ptr))
end

--- @return number  Current thread count for batch processing.
function Context:n_threads_batch()
    return tonumber(self._lib.llama_n_threads_batch(self._ptr))
end

--- Dynamically change thread counts without recreating the context.
--- @param  n_threads       number
--- @param  n_threads_batch number?  Defaults to n_threads.
function Context:set_n_threads(n_threads, n_threads_batch)
    self._lib.llama_set_n_threads(self._ptr, n_threads, n_threads_batch or n_threads)
end

--- Toggle embedding extraction mode.
--- @param  on  bool
function Context:set_embeddings(on)
    self._lib.llama_set_embeddings(self._ptr, on)
end

--- Toggle causal attention (false = full bidirectional, for embeddings).
--- @param  on  bool
function Context:set_causal_attn(on)
    self._lib.llama_set_causal_attn(self._ptr, on)
end

--- Enable/disable warmup mode (skips some operations during warmup pass).
--- @param  on  bool
function Context:set_warmup(on)
    self._lib.llama_set_warmup(self._ptr, on)
end

--- Synchronize GPU operations (ensure all async GPU work is complete).
function Context:synchronize()
    self._lib.llama_synchronize(self._ptr)
end

--- Get the current pooling type of this context.
--- @return string  "none" | "mean" | "cls" | "last" | "rank"
function Context:pooling_type()
    local t = self._lib.llama_pooling_type(self._ptr)
    local map = { [-1]="unspecified", [0]="none", [1]="mean", [2]="cls", [3]="last", [4]="rank" }
    return map[tonumber(t)] or "unknown"
end

--- Get the sampled token at batch position i (after decode+sample).
--- @param  i  number  Batch position.
--- @return number  Token ID, or -1 if not available.
function Context:sampled_token(i)
    return tonumber(self._lib.llama_get_sampled_token_ith(self._ptr, i or 0))
end

-- ── Decode ────────────────────────────────────────────────────────────────────

--- Decode a sequence of tokens and update the KV cache.
---
--- Handles chunking automatically when the batch is larger than n_batch.
--- The last token in the sequence has its logits enabled for sampling.
---
--- @param  tokens      cdata    int32_t array of token IDs.
--- @param  n_tokens    number   Number of tokens to decode.
--- @param  seq_id      number?  Sequence ID (default: 0).
--- @param  pos_offset  number?  Starting KV position (default: ctx:n_past()).
--- @return number  Last chunk size (for sample index calculation).
--- @error  If decoding fails.
function Context:decode(tokens, n_tokens, seq_id, pos_offset)
    seq_id     = seq_id     or 0
    pos_offset = pos_offset or self._n_past

    local lib     = self._lib
    local ffi     = self._ffi
    local n_batch = self:n_batch()
    local done    = 0
    local last_chunk = 0

    while done < n_tokens do
        local chunk    = math.min(n_batch, n_tokens - done)
        local is_last  = (done + chunk >= n_tokens)
        local batch    = lib.llama_batch_init(chunk, 0, 1)

        for i = 0, chunk - 1 do
            batch.token[i]     = tokens[done + i]
            batch.pos[i]       = pos_offset + done + i
            batch.n_seq_id[i]  = 1
            batch.seq_id[i][0] = seq_id
            -- Enable logits only for the last token of the last chunk
            batch.logits[i]    = (is_last and i == chunk - 1) and 1 or 0
        end
        batch.n_tokens = chunk

        local ret = lib.llama_decode(self._ptr, batch)
        lib.llama_batch_free(batch)

        if ret ~= 0 then
            if ret == 1 then
                error("[ion7.core.context] KV cache is full", 2)
            end
            error(string.format(
                "[ion7.core.context] llama_decode returned %d", ret), 2)
        end

        done       = done + chunk
        last_chunk = chunk
    end

    self._n_past = pos_offset + n_tokens
    return last_chunk
end

--- Decode a single token (used during generation loop).
--- Uses a pre-allocated batch (set up in Context.new) to avoid
--- one malloc+free per token.
---
--- @param  token   number  Token ID.
--- @param  seq_id  number? Sequence ID (default: 0).
--- @error  If decoding fails.
function Context:decode_single(token, seq_id)
    seq_id = seq_id or 0
    local lib = self._lib
    local b   = self._decode_batch
    b.token[0]     = token
    b.pos[0]       = self._n_past
    b.n_seq_id[0]  = 1
    b.seq_id[0][0] = seq_id
    b.logits[0]    = 1
    b.n_tokens     = 1

    local ret = lib.llama_decode(self._ptr, b)
    if ret ~= 0 then
        error(string.format(
            "[ion7.core.context] single decode failed: %d", ret), 2)
    end
    self._n_past = self._n_past + 1
end

-- ── Logits ────────────────────────────────────────────────────────────────────

--- Return the logit array for position idx.
---
--- @param  idx  number  Batch position (usually 0 after decode_single).
--- @return cdata  float* pointer into the context's logit buffer (not owned).
function Context:logits(idx)
    return self._lib.llama_get_logits_ith(self._ptr, idx)
end

--- Compute the log-probability of a specific token at position idx.
---
--- @param  idx    number  Batch position.
--- @param  token  number  Token ID.
--- @return number  Log-probability (negative, closer to 0 = more likely).
function Context:logprob(idx, token)
    local ptr = self._lib.llama_get_logits_ith(self._ptr, idx)
    local n   = self._n_vocab or (self._lib.llama_n_vocab and tonumber(self._lib.llama_n_vocab(self._ptr)) or 32000)
    -- Compute log-softmax
    local max = -math.huge
    for i = 0, n - 1 do
        local v = tonumber(ptr[i])
        if v > max then max = v end
    end
    local sum = 0.0
    for i = 0, n - 1 do
        sum = sum + math.exp(tonumber(ptr[i]) - max)
    end
    return tonumber(ptr[token]) - max - math.log(sum)
end

--- Compute the Shannon entropy of the logit distribution at position idx.

---
--- @param  idx  number  Batch position.
--- @return number  Entropy in nats.
function Context:entropy(idx)
    local ptr = self._lib.llama_get_logits_ith(self._ptr, idx)
    local n   = self._n_vocab or (self._lib.llama_n_vocab and tonumber(self._lib.llama_n_vocab(self._ptr)) or 32000)
    local max = -math.huge
    for i = 0, n - 1 do
        local v = tonumber(ptr[i])
        if v > max then max = v end
    end
    local sum = 0.0
    local probs = {}
    for i = 0, n - 1 do
        local p = math.exp(tonumber(ptr[i]) - max)
        probs[i + 1] = p
        sum = sum + p
    end
    local entropy = 0.0
    for _, p in ipairs(probs) do
        if p > 0 then
            local pn = p / sum
            entropy = entropy - pn * math.log(pn)
        end
    end
    return entropy
end

-- ── Embeddings ────────────────────────────────────────────────────────────────

--- Retrieve the pooled embedding for sequence seq_id.
--- Only valid for embedding contexts (pooling != NONE).
---
--- @param  seq_id  number?  Sequence ID (default: 0).
--- @param  dim     number?  Embedding dimension (default: auto from model).
--- @return table   float[] Lua array of floats, or nil if not available.
function Context:embedding(seq_id, dim)
    seq_id = seq_id or 0
    local ptr = self._lib.llama_get_embeddings_seq(self._ptr, seq_id)
    if ptr == nil then
        -- Fallback: last token embedding
        ptr = self._lib.llama_get_embeddings_ith(self._ptr, -1)
    end
    if ptr == nil then return nil end

    dim = dim or 4096  -- caller should provide correct value
    local vec = {}
    for i = 0, dim - 1 do
        vec[i + 1] = tonumber(ptr[i])
    end
    return vec
end

-- ── KV cache ──────────────────────────────────────────────────────────────────

--- Check whether the KV cache supports position shifting.
--- Returns false for recurrent models (Mamba, RWKV) -- never call kv_seq_shift on them.
--- @return bool
function Context:kv_can_shift()
    local mem = self._lib.llama_get_memory(self._ptr)
    return mem ~= nil and self._lib.llama_memory_can_shift(mem)
end

--- Return the smallest KV cache position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default: 0).
--- @return number
function Context:kv_seq_pos_min(seq_id)
    local mem = self._lib.llama_get_memory(self._ptr)
    return mem and tonumber(self._lib.llama_memory_seq_pos_min(mem, seq_id or 0)) or -1
end

--- Return the largest KV cache position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default: 0).
--- @return number
function Context:kv_seq_pos_max(seq_id)
    local mem = self._lib.llama_get_memory(self._ptr)
    return mem and tonumber(self._lib.llama_memory_seq_pos_max(mem, seq_id or 0)) or -1
end

--- Divide all positions in a sequence by d (for context compression).
--- @param  seq_id  number
--- @param  d       number  Divisor > 1.
--- @param  p0      number? Start (default: 0).
--- @param  p1      number? End (default: -1).
function Context:kv_seq_div(seq_id, d, p0, p1)
    local mem = self._lib.llama_get_memory(self._ptr)
    if mem then self._lib.llama_memory_seq_div(mem, seq_id, p0 or 0, p1 or -1, d) end
end

--- Clear the entire KV cache and reset n_past to 0.
--- Must be called between independent conversations.
function Context:kv_clear()
    self._bridge.ion7_kv_clear(self._ptr)
    self._n_past = 0
end

--- Remove tokens in position range [p0, p1) for a sequence.
--- Use p1 = -1 to remove from p0 to the end.
---
--- @param  seq_id  number  Sequence ID.
--- @param  p0      number  Start position (inclusive).
--- @param  p1      number  End position (exclusive). -1 = end.
--- @return bool  true if the sequence was fully removed.
function Context:kv_seq_rm(seq_id, p0, p1)
    -- llama_memory_seq_rm returns bool (false = partial sequence, can't remove)
    local mem = self._lib.llama_get_memory(self._ptr)
    if not mem then return false end
    return self._lib.llama_memory_seq_rm(mem, seq_id, p0, p1 or -1)
end

--- Copy a range of KV cache entries from src to dst sequence.
---
--- @param  src_seq  number
--- @param  dst_seq  number
--- @param  p0       number
--- @param  p1       number  -1 = end.
function Context:kv_seq_cp(src_seq, dst_seq, p0, p1)
    self._bridge.ion7_kv_seq_cp(self._ptr, src_seq, dst_seq, p0, p1 or -1)
end

--- Keep only seq_id; remove all other sequences from the KV cache.
--- @param  seq_id  number
function Context:kv_seq_keep(seq_id)
    self._bridge.ion7_kv_seq_keep(self._ptr, seq_id)
end

--- Shift all positions in a sequence by delta.
--- Used for sliding-window context management.
---
--- @param  seq_id  number
--- @param  delta   number  Shift amount (negative to move backward).
--- @param  p0      number? Start (default: 0).
--- @param  p1      number? End (default: -1).
function Context:kv_seq_shift(seq_id, delta, p0, p1)
    self._bridge.ion7_kv_seq_shift(
        self._ptr, seq_id, p0 or 0, p1 or -1, delta)
end

-- ── State persistence ─────────────────────────────────────────────────────────

--- Save the complete context state (KV cache + RNG) to a file.
--- Loading this file later restores the context exactly, enabling warm starts.
---
--- @param  path     string   Output file path.
--- @param  tokens   cdata?   Token sequence to embed in the file (may be nil).
--- @param  n_tokens number?  Length of tokens array.
--- @return bool  true on success.
function Context:save_state(path, tokens, n_tokens)
    return self._bridge.ion7_state_save_file(
        self._ptr, path,
        tokens or nil,
        n_tokens or 0
    ) == 1
end

--- Restore context state from a file previously saved via save_state().
---
--- @param  path          string  Input file path.
--- @param  token_buf     cdata?  Buffer to receive the stored token sequence.
--- @param  token_capacity number? Capacity of token_buf.
--- @return bool, number  success, number of tokens restored.
function Context:load_state(path, token_buf, token_capacity)
    local ffi  = self._ffi
    local n_out = ffi.new("size_t[1]", 0)
    local ok    = self._bridge.ion7_state_load_file(
        self._ptr, path,
        token_buf or nil,
        token_capacity or 0,
        n_out
    ) == 1
    -- Restore n_past from the saved token count
    if ok and tonumber(n_out[0]) > 0 then
        self._n_past = tonumber(n_out[0])
    end
    return ok, tonumber(n_out[0])
end

--- Serialise state to an in-memory Lua string.
--- Useful for session pooling without hitting the filesystem.
---
--- @return string  Binary blob, or nil on failure.
function Context:snapshot()
    local bridge = self._bridge
    local ffi    = self._ffi
    local sz     = tonumber(bridge.ion7_state_size(self._ptr))
    if sz == 0 then return nil end
    local buf    = ffi.new("uint8_t[?]", sz)
    local written = tonumber(bridge.ion7_state_get(self._ptr, buf, sz))
    if written == 0 then return nil end
    return ffi.string(buf, written)
end

--- Restore state from a snapshot string produced by snapshot().
---
--- @param  blob  string  Binary snapshot blob.
--- @return bool
function Context:restore(blob)
    if not blob or #blob == 0 then return false end
    local ffi = self._ffi
    local buf = ffi.new("uint8_t[?]", #blob)
    ffi.copy(buf, blob, #blob)
    return self._bridge.ion7_state_set(self._ptr, buf, #blob) == 1
end

-- ── Per-sequence state ───────────────────────────────────────────────────────

--- Return the size in bytes of the KV state for a single sequence.
--- @param  seq_id  number?  Sequence ID (default: 0).
--- @return number
function Context:seq_state_size(seq_id)
    return tonumber(self._bridge.ion7_state_seq_size(self._ptr, seq_id or 0))
end

--- Save the KV state for a single sequence to a file.
--- Useful for multi-session caching without saving the entire context.
--- @param  path    string
--- @param  seq_id  number?  Sequence ID (default: 0).
--- @return bool
function Context:seq_save_state(path, seq_id)
    return self._bridge.ion7_state_seq_save(self._ptr, path, seq_id or 0) == 1
end

--- Load the KV state for a single sequence from a file.
--- @param  path        string
--- @param  dst_seq_id  number?  Target sequence ID (default: 0).
--- @return bool
function Context:seq_load_state(path, dst_seq_id)
    return self._bridge.ion7_state_seq_load(self._ptr, path, dst_seq_id or 0) == 1
end

-- ── Performance ───────────────────────────────────────────────────────────────

--- Print performance counters to stderr.
function Context:perf_print()
    self._bridge.ion7_perf_print(self._ptr)
end

--- Reset performance counters.
function Context:perf_reset()
    self._bridge.ion7_perf_reset(self._ptr)
end

--- Return performance data as a Lua table.
--- @return table  { t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval, tokens_per_s }
function Context:perf()
    local data = self._lib.llama_perf_context(self._ptr)
    local n_eval = tonumber(data.n_eval)
    local t_eval = tonumber(data.t_eval_ms)
    return {
        t_load_ms    = tonumber(data.t_load_ms),
        t_p_eval_ms  = tonumber(data.t_p_eval_ms),
        t_eval_ms    = t_eval,
        n_p_eval     = tonumber(data.n_p_eval),
        n_eval       = n_eval,
        n_reused     = tonumber(data.n_reused),
        tokens_per_s = (n_eval > 0 and t_eval > 0)
                       and (n_eval / (t_eval / 1000.0))
                       or 0.0,
    }
end

return Context
