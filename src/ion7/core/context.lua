--- @module ion7.core.context
--- SPDX-License-Identifier: MIT
--- Inference context: decoding, KV cache management, state persistence.
---

local ffi = require "ffi"

-- ── Pre-parsed ctypes (parse cdecl once, reuse as constructor) ───────────────
local _u8arr = ffi.typeof("uint8_t[?]")
local _f32arr = ffi.typeof("float[?]")
local _sz1   = ffi.typeof("size_t[1]")

-- ── Module-level constants ──────────────

local POOLING_NAMES = {
    [-1] = "unspecified",
    [0]  = "none",
    [1]  = "mean",
    [2]  = "cls",
    [3]  = "last",
    [4]  = "rank",
}

-- ── Context ───────────────────────────────────────────────────────────────────

--- @class Context
--- @field _ptr     cdata   llama_context* (freed on GC via ffi.gc).
--- @field _lib     cdata   libllama.so namespace.
--- @field _bridge  cdata   ion7_bridge.so namespace.
--- @field _n_past  number  Current KV cache fill position (tracks how many tokens decoded).
--- @field _n_batch number  Batch capacity (cached at creation, immutable).
--- @field _n_ctx   number  Context window size (cached, immutable).
--- @field _mem     cdata   llama_memory_t (cached, immutable for context lifetime).
--- @field _decode_batch llama_batch  Pre-allocated batch (n_batch capacity), reused forever.
local Context = {}
Context.__index = Context

-- ── Constructor ───────────────────────────────────────────────────────────────

--- Wrap a raw llama_context pointer.
--- Prefer model:context() over calling this directly.
---
--- @param  lib     cdata  libllama.so FFI namespace.
--- @param  bridge  cdata  ion7_bridge.so FFI namespace.
--- @param  ptr     cdata  llama_context* (will be freed on GC).
--- @return Context
function Context.new(lib, bridge, ptr)
    collectgarbage("collect")
    assert(ptr ~= nil, "[ion7.core.context] context pointer is NULL")

    -- Cache immutable properties once - avoids an FFI call on every getter.
    local n_batch = tonumber(lib.llama_n_batch(ptr))
    local n_ctx   = tonumber(lib.llama_n_ctx(ptr))
    local n_ubatch = tonumber(lib.llama_n_ubatch(ptr))
    local n_seq_max = tonumber(lib.llama_n_seq_max(ptr))

    -- Cache memory handle: KV ops use it directly (no llama_get_memory() per call).
    local mem = lib.llama_get_memory(ptr)

    -- Pre-allocate a decode batch with full n_batch capacity.
    -- Reused by decode(), decode_single(), and decode_multi() - zero malloc at runtime.
    local db = lib.llama_batch_init(n_batch, 0, 1)
    -- Pre-initialize n_seq_id = 1 for all positions (constant, never changes).
    for i = 0, n_batch - 1 do db.n_seq_id[i] = 1 end

    -- GC sentinel: frees the batch internal buffers when the Context is collected.
    -- ffi.gc on a 1-byte dummy cdata - reliable across all LuaJIT versions.
    local batch_gc = ffi.gc(ffi.new("int8_t[1]"), function()
        lib.llama_batch_free(db)
    end)

    return setmetatable({
        _ptr          = ffi.gc(ptr, bridge.ion7_context_free),
        _lib          = lib,
        _bridge       = bridge,
        _n_past       = 0,
        _n_batch      = n_batch,
        _n_ctx        = n_ctx,
        _n_ubatch     = n_ubatch,
        _n_seq_max    = n_seq_max,
        _mem          = mem,
        _decode_batch = db,
        _batch_gc     = batch_gc,
    }, Context)
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────────

--- Return the raw llama_context* (for passing to sampler:sample() etc.).
--- @return cdata
function Context:ptr()
    return self._ptr
end

--- Explicitly free the context and release VRAM immediately.
--- Normally handled by GC. Call this in tight loops to avoid OOM.
function Context:free()
    if self._ptr then
        self._bridge.ion7_context_free(self._ptr)
        self._ptr = ffi.gc(self._ptr, nil)
        self._ptr = nil
    end
    -- Force immediate batch cleanup before the GC sentinel fires.
    if self._batch_gc then
        ffi.gc(self._batch_gc, nil)  -- disarm sentinel
        self._batch_gc = nil
        if self._decode_batch then
            self._lib.llama_batch_free(self._decode_batch)
            self._decode_batch = nil
        end
    end
end

-- ── Immutable properties (returned from cache, no FFI call) ──────────────────

--- @return number  Context window size.
function Context:n_ctx()      return self._n_ctx    end

--- @return number  Batch capacity.
function Context:n_batch()    return self._n_batch   end

--- @return number  Micro-batch size.
function Context:n_ubatch()   return self._n_ubatch  end

--- @return number  Current KV fill position.
function Context:n_past()     return self._n_past    end

--- @return number  Context window per sequence.
function Context:n_ctx_seq()
    return tonumber(self._lib.llama_n_ctx_seq(self._ptr))
end

--- @return number  Maximum concurrent sequences.
function Context:n_seq_max()  return self._n_seq_max end

-- ── Mutable properties ────────────────────────────────────────────────────────

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

--- @return string  "none"|"mean"|"cls"|"last"|"rank"|"unspecified"
function Context:pooling_type()
    local t = tonumber(self._lib.llama_pooling_type(self._ptr))
    return POOLING_NAMES[t] or "unknown"
end

--- Toggle embedding extraction mode.
function Context:set_embeddings(on)     self._lib.llama_set_embeddings(self._ptr, on)     end

--- Toggle causal attention (false = bidirectional, for embedding models).
function Context:set_causal_attn(on)   self._lib.llama_set_causal_attn(self._ptr, on)   end

--- Enable/disable warmup mode (skips perf counters during warmup pass).
function Context:set_warmup(on)        self._lib.llama_set_warmup(self._ptr, on)         end

--- Synchronize GPU operations (wait for all async GPU work to complete).
function Context:synchronize()         self._lib.llama_synchronize(self._ptr)            end

--- Print a detailed memory breakdown to stderr (debugging).
function Context:memory_breakdown()    self._lib.llama_memory_breakdown_print(self._ptr) end

--- Register an abort callback (called periodically during decode).
--- Return true from cb to abort the current decode.
--- @param  cb    cdata  C function pointer: bool(*)(void* data).
--- @param  data  cdata? User data passed to cb.
function Context:set_abort_callback(cb, data)
    self._lib.llama_set_abort_callback(self._ptr, cb, data or nil)
end

-- ── Decode ────────────────────────────────────────────────────────────────────

--- Decode a sequence of tokens and update the KV cache.
---
--- Uses a pre-allocated batch (n_batch capacity) - no malloc per call.
--- Handles chunking automatically when n_tokens > n_batch.
--- Only the last token of the last chunk has logits enabled.
---
--- Two inner loops (cdata vs Lua table) avoid a per-token branch in the JIT.
---
--- @param  tokens      cdata|table  int32_t array (0-based cdata) or Lua table (1-based).
--- @param  n_tokens    number?      Token count (required for cdata; auto for table).
--- @param  seq_id      number?      Sequence ID (default 0).
--- @param  pos_offset  number?      Starting KV position (default n_past).
--- @return number  Last chunk size (used as sample index for sampler:sample()).
--- @error  On KV cache full or decode error.
function Context:decode(tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens   = n_tokens   or (is_table and #tokens or 0)
    seq_id     = seq_id     or 0
    pos_offset = pos_offset or self._n_past

    local lib     = self._lib
    local n_batch = self._n_batch
    local b       = self._decode_batch
    -- Pre-load struct field pointers once - avoids repeated member dereferences
    -- inside the tight inner loop, which the JIT cannot hoist automatically.
    local tok_ptr = b.token
    local pos_ptr = b.pos
    local seq_ptr = b.seq_id
    local log_ptr = b.logits
    local done    = 0
    local last_n  = 0

    if is_table then
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local end_i  = n - 1
            local base_p = pos_offset + done
            -- ffi.fill maps to an inlinable SIMD memset - faster than a Lua loop
            ffi.fill(log_ptr, n)   -- zero n int8_t elements
            for i = 0, end_i do
                tok_ptr[i]    = tokens[done + i + 1]   -- Lua: 1-based
                pos_ptr[i]    = base_p + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then log_ptr[end_i] = 1 end
            b.n_tokens = n
            local ret = lib.llama_decode(self._ptr, b)
            if ret ~= 0 then
                if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
                error(string.format("[ion7.core.context] llama_decode returned %d", ret), 2)
            end
            done   = done + n
            last_n = n
        end
    else
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local end_i  = n - 1
            local base_p = pos_offset + done
            ffi.fill(log_ptr, n)
            for i = 0, end_i do
                tok_ptr[i]    = tokens[done + i]        -- cdata: 0-based
                pos_ptr[i]    = base_p + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then log_ptr[end_i] = 1 end
            b.n_tokens = n
            local ret = lib.llama_decode(self._ptr, b)
            if ret ~= 0 then
                if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
                error(string.format("[ion7.core.context] llama_decode returned %d", ret), 2)
            end
            done   = done + n
            last_n = n
        end
    end

    self._n_past = pos_offset + n_tokens
    return last_n
end

--- Decode multiple tokens with ALL logits enabled (for speculative verification).
---
--- After the call, sampler:sample(ctx:ptr(), i) is valid for i = 0 .. n-1.
---
--- @param  tokens  table   Lua 1-based array of token IDs.
--- @param  seq_id  number? Sequence ID (default 0).
--- @error  If decoding fails.
function Context:decode_multi(tokens, seq_id)
    local n = #tokens
    assert(n > 0, "[ion7.core.context] decode_multi: empty token list")
    seq_id    = seq_id or 0
    local lib = self._lib
    local b   = self._decode_batch
    local base = self._n_past
    for i = 0, n - 1 do
        b.token[i]     = tokens[i + 1]
        b.pos[i]       = base + i
        b.seq_id[i][0] = seq_id
        b.logits[i]    = 1
    end
    b.n_tokens = n
    local ret = lib.llama_decode(self._ptr, b)
    if ret ~= 0 then
        if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
        error(string.format("[ion7.core.context] decode_multi failed: %d", ret), 2)
    end
    self._n_past = base + n
end

--- Decode a single token (tight generation loop path).
---
--- Uses the pre-allocated batch - zero malloc per call.
---
--- @param  token   number  Token ID.
--- @param  seq_id  number? Sequence ID (default 0).
--- @error  If decoding fails.
function Context:decode_single(token, seq_id)
    local lib = self._lib
    local b   = self._decode_batch
    b.token[0]     = token
    b.pos[0]       = self._n_past
    b.seq_id[0][0] = seq_id or 0
    b.logits[0]    = 1
    b.n_tokens     = 1
    local ret = lib.llama_decode(self._ptr, b)
    if ret ~= 0 then
        error(string.format("[ion7.core.context] decode_single failed: %d", ret), 2)
    end
    self._n_past = self._n_past + 1
end

--- Encode a sequence (for encoder-decoder models: T5, BART, etc.).
---
--- @param  tokens      cdata|table
--- @param  n_tokens    number?
--- @param  seq_id      number?
--- @param  pos_offset  number?
--- @error  If encoding fails.
function Context:encode(tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens   = n_tokens   or (is_table and #tokens or 0)
    seq_id     = seq_id     or 0
    pos_offset = pos_offset or self._n_past

    local lib     = self._lib
    local n_batch = self._n_batch
    local b       = self._decode_batch
    local done    = 0

    if is_table then
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local base_p = pos_offset + done
            for i = 0, n - 1 do
                b.token[i]     = tokens[done + i + 1]
                b.pos[i]       = base_p + i
                b.seq_id[i][0] = seq_id
                b.logits[i]    = 1
            end
            b.n_tokens = n
            local ret = lib.llama_encode(self._ptr, b)
            if ret ~= 0 then
                error(string.format("[ion7.core.context] llama_encode returned %d", ret), 2)
            end
            done = done + n
        end
    else
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local base_p = pos_offset + done
            for i = 0, n - 1 do
                b.token[i]     = tokens[done + i]
                b.pos[i]       = base_p + i
                b.seq_id[i][0] = seq_id
                b.logits[i]    = 1
            end
            b.n_tokens = n
            local ret = lib.llama_encode(self._ptr, b)
            if ret ~= 0 then
                error(string.format("[ion7.core.context] llama_encode returned %d", ret), 2)
            end
            done = done + n
        end
    end
    self._n_past = pos_offset + n_tokens
end

-- ── Logits & sampling results ─────────────────────────────────────────────────

--- Return the raw logit array for batch position idx.
--- @param  idx  number  Batch position (usually 0 after decode_single).
--- @return cdata  float* into the context's logit buffer (not owned, do not free).
function Context:logits(idx)
    return self._lib.llama_get_logits_ith(self._ptr, idx or 0)
end

--- Compute log-probability of token_id at batch position idx.
--- Delegates to C (ion7_logprob) - iterates n_vocab floats ~50x faster than Lua.
---
--- @param  idx       number  Batch position.
--- @param  token_id  number  Token to evaluate.
--- @return number  Log-probability (negative; closer to 0 = more likely).
function Context:logprob(idx, token_id)
    return tonumber(self._bridge.ion7_logprob(self._ptr, idx, token_id))
end

--- Compute Shannon entropy of the logit distribution at batch position idx.
--- Delegates to C (ion7_entropy).
---
--- @param  idx  number  Batch position.
--- @return number  Entropy in nats (>= 0).
function Context:entropy(idx)
    return tonumber(self._bridge.ion7_entropy(self._ptr, idx or 0))
end

--- Get the sampled token at batch position i (after decode + sampler:sample()).
--- @param  i  number?  Batch position (default 0).
--- @return number  Token ID, or -1 if not available.
function Context:sampled_token(i)
    return tonumber(self._lib.llama_get_sampled_token_ith(self._ptr, i or 0))
end

--- Return the probability array for the sampled token at position i.
--- @param  i  number?  Batch position.
--- @return cdata  float* (raw, not owned). May be NULL.
function Context:sampled_probs(i)
    return self._lib.llama_get_sampled_probs_ith(self._ptr, i or 0)
end

--- Return the count of probabilities in sampled_probs(i).
--- @param  i  number?
--- @return number
function Context:sampled_probs_count(i)
    return tonumber(self._lib.llama_get_sampled_probs_count_ith(self._ptr, i or 0))
end

--- Return the logit array for the sampled candidates at position i.
--- @param  i  number?
--- @return cdata  float* (raw, not owned).
function Context:sampled_logits(i)
    return self._lib.llama_get_sampled_logits_ith(self._ptr, i or 0)
end

--- Return the count of logits in sampled_logits(i).
--- @param  i  number?
--- @return number
function Context:sampled_logits_count(i)
    return tonumber(self._lib.llama_get_sampled_logits_count_ith(self._ptr, i or 0))
end

--- Return the candidate token array at position i.
--- @param  i  number?
--- @return cdata  llama_token* (raw, not owned).
function Context:sampled_candidates(i)
    return self._lib.llama_get_sampled_candidates_ith(self._ptr, i or 0)
end

--- Return the count of candidates in sampled_candidates(i).
--- @param  i  number?
--- @return number
function Context:sampled_candidates_count(i)
    return tonumber(self._lib.llama_get_sampled_candidates_count_ith(self._ptr, i or 0))
end

-- ── Embeddings ────────────────────────────────────────────────────────────────

--- Return the raw embedding pointer for a sequence (zero-copy).
--- Caller must not free this pointer and must not outlive the context.
---
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return cdata  float* or nil.
function Context:embedding_ptr(seq_id)
    local lib = self._lib
    local ptr = lib.llama_get_embeddings_seq(self._ptr, seq_id or 0)
    if ptr == nil then ptr = lib.llama_get_embeddings_ith(self._ptr, -1) end
    return ptr
end

--- Return the pooled embedding as a Lua table (copies floats from C).
---
--- @param  seq_id  number?  Sequence ID (default 0).
--- @param  dim     number?  Embedding dimension (must match model:n_embd()).
--- @return table   float[] (1-based), or nil if not available.
function Context:embedding(seq_id, dim)
    local ptr = self:embedding_ptr(seq_id)
    if ptr == nil then return nil end
    dim       = dim or 4096
    local vec = {}
    for i = 0, dim - 1 do
        vec[i + 1] = ptr[i]   -- LuaJIT: float* index auto-converts to Lua number
    end
    return vec
end

-- ── KV cache / Memory ─────────────────────────────────────────────────────────
-- All ops use the cached _mem pointer - no llama_get_memory() call per operation.

--- Check whether the KV cache supports position shifting.
--- Returns false for recurrent models (Mamba, RWKV).
--- @return bool
function Context:kv_can_shift()
    local mem = self._mem
    return mem ~= nil and self._lib.llama_memory_can_shift(mem)
end

--- Return the smallest KV position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return number
function Context:kv_seq_pos_min(seq_id)
    local mem = self._mem
    if not mem then return -1 end
    return tonumber(self._lib.llama_memory_seq_pos_min(mem, seq_id or 0))
end

--- Return the largest KV position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return number
function Context:kv_seq_pos_max(seq_id)
    local mem = self._mem
    if not mem then return -1 end
    return tonumber(self._lib.llama_memory_seq_pos_max(mem, seq_id or 0))
end

--- Divide all positions in [p0, p1) for seq_id by d (context compression).
--- @param  seq_id  number
--- @param  d       number  Divisor > 1.
--- @param  p0      number? Start (default 0).
--- @param  p1      number? End exclusive (default -1 = all).
function Context:kv_seq_div(seq_id, d, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_div(mem, seq_id, p0 or 0, p1 or -1, d) end
end

--- Clear the entire KV cache and reset n_past to 0.
function Context:kv_clear()
    local mem = self._mem
    if mem then self._lib.llama_memory_clear(mem, true) end
    self._n_past = 0
end

--- Remove KV entries for seq_id in [p0, p1). p1 = -1 means to end.
--- @param  seq_id  number
--- @param  p0      number
--- @param  p1      number  -1 = end.
--- @return bool  true if the sequence was fully removed.
function Context:kv_seq_rm(seq_id, p0, p1)
    local mem = self._mem
    if not mem then return false end
    return self._lib.llama_memory_seq_rm(mem, seq_id, p0, p1 or -1)
end

--- Copy KV entries from src to dst for positions in [p0, p1).
--- @param  src_seq  number
--- @param  dst_seq  number
--- @param  p0       number
--- @param  p1       number  -1 = end.
function Context:kv_seq_cp(src_seq, dst_seq, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1 or -1) end
end

--- Keep only seq_id; discard all other sequences from the KV cache.
--- @param  seq_id  number
function Context:kv_seq_keep(seq_id)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_keep(mem, seq_id) end
end

--- Add delta to all positions in [p0, p1) for seq_id (sliding window shift).
--- Only valid when kv_can_shift() returns true.
--- @param  seq_id  number
--- @param  delta   number  Shift amount (negative moves positions backward).
--- @param  p0      number? Start (default 0).
--- @param  p1      number? End exclusive (default -1 = all).
function Context:kv_seq_shift(seq_id, delta, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_add(mem, seq_id, p0 or 0, p1 or -1, delta) end
end

--- Add delta to positions in [p0, p1) for seq_id (canonical name for kv_seq_shift).
--- @param  seq_id  number
--- @param  p0      number
--- @param  p1      number
--- @param  delta   number
function Context:memory_seq_add(seq_id, p0, p1, delta)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_add(mem, seq_id, p0 or 0, p1 or -1, delta) end
end

-- ── State persistence ─────────────────────────────────────────────────────────

--- Save the full context state to a file (KV cache + RNG state).
--- @param  path     string   Output file path.
--- @param  tokens   cdata?   Token sequence to embed (may be nil).
--- @param  n_tokens number?  Length of tokens array.
--- @return bool
function Context:save_state(path, tokens, n_tokens)
    return self._lib.llama_state_save_file(self._ptr, path, tokens or nil, n_tokens or 0)
end

--- Restore full context state from a file saved via save_state().
--- @param  path          string
--- @param  token_buf     cdata?   Buffer to receive the stored token sequence.
--- @param  token_capacity number?
--- @return bool, number  success, number of tokens restored.
function Context:load_state(path, token_buf, token_capacity)
    local n_out = _sz1(0)
    local ok    = self._lib.llama_state_load_file(
        self._ptr, path, token_buf or nil, token_capacity or 0, n_out)
    if ok and tonumber(n_out[0]) > 0 then
        self._n_past = tonumber(n_out[0])
    end
    return ok, tonumber(n_out[0])
end

--- Serialise the full state to an in-memory Lua string (no filesystem I/O).
--- @return string  Binary blob, or nil on failure.
function Context:snapshot()
    local sz = tonumber(self._lib.llama_state_get_size(self._ptr))
    if sz == 0 then return nil end
    local buf     = _u8arr(sz)
    local written = tonumber(self._lib.llama_state_get_data(self._ptr, buf, sz))
    if written == 0 then return nil end
    return ffi.string(buf, written)
end

--- Restore state from a snapshot string produced by snapshot().
--- @param  blob  string
--- @return bool
function Context:restore(blob)
    if not blob or #blob == 0 then return false end
    local buf = _u8arr(#blob)
    ffi.copy(buf, blob, #blob)
    return self._lib.llama_state_set_data(self._ptr, buf, #blob) > 0
end

-- ── Per-sequence state ────────────────────────────────────────────────────────

--- Return the byte size of the KV state for a single sequence.
--- @param  seq_id  number?  (default 0).
--- @return number
function Context:seq_state_size(seq_id)
    return tonumber(self._lib.llama_state_seq_get_size(self._ptr, seq_id or 0))
end

--- Save the KV state for a single sequence to a file.
--- @param  path    string
--- @param  seq_id  number?  (default 0).
--- @return bool
function Context:seq_save_state(path, seq_id)
    return self._lib.llama_state_seq_save_file(self._ptr, path, seq_id or 0, nil, 0) > 0
end

--- Load the KV state for a single sequence from a file.
--- @param  path        string
--- @param  dst_seq_id  number?  (default 0).
--- @return bool
function Context:seq_load_state(path, dst_seq_id)
    local n_out = _sz1(0)
    return self._lib.llama_state_seq_load_file(
        self._ptr, path, dst_seq_id or 0, nil, 0, n_out) > 0
end

--- Serialise the KV state for a single sequence to a Lua string (no I/O).
--- Mirrors snapshot() but for one sequence only.
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return string?  Binary blob, or nil on failure.
function Context:seq_snapshot(seq_id)
    local lib = self._lib
    local sz  = tonumber(lib.llama_state_seq_get_size(self._ptr, seq_id or 0))
    if sz == 0 then return nil end
    local buf     = _u8arr(sz)
    local written = tonumber(lib.llama_state_seq_get_data(self._ptr, buf, sz, seq_id or 0))
    if written == 0 then return nil end
    return ffi.string(buf, written)
end

--- Restore the KV state for a single sequence from a blob produced by seq_snapshot().
--- @param  blob       string
--- @param  dst_seq_id number?  Destination sequence ID (default 0).
--- @return number  Bytes consumed, or 0 on failure.
function Context:seq_restore(blob, dst_seq_id)
    if not blob or #blob == 0 then return 0 end
    local buf = _u8arr(#blob)
    ffi.copy(buf, blob, #blob)
    return tonumber(self._lib.llama_state_seq_set_data(
        self._ptr, buf, #blob, dst_seq_id or 0))
end

-- ── Adapters ──────────────────────────────────────────────────────────────────

--- Apply a control vector (activation steering) to this context.
--- Takes effect on the next decode.
---
--- @param  data     table|cdata  float[] of n_embd × (il_end - il_start + 1) values.
--- @param  n_embd   number  Embedding dimension.
--- @param  il_start number  First layer (inclusive).
--- @param  il_end   number  Last layer (inclusive).
--- @return bool
function Context:set_control_vector(data, n_embd, il_start, il_end)
    local n, buf
    if type(data) == "table" then
        n   = #data
        buf = _f32arr(n)
        for i = 1, n do buf[i - 1] = data[i] end
    else
        -- cdata float* passed directly - zero copy
        buf = data
        n   = n_embd * (il_end - il_start + 1)
    end
    return self._lib.llama_set_adapter_cvec(self._ptr, buf, n, n_embd, il_start, il_end) == 0
end

--- Clear any active control vector.
function Context:clear_control_vector()
    self._lib.llama_set_adapter_cvec(self._ptr, nil, 0, 0, 0, 0)
end

--- Assign a sampler to a specific sequence. [EXPERIMENTAL]
--- When set, llama_decode() samples automatically using this sampler for seq_id.
--- @param  seq_id  number
--- @param  sampler Sampler  Must remain alive for the duration.
--- @return bool
function Context:set_sampler(seq_id, sampler)
    local chain = sampler._chain or sampler._ptr
    assert(chain ~= nil, "[ion7.core.context] set_sampler: invalid sampler")
    return self._lib.llama_set_sampler(self._ptr, seq_id, chain)
end

--- Attach a custom threadpool to this context.
--- @param  tp        cdata|Threadpool
--- @param  tp_batch  cdata|Threadpool?  Defaults to tp.
function Context:attach_threadpool(tp, tp_batch)
    local p  = type(tp)       == "table" and tp._ptr       or tp
    local pb = tp_batch and (type(tp_batch) == "table" and tp_batch._ptr or tp_batch) or p
    self._bridge.ion7_threadpool_attach(self._ptr, p, pb)
end

--- Detach the current threadpool (reverts to internal auto-managed pool).
function Context:detach_threadpool()
    self._bridge.ion7_threadpool_detach(self._ptr)
end

-- ── Performance ───────────────────────────────────────────────────────────────

--- Print performance counters to stderr.
function Context:perf_print()  self._lib.llama_perf_context_print(self._ptr) end

--- Reset performance counters.
function Context:perf_reset()  self._lib.llama_perf_context_reset(self._ptr) end

--- Return performance data as a Lua table.
--- @return table  { t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval, n_reused, tokens_per_s }
function Context:perf()
    local data  = self._lib.llama_perf_context(self._ptr)
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
