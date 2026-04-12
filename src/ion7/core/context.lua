--- @module ion7.core.context
--- SPDX-License-Identifier: MIT
--- Inference context: decoding, KV cache management, state persistence.
---
--- Context wraps a llama_context* and manages a pre-allocated decode batch
--- (zero malloc per token during generation). All expensive sub-operations
--- are split into focused sub-modules under context/.

local ffi = require "ffi"

-- ── Module-level constants ─────────────────────────────────────────────────────

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
--- @field _ptr        cdata   llama_context* (freed on GC via ffi.gc).
--- @field _lib        cdata   libllama.so namespace.
--- @field _bridge     cdata   ion7_bridge.so namespace.
--- @field _n_past     number  Current KV cache fill position.
--- @field _n_batch    number  Batch capacity (cached, immutable).
--- @field _n_ctx      number  Context window size (cached, immutable).
--- @field _mem        cdata   llama_memory_t (cached, immutable).
--- @field _decode_batch llama_batch  Pre-allocated batch, reused forever.
--- @field _model_ref  table?  Parent Model object (set by Model:context()).
--- @field _n_vocab    number? Vocabulary size (set by Model:context()).
--- @field _is_embed   bool?   True when created via Model:embedding_context().
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

    local n_batch   = tonumber(lib.llama_n_batch(ptr))
    local n_ctx     = tonumber(lib.llama_n_ctx(ptr))
    local n_ubatch  = tonumber(lib.llama_n_ubatch(ptr))
    local n_seq_max = tonumber(lib.llama_n_seq_max(ptr))
    local mem       = lib.llama_get_memory(ptr)

    -- Pre-allocate a decode batch with full n_batch capacity.
    -- Reused by decode(), decode_single(), decode_multi() - zero malloc at runtime.
    local db = lib.llama_batch_init(n_batch, 0, 1)
    for i = 0, n_batch - 1 do db.n_seq_id[i] = 1 end

    -- GC sentinel: frees the batch internal buffers when the Context is collected.
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
    if self._batch_gc then
        ffi.gc(self._batch_gc, nil)   -- disarm sentinel
        self._batch_gc = nil
        if self._decode_batch then
            self._lib.llama_batch_free(self._decode_batch)
            self._decode_batch = nil
        end
    end
end

-- ── Immutable properties (cached, no FFI call) ───────────────────────────────

--- @return number  Context window size.
function Context:n_ctx()
    return self._n_ctx
end

--- @return number  Batch capacity.
function Context:n_batch()
    return self._n_batch
end

--- @return number  Micro-batch size.
function Context:n_ubatch()
    return self._n_ubatch
end

--- @return number  Current KV fill position.
function Context:n_past()
    return self._n_past
end

--- @return number  Context window per sequence.
function Context:n_ctx_seq()
    return tonumber(self._lib.llama_n_ctx_seq(self._ptr))
end

--- @return number  Maximum concurrent sequences.
function Context:n_seq_max()
    return self._n_seq_max
end

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
function Context:set_embeddings(on)
    self._lib.llama_set_embeddings(self._ptr, on)
end

--- Toggle causal attention (false = bidirectional, for embedding models).
function Context:set_causal_attn(on)
    self._lib.llama_set_causal_attn(self._ptr, on)
end

--- Enable/disable warmup mode (skips perf counters during warmup pass).
function Context:set_warmup(on)
    self._lib.llama_set_warmup(self._ptr, on)
end

--- Synchronize GPU operations (wait for all async GPU work to complete).
function Context:synchronize()
    self._lib.llama_synchronize(self._ptr)
end

--- Print a detailed memory breakdown to stderr (debugging).
function Context:memory_breakdown()
    self._lib.llama_memory_breakdown_print(self._ptr)
end

--- Register an abort callback (called periodically during decode).
--- Return true from cb to abort the current decode.
--- @param  cb    cdata  C function pointer: bool(*)(void* data).
--- @param  data  cdata? User data passed to cb.
function Context:set_abort_callback(cb, data)
    self._lib.llama_set_abort_callback(self._ptr, cb, data or nil)
end

-- ── Sub-modules ───────────────────────────────────────────────────────────────

for k, v in pairs(require "ion7.core.context.decode")  do Context[k] = v end
for k, v in pairs(require "ion7.core.context.kv")      do Context[k] = v end
for k, v in pairs(require "ion7.core.context.state")   do Context[k] = v end
for k, v in pairs(require "ion7.core.context.logits")  do Context[k] = v end

return Context
