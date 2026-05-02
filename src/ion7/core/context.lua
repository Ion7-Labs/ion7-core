--- @module ion7.core.context
--- @author  ion7 / Ion7 Project Contributors
---
--- Inference `Context` : decoding, KV cache, state persistence, logits
--- access. Wraps a `llama_context*` and pre-allocates a reusable decode
--- batch sized to `n_batch` so the inner generation loop never mallocs.
---
--- Sub-module split :
---
---   - `context/decode.lua`  : decode / decode_single / decode_multi / encode
---   - `context/kv.lua`      : KV cache memory ops (clear, seq_rm, seq_cp, ...)
---   - `context/state.lua`   : full and per-sequence state save / load / snapshot
---   - `context/logits.lua`  : logits, embeddings, logprob/entropy, control
---                             vector, threadpool, perf, warmup
---
--- The methods exported by each sub-module take `self` (a Context
--- instance) as their first argument and are spliced into `Context`'s
--- metatable at the bottom of this file. The public require path stays
--- `ion7.core.context` regardless of which mixin a method came from.
---
--- Lifecycle :
---   The `llama_context*` is freed via `llama_free` on garbage
---   collection. The pre-allocated batch is freed via a paired GC
---   sentinel — Lua's GC does NOT visit nested cdata fields on its own,
---   so we anchor a tiny dummy cdata whose finalizer disposes of the
---   batch.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_misc = require "ion7.core.ffi.llama.misc" -- get_memory
local llama_context = require "ion7.core.ffi.llama.context" -- n_ctx, n_batch, ...
local llama_batch = require "ion7.core.ffi.llama.batch" -- batch_init, batch_free

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local tonumber = tonumber

-- Symbolic name for the `llama_pooling_type` enum value reported by
-- `llama_pooling_type(ctx)`. Used by `Context:pooling_type` so callers
-- can `if ctx:pooling_type() == "mean"` without remembering numbers.
local POOLING_NAMES = {
    [-1] = "unspecified",
    [0] = "none",
    [1] = "mean",
    [2] = "cls",
    [3] = "last",
    [4] = "rank"
}

--- @class ion7.core.Context
--- @field _ptr          cdata    `llama_context*` (auto-freed via ffi.gc).
--- @field _model_ref    table    Parent Model — keeps the model alive as long as we are.
--- @field _mem          cdata    `llama_memory_t` accessor (cached).
--- @field _decode_batch cdata    Pre-allocated `llama_batch` reused on every decode.
--- @field _batch_gc     cdata    GC sentinel that disposes of `_decode_batch`.
--- @field _n_past       integer  Current KV fill position (Lua-side mirror).
--- @field _n_batch      integer  Batch capacity (cached, immutable).
--- @field _n_ctx        integer  Context window size (cached, immutable).
--- @field _n_ubatch     integer  Micro-batch size (cached, immutable).
--- @field _n_seq_max    integer  Maximum concurrent sequences (cached).
--- @field _is_embed     boolean? Set by `Model:embedding_context`.
local Context = {}
Context.__index = Context

-- ── Constructor ───────────────────────────────────────────────────────────

--- Wrap a raw `llama_context*` returned by `llama_init_from_model`.
--- Prefer `model:context()` over calling this directly — it does the
--- params dance and the OOM-retry cascade for you.
---
--- @param  model ion7.core.Model Parent model (back-reference for GC ordering).
--- @param  ptr   cdata           `llama_context*` (will be freed on GC).
--- @return ion7.core.Context
function Context.new(model, ptr)
    assert(ptr ~= nil, "[ion7.core.context] context pointer is NULL")

    local n_batch = tonumber(llama_context.llama_n_batch(ptr))
    local n_ctx = tonumber(llama_context.llama_n_ctx(ptr))
    local n_ubatch = tonumber(llama_context.llama_n_ubatch(ptr))
    local n_seq_max = tonumber(llama_context.llama_n_seq_max(ptr))
    local mem = llama_misc.llama_get_memory(ptr)

    -- Pre-allocate a decode batch with full n_batch capacity so the inner
    -- token-by-token generation loop NEVER mallocs. We initialise every
    -- `n_seq_id[i] = 1` once here ; in the hot path each row only touches
    -- `token[i]`, `pos[i]`, `seq_id[i][0]` and `logits[i]`.
    local db = llama_batch.llama_batch_init(n_batch, 0, 1)
    for i = 0, n_batch - 1 do
        db.n_seq_id[i] = 1
    end

    -- The decode batch is a cdata struct with internal allocations. Lua's
    -- GC will not call `llama_batch_free` on its own, so we anchor a
    -- 1-byte sentinel whose ffi.gc finalizer disposes of the batch when
    -- the Context is collected.
    local batch_gc =
        ffi_gc(
        ffi_new("int8_t[1]"),
        function()
            llama_batch.llama_batch_free(db)
        end
    )

    return setmetatable(
        {
            _ptr = ffi_gc(ptr, llama_context.llama_free),
            _model_ref = model, -- keeps model:_ptr alive while we exist
            _mem = mem,
            _decode_batch = db,
            _batch_gc = batch_gc,
            _n_past = 0,
            _n_batch = n_batch,
            _n_ctx = n_ctx,
            _n_ubatch = n_ubatch,
            _n_seq_max = n_seq_max
        },
        Context
    )
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────

--- Return the raw `llama_context*` cdata pointer (used by samplers and
--- any FFI call that needs the context handle).
--- @return cdata
function Context:ptr()
    return self._ptr
end

--- Return the cached `llama_memory_t` accessor for this context.
--- Re-calling `llama_get_memory` on every KV op would be wasteful — we
--- cache it once at construction.
--- @return cdata
function Context:memory()
    return self._mem
end

--- Explicitly free the context (and its batch buffers) immediately.
--- Idempotent. Normally the GC handles this ; call it manually inside
--- tight benchmark loops to avoid accumulating dead VRAM allocations
--- between iterations.
function Context:free()
    if self._ptr then
        ffi_gc(self._ptr, nil)
        llama_context.llama_free(self._ptr)
        self._ptr = nil
    end
    if self._batch_gc then
        -- Disarm the sentinel BEFORE freeing the batch so the GC does not
        -- get a chance to call llama_batch_free on already-freed memory.
        ffi_gc(self._batch_gc, nil)
        self._batch_gc = nil
        if self._decode_batch then
            llama_batch.llama_batch_free(self._decode_batch)
            self._decode_batch = nil
        end
    end
end

-- ── Immutable properties (cached at construction, no FFI roundtrip) ───────

--- @return integer Context window size in tokens.
function Context:n_ctx()
    return self._n_ctx
end
--- @return integer Batch capacity (max tokens per `llama_decode` call).
function Context:n_batch()
    return self._n_batch
end
--- @return integer Micro-batch size (the chunk the backend processes at once).
function Context:n_ubatch()
    return self._n_ubatch
end
--- @return integer Maximum concurrent sequences.
function Context:n_seq_max()
    return self._n_seq_max
end

-- ── Mutable / live properties (FFI roundtrip per call) ────────────────────

--- @return integer Per-sequence context window (asks llama.cpp live).
function Context:n_ctx_seq()
    return tonumber(llama_context.llama_n_ctx_seq(self._ptr))
end

--- @return integer Current generation thread count.
function Context:n_threads()
    return tonumber(llama_context.llama_n_threads(self._ptr))
end

--- @return integer Current batch processing thread count.
function Context:n_threads_batch()
    return tonumber(llama_context.llama_n_threads_batch(self._ptr))
end

--- Update the thread counts on a live context — no recreate required.
--- @param  n_threads       integer
--- @param  n_threads_batch integer? Defaults to `n_threads`.
function Context:set_n_threads(n_threads, n_threads_batch)
    llama_context.llama_set_n_threads(self._ptr, n_threads, n_threads_batch or n_threads)
end

--- Symbolic pooling strategy of the context, e.g. `"mean"` for an
--- embedding context. See `POOLING_NAMES` for the full mapping.
--- @return string
function Context:pooling_type()
    local v = tonumber(llama_context.llama_pooling_type(self._ptr))
    return POOLING_NAMES[v] or "unknown"
end

--- Toggle embedding extraction mode at runtime.
--- @param  on boolean
function Context:set_embeddings(on)
    llama_context.llama_set_embeddings(self._ptr, on)
end

--- Toggle causal attention. Pass `false` to use bidirectional attention
--- (the embedding mode used by encoder-style models).
--- @param  on boolean
function Context:set_causal_attn(on)
    llama_context.llama_set_causal_attn(self._ptr, on)
end

--- Mark the context as "in warmup" so llama.cpp does not pollute its
--- perf counters with the dummy decode shaders are JIT-compiling on.
--- See `Context:warmup()` for the high-level helper.
--- @param  on boolean
function Context:set_warmup(on)
    llama_context.llama_set_warmup(self._ptr, on)
end

--- Block until every async GPU command queued so far has finished.
--- Useful before reading logits or tensors out of a backend buffer.
function Context:synchronize()
    llama_context.llama_synchronize(self._ptr)
end

--- Register an abort callback that llama.cpp will poll periodically
--- during a decode. Returning `true` from the callback aborts.
---
--- @param  cb   cdata  Function pointer of type `bool(*)(void* data)`.
--- @param  data cdata? Opaque user data forwarded to `cb`.
function Context:set_abort_callback(cb, data)
    llama_context.llama_set_abort_callback(self._ptr, cb, data or nil)
end

-- ── KV cache fill tracking (Lua-side mirror) ──────────────────────────────

--- @return integer Current Lua-tracked KV fill position. Mirrors what we
---                 wrote into the cache via `decode` / `decode_single`.
function Context:n_past()
    return self._n_past
end

--- Manually realign the Lua-side `n_past` mirror after a state restore
--- (when llama.cpp resumes from a snapshot it knows the position but
--- we don't). Most callers should NOT need this.
--- @param  n integer
function Context:set_n_past(n)
    self._n_past = n
end

-- ── Mixin sub-modules ─────────────────────────────────────────────────────

for k, v in pairs(require "ion7.core.context.decode") do
    Context[k] = v
end
for k, v in pairs(require "ion7.core.context.kv") do
    Context[k] = v
end
for k, v in pairs(require "ion7.core.context.state") do
    Context[k] = v
end
for k, v in pairs(require "ion7.core.context.logits") do
    Context[k] = v
end

return Context
