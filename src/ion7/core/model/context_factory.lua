--- @module ion7.core.model.context_factory
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Model` : Vocab and Context creation factories.
---
--- The Vocab handle is cached on the Model so repeated `Model:vocab()`
--- calls are O(1) ; the underlying `llama_vocab*` is owned by the model
--- itself, so we do not attach a finalizer to it.
---
--- `Model:context()` builds a `llama_context_params` struct from
--- ergonomic Lua options, then calls `llama_init_from_model`. When
--- creation fails with quantised KV (the most common cause of OOM on
--- bring-up) we retry once with f16 KV, and once more with a smaller
--- context size. The cascade lets the caller pass the BIGGEST setting
--- they would like and silently fall back to whatever VRAM allows.
---
--- The `Vocab` and `Context` modules are required INSIDE the methods to
--- break the otherwise-circular dependency at module-init time
--- (`Model` ←→ `Vocab` ←→ `Context`).

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_model = require "ion7.core.ffi.llama.model" -- model_get_vocab
local llama_context = require "ion7.core.ffi.llama.context" -- context_default_params
local llama_misc = require "ion7.core.ffi.llama.misc" -- llama_init_from_model

local math_min = math.min

-- KV cache element type → `enum ggml_type` numeric value. The K cache
-- accepts a few quantisations the V cache does not (because flash
-- attention is required for V-side dequant on the fly). We expose the
-- common subset and document the constraint above.
--
-- All values match the upstream `enum ggml_type` from `ggml.h` :
local KV_TYPES = {
    f16 = 1, -- default, full precision, supported everywhere
    bf16 = 30,
    q8_0 = 8,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    iq4_nl = 20,
    q4_k = 12, -- K cache only ; V needs flash attention + GGML_CUDA_FA_ALL_QUANTS
    q5_k = 13 -- K cache only
}

-- Pooling strategy strings → `enum llama_pooling_type` numeric value.
local POOLING_TYPES = {
    none = -1,
    mean = 1,
    cls = 2,
    last = 3,
    rank = 4
}

local M = {}

-- ── Vocab factory ─────────────────────────────────────────────────────────

--- Return the model's vocabulary as an ion7.core.Vocab handle. Cached
--- on the Model — repeated calls are O(1).
--- @return ion7.core.Vocab
function M.vocab(self)
    if self._vocab then
        return self._vocab
    end
    local Vocab = require "ion7.core.vocab"
    local ptr = llama_model.llama_model_get_vocab(self._ptr)
    assert(ptr ~= nil, "[ion7.core.model] llama_model_get_vocab returned NULL")
    self._vocab = Vocab.new(self, ptr)
    return self._vocab
end

-- ── Inference context factory ─────────────────────────────────────────────

--- Create an inference context from this model.
---
--- @param  opts table?
---   `n_ctx`           (integer, default 4096)
---   `n_batch`         (integer, default 2048)
---   `n_ubatch`        (integer, default `min(512, n_batch)` — matches llama.cpp)
---   `n_seq_max`       (integer, default 1)
---   `n_threads`       (integer, default 4)
---   `n_threads_batch` (integer, default `2 × n_threads` — empirically optimal
---                      for prefill on multi-core CPUs ; reduce to `n_threads`
---                      manually on tiny CPUs to avoid oversubscription)
---   `flash_attn`      (bool,    default false ; auto-forced on for quantised KV)
---   `offload_kqv`     (bool,    default true)
---   `op_offload`      (bool,    default false)
---   `no_perf`         (bool,    default false)
---   `kv_type`         (string,  default `"f16"`) — sets BOTH K and V cache type
---   `kv_type_k`       (string)  — override K independently
---   `kv_type_v`       (string)  — override V independently
---   `swa_full`        (bool,    default false)
---   `kv_unified`      (bool,    default false)
--- @return ion7.core.Context
--- @raise   When context creation fails after the retry cascade.
function M.context(self, opts)
    opts = opts or {}
    local Context = require "ion7.core.context"

    local kv_type_k_name = opts.kv_type_k or opts.kv_type or "f16"
    local kv_type_v_name = opts.kv_type_v or opts.kv_type or "f16"
    local type_k = KV_TYPES[kv_type_k_name]
    local type_v = KV_TYPES[kv_type_v_name]
    assert(type_k, "[ion7.core.model] unknown kv_type_k: " .. kv_type_k_name)
    assert(type_v, "[ion7.core.model] unknown kv_type_v: " .. kv_type_v_name)

    -- Quantised KV always requires flash attention to dequant on the fly.
    local force_flash = (type_k ~= 1) or (type_v ~= 1)

    -- Default n_ubatch matches llama.cpp upstream (`min(512, n_batch)`).
    -- An earlier version of this code halved n_ubatch on CPU under the
    -- assumption that smaller chunks would fit better in cache — it
    -- turns out that on modern x86 the per-forward-pass overhead
    -- (graph build, threadpool wake-up, sync barriers) dominates any
    -- locality win, so larger ubatches are strictly faster for prefill.
    -- Confirmed via ion7-benchmark `ttft` scenario on Ryzen 9950X :
    -- the old 256-on-CPU default cost ~30 % prefill throughput at
    -- pp ≥ 512 versus llama-cpp-python's defaults.
    local n_batch = opts.n_batch or 2048
    local n_ubatch = opts.n_ubatch or math_min(512, n_batch)

    -- Build the params struct ; mutate, then hand to llama_init_from_model.
    -- Each retry rebuilds the struct so we cleanly start from the upstream
    -- defaults again.
    local function try_create(n_ctx_value, force_flash_value, type_k_value, type_v_value)
        local p = llama_context.llama_context_default_params()
        p.n_ctx = n_ctx_value or 4096
        p.n_batch = n_batch
        p.n_ubatch = n_ubatch
        p.n_seq_max = opts.n_seq_max or 1
        p.n_threads = opts.n_threads or 4
        p.n_threads_batch = (opts.n_threads_batch and opts.n_threads_batch > 0) and opts.n_threads_batch or (p.n_threads * 2)
        -- LLAMA_FLASH_ATTN_TYPE_ENABLED = 1, AUTO = -1 (per llama.h enum).
        p.flash_attn_type = (force_flash_value or opts.flash_attn) and 1 or -1
        if opts.offload_kqv ~= false then
            p.offload_kqv = true
        end
        if opts.op_offload then
            p.op_offload = true
        end
        if opts.no_perf then
            p.no_perf = true
        end
        if opts.swa_full then
            p.swa_full = true
        end
        if opts.kv_unified then
            p.kv_unified = true
        end
        if type_k_value > 0 then
            p.type_k = type_k_value
        end
        if type_v_value > 0 then
            p.type_v = type_v_value
        end
        return llama_misc.llama_init_from_model(self._ptr, p)
    end

    local n_ctx_req = opts.n_ctx or 4096
    local ptr = try_create(n_ctx_req, force_flash, type_k, type_v)

    -- Retry cascade : VRAM-tight setups often need to drop quantised KV,
    -- then fall back to a smaller context.
    if ptr == nil and (type_k ~= 1 or type_v ~= 1) then
        ptr = try_create(n_ctx_req, false, 1, 1)
    end
    if ptr == nil and n_ctx_req > 4096 then
        ptr = try_create(4096, false, 1, 1)
    end
    if ptr == nil then
        error(
            string.format(
                "[ion7.core.model] failed to create inference context " .. "(n_ctx=%d, kv=%s/%s) — out of VRAM ?",
                n_ctx_req,
                kv_type_k_name,
                kv_type_v_name
            ),
            2
        )
    end

    return Context.new(self, ptr)
end

-- ── Embedding context factory ─────────────────────────────────────────────

--- Create an embedding context : pooling enabled, no logits, CPU-only
--- by default. Used for vector-search workloads where the model only
--- emits a single vector per input rather than token-by-token output.
---
--- @param  opts table?
---   `n_ctx`     (integer, default 512)
---   `n_seq_max` (integer, default 1)
---   `n_threads` (integer, default 4)
---   `pooling`   (string,  default `"last"`) — one of
---                `"none" | "mean" | "cls" | "last" | "rank"`.
--- @return ion7.core.Context  Marked with `_is_embed = true`.
--- @raise   When creation fails.
function M.embedding_context(self, opts)
    opts = opts or {}
    local Context = require "ion7.core.context"

    local pooling = POOLING_TYPES[opts.pooling or "last"]
    assert(pooling, "[ion7.core.model] unknown pooling type: " .. tostring(opts.pooling))

    local p = llama_context.llama_context_default_params()
    p.n_ctx = opts.n_ctx or 512
    p.n_batch = p.n_ctx
    p.n_ubatch = p.n_batch
    p.n_seq_max = opts.n_seq_max or 1
    p.n_threads = opts.n_threads or 4
    p.n_threads_batch = p.n_threads
    p.embeddings = true
    p.pooling_type = pooling
    p.offload_kqv = false
    p.op_offload = false
    p.no_perf = true
    p.flash_attn_type = -1 -- AUTO

    local ptr = llama_misc.llama_init_from_model(self._ptr, p)
    if ptr == nil then
        error("[ion7.core.model] failed to create embedding context", 2)
    end

    local ctx = Context.new(self, ptr)
    ctx._is_embed = true
    return ctx
end

return M
