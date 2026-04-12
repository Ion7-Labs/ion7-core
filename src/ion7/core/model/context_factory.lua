--- @module ion7.core.model.context_factory
--- SPDX-License-Identifier: MIT
--- Vocabulary and context creation from a Model.
--- All functions receive the Model instance as first argument.

local Vocab   = require "ion7.core.vocab"
local Context = require "ion7.core.context"

-- KV cache type → GGML_TYPE_* value.
-- V cache (type_v) supports: f16, bf16, q8_0, q4_0 (no flash needed for these).
-- K cache (type_k) additionally supports: q4_1, q5_0, q5_1, iq4_nl.
-- Q4_K/Q5_K for V require flash attention + GGML_CUDA_FA_ALL_QUANTS.
local KV_TYPES = {
    f16    =  1,   -- default, best quality
    bf16   = 30,   -- BF16, slightly smaller than f16
    q8_0   =  8,   -- ~50% VRAM savings, minimal quality loss
    q4_0   =  2,   -- ~75% VRAM savings, some quality loss
    q4_1   =  3,   -- ~75% VRAM savings, slightly better than q4_0
    q5_0   =  6,   -- ~60% VRAM savings
    q5_1   =  7,   -- ~60% VRAM savings, slightly better
    iq4_nl = 20,   -- ~75% VRAM savings, better quality than q4_0
    q4_k   = 12,   -- K cache only (requires flash attn for V)
    q5_k   = 13,   -- K cache only (requires flash attn for V)
}

local M = {}

--- Return the Vocab handle for this model. Created once and cached.
--- @return Vocab
function M.vocab(self)
    if self._vocab then return self._vocab end
    local ptr = self._lib.llama_model_get_vocab(self._ptr)
    assert(ptr ~= nil, "[ion7.core.model] llama_model_get_vocab returned NULL")
    self._vocab = Vocab.new(self._lib, self._ptr, ptr)
    return self._vocab
end

--- Create an inference context for text generation.
---
--- @param  opts  table?
---   opts.n_ctx           number?  Context window in tokens (default: 4096).
---   opts.n_batch         number?  Logical batch size (default: 2048).
---   opts.n_ubatch        number?  Physical micro-batch size (default: auto).
---   opts.n_threads       number?  CPU threads for generation (default: 4).
---   opts.n_threads_batch number?  CPU threads for prompt eval (default: 2*n_threads).
---   opts.flash_attn      bool?    Force Flash Attention ON (default: AUTO).
---   opts.offload_kqv     bool?    Keep KV cache on GPU (default: true).
---   opts.op_offload      bool?    Offload host tensor ops to GPU (default: false).
---   opts.no_perf         bool?    Disable perf counters (default: false).
---   opts.kv_type         string?  KV cache type for both K and V (default: "f16").
---   opts.kv_type_k       string?  Override K cache type independently.
---   opts.kv_type_v       string?  Override V cache type independently.
---   opts.swa_full        bool?    Full-size Sliding Window Attention cache.
---   opts.kv_unified      bool?    Unified KV buffer across sequences.
--- @return Context
--- @error  If context creation fails.
function M.context(self, opts)
    opts = opts or {}
    local kv_type_k = opts.kv_type_k or opts.kv_type or "f16"
    local kv_type_v = opts.kv_type_v or opts.kv_type or "f16"
    local type_k    = KV_TYPES[kv_type_k]
    local type_v    = KV_TYPES[kv_type_v]
    assert(type_k, "[ion7-core] unknown kv_type_k: " .. kv_type_k)
    assert(type_v, "[ion7-core] unknown kv_type_v: " .. kv_type_v)

    local flash_attn  = (opts.flash_attn and 1 or 0)
    local offload_kqv = (opts.offload_kqv ~= false and 1 or 0)

    -- Quantized KV requires flash attention for correct results.
    if type_k ~= 1 or type_v ~= 1 then flash_attn = 1 end

    local n_batch  = opts.n_batch or 2048
    local n_ubatch
    if opts.n_ubatch then
        n_ubatch = opts.n_ubatch
    else
        local use_gpu = (opts.n_gpu_layers or 0) ~= 0 or (type_k ~= 1 or type_v ~= 1)
        n_ubatch = use_gpu and math.min(512, n_batch) or math.min(256, n_batch)
    end

    local function try_create(n_ctx_, flash_)
        return self._bridge.ion7_context_create(
            self._ptr, n_ctx_, n_batch, n_ubatch,
            opts.n_seq_max       or 1,
            opts.n_threads       or 4,
            opts.n_threads_batch or 0,
            flash_, offload_kqv,
            opts.op_offload  and 1 or 0,
            opts.no_perf     and 1 or 0,
            type_k, type_v,
            opts.swa_full    and 1 or 0,
            opts.kv_unified  and 1 or 0
        )
    end

    local n_ctx_ = opts.n_ctx or 4096
    local ptr    = try_create(n_ctx_, flash_attn)

    -- Retry cascade when VRAM is tight.
    if ptr == nil and (type_k ~= 1 or type_v ~= 1) then
        ptr = try_create(n_ctx_, 0)          -- drop quantized KV, try f16
    end
    if ptr == nil and n_ctx_ > 4096 then
        ptr = try_create(4096, 0)            -- reduce context
    end
    if ptr == nil then
        error(string.format(
            "[ion7.core.model] failed to create inference context" ..
            " (n_ctx=%d, kv=%s, flash=%d) -- out of VRAM?",
            n_ctx_, kv_type_k .. "/" .. kv_type_v, flash_attn), 2)
    end

    local ctx = Context.new(self._lib, self._bridge, ptr)
    ctx._n_vocab   = self:vocab():n_vocab()
    ctx._model_ref = self
    return ctx
end

--- Create an embedding context (CPU, pooling enabled, no logits).
---
--- @param  opts  table?
---   opts.n_ctx      number?  Context size (default: 512).
---   opts.n_threads  number?  CPU threads (default: 4).
---   opts.pooling    string?  "none"|"mean"|"cls"|"last"(default)|"rank".
--- @return Context
--- @error  If context creation fails.
function M.embedding_context(self, opts)
    opts = opts or {}
    local pooling_map = { none = -1, mean = 1, cls = 2, last = 3, rank = 4 }
    local pooling = pooling_map[opts.pooling or "last"] or 3

    local ptr = self._bridge.ion7_embedding_context_create(
        self._ptr, opts.n_ctx or 512, opts.n_seq_max or 1, opts.n_threads or 4, pooling)
    if ptr == nil then
        error("[ion7.core.model] failed to create embedding context", 2)
    end

    local ctx = Context.new(self._lib, self._bridge, ptr)
    ctx._n_vocab   = self:vocab():n_vocab()
    ctx._model_ref = self
    ctx._is_embed  = true
    return ctx
end

return M
