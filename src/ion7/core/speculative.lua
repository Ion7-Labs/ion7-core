--- @module ion7.core.speculative
--- SPDX-License-Identifier: MIT
--- Speculative decoding engine: n-gram or draft-model prediction.
---
--- Wraps ion7_speculative_* bridge functions.
--- For ngram types (NGRAM_CACHE, NGRAM_SIMPLE, NGRAM_MAP_K), ctx_dft must be nil.
--- For draft-model type, ctx_dft is a second Context loaded from a smaller model.
---
--- Typical usage (zero-config ngram_cache):
---   local Speculative = require "ion7.core.speculative"
---   local spec = Speculative.new(ctx, nil, { type = "ngram_cache", n_draft = 5 })
---   spec:begin({})           -- (re)initialize cache; pass recent token history
---
---   -- inside generation loop, after sampling `last_tok`:
---   local drafts = spec:draft(all_gen_ids, last_tok)   -- table of predicted tokens
---   -- ... decode drafts in target, verify, call spec:accept(n_accepted)
---
--- @author Ion7-Labs
--- @version 0.1.0

local ffi    = require "ffi"
local _i32arr = ffi.typeof("int32_t[?]")

local Speculative = {}
Speculative.__index = Speculative

-- ── Type constants ─────────────────────────────────────────────────────────────
-- Mirror common_speculative_type in bridge_common.cpp

--- Draft model speculative decoding (requires ctx_dft).
Speculative.DRAFT        = 1
--- EAGLE3 speculative heads on the target model itself (no ctx_dft needed).
--- Requires a model that ships EAGLE3 draft heads in its GGUF metadata.
Speculative.EAGLE3       = 2
--- Simple n-gram prediction (no draft model).
Speculative.NGRAM_SIMPLE = 3
--- N-gram with prediction map (no draft model).
Speculative.NGRAM_MAP_K  = 4
--- LRU n-gram cache (≈ Cacheback). Best zero-cost option.
Speculative.NGRAM_CACHE  = 7

local _TYPE_MAP = {
    draft        = 1,
    eagle3       = 2,
    ngram_simple = 3,
    ngram_map_k  = 4,
    ngram_cache  = 7,
}

-- ── Constructor ───────────────────────────────────────────────────────────────

--- Create a new speculative decoding engine.
---
--- @param  ctx_tgt  Context         Target inference context (ion7.core.Context).
--- @param  ctx_dft  Context?        Draft context. nil for all ngram types.
--- @param  opts     table?
---   opts.type        string|number  "ngram_cache" (default) | "ngram_simple" |
---                                   "ngram_map_k" | "draft".
---   opts.n_draft     number?        Max draft tokens per step. Default: 5.
---   opts.ngram_min   number?        Min n-gram order. Default: 3.
---   opts.ngram_max   number?        Max n-gram order. Default: 5.
--- @return Speculative
function Speculative.new(ctx_tgt, ctx_dft, opts)
    opts = opts or {}

    local L      = require("ion7.core.ffi.loader").instance()
    local bridge = L.bridge

    local type_id = opts.type
    if type(type_id) == "string" then
        type_id = assert(_TYPE_MAP[type_id], "[ion7.core.speculative] unknown type: " .. opts.type)
    end
    type_id = type_id or Speculative.NGRAM_CACHE

    local n_draft   = opts.n_draft   or 5
    local ngram_min = opts.ngram_min or 3
    local ngram_max = opts.ngram_max or 5

    local tgt_ptr = type(ctx_tgt) == "table" and ctx_tgt._ptr or ctx_tgt
    local dft_ptr = ctx_dft
        and (type(ctx_dft) == "table" and ctx_dft._ptr or ctx_dft)
        or  nil

    local ptr = bridge.ion7_speculative_init(tgt_ptr, dft_ptr, type_id, n_draft, ngram_min, ngram_max)
    assert(ptr ~= nil, "[ion7.core.speculative] init failed")

    return setmetatable({
        _ptr        = ffi.gc(ptr, bridge.ion7_speculative_free),
        _bridge     = bridge,
        _n_draft    = n_draft,
        -- Reusable C buffers (avoid alloc per call)
        _buf        = _i32arr(n_draft + 1),
        _ctx_buf    = _i32arr(256),
        _ctx_buf_sz = 256,
    }, Speculative)
end

-- ── API ───────────────────────────────────────────────────────────────────────

--- (Re)initialise the n-gram cache with recent token history.
--- Call once after the prompt is encoded, before the first draft() call.
--- Passing an empty table is valid (cache warms up from generated tokens).
---
--- @param  tokens  table  Lua 1-based array of int token IDs.
function Speculative:begin(tokens)
    local n = #tokens
    if n == 0 then
        self._bridge.ion7_speculative_begin(self._ptr, nil, 0)
        return
    end
    if n > self._ctx_buf_sz then
        self._ctx_buf    = _i32arr(n)
        self._ctx_buf_sz = n
    end
    local buf = self._ctx_buf
    for i = 1, n do buf[i - 1] = tokens[i] end
    self._bridge.ion7_speculative_begin(self._ptr, buf, n)
end

--- Generate draft tokens for the next speculative step.
---
--- Uses the n-gram cache (or draft model) to predict continuations of
--- `last_tok` given the recent context in `tokens`.
---
--- @param  tokens    table   All generated token IDs so far (Lua 1-based).
---                           Only the trailing ngram_max tokens matter for lookup.
--- @param  last_tok  number  The most recently sampled / committed token (int32).
--- @return table     Lua 1-based array of up to n_draft candidate tokens.
---                   Empty when no prediction is available.
function Speculative:draft(tokens, last_tok)
    local n       = #tokens
    local ctx_buf = nil
    if n > 0 then
        if n > self._ctx_buf_sz then
            self._ctx_buf    = _i32arr(n)
            self._ctx_buf_sz = n
        end
        ctx_buf = self._ctx_buf
        for i = 1, n do ctx_buf[i - 1] = tokens[i] end
    end

    local k = tonumber(self._bridge.ion7_speculative_draft(
        self._ptr, ctx_buf, n, last_tok,
        self._buf, self._n_draft))

    if k <= 0 then return {} end

    local result = {}
    local buf    = self._buf
    for i = 0, k - 1 do
        result[i + 1] = buf[i]   -- int32_t auto-converts to Lua number
    end
    return result
end

--- Inform the engine how many consecutive draft tokens were accepted.
--- Call after every speculative step, even when n_accepted = 0.
---
--- @param  n_accepted  number  Count of accepted tokens from the last draft().
function Speculative:accept(n_accepted)
    self._bridge.ion7_speculative_accept(self._ptr, n_accepted or 0)
end

--- Print speculative decoding statistics to stderr.
--- Useful after generation to see the acceptance rate.
function Speculative:stats()
    self._bridge.ion7_speculative_stats(self._ptr)
end

--- Explicitly free the speculative engine.
--- Optional - the GC finalizer handles it automatically.
function Speculative:free()
    if self._ptr then
        self._bridge.ion7_speculative_free(self._ptr)
        self._ptr = ffi.gc(self._ptr, nil)
        self._ptr = nil
    end
end

return Speculative
