--- @module ion7.core.speculative
--- @author  ion7 / Ion7 Project Contributors
---
--- Speculative-decoding engine : either an n-gram cache built from
--- recent context (no extra model required) or a separate small "draft"
--- model that produces guesses the target model verifies in batch.
---
---   - n-gram variants (`NGRAM_CACHE`, `NGRAM_SIMPLE`, `NGRAM_MAP_K`,
---     `NGRAM_MAP_K4V`, `NGRAM_MOD`) : `ctx_dft` MUST be `nil`.
---   - `DRAFT` : `ctx_dft` is the second `Context` to query for draft
---     tokens.
---   - `EAGLE3` : the target model itself ships EAGLE3 draft heads in
---     its GGUF metadata ; `ctx_dft` is `nil`.
---
--- Drives the libcommon `common_speculative` API through the bridge,
--- which absorbs its `std::vector<llama_token>` parameters into flat C
--- arrays.
---
---   local Speculative = require "ion7.core.speculative"
---   local spec = Speculative.new(ctx_tgt, nil, {
---     type = "ngram_cache", n_draft = 5
---   })
---   spec:begin(prompt_tokens)
---   -- inside the generation loop, after sampling `last_tok` :
---   local drafts = spec:draft(all_generated, last_tok)
---   -- decode drafts in target, count how many matched, then :
---   spec:accept(n_accepted)
---
--- Resource management :
---   The native handle is `ffi.gc`-attached to the bridge's
---   `ion7_speculative_free`. The token-buffer cdata is reused across
---   calls and grown on demand to avoid per-step allocations in the
---   tight verification loop.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local bridge = require "ion7.core.ffi.bridge"

local ffi_gc = ffi.gc
local tonumber = tonumber

-- Pre-resolved typeof for the int32 token arrays we pass to the bridge.
local I32ARR = ffi.typeof("int32_t[?]")

-- Speculative-type integer codes — must stay 1:1 with the
-- `common_speculative_type` enum in libcommon. Re-exported as both
-- module constants AND a string alias map so callers can write either
-- `Speculative.NGRAM_CACHE` or `{ type = "ngram_cache" }`.
local TYPE_BY_NUMBER = {
    NONE = 0, -- speculative disabled (sentinel)
    DRAFT = 1, -- separate draft model (ctx_dft required)
    EAGLE3 = 2, -- EAGLE-3 heads on the target model
    NGRAM_SIMPLE = 3, -- recent-context n-gram lookup
    NGRAM_MAP_K = 4, -- n-gram map with k counters
    NGRAM_MAP_K4V = 5, -- n-gram map with k + 4 m-gram values
    NGRAM_MOD = 6, -- n-gram with modular prediction
    NGRAM_CACHE = 7 -- LRU n-gram cache (Cacheback paper)
}

local TYPE_BY_NAME = {
    none = 0,
    draft = 1,
    eagle3 = 2,
    ngram_simple = 3,
    ngram_map_k = 4,
    ngram_map_k4v = 5,
    ngram_mod = 6,
    ngram_cache = 7
}

--- @class ion7.core.Speculative
--- @field _ptr        cdata  `ion7_speculative_t*` (auto-freed via ffi.gc).
--- @field _n_draft    integer Per-step max draft tokens.
--- @field _draft_buf  cdata  Reusable buffer for `:draft` output.
--- @field _ctx_buf    cdata  Reusable token-history buffer (grows on demand).
--- @field _ctx_buf_sz integer Current capacity of `_ctx_buf` in tokens.
local Speculative = {}
Speculative.__index = Speculative

-- Re-export the type constants on the module table for callers that
-- prefer numeric literals over the string-based config form.
for k, v in pairs(TYPE_BY_NUMBER) do
    Speculative[k] = v
end

-- ── Constructor ──────────────────────────────────────────────────────────

--- Create a speculative-decoding engine.
---
--- @param  ctx_tgt cdata|ion7.core.Context Target context (mandatory).
--- @param  ctx_dft cdata|ion7.core.Context|nil Draft context, only for `DRAFT`.
--- @param  opts    table?
---   `type`       (string|integer, default `"ngram_cache"`) — either a
---                `Speculative.*` numeric constant or one of the lower-
---                case string aliases (`"ngram_cache"`, `"draft"`, ...).
---   `n_draft`    (integer, default 5)  — max draft tokens per step.
---   `ngram_min`  (integer, default 3)  — min n-gram order (n-gram types).
---   `ngram_max`  (integer, default 5)  — max n-gram order (n-gram types).
--- @return ion7.core.Speculative
--- @raise   When `ion7_speculative_init` returns NULL.
function Speculative.new(ctx_tgt, ctx_dft, opts)
    opts = opts or {}

    -- Resolve the type input : numeric stays as-is, strings go through
    -- the alias map, nil defaults to NGRAM_CACHE (best zero-config).
    local t = opts.type
    if type(t) == "string" then
        t = TYPE_BY_NAME[t]
        assert(t, "[ion7.core.speculative] unknown type : " .. tostring(opts.type))
    end
    t = t or Speculative.NGRAM_CACHE

    local n_draft = opts.n_draft or 5
    local ngram_min = opts.ngram_min or 3
    local ngram_max = opts.ngram_max or 5

    local tgt_ptr = type(ctx_tgt) == "table" and ctx_tgt:ptr() or ctx_tgt
    local dft_ptr = ctx_dft and (type(ctx_dft) == "table" and ctx_dft:ptr() or ctx_dft) or nil

    local ptr = bridge.ion7_speculative_init(tgt_ptr, dft_ptr, t, n_draft, ngram_min, ngram_max)
    assert(ptr ~= nil, "[ion7.core.speculative] init failed")

    -- Pre-allocate the per-call buffers ; n_draft + 1 covers the upper
    -- bound for `:draft` output, and 256 tokens is a safe starting size
    -- for the context buffer (grows on demand below).
    return setmetatable(
        {
            _ptr = ffi_gc(ptr, bridge.ion7_speculative_free),
            _n_draft = n_draft,
            _draft_buf = I32ARR(n_draft + 1),
            _ctx_buf = I32ARR(256),
            _ctx_buf_sz = 256
        },
        Speculative
    )
end

-- ── Internal : grow `_ctx_buf` to fit at least `need` tokens ─────────────

local function ensure_ctx_buf(self, need)
    if need > self._ctx_buf_sz then
        -- Grow geometrically to avoid death-by-a-thousand-reallocs when
        -- the prompt grows incrementally during streaming generation.
        local new_sz = math.max(need, self._ctx_buf_sz * 2)
        self._ctx_buf = I32ARR(new_sz)
        self._ctx_buf_sz = new_sz
    end
end

-- Internal : copy a 1-based Lua token table into `_ctx_buf` (0-based
-- cdata) and return the cdata pointer + length.
local function copy_tokens(self, tokens)
    local n = #tokens
    if n == 0 then
        return nil, 0
    end
    ensure_ctx_buf(self, n)
    local buf = self._ctx_buf
    for i = 1, n do
        buf[i - 1] = tokens[i]
    end
    return buf, n
end

-- ── Public API ───────────────────────────────────────────────────────────

--- (Re)initialise the engine with the current prompt history. Pass an
--- empty table on a fresh conversation — the cache will warm up on
--- subsequent `draft()` calls.
--- @param  tokens integer[] 1-based Lua array of token ids.
function Speculative:begin(tokens)
    local buf, n = copy_tokens(self, tokens)
    bridge.ion7_speculative_begin(self._ptr, buf, n)
end

--- Generate up to `n_draft` predicted tokens for the next step.
---
--- @param  tokens   integer[] All generated tokens so far (1-based).
---                            Only the trailing `ngram_max` tokens
---                            actually matter for n-gram lookup.
--- @param  last_tok integer   The most recently committed token id.
--- @return integer[]          1-based array of draft tokens (possibly empty).
function Speculative:draft(tokens, last_tok)
    local buf, n = copy_tokens(self, tokens)
    local k = tonumber(bridge.ion7_speculative_draft(self._ptr, buf, n, last_tok, self._draft_buf, self._n_draft))

    if k <= 0 then
        return {}
    end

    local out = {}
    local d = self._draft_buf
    for i = 0, k - 1 do
        out[i + 1] = d[i]
    end
    return out
end

--- Tell the engine how many consecutive draft tokens the target model
--- accepted. Call after every speculative step, even when `0`.
--- @param  n_accepted integer
function Speculative:accept(n_accepted)
    bridge.ion7_speculative_accept(self._ptr, n_accepted or 0)
end

--- Print acceptance-rate / effective-speedup stats to stderr.
function Speculative:stats()
    bridge.ion7_speculative_stats(self._ptr)
end

--- Explicit release. Idempotent. Disarms the GC finalizer first to
--- avoid a double-free if the GC runs later.
function Speculative:free()
    if self._ptr then
        ffi_gc(self._ptr, nil)
        bridge.ion7_speculative_free(self._ptr)
        self._ptr = nil
    end
end

return Speculative
