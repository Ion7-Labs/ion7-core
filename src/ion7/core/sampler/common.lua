--- @module ion7.core.sampler.common
--- SPDX-License-Identifier: MIT
--- CSampler: wrapper around ion7_csampler_t (common_sampler from libcommon).
---
--- Supports DRY, XTC, mirostat, grammar_lazy (CRANE-style), logit_bias,
--- adaptive sampling, and top-N-sigma in a single C object.
---
--- Drop-in compatible with Sampler in generator.lua: sample() auto-accepts.
---
--- @usage
---   local CSampler = require "ion7.core.sampler.common"
---   local s = CSampler.new(model, {
---       temp = 0.8, top_k = 40, top_p = 0.95,
---       dry_mult = 0.8, dry_base = 1.75,
---       xtc_probability = 0.1, xtc_threshold = 0.1,
---   })
---   local tok = s:sample(ctx:ptr(), -1)

local Loader = require "ion7.core.ffi.loader"

--- @class CSampler
local CSampler = {}
CSampler.__index = CSampler

--- Create a CSampler from a model + options table.
---
--- @param  model  cdata|table  llama_model* or ion7-core Model object.
--- @param  opts   table?
---   opts.seed              number?   RNG seed (default: random)
---   opts.top_k             number?   Top-K (default: 40)
---   opts.top_p             number?   Top-P (default: 0.95)
---   opts.min_p             number?   Min-P (default: 0.05)
---   opts.temp              number?   Temperature (default: 0.8)
---   opts.repeat_penalty    number?   Repetition penalty (default: 1.0)
---   opts.freq_penalty      number?   Frequency penalty (default: 0.0)
---   opts.pres_penalty      number?   Presence penalty (default: 0.0)
---   opts.repeat_last_n     number?   Window for penalties (default: 64)
---   opts.dry_mult          number?   DRY multiplier (default: 0 = off)
---   opts.dry_base          number?   DRY base (default: 1.75)
---   opts.dry_allowed_len   number?   DRY min seq len (default: 2)
---   opts.dry_last_n        number?   DRY context window (default: -1 = all)
---   opts.xtc_probability   number?   XTC firing probability (default: 0 = off)
---   opts.xtc_threshold     number?   XTC logit threshold (default: 0.1)
---   opts.mirostat          number?   0 = off, 1 = v1, 2 = v2 (default: 0)
---   opts.mirostat_tau      number?   Mirostat target entropy (default: 5.0)
---   opts.mirostat_eta      number?   Mirostat learning rate (default: 0.1)
---   opts.top_n_sigma       number?   Top-N-Sigma cutoff (default: -1.0 = off)
---   opts.adaptive_target   number?   Adaptive-P target probability (default: -1.0 = off)
---   opts.adaptive_decay    number?   Adaptive-P EMA decay (default: 0.9)
---   opts.grammar           string?   GBNF grammar string (optional)
---   opts.grammar_lazy      bool?     Activate grammar only on trigger (default: false)
---   opts.trigger_words     table?    Strings that activate lazy grammar
---   opts.logit_bias        table?    { [token_id] = delta_logit, ... }
--- @return CSampler
function CSampler.new(model, opts)
    local L      = Loader.instance()
    local ffi    = L.ffi
    local bridge = L.bridge
    opts = opts or {}

    local model_ptr = type(model) == "table" and model._ptr or model
    assert(model_ptr ~= nil, "[ion7.core.sampler] CSampler: model pointer is NULL")

    local params = ffi.new("ion7_csampler_params_t", {
        seed            = opts.seed            or 0xFFFFFFFF,
        top_k           = opts.top_k           or 40,
        top_p           = opts.top_p           or 0.95,
        min_p           = opts.min_p           or 0.05,
        xtc_probability = opts.xtc_probability or 0.0,
        xtc_threshold   = opts.xtc_threshold   or 0.1,
        temp            = opts.temp or opts.temperature or 0.8,
        repeat_penalty  = opts.repeat_penalty  or 1.0,
        freq_penalty    = opts.freq_penalty    or 0.0,
        pres_penalty    = opts.pres_penalty    or 0.0,
        repeat_last_n   = opts.repeat_last_n   or 64,
        dry_mult        = opts.dry_mult        or 0.0,
        dry_base        = opts.dry_base        or 1.75,
        dry_allowed_len = opts.dry_allowed_len or 2,
        dry_last_n      = opts.dry_last_n      or -1,
        mirostat        = opts.mirostat        or 0,
        mirostat_tau    = opts.mirostat_tau    or 5.0,
        mirostat_eta    = opts.mirostat_eta    or 0.1,
        top_n_sigma     = opts.top_n_sigma     or -1.0,
        adaptive_target = opts.adaptive_target or -1.0,
        adaptive_decay  = opts.adaptive_decay  or 0.9,
        grammar_lazy    = (opts.grammar_lazy and 1 or 0),
    })

    -- Trigger words for lazy grammar activation.
    local tw_arr, tw_refs, n_tw = nil, {}, 0
    if opts.trigger_words and #opts.trigger_words > 0 then
        n_tw   = #opts.trigger_words
        tw_arr = ffi.new("const char*[?]", n_tw)
        for i, w in ipairs(opts.trigger_words) do
            tw_refs[i] = w
            tw_arr[i - 1] = w
        end
    end

    -- Logit bias: two parallel arrays { token_id } / { float delta }.
    local lb_ids, lb_vals, n_lb = nil, nil, 0
    if opts.logit_bias then
        local ids, vals = {}, {}
        for tok_id, delta in pairs(opts.logit_bias) do
            ids[#ids + 1]   = tok_id
            vals[#vals + 1] = delta
        end
        n_lb    = #ids
        lb_ids  = ffi.new("int32_t[?]", n_lb)
        lb_vals = ffi.new("float[?]",   n_lb)
        for i = 1, n_lb do lb_ids[i-1] = ids[i]; lb_vals[i-1] = vals[i] end
    end

    local ptr = bridge.ion7_csampler_init(
        model_ptr, params, opts.grammar or nil,
        tw_arr, n_tw,
        lb_ids, lb_vals, n_lb)

    assert(ptr ~= nil, "[ion7.core.sampler] ion7_csampler_init returned NULL")

    return setmetatable({
        _ptr     = ffi.gc(ptr, bridge.ion7_csampler_free),
        _bridge  = bridge,
        _ffi     = ffi,
        _tw_refs = tw_refs,   -- keep trigger word strings alive (GC anchor)
    }, CSampler)
end

--- Sample the next token. Auto-accepts so it is drop-in compatible with Sampler.
--- @param  ctx  cdata   llama_context*.
--- @param  idx  number  Logit position (-1 = last decoded position).
--- @return number  Token ID.
function CSampler:sample(ctx, idx)
    return tonumber(self._bridge.ion7_csampler_sample_accept(
        self._ptr, ctx, idx or -1, 0)) or -1
end

--- Explicitly notify the sampler that a token was accepted (rarely needed).
--- @param  token  number
function CSampler:accept(token)
    self._bridge.ion7_csampler_accept(self._ptr, token)
end

--- Reset sampler state (penalty history, mirostat state, grammar position).
function CSampler:reset()
    self._bridge.ion7_csampler_reset(self._ptr)
end

--- Last accepted token ID (-1 if none yet).
--- @return number
function CSampler:last()
    return tonumber(self._bridge.ion7_csampler_last(self._ptr))
end

--- Current RNG seed.
--- @return number
function CSampler:seed()
    return tonumber(self._bridge.ion7_csampler_get_seed(self._ptr))
end

--- Explicitly free the CSampler before GC.
function CSampler:free()
    if self._ptr then
        self._ffi.gc(self._ptr, nil)
        self._bridge.ion7_csampler_free(self._ptr)
        self._ptr = nil
    end
end

return CSampler
