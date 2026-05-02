--- @module ion7.core.sampler.common
--- @author  ion7 / Ion7 Project Contributors
---
--- `CSampler` : a single-object sampler backed by libcommon's
--- `common_sampler`. Bundles the advanced features that would otherwise
--- need to be assembled step-by-step in a `Sampler` chain :
---
---   - DRY                 (don't-repeat-yourself penalty)
---   - XTC                 (exclude-top-choices)
---   - Mirostat v1 / v2    (entropy-targeting sampling)
---   - top-N-sigma         (logit standard-deviation cutoff)
---   - adaptive-p          (decayed probability threshold)
---   - grammar + grammar_lazy + trigger words
---   - logit_bias          (`{ token_id = delta_logit, ... }`)
---
--- Routed through the libcommon bridge because the `common_sampler` C++
--- type uses `std::vector` and `std::string` parameters — the bridge
--- adapts those to a flat C ABI.
---
--- Drop-in compatible with `Sampler` : `:sample(ctx, idx)` auto-accepts
--- (it calls the bridge's `ion7_csampler_sample_accept` which crosses
--- the FFI boundary ONCE per token).
---
---   local CSampler = require "ion7.core.sampler.common"
---   local s = CSampler.new(model, {
---     temp = 0.8, top_k = 40, top_p = 0.95,
---     dry_mult = 0.8, dry_base = 1.75,
---     xtc_probability = 0.1, xtc_threshold = 0.1,
---     logit_bias = { [32007] = -10.0 },     -- soft-ban a token
---   })
---   local tok = s:sample(ctx:ptr(), -1)

local ffi = require "ffi"
require "ion7.core.ffi.types"

local bridge = require "ion7.core.ffi.bridge"

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local tonumber = tonumber

--- @class ion7.core.CSampler
--- @field _ptr      cdata `ion7_csampler_t*` (auto-freed via ffi.gc).
--- @field _tw_anchor table String anchors for `trigger_words` (GC-pinning).
local CSampler = {}
CSampler.__index = CSampler

--- Create a CSampler.
---
--- @param  model cdata|table `llama_model*` or a `Model` instance.
--- @param  opts  table?      Sampling parameters (see module header for
---                           the full list and defaults).
--- @return ion7.core.CSampler
--- @raise   When the bridge call returns NULL (typically a malformed
---          grammar or a model pointer mismatch).
function CSampler.new(model, opts)
    opts = opts or {}

    local model_ptr = type(model) == "table" and model:ptr() or model
    assert(model_ptr ~= nil, "[ion7.core.sampler.common] model pointer is NULL")

    -- Initialise the params struct field-by-field rather than via a
    -- table constructor so default values match the bridge contract
    -- exactly (e.g. `dry_last_n = -1` means "full ctx", not "ignore").
    local params = ffi_new("ion7_csampler_params_t")
    params.seed = opts.seed or 0xFFFFFFFF
    params.top_k = opts.top_k or 40
    params.top_p = opts.top_p or 0.95
    params.min_p = opts.min_p or 0.05
    params.xtc_probability = opts.xtc_probability or 0.0
    params.xtc_threshold = opts.xtc_threshold or 0.1
    params.temp = opts.temp or opts.temperature or 0.8
    params.repeat_penalty = opts.repeat_penalty or 1.0
    params.freq_penalty = opts.freq_penalty or 0.0
    params.pres_penalty = opts.pres_penalty or 0.0
    params.repeat_last_n = opts.repeat_last_n or 64
    params.dry_mult = opts.dry_mult or 0.0
    params.dry_base = opts.dry_base or 1.75
    params.dry_allowed_len = opts.dry_allowed_len or 2
    params.dry_last_n = opts.dry_last_n or -1
    params.mirostat = opts.mirostat or 0
    params.mirostat_tau = opts.mirostat_tau or 5.0
    params.mirostat_eta = opts.mirostat_eta or 0.1
    params.top_n_sigma = opts.top_n_sigma or -1.0
    params.adaptive_target = opts.adaptive_target or -1.0
    params.adaptive_decay = opts.adaptive_decay or 0.9
    params.grammar_lazy = (opts.grammar_lazy and 1) or 0

    -- Trigger words for `grammar_lazy`. We pin the Lua strings on
    -- `tw_anchor` (a Lua table) so the GC cannot collect them between
    -- the `ion7_csampler_init` call and now.
    local tw_anchor = {}
    local tw_arr, n_tw = nil, 0
    if opts.trigger_words and #opts.trigger_words > 0 then
        n_tw = #opts.trigger_words
        tw_arr = ffi_new("const char*[?]", n_tw)
        for i, w in ipairs(opts.trigger_words) do
            tw_anchor[i] = w
            tw_arr[i - 1] = w
        end
    end

    -- Logit bias is exposed to the user as a sparse map ; the bridge
    -- expects two parallel arrays.
    local lb_ids, lb_vals, n_lb = nil, nil, 0
    if opts.logit_bias then
        local ids, vals = {}, {}
        for tok_id, delta in pairs(opts.logit_bias) do
            ids[#ids + 1] = tok_id
            vals[#vals + 1] = delta
        end
        n_lb = #ids
        lb_ids = ffi_new("int32_t[?]", n_lb)
        lb_vals = ffi_new("float[?]", n_lb)
        for i = 1, n_lb do
            lb_ids[i - 1] = ids[i]
            lb_vals[i - 1] = vals[i]
        end
    end

    local ptr = bridge.ion7_csampler_init(model_ptr, params, opts.grammar or nil, tw_arr, n_tw, lb_ids, lb_vals, n_lb)

    assert(ptr ~= nil, "[ion7.core.sampler.common] ion7_csampler_init returned NULL")

    return setmetatable(
        {
            _ptr = ffi_gc(ptr, bridge.ion7_csampler_free),
            _tw_anchor = tw_anchor
        },
        CSampler
    )
end

--- Sample the next token AND auto-accept it. Drop-in compatible with
--- `Sampler:sample` for the same chain semantics. Single FFI crossing
--- per generated token (the bridge wraps sample + accept in one call).
---
--- @param  ctx cdata    `llama_context*`.
--- @param  idx integer? Logit position (default `-1` = last).
--- @return integer Sampled token id.
function CSampler:sample(ctx, idx)
    return tonumber(bridge.ion7_csampler_sample_accept(self._ptr, ctx, idx or -1, 0)) or -1
end

--- Manually notify the sampler that `token` was accepted. Rarely
--- needed (`sample` already auto-accepts) ; useful when you sample via
--- another path and still want the DRY / Mirostat state updated.
function CSampler:accept(token)
    bridge.ion7_csampler_accept(self._ptr, token)
end

--- Reset every internal counter (DRY history, Mirostat state, grammar
--- automaton position).
function CSampler:reset()
    bridge.ion7_csampler_reset(self._ptr)
end

--- Last accepted token id, or `-1` if no token has been accepted yet
--- on this sampler.
--- @return integer
function CSampler:last()
    return tonumber(bridge.ion7_csampler_last(self._ptr))
end

--- Effective RNG seed used by this instance.
--- @return integer
function CSampler:seed()
    return tonumber(bridge.ion7_csampler_get_seed(self._ptr))
end

--- Free the sampler before GC. Idempotent. Disarms the `ffi.gc`
--- finalizer so the GC does not double-free later.
function CSampler:free()
    if self._ptr then
        ffi_gc(self._ptr, nil)
        bridge.ion7_csampler_free(self._ptr)
        self._ptr = nil
    end
end

return CSampler
