--- @module ion7.core.custom_sampler
--- SPDX-License-Identifier: MIT
--- Write sampling logic in pure Lua and plug it into the llama.cpp chain.
---
--- Uses the bridge's ion7_sampler_create() which constructs the llama_sampler_i
--- struct in C (safe, no FFI struct layout issues). LuaJIT callbacks are
--- created via ffi.cast() and stored to prevent GC.
---
--- Usage:
---   local cs = llama.CustomSampler.new("lua_greedy", {
---       apply = function(candidates, n)
---           -- candidates: cdata llama_token_data* array
---           -- n: number of candidates
---           -- Return the index of the selected token (0-based)
---           local best_i, best_logit = 0, -math.huge
---           for i = 0, n - 1 do
---               if candidates[i].logit > best_logit then
---                   best_logit = candidates[i].logit
---                   best_i = i
---               end
---           end
---           return best_i
---       end,
---   })
---
---   local chain = llama.Sampler.chain():custom(cs):build(vocab)
---
--- Callback signatures:
---   apply(candidates cdata, n number)  -> number  Required. Selected index (0-based).
---   accept(token_id number)                        Called after token accepted.
---   reset()                                        Called on chain reset.
---
--- Keep the CustomSampler object alive as long as the chain is in use.

local Loader = require "ion7.core.ffi.loader"

local CustomSampler = {}
CustomSampler.__index = CustomSampler

--- Create a custom sampler from Lua callbacks.
--- Uses ion7_sampler_create (bridge) to safely wrap Lua functions as C callbacks.
---
--- @param  name      string
--- @param  callbacks table
---   callbacks.apply(candidates cdata, n number) -> number  Required.
---   callbacks.accept(token number)?
---   callbacks.reset()?
--- @return CustomSampler
function CustomSampler.new(name, callbacks)
    assert(type(callbacks) == "table" and type(callbacks.apply) == "function",
        "[ion7.core.custom_sampler] callbacks.apply is required")

    local L   = Loader.instance()
    local ffi = L.ffi

    -- Build C function pointers via ffi.cast().
    -- The bridge typedefs ion7_sampler_*_fn with a void* userdata parameter.
    -- Our closures capture the Lua callbacks directly -- no userdata needed.

    -- apply: (llama_token_data_array* cur_p, void* ud) -> void
    -- We receive cur_p as a pointer; LuaJIT auto-dereferences fields.
    local apply_fn = callbacks.apply
    local apply_cb = ffi.cast("ion7_sampler_apply_fn", function(cur_p, _)
        -- LuaJIT rule: Lua errors MUST NOT cross FFI callback boundaries.
        -- Any error inside ffi.cast callbacks causes PANIC: bad callback.
        -- We wrap in pcall and fall back to token 0 on error.
        local ok, idx = pcall(apply_fn, cur_p, tonumber(cur_p.size))
        if ok and type(idx) == "number" and idx >= 0 then
            cur_p.selected = idx
        elseif not ok then
            cur_p.selected = 0  -- fallback: pick first token
        end
    end)

    -- accept: (int32_t token, void* ud) -> void
    local accept_cb = nil
    if type(callbacks.accept) == "function" then
        local accept_fn = callbacks.accept
        accept_cb = ffi.cast("ion7_sampler_accept_fn", function(tok, _)
            pcall(accept_fn, tonumber(tok))
        end)
    end

    -- reset: (void* ud) -> void
    local reset_cb = nil
    if type(callbacks.reset) == "function" then
        local reset_fn = callbacks.reset
        reset_cb = ffi.cast("ion7_sampler_reset_fn", function(_)
            pcall(reset_fn)
        end)
    end

    -- Create the sampler via the bridge (constructs llama_sampler_i in C)
    local raw = L.bridge.ion7_sampler_create(
        name or "custom",
        apply_cb,
        accept_cb,
        reset_cb,
        nil,   -- free_fn: callbacks are GC'd Lua objects, bridge doesn't own them
        nil    -- userdata: not used (closures capture state directly)
    )

    if raw == nil then
        error("[ion7.core.custom_sampler] ion7_sampler_create returned NULL", 2)
    end

    return setmetatable({
        _name      = name or "custom",
        _ptr       = raw,        -- raw llama_sampler* (NO GC finalizer -- chain owns it)
        _apply_cb  = apply_cb,   -- kept alive to prevent GC of C callback
        _accept_cb = accept_cb,
        _reset_cb  = reset_cb,
    }, CustomSampler)
end

--- @return cdata  Raw llama_sampler* for passing to the sampler chain.
function CustomSampler:ptr()
    return self._ptr
end

--- @return string
function CustomSampler:name()
    return self._name
end

return CustomSampler
