--- @module ion7.core.model.lora
--- SPDX-License-Identifier: MIT
--- LoRA adapter loading and application.
--- All functions receive the Model instance as first argument.

local M = {}

--- Load a LoRA adapter from a GGUF file.
---
--- @param  path   string  Path to the adapter .gguf file.
--- @param  scale  number?  Initial scale (default: 1.0).
--- @return table  { _ptr, path, scale, _bridge, meta }
--- @error  If loading fails.
function M.load_lora(self, path, scale)
    local ptr = self._bridge.ion7_lora_load(self._ptr, path)
    if ptr == nil then
        error(string.format(
            "[ion7.core.model] failed to load LoRA from '%s'", path), 2)
    end
    local bridge = self._bridge
    local ffi    = self._ffi
    return {
        _ptr    = ffi.gc(ptr, bridge.ion7_lora_free),
        _bridge = bridge,
        path    = path,
        scale   = scale or 1.0,
        meta    = function(adapter, key)
            local buf = ffi.new("char[512]")
            local n   = bridge.ion7_lora_meta_val(adapter._ptr, key, buf, 512)
            return n >= 0 and ffi.string(buf, n) or nil
        end,
    }
end

--- Alias for load_lora() (backwards compatibility).
function M.lora_load(self, path)
    return M.load_lora(self, path)
end

--- Apply a LoRA adapter to a context.
---
--- @param  ctx      Context  Target context.
--- @param  adapter  table    LoRA handle from load_lora().
--- @param  scale    number?  Scale override (default: adapter.scale).
--- @return bool  true on success.
function M.apply_lora(self, ctx, adapter, scale)
    return self._bridge.ion7_lora_apply(
        ctx:ptr(), adapter._ptr, scale or adapter.scale) == 0
end

--- Remove all LoRA adapters from a context.
---
--- @param  ctx      Context  Target context.
--- @return bool  true on success.
function M.remove_lora(self, ctx)
    return self._bridge.ion7_lora_remove(ctx:ptr()) == 0
end

return M
