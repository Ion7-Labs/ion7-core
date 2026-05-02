--- @module ion7.core.model.lora
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Model` : LoRA adapter loading and application.
---
--- The adapter handle returned by `Model:load_lora()` is a plain table
--- with a `:meta(key)` method ; we deliberately do NOT promote it to a
--- full class to keep the public surface small. Apply / remove live as
--- Model methods so callers can stick to the `model:do_thing(...)`
--- convention regardless of which lifecycle stage they touch.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_adapter = require "ion7.core.ffi.llama.adapter" -- adapter_lora_*
local llama_context = require "ion7.core.ffi.llama.context" -- set_adapters_lora

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local ffi_string = ffi.string

local META_BUF_SIZE = 512

local M = {}

--- @class ion7.core.LoraHandle
--- @field _ptr  cdata    `llama_adapter_lora*` (auto-freed via ffi.gc).
--- @field path  string   Origin path on disk.
--- @field scale number   Default blend strength used by `apply_lora`.

--- Load a LoRA adapter from disk.
---
--- @param  path  string  Path to the adapter `.gguf` file.
--- @param  scale number? Default scale used by `Model:apply_lora` (1.0).
--- @return ion7.core.LoraHandle
--- @raise   When the adapter file fails to open.
function M.load_lora(self, path, scale)
    local ptr = llama_adapter.llama_adapter_lora_init(self._ptr, path)
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load LoRA from '%s'", path), 2)
    end

    local handle = {
        _ptr = ffi_gc(ptr, llama_adapter.llama_adapter_lora_free),
        path = path,
        scale = scale or 1.0
    }

    --- Read a metadata value from this LoRA adapter's GGUF header.
    --- @param  key string
    --- @return string|nil
    function handle:meta(key)
        local buf = ffi_new("char[?]", META_BUF_SIZE)
        local n = llama_adapter.llama_adapter_meta_val_str(self._ptr, key, buf, META_BUF_SIZE)
        return n >= 0 and ffi_string(buf, n) or nil
    end

    return handle
end

--- Backwards-compatible alias for `load_lora`.
function M.lora_load(self, path)
    return M.load_lora(self, path)
end

--- Apply a LoRA adapter to a context with the given blend strength.
--- The underlying `llama_set_adapters_lora` takes ARRAYS, so we
--- materialise a 1-element array of pointers and a matching float[] for
--- the scale.
---
--- @param  ctx     ion7.core.Context  Target inference context.
--- @param  adapter ion7.core.LoraHandle Adapter from `load_lora`.
--- @param  scale   number?            Override the adapter's default scale.
--- @return boolean true on success.
function M.apply_lora(self, ctx, adapter, scale)
    local arr_ptrs = ffi_new("struct llama_adapter_lora*[1]", adapter._ptr)
    local arr_scales = ffi_new("float[1]", scale or adapter.scale)
    return llama_context.llama_set_adapters_lora(ctx:ptr(), arr_ptrs, 1, arr_scales) == 0
end

--- Remove all LoRA adapters from a context. Equivalent to passing an
--- empty array to `llama_set_adapters_lora`.
---
--- @param  ctx ion7.core.Context
--- @return boolean true on success.
function M.remove_lora(self, ctx)
    return llama_context.llama_set_adapters_lora(ctx:ptr(), nil, 0, nil) == 0
end

return M
