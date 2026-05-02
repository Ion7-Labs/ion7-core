--- @module ion7.core.model.meta
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Model` : GGUF metadata access. Wraps llama.cpp's
--- `llama_model_meta_*` family directly through the auto-generated FFI.
---
--- The 4 KiB temporary buffer fits every realistic GGUF metadata value
--- (the largest in current model checkpoints — a tokenizer JSON dump —
--- comes in well below 1 KiB). For values that exceed it, the underlying
--- `llama_model_meta_val_str*` returns the required size and we'd need a
--- two-pass query. We do not bother with that here because it would be
--- exotic in practice and trivial to add later.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_model = require "ion7.core.ffi.llama.model" -- llama_model_meta_*

local ffi_new = ffi.new
local ffi_string = ffi.string
local tonumber = tonumber

local META_BUF_SIZE = 4096

local M = {}

-- ── Counts and indexed access ─────────────────────────────────────────────

--- Total number of GGUF metadata entries embedded in the model.
--- @return integer
function M.meta_count(self)
    return tonumber(llama_model.llama_model_meta_count(self._ptr))
end

--- Read the metadata key at index `i` (0-based).
--- @param  i integer
--- @return string|nil
function M.meta_key_at(self, i)
    local buf = ffi_new("char[?]", META_BUF_SIZE)
    local n = llama_model.llama_model_meta_key_by_index(self._ptr, i, buf, META_BUF_SIZE)
    return n >= 0 and ffi_string(buf, n) or nil
end

--- Read the metadata value at index `i` (0-based).
--- @param  i integer
--- @return string|nil
function M.meta_val_at(self, i)
    local buf = ffi_new("char[?]", META_BUF_SIZE)
    local n = llama_model.llama_model_meta_val_str_by_index(self._ptr, i, buf, META_BUF_SIZE)
    return n >= 0 and ffi_string(buf, n) or nil
end

-- ── Lookup by key ─────────────────────────────────────────────────────────

--- Look up a metadata value by key, e.g. `"general.name"`.
--- @param  key string
--- @return string|nil
function M.meta(self, key)
    local buf = ffi_new("char[?]", META_BUF_SIZE)
    local n = llama_model.llama_model_meta_val_str(self._ptr, key, buf, META_BUF_SIZE)
    return n >= 0 and ffi_string(buf, n) or nil
end

--- Backwards-compatible alias for `meta`.
function M.meta_val(self, key)
    return M.meta(self, key)
end

-- ── Bulk export ───────────────────────────────────────────────────────────

--- Return ALL GGUF metadata as a flat `{ [key] = value }` table.
--- Useful for one-shot logging or dumping to JSON. We allocate the two
--- scratch buffers once and reuse them across the iteration.
--- @return table<string, string>
function M.meta_all(self)
    local result = {}
    local n = llama_model.llama_model_meta_count(self._ptr)
    local kbuf = ffi_new("char[?]", META_BUF_SIZE)
    local vbuf = ffi_new("char[?]", META_BUF_SIZE)
    for i = 0, n - 1 do
        local kn = llama_model.llama_model_meta_key_by_index(self._ptr, i, kbuf, META_BUF_SIZE)
        local vn = llama_model.llama_model_meta_val_str_by_index(self._ptr, i, vbuf, META_BUF_SIZE)
        if kn >= 0 and vn >= 0 then
            result[ffi_string(kbuf, kn)] = ffi_string(vbuf, vn)
        end
    end
    return result
end

-- ── Embedded chat template ────────────────────────────────────────────────

--- Read the model's built-in Jinja2 chat template.
---
--- @param  name string|nil Template variant name (e.g. `"tool_use"`),
---                         or nil for the default template.
--- @return string|nil      Template string, or nil if none is embedded.
function M.chat_template(self, name)
    local p = llama_model.llama_model_chat_template(self._ptr, name)
    return p ~= nil and ffi_string(p) or nil
end

return M
