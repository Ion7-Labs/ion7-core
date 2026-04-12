--- @module ion7.core.model.meta
--- SPDX-License-Identifier: MIT
--- Model GGUF metadata access.
--- All functions receive the Model instance as first argument.

local M = {}

local META_BUF_SIZE = 4096

--- @return number  Total number of GGUF metadata entries.
function M.meta_count(self)
    return tonumber(self._bridge.ion7_model_meta_count(self._ptr))
end

--- @param  i  number  0-based index.
--- @return string?
function M.meta_key_at(self, i)
    local buf = self._ffi.new("char[512]")
    local n   = self._bridge.ion7_model_meta_key_at(self._ptr, i, buf, 512)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

--- @param  i  number  0-based index.
--- @return string?
function M.meta_val_at(self, i)
    local buf = self._ffi.new("char[4096]")
    local n   = self._bridge.ion7_model_meta_val_at(self._ptr, i, buf, 4096)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

--- Get a GGUF metadata value by key name.
--- @param  key  string  e.g. "general.name", "llama.context_length".
--- @return string?
function M.meta(self, key)
    local buf = self._ffi.new("char[?]", META_BUF_SIZE)
    local n   = self._bridge.ion7_model_meta_val(self._ptr, key, buf, META_BUF_SIZE)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

--- Alias for meta().
function M.meta_val(self, key)
    return M.meta(self, key)
end

--- Return all GGUF metadata as a Lua table { key = value, ... }.
--- @return table
function M.meta_all(self)
    local result = {}
    local n      = self._bridge.ion7_model_meta_count(self._ptr)
    local kbuf   = self._ffi.new("char[?]", META_BUF_SIZE)
    local vbuf   = self._ffi.new("char[?]", META_BUF_SIZE)
    for i = 0, n - 1 do
        local kn = self._bridge.ion7_model_meta_key_at(self._ptr, i, kbuf, META_BUF_SIZE)
        local vn = self._bridge.ion7_model_meta_val_at(self._ptr, i, vbuf, META_BUF_SIZE)
        if kn >= 0 and vn >= 0 then
            result[self._ffi.string(kbuf, kn)] = self._ffi.string(vbuf, vn)
        end
    end
    return result
end

--- Get the model's built-in Jinja2 chat template string.
--- @param  name  string?  Template variant name, or nil for the default.
--- @return string?
function M.chat_template(self, name)
    local p = self._bridge.ion7_model_chat_template(self._ptr, name)
    return (p ~= nil) and self._ffi.string(p) or nil
end

return M
