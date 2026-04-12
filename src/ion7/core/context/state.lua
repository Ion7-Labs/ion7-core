--- @module ion7.core.context.state
--- SPDX-License-Identifier: MIT
--- Context state persistence: file-based and in-memory snapshots.
--- All functions receive the Context instance as first argument.

local ffi    = require "ffi"
local _u8arr = ffi.typeof("uint8_t[?]")
local _sz1   = ffi.typeof("size_t[1]")

local M = {}

--- Save the full context state to a file (KV cache + RNG state).
--- @param  path     string   Output file path.
--- @param  tokens   cdata?   Token sequence to embed (may be nil).
--- @param  n_tokens number?  Length of tokens array.
--- @return bool
function M.save_state(self, path, tokens, n_tokens)
    return self._lib.llama_state_save_file(self._ptr, path, tokens or nil, n_tokens or 0)
end

--- Restore full context state from a file saved via save_state().
--- @param  path           string
--- @param  token_buf      cdata?   Buffer to receive the stored token sequence.
--- @param  token_capacity number?
--- @return bool, number  success, number of tokens restored.
function M.load_state(self, path, token_buf, token_capacity)
    local n_out = _sz1(0)
    local ok    = self._lib.llama_state_load_file(
        self._ptr, path, token_buf or nil, token_capacity or 0, n_out)
    if ok and tonumber(n_out[0]) > 0 then
        self._n_past = tonumber(n_out[0])
    end
    return ok, tonumber(n_out[0])
end

--- Serialise the full state to an in-memory Lua string (no filesystem I/O).
--- @return string?  Binary blob, or nil on failure.
function M.snapshot(self)
    local sz = tonumber(self._lib.llama_state_get_size(self._ptr))
    if sz == 0 then return nil end
    local buf     = _u8arr(sz)
    local written = tonumber(self._lib.llama_state_get_data(self._ptr, buf, sz))
    if written == 0 then return nil end
    return ffi.string(buf, written)
end

--- Restore state from a snapshot string produced by snapshot().
--- @param  blob  string
--- @return bool
function M.restore(self, blob)
    if not blob or #blob == 0 then return false end
    local buf = _u8arr(#blob)
    ffi.copy(buf, blob, #blob)
    return self._lib.llama_state_set_data(self._ptr, buf, #blob) > 0
end

--- Return the byte size of the KV state for a single sequence.
--- @param  seq_id  number?  (default 0).
--- @return number
function M.seq_state_size(self, seq_id)
    return tonumber(self._lib.llama_state_seq_get_size(self._ptr, seq_id or 0))
end

--- Save the KV state for a single sequence to a file.
--- @param  path    string
--- @param  seq_id  number?  (default 0).
--- @return bool
function M.seq_save_state(self, path, seq_id)
    return self._lib.llama_state_seq_save_file(self._ptr, path, seq_id or 0, nil, 0) > 0
end

--- Load the KV state for a single sequence from a file.
--- @param  path        string
--- @param  dst_seq_id  number?  (default 0).
--- @return bool
function M.seq_load_state(self, path, dst_seq_id)
    local n_out = _sz1(0)
    return self._lib.llama_state_seq_load_file(
        self._ptr, path, dst_seq_id or 0, nil, 0, n_out) > 0
end

--- Serialise the KV state for a single sequence to a Lua string (no I/O).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return string?  Binary blob, or nil on failure.
function M.seq_snapshot(self, seq_id)
    local lib = self._lib
    local sz  = tonumber(lib.llama_state_seq_get_size(self._ptr, seq_id or 0))
    if sz == 0 then return nil end
    local buf     = _u8arr(sz)
    local written = tonumber(lib.llama_state_seq_get_data(self._ptr, buf, sz, seq_id or 0))
    if written == 0 then return nil end
    return ffi.string(buf, written)
end

--- Restore the KV state for a single sequence from a blob produced by seq_snapshot().
--- @param  blob       string
--- @param  dst_seq_id number?  Destination sequence ID (default 0).
--- @return number  Bytes consumed, or 0 on failure.
function M.seq_restore(self, blob, dst_seq_id)
    if not blob or #blob == 0 then return 0 end
    local buf = _u8arr(#blob)
    ffi.copy(buf, blob, #blob)
    return tonumber(self._lib.llama_state_seq_set_data(
        self._ptr, buf, #blob, dst_seq_id or 0))
end

return M
