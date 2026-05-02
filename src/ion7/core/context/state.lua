--- @module ion7.core.context.state
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Context` : full-context and per-sequence state
--- persistence (KV cache + RNG seed + everything llama.cpp considers
--- restorable).
---
--- Two storage backends are exposed for each granularity :
---
---   - File on disk  : `save_state` / `load_state` (whole context),
---                     `seq_save_state` / `seq_load_state` (single seq).
---   - Lua string    : `snapshot` / `restore` (whole context),
---                     `seq_snapshot` / `seq_restore` (single seq).
---
--- The Lua-string variants are convenient for in-memory workflows
--- (sticking the state in Redis, cloning a context for parallel
--- exploration, ...). The cdata buffers are sized exactly to what
--- `llama_state_get_size` reports — no over-allocation.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_state = require "ion7.core.ffi.llama.state" -- llama_state_*

local ffi_new = ffi.new
local ffi_string = ffi.string
local ffi_copy = ffi.copy
local tonumber = tonumber

-- Reusable cdata typeof handles to dodge per-call type lookups.
local U8ARR = ffi.typeof("uint8_t[?]")
local SZ1 = ffi.typeof("size_t[1]")

local M = {}

-- ── Whole-context state (file-backed) ─────────────────────────────────────

--- Save the full context state to a file. The optional `tokens` cdata
--- is embedded so a downstream `load_state` can recover the original
--- prompt alongside the KV cache.
---
--- @param  path     string  Output file path.
--- @param  tokens   cdata?  Optional `llama_token[]` to embed.
--- @param  n_tokens integer? Length of `tokens`.
--- @return boolean
function M.save_state(self, path, tokens, n_tokens)
    return llama_state.llama_state_save_file(self._ptr, path, tokens or nil, n_tokens or 0) == true
end

--- Restore the full context state from a file written by `save_state`.
--- When `token_buf` is provided we also recover the embedded prompt
--- token sequence ; the Lua-side `n_past` mirror is updated to match
--- the restored fill position.
---
--- @param  path           string
--- @param  token_buf      cdata?   Buffer to receive the embedded tokens.
--- @param  token_capacity integer? Capacity of `token_buf`.
--- @return boolean ok
--- @return integer n_tokens_restored
function M.load_state(self, path, token_buf, token_capacity)
    local n_out = SZ1(0)
    local ok = llama_state.llama_state_load_file(self._ptr, path, token_buf or nil, token_capacity or 0, n_out)
    local n = tonumber(n_out[0])
    if ok and n > 0 then
        self._n_past = n
    end
    return ok == true, n
end

-- ── Whole-context state (Lua-string backed) ───────────────────────────────

--- Serialise the full state into a Lua string (zero file I/O).
--- Returns nil if `llama_state_get_size` reports 0 bytes.
--- @return string|nil
function M.snapshot(self)
    local sz = tonumber(llama_state.llama_state_get_size(self._ptr))
    if sz == 0 then
        return nil
    end
    local buf = U8ARR(sz)
    local written = tonumber(llama_state.llama_state_get_data(self._ptr, buf, sz))
    if written == 0 then
        return nil
    end
    return ffi_string(buf, written)
end

--- Restore state from a `snapshot` blob. Does NOT update the Lua-side
--- `n_past` mirror (the snapshot does not carry it) — call
--- `Context:set_n_past(n)` after if you need to resume position-aware
--- decoding.
---
--- @param  blob string
--- @return boolean
function M.restore(self, blob)
    if not blob or #blob == 0 then
        return false
    end
    local buf = U8ARR(#blob)
    ffi_copy(buf, blob, #blob)
    return tonumber(llama_state.llama_state_set_data(self._ptr, buf, #blob)) > 0
end

-- ── Per-sequence state (file-backed) ──────────────────────────────────────

--- Byte size of the state for a single sequence — useful to pre-size
--- caller-side buffers before `seq_snapshot`.
--- @param  seq_id integer? Default 0.
--- @return integer
function M.seq_state_size(self, seq_id)
    return tonumber(llama_state.llama_state_seq_get_size(self._ptr, seq_id or 0))
end

--- Save a single sequence's KV state to a file.
--- @param  path   string
--- @param  seq_id integer? Default 0.
--- @return boolean
function M.seq_save_state(self, path, seq_id)
    return tonumber(llama_state.llama_state_seq_save_file(self._ptr, path, seq_id or 0, nil, 0)) > 0
end

--- Load a sequence's KV state from a file into `dst_seq_id`.
--- @param  path       string
--- @param  dst_seq_id integer? Default 0.
--- @return boolean
function M.seq_load_state(self, path, dst_seq_id)
    local n_out = SZ1(0)
    return tonumber(llama_state.llama_state_seq_load_file(self._ptr, path, dst_seq_id or 0, nil, 0, n_out)) > 0
end

-- ── Per-sequence state (Lua-string backed) ────────────────────────────────

--- Serialise a single sequence's KV state into a Lua string.
--- @param  seq_id integer? Default 0.
--- @return string|nil
function M.seq_snapshot(self, seq_id)
    seq_id = seq_id or 0
    local sz = tonumber(llama_state.llama_state_seq_get_size(self._ptr, seq_id))
    if sz == 0 then
        return nil
    end
    local buf = U8ARR(sz)
    local written = tonumber(llama_state.llama_state_seq_get_data(self._ptr, buf, sz, seq_id))
    if written == 0 then
        return nil
    end
    return ffi_string(buf, written)
end

--- Restore a sequence's KV state from a `seq_snapshot` blob.
--- @param  blob       string
--- @param  dst_seq_id integer? Default 0.
--- @return integer Bytes consumed (0 on failure).
function M.seq_restore(self, blob, dst_seq_id)
    if not blob or #blob == 0 then
        return 0
    end
    local buf = U8ARR(#blob)
    ffi_copy(buf, blob, #blob)
    return tonumber(llama_state.llama_state_seq_set_data(self._ptr, buf, #blob, dst_seq_id or 0))
end

return M
