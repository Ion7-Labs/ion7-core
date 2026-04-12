--- @module ion7.core.context.decode
--- SPDX-License-Identifier: MIT
--- Token decoding (decode, decode_single, decode_multi) and encoding.
--- All functions receive the Context instance as first argument.

local ffi = require "ffi"

local M = {}

--- Decode a sequence of tokens and update the KV cache.
---
--- Uses the pre-allocated batch (n_batch capacity) - no malloc per call.
--- Handles chunking automatically when n_tokens > n_batch.
--- Only the last token of the last chunk has logits enabled.
---
--- @param  tokens      cdata|table  int32_t array (0-based) or Lua table (1-based).
--- @param  n_tokens    number?      Token count (required for cdata; auto for table).
--- @param  seq_id      number?      Sequence ID (default 0).
--- @param  pos_offset  number?      Starting KV position (default n_past).
--- @return number  Last chunk size (used as sample index for sampler:sample()).
--- @error  On KV cache full or decode error.
function M.decode(self, tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens   = n_tokens   or (is_table and #tokens or 0)
    seq_id     = seq_id     or 0
    pos_offset = pos_offset or self._n_past

    local lib     = self._lib
    local n_batch = self._n_batch
    local b       = self._decode_batch
    local tok_ptr = b.token
    local pos_ptr = b.pos
    local seq_ptr = b.seq_id
    local log_ptr = b.logits
    local done    = 0
    local last_n  = 0

    if is_table then
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local end_i  = n - 1
            local base_p = pos_offset + done
            ffi.fill(log_ptr, n)
            for i = 0, end_i do
                tok_ptr[i]    = tokens[done + i + 1]   -- Lua: 1-based
                pos_ptr[i]    = base_p + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then log_ptr[end_i] = 1 end
            b.n_tokens = n
            local ret = lib.llama_decode(self._ptr, b)
            if ret ~= 0 then
                if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
                error(string.format("[ion7.core.context] llama_decode returned %d", ret), 2)
            end
            done   = done + n
            last_n = n
        end
    else
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local end_i  = n - 1
            local base_p = pos_offset + done
            ffi.fill(log_ptr, n)
            for i = 0, end_i do
                tok_ptr[i]    = tokens[done + i]        -- cdata: 0-based
                pos_ptr[i]    = base_p + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then log_ptr[end_i] = 1 end
            b.n_tokens = n
            local ret = lib.llama_decode(self._ptr, b)
            if ret ~= 0 then
                if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
                error(string.format("[ion7.core.context] llama_decode returned %d", ret), 2)
            end
            done   = done + n
            last_n = n
        end
    end

    self._n_past = pos_offset + n_tokens
    return last_n
end

--- Decode multiple tokens with ALL logits enabled (for speculative verification).
---
--- After the call, sampler:sample(ctx:ptr(), i) is valid for i = 0 .. n-1.
---
--- @param  tokens  table   Lua 1-based array of token IDs.
--- @param  seq_id  number? Sequence ID (default 0).
--- @error  If decoding fails.
function M.decode_multi(self, tokens, seq_id)
    local n = #tokens
    assert(n > 0, "[ion7.core.context] decode_multi: empty token list")
    assert(n <= self._n_batch, string.format(
        "[ion7.core.context] decode_multi: %d tokens exceeds batch capacity %d",
        n, self._n_batch))
    seq_id     = seq_id or 0
    local lib  = self._lib
    local b    = self._decode_batch
    local base = self._n_past
    for i = 0, n - 1 do
        b.token[i]     = tokens[i + 1]
        b.pos[i]       = base + i
        b.seq_id[i][0] = seq_id
        b.logits[i]    = 1
    end
    b.n_tokens = n
    local ret = lib.llama_decode(self._ptr, b)
    if ret ~= 0 then
        if ret == 1 then error("[ion7.core.context] KV cache is full", 2) end
        error(string.format("[ion7.core.context] decode_multi failed: %d", ret), 2)
    end
    self._n_past = base + n
end

--- Decode a single token (tight generation loop path).
---
--- Uses the pre-allocated batch - zero malloc per call.
---
--- @param  token   number  Token ID.
--- @param  seq_id  number? Sequence ID (default 0).
--- @error  If decoding fails.
function M.decode_single(self, token, seq_id)
    local lib = self._lib
    local b   = self._decode_batch
    b.token[0]     = token
    b.pos[0]       = self._n_past
    b.seq_id[0][0] = seq_id or 0
    b.logits[0]    = 1
    b.n_tokens     = 1
    local ret = lib.llama_decode(self._ptr, b)
    if ret ~= 0 then
        error(string.format("[ion7.core.context] decode_single failed: %d", ret), 2)
    end
    self._n_past = self._n_past + 1
end

--- Encode a sequence (for encoder-decoder models: T5, BART, etc.).
---
--- @param  tokens      cdata|table
--- @param  n_tokens    number?
--- @param  seq_id      number?
--- @param  pos_offset  number?
--- @error  If encoding fails.
function M.encode(self, tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens   = n_tokens   or (is_table and #tokens or 0)
    seq_id     = seq_id     or 0
    pos_offset = pos_offset or self._n_past

    local lib     = self._lib
    local n_batch = self._n_batch
    local b       = self._decode_batch
    local done    = 0

    if is_table then
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local base_p = pos_offset + done
            for i = 0, n - 1 do
                b.token[i]     = tokens[done + i + 1]
                b.pos[i]       = base_p + i
                b.seq_id[i][0] = seq_id
                b.logits[i]    = 1
            end
            b.n_tokens = n
            local ret = lib.llama_encode(self._ptr, b)
            if ret ~= 0 then
                error(string.format("[ion7.core.context] llama_encode returned %d", ret), 2)
            end
            done = done + n
        end
    else
        while done < n_tokens do
            local n      = n_tokens - done
            if n > n_batch then n = n_batch end
            local base_p = pos_offset + done
            for i = 0, n - 1 do
                b.token[i]     = tokens[done + i]
                b.pos[i]       = base_p + i
                b.seq_id[i][0] = seq_id
                b.logits[i]    = 1
            end
            b.n_tokens = n
            local ret = lib.llama_encode(self._ptr, b)
            if ret ~= 0 then
                error(string.format("[ion7.core.context] llama_encode returned %d", ret), 2)
            end
            done = done + n
        end
    end
    self._n_past = pos_offset + n_tokens
end

return M
