--- @module ion7.core.utf8
--- @author  ion7 / Ion7 Project Contributors
---
--- UTF-8 helpers for streaming token emission.
---
--- The point of this module is to let the inference layer decide whether a
--- byte buffer ends on a complete UTF-8 character boundary BEFORE handing
--- it off to upper layers (the renderer, an SSE stream, ...). Splitting a
--- multi-byte codepoint mid-sequence makes the partial bytes display as
--- a replacement character (U+FFFD) on the way out, which is visible to
--- the user as flickering or garbled diacritics.
---
--- Pure Lua / LuaJIT, no FFI. Functions are JIT-friendly : hot paths
--- avoid string allocation and reuse local aliases for the bit and string
--- libraries.

local bit = require "bit"

local bit_band = bit.band
local string_byte = string.byte

local M = {}

-- ── UTF-8 leading-byte masks ───────────────────────────────────────────────
--
-- A leading byte's top bits encode how many trailing continuation bytes
-- the codepoint requires :
--
--   0xxx_xxxx  → 1 byte total  (ASCII)
--   110x_xxxx  → 2 bytes total (e.g. accented Latin)
--   1110_xxxx  → 3 bytes total (most BMP)
--   1111_0xxx  → 4 bytes total (supplementary planes, emoji)
--   10xx_xxxx  → continuation byte (invalid as a leader)
--
-- Each pair below is `(mask, expected)` : we apply the mask and compare
-- to the expected pattern.

local M1, T1 = 0x80, 0x00 -- 1000_0000 → 0xxx_xxxx
local M2, T2 = 0xE0, 0xC0 -- 1110_0000 → 110x_xxxx
local M3, T3 = 0xF0, 0xE0 -- 1111_0000 → 1110_xxxx
local M4, T4 = 0xF8, 0xF0 -- 1111_1000 → 1111_0xxx

--- Return the expected total byte length of the UTF-8 sequence starting
--- with `b`.
---
--- @param  b integer Byte value (0-255) of the leading byte.
--- @return integer   1, 2, 3 or 4 for valid leading bytes; 0 for
---                   continuation bytes or invalid input.
function M.seq_len(b)
    if bit_band(b, M1) == T1 then
        return 1
    end
    if bit_band(b, M2) == T2 then
        return 2
    end
    if bit_band(b, M3) == T3 then
        return 3
    end
    if bit_band(b, M4) == T4 then
        return 4
    end
    return 0
end

--- Return `true` if `buf` ends on a complete UTF-8 character boundary,
--- `false` otherwise. Empty buffers count as complete.
---
--- This is the streaming-friendly check : it walks the buffer leading
--- byte by leading byte, advancing by the expected sequence length each
--- step. As soon as a sequence is malformed (zero-length leader) or
--- truncated (would advance past `#buf`), we report incomplete and stop.
---
--- The implementation does NOT validate continuation byte structure
--- (i.e. it trusts that a valid leader is followed by `seq-1` bytes
--- with `10xx_xxxx`); this matches the historical bridge behaviour and
--- is sufficient for splitting llama.cpp's tokenised output.
---
--- @param  buf string|nil Byte buffer (typically a partial token stream).
--- @return boolean
function M.is_complete(buf)
    if not buf or buf == "" then
        return true
    end
    local seq_len = M.seq_len
    local n = #buf
    local i = 1
    while i <= n do
        local b = string_byte(buf, i)
        local seq = seq_len(b)
        if seq == 0 then
            return false
        end
        if i + seq - 1 > n then
            return false
        end
        i = i + seq
    end
    return true
end

return M
