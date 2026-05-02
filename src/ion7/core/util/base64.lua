--- @module ion7.core.base64
--- @author  ion7 / Ion7 Project Contributors
---
--- Standard Base64 (RFC 4648) encoder and decoder.
---
--- Originally provided by the C bridge as wrappers around libcommon's
--- `base64::encode` / `base64::decode`. The pure-Lua port keeps the same
--- semantics (NUL-byte safe, `=` padding, 64-character alphabet) and is
--- fast enough for ion7's typical workload — base64 is only invoked on
--- multimodal image attachments, never inside the per-token hot path.
---
--- API differences from the C bridge :
---   - Lua functions return STRINGS (no caller-managed `out` buffers).
---     Lua strings are length-tagged byte sequences; nothing prevents
---     embedding NUL bytes.
---   - `decode` returns `nil` on malformed input rather than `-1`.
---
--- The implementation is byte-oriented and uses LuaJIT's `bit` library
--- for the 6-bit/8-bit shuffling. No allocations in the inner loop
--- beyond the per-step indexing of the alphabet table.

local bit = require "bit"

local bit_band = bit.band
local bit_bor = bit.bor
local bit_lshift = bit.lshift
local bit_rshift = bit.rshift
local string_byte = string.byte
local string_char = string.char
local string_sub = string.sub
local string_rep = string.rep
local table_concat = table.concat

local M = {}

-- ── Alphabets ─────────────────────────────────────────────────────────────

local ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- Forward lookup `[0..63] → ascii` as a 64-entry array indexed from 1.
local ENC = {}
for i = 1, #ALPHABET do
    ENC[i - 1] = string_byte(ALPHABET, i)
end

-- Reverse lookup `[ascii] → [0..63]` (or `nil` for invalid bytes).
local DEC = {}
for i = 1, #ALPHABET do
    DEC[string_byte(ALPHABET, i)] = i - 1
end

-- ── Encode ────────────────────────────────────────────────────────────────

--- Encode binary `data` into a Base64 ASCII string.
---
--- Trailing partial groups are padded with `=` so the output length is
--- always a multiple of 4. NUL bytes inside `data` are handled correctly.
---
--- @param  data string Raw byte sequence (Lua string).
--- @return string      Base64-encoded ASCII (only [A-Za-z0-9+/=]).
function M.encode(data)
    local n = #data
    if n == 0 then
        return ""
    end

    -- Process full 3-byte groups first; output 4 chars per group.
    local out, oi = {}, 0
    local full_end = n - (n % 3)

    for i = 1, full_end, 3 do
        local b1, b2, b3 = string_byte(data, i, i + 2)
        out[oi + 1] = ENC[bit_rshift(b1, 2)]
        out[oi + 2] = ENC[bit_bor(bit_lshift(bit_band(b1, 0x03), 4), bit_rshift(b2, 4))]
        out[oi + 3] = ENC[bit_bor(bit_lshift(bit_band(b2, 0x0F), 2), bit_rshift(b3, 6))]
        out[oi + 4] = ENC[bit_band(b3, 0x3F)]
        oi = oi + 4
    end

    -- Handle the trailing 1- or 2-byte tail with explicit `=` padding.
    local rem = n - full_end
    if rem == 1 then
        local b1 = string_byte(data, full_end + 1)
        out[oi + 1] = ENC[bit_rshift(b1, 2)]
        out[oi + 2] = ENC[bit_lshift(bit_band(b1, 0x03), 4)]
        out[oi + 3] = 0x3D -- '='
        out[oi + 4] = 0x3D -- '='
        oi = oi + 4
    elseif rem == 2 then
        local b1, b2 = string_byte(data, full_end + 1, full_end + 2)
        out[oi + 1] = ENC[bit_rshift(b1, 2)]
        out[oi + 2] = ENC[bit_bor(bit_lshift(bit_band(b1, 0x03), 4), bit_rshift(b2, 4))]
        out[oi + 3] = ENC[bit_lshift(bit_band(b2, 0x0F), 2)]
        out[oi + 4] = 0x3D -- '='
        oi = oi + 4
    end

    -- `out` is a flat array of byte values; convert to a string in one shot.
    return string_char(unpack(out, 1, oi))
end

-- ── Decode ────────────────────────────────────────────────────────────────

local PAD_BYTE = 0x3D -- '='

--- Decode a Base64 ASCII `src` string back to bytes.
---
--- Whitespace inside the input is silently ignored (matches RFC 4648
--- common practice). Invalid characters or malformed padding return
--- `nil` rather than raising, mirroring the bridge's `-1` sentinel.
---
--- @param  src string Base64-encoded input.
--- @return string|nil Decoded bytes, or `nil` on parse error.
function M.decode(src)
    if not src or src == "" then
        return ""
    end

    -- First pass : strip whitespace so the rest of the routine can assume a
    -- clean stream where each non-`=` char maps to a 6-bit value.
    local cleaned, ci = {}, 0
    for i = 1, #src do
        local b = string_byte(src, i)
        -- ASCII whitespace : space, tab, LF, CR, VT, FF.
        if b ~= 0x20 and b ~= 0x09 and b ~= 0x0A and b ~= 0x0D and b ~= 0x0B and b ~= 0x0C then
            ci = ci + 1
            cleaned[ci] = b
        end
    end
    if ci == 0 then
        return ""
    end
    if (ci % 4) ~= 0 then
        return nil
    end

    -- Output capacity : 3 bytes per quartet, minus 1 per `=` padding char.
    local pad = 0
    if cleaned[ci] == PAD_BYTE then
        pad = pad + 1
    end
    if cleaned[ci - 1] == PAD_BYTE then
        pad = pad + 1
    end
    local out, oi = {}, 0

    for i = 1, ci, 4 do
        local v1 = DEC[cleaned[i]]
        local v2 = DEC[cleaned[i + 1]]
        if not v1 or not v2 then
            return nil
        end

        local b3 = cleaned[i + 2]
        local b4 = cleaned[i + 3]
        local v3 = (b3 == PAD_BYTE) and 0 or DEC[b3]
        local v4 = (b4 == PAD_BYTE) and 0 or DEC[b4]
        if not v3 or not v4 then
            return nil
        end

        out[oi + 1] = bit_bor(bit_lshift(v1, 2), bit_rshift(v2, 4))
        out[oi + 2] = bit_band(bit_bor(bit_lshift(v2, 4), bit_rshift(v3, 2)), 0xFF)
        out[oi + 3] = bit_band(bit_bor(bit_lshift(v3, 6), v4), 0xFF)
        oi = oi + 3
    end

    -- Trim the bytes that correspond to `=` padding chars.
    return string_char(unpack(out, 1, oi - pad))
end

return M
