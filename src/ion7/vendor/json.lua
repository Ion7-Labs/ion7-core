--- @module ion7.vendor.json
--- SPDX-License-Identifier: MIT
---
--- Minimal JSON encoder / decoder - pure Lua, zero dependencies.
---
--- Handles: objects, arrays, strings (full escape sequences + \uXXXX),
---          numbers, booleans, null (decoded as nil).
---
--- This is bundled as a vendor lib because the bridge already links
--- nlohmann/json for C++ operations. This module handles the Lua side:
--- parsing JSON strings returned by bridge functions (e.g. tool call arrays
--- from ion7_chat_parse) into Lua tables, and serialising Lua tables back
--- to JSON for building schemas or request bodies.
---
--- @usage
---   local json = require "ion7.vendor.json"
---   local t  = json.decode('{"name":"Joi","version":2}')
---   local s  = json.encode({ role = "user", content = "hello" })
---   local ok = pcall(json.decode, "not json")  -- false

local json = {}

-- ── Decode ────────────────────────────────────────────────────────────────

local function skip_ws(s, i)
    while i <= #s do
        local b = s:byte(i)
        if b ~= 32 and b ~= 9 and b ~= 10 and b ~= 13 then break end
        i = i + 1
    end
    return i
end

local decode_value  -- forward declaration

local function decode_string(s, i)
    local buf, j = {}, i + 1   -- skip opening "
    while j <= #s do
        local c = s:sub(j, j)
        if c == '"' then
            return table.concat(buf), j + 1
        elseif c == '\\' then
            j = j + 1
            c = s:sub(j, j)
            if     c == '"'  then buf[#buf+1] = '"'
            elseif c == '\\' then buf[#buf+1] = '\\'
            elseif c == '/'  then buf[#buf+1] = '/'
            elseif c == 'n'  then buf[#buf+1] = '\n'
            elseif c == 'r'  then buf[#buf+1] = '\r'
            elseif c == 't'  then buf[#buf+1] = '\t'
            elseif c == 'b'  then buf[#buf+1] = '\8'
            elseif c == 'f'  then buf[#buf+1] = '\12'
            elseif c == 'u'  then
                local hex = s:sub(j+1, j+4)
                local cp  = tonumber(hex, 16) or 0
                j = j + 4
                if cp < 0x80 then
                    buf[#buf+1] = string.char(cp)
                elseif cp < 0x800 then
                    buf[#buf+1] = string.char(
                        0xC0 + math.floor(cp / 64),
                        0x80 + cp % 64)
                else
                    buf[#buf+1] = string.char(
                        0xE0 + math.floor(cp / 4096),
                        0x80 + math.floor(cp / 64) % 64,
                        0x80 + cp % 64)
                end
            else
                buf[#buf+1] = c
            end
        else
            buf[#buf+1] = c
        end
        j = j + 1
    end
    error("[ion7.vendor.json] unterminated string", 2)
end

local function decode_number(s, i)
    local j = i
    -- optional leading minus
    if s:sub(j, j) == '-' then j = j + 1 end
    while j <= #s and s:sub(j, j):match('[%deE%.%+%-]') do j = j + 1 end
    local n = tonumber(s:sub(i, j - 1))
    if not n then error("[ion7.vendor.json] invalid number at " .. i, 2) end
    return n, j
end

local function decode_array(s, i)
    local arr, j = {}, i + 1   -- skip [
    j = skip_ws(s, j)
    if s:sub(j, j) == ']' then return arr, j + 1 end
    while true do
        local v
        v, j = decode_value(s, j)
        arr[#arr+1] = v
        j = skip_ws(s, j)
        local c = s:sub(j, j)
        if     c == ']' then return arr, j + 1
        elseif c ~= ',' then error("[ion7.vendor.json] expected ',' or ']' in array", 2)
        end
        j = skip_ws(s, j + 1)
    end
end

local function decode_object(s, i)
    local obj, j = {}, i + 1   -- skip {
    j = skip_ws(s, j)
    if s:sub(j, j) == '}' then return obj, j + 1 end
    while true do
        if s:sub(j, j) ~= '"' then
            error("[ion7.vendor.json] expected string key in object at " .. j, 2)
        end
        local k
        k, j = decode_string(s, j)
        j = skip_ws(s, j)
        if s:sub(j, j) ~= ':' then
            error("[ion7.vendor.json] expected ':' in object at " .. j, 2)
        end
        j = skip_ws(s, j + 1)
        local v
        v, j = decode_value(s, j)
        obj[k] = v
        j = skip_ws(s, j)
        local c = s:sub(j, j)
        if     c == '}' then return obj, j + 1
        elseif c ~= ',' then error("[ion7.vendor.json] expected ',' or '}' in object", 2)
        end
        j = skip_ws(s, j + 1)
    end
end

decode_value = function(s, i)
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if     c == '"'  then return decode_string(s, i)
    elseif c == '['  then return decode_array(s, i)
    elseif c == '{'  then return decode_object(s, i)
    elseif c == 't'  then
        if s:sub(i, i+3) == 'true'  then return true,  i + 4 end
    elseif c == 'f'  then
        if s:sub(i, i+4) == 'false' then return false, i + 5 end
    elseif c == 'n'  then
        if s:sub(i, i+3) == 'null'  then return nil,   i + 4 end
    elseif c == '-' or c:match('%d') then
        return decode_number(s, i)
    end
    error("[ion7.vendor.json] unexpected token at position " .. i ..
          ": '" .. c .. "'", 2)
end

--- Decode a JSON string into a Lua value.
--- @param  s  string  JSON input.
--- @return    any     Lua value (table, string, number, boolean, or nil for JSON null).
--- @error     If the input is not valid JSON.
function json.decode(s)
    assert(type(s) == "string", "[ion7.vendor.json] decode() expects a string")
    local v = decode_value(s, 1)
    return v
end

-- ── Encode ────────────────────────────────────────────────────────────────

-- Single-char escape table
local ESC = {
    ['"']  = '\\"',  ['\\'] = '\\\\',
    ['\n'] = '\\n',  ['\r'] = '\\r',
    ['\t'] = '\\t',  ['\8'] = '\\b',
    ['\12'] = '\\f',
}
-- Control characters 0x00-0x1F without explicit entries above
for i = 0, 0x1F do
    local ch = string.char(i)
    if not ESC[ch] then
        ESC[ch] = string.format('\\u%04x', i)
    end
end

local encode  -- forward declaration

local function encode_string(v)
    return '"' .. v:gsub('[\1-\31"\\]', ESC):gsub('%z', ESC) .. '"'
end

local function encode_array(t, n)
    local parts = {}
    for idx = 1, n do
        parts[idx] = encode(t[idx])
    end
    return '[' .. table.concat(parts, ',') .. ']'
end

local function encode_object(t)
    local parts = {}
    for k, v in pairs(t) do
        if type(k) == 'string' then
            parts[#parts+1] = encode_string(k) .. ':' .. encode(v)
        end
    end
    return '{' .. table.concat(parts, ',') .. '}'
end

encode = function(v)
    local t = type(v)
    if     t == 'string'  then
        return encode_string(v)
    elseif t == 'number'  then
        if v ~= v         then return '"nan"'  end   -- NaN
        if v ==  math.huge then return '"inf"'  end
        if v == -math.huge then return '"-inf"' end
        -- Integer representation when lossless
        if math.floor(v) == v and math.abs(v) < 2^53 then
            return string.format('%d', v)
        end
        return string.format('%.17g', v)
    elseif t == 'boolean' then
        return v and 'true' or 'false'
    elseif t == 'nil'     then
        return 'null'
    elseif t == 'table'   then
        -- Array detection: keys must be consecutive integers 1..n with no extras
        local n = #v
        if n > 0 then
            local count = 0
            for _ in pairs(v) do count = count + 1 end
            if count == n then return encode_array(v, n) end
        end
        return encode_object(v)
    end
    error("[ion7.vendor.json] cannot encode value of type '" .. t .. "'", 2)
end

--- Encode a Lua value as a JSON string.
--- Tables with consecutive integer keys become JSON arrays.
--- Tables with string keys become JSON objects.
--- nil encodes as "null", NaN/inf as string literals.
--- @param  v  any     Lua value to encode.
--- @return    string  JSON string.
--- @error     If the value contains a type that cannot be serialised.
function json.encode(v)
    return encode(v)
end

return json
