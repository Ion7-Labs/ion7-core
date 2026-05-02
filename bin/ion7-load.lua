#!/usr/bin/env luajit
--- @module bin.ion7-load
--- @author  ion7 / Ion7 Project Contributors
---
--- Preamble loader for distributable tarballs of ion7-core.
---
--- A `bin/ion7-load.lua` file ships at the root of every release
--- tarball. Calling `dofile` on it from any working directory wires
--- the three things `require "ion7.core"` needs in order to find the
--- bundled copy of the library :
---
---   1. `package.path`   so `require "ion7.core"` resolves to the
---      Lua runtime under `<install>/src/ion7/...` rather than any
---      system-installed clone.
---   2. `ION7_LIBLLAMA_PATH` env var so the FFI loader picks the
---      `libllama.*` shipped under `<install>/lib/` instead of
---      whatever the system loader finds first.
---   3. `ION7_BRIDGE_PATH` env var so the bridge module loads the
---      `ion7_bridge.*` placed alongside `libllama.*`.
---
--- Typical use :
---
---   dofile("/opt/ion7-core/bin/ion7-load.lua")
---   local ion7 = require "ion7.core"
---   ion7.init({ log_level = 0 })

local ffi = require "ffi"

-- ── Locate ourselves on disk ─────────────────────────────────────────────
--
-- `debug.getinfo(1, "S").source` is the source path the script was
-- loaded from, prefixed with `@`. Two parents up gets us out of
-- `<install>/bin/` and back to the install root.

local function script_path()
    local s = debug.getinfo(1, "S").source
    if s:sub(1, 1) == "@" then s = s:sub(2) end
    return s
end

local function dirname(p)
    return p:match("^(.+)[/\\][^/\\]+$") or "."
end

local SCRIPT  = script_path()
local INSTALL = dirname(dirname(SCRIPT))    -- parent of bin/

-- Path separator. On Windows package.config:sub(1,1) == "\\".
local SEP = package.config:sub(1, 1)

local LIB_DIR = INSTALL .. SEP .. "lib"
local SRC_DIR = INSTALL .. SEP .. "src"

-- ── 1. Lua module path ───────────────────────────────────────────────────
package.path = SRC_DIR .. SEP .. "?.lua;" ..
               SRC_DIR .. SEP .. "?" .. SEP .. "init.lua;" ..
               package.path

-- ── 2. Native loader env vars ────────────────────────────────────────────
--
-- LuaJIT 2.x has no `os.setenv`. We poke libc directly through the
-- FFI ; both Linux (`setenv`) and Windows (`_putenv_s`) are covered.
-- The result is visible to the SAME process via `os.getenv`, which
-- is what the ion7 loader reads.

ffi.cdef [[
    int setenv  (const char *name, const char *value, int overwrite);
    int _putenv_s(const char *name, const char *value);
]]

local function set_env(name, value)
    local ok = pcall(function() ffi.C.setenv(name, value, 1) end)
    if not ok then
        pcall(function() ffi.C._putenv_s(name, value) end)
    end
end

-- Prefer `.so` (Linux) ; fall back to `.dylib` (macOS) and `.dll`
-- (Windows). We only set the env var to the FIRST file that exists —
-- ion7.core.ffi.loader handles the rest of the search list.
local function first_existing(candidates)
    for _, p in ipairs(candidates) do
        local f = io.open(p, "rb")
        if f then f:close() ; return p end
    end
    return nil
end

local function pick(stem)
    -- libllama / libggml ship as soversion-suffixed files on Linux
    -- (`libllama.so.0`) ; we point at the unsuffixed symlink that the
    -- packaging step also copies. macOS has no soversion, Windows
    -- uses `.dll`. Probe all three suffixes in that order.
    return first_existing({
        LIB_DIR .. SEP .. stem .. ".so",
        LIB_DIR .. SEP .. stem .. ".dylib",
        LIB_DIR .. SEP .. stem .. ".dll",
    })
end

local libllama = pick("libllama")
local libggml  = pick("libggml")
local bridge   = pick("ion7_bridge")

if libllama then set_env("ION7_LIBLLAMA_PATH", libllama) end
if libggml  then set_env("ION7_LIBGGML_PATH",  libggml)  end
if bridge   then set_env("ION7_BRIDGE_PATH",   bridge)   end

-- ── 3. Return the install root for callers that want it ─────────────────
return INSTALL
