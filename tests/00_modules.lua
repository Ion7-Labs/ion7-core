#!/usr/bin/env luajit
--- @module tests.00_modules
--- @author  ion7 / Ion7 Project Contributors
---
--- Verify that EVERY Lua module under `src/ion7/core/` loads cleanly,
--- WITHOUT linking the underlying shared libraries (`libllama.so`,
--- `ion7_bridge.so`).
---
--- Why a "load only" suite first
--- ─────────────────────────────
--- A normal smoke test that loads a model already detects most
--- regressions, but it has a glaring blind spot : if a Lua module has
--- a typo in a `require` path, a syntax error, or a missing FFI symbol
--- it never references on the happy path, the bug only surfaces the
--- day a user calls into that code path. By walking every `.lua` file
--- under `src/ion7/core/` and `require`-ing it through `pcall`, we
--- catch :
---
---   - typos in `require "ion7.core.xxx"`,
---   - syntax errors in any module,
---   - circular requires that crash on first use,
---   - mismatched FFI symbol names (a `cdef` parses fine, but a
---     `local fn = llama_x.does_not_exist` would be a load-time error),
---   - missing files referenced from `init.lua`.
---
--- All before we burn 5 GB of VRAM loading a model.
---
--- Why FFI is stubbed
--- ──────────────────
--- A module that does `local L = ffi.load("libllama")` at file scope
--- would otherwise force this suite to depend on a built `.so`. We
--- monkey-patch `ffi.load` to return a "noop" namespace : an empty
--- table with a metatable that turns any `__index` into a function
--- that does nothing. The cdef + require chain still has to resolve,
--- but no FFI call is actually crossed.
---
--- The bridge module additionally calls into the .so at require time,
--- so we register a `package.preload` entry for it that hands back the
--- same stub table.
---
--- Exit status
--- ───────────
---   0 — every module loaded cleanly.
---   1 — at least one module failed (failures printed on stderr).

local T = require "tests.framework"
require "tests.helpers" -- sets package.path for the in-tree sources

T.suite("Module load — every file under src/ion7/core/ requires cleanly")

-- ── Stub the FFI surface ────────────────────────────────────────────────
--
-- The stub table swallows every access. It has to be callable as a
-- ffi.load result (so we hand back functions that return zero / nil),
-- and it has to be safe to use as a "value" container so that things
-- like `local C = ffi.load(...) ; local fn = C.foo` simply yield a
-- noop function.

local ffi = require "ffi"

local _stub = setmetatable({}, {
    __index = function() return function() end end,
    __call  = function() return nil end,
})

-- Patch ffi.load globally for the duration of this test process.
ffi.load = function() return _stub end

-- Some modules have side-effecting `ffi.load` calls AT REQUIRE TIME
-- (e.g. the bridge does its own dlopen on require). Pre-register a
-- preload entry that returns the stub instead.
package.preload["ion7.core.ffi.bridge"] = function() return _stub end

-- ── Walk the src tree ────────────────────────────────────────────────────
--
-- We use `find` instead of LuaFileSystem to avoid an external
-- dependency. The `-not -path '*/.old/*'` filter skips any `.old/`
-- subtrees a developer may keep in their working copy.

--- List every `.lua` file under `dir`, excluding archived `.old/`
--- subtrees. Returned paths are relative to `dir`.
--- @param  dir string
--- @return string[]
local function find_lua(dir)
    local out = {}
    local p = io.popen(
        "find " .. dir ..
        " -type f -name '*.lua' -not -path '*/.old/*'"
    )
    if not p then return out end
    for line in p:lines() do out[#out + 1] = line end
    p:close()
    return out
end

local files = find_lua("./src/ion7/core")
table.sort(files)

T.test("at least one module discovered (sanity check)", function()
    T.gt(#files, 0, "find returned no .lua files — wrong CWD?")
end)

-- ── Try to require each file ────────────────────────────────────────────
--
-- We translate the relative path into a Lua module name :
---   ./src/ion7/core/model/inspect.lua → ion7.core.model.inspect
---   ./src/ion7/core/init.lua          → ion7.core
--
-- Each `require` is wrapped in `pcall` so one broken module does not
-- abort the whole walk — we want to see every failure in one run.

for _, file in ipairs(files) do
    local mod = file
        :gsub("^%./src/", "")
        :gsub("%.lua$",   "")
        :gsub("/",        ".")
        :gsub("%.init$",  "")

    T.test("require '" .. mod .. "'", function()
        local ok, err = pcall(require, mod)
        if not ok then
            error(tostring(err):sub(1, 400), 0)
        end
    end)
end

-- ── Final verdict ───────────────────────────────────────────────────────

local ok = T.summary()
os.exit(ok and 0 or 1)
