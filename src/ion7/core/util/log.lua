--- @module ion7.core.log
--- @author  ion7 / Ion7 Project Contributors
---
--- Routing for llama.cpp / ggml log output. Originally a 3-global module
--- in the C bridge (`g_log_level`, `g_log_file`, `g_log_timestamps`); now
--- a Lua-side callback registered via the auto-generated FFI binding to
--- `llama_log_set`.
---
--- The level model matches the historical bridge for source compatibility :
---
---    level | what passes through
---   -------+----------------------------------------------
---     0    | silent — no output
---     1    | error only                          (default)
---     2    | error + warn
---     3    | error + warn + info
---     4    | error + warn + info + debug
---
--- Internally the user level is mapped to ggml's `GGML_LOG_LEVEL_*` enum
--- (DEBUG=1, INFO=2, WARN=3, ERROR=4) by `5 - user_level` to derive the
--- minimum ggml level allowed through.
---
--- LuaJIT FFI gotcha :
---   `ffi.cast("ggml_log_callback", fn)` returns a cdata pointer that
---   MUST stay reachable from Lua for as long as the callback is
---   registered C-side — otherwise the GC may collect the trampoline
---   and llama.cpp will crash on its next log call. We anchor it on the
---   module-local `state.callback_cdata`.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama = require "ion7.core.ffi.llama.backend" -- llama_log_set

local ffi_string = ffi.string
local ffi_cast = ffi.cast
local io_stderr = io.stderr

local M = {}

-- ── ggml_log_level constants ──────────────────────────────────────────────
-- Cached locally so the dispatch hot path does not pay the FFI enum lookup.
local GGML_LOG_LEVEL_NONE = 0
local GGML_LOG_LEVEL_DEBUG = 1
local GGML_LOG_LEVEL_INFO = 2
local GGML_LOG_LEVEL_WARN = 3
local GGML_LOG_LEVEL_ERROR = 4
-- CONT (5) is intentionally not named : it means "continuation of the
-- previous message" and always passes when its predecessor passed.

-- ── Module state ──────────────────────────────────────────────────────────

local state = {
    level = 1, --- @type integer  user-facing level (see header doc)
    file = nil, --- @type file*?   destination, nil → stderr
    timestamps = false, --- @type boolean  prepend ISO-8601 ts on each line
    callback_cdata = nil, --- @type cdata?   anchored ffi.cast — DO NOT drop
    installed = false --- @type boolean  is our callback active C-side?
}

-- ── The actual log dispatcher ─────────────────────────────────────────────

--- C-side dispatcher : invoked by llama.cpp / ggml whenever they emit a
--- log message. Receives the ggml level, a NUL-terminated `const char*`
--- and an opaque user_data pointer (we always pass NULL).
---
--- Filtering happens here rather than C-side because the level threshold
--- is owned by `state.level` and can be flipped at runtime by `set_level`.
---
--- @param  level integer  Raw `ggml_log_level` enum value.
--- @param  text  cdata    `const char*` pointing to the message text.
--- @param  ud    cdata?   `void*` user data (unused, kept for sig).
local function dispatch(level, text, ud)
    if state.level <= 0 then
        return
    end
    local threshold = 5 - state.level
    -- ggml emits enums as cdata ; tonumber() forces integer comparison.
    if tonumber(level) < threshold then
        return
    end
    local dst = state.file or io_stderr
    if state.timestamps then
        dst:write(os.date("%Y-%m-%dT%H:%M:%S "))
    end
    dst:write(ffi_string(text))
    if state.file then
        dst:flush()
    end
end

-- ── Public API ────────────────────────────────────────────────────────────

--- Set the verbosity threshold. See module header for the mapping.
--- Takes effect immediately for subsequent log calls.
--- @param  level integer 0 (silent) to 4 (debug).
function M.set_level(level)
    state.level = level or 0
end

--- Direct log output to a file path, or restore stderr when `path` is
--- nil / empty. The previous file handle (if any) is closed.
--- @param  path string|nil File path, or nil for stderr.
function M.to_file(path)
    if state.file then
        state.file:close()
        state.file = nil
    end
    if path and path ~= "" then
        state.file = io.open(path, "a")
    end
end

--- Toggle ISO-8601 timestamp prefix on each emitted line.
--- @param  enable boolean
function M.set_timestamps(enable)
    state.timestamps = enable and true or false
end

--- Register our dispatcher with llama.cpp. Idempotent : a second call is
--- a no-op. Call once at startup, before any model load, so even the
--- backend init messages route through the configured destination.
function M.install()
    if state.installed then
        return
    end
    -- Anchor the cast cdata on the module so the GC keeps it alive for
    -- as long as `llama_log_set` holds a reference to the trampoline.
    state.callback_cdata = ffi_cast("ggml_log_callback", dispatch)
    llama.llama_log_set(state.callback_cdata, nil)
    state.installed = true
end

--- Detach our dispatcher and restore llama.cpp's default logging
--- (which writes to stderr unconditionally).
function M.uninstall()
    if not state.installed then
        return
    end
    llama.llama_log_set(nil, nil)
    -- Drop the anchor so the trampoline can be GC'd.
    state.callback_cdata = nil
    state.installed = false
end

--- Inspect the current configuration (useful for tests / diagnostics).
--- @return table { level, has_file, timestamps, installed }
function M.snapshot()
    return {
        level = state.level,
        has_file = state.file ~= nil,
        timestamps = state.timestamps,
        installed = state.installed
    }
end

return M
