--- @module ffi_gen.util
--- @author  ion7 / generate_ffi.lua
---
--- Path, file and string helpers shared across the generator. Pure stdlib,
--- no external dependencies. Performance-sensitive call sites cache the
--- relevant function locally; the API is designed for that.

local M = {}

-- ── Hot-path local aliases (LuaJIT canon: faster than global lookups) ──────
local io_open       = io.open
local os_execute    = os.execute
local os_getenv     = os.getenv
local os_tmpname    = os.tmpname
local string_gsub   = string.gsub
local string_sub    = string.sub
local string_format = string.format
local table_concat  = table.concat
local table_sort    = table.sort

--- Whether the host platform is Windows (path separator detection).
--- @type boolean
M.IS_WINDOWS = (package.config:sub(1, 1) == "\\")

-- ── String / path helpers ──────────────────────────────────────────────────

-- basename() is called many times on the same paths during AST walking, so we
-- memoise it. The cache is bounded in practice (one entry per unique header).
local _basename_cache = {}

--- Return the final path component (filename) of a path.
--- Forward and backward slashes are both accepted.
--- @param  path string Filesystem path.
--- @return string     The basename (last `/` or `\\` segment).
function M.basename(path)
  local cached = _basename_cache[path]
  if cached then return cached end
  local n = (string_gsub(path, "\\", "/"):match("([^/]+)$")) or path
  _basename_cache[path] = n
  return n
end

--- Normalise backslashes to forward slashes. Useful when comparing paths
--- coming back from clang on Windows (mixed separators are common).
--- @param  path string A possibly Windows-flavoured path.
--- @return string     Path with `/` separators only.
function M.normalize_path(path)
  return (string_gsub(path, "\\", "/"))
end

--- Strip leading and trailing whitespace (spaces, tabs, newlines).
--- @param  s string
--- @return string
function M.trim(s)
  return (string_gsub(string_gsub(s, "^%s+", ""), "%s+$", ""))
end

--- Test whether `s` starts with `prefix` (cheaper than `s:match("^"..prefix)`).
--- @param  s      string
--- @param  prefix string
--- @return boolean
function M.starts_with(s, prefix)
  return string_sub(s, 1, #prefix) == prefix
end

--- Indent every line of `s` with `prefix`. Empty lines are preserved as-is
--- (i.e. NOT prefixed).
--- @param  s      string
--- @param  prefix string Indentation to prepend on non-empty lines.
--- @return string
function M.indent(s, prefix)
  local out, n = {}, 0
  for line in (s .. "\n"):gmatch("([^\n]*)\n") do
    n = n + 1
    out[n] = (#line > 0) and (prefix .. line) or line
  end
  -- gmatch leaves a trailing empty line we want to drop.
  if out[n] == "" then out[n] = nil end
  return table_concat(out, "\n")
end

--- Return a table's keys, sorted alphabetically.
--- @generic K
--- @param   t table<K, any>
--- @return  K[]
function M.sorted_keys(t)
  local keys, n = {}, 0
  for k in pairs(t) do n = n + 1; keys[n] = k end
  table_sort(keys)
  return keys
end

-- ── File I/O ───────────────────────────────────────────────────────────────

--- Test whether a file exists and is readable.
--- @param  path string
--- @return boolean
function M.file_exists(path)
  local f = io_open(path, "rb")
  if f then f:close(); return true end
  return false
end

--- Read an entire file into memory (binary mode, byte-exact).
--- @param  path string
--- @return string|nil contents Whole file as a string, or nil on error.
--- @return string|nil err      Error message if reading failed.
function M.read_file(path)
  local f, err = io_open(path, "rb")
  if not f then return nil, err end
  local s = f:read("*a")
  f:close()
  return s
end

--- Write `content` to `path`, overwriting any prior contents (binary mode).
--- @param  path    string
--- @param  content string
--- @return boolean|nil ok True on success, nil otherwise.
--- @return string|nil   err Error message on failure.
function M.write_file(path, content)
  local f, err = io_open(path, "wb")
  if not f then return nil, err end
  f:write(content)
  f:close()
  return true
end

--- Recursively create a directory (no-op if it already exists). Best-effort
--- — failures are silenced because the next file write will surface them.
--- @param  path string
--- @return nil
function M.mkdir_p(path)
  -- Try Unix-style first (works in Git Bash and Linux).
  os_execute("mkdir -p '" .. path .. "' 2>/dev/null")
  -- Fallback: Windows cmd. `mkdir` creates intermediate dirs by default and
  -- swallows the error if the leaf already exists.
  if M.IS_WINDOWS then
    os_execute('mkdir "' .. string_gsub(path, "/", "\\") .. '" 2>nul')
  end
end

-- ── Subprocess helpers ─────────────────────────────────────────────────────

--- Normalise the return value of `os.execute` between Lua 5.1 (number) and
--- Lua 5.2+ / LuaJIT-extended (boolean + status table).
--- @param  rc any
--- @return boolean ok True if the command exited cleanly.
function M.exec_ok(rc)
  return rc == 0 or rc == true
end

--- Build a unique temp file path. Cross-platform: handles both the Linux
--- shape (`/tmp/lua_xxxx`) and the Windows shape (a full
--- `%TEMP%\\xxxx.` path). We extract just the random-looking trailing
--- component and rebuild a clean `<dir>/ion7_ffi_<id><suffix>` path.
--- @param  suffix string|nil Optional extension (e.g. `".json"`).
--- @return string
function M.tmp_path(suffix)
  local raw = os_tmpname()
  -- Keep only the last path segment, drop any trailing dot.
  local base = raw:match("([^/\\]+)$") or raw
  base = string_gsub(base, "%.$", "")
  local dir
  if M.IS_WINDOWS then
    dir = os_getenv("TEMP") or os_getenv("TMP") or "C:/Windows/Temp"
    dir = string_gsub(dir, "\\", "/")
  else
    dir = "/tmp"
  end
  return string_format("%s/ion7_ffi_%s%s", dir, base, suffix or "")
end

--- Wrap a shell command for safe execution under Windows `cmd.exe`. cmd
--- strips ONE pair of outer double-quotes from `cmd /c "..."`, so commands
--- whose exe path contains spaces (e.g. `C:\Program Files\...`) need their
--- ENTIRE pipeline re-wrapped. On Unix shells, returns the command verbatim.
--- @param  cmd string
--- @return string
function M.wrap_for_shell(cmd)
  if M.IS_WINDOWS then return '"' .. cmd .. '"' end
  return cmd
end

return M
