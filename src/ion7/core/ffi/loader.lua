-- src/ion7/core/ffi/loader.lua
-- Loads the native shared libs (libllama, libggml) and exposes their handles.
-- Resolution order: ION7_LIBLLAMA_PATH > vendor/llama.cpp/build/ > system.

local ffi = require "ffi"
require "ion7.core.ffi.types"   -- ensure types are ready

-- Filter nil candidates (env vars not set) up front : `ipairs` would stop
-- at the first nil and `table.concat` in the error message would crash on
-- a sparse table.
local function _try_load(name, candidates)
  local clean, n = {}, 0
  for _, p in pairs(candidates) do
    if p and p ~= "" then n = n + 1; clean[n] = p end
  end
  for _, path in ipairs(clean) do
    local ok, lib = pcall(function() return ffi.load(path) end)
    if ok then return lib end
  end
  error(string.format("[ion7-core] %s not found. Candidates: %s",
    name, table.concat(clean, " | ")))
end

--- Resolve the directory of THIS file. Used to find libraries that the
--- rockspec's install step copies into `<package>/_libs/` next to the
--- top-level `core/` directory.
local function _module_dir()
  local info = debug.getinfo(1, "S")
  local src  = info and info.source or ""
  if src:sub(1, 1) ~= "@" then return nil end
  return (src:sub(2):gsub("/[^/]*$", ""))
end

local function _candidates(env_var, base_name)
  local mdir = _module_dir()
  return {
    os.getenv(env_var),
    mdir and (mdir .. "/../_libs/lib" .. base_name .. ".so"),
    mdir and (mdir .. "/../_libs/lib" .. base_name .. ".dylib"),
    "vendor/llama.cpp/build/bin/lib" .. base_name .. ".so",
    "vendor/llama.cpp/build/lib/lib" .. base_name .. ".so",
    "vendor/llama.cpp/build/" .. base_name .. ".dll",
    "/usr/local/lib/lib" .. base_name .. ".so",
    "lib" .. base_name .. ".so",
  }
end

local M = {}

M.llama = _try_load("libllama", _candidates("ION7_LIBLLAMA_PATH", "llama"))
M.ggml  = _try_load("libggml",  _candidates("ION7_LIBGGML_PATH",  "ggml"))
-- libggml-vulkan is usually statically linked into libggml — uncomment if not:
-- M.ggml_vulkan = _try_load("libggml-vulkan", _candidates("ION7_LIBGGML_VULKAN_PATH", "ggml-vulkan"))
return M
