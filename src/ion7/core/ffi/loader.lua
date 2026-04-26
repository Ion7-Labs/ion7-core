-- src/ion7/core/ffi/loader.lua
-- Loads the native shared libs (libllama, libggml) and exposes their handles.
-- Resolution order: ION7_LIBLLAMA_PATH > vendor/llama.cpp/build/ > system.

local ffi = require "ffi"
require "ion7.core.ffi.types"   -- ensure types are ready

local function _try_load(name, candidates)
  for _, path in ipairs(candidates) do
    if path then
      local ok, lib = pcall(function() return ffi.load(path) end)
      if ok then return lib end
    end
  end
  error(string.format("[ion7-core] %s not found. Candidates: %s",
    name, table.concat(candidates, " | ")))
end

local function _candidates(env_var, base_name)
  return {
    os.getenv(env_var),
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
