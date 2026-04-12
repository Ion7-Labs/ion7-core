--- @module ion7.core.model
--- SPDX-License-Identifier: MIT
--- Top-level model object: loading, context creation, vocabulary, metadata.
---
--- A Model wraps a llama_model* loaded from a GGUF file. It is the root
--- of the ion7-core object graph. All contexts and vocabulary handles are
--- created from it.
---
--- A single Model may host multiple simultaneous Context instances, enabling
--- multi-session setups where a cheap embedding context and a heavier
--- generation context share the same weights.
---
--- Resource management: the llama_model* is freed automatically when the
--- Model object is garbage-collected. All Context instances keep a Lua
--- reference to their parent Model, so the GC ordering is safe.
---
--- @usage
---   local llama = require "ion7.core"
---   llama.init({ log_level = 1 })
---
---   local model = llama.Model.load("qwen3.5-27b.gguf", { n_gpu_layers = 25 })
---   print(model:desc())           -- "Qwen3 27B Q4_K_M"
---   print(model:n_params() / 1e9) -- 27.36 (billions)
---   print(model:is_recurrent())   -- false
---
---   local ctx   = model:context({ n_ctx = 65536, kv_type = "q8_0" })
---   local vocab = model:vocab()

local Loader = require "ion7.core.ffi.loader"

-- ── Model ─────────────────────────────────────────────────────────────────────

--- @class Model
--- @field _ptr   cdata   llama_model* (freed on GC).
--- @field path   string  Path to the source GGUF file.
local Model = {}
Model.__index = Model

-- ── Constructors ──────────────────────────────────────────────────────────────

--- Load a GGUF model from disk.
---
--- @param  path  string  Absolute path to the .gguf file.
--- @param  opts  table?
---   opts.n_gpu_layers  number?  Layers to offload. 0=CPU, -1=all (default: 0).
---   opts.use_mmap      bool?    Memory-map the file (default: true).
---   opts.use_mlock     bool?    Lock pages in RAM (default: false).
---   opts.vocab_only    bool?    Load vocabulary only, no weights (default: false).
--- @return Model
--- @error  If the file cannot be loaded.
function Model.load(path, opts)
    assert(type(path) == "string" and #path > 0,
        "[ion7.core.model] path must be a non-empty string")
    opts = opts or {}
    local L = Loader.instance()

    local ptr = L.bridge.ion7_model_load(
        path,
        opts.n_gpu_layers or 0,
        opts.use_mmap ~= false and 1 or 0,
        opts.use_mlock  and 1 or 0,
        opts.vocab_only and 1 or 0
    )
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load '%s'", path), 2)
    end

    return setmetatable({
        _ptr    = L.ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib,
        _bridge = L.bridge,
        _ffi    = L.ffi,
        _vocab  = nil,
        path    = path,
    }, Model)
end

--- Load a sharded GGUF model from multiple files.
---
--- @param  paths  table   Array of paths to GGUF shard files (in shard order).
--- @param  opts   table?  Same options as Model.load().
--- @return Model
--- @error  If loading fails.
function Model.load_splits(paths, opts)
    assert(type(paths) == "table" and #paths > 0,
        "[ion7.core.model] paths must be a non-empty table of strings")
    opts = opts or {}
    collectgarbage("collect")
    local L   = Loader.instance()
    local ffi = L.ffi
    local n   = #paths
    local arr = ffi.new("const char*[?]", n)
    local _refs = {}
    for i, p in ipairs(paths) do _refs[i] = p; arr[i-1] = p end
    local ptr = L.bridge.ion7_model_load_splits(arr, n, opts.n_gpu_layers or 0)
    if ptr == nil then
        error(string.format(
            "[ion7.core.model] failed to load sharded model (%d parts)", n), 2)
    end
    return setmetatable({
        _ptr    = ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib, _bridge = L.bridge, _ffi = ffi, _vocab = nil,
        path    = paths[1],
    }, Model)
end

--- Load a model from an already-open FILE* handle.
---
--- The caller owns the file - it is NOT closed by this call.
---
--- @param  file  cdata   FILE* (from ffi.C.fopen or similar).
--- @param  opts  table?
---   opts.n_gpu_layers  number?  (default: 0)
--- @return Model
--- @error  If loading fails.
function Model.load_from_fd(file, opts)
    assert(file ~= nil, "[ion7.core.model] load_from_fd: FILE* is nil")
    opts = opts or {}
    local L   = Loader.instance()
    local ptr = L.bridge.ion7_model_load_fd(file, opts.n_gpu_layers or 0)
    if ptr == nil then
        error("[ion7.core.model] load_from_fd: failed to load model", 2)
    end
    return setmetatable({
        _ptr    = L.ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib,
        _bridge = L.bridge,
        _ffi    = L.ffi,
        _vocab  = nil,
        path    = "<file_ptr>",
    }, Model)
end

--- Auto-fit model and context parameters to available device VRAM.
---
--- @param  path  string  Path to GGUF file (for size estimation).
--- @param  opts  table?
---   opts.n_ctx      number?  Desired context size (default: 4096).
---   opts.n_ctx_min  number?  Minimum acceptable context (default: 512).
--- @return table?  { n_gpu_layers, n_ctx } or nil if no fit found.
function Model.fit_params(path, opts)
    opts = opts or {}
    local L   = Loader.instance()
    local ffi = L.ffi
    local n_gpu = ffi.new("int32_t[1]",  0)
    local n_ctx = ffi.new("uint32_t[1]", opts.n_ctx or 4096)
    local min   = opts.n_ctx_min or 512
    local status = L.bridge.ion7_params_fit(path, n_gpu, n_ctx, min)
    if status ~= 0 then return nil end
    return {
        n_gpu_layers = tonumber(n_gpu[0]),
        n_ctx        = tonumber(n_ctx[0]),
    }
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────────

--- Return the raw llama_model*.
--- @return cdata
function Model:ptr() return self._ptr end

--- Explicitly free the model and release VRAM immediately.
--- After free(), this object must not be used.
function Model:free()
    if self._ptr then
        self._bridge.ion7_model_free(self._ptr)
        self._ptr = self._ffi.gc(self._ptr, nil)
        self._ptr = nil
        self._vocab = nil
    end
end

--- Save the model to a GGUF file.
--- Useful after applying quantization or merging LoRA adapters.
--- @param  path  string  Output file path.
function Model:save(path)
    self._bridge.ion7_model_save(self._ptr, path)
end

-- ── Sub-modules ───────────────────────────────────────────────────────────────
-- Each sub-module returns a plain table of methods (self = Model instance).
-- Iterating and assigning them here keeps this file slim while preserving
-- the single public require path: require "ion7.core.model".

for k, v in pairs(require "ion7.core.model.inspect")          do Model[k] = v end
for k, v in pairs(require "ion7.core.model.meta")             do Model[k] = v end
for k, v in pairs(require "ion7.core.model.lora")             do Model[k] = v end
for k, v in pairs(require "ion7.core.model.context_factory")  do Model[k] = v end
for k, v in pairs(require "ion7.core.model.quantize")         do Model[k] = v end

return Model
