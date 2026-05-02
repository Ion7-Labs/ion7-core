--- @module ion7.core.model
--- @author  ion7 / Ion7 Project Contributors
---
--- Top-level `Model` object : wraps a `llama_model*` loaded from a GGUF
--- file. The Model is the root of the ion7-core object graph — every
--- `Context`, `Vocab` and LoRA adapter is created from one.
---
--- A single Model may host multiple simultaneous Context instances. Use
--- this to share weights between, say, a cheap embedding context and a
--- heavier generation context without paying the model-load cost twice.
---
--- Resource management : the underlying `llama_model*` is freed
--- automatically on garbage collection via `ffi.gc`. Every Context
--- created from a Model holds a Lua-level reference back to its parent,
--- which keeps the GC ordering safe (the model can never be freed while
--- a context still depends on it).
---
---   local ion7  = require "ion7.core"
---   ion7.init({ log_level = 1 })
---
---   local model = ion7.Model.load("qwen3-32b.gguf", { n_gpu_layers = 25 })
---   print(model:desc())              --> "Qwen3 32B Q4_K_M"
---   print(model:n_params() / 1e9)    --> 32.79  (billions)
---
---   local ctx   = model:context({ n_ctx = 65536, kv_type = "q8_0" })
---   local vocab = model:vocab()
---
--- Implementation notes :
---   - All llama.cpp pass-through wrappers from the old bridge are gone.
---     We talk to `llama_model_*` directly via the FFI bindings under
---     `ion7.core.ffi.llama.model`.
---   - VRAM auto-fit (`Model.fit_params`) still goes through the bridge
---     because `common_fit_params` lives in libcommon, not `llama.h`.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_model = require "ion7.core.ffi.llama.model" -- model_*
local llama_misc = require "ion7.core.ffi.llama.misc" -- llama_init_from_model

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local ffi_string = ffi.string

--- @class ion7.core.Model
--- @field _ptr cdata    Underlying `llama_model*` (auto-freed via ffi.gc).
--- @field path string   Origin path (or "<file_ptr>" / "<splits>").
local Model = {}
Model.__index = Model

-- ── Constructors ──────────────────────────────────────────────────────────

--- Build a `llama_model_params` cdata from the user-facing options table.
--- Defaults match `llama_model_default_params` ; only the four flags we
--- expose at the Lua level get overridden.
---
--- @param  opts table?
--- @return cdata `struct llama_model_params`
local function build_model_params(opts)
    local p = llama_model.llama_model_default_params()
    p.n_gpu_layers = opts.n_gpu_layers or 0
    -- `use_mmap` defaults to true in llama.cpp ; we honour the explicit
    -- boolean only when the caller passed `false`.
    if opts.use_mmap == false then
        p.use_mmap = false
    end
    if opts.use_mlock then
        p.use_mlock = true
    end
    if opts.vocab_only then
        p.vocab_only = true
    end
    return p
end

--- Load a GGUF model from disk.
---
--- @param  path string Absolute path to the `.gguf` file.
--- @param  opts table? Optional :
---           `n_gpu_layers` (number, default 0 — 0 = CPU, -1 = all),
---           `use_mmap`     (bool,   default true),
---           `use_mlock`    (bool,   default false),
---           `vocab_only`   (bool,   default false).
--- @return ion7.core.Model
--- @raise   When the file cannot be opened or the model fails to load.
function Model.load(path, opts)
    assert(type(path) == "string" and #path > 0, "[ion7.core.model] path must be a non-empty string")
    opts = opts or {}

    local params = build_model_params(opts)
    local ptr = llama_model.llama_model_load_from_file(path, params)
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load '%s'", path), 2)
    end

    return setmetatable(
        {
            _ptr = ffi_gc(ptr, llama_model.llama_model_free),
            _vocab = nil, -- lazily populated by Model:vocab()
            path = path
        },
        Model
    )
end

--- Load a sharded GGUF model from an explicit list of paths.
---
--- Use when the shards do not follow the standard auto-discovery naming
--- (`model-00001-of-NNNN.gguf`). Paths must be in shard order.
---
--- @param  paths string[] Array of GGUF shard file paths.
--- @param  opts  table?   Same shape as `Model.load`.
--- @return ion7.core.Model
--- @raise   When loading fails.
function Model.load_splits(paths, opts)
    assert(type(paths) == "table" and #paths > 0, "[ion7.core.model] paths must be a non-empty table of strings")
    opts = opts or {}

    -- We hold Lua refs to the path strings in `_anchor` so they cannot be
    -- collected while the const-char-pointer array still references them.
    local n = #paths
    local arr = ffi_new("const char*[?]", n)
    local _anchor = {}
    for i, p in ipairs(paths) do
        _anchor[i] = p
        arr[i - 1] = p
    end

    local params = build_model_params(opts)
    local ptr = llama_model.llama_model_load_from_splits(arr, n, params)
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load sharded model (%d parts)", n), 2)
    end

    return setmetatable(
        {
            _ptr = ffi_gc(ptr, llama_model.llama_model_free),
            _vocab = nil,
            path = paths[1]
        },
        Model
    )
end

--- Load a model from an already-open `FILE*` handle. The caller retains
--- ownership of the FILE — it is NOT closed by this call.
---
--- @param  file cdata `FILE*` (from `ffi.C.fopen` or similar).
--- @param  opts table? Same shape as `Model.load`.
--- @return ion7.core.Model
--- @raise   When loading fails.
function Model.load_from_fd(file, opts)
    assert(file ~= nil, "[ion7.core.model] load_from_fd: FILE* is nil")
    opts = opts or {}

    local params = build_model_params(opts)
    local ptr = llama_model.llama_model_load_from_file_ptr(file, params)
    if ptr == nil then
        error("[ion7.core.model] load_from_fd: failed to load model", 2)
    end

    return setmetatable(
        {
            _ptr = ffi_gc(ptr, llama_model.llama_model_free),
            _vocab = nil,
            path = "<file_ptr>"
        },
        Model
    )
end

--- Auto-fit a model's `n_gpu_layers` and `n_ctx` to the available VRAM.
---
--- Wraps the bridge's `ion7_params_fit`, which itself wraps libcommon's
--- `common_fit_params`. NOT thread-safe — call before any other model
--- load on the same process.
---
--- @param  path string Absolute path to the `.gguf` file.
--- @param  opts table? Optional :
---           `n_ctx`     (number, default 4096) — desired context size,
---           `n_ctx_min` (number, default 512)  — abort if cannot meet.
--- @return table|nil  `{ n_gpu_layers, n_ctx }` on success, nil if the
---                    model cannot fit even at minimum settings.
function Model.fit_params(path, opts)
    -- Lazy require so consumers that never touch fit_params don't have to
    -- have the bridge .so on disk.
    local bridge = require "ion7.core.ffi.bridge"
    opts = opts or {}

    local n_gpu = ffi_new("int32_t[1]", 0)
    local n_ctx = ffi_new("uint32_t[1]", opts.n_ctx or 4096)
    local rc = bridge.ion7_params_fit(path, n_gpu, n_ctx, opts.n_ctx_min or 512)
    if rc ~= 0 then
        return nil
    end
    return {
        n_gpu_layers = tonumber(n_gpu[0]),
        n_ctx = tonumber(n_ctx[0])
    }
end

-- ── Lifecycle ─────────────────────────────────────────────────────────────

--- Return the raw `llama_model*` cdata pointer.
--- @return cdata
function Model:ptr()
    return self._ptr
end

--- Explicitly free the model and release VRAM immediately. After this
--- the Model object is dead — calling any other method on it is
--- undefined. The `ffi.gc` finalizer is disarmed first to avoid a
--- double-free if the GC runs later.
function Model:free()
    if self._ptr then
        ffi_gc(self._ptr, nil)
        llama_model.llama_model_free(self._ptr)
        self._ptr = nil
        self._vocab = nil
    end
end

--- Save the (possibly modified) model back to a GGUF file. Useful after
--- applying quantization or merging LoRA adapters.
--- @param  path string Output GGUF file path.
function Model:save(path)
    llama_model.llama_model_save_to_file(self._ptr, path)
end

-- ── Mixin sub-modules ─────────────────────────────────────────────────────
--
-- Each sub-module returns a flat table of methods that take the Model
-- instance as their first argument. We splice them into the metatable
-- below so the public require path remains `ion7.core.model` — exactly
-- like the historical layout.
--
-- The sub-modules are loaded at Model.lua load time, but the Vocab /
-- Context modules they may reference internally are lazy-required from
-- inside the methods to avoid a circular dependency at module init.

for k, v in pairs(require "ion7.core.model.inspect") do
    Model[k] = v
end
for k, v in pairs(require "ion7.core.model.meta") do
    Model[k] = v
end
for k, v in pairs(require "ion7.core.model.lora") do
    Model[k] = v
end
for k, v in pairs(require "ion7.core.model.quantize") do
    Model[k] = v
end
for k, v in pairs(require "ion7.core.model.context_factory") do
    Model[k] = v
end

return Model
