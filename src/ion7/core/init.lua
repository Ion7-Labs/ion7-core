--- @module ion7.core
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- Ion7 Core - Silicon-level LuaJIT runtime for llama.cpp.
---
--- This is the foundation layer of the Ion7 ecosystem.
--- It provides direct, zero-overhead access to every llama.cpp primitive.
---
--- What ion7-core IS:
---   Model loading, context management, tokenization, KV cache ops,
---   state persistence, sampler chain construction, custom Lua samplers,
---   threadpool management, performance monitoring.
---
--- What ion7-core is NOT:
---   Chat pipelines, stop-string handling, streaming, RAG, embeddings,
---   grammar compilers. Those live in dedicated modules (ion7-llm, etc.)
---
--- Usage:
---   local ion7    = require "ion7.core"
---   ion7.init({ log_level = 0 })
---
---   local model   = ion7.Model.load("/path/to/model.gguf")
---   local ctx     = model:context()
---   local vocab   = model:vocab()
---   local sampler = ion7.Sampler.chain():top_k(50):temp(0.8):dist():build(vocab)
---
---   local tokens, n = vocab:tokenize("Hello", true, false)
---   ctx:decode(tokens, n, 0, 0)
---   local token = sampler:sample(ctx:ptr(), 0)
---   print(vocab:piece(token))
---
---   ion7.shutdown()
---
--- See examples/ for runnable demos.

local ion7 = {}

-- ── Lazy module registry ──────────────────────────────────────────────────────
-- Modules load on first access to keep startup time minimal.

local _loaded = {}
local _modules = {
    Model         = "ion7.core.model",
    Context       = "ion7.core.context",
    Vocab         = "ion7.core.vocab",
    Sampler       = "ion7.core.sampler",
    CustomSampler = "ion7.core.custom_sampler",
    Threadpool    = "ion7.core.threadpool",
}

setmetatable(ion7, {
    __index = function(t, k)
        if _modules[k] then
            if not _loaded[k] then
                _loaded[k] = require(_modules[k])
            end
            rawset(t, k, _loaded[k])
            return _loaded[k]
        end
    end
})

local Loader = require "ion7.core.ffi.loader"

-- ── Lifecycle ─────────────────────────────────────────────────────────────────

--- Initialize the llama.cpp backend.
--- Must be called once before any model is loaded.
---
--- @param opts table?
---   opts.log_level   number?  0=silent(default) 1=error 2=warn 3=info 4=debug
---   opts.llama_path  string?  Directory containing libllama.so
---   opts.bridge_path string?  Directory containing ion7_bridge.so
function ion7.init(opts)
    opts = opts or {}
    Loader.init({
        log_level   = opts.log_level   or 0,
        llama_path  = opts.llama_path,
        bridge_path = opts.bridge_path,
    })
    local L = Loader.instance()
    L.bridge.ion7_backend_init()
    L.bridge.ion7_set_log_level(opts.log_level or 0)
end

--- Release all llama.cpp resources. Call once at process exit.
--- Safe to call multiple times.
function ion7.shutdown()
    if Loader._instance then
        pcall(function() Loader.instance().bridge.ion7_backend_free() end)
    end
end

-- ── Runtime info ──────────────────────────────────────────────────────────────

--- Return a table of build-time and runtime capabilities.
--- Downstream modules (ion7-llm, ion7-embed) use this to adapt behavior.
---
--- @return table
---   { mmap, mlock, gpu_offload, rpc, max_devices, max_parallel_seqs,
---     bridge_ver, llama_info }
function ion7.capabilities()
    local L  = Loader.instance()
    local br = L.bridge
    return {
        mmap              = br.ion7_supports_mmap()           == 1,
        mlock             = br.ion7_supports_mlock()          == 1,
        gpu_offload       = br.ion7_supports_gpu_offload()    == 1,
        rpc               = br.ion7_supports_rpc()            == 1,
        max_devices       = tonumber(br.ion7_max_devices()),
        max_parallel_seqs = tonumber(br.ion7_max_parallel_sequences()),
        bridge_ver        = L.ffi.string(br.ion7_bridge_version()),
        llama_info        = L.ffi.string(br.ion7_llama_info()),
    }
end

--- Microsecond timestamp from llama.cpp internal clock.
--- Useful for precise timing inside generation loops.
--- @return number
function ion7.time_us()
    return tonumber(Loader.instance().lib.llama_time_us())
end

-- ── NUMA ──────────────────────────────────────────────────────────────────────

--- Strategy constants for ion7.numa_init() (mirrors ggml_numa_strategy).
ion7.NUMA_DISABLED   = 0  -- No NUMA optimization (default).
ion7.NUMA_DISTRIBUTE = 1  -- Distribute threads across NUMA nodes.
ion7.NUMA_ISOLATE    = 2  -- Isolate to a specific node.
ion7.NUMA_NUMACTL    = 3  -- Use numactl settings.
ion7.NUMA_MIRROR     = 4  -- Mirror across nodes.

--- Initialize NUMA optimization for multi-socket servers.
--- Call BEFORE ion7.init() for best effect.
--- @param strategy number?  Default: NUMA_DISTRIBUTE.
function ion7.numa_init(strategy)
    Loader.instance().lib.llama_numa_init(strategy or ion7.NUMA_DISTRIBUTE)
end

--- Reset the library state (backend freed, singleton cleared).
--- Allows calling ion7.init() again in the same process.
--- Primarily useful for test isolation.
function ion7.reset()
    require("ion7.core.ffi.loader").reset()
end

return ion7
