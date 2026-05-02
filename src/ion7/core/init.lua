--- @module ion7.core
--- @author  ion7 / Ion7 Project Contributors
---
--- Ion7 Core — silicon-level LuaJIT runtime for llama.cpp.
---
--- The foundation layer of the Ion7 ecosystem. Provides a Lua-ergonomic
--- OOP wrapper over llama.cpp's stable C API, plus a small libcommon
--- bridge for the C++ helpers that cannot be FFI'd directly (chat
--- templates, common_sampler, common_speculative, JSON-schema-to-GBNF,
--- VRAM auto-fit).
---
--- What ion7-core IS :
---   - Model loading + introspection + metadata + LoRA + quantize.
---   - Context management : decode / encode, KV cache, state I/O.
---   - Vocabulary : tokenize / detokenize / Jinja2 templates.
---   - Sampler chains : 20+ samplers with a fluent builder + custom
---     Lua-implemented samplers via `ffi.cast` trampolines.
---   - Threadpool, speculative decoding, perf counters.
---   - Pure-Lua utilities : UTF-8 streaming, base64 codec, log routing,
---     tensor inspection.
---
--- What ion7-core is NOT :
---   Chat pipelines, stop-string detection, streaming protocols, RAG,
---   embedding stores, grammar compilers. Those live in dedicated
---   downstream modules (`ion7-llm`, `ion7-embed`, `ion7-grammar`,
---   `ion7-engram`).
---
---   local ion7 = require "ion7.core"
---   ion7.init({ log_level = 1 })
---
---   local model   = ion7.Model.load("qwen3-32b.gguf", { n_gpu_layers = 25 })
---   local ctx     = model:context({ n_ctx = 65536, kv_type = "q8_0" })
---   local vocab   = model:vocab()
---   local sampler = ion7.Sampler.default()
---
---   ctx:warmup()                                 -- pre-JIT GPU shaders
---
---   local toks, n = vocab:tokenize("Hello, world!", true, false)
---   ctx:decode(toks, n)
---   for _ = 1, 20 do
---     local tok = sampler:sample(ctx:ptr(), 0)
---     io.write(vocab:piece(tok))
---     ctx:decode_single(tok)
---   end
---
---   ion7.shutdown()
---
--- Lazy module loading :
---   The class modules (`Model`, `Context`, ...) are loaded on FIRST
---   ACCESS via a `__index` metatable hook ; importing `ion7.core`
---   does not pay the cost of pulling in every sub-module up front.
---   The utility modules under `util/` are similarly lazy.

local ion7 = {}

-- ── Class registry (lazy) ─────────────────────────────────────────────────
--
-- Each entry is `field_name → require_path`. The `__index` hook below
-- requires the module on first access and caches the result via
-- `rawset`, so subsequent reads are direct table lookups.

local _CLASSES = {
    Model = "ion7.core.model",
    Context = "ion7.core.context",
    Vocab = "ion7.core.vocab",
    Sampler = "ion7.core.sampler",
    CustomSampler = "ion7.core.custom_sampler",
    Threadpool = "ion7.core.threadpool",
    Speculative = "ion7.core.speculative"
}

-- ── Utility re-exports (lazy) ─────────────────────────────────────────────
--
-- The four `util/*` helpers are exposed at the top level as well so
-- callers can do `ion7.utf8.is_complete(...)` without remembering the
-- sub-path. Same lazy-load mechanism as the classes above.

local _UTILS = {
    utf8 = "ion7.core.util.utf8",
    base64 = "ion7.core.util.base64",
    log = "ion7.core.util.log",
    tensor = "ion7.core.util.tensor"
}

setmetatable(
    ion7,
    {
        __index = function(t, k)
            local p = _CLASSES[k] or _UTILS[k]
            if not p then
                return nil
            end
            local mod = require(p)
            rawset(t, k, mod)
            return mod
        end
    }
)

-- ── Eagerly-required FFI surfaces used by the lifecycle helpers ──────────
--
-- These modules are tiny (a few cdef + ffi.load lines each) and almost
-- every consumer ends up touching them via the lifecycle functions
-- below — pre-loading saves a metatable lookup on first use without
-- meaningful cost.

local llama_backend = require "ion7.core.ffi.llama.backend" -- backend_init/free, supports_*
local ffi = require "ffi"
local ffi_string = ffi.string

-- ── Lifecycle ────────────────────────────────────────────────────────────

--- Initialise the llama.cpp + ggml backend and configure logging.
--- Must be called once at process startup, BEFORE loading any model.
---
--- @param opts table?
---   `log_level` (integer, default 0) — 0=silent, 1=error, 2=warn,
---                3=info, 4=debug. See `ion7.core.util.log`.
---   `log_file`  (string, optional)   — redirect output to a file path
---                instead of stderr.
---   `log_timestamps` (bool, optional) — prepend ISO-8601 timestamps.
function ion7.init(opts)
    opts = opts or {}

    -- Wire up our log dispatcher BEFORE backend_init so the backend's
    -- own boot messages route through our level filter.
    local log = ion7.log -- triggers lazy require
    if opts.log_level then
        log.set_level(opts.log_level)
    end
    if opts.log_file then
        log.to_file(opts.log_file)
    end
    if opts.log_timestamps ~= nil then
        log.set_timestamps(opts.log_timestamps)
    end
    log.install()

    llama_backend.llama_backend_init()
end

--- Release every llama.cpp / ggml resource. Safe to call multiple
--- times (the second call is a no-op). Pair with `init` at process
--- exit to release VRAM cleanly.
function ion7.shutdown()
    -- Wrap in pcall : on a Ctrl-C path the FFI lib may already be
    -- partially torn down ; we still want shutdown() to be best-effort.
    pcall(llama_backend.llama_backend_free)
    pcall(
        function()
            ion7.log.uninstall()
        end
    )
end

-- ── Runtime capabilities ─────────────────────────────────────────────────

--- Snapshot of the build-time and runtime capabilities of the linked
--- libllama / bridge. Downstream modules (`ion7-llm`, `ion7-embed`)
--- consult this to adapt behaviour (e.g. enable mmap when supported).
---
--- @return table {
---   mmap, mlock, gpu_offload, rpc,
---   max_devices, max_parallel_seqs,
---   bridge_ver, llama_info
--- }
function ion7.capabilities()
    -- Lazy-require the bridge so a Vocab-only consumer (no .so) can
    -- still use the rest of the API surface.
    local bridge_ok, bridge = pcall(require, "ion7.core.ffi.bridge")

    return {
        mmap = llama_backend.llama_supports_mmap() == true,
        mlock = llama_backend.llama_supports_mlock() == true,
        gpu_offload = llama_backend.llama_supports_gpu_offload() == true,
        rpc = llama_backend.llama_supports_rpc() == true,
        max_devices = tonumber(llama_backend.llama_max_devices()),
        max_parallel_seqs = tonumber(llama_backend.llama_max_parallel_sequences()),
        bridge_ver = bridge_ok and ffi_string(bridge.ion7_bridge_version()) or nil,
        llama_info = ffi_string(llama_backend.llama_print_system_info())
    }
end

--- Microsecond timestamp from llama.cpp's internal monotonic clock.
--- Useful for precise inner-loop timing without `os.clock`'s
--- platform-dependent resolution.
--- @return integer
function ion7.time_us()
    return tonumber(llama_backend.llama_time_us())
end

-- ── NUMA ─────────────────────────────────────────────────────────────────

--- NUMA strategy constants — match the upstream `ggml_numa_strategy`
--- enum values. Pass to `ion7.numa_init`.
ion7.NUMA_DISABLED = 0
ion7.NUMA_DISTRIBUTE = 1
ion7.NUMA_ISOLATE = 2
ion7.NUMA_NUMACTL = 3
ion7.NUMA_MIRROR = 4

--- Configure NUMA placement. Call BEFORE `ion7.init` for the policy to
--- apply to llama.cpp's own thread spawning.
--- @param  strategy integer? Default `NUMA_DISTRIBUTE`.
function ion7.numa_init(strategy)
    llama_backend.llama_numa_init(strategy or ion7.NUMA_DISTRIBUTE)
end

return ion7