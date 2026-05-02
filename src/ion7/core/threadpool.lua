--- @module ion7.core.threadpool
--- @author  ion7 / Ion7 Project Contributors
---
--- CPU threadpool for `llama_context` instances.
---
--- llama.cpp creates one internal threadpool per context by default. A
--- shared `Threadpool` lets multiple contexts reuse the same worker
--- threads, which amortises the OS thread-creation cost (significant on
--- Windows) and keeps CPU utilisation predictable when several contexts
--- run side by side.
---
---   local tp = ion7.Threadpool.new(16)   -- 16 worker threads
---   ctx1:attach_threadpool(tp)
---   ctx2:attach_threadpool(tp)           -- shared pool
---   -- ... inference work ...
---   tp:free()                            -- or let GC handle it
---
--- Implementation notes :
---   - Pools are created and freed via the CPU backend's registry, not
---     by calling `ggml_threadpool_new` / `_free` directly. The CPU
---     backend looks up its pool functions via
---     `ggml_backend_reg_get_proc_address`, so a directly-created pool
---     is invisible to the backend and crashes on first decode. We
---     mirror the `llama-bench` upstream pattern : look up the proc
---     addresses once at module load, cast them to function pointers,
---     and call them on every `new` / `free`.
---   - `ffi.gc` attaches the resolved free function pointer to the
---     pool cdata so a pool that goes out of scope cleans up on its
---     own. `Threadpool:free()` disarms the finalizer first to avoid
---     a double-free if both happen.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local ggml_misc    = require "ion7.core.ffi.ggml.misc"    -- threadpool_params_init
local ggml_backend = require "ion7.core.ffi.ggml.backend" -- backend registry

local ffi_new  = ffi.new
local ffi_gc   = ffi.gc
local ffi_cast = ffi.cast

-- ── CPU backend proc-address lookup (lazy) ───────────────────────────────
--
-- The lookup needs `llama_backend_init` to have run first so the CPU
-- backend has registered itself, and it touches the FFI shared
-- library. Both make eager lookup at module-load time fragile. We
-- resolve once on first `Threadpool.new` and cache the cast function
-- pointers in module-local upvalues.

local TP_NEW_T  = ffi.typeof(
    "struct ggml_threadpool * (*)(struct ggml_threadpool_params *)")
local TP_FREE_T = ffi.typeof(
    "void (*)(struct ggml_threadpool *)")

-- GGML_BACKEND_DEVICE_TYPE_CPU is the first member of the enum (= 0).
local GGML_BACKEND_DEVICE_TYPE_CPU = 0

local threadpool_new_fn, threadpool_free_fn

local function resolve_cpu_procs()
    if threadpool_new_fn then return end

    local cpu_dev = ggml_backend.ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
    assert(cpu_dev ~= nil,
           "[ion7.core.threadpool] no CPU backend device registered " ..
           "(did you forget ion7.init() ?)")

    local cpu_reg = ggml_backend.ggml_backend_dev_backend_reg(cpu_dev)
    assert(cpu_reg ~= nil, "[ion7.core.threadpool] CPU backend has no registry")

    local new_proc  = ggml_backend.ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new")
    local free_proc = ggml_backend.ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free")
    assert(new_proc  ~= nil, "[ion7.core.threadpool] CPU backend missing ggml_threadpool_new")
    assert(free_proc ~= nil, "[ion7.core.threadpool] CPU backend missing ggml_threadpool_free")

    threadpool_new_fn  = ffi_cast(TP_NEW_T,  new_proc)
    threadpool_free_fn = ffi_cast(TP_FREE_T, free_proc)
end

--- @class ion7.core.Threadpool
--- @field _ptr cdata    `ggml_threadpool_t` opaque handle.
--- @field _n   integer  Worker thread count, cached at creation.
local Threadpool = {}
Threadpool.__index = Threadpool

--- Create a CPU threadpool with `n_threads` worker threads.
---
--- @param  n_threads integer Worker count (must be > 0).
--- @return ion7.core.Threadpool
--- @raise   When `n_threads` is invalid or `ggml_threadpool_new` fails.
function Threadpool.new(n_threads)
    assert(type(n_threads) == "number" and n_threads > 0,
           "[ion7.core.threadpool] n_threads must be > 0")

    -- Resolve the CPU backend's proc addresses on first use. Cached
    -- in module-level upvalues for every subsequent call.
    resolve_cpu_procs()

    -- Populate the params struct with sensible defaults via the
    -- in-place initialiser. `ffi.new` zero-fills first ; `params_init`
    -- writes n_threads, prio, poll, strict_cpu and paused into it.
    local params = ffi_new("struct ggml_threadpool_params")
    ggml_misc.ggml_threadpool_params_init(params, n_threads)

    -- Create through the CPU-registered factory. Using
    -- `ggml_threadpool_new` directly produces a pool the CPU backend
    -- cannot see, which crashes on the first decode after attach.
    local ptr = threadpool_new_fn(params)
    if ptr == nil then
        error(string.format(
            "[ion7.core.threadpool] failed to create pool with %d threads",
            n_threads), 2)
    end

    return setmetatable({
        _ptr = ffi_gc(ptr, threadpool_free_fn),
        _n   = n_threads,
    }, Threadpool)
end

--- Return the raw `ggml_threadpool_t` cdata pointer for use with
--- `Context:attach_threadpool` (or any other ggml API that wants one).
--- @return cdata
function Threadpool:ptr()
    return self._ptr
end

--- Return the cached worker count. Cheap : avoids the FFI roundtrip of
--- `ggml_threadpool_get_n_threads`. The value never changes after
--- construction so caching is exact.
--- @return integer
function Threadpool:n_threads()
    return self._n
end

--- Pause every worker thread. Workers stop accepting new jobs until
--- `resume` is called. No-op when the pool is already freed.
function Threadpool:pause()
    if self._ptr then
        ggml_misc.ggml_threadpool_pause(self._ptr)
    end
end

--- Resume previously paused workers. No-op when the pool is freed.
function Threadpool:resume()
    if self._ptr then
        ggml_misc.ggml_threadpool_resume(self._ptr)
    end
end

--- Manually release the threadpool's native resources. Idempotent — safe
--- to call multiple times. The `ffi.gc` finalizer is disarmed BEFORE the
--- explicit free so the GC does not double-free the same pointer later.
function Threadpool:free()
    if self._ptr then
        ffi_gc(self._ptr, nil)
        -- The free fn is cached as soon as `new` ran, which is a
        -- prerequisite for owning a non-nil _ptr.
        threadpool_free_fn(self._ptr)
        self._ptr = nil
    end
end

return Threadpool
