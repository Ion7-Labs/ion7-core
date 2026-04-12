--- @module ion7.core.threadpool
--- SPDX-License-Identifier: MIT
--- CPU threadpool for llama.cpp contexts.
---
--- By default llama.cpp creates one internal threadpool per context.
--- A custom Threadpool lets multiple contexts share a single pool,
--- reducing thread creation overhead and improving CPU utilization.
---
--- Usage:
---   local tp = llama.Threadpool.new(16)  -- 16 worker threads
---   ctx:attach_threadpool(tp)
---   ctx2:attach_threadpool(tp)           -- shared pool
---   -- ... work ...
---   tp:free()                            -- or let GC handle it

local Loader = require "ion7.core.ffi.loader"

--- @class Threadpool
local Threadpool = {}
Threadpool.__index = Threadpool

--- Create a CPU threadpool.
--- @param  n_threads  number  Worker thread count.
--- @return Threadpool
--- @error  If creation fails.
function Threadpool.new(n_threads)
    assert(type(n_threads) == "number" and n_threads > 0,
        "[ion7.core.threadpool] n_threads must be > 0")
    local L   = Loader.instance()
    local ffi = L.ffi
    local ptr = L.bridge.ion7_threadpool_create(n_threads)
    if ptr == nil then
        error(string.format(
            "[ion7.core.threadpool] failed to create pool with %d threads", n_threads), 2)
    end
    return setmetatable({
        _ptr      = ffi.gc(ptr, L.bridge.ion7_threadpool_free),
        _bridge   = L.bridge,
        _n        = n_threads,
    }, Threadpool)
end

--- Return the raw ggml_threadpool_t pointer.
--- Pass to ctx:attach_threadpool().
--- @return cdata
function Threadpool:ptr() return self._ptr end

--- @return number  Thread count (stored at creation time).
function Threadpool:n_threads()
    return self._n
end

--- Pause all worker threads (they stop accepting new work).
function Threadpool:pause()
    if self._ptr then self._bridge.ion7_threadpool_pause(self._ptr) end
end

--- Resume paused worker threads.
function Threadpool:resume()
    if self._ptr then self._bridge.ion7_threadpool_resume(self._ptr) end
end

--- Manually free the threadpool.
--- Safe to call multiple times (ffi.gc is reset after first call).
function Threadpool:free()
    if self._ptr then
        self._bridge.ion7_threadpool_free(self._ptr)
        -- Disarm the ffi.gc finalizer to prevent double-free when GC runs
        self._ptr = require("ion7.core.ffi.loader").instance().ffi.gc(self._ptr, nil)
        self._ptr = nil
    end
end

return Threadpool
