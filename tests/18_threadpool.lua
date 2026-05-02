#!/usr/bin/env luajit
--- @module tests.18_threadpool
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 18 — `Threadpool` : create, attach, pause/resume, free
--- ════════════════════════════════════════════════════════════════════════
---
--- A `Threadpool` is a shared bag of CPU worker threads usable by one
--- or more `Context` instances. Sharing pools across contexts amortises
--- the OS thread-creation cost (notable on Windows) and keeps overall
--- CPU utilisation predictable when several contexts run side by side.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/18_threadpool.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Construction
-- ════════════════════════════════════════════════════════════════════════

T.suite("Construction")

T.test("Threadpool.new returns a handle with the requested size", function()
    local tp = ion7.Threadpool.new(2)
    T.is_type(tp,         "table")
    T.is_type(tp:ptr(),   "cdata")
    T.eq(tp:n_threads(),  2)
    tp:free()
end)

T.test("Threadpool.new rejects non-positive sizes", function()
    T.err(function() ion7.Threadpool.new(0)  end, "n_threads")
    T.err(function() ion7.Threadpool.new(-1) end, "n_threads")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Pause / resume
-- ════════════════════════════════════════════════════════════════════════

T.suite("Pause / resume")

T.test("pause and resume do not raise", function()
    local tp = ion7.Threadpool.new(2)
    T.no_error(function() tp:pause()  end)
    T.no_error(function() tp:resume() end)
    tp:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Attach / detach to a Context
-- ════════════════════════════════════════════════════════════════════════
--
-- The full attach / decode / detach cycle. We attach a 2-thread pool,
-- run a small decode through it, then detach and confirm subsequent
-- decodes still work on llama.cpp's own internal pool.

T.suite("Attach / detach round-trip")

-- llama.cpp's CPU backend dispatches as many workers per ubatch as
-- `n_threads_batch` requests ; passing more workers than the pool
-- holds segfaults inside the backend's worker dispatch. We size both
-- parameters identically below so the pool capacity matches what the
-- context will actually ask for.
local POOL_SIZE = 2

T.test("attached pool participates in a real decode", function()
    local tp  = ion7.Threadpool.new(POOL_SIZE)
    local ctx = model:context({
        n_ctx           = 256,
        n_threads       = POOL_SIZE,
        n_threads_batch = POOL_SIZE,
    })

    ctx:attach_threadpool(tp)
    local toks, n = vocab:tokenize("attached", true, true)
    T.no_error(function() ctx:decode(toks, n) end)

    ctx:detach_threadpool()
    -- A second decode after detach uses llama.cpp's internal pool ;
    -- the call must still succeed.
    ctx:kv_clear()
    T.no_error(function() ctx:decode(toks, n) end)

    ctx:free()
    tp:free()
end)

T.test("a single pool can be attached to two contexts simultaneously",
    function()
        local tp = ion7.Threadpool.new(POOL_SIZE)
        local a  = model:context({
            n_ctx = 256, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE })
        local b  = model:context({
            n_ctx = 256, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE })
        a:attach_threadpool(tp)
        b:attach_threadpool(tp)

        local toks, n = vocab:tokenize("share", true, true)
        T.no_error(function() a:decode(toks, n) end)
        T.no_error(function() b:decode(toks, n) end)

        a:free() b:free() tp:free()
    end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — Lifecycle
-- ════════════════════════════════════════════════════════════════════════

T.suite("Lifecycle — explicit free is idempotent")

T.test("free() then free() is safe", function()
    local tp = ion7.Threadpool.new(2)
    T.no_error(function() tp:free() end)
    T.no_error(function() tp:free() end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
