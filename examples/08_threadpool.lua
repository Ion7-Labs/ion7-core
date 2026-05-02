#!/usr/bin/env luajit
--- @example examples.08_threadpool
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 08 — Threadpool : shared CPU workers across multiple contexts
--- ════════════════════════════════════════════════════════════════════════
---
--- A `Threadpool` is a pool of OS-level worker threads that one or
--- more `Context` instances can borrow at decode time. Without a
--- shared pool, llama.cpp spins up a fresh internal pool per context
--- — wasteful when several contexts run side by side.
---
--- Important sizing rule : the pool size MUST equal both `n_threads`
--- AND `n_threads_batch` of every attached context. A mismatched
--- size segfaults inside the CPU backend's worker dispatch — there
--- is no graceful failure for it.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/08_threadpool.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit   = ion7.Model.fit_params(MODEL)
local model = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local vocab = model:vocab()

local POOL_SIZE = 4
local sampler   = ion7.Sampler.chain():top_k(1):dist(42):build()

--- Quick generation helper. Decodes the prompt then samples up to
--- `max_gen` tokens, returning the trimmed reply.
local function quick(ctx, user_text, max_gen)
    local prompt   = vocab:apply_template(
        { { role = "user", content = user_text } }, true, -1)
    local toks, n  = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(toks, n)
    sampler:reset()
    local pieces = {}
    for _ = 1, max_gen do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        pieces[#pieces + 1] = vocab:piece(tok)
        ctx:decode_single(tok)
    end
    return (table.concat(pieces):gsub("^%s+", ""):gsub("%s+$", ""))
end

-- ════════════════════════════════════════════════════════════════════════
-- 1. Two contexts attached to one shared pool
-- ════════════════════════════════════════════════════════════════════════

print("\n[1] Two contexts sharing one threadpool of " .. POOL_SIZE .. " workers")

local tp = ion7.Threadpool.new(POOL_SIZE)
print("  pool created : " .. tp:n_threads() .. " worker threads")

local ctx_a = model:context({
    n_ctx = 1024, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE,
})
local ctx_b = model:context({
    n_ctx = 1024, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE,
})
ctx_a:attach_threadpool(tp)
ctx_b:attach_threadpool(tp)

print("  ctx A : " .. quick(ctx_a, "What is 2 + 2 ?", 24):gsub("\n", " "))
print("  ctx B : " .. quick(ctx_b, "What is 3 + 3 ?", 24):gsub("\n", " "))

ctx_a:detach_threadpool() ; ctx_b:detach_threadpool()
ctx_a:free() ; ctx_b:free()

-- ════════════════════════════════════════════════════════════════════════
-- 2. Pause / resume for cooperative CPU sharing
-- ════════════════════════════════════════════════════════════════════════
--
-- Pausing the pool puts the workers to sleep so an unrelated CPU-
-- bound task on the same machine gets the cores for itself. Resuming
-- wakes them back up. A bit niche, but useful for dropping inference
-- priority while a heavy CPU job runs nearby.

print("\n[2] Pause / resume the pool")

local ctx = model:context({
    n_ctx = 1024, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE,
})
ctx:attach_threadpool(tp)

print("  reply 1 : " .. quick(ctx, "Say hi.", 10):gsub("\n", " "))

tp:pause()  ; print("  pool paused")
-- ... another CPU-heavy task could run here without thread contention ...
tp:resume() ; print("  pool resumed")

print("  reply 2 : " .. quick(ctx, "Say bye.", 10):gsub("\n", " "))

ctx:detach_threadpool() ; ctx:free()

-- ════════════════════════════════════════════════════════════════════════
-- 3. Prefill latency : shared pool vs per-context default
-- ════════════════════════════════════════════════════════════════════════
--
-- llama.cpp creates its own internal pool when none is attached. The
-- difference per single decode is small ; the win shows up when you
-- run many contexts back to back (each default pool is its own
-- pthread_create cascade).

print("\n[3] Prefill timing — shared pool vs default pool")

local prompt   = vocab:apply_template(
    { { role = "user", content = "List five facts about LuaJIT." } },
    true, -1)
local toks, n  = vocab:tokenize(prompt, false, true)

local function average_prefill_ms(ctx_inst, n_runs)
    local t0 = ion7.time_us()
    for _ = 1, n_runs do
        ctx_inst:kv_clear()
        ctx_inst:decode(toks, n)
    end
    return (ion7.time_us() - t0) / 1000 / n_runs
end

local N_RUNS = 5

local ctx_shared = model:context({
    n_ctx = 1024, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE,
})
ctx_shared:attach_threadpool(tp)
local ms_shared = average_prefill_ms(ctx_shared, N_RUNS)
ctx_shared:detach_threadpool() ; ctx_shared:free()

local ctx_default = model:context({
    n_ctx = 1024, n_threads = POOL_SIZE, n_threads_batch = POOL_SIZE,
})
local ms_default = average_prefill_ms(ctx_default, N_RUNS)
ctx_default:free()

print(string.format("  shared pool   : %.1f ms/prefill (avg of %d)",
    ms_shared, N_RUNS))
print(string.format("  default pool  : %.1f ms/prefill (avg of %d)",
    ms_default, N_RUNS))

sampler:free() ; tp:free() ; model:free()
ion7.shutdown()
