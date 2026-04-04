#!/usr/bin/env luajit
--- 08_threadpool.lua - Shared CPU threadpool across multiple contexts.
---
--- By default each context creates its own internal threadpool, which wastes
--- threads and memory when running multiple contexts simultaneously.
--- A shared Threadpool amortises thread creation and reduces contention.
---
--- Use case: serving multiple users, batch embedding, multi-sequence decoding.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/08_threadpool.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)
local model  = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab  = model:vocab()

local N_THREADS = 4   -- adjust to your CPU

-- ── 1. Basic shared threadpool ────────────────────────────────────────────────

io.write("\n[1. Shared threadpool across two contexts]\n")

-- One threadpool for prefill (batched), one for generation (sequential)
local tp_prefill = ion7.Threadpool.new(N_THREADS)
local tp_gen     = ion7.Threadpool.new(N_THREADS)

io.write(string.format("  Created: tp_prefill(%d threads)  tp_gen(%d threads)\n",
    tp_prefill:n_threads(), tp_gen:n_threads()))

local ctx1 = model:context({ n_ctx = 512, n_threads = N_THREADS })
local ctx2 = model:context({ n_ctx = 512, n_threads = N_THREADS })

-- Attach the shared pool - both contexts use the same OS threads
ctx1:attach_threadpool(tp_prefill, tp_gen)
ctx2:attach_threadpool(tp_prefill, tp_gen)
io.write("  Both contexts attached to shared pool\n")

-- Simple generation helper
local sampler = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)

local function quick_gen(ctx, prompt, max)
    local msgs    = { { role = "user", content = prompt } }
    local fmt     = vocab:apply_template(msgs, true)
    local toks, n = vocab:tokenize(fmt, false, true)
    ctx:kv_clear()
    ctx:decode(toks, n, 0, 0)
    sampler:reset()
    local parts = {}
    for _ = 1, max do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        parts[#parts+1] = vocab:piece(tok, true)
        ctx:decode_single(tok, 0)
    end
    return table.concat(parts):match("^%s*(.-)%s*$")
end

io.write("  ctx1: " .. quick_gen(ctx1, "What is 2+2?", 20) .. "\n")
io.write("  ctx2: " .. quick_gen(ctx2, "What is 3+3?", 20) .. "\n")

-- Detach before freeing pool
ctx1:detach_threadpool()
ctx2:detach_threadpool()

-- ── 2. Pause / resume for priority control ────────────────────────────────────
-- Useful when you want to dedicate CPU to a high-priority task temporarily.

io.write("\n[2. Pause / resume for priority]\n")

local tp = ion7.Threadpool.new(N_THREADS)
local ctx = model:context({ n_ctx = 256 })
ctx:attach_threadpool(tp)

io.write("  Running with shared pool...\n")
quick_gen(ctx, "Say hello.", 10)

-- Pause: workers go idle, reducing CPU interference with other processes
tp:pause()
io.write("  Pool paused (workers idle)\n")

-- Do some other work here (e.g. heavy embedding computation)

-- Resume when ready for the next inference
tp:resume()
io.write("  Pool resumed\n")

quick_gen(ctx, "Say goodbye.", 10)

ctx:detach_threadpool()
ctx:free()

-- ── 3. Performance comparison ─────────────────────────────────────────────────

io.write("\n[3. Shared pool vs default (per-context) pool]\n")

local N_RUNS = 5
local prompt  = "List 5 facts about LuaJIT:"
local msgs    = { { role = "user", content = prompt } }
local fmt     = vocab:apply_template(msgs, true)
local toks, n = vocab:tokenize(fmt, false, true)

-- With shared pool
local ctx_shared = model:context({ n_ctx = 512 })
local tp_shared  = ion7.Threadpool.new(N_THREADS)
ctx_shared:attach_threadpool(tp_shared)

local t0 = ion7.time_us()
for _ = 1, N_RUNS do
    ctx_shared:kv_clear()
    ctx_shared:decode(toks, n, 0, 0)
end
local shared_ms = (ion7.time_us() - t0) / 1000 / N_RUNS
ctx_shared:detach_threadpool()
ctx_shared:free()
tp_shared:free()

-- Without shared pool (default internal threadpool)
local ctx_default = model:context({ n_ctx = 512 })

t0 = ion7.time_us()
for _ = 1, N_RUNS do
    ctx_default:kv_clear()
    ctx_default:decode(toks, n, 0, 0)
end
local default_ms = (ion7.time_us() - t0) / 1000 / N_RUNS
ctx_default:free()

io.write(string.format("  Shared pool:   %.1f ms/prefill (avg %d runs)\n", shared_ms, N_RUNS))
io.write(string.format("  Default pool:  %.1f ms/prefill (avg %d runs)\n", default_ms, N_RUNS))
io.write(string.format("  Difference:    %+.1f ms\n", shared_ms - default_ms))
io.write("  (difference is typically small for single context; benefit shows at scale)\n")

-- Cleanup
tp_prefill:free(); tp_gen:free()
ctx1:free(); ctx2:free()
sampler:free()

ion7.shutdown()
