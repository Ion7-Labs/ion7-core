#!/usr/bin/env luajit
--- 06_kv_reuse.lua - KV cache reuse patterns for efficiency.
---
--- Demonstrates three KV cache techniques:
---   1. Snapshot/restore  - save and restore conversation state
---   2. Sequence forking  - branch a conversation into parallel paths
---   3. Sliding window    - keep context under control for long sessions
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/06_kv_reuse.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)
local model  = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab  = model:vocab()
local ctx    = model:context({ n_ctx = 2048, n_seq_max = 4 })
local sampler = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)

local function prefill(text, seq_id)
    local msgs    = { { role = "user", content = text } }
    local fmt     = vocab:apply_template(msgs, true)
    local toks, n = vocab:tokenize(fmt, false, true)
    ctx:decode(toks, n, seq_id or 0, 0)
    return n
end

local function gen(max, seq_id)
    sampler:reset()
    local parts = {}
    for _ = 1, max do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        parts[#parts+1] = vocab:piece(tok, true)
        ctx:decode_single(tok, seq_id or 0)
    end
    return table.concat(parts):match("^%s*(.-)%s*$")
end

-- ── 1. Snapshot / restore ────────────────────────────────────────────────────
-- Useful for: saving a conversation checkpoint, branch-and-compare,
-- retrying with a different sampler.

io.write("\n[1. Snapshot / restore]\n")

ctx:kv_clear()
prefill("The capital of France is", 0)
local response1 = gen(20, 0)
io.write("  First generation:    " .. response1 .. "\n")

-- Save state after the question is processed
local t0 = ion7.time_us()
local blob = ctx:snapshot()
local snap_ms = (ion7.time_us() - t0) / 1000

io.write(string.format("  Snapshot: %.1f KB in %.2f ms\n", #blob / 1024, snap_ms))

-- Restore and re-generate - should produce the same output (same KV = same context)
ctx:restore(blob)
local response2 = gen(20, 0)
io.write("  After restore:       " .. response2 .. "\n")
io.write("  Match: " .. tostring(response1 == response2) .. "\n")

-- ── 2. Sequence forking ───────────────────────────────────────────────────────
-- Useful for: beam search, parallel exploration of conversation branches,
-- A/B testing different continuations from the same context.

io.write("\n[2. Sequence forking]\n")

ctx:kv_clear()
prefill("List three programming languages:", 0)

-- Fork sequence 0 into sequences 1 and 2
ctx:kv_seq_cp(0, 1, 0, -1)
ctx:kv_seq_cp(0, 2, 0, -1)

io.write("  [forked seq 0 → seq 1, 2]\n")

-- Generate from each branch independently with different seeds
local s1 = ion7.Sampler.chain():top_k(40):temperature(0.9):dist(111):build(vocab)
local s2 = ion7.Sampler.chain():top_k(40):temperature(0.9):dist(999):build(vocab)

local function gen_with(samp, seq, max)
    samp:reset()
    local parts = {}
    for _ = 1, max do
        local tok = samp:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        parts[#parts+1] = vocab:piece(tok, true)
        ctx:decode_single(tok, seq)
    end
    return table.concat(parts):match("^%s*(.-)%s*$")
end

io.write("  Branch A (seed 111): " .. gen_with(s1, 1, 60):gsub("\n", " ") .. "\n")
io.write("  Branch B (seed 999): " .. gen_with(s2, 2, 60):gsub("\n", " ") .. "\n")
s1:free(); s2:free()

-- ── 3. Sliding window ────────────────────────────────────────────────────────
-- When n_past approaches n_ctx, shift old tokens out of the cache.
-- Only works on models where kv_can_shift() = true.

io.write("\n[3. Sliding window]\n")

if not ctx:kv_can_shift() then
    io.write("  [SKIP] kv_can_shift=false on this model (Qwen3 uses chunked attention)\n")
    io.write("  [INFO] For sliding window, use models with standard RoPE (e.g. Llama, Mistral)\n")
else
    local WINDOW = 256  -- tokens to keep
    local n_gen  = 0
    ctx:kv_clear()
    prefill("Write a very long story: Once upon a time", 0)

    io.write("  Generating with sliding window (max 500 tokens)...\n")
    for _ = 1, 500 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        n_gen = n_gen + 1

        -- Slide: remove the oldest tokens when window is full
        if ctx:n_past() >= WINDOW then
            local shift = math.floor(WINDOW / 4)   -- discard 25% oldest
            ctx:kv_seq_shift(0, -shift, 0, -1)
        end
    end
    io.write(string.format("  Generated %d tokens within %d-token window\n", n_gen, WINDOW))
end

ion7.shutdown()
