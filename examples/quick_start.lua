#!/usr/bin/env luajit
--- Quick start example - ion7-core
---
--- This file shows the MINIMUM viable usage of ion7-core.
--- For a real application, use ion7-llm which provides a proper
--- Generator, streaming, stop strings, and chat pipeline.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/quick_start.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL_PATH = os.getenv("ION7_MODEL")
if not MODEL_PATH then
    print("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/quick_start.lua")
    os.exit(1)
end

local ion7 = require "ion7.core"

-- ── Step 1: Initialize backend ────────────────────────────────────────────────
ion7.init({ log_level = 0 })

-- ── Step 2: Load model (auto-fit VRAM) ───────────────────────────────────────
local fit   = ion7.Model.fit_params(MODEL_PATH)
local model = ion7.Model.load(MODEL_PATH, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})

-- ── Step 3: Create primitives ─────────────────────────────────────────────────
local ctx   = model:context({ n_ctx = math.min(fit and fit.n_ctx or 2048, 4096) })
local vocab = model:vocab()

-- ── Step 4: Build sampler ─────────────────────────────────────────────────────
local sampler = ion7.Sampler.chain()
    :top_k(50)
    :top_p(0.9, 1)
    :temp(0.8)
    :dist(0xFFFFFFFF)
    :build(vocab)

-- ── Step 5: Tokenize prompt ───────────────────────────────────────────────────
local messages = { { role = "user", content = "Say hello in one sentence." } }
local prompt   = vocab:apply_template(messages, true)
local tokens, n = vocab:tokenize(prompt, false, true)

print(string.format("[ion7-core] prompt: %d tokens", n))

-- ── Step 6: Decode prompt (prefill) ──────────────────────────────────────────
ctx:kv_clear()
ctx:decode(tokens, n, 0, 0)

-- ── Step 7: Generate tokens ───────────────────────────────────────────────────
io.write("[ion7-core] output: ")
for _ = 1, 100 do
    local token = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(token) then break end
    io.write(vocab:piece(token, true))
    io.flush()
    ctx:decode_single(token, 0)
end
io.write("\n")

-- ── Step 8: Stats ─────────────────────────────────────────────────────────────
local p = ctx:perf()
print(string.format("[ion7-core] %.1f tok/s  (n_eval=%d  n_reused=%d)",
    p.tokens_per_s, p.n_eval, p.n_reused))

ion7.shutdown()
