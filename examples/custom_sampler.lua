#!/usr/bin/env luajit
--- Custom sampler example - write sampling logic in Lua.
---
--- Demonstrates how to implement a sampler entirely in Lua
--- and plug it into the llama.cpp chain via ion7-core.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/custom_sampler.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL_PATH = os.getenv("ION7_MODEL")
if not MODEL_PATH then
    print("Usage: ION7_MODEL=/path/to/model.gguf luajit examples/custom_sampler.lua")
    os.exit(1)
end

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit   = ion7.Model.fit_params(MODEL_PATH)
local model = ion7.Model.load(MODEL_PATH, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local ctx   = model:context({ n_ctx = 2048 })
local vocab = model:vocab()

-- ── Lua-defined sampler: "Top-K with logging" ─────────────────────────────────
-- This sampler logs the top-3 candidates at each step, then selects the best.

local step = 0
local top_k_logged = ion7.CustomSampler.new("top_k_logger", {
    apply = function(candidates, n)
        step = step + 1
        -- Print top 3 logits at step 1
        if step == 1 then
            io.write("  [step 1 top-3 candidates]\n")
            local tops = {}
            for i = 0, math.min(n, 200) - 1 do
                tops[#tops + 1] = { i = i, logit = candidates[i].logit }
            end
            table.sort(tops, function(a, b) return a.logit > b.logit end)
            for k = 1, math.min(3, #tops) do
                io.write(string.format("    [%d] logit=%.4f\n", tops[k].i, tops[k].logit))
            end
        end
        -- Select highest logit (greedy)
        local best_i, best_logit = 0, -math.huge
        for i = 0, n - 1 do
            if candidates[i].logit > best_logit then
                best_logit = candidates[i].logit
                best_i = i
            end
        end
        candidates.selected = best_i
        return best_i
    end,
})

-- Plug our Lua sampler into a chain
local sampler = ion7.Sampler.chain()
    :custom(top_k_logged)  -- our Lua sampler runs first
    :build(vocab)

-- Run a short generation
local messages = { { role = "user", content = "What is 2+2?" } }
local prompt   = vocab:apply_template(messages, true)
local tokens, n = vocab:tokenize(prompt, false, true)

ctx:kv_clear()
ctx:decode(tokens, n, 0, 0)

io.write("[custom sampler] output: ")
for _ = 1, 30 do
    local token = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(token) then break end
    io.write(vocab:piece(token, true))
    io.flush()
    sampler:accept(token)
    ctx:decode_single(token, 0)
end
io.write("\n")

ion7.shutdown()
