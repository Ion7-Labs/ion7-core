#!/usr/bin/env luajit
--- 03_streaming.lua - Token-by-token streaming with a callback.
---
--- Shows how to wire a per-token callback for streaming output,
--- performance tracking, and early stopping.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/03_streaming.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)

local model   = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab   = model:vocab()
local ctx     = model:context({ n_ctx = 2048 })
local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95):temperature(0.8):dist(42)
    :build(vocab)

--- Stream tokens from a prompt, calling on_token for each generated token.
---
--- @param prompt    string   Raw user message (will be formatted with chat template)
--- @param max_gen   number   Max tokens to generate
--- @param on_token  function Called with (piece: string, token: number, n: number)
---                           Return false to stop generation early
--- @return          string   The full generated text
local function stream(prompt, max_gen, on_token)
    local msgs   = { { role = "user", content = prompt } }
    local fmt    = vocab:apply_template(msgs, true)
    local toks, n = vocab:tokenize(fmt, false, true)

    ctx:kv_clear()
    ctx:decode(toks, n, 0, 0)
    sampler:reset()
    ctx:perf_reset()

    local parts = {}
    local count = 0
    for _ = 1, max_gen do
        local tok   = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        local piece = vocab:piece(tok, true)
        parts[#parts+1] = piece
        count = count + 1
        if on_token(piece, tok, count) == false then break end
        ctx:decode_single(tok, 0)
    end

    return table.concat(parts)
end

-- ── Example 1: plain streaming ────────────────────────────────────────────────

io.write("\n[Example 1 - plain streaming]\n")
io.write("Response: ")

local t0 = ion7.time_us()
stream("Count from 1 to 20, numbers only, one per line.", 200,
    function(piece)
        io.write(piece); io.flush()
        return true
    end
)

local elapsed_ms = (ion7.time_us() - t0) / 1000
local p = ctx:perf()
io.write(string.format(
    "\n\n[%.1f tok/s  |  prefill %d tok / %.0f ms  |  gen %d tok / %.0f ms]\n",
    p.tokens_per_s, p.n_p_eval, p.t_p_eval_ms, p.n_eval, p.t_eval_ms
))

-- ── Example 2: stop on keyword ────────────────────────────────────────────────

io.write("\n[Example 2 - stop on '10']\n")
io.write("Response: ")

local buffer = ""
stream("Count from 1 to 20.", 200, function(piece)
    buffer = buffer .. piece
    io.write(piece); io.flush()
    -- Stop as soon as we see "10" in the accumulated output
    if buffer:find("10") then return false end
    return true
end)
io.write("\n[stopped early]\n")

-- ── Example 3: collect and post-process ──────────────────────────────────────

io.write("\n[Example 3 - word count after generation]\n")
local full = stream("Describe LuaJIT FFI in three sentences.", 256, function() return true end)
local words = 0
for _ in full:gmatch("%S+") do words = words + 1 end
io.write(string.format("Response (%d words): %s\n", words, full))

ion7.shutdown()
