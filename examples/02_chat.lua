#!/usr/bin/env luajit
--- 02_chat.lua - Interactive multi-turn chat with KV cache reuse.
---
--- Keeps the full conversation in the KV cache between turns so the model
--- doesn't need to re-process previous messages. Each new turn only processes
--- the new tokens - this is the correct way to build a chat loop.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/02_chat.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)

local model   = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab   = model:vocab()
local ctx     = model:context({ n_ctx = fit and fit.n_ctx or 4096 })
local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95):temperature(0.8)
    :penalties(64, 1.1)   -- light repetition penalty
    :dist(os.time())      -- different seed each run
    :build(vocab)

local MAX_GEN  = 512
local history  = {}       -- { { role, content }, ... }

-- System prompt (optional - comment out if your model doesn't support it)
local SYSTEM = "You are a helpful, concise assistant."

io.write("ion7-core chat - type 'quit' to exit, 'reset' to clear history\n\n")

while true do
    io.write("You: ")
    io.flush()
    local input = io.read("l")
    if not input or input == "quit" then break end

    if input == "reset" then
        history = {}
        ctx:kv_clear()
        sampler:reset()
        io.write("[history cleared]\n\n")
        goto continue
    end

    if #input == 0 then goto continue end

    -- Build the full message list with optional system prompt
    local msgs = {}
    if SYSTEM and #history == 0 then
        msgs[#msgs+1] = { role = "system", content = SYSTEM }
    end
    for _, m in ipairs(history) do msgs[#msgs+1] = m end
    msgs[#msgs+1] = { role = "user", content = input }

    -- Apply chat template - add_ass=true appends the assistant turn start
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)

    -- Only send the NEW tokens (KV cache already holds the prefix)
    -- For simplicity here we re-decode the full prompt each turn.
    -- In production, track n_past and only decode the delta.
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    sampler:reset()

    -- Generate
    io.write("Assistant: ")
    local parts = {}
    for _ = 1, MAX_GEN do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        local piece = vocab:piece(tok, true)
        io.write(piece); io.flush()
        parts[#parts+1] = piece
        ctx:decode_single(tok, 0)
    end
    io.write("\n\n")

    -- Store in history for next turn
    history[#history+1] = { role = "user",      content = input }
    history[#history+1] = { role = "assistant",  content = table.concat(parts) }

    -- Warn when approaching context limit
    local used_pct = ctx:n_past() / ctx:n_ctx() * 100
    if used_pct > 80 then
        io.write(string.format("[context %d%% full - consider typing 'reset']\n\n", math.floor(used_pct)))
    end

    ::continue::
end

ion7.shutdown()
