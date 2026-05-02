#!/usr/bin/env luajit
--- @example examples.02_chat
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 02 — Multi-turn chat with KV-delta prefilling
--- ════════════════════════════════════════════════════════════════════════
---
--- A real chat loop must NOT re-tokenise + re-decode the whole history
--- on every turn — that's quadratic in turn count and ruins throughput.
--- The trick is to keep `n_past` synchronised with the formatted prompt
--- prefix : on each turn we tokenise the FULL formatted history, then
--- decode only the suffix that the cache hasn't seen yet.
---
--- The example is non-interactive : it walks through three pre-recorded
--- user turns so it can be run from CI / smoke tests. Swap the
--- `USER_TURNS` table for an `io.read()` loop to make it interactive.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/02_chat.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit   = ion7.Model.fit_params(MODEL)
local model = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local vocab   = model:vocab()
local ctx     = model:context({ n_ctx = fit and fit.n_ctx or 4096 })
local sampler = ion7.Sampler.chain()
    :top_k(40)
    :top_p(0.95)
    :temperature(0.8)
    :penalties(64, 1.1, 0.0, 0.0)
    :dist(os.time())
    :build()

-- The conversation : a system prompt + N user turns we hardcode for the
-- demo, plus the assistant replies the model will fill in.
local SYSTEM = "You are a helpful and concise assistant."
local USER_TURNS = {
    "Reply with a single sentence : what does LuaJIT do ?",
    "And what about its FFI ?",
    "Thanks. In one word, would you call it fast ?",
}
local MAX_GEN_PER_TURN = 200

local history = { { role = "system", content = SYSTEM } }

--- Decode just the tokens that come AFTER the part already in the KV
--- cache. We re-render the chat template on every turn (the template
--- output is stable for the prefix), tokenise the full prompt, then
--- compare against the prefix already loaded into the cache : tokens
--- past `n_seen` are the new suffix that needs decoding.
---
--- @param  toks   cdata   `int32_t[?]` tokens of the FULL formatted prompt.
--- @param  n      integer Total token count.
--- @param  n_seen integer Tokens already in the KV cache (== ctx:n_past()).
local function decode_delta(toks, n, n_seen)
    if n <= n_seen then
        -- Prompt got shorter (history was reset / truncated) : start
        -- over from a clean cache.
        ctx:kv_clear()
        ctx:decode(toks, n)
        return
    end

    -- Build a cdata slice starting at n_seen. We could allocate a new
    -- array but reusing the original one with a pointer offset is
    -- cheaper. LuaJIT's cdata pointer arithmetic returns a `int32_t*`.
    local ffi = require "ffi"
    local suffix = ffi.cast("int32_t *", toks) + n_seen
    ctx:decode(suffix, n - n_seen)
end

for turn_idx, user_msg in ipairs(USER_TURNS) do
    history[#history + 1] = { role = "user", content = user_msg }

    -- Render the FULL conversation (template prefix + every turn so far)
    -- and tokenise once. The template writes a stable prefix as long as
    -- the leading messages don't change, which is exactly the case
    -- between turns of a real chat.
    local prompt   = vocab:apply_template(history, true, -1)
    local toks, n  = vocab:tokenize(prompt, false, true)

    -- Decode only the new tokens. On turn 1 `ctx:n_past()` is 0 so the
    -- whole prompt is new ; on turn 2+ only the appended user message
    -- (and assistant prefix) is new.
    decode_delta(toks, n, ctx:n_past())
    sampler:reset()

    io.write(string.format("\n[turn %d] You : %s\n", turn_idx, user_msg))
    io.write("[turn " .. turn_idx .. "] Assistant : ")

    local reply_pieces = {}
    for _ = 1, MAX_GEN_PER_TURN do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        local p = vocab:piece(tok)
        io.write(p) ; io.flush()
        reply_pieces[#reply_pieces + 1] = p
        ctx:decode_single(tok)
    end
    io.write("\n")

    -- Persist the assistant reply in the history so the next template
    -- render contains it as a plain string.
    history[#history + 1] = {
        role    = "assistant",
        content = table.concat(reply_pieces),
    }

    -- Friendly reminder when the cache is filling up.
    local pct = ctx:n_past() / ctx:n_ctx() * 100
    if pct > 80 then
        io.write(string.format("[ctx %.0f%% full]\n", pct))
    end
end

print(string.format("\n[done] %d turns, KV used %d / %d tokens",
    #USER_TURNS, ctx:n_past(), ctx:n_ctx()))

sampler:free() ; ctx:free() ; model:free()
ion7.shutdown()
