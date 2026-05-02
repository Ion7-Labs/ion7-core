#!/usr/bin/env luajit
--- @example examples.03_streaming
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 03 — Streaming output : per-token callback, UTF-8 safety, early stop
--- ════════════════════════════════════════════════════════════════════════
---
--- Three things every real streamer needs :
---
---   1. A per-token callback — so the UI can display text as it arrives.
---   2. UTF-8 boundary safety — multi-byte codepoints (CJK, emoji) often
---      span two tokens. Emitting a half-codepoint to the terminal
---      shows up as a U+FFFD square. `ion7.utf8.is_complete` lets us
---      hold partial bytes until the next token completes them.
---   3. Early stop — let the caller cancel generation by returning
---      false from the callback.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/03_streaming.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit     = ion7.Model.fit_params(MODEL)
local model   = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local vocab   = model:vocab()
local ctx     = model:context({ n_ctx = 2048 })
local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95):temperature(0.8):dist(42):build()

--- Run a prompt through the model and call `on_chunk` with every UTF-8
--- complete piece of output. `on_chunk` returns `false` to stop early.
---
--- @param  user_text string  Plain user message ; we wrap it with the
---                           model's chat template here.
--- @param  max_gen   integer Maximum new tokens.
--- @param  on_chunk  fun(chunk:string, count:integer):boolean
--- @return string  The full reply.
--- @return integer Number of tokens generated.
local function stream(user_text, max_gen, on_chunk)
    local prompt = vocab:apply_template(
        { { role = "user", content = user_text } }, true, -1)
    local toks, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(toks, n)
    sampler:reset()
    ctx:perf_reset()

    local pending = ""    -- bytes whose codepoint has not yet completed
    local full    = {}
    local count   = 0

    for _ = 1, max_gen do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        count = count + 1

        -- Concatenate with any pending fragment from the previous step.
        local piece = pending .. vocab:piece(tok)
        if ion7.utf8.is_complete(piece) then
            -- Whole codepoints — safe to emit.
            full[#full + 1] = piece
            pending = ""
            if on_chunk(piece, count) == false then break end
        else
            -- Last codepoint is half-emitted ; hold it back.
            pending = piece
        end

        ctx:decode_single(tok)
    end

    -- Flush whatever bytes we held back (final partial codepoint, if any).
    if #pending > 0 then full[#full + 1] = pending end
    return table.concat(full), count
end

-- ── Demo 1 : plain stream, all tokens ────────────────────────────────────
print("\n[demo 1] streaming a short reply")
io.write("Reply : ")
local reply, n_gen = stream("Count from one to five.", 64, function(chunk)
    io.write(chunk) ; io.flush()
    return true
end)
local p = ctx:perf()
print(string.format(
    "\n  → %d tokens, %.1f tok/s (prefill %d in %.0f ms, gen %d in %.0f ms)",
    n_gen, p.tokens_per_s, p.n_p_eval, p.t_p_eval_ms, p.n_eval, p.t_eval_ms))

-- ── Demo 2 : early stop on a keyword ─────────────────────────────────────
print("\n[demo 2] stop the stream as soon as the word \"five\" appears")
io.write("Reply : ")
local buffer, stopped = "", false
stream("Count slowly from one to ten.", 200, function(chunk)
    buffer = buffer .. chunk
    io.write(chunk) ; io.flush()
    -- Case-insensitive plain search : the model often capitalises
    -- single-word counts ("Five.") and a case-sensitive `find` would
    -- silently miss the trigger.
    if buffer:lower():find("five", 1, true) then
        stopped = true
        return false
    end
    return true
end)
print(stopped and "\n  → stopped early on keyword"
                or "\n  → keyword never appeared, generation reached its natural end")

-- ── Demo 3 : collect first, post-process ─────────────────────────────────
print("\n[demo 3] collect the full reply, then count words")
local full = stream("In one paragraph, what is a coroutine ?", 256,
                    function() return true end)
local words = 0
for _ in full:gmatch("%S+") do words = words + 1 end
print(string.format("Reply (%d words) : %s", words, full:gsub("\n", " ")))

sampler:free() ; ctx:free() ; model:free()
ion7.shutdown()
