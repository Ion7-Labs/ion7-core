#!/usr/bin/env luajit
--- @module tests.17_custom_sampler
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 17 — `CustomSampler` : Lua-implemented sampling step
--- ════════════════════════════════════════════════════════════════════════
---
--- A `CustomSampler` lets you plug a Lua function into a sampler chain
--- as a real `llama_sampler*`. The implementation glues three
--- `ffi.cast` trampolines (`apply`, `accept`, `reset`) into a struct
--- llama.cpp owns by pointer ; the Lua wrapper anchors the cdata
--- callbacks so the GC cannot collect them while llama.cpp holds the
--- pointers.
---
--- This file demonstrates :
---
---   1. The `apply` callback contract :
---      `apply(candidates_cdata, n_int) -> chosen_index`.
---   2. Plugging a CustomSampler into a chain via `:custom(cs)`.
---   3. The `accept` and `reset` optional callbacks.
---   4. The "errors are caught" property : a buggy `apply` falls back
---      to selecting candidate 0 rather than crashing the process.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/17_custom_sampler.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab   = model:vocab()
local n_vocab = vocab:n_vocab()
local ctx     = model:context({ n_ctx = 512, n_threads = 2 })

do
    local toks, n = vocab:tokenize("Hello", true, true)
    ctx:decode(toks, n)
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — A pure-Lua greedy sampler
-- ════════════════════════════════════════════════════════════════════════
--
-- Greedy is the simplest non-trivial demonstration : pick the argmax
-- of the logit array. The same logic in 6 lines of Lua reproduces
-- what `llama_sampler_init_greedy` does in C.

T.suite("Greedy CustomSampler")

local apply_calls = 0

local greedy_lua = ion7.CustomSampler.new("greedy_lua", {
    apply = function(candidates, n)
        apply_calls = apply_calls + 1
        local best, best_logit = 0, -math.huge
        for i = 0, n - 1 do
            local v = candidates[i].logit
            if v > best_logit then
                best_logit = v
                best = i
            end
        end
        return best
    end,
})

T.test("CustomSampler.new returns an instance with name + ptr", function()
    T.is_type(greedy_lua,        "table")
    T.eq(greedy_lua:name(),      "greedy_lua")
    T.is_type(greedy_lua:ptr(),  "cdata")
end)

T.test("plugged into a chain, the Lua sampler is actually invoked", function()
    apply_calls = 0
    local chain = ion7.Sampler.chain():custom(greedy_lua):build()

    local tok = chain:sample(ctx:ptr(), -1)
    T.gte(apply_calls, 1, "apply must have been called at least once")
    T.gte(tok, 0)
    T.gt(n_vocab, tok)

    -- A second sample on the same chain must keep working.
    chain:sample(ctx:ptr(), -1)
    T.gte(apply_calls, 2)
    chain:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — accept / reset callbacks
-- ════════════════════════════════════════════════════════════════════════
--
-- Both are optional : when omitted, the corresponding slot in the
-- `llama_sampler_i` interface stays NULL and llama.cpp skips it. We
-- pass both here and verify they fire.

T.suite("Optional accept / reset callbacks")

-- Sampler chains take ownership of the CustomSampler's underlying
-- `llama_sampler*` and free it on chain teardown, so a single
-- CustomSampler instance must NOT be reused across multiple chains.
-- Each test builds a fresh wrapper with its own callbacks table.

T.test("accept fires once per chain.sample call", function()
    local accepted = {}
    local cs = ion7.CustomSampler.new("accept-hook", {
        apply  = function(_, _) return 0 end,
        accept = function(tok) accepted[#accepted + 1] = tok end,
    })
    local chain = ion7.Sampler.chain():custom(cs):build()
    chain:sample(ctx:ptr(), -1)
    chain:sample(ctx:ptr(), -1)
    -- A chain auto-accepts on every sample, so we should see two entries.
    T.eq(#accepted, 2)
    chain:free()
end)

T.test("reset fires when the chain is reset", function()
    local reset_count = 0
    local cs = ion7.CustomSampler.new("reset-hook", {
        apply = function(_, _) return 0 end,
        reset = function()    reset_count = reset_count + 1 end,
    })
    local chain = ion7.Sampler.chain():custom(cs):build()
    chain:reset()
    T.eq(reset_count, 1)
    chain:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Error containment
-- ════════════════════════════════════════════════════════════════════════
--
-- Any uncaught Lua error inside an `ffi.cast` callback aborts the
-- process. The wrapper traps every callback in `pcall` and falls back
-- to selecting candidate 0. We verify by deliberately raising in
-- `apply` and confirming the chain still returns a valid token id.

T.suite("Error containment")

T.test("a raising apply() falls back to candidate 0 instead of crashing",
    function()
        local boom = ion7.CustomSampler.new("boom", {
            apply = function(_, _)
                error("intentional", 0)
            end,
        })
        local chain = ion7.Sampler.chain():custom(boom):build()
        local tok = chain:sample(ctx:ptr(), -1)
        T.is_type(tok, "number")
        T.gte(tok, 0)
        chain:free()
    end)

T.test("an out-of-range return falls back to candidate 0", function()
    local cs = ion7.CustomSampler.new("oob", {
        apply = function(_, _) return -1 end, -- invalid
    })
    local chain = ion7.Sampler.chain():custom(cs):build()
    local tok = chain:sample(ctx:ptr(), -1)
    T.is_type(tok, "number")
    chain:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
