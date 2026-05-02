#!/usr/bin/env luajit
--- @example examples.07_custom_sampler
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 07 — CustomSampler : sampling logic written in pure Lua
--- ════════════════════════════════════════════════════════════════════════
---
--- A `CustomSampler` plugs a Lua function into the `llama_sampler`
--- chain. The trampoline gives your callback two arguments :
---
---   apply(candidates, n) -> chosen_index
---     candidates : `llama_token_data*` cdata array (0-based).
---                  Each element has `.id` (token), `.logit`, `.p`.
---     n          : number of candidates after upstream filtering.
---     return       0-based index of the chosen candidate.
---
--- Optional callbacks `accept(token)` and `reset()` are called by the
--- chain when a token is committed and when the chain is reset.
---
--- Three samplers below : a Lua greedy (sanity check), a contrastive
--- sampler (penalises tokens that lie above the mean logit), and a
--- stateful "no-recent-repeat" sampler that uses upvalues to remember
--- the last N tokens.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/07_custom_sampler.lua

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
local ctx   = model:context({ n_ctx = 1024, n_threads = 2 })

--- Run a fresh prompt through `sampler` and return the trimmed reply.
local function generate(prompt_text, sampler, max_gen)
    local prompt = vocab:apply_template(
        { { role = "user", content = prompt_text } }, true, -1)
    local toks, n = vocab:tokenize(prompt, false, true)
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
-- 1. Lua greedy — sanity check
-- ════════════════════════════════════════════════════════════════════════
--
-- The simplest non-trivial sampler : pick the highest-logit candidate.
-- Equivalent to the C++ `:greedy()` step ; running both side by side
-- on the same prompt should produce identical output.

print("\n[1] Lua greedy vs native greedy — must match")

local function pick_argmax(candidates, n)
    local best, best_logit = 0, -math.huge
    for i = 0, n - 1 do
        local v = candidates[i].logit
        if v > best_logit then
            best_logit = v
            best       = i
        end
    end
    return best
end

local lua_greedy   = ion7.CustomSampler.new("lua_greedy", { apply = pick_argmax })
local lua_chain    = ion7.Sampler.chain():custom(lua_greedy):build()
local native_chain = ion7.Sampler.greedy()

local PROMPT = "List the first three prime numbers."
local from_lua    = generate(PROMPT, lua_chain, 32):gsub("\n", " ")
local from_native = generate(PROMPT, native_chain, 32):gsub("\n", " ")
print("  Lua greedy    : " .. from_lua)
print("  Native greedy : " .. from_native)
print("  match : " .. tostring(from_lua == from_native))

lua_chain:free() ; native_chain:free()

-- ════════════════════════════════════════════════════════════════════════
-- 2. Contrastive sampler
-- ════════════════════════════════════════════════════════════════════════
--
-- For every candidate, subtract a penalty proportional to "how far
-- above the mean logit" it sits. Predictable tokens (high logit)
-- lose more, surprising ones (near the mean) lose less. The result
-- is more variety in the output without dropping coherence.
---
--- An upstream `:top_k(50)` keeps the candidate set sane — without it
--- our O(n) loop scans 130 000 tokens every step.

print("\n[2] Contrastive sampler (penalise predictable tokens)")

local ALPHA = 0.5
local function contrastive(candidates, n)
    -- Pass 1 : compute the mean logit for the kept candidates.
    local sum = 0
    for i = 0, n - 1 do sum = sum + candidates[i].logit end
    local mean = sum / n

    -- Pass 2 : pick the token whose adjusted logit is highest.
    local best, best_adj = 0, -math.huge
    for i = 0, n - 1 do
        local adj = candidates[i].logit
                  - ALPHA * math.max(0, candidates[i].logit - mean)
        if adj > best_adj then best_adj = adj ; best = i end
    end
    return best
end

local contrastive_sampler = ion7.CustomSampler.new("contrastive",
    { apply = contrastive })
local contrastive_chain = ion7.Sampler.chain()
    :top_k(50)
    :custom(contrastive_sampler)
    :build()

print("  output : " ..
    generate("Tell me one surprising fact about octopuses.",
             contrastive_chain, 80):gsub("\n", " "))
contrastive_chain:free()

-- ════════════════════════════════════════════════════════════════════════
-- 3. Stateful no-recent-repeat sampler
-- ════════════════════════════════════════════════════════════════════════
--
-- Tracks the last `WINDOW` accepted tokens via the `accept` callback
-- and slaps a flat penalty on any candidate whose id is in that ring
-- buffer. Demonstrates upvalues persisting across `apply` and
-- `accept` invocations for the same `CustomSampler` instance, plus
-- the `reset` callback wiping the state on chain reset.

print("\n[3] Stateful sampler — penalise the last 8 generated tokens")

local function make_no_repeat(window, penalty)
    local recent = {}
    return ion7.CustomSampler.new("no_repeat", {
        apply = function(candidates, n)
            -- Build the blocked set on the hot path. For tiny `window`
            -- (≤ 16) a plain table beats a sorted-array bsearch.
            local blocked = {}
            for _, id in ipairs(recent) do blocked[id] = true end

            local best, best_adj = 0, -math.huge
            for i = 0, n - 1 do
                local adj = candidates[i].logit
                if blocked[candidates[i].id] then adj = adj - penalty end
                if adj > best_adj then best_adj = adj ; best = i end
            end
            return best
        end,

        accept = function(tok)
            recent[#recent + 1] = tok
            if #recent > window then table.remove(recent, 1) end
        end,

        reset = function()
            recent = {}
        end,
    })
end

local no_repeat = make_no_repeat(8, 2.0)
local no_repeat_chain = ion7.Sampler.chain()
    :top_k(40)
    :temperature(0.8)
    :dist(42)
    :custom(no_repeat)
    :build()

print("  output : " ..
    generate("Continue the list of colours : red, blue, green,",
             no_repeat_chain, 80):gsub("\n", " "))
no_repeat_chain:free()

ctx:free() ; model:free()
ion7.shutdown()
