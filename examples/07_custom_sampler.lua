#!/usr/bin/env luajit
--- 07_custom_sampler.lua - Custom sampling logic written in pure Lua.
---
--- ion7-core allows you to implement any sampling strategy in Lua and plug it
--- directly into the C sampler chain with near-zero overhead (~12% vs native).
--- The apply callback receives the raw logit array and can do anything.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/07_custom_sampler.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)
local model  = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab  = model:vocab()
local ctx    = model:context({ n_ctx = 512 })

local function prefill_and_gen(prompt, sampler, max)
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
    return table.concat(parts)
end

-- ── 1. Lua greedy sampler ─────────────────────────────────────────────────────
-- Equivalent to built-in greedy, but written in Lua.

local function make_lua_greedy(name)
    return ion7.CustomSampler.new(name or "lua_greedy", {
        apply = function(cur_p, n)
            local best_i, best_logit = 0, -math.huge
            local data = cur_p.data
            for i = 0, n - 1 do
                if data[i].logit > best_logit then
                    best_logit = data[i].logit
                    best_i = i
                end
            end
            cur_p.selected = best_i
        end,
    })
end

io.write("\n[1. Lua greedy vs native greedy - should produce identical output]\n")

local cs_greedy  = make_lua_greedy()
local lua_chain  = ion7.Sampler.chain():custom(cs_greedy):build(vocab)
local native_chain = ion7.Sampler.chain():greedy():build(vocab)

local PROMPT = "The first three prime numbers are"
local out_lua    = prefill_and_gen(PROMPT, lua_chain,    32)
local out_native = prefill_and_gen(PROMPT, native_chain, 32)

io.write("  Lua greedy:    " .. out_lua:gsub("\n", " ")    .. "\n")
io.write("  Native greedy: " .. out_native:gsub("\n", " ") .. "\n")
io.write("  Match: " .. tostring(out_lua == out_native) .. "\n")

-- ── 2. Contrastive sampling ───────────────────────────────────────────────────
-- Penalises tokens that appear in a "negative" distribution.
-- Concept: for each candidate, logit = logit_pos - alpha * logit_neg
-- Here simplified: penalise tokens that are too "common" (high raw probability).

local ALPHA = 0.5  -- penalty strength

local function make_contrastive(alpha)
    return ion7.CustomSampler.new("contrastive", {
        apply = function(cur_p, n)
            local data = cur_p.data

            -- Compute mean logit as a "common" baseline
            local sum = 0
            for i = 0, n - 1 do sum = sum + data[i].logit end
            local mean = sum / n

            -- Penalise tokens above mean (too predictable)
            local best_i, best_adjusted = 0, -math.huge
            for i = 0, n - 1 do
                local adjusted = data[i].logit - alpha * math.max(0, data[i].logit - mean)
                if adjusted > best_adjusted then
                    best_adjusted = adjusted
                    best_i = i
                end
            end
            cur_p.selected = best_i
        end,
    })
end

io.write("\n[2. Contrastive sampler (penalises predictable tokens)]\n")

local cs_contrast = make_contrastive(ALPHA)
local contrast_chain = ion7.Sampler.chain()
    :top_k(50)
    :custom(cs_contrast)
    :build(vocab)

local out_contrast = prefill_and_gen("Tell me something surprising about space", contrast_chain, 80)
io.write("  Output: " .. out_contrast:gsub("\n", " ") .. "\n")

-- ── 3. Stateful sampler - tracks token history ────────────────────────────────
-- Custom samplers can maintain state between tokens using upvalues.

io.write("\n[3. Stateful sampler - penalises last 8 tokens (no C++ needed)]\n")

local function make_recent_penalty(window, penalty)
    local recent = {}  -- ring buffer of recent token ids
    return ion7.CustomSampler.new("recent_penalty", {
        apply = function(cur_p, n)
            local data = cur_p.data
            -- Build a set of recent tokens for O(1) lookup
            local blocked = {}
            for _, id in ipairs(recent) do blocked[id] = true end

            local best_i, best_logit = 0, -math.huge
            for i = 0, n - 1 do
                local logit = data[i].logit
                if blocked[data[i].id] then logit = logit - penalty end
                if logit > best_logit then best_logit = logit; best_i = i end
            end
            cur_p.selected = best_i
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

local cs_penalty   = make_recent_penalty(8, 2.0)
local penalty_chain = ion7.Sampler.chain()
    :top_k(40):temperature(0.8):dist(42)
    :custom(cs_penalty)
    :build(vocab)

local out_penalty = prefill_and_gen(
    "List colours: red, blue, green,", penalty_chain, 60
)
io.write("  Output: " .. out_penalty:gsub("\n", " ") .. "\n")

-- Cleanup
lua_chain:free(); native_chain:free()
contrast_chain:free(); penalty_chain:free()

ion7.shutdown()
