#!/usr/bin/env luajit
--- @example examples.06_kv_reuse
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 06 — KV cache reuse : snapshots, sequence forking, sliding window
--- ════════════════════════════════════════════════════════════════════════
---
--- A long system prompt (or a long shared chat history) is expensive
--- to prefill but cheap to reuse. Three patterns covered here :
---
---   1. Snapshot / restore   : serialise the cache to a Lua string,
---                             reuse it later or in a different ctx.
---   2. Sequence forking     : duplicate seq 0 onto seq 1 so two
---                             samplers can branch from the same prefix
---                             without re-running the prefill.
---   3. Sliding window       : when n_past approaches n_ctx, shift
---                             old tokens out of the cache to free
---                             room. Only works on models whose KV
---                             reports `kv_can_shift() = true`.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/06_kv_reuse.lua

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

-- n_seq_max ≥ 2 so we can fork into a second sequence in demo 2.
local ctx = model:context({ n_ctx = 4096, n_seq_max = 4 })

--- Greedy generator helper. Operates on `seq_id`, returns the trimmed
--- text and stops on EOG / max_gen.
local function gen(seq_id, max_gen, sampler)
    sampler:reset()
    local pieces = {}
    for _ = 1, max_gen do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        pieces[#pieces + 1] = vocab:piece(tok)
        ctx:decode_single(tok, seq_id)
    end
    return (table.concat(pieces):gsub("^%s+", ""):gsub("%s+$", ""))
end

local greedy = ion7.Sampler.greedy()

-- ════════════════════════════════════════════════════════════════════════
-- 1. Snapshot / restore
-- ════════════════════════════════════════════════════════════════════════
--
-- Run a prefill, snapshot the state, restore later, verify the
-- restored context picks up exactly where the snapshot left off.
-- Greedy sampling makes the test deterministic — both runs must agree.

print("\n[1] Snapshot / restore")

ctx:kv_clear()
local prompt = vocab:apply_template(
    { { role = "user", content = "The capital of France is" } }, true, -1)
local toks, n = vocab:tokenize(prompt, false, true)
ctx:decode(toks, n)

-- We need a probe-decode AFTER snapshot so each side regenerates its
-- logits buffer (snapshots do not carry logits — those are recomputed
-- on the next decode). We sample the FIRST token now to anchor the
-- comparison.
local first_tok = greedy:sample(ctx:ptr(), -1)
print(string.format("  next-token after prefill : %d (%q)",
    first_tok, vocab:piece(first_tok)))

-- Take the snapshot AT this moment, then continue from it twice.
local blob   = ctx:snapshot()
local n_past = ctx:n_past()
print(string.format("  snapshot size : %.1f KB", #blob / 1024))

ctx:decode_single(first_tok)
local from_a = gen(0, 12, greedy)
print(string.format("  branch A (no restore) : %q", from_a))

-- Restore into a fresh context. The destination MUST mirror the
-- source's shape (n_ctx, n_seq_max, kv_type) — restore is a memcpy of
-- the cache layout, not a logical recreation. Mismatched contexts
-- silently corrupt or fail with "KV cache is full".
local ctx_b = model:context({ n_ctx = 4096, n_seq_max = 4 })
ctx_b:restore(blob)
ctx_b:set_n_past(n_past)
ctx_b:decode_single(first_tok)
greedy:reset()
local pieces_b = {}
for _ = 1, 12 do
    local tok = greedy:sample(ctx_b:ptr(), -1)
    if vocab:is_eog(tok) then break end
    pieces_b[#pieces_b + 1] = vocab:piece(tok)
    ctx_b:decode_single(tok)
end
local from_b = (table.concat(pieces_b):gsub("^%s+", ""):gsub("%s+$", ""))
print(string.format("  branch B (restored)    : %q", from_b))
print("  match : " .. tostring(from_a == from_b))
ctx_b:free()

-- ════════════════════════════════════════════════════════════════════════
-- 2. Sequence forking
-- ════════════════════════════════════════════════════════════════════════
--
-- Same prefill, two divergent continuations sampled with different
-- seeds. `kv_seq_cp` duplicates seq 0 onto seq 1 inside the same KV
-- cache, no extra prefill cost.

print("\n[2] Sequence forking")

ctx:kv_clear()
local p2 = vocab:apply_template(
    { { role = "user", content = "Name three programming languages :" } },
    true, -1)
local t2, n2 = vocab:tokenize(p2, false, true)
ctx:decode(t2, n2)
local prefill_pos = ctx:n_past()
print(string.format("  prefill done (%d tokens) — fork seq 0 → seq 1, seq 2",
    prefill_pos))

ctx:kv_seq_cp(0, 1, 0, -1)
ctx:kv_seq_cp(0, 2, 0, -1)

local s_warm = ion7.Sampler.chain():top_k(40):temperature(1.0):dist(111):build()
local s_cool = ion7.Sampler.chain():top_k(40):temperature(0.4):dist(999):build()

-- The Lua-side `_n_past` counter tracks the LAST decoded position,
-- not a per-sequence value. To run two parallel branches from the
-- same prefill we reset it back to `prefill_pos` between branches —
-- otherwise the second branch would try to write tokens at positions
-- past the end of its own sequence.
ctx:set_n_past(prefill_pos)
print("  branch A (T=1.0) : " .. gen(1, 60, s_warm):gsub("\n", " "))
ctx:set_n_past(prefill_pos)
print("  branch B (T=0.4) : " .. gen(2, 60, s_cool):gsub("\n", " "))
s_warm:free() ; s_cool:free()

-- ════════════════════════════════════════════════════════════════════════
-- 3. Sliding window (gated on kv_can_shift)
-- ════════════════════════════════════════════════════════════════════════
--
-- When the cache fills past `WINDOW`, rotate the oldest 25 % out by
-- shifting positions left. The model now attends to a moving window
-- rather than the original prompt. Models with chunked / segment
-- attention (some recent Qwen variants) report `kv_can_shift = false`
-- and skip this demo.

print("\n[3] Sliding window")

if not ctx:kv_can_shift() then
    print("  [SKIP] this model does not support position shifting")
else
    local WINDOW = 384
    ctx:kv_clear()
    local p3 = vocab:apply_template(
        { { role = "user",
            content = "Write a long imaginative paragraph about a brave fox." } },
        true, -1)
    local t3, n3 = vocab:tokenize(p3, false, true)
    ctx:decode(t3, n3)

    local fluent = ion7.Sampler.chain():top_k(40):temperature(0.8):dist(7):build()
    fluent:reset()
    local n_gen = 0
    for _ = 1, 400 do
        local tok = fluent:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok)
        n_gen = n_gen + 1
        if ctx:n_past() >= WINDOW then
            -- Slide the oldest 25 % out of the cache. `kv_seq_shift`
            -- moves cache positions, but the Lua-side `_n_past`
            -- mirror does not decrement automatically — we have to
            -- pull it back ourselves so the next decode lands at
            -- the correct position.
            local shift = math.floor(WINDOW / 4)
            ctx:kv_seq_shift(0, -shift, 0, -1)
            ctx:set_n_past(ctx:n_past() - shift)
        end
    end
    print(string.format("  generated %d tokens, KV held to ≤ %d",
        n_gen, WINDOW))
    fluent:free()
end

greedy:free() ; ctx:free() ; model:free()
ion7.shutdown()
