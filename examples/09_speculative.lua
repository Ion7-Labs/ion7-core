#!/usr/bin/env luajit
--- @example examples.09_speculative
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 09 — Speculative decoding : accelerate generation with draft tokens
--- ════════════════════════════════════════════════════════════════════════
---
--- Speculative decoding lets a fast "drafter" propose a short batch of
--- tokens that the target model verifies in a single forward pass. If
--- many drafts are correct, the target advances by several positions
--- per forward pass instead of one — a 1.5× to 3× speedup on
--- agreement-friendly workloads (code, structured text, long quoting
--- tasks).
---
--- Two drafter strategies are demonstrated :
---
---   1. n-gram cache : a pure look-up table built from the recent
---      context. No second model needed — works on any prompt.
---   2. Draft model  : a smaller model proposes tokens, the target
---      verifies. Set `ION7_DRAFT` to a separate small `.gguf`. When
---      unset, the example falls back to loading the target a second
---      time, which still exercises the API end-to-end.
---
---   ION7_MODEL=/path/to/model.gguf [ION7_DRAFT=/path/to/draft.gguf] \
---       luajit examples/09_speculative.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)
local DRAFT = os.getenv("ION7_DRAFT") or MODEL  -- fallback : same file

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit     = ion7.Model.fit_params(MODEL)
local model   = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local vocab   = model:vocab()
local ctx     = model:context({ n_ctx = 1024, n_threads = 4 })
local sampler = ion7.Sampler.greedy()

-- ════════════════════════════════════════════════════════════════════════
-- 1. n-gram cache speculative — no second model
-- ════════════════════════════════════════════════════════════════════════
--
-- The simplest variant : at every step the engine scans recent
-- context for an n-gram match (3 to 5 tokens) and proposes its
-- continuation as the draft. Free in compute, sometimes free in
-- accuracy too (great for boilerplate / repeating structures).

print("\n[1] n-gram cache speculative")

local PROMPT = "Repeat the sentence three times exactly : Today the sun rises in the east."
local prompt   = vocab:apply_template(
    { { role = "user", content = PROMPT } }, true, -1)
local toks, n  = vocab:tokenize(prompt, false, true)

ctx:kv_clear()
ctx:decode(toks, n)

local spec = ion7.Speculative.new(ctx, nil, {
    type = "ngram_cache", n_draft = 5, ngram_min = 3, ngram_max = 5,
})

-- Seed the engine with the prompt history so the cache has something
-- to look up against on the first iteration.
local prompt_history = {}
for i = 0, n - 1 do prompt_history[i + 1] = toks[i] end
spec:begin(prompt_history)

io.write("  reply : ")
local generated, n_drafts_total, n_accepted_total = {}, 0, 0
local last_tok = prompt_history[#prompt_history]

for _ = 1, 100 do
    -- Ask the drafter for up to `n_draft` likely-next tokens.
    local drafts = spec:draft(prompt_history, last_tok)
    n_drafts_total = n_drafts_total + #drafts

    -- For the n-gram demo we only sample one token per round (no
    -- batched verification) — the API is the same as a regular
    -- generation loop. A real consumer would use `decode_multi` to
    -- verify all drafts in one forward pass and `spec:accept(k)` with
    -- the agreement count. We use a single-token loop here for
    -- pedagogical clarity.
    local tok = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(tok) then spec:accept(0) ; break end

    -- Count how many of the drafts matched our greedy choice.
    local k = 0
    for i, d in ipairs(drafts) do
        if d == tok and i == 1 then k = 1 end
    end
    n_accepted_total = n_accepted_total + k
    spec:accept(k)

    io.write(vocab:piece(tok)) ; io.flush()
    generated[#generated + 1]      = tok
    prompt_history[#prompt_history + 1] = tok
    last_tok = tok
    ctx:decode_single(tok)
end

io.write("\n")
print(string.format("  drafts proposed : %d, accepted : %d (%.0f %% hit-rate)",
    n_drafts_total, n_accepted_total,
    n_drafts_total > 0 and (100 * n_accepted_total / n_drafts_total) or 0))
spec:free()

-- ════════════════════════════════════════════════════════════════════════
-- 2. Draft-model speculative — second model proposes, target verifies
-- ════════════════════════════════════════════════════════════════════════
--
-- The DRAFT-type engine takes a separate `Context` from a smaller
-- model. We pre-create that context just like any other inference
-- context. The bridge auto-shapes the draft window from the target's
-- size, so the same model used twice is a valid (no-speedup but
-- correctness-validating) configuration.

print("\n[2] Draft-model speculative")
print("  loading draft model : " .. DRAFT:match("[^/]+$"))

local draft_model = ion7.Model.load(DRAFT, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local ctx_dft = draft_model:context({ n_ctx = 1024, n_threads = 4 })

local spec2 = ion7.Speculative.new(ctx, ctx_dft, {
    type = "draft", n_draft = 5,
})

ctx:kv_clear()
ctx:decode(toks, n)
spec2:begin(prompt_history)
local drafts2 = spec2:draft(prompt_history, last_tok)
print(string.format("  first round drafts (n=%d) : %s", #drafts2,
    table.concat((function()
        local labels = {}
        for _, d in ipairs(drafts2) do
            labels[#labels + 1] = vocab:piece(d):gsub("\n", "\\n")
        end
        return labels
    end)(), " | ")))
spec2:accept(0)
spec2:free()

ctx_dft:free() ; draft_model:free()

sampler:free() ; ctx:free() ; model:free()
ion7.shutdown()
