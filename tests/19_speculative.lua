#!/usr/bin/env luajit
--- @module tests.19_speculative
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 19 — Speculative decoding (n-gram and draft-model variants)
--- ════════════════════════════════════════════════════════════════════════
---
--- Speculative decoding accelerates token generation by predicting a
--- short batch of candidate tokens ahead of the target model, then
--- verifying them all in a single forward pass. Two flavours :
---
---   - n-gram cache (`ngram_cache`) : a recent-context lookup table.
---     Needs no second model — works on any context.
---   - draft model (`draft`)        : a small fast model produces
---     guesses the bigger target verifies. Set `ION7_DRAFT` to the
---     small `.gguf` to exercise this path ; the suite skips it
---     gracefully when unset.
---
---   ION7_MODEL=/path/to/model.gguf [ION7_DRAFT=/path/to/draft.gguf] \
---       luajit tests/19_speculative.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()
local ctx   = model:context({ n_ctx = 512, n_threads = 2 })

do
    local toks, n = vocab:tokenize("Hello world", true, true)
    ctx:decode(toks, n)
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Type aliases and constants
-- ════════════════════════════════════════════════════════════════════════

T.suite("Type constants")

T.test("Speculative.<TYPE> constants are integers", function()
    T.eq(ion7.Speculative.NONE,          0)
    T.eq(ion7.Speculative.DRAFT,         1)
    T.eq(ion7.Speculative.EAGLE3,        2)
    T.eq(ion7.Speculative.NGRAM_SIMPLE,  3)
    T.eq(ion7.Speculative.NGRAM_CACHE,   7)
end)

T.test("string aliases resolve at construction time", function()
    -- Build with a string alias ; the constructor must accept it
    -- exactly like the numeric constant.
    local spec = ion7.Speculative.new(ctx, nil, { type = "ngram_cache" })
    T.is_type(spec, "table")
    spec:free()
end)

T.test("an unknown type alias raises with a helpful message", function()
    T.err(
        function() ion7.Speculative.new(ctx, nil, { type = "no_such" }) end,
        "unknown type"
    )
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — N-gram-cache flow (no draft model needed)
-- ════════════════════════════════════════════════════════════════════════
--
-- We exercise the full lifecycle :
--   begin(prompt) → draft(history, last_tok) → accept(k) → free.
--
-- The acceptance counter does not have to be > 0 on the first call —
-- the cache warms up over several rounds. We assert only that each
-- step returns the right Lua type, not specific outcomes.

T.suite("N-gram cache flow")

T.test("begin / draft / accept on a fresh n-gram engine", function()
    local spec = ion7.Speculative.new(ctx, nil, {
        type = "ngram_cache", n_draft = 4, ngram_min = 2, ngram_max = 4,
    })

    -- Tokens we have produced so far. begin() seeds the engine with
    -- the prompt history.
    local prompt_table = {}
    local toks, n = vocab:tokenize("Hello world", false, true)
    for i = 0, n - 1 do prompt_table[i + 1] = toks[i] end
    T.no_error(function() spec:begin(prompt_table) end)

    -- Ask for a batch of drafts. The result is a 1-based Lua array
    -- (possibly empty when the cache is cold).
    local last_tok = prompt_table[#prompt_table]
    local drafts = spec:draft(prompt_table, last_tok)
    T.is_type(drafts, "table")
    -- accept() acknowledges how many of the drafts the target
    -- actually matched ; passing 0 is the canonical "we used none".
    T.no_error(function() spec:accept(0) end)
    -- stats() prints a summary line to stderr — must not raise.
    T.no_error(function() spec:stats() end)
    spec:free()
end)

T.test("free() then free() is idempotent", function()
    local spec = ion7.Speculative.new(ctx, nil, { type = "ngram_cache" })
    T.no_error(function() spec:free() end)
    T.no_error(function() spec:free() end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Draft-model flow
-- ════════════════════════════════════════════════════════════════════════
--
-- Two distinct contexts are required ; the underlying model can be
-- shared. We default to loading the test model TWICE so the path is
-- always exercised — picking a smaller dedicated draft model via
-- `ION7_DRAFT` only matters for actual speedup measurement, not for
-- API correctness.

T.suite("Draft-model flow")

T.test("Speculative.new with type='draft' on two contexts", function()
    local draft_path  = H.draft_model_path() or H.require_model(T)
    local draft_model = ion7.Model.load(draft_path,
                                         { n_gpu_layers = H.gpu_layers() })
    local ctx_dft     = draft_model:context({ n_ctx = 512, n_threads = 2 })

    local spec = ion7.Speculative.new(ctx, ctx_dft, {
        type = "draft", n_draft = 5,
    })
    T.is_type(spec, "table")

    -- Smoke-test the same begin / draft / accept path on a
    -- prompted target.
    local prompt_table = {}
    local toks, n = vocab:tokenize("Once upon a time", false, true)
    for i = 0, n - 1 do prompt_table[i + 1] = toks[i] end
    T.no_error(function() spec:begin(prompt_table) end)
    local drafts = spec:draft(prompt_table, prompt_table[#prompt_table])
    T.is_type(drafts, "table")
    T.no_error(function() spec:accept(0) end)

    spec:free()
    ctx_dft:free()
    draft_model:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
