#!/usr/bin/env luajit
--- @module tests.16_sampler
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 16 — `Sampler` : fluent chains, sub-samplers, factories
--- ════════════════════════════════════════════════════════════════════════
---
--- A `Sampler` is a chain of token-selection steps applied in order to
--- the logits coming out of `llama_decode`. The fluent builder lets a
--- caller compose the chain top-down :
---
---   ion7.Sampler.chain()
---       :penalties(64, 1.1)
---       :top_k(40)
---       :top_p(0.95)
---       :min_p(0.05)
---       :temperature(0.8)
---       :dist()
---       :build()
---
--- This file walks the API in three movements :
---
---   1. The builder       : every per-sampler method, end-to-end.
---   2. The runtime       : `sample`, `accept`, `reset`, `clone`,
---                          chain introspection (`n`, `get`, `remove`).
---   3. The factories     : `default()`, `greedy()`, `creative()`.
---
--- We use a small live `Context` to actually call `:sample` and verify
--- the return is a token in `[0, n_vocab)`.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/16_sampler.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab   = model:vocab()
local n_vocab = vocab:n_vocab()

-- A short prompt decoded once so the context has logits to sample from.
local ctx = model:context({ n_ctx = 512, n_threads = 2 })
do
    local toks, n = vocab:tokenize("Hello", true, true)
    ctx:decode(toks, n)
end

local function valid_token(tok)
    -- Every sampler must return a token id in `[0, n_vocab)`. -1 is
    -- the "no logits available" sentinel — a real failure mode.
    T.is_type(tok, "number")
    T.gte(tok, 0,         "token id must be non-negative")
    T.gt(n_vocab, tok,    "token id must be < n_vocab")
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Builder API
-- ════════════════════════════════════════════════════════════════════════

T.suite("Builder API — per-sampler steps")

T.test("chain() returns a SamplerBuilder", function()
    local b = ion7.Sampler.chain()
    T.is_type(b,       "table")
    T.is_type(b.top_k, "function")
    T.is_type(b.dist,  "function")
    T.is_type(b.build, "function")
end)

T.test("greedy : the simplest possible chain", function()
    local s   = ion7.Sampler.chain():greedy():build()
    local tok = s:sample(ctx:ptr(), -1)
    valid_token(tok)
    s:free()
end)

T.test("dist with explicit seed reproduces the same draw twice", function()
    local s1  = ion7.Sampler.chain():dist(42):build()
    local t1  = s1:sample(ctx:ptr(), -1)
    s1:free()

    local s2  = ion7.Sampler.chain():dist(42):build()
    local t2  = s2:sample(ctx:ptr(), -1)
    s2:free()
    T.eq(t1, t2, "same seed must give the same token")
end)

T.test("temperature(0) is effectively greedy", function()
    -- temp 0 collapses the distribution onto the argmax.
    local s = ion7.Sampler.chain():temperature(0):dist():build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("temperature_dynamic accepts a base/range/exponent triplet", function()
    local s = ion7.Sampler.chain():temperature_dynamic(0.7, 0.2, 1.0):dist():build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("top_k narrows to the K most probable tokens", function()
    local s = ion7.Sampler.chain():top_k(5):dist(123):build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("top_p / min_p / typical / top_n_sigma all build and sample", function()
    local b = ion7.Sampler.chain()
        :top_p(0.95)
        :min_p(0.05)
        :typical(0.95)
        :top_n_sigma(1.0)
        :temperature(0.8)
        :dist(7)
    local s = b:build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("xtc with a low probability does not exclude the obvious top", function()
    -- probability=0 means XTC never triggers ; the chain reduces to dist.
    local s = ion7.Sampler.chain():xtc(0.0, 0.1):dist(1):build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("mirostat / mirostat_v2 build and sample", function()
    local s1 = ion7.Sampler.chain():mirostat(n_vocab, 5.0, 0.1):build()
    valid_token(s1:sample(ctx:ptr(), -1))
    s1:free()

    local s2 = ion7.Sampler.chain():mirostat_v2(5.0, 0.1):build()
    valid_token(s2:sample(ctx:ptr(), -1))
    s2:free()
end)

T.test("penalties(last_n, repeat, freq, present) builds and samples", function()
    local s = ion7.Sampler.chain()
        :penalties(64, 1.1, 0.0, 0.0)
        :dist(99)
        :build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("dry requires a vocab pointer and accepts seq breakers", function()
    local s = ion7.Sampler.chain():dry(
        { multiplier = 0.8, allowed_length = 2 },
        vocab._ptr,
        model:n_ctx_train()
    ):dist():build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("logit_bias passes through a sparse table of {token = delta}", function()
    -- Bias the EOS down so it does not get sampled by accident.
    local biases = {}
    if vocab:eos() >= 0 then biases[vocab:eos()] = -100.0 end
    local s = ion7.Sampler.chain():logit_bias(biases):dist(2):build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("adaptive_p builds and samples", function()
    local s = ion7.Sampler.chain():adaptive_p(0.45, 0.99, 11):build()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Runtime API
-- ════════════════════════════════════════════════════════════════════════

T.suite("Runtime — sample / reset / clone / introspection")

local function make_chain()
    return ion7.Sampler.chain()
        :top_k(40)
        :top_p(0.95)
        :temperature(0.8)
        :dist(1234)
        :build()
end

T.test("sample / accept / reset round-trip without raising", function()
    local s   = make_chain()
    local tok = s:sample(ctx:ptr(), -1)
    valid_token(tok)
    -- A `Sampler:sample` call on a CHAIN auto-accepts internally — we
    -- do NOT call `accept` after it. Reset clears any per-step state
    -- (mirostat, grammar, dry, ...).
    T.no_error(function() s:reset() end)
    s:free()
end)

T.test("seed() reports the chain's current RNG seed", function()
    local s = make_chain()
    local seed = s:seed()
    T.is_type(seed, "number")
    s:free()
end)

T.test("name() returns a non-empty string", function()
    local s = make_chain()
    T.gt(#s:name(), 0, "chain head should have a name")
    s:free()
end)

T.test("n() / get(i) / remove(i) operate on chain children", function()
    local s = make_chain()
    local n = s:n()
    T.gt(n, 0, "chain has children")
    -- Read back each child's name.
    for i = 0, n - 1 do
        local sub = s:get(i)
        T.is_type(sub, "table")
        T.is_type(sub:name(), "string")
    end
    -- Remove the first child and confirm the count drops.
    local removed = s:remove(0)
    T.is_type(removed, "table")
    T.eq(s:n(), n - 1)
    s:free()
end)

T.test("clone() produces an independent chain", function()
    local s  = make_chain()
    local c  = s:clone()
    T.is_type(c, "table")
    T.eq(c:n(), s:n())
    -- Both chains can still sample.
    valid_token(s:sample(ctx:ptr(), -1))
    valid_token(c:sample(ctx:ptr(), -1))
    s:free()
    c:free()
end)

T.test("perf() returns the documented counters", function()
    local s   = make_chain()
    local p   = s:perf()
    T.is_type(p,            "table")
    T.is_type(p.t_sample_ms,"number")
    T.is_type(p.n_sample,   "number")
    s:free()
end)

T.test("perf_reset and perf_print do not raise", function()
    local s = make_chain()
    s:sample(ctx:ptr(), -1)
    T.no_error(function() s:perf_reset() end)
    -- perf_print writes to stderr ; we only check it does not throw.
    T.no_error(function() s:perf_print() end)
    s:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Factories
-- ════════════════════════════════════════════════════════════════════════
--
-- Three named factories cover the bulk of typical use. They wrap the
-- builder with sensible defaults so a caller does not have to remember
-- the right top-k / top-p / temp triplet for "balanced" output.

T.suite("Factories — Sampler.default / greedy / creative")

T.test("Sampler.default() builds a balanced chain", function()
    local s = ion7.Sampler.default()
    T.gte(s:n(), 5, "default chain should have several steps")
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("Sampler.greedy() is a single-step chain", function()
    local s = ion7.Sampler.greedy()
    T.eq(s:n(), 1, "greedy chain has exactly one step")
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

T.test("Sampler.creative() is a more adventurous default", function()
    local s = ion7.Sampler.creative()
    valid_token(s:sample(ctx:ptr(), -1))
    s:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
