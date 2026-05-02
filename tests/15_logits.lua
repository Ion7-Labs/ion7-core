#!/usr/bin/env luajit
--- @module tests.15_logits
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 15 — Logits, log-probabilities, entropy, embeddings, sampled accessors
--- ════════════════════════════════════════════════════════════════════════
---
--- After a `decode`, the context exposes a logit vector at every
--- position whose batch slot had `logits = 1`. This file covers :
---
---   1. Raw logit pointer access (`logits(idx)`).
---   2. The numerical helpers `logprob`, `entropy`, and the combined
---      single-pass `logprob_entropy`.
---   3. The "what did we sample on the C side" accessors
---      (`sampled_token`, `sampled_probs`, `sampled_logits`,
---      `sampled_candidates`) that an attached `set_sampler`
---      callback exposes.
---   4. Embedding extraction for sequence-level vectors (only meaningful
---      with an embedding context — skipped on a generation context).
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/15_logits.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab   = model:vocab()
local n_vocab = vocab:n_vocab()
local ctx     = model:context({ n_ctx = 1024, n_threads = 2 })

-- Decode a short prompt so the context has logits to read.
local toks, n = vocab:tokenize("Hello, world!", true, true)
ctx:decode(toks, n)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Raw logits
-- ════════════════════════════════════════════════════════════════════════

T.suite("Raw logits — logits(idx) and immediate sampling-result accessors")

T.test("logits(-1) returns a non-NULL float* cdata", function()
    local p = ctx:logits(-1)
    T.is_type(p, "cdata")
    -- A non-NULL pointer can be indexed without crashing ; we read the
    -- first slot to confirm the cdata is the right type.
    T.is_type(tonumber(p[0]), "number")
end)

T.test("sampled_* accessors return numbers/cdata even with no sampler attached",
    function()
        -- Without `set_sampler`, the sampled-* accessors return
        -- pointer-or-zero defaults. We only verify the FFI plumbing
        -- works end-to-end (no crash, no Lua error).
        T.is_type(ctx:sampled_token(0),             "number")
        T.is_type(ctx:sampled_probs_count(0),       "number")
        T.is_type(ctx:sampled_logits_count(0),      "number")
        T.is_type(ctx:sampled_candidates_count(0),  "number")
    end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — logprob, entropy, combined readout
-- ════════════════════════════════════════════════════════════════════════
--
-- `logprob` and `entropy` perform a numerically-stable softmax pass
-- over the logit vector. The combined `logprob_entropy` does both in
-- a single sweep — half the work when both are needed.

T.suite("logprob / entropy — numerical softmax helpers")

T.test("logprob is a finite non-positive number for an in-range token", function()
    -- Token 0 is always in vocabulary range. log-probabilities of a
    -- proper distribution are <= 0 (probabilities are <= 1).
    local lp = ctx:logprob(-1, 0)
    T.is_type(lp, "number")
    T.gte(0, lp,    "log probability must be <= 0")
    T.gt(lp, -1e9,  "log probability must be finite")
end)

T.test("logprob is -math.huge for an out-of-vocab token", function()
    T.eq(ctx:logprob(-1, n_vocab + 100), -math.huge)
    T.eq(ctx:logprob(-1, -1),            -math.huge)
end)

T.test("entropy is a non-negative finite number", function()
    local h = ctx:entropy(-1)
    T.is_type(h, "number")
    T.gte(h, 0, "entropy is always >= 0")
    -- Upper bound : log(n_vocab) is the entropy of a uniform distribution.
    T.gt(math.log(n_vocab) + 1, h, "entropy <= log(n_vocab)")
end)

T.test("logprob_entropy returns the same numbers as the two single calls",
    function()
        local lp_solo = ctx:logprob(-1, 0)
        local h_solo  = ctx:entropy(-1)
        local lp, h   = ctx:logprob_entropy(-1, 0)
        T.near(lp, lp_solo, 1e-6, "logprob")
        T.near(h,  h_solo,  1e-6, "entropy")
    end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Threadpool attach / detach
-- ════════════════════════════════════════════════════════════════════════
--
-- Live attach / detach round-trip. We do not run a decode under the
-- attached pool here ; the threadpool itself has its own dedicated
-- file (`18_threadpool.lua`).

T.suite("Threadpool — attach and detach")

T.test("attach_threadpool / detach_threadpool", function()
    local tp = ion7.Threadpool.new(2)
    T.no_error(function() ctx:attach_threadpool(tp) end)
    T.no_error(function() ctx:detach_threadpool() end)
    tp:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — Control vector
-- ════════════════════════════════════════════════════════════════════════
--
-- A control vector steers the residual stream by adding a small bias
-- to every layer's hidden state. Setting one with all zeros is a
-- functional no-op but exercises the full plumbing path.

T.suite("Control vector — set / clear")

T.test("set_control_vector(zero vector) returns true", function()
    local n_embd = model:n_embd()
    local zeros = {}
    -- One layer's worth of zeros is enough to test the FFI shape.
    for i = 1, n_embd do zeros[i] = 0.0 end
    T.eq(ctx:set_control_vector(zeros, n_embd, 0, 0), true)
end)

T.test("clear_control_vector does not raise", function()
    T.no_error(function() ctx:clear_control_vector() end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 5 — Embeddings (gated on an embedding context)
-- ════════════════════════════════════════════════════════════════════════
--
-- A generation context does NOT produce a per-sequence embedding
-- vector ; that requires `model:embedding_context(...)`. We build a
-- second short-lived embedding context just for this suite, decode a
-- single sentence, and confirm we can read out a vector of the right
-- dimension.

T.suite("Embeddings — embedding_context flow")

T.test("embedding_context produces an n_embd-sized vector", function()
    local ec = model:embedding_context({ n_ctx = 256, pooling = "mean" })
    local etk, en = vocab:tokenize("This is a sentence.", true, true)
    ec:decode(etk, en)

    -- Pooled embedding as a Lua table (1-based, dim == n_embd).
    local v = ec:embedding(0, model:n_embd())
    T.is_type(v, "table")
    T.eq(#v, model:n_embd(), "vector length should equal n_embd")
    -- The vector must contain at least one non-zero coefficient.
    local any_nonzero = false
    for _, x in ipairs(v) do
        if x ~= 0 then any_nonzero = true; break end
    end
    T.ok(any_nonzero, "embedding should not be all zeros")
    ec:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
