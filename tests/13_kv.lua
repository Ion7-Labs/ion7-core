#!/usr/bin/env luajit
--- @module tests.13_kv
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 13 — KV cache : window queries and mutations
--- ════════════════════════════════════════════════════════════════════════
---
--- The KV cache is the working memory of a transformer : every decoded
--- token leaves a key/value pair behind that future tokens attend to.
--- Direct manipulation is what enables :
---
---   - prompt-prefix caching   (`kv_clear` + decode prompt + snapshot),
---   - parallel sequences      (`kv_seq_cp` to fork, `kv_seq_keep` to prune),
---   - sliding window          (`kv_seq_shift` to drop old tokens),
---   - context compression     (`kv_seq_div` to halve positions).
---
--- The `(p0, p1)` window is `[inclusive, exclusive)`. `-1` for either
--- side means "from beginning" / "to end".
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/13_kv.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()
local ctx   = model:context({ n_ctx = 1024, n_threads = 2 })

-- Helper : decode a short prompt to seed the cache.
local function seed(seq_id, text)
    local toks, n = vocab:tokenize(text, false, true)
    ctx:decode(toks, n, seq_id or 0)
    return n
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Window queries
-- ════════════════════════════════════════════════════════════════════════

T.suite("Window queries — kv_seq_pos_min / max / can_shift")

T.test("an empty cache reports -1 for both bounds", function()
    ctx:kv_clear()
    T.eq(ctx:kv_seq_pos_min(0), -1)
    T.eq(ctx:kv_seq_pos_max(0), -1)
end)

T.test("after a decode, bounds report the live range", function()
    ctx:kv_clear()
    local n = seed(0, "Hello, world.")
    T.gte(ctx:kv_seq_pos_min(0),     0)
    T.gte(ctx:kv_seq_pos_max(0), n - 1)
end)

T.test("kv_can_shift returns a boolean (true for non-recurrent models)", function()
    -- Non-recurrent models support position shifting ; recurrent ones
    -- (Mamba, RWKV) return false. We accept either, only assert the type.
    T.is_type(ctx:kv_can_shift(), "boolean")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — kv_clear
-- ════════════════════════════════════════════════════════════════════════

T.suite("kv_clear — wipe everything")

T.test("kv_clear resets bounds AND the n_past mirror", function()
    seed(0, "some prompt")
    T.gt(ctx:n_past(), 0)
    ctx:kv_clear()
    T.eq(ctx:n_past(), 0)
    T.eq(ctx:kv_seq_pos_min(0), -1)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Per-sequence ops
-- ════════════════════════════════════════════════════════════════════════
--
-- Sequence IDs are independent rows in the cache. With `n_seq_max = 1`
-- we still have seq 0, but copy / fork operations need at least 2.

T.suite("Per-sequence ops — seq_rm / seq_cp / seq_keep")

T.test("kv_seq_rm shrinks the live range", function()
    ctx:kv_clear()
    local n = seed(0, "abcdef")
    -- Drop the trailing half of the cache.
    local cut = math.floor(n / 2)
    T.eq(ctx:kv_seq_rm(0, cut, -1), true)
    T.eq(ctx:kv_seq_pos_max(0), cut - 1)
end)

T.test("kv_seq_keep on a single-seq cache is a no-op (does not raise)",
    function()
        ctx:kv_clear()
        seed(0, "x")
        T.no_error(function() ctx:kv_seq_keep(0) end)
    end)

if ctx:n_seq_max() >= 2 then

    T.test("kv_seq_cp duplicates seq 0 onto seq 1", function()
        ctx:kv_clear()
        local n = seed(0, "fork me")
        ctx:kv_seq_cp(0, 1, 0, -1)
        -- Both sequences should now report the same upper bound.
        T.eq(ctx:kv_seq_pos_max(0), ctx:kv_seq_pos_max(1))
        -- Lengths agree with the source decode.
        T.eq(ctx:kv_seq_pos_max(1), n - 1)
    end)

else
    T.skip("kv_seq_cp",
           "context was created with n_seq_max=1 — sequence forking unavailable")
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — Position arithmetic (shift, div)
-- ════════════════════════════════════════════════════════════════════════
--
-- These two operations rewrite the *positions* attached to KV entries
-- without re-running the model. `shift` is sliding-window attention
-- bookkeeping ; `div` is the context-compression trick from the
-- "StreamingLLM" / "Self-Extend" literature.
--
-- Both are no-ops for recurrent / state-space architectures, and
-- `kv_can_shift()` is the gate that tells us whether to bother.

T.suite("Position arithmetic — shift / div")

if ctx:kv_can_shift() then
    T.test("kv_seq_shift survives a small forward slide", function()
        ctx:kv_clear()
        seed(0, "shiftable")
        T.no_error(function() ctx:kv_seq_shift(0, 2, 0, -1) end)
    end)

    T.test("kv_seq_div halves every position in a window", function()
        ctx:kv_clear()
        seed(0, "compress me")
        T.no_error(function() ctx:kv_seq_div(0, 2, 0, -1) end)
    end)
else
    T.skip("kv_seq_shift / kv_seq_div",
           "model does not support position shifting (recurrent / SSM)")
end

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
