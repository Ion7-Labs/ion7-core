#!/usr/bin/env luajit
--- @module tests.12_context
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 12 — `Context` : creation, properties, decode, encode, warmup, perf
--- ════════════════════════════════════════════════════════════════════════
---
--- A `Context` is the inference handle. It owns the KV cache and a
--- pre-allocated decode batch ; every token your generator produces
--- transits through it. This file covers the lifecycle and the four
--- decode entry points :
---
---   1. Construction      : `Model:context(opts)` and option parsing.
---   2. Cached properties : `n_ctx`, `n_batch`, `n_seq_max`, `n_threads`.
---   3. Live properties   : `pooling_type`, `n_threads`, etc.
---   4. Decode variants   : `decode_single`, `decode`, `decode_multi`.
---   5. Encode pass       : `encode` (skipped on decoder-only models).
---   6. Warmup            : `warmup()` JIT-pre-compiles GPU shaders.
---   7. Perf counters     : `perf()`, `perf_print()`, `perf_reset()`.
---
--- KV cache mutations, state persistence and logits/embedding readers
--- live in their own files (13_kv, 14_state, 15_logits) to keep this
--- one focused on the decode lifecycle.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/12_context.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Construction & cached properties
-- ════════════════════════════════════════════════════════════════════════
--
-- `Model:context(opts)` creates a `llama_context*` and wraps it in a
-- Lua handle. The constructor caches a few immutable shape numbers
-- (`n_ctx`, `n_batch`, ...) so the inner generation loop never pays a
-- per-call FFI roundtrip just to read its own dimensions.

T.suite("Construction & cached properties")

T.test("model:context() returns a Context with positive dimensions", function()
    local ctx = model:context({ n_ctx = 1024, n_threads = 2 })
    T.is_type(ctx,         "table")
    T.is_type(ctx:ptr(),   "cdata")
    T.gte(ctx:n_ctx(),     1024)
    T.gt(ctx:n_batch(),    0)
    T.gt(ctx:n_ubatch(),   0)
    T.gte(ctx:n_seq_max(), 1)
    ctx:free()
end)

T.test("kv_type='q8_0' opts in a quantised KV cache (auto-flash)", function()
    -- Quantised KV requires flash attention to dequant on the fly. The
    -- factory forces flash on automatically when the user picks a
    -- quantised type ; the call must therefore succeed even if the
    -- caller does not pass `flash_attn = true`.
    local ctx = model:context({ n_ctx = 1024, kv_type = "q8_0" })
    T.gte(ctx:n_ctx(), 1024)
    ctx:free()
end)

T.test("kv_type_k and kv_type_v override independently", function()
    local ctx = model:context({
        n_ctx     = 1024,
        kv_type_k = "q8_0",
        kv_type_v = "f16",
    })
    T.gte(ctx:n_ctx(), 1024)
    ctx:free()
end)

T.test("invalid kv_type is rejected with an explicit message", function()
    T.err(function()
        model:context({ n_ctx = 512, kv_type = "not_a_real_quant" })
    end, "unknown kv_type")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Live properties
-- ════════════════════════════════════════════════════════════════════════
--
-- A few properties are queried live from llama.cpp on each call —
-- they may have been changed by the user via setter methods.

T.suite("Live properties")

local ctx = model:context({ n_ctx = 1024, n_threads = 2 })

T.test("pooling_type returns a documented name", function()
    -- A generation context with no embedding pooling reports
    -- "unspecified" or "none" depending on llama.cpp version. We accept
    -- the entire documented set rather than pin a specific one.
    T.one_of(ctx:pooling_type(),
             { "unspecified", "none", "mean", "cls", "last", "rank" })
end)

T.test("set_n_threads + n_threads round-trip", function()
    ctx:set_n_threads(3, 5)
    T.eq(ctx:n_threads(),       3)
    T.eq(ctx:n_threads_batch(), 5)
    -- Restore so subsequent decode tests run on a sensible thread count.
    ctx:set_n_threads(2, 2)
end)

T.test("set_embeddings / set_causal_attn / set_warmup do not raise", function()
    -- We are not decoding here, so the side effects are invisible — we
    -- are just confirming the FFI plumbing for these toggles works.
    T.no_error(function() ctx:set_embeddings(false)  end)
    T.no_error(function() ctx:set_causal_attn(true)  end)
    T.no_error(function() ctx:set_warmup(false)      end)
end)

T.test("synchronize blocks cleanly with no work queued", function()
    T.no_error(function() ctx:synchronize() end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Decode variants
-- ════════════════════════════════════════════════════════════════════════
--
-- `decode` is the bread-and-butter entry point. It accepts both Lua
-- tables (the typical shape of `vocab:tokenize` output) and raw cdata
-- int32 arrays. It chunks automatically when the input exceeds
-- `n_batch`. Logits are produced for the LAST token of the LAST chunk.
--
-- `decode_single` is the per-token fast path used inside the
-- generation loop ; `decode_multi` enables logits for EVERY position
-- and is what speculative decoding uses to verify multiple drafts in
-- one forward pass.

T.suite("Decode variants")

T.test("decode of a Lua-table prompt advances n_past", function()
    ctx:kv_clear() -- start clean
    local toks, n = vocab:tokenize("Hello, world.", true, true)

    -- The cdata-array shape works too ; we use the table form here for
    -- the tutorial value of showing the simplest call site.
    local table_toks = {}
    for i = 0, n - 1 do table_toks[i + 1] = toks[i] end

    local last = ctx:decode(table_toks)
    T.gt(last, 0,            "decode returns last-chunk size")
    T.eq(ctx:n_past(), n,    "n_past advanced by the prompt length")
end)

T.test("decode_single appends one token", function()
    local before = ctx:n_past()
    -- BOS is always a safe single-token to push.
    local bos = vocab:bos()
    if bos < 0 then bos = 0 end
    ctx:decode_single(bos)
    T.eq(ctx:n_past(), before + 1)
end)

T.test("decode of a cdata int32 array works identically", function()
    ctx:kv_clear()
    local toks, n = vocab:tokenize("Test", false, true)
    local last = ctx:decode(toks, n)
    T.gt(last, 0)
    T.eq(ctx:n_past(), n)
end)

T.test("decode_multi enables logits at every position", function()
    ctx:kv_clear()
    local toks, n = vocab:tokenize("abc", false, true)
    -- decode_multi takes a 1-based table.
    local table_toks = {}
    for i = 0, n - 1 do table_toks[i + 1] = toks[i] end
    -- Cap at n_batch so the assertion inside decode_multi does not fire.
    if #table_toks > ctx:n_batch() then
        for i = ctx:n_batch() + 1, #table_toks do table_toks[i] = nil end
    end
    T.no_error(function() ctx:decode_multi(table_toks) end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — Encode (T5-style models only)
-- ════════════════════════════════════════════════════════════════════════
--
-- `encode` runs the model's encoder stack. Only meaningful for
-- encoder-decoder architectures (T5, BART, ...). On a decoder-only
-- LLM the call is invalid — we skip the suite then.

T.suite("Encode pass")

if model:has_encoder() then
    T.test("encode produces no error on a small prompt", function()
        ctx:kv_clear()
        local toks, n = vocab:tokenize("source", false, true)
        T.no_error(function() ctx:encode(toks, n) end)
    end)
else
    T.skip("encode", "model has no encoder stack — decoder-only family")
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 5 — Warmup & perf counters
-- ════════════════════════════════════════════════════════════════════════
--
-- `warmup` runs a single dummy decode so the GPU backend can JIT-
-- compile its shaders. It clears the KV cache afterwards, leaving the
-- context in the same state as right after construction.

T.suite("Warmup & perf counters")

T.test("warmup completes and resets n_past", function()
    ctx:kv_clear()
    ctx:decode_single(vocab:bos() >= 0 and vocab:bos() or 0)
    T.gt(ctx:n_past(), 0, "preliminary decode advanced n_past")
    ctx:warmup()
    T.eq(ctx:n_past(), 0, "warmup clears the KV mirror")
end)

T.test("perf returns a populated table after a decode", function()
    ctx:kv_clear()
    ctx:perf_reset()
    local toks, n = vocab:tokenize("perf bench", false, true)
    ctx:decode(toks, n)
    local p = ctx:perf()
    T.is_type(p,          "table")
    T.is_type(p.t_load_ms, "number")
    T.is_type(p.t_p_eval_ms, "number")
    T.is_type(p.t_eval_ms, "number")
    T.is_type(p.n_p_eval, "number")
    T.is_type(p.n_eval,   "number")
    T.is_type(p.tokens_per_s, "number")
end)

T.test("perf_reset does not raise and returns numeric counters", function()
    -- llama.cpp's reset does not guarantee an exact-zero state for
    -- every counter (some are bookkeeping internals it keeps to render
    -- the next print sensibly). We only assert the call is wired
    -- correctly and the counters remain numeric afterwards.
    T.no_error(function() ctx:perf_reset() end)
    local p = ctx:perf()
    T.is_type(p.n_p_eval, "number")
    T.is_type(p.n_eval,   "number")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 6 — Lifecycle
-- ════════════════════════════════════════════════════════════════════════

T.suite("Lifecycle — explicit free is idempotent")

T.test("free() then free() is safe", function()
    local c = model:context({ n_ctx = 256 })
    T.no_error(function() c:free() end)
    T.no_error(function() c:free() end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ctx:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
