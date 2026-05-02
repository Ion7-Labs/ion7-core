#!/usr/bin/env luajit
--- @module tests.14_state
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 14 — Context state persistence : snapshot, restore, save, load
--- ════════════════════════════════════════════════════════════════════════
---
--- The KV cache is the most expensive thing a generator builds. A
--- well-warmed context (long system prompt, many tool calls) takes
--- seconds to recompute from scratch ; serialising it after the build
--- and restoring later is essentially free.
---
--- Two granularities × two storage backends :
---
---                  | whole context      | per-sequence
---   ───────────────┼─────────────────────────────────────────────
---   file-backed    | save_state         | seq_save_state
---                  | load_state         | seq_load_state
---   ───────────────┼─────────────────────────────────────────────
---   Lua-string     | snapshot           | seq_snapshot
---                  | restore            | seq_restore
---
--- We exercise the round-trip for both backends and confirm the
--- restored context can be sampled from like the original.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/14_state.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()
local n_vocab = vocab:n_vocab()

-- Use two sibling contexts so we can snapshot one and restore into the
-- other — the crispest demonstration of "transferable state".
local ctx_src = model:context({ n_ctx = 1024, n_threads = 2 })
local ctx_dst = model:context({ n_ctx = 1024, n_threads = 2 })

-- Decode a short prompt into the source so the cache holds something
-- non-trivial to round-trip.
local toks, n = vocab:tokenize("The quick brown fox", true, true)
ctx_src:decode(toks, n)

local sampler = ion7.Sampler.greedy()

-- A fixed token we decode in both contexts after a restore so each
-- side regenerates its logits on identical state. Logits are NOT part
-- of the state blob — they are recomputed per decode — so we always
-- need at least one decode call between `restore` and `sample`.
local probe_token = vocab:eot()
if probe_token < 0 then probe_token = vocab:eos() end
if probe_token < 0 then probe_token = 0 end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Lua-string round-trip (whole context)
-- ════════════════════════════════════════════════════════════════════════
--
-- `snapshot()` returns the entire context state as a Lua string. The
-- shape is opaque (binary blob) but Lua strings are byte-safe so it
-- can be stashed in Redis, sent across the network, or held in
-- memory. The blob carries the KV cache and the RNG seed but NOT the
-- logits buffer — that gets recomputed on the next decode.

T.suite("Whole-context Lua-string round-trip")

local blob

T.test("snapshot returns a non-empty Lua string", function()
    blob = ctx_src:snapshot()
    T.is_type(blob, "string")
    T.gt(#blob, 0)
end)

T.test("restore() loads the blob into a fresh context", function()
    -- The destination starts empty. After restore the cache rows
    -- should match the source ; we realign the Lua mirror manually
    -- because the blob does not carry it.
    T.eq(ctx_dst:restore(blob), true)
    ctx_dst:set_n_past(ctx_src:n_past())
    T.eq(ctx_dst:n_past(), ctx_src:n_past())
end)

T.test("restored context produces the same greedy continuation", function()
    -- Decode the SAME probe token in both contexts so each one
    -- regenerates its logits buffer on identical KV state. Greedy
    -- sampling is deterministic, so the resulting token must match.
    ctx_src:decode_single(probe_token)
    ctx_dst:decode_single(probe_token)

    local from_src = sampler:sample(ctx_src:ptr(), -1)
    local from_dst = sampler:sample(ctx_dst:ptr(), -1)
    T.gte(from_src, 0)
    T.gt(n_vocab, from_src)
    T.eq(from_dst, from_src,
         "restored state + same decode must produce the same greedy token")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — File round-trip (whole context)
-- ════════════════════════════════════════════════════════════════════════

T.suite("Whole-context file round-trip")

local state_path = H.tmpfile("ion7-state-" .. os.time() .. ".bin")

T.test("save_state writes a non-empty file", function()
    T.eq(ctx_src:save_state(state_path), true)
    local f = io.open(state_path, "rb")
    T.ok(f ~= nil, "file should exist after save_state")
    f:seek("end")
    T.gt(f:seek(), 0, "file should be non-empty")
    f:close()
end)

T.test("load_state populates a fresh context", function()
    -- Reset the destination so we observe the load effect cleanly.
    ctx_dst:kv_clear()
    local ok, n_restored = ctx_dst:load_state(state_path)
    T.eq(ok, true)
    T.is_type(n_restored, "number")

    -- Realign the Lua mirror from the live KV bound : `load_state`
    -- only auto-fills `_n_past` when a token_buf was provided.
    local last = ctx_dst:kv_seq_pos_max(0)
    ctx_dst:set_n_past(last + 1)

    -- A subsequent decode + greedy sample must produce a valid token id.
    ctx_dst:decode_single(probe_token)
    local tok = sampler:sample(ctx_dst:ptr(), -1)
    T.gte(tok, 0)
    T.gt(n_vocab, tok)
end)

H.try_remove(state_path)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Per-sequence round-trip
-- ════════════════════════════════════════════════════════════════════════
--
-- The per-seq variants serialise just the rows belonging to one
-- sequence id. Useful for micro-sharding a parallel-decoding setup
-- across multiple worker processes.

T.suite("Per-sequence Lua-string round-trip")

T.test("seq_state_size reports a positive byte count", function()
    local sz = ctx_src:seq_state_size(0)
    T.is_type(sz, "number")
    T.gt(sz, 0)
end)

T.test("seq_snapshot / seq_restore round-trip a single sequence", function()
    local seq_blob = ctx_src:seq_snapshot(0)
    T.is_type(seq_blob, "string")
    T.gt(#seq_blob, 0)

    -- Restore into seq 0 of a freshly cleared destination.
    ctx_dst:kv_clear()
    local consumed = ctx_dst:seq_restore(seq_blob, 0)
    T.gt(consumed, 0, "seq_restore returns bytes consumed")

    -- Realign the Lua mirror from the live KV bound, then verify the
    -- restored cache supports a fresh decode + sample without raising.
    local last = ctx_dst:kv_seq_pos_max(0)
    ctx_dst:set_n_past(last + 1)
    ctx_dst:decode_single(probe_token)
    local tok = sampler:sample(ctx_dst:ptr(), -1)
    T.gte(tok, 0)
    T.gt(n_vocab, tok)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

sampler:free()
ctx_src:free()
ctx_dst:free()
model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
