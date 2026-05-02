#!/usr/bin/env luajit
--- @module tests.20_chat_templates
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 20 — Chat templates : apply, parse, reasoning-budget sampler
--- ════════════════════════════════════════════════════════════════════════
---
--- Three pieces of the chat pipeline that route through the libcommon
--- bridge :
---
---   1. `Vocab:apply_template`                — message list → prompt string.
---      (basic round-trip already covered in 11_vocab ; here we test the
---      thinking-mode toggle).
---   2. `bridge.ion7_chat_parse`              — generated text → structured
---      `{ content, thinking, tools }` triple. Used by chat front-ends
---      to split out `<think>...</think>` blocks and tool-call payloads.
---   3. The reasoning-budget sampler          — caps the number of tokens
---      a model emits inside a `<think>...</think>` block. Inserted via
---      the SamplerBuilder's `:reasoning_budget(model, n)` step.
---
--- Skipped automatically when the model has no embedded chat template.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/20_chat_templates.lua

local ffi = require "ffi"
local T   = require "tests.framework"
local H   = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab = model:vocab()

if not model:chat_template(nil) then
    T.skip("(this whole file)",
           "model has no embedded chat template — likely a base model")
    T.summary()
    os.exit(0)
end

local bridge = require "ion7.core.ffi.bridge"

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — apply_template thinking toggle
-- ════════════════════════════════════════════════════════════════════════

T.suite("apply_template — thinking toggle")

T.test("supports_thinking returns a boolean", function()
    T.is_type(vocab:supports_thinking(), "boolean")
end)

T.test("apply_template with thinking=-1 (default), 0 (off), 1 (on)", function()
    -- The exact prompt strings depend on the model's Jinja template.
    -- We only assert each call returns a non-empty string ; specific
    -- output shape is the model's responsibility.
    local msgs = { { role = "user", content = "Hello" } }
    for _, mode in ipairs({ -1, 0, 1 }) do
        local p = vocab:apply_template(msgs, true, mode)
        T.gt(#p, 0, "thinking=" .. mode .. " should produce a prompt")
    end
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Chat parse (bridge)
-- ════════════════════════════════════════════════════════════════════════
--
-- `ion7_chat_parse` takes a model-generated string and decomposes it
-- into a `(content, thinking, tools)` triple plus a `has_tools` flag.
-- The bridge expects three caller-owned buffers + a single int*. We
-- thread them in via `ffi.new` here rather than wrap the call — the
-- raw FFI shape is what a Lua-level helper would call internally.

T.suite("Chat output parsing (ion7_chat_parse)")

local templates = vocab._tmpls -- internal handle ; OK to read in tests

T.test("parses a plain assistant reply (no thinking, no tools)", function()
    local content_buf  = ffi.new("char[?]", 4096)
    local thinking_buf = ffi.new("char[?]", 256)
    local tools_buf    = ffi.new("char[?]", 256)
    local has_tools    = ffi.new("int[1]", 0)

    local rc = bridge.ion7_chat_parse(
        templates,
        "The capital of France is Paris.",
        0,                          -- enable_thinking = off
        content_buf,  4096,
        thinking_buf, 256,
        tools_buf,    256,
        has_tools)

    T.gte(rc, 0, "ion7_chat_parse should return >= 0 on success")
    -- The plain-prose content must round-trip into content_buf.
    local content = ffi.string(content_buf)
    T.gt(#content, 0)
    T.contains(content, "Paris")
    -- No thinking block, no tools.
    T.eq(has_tools[0], 0)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Reasoning-budget sampler
-- ════════════════════════════════════════════════════════════════════════
--
-- `:reasoning_budget(model, n)` inserts a libcommon-backed sampler
-- that limits the number of tokens a model can emit inside a
-- `<think>...</think>` block. We don't actually drive a long
-- generation here — the goal is to confirm the builder accepts the
-- step and that `:sample` returns a valid token id from a chain
-- where reasoning_budget is the first step.

T.suite("Reasoning-budget sampler step")

T.test("builder accepts :reasoning_budget(model, n) as the first step",
    function()
        local ctx = model:context({ n_ctx = 256, n_threads = 2 })
        local toks, n = vocab:tokenize("Hello", true, true)
        ctx:decode(toks, n)

        local s = ion7.Sampler.chain()
            :reasoning_budget(model, 64)
            :greedy()
            :build()

        local tok = s:sample(ctx:ptr(), -1)
        T.gte(tok, 0)
        T.gt(vocab:n_vocab(), tok)

        s:free()
        ctx:free()
    end)

T.test("reasoning_budget accepts a Model instance OR a raw cdata", function()
    -- Both `model` (table) and `model:ptr()` (cdata) must work — the
    -- builder normalises internally.
    local ctx = model:context({ n_ctx = 256, n_threads = 2 })
    local toks, n = vocab:tokenize("Hi", true, true)
    ctx:decode(toks, n)

    local s = ion7.Sampler.chain()
        :reasoning_budget(model:ptr(), 32)
        :greedy()
        :build()

    T.gte(s:sample(ctx:ptr(), -1), 0)
    s:free()
    ctx:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
