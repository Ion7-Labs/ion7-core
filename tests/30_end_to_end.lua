#!/usr/bin/env luajit
--- @module tests.30_end_to_end
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 30 — End-to-end tutorial : prompt → tokens → decode → sample → stream
--- ════════════════════════════════════════════════════════════════════════
---
--- This is the integration tour. Every other file in the suite focuses
--- on one corner of the API ; this one stitches them together into a
--- minimal "complete a chat turn" pipeline so a reader can see how the
--- pieces compose :
---
---   1. Boot the backend and load the model.
---   2. Format a chat with the model's embedded Jinja2 template.
---   3. Tokenise the formatted prompt.
---   4. Build a sampler chain (top-p / temperature / dist).
---   5. Decode the prompt in one batch ; sample the first reply token.
---   6. Loop : decode_single + sample, with UTF-8-safe streaming output.
---   7. Stop on EOS / EOT or after `MAX_NEW_TOKENS`.
---   8. Read out a few perf counters at the end.
---
--- The assertions here are deliberately loose : we are not testing
--- model quality or specific token IDs (those drift with quantisation
--- and llama.cpp version). We assert that the pipeline RUNS, that
--- detokenisation produces UTF-8-clean output, and that perf reports
--- non-zero work was done.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/30_end_to_end.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)
local vocab   = model:vocab()
local n_vocab = vocab:n_vocab()

-- Cap the generation at a small number of tokens so the test stays
-- under a minute on CPU. Long enough to actually exercise the loop
-- (and to require at least one UTF-8 boundary check on most tokens).
local MAX_NEW_TOKENS = 24

-- ════════════════════════════════════════════════════════════════════════
-- Suite — end-to-end inference
-- ════════════════════════════════════════════════════════════════════════

T.suite("End-to-end : chat → tokens → decode → sample → stream")

T.test("full pipeline produces a non-empty UTF-8 reply", function()
    -- ── Step 1. Build the prompt ────────────────────────────────────────
    -- A model with an embedded chat template gets the proper Jinja2
    -- formatting ; a base model falls back to raw concatenation.
    local prompt
    if model:chat_template(nil) then
        prompt = vocab:apply_template({
            { role = "user", content = "Reply with exactly: hello world" },
        }, true, 0)
    else
        prompt = "User: Reply with exactly: hello world\nAssistant:"
    end
    T.gt(#prompt, 0)

    -- ── Step 2. Create the inference context ────────────────────────────
    local ctx = model:context({
        n_ctx     = 1024,
        n_threads = 4,
    })
    -- Warmup compiles GPU shaders / pre-touches KV memory ; the
    -- following decode does not pay that cost.
    ctx:warmup()

    -- ── Step 3. Tokenise + decode the prompt ────────────────────────────
    local toks, n = vocab:tokenize(prompt, true, true)
    T.gt(n, 0)
    ctx:decode(toks, n)
    T.eq(ctx:n_past(), n)

    -- ── Step 4. Build the sampler ───────────────────────────────────────
    -- Greedy makes the test deterministic. For a real chat front-end
    -- you would use `Sampler.default()` (top-p + min-p + temperature).
    local sampler = ion7.Sampler.greedy()

    -- ── Step 5. Generation loop ─────────────────────────────────────────
    -- We collect the generated token ids first, then detokenise once
    -- at the end. A streaming UI would call `vocab:piece(tok)` after
    -- every sample and emit the bytes through the UTF-8 buffer
    -- (`ion7.utf8.is_complete`) to never split a multi-byte codepoint.
    local generated = {}
    local stop_tokens = { [vocab:eos()] = true,
                          [vocab:eot()] = true }
    -- nil-protect : a model with no EOT reports -1 — the lookup
    -- against `stop_tokens` would always miss because -1 is its own
    -- valid Lua key. We tolerate either case.

    local stopped_naturally = false
    for _ = 1, MAX_NEW_TOKENS do
        local tok = sampler:sample(ctx:ptr(), -1)
        T.gte(tok, 0)
        T.gt(n_vocab, tok)
        if stop_tokens[tok] then
            stopped_naturally = true
            break
        end
        generated[#generated + 1] = tok
        ctx:decode_single(tok)
    end

    -- We do not assert on stopped_naturally — the model may or may
    -- not emit EOS within 24 tokens. Both are valid outcomes.

    -- ── Step 6. Detokenise the captured tokens ──────────────────────────
    -- Build a cdata int32 array for `vocab:detokenize`, which is the
    -- bulk path. In a streaming setup we'd already have emitted the
    -- pieces token-by-token via `vocab:piece`.
    local ffi = require "ffi"
    local arr = ffi.new("int32_t[?]", #generated)
    for i, t in ipairs(generated) do arr[i - 1] = t end
    local reply = vocab:detokenize(arr, #generated)
    T.is_type(reply, "string")
    T.gt(#reply, 0, "the model produced at least one byte of output")
    T.ok(ion7.utf8.is_complete(reply),
         "the streamed bytes form a complete UTF-8 sequence")

    -- ── Step 7. Perf counters ───────────────────────────────────────────
    local p = ctx:perf()
    T.gt(p.n_eval + p.n_p_eval, 0,
         "perf must show some evaluation work")

    sampler:free()
    ctx:free()
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
