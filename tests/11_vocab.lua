#!/usr/bin/env luajit
--- @module tests.11_vocab
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 11 — `Vocab` : tokenisation, detokenisation, special tokens, templates
--- ════════════════════════════════════════════════════════════════════════
---
--- The `Vocab` is the second handle a downstream caller touches after
--- `Model`. Almost every interaction with an LLM goes through it :
---
---   - `tokenize(text)`     : UTF-8 string → int32 token array.
---   - `detokenize(tokens)` : int32 token array → UTF-8 string.
---   - `piece(token)`       : single-token text view (memoised).
---   - `bos / eos / eot ...`: special-token IDs the prompt builder needs.
---   - `apply_template`     : Jinja2 chat template via the libcommon bridge.
---
--- Tokenise/detokenise round-trip is the foundational invariant. We
--- exercise it on a few representative inputs (ASCII, accented Latin,
--- CJK, emoji) plus a longer paragraph that probably exceeds the
--- pre-allocated scratch buffer and exercises the auto-grow path.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/11_vocab.lua

local T = require "tests.framework"
local H = require "tests.helpers"

local ion7, model = H.boot(T)

-- We share one `Vocab` across the whole file. The Model caches the
-- handle ; calling :vocab() again would return the same table.
local vocab = model:vocab()

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Identity
-- ════════════════════════════════════════════════════════════════════════

T.suite("Identity")

T.test("n_vocab is positive", function()
    T.gt(vocab:n_vocab(), 0)
    -- `n_tokens` is an alias of `n_vocab`. The two must agree on the
    -- exact same instance.
    T.eq(vocab:n_tokens(), vocab:n_vocab())
end)

T.test("type() returns one of the documented kinds", function()
    T.one_of(vocab:type(),
             { "none", "spm", "bpe", "wpm", "ugm", "rwkv", "unknown" })
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Tokenise / detokenise round-trip
-- ════════════════════════════════════════════════════════════════════════
--
-- The contract :
--   tokenize(text, false, true)   tokens that DO NOT include BOS/EOS.
--   detokenize(tokens, n)         text reconstruction.
--
-- For most modern tokenisers (SPM, BPE) the round-trip is byte-exact
-- on UTF-8 input. We accept either an exact match OR a leading-space
-- difference (BPE often leaves a leading whitespace token at the
-- start of the encoded sequence).

T.suite("Tokenise / detokenise round-trip")

local function roundtrip(label, text)
    T.test(label, function()
        local toks, n = vocab:tokenize(text, false, true)
        T.gt(n, 0,                      "should produce at least one token")
        T.is_type(toks, "cdata",        "tokens are returned as a cdata array")

        local back = vocab:detokenize(toks, n)
        -- Allow leading whitespace difference but the rest must match.
        local stripped = back:gsub("^%s+", "")
        T.ok(stripped == text or back == text,
             string.format("round-trip differs : got %q, expected %q",
                           back, text))
    end)
end

roundtrip("ASCII",                 "Hello, world.")
roundtrip("accented Latin",        "déjà vu — café")
roundtrip("CJK",                   "你好世界")
roundtrip("emoji",                 "ship it 🚀 now")
roundtrip("long paragraph (forces auto-grow of the detok buffer)",
    string.rep(
        "The quick brown fox jumps over the lazy dog. ", 200))

T.test("add_special prepends BOS when requested", function()
    -- `add_special = true` asks the vocab to add BOS / EOS as
    -- configured in the GGUF. Most chat models add BOS but not EOS.
    local with,    n_with    = vocab:tokenize("hi", true,  false)
    local without, n_without = vocab:tokenize("hi", false, false)
    T.gte(n_with, n_without,
        "add_special should not produce fewer tokens than without")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Per-token text views (`piece`)
-- ════════════════════════════════════════════════════════════════════════
--
-- `piece(token)` returns the displayable text of a single token.
-- It is the per-token output of a streaming generator, so the hot path
-- of inference. The result is memoised — second calls for the same
-- token id return a cached Lua string without crossing FFI.

T.suite("Per-token piece accessors")

T.test("piece returns a string for every reachable token", function()
    local toks, n = vocab:tokenize("Hello", false, true)
    for i = 0, n - 1 do
        local p = vocab:piece(toks[i])
        T.is_type(p, "string")
    end
end)

T.test("piece is memoised (same identity on repeat call)", function()
    local toks = vocab:tokenize("hello", false, true)
    local first  = vocab:piece(toks[0])
    local second = vocab:piece(toks[0])
    -- Lua interns identical short strings, but the cache also keeps a
    -- single concrete reference ; either invariant satisfies us.
    T.eq(first, second)
end)

T.test("text(token) and piece(token) coexist sensibly", function()
    local toks = vocab:tokenize("a", false, true)
    -- text() is a raw view ; piece() applies SPM whitespace normalisation.
    -- They may differ on leading-space details, but neither should crash.
    T.is_type(vocab:text(toks[0]),  "string")
    T.is_type(vocab:piece(toks[0]), "string")
end)

T.test("score and attr return numbers", function()
    local toks = vocab:tokenize("a", false, true)
    T.is_type(vocab:score(toks[0]), "number")
    T.is_type(vocab:attr(toks[0]),  "number")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — Special tokens
-- ════════════════════════════════════════════════════════════════════════
--
-- Every model declares a handful of special tokens : BOS, EOS, EOT,
-- newline, padding, etc. The accessor returns the token id, or `-1`
-- when the model has no such token. We assert each is an integer in
-- `[-1, n_vocab)` — that is the entire valid range.

T.suite("Special-token accessors")

local function valid_id(id, name)
    T.is_type(id, "number", name .. " : not a number")
    T.gte(id, -1, name .. " : id must be >= -1 (sentinel)")
    T.gt(vocab:n_vocab(), id,
         name .. " : id must be < n_vocab")
end

T.test("special-token IDs are integers in [-1, n_vocab)", function()
    valid_id(vocab:bos(),     "bos")
    valid_id(vocab:eos(),     "eos")
    valid_id(vocab:eot(),     "eot")
    valid_id(vocab:nl(),      "nl")
    valid_id(vocab:pad(),     "pad")
    valid_id(vocab:sep(),     "sep")
    valid_id(vocab:mask(),    "mask")
    valid_id(vocab:fim_pre(), "fim_pre")
    valid_id(vocab:fim_suf(), "fim_suf")
    valid_id(vocab:fim_mid(), "fim_mid")
    valid_id(vocab:fim_pad(), "fim_pad")
    valid_id(vocab:fim_rep(), "fim_rep")
    valid_id(vocab:fim_sep(), "fim_sep")
end)

T.test("EOS / EOT (when present) are flagged as end-of-generation", function()
    local eos = vocab:eos()
    if eos >= 0 then T.ok(vocab:is_eog(eos), "EOS should be EOG") end
    local eot = vocab:eot()
    if eot >= 0 then T.ok(vocab:is_eog(eot), "EOT should be EOG") end
end)

T.test("auto-prepend / append flags return booleans", function()
    T.is_type(vocab:get_add_bos(), "boolean")
    T.is_type(vocab:get_add_eos(), "boolean")
    T.is_type(vocab:get_add_sep(), "boolean")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 5 — Built-in chat templates
-- ════════════════════════════════════════════════════════════════════════

T.suite("Built-in chat templates")

T.test("builtin_templates returns a non-empty list of strings", function()
    local names = vocab:builtin_templates()
    T.is_type(names, "table")
    T.gt(#names, 0)
    for _, name in ipairs(names) do
        T.is_type(name, "string")
        T.gt(#name, 0)
    end
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 6 — Jinja2 chat template (libcommon bridge)
-- ════════════════════════════════════════════════════════════════════════
--
-- `apply_template` formats a message list according to the model's
-- own embedded Jinja2 template. The output is the prompt string the
-- generator should tokenise. We test it ONLY when the model embeds a
-- template (instruct-tuned models do, base models often do not).

T.suite("Jinja2 chat template")

if model:chat_template(nil) then

    T.test("apply_template produces a non-empty prompt", function()
        local prompt = vocab:apply_template({
            { role = "user", content = "Hello!" },
        }, true, -1)
        T.is_type(prompt, "string")
        T.gt(#prompt, 0)
    end)

    T.test("apply_template grows past the scratch buffer (>128 KB) cleanly",
        function()
            -- Force the auto-grow path by filling a big content string.
            -- 130 000 bytes is just over the 128 KiB scratch.
            local big = string.rep("x ", 65000) -- 130 000 bytes
            local prompt = vocab:apply_template({
                { role = "user", content = big },
            }, true, -1)
            T.gt(#prompt, 130000)
        end)

else
    T.skip("apply_template",
           "model has no embedded chat template (likely a base model)")
end

T.test("supports_thinking returns a boolean", function()
    T.is_type(vocab:supports_thinking(), "boolean")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
