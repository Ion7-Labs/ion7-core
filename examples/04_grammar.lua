#!/usr/bin/env luajit
--- 04_grammar.lua - Constrained generation with GBNF grammars.
---
--- GBNF (GGML BNF) constrains the model to only produce tokens that match
--- a formal grammar. Useful for structured output: JSON, enums, code, etc.
--- The model never generates invalid output - it's enforced at the logit level.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/04_grammar.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)
local model  = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab  = model:vocab()
local ctx    = model:context({ n_ctx = 1024 })

--- Run a grammar-constrained generation and return the output.
local function generate(prompt, gbnf, root, max_tokens)
    local sampler = ion7.Sampler.chain()
        :grammar(gbnf, root or "root", vocab)
        :temperature(0.0)   -- deterministic within grammar constraint
        :dist(42)
        :build(vocab)

    local msgs   = { { role = "user", content = prompt } }
    local fmt    = vocab:apply_template(msgs, true)
    local toks, n = vocab:tokenize(fmt, false, true)

    ctx:kv_clear()
    ctx:decode(toks, n, 0, 0)
    sampler:reset()

    local parts = {}
    for _ = 1, max_tokens or 128 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        parts[#parts+1] = vocab:piece(tok, true)
        ctx:decode_single(tok, 0)
    end

    sampler:free()
    return table.concat(parts):match("^%s*(.-)%s*$")  -- trim
end

-- ── 1. Sentiment classification ───────────────────────────────────────────────

local SENTIMENT_GBNF = 'root ::= "positive" | "negative" | "neutral"'

local reviews = {
    "This product is absolutely amazing, best purchase ever!",
    "Terrible quality, broke after one day. Very disappointed.",
    "It's okay, nothing special but gets the job done.",
}

io.write("\n[1. Sentiment classification - grammar: positive | negative | neutral]\n")
for _, review in ipairs(reviews) do
    local sentiment = generate(
        string.format('Review: "%s"\nSentiment:', review),
        SENTIMENT_GBNF
    )
    io.write(string.format("  %-60s → %s\n", review:sub(1, 58) .. "..", sentiment))
end

-- ── 2. Structured JSON output ─────────────────────────────────────────────────

-- Grammar that produces a JSON object with name, age, and city fields
local JSON_PERSON_GBNF = [[
root   ::= "{" ws "\"name\":" ws string "," ws "\"age\":" ws number "," ws "\"city\":" ws string "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
]]

io.write("\n[2. Structured JSON output]\n")
local json_out = generate(
    "Extract the person info as JSON: John Smith is 34 years old and lives in Paris.",
    JSON_PERSON_GBNF, "root", 128
)
io.write("  Output: " .. json_out .. "\n")

-- ── 3. Yes/No with explanation using grammar_lazy ────────────────────────────
-- grammar_lazy activates only when a trigger word appears in the output.
-- The model writes freely until it says "Answer:" then must respond yes/no.

local YESNO_GBNF = 'root ::= "yes" | "no"'

io.write("\n[3. Yes/No forced answer]\n")
local sampler_lazy = ion7.Sampler.chain()
    :grammar_lazy(YESNO_GBNF, "root", vocab,
        { "Answer:" },   -- trigger words
        {},              -- trigger tokens
        {}               -- trigger patterns
    )
    :temperature(0.3)
    :dist(42)
    :build(vocab)

local msgs  = {{ role="user", content="Is the sky blue? Think briefly, then say Answer: yes or no." }}
local fmt   = vocab:apply_template(msgs, true)
local toks, n = vocab:tokenize(fmt, false, true)
ctx:kv_clear()
ctx:decode(toks, n, 0, 0)
sampler_lazy:reset()

io.write("  Response: ")
for _ = 1, 128 do
    local tok = sampler_lazy:sample(ctx:ptr(), -1)
    if vocab:is_eog(tok) then break end
    io.write(vocab:piece(tok, true)); io.flush()
    ctx:decode_single(tok, 0)
end
io.write("\n")
sampler_lazy:free()

ion7.shutdown()
