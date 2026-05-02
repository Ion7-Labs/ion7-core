#!/usr/bin/env luajit
--- @example examples.04_grammar
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 04 — Grammar-constrained output (GBNF + JSON Schema)
--- ════════════════════════════════════════════════════════════════════════
---
--- Three flavours of constrained generation, in order of increasing
--- power :
---
---   1. Inline GBNF      : a tiny grammar embedded in the source. Best
---                         for enums and other one-of-N classifications.
---   2. JSON Schema      : compile a JSON Schema into GBNF via the
---                         libcommon bridge — no need to hand-write the
---                         grammar for a structured object.
---   3. Lazy grammar     : the grammar only kicks in once a trigger
---                         word appears. Lets the model write free-form
---                         prose, then locks down a structured tail.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/04_grammar.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ffi   = require "ffi"
local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

local fit   = ion7.Model.fit_params(MODEL)
local model = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local vocab = model:vocab()
local ctx   = model:context({ n_ctx = 1024 })

--- Run a single generation through `sampler` and return the trimmed
--- assistant reply.
local function run(sampler, user_text, max_gen)
    local prompt   = vocab:apply_template(
        { { role = "user", content = user_text } }, true, -1)
    local toks, n  = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(toks, n)
    sampler:reset()

    local pieces = {}
    for _ = 1, max_gen or 128 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        pieces[#pieces + 1] = vocab:piece(tok)
        ctx:decode_single(tok)
    end
    return (table.concat(pieces):gsub("^%s+", ""):gsub("%s+$", ""))
end

-- ════════════════════════════════════════════════════════════════════════
-- 1. Inline GBNF — sentiment classification
-- ════════════════════════════════════════════════════════════════════════
--
-- The grammar restricts the entire output to one of three string
-- literals. Using temperature 0 makes the choice deterministic for the
-- highest-logit candidate inside the grammar mask.

print("\n[1] Sentiment classification — grammar : positive | negative | neutral")
local SENTIMENT = 'root ::= "positive" | "negative" | "neutral"'

local reviews = {
    "Best purchase I have made all year, absolutely thrilled.",
    "Broken on arrival, customer service was unhelpful.",
    "It does the job, nothing more nothing less.",
}
for _, text in ipairs(reviews) do
    local s = ion7.Sampler.chain()
        :grammar(SENTIMENT, "root", vocab)
        :greedy()
        :build()
    print(string.format("  %-50s → %s",
        text:sub(1, 48) .. "..", run(s, "Review : " .. text .. "\nSentiment :", 16)))
    s:free()
end

-- ════════════════════════════════════════════════════════════════════════
-- 2. JSON Schema → GBNF (via bridge)
-- ════════════════════════════════════════════════════════════════════════
--
-- Hand-writing a GBNF for a non-trivial JSON shape is tedious and
-- error-prone. The libcommon bridge ships a JSON-Schema → GBNF
-- compiler we expose via `ion7.core.ffi.bridge`. Pass the schema in,
-- get the grammar string back, plug it into a `:grammar(...)` step.

print("\n[2] JSON Schema → GBNF")
local bridge = require "ion7.core.ffi.bridge"

local SCHEMA = [[
{
  "type": "object",
  "properties": {
    "name":  { "type": "string" },
    "age":   { "type": "integer" },
    "city":  { "type": "string" }
  },
  "required": ["name", "age", "city"]
}
]]

-- Compile once, reuse N times. 64 KiB output buffer is more than
-- enough for any reasonable schema.
local out  = ffi.new("char[?]", 65536)
local nout = bridge.ion7_json_schema_to_grammar(SCHEMA, out, 65536)
assert(nout > 0, "json_schema_to_grammar failed")
local gbnf = ffi.string(out, nout)
print(string.format("  compiled grammar : %d bytes", #gbnf))

local s = ion7.Sampler.chain()
    :grammar(gbnf, "root", vocab)
    :temperature(0.3)
    :dist(42)
    :build()
print("  output :")
print("    " .. run(s,
    "Extract the structured info as JSON. " ..
    "Sentence : John Smith is 34 years old and lives in Paris.",
    256))
s:free()

-- ════════════════════════════════════════════════════════════════════════
-- 3. Nested GBNF — list output with disjunction at every position
-- ════════════════════════════════════════════════════════════════════════
--
-- A bigger grammar that produces a comma-separated list of three
-- colours drawn from a fixed enum. Demonstrates :
---  - a `root` rule that composes sub-rules (`item`, `colour`),
---  - alternation (`|`),
---  - whitespace handling (`ws`).

print("\n[3] Nested GBNF — list of three colours from a fixed enum")

local COLOURS = [[
root   ::= "[" ws colour ws "," ws colour ws "," ws colour ws "]"
colour ::= "\"red\"" | "\"green\"" | "\"blue\"" |
           "\"yellow\"" | "\"purple\"" | "\"orange\""
ws     ::= [ \t\n]*
]]

local list_sampler = ion7.Sampler.chain()
    :grammar(COLOURS, "root", vocab)
    :temperature(0.5)
    :dist(7)
    :build()

print("  output : " .. run(list_sampler,
    "Pick three vivid colours and return them as a JSON array.",
    96))
list_sampler:free()

ctx:free() ; model:free()
ion7.shutdown()
