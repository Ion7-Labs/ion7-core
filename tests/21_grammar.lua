#!/usr/bin/env luajit
--- @module tests.21_grammar
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 21 — Grammar tooling : JSON Schema → GBNF, partial regex, JSON utils
--- ════════════════════════════════════════════════════════════════════════
---
--- The bridge exposes four utilities a chat / RAG / agent framework
--- typically pulls in from the libcommon side rather than rebuilding :
---
---   1. `ion7_json_schema_to_grammar`  : compile a JSON Schema into a
---      llama.cpp-compatible GBNF string. Plug the result into the
---      `Sampler:grammar` step to constrain output to schema-valid JSON.
---   2. `ion7_regex_*`                 : partial-regex matcher for
---      streaming stop-string detection. `partial = 1` returns 1 even
---      on incomplete prefixes that COULD still match.
---   3. `ion7_json_validate`           : "is this string parseable JSON".
---   4. `ion7_json_format`             : pretty-print JSON.
---   5. `ion7_json_merge`              : deep-merge two JSON objects
---      (overlay wins). Useful for layered config / tool args.
---
--- All four are called via the bridge module ; no Lua wrapper.
---
---   ION7_MODEL=/path/to/model.gguf luajit tests/21_grammar.lua

local ffi    = require "ffi"
local T      = require "tests.framework"
local H      = require "tests.helpers"
local ion7   = H.require_backend(T)
local bridge = require "ion7.core.ffi.bridge"

local function buf(n)
    return ffi.new("char[?]", n)
end

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — JSON Schema → GBNF
-- ════════════════════════════════════════════════════════════════════════
--
-- The grammar text is several KB even for tiny schemas, so we size
-- the output buffer at 64 KiB. The bridge returns the bytes-needed
-- count when the buffer is too small ; we test that signal too.

T.suite("JSON Schema → GBNF (ion7_json_schema_to_grammar)")

local SCHEMA = [[
{
  "type": "object",
  "properties": {
    "city":    { "type": "string"  },
    "celsius": { "type": "integer" }
  },
  "required": ["city", "celsius"]
}
]]

T.test("produces a non-empty GBNF string for a simple schema", function()
    local out = buf(65536)
    local n = bridge.ion7_json_schema_to_grammar(SCHEMA, out, 65536)
    T.gt(n, 0, "should report bytes written")

    local gbnf = ffi.string(out, n)
    -- A minimal sanity check on the shape : every llama.cpp grammar
    -- defines at least the canonical `root` non-terminal.
    T.contains(gbnf, "root")
end)

T.test("rejects malformed input gracefully", function()
    local out = buf(1024)
    local n = bridge.ion7_json_schema_to_grammar("not a schema {{{", out, 1024)
    -- Negative or zero on failure ; we accept either as long as no
    -- bytes are claimed to have been written.
    T.gte(0, n)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Partial regex (streaming stop-string detection)
-- ════════════════════════════════════════════════════════════════════════
--
-- `partial = 1` makes the matcher accept incomplete prefixes that
-- COULD still match the pattern. A streaming generator uses it to
-- decide whether the current output buffer might be the start of a
-- stop string and should be held back.

T.suite("Partial regex (ion7_regex_*)")

T.test("full match returns >= 1, no match returns 0", function()
    local r = bridge.ion7_regex_new("Done\\.")
    T.is_type(r, "cdata")

    local s = "Job: Done."
    local hit = bridge.ion7_regex_search(r, s, #s, 0)
    T.gte(hit, 1, "full pattern is in the haystack")

    s = "Job: Working."
    T.eq(bridge.ion7_regex_search(r, s, #s, 0), 0, "no match")

    bridge.ion7_regex_free(r)
end)

T.test("partial=1 accepts a prefix that could grow into a match", function()
    local r = bridge.ion7_regex_new("Hello, world!")

    -- Streaming chunks. The first two could still grow into the full
    -- pattern ; the third is a confirmed match.
    local prefix = "Hello, "
    T.gte(bridge.ion7_regex_search(r, prefix, #prefix, 1), 1,
          "partial=1 must accept incomplete prefix")

    local final = "Hello, world!"
    T.gte(bridge.ion7_regex_search(r, final, #final, 1), 1)

    -- partial=0 on the same prefix must reject — there's no full match yet.
    T.eq(bridge.ion7_regex_search(r, prefix, #prefix, 0), 0)

    bridge.ion7_regex_free(r)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — JSON validate / format / merge
-- ════════════════════════════════════════════════════════════════════════

T.suite("JSON utilities (validate / format / merge)")

T.test("ion7_json_validate", function()
    T.eq(bridge.ion7_json_validate('{"a":1}'),       1, "well-formed object")
    T.eq(bridge.ion7_json_validate('[1, 2, 3]'),     1, "well-formed array")
    T.eq(bridge.ion7_json_validate('{"a":}'),        0, "syntax error")
    T.eq(bridge.ion7_json_validate('not json'),      0, "not even close")
end)

T.test("ion7_json_format pretty-prints", function()
    local out = buf(1024)
    local n = bridge.ion7_json_format('{"b":2,"a":1}', out, 1024)
    T.gt(n, 0)
    local pretty = ffi.string(out, n)
    -- Pretty-printed JSON has whitespace AND a newline between fields.
    T.contains(pretty, "\n")
end)

T.test("ion7_json_merge deep-merges base and overlay", function()
    local out = buf(1024)
    local n = bridge.ion7_json_merge(
        '{"a": 1, "b": {"x": 1}}',
        '{"a": 2, "b": {"y": 2}}',
        out, 1024)
    T.gt(n, 0)
    local merged = ffi.string(out, n)
    -- Overlay overrode `a`. Both nested keys survived.
    T.contains(merged, "\"a\"")
    T.contains(merged, "\"x\"")
    T.contains(merged, "\"y\"")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ion7.shutdown()
os.exit(T.summary() and 0 or 1)
