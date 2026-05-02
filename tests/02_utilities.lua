#!/usr/bin/env luajit
--- @module tests.02_utilities
--- @author  ion7 / Ion7 Project Contributors
---
--- Pure-Lua utility coverage : `ion7.core.util.utf8` and
--- `ion7.core.util.base64`.
---
--- These two modules are pure LuaJIT — no FFI to libllama, no bridge —
--- so they can be exercised without loading a model, which makes them
--- a good first stop in the suite : when the basic pieces are wrong,
--- every higher layer misbehaves silently (mid-codepoint streaming
--- chunks, mangled multimodal blobs, ...).
---
--- Out of scope here :
---   - `ion7.core.util.log`     — configuration is pure Lua but module
---                                 load registers an FFI callback into
---                                 libllama (`llama_log_set`). Covered
---                                 by `01_capabilities.lua`, which has
---                                 the backend up already.
---   - `ion7.core.util.tensor`  — its accessors take a tensor cdata
---                                 produced by a live decode. Covered
---                                 by `15_logits.lua` once a context
---                                 exists.

local T = require "tests.framework"
require "tests.helpers" -- ensures src/?.lua is on package.path

local utf8   = require "ion7.core.util.utf8"
local base64 = require "ion7.core.util.base64"

-- ════════════════════════════════════════════════════════════════════════
-- UTF-8 — leading-byte inspection and stream completeness
-- ════════════════════════════════════════════════════════════════════════
--
-- A token-by-token streamer emits whatever bytes llama.cpp gives it. For
-- a multi-byte codepoint (e.g. the é in "café" → 0xC3 0xA9) it is normal
-- for the boundary to fall mid-codepoint after one of the tokens. The
-- caller must hold the trailing partial bytes back until the next token
-- completes the codepoint, otherwise the user sees a U+FFFD square.
--
-- `utf8.is_complete(buf)` answers exactly that question. We test it
-- against every leading-byte pattern (1..4 bytes) and against the
-- truncation cases the streamer cares about.

T.suite("UTF-8 utilities (ion7.core.util.utf8)")

T.test("seq_len classifies every leading-byte pattern", function()
    -- 0xxx_xxxx — ASCII, 1-byte sequence.
    T.eq(utf8.seq_len(0x41), 1, "'A' should be a 1-byte leader")
    T.eq(utf8.seq_len(0x00), 1, "NUL is a valid 1-byte leader")
    -- 110x_xxxx — 2-byte sequence (Latin Extended, é = 0xC3 0xA9).
    T.eq(utf8.seq_len(0xC3), 2, "first byte of 'é' is a 2-byte leader")
    -- 1110_xxxx — 3-byte sequence (most BMP, e.g. CJK).
    T.eq(utf8.seq_len(0xE4), 3, "first byte of 你 is a 3-byte leader")
    -- 1111_0xxx — 4-byte sequence (supplementary planes, emoji).
    T.eq(utf8.seq_len(0xF0), 4, "emoji leaders need 4 bytes")
    -- 10xx_xxxx — continuation byte, NOT a valid leader.
    T.eq(utf8.seq_len(0xA9), 0, "continuation bytes return 0")
    -- 1111_1xxx — invalid (5+ byte sequences are forbidden in modern UTF-8).
    T.eq(utf8.seq_len(0xF8), 0, "5-byte-style leaders are invalid")
end)

T.test("is_complete accepts well-formed sequences", function()
    T.ok(utf8.is_complete(""),       "empty string is trivially complete")
    T.ok(utf8.is_complete("hello"),  "ASCII-only string is complete")
    T.ok(utf8.is_complete("café"),   "complete 2-byte tail (é)")
    T.ok(utf8.is_complete("日本語"), "complete 3-byte sequences (CJK)")
    T.ok(utf8.is_complete("🚀"),     "complete 4-byte sequence (emoji)")
end)

T.test("is_complete detects mid-sequence truncation", function()
    -- "café" = 63 61 66 C3 A9. Drop the trailing A9 — boundary is now
    -- right after the 0xC3 leader, which announced 2 bytes but only
    -- delivered 1. is_complete must reject that.
    T.eq(utf8.is_complete("caf\xC3"),              false, "1-byte tail of é missing")
    -- 4-byte emoji truncated to its first byte.
    T.eq(utf8.is_complete("\xF0"),                 false, "lone 4-byte leader")
    -- 4-byte emoji truncated mid-stream (got 3 of 4 bytes).
    T.eq(utf8.is_complete("\xF0\x9F\x9A"),         false, "missing last continuation byte")
    -- A standalone continuation byte is not a valid sequence.
    T.eq(utf8.is_complete("\xA9"),                 false, "continuation byte alone is invalid")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Base64 — RFC 4648 encode / decode
-- ════════════════════════════════════════════════════════════════════════
--
-- Base64 is invoked on multimodal image attachments (Vision-Language
-- models) and on opaque blobs the chat layer might want to passthrough
-- intact. The implementation is byte-oriented and NUL-safe, so we test
-- both ASCII roundtrips and embedded NULs.

T.suite("Base64 utilities (ion7.core.util.base64)")

T.test("encode produces standard RFC 4648 output", function()
    T.eq(base64.encode(""),     "",         "empty input → empty output")
    T.eq(base64.encode("f"),    "Zg==",     "1-byte input pads with two '='")
    T.eq(base64.encode("fo"),   "Zm8=",     "2-byte input pads with one '='")
    T.eq(base64.encode("foo"),  "Zm9v",     "3-byte input has no padding")
    T.eq(base64.encode("foob"), "Zm9vYg==", "4-byte input pads with two '='")
    -- The "Hello, World!" canonical example.
    T.eq(base64.encode("Hello, World!"), "SGVsbG8sIFdvcmxkIQ==")
end)

T.test("decode is the inverse of encode", function()
    -- A roundtrip across a wide range of byte values catches sign-bit
    -- and shift-cast bugs in the bit-twiddling.
    local sample = ""
    for b = 0, 255 do
        sample = sample .. string.char(b)
    end
    T.eq(base64.decode(base64.encode(sample)), sample,
         "256-byte alphabet roundtrip preserves bytes")
end)

T.test("decode is NUL-safe (Lua strings carry length, not C strings)", function()
    local with_nul = "a\0b\0c"
    T.eq(base64.decode(base64.encode(with_nul)), with_nul)
end)

T.test("decode silently strips ASCII whitespace", function()
    -- RFC 4648 §3.3 allows implementations to ignore embedded whitespace.
    -- This matters when consuming PEM-style blocks split across lines.
    T.eq(base64.decode("Zm9v"),                 "foo")
    T.eq(base64.decode("Zm9v\n"),               "foo", "trailing LF stripped")
    T.eq(base64.decode("Zm  9\nv"),             "foo", "embedded whitespace stripped")
    T.eq(base64.decode("\t Zm9v \r\n"),         "foo", "leading + trailing whitespace")
end)

T.test("decode returns nil on malformed input", function()
    -- Length not a multiple of 4 once whitespace is stripped.
    T.eq(base64.decode("Zm9"), nil, "3-char body is incomplete")
    -- An invalid alphabet byte (`!`) cannot map to a 6-bit value.
    T.eq(base64.decode("Z!9v"), nil, "non-alphabet character rejected")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

local ok = T.summary()
os.exit(ok and 0 or 1)
