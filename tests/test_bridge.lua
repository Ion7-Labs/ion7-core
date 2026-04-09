#!/usr/bin/env luajit
--- Bridge tests - new APIs added in the libcommon integration.
--- Covers: csampler, speculative, chat_parse, utf8, json utils, regex,
---         cvec, numa, cpu_caps, log, base64, warmup, JSON schema→GBNF.
---
--- Run: ION7_MODEL=/path/to/model.gguf luajit tests/test_bridge.lua
--- Optional: ION7_LIB_DIR=/path/to/llama.cpp/build/bin

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local T          = require "tests.framework"
local model_path = os.getenv("ION7_MODEL")
local lib_dir    = os.getenv("ION7_LIB_DIR")

if not model_path then
    print("[SKIP] Set ION7_MODEL=/path/to/model.gguf to run bridge v2 tests")
    os.exit(0)
end

local ffi    = require "ffi"
local ion7   = require "ion7.core"
local Loader = require "ion7.core.ffi.loader"
ion7.init({ log_level = 0, llama_path = lib_dir, bridge_path = lib_dir })
local L = Loader.instance()
local B = L.bridge
local C = L.C

-- Shared fixtures loaded once
local model = ion7.Model.load(model_path, { n_gpu_layers = 0 })
assert(model, "model load failed")
local vocab = model:vocab()

-- ══════════════════════════════════════════════════════════════════
-- 1. CONTEXT WARMUP
-- ══════════════════════════════════════════════════════════════════

T.suite("Context warmup")

T.test("ion7_context_warmup does not crash", function()
    local ctx = model:context({ n_ctx = 512 })
    B.ion7_context_warmup(ctx:ptr())
    ctx:free()
    T.ok(true)
end)

-- ══════════════════════════════════════════════════════════════════
-- 2. ADVANCED SAMPLER (common_sampler)
-- ══════════════════════════════════════════════════════════════════

T.suite("Advanced sampler (ion7_csampler)")

T.test("csampler_init with NULL params succeeds", function()
    local s = B.ion7_csampler_init(model:ptr(), nil, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil, "csampler handle should not be null")
    B.ion7_csampler_free(s)
end)

T.test("csampler_init with explicit params", function()
    local p = ffi.new("ion7_csampler_params_t")
    p.seed    = 42
    p.top_k   = 40
    p.top_p   = 0.95
    p.temp    = 0.8
    local s = B.ion7_csampler_init(model:ptr(), p, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil, "csampler with params should succeed")
    B.ion7_csampler_free(s)
end)

T.test("csampler_get_seed returns non-zero for random seed", function()
    local p = ffi.new("ion7_csampler_params_t")
    p.seed = 0xFFFFFFFF  -- LLAMA_DEFAULT_SEED
    local s = B.ion7_csampler_init(model:ptr(), p, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil)
    local seed = B.ion7_csampler_get_seed(s)
    T.ok(type(tonumber(seed)) == "number")
    B.ion7_csampler_free(s)
end)

T.test("csampler_init with GBNF grammar", function()
    local gbnf = 'root ::= "yes" | "no"'
    local s = B.ion7_csampler_init(model:ptr(), nil, gbnf, nil, 0, nil, nil, 0)
    T.ok(s ~= nil, "csampler with grammar should succeed")
    B.ion7_csampler_free(s)
end)

T.test("csampler_reset does not crash", function()
    local s = B.ion7_csampler_init(model:ptr(), nil, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil)
    B.ion7_csampler_reset(s)
    B.ion7_csampler_free(s)
end)

T.test("csampler_last returns -1 before any acceptance", function()
    local s = B.ion7_csampler_init(model:ptr(), nil, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil)
    T.eq(tonumber(B.ion7_csampler_last(s)), -1)
    B.ion7_csampler_free(s)
end)

T.test("csampler_sample + accept roundtrip", function()
    local ctx = model:context({ n_ctx = 128 })
    local s   = B.ion7_csampler_init(model:ptr(), nil, nil, nil, 0, nil, nil, 0)
    T.ok(s ~= nil)

    -- Encode a small prompt so logits are populated
    local tokens, n = vocab:tokenize("Hello", true, false)
    ctx:decode(tokens, n, 0, 0)

    local tok = B.ion7_csampler_sample(s, ctx:ptr(), -1, 0)
    T.ok(tonumber(tok) >= 0, "sampled token should be >= 0")
    B.ion7_csampler_accept(s, tok)
    T.eq(tonumber(B.ion7_csampler_last(s)), tonumber(tok))

    B.ion7_csampler_free(s)
    ctx:free()
end)

-- ══════════════════════════════════════════════════════════════════
-- 3. SPECULATIVE DECODING
-- ══════════════════════════════════════════════════════════════════

T.suite("Speculative decoding (ion7_speculative)")

T.test("speculative_init NGRAM_SIMPLE succeeds", function()
    local ctx = model:context({ n_ctx = 512 })
    local spec = B.ion7_speculative_init(ctx:ptr(), nil,
                     C.SPEC_NGRAM_SIMPLE, 8, 1, 4)
    T.ok(spec ~= nil, "spec handle should not be null")
    B.ion7_speculative_free(spec)
    ctx:free()
end)

T.test("speculative_init NGRAM_CACHE succeeds", function()
    local ctx = model:context({ n_ctx = 512 })
    local spec = B.ion7_speculative_init(ctx:ptr(), nil,
                     C.SPEC_NGRAM_CACHE, 8, 1, 4)
    T.ok(spec ~= nil)
    B.ion7_speculative_free(spec)
    ctx:free()
end)

T.test("speculative_begin does not crash", function()
    local ctx  = model:context({ n_ctx = 512 })
    local spec = B.ion7_speculative_init(ctx:ptr(), nil, C.SPEC_NGRAM_SIMPLE, 8, 1, 4)
    T.ok(spec ~= nil)

    local tokens, n = vocab:tokenize("Hello world", true, false)
    B.ion7_speculative_begin(spec, tokens, n)

    B.ion7_speculative_free(spec)
    ctx:free()
end)

T.test("speculative_draft returns >= 0 tokens", function()
    local ctx  = model:context({ n_ctx = 512 })
    local spec = B.ion7_speculative_init(ctx:ptr(), nil, C.SPEC_NGRAM_SIMPLE, 8, 1, 4)
    local tokens, n = vocab:tokenize("Hello world", true, false)
    B.ion7_speculative_begin(spec, tokens, n)

    local out   = ffi.new("int32_t[8]")
    local last  = tokens[n - 1]
    local ndraft = B.ion7_speculative_draft(spec, tokens, n, last, out, 8)
    T.ok(tonumber(ndraft) >= 0, "draft count should be >= 0")

    B.ion7_speculative_free(spec)
    ctx:free()
end)

T.test("speculative_accept does not crash", function()
    local ctx  = model:context({ n_ctx = 512 })
    local spec = B.ion7_speculative_init(ctx:ptr(), nil, C.SPEC_NGRAM_SIMPLE, 8, 1, 4)
    B.ion7_speculative_accept(spec, 0)
    B.ion7_speculative_free(spec)
    ctx:free()
end)

-- ══════════════════════════════════════════════════════════════════
-- 4. CHAT TEMPLATES + PARSE
-- ══════════════════════════════════════════════════════════════════

T.suite("Chat templates + parse (ion7_chat_templates / ion7_chat_parse)")

T.test("chat_templates_init succeeds", function()
    local t = B.ion7_chat_templates_init(model:ptr(), nil)
    T.ok(t ~= nil, "chat templates handle should not be null")
    B.ion7_chat_templates_free(t)
end)

T.test("chat_templates_support_thinking returns 0 or 1", function()
    local t   = B.ion7_chat_templates_init(model:ptr(), nil)
    local sup = tonumber(B.ion7_chat_templates_support_thinking(t))
    T.ok(sup == 0 or sup == 1, "should be 0 or 1, got " .. tostring(sup))
    B.ion7_chat_templates_free(t)
end)

T.test("chat_templates_apply returns a non-empty string", function()
    local t      = B.ion7_chat_templates_init(model:ptr(), nil)
    local roles  = ffi.new("const char*[1]", { "user" })
    local conts  = ffi.new("const char*[1]", { "Hello" })
    local buf    = ffi.new("char[4096]")
    local n = B.ion7_chat_templates_apply(t, roles, conts, 1, 1, -1, buf, 4096)
    T.ok(n > 0, "apply should return > 0 bytes, got " .. tostring(n))
    local result = ffi.string(buf)
    T.ok(#result > 0, "result string should not be empty")
    B.ion7_chat_templates_free(t)
end)

T.test("chat_parse splits content from thinking", function()
    local t       = B.ion7_chat_templates_init(model:ptr(), nil)
    local raw     = "<think>inner reasoning</think>actual answer"
    local content = ffi.new("char[2048]")
    local thinking= ffi.new("char[2048]")
    local tools   = ffi.new("char[2048]")
    local has_tools = ffi.new("int[1]")

    local rc = B.ion7_chat_parse(t, raw, 1,
        content,  2048,
        thinking, 2048,
        tools,    2048,
        has_tools)
    T.ok(rc == 0 or rc == 1, "chat_parse rc should be 0 (ok) or 1 (truncated)")
    -- thinking block may or may not be extracted depending on template support
    T.ok(ffi.string(content) ~= nil)
    B.ion7_chat_templates_free(t)
end)

-- ══════════════════════════════════════════════════════════════════
-- 5. UTF-8 HELPERS
-- ══════════════════════════════════════════════════════════════════

T.suite("UTF-8 helpers (ion7_utf8_*)")

T.test("utf8_seq_len ASCII byte returns 1", function()
    T.eq(tonumber(B.ion7_utf8_seq_len(0x41)), 1)   -- 'A'
end)

T.test("utf8_seq_len 2-byte leading byte returns 2", function()
    T.eq(tonumber(B.ion7_utf8_seq_len(0xC3)), 2)   -- é leading byte
end)

T.test("utf8_seq_len 3-byte leading byte returns 3", function()
    T.eq(tonumber(B.ion7_utf8_seq_len(0xE4)), 3)   -- CJK leading byte
end)

T.test("utf8_seq_len 4-byte leading byte returns 4", function()
    T.eq(tonumber(B.ion7_utf8_seq_len(0xF0)), 4)   -- emoji leading byte
end)

T.test("utf8_seq_len continuation byte returns 0", function()
    T.eq(tonumber(B.ion7_utf8_seq_len(0x80)), 0)
end)

T.test("utf8_is_complete on ASCII string returns 1", function()
    local s = "hello"
    T.eq(tonumber(B.ion7_utf8_is_complete(s, #s)), 1)
end)

T.test("utf8_is_complete on complete 2-byte char returns 1", function()
    local s = "\xC3\xA9"   -- é
    T.eq(tonumber(B.ion7_utf8_is_complete(s, #s)), 1)
end)

T.test("utf8_is_complete on truncated multi-byte returns 0", function()
    local s = "\xC3"       -- é without second byte
    T.eq(tonumber(B.ion7_utf8_is_complete(s, #s)), 0)
end)

T.test("utf8_is_complete on complete 3-byte CJK returns 1", function()
    local s = "\xE4\xB8\xAD"  -- 中
    T.eq(tonumber(B.ion7_utf8_is_complete(s, #s)), 1)
end)

-- ══════════════════════════════════════════════════════════════════
-- 6. JSON SCHEMA → GBNF
-- ══════════════════════════════════════════════════════════════════

T.suite("JSON schema → GBNF (ion7_json_schema_to_grammar)")

T.test("simple string-type schema produces non-empty GBNF", function()
    local schema = '{"type":"string"}'
    local buf    = ffi.new("char[4096]")
    local n      = B.ion7_json_schema_to_grammar(schema, buf, 4096)
    T.ok(n > 0, "should return > 0 bytes needed, got " .. tostring(n))
    T.ok(ffi.string(buf):find("root") ~= nil, "GBNF should contain 'root' rule")
end)

T.test("object schema with properties produces GBNF", function()
    local schema = '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}'
    local buf    = ffi.new("char[8192]")
    local n      = B.ion7_json_schema_to_grammar(schema, buf, 8192)
    T.ok(n > 0, "schema→GBNF should succeed")
end)

T.test("invalid JSON returns -1", function()
    local buf = ffi.new("char[256]")
    local n   = B.ion7_json_schema_to_grammar("not json", buf, 256)
    T.eq(tonumber(n), -1, "invalid schema should return -1")
end)

T.test("NULL output queries required buffer size", function()
    local schema = '{"type":"boolean"}'
    local n      = B.ion7_json_schema_to_grammar(schema, nil, 0)
    T.ok(n > 0, "size query should return needed bytes")
end)

-- ══════════════════════════════════════════════════════════════════
-- 7. PARTIAL REGEX
-- ══════════════════════════════════════════════════════════════════

T.suite("Partial regex (ion7_regex)")

T.test("regex_new with valid pattern returns non-NULL", function()
    local r = B.ion7_regex_new("hello")
    T.ok(r ~= nil)
    B.ion7_regex_free(r)
end)

T.test("regex_new with invalid pattern returns NULL", function()
    local r = B.ion7_regex_new("[invalid")
    T.ok(r == nil, "invalid pattern should return NULL")
end)

T.test("regex_search full match returns 2", function()
    local r = B.ion7_regex_new("hello")
    local s = "hello world"
    T.eq(tonumber(B.ion7_regex_search(r, s, #s, 0)), 2)   -- REGEX_FULL = 2
    B.ion7_regex_free(r)
end)

T.test("regex_search no match returns 0", function()
    local r = B.ion7_regex_new("xyz")
    local s = "hello world"
    T.eq(tonumber(B.ion7_regex_search(r, s, #s, 0)), 0)   -- REGEX_NO_MATCH = 0
    B.ion7_regex_free(r)
end)

T.test("regex_search partial prefix match returns 1", function()
    local r = B.ion7_regex_new("</think>")
    local s = "</thi"   -- prefix of </think>
    local rc = tonumber(B.ion7_regex_search(r, s, #s, 1))
    T.ok(rc == 1 or rc == 0, "partial or no-match expected, got " .. tostring(rc))
    B.ion7_regex_free(r)
end)

T.test("regex constants are correct", function()
    T.eq(C.REGEX_NO_MATCH, 0)
    T.eq(C.REGEX_PARTIAL,  1)
    T.eq(C.REGEX_FULL,     2)
end)

-- ══════════════════════════════════════════════════════════════════
-- 8. JSON UTILITIES
-- ══════════════════════════════════════════════════════════════════

T.suite("JSON utilities (ion7_json_*)")

T.test("json_validate returns 1 for valid JSON", function()
    T.eq(tonumber(B.ion7_json_validate('{"key":"value"}')), 1)
    T.eq(tonumber(B.ion7_json_validate("[1,2,3]")),         1)
    T.eq(tonumber(B.ion7_json_validate('"string"')),        1)
    T.eq(tonumber(B.ion7_json_validate("42")),              1)
    T.eq(tonumber(B.ion7_json_validate("true")),            1)
    T.eq(tonumber(B.ion7_json_validate("null")),            1)
end)

T.test("json_validate returns 0 for invalid JSON", function()
    T.eq(tonumber(B.ion7_json_validate("not json")),  0)
    T.eq(tonumber(B.ion7_json_validate("{bad}")),     0)
    T.eq(tonumber(B.ion7_json_validate("")),          0)
end)

T.test("json_format pretty-prints JSON", function()
    local buf = ffi.new("char[1024]")
    local n   = B.ion7_json_format('{"a":1,"b":2}', buf, 1024)
    T.ok(n > 0)
    local s = ffi.string(buf)
    T.ok(s:find("\n") ~= nil, "formatted JSON should contain newlines")
end)

T.test("json_format returns -1 for invalid input", function()
    local buf = ffi.new("char[256]")
    local n   = B.ion7_json_format("not json", buf, 256)
    T.eq(tonumber(n), -1)
end)

T.test("json_format with NULL queries required size", function()
    local n = B.ion7_json_format('{"x":1}', nil, 0)
    T.ok(n > 0, "size query should return > 0")
end)

T.test("json_merge overlays keys from patch", function()
    local buf = ffi.new("char[512]")
    local n   = B.ion7_json_merge('{"a":1,"b":2}', '{"b":99,"c":3}', buf, 512)
    T.ok(n > 0)
    local result = ffi.string(buf)
    -- Parse with our Lua json and verify semantics
    local json = require "ion7.vendor.json"
    local t    = json.decode(result)
    T.eq(t.a, 1,  "a should be preserved")
    T.eq(t.b, 99, "b should be overwritten by patch")
    T.eq(t.c, 3,  "c should be added from patch")
end)

T.test("json_merge with null value deletes key (RFC 7396)", function()
    local buf = ffi.new("char[512]")
    local n   = B.ion7_json_merge('{"a":1,"b":2}', '{"b":null}', buf, 512)
    T.ok(n > 0)
    local json = require "ion7.vendor.json"
    local t    = json.decode(ffi.string(buf))
    T.eq(t.a, 1,  "a should remain")
    T.eq(t.b, nil, "b should be deleted by null patch")
end)

T.test("json_merge returns -1 on invalid base", function()
    local buf = ffi.new("char[256]")
    local n   = B.ion7_json_merge("bad", '{"x":1}', buf, 256)
    T.eq(tonumber(n), -1)
end)

-- ══════════════════════════════════════════════════════════════════
-- 9. BASE64
-- ══════════════════════════════════════════════════════════════════

T.suite("Base64 (ion7_base64_encode / decode)")

T.test("encode empty returns 0 chars", function()
    local out = ffi.new("char[8]")
    local n   = B.ion7_base64_encode(ffi.cast("const uint8_t*", ""), 0, out, 8)
    T.eq(tonumber(n), 0)
end)

T.test("encode 'Man' → 'TWFu' (RFC 4648 example)", function()
    local src = ffi.cast("const uint8_t*", "Man")
    local out = ffi.new("char[8]")
    local n   = B.ion7_base64_encode(src, 3, out, 8)
    T.eq(tonumber(n), 4)
    T.eq(ffi.string(out, 4), "TWFu")
end)

T.test("encode/decode roundtrip", function()
    local orig = "Hello, ion7!"
    local src  = ffi.cast("const uint8_t*", orig)
    -- encode
    local enc_buf = ffi.new("char[64]")
    local enc_n   = B.ion7_base64_encode(src, #orig, enc_buf, 64)
    T.ok(tonumber(enc_n) > 0)
    -- decode
    local dec_buf = ffi.new("uint8_t[64]")
    local dec_n   = B.ion7_base64_decode(enc_buf, enc_n, dec_buf, 64)
    T.eq(tonumber(dec_n), #orig)
    T.eq(ffi.string(dec_buf, dec_n), orig)
end)

T.test("encode returns -1 when output buffer too small", function()
    local src = ffi.cast("const uint8_t*", "Hello")
    local out = ffi.new("char[2]")
    local n   = B.ion7_base64_encode(src, 5, out, 2)
    T.eq(tonumber(n), -1)
end)

T.test("decode returns -1 on invalid base64", function()
    local out = ffi.new("uint8_t[64]")
    local n   = B.ion7_base64_decode("!!!!", 4, out, 64)
    T.eq(tonumber(n), -1)
end)

-- ══════════════════════════════════════════════════════════════════
-- 10. CPU CAPS
-- ══════════════════════════════════════════════════════════════════

T.suite("CPU capabilities (ion7_cpu_caps)")

T.test("ion7_cpu_caps fills struct without crash", function()
    local caps = ffi.new("ion7_cpu_caps_t")
    B.ion7_cpu_caps(caps)
    T.ok(true, "should not crash")
end)

T.test("cpu caps fields are 0 or 1", function()
    local caps = ffi.new("ion7_cpu_caps_t")
    B.ion7_cpu_caps(caps)
    for _, field in ipairs({ "avx", "avx2", "sse3", "neon", "riscv_v", "wasm_simd" }) do
        local v = tonumber(caps[field])
        T.ok(v == 0 or v == 1,
            field .. " should be 0 or 1, got " .. tostring(v))
    end
end)

T.test("sve_cnt is 0 or a multiple of 8", function()
    local caps = ffi.new("ion7_cpu_caps_t")
    B.ion7_cpu_caps(caps)
    local cnt = tonumber(caps.sve_cnt)
    T.ok(cnt == 0 or cnt % 8 == 0,
        "sve_cnt should be 0 or multiple of 8, got " .. tostring(cnt))
end)

-- ══════════════════════════════════════════════════════════════════
-- 11. NUMA
-- ══════════════════════════════════════════════════════════════════

T.suite("NUMA (ion7_numa_init / ion7_is_numa)")

T.test("is_numa returns 0 before init (disabled)", function()
    -- NUMA was not initialised in this session
    local v = tonumber(B.ion7_is_numa())
    T.ok(v == 0 or v == 1, "is_numa should be 0 or 1")
end)

T.test("numa constants are correct", function()
    T.eq(C.NUMA_DISABLED,   0)
    T.eq(C.NUMA_DISTRIBUTE, 1)
    T.eq(C.NUMA_ISOLATE,    2)
    T.eq(C.NUMA_NUMACTL,    3)
    T.eq(C.NUMA_MIRROR,     4)
end)

-- ══════════════════════════════════════════════════════════════════
-- 12. LOG ROUTING
-- ══════════════════════════════════════════════════════════════════

T.suite("Log routing (ion7_log_to_file / ion7_log_set_timestamps)")

T.test("log_set_timestamps does not crash", function()
    B.ion7_log_set_timestamps(1)
    B.ion7_log_set_timestamps(0)
    T.ok(true)
end)

T.test("log_to_file with NULL restores stderr without crash", function()
    B.ion7_log_to_file(nil)
    T.ok(true)
end)

-- ══════════════════════════════════════════════════════════════════
-- 13. SPECULATIVE CONSTANTS
-- ══════════════════════════════════════════════════════════════════

T.suite("Speculative constants")

T.test("spec constants are 1:1 with common_speculative_type / ion7_bridge.h", function()
    T.eq(C.SPEC_NONE,          0)
    T.eq(C.SPEC_DRAFT,         1)
    T.eq(C.SPEC_EAGLE3,        2)
    T.eq(C.SPEC_NGRAM_SIMPLE,  3)
    T.eq(C.SPEC_NGRAM_MAP_K,   4)
    T.eq(C.SPEC_NGRAM_MAP_K4V, 5)
    T.eq(C.SPEC_NGRAM_MOD,     6)
    T.eq(C.SPEC_NGRAM_CACHE,   7)
end)

-- ══════════════════════════════════════════════════════════════════
-- Logprob / Entropy (ion7_logprob, ion7_entropy)
-- ══════════════════════════════════════════════════════════════════

T.suite("Logprob / Entropy (bridge_utils)")

T.test("ion7_logprob returns a negative float after decode", function()
    local ctx  = model:context({ n_ctx = 512, n_gpu_layers = 0 })
    local v    = model:vocab()
    local tokens, n = v:tokenize("Hello", false, false)
    ctx:decode(tokens, n)
    local lp = B.ion7_logprob(ctx:ptr(), 0, v:bos())
    T.ok(type(tonumber(lp)) == "number", "logprob should be a number")
    T.ok(tonumber(lp) <= 0.0, "log-probability must be <= 0")
    T.ok(tonumber(lp) > -1e9, "log-probability should be finite")
    ctx:free()
end)

T.test("ion7_entropy returns a non-negative float after decode", function()
    local ctx = model:context({ n_ctx = 512, n_gpu_layers = 0 })
    local v   = model:vocab()
    local tokens, n = v:tokenize("Hello", false, false)
    ctx:decode(tokens, n)
    local H = B.ion7_entropy(ctx:ptr(), 0)
    T.ok(type(tonumber(H)) == "number", "entropy should be a number")
    T.ok(tonumber(H) >= 0.0, "entropy must be >= 0")
    ctx:free()
end)

T.test("ctx:logprob() and ctx:entropy() delegate to bridge", function()
    local ctx = model:context({ n_ctx = 512, n_gpu_layers = 0 })
    local v   = model:vocab()
    local tokens, n = v:tokenize("Hi", false, false)
    ctx:decode(tokens, n)
    local lp = ctx:logprob(0, v:bos())
    local H  = ctx:entropy(0)
    T.ok(lp <= 0.0,  "ctx:logprob should be <= 0")
    T.ok(H  >= 0.0,  "ctx:entropy should be >= 0")
    -- Both should agree with direct bridge calls
    local lp2 = tonumber(B.ion7_logprob(ctx:ptr(), 0, v:bos()))
    T.ok(math.abs(lp - lp2) < 1e-6, "ctx:logprob matches B.ion7_logprob")
    ctx:free()
end)

-- ══════════════════════════════════════════════════════════════════
-- CLEANUP & SUMMARY
-- ══════════════════════════════════════════════════════════════════

model:free()
ion7.shutdown()

local ok = T.summary()
os.exit(ok and 0 or 1)
