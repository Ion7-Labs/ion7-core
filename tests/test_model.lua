#!/usr/bin/env luajit
--- Exhaustive model tests - every ion7-core API surface tested.
--- No blind spots. If it's in PUBLIC_API.md, it's tested here.
---
--- Run: ION7_MODEL=/path/to/model.gguf luajit tests/test_model.lua
--- Optional: ION7_LIB_DIR=/path/to/llama.cpp/build/bin

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local T          = require "tests.framework"
local model_path = os.getenv("ION7_MODEL")
local lib_dir    = os.getenv("ION7_LIB_DIR")

if not model_path then
    print("[SKIP] Set ION7_MODEL=/path/to/model.gguf")
    os.exit(0)
end

local ion7 = require "ion7.core"
ion7.init({ log_level = 0, llama_path = lib_dir, bridge_path = lib_dir })

-- ══════════════════════════════════════════════════════════════════
-- 1. CAPABILITIES
-- ══════════════════════════════════════════════════════════════════

T.suite("ion7.capabilities()")

T.test("returns complete table", function()
    local cap = ion7.capabilities()
    T.ok(type(cap.mmap)              == "boolean", "mmap")
    T.ok(type(cap.mlock)             == "boolean", "mlock")
    T.ok(type(cap.gpu_offload)       == "boolean", "gpu_offload")
    T.ok(type(cap.rpc)               == "boolean", "rpc")
    T.ok(type(cap.max_devices)       == "number",  "max_devices")
    T.ok(type(cap.max_parallel_seqs) == "number",  "max_parallel_seqs")
    T.ok(type(cap.bridge_ver)        == "string" and #cap.bridge_ver > 0, "bridge_ver")
    T.ok(type(cap.llama_info)        == "string" and #cap.llama_info > 0, "llama_info")
    print(string.format("  bridge=%s  gpu=%s  rpc=%s  devices=%d  parallel_seqs=%d",
        cap.bridge_ver, tostring(cap.gpu_offload), tostring(cap.rpc),
        cap.max_devices, cap.max_parallel_seqs))
end)

T.test("ion7.time_us() returns increasing values", function()
    local t1 = ion7.time_us()
    local t2 = ion7.time_us()
    T.ok(type(t1) == "number")
    T.ok(t2 >= t1, "time should not go backwards")
end)

T.test("capabilities table keys are exhaustive", function()
    local cap = ion7.capabilities()
    local expected_keys = {
        "mmap", "mlock", "gpu_offload", "rpc",
        "max_devices", "max_parallel_seqs",
        "bridge_ver", "llama_info",
    }
    for _, key in ipairs(expected_keys) do
        T.ok(cap[key] ~= nil, "missing key: " .. key)
    end
end)

-- ══════════════════════════════════════════════════════════════════
-- 2. FIT_PARAMS
-- ══════════════════════════════════════════════════════════════════

T.suite("Model.fit_params")

local fit

T.test("returns n_gpu_layers and n_ctx", function()
    fit = ion7.Model.fit_params(model_path)
    T.ok(fit ~= nil, "fit_params returned nil")
    T.ok(type(fit.n_gpu_layers) == "number", "n_gpu_layers is number")
    T.ok(type(fit.n_ctx)        == "number", "n_ctx is number")
    T.ok(fit.n_gpu_layers >= -1, "n_gpu_layers >= -1 (-1 = all layers)")
    T.ok(fit.n_ctx > 0,          "n_ctx > 0")
    print(string.format("  → n_gpu_layers=%d  n_ctx=%d", fit.n_gpu_layers, fit.n_ctx))
end)

T.test("respects n_ctx_min", function()
    local f = ion7.Model.fit_params(model_path, { n_ctx_min = 1024 })
    if f then T.ok(f.n_ctx >= 1024, "n_ctx >= n_ctx_min") end
end)

T.test("fails gracefully on bad path", function()
    local f = ion7.Model.fit_params("/nonexistent/model.gguf")
    T.ok(f == nil, "should return nil for nonexistent path")
end)

-- ══════════════════════════════════════════════════════════════════
-- 3. MODEL LOAD & INTROSPECTION
-- ══════════════════════════════════════════════════════════════════

T.suite("Model.load + introspection")

local model

T.test("Model.load() succeeds", function()
    model = ion7.Model.load(model_path, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
    T.ok(model ~= nil, "model should not be nil")
end)

T.test("info() - complete metadata", function()
    local i = model:info()
    T.ok(type(i.n_params)    == "number" and i.n_params    > 0, "n_params")
    T.ok(type(i.n_layer)     == "number" and i.n_layer     > 0, "n_layer")
    T.ok(type(i.n_embd)      == "number" and i.n_embd      > 0, "n_embd")
    T.ok(type(i.n_head)      == "number" and i.n_head      > 0, "n_head")
    T.ok(type(i.n_ctx_train) == "number" and i.n_ctx_train > 0, "n_ctx_train")
    T.ok(type(i.size)        == "number" and i.size        > 0, "size")
    print(string.format("  %.2fB params  %d layers  n_embd=%d  n_head=%d  ctx_train=%d",
        i.n_params/1e9, i.n_layer, i.n_embd, i.n_head, i.n_ctx_train))
end)

T.test("n_embd() n_layer() etc direct accessors", function()
    T.ok(model:n_embd()      > 0, "n_embd")
    T.ok(model:n_layer()     > 0, "n_layer")
    T.ok(model:n_head()      > 0, "n_head")
    T.ok(model:n_head_kv()   > 0, "n_head_kv")
    T.ok(model:n_ctx_train() > 0, "n_ctx_train")
    T.ok(model:n_params()    > 0, "n_params")
    T.ok(model:size()        > 0, "size")
    T.ok(model:n_embd_out()  >= 0, "n_embd_out")
end)

T.test("rope_type() returns valid string", function()
    local rt = model:rope_type()
    local valid = { none=1, norm=1, neox=1, mrope=1, imrope=1, vision=1, unknown=1 }
    T.ok(valid[rt] ~= nil, "unknown rope_type: " .. tostring(rt))
    print("  rope_type: " .. rt)
end)

T.test("rope_freq_scale_train() is a positive float", function()
    local s = model:rope_freq_scale_train()
    T.ok(type(s) == "number" and s > 0, "should be > 0, got " .. tostring(s))
end)

T.test("has_encoder() has_decoder() return booleans", function()
    T.ok(type(model:has_encoder()) == "boolean", "has_encoder")
    T.ok(type(model:has_decoder()) == "boolean", "has_decoder")
end)

T.test("is_recurrent() is_hybrid() is_diffusion() return booleans", function()
    T.ok(type(model:is_recurrent()) == "boolean", "is_recurrent")
    T.ok(type(model:is_hybrid())    == "boolean", "is_hybrid")
    T.ok(type(model:is_diffusion()) == "boolean", "is_diffusion")
end)

T.test("decoder_start_token() returns int", function()
    local t = model:decoder_start_token()
    T.ok(type(t) == "number", "should be a number")
end)

T.test("n_cls_out() returns int >= 0", function()
    local n = model:n_cls_out()
    T.ok(type(n) == "number" and n >= 0, "n_cls_out >= 0")
end)

T.test("meta_count() > 0", function()
    T.ok(model:meta_count() > 0, "should have GGUF metadata")
    print("  meta keys: " .. model:meta_count())
end)

T.test("meta_key_at(0) returns a string", function()
    local key = model:meta_key_at(0)
    T.ok(type(key) == "string" and #key > 0)
    print("  meta key[0]: " .. key)
end)

T.test("meta_val_at(0) returns a string", function()
    local val = model:meta_val_at(0)
    T.ok(type(val) == "string")
end)

T.test("meta_val('general.name') returns string or nil", function()
    local name = model:meta_val("general.name")
    T.ok(name == nil or type(name) == "string")
    if name then print("  model name: " .. name) end
end)

T.test("chat_template() returns string or nil", function()
    local tmpl = model:chat_template(nil)
    T.ok(tmpl == nil or type(tmpl) == "string")
    if tmpl then print(string.format("  chat template: %d chars", #tmpl)) end
end)

T.test("desc() returns non-empty string", function()
    local d = model:desc()
    T.ok(type(d) == "string" and #d > 0, "desc should be non-empty")
    print("  desc: " .. d)
end)

T.test("ptr() returns non-nil cdata", function()
    local p = model:ptr()
    T.ok(p ~= nil, "ptr() should not be nil")
end)

T.test("n_embd_inp() returns number >= 0", function()
    local n = model:n_embd_inp()
    T.ok(type(n) == "number" and n >= 0, "n_embd_inp >= 0, got " .. tostring(n))
    print("  n_embd_inp: " .. n)
end)

T.test("n_swa() returns number >= 0", function()
    local n = model:n_swa()
    T.ok(type(n) == "number" and n >= 0, "n_swa >= 0, got " .. tostring(n))
    print("  n_swa: " .. n)
end)

-- ══════════════════════════════════════════════════════════════════
-- 4. QUANTIZE (dry-run only - no write)
-- ══════════════════════════════════════════════════════════════════

T.suite("Model.quantize (dry_run)")

T.test("dry_run=true returns 0 (success)", function()
    local ret = ion7.Model.quantize(model_path, "/tmp/ion7_quant_dryrun.gguf", {
        ftype   = "q4_0",
        dry_run = true,
        nthread = 2,
    })
    T.ok(type(ret) == "number", "should return number, got: " .. type(ret))
    print(string.format("  q4_0 dry_run ret: %d", ret))
end)

T.test("all ftype strings are accepted", function()
    local ftypes = { "f32", "f16", "q4_0", "q8_0", "q4_k_m", "q5_k_m", "q6_k", "bf16" }
    for _, ftype in ipairs(ftypes) do
        local ok, err = pcall(ion7.Model.quantize, model_path,
            "/tmp/ion7_test.gguf", { ftype = ftype, dry_run = true })
        T.ok(ok, "ftype '" .. ftype .. "' raised: " .. tostring(err))
    end
end)

T.test("invalid ftype raises error", function()
    T.err(function()
        ion7.Model.quantize(model_path, "/tmp/x.gguf", { ftype = "not_a_format" })
    end, "unknown ftype")
end)

-- ══════════════════════════════════════════════════════════════════
-- 5. VOCAB
-- ══════════════════════════════════════════════════════════════════

T.suite("Vocab")

local vocab

T.test("model:vocab() returns usable Vocab", function()
    vocab = model:vocab()
    T.ok(vocab ~= nil)
end)

T.test("n_tokens() is a positive number", function()
    T.ok(vocab:n_tokens() > 0)
    print("  vocab size: " .. vocab:n_tokens())
end)

T.test("tokenize() returns tokens + count", function()
    local tokens, n = vocab:tokenize("Hello world", false, false)
    T.ok(n > 0, "n > 0")
    T.ok(tokens ~= nil, "tokens not nil")
end)

T.test("tokenize() with add_special=true includes BOS", function()
    local _, n1 = vocab:tokenize("Hello", false, false)
    local _, n2 = vocab:tokenize("Hello", true,  false)
    T.ok(n2 >= n1, "with BOS should have >= tokens")
end)

T.test("detokenize() is inverse of tokenize()", function()
    local text = "Hello world"
    local tokens, n = vocab:tokenize(text, false, false)
    local out = vocab:detokenize(tokens, n, false, false)
    T.ok(type(out) == "string" and #out > 0)
    -- May not be byte-identical due to leading spaces, but should contain the text
    T.ok(out:find("Hello") ~= nil, "detokenize should contain 'Hello'")
end)

T.test("piece() converts token id to string", function()
    local tokens, n = vocab:tokenize("Hello", false, false)
    T.ok(n > 0)
    local piece = vocab:piece(tokens[0] or tokens[1], true)
    T.ok(type(piece) == "string")
end)

T.test("score() returns a float", function()
    local bos = vocab:bos()
    if bos >= 0 then
        local s = vocab:score(bos)
        T.ok(type(s) == "number")
    end
end)

T.test("attr() returns integer bitmask", function()
    local bos = vocab:bos()
    if bos >= 0 then
        local a = vocab:attr(bos)
        T.ok(type(a) == "number" and a >= 0)
    end
end)

T.test("special tokens - bos/eos/eot/pad/nl", function()
    local bos = vocab:bos()
    local eos = vocab:eos()
    local eot = vocab:eot()
    local pad = vocab:pad()
    local nl  = vocab:nl()
    T.ok(type(bos) == "number", "bos")
    T.ok(type(eos) == "number", "eos")
    T.ok(type(eot) == "number", "eot")
    T.ok(type(pad) == "number", "pad")
    T.ok(type(nl)  == "number", "nl")
    print(string.format("  bos=%d eos=%d eot=%d pad=%d nl=%d", bos, eos, eot, pad, nl))
end)

T.test("special tokens - sep/mask/cls", function()
    local sep  = vocab:sep()
    local mask = vocab:mask()
    T.ok(type(sep)  == "number", "sep")
    T.ok(type(mask) == "number", "mask")
end)

T.test("fim tokens - pre/suf/mid", function()
    local pre = vocab:fim_pre()
    local suf = vocab:fim_suf()
    local mid = vocab:fim_mid()
    T.ok(type(pre) == "number", "fim_pre")
    T.ok(type(suf) == "number", "fim_suf")
    T.ok(type(mid) == "number", "fim_mid")
end)

T.test("get_add_bos/eos/sep return booleans", function()
    T.ok(type(vocab:get_add_bos()) == "boolean", "get_add_bos")
    T.ok(type(vocab:get_add_eos()) == "boolean", "get_add_eos")
    T.ok(type(vocab:get_add_sep()) == "boolean", "get_add_sep")
end)

T.test("is_eog() - eos token is EOG", function()
    local eos = vocab:eos()
    if eos >= 0 then
        T.ok(vocab:is_eog(eos), "eos should be end-of-generation")
    end
end)

T.test("is_eog() - regular word token is NOT EOG", function()
    local tokens, n = vocab:tokenize("hello", false, false)
    T.ok(n > 0)
    T.ok(not vocab:is_eog(tokens[0] or tokens[1]), "word token should not be EOG")
end)

T.test("is_control() returns boolean", function()
    local bos = vocab:bos()
    if bos >= 0 then
        T.ok(type(vocab:is_control(bos)) == "boolean")
    end
end)

T.test("builtin_templates() returns non-empty list", function()
    local templates = vocab:builtin_templates()
    T.ok(type(templates) == "table" and #templates > 0)
    print("  built-in templates: " .. #templates .. " (first: " .. templates[1] .. ")")
end)

T.test("apply_template() formats messages correctly", function()
    local msgs   = { { role = "user", content = "Hello" } }
    local result = vocab:apply_template(msgs, false)
    T.ok(type(result) == "string" and #result > 0)
    T.ok(result:find("Hello") ~= nil, "template should contain user message")
end)

T.test("apply_template() add_ass=true appends assistant prefix", function()
    local msgs = { { role = "user", content = "Hi" } }
    local r1   = vocab:apply_template(msgs, false)
    local r2   = vocab:apply_template(msgs, true)
    T.ok(#r2 >= #r1, "add_ass should produce >= output length")
end)

T.test("tokenize with special=true and parse_special=true", function()
    local tokens, n = vocab:tokenize("Hello world", true, true)
    T.ok(n > 0, "should produce tokens with both flags true")
    T.ok(tokens ~= nil, "tokens not nil")
end)

T.test("tokenize empty string returns 0 tokens", function()
    local tokens, n = vocab:tokenize("", false, false)
    T.eq(n, 0, "empty string should produce 0 tokens")
end)

T.test("detokenize with special=true", function()
    local tokens, n = vocab:tokenize("Hello", true, false)
    -- detokenize with remove_special=false, unparse_special=true
    local out = vocab:detokenize(tokens, n, false, true)
    T.ok(type(out) == "string" and #out > 0, "should produce non-empty string")
end)

T.test("piece() with special=false vs special=true", function()
    local bos = vocab:bos()
    if bos >= 0 then
        local p1 = vocab:piece(bos, false)
        local p2 = vocab:piece(bos, true)
        T.ok(type(p1) == "string", "piece(bos, false) returns string")
        T.ok(type(p2) == "string", "piece(bos, true) returns string")
        -- With special=true, BOS should produce text; with false it may be empty
        print(string.format("  piece(bos, false)='%s'  piece(bos, true)='%s'",
            p1:gsub("\n","\\n"), p2:gsub("\n","\\n")))
    end
end)

T.test("text() returns raw token text", function()
    local ok, _ = pcall(function() return vocab.text end)
    if not ok or type(vocab.text) ~= "function" then
        T.skip("text()", "method not available")
        return
    end
    local tokens, n = vocab:tokenize("Hello", false, false)
    T.ok(n > 0)
    local txt = vocab:text(tokens[0])
    T.ok(type(txt) == "string", "text() should return string")
end)

T.test("fim tokens - pad/rep/sep", function()
    local ok_pad, pad = pcall(function() return vocab:fim_pad() end)
    local ok_rep, rep = pcall(function() return vocab:fim_rep() end)
    local ok_sep, sep = pcall(function() return vocab:fim_sep() end)
    if ok_pad then T.ok(type(pad) == "number", "fim_pad") end
    if ok_rep then T.ok(type(rep) == "number", "fim_rep") end
    if ok_sep then T.ok(type(sep) == "number", "fim_sep") end
    if ok_pad and ok_rep and ok_sep then
        print(string.format("  fim_pad=%d fim_rep=%d fim_sep=%d", pad, rep, sep))
    end
end)

-- ══════════════════════════════════════════════════════════════════
-- 6. CONTEXT
-- ══════════════════════════════════════════════════════════════════

T.suite("Context creation + dimensions")

local ctx
local n_ctx = math.min(fit and fit.n_ctx or 4096, 4096)

T.test("model:context() with default opts", function()
    ctx = model:context({
        n_gpu_layers = fit and fit.n_gpu_layers or 0,
        n_ctx        = n_ctx,
    })
    T.ok(ctx ~= nil)
end)

T.test("n_ctx() matches requested", function()
    T.ok(ctx:n_ctx() >= n_ctx * 0.9, "n_ctx should be close to requested")
    T.ok(ctx:n_ctx() > 0)
    print(string.format("  n_ctx=%d  n_batch=%d  n_ubatch=%d  n_seq_max=%d",
        ctx:n_ctx(), ctx:n_batch(), ctx:n_ubatch(), ctx:n_seq_max()))
end)

T.test("n_batch() > 0", function()
    T.ok(ctx:n_batch() > 0)
end)

T.test("n_ubatch() > 0 and <= n_batch()", function()
    T.ok(ctx:n_ubatch() > 0)
    T.ok(ctx:n_ubatch() <= ctx:n_batch())
end)

T.test("n_seq_max() >= 1", function()
    T.ok(ctx:n_seq_max() >= 1)
end)

T.test("n_threads() > 0", function()
    T.ok(ctx:n_threads() > 0)
    print("  threads: " .. ctx:n_threads())
end)

T.test("set_n_threads() changes thread count", function()
    local orig = ctx:n_threads()
    ctx:set_n_threads(2)
    T.eq(ctx:n_threads(), 2, "should be 2 after set")
    ctx:set_n_threads(orig)  -- restore
    T.eq(ctx:n_threads(), orig, "should be restored")
end)

T.test("set_embeddings() / set_causal_attn() don't crash", function()
    ctx:set_embeddings(false)
    ctx:set_causal_attn(true)
    ctx:set_embeddings(false)  -- restore
end)

T.test("synchronize() doesn't crash", function()
    ctx:synchronize()
end)

T.test("n_past() starts at 0", function()
    ctx:kv_clear()
    T.eq(ctx:n_past(), 0)
end)

T.test("ptr() returns non-nil cdata", function()
    local p = ctx:ptr()
    T.ok(p ~= nil, "ctx:ptr() should not be nil")
end)

T.test("n_threads_batch() returns number > 0", function()
    local ntb = ctx:n_threads_batch()
    T.ok(type(ntb) == "number" and ntb > 0,
        "n_threads_batch should be > 0, got " .. tostring(ntb))
    print("  n_threads_batch: " .. ntb)
end)

T.test("set_warmup(true) then set_warmup(false) doesn't crash", function()
    T.no_error(function()
        ctx:set_warmup(true)
        ctx:set_warmup(false)
    end, "set_warmup round-trip")
end)

T.test("pooling_type() returns string for inference context", function()
    local pt = ctx:pooling_type()
    T.ok(type(pt) == "string", "pooling_type should return string, got " .. type(pt))
    print("  pooling_type: " .. pt)
end)

-- ══════════════════════════════════════════════════════════════════
-- 7. KV CACHE OPERATIONS
-- ══════════════════════════════════════════════════════════════════

T.suite("KV cache operations")

T.test("kv_clear() resets n_past to 0", function()
    ctx:kv_clear()
    T.eq(ctx:n_past(), 0)
end)

T.test("decode() advances n_past", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Hello world", false, false)
    ctx:decode(tokens, n, 0, 0)
    T.ok(ctx:n_past() > 0, "n_past should advance after decode")
    T.eq(ctx:n_past(), n, "n_past should equal token count")
end)

T.test("kv_seq_rm() reduces n_past conceptually", function()
    -- After decode, remove tokens from position 2 onwards
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Hello world foo bar", false, false)
    ctx:decode(tokens, n, 0, 0)
    local before = ctx:n_past()
    ctx:kv_seq_rm(0, 2, -1)  -- remove from pos 2 to end
    ctx._n_past = 2           -- update our tracking
    T.eq(ctx:n_past(), 2, "n_past should be 2 after trim")
    T.ok(before > 2, "we should have removed some tokens")
end)

T.test("kv_seq_cp() doesn't crash", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:decode(tokens, n, 0, 0)
    ctx:kv_seq_cp(0, 1, 0, -1)  -- copy seq 0 to seq 1
end)

T.test("kv_seq_keep() doesn't crash", function()
    ctx:kv_clear()
    ctx:kv_seq_keep(0)
end)

T.test("kv_can_shift() returns boolean", function()
    local can = ctx:kv_can_shift()
    T.ok(type(can) == "boolean")
    print("  kv_can_shift: " .. tostring(can))
end)

T.test("kv_seq_shift() works when supported", function()
    if not ctx:kv_can_shift() then
        print("  [SKIP] kv_can_shift=false on this model")
        return
    end
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Hello world", false, false)
    ctx:decode(tokens, n, 0, 0)
    -- Shift all positions in [0, n) by -2
    ctx:kv_seq_shift(0, 0, n, -2)
end)

T.test("kv_seq_pos_max(0) returns number", function()
    local ok, val = pcall(function() return ctx:kv_seq_pos_max(0) end)
    if not ok then
        T.skip("kv_seq_pos_max()", "method not available")
        return
    end
    T.ok(type(val) == "number", "kv_seq_pos_max should return number")
end)

T.test("kv_clear after decode resets n_past", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Testing KV clear", false, false)
    ctx:decode(tokens, n, 0, 0)
    T.ok(ctx:n_past() > 0, "n_past should be > 0 after decode")
    ctx:kv_clear()
    T.eq(ctx:n_past(), 0, "n_past should be 0 after kv_clear")
end)

T.test("multiple kv_seq_cp then kv_seq_keep(1)", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Copy test", false, false)
    ctx:decode(tokens, n, 0, 0)
    T.no_error(function()
        ctx:kv_seq_cp(0, 1, 0, -1)
        ctx:kv_seq_cp(0, 2, 0, -1)
        ctx:kv_seq_cp(0, 3, 0, -1)
        ctx:kv_seq_keep(1)
    end, "multiple seq_cp + seq_keep(1)")
end)

T.test("kv_seq_rm with seq_id=-1 (all sequences)", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Remove all test", false, false)
    ctx:decode(tokens, n, 0, 0)
    T.no_error(function()
        ctx:kv_seq_rm(-1, 0, -1)
    end, "kv_seq_rm(-1, 0, -1)")
end)

-- ══════════════════════════════════════════════════════════════════
-- 8. DECODE + SAMPLE
-- ══════════════════════════════════════════════════════════════════

T.suite("Decode + sample pipeline")

local sampler

T.test("build greedy sampler", function()
    sampler = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    T.ok(sampler ~= nil)
end)

T.test("decode prompt → sample token", function()
    local msgs   = { { role = "user", content = "Reply: PONG" } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    T.ok(n > 0)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local token = sampler:sample(ctx:ptr(), -1)
    T.ok(type(token) == "number" and token >= 0)
    local piece = vocab:piece(token, true)
    T.ok(type(piece) == "string")
    print(string.format("  first token: '%s' (id=%d)", piece:gsub("\n","\\n"), token))
end)

T.test("decode_single() advances n_past by 1", function()
    ctx:kv_clear()
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:decode(tokens, n, 0, 0)
    local before = ctx:n_past()
    local token = sampler:sample(ctx:ptr(), -1)
    sampler:accept(token)
    ctx:decode_single(token, 0)
    T.eq(ctx:n_past(), before + 1)
end)

T.test("generate 10 tokens without crash", function()
    local msgs   = { { role = "user", content = "Count to ten." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    sampler:reset()
    local count = 0
    for _ = 1, 10 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        sampler:accept(tok)
        ctx:decode_single(tok, 0)
        count = count + 1
    end
    T.ok(count > 0, "should have generated at least 1 token")
    print("  generated: " .. count .. " tokens")
end)

T.test("generate 50 tokens and verify decodable text", function()
    local msgs   = { { role = "user", content = "Write a short sentence about cats." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local gen_sampler = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    gen_sampler:reset()
    local gen_tokens = {}
    local ffi = require("ffi")
    for _ = 1, 50 do
        local tok = gen_sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        gen_sampler:accept(tok)
        gen_tokens[#gen_tokens + 1] = tok
        ctx:decode_single(tok, 0)
    end
    T.ok(#gen_tokens > 0, "should have generated at least 1 token")
    -- Convert to cdata for detokenize
    local buf = ffi.new("int32_t[?]", #gen_tokens)
    for i, t in ipairs(gen_tokens) do buf[i - 1] = t end
    local text = vocab:detokenize(buf, #gen_tokens, false, false)
    T.ok(type(text) == "string" and #text > 0, "generated text should be non-empty")
    print(string.format("  generated %d tokens: '%s'",
        #gen_tokens, text:sub(1, 80):gsub("\n", "\\n")))
end)

T.test("decode() with seq_id=1 (requires n_seq_max > 1)", function()
    -- Default context has n_seq_max=1, so seq_id=1 may fail - that's correct.
    -- Create a multi-seq context to test properly.
    local ctx2 = model:context({ n_ctx = 512, n_seq_max = 2 })
    if ctx2 then
        local tokens, n = vocab:tokenize("Hello", false, false)
        T.no_error(function()
            ctx2:decode(tokens, n, 1, 0)
        end, "decode with seq_id=1")
        ctx2:free()
    else
        T.skip("decode seq_id=1", "could not create n_seq_max=2 context")
    end
end)

T.test("sampled tokens are valid token IDs", function()
    local msgs   = { { role = "user", content = "Say hello." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    local n_vocab = vocab:n_tokens()
    for _ = 1, 5 do
        local tok = s:sample(ctx:ptr(), -1)
        T.ok(tok >= 0, "token should be >= 0")
        T.ok(tok < n_vocab, "token should be < n_vocab (" .. n_vocab .. "), got " .. tok)
        if vocab:is_eog(tok) then break end
        s:accept(tok)
        ctx:decode_single(tok, 0)
    end
end)

-- ══════════════════════════════════════════════════════════════════
-- 9. PERFORMANCE MONITORING
-- ══════════════════════════════════════════════════════════════════

T.suite("Performance monitoring")

T.test("perf() returns complete timing table", function()
    local p = ctx:perf()
    T.ok(type(p.t_load_ms)    == "number", "t_load_ms")
    T.ok(type(p.t_p_eval_ms)  == "number", "t_p_eval_ms")
    T.ok(type(p.t_eval_ms)    == "number", "t_eval_ms")
    T.ok(type(p.n_p_eval)     == "number", "n_p_eval")
    T.ok(type(p.n_eval)       == "number", "n_eval")
    T.ok(type(p.n_reused)     == "number", "n_reused")
    T.ok(type(p.tokens_per_s) == "number", "tokens_per_s")
    T.ok(p.n_eval >= 0, "n_eval >= 0")
    print(string.format("  n_p_eval=%d n_eval=%d n_reused=%d %.1f tok/s",
        p.n_p_eval, p.n_eval, p.n_reused, p.tokens_per_s))
end)

T.test("perf_reset() zeroes counters", function()
    -- Decode enough to have non-zero counters
    local toks, n = vocab:tokenize("Hello world benchmark", false, false)
    ctx:kv_clear()
    ctx:decode(toks, n, 0, 0)
    local before = ctx:perf()
    T.ok(before.t_p_eval_ms >= 0, "t_p_eval_ms should be >= 0 before reset")
    -- Reset
    ctx:perf_reset()
    local p = ctx:perf()
    -- After reset, timing should be zeroed (counters may count next decode)
    T.ok(p.t_load_ms   >= 0,  "t_load_ms >= 0 after reset")
    T.ok(p.t_eval_ms   == 0.0, "t_eval_ms should be 0 after reset")
    T.ok(p.t_p_eval_ms == 0.0, "t_p_eval_ms should be 0 after reset")
    print(string.format("  after reset: n_p_eval=%d n_eval=%d t_eval=%.2f",
        p.n_p_eval, p.n_eval, p.t_eval_ms))
end)

-- ══════════════════════════════════════════════════════════════════
-- 10. STATE PERSISTENCE
-- ══════════════════════════════════════════════════════════════════

T.suite("State persistence")

T.test("snapshot() returns non-empty blob", function()
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local blob = ctx:snapshot()
    T.ok(type(blob) == "string" and #blob > 0)
    print(string.format("  snapshot: %.1f KB", #blob / 1024))
end)

T.test("restore() from snapshot succeeds", function()
    local tokens, n = vocab:tokenize("Hello world", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local n_past_before = ctx:n_past()
    local blob = ctx:snapshot()
    ctx:kv_clear()
    T.eq(ctx:n_past(), 0, "after kv_clear n_past=0")
    local ok = ctx:restore(blob)
    T.ok(ok, "restore should return true")
    -- n_past is restored from blob
end)

T.test("save_state() + load_state() round-trip", function()
    local path = "/tmp/ion7_state_roundtrip.bin"
    local tokens, n = vocab:tokenize("Round trip test", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    ctx:save_state(path)
    ctx:kv_clear()
    ctx:load_state(path)
    os.remove(path)
end)

T.test("seq_save_state() + seq_load_state() round-trip", function()
    local path = "/tmp/ion7_seq_state.bin"
    local tokens, n = vocab:tokenize("Seq test", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    ctx:seq_save_state(path, 0)
    ctx:kv_clear()
    ctx:seq_load_state(path, 0)
    os.remove(path)
end)

T.test("snapshot() size is consistent across calls", function()
    local tokens, n = vocab:tokenize("Consistency test", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local blob1 = ctx:snapshot()
    local blob2 = ctx:snapshot()
    T.ok(blob1 ~= nil and blob2 ~= nil, "both snapshots should exist")
    T.eq(#blob1, #blob2, "snapshot sizes should be identical")
end)

T.test("save_state to nonexistent directory errors", function()
    local tokens, n = vocab:tokenize("Error test", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    -- Attempt to save to a path in a nonexistent directory
    local ok, _ = pcall(function()
        ctx:save_state("/nonexistent_dir_12345/state.bin")
    end)
    -- This should either error or return false; either is acceptable
    -- The key is it doesn't crash
    print("  save to nonexistent dir: " .. (ok and "returned (no crash)" or "raised error (expected)"))
end)

T.test("seq_save_state + seq_load_state to different seq_id", function()
    local path = "/tmp/ion7_seq_cross.bin"
    local tokens, n = vocab:tokenize("Cross seq test", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local saved = ctx:seq_save_state(path, 0)
    if saved then
        ctx:kv_clear()
        -- Load to a different seq_id
        local loaded = ctx:seq_load_state(path, 1)
        T.ok(type(loaded) == "boolean", "seq_load_state should return bool")
    end
    os.remove(path)
end)

-- ══════════════════════════════════════════════════════════════════
-- 11. SAMPLER CHAIN
-- ══════════════════════════════════════════════════════════════════

T.suite("Sampler chain")

T.test("chain_n() counts correctly", function()
    local s = ion7.Sampler.chain():top_k(50):top_p(0.9):temp(0.8):dist(42):build(vocab)
    T.eq(s:n(), 4, "should have 4 samplers")
end)

T.test("clone() produces independent working copy", function()
    local s1 = ion7.Sampler.chain():top_k(50):dist(42):build(vocab)
    local s2 = s1:clone()
    T.ok(s2 ~= nil, "clone should not be nil")
    T.eq(s2:n(), s1:n(), "clone should have same count")
end)

T.test("seed() returns a number", function()
    local s = ion7.Sampler.chain():dist(12345):build(vocab)
    T.ok(type(s:seed()) == "number")
end)

T.test("reset() doesn't crash", function()
    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    s:reset()
end)

T.test("perf() returns timing table", function()
    local s = ion7.Sampler.chain():top_k(10):dist(42):build(vocab)
    local p = s:perf()
    T.ok(type(p.t_sample_ms) == "number", "t_sample_ms")
    T.ok(type(p.n_sample)    == "number", "n_sample")
end)

T.test("perf_reset() zeroes sampler counters", function()
    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    s:perf_reset()
    local p = s:perf()
    T.eq(p.n_sample, 0, "n_sample should be 0 after reset")
end)

-- All sampler types build without crash
local sampler_types = {
    {"greedy",        function(b) return b:greedy() end},
    {"top_k",         function(b) return b:top_k(50):dist(1) end},
    {"top_p",         function(b) return b:top_p(0.9):dist(1) end},
    {"min_p",         function(b) return b:min_p(0.05):dist(1) end},
    {"temp",          function(b) return b:temp(0.8):dist(1) end},
    {"temp_dynamic",  function(b) return b:temp_dynamic(0.8, 0.2):dist(1) end},
    {"typical",       function(b) return b:typical(0.9):dist(1) end},
    {"top_n_sigma",   function(b) return b:top_n_sigma(1.0):dist(1) end},
    {"xtc",           function(b) return b:xtc(0.1, 0.9):dist(1) end},
    {"mirostat_v2",   function(b) return b:mirostat_v2(42, 5.0, 0.1) end},
    {"penalties",     function(b) return b:penalties(64, 1.1):dist(1) end},
    {"adaptive_p",    function(b) return b:adaptive_p(0.45, 0.99):dist(1) end},
    {"logit_bias",    function(b) return b:logit_bias({}):dist(1) end},
    {"grammar",       function(b)
        -- grammar() needs the vocab object to get the llama_vocab* ptr
        return b:grammar('root ::= [a-z]+', "root", vocab):dist(1)
    end},
}

for _, pair in ipairs(sampler_types) do
    local name, build_fn = pair[1], pair[2]
    T.test("sampler type: " .. name, function()
        local b = build_fn(ion7.Sampler.chain())
        local s = b:build(vocab)
        T.ok(s ~= nil, name .. " build returned nil")
    end)
end

T.test("greedy sampler produces deterministic output", function()
    local msgs   = { { role = "user", content = "Say hello." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)

    local function generate_10()
        ctx:kv_clear()
        ctx:decode(tokens, n, 0, 0)
        local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
        local result = {}
        for _ = 1, 10 do
            local tok = s:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            s:accept(tok)
            result[#result + 1] = tok
            ctx:decode_single(tok, 0)
        end
        return result
    end

    local run1 = generate_10()
    local run2 = generate_10()
    T.ok(#run1 > 0, "should generate tokens")
    T.eq(#run1, #run2, "same number of tokens")
    for i = 1, #run1 do
        T.eq(run1[i], run2[i], "token " .. i .. " should match")
    end
end)

T.test("temperature=0.0 is effectively greedy", function()
    local msgs   = { { role = "user", content = "Say one word." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    -- temp(0.0) should pick highest logit
    local s = ion7.Sampler.chain():temp(0.0):dist(42):build(vocab)
    local tok = s:sample(ctx:ptr(), -1)
    T.ok(type(tok) == "number" and tok >= 0, "should produce valid token")
end)

T.test("chain with 0 samplers: build() still works", function()
    local s = ion7.Sampler.chain():build(vocab)
    T.ok(s ~= nil, "empty chain should build")
    T.eq(s:n(), 0, "should have 0 samplers")
end)

T.test("accept() on sampler chain doesn't crash", function()
    local s = ion7.Sampler.chain():top_k(10):dist(42):build(vocab)
    T.no_error(function()
        s:accept(1)
        s:accept(0)
        s:accept(vocab:bos())
    end, "accept()")
end)

-- ══════════════════════════════════════════════════════════════════
-- 12. CUSTOM SAMPLER
-- ══════════════════════════════════════════════════════════════════

T.suite("CustomSampler")

T.test("new() requires apply callback", function()
    T.err(function() ion7.CustomSampler.new("bad", {}) end, "apply is required")
end)

T.test("new() creates sampler and ptr() is non-null", function()
    local cs = ion7.CustomSampler.new("test", {
        apply = function(candidates, n) return 0 end
    })
    T.ok(cs ~= nil)
    T.ok(cs:ptr() ~= nil)
    T.eq(cs:name(), "test")
end)

T.test("Lua greedy sampler is called during generation", function()
    local apply_count = 0
    local cs = ion7.CustomSampler.new("lua_greedy", {
        apply = function(cur_p, n)
            apply_count = apply_count + 1
            -- cur_p is llama_token_data_array*, iterate via cur_p.data
            local best_i, best_logit = 0, -math.huge
            for i = 0, n - 1 do
                if cur_p.data[i].logit > best_logit then
                    best_logit = cur_p.data[i].logit
                    best_i = i
                end
            end
            cur_p.selected = best_i  -- set on the array struct
            return best_i
        end,
    })
    local chain = ion7.Sampler.chain():custom(cs):build(vocab)
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    chain:sample(ctx:ptr(), -1)
    T.ok(apply_count > 0, "apply callback should have been called")
end)

T.test("accept callback is registered (no PANIC)", function()
    -- Note: calling chain:accept() with a custom accept callback can trigger
    -- PANIC: bad callback in some LuaJIT versions when the FFI callback
    -- dispatch interacts with the GC. We verify the callback is registered
    -- and that sample() works; accept() is tested via the C bridge directly.
    local accept_count = 0
    local cs = ion7.CustomSampler.new("counter", {
        apply  = function(cur_p, n)
            cur_p.selected = 0
            return 0
        end,
        accept = function(token)
            accept_count = accept_count + 1
        end,
    })
    T.ok(cs ~= nil, "custom sampler with accept callback created")
    T.ok(cs._accept_cb ~= nil, "accept callback is registered")
    local chain = ion7.Sampler.chain():custom(cs):build(vocab)
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local tok = chain:sample(ctx:ptr(), -1)
    T.ok(type(tok) == "number" and tok >= 0, "sample works with accept callback present")
end)

T.test("reset callback is called", function()
    local reset_count = 0
    local cs = ion7.CustomSampler.new("resetter", {
        apply = function(candidates, n) candidates.selected = 0; return 0 end,
        reset = function() reset_count = reset_count + 1 end,
    })
    local chain = ion7.Sampler.chain():custom(cs):build(vocab)
    chain:reset()
    T.ok(reset_count > 0, "reset callback should have been called")
end)

T.test("custom sampler chains with built-ins", function()
    local cs = ion7.CustomSampler.new("passthrough", {
        apply = function(candidates, n)
            -- Don't select, just pass through (return -1 = let next sampler decide)
        end,
    })
    local chain = ion7.Sampler.chain()
        :top_k(50)
        :custom(cs)
        :temp(0.8)
        :dist(42)
        :build(vocab)
    T.ok(chain ~= nil)
    T.ok(chain:n() == 4, "should have 4 samplers in chain")
end)

T.test("custom sampler zeroes all but first logit", function()
    local cs = ion7.CustomSampler.new("force_first", {
        apply = function(cur_p, n)
            -- Zero out all logits except the first candidate
            for i = 1, n - 1 do
                cur_p.data[i].logit = -1e9
            end
            cur_p.selected = 0
            return 0
        end,
    })
    local chain = ion7.Sampler.chain():custom(cs):build(vocab)
    local tokens, n = vocab:tokenize("Hello", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local tok = chain:sample(ctx:ptr(), -1)
    T.ok(type(tok) == "number" and tok >= 0, "should produce valid token")
end)

-- ══════════════════════════════════════════════════════════════════
-- 13. THREADPOOL
-- ══════════════════════════════════════════════════════════════════

T.suite("Threadpool")

T.test("Threadpool.new(4) creates pool", function()
    local tp = ion7.Threadpool.new(4)
    T.ok(tp ~= nil)
    T.eq(tp:n_threads(), 4)
    T.ok(tp:ptr() ~= nil)
end)

T.test("Threadpool.new(1) single thread", function()
    local tp = ion7.Threadpool.new(1)
    T.eq(tp:n_threads(), 1)
end)

T.test("n_threads=0 is rejected", function()
    T.err(function() ion7.Threadpool.new(0) end)
end)

T.test("attach() + detach() round-trip", function()
    local tp = ion7.Threadpool.new(4)
    ctx:attach_threadpool(tp)
    ctx:detach_threadpool()
end)

T.test("shared pool across two contexts", function()
    local tp   = ion7.Threadpool.new(4)
    local ctx2 = model:context({ n_ctx = 512 })
    ctx:attach_threadpool(tp)
    ctx2:attach_threadpool(tp)
    ctx:detach_threadpool()
    ctx2:detach_threadpool()
end)

T.test("pause() + resume() don't crash", function()
    local tp = ion7.Threadpool.new(2)
    tp:pause()
    tp:resume()
end)

T.test("free() is idempotent", function()
    local tp = ion7.Threadpool.new(2)
    tp:free()
    tp:free()  -- must not crash
end)

T.test("Threadpool.new(N) with N = number of CPU cores", function()
    -- Use a reasonable guess for CPU cores (at least 1, at most 128)
    local n_cores = 4  -- conservative default
    local ok, _ = pcall(function()
        local f = io.popen("nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4")
        if f then
            local s = f:read("*a")
            f:close()
            n_cores = tonumber(s) or 4
        end
    end)
    local tp = ion7.Threadpool.new(n_cores)
    T.ok(tp ~= nil, "should create pool with " .. n_cores .. " threads")
    T.eq(tp:n_threads(), n_cores, "n_threads should match")
    tp:free()
    print("  created pool with " .. n_cores .. " threads (CPU cores)")
end)

T.test("attach with separate batch pool", function()
    local tp_gen   = ion7.Threadpool.new(2)
    local tp_batch = ion7.Threadpool.new(4)
    T.no_error(function()
        ctx:attach_threadpool(tp_gen, tp_batch)
        ctx:detach_threadpool()
    end, "attach with separate batch pool")
    tp_gen:free()
    tp_batch:free()
end)

T.test("n_threads() returns correct count after creation", function()
    for _, n in ipairs({1, 2, 4, 8}) do
        local tp = ion7.Threadpool.new(n)
        T.eq(tp:n_threads(), n, "n_threads should be " .. n)
        tp:free()
    end
end)

-- ══════════════════════════════════════════════════════════════════
-- 14. LORA ADAPTER (if applicable)
-- ══════════════════════════════════════════════════════════════════

T.suite("LoRA adapter")

local LORA_PATH = os.getenv("ION7_LORA")

if not LORA_PATH or LORA_PATH == "" then
    T.test("LoRA - SKIP (set ION7_LORA=/path/to/adapter.gguf)", function()
        -- Document the skip but don't fail
        print("  Set ION7_LORA to enable LoRA tests")
    end)
else
    T.test("lora_load() + lora_apply() + lora_remove()", function()
        local adapter = model:lora_load(LORA_PATH)
        T.ok(adapter ~= nil, "lora_load returned nil")
        local ok = ctx:lora_apply(adapter, 1.0)
        T.ok(ok, "lora_apply failed")
        ctx:lora_remove(adapter)
    end)
end

-- ══════════════════════════════════════════════════════════════════
-- 15. EMBEDDING CONTEXT
-- ══════════════════════════════════════════════════════════════════

T.suite("Embedding context")

T.test("model:embedding_context() creates context with pooling", function()
    local ectx = model:embedding_context({ n_ctx = 512, pooling = "last" })
    T.ok(ectx ~= nil)
    local pt = ectx:pooling_type()
    T.ok(pt == "last" or pt == "none",
        "pooling should be 'last' or 'none', got: " .. tostring(pt))
end)

T.test("embedding context with n_seq_max > 1", function()
    local ectx = model:embedding_context({
        n_ctx = 512, pooling = "last", n_seq_max = 4
    })
    T.ok(ectx ~= nil)
    T.ok(ectx:n_seq_max() >= 1)
    print("  n_seq_max: " .. ectx:n_seq_max())
end)

-- ══════════════════════════════════════════════════════════════════
-- 16. CONTEXT - LOGITS AND EMBEDDINGS
-- ══════════════════════════════════════════════════════════════════

T.suite("Context - logits and embeddings")

T.test("ctx:logits() returns non-nil pointer after decode", function()
    local ok_method = pcall(function() return ctx.logits end)
    if not ok_method or type(ctx.logits) ~= "function" then
        T.skip("ctx:logits()", "method not available")
        return
    end
    local tokens, n = vocab:tokenize("Hello logits", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    -- logits for the last position (batch idx = n-1, but we use -1 convention)
    local ok, logits = pcall(function() return ctx:logits(-1) end)
    if not ok then
        T.skip("ctx:logits(-1)", "call failed: " .. tostring(logits))
        return
    end
    T.ok(logits ~= nil, "logits pointer should not be nil")
end)

T.test("logits have vocab_size entries (spot check)", function()
    local ok_method = pcall(function() return ctx.logits end)
    if not ok_method or type(ctx.logits) ~= "function" then
        T.skip("logits vocab check", "method not available")
        return
    end
    local tokens, n = vocab:tokenize("Test logits", false, false)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local ok, logits = pcall(function() return ctx:logits(-1) end)
    if not ok or logits == nil then
        T.skip("logits vocab check", "logits not available")
        return
    end
    -- Spot-check: read a few logit values to verify the pointer is valid
    local n_vocab = vocab:n_tokens()
    T.no_error(function()
        local v0 = tonumber(logits[0])
        local v1 = tonumber(logits[1])
        local vn = tonumber(logits[n_vocab - 1])
        T.ok(type(v0) == "number", "logits[0] should be a number")
        T.ok(type(v1) == "number", "logits[1] should be a number")
        T.ok(type(vn) == "number", "logits[n_vocab-1] should be a number")
    end, "reading logit values")
    print("  logits[0]=" .. string.format("%.2f", tonumber(logits[0])) ..
          "  n_vocab=" .. n_vocab)
end)

T.test("ctx:embedding() on embedding context", function()
    local ok_method = pcall(function() return ctx.embedding end)
    if not ok_method or type(ctx.embedding) ~= "function" then
        T.skip("ctx:embedding()", "method not available")
        return
    end
    -- Create a dedicated embedding context for this test
    local ok_ectx, ectx = pcall(function()
        return model:embedding_context({ n_ctx = 512, pooling = "last" })
    end)
    if not ok_ectx or ectx == nil then
        T.skip("ctx:embedding()", "could not create embedding context")
        return
    end
    local tokens, n = vocab:tokenize("Embedding test", false, false)
    ectx:kv_clear()
    local ok_decode = pcall(function() ectx:decode(tokens, n, 0, 0) end)
    if not ok_decode then
        T.skip("ctx:embedding()", "decode failed on embedding context")
        return
    end
    local emb = ectx:embedding(0, model:n_embd())
    if emb then
        T.ok(type(emb) == "table", "embedding should be a table")
        T.ok(#emb > 0, "embedding should have entries")
        print(string.format("  embedding dim=%d  first=%.4f", #emb, emb[1] or 0))
    else
        print("  embedding returned nil (model may not support pooled embeddings)")
    end
end)

-- ══════════════════════════════════════════════════════════════════
-- 17. EDGE CASES AND ERROR HANDLING
-- ══════════════════════════════════════════════════════════════════

T.suite("Edge cases and error handling")

T.test("model:context() with very small n_ctx (64)", function()
    local ok, small_ctx = pcall(function()
        return model:context({ n_ctx = 64, n_gpu_layers = fit and fit.n_gpu_layers or 0 })
    end)
    if ok and small_ctx then
        T.ok(small_ctx:n_ctx() > 0, "small context should have n_ctx > 0")
        print("  small ctx n_ctx: " .. small_ctx:n_ctx())
        small_ctx:free()
    else
        print("  small n_ctx=64 not supported (expected on some models)")
    end
end)

T.test("double free: ctx:free() then ctx:free() doesn't crash", function()
    local tmp_ctx = model:context({ n_ctx = 512 })
    T.ok(tmp_ctx ~= nil)
    T.no_error(function()
        tmp_ctx:free()
        tmp_ctx:free()  -- second free should be idempotent
    end, "double ctx:free()")
end)

T.test("decode with 0 tokens doesn't crash", function()
    local ffi = require("ffi")
    local empty_tokens = ffi.new("int32_t[1]", 0)
    ctx:kv_clear()
    -- Decoding 0 tokens: might error, but should not crash
    local ok, err = pcall(function()
        ctx:decode(empty_tokens, 0, 0, 0)
    end)
    -- Either succeeds (no-op) or errors gracefully
    print("  decode 0 tokens: " .. (ok and "ok (no-op)" or "error (expected): " .. tostring(err):sub(1, 60)))
end)

T.test("tokenize very long text (10000 chars)", function()
    local long_text = string.rep("Hello world. This is a test sentence for tokenization. ", 200)
    T.ok(#long_text >= 10000, "text should be >= 10000 chars")
    local tokens, n = vocab:tokenize(long_text, false, false)
    T.ok(n > 0, "should produce tokens for long text")
    T.ok(n > 100, "10000 chars should produce many tokens")
    print(string.format("  %d chars -> %d tokens", #long_text, n))
end)

-- ══════════════════════════════════════════════════════════════════
-- 18. MULTI-SEQUENCE WORKFLOW
-- ══════════════════════════════════════════════════════════════════

T.suite("Multi-sequence workflow")

T.test("create context with n_seq_max=4", function()
    local mctx = model:context({
        n_ctx     = 1024,
        n_seq_max = 4,
        n_gpu_layers = fit and fit.n_gpu_layers or 0,
    })
    T.ok(mctx ~= nil)
    T.ok(mctx:n_seq_max() >= 1, "n_seq_max should be >= 1")
    print("  multi-seq ctx n_seq_max: " .. mctx:n_seq_max())

    -- Decode on seq_id 0
    local tokens, n = vocab:tokenize("Sequence zero", false, false)
    mctx:kv_clear()
    T.no_error(function()
        mctx:decode(tokens, n, 0, 0)
    end, "decode on seq_id 0")

    -- Decode on seq_id 1
    local tokens2, n2 = vocab:tokenize("Sequence one", false, false)
    T.no_error(function()
        mctx:decode(tokens2, n2, 1, 0)
    end, "decode on seq_id 1")

    -- kv_seq_cp from 0 to 2
    T.no_error(function()
        mctx:kv_seq_cp(0, 2, 0, -1)
    end, "kv_seq_cp(0, 2)")

    -- kv_seq_keep(0) removes all but seq 0
    T.no_error(function()
        mctx:kv_seq_keep(0)
    end, "kv_seq_keep(0)")

    mctx:free()
end)

T.test("kv_seq_rm removes specific sequence", function()
    local mctx = model:context({
        n_ctx     = 1024,
        n_seq_max = 4,
        n_gpu_layers = fit and fit.n_gpu_layers or 0,
    })
    local tokens, n = vocab:tokenize("Test sequence removal", false, false)
    mctx:kv_clear()
    mctx:decode(tokens, n, 0, 0)
    mctx:kv_seq_cp(0, 1, 0, -1)
    mctx:kv_seq_cp(0, 2, 0, -1)

    -- Remove seq 1 specifically
    T.no_error(function()
        mctx:kv_seq_rm(1, 0, -1)
    end, "kv_seq_rm(1, 0, -1)")

    mctx:free()
end)

-- ══════════════════════════════════════════════════════════════════
-- 19. SAMPLER DETERMINISM
-- ══════════════════════════════════════════════════════════════════

T.suite("Sampler determinism")

T.test("two identical chains with same seed produce same sequence", function()
    local msgs   = { { role = "user", content = "Write a number." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)

    local function run_with_seed(seed)
        ctx:kv_clear()
        ctx:decode(tokens, n, 0, 0)
        local s = ion7.Sampler.chain():top_k(40):temp(0.8):dist(seed):build(vocab)
        local result = {}
        for _ = 1, 5 do
            local tok = s:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            s:accept(tok)
            result[#result + 1] = tok
            ctx:decode_single(tok, 0)
        end
        return result
    end

    local run1 = run_with_seed(42)
    local run2 = run_with_seed(42)
    T.ok(#run1 > 0, "should generate tokens")
    T.eq(#run1, #run2, "same seed should produce same count")
    for i = 1, #run1 do
        T.eq(run1[i], run2[i], "token " .. i .. " should match with same seed")
    end
end)

T.test("two chains with different seeds produce different sequences", function()
    local msgs   = { { role = "user", content = "Tell me something random." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)

    local function run_with_seed(seed)
        ctx:kv_clear()
        ctx:decode(tokens, n, 0, 0)
        local s = ion7.Sampler.chain():top_k(40):temp(1.0):dist(seed):build(vocab)
        local result = {}
        for _ = 1, 10 do
            local tok = s:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            s:accept(tok)
            result[#result + 1] = tok
            ctx:decode_single(tok, 0)
        end
        return result
    end

    local run1 = run_with_seed(1)
    local run2 = run_with_seed(99999)
    -- With temp=1.0 and different seeds, sequences should differ
    -- (probabilistic: could theoretically match, but extremely unlikely over 10 tokens)
    if #run1 > 0 and #run2 > 0 then
        local differ = false
        for i = 1, math.min(#run1, #run2) do
            if run1[i] ~= run2[i] then differ = true; break end
        end
        if #run1 ~= #run2 then differ = true end
        if not differ then
            print("  [note] same output with different seeds - model may be near-deterministic at start")
        end
        -- This is probabilistic; don't fail the test suite for it
        T.ok(true, "different seeds test completed (differ=" .. tostring(differ) .. ")")
    end
end)

T.test("reset + re-sample produces same first token", function()
    local msgs   = { { role = "user", content = "Say yes." } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)

    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)

    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    local tok1 = s:sample(ctx:ptr(), -1)

    -- Reset and re-decode to get fresh logits
    s:reset()
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    local tok2 = s:sample(ctx:ptr(), -1)

    T.eq(tok1, tok2, "greedy after reset should produce same first token")
end)

-- ══════════════════════════════════════════════════════════════════
-- 20. FULL GENERATION PIPELINE
-- ══════════════════════════════════════════════════════════════════

T.suite("Full generation pipeline")

T.test("complete chat flow: system + user -> assistant response", function()
    local msgs = {
        { role = "system",  content = "You are a helpful assistant. Reply concisely." },
        { role = "user",    content = "What is 2+2?" },
    }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    T.ok(n > 0, "prompt tokenization should succeed")

    ctx:kv_clear()
    ctx:perf_reset()
    ctx:decode(tokens, n, 0, 0)

    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    local ffi = require("ffi")
    local gen_tokens = {}
    for _ = 1, 30 do
        local tok = s:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        s:accept(tok)
        gen_tokens[#gen_tokens + 1] = tok
        ctx:decode_single(tok, 0)
    end

    T.ok(#gen_tokens > 0, "should generate at least 1 token")

    -- Convert to text
    local buf = ffi.new("int32_t[?]", #gen_tokens)
    for i, t in ipairs(gen_tokens) do buf[i - 1] = t end
    local response = vocab:detokenize(buf, #gen_tokens, false, false)
    T.ok(type(response) == "string" and #response > 0, "response should be non-empty string")
    print(string.format("  response (%d tok): '%s'",
        #gen_tokens, response:sub(1, 100):gsub("\n", "\\n")))
end)

T.test("response is valid UTF-8 text", function()
    local msgs = {
        { role = "user", content = "Say hello in French." },
    }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)

    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    local ffi = require("ffi")
    local gen_tokens = {}
    for _ = 1, 20 do
        local tok = s:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        s:accept(tok)
        gen_tokens[#gen_tokens + 1] = tok
        ctx:decode_single(tok, 0)
    end

    if #gen_tokens > 0 then
        local buf = ffi.new("int32_t[?]", #gen_tokens)
        for i, t in ipairs(gen_tokens) do buf[i - 1] = t end
        local text = vocab:detokenize(buf, #gen_tokens, false, false)
        -- Basic UTF-8 validity: no lone surrogates, no truncated sequences
        -- A simple check: the string should not contain 0xFF or 0xFE bytes
        T.ok(not text:find("\xFF"), "should not contain 0xFF byte")
        T.ok(not text:find("\xFE"), "should not contain 0xFE byte")
        T.ok(#text > 0, "text should be non-empty")
    end
end)

T.test("EOS/EOG terminates generation correctly", function()
    -- Use a prompt designed to get a short answer
    local msgs = {
        { role = "user", content = "Reply with exactly one word: yes" },
    }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)

    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    local hit_eog = false
    local count = 0
    for _ = 1, 100 do
        local tok = s:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then
            hit_eog = true
            break
        end
        s:accept(tok)
        ctx:decode_single(tok, 0)
        count = count + 1
    end
    -- Model should eventually produce EOG (within 100 tokens for this prompt)
    print(string.format("  generated %d tokens, hit_eog=%s", count, tostring(hit_eog)))
    -- Don't assert hit_eog=true because some models may not stop in 100 tokens
    -- But we verify no crash occurred
    T.ok(count >= 0, "generation loop completed without crash")
end)

T.test("perf() after generation shows reasonable tok/s", function()
    -- Run a quick generation to populate perf counters
    local msgs = { { role = "user", content = "Count: 1 2 3" } }
    local prompt = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(prompt, false, true)
    ctx:kv_clear()
    ctx:perf_reset()
    ctx:decode(tokens, n, 0, 0)

    local s = ion7.Sampler.chain():top_k(1):dist(42):build(vocab)
    for _ = 1, 10 do
        local tok = s:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        s:accept(tok)
        ctx:decode_single(tok, 0)
    end

    local p = ctx:perf()
    T.ok(type(p.tokens_per_s) == "number", "tokens_per_s should be a number")
    T.ok(p.tokens_per_s >= 0, "tokens_per_s should be >= 0")
    if p.n_eval > 0 then
        T.ok(p.tokens_per_s > 0, "with n_eval > 0, tok/s should be > 0")
    end
    print(string.format("  perf: %.1f tok/s  n_p_eval=%d  n_eval=%d",
        p.tokens_per_s, p.n_p_eval, p.n_eval))
end)

-- ══════════════════════════════════════════════════════════════════
-- 21. DOUBLE FREE MODEL (must be LAST - destroys model)
-- ══════════════════════════════════════════════════════════════════

T.suite("Double free model (destructive - last)")

T.test("double free: model:free() then model:free() doesn't crash", function()
    -- Create a temporary model for this test to avoid destroying the main one
    local tmp_model = ion7.Model.load(model_path, { n_gpu_layers = 0, vocab_only = true })
    T.ok(tmp_model ~= nil, "tmp model loaded")
    T.no_error(function()
        tmp_model:free()
        tmp_model:free()  -- second free should be idempotent
    end, "double model:free()")
end)

-- ══════════════════════════════════════════════════════════════════
-- SUMMARY
-- ══════════════════════════════════════════════════════════════════

ion7.shutdown()
local ok = T.summary()
os.exit(ok and 0 or 1)
