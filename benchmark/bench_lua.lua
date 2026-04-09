--- benchmark/bench_lua.lua - ion7-core side of the comparison.
---
--- Copyright (C) 2026 Ion7 Project Contributors
--- SPDX-License-Identifier: MIT
---
--- Aligned with bench_python.py: same texts, same BOS settings, same iteration
--- counts on every shared vector.  Ion7-exclusive sections (9-14) are clearly
--- marked and have no Python counterpart.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit benchmark/bench_lua.lua
---   ION7_MODEL=/path/to/model.gguf LLAMA_LIB=/path/to/libllama.so \
---     luajit benchmark/bench_lua.lua --n-gpu-layers 35 --n-ctx 4096

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

-- ── CLI ───────────────────────────────────────────────────────────────────────

local args = {
    n_gpu_layers = nil,
    n_ctx        = 2048,
    n_gen        = 128,
    n_repeat     = 3,
    seed         = 42,
    stress       = false,
    n_ubatch     = nil,
    flash_attn   = false,
    n_batch      = nil,
}

do
    local i = 1
    while i <= #arg do
        if     arg[i] == "--n-gpu-layers" then args.n_gpu_layers = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--n-ctx"        then args.n_ctx        = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--n-gen"        then args.n_gen        = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--n-repeat"     then args.n_repeat     = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--seed"         then args.seed         = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--stress"       then args.stress       = true; i = i+1
        elseif arg[i] == "--n-ubatch"     then args.n_ubatch     = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--n-batch"      then args.n_batch      = tonumber(arg[i+1]); i = i+2
        elseif arg[i] == "--flash"        then args.flash_attn   = true; i = i+1
        else i = i+1 end
    end
end

-- ── Dependencies ──────────────────────────────────────────────────────────────

local ok_json, json = pcall(require, "dkjson")
if not ok_json then ok_json, json = pcall(require, "cjson") end
if not ok_json then
    io.stderr:write("Need dkjson or cjson: luarocks install dkjson\n")
    os.exit(1)
end

local MODEL = os.getenv("ION7_MODEL")
if not MODEL then
    io.stdout:write(json.encode({ error = "Set ION7_MODEL=/path/to/model.gguf" }) .. "\n")
    os.exit(1)
end

local ion7 = require "ion7.core"

local function find_so(hint)
    if not hint then return nil end
    if hint:match("%.so[%.%d]*$") or hint:match("%.dylib$") then return hint end
    for _, p in ipairs({
        hint .. "/libllama.so", hint .. "/bin/libllama.so",
        hint:gsub("/bin$", "") .. "/libllama.so",
    }) do
        local f = io.open(p, "rb"); if f then f:close(); return p end
    end
end

local llama_so = find_so(os.getenv("ION7_LIB_DIR"))
                 or os.getenv("LLAMA_LIB")

local ok_init, err_init = pcall(ion7.init, {
    log_level  = 0,
    llama_path = llama_so,
})
if not ok_init then
    io.stdout:write(json.encode({ error = "ion7.init() failed", detail = tostring(err_init) }) .. "\n")
    os.exit(1)
end

-- ── Helpers ───────────────────────────────────────────────────────────────────

local _time_fn
do
    if type(ion7.time_us) == "function" then
        local ok, v = pcall(ion7.time_us)
        if ok and type(v) == "number" then
            _time_fn = function() return ion7.time_us() / 1000.0 end
        end
    end
    if not _time_fn then
        local ok, socket = pcall(require, "socket")
        if ok and socket.gettime then
            _time_fn = function() return socket.gettime() * 1000.0 end
        end
    end
    if not _time_fn then
        _time_fn = function() return os.clock() * 1000.0 end
        io.stderr:write("[bench] warn: using os.clock() - CPU time only\n")
    end
end

local function now_ms() return _time_fn() end

--- Read process RSS from /proc/self/status. Returns MB, or -1 if unavailable.
local function rss_mb()
    local f = io.open("/proc/self/status", "r")
    if not f then return -1.0 end
    local content = f:read("*a"); f:close()
    local kb = content:match("VmRSS:%s*(%d+)%s*kB")
    return kb and tonumber(kb) / 1024.0 or -1.0
end

local function median(xs)
    if #xs == 0 then return 0 end
    local s = {}; for _, v in ipairs(xs) do s[#s+1] = v end
    table.sort(s)
    local n = #s
    return n % 2 == 0 and (s[n/2] + s[n/2+1]) / 2 or s[math.ceil(n/2)]
end

local function mean(xs)
    if #xs == 0 then return 0 end
    local sum = 0; for _, v in ipairs(xs) do sum = sum + v end
    return sum / #xs
end

local function round(x, d)
    local f = 10^(d or 2)
    return math.floor(x * f + 0.5) / f
end

--- Run fn n times with one warm-up, return { min, median, max } in ms.
local function measure(fn, n)
    n = n or args.n_repeat
    fn()
    collectgarbage("collect")
    local times = {}
    for _ = 1, n do
        local t0 = now_ms(); fn(); times[#times+1] = now_ms() - t0
        collectgarbage("collect")
    end
    table.sort(times)
    return { min = times[1], median = times[math.ceil(#times/2)], max = times[#times] }
end

local RESULTS = {
    backend   = "ion7-core (LuaJIT)",
    version   = "0.1.0",
    model     = MODEL:match("[^/]+$"),
    n_ctx     = args.n_ctx,
    n_gen     = args.n_gen,
    n_repeat  = args.n_repeat,
    benchmarks = {},
    errors    = {},
}

local function record(name, data) RESULTS.benchmarks[name] = data end
local function bench_err(name, msg)
    RESULTS.errors[#RESULTS.errors+1] = { benchmark = name, error = msg }
end

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION A - COMPARABLE WITH llama-cpp-python
-- Same texts, same BOS settings, same iteration counts on both sides.
-- ═══════════════════════════════════════════════════════════════════════════════

-- ── 1. Model load ─────────────────────────────────────────────────────────────

io.stderr:write("[bench] 1/8 loading model...\n"); io.stderr:flush()

local fit = ion7.Model.fit_params(MODEL, { n_ctx = args.n_ctx })
local ngl = args.n_gpu_layers or (fit and fit.n_gpu_layers or 0)

local mem_before = rss_mb()
local t0 = now_ms()
local model = ion7.Model.load(MODEL, { n_gpu_layers = ngl })
local load_ms = now_ms() - t0
local mem_after_load = rss_mb()

if not model then
    io.stdout:write(json.encode({ error = "Model load failed", model = MODEL }) .. "\n")
    os.exit(1)
end

local info = model:info()
local vocab = model:vocab()
local ctx   = model:context({
    n_gpu_layers = ngl,
    n_ctx        = args.n_ctx,
    n_ubatch     = args.n_ubatch,
    flash_attn   = args.flash_attn or nil,
    n_batch      = args.n_batch,
})

record("model_load", {
    load_ms       = round(load_ms),
    n_gpu_layers  = ngl,
    n_batch       = ctx:n_batch(),
    n_ubatch      = ctx:n_ubatch(),
    n_params_b    = round(info.n_params / 1e9, 3),
    n_layers      = info.n_layer,
    n_embd        = info.n_embd,
    size_gb       = round(info.size / 1e9, 2),
    rss_delta_mb  = mem_after_load >= 0 and round(mem_after_load - mem_before, 1) or nil,
})

-- ── 2. Tokenization ───────────────────────────────────────────────────────────
-- add_special=true, parse_special=false - same as Python add_bos=True, special=False.

io.stderr:write("[bench] 2/8 tokenization...\n"); io.stderr:flush()

-- Texts identical to bench_python.py
local TOKENIZE_TEXTS = {
    "Hello world",
    ("The quick brown fox jumps over the lazy dog. "):rep(10),
    ("Explain the attention mechanism in transformers. "):rep(20),
}

local tok_results = {}
for _, text in ipairs(TOKENIZE_TEXTS) do
    local times = {}
    local n_tokens
    for _ = 1, args.n_repeat do
        local t1 = now_ms()
        local _, n = vocab:tokenize(text, true, false)  -- add_bos=true, parse_special=false
        times[#times+1] = now_ms() - t1
        n_tokens = n
    end
    local tok_ms = median(times)
    tok_results[#tok_results+1] = {
        text_chars   = #text,
        n_tokens     = n_tokens,
        median_ms    = round(tok_ms, 3),
        tokens_per_s = tok_ms > 0 and round(n_tokens / (tok_ms / 1000), 0) or 0,
    }
end

record("tokenization", {
    cases            = tok_results,
    avg_tokens_per_s = round(mean(
        (function() local r = {}; for _, v in ipairs(tok_results) do r[#r+1] = v.tokens_per_s end; return r end)()
    ), 0),
    note = "add_bos=true, parse_special=false - aligned with Python add_bos=True",
})

-- ── 3. Prompt prefill ─────────────────────────────────────────────────────────
-- Raw tokenization with add_bos=true. No chat template (Python has no equivalent).
-- Times the decode() call only, not tokenization.

io.stderr:write("[bench] 3/8 prefill...\n"); io.stderr:flush()

-- Prompts identical to bench_python.py
local PROMPTS = {
    "What is 2+2?",
    "Explain in detail the history of the Roman Empire and its cultural impact on Western civilization.",
    ("Write a comprehensive technical guide about transformer architectures. "):rep(8),
}

local prefill_results = {}
for _, prompt in ipairs(PROMPTS) do
    local tokens, n = vocab:tokenize(prompt, true, false)  -- add_bos=true
    local times_ms = {}
    for _ = 1, args.n_repeat do
        ctx:kv_clear()
        local t1 = now_ms()
        ctx:decode(tokens, n, 0, 0)
        times_ms[#times_ms+1] = now_ms() - t1
    end
    local med = median(times_ms)
    prefill_results[#prefill_results+1] = {
        prompt_tokens = n,
        median_ms     = round(med),
        tokens_per_s  = med > 0 and round(n / (med / 1000), 1) or 0,
    }
end

record("prefill", {
    cases            = prefill_results,
    avg_tokens_per_s = round(mean(
        (function() local r = {}; for _, v in ipairs(prefill_results) do r[#r+1] = v.tokens_per_s end; return r end)()
    ), 1),
    note = "raw tokenize(add_bos=true) + decode() - no chat template",
})

-- ── 4. Generation throughput ──────────────────────────────────────────────────
-- Prompt and sampler params identical to bench_python.py.

io.stderr:write("[bench] 4/8 generation...\n"); io.stderr:flush()

local GEN_PROMPT   = "Count from 1 to 100 as fast as possible:"
local gen_tokens, gen_n = vocab:tokenize(GEN_PROMPT, true, false)

local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95, 1):temperature(0.8):dist(args.seed):build(vocab)

local gen_times = {}
local gen_tok_s = {}

for _ = 1, args.n_repeat do
    ctx:kv_clear()
    ctx:decode(gen_tokens, gen_n, 0, 0)
    sampler:reset()

    local t1 = now_ms()
    local count = 0
    local ok, err_g = pcall(function()
        for _ = 1, args.n_gen do
            local tok = sampler:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            ctx:decode_single(tok, 0)
            count = count + 1
        end
    end)
    if not ok then io.stderr:write("[bench] gen error: " .. tostring(err_g) .. "\n"); break end
    local elapsed = now_ms() - t1
    if elapsed > 0 then
        gen_times[#gen_times+1] = elapsed
        gen_tok_s[#gen_tok_s+1] = count / (elapsed / 1000)
    end
end

record("generation", {
    prompt       = GEN_PROMPT,
    n_gen        = args.n_gen,
    median_ms    = round(median(gen_times)),
    median_tok_s = round(median(gen_tok_s), 2),
    ms_per_token = round(median(gen_times) / args.n_gen, 2),
})

-- ── 5. Grammar-constrained generation ────────────────────────────────────────
-- Identical GBNF, prompt and max_tokens to bench_python.py.

io.stderr:write("[bench] 5/8 grammar...\n"); io.stderr:flush()

local GBNF          = 'root ::= "positive" | "negative" | "neutral"'
local GRAMMAR_PROMPT = "Review: 'This product is excellent!' Sentiment:"
local VALID = { positive = true, negative = true, neutral = true }

local grammar_sampler = ion7.Sampler.chain()
    :grammar(GBNF, "root", vocab):top_k(40):temperature(0.0):dist(args.seed):build(vocab)

local grammar_tokens, grammar_n = vocab:tokenize(GRAMMAR_PROMPT, true, false)
local grammar_times   = {}
local grammar_outputs = {}

for _ = 1, args.n_repeat do
    ctx:kv_clear()
    ctx:decode(grammar_tokens, grammar_n, 0, 0)
    grammar_sampler:reset()
    local t1 = now_ms()
    local parts = {}
    for _ = 1, 8 do
        local tok = grammar_sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        parts[#parts+1] = vocab:piece(tok)
    end
    grammar_times[#grammar_times+1]    = now_ms() - t1
    grammar_outputs[#grammar_outputs+1] = table.concat(parts):match("^%s*(.-)%s*$")
end

grammar_sampler:free()

local all_valid = true
for _, o in ipairs(grammar_outputs) do if not VALID[o] then all_valid = false end end

record("grammar_constrained", {
    gbnf      = GBNF,
    outputs   = grammar_outputs,
    all_valid = all_valid,
    median_ms = round(median(grammar_times)),
    supported = true,
})

-- ── 6. KV snapshot ────────────────────────────────────────────────────────────
-- Both sides use in-memory snapshot (no file I/O). n_repeat*2 iterations each.

io.stderr:write("[bench] 6/8 kv snapshot...\n"); io.stderr:flush()

do
    local snap_tokens, snap_n = vocab:tokenize(
        "Hello, I am a benchmark. Please remember this message.", true, false)
    ctx:kv_clear()
    ctx:decode(snap_tokens, snap_n, 0, 0)

    local save_times = {}
    local load_times = {}
    local blob, snap_size_kb

    for _ = 1, args.n_repeat * 2 do
        local t1 = now_ms(); blob = ctx:snapshot(); save_times[#save_times+1] = now_ms() - t1
        local t2 = now_ms(); ctx:restore(blob);     load_times[#load_times+1] = now_ms() - t2
    end
    snap_size_kb = round(#blob / 1024, 1)

    record("kv_snapshot", {
        n_past        = snap_n,
        state_size_kb = snap_size_kb,
        save_ms       = round(median(save_times), 3),
        load_ms       = round(median(load_times), 3),
        in_memory     = true,
    })
end

-- ── 7. Detokenization ────────────────────────────────────────────────────────
-- N_DTOK=1000 calls (same as bench_python.py).
-- Text and tokenization settings identical on both sides.

io.stderr:write("[bench] 7/8 detokenization...\n"); io.stderr:flush()

do
    local DTOK_TEXT  = "The quick brown fox jumps over the lazy dog."
    local N_DTOK     = 1000
    local dtok_toks, dtok_n = vocab:tokenize(DTOK_TEXT, false, false)
    local dtok_times = {}

    -- One warm-up pass
    vocab:detokenize(dtok_toks, dtok_n, false, false)
    collectgarbage("collect")

    -- N_DTOK consecutive calls per measurement round
    for _ = 1, args.n_repeat do
        local t1 = now_ms()
        for _ = 1, N_DTOK do vocab:detokenize(dtok_toks, dtok_n, false, false) end
        dtok_times[#dtok_times+1] = (now_ms() - t1) / N_DTOK  -- ms per call
    end

    local med_per_call = median(dtok_times)
    record("detokenization", {
        n_tokens       = dtok_n,
        n_calls        = N_DTOK,
        median_ms      = round(med_per_call, 6),
        calls_per_s    = round(1000 / med_per_call, 0),
        note           = N_DTOK .. " calls per measurement, median ms/call reported",
    })
end

-- ── 7b. Memory footprint ─────────────────────────────────────────────────────
-- Comparable with bench_python.py section 8.
-- RSS after full init (model + context), delta vs pre-load.

do
    local rss_now = rss_mb()
    record("memory", {
        rss_after_load_mb = rss_now >= 0 and round(rss_now, 1) or nil,
        rss_delta_load_mb = (mem_before >= 0 and rss_now >= 0)
                            and round(rss_now - mem_before, 1) or nil,
        proc_available    = mem_before >= 0,
    })
end

-- ── 8. Single-token decode loop ───────────────────────────────────────────────
-- N=200 tokens. Measures per-token decode+sample latency.
-- Python equivalent: llm.generate() iterator with per-token timestamps.

io.stderr:write("[bench] 8/8 single-token loop...\n"); io.stderr:flush()

do
    local ok8, err8 = pcall(function()
        local N_SINGLE = 200
        local loop_tokens, loop_n = vocab:tokenize("Write a long story about a knight:", true, false)
        ctx:kv_clear()
        ctx:decode(loop_tokens, loop_n, 0, 0)
        sampler:reset()

        -- Collect per-token timestamps (mirror of Python's approach)
        local tok_stamps = {}
        local t_start    = now_ms()
        for _ = 1, N_SINGLE do
            local tok = sampler:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            ctx:decode_single(tok, 0)
            tok_stamps[#tok_stamps+1] = now_ms()
        end
        local total_ms = now_ms() - t_start
        local n_actual = #tok_stamps

        local deltas = {}
        if n_actual >= 2 then
            deltas[1] = tok_stamps[1] - t_start
            for i = 2, n_actual do deltas[i] = tok_stamps[i] - tok_stamps[i-1] end
        end

        record("single_token_loop", {
            n_tokens            = n_actual,
            tokens_per_s        = round(n_actual / (total_ms / 1000), 2),
            median_ms_per_token = round(#deltas > 0 and median(deltas) or total_ms / math.max(n_actual, 1), 3),
            total_ms            = round(total_ms, 2),
            note                = "decode_single + sampler:sample per token, per-token timestamps",
        })
    end)
    if not ok8 then bench_err("single_token_loop", tostring(err8)) end
end

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION B - ION7-CORE EXCLUSIVE
-- No Python equivalent. Not shown in the ratio columns.
-- ═══════════════════════════════════════════════════════════════════════════════

-- ── 9. Context creation (KV quant variants) ───────────────────────────────────

io.stderr:write("[bench] ion7] context creation...\n"); io.stderr:flush()

do
    local ok9, err9 = pcall(function()
        local function make_ctx(kv)
            return model:context({
                n_gpu_layers = ngl, n_ctx = args.n_ctx, kv_quant = kv,
                n_ubatch = args.n_ubatch, flash_attn = args.flash_attn or nil,
                n_batch = args.n_batch,
            })
        end
        local t_f16 = measure(function() make_ctx("f16"):free()  end)
        local t_q8  = measure(function() make_ctx("q8_0"):free() end)
        local t_q4  = measure(function() make_ctx("q4_0"):free() end)
        record("context_creation", {
            cases = {
                { kv_quant = "f16",  median_ms = round(t_f16.median, 3) },
                { kv_quant = "q8_0", median_ms = round(t_q8.median, 3)  },
                { kv_quant = "q4_0", median_ms = round(t_q4.median, 3)  },
            },
        })
    end)
    if not ok9 then bench_err("context_creation", tostring(err9)) end
end

-- ── 10. Sampler chain overhead ────────────────────────────────────────────────

do
    local ok10, err10 = pcall(function()
        local profiles = {
            greedy   = ion7.Sampler.chain():greedy():build(vocab),
            minimal  = ion7.Sampler.chain():top_k(1):dist(42):build(vocab),
            standard = ion7.Sampler.chain():top_k(40):top_p(0.95,1):temperature(0.8):dist(42):build(vocab),
        }
        ctx:kv_clear()
        ctx:decode(gen_tokens, gen_n, 0, 0)
        local profile_results = {}
        for name, samp in pairs(profiles) do
            local t = measure(function()
                for _ = 1, 100 do samp:sample(ctx:ptr(), -1) end
            end)
            profile_results[name] = { median_ms = round(t.median / 100, 4) }
            samp:free()
        end
        record("sampler_overhead", { profiles = profile_results })
    end)
    if not ok10 then bench_err("sampler_overhead", tostring(err10)) end
end

-- ── 11. KV cache operations ───────────────────────────────────────────────────

do
    local ok11, err11 = pcall(function()
        local kv_tok, kv_n = vocab:tokenize("Hello world test", false, false)
        ctx:kv_clear(); ctx:decode(kv_tok, kv_n, 0, 0)

        local N = 10000
        local t_clear = measure(function() for _ = 1, N do ctx:kv_clear() end end)
        ctx:decode(kv_tok, kv_n, 0, 0)
        local t_rm = measure(function() for _ = 1, N do ctx:kv_seq_rm(0, 0, -1) end end)
        ctx:kv_clear(); ctx:decode(kv_tok, kv_n, 0, 0)
        local t_cp = measure(function() for _ = 1, 1000 do ctx:kv_seq_cp(0, 1, 0, -1) end end)

        record("kv_operations", {
            kv_clear  = { n = N,    calls_per_s = round(N    / (t_clear.median / 1000), 0) },
            kv_seq_rm = { n = N,    calls_per_s = round(N    / (t_rm.median    / 1000), 0) },
            kv_seq_cp = { n = 1000, calls_per_s = round(1000 / (t_cp.median    / 1000), 0) },
        })
    end)
    if not ok11 then bench_err("kv_operations", tostring(err11)) end
end

-- ── 12. State persistence (file I/O) ──────────────────────────────────────────

do
    local ok12, err12 = pcall(function()
        local sp_tok, sp_n = vocab:tokenize("State persistence benchmark", false, false)
        ctx:kv_clear(); ctx:decode(sp_tok, sp_n, 0, 0)

        local t_snap    = measure(function() ctx:snapshot() end)
        local blob      = ctx:snapshot()
        local t_restore = measure(function() ctx:restore(blob) end)

        local path = "/tmp/ion7_bench_state.bin"
        local t_save = measure(function() ctx:save_state(path) end)
        local t_load = measure(function() ctx:load_state(path) end)
        os.remove(path)

        record("state_persistence", {
            snapshot_ms      = round(t_snap.median, 3),
            restore_ms       = round(t_restore.median, 3),
            snapshot_size_kb = round(#blob / 1024, 1),
            save_file_ms     = round(t_save.median, 3),
            load_file_ms     = round(t_load.median, 3),
        })
    end)
    if not ok12 then bench_err("state_persistence", tostring(err12)) end
end

-- ── 13. Custom Lua sampler (callback overhead) ────────────────────────────────

do
    local ok13, err13 = pcall(function()
        local lua_greedy = ion7.CustomSampler.new("bench_greedy", {
            apply = function(cur_p, nc)
                local best_i, best_v = 0, -math.huge
                local data = cur_p.data
                for ci = 0, nc - 1 do
                    if data[ci].logit > best_v then best_v = data[ci].logit; best_i = ci end
                end
                cur_p.selected = best_i
            end,
        })
        local lua_chain    = ion7.Sampler.chain():custom(lua_greedy):build(vocab)
        local native_chain = ion7.Sampler.chain():greedy():build(vocab)

        ctx:kv_clear(); ctx:decode(gen_tokens, gen_n, 0, 0)
        local N = 1000
        local t_lua    = measure(function() for _ = 1, N do lua_chain:sample(ctx:ptr(), -1) end end)
        local t_native = measure(function() for _ = 1, N do native_chain:sample(ctx:ptr(), -1) end end)

        lua_chain:free(); native_chain:free()

        record("custom_sampler", {
            lua_greedy    = { samples_per_s = round(N / (t_lua.median    / 1000), 0) },
            native_greedy = { samples_per_s = round(N / (t_native.median / 1000), 0) },
            overhead_pct  = round((t_lua.median / t_native.median - 1.0) * 100.0, 1),
        })
    end)
    if not ok13 then bench_err("custom_sampler", tostring(err13)) end
end

-- ── 14. Vocab operations ──────────────────────────────────────────────────────

do
    local ok14, err14 = pcall(function()
        local N = 500000
        local bos_tok = vocab:bos()
        local t_bos = measure(function() for _ = 1, N do vocab:bos() end end)
        local t_eog = measure(function() for _ = 1, N do vocab:is_eog(bos_tok) end end)

        local pt = vocab:tokenize("Hello world", false, false)
        local n_piece = math.floor(N / 10)
        local t_piece = measure(function()
            for _ = 1, n_piece do vocab:piece(pt[0], true) end
        end)

        -- tokenize() loop - same text/N as Python section 14
        local VOC_TEXT   = "The quick brown fox jumps over the lazy dog."
        local N_VOC      = 1000
        local voc_toks, voc_n = vocab:tokenize(VOC_TEXT, false, false)
        local t_tok_loop = measure(function()
            for _ = 1, N_VOC do vocab:tokenize(VOC_TEXT, false, false) end
        end)
        -- detokenize() loop - same tokens/N as Python
        local t_dtok_loop = measure(function()
            for _ = 1, N_VOC do vocab:detokenize(voc_toks, voc_n, false, false) end
        end)

        record("vocab_operations", {
            bos       = { n = N,       calls_per_s = round(N       / (t_bos.median      / 1000), 0) },
            is_eog    = { n = N,       calls_per_s = round(N       / (t_eog.median      / 1000), 0) },
            piece     = { n = n_piece, calls_per_s = round(n_piece / (t_piece.median    / 1000), 0) },
            tokenize  = { n = N_VOC,   calls_per_s = round(N_VOC   / (t_tok_loop.median  / 1000), 0) },
            detokenize = { n = N_VOC,  calls_per_s = round(N_VOC   / (t_dtok_loop.median / 1000), 0) },
        })
    end)
    if not ok14 then bench_err("vocab_operations", tostring(err14)) end
end

-- ── Stress tests ──────────────────────────────────────────────────────────────

if args.stress then
    io.stderr:write("[bench] STRESS: sustained generation...\n"); io.stderr:flush()

    local LONG_PROMPT = "Write a detailed technical essay about transformer architecture, "
                     .. "attention mechanisms, and how large language models work."
    local tok_long, n_long = vocab:tokenize(LONG_PROMPT, true, false)
    local N_LONG = math.min(args.n_gen * 8, 512)

    ctx:kv_clear(); ctx:decode(tok_long, n_long, 0, 0); sampler:reset()

    local window_rates, window_count, window_start = {}, 0, now_ms()
    local total_count = 0

    for _ = 1, N_LONG do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        total_count = total_count + 1
        window_count = window_count + 1
        if window_count == 32 then
            window_rates[#window_rates+1] = round(32 / ((now_ms() - window_start) / 1000), 1)
            window_start = now_ms(); window_count = 0
        end
    end

    local perf_long = ctx:perf()
    record("stress_sustained_generation", {
        n_tokens_generated = total_count, n_prompt_tokens = n_long,
        overall_tok_s = round(perf_long.tokens_per_s, 2),
        window_tok_s  = window_rates,
        t_eval_ms     = round(perf_long.t_eval_ms, 2),
    })

    io.stderr:write("[bench] STRESS: back-to-back sessions...\n"); io.stderr:flush()

    local TURNS = {
        "What is 2+2?", "And what is 3+3?", "What about 10*10?",
        "Name the first 5 prime numbers.", "What color is the sky?",
        "How many days in a week?", "What is the capital of France?",
        "How many continents are there?", "What is pi approximately?", "Is water wet?",
    }
    local session_times = {}
    for _, turn in ipairs(TURNS) do
        local t_tok, t_n = vocab:tokenize(
            "You are a helpful assistant.\n\nUser: " .. turn .. "\nAssistant:", true, false)
        ctx:kv_clear()
        local t1 = now_ms(); ctx:decode(t_tok, t_n, 0, 0); sampler:reset()
        for _ = 1, 16 do
            local tok = sampler:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            ctx:decode_single(tok, 0)
        end
        session_times[#session_times+1] = round(now_ms() - t1, 2)
    end
    record("stress_back_to_back", {
        n_sessions        = #TURNS,
        session_ms        = session_times,
        median_session_ms = round(median(session_times), 2),
        min_session_ms    = round(math.min(table.unpack(session_times)), 2),
        max_session_ms    = round(math.max(table.unpack(session_times)), 2),
    })

    io.stderr:write("[bench] STRESS: context pressure...\n"); io.stderr:flush()

    local fill_text = ("The quick brown fox jumps over the lazy dog. "):rep(100)
    local fill_tok, fill_n = vocab:tokenize(fill_text, false, false)
    fill_n = math.min(fill_n, math.floor(args.n_ctx * 0.75) - 32)
    ctx:kv_clear()
    local t_fill = now_ms(); ctx:decode(fill_tok, fill_n, 0, 0); local fill_ms = now_ms() - t_fill
    sampler:reset()
    local t_gen = now_ms(); local gen_full = 0
    for _ = 1, 32 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0); gen_full = gen_full + 1
    end
    local gen_full_ms = now_ms() - t_gen
    record("stress_context_pressure", {
        n_ctx = args.n_ctx, fill_tokens = fill_n,
        fill_pct        = round(fill_n / args.n_ctx * 100, 1),
        prefill_tok_s   = round(fill_n / (fill_ms / 1000), 1),
        gen_on_full_tok_s = round(gen_full / (gen_full_ms / 1000), 1),
    })

    io.stderr:write("[bench] STRESS: sampler throughput...\n"); io.stderr:flush()

    local samp_stress = ion7.Sampler.chain()
        :top_k(40):top_p(0.95, 1):temperature(0.8):dist(args.seed):build(vocab)
    ctx:kv_clear(); ctx:decode(tok_long, math.min(n_long, 64), 0, 0)
    local N_S = 1000; local t_s = now_ms()
    for _ = 1, N_S do samp_stress:sample(ctx:ptr(), -1) end
    local sm = now_ms() - t_s
    samp_stress:free()
    record("stress_sampler_throughput", {
        n_samples = N_S, avg_us_per_call = round(sm * 1000 / N_S, 2),
        samples_per_s = round(N_S / (sm / 1000), 0),
    })
end

-- ── Cleanup & output ──────────────────────────────────────────────────────────

sampler:free()
ctx:free()
model:free()
ion7.shutdown()

io.stdout:write(json.encode(RESULTS) .. "\n")
io.stdout:flush()
