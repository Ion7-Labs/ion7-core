--- benchmark/bench_lua.lua - ion7-core side of the comparison.

---
--- Copyright (C) 2026 Ion7 Project Contributors
--- SPDX-License-Identifier: MIT
---

--- Measures the same operations as bench_python.py so results are directly
--- comparable. Emits JSON to stdout with the exact same schema.
--- Sections 1-8: core benchmarks. Sections 9-14: deep internals.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit benchmark/bench_lua.lua
---   ION7_MODEL=/path/to/model.gguf ION7_LIB_DIR=/path/to/llama/build/bin \
---     luajit benchmark/bench_lua.lua --n-gpu-layers 20 --n-gen 128
---
--- @author Ion7-Labs

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
    n_batch      = nil,   -- nil = auto (512 GPU / 256 CPU)
}

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

-- ── Dependencies ──────────────────────────────────────────────────────────────

local ok_json, json = pcall(require, "dkjson")
if not ok_json then ok_json, json = pcall(require, "cjson") end
if not ok_json then
    io.stderr:write("Need dkjson or cjson: luarocks install dkjson\n")
    os.exit(1)
end

local MODEL   = os.getenv("ION7_MODEL")
local LIB_DIR = os.getenv("ION7_LIB_DIR")

if not MODEL then
    io.stdout:write(json.encode({ error = "Set ION7_MODEL=/path/to/model.gguf" }) .. "\n")
    os.exit(1)
end

local ion7 = require "ion7.core"

-- Resolve the full path to libllama.so.
-- ION7_LIB_DIR can be either:
--   - A directory containing libllama.so  → we append the filename
--   - The full path to libllama.so itself → we use it directly
-- The LLAMA_LIB env var is also supported by ion7-core directly.
local function find_so(hint)
    if not hint then return nil end
    -- If it already ends in .so it's a full path
    if hint:match("%.so[%.%d]*$") then return hint end
    -- Otherwise treat as directory
    local candidates = {
        hint .. "/libllama.so",
        hint .. "/bin/libllama.so",
        hint:gsub("/bin$", "") .. "/libllama.so",
        hint .. "/../libllama.so",
    }
    for _, p in ipairs(candidates) do
        local f = io.open(p, "rb")
        if f then f:close(); return p end
    end
    return nil
end

local llama_so = find_so(LIB_DIR) or os.getenv("LLAMA_LIB")  -- ion7-core also checks this env var

if LIB_DIR and not llama_so then
    io.stderr:write("[bench] warn: could not find libllama.so in " .. LIB_DIR .. "\n")
end
if llama_so then
    io.stderr:write("[bench] using: " .. llama_so .. "\n")
end

local ok_init, err_init = pcall(ion7.init, {
    log_level  = 0,
    llama_path = llama_so,   -- full path to .so, or nil to let ion7-core auto-detect
})
if not ok_init then
    io.stdout:write(json.encode({
        error  = "ion7.init() failed",
        detail = tostring(err_init),
        hints  = {
            "ION7_LIB_DIR must point to the directory containing libllama.so",
            "OR set it to the full path: ION7_LIB_DIR=/path/to/libllama.so",
            "OR set LLAMA_LIB=/path/to/libllama.so",
            "Find it with: find ~ -name 'libllama.so' 2>/dev/null",
        }
    }) .. "\n")
    os.exit(1)
end

-- ── Helpers ───────────────────────────────────────────────────────────────────

--- High-resolution timestamp in milliseconds.
--- Falls back through multiple sources for compatibility.
local _time_fn
do
    -- Prefer ion7.time_us() (microsecond precision, monotonic)
    if type(ion7.time_us) == "function" then
        local ok, v = pcall(ion7.time_us)
        if ok and type(v) == "number" then
            _time_fn = function() return ion7.time_us() / 1000.0 end
        end
    end
    -- Fallback: luasocket high-res clock
    if not _time_fn then
        local ok, socket = pcall(require, "socket")
        if ok and socket.gettime then
            _time_fn = function() return socket.gettime() * 1000.0 end
        end
    end
    -- Fallback: os.clock (CPU time, not wall time - less accurate)
    if not _time_fn then
        _time_fn = function() return os.clock() * 1000.0 end
        io.stderr:write("[bench] warn: using os.clock() - CPU time only, not wall time\n")
    end
end

local function now_ms()
    return _time_fn()
end

--- Median of a table of numbers.
local function median(xs)
    if #xs == 0 then return 0 end
    local s = {}
    for _, v in ipairs(xs) do s[#s+1] = v end
    table.sort(s)
    local n = #s
    if n % 2 == 0 then return (s[n/2] + s[n/2+1]) / 2
    else return s[math.ceil(n/2)] end
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

--- Run fn n times with warm-up, return { min, median, max } in ms.
--- Used by deep sections 9-14.
local function measure(fn, n_repeat_override)
    local nr = n_repeat_override or args.n_repeat
    fn()  -- warm-up
    collectgarbage("collect")
    local times = {}
    for mi = 1, nr do
        local t0 = now_ms()
        fn()
        times[#times+1] = now_ms() - t0
        collectgarbage("collect")
    end
    table.sort(times)
    return {
        min    = times[1],
        median = times[math.ceil(#times / 2)],
        max    = times[#times],
    }
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

local function record(name, data)
    RESULTS.benchmarks[name] = data
end

local function bench_err(name, msg)
    RESULTS.errors[#RESULTS.errors+1] = { benchmark = name, error = msg }
end

-- ── 1. Model load ─────────────────────────────────────────────────────────────

io.stderr:write("[bench] 1/14 loading model...\n"); io.stderr:flush()
local fit = ion7.Model.fit_params(MODEL, { n_ctx = args.n_ctx })
local ngl = args.n_gpu_layers or (fit and fit.n_gpu_layers or 0)

local t0 = now_ms()
local model = ion7.Model.load(MODEL, { n_gpu_layers = ngl })
local load_ms = now_ms() - t0

if not model then
    io.stdout:write(json.encode({
        error   = "Model load failed",
        hint    = "Check ION7_MODEL and ION7_LIB_DIR paths",
        model   = MODEL,
        lib_dir = LIB_DIR or "(auto)",
    }) .. "\n")
    os.exit(1)
end

local info = model:info()

-- ── Setup context + vocab ─────────────────────────────────────────────────────

local vocab = model:vocab()
local ctx   = model:context({
    n_gpu_layers = ngl,
    n_ctx        = args.n_ctx,
    n_ubatch     = args.n_ubatch,  -- nil = auto
    flash_attn   = args.flash_attn or nil,
    n_batch      = args.n_batch,
})

record("model_load", {
    load_ms      = round(load_ms),
    n_gpu_layers = ngl,
    n_ubatch     = ctx:n_ubatch() or "(bug:nil)",
    n_batch      = ctx:n_batch() or "(bug:nil)",
    n_params_b   = round(info.n_params / 1e9, 3),
    n_layers     = info.n_layer,
    n_embd       = info.n_embd,
    size_gb      = round(info.size / 1e9, 2),
})

io.stderr:write("[bench] 2/14 tokenization...\n"); io.stderr:flush()
-- ── 2. Tokenization ───────────────────────────────────────────────────────────

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
        local _, n = vocab:tokenize(text, false, false)
        local elapsed = now_ms() - t1
        times[#times+1] = elapsed
        n_tokens = n
    end
    local tok_ms = median(times)
    tok_results[#tok_results+1] = {
        text_chars    = #text,
        n_tokens      = n_tokens,
        median_ms     = round(tok_ms, 3),
        tokens_per_s  = tok_ms > 0 and round(n_tokens / (tok_ms / 1000), 0) or 0,
    }
end

local avg_tok_s = 0
for _, r in ipairs(tok_results) do avg_tok_s = avg_tok_s + r.tokens_per_s end
avg_tok_s = round(avg_tok_s / #tok_results, 0)

record("tokenization", {
    cases           = tok_results,
    avg_tokens_per_s = avg_tok_s,
})

io.stderr:write("[bench] 3/14 prefill...\n"); io.stderr:flush()
-- ── 3. Prompt prefill ─────────────────────────────────────────────────────────

local PROMPTS = {
    "What is 2+2?",
    "Explain in detail the history of the Roman Empire and its cultural impact on Western civilization.",
    ("Write a comprehensive technical guide about transformer architectures. "):rep(8),
}

local prefill_results = {}
for _, prompt in ipairs(PROMPTS) do
    local times_ms  = {}
    local tok_rates = {}
    local n_prompt_tokens

    local msgs = { { role = "user", content = prompt } }
    local formatted = vocab:apply_template(msgs, true)
    local tokens, n = vocab:tokenize(formatted, false, true)
    n_prompt_tokens = n

    for _ = 1, args.n_repeat do
        ctx:kv_clear()
        local t1 = now_ms()
        ctx:decode(tokens, n, 0, 0)
        local elapsed = now_ms() - t1
        times_ms[#times_ms+1]  = elapsed
        tok_rates[#tok_rates+1] = n / (elapsed / 1000)
    end

    prefill_results[#prefill_results+1] = {
        prompt_tokens = n_prompt_tokens,
        median_ms     = round(median(times_ms)),
        tokens_per_s  = round(median(tok_rates), 1),
    }
end

local avg_prefill_s = 0
for _, r in ipairs(prefill_results) do avg_prefill_s = avg_prefill_s + r.tokens_per_s end
avg_prefill_s = round(avg_prefill_s / #prefill_results, 1)

record("prefill", {
    cases            = prefill_results,
    avg_tokens_per_s = avg_prefill_s,
})

io.stderr:write("[bench] 4/14 generation...\n"); io.stderr:flush()
-- ── 4. Generation throughput ──────────────────────────────────────────────────

local GEN_PROMPT = "Count from 1 to 100 as fast as possible:"
local sampler = ion7.Sampler.chain()
    :top_k(40)
    :top_p(0.95, 1)
    :temperature(0.8)
    :dist(args.seed)
    :build(vocab)

local gen_times   = {}
local gen_tok_s   = {}

local msgs = { { role = "user", content = GEN_PROMPT } }
local formatted = vocab:apply_template(msgs, true)
local tokens, n = vocab:tokenize(formatted, false, true)

for _ = 1, args.n_repeat do
    ctx:kv_clear()
    ctx:decode(tokens, n, 0, 0)
    sampler:reset()

    local t1 = now_ms()
    local count = 0
    local ok, err_gen = pcall(function()
        for _ = 1, args.n_gen do
            local tok = sampler:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            ctx:decode_single(tok, 0)
            count = count + 1
        end
    end)
    if not ok then
        io.stderr:write("[bench] generation error: " .. tostring(err_gen) .. "\n")
        break
    end
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

io.stderr:write("[bench] 5/14 grammar...\n"); io.stderr:flush()
-- ── 5. Grammar-constrained generation ────────────────────────────────────────

local GBNF = 'root ::= "positive" | "negative" | "neutral"'
local grammar_sampler = ion7.Sampler.chain()
    :grammar(GBNF, "root", vocab)
    :top_k(40)
    :temperature(0.0)
    :dist(args.seed)
    :build(vocab)

local grammar_times   = {}
local grammar_outputs = {}
local VALID = { positive = true, negative = true, neutral = true }

local grammar_prompt = "Review: 'This product is excellent!' Sentiment:"
local msgs_g = { { role = "user", content = grammar_prompt } }
local formatted_g = vocab:apply_template(msgs_g, true)
local tokens_g, n_g = vocab:tokenize(formatted_g, false, true)

for _ = 1, args.n_repeat do
    ctx:kv_clear()
    ctx:decode(tokens_g, n_g, 0, 0)
    grammar_sampler:reset()

    local t1    = now_ms()
    local parts = {}
    for _ = 1, 8 do
        local tok = grammar_sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        parts[#parts+1] = vocab:piece(tok)
    end
    local elapsed = now_ms() - t1
    grammar_times[#grammar_times+1]   = elapsed
    grammar_outputs[#grammar_outputs+1] = table.concat(parts):match("^%s*(.-)%s*$")
end

grammar_sampler:free()

local all_valid = true
for _, o in ipairs(grammar_outputs) do
    if not VALID[o] then all_valid = false end
end

record("grammar_constrained", {
    gbnf      = GBNF,
    outputs   = grammar_outputs,
    all_valid = all_valid,
    median_ms = round(median(grammar_times)),
    supported = true,
})

io.stderr:write("[bench] 6/14 kv snapshot...\n"); io.stderr:flush()
-- ── 6. KV snapshot / restore ─────────────────────────────────────────────────

local snap_prompt = "Hello, I am a benchmark. Please remember this message."
local msgs_s = { { role = "user", content = snap_prompt } }
local formatted_s  = vocab:apply_template(msgs_s, true)
local tokens_s, n_s = vocab:tokenize(formatted_s, false, true)

ctx:kv_clear()
ctx:decode(tokens_s, n_s, 0, 0)

local snap_times  = {}
local restore_times = {}
local snap_size_kb

for _ = 1, args.n_repeat * 2 do
    local t1 = now_ms()
    local blob = ctx:snapshot()
    snap_times[#snap_times+1] = now_ms() - t1
    snap_size_kb = #blob / 1024

    local t2 = now_ms()
    ctx:restore(blob)
    restore_times[#restore_times+1] = now_ms() - t2
end

record("kv_snapshot", {
    method         = "snapshot()/restore() - in-memory blob",
    n_past         = n_s,
    state_size_kb  = round(snap_size_kb, 1),
    save_ms        = round(median(snap_times), 3),
    load_ms        = round(median(restore_times), 3),
    in_memory      = true,
    note           = "ion7-core uses in-memory blobs; no file I/O involved",
})

io.stderr:write("[bench] 7/14 detokenization...\n"); io.stderr:flush()
-- ── 7. Detokenization ────────────────────────────────────────────────────────

local detok_text   = "Hello world this is a test of detokenization speed"
local detok_tokens, detok_n = vocab:tokenize(detok_text, false, false)
local detok_times  = {}

for _ = 1, args.n_repeat * 10 do
    local t1 = now_ms()
    vocab:detokenize(detok_tokens, detok_n, false, false)
    detok_times[#detok_times+1] = now_ms() - t1
end

record("detokenization", {
    n_tokens   = detok_n,
    median_ms  = round(median(detok_times), 4),
})

io.stderr:write("[bench] 8/14 sampler overhead...\n"); io.stderr:flush()
-- ── 8. Sampler chain overhead ─────────────────────────────────────────────────
-- Ion7-core specific: measure how much overhead the sampler chain adds
-- (not available in llama-cpp-python benchmarks - but good to show)

local profiles = {
    greedy   = ion7.Sampler.chain():greedy():build(vocab),
    minimal  = ion7.Sampler.chain():top_k(1):dist(42):build(vocab),
    standard = ion7.Sampler.chain():top_k(40):top_p(0.95,1):temperature(0.8):dist(42):build(vocab),
}

ctx:kv_clear()
ctx:decode(tokens, n, 0, 0)

local profile_results = {}
for name, samp in pairs(profiles) do
    local times = {}
    for _ = 1, args.n_repeat * 5 do
        local t1 = now_ms()
        samp:sample(ctx:ptr(), -1)
        times[#times+1] = now_ms() - t1
    end
    profile_results[name] = { median_ms = round(median(times), 4) }
    samp:free()
end

record("sampler_overhead", {
    note    = "ion7-core specific - time for one sample() call excluding decode",
    profiles = profile_results,
})

-- ── 9. Context creation ──────────────────────────────────────────────────────

io.stderr:write("[bench] 9/14 context creation...\n"); io.stderr:flush()

do
    local ok9, err9 = pcall(function()
        local function make_ctx_kv(kv_quant)
            return model:context({
                n_gpu_layers = ngl,
                n_ctx        = args.n_ctx,
                kv_quant     = kv_quant or "f16",
                n_ubatch     = args.n_ubatch,
                flash_attn   = args.flash_attn or nil,
                n_batch      = args.n_batch,
            })
        end

        local t_f16 = measure(function() local c = make_ctx_kv("f16");  c:free() end)
        local t_q8  = measure(function() local c = make_ctx_kv("q8_0"); c:free() end)
        local t_q4  = measure(function() local c = make_ctx_kv("q4_0"); c:free() end)

        record("context_creation", {
            cases = {
                { kv_quant = "f16",  median_ms = round(t_f16.median, 3), min_ms = round(t_f16.min, 3), max_ms = round(t_f16.max, 3) },
                { kv_quant = "q8_0", median_ms = round(t_q8.median, 3),  min_ms = round(t_q8.min, 3),  max_ms = round(t_q8.max, 3) },
                { kv_quant = "q4_0", median_ms = round(t_q4.median, 3),  min_ms = round(t_q4.min, 3),  max_ms = round(t_q4.max, 3) },
            },
        })
    end)
    if not ok9 then bench_err("context_creation", tostring(err9)) end
end

-- ── 10. decode_single (the hot path) ─────────────────────────────────────────

io.stderr:write("[bench] 10/14 decode_single...\n"); io.stderr:flush()

do
    local ok10, err10 = pcall(function()
        local tok = vocab:bos()
        local N   = 200

        -- With kv_clear between calls (worst case - no KV reuse)
        local t_nocache = measure(function()
            ctx:kv_clear()
            ctx._n_past = 0
            for _ = 1, N do
                ctx:decode_single(tok, 0)
                ctx:kv_clear()
                ctx._n_past = 0
            end
        end)

        -- With KV accumulation (realistic)
        local N_accum = math.min(N, args.n_ctx - 10)
        local t_cache = measure(function()
            ctx:kv_clear()
            ctx._n_past = 0
            for _ = 1, N_accum do
                ctx:decode_single(tok, 0)
            end
        end)

        record("decode_single", {
            no_cache = {
                n           = N,
                calls_per_s = round(N / (t_nocache.median / 1000), 1),
                median_ms   = round(t_nocache.median, 3),
            },
            with_cache = {
                n           = N_accum,
                calls_per_s = round(N_accum / (t_cache.median / 1000), 1),
                median_ms   = round(t_cache.median, 3),
            },
        })
    end)
    if not ok10 then bench_err("decode_single", tostring(err10)) end
end

-- ── 11. KV cache operations ──────────────────────────────────────────────────

io.stderr:write("[bench] 11/14 kv operations...\n"); io.stderr:flush()

do
    local ok11, err11 = pcall(function()
        -- Fill with some tokens first
        local kv_tokens, kv_n = vocab:tokenize("Hello world test", false, false)
        ctx:kv_clear()
        ctx:decode(kv_tokens, kv_n, 0, 0)

        local N_CLEAR = 10000
        local t_clear = measure(function()
            for _ = 1, N_CLEAR do ctx:kv_clear() end
        end)

        -- Refill after clear
        ctx:decode(kv_tokens, kv_n, 0, 0)
        local N_RM = 10000
        local t_rm = measure(function()
            for _ = 1, N_RM do ctx:kv_seq_rm(0, 0, -1) end
        end)

        -- seq_cp
        ctx:kv_clear()
        ctx:decode(kv_tokens, kv_n, 0, 0)
        local N_CP = 1000
        local t_cp = measure(function()
            for _ = 1, N_CP do ctx:kv_seq_cp(0, 1, 0, -1) end
        end)

        record("kv_operations", {
            kv_clear = {
                n           = N_CLEAR,
                calls_per_s = round(N_CLEAR / (t_clear.median / 1000), 0),
                median_ms   = round(t_clear.median, 3),
            },
            kv_seq_rm = {
                n           = N_RM,
                calls_per_s = round(N_RM / (t_rm.median / 1000), 0),
                median_ms   = round(t_rm.median, 3),
            },
            kv_seq_cp = {
                n           = N_CP,
                calls_per_s = round(N_CP / (t_cp.median / 1000), 0),
                median_ms   = round(t_cp.median, 3),
            },
        })
    end)
    if not ok11 then bench_err("kv_operations", tostring(err11)) end
end

-- ── 12. State persistence ────────────────────────────────────────────────────

io.stderr:write("[bench] 12/14 state persistence...\n"); io.stderr:flush()

do
    local ok12, err12 = pcall(function()
        local sp_tokens, sp_n = vocab:tokenize("State persistence benchmark test", false, false)
        ctx:kv_clear()
        ctx:decode(sp_tokens, sp_n, 0, 0)

        -- In-memory snapshot/restore
        local t_snap    = measure(function() ctx:snapshot() end)
        local blob      = ctx:snapshot()
        local t_restore = measure(function() ctx:restore(blob) end)

        -- File I/O round-trip
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

-- ── 13. Custom Lua sampler - callback overhead ───────────────────────────────

io.stderr:write("[bench] 13/14 custom sampler...\n"); io.stderr:flush()

do
    local ok13, err13 = pcall(function()
        local call_count = 0
        local lua_greedy = ion7.CustomSampler.new("bench_greedy", {
            apply = function(cur_p, nc)
                call_count = call_count + 1
                local best_i, best_v = 0, -math.huge
                local data = cur_p.data
                for ci = 0, nc - 1 do
                    if data[ci].logit > best_v then
                        best_v = data[ci].logit
                        best_i = ci
                    end
                end
                cur_p.selected = best_i
                return best_i
            end,
        })
        local lua_chain = ion7.Sampler.chain():custom(lua_greedy):build(vocab)

        -- Need a native greedy sampler for comparison
        local native_greedy = ion7.Sampler.chain():greedy():build(vocab)

        local cs_tokens, cs_n = vocab:tokenize("Custom sampler benchmark", false, false)
        ctx:kv_clear()
        ctx:decode(cs_tokens, cs_n, 0, 0)

        local N_CS = 1000
        call_count = 0
        local t_lua = measure(function()
            for _ = 1, N_CS do lua_chain:sample(ctx:ptr(), -1) end
        end)
        local t_native = measure(function()
            for _ = 1, N_CS do native_greedy:sample(ctx:ptr(), -1) end
        end)

        local overhead_pct = (t_lua.median / t_native.median - 1.0) * 100.0

        lua_chain:free()
        native_greedy:free()

        record("custom_sampler", {
            lua_greedy = {
                n             = N_CS,
                samples_per_s = round(N_CS / (t_lua.median / 1000), 0),
                median_ms     = round(t_lua.median, 3),
            },
            native_greedy = {
                n             = N_CS,
                samples_per_s = round(N_CS / (t_native.median / 1000), 0),
                median_ms     = round(t_native.median, 3),
            },
            overhead_pct = round(overhead_pct, 1),
        })
    end)
    if not ok13 then bench_err("custom_sampler", tostring(err13)) end
end

-- ── 14. Vocab operations ─────────────────────────────────────────────────────

io.stderr:write("[bench] 14/14 vocab operations...\n"); io.stderr:flush()

do
    local ok14, err14 = pcall(function()
        local N_BOS = 500000
        local t_bos = measure(function()
            for _ = 1, N_BOS do vocab:bos() end
        end)

        local N_EOG = 500000
        local bos_tok = vocab:bos()
        local t_eog = measure(function()
            for _ = 1, N_EOG do vocab:is_eog(bos_tok) end
        end)

        local N_PIECE = 50000
        local piece_tokens, piece_n = vocab:tokenize("Hello world test benchmark", false, false)
        local t_piece = measure(function()
            for _ = 1, N_PIECE do vocab:piece(piece_tokens[0], true) end
        end)

        record("vocab_operations", {
            bos = {
                n           = N_BOS,
                calls_per_s = round(N_BOS / (t_bos.median / 1000), 0),
            },
            is_eog = {
                n           = N_EOG,
                calls_per_s = round(N_EOG / (t_eog.median / 1000), 0),
            },
            piece = {
                n           = N_PIECE,
                calls_per_s = round(N_PIECE / (t_piece.median / 1000), 0),
            },
        })
    end)
    if not ok14 then bench_err("vocab_operations", tostring(err14)) end
end

-- ── Stress tests (--stress only) ─────────────────────────────────────────────

if args.stress then
    io.stderr:write("[bench] STRESS: back-to-back sessions...\n"); io.stderr:flush()

    -- S1. Sustained generation - long output, stable tok/s over time
    local LONG_PROMPT = "Write a detailed technical essay about transformer architecture, "
                     .. "attention mechanisms, and how large language models work. "
                     .. "Cover all major components including embeddings, positional "
                     .. "encoding, multi-head attention, feed-forward layers, and "
                     .. "the training process. Be thorough and technical."
    local msgs_long = { { role = "user", content = LONG_PROMPT } }
    local fmt_long  = vocab:apply_template(msgs_long, true)
    local tok_long, n_long = vocab:tokenize(fmt_long, false, true)
    local N_LONG = math.min(args.n_gen * 8, 512)

    local sustained_rates = {}
    local window_rates    = {}  -- tok/s per 32-token window

    ctx:kv_clear()
    ctx:decode(tok_long, n_long, 0, 0)
    sampler:reset()
    ctx:perf_reset()

    local window_start = now_ms()
    local window_count = 0
    local total_count  = 0

    for i = 1, N_LONG do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        total_count  = total_count + 1
        window_count = window_count + 1
        if window_count == 32 then
            local window_ms = now_ms() - window_start
            window_rates[#window_rates+1] = round(32 / (window_ms / 1000), 1)
            window_start = now_ms()
            window_count = 0
        end
    end

    local perf_long = ctx:perf()
    sustained_rates[#sustained_rates+1] = perf_long.tokens_per_s

    record("stress_sustained_generation", {
        n_tokens_generated = total_count,
        n_prompt_tokens    = n_long,
        overall_tok_s      = round(perf_long.tokens_per_s, 2),
        window_tok_s       = window_rates,  -- tok/s per 32-token window
        t_eval_ms          = round(perf_long.t_eval_ms, 2),
        note               = "32-token sliding window shows throughput stability over time",
    })

    -- S2. Back-to-back sessions with KV reuse
    io.stderr:write("[bench] STRESS: KV reuse back-to-back...\n"); io.stderr:flush()

    local N_SESSIONS = 10
    local sys_prompt  = "You are a helpful assistant."
    local user_turns  = {
        "What is 2+2?",
        "And what is 3+3?",
        "What about 10*10?",
        "Name the first 5 prime numbers.",
        "What color is the sky?",
        "How many days in a week?",
        "What is the capital of France?",
        "How many continents are there?",
        "What is pi approximately?",
        "Is water wet?",
    }

    local session_times = {}
    local reuse_hits    = 0

    local prev_snap = nil
    local prev_n    = 0

    for i = 1, N_SESSIONS do
        local turn = user_turns[i]
        local msgs_s2 = {
            { role = "system", content = sys_prompt },
            { role = "user",   content = turn },
        }
        local fmt_s2 = vocab:apply_template(msgs_s2, true)
        local tok_s2, n_s2 = vocab:tokenize(fmt_s2, false, true)

        ctx:kv_clear()
        local t1 = now_ms()
        ctx:decode(tok_s2, n_s2, 0, 0)
        sampler:reset()

        local parts = {}
        for _ = 1, 16 do
            local tok = sampler:sample(ctx:ptr(), -1)
            if vocab:is_eog(tok) then break end
            ctx:decode_single(tok, 0)
            parts[#parts+1] = vocab:piece(tok)
        end
        local session_ms = now_ms() - t1

        local snap = ctx:snapshot()
        session_times[#session_times+1] = round(session_ms, 2)
        prev_snap = snap
        prev_n    = n_s2
    end

    record("stress_back_to_back", {
        n_sessions        = N_SESSIONS,
        session_ms        = session_times,
        median_session_ms = round(median(session_times), 2),
        min_session_ms    = round(math.min(table.unpack(session_times)), 2),
        max_session_ms    = round(math.max(table.unpack(session_times)), 2),
        note              = "prefill + 16 tokens per session, no KV reuse (fresh kv_clear each)",
    })

    -- S3. Context fill stress - fill to 75% of n_ctx, measure perf degradation
    io.stderr:write("[bench] STRESS: context fill pressure...\n"); io.stderr:flush()

    local ctx_target   = math.floor(args.n_ctx * 0.75)
    local filler_text  = ("The quick brown fox jumps over the lazy dog. "):rep(100)
    local filler_tok, filler_n = vocab:tokenize(filler_text, false, false)
    filler_n = math.min(filler_n, ctx_target - 32)

    ctx:kv_clear()
    ctx:perf_reset()

    -- Fill context
    local t_fill_start = now_ms()
    ctx:decode(filler_tok, filler_n, 0, 0)
    local fill_ms = now_ms() - t_fill_start

    -- Now generate on a nearly-full context
    sampler:reset()
    local t_gen_start = now_ms()
    local gen_on_full = 0
    for _ = 1, 32 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        ctx:decode_single(tok, 0)
        gen_on_full = gen_on_full + 1
    end
    local gen_full_ms = now_ms() - t_gen_start

    record("stress_context_pressure", {
        n_ctx              = args.n_ctx,
        fill_tokens        = filler_n,
        fill_pct           = round(filler_n / args.n_ctx * 100, 1),
        prefill_ms         = round(fill_ms, 2),
        prefill_tok_s      = round(filler_n / (fill_ms / 1000), 1),
        gen_on_full_ms     = round(gen_full_ms, 2),
        gen_on_full_tok_s  = round(gen_on_full / (gen_full_ms / 1000), 1),
        note               = "prefill + generation at 75% context fill",
    })

    -- S4. Sampler chain throughput under load
    io.stderr:write("[bench] STRESS: sampler throughput...\n"); io.stderr:flush()

    local sampler_stress = ion7.Sampler.chain()
        :top_k(40):top_p(0.95, 1):temperature(0.8):dist(args.seed):build(vocab)

    local N_SAMPLES = 1000
    ctx:kv_clear()
    ctx:decode(tok_long, math.min(n_long, 64), 0, 0)

    local t_samp = now_ms()
    for _ = 1, N_SAMPLES do
        sampler_stress:sample(ctx:ptr(), -1)
    end
    local samp_total_ms = now_ms() - t_samp
    sampler_stress:free()

    record("stress_sampler_throughput", {
        n_samples       = N_SAMPLES,
        total_ms        = round(samp_total_ms, 2),
        avg_us_per_call = round(samp_total_ms * 1000 / N_SAMPLES, 2),
        samples_per_s   = round(N_SAMPLES / (samp_total_ms / 1000), 0),
        note            = "pure sample() calls, no decode between them",
    })
end

-- ── Cleanup ───────────────────────────────────────────────────────────────────

sampler:free()
ctx:free()
model:free()
ion7.shutdown()

-- ── Output ────────────────────────────────────────────────────────────────────

local output = json.encode(RESULTS) .. "\n"
io.stdout:write(output)
io.stdout:flush()
