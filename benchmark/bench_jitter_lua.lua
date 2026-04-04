--- benchmark/bench_jitter_lua.lua - timing stability benchmark (ion7-core).

---
--- Copyright (C) 2026 Ion7 Project Contributors
--- SPDX-License-Identifier: AGPL-3.0-or-later
---

--- Runs N iterations of each operation and records every individual timing.
--- Outputs JSON with full timing arrays for statistical analysis.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit benchmark/bench_jitter_lua.lua
---   ION7_MODEL=... luajit benchmark/bench_jitter_lua.lua --n-gpu-layers 32 --n-iter 100

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

-- ── CLI ───────────────────────────────────────────────────────────────────────

local args = { n_gpu_layers = nil, n_ctx = 2048, n_iter = 100, seed = 42 }
local i = 1
while i <= #arg do
    if     arg[i] == "--n-gpu-layers" then args.n_gpu_layers = tonumber(arg[i+1]); i=i+2
    elseif arg[i] == "--n-ctx"        then args.n_ctx        = tonumber(arg[i+1]); i=i+2
    elseif arg[i] == "--n-iter"       then args.n_iter       = tonumber(arg[i+1]); i=i+2
    elseif arg[i] == "--seed"         then args.seed         = tonumber(arg[i+1]); i=i+2
    else i=i+1 end
end

-- ── Dependencies ──────────────────────────────────────────────────────────────

local ok_json, json = pcall(require, "dkjson")
if not ok_json then ok_json, json = pcall(require, "cjson") end
if not ok_json then io.stderr:write("Need dkjson or cjson\n"); os.exit(1) end

local MODEL   = os.getenv("ION7_MODEL")
local LIB_DIR = os.getenv("ION7_LIB_DIR")
if not MODEL then
    io.stdout:write(json.encode({ error = "Set ION7_MODEL" }) .. "\n"); os.exit(1)
end

local ion7 = require "ion7.core"
local function find_so(hint)
    if not hint then return nil end
    if hint:match("%.so[%.%d]*$") then return hint end
    for _, p in ipairs({
        hint.."/libllama.so", hint.."/bin/libllama.so",
        hint:gsub("/bin$","").."/libllama.so",
    }) do
        local f = io.open(p, "rb"); if f then f:close(); return p end
    end
end
local llama_so = find_so(LIB_DIR) or os.getenv("LLAMA_LIB")
ion7.init({ log_level = 0, llama_path = llama_so })

-- ── Time ──────────────────────────────────────────────────────────────────────

local _time_fn
if type(ion7.time_us) == "function" then
    local ok, v = pcall(ion7.time_us)
    if ok and type(v) == "number" then
        _time_fn = function() return ion7.time_us() / 1000.0 end
    end
end
if not _time_fn then
    local ok, socket = pcall(require, "socket")
    if ok and socket.gettime then _time_fn = function() return socket.gettime() * 1000.0 end end
end
if not _time_fn then _time_fn = function() return os.clock() * 1000.0 end end
local function now_ms() return _time_fn() end

-- ── Stats ─────────────────────────────────────────────────────────────────────

local function stats(times)
    local n = #times
    if n == 0 then return {} end
    local s = {}
    for _, v in ipairs(times) do s[#s+1] = v end
    table.sort(s)

    local sum = 0; for _, v in ipairs(s) do sum = sum + v end
    local mean = sum / n

    local variance = 0
    for _, v in ipairs(s) do variance = variance + (v - mean)^2 end
    local stddev = math.sqrt(variance / n)

    local function percentile(p)
        local idx = math.max(1, math.ceil(p / 100 * n))
        return s[idx]
    end

    return {
        n       = n,
        mean    = math.floor(mean   * 1000 + 0.5) / 1000,
        median  = math.floor(percentile(50) * 1000 + 0.5) / 1000,
        p95     = math.floor(percentile(95) * 1000 + 0.5) / 1000,
        p99     = math.floor(percentile(99) * 1000 + 0.5) / 1000,
        min     = math.floor(s[1]   * 1000 + 0.5) / 1000,
        max     = math.floor(s[n]   * 1000 + 0.5) / 1000,
        stddev  = math.floor(stddev * 1000 + 0.5) / 1000,
        cv_pct  = math.floor((stddev / mean) * 10000 + 0.5) / 100,
        times   = times,
    }
end

-- ── Setup ─────────────────────────────────────────────────────────────────────

io.stderr:write("[jitter] loading model...\n"); io.stderr:flush()
local fit = ion7.Model.fit_params(MODEL)
local ngl = args.n_gpu_layers or (fit and fit.n_gpu_layers or 0)
local model = ion7.Model.load(MODEL, { n_gpu_layers = ngl })
local vocab = model:vocab()
local ctx   = model:context({ n_ctx = args.n_ctx })
local N     = args.n_iter

local RESULTS = {
    backend  = "ion7-core (LuaJIT)",
    n_iter   = N,
    model    = MODEL:match("[^/]+$"),
    ops      = {},
}

-- ── Operations ────────────────────────────────────────────────────────────────

-- Helper: warm up, then time N iterations
local function measure_n(setup_fn, op_fn)
    if setup_fn then setup_fn() end
    op_fn()  -- warm-up (not recorded)
    collectgarbage("collect")
    local times = {}
    for _ = 1, N do
        if setup_fn then setup_fn() end
        local t0 = now_ms()
        op_fn()
        times[#times+1] = now_ms() - t0
        collectgarbage("collect")
    end
    return times
end

-- 1. Tokenization - short text
io.stderr:write("[jitter] 1/6 tokenization...\n"); io.stderr:flush()
local short_text = "Hello world"
RESULTS.ops.tokenize_short = stats(measure_n(nil, function()
    vocab:tokenize(short_text, false, false)
end))

-- 2. Tokenization - medium text
local medium_text = ("The quick brown fox jumps over the lazy dog. "):rep(10)
RESULTS.ops.tokenize_medium = stats(measure_n(nil, function()
    vocab:tokenize(medium_text, false, false)
end))

-- 3. Prefill - small prompt
io.stderr:write("[jitter] 2/6 prefill small...\n"); io.stderr:flush()
local small_prompt_msgs = {{ role="user", content="What is 2+2?" }}
local small_fmt   = vocab:apply_template(small_prompt_msgs, true)
local small_toks, small_n = vocab:tokenize(small_fmt, false, true)
RESULTS.ops.prefill_small = stats(measure_n(
    function() ctx:kv_clear() end,
    function() ctx:decode(small_toks, small_n, 0, 0) end
))
RESULTS.ops.prefill_small.prompt_tokens = small_n

-- 4. Prefill - medium prompt
io.stderr:write("[jitter] 3/6 prefill medium...\n"); io.stderr:flush()
local med_prompt_msgs = {{ role="user", content=("Explain transformers. "):rep(8) }}
local med_fmt  = vocab:apply_template(med_prompt_msgs, true)
local med_toks, med_n = vocab:tokenize(med_fmt, false, true)
RESULTS.ops.prefill_medium = stats(measure_n(
    function() ctx:kv_clear() end,
    function() ctx:decode(med_toks, med_n, 0, 0) end
))
RESULTS.ops.prefill_medium.prompt_tokens = med_n

-- 5. Single-token generation (decode_single hot path)
io.stderr:write("[jitter] 4/6 decode_single...\n"); io.stderr:flush()
local sampler = ion7.Sampler.chain():top_k(1):dist(args.seed):build(vocab)
ctx:kv_clear()
ctx:decode(small_toks, small_n, 0, 0)
local last_tok = sampler:sample(ctx:ptr(), -1)

RESULTS.ops.decode_single = stats(measure_n(
    function()
        ctx:kv_clear()
        ctx:decode(small_toks, small_n, 0, 0)
        last_tok = sampler:sample(ctx:ptr(), -1)
        sampler:reset()
    end,
    function()
        ctx:decode_single(last_tok, 0)
    end
))

-- 6. Detokenization
io.stderr:write("[jitter] 5/6 detokenization...\n"); io.stderr:flush()
local dtok_text = "Hello world this is a test of detokenization speed"
local dtok_tokens, dtok_n = vocab:tokenize(dtok_text, false, false)
RESULTS.ops.detokenize = stats(measure_n(nil, function()
    vocab:detokenize(dtok_tokens, dtok_n, false, false)
end))

-- 6. KV snapshot (in-memory)
io.stderr:write("[jitter] 6/6 kv snapshot...\n"); io.stderr:flush()
ctx:kv_clear()
ctx:decode(small_toks, small_n, 0, 0)
local blob = ctx:snapshot()
RESULTS.ops.kv_snapshot_save = stats(measure_n(nil, function()
    ctx:snapshot()
end))
RESULTS.ops.kv_snapshot_load = stats(measure_n(nil, function()
    ctx:restore(blob)
end))

-- ── Cleanup ───────────────────────────────────────────────────────────────────

sampler:free(); ctx:free(); model:free(); ion7.shutdown()

io.stdout:write(json.encode(RESULTS) .. "\n")
io.stdout:flush()
