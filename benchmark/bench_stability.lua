#!/usr/bin/env luajit
--- ion7-core stability benchmark.
--- Runs a generation loop for N tokens and monitors RSS memory.
--- A stable RSS = no memory leaks. That's the claim.

---
--- Copyright (C) 2026 Ion7 Project Contributors
--- SPDX-License-Identifier: AGPL-3.0-or-later
---

--- Run:
---   ION7_MODEL=/path/to/model.gguf luajit bench/bench_stability.lua
---
--- Optional:
---   ION7_TOKENS=100000   (default: 100000)
---   ION7_REPORT=1000     (print stats every N tokens, default: 5000)

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL_PATH  = os.getenv("ION7_MODEL")
local N_TOKENS    = tonumber(os.getenv("ION7_TOKENS"))  or 100000
local REPORT_EVERY = tonumber(os.getenv("ION7_REPORT")) or 5000

if not MODEL_PATH then
    io.stderr:write("Usage: ION7_MODEL=/path/to/model.gguf luajit bench/bench_stability.lua\n")
    os.exit(1)
end

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

-- ── RSS helper (Linux /proc/self/status) ──────────────────────────────────────

local function rss_kb()
    local f = io.open("/proc/self/status", "r")
    if not f then return 0 end
    for line in f:lines() do
        local v = line:match("^VmRSS:%s+(%d+)")
        if v then f:close(); return tonumber(v) end
    end
    f:close()
    return 0
end

local function now_ms()
    return os.clock() * 1000.0
end

-- ── Setup ─────────────────────────────────────────────────────────────────────

local fit   = ion7.Model.fit_params(MODEL_PATH)
local model = ion7.Model.load(MODEL_PATH, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
local ctx   = model:context({ n_ctx = math.min(fit and fit.n_ctx or 4096, 4096) })
local vocab = model:vocab()
local sampler = ion7.Sampler.chain():top_k(50):temp(0.8):dist(42):build(vocab)

-- Simple rotating prompt to keep generation going
local prompts = {
    "Tell me a short story about",
    "Explain in one paragraph why",
    "List three facts about",
    "Describe briefly the concept of",
}

local rss_start    = rss_kb()
local t_start      = now_ms()
local t_last       = t_start   -- for per-interval delta
local total_tokens = 0
local total_resets = 0
local rss_min      = rss_start
local rss_max      = rss_start

io.write(string.format(
    "\n\27[1mion7-core stability benchmark\27[0m\n" ..
    "  model:    %s\n" ..
    "  target:   %d tokens\n" ..
    "  rss_start: %.1f MB\n" ..
    "%s\n",
    MODEL_PATH:match("[^/]+$"),
    N_TOKENS,
    rss_start / 1024,
    string.rep("─", 72)
))

io.write(string.format("  %-8s  %-8s  %-8s  %-8s  %-10s  %-8s  %-8s\n",
    "tokens", "tok/s", "rss_MB", "rss_Δ", "t_total", "t_Δ", "resets"))
io.write(string.rep("─", 72) .. "\n")

-- ── Generation loop ───────────────────────────────────────────────────────────

local prompt_idx = 1

while total_tokens < N_TOKENS do
    -- Pick next prompt. /no_think suppresses Qwen3 thinking blocks which
    -- would fill the context in 1-2 prompts and distort generation speed.
    local prompt_text = "/no_think " .. prompts[prompt_idx] .. " the universe"
    prompt_idx = (prompt_idx % #prompts) + 1

    local msgs = { { role = "user", content = prompt_text } }
    local formatted = vocab:apply_template(msgs, true)
    local tokens, n_prompt = vocab:tokenize(formatted, false, true)

    -- Always start fresh. The stability bench measures memory stability,
    -- not KV sliding window. Avoiding n_past drift keeps the bench clean.
    ctx:kv_clear()
    total_resets = total_resets + 1
    ctx:decode(tokens, n_prompt, 0, 0)
    sampler:reset()

    -- Generate up to 128 tokens per prompt or until EOG
    local gen_count = 0
    for _ = 1, 128 do
        local tok = sampler:sample(ctx:ptr(), -1)
        if vocab:is_eog(tok) then break end
        sampler:accept(tok)
        ctx:decode_single(tok, 0)
        gen_count = gen_count + 1
        total_tokens = total_tokens + 1

        -- Report every REPORT_EVERY tokens
        if total_tokens % REPORT_EVERY == 0 then
            local rss     = rss_kb()
            local elapsed = (now_ms() - t_start) / 1000.0
            local tps     = total_tokens / elapsed
            rss_min = math.min(rss_min, rss)
            rss_max = math.max(rss_max, rss)
            local dt = (now_ms() - t_last) / 1000.0
            t_last = now_ms()
            io.write(string.format(
                "  %-8d  %-8.1f  %-8.1f  %-8.1f  %-10.1fs  %-7.1fs  %-8d\n",
                total_tokens,
                tps,
                rss / 1024,
                (rss - rss_start) / 1024,
                elapsed,
                dt,
                total_resets
            ))
            io.flush()
        end

        if total_tokens >= N_TOKENS then break end
    end
end

-- ── Final report ──────────────────────────────────────────────────────────────

local rss_end  = rss_kb()
local elapsed  = (now_ms() - t_start) / 1000.0
local tps_avg  = total_tokens / elapsed

io.write(string.rep("─", 72) .. "\n")
io.write(string.format(
    "\n\27[1mResults - %d tokens\27[0m\n",
    total_tokens
))
io.write(string.format("  avg throughput:  %.2f tok/s\n", tps_avg))
io.write(string.format("  total time:      %.1f s\n", elapsed))
io.write(string.format("  context resets:  %d\n", total_resets))
io.write(string.format("  rss start:       %.1f MB\n", rss_start / 1024))
io.write(string.format("  rss end:         %.1f MB\n", rss_end   / 1024))
io.write(string.format("  rss min:         %.1f MB\n", rss_min   / 1024))
io.write(string.format("  rss max:         %.1f MB\n", rss_max   / 1024))

local drift = rss_end - rss_start
-- Measure drift AFTER the first report (context creation is one-time)
-- A true leak grows continuously; a one-time jump then flat is normal.
local drift_after_warmup = rss_max - rss_min
local verdict
if drift_after_warmup < 5 * 1024 then       -- < 5 MB variation after warmup
    verdict = "\27[32m✓ STABLE\27[0m  (one-time context alloc: " ..
              string.format("%.1f", drift/1024) .. " MB)"
elseif drift_after_warmup < 20 * 1024 then  -- < 20 MB variation
    verdict = "\27[33m~ ACCEPTABLE\27[0m  (small variance: " ..
              string.format("%.1f", drift_after_warmup/1024) .. " MB)"
else
    verdict = "\27[31m⚠ POSSIBLE LEAK\27[0m  (growing: +" ..
              string.format("%.1f", drift_after_warmup/1024) .. " MB)"
end
io.write(string.format(
    "\n  \27[1mMemory drift: %+.1f MB total\27[0m  |  post-warmup variation: %.1f MB\n  %s\n",
    drift / 1024, drift_after_warmup / 1024, verdict
))

io.write(string.rep("─", 72) .. "\n")
ion7.shutdown()
