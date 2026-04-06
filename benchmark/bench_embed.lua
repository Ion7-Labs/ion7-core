#!/usr/bin/env luajit
---
--- Copyright (C) 2026 Ion7 Project Contributors
--- SPDX-License-Identifier: MIT
---

--- ion7-core embedding benchmark.
--- Loads the embedding model on clean VRAM (inference model NOT loaded).
---
--- Run:
---   ION7_EMBED=/path/to/embed.gguf luajit bench/bench_embed.lua
---
--- Optional:
---   ION7_LIB_DIR=/path/to/llama.cpp/build/bin
---   ION7_RUNS=3

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local EMBED_PATH = os.getenv("ION7_EMBED")
local LIB_DIR    = os.getenv("ION7_LIB_DIR")
local N_RUNS     = tonumber(os.getenv("ION7_RUNS")) or 3

if not EMBED_PATH then
    io.stderr:write("Usage: ION7_EMBED=/path/to/embed.gguf luajit bench/bench_embed.lua\n")
    os.exit(1)
end

local ion7 = require "ion7.core"
ion7.init({ log_level = 0, llama_path = LIB_DIR, bridge_path = LIB_DIR })

local function now_ms() return os.clock() * 1000.0 end
local W = 52

local function header(title)
    io.write(string.format("\n\27[1m── %s \27[0m%s\n", title,
        string.rep("─", W - #title - 1)))
end

local function row(label, value, unit, extra)
    io.write(string.format("  %-44s %8.2f  %-8s %s\n",
        label, value, unit, extra or ""))
end

local function measure(fn)
    fn()
    collectgarbage("collect")
    local times = {}
    for i = 1, N_RUNS do
        local t0 = now_ms()
        fn()
        times[#times + 1] = now_ms() - t0
        collectgarbage("collect")
    end
    table.sort(times)
    return {
        min    = times[1],
        median = times[math.ceil(#times / 2)],
        max    = times[#times],
    }
end

local cap = ion7.capabilities()
io.write(string.format(
    "\n\27[1mion7-core embedding benchmark\27[0m\n" ..
    "  bridge:  %s\n" ..
    "  gpu:     %s\n" ..
    "  model:   %s\n" ..
    "  runs:    %d\n" ..
    "%s\n",
    cap.bridge_ver, tostring(cap.gpu_offload),
    EMBED_PATH:match("[^/]+$"), N_RUNS,
    string.rep("─", W + 22)
))

-- Load embed model on clean VRAM
local emodel = ion7.Model.load(EMBED_PATH, { n_gpu_layers = -1 })
local ectx1  = emodel:embedding_context({ n_ctx = 512, pooling = "last", n_seq_max = 1 })
local ectx4  = emodel:embedding_context({ n_ctx = 512, pooling = "last", n_seq_max = 4 })
local evocab = emodel:vocab()

local n_embd = emodel:n_embd_out() > 0 and emodel:n_embd_out() or emodel:n_embd()
io.write(string.format("  n_embd=%d  n_tokens=%d\n", n_embd, evocab:n_tokens()))

local Loader = require "ion7.core.ffi.loader"
local lib    = Loader.instance().lib

local texts = {
    "The quick brown fox jumps over the lazy dog.",
    "LuaJIT is a fast just-in-time compiler for Lua.",
    "llama.cpp runs LLMs locally on consumer hardware.",
    "Embeddings capture semantic meaning in vector space.",
}

local function embed_texts(ctx, n)
    local texts_to_use = {}
    for i = 1, n do texts_to_use[i] = texts[((i-1) % #texts) + 1] end
    for _, text in ipairs(texts_to_use) do
        local toks, tok_n = evocab:tokenize(text, false, true)
        tok_n = math.min(tok_n, ctx:n_ctx() - 1)
        local batch = lib.llama_batch_init(tok_n, 0, 1)
        for i = 0, tok_n - 1 do
            batch.token[i]     = toks[i]
            batch.pos[i]       = i
            batch.n_seq_id[i]  = 1
            batch.seq_id[i][0] = 0
            batch.logits[i]    = (i == tok_n - 1) and 1 or 0
        end
        batch.n_tokens = tok_n
        ctx:kv_clear()
        lib.llama_decode(ctx:ptr(), batch)
        lib.llama_batch_free(batch)
    end
end

header("1. Single embedding (sequential)")

for _, n in ipairs({ 1, 4, 16 }) do
    local t = measure(function() embed_texts(ectx1, n) end)
    row(string.format("sequential  %2d text(s)", n),
        n / (t.median / 1000), "embeds/s",
        string.format("(%.1fms each)", t.median / n))
end

header("2. Batch embedding (parallel, n_seq_max=4)")

if ectx4:n_seq_max() >= 4 then
    for _, n in ipairs({ 1, 4, 8, 16 }) do
        local t_seq = measure(function() embed_texts(ectx1, n) end)
        local t_par = measure(function() embed_texts(ectx4, n) end)
        local speedup = t_seq.median / t_par.median
        row(string.format("parallel  %2d text(s)  vs sequential", n),
            n / (t_par.median / 1000), "embeds/s",
            string.format("%.2fx speedup", speedup))
    end
else
    io.write("  [SKIP] n_seq_max < 4\n")
end

header("3. Throughput at scale")

for _, n in ipairs({ 32, 64, 128 }) do
    local t = measure(function() embed_texts(ectx4, n) end)
    row(string.format("batch %3d texts  (parallel)", n),
        n / (t.median / 1000), "embeds/s")
end

io.write("\n" .. string.rep("─", W + 22) .. "\n")
ion7.shutdown()
