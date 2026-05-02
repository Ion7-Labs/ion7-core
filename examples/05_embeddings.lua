#!/usr/bin/env luajit
--- @example examples.05_embeddings
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 05 — Embeddings : encode text into vectors and rank by similarity
--- ════════════════════════════════════════════════════════════════════════
---
--- An embedding model compresses any string into a single dense
--- vector ; cosine similarity between vectors approximates semantic
--- closeness of the source texts. This is the foundation of RAG
--- (retrieve relevant snippets) and semantic search.
---
--- The example loads an embedding-tuned GGUF (set `ION7_EMBED`),
--- encodes a small set of sentences, and prints the cosine similarity
--- matrix. Pairs that mean the same thing should land near 1.0 ;
--- unrelated pairs should land near 0.
---
---   ION7_EMBED=/path/to/embed.gguf luajit examples/05_embeddings.lua
---
--- Recommended embedding models (any GGUF with `general.architecture`
--- set to `bert`, `nomic-bert`, `gte-qwen2`, ...) :
---   - nomic-embed-text-v1.5 (768 dim, English)
---   - bge-small-en-v1.5      (384 dim, English)
---   - multilingual-e5-base   (768 dim, 100+ languages)

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local EMBED = os.getenv("ION7_EMBED") or
    error("Set ION7_EMBED=/path/to/embed.gguf", 0)

local ion7 = require "ion7.core"
ion7.init({ log_level = 0 })

-- Embeddings run fine on CPU ; many embedders are too small to benefit
-- from GPU offload. Override with ION7_GPU_LAYERS if you have a GPU.
local model = ion7.Model.load(EMBED, {
    n_gpu_layers = tonumber(os.getenv("ION7_GPU_LAYERS") or "0") or 0,
})
local vocab = model:vocab()

-- Pooling strategy depends on the architecture :
--   - BERT / RoBERTa derivatives    : "cls"
--   - GTE / E5 / Qwen3-Embedding    : "last"
--   - SBERT / Sentence-Transformers : "mean"
-- "last" is the safest default for modern embedders ; flip to "cls"
-- if your similarities all come out near 1.0 (a giveaway that the
-- pooling is wrong).
local ctx = model:embedding_context({
    n_ctx     = 512,
    pooling   = "last",
    n_seq_max = 1,
})

-- The output dimension is what determines the cosine similarity work.
-- Some models report a different output size from their internal
-- embedding (when a projection head fans out wider).
local n_embd = model:n_embd_out() > 0 and model:n_embd_out() or model:n_embd()

print(string.format("[embedding model] %s — dim = %d, pooling = %s",
    model:meta("general.name") or model.path:match("[^/]+$"),
    n_embd, ctx:pooling_type()))

--- Embed `text` and return the pooled vector as a Lua table of floats.
--- The context is reused across calls — `kv_clear()` resets the cache
--- between texts so each one gets a clean state.
local function embed(text)
    local toks, n = vocab:tokenize(text, true, true)
    -- Embedding contexts have small windows ; truncate hard rather
    -- than crash on a long input.
    if n > ctx:n_ctx() then n = ctx:n_ctx() end
    ctx:kv_clear()
    ctx:decode(toks, n)
    -- `embedding(seq_id, dim)` copies the pooled vector out as a
    -- regular Lua table so we can hold it past the next decode.
    return ctx:embedding(0, n_embd)
end

--- Standard cosine similarity, in `[-1, 1]`.
local function cosine(a, b)
    local dot, na, nb = 0, 0, 0
    for i = 1, #a do
        dot = dot + a[i] * b[i]
        na  = na  + a[i] * a[i]
        nb  = nb  + b[i] * b[i]
    end
    return dot / (math.sqrt(na) * math.sqrt(nb))
end

-- ── Demo : six sentences, three thematic pairs ──────────────────────────
local texts = {
    "LuaJIT is a fast just-in-time compiler for Lua.",
    "llama.cpp runs large language models on consumer hardware.",
    "The cat sat on the mat.",
    "A small feline rested on a rug.",
    "Neural networks learn patterns from training data.",
    "Deep learning models are trained on large datasets.",
}

print("\n[embedding " .. #texts .. " sentences]")
local vecs = {}
for i, t in ipairs(texts) do
    vecs[i] = embed(t)
    print(string.format("  [%d] %s", i, t))
end

print("\n[cosine similarity matrix]")
io.write("       ")
for j = 1, #texts do io.write(string.format("  [%d] ", j)) end
io.write("\n")
for i = 1, #texts do
    io.write(string.format("  [%d] ", i))
    for j = 1, #texts do
        local sim = (i == j) and 1.0 or cosine(vecs[i], vecs[j])
        if i == j then
            io.write("   -- ")
        elseif sim > 0.8 then
            io.write(string.format(" \027[32m%5.2f\027[0m", sim))
        elseif sim < 0.4 then
            io.write(string.format(" \027[31m%5.2f\027[0m", sim))
        else
            io.write(string.format(" %5.2f", sim))
        end
    end
    io.write("\n")
end
print("\n  green ≥ 0.80 = clearly similar    red ≤ 0.40 = clearly different")
print("  pairs (3,4) and (5,6) are paraphrases ; pairs across themes should be low.")

ctx:free() ; model:free()
ion7.shutdown()
