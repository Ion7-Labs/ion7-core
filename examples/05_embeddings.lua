#!/usr/bin/env luajit
--- 05_embeddings.lua - Semantic similarity with embedding models.
---
--- Uses a separate embedding model (e.g. Qwen3-Embedding) to encode texts
--- into dense vectors, then computes cosine similarity between them.
---
--- This is the foundation of RAG (Retrieval-Augmented Generation).
---
--- Usage:
---   ION7_EMBED=/path/to/embed.gguf luajit examples/05_embeddings.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

ion7.init({ log_level = 0 })

local EMBED_PATH = assert(os.getenv("ION7_EMBED"), "Set ION7_EMBED=/path/to/embed.gguf")

local model = ion7.Model.load(EMBED_PATH, { n_gpu_layers = 0 })  -- CPU for embedding
local vocab  = model:vocab()
local ctx    = model:embedding_context({
    n_ctx    = 512,
    pooling  = "last",   -- Qwen3-Embedding, GTE, E5 → "last"
                         -- BERT, RoBERTa           → "cls"
    n_seq_max = 1,
})
local n_embd = model:n_embd_out() > 0 and model:n_embd_out() or model:n_embd()

io.write(string.format("[embedding model: %s  dim=%d]\n\n",
    model:meta_val("general.name") or EMBED_PATH:match("[^/]+$"), n_embd))

--- Embed a single text. Returns a Lua table of floats.
local function embed(text)
    local toks, n = vocab:tokenize(text, false, true)
    n = math.min(n, ctx:n_ctx() - 1)

    -- Build a raw batch via the FFI loader
    local Loader = require "ion7.core.ffi.loader"
    local lib = Loader.instance().lib
    local batch = lib.llama_batch_init(n, 0, 1)
    for i = 0, n - 1 do
        batch.token[i]     = toks[i]
        batch.pos[i]       = i
        batch.n_seq_id[i]  = 1
        batch.seq_id[i][0] = 0
        batch.logits[i]    = (i == n - 1) and 1 or 0
    end
    batch.n_tokens = n
    ctx:kv_clear()
    lib.llama_decode(ctx:ptr(), batch)
    lib.llama_batch_free(batch)

    -- Extract the pooled embedding (seq 0)
    return ctx:embedding(0, n_embd)
end

--- Cosine similarity between two float tables.
local function cosine(a, b)
    local dot, na, nb = 0, 0, 0
    for i = 1, #a do
        dot = dot + a[i] * b[i]
        na  = na  + a[i] * a[i]
        nb  = nb  + b[i] * b[i]
    end
    return dot / (math.sqrt(na) * math.sqrt(nb))
end

-- ── Semantic similarity demo ──────────────────────────────────────────────────

local texts = {
    "LuaJIT is a fast just-in-time compiler for Lua.",
    "llama.cpp runs large language models on consumer hardware.",
    "The cat sat on the mat.",
    "A small feline rested on a rug.",
    "Neural networks learn patterns from training data.",
    "Deep learning models are trained on large datasets.",
}

io.write("[Computing embeddings...]\n")
local vecs = {}
for i, text in ipairs(texts) do
    vecs[i] = embed(text)
    io.write(string.format("  [%d/6] %s\n", i, text:sub(1, 50)))
end

io.write("\n[Cosine similarity matrix]\n")
io.write(string.format("  %-52s", ""))
for i = 1, #texts do io.write(string.format(" [%d] ", i)) end
io.write("\n")

for i = 1, #texts do
    io.write(string.format("  [%d] %-50s", i, texts[i]:sub(1, 48) .. ".."))
    for j = 1, #texts do
        local sim = cosine(vecs[i], vecs[j])
        if i == j then
            io.write("  1.00")
        elseif sim > 0.8 then
            io.write(string.format(" \27[32m%.2f\27[0m", sim))   -- green: similar
        elseif sim < 0.4 then
            io.write(string.format(" \27[31m%.2f\27[0m", sim))   -- red: dissimilar
        else
            io.write(string.format("  %.2f", sim))
        end
    end
    io.write("\n")
end

io.write("\n[Key observations]\n")
io.write("  Pairs 3-4 (cat/feline) and 5-6 (neural net/deep learning) should be highly similar.\n")
io.write("  Pairs 1-3 (Lua vs cat) should be dissimilar.\n")

ion7.shutdown()
