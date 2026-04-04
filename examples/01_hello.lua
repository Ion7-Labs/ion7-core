#!/usr/bin/env luajit

--- 01_hello.lua - Minimal inference. Load a model and generate one response.
---
--- Usage:
---   ION7_MODEL=/path/to/model.gguf luajit examples/01_hello.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local ion7 = require "ion7.core"

-- Initialise the backend (silent mode)
ion7.init({ log_level = 0 })

-- Auto-fit: detect how many GPU layers and context size fit in VRAM
local MODEL = assert(os.getenv("ION7_MODEL"), "Set ION7_MODEL=/path/to/model.gguf")
local fit   = ion7.Model.fit_params(MODEL)

-- Load the model
local model = ion7.Model.load(MODEL, { n_gpu_layers = fit and fit.n_gpu_layers or 0 })
local vocab  = model:vocab()
local ctx    = model:context({ n_ctx = 512 })
local sampler = ion7.Sampler.chain()
    :top_k(40):top_p(0.95):temperature(0.8):dist(42)
    :build(vocab)

-- Format and tokenise the prompt
local msgs   = { { role = "user", content = "What is LuaJIT in one sentence?" } }
local prompt = vocab:apply_template(msgs, true)
local tokens, n = vocab:tokenize(prompt, false, true)

-- Prefill
ctx:decode(tokens, n, 0, 0)

-- Generate up to 128 tokens
io.write("Assistant: ")
for _ = 1, 128 do
    local tok = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(tok) then break end
    io.write(vocab:piece(tok, true))
    io.flush()
    ctx:decode_single(tok, 0)
end
io.write("\n")

ion7.shutdown()
