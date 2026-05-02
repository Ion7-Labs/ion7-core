#!/usr/bin/env luajit
--- @example examples.01_hello
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 01 — Hello, model : the smallest possible end-to-end pipeline
--- ════════════════════════════════════════════════════════════════════════
---
--- The minimum viable use of ion7-core : load a model, ask it a
--- question, print the reply. Eight steps, ~30 lines of actual code.
---
---   ION7_MODEL=/path/to/model.gguf luajit examples/01_hello.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local MODEL = os.getenv("ION7_MODEL") or
    error("Set ION7_MODEL=/path/to/model.gguf", 0)

local ion7 = require "ion7.core"

-- ── 1. Bring the backend up ───────────────────────────────────────────────
ion7.init({ log_level = 0 })

-- ── 2. Load the model. `Model.fit_params` probes available VRAM and
--      picks the largest n_gpu_layers / n_ctx that will fit ; we fall
--      back to CPU-only when the helper cannot determine a fit.
local fit   = ion7.Model.fit_params(MODEL)
local model = ion7.Model.load(MODEL, {
    n_gpu_layers = fit and fit.n_gpu_layers or 0,
})
print(string.format("[loaded] %s (%.2f B params)",
    model:desc(), model:n_params() / 1e9))

-- ── 3. Vocabulary + inference context ────────────────────────────────────
local vocab = model:vocab()
local ctx   = model:context({ n_ctx = fit and fit.n_ctx or 2048 })

-- ── 4. Sampler chain : top-k / top-p / temperature / dist. The default
--      factory wraps the same defaults — we spell them out here for
--      visibility.
local sampler = ion7.Sampler.chain()
    :top_k(40)
    :top_p(0.95)
    :temperature(0.8)
    :dist(42)
    :build()

-- ── 5. Format the prompt with the model's embedded chat template,
--      then tokenise.
local prompt = vocab:apply_template(
    { { role = "user", content = "What is LuaJIT in one sentence ?" } },
    true,    -- add_ass : append the assistant turn header
    -1)      -- enable_thinking : -1 = use the model default
local toks, n = vocab:tokenize(prompt, false, true)

-- ── 6. Prefill : feed the prompt through the model in one batch.
ctx:decode(toks, n)

-- ── 7. Generate up to 128 tokens, streamed to stdout. Stop on EOG.
io.write("Assistant : ")
for _ = 1, 128 do
    local tok = sampler:sample(ctx:ptr(), -1)
    if vocab:is_eog(tok) then break end
    io.write(vocab:piece(tok))
    io.flush()
    ctx:decode_single(tok)
end
io.write("\n")

-- ── 8. Shutdown. The Lua GC would clean up eventually but explicit
--      teardown keeps VRAM bookkeeping predictable.
sampler:free()
ctx:free()
model:free()
ion7.shutdown()
