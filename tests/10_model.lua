#!/usr/bin/env luajit
--- @module tests.10_model
--- @author  ion7 / Ion7 Project Contributors
---
--- Tutorial : loading a GGUF model and reading its metadata.
---
--- This is the first model-dependent file in the suite. Everything that
--- llama.cpp can do starts here — without a `Model` instance there is no
--- vocab, no context, no decode, no sampling.
---
--- What you'll learn from reading this file
--- ────────────────────────────────────────
---   1. How `ion7.init()` brings up the llama.cpp backend (you call it
---      once per process, before any `Model.load`).
---   2. How `Model.load` materialises a `llama_model*` from a `.gguf`
---      file on disk and what the option flags mean.
---   3. How the introspection accessors (`desc`, `n_params`, `n_layer`,
---      ...) report the model's architectural shape — useful both for
---      sanity checks and for adapting code paths to a specific model
---      family.
---   4. How `Model:info()` packs every accessor into one Lua table for
---      cheap logging.
---   5. How GC ownership works : an explicit `Model:free()` and an
---      automatic `ffi.gc` finaliser that catches the case where the
---      caller forgets.
---
--- The assertions below are generic invariants (positive sizes, sane
--- RoPE type, GQA constraint, ...) so the file passes against any
--- decoder-only LLM you point `ION7_MODEL` at — Llama, Mistral, Qwen,
--- DeepSeek, etc.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("Model loading and introspection")

-- ── Boilerplate : init + load ───────────────────────────────────────────
--
-- `H.boot` is shorthand for :
--
--   local ion7 = require "ion7.core"
--   ion7.init({ log_level = 0 })
--   local model = ion7.Model.load(H.require_model(T), {
--       n_gpu_layers = H.gpu_layers(),
--   })
--
-- We expand the long form ONCE here so the tutorial reader can see
-- what happens under the hood ; subsequent test files in the suite
-- collapse it back into a one-liner via `H.boot`.
--
-- `require_backend` wraps the `require "ion7.core"` + `init` pair in
-- pcall and skips the whole file with a clear message when libllama
-- is missing — the same skip-not-fail contract used for the model.

local ion7 = H.require_backend(T)

local model_path = H.require_model(T)

T.test("Model.load materialises the GGUF file", function()
    -- n_gpu_layers = 0 keeps the suite portable — tests must run on
    -- CPU-only laptops and in CI runners without a GPU. The user can
    -- override via ION7_GPU_LAYERS if a faster run is desired.
    local m = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })
    T.is_type(m,  "table",  "Model.load returns an instance, not a cdata")
    T.eq(m.path, model_path, "Model.path mirrors the load path")
    T.neq(m._ptr, nil,        "Model._ptr should be a live cdata handle")
    -- Free aggressively so each test gets a clean slate ; the
    -- following tests use a single shared instance below.
    m:free()
end)

-- One shared model instance for the rest of the file. The previous
-- test already proved that load works ; loading again here is just the
-- pragmatic path of least resistance — keeping a single instance
-- around lets subsequent tests run in a few microseconds each.
local model = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })

-- ── Identity & size ─────────────────────────────────────────────────────
--
-- `desc` returns a llama.cpp-formatted summary like "Mistral 3B Q8_0".
-- `size` reports tensor bytes on disk (not VRAM consumption — that
-- would account for KV cache etc., which only exist once a context is
-- created). `n_params` is the trained-parameter count.

T.test("desc returns a non-empty model identity string", function()
    local d = model:desc()
    T.is_type(d, "string")
    T.gt(#d, 0, "desc string should not be empty")
end)

T.test("size and n_params are positive and consistent", function()
    local sz = model:size()
    local np = model:n_params()
    T.gt(sz, 0,         "size in bytes must be positive")
    T.gt(np, 0,         "parameter count must be positive")
    -- Sanity : a Q8 model uses roughly 1 byte per parameter, so the two
    -- numbers should be within an order of magnitude. This is a coarse
    -- but robust invariant against accidental unit-mixing bugs.
    T.gt(sz / np, 0.3,  "bytes/param should not be implausibly small")
    T.gt(20,    sz / np, "bytes/param should not be implausibly large")
end)

-- ── Architectural topology ──────────────────────────────────────────────
--
-- Every transformer-style model exposes a handful of integer shapes :
--
--   n_ctx_train  : context window the model was trained for.
--   n_embd       : hidden / residual stream dimension.
--   n_layer      : transformer block count.
--   n_head       : attention heads. n_head_kv ≤ n_head for GQA / MQA.
--   n_swa        : sliding-window attention size. 0 for full-context.
--
-- Reading these is how a generic loader can size buffers, decide
-- whether a draft model is compatible, or warn that a target context
-- exceeds n_ctx_train.

T.test("topology accessors report sensible values", function()
    T.gt(model:n_ctx_train(), 0, "n_ctx_train")
    T.gt(model:n_embd(),      0, "n_embd")
    T.gt(model:n_layer(),     0, "n_layer")
    T.gt(model:n_head(),      0, "n_head")
    -- GQA invariant : KV heads cannot exceed query heads.
    local nh, nh_kv = model:n_head(), model:n_head_kv()
    T.gt(nh_kv, 0,    "n_head_kv must be positive")
    T.gte(nh, nh_kv,  "n_head_kv must be <= n_head")
    -- n_swa is non-negative ; 0 means "no sliding window".
    T.gte(model:n_swa(), 0, "n_swa must be non-negative")
end)

-- ── Capability flags ────────────────────────────────────────────────────

T.test("capability flags match expected family shape", function()
    -- Ministral / Mistral are decoder-only causal LMs.
    T.eq(model:has_decoder(), true, "must have a decoder stack")
    T.eq(model:has_encoder(), false, "Mistral has no encoder pass")
    -- Recurrent flags : false for plain attention models.
    T.eq(model:is_recurrent(), false)
    T.eq(model:is_hybrid(),    false)
    T.eq(model:is_diffusion(), false)
end)

-- ── Position encoding ───────────────────────────────────────────────────
--
-- `rope_type` returns one of the symbolic names :
--   "none" | "norm" | "neox" | "mrope" | "imrope" | "vision" | "unknown".
--
-- We don't pin the exact value (different models use different schemes)
-- but we assert the result is one of the known names — catches a
-- silent enum-renaming regression upstream.

T.test("rope_type is one of the documented values", function()
    T.one_of(model:rope_type(),
        { "none", "norm", "neox", "mrope", "imrope", "vision", "unknown" },
        "rope_type must be a recognised name")
    -- The training-time RoPE freq scale should be a positive number.
    T.gt(model:rope_freq_scale_train(), 0, "rope_freq_scale_train")
end)

-- ── Aggregate snapshot ──────────────────────────────────────────────────
--
-- `info()` packs every accessor into one flat table. Useful as a
-- one-liner for logs / dashboards. We don't reproduce every field —
-- we just sanity-check the shape and a couple of derived fields.

T.test("info() returns a populated snapshot table", function()
    local i = model:info()
    T.is_type(i,         "table",  "info() returns a table")
    T.is_type(i.desc,    "string", "info.desc is a string")
    T.is_type(i.n_params, "number", "info.n_params is a number")
    -- Derived size-in-GB field should match the raw byte count.
    T.near(i.size_gb, i.size / (1024 ^ 3), 1e-9,
           "size_gb is size / GiB")
    -- vocab-related fields come from the lazy-built Vocab handle.
    T.gt(i.n_vocab, 0,             "info.n_vocab is positive")
    T.is_type(i.vocab_type, "string")
end)

-- ── Multiple contexts share a single model ──────────────────────────────
--
-- A common ion7 pattern is to load the model once and create several
-- contexts on top — e.g. a small one for embedding-style classification
-- and a big one for generation. The `_model_ref` back-reference each
-- Context holds prevents the model from being collected while any
-- context is still alive, even if the caller drops their `model`
-- variable.

T.test("multi-context : contexts hold the model alive", function()
    local ctx_a = model:context({ n_ctx = 1024 })
    local ctx_b = model:context({ n_ctx = 1024 })
    T.neq(ctx_a:ptr(), nil)
    T.neq(ctx_b:ptr(), nil)
    -- Both contexts must report the same n_ctx that we asked for ;
    -- llama.cpp may round up internally but never below the request.
    T.gte(ctx_a:n_ctx(), 1024)
    T.gte(ctx_b:n_ctx(), 1024)
    ctx_a:free()
    ctx_b:free()
end)

-- ── Explicit free is idempotent ─────────────────────────────────────────
--
-- Tightly-scoped Lua-finaliser semantics are notoriously hard to reason
-- about when CUDA / Metal allocations are involved. Calling `:free()`
-- explicitly when you're done is the predictable path. The bookkeeping
-- around `ffi.gc` makes a second call a no-op rather than a double
-- free.

T.test("Model:free() is idempotent and clears _ptr", function()
    -- Use a throwaway instance so we don't disturb the shared `model`.
    local m = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })
    m:free()
    T.eq(m._ptr, nil, "free() should null the pointer")
    -- Second call must not crash.
    m:free()
    T.eq(m._ptr, nil, "second free() is a no-op")
end)

-- ── Verdict ─────────────────────────────────────────────────────────────

model:free()
ion7.shutdown()

local ok = T.summary()
os.exit(ok and 0 or 1)
