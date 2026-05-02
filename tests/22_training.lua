#!/usr/bin/env luajit
--- @module tests.22_training
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 22 — Training (`llama_opt`) : opt-in
--- ════════════════════════════════════════════════════════════════════════
---
--- The bridge exposes a thin wrapper over libcommon's training loop :
---
---   ion7_opt_init(ctx, model, optimizer, lr)
---       → ion7_opt_state_t*       (AdamW = 0, SGD = 1)
---
---   ion7_opt_dataset_create(ctx, tokens, n_tokens, stride)
---       → ggml_opt_dataset_t      (libcommon-managed)
---
---   ion7_opt_epoch(ctx, dataset, val_split)
---       → float                   loss after one pass
---
--- A real training step on a 3B model takes minutes, so this file is
--- gated on `ION7_TRAIN=1`. The default run only verifies the API
--- surface : `init` succeeds, `dataset_create` succeeds, both can be
--- freed without crashing.
---
---   ION7_MODEL=/path/to/model.gguf ION7_TRAIN=1 luajit tests/22_training.lua

local T = require "tests.framework"
local H = require "tests.helpers"

-- The training entry points share fate : `ion7_opt_init` issues a
-- `GGML_ASSERT` that aborts the process when the context has not been
-- shaped for backward passes (n_ctx_train alignment, KV cache layout,
-- mmap'd weights, ...). There is no safe "surface only" probe — the
-- assertion fires at init, before the wrapper can return. We therefore
-- gate the entire file on `ION7_TRAIN=1` rather than risk an abort()
-- in the default suite run.

if not H._env("ION7_TRAIN") then
    T.suite("Training (gated)")
    T.skip("(this whole file)",
           "set ION7_TRAIN=1 to exercise the llama_opt path " ..
           "(builds a backward graph — slow and memory-hungry)")
    T.summary()
    os.exit(0)
end

local ion7, model = H.boot(T, { use_mmap = false })
local vocab = model:vocab()

-- ════════════════════════════════════════════════════════════════════════
-- Constraints layered on the training context
-- ════════════════════════════════════════════════════════════════════════
--
-- `llama_opt_init` is single-shot per context : it stores its
-- gradient state on the context as `opt_ctx`, and re-calling it later
-- triggers `GGML_ASSERT(!opt_ctx)`. We therefore build a FRESH
-- context for the entire training cycle and run init / dataset /
-- epoch in one shot.
--
-- The context itself must satisfy :
--   - F16 KV cache (quantised KV has no gradient path).
--   - n_ctx % n_batch == 0 (alignment assertion in opt_init).
--   - n_batch % n_ubatch == 0 (same).
--   - mmap = false (gradient updates would write through a read-only map).
--
-- And the training corpus must satisfy :
--   - common_opt_dataset_init computes
--         ndata = (tokens.size() - n_ctx - 1) / stride
--     and asserts `ndata > 0`. With n_ctx = 64 we therefore need at
--     least 66 tokens — ~3000 tokens give us ~46 training windows.

local TRAIN_N_CTX = 64
local TRAIN_TEXT  = string.rep(
    "The quick brown fox jumps over the lazy dog. ", 100)

local bridge = require "ion7.core.ffi.bridge"

T.suite("Training cycle (init + dataset + epoch)")

T.test("full training cycle on a single fresh context", function()
    local ctx = model:context({
        n_ctx           = TRAIN_N_CTX,
        n_batch         = TRAIN_N_CTX,
        n_ubatch        = TRAIN_N_CTX,
        n_threads       = 2,
        n_threads_batch = 2,
        kv_type         = "f16",
    })

    -- ── init ────────────────────────────────────────────────────────────
    local state = bridge.ion7_opt_init(ctx:ptr(), model:ptr(), 0, 1e-4)
    T.is_type(state, "cdata")

    -- ── dataset ─────────────────────────────────────────────────────────
    local toks, n = vocab:tokenize(TRAIN_TEXT, true, true)
    T.gt(n, TRAIN_N_CTX + 1,
         "training corpus must produce at least n_ctx + 2 tokens")
    local ds = bridge.ion7_opt_dataset_create(ctx:ptr(), toks, n, 1)
    T.is_type(ds, "cdata")

    -- ── one epoch ───────────────────────────────────────────────────────
    local loss = tonumber(bridge.ion7_opt_epoch(ctx:ptr(), ds, 0.0))
    T.is_type(loss, "number")
    T.gte(loss, 0,                      "loss must be >= 0")
    T.gt(1e9, math.abs(loss),           "loss must be finite")

    -- ── teardown ────────────────────────────────────────────────────────
    bridge.ion7_opt_dataset_free(ds)
    bridge.ion7_opt_free(state)
    ctx:free()
end)

model:free()
ion7.shutdown()
os.exit(T.summary() and 0 or 1)
