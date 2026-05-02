--- @module tests.helpers
--- @author  ion7 / Ion7 Project Contributors
---
--- Shared scaffolding for the ion7-core test suite.
---
--- The whole suite is split into many small files so each one reads as a
--- focused mini-tutorial on a single topic (model loading, sampling,
--- chat templates, ...). Most of those files end up needing the same
--- two things :
---
---   1. A path to a real GGUF model on disk — without one, half the
---      surface (every llama.cpp call that requires weights) cannot be
---      tested. We read the path from the `ION7_MODEL` env var so the
---      same suite works on a developer laptop and in CI.
---
---   2. A tiny pre-amble that loads `ion7.core`, sets `package.path` so
---      the in-tree sources are visible, and (when relevant) loads the
---      model with sane defaults.
---
--- The two helpers below cover both. Importing this module also has the
--- side effect of fixing `package.path` for the whole test process, so
--- a test file can simply do :
---
---   local T = require "tests.framework"
---   local H = require "tests.helpers"
---
---   local model_path = H.require_model(T)
---   ...
---
--- and stop worrying about loader details.
---
--- Environment variables read :
---
---   ION7_MODEL    Required for the model-dependent suites. Path to a
---                 generation GGUF. The suite never falls back to a
---                 hardcoded location — set the variable explicitly,
---                 either inline (`ION7_MODEL=/path bash run_all.sh`)
---                 or via your shell rc.
---   ION7_EMBED    Required for embedding suites. Path to an embedding
---                 model (e.g. nomic-embed-text-v1.5.gguf).
---   ION7_DRAFT    Optional. Path to a small draft model for the
---                 speculative-decoding suite.
---   ION7_LORA     Optional. Path to a LoRA adapter for the LoRA suite.
---   ION7_GPU_LAYERS  Override `n_gpu_layers` (default 0 — pure CPU,
---                    keeps the suite portable on machines with no GPU).

-- ── package.path bootstrap ────────────────────────────────────────────────
--
-- Tests are launched from the repo root (`luajit tests/00_modules.lua`)
-- but the in-tree sources live under `src/ion7/...`. Prepend both the
-- `src/?.lua` and `src/?/init.lua` roots so `require "ion7.core"`
-- resolves the local code instead of any system-installed copy.

local function _bootstrap_paths()
    local extras = "./src/?.lua;./src/?/init.lua"
    if not package.path:find(extras, 1, true) then
        package.path = extras .. ";" .. package.path
    end
end

_bootstrap_paths()

local M = {}

-- ── Environment helpers ───────────────────────────────────────────────────

--- Read an environment variable, returning `nil` for empty strings as
--- well as unset variables. The shell often turns "unset" into "" so we
--- normalise both into `nil` to keep call sites short.
---
--- @param  name string
--- @return string|nil
local function _env(name)
    local v = os.getenv(name)
    if v == nil or v == "" then
        return nil
    end
    return v
end

M._env = _env

--- Resolve the generation model path from `ION7_MODEL`. Returns nil
--- when the variable is unset or empty — there is intentionally NO
--- fallback to a hardcoded location (see `feedback_no_hardcoded_fallbacks`).
---
--- Exposed for test files that want to skip selectively rather than
--- aborting the whole file the way `require_model` does.
---
--- @return string|nil
function M.model_path()
    return _env("ION7_MODEL")
end

--- Require a generation model. When `ION7_MODEL` is unset, mark every
--- test in the current file as `[SKIP]` (using the test framework's
--- `skip` helper) and call `os.exit(0)` so the suite continues without
--- spurious failures.
---
--- The exit-cleanly behaviour is important : `run_all.sh` interprets a
--- non-zero exit as a SUITE failure. Skipping is not failing.
---
--- @param  T tests.framework The framework module already required.
--- @return string Absolute path to the model file.
function M.require_model(T)
    local p = M.model_path()
    if not p then
        T.skip(
            "(this whole file)",
            "ION7_MODEL not set — export ION7_MODEL=/path/to/model.gguf"
        )
        T.summary()
        os.exit(0)
    end
    return p
end

--- Require an embedding model. Same contract as `require_model` but
--- looks at `ION7_EMBED`.
function M.require_embed_model(T)
    local p = _env("ION7_EMBED")
    if not p then
        T.skip("(this whole file)", "ION7_EMBED not set — embedding tests skipped")
        T.summary()
        os.exit(0)
    end
    return p
end

--- Optional helpers — return the path or `nil`. Caller decides whether
--- to skip individual tests.
function M.draft_model_path() return _env("ION7_DRAFT") end
function M.lora_path()        return _env("ION7_LORA")  end

--- Default `n_gpu_layers` for the suite. The suite must run on
--- machines without a GPU, so the default is `0`. CI / workstation
--- runs can override with `ION7_GPU_LAYERS=99`.
--- @return integer
function M.gpu_layers()
    return tonumber(_env("ION7_GPU_LAYERS") or "0") or 0
end

-- ── Filesystem helpers ────────────────────────────────────────────────────

--- Build a path inside the system temp directory. Used by the state /
--- save / load suites that need to write a few KB to disk.
---
---   local p = H.tmpfile("ion7-state-" .. os.time())
---
--- @param  basename string
--- @return string Absolute path under `/tmp` (or `%TEMP%` on Windows
---                via the `TMPDIR` / `TEMP` env vars).
function M.tmpfile(basename)
    local dir = _env("TMPDIR") or _env("TEMP") or "/tmp"
    return dir .. "/" .. basename
end

--- Best-effort `os.remove` that swallows errors — used in test
--- teardowns where the file may already be gone.
--- @param  path string
function M.try_remove(path)
    pcall(os.remove, path)
end

-- ── Backend bring-up ──────────────────────────────────────────────────────

--- Require the native libraries (`libllama`, `libggml`, optionally the
--- bridge `ion7_bridge.so`) to be reachable, then return the loaded
--- `ion7.core` module with the backend already initialised.
---
--- When the shared libraries are not on the loader search path, mark
--- every test in the current file as `[SKIP]` with a message that
--- points to the missing dependency and exit cleanly. This is the same
--- skip-not-fail contract `require_model` follows.
---
--- The `require "ion7.core"` call is wrapped in `pcall` because
--- `init.lua` eagerly imports `ffi/llama/backend.lua`, which itself
--- triggers `ffi.load("libllama")`. A failure there manifests as a Lua
--- error during `require`, not during `ion7.init`.
---
--- @param  T tests.framework
--- @return table The `ion7.core` module with backend initialised.
function M.require_backend(T)
    local ok, ion7 = pcall(require, "ion7.core")
    if not ok then
        T.skip(
            "(this whole file)",
            "ion7.core failed to load — build vendor/llama.cpp and the bridge first " ..
            "(make llama && make build), or set ION7_LIBLLAMA_PATH. " ..
            "Underlying error: " .. tostring(ion7):sub(1, 200)
        )
        T.summary()
        os.exit(0)
    end

    local init_ok, err = pcall(ion7.init, { log_level = 0 })
    if not init_ok then
        T.skip(
            "(this whole file)",
            "ion7.init() failed: " .. tostring(err):sub(1, 200)
        )
        T.summary()
        os.exit(0)
    end

    return ion7
end

-- ── Boilerplate model load ────────────────────────────────────────────────

--- One-shot helper : require the backend, load the model, return the
--- `(ion7, model)` pair. Most model-dependent test files start with
---
---   local ion7, model = H.boot(T)
---
--- which removes ~10 lines of preamble at the top of every file.
---
--- Skips the whole file (with a clear message) when either the backend
--- libraries or the model path are missing. A model that is present
--- but fails to load still raises — that is a real bug, not a missing
--- dependency.
---
--- @param  T    tests.framework
--- @param  opts table? Forwarded to `Model.load` (default
---                     `{ n_gpu_layers = H.gpu_layers() }`).
--- @return table The `ion7.core` module.
--- @return ion7.core.Model
function M.boot(T, opts)
    local ion7 = M.require_backend(T)

    opts = opts or {}
    if opts.n_gpu_layers == nil then
        opts.n_gpu_layers = M.gpu_layers()
    end

    local path  = M.require_model(T)
    local model = ion7.Model.load(path, opts)
    return ion7, model
end

return M
