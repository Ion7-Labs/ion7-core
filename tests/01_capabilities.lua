#!/usr/bin/env luajit
--- @module tests.01_capabilities
--- @author  ion7 / Ion7 Project Contributors
---
--- ════════════════════════════════════════════════════════════════════════
--- 01 — Backend lifecycle and runtime capabilities
--- ════════════════════════════════════════════════════════════════════════
---
--- The first model-free integration test. This file exercises the parts
--- of the public surface that reach native code but do NOT require a
--- model on disk :
---
---   - `ion7.init()`              → `llama_backend_init`
---   - `ion7.shutdown()`          → `llama_backend_free` (idempotent)
---   - `ion7.capabilities()`      → mmap / mlock / GPU / RPC / device counts
---   - `ion7.time_us()`           → llama.cpp's monotonic microsecond clock
---   - `ion7.numa_init()`         → NUMA strategy selector
---   - `ion7.NUMA_*` constants    → enum mirror of `ggml_numa_strategy`
---   - `ion7.log` (`util.log`)    → log-level routing through `llama_log_set`
---
--- Skipped automatically when libllama / the bridge cannot be loaded.
--- See `tests/helpers.lua` (`require_backend`).

local T = require "tests.framework"
local H = require "tests.helpers"

-- ════════════════════════════════════════════════════════════════════════
-- Suite 1 — Backend lifecycle
-- ════════════════════════════════════════════════════════════════════════
--
-- `ion7.init` must run successfully before any other model-related call
-- can succeed. We rely on `H.require_backend` to do the call AND skip
-- the whole file with a clear message when libllama is not on the
-- loader path. Anything that follows can therefore assume the backend
-- is up.

local ion7 = H.require_backend(T)

T.suite("Backend lifecycle")

T.test("ion7.init returned without raising", function()
    -- `require_backend` already called `init`. We are merely confirming
    -- the resulting `ion7` namespace looks healthy.
    T.is_type(ion7,             "table")
    T.is_type(ion7.capabilities, "function")
end)

T.test("ion7.shutdown is idempotent", function()
    -- llama.cpp tolerates being torn down twice — the second call is a
    -- no-op. We need this property so a Ctrl-C path that triggers
    -- shutdown twice (signal handler + atexit) does not crash.
    T.no_error(function() ion7.shutdown() end, "first shutdown")
    T.no_error(function() ion7.shutdown() end, "second shutdown")
    -- Bring the backend back up so the rest of the suite can run.
    ion7.init({ log_level = 0 })
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 2 — Capability discovery
-- ════════════════════════════════════════════════════════════════════════
--
-- `capabilities()` is the canonical "what can this build do" query.
-- Downstream modules consult it to decide whether to enable mmap,
-- whether GPU offload is even an option, etc. We assert types only
-- (the values are environment-dependent) — the goal is to confirm the
-- field plumbing and the cdata → number / cdata → string conversions.

T.suite("Runtime capabilities")

local caps = ion7.capabilities()

T.test("capabilities() returns a flat table of expected fields", function()
    T.is_type(caps,                  "table")
    T.is_type(caps.mmap,             "boolean")
    T.is_type(caps.mlock,            "boolean")
    T.is_type(caps.gpu_offload,      "boolean")
    T.is_type(caps.rpc,              "boolean")
    T.is_type(caps.max_devices,      "number")
    T.is_type(caps.max_parallel_seqs,"number")
    T.is_type(caps.llama_info,       "string")
    -- `bridge_ver` is `nil` when the bridge .so is absent — that is a
    -- documented contract of `capabilities()`, not a bug. Accept either.
    T.ok(caps.bridge_ver == nil or type(caps.bridge_ver) == "string",
         "bridge_ver must be string or nil")
end)

T.test("max_devices and max_parallel_seqs are sane", function()
    -- llama.cpp returns at least 1 here even on a CPU-only build.
    T.gte(caps.max_devices,       1, "at least one device")
    T.gte(caps.max_parallel_seqs, 1, "at least one parallel sequence")
end)

T.test("llama_info contains the CPU feature string", function()
    -- The system-info string mentions the build's enabled extensions
    -- ("AVX = 1 | AVX2 = 1 | ..."). It is too implementation-specific
    -- to test exact content, so we assert it is non-empty and the
    -- canonical "AVX" substring is somewhere in it on x86_64.
    T.gt(#caps.llama_info, 0)
    if jit and jit.arch == "x86" or jit.arch == "x64" then
        T.contains(caps.llama_info, "AVX")
    end
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 3 — Monotonic clock
-- ════════════════════════════════════════════════════════════════════════
--
-- `ion7.time_us` returns llama.cpp's `llama_time_us()` — a monotonic
-- microsecond counter useful for token-time measurement inside hot
-- generation loops. We confirm it produces a positive number and that
-- it advances between two reads separated by a no-op sleep.

T.suite("Monotonic clock — ion7.time_us")

T.test("returns a positive integer", function()
    local t = ion7.time_us()
    T.is_type(t, "number")
    T.gt(t, 0)
end)

T.test("advances between two reads", function()
    local a = ion7.time_us()
    -- A tight loop is enough to push the counter forward — sleeping
    -- would slow the test down with no benefit.
    for _ = 1, 1000 do end
    local b = ion7.time_us()
    T.gte(b, a, "clock must be monotonic non-decreasing")
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 4 — NUMA
-- ════════════════════════════════════════════════════════════════════════
--
-- The NUMA helpers mirror `ggml_numa_strategy` from upstream. We do
-- not actually flip the strategy here (that would only matter on a
-- multi-socket host and could perturb the rest of the suite) — we
-- check the constant values match the upstream enum and that
-- `numa_init` accepts each one without raising.

T.suite("NUMA — strategy constants and selector")

T.test("NUMA_* mirror ggml_numa_strategy values", function()
    T.eq(ion7.NUMA_DISABLED,   0)
    T.eq(ion7.NUMA_DISTRIBUTE, 1)
    T.eq(ion7.NUMA_ISOLATE,    2)
    T.eq(ion7.NUMA_NUMACTL,    3)
    T.eq(ion7.NUMA_MIRROR,     4)
end)

T.test("numa_init accepts the documented strategies", function()
    -- We only test DISABLED to avoid altering the live process state on
    -- a machine that may or may not be NUMA. The point is to confirm
    -- the FFI plumbing returns cleanly — not to actually toggle NUMA.
    T.no_error(function() ion7.numa_init(ion7.NUMA_DISABLED) end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Suite 5 — Log routing (`ion7.log`)
-- ════════════════════════════════════════════════════════════════════════
--
-- `ion7.log` registers a Lua-side dispatcher with llama.cpp via
-- `llama_log_set`. We verify the basic public API exists and that
-- toggling levels does not crash. Capturing actual log output is
-- intentionally out of scope here — it would couple the test to the
-- exact wording of llama.cpp's diagnostic strings.

T.suite("Log routing — ion7.log")

T.test("public API surface", function()
    T.is_type(ion7.log,                 "table")
    T.is_type(ion7.log.set_level,       "function")
    T.is_type(ion7.log.set_timestamps,  "function")
    T.is_type(ion7.log.install,         "function")
    T.is_type(ion7.log.uninstall,       "function")
end)

T.test("set_level accepts the four documented levels", function()
    -- 0 = silent, 1 = error, 2 = warn, 3 = info, 4 = debug.
    -- Re-set to 0 at the end so we don't spam stderr for the rest.
    for level = 0, 4 do
        T.no_error(function() ion7.log.set_level(level) end,
                   "set_level " .. level)
    end
    ion7.log.set_level(0)
end)

T.test("set_timestamps takes a boolean", function()
    T.no_error(function() ion7.log.set_timestamps(true)  end)
    T.no_error(function() ion7.log.set_timestamps(false) end)
end)

-- ════════════════════════════════════════════════════════════════════════
-- Verdict
-- ════════════════════════════════════════════════════════════════════════

ion7.shutdown()
os.exit(T.summary() and 0 or 1)
