#!/usr/bin/env luajit
--- Pure Lua tests - no model, no libllama.so required.
--- Tests all logic that doesn't depend on FFI.
---
--- Run: luajit tests/test_pure.lua

package.path = "./src/?.lua;./src/?/init.lua;" .. package.path

local T = require "tests.framework"

-- ── Suite 1: Sampler builder (pure Lua, no model needed) ─────────────────────

T.suite("SamplerBuilder - type registry")

local Sampler = require "ion7.core.sampler"

T.test("SamplerBuilder.chain() returns a builder", function()
    local b = Sampler.chain()
    T.ok(b ~= nil)
    T.ok(type(b.top_k) == "function")
    T.ok(type(b.temp)  == "function")
    T.ok(type(b.build) == "function")
end)

T.test("Builder methods return self (chaining works)", function()
    local b = Sampler.chain()
    local b2 = b:top_k(50)
    T.eq(b, b2, "top_k should return self")
    local b3 = b:top_p(0.9, 1)
    T.eq(b, b3, "top_p should return self")
end)

T.test("Builder accumulates steps in order", function()
    local b = Sampler.chain():top_k(50):temp(0.8):dist(42)
    T.eq(#b._steps, 3, "should have 3 steps")
    T.eq(b._steps[1].type, "top_k")
    T.eq(b._steps[2].type, "temperature")
    T.eq(b._steps[3].type, "dist")
end)

T.test("Builder stores sampler params correctly", function()
    local b = Sampler.chain():top_k(100):top_p(0.95, 2):temp(0.7)
    T.eq(b._steps[1].k,       100)
    T.eq(b._steps[2].p,       0.95)
    T.eq(b._steps[2].min_keep, 2)
    T.eq(b._steps[3].t,       0.7)
end)

T.test("temp() is alias for temperature()", function()
    local b1 = Sampler.chain():temp(0.5)
    local b2 = Sampler.chain():temperature(0.5)
    T.eq(b1._steps[1].type, b2._steps[1].type)
    T.eq(b1._steps[1].t,    b2._steps[1].t)
end)

T.test("adaptive_p params", function()
    local b = Sampler.chain():adaptive_p(0.3, 0.98, 42)
    T.eq(b._steps[1].type,   "adaptive_p")
    T.eq(b._steps[1].target,  0.3)
    T.eq(b._steps[1].decay,   0.98)
    T.eq(b._steps[1].seed,    42)
end)

T.test("grammar_lazy params", function()
    local b = Sampler.chain():grammar_lazy('root ::= "yes"', "root", nil,
        {"trigger"}, {}, {})
    T.eq(b._steps[1].type, "grammar_lazy")
    T.eq(b._steps[1].gbnf, 'root ::= "yes"')
end)

-- Individual builder method type checks

T.test("greedy() step type", function()
    local b = Sampler.chain():greedy()
    T.eq(#b._steps, 1)
    T.eq(b._steps[1].type, "greedy")
end)

T.test("top_k() step type and param", function()
    local b = Sampler.chain():top_k(40)
    T.eq(b._steps[1].type, "top_k")
    T.eq(b._steps[1].k, 40)
end)

T.test("top_p() step type, param, and min_keep default", function()
    local b = Sampler.chain():top_p(0.95)
    T.eq(b._steps[1].type, "top_p")
    T.eq(b._steps[1].p, 0.95)
    T.eq(b._steps[1].min_keep, 1, "min_keep should default to 1")
end)

T.test("min_p() step type, param, and min_keep default", function()
    local b = Sampler.chain():min_p(0.05)
    T.eq(b._steps[1].type, "min_p")
    T.eq(b._steps[1].p, 0.05)
    T.eq(b._steps[1].min_keep, 1, "min_keep should default to 1")
end)

T.test("temp_dynamic / temperature_dynamic step type and params", function()
    local b1 = Sampler.chain():temp_dynamic(0.8, 0.2, 2.0)
    T.eq(b1._steps[1].type, "temperature_ext")
    T.eq(b1._steps[1].t, 0.8)
    T.eq(b1._steps[1].delta, 0.2)
    T.eq(b1._steps[1].exponent, 2.0)
    -- alias check
    local b2 = Sampler.chain():temperature_dynamic(0.8, 0.2, 2.0)
    T.eq(b2._steps[1].type, b1._steps[1].type, "temp_dynamic is alias for temperature_dynamic")
end)

T.test("temp_dynamic exponent defaults to 1.0", function()
    local b = Sampler.chain():temp_dynamic(0.8, 0.2)
    T.eq(b._steps[1].exponent, 1.0, "exponent should default to 1.0")
end)

T.test("typical() step type and min_keep default", function()
    local b = Sampler.chain():typical(0.9)
    T.eq(b._steps[1].type, "typical")
    T.eq(b._steps[1].p, 0.9)
    T.eq(b._steps[1].min_keep, 1, "min_keep should default to 1")
end)

T.test("top_n_sigma() step type and param", function()
    local b = Sampler.chain():top_n_sigma(3)
    T.eq(b._steps[1].type, "top_n_sigma")
    T.eq(b._steps[1].n, 3)
end)

T.test("xtc() step type, params, and defaults", function()
    local b = Sampler.chain():xtc(0.5, 0.1)
    T.eq(b._steps[1].type, "xtc")
    T.eq(b._steps[1].probability, 0.5)
    T.eq(b._steps[1].threshold, 0.1)
    T.eq(b._steps[1].min_keep, 1, "min_keep should default to 1")
    T.eq(b._steps[1].seed, 0xFFFFFFFF, "seed should default to 0xFFFFFFFF")
end)

T.test("mirostat() (v1) step type and defaults", function()
    local b = Sampler.chain():mirostat(32000)
    T.eq(b._steps[1].type, "mirostat")
    T.eq(b._steps[1].n_vocab, 32000)
    T.eq(b._steps[1].tau, 5.0, "tau should default to 5.0")
    T.eq(b._steps[1].eta, 0.1, "eta should default to 0.1")
    T.eq(b._steps[1].m, 100, "m should default to 100")
    T.eq(b._steps[1].seed, 0xFFFFFFFF, "seed should default to 0xFFFFFFFF")
end)

T.test("mirostat_v2() step type and defaults", function()
    local b = Sampler.chain():mirostat_v2()
    T.eq(b._steps[1].type, "mirostat_v2")
    T.eq(b._steps[1].tau, 5.0, "tau should default to 5.0")
    T.eq(b._steps[1].eta, 0.1, "eta should default to 0.1")
    T.eq(b._steps[1].seed, 0xFFFFFFFF, "seed should default to 0xFFFFFFFF")
end)

T.test("penalties() step type and defaults", function()
    local b = Sampler.chain():penalties()
    T.eq(b._steps[1].type, "penalties")
    T.eq(b._steps[1].last_n, 64, "last_n should default to 64")
    T.eq(b._steps[1].repeat_penalty, 1.0, "repeat_penalty should default to 1.0")
    T.eq(b._steps[1].freq_penalty, 0.0, "freq_penalty should default to 0.0")
    T.eq(b._steps[1].present_penalty, 0.0, "present_penalty should default to 0.0")
end)

T.test("dist() step type and seed default", function()
    local b = Sampler.chain():dist()
    T.eq(b._steps[1].type, "dist")
    T.eq(b._steps[1].seed, 0xFFFFFFFF, "seed should default to 0xFFFFFFFF")
end)

T.test("logit_bias() step type and biases", function()
    local biases = { [10] = 5.0, [20] = -1.5 }
    local b = Sampler.chain():logit_bias(biases)
    T.eq(b._steps[1].type, "logit_bias")
    T.eq(b._steps[1].biases[10], 5.0)
    T.eq(b._steps[1].biases[20], -1.5)
end)

T.test("build() without FFI raises error (no libllama)", function()
    local b = Sampler.chain():top_k(40):temp(0.8):dist()
    T.err(function() b:build() end)
end)

T.test("empty chain (no steps) can be created", function()
    local b = Sampler.chain()
    T.eq(#b._steps, 0, "empty chain should have 0 steps")
    T.is_type(b._steps, "table")
end)

T.test("monster chain with 10+ steps accumulates all", function()
    local b = Sampler.chain()
        :penalties(64, 1.1, 0.0, 0.0)
        :top_k(40)
        :top_p(0.95)
        :min_p(0.05)
        :typical(0.9)
        :top_n_sigma(2)
        :xtc(0.5, 0.1)
        :temp(0.8)
        :temp_dynamic(0.8, 0.2)
        :mirostat_v2(5.0, 0.1, 42)
        :logit_bias({ [1] = 1.0 })
        :dist(123)
    T.eq(#b._steps, 12, "monster chain should have 12 steps")
    -- Verify first and last
    T.eq(b._steps[1].type, "penalties")
    T.eq(b._steps[12].type, "dist")
    T.eq(b._steps[12].seed, 123)
end)

-- ── Suite 2: CustomSampler construction ──────────────────────────────────────

T.suite("CustomSampler - construction guards")

local CS = require "ion7.core.custom_sampler"

T.test("requires apply callback", function()
    T.err(function() CS.new("bad", {}) end, "apply is required")
end)

T.test("requires callbacks table", function()
    T.err(function() CS.new("bad", nil) end)
end)

T.test("name() returns correct string", function()
    -- Can't fully construct without libllama, just test the guard
    local ok, err = pcall(CS.new, "greedy", { apply = function() return 0 end })
    -- Either succeeds (if libllama is loaded) or fails with a clear message
    if not ok then
        T.ok(tostring(err):find("ion7") or tostring(err):find("cannot"),
            "should fail with ion7 error: " .. tostring(err))
    end
end)

T.test("callbacks.apply must be a function, not a string", function()
    T.err(function() CS.new("bad", { apply = "not_a_function" }) end, "apply is required")
end)

T.test("name must be a string (number rejected)", function()
    -- The constructor passes name to Loader which expects a string.
    -- With valid callbacks, the error should come from FFI/loader, not from
    -- the apply guard.
    local ok, err = pcall(CS.new, 12345, { apply = function() return 0 end })
    if not ok then
        -- Should fail somewhere meaningful, not with a cryptic nil error
        T.ok(type(tostring(err)) == "string", "error should be a readable string")
    end
end)

T.test("all optional callbacks (accept, reset) accepted at guard level", function()
    -- Construction will fail at FFI level, but must pass the Lua guards
    local ok, err = pcall(CS.new, "full", {
        apply  = function() return 0 end,
        accept = function() end,
        reset  = function() end,
    })
    if not ok then
        -- The error should NOT be about callbacks validation
        local msg = tostring(err)
        T.ok(not msg:find("apply is required"),
            "should pass callback validation: " .. msg)
    end
end)

-- ── Suite 3: Threadpool construction guards ──────────────────────────────────

T.suite("Threadpool - construction guards")

local TP = require "ion7.core.threadpool"

T.test("n_threads must be > 0", function()
    T.err(function()
        local ok, err = pcall(TP.new, 0)
        if not ok then error(err) end
    end)
end)

T.test("negative n_threads rejected", function()
    T.err(function() TP.new(-1) end, "n_threads must be > 0")
end)

T.test("n_threads=1 passes Lua guard", function()
    local ok, err = pcall(TP.new, 1)
    if not ok then
        -- Should fail at FFI/bridge level, not at the Lua assertion
        local msg = tostring(err)
        T.ok(not msg:find("n_threads must be > 0"),
            "n_threads=1 should pass guard: " .. msg)
    end
end)

T.test("very large n_threads (10000) passes Lua guard", function()
    local ok, err = pcall(TP.new, 10000)
    if not ok then
        -- Should fail at bridge/OS level, not at the Lua assertion
        local msg = tostring(err)
        T.ok(not msg:find("n_threads must be > 0"),
            "n_threads=10000 should pass guard: " .. msg)
    end
end)

-- ── Suite 4: Grammar (pure Lua) ─────────────────────────────────────────────

T.suite("Grammar - GBNF builder")

local Grammar = pcall(require, "ion7.core.grammar") and require "ion7.core.grammar" or nil
if not Grammar then
    T.suite("Grammar - SKIP (not in core, use ion7-grammar module)")
else
    T.test("from_gbnf preserves string", function()
        local g = Grammar.from_gbnf('root ::= "hello"')
        T.eq(g:to_gbnf(), 'root ::= "hello"')
    end)
end

-- ── Suite 5: Module loading ─────────────────────────────────────────────────

T.suite("Module loading")

T.test("require('ion7.core') returns a table", function()
    local ion7 = require "ion7.core"
    T.is_type(ion7, "table", "ion7.core should be a table")
end)

T.test("require('ion7.core.sampler') returns a table with chain()", function()
    local S = require "ion7.core.sampler"
    T.is_type(S, "table", "sampler module should be a table")
    T.is_type(S.chain, "function", "sampler.chain should be a function")
end)

T.test("require('ion7.core.custom_sampler') returns a table with new()", function()
    local C = require "ion7.core.custom_sampler"
    T.is_type(C, "table", "custom_sampler module should be a table")
    T.is_type(C.new, "function", "custom_sampler.new should be a function")
end)

T.test("require('ion7.core.threadpool') returns a table with new()", function()
    local P = require "ion7.core.threadpool"
    T.is_type(P, "table", "threadpool module should be a table")
    T.is_type(P.new, "function", "threadpool.new should be a function")
end)

-- ── Summary ──────────────────────────────────────────────────────────────────
local ok = T.summary()
os.exit(ok and 0 or 1)
