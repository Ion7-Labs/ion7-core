--- @module ion7.core.context.fast
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Context` : composite fast-path methods that fuse multiple
--- 1:1 wrappers into a single bridge call. This is the upper tier of
--- ion7-core's two-level API :
---
---   - `ion7.core.context.decode` / `sampler.lua` / `vocab.lua`
---     One method per llama.cpp function. Build whatever pipeline you
---     want from the primitives.
---
---   - `ion7.core.context.fast` (this file)
---     Composite shortcuts for common pipelines. Lower line count in
---     user code ; same performance envelope as the primitives because
---     LuaJIT's JIT already collapses sequences of FFI calls into a
---     single trace. The fast-path is an ergonomics layer, not a
---     perf-magic layer.
---
--- Pattern : every fast-path method here MUST be expressible as a
--- straightforward composition of the primitive API. If a user wants
--- to inspect intermediate state (e.g. log every sampled token before
--- decoding), they drop back to the primitive API ; the fast-path is
--- not in their way.

local bridge      = require "ion7.core.ffi.bridge"
local ion7_step   = bridge.ion7_context_step

local M = {}

-- ── step (sample + EOG check + decode in one bridge call) ─────────────────

--- Single-token generation step. Equivalent to
--- `sampler:sample(ctx:ptr(), idx)` + `vocab:is_eog(tok)` (early exit
--- on EOG) + `ctx:decode_single(tok, seq_id)`, fused into one bridge
--- call.
---
--- Pre-conditions : `prepare_step(seq_id)` was called once after the
--- prefill / kv_clear so the per-context decode batch is configured
--- for n_tokens=1.
---
--- @param  sampler ion7.core.Sampler  Sampler chain.
--- @param  vocab   ion7.core.Vocab    Vocabulary (for EOG check).
--- @param  idx     integer?           Logits index (default -1 = last).
--- @return integer? Sampled token ID, or `nil` on EOG.
--- @raise            On `llama_decode` error.
function M.step(self, sampler, vocab, idx)
    local b = self._decode_batch
    b.pos[0] = self._n_past
    local tok = ion7_step(self._ptr, b, sampler._chain, vocab._ptr, idx or -1)
    if tok == -2 then
        error("[ion7.core.context] step: llama_decode failed", 2)
    end
    if tok < 0 then return nil end
    self._n_past = self._n_past + 1
    return tok
end

--- Configure the per-context decode batch for repeated `step()` calls.
--- Call once after `kv_clear` or after the prefill — sets n_tokens=1,
--- seq_id[0][0]=<seq>, logits[0]=1. The hot loop then only touches
--- pos+token via the bridge.
---
--- @param  seq_id integer? Sequence ID (default 0).
function M.prepare_step(self, seq_id)
    local b = self._decode_batch
    b.n_tokens     = 1
    b.seq_id[0][0] = seq_id or 0
    b.logits[0]    = 1
end

return M
