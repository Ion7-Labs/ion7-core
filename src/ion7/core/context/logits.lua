--- @module ion7.core.context.logits
--- SPDX-License-Identifier: MIT
--- Logits, embeddings, adapters, threadpool, and performance counters.
--- All functions receive the Context instance as first argument.

local ffi     = require "ffi"
local _f32arr = ffi.typeof("float[?]")

local M = {}

-- ── Logits & sampling results ─────────────────────────────────────────────────

--- Return the raw logit array for batch position idx.
--- @param  idx  number  Batch position (usually 0 after decode_single).
--- @return cdata  float* into the context's logit buffer (not owned, do not free).
function M.logits(self, idx)
    return self._lib.llama_get_logits_ith(self._ptr, idx or 0)
end

--- Compute log-probability of token_id at batch position idx.
--- Delegates to C (ion7_logprob) - iterates n_vocab floats ~50x faster than Lua.
--- @param  idx       number  Batch position.
--- @param  token_id  number  Token to evaluate.
--- @return number  Log-probability (negative; closer to 0 = more likely).
function M.logprob(self, idx, token_id)
    return tonumber(self._bridge.ion7_logprob(self._ptr, idx, token_id))
end

--- Compute Shannon entropy of the logit distribution at batch position idx.
--- @param  idx  number  Batch position.
--- @return number  Entropy in nats (>= 0).
function M.entropy(self, idx)
    return tonumber(self._bridge.ion7_entropy(self._ptr, idx or 0))
end

--- Get the sampled token at batch position i (after decode + sampler:sample()).
--- @param  i  number?  Batch position (default 0).
--- @return number  Token ID, or -1 if not available.
function M.sampled_token(self, i)
    return tonumber(self._lib.llama_get_sampled_token_ith(self._ptr, i or 0))
end

--- Return the probability array for the sampled token at position i.
--- @param  i  number?  Batch position.
--- @return cdata  float* (raw, not owned). May be NULL.
function M.sampled_probs(self, i)
    return self._lib.llama_get_sampled_probs_ith(self._ptr, i or 0)
end

--- Return the count of probabilities in sampled_probs(i).
--- @param  i  number?
--- @return number
function M.sampled_probs_count(self, i)
    return tonumber(self._lib.llama_get_sampled_probs_count_ith(self._ptr, i or 0))
end

--- Return the logit array for the sampled candidates at position i.
--- @param  i  number?
--- @return cdata  float* (raw, not owned).
function M.sampled_logits(self, i)
    return self._lib.llama_get_sampled_logits_ith(self._ptr, i or 0)
end

--- Return the count of logits in sampled_logits(i).
--- @param  i  number?
--- @return number
function M.sampled_logits_count(self, i)
    return tonumber(self._lib.llama_get_sampled_logits_count_ith(self._ptr, i or 0))
end

--- Return the candidate token array at position i.
--- @param  i  number?
--- @return cdata  llama_token* (raw, not owned).
function M.sampled_candidates(self, i)
    return self._lib.llama_get_sampled_candidates_ith(self._ptr, i or 0)
end

--- Return the count of candidates in sampled_candidates(i).
--- @param  i  number?
--- @return number
function M.sampled_candidates_count(self, i)
    return tonumber(self._lib.llama_get_sampled_candidates_count_ith(self._ptr, i or 0))
end

-- ── Embeddings ────────────────────────────────────────────────────────────────

--- Return the raw embedding pointer for a sequence (zero-copy).
--- Caller must not free this pointer and must not outlive the context.
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return cdata  float* or nil.
function M.embedding_ptr(self, seq_id)
    local lib = self._lib
    local ptr = lib.llama_get_embeddings_seq(self._ptr, seq_id or 0)
    if ptr == nil then ptr = lib.llama_get_embeddings_ith(self._ptr, -1) end
    return ptr
end

--- Return the pooled embedding as a Lua table (copies floats from C).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @param  dim     number?  Embedding dimension (default: from model ref or 4096).
--- @return table   float[] (1-based), or nil if not available.
function M.embedding(self, seq_id, dim)
    local ptr = self:embedding_ptr(seq_id)
    if ptr == nil then return nil end
    if not dim then
        dim = self._model_ref and tonumber(self._model_ref:n_embd()) or 4096
    end
    local vec = {}
    for i = 0, dim - 1 do
        vec[i + 1] = ptr[i]
    end
    return vec
end

-- ── Adapters ──────────────────────────────────────────────────────────────────

--- Apply a control vector (activation steering) to this context.
--- Takes effect on the next decode.
---
--- @param  data     table|cdata  float[] of n_embd × (il_end - il_start + 1) values.
--- @param  n_embd   number  Embedding dimension.
--- @param  il_start number  First layer (inclusive).
--- @param  il_end   number  Last layer (inclusive).
--- @return bool
function M.set_control_vector(self, data, n_embd, il_start, il_end)
    local n, buf
    if type(data) == "table" then
        n   = #data
        buf = _f32arr(n)
        for i = 1, n do buf[i - 1] = data[i] end
    else
        buf = data   -- cdata float* passed directly - zero copy
        n   = n_embd * (il_end - il_start + 1)
    end
    return self._lib.llama_set_adapter_cvec(self._ptr, buf, n, n_embd, il_start, il_end) == 0
end

--- Clear any active control vector.
function M.clear_control_vector(self)
    self._lib.llama_set_adapter_cvec(self._ptr, nil, 0, 0, 0, 0)
end

--- Assign a sampler to a specific sequence. [EXPERIMENTAL]
--- When set, llama_decode() samples automatically using this sampler for seq_id.
--- @param  seq_id  number
--- @param  sampler Sampler  Must remain alive for the duration.
--- @return bool
function M.set_sampler(self, seq_id, sampler)
    local chain = sampler._chain or sampler._ptr
    assert(chain ~= nil, "[ion7.core.context] set_sampler: invalid sampler")
    return self._lib.llama_set_sampler(self._ptr, seq_id, chain)
end

--- Attach a custom threadpool to this context.
--- @param  tp        cdata|Threadpool
--- @param  tp_batch  cdata|Threadpool?  Defaults to tp.
function M.attach_threadpool(self, tp, tp_batch)
    local p  = type(tp)       == "table" and tp._ptr       or tp
    local pb = tp_batch and (type(tp_batch) == "table" and tp_batch._ptr or tp_batch) or p
    self._bridge.ion7_threadpool_attach(self._ptr, p, pb)
end

--- Detach the current threadpool (reverts to internal auto-managed pool).
function M.detach_threadpool(self)
    self._bridge.ion7_threadpool_detach(self._ptr)
end

-- ── Performance ───────────────────────────────────────────────────────────────

--- Print performance counters to stderr.
function M.perf_print(self)  self._lib.llama_perf_context_print(self._ptr) end

--- Reset performance counters.
function M.perf_reset(self)  self._lib.llama_perf_context_reset(self._ptr) end

--- Return performance data as a Lua table.
--- @return table  { t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval, n_reused, tokens_per_s }
function M.perf(self)
    local data   = self._lib.llama_perf_context(self._ptr)
    local n_eval = tonumber(data.n_eval)
    local t_eval = tonumber(data.t_eval_ms)
    return {
        t_load_ms    = tonumber(data.t_load_ms),
        t_p_eval_ms  = tonumber(data.t_p_eval_ms),
        t_eval_ms    = t_eval,
        n_p_eval     = tonumber(data.n_p_eval),
        n_eval       = n_eval,
        n_reused     = tonumber(data.n_reused),
        tokens_per_s = (n_eval > 0 and t_eval > 0)
                       and (n_eval / (t_eval / 1000.0))
                       or 0.0,
    }
end

return M
