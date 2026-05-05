--- @module ion7.core.context.logits
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Context` : everything that reads from a freshly-decoded
--- context — logits, embeddings, the sampled-token/probs accessors, plus
--- runtime adapters (control vector, threadpool) and perf counters.
---
--- Two former standalone modules are folded in here :
---
---   - `warmup.lua`  → `Context:warmup()`
---   - `stats.lua`   → `Context:logprob` / `entropy` / `logprob_entropy`
---
--- The math (`logprob`/`entropy`) is implemented in pure Lua + LuaJIT
--- FFI cdata access. The historical bridge implemented these in C
--- because plain Lua's per-cdata `tonumber()` made the n_vocab loop
--- ~50× slower than C. With direct cdata access the JIT compiles the
--- loop down to roughly ~2-3× of C speed — close enough that keeping
--- this off the bridge is a clean win.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_logits = require "ion7.core.ffi.llama.logits" -- logits / embeddings
local llama_misc = require "ion7.core.ffi.llama.misc" -- sampled_*, threadpool attach
local llama_context = require "ion7.core.ffi.llama.context" -- set_adapter_cvec, set_sampler
local llama_batch = require "ion7.core.ffi.llama.batch" -- batch_get_one (used by warmup)
local llama_vocab = require "ion7.core.ffi.llama.vocab" -- vocab_bos, vocab_n_tokens
local llama_model = require "ion7.core.ffi.llama.model" -- model_get_vocab
local llama_perf = require "ion7.core.ffi.llama.perf" -- perf_context_*

local ffi_new = ffi.new
local math_exp = math.exp
local math_log = math.log
local math_huge = math.huge
local tonumber = tonumber

local F32ARR = ffi.typeof("float[?]")

-- Sentinel returned by `logprob` / `logprob_entropy` when the call
-- cannot produce a value (no logits at idx, or token out of vocab).
local NEG_INF = -math_huge

local M = {}

-- ── Logits & sampling-result accessors ────────────────────────────────────

--- Raw logits buffer for batch position `idx`. Returns a `float*` cdata
--- that is OWNED by the context — do not free, do not retain past the
--- next decode.
--- @param  idx integer? Default 0 (typical after `decode_single`).
--- @return cdata
function M.logits(self, idx)
    return llama_logits.llama_get_logits_ith(self._ptr, idx or 0)
end

--- Sampled token at batch position `i`, or `-1` if not available.
--- (Set by an attached `llama_set_sampler` ; see `Context:set_sampler`.)
--- @param  i integer?
--- @return integer
function M.sampled_token(self, i)
    return tonumber(llama_misc.llama_get_sampled_token_ith(self._ptr, i or 0))
end

--- Probability array for the sampled token at position `i`.
--- @param  i integer?
--- @return cdata float* (raw, not owned).
function M.sampled_probs(self, i)
    return llama_misc.llama_get_sampled_probs_ith(self._ptr, i or 0)
end

--- Count matching `sampled_probs(i)`.
--- @param  i integer?
--- @return integer
function M.sampled_probs_count(self, i)
    return tonumber(llama_misc.llama_get_sampled_probs_count_ith(self._ptr, i or 0))
end

--- Logit array for the sampled candidates at position `i`.
--- @param  i integer?
--- @return cdata float* (raw, not owned).
function M.sampled_logits(self, i)
    return llama_misc.llama_get_sampled_logits_ith(self._ptr, i or 0)
end

--- Count matching `sampled_logits(i)`.
--- @param  i integer?
--- @return integer
function M.sampled_logits_count(self, i)
    return tonumber(llama_misc.llama_get_sampled_logits_count_ith(self._ptr, i or 0))
end

--- Candidate-token array at position `i`.
--- @param  i integer?
--- @return cdata llama_token* (raw, not owned).
function M.sampled_candidates(self, i)
    return llama_misc.llama_get_sampled_candidates_ith(self._ptr, i or 0)
end

--- Count matching `sampled_candidates(i)`.
--- @param  i integer?
--- @return integer
function M.sampled_candidates_count(self, i)
    return tonumber(llama_misc.llama_get_sampled_candidates_count_ith(self._ptr, i or 0))
end

-- ── logprob / entropy (pure-Lua softmax with cdata fast path) ─────────────

--- Internal : numerically-stable log-softmax preparation. Returns
--- `(logits, n, max_l, sum)` for reuse by the public methods. When the
--- batch has no logits at `idx`, `logits` is nil.
local function prepare_softmax(self, idx)
    local lp = llama_logits.llama_get_logits_ith(self._ptr, idx)
    if lp == nil then
        return nil, 0, NEG_INF, 0.0
    end

    -- We get vocab size from the parent Model when available (it caches
    -- the Vocab handle), otherwise fall back to a live FFI query.
    local n
    if self._model_ref then
        n = self._model_ref:vocab():n_vocab()
    else
        local vocab = llama_model.llama_model_get_vocab(llama_misc.llama_get_model(self._ptr))
        n = tonumber(llama_vocab.llama_vocab_n_tokens(vocab))
    end

    -- Pass 1 : find the max for numerical stability.
    local max_l = NEG_INF
    for i = 0, n - 1 do
        local v = lp[i]
        if v > max_l then
            max_l = v
        end
    end
    -- Pass 2 : sum exp(l - max). LuaJIT specialises the cdata loads.
    local sum = 0.0
    for i = 0, n - 1 do
        sum = sum + math_exp(lp[i] - max_l)
    end
    return lp, n, max_l, sum
end

--- Log-probability of `token_id` given the logits at batch position
--- `idx`. Returns `-math.huge` when the token is out of vocab range or
--- the batch slot has no logits.
--- @param  idx      integer
--- @param  token_id integer
--- @return number
function M.logprob(self, idx, token_id)
    local lp, n, max_l, sum = prepare_softmax(self, idx)
    if lp == nil then
        return NEG_INF
    end
    if token_id < 0 or token_id >= n then
        return NEG_INF
    end
    return lp[token_id] - max_l - math_log(sum)
end

--- Shannon entropy (in nats) of the logit distribution at batch
--- position `idx`. Returns `0` when no logits are available.
--- @param  idx integer
--- @return number
function M.entropy(self, idx)
    local lp, n, max_l, sum = prepare_softmax(self, idx)
    if lp == nil then
        return 0.0
    end
    local H = 0.0
    for i = 0, n - 1 do
        local p = math_exp(lp[i] - max_l) / sum
        if p > 0.0 then
            H = H - p * math_log(p)
        end
    end
    return H
end

--- Combined `logprob` and `entropy` in ONE pass over `n_vocab`. Use
--- when both metrics are needed — halves the work compared to two
--- separate calls.
---
--- @param  idx      integer
--- @param  token_id integer
--- @return number logprob (`-math.huge` on error).
--- @return number entropy (`0` on error).
function M.logprob_entropy(self, idx, token_id)
    local lp, n, max_l, sum = prepare_softmax(self, idx)
    if lp == nil then
        return NEG_INF, 0.0
    end

    local lpr
    if token_id >= 0 and token_id < n then
        lpr = lp[token_id] - max_l - math_log(sum)
    else
        lpr = NEG_INF
    end

    local H = 0.0
    for i = 0, n - 1 do
        local p = math_exp(lp[i] - max_l) / sum
        if p > 0.0 then
            H = H - p * math_log(p)
        end
    end
    return lpr, H
end

-- ── Embeddings ────────────────────────────────────────────────────────────

--- Raw embedding cdata pointer for a sequence (zero-copy). Falls back
--- to the per-batch embedding when the sequence-keyed accessor returns
--- nil. Caller must NOT free and must NOT retain past the next decode.
---
--- @param  seq_id integer? Default 0.
--- @return cdata|nil  `float*` or nil if no embedding is available.
function M.embedding_ptr(self, seq_id)
    local p = llama_logits.llama_get_embeddings_seq(self._ptr, seq_id or 0)
    if p == nil then
        p = llama_logits.llama_get_embeddings_ith(self._ptr, -1)
    end
    return p
end

--- Pooled embedding as a Lua table (1-based, copies the floats out so
--- the caller can hold it past the next decode).
---
--- @param  seq_id integer? Default 0.
--- @param  dim    integer? Embedding dimension. Defaults to the model's
---                         `n_embd` if the Context has a back-reference
---                         to its parent Model, else `4096`.
--- @return table|nil
function M.embedding(self, seq_id, dim)
    local p = self:embedding_ptr(seq_id)
    if p == nil then
        return nil
    end

    if not dim then
        dim = (self._model_ref and tonumber(self._model_ref:n_embd())) or 4096
    end

    local out = {}
    for i = 0, dim - 1 do
        out[i + 1] = p[i]
    end
    return out
end

-- ── Per-token embeddings (pooling = "none" only) ──────────────────────────
--
-- When the context was built with `pooling = "none"`, llama.cpp returns
-- one embedding vector per decoded token (rather than a single pooled
-- vector). The functions below expose both the raw row-major float*
-- buffer (zero-copy fast path) and a per-token Lua-table accessor.
--
-- Use cases : late chunking (Jina, arXiv:2409.04701), token-level
-- attribution, contrastive sentence-pair scoring without re-decoding.

--- Raw float* pointer to ALL token embeddings of the most recent
--- decode. Layout : `[n_tokens][n_embd]` row-major. Owned by the
--- context — do NOT free, do NOT retain past the next decode. Returns
--- nil when the context was not built with `pooling = "none"` or when
--- nothing has been decoded yet.
function M.embeddings_all_ptr(self)
    return llama_logits.llama_get_embeddings(self._ptr)
end

--- Raw float* pointer to the embedding of the i-th decoded token.
--- 0-based. Same ownership rules as `embeddings_all_ptr`.
function M.embedding_token_ptr(self, i)
    return llama_logits.llama_get_embeddings_ith(self._ptr, i or 0)
end

--- Per-token embedding as a copied Lua table. Convenience wrapper over
--- `embedding_token_ptr` for callers that want to retain the vector
--- across decodes. Returns nil when no embedding is available at idx.
---
--- @param  idx integer  0-based token index in the most recent batch.
--- @param  dim integer? Embedding dimension. Defaults to the parent
---                       Model's `n_embd` when reachable.
--- @return table|nil
function M.embedding_token(self, idx, dim)
    local p = llama_logits.llama_get_embeddings_ith(self._ptr, idx)
    if p == nil then return nil end

    if not dim then
        dim = (self._model_ref and tonumber(self._model_ref:n_embd())) or 4096
    end

    local out = {}
    for i = 0, dim - 1 do
        out[i + 1] = p[i]
    end
    return out
end

-- ── Adapters (control vector, sampler-per-sequence) ───────────────────────

--- Apply a control vector (activation steering) to this context.
--- Effective on the next decode. Pass either a Lua table (we'll copy
--- to a float buffer) or a pre-built cdata `float*`.
---
--- @param  data     table|cdata  `n_embd × (il_end - il_start + 1)` floats.
--- @param  n_embd   integer Embedding dimension.
--- @param  il_start integer First layer (inclusive).
--- @param  il_end   integer Last layer  (inclusive).
--- @return boolean
function M.set_control_vector(self, data, n_embd, il_start, il_end)
    local n, buf
    if type(data) == "table" then
        n = #data
        buf = F32ARR(n)
        for i = 1, n do
            buf[i - 1] = data[i]
        end
    else
        -- Caller passed a cdata float* directly — zero-copy fast path.
        buf = data
        n = n_embd * (il_end - il_start + 1)
    end
    return llama_context.llama_set_adapter_cvec(self._ptr, buf, n, n_embd, il_start, il_end) == 0
end

--- Remove any previously-applied control vector.
function M.clear_control_vector(self)
    llama_context.llama_set_adapter_cvec(self._ptr, nil, 0, 0, 0, 0)
end

--- Attach a sampler to a specific sequence. After this, `llama_decode`
--- will sample automatically using `sampler` whenever it processes
--- `seq_id`. EXPERIMENTAL upstream — the surface may shift.
---
--- @param  seq_id  integer
--- @param  sampler ion7.core.Sampler|cdata Either a Sampler instance
---                                         or a raw `llama_sampler*`.
--- @return boolean
function M.set_sampler(self, seq_id, sampler)
    local chain = sampler._chain or sampler._ptr or sampler
    assert(chain ~= nil, "[ion7.core.context] set_sampler: invalid sampler")
    return llama_context.llama_set_sampler(self._ptr, seq_id, chain) == true
end

-- ── Threadpool attach / detach ────────────────────────────────────────────

--- Attach a CPU threadpool to this context. `tp` may be either a raw
--- `ggml_threadpool_t` cdata or a `Threadpool` instance.
---
--- The second pool slot is for batch operations and defaults to NULL,
--- which tells llama.cpp to fall back to the primary pool internally
--- — passing the same pool twice via two distinct attach calls would
--- corrupt the pool's wait state.
---
--- @param  tp       cdata|ion7.core.Threadpool Primary pool.
--- @param  tp_batch cdata|ion7.core.Threadpool? Optional dedicated batch pool.
function M.attach_threadpool(self, tp, tp_batch)
    local p  = type(tp) == "table" and tp._ptr or tp
    local pb = tp_batch and
               (type(tp_batch) == "table" and tp_batch._ptr or tp_batch) or
               nil
    llama_misc.llama_attach_threadpool(self._ptr, p, pb)
end

--- Detach the current threadpool (reverts to llama.cpp's internal one).
function M.detach_threadpool(self)
    llama_misc.llama_detach_threadpool(self._ptr)
end

-- ── Performance counters ──────────────────────────────────────────────────

--- Print perf counters to stderr.
function M.perf_print(self)
    llama_perf.llama_perf_context_print(self._ptr)
end

--- Reset every perf counter to zero.
function M.perf_reset(self)
    llama_perf.llama_perf_context_reset(self._ptr)
end

--- Read the perf counters into a Lua table.
--- @return table { t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval, n_reused, tokens_per_s }
function M.perf(self)
    local d = llama_perf.llama_perf_context(self._ptr)
    local n_eval = tonumber(d.n_eval)
    local t_eval = tonumber(d.t_eval_ms)
    return {
        t_load_ms = tonumber(d.t_load_ms),
        t_p_eval_ms = tonumber(d.t_p_eval_ms),
        t_eval_ms = t_eval,
        n_p_eval = tonumber(d.n_p_eval),
        n_eval = n_eval,
        n_reused = tonumber(d.n_reused),
        tokens_per_s = (n_eval > 0 and t_eval > 0) and (n_eval / (t_eval / 1000.0)) or 0.0
    }
end

-- ── Warmup (single dummy decode + KV clear) ───────────────────────────────

--- Run a single dummy decode through this context to force the GPU
--- backend to JIT-compile its shaders. After the call the KV cache is
--- wiped and `n_past` reset, so the context is left in the same state
--- it would have been right after creation — minus the cold-shader
--- penalty on the next real decode.
---
--- Use once, right after `model:context()` and before any real work, to
--- shave 1-3 s off the time-to-first-token of the first real request.
function M.warmup(self)
    if self._ptr == nil then
        return
    end

    local model = llama_misc.llama_get_model(self._ptr)
    local vocab = llama_model.llama_model_get_vocab(model)
    local bos = llama_vocab.llama_vocab_bos(vocab)
    if bos < 0 then
        bos = 0
    end

    local tokens = ffi_new("llama_token[1]", bos)
    local batch = llama_batch.llama_batch_get_one(tokens, 1)

    -- Tell llama.cpp this is a warmup so it does NOT count it in the
    -- perf summary.
    llama_context.llama_set_warmup(self._ptr, true)
    llama_context.llama_decode(self._ptr, batch)
    llama_context.llama_set_warmup(self._ptr, false)

    -- Wipe the dummy state so the next decode sees a clean context.
    self:kv_clear()
end

return M
