--- @module ion7.core.sampler
--- @author  ion7 / Ion7 Project Contributors
---
--- Composable sampler chains via a fluent builder pattern.
---
--- The chain is assembled by calling `Sampler.chain()` and chaining
--- per-sampler methods, then finalised with `:build()`. The order of
--- additions matters — llama.cpp applies samplers left-to-right at
--- sample time. Typical orders :
---
---   penalties → dry → top-k → top-p → min-p → xtc → temperature → dist
---
---   local s = ion7.Sampler.chain()
---     :penalties(64, 1.1)
---     :top_k(40)
---     :top_p(0.95)
---     :min_p(0.05)
---     :temperature(0.8)
---     :dist()
---     :build()
---
---   local tok = s:sample(ctx:ptr(), 0)   -- auto-accepts when called on a chain
---
--- Resource management :
---   - Sub-sampler `llama_sampler*` handles are OWNED by the chain after
---     `chain_add` ; do not free them yourself.
---   - The chain itself is `ffi.gc`-attached to `llama_sampler_free`.
---   - For Lua-implemented samplers (`CustomSampler`), the builder pins
---     the wrapper instance in `_custom_refs` so the trampolines stay
---     alive for the chain's lifetime.
---
--- The `reasoning_budget` step is the only one that uses the libcommon
--- bridge ; everything else is direct FFI to llama.cpp's stable API.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_sampler = require "ion7.core.ffi.llama.sampler" -- llama_sampler_*
local llama_perf = require "ion7.core.ffi.llama.perf" -- llama_perf_sampler*

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local ffi_string = ffi.string
local tonumber = tonumber

-- Forward declaration : `Sampler` is defined after `SamplerBuilder` but
-- `SamplerBuilder:build` needs to reference it. Lua locals are not
-- hoisted ; declaring upfront lets the assignment below capture the
-- same upvalue.
local Sampler

-- ── SamplerBuilder ────────────────────────────────────────────────────────

--- @class ion7.core.SamplerBuilder
--- @field _steps        table[]               Ordered list of per-sampler specs.
--- @field _custom_refs  ion7.core.CustomSampler[]? Pinned CustomSampler instances.
local SamplerBuilder = {}
SamplerBuilder.__index = SamplerBuilder

--- Start a new chain. Returns a fluent builder ; chain method calls
--- (`:top_k(...)`, `:dist()`, ...) and finalise with `:build()`.
--- @return ion7.core.SamplerBuilder
function SamplerBuilder.new()
    return setmetatable({_steps = {}}, SamplerBuilder)
end

-- Internal : append one step spec and return self for chaining.
local function push(self, spec)
    self._steps[#self._steps + 1] = spec
    return self
end

-- ── Per-sampler builder methods ───────────────────────────────────────────

--- Pick the highest-probability token at every step.
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:greedy()
    return push(self, {type = "greedy"})
end

--- Sample from the (post-temperature) probability distribution.
--- @param  seed integer? RNG seed (default `0xFFFFFFFF` = random).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:dist(seed)
    return push(self, {type = "dist", seed = seed or 0xFFFFFFFF})
end

--- Apply a temperature to the logits. `0` = greedy ; `1` = unmodified.
--- @param  t number
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:temperature(t)
    return push(self, {type = "temperature", t = t})
end

--- Alias : shorter spelling of `:temperature`.
function SamplerBuilder:temp(t)
    return self:temperature(t)
end

--- Dynamic temperature : final temperature drawn from `[base-range, base+range]`.
--- @param  base     number Centre value.
--- @param  range    number Half-width.
--- @param  exponent number? Exponent of the distribution (default 1).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:temperature_dynamic(base, range, exponent)
    return push(
        self,
        {
            type = "temperature_ext",
            t = base,
            delta = range,
            exponent = exponent or 1.0
        }
    )
end

--- Alias : shorter spelling of `:temperature_dynamic`.
function SamplerBuilder:temp_dynamic(base, range, exponent)
    return self:temperature_dynamic(base, range, exponent)
end

--- Top-K filtering : keep only the K most probable tokens.
--- @param  k integer
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:top_k(k)
    return push(self, {type = "top_k", k = k})
end

--- Nucleus (top-P) filtering : keep tokens up to cumulative probability `p`.
--- @param  p        number Threshold in `(0, 1]`.
--- @param  min_keep integer? Floor (default 1).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:top_p(p, min_keep)
    return push(self, {type = "top_p", p = p, min_keep = min_keep or 1})
end

--- Min-P filtering : keep tokens whose probability is at least `p × max_prob`.
--- @param  p        number Threshold relative to top token.
--- @param  min_keep integer? Floor (default 1).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:min_p(p, min_keep)
    return push(self, {type = "min_p", p = p, min_keep = min_keep or 1})
end

--- Top-N-sigma : keep tokens within `n` standard deviations of the
--- maximum logit. Newer (2025) alternative to top-k / top-p.
--- @param  n number
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:top_n_sigma(n)
    return push(self, {type = "top_n_sigma", n = n})
end

--- Locally-typical sampling : selects tokens whose information content
--- matches the expected entropy of the distribution. More natural
--- output than top-p alone for free-form generation.
--- @param  p        number Typicality threshold in `(0, 1]`.
--- @param  min_keep integer? Floor (default 1).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:typical(p, min_keep)
    return push(self, {type = "typical", p = p, min_keep = min_keep or 1})
end

--- XTC (Exclude Top Choices) : with probability `probability`, drop
--- candidates above `threshold` to encourage diversity.
--- @param  probability number
--- @param  threshold   number Logit threshold for exclusion.
--- @param  min_keep    integer?
--- @param  seed        integer?
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:xtc(probability, threshold, min_keep, seed)
    return push(
        self,
        {
            type = "xtc",
            probability = probability,
            threshold = threshold,
            min_keep = min_keep or 1,
            seed = seed or 0xFFFFFFFF
        }
    )
end

--- Mirostat v1 : adaptive sampling that targets a constant entropy.
--- @param  n_vocab integer Vocabulary size.
--- @param  tau     number? Target entropy (default 5).
--- @param  eta     number? Learning rate (default 0.1).
--- @param  m       integer? Mirostat M (default 100).
--- @param  seed    integer?
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:mirostat(n_vocab, tau, eta, m, seed)
    return push(
        self,
        {
            type = "mirostat",
            n_vocab = n_vocab,
            seed = seed or 0xFFFFFFFF,
            tau = tau or 5.0,
            eta = eta or 0.1,
            m = m or 100
        }
    )
end

--- Mirostat v2 : simpler than v1, no M parameter.
--- @param  tau  number? (default 5).
--- @param  eta  number? (default 0.1).
--- @param  seed integer?
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:mirostat_v2(tau, eta, seed)
    return push(
        self,
        {
            type = "mirostat_v2",
            seed = seed or 0xFFFFFFFF,
            tau = tau or 5.0,
            eta = eta or 0.1
        }
    )
end

--- Repetition / frequency / presence penalties.
--- @param  last_n          integer? Window in tokens (default 64).
--- @param  repeat_penalty  number?  > 1 reduces repeats (default 1).
--- @param  freq_penalty    number?  Per-token frequency penalty.
--- @param  present_penalty number?  Presence penalty.
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:penalties(last_n, repeat_penalty, freq_penalty, present_penalty)
    return push(
        self,
        {
            type = "penalties",
            last_n = last_n or 64,
            repeat_penalty = repeat_penalty or 1.0,
            freq_penalty = freq_penalty or 0.0,
            present_penalty = present_penalty or 0.0
        }
    )
end

--- DRY (Don't Repeat Yourself) repetition sampler.
--- @param  opts        table?  `{ multiplier, base, allowed_length, penalty_last_n, seq_breakers }`.
--- @param  vocab       cdata   `llama_vocab*` (required by llama.cpp).
--- @param  n_ctx_train integer Training context size (used for sizing).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:dry(opts, vocab, n_ctx_train)
    opts = opts or {}
    return push(
        self,
        {
            type = "dry",
            vocab = vocab,
            n_ctx_train = n_ctx_train or 4096,
            multiplier = opts.multiplier or 0.0,
            base = opts.base or 1.75,
            allowed_length = opts.allowed_length or 2,
            penalty_last_n = opts.penalty_last_n or -1,
            seq_breakers = opts.seq_breakers or {"\n", ":"}
        }
    )
end

--- Constrain output to a GBNF grammar.
--- @param  gbnf  string  GBNF grammar source.
--- @param  root  string? Root rule name (default `"root"`).
--- @param  vocab cdata|table `llama_vocab*` or a `Vocab` instance.
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:grammar(gbnf, root, vocab)
    local vptr = vocab and (type(vocab) == "table" and vocab._ptr or vocab) or nil
    return push(
        self,
        {
            type = "grammar",
            gbnf = gbnf or "",
            root = root or "root",
            vocab_ptr = vptr
        }
    )
end

--- Lazy grammar : the grammar only kicks in once one of the trigger
--- conditions is met. Useful for "free-form prose, then JSON when the
--- model emits `{`" workflows (CRANE-style).
function SamplerBuilder:grammar_lazy(gbnf, root, vocab, trigger_words, trigger_tokens, trigger_patterns)
    local vptr = vocab and (type(vocab) == "table" and vocab._ptr or vocab) or nil
    return push(
        self,
        {
            type = "grammar_lazy",
            gbnf = gbnf,
            root = root or "root",
            vocab_ptr = vptr,
            trigger_words = trigger_words or {},
            trigger_tokens = trigger_tokens or {},
            trigger_patterns = trigger_patterns or {}
        }
    )
end

--- Logit bias : `{ [token_id] = delta_logit, ... }`.
function SamplerBuilder:logit_bias(biases)
    return push(self, {type = "logit_bias", biases = biases})
end

--- Adaptive-p sampler : exponentially-decayed probability threshold.
function SamplerBuilder:adaptive_p(target, decay, seed)
    return push(
        self,
        {
            type = "adaptive_p",
            target = target or 0.45,
            decay = decay or 0.99,
            seed = seed or 0xFFFFFFFF
        }
    )
end

--- Plug a Lua-implemented `CustomSampler` into the chain. The builder
--- pins the wrapper so its trampolines outlive the chain.
function SamplerBuilder:custom(cs)
    return push(self, {type = "custom", sampler = cs})
end

--- Fill-in-the-Middle sampler. Requires a `vocab` cdata or `Vocab`.
function SamplerBuilder:infill(vocab)
    local vptr = vocab and (type(vocab) == "table" and vocab._ptr or vocab) or nil
    return push(self, {type = "infill", vocab_ptr = vptr})
end

--- Reasoning-budget sampler : caps the token count inside `<think>...</think>`
--- blocks. MUST be inserted FIRST in the chain. Routed through the
--- libcommon bridge.
--- @param  model    cdata|table `llama_model*` or a `Model` instance.
--- @param  n_budget integer     Max tokens inside the block (default 512).
--- @return ion7.core.SamplerBuilder
function SamplerBuilder:reasoning_budget(model, n_budget)
    local mptr = type(model) == "table" and model:ptr() or model
    return push(
        self,
        {
            type = "reasoning_budget",
            model_ptr = mptr,
            n_budget = n_budget or 512
        }
    )
end

-- ── build() : turn the spec list into a live chain ────────────────────────

--- Materialise the chain into a `Sampler` instance ready to use.
--- @return ion7.core.Sampler
--- @raise   When any sampler init returns NULL.
function SamplerBuilder:build()
    local chain_params = ffi_new("struct llama_sampler_chain_params")
    chain_params.no_perf = true
    local chain = llama_sampler.llama_sampler_chain_init(chain_params)
    if chain == nil then
        error("[ion7.core.sampler] failed to initialise sampler chain", 2)
    end

    for _, s in ipairs(self._steps) do
        local smpl

        if s.type == "greedy" then
            smpl = llama_sampler.llama_sampler_init_greedy()
        elseif s.type == "dist" then
            smpl = llama_sampler.llama_sampler_init_dist(s.seed)
        elseif s.type == "temperature" then
            smpl = llama_sampler.llama_sampler_init_temp(s.t)
        elseif s.type == "temperature_ext" then
            smpl = llama_sampler.llama_sampler_init_temp_ext(s.t or 0.8, s.delta or 0.0, s.exponent or 1.0)
        elseif s.type == "top_k" then
            smpl = llama_sampler.llama_sampler_init_top_k(s.k)
        elseif s.type == "top_p" then
            smpl = llama_sampler.llama_sampler_init_top_p(s.p, s.min_keep)
        elseif s.type == "min_p" then
            smpl = llama_sampler.llama_sampler_init_min_p(s.p, s.min_keep)
        elseif s.type == "top_n_sigma" then
            smpl = llama_sampler.llama_sampler_init_top_n_sigma(s.n)
        elseif s.type == "typical" then
            smpl = llama_sampler.llama_sampler_init_typical(s.p, s.min_keep)
        elseif s.type == "xtc" then
            smpl = llama_sampler.llama_sampler_init_xtc(s.probability, s.threshold, s.min_keep, s.seed)
        elseif s.type == "mirostat" then
            smpl = llama_sampler.llama_sampler_init_mirostat(s.n_vocab, s.seed, s.tau, s.eta, s.m)
        elseif s.type == "mirostat_v2" then
            smpl = llama_sampler.llama_sampler_init_mirostat_v2(s.seed, s.tau, s.eta)
        elseif s.type == "penalties" then
            smpl =
                llama_sampler.llama_sampler_init_penalties(
                s.last_n,
                s.repeat_penalty,
                s.freq_penalty,
                s.present_penalty
            )
        elseif s.type == "dry" then
            assert(s.vocab, "[ion7.core.sampler] DRY requires vocab")
            local n = #s.seq_breakers
            local breakers = ffi_new("const char*[?]", n)
            local _anchor = {} -- keep Lua strings alive
            for i, b in ipairs(s.seq_breakers) do
                _anchor[i] = b
                breakers[i - 1] = b
            end
            smpl =
                llama_sampler.llama_sampler_init_dry(
                s.vocab,
                s.n_ctx_train,
                s.multiplier,
                s.base,
                s.allowed_length,
                s.penalty_last_n,
                breakers,
                n
            )
        elseif s.type == "grammar" then
            assert(s.vocab_ptr, "[ion7.core.sampler] grammar requires vocab")
            smpl = llama_sampler.llama_sampler_init_grammar(s.vocab_ptr, s.gbnf, s.root)
        elseif s.type == "grammar_lazy" then
            assert(s.vocab_ptr, "[ion7.core.sampler] grammar_lazy requires vocab")
            local n_tw = #s.trigger_words
            local n_tt = #s.trigger_tokens
            local n_tp = #s.trigger_patterns
            local tt = (n_tt > 0) and ffi_new("int32_t[?]", n_tt) or nil
            for i, t in ipairs(s.trigger_tokens) do
                tt[i - 1] = t
            end

            -- llama.cpp exposes two distinct factories : `_grammar_lazy`
            -- takes plain trigger words ; `_grammar_lazy_patterns`
            -- takes regex patterns. Passing both at once is not a
            -- supported shape — we route by which list the caller
            -- populated.
            if n_tp > 0 then
                local tp = ffi_new("const char*[?]", n_tp)
                local _anchor = {}
                for i, p in ipairs(s.trigger_patterns) do
                    _anchor[#_anchor + 1] = p
                    tp[i - 1] = p
                end
                smpl = llama_sampler.llama_sampler_init_grammar_lazy_patterns(
                    s.vocab_ptr, s.gbnf, s.root, tp, n_tp, tt, n_tt)
            else
                local tw = (n_tw > 0) and ffi_new("const char*[?]", n_tw) or nil
                local _anchor = {}
                for i, w in ipairs(s.trigger_words) do
                    _anchor[#_anchor + 1] = w
                    tw[i - 1] = w
                end
                smpl = llama_sampler.llama_sampler_init_grammar_lazy(
                    s.vocab_ptr, s.gbnf, s.root, tw, n_tw, tt, n_tt)
            end
        elseif s.type == "logit_bias" then
            local list = {}
            for tok, delta in pairs(s.biases) do
                list[#list + 1] = {token = tok, bias = delta}
            end
            local arr = ffi_new("llama_logit_bias[?]", #list)
            for i, lb in ipairs(list) do
                arr[i - 1].token = lb.token
                arr[i - 1].bias = lb.bias
            end
            -- n_vocab is unused by this sampler ; passing 0 is canonical.
            smpl = llama_sampler.llama_sampler_init_logit_bias(0, #list, arr)
        elseif s.type == "adaptive_p" then
            smpl = llama_sampler.llama_sampler_init_adaptive_p(s.target, s.decay, s.seed)
        elseif s.type == "infill" then
            assert(s.vocab_ptr, "[ion7.core.sampler] infill requires vocab")
            smpl = llama_sampler.llama_sampler_init_infill(s.vocab_ptr)
        elseif s.type == "custom" then
            assert(
                s.sampler and type(s.sampler.ptr) == "function",
                "[ion7.core.sampler] custom requires a CustomSampler"
            )
            -- Pin the Lua wrapper so its trampolines stay alive.
            self._custom_refs = self._custom_refs or {}
            self._custom_refs[#self._custom_refs + 1] = s.sampler
            smpl = s.sampler:ptr()
        elseif s.type == "reasoning_budget" then
            local bridge = require "ion7.core.ffi.bridge"
            smpl = bridge.ion7_reasoning_budget_init(s.model_ptr, s.n_budget)
        else
            error("[ion7.core.sampler] unknown sampler type : " .. tostring(s.type), 2)
        end

        if smpl == nil then
            llama_sampler.llama_sampler_free(chain)
            error("[ion7.core.sampler] " .. s.type .. " init returned NULL", 2)
        end
        llama_sampler.llama_sampler_chain_add(chain, smpl)
    end

    return Sampler._wrap(ffi_gc(chain, llama_sampler.llama_sampler_free), self._custom_refs)
end

-- ── Sampler (the runtime handle) ──────────────────────────────────────────

--- @class ion7.core.Sampler
--- @field _chain        cdata `llama_sampler*` (chain head, GC-managed).
--- @field _custom_refs  table? Pinned CustomSampler instances (anchors).
Sampler = {}
Sampler.__index = Sampler

--- Internal : wrap a raw chain pointer into a Sampler instance.
function Sampler._wrap(chain, custom_refs)
    return setmetatable(
        {
            _chain = chain,
            _custom_refs = custom_refs
        },
        Sampler
    )
end

--- Sample the next token from the logits at position `idx`.
---
--- When called on a CHAIN (the typical case), `llama_sampler_sample`
--- internally calls `llama_sampler_accept` on every step. Do NOT call
--- `Sampler:accept(token)` separately afterwards — that would be a
--- double-accept and would corrupt grammar-state and DRY history.
---
--- @param  ctx cdata   `llama_context*`.
--- @param  idx integer Logit position (`-1` for the last decoded token).
--- @return integer Sampled token id.
function Sampler:sample(ctx, idx)
    return tonumber(llama_sampler.llama_sampler_sample(self._chain, ctx, idx))
end

--- Manually notify the chain that a token was accepted. Only useful
--- for sub-samplers obtained via `Sampler:get(i)` ; not needed when
--- you call `:sample` on the chain itself.
function Sampler:accept(token)
    llama_sampler.llama_sampler_accept(self._chain, token)
end

--- Reset every sub-sampler's state (penalty history, mirostat, grammar).
function Sampler:reset()
    llama_sampler.llama_sampler_reset(self._chain)
end

--- @return integer Current RNG seed of the chain.
function Sampler:seed()
    return tonumber(llama_sampler.llama_sampler_get_seed(self._chain))
end

--- @return string Sampler name string (top-level chain only).
function Sampler:name()
    local p = llama_sampler.llama_sampler_name(self._chain)
    return p ~= nil and ffi_string(p) or ""
end

--- @return integer Number of sub-samplers in the chain.
function Sampler:n()
    return tonumber(llama_sampler.llama_sampler_chain_n(self._chain))
end

--- Get the sub-sampler at index `i`. Returned handle is NOT GC-managed
--- (the chain owns it).
--- @param  i integer 0-based.
--- @return ion7.core.Sampler|nil
function Sampler:get(i)
    local s = llama_sampler.llama_sampler_chain_get(self._chain, i)
    if s == nil then
        return nil
    end
    return setmetatable({_chain = s}, Sampler)
end

--- Remove the sub-sampler at index `i` and return it. Caller takes
--- ownership — the returned Sampler is NOT GC-managed.
function Sampler:remove(i)
    local s = llama_sampler.llama_sampler_chain_remove(self._chain, i)
    if s == nil then
        return nil
    end
    return setmetatable({_chain = s}, Sampler)
end

--- Deep-clone the chain. The clone is independently GC-managed.
--- @return ion7.core.Sampler
function Sampler:clone()
    local c = llama_sampler.llama_sampler_clone(self._chain)
    if c == nil then
        error("[ion7.core.sampler] clone failed", 2)
    end
    return Sampler._wrap(ffi_gc(c, llama_sampler.llama_sampler_free), self._custom_refs)
end

--- Read sampler perf counters into a Lua table.
--- @return table { t_sample_ms, n_sample }
function Sampler:perf()
    local d = llama_perf.llama_perf_sampler(self._chain)
    return {
        t_sample_ms = tonumber(d.t_sample_ms),
        n_sample = tonumber(d.n_sample)
    }
end

--- Print sampler perf to stderr.
function Sampler:perf_print()
    llama_perf.llama_perf_sampler_print(self._chain)
end

--- Reset sampler perf counters.
function Sampler:perf_reset()
    llama_perf.llama_perf_sampler_reset(self._chain)
end

--- Free the chain immediately. Idempotent. Disarms `ffi.gc` so the GC
--- does not double-free later.
function Sampler:free()
    if self._chain == nil then
        return
    end
    ffi_gc(self._chain, nil)
    llama_sampler.llama_sampler_free(self._chain)
    self._chain = nil
    self._custom_refs = nil
end

-- ── Factory shortcuts ─────────────────────────────────────────────────────

--- Entry point of the fluent API.
--- @return ion7.core.SamplerBuilder
function Sampler.chain()
    return SamplerBuilder.new()
end

--- Pre-built balanced sampler : penalties + top-k 40 + top-p 0.95 +
--- min-p 0.05 + temp 0.8 + dist. Suitable for most chat workloads.
--- @return ion7.core.Sampler
function Sampler.default()
    return Sampler.chain():penalties(64, 1.0, 0.0, 0.0):top_k(40):top_p(0.95):min_p(0.05):temperature(0.8):dist():build(

    )
end

--- Pure greedy sampler. Deterministic, ideal for grammar-constrained
--- structured output where any randomness would break parsing.
--- @return ion7.core.Sampler
function Sampler.greedy()
    return Sampler.chain():greedy():build()
end

--- More adventurous defaults : higher temp, mild repetition control.
--- @return ion7.core.Sampler
function Sampler.creative()
    return Sampler.chain():penalties(128, 1.05, 0.0, 0.0):top_k(80):top_p(0.98):min_p(0.02):temperature(1.2):dist():build(

    )
end

-- ── CSampler shortcut ─────────────────────────────────────────────────────

-- Sampler.common is the libcommon-backed sampler (DRY/XTC/Mirostat/...
-- in a single bridge object). The implementation lives in the
-- `sampler/common` sub-module so this file stays under control.
Sampler.common = require("ion7.core.sampler.common").new

return Sampler
