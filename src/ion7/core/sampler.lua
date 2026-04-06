--- @module ion7.core.sampler
--- SPDX-License-Identifier: MIT
--- Composable sampler chains for token sampling.
---
--- Uses the Builder pattern: call Sampler.chain() to start, chain method
--- calls to add samplers in order, finish with :build() to get a live
--- llama_sampler* wrapped in a managed Sampler object.
---
--- The sampler chain is applied in the order samplers were added.
--- llama.cpp internally applies: logit-bias -> penalties -> dry ->
--- top-k -> typical -> top-p -> min-p -> xtc -> temperature -> dist
--- Use the order that matches your needs.
---
--- Resource management: the underlying llama_sampler* is freed automatically
--- when the Sampler object is garbage-collected (via ffi.gc).
---
--- @usage
---   local sampler = Sampler.chain()
---       :penalties({ repeat_last_n = 64, repeat_penalty = 1.1 })
---       :top_k(40)
---       :top_p(0.95)
---       :min_p(0.05)
---       :temperature(0.8)
---       :build()
---
---   local token = sampler:sample(ctx, 0)
---   sampler:accept(token)

local Loader = require "ion7.core.ffi.loader"

-- ── Forward declaration ─────────────────────────────────────────────────────────
-- Sampler is defined below SamplerBuilder but build() needs to reference it.
-- Declaring it here lets build() capture it as an upvalue by reference.
local Sampler

-- ── Builder ───────────────────────────────────────────────────────────────────

--- @class SamplerBuilder
--- @field _steps table  Ordered list of sampler specs.
local SamplerBuilder = {}
SamplerBuilder.__index = SamplerBuilder

--- Start building a sampler chain.
--- @return SamplerBuilder
function SamplerBuilder.new()
    return setmetatable({ _steps = {} }, SamplerBuilder)
end

-- Internal: append a step descriptor.
local function push(self, spec)
    self._steps[#self._steps + 1] = spec
    return self  -- fluent
end

--- Add a greedy sampler (always picks the highest-probability token).
--- @return SamplerBuilder
function SamplerBuilder:greedy()
    return push(self, { type = "greedy" })
end

--- Add a distribution sampler (samples from the probability distribution).
--- @param  seed  number?  RNG seed. Default: random.
--- @return SamplerBuilder
function SamplerBuilder:dist(seed)
    return push(self, { type = "dist", seed = seed or 0xFFFFFFFF })
end

--- Add a temperature sampler.
--- @param  t  number  Temperature (0 = greedy, 1 = unmodified, >1 = more random).
--- @return SamplerBuilder
--- Alias for temperature() -- shorter name.
--- @param  t  number
--- @return SamplerBuilder
function SamplerBuilder:temp(t) return self:temperature(t) end

function SamplerBuilder:temperature(t)
    return push(self, { type = "temperature", t = t })
end

--- Add a dynamic temperature sampler.
---
--- Final temperature is drawn from [base-range, base+range].
---
--- @param  base      number  Center temperature.
--- @param  range     number  Half-width of the temperature range.
--- @param  exponent  number? Exponent for the distribution (default: 1.0).
--- @return SamplerBuilder
--- Alias for temperature_dynamic().
function SamplerBuilder:temp_dynamic(base, range, exponent) return self:temperature_dynamic(base, range, exponent) end

function SamplerBuilder:temperature_dynamic(base, range, exponent)
    return push(self, {
        type     = "temperature_ext",
        t        = base,
        delta    = range,
        exponent = exponent or 1.0,
    })
end

--- Add a Top-K sampler.
--- @param  k  number  Keep only the K most probable tokens.
--- @return SamplerBuilder
function SamplerBuilder:top_k(k)
    return push(self, { type = "top_k", k = k })
end

--- Add a Nucleus (Top-P) sampler.
--- @param  p         number   Cumulative probability threshold (e.g. 0.95).
--- @param  min_keep  number?  Minimum tokens to keep (default: 1).
--- @return SamplerBuilder
function SamplerBuilder:top_p(p, min_keep)
    return push(self, { type = "top_p", p = p, min_keep = min_keep or 1 })
end

--- Add a Min-P sampler.
--- @param  p         number   Minimum probability relative to the top token.
--- @param  min_keep  number?  Minimum tokens to keep (default: 1).
--- @return SamplerBuilder
function SamplerBuilder:min_p(p, min_keep)
    return push(self, { type = "min_p", p = p, min_keep = min_keep or 1 })
end

--- Add a Top-N-Sigma sampler (2025).
--- Keeps tokens within N standard deviations of the maximum logit.
--- @param  n  number
--- @return SamplerBuilder
function SamplerBuilder:top_n_sigma(n)
    return push(self, { type = "top_n_sigma", n = n })
end

--- Add a Locally Typical sampling.
--- @param  p         number
--- @param  min_keep  number?
--- @return SamplerBuilder

function SamplerBuilder:typical(p, min_keep)
    return push(self, { type = "typical", p = p, min_keep = min_keep or 1 })
end

--- Add XTC (Exclude Top Choices) sampler.
--- @param  probability  number   Probability that XTC fires for a token.
--- @param  threshold    number   Logit threshold for exclusion.
--- @param  min_keep     number?
--- @param  seed         number?
--- @return SamplerBuilder
function SamplerBuilder:xtc(probability, threshold, min_keep, seed)
    return push(self, {
        type        = "xtc",
        probability = probability,
        threshold   = threshold,
        min_keep    = min_keep or 1,
        seed        = seed or 0xFFFFFFFF,
    })
end

--- Add Mirostat v1 sampler.
--- @param  n_vocab  number  Vocabulary size.
--- @param  tau      number  Target entropy (default: 5.0).
--- @param  eta      number  Learning rate (default: 0.1).
--- @param  m        number? M parameter (default: 100).
--- @param  seed     number?
--- @return SamplerBuilder
function SamplerBuilder:mirostat(n_vocab, tau, eta, m, seed)
    return push(self, {
        type    = "mirostat",
        n_vocab = n_vocab,
        seed    = seed or 0xFFFFFFFF,
        tau     = tau  or 5.0,
        eta     = eta  or 0.1,
        m       = m    or 100,
    })
end

--- Add Mirostat v2 sampler.
--- @param  tau   number?  Target entropy (default: 5.0).
--- @param  eta   number?  Learning rate (default: 0.1).
--- @param  seed  number?
--- @return SamplerBuilder
function SamplerBuilder:mirostat_v2(tau, eta, seed)
    return push(self, {
        type = "mirostat_v2",
        seed = seed or 0xFFFFFFFF,
        tau  = tau  or 5.0,
        eta  = eta  or 0.1,
    })
end

--- Add repetition/frequency/presence penalties.
---
--- @param  opts  table
---   opts.repeat_last_n    number?  Window for penalty (default: 64).
---   opts.repeat_penalty   number?  Repetition penalty > 1.0 (default: 1.0).
---   opts.frequency_penalty number? Frequency penalty (default: 0.0).
---   opts.presence_penalty  number? Presence penalty (default: 0.0).
--- @return SamplerBuilder
function SamplerBuilder:penalties(last_n, repeat_penalty, freq_penalty, present_penalty)
    self._steps[#self._steps + 1] = {
        type           = "penalties",
        last_n         = last_n or 64,
        repeat_penalty = repeat_penalty or 1.0,
        freq_penalty   = freq_penalty   or 0.0,
        present_penalty = present_penalty or 0.0,
    }
    return self
end

--- Add DRY (Don't Repeat Yourself) repetition sampler.
---
--- @param  opts  table
---   opts.multiplier      number?   DRY multiplier (default: 0.0 = disabled).
---   opts.base            number?   DRY base (default: 1.75).
---   opts.allowed_length  number?   Minimum sequence length before penalty (default: 2).
---   opts.penalty_last_n  number?   Context length for penalty (default: -1 = all).
---   opts.seq_breakers    table?    Strings that reset the penalty (default: {"\n",":"}).
--- @param  vocab         cdata  llama_vocab* (required for DRY).
--- @param  n_ctx_train   number Model's training context size.
--- @return SamplerBuilder
function SamplerBuilder:dry(opts, vocab, n_ctx_train)
    opts = opts or {}
    return push(self, {
        type             = "dry",
        vocab            = vocab,
        n_ctx_train      = n_ctx_train or 4096,
        multiplier       = opts.multiplier      or 0.0,
        base             = opts.base            or 1.75,
        allowed_length   = opts.allowed_length  or 2,
        penalty_last_n   = opts.penalty_last_n  or -1,
        seq_breakers     = opts.seq_breakers    or { "\n", ":" },
    })
end

--- Add a GBNF grammar sampler.
---
--- Constrains generation to text that matches the given grammar.
--- Use Grammar.json() or Grammar.from_json_schema() for structured output.
---
--- @param  gbnf_string  string  Grammar in GBNF format.
--- @param  root         string? Root rule name (default: "root").
--- @param  vocab        cdata   llama_vocab* (required).
--- @return SamplerBuilder
function SamplerBuilder:grammar(gbnf_string, root, vocab)
    -- vocab: ion7.Vocab object or raw llama_vocab* cdata
    local vptr = nil
    if vocab then
        vptr = (type(vocab) == "table" and vocab._ptr) or vocab
    end
    return push(self, {
        type      = "grammar",
        gbnf      = gbnf_string or "",
        root      = root or "root",
        vocab_ptr = vptr,
    })
end

--- Add a logit bias sampler.
---
--- Adjusts logits for specific tokens before sampling.
--- Positive bias increases the probability of a token; negative decreases it.
---
--- @param  biases  table  { [token_id] = delta_logit, ... }
--- @return SamplerBuilder
function SamplerBuilder:logit_bias(biases)
    return push(self, { type = "logit_bias", biases = biases })
end

--- Add an adaptive-p sampler (ref: https://github.com/ggml-org/llama.cpp/pull/17927).
--- Dynamically adjusts the probability threshold using exponential decay.
--- @param  target  number?  Target probability threshold (default: 0.45).
--- @param  decay   number?  Decay factor per step (default: 0.99).
--- @param  seed    number?  RNG seed (default: random).
--- @return SamplerBuilder
function SamplerBuilder:adaptive_p(target, decay, seed)
    return push(self, {
        type   = "adaptive_p",
        target = target or 0.45,
        decay  = decay  or 0.99,
        seed   = seed   or 0xFFFFFFFF,
    })
end

--- Add a lazy grammar sampler with trigger patterns.
--- Grammar is only activated when one of the trigger conditions is met.
--- @param  gbnf_string      string    Grammar in GBNF format.
--- @param  root             string?   Root rule name (default: "root").
--- @param  vocab            cdata     llama_vocab*.
--- @param  trigger_words    table?    Strings that trigger grammar activation.
--- @param  trigger_tokens   table?    Token IDs that trigger grammar activation.
--- @param  trigger_patterns table?    Regex patterns that trigger grammar activation.
--- @return SamplerBuilder
function SamplerBuilder:grammar_lazy(gbnf_string, root, vocab, trigger_words, trigger_tokens, trigger_patterns)
    return push(self, {
        type             = "grammar_lazy",
        gbnf             = gbnf_string,
        root             = root or "root",
        vocab            = vocab,
        trigger_words    = trigger_words    or {},
        trigger_tokens   = trigger_tokens   or {},
        trigger_patterns = trigger_patterns or {},
    })
end

--- Add a custom sampler written in Lua.
--- @param  custom_sampler  CustomSampler  From llama.CustomSampler.new().
--- @return SamplerBuilder
function SamplerBuilder:custom(custom_sampler)
    return push(self, { type = "custom", sampler = custom_sampler })
end

--- Add an infill (fill-in-the-middle) sampler.
--- @param  vocab  cdata  llama_vocab*.
--- @return SamplerBuilder
function SamplerBuilder:infill(vocab)
    return push(self, { type = "infill", vocab = vocab })
end

--- Add a custom Lua-implemented sampler to the chain.
--- @param  cs  CustomSampler  Created via llama.CustomSampler.new().
--- Build the sampler chain.
--- @return Sampler
--- @error  If any sampler initialisation fails.
function SamplerBuilder:build()
    local L      = Loader.instance()
    local lib    = L.lib
    local ffi    = L.ffi
    local C      = L.C

    local chain_params = ffi.new("llama_sampler_chain_params", { no_perf = true })
    local chain = lib.llama_sampler_chain_init(chain_params)
    if chain == nil then
        error("[ion7.core.sampler] failed to initialise sampler chain", 2)
    end

    for _, s in ipairs(self._steps) do
        local smpl

        if s.type == "greedy" then
            smpl = lib.llama_sampler_init_greedy()

        elseif s.type == "dist" then
            smpl = lib.llama_sampler_init_dist(s.seed)

        elseif s.type == "temperature" then
            smpl = lib.llama_sampler_init_temp(s.t)

        elseif s.type == "temperature_ext" then
            smpl = lib.llama_sampler_init_temp_ext(
                s.base or s.t or 0.8,
                s.range or s.delta or 0.0,
                s.exponent or 1.0)

        elseif s.type == "top_k" then
            smpl = lib.llama_sampler_init_top_k(s.k)

        elseif s.type == "top_p" then
            smpl = lib.llama_sampler_init_top_p(s.p, s.min_keep)

        elseif s.type == "min_p" then
            smpl = lib.llama_sampler_init_min_p(s.p, s.min_keep)

        elseif s.type == "top_n_sigma" then
            smpl = lib.llama_sampler_init_top_n_sigma(s.n)

        elseif s.type == "typical" then
            smpl = lib.llama_sampler_init_typical(s.p, s.min_keep)

        elseif s.type == "xtc" then
            smpl = lib.llama_sampler_init_xtc(
                s.probability, s.threshold, s.min_keep, s.seed)

        elseif s.type == "mirostat" then
            smpl = lib.llama_sampler_init_mirostat(
                s.n_vocab, s.seed, s.tau, s.eta, s.m)

        elseif s.type == "mirostat_v2" then
            smpl = lib.llama_sampler_init_mirostat_v2(s.seed, s.tau, s.eta)

        elseif s.type == "penalties" then
            smpl = lib.llama_sampler_init_penalties(
                s.last_n         or s.penalty_last_n  or 64,
                s.repeat_penalty or s.penalty_repeat  or 1.0,
                s.freq_penalty   or s.penalty_freq    or 0.0,
                s.present_penalty or s.penalty_present or 0.0
            )

        elseif s.type == "dry" then
            assert(s.vocab, "[ion7.core.sampler] DRY requires vocab")
            local n_breakers = #s.seq_breakers
            local breakers   = ffi.new("const char*[?]", n_breakers)
            local _bk_refs   = {}  -- keep alive
            for i, b in ipairs(s.seq_breakers) do
                _bk_refs[i] = b
                breakers[i - 1] = b
            end
            smpl = lib.llama_sampler_init_dry(
                s.vocab, s.n_ctx_train,
                s.multiplier, s.base,
                s.allowed_length, s.penalty_last_n,
                breakers, n_breakers
            )

        elseif s.type == "grammar" then
            assert(s.vocab_ptr, "[ion7.core.sampler] grammar requires vocab")
            smpl = lib.llama_sampler_init_grammar(s.vocab_ptr, s.gbnf, s.root or "root")

        elseif s.type == "logit_bias" then
            local pairs_list = {}
            local n = 0
            for tok_id, delta in pairs(s.biases) do
                pairs_list[#pairs_list + 1] = { token = tok_id, bias = delta }
                n = n + 1
            end
            local arr = ffi.new("llama_logit_bias[?]", n)
            for i, p in ipairs(pairs_list) do
                arr[i - 1].token = p.token
                arr[i - 1].bias  = p.bias
            end
            -- n_vocab is 0 here -- llama.cpp ignores it for this sampler
            smpl = lib.llama_sampler_init_logit_bias(0, n, arr)

        elseif s.type == "adaptive_p" then
            smpl = lib.llama_sampler_init_adaptive_p(s.target, s.decay, s.seed)

        elseif s.type == "grammar_lazy" then
            assert(s.vocab, "[ion7.core.sampler] grammar_lazy requires vocab")
            local n_tw = #s.trigger_words
            local n_tt = #s.trigger_tokens
            local n_tp = #s.trigger_patterns
            local tw   = n_tw > 0 and ffi.new("const char*[?]", n_tw) or nil
            local tt   = n_tt > 0 and ffi.new("int32_t[?]", n_tt) or nil
            local tp   = n_tp > 0 and ffi.new("const char*[?]", n_tp) or nil
            local _tw_refs = {}
            for i, w in ipairs(s.trigger_words)    do _tw_refs[i]=w; tw[i-1]=w end
            for i, t in ipairs(s.trigger_tokens)   do tt[i-1]=t end
            for i, p in ipairs(s.trigger_patterns) do _tw_refs[#_tw_refs+1]=p; tp[i-1]=p end
            smpl = lib.llama_sampler_init_grammar_lazy_patterns(
                s.vocab, s.gbnf, s.root,
                tw, n_tw, tt, n_tt, tp, n_tp)

        elseif s.type == "infill" then
            assert(s.vocab, "[ion7.core.sampler] infill requires vocab")
            smpl = lib.llama_sampler_init_infill(s.vocab)

        elseif s.type == "custom" then
            assert(s.sampler and type(s.sampler.ptr) == "function",
                "[ion7.core.sampler] custom requires a CustomSampler object")
            -- Keep the CustomSampler alive to prevent GC of Lua callbacks
            self._custom_refs = self._custom_refs or {}
            self._custom_refs[#self._custom_refs + 1] = s.sampler
            -- Let the general llama_sampler_chain_add below handle the add
            smpl = s.sampler:ptr()

        else
            error(string.format(
                "[ion7.core.sampler] unknown sampler type: %s", s.type), 2)
        end

        if smpl == nil then
            lib.llama_sampler_free(chain)
            error(string.format(
                "[ion7.core.sampler] %s init returned NULL", s.type), 2)
        end
        lib.llama_sampler_chain_add(chain, smpl)
    end

    -- Wrap in managed object
    return Sampler._wrap(ffi.gc(chain, lib.llama_sampler_free), lib)
end

-- ── Sampler ───────────────────────────────────────────────────────────────────

--- @class Sampler
--- @field _chain  cdata  llama_sampler* (freed on GC).
--- @field _lib    cdata  libllama.so namespace.
Sampler = {}  -- assigns to the forward-declared local above
Sampler.__index = Sampler

--- Internal: wrap a raw chain pointer.
function Sampler._wrap(chain, lib)
    return setmetatable({ _chain = chain, _lib = lib }, Sampler)
end

--- Sample the next token from context position idx.
---
--- When called on a sampler CHAIN (the common case), llama_sampler_sample()
--- internally calls llama_sampler_accept() on all samplers in the chain.
--- Do NOT call Sampler:accept() separately after sample() - it would be a
--- double-accept and will corrupt grammar sampler state, causing a crash.
---
--- @param  ctx  cdata   llama_context*.
--- @param  idx  number  Logit position (-1 = last decoded position).
--- @return number  Token ID.
function Sampler:sample(ctx, idx)
    return tonumber(self._lib.llama_sampler_sample(self._chain, ctx, idx))
end

--- Inform the sampler chain that a token was accepted.
--- Must be called after each generated token to update penalty state.
---
--- @param  token  number  Accepted token ID.
function Sampler:accept(token)
    self._lib.llama_sampler_accept(self._chain, token)
end

--- Reset the sampler chain state (penalty history, mirostat state, etc.).
function Sampler:reset()
    self._lib.llama_sampler_reset(self._chain)
end

--- @return number  Current RNG seed of the chain.
function Sampler:seed()
    return tonumber(self._lib.llama_sampler_get_seed(self._chain))
end

--- @return string  Name of the sampler.
function Sampler:name()
    local p = self._lib.llama_sampler_name(self._chain)
    return p ~= nil and Loader.instance().ffi.string(p) or ""
end

--- @return number  Number of samplers in the chain.
function Sampler:n()
    return tonumber(self._lib.llama_sampler_chain_n(self._chain))
end

--- Return the sampler at position i in the chain.
--- @param  i  number  0-based index.
--- @return Sampler  (not GC-managed -- owned by the chain)
function Sampler:get(i)
    local smpl = self._lib.llama_sampler_chain_get(self._chain, i)
    if smpl == nil then return nil end
    return setmetatable({ _chain = smpl, _lib = self._lib }, Sampler)
end

--- Clone this sampler chain.
--- The clone is independently GC-managed.
--- @return Sampler
function Sampler:clone()
    local cloned = self._lib.llama_sampler_clone(self._chain)
    if cloned == nil then error("[ion7.core.sampler] clone failed", 2) end
    local ffi = Loader.instance().ffi
    return Sampler._wrap(ffi.gc(cloned, self._lib.llama_sampler_free), self._lib)
end

--- Return performance data for this sampler chain.
--- @return table  { t_sample_ms, n_sample }
function Sampler:perf()
    local d = self._lib.llama_perf_sampler(self._chain)
    return {
        t_sample_ms = tonumber(d.t_sample_ms),
        n_sample    = tonumber(d.n_sample),
    }
end

--- Print sampler performance to stderr.
function Sampler:perf_print()
    self._lib.llama_perf_sampler_print(self._chain)
end

--- Reset sampler performance counters.
function Sampler:perf_reset()
    self._lib.llama_perf_sampler_reset(self._chain)
end

--- Explicitly free the sampler chain and cancel the GC finalizer.
---
--- The sampler is freed automatically on GC, so calling this is optional.
--- Use it in tight generation loops where you want deterministic cleanup
--- without waiting for the garbage collector.
---
--- After free(), the Sampler object is invalid and must not be used.
function Sampler:free()
    if self._chain == nil then return end
    local ffi = Loader.instance().ffi
    -- Cancel the ffi.gc finalizer to prevent double-free,
    -- then free the chain immediately.
    ffi.gc(self._chain, nil)
    self._lib.llama_sampler_free(self._chain)
    self._chain = nil
end

-- ── Factory shortcuts ─────────────────────────────────────────────────────────

--- Returns a SamplerBuilder (entry point for the fluent API).
--- @return SamplerBuilder
function Sampler.chain()
    return SamplerBuilder.new()
end

--- Default sampler: temperature=0.8, top_k=40, top_p=0.95, min_p=0.05.
--- Suitable for most conversational use cases.
--- @return Sampler
function Sampler.default()
    return Sampler.chain()
        :penalties(64, 1.0, 0.0, 0.0)
        :top_k(40)
        :top_p(0.95)
        :min_p(0.05)
        :temperature(0.8)
        :dist()
        :build()
end

--- Pure greedy sampler (deterministic, best-token at each step).
--- Useful for structured output tasks alongside grammar sampling.
--- @return Sampler
function Sampler.greedy()
    return Sampler.chain():greedy():build()
end

--- Creative sampler: higher temperature, mild repetition control.
--- @return Sampler
function Sampler.creative()
    return Sampler.chain()
        :penalties(128, 1.05, 0.0, 0.0)
        :top_k(80)
        :top_p(0.98)
        :min_p(0.02)
        :temperature(1.2)
        :dist()
        :build()
end

-- Sampler.json() is available in ion7-grammar module (not in core).


return Sampler
