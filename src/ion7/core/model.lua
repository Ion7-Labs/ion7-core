--- @module ion7.core.model
--- SPDX-License-Identifier: MIT
--- Top-level model object: loading, context creation, vocabulary, metadata.
---
--- A Model wraps a llama_model* loaded from a GGUF file. It is the root
--- of the ion7-core object graph. All contexts and vocabulary handles are
--- created from it.
---
--- A single Model may host multiple simultaneous Context instances, enabling
--- multi-session setups where a cheap embedding context and a heavier
--- generation context share the same weights.
---
--- Resource management: the llama_model* is freed automatically when the
--- Model object is garbage-collected. All Context instances keep a Lua
--- reference to their parent Model, so the GC ordering is safe.
---
--- @usage
---   local llama = require "ion7.core"
---   llama.init({ log_level = 1 })
---
---   local model = llama.Model.load("qwen3.5-27b.gguf", {
---       n_gpu_layers = 25,
---   })
---   print(model:desc())           -- "Qwen3 27B Q4_K_M"
---   print(model:n_params() / 1e9) -- 27.36 (billions)
---   print(model:is_recurrent())   -- false
---
---   local ctx   = model:context({ n_ctx = 65536, kv_type = "q8_0" })
---   local vocab = model:vocab()

local Loader  = require "ion7.core.ffi.loader"
local Vocab   = require "ion7.core.vocab"
local Context = require "ion7.core.context"

-- ── KV quant map ──────────────────────────────────────────────────────────────

-- KV cache type → GGML_TYPE_* value passed directly to bridge.
-- V cache (type_v) supports: f16, bf16, q8_0, q4_0 (no flash needed for these).
-- K cache (type_k) additionally supports: q4_1, q5_0, q5_1, iq4_nl.
-- Q4_K/Q5_K for V require flash attention + GGML_CUDA_FA_ALL_QUANTS.
local KV_TYPES = {
    f16    =  1,   -- default, best quality
    bf16   = 30,   -- BF16 - good quality, slightly smaller than f16
    q8_0   =  8,   -- ~50% VRAM savings, minimal quality loss
    q4_0   =  2,   -- ~75% VRAM savings, some quality loss
    q4_1   =  3,   -- ~75% VRAM savings, slightly better than q4_0
    q5_0   =  6,   -- ~60% VRAM savings
    q5_1   =  7,   -- ~60% VRAM savings, slightly better than q5_0
    iq4_nl = 20,   -- ~75% VRAM savings, better quality than q4_0
    q4_k   = 12,   -- K cache only (requires flash attn for V)
    q5_k   = 13,   -- K cache only (requires flash attn for V)
}

-- ── Scratch buffer for string queries ─────────────────────────────────────────

local META_BUF_SIZE = 4096

-- ── Model ─────────────────────────────────────────────────────────────────────

--- @class Model
--- @field _ptr   cdata   llama_model* (freed on GC).
--- @field path   string  Path to the source GGUF file.
local Model = {}
Model.__index = Model

--- Load a sharded GGUF model from multiple files.
--- Used for large models split across several files (e.g. 70B models).
--- The paths must be in the correct shard order.
---
--- @param  paths  table   Array of paths to GGUF shard files.
--- @param  opts   table?  Same options as Model.load().
--- @--- Load a model from an open file descriptor.
--- Useful for loading models from memory-mapped files, pipes, or custom sources.
---
--- @param  fd            number   Open file descriptor (readable, seekable).
--- @param  n_gpu_layers  number?  GPU layers (-1 = all, 0 = CPU). Default: auto-fit.
--- @return Model
--- Quantize a model file and save the result.
---
--- @param  inp_path  string   Path to the source GGUF model.
--- @param  out_path  string   Output path for the quantized GGUF.
--- @param  opts      table?
---   opts.ftype                 string?   Quantization type (default: "q4_k_m").
---                              Classic: "f32","f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0"
---                              K-quant: "q2_k","q2_k_s","q3_k_s","q3_k_m","q3_k_l",
---                                       "q4_k_s","q4_k_m","q5_k_s","q5_k_m","q6_k"
---                              IQ:      "iq1_s","iq1_m","iq2_xxs","iq2_xs","iq2_s","iq2_m",
---                                       "iq3_xxs","iq3_xs","iq3_s","iq3_m","iq4_nl","iq4_xs"
---                              TQ/new:  "tq1_0","tq2_0","mxfp4_moe","nvfp4","q1_0"
---                              Other:   "copy"
---   opts.nthread               number?   Threads (default: 0 = hardware concurrency).
---   opts.allow_requantize      bool?     Re-quantize already-quantized tensors.
---   opts.quantize_output       bool?     Quantize the output tensor too.
---   opts.pure                  bool?     Quantize ALL tensors (no 1d exceptions).
---   opts.keep_split            bool?     Preserve shard count.
---   opts.dry_run               bool?     Compute size without writing.
--- @return number  0 on success, error code otherwise.
function Model.quantize(inp_path, out_path, opts)
    assert(type(inp_path) == "string", "[ion7.core.model] inp_path must be a string")
    assert(type(out_path) == "string", "[ion7.core.model] out_path must be a string")
    opts = opts or {}

    local L   = Loader.instance()
    local lib = L.lib

    local ftype_map = {
        f32        =  0,   f16        =  1,
        q4_0       =  2,   q4_1       =  3,
        q8_0       =  7,   q5_0       =  8,   q5_1       =  9,
        q2_k       = 10,   q3_k_s     = 11,   q3_k_m     = 12,   q3_k_l     = 13,
        q4_k_s     = 14,   q4_k_m     = 15,   q5_k_s     = 16,   q5_k_m     = 17,
        q6_k       = 18,
        iq2_xxs    = 19,   iq2_xs     = 20,   q2_k_s     = 21,   iq3_xs     = 22,
        iq3_xxs    = 23,   iq1_s      = 24,   iq4_nl     = 25,   iq3_s      = 26,
        iq3_m      = 27,   iq2_s      = 28,   iq2_m      = 29,   iq4_xs     = 30,
        iq1_m      = 31,   bf16       = 32,
        tq1_0      = 36,   tq2_0      = 37,   mxfp4_moe  = 38,   nvfp4      = 39,
        q1_0       = 40,   copy       = -1,
    }
    local ftype_key = (opts.ftype or "q4_k_m"):lower():gsub("-", "_")
    local ftype_val = ftype_map[ftype_key]
    assert(ftype_val ~= nil, string.format(
        "[ion7.core.model] unknown ftype '%s'", opts.ftype or "q4_k_m"))

    local params = lib.llama_model_quantize_default_params()
    params.nthread                = opts.nthread            or 0
    params.ftype                  = ftype_val
    params.allow_requantize       = opts.allow_requantize   and true or false
    params.quantize_output_tensor = opts.quantize_output    and true or false
    params.pure                   = opts.pure               and true or false
    params.keep_split             = opts.keep_split         and true or false
    params.dry_run                = opts.dry_run            and true or false

    return tonumber(Loader.instance().bridge.ion7_model_quantize(
        inp_path, out_path,
        params.ftype,
        params.nthread,
        params.pure and 1 or 0,
        params.allow_requantize and 1 or 0,
        params.quantize_output_tensor and 1 or 0,
        params.dry_run and 1 or 0
    ))
end

--- @error  If loading fails.
function Model.load_splits(paths, opts)
    assert(type(paths) == "table" and #paths > 0,
        "[ion7.core.model] paths must be a non-empty table of strings")
    opts = opts or {}
    -- Hint GC before large allocation
    collectgarbage("collect")
    local L   = Loader.instance()
    local ffi = L.ffi
    local n   = #paths
    local arr = ffi.new("const char*[?]", n)
    local _refs = {}
    for i, p in ipairs(paths) do _refs[i] = p; arr[i-1] = p end
    local ptr = L.bridge.ion7_model_load_splits(arr, n, opts.n_gpu_layers or 0)
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load sharded model (%d parts)", n), 2)
    end
    return setmetatable({
        _ptr    = ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib, _bridge = L.bridge, _ffi = ffi, _vocab = nil,
        path    = paths[1],
    }, Model)
end

--- Load a model from an already-open FILE* handle.
---
--- Useful for in-memory models, pipe-loading, or when the file is already
--- open for inspection. The caller owns the file - it is NOT closed.
---
--- @param  file          cdata   FILE* (from ffi.C.fopen or similar).
--- @param  opts          table?
---   opts.n_gpu_layers   number? (default: 0)
--- @return Model
--- @error  If loading fails.
function Model.load_from_fd(file, opts)
    assert(file ~= nil, "[ion7.core.model] load_from_fd: FILE* is nil")
    opts = opts or {}
    local L   = Loader.instance()
    local ptr = L.bridge.ion7_model_load_fd(file, opts.n_gpu_layers or 0)
    if ptr == nil then
        error("[ion7.core.model] load_from_fd: failed to load model", 2)
    end
    return setmetatable({
        _ptr    = L.ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib,
        _bridge = L.bridge,
        _ffi    = L.ffi,
        _vocab  = nil,
        path    = "<file_ptr>",
    }, Model)
end

--- Load a GGUF model from disk.
---
--- @param  path  string  Absolute path to the .gguf file.
--- @param  opts  table?
---   opts.n_gpu_layers  number?  Layers to offload. 0=CPU, -1=all (default: 0).
---   opts.use_mmap      bool?    Memory-map the file (default: true).
---   opts.use_mlock     bool?    Lock pages in RAM (default: false).
---   opts.vocab_only    bool?    Load vocabulary only, no weights (default: false).
--- @return Model
--- @error  If the file cannot be loaded.
function Model.load(path, opts)
    assert(type(path) == "string" and #path > 0,
        "[ion7.core.model] path must be a non-empty string")
    opts = opts or {}
    local L = Loader.instance()

    local ptr = L.bridge.ion7_model_load(
        path,
        opts.n_gpu_layers or 0,
        opts.use_mmap ~= false and 1 or 0,
        opts.use_mlock and 1 or 0,
        opts.vocab_only and 1 or 0
    )
    if ptr == nil then
        error(string.format("[ion7.core.model] failed to load '%s'", path), 2)
    end

    return setmetatable({
        _ptr    = L.ffi.gc(ptr, L.bridge.ion7_model_free),
        _lib    = L.lib,
        _bridge = L.bridge,
        _ffi    = L.ffi,
        _vocab  = nil,
        path    = path,
    }, Model)
end

--- Auto-fit model and context parameters to available device VRAM.
---
--- Simulates allocations and finds the maximum n_gpu_layers + n_ctx that
--- fit without running out of VRAM. Recommended call before creating a
--- context on machines with limited GPU memory.
---
--- @param  path     string  Path to GGUF file (for size estimation).
--- @param  opts     table?
---   opts.n_gpu_layers  number?  Starting point (default: -1 = try all).
---   opts.n_ctx         number?  Desired context size (default: 4096).
---   opts.n_ctx_min     number?  Minimum acceptable context (default: 512).
--- @return table?  { n_gpu_layers, n_ctx } or nil if no fit found.
function Model.fit_params(path, opts)
    opts = opts or {}
    local L   = Loader.instance()
    local ffi = L.ffi

    -- Always start from 0 so llama_params_fit can compute the optimal
    -- value upward. llama_params_fit only modifies fields still at their
    -- default value; passing a non-default would skip the computation.
    local n_gpu = ffi.new("int32_t[1]", 0)
    local n_ctx = ffi.new("uint32_t[1]", opts.n_ctx        or 4096)
    local min   = opts.n_ctx_min or 512

    local status = L.bridge.ion7_params_fit(path, n_gpu, n_ctx, min)
    if status ~= 0 then return nil end

    return {
        n_gpu_layers = tonumber(n_gpu[0]),
        n_ctx        = tonumber(n_ctx[0]),
    }
end

--- Return the raw llama_model* pointer.
--- @return cdata
--- Explicitly free the model and release VRAM immediately.
--- After free(), this object must not be used.
function Model:free()
    if self._ptr then
        self._bridge.ion7_model_free(self._ptr)
        self._ptr = self._ffi.gc(self._ptr, nil)
        self._ptr = nil
        self._vocab = nil
    end
end

function Model:ptr() return self._ptr end

-- ── Introspection ─────────────────────────────────────────────────────────────

--- Get a human-readable description of the model type.
---
--- @return string  e.g. "LLaMA 3.1 8B Q4_K_M"
function Model:desc()
    local buf = self._ffi.new("char[512]")
    local n   = self._bridge.ion7_model_desc(self._ptr, buf, 512)
    return n > 0 and self._ffi.string(buf, n) or "(unknown)"
end

--- Total size of all model tensors in bytes.
--- @return number
function Model:size()
    -- uint64_t → cast to double (safe for values < 2^53, i.e. up to 8 PB)
    return tonumber(self._ffi.cast("double", self._bridge.ion7_model_size(self._ptr)))
end

--- Total number of parameters.
---
--- @return number  e.g. 8030000000 for an 8B model.
function Model:n_params()
    return tonumber(self._ffi.cast("double", self._bridge.ion7_model_n_params(self._ptr)))
end

--- Training context size (maximum safe context).
--- @return number
function Model:n_ctx_train() return tonumber(self._bridge.ion7_model_n_ctx_train(self._ptr)) end

--- Hidden embedding dimension.
--- @return number
function Model:n_embd() return tonumber(self._bridge.ion7_model_n_embd(self._ptr)) end

--- Input embedding dimension (may differ from n_embd for multimodal models).
--- @return number
function Model:n_embd_inp() return tonumber(self._bridge.ion7_model_n_embd_inp(self._ptr)) end

--- Output embedding dimension.
--- @return number
function Model:n_embd_out() return tonumber(self._bridge.ion7_model_n_embd_out(self._ptr)) end

--- Number of transformer layers.
--- @return number
function Model:n_layer() return tonumber(self._bridge.ion7_model_n_layer(self._ptr)) end

--- Number of attention heads.
--- @return number
function Model:n_head() return tonumber(self._bridge.ion7_model_n_head(self._ptr)) end

--- Number of key-value heads (less than n_head for GQA models like Llama3, Qwen).
--- @return number
function Model:n_head_kv() return tonumber(self._bridge.ion7_model_n_head_kv(self._ptr)) end

--- Sliding window attention size (0 if not applicable).
--- @return number
function Model:n_swa() return tonumber(self._bridge.ion7_model_n_swa(self._ptr)) end

--- Returns true if the model requires llama_encode() (encoder-decoder: T5, Whisper).
--- @return bool
function Model:has_encoder() return self._bridge.ion7_model_has_encoder(self._ptr) == 1 end

--- Returns true if the model uses llama_decode() (virtually all LLMs do).
--- @return bool
function Model:has_decoder() return self._bridge.ion7_model_has_decoder(self._ptr) == 1 end

--- Returns true for recurrent-architecture models (Mamba, RWKV, etc.).
--- @return bool
function Model:is_recurrent() return self._bridge.ion7_model_is_recurrent(self._ptr) == 1 end

--- Returns true for hybrid attention+SSM models (Jamba, Granite).
--- @return bool
function Model:is_hybrid() return self._bridge.ion7_model_is_hybrid(self._ptr) == 1 end

--- Returns true for diffusion-based models (LLaDA, Dream).
--- @return bool
function Model:is_diffusion() return self._bridge.ion7_model_is_diffusion(self._ptr) == 1 end

--- Get the RoPE frequency scale factor from training.
--- @return number
function Model:rope_freq_scale_train()
    return tonumber(self._lib.llama_model_rope_freq_scale_train(self._ptr))
end

--- Number of classifier output classes (only valid for classifier models).
--- @return number
function Model:n_cls_out()
    return tonumber(self._lib.llama_model_n_cls_out(self._ptr))
end

--- Label of classifier output class at index i.
--- @param  i  number  0-based index.
--- @return string?
function Model:cls_label(i)
    local p = self._lib.llama_model_cls_label(self._ptr, i)
    return p ~= nil and self._ffi.string(p) or nil
end

--- Return the RoPE type used by this model.
--- Useful to detect MRoPE models (Qwen3.5, etc.) where kv_seq_rm is unreliable.
--- Returns: "none" | "norm" | "neox" | "mrope" | "imrope" | "vision"
--- @return string
function Model:rope_type()
    local p = self._bridge.ion7_model_rope_type(self._ptr)
    return (p ~= nil) and self._ffi.string(p) or "unknown"
end

--- Returns the decoder start token for encoder-decoder models (-1 for decoder-only).
--- @return number
function Model:decoder_start_token()
    return tonumber(self._lib.llama_model_decoder_start_token(self._ptr))
end

--- Return a summary table of model properties.
---
--- @return table
---   { path, desc, n_params, size_gb, n_ctx_train, n_embd, n_layer,
---     n_head, n_head_kv, n_swa, n_vocab, vocab_type,
---     has_encoder, is_recurrent, is_hybrid, is_diffusion }
function Model:info()
    local v  = self:vocab()
    local sz = self:size()
    return {
        path         = self.path,
        desc         = self:desc(),
        n_params     = self:n_params(),
        size         = sz,
        size_gb      = sz / (1024^3),
        n_ctx_train  = self:n_ctx_train(),
        n_embd       = self:n_embd(),
        n_embd_inp   = self:n_embd_inp(),
        n_embd_out   = self:n_embd_out(),
        n_layer      = self:n_layer(),
        n_head       = self:n_head(),
        n_head_kv    = self:n_head_kv(),
        n_swa        = self:n_swa(),
        n_vocab      = v:n_vocab(),
        vocab_type   = v:type(),
        has_encoder  = self:has_encoder(),
        has_decoder  = self:has_decoder(),
        is_recurrent = self:is_recurrent(),
        is_hybrid    = self:is_hybrid(),
        is_diffusion = self:is_diffusion(),
    }
end

-- ── Metadata ─────────────────────────────────────────────────────────────────

--- Get a GGUF metadata value by key name.
---
--- @param  key  string  Metadata key, e.g. "general.name", "llama.context_length".
--- @return string?  Value string, or nil if the key does not exist.
-- Stable public API aliases
function Model:meta_count()
    return tonumber(self._bridge.ion7_model_meta_count(self._ptr))
end

function Model:meta_key_at(i)
    local buf = self._ffi.new("char[512]")
    local n = self._bridge.ion7_model_meta_key_at(self._ptr, i, buf, 512)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

function Model:meta_val_at(i)
    local buf = self._ffi.new("char[4096]")
    local n = self._bridge.ion7_model_meta_val_at(self._ptr, i, buf, 4096)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

function Model:meta_val(key)
    return self:meta(key)
end

function Model:lora_load(path)
    return self:load_lora(path)
end

function Model:meta(key)
    local buf = self._ffi.new("char[?]", META_BUF_SIZE)
    local n   = self._bridge.ion7_model_meta_val(self._ptr, key, buf, META_BUF_SIZE)
    return n >= 0 and self._ffi.string(buf, n) or nil
end

--- Return all GGUF metadata as a Lua table.
---
--- @return table  { key = value, ... } (all values are strings).
function Model:meta_all()
    local result = {}
    local n      = self._bridge.ion7_model_meta_count(self._ptr)
    local kbuf   = self._ffi.new("char[?]", META_BUF_SIZE)
    local vbuf   = self._ffi.new("char[?]", META_BUF_SIZE)

    for i = 0, n - 1 do
        local kn = self._bridge.ion7_model_meta_key_at(self._ptr, i, kbuf, META_BUF_SIZE)
        local vn = self._bridge.ion7_model_meta_val_at(self._ptr, i, vbuf, META_BUF_SIZE)
        if kn >= 0 and vn >= 0 then
            local k = self._ffi.string(kbuf, kn)
            local v = self._ffi.string(vbuf, vn)
            result[k] = v
        end
    end
    return result
end

--- Get the model's built-in Jinja2 chat template string.
---
--- @param  name  string?  Template variant name, or nil for the default.
--- @return string?  Template string, or nil if not embedded in the GGUF.
function Model:chat_template(name)
    local p = self._bridge.ion7_model_chat_template(self._ptr, name)
    return (p ~= nil) and self._ffi.string(p) or nil
end

-- ── Vocabulary ────────────────────────────────────────────────────────────────

--- Return the Vocab handle for this model. Created once and cached.
--- @return Vocab
function Model:vocab()
    if self._vocab then return self._vocab end
    local ptr = self._lib.llama_model_get_vocab(self._ptr)
    assert(ptr ~= nil, "[ion7.core.model] llama_model_get_vocab returned NULL")
    self._vocab = Vocab.new(self._lib, self._ptr, ptr)
    return self._vocab
end

-- ── Context creation ─────────────────────────────────────────────────────────

--- Create an inference context for text generation.
---
--- @param  opts  table?
---   opts.n_ctx           number?  Context window in tokens (default: 4096).
---   opts.n_batch         number?  Logical batch size (default: 2048).
---   opts.n_ubatch        number?  Physical micro-batch size. Default: "auto" (512 for
---                              GPU contexts, min(n_batch, 256) for CPU) for lower
---                              Time-To-First-Token latency. Set explicitly to override.
---   opts.n_threads       number?  CPU threads for generation (default: 4).
---   opts.n_threads_batch number?  CPU threads for prompt eval (default: 2*n_threads).
---   opts.flash_attn      bool?    Force Flash Attention ON (default: AUTO).
---   opts.offload_kqv     bool?    Keep KV cache on GPU (default: true).
---   opts.op_offload      bool?    Offload host tensor ops to GPU (default: false).
---   opts.no_perf         bool?    Disable perf counters (default: false).
---   opts.kv_type         string?  KV cache type for both K and V:
---                                 "f16" (default), "bf16", "q8_0", "q4_0", "q4_1",
---                                 "q5_0", "q5_1", "iq4_nl".
---                                 Or set opts.kv_type_k / opts.kv_type_v separately.
---   opts.kv_type_k       string?  Override K cache type independently.
---   opts.kv_type_v       string?  Override V cache type independently.
---   opts.swa_full        bool?    Full-size Sliding Window Attention cache (default: false).
---   opts.kv_unified      bool?    Unified KV buffer across sequences (default: false).
--- @return Context
--- @error  If context creation fails.
function Model:context(opts)
    opts = opts or {}
    local kv_type_k = opts.kv_type_k or opts.kv_type or "f16"
    local kv_type_v = opts.kv_type_v or opts.kv_type or "f16"
    local type_k    = KV_TYPES[kv_type_k]
    local type_v    = KV_TYPES[kv_type_v]
    assert(type_k, "[ion7-core] unknown kv_type_k: " .. kv_type_k)
    assert(type_v, "[ion7-core] unknown kv_type_v: " .. kv_type_v)
    local flash_attn  = (opts.flash_attn and 1 or 0)
    local offload_kqv = (opts.offload_kqv ~= false and 1 or 0)

    -- kv quant (q8_0/q4_0) requires flash attention
    -- Enable flash attention automatically when KV cache is quantized
    -- (quantized V cache requires flash attention for correct results)
    if type_k ~= 1 or type_v ~= 1 then flash_attn = 1 end

    -- n_ubatch auto-tuning:
    -- A smaller n_ubatch reduces Time-To-First-Token by processing prompts
    -- in smaller physical chunks. GPU: 512 is a good balance. CPU: 256 or less.
    local n_batch  = opts.n_batch or 2048
    local n_ubatch
    if opts.n_ubatch then
        n_ubatch = opts.n_ubatch
    else
        local use_gpu = (opts.n_gpu_layers or 0) ~= 0 or
                        (type_k ~= 1 or type_v ~= 1)  -- quantized KV implies GPU
        n_ubatch = use_gpu and math.min(512, n_batch) or math.min(256, n_batch)
    end

    local function try_create(n_ctx_, flash_)
        return self._bridge.ion7_context_create(
            self._ptr,
            n_ctx_,
            n_batch,
            n_ubatch,
            opts.n_seq_max       or 1,
            opts.n_threads       or 4,
            opts.n_threads_batch or 0,
            flash_,
            offload_kqv,
            opts.op_offload      and 1 or 0,
            opts.no_perf         and 1 or 0,
            type_k, type_v,
            opts.swa_full        and 1 or 0,
            opts.kv_unified      and 1 or 0
        )
    end

    local n_ctx_ = opts.n_ctx or 4096
    local ptr    = try_create(n_ctx_, flash_attn)

    -- Retry cascade: if VRAM is tight, progressively relax settings
    if ptr == nil and (type_k ~= 1 or type_v ~= 1) then
        -- Try f16 KV (no flash needed) -- saves ~30% VRAM on KV cache
        ptr = try_create(n_ctx_, 0)
    end
    if ptr == nil and n_ctx_ > 4096 then
        -- Try with smaller context
        ptr = try_create(4096, 0)
    end
    if ptr == nil then
        error(string.format(
            "[ion7.core.model] failed to create inference context" ..
            " (n_ctx=%d, kv=%s, flash=%d) -- out of VRAM?",
            n_ctx_, kv_type_k .. "/" .. kv_type_v, flash_attn), 2)
    end
    local ctx = Context.new(self._lib, self._bridge, ptr)
    ctx._n_vocab   = self:vocab():n_vocab()
    ctx._model_ref = self
    return ctx
end

--- Create an embedding context (CPU, pooling enabled, no logits).
---
--- Optimized for embedding models like Qwen3-Embedding.
--- Uses no GPU to avoid VRAM conflicts with the main generation context.
---
--- @param  opts  table?
---   opts.n_ctx      number?  Context size (default: 512).
---   opts.n_threads  number?  CPU threads (default: 4).
---   opts.pooling    string?  "none"|"mean"|"cls"|"last"(default)|"rank".
--- @return Context
--- @error  If context creation fails.
function Model:embedding_context(opts)
    opts = opts or {}
    local pooling_map = { none=-1, mean=1, cls=2, last=3, rank=4 }
    local pooling = pooling_map[opts.pooling or "last"] or 3

    local ptr = self._bridge.ion7_embedding_context_create(
        self._ptr,
        opts.n_ctx     or 512,
        opts.n_seq_max or 1,
        opts.n_threads or 4,
        pooling
    )
    if ptr == nil then
        error("[ion7.core.model] failed to create embedding context", 2)
    end
    local ctx = Context.new(self._lib, self._bridge, ptr)
    ctx._n_vocab   = self:vocab():n_vocab()
    ctx._model_ref = self
    ctx._is_embed  = true
    return ctx
end

-- ── Persistence ───────────────────────────────────────────────────────────────

--- Save the model to a GGUF file.
--- Useful after applying quantization or merging LoRA adapters.
---
--- @param  path  string  Output file path.
function Model:save(path)
    self._bridge.ion7_model_save(self._ptr, path)
end

-- ── LoRA adapters ─────────────────────────────────────────────────────────────

--- Load a LoRA adapter from a GGUF file.
---
--- @param  path   string  Path to the adapter .gguf file.
--- @param  scale  number?  Initial scale (default: 1.0).
--- @return table  { _ptr, path, scale, _bridge }
--- @error  If loading fails.
function Model:load_lora(path, scale)
    local ptr = self._bridge.ion7_lora_load(self._ptr, path)
    if ptr == nil then
        error(string.format(
            "[ion7.core.model] failed to load LoRA from '%s'", path), 2)
    end
    local bridge = self._bridge
    local ffi    = self._ffi
    return {
        _ptr    = ffi.gc(ptr, bridge.ion7_lora_free),
        _bridge = bridge,
        path    = path,
        scale   = scale or 1.0,
        meta    = function(self, key)
            local buf = ffi.new("char[512]")
            local n   = bridge.ion7_lora_meta_val(self._ptr, key, buf, 512)
            return n >= 0 and ffi.string(buf, n) or nil
        end,
    }
end

--- Apply a LoRA adapter to a context.
---
--- @param  ctx      Context  Target context.
--- @param  adapter  table    LoRA handle from load_lora().
--- @param  scale    number?  Scale override (default: adapter.scale).
--- @return bool  true on success.
function Model:apply_lora(ctx, adapter, scale)
    return self._bridge.ion7_lora_apply(
        ctx:ptr(), adapter._ptr, scale or adapter.scale) == 0
end

--- Remove all LoRA adapters from a context.
---
--- @param  ctx      Context  Target context.
--- @param  adapter  table    LoRA handle (used as key, pass any loaded adapter).
--- @return bool  true on success.
function Model:remove_lora(ctx, adapter)
    return self._bridge.ion7_lora_remove(ctx:ptr()) == 0
end

return Model
