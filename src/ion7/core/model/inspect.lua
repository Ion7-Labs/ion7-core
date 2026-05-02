--- @module ion7.core.model.inspect
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Model` : property getters and a one-shot `info()` summary.
---
--- Functions take `self` (a Model instance) as their first argument so
--- they can be spliced directly into `Model`'s metatable from
--- `model.lua`. They never reach into the bridge — every call goes
--- through the auto-generated FFI bindings under `ion7.core.ffi.llama.*`.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_model = require "ion7.core.ffi.llama.model" -- llama_model_*

local ffi_new = ffi.new
local ffi_string = ffi.string
local tonumber = tonumber

-- LLAMA_ROPE_TYPE_* numeric values → human-readable string. Pulled by
-- numeric value from `llama.h` so a future llama.cpp renaming of one of
-- the enum members surfaces here as a missing key (default = "unknown").
local ROPE_TYPE_NAMES = {
    [-1] = "none",
    [0] = "norm",
    [2] = "neox",
    [8] = "mrope",
    [40] = "imrope",
    [24] = "vision"
}

local M = {}

-- ── Identity / size ───────────────────────────────────────────────────────

--- Human-readable description string, e.g. `"Qwen3 32B Q4_K_M"`.
--- @return string
function M.desc(self)
    local buf = ffi_new("char[512]")
    local n = llama_model.llama_model_desc(self._ptr, buf, 512)
    return n > 0 and ffi_string(buf, n) or "(unknown)"
end

--- Total byte size of the model's tensors on disk.
--- We cast through `double` so the value survives the `tonumber` cast
--- when it exceeds 2^31 (any modern 7B+ model qualifies).
--- @return number
function M.size(self)
    return tonumber(ffi.cast("double", llama_model.llama_model_size(self._ptr)))
end

--- Total number of trained parameters (weights + biases).
--- @return number
function M.n_params(self)
    return tonumber(ffi.cast("double", llama_model.llama_model_n_params(self._ptr)))
end

-- ── Architecture topology ─────────────────────────────────────────────────

--- Training-time context length declared by the GGUF header.
--- @return integer
function M.n_ctx_train(self)
    return tonumber(llama_model.llama_model_n_ctx_train(self._ptr))
end

--- Hidden embedding dimension used internally by the transformer stack.
--- @return integer
function M.n_embd(self)
    return tonumber(llama_model.llama_model_n_embd(self._ptr))
end

--- Input embedding dimension (matches `n_embd` for most models, may
--- differ for projection-heavy designs).
--- @return integer
function M.n_embd_inp(self)
    return tonumber(llama_model.llama_model_n_embd_inp(self._ptr))
end

--- Output embedding dimension. Differs from `n_embd_inp` for
--- classifier-style models with a projection head.
--- @return integer
function M.n_embd_out(self)
    return tonumber(llama_model.llama_model_n_embd_out(self._ptr))
end

--- Number of transformer blocks.
--- @return integer
function M.n_layer(self)
    return tonumber(llama_model.llama_model_n_layer(self._ptr))
end

--- Number of attention heads.
--- @return integer
function M.n_head(self)
    return tonumber(llama_model.llama_model_n_head(self._ptr))
end

--- Number of KV heads. Lower than `n_head` for GQA / MQA models.
--- @return integer
function M.n_head_kv(self)
    return tonumber(llama_model.llama_model_n_head_kv(self._ptr))
end

--- Sliding-window attention size. `0` for full-context models.
--- @return integer
function M.n_swa(self)
    return tonumber(llama_model.llama_model_n_swa(self._ptr))
end

-- ── Capability flags ──────────────────────────────────────────────────────

--- True for encoder-decoder architectures (T5-like).
--- @return boolean
function M.has_encoder(self)
    return llama_model.llama_model_has_encoder(self._ptr) == true
end

--- True for decoder-bearing models — virtually all LLMs.
--- @return boolean
function M.has_decoder(self)
    return llama_model.llama_model_has_decoder(self._ptr) == true
end

--- True for state-space / RNN-like models (Mamba, RWKV).
--- @return boolean
function M.is_recurrent(self)
    return llama_model.llama_model_is_recurrent(self._ptr) == true
end

--- True for hybrid attention + SSM models (Jamba).
--- @return boolean
function M.is_hybrid(self)
    return llama_model.llama_model_is_hybrid(self._ptr) == true
end

--- True for diffusion-based models (LLaDA et al.).
--- @return boolean
function M.is_diffusion(self)
    return llama_model.llama_model_is_diffusion(self._ptr) == true
end

-- ── Position encoding ─────────────────────────────────────────────────────

--- Training-time RoPE frequency scale.
--- @return number
function M.rope_freq_scale_train(self)
    return tonumber(llama_model.llama_model_rope_freq_scale_train(self._ptr))
end

--- Symbolic name of the RoPE scheme :
---   `"none" | "norm" | "neox" | "mrope" | "imrope" | "vision" | "unknown"`.
--- @return string
function M.rope_type(self)
    local v = tonumber(llama_model.llama_model_rope_type(self._ptr))
    return ROPE_TYPE_NAMES[v] or "unknown"
end

-- ── Classifier head (only meaningful for classifier-style models) ─────────

--- Number of classifier output classes ; 0 for non-classifier models.
--- @return integer
function M.n_cls_out(self)
    return tonumber(llama_model.llama_model_n_cls_out(self._ptr))
end

--- Label string of the classifier output at index `i` (0-based), or nil
--- if `i` exceeds `n_cls_out`.
--- @param  i integer 0-based index.
--- @return string|nil
function M.cls_label(self, i)
    local p = llama_model.llama_model_cls_label(self._ptr, i)
    return p ~= nil and ffi_string(p) or nil
end

--- Decoder start token id for encoder-decoder models, `-1` for plain
--- decoders.
--- @return integer
function M.decoder_start_token(self)
    return tonumber(llama_model.llama_model_decoder_start_token(self._ptr))
end

-- ── Aggregate summary ─────────────────────────────────────────────────────

--- Return a flat table snapshot of every Model property in one call.
--- Convenient for logging or sending to `print` / dkjson.
--- @return table
function M.info(self)
    local v = self:vocab() -- depends on context_factory mixin
    local sz = self:size()
    return {
        path = self.path,
        desc = self:desc(),
        n_params = self:n_params(),
        size = sz,
        size_gb = sz / (1024 ^ 3),
        n_ctx_train = self:n_ctx_train(),
        n_embd = self:n_embd(),
        n_embd_inp = self:n_embd_inp(),
        n_embd_out = self:n_embd_out(),
        n_layer = self:n_layer(),
        n_head = self:n_head(),
        n_head_kv = self:n_head_kv(),
        n_swa = self:n_swa(),
        n_vocab = v:n_vocab(),
        vocab_type = v:type(),
        rope_type = self:rope_type(),
        has_encoder = self:has_encoder(),
        has_decoder = self:has_decoder(),
        is_recurrent = self:is_recurrent(),
        is_hybrid = self:is_hybrid(),
        is_diffusion = self:is_diffusion()
    }
end

return M
