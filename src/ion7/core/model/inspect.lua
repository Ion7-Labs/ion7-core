--- @module ion7.core.model.inspect
--- SPDX-License-Identifier: MIT
--- Model property getters and info summary.
--- All functions receive the Model instance as first argument (standard Lua OOP).

local M = {}

--- @return string  e.g. "Qwen3 27B Q4_K_M"
function M.desc(self)
    local buf = self._ffi.new("char[512]")
    local n   = self._bridge.ion7_model_desc(self._ptr, buf, 512)
    return n > 0 and self._ffi.string(buf, n) or "(unknown)"
end

--- Total size of all model tensors in bytes.
--- @return number
function M.size(self)
    return tonumber(self._ffi.cast("double", self._bridge.ion7_model_size(self._ptr)))
end

--- Total number of parameters.
--- @return number
function M.n_params(self)
    return tonumber(self._ffi.cast("double", self._bridge.ion7_model_n_params(self._ptr)))
end

--- Training context size.
--- @return number
function M.n_ctx_train(self)
    return tonumber(self._bridge.ion7_model_n_ctx_train(self._ptr))
end

--- Hidden embedding dimension.
--- @return number
function M.n_embd(self)
    return tonumber(self._bridge.ion7_model_n_embd(self._ptr))
end

--- Input embedding dimension.
--- @return number
function M.n_embd_inp(self)
    return tonumber(self._bridge.ion7_model_n_embd_inp(self._ptr))
end

--- Output embedding dimension.
--- @return number
function M.n_embd_out(self)
    return tonumber(self._bridge.ion7_model_n_embd_out(self._ptr))
end

--- Number of transformer layers.
--- @return number
function M.n_layer(self)
    return tonumber(self._bridge.ion7_model_n_layer(self._ptr))
end

--- Number of attention heads.
--- @return number
function M.n_head(self)
    return tonumber(self._bridge.ion7_model_n_head(self._ptr))
end

--- Number of KV heads (GQA).
--- @return number
function M.n_head_kv(self)
    return tonumber(self._bridge.ion7_model_n_head_kv(self._ptr))
end

--- Sliding window attention size.
--- @return number
function M.n_swa(self)
    return tonumber(self._bridge.ion7_model_n_swa(self._ptr))
end

--- Returns true if encoder-decoder model.
--- @return bool
function M.has_encoder(self)
    return self._bridge.ion7_model_has_encoder(self._ptr) == 1
end

--- Returns true for decoder models (virtually all LLMs).
--- @return bool
function M.has_decoder(self)
    return self._bridge.ion7_model_has_decoder(self._ptr) == 1
end

--- Returns true for recurrent models (Mamba, RWKV).
--- @return bool
function M.is_recurrent(self)
    return self._bridge.ion7_model_is_recurrent(self._ptr) == 1
end

--- Returns true for hybrid attention+SSM models (Jamba).
--- @return bool
function M.is_hybrid(self)
    return self._bridge.ion7_model_is_hybrid(self._ptr) == 1
end

--- Returns true for diffusion-based models (LLaDA).
--- @return bool
function M.is_diffusion(self)
    return self._bridge.ion7_model_is_diffusion(self._ptr) == 1
end

--- RoPE frequency scale from training.
--- @return number
function M.rope_freq_scale_train(self)
    return tonumber(self._lib.llama_model_rope_freq_scale_train(self._ptr))
end

--- Number of classifier output classes.
--- @return number
function M.n_cls_out(self)
    return tonumber(self._lib.llama_model_n_cls_out(self._ptr))
end

--- Label of classifier output class at index i.
--- @param  i  number  0-based index.
--- @return string?
function M.cls_label(self, i)
    local p = self._lib.llama_model_cls_label(self._ptr, i)
    return p ~= nil and self._ffi.string(p) or nil
end

--- RoPE type: "none" | "norm" | "neox" | "mrope" | "imrope" | "vision"
--- @return string
function M.rope_type(self)
    local p = self._bridge.ion7_model_rope_type(self._ptr)
    return (p ~= nil) and self._ffi.string(p) or "unknown"
end

--- Decoder start token for encoder-decoder models (-1 for decoder-only).
--- @return number
function M.decoder_start_token(self)
    return tonumber(self._lib.llama_model_decoder_start_token(self._ptr))
end

--- Return a summary table of model properties.
--- @return table
function M.info(self)
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

return M
