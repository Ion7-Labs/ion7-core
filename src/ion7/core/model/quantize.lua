--- @module ion7.core.model.quantize
--- @author  ion7 / Ion7 Project Contributors
---
--- Static model quantisation. `quantize` is a class-method (no Model
--- instance needed) — it operates on file paths only — so we expose it
--- as `Model.quantize` rather than `Model:quantize`.
---
--- The `ftype` map is the canonical name → numeric translation that
--- mirrors `enum llama_ftype` in `llama.h`. Names are lowercase with
--- underscores ; the function normalises caller input by lowercasing
--- and `-` → `_` before lookup so callers can pass `"Q4_K_M"` or
--- `"q4-k-m"` interchangeably.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_model = require "ion7.core.ffi.llama.model" -- model_quantize_*

local tonumber = tonumber

-- Canonical name → `enum llama_ftype` value. Keep this aligned with
-- llama.h ; the upstream enum changes infrequently but DOES change
-- (most recently when MoE / FP4 quantisations were added).
local FTYPE_MAP = {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q8_0 = 7,
    q5_0 = 8,
    q5_1 = 9,
    q2_k = 10,
    q3_k_s = 11,
    q3_k_m = 12,
    q3_k_l = 13,
    q4_k_s = 14,
    q4_k_m = 15,
    q5_k_s = 16,
    q5_k_m = 17,
    q6_k = 18,
    iq2_xxs = 19,
    iq2_xs = 20,
    q2_k_s = 21,
    iq3_xs = 22,
    iq3_xxs = 23,
    iq1_s = 24,
    iq4_nl = 25,
    iq3_s = 26,
    iq3_m = 27,
    iq2_s = 28,
    iq2_m = 29,
    iq4_xs = 30,
    iq1_m = 31,
    bf16 = 32,
    tq1_0 = 36,
    tq2_0 = 37,
    mxfp4_moe = 38,
    nvfp4 = 39,
    q1_0 = 40,
    copy = -1
}

local M = {}

--- Quantise a GGUF model to a new file.
---
--- @param  inp_path string Path to source GGUF.
--- @param  out_path string Path to write the quantised GGUF.
--- @param  opts     table?
---   `ftype`              (string, default `"q4_k_m"`) — see `FTYPE_MAP`.
---   `nthread`            (integer, default 0 = hardware concurrency).
---   `allow_requantize`   (bool)   — re-quantise already-quantised tensors.
---   `quantize_output`    (bool)   — also quantise the output / lm_head tensor.
---   `pure`               (bool)   — quantise EVERY tensor (no 1d exceptions).
---   `keep_split`         (bool)   — preserve the source's shard layout.
---   `dry_run`            (bool)   — compute the plan without writing.
--- @return integer 0 on success, non-zero llama_ftype error code otherwise.
function M.quantize(inp_path, out_path, opts)
    assert(type(inp_path) == "string", "[ion7.core.model] inp_path must be a string")
    assert(type(out_path) == "string", "[ion7.core.model] out_path must be a string")
    opts = opts or {}

    local ftype_key = (opts.ftype or "q4_k_m"):lower():gsub("-", "_")
    local ftype_val = FTYPE_MAP[ftype_key]
    assert(ftype_val ~= nil, string.format("[ion7.core.model] unknown ftype '%s'", opts.ftype or "q4_k_m"))

    -- llama_model_quantize_default_params returns the struct by value ;
    -- we mutate its fields then pass a pointer. The default param block
    -- already initialises sensible boolean defaults (false), so we only
    -- override the ones the caller explicitly sets.
    local params = ffi.new("struct llama_model_quantize_params")
    -- Initialise from the default by copying the value-returned struct.
    -- (LuaJIT FFI assignment between same-type structs does memcpy.)
    ffi.copy(
        params,
        llama_model.llama_model_quantize_default_params(),
        ffi.sizeof("struct llama_model_quantize_params")
    )

    params.ftype = ftype_val
    params.nthread = opts.nthread or 0
    if opts.allow_requantize ~= nil then
        params.allow_requantize = opts.allow_requantize
    end
    if opts.quantize_output ~= nil then
        params.quantize_output_tensor = opts.quantize_output
    end
    if opts.pure ~= nil then
        params.pure = opts.pure
    end
    if opts.keep_split ~= nil then
        params.keep_split = opts.keep_split
    end
    if opts.dry_run ~= nil then
        params.dry_run = opts.dry_run
    end

    return tonumber(llama_model.llama_model_quantize(inp_path, out_path, params))
end

return M
