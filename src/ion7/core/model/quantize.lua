--- @module ion7.core.model.quantize
--- SPDX-License-Identifier: MIT
--- Static model quantization (Model.quantize).

local Loader = require "ion7.core.ffi.loader"

-- ftype name → llama_ftype int value.
-- Keys are normalised to lowercase with underscores.
local ftype_map = {
    f32        =  0,   f16        =  1,   q4_0       =  2,   q4_1       =  3,
    q8_0       =  7,   q5_0       =  8,   q5_1       =  9,   q2_k       = 10,
    q3_k_s     = 11,   q3_k_m     = 12,   q3_k_l     = 13,   q4_k_s     = 14,
    q4_k_m     = 15,   q5_k_s     = 16,   q5_k_m     = 17,   q6_k       = 18,
    iq2_xxs    = 19,   iq2_xs     = 20,   q2_k_s     = 21,   iq3_xs     = 22,
    iq3_xxs    = 23,   iq1_s      = 24,   iq4_nl     = 25,   iq3_s      = 26,
    iq3_m      = 27,   iq2_s      = 28,   iq2_m      = 29,   iq4_xs     = 30,
    iq1_m      = 31,   bf16       = 32,   tq1_0      = 36,   tq2_0      = 37,
    mxfp4_moe  = 38,   nvfp4      = 39,   q1_0       = 40,   copy       = -1,
}

local M = {}

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
function M.quantize(inp_path, out_path, opts)
    assert(type(inp_path) == "string", "[ion7.core.model] inp_path must be a string")
    assert(type(out_path) == "string", "[ion7.core.model] out_path must be a string")
    opts = opts or {}

    local L         = Loader.instance()
    local ftype_key = (opts.ftype or "q4_k_m"):lower():gsub("-", "_")
    local ftype_val = ftype_map[ftype_key]
    assert(ftype_val ~= nil, string.format(
        "[ion7.core.model] unknown ftype '%s'", opts.ftype or "q4_k_m"))

    local params = L.lib.llama_model_quantize_default_params()
    params.nthread                = opts.nthread         or 0
    params.ftype                  = ftype_val
    params.allow_requantize       = opts.allow_requantize  and true or false
    params.quantize_output_tensor = opts.quantize_output   and true or false
    params.pure                   = opts.pure              and true or false
    params.keep_split             = opts.keep_split        and true or false
    params.dry_run                = opts.dry_run           and true or false

    return tonumber(L.bridge.ion7_model_quantize(
        inp_path, out_path,
        params.ftype,
        params.nthread,
        params.pure                   and 1 or 0,
        params.allow_requantize       and 1 or 0,
        params.quantize_output_tensor and 1 or 0,
        params.dry_run                and 1 or 0
    ))
end

return M
