--- @module ion7.core.tensor
--- @author  ion7 / Ion7 Project Contributors
---
--- Tensor inspection and bulk readout for use inside `cb_eval` callbacks
--- (or anywhere else a `struct ggml_tensor *` lands in Lua space).
---
--- The four metadata helpers (`name`, `type`, `ne`, `nbytes`) are direct
--- field reads on the cdata pointer — no FFI call cost.
---
--- `copy_f32` reads the raw tensor bytes via `ggml_backend_tensor_get`
--- (which transparently handles CPU and GPU backends), then converts to
--- F32 via `ggml_fp16_to_fp32_row` / `ggml_bf16_to_fp32_row` for the
--- non-F32 sources. Row-wise conversion keeps the per-element overhead
--- to a single C call regardless of tensor size.
---
--- ⚠ Layout caveat for `name` :
---   `ggml_tensor.name` is a fixed-size `char[64]` field, NUL-terminated
---   when the original C string was shorter than 64 bytes. We stop at
---   the first NUL via `ffi.string(ptr)` (no length argument).

local ffi = require "ffi"
require "ion7.core.ffi.types"

local ggml_tensor = require "ion7.core.ffi.ggml.tensor" -- nelements, nbytes
local ggml_backend = require "ion7.core.ffi.ggml.backend" -- backend_tensor_get
local ggml_misc = require "ion7.core.ffi.ggml.misc" -- fp16/bf16 row converters

local ffi_string = ffi.string
local ffi_new = ffi.new
local ffi_sizeof = ffi.sizeof
local tonumber = tonumber

-- Cached enum values for the tensor types we know how to materialise.
-- Hardcoded against the upstream `enum ggml_type` :
--   F32 = 0, F16 = 1, BF16 = 30.
-- If those numerals ever change upstream, the compat layer should
-- re-pin them.
local GGML_TYPE_F32 = 0
local GGML_TYPE_F16 = 1
local GGML_TYPE_BF16 = 30

local M = {}

-- ── Metadata accessors ────────────────────────────────────────────────────

--- Return the tensor's name as a Lua string. Empty string if `t` is `nil`
--- (matches the C bridge's `""` sentinel).
---
--- @param  t cdata `ggml_tensor*`, may be NULL.
--- @return string
function M.name(t)
    if t == nil then
        return ""
    end
    return ffi_string(t.name) -- fixed-size buffer, stops at first NUL.
end

--- Return the tensor's `ggml_type` enum value as a Lua integer, or `-1`
--- if `t` is `nil`.
---
--- @param  t cdata `ggml_tensor*`, may be NULL.
--- @return integer
function M.type(t)
    if t == nil then
        return -1
    end
    return tonumber(t.type)
end

--- Return the size of dimension `dim` (0..3) in elements, or `0` if `t`
--- is `nil` or `dim` is out of range.
---
--- @param  t   cdata   `ggml_tensor*`.
--- @param  dim integer Dimension index in `[0, 3]`.
--- @return integer
function M.ne(t, dim)
    if t == nil or dim < 0 or dim >= 4 then
        return 0
    end
    return tonumber(t.ne[dim])
end

--- Return the total byte size of the tensor data (calls `ggml_nbytes`).
---
--- @param  t cdata `ggml_tensor*`.
--- @return integer  Byte count, or 0 when `t` is `nil`.
function M.nbytes(t)
    if t == nil then
        return 0
    end
    return tonumber(ggml_tensor.ggml_nbytes(t))
end

-- ── Bulk readout ──────────────────────────────────────────────────────────

--- Copy the tensor's data into a caller-provided F32 cdata buffer.
---
--- Handles CPU, CUDA, Metal and Vulkan backends transparently via
--- `ggml_backend_tensor_get`. F16 and BF16 source tensors are
--- up-converted to F32 row-wise. Quantised types (Q4_K, Q5_K, ...) are
--- intentionally NOT supported — callers should dequantise upstream.
---
--- @param  t         cdata    `ggml_tensor*`.
--- @param  dst       cdata    `float*` writable buffer.
--- @param  dst_count integer  Capacity of `dst` in floats.
--- @return integer            Number of floats actually written, or `-1`
---                            on error (NULL inputs, capacity exceeded,
---                            unsupported tensor type).
function M.copy_f32(t, dst, dst_count)
    if t == nil or dst == nil or dst_count == 0 then
        return -1
    end

    local n_elem = tonumber(ggml_tensor.ggml_nelements(t))
    if n_elem <= 0 or n_elem > dst_count then
        return -1
    end

    local kind = tonumber(t.type)

    if kind == GGML_TYPE_F32 then
        -- Direct read into the destination ; no temporary allocation.
        ggml_backend.ggml_backend_tensor_get(t, dst, 0, n_elem * ffi_sizeof("float"))
        return n_elem
    end

    if kind == GGML_TYPE_F16 then
        -- Stage into an f16 buffer, then upconvert in one row-wise call.
        local tmp = ffi_new("ggml_fp16_t[?]", n_elem)
        ggml_backend.ggml_backend_tensor_get(t, tmp, 0, n_elem * ffi_sizeof("ggml_fp16_t"))
        ggml_misc.ggml_fp16_to_fp32_row(tmp, dst, n_elem)
        return n_elem
    end

    if kind == GGML_TYPE_BF16 then
        local tmp = ffi_new("ggml_bf16_t[?]", n_elem)
        ggml_backend.ggml_backend_tensor_get(t, tmp, 0, n_elem * ffi_sizeof("ggml_bf16_t"))
        ggml_misc.ggml_bf16_to_fp32_row(tmp, dst, n_elem)
        return n_elem
    end

    -- Quantised or otherwise unsupported : caller must dequantise upstream.
    return -1
end

return M
