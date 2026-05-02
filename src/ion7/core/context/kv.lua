--- @module ion7.core.context.kv
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Context` : KV cache management.
---
--- All operations route through `llama_memory_*` (the new API that
--- replaced the deprecated `llama_kv_self_*` family). The cached
--- `self._mem` accessor avoids one FFI call per op.
---
--- A note on the `(p0, p1)` window argument convention :
---   - `p0` is INCLUSIVE, `p1` is EXCLUSIVE. Pass `-1` for either to
---     mean "from beginning" / "to end".
---   - Defaults : `p0 = 0`, `p1 = -1` (i.e. the whole sequence).

local llama_kv = require "ion7.core.ffi.llama.kv_cache" -- llama_memory_*

local tonumber = tonumber

local M = {}

-- ── Capability + read access ──────────────────────────────────────────────

--- True if this context's KV cache supports position shifting. Returns
--- `false` for recurrent models (Mamba, RWKV) — they have no positional
--- index to shift.
--- @return boolean
function M.kv_can_shift(self)
    local mem = self._mem
    return mem ~= nil and llama_kv.llama_memory_can_shift(mem) == true
end

--- Smallest KV position currently held by `seq_id`, or `-1` when the
--- sequence is empty / the memory accessor is missing.
--- @param  seq_id integer? Default 0.
--- @return integer
function M.kv_seq_pos_min(self, seq_id)
    local mem = self._mem
    if not mem then
        return -1
    end
    return tonumber(llama_kv.llama_memory_seq_pos_min(mem, seq_id or 0))
end

--- Largest KV position currently held by `seq_id`, or `-1` for empty
--- sequences.
--- @param  seq_id integer? Default 0.
--- @return integer
function M.kv_seq_pos_max(self, seq_id)
    local mem = self._mem
    if not mem then
        return -1
    end
    return tonumber(llama_kv.llama_memory_seq_pos_max(mem, seq_id or 0))
end

-- ── Mutating operations ───────────────────────────────────────────────────

--- Wipe the entire KV cache and reset our Lua-tracked `n_past` mirror
--- to 0. This is the brutal "start over" reset used after big context
--- changes (e.g. switching prompt entirely).
function M.kv_clear(self)
    local mem = self._mem
    if mem then
        llama_kv.llama_memory_clear(mem, true)
    end
    self._n_past = 0
end

--- Drop KV entries for `seq_id` in `[p0, p1)`.
--- @param  seq_id integer
--- @param  p0     integer Inclusive start.
--- @param  p1     integer Exclusive end (`-1` for "to end").
--- @return boolean true if the request fully removed the slice.
function M.kv_seq_rm(self, seq_id, p0, p1)
    local mem = self._mem
    if not mem then
        return false
    end
    return llama_kv.llama_memory_seq_rm(mem, seq_id, p0, p1 or -1) == true
end

--- Copy KV entries from `src_seq` to `dst_seq` in `[p0, p1)`. Used to
--- fork sequences for parallel decoding or beam search.
--- @param  src_seq integer
--- @param  dst_seq integer
--- @param  p0      integer
--- @param  p1      integer
function M.kv_seq_cp(self, src_seq, dst_seq, p0, p1)
    local mem = self._mem
    if mem then
        llama_kv.llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1 or -1)
    end
end

--- Drop EVERY sequence in the KV cache except `seq_id`.
--- @param  seq_id integer
function M.kv_seq_keep(self, seq_id)
    local mem = self._mem
    if mem then
        llama_kv.llama_memory_seq_keep(mem, seq_id)
    end
end

--- Add `delta` to every position in `[p0, p1)` for `seq_id`. Positive
--- shifts the window forward, negative drops it back. Only valid when
--- `kv_can_shift()` returns true.
---
--- @param  seq_id integer
--- @param  delta  integer Shift amount.
--- @param  p0     integer? Inclusive start (default 0).
--- @param  p1     integer? Exclusive end   (default -1).
function M.kv_seq_shift(self, seq_id, delta, p0, p1)
    local mem = self._mem
    if mem then
        llama_kv.llama_memory_seq_add(mem, seq_id, p0 or 0, p1 or -1, delta)
    end
end

--- Divide every position in `[p0, p1)` for `seq_id` by `d`. Used for
--- context compression (group multiple positions together to fit a
--- longer effective prompt in the same window).
---
--- @param  seq_id integer
--- @param  d      integer Divisor (must be > 1).
--- @param  p0     integer? Inclusive start (default 0).
--- @param  p1     integer? Exclusive end   (default -1).
function M.kv_seq_div(self, seq_id, d, p0, p1)
    local mem = self._mem
    if mem then
        llama_kv.llama_memory_seq_div(mem, seq_id, p0 or 0, p1 or -1, d)
    end
end

return M
