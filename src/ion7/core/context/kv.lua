--- @module ion7.core.context.kv
--- SPDX-License-Identifier: MIT
--- KV cache and memory operations.
--- All functions receive the Context instance as first argument.

local M = {}

--- Check whether the KV cache supports position shifting.
--- Returns false for recurrent models (Mamba, RWKV).
--- @return bool
function M.kv_can_shift(self)
    local mem = self._mem
    return mem ~= nil and self._lib.llama_memory_can_shift(mem)
end

--- Return the smallest KV position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return number
function M.kv_seq_pos_min(self, seq_id)
    local mem = self._mem
    if not mem then return -1 end
    return tonumber(self._lib.llama_memory_seq_pos_min(mem, seq_id or 0))
end

--- Return the largest KV position for a sequence (-1 if empty).
--- @param  seq_id  number?  Sequence ID (default 0).
--- @return number
function M.kv_seq_pos_max(self, seq_id)
    local mem = self._mem
    if not mem then return -1 end
    return tonumber(self._lib.llama_memory_seq_pos_max(mem, seq_id or 0))
end

--- Divide all positions in [p0, p1) for seq_id by d (context compression).
--- @param  seq_id  number
--- @param  d       number  Divisor > 1.
--- @param  p0      number? Start (default 0).
--- @param  p1      number? End exclusive (default -1 = all).
function M.kv_seq_div(self, seq_id, d, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_div(mem, seq_id, p0 or 0, p1 or -1, d) end
end

--- Clear the entire KV cache and reset n_past to 0.
function M.kv_clear(self)
    local mem = self._mem
    if mem then self._lib.llama_memory_clear(mem, true) end
    self._n_past = 0
end

--- Remove KV entries for seq_id in [p0, p1). p1 = -1 means to end.
--- @param  seq_id  number
--- @param  p0      number
--- @param  p1      number  -1 = end.
--- @return bool  true if the sequence was fully removed.
function M.kv_seq_rm(self, seq_id, p0, p1)
    local mem = self._mem
    if not mem then return false end
    return self._lib.llama_memory_seq_rm(mem, seq_id, p0, p1 or -1)
end

--- Copy KV entries from src to dst for positions in [p0, p1).
--- @param  src_seq  number
--- @param  dst_seq  number
--- @param  p0       number
--- @param  p1       number  -1 = end.
function M.kv_seq_cp(self, src_seq, dst_seq, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1 or -1) end
end

--- Keep only seq_id; discard all other sequences from the KV cache.
--- @param  seq_id  number
function M.kv_seq_keep(self, seq_id)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_keep(mem, seq_id) end
end

--- Add delta to all positions in [p0, p1) for seq_id (sliding window shift).
--- Only valid when kv_can_shift() returns true.
--- @param  seq_id  number
--- @param  delta   number  Shift amount (negative moves positions backward).
--- @param  p0      number? Start (default 0).
--- @param  p1      number? End exclusive (default -1 = all).
function M.kv_seq_shift(self, seq_id, delta, p0, p1)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_add(mem, seq_id, p0 or 0, p1 or -1, delta) end
end

--- Add delta to positions in [p0, p1) for seq_id (canonical name for kv_seq_shift).
--- @param  seq_id  number
--- @param  p0      number
--- @param  p1      number
--- @param  delta   number
function M.memory_seq_add(self, seq_id, p0, p1, delta)
    local mem = self._mem
    if mem then self._lib.llama_memory_seq_add(mem, seq_id, p0 or 0, p1 or -1, delta) end
end

return M
