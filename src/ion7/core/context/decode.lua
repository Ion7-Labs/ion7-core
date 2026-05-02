--- @module ion7.core.context.decode
--- @author  ion7 / Ion7 Project Contributors
---
--- Mixin for `Context` : token decode and encode operations.
---
--- All four entry points reuse the Context's pre-allocated `_decode_batch`
--- — zero malloc per call, zero malloc per token in the inner generation
--- loop. The `decode` variant accepts both Lua tables (1-based, the
--- typical shape of `vocab:tokenize` output) and raw cdata int32 arrays
--- (0-based, what you get back from a custom prompt builder). Detection
--- happens once via `type(tokens)`.
---
--- Logit policy :
---
---   - `decode`        : only the LAST token of the LAST chunk has logits
---                       enabled (the typical "process this prompt, give
---                       me the next-token logits" use case).
---   - `decode_single` : that one token has logits — fast path.
---   - `decode_multi`  : EVERY position has logits enabled (used for
---                       speculative-decoding verification, where we
---                       need to re-evaluate N drafts in one pass).
---   - `encode`        : every position has logits enabled (T5-style
---                       encoder pass).

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_context = require "ion7.core.ffi.llama.context" -- llama_decode, llama_encode

local ffi_fill = ffi.fill

local M = {}

-- Internal helper : raise the right error for a non-zero llama_decode rc.
-- llama.cpp's convention is `0 = ok`, `1 = KV full`, others = generic.
local function decode_failed(op, rc)
    if rc == 1 then
        error("[ion7.core.context] " .. op .. ": KV cache is full", 3)
    end
    error(string.format("[ion7.core.context] %s returned %d", op, rc), 3)
end

-- ── decode (the main entry point — handles chunking + table/cdata) ────────

--- Decode `n_tokens` tokens, updating the KV cache and producing logits
--- for the LAST token only.
---
--- The function chunks automatically when `n_tokens > n_batch` so the
--- caller can pass an arbitrarily long prompt. The returned value is the
--- size of the LAST chunk — that's the upper bound for `idx` arguments
--- to `Sampler:sample(ctx:ptr(), idx)` afterwards.
---
--- @param  tokens     cdata|table Token IDs. Lua tables are 1-based ;
---                                cdata arrays are 0-based.
--- @param  n_tokens   integer?    Required for cdata, auto for tables.
--- @param  seq_id     integer?    Sequence ID (default 0).
--- @param  pos_offset integer?    Starting KV position (default `n_past`).
--- @return integer                Last chunk size.
--- @raise   On KV-full or any other `llama_decode` error.
function M.decode(self, tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens = n_tokens or (is_table and #tokens or 0)
    seq_id = seq_id or 0
    pos_offset = pos_offset or self._n_past

    local n_batch = self._n_batch
    local b = self._decode_batch
    local tok_ptr = b.token
    local pos_ptr = b.pos
    local seq_ptr = b.seq_id
    local log_ptr = b.logits
    local done = 0
    local last_n = 0

    -- Two near-identical loop bodies because the inner index translation
    -- (`tokens[done + i + 1]` for table vs `tokens[done + i]` for cdata)
    -- is the only thing that differs. Keeping them separate lets the JIT
    -- specialise each path without branch checks per iteration.
    if is_table then
        while done < n_tokens do
            local n = n_tokens - done
            if n > n_batch then
                n = n_batch
            end
            local end_i = n - 1
            local base = pos_offset + done
            ffi_fill(log_ptr, n)
            for i = 0, end_i do
                tok_ptr[i] = tokens[done + i + 1] -- 1-based table
                pos_ptr[i] = base + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then
                log_ptr[end_i] = 1
            end
            b.n_tokens = n
            local rc = llama_context.llama_decode(self._ptr, b)
            if rc ~= 0 then
                decode_failed("decode", rc)
            end
            done = done + n
            last_n = n
        end
    else
        while done < n_tokens do
            local n = n_tokens - done
            if n > n_batch then
                n = n_batch
            end
            local end_i = n - 1
            local base = pos_offset + done
            ffi_fill(log_ptr, n)
            for i = 0, end_i do
                tok_ptr[i] = tokens[done + i] -- 0-based cdata
                pos_ptr[i] = base + i
                seq_ptr[i][0] = seq_id
            end
            if done + n >= n_tokens then
                log_ptr[end_i] = 1
            end
            b.n_tokens = n
            local rc = llama_context.llama_decode(self._ptr, b)
            if rc ~= 0 then
                decode_failed("decode", rc)
            end
            done = done + n
            last_n = n
        end
    end

    self._n_past = pos_offset + n_tokens
    return last_n
end

-- ── decode_single (fast path for the per-token generation loop) ───────────

--- Decode ONE token. Logits enabled for it. Skips the chunking loop and
--- the table-vs-cdata branch — the JIT loves this path.
---
--- @param  token  integer Token ID to decode.
--- @param  seq_id integer? Sequence ID (default 0).
--- @raise   On any `llama_decode` error.
function M.decode_single(self, token, seq_id)
    local b = self._decode_batch
    b.token[0] = token
    b.pos[0] = self._n_past
    b.seq_id[0][0] = seq_id or 0
    b.logits[0] = 1
    b.n_tokens = 1
    local rc = llama_context.llama_decode(self._ptr, b)
    if rc ~= 0 then
        decode_failed("decode_single", rc)
    end
    self._n_past = self._n_past + 1
end

-- ── decode_multi (every position has logits — for speculative verify) ─────

--- Decode `#tokens` tokens with logits enabled for EVERY position. After
--- this call, `Sampler:sample(ctx:ptr(), i)` is valid for `i` in
--- `0 .. #tokens - 1`. Used by speculative decoding to verify all draft
--- positions in a single forward pass.
---
--- @param  tokens table   1-based Lua table of token IDs.
--- @param  seq_id integer? Sequence ID (default 0).
--- @raise   When `#tokens` exceeds the batch capacity, or on decode error.
function M.decode_multi(self, tokens, seq_id)
    local n = #tokens
    assert(n > 0, "[ion7.core.context] decode_multi: empty token list")
    assert(
        n <= self._n_batch,
        string.format("[ion7.core.context] decode_multi: %d tokens exceeds batch capacity %d", n, self._n_batch)
    )

    seq_id = seq_id or 0
    local b = self._decode_batch
    local base = self._n_past
    for i = 0, n - 1 do
        b.token[i] = tokens[i + 1]
        b.pos[i] = base + i
        b.seq_id[i][0] = seq_id
        b.logits[i] = 1
    end
    b.n_tokens = n

    local rc = llama_context.llama_decode(self._ptr, b)
    if rc ~= 0 then
        decode_failed("decode_multi", rc)
    end
    self._n_past = base + n
end

-- ── encode (T5-style encoder pass) ────────────────────────────────────────

--- Encode `n_tokens` tokens through the model's encoder stack. Used by
--- encoder-decoder architectures (T5, BART, ...). Like `decode`, handles
--- chunking and both table / cdata token layouts.
---
--- @param  tokens     cdata|table
--- @param  n_tokens   integer?
--- @param  seq_id     integer?
--- @param  pos_offset integer?
--- @raise   On any `llama_encode` error.
function M.encode(self, tokens, n_tokens, seq_id, pos_offset)
    local is_table = type(tokens) == "table"
    n_tokens = n_tokens or (is_table and #tokens or 0)
    seq_id = seq_id or 0
    pos_offset = pos_offset or self._n_past

    local n_batch = self._n_batch
    local b = self._decode_batch
    local done = 0

    if is_table then
        while done < n_tokens do
            local n = n_tokens - done
            if n > n_batch then
                n = n_batch
            end
            local base = pos_offset + done
            for i = 0, n - 1 do
                b.token[i] = tokens[done + i + 1]
                b.pos[i] = base + i
                b.seq_id[i][0] = seq_id
                b.logits[i] = 1
            end
            b.n_tokens = n
            local rc = llama_context.llama_encode(self._ptr, b)
            if rc ~= 0 then
                decode_failed("encode", rc)
            end
            done = done + n
        end
    else
        while done < n_tokens do
            local n = n_tokens - done
            if n > n_batch then
                n = n_batch
            end
            local base = pos_offset + done
            for i = 0, n - 1 do
                b.token[i] = tokens[done + i]
                b.pos[i] = base + i
                b.seq_id[i][0] = seq_id
                b.logits[i] = 1
            end
            b.n_tokens = n
            local rc = llama_context.llama_encode(self._ptr, b)
            if rc ~= 0 then
                decode_failed("encode", rc)
            end
            done = done + n
        end
    end

    self._n_past = pos_offset + n_tokens
end

return M
