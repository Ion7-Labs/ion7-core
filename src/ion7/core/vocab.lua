--- @module ion7.core.vocab
--- SPDX-License-Identifier: AGPL-3.0-or-later
--- Vocabulary operations: tokenisation, detokenisation, chat templates.
---
--- A Vocab wraps a `const llama_vocab*` obtained from a loaded model.
--- It is a lightweight, allocation-free handle -- no C objects are owned.
---
--- All string conversions use a shared 4KB internal buffer for performance.
--- This buffer is not thread-safe; use one Vocab per thread.
---
--- @usage
---   local vocab = model:vocab()
---   local tokens, n = vocab:tokenize("Hello, world!", true, true)
---   local text = vocab:detokenize(tokens, n)
---   print(text)  --> "Hello, world!"

local Loader = require "ion7.core.ffi.loader"

-- ── Internal buffer for token-to-piece conversion ─────────────────────────────

local PIECE_BUF_SIZE  = 256
local TMPL_BUF_SIZE   = 131072  -- 128 KB for chat template output
local DETOK_BUF_SIZE  = 65536   -- 64  KB for detokenize output

-- ── Vocab ─────────────────────────────────────────────────────────────────────

--- @class Vocab
--- @field _ptr  cdata   Opaque llama_vocab* (not owned, do not free).
--- @field _lib  cdata   libllama.so namespace.
--- @field _model cdata  llama_model* for chat template calls.
local Vocab = {}
Vocab.__index = Vocab

--- Create a Vocab from a raw llama_vocab pointer.
--- Prefer model:vocab() over calling this directly.
---
--- @param  lib    cdata  libllama.so FFI namespace.
--- @param  model  cdata  llama_model* (needed for chat template).
--- @param  ptr    cdata  llama_vocab* from llama_model_get_vocab().
--- @return Vocab
function Vocab.new(lib, model, ptr)
    assert(ptr ~= nil, "[ion7.core.vocab] vocab pointer is NULL")
    local L   = Loader.instance()
    local ffi = L.ffi
    return setmetatable({
        _ptr    = ptr,
        _lib    = lib,
        _bridge = L.bridge,
        _model  = model,
        _ffi    = ffi,
        -- Pre-allocated scratch buffers
        _piece_buf = ffi.new("char[?]", PIECE_BUF_SIZE),
        _tmpl_buf  = ffi.new("char[?]", TMPL_BUF_SIZE),
        _dtok_buf  = ffi.new("char[?]", DETOK_BUF_SIZE),
    }, Vocab)
end

--- @return number  Total number of tokens in the vocabulary.
function Vocab:n_vocab()
    return tonumber(self._lib.llama_vocab_n_tokens(self._ptr))
end

--- @return string  Vocabulary type: "spm", "bpe", "wpm", "ugm", "rwkv", or "none".
function Vocab:type()
    local t = self._lib.llama_vocab_type(self._ptr)
    local types = { [0]="none", [1]="spm", [2]="bpe", [3]="wpm", [4]="ugm", [5]="rwkv" }
    return types[tonumber(t)] or "unknown"
end

--- Tokenize a text string.
---
--- @param  text          string   UTF-8 text to tokenize.
--- @param  add_special   bool?    Add BOS/EOS tokens (default: false).
--- @param  parse_special bool?    Parse special token text (default: true).
--- @return cdata, number  int32_t array and token count.
--- @error  If tokenization fails.
function Vocab:tokenize(text, add_special, parse_special)
    if add_special   == nil then add_special   = false end
    if parse_special == nil then parse_special = true  end

    -- Estimate worst case: 1 token per byte + 16 for special tokens
    local max_tokens = math.max(#text + 16, 64)
    local buf        = self._ffi.new("int32_t[?]", max_tokens)
    local n          = self._lib.llama_tokenize(
        self._ptr, text, #text,
        buf, max_tokens,
        add_special, parse_special
    )
    if n < 0 then
        -- Buffer was too small; reallocate and retry
        max_tokens = -n + 16
        buf        = self._ffi.new("int32_t[?]", max_tokens)
        n          = self._lib.llama_tokenize(
            self._ptr, text, #text,
            buf, max_tokens,
            add_special, parse_special
        )
        if n < 0 then
            error(string.format(
                "[ion7.core.vocab] tokenize failed (n=%d)", n), 2)
        end
    end
    return buf, tonumber(n)
end

--- Detokenize a sequence of tokens back into text.
---
--- @param  tokens         cdata    int32_t array.
--- @param  n              number   Number of tokens.
--- @param  remove_special bool?    Strip BOS/EOS (default: false).
--- @param  unparse_special bool?   Convert specials back to text (default: false).
--- @return string
function Vocab:detokenize(tokens, n, remove_special, unparse_special)
    remove_special  = remove_special  or false
    unparse_special = unparse_special or false

    local len = self._lib.llama_detokenize(
        self._ptr, tokens, n,
        self._dtok_buf, DETOK_BUF_SIZE,
        remove_special, unparse_special
    )
    if len < 0 then
        error("[ion7.core.vocab] detokenize buffer too small", 2)
    end
    return self._ffi.string(self._dtok_buf, len)
end

--- Convert a single token to its text piece.
---
--- @param  token    number  Token ID.
--- @param  special  bool?   Include special token text (default: true).
--- @return string
function Vocab:piece(token, special)
    if special == nil then special = true end
    local n = self._lib.llama_token_to_piece(
        self._ptr, token,
        self._piece_buf, PIECE_BUF_SIZE,
        0, special
    )
    if n < 0 then
        -- Expand buffer for large pieces (rare)
        local big = self._ffi.new("char[?]", -n + 1)
        n = self._lib.llama_token_to_piece(
            self._ptr, token, big, -n + 1, 0, special)
        return n > 0 and self._ffi.string(big, n) or ""
    end
    return n > 0 and self._ffi.string(self._piece_buf, n) or ""
end

--- Get the raw text representation of a token (without lstrip behavior).
---
--- @param  token  number  Token ID.
--- @return string
function Vocab:text(token)
    local p = self._lib.llama_vocab_get_text(self._ptr, token)
    return p ~= nil and self._ffi.string(p) or ""
end

--- Check whether a token is an end-of-generation token.
---
--- @param  token  number  Token ID.
--- @return bool
function Vocab:is_eog(token)
    return self._lib.llama_vocab_is_eog(self._ptr, token)
end

--- @return number  Beginning-of-sequence token ID.
function Vocab:bos() return tonumber(self._lib.llama_vocab_bos(self._ptr)) end

--- @return number  End-of-sequence token ID.
function Vocab:eos() return tonumber(self._lib.llama_vocab_eos(self._ptr)) end

--- @return number  End-of-turn token ID.
function Vocab:eot() return tonumber(self._lib.llama_vocab_eot(self._ptr)) end

--- @return number  Newline token ID.
function Vocab:nl()  return tonumber(self._lib.llama_vocab_nl(self._ptr))  end

--- @return number  Padding token ID.
function Vocab:pad() return tonumber(self._lib.llama_vocab_pad(self._ptr)) end

--- Fill-in-the-Middle token accessors.
--- @return number
function Vocab:fim_pre() return tonumber(self._lib.llama_vocab_fim_pre(self._ptr)) end
function Vocab:fim_suf() return tonumber(self._lib.llama_vocab_fim_suf(self._ptr)) end
function Vocab:fim_mid() return tonumber(self._lib.llama_vocab_fim_mid(self._ptr)) end
function Vocab:fim_pad() return tonumber(self._lib.llama_vocab_fim_pad(self._ptr)) end
function Vocab:fim_rep() return tonumber(self._lib.llama_vocab_fim_rep(self._ptr)) end
function Vocab:fim_sep() return tonumber(self._lib.llama_vocab_fim_sep(self._ptr)) end

--- @return number  Sentence separator token ID.
function Vocab:sep()  return tonumber(self._lib.llama_vocab_sep(self._ptr))  end
--- @return number  Mask token ID.
function Vocab:mask() return tonumber(self._lib.llama_vocab_mask(self._ptr)) end
--- @return number  CLS (classification) token ID.
--- @return bool  Whether add_sep is set.
function Vocab:get_add_sep() return self._lib.llama_vocab_get_add_sep(self._ptr) == 1 end
function Vocab:get_add_bos() return self._lib.llama_vocab_get_add_bos(self._ptr) == 1 end
function Vocab:get_add_eos() return self._lib.llama_vocab_get_add_eos(self._ptr) == 1 end

--- Alias: n_tokens() = n_vocab()
function Vocab:n_tokens() return self:n_vocab() end

--- CLS token ID (-1 if not present).
function Vocab:cls() return tonumber(self._lib.llama_vocab_cls(self._ptr)) end

--- Get the float score of a token (used in Unigram/SPM tokenizers).
--- @param  token  number
--- @return number
function Vocab:score(token)
    return tonumber(self._lib.llama_vocab_get_score(self._ptr, token))
end

--- Get the attribute flags of a token.
--- Returns a bitmask of LLAMA_TOKEN_ATTR_* values.
--- @param  token  number
--- @return number
function Vocab:attr(token)
    return tonumber(self._lib.llama_vocab_get_attr(self._ptr, token))
end

--- Returns true if a token is a control token.
--- @param  token  number
--- @return bool
function Vocab:is_control(token)
    return self._lib.llama_vocab_is_control(self._ptr, token)
end

--- List the names of all built-in chat templates available in llama.cpp.
--- @return table  Array of template name strings.
function Vocab:builtin_templates()
    local ffi  = self._ffi
    local n    = 64
    local bufs = ffi.new("const char*[?]", n)
    local count = self._lib.llama_chat_builtin_templates(bufs, n)
    if count < 0 then return {} end
    local result = {}
    for i = 0, math.min(count, n) - 1 do
        result[i + 1] = ffi.string(bufs[i])
    end
    return result
end

--- Apply the model's built-in chat template to a list of messages.
---
--- @param  messages   table   Array of { role = string, content = string }.
--- @param  add_ass    bool?   Append the assistant generation prompt (default: true).
--- @param  template   string? Override the model's default template (rarely needed).
--- @return string     Formatted prompt string ready for tokenisation.
--- @error  If the template application fails or the buffer overflows.
function Vocab:apply_template(messages, add_ass, template)
    if add_ass == nil then add_ass = true end

    local n   = #messages
    local ffi = self._ffi
    local msgs = ffi.new("llama_chat_message[?]", n)
    -- Anchor Lua strings to prevent GC during the C call
    local role_refs    = {}
    local content_refs = {}

    for i, msg in ipairs(messages) do
        local role    = tostring(msg.role    or "user")
        local content = tostring(msg.content or "")
        role_refs[i]    = role
        content_refs[i] = content
        msgs[i - 1].role    = role
        msgs[i - 1].content = content
    end

    -- Use bridge wrapper: handles API change between llama.cpp versions
    -- (model param removed in master, ion7_chat_apply_template abstracts this)
    local needed = self._bridge.ion7_chat_apply_template(
        template or nil,
        msgs, n,
        add_ass,
        self._tmpl_buf, TMPL_BUF_SIZE
    )
    if needed < 0 then
        error("[ion7.core.vocab] chat template output buffer too small", 2)
    end
    if needed > TMPL_BUF_SIZE then
        -- Grow buffer and retry
        local big = ffi.new("char[?]", needed + 1)
        needed = self._bridge.ion7_chat_apply_template(
            template or nil,
            msgs, n, add_ass,
            big, needed + 1
        )
        if needed < 0 then
            error("[ion7.core.vocab] chat template failed on retry", 2)
        end
        return ffi.string(big, needed)
    end

    return ffi.string(self._tmpl_buf, needed)
end

return Vocab
