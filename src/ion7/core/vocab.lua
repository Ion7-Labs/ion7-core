--- @module ion7.core.vocab
--- SPDX-License-Identifier: MIT
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
local ffi    = require "ffi"

-- ── Pre-parsed VLA ctypes ─────────────────────────────────────────────────────
-- doc: "parse the cdecl only once and get its ctype with ffi.typeof().
--       Then use the ctype as a constructor repeatedly."
-- This avoids the C-declaration string parse on every ffi.new() call.
local _i32arr  = ffi.typeof("int32_t[?]")
local _chararr = ffi.typeof("char[?]")
local _cptrarr = ffi.typeof("const char*[?]")

-- ── Internal buffer for token-to-piece conversion ─────────────────────────────

local PIECE_BUF_SIZE  = 256
local PIECE_BIG_SIZE  = 4096    -- Pre-alloc for rare oversized pieces (avoids alloc on fallback)
local TMPL_BUF_SIZE   = 131072  -- 128 KB for chat template output
local DETOK_BUF_SIZE  = 65536   -- 64  KB for detokenize output

-- ── Vocab ─────────────────────────────────────────────────────────────────────

--- @class Vocab
--- @field _ptr    cdata   Opaque llama_vocab* (not owned, do not free).
--- @field _lib    cdata   libllama.so namespace.
--- @field _model  cdata   llama_model* for chat template calls.
--- @field _bridge cdata   ion7_bridge.so namespace.
--- @field _ffi    cdata   LuaJIT ffi namespace.
--- @field _tmpls  cdata   ion7_chat_templates_t* (owned, freed via ffi.gc).
--- @field _piece_buf   cdata  Scratch buffer for token-to-piece conversion.
--- @field _piece_big   cdata  Pre-alloc fallback buffer for oversized pieces.
--- @field _piece_cache table  Strong cache: token id → piece string (integer keys, GC-safe).
--- @field _tmpl_buf    cdata  Scratch buffer for chat template output.
--- @field _dtok_buf    cdata  Scratch buffer for detokenize output.
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
    local L      = Loader.instance()
    local ffi    = L.ffi
    local bridge = L.bridge

    -- Initialise Jinja2 chat templates from the model (libcommon).
    -- ffi.gc ensures the handle is freed when the Vocab is collected.
    local tmpls = ffi.gc(
        bridge.ion7_chat_templates_init(model, nil),
        bridge.ion7_chat_templates_free
    )

    return setmetatable({
        _ptr    = ptr,
        _lib    = lib,
        _bridge = bridge,
        _model  = model,
        _ffi    = ffi,
        _tmpls  = tmpls,
        -- Pre-allocated scratch buffers
        _piece_buf   = _chararr(PIECE_BUF_SIZE),
        _piece_big   = _chararr(PIECE_BIG_SIZE),  -- reused for oversized pieces
        _piece_cache = {},  -- strong cache: integer token id → piece string
        _tmpl_buf    = _chararr(TMPL_BUF_SIZE),
        _dtok_buf    = _chararr(DETOK_BUF_SIZE),
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
    local buf        = _i32arr(max_tokens)
    local n          = self._lib.llama_tokenize(
        self._ptr, text, #text,
        buf, max_tokens,
        add_special, parse_special
    )
    if n < 0 then
        -- Buffer was too small; reallocate and retry
        max_tokens = -n + 16
        buf        = _i32arr(max_tokens)
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
    local cache = self._piece_cache
    local cached = cache[token]
    if cached then return cached end

    local n = self._lib.llama_token_to_piece(
        self._ptr, token,
        self._piece_buf, PIECE_BUF_SIZE,
        0, special
    )
    local s
    if n < 0 then
        -- Expand into pre-allocated big buffer; only heap-alloc if oversized
        local sz  = -n + 1
        local big = sz <= PIECE_BIG_SIZE and self._piece_big or _chararr(sz)
        n = self._lib.llama_token_to_piece(self._ptr, token, big, sz, 0, special)
        s = n > 0 and self._ffi.string(big, n) or ""
    else
        s = n > 0 and self._ffi.string(self._piece_buf, n) or ""
    end
    cache[token] = s
    return s
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

--- CLS token ID. CLS == BOS in llama.cpp (llama_vocab_cls is deprecated).
function Vocab:cls() return tonumber(self._lib.llama_vocab_bos(self._ptr)) end

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
    local bufs = _cptrarr(n)
    local count = self._lib.llama_chat_builtin_templates(bufs, n)
    if count < 0 then return {} end
    local result = {}
    for i = 0, math.min(count, n) - 1 do
        result[i + 1] = ffi.string(bufs[i])
    end
    return result
end

--- Apply the model's Jinja2 chat template to a list of messages.
---
--- @param  messages         table   Array of { role = string, content = string }.
--- @param  add_ass          bool?   Append the assistant generation prompt (default: true).
--- @param  enable_thinking  int?    -1 = model default, 0 = disable, 1 = enable (default: -1).
--- @return string           Formatted prompt string ready for tokenisation.
--- @error  If the template application fails.
function Vocab:apply_template(messages, add_ass, enable_thinking)
    if add_ass         == nil then add_ass         = true end
    if enable_thinking == nil then enable_thinking = -1   end

    local n      = #messages
    local ffi    = self._ffi
    local bridge = self._bridge

    -- Build parallel C string pointer arrays.
    -- Lua strings are anchored in role_refs/content_refs to prevent GC
    -- during the bridge call (ffi pointers don't pin Lua strings).
    local roles    = _cptrarr(n)
    local contents = _cptrarr(n)
    local role_refs    = {}
    local content_refs = {}

    for i, msg in ipairs(messages) do
        local role    = tostring(msg.role    or "user")
        local content = tostring(msg.content or "")
        role_refs[i]    = role
        content_refs[i] = content
        roles[i - 1]    = role
        contents[i - 1] = content
    end

    local needed = bridge.ion7_chat_templates_apply(
        self._tmpls,
        roles, contents, n,
        add_ass and 1 or 0,
        enable_thinking,
        self._tmpl_buf, TMPL_BUF_SIZE
    )
    if needed < 0 then
        error("[ion7.core.vocab] chat template failed", 2)
    end
    if needed > TMPL_BUF_SIZE then
        local big = _chararr(needed)
        needed = bridge.ion7_chat_templates_apply(
            self._tmpls,
            roles, contents, n,
            add_ass and 1 or 0,
            enable_thinking,
            big, needed
        )
        if needed < 0 then
            error("[ion7.core.vocab] chat template failed on retry", 2)
        end
        return ffi.string(big, needed - 1)
    end

    return ffi.string(self._tmpl_buf, needed - 1)
end

--- Returns true if the model's template supports enable_thinking.
--- (Qwen3/3.5, DeepSeek-R1 and similar reasoning models return true.)
--- @return bool
function Vocab:supports_thinking()
    return self._bridge.ion7_chat_templates_support_thinking(self._tmpls) == 1
end

return Vocab
