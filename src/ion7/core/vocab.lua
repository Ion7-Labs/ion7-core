--- @module ion7.core.vocab
--- @author  ion7 / Ion7 Project Contributors
---
--- Vocabulary handle : tokenisation, detokenisation, special-token
--- queries, chat-template application.
---
--- A `Vocab` wraps a `const llama_vocab*` returned by
--- `llama_model_get_vocab`. The pointer is OWNED by the model — we do
--- not free it. Lifetime is anchored on the parent Model via
--- `self._model_ref`.
---
--- The `_tmpls` field is the only owned native resource : a
--- `common_chat_templates*` allocated by the bridge for Jinja2 chat
--- templating. `ffi.gc` frees it when the Vocab is collected.
---
--- Performance note : every text→bytes round-trip uses a pre-allocated
--- scratch buffer ; the per-token `piece()` cache stores Lua strings so
--- the hot path of streaming inference (one piece per generated token)
--- never allocates a new buffer.
---
---   local model = ion7.Model.load("...gguf")
---   local vocab = model:vocab()
---   local toks, n = vocab:tokenize("Hello, world!", true, true)
---   local txt     = vocab:detokenize(toks, n)
---
--- The `Vocab` is created by `Model:vocab()` ; calling `Vocab.new`
--- directly is fine but typically unnecessary.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_vocab = require "ion7.core.ffi.llama.vocab" -- llama_vocab_*, llama_tokenize, ...
local llama_chat = require "ion7.core.ffi.llama.chat" -- llama_chat_builtin_templates

local ffi_new = ffi.new
local ffi_gc = ffi.gc
local ffi_string = ffi.string
local tonumber = tonumber
local math_max = math.max
local math_min = math.min

-- Pre-resolved cdata typeofs : repeating `ffi.typeof("char[?]")` on the
-- hot path forces LuaJIT to re-parse the cdecl. Caching the typeof
-- handles makes `_chararr(n)` a straight cdata constructor call.
local I32ARR = ffi.typeof("int32_t[?]")
local CHARARR = ffi.typeof("char[?]")
local CPTRARR = ffi.typeof("const char*[?]")

-- Scratch buffer sizes. Empirically chosen :
--   PIECE_BUF_SIZE   : ASCII pieces fit in 256 bytes ; specials/CJK overflow.
--   PIECE_BIG_SIZE   : pre-allocated fallback for the rare oversized piece.
--   TMPL_BUF_SIZE    : 128 KB covers prompts up to ~25k tokens of context.
--   DETOK_BUF_SIZE   :  64 KB covers ~16k tokens of output ; we retry-grow
--                      automatically when the model produces more.
local PIECE_BUF_SIZE = 256
local PIECE_BIG_SIZE = 4096
local TMPL_BUF_SIZE = 131072
local DETOK_BUF_SIZE = 65536

-- Symbolic name for the `enum llama_vocab_type` returned by
-- `llama_vocab_type` — easier to branch on than the raw int.
local VOCAB_TYPE_NAMES = {
    [0] = "none",
    [1] = "spm",
    [2] = "bpe",
    [3] = "wpm",
    [4] = "ugm",
    [5] = "rwkv"
}

--- @class ion7.core.Vocab
--- @field _ptr        cdata    `const llama_vocab*` (not owned).
--- @field _model_ref  table    Parent Model — keeps the model alive.
--- @field _tmpls      cdata?   `common_chat_templates*` (owned, GC-freed).
--- @field _piece_buf  cdata    Pre-allocated 256-byte buffer for `piece()`.
--- @field _piece_big  cdata    Pre-allocated 4-KB fallback buffer.
--- @field _piece_cache table   Memoised `[token_id] = piece_string` cache.
--- @field _tmpl_buf   cdata    Pre-allocated buffer for chat templates.
--- @field _dtok_buf   cdata    Pre-allocated buffer for detokenisation.
local Vocab = {}
Vocab.__index = Vocab

-- ── Constructor ───────────────────────────────────────────────────────────

--- Wrap a raw `llama_vocab*` returned by `llama_model_get_vocab`. The
--- caller MUST keep the parent Model alive at least as long as the
--- Vocab — we anchor it as `self._model_ref` for that very purpose.
---
--- The chat-templates handle is initialised eagerly. If the bridge
--- shared library is missing, the `require` of `ion7.core.ffi.bridge`
--- would have failed at module load already — but we degrade
--- gracefully here by checking the call result, leaving `_tmpls = nil`
--- when initialisation fails (e.g. for a model with no embedded
--- template). Calls to `apply_template` will then raise a clear error.
---
--- @param  model ion7.core.Model Parent model.
--- @param  ptr   cdata           `const llama_vocab*`.
--- @return ion7.core.Vocab
function Vocab.new(model, ptr)
    assert(ptr ~= nil, "[ion7.core.vocab] vocab pointer is NULL")

    -- Lazy-require the bridge so a Vocab can still be created in
    -- environments where the bridge .so is absent (purely tokenisation
    -- workloads). Chat-template features will then degrade with a
    -- clear error at use site.
    local bridge_ok, bridge = pcall(require, "ion7.core.ffi.bridge")

    local tmpls
    if bridge_ok then
        local raw = bridge.ion7_chat_templates_init(model:ptr(), nil)
        if raw ~= nil then
            tmpls = ffi_gc(raw, bridge.ion7_chat_templates_free)
        end
    end

    return setmetatable(
        {
            _ptr = ptr,
            _model_ref = model,
            _tmpls = tmpls,
            _piece_buf = CHARARR(PIECE_BUF_SIZE),
            _piece_big = CHARARR(PIECE_BIG_SIZE),
            _piece_cache = {},
            _tmpl_buf = CHARARR(TMPL_BUF_SIZE),
            _dtok_buf = CHARARR(DETOK_BUF_SIZE)
        },
        Vocab
    )
end

-- ── Identity ──────────────────────────────────────────────────────────────

--- @return integer Total number of tokens in the vocabulary.
function Vocab:n_vocab()
    return tonumber(llama_vocab.llama_vocab_n_tokens(self._ptr))
end

--- Backwards-compatible alias for `n_vocab`.
function Vocab:n_tokens()
    return self:n_vocab()
end

--- @return string Vocab kind : `"spm"`, `"bpe"`, `"wpm"`, `"ugm"`,
---                `"rwkv"`, `"none"` or `"unknown"`.
function Vocab:type()
    local t = tonumber(llama_vocab.llama_vocab_type(self._ptr))
    return VOCAB_TYPE_NAMES[t] or "unknown"
end

-- ── Tokenisation ──────────────────────────────────────────────────────────

--- Tokenise UTF-8 `text` into an int32 cdata array.
---
--- `llama_tokenize` returns the negated required size when our buffer
--- is too small ; we honour that contract by reallocating once and
--- retrying. The initial heuristic (1 token per byte + 16 specials) is
--- a generous overestimate that almost always avoids the retry.
---
--- @param  text          string UTF-8 input.
--- @param  add_special   boolean? Add BOS/EOS tokens (default false).
--- @param  parse_special boolean? Parse `<...>` special tokens (default true).
--- @return cdata    `int32_t[?]` token array (0-based).
--- @return integer  Token count.
--- @raise   When tokenisation fails after the retry.
function Vocab:tokenize(text, add_special, parse_special)
    if add_special == nil then
        add_special = false
    end
    if parse_special == nil then
        parse_special = true
    end

    local cap = math_max(#text + 16, 64)
    local buf = I32ARR(cap)
    local n = llama_vocab.llama_tokenize(self._ptr, text, #text, buf, cap, add_special, parse_special)

    if n < 0 then
        -- Buffer too small — retry with the requested capacity + headroom.
        cap = -n + 16
        buf = I32ARR(cap)
        n = llama_vocab.llama_tokenize(self._ptr, text, #text, buf, cap, add_special, parse_special)
        if n < 0 then
            error(string.format("[ion7.core.vocab] tokenize failed (n=%d)", n), 2)
        end
    end
    return buf, tonumber(n)
end

--- Detokenise back into a UTF-8 Lua string.
---
--- Uses the pre-allocated `_dtok_buf` (64 KB). For exceptionally long
--- outputs we retry once with a heap-allocated buffer sized to the
--- exact requested capacity.
---
--- @param  tokens          cdata   `int32_t[?]` token array (0-based).
--- @param  n               integer Token count.
--- @param  remove_special  boolean? Strip BOS/EOS (default false).
--- @param  unparse_special boolean? Convert specials back to text (default false).
--- @return string
function Vocab:detokenize(tokens, n, remove_special, unparse_special)
    remove_special = remove_special or false
    unparse_special = unparse_special or false

    local len =
        llama_vocab.llama_detokenize(
        self._ptr,
        tokens,
        n,
        self._dtok_buf,
        DETOK_BUF_SIZE,
        remove_special,
        unparse_special
    )

    if len < 0 then
        -- Buffer too small ; allocate one of the exact required size.
        local need = -len
        local big = CHARARR(need)
        len = llama_vocab.llama_detokenize(self._ptr, tokens, n, big, need, remove_special, unparse_special)
        if len < 0 then
            error("[ion7.core.vocab] detokenize failed after retry", 2)
        end
        return ffi_string(big, len)
    end
    return ffi_string(self._dtok_buf, len)
end

-- ── Single-token text accessors ───────────────────────────────────────────

--- Convert a single token to its visible text piece. Memoised : the
--- second call with the same token id returns a cached Lua string.
---
--- @param  token   integer Token id.
--- @param  special boolean? Render special-token text too (default true).
--- @return string
function Vocab:piece(token, special)
    if special == nil then
        special = true
    end
    local cache = self._piece_cache
    local cached = cache[token]
    if cached then
        return cached
    end

    local n = llama_vocab.llama_token_to_piece(self._ptr, token, self._piece_buf, PIECE_BUF_SIZE, 0, special)

    local s
    if n < 0 then
        -- Piece overflows our 256-byte stash : fall back to the 4-KB
        -- pre-allocated buffer ; only allocate fresh when even THAT is
        -- too small (CJK script + emoji combined can require ~6 KB).
        local sz = -n + 1
        local big = sz <= PIECE_BIG_SIZE and self._piece_big or CHARARR(sz)
        n = llama_vocab.llama_token_to_piece(self._ptr, token, big, sz, 0, special)
        s = (n > 0) and ffi_string(big, n) or ""
    else
        s = (n > 0) and ffi_string(self._piece_buf, n) or ""
    end
    cache[token] = s
    return s
end

--- Raw text representation of a token (no lstrip / SPM whitespace
--- normalisation). Empty string if the token has no text view.
--- @param  token integer
--- @return string
function Vocab:text(token)
    local p = llama_vocab.llama_vocab_get_text(self._ptr, token)
    return p ~= nil and ffi_string(p) or ""
end

--- Float score of a token (only meaningful for SPM/Unigram vocabs).
--- @param  token integer
--- @return number
function Vocab:score(token)
    return tonumber(llama_vocab.llama_vocab_get_score(self._ptr, token))
end

--- Attribute bitmask for a token (`LLAMA_TOKEN_ATTR_*`).
--- @param  token integer
--- @return integer
function Vocab:attr(token)
    return tonumber(llama_vocab.llama_vocab_get_attr(self._ptr, token))
end

-- ── Token kind queries ────────────────────────────────────────────────────

--- True for end-of-generation tokens (EOS, EOT, custom STOP, ...).
--- @param  token integer
--- @return boolean
function Vocab:is_eog(token)
    return llama_vocab.llama_vocab_is_eog(self._ptr, token) == true
end

--- True for control / special tokens.
--- @param  token integer
--- @return boolean
function Vocab:is_control(token)
    return llama_vocab.llama_vocab_is_control(self._ptr, token) == true
end

-- ── Special-token IDs ─────────────────────────────────────────────────────

--- Beginning-of-sequence token id, or `-1` if the model has none.
function Vocab:bos()
    return tonumber(llama_vocab.llama_vocab_bos(self._ptr))
end
--- End-of-sequence token id, or `-1` if absent.
function Vocab:eos()
    return tonumber(llama_vocab.llama_vocab_eos(self._ptr))
end
--- End-of-turn token id (chat models).
function Vocab:eot()
    return tonumber(llama_vocab.llama_vocab_eot(self._ptr))
end
--- Newline token id, or `-1`.
function Vocab:nl()
    return tonumber(llama_vocab.llama_vocab_nl(self._ptr))
end
--- Padding token id.
function Vocab:pad()
    return tonumber(llama_vocab.llama_vocab_pad(self._ptr))
end
--- Sentence-separator token id.
function Vocab:sep()
    return tonumber(llama_vocab.llama_vocab_sep(self._ptr))
end
--- Mask token id (for masked-LM models).
function Vocab:mask()
    return tonumber(llama_vocab.llama_vocab_mask(self._ptr))
end

-- CLS upstream is deprecated in favour of BOS — keep the alias for
-- backwards compatibility with consumers that were calling it.
function Vocab:cls()
    return tonumber(llama_vocab.llama_vocab_bos(self._ptr))
end

--- Fill-in-the-Middle prefix token id. `-1` when the model has no FIM.
function Vocab:fim_pre()
    return tonumber(llama_vocab.llama_vocab_fim_pre(self._ptr))
end
--- FIM suffix token id.
function Vocab:fim_suf()
    return tonumber(llama_vocab.llama_vocab_fim_suf(self._ptr))
end
--- FIM middle token id.
function Vocab:fim_mid()
    return tonumber(llama_vocab.llama_vocab_fim_mid(self._ptr))
end
--- FIM padding token id.
function Vocab:fim_pad()
    return tonumber(llama_vocab.llama_vocab_fim_pad(self._ptr))
end
--- FIM repository-marker token id.
function Vocab:fim_rep()
    return tonumber(llama_vocab.llama_vocab_fim_rep(self._ptr))
end
--- FIM separator token id.
function Vocab:fim_sep()
    return tonumber(llama_vocab.llama_vocab_fim_sep(self._ptr))
end

-- ── Auto-prepend / append flags ───────────────────────────────────────────

--- True if the vocab automatically prepends BOS during tokenisation.
function Vocab:get_add_bos()
    return llama_vocab.llama_vocab_get_add_bos(self._ptr) == true
end
--- True if the vocab automatically appends EOS during tokenisation.
function Vocab:get_add_eos()
    return llama_vocab.llama_vocab_get_add_eos(self._ptr) == true
end
--- True if the vocab automatically appends a sentence separator.
function Vocab:get_add_sep()
    return llama_vocab.llama_vocab_get_add_sep(self._ptr) == true
end

-- ── Built-in chat templates ───────────────────────────────────────────────

--- Names of the chat templates llama.cpp ships with (separate from the
--- per-model template stored inside the GGUF). Useful for debugging /
--- introspection ; production code should generally rely on
--- `apply_template` which uses the model's own template.
--- @return string[] Template name strings.
function Vocab:builtin_templates()
    local cap = 64
    local out = CPTRARR(cap)
    local n = llama_chat.llama_chat_builtin_templates(out, cap)
    if n < 0 then
        return {}
    end
    local result = {}
    for i = 0, math_min(n, cap) - 1 do
        result[i + 1] = ffi_string(out[i])
    end
    return result
end

-- ── Jinja2 chat template (bridge) ─────────────────────────────────────────

--- Apply the model's embedded Jinja2 chat template to a sequence of
--- messages. Returns the formatted prompt ready to feed to
--- `Vocab:tokenize`.
---
--- The return string excludes the trailing NUL byte (that's what
--- `needed - 1` is computing — `ion7_chat_templates_apply` reports the
--- byte count INCLUDING NUL).
---
--- @param  messages        table[] Array of `{ role = string, content = string }`.
--- @param  add_ass         boolean? Append assistant generation prefix (default true).
--- @param  enable_thinking integer? `-1` model default, `0` off, `1` on.
--- @return string Formatted prompt.
--- @raise   When the template engine fails (missing `_tmpls` or runtime error).
function Vocab:apply_template(messages, add_ass, enable_thinking)
    if not self._tmpls then
        error(
            "[ion7.core.vocab] chat templates unavailable " .. "(bridge .so missing or model has no embedded template)",
            2
        )
    end
    if add_ass == nil then
        add_ass = true
    end
    if enable_thinking == nil then
        enable_thinking = -1
    end

    local bridge = require "ion7.core.ffi.bridge"
    local n = #messages
    local roles = CPTRARR(n)
    local conts = CPTRARR(n)
    -- Anchor the Lua strings so the GC does not free them while the
    -- bridge call is reading the C pointers we set up.
    local anchor_r, anchor_c = {}, {}
    for i, msg in ipairs(messages) do
        local r = tostring(msg.role or "user")
        local c = tostring(msg.content or "")
        anchor_r[i] = r
        anchor_c[i] = c
        roles[i - 1] = r
        conts[i - 1] = c
    end

    local needed =
        bridge.ion7_chat_templates_apply(
        self._tmpls,
        roles,
        conts,
        n,
        add_ass and 1 or 0,
        enable_thinking,
        self._tmpl_buf,
        TMPL_BUF_SIZE
    )

    if needed < 0 then
        error("[ion7.core.vocab] chat template failed (rc=" .. needed .. ")", 2)
    end

    -- Re-call with a heap buffer when the prompt outgrew our scratch.
    if needed > TMPL_BUF_SIZE then
        local big = CHARARR(needed)
        needed =
            bridge.ion7_chat_templates_apply(
            self._tmpls,
            roles,
            conts,
            n,
            add_ass and 1 or 0,
            enable_thinking,
            big,
            needed
        )
        if needed < 0 then
            error("[ion7.core.vocab] chat template failed on retry " .. "(rc=" .. needed .. ")", 2)
        end
        return ffi_string(big, needed - 1)
    end
    return ffi_string(self._tmpl_buf, needed - 1)
end

--- True if the embedded template recognises `enable_thinking` (Qwen3,
--- DeepSeek-R1 et al.).
--- @return boolean
function Vocab:supports_thinking()
    if not self._tmpls then
        return false
    end
    local bridge = require "ion7.core.ffi.bridge"
    return bridge.ion7_chat_templates_support_thinking(self._tmpls) == 1
end

return Vocab
