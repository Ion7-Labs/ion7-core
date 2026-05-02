--- @module ion7.core.custom_sampler
--- @author  ion7 / Ion7 Project Contributors
---
--- Plug-your-own-sampler-into-llama.cpp from pure Lua.
---
--- The historical implementation went through the bridge's
--- `ion7_sampler_create` because LuaJIT's `ffi.cast`-as-callback pattern
--- has subtle GC-lifetime hazards : if the cast cdata is collected
--- while llama.cpp still holds a function pointer to it, the next call
--- crashes. The bridge wrapped that in C and stored the trampolines in
--- a malloc'd struct it owned.
---
--- The V2 rewrite replicates the same dance in pure Lua + LuaJIT FFI :
---
---   1. Allocate a `struct llama_sampler_i` cdata, fill its function-
---      pointer fields with `ffi.cast`-built trampolines.
---   2. Allocate a `char[64]` cdata to hold the sampler's name (which
---      `iface.name` returns as a `const char*`).
---   3. Anchor every cast cdata AND the iface struct on the
---      CustomSampler instance ; as long as the instance is reachable
---      from Lua, the trampolines and the iface stay alive.
---   4. Call `llama_sampler_init(&iface, nil)` to get the
---      `llama_sampler*` handle suitable for plugging into a chain.
---
--- Critical safety rule for the trampolines :
---   ANY uncaught Lua error inside an `ffi.cast` callback aborts the
---   process with `PANIC: bad callback`. We wrap every Lua callback
---   invocation in `pcall` and silently fall back to a sensible
---   default. The user sees the error in their inference output as a
---   degenerate sample (token 0), not a crash.
---
---   local cs = ion7.CustomSampler.new("greedy_lua", {
---     apply = function(candidates, n)
---       -- candidates is a `llama_token_data*` cdata array (0-based).
---       -- Returning the chosen index sets `cur_p.selected` C-side.
---       local best, best_logit = 0, -math.huge
---       for i = 0, n - 1 do
---         if candidates[i].logit > best_logit then
---           best_logit = candidates[i].logit
---           best = i
---         end
---       end
---       return best
---     end,
---   })
---
---   local chain = ion7.Sampler.chain():custom(cs):build(vocab)
---
--- Keep the CustomSampler instance reachable for as long as the chain
--- using it stays in use — the chain holds POINTERS into our cdata, not
--- managed references.

local ffi = require "ffi"
require "ion7.core.ffi.types"

local llama_sampler = require "ion7.core.ffi.llama.sampler" -- llama_sampler_init

local ffi_new = ffi.new
local ffi_cast = ffi.cast
local ffi_copy = ffi.copy
local ffi_gc = ffi.gc
local pcall = pcall
local tonumber = tonumber

local NAME_BUF_SIZE = 64

--- @class ion7.core.CustomSampler
--- @field _ptr      cdata  Raw `llama_sampler*` for chain insertion.
--- @field _name     string Human-readable name.
--- @field _name_buf cdata  `char[NAME_BUF_SIZE]` holding `_name`.
--- @field _iface    cdata  `struct llama_sampler_i` populated with trampolines.
--- @field _cb_*     cdata  Anchored trampolines : DO NOT lose these refs.
local CustomSampler = {}
CustomSampler.__index = CustomSampler

--- Build a custom sampler from a Lua callback table.
---
--- @param  name      string  Display name (≤ 63 bytes ; truncated otherwise).
--- @param  callbacks table   `{ apply = fn, accept = fn?, reset = fn? }`
---                           - `apply(candidates_cdata, n_int)` is REQUIRED.
---                             It must return the 0-based index of the
---                             chosen token in `candidates`.
---                           - `accept(token_int)` is called after the
---                             chain accepts a token. Optional.
---                           - `reset()` is called when the chain is
---                             reset (between conversations). Optional.
--- @return ion7.core.CustomSampler
--- @raise   When `callbacks.apply` is missing or `llama_sampler_init` fails.
function CustomSampler.new(name, callbacks)
    assert(
        type(callbacks) == "table" and type(callbacks.apply) == "function",
        "[ion7.core.custom_sampler] callbacks.apply is required"
    )

    name = name or "custom"

    -- Persistent buffer for the name : `iface.name` returns a `const char*`,
    -- so we need somewhere stable to point to. We copy at most 63 bytes
    -- (leave a slot for the NUL terminator).
    local name_buf = ffi_new("char[?]", NAME_BUF_SIZE)
    local copy_n = math.min(#name, NAME_BUF_SIZE - 1)
    ffi_copy(name_buf, name, copy_n)
    name_buf[copy_n] = 0

    -- ── Build the trampolines ───────────────────────────────────────────────
    -- Each trampoline wraps a Lua function in pcall so a buggy callback
    -- becomes a benign default rather than a process abort.

    -- name(smpl) -> const char*
    local cb_name =
        ffi_cast(
        "const char *(*)(const struct llama_sampler *)",
        function(_)
            return name_buf
        end
    )

    -- apply(smpl, cur_p) -> void
    local apply_fn = callbacks.apply
    local cb_apply =
        ffi_cast(
        "void (*)(struct llama_sampler *, llama_token_data_array *)",
        function(_, cur_p)
            local ok, idx = pcall(apply_fn, cur_p.data, tonumber(cur_p.size))
            if ok and type(idx) == "number" and idx >= 0 then
                cur_p.selected = idx
            else
                -- Fallback : pick the first candidate. Keeps inference moving
                -- even when the user's callback throws.
                cur_p.selected = 0
            end
        end
    )

    -- accept(smpl, token) -> void
    local cb_accept
    if type(callbacks.accept) == "function" then
        local accept_fn = callbacks.accept
        cb_accept =
            ffi_cast(
            "void (*)(struct llama_sampler *, llama_token)",
            function(_, tok)
                pcall(accept_fn, tonumber(tok))
            end
        )
    end

    -- reset(smpl) -> void
    local cb_reset
    if type(callbacks.reset) == "function" then
        local reset_fn = callbacks.reset
        cb_reset =
            ffi_cast(
            "void (*)(struct llama_sampler *)",
            function(_)
                pcall(reset_fn)
            end
        )
    end

    -- free(smpl) -> void  (no-op : ownership stays Lua-side)
    local cb_free =
        ffi_cast(
        "void (*)(struct llama_sampler *)",
        function(_)
        end
    )

    -- ── Populate the iface struct ───────────────────────────────────────────
    -- We allocate the struct as cdata so the addressable lifetime is tied
    -- to our anchor below ; field assignment then writes the function-
    -- pointer values directly. The 4 backend_* slots stay nil — those are
    -- experimental upstream and unused by every shipping sampler chain.

    local iface = ffi_new("struct llama_sampler_i")
    iface.name = cb_name
    iface.apply = cb_apply
    iface.accept = cb_accept -- nil is a valid value (LuaJIT writes a NULL pointer)
    iface.reset = cb_reset
    iface.clone = nil -- chain never clones our sampler
    iface["free"] = cb_free -- `free` is a Lua keyword for tables — quote it

    -- ── Hand the iface to llama.cpp ─────────────────────────────────────────
    -- ctx is nil : closures capture all the state the trampolines need.
    local ptr = llama_sampler.llama_sampler_init(iface, nil)
    if ptr == nil then
        error("[ion7.core.custom_sampler] llama_sampler_init returned NULL", 2)
    end

    return setmetatable(
        {
            _ptr = ptr, -- raw llama_sampler* (no ffi.gc — chain owns it)
            _name = name,
            _name_buf = name_buf, -- anchor : `iface.name` reads from this
            _iface = iface, -- anchor : llama.cpp dereferences `iface.*` lazily
            _cb_name = cb_name, -- anchor : the trampolines must outlive llama.cpp's refs
            _cb_apply = cb_apply,
            _cb_accept = cb_accept,
            _cb_reset = cb_reset,
            _cb_free = cb_free
        },
        CustomSampler
    )
end

--- Raw `llama_sampler*` cdata for plugging into a sampler chain.
--- @return cdata
function CustomSampler:ptr()
    return self._ptr
end

--- Display name (the same string passed to `new`).
--- @return string
function CustomSampler:name()
    return self._name
end

return CustomSampler
