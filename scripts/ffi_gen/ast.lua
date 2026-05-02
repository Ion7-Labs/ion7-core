--- @module ffi_gen.ast
--- @author  ion7 / generate_ffi.lua
---
--- AST walking and per-kind C-spelling rendering.
---
--- Consumes the JSON tree produced by clang (`-Xclang -ast-dump=json`) and
--- yields a flat list of `Decl` records ready for the emit phase.
---
--- The AST format used by clang has two quirks the walker has to handle:
---   1. The `loc.file` field is only emitted on the FIRST node from each
---      source file, in document order. Subsequent nodes inherit that file
---      until a different one shows up. We track `current_file` as we walk.
---   2. clang adds implicit builtin declarations (`__int128_t`, `size_t`,
---      `__NSConstantString` ...) that we want to skip. They are flagged
---      with `isImplicit = true`.

local util = require "ffi_gen.util"

local string_format = string.format
local string_match  = string.match
local string_find   = string.find
local string_sub    = string.sub
local table_concat  = table.concat

local M = {}

--- @class ffi_gen.ast.Decl
--- @field kind         "function"|"struct"|"struct_fwd"|"typedef"|"enum"
--- @field name         string  Symbol name.
--- @field spelling     string  Reconstructed C declaration (single statement).
--- @field source_file  string  Originating header (full path).
--- @field extent_start integer Source line for stable ordering.

-- ── qualType helpers (used by every renderer below) ───────────────────────

--- Strip C qualifiers and macros that LuaJIT's `ffi.cdef` parser does not
--- understand. Currently:
---
---   - `restrict` / `__restrict` / `__restrict__` (compiler hints with no
---     ABI relevance — should never appear in our output, but we belt-and-
---     brace it because clang sometimes resurfaces them despite the
---     `-DGGML_RESTRICT=` neutralisation).
---
--- Returns the input unchanged if there is nothing to strip.
---
--- @param  qt string|nil The raw qualType string from the AST.
--- @return string        Cleaned type spelling.
local function clean_qualtype(qt)
  if not qt or qt == "" then return qt or "" end
  qt = qt:gsub("%s*__restrict__%s*", " ")
         :gsub("%s*__restrict%s*",   " ")
         :gsub("%s*restrict%s*",     " ")
         :gsub("%s+$", "")
         :gsub("%s%s+",  " ")
  return qt
end

--- Pull the qualified type spelling from a node, defaulting to `fallback`
--- when the field is missing. Wraps `clean_qualtype` so every render
--- function gets stripped strings without sprinkling the call sites.
---
--- @param  node     table   AST node with optional `node.type.qualType`.
--- @param  fallback string  Type to return when the qualType is absent.
--- @return string
local function qualtype_of(node, fallback)
  local qt = node.type and node.type.qualType
  if not qt then return fallback end
  return clean_qualtype(qt)
end

-- ── Per-kind C spelling renderers ──────────────────────────────────────────

--- Reconstruct a function prototype from a `FunctionDecl` node.
--- Format: `RetType name(arg1Type arg1Name, arg2Type arg2Name);`
---
--- The return type is extracted from `node.type.qualType` (which has the
--- shape `RetType(arg1Type, arg2Type)`); we slice off everything before the
--- first `(`. This is robust for the public surface of llama.cpp — function
--- pointers as return types are not used there.
---
--- Argument names are pulled from each `ParmVarDecl` child; types come from
--- `child.type.qualType` to preserve `const`, pointer levels, etc.
---
--- @param  node table A `FunctionDecl` AST node.
--- @return string     C prototype, terminated with `;`.
local function render_function(node)
  local ret_full = qualtype_of(node, "void")
  local paren_pos = string_find(ret_full, "%(")
  local ret = paren_pos and util.trim(string_sub(ret_full, 1, paren_pos - 1)) or ret_full

  local params, n = {}, 0
  for _, child in ipairs(node.inner or {}) do
    if child.kind == "ParmVarDecl" then
      n = n + 1
      local ptype = qualtype_of(child, "void")
      local pname = child.name or ""
      if pname ~= "" then
        params[n] = ptype .. " " .. pname
      else
        params[n] = ptype
      end
    end
  end
  local sig_args = (n > 0) and table_concat(params, ", ") or "void"
  return string_format("%s %s(%s);", ret, node.name, sig_args)
end

--- Render a single C field declaration prefixed with `indent` spaces.
---
--- Three syntactic shapes are recognised :
---
---   - Plain type            → `<indent><type> <name>;`
---   - Array type            → `<indent><base> <name>[<count>];`
---     (clang gives `int [4]`, we move the brackets next to the name).
---   - Function-pointer type → `<indent><RetType> (*<name>)(<args>);`
---     (clang gives `RetType (*)(args)`, we insert the name into the
---      parens — same shape as a function-pointer typedef).
---
--- @param  ftype  string A C type spelling.
--- @param  fname  string Field name.
--- @param  indent string Leading whitespace for the line.
--- @return string        A single-line field declaration ending with `;`.
local function render_field(ftype, fname, indent)
  -- Function-pointer field : insert the field name into the `(*)` slot.
  if ftype:find("%(%*%)") then
    return string_format("%s%s;", indent,
                         (ftype:gsub("%(%*%)", "(*" .. fname .. ")", 1)))
  end
  -- Array field : move brackets from the type next to the field name.
  local base, arr = string_match(ftype, "^(.-)%s*%[(.-)%]$")
  if base then
    return string_format("%s%s %s[%s];", indent, util.trim(base), fname, arr)
  end
  return string_format("%s%s %s;", indent, ftype, fname)
end

--- Render the body (fields only) of an anonymous nested record into an
--- indented multi-line string. Used by `render_struct` for both the fully
--- anonymous case (`union { ... };`) and the named-instance case
--- (`struct { ... } name;`).
---
--- @param  record table  An anonymous `RecordDecl` node.
--- @return string        Indented body with leading `\n` and no trailing newline.
local function render_anon_body(record)
  local sub, sn = {}, 0
  for _, gch in ipairs(record.inner or {}) do
    if gch.kind == "FieldDecl" and gch.name and gch.name ~= "" then
      local gtype = qualtype_of(gch, "int")
      sn = sn + 1
      sub[sn] = render_field(gtype, gch.name, "        ")
    end
  end
  return table_concat(sub, "\n")
end

--- Reconstruct a struct definition from a `RecordDecl` node.
---
--- clang spreads several shapes across the inner array; we walk it with
--- an explicit index so we can pair adjacent items :
---
---   - `FieldDecl` (named)         → render as a regular field.
---   - `RecordDecl` (no name)      → an anonymous nested union/struct.
---     Look at the IMMEDIATELY following item :
---       - if it's a `FieldDecl` with EMPTY name → fully anonymous,
---         emit `union { ... };` and consume both items.
---       - if it's a `FieldDecl` with a NAME and a type pointing back
---         at the anonymous record → emit `struct { ... } name;` and
---         consume both items.
---       - otherwise → emit the body alone (rare safety net).
---   - `IndirectFieldDecl`         → clang shortcut entries for anonymous
---                                   union members. SKIP (already
---                                   accessible via the inlined body).
---
--- @param  node table A `RecordDecl` node with `completeDefinition = true`.
--- @return string     C struct body, terminated with `;`.
local function render_struct(node)
  local inner = node.inner or {}
  local fields, n = {}, 0
  local i = 1
  while i <= #inner do
    local child = inner[i]
    local kind  = child.kind

    if kind == "FieldDecl" then
      local fname = child.name or ""
      if fname ~= "" then
        local ftype = qualtype_of(child, "int")
        n = n + 1
        fields[n] = render_field(ftype, fname, "    ")
      end
      -- Anonymous FieldDecl with no preceding RecordDecl : extremely
      -- rare and would be rendered with the mangled C++ type, so we skip.

    elseif kind == "RecordDecl" and (not child.name or child.name == "") then
      local tag  = child.tagUsed or "struct"
      local body = render_anon_body(child)
      local nxt  = inner[i + 1]

      if nxt and nxt.kind == "FieldDecl" then
        local nname = nxt.name or ""
        if nname == "" then
          -- Fully anonymous : `union { ... };`
          n = n + 1
          fields[n] = string_format("    %s {\n%s\n    };", tag, body)
          i = i + 1  -- consume the FieldDecl placeholder
        else
          -- Named instance of an anonymous record : `struct { ... } name;`
          n = n + 1
          fields[n] = string_format("    %s {\n%s\n    } %s;", tag, body, nname)
          i = i + 1  -- consume the FieldDecl that named the record
        end
      else
        -- Anonymous record with no following FieldDecl : emit body alone.
        n = n + 1
        fields[n] = string_format("    %s {\n%s\n    };", tag, body)
      end
    end
    -- IndirectFieldDecl is intentionally skipped.

    i = i + 1
  end
  if n == 0 then
    return string_format("struct %s;", node.name)
  end
  -- Emit a self-typedef alongside the struct definition so that field
  -- types using the bare tag name (`Foo *` rather than `struct Foo *`)
  -- resolve under `ffi.cdef`. clang's `qualType` strings normalise away
  -- the `struct` keyword once the matching typedef exists, but if we
  -- only emitted the bare struct definition the cdef parser would reject
  -- those bare-tag references. C99 allows duplicate identical typedefs,
  -- so the extra alias is harmless even when the upstream header already
  -- declares it.
  return string_format("struct %s {\n%s\n};\ntypedef struct %s %s;",
                       node.name, table_concat(fields, "\n"),
                       node.name, node.name)
end

--- Reconstruct a `typedef X Y;` from a `TypedefDecl` node.
---
--- Handles three syntactic shapes that all show up in llama.h / ggml.h :
---
---   - Plain typedef           → `typedef <underlying> <name>;`
---   - Function-pointer typedef → clang gives `RetType (*)(args)`; the
---                                name must be inserted INSIDE the parens,
---                                yielding `typedef RetType (*name)(args);`
---   - Array typedef            → clang gives `Base [N]`; the brackets
---                                must move next to the name, yielding
---                                `typedef Base name[N];`
---
--- @param  node table A `TypedefDecl` node.
--- @return string     A valid C typedef declaration, terminated with `;`.
local function render_typedef(node)
  local underlying = qualtype_of(node, "int")
  local name       = node.name

  -- Function-pointer typedef : the `(*)` slot must receive the name.
  if underlying:find("%(%*%)") then
    return string_format("typedef %s;",
                         (underlying:gsub("%(%*%)", "(*" .. name .. ")", 1)))
  end

  -- Array typedef : `int [10]` → `int name[10]`.
  local base, arr = string_match(underlying, "^(.-)%s*%[(.-)%]$")
  if base then
    return string_format("typedef %s %s[%s];", util.trim(base), name, arr)
  end

  return string_format("typedef %s %s;", underlying, name)
end

--- Reconstruct an enum definition from an `EnumDecl` node.
---
--- For `EnumConstantDecl` nodes that have an explicit value, clang nests
--- a `ConstantExpr` (or similar) under `inner[1]` carrying the numeric
--- string in `value`. For implicit auto-incremented constants, `inner`
--- is empty — in that case we EMIT THE NAME ALONE, letting the C
--- compiler (or `ffi.cdef` parser) resume the auto-increment cleanly.
--- This avoids a class of bugs where every implicit member would be
--- rendered with the wrong literal value (e.g. all 0).
---
--- @param  node table An `EnumDecl` node.
--- @return string     `enum <name> { ... };`.
local function render_enum(node)
  local consts, n = {}, 0
  for _, child in ipairs(node.inner or {}) do
    if child.kind == "EnumConstantDecl" then
      n = n + 1
      local explicit_value
      if child.inner and child.inner[1] and child.inner[1].value ~= nil then
        explicit_value = tostring(child.inner[1].value)
      end
      if explicit_value then
        consts[n] = string_format("    %s = %s,", child.name, explicit_value)
      else
        -- Implicit value : let the C parser auto-increment from the
        -- previous member (defaults to 0 for the first).
        consts[n] = string_format("    %s,", child.name)
      end
    end
  end
  if n == 0 then
    return string_format("enum %s {};", node.name or "")
  end
  return string_format("enum %s {\n%s\n};", node.name or "", table_concat(consts, "\n"))
end

-- ── Decl extraction ────────────────────────────────────────────────────────

--- Convert one AST node into a `Decl` record (or nil if it should be skipped).
---
--- Skip rules for `FunctionDecl`:
---   - `static` storage class → not exported, dropped.
---   - `variadic = true`     → painful in FFI, dropped.
---   - render returned nil   → broken, dropped (logged).
---
--- For `RecordDecl`, lack of `completeDefinition` produces a forward-decl
--- entry; the dedup pass in `emit.render_types_lua` keeps the complete
--- definition over the forward when both exist.
---
--- @param  node        table   AST node.
--- @param  source_file string  Originating header path.
--- @param  ignored_macros table<string, true> Names to drop unconditionally.
--- @param  stats       ffi_gen.stats Coverage tracker (for skip reasons).
--- @return ffi_gen.ast.Decl|nil
local function extract_decl(node, source_file, ignored_macros, stats)
  local name = node.name
  if not name or name == "" or ignored_macros[name] then return nil end

  local kind = node.kind
  local line = (node.loc and node.loc.line)
            or (node.range and node.range.begin and node.range.begin.line)
            or 0

  if kind == "FunctionDecl" then
    if node.storageClass == "static" then
      stats:add_skip(name, source_file, "static inline")
      return nil
    end
    if node.variadic then
      stats:add_skip(name, source_file, "variadic")
      return nil
    end
    local sp = render_function(node)
    if sp then
      return {
        kind = "function", name = name, spelling = sp,
        source_file = source_file, extent_start = line,
      }
    end
    stats:add_skip(name, source_file, "render failed")
    return nil

  elseif kind == "RecordDecl" then
    if node.completeDefinition ~= true then
      return {
        kind = "struct_fwd", name = name,
        spelling = "struct " .. name .. ";",
        source_file = source_file, extent_start = line,
      }
    end
    return {
      kind = "struct", name = name, spelling = render_struct(node),
      source_file = source_file, extent_start = line,
    }

  elseif kind == "TypedefDecl" then
    return {
      kind = "typedef", name = name, spelling = render_typedef(node),
      source_file = source_file, extent_start = line,
    }

  elseif kind == "EnumDecl" then
    return {
      kind = "enum", name = name, spelling = render_enum(node),
      source_file = source_file, extent_start = line,
    }
  end
  return nil
end

-- ── Walking the translation unit ───────────────────────────────────────────

--- Walk the top-level `inner` of a `TranslationUnitDecl`, collect Decls into
--- `decls_out` and update `stats`.
---
--- Function-level deduplication: even with the combined-header strategy,
--- the same function may appear in two files that the include-guards do not
--- shield (rare but it happens with `extern` redeclarations). The
--- `seen_fns` set is the single source of truth for "have I emitted this
--- name already". Types are NOT deduplicated here — that pass is done at
--- emit time, where the renderer can prefer the full struct over a forward
--- declaration.
---
--- @param  ast            table              Decoded TU root from clang.
--- @param  target_filter  fun(path:string):boolean Returns true if a header should be kept.
--- @param  ignored_macros table<string, true> Names to drop unconditionally.
--- @param  stats          ffi_gen.stats      Coverage tracker.
--- @param  decls_out      ffi_gen.ast.Decl[] Output array (mutated in place).
--- @param  seen_fns       table<string, true> Cross-call dedup set for function names.
function M.walk(ast, target_filter, ignored_macros, stats, decls_out, seen_fns)
  local current_file = nil
  local out_n = #decls_out
  for _, node in ipairs(ast.inner or {}) do
    -- Update current_file when this node carries a fresh `loc.file`.
    if node.loc and node.loc.file then
      current_file = util.normalize_path(node.loc.file)
    end
    -- Skip clang-injected builtins like `__int128_t`, `size_t`, ...
    if node.isImplicit then
      -- noop
    elseif current_file and target_filter(current_file) then
      if node.kind == "FunctionDecl" and node.name and node.name ~= "" then
        if not seen_fns[node.name] then
          seen_fns[node.name] = true
          local hn = util.basename(current_file)
          stats.fns_per_header[hn] = (stats.fns_per_header[hn] or 0) + 1
          local decl = extract_decl(node, current_file, ignored_macros, stats)
          if decl then
            out_n = out_n + 1
            decls_out[out_n] = decl
          end
        end
      else
        local decl = extract_decl(node, current_file, ignored_macros, stats)
        if decl then
          out_n = out_n + 1
          decls_out[out_n] = decl
        end
      end
    end
  end
end

-- ── Group assignment ───────────────────────────────────────────────────────

--- Find the first group whose prefix list matches `name`. The order of
--- `groups` matters — the FIRST matching prefix wins.
---
--- @param  name   string Function name to classify.
--- @param  groups { [1]: string, [2]: string[] }[] Group config (see `ffi_gen.groups`).
--- @return string  group_name The group bucket (`"misc"` if no prefix matched).
--- @return boolean matched    False when the name fell through into `misc`.
function M.assign_group(name, groups)
  for i = 1, #groups do
    local entry = groups[i]
    local prefixes = entry[2]
    for j = 1, #prefixes do
      local p = prefixes[j]
      if string_sub(name, 1, #p) == p then
        return entry[1], true
      end
    end
  end
  return "misc", false
end

return M
