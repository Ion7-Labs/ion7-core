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
  local ret_full = (node.type and node.type.qualType) or "void"
  local paren_pos = string_find(ret_full, "%(")
  local ret = paren_pos and util.trim(string_sub(ret_full, 1, paren_pos - 1)) or ret_full

  local params, n = {}, 0
  for _, child in ipairs(node.inner or {}) do
    if child.kind == "ParmVarDecl" then
      n = n + 1
      local ptype = (child.type and child.type.qualType) or "void"
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

--- Reconstruct a struct definition from a `RecordDecl` node.
---
--- Each `FieldDecl` child becomes one indented field. Array types
--- (`int [4]`) are rewritten to the conventional placement (`int name[4]`).
---
--- @param  node table A `RecordDecl` node with `completeDefinition = true`.
--- @return string     C struct body, terminated with `;`.
local function render_struct(node)
  local fields, n = {}, 0
  for _, child in ipairs(node.inner or {}) do
    if child.kind == "FieldDecl" then
      n = n + 1
      local ftype = (child.type and child.type.qualType) or "int"
      local fname = child.name or "_unnamed"
      -- Move array brackets from the type onto the field name.
      local base, arr = string_match(ftype, "^(.-)%s*%[(.-)%]$")
      if base then
        fields[n] = string_format("    %s %s[%s];", util.trim(base), fname, arr)
      else
        fields[n] = string_format("    %s %s;", ftype, fname)
      end
    end
  end
  if n == 0 then
    return string_format("struct %s;", node.name)
  end
  return string_format("struct %s {\n%s\n};", node.name, table_concat(fields, "\n"))
end

--- Reconstruct a `typedef X Y;` from a `TypedefDecl` node.
--- @param  node table A `TypedefDecl` node.
--- @return string     `typedef <underlying> <name>;`.
local function render_typedef(node)
  local underlying = (node.type and node.type.qualType) or "int"
  return string_format("typedef %s %s;", underlying, node.name)
end

--- Reconstruct an enum definition from an `EnumDecl` node.
---
--- The numeric value of each `EnumConstantDecl` lives in `inner[1].value`
--- (a string in the JSON, since clang preserves user-written radix/sign).
---
--- @param  node table An `EnumDecl` node.
--- @return string     `enum <name> { ... };`.
local function render_enum(node)
  local consts, n = {}, 0
  for _, child in ipairs(node.inner or {}) do
    if child.kind == "EnumConstantDecl" then
      n = n + 1
      local val = "0"
      if child.inner and child.inner[1] then
        local v = child.inner[1].value
        if v ~= nil then val = tostring(v) end
      end
      consts[n] = string_format("    %s = %s,", child.name, val)
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
