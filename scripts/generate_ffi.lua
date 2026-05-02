#!/usr/bin/env luajit
--[[
  generate_ffi.lua — entry point for the LuaJIT FFI binding generator.

  Auto-generates ion7-core's FFI declarations for `llama.cpp`, `ggml` and
  `gguf` from their public C headers, by feeding clang's `-ast-dump=json`
  output into a pure-Lua emitter.

  Headers parsed by default (~95% of ion7 needs):
    llama.h, ggml.h, ggml-alloc.h, ggml-backend.h, ggml-cpu.h,
    ggml-blas.h, ggml-opt.h, gguf.h, ggml-vulkan.h.

  Opt-in extras:
    --include-cuda / --include-metal / --include-opencl / --include-sycl
    --include-rpc / --include-webgpu / --include-all-backends
    --no-vulkan to drop the default Vulkan binding

  Other flags:
    --llama-cpp-dir <path>   override the vendored llama.cpp root
    --output-dir <path>      override the destination (default src/ion7/core/ffi)
    --clang <path>           override the clang executable path
    --audit                  parse + report only, do not write files
    -h / --help              print this help

  Prerequisites:
    - LuaJIT 2.1+ (any LuaJIT can run this; uses no LuaJIT-only stdlib)
    - clang on PATH, or via $ION7_CLANG, or installed at C:\Program Files\LLVM
    - ion7.vendor.json available at src/ion7/vendor/json.lua (already vendored)

  Architecture overview:
    - ffi_gen.groups   : prefix-to-bucket configuration (the "schema").
    - ffi_gen.util     : path/file helpers, hot-path locals, IS_WINDOWS.
    - ffi_gen.stats    : coverage report accumulator.
    - ffi_gen.clang    : spawn clang, decode JSON.
    - ffi_gen.ast      : walk the AST, extract Decls, render C spellings.
    - ffi_gen.emit     : write the Lua FFI files.

  This file is the orchestrator: it parses CLI flags, builds the include
  list, drives the pipeline and prints the final summary. Domain logic
  belongs in the modules above.
]]

-- ── Module path bootstrap ──────────────────────────────────────────────────

--- Return the directory holding the currently-executing script (with a
--- trailing slash). Used to resolve the `ffi_gen/*` modules and the vendored
--- `ion7.vendor.json` regardless of the caller's CWD.
--- @return string
local function script_dir()
  local p = arg[0] or ""
  return (p:match("(.*[/\\])")) or "./"
end

local SCRIPT_DIR   = script_dir()
local PROJECT_ROOT = SCRIPT_DIR .. "../"

package.path = SCRIPT_DIR   .. "?.lua;"
            .. SCRIPT_DIR   .. "?/init.lua;"
            .. PROJECT_ROOT .. "src/?.lua;"
            .. PROJECT_ROOT .. "src/?/init.lua;"
            .. package.path

local groups = require "ffi_gen.groups"
local util   = require "ffi_gen.util"
local Stats  = require "ffi_gen.stats"
local clang  = require "ffi_gen.clang"
local astlib = require "ffi_gen.ast"
local emit   = require "ffi_gen.emit"

-- ── CLI parsing ────────────────────────────────────────────────────────────

--- Parse the command-line arguments into a flat options table.
--- @param  argv string[] Raw `arg` table passed by LuaJIT.
--- @return table         Parsed options.
local function parse_args(argv)
  local opts = {
    llama_cpp_dir        = PROJECT_ROOT .. "vendor/llama.cpp",
    output_dir           = PROJECT_ROOT .. "src/ion7/core/ffi",
    include_vulkan       = true,
    include_cuda         = false,
    include_metal        = false,
    include_opencl       = false,
    include_sycl         = false,
    include_rpc          = false,
    include_webgpu       = false,
    include_all_backends = false,
    audit                = false,
    clang                = nil,
  }

  local i = 1
  while i <= #argv do
    local a = argv[i]
    if     a == "--llama-cpp-dir" then opts.llama_cpp_dir = argv[i+1]; i = i + 2
    elseif a == "--output-dir"    then opts.output_dir    = argv[i+1]; i = i + 2
    elseif a == "--clang"         then opts.clang         = argv[i+1]; i = i + 2
    elseif a == "--include-vulkan" then opts.include_vulkan = true; i = i + 1
    elseif a == "--no-vulkan"      then opts.include_vulkan = false; i = i + 1
    elseif a == "--include-cuda"   then opts.include_cuda  = true; i = i + 1
    elseif a == "--include-metal"  then opts.include_metal = true; i = i + 1
    elseif a == "--include-opencl" then opts.include_opencl = true; i = i + 1
    elseif a == "--include-sycl"   then opts.include_sycl  = true; i = i + 1
    elseif a == "--include-rpc"    then opts.include_rpc   = true; i = i + 1
    elseif a == "--include-webgpu" then opts.include_webgpu = true; i = i + 1
    elseif a == "--include-all-backends" then opts.include_all_backends = true; i = i + 1
    elseif a == "--audit"          then opts.audit = true; i = i + 1
    elseif a == "-h" or a == "--help" then
      io.write("Usage: luajit scripts/generate_ffi.lua [options]\n")
      io.write("Options:\n")
      io.write("  --llama-cpp-dir <path>   override vendored llama.cpp root\n")
      io.write("  --output-dir <path>      override Lua FFI destination\n")
      io.write("  --clang <path>           override the clang executable path\n")
      io.write("  --include-{cuda,metal,opencl,sycl,rpc,webgpu}\n")
      io.write("                           opt-in additional backend bindings\n")
      io.write("  --include-all-backends   shorthand for all of the above\n")
      io.write("  --no-vulkan              drop the default Vulkan binding\n")
      io.write("  --audit                  parse + report only, no files written\n")
      os.exit(0)
    else
      io.stderr:write("ERROR: unknown option: " .. a .. "\n")
      os.exit(2)
    end
  end

  if opts.include_all_backends then
    opts.include_cuda   = true
    opts.include_metal  = true
    opts.include_opencl = true
    opts.include_sycl   = true
    opts.include_rpc    = true
    opts.include_webgpu = true
  end
  return opts
end

-- ── Header set assembly ────────────────────────────────────────────────────

--- Build the absolute paths of every header to feed to clang, based on the
--- requested defaults + opt-ins.
---
--- @param  opts        table  Parsed CLI options.
--- @param  llama_root  string Resolved absolute path to the llama.cpp root.
--- @return string[]           Absolute header paths.
local function collect_headers(opts, llama_root)
  local out, n = {}, 0
  for _, rel in ipairs(groups.DEFAULT_HEADERS) do
    n = n + 1
    out[n] = llama_root .. "/" .. rel
  end
  local opt_flags = {
    vulkan = opts.include_vulkan, cuda = opts.include_cuda,
    metal  = opts.include_metal,  opencl = opts.include_opencl,
    sycl   = opts.include_sycl,   rpc    = opts.include_rpc,
    webgpu = opts.include_webgpu,
  }
  for key, rel in pairs(groups.OPTIONAL_HEADERS) do
    if opt_flags[key] then
      n = n + 1
      out[n] = llama_root .. "/" .. rel
    end
  end
  return out
end

--- Filter callback for `ffi_gen.ast.walk` — decides whether a `loc.file`
--- should be included in the extraction. We keep llama.h, anything starting
--- with ggml, and gguf.h (the latter is what the .py was missing).
---
--- @param  src string Source file path coming from `loc.file`.
--- @return boolean
local function target_filter(src)
  local n = util.basename(src)
  return n == "llama.h"
      or util.starts_with(n, "ggml")
      or n == "gguf.h"
end

-- ── Pipeline ───────────────────────────────────────────────────────────────

--- Drive the whole generation pipeline. Returns nothing; prints progress and
--- the coverage report to stdout, and writes files unless `opts.audit`.
---
--- @param opts table Parsed CLI options.
local function run(opts)
  local llama_root = util.normalize_path(opts.llama_cpp_dir)
  if not util.file_exists(llama_root .. "/include/llama.h") then
    io.stderr:write(string.format(
      "ERROR: %s/include/llama.h not found. Vendor llama.cpp first.\n",
      llama_root))
    os.exit(1)
  end

  local output_dir = util.normalize_path(opts.output_dir)
  util.mkdir_p(output_dir)

  local headers = collect_headers(opts, llama_root)
  local missing = {}
  for _, h in ipairs(headers) do
    if not util.file_exists(h) then missing[#missing+1] = h end
  end
  if #missing > 0 then
    for _, h in ipairs(missing) do
      io.stderr:write("WARN: missing header (skipped): " .. h .. "\n")
    end
    -- Drop the missing ones from the include list so clang does not error.
    local kept, kn = {}, 0
    for _, h in ipairs(headers) do
      if util.file_exists(h) then kn = kn + 1; kept[kn] = h end
    end
    headers = kept
  end

  -- Single combined parse: huge speedup over per-header invocations.
  print(string.format("[generate_ffi] parsing %d headers (combined) from %s",
                      #headers, llama_root))
  for _, h in ipairs(headers) do
    print("  · " .. h:sub(#llama_root + 2))
  end

  local include_paths = {}
  for i, sub in ipairs(groups.INCLUDE_SUBDIRS) do
    include_paths[i] = llama_root .. "/" .. sub
  end

  local clang_path = clang.find_clang(opts.clang)
  local combined = clang.build_combined_header(headers)

  local t_parse = os.clock()
  local ast, err = clang.dump_ast(clang_path, combined, include_paths,
                                  groups.CLANG_DEFINES)
  os.remove(combined)
  if not ast then
    io.stderr:write("ERROR: " .. err .. "\n"); os.exit(1)
  end
  local parse_dt = os.clock() - t_parse

  -- Walk + dedup.
  local stats = Stats.new()
  local all_decls = {}
  local seen_fns = {}
  astlib.walk(ast, target_filter, groups.IGNORED_MACROS,
              stats, all_decls, seen_fns)

  -- Split types vs functions.
  local types, fns, tn, fn = {}, {}, 0, 0
  for i = 1, #all_decls do
    local d = all_decls[i]
    if d.kind == "function" then fn = fn + 1; fns[fn] = d
    else                         tn = tn + 1; types[tn] = d
    end
  end
  print(string.format(
    "[generate_ffi] parsed in %.2fs — %d decls (%d types, %d functions)",
    parse_dt, #all_decls, tn, fn))

  -- Group functions by namespace + bucket.
  local llama_groups, ggml_groups, gguf_groups = {}, {}, {}
  for i = 1, #fns do
    local f = fns[i]
    local name = f.name
    if util.starts_with(name, "llama_") then
      local g, matched = astlib.assign_group(name, groups.LLAMA_GROUPS)
      llama_groups[g] = llama_groups[g] or {}
      table.insert(llama_groups[g], f)
      if not matched then stats:add_misc(name, "llama") end
    elseif util.starts_with(name, "gguf_") then
      local g, matched = astlib.assign_group(name, groups.GGUF_GROUPS)
      gguf_groups[g] = gguf_groups[g] or {}
      table.insert(gguf_groups[g], f)
      if not matched then stats:add_misc(name, "gguf") end
    elseif util.starts_with(name, "ggml_") then
      local g, matched = astlib.assign_group(name, groups.GGML_GROUPS)
      ggml_groups[g] = ggml_groups[g] or {}
      table.insert(ggml_groups[g], f)
      if not matched then stats:add_misc(name, "ggml") end
    else
      stats:add_skip(name, f.source_file,
                     "unknown namespace (not llama_/ggml_/gguf_)")
    end
  end

  -- Emit (unless --audit).
  if opts.audit then
    print("\n[AUDIT MODE] no files written.")
  else
    emit.write_types(types, output_dir)
    print("[generate_ffi] wrote " .. output_dir .. "/types.lua")
    for _, g in ipairs(util.sorted_keys(llama_groups)) do
      emit.write_group(g, llama_groups[g], "llama", output_dir)
      print(string.format("  llama/%s.lua: %d functions", g, #llama_groups[g]))
    end
    for _, g in ipairs(util.sorted_keys(ggml_groups)) do
      emit.write_group(g, ggml_groups[g], "ggml", output_dir)
      print(string.format("  ggml/%s.lua: %d functions", g, #ggml_groups[g]))
    end
    for _, g in ipairs(util.sorted_keys(gguf_groups)) do
      emit.write_group(g, gguf_groups[g], "gguf", output_dir)
      print(string.format("  gguf/%s.lua: %d functions", g, #gguf_groups[g]))
    end
    emit.write_loader(output_dir, next(ggml_groups) ~= nil, opts.include_vulkan)
    print("[generate_ffi] wrote " .. output_dir .. "/loader.lua")
  end

  -- Coverage report.
  print(stats:report())

  local total_public = 0
  for _, n in pairs(stats.fns_per_header) do total_public = total_public + n end
  local exported = #fns
  local pct = (total_public > 0) and (exported / total_public * 100) or 0
  print(string.format("  Extraction coverage: %d/%d functions (%.1f %%)",
                      exported, total_public, pct))

  if not opts.audit then
    local function sum_groups(g) local s = 0; for _, v in pairs(g) do s = s + #v end; return s end
    local function count_keys(g) local s = 0; for _ in pairs(g) do s = s + 1 end; return s end
    local wl  = sum_groups(llama_groups)
    local wg  = sum_groups(ggml_groups)
    local wgg = sum_groups(gguf_groups)
    print(string.format("  Write coverage     : %d/%d functions exposed in Lua",
                        wl + wg + wgg, exported))
    print(string.format("    llama : %4d in %2d files", wl, count_keys(llama_groups)))
    print(string.format("    ggml  : %4d in %2d files", wg, count_keys(ggml_groups)))
    print(string.format("    gguf  : %4d in %2d files", wgg, count_keys(gguf_groups)))
    print("\n✓ Done. Output written to " .. output_dir)
    print("  Usage: require 'ion7.core.ffi.llama.context'  (lazy load by domain)")
  else
    print("\n✓ Audit done (no files modified).")
  end
end

run(parse_args(arg))
