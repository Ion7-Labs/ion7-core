--- @module ffi_gen.stats
--- @author  ion7 / generate_ffi.lua
---
--- Lightweight coverage tracker. Counts how many public functions each header
--- exposes, what the script chose to skip and why, and which symbols fell
--- through into the catch-all `misc` group.
---
--- The `report()` method renders a human-readable summary printed at the end
--- of every `generate_ffi.lua` run; it is the single source of truth for
--- noticing when an upstream change in llama.cpp introduces new symbol
--- prefixes that we have not yet routed in `ffi_gen.groups`.

local util = require "ffi_gen.util"

local string_format = string.format
local string_rep    = string.rep
local table_concat  = table.concat
local table_insert  = table.insert
local math_min      = math.min

local Stats = {}
Stats.__index = Stats

--- @class ffi_gen.stats.Skip
--- @field name   string  Symbol name that was dropped.
--- @field header string  Originating header (basename only).
--- @field reason string  Human-readable reason (e.g. `"variadic"`, `"static inline"`).

--- @class ffi_gen.stats.Misc
--- @field name string  Symbol name that did not match any prefix group.
--- @field ns   string  Namespace it landed in (`"llama"` / `"ggml"` / `"gguf"`).

--- Create a fresh, empty stats accumulator.
--- @return ffi_gen.stats
function Stats.new()
  return setmetatable({
    --- Per-header public function counts (header basename → count).
    --- @type table<string, integer>
    fns_per_header = {},
    --- Functions explicitly dropped (variadic, static inline, render fail).
    --- @type ffi_gen.stats.Skip[]
    skipped = {},
    --- Functions that matched no prefix group and landed in `misc`.
    --- @type ffi_gen.stats.Misc[]
    misc_fns = {},
  }, Stats)
end

--- Record a skipped function with its reason.
--- @param name   string Symbol name.
--- @param header string Originating header path (basename will be extracted).
--- @param reason string Why it was dropped.
function Stats:add_skip(name, header, reason)
  table_insert(self.skipped, {
    name   = name,
    header = util.basename(header),
    reason = reason,
  })
end

--- Record a function that matched no prefix group.
--- @param name string Symbol name.
--- @param ns   string Namespace (`"llama"`, `"ggml"`, or `"gguf"`).
function Stats:add_misc(name, ns)
  table_insert(self.misc_fns, { name = name, ns = ns })
end

--- Render a human-readable coverage report.
---
--- Output sections:
---   1. Per-header function counts and grand total.
---   2. Skip reasons grouped by category, with up to 5 examples each.
---   3. Misc bucket warnings (up to 15 examples) — these typically signal a
---      prefix group that needs to be added in `ffi_gen.groups`.
---
--- @return string
function Stats:report()
  local lines, n = {}, 0
  local function push(s) n = n + 1; lines[n] = s end

  push("")
  push("═══════════════════════════════════════════════════════════")
  push("  COVERAGE REPORT")
  push("═══════════════════════════════════════════════════════════")

  -- Per-header function counts.
  local total = 0
  for _, header in ipairs(util.sorted_keys(self.fns_per_header)) do
    local count = self.fns_per_header[header]
    push(string_format("  %-30s %5d public functions", header, count))
    total = total + count
  end
  push("  " .. string_rep("─", 36))
  push(string_format("  %-30s %5d", "TOTAL", total))
  push("")

  -- Skip reasons (grouped).
  if #self.skipped > 0 then
    push(string_format("  ⚠ %d function(s) skipped:", #self.skipped))
    local by_reason = {}
    for _, sk in ipairs(self.skipped) do
      by_reason[sk.reason] = by_reason[sk.reason] or {}
      table_insert(by_reason[sk.reason], sk.header .. "::" .. sk.name)
    end
    for _, reason in ipairs(util.sorted_keys(by_reason)) do
      local items = by_reason[reason]
      push(string_format("    [%s] (%d)", reason, #items))
      for i = 1, math_min(5, #items) do
        push("      - " .. items[i])
      end
      if #items > 5 then
        push(string_format("      ... +%d more", #items - 5))
      end
    end
    push("")
  end

  -- Misc bucket warnings.
  if #self.misc_fns > 0 then
    push(string_format("  ⚠ %d function(s) landed in 'misc' ".. "(add their prefix to *_GROUPS to route them):", #self.misc_fns))
    for i = 1, math_min(15, #self.misc_fns) do
      local f = self.misc_fns[i]
      push(string_format("      - %s::%s", f.ns, f.name))
    end
    if #self.misc_fns > 15 then
      push(string_format("      ... +%d more", #self.misc_fns - 15))
    end
    push("")
  end

  push("═══════════════════════════════════════════════════════════")
  return table_concat(lines, "\n")
end

return Stats
