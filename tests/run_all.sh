#!/usr/bin/env bash
# Run the full ion7-core test suite.
#
# Usage:
#   ION7_MODEL=/path/to/model.gguf bash tests/run_all.sh
#
# Optional:
#   ION7_LIB_DIR=/path/to/llama.cpp/build/bin
#   ION7_EMBED=/path/to/embed.gguf
#   ION7_LORA=/path/to/adapter.gguf

set -e

PASS=0
FAIL=0

run_suite() {
    local name="$1"
    local file="$2"
    printf "\n\033[1m══ %-40s \033[0m%s\n" "$name" "$(printf '═%.0s' {1..18})"
    if ION7_MODEL="$ION7_MODEL" \
       ION7_LIB_DIR="$ION7_LIB_DIR" \
       ION7_EMBED="$ION7_EMBED" \
       ION7_LORA="$ION7_LORA" \
       luajit "$file"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

# ── Suite 1: Pure Lua (no model) ──────────────────────────────────────────────
run_suite "Pure Lua - no model required" tests/test_pure.lua

# ── Suite 2: Model-dependent ──────────────────────────────────────────────────
if [ -n "$ION7_MODEL" ]; then
    run_suite "Model - full API coverage" tests/test_model.lua
else
    printf "\n  \033[33m[SKIP]\033[0m Model tests - set ION7_MODEL=/path/to/model.gguf\n"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"
printf "  Suites: \033[32m%d passed\033[0m" "$PASS"
[ "$FAIL" -gt 0 ] && printf "  \033[31m%d FAILED\033[0m" "$FAIL"
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"

[ "$FAIL" -eq 0 ]
