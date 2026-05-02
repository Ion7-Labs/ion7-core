#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run the full ion7-core test suite.
#
# The suite is split into many small files numbered by topic. Files
# starting with `0` need no model (pure-Lua + module-load tests), files
# starting with `1` need a generation model (ION7_MODEL), and files
# starting with `2` may need extra resources (draft model, embedding
# model, LoRA). The runner discovers files alphabetically so the order
# matches the prefix.
#
# Usage:
#   ION7_MODEL=/path/to/model.gguf bash tests/run_all.sh
#
# Optional environment:
#   ION7_LIB_DIR     Directory containing libllama.so / ion7_bridge.so
#   ION7_EMBED       Embedding model (.gguf) for the embedding suite
#   ION7_DRAFT       Draft model (.gguf) for speculative decoding
#   ION7_LORA        LoRA adapter (.gguf) for the LoRA suite
#   ION7_GPU_LAYERS  Override n_gpu_layers (default 0 — pure CPU)
#   ION7_SKIP        Whitespace-separated list of file basenames to skip
#                    (e.g. ION7_SKIP="22_training.lua")
# ──────────────────────────────────────────────────────────────────────────

set -e

PASS=0
FAIL=0
SKIP=0

cd "$(dirname "$0")/.."

# Print a one-line header for a suite file and run it. luajit's exit
# status drives our PASS / FAIL counter — 0 = suite green, anything
# else = at least one test in that file failed.
run_suite() {
    local file="$1"
    local name
    name="$(basename "$file" .lua)"

    # Honour ION7_SKIP : skip suites listed in the env var.
    if [ -n "$ION7_SKIP" ] && echo "$ION7_SKIP" | grep -qw "$(basename "$file")"; then
        printf "\n\033[33m══ SKIP %-40s\033[0m\n" "$name (in ION7_SKIP)"
        SKIP=$((SKIP + 1))
        return
    fi

    printf "\n\033[1m══ %-40s \033[0m%s\n" "$name" "$(printf '═%.0s' {1..18})"
    if ION7_MODEL="$ION7_MODEL" \
       ION7_LIB_DIR="$ION7_LIB_DIR" \
       ION7_EMBED="$ION7_EMBED" \
       ION7_DRAFT="$ION7_DRAFT" \
       ION7_LORA="$ION7_LORA" \
       ION7_GPU_LAYERS="$ION7_GPU_LAYERS" \
       luajit "$file"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

# Iterate over every numbered test file in lexicographic order. The
# `0X_*` files run first (no model needed), then `1X_*` (model), then
# `2X_*` (extras). Files that exit early via `H.require_model` count
# as PASS — that is what the helper guarantees when the env var is
# missing, so we don't spam the user with red FAILs on a fresh clone.
for f in tests/[0-9][0-9]_*.lua; do
    [ -f "$f" ] || continue
    run_suite "$f"
done

# ── Summary ──────────────────────────────────────────────────────────────
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"
printf "  Suites: \033[32m%d passed\033[0m" "$PASS"
[ "$FAIL" -gt 0 ] && printf "  \033[31m%d FAILED\033[0m" "$FAIL"
[ "$SKIP" -gt 0 ] && printf "  \033[33m%d skipped\033[0m" "$SKIP"
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"

[ "$FAIL" -eq 0 ]
