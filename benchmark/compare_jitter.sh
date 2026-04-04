#!/usr/bin/env bash
# benchmark/compare_jitter.sh - timing stability comparison: LuaJIT vs Python.
#
# Runs N iterations of each operation in both backends, then compares
# mean / median / p95 / p99 / stddev and CV% (coefficient of variation).
#
# CV% = stddev / mean * 100 - the key stability metric.
# Lower CV% = more predictable = better for real-time use.
#
# Usage:
#   ION7_MODEL=/path/to/model.gguf bash benchmark/compare_jitter.sh
#   ION7_MODEL=...  bash benchmark/compare_jitter.sh --n-gpu-layers 32 --n-iter 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[0;36m'; NC='\033[0m'

need() { command -v "$1" &>/dev/null || { echo "Missing: $1"; exit 1; }; }
need luajit; need python3; need jq

if [ -z "${ION7_MODEL:-}" ]; then
    echo "Set ION7_MODEL=/path/to/model.gguf"; exit 1
fi

if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo -e "${YELLOW}[warn] llama-cpp-python not installed - Lua only${NC}"
    PYTHON_AVAILABLE=0
else
    PYTHON_AVAILABLE=1
fi

if [ -z "${ION7_LIB_DIR:-}" ]; then
    FOUND_LIB=$(find "${HOME}" -name "libllama.so" 2>/dev/null | head -1)
    if [ -n "$FOUND_LIB" ]; then
        export ION7_LIB_DIR="$(dirname "$FOUND_LIB")"
        echo -e "${DIM}[auto] ION7_LIB_DIR=${ION7_LIB_DIR}${NC}"
    fi
fi

EXTRA_ARGS="${*:-}"
TMP_LUA=$(mktemp /tmp/ion7_jitter_lua_XXXX.json)
TMP_PY=$(mktemp /tmp/ion7_jitter_py_XXXX.json)

echo -e "${BOLD}═══ Timing Stability: ion7-core vs llama-cpp-python ══════════════${NC}"
echo -e "${DIM}Model: $(basename "$ION7_MODEL")${NC}"
echo -e "${DIM}CV% = stddev/mean × 100 - lower is more stable${NC}"
echo ""

cd "$ROOT_DIR"

echo -e "${CYAN}▶ Running ion7-core (LuaJIT)...${NC}"
luajit benchmark/bench_jitter_lua.lua $EXTRA_ARGS > "$TMP_LUA" 2>/tmp/ion7_jitter_lua_err.txt
echo -e "${GREEN}  ✓ done${NC}"

if [ $PYTHON_AVAILABLE -eq 1 ]; then
    echo -e "${CYAN}▶ Running llama-cpp-python...${NC}"
    python3 benchmark/bench_jitter_py.py $EXTRA_ARGS > "$TMP_PY" 2>/tmp/ion7_jitter_py_err.txt || PYTHON_AVAILABLE=0
    [ $PYTHON_AVAILABLE -eq 1 ] && echo -e "${GREEN}  ✓ done${NC}"
fi

echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────

jl() { jq -r "$1" "$TMP_LUA" 2>/dev/null || echo "N/A"; }
jp() { jq -r "$1" "$TMP_PY"  2>/dev/null || echo "N/A"; }

N_ITER=$(jl ".n_iter")

# Colour CV%: < 5% green, < 15% yellow, >= 15% red
cv_color() {
    local v="$1"
    if [[ "$v" == "N/A" ]]; then echo "$v"; return; fi
    local cmp
    cmp=$(python3 -c "v=float('$v'); print('lo' if v<5 else ('mid' if v<15 else 'hi'))" 2>/dev/null || echo "mid")
    case "$cmp" in
        lo)  echo -e "${GREEN}${v}%${NC}" ;;
        mid) echo -e "${YELLOW}${v}%${NC}" ;;
        *)   echo -e "${RED}${v}%${NC}" ;;
    esac
}

# Winner: lower ms = better
winner_ms() {
    local lua="$1" py="$2"
    if [[ "$lua" == "N/A" || "$py" == "N/A" ]]; then echo ""; return; fi
    local w
    w=$(python3 -c "print('lua' if float('$lua') < float('$py') else 'py')" 2>/dev/null || echo "")
    [[ "$w" == "lua" ]] && echo -e " ${GREEN}← LuaJIT${NC}" || echo -e " ${RED}← Python${NC}"
}

# Print a full stats block for one operation
print_op() {
    local title="$1" lua_key="$2" py_key="$3"
    local unit="${4:-ms}"

    local lmean  lmed  lp95  lp99  lstd  lcv
    local pymean pymed pyp95 pyp99 pystd pycv

    lmean=$(jl ".ops.${lua_key}.mean")
    lmed=$(jl  ".ops.${lua_key}.median")
    lp95=$(jl  ".ops.${lua_key}.p95")
    lp99=$(jl  ".ops.${lua_key}.p99")
    lstd=$(jl  ".ops.${lua_key}.stddev")
    lcv=$(jl   ".ops.${lua_key}.cv_pct")

    if [ $PYTHON_AVAILABLE -eq 1 ]; then
        pymean=$(jp ".ops.${py_key}.mean")
        pymed=$(jp  ".ops.${py_key}.median")
        pyp95=$(jp  ".ops.${py_key}.p95")
        pyp99=$(jp  ".ops.${py_key}.p99")
        pystd=$(jp  ".ops.${py_key}.stddev")
        pycv=$(jp   ".ops.${py_key}.cv_pct")
    else
        pymean="N/A"; pymed="N/A"; pyp95="N/A"; pyp99="N/A"; pystd="N/A"; pycv="N/A"
    fi

    printf "\n${BOLD}  %s${NC}\n" "$title"
    printf "  ${DIM}%-14s  %10s  %10s${NC}\n" "" "ion7-core" "Python"
    printf "  ${DIM}%s${NC}\n" "$(printf '─%.0s' {1..44})"
    printf "  %-14s  %9s${unit}  %9s${unit}%s\n" "mean"   "$lmean"  "$pymean" "$(winner_ms "$lmean" "$pymean")"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "median" "$lmed"   "$pymed"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "p95"    "$lp95"   "$pyp95"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "p99"    "$lp99"   "$pyp99"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "min"    "$(jl ".ops.${lua_key}.min")" "$(jp ".ops.${py_key}.min" 2>/dev/null || echo N/A)"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "max"    "$(jl ".ops.${lua_key}.max")" "$(jp ".ops.${py_key}.max" 2>/dev/null || echo N/A)"
    printf "  %-14s  %9s${unit}  %9s${unit}\n"   "stddev" "$lstd"   "$pystd"
    printf "  %-14s  %10s  %10s\n" \
        "CV% (stability)" "$(cv_color "$lcv")" "$(cv_color "$pycv")"
}

# ── Report ────────────────────────────────────────────────────────────────────

echo -e "${BOLD}═══ Results (N=${N_ITER} iterations per operation) ════════════════════${NC}"
echo -e "${DIM}  CV% < 5% = stable, 5-15% = acceptable, > 15% = jittery${NC}"

print_op "1. Tokenization - short (~2 tokens)" \
    "tokenize_short" "tokenize_short"

print_op "2. Tokenization - medium (~100 tokens)" \
    "tokenize_medium" "tokenize_medium"

lua_small_n=$(jl ".ops.prefill_small.prompt_tokens")
py_small_n=$(jp  ".ops.prefill_small.prompt_tokens" 2>/dev/null || echo "?")
print_op "3. Prefill - small (${lua_small_n} tokens)" \
    "prefill_small" "prefill_small"

lua_med_n=$(jl ".ops.prefill_medium.prompt_tokens")
print_op "4. Prefill - medium (~${lua_med_n} tokens)" \
    "prefill_medium" "prefill_medium"

print_op "5. Single-token decode" \
    "decode_single" "decode_single"

print_op "6. Detokenization" \
    "detokenize" "detokenize"

# KV snapshot (may not be available on Python)
lua_kv_cv=$(jl ".ops.kv_snapshot_save.cv_pct")
if [ "$lua_kv_cv" != "N/A" ]; then
    print_op "7. KV Snapshot - save" "kv_snapshot_save" "kv_snapshot_save"
    print_op "8. KV Snapshot - load" "kv_snapshot_load" "kv_snapshot_load"
fi

# ── Stability summary ─────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══ Stability Summary ══════════════════════════════════════════════${NC}"
printf "  ${BOLD}%-34s  %10s  %10s${NC}\n" "Operation" "Lua CV%" "Python CV%"
printf "  %s\n" "$(printf '─%.0s' {1..58})"

ops=("tokenize_short" "tokenize_medium" "prefill_small" "prefill_medium" "decode_single" "detokenize")
labels=("Tokenize short" "Tokenize medium" "Prefill small" "Prefill medium" "Decode single" "Detokenize")

for idx in "${!ops[@]}"; do
    op="${ops[$idx]}"
    label="${labels[$idx]}"
    lcv=$(jl ".ops.${op}.cv_pct")
    pycv=$([ $PYTHON_AVAILABLE -eq 1 ] && jp ".ops.${op}.cv_pct" || echo "N/A")
    printf "  %-34s  %10s  %10s\n" "$label" "$(cv_color "$lcv")" "$(cv_color "$pycv")"
done

echo ""
echo -e "${DIM}Raw JSON saved to: $TMP_LUA (lua)  $TMP_PY (python)${NC}"
cp "$TMP_LUA" benchmark/last_jitter_lua.json
[ $PYTHON_AVAILABLE -eq 1 ] && cp "$TMP_PY" benchmark/last_jitter_python.json
echo -e "${DIM}Results saved to benchmark/last_jitter_{lua,python}.json${NC}"
echo ""
