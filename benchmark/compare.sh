#!/usr/bin/env bash
# benchmark/compare.sh - run both backends and produce a side-by-side report.
#
# Usage:
#   ION7_MODEL=/path/to/model.gguf bash benchmark/compare.sh
#   ION7_MODEL=/path/to/model.gguf bash benchmark/compare.sh --n-gpu-layers 20 --n-gen 64
#
# Requirements:
#   - luajit + ion7-core (src/ in cwd)
#   - python3 + llama-cpp-python
#   - jq  (brew install jq / apt install jq)
#
# Options forwarded to both bench scripts:
#   --n-gpu-layers  N    (default: auto-fit)
#   --n-ctx         N    (default: 2048)
#   --n-gen         N    (default: 128, tokens to generate)
#   --n-repeat      N    (default: 3, repetitions per benchmark)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ── Dependency check ──────────────────────────────────────────────────────────
need() { command -v "$1" &>/dev/null || { echo "Missing: $1"; exit 1; }; }
need luajit
need python3
need jq

if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo -e "${YELLOW}[warn] llama-cpp-python not installed.${NC}"
    echo -e "       Run: pip install llama-cpp-python"
    echo -e "       CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall"
    echo ""
    echo -e "${YELLOW}Running ion7-core only (no comparison).${NC}"
    PYTHON_AVAILABLE=0
else
    PYTHON_AVAILABLE=1
fi

if [ -z "${ION7_MODEL:-}" ]; then
    echo "Set ION7_MODEL=/path/to/model.gguf"
    exit 1
fi

# Resolve ION7_LIB_DIR: accept LLAMA_LIB (path to .so) as an alias
if [ -z "${ION7_LIB_DIR:-}" ] && [ -n "${LLAMA_LIB:-}" ]; then
    export ION7_LIB_DIR="${LLAMA_LIB}"
fi

# Auto-detect libllama.so when neither variable is set
if [ -z "${ION7_LIB_DIR:-}" ]; then
    # find -print -quit avoids SIGPIPE that `find ... | head -1` causes under pipefail
    FOUND_LIB=$(find "${HOME}" -name "libllama.so" -print -quit 2>/dev/null)
    if [ -n "$FOUND_LIB" ]; then
        export ION7_LIB_DIR="$(dirname "$FOUND_LIB")"
        echo -e "${DIM}[auto] ION7_LIB_DIR=${ION7_LIB_DIR}${NC}"
    else
        echo -e "${YELLOW}[hint] ION7_LIB_DIR not set. If init fails, run:${NC}"
        echo -e "${YELLOW}       find ~ -name 'libllama.so' 2>/dev/null${NC}"
        echo -e "${YELLOW}       export ION7_LIB_DIR=/path/to/llama.cpp/build${NC}"
    fi
fi

# Forward all extra args to both scripts
EXTRA_ARGS="${*:-}"

# ── Run backends ──────────────────────────────────────────────────────────────
TMP_LUA=$(mktemp /tmp/ion7_bench_lua_XXXX.json)
TMP_PY=$(mktemp /tmp/ion7_bench_py_XXXX.json)

echo -e "${BOLD}═══ ion7-core vs llama-cpp-python ════════════════════════════${NC}"
echo -e "${DIM}Model: $(basename "$ION7_MODEL")${NC}"
echo ""

# Run Lua
echo -e "${BLUE}▶ Running ion7-core (LuaJIT)...${NC}"
echo -e "${DIM}  (model load may take 10-30s)${NC}"
cd "$ROOT_DIR"
luajit benchmark/bench_lua.lua $EXTRA_ARGS > "$TMP_LUA" 2>/tmp/ion7_lua_err.txt
LUA_EXIT=$?
if [ $LUA_EXIT -ne 0 ]; then
    echo -e "${RED}[ERROR] ion7-core exited with code $LUA_EXIT${NC}"
    cat /tmp/ion7_lua_err.txt
    exit 1
fi
# Validate JSON output
if ! jq empty "$TMP_LUA" 2>/dev/null; then
    echo -e "${RED}[ERROR] ion7-core output is not valid JSON:${NC}"
    cat "$TMP_LUA"
    echo "stderr:"
    cat /tmp/ion7_lua_err.txt
    exit 1
fi
echo -e "${GREEN}  ✓ ion7-core done${NC}"

# Run Python
if [ $PYTHON_AVAILABLE -eq 1 ]; then
    echo -e "${BLUE}▶ Running llama-cpp-python...${NC}"
    python3 benchmark/bench_python.py $EXTRA_ARGS > "$TMP_PY" 2> >(tee /tmp/ion7_py_err.txt >&2)
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] llama-cpp-python failed:${NC}"
        cat /tmp/ion7_py_err.txt
        PYTHON_AVAILABLE=0
    else
        echo -e "${GREEN}  ✓ llama-cpp-python done${NC}"
    fi
fi

echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
jq_lua() { jq -r "$1" "$TMP_LUA" 2>/dev/null || echo "N/A"; }
jq_py()  { jq -r "$1" "$TMP_PY"  2>/dev/null || echo "N/A"; }

# Format a ratio with colour: green if LuaJIT is faster (higher tok/s or lower ms)
ratio_tok_s() {
    local lua_val="$1" py_val="$2"
    if [[ "$lua_val" == "N/A" || "$py_val" == "N/A" ]]; then echo "N/A"; return; fi
    local ratio
    ratio=$(python3 -c "
lua=$lua_val; py=$py_val
if py == 0: print('N/A')
else: print(f'{lua/py:.2f}x')
" 2>/dev/null || echo "N/A")
    local cmp
    cmp=$(python3 -c "
lua=$lua_val; py=$py_val
print('gt' if lua > py else 'lt')
" 2>/dev/null || echo "eq")
    if [[ "$cmp" == "gt" ]]; then
        echo -e "${GREEN}${ratio}${NC}"
    else
        echo -e "${RED}${ratio}${NC}"
    fi
}

ratio_ms() {
    local lua_val="$1" py_val="$2"
    if [[ "$lua_val" == "N/A" || "$py_val" == "N/A" ]]; then echo "N/A"; return; fi
    local ratio
    ratio=$(python3 -c "
lua=$lua_val; py=$py_val
if lua == 0: print('N/A')
else: print(f'{py/lua:.2f}x')
" 2>/dev/null || echo "N/A")
    local cmp
    cmp=$(python3 -c "
lua=$lua_val; py=$py_val
print('lt' if lua < py else 'gt')
" 2>/dev/null || echo "eq")
    if [[ "$cmp" == "lt" ]]; then
        echo -e "${GREEN}${ratio}${NC}"
    else
        echo -e "${RED}${ratio}${NC}"
    fi
}

# Two-column row: label | lua | python | ratio
row() {
    local label="$1" lua_val="$2" py_val="$3" ratio_val="$4"
    printf "  %-34s  %12s  %12s  %10s\n" "$label" "$lua_val" "$py_val" "$ratio_val"
}

header() {
    printf "\n${BOLD}  %-34s  %12s  %12s  %10s${NC}\n" "$1" "ion7-core" "llama-cpp-py" "ratio"
    printf "  %s\n" "$(printf '─%.0s' {1..76})"
}

# ── Report ────────────────────────────────────────────────────────────────────

lua_version=$(jq_lua ".version")
py_version=$(jq_py ".version")
echo -e "${BOLD}Versions:${NC}  ion7-core ${lua_version}  |  llama-cpp-python ${py_version}"
echo -e "${DIM}n_ctx=$(jq_lua ".n_ctx")  n_gen=$(jq_lua ".n_gen")  n_repeat=$(jq_lua ".n_repeat")${NC}"
echo ""

# 1. Model load
header "1. Model Load"
lua_load=$(jq_lua ".benchmarks.model_load.load_ms")
py_load=$(jq_py  ".benchmarks.model_load.load_ms")
r=$(ratio_ms "$lua_load" "$py_load")
row "Load time" "${lua_load}ms" "${py_load}ms" "$r"

lua_ngl=$(jq_lua ".benchmarks.model_load.n_gpu_layers")
row "GPU layers" "$lua_ngl" "$(jq_py '.benchmarks.model_load.n_gpu_layers')" ""

lua_size=$(jq_lua ".benchmarks.model_load.size_gb // \"N/A\"")
row "Model size" "${lua_size}GB" "N/A" ""

lua_rss_delta=$(jq_lua ".benchmarks.model_load.rss_delta_mb // \"N/A\"")
py_rss_delta=$(jq_py  ".benchmarks.model_load.rss_delta_mb // \"N/A\"")
row "RSS delta (load)" "${lua_rss_delta}MB" "${py_rss_delta}MB" ""

# 2. Tokenization
header "2. Tokenization"
lua_tok=$(jq_lua ".benchmarks.tokenization.avg_tokens_per_s")
py_tok=$(jq_py  ".benchmarks.tokenization.avg_tokens_per_s")
r=$(ratio_tok_s "$lua_tok" "$py_tok")
row "Avg tok/s (3 prompts)" "$lua_tok" "$py_tok" "$r"

# Per-case breakdown
cases=$(jq_lua "[.benchmarks.tokenization.cases[].n_tokens] | length")
for idx in $(seq 0 $((cases - 1))); do
    lua_c=$(jq_lua ".benchmarks.tokenization.cases[$idx].tokens_per_s")
    py_c=$(jq_py  ".benchmarks.tokenization.cases[$idx].tokens_per_s")
    n=$(jq_lua    ".benchmarks.tokenization.cases[$idx].n_tokens")
    r=$(ratio_tok_s "$lua_c" "$py_c")
    row "  case $((idx+1)) (${n} tokens)" "$lua_c tok/s" "$py_c tok/s" "$r"
done

# 3. Prefill
header "3. Prompt Prefill"
lua_pfill=$(jq_lua ".benchmarks.prefill.avg_tokens_per_s")
py_pfill=$(jq_py  ".benchmarks.prefill.avg_tokens_per_s")
r=$(ratio_tok_s "$lua_pfill" "$py_pfill")
row "Avg prefill tok/s" "$lua_pfill" "$py_pfill" "$r"

cases=$(jq_lua "[.benchmarks.prefill.cases[].prompt_tokens] | length")
for idx in $(seq 0 $((cases - 1))); do
    lua_c=$(jq_lua ".benchmarks.prefill.cases[$idx].tokens_per_s")
    py_c=$(jq_py  ".benchmarks.prefill.cases[$idx].tokens_per_s")
    n=$(jq_lua    ".benchmarks.prefill.cases[$idx].prompt_tokens")
    r=$(ratio_tok_s "$lua_c" "$py_c")
    row "  case $((idx+1)) (${n} tokens)" "$lua_c tok/s" "$py_c tok/s" "$r"
done

# 4. Generation
header "4. Token Generation"
lua_gen=$(jq_lua ".benchmarks.generation.median_tok_s")
py_gen=$(jq_py  ".benchmarks.generation.median_tok_s")
r=$(ratio_tok_s "$lua_gen" "$py_gen")
row "Throughput (tok/s)" "$lua_gen" "$py_gen" "$r"

lua_ms=$(jq_lua ".benchmarks.generation.ms_per_token")
py_ms=$(jq_py  ".benchmarks.generation.ms_per_token")
r=$(ratio_ms "$lua_ms" "$py_ms")
row "Latency (ms/token)" "$lua_ms" "$py_ms" "$r"

lua_total=$(jq_lua ".benchmarks.generation.median_ms")
py_total=$(jq_py  ".benchmarks.generation.median_ms")
r=$(ratio_ms "$lua_total" "$py_total")
row "Total (ms, n=$(jq_lua '.n_gen') tokens)" "$lua_total" "$py_total" "$r"

# 5. Grammar
header "5. Grammar-Constrained Generation"
lua_g=$(jq_lua ".benchmarks.grammar_constrained.supported")
py_g=$(jq_py  ".benchmarks.grammar_constrained.supported")
row "Supported" "$lua_g" "$py_g" ""

lua_gms=$(jq_lua ".benchmarks.grammar_constrained.median_ms")
py_gms=$(jq_py  ".benchmarks.grammar_constrained.median_ms")
if [[ "$lua_g" == "true" && "$py_g" == "true" ]]; then
    r=$(ratio_ms "$lua_gms" "$py_gms")
    row "Median ms" "$lua_gms" "$py_gms" "$r"
fi

lua_gv=$(jq_lua ".benchmarks.grammar_constrained.all_valid")
py_gv=$(jq_py  ".benchmarks.grammar_constrained.all_valid")
row "All outputs valid" "$lua_gv" "$py_gv" ""

# Grammar outputs
lua_out=$(jq_lua "[.benchmarks.grammar_constrained.outputs[]] | join(\", \")")
py_out=$(jq_py  "[.benchmarks.grammar_constrained.outputs[]] | join(\", \")")
printf "  %-34s  %s\n" "  Lua outputs:" "$lua_out"
printf "  %-34s  %s\n" "  Python outputs:" "$py_out"

# 6. KV snapshot
header "6. KV Cache Snapshot"
lua_kv_save=$(jq_lua ".benchmarks.kv_snapshot.save_ms")
lua_kv_load=$(jq_lua ".benchmarks.kv_snapshot.load_ms")
py_kv_save=$(jq_py  ".benchmarks.kv_snapshot.save_ms")
py_kv_load=$(jq_py  ".benchmarks.kv_snapshot.load_ms")
lua_kv_mem=$(jq_lua ".benchmarks.kv_snapshot.in_memory")
py_kv_mem=$(jq_py  ".benchmarks.kv_snapshot.in_memory")

r_save=$(ratio_ms "$lua_kv_save" "$py_kv_save")
r_load=$(ratio_ms "$lua_kv_load" "$py_kv_load")
row "Snapshot save_ms"   "${lua_kv_save}ms"  "${py_kv_save}ms"  "$r_save"
row "Snapshot load_ms"   "${lua_kv_load}ms"  "${py_kv_load}ms"  "$r_load"
row "In-memory (no I/O)" "$lua_kv_mem"       "$py_kv_mem"       ""

lua_kv_sz=$(jq_lua ".benchmarks.kv_snapshot.state_size_kb")
py_kv_sz=$(jq_py  ".benchmarks.kv_snapshot.state_size_kb")
row "Snapshot size"  "${lua_kv_sz}KB"  "${py_kv_sz}KB"  ""

# 7. Detokenization
header "7. Detokenization"
lua_dtok=$(jq_lua ".benchmarks.detokenization.median_ms")
py_dtok=$(jq_py  ".benchmarks.detokenization.median_ms")
r=$(ratio_ms "$lua_dtok" "$py_dtok")
row "Median ms/call" "${lua_dtok}ms" "${py_dtok}ms" "$r"
lua_dtok_cps=$(jq_lua ".benchmarks.detokenization.calls_per_s")
py_dtok_cps=$(jq_py  ".benchmarks.detokenization.calls_per_s")
r2=$(ratio_tok_s "$lua_dtok_cps" "$py_dtok_cps")
row "Throughput (calls/s)" "$lua_dtok_cps" "$py_dtok_cps" "$r2"

# 7b. Memory footprint
header "7b. Memory Footprint"
lua_rss=$(jq_lua ".benchmarks.memory.rss_after_load_mb // \"N/A\"")
py_rss=$(jq_py  ".benchmarks.memory.rss_after_load_mb // \"N/A\"")
lua_rss_d=$(jq_lua ".benchmarks.memory.rss_delta_load_mb // \"N/A\"")
py_rss_d=$(jq_py  ".benchmarks.memory.rss_delta_load_mb // \"N/A\"")
row "RSS after load (MB)"  "${lua_rss}MB"   "${py_rss}MB"   ""
row "RSS delta load (MB)"  "${lua_rss_d}MB" "${py_rss_d}MB" "$(python3 -c "
a,b='${lua_rss_d}','${py_rss_d}'
try:
    r=float(b)/float(a)
    print(f'{r:.2f}x less' if r>1 else f'{1/r:.2f}x more')
except: print('N/A')
" 2>/dev/null)"

# 8. Sampler overhead (ion7-core only)
header "8. Sampler Chain Overhead (ion7-core only)"
for profile in greedy minimal standard; do
    lua_sp=$(jq_lua ".benchmarks.sampler_overhead.profiles.${profile}.median_ms")
    row "  ${profile} profile (ms/sample)" "$lua_sp" "N/A" ""
done

# 9. Context creation
header "9. Context Creation"
lua_ctx_f16=$(jq_lua ".benchmarks.context_creation.cases[]? | select(.kv_quant==\"f16\") | .median_ms // \"N/A\"")
lua_ctx_q8=$(jq_lua  ".benchmarks.context_creation.cases[]? | select(.kv_quant==\"q8_0\") | .median_ms // \"N/A\"")
lua_ctx_q4=$(jq_lua  ".benchmarks.context_creation.cases[]? | select(.kv_quant==\"q4_0\") | .median_ms // \"N/A\"")
py_ctx=$(jq_py ".benchmarks.context_creation.median_ms // \"N/A\"")
row "kv=f16 (ms)"  "${lua_ctx_f16:-N/A}" "${py_ctx:-N/A}" ""
row "kv=q8_0 (ms)" "${lua_ctx_q8:-N/A}"  "N/A" ""
row "kv=q4_0 (ms)" "${lua_ctx_q4:-N/A}"  "N/A" ""

# 10. Single-token decode
header "10. Single-Token Decode"
lua_stok=$(jq_lua ".benchmarks.single_token_loop.tokens_per_s // \"N/A\"")
lua_sms=$(jq_lua  ".benchmarks.single_token_loop.median_ms_per_token // \"N/A\"")
py_stok=$(jq_py   ".benchmarks.single_token_loop.tokens_per_s // \"N/A\"")
py_sms=$(jq_py    ".benchmarks.single_token_loop.median_ms_per_token // \"N/A\"")
r_stok=$(ratio_tok_s "$lua_stok" "$py_stok")
r_sms=$(ratio_ms "$lua_sms" "$py_sms")
row "Throughput (tok/s)"   "$lua_stok" "$py_stok" "$r_stok"
row "Median ms/token"      "${lua_sms}ms" "${py_sms}ms" "$r_sms"

# 11. KV operations (ion7-core only)
header "11. KV Operations (ion7-core only)"
lua_kvc=$(jq_lua ".benchmarks.kv_operations.kv_clear.calls_per_s // \"N/A\"")
lua_kvr=$(jq_lua ".benchmarks.kv_operations.kv_seq_rm.calls_per_s // \"N/A\"")
lua_kvcp=$(jq_lua ".benchmarks.kv_operations.kv_seq_cp.calls_per_s // \"N/A\"")
row "kv_clear() (calls/s)"  "$lua_kvc"  "N/A" ""
row "kv_seq_rm() (calls/s)" "$lua_kvr"  "N/A" ""
row "kv_seq_cp() (calls/s)" "$lua_kvcp" "N/A" ""

# 12. State persistence
header "12. State Persistence (file I/O)"
lua_save_f=$(jq_lua ".benchmarks.state_persistence.save_file_ms // \"N/A\"")
lua_load_f=$(jq_lua ".benchmarks.state_persistence.load_file_ms // \"N/A\"")
row "save_state() ms" "$lua_save_f" "N/A" ""
row "load_state() ms" "$lua_load_f" "N/A" ""

# 13. Custom sampler (ion7-core only)
header "13. Custom Lua Sampler (ion7-core only)"
lua_custom=$(jq_lua ".benchmarks.custom_sampler.lua_greedy.samples_per_s // \"N/A\"")
lua_native=$(jq_lua ".benchmarks.custom_sampler.native_greedy.samples_per_s // \"N/A\"")
lua_oh=$(jq_lua     ".benchmarks.custom_sampler.overhead_pct // \"N/A\"")
row "Lua greedy (samples/s)"    "$lua_custom" "N/A" ""
row "Native greedy (samples/s)" "$lua_native" "N/A" ""
row "Callback overhead"         "${lua_oh}%"  "N/A" ""

# 14. Vocab operations
header "14. Vocab Operations"
lua_bos=$(jq_lua ".benchmarks.vocab_operations.bos.calls_per_s // \"N/A\"")
lua_eog=$(jq_lua ".benchmarks.vocab_operations.is_eog.calls_per_s // \"N/A\"")
lua_tok_ops=$(jq_lua  ".benchmarks.vocab_operations.tokenize.calls_per_s // \"N/A\"")
lua_dtok_ops=$(jq_lua ".benchmarks.vocab_operations.detokenize.calls_per_s // \"N/A\"")
py_tok_ops=$(jq_py    ".benchmarks.vocab_operations.tokenize.calls_per_s // \"N/A\"")
py_dtok_ops=$(jq_py   ".benchmarks.vocab_operations.detokenize.calls_per_s // \"N/A\"")
row "bos() (calls/s)"             "$lua_bos"     "N/A"           ""
row "is_eog() (calls/s)"          "$lua_eog"     "N/A"           ""
row "tokenize() loop (calls/s)"   "$lua_tok_ops"  "$py_tok_ops"  "$(ratio_tok_s "$lua_tok_ops"  "$py_tok_ops")"
row "detokenize() loop (calls/s)" "$lua_dtok_ops" "$py_dtok_ops" "$(ratio_tok_s "$lua_dtok_ops" "$py_dtok_ops")"

# ── Errors ────────────────────────────────────────────────────────────────────
lua_errors=$(jq_lua "[.errors[]] | length")
py_errors=$(jq_py  "[.errors[]] | length")
if [[ "$lua_errors" -gt 0 || "$py_errors" -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Errors:${NC}"
    if [[ "$lua_errors" -gt 0 ]]; then
        jq_lua ".errors[]" | sed 's/^/  [lua] /'
    fi
    if [[ "$py_errors" -gt 0 ]]; then
        jq_py ".errors[]" | sed 's/^/  [py]  /'
    fi
fi

# ── Notes ─────────────────────────────────────────────────────────────────────
echo ""
# ── Stress results (if present) ──────────────────────────────────────────────
has_stress=$(jq_lua "if .benchmarks.stress_sustained_generation then \"yes\" else \"no\" end")
if [[ "$has_stress" == "yes" ]]; then
    echo ""
    echo -e "${BOLD}═══ Stress Results (ion7-core) ════════════════════════════════${NC}"

    header "S1. Sustained Generation"
    lua_stok=$(jq_lua ".benchmarks.stress_sustained_generation.overall_tok_s")
    lua_sn=$(jq_lua ".benchmarks.stress_sustained_generation.n_tokens_generated")
    row "Overall tok/s ($lua_sn tokens)" "$lua_stok" "N/A" ""
    echo "  Window tok/s (32-token windows):"
    jq_lua "[.benchmarks.stress_sustained_generation.window_tok_s[]] | @csv" | \
        tr ',' ' ' | xargs printf "    %s\n" 2>/dev/null || true

    header "S2. Back-to-Back Sessions"
    lua_ns=$(jq_lua  ".benchmarks.stress_back_to_back.n_sessions")
    lua_med=$(jq_lua ".benchmarks.stress_back_to_back.median_session_ms")
    lua_min=$(jq_lua ".benchmarks.stress_back_to_back.min_session_ms")
    lua_max=$(jq_lua ".benchmarks.stress_back_to_back.max_session_ms")
    py_ns=$(jq_py    ".benchmarks.stress_back_to_back.n_sessions")
    py_med=$(jq_py   ".benchmarks.stress_back_to_back.median_session_ms")
    py_min=$(jq_py   ".benchmarks.stress_back_to_back.min_session_ms")
    py_max=$(jq_py   ".benchmarks.stress_back_to_back.max_session_ms")
    r_med=$(ratio_ms "$lua_med" "$py_med")
    row "Sessions" "$lua_ns" "$py_ns" ""
    row "Median ms/session" "${lua_med}ms" "${py_med}ms" "$r_med"
    row "Min ms/session"    "${lua_min}ms" "${py_min}ms" ""
    row "Max ms/session"    "${lua_max}ms" "${py_max}ms" ""

    header "S3. Context Fill Pressure"
    lua_fill=$(jq_lua ".benchmarks.stress_context_pressure.fill_pct")
    py_fill=$(jq_py   ".benchmarks.stress_context_pressure.fill_pct")
    lua_ptok=$(jq_lua ".benchmarks.stress_context_pressure.prefill_tok_s")
    py_ptok=$(jq_py   ".benchmarks.stress_context_pressure.prefill_tok_s")
    lua_gtok=$(jq_lua ".benchmarks.stress_context_pressure.gen_on_full_tok_s")
    py_gtok=$(jq_py   ".benchmarks.stress_context_pressure.gen_on_full_tok_s")
    r_p=$(ratio_tok_s "$lua_ptok" "$py_ptok")
    r_g=$(ratio_tok_s "$lua_gtok" "$py_gtok")
    row "Context fill"          "${lua_fill}%"  "${py_fill}%"  ""
    row "Prefill tok/s"         "$lua_ptok"     "$py_ptok"     "$r_p"
    row "Gen tok/s (full ctx)"  "$lua_gtok"     "$py_gtok"     "$r_g"

    header "S4. Sampler Throughput (1000 calls)"
    lua_sus=$(jq_lua    ".benchmarks.stress_sampler_throughput.samples_per_s")
    lua_sus_us=$(jq_lua ".benchmarks.stress_sampler_throughput.avg_us_per_call")
    py_sus=$(jq_py      ".benchmarks.stress_sampler_throughput.samples_per_s")
    py_sus_us=$(jq_py   ".benchmarks.stress_sampler_throughput.avg_us_per_call")
    r_sus=$(ratio_tok_s "$lua_sus" "$py_sus")
    row "Samples/s" "$lua_sus" "$py_sus" "$r_sus"
    row "μs/call"   "$lua_sus_us" "$py_sus_us" ""
    echo -e "  ${DIM}Note: Python uses generate() single-step approximation; ion7-core uses raw sample()${NC}"
fi

echo -e "${DIM}Notes:${NC}"
echo -e "${DIM}  - ratio > 1.0x = ion7-core is faster (green) or slower (red)${NC}"
echo -e "${DIM}  - KV snapshot: ion7-core uses in-memory blobs; llama-cpp-python uses file I/O${NC}"
echo -e "${DIM}  - Sampler overhead benchmark is ion7-core specific (no Python equivalent)${NC}"
echo -e "${DIM}  - Raw JSON: ${TMP_LUA} (lua) and ${TMP_PY} (python)${NC}"
echo ""

# Copy JSON results to outputs
cp "$TMP_LUA" ./benchmark/last_results_lua.json
if [ $PYTHON_AVAILABLE -eq 1 ]; then
    cp "$TMP_PY" ./benchmark/last_results_python.json
fi
echo -e "${DIM}Results saved to benchmark/last_results_{lua,python}.json${NC}"
