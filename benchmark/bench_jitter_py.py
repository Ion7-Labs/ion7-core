#!/usr/bin/env python3
"""
benchmark/bench_jitter_py.py - timing stability benchmark (llama-cpp-python).

Runs N iterations of each operation and records every individual timing.
Same schema as bench_jitter_lua.lua for direct statistical comparison.
"""

import os, sys, json, gc, time, math, argparse, statistics

try:
    from llama_cpp import Llama
except ImportError:
    print(json.dumps({"error": "llama-cpp-python not installed"})); sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--n-gpu-layers", type=int, default=None)
parser.add_argument("--n-ctx",        type=int, default=2048)
parser.add_argument("--n-iter",       type=int, default=100)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

MODEL = os.environ.get("ION7_MODEL")
if not MODEL:
    print(json.dumps({"error": "Set ION7_MODEL"})); sys.exit(1)

def now_ms(): return time.perf_counter() * 1000.0

# ── Stats ─────────────────────────────────────────────────────────────────────

def calc_stats(times):
    if not times: return {}
    s = sorted(times)
    n = len(s)
    mean   = sum(s) / n
    stddev = math.sqrt(sum((x - mean)**2 for x in s) / n)
    def pct(p): return s[max(0, math.ceil(p / 100 * n) - 1)]
    return {
        "n":      n,
        "mean":   round(mean,          3),
        "median": round(pct(50),       3),
        "p95":    round(pct(95),       3),
        "p99":    round(pct(99),       3),
        "min":    round(s[0],          3),
        "max":    round(s[-1],         3),
        "stddev": round(stddev,        3),
        "cv_pct": round(stddev / mean * 100, 2) if mean > 0 else 0,
        "times":  [round(t, 3) for t in times],
    }

# ── Setup ─────────────────────────────────────────────────────────────────────

print("[jitter] loading model...", file=sys.stderr, flush=True)
ngl = args.n_gpu_layers if args.n_gpu_layers is not None else 0

llm = Llama(
    model_path   = MODEL,
    n_gpu_layers = ngl,
    n_ctx        = args.n_ctx,
    n_batch      = 512,
    verbose      = False,
    seed         = args.seed,
)

N = args.n_iter
RESULTS = {
    "backend": "llama-cpp-python",
    "n_iter":  N,
    "model":   os.path.basename(MODEL),
    "ops":     {},
}

# Helper: warm up then time N iterations
def measure_n(setup_fn, op_fn):
    if setup_fn: setup_fn()
    op_fn()  # warm-up
    gc.collect()
    times = []
    for _ in range(N):
        if setup_fn: setup_fn()
        t0 = now_ms()
        op_fn()
        times.append(now_ms() - t0)
        gc.collect()
    return times

# ── Operations ────────────────────────────────────────────────────────────────

# 1. Tokenization - short
print("[jitter] 1/6 tokenization...", file=sys.stderr, flush=True)
RESULTS["ops"]["tokenize_short"] = calc_stats(
    measure_n(None, lambda: llm.tokenize(b"Hello world"))
)

# 2. Tokenization - medium
medium_text = ("The quick brown fox jumps over the lazy dog. " * 10).encode()
RESULTS["ops"]["tokenize_medium"] = calc_stats(
    measure_n(None, lambda: llm.tokenize(medium_text))
)

# 3. Prefill - small
print("[jitter] 2/6 prefill small...", file=sys.stderr, flush=True)
small_tokens = llm.tokenize(b"What is 2+2?", add_bos=True)
small_n = len(small_tokens)

def prefill_small():
    llm.reset()
    llm.eval(small_tokens)

times = []
prefill_small()  # warm-up
gc.collect()
for _ in range(N):
    llm.reset()
    t0 = now_ms()
    llm.eval(small_tokens)
    times.append(now_ms() - t0)
    gc.collect()

op = calc_stats(times)
op["prompt_tokens"] = small_n
RESULTS["ops"]["prefill_small"] = op

# 4. Prefill - medium
print("[jitter] 3/6 prefill medium...", file=sys.stderr, flush=True)
med_text = ("Explain transformers. " * 8).encode()
med_tokens = llm.tokenize(med_text, add_bos=True)
med_n = len(med_tokens)

times = []
llm.reset(); llm.eval(med_tokens); gc.collect()  # warm-up
for _ in range(N):
    llm.reset()
    t0 = now_ms()
    llm.eval(med_tokens)
    times.append(now_ms() - t0)
    gc.collect()

op = calc_stats(times)
op["prompt_tokens"] = med_n
RESULTS["ops"]["prefill_medium"] = op

# 5. Single-token generation
print("[jitter] 4/6 single token...", file=sys.stderr, flush=True)
times = []
llm.reset(); llm.eval(small_tokens); gc.collect()  # warm-up
next(llm.generate(small_tokens, temp=0.0, reset=False))

for _ in range(N):
    llm.reset()
    llm.eval(small_tokens)
    t0 = now_ms()
    next(llm.generate(small_tokens, temp=0.0, reset=False))
    times.append(now_ms() - t0)
    gc.collect()

RESULTS["ops"]["decode_single"] = calc_stats(times)

# 6. Detokenization
print("[jitter] 5/6 detokenization...", file=sys.stderr, flush=True)
dtok = llm.tokenize(b"Hello world this is a test of detokenization speed", add_bos=False)
RESULTS["ops"]["detokenize"] = calc_stats(
    measure_n(None, lambda: llm.detokenize(dtok))
)

# 7. KV snapshot
print("[jitter] 6/6 kv snapshot...", file=sys.stderr, flush=True)
try:
    llm.reset(); llm.eval(small_tokens); gc.collect()
    blob = llm.save_state()

    save_times = measure_n(None, lambda: llm.save_state())
    load_times = measure_n(None, lambda: llm.load_state(blob))
    RESULTS["ops"]["kv_snapshot_save"] = calc_stats(save_times)
    RESULTS["ops"]["kv_snapshot_load"] = calc_stats(load_times)
except Exception as e:
    RESULTS["ops"]["kv_snapshot_save"] = {"error": str(e)}
    RESULTS["ops"]["kv_snapshot_load"] = {"error": str(e)}

# ── Output ────────────────────────────────────────────────────────────────────

print(json.dumps(RESULTS, indent=2))
