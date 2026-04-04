#!/usr/bin/env python3
"""
benchmark/bench_python.py - llama-cpp-python side of the comparison.

Measures the same operations as bench_lua.lua so results are directly comparable:
  1. Model load time
  2. Tokenization throughput
  3. Prompt prefill speed (tokens/s)
  4. Generation throughput (tokens/s)
  5. Grammar-constrained generation
  6. KV snapshot/restore latency (file-based)
  7. Detokenization
  8. Memory (RSS tracking)
  9. Context creation overhead
  10. Single-token decode loop
  11. Vocab operations throughput

Output: JSON to stdout - same schema as bench_lua.lua.

Usage:
  ION7_MODEL=/path/to/model.gguf python3 benchmark/bench_python.py
  ION7_MODEL=/path/to/model.gguf python3 benchmark/bench_python.py --n-gpu-layers 20
"""

import os, sys, time, json, gc, argparse, statistics
from typing import Optional

# ── Dependency check ──────────────────────────────────────────────────────────

try:
    from llama_cpp import Llama, LlamaGrammar
    import llama_cpp
    LLAMA_CPP_VERSION = getattr(llama_cpp, "__version__", "unknown")
except ImportError:
    print(json.dumps({
        "error": "llama-cpp-python not installed. Run: pip install llama-cpp-python",
        "install": "CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall"
    }))
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n-gpu-layers", type=int, default=None)
parser.add_argument("--n-ctx",        type=int, default=2048)
parser.add_argument("--n-gen",        type=int, default=128,  help="tokens to generate per benchmark")
parser.add_argument("--n-repeat",     type=int, default=3,    help="repetitions per benchmark")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--stress",       action="store_true")
parser.add_argument("--n-ubatch",     type=int, default=512, help="Physical batch size (passed to llama-cpp-python n_batch param)")
args = parser.parse_args()

MODEL = os.environ.get("ION7_MODEL")
if not MODEL:
    print(json.dumps({"error": "Set ION7_MODEL=/path/to/model.gguf"}))
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def rss_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return -1.0

def median(xs):
    return statistics.median(xs) if xs else 0.0

def fmt(ms: float) -> str:
    return f"{ms:.2f}ms"

RESULTS = {
    "backend":      "llama-cpp-python",
    "version":      LLAMA_CPP_VERSION,
    "model":        os.path.basename(MODEL),
    "n_ctx":        args.n_ctx,
    "n_gen":        args.n_gen,
    "n_repeat":     args.n_repeat,
    "benchmarks":   {},
    "errors":       [],
}

def record(name: str, data: dict):
    RESULTS["benchmarks"][name] = data

def err(name: str, msg: str):
    RESULTS["errors"].append({"benchmark": name, "error": msg})

# ── 1. Model load ─────────────────────────────────────────────────────────────

mem_before = rss_mb()
t0 = now_ms()

ngl = args.n_gpu_layers if args.n_gpu_layers is not None else 0

try:
    llm = Llama(
        model_path      = MODEL,
        n_gpu_layers    = ngl,
        n_ctx           = args.n_ctx,
        n_batch         = 512,    # llama-cpp-python default
        verbose         = False,
        seed            = args.seed,
    )
    load_ms = now_ms() - t0
    mem_after = rss_mb()

    record("model_load", {
        "load_ms":      round(load_ms, 2),
        "rss_delta_mb": round(mem_after - mem_before, 1),
        "n_gpu_layers": ngl,
        "n_batch": 512,
        "n_ubatch": 512,
    })
except Exception as e:
    print(json.dumps({"error": f"Model load failed: {e}"}))
    sys.exit(1)

# ── 2. Tokenization ───────────────────────────────────────────────────────────

TOKENIZE_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog. " * 10,
    "Explain the attention mechanism in transformers. " * 20,
]

tok_results = []
for text in TOKENIZE_TEXTS:
    times = []
    for _ in range(args.n_repeat):
        t0 = now_ms()
        tokens = llm.tokenize(text.encode())
        elapsed = now_ms() - t0
        times.append(elapsed)
    n_tokens = len(tokens)
    tok_ms = median(times)
    tok_results.append({
        "text_chars":  len(text),
        "n_tokens":    n_tokens,
        "median_ms":   round(tok_ms, 3),
        "tokens_per_s": round(n_tokens / (tok_ms / 1000), 0) if tok_ms > 0 else 0,
    })

record("tokenization", {
    "cases": tok_results,
    "avg_tokens_per_s": round(statistics.mean(r["tokens_per_s"] for r in tok_results), 0),
})

# ── 3. Prompt prefill ─────────────────────────────────────────────────────────

PROMPTS = [
    "What is 2+2?",
    "Explain in detail the history of the Roman Empire and its cultural impact on Western civilization.",
    ("Write a comprehensive technical guide about transformer architectures. " * 8),
]

prefill_results = []
for prompt in PROMPTS:
    times_ms  = []
    tok_rates = []

    for _ in range(args.n_repeat):
        # Reset state
        llm.reset()
        tokens = llm.tokenize(prompt.encode(), add_bos=True)
        n = len(tokens)

        t0 = now_ms()
        # Force prefill by evaluating the prompt
        llm.eval(tokens)
        elapsed = now_ms() - t0

        times_ms.append(elapsed)
        tok_rates.append(n / (elapsed / 1000) if elapsed > 0 else 0)

    prefill_results.append({
        "prompt_tokens":   len(llm.tokenize(prompt.encode(), add_bos=True)),
        "median_ms":       round(median(times_ms), 2),
        "tokens_per_s":    round(median(tok_rates), 1),
    })

record("prefill", {
    "cases":           prefill_results,
    "avg_tokens_per_s": round(statistics.mean(r["tokens_per_s"] for r in prefill_results), 1),
})

# ── 4. Generation throughput ──────────────────────────────────────────────────

GEN_PROMPT = "Count from 1 to 100 as fast as possible:"

gen_times   = []
gen_tok_s   = []

for _ in range(args.n_repeat):
    llm.reset()
    tokens = llm.tokenize(GEN_PROMPT.encode(), add_bos=True)
    llm.eval(tokens)

    t0 = now_ms()
    count = 0
    for tok in llm.generate(tokens, temp=0.8, top_k=40, top_p=0.95):
        count += 1
        if count >= args.n_gen:
            break
    elapsed = now_ms() - t0

    gen_times.append(elapsed)
    gen_tok_s.append(count / (elapsed / 1000) if elapsed > 0 else 0)

record("generation", {
    "prompt":         GEN_PROMPT,
    "n_gen":          args.n_gen,
    "median_ms":      round(median(gen_times), 2),
    "median_tok_s":   round(median(gen_tok_s), 2),
    "ms_per_token":   round(median(gen_times) / args.n_gen, 2),
})

# ── 5. Grammar-constrained generation ────────────────────────────────────────

try:
    # llama-cpp-python supports grammar via LlamaGrammar
    GBNF = 'root ::= "positive" | "negative" | "neutral"'
    grammar = LlamaGrammar.from_string(GBNF)

    grammar_times = []
    grammar_outputs = []

    for _ in range(args.n_repeat):
        llm.reset()
        prompt = "Review: 'This product is excellent!' Sentiment:"
        t0 = now_ms()
        output = llm(
            prompt,
            max_tokens  = 8,
            grammar     = grammar,
            temperature = 0.0,
        )
        elapsed = now_ms() - t0
        grammar_times.append(elapsed)
        text = output["choices"][0]["text"].strip()
        grammar_outputs.append(text)

    record("grammar_constrained", {
        "gbnf":         GBNF,
        "outputs":      grammar_outputs,
        "all_valid":    all(o in ("positive", "negative", "neutral") for o in grammar_outputs),
        "median_ms":    round(median(grammar_times), 2),
        "supported":    True,
    })
except Exception as e:
    record("grammar_constrained", {
        "supported": False,
        "error":     str(e),
    })

# ── 6. KV snapshot ───────────────────────────────────────────────────────────
# llama-cpp-python 0.3.x: save_state() returns bytes (in-memory).
# Earlier versions used save_state(path) with file I/O.

try:
    llm.reset()
    prompt = "Hello, I am a benchmark. Please remember this message."
    tokens = llm.tokenize(prompt.encode(), add_bos=True)
    llm.eval(tokens)
    n_past_after_prefill = llm.n_tokens

    save_times, load_times = [], []
    state_size_kb = -1

    for _ in range(args.n_repeat * 2):
        t0 = now_ms()
        state = llm.save_state()
        save_times.append(now_ms() - t0)
        state_size_kb = round(len(state) / 1024, 1)

        t0 = now_ms()
        llm.load_state(state)
        load_times.append(now_ms() - t0)

    record("kv_snapshot", {
        "method":        "save_state()/load_state() - in-memory bytes (0.3.x API)",
        "n_past":        n_past_after_prefill,
        "state_size_kb": state_size_kb,
        "save_ms":       round(median(save_times), 3),
        "load_ms":       round(median(load_times), 3),
        "in_memory":     True,
        "note":          "llama-cpp-python 0.3.x returns bytes; no file I/O.",
    })
except Exception as e:
    # Fallback: try old file-based API (< 0.3.x)
    try:
        import tempfile, os as _os
        llm.reset()
        tokens = llm.tokenize(b"Hello benchmark", add_bos=True)
        llm.eval(tokens)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            state_path = tmp.name
        t0 = now_ms(); llm.save_state(state_path); save_ms = round(now_ms() - t0, 2)
        state_size_kb = round(_os.path.getsize(state_path) / 1024, 1) if _os.path.exists(state_path) else -1
        t0 = now_ms(); llm.load_state(state_path); load_ms_state = round(now_ms() - t0, 2)
        try: _os.unlink(state_path)
        except: pass
        record("kv_snapshot", {
            "method": "save_state(path)/load_state(path) - file-based (legacy API)",
            "n_past": llm.n_tokens, "state_size_kb": state_size_kb,
            "save_ms": save_ms, "load_ms": load_ms_state,
            "in_memory": False, "note": "llama-cpp-python < 0.3.x file-based API.",
        })
    except Exception as e2:
        record("kv_snapshot", {"supported": False, "error": str(e2)})

# ── 7. Detokenization ─────────────────────────────────────────────────────────

detok_times = []
sample_tokens = llm.tokenize(b"Hello world this is a test of detokenization speed", add_bos=False)
for _ in range(args.n_repeat * 10):
    t0 = now_ms()
    text = llm.detokenize(sample_tokens)
    detok_times.append(now_ms() - t0)

record("detokenization", {
    "n_tokens":     len(sample_tokens),
    "median_ms":    round(median(detok_times), 4),
})

# ── 8. Memory footprint ───────────────────────────────────────────────────────

record("memory", {
    "rss_after_load_mb":  round(rss_mb(), 1),
    "rss_delta_load_mb":  round(rss_mb() - mem_before, 1),
    "psutil_available":   HAS_PSUTIL,
})

# ── 9. Context creation ─────────────────────────────────────────────────────
# llama-cpp-python doesn't expose context creation separately from model load.
# We time the Llama() constructor with the model already cached by the OS.
# This measures context allocation overhead (not model loading from disk).

try:
    ctx_times = []
    for _ in range(3):
        gc.collect()
        t0 = now_ms()
        tmp_llm = Llama(
            model_path   = MODEL,
            n_gpu_layers = ngl,
            n_ctx        = args.n_ctx,
            n_batch      = 512,
            verbose      = False,
            seed         = args.seed,
        )
        elapsed = now_ms() - t0
        ctx_times.append(elapsed)
        # Free the temporary instance before next iteration
        del tmp_llm
        gc.collect()

    record("context_creation", {
        "median_ms": round(median(ctx_times), 2),
        "all_ms":    [round(t, 2) for t in ctx_times],
        "note":      "Llama() constructor (model cached in OS page cache)",
    })
except Exception as e:
    # Creating multiple instances may fail on low VRAM
    record("context_creation", {
        "median_ms": -1,
        "error":     str(e),
        "note":      "Llama() constructor failed - likely VRAM exhaustion",
    })

# ── 10. Single-token decode loop ────────────────────────────────────────────
# Generate tokens one at a time in a tight loop.
# This measures the per-token overhead of llama-cpp-python's generate() iterator.

try:
    N_SINGLE = 200

    llm.reset()
    prompt_tokens = llm.tokenize(b"Write a long story about a knight:", add_bos=True)
    llm.eval(prompt_tokens)

    token_times = []
    t_total_start = now_ms()

    for tok in llm.generate(prompt_tokens, temp=0.8, top_k=40, top_p=0.95, reset=False):
        t_tok = now_ms()
        token_times.append(t_tok)
        if len(token_times) >= N_SINGLE:
            break

    t_total_end = now_ms()
    total_ms = t_total_end - t_total_start

    # Compute per-token deltas from consecutive timestamps
    if len(token_times) >= 2:
        deltas = []
        # First token delta: from total_start to first token
        deltas.append(token_times[0] - t_total_start)
        for i in range(1, len(token_times)):
            deltas.append(token_times[i] - token_times[i - 1])
        median_ms_per_tok = median(deltas)
    else:
        deltas = []
        median_ms_per_tok = total_ms / max(len(token_times), 1)

    n_actual = len(token_times)
    tok_per_s = n_actual / (total_ms / 1000) if total_ms > 0 else 0

    record("single_token_loop", {
        "n_tokens":            n_actual,
        "tokens_per_s":        round(tok_per_s, 2),
        "median_ms_per_token": round(median_ms_per_tok, 3),
        "total_ms":            round(total_ms, 2),
        "note":                "Single-step generate() loop - includes Python iterator overhead",
    })
except Exception as e:
    err("single_token_loop", str(e))

# ── 11. Vocab operations ────────────────────────────────────────────────────
# Measure tokenize() and detokenize() throughput over many calls.

try:
    N_VOCAB_OPS = 1000

    # a) tokenize() short text x1000
    short_text = b"The quick brown fox jumps over the lazy dog."
    t0 = now_ms()
    for _ in range(N_VOCAB_OPS):
        llm.tokenize(short_text)
    tok_total_ms = now_ms() - t0
    tok_calls_per_s = N_VOCAB_OPS / (tok_total_ms / 1000) if tok_total_ms > 0 else 0

    # b) detokenize() x1000
    vocab_sample_tokens = llm.tokenize(short_text, add_bos=False)
    t0 = now_ms()
    for _ in range(N_VOCAB_OPS):
        llm.detokenize(vocab_sample_tokens)
    detok_total_ms = now_ms() - t0
    detok_calls_per_s = N_VOCAB_OPS / (detok_total_ms / 1000) if detok_total_ms > 0 else 0

    record("vocab_operations", {
        "tokenize": {
            "n":            N_VOCAB_OPS,
            "total_ms":     round(tok_total_ms, 2),
            "calls_per_s":  round(tok_calls_per_s, 0),
        },
        "detokenize": {
            "n":            N_VOCAB_OPS,
            "total_ms":     round(detok_total_ms, 2),
            "calls_per_s":  round(detok_calls_per_s, 0),
        },
    })
except Exception as e:
    err("vocab_operations", str(e))


# ── Stress ────────────────────────────────────────────────────────────────────

if args.stress:

    # S1. Sustained generation with 32-token sliding window
    LONG_PROMPT = ("Write a detailed technical essay about transformer architecture, "
        "attention mechanisms, and how large language models work. "
        "Cover embeddings, positional encoding, multi-head attention, "
        "feed-forward layers, and the training process. Be thorough.")
    N_LONG = min(args.n_gen * 8, 512)
    try:
        llm.reset()
        tl = llm.tokenize(LONG_PROMPT.encode(), add_bos=True)
        llm.eval(tl)
        wins, ws, wc, tc = [], now_ms(), 0, 0
        t0 = now_ms()
        for tok in llm.generate(tl, temp=0.8, top_k=40, top_p=0.95):
            tc += 1; wc += 1
            if wc == 32:
                wins.append(round(32 / ((now_ms()-ws)/1000), 1))
                ws = now_ms(); wc = 0
            if tc >= N_LONG: break
        tm = now_ms() - t0
        record("stress_sustained_generation", {
            "n_tokens_generated": tc, "n_prompt_tokens": len(tl),
            "overall_tok_s": round(tc/(tm/1000), 2) if tm>0 else 0,
            "window_tok_s": wins, "t_eval_ms": round(tm, 2),
            "note": "32-token sliding window",
        })
    except Exception as e:
        err("stress_sustained_generation", str(e))

    # S2. Back-to-back sessions
    try:
        TURNS = ["What is 2+2?","And what is 3+3?","What about 10*10?",
                 "Name the first 5 prime numbers.","What color is the sky?",
                 "How many days in a week?","What is the capital of France?",
                 "How many continents are there?","What is pi approximately?","Is water wet?"]
        stimes = []
        for turn in TURNS:
            llm.reset()
            p = f"You are a helpful assistant.\nUser: {turn}\nAssistant:"
            ts = llm.tokenize(p.encode(), add_bos=True); llm.eval(ts)
            t1 = now_ms(); n = 0
            for tok in llm.generate(ts, temp=0.8, top_k=40):
                n += 1
                if n >= 16: break
            stimes.append(round(now_ms()-t1, 2))
        record("stress_back_to_back", {
            "n_sessions": len(TURNS), "session_ms": stimes,
            "median_session_ms": round(median(stimes), 2),
            "min_session_ms": round(min(stimes), 2),
            "max_session_ms": round(max(stimes), 2),
            "note": "prefill + 16 tokens per session",
        })
    except Exception as e:
        err("stress_back_to_back", str(e))

    # S3. Context fill pressure
    try:
        ft = llm.tokenize(("The quick brown fox jumps over the lazy dog. "*100).encode(),
                          add_bos=False)[:int(args.n_ctx*0.75)-32]
        llm.reset()
        t0 = now_ms(); llm.eval(ft); fm = now_ms()-t0
        t0 = now_ms(); n = 0
        for tok in llm.generate(ft, temp=0.8, top_k=40):
            n += 1
            if n >= 32: break
        gm = now_ms()-t0
        record("stress_context_pressure", {
            "n_ctx": args.n_ctx, "fill_tokens": len(ft),
            "fill_pct": round(len(ft)/args.n_ctx*100, 1),
            "prefill_ms": round(fm, 2),
            "prefill_tok_s": round(len(ft)/(fm/1000), 1) if fm>0 else 0,
            "gen_on_full_ms": round(gm, 2),
            "gen_on_full_tok_s": round(n/(gm/1000), 1) if gm>0 else 0,
            "note": "prefill + generation at 75% context fill",
        })
    except Exception as e:
        err("stress_context_pressure", str(e))

    # S4. Sampler throughput approximation
    try:
        llm.reset()
        pt = llm.tokenize(b"Hello", add_bos=True); llm.eval(pt)
        N = 1000; t0 = now_ms()
        for _ in range(N):
            next(llm.generate(pt, temp=0.8, top_k=40, reset=False))
        sm = now_ms()-t0
        record("stress_sampler_throughput", {
            "n_samples": N, "total_ms": round(sm, 2),
            "avg_us_per_call": round(sm*1000/N, 2),
            "samples_per_s": round(N/(sm/1000), 0),
            "note": "approx via generate() single-step (includes decode overhead)",
        })
    except Exception as e:
        err("stress_sampler_throughput", str(e))

# ── Output ────────────────────────────────────────────────────────────────────

print(json.dumps(RESULTS, indent=2))
