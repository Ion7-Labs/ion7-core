#!/usr/bin/env python3
"""
benchmark/gen_charts.py
Generate SVG charts from ion7-core benchmark JSON results.

Usage:
    python3 benchmark/gen_charts.py
    python3 benchmark/gen_charts.py --out benchmark/charts
    python3 benchmark/gen_charts.py --lua benchmark/last_results_lua.json \
                                    --python benchmark/last_results_python.json \
                                    --stability benchmark/last_results_stability.json
"""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("error: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# ── Palette ───────────────────────────────────────────────────────────────────

C_ION7    = "#4f9cf9"   # blue
C_PY      = "#f97a4f"   # orange
C_GREEN   = "#34c759"   # speedup high
C_GREEN2  = "#6bd98a"   # speedup mid
C_BLUE2   = "#a8c8f8"   # speedup low
C_GRID    = "#efefef"
C_FG      = "#1c1c2e"
C_DIM     = "#8892a4"
C_BG      = "#ffffff"

plt.rcParams.update({
    "figure.facecolor":   C_BG,
    "axes.facecolor":     C_BG,
    "axes.edgecolor":     C_GRID,
    "axes.grid":          True,
    "grid.color":         C_GRID,
    "grid.linewidth":     1.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "text.color":         C_FG,
    "axes.labelcolor":    C_DIM,
    "xtick.color":        C_DIM,
    "ytick.color":        C_DIM,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "font.family":        "DejaVu Sans",
    "font.size":          11,
})

_out_dir = "benchmark/charts"


def _save(fig, name):
    os.makedirs(_out_dir, exist_ok=True)
    path = os.path.join(_out_dir, name)
    fig.savefig(path, format="svg", bbox_inches="tight",
                metadata={"Creator": "ion7-core gen_charts.py"})
    plt.close(fig)
    print(f"  → {path}")


def _bar_color(ratio):
    if ratio >= 10:  return C_GREEN
    if ratio >= 2:   return C_GREEN2
    return C_BLUE2


# ── Chart 1: Speedup ratios ───────────────────────────────────────────────────

def chart_speedup(lua: dict, py: dict):
    b_lua = lua["benchmarks"]
    b_py  = py["benchmarks"]

    def _tok_ratio(lua_val, py_val):
        """Higher tok/s is better → ratio = lua/py."""
        return lua_val / py_val if py_val else 0

    def _ms_ratio(lua_ms, py_ms):
        """Lower ms is better → ratio = py/lua."""
        return py_ms / lua_ms if lua_ms else 0

    lua_ctx = b_lua["context_creation"]["cases"][0]["median_ms"]
    py_ctx  = b_py["context_creation"].get("median_ms", None)

    rows = [
        ("tokenize",    _tok_ratio(b_lua["tokenization"]["avg_tokens_per_s"],
                                   b_py["tokenization"]["avg_tokens_per_s"])),
        ("generation",  _tok_ratio(b_lua["generation"]["median_tok_s"],
                                   b_py["generation"]["median_tok_s"])),
        ("grammar",     _ms_ratio(b_lua["grammar_constrained"]["median_ms"],
                                  b_py["grammar_constrained"]["median_ms"])),
        ("model load",  _ms_ratio(b_lua["model_load"]["load_ms"],
                                  b_py["model_load"]["load_ms"])),
        ("prefill avg", _tok_ratio(b_lua["prefill"]["avg_tokens_per_s"],
                                   b_py["prefill"]["avg_tokens_per_s"])),
        ("detokenize",  _ms_ratio(b_lua["detokenization"]["median_ms"],
                                  b_py["detokenization"]["median_ms"])),
    ]
    if py_ctx:
        rows.append(("ctx create", _ms_ratio(lua_ctx, py_ctx)))

    rows.sort(key=lambda r: r[1])   # ascending → biggest bar at top
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [_bar_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C_BG)

    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3,
                   linewidth=0)

    ax.axvline(1.0, color=C_DIM, linewidth=1.0, linestyle="--", zorder=2,
               alpha=0.7)

    for bar, val in zip(bars, values):
        lbl = f"{val:.0f}×" if val >= 5 else f"{val:.2f}×"
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                lbl, va="center", ha="left", fontsize=10,
                fontweight="bold", color=C_FG)

    ax.set_xlabel("× faster than llama-cpp-python", labelpad=10)
    ax.set_title("ion7-core speedup vs llama-cpp-python",
                 fontsize=13, fontweight="bold", pad=16, color=C_FG)
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    py_ver = py.get("version", "")
    fig.text(0.99, 0.01, f"llama-cpp-python {py_ver}  ·  ion7-core {lua.get('version','')}",
             ha="right", va="bottom", fontsize=8, color=C_DIM)

    fig.tight_layout()
    _save(fig, "speedup.svg")


# ── Chart 2: Side-by-side performance ────────────────────────────────────────

def chart_compare(lua: dict, py: dict):
    b_lua = lua["benchmarks"]
    b_py  = py["benchmarks"]

    panels = [
        {
            "title":   "Token generation",
            "ylabel":  "tok / s",
            "ion7":    b_lua["generation"]["median_tok_s"],
            "python":  b_py["generation"]["median_tok_s"],
            "higher":  True,
        },
        {
            "title":   "Prompt prefill (avg)",
            "ylabel":  "tok / s",
            "ion7":    b_lua["prefill"]["avg_tokens_per_s"],
            "python":  b_py["prefill"]["avg_tokens_per_s"],
            "higher":  True,
        },
        {
            "title":   "Model load",
            "ylabel":  "ms",
            "ion7":    b_lua["model_load"]["load_ms"],
            "python":  b_py["model_load"]["load_ms"],
            "higher":  False,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor(C_BG)

    for ax, p in zip(axes, panels):
        iv, pv = p["ion7"], p["python"]
        ion7_wins = (p["higher"] and iv >= pv) or (not p["higher"] and iv <= pv)
        bar_colors = [C_ION7, C_PY] if ion7_wins else [C_PY, C_ION7]
        labels = ["ion7-core", "llama-cpp-py"]

        bars = ax.bar(labels, [iv, pv], color=bar_colors, width=0.42,
                      zorder=3, linewidth=0)

        top = max(iv, pv)
        for bar, val in zip(bars, [iv, pv]):
            if p["ylabel"] == "tok / s" and val >= 1000:
                lbl = f"{val/1000:.1f}k"
            else:
                lbl = f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.03,
                    lbl, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=C_FG)

        ratio = (iv / pv) if p["higher"] else (pv / iv)
        ratio_lbl = f"{ratio:.2f}×"
        ratio_color = C_GREEN if ion7_wins else C_PY
        ax.text(0.5, 0.97, f"ion7 {ratio_lbl}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold", color=ratio_color)

        ax.set_title(p["title"], fontsize=11, fontweight="bold",
                     pad=12, color=C_FG)
        ax.set_ylabel(p["ylabel"], labelpad=6)
        ax.set_ylim(0, top * 1.25)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

    model = lua.get("model", "")
    fig.suptitle(
        f"{model}  ·  n_ctx {lua.get('n_ctx','')}  ·  n_repeat {lua.get('n_repeat','')}",
        fontsize=10, color=C_DIM, y=1.02)

    fig.tight_layout()
    _save(fig, "compare.svg")


# ── Chart 3: Stability ────────────────────────────────────────────────────────

def chart_stability(stab: dict):
    chk    = stab["checkpoints"]
    tokens = [c["tokens"]  for c in chk]
    tps    = [c["tps"]     for c in chk]
    rss    = [c["rss_mb"]  for c in chk]

    fig, (ax_rss, ax_tps) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(hspace=0.06)

    # — RSS panel —
    ax_rss.plot(tokens, rss, color=C_ION7, linewidth=2.5, zorder=3,
                solid_capstyle="round")
    ax_rss.fill_between(tokens, rss, min(rss) - 1, alpha=0.10, color=C_ION7)
    ax_rss.set_ylabel("RSS (MB)", labelpad=10)
    ax_rss.set_ylim(min(rss) - 8, max(rss) + 12)
    ax_rss.tick_params(axis="x", length=0)
    ax_rss.tick_params(axis="y", length=0)
    ax_rss.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    drift = stab.get("rss_drift_mb", max(rss) - min(rss))
    ax_rss.text(0.98, 0.85, f"post-warmup drift: +{drift:.1f} MB",
                transform=ax_rss.transAxes, ha="right", va="top",
                fontsize=9, color=C_DIM)

    n_tok   = stab.get("n_tokens_target", tokens[-1])
    resets  = stab.get("context_resets", "?")
    model   = stab.get("model", "")
    ax_rss.set_title(
        f"Stability — {n_tok // 1000}k tokens · {resets} KV resets · {model}",
        fontsize=12, fontweight="bold", pad=14, color=C_FG)

    # — tok/s panel —
    ax_tps.plot(tokens, tps, color=C_GREEN, linewidth=2.5, zorder=3,
                solid_capstyle="round")
    ax_tps.fill_between(tokens, tps, min(tps) - 0.2, alpha=0.10, color=C_GREEN)
    ax_tps.set_ylabel("tok / s", labelpad=10)
    ax_tps.set_xlabel("tokens generated", labelpad=10)
    tps_range = max(tps) - min(tps)
    ax_tps.set_ylim(min(tps) - max(1, tps_range * 3),
                    max(tps) + max(1, tps_range * 3))
    ax_tps.tick_params(axis="x", length=0)
    ax_tps.tick_params(axis="y", length=0)
    ax_tps.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x) // 1000}k"))
    ax_tps.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax_tps.text(0.98, 0.12,
                f"variance: ±{tps_range / 2:.2f} tok/s",
                transform=ax_tps.transAxes, ha="right", va="bottom",
                fontsize=9, color=C_DIM)

    for ax in (ax_rss, ax_tps):
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "stability.svg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate SVG charts for ion7-core benchmarks.")
    ap.add_argument("--lua",       default="benchmark/last_results_lua.json")
    ap.add_argument("--python",    default="benchmark/last_results_python.json")
    ap.add_argument("--stability", default="benchmark/last_results_stability.json")
    ap.add_argument("--out",       default="benchmark/charts")
    args = ap.parse_args()

    global _out_dir
    _out_dir = args.out

    if not os.path.exists(args.lua):
        print(f"error: {args.lua} not found — run 'make bench' first", file=sys.stderr)
        sys.exit(1)

    with open(args.lua) as f:
        lua = json.load(f)

    py = None
    if os.path.exists(args.python):
        with open(args.python) as f:
            py = json.load(f)
    else:
        print(f"  (no Python results at {args.python} — skipping speedup/compare)")

    print("Generating charts...")

    if py:
        chart_speedup(lua, py)
        chart_compare(lua, py)

    if os.path.exists(args.stability):
        with open(args.stability) as f:
            stab = json.load(f)
        chart_stability(stab)
    else:
        print(f"  (no stability results at {args.stability} — skipping)")

    print("Done.")


if __name__ == "__main__":
    main()
