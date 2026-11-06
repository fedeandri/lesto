# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Chart llmprobe reports for a PR: one line per arm across context size, every point labelled.

    uv run tests/bench_chart.py --out charts/ main='~/claude-tmp/bench-main-*/*.json' pr='~/claude-tmp/bench-pr-*/*.json'

Each ARM=GLOB names one arm and the llmprobe reports (`tests/bench.sh`, `llmprobe --save`) that
measured it. Several reports per arm (arms alternated across runs) are reduced to the median per
context size. Reads `bench.contextScaling`; measures nothing itself. Writes decode.png,
prefill.png and ttft.png, with the machine, model, engine version and report count in the caption.
"""
import argparse
import glob
import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#3b7dd8", "#e0822a", "#2e9e5b", "#c43d5f", "#8a5cc8", "#1a9aa6"]
METRICS = [
    ("decodeTokPerSec", "decode.png", "Generation across context size", "tok/s", "{:.1f}"),
    ("prefillTokPerSec", "prefill.png", "Prompt processing across context size", "tok/s", "{:.0f}"),
    ("ttftMs", "ttft.png", "Time to first token across context size", "ms", "{:.0f}"),
]


def load_arm(pattern):
    paths = sorted(glob.glob(os.path.expanduser(pattern)))
    if not paths:
        raise SystemExit(f"no reports match {pattern}")
    reports = [json.loads(Path(p).read_text()) for p in paths]
    points = {}
    for r in reports:
        for row in r["bench"].get("contextScaling") or []:
            points.setdefault(row["targetTokens"], []).append(row)
    return reports, points


def ctx_label(n):
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def caption(arms):
    first = next(iter(arms.values()))[0][0]
    m, t = first["machine"], first["target"]
    counts = ", ".join(f"{name} n={len(reports)}" for name, (reports, _) in arms.items())
    version = (t.get("engineSettings") or {}).get("version", "?")
    return (f"{m['cpu']}, {m['memGB']} GB · {t['model']} · {t['engine']} {version} · "
            f"llmprobe contextScaling, median of {counts}")


def chart(arms, metric, title, ylabel, fmt, path):
    ctxs = sorted({c for _, points in arms.values() for c in points})
    xi = {c: i for i, c in enumerate(ctxs)}
    lines = {}
    for name, (_, points) in arms.items():
        vals = {c: [row[metric] for row in points.get(c, []) if row.get(metric) is not None] for c in ctxs}
        lines[name] = {c: statistics.median(v) for c, v in vals.items() if v}
    top = {c: max((ln[c] for ln in lines.values() if c in ln), default=None) for c in ctxs}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (name, line) in enumerate(lines.items()):
        color = COLORS[k % len(COLORS)]
        ax.plot([xi[c] for c in line], list(line.values()), marker="o", color=color, linewidth=2, markersize=5, label=name)
        for c, v in line.items():  # the highest value at a context labels above its point, the rest below
            ax.annotate(fmt.format(v), (xi[c], v), textcoords="offset points",
                        xytext=(0, 7 if v == top[c] else -13), ha="center", fontsize=8, color=color)
    ax.set_xticks(range(len(ctxs)), [ctx_label(c) for c in ctxs])
    ax.set_xlabel("context (prompt tokens)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    fig.text(0.01, 0.005, caption(arms), fontsize=7, color="#555", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+", metavar="ARM=GLOB")
    ap.add_argument("--out", default="charts")
    a = ap.parse_args()
    arms = {}
    for spec in a.arms:
        name, _, pattern = spec.partition("=")
        if not pattern:
            raise SystemExit(f"expected ARM=GLOB, got {spec}")
        arms[name] = load_arm(pattern)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for metric, fname, title, ylabel, fmt in METRICS:
        chart(arms, metric, title, ylabel, fmt, out / fname)
        print("wrote", out / fname)


if __name__ == "__main__":
    main()
