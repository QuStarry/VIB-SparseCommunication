r"""
Phase 3-b · Figure 3-a: Pareto-optimal bubble chart.

x-axis  : VIB relative to Full → message compression (%)
y-axis  : VIB relative to Full → Δ Pass@1 (pp), with paired Bootstrap 95% CI whiskers
bubble  : total tasks (N)
color   : dataset (HumanEval / MBPP / GSM8K / MATH)

Outputs (paper/figures/):
  figure3_pareto_bubble.pdf   (vector, for paper)
  figure3_pareto_bubble.png   (raster preview)

Usage:
  python3 scripts/plot_pareto_3a.py
  python3 scripts/plot_pareto_3a.py --main results/main_with_math_merged.codeeval_fixed.json --outdir paper/figures
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")   # no display
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import analyze_multi_benchmark as an


# ---------------- Theme (brand-less, conference neutral) ----------------

PALETTE = {
    "HumanEval": "#1f77b4",
    "MBPP":      "#2ca02c",
    "GSM8K":     "#ff7f0e",
    "MATH":      "#9467bd",
}
EDGE = "#222222"
GRID = "#e5e7eb"


@dataclass
class Point:
    label: str
    ds: str
    x: float       # msg compression %
    y: float       # Δpp VIB-Full
    y_ci: Tuple[float, float]
    n: int
    color: str


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def build_points(main_path: Path) -> List[Point]:
    data = load_json(main_path)
    pts: List[Point] = []
    order = [
        ("humaneval", "HumanEval"), ("mbpp", "MBPP"),
        ("gsm8k", "GSM8K"), ("math", "MATH"),
    ]
    for ds, lab in order:
        if ds not in data["summary"]:
            continue
        s = data["summary"][ds]
        rows = [{
            "task_id": r["task_id"],
            "full_passed": r["full_passed"], "vib_passed": r["vib_passed"],
            "full_messages": r["full_messages"], "vib_messages": r["vib_messages"],
            "full_tokens": r["full_tokens"], "vib_tokens": r["vib_tokens"],
        } for r in data["detailed_results"][ds]]
        a = an.analyze_dataset(ds, rows, s)
        pts.append(Point(
            label=lab,
            ds=lab,
            x=float(s["msg_compression_pct"]),
            y=float(a["pass_delta_pp"]),
            y_ci=tuple(a["pass_delta_bootstrap_ci_95"]),
            n=int(s["total_tasks"]),
            color=PALETTE[lab],
        ))
    return pts


def savefig(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05, dpi=300)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"  ✔ wrote {pdf} ({pdf.stat().st_size} bytes)")
    print(f"  ✔ wrote {png} ({png.stat().st_size} bytes)")


def plot(pts: List[Point], outdir: Path, with_arrows: bool):
    # Bubble area follows sqrt(N), matching the manuscript caption.
    ns = np.array([p.n for p in pts])
    # Map N→area. Use sqrt for radius. Set min area to 180 (so MBPP doesn't vanish).
    n_ref = 1000.0
    base_area = 260.0
    sizes = base_area * np.sqrt(ns / n_ref)
    sizes = np.clip(sizes, 180, 1600)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # --- guides: x=0 (no msg compress) and y=0 (no accuracy change)
    ax.axhline(0, color="#9ca3af", linewidth=1.1, linestyle="-", zorder=0)
    ax.axvline(0, color="#9ca3af", linewidth=1.1, linestyle="-", zorder=0)

    # --- x>0 band (positive compression) shade light
    ax.axvspan(0, 50, facecolor="#eef7ff", alpha=0.4, zorder=-1)
    ax.set_xlim(-6, 42)
    ax.set_ylim(min(-4.2, min(p.y_ci[0] for p in pts) - 1.0),
                max(6.8, max(p.y_ci[1] for p in pts) + 1.0))

    # grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    # --- CI whiskers
    for p in pts:
        yerr_lo = p.y - p.y_ci[0]
        yerr_hi = p.y_ci[1] - p.y
        ax.errorbar(p.x, p.y,
                    yerr=np.array([[yerr_lo], [yerr_hi]]),
                    fmt="none",
                    ecolor=p.color, elinewidth=1.6, capsize=3.2, alpha=0.9, zorder=2)

    # --- bubbles
    sc = ax.scatter(
        [p.x for p in pts], [p.y for p in pts],
        s=sizes,
        c=[p.color for p in pts],
        alpha=0.78,
        edgecolors=EDGE,
        linewidths=1.0,
        zorder=3,
    )

    # --- dataset label near bubble
    for p, sz in zip(pts, sizes):
        r = np.sqrt(sz / np.pi)
        # convert points to data coords: offset by ~1 radius in data y
        off_y = 0.0 + (sz ** 0.5) * 0.022
        ax.annotate(
            f"{p.label}\nN={p.n}",
            xy=(p.x, p.y),
            xytext=(0, r + 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=p.color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=p.color, alpha=0.85, linewidth=0.8),
            zorder=4,
        )

    # axes labels
    ax.set_xlabel("Message Compression vs. Full-Comm (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Δ Pass@1 vs. Full-Comm (pp, 95% Boot CI)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3-a · Observed accuracy–communication trade-offs",
                 fontsize=12, fontweight="bold", pad=10, loc="left")

    # arrow: Pareto direction NE → better
    if with_arrows:
        ax.annotate("", xy=(38, 5.6), xytext=(14, -3.2),
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.2, linestyle="--"))
        ax.text(23, 1.3, "Pareto improvement", fontsize=8.5, color="#4b5563",
                rotation=28, fontstyle="italic")

    # ---- N size legend (custom) ----
    legend_n = [100, 1000, 5000]
    legend_sizes = base_area * np.sqrt(np.array(legend_n) / n_ref)
    legend_sizes = np.clip(legend_sizes, 180, 1600)
    handles = [
        plt.scatter([], [], s=s, c="#cccccc", alpha=0.85, edgecolors=EDGE, linewidths=1.0)
        for s in legend_sizes
    ]
    labels = [f"N = {n}" for n in legend_n]
    leg1 = ax.legend(handles, labels, title="Total tasks",
                     loc="lower right", scatterpoints=1,
                     frameon=True, framealpha=0.92,
                     labelspacing=1.5, borderpad=0.8, fontsize=8.5)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # ---- dataset color legend ----
    from matplotlib.lines import Line2D
    handles2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[l],
               markersize=10, markeredgecolor=EDGE, markeredgewidth=1.0, label=l)
        for l in (point.label for point in pts)
    ]
    ax.legend(handles=handles2, title="Dataset",
              loc="upper left", frameon=True, framealpha=0.92,
              fontsize=8.5)

    fig.tight_layout()
    savefig(fig, outdir, "figure3_pareto_bubble")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main_with_math_merged.codeeval_fixed.json",
                    help="主结果 JSON；默认使用 scripts/recompute_code_eval.py 修正过"
                         "代码基准通过标记的产物，与 paper/tables 保持同一数据源")
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--no-arrows", action="store_true", help="Disable Pareto arrow")
    args = ap.parse_args()

    pts = build_points(Path(args.main))
    print(f"[3-a] plotting {len(pts)} datasets…")
    for p in pts:
        print(f"   · {p.label:<10} x={p.x:+.2f}%  y={p.y:+.2f}pp  CI=[{p.y_ci[0]:+.2f}, {p.y_ci[1]:+.2f}]  N={p.n}")
    plot(pts, Path(args.outdir), with_arrows=(not args.no_arrows))


if __name__ == "__main__":
    main()
