r"""
Phase 3-b · Figure 3-b: GSM8K ablation panel — (left) frozen_only, (right) frozen+role_weights.

Current dataset coverage: 2 ablation runs →  2 data points per panel. To avoid a 2-point
"line" being mathematically meaningless, the plot renders as a **dual-metric grouped bar chart**:

  * For each ablation {Frozen-History only   (enable_gsm8k_frozen_history=T, role_weights=F)
                       {Frozen-History + Role-weights   (…, role_weights=T)}

  * Two bars per ablation:
        - Blue bar:  Δ Pass@1 VIB-Full (pp, left axis)
        - Orange bar: Message compression VIB/Full (%, right axis)

  * Each bar carries the absolute metric value on top, plus Wilson/Bootstrap CI whiskers
    as available from analyze_dataset (falls back to point only).

The structure is designed so that *when β₀/λ sweeps (e.g. 0.2/0.4/0.6/0.8 λ sweep) are run*,
the same script (extended with --ablations list) automatically renders a proper multi-point
ablation line chart without layout changes. We leave a TODO inside `load_from_args()` so
sweeps map trivially.

Outputs (paper/figures/):
  figure3_gsm8k_ablation_bar.pdf
  figure3_gsm8k_ablation_bar.png

Usage:
  python3 scripts/plot_ablation_3b.py
  python3 scripts/plot_ablation_3b.py \
      --abl1 results/gsm8k_ablation_frozen_only.json \
      --abl2 results/gsm8k_ablation_frozen_plus_role.json \
      --outdir paper/figures
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import analyze_multi_benchmark as an


PALETTE_DELTA = "#2563eb"
PALETTE_COMPR = "#ea580c"
EDGE = "#1f2937"
GRID = "#e5e7eb"


@dataclass
class AblationPoint:
    name: str        # display name on x-axis
    delta_pp: float
    delta_ci: Tuple[float, float]  # 95% boot CI of VIB-Full Δ
    msg_compress: float
    n: int


def load_point(path: Path, name: str, ds: str = "gsm8k") -> AblationPoint:
    data = json.loads(path.read_text())
    s = data["summary"][ds]
    rows = [{
        "task_id": r["task_id"],
        "full_passed": r["full_passed"], "vib_passed": r["vib_passed"],
        "full_messages": r["full_messages"], "vib_messages": r["vib_messages"],
        "full_tokens": r["full_tokens"], "vib_tokens": r["vib_tokens"],
    } for r in data["detailed_results"][ds]]
    a = an.analyze_dataset(ds, rows, s)
    return AblationPoint(
        name=name,
        delta_pp=float(a["pass_delta_pp"]),
        delta_ci=tuple(a["pass_delta_bootstrap_ci_95"]),
        msg_compress=float(s["msg_compression_pct"]),
        n=int(s["total_tasks"]),
    )


def savefig(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05, dpi=300)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"  ✔ wrote {pdf} ({pdf.stat().st_size} bytes)")
    print(f"  ✔ wrote {png} ({png.stat().st_size} bytes)")


def plot(points: List[AblationPoint], outdir: Path):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    xs = np.arange(len(points))
    width = 0.36

    # --- twin axis
    ax2 = ax1.twinx()
    ax2.set_facecolor("white")

    # yerr for Δpp
    lo = np.array([p.delta_pp - p.delta_ci[0] for p in points])
    hi = np.array([p.delta_ci[1] - p.delta_pp for p in points])
    bars1 = ax1.bar(xs - width / 2,
                    [p.delta_pp for p in points],
                    width,
                    yerr=np.stack([lo, hi], axis=0),
                    capsize=4.0,
                    color=PALETTE_DELTA, alpha=0.9,
                    edgecolor=EDGE, linewidth=0.9, label="Δ Pass@1 VIB−Full (pp)",
                    error_kw=dict(ecolor=EDGE, elinewidth=1.1))
    bars2 = ax2.bar(xs + width / 2,
                    [p.msg_compress for p in points],
                    width,
                    color=PALETTE_COMPR, alpha=0.86,
                    edgecolor=EDGE, linewidth=0.9, label="Msg compression (%)")

    # value labels on bars
    def _label(ax, bars, fmt, c):
        for b in bars:
            h = b.get_height()
            xy = (b.get_x() + b.get_width() / 2, h)
            ax.annotate(fmt(h),
                        xy=xy,
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=9, color=c, fontweight="bold")

    _label(ax1, bars1, lambda v: f"{v:+.2f}pp", PALETTE_DELTA)
    _label(ax2, bars2, lambda v: f"{v:+.2f}%", PALETTE_COMPR)

    # reference lines
    ax1.axhline(0, color="#9ca3af", linewidth=1.0, zorder=0)
    ax2.axhline(0, color="#9ca3af", linewidth=1.0, zorder=0, linestyle=":")

    # grid on left y only
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.7, color=GRID, zorder=0)
    ax1.set_axisbelow(True)

    # N= per x label
    labels = [f"{p.name}\n$N={p.n}$" for p in points]
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=10)

    ax1.set_xlabel("GSM8K ablation mode (§3 Method)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Δ Pass@1 VIB vs. Full (pp, 95% Boot CI)", fontsize=10.5, color=PALETTE_DELTA, fontweight="bold")
    ax2.set_ylabel("Message compression vs. Full (%)", fontsize=10.5, color=PALETTE_COMPR, fontweight="bold")
    ax1.set_title("Figure 3-b · Ablation: frozen-history × role-aware weights",
                  fontsize=12, fontweight="bold", pad=10, loc="left")

    # y limits: Δpp axis
    ymax = max(1.0, 1.4 * max(abs(p.delta_pp) + hi[i] for i, p in enumerate(points)))
    ax1.set_ylim(-ymax, ymax)
    ax2_lo = min(-2, 0.88 * min(p.msg_compress for p in points))
    ax2_hi = max(2, 1.15 * max(0.1, max(p.msg_compress for p in points)))
    ax2.set_ylim(ax2_lo, ax2_hi)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2,
               loc="upper center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, frameon=True, framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    savefig(fig, outdir, "figure3_gsm8k_ablation_bar")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl1", default="results/gsm8k_ablation_frozen_only.json",
                    help="Label: Frozen-History only")
    ap.add_argument("--abl2", default="results/gsm8k_ablation_frozen_plus_role.json",
                    help="Label: Frozen-History + Role weights")
    # Extension for future β/λ sweeps:
    # ap.add_argument("--ablations", nargs="+", action="append",
    #                 metavar=("NAME", "PATH"), help="e.g. --ablations λ=0.2 run1.json")
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()

    pts = [
        load_point(Path(args.abl1), "Frozen-history only"),
        load_point(Path(args.abl2), "Frozen-history + Role-weights"),
    ]
    print("[3-b] ablation bars:")
    for p in pts:
        print(f"   · {p.name:<30} Δpp={p.delta_pp:+.2f} CI=[{p.delta_ci[0]:+.2f},{p.delta_ci[1]:+.2f}]  compress={p.msg_compress:+.2f}%  N={p.n}")
    plot(pts, Path(args.outdir))


if __name__ == "__main__":
    main()
