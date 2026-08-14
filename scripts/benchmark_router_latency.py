r"""
MED-1: Benchmark VIB Router pure embedding latency (no LLM calls).

Measures latency (a) first-call model load (one-off) and (b) per-message evaluate_detailed() latency
sweeping |H| ∈ {1, 4, 8, 16, 32}.

Plots a bar chart (p50 / p95) and prints statistics.

Outputs:
  paper/figures/router_latency.png
  paper/figures/router_latency.pdf
  /tmp/router_latency.csv
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vib_router import VIBSparseCommRouter


def _random_paragraph(rng, size=120):
    words = (
        "optimiz quicksort merge insertion complexity average worst case runtime arrays sorting algorithm array stable "
        "edge boundary handle empty null pointer coder tester reviewer planner solver verifier test case "
        "assertion verify bug fix test cases docstring docstring code function return class method loop recursion "
        "dynamic programming greedy bfs dfs traversal binary search dp table dp table dict list map"
    ).split()
    return " ".join(rng.choice(words, size=size))


def main():
    rng = np.random.default_rng(20260813)
    print("[load] warming up model = VIBSparseCommRouter default (all-MiniLM-L6-v2) ...")
    t0 = time.perf_counter()
    VIBSparseCommRouter.warmup_shared_encoder()
    t1 = time.perf_counter()
    print(f"   warmup latency: {t1-t0:.2f}s")

    router = VIBSparseCommRouter()

    HS = [1, 4, 8, 16, 32]
    N_MSGS = 100
    print(f"[bench] running {N_MSGS} messages × |H| sweep {HS}")
    print()

    results = {}
    for h in HS:
        goal = "Optimize the sorting performance of this Python code."
        history = [_random_paragraph(rng, 140) for _ in range(h)]
        latencies = []
        # prime one dry-run for fair compare
        cand0 = _random_paragraph(rng, 140)
        _ = router.evaluate_detailed(cand0, history, goal, step_t=2, role="Coder")
        for i in range(N_MSGS):
            cand = _random_paragraph(rng, rng.integers(80, 200))
            role = rng.choice(["Architect", "Coder", "Reviewer"])
            step = int(rng.integers(1, 5))
            t_s = time.perf_counter()
            router.evaluate_detailed(cand, history, goal, step_t=step, role=role)
            t_e = time.perf_counter()
            latencies.append((t_e - t_s) * 1000)  # ms
        lat_ms = np.array(latencies)
        p50 = float(np.percentile(lat_ms, 50))
        p95 = float(np.percentile(lat_ms, 95))
        mean = float(lat_ms.mean())
        results[h] = {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean}
        print(f"   |H|={h:>2}:  p50={p50:6.2f} ms   p95={p95:6.2f} ms   mean={mean:6.2f} ms  "
              f"min={lat_ms.min():6.2f} ms  max={lat_ms.max():6.2f} ms")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(len(HS))
    w = 0.38
    p50s = [results[h]["p50_ms"] for h in HS]
    p95s = [results[h]["p95_ms"] for h in HS]
    ax.bar(xs - w / 2, p50s, w, label="p50", color="#2563eb", edgecolor="#111")
    ax.bar(xs + w / 2, p95s, w, label="p95", color="#ea580c", edgecolor="#111")
    for i, (p5, p9) in enumerate(zip(p50s, p95s)):
        ax.text(i - w/2, p5 + max(p50s) * 0.02, f"{p5:.1f}", ha="center", fontsize=8.5, color="#2563eb")
        ax.text(i + w/2, p9 + max(p95s) * 0.02, f"{p9:.1f}", ha="center", fontsize=8.5, color="#ea580c")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(h) for h in HS])
    ax.set_xlabel("|H| = number of messages in history")
    ax.set_ylabel("Latency per gate decision (ms)")
    ax.set_title("VIB router latency vs history length (all-MiniLM-L6-v2)")
    ax.grid(axis="y", linestyle=":", color="#e5e7eb")
    ax.legend()
    fig.tight_layout()
    out_dir = ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "router_latency.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "router_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] saved plot to {out_dir / 'router_latency.png'}")
    json.dump(results, open("/tmp/router_latency.json", "w"), indent=2)
    # CSV:
    with open("/tmp/router_latency.csv", "w") as f:
        f.write("history_size,p50_ms,p95_ms,mean_ms\n")
        for h in HS:
            f.write(f"{h},{results[h]['p50_ms']:.3f},{results[h]['p95_ms']:.3f},{results[h]['mean_ms']:.3f}\n")
    print(f"[done] csv/json → /tmp/router_latency.csv")


if __name__ == "__main__":
    main()
