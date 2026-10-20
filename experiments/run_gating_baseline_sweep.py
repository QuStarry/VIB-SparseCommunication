#!/usr/bin/env python3
"""Naive-gating control experiment: VIB vs random-drop vs cosine top-k.

Motivation
----------
The VIB gate is the paper's central mechanism, yet every reported comparison so
far is against *Full* communication (no gate at all). That isolates the effect of
"gating versus not gating", not "this gate versus a cheaper gate". A reviewer is
therefore free to claim the gain comes from merely dropping messages. This driver
removes that confound by running two zero-cost gating policies through the exact
same harness, backbone and prompt stack:

  random   Bernoulli keep at a probability calibrated to the VIB arm's own
           admission rate (matched message budget, no semantic signal at all).
  topk     cosine-diversity gate: drop a candidate whose maximum cosine against
           the already-broadcast history exceeds a threshold. Same shared encoder
           as VIB, but no goal term, no beta schedule, no role prior, no guards.

Budget matching
---------------
Both naive arms are pinned to the compression ratio the VIB anchor realises on
the same day. The VIB anchor is re-run here on purpose so that all three arms
share one backbone snapshot: the public ``deepseek-chat`` alias is re-pointed by
the vendor without notice, which is why these controls run against the pinned
release (see the snapshot-drift note in the paper).

The default top-k grid is calibrated from measured compression, not from an
offline estimate: lowering theta by 0.05 buys roughly 9 pp of extra message
reduction on this subset, so theta=0.75 lands on the VIB anchor's own budget
(~30% compression) and serves as the matched point, while the wider thetas
trace the budget/accuracy trend above it.

Restart safety / fail-fast
--------------------------
Identical contract to ``run_hyper_sweep.py``: valid points are skipped on
restart, invalid ones are archived to ``invalid/`` and re-run, and a freshly
computed point that fails the guard aborts the sweep unless ``--keep-going`` is
given (gateway quota exhaustion silently scores every sample wrong).

Usage
-----
    python3 experiments/run_gating_baseline_sweep.py --dry-run
    python3 experiments/run_gating_baseline_sweep.py --limit 3 --workers 2   # smoke
    python3 experiments/run_gating_baseline_sweep.py                        # full N=200
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hyper_sweep import (  # noqa: E402
    MAX_INVALID_TASKS,
    _broken_task_count,
    _is_valid,
    _load_summary,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "run_multi_benchmark.py"
OUTDIR = ROOT / "results" / "gating_sweep"
INVALID_DIR = OUTDIR / "invalid"

# Ordered so the matched-budget point lands first, since that is the one the
# paper's control comparison rests on.
DEFAULT_TOPK_THRESHOLDS = [0.75, 0.80, 0.85, 0.90]

ANCHOR_NAME = "gating_vib_anchor"

# The hyper-parameter sweep keeps a 95% floor because it compares many nearby
# configurations. This control is a different question -- it only needs the
# backbone to be *healthy* (no gateway exhaustion), and the Full arm sits at
# ~94.5% here, so a 95% floor would reject perfectly good data.
GATING_MIN_FULL_PASS_RATE = 90.0


def _tag(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def _evaluate_point(
    name: str,
    extra_flags: list,
    args,
    env: dict,
    min_full_pass_rate: float = GATING_MIN_FULL_PASS_RATE,
) -> tuple:
    """Run one sweep point (or reuse a valid existing one).

    Returns ``(summary, reused, valid)``.
    """
    out = OUTDIR / f"{name}.json"
    summary = _load_summary(out)

    if out.exists() and not args.no_resume and not args.dry_run:
        if _is_valid(summary, out, min_full_pass_rate):
            print(f"[skip] {name} already valid "
                  f"(full_pass_rate={summary.get('full_pass_rate')}, "
                  f"compression={summary.get('msg_compression_pct')}%, "
                  f"broken={_broken_task_count(out)})", flush=True)
            return summary, True, True
        archived = INVALID_DIR / out.name
        shutil.move(str(out), str(archived))
        print(f"[redo] {name} invalid "
              f"(full_pass_rate={summary.get('full_pass_rate')}) "
              f"-> archived {archived.name}", flush=True)

    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--datasets", "gsm8k",
        "--limit", str(args.limit),
        "--workers", str(args.workers),
        "--model", args.model,
        "--provider", args.provider,
        *extra_flags,
        "--output", str(out),
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {}, False, True
    subprocess.run(cmd, check=True, env=env)

    summary = _load_summary(out)
    valid = _is_valid(summary, out, min_full_pass_rate)
    if not valid:
        print(f"[abort] {name} came back invalid "
              f"(full_pass_rate={summary.get('full_pass_rate')}, "
              f"broken={_broken_task_count(out)}).", flush=True)
    return summary, False, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="GSM8K sample size.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="deepseek-v4-flash-0817",
                    help="Pinned backbone snapshot (the release the public "
                         "deepseek-chat alias serves).")
    ap.add_argument("--provider", default="private", choices=["deepseek", "private"])
    ap.add_argument("--topk-thresholds", type=float, nargs="+",
                    default=DEFAULT_TOPK_THRESHOLDS,
                    help="Cosine thresholds for the top-k diversity gate.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing outputs and re-run every point.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Keep sweeping even when a point comes back invalid.")
    ap.add_argument("--dry-run", action="store_true", help="Print cmds, do not launch.")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    INVALID_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("VIB_LLM_PROVIDER", args.provider)
    env.setdefault("VIB_LLM_MODEL", args.model)

    rows = []
    aborted = False

    # --- 1) Same-day VIB anchor -------------------------------------------
    anchor_summary, reused, valid = _evaluate_point(
        ANCHOR_NAME, ["--router-gating-mode", "vib"], args, env)
    rows.append({"arm": "vib", "name": ANCHOR_NAME, "flags": [],
                 "summary": anchor_summary, "reused": reused})

    if not valid and not args.keep_going:
        print("[abort] VIB anchor invalid -- cannot calibrate the naive arms.",
              flush=True)
        aborted = True

    # The naive arms are pinned to the anchor's realised admission rate so that
    # every gate pays the same number of messages.
    if not aborted and not args.dry_run:
        keep_rate = round(
            float(anchor_summary.get("vib_messages", 0))
            / max(1, int(anchor_summary.get("full_messages", 0))), 4)
    else:
        keep_rate = 0.70
    print(f"[calib] matched keep_rate = {keep_rate} "
          f"(anchor compression={anchor_summary.get('msg_compression_pct')}%)",
          flush=True)

    # --- 2) Naive gating arms ---------------------------------------------
    points = [(
        "gating_random_matched",
        "random",
        ["--router-gating-mode", "random", "--router-keep-rate", str(keep_rate)],
        keep_rate,
    )]
    for theta in args.topk_thresholds:
        points.append((
            f"gating_topk_{_tag(theta)}",
            "topk",
            ["--router-gating-mode", "topk", "--router-sim-threshold", str(theta)],
            theta,
        ))

    for name, arm, flags, knob in points:
        if aborted:
            break
        summary, reused, valid = _evaluate_point(name, flags, args, env)
        rows.append({"arm": arm, "name": name, "flags": flags, "knob": knob,
                     "summary": summary, "reused": reused})
        if not valid and not args.keep_going:
            print(f"[abort] {name} invalid -- stopping sweep.", flush=True)
            aborted = True
            break

    summary_path = OUTDIR / "gating_summary.json"
    json.dump(
        {"limit": args.limit, "model": args.model, "provider": args.provider,
         "matched_keep_rate": keep_rate,
         "min_full_pass_rate": GATING_MIN_FULL_PASS_RATE,
         "max_invalid_tasks": MAX_INVALID_TASKS,
         "aborted": aborted, "rows": rows},
        open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False,
    )
    print(f"\n[done] wrote {summary_path} with {len(rows)} rows"
          f"{' (ABORTED EARLY)' if aborted else ''}", flush=True)
    if aborted:
        sys.exit(2)


if __name__ == "__main__":
    main()
