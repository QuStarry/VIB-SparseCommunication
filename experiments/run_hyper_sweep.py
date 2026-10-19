#!/usr/bin/env python3
"""One-factor-at-a-time hyperparameter sweep (β₀, λ, γ) × GSM8K N=200 on private gateway.

Paper defaults anchor the sweep (from experiments/run_multi_benchmark.py):
  beta_0 = 3.2,  lambda_decay = 0.08,  gamma = -0.4

Sweep grids (OFAT: vary one, keep the other two fixed at the default):
  beta_0 ∈ {1.6, 2.4, 3.2, 4.0, 4.8}
  λ        ∈ {0.04, 0.06, 0.08, 0.10, 0.12}
  γ        ∈ {-0.8, -0.6, -0.4, -0.2,  0.0}

Restart safety
--------------
``LLMClient.chat`` swallows gateway failures and returns the error text as the
answer content, so a run killed by quota exhaustion still exits 0 and writes a
syntactically valid JSON — with every sample scored wrong.

Validity therefore needs two independent signals: the pass rate of the
uncompressed baseline (it never touches the router, so it should stay flat
across grid points) *and* the count of task-level exceptions, because a crash in
the VIB arm aborts the whole task and drags the Full number down with it.

Valid points are skipped on restart; invalid ones are archived under
``invalid/`` and re-run. If a freshly computed point turns out invalid the
sweep stops immediately (``--keep-going`` disables that) instead of burning
hours producing garbage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "run_multi_benchmark.py"
OUTDIR = ROOT / "results" / "sweep_v1"
INVALID_DIR = OUTDIR / "invalid"

SWEEP = OrderedDict([
    ("beta0",  {"flag": "--router-beta0",  "values": [1.6, 2.4, 3.2, 4.0, 4.8]}),
    ("lambda", {"flag": "--router-lambda", "values": [0.04, 0.06, 0.08, 0.10, 0.12]}),
    ("gamma",  {"flag": "--router-gamma",  "values": [-0.8, -0.6, -0.4, -0.2, 0.0]}),
])
DEFAULT_ANCHOR = {"beta0": 3.2, "lambda": 0.08, "gamma": -0.4}

# A point is only trustworthy when the uncompressed baseline (router-independent)
# is healthy AND no task died on an exception. Task-level exceptions wipe both
# arms, so they drag ``full_pass_rate`` down even though Full never touches the
# router — which is why the pass rate alone is not a sufficient guard.
MIN_FULL_PASS_RATE = 95.0
MAX_INVALID_TASKS = 2


def _name(param: str, val) -> str:
    tag = f"{param}_{val}".replace("-", "m").replace(".", "p")
    return f"sweep_{tag}"


def _load_summary(out: Path) -> dict:
    if not out.exists():
        return {}
    try:
        data = json.load(open(out, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("summary", {}).get("gsm8k", {})


def _broken_task_count(out: Path) -> int:
    """Tasks whose whole trace died on an exception (both arms lost)."""
    try:
        data = json.load(open(out, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 10**6
    traces = (data.get("trace_results") or {}).get("gsm8k") or []
    if not traces:
        return 10**6
    return sum(1 for t in traces if t.get("error_msg"))


def _is_valid(summary: dict, out: Path) -> bool:
    if not summary or not summary.get("total_tasks"):
        return False
    if float(summary.get("full_pass_rate") or 0.0) < MIN_FULL_PASS_RATE:
        return False
    return _broken_task_count(out) <= MAX_INVALID_TASKS


def _run(cmd, env, dry_run: bool) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=True, env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="GSM8K sample size.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="deepseek-v4-pro-0817", help="Private-gateway model id.")
    ap.add_argument("--provider", default="private", choices=["deepseek", "private"])
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
    for param, cfg in SWEEP.items():
        if aborted:
            break
        for val in cfg["values"]:
            name = _name(param, val)
            out = OUTDIR / f"{name}.json"
            summary = _load_summary(out)

            if out.exists() and not args.no_resume and not args.dry_run:
                if _is_valid(summary, out):
                    print(f"[skip] {name} already valid "
                          f"(full_pass_rate={summary.get('full_pass_rate')}, "
                          f"broken={_broken_task_count(out)})", flush=True)
                    rows.append({"param": param, "value": val, "name": name,
                                 "summary": summary, "rc": 0, "reused": True})
                    continue
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
                cfg["flag"], str(val),
                "--output", str(out),
            ]
            rc = _run(cmd, env, args.dry_run)
            summary = {} if args.dry_run else _load_summary(out)
            rows.append({"param": param, "value": val, "name": name,
                         "summary": summary, "rc": rc, "reused": False})

            if not args.dry_run and not _is_valid(summary, out):
                print(f"[abort] {name} came back invalid "
                      f"(full_pass_rate={summary.get('full_pass_rate')}, "
                      f"broken={_broken_task_count(out)}) — stopping sweep.",
                      flush=True)
                if not args.keep_going:
                    aborted = True
                    break

    summary_path = OUTDIR / "sweep_summary.json"
    json.dump(
        {"anchor": DEFAULT_ANCHOR, "limit": args.limit,
         "model": args.model, "provider": args.provider,
         "min_full_pass_rate": MIN_FULL_PASS_RATE,
         "aborted": aborted, "rows": rows},
        open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False,
    )
    print(f"\n[done] wrote {summary_path} with {len(rows)} rows"
          f"{' (ABORTED EARLY)' if aborted else ''}", flush=True)
    if aborted:
        sys.exit(2)


if __name__ == "__main__":
    main()
