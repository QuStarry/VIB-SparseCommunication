"""Re-score archived MATH whole answers into separate, provenance-checked artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_multi_benchmark import analyze_dataset
from experiments.math_answer_matching import VERSION, score

PASS_KEYS = (
    "full_pass_rate", "vib_pass_rate", "pass_delta_pp", "full_only_count",
    "vib_only_count", "both_pass_count", "both_fail_count", "mcnemar_pvalue",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invariant_digest(data):
    """Hash all fields except MATH labels, their derived summary and correction metadata."""
    digest = hashlib.sha256()
    for dataset, rows in data["trace_results"].items():
        for trace in rows:
            for key, value in trace.items():
                if dataset == "math" and key in ("full", "vib"):
                    value = {field: item for field, item in value.items() if field != "passed"}
                digest.update(json.dumps([dataset, trace["task_id"], key, value],
                                         sort_keys=True, ensure_ascii=False).encode())
    for dataset, rows in data["detailed_results"].items():
        for row in rows:
            value = {key: value for key, value in row.items()
                     if dataset != "math" or key not in ("full_passed", "vib_passed")}
            digest.update(json.dumps([dataset, value], sort_keys=True).encode())
    for dataset, summary in data["summary"].items():
        value = {key: value for key, value in summary.items()
                 if dataset != "math" or key not in PASS_KEYS}
        digest.update(json.dumps([dataset, value], sort_keys=True).encode())
    for key, value in data.items():
        if key not in ("trace_results", "detailed_results", "summary", "math_eval_correction"):
            digest.update(json.dumps([key, value], sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def recompute(data):
    before = invariant_digest(data)
    traces = data["trace_results"]["math"]
    detailed = {row["task_id"]: row for row in data["detailed_results"]["math"]}
    if len(detailed) != len(traces) or len({row["task_id"] for row in traces}) != len(traces):
        raise ValueError("MATH task IDs are duplicated or inconsistent")
    audit = []
    formats = Counter()
    flips = Counter()
    expression_ids = set()
    for trace in traces:
        row = detailed[trace["task_id"]]
        if trace["full"]["reference_answer"] != trace["vib"]["reference_answer"]:
            raise ValueError("Paired reference answers differ")
        reference = trace["full"]["reference_answer"]
        # Define coverage from the reference alone, never from candidate success or syntax.
        reference_format = score(reference, reference).get("reference_kind", "unsupported")
        if reference_format == "expression":
            expression_ids.add(trace["task_id"])
        for arm in ("full", "vib"):
            outcome = score(trace[arm]["output"], trace[arm]["reference_answer"])
            old = bool(trace[arm]["passed"])
            if old != bool(row[f"{arm}_passed"]):
                raise ValueError("Archived trace/detailed labels disagree")
            passed = outcome["passed"]
            formats[(arm, reference_format)] += 1
            if old != passed:
                flips[(arm, "false_to_true" if passed else "true_to_false")] += 1
            audit.append({"task_id": trace["task_id"], "arm": arm,
                          "recorded": old, "recomputed": passed,
                          "reference_format": reference_format, **outcome})
            trace[arm]["passed"] = passed
            row[f"{arm}_passed"] = passed
    summary = data["summary"]["math"]
    for arm in ("full", "vib"):
        summary[f"{arm}_pass_rate"] = round(
            100 * sum(row[f"{arm}_passed"] for row in detailed.values()) / len(detailed), 2
        )
    # The analyzer reads percentages from summary, so refresh them before any derived CI.
    analysis = analyze_dataset("math", list(detailed.values()), summary)
    for key in PASS_KEYS:
        if key not in ("full_pass_rate", "vib_pass_rate"):
            summary[key] = analysis[key]
    if invariant_digest(data) != before:
        raise AssertionError("Non-label data changed during re-scoring")
    expression_rows = [row for task_id, row in detailed.items() if task_id in expression_ids]
    sensitivity = None
    if expression_rows:
        expression_summary = {
            f"{arm}_pass_rate": 100 * sum(row[f"{arm}_passed"] for row in expression_rows)
            / len(expression_rows) for arm in ("full", "vib")
        }
        sensitivity = analyze_dataset("math_expression_reference", expression_rows, expression_summary)
    return {
        "total_tasks": len(traces), "label_changes": [
            {"arm": arm, "direction": direction, "count": count}
            for (arm, direction), count in sorted(flips.items())
        ],
        "reference_formats": [
            {"arm": arm, "kind": kind, "count": count} for (arm, kind), count in sorted(formats.items())
        ],
        "analysis": analysis, "reference_expression_sensitivity": sensitivity,
        "rows": audit, "invariant_sha256": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Original input cannot be overwritten")
    audit_path = args.output.with_suffix(".audit.json")
    provenance = {
        "source": str(args.input), "source_sha256": sha(args.input),
        "script_sha256": sha(Path(__file__)),
        "matcher_sha256": sha(ROOT / "experiments/math_answer_matching.py"),
        "method": VERSION, "sympy_version": sympy.__version__,
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("math_eval_correction", {}).get("provenance") != provenance:
            raise ValueError("Existing correction provenance differs; refuse overwrite")
        if not audit_path.exists():
            raise ValueError("Correction audit missing")
        print(f"Reused {args.output}")
        return
    data = json.loads(args.input.read_text())
    report = recompute(data)
    data["math_eval_correction"] = {
        "provenance": provenance, "invariant_sha256": report["invariant_sha256"],
        "total_tasks": report["total_tasks"], "label_changes": report["label_changes"],
        "limitations": [
            "No last-number fallback. Bare comma sequences and tuples preserve order; "
            "explicit sets and unions compare without order.",
            "Unrecognized expressions use normalized whole-text equality; unsupported "
            "mismatches can remain false negatives, not mathematical correctness proofs.",
            "Reference-box extraction is balanced; archived full solutions may have "
            "different annotation quality from boxed final-answer references.",
            "Generation protocols, model snapshots and billed-cost omissions remain unchanged.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({"provenance": provenance, **report},
                                     indent=2, ensure_ascii=False))
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps({key: data["summary"]["math"][key] for key in PASS_KEYS}))
    print(f"Saved {args.output} and {audit_path}")


if __name__ == "__main__":
    main()
