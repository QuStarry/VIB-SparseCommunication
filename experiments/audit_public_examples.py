"""Evaluate literal public examples on all replay tasks without task-ID rules or API calls."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_frozen_replay import analyze
from experiments.audit_public_evidence import execute_public_checks
from experiments.run_frozen_replay import fingerprint, write_json


def extract_examples(problem):
    """Accept literal call/result or call==literal examples; unsupported syntax abstains."""
    functions = set(re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", problem, re.MULTILINE))
    lines = problem.splitlines()
    checks = []
    rejected = []
    for index, line in enumerate(lines):
        if ">>>" not in line:
            continue
        source = line.partition(">>>")[2].strip()
        try:
            expression = ast.parse(source, mode="eval").body
            if isinstance(expression, ast.Compare):
                if len(expression.ops) != 1 or not isinstance(expression.ops[0], ast.Eq):
                    raise ValueError("unsupported comparison")
                expected = ast.literal_eval(expression.comparators[0])
                expression = expression.left
            else:
                if index + 1 >= len(lines):
                    raise ValueError("missing literal result")
                expected = ast.literal_eval(lines[index + 1].strip())
            if not isinstance(expression, ast.Call):
                raise ValueError("not a function call")
            for node in ast.walk(expression):
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name):
                        raise ValueError("attribute calls unsupported")
                    if node.func.id not in functions | {"round"} or node.func.id.startswith("__"):
                        raise ValueError("not a public function")
                    for argument in (*node.args, *(item.value for item in node.keywords)):
                        if not isinstance(argument, ast.Call):
                            ast.literal_eval(argument)
                    if any(item.arg is None for item in node.keywords):
                        raise ValueError("expanded keywords unsupported")
            check = {
                "expression": ast.unparse(expression), "expected": expected,
                "source_line": index + 1, "source_text": source,
            }
            encoded_expected = json.dumps(expected, allow_nan=False)
            if json.loads(encoded_expected) != expected:
                raise ValueError("literal type cannot round-trip through execution JSON")
            checks.append(check)
        except (SyntaxError, ValueError, TypeError):
            rejected.append({"source_line": index + 1, "source_text": source})
    by_expression = {}
    conflicts = []
    for check in checks:
        previous = by_expression.get(check["expression"])
        if previous and previous["expected"] != check["expected"]:
            conflicts.append(check["expression"])
        by_expression[check["expression"]] = check
    return {
        "checks": [] if conflicts else list(by_expression.values()),
        "rejected_examples": rejected,
        "conflicting_expressions": sorted(set(conflicts)),
    }


def classify_evaluation(evaluation):
    """Keep literal contradictions, execution errors and unsupported cases distinct."""
    if evaluation["status"] != "executed":
        return "execution_error"
    checks = evaluation["checks"]
    if any("error_type" not in check and not check["satisfied"] for check in checks):
        return "literal_contradiction"
    if any("error_type" in check for check in checks):
        return "execution_error"
    return "examples_satisfied"


def diagnostic_metrics(rows):
    """Benchmark labels are joined only after public evaluation for descriptive diagnostics."""
    counts = Counter((row["public_status"], row["benchmark_passed"]) for row in rows)
    passed = sum(row["benchmark_passed"] for row in rows)
    failed = len(rows) - passed
    false_alarms = counts["literal_contradiction", True]
    detected_failures = counts["literal_contradiction", False]
    contradicted = false_alarms + detected_failures
    return {
        "covered_outputs": len(rows),
        "benchmark_passed_outputs": passed,
        "benchmark_failed_outputs": failed,
        "literal_contradiction_on_passed": false_alarms,
        "literal_contradiction_on_failed": detected_failures,
        "false_alarm_rate_on_covered_passed": false_alarms / passed if passed else None,
        "failure_detection_rate_on_covered_failed": detected_failures / failed if failed else None,
        "benchmark_failure_fraction_among_contradictions": (
            detected_failures / contradicted if contradicted else None
        ),
        "benchmark_failures_despite_satisfied_examples": counts["examples_satisfied", False],
        "execution_error_outputs": sum(
            count for (status, _), count in counts.items() if status == "execution_error"
        ),
    }


def evaluate_job(job):
    task_id, record, checks, support_code = job
    if support_code:
        from experiments.public_semantics import property_output
        candidate = property_output(record["output"], support_code)
    else:
        candidate = record["output"]
    started = time.perf_counter()
    evaluation = execute_public_checks(candidate, checks)
    return {
        "task_id": task_id, "request_id": record["request_id"],
        "policy": record["policy"], "repeat": record["repeat"],
        "output_sha256": hashlib.sha256(record["output"].encode()).hexdigest(),
        "public_status": classify_evaluation(evaluation),
        "public_evaluation": evaluation,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def compare_snapshots(literal, semantic):
    """Compare identical request IDs, separating changed verification from abstention."""
    for key in ("manifest_sha256", "request_content_sha256"):
        if literal[key] != semantic[key]:
            raise ValueError("公开核验比较必须使用相同原始输入")
    baseline = {row["request_id"]: row for row in literal["outputs"]}
    revised = {row["request_id"]: row for row in semantic["outputs"]}
    common = baseline.keys() & revised.keys()
    if any(baseline[key]["benchmark_passed"] != revised[key]["benchmark_passed"] for key in common):
        raise ValueError("原始 benchmark 标签不一致，禁止比较")
    policies = sorted({revised[key]["policy"] for key in common})

    def metric_rows(snapshot, ids):
        return [{
            "public_status": snapshot[key]["public_status"].replace(
                "public_check_violation", "literal_contradiction"
            ),
            "benchmark_passed": snapshot[key]["benchmark_passed"],
        } for key in ids]

    return {
        "common_outputs": len(common),
        "abstained_previously_covered_outputs": len(baseline.keys() - revised.keys()),
        "newly_covered_outputs": len(revised.keys() - baseline.keys()),
        "matched_by_policy": {
            policy: {
                "literal": diagnostic_metrics(metric_rows(
                    baseline, [key for key in common if revised[key]["policy"] == policy]
                )),
                "semantic": diagnostic_metrics(metric_rows(
                    revised, [key for key in common if revised[key]["policy"] == policy]
                )),
            } for policy in policies
        },
        "changed_outcomes_on_common_outputs": [
            {
                "request_id": key, "task_id": revised[key]["task_id"],
                "policy": revised[key]["policy"], "repeat": revised[key]["repeat"],
                "literal_status": baseline[key]["public_status"],
                "semantic_status": revised[key]["public_status"],
                "benchmark_passed": revised[key]["benchmark_passed"],
            } for key in sorted(common)
            if baseline[key]["public_status"] != revised[key]["public_status"].replace(
                "public_check_violation", "literal_contradiction"
            )
        ],
    }


def audit_examples(directory, workers=4, semantic=False):
    analyze(directory)  # Verify complete contract first; do not use labels for selection.
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = manifest.get("dataset", "math")
    paths = sorted(directory.glob("request_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if semantic:
        from experiments.public_semantics import semantic_contract
        extractor = semantic_contract
    else:
        extractor = extract_examples
    extracted = {
        task["task_id"]: extractor(task["problem"])
        if dataset == "humaneval" else
        {"checks": [], "rejected_examples": [], "conflicting_expressions": []}
        for task in manifest["tasks"]
    }
    jobs = [
        (row["task_id"],
         {key: row[key] for key in ("request_id", "policy", "repeat", "output")},
         extracted[row["task_id"]]["checks"],
         extracted[row["task_id"]].get("support_code", ""))
        for row in records if extracted[row["task_id"]]["checks"]
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(evaluate_job, jobs))
    wall_ms = (time.perf_counter() - started) * 1000
    labels = {row["request_id"]: bool(row["passed"]) for row in records}
    for output in outputs:
        output["benchmark_passed"] = labels[output["request_id"]]
    times = [row["elapsed_ms"] for row in outputs]
    report = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "request_content_sha256": fingerprint([
            [path.name, hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths
        ]),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executor_script_sha256": hashlib.sha256(
            (ROOT / "experiments/audit_public_evidence.py").read_bytes()
        ).hexdigest(),
        "verification_mode": "semantic_prototype" if semantic else "literal",
        "api_requests": 0,
        "coverage": {
            "dataset": dataset, "total_tasks": len(extracted),
            "tasks_with_checks": sum(bool(item["checks"]) for item in extracted.values()),
            "total_outputs": len(records), "evaluated_outputs": len(outputs),
            "accepted_checks": sum(len(item["checks"]) for item in extracted.values()),
        },
        "performance": {
            "workers": workers, "wall_ms": wall_ms,
            "execution_elapsed_ms_including_process_startup": {
                str(percentile): float(np.percentile(times, percentile))
                for percentile in (50, 95, 99)
            } if times else {},
            "extractor_timing_included": False,
        },
        "by_policy": {
            policy: diagnostic_metrics([row for row in outputs if row["policy"] == policy])
            for policy in manifest["policies"]
        },
        "tasks": [{"task_id": task_id, **item} for task_id, item in extracted.items()],
        "outputs": outputs,
        "limitations": [
            "Literal >>> examples only; prose Input/Output, mathematical derivations "
            "and unknown syntax abstain. No task-ID exceptions or manual certificates.",
            "A public example may contradict the prose or benchmark specification. "
            "False alarms are disagreement with recorded benchmark labels, not proof "
            "that a general-purpose verifier is wrong.",
            "Repeated outputs are descriptive diagnostics, not independent tasks. "
            "No datasets, policies or protocols are pooled to infer predictive validity.",
            "Satisfying examples does not certify correctness; execution errors are "
            "not counted as literal counterexamples.",
            "Labels are joined after evaluation; hidden tests and reference answers "
            "are never used to extract examples or choose public outcomes.",
            "Timings are one local run including child-process startup, not a production "
            "latency benchmark or an optimized concurrency recommendation.",
            "Executes archived code in timed local processes, not a security sandbox.",
        ],
    }
    if semantic:
        report["semantic_adapter_sha256"] = hashlib.sha256(
            (ROOT / "experiments/public_semantics.py").read_bytes()
        ).hexdigest()
        report["coverage"]["task_modes"] = dict(Counter(
            item.get("mode", "unsupported") for item in extracted.values()
        ))
        for policy, metrics in report["by_policy"].items():
            report["by_policy"][policy] = {
                key.replace("literal_contradiction", "public_check_violation"): value
                for key, value in metrics.items()
            }
        for row in outputs:
            if row["public_status"] == "literal_contradiction":
                row["public_status"] = "public_check_violation"
        report["limitations"] += [
            "Semantic adapters were designed after observing failures; applicability "
            "depends on public syntax and wording, not task IDs, but is not held-out validation.",
            "Only explicitly provided coefficient-order polynomials get a residual test; "
            "recognized popcount specification conflicts cause abstention, not correction.",
            "No new inputs are generated; coverage loss through abstention must be "
            "reported alongside false-alarm changes, not hidden by denominator changes.",
        ]
        baseline_path = directory / "public_examples_audit.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            report["comparison_with_literal_snapshot"] = compare_snapshots(baseline, report)
            report["literal_snapshot_sha256"] = hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        report = audit_examples(directory, args.workers, args.semantic)
        filename = "public_semantics_audit.json" if args.semantic else "public_examples_audit.json"
        write_json(directory / filename, report)
        print(json.dumps({"directory": str(directory), "coverage": report["coverage"],
                          "metrics": report["by_policy"], "performance": report["performance"]}))


if __name__ == "__main__":
    main()
