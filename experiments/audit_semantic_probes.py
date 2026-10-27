"""Frozen, zero-API polynomial input-family checks on archived code, not new-task evaluation."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_frozen_replay import analyze
from experiments.audit_public_evidence import execute_public_checks
from experiments.public_semantics import property_output, semantic_contract
from experiments.run_frozen_replay import fingerprint, write_json

FAMILIES = {
    "linear_integer_root": "development",
    "cubic_distinct_integer_roots": "development",
    "quintic_distinct_integer_roots": "validation",
    "cubic_stationary_origin": "validation",
}


def coefficients_from_roots(roots):
    """Expand a monic polynomial in ascending coefficient order using exact integers."""
    coefficients = [1]
    for root in roots:
        expanded = [0] * (len(coefficients) + 1)
        for index, coefficient in enumerate(coefficients):
            expanded[index] -= root * coefficient
            expanded[index + 1] += coefficient
        coefficients = expanded
    return coefficients


def exact_polynomial(coefficients, point):
    value = 0
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def generate_probes(seed=20261026, per_family=32):
    """Freeze structurally disjoint families without inspecting candidate outputs or labels."""
    if not 1 <= per_family <= 32:
        raise ValueError("每族探针数必须在1到32之间")
    probes = []
    for family, split in FAMILIES.items():
        rng = random.Random(f"{seed}:{family}")
        if family == "linear_integer_root":
            root_sets = [[root] for root in rng.sample(list(range(-25, 26)), per_family)]
        elif family == "cubic_stationary_origin":
            root_sets = [[root] for root in rng.sample(
                [value for value in range(-32, 33) if value], per_family
            )]
        else:
            degree = 3 if family.startswith("cubic") else 5
            root_sets = []
            while len(root_sets) < per_family:
                roots = sorted(rng.sample(list(range(-15, 16)), degree))
                if roots not in root_sets:
                    root_sets.append(roots)
        for index, roots in enumerate(root_sets):
            coefficients = (
                [-roots[0]**3, 0, 0, 1] if family == "cubic_stationary_origin"
                else coefficients_from_roots(roots)
            )
            witness = roots[0]
            if len(coefficients) % 2 or coefficients[-1] == 0:
                raise ValueError("探针不满足公开偶数系数数目契约")
            if exact_polynomial(coefficients, witness) != 0:
                raise ValueError("探针见证无效")
            probes.append({
                "probe_id": f"{family}/{index}", "family": family, "split": split,
                "coefficients": coefficients, "witness_root": witness,
            })
    return probes


def public_function(contract):
    expression = ast.parse(contract["checks"][0]["expression"], mode="eval").body
    return expression.args[0].func.id


def evaluate_family(output, function, contract, probes):
    """Accept any root by residual, never compare with the construction witness."""
    checks = [{
        "expression": (
            f"_audit_public_root({function}({probe['coefficients']!r}), "
            f"{probe['coefficients']!r})"
        ),
        "expected": True,
    } for probe in probes]
    started = time.perf_counter()
    evaluation = execute_public_checks(property_output(output, contract["support_code"]), checks)
    rows = evaluation.get("checks", [])
    if evaluation["status"] == "executed" and len(rows) != len(probes):
        raise ValueError("探针执行覆盖不完整")
    counts = Counter(
        "execution_error" if "error_type" in row else
        "satisfied" if row["satisfied"] else "property_violation"
        for row in rows
    )
    return {
        "evaluation": evaluation, "probe_ids": [probe["probe_id"] for probe in probes],
        "satisfied": counts["satisfied"], "property_violations": counts["property_violation"],
        "check_execution_errors": counts["execution_error"],
        "all_probes_satisfied": (
            evaluation["status"] == "executed" and counts["satisfied"] == len(probes)
        ),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def summarize_family(rows):
    return {
        "code_outputs": len(rows),
        "code_outputs_satisfying_all_probes": sum(row["all_probes_satisfied"] for row in rows),
        "satisfied_probes": sum(row["satisfied"] for row in rows),
        "property_violations": sum(row["property_violations"] for row in rows),
        "check_execution_errors": sum(row["check_execution_errors"] for row in rows),
        "family_process_errors": sum(row["evaluation"]["status"] != "executed" for row in rows),
    }


def run(directory, dry_run=False):
    analyze(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(directory.glob("request_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    tasks = {}
    if manifest.get("dataset", "math") == "humaneval":
        for task in manifest["tasks"]:
            contract = semantic_contract(task["problem"])
            if contract["mode"] == "polynomial_root_residual":
                tasks[task["task_id"]] = contract
    probes = generate_probes()
    protocol = {
        "version": 1, "seed": 20261026, "probes": probes,
        "families": FAMILIES,
        "selection": "all_tasks_with_public_polynomial_contract_no_pass_label_filter",
        "task_contracts": tasks,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_requests_sha256": fingerprint([
            [path.name, hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths
        ]),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((ROOT / "experiments/public_semantics.py").read_bytes()).hexdigest(),
        "executor_sha256": hashlib.sha256(
            (ROOT / "experiments/audit_public_evidence.py").read_bytes()
        ).hexdigest(),
    }
    protocol_path = directory / "public_probe_manifest.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("探针协议变更，禁止覆盖已有冻结协议")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    jobs = [(row, family) for row in records if row["task_id"] in tasks for family in FAMILIES]
    protocol_hash = fingerprint(protocol)
    if dry_run:
        print(json.dumps({"directory": str(directory), "protocol_sha256": protocol_hash,
                          "tasks": len(tasks), "family_jobs": len(jobs), "api_requests": 0}))
        return
    result_path = directory / "public_probe_audit.json"
    completed = {}
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing["protocol_sha256"] != protocol_hash:
            raise ValueError("探针结果协议不一致")
        completed = {row["job_id"]: row for row in existing["outputs"]}
        expected_ids = {
            fingerprint([protocol_hash, record["request_id"], family]) for record, family in jobs
        }
        if len(completed) != len(existing["outputs"]) or not completed.keys() <= expected_ids:
            raise ValueError("探针缓存包含重复或未知任务")
    report = {
        "protocol_sha256": protocol_hash, "api_requests": 0,
        "total_tasks_in_source": len(manifest["tasks"]), "applicable_tasks": len(tasks),
        "probes_per_family": 32, "family_jobs": len(jobs), "complete": False,
        "outputs": list(completed.values()),
        "limitations": [
            "Development and validation separate constructed input families, not independent tasks.",
            "Adapters and family designs are discovery-informed; not blinded or held-out method evaluation.",
            "Any real root is accepted; construction witnesses are not expected return values.",
            "Residual tolerance and finite probes do not prove correctness for all valid inputs.",
            "No policy training, threshold tuning, new model calls or candidate generation.",
            "Failures include exceptions and invalid return values, not only wrong numeric answers.",
            "Timed local execution is not a security sandbox; original benchmark labels remain unchanged.",
        ],
    }
    for record, family in jobs:
        job_id = fingerprint([protocol_hash, record["request_id"], family])
        if job_id in completed:
            continue
        selected = [probe for probe in probes if probe["family"] == family]
        contract = tasks[record["task_id"]]
        row = {
            "job_id": job_id, "request_id": record["request_id"],
            "task_id": record["task_id"], "policy": record["policy"],
            "repeat": record["repeat"], "family": family, "split": FAMILIES[family],
            **evaluate_family(record["output"], public_function(contract), contract, selected),
        }
        completed[job_id] = row
        report["outputs"] = list(completed.values())
        write_json(result_path, report)
    report["complete"] = True
    report["by_family_policy"] = {}
    for family in FAMILIES:
        by_policy = {}
        for policy in manifest["policies"]:
            rows = [row for row in completed.values()
                    if row["family"] == family and row["policy"] == policy]
            by_policy[policy] = summarize_family(rows)
        report["by_family_policy"][family] = by_policy
    write_json(result_path, report)
    print(json.dumps({"directory": str(directory), "complete": True,
                      "tasks": len(tasks), "family_jobs": len(jobs), "api_requests": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        run(directory, args.dry_run)


if __name__ == "__main__":
    main()
