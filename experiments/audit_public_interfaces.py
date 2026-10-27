"""Separate public-module compatibility from semantic evidence; never decide discussion policy."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_frozen_replay import analyze
from experiments.audit_public_evidence import execute_public_checks
from experiments.public_semantics import property_output
from experiments.run_frozen_replay import fingerprint, write_json


def public_interfaces(problem):
    """Extract public top-level declarations, never hidden entry points or benchmark tests."""
    try:
        tree = ast.parse(problem)
    except SyntaxError:
        return []
    declarations = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        positional = function.args.posonlyargs + function.args.args
        required = len(positional) - len(function.args.defaults)
        required_keywords = [arg.arg for arg, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        ) if default is None]
        shapes = [{"positional_count": count, "keyword_names": required_keywords}
                  for count in sorted({required, len(positional)})]
        shapes.append({
            "positional_count": len(function.args.posonlyargs),
            "keyword_names": [arg.arg for arg in function.args.args] + required_keywords,
        })
        declarations.append({"name": function.name, "call_shapes": shapes})
    return declarations


INTERFACE_SUPPORT = """
def _audit_public_interface(name, shapes):
    import inspect
    candidate = globals().get(name)
    if not callable(candidate):
        return {"name": name, "status": "missing_or_noncallable"}
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return {"name": name, "status": "introspection_unsupported"}
    incompatible = []
    for shape in shapes:
        try:
            signature.bind(
                *[None] * shape["positional_count"],
                **{keyword: None for keyword in shape["keyword_names"]}
            )
        except TypeError:
            incompatible.append(shape)
    return {
        "name": name, "status": "signature_incompatible" if incompatible else "compatible",
        "incompatible_public_call_shapes": incompatible,
    }
"""


def evaluate_interface(output, declarations):
    """Load code but never invoke the declared functions to check their signatures."""
    if not declarations:
        return {"status": "unsupported_public_prompt", "symbols": []}
    checks = [{
        "expression": f"_audit_public_interface({item['name']!r}, {item['call_shapes']!r})",
        "expected": None,
    } for item in declarations]
    evaluation = execute_public_checks(property_output(output, INTERFACE_SUPPORT), checks)
    if evaluation["status"] != "executed":
        return {"status": "candidate_load_or_process_error", "evaluation": evaluation}
    rows = evaluation["checks"]
    if len(rows) != len(declarations) or any("error_type" in row for row in rows):
        return {"status": "interface_check_error", "evaluation": evaluation}
    symbols = [row["actual"] for row in rows]
    statuses = {row["status"] for row in symbols}
    if "missing_or_noncallable" in statuses:
        status = "public_symbol_missing"
    elif "signature_incompatible" in statuses:
        status = "public_signature_incompatible"
    elif "introspection_unsupported" in statuses:
        status = "interface_introspection_unsupported"
    else:
        status = "public_interface_compatible"
    return {"status": status, "symbols": symbols}


def evidence_layers(request_id, snapshots):
    layers = []
    for source, rows in snapshots.items():
        observations = [row for row in rows if row["request_id"] == request_id]
        if not observations:
            continue
        if source == "public_semantics":
            status = observations[0]["public_status"]
        elif source == "public_inverse":
            observation = observations[0]
            if observation["all_roundtrips_satisfied"]:
                status = "checks_satisfied"
            elif observation["property_violations"]:
                status = "property_violation"
            else:
                status = "check_execution_error"
        else:
            if all(row["all_probes_satisfied"] for row in observations):
                status = "checks_satisfied"
            elif any(row["property_violations"] for row in observations):
                status = "property_violation"
            else:
                status = "check_execution_error"
        layers.append({"source": source, "status": status})
    return layers


def verified_snapshots(directory, manifest_sha, requests_sha):
    """Refuse stale sidecar joins; absent coverage is not a positive semantic result."""
    snapshots = {}
    digests = {}
    for source in ("public_semantics", "public_inverse", "public_probe"):
        path = directory / f"{source}_audit.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if source == "public_semantics":
            source_manifest = report["manifest_sha256"]
            source_requests = report["request_content_sha256"]
        else:
            protocol_path = directory / f"{source}_manifest.json"
            protocol = json.loads(protocol_path.read_text())
            if fingerprint(protocol) != report["protocol_sha256"] or not report["complete"]:
                raise ValueError("侧车核验协议不完整或不一致")
            if source == "public_inverse":
                source_manifest, source_requests = protocol["manifest_sha256"], protocol["requests_sha256"]
            else:
                source_manifest = protocol["source_manifest_sha256"]
                source_requests = protocol["source_requests_sha256"]
            digests[protocol_path.name] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if source_manifest != manifest_sha or source_requests != requests_sha:
            raise ValueError("侧车核验与原始请求不一致")
        snapshots[source] = report["outputs"]
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshots, digests


def audit(directory, workers=4):
    analyze(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset", "math") != "humaneval":
        raise ValueError("公开Python接口审计仅适用于HumanEval")
    paths = sorted(directory.glob("request_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    requests_sha = fingerprint([
        [path.name, hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths
    ])
    snapshots, snapshot_digests = verified_snapshots(directory, manifest_sha, requests_sha)
    declarations = {task["task_id"]: public_interfaces(task["problem"]) for task in manifest["tasks"]}
    fingerprint_data = {
        "manifest_sha256": manifest_sha, "requests_sha256": requests_sha,
        "sidecar_digests": snapshot_digests,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executor_sha256": hashlib.sha256(
            (ROOT / "experiments/audit_public_evidence.py").read_bytes()
        ).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            (ROOT / "experiments/public_semantics.py").read_bytes()
        ).hexdigest(),
    }
    report_path = directory / "public_interface_audit.json"
    audit_hash = fingerprint(fingerprint_data)
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["audit_sha256"] == audit_hash and previous["complete"]:
            return previous
        raise ValueError("接口审计输入或实现变更，拒绝覆盖已完成结果")

    def evaluate(record):
        return {
            "request_id": record["request_id"], "task_id": record["task_id"],
            "policy": record["policy"], "repeat": record["repeat"],
            "interface": evaluate_interface(record["output"], declarations[record["task_id"]]),
            "semantic_layers": evidence_layers(record["request_id"], snapshots),
            "discussion_decision": "not_evaluated",
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(evaluate, records))
    labels = {row["request_id"]: bool(row["passed"]) for row in records}
    for row in outputs:
        row["recorded_benchmark_passed"] = labels[row["request_id"]]
    summary = {}
    for policy in manifest["policies"]:
        rows = [row for row in outputs if row["policy"] == policy]
        summary[policy] = {
            "code_outputs": len(rows),
            "interface_status_counts": dict(Counter(row["interface"]["status"] for row in rows)),
            "benchmark_failed_despite_compatible_interface": sum(
                not row["recorded_benchmark_passed"]
                and row["interface"]["status"] == "public_interface_compatible" for row in rows
            ),
            "benchmark_passed_with_interface_diagnostic": sum(
                row["recorded_benchmark_passed"]
                and row["interface"]["status"] in (
                    "public_symbol_missing", "public_signature_incompatible"
                ) for row in rows
            ),
        }
    report = {
        **fingerprint_data, "audit_sha256": audit_hash, "api_requests": 0, "complete": True,
        "coverage": {"total_tasks": len(declarations),
                     "tasks_with_public_declarations": sum(bool(items) for items in declarations.values()),
                     "total_outputs": len(outputs)},
        "public_declarations": declarations, "by_policy": summary, "outputs": outputs,
        "limitations": [
            "Checks public declarations and representative positional/keyword call shapes only; "
            "not all possible calls or the benchmark's exact module interface requirements.",
            "No declared function is called for signature checking; candidate module load can still execute code.",
            "A compatible interface is not evidence of functional correctness.",
            "Semantic coverage absence is not semantic success; finite checks do not certify correctness.",
            "Interface diagnoses do not establish a discussion-stop or repair policy.",
            "Existing seen tasks and correlated code outputs, not independent held-out validation.",
            "Timed local execution is not a security sandbox; no model/API calls.",
        ],
    }
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    for directory in args.directories:
        report = audit(directory, args.workers)
        print(json.dumps({"directory": str(directory), "coverage": report["coverage"],
                          "by_policy": report["by_policy"], "api_requests": 0}))


if __name__ == "__main__":
    main()
