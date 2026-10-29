"""Locally repair one exported oracle and rescore frozen outputs transparently."""

import argparse
import ast
import json
import os
from pathlib import Path
import time

from experiments.analyze_prompt_contrast_replication import replication_statistics
from experiments.rescore_feedback_plus import (
    _load_signed, _score, _signed, reference_acceptance, validate_manifest,
)
from experiments.run_feedback_experiment import immutable_json
from experiments.run_prompt_contrast_pilot import ROOT, write_json
from src.experiment_driver import file_sha256
from src.request_ledger import fingerprint


OLD = "assert _poly(*candidate(*inp), inp) <= 0.0001"
NEW = "assert _vib_local_root_check(inp[0], candidate(*inp))"
HELPER = '''
def _vib_local_root_check(coefficients, root):
    from fractions import Fraction
    import math
    from numbers import Real

    if isinstance(root, bool) or not isinstance(root, Real):
        return False
    try:
        root = float(root)
        if not math.isfinite(root):
            return False
        exact_coefficients = [Fraction(value) for value in coefficients]

        def evaluate(value):
            argument = Fraction.from_float(value)
            residual = Fraction(0)
            for coefficient in reversed(exact_coefficients):
                residual = residual * argument + coefficient
            return residual

        if abs(evaluate(root)) <= Fraction(1, 10000):
            return True
        lower = math.nextafter(root, -math.inf)
        upper = math.nextafter(root, math.inf)
        if not math.isfinite(lower) or not math.isfinite(upper):
            return False
        lower_residual, upper_residual = evaluate(lower), evaluate(upper)
        return (lower_residual <= 0 <= upper_residual
                or upper_residual <= 0 <= lower_residual)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
'''
PARENT = ROOT / "results/feedback_plus_rescore_20261029"


def repair_task32(task):
    """Use an exact residual or a certified adjacent-float root interval."""
    if task["task_id"] != "HumanEval/32" or task["test"].count(OLD) != 1:
        raise ValueError("Repair requires exactly the known task32 assertion")
    return {**task, "test": task["test"].replace(OLD, NEW, 1) + "\n" + HELPER}


def _parent():
    manifest = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    tasks = validate_manifest(manifest)
    return manifest, tasks


def validation_reference32(task):
    """Build a reference from publisher roots, not from any model output."""
    check = next(node for node in ast.parse(task["test"]).body
                 if isinstance(node, ast.FunctionDef) and node.name == "check")
    values = {target.id: ast.literal_eval(node.value) for node in check.body
              if isinstance(node, ast.Assign) for target in node.targets
              if isinstance(target, ast.Name) and target.id in ("inputs", "results")}
    if len(values["inputs"]) != len(values["results"]):
        raise ValueError("Task32 publisher inputs and roots do not align")
    roots = {}
    for arguments, root in zip(values["inputs"], values["results"]):
        if len(arguments) != 1:
            raise ValueError("Task32 expects one coefficient-list argument")
        roots[tuple(arguments[0])] = root
    body = f"\n    roots = {roots!r}\n    return roots[tuple(xs)]\n"
    return {**task, "canonical_solution": body}, len(values["inputs"])


def _expected(output):
    parent, tasks = _parent()
    original = Path(parent["dataset"]["path"]).read_bytes()
    repaired_task = repair_task32(tasks["HumanEval/32"])
    before_field = json.dumps(tasks["HumanEval/32"]["test"], ensure_ascii=False).encode("utf-8")
    after_field = json.dumps(repaired_task["test"], ensure_ascii=False).encode("utf-8")
    if original.count(before_field) != 1:
        raise ValueError("Byte-level task32 test-field patch is not unique")
    repaired = original.replace(before_field, after_field, 1)
    patched_tasks = {row["task_id"]: row for row in
                     (json.loads(line) for line in repaired.decode("utf-8").splitlines() if line.strip())}
    expected_tasks = {**tasks, "HumanEval/32": repair_task32(tasks["HumanEval/32"])}
    if patched_tasks != expected_tasks:
        raise ValueError("Patch changed data outside the authorized assertion")
    path = output / "local_repaired_test.jsonl"
    if path.exists():
        if path.read_bytes() != repaired:
            raise ValueError("Local repaired dataset changed")
    else:
        with path.open("xb") as stream:
            stream.write(repaired)
            stream.flush()
            os.fsync(stream.fileno())
    patch = {
        "version": "humaneval32_exact_residual_adjacent_float_local_repair_v2",
        "task_id": "HumanEval/32", "before": OLD, "after": NEW,
        "helper_source": HELPER,
        "reason": "scalar_root_not_iterable_absolute_residual_and_finite_precision_root_interval",
        "tolerance": 0.0001, "tolerance_changed": False, "inputs_changed": False,
        "numerical_acceptance_extended": True,
        "precision_guard": "exact_fraction_sign_change_between_adjacent_float_neighbors",
        "precision_guard_user_confirmed": True,
        "reference_changed": False, "changed_tasks": 1,
        "original_file_sha256": file_sha256(parent["dataset"]["path"]),
        "local_file_sha256": file_sha256(path),
        "task32_original_test_sha256": fingerprint(tasks["HumanEval/32"]["test"]),
        "task32_local_test_sha256": fingerprint(patched_tasks["HumanEval/32"]["test"]),
        "task32_in_candidate_cohort": "HumanEval/32" in parent["task_ids"],
        "scope": "user_authorized_local_oracle_revision_not_official_evalplus_score",
    }
    validation_task, count = validation_reference32(patched_tasks["HumanEval/32"])
    patch["reference_validation_override"] = {
        "task_id": "HumanEval/32", "mode": "publisher_expected_root_table",
        "user_confirmed": True, "input_root_pairs": count,
        "original_canonical_preserved": True, "remaining_original_canonicals": 163,
        "original_code_sha256": fingerprint(tasks["HumanEval/32"]["prompt"]
                                             + tasks["HumanEval/32"]["canonical_solution"]),
        "validation_code_sha256": fingerprint(validation_task["prompt"]
                                               + validation_task["canonical_solution"]),
        "reason": "original_newton_can_fail_to_converge_under_valid_expanded_inputs",
    }
    manifest = {
        **parent, "version": "local_repaired_feedback_plus_rescoring_v1",
        "dataset": {**parent["dataset"], "path": str(path.resolve()),
                    "local_patched_sha256": file_sha256(path),
                    "scope": patch["scope"]},
        "parent_manifest": {"path": str((PARENT / "manifest.json").resolve()),
                            "sha256": file_sha256(PARENT / "manifest.json")},
        "patch": patch, "interpretation":
            "same_frozen_outputs_local_repaired_exported_tests_not_official_or_new_confirmation",
        "sources_sha256": {**parent["sources_sha256"],
                           "experiments/rescore_repaired_plus.py": file_sha256(__file__)},
    }
    return manifest, patched_tasks


def prepare_repaired(output_dir):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    manifest, _ = _expected(output)
    immutable_json(output / "patch.json", manifest["patch"])
    immutable_json(output / "manifest.json", manifest)
    return manifest


def _validate(output, manifest):
    expected, tasks = _expected(output)
    if manifest != expected or json.loads((output / "patch.json").read_text()) != manifest["patch"]:
        raise ValueError("Frozen local repair protocol changed")
    return tasks


def rescore_repaired(output_dir):
    """Reuse strict reference acceptance; never modify old scoring modules."""
    output = Path(output_dir)
    lock = output / ".rescore.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        tasks = _validate(output, manifest)
        diagnostic_path = output / "original_task32_reference.json"
        diagnostic_identity = {"manifest_sha256": fingerprint(manifest), "task_id": "HumanEval/32"}
        if diagnostic_path.exists():
            diagnostic = _load_signed(diagnostic_path, diagnostic_identity)
        else:
            original_code = tasks["HumanEval/32"]["prompt"] + tasks["HumanEval/32"]["canonical_solution"]
            diagnostic = _signed({**diagnostic_identity,
                                  **_score(original_code, tasks["HumanEval/32"], manifest),
                                  "scope": "original_newton_diagnostic_not_validation_override"})
            write_json(diagnostic_path, diagnostic)
        validation_task, _ = validation_reference32(tasks["HumanEval/32"])
        validation_tasks = {**tasks, "HumanEval/32": validation_task}
        acceptance = reference_acceptance(output, manifest, validation_tasks)
        identity = {"manifest_sha256": fingerprint(manifest),
                    "reference_acceptance_sha256": file_sha256(output / "reference_acceptance.json")}
        analysis_path = output / "analysis.json"
        if analysis_path.exists():
            result = _load_signed(analysis_path, identity)
            for row in result["evaluations"]:
                record = _load_signed(output / "candidate_checks" / f"{fingerprint(row['job_id'])}.json",
                                      {**identity, "job_id": row["job_id"]})
                if record != row:
                    raise ValueError("Cached local score record differs")
            return result
        folder = output / "candidate_checks"
        folder.mkdir(exist_ok=True)
        rows = []
        for job in manifest["jobs"]:
            task = tasks[job["task_id"]]
            record_identity = {**identity, **job, "test_sha256": fingerprint(task["test"])}
            path = folder / f"{fingerprint(job['job_id'])}.json"
            if path.exists():
                row = _load_signed(path, record_identity)
            else:
                raw = json.loads(Path(job["raw_response"]["path"]).read_text())
                if file_sha256(job["raw_response"]["path"]) != job["raw_response"]["sha256"]:
                    raise ValueError("Original raw output changed")
                started = time.monotonic()
                scored = ({"passed": False, "status": "original_truncation", "independent_checks": 0}
                          if job["original_status"] == "truncated" else _score(raw["content"], task, manifest))
                row = _signed({**record_identity, **scored,
                               "local_evaluation_seconds": time.monotonic() - started})
                write_json(path, row)
            rows.append(row)
            print(json.dumps({"phase": "local_repaired_candidates", "completed": len(rows), "total": 128}), flush=True)
        statistics, primary = replication_statistics(rows, manifest["statistics_settings"])
        original_statistics, _ = replication_statistics(
            [{**row, "passed": row["original_passed"]} for row in rows], manifest["statistics_settings"])
        transitions = {}
        for policy in statistics["groups"]:
            group = [row for row in rows if row["prompt_variant"] == policy]
            transitions[policy] = {
                "pass_to_fail": sum(row["original_passed"] and not row["passed"] for row in group),
                "fail_to_pass": sum(not row["original_passed"] and row["passed"] for row in group),
                "both_pass": sum(row["original_passed"] and row["passed"] for row in group),
                "both_fail": sum(not row["original_passed"] and not row["passed"] for row in group),
            }
        _validate(output, manifest)
        result = _signed({
            **identity, "statistics": statistics, "primary_comparison": primary,
            "original_statistics": original_statistics, "transitions": transitions,
            "evaluations": rows, "patch": manifest["patch"], "cost": manifest["cost"],
            "new_model_requests": 0, "original_artifacts_unchanged": True,
            "reference_acceptance": {"all_passed": acceptance["all_passed"], "tasks": 164},
            "reference_validation_scope": "163_original_canonicals_plus_task32_publisher_root_table",
            "original_task32_reference_diagnostic": diagnostic,
            "interpretation": manifest["interpretation"],
            "independent_candidate_executions": sum(row["independent_checks"] for row in rows),
            "local_reference_evaluation_seconds": sum(row["local_evaluation_seconds"] for row in acceptance["checks"]),
            "local_candidate_evaluation_seconds": sum(row["local_evaluation_seconds"] for row in rows),
        })
        write_json(analysis_path, result)
        return result
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_repaired(args.output_dir)
    result = rescore_repaired(args.output_dir)
    print(json.dumps({key: result[key] for key in ("statistics", "primary_comparison", "transitions")}, indent=2))
