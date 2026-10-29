"""Expand frozen feedback coverage without model calls or answer repair."""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import time

from experiments.analyze_prompt_contrast_replication import replication_statistics
from experiments.run_feedback_experiment import immutable_json, proof, run_feedback
from experiments.run_prompt_contrast_pilot import ROOT, response_path, write_json
from src.experiment_driver import file_sha256
from src.isolated_plus_evaluation import plus_protocol, score_plus
from src.request_ledger import fingerprint


REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
SOURCE_SHA256 = "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492"
PUBLISHER = "evalplus/humanevalplus"
ORIGINAL = ROOT / "results/feedback_experiment_20261028"
SOURCE_FILES = (
    "experiments/rescore_feedback_plus.py", "src/isolated_plus_evaluation.py",
    "src/isolated_code_evaluation.py", "experiments/run_feedback_experiment.py",
    "experiments/analyze_prompt_contrast_replication.py",
    "experiments/analyze_prompt_contrast_pilot.py",
)


def _deny_client(path):
    raise AssertionError("Expanded rescoring must never initialize a model client")


def load_dataset(path):
    """Verify the publisher LFS checksum before parsing or executing any tests."""
    if file_sha256(path) != SOURCE_SHA256:
        raise ValueError("Publisher dataset SHA-256 differs")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {row["task_id"]: row for row in rows}
    expected = {f"HumanEval/{index}" for index in range(164)}
    if len(rows) != 164 or set(tasks) != expected:
        raise ValueError("Require all 164 unique publisher task IDs")
    for row in rows:
        for key in ("prompt", "canonical_solution", "entry_point", "test"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Incomplete publisher exported task")
        if not row["entry_point"].isidentifier():
            raise ValueError("Invalid publisher entry point")
        checks = [node for node in ast.parse(row["test"]).body
                  if isinstance(node, ast.FunctionDef) and node.name == "check"]
        if len(checks) != 1 or [argument.arg for argument in checks[0].args.args] != ["candidate"]:
            raise ValueError("Exported task must define check(candidate)")
        tree = ast.parse(row["prompt"] + row["canonical_solution"])
        if not any(isinstance(node, ast.FunctionDef) and node.name == row["entry_point"] for node in tree.body):
            raise ValueError("Publisher reference assembly lacks declared entry point")
    return tasks


def original_evidence():
    """Validate previous raw outputs and ledgers with a denied transport factory."""
    if not (ORIGINAL / "analysis.json").exists():
        raise ValueError("Original cached analysis must already exist")
    recovered = run_feedback(ORIGINAL, _deny_client)
    if not recovered.get("complete"):
        raise ValueError("Complete original feedback generation required")
    protocol = json.loads((ORIGINAL / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((ORIGINAL / "synthesis/manifest.json").read_text(encoding="utf-8"))
    original = json.loads((ORIGINAL / "analysis.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in original.items() if key != "content_sha256"}
    if fingerprint(payload) != original.get("content_sha256"):
        raise ValueError("Original cached analysis content changed")
    evaluations = {row["job_id"]: row for row in original["evaluations"]}
    if len(manifest["jobs"]) != 128 or set(evaluations) != {job["job_id"] for job in manifest["jobs"]}:
        raise ValueError("Original complete 128-output scoring set required")
    jobs = []
    for job in manifest["jobs"]:
        path = response_path(ORIGINAL / "synthesis", job)
        raw = json.loads(path.read_text(encoding="utf-8"))
        previous = evaluations[job["job_id"]]
        if raw["output_sha256"] != previous["output_sha256"]:
            raise ValueError("Original score no longer matches raw output")
        jobs.append({key: job[key] for key in ("job_id", "task_id", "prompt_variant", "repetition")}
                    | {"raw_response": proof(path), "output_sha256": raw["output_sha256"],
                       "original_passed": previous["passed"], "original_status": previous["status"]})
    files = ["protocol.json", "pipeline_report.json", "analysis.json",
             "synthesis/manifest.json", "synthesis/generation_report.json"]
    files.extend(f"{phase}/requests.jsonl" for phase in ("round1", "round2", "synthesis"))
    return protocol, jobs, {name: proof(ORIGINAL / name) for name in files}


def prepare_plus(source_path, output_dir):
    """Freeze expanded tests only after all original generation has ended."""
    output = Path(output_dir)
    source = Path(source_path)
    tasks = load_dataset(source)
    protocol, jobs, evidence = original_evidence()
    original_bundle = json.loads(Path(protocol["evaluation"]["path"]).read_text(encoding="utf-8"))
    entries = {task["task_id"]: task["entry_point"] for task in original_bundle["tasks"]}
    if any(tasks[job["task_id"]]["entry_point"] != entries[job["task_id"]] for job in jobs):
        raise ValueError("Expanded and original task entry points differ")
    output.mkdir(exist_ok=True)
    dataset_path = output / "publisher_test.jsonl"
    if dataset_path.exists():
        load_dataset(dataset_path)
    else:
        # Preserve the byte-exact publisher asset, including the full test pool.
        with source.open("rb") as origin, dataset_path.open("xb") as destination:
            shutil.copyfileobj(origin, destination)
            destination.flush()
            os.fsync(destination.fileno())
    manifest = {
        "version": "frozen_feedback_publisher_plus_rescoring_v1",
        "dataset": {"publisher": PUBLISHER, "revision": REVISION,
                    "source_url": f"https://huggingface.co/datasets/{PUBLISHER}/resolve/{REVISION}/test.jsonl",
                    "download_url": f"https://hf-mirror.com/datasets/{PUBLISHER}/resolve/{REVISION}/test.jsonl",
                    "verified_publisher_lfs_sha256": SOURCE_SHA256,
                    "path": str(dataset_path.resolve()), "rows": 164,
                    "license_declared_by_publisher": "apache-2.0",
                    "scope": "publisher_hf_exported_tests_not_authenticated_github_release_or_evalplus_cli"},
        "task_ids": protocol["task_ids"], "task_count": 16, "output_count": 128,
        "reference_count": 164, "jobs": jobs, "original_evidence": evidence,
        "scoring": plus_protocol(seed=20261029, timeout=30.0),
        "statistics_settings": protocol["analysis"],
        "selection": "all_original_128_outputs_no_label_selection_no_regeneration",
        "interpretation": "same_tasks_stronger_test_coverage_not_new_dataset_or_independent_confirmation",
        "cost": {"new_model_requests": 0, "new_model_proxy_usd": "0",
                 "original_actual_experiment_cost": json.loads((ORIGINAL / "analysis.json").read_text())["actual_experiment_cost"],
                 "scope": "original_generation_usage_unchanged_no_new_invoice_claim"},
        "sources_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
    }
    immutable_json(output / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest):
    tasks = load_dataset(manifest["dataset"]["path"])
    if (manifest["dataset"]["revision"] != REVISION
            or manifest["dataset"]["verified_publisher_lfs_sha256"] != SOURCE_SHA256
            or manifest["scoring"] != plus_protocol(seed=20261029, timeout=30.0)
            or manifest["sources_sha256"] != {name: file_sha256(ROOT / name) for name in SOURCE_FILES}):
        raise ValueError("Frozen enhanced scoring protocol or implementation changed")
    protocol, jobs, evidence = original_evidence()
    if (manifest["task_ids"] != protocol["task_ids"] or manifest["jobs"] != jobs
            or manifest["original_evidence"] != evidence
            or manifest["statistics_settings"] != protocol["analysis"]):
        raise ValueError("Original frozen evidence differs")
    if manifest["output_count"] != 128 or manifest["reference_count"] != 164:
        raise ValueError("Enhanced comparison set changed")
    for proof_row in evidence.values():
        if file_sha256(proof_row["path"]) != proof_row["sha256"]:
            raise ValueError("Original source evidence changed")
    expected = prepare_plus(manifest["dataset"]["path"], Path(manifest["dataset"]["path"]).parent)
    if manifest != expected:
        raise ValueError("Frozen enhanced manifest fields changed")
    return tasks


def _signed(payload):
    return {**payload, "content_sha256": fingerprint(payload)}


def _load_signed(path, identity):
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {key: value for key, value in stored.items() if key != "content_sha256"}
    if (fingerprint(payload) != stored.get("content_sha256")
            or any(stored.get(key) != value for key, value in identity.items())):
        raise ValueError("Cached enhanced artifact content or identity changed")
    return stored


def _score(output, task, manifest):
    return score_plus(output, task, seed=manifest["scoring"]["seed"],
                      timeout=manifest["scoring"]["timeout_seconds"])


def reference_acceptance(output, manifest, tasks):
    """Do not publish candidate comparisons if any publisher reference fails."""
    path = output / "reference_acceptance.json"
    identity = {"manifest_sha256": fingerprint(manifest)}
    if path.exists():
        acceptance = _load_signed(path, identity)
    else:
        folder = output / "reference_checks"
        folder.mkdir(exist_ok=True)
        rows = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            code = task["prompt"] + task["canonical_solution"]
            record_identity = {**identity, "task_id": task_id, "reference_sha256": fingerprint(code),
                               "test_sha256": fingerprint(task["test"])}
            record_path = folder / f"{fingerprint(task_id)}.json"
            if record_path.exists():
                row = _load_signed(record_path, record_identity)
            else:
                started = time.monotonic()
                row = _signed({**record_identity, **_score(code, task, manifest),
                               "local_evaluation_seconds": time.monotonic() - started})
                write_json(record_path, row)
            rows.append(row)
            print(json.dumps({"phase": "reference_acceptance", "completed": len(rows), "total": 164,
                              "task_id": task_id, "status": row["status"]}), flush=True)
            if not row["passed"]:
                raise ValueError(f"Publisher reference failed; no candidate comparison: {task_id}")
        acceptance = _signed({**identity, "reference_count": len(rows),
                              "all_passed": all(row["passed"] for row in rows), "checks": rows,
                              "independent_interpreter_executions": 2 * len(rows)})
        write_json(path, acceptance)
    if (not acceptance["all_passed"] or acceptance["reference_count"] != 164
            or len(acceptance["checks"]) != 164
            or {row["task_id"] for row in acceptance["checks"]} != set(tasks)):
        raise ValueError("Incomplete reference acceptance")
    folder = output / "reference_checks"
    for row in acceptance["checks"]:
        if _load_signed(folder / f"{fingerprint(row['task_id'])}.json",
                        {**identity, "task_id": row["task_id"]}) != row:
            raise ValueError("Reference record differs from acceptance")
        task = tasks[row["task_id"]]
        if (not row["passed"] or row["status"] != "pass"
                or row["reference_sha256"] != fingerprint(task["prompt"] + task["canonical_solution"])
                or row["test_sha256"] != fingerprint(task["test"])):
            raise ValueError("Reference acceptance differs from frozen publisher tests")
    return acceptance


def rescore_feedback_plus(output_dir):
    """Resume settled local scoring records; never construct an API client."""
    output = Path(output_dir)
    descriptor = os.open(output / ".rescore.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        tasks = validate_manifest(manifest)
        acceptance = reference_acceptance(output, manifest, tasks)
        identity = {"manifest_sha256": fingerprint(manifest),
                    "reference_acceptance_sha256": file_sha256(output / "reference_acceptance.json")}
        path = output / "analysis.json"
        if path.exists():
            result = _load_signed(path, identity)
            for row in result["evaluations"]:
                record_path = output / "candidate_checks" / f"{fingerprint(row['job_id'])}.json"
                if _load_signed(record_path, {**identity, "job_id": row["job_id"]}) != row:
                    raise ValueError("Cached individual score differs")
            return result
        folder = output / "candidate_checks"
        folder.mkdir(exist_ok=True)
        rows = []
        for job in manifest["jobs"]:
            task = tasks[job["task_id"]]
            record_identity = {**identity, **job, "test_sha256": fingerprint(task["test"])}
            record_path = folder / f"{fingerprint(job['job_id'])}.json"
            if record_path.exists():
                row = _load_signed(record_path, record_identity)
            else:
                raw = json.loads(Path(job["raw_response"]["path"]).read_text(encoding="utf-8"))
                started = time.monotonic()
                scored = ({"passed": False, "status": "original_truncation", "independent_checks": 0}
                          if job["original_status"] == "truncated" else _score(raw["content"], task, manifest))
                row = _signed({**record_identity, **scored, "local_evaluation_seconds": time.monotonic() - started})
                write_json(record_path, row)
            rows.append(row)
            print(json.dumps({"phase": "candidate_rescoring", "completed": len(rows), "total": 128}), flush=True)
        stats, primary = replication_statistics(rows, manifest["statistics_settings"])
        original_rows = [{**row, "passed": row["original_passed"]} for row in rows]
        original_stats, _ = replication_statistics(original_rows, manifest["statistics_settings"])
        transitions = {}
        for policy in original_stats["groups"]:
            group = [row for row in rows if row["prompt_variant"] == policy]
            transitions[policy] = {
                "pass_to_fail": sum(row["original_passed"] and not row["passed"] for row in group),
                "fail_to_pass": sum(not row["original_passed"] and row["passed"] for row in group),
                "both_pass": sum(row["original_passed"] and row["passed"] for row in group),
                "both_fail": sum(not row["original_passed"] and not row["passed"] for row in group),
            }
        validate_manifest(manifest)
        result = _signed({
            **identity, "statistics": stats, "primary_comparison": primary,
            "original_statistics": original_stats, "transitions": transitions, "evaluations": rows,
            "reference_acceptance": {"all_passed": acceptance["all_passed"], "tasks": 164},
            "cost": manifest["cost"], "new_model_requests": 0,
            "original_artifacts_unchanged": True, "interpretation": manifest["interpretation"],
            "independent_candidate_executions": sum(row["independent_checks"] for row in rows),
            "local_reference_evaluation_seconds": sum(row["local_evaluation_seconds"] for row in acceptance["checks"]),
            "local_candidate_evaluation_seconds": sum(row["local_evaluation_seconds"] for row in rows),
        })
        write_json(path, result)
        return result
    finally:
        (output / ".rescore.lock").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_plus(args.source, args.output_dir)
    result = rescore_feedback_plus(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "statistics", "primary_comparison", "transitions", "new_model_requests")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
