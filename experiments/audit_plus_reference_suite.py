"""Finish source acceptance diagnostics without authorizing candidate scoring."""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.rescore_feedback_plus import (
    _load_signed, _score, _signed, validate_manifest,
)
from experiments.run_prompt_contrast_pilot import write_json
from src.experiment_driver import file_sha256
from src.request_ledger import fingerprint


def _audit_reference_suite(output_dir):
    """Record every reference outcome; any failure keeps candidate scoring blocked."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    tasks = validate_manifest(manifest)
    identity = {"manifest_sha256": fingerprint(manifest),
                "audit_implementation_sha256": file_sha256(__file__)}
    path = output / "source_audit.json"
    if path.exists():
        result = _load_signed(path, identity)
        for row in result["checks"]:
            if _load_signed(output / "reference_checks" / f"{fingerprint(row['task_id'])}.json",
                            {"manifest_sha256": identity["manifest_sha256"], "task_id": row["task_id"]}) != row:
                raise ValueError("Cached source audit reference changed")
        return result
    folder = output / "reference_checks"
    folder.mkdir(exist_ok=True)
    rows = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        code = task["prompt"] + task["canonical_solution"]
        row_identity = {"manifest_sha256": identity["manifest_sha256"], "task_id": task_id,
                        "reference_sha256": fingerprint(code), "test_sha256": fingerprint(task["test"])}
        row_path = folder / f"{fingerprint(task_id)}.json"
        if row_path.exists():
            row = _load_signed(row_path, row_identity)
        else:
            started = time.monotonic()
            row = _signed({**row_identity, **_score(code, task, manifest),
                           "local_evaluation_seconds": time.monotonic() - started})
            write_json(row_path, row)
        rows.append(row)
        print(json.dumps({"phase": "reference_source_audit", "completed": len(rows), "total": 164,
                          "task_id": task_id, "status": row["status"]}), flush=True)
    validate_manifest(manifest)
    result = _signed({
        **identity, "reference_count": len(rows), "passed_references": sum(row["passed"] for row in rows),
        "failed_references": [{"task_id": row["task_id"], "status": row["status"]}
                              for row in rows if not row["passed"]],
        "checks": rows, "all_passed": all(row["passed"] for row in rows),
        "candidate_rescoring_authorized": False,
        "candidate_outputs_evaluated": 0, "new_model_requests": 0,
        "interpretation": "source_acceptance_audit_no_candidate_accuracy_comparison",
        "local_reference_evaluation_seconds": sum(row["local_evaluation_seconds"] for row in rows),
        "independent_reference_executions": sum(row["independent_checks"] for row in rows),
    })
    write_json(path, result)
    return result


def audit_reference_suite(output_dir):
    lock = Path(output_dir) / ".rescore.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        return _audit_reference_suite(output_dir)
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = audit_reference_suite(parser.parse_args().output_dir)
    print(json.dumps({key: result[key] for key in (
        "reference_count", "passed_references", "failed_references",
        "candidate_outputs_evaluated", "new_model_requests")}, ensure_ascii=False, indent=2))
