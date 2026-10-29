"""Verify complete local rescoring using denied scoring and preserved evidence."""

import json
from pathlib import Path
from unittest.mock import patch

from experiments.rescore_repaired_plus import PARENT, rescore_repaired
from experiments.run_feedback_experiment import immutable_json
from experiments.run_prompt_contrast_pilot import ROOT
from src.experiment_driver import file_sha256


def check_local_acceptance(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    before = {name: file_sha256(item["path"]) for name, item in manifest["original_evidence"].items()}
    checks_before = {str(path): file_sha256(path) for folder in ("reference_checks", "candidate_checks")
                     for path in (output / folder).glob("*.json")}
    # This standalone acceptance process forbids fallback computation explicitly.
    with patch("experiments.rescore_repaired_plus._score",
               side_effect=AssertionError("Unexpected candidate scoring")) as candidates:
        with patch("experiments.rescore_feedback_plus._score",
                   side_effect=AssertionError("Unexpected reference scoring")) as references:
            result = rescore_repaired(output)
            if candidates.call_count or references.call_count:
                raise ValueError("Cached validation performed new scoring")
    after = {name: file_sha256(item["path"]) for name, item in manifest["original_evidence"].items()}
    checks_after = {str(path): file_sha256(path) for folder in ("reference_checks", "candidate_checks")
                    for path in (output / folder).glob("*.json")}
    prior = json.loads((PARENT / "acceptance.json").read_text())
    if file_sha256(PARENT / "source_audit.json") != prior["source_audit_sha256"]:
        raise ValueError("Original source failure evidence changed")
    if before != after or checks_before != checks_after:
        raise ValueError("Cache validation changed evidence")
    if len(result["evaluations"]) != 128 or not result["reference_acceptance"]["all_passed"]:
        raise ValueError("Incomplete local comparison")
    report = {
        "original_files_sha256_before": before, "original_files_sha256_after": after,
        "original_files_unchanged": True, "local_records_unchanged": True,
        "old_source_failure_preserved": True, "new_model_requests": 0,
        "new_local_scoring_calls_on_resume": 0, "candidate_outputs": 128,
        "independent_candidate_executions": result["independent_candidate_executions"],
        "validation_reference_scope": result["reference_validation_scope"],
        "analysis_sha256": file_sha256(output / "analysis.json"),
        "patch_sha256": file_sha256(output / "patch.json"),
        "scope": "user_authorized_local_revision_not_official_evalplus_score",
    }
    immutable_json(output / "acceptance.json", report)
    return report


if __name__ == "__main__":
    report = check_local_acceptance(ROOT / "results/feedback_plus_repaired_20261029")
    print(json.dumps({key: report[key] for key in (
        "original_files_unchanged", "local_records_unchanged",
        "old_source_failure_preserved", "new_model_requests",
        "new_local_scoring_calls_on_resume", "candidate_outputs")}, indent=2))
