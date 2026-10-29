"""Frozen 48-task replication, disjoint from the first 24-task prompt pilot."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import random

from experiments.calibrate_prompt_protocol import _bytes, _cost
from experiments.check_prompt_contrast_readiness import _load_draft, endpoint_metadata
from experiments.run_prompt_contrast_pilot import (
    ROOT, configure_live_client, freeze_pilot,
)
from src.experiment_driver import file_sha256, implementation_sha256
from src.exploratory_prompt_session import execute_exploratory
from src.llm_client import DEFAULT_PRO_ENV


def freeze_replication(draft_path, prior_manifest_path, endpoint):
    """Sample task IDs only; historical labels and new pilot scores are unused."""
    if (endpoint.get("provider") != "private" or not endpoint.get("credential_present")
            or endpoint.get("credential_env") != "PRIVATE_API_KEY_1"):
        raise ValueError("Explicit second private credential required")
    path, prior_path = Path(draft_path).resolve(), Path(prior_manifest_path).resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior != freeze_pilot(path, prior["endpoint"]):
        raise ValueError("First pilot contract or source changed")
    if any(endpoint[key] != prior["endpoint"][key] for key in ("provider", "model", "base_url")):
        raise ValueError("Replication must preserve the pilot endpoint")
    draft, digest = _load_draft(path)
    excluded = set(prior["task_ids"])
    remaining = sorted({job["task_id"] for job in draft["jobs"]} - excluded)
    selected = random.Random(20261030).sample(remaining, 48)
    selected_set = set(selected)
    fields = ("job_id", "task_id", "prompt_variant", "repetition", "messages",
              "messages_sha256", "parameters", "candidate_stream_sha256", "stage")
    jobs = [{**{field: job[field] for field in fields},
             "planning_input_units": _bytes(job["messages"]) + 1024}
            for job in draft["jobs"] if job["task_id"] in selected_set]
    random.Random(20261031).shuffle(jobs)
    scenario = sum(_cost(job["planning_input_units"], 1024) for job in jobs)
    if len(jobs) != 288 or scenario > Decimal(1):
        raise ValueError("Full replication does not fit the proxy planning scenario")
    return {
        "version": "archival_prompt_replication_48_v1",
        "scope": "prospective_disjoint_archival_prompt_only_replication_not_new_corpus",
        "endpoint": endpoint, "draft_path": str(path), "draft_sha256": digest,
        "prior_manifest_path": str(prior_path), "prior_manifest_sha256": file_sha256(prior_path),
        "excluded_task_ids": sorted(excluded), "task_ids": selected, "task_count": 48,
        "sample_seed": 20261030, "order_seed": 20261031, "jobs": jobs, "repeats": 2,
        "evaluation": {**draft["evaluation"],
                       "path": str((path.parent / draft["evaluation"]["path"]).resolve())},
        "analysis": {
            "unit": "task_mean_over_two_repetitions", "primary": ["legacy_compact", "current_arbitration"],
            "bootstrap_samples": 10000, "seed": 20261030,
            "exact_sign_test": "two_sided_direction_on_non_tied_task_means",
            "primary_alpha": 0.05, "secondary_pairs": "exploratory_unadjusted",
            "interpretation": "no_equivalence_claim_no_vib_causal_claim",
            "truncated_response": "failure_without_regeneration",
            "incomplete_generation": "no_accuracy_comparison",
            "first_pilot": "report_separately_no_pooled_confirmation",
        },
        "proxy_stop_usd": "1", "planned_proxy_scenario_usd": str(scenario),
        "price_date": "2026-10-23", "price_scope": "historical_proxy_not_actual_invoice",
        "input_usd_per_million": "0.30", "output_usd_per_million": "1.20",
        "budget_scope": "observed_proxy_fail_fast_not_guaranteed_provider_invoice_cap",
        "planning_input_measure": "utf8_bytes_plus_1024_not_verified_bound",
        "authorization_scope": "user_continue_small_actual_testing_20261028",
        "full_benchmark_authorized": False, "workers": 1, "sdk_retries": 0,
        "attempts_per_job": 1, "timeout_seconds": 60,
        "implementation_sha256": file_sha256(__file__),
        "analysis_implementation_sha256": file_sha256(ROOT / "experiments/analyze_prompt_contrast_replication.py"),
        "request_implementation_sha256": implementation_sha256(),
        "dependencies_sha256": {
            relative: file_sha256(ROOT / relative) for relative in (
                "src/exploratory_prompt_session.py", "src/isolated_code_evaluation.py",
                "experiments/run_prompt_contrast_pilot.py", "experiments/analyze_prompt_contrast_pilot.py",
                "experiments/check_prompt_contrast_readiness.py", "experiments/calibrate_prompt_protocol.py")},
    }


def validate_replication(manifest):
    if manifest != freeze_replication(
            manifest["draft_path"], manifest["prior_manifest_path"], manifest["endpoint"]):
        raise ValueError("Replication inputs, contract or implementation changed")


def execute_replication(manifest, output_dir, client_factory):
    return execute_exploratory(manifest, output_dir, client_factory, validate_replication)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "results/prompt_contrast_preparation_20261028/draft.json")
    parser.add_argument("--prior-manifest", type=Path, default=ROOT / "results/prompt_contrast_pilot_20261028/manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    endpoint = endpoint_metadata(ROOT / DEFAULT_PRO_ENV)
    manifest = freeze_replication(args.draft, args.prior_manifest, endpoint)
    result = execute_replication(manifest, args.output_dir,
                                 (lambda path: configure_live_client(endpoint, path)) if args.execute else None)
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2))
    return 0 if not args.execute or result["summary"]["run_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
