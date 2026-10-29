"""Run frozen shared-candidate receiver interventions, not fresh collaboration."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import random

from experiments.calibrate_prompt_protocol import _bytes, _cost
from experiments.check_prompt_contrast_readiness import endpoint_metadata
from experiments.prepare_gate_receiver import load_gate_design, POLICIES
from experiments.run_prompt_contrast_pilot import ROOT, configure_live_client
from src.experiment_driver import file_sha256, implementation_sha256
from src.exploratory_prompt_session import execute_exploratory
from src.llm_client import DEFAULT_PRO_ENV
from src.request_ledger import fingerprint


def freeze_gate_session(design_path, endpoint):
    """Use one immutable renderer for every policy; only admitted histories vary."""
    if (endpoint.get("provider") != "private" or not endpoint.get("credential_present")
            or endpoint.get("credential_env") != "PRIVATE_API_KEY_1"):
        raise ValueError("Require explicit available second private credential")
    path = Path(design_path).resolve()
    design = load_gate_design(path)
    if (design["encoder"]["dimension"] != 384 or design["encoder"]["injected_encoder"]
            or not design["encoder"]["eval_mode"]):
        raise ValueError("Frozen real encoder identity required")
    namespace = {}
    prompt_path = Path(design["prompt_source"]["path"])
    # Only the previously trusted and hash-bound prompt snapshot is executed.
    exec(compile(prompt_path.read_text(encoding="utf-8"), str(prompt_path), "exec"), namespace)
    renderer = namespace["specification_priority"]
    jobs = []
    for stream in design["streams"]:
        for policy in POLICIES:
            projection = design["projections"][stream["task_id"]][policy]
            messages = [{"role": "user", "content": renderer(stream["prompt"], projection["history"])}]
            for repetition in (0, 1):
                jobs.append({
                    "job_id": f"{stream['task_id']}/{policy}/{repetition}",
                    "task_id": stream["task_id"], "prompt_variant": policy, "repetition": repetition,
                    "stage": "code", "messages": messages, "messages_sha256": fingerprint(messages),
                    "candidate_stream_sha256": stream["sha256"],
                    "parameters": {"temperature": 0, "max_tokens": 1024},
                    "admitted_indices": projection["indices"], "round_counts": projection["round_counts"],
                    "admitted_history_sha256": projection["history_sha256"],
                    "admitted_history_bytes": projection["history_utf8_bytes"],
                    "planning_input_units": _bytes(messages) + 1024,
                })
    random.Random(20261102).shuffle(jobs)
    scenario = sum(_cost(job["planning_input_units"], 1024) for job in jobs)
    if len(jobs) != 256 or scenario > Decimal(1):
        raise ValueError("Complete receiver design exceeds proxy planning budget")
    return {
        "version": "gate_receiver_session_v1", "scope": "receiver_only_archival_intervention_not_end_to_end",
        "design_path": str(path), "design_sha256": file_sha256(path), "endpoint": endpoint,
        "task_ids": design["task_ids"], "task_count": 32, "repeats": 2, "jobs": jobs,
        "policies": list(POLICIES), "encoder": design["encoder"],
        "router_config": design["router_config"], "same_round_history": "incremental",
        "evaluation": design["evaluation"], "prompt_source": design["prompt_source"],
        "budget_matching": design["budget_matching"],
        "common_generation_budget": {"calls_per_task_per_policy": 2, "max_output_tokens_per_call": 1024},
        "cost_scope": "new_synthesis_only_candidate_generation_already_occurred_no_end_to_end_saving_claim",
        "analysis": {
            "unit": "task_mean_over_two_repetitions", "primary": ["vib", "random_matched"],
            "primary_alpha": 0.05, "seed": 20261101, "bootstrap_samples": 10000,
            "exact_sign_test": "two_sided_non_tied_task_means", "secondary": "exploratory_unadjusted",
            "truncation": "failure_no_regeneration", "incomplete": "no_accuracy_comparison",
            "inference": "conditional_receiver_effect_not_end_to_end_causal_superiority_or_equivalence",
        },
        "proxy_stop_usd": "1", "planned_proxy_scenario_usd": str(scenario),
        "price_scope": "historical_proxy_not_actual_invoice", "price_date": "2026-10-23",
        "input_usd_per_million": "0.30", "output_usd_per_million": "1.20",
        "budget_scope": "observed_proxy_fail_fast_not_invoice_hard_cap",
        "authorization_scope": "user_continue_small_actual_testing_20261028",
        "workers": 1, "timeout_seconds": 60, "sdk_retries": 0, "attempts_per_job": 1,
        "implementation_sha256": file_sha256(__file__),
        "analysis_implementation_sha256": file_sha256(ROOT / "experiments/analyze_gate_receiver.py"),
        "request_implementation_sha256": implementation_sha256(),
        "dependencies_sha256": {name: file_sha256(ROOT / name) for name in (
            "src/exploratory_prompt_session.py", "src/isolated_code_evaluation.py",
            "src/vib_router.py", "experiments/prepare_gate_receiver.py",
            "experiments/run_prompt_contrast_pilot.py", "experiments/calibrate_prompt_protocol.py",
            "experiments/analyze_prompt_contrast_replication.py", "experiments/analyze_prompt_contrast_pilot.py")},
    }


def validate_gate_session(manifest):
    if manifest != freeze_gate_session(manifest["design_path"], manifest["endpoint"]):
        raise ValueError("Receiver session input or implementation changed")


def execute_gate_session(manifest, output_dir, client_factory):
    return execute_exploratory(manifest, output_dir, client_factory, validate_gate_session)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    endpoint = endpoint_metadata(ROOT / DEFAULT_PRO_ENV)
    manifest = freeze_gate_session(args.design, endpoint)
    report = execute_gate_session(manifest, args.output_dir,
                                 (lambda path: configure_live_client(endpoint, path)) if args.execute else None)
    print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))
    return 0 if not args.execute or report["summary"]["run_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
