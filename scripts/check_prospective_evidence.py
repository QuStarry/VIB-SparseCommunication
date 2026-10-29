"""Audit completed prospective cohorts using strictly client-free recovery."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from experiments.analyze_feedback_experiment import analyze_feedback
from experiments.analyze_gate_receiver import analyze_gate_session
from experiments.analyze_prompt_contrast_pilot import analyze_pilot
from experiments.analyze_prompt_contrast_replication import analyze_replication
from experiments.run_feedback_experiment import run_feedback
from experiments.run_gate_receiver import execute_gate_session
from experiments.run_prompt_contrast_pilot import ROOT, execute_pilot, write_json
from experiments.run_prompt_contrast_replication import execute_replication
from src.experiment_driver import file_sha256
from src.request_ledger import fingerprint


def check_evidence(output):
    """Validate all raw outputs, ledgers, sources and cached statistics locally."""
    factories = []

    def deny_client(path):
        factories.append(str(path))
        raise AssertionError("Cached acceptance must never construct an API client")

    cohorts = (
        ("prompt_contrast_pilot_20261028", execute_pilot, analyze_pilot),
        ("prompt_contrast_replication_20261028", execute_replication, analyze_replication),
        ("gate_receiver_20261028", execute_gate_session, analyze_gate_session),
    )
    before, summaries, analyses = {}, {}, {}
    for name, execute, analyze in cohorts:
        folder = ROOT / "results" / name
        ledger = folder / "requests.jsonl"
        before[str(ledger)] = file_sha256(ledger)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        report = execute(manifest, folder, deny_client)
        if not report["summary"]["run_complete"] or not report["summary"]["usage_complete"]:
            raise ValueError("Incomplete prior cohort")
        summaries[name] = report["summary"]
        analyses[name] = fingerprint(analyze(folder))
    feedback_folder = ROOT / "results/feedback_experiment_20261028"
    for phase in ("round1", "round2", "synthesis"):
        ledger = feedback_folder / phase / "requests.jsonl"
        before[str(ledger)] = file_sha256(ledger)
    recovered = run_feedback(feedback_folder, deny_client)
    if not recovered["complete"]:
        raise ValueError("Incomplete feedback pipeline")
    feedback = analyze_feedback(feedback_folder)
    if feedback["actual_experiment_cost"]["requests"] != 368:
        raise ValueError("Unexpected actual feedback request count")
    route = json.loads((feedback_folder / "routes_round2.json").read_text(encoding="utf-8"))
    for arms in route["histories"].values():
        if any(arms[policy]["round_counts"] != arms["vib"]["round_counts"]
               for policy in ("random_matched", "cosine_matched")):
            raise ValueError("Count matching failed")
    for phase, summary in recovered["summaries"].items():
        manifest = json.loads((feedback_folder / phase / "manifest.json").read_text(encoding="utf-8"))
        if Decimal(summary["proxy_cost_usd"]) > Decimal(manifest["proxy_stop_usd"]):
            raise ValueError("Observed phase proxy exceeded declared threshold")
    after = {path: file_sha256(path) for path in before}
    if factories or before != after:
        raise ValueError("Cache recovery mutated a ledger or initialized a client")
    protocol = json.loads((feedback_folder / "protocol.json").read_text(encoding="utf-8"))
    report = {
        "new_api_client_constructions": len(factories), "new_api_requests": 0,
        "ledger_sha256_before": before, "ledger_sha256_after": after,
        "all_ledgers_unchanged": True, "prior_generation_summaries": summaries,
        "prior_analysis_sha256": analyses, "feedback_analysis_sha256": fingerprint(feedback),
        "feedback_protocol_sha256": fingerprint(protocol),
        "actual_feedback_requests": 368, "all_usage_complete": True,
        "matched_every_task_every_round": True,
        "isolated_scoring_executions": 2 * sum(
            row["status"] != "truncated" for row in feedback["evaluations"]),
        "cached_analysis_reused_without_rescoring": True,
        "cost_scope": "historical_proxy_no_actual_invoice",
        "implementation_sha256": file_sha256(__file__),
    }
    report["content_sha256"] = fingerprint(report)
    path = Path(output)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise ValueError("Existing acceptance artifact changed")
    else:
        write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/feedback_experiment_20261028/acceptance.json")
    args = parser.parse_args()
    report = check_evidence(args.output)
    print(json.dumps({key: report[key] for key in (
        "new_api_requests", "all_ledgers_unchanged", "actual_feedback_requests",
        "all_usage_complete", "matched_every_task_every_round", "isolated_scoring_executions")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
