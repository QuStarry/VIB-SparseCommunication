"""Score completed feedback outputs and separate actual from standalone costs."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from experiments.analyze_prompt_contrast_replication import replication_statistics
from experiments.calibrate_prompt_protocol import _cost
from experiments.run_feedback_experiment import run_feedback, validate_phase_report
from experiments.run_prompt_contrast_pilot import response_path, write_json
from src.experiment_driver import file_sha256
from src.isolated_code_evaluation import load_evaluators
from src.request_ledger import fingerprint


def aggregate_cost(records):
    prompt = sum(row["usage"]["prompt_tokens"] for row in records)
    completion = sum(row["usage"]["completion_tokens"] for row in records)
    return {"requests": len(records), "prompt_tokens": prompt, "completion_tokens": completion,
            "proxy_usd": str(_cost(prompt, completion))}


def analyze_feedback(output_dir):
    """No hidden scoring until all 368 requests are settled and validated."""
    output = Path(output_dir)
    if not run_feedback(output).get("complete"):
        raise ValueError("Complete pipeline required before hidden scoring")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    manifests, records, phase_costs = {}, {}, {}
    identity = {"protocol_sha256": file_sha256(output / "protocol.json"),
                "implementation_sha256": file_sha256(__file__), "phase_report_sha256": {}}
    for phase in protocol["phase_counts"]:
        folder = output / phase
        manifests[phase] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        report_path = folder / "generation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records[phase] = validate_phase_report(protocol, manifests[phase], folder, report)
        identity["phase_report_sha256"][phase] = file_sha256(report_path)
        phase_costs[phase] = aggregate_cost(records[phase])
    path = output / "analysis.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in stored.items() if key != "content_sha256"}
        if any(stored.get(key) != value for key, value in identity.items()) or fingerprint(payload) != stored.get("content_sha256"):
            raise ValueError("Cached feedback analysis changed")
        return stored
    evaluators = load_evaluators(protocol["evaluation"]["path"], protocol["evaluation"]["sha256"])
    manifest = manifests["synthesis"]
    by_operation = {row["operation_id"]: row for row in records["synthesis"]}
    manifest_id = fingerprint(manifest)
    rows = []
    for job in manifest["jobs"]:
        record = by_operation[fingerprint([manifest_id, job["job_id"]])]
        raw = json.loads(response_path(output / "synthesis", job).read_text(encoding="utf-8"))
        passed = False if record["status"] == "truncated" else evaluators[job["task_id"]](raw["content"])
        if type(passed) is not bool:
            raise ValueError("Scorer must return a boolean")
        rows.append({key: job[key] for key in ("task_id", "job_id", "prompt_variant", "repetition")}
                    | {"passed": passed, "status": record["status"], "output_sha256": raw["output_sha256"]})
    stats, primary = replication_statistics(rows, protocol["analysis"])
    route = json.loads((output / "routes_round2.json").read_text(encoding="utf-8"))
    policy_costs, communication = {}, {}
    for policy in protocol["policies"]:
        stage_costs = {}
        for phase in ("round2", "synthesis"):
            phase_manifest = manifests[phase]
            selected = {fingerprint([fingerprint(phase_manifest), job["job_id"]])
                        for job in phase_manifest["jobs"] if job["prompt_variant"] == policy}
            stage_costs[phase] = aggregate_cost([row for row in records[phase] if row["operation_id"] in selected])
        standalone = (Decimal(phase_costs["round1"]["proxy_usd"])
                      + Decimal(stage_costs["round2"]["proxy_usd"])
                      + Decimal(stage_costs["synthesis"]["proxy_usd"]) / 2)
        policy_costs[policy] = {**stage_costs, "standalone_one_synthesis_proxy_usd": str(standalone),
                              "standalone_scope": "shared_round1_full_plus_own_round2_plus_mean_one_synthesis",
                              "standalone_requests": 112}
        arms = [route["histories"][task][policy] for task in protocol["task_ids"]]
        communication[policy] = {
            "messages": sum(sum(arm["round_counts"]) for arm in arms),
            "history_utf8_bytes": sum(arm["history_utf8_bytes"] for arm in arms),
            "round_messages": [sum(arm["round_counts"][index] for arm in arms) for index in (0, 1)],
            "unit": "unique_task_history_not_repetition_weighted",
        }
    result = {**identity, "statistics": stats, "primary_comparison": primary, "evaluations": rows,
              "phase_costs": phase_costs, "policy_costs": policy_costs, "communication": communication,
              "actual_experiment_cost": aggregate_cost([row for values in records.values() for row in values]),
              "truncated_by_phase": {phase: sum(row["status"] == "truncated" for row in values)
                                     for phase, values in records.items()},
              "inference_scope": protocol["scope"]}
    # Publishing must not race a changed report or raw output.
    for phase in records:
        folder = output / phase
        report = json.loads((folder / "generation_report.json").read_text(encoding="utf-8"))
        validate_phase_report(protocol, manifests[phase], folder, report)
        if file_sha256(folder / "generation_report.json") != identity["phase_report_sha256"][phase]:
            raise ValueError("Generation evidence changed during scoring")
    result["content_sha256"] = fingerprint(result)
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_feedback(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "statistics", "primary_comparison", "policy_costs", "actual_experiment_cost",
        "communication", "truncated_by_phase")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
