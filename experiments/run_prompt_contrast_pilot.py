"""Exploratory 24-task prompt pilot; generation never consults hidden scoring."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

from experiments.calibrate_prompt_protocol import _bytes, _cost, _record_reason
from experiments.check_prompt_contrast_readiness import _load_draft, endpoint_metadata
from src.experiment_driver import file_sha256, implementation_sha256
from src.llm_client import build_client, DEFAULT_PRO_ENV
from src.request_ledger import RequestLedger, fingerprint, summarize_requests


ROOT = Path(__file__).resolve().parents[1]
STOP_USD = Decimal("1")
SAMPLE_SEED = 20261028
ORDER_SEED = 20261029


def freeze_pilot(draft_path, endpoint):
    """Freeze all pairs using task IDs only, never historical pass labels."""
    if (not endpoint.get("credential_present") or endpoint.get("credential_env") != "PRIVATE_API_KEY_1"
            or endpoint.get("provider") != "private"):
        raise ValueError("Explicit available second private credential is required")
    path = Path(draft_path).resolve()
    draft, digest = _load_draft(path)
    if draft["repeats"] != 2 or draft["task_count"] != 164:
        raise ValueError("Require the frozen 164-task two-repetition preparation")
    selected = random.Random(SAMPLE_SEED).sample(
        sorted({job["task_id"] for job in draft["jobs"]}), 24)
    selected_set = set(selected)
    fields = ("job_id", "task_id", "prompt_variant", "repetition", "messages",
              "messages_sha256", "parameters", "candidate_stream_sha256", "stage")
    jobs = [{**{field: job[field] for field in fields},
             "planning_input_units": _bytes(job["messages"]) + 1024}
            for job in draft["jobs"] if job["task_id"] in selected_set]
    random.Random(ORDER_SEED).shuffle(jobs)
    planned = sum(_cost(job["planning_input_units"], 1024) for job in jobs)
    if len(jobs) != 144 or planned > STOP_USD:
        raise ValueError("Pilot request count or historical planning budget exceeded")
    return {
        "version": "archival_prompt_contrast_pilot_v1",
        "scope": "exploratory_archival_same_full_stream_prompt_only",
        "endpoint": endpoint, "draft_path": str(path), "draft_sha256": digest,
        "task_ids": selected, "sample_seed": SAMPLE_SEED, "order_seed": ORDER_SEED,
        "jobs": jobs, "task_count": 24, "repeats": 2,
        "evaluation": {**draft["evaluation"],
                       "path": str((path.parent / draft["evaluation"]["path"]).resolve())},
        "analysis": {"unit": "task_mean_over_two_repetitions", "bootstrap_samples": 10000,
                     "seed": SAMPLE_SEED, "truncated_response": "failure_without_regeneration",
                     "incomplete_generation": "no_accuracy_comparison",
                     "multiple_comparisons": "exploratory_unadjusted_intervals"},
        "proxy_stop_usd": str(STOP_USD), "planned_proxy_scenario_usd": str(planned),
        "price_date": "2026-10-23", "price_scope": "historical_proxy_not_actual_invoice",
        "input_usd_per_million": "0.30", "output_usd_per_million": "1.20",
        "planning_input_measure": "utf8_content_bytes_plus_1024_not_verified_token_bound",
        "budget_scope": "observed_proxy_fail_fast_not_guaranteed_provider_invoice_cap",
        "authorization_scope": "user_requested_actual_small_testing_and_continue_20261028",
        "full_benchmark_authorized": False, "sdk_retries": 0, "attempts_per_job": 1,
        "workers": 1, "timeout_seconds": 60,
        "implementation_sha256": file_sha256(__file__),
        "analysis_implementation_sha256": file_sha256(
            ROOT / "experiments/analyze_prompt_contrast_pilot.py"),
        "request_implementation_sha256": implementation_sha256(),
        "dependencies_sha256": {
            name: file_sha256(ROOT / "experiments" / name)
            for name in ("calibrate_prompt_protocol.py", "check_prompt_contrast_readiness.py")},
    }


def write_json(path, value):
    """Exclusive durable artifacts; existing evidence is never overwritten."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def response_path(output, job):
    return Path(output) / "responses" / f"{fingerprint(job['job_id'])}.json"


def _operation(manifest, job):
    return fingerprint([fingerprint(manifest), job["job_id"]])


def _capture_transport(client, output, active_job):
    # Capture before the client rejects length responses, retaining paid text.
    transport = client.client.chat.completions.create

    def create(**kwargs):
        response = transport(**kwargs)
        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        content = getattr(getattr(choice, "message", None), "content", None)
        job = active_job[0]
        write_json(response_path(output, job), {
            "job_id": job["job_id"], "content": content, "output_sha256": fingerprint(content),
            "response_id": getattr(response, "id", None), "served_model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
        })
        return response

    client._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))


def _artifact_reason(output, job, record):
    path = response_path(output, job)
    if not path.exists():
        return "missing_response_artifact"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    content = artifact.get("content")
    if (artifact.get("job_id") != job["job_id"] or not isinstance(content, str)
            or not content.strip() or artifact.get("output_sha256") != fingerprint(content)
            or record.get("output_sha256") != fingerprint(content)
            or any(artifact.get(key) != record.get(key)
                   for key in ("response_id", "served_model", "finish_reason"))):
        return "response_artifact_mismatch"
    if record["status"] == "completed" and record["finish_reason"] != "stop":
        return "unexpected_finish_reason"
    return None


def _index_records(manifest, output, records):
    manifest_id = fingerprint(manifest)
    jobs = {fingerprint([manifest_id, job["job_id"]]): job for job in manifest["jobs"]}
    indexed = {}
    for record in records:
        operation = record.get("operation_id")
        if operation not in jobs or operation in indexed:
            return indexed, "foreign_or_duplicate_request_evidence"
        indexed[operation] = record
        reason = _record_reason(record, manifest, jobs[operation])
        if not reason:
            reason = _artifact_reason(output, jobs[operation], record)
        if reason:
            return indexed, reason
    return indexed, None


def _summary(manifest, records, stop_reason, attempts):
    stats = summarize_requests(records)
    cost = _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"])
    usage_complete = stats["usage_complete"] and len(records) >= attempts
    return {
        "requests_with_evidence": len(records), "planned_requests": len(manifest["jobs"]),
        "new_attempts": attempts, "prompt_tokens": stats["total_prompt_tokens"],
        "completion_tokens": stats["total_completion_tokens"], "total_tokens": stats["total_tokens"],
        "usage_complete": usage_complete,
        "proxy_cost_usd": str(cost) if usage_complete else None,
        "known_usage_proxy_cost_usd": str(cost), "price_scope": manifest["price_scope"],
        "actual_invoice_usd": None, "actual_invoice_rmb": None,
        "truncated_responses": sum(record.get("status") == "truncated" for record in records),
        "stop_reason": stop_reason,
        "run_complete": not stop_reason and len(records) == len(manifest["jobs"]),
        "full_benchmark_executed": False, "full_prompt_token_bounds_proven": False,
    }


def validate_generation(manifest, output, report):
    """Check frozen report and underlying evidence without constructing a client."""
    records = RequestLedger(Path(output) / "requests.jsonl").records()
    if (report["manifest_sha256"] != fingerprint(manifest) or report["requests"] != records
            or report["ledger_sha256"] != file_sha256(Path(output) / "requests.jsonl")):
        raise ValueError("Generation report or ledger fingerprint differs")
    _, reason = _index_records(manifest, output, records)
    if report["summary"]["run_complete"] and reason:
        raise ValueError(f"Complete generation evidence is invalid: {reason}")
    for job in manifest["jobs"]:
        path = response_path(output, job)
        expected = report["response_sha256"].get(job["job_id"])
        if (file_sha256(path) if path.exists() else None) != expected:
            raise ValueError("Response artifact fingerprint differs")
    expected = _summary(manifest, records, report["summary"]["stop_reason"],
                        report["summary"]["new_attempts"])
    if report["summary"] != expected:
        raise ValueError("Generation summary differs from request evidence")
    return records


def execute_pilot(manifest, output_dir, client_factory):
    """Resume settled jobs only; unknown requests permanently block new transport."""
    if manifest != freeze_pilot(manifest["draft_path"], manifest["endpoint"]):
        raise ValueError("Pilot protocol, inputs or implementation changed")
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    lock = output / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        path = output / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Frozen manifest differs; use a new run")
        else:
            write_json(path, manifest)
        (output / "responses").mkdir(exist_ok=True)
        report_path = output / "generation_report.json"
        if report_path.exists():
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            validate_generation(manifest, output, stored)
            return stored
        if client_factory is None:
            return {"prepared": True, "manifest_sha256": fingerprint(manifest)}
        ledger_path = output / "requests.jsonl"
        ledger_path.touch(exist_ok=True)
        records = RequestLedger(ledger_path).records()
        indexed, stop_reason = _index_records(manifest, output, records)
        stats = summarize_requests(records)
        if not stop_reason and _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"]) > STOP_USD:
            stop_reason = "observed_proxy_budget_exceeded"
        client, attempts, active_job = None, 0, [None]
        for job in manifest["jobs"]:
            if stop_reason:
                break
            operation = _operation(manifest, job)
            if operation in indexed:
                continue
            if response_path(output, job).exists():
                stop_reason = "orphan_response_artifact"
                break
            stats = summarize_requests(records)
            spent = _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"])
            if spent + _cost(job["planning_input_units"], 1024) > STOP_USD:
                stop_reason = "proxy_budget_stop"
                break
            if client is None:
                try:
                    client = client_factory(ledger_path)
                    expected = manifest["endpoint"]
                    if (client.provider, client.model, client.base_url) != (
                            expected["provider"], expected["model"], expected["base_url"]):
                        stop_reason = "client_endpoint_mismatch"
                        break
                    _capture_transport(client, output, active_job)
                except Exception:
                    stop_reason = "client_initialization_failed"
                    break
            active_job[0] = job
            client.request_context = {"pilot_job_id": job["job_id"], "source_task_id": job["task_id"]}
            attempts += 1
            error = None
            try:
                client.chat_single_attempt(job["messages"], **job["parameters"],
                                           operation_id=operation, stage="code")
            except Exception as exception:
                error = {"error_type": type(exception).__name__,
                         "http_status": getattr(exception, "status_code", None)}
            records = client.get_request_records()
            matches = [record for record in records if record["operation_id"] == operation]
            if len(matches) != 1:
                stop_reason = "missing_or_duplicate_request_evidence"
                break
            if error and matches[0].get("status") != "truncated":
                client.request_ledger.finish(matches[0]["request_id"], **error)
                records = client.get_request_records()
            indexed, stop_reason = _index_records(manifest, output, records)
            stats = summarize_requests(records)
            if not stop_reason and _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"]) > STOP_USD:
                stop_reason = "observed_proxy_budget_exceeded"
        report = {
            "manifest_sha256": fingerprint(manifest), "ledger_sha256": file_sha256(ledger_path),
            "response_sha256": {
                job["job_id"]: file_sha256(response_path(output, job))
                for job in manifest["jobs"] if response_path(output, job).exists()},
            "summary": _summary(manifest, records, stop_reason, attempts), "requests": records,
        }
        write_json(report_path, report)
        return report
    finally:
        lock.unlink()


def configure_live_client(endpoint, ledger_path):
    """Pilot-only SDK limits; old benchmark client defaults remain unchanged."""
    if os.environ.get("VIB_PRIVATE_API_KEY_ENV") != endpoint["credential_env"]:
        raise ValueError("Select the second credential explicitly; no fallback")
    client = build_client(endpoint["model"], "private", env_path=str(ROOT / DEFAULT_PRO_ENV),
                          temperature=0, max_tokens=1024, log=False, ledger_path=str(ledger_path))
    client._client = client.client.with_options(timeout=60, max_retries=0)
    return client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "results/prompt_contrast_preparation_20261028/draft.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    endpoint = endpoint_metadata(ROOT / DEFAULT_PRO_ENV)
    manifest = freeze_pilot(args.draft, endpoint)
    report = execute_pilot(manifest, args.output_dir,
                           (lambda path: configure_live_client(endpoint, path)) if args.execute else None)
    print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))
    return 0 if not args.execute or report["summary"]["run_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
