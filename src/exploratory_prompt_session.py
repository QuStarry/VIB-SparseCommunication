"""Prospective serial prompt sessions with observed, not invoice, budgets."""

from decimal import Decimal
import json
import os
from pathlib import Path

from experiments.calibrate_prompt_protocol import _cost
from experiments.run_prompt_contrast_pilot import (
    _capture_transport, _index_records, _summary, response_path,
    validate_generation, write_json,
)
from .experiment_driver import file_sha256
from .request_ledger import RequestLedger, fingerprint, summarize_requests


def _attempt(client, manifest, job, operation):
    error = None
    try:
        client.chat_single_attempt(job["messages"], **job["parameters"],
                                   operation_id=operation, stage="code")
    except Exception as exception:
        error = {"error_type": type(exception).__name__,
                 "http_status": getattr(exception, "status_code", None)}
    records = client.get_request_records()
    matches = [row for row in records if row.get("operation_id") == operation]
    if len(matches) != 1:
        return records, "missing_or_duplicate_request_evidence"
    if error and matches[0].get("status") != "truncated":
        client.request_ledger.finish(matches[0]["request_id"], **error)
        records = client.get_request_records()
    return records, None


def _spent(records):
    stats = summarize_requests(records)
    return _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"])


def _session_summary(manifest, records, stop_reason, attempts):
    summary = _summary(manifest, records, stop_reason, attempts)
    # Missing new evidence is unknown even when earlier settled usage exists.
    if stop_reason == "missing_or_duplicate_request_evidence":
        summary.update(usage_complete=False, proxy_cost_usd=None)
    return summary


def validate_exploratory_generation(manifest, output, report):
    """Reuse frozen artifact checks while preserving stricter missing-evidence cost."""
    summary = report["summary"]
    expected = _session_summary(manifest, report["requests"], summary["stop_reason"], summary["new_attempts"])
    if summary != expected:
        raise ValueError("Exploratory summary differs from request evidence")
    legacy_summary = _summary(manifest, report["requests"], summary["stop_reason"], summary["new_attempts"])
    return validate_generation(manifest, output, {**report, "summary": legacy_summary})


def execute_exploratory(manifest, output_dir, client_factory, validator):
    """Execute a validated contract without consulting any correctness labels.

    The validator is trusted code bound by the caller's frozen manifest. This
    runner uses the historical proxy rates, with a maximum one-dollar observed
    threshold; it does not assert verified token bounds or invoice pricing.
    """
    validator(manifest)
    budget = Decimal(manifest["proxy_stop_usd"])
    if not Decimal(0) < budget <= Decimal(1):
        raise ValueError("Exploratory proxy budget must be positive and at most one dollar")
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    lock = output / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        path = output / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Existing immutable manifest differs")
        else:
            write_json(path, manifest)
        (output / "responses").mkdir(exist_ok=True)
        report_path = output / "generation_report.json"
        if report_path.exists():
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            validate_exploratory_generation(manifest, output, stored)
            return stored
        if client_factory is None:
            return {"prepared": True, "manifest_sha256": fingerprint(manifest)}
        ledger_path = output / "requests.jsonl"
        ledger_path.touch(exist_ok=True)
        records = RequestLedger(ledger_path).records()
        indexed, stop_reason = _index_records(manifest, output, records)
        if not stop_reason and _spent(records) > budget:
            stop_reason = "observed_proxy_budget_exceeded"
        manifest_id = fingerprint(manifest)
        client, active_job, attempts = None, [None], 0
        for job in manifest["jobs"]:
            if stop_reason:
                break
            operation = fingerprint([manifest_id, job["job_id"]])
            if operation in indexed:
                continue
            if response_path(output, job).exists():
                stop_reason = "orphan_response_artifact"
                break
            reserve = _cost(job["planning_input_units"], job["parameters"]["max_tokens"])
            if _spent(records) + reserve > budget:
                stop_reason = "proxy_budget_stop"
                break
            if client is None:
                try:
                    client = client_factory(ledger_path)
                    endpoint = manifest["endpoint"]
                    if (client.provider, client.model, client.base_url) != (
                            endpoint["provider"], endpoint["model"], endpoint["base_url"]):
                        stop_reason = "client_endpoint_mismatch"
                        break
                    _capture_transport(client, output, active_job)
                except Exception:
                    stop_reason = "client_initialization_failed"
                    break
            active_job[0] = job
            client.request_context = {"replication_job_id": job["job_id"],
                                      "source_task_id": job["task_id"]}
            attempts += 1
            records, stop_reason = _attempt(client, manifest, job, operation)
            if not stop_reason:
                indexed, stop_reason = _index_records(manifest, output, records)
            if not stop_reason and _spent(records) > budget:
                stop_reason = "observed_proxy_budget_exceeded"
        summary = _session_summary(manifest, records, stop_reason, attempts)
        report = {
            "manifest_sha256": manifest_id, "ledger_sha256": file_sha256(ledger_path),
            "response_sha256": {
                job["job_id"]: file_sha256(response_path(output, job))
                for job in manifest["jobs"] if response_path(output, job).exists()},
            "summary": summary, "requests": records,
        }
        write_json(report_path, report)
        return report
    finally:
        lock.unlink()
