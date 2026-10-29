"""Six-request protocol calibration only; not benchmark execution or scoring."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path

from experiments.check_prompt_contrast_readiness import _load_draft, endpoint_metadata, VARIANTS
from src.experiment_driver import file_sha256, implementation_sha256
from src.llm_client import build_client, DEFAULT_PRO_ENV
from src.request_ledger import RequestLedger, fingerprint, summarize_requests


ROOT = Path(__file__).resolve().parents[1]
INPUT_RATE, OUTPUT_RATE = Decimal("0.30"), Decimal("1.20")
STOP_USD = Decimal("0.05")


def _cost(prompt, completion):
    return (Decimal(prompt) * INPUT_RATE + Decimal(completion) * OUTPUT_RATE) / Decimal(1000000)


def _bytes(messages):
    return sum(len(message["content"].encode("utf-8")) for message in messages)


def freeze_probe(draft_path, endpoint):
    """Select length extremes without labels; freeze a separate tiny protocol."""
    if not endpoint.get("credential_present"):
        raise ValueError("Selected credential is missing; no fallback is permitted")
    draft, digest = _load_draft(Path(draft_path))
    jobs = []
    for variant in sorted(VARIANTS):
        candidates = [job for job in draft["jobs"] if job["prompt_variant"] == variant and job["repetition"] == 0]
        ordered = sorted(candidates, key=lambda job: (_bytes(job["messages"]), job["job_id"]))
        for label, job in (("shortest", ordered[0]), ("longest", ordered[-1])):
            jobs.append({
                "job_id": f"probe/{variant}/{label}", "source_job_id": job["job_id"],
                "task_id": job["task_id"], "prompt_variant": variant, "length_selection": label,
                "messages": job["messages"], "parameters": {"temperature": 0, "max_tokens": 32},
                "planning_input_units": _bytes(job["messages"]) + 1024,
            })
    planned = sum(_cost(job["planning_input_units"], 32) for job in jobs)
    if planned > STOP_USD:
        raise ValueError("Small calibration planning scenario exceeds proxy stop threshold")
    return {
        "version": "small_prompt_protocol_calibration_v1",
        "scope": "small_protocol_calibration_not_benchmark", "endpoint": endpoint,
        "draft_sha256": digest, "jobs": jobs, "planned_proxy_scenario_usd": str(planned),
        "proxy_stop_usd": str(STOP_USD), "price_date": "2026-10-23",
        "price_scope": "historical_proxy_not_actual_invoice",
        "input_usd_per_million": str(INPUT_RATE), "output_usd_per_million": str(OUTPUT_RATE),
        "sdk_retries": 0, "attempts_per_job": 1, "workers": 1, "timeout_seconds": 60,
        "budget_scope": "observed_proxy_fail_fast_not_guaranteed_provider_invoice_cap",
        "authorization_scope": "user_requested_actual_small_testing_20261028",
        "full_benchmark_authorized": False,
        "implementation_sha256": file_sha256(__file__),
        "request_implementation_sha256": implementation_sha256(),
    }


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _record_reason(record, manifest, job):
    if record.get("http_status") in (401, 403, 429):
        return f"http_{record['http_status']}"
    if record.get("usage_status") != "known":
        return "unknown_usage"
    if record.get("served_model") != manifest["endpoint"]["model"]:
        return "served_model_mismatch"
    if record.get("parameters") != job["parameters"] or record.get("prompt_sha256") != fingerprint(job["messages"]):
        return "request_parameters_or_prompt_mismatch"
    if not record.get("response_id"):
        return "missing_response_identity"
    if record["usage"]["completion_tokens"] > job["parameters"]["max_tokens"]:
        return "completion_cap_violated"
    if record.get("status") not in ("completed", "truncated"):
        return "invalid_response"
    return None


def _report(manifest, records, stop_reason, *, new_attempts):
    stats = summarize_requests(records)
    cost = _cost(stats["total_prompt_tokens"], stats["total_completion_tokens"])
    usage_complete = stats["usage_complete"] and len(records) >= new_attempts
    summary = {
        "requests_with_evidence": len(records), "new_attempts": new_attempts,
        "prompt_tokens": stats["total_prompt_tokens"], "completion_tokens": stats["total_completion_tokens"],
        "total_tokens": stats["total_tokens"], "usage_complete": usage_complete,
        "proxy_cost_usd": str(cost) if usage_complete else None,
        "known_usage_proxy_cost_usd": str(cost), "price_scope": manifest["price_scope"],
        "actual_invoice_usd": None, "actual_invoice_rmb": None, "stop_reason": stop_reason,
        "truncated_responses": sum(record.get("status") == "truncated" for record in records),
        "protocol_checks_passed": not stop_reason and len(records) == len(manifest["jobs"]),
        "full_prompt_token_bounds_proven": False, "full_benchmark_executed": False,
    }
    return {"manifest_sha256": fingerprint(manifest), "summary": summary, "requests": records}


def execute_probe(manifest, output_dir, client_factory):
    """Persist before transport, never retry, and never resend preexisting evidence.

    A None factory prepares files only. Injected factories must own the one
    durable ledger, use single-attempt transport and respect the frozen endpoint.
    Length responses are expected accounting samples, not valid code solutions.
    """
    if (manifest["scope"] != "small_protocol_calibration_not_benchmark"
            or len(manifest["jobs"]) != 6
            or not Decimal(0) < Decimal(manifest["proxy_stop_usd"]) <= STOP_USD
            or manifest["implementation_sha256"] != file_sha256(__file__)
            or manifest["request_implementation_sha256"] != implementation_sha256()
            or any(job["parameters"] != {"temperature": 0, "max_tokens": 32} for job in manifest["jobs"])):
        raise ValueError("Small calibration contract or implementation changed")
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    manifest_path, report_path = output / "manifest.json", output / "report.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Frozen calibration manifest differs; use a new run")
    else:
        _write_json(manifest_path, manifest)
    if report_path.exists():
        stored = json.loads(report_path.read_text(encoding="utf-8"))
        if stored["manifest_sha256"] != fingerprint(manifest):
            raise ValueError("Stored report manifest identity differs")
        return stored
    if client_factory is None:
        return {"prepared": True, "manifest_sha256": fingerprint(manifest)}
    lock = output / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        ledger_path = output / "requests.jsonl"
        records = RequestLedger(ledger_path).records()
        stop_reason = "preexisting_request_evidence" if records else None
        client, attempts = None, 0
        for job in manifest["jobs"]:
            if stop_reason:
                break
            spent = _cost(*(
                sum(record.get("usage", {}).get(field, 0) for record in records if record.get("usage"))
                for field in ("prompt_tokens", "completion_tokens")
            ))
            planned_reserve = _cost(job["planning_input_units"], job["parameters"]["max_tokens"])
            if spent + planned_reserve > Decimal(manifest["proxy_stop_usd"]):
                stop_reason = "proxy_budget_stop"
                break
            if client is None:
                try:
                    client = client_factory(ledger_path)
                except Exception:
                    stop_reason = "client_initialization_failed"
                    break
                expected = manifest["endpoint"]
                if (client.provider, client.model, client.base_url) != (expected["provider"], expected["model"], expected["base_url"]):
                    stop_reason = "client_endpoint_mismatch"
                    break
            operation_id = fingerprint([fingerprint(manifest), job["job_id"]])
            client.request_context = {"calibration_job_id": job["job_id"], "source_task_id": job["task_id"]}
            error = None
            attempts += 1
            try:
                client.chat_single_attempt(job["messages"], **job["parameters"], operation_id=operation_id, stage="code")
            except Exception as exception:
                error = {"error_type": type(exception).__name__, "http_status": getattr(exception, "status_code", None)}
            records = client.get_request_records()
            matches = [record for record in records if record["operation_id"] == operation_id]
            if len(matches) != 1:
                stop_reason = "missing_or_duplicate_request_evidence"
                break
            record = matches[0]
            if error and record.get("status") != "truncated":
                client.request_ledger.finish(record["request_id"], **error)
                records = client.get_request_records()
                record = next(item for item in records if item["request_id"] == record["request_id"])
            stop_reason = _record_reason(record, manifest, job)
            if not stop_reason and _cost(record["usage"]["prompt_tokens"], record["usage"]["completion_tokens"]) + spent > Decimal(manifest["proxy_stop_usd"]):
                stop_reason = "observed_proxy_budget_exceeded"
        report = _report(manifest, records, stop_reason, new_attempts=attempts)
        _write_json(report_path, report)
        summary = report["summary"]
        text = f"""# 小响应真实协议标定

请求证据数：{summary["requests_with_evidence"]}；新尝试数：{attempts}。协议检查通过：{summary["protocol_checks_passed"]}。

实测prompt Token：{summary["prompt_tokens"]}；output Token：{summary["completion_tokens"]}；合计：{summary["total_tokens"]}。用量完整：{summary["usage_complete"]}。

按历史代理价计费：{("$" + summary["proxy_cost_usd"]) if summary["usage_complete"] else "未知"}；已知用量小计${summary["known_usage_proxy_cost_usd"]}。这不是网关实付账单；美元/人民币实际账单均未知。停机原因：{stop_reason or "无"}。

温度0、输出上限32、最多6次串行请求、无自动重试。低输出上限导致的截断计入费用，不作为代码正确输出；没有评分、未运行984请求全量，也不据少量观测证明全量Token上界。

| 内容 | 入口 |
|---|---|
| 冻结标定协议 | [manifest.json](manifest.json) |
| 实测请求、模型、usage与耗时 | [report.json](report.json) |
| 发送前pending及终态事件 | [requests.jsonl](requests.jsonl) |
"""
        with (output / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(text)
        return report
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "results/prompt_contrast_preparation_20261028/draft.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    endpoint = endpoint_metadata(ROOT / DEFAULT_PRO_ENV)
    manifest = freeze_probe(args.draft, endpoint)

    def live_factory(path):
        if os.environ.get("VIB_PRIVATE_API_KEY_ENV") != endpoint["credential_env"]:
            raise ValueError("Set the explicit selected private key variable; no fallback")
        return configure_live_client(endpoint, path)

    report = execute_probe(manifest, args.output_dir, live_factory if args.execute else None)
    print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))
    return 0 if not args.execute or report["summary"]["protocol_checks_passed"] else 2


def configure_live_client(endpoint, ledger_path):
    """Apply calibration-specific timeout without changing old benchmark clients."""
    client = build_client(endpoint["model"], "private", env_path=str(ROOT / DEFAULT_PRO_ENV),
                          temperature=0, max_tokens=32, log=False, ledger_path=str(ledger_path))
    client._client = client.client.with_options(timeout=60, max_retries=0)
    return client


if __name__ == "__main__":
    raise SystemExit(main())
