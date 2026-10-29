import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.run_prompt_contrast_pilot import execute_pilot, freeze_pilot
from src.llm_client import LLMClient, ResolvedModel
from src.request_ledger import RequestLedger, fingerprint


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/prompt_contrast_preparation_20261028/draft.json"
ENDPOINT = {"provider": "private", "model": "deepseek-v4-flash-0817",
            "base_url": "https://example.invalid/v1", "credential_env": "PRIVATE_API_KEY_1",
            "credential_present": True}


@pytest.fixture(autouse=True)
def fake_disk_flush(monkeypatch):
    # Fake transports test control flow, not filesystem durability benchmarks.
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def factory(calls, *, finish="stop", missing_usage=False, failure=None,
            model=None, prompt_tokens=100, completion_tokens=20, interrupt_after=None):
    def create(path):
        client = LLMClient(_resolved=ResolvedModel(
            "private", ENDPOINT["model"], ENDPOINT["base_url"], "fake-secret", "fake"),
            ledger_path=path)

        def transport(**kwargs):
            calls.append(kwargs)
            if interrupt_after is not None and len(calls) > interrupt_after:
                raise KeyboardInterrupt()
            if failure:
                exception = RuntimeError("fake-secret")
                exception.status_code = failure
                raise exception
            return SimpleNamespace(
                id=f"response/{len(calls)}", _request_id=f"request/{len(calls)}",
                model=model or ENDPOINT["model"],
                usage=None if missing_usage else SimpleNamespace(
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
                    content="def candidate():\n    return 1"))],
            )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))
        return client
    return create


def test_freezes_24_tasks_all_pairs_without_mutation():
    before = SOURCE.read_bytes()
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    assert manifest == freeze_pilot(SOURCE, ENDPOINT)
    assert len(manifest["jobs"]) == 144
    assert len(set(job["task_id"] for job in manifest["jobs"])) == 24
    assert all(job["parameters"] == {"temperature": 0, "max_tokens": 1024}
               for job in manifest["jobs"])
    assert all("test_cases" not in job and "passed" not in job for job in manifest["jobs"])
    assert SOURCE.read_bytes() == before


def test_complete_outputs_persist_before_scoring_and_cache_without_client(tmp_path):
    calls = []
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    first = execute_pilot(manifest, output, factory(calls))
    assert len(calls) == 144
    assert first["summary"]["run_complete"]
    assert first["summary"]["total_tokens"] == 17280
    assert len(list((output / "responses").glob("*.json"))) == 144
    assert "accuracy" not in first["summary"]
    assert not (output / "analysis.json").exists()
    before = (output / "requests.jsonl").read_bytes()
    assert execute_pilot(manifest, output, lambda path: pytest.fail("cached")) == first
    assert before == (output / "requests.jsonl").read_bytes()


def test_truncated_outputs_are_retained_billed_and_not_regenerated(tmp_path):
    calls = []
    report = execute_pilot(freeze_pilot(SOURCE, ENDPOINT), tmp_path / "pilot",
                           factory(calls, finish="length", completion_tokens=1024))
    assert len(calls) == 144
    assert report["summary"]["run_complete"]
    assert report["summary"]["truncated_responses"] == 144
    assert report["summary"]["completion_tokens"] == 147456
    assert all(row["status"] == "truncated" for row in report["requests"])


@pytest.mark.parametrize("kwargs,reason", [
    ({"missing_usage": True}, "unknown_usage"),
    ({"failure": 429}, "http_429"),
    ({"model": "different"}, "served_model_mismatch"),
    ({"completion_tokens": 1025}, "completion_cap_violated"),
    ({"prompt_tokens": 4000000}, "observed_proxy_budget_exceeded"),
])
def test_fail_fast_no_retry_preserves_evidence(tmp_path, kwargs, reason):
    calls = []
    report = execute_pilot(freeze_pilot(SOURCE, ENDPOINT), tmp_path / "pilot",
                           factory(calls, **kwargs))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == reason
    assert not report["summary"]["run_complete"]
    assert "fake-secret" not in str(report)
    if kwargs.get("missing_usage") or kwargs.get("failure"):
        assert report["summary"]["proxy_cost_usd"] is None


def test_pending_on_resume_never_resends(tmp_path):
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    execute_pilot(manifest, output, None)
    ledger = RequestLedger(output / "requests.jsonl")
    job = manifest["jobs"][0]
    ledger.start({"parameters": job["parameters"], "prompt_sha256": fingerprint(job["messages"])},
                 operation_id=fingerprint([fingerprint(manifest), job["job_id"]]), attempt=1)
    report = execute_pilot(manifest, output, lambda path: pytest.fail("pending"))
    assert report["summary"]["stop_reason"] == "unknown_usage"


def test_settled_responses_resume_only_unsent_jobs(tmp_path):
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    calls = []
    execute_pilot(manifest, output, factory(calls))
    (output / "generation_report.json").unlink()
    resumed = execute_pilot(manifest, output, lambda path: pytest.fail("all settled"))
    assert resumed["summary"]["run_complete"]
    assert resumed["summary"]["new_attempts"] == 0


def test_interrupted_transport_stops_on_unresolved_evidence(tmp_path):
    calls = []
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    with pytest.raises(KeyboardInterrupt):
        execute_pilot(manifest, output, factory(calls, interrupt_after=2))
    report = execute_pilot(manifest, output, lambda path: pytest.fail("unresolved"))
    assert report["summary"]["requests_with_evidence"] == 3
    assert report["summary"]["stop_reason"] == "unknown_usage"


def test_settled_partial_run_continues_without_resending_received_outputs(tmp_path):
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    first_calls, next_calls = [], []

    def interrupted_factory(path):
        client = factory(first_calls)(path)
        finish = client.request_ledger.finish

        def interrupted_finish(request_id, **fields):
            finish(request_id, **fields)
            if len(first_calls) == 2:
                raise KeyboardInterrupt()

        client.request_ledger.finish = interrupted_finish
        return client

    with pytest.raises(KeyboardInterrupt):
        execute_pilot(manifest, output, interrupted_factory)
    resumed = execute_pilot(manifest, output, factory(next_calls))
    assert len(first_calls) == 2
    assert len(next_calls) == 142
    assert resumed["summary"]["run_complete"]
    assert resumed["summary"]["new_attempts"] == 142


def test_response_tampering_and_manifest_drift_rejected_before_transport(tmp_path):
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    execute_pilot(manifest, output, factory([]))
    (output / "generation_report.json").unlink()
    raw_path = next((output / "responses").glob("*.json"))
    raw = json.loads(raw_path.read_text())
    raw["content"] = "changed"
    raw_path.write_text(json.dumps(raw))
    report = execute_pilot(manifest, output, lambda path: pytest.fail("tampering"))
    assert report["summary"]["stop_reason"] == "response_artifact_mismatch"
    changed = copy.deepcopy(manifest)
    changed["jobs"][0]["parameters"]["temperature"] = 1
    with pytest.raises(ValueError):
        execute_pilot(changed, output, lambda path: pytest.fail("drift"))


def test_explicit_selected_key_required():
    with pytest.raises(ValueError):
        freeze_pilot(SOURCE, {**ENDPOINT, "credential_present": False})


def test_missing_request_evidence_is_not_zero_cost(tmp_path):
    def broken_factory(path):
        client = factory([])(path)
        client.chat_single_attempt = lambda *args, **kwargs: {"content": "untracked"}
        return client
    report = execute_pilot(freeze_pilot(SOURCE, ENDPOINT), tmp_path / "pilot", broken_factory)
    assert report["summary"]["stop_reason"] == "missing_or_duplicate_request_evidence"
    assert report["summary"]["new_attempts"] == 1
    assert report["summary"]["proxy_cost_usd"] is None
    assert not report["summary"]["usage_complete"]


def test_live_client_configuration(monkeypatch, tmp_path):
    from experiments import run_prompt_contrast_pilot as pilot
    monkeypatch.setenv("VIB_PRIVATE_API_KEY_ENV", "PRIVATE_API_KEY_1")
    options = []
    parameters = []
    client = SimpleNamespace(client=SimpleNamespace(
        with_options=lambda **kwargs: options.append(kwargs) or "configured"))
    monkeypatch.setattr(pilot, "build_client",
                        lambda *args, **kwargs: parameters.append(kwargs) or client)
    assert pilot.configure_live_client(ENDPOINT, tmp_path / "ledger") is client
    assert options == [{"timeout": 60, "max_retries": 0}]
    assert parameters[0]["max_tokens"] == 1024


def test_durable_pending_precedes_transport_and_response_precedes_terminal(tmp_path, monkeypatch):
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    flushes = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: flushes.append(descriptor))
    calls = []

    def create(path):
        client = factory(calls)(path)
        transport = client.client.chat.completions.create

        def checked_transport(**kwargs):
            assert flushes
            assert RequestLedger(path).records()[-1]["status"] == "pending"
            return transport(**kwargs)

        client._client.chat.completions.create = checked_transport
        original_finish = client.request_ledger.finish

        def checked_finish(request_id, **fields):
            if fields.get("status") == "completed":
                raw_path = output / "responses" / f"{fingerprint(manifest['jobs'][len(calls) - 1]['job_id'])}.json"
                assert raw_path.exists()
            original_finish(request_id, **fields)

        client.request_ledger.finish = checked_finish
        return client

    assert execute_pilot(manifest, output, create)["summary"]["run_complete"]
    assert len(flushes) >= 144 * 3
