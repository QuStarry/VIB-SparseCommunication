import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.calibrate_prompt_protocol import freeze_probe, execute_probe
from src.llm_client import LLMClient, ResolvedModel
from src.request_ledger import RequestLedger


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/prompt_contrast_preparation_20261028/draft.json"
ENDPOINT = {"provider": "private", "model": "deepseek-v4-flash-0817",
            "base_url": "https://example.invalid/v1", "credential_env": "PRIVATE_API_KEY_1",
            "credential_present": True}


def fake_factory(calls, *, missing_usage=False, served_model=None, failure=False,
                 prompt_tokens=100, completion_tokens=32, http_status=None):
    def create(path):
        resolved = ResolvedModel("private", ENDPOINT["model"], ENDPOINT["base_url"], "fake-secret", "fake")
        client = LLMClient(_resolved=resolved, ledger_path=path)

        def transport(**kwargs):
            calls.append(kwargs)
            if failure:
                error = RuntimeError("fake-secret")
                error.status_code = http_status
                raise error
            return SimpleNamespace(
                id=f"response/{len(calls)}", _request_id=f"request/{len(calls)}",
                model=served_model or ENDPOINT["model"],
                usage=None if missing_usage else SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial code"))],
            )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))
        return client
    return create


def test_freezes_six_length_extremes_without_mutating_original():
    before = SOURCE.read_bytes()
    manifest = freeze_probe(SOURCE, ENDPOINT)
    assert len(manifest["jobs"]) == 6
    assert {job["length_selection"] for job in manifest["jobs"]} == {"shortest", "longest"}
    assert all(job["parameters"] == {"temperature": 0, "max_tokens": 32} for job in manifest["jobs"])
    assert SOURCE.read_bytes() == before
    assert manifest["scope"] == "small_protocol_calibration_not_benchmark"


def test_truncated_responses_are_billed_but_not_scored(tmp_path):
    manifest = freeze_probe(SOURCE, ENDPOINT)
    calls = []
    report = execute_probe(manifest, tmp_path / "probe", fake_factory(calls))
    assert len(calls) == 6
    assert report["summary"]["protocol_checks_passed"]
    assert report["summary"]["prompt_tokens"] == 600
    assert report["summary"]["completion_tokens"] == 192
    assert report["summary"]["truncated_responses"] == 6
    assert all(call["temperature"] == 0 and call["max_tokens"] == 32 for call in calls)
    assert "accuracy" not in report["summary"]


def test_unknown_usage_stops_without_retry(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, missing_usage=True))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "unknown_usage"
    assert not report["summary"]["usage_complete"]
    assert report["summary"]["proxy_cost_usd"] is None


def test_network_failure_stops_and_error_text_is_redacted(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, failure=True))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "unknown_usage"
    assert "fake-secret" not in str(report)


def test_returned_model_drift_stops_after_preserving_known_usage(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, served_model="other"))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "served_model_mismatch"
    assert report["summary"]["prompt_tokens"] == 100


def test_completed_report_is_reused_without_constructing_client(tmp_path):
    manifest = freeze_probe(SOURCE, ENDPOINT)
    output = tmp_path / "probe"
    calls = []
    first = execute_probe(manifest, output, fake_factory(calls))
    second = execute_probe(manifest, output, lambda path: pytest.fail("No second client"))
    assert first == second
    assert len(calls) == 6


def test_pending_ledger_on_resume_never_resends(tmp_path):
    manifest = freeze_probe(SOURCE, ENDPOINT)
    output = tmp_path / "probe"
    execute_probe(manifest, output, None)
    ledger = RequestLedger(output / "requests.jsonl")
    ledger.start({"model": ENDPOINT["model"], "parameters": {"temperature": 0, "max_tokens": 32}},
                 operation_id="interrupted", attempt=1)
    report = execute_probe(manifest, output, lambda path: pytest.fail("No unresolved retry"))
    assert report["summary"]["stop_reason"] == "preexisting_request_evidence"


def test_manifest_drift_is_rejected_without_network(tmp_path):
    manifest = freeze_probe(SOURCE, ENDPOINT)
    output = tmp_path / "probe"
    execute_probe(manifest, output, None)
    changed = copy.deepcopy(manifest)
    changed["endpoint"]["model"] = "other"
    with pytest.raises(ValueError):
        execute_probe(changed, output, lambda path: pytest.fail("No drifted client"))


def test_missing_selected_credential_is_rejected():
    with pytest.raises(ValueError):
        freeze_probe(SOURCE, {**ENDPOINT, "credential_present": False})


def test_429_stops_single_attempt_without_retry(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, failure=True, http_status=429))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "http_429"
    assert report["summary"]["proxy_cost_usd"] is None


def test_observed_over_budget_is_preserved_not_clamped(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, prompt_tokens=200000))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "observed_proxy_budget_exceeded"
    assert report["summary"]["proxy_cost_usd"] == "0.0600384"


def test_provider_output_cap_violation_stops(tmp_path):
    calls = []
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe",
                           fake_factory(calls, completion_tokens=33))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "completion_cap_violated"
    assert report["summary"]["completion_tokens"] == 33


def test_live_client_sets_explicit_sdk_timeout_and_no_retries(monkeypatch, tmp_path):
    from experiments import calibrate_prompt_protocol as calibration
    options = []
    sdk = SimpleNamespace(with_options=lambda **kwargs: options.append(kwargs) or "configured")
    client = SimpleNamespace(client=sdk)
    monkeypatch.setattr(calibration, "build_client", lambda *args, **kwargs: client)
    assert calibration.configure_live_client(ENDPOINT, tmp_path / "requests.jsonl") is client
    assert options == [{"timeout": 60, "max_retries": 0}]
    assert client._client == "configured"


def test_client_initialization_failure_does_not_invent_a_sent_request(tmp_path):
    def broken(path):
        raise RuntimeError("fake-secret")
    report = execute_probe(freeze_probe(SOURCE, ENDPOINT), tmp_path / "probe", broken)
    assert report["summary"]["new_attempts"] == 0
    assert report["summary"]["stop_reason"] == "client_initialization_failed"
    assert "fake-secret" not in str(report)
