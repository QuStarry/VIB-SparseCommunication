import copy
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from types import SimpleNamespace
import threading

import pytest

from src.experiment_session import (
    BudgetExceeded, ExperimentSession, ManifestMismatch, SessionStopped,
)
from src.llm_client import LLMClient


def manifest(count=1, budget="0.0002"):
    return {
        "version": 1, "model": "test-model", "provider": "custom",
        "base_url": "https://example.invalid/v1",
        "budget_usd": budget, "evaluation_version": "test-evaluation-v1",
        "price": {"version": "test-price-v1", "input_usd_per_million": "1",
                  "output_usd_per_million": "1", "scope": "proxy_not_invoice"},
        "jobs": [
            {"job_id": f"job/{index}", "task_id": f"task/{index}", "stage": "math",
             "messages": [{"role": "user", "content": "Compute 1+1"}],
             "parameters": {"temperature": 0, "max_tokens": 100}, "prompt_token_limit": 100}
            for index in range(count)
        ],
    }


def fake_client(outcomes=None):
    client = LLMClient(api_key="fake-secret", base_url="https://example.invalid/v1", model="test-model")
    calls = []
    outcomes = list(outcomes or [("答案: 2", 10, 5, "test-model")])

    def create(**kwargs):
        calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        content, prompt, completion, served = outcome
        return SimpleNamespace(
            id="response-id", _request_id="request-id", model=served,
            usage=(SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
                   if prompt is not None else None),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        )

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_concurrent_reservations_never_exceed_shared_budget(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest(48))

    def reserve(index):
        try:
            return session.claim(f"job/{index}")
        except (BudgetExceeded, SessionStopped):
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(48)))
    assert sum(results) == 1
    summary = session.summary()
    assert summary["committed_units"] == summary["budget_units"]
    assert summary["reserved_requests"] == 1


def test_duplicate_job_claim_does_not_reserve_twice(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    assert session.claim("job/0")
    with pytest.raises(SessionStopped, match="reserved"):
        session.claim("job/0")
    assert session.summary()["reserved_requests"] == 1


def test_pending_recovery_blocks_new_requests_without_releasing_reserve(tmp_path):
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest(2, "0.0004"))
    session.claim("job/0")
    restored = ExperimentSession(path, manifest(2, "0.0004"))
    client, calls = fake_client()
    with pytest.raises(SessionStopped):
        restored.execute("job/1", client)
    assert calls == []
    assert restored.summary()["committed_units"] == session.summary()["committed_units"]


def test_received_output_recovers_without_repeating_request(tmp_path):
    path = tmp_path / "session.db"
    client, calls = fake_client()
    session = ExperimentSession(path, manifest())
    result = session.execute("job/0", client)
    assert session.job("job/0")["state"] == "received"
    restored = ExperimentSession(path, manifest())
    assert restored.execute("job/0", client) == result
    assert len(calls) == 1
    assert restored.evaluate("job/0", lambda output: output == "答案: 2", "test-evaluation-v1") is True
    assert restored.job("job/0")["state"] == "completed"
    assert restored.summary()["known_spent_units"] == 15000


def test_completed_output_and_evaluation_are_both_reused(tmp_path):
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest())
    client, calls = fake_client()
    session.execute("job/0", client)
    session.evaluate("job/0", lambda output: True, "test-evaluation-v1")
    restored = ExperimentSession(path, manifest())
    restored.execute("job/0", client)

    def should_not_run(output):
        raise AssertionError("Evaluation was already completed")

    assert restored.evaluate("job/0", should_not_run, "test-evaluation-v1")
    assert len(calls) == 1


def test_unknown_usage_halts_all_jobs_and_retains_reservation(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest(2, "0.0004"))
    client, calls = fake_client([("答案: 2", None, None, "test-model")])
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    with pytest.raises(SessionStopped):
        session.execute("job/1", client)
    assert len(calls) == 1
    assert session.job("job/0")["state"] == "unknown"
    assert session.summary()["unknown_requests"] == 1
    assert session.summary()["held_units"] == 200000


def test_network_failure_is_not_retried_or_assumed_free(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client([TimeoutError("fake-secret")])
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    assert len(calls) == 1
    assert session.summary()["known_spent_units"] == 0
    assert session.summary()["held_units"] == 200000
    assert "fake-secret" not in str(session.job("job/0"))


def test_insufficient_budget_prevents_network_request(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest(budget="0.000199999"))
    client, calls = fake_client()
    with pytest.raises(BudgetExceeded):
        session.execute("job/0", client)
    assert calls == []


@pytest.mark.parametrize("field", ["model", "budget_usd", "evaluation_version"])
def test_manifest_drift_is_rejected(tmp_path, field):
    original = manifest()
    ExperimentSession(tmp_path / "session.db", original)
    changed = copy.deepcopy(original)
    changed[field] = "0.0003" if field == "budget_usd" else "changed"
    with pytest.raises(ManifestMismatch):
        ExperimentSession(tmp_path / "session.db", changed)


def test_duplicate_job_ids_are_rejected(tmp_path):
    contract = manifest(2)
    contract["jobs"][1]["job_id"] = "job/0"
    with pytest.raises(ValueError, match="Duplicate"):
        ExperimentSession(tmp_path / "session.db", contract)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0"])
def test_invalid_prices_are_rejected(tmp_path, value):
    contract = manifest()
    contract["price"]["input_usd_per_million"] = value
    with pytest.raises(ValueError):
        ExperimentSession(tmp_path / "session.db", contract)


def test_credentials_cannot_be_frozen_into_manifest(tmp_path):
    contract = manifest()
    contract["api_key"] = "fake-secret"
    with pytest.raises(ValueError, match="credential"):
        ExperimentSession(tmp_path / "session.db", contract)


def test_wrong_endpoint_is_rejected_before_reserving_or_sending(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client()
    client.model = "other-model"
    with pytest.raises(ManifestMismatch):
        session.execute("job/0", client)
    assert not calls
    assert session.job("job/0")["state"] == "planned"


def test_served_model_drift_is_recorded_and_stops_requests(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client([("答案: 2", 10, 5, "other-model")])
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    assert len(calls) == 1
    assert session.summary()["known_spent_units"] == 15000
    assert session.job("job/0")["response"]["served_model"] == "other-model"


def test_provider_bound_violation_preserves_actual_overspend(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, _ = fake_client([("答案: 2", 1000, 5, "test-model")])
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    summary = session.summary()
    assert summary["known_spent_units"] == 1005000
    assert summary["committed_units"] > summary["budget_units"]
    assert summary["stop_reason"] == "provider_bound_violation"


def test_evaluation_failure_never_regenerates_received_output(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client()
    session.execute("job/0", client)

    def broken_evaluator(output):
        raise ValueError("Evaluation failed")

    with pytest.raises(SessionStopped):
        session.evaluate("job/0", broken_evaluator, "test-evaluation-v1")
    assert session.execute("job/0", client)["content"] == "答案: 2"
    assert session.evaluate("job/0", lambda output: True, "test-evaluation-v1")
    assert len(calls) == 1


def test_response_tampering_is_rejected_before_cache_reuse(tmp_path):
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest())
    client, calls = fake_client()
    session.execute("job/0", client)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE jobs SET response_json = ? WHERE job_id = ?", ('{"content":"tampered"}', "job/0"))
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    assert len(calls) == 1


def test_evaluation_protocol_drift_is_rejected_without_running_evaluator(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, _ = fake_client()
    session.execute("job/0", client)
    with pytest.raises(ManifestMismatch):
        session.evaluate("job/0", lambda output: True, "changed-version")


@pytest.mark.parametrize("workers", [2, 8, 16])
def test_independent_connections_share_atomic_budget(tmp_path, workers):
    path = tmp_path / "session.db"
    contract = manifest(48)
    sessions = [ExperimentSession(path, contract) for _ in range(workers)]

    def reserve(index):
        try:
            return sessions[index % workers].claim(f"job/{index}")
        except (BudgetExceeded, SessionStopped):
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        accepted = list(executor.map(reserve, range(48)))
    assert sum(accepted) == 1
    assert sessions[0].summary()["committed_units"] == 200000


def test_transport_is_called_only_after_durable_reservation(tmp_path):
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest())
    client, calls = fake_client()
    original = client.chat_single_attempt

    def inspect_reservation(*args, **kwargs):
        with sqlite3.connect(path) as connection:
            state, units = connection.execute(
                "SELECT state, reserved_units FROM jobs WHERE job_id = 'job/0'"
            ).fetchone()
        assert state == "reserved"
        assert units == 200000
        return original(*args, **kwargs)

    client.chat_single_attempt = inspect_reservation
    session.execute("job/0", client)
    assert len(calls) == 1


def test_crash_between_send_and_response_blocks_paid_resume(tmp_path):
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest())
    client, _ = fake_client()

    def simulate_interruption(*args, **kwargs):
        raise KeyboardInterrupt()

    client.chat_single_attempt = simulate_interruption
    with pytest.raises(KeyboardInterrupt):
        session.execute("job/0", client)
    assert session.job("job/0")["state"] == "reserved"
    restored = ExperimentSession(path, manifest())
    with pytest.raises(SessionStopped):
        restored.execute("job/0", client)
    assert restored.summary()["held_units"] == 200000


def test_duplicate_workers_do_not_send_same_job_twice(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client()
    entered = threading.Event()
    release = threading.Event()
    original = client.chat_single_attempt

    def pause_transport(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    client.chat_single_attempt = pause_transport
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(session.execute, "job/0", client)
        assert entered.wait(5)
        try:
            with pytest.raises(SessionStopped):
                session.execute("job/0", client)
        finally:
            release.set()
        assert first.result()["content"] == "答案: 2"
    assert len(calls) == 1


def test_truncated_billed_response_is_retained_without_retry(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, calls = fake_client()
    original_create = client._client.chat.completions.create

    def truncated(**kwargs):
        result = original_create(**kwargs)
        result.choices[0].finish_reason = "length"
        return result

    client._client.chat.completions.create = truncated
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    assert len(calls) == 1
    assert session.summary()["known_spent_units"] == 15000
    assert session.job("job/0")["response"]["request_evidence"]["status"] == "truncated"


def test_quota_failure_halts_without_retrying_or_logging_secret(tmp_path):
    class QuotaFailure(RuntimeError):
        status_code = 429

    session = ExperimentSession(tmp_path / "session.db", manifest(2, "0.0004"))
    client, calls = fake_client([QuotaFailure("fake-secret")])
    with pytest.raises(SessionStopped):
        session.execute("job/0", client)
    with pytest.raises(SessionStopped):
        session.execute("job/1", client)
    assert len(calls) == 1
    assert session.summary()["held_units"] == 200000
    assert session.job("job/0")["response"]["error_type"] == "QuotaFailure"


def test_hidden_evaluation_fields_are_rejected_from_inference_jobs(tmp_path):
    contract = manifest()
    contract["jobs"][0]["test_cases"] = ["hidden check"]
    with pytest.raises(ValueError, match="Evaluation material"):
        ExperimentSession(tmp_path / "session.db", contract)


def test_caller_cannot_mutate_frozen_manifest_after_initialization(tmp_path):
    contract = manifest()
    session = ExperimentSession(tmp_path / "session.db", contract)
    contract["jobs"][0]["parameters"]["temperature"] = 1.3
    client, calls = fake_client()
    session.execute("job/0", client)
    assert calls[0]["temperature"] == 0


def test_non_boolean_evaluation_is_not_accepted(tmp_path):
    session = ExperimentSession(tmp_path / "session.db", manifest())
    client, _ = fake_client()
    session.execute("job/0", client)
    with pytest.raises(SessionStopped):
        session.evaluate("job/0", lambda output: "True", "test-evaluation-v1")
    assert session.job("job/0")["state"] == "received"


def test_budget_rounds_reservations_up_not_down(tmp_path):
    contract = manifest()
    contract["price"]["input_usd_per_million"] = "0.0000001"
    contract["price"]["output_usd_per_million"] = "0.0000001"
    session = ExperimentSession(tmp_path / "session.db", contract)
    session.claim("job/0")
    assert session.summary()["held_units"] == 1


def test_changed_session_implementation_blocks_resume(tmp_path):
    path = tmp_path / "session.db"
    ExperimentSession(path, manifest())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value = 'changed' WHERE name = 'implementation_sha256'")
    with pytest.raises(ManifestMismatch, match="implementation"):
        ExperimentSession(path, manifest())


def test_same_job_id_in_different_manifests_has_independent_request_identity(tmp_path):
    first = ExperimentSession(tmp_path / "first.db", manifest())
    second = ExperimentSession(tmp_path / "second.db", manifest(budget="0.0004"))
    client, calls = fake_client()
    result = first.execute("job/0", client)
    assert result["manifest_sha256"] == first.summary()["manifest_sha256"]

    def fail_before_request(*args, **kwargs):
        assert kwargs["operation_id"] != result["operation_id"]
        raise RuntimeError("Transport initialization failed")

    client.chat_single_attempt = fail_before_request
    with pytest.raises(SessionStopped):
        second.execute("job/0", client)
    assert second.summary()["held_units"] == 200000
    assert second.summary()["known_spent_units"] == 0
    assert len(calls) == 1
