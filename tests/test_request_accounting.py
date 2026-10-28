import copy
import json
from types import SimpleNamespace
import pytest

from src.llm_client import LLMClient


def response(content="answer", usage=True):
    return SimpleNamespace(
        id="response-123", _request_id="provider-request-456", model="served-model",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7) if usage else None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def fake_client(monkeypatch, outcomes, **kwargs):
    client = LLMClient(api_key="secret-test-only", base_url="https://example.invalid/v1", **kwargs)
    calls = []

    def create(**request):
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("src.llm_client.time.sleep", lambda delay: None)
    return client, calls


def test_successful_request_binds_response_usage_and_parameters(monkeypatch):
    client, _ = fake_client(monkeypatch, [response()], temperature=0, max_tokens=77)
    result = client.chat([{"role": "user", "content": "question"}])
    record = client.get_request_records()[0]
    assert record["status"] == "completed"
    assert record["usage_status"] == "known"
    assert record["response_id"] == "response-123"
    assert record["provider_request_id"] == "provider-request-456"
    assert record["served_model"] == "served-model"
    assert record["parameters"] == {"temperature": 0, "max_tokens": 77}
    assert len(record["prompt_sha256"]) == len(record["output_sha256"]) == 64
    assert record["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert result["request_id"] == record["request_id"]
    assert client.get_token_stats()["usage_complete"] is True


def test_missing_usage_is_unknown_not_zero_cost(monkeypatch):
    client, _ = fake_client(monkeypatch, [response(usage=False)])
    result = client.chat([{"role": "user", "content": "question"}])
    assert result["usage_status"] == "unknown"
    stats = client.get_token_stats()
    assert stats["usage_complete"] is False
    assert stats["unknown_usage_requests"] == 1
    assert client.get_request_records()[0]["usage"] is None


def test_empty_billed_response_is_counted_before_retry(monkeypatch):
    client, _ = fake_client(monkeypatch, [response(content=None), response()])
    client.chat([{"role": "user", "content": "question"}])
    records = client.get_request_records()
    assert [record["status"] for record in records] == ["invalid_response", "completed"]
    assert records[0]["operation_id"] == records[1]["operation_id"]
    assert client.get_token_stats()["total_tokens"] == 36
    assert client.get_token_stats()["usage_complete"] is True


def test_failed_attempt_preserves_unknown_usage_and_redacts_error(monkeypatch, capsys):
    client, _ = fake_client(monkeypatch, [RuntimeError("secret-test-only"), response()])
    client.chat([{"role": "user", "content": "question"}])
    assert [record["usage_status"] for record in client.get_request_records()] == ["unknown", "known"]
    assert client.get_token_stats()["total_tokens"] == 18
    assert client.get_token_stats()["usage_complete"] is False
    assert "secret-test-only" not in capsys.readouterr().out
    assert "secret-test-only" not in json.dumps(client.get_request_records())


def test_pending_event_is_durable_before_transport(monkeypatch, tmp_path):
    path = tmp_path / "requests.jsonl"
    client, _ = fake_client(monkeypatch, [response()], ledger_path=str(path))
    original_create = client._client.chat.completions.create

    def inspect_pending(**kwargs):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events[-1]["status"] == "pending"
        assert events[-1]["usage_status"] == "unknown"
        return original_create(**kwargs)

    client._client.chat.completions.create = inspect_pending
    client.chat([{"role": "user", "content": "question"}])
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["status"] for event in events] == ["pending", "completed"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert "secret-test-only" not in path.read_text()


def test_recovered_pending_request_stays_unknown_without_resending(tmp_path):
    from src.request_ledger import RequestLedger

    path = tmp_path / "requests.jsonl"
    ledger = RequestLedger(str(path))
    ledger.start({"prompt_sha256": "hash", "parameters": {}}, operation_id="operation", attempt=1)
    restored = RequestLedger(str(path))
    assert restored.records()[0]["status"] == "pending"
    assert restored.summary()["unknown_usage_requests"] == 1


def test_reset_stats_cannot_erase_request_evidence(monkeypatch):
    client, _ = fake_client(monkeypatch, [response()])
    client.chat([{"role": "user", "content": "question"}])
    client.reset_stats()
    assert client.get_token_stats()["total_tokens"] == 0
    assert len(client.get_request_records()) == 1


def test_refresh_preserves_history_counters_and_marks_combined_cost_unknown(monkeypatch):
    from experiments import targeted_refresh_synthesis as refresh

    client, _ = fake_client(monkeypatch, [response(content="def f(): return 1")])
    monkeypatch.setattr(refresh, "_evaluate_code", lambda *args: True)
    sample = refresh.mb.TaskSample(task_id="HumanEval/test", dataset_name="HumanEval", prompt="def f(): pass",
                                   reference_answer="", entry_point="f", test_cases=["check"])
    original = {"output": "def f(): return 0", "passed": False, "history": ["candidate"],
                "token_stats": {"total_tokens": 100}}
    before = copy.deepcopy(original)
    result = refresh._process_variant(original, sample, client, "full")
    assert original == before
    assert result["token_stats"] == before["token_stats"]
    accounting = result["refresh_accounting"]
    assert accounting["combined_cost_status"] == "unknown"
    assert accounting["incremental_usage"]["total_tokens"] == 18
    assert len(accounting["parent_output_sha256"]) == 64
    assert len(accounting["requests"]) == 1
    assert accounting["output_sha256"] != accounting["parent_output_sha256"]
    # Completed refreshes reuse their linked output, not another synthesis.
    again = refresh._process_variant(result, sample, client, "full")
    assert again == result
    assert len(client.get_request_records()) == 1


def test_discarded_probe_cost_is_not_lost():
    from experiments.run_multi_benchmark import MultiDatasetHarness

    def variant(tokens, output):
        return {"output": output, "history": [output], "request_records": [{"output": output}],
                "token_stats": {"total_tokens": tokens, "total_calls": 1, "total_attempts": 1,
                                "total_prompt_tokens": tokens, "total_completion_tokens": 0,
                                "known_usage_requests": 1, "unknown_usage_requests": 0,
                                "usage_complete": True}}

    selected = variant(20, "selected")
    probe = variant(10, "discarded")
    result = MultiDatasetHarness._include_discarded_probe(selected, probe)
    assert result["output"] == "selected"
    assert result["history"] == ["selected"]
    assert result["token_stats"]["total_tokens"] == 30
    assert result["selected_token_stats"]["total_tokens"] == 20
    assert result["discarded_attempts"] == [probe]
    assert len(result["request_records"]) == 2
    assert selected["token_stats"]["total_tokens"] == 20


def test_ledger_failure_after_response_never_retries_transport(monkeypatch):
    from src.request_ledger import LedgerError

    client, calls = fake_client(monkeypatch, [response()])

    def fail_persistence(*args, **kwargs):
        raise LedgerError("disk full")

    monkeypatch.setattr(client.request_ledger, "finish", fail_persistence)
    with pytest.raises(LedgerError):
        client.chat([{"role": "user", "content": "question"}])
    assert len(calls) == 1
    assert client.get_request_records()[0]["status"] == "pending"


def test_sdk_hidden_retries_are_disabled(monkeypatch):
    import openai

    captured = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.append(kwargs) or object())
    client = LLMClient(api_key="test-key", base_url="https://example.invalid/v1")
    assert client.client is not None
    assert captured[0]["max_retries"] == 0


def test_truncated_response_is_recorded_and_not_silently_accepted(monkeypatch):
    truncated = response(content="partial")
    truncated.choices[0].finish_reason = "length"
    client, calls = fake_client(monkeypatch, [truncated, response()])
    assert client.chat([{"role": "user", "content": "question"}])["content"] == "answer"
    assert len(calls) == 2
    assert client.get_request_records()[0]["status"] == "truncated"
    assert client.get_token_stats()["total_tokens"] == 36


def test_invalid_usage_stays_unknown(monkeypatch):
    invalid = response()
    invalid.usage.prompt_tokens = -1
    client, _ = fake_client(monkeypatch, [invalid])
    client.chat([{"role": "user", "content": "question"}])
    assert client.get_request_records()[0]["usage"] is None
    assert client.get_token_stats()["unknown_usage_requests"] == 1


def test_endpoint_description_does_not_expose_credentials(monkeypatch):
    client, _ = fake_client(monkeypatch, [response()])
    assert "secret-test-only" not in repr(client.resolved)
    assert "secret-test-only" not in client.resolved.describe()


def test_refresh_rejects_changed_provenance_without_resending(monkeypatch):
    from experiments import targeted_refresh_synthesis as refresh

    client, calls = fake_client(monkeypatch, [response(content="def f(): return 1")])
    monkeypatch.setattr(refresh, "_evaluate_code", lambda *args: True)
    sample = refresh.mb.TaskSample(task_id="HumanEval/test", dataset_name="HumanEval",
                                  prompt="def f(): pass", reference_answer="")
    original = {"output": "old", "passed": False, "history": [], "token_stats": {"total_tokens": 10}}
    refreshed = refresh._process_variant(original, sample, client, "full")
    assert refreshed["refresh_accounting"]["matching_artifact_request_ids"]
    refreshed["history"] = ["changed history"]
    with pytest.raises(ValueError, match="provenance"):
        refresh._process_variant(refreshed, sample, client, "full")
    assert len(calls) == 1
