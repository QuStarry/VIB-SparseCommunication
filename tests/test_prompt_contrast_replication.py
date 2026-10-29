import copy
import json
import os
from pathlib import Path

import pytest

from experiments.run_prompt_contrast_replication import freeze_replication, execute_replication
from experiments.run_prompt_contrast_pilot import execute_pilot, freeze_pilot
from src.request_ledger import RequestLedger, fingerprint
from tests.test_prompt_contrast_pilot import SOURCE, ENDPOINT, factory


LIVE_PRIOR = SOURCE.parents[1] / "prompt_contrast_pilot_20261028/manifest.json"
PRIOR = LIVE_PRIOR


@pytest.fixture(autouse=True)
def fake_flush(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)
    prior = tmp_path / "fake_prior.json"
    prior.write_text(json.dumps(freeze_pilot(SOURCE, ENDPOINT)))
    monkeypatch.setattr(__import__(__name__, fromlist=["PRIOR"]), "PRIOR", prior)


def freeze():
    return freeze_replication(SOURCE, PRIOR, ENDPOINT)


def test_freezes_48_unseen_pilot_tasks_without_labels_or_message_changes():
    prior = json.loads(PRIOR.read_text())
    manifest = freeze()
    assert manifest == freeze()
    assert len(manifest["jobs"]) == 288
    assert len(set(manifest["task_ids"])) == 48
    assert not set(manifest["task_ids"]) & set(prior["task_ids"])
    source_jobs = {job["job_id"]: job for job in json.loads(SOURCE.read_text())["jobs"]}
    for job in manifest["jobs"]:
        assert job["messages_sha256"] == source_jobs[job["job_id"]]["messages_sha256"]
        assert job["parameters"] == {"temperature": 0, "max_tokens": 1024}
        assert "passed" not in job and "test_cases" not in job
    assert manifest["analysis"]["primary"] == ["legacy_compact", "current_arbitration"]


def test_complete_replication_no_hidden_scoring_and_cached_no_client(tmp_path):
    output, calls = tmp_path / "run", []
    manifest = freeze()
    result = execute_replication(manifest, output, factory(calls))
    assert len(calls) == 288
    assert result["summary"]["run_complete"]
    assert len(list((output / "responses").glob("*.json"))) == 288
    assert not (output / "analysis.json").exists()
    before = (output / "requests.jsonl").read_bytes()
    assert execute_replication(manifest, output, lambda path: pytest.fail("cached")) == result
    assert before == (output / "requests.jsonl").read_bytes()


@pytest.mark.parametrize("kwargs,reason", [
    ({"missing_usage": True}, "unknown_usage"),
    ({"failure": 429}, "http_429"),
    ({"model": "other"}, "served_model_mismatch"),
    ({"prompt_tokens": 4000000}, "observed_proxy_budget_exceeded"),
    ({"completion_tokens": 1025}, "completion_cap_violated"),
])
def test_fail_fast_single_attempt_preserves_accounting(tmp_path, kwargs, reason):
    calls = []
    result = execute_replication(freeze(), tmp_path / "run", factory(calls, **kwargs))
    assert len(calls) == 1
    assert result["summary"]["stop_reason"] == reason
    if kwargs.get("missing_usage") or kwargs.get("failure"):
        assert result["summary"]["proxy_cost_usd"] is None
    assert "fake-secret" not in str(result)


def test_truncated_outputs_are_billed_once_without_repair(tmp_path):
    calls = []
    result = execute_replication(freeze(), tmp_path / "run",
                                 factory(calls, finish="length", completion_tokens=1024))
    assert result["summary"]["run_complete"]
    assert len(calls) == 288
    assert result["summary"]["truncated_responses"] == 288


def test_partial_received_resume_only_unsent(tmp_path):
    manifest, output = freeze(), tmp_path / "run"
    calls, resumed_calls = [], []

    def interrupted(path):
        client = factory(calls)(path)
        finish = client.request_ledger.finish

        def interrupt_after_received(request_id, **fields):
            finish(request_id, **fields)
            if len(calls) == 2:
                raise KeyboardInterrupt()
        client.request_ledger.finish = interrupt_after_received
        return client

    with pytest.raises(KeyboardInterrupt):
        execute_replication(manifest, output, interrupted)
    result = execute_replication(manifest, output, factory(resumed_calls))
    assert len(resumed_calls) == 286
    assert result["summary"]["run_complete"]


def test_pending_unknown_blocks_resume(tmp_path):
    manifest, output = freeze(), tmp_path / "run"
    execute_replication(manifest, output, None)
    job = manifest["jobs"][0]
    RequestLedger(output / "requests.jsonl").start(
        {"parameters": job["parameters"], "prompt_sha256": fingerprint(job["messages"])},
        operation_id=fingerprint([fingerprint(manifest), job["job_id"]]), attempt=1)
    result = execute_replication(manifest, output, lambda path: pytest.fail("pending"))
    assert result["summary"]["stop_reason"] == "unknown_usage"


def test_mutated_manifest_rejected_without_transport(tmp_path):
    manifest = copy.deepcopy(freeze())
    manifest["jobs"][0]["parameters"]["temperature"] = 1
    with pytest.raises(ValueError):
        execute_replication(manifest, tmp_path / "run", lambda path: pytest.fail("drift"))


def test_response_tampering_rejected_on_cached_recovery(tmp_path):
    manifest, output = freeze(), tmp_path / "run"
    execute_replication(manifest, output, factory([]))
    raw = next((output / "responses").glob("*.json"))
    value = json.loads(raw.read_text())
    value["content"] = "changed"
    raw.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        execute_replication(manifest, output, lambda path: pytest.fail("modified"))


def test_first_pilot_still_reuses_existing_frozen_evidence():
    output = LIVE_PRIOR.parent
    before = (output / "requests.jsonl").read_bytes()
    result = execute_pilot(json.loads(LIVE_PRIOR.read_text()), output, lambda path: pytest.fail("old"))
    assert result["summary"]["run_complete"]
    assert before == (output / "requests.jsonl").read_bytes()


def test_source_sampling_does_not_read_first_pilot_labels(tmp_path):
    prior = tmp_path / "manifest.json"
    prior.write_bytes(PRIOR.read_bytes())
    (tmp_path / "analysis.json").write_text("invalid labels deliberately not readable")
    result = freeze_replication(SOURCE, prior, ENDPOINT)
    assert result["task_ids"] == freeze()["task_ids"]


def test_lock_prevents_second_active_run(tmp_path):
    output = tmp_path / "run"
    execute_replication(freeze(), output, None)
    (output / ".run.lock").touch()
    with pytest.raises(FileExistsError):
        execute_replication(freeze(), output, lambda path: pytest.fail("locked"))


def test_missing_evidence_after_received_resume_remains_unknown(tmp_path):
    manifest, output, calls = freeze(), tmp_path / "run", []

    def interrupt(path):
        client = factory(calls)(path)
        finish = client.request_ledger.finish

        def settled_then_interrupt(request_id, **fields):
            finish(request_id, **fields)
            raise KeyboardInterrupt()
        client.request_ledger.finish = settled_then_interrupt
        return client
    with pytest.raises(KeyboardInterrupt):
        execute_replication(manifest, output, interrupt)

    def missing(path):
        client = factory([])(path)
        client.chat_single_attempt = lambda *args, **kwargs: {}
        return client
    result = execute_replication(manifest, output, missing)
    assert result["summary"]["requests_with_evidence"] == 1
    assert result["summary"]["proxy_cost_usd"] is None
    assert not result["summary"]["usage_complete"]
    assert execute_replication(manifest, output, lambda path: pytest.fail("cached unknown")) == result


def test_proxy_reserve_blocks_next_request_before_dispatch(monkeypatch, tmp_path):
    from src import exploratory_prompt_session as session
    from decimal import Decimal
    calls = []
    actual_cost = session._cost

    def large_reserve(prompt, completion):
        return Decimal(2) if completion == 1024 else actual_cost(prompt, completion)
    monkeypatch.setattr(session, "_cost", large_reserve)
    result = execute_replication(freeze(), tmp_path / "run", factory(calls))
    assert calls == []
    assert result["summary"]["stop_reason"] == "proxy_budget_stop"


def test_initializer_failure_has_no_sent_attempt(tmp_path):
    def broken(path):
        raise RuntimeError("fake-secret")
    result = execute_replication(freeze(), tmp_path / "run", broken)
    assert result["summary"]["new_attempts"] == 0
    assert result["summary"]["stop_reason"] == "client_initialization_failed"
    assert "fake-secret" not in str(result)
