import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from src.experiment_driver import driver_sha256, implementation_sha256, preflight, run_session
from src.experiment_session import ExperimentSession
from src.llm_client import LLMClient
from src.request_ledger import fingerprint


def contract(tmp_path, count=3):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Synthetic evidence for offline tests only.", encoding="utf-8")
    artifact = {"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    endpoint = {"provider": "custom", "model": "test-model", "base_url": "https://example.invalid/v1"}
    price = {"version": "synthetic-v1", "input_usd_per_million": "1",
             "output_usd_per_million": "1", "scope": "proxy_not_invoice"}
    jobs = [
        {"job_id": f"job/{index}", "task_id": f"task/{index}", "stage": "math",
         "messages": [{"role": "user", "content": f"Compute {index}+1"}],
         "parameters": {"temperature": 0, "max_tokens": 100}, "prompt_token_limit": 100}
        for index in range(count)
    ]
    return {
        "version": 1, **endpoint, "budget_usd": "0.001",
        "evaluation_version": "eval-v1", "price": price, "jobs": jobs,
        "preflight": {
            "driver_sha256": driver_sha256(),
            "implementation_sha256": implementation_sha256(),
            "price_evidence": {**artifact, "endpoint": endpoint, "price": price,
                               "reviewed_upper_bound": True},
            "evaluation_evidence": {**artifact, "version": "eval-v1"},
            "prompt_bounds": {
                job["job_id"]: {**artifact, "endpoint": endpoint,
                                "messages_sha256": fingerprint(job["messages"]),
                                "prompt_token_limit": 100, "max_tokens": 100,
                                "reviewed_provider_bounds": True}
                for job in jobs
            },
        },
    }


def approval(manifest):
    return {"manifest_sha256": fingerprint(manifest), "budget_usd": manifest["budget_usd"],
            "approved": True}


def factory(calls, *, missing_usage=False):
    lock = threading.Lock()

    def create(job_id):
        client = LLMClient(api_key="fake-secret", base_url="https://example.invalid/v1", model="test-model")

        def transport(**kwargs):
            with lock:
                calls.append((job_id, kwargs))
            return SimpleNamespace(
                id=f"response-{job_id}", model="test-model", _request_id=f"request-{job_id}",
                usage=None if missing_usage else SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="2"))],
            )

        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))
        return client

    return create


def evaluators(manifest):
    return {job["task_id"]: lambda output: output == "2" for job in manifest["jobs"]}


def test_preflight_quotes_all_jobs_without_creating_persistent_session(tmp_path):
    manifest = contract(tmp_path)
    report = preflight(manifest)
    assert report["ready"] is True
    assert report["reservation_units"] == 600000
    assert report["job_count"] == 3
    assert not list(tmp_path.glob("*.db"))


@pytest.mark.parametrize("field", ["price_evidence", "evaluation_evidence", "prompt_bounds"])
def test_missing_evidence_blocks_client_construction(tmp_path, field):
    manifest = contract(tmp_path)
    del manifest["preflight"][field]
    created = []
    with pytest.raises(ValueError, match="preflight"):
        run_session(manifest, tmp_path / "session.db", lambda job: created.append(job),
                    evaluators(manifest), approval=approval(manifest))
    assert created == []
    assert not (tmp_path / "session.db").exists()


def test_changed_evidence_file_is_rejected(tmp_path):
    manifest = contract(tmp_path)
    Path(manifest["preflight"]["price_evidence"]["path"]).write_text("Changed.", encoding="utf-8")
    assert preflight(manifest)["ready"] is False


@pytest.mark.parametrize("field", ["messages_sha256", "prompt_token_limit", "max_tokens", "endpoint"])
def test_prompt_evidence_must_bind_actual_request(tmp_path, field):
    manifest = contract(tmp_path)
    manifest["preflight"]["prompt_bounds"]["job/0"][field] = "changed"
    assert preflight(manifest)["ready"] is False


def test_unreviewed_provider_bounds_do_not_become_verified_by_hashing(tmp_path):
    manifest = contract(tmp_path)
    manifest["preflight"]["prompt_bounds"]["job/0"]["reviewed_provider_bounds"] = False
    assert preflight(manifest)["ready"] is False


def test_price_schedule_drift_blocks_preflight(tmp_path):
    manifest = contract(tmp_path)
    manifest["price"]["input_usd_per_million"] = "2"
    # Detach the evidence: real JSON files have no shared dictionary aliases.
    manifest["preflight"]["price_evidence"]["price"] = {
        "version": "synthetic-v1", "input_usd_per_million": "1",
        "output_usd_per_million": "1", "scope": "proxy_not_invoice",
    }
    assert preflight(manifest)["ready"] is False


def test_driver_source_drift_blocks_preflight(tmp_path):
    manifest = contract(tmp_path)
    manifest["preflight"]["driver_sha256"] = "0" * 64
    assert preflight(manifest)["ready"] is False


def test_request_implementation_drift_blocks_preflight(tmp_path):
    manifest = contract(tmp_path)
    manifest["preflight"]["implementation_sha256"]["llm_client.py"] = "0" * 64
    assert "request_implementation_mismatch" in preflight(manifest)["blockers"]


def test_budget_must_cover_entire_frozen_request_list(tmp_path):
    manifest = contract(tmp_path)
    manifest["budget_usd"] = "0.0002"
    assert "budget_cannot_cover_all_reservations" in preflight(manifest)["blockers"]


@pytest.mark.parametrize("change", ["manifest_sha256", "budget_usd", "approved"])
def test_approval_binds_manifest_and_budget_before_any_client(tmp_path, change):
    manifest = contract(tmp_path)
    accepted = approval(manifest)
    accepted[change] = False if change == "approved" else "changed"
    calls = []
    with pytest.raises(ValueError, match="approval"):
        run_session(manifest, tmp_path / "session.db", factory(calls),
                    evaluators(manifest), approval=accepted)
    assert not calls


def test_missing_evaluator_fails_before_network(tmp_path):
    manifest = contract(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="evaluators"):
        run_session(manifest, tmp_path / "session.db", factory(calls), {}, approval=approval(manifest))
    assert not calls


def test_parallel_driver_completes_and_resume_creates_no_clients(tmp_path):
    manifest = contract(tmp_path, 12)
    manifest["budget_usd"] = "0.003"
    calls = []
    path = tmp_path / "session.db"
    report = run_session(manifest, path, factory(calls), evaluators(manifest),
                         approval=approval(manifest), workers=4)
    assert report["summary"]["run_complete"]
    assert len(calls) == 12

    def forbidden(job_id):
        raise AssertionError("Completed jobs must not create a client")

    restored = run_session(manifest, path, forbidden, evaluators(manifest),
                           approval=approval(manifest), workers=4)
    assert restored["summary"]["run_complete"]
    assert len(calls) == 12


def test_received_response_resumes_only_local_evaluation(tmp_path):
    manifest = contract(tmp_path, 1)
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest)
    calls = []
    session.execute("job/0", factory(calls)("job/0"))
    report = run_session(manifest, path, lambda job: pytest.fail("No client needed"),
                         evaluators(manifest), approval=approval(manifest))
    assert report["summary"]["run_complete"]
    assert len(calls) == 1


def test_unknown_usage_stops_new_submissions(tmp_path):
    manifest = contract(tmp_path, 8)
    manifest["budget_usd"] = "0.002"
    calls = []
    report = run_session(manifest, tmp_path / "session.db", factory(calls, missing_usage=True),
                         evaluators(manifest), approval=approval(manifest))
    assert len(calls) == 1
    assert report["summary"]["held_units"] == 200000
    assert report["summary"]["planned_requests"] == 7
    assert not report["summary"]["run_complete"]
    assert report["errors"] == [{"job_id": "job/0", "error_type": "SessionStopped"}]


def test_inflight_request_is_settled_after_another_worker_fails(tmp_path, monkeypatch):
    manifest = contract(tmp_path, 4)
    barrier = threading.Barrier(2)
    stopped = threading.Event()
    calls = []
    normal = factory(calls)
    receive = ExperimentSession._receive

    def received(session, job_id, response, failure=None):
        try:
            return receive(session, job_id, response, failure)
        finally:
            if job_id == "job/0":
                stopped.set()

    monkeypatch.setattr(ExperimentSession, "_receive", received)

    def create(job_id):
        client = normal(job_id)
        transport = client._client.chat.completions.create

        def synchronized(**kwargs):
            barrier.wait(timeout=5)
            result = transport(**kwargs)
            if job_id == "job/0":
                result.usage = None
            else:
                assert stopped.wait(timeout=5)
            return result

        client._client.chat.completions.create = synchronized
        return client

    report = run_session(manifest, tmp_path / "session.db", create, evaluators(manifest),
                         approval=approval(manifest), workers=2)
    assert len(calls) == 2
    assert report["summary"]["unknown_requests"] == 1
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["known_spent_units"] == 15000


def test_client_factory_failure_stops_without_secret_in_report(tmp_path):
    manifest = contract(tmp_path)

    def broken(job_id):
        raise RuntimeError("fake-secret")

    report = run_session(manifest, tmp_path / "session.db", broken, evaluators(manifest),
                         approval=approval(manifest))
    assert report["summary"]["stop_reason"] == "driver_failure"
    assert "fake-secret" not in json.dumps(report)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_is_rejected(tmp_path, workers):
    manifest = contract(tmp_path)
    with pytest.raises(ValueError, match="workers"):
        run_session(manifest, tmp_path / "session.db", factory([]), evaluators(manifest),
                    approval=approval(manifest), workers=workers)


def test_mutation_of_callers_manifest_does_not_change_requests(tmp_path):
    manifest = contract(tmp_path, 1)
    original = copy.deepcopy(manifest)
    calls = []
    normal = factory(calls)

    def mutate(job_id):
        manifest["jobs"][0]["messages"][0]["content"] = "Mutated"
        return normal(job_id)

    report = run_session(manifest, tmp_path / "session.db", mutate, evaluators(manifest),
                         approval=approval(manifest))
    assert report["summary"]["run_complete"]
    assert calls[0][1]["messages"] == original["jobs"][0]["messages"]


def test_evidence_drift_during_client_creation_prevents_transport(tmp_path):
    manifest = contract(tmp_path, 1)
    calls = []
    normal = factory(calls)

    def mutate(job_id):
        Path(manifest["preflight"]["price_evidence"]["path"]).write_text("Changed.", encoding="utf-8")
        return normal(job_id)

    report = run_session(manifest, tmp_path / "session.db", mutate, evaluators(manifest),
                         approval=approval(manifest))
    assert not calls
    assert report["summary"]["stop_reason"] == "evidence_drift"
    assert report["summary"]["planned_requests"] == 1


def test_pending_resume_does_not_create_clients(tmp_path):
    manifest = contract(tmp_path, 2)
    path = tmp_path / "session.db"
    session = ExperimentSession(path, manifest)
    session.claim("job/0")
    report = run_session(manifest, path, lambda job: pytest.fail("Unresolved session cannot call"),
                         evaluators(manifest), approval=approval(manifest))
    assert report["summary"]["stop_reason"] == "unresolved_requests"
    assert report["summary"]["held_units"] == 200000


@pytest.mark.parametrize("invalid", [None, [], {"version": 1}])
def test_malformed_manifest_returns_blockers_not_raw_exception(tmp_path, invalid):
    assert preflight(invalid)["ready"] is False


def test_task_id_is_required_for_driver_preflight(tmp_path):
    manifest = contract(tmp_path)
    del manifest["jobs"][0]["task_id"]
    assert preflight(manifest)["ready"] is False


def test_cli_is_offline_and_refuses_overwrite(tmp_path, monkeypatch):
    import socket
    from experiments.preflight_session import main

    def forbidden(*args, **kwargs):
        pytest.fail("Offline CLI must not create clients or connect to network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(LLMClient, "__init__", forbidden)
    manifest = contract(tmp_path)
    source = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["--manifest", str(source), "--output", str(output)]) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--manifest", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_cli_writes_blockers_for_unready_contract(tmp_path):
    from experiments.preflight_session import main

    manifest = contract(tmp_path)
    del manifest["preflight"]
    source = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["--manifest", str(source), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["ready"] is False


def test_cli_exposes_no_paid_execution_switch(tmp_path):
    from experiments.preflight_session import main

    with pytest.raises(SystemExit) as error:
        main(["--manifest", "manifest.json", "--output", "report.json", "--live"])
    assert error.value.code == 2


def test_manifest_credential_rejection_does_not_leak_secret_in_report(tmp_path):
    manifest = contract(tmp_path)
    manifest["api_key"] = "fake-secret"
    report = preflight(manifest)
    assert not report["ready"]
    assert "fake-secret" not in json.dumps(report)
