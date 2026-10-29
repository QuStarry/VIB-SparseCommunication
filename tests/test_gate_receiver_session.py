import copy
import json
import os
from pathlib import Path

import pytest

from experiments.run_gate_receiver import freeze_gate_session, execute_gate_session
from tests.test_prompt_contrast_pilot import ENDPOINT, factory
from src.request_ledger import RequestLedger, fingerprint


DESIGN = Path(__file__).resolve().parents[1] / "results/gate_receiver_design_20261028/design.json"


@pytest.fixture(autouse=True)
def fake_flush(monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def freeze():
    return freeze_gate_session(DESIGN, ENDPOINT)


def test_frozen_all_policy_pairs_only_history_changes():
    manifest = freeze()
    assert len(manifest["jobs"]) == 256
    assert manifest["task_count"] == 32
    assert manifest["analysis"]["primary"] == ["vib", "random_matched"]
    tasks = {}
    for job in manifest["jobs"]:
        assert job["parameters"] == {"temperature": 0, "max_tokens": 1024}
        tasks.setdefault(job["task_id"], {})[job["prompt_variant"]] = job
        assert "test_cases" not in job and "passed" not in job
    for arms in tasks.values():
        assert len({job["candidate_stream_sha256"] for job in arms.values()}) == 1
        assert arms["vib"]["round_counts"] == arms["random_matched"]["round_counts"]
        assert arms["vib"]["round_counts"] == arms["cosine_matched"]["round_counts"]
    assert manifest["encoder"]["dimension"] == 384


def test_complete_cached_no_client_and_no_hidden_scoring(tmp_path):
    manifest, calls, output = freeze(), [], tmp_path / "run"
    report = execute_gate_session(manifest, output, factory(calls))
    assert len(calls) == 256
    assert report["summary"]["run_complete"]
    assert not (output / "analysis.json").exists()
    before = (output / "requests.jsonl").read_bytes()
    assert execute_gate_session(manifest, output, lambda path: pytest.fail("cached")) == report
    assert before == (output / "requests.jsonl").read_bytes()


@pytest.mark.parametrize("kwargs,reason", [
    ({"missing_usage": True}, "unknown_usage"),
    ({"failure": 429}, "http_429"),
    ({"model": "different"}, "served_model_mismatch"),
])
def test_unknown_or_drift_stops_without_retry(tmp_path, kwargs, reason):
    calls = []
    report = execute_gate_session(freeze(), tmp_path / "run", factory(calls, **kwargs))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == reason
    if kwargs.get("missing_usage") or kwargs.get("failure"):
        assert report["summary"]["proxy_cost_usd"] is None


def test_pending_resume_never_sends_again(tmp_path):
    manifest, output = freeze(), tmp_path / "run"
    execute_gate_session(manifest, output, None)
    job = manifest["jobs"][0]
    RequestLedger(output / "requests.jsonl").start(
        {"parameters": job["parameters"], "prompt_sha256": fingerprint(job["messages"])},
        operation_id=fingerprint([fingerprint(manifest), job["job_id"]]), attempt=1)
    report = execute_gate_session(manifest, output, lambda path: pytest.fail("pending"))
    assert report["summary"]["stop_reason"] == "unknown_usage"


def test_manifest_drift_rejected_before_transport(tmp_path):
    manifest = copy.deepcopy(freeze())
    manifest["jobs"][0]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        execute_gate_session(manifest, tmp_path / "run", lambda path: pytest.fail("drift"))


def test_original_design_not_modified_by_freezing():
    before = DESIGN.read_bytes()
    freeze()
    assert before == DESIGN.read_bytes()
