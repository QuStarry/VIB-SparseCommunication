import json
import os
import sys
from types import SimpleNamespace

import pytest

from experiments.feedback_protocol import freeze_feedback_protocol, stage_manifest, agent_messages
from experiments.run_feedback_experiment import execute_phase, route_round, run_feedback
from tests.test_prompt_contrast_pilot import ENDPOINT, factory
from src.request_ledger import fingerprint


@pytest.fixture(autouse=True)
def fast_flush(monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def protocol():
    return freeze_feedback_protocol(ENDPOINT)


def test_disjoint_sampling_and_declared_budget():
    frozen = protocol()
    assert len(frozen["tasks"]) == 16
    assert not set(frozen["task_ids"]) & set(frozen["excluded_task_ids"])
    assert len(frozen["excluded_task_ids"]) == 104
    assert frozen["phase_counts"] == {"round1": 48, "round2": 192, "synthesis": 128}
    assert frozen["phase_budgets"] == {"round1": "0.15", "round2": "0.45", "synthesis": "0.40"}
    assert all(set(task) == {"task_id", "prompt"} for task in frozen["tasks"])


def test_agent_template_reads_real_history():
    messages = agent_messages("public task", ["[Coder]: actual earlier output"], 2, "Reviewer")
    assert "actual earlier output" in messages[1]["content"]
    assert "第 2 轮" in messages[1]["content"]


def test_round1_real_stage_and_cache(tmp_path):
    frozen = protocol()
    manifest = stage_manifest(frozen, "round1", {}, None)
    calls = []
    output = tmp_path / "round1"
    report = execute_phase(frozen, manifest, output, factory(calls))
    assert len(calls) == 48
    assert report["summary"]["run_complete"]
    assert all(row["stage"] == "agent" for row in report["requests"])
    before = (output / "requests.jsonl").read_bytes()
    assert execute_phase(frozen, manifest, output, lambda path: pytest.fail("cached")) == report
    assert (output / "requests.jsonl").read_bytes() == before


def test_unknown_stops_single_attempt(tmp_path):
    frozen, calls = protocol(), []
    report = execute_phase(frozen, stage_manifest(frozen, "round1", {}, None),
                           tmp_path / "round1", factory(calls, missing_usage=True))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "unknown_usage"
    assert report["summary"]["proxy_cost_usd"] is None


def test_phase_manifest_drift_rejected(tmp_path):
    frozen = protocol()
    manifest = stage_manifest(frozen, "round1", {}, None)
    manifest["jobs"][0]["stage"] = "code"
    with pytest.raises(ValueError):
        execute_phase(frozen, manifest, tmp_path / "round1", lambda path: pytest.fail("drift"))


def test_matched_routes_use_own_candidates():
    import numpy as np

    class Encoder:
        def encode(self, texts, **kwargs):
            vectors = np.ones((len(texts), 384), dtype=np.float32)
            vectors[:, 0] = np.arange(len(texts)) + 1
            return vectors

    frozen = protocol()
    candidates = {task: {policy: [f"{policy} contribution {index}" for index in range(3)]
                         for policy in frozen["policies"]} for task in frozen["task_ids"]}
    routes = route_round(frozen, candidates, None, 1, Encoder())
    for task in frozen["task_ids"]:
        count = routes["histories"][task]["vib"]["round_counts"]
        for policy in ("random_matched", "cosine_matched"):
            assert routes["histories"][task][policy]["round_counts"] == count
            assert all(policy in text for text in routes["histories"][task][policy]["history"])


class FakeEncoder:
    max_seq_length = 256

    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        pass

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        import numpy as np
        vectors = np.ones((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = [len(text) % 31 + 1 for text in texts]
        return vectors


@pytest.fixture
def fake_pipeline(monkeypatch):
    import experiments.run_feedback_experiment as runner
    monkeypatch.setattr(runner, "endpoint_metadata", lambda path: ENDPOINT)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=FakeEncoder))


def test_full_pipeline_368_and_real_second_round_context(tmp_path, fake_pipeline):
    calls = []
    output = tmp_path / "pipeline"
    result = run_feedback(output, factory(calls))
    assert result["complete"]
    assert len(calls) == 368
    manifest = json.loads((output / "round2/manifest.json").read_text())
    routes = json.loads((output / "routes_round1.json").read_text())
    for job in manifest["jobs"]:
        history = routes["histories"][job["task_id"]][job["prompt_variant"]]["history"]
        assert job["history_sha256"] == fingerprint(history)
        for text in history:
            assert text in job["messages"][1]["content"]
    before = {phase: (output / phase / "requests.jsonl").read_bytes()
              for phase in ("round1", "round2", "synthesis")}
    assert run_feedback(output, lambda path: pytest.fail("cached")) == result
    for phase, content in before.items():
        assert (output / phase / "requests.jsonl").read_bytes() == content


def test_pipeline_unknown_blocks_all_downstream(tmp_path, fake_pipeline):
    calls = []
    result = run_feedback(tmp_path / "pipeline", factory(calls, missing_usage=True))
    assert not result["complete"]
    assert len(calls) == 1
    assert not (tmp_path / "pipeline/round2").exists()


def test_candidate_truncation_retained_without_repair(tmp_path):
    frozen, calls = protocol(), []
    report = execute_phase(frozen, stage_manifest(frozen, "round1", {}, None),
                           tmp_path / "round1", factory(calls, finish="length", completion_tokens=256))
    assert report["summary"]["run_complete"]
    assert report["summary"]["truncated_responses"] == 48
    assert len(calls) == 48


def test_parent_tamper_fails_before_new_client(tmp_path, fake_pipeline):
    output = tmp_path / "pipeline"
    run_feedback(output, factory([]))
    route_path = output / "routes_round1.json"
    route = json.loads(route_path.read_text())
    route["candidates_sha256"] = "changed"
    route_path.write_text(json.dumps(route))
    with pytest.raises(ValueError):
        run_feedback(output, lambda path: pytest.fail("tampered"))


def test_interrupted_pending_never_reissued(tmp_path):
    frozen, calls = protocol(), []
    manifest = stage_manifest(frozen, "round1", {}, None)
    output = tmp_path / "round1"
    with pytest.raises(KeyboardInterrupt):
        execute_phase(frozen, manifest, output, factory(calls, interrupt_after=2))
    report = execute_phase(frozen, manifest, output, lambda path: pytest.fail("pending"))
    assert report["summary"]["requests_with_evidence"] == 3
    assert report["summary"]["stop_reason"] == "unknown_usage"
    assert report["summary"]["proxy_cost_usd"] is None


def test_stage_preflight_never_shrinks_overbudget_cohort():
    frozen = protocol()
    frozen["tasks"][0]["prompt"] = "long public text " * 100000
    with pytest.raises(ValueError, match="Complete phase planning"):
        stage_manifest(frozen, "round1", {}, None)


def test_settled_interruption_resumes_only_unsent(tmp_path):
    from experiments.run_prompt_contrast_pilot import write_json
    from src.request_ledger import RequestLedger

    frozen = protocol()
    manifest = stage_manifest(frozen, "round1", {}, None)
    output = tmp_path / "round1"
    execute_phase(frozen, manifest, output, None)
    job = manifest["jobs"][0]
    ledger = RequestLedger(output / "requests.jsonl")
    request = ledger.start(
        {"parameters": job["parameters"], "prompt_sha256": fingerprint(job["messages"]), "stage": "agent"},
        operation_id=fingerprint([fingerprint(manifest), job["job_id"]]), attempt=1)
    content = "already received contribution"
    raw = {"job_id": job["job_id"], "content": content, "output_sha256": fingerprint(content),
           "response_id": "settled", "served_model": ENDPOINT["model"], "finish_reason": "stop"}
    from experiments.run_prompt_contrast_pilot import response_path
    write_json(response_path(output, job), raw)
    ledger.finish(request, status="completed", usage_status="known",
                  usage={"prompt_tokens": 100, "completion_tokens": 20}, **{key: raw[key] for key in (
                      "output_sha256", "response_id", "served_model", "finish_reason")})
    calls = []
    report = execute_phase(frozen, manifest, output, factory(calls))
    assert report["summary"]["run_complete"]
    assert len(calls) == 47
    assert json.loads(response_path(output, job).read_text()) == raw


def test_http_quota_failure_never_retries(tmp_path):
    frozen, calls = protocol(), []
    report = execute_phase(frozen, stage_manifest(frozen, "round1", {}, None),
                           tmp_path / "round1", factory(calls, failure=429))
    assert len(calls) == 1
    assert report["summary"]["stop_reason"] == "http_429"
    assert report["summary"]["proxy_cost_usd"] is None
    assert "fake-secret" not in str(report)
