import json

import pytest

import experiments.rescore_feedback_plus as module
from src.experiment_driver import file_sha256
from src.request_ledger import fingerprint


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    tasks = [{"task_id": f"HumanEval/{index}", "prompt": "def f(x):\n",
              "canonical_solution": "    return x * 2\n", "entry_point": "f",
              "test": "def check(candidate):\n    assert candidate(3) == 6"}
             for index in range(164)]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(task) for task in tasks))
    monkeypatch.setattr(module, "SOURCE_SHA256", file_sha256(source))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"tasks": tasks}))
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"content": "def f(x): return x * 2", "output_sha256":
                              fingerprint("def f(x): return x * 2")}))
    evidence = {"raw": {"path": str(raw), "sha256": file_sha256(raw)}}
    protocol = {
        "evaluation": {"path": str(bundle)}, "task_ids": [task["task_id"] for task in tasks[:16]],
        "analysis": {"primary": ["vib", "random_matched"], "primary_alpha": 0.05,
                     "bootstrap_samples": 20, "seed": 20261029},
    }
    jobs = [{"task_id": task, "job_id": f"{task}/{policy}/{repeat}",
             "prompt_variant": policy, "repetition": repeat, "original_passed": True,
             "original_status": "completed", "raw_response": evidence["raw"],
             "output_sha256": fingerprint("def f(x): return x * 2")}
            for task in protocol["task_ids"] for policy in ("full", "vib", "random_matched", "cosine_matched")
            for repeat in (0, 1)]
    monkeypatch.setattr(module, "original_evidence", lambda: (protocol, jobs, evidence))
    monkeypatch.setattr(module, "_score", lambda *args: {"passed": True, "status": "pass", "independent_checks": 2})
    return source, tmp_path / "results"


def test_checksum_drift_rejected_before_parsing(tmp_path):
    source = tmp_path / "changed.jsonl"
    source.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        module.load_dataset(source)


def test_freezes_all_original_outputs_separate_from_tests(synthetic_source):
    source, output = synthetic_source
    manifest = module.prepare_plus(source, output)
    assert len(manifest["jobs"]) == 128
    assert manifest["reference_count"] == 164
    assert manifest["cost"]["new_model_requests"] == 0
    assert all("test" not in job and "canonical_solution" not in job for job in manifest["jobs"])
    assert file_sha256(output / "publisher_test.jsonl") == file_sha256(source)


def test_reference_acceptance_precedes_all_candidates(synthetic_source, monkeypatch):
    source, output = synthetic_source
    manifest = module.prepare_plus(source, output)
    calls = []
    def score(code, task, protocol):
        calls.append(code)
        assert len(calls) <= 164 or (output / "reference_acceptance.json").exists()
        return {"passed": True, "status": "pass", "independent_checks": 2}
    monkeypatch.setattr(module, "_score", score)
    result = module.rescore_feedback_plus(output)
    assert len(calls) == 164 + 128
    assert result["independent_candidate_executions"] == 256
    assert result["new_model_requests"] == 0
    assert result["statistics"]["groups"]["vib"]["passed_generations"] == 32
    assert result["original_statistics"]["groups"]["vib"]["passed_generations"] == 32
    monkeypatch.setattr(module, "_score", lambda *args: pytest.fail("cached rescoring"))
    assert module.rescore_feedback_plus(output) == result


def test_reference_failure_never_scores_candidates(synthetic_source, monkeypatch):
    source, output = synthetic_source
    module.prepare_plus(source, output)
    calls = []
    def fail(*args):
        calls.append(args)
        return {"passed": False, "status": "assertion_failure", "independent_checks": 2}
    monkeypatch.setattr(module, "_score", fail)
    with pytest.raises(ValueError, match="reference failed"):
        module.rescore_feedback_plus(output)
    assert len(calls) == 1
    assert not (output / "candidate_checks").exists()
    assert not (output / "analysis.json").exists()


def test_new_failures_preserved_with_original_pass_labels(synthetic_source, monkeypatch):
    source, output = synthetic_source
    manifest = module.prepare_plus(source, output)
    tasks = module.load_dataset(manifest["dataset"]["path"])
    module.reference_acceptance(output, manifest, tasks)
    monkeypatch.setattr(module, "_score", lambda *args: {
        "passed": False, "status": "assertion_failure", "independent_checks": 2})
    result = module.rescore_feedback_plus(output)
    assert result["transitions"]["vib"]["pass_to_fail"] == 32
    assert all(row["original_passed"] and not row["passed"] for row in result["evaluations"])


def test_original_evidence_change_stops_before_scoring(synthetic_source, monkeypatch):
    source, output = synthetic_source
    manifest = module.prepare_plus(source, output)
    raw = manifest["jobs"][0]["raw_response"]["path"]
    from pathlib import Path
    Path(raw).write_text("changed")
    monkeypatch.setattr(module, "_score", lambda *args: pytest.fail("changed source"))
    with pytest.raises(ValueError, match="evidence changed"):
        module.rescore_feedback_plus(output)


def test_cached_content_tamper_rejected(synthetic_source):
    source, output = synthetic_source
    module.prepare_plus(source, output)
    module.rescore_feedback_plus(output)
    path = output / "analysis.json"
    result = json.loads(path.read_text())
    result["statistics"]["groups"]["vib"]["passed_generations"] = 100
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Cached"):
        module.rescore_feedback_plus(output)


def test_manifest_cost_or_scope_drift_rejected(synthetic_source):
    source, output = synthetic_source
    module.prepare_plus(source, output)
    path = output / "manifest.json"
    changed = json.loads(path.read_text())
    changed["cost"]["new_model_requests"] = 99
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="artifact differs"):
        module.rescore_feedback_plus(output)


def test_reference_interrupt_reuses_completed_checks(synthetic_source, monkeypatch):
    source, output = synthetic_source
    manifest = module.prepare_plus(source, output)
    tasks = module.load_dataset(manifest["dataset"]["path"])
    calls = []
    def interrupted(*args):
        calls.append(args)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return {"passed": True, "status": "pass", "independent_checks": 2}
    monkeypatch.setattr(module, "_score", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.reference_acceptance(output, manifest, tasks)
    assert len(list((output / "reference_checks").glob("*.json"))) == 2
    resumed = []
    def score(*args):
        resumed.append(args)
        return {"passed": True, "status": "pass", "independent_checks": 2}
    monkeypatch.setattr(module, "_score", score)
    assert module.reference_acceptance(output, manifest, tasks)["all_passed"]
    assert len(resumed) == 162
