import json
import os

import pytest

from experiments.analyze_prompt_contrast_replication import exact_sign_test, analyze_replication
from experiments.run_prompt_contrast_pilot import freeze_pilot
from experiments.run_prompt_contrast_replication import freeze_replication, execute_replication
from tests.test_prompt_contrast_pilot import SOURCE, ENDPOINT, factory


@pytest.fixture(autouse=True)
def fake_flush(monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def frozen(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(freeze_pilot(SOURCE, ENDPOINT)))
    return freeze_replication(SOURCE, prior, ENDPOINT)


def test_exact_direction_test_small_samples():
    assert exact_sign_test(0, 4) == 0.125
    assert exact_sign_test(0, 3) == 0.25
    assert exact_sign_test(0, 1) == 1
    assert exact_sign_test(0, 0) == 1
    assert exact_sign_test(3, 3) == 1


def test_score_all_after_generation_and_cache(monkeypatch, tmp_path):
    from experiments import analyze_prompt_contrast_replication as analysis
    manifest, output, calls, scored = frozen(tmp_path), tmp_path / "run", [], []
    execute_replication(manifest, output, factory(calls))

    def evaluate(content):
        assert len(calls) == 288
        scored.append(content)
        return True

    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: evaluate for task in manifest["task_ids"]})
    result = analyze_replication(output)
    assert len(scored) == 288
    assert result["primary_comparison"]["left"] == "legacy_compact"
    assert result["primary_comparison"]["right"] == "current_arbitration"
    assert result["primary_comparison"]["exact_sign_p"] == 1
    assert result["statistics"]["groups"]["legacy_compact"]["independent_tasks"] == 48
    assert "pooled_statistics" not in result
    assert analyze_replication(output) == result
    assert len(scored) == 288


def test_incomplete_generation_not_scored(monkeypatch, tmp_path):
    from experiments import analyze_prompt_contrast_replication as analysis
    manifest, output = frozen(tmp_path), tmp_path / "run"
    execute_replication(manifest, output, factory([], missing_usage=True))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: pytest.fail("incomplete"))
    with pytest.raises(ValueError):
        analyze_replication(output)


def test_truncated_failure_without_hidden_evaluation(monkeypatch, tmp_path):
    from experiments import analyze_prompt_contrast_replication as analysis
    manifest, output = frozen(tmp_path), tmp_path / "run"
    execute_replication(manifest, output, factory([], finish="length", completion_tokens=1024))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: lambda content: pytest.fail("truncated") for task in manifest["task_ids"]})
    result = analyze_replication(output)
    assert not any(row["passed"] for row in result["evaluations"])


def test_evaluation_failure_never_regenerates(monkeypatch, tmp_path):
    from experiments import analyze_prompt_contrast_replication as analysis
    manifest, output, calls = frozen(tmp_path), tmp_path / "run", []
    execute_replication(manifest, output, factory(calls))

    def broken(content):
        raise RuntimeError("Scoring facility failure")
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: broken for task in manifest["task_ids"]})
    with pytest.raises(RuntimeError):
        analyze_replication(output)
    assert not (output / "analysis.json").exists()
    assert len(calls) == 288


def test_score_cache_tampering_rejected(monkeypatch, tmp_path):
    from experiments import analyze_prompt_contrast_replication as analysis
    manifest, output = frozen(tmp_path), tmp_path / "run"
    execute_replication(manifest, output, factory([]))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: lambda content: False for task in manifest["task_ids"]})
    analyze_replication(output)
    path = output / "analysis.json"
    result = json.loads(path.read_text())
    result["evaluations"][0]["passed"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        analyze_replication(output)
